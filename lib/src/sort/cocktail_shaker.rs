use core::mem::ManuallyDrop;
use core::ptr;

/// Sorts a slice in place using cocktail shaker sort.
///
/// Cocktail shaker sort is bubble sort that alternates direction. A forward pass walks the
/// unsorted range from left to right, carrying the largest element seen so far towards the
/// end; a backward pass then walks it from right to left, carrying the smallest element seen
/// so far towards the front. Each pass leaves the elements beyond its last move in their final
/// places, so the unsorted range shrinks from both ends, and a pass that moves nothing ends
/// the sort.
///
/// Bubble sort moves a small element near the end (a "turtle") only one slot left per pass,
/// so a single one costs it `n - 1` passes. Here the next backward pass carries it all the way
/// to its place: `[1, 2, ..., n - 1, 0]` costs `3n - 6` comparisons (for `n ≥ 3`) instead of
/// the `n(n - 1)/2` of bubble sort. On random input it makes about a third fewer comparisons
/// than bubble sort (`≈ n²/3` against `≈ n²/2`); on reverse-sorted input both make
/// `n(n - 1)/2`.
///
/// Two refinements keep the algorithm as cheap as cocktail shaker sort can be:
///
/// * **Last-move tracking at both ends.** A forward pass ends the unsorted range at the last
///   element it moved, and a backward pass starts it just after the last element it moved, so
///   already-sorted input finishes after a single pass of `n - 1` comparisons and sorted runs
///   at either end are never rescanned.
/// * **Hole shifting instead of swapping.** An element travelling `k` slots is read once, the
///   `k` elements it overtakes are each shifted by one slot towards where it came from as
///   they are compared with it, and it is written once into its final slot: `k + 1` moves
///   instead of the `3k` moves of `k` swaps. Shifting while comparing, rather than finding the
///   distance first and moving a block afterwards, touches each element once and avoids a
///   `memmove` call per carry, most of which are only a slot or two long; on random `u64`
///   input that is about twice as fast.
///
/// The number of comparisons is exactly that of classic cocktail shaker sort with early exit
/// and last-swap bounds at both ends.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order (only strictly smaller elements are
///   overtaken going right, and only strictly greater ones going left).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the cost grows with how far elements are from their sorted positions, in
///   either direction.
///
/// Let `I` be the number of inversions and `C` the number of carries (elements picked up and
/// moved at least one slot, counted once per pass). Each overtaken element moves one slot and
/// removes one inversion, so every inversion costs exactly one move, and each carry costs one
/// more for the final write.
///
/// | Case                      | Comparisons     | Moves                 |
/// |---------------------------|-----------------|-----------------------|
/// | Best (already sorted)     | `n - 1`         | `0`                   |
/// | Any input                 | `≤ n(n - 1)/2`  | `I + C ≤ 2I`          |
/// | Average                   | `O(n²)`         | `O(n²)`               |
/// | Worst (reverse sorted)    | `n(n - 1)/2`    | `n(n + 1)/2 - 1`      |
///
/// Cocktail shaker sort is quadratic and is mainly of educational value; prefer
/// [`slice::sort`] for real workloads.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: nothing is dropped twice or leaked.
///
/// # Examples
///
/// ```
/// use a::sort::cocktail_shaker;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// cocktail_shaker(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // The 0 at the end reaches the front in one backward pass.
/// let mut v = [1, 2, 3, 4, 5, 0];
/// cocktail_shaker(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// cocktail_shaker(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn cocktail_shaker<T: Ord>(arr: &mut [T]) {
    // Everything before `lo` and from `hi` onwards is already in its final position.
    let (mut lo, mut hi) = (0, arr.len());
    while hi - lo > 1 {
        hi = lo + forward(&mut arr[lo..hi]);
        if hi - lo <= 1 {
            break;
        }
        lo += backward(&mut arr[lo..hi]);
    }
}

/// Carries large elements of `s` to the right, and returns the final index of the last
/// element moved (`0` if none was): everything from there onwards is in its final position.
fn forward<T: Ord>(s: &mut [T]) -> usize {
    let len = s.len();
    let mut last = 0;
    let mut j = 1;
    while j < len {
        if s[j] < s[j - 1] {
            // `s[j - 1]` is the largest element seen in this pass: carry it right past every
            // strictly smaller element, shifting each of those left by one slot.
            let base = s.as_mut_ptr();
            // SAFETY: every offset is `< len`, so all pointers stay inside `s`. The element at
            // `j - 1` is moved into `hole`, which leaves exactly one slot logically
            // uninitialised: `hole.dst`. Each step moves the next element into that slot and
            // advances `hole.dst` onto the one just vacated, so source and destination never
            // overlap, and the element compared next is never the vacant one. When `hole` is
            // dropped — normally, or while unwinding from a panicking comparison — it writes
            // the carried element back into `hole.dst`, so `s` again holds every element
            // exactly once.
            unsafe {
                let mut hole = Hole {
                    value: ManuallyDrop::new(ptr::read(base.add(j - 1))),
                    dst: base.add(j - 1),
                };
                loop {
                    ptr::copy_nonoverlapping(base.add(j), hole.dst, 1);
                    hole.dst = base.add(j);
                    j += 1;
                    if j == len || *base.add(j) >= *hole.value {
                        break;
                    }
                }
            }
            // The carried element now sits at `j - 1`, and `s[j]` is known not to be smaller
            // than it, so the scan resumes from `j + 1`.
            last = j - 1;
        }
        j += 1;
    }
    last
}

/// Carries small elements of `s` to the left, and returns one past the final index of the
/// last element moved (`s.len()` if none was): everything before there is in its final
/// position.
fn backward<T: Ord>(s: &mut [T]) -> usize {
    let len = s.len();
    let mut first = len;
    // Compares `s[i - 1]` with `s[i]`.
    let mut i = len - 1;
    while i > 0 {
        if s[i] < s[i - 1] {
            // `s[i]` is the smallest element seen in this pass: carry it left past every
            // strictly greater element, shifting each of those right by one slot.
            let base = s.as_mut_ptr();
            // SAFETY: every offset is `< len`, so all pointers stay inside `s`. The element at
            // `i` is moved into `hole`, which leaves exactly one slot logically uninitialised:
            // `hole.dst`. Each step moves the previous element into that slot and moves
            // `hole.dst` back onto the one just vacated, so source and destination never
            // overlap, and the element compared next is never the vacant one. When `hole` is
            // dropped — normally, or while unwinding from a panicking comparison — it writes
            // the carried element back into `hole.dst`, so `s` again holds every element
            // exactly once.
            unsafe {
                let mut hole = Hole {
                    value: ManuallyDrop::new(ptr::read(base.add(i))),
                    dst: base.add(i),
                };
                loop {
                    ptr::copy_nonoverlapping(base.add(i - 1), hole.dst, 1);
                    hole.dst = base.add(i - 1);
                    i -= 1;
                    if i == 0 || *base.add(i - 1) <= *hole.value {
                        break;
                    }
                }
            }
            // The carried element now sits at `i`, and `s[i - 1]` (if any) is known not to be
            // greater than it, so the scan resumes by comparing `s[i - 2]` with `s[i - 1]`.
            first = i + 1;
            if i == 0 {
                break;
            }
        }
        i -= 1;
    }
    first
}

/// An element temporarily moved out of a slice, plus the slot it will be written back to.
///
/// Writing back on drop is what keeps [`cocktail_shaker`] panic-safe: the slice is whole
/// again even if a comparison unwinds while the element is out.
struct Hole<T> {
    value: ManuallyDrop<T>,
    dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to the single uninitialised slot of the slice being sorted,
        // and `value` is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(&*self.value, self.dst, 1) }
    }
}
