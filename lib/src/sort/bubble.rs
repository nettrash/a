use core::mem::ManuallyDrop;
use core::ptr;

/// Sorts a slice in place using bubble sort.
///
/// Each pass walks the unsorted prefix from left to right, carrying the largest element
/// seen so far towards the end until it meets an element that is not smaller than it.
/// After a pass, everything from the position of the last move onwards is in its final
/// place, so the next pass stops there; a pass that moves nothing ends the sort.
///
/// Two refinements keep the algorithm as cheap as bubble sort can be:
///
/// * **Last-move tracking.** The next pass scans only up to the last element that moved,
///   so already-sorted input finishes after a single pass of `n - 1` comparisons and
///   partially sorted tails are never rescanned.
/// * **Hole shifting instead of swapping.** An element travelling `k` slots to the right is
///   read once, the `k` smaller elements are each shifted left by one, and the element is
///   written once into its final slot: `k + 1` moves instead of the `3k` moves of `k` swaps.
///
/// The number of comparisons is exactly that of classic bubble sort with early exit.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order (only strictly smaller elements
///   are overtaken).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the cost grows with how far elements are from their sorted positions.
///
/// | Case                      | Comparisons | Moves  |
/// |---------------------------|-------------|--------|
/// | Best (already sorted)     | `O(n)`      | `0`    |
/// | Average                   | `O(n²)`     | `O(n²)`|
/// | Worst (reverse sorted)    | `O(n²)`     | `O(n²)`|
///
/// Bubble sort is quadratic and is mainly of educational value; prefer
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
/// use a::sort::bubble;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// bubble(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// bubble(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn bubble<T: Ord>(arr: &mut [T]) {
    // Everything at index `n` and beyond is already in its final position.
    let mut n = arr.len();
    while n > 1 {
        let s = &mut arr[..n];
        let len = s.len();
        let mut last = 0; // final index of the last element moved in this pass
        let mut j = 1;
        while j < len {
            if s[j] < s[j - 1] {
                // `s[j - 1]` is the largest element seen in this pass: carry it right past
                // every strictly smaller element, shifting each of those left by one slot.
                let base = s.as_mut_ptr();
                // SAFETY: every offset is `< len`, so all pointers stay inside `s`. The
                // element at `j - 1` is moved into `hole`, which leaves exactly one slot
                // logically uninitialised: `hole.dst`. Each step moves the next element into
                // that slot and advances `hole.dst` onto the one just vacated, so source and
                // destination never overlap. When `hole` is dropped — normally, or while
                // unwinding from a panicking comparison — it writes the carried element back
                // into `hole.dst`, so `s` again holds every element exactly once.
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
                // The carried element now sits at `j - 1`, and `s[j]` is known not to be
                // smaller than it, so the scan resumes from `j + 1`.
                last = j - 1;
            }
            j += 1;
        }
        n = last;
    }
}

/// An element temporarily moved out of a slice, plus the slot it will be written back to.
///
/// Writing back on drop is what keeps [`bubble`] panic-safe: the slice is whole again even
/// if a comparison unwinds while the element is out.
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
