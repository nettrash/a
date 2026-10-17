use core::mem::ManuallyDrop;
use core::ptr;

/// Sorts a slice in place using gnome sort.
///
/// A gnome walks along the slice comparing each element with the one before it. While they
/// are in order it steps forward; when they are not, it carries the element back, one slot at
/// a time, until it meets an element that is not greater than it or reaches the front. The
/// slice behind the gnome is always sorted, so once it walks off the end the sort is done.
///
/// Two refinements keep the algorithm as cheap as gnome sort can be:
///
/// * **Teleportation.** Classic gnome sort, having carried an element back `d` slots, walks
///   forward again over those `d` slots, comparing every pair on the way, although each of
///   them is already known to be in order. This gnome remembers where it turned back and
///   returns there at once, which saves exactly one comparison per inversion: about half of
///   them on random input, and `n(n - 1)/2` of the `n(n - 1)` on reverse-sorted input.
/// * **Hole shifting instead of swapping.** An element travelling `d` slots is read once, the
///   `d` greater elements it passes are each shifted right by one slot as they are compared
///   with it, and it is written once into its final slot: `d + 1` moves instead of the `3d`
///   moves of `d` swaps.
///
/// Every comparison made is one that classic gnome sort also makes, in the same order; only
/// the redundant ones of the walk forward are skipped. With both refinements gnome sort makes
/// the same comparisons and moves as insertion sort with a linear search; [`insertion`] in this
/// crate searches by galloping instead and makes far fewer comparisons.
///
/// [`insertion`]: fn@crate::sort::insertion
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order (only strictly greater elements
///   are passed).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the cost grows with how far elements are from their sorted positions.
///
/// Let `I` be the number of inversions, `C` the number of elements that move at least one slot
/// left, and `F` the number of elements other than the first that are smaller than every
/// element before them (they are carried all the way to the front, where no comparison is
/// needed to stop). Each passed element moves one slot and removes one inversion, so every
/// inversion costs exactly one comparison and one move.
///
/// | Case                      | Comparisons         | Moves                 |
/// |---------------------------|---------------------|-----------------------|
/// | Best (already sorted)     | `n - 1`             | `0`                   |
/// | Any input                 | `n - 1 + I - F`     | `I + C`               |
/// | Average                   | `≈ n²/4`            | `≈ n²/4`              |
/// | Worst (reverse sorted)    | `n(n - 1)/2`        | `n(n + 1)/2 - 1`      |
///
/// Gnome sort is quadratic and is mainly of educational value; prefer [`slice::sort`] for
/// real workloads.
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
/// use a::sort::gnome;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// gnome(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // The 0 at the end is carried to the front in a single walk back.
/// let mut v = [1, 2, 3, 4, 5, 0];
/// gnome(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// gnome(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn gnome<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    // `arr[..pos]` is sorted; the gnome compares `arr[pos - 1]` with `arr[pos]`.
    let mut pos = 1;
    while pos < len {
        if arr[pos] < arr[pos - 1] {
            // Carry `arr[pos]` left past every strictly greater element, shifting each of
            // those right by one slot.
            let base = arr.as_mut_ptr();
            // SAFETY: `hole.dst` starts at `base + pos`, with `0 < pos < len`, and the loop
            // ends once it reaches `base`, so every pointer stays inside `arr`. The element at
            // `pos` is moved into `hole`, which leaves exactly one slot logically
            // uninitialised: `hole.dst`. Each step moves the previous element into that slot
            // and moves `hole.dst` back onto the one just vacated, so source and destination
            // never overlap, and the element compared next is never the vacant one. When
            // `hole` is dropped — normally, or while unwinding from a panicking comparison —
            // it writes the carried element back into `hole.dst`, so `arr` again holds every
            // element exactly once.
            //
            // The walk back steps `hole.dst` itself rather than an index alongside it: with a
            // single induction variable the loop compiles to the same tight code whether or
            // not `gnome` is inlined into its caller, where the index version was measured at
            // half the speed on `u64` once inlined.
            unsafe {
                let mut hole = Hole {
                    value: ManuallyDrop::new(ptr::read(base.add(pos))),
                    dst: base.add(pos),
                };
                loop {
                    let prev = hole.dst.sub(1);
                    ptr::copy_nonoverlapping(prev, hole.dst, 1);
                    hole.dst = prev;
                    if prev == base || *prev.sub(1) <= *hole.value {
                        break;
                    }
                }
            }
        }
        // Teleport: everything up to `pos` is sorted again, so the walk resumes just past it.
        pos += 1;
    }
}

/// An element temporarily moved out of a slice, plus the slot it will be written back to.
///
/// Writing back on drop is what keeps [`gnome`] panic-safe: the slice is whole again even if
/// a comparison unwinds while the element is out.
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
