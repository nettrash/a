use core::mem::{self, ManuallyDrop};
use core::ptr;

use super::bubble;

/// Sorts a slice in place using comb sort.
///
/// Comb sort is bubble sort with a gap. A pass compares every element with the one `gap`
/// slots after it and swaps the two if they are out of order, so a small element near the end
/// (a "turtle") moves `gap` slots left in one step instead of one. The gap starts at the length
/// of the slice and shrinks by a factor of 1.3 after each pass; once it reaches 1, the slice is
/// nearly sorted and plain bubble sort finishes it.
///
/// Four refinements keep the algorithm as cheap as comb sort can be:
///
/// * **Rule of 11 (Combsort11).** A gap of 9 or 10 is replaced by 11. The sequences that go
///   through 9 or 10 leave more turtles for the final bubble sort; 11, 8, 6, 4, 3, 2 does not.
///   On random input of 10,000 elements this saves about 6% of the comparisons and narrows
///   their spread.
/// * **An adaptive finish.** The final phase is this crate's [`bubble`](fn@bubble), which
///   scans only up to the last element it moved and moves elements by hole shifting, instead
///   of the repeated full passes of classic comb sort.
/// * **Branchless swaps for small elements.** Whether a gapped pass swaps a pair is close to a
///   coin flip, which defeats the branch predictor. For elements of at most 32 bytes the swap
///   is made unconditional: both slots are rewritten from addresses chosen by the comparison,
///   which the processor turns into conditional moves. A pass is also walked in blocks of
///   `gap` pairs taken from two non-overlapping slices, which the compiler can schedule
///   freely. On random input this is 3–4.5 times faster for integers and 1.5–1.8 times for
///   strings. On already sorted input, where every branch would be predicted, it is about a
///   quarter slower, a price worth paying for a sort that is not adaptive anyway. Larger
///   elements cost more to rewrite than a misprediction does, so they are swapped only when
///   needed.
/// * **Carrying along each chain at small gaps, for elements of at most 8 bytes.** At a gap
///   of 8 or less, each pair overlaps the one `gap` slots before it, so a swap has to reach
///   memory before the next comparison can read it back, and a pass at gap 2 runs at a third
///   of the speed of one at a large gap. Instead, each chain `c, c + gap, c + 2·gap, ...` is
///   walked holding the largest element seen so far in a register, and only the smaller of it
///   and the next element is written back: the same comparisons with the same outcomes, the
///   same result, and one move per swap plus one per carried element instead of two per swap.
///   This makes the whole sort 10–50% faster for integers and pointers. It only pays when a
///   comparison is cheap: each one waits for the previous one, so larger elements, whose
///   comparisons usually read memory (strings, for one), are much faster with the
///   independent pairs of a plain pass.
///
/// The comparisons made are exactly those of comb sort with the rule of 11, followed by bubble
/// sort with early exit and last-swap bound, though not always in the same order.
///
/// # Properties
///
/// * **Unstable**: equal elements may be reordered (a gapped swap can carry an element past
///   an equal one).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Not adaptive**: the gapped passes make the same comparisons whatever the input; only
///   the final bubble sort gets cheaper on nearly sorted input.
///
/// Let `g₁ > g₂ > ... > gₖ ≥ 2` be the gaps: `k ≈ log₁.₃ n − 4` of them, each the previous one
/// divided by 1.3 and rounded down, with the rule of 11 applied. The gapped passes make exactly
/// `Σ (n − gᵢ) ≈ n·(log₁.₃ n − 7.3)` comparisons on every input; the final bubble sort adds at
/// least `n − 1`, and on random input about `3n`.
///
/// | Case                               | Comparisons                     | Moves                 |
/// |------------------------------------|---------------------------------|-----------------------|
/// | Best (already sorted)              | `Σ (n − gᵢ) + n − 1`            | `0`                   |
/// | Average (random)                   | `≈ n·(log₁.₃ n − 4.5)`          | `O(n log n)`          |
/// | Worst                              | `O(n²)`                         | `O(n²)`               |
///
/// For `n = 1000`: 19,706 comparisons on sorted input and about 21,300 on random input,
/// against about 499,000 for bubble sort. No known input makes comb sort quadratic in
/// practice, but no `O(n log n)` bound has been proved either.
///
/// Comb sort is mainly of educational value; prefer [`slice::sort_unstable`] for real
/// workloads.
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
/// use a::sort::comb;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// comb(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // Turtles at the end move many slots per step.
/// let mut v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
/// comb(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// comb(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn comb<T: Ord>(arr: &mut [T]) {
    let mut gap = arr.len();
    loop {
        gap = shrink(gap);
        if gap <= 1 {
            break;
        }
        // The sizes are compile-time constants: for each `T`, the passes it cannot use are
        // compiled away.
        let size = mem::size_of::<T>();
        if size <= CARRY_MAX_SIZE && gap <= CARRY_MAX_GAP {
            pass_carry(arr, gap);
        } else if size <= BRANCHLESS_MAX_SIZE {
            pass_branchless(arr, gap);
        } else {
            pass_branchy(arr, gap);
        }
    }
    bubble(arr);
}

/// The largest element size, in bytes, for which a gapped pass swaps without branching.
///
/// Measured on random input: integers, `u128`, pairs of integers, `&str`, `String` and `Box`
/// sort 1.6–4.5 times faster branchless; elements of 64 bytes or more about 1.2–2 times slower.
const BRANCHLESS_MAX_SIZE: usize = 32;

/// The largest element size, in bytes, for which small gaps are passed by [`pass_carry`].
///
/// Measured per pass at gaps 2 to 8: 2–3 times faster than [`pass_branchless`] for 8-byte
/// integers and pointers, but slower for every 16-byte type tried (up to 1.6 times for pairs of
/// integers) and up to 4 times slower for `String`.
const CARRY_MAX_SIZE: usize = 8;

/// The largest gap passed by [`pass_carry`]. From 11 up (9 and 10 never occur), the
/// independent pairs of [`pass_branchless`], which the compiler vectorizes for integers, win.
const CARRY_MAX_GAP: usize = 8;

/// The gap after `gap`: `gap / 1.3` rounded down, with 9 and 10 replaced by 11.
fn shrink(gap: usize) -> usize {
    // `⌊10·gap / 13⌋`, computed without overflowing for any `gap`.
    let next = gap / 13 * 10 + gap % 13 * 10 / 13;
    if next == 9 || next == 10 { 11 } else { next }
}

/// Compares every element of `arr` with the one `gap` slots after it, in increasing order of
/// position, and swaps the two if the second is smaller, with no branch on the outcome.
fn pass_branchless<T: Ord>(arr: &mut [T], gap: usize) {
    // Pairs are taken a block of `gap` at a time: `lo` and `hi` are disjoint, and the pairs
    // of a block share no element, so the order is that of a plain scan.
    let mut start = 0;
    while arr.len() - start > gap {
        let (lo, hi) = arr[start..].split_at_mut(gap);
        for (x, y) in lo.iter_mut().zip(hi) {
            let (a, b): (*mut T, *mut T) = (x, y);
            // SAFETY: `a` and `b` come from two disjoint mutable references, so both are valid
            // and do not overlap. The comparison runs while both elements are in place, so a
            // panic leaves the slice untouched. Then the element that belongs in `b` is read
            // out, the one that belongs in `a` is copied there (onto itself if no swap is
            // needed, which `ptr::copy` allows), and the element read out is written to `b`.
            // Nothing in between can panic, and both slots end up holding one of the two
            // original elements each.
            unsafe {
                let swap = *b < *a;
                let (to_a, to_b) = if swap { (b, a) } else { (a, b) };
                let tmp = ManuallyDrop::new(ptr::read(to_b));
                ptr::copy(to_a, a, 1);
                ptr::copy_nonoverlapping(&*tmp, b, 1);
            }
        }
        start += gap;
    }
}

/// Compares every element of `arr` with the one `gap` slots after it, in increasing order of
/// position, and swaps the two if the second is smaller.
fn pass_branchy<T: Ord>(arr: &mut [T], gap: usize) {
    for i in 0..arr.len() - gap {
        if arr[i + gap] < arr[i] {
            arr.swap(i, i + gap);
        }
    }
}

/// Does what [`pass_branchless`] does, one chain `c, c + gap, c + 2·gap, ...` at a time.
///
/// Along a chain, a pass carries the largest element seen so far: each pair is that element
/// and the next one, and the smaller of the two stays behind. So the carried element is held
/// outside the slice, and at each step the smaller of it and the next element is written into
/// the vacant slot behind, chosen by value so that the carried element can stay in a register.
/// Chains share no element, so taking them one after the other gives the same result as the
/// interleaved order of a plain pass.
fn pass_carry<T: Ord>(arr: &mut [T], gap: usize) {
    let len = arr.len();
    let base = arr.as_mut_ptr();
    // Chain `c` has a pair only if `c + gap < len`; `gap < len` always holds here.
    for c in 0..gap.min(len - gap) {
        // SAFETY: every index is `< len`, so all pointers stay inside `arr`. The first
        // comparison is made with both elements in place, so a panic there leaves the slice
        // untouched. Then the head of the chain is read out into `held`, which leaves exactly
        // one slot logically uninitialised: `hole.dst`. Each step reads the next element and
        // the held one bitwise, writes the smaller into `hole.dst` and the larger back into
        // `held`, and moves `hole.dst` onto the slot of the element just read, which is now the
        // vacant one; nothing in it can panic. Every later comparison is made with the next
        // element in its slot and the held one in `held`. When `hole` is dropped — normally, or
        // while unwinding from a panicking comparison — it writes the held element into
        // `hole.dst`, so `arr` again holds every element exactly once. `held` is accessed only
        // through `held_ptr` once that exists, so `hole.src` stays valid.
        unsafe {
            let mut j = c + gap;
            let mut smaller = *base.add(j) < *base.add(c);
            let mut held = ManuallyDrop::new(ptr::read(base.add(c)));
            let held_ptr: *mut ManuallyDrop<T> = &mut held;
            let mut hole = Hole {
                src: held_ptr.cast::<T>(),
                dst: base.add(c),
            };
            loop {
                let next = ManuallyDrop::new(ptr::read(base.add(j)));
                let carried = ptr::read(held_ptr);
                let (stays, goes_on) = if smaller {
                    (next, carried)
                } else {
                    (carried, next)
                };
                ptr::write(hole.dst, ManuallyDrop::into_inner(stays));
                ptr::write(held_ptr, goes_on);
                hole.dst = base.add(j);
                j += gap;
                if j >= len {
                    break;
                }
                smaller = *base.add(j) < **held_ptr;
            }
        }
    }
}

/// A vacant slot of a slice, plus the element held outside the slice that belongs there.
///
/// Writing the held element back on drop is what keeps [`pass_carry`] panic-safe: the slice is
/// whole again even if a comparison unwinds while the element is out.
struct Hole<T> {
    src: *const T,
    dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to the single uninitialised slot of the slice being sorted, and
        // `src` to the held element, which is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) }
    }
}
