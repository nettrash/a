use core::hint::select_unpredictable;
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
///   of the repeated full passes of classic comb sort. For integers, its branchless passes
///   make the whole comb sort 1.1–1.2 times faster than a finish that branches on every
///   comparison.
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
/// * **Carrying in registers at small gaps, for elements of at most 16 bytes.** At a gap of
///   14 or less, each pair overlaps the one `gap` slots before it, so a swap has to reach
///   memory before the next comparison can read it back, and a pass at gap 2 runs at a third
///   of the speed of one at a large gap. But the pairs form `gap` chains
///   `c, c + gap, c + 2·gap, ...`, and along a chain a pass carries the largest element seen
///   so far. So the pass holds the carried element of every chain in a register, one per
///   chain, and at each position writes back only the smaller of the carried element and the
///   next one: the comparisons of a plain pass in its order, with the same outcomes and the
///   same result, and no round trip through memory. The `gap` chains are independent, so
///   their comparisons overlap. Against walking one chain at a time this makes the whole sort
///   1.1–1.3 times faster for 4- and 8-byte integers, and against a plain pass 1.1–1.4 times
///   for pairs of integers and boxed strings; it is slower for `String`, whose comparisons
///   take long to start, which keeps the independent pairs of a plain pass. Chains shorter
///   than four elements, which only short slices have, are not worth picking up.
///
/// The comparisons made are exactly those of comb sort with the rule of 11, in the order of a
/// plain pass, followed by those of bubble sort with early exit and last-swap bound (which
/// interleaves two of its passes for small elements).
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
        if size <= LANES_MAX_SIZE && gap <= LANES_MAX_GAP && arr.len() >= LANES_MIN_CHAIN * gap {
            // Every gap of 14 or less that occurs for 15 elements or more: the others, 5, 7,
            // 12 and 13, only start the sort of a shorter slice.
            match gap {
                2 => pass_lanes::<T, 2>(arr),
                3 => pass_lanes::<T, 3>(arr),
                4 => pass_lanes::<T, 4>(arr),
                6 => pass_lanes::<T, 6>(arr),
                8 => pass_lanes::<T, 8>(arr),
                11 => pass_lanes::<T, 11>(arr),
                14 => pass_lanes::<T, 14>(arr),
                _ => pass_branchless(arr, gap),
            }
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

/// The largest element size, in bytes, for which small gaps are passed by [`pass_lanes`].
///
/// Measured on the whole sort against [`pass_branchless`] at those gaps: 1.05–1.6 times faster
/// for `u32`, `u64`, `Box<u64>`, `(u64, u64)` and `Box<str>`, 1.05–1.2 times slower for
/// `String`.
const LANES_MAX_SIZE: usize = 16;

/// The largest gap passed by [`pass_lanes`]. From 18 up, the independent pairs of
/// [`pass_branchless`], which the compiler vectorizes for integers, are as fast or faster.
const LANES_MAX_GAP: usize = 14;

/// The shortest chains, in elements, worth holding in registers: shorter ones cost more to
/// pick up and put back than they save.
const LANES_MIN_CHAIN: usize = 4;

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
                let (to_a, to_b) = select_unpredictable(swap, (b, a), (a, b));
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

/// Does what [`pass_branchless`] does at gap `G`, with the carried element of each chain held
/// outside the slice.
///
/// Position `j` belongs to chain `j % G`. Along a chain, a pass carries the largest element
/// seen so far: each pair is that element and the next one, and the smaller of the two stays
/// behind. So the carried element of chain `c` is held in `lanes.held[c]`, the slot it came
/// from is vacant, and at each position the smaller of the held element and the next one is
/// written into the vacant slot, chosen by value so that the held elements can stay in
/// registers. Positions are taken in increasing order, as in a plain pass, and the head of a
/// chain is compared in its slot before it is picked up.
///
/// `G` must be less than `arr.len()`.
#[inline(always)]
fn pass_lanes<T: Ord, const G: usize>(arr: &mut [T]) {
    let len = arr.len();
    let base = arr.as_mut_ptr();
    // SAFETY: `G < len`, and every index is `< len`. Picking up the head of a chain leaves its
    // slot logically uninitialised, and `Lanes` keeps track of that slot: each step reads the
    // next element of a chain and its held element bitwise, writes the smaller into the
    // chain's vacant slot and the larger into `held`, and moves the vacant slot onto the slot
    // of the element just read; nothing between the comparison and the end of a step can
    // panic. When `lanes` is dropped — normally, or while unwinding from a panicking
    // comparison — it writes every held element into its chain's vacant slot, so `arr` again
    // holds every element exactly once.
    unsafe {
        // The first row compares each chain's head with the next element while both are in
        // their slots; its pairs share no element, so all of them can be compared before any
        // is moved, and a panic there leaves the slice untouched.
        let first = G.min(len - G);
        let mut smaller = [false; G];
        for (c, s) in smaller.iter_mut().enumerate().take(first) {
            *s = *base.add(G + c) < *base.add(c);
        }
        let mut lanes = Lanes::<T, G> {
            held: core::array::from_fn(|c| ManuallyDrop::new(ptr::read(base.add(c)))),
            base,
            row: G,
            done: 0,
        };
        for (c, &s) in smaller.iter().enumerate().take(first) {
            lanes.step(c, s);
        }
        if len < 2 * G {
            return;
        }
        lanes.row += G;
        lanes.done = 0;
        while lanes.row + G <= len {
            for c in 0..G {
                let s = *base.add(lanes.row + c) < *lanes.held[c];
                lanes.step(c, s);
            }
            lanes.row += G;
            lanes.done = 0;
        }
        for c in 0..len - lanes.row {
            let s = *base.add(lanes.row + c) < *lanes.held[c];
            lanes.step(c, s);
        }
    }
}

/// The carried elements of the `G` chains of a pass at gap `G`, walked a row of `G`
/// positions at a time.
///
/// Writing the held elements back on drop is what keeps [`pass_lanes`] panic-safe: the slice
/// is whole again even if a comparison unwinds while the elements are out.
struct Lanes<T, const G: usize> {
    held: [ManuallyDrop<T>; G],
    base: *mut T,
    /// The first position of the row being walked, a multiple of `G` and at least `G`.
    row: usize,
    /// How many chains have taken their step in this row. Chain `c`'s vacant slot is
    /// `row + c` if it has, and `row - G + c` if it has not.
    done: usize,
}

impl<T: Ord, const G: usize> Lanes<T, G> {
    /// Takes chain `c`'s step at position `row + c`, which must be inside the slice, given
    /// whether the element there is `smaller` than the held one, with `c == done`.
    #[inline(always)]
    unsafe fn step(&mut self, c: usize, smaller: bool) {
        // SAFETY: see `pass_lanes`; the vacant slot of chain `c` is `row - G + c` until now.
        unsafe {
            let x = self.base.add(self.row + c);
            let held: *mut ManuallyDrop<T> = &mut self.held[c];
            let stays =
                select_unpredictable(smaller, ManuallyDrop::new(ptr::read(x)), ptr::read(held));
            let goes_on =
                select_unpredictable(smaller, ptr::read(held), ManuallyDrop::new(ptr::read(x)));
            ptr::write(
                self.base.add(self.row - G + c),
                ManuallyDrop::into_inner(stays),
            );
            ptr::write(held, goes_on);
            self.done = c + 1;
        }
    }
}

impl<T, const G: usize> Drop for Lanes<T, G> {
    fn drop(&mut self) {
        for (c, held) in self.held.iter().enumerate() {
            let slot = if c < self.done {
                self.row + c
            } else {
                self.row - G + c
            };
            // SAFETY: `slot` is chain `c`'s vacant slot, and its held element is never used
            // again after this write.
            unsafe { ptr::copy_nonoverlapping(&**held, self.base.add(slot), 1) }
        }
    }
}
