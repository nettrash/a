use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

/// Sorts a slice in place using Shell sort.
///
/// Shell sort, published by Donald Shell in 1959, is insertion sort with a gap. A pass with
/// gap `h` insertion sorts every chain of elements `h` apart, `v[c], v[c + h], v[c + 2h], ...`,
/// so an element far from its place travels there in steps of `h` instead of one slot at a
/// time. The passes use a decreasing sequence of gaps ending with 1, a plain insertion sort,
/// which the earlier passes leave with little to do. How fast the sort is depends almost
/// entirely on the gap sequence: with the ones used here it makes about `1.6–2·n·log₂ n`
/// comparisons on random input of up to a million elements, though no gap sequence is known
/// to make Shell sort `O(n log n)`.
///
/// Five refinements keep the algorithm as cheap as Shell sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or strictly descending run; if it covers the whole slice, the slice is left
///   alone or reversed. This costs `n − 1` comparisons on such input and usually two or three
///   on any other.
/// * **Hole shifting.** An element is read out once, the greater elements before it on its
///   chain shift one place along the chain each, and it is written once into the slot left
///   free: one move per step instead of the three of a swap.
/// * **The gaps that are fastest, not the fewest comparisons.** Ciura's sequence (`1, 4, 10,
///   23, 57, 132, 301, 701`, then times 2.25) makes the fewest comparisons of the sequences
///   tried, 1.1–1.27 times fewer than Sedgewick's 1982 one (`1, 8, 23, 77, 281, 1073, ...`).
///   But every pass ends nearly every insertion with a mispredicted branch, and Sedgewick's
///   gaps grow by a factor of about 4 instead of 2.25, so they take fewer passes: for elements
///   larger than 8 bytes they are 1.05–1.35 times faster, strings included (except random
///   strings from a million elements on, 1.1 times slower, where each comparison is a cache
///   miss). Elements of at most 8 bytes use Ciura's gaps, which suit the window below.
/// * **A branchless window for small elements.** In a pass over a slice that is still far from
///   sorted, how far an element moves along its chain is about as hard to predict as a coin
///   toss: with Ciura's gaps, 0 steps 44% of the time, 1 step 26%, 2 steps 14%, 3 steps 8%
///   and 4 or more 8%. For elements of at most 8 bytes, each element is compared with the 4
///   elements before it on its chain at once, how many of them it passes is computed as a
///   number, and the 5 slots are rewritten from copies chosen by an index computed from that
///   number: no branch, at the cost of comparisons that turn out unnecessary. Only the 8% of
///   elements that pass all 4 go on with a branch per step. On random input this is 1.4–1.8
///   times faster for integers.
/// * **The window only while it pays.** On a slice that is nearly sorted, insertion's branches
///   are predictable and the window's extra comparisons and writes are wasted: up to 2.6 times
///   slower. So a pass takes the window only if the pass before it moved more than a quarter
///   of the elements; the first pass assumes it did, and the final pass, with gap 1, never
///   does, since each element there depends on the one just placed.
///
/// # Properties
///
/// * **Unstable**: equal elements may be reordered (an insertion along a chain can carry an
///   element past an equal one that lies between the two in the slice).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the passes do less on nearly sorted input, and sorted or reverse-sorted
///   input costs one scan.
///
/// | Case                              | Comparisons                    | Moves                  |
/// |-----------------------------------|--------------------------------|------------------------|
/// | Best (sorted, reverse, all equal) | `n − 1`                        | `0` (`n` reversed)     |
/// | Average, Sedgewick's gaps         | `≈ (1.6–2)·n·log₂ n`           | `O(comparisons)`       |
/// | Average, Ciura's gaps and window  | `≈ (2.1–2.8)·n·log₂ n`         | `O(comparisons)`       |
/// | Worst                             | `O(n^(4/3))`, Sedgewick's gaps | the same               |
///
/// No bound is known for Ciura's gaps, which were found by experiment.
///
/// For `n = 1000`: 999 comparisons on sorted or reverse-sorted input and about 15,900 on random
/// input, or 21,300 with the window, whose unnecessary comparisons are cheap for small
/// elements; heap sort makes about 10,500.
///
/// Against classic Shell sort with Ciura's gaps, plain insertion and the same scan for a run,
/// this is 1.2–1.75 times faster on random integers, 1.1–1.65 times on nearly sorted ones and
/// 1.15–1.3 times with few distinct keys; 1.05–1.4 times faster on strings except random ones
/// from a million elements on, and 1.03–1.3 times on 64-byte elements. Against this crate's
/// [`comb`](crate::sort::comb) sort, whose fixed compare-exchanges the compiler vectorizes for
/// integers: 2.4–3.3 times slower on random integers from 1000 elements on, but faster on
/// nearly sorted input and with few distinct keys up to 1000 elements, and 1.3–2.5 times
/// faster on strings that are nearly sorted, have few distinct keys or follow a pattern.
/// Against [`heap`](crate::sort::heap) sort: 1.3–1.7 times slower on random integers from
/// 1000 elements on, about as fast on random strings, and faster on everything nearly
/// sorted or patterned.
///
/// Shell sort is mainly of educational value; prefer [`slice::sort_unstable`] for real
/// workloads.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: nothing is dropped twice or leaked. An [`Ord`] that is not a total
/// order leaves `arr` in an unspecified order with every element exactly once as well.
///
/// # Examples
///
/// ```
/// use a::sort::shell;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// shell(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Small elements far from their place travel in long steps.
/// let mut v: Vec<u32> = (0..1000).rev().chain(1000..2000).collect();
/// v.swap(0, 1999);
/// shell(&mut v);
/// assert!(v.windows(2).all(|w| w[0] <= w[1]));
///
/// let mut words = vec!["pear", "apple", "fig"];
/// shell(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn shell<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    // An ascending run stops at the first descent; a strictly descending one at the first
    // element that is not smaller, so reversing it is all it takes.
    let descending = arr[1] < arr[0];
    let run = 2 + if descending {
        arr[1..].windows(2).take_while(|w| w[1] < w[0]).count()
    } else {
        arr[1..].windows(2).take_while(|w| w[1] >= w[0]).count()
    };
    if run == len {
        if descending {
            arr.reverse();
        }
        return;
    }

    let mut gaps = [0; MAX_GAPS];
    if mem::size_of::<T>() <= WINDOW_MAX_SIZE {
        let count = ciura_gaps(len, &mut gaps);
        // Whether the last pass moved many elements: the input is still far from sorted, and
        // the next pass had better not branch on every element.
        let mut busy = true;
        for &gap in gaps[..count].iter().rev() {
            let moved = if busy && gap > 1 {
                pass_window(arr, gap)
            } else {
                pass(arr, gap)
            };
            busy = moved > len / BUSY;
        }
    } else {
        let count = sedgewick_gaps(len, &mut gaps);
        for &gap in gaps[..count].iter().rev() {
            pass(arr, gap);
        }
    }
}

/// Room for the gaps of any slice: both sequences grow by a factor of at least 2.25.
const MAX_GAPS: usize = 64;

/// The largest element size, in bytes, for which passes may insert through a window
/// ([`pass_window`]) and the gaps are Ciura's; larger elements use Sedgewick's gaps and plain
/// insertion only.
///
/// Measured on random input: integers sort 1.4–1.8 times faster with the window, and `u128`
/// 1.15–1.4 times; but `&str` and `String`, as small as a `u128` and slower to compare, sort
/// 1.1–1.45 times slower with it on input with few distinct keys or in a sawtooth pattern;
/// 32-byte elements gain on random input but lose up to 1.75 times elsewhere, and 64-byte
/// ones lose on all input.
const WINDOW_MAX_SIZE: usize = 8;

/// How many steps [`pass_window`] takes without branching.
///
/// Measured against 3 and 6: 1.05–1.1 times faster than 3 on random input from 100,000
/// elements on, and up to 1.1 times slower around 1000 elements; 6 is slower on all input.
const WINDOW: usize = 4;

/// The next pass inserts through a window if this pass moved more than `1 / BUSY` of the
/// elements, and plainly otherwise.
///
/// Measured against 2 and 8: within 3% of 8, and 1.05–1.2 times faster than 2.
const BUSY: usize = 4;

/// Writes the gaps below `n` of Ciura's sequence, `1, 4, 10, 23, 57, 132, 301, 701`, extended
/// by multiplying by 2.25 and rounding down, into `gaps` in increasing order, and returns how
/// many there are. Found by experiment to make about the fewest comparisons of any sequence.
fn ciura_gaps(n: usize, gaps: &mut [usize; MAX_GAPS]) -> usize {
    const CIURA: [usize; 8] = [1, 4, 10, 23, 57, 132, 301, 701];
    let mut count = 0;
    let mut gap = 1;
    while gap < n {
        gaps[count] = gap;
        count += 1;
        gap = match CIURA.get(count) {
            Some(&next) => next,
            // `⌊2.25·gap⌋`; past `usize::MAX`, the gap is too large for any slice.
            None => gap.checked_mul(9).map_or(usize::MAX, |g| g / 4),
        };
    }
    count
}

/// Writes the gaps below `n` of Sedgewick's 1982 sequence, `1` and `4^k + 3·2^(k−1) + 1`
/// (`8, 23, 77, 281, 1073, ...`), into `gaps` in increasing order, and returns how many there
/// are. Its gaps grow by a factor of about 4, so it takes fewer passes than Ciura's.
fn sedgewick_gaps(n: usize, gaps: &mut [usize; MAX_GAPS]) -> usize {
    let mut count = 0;
    let mut gap = 1;
    let mut k = 0u32;
    while gap < n {
        gaps[count] = gap;
        count += 1;
        k += 1;
        // Past `usize::MAX`, the gap is too large for any slice.
        gap = 4usize
            .checked_pow(k)
            .and_then(|p| p.checked_add(3 << (k - 1)))
            .map_or(usize::MAX, |g| g.saturating_add(1));
    }
    count
}

/// Sorts each chain `v[c], v[c + gap], v[c + 2·gap], ...` by insertion: each element is
/// carried back along its chain past the greater elements before it, which shift one place
/// down the chain. Returns how many elements moved.
fn pass<T: Ord>(v: &mut [T], gap: usize) -> usize {
    let len = v.len();
    let base = v.as_mut_ptr();
    let mut moved = 0;
    for i in gap..len {
        // SAFETY: `gap <= i < len`, so both elements are inside `v`.
        if unsafe { *base.add(i) >= *base.add(i - gap) } {
            continue;
        }
        moved += 1;
        // SAFETY: every offset is `i - m·gap >= 0` for some `m`, inside `v`. The element is
        // read out, and `hole.dst` is always the one vacant slot: each step shifts the chain
        // element before it into it. If a comparison panics, dropping `hole` writes the
        // element into the vacant slot, so `v` holds every element exactly once.
        unsafe {
            let x = ManuallyDrop::new(ptr::read(base.add(i)));
            ptr::copy_nonoverlapping(base.add(i - gap), base.add(i), 1);
            let mut hole = Hole {
                src: &*x,
                dst: base.add(i - gap),
            };
            let mut j = i - gap;
            while j >= gap && *x < *base.add(j - gap) {
                ptr::copy_nonoverlapping(base.add(j - gap), base.add(j), 1);
                j -= gap;
                hole.dst = base.add(j);
            }
            // Dropping `hole` writes the element into the vacant slot.
        }
    }
    moved
}

/// Like [`pass`], but takes the first [`WINDOW`] steps of each insertion without branching.
/// Returns how many elements moved.
///
/// Each element is compared with the `WINDOW` elements before it on its chain, and `k`, the
/// number of those (counted from the nearest, up to the first that is not greater) it moves
/// past, is computed as a number. Then each of those `WINDOW` slots is rewritten from a copy
/// chosen by an index computed from `k`: the slot `m` places back takes the element `m + 1`
/// places back if `m < k`, the element itself if `m == k`, and its own element otherwise. Only
/// an element that moves past all `WINDOW` goes on along its chain with a branch per step.
///
/// The comparisons beyond the first that fails are made for nothing, but in a pass of a
/// partly sorted slice how far an element moves is about as hard to predict as a coin toss,
/// and each misprediction costs more than several comparisons of small elements.
fn pass_window<T: Ord>(v: &mut [T], gap: usize) -> usize {
    let len = v.len();
    let head = WINDOW.saturating_mul(gap).min(len);
    let mut moved = pass(&mut v[..head], gap);
    let base = v.as_mut_ptr();
    for i in head..len {
        // SAFETY: `i >= WINDOW·gap`, so every `i - m·gap` with `m <= WINDOW` is inside `v`.
        // The elements of the window are copied out bit for bit into `window` before any is
        // compared or written: `window[0]` is the element being inserted, `window[m]` the one
        // `m` places back. A panicking comparison therefore leaves `v` untouched. The writes
        // that follow cannot panic, and whatever `Ord` answered, `k <= WINDOW` and they place
        // `window[0..=k]` into the slots `0..=k` places back in a rotated order and leave the
        // others as they were: a permutation of the window, except when `k == WINDOW`, where
        // the slot `WINDOW` places back is left vacant for the rest of the insertion, which
        // is `pass`'s loop with its hole.
        unsafe {
            let slot = |m: usize| base.add(i - m * gap);
            let mut window: [MaybeUninit<T>; WINDOW + 1] =
                [const { MaybeUninit::uninit() }; WINDOW + 1];
            for (m, copy) in window.iter_mut().enumerate() {
                ptr::copy_nonoverlapping(slot(m), copy.as_mut_ptr(), 1);
            }
            let x = window[0].as_ptr();
            let mut k = 0;
            let mut before = 1;
            for copy in &window[1..] {
                before &= usize::from(*x < *copy.as_ptr());
                k += before;
            }
            moved += usize::from(k > 0);
            for m in 0..WINDOW {
                let shifted = usize::from(m < k);
                let placed = usize::from(m == k);
                let from = shifted * (m + 1) + (1 - shifted - placed) * m;
                // `from <= m + 1 <= WINDOW`; indexing `window` with a check instead made the
                // whole sort up to 1.2 times slower.
                ptr::copy_nonoverlapping(window.as_ptr().add(from).cast::<T>(), slot(m), 1);
            }
            if k == WINDOW {
                let mut hole = Hole {
                    src: x,
                    dst: slot(WINDOW),
                };
                let mut j = i - WINDOW * gap;
                while j >= gap && *x < *base.add(j - gap) {
                    ptr::copy_nonoverlapping(base.add(j - gap), base.add(j), 1);
                    j -= gap;
                    hole.dst = base.add(j);
                }
                // Dropping `hole` writes the element into the vacant slot.
            }
        }
    }
    moved
}

/// An element held outside a slice and the vacant slot it belongs in: dropping it, when the
/// insertion is done or a comparison panics, writes the element there.
struct Hole<T> {
    src: *const T,
    dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` is the vacant slot of the slice, and `src` the element read out of it.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) }
    }
}
