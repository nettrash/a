use core::hint::select_unpredictable;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::Range;
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
///   number, and the 4 slots it can move into are rewritten from copies held in registers,
///   each chosen by two selects on that number: no branch, at the cost of comparisons that
///   turn out unnecessary. Only the 8% of elements that pass all 4 go on with a branch per
///   step. On random input this is 1.65–1.9 times faster for integers.
/// * **The window only where it pays, judged by the insertions themselves.** Where the slice
///   is nearly sorted or its order follows a pattern, insertion's branches are predictable and
///   the window's extra comparisons and writes are wasted: a pass can be up to 5 times slower
///   with it. How many elements move does not tell the two apart (sawtooth input moves as
///   many as random input), but how regularly they move does. So a pass is made in blocks of
///   512 elements, and each block counts, at no measurable cost, how predictable its insertions
///   were and chooses the window for the next block only if they were not. Inserted plainly,
///   it counts the runs of consecutive elements that moved and the steps they took beyond the
///   first, and takes `2·runs + 0.35·steps > 0.42` per element as unpredictable; through the
///   window, it counts how often the number of elements passed differs from the one before,
///   and takes more than `0.42` per element as unpredictable. As branch predictors also learn
///   short patterns, the last 32 elements of each block note their steps too, and a block whose
///   steps repeat with a period of at most 16 is followed by plain insertion. The first pass
///   assumes the window pays, and the final pass, with gap 1, never takes it, since each
///   element there depends on the one just placed. Against taking the window whenever the
///   pass before moved more than a quarter of the elements, this is 1.2–1.85 times faster on
///   sawtooth input of 1000 to a million integers with periods from 4 to `n/8` (but a period
///   of `√n` from 100,000 elements on, as fast to 1.1 times slower), 2.1–2.45 times on
///   organ-pipe input, 1.05–1.15 times on nearly sorted input, 1.05–1.25 times with 4 distinct
///   keys, and as fast on random input (within 3%; sorted runs of about `√n` random elements,
///   which only the old count read right, are 1.03–1.25 times slower below 10,000 elements).
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
/// this is 1.65–1.9 times faster on random integers from 1000 elements on, up to 1.75 times on
/// nearly sorted ones, 1.3–1.4 times with few distinct keys and 1.1–1.2 times on organ-pipe
/// input; on sawtooth input of 1000–100,000 integers with periods from 4 to `n/8` it ranges
/// from 1.15 times slower (a period of 16 at 1000 elements, which the branch predictor learns
/// from history the window cannot see) to 1.35 times faster, mostly within 1.1 times either
/// way; 1.05–1.4 times faster on strings except random ones from a million elements on, and
/// 1.03–1.3 times on 64-byte elements. Against this crate's [`comb`](crate::sort::comb) sort,
/// whose fixed compare-exchanges the compiler vectorizes for integers: 3.55–4 times slower on
/// random integers from 1000 elements on and 1.8–2.65 times on nearly sorted ones or with few
/// distinct keys (1.15 times faster on 100 nearly sorted ones), but 1.1–2.5 times faster on
/// strings that have few distinct keys or follow a pattern, and 1.6 times on 1000 nearly
/// sorted strings (as fast on longer ones).
/// Against [`heap`](crate::sort::heap) sort: 1.45–1.8 times slower on random integers from
/// 1000 elements on, 1.1–1.6 times on nearly sorted ones and 1.1–1.25 times with few distinct
/// keys, about as fast on random strings, and faster on strings that are nearly sorted or
/// patterned.
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
        // Whether the elements inserted last were hard to predict, so that the next ones had
        // better go through the window; the first pass assumes they were.
        let mut window = true;
        for &gap in gaps[..count].iter().rev() {
            if gap > 1 {
                window = pass_choosing(arr, gap, window);
            } else {
                insert(arr, 1, 1..len, &mut ());
            }
        }
    } else {
        let count = sedgewick_gaps(len, &mut gaps);
        for &gap in gaps[..count].iter().rev() {
            insert(arr, gap, gap..len, &mut ());
        }
    }
}

/// Room for the gaps of any slice: both sequences grow by a factor of at least 2.25.
const MAX_GAPS: usize = 64;

/// The largest element size, in bytes, for which passes may insert through a window
/// ([`insert_window`]) and the gaps are Ciura's; larger elements use Sedgewick's gaps and
/// plain insertion only.
///
/// Measured on random input: integers sort 1.65–1.9 times faster with the window, and `u128`
/// 1.15–1.4 times; but `&str` and `String`, as small as a `u128` and slower to compare, sort
/// 1.1–1.45 times slower with it on input with few distinct keys or in a sawtooth pattern;
/// 32-byte elements gain on random input but lose up to 1.75 times elsewhere, and 64-byte
/// ones lose on all input.
const WINDOW_MAX_SIZE: usize = 8;

/// How many steps [`insert_window`] takes without branching.
///
/// Measured against 3 and 6: 1.05–1.1 times faster than 3 on random input from 100,000
/// elements on, and up to 1.1 times slower around 1000 elements; 6 is slower on all input.
const WINDOW: usize = 4;

/// How many elements [`pass_choosing`] inserts the same way before it chooses again.
///
/// Measured against 256 and 1024: within 2% of either on random input; 256 is up to 1.1
/// times slower on sawtooth input, whose blocks then differ more from one to the next.
const BLOCK: usize = 512;

/// How many elements of the head of a pass, where the window cannot reach, it takes to
/// choose from them: fewer, and the choice made at the end of the pass before stands.
const SAMPLE: usize = 128;

/// How many elements at the end of each block note their steps for [`History::periodic`].
const HISTORY: usize = 32;

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

/// Makes the pass with gap `gap > 1` over a slice of small elements in blocks of [`BLOCK`]
/// elements, each inserted plainly or through the window, whichever the block before it found
/// to pay; `window` is the choice the pass before ended with. Returns the choice this pass
/// ends with.
///
/// Plain insertion is fast when its branches are predictable, and the window when they are
/// not, so a block chooses the window for the next one if its own insertions were hard to
/// predict. Inserted plainly, a block counts the runs of consecutive elements that moved and
/// the steps they took beyond the first: a branch predictor guesses well inside runs, and
/// each extra step is another branch that ends the insertion somewhere. Through the window it
/// counts how often the number of places an element passes differs from the one before it.
/// Each count is free: the first two are kept on the path of elements that move only, the
/// third replaces a count the window kept before. And as a branch predictor also learns short
/// patterns, the last [`HISTORY`] elements of the block note how far each moved, and a block
/// whose steps repeat with a short period ([`History::periodic`]) chooses plain insertion.
fn pass_choosing<T: Ord>(v: &mut [T], gap: usize, mut window: bool) -> bool {
    let len = v.len();
    // The window reaches back `WINDOW` places along the chain, so the elements before `head`
    // are inserted plainly.
    let head = WINDOW.saturating_mul(gap).min(len);
    let mut start = gap;
    while start < head {
        let end = start + (head - start).min(BLOCK);
        let mut runs = Runs::new(start);
        insert(v, gap, start..end, &mut runs);
        if end - start >= SAMPLE {
            window = runs.unpredictable(gap, end - start);
        }
        start = end;
    }
    while start < len {
        let end = start + (len - start).min(BLOCK);
        // The last `HISTORY` elements note their steps, unless the block is short.
        let split = if end - start >= 2 * HISTORY {
            end - HISTORY
        } else {
            end
        };
        let unpredictable = if window {
            let mut changes = Changes::default();
            insert_window(v, gap, start..split, &mut changes);
            changes.unpredictable(split - start)
        } else {
            let mut runs = Runs::new(start);
            insert(v, gap, start..split, &mut runs);
            runs.unpredictable(gap, split - start)
        };
        if split < end {
            let mut history = History::default();
            if window {
                insert_window(v, gap, split..end, &mut history);
            } else {
                insert(v, gap, split..end, &mut history);
            }
            window = unpredictable && !history.periodic();
        } else {
            window = unpredictable;
        }
        start = end;
    }
    window
}

/// What an insertion pass notes about the elements it inserts.
trait Tally {
    /// Plain insertion left the element where it was.
    #[inline(always)]
    fn stayed(&mut self) {}

    /// Plain insertion carried element `i` back along its chain to `j < i`.
    #[inline(always)]
    fn moved(&mut self, i: usize, j: usize, gap: usize) {
        let _ = (i, j, gap);
    }

    /// The window found that element passed `k` of the [`WINDOW`] elements before it on its
    /// chain (all of them if `k == WINDOW`, after which it went on).
    #[inline(always)]
    fn passed(&mut self, k: usize) {
        let _ = k;
    }
}

/// Notes nothing.
impl Tally for () {}

/// The runs of consecutive elements that plain insertion moved, and the steps they took
/// beyond the first.
struct Runs {
    runs: usize,
    /// The extra steps times the gap: `i - gap - j` summed, wrapping rather than overflowing
    /// for slices of zero-sized elements longer than memory could hold.
    extra: u64,
    /// The last element that moved, or one at least two before the first of the block.
    last: usize,
}

impl Runs {
    fn new(start: usize) -> Self {
        Runs {
            runs: 0,
            extra: 0,
            last: start.wrapping_sub(2),
        }
    }

    /// Whether `2·runs + 0.35·extra steps > 0.42·n` for the `n` elements counted: the next
    /// elements had better go through the window.
    ///
    /// Fitted to the time of each pass in either way, for integers of 100 to a million
    /// elements in 13 patterns: 1.03 times the time of the faster way of each pass on
    /// average, against 1.25 times for choosing the window when a quarter of the elements of
    /// the pass before moved.
    fn unpredictable(&self, gap: usize, n: usize) -> bool {
        let extra = self.extra / gap as u64;
        let score = (200 * self.runs as u64).saturating_add(extra.saturating_mul(35));
        score > 42 * n as u64
    }
}

impl Tally for Runs {
    #[inline(always)]
    fn moved(&mut self, i: usize, j: usize, gap: usize) {
        self.runs += usize::from(self.last.wrapping_add(1) != i);
        self.last = i;
        self.extra = self.extra.wrapping_add((i - gap - j) as u64);
    }
}

/// How often the number of elements the window found an element passed differs from the
/// one before it.
#[derive(Default)]
struct Changes {
    changes: usize,
    last: usize,
}

impl Changes {
    /// Whether the number changed for more than `0.42` of the `n` elements counted: the next
    /// elements had better go through the window too.
    fn unpredictable(&self, n: usize) -> bool {
        100 * self.changes > 42 * n
    }
}

impl Tally for Changes {
    #[inline(always)]
    fn passed(&mut self, k: usize) {
        self.changes += usize::from(k != self.last);
        self.last = k;
    }
}

/// The steps of the last 32 elements inserted, 0 to 3 (or more) in 2 bits each, the latest
/// in the lowest bits.
#[derive(Default)]
struct History(u64);

impl History {
    /// Whether the steps repeat with a period of at most 16 elements, but for one in 16
    /// mismatches: a pattern a branch predictor learns, so the next elements had better be
    /// inserted plainly however often the steps change. Sawtooth input with a period of 4
    /// sorts up to 1.25 times faster with this, and random input as fast.
    fn periodic(&self) -> bool {
        const LOW: u64 = 0x5555_5555_5555_5555;
        let steps = self.0;
        (1..=16u32).any(|lag| {
            let differ = steps ^ (steps >> (2 * lag));
            // One bit per pair of steps `lag` apart that differ, for the `32 - lag` pairs.
            let mismatches = (differ | (differ >> 1)) & LOW & (u64::MAX >> (2 * lag));
            mismatches.count_ones() * 16 <= 32 - lag
        })
    }
}

impl Tally for History {
    #[inline(always)]
    fn stayed(&mut self) {
        self.0 <<= 2;
    }

    #[inline(always)]
    fn moved(&mut self, i: usize, j: usize, gap: usize) {
        let distance = i - j;
        let steps = 1 + u64::from(distance > gap) + u64::from(distance > 2 * gap);
        self.0 = (self.0 << 2) | steps;
    }

    #[inline(always)]
    fn passed(&mut self, k: usize) {
        self.0 = (self.0 << 2) | k.min(3) as u64;
    }
}

/// Inserts each element of `range` into the chain `v[i - gap], v[i - 2·gap], ...` before
/// it: the element is carried back past the greater elements before it, which shift one
/// place down the chain, and `tally` notes where it went.
///
/// # Panics
///
/// If `range` does not lie within `gap..v.len()`.
#[inline(always)]
fn insert<T: Ord, N: Tally>(v: &mut [T], gap: usize, range: Range<usize>, tally: &mut N) {
    assert!(gap <= range.start && range.end <= v.len());
    let base = v.as_mut_ptr();
    for i in range {
        // SAFETY: `gap <= i < len`, so both elements are inside `v`.
        if unsafe { *base.add(i) >= *base.add(i - gap) } {
            tally.stayed();
            continue;
        }
        // SAFETY: every offset is `i - m·gap >= 0` for some `m`, inside `v`. The element is
        // read out, and `hole.dst` is always the one vacant slot: each step shifts the chain
        // element before it into it. If a comparison panics, dropping `hole` writes the
        // element into the vacant slot, so `v` holds every element exactly once.
        let j = unsafe {
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
            j
        };
        tally.moved(i, j, gap);
    }
}

/// Like [`insert`], but takes the first [`WINDOW`] steps of each insertion without
/// branching, and `tally` notes how many of them each element took.
///
/// Each element is compared with the `WINDOW` elements before it on its chain, and `k`, the
/// number of those (counted from the nearest, up to the first that is not greater) it moves
/// past, is computed as a number. Then each of those `WINDOW` slots is rewritten from a copy
/// chosen by two selects on `k`: the slot `m` places back takes the element `m + 1` places
/// back if `m < k`, the element itself if `m == k`, and its own element otherwise. Only
/// an element that moves past all `WINDOW` goes on along its chain with a branch per step.
///
/// The comparisons beyond the first that fails are made for nothing, but in a pass of a
/// partly sorted slice how far an element moves is about as hard to predict as a coin toss,
/// and each misprediction costs more than several comparisons of small elements.
///
/// # Panics
///
/// If `range` does not lie within `WINDOW·gap..v.len()`.
#[inline(always)]
fn insert_window<T: Ord, N: Tally>(v: &mut [T], gap: usize, range: Range<usize>, tally: &mut N) {
    assert!(WINDOW.saturating_mul(gap) <= range.start && range.end <= v.len());
    let base = v.as_mut_ptr();
    for i in range {
        // SAFETY: `i >= WINDOW·gap`, so every `i - m·gap` with `m <= WINDOW` is inside `v`.
        // The elements of the window are copied out bit for bit into `window` before any is
        // compared or written: `window[0]` is the element being inserted, `window[m]` the one
        // `m` places back. A panicking comparison therefore leaves `v` untouched. The writes
        // that follow cannot panic, and whatever `Ord` answered, `k <= WINDOW` and they place
        // `window[0..=k]` into the slots `0..=k` places back in a rotated order and leave the
        // others as they were: a permutation of the window, except when `k == WINDOW`, where
        // the slot `WINDOW` places back is left vacant for the rest of the insertion, which
        // is `insert`'s loop with its hole.
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
            tally.passed(k);
            for m in 0..WINDOW {
                // The slot `m` places back takes the element `m + 1` places back if `m < k`,
                // the element itself if `m == k`, and keeps its own otherwise: two selects of
                // copies held in registers. Loading the copy from `window` at an index
                // computed from `k`, which keeps the compiler from branching just as well,
                // goes through memory and was 1.05–1.18 times slower.
                let own =
                    select_unpredictable(m == k, ptr::read(&window[0]), ptr::read(&window[m]));
                let value = select_unpredictable(m < k, ptr::read(&window[m + 1]), own);
                slot(m).cast::<MaybeUninit<T>>().write(value);
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
