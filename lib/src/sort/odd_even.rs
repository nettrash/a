use core::hint::select_unpredictable;
use core::mem::{ManuallyDrop, MaybeUninit, size_of};
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::panic::resume_unwind;
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread;

/// Sorts a slice in place using odd-even sort.
///
/// Odd-even sort, also called brick sort or odd-even transposition sort, compares neighbours
/// in alternating phases. An *odd* phase compare-exchanges the pairs `(1, 2)`, `(3, 4)`, ...,
/// and an *even* phase the pairs `(0, 1)`, `(2, 3)`, ...: each pair is swapped if its second
/// element is smaller. An odd phase followed by an even phase is a round, and the textbook
/// version repeats rounds until one swaps nothing. No two pairs of a phase share an element, so
/// all the compare-exchanges of a phase are independent and could run at once: the algorithm
/// comes from parallel computing, where a row of processors that only talk to their neighbours
/// sorts `n` elements in `n` phases (Habermann's "parallel neighbor-sort", 1972), and it is the
/// sorting network of such a row. On one processor it is bubble sort with each pass cut into
/// independent pairs; [`odd_even_parallel`] runs it on several threads.
///
/// Six refinements keep the realisation as cheap as odd-even sort can be. None of them
/// changes which compare-exchanges run in which phase, so every swap is one the textbook
/// version makes, and the slice after every round is the same.
///
/// * **Bounds from the last swaps.** A pair that is in order stays in order until one of its
///   elements changes, and in the odd phase of a round an element can only have changed if the
///   even phase before it swapped it. So the odd phase only compares the pairs from one before
///   the first to one after the last pair that the previous even phase swapped, and the even
///   phase one pair more at each end, which covers every pair the odd phase can have disturbed;
///   all other pairs are known to be in order. The textbook version compares all `n - 1` pairs
///   in every round. This is the last-swap bound of this crate's [`bubble`](fn@super::bubble)
///   sort at both ends: on random input it saves 7% of the comparisons for 1000 elements (2%
///   for 10,000), and makes nearly sorted input 1.25–1.8 times faster, input with few distinct
///   keys 1.2–1.4 times and an element out of place at either end (a "turtle" or a "rabbit")
///   cost about `3.5n` comparisons instead of about `n²/2`.
/// * **At most `⌈n/2⌉` rounds.** `n` phases sort any input (by the 0-1 principle: Knuth, *The
///   Art of Computer Programming* 3, 5.3.4), so the sort stops after `⌈n/2⌉` rounds even if the
///   last one swapped. This saves the textbook's final round, which finds nothing to swap, on
///   input that needs every phase (reverse-sorted input among it); above all, it guarantees that
///   the sort ends with an inconsistent [`Ord`], whose answers would otherwise almost never let
///   a round pass without a swap.
/// * **One sweep per round.** The even pair `(k, k + 1)` depends only on the odd pairs on either
///   side of it, `(k - 1, k)` and `(k + 1, k + 2)`. So a single sweep can compare an odd pair and
///   then the even pair just behind it: a wavefront that runs the even phase on each part of the
///   slice as soon as the odd phase has finished there, making the same compare-exchanges, each
///   on the same two elements as if the phases ran one after the other. Each element is loaded
///   once per round instead of twice: 2–2.2 times faster for integers, up to 1.7 times for
///   strings and 1.6 times for 64-byte elements.
/// * **Branchless compare-exchanges for elements of at most 8 bytes.** On random input whether
///   a pair swaps is close to a coin flip. Both elements are read, and the smaller and the
///   larger written back, chosen by the comparison instead of a branch, so that the compiler
///   turns a sweep of 32 odd and 32 even pairs into vector code for integers; the pairs that
///   swapped come out as a bit mask, which gives the bounds. This is 1.3–2 times faster than
///   branching on random input or few distinct keys, and 2–10% slower on nearly sorted input,
///   where the branches are predictable. Sweeps of 64 pairs are 1.2–1.25 times slower (more
///   than the processor can keep in flight), and 16-byte elements 2.4–3.9 times slower without
///   branches than with them.
/// * **Elements of at most 4 bytes split into even and odd positions.** For these, a chunk of
///   32 steps is moved out of the slice into two small arrays, one for the elements at even
///   positions and one for those at odd ones, and back after both phases; the element where
///   the next chunk starts is carried over in a register. Every pair of a phase then sits at
///   the same place of the two arrays (one place apart for the odd phase), and the compiler
///   runs each phase as a few vector minimums and maximums, where in the slice it picked
///   lanes apart and widened every swap bit of a narrow element to 64 bits for the mask. The
///   same pairs are compared and the same swaps made, in the same rounds, but the elements are
///   compared where the chunk holds them, outside the slice: 1.3–1.5 times faster for 4-byte
///   elements from 1000 elements on, 1.7–2.35 times for 2-byte and 3.3–4.3 times for 1-byte
///   ones, on every input that takes more than one round (1.05–1.5 times on 100); for `u64`
///   it made no difference.
/// * **For larger elements, swaps by plain copies, with the even phase two pairs behind.** A
///   swap reads both elements and writes them back crossed; [`core::mem::swap`] compiled to
///   copies that the processor could not forward to the next comparison, 1.2–1.9 times slower
///   for 16- and 64-byte elements. For the same reason the even phase compares its pair two
///   pairs behind the odd phase instead of right behind it: 1.07–1.45 times faster for 64-byte
///   elements, about the same for others.
///
/// # Properties
///
/// * **Stable**: only neighbours are swapped, and only when the second is strictly smaller, so
///   equal elements never pass each other.
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the bounds shrink to the part of the slice that is still changing.
///
/// Every swap removes exactly one inversion (a pair of elements out of order), so the sort
/// makes exactly `I` swaps, the number of inversions, as the textbook version does. Comparisons
/// are never more than the textbook's.
///
/// | Case                          | Comparisons                                     | Textbook                       | Swaps          |
/// |-------------------------------|-------------------------------------------------|--------------------------------|----------------|
/// | Sorted                        | `n - 1`                                         | `n - 1`                        | `0`            |
/// | Reverse sorted                | `n(n - 1)/2` (`n` even), `(n² - 1)/2` (`n` odd) | the same plus `n - 1`          | `n(n - 1)/2`   |
/// | Turtle `[1, 2, ..., n - 1, 0]`| `(7n - 16)/2` (`n` even), `(7n - 13)/2` (odd)   | `(n - 1)(⌊n/2⌋ + 1)`           | `n - 1`        |
/// | Random (`n = 1000`)           | `≈ 453,400`                                     | `≈ 486,800`                    | `≈ n²/4`       |
/// | Worst                         | `O(n²)`                                         | `O(n²)`                        | `n(n - 1)/2`   |
///
/// Against the textbook version, measured on 1000 and 5000 elements: 1.15–3.4 times faster for
/// integers (1.15 on reverse-sorted input, 2–2.6 on random), 1.03–1.75 times for strings and
/// 1.0–2.1 times for 64-byte elements. Against this crate's [`bubble`](fn@super::bubble) sort,
/// 1.6–1.7 times faster on random integers, but on strings and 64-byte elements from 1.1 times
/// faster to 2 times slower (reverse-sorted 64-byte elements): bubble sort shifts an element
/// one write per slot instead of swapping. [`cocktail_shaker`](fn@super::cocktail_shaker)
/// sort makes a third fewer comparisons on random input and carries a turtle home in one pass:
/// odd-even sort is 1.45 times faster on random integers and 2.1 times on few distinct keys,
/// but 1.2–1.4 times slower on random strings, 1.5–2.2 times on 64-byte elements and 20–50
/// times on nearly sorted input. Odd-even sort is quadratic: on 1000 random integers it
/// is about 20 times slower than [`slice::sort_unstable`], on 5000 about 85 times; prefer
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
/// use a::sort::odd_even;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// odd_even(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // A turtle costs a few comparisons per round, not a pass over the whole slice.
/// let mut v = [1, 2, 3, 4, 5, 0];
/// odd_even(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// odd_even(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn odd_even<T: Ord>(arr: &mut [T]) {
    if arr.len() < 2 {
        return;
    }
    let mut rounds = Rounds::new(arr.len());
    rounds.run(arr, usize::MAX);
}

/// Sorts a slice in place using odd-even sort on several threads.
///
/// The result is that of [`odd_even`], and so is every swap: this runs the same phases with the
/// same refinements, on [`std::thread::scope`] threads (no dependencies) and as many of them as
/// [`std::thread::available_parallelism`] reports.
///
/// The compare-exchanges of a phase are independent, but a thread per pair (the model the
/// algorithm was designed for) or a barrier after every phase would cost far more than the
/// compare-exchanges themselves. Instead, the threads run many phases between two barriers, by
/// tiling the plane of pairs and phases ("temporal blocking", as for stencil computations):
///
/// * The slice is cut into blocks, which the threads claim in turn. On its block a thread runs
///   `K` phases (a *super-step*), each one leaving out one more pair at each end than the one
///   before, where the pair would need an element from the neighbouring block that has not yet
///   reached that phase: the pairs it compares form a trapezoid.
/// * The pairs left out form triangles around the block boundaries. In the next super-step the
///   slice is cut at the midpoints of the previous blocks, so that each new block holds one
///   triangle: the thread first completes the previous super-step there, and then runs the
///   trapezoid of the next `K` phases on the whole block.
///
/// Every compare-exchange still runs exactly once, after the two it depends on (the pairs on
/// either side of it in the phase before), so the slice goes through the same states as in
/// [`odd_even`]: this is still odd-even transposition sort, unlike the block odd-even
/// merge-split sort (Baudet and Stevenson, 1978), which sorts blocks and merges neighbours, a
/// different algorithm. Threads meet only once per super-step, at a barrier where they combine
/// the bounds of the last complete phase and decide whether to go on: the first super-step of
/// a run is 8 phases and each next one twice as long, up to half a block, so a slice that
/// is nearly done does not pay for a long super-step. A run stops at the first barrier after a
/// phase that swapped nothing, so it may compare up to two super-steps more than [`odd_even`];
/// and its phases cover a window around the bounds rather than the bounds themselves, which
/// costs it more comparisons than [`odd_even`] on input whose unsorted part shrinks (1.4 times
/// on nearly sorted input of 5500 elements), but every one of them is in parallel. When the
/// unsorted part has shrunk to half the window, the run ends and a new one starts on a window
/// around what is left.
///
/// Threads are not worth starting for short slices or for input that is sorted soon: below
/// 4096 elements of at most 8 bytes (2048 larger ones) the sort runs on the calling thread
/// only, and above, the first rounds do too, until `2²⁰` (`2¹⁶`) pairs have been compared or
/// 64 rounds have run. Measured on a 16-core machine (12 performance cores) under other load,
/// against [`odd_even`]: 4.3–6 times faster on 20,000 integers, 7–8 times on strings and
/// 6.6–10.5 times on 64-byte elements; on 8192 elements 1.9–3.8, 4.4–6.5 and 3.5–6.6 times; on
/// 4096, 1.3–2.1, 2.7–4 and 2–4.4 times. On 2048 strings it is 1.5–2.5 times faster. Elements
/// of at most 4 bytes, split into even and odd positions as in [`odd_even`], sort 1.2–1.25
/// (`u32`) and 2.1–2.2 (`u8`) times faster than without that on 8192 and 20,000 elements.
///
/// # Panics
///
/// Does not panic by itself, except if the system cannot start a thread, which it treats by
/// sorting on the calling thread. If the [`Ord`] implementation of `T` panics, on any thread,
/// the other threads stop, the panic is propagated to the caller with its original payload,
/// and `arr` is left in an unspecified order, but it still holds every original element exactly
/// once: nothing is dropped twice or leaked.
///
/// # Examples
///
/// ```
/// use a::sort::odd_even_parallel;
///
/// let mut v: Vec<u32> = (0..10_000).rev().collect();
/// odd_even_parallel(&mut v);
/// assert!(v.windows(2).all(|w| w[0] <= w[1]));
///
/// let mut words = vec!["pear", "apple", "fig"];
/// odd_even_parallel(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn odd_even_parallel<T: Ord + Send>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }
    let mut rounds = Rounds::new(n);
    let min_len = parallel_min_len(size_of::<T>());
    if n >= min_len {
        rounds.run(arr, (warmup_pairs(size_of::<T>()) / n).max(WARMUP_ROUNDS));
        // Asked only now: most of what finishes in the warm-up is sorted input, for which the
        // system call would be most of the cost.
        let mut threads = None;
        while let Some((lo, m)) = rounds.next {
            // The next round compares pairs from `lo - 1` to `lo + 2m - 1`.
            let e0 = lo - 1;
            let e1 = (lo + 2 * m + 1).min(n);
            let active = e1 - e0;
            let threads = *threads
                .get_or_insert_with(|| thread::available_parallelism().map_or(1, |p| p.get()));
            if threads < 2 || active < min_len / 2 {
                break;
            }
            let margin = active / 2;
            let w0 = e0.saturating_sub(margin);
            let w1 = (e1 + margin).min(n);
            if !session(arr, w0, w1, &mut rounds, threads) {
                break;
            }
        }
    }
    rounds.run(arr, usize::MAX);
}

/// The slice length from which [`odd_even_parallel`] uses threads, for elements of `size`
/// bytes; a run of the threads also needs the part of the slice still unsorted to be at least
/// half of it. Measured: integers are 1.2–1.35 times slower on 3000 elements with threads and
/// 1.3–2 times faster on 4096; strings and 64-byte elements 1.5–2.5 times faster on 2048, but
/// some 64-byte inputs slower on 1536.
///
/// Under Miri, which interprets every step thousands of times slower, this and the other
/// sizes of the parallel sort are scaled down, so that its checks reach the threads at all.
const fn parallel_min_len(size: usize) -> usize {
    if cfg!(miri) {
        64
    } else if size <= BRANCHLESS_MAX_SIZE {
        4096
    } else {
        2048
    }
}

/// Pairs compared on one thread before [`odd_even_parallel`] starts any thread: a thread costs
/// tens of microseconds to start, and input that is nearly sorted is often done by then.
const fn warmup_pairs(size: usize) -> usize {
    if cfg!(miri) {
        256
    } else if size <= BRANCHLESS_MAX_SIZE {
        1 << 20
    } else {
        1 << 16
    }
}

/// The smallest block a parallel session splits a window into. A super-step runs at most half
/// a block's length in phases, and threads meet once per super-step, so small blocks
/// mean many meetings: for integers, blocks of 512 were 1.1–1.7 times slower than 1024 on 8192
/// to 20,000 elements; for larger elements, 256 was 1.1–1.7 times slower than 512 on 64-byte
/// ones, though up to 1.2 times faster on strings.
const fn min_block(size: usize) -> usize {
    if cfg!(miri) {
        16
    } else if size <= BRANCHLESS_MAX_SIZE {
        1024
    } else {
        512
    }
}

/// The fewest rounds run on one thread before [`odd_even_parallel`] starts any thread, so that
/// input whose elements are only a few places out of order (64 rounds sort elements up to 128
/// places away) never starts them.
const WARMUP_ROUNDS: usize = 64;

/// The length of the first super-step of a session; each next one is twice as long, so that a
/// session that ends early has not paid for a long one. Starting at 32 measured the same.
const FIRST_K: usize = 8;

/// Blocks per thread in a parallel session: threads claim blocks as they finish, so one slowed
/// down (an efficiency core, or another program) takes fewer. One block per thread was up to
/// 1.25 times slower on 20,000 elements.
const BLOCKS_PER_THREAD: usize = 2;

/// The pairs that swapped in a phase: the first and the last, by the index of their first
/// element, or none. Only the ends are kept: the bounds of the next round need nothing else.
#[derive(Clone, Copy, Debug)]
struct Swaps {
    first: usize,
    last: usize,
}

impl Swaps {
    const NONE: Swaps = Swaps {
        first: usize::MAX,
        last: 0,
    };

    /// Whether no pair swapped.
    #[inline(always)]
    fn is_none(self) -> bool {
        self.first == usize::MAX
    }

    /// The swaps of two parts of the same phase.
    #[inline(always)]
    fn merge(self, other: Swaps) -> Swaps {
        Swaps {
            first: self.first.min(other.first),
            last: self.last.max(other.last),
        }
    }

    /// Records that pair `k` swapped; `k` comes after every pair recorded before, which saves a
    /// comparison per swap (keeping a maximum was 4–15% slower for 16-byte elements).
    #[inline(always)]
    fn push(&mut self, k: usize) {
        self.first = self.first.min(k);
        self.last = k;
    }
}

/// The state of the sort between rounds: the bounds of the next round and how many rounds are
/// left.
struct Rounds {
    n: usize,
    /// The odd phase of the next round compares the pairs starting at `lo`, `lo + 2`, ...,
    /// `lo + 2(m - 1)`; `None` once the slice is sorted.
    next: Option<(usize, usize)>,
    /// Rounds left before the cap of `⌈n/2⌉` rounds.
    left: usize,
}

impl Rounds {
    /// The first round: every pair, since nothing is known yet (`n ≥ 2`).
    fn new(n: usize) -> Self {
        Rounds {
            n,
            next: Some((1, (n - 1) / 2)),
            left: n.div_ceil(2),
        }
    }

    /// Records the pairs swapped by the even phase that ended the last round, and sets the
    /// bounds of the next: its odd phase covers the pairs from one before the first to one after
    /// the last of them. No swap, or no round left, ends the sort.
    fn after(&mut self, swaps: Swaps) {
        self.next = if swaps.is_none() || self.left == 0 {
            None
        } else {
            let lo = if swaps.first == 0 { 1 } else { swaps.first - 1 };
            let hi = if swaps.last + 2 < self.n {
                Some(swaps.last + 1)
            } else {
                swaps.last.checked_sub(1)
            };
            match hi {
                Some(hi) if hi >= lo => Some((lo, (hi - lo) / 2 + 1)),
                _ => None,
            }
        };
    }

    /// Runs at most `max` rounds, or until the sort ends.
    fn run<T: Ord>(&mut self, arr: &mut [T], max: usize) {
        let base = arr.as_mut_ptr();
        let mut budget = max;
        while let Some((lo, m)) = self.next {
            if budget == 0 {
                return;
            }
            budget -= 1;
            self.left -= 1;
            let nb = if lo + 2 * m < self.n { m + 1 } else { m };
            // SAFETY: `lo ≥ 1` and the pairs of the round lie within `arr` (see `after`).
            let swaps = unsafe { round(base, lo, m, lo - 1, nb) };
            self.after(swaps);
        }
    }
}

/// The largest element size, in bytes, compare-exchanged without a branch. Branching was
/// 1.3–2 times slower for integers on random input; 16-byte elements were 2.4–3.9 times slower
/// without branches.
const BRANCHLESS_MAX_SIZE: usize = 8;

/// Steps per chunk of the branchless round, one bit of the swap mask each. With a fixed count
/// the compiler unrolls a chunk into vector code; 64 was 1.2–1.25 times slower, 16 within 6%.
const CHUNK: usize = 32;

/// How many steps the second phase trails the first by in the branchy round. Right behind (0),
/// each comparison reads an element the swap just before it wrote, and for 64-byte elements the
/// processor could not forward those writes: 1.07–1.45 times slower than 2.
const LAG: usize = 2;

/// Compares `*a` with `*b` and swaps them if `*b < *a`, with no branch on the outcome: both
/// are read, and the smaller and the larger written back, chosen by the comparison. Returns
/// whether they were swapped.
///
/// # Safety
///
/// `a` and `b` must be valid, distinct and not otherwise in use.
#[inline(always)]
unsafe fn cx_branchless<T: Ord>(a: *mut T, b: *mut T) -> bool {
    // SAFETY: the comparison runs with both elements in place, so a panic leaves them there.
    // Then both are copied out and written back, each into one slot, from the copies chosen by
    // the comparison (`ManuallyDrop`, so that the copies are never dropped); nothing in between
    // can panic, and both slots end up holding one of the two original elements each.
    unsafe {
        let swap = *b < *a;
        let x = ManuallyDrop::new(ptr::read(a));
        let y = ManuallyDrop::new(ptr::read(b));
        let lo = select_unpredictable(swap, &y, &x);
        let hi = select_unpredictable(swap, &x, &y);
        ptr::copy_nonoverlapping::<T>(&**lo, a, 1);
        ptr::copy_nonoverlapping::<T>(&**hi, b, 1);
        swap
    }
}

/// Compares `*a` with `*b` and swaps them if `*b < *a`. Returns whether they were swapped.
///
/// # Safety
///
/// `a` and `b` must be valid, distinct and not otherwise in use.
#[inline(always)]
unsafe fn cx_branchy<T: Ord>(a: *mut T, b: *mut T) -> bool {
    // SAFETY: the comparison runs with both elements in place; the swap reads both out and
    // writes each into the other slot, and nothing in between can panic.
    unsafe {
        if *b < *a {
            let x = ManuallyDrop::new(ptr::read(a));
            let y = ManuallyDrop::new(ptr::read(b));
            ptr::write(a, ManuallyDrop::into_inner(y));
            ptr::write(b, ManuallyDrop::into_inner(x));
            true
        } else {
            false
        }
    }
}

/// Compare-exchanges the pair starting at `k`, with the kernel for elements of `T`'s size.
///
/// # Safety
///
/// `base + k` and `base + k + 1` must be valid and not otherwise in use.
#[inline(always)]
unsafe fn cx<T: Ord>(base: *mut T, k: usize) -> bool {
    // SAFETY: as required of the caller.
    unsafe {
        if size_of::<T>() <= BRANCHLESS_MAX_SIZE {
            cx_branchless(base.add(k), base.add(k + 1))
        } else {
            cx_branchy(base.add(k), base.add(k + 1))
        }
    }
}

/// `c` steps of the wavefront starting at `p`: step `i` compares the pair of the first phase
/// at `p + 2i + 1` and then the pair of the second phase at `p + 2i`, which depends on that
/// one and the one before. Returns the mask of second-phase pairs swapped (bit `i` for step
/// `i`), from which the bounds are read once per chunk; `c ≤ 64`.
///
/// # Safety
///
/// The elements `p .. p + 2c + 1` must be valid and not otherwise in use, and the first-phase
/// pair before `p` must have been compared already (or not need to be).
#[inline(always)]
unsafe fn steps_branchless<T: Ord>(p: *mut T, c: usize) -> u64 {
    let mut mask = 0;
    for i in 0..c {
        // SAFETY: the three elements lie within `p .. p + 2c + 1`.
        unsafe {
            let q = p.add(2 * i);
            cx_branchless(q.add(1), q.add(2));
            let s = cx_branchless(q, q.add(1));
            mask |= (s as u64) << i;
        }
    }
    mask
}

/// The largest element size, in bytes, for which the branchless round moves each chunk out of
/// the slice into a [`Chunk`] of local arrays, one of the elements at even and one of those at
/// odd positions ([`chunk_steps`]).
///
/// Measured against [`steps_branchless`], on 1000 and 5000 elements: 1.3–1.5 times faster for
/// 4-byte elements on every input, 1.7–2.35 times for 2-byte and 3.3–4.3 times for 1-byte
/// elements, and 1.05–1.5 times on 100. For `u64` it made no difference (within 1.5%), and
/// chunks of 16 steps made `u64` 1.15–1.2 times slower: two elements to a vector register
/// leave little to gain. Chunks of 16 or 64 steps made narrow elements 1.5–2.9 times slower
/// than [`CHUNK`]: the compiler no longer keeps the arrays in vector registers.
const SPLIT_MAX_SIZE: usize = 4;

/// An element held outside the slice, and the slot it goes back to when dropped.
struct Carry<T> {
    val: MaybeUninit<T>,
    dst: *mut T,
}

impl<T> Drop for Carry<T> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: `dst` is the vacant slot of the slice that `val` belongs in (see `Chunk`).
        unsafe { ptr::copy_nonoverlapping(self.val.as_ptr(), self.dst, 1) }
    }
}

/// The `2·CHUNK + 1` elements from `p` on, moved out of the slice: `even[i]` is the element of
/// `p + 2i`, `odd[i]` that of `p + 2i + 1` and `last` that of `p + 2·CHUNK`. The element of `p`
/// comes from `carry`, which holds it outside the slice, and dropping the chunk writes every
/// element back but `last`, which goes into `carry` instead, for `p + 2·CHUNK`: the next chunk
/// starts there and reads it from a register, not from a slot the processor has only just
/// been told to write.
struct Chunk<'a, T> {
    even: [MaybeUninit<T>; CHUNK],
    odd: [MaybeUninit<T>; CHUNK],
    last: MaybeUninit<T>,
    p: *mut T,
    carry: &'a mut Carry<T>,
}

impl<T> Drop for Chunk<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: the slots `p .. p + 2·CHUNK` are vacant, their elements in `even` and `odd`,
        // and `carry.val` is vacant, its element in `even[0]` (see `chunk_steps`). Each goes
        // back to its slot, and `last` into `carry`, whose slot `p + 2·CHUNK` is now vacant.
        unsafe {
            for i in 0..CHUNK {
                ptr::copy_nonoverlapping(self.even[i].as_ptr(), self.p.add(2 * i), 1);
                ptr::copy_nonoverlapping(self.odd[i].as_ptr(), self.p.add(2 * i + 1), 1);
            }
            ptr::copy_nonoverlapping(self.last.as_ptr(), self.carry.val.as_mut_ptr(), 1);
            self.carry.dst = self.p.add(2 * CHUNK);
        }
    }
}

/// [`cx_branchless`] on elements held in a [`Chunk`].
///
/// # Safety
///
/// `a` and `b` must be valid, distinct and initialised.
#[inline(always)]
unsafe fn cx_held<T: Ord>(a: *mut MaybeUninit<T>, b: *mut MaybeUninit<T>) -> bool {
    // SAFETY: as in `cx_branchless`: the comparison runs with both elements where they are,
    // then each of the two places gets one of them, and nothing in between can panic.
    unsafe {
        let swap = (*b).assume_init_ref() < (*a).assume_init_ref();
        let (x, y) = (ptr::read(a), ptr::read(b));
        let (lo, hi) = select_unpredictable(swap, (ptr::read(&y), ptr::read(&x)), (x, y));
        ptr::write(a, lo);
        ptr::write(b, hi);
        swap
    }
}

/// [`steps_branchless`] for [`CHUNK`] steps from `p`, on elements of at most
/// [`SPLIT_MAX_SIZE`] bytes, with the element of `p` held in `carry`; afterwards `carry` holds
/// that of `p + 2·CHUNK` (see [`Chunk`]).
///
/// The elements are moved into two arrays, one for even positions and one for odd ones, so
/// that every compare-exchange of a phase pairs equal lanes of the two (shifted by one for the
/// first phase), and the compiler runs the phase as a few vector minimums and maximums:
/// [`steps_branchless`] makes it pick lanes apart, and widens the comparison results of
/// narrow elements to 64 bits to build the mask. Every comparison and every swap is still the
/// one of the phase, on the same two elements.
///
/// # Safety
///
/// As for [`steps_branchless`], with `c` = [`CHUNK`]; the slot `p` is vacant, its element in
/// `carry`, and no other element is held out of the slice.
#[inline(always)]
unsafe fn chunk_steps<T: Ord>(p: *mut T, carry: &mut Carry<T>) -> u64 {
    // SAFETY: the chunk takes the elements of `p + 1 ..= p + 2·CHUNK` from the slice and that
    // of `p` from `carry`, by bit copies that leave their old places vacant; it compares and
    // exchanges them among its own places only, and on drop (also if a comparison panics)
    // puts each back, as described for `Chunk`.
    unsafe {
        let mut chunk = Chunk::<T> {
            even: [const { MaybeUninit::uninit() }; CHUNK],
            odd: [const { MaybeUninit::uninit() }; CHUNK],
            last: ptr::read(p.add(2 * CHUNK).cast()),
            p,
            carry,
        };
        ptr::copy_nonoverlapping(chunk.carry.val.as_ptr(), chunk.even[0].as_mut_ptr(), 1);
        for i in 1..CHUNK {
            chunk.even[i] = ptr::read(p.add(2 * i).cast());
        }
        for i in 0..CHUNK {
            chunk.odd[i] = ptr::read(p.add(2 * i + 1).cast());
        }
        let even = chunk.even.as_mut_ptr();
        let odd = chunk.odd.as_mut_ptr();
        // The first phase: the pairs `(p + 2i + 1, p + 2i + 2)`.
        for i in 0..CHUNK - 1 {
            cx_held(odd.add(i), even.add(i + 1));
        }
        cx_held(odd.add(CHUNK - 1), &raw mut chunk.last);
        // The second phase: the pairs `(p + 2i, p + 2i + 1)`, which swapped as bytes, turned
        // into bits below: a 64-bit mask built lane by lane is what made narrow elements slow.
        let mut swapped = [0u8; CHUNK];
        for (i, s) in swapped.iter_mut().enumerate() {
            *s = cx_held(even.add(i), odd.add(i)) as u8;
        }
        drop(chunk);
        let mut mask = 0;
        for (k, bytes) in swapped.as_chunks::<8>().0.iter().enumerate() {
            // Byte `j` of `x` is 0 or 1; the product gathers them into its top byte, bit `j`.
            let x = u64::from_le_bytes(*bytes);
            mask |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
        }
        mask
    }
}

/// Runs two consecutive phases: the first compares the `na` pairs starting at `a0`, `a0 + 2`,
/// ..., the second the `nb` pairs starting at `b0`, `b0 + 2`, .... Returns the pairs the second
/// phase swapped.
///
/// If both are non-empty, `b0` is `a0 - 1` or `a0 + 1`. A pair of the second phase depends only
/// on the pairs of the first on either side of it, so the two phases run as a wavefront: first
/// the pairs of the first phase that come before any of the second (`head`), then steps of one
/// pair of each, then whatever is left of the first and then of the second. A first-phase pair
/// outside the range is known not to swap, or lies outside the slice.
///
/// # Safety
///
/// Every element of every pair must be valid and not otherwise in use.
#[inline(always)]
unsafe fn round<T: Ord>(base: *mut T, a0: usize, na: usize, b0: usize, nb: usize) -> Swaps {
    // The first-phase pairs to compare before the first second-phase pair can be.
    let head = if nb == 0 {
        na
    } else if na == 0 || b0 + 1 == a0 {
        0
    } else {
        1
    };
    let mut swaps = Swaps::NONE;
    // SAFETY: every pair compared is one of the two phases, as required of the caller.
    unsafe {
        for i in 0..head {
            cx(base, a0 + 2 * i);
        }
        let steps = (na - head).min(nb);
        if size_of::<T>() <= BRANCHLESS_MAX_SIZE {
            let mut done = 0;
            if size_of::<T>() <= SPLIT_MAX_SIZE && steps >= CHUNK {
                // The element of the slot each chunk starts at comes from the chunk before.
                let mut carry = Carry {
                    val: MaybeUninit::new(ptr::read(base.add(b0))),
                    dst: base.add(b0),
                };
                while steps - done >= CHUNK {
                    let p = b0 + 2 * done;
                    let mask = chunk_steps(base.add(p), &mut carry);
                    if mask != 0 {
                        swaps.first = swaps.first.min(p + 2 * mask.trailing_zeros() as usize);
                        swaps.last = p + 2 * (63 - mask.leading_zeros() as usize);
                    }
                    done += CHUNK;
                }
                // Dropping `carry` puts the element of the next slot back.
            }
            while done < steps {
                let p = b0 + 2 * done;
                let c = if steps - done >= CHUNK {
                    CHUNK
                } else {
                    steps - done
                };
                let mask = if c == CHUNK {
                    steps_branchless(base.add(p), CHUNK)
                } else {
                    steps_branchless(base.add(p), c)
                };
                if mask != 0 {
                    swaps.first = swaps.first.min(p + 2 * mask.trailing_zeros() as usize);
                    swaps.last = p + 2 * (63 - mask.leading_zeros() as usize);
                }
                done += c;
            }
        } else {
            for i in 0..steps {
                cx(base, a0 + 2 * (head + i));
                if i >= LAG {
                    let k = b0 + 2 * (i - LAG);
                    if cx(base, k) {
                        swaps.push(k);
                    }
                }
            }
        }
        for i in head + steps..na {
            cx(base, a0 + 2 * i);
        }
        // The branchy steps leave their last `LAG` second-phase pairs to here.
        let from = if size_of::<T>() <= BRANCHLESS_MAX_SIZE {
            steps
        } else {
            steps.saturating_sub(LAG)
        };
        for i in from..nb {
            let k = b0 + 2 * i;
            if cx(base, k) {
                swaps.push(k);
            }
        }
    }
    swaps
}

/// The pairs from `lo` to `hi` (window-local, inclusive, clipped at 0) whose index in the slice,
/// `k + w0`, is odd if `odd` and even otherwise: the first one and how many. Phases are odd or
/// even in the slice, not in the window.
fn pairs(lo: isize, hi: isize, w0: usize, odd: bool) -> (usize, usize) {
    if hi < lo.max(0) {
        return (0, 0);
    }
    let mut first = lo.max(0) as usize;
    if ((first + w0) & 1 == 1) != odd {
        first += 1;
    }
    let hi = hi as usize;
    if first > hi {
        (0, 0)
    } else {
        (first, (hi - first) / 2 + 1)
    }
}

/// A raw pointer to the window being sorted, shared by the threads of a session. The blocks a
/// thread works on change from one super-step to the next and are not contiguous for the
/// thread that takes both edge blocks, so the threads cannot each be handed a `&mut` slice for
/// the whole session; they all reach the window through this one pointer instead, which the
/// session takes from its `&mut` slice and does not use otherwise until they are done.
struct Shared<T>(*mut T);

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}

impl<T> Shared<T> {
    /// The pointer; a method, so that a closure captures the whole `Shared` (which is `Send`)
    /// rather than its field.
    fn get(self) -> *mut T {
        self.0
    }
}

// SAFETY: the threads of a session access disjoint parts of the window between barriers, which
// order their accesses, so sharing the pointer only lets elements be used — compared and
// moved — by one thread at a time, on another thread than the caller's: what sending a `&mut`
// slice of them would allow, which `T: Send` does.
unsafe impl<T: Send> Send for Shared<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Shared<T> {}

/// A block of the window: elements `x0..x1`, whether each end is interior to the window
/// (so that the trapezoid shrinks there), and the boundary of the other partition inside it.
#[derive(Clone, Copy)]
struct Block {
    x0: usize,
    x1: usize,
    shrink_left: bool,
    shrink_right: bool,
    mid: Option<usize>,
}

/// How a window is split into blocks: the first partition at `b`, the second at the midpoints
/// `c` of the first's blocks, with its two half-blocks at the edges.
struct Plan {
    /// Window length.
    w: usize,
    /// Boundaries of the blocks of the first partition: `b[0] = 0`, ..., `b[nb] = w`.
    b: Vec<usize>,
    /// Midpoints of those blocks, the inner boundaries of the second partition.
    c: Vec<usize>,
}

impl Plan {
    /// Splits a window of `w` elements into `nb` blocks of the first partition.
    fn new(w: usize, nb: usize) -> Plan {
        let b: Vec<usize> = (0..=nb).map(|i| w * i / nb).collect();
        let c: Vec<usize> = (0..nb).map(|i| (b[i] + b[i + 1]) / 2).collect();
        Plan { w, b, c }
    }

    /// The largest number of phases a super-step can run: the shortest half-block. A triangle
    /// of `k` phases reaches `k` elements to either side of its boundary, which must stay inside
    /// the block that holds it.
    fn max_k(&self) -> usize {
        let nb = self.c.len();
        (0..nb)
            .map(|i| (self.c[i] - self.b[i]).min(self.b[i + 1] - self.c[i]))
            .min()
            .unwrap_or(0)
    }

    /// The number of blocks of a partition.
    fn blocks(&self, second: bool) -> usize {
        if second {
            self.c.len() + 1
        } else {
            self.c.len()
        }
    }

    /// Block `i` of a partition.
    fn block(&self, second: bool, i: usize) -> Block {
        let nb = self.c.len();
        if !second {
            Block {
                x0: self.b[i],
                x1: self.b[i + 1],
                shrink_left: i > 0,
                shrink_right: i + 1 < nb,
                mid: Some(self.c[i]),
            }
        } else if i == 0 {
            Block {
                x0: 0,
                x1: self.c[0],
                shrink_left: false,
                shrink_right: true,
                mid: None,
            }
        } else if i == nb {
            Block {
                x0: self.c[nb - 1],
                x1: self.w,
                shrink_left: true,
                shrink_right: false,
                mid: None,
            }
        } else {
            Block {
                x0: self.c[i - 1],
                x1: self.c[i],
                shrink_left: true,
                shrink_right: true,
                mid: Some(self.b[i]),
            }
        }
    }
}

/// The phases run on the trapezoid of `block`: `k` phases (`k` even, the first odd), each one
/// pair shorter at the interior ends than the one before, run as `k/2` rounds. Returns the swaps
/// of the last phase.
///
/// # Safety
///
/// The block's elements must be valid and in use by this thread only.
unsafe fn trapezoid<T: Ord>(base: *mut T, w0: usize, block: Block, k: usize) -> Swaps {
    let range = |j: usize| {
        let lo = block.x0 + if block.shrink_left { j } else { 0 };
        let hi = block.x1 as isize - 2 - if block.shrink_right { j as isize } else { 0 };
        (lo as isize, hi)
    };
    let mut swaps = Swaps::NONE;
    for j in (0..k).step_by(2) {
        let (lo, hi) = range(j);
        let (a0, na) = pairs(lo, hi, w0, true);
        let (lo, hi) = range(j + 1);
        let (b0, nb) = pairs(lo, hi, w0, false);
        // SAFETY: the pairs lie within the block.
        swaps = unsafe { round(base, a0, na, b0, nb) };
    }
    swaps
}

/// The phases left out by the trapezoids on either side of `mid`: `k` phases (`k` even, the
/// first odd), each one pair wider at both ends than the one before, run as `k/2` rounds.
/// Returns the swaps of the last phase.
///
/// # Safety
///
/// The `k + 1` elements on either side of `mid` must be valid and in use by this thread only.
unsafe fn triangle<T: Ord>(base: *mut T, w0: usize, mid: usize, k: usize) -> Swaps {
    let mid = mid as isize;
    let mut swaps = Swaps::NONE;
    for j in (0..k as isize).step_by(2) {
        let (a0, na) = pairs(mid - 1 - j, mid + j - 1, w0, true);
        let (b0, nb) = pairs(mid - 2 - j, mid + j, w0, false);
        // SAFETY: the pairs lie within `k` elements of `mid`, inside its block.
        swaps = unsafe { round(base, a0, na, b0, nb) };
    }
    swaps
}

/// What the threads of a session do next.
#[derive(Clone, Copy)]
enum Decision {
    /// Run super-step `s` (on the first partition if even, the second if odd): first the
    /// triangles of the previous super-step (`kp` phases), then `k` phases on the trapezoids.
    Run {
        s: usize,
        k: usize,
        kp: usize,
    },
    Stop,
}

/// How a session ended.
#[derive(Clone, Copy)]
enum Outcome {
    /// Unfinished; the slice is as it was.
    Aborted,
    /// The slice is sorted.
    Sorted,
    /// Rounds were run; these pairs swapped in the last (even) phase.
    Swapped(Swaps),
}

/// The bookkeeping of a session, updated by the last thread to reach each barrier.
struct Leader {
    started: bool,
    finishing: bool,
    /// The super-step being run, and its length.
    s: usize,
    k: usize,
    /// The super-step length limit.
    k_max: usize,
    rounds_left: usize,
    /// Swaps of the last phase of the current super-step, from the trapezoids.
    trap: Swaps,
    /// The pairs that may swap in the first phase of the session.
    init: (usize, usize),
    w: usize,
    /// Whether the window may be left on the left (right): the slice continues there.
    open_left: bool,
    open_right: bool,
    outcome: Outcome,
}

impl Leader {
    /// Decides what the threads do next, from the swaps they found in the last phase of the
    /// previous super-step (on the triangles: with those found on its trapezoids before, the
    /// phase is complete) and of the current one (on the trapezoids). The run goes on while the
    /// slice is unsorted, rounds are left, the unsorted part has not shrunk to half of what it
    /// was at the start, and it cannot reach the window's edges within the next super-step (the
    /// bounds widen by at most a pair per phase), unless the slice ends there.
    fn decide(&mut self, tri: Swaps, trap: Swaps) -> Decision {
        if !self.started {
            self.started = true;
            self.rounds_left -= self.k / 2;
            return Decision::Run {
                s: 0,
                k: self.k,
                kp: 0,
            };
        }
        if self.finishing {
            self.outcome = Outcome::Swapped(self.trap.merge(tri));
            return Decision::Stop;
        }
        // Super-step `s` has run its trapezoids and the triangles of super-step `s - 1`.
        let s = self.s;
        // The pairs that may swap from the first phase after the known one, and how many
        // phases ago that was.
        let (lo, hi) = if s == 0 {
            self.init
        } else {
            let known = self.trap.merge(tri);
            if known.is_none() {
                self.outcome = Outcome::Sorted;
                return Decision::Stop;
            }
            (known.first.saturating_sub(1), known.last + 1)
        };
        self.trap = trap;
        let next_k = self.k_max.min(2 * self.k).min(2 * self.rounds_left);
        let reach = self.k + next_k;
        let fits =
            (!self.open_left || lo >= reach) && (!self.open_right || hi + reach + 2 <= self.w);
        let shrunk = s > 0 && (hi - lo + 2) * 2 < self.init.1 - self.init.0 + 2;
        if next_k == 0 || !fits || shrunk {
            self.finishing = true;
            return Decision::Run {
                s: s + 1,
                k: 0,
                kp: self.k,
            };
        }
        self.rounds_left -= next_k / 2;
        let kp = self.k;
        self.s = s + 1;
        self.k = next_k;
        Decision::Run {
            s: s + 1,
            k: next_k,
            kp,
        }
    }
}

/// What the threads of a session share at a barrier.
struct BarrierState {
    arrived: usize,
    generation: usize,
    poisoned: bool,
    tri: Swaps,
    trap: Swaps,
    decision: Decision,
    leader: Leader,
}

/// A barrier at which the threads of a session combine what they found, and the last to
/// arrive decides what all do next.
struct Barrier {
    state: Mutex<BarrierState>,
    cv: Condvar,
    parties: usize,
    /// The next block of the current super-step to be claimed.
    next_block: AtomicUsize,
}

impl Barrier {
    /// Waits for all threads, adding this one's swaps. Returns the decision of the last one to
    /// arrive, or `None` if a thread has panicked (or could not be started).
    fn wait(&self, tri: Swaps, trap: Swaps) -> Option<Decision> {
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if st.poisoned {
            return None;
        }
        st.tri = st.tri.merge(tri);
        st.trap = st.trap.merge(trap);
        st.arrived += 1;
        if st.arrived == self.parties {
            let (tri, trap) = (st.tri, st.trap);
            st.tri = Swaps::NONE;
            st.trap = Swaps::NONE;
            st.arrived = 0;
            st.decision = st.leader.decide(tri, trap);
            st.generation += 1;
            self.next_block.store(0, Ordering::Relaxed);
            self.cv.notify_all();
            Some(st.decision)
        } else {
            let generation = st.generation;
            while st.generation == generation && !st.poisoned {
                st = self.cv.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
            if st.poisoned { None } else { Some(st.decision) }
        }
    }

    /// Releases every waiting thread with `None`, now and at every later wait.
    fn poison(&self) {
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        st.poisoned = true;
        self.cv.notify_all();
    }
}

/// Poisons the barrier if its thread unwinds, so that the others stop instead of waiting.
struct PoisonOnUnwind<'a>(&'a Barrier);

impl Drop for PoisonOnUnwind<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.poison();
        }
    }
}

/// What each thread of a session runs: between barriers, it claims blocks of the current
/// partition and runs on each the triangle of the previous super-step and then the trapezoid of
/// the current one.
fn worker<T: Ord>(base: Shared<T>, w0: usize, plan: &Plan, barrier: &Barrier) {
    let _guard = PoisonOnUnwind(barrier);
    let base = base.get();
    let (mut tri, mut trap) = (Swaps::NONE, Swaps::NONE);
    while let Some(Decision::Run { s, k, kp }) = barrier.wait(tri, trap) {
        tri = Swaps::NONE;
        trap = Swaps::NONE;
        let second = s % 2 == 1;
        loop {
            let i = barrier.next_block.fetch_add(1, Ordering::Relaxed);
            if i >= plan.blocks(second) {
                break;
            }
            let block = plan.block(second, i);
            // SAFETY: blocks of one partition are disjoint and lie within the window, each is
            // claimed by one thread, and the barrier orders the accesses of one super-step after
            // those of the one before. A triangle reaches at most `kp ≤ k_max` elements from its
            // boundary, which stays inside the block (see `Plan::max_k`).
            unsafe {
                if let (Some(mid), true) = (block.mid, kp > 0) {
                    tri = tri.merge(triangle(base, w0, mid, kp));
                }
                if k > 0 {
                    trap = trap.merge(trapezoid(base, w0, block, k));
                }
            }
        }
    }
}

/// Runs rounds on the window `arr[w0..w1]` with up to `threads` threads, from the state in
/// `rounds`, and updates it. Returns `false` if it ran none: the window is too short for two
/// blocks, or a thread could not be started.
///
/// Pairs outside the window are not compared, which is right as long as the bounds of every
/// phase run stay inside it: the window has a margin of half the unsorted part on each side, a
/// super-step is at most half the margin long at first, and the leader ends the session before
/// the bounds could cross the margin.
fn session<T: Ord + Send>(
    arr: &mut [T],
    w0: usize,
    w1: usize,
    rounds: &mut Rounds,
    threads: usize,
) -> bool {
    let n = arr.len();
    let Some((lo, m)) = rounds.next else {
        return false;
    };
    let w = w1 - w0;
    let nb = (threads * BLOCKS_PER_THREAD).min(w / min_block(size_of::<T>()));
    if nb < 2 {
        return false;
    }
    let plan = Plan::new(w, nb);
    // The pairs that may swap in the first phase, window-local.
    let init = (lo - 1 - w0, lo + 2 * m - 1 - w0);
    let mut k_max = plan.max_k();
    if w0 > 0 {
        k_max = k_max.min(init.0 / 2);
    }
    if w1 < n {
        k_max = k_max.min((w - init.1 - 2) / 2);
    }
    let k = k_max.min(FIRST_K).min(2 * rounds.left) & !1;
    if k == 0 {
        return false;
    }
    let p = threads.min(nb);
    let barrier = Barrier {
        state: Mutex::new(BarrierState {
            arrived: 0,
            generation: 0,
            poisoned: false,
            tri: Swaps::NONE,
            trap: Swaps::NONE,
            decision: Decision::Stop,
            leader: Leader {
                started: false,
                finishing: false,
                s: 0,
                k,
                k_max: k_max & !1,
                rounds_left: rounds.left,
                trap: Swaps::NONE,
                init,
                w,
                open_left: w0 > 0,
                open_right: w1 < n,
                outcome: Outcome::Aborted,
            },
        }),
        cv: Condvar::new(),
        parties: p,
        next_block: AtomicUsize::new(0),
    };
    // From here until the threads are done, the window is reached only through `base`.
    let base = Shared(arr[w0..w1].as_mut_ptr());
    let (barrier, plan) = (&barrier, &plan);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(p - 1);
        for _ in 1..p {
            let spawned =
                thread::Builder::new().spawn_scoped(scope, move || worker(base, w0, plan, barrier));
            match spawned {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    barrier.poison();
                    break;
                }
            }
        }
        worker(base, w0, plan, barrier);
        let mut panic = None;
        for handle in handles {
            if let Err(payload) = handle.join() {
                panic.get_or_insert(payload);
            }
        }
        if let Some(payload) = panic {
            resume_unwind(payload);
        }
    });
    let st = barrier.state.lock().unwrap_or_else(PoisonError::into_inner);
    rounds.left = st.leader.rounds_left;
    match st.leader.outcome {
        Outcome::Aborted => false,
        Outcome::Sorted => {
            rounds.next = None;
            true
        }
        Outcome::Swapped(s) => {
            let shifted = if s.is_none() {
                s
            } else {
                Swaps {
                    first: s.first + w0,
                    last: s.last + w0,
                }
            };
            rounds.after(shifted);
            true
        }
    }
}
