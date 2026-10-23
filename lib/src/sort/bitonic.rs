use core::hint::select_unpredictable;
use core::mem::{self, MaybeUninit};
use core::{ptr, slice};
use std::any::Any;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{self, AtomicBool, AtomicUsize};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;

/// Sorts a slice in place using bitonic sort.
///
/// Bitonic sort, published by Kenneth Batcher in 1968, is a sorting network: a fixed sequence
/// of compare-exchanges, each of which puts the elements at two given positions in order,
/// chosen in advance and never depending on what the comparisons answer. A sequence that
/// first rises and then falls is called bitonic, and one stage of compare-exchanges, of each
/// element of the first half with the element half a slice further on, splits a bitonic
/// sequence into two bitonic halves with nothing in the first greater than anything in the
/// second; repeating this on the halves merges it. Bitonic sort sorts the two halves of the
/// slice in opposite directions, which makes the whole slice bitonic, and merges it. That
/// takes `n·k·(k + 1) / 4` compare-exchanges for `n = 2^k`, `O(n log² n)`, in `k·(k + 1) / 2`
/// stages whose compare-exchanges are independent of each other, which is why the network
/// suits hardware, vector units and GPUs, which can run a whole stage at once, and threads
/// ([`bitonic_parallel`]).
///
/// Batcher's network is for powers of two. For other lengths this is H. W. Lang's
/// formulation: the first `⌊n/2⌋` elements are sorted descending and the rest ascending, and
/// the merge of `n` elements compares each element `i < n − m` with element `i + m`, for `m`
/// the largest power of two below `n`, then merges the first `m` elements and the rest. It is
/// the network for the next power of two without the compare-exchanges that would involve
/// the missing elements, as if they were padding greater than every element, which never
/// moves; nothing is actually padded, so any `T` can be sorted.
///
/// The network is fixed, so the five refinements change only how it is carried out: the
/// sort makes exactly the compare-exchanges of the textbook recursion, each once, in an
/// order that respects which ones depend on which.
///
/// * **Small networks in straight-line code.** Sorts of at most 16 elements and merges of
///   at most 16 are not recursed into: each length has its own copy of its network, with
///   every compare-exchange written out at constant positions, so small elements stay in
///   registers. Against recursing down to single elements, this is 2.2–4.2 times faster for
///   integers, 1.6–2 times for strings and 1.35–1.6 times for 64-byte elements; against a
///   loop over the same networks, 1.9–2.4, 1.1–1.15 and 1.35–1.55 times.
/// * **Two or three stages per pass.** A merge of a power-of-two length runs its first two
///   or three stages as one pass over groups of 4 or 8 elements a quarter or an eighth of the
///   merge apart, each read once, put through the 4 or 12 compare-exchanges of those stages
///   and written once, then merges the parts depth first, so the later stages run in cache.
///   Three stages for elements of at most 8 bytes make integers 1.04–1.14 times faster than
///   one, two for elements of more than 24 bytes make 64-byte elements 1.03–1.08 times
///   faster; strings, whose comparisons cost more than the passes, take one.
/// * **Compare-exchanges without branches.** Elements of at most 8 bytes are exchanged by
///   selecting the values to write back, which lets the compiler run a pass's independent
///   compare-exchanges in vector registers: 4.6–5 times faster than branching on each
///   comparison. Within a merge, whether a compare-exchange swaps comes in long runs, as it
///   acts on bitonic sequences, but in the small networks it is a coin toss on random input,
///   so elements of 9 to 24 bytes, such as `String`, are exchanged there by selecting the
///   addresses to copy from, and by a branch in the passes: 1.25–1.45 times faster for random
///   strings, 1.05–1.1 times for nearly sorted ones. The compiler turns the selection back
///   into a branch for a comparison that is itself a chain of branches, such as that of
///   arrays and tuples, and these lose up to 1.4 times; types such as `u128` and `(u64, u64)`
///   are 1.4–1.9 times faster on random input, but up to 1.8 times slower on nearly sorted
///   input, whose branches are predictable.
/// * **The direction is a type.** Each part is sorted ascending or descending through a type
///   parameter rather than a flag, so every compare-exchange is compiled for its direction.
/// * **Narrow elements are merged stage by stage.** For elements of at most 4 bytes, a merge
///   of a power-of-two length of at most 2048 runs its stages one after the other, each in one
///   loop over the whole merge, which stays in the first-level cache: the compiler runs every
///   stage, down to the one pairing neighbours, as vector minimums and maximums of 4 to 16
///   elements at once, where the straight-line merges of 16 compare one pair at a time. This
///   is 1.25 times faster for `u32`, `u16` and `u8` at 1000 elements, 1.45–1.6 times at
///   16,384 and 1.3–1.55 times at a million, on every input; for `u64`, of which a vector holds
///   two, it was 1.1–1.25 times slower.
///
/// Together, against the textbook recursion (Lang's, with the direction as a flag and a
/// branch and a swap for each compare-exchange): 10.5–11.8 times faster on random integers
/// and 4.8–9.8 times on other input, 1.75–2.1 times on random strings and 1.3–1.6 times on
/// other strings, and 1.4–1.8 times for random 64-byte elements, 1.05–1.5 times on other
/// input.
///
/// # Properties
///
/// * **Unstable**: equal elements may be reordered.
/// * **In place**: no allocation, `O(log n)` stack.
/// * **Not adaptive**: the same compare-exchanges on every input of a length, sorted or not.
///
/// | `n`                  | Compare-exchanges, on every input       | Stages            |
/// |----------------------|-----------------------------------------|-------------------|
/// | `2^k`                | `n·k·(k + 1) / 4`                       | `k·(k + 1) / 2`   |
/// | `2^(k−1) < n < 2^k`  | between `n·(k − 1)² / 4` and `n·k·(k + 1) / 4` | `k·(k + 1) / 2` |
///
/// For `n = 1000`: 26,984 comparisons on every input, against about 10,500 for heap sort
/// (14,600 for elements of at most 8 bytes); for `n = 1,000,000`: 102,931,040, about
/// `5.2·n·log₂ n`.
///
/// With integers, whose compare-exchanges run in vector registers, this sort is as fast as
/// [`slice::sort_unstable`] around 1000 elements, and 1.25 times faster than this crate's
/// [`quick`](fn@super::quick) sort; from there on its `log₂ n` factor shows: `sort_unstable`
/// is 1.35 times faster at 16,384 elements and 1.85 times at a million, and 20–70 times
/// faster on sorted input, which it recognises. Random `u32`s and `u16`s, merged stage by
/// stage, are sorted 1.1–1.3 times faster than by `sort_unstable` from 1000 to 16,384
/// elements, and 1.25–1.45 times slower at a million. Elements with costlier comparisons pay for
/// every one of them: `sort_unstable` is 2.5–5 times faster on random strings and 2.3–4.1
/// times for random 64-byte elements. Against this crate's [`shell`](fn@super::shell) sort:
/// 2.25–3.6 times faster on random integers from 1000 elements to a million, as fast on
/// random strings, 1.35–1.6 times slower for 64-byte elements.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: nothing is dropped twice or leaked. An [`Ord`] that is not a total
/// order leaves `arr` in an unspecified order with every element exactly once as well, after
/// the same comparisons as a total order.
///
/// # Examples
///
/// ```
/// use a::sort::bitonic;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// bitonic(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Any length, not only powers of two.
/// let mut v: Vec<u32> = (0..1000).map(|i| i * 7919 % 1000).collect();
/// bitonic(&mut v);
/// assert!(v.iter().copied().eq(0..1000));
///
/// let mut words = vec!["pear", "apple", "fig"];
/// bitonic(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn bitonic<T: Ord>(arr: &mut [T]) {
    sort::<T, Ascending>(arr);
}

/// Sorts a slice in place using bitonic sort on several threads.
///
/// Makes exactly the compare-exchanges of [`bitonic`], so it leaves the slice in the same
/// order, equal elements included, but spreads them over threads: one per 16,384 elements
/// of at most 8 bytes, or per 4,096 larger elements, up to as many as
/// [`std::thread::available_parallelism`] reports. A slice too short for two threads is
/// sorted by [`bitonic`] on the calling thread, with no allocation.
///
/// The slice is cut along the sort's own halving into parts, two per thread for elements of
/// at most 8 bytes and four for larger ones, and a first round sorts each part. Then each
/// level of merges, from the smallest, takes two rounds. The parts of a merge become its
/// blocks: the merge's first stages compare elements at the same position of different
/// blocks, so the first round runs them for one range of positions at a time, in all blocks
/// at once, and the second merges each block on its own. Threads take the tasks of a round,
/// which touch disjoint elements, one at a time until none is left, so threads that run
/// faster take more, and meet at a barrier before the next round.
///
/// The threads are spawned once per call and kept for all rounds: spawning them for each
/// round instead made 32,768 to 131,072 elements sort 1.15–1.6 times slower, and a million
/// integers 1.2 times. Against [`bitonic`] on
/// the 16 threads of an Apple M4 Max (12 performance and 4 efficiency cores): integers sort
/// 1.6 times faster at 32,768 elements, 2.5 times at 65,536, 2.9–3.2 times up to 262,144, 7.2
/// times at a million and 8.8 times at four million; strings 1.8 times faster at 8,192
/// elements, 3.3 times at 16,384, 4.4–5.7 times at 32,768 and 10–11 times from 262,144 on;
/// 64-byte elements 1.7 times faster at 8,192 elements, 2.9 times at 16,384 and 8–9.7 times
/// from 262,144 on. Between 262,144 and a million integers, 8 to 12 threads would be up to
/// 1.25 times faster than 16: each round waits for its slowest thread, and on the efficiency
/// cores a task takes longer. Against the single-threaded [`slice::sort_unstable`], this is
/// faster from 32,768 elements on: for integers 1.2 times there and 3.9 times at a million,
/// for strings 1.15 and 2.2 times, for 64-byte elements 1.05 and 2.4 times. Elements of at
/// most 4 bytes, merged stage by stage as in [`bitonic`], sort 1.05–1.25 (`u32`) and
/// 1.15–1.35 (`u8`) times faster than without that, from 65,536 to a million elements.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics on any thread, the
/// threads finish the tasks they are on and skip the rest, and the first panic is propagated
/// to the caller; `arr` still holds every original element exactly once, as with
/// [`bitonic`].
///
/// # Examples
///
/// ```
/// use a::sort::bitonic_parallel;
///
/// let mut v: Vec<u64> = (0..100_000).map(|i| i * 7919 % 100_003).collect();
/// bitonic_parallel(&mut v);
/// assert!(v.windows(2).all(|w| w[0] <= w[1]));
///
/// let mut words: Vec<String> = (0..10_000).map(|i| format!("{}", i * 7 % 10_000)).collect();
/// bitonic_parallel(&mut words);
/// assert!(words.windows(2).all(|w| w[0] <= w[1]));
/// ```
pub fn bitonic_parallel<T: Ord + Send>(arr: &mut [T]) {
    // Asking for the parallelism can allocate, so shorter slices do not.
    let most = arr.len() / per_thread::<T>();
    let threads = if most < 2 { 1 } else { most.min(threads()) };
    if threads < 2 {
        bitonic(arr);
        return;
    }
    let rounds = schedule(
        arr.len(),
        threads,
        parts_per_thread::<T>(),
        per_thread::<T>(),
    );
    run(arr, &rounds, threads);
}

/// The parallel sort cuts the slice into about this many parts per thread, so that threads
/// that run faster, on a faster core or a less busy one, take over more of them, at the cost
/// of two more rounds for each doubling.
///
/// Measured on 16 threads, against two parts per thread: four make strings and 64-byte
/// elements sort 1.05–1.15 times faster from 262,144 elements on, but integers 1.05–1.2 times
/// slower from 131,072 to a million, whose tasks are short enough for the rounds to cost more
/// than the imbalance.
const fn parts_per_thread<T>() -> usize {
    if mem::size_of::<T>() <= VALUES_MAX {
        2
    } else {
        4
    }
}

/// [`bitonic_parallel`] uses no more threads than one per this many elements: each round
/// costs a barrier, which a thread must have enough work to pay for. Its parts are no
/// shorter either, for the same reason.
///
/// Measured: with 16 threads, 32,768 integers sort 1.6 times slower than on one thread, but
/// 1.5 times faster on two to four; 131,072 integers sort fastest on eight. Strings and
/// 64-byte elements, whose comparisons and moves cost more, gain 1.7–1.9 times on two
/// threads from 8,192 elements and sort fastest on eight at 32,768. As the shortest part,
/// 16,384 integers instead of 4,096 sort 1.04–1.15 times faster from 32,768 to 262,144
/// elements; for larger elements, 16,384 is up to 3 times slower and 1,024 up to 1.25 times.
const fn per_thread<T>() -> usize {
    if mem::size_of::<T>() <= VALUES_MAX {
        1 << 14
    } else {
        1 << 12
    }
}

/// The number of threads [`bitonic_parallel`] uses.
fn threads() -> usize {
    thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

/// A part of the work of [`bitonic_parallel`], on positions `start..start + len` of the
/// slice, into ascending order if `ascending`.
#[derive(Clone, Copy)]
enum Task {
    /// Sort the positions.
    Sort {
        start: usize,
        len: usize,
        ascending: bool,
    },
    /// Run the first `log₂ blocks` stages of the merge of the positions, which compare
    /// elements of different blocks, on the `range`-th range of positions within each block.
    Across {
        start: usize,
        len: usize,
        ascending: bool,
        blocks: usize,
        range: usize,
    },
    /// Merge the positions, the rest of a merge whose stages across blocks are done.
    Merge {
        start: usize,
        len: usize,
        ascending: bool,
    },
}

/// The rounds of tasks that sort `n` elements on `threads` threads: the tasks of a round
/// touch disjoint positions and can run in any order, and a round starts only once the one
/// before it is done.
///
/// The slice is cut into `2^depth` parts, about `parts_per_thread` per thread but none
/// shorter than `min_part`, along the sort's own halving, and the first round sorts them. Then each level of merges, from the
/// smallest, takes two rounds. A merge of the level has `blocks` sorted parts; its first
/// `log₂ blocks` stages compare elements at the same position of different blocks, so the
/// first round runs them for one range of positions per task, in all blocks at once, and the
/// second merges each block on its own.
fn schedule(n: usize, threads: usize, parts_per_thread: usize, min_part: usize) -> Vec<Vec<Task>> {
    let mut depth = (threads * parts_per_thread).next_power_of_two().ilog2();
    while depth > 0 && n >> depth < min_part {
        depth -= 1;
    }
    let mut rounds = Vec::new();
    let mut parts = Vec::new();
    split(0, n, true, depth, &mut parts);
    rounds.push(
        parts
            .iter()
            .map(|&(start, len, ascending)| Task::Sort {
                start,
                len,
                ascending,
            })
            .collect(),
    );
    for level in (0..depth).rev() {
        // A merge of `len` elements is the merge of the next power of two, `P`, without the
        // compare-exchanges beyond `len`, so its blocks have `P / blocks` positions, the
        // last ones fewer or none.
        let blocks = 1 << (depth - level);
        let mut merges = Vec::new();
        split(0, n, true, level, &mut merges);
        let mut across = Vec::new();
        let mut within = Vec::new();
        for &(start, len, ascending) in &merges {
            let block = len.next_power_of_two() / blocks;
            let ranges = block.div_ceil(block.div_ceil(blocks));
            across.extend((0..ranges).map(|range| Task::Across {
                start,
                len,
                ascending,
                blocks,
                range,
            }));
            within.extend((0..len.div_ceil(block)).map(|b| Task::Merge {
                start: start + b * block,
                len: block.min(len - b * block),
                ascending,
            }));
        }
        rounds.push(across);
        rounds.push(within);
    }
    rounds
}

/// Appends to `out` the `2^depth` parts of positions `start..start + len` that [`sort`]
/// sorts on its own at recursion depth `depth` when sorting them ascending if `ascending`,
/// as `(start, len, ascending)`.
fn split(
    start: usize,
    len: usize,
    ascending: bool,
    depth: u32,
    out: &mut Vec<(usize, usize, bool)>,
) {
    if depth == 0 {
        out.push((start, len, ascending));
        return;
    }
    let half = len / 2;
    split(start, half, !ascending, depth - 1, out);
    split(start + half, len - half, ascending, depth - 1, out);
}

/// Carries out `task` on the slice starting at `base`.
///
/// # Safety
///
/// `base` must be valid for reads and writes of every position the task touches, and no
/// other thread may access those positions while it runs.
unsafe fn perform<T: Ord>(base: *mut T, task: Task) {
    // SAFETY: the positions of the task are valid and this thread's alone, as the caller
    // guarantees; the slices made from them below are disjoint.
    unsafe {
        let part = |start: usize, len: usize| slice::from_raw_parts_mut(base.add(start), len);
        match task {
            Task::Sort {
                start,
                len,
                ascending,
            } => {
                if ascending {
                    sort::<T, Ascending>(part(start, len));
                } else {
                    sort::<T, Descending>(part(start, len));
                }
            }
            Task::Merge {
                start,
                len,
                ascending,
            } => {
                if ascending {
                    merge::<T, Ascending>(part(start, len));
                } else {
                    merge::<T, Descending>(part(start, len));
                }
            }
            Task::Across {
                start,
                len,
                ascending,
                blocks,
                range,
            } => {
                let block = len.next_power_of_two() / blocks;
                let width = block.div_ceil(blocks);
                let mut pieces: Vec<&mut [T]> = (0..blocks)
                    .map(|b| {
                        let from = (b * block + range * width).min(len);
                        let to = (b * block + ((range + 1) * width).min(block)).min(len);
                        part(start + from, to - from)
                    })
                    .collect();
                if ascending {
                    merge_across::<T, Ascending>(&mut pieces);
                } else {
                    merge_across::<T, Descending>(&mut pieces);
                }
            }
        }
    }
}

/// Runs the first `log₂ pieces.len()` stages of a merge on `pieces`, one range of positions
/// from each of its blocks: the stage that compares blocks `s` apart compares each piece
/// with the one `s` further on, position by position. Later blocks may be cut short at the
/// end of the slice; a piece is never longer than the one before it, so pairing positions
/// up to the end of the later piece makes exactly the compare-exchanges of the merge.
fn merge_across<T: Ord, D: Direction>(pieces: &mut [&mut [T]]) {
    let blocks = pieces.len();
    let mut s = blocks / 2;
    while s > 0 {
        for a in 0..blocks {
            if a & s == 0 {
                let (x, y) = pieces.split_at_mut(a + s);
                clean2::<T, D>(x[a], y[0]);
            }
        }
        s /= 2;
    }
}

/// The start of the slice being sorted, shared by the threads of [`bitonic_parallel`].
struct Shared<T>(*mut T);

impl<T> Shared<T> {
    /// The pointer; a method, so that closures capture the whole `Shared` rather than the
    /// bare pointer field.
    fn get(&self) -> *mut T {
        self.0
    }
}

// SAFETY: the threads only ever touch disjoint positions at the same time (see `run`), so
// sharing the pointer moves elements between threads, which `T: Send` allows, but never
// lets two threads reach the same element at once, which would need `T: Sync`.
unsafe impl<T: Send> Sync for Shared<T> {}

/// Carries out the `rounds` on `arr`, on `threads` threads, the calling one included.
///
/// The threads are spawned once. Each takes the tasks of a round one at a time, from a
/// counter, until none is left, then waits at a barrier for the others to finish the round.
/// If a task panics, the panic is caught, the remaining tasks are skipped but the threads
/// still meet at every barrier, and the first panic is propagated once all have finished.
fn run<T: Ord + Send>(arr: &mut [T], rounds: &[Vec<Task>], threads: usize) {
    let base = Shared(arr.as_mut_ptr());
    let next: Vec<AtomicUsize> = rounds.iter().map(|_| AtomicUsize::new(0)).collect();
    let barrier = Barrier::new();
    let failed = AtomicBool::new(false);
    let panic: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);
    let work = || {
        for (round, next) in rounds.iter().zip(&next) {
            loop {
                let i = next.fetch_add(1, atomic::Ordering::Relaxed);
                let Some(&task) = round.get(i) else {
                    break;
                };
                if failed.load(atomic::Ordering::Relaxed) {
                    continue;
                }
                // SAFETY: the tasks of a round touch disjoint positions of `arr`, each task
                // is taken by exactly one thread, and the barrier makes every write of a
                // round visible before the next one starts.
                let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
                    perform(base.get(), task);
                }));
                if let Err(payload) = result {
                    failed.store(true, atomic::Ordering::Relaxed);
                    let mut first = panic.lock().unwrap_or_else(PoisonError::into_inner);
                    first.get_or_insert(payload);
                }
            }
            barrier.wait();
        }
    };
    thread::scope(|s| {
        let mut spawned = 0;
        for _ in 1..threads {
            if thread::Builder::new().spawn_scoped(s, work).is_err() {
                break;
            }
            spawned += 1;
        }
        barrier.set_parties(spawned + 1);
        work();
    });
    if let Some(payload) = panic.into_inner().unwrap_or_else(PoisonError::into_inner) {
        panic::resume_unwind(payload);
    }
}

/// A barrier whose number of parties is set once they are known: the threads of [`run`]
/// may reach it before all of them have been spawned, and a spawn may fail.
struct Barrier {
    state: Mutex<BarrierState>,
    released: Condvar,
}

struct BarrierState {
    /// How many threads must arrive to release them; until set, more than can ever arrive.
    parties: usize,
    arrived: usize,
    /// Counts the releases, so that a thread waits for its own.
    generation: usize,
}

impl Barrier {
    fn new() -> Self {
        Barrier {
            state: Mutex::new(BarrierState {
                parties: usize::MAX,
                arrived: 0,
                generation: 0,
            }),
            released: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, BarrierState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Sets the number of parties; no more than that many may have arrived yet.
    fn set_parties(&self, parties: usize) {
        self.lock().parties = parties;
    }

    /// Waits until all parties have arrived, then releases them together.
    fn wait(&self) {
        let mut state = self.lock();
        state.arrived += 1;
        if state.arrived == state.parties {
            state.arrived = 0;
            state.generation += 1;
            self.released.notify_all();
            return;
        }
        let generation = state.generation;
        while state.generation == generation {
            state = self
                .released
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// The direction a part is sorted in, as a type, so that each compare-exchange is compiled
/// for its direction.
trait Direction {
    /// The opposite direction.
    type Reverse: Direction;
    const ASCENDING: bool;
}

struct Ascending;
struct Descending;

impl Direction for Ascending {
    type Reverse = Descending;
    const ASCENDING: bool = true;
}

impl Direction for Descending {
    type Reverse = Ascending;
    const ASCENDING: bool = false;
}

/// Sorts of at most this many elements run a straight-line network ([`network`]).
///
/// Measured against 8: 1.02–1.07 times faster for integers up to 16,384 elements, no change
/// beyond or for larger elements.
const SORT_LEAF: usize = 16;

/// Merges of at most this many elements run a straight-line network ([`network`]).
///
/// Measured against 8: 1.2–1.3 times faster for integers, 1.03–1.09 times for larger
/// elements. 32 would make integers and strings 1.02–1.035 times faster still, and 64-byte
/// elements up to 1.03 times slower, for twice the code.
const MERGE_LEAF: usize = 16;

/// Elements of at most this many bytes are exchanged by selecting values ([`VALUES`]).
const VALUES_MAX: usize = 8;

/// Elements of at most this many bytes, and more than [`VALUES_MAX`], are exchanged by
/// selecting addresses ([`POINTERS`]) in the straight-line networks: `String`, `Vec` and
/// pairs of words.
///
/// Measured: 1.25–1.45 times faster for random strings, 1.05–1.1 times for nearly sorted ones
/// (see [`bitonic`] for the types it does not suit). Arrays of four to eight `u64`s, whose
/// comparison is a chain of branches, sort 1.15–1.35 times faster by branching.
const POINTERS_MAX: usize = 24;

/// Exchanges by selecting the values to write back: no branch, and the compiler can run
/// independent compare-exchanges in vector registers.
const VALUES: u8 = 0;
/// Exchanges by selecting the addresses to copy from: no branch, at the cost of copying both
/// elements whether or not they swap.
const POINTERS: u8 = 1;
/// Exchanges by branching on the comparison: only swapped elements are moved.
const BRANCH: u8 = 2;

/// How the straight-line networks exchange elements of type `T`.
const fn network_exchange<T>() -> u8 {
    if mem::size_of::<T>() <= VALUES_MAX {
        VALUES
    } else if mem::size_of::<T>() <= POINTERS_MAX {
        POINTERS
    } else {
        BRANCH
    }
}

/// How the passes of the merges exchange elements of type `T`.
const fn pass_exchange<T>() -> u8 {
    if mem::size_of::<T>() <= VALUES_MAX {
        VALUES
    } else {
        BRANCH
    }
}

/// How many stages a pass of a merge runs at once, as the number of parts it compares: 2, 4
/// or 8 for one, two or three stages.
///
/// Measured against one stage per pass: three make integers 1.04–1.14 times faster (two are
/// within 2% of three), two make 64-byte elements 1.03–1.08 times faster, and three 1.1–1.2
/// times slower than two. Strings sort 1.03–1.08 times faster with one: a pass that branches
/// on several comparisons at once mispredicts more.
const fn radix<T>() -> usize {
    if mem::size_of::<T>() <= VALUES_MAX {
        8
    } else if mem::size_of::<T>() <= POINTERS_MAX {
        2
    } else {
        4
    }
}

/// Sorts `v` in direction `D`: the first half in the opposite direction, the rest in `D`,
/// which makes `v` bitonic, then merges it.
fn sort<T: Ord, D: Direction>(v: &mut [T]) {
    let n = v.len();
    if n <= SORT_LEAF {
        leaf::<T, D, false>(v);
        return;
    }
    let (a, b) = v.split_at_mut(n / 2);
    sort::<T, D::Reverse>(a);
    sort::<T, D>(b);
    merge::<T, D>(v);
}

/// Merges `v`, whose first `⌊n/2⌋` elements are sorted against direction `D` and the rest
/// in it, or which is any other sequence such a merge leaves to its parts, into direction `D`.
///
/// Compares each element `i < n − m` with element `i + m`, for `m` the largest power of two
/// below `n`, then merges `v[..m]` and `v[m..]`.
fn merge<T: Ord, D: Direction>(v: &mut [T]) {
    let n = v.len();
    if n <= MERGE_LEAF {
        leaf::<T, D, true>(v);
        return;
    }
    let m = 1 << (n - 1).ilog2();
    if n == 2 * m {
        merge_pow2::<T, D>(v);
        return;
    }
    let (lo, hi) = v.split_at_mut(m);
    clean2::<T, D>(&mut lo[..n - m], hi);
    merge_pow2::<T, D>(lo);
    merge::<T, D>(hi);
}

/// [`merge`] for a power-of-two length: runs its first stages, as many as [`radix`] says,
/// in one pass, then merges each part depth first.
fn merge_pow2<T: Ord, D: Direction>(v: &mut [T]) {
    let n = v.len();
    if mem::size_of::<T>() <= STAGED_MAX_SIZE && n <= STAGED_LEN {
        return staged::<T, D>(v);
    }
    if n <= MERGE_LEAF {
        leaf::<T, D, true>(v);
        return;
    }
    // The parts are merged in a loop of its own in each branch, where the compiler knows it
    // runs 8, 4 or 2 times: one loop after the branches, over parts of a size it does not
    // know, made the whole sort up to 1.03 times slower.
    let radix = radix::<T>();
    if radix >= 8 && n >= 8 * MERGE_LEAF {
        let e = n / 8;
        let (v0, v1) = v.split_at_mut(4 * e);
        let (v00, v01) = v0.split_at_mut(2 * e);
        let (v10, v11) = v1.split_at_mut(2 * e);
        let (a, b) = v00.split_at_mut(e);
        let (c, d) = v01.split_at_mut(e);
        let (e_, f) = v10.split_at_mut(e);
        let (g, h) = v11.split_at_mut(e);
        clean8::<T, D>([a, b, c, d, e_, f, g, h]);
        for part in v.chunks_exact_mut(e) {
            merge_pow2::<T, D>(part);
        }
    } else if radix >= 4 && n >= 4 * MERGE_LEAF {
        let q = n / 4;
        let (ab, cd) = v.split_at_mut(2 * q);
        let (a, b) = ab.split_at_mut(q);
        let (c, d) = cd.split_at_mut(q);
        clean4::<T, D>(a, b, c, d);
        for part in v.chunks_exact_mut(q) {
            merge_pow2::<T, D>(part);
        }
    } else {
        let (a, b) = v.split_at_mut(n / 2);
        clean2::<T, D>(a, b);
        merge_pow2::<T, D>(a);
        merge_pow2::<T, D>(b);
    }
}

/// Elements of at most this many bytes are merged stage by stage ([`staged`]) once a merge of
/// a power-of-two length is at most [`STAGED_LEN`] long.
///
/// Measured against the straight-line merges of 16 for `u32`, `u16` and `u8`: 1.25 times
/// faster on 1000 random elements, 1.4–1.45 times on 16,384 and 1.3 times on a million for
/// `u32`, 1.6–1.8 times for `u8`. For `u64` stage by stage is 1.1–1.25 times slower: two
/// elements to a vector register do not pay for a pass over memory per stage, and the
/// straight-line merges keep them in registers.
const STAGED_MAX_SIZE: usize = 4;

/// The longest merge [`staged`] runs, for elements of at most [`STAGED_MAX_SIZE`] bytes:
/// 2048 elements, at most 8 KiB, stay in the first-level cache through all its stages.
/// Measured: from 2048 to 32,768 within 2–3% of each other, and 64 to 256 up to 1.3 times
/// slower.
const STAGED_LEN: usize = 2048;

/// Merges `v`, of a power-of-two length, into direction `D`, one stage at a time over all of
/// `v`: the compare-exchanges `2^s` apart, for `s` from `log₂ n − 1` down to 0, each stage in
/// one loop over the slice.
///
/// These are the compare-exchanges of [`merge_pow2`], in an order that still runs every one
/// after the ones it depends on, but the loops of a stage are long and regular: the compiler
/// runs a stage as vector minimums and maximums of 16 bytes, also the last three, which pair
/// neighbours 4, 2 and 1 apart (with interleaving loads and stores for the last two). The
/// straight-line merges of 16 elements, which [`merge_pow2`] recurses into otherwise, compare
/// one pair at a time: for narrow elements, a vector compares 4 to 16.
fn staged<T: Ord, D: Direction>(v: &mut [T]) {
    let n = v.len();
    let mut d = n / 2;
    while d >= 8 {
        for block in v.chunks_exact_mut(2 * d) {
            let (a, b) = block.split_at_mut(d);
            clean2::<T, D>(a, b);
        }
        d /= 2;
    }
    // Stages of 4, 2 and 1 positions apart in loops of their own: with the loop above, a
    // stage 4 apart has too few pairs per block for the compiler to use vectors.
    if d == 4 {
        for [a, b, c, e, f, g, h, i] in v.as_chunks_mut::<8>().0 {
            cx::<T, D>(a, f);
            cx::<T, D>(b, g);
            cx::<T, D>(c, h);
            cx::<T, D>(e, i);
        }
        d = 2;
    }
    if d == 2 {
        for [a, b, c, e] in v.as_chunks_mut::<4>().0 {
            cx::<T, D>(a, c);
            cx::<T, D>(b, e);
        }
        d = 1;
    }
    if d == 1 {
        for [a, b] in v.as_chunks_mut::<2>().0 {
            cx::<T, D>(a, b);
        }
    }
}

/// One stage: compare-exchanges each element of `x` with the element of `y` at the same
/// position, as far as the shorter of the two goes.
fn clean2<T: Ord, D: Direction>(x: &mut [T], y: &mut [T]) {
    for (a, b) in x.iter_mut().zip(y) {
        cx::<T, D>(a, b);
    }
}

/// Two stages: for each position, the elements of `a`, `b`, `c` and `d` there, which are
/// a quarter of a merge apart, through the stages comparing them half and a quarter of the
/// merge apart.
fn clean4<T: Ord, D: Direction>(a: &mut [T], b: &mut [T], c: &mut [T], d: &mut [T]) {
    for (((a, b), c), d) in a.iter_mut().zip(b).zip(c).zip(d) {
        cx::<T, D>(a, c);
        cx::<T, D>(b, d);
        cx::<T, D>(a, b);
        cx::<T, D>(c, d);
    }
}

/// Three stages: for each position, the elements of the eight parts there, an eighth of a
/// merge apart, through the stages comparing them half, a quarter and an eighth of the merge
/// apart.
fn clean8<T: Ord, D: Direction>(parts: [&mut [T]; 8]) {
    let [a, b, c, d, e, f, g, h] = parts;
    let groups = a
        .iter_mut()
        .zip(b)
        .zip(c)
        .zip(d)
        .zip(e)
        .zip(f)
        .zip(g)
        .zip(h);
    for (((((((a, b), c), d), e), f), g), h) in groups {
        cx::<T, D>(a, e);
        cx::<T, D>(b, f);
        cx::<T, D>(c, g);
        cx::<T, D>(d, h);
        cx::<T, D>(a, c);
        cx::<T, D>(b, d);
        cx::<T, D>(e, g);
        cx::<T, D>(f, h);
        cx::<T, D>(a, b);
        cx::<T, D>(c, d);
        cx::<T, D>(e, f);
        cx::<T, D>(g, h);
    }
}

/// Compare-exchanges `*a`, at the lower position, and `*b` into direction `D`, the way the
/// passes of the merges do.
#[inline(always)]
fn cx<T: Ord, D: Direction>(a: &mut T, b: &mut T) {
    let (a, b): (*mut T, *mut T) = (a, b);
    // SAFETY: two distinct mutable references are valid, non-overlapping pointers.
    unsafe {
        if D::ASCENDING {
            exchange(a, b, pass_exchange::<T>());
        } else {
            exchange(b, a, pass_exchange::<T>());
        }
    }
}

/// Puts the smaller of `*lo` and `*hi` into `*lo` and the other into `*hi`, by `how`:
/// [`VALUES`], [`POINTERS`] or [`BRANCH`]. Equal elements are left where they are.
///
/// # Safety
///
/// `lo` and `hi` must be valid for reads and writes and must not overlap.
#[inline(always)]
unsafe fn exchange<T: Ord>(lo: *mut T, hi: *mut T, how: u8) {
    // SAFETY: the caller guarantees two valid, distinct elements. They are compared in
    // place, so a panicking comparison leaves both untouched. After it, nothing can panic:
    // the elements are read as bit copies that are never dropped (`MaybeUninit`) and written
    // back so that each slot again holds one of the two, either swapped or not.
    unsafe {
        let swap = *hi < *lo;
        let (lo, hi) = (lo.cast::<MaybeUninit<T>>(), hi.cast::<MaybeUninit<T>>());
        if how == VALUES {
            let (new_lo, new_hi) = select_unpredictable(
                swap,
                (ptr::read(hi), ptr::read(lo)),
                (ptr::read(lo), ptr::read(hi)),
            );
            ptr::write(lo, new_lo);
            ptr::write(hi, new_hi);
        } else if how == POINTERS {
            let to_lo = select_unpredictable(swap, hi, lo);
            let to_hi = select_unpredictable(swap, lo, hi);
            let held = ptr::read(to_hi);
            // Onto itself if there is no swap, which `ptr::copy` allows.
            ptr::copy(to_lo, lo, 1);
            ptr::write(hi, held);
        } else if swap {
            ptr::swap_nonoverlapping(lo, hi, 1);
        }
    }
}

/// A sorting or merging network for at most [`SORT_LEAF`] or [`MERGE_LEAF`] elements: the
/// positions of its compare-exchanges, in order, each as the position that receives the
/// smaller element and the one that receives the larger, for an ascending result.
struct Network {
    pairs: [[u8; 2]; MAX_PAIRS],
    len: usize,
}

/// The most compare-exchanges of a [`Network`]: 80 for a sort of 16 elements; a merge of 16
/// has 32.
const MAX_PAIRS: usize = 80;

/// Appends the compare-exchange of positions `i < j` into direction `ascending` to `net`.
const fn push(net: &mut Network, i: usize, j: usize, ascending: bool) {
    net.pairs[net.len] = if ascending {
        [i as u8, j as u8]
    } else {
        [j as u8, i as u8]
    };
    net.len += 1;
}

/// Appends the network of [`merge`] for `v[lo..lo + n]` to `net`.
const fn network_merge(net: &mut Network, lo: usize, n: usize, ascending: bool) {
    if n > 1 {
        let m = 1 << (n - 1).ilog2();
        let mut i = lo;
        while i < lo + n - m {
            push(net, i, i + m, ascending);
            i += 1;
        }
        network_merge(net, lo, m, ascending);
        network_merge(net, lo + m, n - m, ascending);
    }
}

/// Appends the network of [`sort`] for `v[lo..lo + n]` to `net`.
const fn network_sort(net: &mut Network, lo: usize, n: usize, ascending: bool) {
    if n > 1 {
        let m = n / 2;
        network_sort(net, lo, m, !ascending);
        network_sort(net, lo + m, n - m, ascending);
        network_merge(net, lo, n, ascending);
    }
}

/// The merging networks for 0 to [`MERGE_LEAF`] elements if `merge`, otherwise the sorting
/// networks for 0 to [`SORT_LEAF`]; built at compile time, which also checks that they fit.
const fn networks(merge: bool) -> [Network; MERGE_LEAF + 1] {
    let mut nets = [const {
        Network {
            pairs: [[0; 2]; MAX_PAIRS],
            len: 0,
        }
    }; MERGE_LEAF + 1];
    let mut n = 0;
    while n <= MERGE_LEAF {
        if merge {
            network_merge(&mut nets[n], 0, n, true);
        } else if n <= SORT_LEAF {
            network_sort(&mut nets[n], 0, n, true);
        }
        n += 1;
    }
    nets
}

const SORT_NETWORKS: [Network; MERGE_LEAF + 1] = networks(false);
const MERGE_NETWORKS: [Network; MERGE_LEAF + 1] = networks(true);

/// Sorts `v`, of [`SORT_LEAF`] elements at most, or merges it, of [`MERGE_LEAF`] at most,
/// if `MERGE`, into direction `D`, by its straight-line network ([`network`]).
fn leaf<T: Ord, D: Direction, const MERGE: bool>(v: &mut [T]) {
    macro_rules! by_length {
        ($($n:literal)*) => {
            match v.len() {
                $($n => network::<T, D, $n, MERGE>(v),)*
                _ => {}
            }
        };
    }
    by_length!(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
}

/// Runs the network of [`sort`], or of [`merge`] if `MERGE`, for `N` elements on `v`, which
/// has `N` elements, into direction `D`.
///
/// Each compare-exchange is written out, with the positions read from the network at compile
/// time, so the compiler sees constant positions and keeps small elements in registers. A
/// loop over the network instead reloads both elements and both positions for each of them:
/// that made the whole sort 1.9–2.4 times slower for integers, 1.1–1.15 times for strings
/// and 1.35–1.55 times for 64-byte elements.
#[inline]
fn network<T: Ord, D: Direction, const N: usize, const MERGE: bool>(v: &mut [T]) {
    let v: &mut [T; N] = v.try_into().expect("`v` has `N` elements");
    let net = if MERGE {
        &MERGE_NETWORKS[N]
    } else {
        &SORT_NETWORKS[N]
    };
    let base = v.as_mut_ptr();
    macro_rules! compare_exchanges {
        ($($k:literal)*) => {
            $(
                // Decided at compile time, so the compare-exchanges past the network's length
                // are not generated at all, also without optimizations: checking `net.len`
                // when sorting instead let the build of the tests run out of memory in CI.
                if const { $k < if MERGE { MERGE_NETWORKS[N].len } else { SORT_NETWORKS[N].len } } {
                    let [lo, hi] = net.pairs[$k];
                    // SAFETY: the networks for `N` elements only name positions below `N`,
                    // and the two of a pair differ, so both are distinct elements of `v`.
                    unsafe {
                        let (lo, hi) = (base.add(lo as usize), base.add(hi as usize));
                        if D::ASCENDING {
                            exchange(lo, hi, network_exchange::<T>());
                        } else {
                            exchange(hi, lo, network_exchange::<T>());
                        }
                    }
                }
            )*
        };
    }
    // Merges of up to 16 elements have up to 32 compare-exchanges, sorts up to 80; a sort
    // also writing out the 48 that merges never use made the build of the tests take twice
    // as long.
    if MERGE {
        compare_exchanges!(
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        );
    } else {
        compare_exchanges!(
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
            60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
        );
    }
}
