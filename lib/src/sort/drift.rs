use core::mem::{self, MaybeUninit};
use core::{hint, ptr};

use super::merge::{PINGPONG_MAX, descents, merge_halves, sort_pingpong, sort_run};
use super::tim::{Buffer, MIN_GALLOP, build_run, extend_run, merge, node_power, power_scale};

/// Sorts a slice using drift sort.
///
/// Drift sort, designed by Orson Peters and Lukas Bergdoll in 2024 and the stable sort of
/// Rust's standard library since 1.81, drifts between merge sort and quick sort. It scans the
/// slice for natural runs, ascending or strictly descending, but keeps only those of at least
/// about `√n` elements (`min(n/2, 32)` below 4096 elements); everything else becomes chunks
/// that are left unsorted for now. Runs and chunks are combined in the order of a merge tree,
/// powersort's: two adjacent unsorted chunks are simply taken together, still unsorted, while
/// an unsorted chunk that meets a sorted run is first sorted by a stable quick sort and then
/// merged. Random input thus ends up as one big chunk for quick sort, which handles repeated
/// keys in `O(n log k)`, while input made of long runs is merged like Tim sort's, and anything
/// in between gets the cheaper of the two for each part.
///
/// Nine refinements keep the algorithm as cheap as drift sort can be. The ratios quoted are
/// measured on an Apple M4 Max with new input for every sort, the way a program sorts; where
/// sorting the same input again and again, which lets the processor learn every branch,
/// gives a different picture, that is said.
///
/// * **This crate's best parts.** Runs are merged by [`tim`](crate::sort::tim) sort's merge,
///   which skips the elements already in place, merges from both ends at once without
///   branching, gallops through runs that interleave in blocks and, once galloping has not
///   paid off, merges runs whose content interleaves finely as two merges at once (four
///   chains of comparisons the processor overlaps: 1.25–1.5 times faster on input made of
///   runs of random content); short parts are sorted by [`merge`](crate::sort::merge) sort's
///   branchless short-run sort (see below); a quick sort still partitioning after `2·log₂ n`
///   levels hands the part to merge sort, so the sort is `O(n log n)` on any input.
/// * **Pivot samples at several scales.** Pivots are chosen as in [`quick`](crate::sort::quick)
///   sort, from a median of 3 up to a pseudo-median of hundreds of samples, but quick sort's
///   ninther and evenly spaced pseudo-median sample at a single scale, which periodic input
///   whose period divides the spacing defeats: every sample shows the same key, each
///   partition peels off one key, and a sawtooth of 1000 elements cost 2.4 times the standard
///   library's time and even exhausted the quick sort limit. Here the samples lie in three
///   sections of an eighth of the width, centred at the middle and `3/8` of the width to
///   either side, recursively, so every level samples at its own scale: 9 samples at two
///   scales from 64 elements, 27 at three from 1024, three times as many for every 16 times
///   the length. The sawtooth of 1000 elements then costs 3.3–3.7 times less and random input
///   1–2% fewer comparisons; integers are 1–2% slower on random input, and a sawtooth of 100
///   integers 1.12 times, whose exact median the old samples happened to hit. The layout is
///   symmetric, so sorted input still gets the middle element.
/// * **Branchless stable partitioning.** Each element is compared with the pivot and copied
///   into the buffer, to the front if it goes left and to the back otherwise, from an address
///   the comparison selects; then both parts are copied back, the back one reversed. Every
///   element is compared in place before it is copied, and the pivot is copied last, after
///   every comparison with it, so a comparison that panics or mutates its arguments leaves no
///   stale copy behind. A partition that moved nothing, because nothing or everything went
///   left, skips the copies back: 1.05–1.3 times faster for integers with 2 distinct keys, up
///   to 1.07 times with 16. Elements larger than 8 bytes, whose copies cost more than the
///   bookkeeping, are instead compacted in place as they go left, only the right part passing
///   through the buffer, in order, with the pivot moved out into a local for the scan, its
///   only live copy, and written back last; a guard restores every element exactly once if a
///   comparison panics. Measured: 1.2–1.3 times faster on random 64-byte elements, 1.2–1.5
///   times with repeated keys, 1.1–1.2 times on random strings from 10,000 elements; strings
///   up to 1000 gain nothing.
/// * **Repeated keys in one pass, without copying the pivot.** When a part's pivot is no
///   greater than the pivot of the partition that produced the part, which is no greater than
///   anything in it, every element equal to the pivot is taken out in one pass, and only the
///   rest is sorted further. The standard library compares against a bitwise copy of the
///   earlier pivot, which is only sound for types without interior mutability; here that
///   pivot is found in place instead, by counting during each partition how many elements
///   went left before it. Input with `k` distinct keys costs `O(n log k)`: 1.3–1.45 times
///   faster for integers with few distinct keys than without the check, 1.1–1.2 times for
///   strings and 64-byte elements. Counting splits the partition loop once more, so for
///   elements of at most 8 bytes only parts of 128 elements or more count, whose left part is
///   likely to be partitioned again: 1.04–1.07 times faster for random integers, and as fast
///   with few distinct keys. Where no earlier pivot bounds a part from below (at the top, and
///   after every such pass), a pivot that is the part's minimum would cost a partition that
///   moves nothing before the pass that takes out its equals, so such a part is first scanned
///   for a smaller element, a comparison or two on most input: another 1.08–1.25 times
///   faster for integers with 2 distinct keys, 1.03–1.1 times for strings with 2 or 16, and
///   nothing for 64-byte elements or on random input.
/// * **No per-element bookkeeping.** The partition loop is split at the pivot and at the
///   earlier pivot instead of checking for them at every element, and for elements of at most
///   8 bytes it is unrolled by 4: 1.03–1.07 times faster for integers.
/// * **Short parts noted for order and merged in place.** A part of up to 32 elements of at
///   most 8 bytes is first scanned for descents, elements smaller than the one before, with
///   one comparison per element and no branch on the answers, into a mask of bits. Runs of 8
///   with no descent, and longer stretches of the merge tree without one, are left alone; the
///   other runs are sorted by merge sort's branchless short-run sort; and every merge reads
///   its halves where the level below left them, in the part or in the buffer, and writes the
///   other place, so no level is copied out before it is merged, and halves that come out in
///   order are not merged either. Larger elements, whose comparisons are calls, keep merge
///   sort's short-run sort for parts of up to 20 elements, and from 17 sort each half's first
///   8 elements by it, insert the rest and merge once. The merges, like the partition, leave
///   no stale copy behind when a comparison mutates its arguments: a merge of up to 16
///   elements decides the order before it copies anything, and a longer one copies the few
///   elements compared after their copy was written once more (see
///   [`merge`](crate::sort::merge)), while a merge of larger elements stops as soon as a half
///   has run out; this costs integers 1.02–1.05 times on random input of 100 elements or more
///   and strings 1.03–1.06 times up to 1000, against merges that compare stale copies, which
///   are sound only for types without interior mutability.
///   The former scheme for integers, each half's first 8 elements sorted and the rest
///   inserted, is quick only on input sorted again and again, once the branch predictor has
///   learnt its insertions: with new input, the mask is 1.02–1.04 times faster on random
///   integers, 1.01–1.04 times on nearly sorted input, runs and `√n` distinct keys, but 1.2
///   times slower on a sawtooth of 100 elements, whose repeated runs of 10 the insertions
///   handled once learnt.
/// * **Short slices adapt too.** A slice of up to 64 elements of at most 8 bytes gets the same
///   scan for descents: none, and it is left alone; one at every element, and it is reversed;
///   at most one per 8 elements, and the elements after the first descent are inserted into
///   the sorted prefix, as Tim sort extends its runs; anything else is sorted with the
///   descents already found. Sorted and strictly descending input costs `n − 1` comparisons.
///   Against scanning for a prefix run and sorting the rest by merge sort (up to 32 elements)
///   or as chunks of 32 (33 to 64), with new input every time: 1.2–1.5 times faster on random
///   integers and with few distinct keys, 1.1–1.8 times on input made of runs, 1.3–2 times on
///   reverse-sorted input, 1.1–2.4 times on sorted input from 12 elements on, 1.1–2.9 times
///   on sorted input with elements appended from 16 elements on and 1.3–1.6 times on nearly
///   sorted input from 40 elements on, but 1.1–1.25 times slower on 8 to 28 nearly sorted
///   elements, and 1.2–2.9 times slower on a sawtooth of 8 to 64 elements or blocks of 8 to
///   32, where the former entry's insertions were learnt (such input is periodic, so the
///   processor learns it even when it is new); with the same input again and again, the
///   learnt insertions are up to 1.3 times faster on 8 to 16 random elements and 1.1–1.9
///   times on 26 to 64 elements made of runs. Larger elements are scanned for a sorted or
///   strictly descending prefix and the rest inserted, which needs no buffer, up to 28
///   elements, or 40 for elements larger than 32 bytes, whose moves cost more than their
///   comparisons: 1.1–1.5 times faster than a chunk of 20 and the rest merged on random
///   strings of 24 to 28 elements, 1.3–2 times on nearly sorted or appended ones; 64-byte
///   elements gain 1.1–1.25 times on random input up to 40 elements and 1.3–1.9 times on
///   input with structure. Longer slices up to twice the small sort's length (40 elements
///   larger than 8 bytes) sort each chunk of that length the same way as soon as it is found,
///   its prefix being the run the scan just found, so no chunk is scanned twice, and merge
///   the chunks: 1.05–1.1 times faster on nearly sorted or block-wise sorted slices of 24–40
///   strings or 64-byte elements.
/// * **Runs cut by a chunk boundary are taken back.** A chunk ends where its length is up,
///   often inside a natural run, whose rest then falls short of the length a run must have.
///   A rest of at least half that length is probed with one or two comparisons and, if they
///   are in order, scanned backwards into the chunk, which gives the run back what it cut
///   off, the boundary moving with it, if that makes the run long enough: 1.05–1.2 times
///   faster on input made of runs of about `√n` elements, up to 1.04 times on nearly sorted
///   input below 4096 elements, and one comparison per chunk on sawtooth input, whose runs
///   stop just short.
/// * **No allocation for small slices or single runs.** The buffer lives on the stack while it
///   fits in 4 KiB, and is otherwise allocated when first needed; short slices of large
///   elements are insertion sorted and need none.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order. Runs are reversed only if strictly
///   descending, both parts of a partition keep their order, and merges and the small sort
///   are stable.
/// * **Not in place**: a buffer of `n` elements (the standard library's drift sort makes do
///   with about `n/2` for large inputs, at the price of a top-level merge that costs it
///   1.3–1.8 times on a million random strings), on the stack if it fits in 4 KiB and
///   allocated once otherwise.
/// * **Adaptive** to runs of `√n` elements or more, ascending or strictly descending, even
///   where a chunk boundary cuts them, and to repeated keys.
/// * **Sound for any `Ord`**: a comparison that panics, answers inconsistently or mutates its
///   arguments through interior mutability never leaves a stale copy, a duplicate or a hole
///   behind (see below).
///
/// | Case                                | Comparisons                  | Moves              |
/// |-------------------------------------|------------------------------|--------------------|
/// | Best (sorted, strictly descending)  | `n − 1`                      | `0` (`n` reversed) |
/// | `k` distinct keys                   | `O(n log k)`                 | `O(n log k)`       |
/// | Average (random)                    | `≈ (1.06–1.12)·n·log₂ n`     | `O(n log n)`       |
/// | Worst                               | `O(n log n)`                 | `O(n log n)`       |
///
/// For `n = 1000`: 999 comparisons on sorted or strictly descending input, about 11,200 on
/// random input, 5,700 with 16 distinct keys and 2,800 with 2 (Tim sort: 9,870, 8,780 and
/// 5,610), about 2,050 with 10 random pairs swapped and 7,100 on a sawtooth (the standard
/// library: 10,600, 5,660, 3,100, 6,600 and 8,900). Elements of at most 8 bytes trade about
/// 4% more comparisons on random input, and 6–13% on slices of up to 64 elements, for fewer
/// mispredicted branches.
///
/// Against [`slice::sort`], the standard library's drift sort, with new input for every sort:
/// 1.3–3 times faster on 8 to 64 random integers (it insertion sorts up to 20 elements and
/// merges chunks up to 64, which the processor only learns when the input repeats), 1.03–1.06
/// times slower on 100 to a million; 1.07–1.08 times slower on random strings of 100 to 1000
/// elements, as fast at 10,000 and 1.06–1.75 times faster beyond; 1.1–1.15 times slower on
/// random 64-byte elements up to 100, 1.06–1.25 times faster from 1000; 1.05–1.9 times
/// faster with few or `√n` distinct keys from 1000 elements on (integers with 16 keys within
/// 1.06 either way) and 1.25–1.65 times on a sawtooth of 1000 (3 times on one of a million
/// integers or strings); and 1.1–8 times faster on input with structure from 24 elements on:
/// nearly sorted, sorted with elements appended, made of runs of `√n` elements (from 10,000;
/// as fast below) or of interleaved blocks. Up to 20 nearly sorted elements its insertion
/// sort is 1.2–1.5 times faster. Against this crate's Tim sort: 1.2–1.85 times faster on
/// random integers and strings (1.3 times on 64-byte elements), 1.6–11 times with few
/// distinct keys and 1.3–2.4 times on sawtooth input up to 100,000 elements, but 1.1–1.35
/// times slower on nearly sorted integers (up to 2 times for strings at 100,000) and 1.05–1.4
/// times slower on input made of runs of `√n` elements, which Tim sort keeps whole.
///
/// Prefer [`slice::sort`] for real workloads.
///
/// # Panics
///
/// Panics only if allocating the buffer fails. If the [`Ord`] implementation of `T` panics,
/// the panic is propagated and `arr` is left in an unspecified order, but it still holds every
/// original element exactly once: nothing is dropped twice or leaked. An [`Ord`] that is not a
/// total order leaves `arr` in an unspecified order with every element exactly once as well,
/// and one that mutates its arguments through interior mutability sees every mutation kept:
/// no element is ever compared through a copy whose twin is live, nor compared with itself.
///
/// # Examples
///
/// ```
/// use a::sort::drift;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// drift(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Many repeated keys are taken out a key at a time.
/// let mut v: Vec<u8> = (0..1000).map(|i| (i * 7 % 3) as u8).collect();
/// drift(&mut v);
/// assert!(v.windows(2).all(|w| w[0] <= w[1]));
///
/// // Equal keys keep their order: cards are compared by rank only.
/// #[derive(Debug, PartialEq, Eq)]
/// struct Card(u8, char);
/// impl Ord for Card {
///     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
///         self.0.cmp(&other.0)
///     }
/// }
/// impl PartialOrd for Card {
///     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
///         Some(self.cmp(other))
///     }
/// }
/// let mut cards: Vec<Card> = "abcdefghij".chars().zip([3, 1, 3, 0, 1, 3, 0, 2, 1, 0])
///     .map(|(c, rank)| Card(rank, c))
///     .collect();
/// drift(&mut cards);
/// let order: String = cards.iter().map(|c| c.1).collect();
/// assert_eq!(order, "dgjbeihacf");
/// ```
pub fn drift<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 || mem::size_of::<T>() == 0 {
        // Zero-sized elements are all alike: there is nothing to reorder.
        return;
    }
    if mem::size_of::<T>() > SMALL_MAX_SIZE && len <= short_slice_len::<T>() {
        let (run, descending) = find_run(arr);
        if descending {
            arr[..run].reverse();
        }
        if run == len {
            return;
        }
        // Tim sort's way with short runs of large elements: by insertion, which is quick on
        // input in order but for a few elements and needs no buffer.
        extend_run(arr, run);
        return;
    }
    let mut buf = Buffer::<T>::new(len);
    if mem::size_of::<T>() <= SMALL_MAX_SIZE && len <= SHORT_MAX {
        sort_short(arr, &mut buf);
        return;
    }
    // Short slices sort their chunks as soon as they are found instead of leaving them for
    // quick sort.
    sort(arr, &mut buf, len <= 2 * small_sort_len::<T>());
}

/// Slices of at most this many elements of at most [`SMALL_MAX_SIZE`] bytes are sorted by
/// [`sort_short`]: as many as the descents of a slice that fit in a `u64`. Measured against
/// sorting 33 to 64 elements as two chunks and a merge, with new input every time: 1.2–1.45
/// times faster on random integers and with few distinct keys, 1.1–1.8 times on input made
/// of runs, 1.3–1.6 times on nearly sorted input and 1.9–3.2 times on sorted, reverse-sorted
/// or appended input; with the same input again and again, 1.1–1.3 times slower on integers
/// made of runs. Quick sorting 33 to 64 random integers at once, which a count of descents
/// told from input with order, was 1.1–1.3 times faster than the chunks but 1.2–1.35 times
/// slower than this.
const SHORT_MAX: usize = PINGPONG_MAX;

/// Sorts `arr`, at most [`SHORT_MAX`] elements of at most [`SMALL_MAX_SIZE`] bytes, with
/// `buf`, which has room for all of it. One pass of `len − 1` comparisons, with no branch on
/// their outcome, notes every descent: none means the slice is sorted, one at every element
/// that it is strictly descending and is reversed, at most `len / DESCENTS_DIV` that a few
/// elements are out of place and are inserted into the sorted prefix, and anything else goes
/// to the small sort with the descents it found, which spares it a scan of its own.
fn sort_short<T: Ord>(arr: &mut [T], buf: &mut Buffer<T>) {
    let len = arr.len();
    let descents = descents(arr);
    if descents == 0 {
        return;
    }
    if descents == all_descents(len) {
        arr.reverse();
        return;
    }
    if descents.count_ones() as usize <= len / DESCENTS_DIV {
        // The sorted prefix ends at the first descent.
        extend_run(arr, descents.trailing_zeros() as usize);
        return;
    }
    // SAFETY: `len <= SHORT_MAX == PINGPONG_MAX`, and `buf` has room for `len` elements,
    // which it does not overlap.
    unsafe { sort_pingpong(arr, buf.get(), descents) }
}

/// The descents of a strictly descending slice of `2 <= len <= 64` elements: every bit from
/// 1 to `len − 1`.
fn all_descents(len: usize) -> u64 {
    debug_assert!((2..=PINGPONG_MAX).contains(&len));
    (u64::MAX >> (PINGPONG_MAX - len)) & !1
}

/// A short slice with at most `len / DESCENTS_DIV` descents (elements smaller than the one
/// before) has its elements after the first inserted into the sorted prefix, which moves
/// little on such input; one with more goes to the small sort, whose branchless merges cost
/// the same whatever the order. Measured against inserting never: 2.3–4 times faster on
/// sorted integers with one appended and 1.3–1.5 times on nearly sorted ones, at 16 to 32
/// elements; and against inserting from 2 descents on however short the slice: 1.3–1.6 times
/// faster on 8 random integers, with few distinct keys or made of runs.
const DESCENTS_DIV: usize = 8;

/// The largest element size, in bytes, for which parts are sorted by [`small_sort`] from 32
/// elements down instead of 20, partitioning is unrolled, short parts and slices are checked
/// for order without branching (see [`sort_short`] and [`small_sort`]), and parts shorter
/// than [`track_min`] do not follow the ancestor pivot into their left part.
///
/// Measured: for integers, small parts of up to 32 elements make random input 1.05–1.1 times
/// faster than up to 20, and unrolling the partition loop by 4 another 1.03–1.07 times. For
/// strings and 64-byte elements, 32 is up to 1.2 times slower on short slices with few
/// distinct keys, and unrolling 1.2–1.35 times slower: their comparisons are calls, and the
/// longer loop only adds code. Insertion sort instead of merge sort's short-run sort would be
/// 1.1–1.5 times slower on random input for them.
const SMALL_MAX_SIZE: usize = 8;

/// Parts of at most this many elements are sorted by [`small_sort`], and chunks of this many
/// elements are sorted as they are found (see [`create_run`]).
const fn small_sort_len<T>() -> usize {
    if mem::size_of::<T>() <= SMALL_MAX_SIZE {
        32
    } else {
        20
    }
}

/// Slices of at most this many elements of more than [`SMALL_MAX_SIZE`] bytes are sorted
/// whole, the way Tim sort builds a short run: scanned for a sorted or strictly descending
/// prefix and the rest inserted into it, in place and without a buffer, instead of chunk by
/// chunk (see [`create_run`]): up to [`INSERTION_LEN`] elements, or twice [`small_sort_len`]
/// for elements larger than [`LARGE_INSERTION_SIZE`] bytes.
const fn short_slice_len<T>() -> usize {
    if mem::size_of::<T>() > LARGE_INSERTION_SIZE {
        2 * small_sort_len::<T>()
    } else {
        INSERTION_LEN
    }
}

/// See [`short_slice_len`]: how many elements of 9 to [`LARGE_INSERTION_SIZE`] bytes are insertion
/// sorted whole rather than as a chunk of [`small_sort_len`] and the rest, merged.
///
/// Measured for strings on slices of 21–28 elements: 1.03–1.5 times faster than the chunks on
/// random input and with few distinct keys, 1.3–2 times on nearly sorted or appended input,
/// whether the input is new to the branch predictor or not. From 30 elements on the
/// insertions lose on random input, few distinct keys or runs whenever the same data is
/// sorted repeatedly, up to 1.08 times at 30–32 and 1.3 times at 40 (and still win 1.02–1.1
/// times on new input), so the bound stops at 28. Elements larger than 32 bytes, whose moves
/// cost more than their comparisons unless they compare by a heap-allocated key, win up to 40
/// elements in both regimes: 1.02–1.2 times on random input and 1.1–2 times on input with
/// structure for 40-, 48- and 64-byte elements.
const INSERTION_LEN: usize = 28;

/// See [`short_slice_len`]: elements larger than this many bytes, whose moves cost more than
/// their comparisons unless they compare by a heap-allocated key, are insertion sorted whole
/// up to twice [`small_sort_len`] elements.
const LARGE_INSERTION_SIZE: usize = 32;

/// Parts of this many elements up to [`HALVES_MAX`], of more than [`SMALL_MAX_SIZE`] bytes,
/// are sorted by [`small_sort`] as two halves of at least 8 elements.
const HALVES_MIN: usize = 17;

/// See [`HALVES_MIN`]: every part [`small_sort`] gets for such elements.
const HALVES_MAX: usize = 20;

/// Parts shorter than this do not follow the ancestor pivot into their left part: counting
/// where it goes splits the partition loop once more, which for elements of at most
/// [`SMALL_MAX_SIZE`] bytes costs more than it saves when the left part is likely short enough
/// for [`small_sort`]. Measured for integers: 1.04–1.07 times faster on random input, and the
/// same with few distinct keys. Strings and 64-byte elements, whose comparisons dominate, lose
/// up to 1.06 times with few distinct keys without it, and follow it in every part.
const fn track_min<T>() -> usize {
    if mem::size_of::<T>() <= SMALL_MAX_SIZE {
        128
    } else {
        0
    }
}

/// Below `MIN_SQRT_RUN² = 4096` elements, a natural run must be `min(len / 2, 32)` elements
/// long to be kept; from there on, about `√len`.
const MIN_SQRT_RUN: usize = 64;
const MIN_MERGE_RUN: usize = 32;

/// A natural run that falls short of the length a run must have reaches back into the
/// unsorted chunk before it only if it has at least `1 / STEAL_DIV` of that length, which is
/// the most the chunk can have cut off a run worth keeping, see [`steal`]. Measured against
/// 4: the same.
const STEAL_DIV: usize = 2;

/// How many elements of at most [`SMALL_MAX_SIZE`] bytes [`stable_partition`] compares per
/// round of its loop.
const UNROLL: usize = 4;

/// Room on the run stack: the depths of the boundaries between runs on it strictly increase
/// and lie between 0 and 64.
const MAX_RUNS: usize = 66;

/// A stretch of the slice waiting to be merged: its length and whether it is sorted yet.
#[derive(Clone, Copy)]
struct Run {
    len: usize,
    sorted: bool,
}

/// Sorts `v` (more than [`small_sort_len`] elements) with `buf`, which has room for all of it.
/// If `eager`, every chunk is sorted as soon as it is found.
fn sort<T: Ord>(v: &mut [T], buf: &mut Buffer<T>, eager: bool) {
    let len = v.len();
    let scale = power_scale(len);
    let min_good_run = if len <= MIN_SQRT_RUN * MIN_SQRT_RUN {
        (len - len / 2).min(MIN_MERGE_RUN)
    } else {
        sqrt_approx(len)
    };
    let mut min_gallop = MIN_GALLOP;

    // The stack of runs waiting to be merged, with the depth of the boundary before each.
    let mut runs = [MaybeUninit::<Run>::uninit(); MAX_RUNS];
    let mut depths = [0u8; MAX_RUNS];
    let mut count = 0;

    // The length of the chunks sorted as soon as they are found, 0 while chunks are left
    // unsorted for quick sort.
    let eager_len = if eager { small_sort_len::<T>() } else { 0 };
    let mut scan = 0;
    let mut prev = Run {
        len: 0,
        sorted: true,
    };
    loop {
        let (next, depth) = if scan < len {
            let (next, stolen) = create_run(v, scan, prev, buf, min_good_run, eager_len);
            // A run that reached back into `prev` starts before `scan`: the boundary between
            // the two moves, before `prev` goes on the stack.
            prev.len -= stolen;
            scan -= stolen;
            let depth = node_power(scale, scan - prev.len, scan, scan + next.len);
            (next, depth)
        } else {
            (
                Run {
                    len: 0,
                    sorted: true,
                },
                0,
            )
        };
        // Merge the runs whose boundaries lie deeper in the merge tree than the boundary
        // between `prev` and `next` (powersort).
        while count > 1 && u32::from(depths[count - 1]) >= depth {
            // SAFETY: the first `count` slots have been written.
            let left = unsafe { runs[count - 1].assume_init() };
            let merged = left.len + prev.len;
            let start = scan - merged;
            prev = logical_merge(&mut v[start..scan], buf, left, prev, &mut min_gallop);
            count -= 1;
        }
        runs[count].write(prev);
        depths[count] = depth as u8;
        count += 1;
        if scan >= len {
            break;
        }
        scan += next.len;
        prev = next;
    }
    if !prev.sorted {
        stable_quicksort(v, buf, quicksort_limit(len), None);
    }
}

/// `⌊√n⌋`, roughly: `2^⌈log₂(n) / 2⌉` refined by one step of Newton's method.
fn sqrt_approx(n: usize) -> usize {
    let lg = n.ilog2();
    let guess = 1usize << lg.div_ceil(2);
    (guess + n / guess) / 2
}

fn quicksort_limit(len: usize) -> u32 {
    2 * (len | 1).ilog2()
}

/// Finds the run at `v[scan..]`: a sorted run if a natural one of at least `min_good_run`
/// elements is there (reversed if strictly descending), and otherwise the next chunk, a chunk
/// of `eager_len` elements sorted now if that is not 0 and a chunk of `min_good_run` elements
/// left unsorted otherwise. `prev` is the run before `scan`: a natural run that falls short of
/// `min_good_run` but goes on backwards into `prev`, if that is an unsorted chunk, takes those
/// elements from it (see [`steal`]). The second value returned is how many, by which `prev`
/// shrinks and the run starts before `scan`.
fn create_run<T: Ord>(
    v: &mut [T],
    scan: usize,
    prev: Run,
    buf: &mut Buffer<T>,
    min_good_run: usize,
    eager_len: usize,
) -> (Run, usize) {
    let len = v.len() - scan;
    // How much of `prev` a run could take: all but one element of an unsorted chunk.
    let stealable = if prev.sorted { 0 } else { prev.len - 1 };
    // The least a run must have to reach back into `prev`.
    let reach = min_good_run / STEAL_DIV;
    let unsorted = Run {
        len: len.min(min_good_run),
        sorted: false,
    };
    if len < min_good_run && eager_len == 0 && (len < reach || len + stealable < min_good_run) {
        // The last chunk, too short for a run even if it reached back.
        return (unsorted, 0);
    }
    let (run, descending) = find_run(&v[scan..]);
    if run >= min_good_run {
        if descending {
            v[scan..scan + run].reverse();
        }
        return (
            Run {
                len: run,
                sorted: true,
            },
            0,
        );
    }
    // Random input hardly ever gets here: its runs are 2 or 3 elements long.
    if run >= reach && run + stealable >= min_good_run {
        let stolen = steal(v, scan, run, descending, min_good_run - run, stealable);
        if stolen > 0 {
            return (
                Run {
                    len: run + stolen,
                    sorted: true,
                },
                stolen,
            );
        }
    }
    if eager_len == 0 {
        return (unsorted, 0);
    }
    // Sorted as found, Tim sort's way with short runs (see `build_run`). The run found is the
    // chunk's prefix, cut to the chunk if it reaches past it: the first elements of a strictly
    // descending run are strictly descending too.
    let chunk_len = len.min(eager_len);
    let chunk = &mut v[scan..scan + chunk_len];
    let sorted = run.min(chunk_len);
    if descending {
        chunk[..sorted].reverse();
    }
    build_run(chunk, sorted, buf);
    (
        Run {
            len: chunk_len,
            sorted: true,
        },
        0,
    )
}

/// Extends the natural run of `run` elements at `v[scan..]`, strictly descending if
/// `descending`, backwards over the elements before `scan` that continue it, if at least
/// `need` of them do (`need` is at least 1 and at most `stealable`, which is less than
/// `scan`): the chunk before the run cut them off it. Takes at most `stealable` of them and
/// returns how many, 0 if fewer than `need` continue the run, which is then left alone. A
/// descending run is reversed as a whole, stolen elements included.
///
/// An ascending run takes an element equal to its first, so equal elements keep their order;
/// a descending one only takes strictly greater elements, so reversing it cannot swap equal
/// elements. Every comparison is made in place, between two different elements.
fn steal<T: Ord>(
    v: &mut [T],
    scan: usize,
    run: usize,
    descending: bool,
    need: usize,
    stealable: usize,
) -> usize {
    // Whether `before`, the element just before `after` in `v`, continues the run.
    let continues = |before: &T, after: &T| {
        if descending {
            before > after
        } else {
            before <= after
        }
    };
    // The run can only reach `need` elements back if the farthest of them, and the one
    // halfway, are in order with the last element before `scan`: a comparison or two rule
    // out most false starts, such as blocks of two interleaved runs.
    if need > 1 && !continues(&v[scan - need], &v[scan - 1]) {
        return 0;
    }
    if need > 3 && !continues(&v[scan - need / 2], &v[scan - 1]) {
        return 0;
    }
    let mut taken = 0;
    while taken < need && continues(&v[scan - taken - 1], &v[scan - taken]) {
        taken += 1;
    }
    if taken < need {
        return 0;
    }
    // The run reaches far enough to keep: take all of it.
    while taken < stealable && continues(&v[scan - taken - 1], &v[scan - taken]) {
        taken += 1;
    }
    if descending {
        v[scan - taken..scan + run].reverse();
    }
    taken
}

/// Returns the length of the run at the start of `v` (at least 2 elements long), the longest
/// prefix that is ascending or strictly descending, and whether it descends.
fn find_run<T: Ord>(v: &[T]) -> (usize, bool) {
    let len = v.len();
    if len < 2 {
        return (len, false);
    }
    let mut end = 2;
    if v[1] < v[0] {
        while end < len && v[end] < v[end - 1] {
            end += 1;
        }
        (end, true)
    } else {
        while end < len && v[end] >= v[end - 1] {
            end += 1;
        }
        (end, false)
    }
}

/// Combines the adjacent runs `left` and `right`, which make up `v`. Two unsorted runs are
/// simply taken together, unsorted; otherwise an unsorted one is sorted first and the two are
/// merged.
fn logical_merge<T: Ord>(
    v: &mut [T],
    buf: &mut Buffer<T>,
    left: Run,
    right: Run,
    min_gallop: &mut usize,
) -> Run {
    let len = v.len();
    if !left.sorted && !right.sorted {
        return Run { len, sorted: false };
    }
    if !left.sorted {
        stable_quicksort(&mut v[..left.len], buf, quicksort_limit(left.len), None);
    }
    if !right.sorted {
        stable_quicksort(&mut v[left.len..], buf, quicksort_limit(right.len), None);
    }
    if left.len > 0 && right.len > 0 {
        merge(v, left.len, buf, min_gallop);
    }
    Run { len, sorted: true }
}

/// Sorts `v` stably by quick sort through `buf`, falling back to merge sort once `limit` more
/// levels of partitioning have not finished the job. `ancestor`, if set, is the index of an
/// element of `v` no greater than any other: the pivot of an enclosing partition.
fn stable_quicksort<T: Ord>(
    mut v: &mut [T],
    buf: &mut Buffer<T>,
    mut limit: u32,
    mut ancestor: Option<usize>,
) {
    loop {
        let len = v.len();
        if len <= small_sort_len::<T>() {
            small_sort(v, buf);
            return;
        }
        if limit == 0 {
            super::merge::sort(v, buf.get());
            return;
        }
        limit -= 1;
        let pivot = choose_pivot(v);
        let mut equal_partition = match ancestor {
            // The pivot may be the ancestor itself: equal, without a comparison.
            Some(a) => a == pivot || v[a] >= v[pivot],
            // No earlier pivot bounds the part from below. A pivot that is the part's minimum
            // would cost a partition that moves nothing before the one that takes out its
            // equals, so look for a smaller element first: a comparison or two on most input,
            // and a pass without moves when there is none. Parts too short to follow an
            // ancestor skip the check as well.
            None => len >= track_min::<T>() && is_minimum(v, pivot),
        };
        // Whether the ancestor is followed into the left part, which costs the partition loop
        // one more split.
        let track = ancestor.is_some() && len >= track_min::<T>();
        let mut split = None;
        if !equal_partition {
            let tracked = if track { ancestor.unwrap_or(0) } else { len };
            let scratch = scratch_of(buf, len);
            let p = stable_partition(v, scratch, pivot, false, tracked, |e, p| e < p);
            equal_partition = p.left == 0;
            split = Some(p);
        }
        let Some(p) = split.filter(|_| !equal_partition) else {
            // No element is smaller than the pivot: take out every element equal to it.
            let scratch = scratch_of(buf, len);
            let equal = stable_partition(v, scratch, pivot, true, 0, |e, p| e <= p).left;
            v = &mut v[equal..];
            ancestor = None;
            continue;
        };
        let left = p.left;
        // The pivot never goes left in a strict partition, so `left < len` whatever the
        // comparisons answer, and `left > 0` here: both parts are shorter than `v`.
        debug_assert!(0 < left && left < len);
        // The pivot, in the right part, is no greater than anything there; the ancestor, if
        // any, went left.
        let left_ancestor = Some(p.left_before_tracked).filter(|&a| track && a < left);
        let right_ancestor = Some(pivot - p.left_before_pivot).filter(|&a| a < len - left);
        let (l, r) = v.split_at_mut(left);
        if l.len() < r.len() {
            stable_quicksort(l, buf, limit, left_ancestor);
            v = r;
            ancestor = right_ancestor;
        } else {
            stable_quicksort(r, buf, limit, right_ancestor);
            v = l;
            ancestor = left_ancestor;
        }
    }
}

/// Whether no element of `v` is smaller than `v[pivot]`: every element is compared in place,
/// up to the first smaller one, and the pivot is not compared with itself.
fn is_minimum<T: Ord>(v: &[T], pivot: usize) -> bool {
    let p = &v[pivot];
    !v[..pivot].iter().any(|e| e < p) && !v[pivot + 1..].iter().any(|e| e < p)
}

/// `buf` as a slice of `len` slots, for a part of that many elements. The slice, unlike the
/// raw pointer, tells the compiler that nothing written through it aliases the part, which
/// lets a partition keep the pivot in a register (it reloaded it for every element otherwise).
fn scratch_of<T>(buf: &mut Buffer<T>, len: usize) -> &mut [MaybeUninit<T>] {
    // SAFETY: the buffer has room for the whole slice being sorted, aligned for `T`, so for
    // the `len` elements of a part of it; the slots are `MaybeUninit`, so what they hold does
    // not matter, and the slice borrows `buf`, which keeps the storage alive and unaliased.
    unsafe { core::slice::from_raw_parts_mut(buf.get().cast::<MaybeUninit<T>>(), len) }
}

/// Below this length the pivot is the median of three elements, at `len/4`, `len/2` and
/// `3·len/4`; from here on a pseudo-median of 9 samples at two scales (see [`pseudo_median`]).
const NINTHER: usize = 64;

/// From this length on, the pivot is the pseudo-median of 27 samples at three scales, and of
/// three times as many, one scale deeper, for every 16 times the length (81 from 16,384
/// elements, 243 from 262,144, 729 from 4,194,304).
///
/// Measured: 27 samples from 512 elements, the standard library's count, save about 1% of
/// the comparisons on random integers from 10,000 elements but no time, and cost 3–5% with
/// 16 distinct keys at 1000–10,000 elements, where partitions are cheap and few; 27 from
/// 4096, as before, leaves sizes below that with two scales only, which a sawtooth whose
/// period divides both offsets still defeats (`3.7·n·log₂ n` comparisons at 65,535 elements
/// with a period of 128).
const PSEUDO_MEDIAN: usize = 1024;

/// Returns the index of the pivot for partitioning `v`: the median of three samples below
/// [`NINTHER`] elements, and from there on a pseudo-median of `3^depth` samples laid out at
/// `depth − 1` scales (see [`pseudo_median`]), 9 from 64 elements, 27 from [`PSEUDO_MEDIAN`]
/// and three times as many for every 16 times the length.
///
/// Not [`quick`](crate::sort::quick) sort's: its ninther samples `len/4`, `len/2` and
/// `3·len/4`, each `± len/8`, and its pseudo-median an evenly spaced grid, both at a single
/// scale. A periodic input whose period divides that spacing (a sawtooth of 1000 elements
/// with a period of 31, say: `125 ≡ 1 (mod 31)`, so the samples at `k·125` are the keys 1 to
/// 7) then shows every sample about the same key, each partition peels off a key, and the
/// sort costs 2–4 times what it should, or even exhausts the quick sort limit.
fn choose_pivot<T: Ord>(v: &[T]) -> usize {
    let len = v.len();
    if len < NINTHER {
        return median_of_3(v, len / 4, len / 2, len / 4 * 3);
    }
    if len < PSEUDO_MEDIAN {
        return ninther(v, len / 2, len);
    }
    let mut depth = 3;
    let mut reach = PSEUDO_MEDIAN * 16;
    while len >= reach {
        depth += 1;
        reach = reach.saturating_mul(16);
    }
    pseudo_median(v, len / 2, len, depth)
}

/// The offset of a section's outer samples from its centre, `3/8` of its `width` (see
/// [`pseudo_median`]), computed so that it cannot overflow.
const fn offset(width: usize) -> usize {
    width / 8 * 3 + width % 8 * 3 / 8
}

/// Returns the index of the pseudo-median of 9 samples of `v` at two scales, the median of
/// the medians of three sections of `width / 8` elements centred at `mid` and `mid ± off`,
/// each sampled at its centre and `3/8` of its width to either side: [`pseudo_median`] at
/// depth 2, written out so that it costs no calls, which for parts of 64 to 1023 elements is
/// most of what the pivot costs (measured for integers against the recursion: 1.03–1.07
/// times faster on random input and with 16 or `√n` keys at 1000–10,000 elements, with
/// fresh input for every sort).
#[inline(always)]
fn ninther<T: Ord>(v: &[T], mid: usize, width: usize) -> usize {
    let off = offset(width);
    let inner = offset(width / 8);
    let a = median_of_3(v, mid - off - inner, mid - off, mid - off + inner);
    let b = median_of_3(v, mid - inner, mid, mid + inner);
    let c = median_of_3(v, mid + off - inner, mid + off, mid + off + inner);
    median_of_3(v, a, b, c)
}

/// Returns the index of the pseudo-median of `3^depth` samples of `v` spread over `width`
/// elements around `mid`: at depth 1, the median of the elements at `mid − off`, `mid` and
/// `mid + off`, where `off = 3·width/8`; deeper, the median of the pseudo-medians, one level
/// shallower, of three sections of `width/8` elements centred there (at depth 2, the
/// [`ninther`]).
///
/// Every level samples at its own scale (offsets of `3/8`, `3/64`, `3/512`, … of the width),
/// so the samples are not an arithmetic progression: a periodic input whose period divides
/// one level's offset is still spread by the other levels. Counted over every length from
/// 64 to a million elements with a sawtooth of period `√n`, or of 8 to 4096, the worst
/// length costs `1.6·n·log₂ n` comparisons (random input: about `1.1·n·log₂ n`), against
/// `3.9·n·log₂ n` with samples at a single scale. The layout is symmetric around `mid`, so
/// on sorted input the pivot is the middle element; the standard library's `0`, `n/2`,
/// `7n/8` costs 1.08 times as many comparisons on nearly sorted input.
///
/// Bounds: each offset is at most `3/8` of its level's width, and widths shrink 8 times per
/// level, so the farthest sample lies within `(3/8)·(1 + 1/8 + 1/64 + …)·width < 3·width/7`
/// of `mid` (rounding down only shrinks offsets), and so does every section's centre; with
/// `mid = len/2` and `width = len` every index lies in `[len/14 − 1, 13·len/14]`, inside `v`,
/// and no subtraction underflows. Sections at one level are `3·width/8` apart and reach less
/// than `3·width/56` from their centres, so they do not overlap, and the depth grows only
/// every 16 times the length while sections shrink 8 times, so the finest width is at least
/// 8 and the finest offset at least 3: the samples are all distinct.
fn pseudo_median<T: Ord>(v: &[T], mid: usize, width: usize, depth: u32) -> usize {
    let off = offset(width);
    if depth == 1 {
        return median_of_3(v, mid - off, mid, mid + off);
    }
    let a = pseudo_median(v, mid - off, width / 8, depth - 1);
    let b = pseudo_median(v, mid, width / 8, depth - 1);
    let c = pseudo_median(v, mid + off, width / 8, depth - 1);
    median_of_3(v, a, b, c)
}

/// Returns whichever of `a`, `b` and `c` indexes the median of the three elements, with two or
/// three comparisons. Two equal indices name the same element, which is the median then and
/// is not compared with itself (an `Ord` that borrows both arguments mutably would panic): a
/// safeguard, since the sample layouts keep every sample distinct.
fn median_of_3<T: Ord>(v: &[T], a: usize, b: usize, c: usize) -> usize {
    if a == b || b == c {
        return b;
    }
    if a == c {
        return a;
    }
    let b_below_a = v[b] < v[a];
    let c_below_b = v[c] < v[b];
    if b_below_a == c_below_b {
        // `a <= b <= c` or `a > b > c`.
        return b;
    }
    // `b` is the smallest or the largest: the median is the other extreme of `a` and `c`.
    let c_below_a = v[c] < v[a];
    if b_below_a == c_below_a { c } else { a }
}

/// What [`stable_partition`] found: how many elements went left, and how many of those were
/// before the pivot and before the tracked element.
struct Partition {
    left: usize,
    left_before_pivot: usize,
    left_before_tracked: usize,
}

/// Partitions `v` stably around the element at `pivot`: the elements for which
/// `goes_left(element, pivot)` holds come first, in their original order, then the others, in
/// their original order; the pivot itself goes left if `pivot_goes_left`. Returns how many go
/// left, and how many of those were before the pivot and before `tracked`. `scratch` has room
/// for all of `v`.
///
/// Every element is compared in place and copied into `scratch`, the ones going left to the
/// front and the others to the back, from where they are copied back in reverse; the pivot is
/// copied last, after every comparison with it. A partition that moved nothing, because
/// nothing or everything went left, skips the copies back: `v`, only read so far, already is
/// the partition. Elements larger than [`SMALL_MAX_SIZE`] bytes are partitioned by
/// [`stable_partition_inplace`] instead.
fn stable_partition<T: Ord>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot: usize,
    pivot_goes_left: bool,
    tracked: usize,
    goes_left: impl Fn(&T, &T) -> bool,
) -> Partition {
    if mem::size_of::<T>() > SMALL_MAX_SIZE {
        return stable_partition_inplace(v, scratch, pivot, pivot_goes_left, tracked, goes_left);
    }
    let len = v.len();
    let scratch = scratch[..len].as_mut_ptr().cast::<T>();
    let base = v.as_mut_ptr();
    assert!(pivot < len);
    let mut left = 0;
    // SAFETY: `scratch` has room for `len` elements (checked above) and does not overlap `v`,
    // and `pivot < len`. After `i` elements, `left` of them went to `scratch[..left]` and
    // `i - left` to the back, `scratch[len - (i - left)..]`, so element `i` goes to
    // `scratch[left]` or to `scratch[len - 1 - i + left]`: free slots inside `scratch`,
    // whatever the comparisons answer. The pivot's turn only reserves its slot; it is copied
    // after the loops, once every comparison with it is made, and every other element is
    // copied after its own comparison. `v` is only read until then, so a panicking comparison
    // leaves it untouched, and the copies in `scratch` are never dropped. The copies back then
    // fill `v` with every element exactly once: the left part in order, the right part, stored
    // back to front, reversed. If nothing went one way, `v`, only read so far, already is the
    // partition, with every element's latest state, and the copies are left behind.
    unsafe {
        let pivot_ptr = base.add(pivot);
        // Copies `v[from..to]`, none of them the pivot, to where they go.
        let scan = |from: usize, to: usize, left: &mut usize| {
            let step = |i: usize, left: &mut usize| {
                let e = base.add(i);
                let goes = goes_left(&*e, &*pivot_ptr);
                let back = scratch.add(len - 1 - i);
                let dst = hint::select_unpredictable(goes, scratch, back).add(*left);
                ptr::copy_nonoverlapping(e, dst, 1);
                *left += usize::from(goes);
            };
            let mut i = from;
            while i + UNROLL <= to {
                for k in 0..UNROLL {
                    step(i + k, left);
                }
                i += UNROLL;
            }
            while i < to {
                step(i, left);
                i += 1;
            }
        };
        let tracked = tracked.min(len);
        let mut left_before_tracked = 0;
        if tracked < pivot {
            scan(0, tracked, &mut left);
            left_before_tracked = left;
            scan(tracked, pivot, &mut left);
        } else {
            scan(0, pivot, &mut left);
        }
        let left_before_pivot = left;
        let pivot_dst = if pivot_goes_left {
            scratch.add(left)
        } else {
            scratch.add(len - 1 - pivot + left)
        };
        left += usize::from(pivot_goes_left);
        if tracked > pivot {
            scan(pivot + 1, tracked, &mut left);
            left_before_tracked = left;
            scan(tracked, len, &mut left);
        } else {
            scan(pivot + 1, len, &mut left);
        }
        if tracked == pivot {
            left_before_tracked = left_before_pivot;
        }
        // Everything goes left only with the pivot.
        if left != 0 && left != len {
            ptr::copy_nonoverlapping(pivot_ptr, pivot_dst, 1);
            ptr::copy_nonoverlapping(scratch, base, left);
            for i in 0..len - left {
                ptr::copy_nonoverlapping(scratch.add(len - 1 - i), base.add(left + i), 1);
            }
        }
        Partition {
            left,
            left_before_pivot,
            left_before_tracked,
        }
    }
}

/// The progress of [`stable_partition_inplace`], which writes into `v` while comparisons that
/// may panic remain, kept so that `v` can be restored with every element exactly once. The
/// pivot was moved out into `pivot` before the scan, which leaves its slot `pivot_pos` of `v`
/// stale. `state` is `(i, left)`: of the first `i` elements, `left` went left and sit in
/// `v[..left]`, the others in `scratch[..i − left]`, both in order; `v[left..i]` is stale and
/// `v[i..]` unread. Once `i > pivot_pos`, the pivot's slot is only reserved: at
/// `v[left_before_pivot]` if it goes left (it is counted in `left` then), else at
/// `scratch[pivot_pos − left_before_pivot]`, a gap among the elements that went right.
struct InplaceGuard<T> {
    base: *mut T,
    scratch: *mut T,
    pivot: MaybeUninit<T>,
    pivot_pos: usize,
    pivot_goes_left: bool,
    left_before_pivot: usize,
    state: (usize, usize),
}

impl<T> Drop for InplaceGuard<T> {
    fn drop(&mut self) {
        let (i, left) = self.state;
        // SAFETY: by the invariant above, the copy fills exactly the stale slots `v[left..i]`
        // with the elements in `scratch`, in order, among them the pivot's gap if its turn has
        // come and it goes right; the pivot then goes to the one slot that is still stale: its
        // original one if its turn has not come, else the slot reserved in `v[..left]` or the
        // slot its gap was copied to. `v` then holds every element exactly once, and the
        // copies in `scratch` are never read again.
        unsafe {
            ptr::copy_nonoverlapping(self.scratch, self.base.add(left), i - left);
            let slot = if i <= self.pivot_pos {
                self.pivot_pos
            } else if self.pivot_goes_left {
                self.left_before_pivot
            } else {
                left + self.pivot_pos - self.left_before_pivot
            };
            ptr::copy_nonoverlapping(self.pivot.as_ptr(), self.base.add(slot), 1);
        }
    }
}

/// [`stable_partition`] for elements larger than [`SMALL_MAX_SIZE`] bytes, whose copies cost
/// more than the bookkeeping: the elements going left are compacted in place, into
/// `v[..left]`, the others copied in order into `scratch`, and only those are copied back, in
/// one block, so each element is copied once or twice instead of twice and the right part is
/// never reversed. Elements of at most 8 bytes gain nothing: their scan, whose stores are
/// the limit, would be slower and their copies back are cheap. Measured for a partition
/// alone, 1.3 times faster for random 64-byte elements, 1.05–1.13 times for strings.
///
/// The pivot is moved out into a local for the whole scan, the only live copy of it: the
/// comparisons read it there, which also lets the compiler keep it in registers while the
/// scan writes into `v`, and it is written into its final slot last, after every comparison.
/// Element `i` goes to `v[left]` (`left ≤ i`, a slot already read) or to `scratch[i − left]`;
/// the pivot's final slot is only reserved at its turn. Since `v` is written while
/// comparisons remain, an [`InplaceGuard`] records the progress before each comparison and
/// restores `v`, every element exactly once, if one panics; for comparisons that cannot
/// unwind the compiler drops those stores.
fn stable_partition_inplace<T: Ord>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot: usize,
    pivot_goes_left: bool,
    tracked: usize,
    goes_left: impl Fn(&T, &T) -> bool,
) -> Partition {
    let len = v.len();
    let scratch = scratch[..len].as_mut_ptr().cast::<T>();
    let base = v.as_mut_ptr();
    assert!(pivot < len);
    let mut left = 0;
    // SAFETY: `scratch` has room for `len` elements (checked above) and does not overlap `v`,
    // and `pivot < len`. Every element is compared in place, the pivot in its local, and then
    // copied to a slot that is free whatever the comparisons answer: `v[left]` has been read
    // (`left <= i`; a copy onto itself when equal) and `scratch[i - left]` not been written,
    // and the pivot's reserved slot is never reached again since both counts only grow. The
    // guard's state is written before each comparison, so a panic finds it current, and
    // nothing else between two states can panic; dropping `guard` then restores `v` (see
    // [`InplaceGuard`]). Otherwise the pivot and the right part are copied in at the end,
    // which completes `v`, and `guard` is forgotten. The copies left in `scratch` and the
    // pivot's stale slot are never dropped or compared: the local is the pivot's only live
    // copy, so a comparison that mutates it through interior mutability is not lost.
    unsafe {
        let mut guard = InplaceGuard {
            base,
            scratch,
            pivot: MaybeUninit::uninit(),
            pivot_pos: pivot,
            pivot_goes_left,
            left_before_pivot: 0,
            state: (0, 0),
        };
        ptr::copy_nonoverlapping(base.add(pivot), guard.pivot.as_mut_ptr(), 1);
        let pivot_ref: &T = &*guard.pivot.as_ptr();
        let state: *mut (usize, usize) = &mut guard.state;
        // Copies `v[from..to]`, none of them the pivot, to where they go.
        let scan = |from: usize, to: usize, left: &mut usize| {
            for i in from..to {
                let l = *left;
                ptr::write(state, (i, l));
                let e = base.add(i);
                let goes = goes_left(&*e, pivot_ref);
                let dst = hint::select_unpredictable(goes, base.add(l), scratch.add(i - l));
                ptr::copy(e, dst, 1);
                *left = l + usize::from(goes);
            }
        };
        let tracked = tracked.min(len);
        let mut left_before_tracked = 0;
        if tracked < pivot {
            scan(0, tracked, &mut left);
            left_before_tracked = left;
            scan(tracked, pivot, &mut left);
        } else {
            scan(0, pivot, &mut left);
        }
        let left_before_pivot = left;
        guard.left_before_pivot = left;
        left += usize::from(pivot_goes_left);
        ptr::write(state, (pivot + 1, left));
        if tracked > pivot {
            scan(pivot + 1, tracked, &mut left);
            left_before_tracked = left;
            scan(tracked, len, &mut left);
        } else {
            scan(pivot + 1, len, &mut left);
        }
        if tracked == pivot {
            left_before_tracked = left_before_pivot;
        }
        // Nothing going left (`left == 0`, a whole copy back) takes an `Ord` that is not a
        // total order: the minimum check or the ancestor precedes every strict partition, and
        // the pivot itself goes left in the other kind; it is not worth a branch.
        let pivot_dst = if pivot_goes_left {
            base.add(left_before_pivot)
        } else {
            scratch.add(pivot - left_before_pivot)
        };
        ptr::copy_nonoverlapping(guard.pivot.as_ptr(), pivot_dst, 1);
        ptr::copy_nonoverlapping(scratch, base.add(left), len - left);
        mem::forget(guard);
        Partition {
            left,
            left_before_pivot,
            left_before_tracked,
        }
    }
}

/// Sorts `v`, at most [`small_sort_len`] elements, stably. Elements of at most
/// [`SMALL_MAX_SIZE`] bytes go to merge sort's short-run sort in its ping-pong form (see
/// [`sort_pingpong`]), which notes the descents of the part by one pass of `len − 1`
/// comparisons with no branch on their outcome, leaves the runs and stretches without one
/// alone, sorts the other runs without branching and merges every level in place of the one
/// below instead of copying it out first. Larger elements go to merge sort's short-run sort
/// as it is, and from [`HALVES_MIN`] elements up to [`HALVES_MAX`] as two halves, each of
/// whose first 8 elements are sorted by it and the rest inserted, merged once: merge sort
/// alone would split 17 to 20 elements into runs of 4 to 5 and merge three times.
fn small_sort<T: Ord>(v: &mut [T], buf: &mut Buffer<T>) {
    let len = v.len();
    let scratch = buf.get();
    if mem::size_of::<T>() <= SMALL_MAX_SIZE {
        // SAFETY: `len <= small_sort_len::<T>() <= PINGPONG_MAX`, and `scratch` has room for
        // all of `v`, which it does not overlap.
        unsafe { sort_pingpong(v, scratch, descents(v)) };
    } else if (HALVES_MIN..=HALVES_MAX).contains(&len) {
        let (left, right) = v.split_at_mut(len / 2);
        for half in [left, right] {
            // SAFETY: a half has at least `HALVES_MIN / 2 >= 8` elements, and `scratch` room
            // for all of `v`, which it does not overlap.
            unsafe { sort_run::<T, true>(&mut half[..8], scratch) };
            extend_run(half, 8);
        }
        // SAFETY: as above, and `len >= 2`.
        unsafe { merge_halves(v, scratch) };
    } else {
        super::merge::sort(v, scratch);
    }
}
