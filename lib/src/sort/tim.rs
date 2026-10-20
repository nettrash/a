use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::{hint, ptr, slice};

/// Sorts a slice using Tim sort.
///
/// Tim sort, designed by Tim Peters for Python's `list.sort` in 2002, is a merge sort that
/// makes use of the order already in the input. It splits the slice into runs, the longest
/// stretches that are ascending or strictly descending, and reverses the descending ones. Runs
/// shorter than a minimum length, between 16 and 32 elements and chosen so that the slice
/// splits into a power of two runs or slightly fewer, are extended to it by insertion sort.
/// Runs are merged as they are found, in an order kept on a stack, until one is left. Random
/// input has short runs and costs `O(n log n)` comparisons, like any merge sort, but input
/// made of a few long runs costs little more than finding them: sorted and strictly
/// descending input costs `n − 1` comparisons.
///
/// Six refinements keep the algorithm as cheap as Tim sort can be:
///
/// * **Galloping.** Before a merge, two searches find the elements of the left run that are
///   not greater than the first element of the right run, and those of the right run that are
///   not smaller than the last of the left run: they are already in place and stay out of the
///   merge. During a merge, once one run has supplied seven elements in a row, the merge
///   gallops: it finds how many more elements of that run come next by probing 1, 2, 4, ...
///   elements ahead and binary searching the last gap, and moves them as a block, `k` elements
///   for about `2·log₂ k` comparisons. The threshold drops while galloping pays off and rises
///   when it does not, so random input hardly ever gallops.
/// * **Powersort merge policy.** Two runs are merged once the boundary between them lies deeper
///   in a perfectly balanced merge tree over the slice than the boundary after them (Munro and
///   Wild's powersort, which CPython's Tim sort has used since Python 3.11). This merges in
///   close to the cheapest order possible, and the stack never holds more than 65 runs.
///   Peters' original rules, which keep the lengths of the runs on the stack growing like
///   Fibonacci numbers, are 1.04–1.12 times slower on input made of runs of varying lengths
///   and no faster on any other input measured.
/// * **Bidirectional, branchless merging.** A merge copies both runs into a buffer and merges
///   them back from both ends at once, like this crate's [`merge`](crate::sort::merge) sort:
///   the front takes the smaller of the two first elements and the back the larger of the two
///   last ones, two independent chains of comparisons that the processor overlaps, and each
///   step copies from an address chosen by the comparison instead of branching on it. Each end
///   counts its own winning streaks and gallops on its own. Tim sort's classic merge copies
///   only the shorter run out and merges from one end; it is 1.15–1.45 times slower on random
///   input and 1.4–1.6 times on runs of varying lengths, but it moves less when one run is
///   much longer than the other, so merges of runs more than four times as long as each other
///   still use it. For elements of at most 8 bytes, each end advances its pointer by
///   selecting between it and the next one rather than adding the comparison's outcome, one
///   instruction less on the chain from a comparison to the next load, and the merge checks
///   for a run that ran out once per batch of steps that cannot run past one rather than at
///   every step.
/// * **Two merges at once when galloping does not pay.** The merge is bound by that chain
///   from each comparison to the next load, not by instructions. Once galloping has failed
///   to pay off in a sort, every merge long enough is split at the middle of its result by a
///   binary search (co-ranking) into two independent merges, both from both ends, whose four
///   chains the processor overlaps; an end that takes from the same run for a whole batch of
///   steps hands what is left back to the galloping merge. With the select and the batches
///   above, and new input for every sort: input made of runs of random content sorts 1.2–1.8
///   times faster for integers and strings and 1.15 times for 64-byte elements, random input
///   1.1–1.4 times (which merges the short runs it starts from), and input with `√n` distinct
///   keys 1.05–1.2 times; input whose runs interleave in blocks keeps galloping.
/// * **Pair insertion.** Short runs are extended by inserting elements two at a time, scanning
///   from the right: the larger of the two is carried left first, and the smaller one goes on
///   from where it stopped, so the elements both of them pass move once. Peters' binary
///   insertion makes fewer comparisons, but each probe of a binary search waits for the
///   previous one, and its moves go through `memmove`: on random input it is 1.2–1.9 times
///   slower, strings included, most on short slices. Inserting one element at a time is
///   1.03–1.1 times slower for integers and large elements, and no faster for strings.
/// * **Branchless short runs for small elements.** Each insertion ends in a mispredicted
///   branch, which costs elements of at most 8 bytes, cheap to compare and move, more than
///   everything else. For them, a short run that starts with fewer than four sorted elements
///   is instead sorted by this crate's merge sort, which sorts pieces of at most four elements
///   by fixed sequences of comparisons that select where each element goes, and merges them
///   from both ends without branching. This is 1.1–1.3 times faster for integers on random
///   input and 1.1–1.2 times with few distinct keys, and up to 1.25 times slower on short
///   slices that are nearly sorted in small steps, where insertion barely moves anything.
/// * **No allocation for small slices or single runs.** The buffer lives on the stack while it
///   fits in 4 KiB (512 `u64`s, say), and is otherwise allocated when the first merge needs it:
///   sorted and strictly descending input never allocates.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order. Only strictly descending runs are
///   reversed, insertion carries an element past strictly greater ones only, and every
///   comparison that decides a placement lets an element of the right run overtake one of the
///   left run only if it is strictly smaller.
/// * **Not in place**: a buffer of `n` elements, on the stack if it fits in 4 KiB and allocated
///   once otherwise.
/// * **Adaptive** to runs of any length, ascending or strictly descending, and to runs whose
///   elements interleave in blocks.
///
/// Let `H` be the entropy of the run lengths, `H = Σ (lᵢ/n)·log₂(n/lᵢ)` for runs of lengths
/// `lᵢ`, which is at most `log₂ r` for `r` runs and small when a few runs hold most elements.
///
/// | Case                                | Comparisons                  | Moves              |
/// |-------------------------------------|------------------------------|--------------------|
/// | Best (sorted, strictly descending)  | `n − 1`                      | `0` (`n` reversed) |
/// | Runs of at least 32 elements        | `O(n + n·H)`                 | `O(n + n·H)`       |
/// | Average (random)                    | `≈ (0.98–1.17)·n·log₂ n`     | `O(n log n)`       |
/// | Worst                               | `O(n log n)`                 | `O(n log n)`       |
///
/// For `n = 1000`: 999 comparisons on sorted or strictly descending input, about 1,180 on
/// sorted input with 10 random elements appended and about 1,800 with 10 random pairs
/// swapped (this crate's merge sort makes about 3,020 and 6,100), and about 9,770 on random
/// input for elements of at most 8 bytes and 11,700 for larger ones (merge sort: about 9,640).
/// Linear insertion into runs of 16 to 32 elements costs more comparisons than a binary search
/// would, but less time.
///
/// Against this crate's merge sort, this is as fast or up to 1.3 times faster on random
/// integers, 1.05–1.4 times faster on random strings and 1.03–1.1 times on 64-byte elements
/// from 1000 elements on, 2.7–4.3 times faster for integers and strings on nearly sorted
/// input, sorted input with elements appended, or two interleaved runs (1.3–3.1 times for
/// 64-byte elements), and 7–11 times on reverse-sorted input. Against [`slice::sort`], which
/// also finds runs: 1.35–1.7 times slower on random integers from 1000 elements on and
/// 1.3–1.55 times on random strings, but 1.6–4.2 times faster on nearly sorted input,
/// 2.6–3.9 times on appended elements or two interleaved runs and 1.4–1.8 times on input
/// made of many runs, for integers and strings.
///
/// Prefer [`slice::sort`], a far more refined stable sort that also finds runs, for real
/// workloads.
///
/// # Panics
///
/// Panics only if allocating the buffer fails. If the [`Ord`] implementation of `T` panics,
/// the panic is propagated and `arr` is left in an unspecified order, but it still holds every
/// original element exactly once: nothing is dropped twice or leaked. An [`Ord`] that is not a
/// total order leaves `arr` in an unspecified order with every element exactly once as well.
///
/// # Examples
///
/// ```
/// use a::sort::tim;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// tim(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // A sorted log with late entries appended is two runs: one merge, mostly galloping.
/// let mut times: Vec<u32> = (0..1000).map(|i| i * 10).collect();
/// times.extend([15, 4005, 9999]);
/// tim(&mut times);
/// assert!(times.windows(2).all(|w| w[0] <= w[1]));
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
/// tim(&mut cards);
/// let order: String = cards.iter().map(|c| c.1).collect();
/// assert_eq!(order, "dgjbeihacf");
/// ```
pub fn tim<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 || mem::size_of::<T>() == 0 {
        // Zero-sized elements are all alike: there is nothing to reorder.
        return;
    }
    let mut buf = Buffer::<T>::new(len);
    if len < MIN_MERGE {
        let run = find_run(arr);
        build_run(arr, run, &mut buf);
        return;
    }

    let min_run = min_run_length(len);
    let scale = power_scale(len);
    let mut min_gallop = MIN_GALLOP;
    let mut runs = RunStack::new();

    let mut start = 0;
    while start < len {
        let mut run = find_run(&mut arr[start..]);
        if run < min_run {
            let end = len.min(start + min_run);
            build_run(&mut arr[start..end], run, &mut buf);
            run = end - start;
        }
        // Powersort: first merge the runs whose boundaries lie deeper in the merge tree than
        // the boundary between the top run and this one.
        let power = match runs.top(0) {
            Some(top) => node_power(scale, top.start, start, start + run),
            None => 0,
        };
        while runs.len > 1 && runs.top(0).is_some_and(|top| top.power >= power) {
            merge_top(arr, &mut runs, &mut buf, &mut min_gallop);
        }
        runs.push(Run {
            start,
            len: run,
            power,
        });
        start += run;
    }
    while runs.len > 1 {
        merge_top(arr, &mut runs, &mut buf, &mut min_gallop);
    }
}

/// Slices shorter than this are sorted by [`extend_run`] alone; longer ones are split into
/// runs of at least half this length.
///
/// Measured against 64 (runs of 32 to 64 elements, as in CPython): 1.1–1.2 times faster on
/// random input of 100 to 1000 elements and 1.02–1.07 times beyond, for integers, strings and
/// large elements alike. Against 16: 1.1–1.3 times faster on slices of 20 to 100 elements and
/// within 4% either way beyond.
const MIN_MERGE: usize = 32;

/// How many times in a row one run must win before a merge starts galloping.
pub(super) const MIN_GALLOP: usize = 7;

/// A merge of runs one of which is more than this many times as long as the other copies only
/// the shorter one out and merges from one end, as Tim sort originally does; other merges copy
/// both out and merge from both ends.
///
/// Measured against merging from both ends always: 1.05–1.55 times faster on a sorted slice
/// with 1% of random elements appended, no different on random input or input of runs of
/// random lengths. Merging from one end whenever galloping had paid off in earlier merges
/// would also be 1.05–1.2 times faster on nearly sorted input and few distinct keys, but
/// 1.1–1.15 times slower on input of runs of random lengths.
const LOPSIDED: usize = 4;

/// The size of the merge buffer kept on the stack, in bytes.
const STACK_BYTES: usize = 4096;

/// Room on the run stack. Every run but the bottom one carries the power of the boundary
/// before it; these strictly increase from bottom to top and lie between 0 and 63, so the
/// stack never holds more than 65 runs.
const MAX_RUNS: usize = 65;

/// A sorted run on the stack, with the power of the boundary between it and the run before.
#[derive(Clone, Copy)]
struct Run {
    start: usize,
    len: usize,
    power: u32,
}

/// The runs waiting to be merged, bottom first. Its slots are left uninitialised until
/// pushed to, which saves clearing 1.5 KiB on every call: measurable on short slices.
struct RunStack {
    runs: [MaybeUninit<Run>; MAX_RUNS],
    len: usize,
}

impl RunStack {
    fn new() -> Self {
        Self {
            runs: [MaybeUninit::uninit(); MAX_RUNS],
            len: 0,
        }
    }

    /// Returns the run `depth` places below the top one, if there is one.
    fn top(&self, depth: usize) -> Option<Run> {
        let i = self.len.checked_sub(depth + 1)?;
        // SAFETY: the first `len` slots have been written by `push` or `merge_top`.
        Some(unsafe { self.runs[i].assume_init() })
    }

    fn push(&mut self, run: Run) {
        self.runs[self.len].write(run);
        self.len += 1;
    }
}

/// Room for `STACK_BYTES` bytes, aligned for `T`.
#[repr(C)]
struct StackBuffer<T> {
    _align: [T; 0],
    bytes: [MaybeUninit<u8>; STACK_BYTES],
}

/// Raw storage for `len` elements: on the stack if they fit in it, and otherwise allocated
/// when the first merge asks for it, so that input made of one run allocates nothing.
pub(super) struct Buffer<T> {
    stack: StackBuffer<T>,
    // Its length stays 0, so it never drops what passes through it.
    heap: Vec<T>,
    len: usize,
}

impl<T> Buffer<T> {
    pub(super) fn new(len: usize) -> Self {
        Self {
            stack: StackBuffer {
                _align: [],
                bytes: [MaybeUninit::uninit(); STACK_BYTES],
            },
            heap: Vec::new(),
            len,
        }
    }

    /// Returns the stack storage, room for `STACK_BYTES` bytes of elements, which does not
    /// overlap the slice being sorted.
    fn on_stack(&mut self) -> *mut T {
        self.stack.bytes.as_mut_ptr().cast::<T>()
    }

    /// Returns storage for `len` elements, which does not overlap the slice being sorted.
    pub(super) fn get(&mut self) -> *mut T {
        if self.len <= STACK_BYTES / mem::size_of::<T>() {
            return self.on_stack();
        }
        if self.heap.capacity() == 0 {
            self.heap = Vec::with_capacity(self.len);
        }
        self.heap.as_mut_ptr()
    }
}

/// Returns the minimum run length for a slice of `n >= MIN_MERGE` elements: `n` divided by the
/// power of two that brings it below `MIN_MERGE`, rounded up. The result lies between
/// `MIN_MERGE / 2` and `MIN_MERGE`, and `n` splits into a power of two runs of that length, or
/// slightly fewer, which merge in balanced pairs.
fn min_run_length(mut n: usize) -> usize {
    let mut rest = 0;
    while n >= MIN_MERGE {
        rest |= n & 1;
        n >>= 1;
    }
    n + rest
}

/// The factor that maps a sum of two positions in a slice of `len` elements onto a fraction
/// of `2^63`, for [`node_power`].
pub(super) fn power_scale(len: usize) -> u64 {
    (1u64 << 62).div_ceil(len as u64)
}

/// Returns the power of the boundary between the adjacent runs `start..mid` and `mid..end`:
/// how many leading bits the positions of the two runs' midpoints, as fractions of the slice,
/// have in common. It is the depth of the boundary in a perfectly balanced merge tree over the
/// whole slice; runs are merged bottom-up in that order (Munro and Wild's powersort).
pub(super) fn node_power(scale: u64, start: usize, mid: usize, end: usize) -> u32 {
    // Twice the midpoints, below `2·len`, so the sums do not overflow and neither do the
    // products, which stay below `2^63 + 2·len`. They differ, since `start < end` and
    // `scale >= 1`.
    let left = (start + mid) as u64 * scale;
    let right = (mid + end) as u64 * scale;
    (left ^ right).leading_zeros()
}

/// Merges the top two runs of the stack into one.
fn merge_top<T: Ord>(
    arr: &mut [T],
    runs: &mut RunStack,
    buf: &mut Buffer<T>,
    min_gallop: &mut usize,
) {
    let (Some(left), Some(right)) = (runs.top(1), runs.top(0)) else {
        return;
    };
    let v = &mut arr[left.start..right.start + right.len];
    merge(v, left.len, buf, min_gallop);
    runs.len -= 1;
    runs.runs[runs.len - 1].write(Run {
        len: left.len + right.len,
        ..left
    });
}

/// Returns the length of the run at the start of `v`: its longest prefix that is ascending, or
/// strictly descending, in which case it is reversed.
fn find_run<T: Ord>(v: &mut [T]) -> usize {
    let len = v.len();
    if len < 2 {
        return len;
    }
    // Reversing a strictly descending run cannot reorder equal elements.
    let mut end = 2;
    if v[1] < v[0] {
        while end < len && v[end] < v[end - 1] {
            end += 1;
        }
        v[..end].reverse();
    } else {
        while end < len && v[end] >= v[end - 1] {
            end += 1;
        }
    }
    end
}

/// Sorts `v`, at most `MIN_MERGE` elements whose first `sorted` are already sorted, into a run.
pub(super) fn build_run<T: Ord>(v: &mut [T], sorted: usize, buf: &mut Buffer<T>) {
    if mem::size_of::<T>() <= SMALL_SORT_MAX_SIZE && sorted < SMALL_SORT_RUN {
        // The stack buffer has room for `STACK_BYTES / 8 >= MIN_MERGE` such elements.
        super::merge::sort(v, buf.on_stack());
    } else {
        extend_run(v, sorted);
    }
}

/// The largest element size, in bytes, for which a run is built by merge sort's branchless
/// short-run sort when it starts with fewer than `SMALL_SORT_RUN` sorted elements; larger
/// elements and longer starts are extended by [`extend_run`].
///
/// Measured for integers: 1.1–1.3 times faster on random input and 1.1–1.2 times with few
/// distinct keys, where every insertion ends in a mispredicted branch; up to 1.25 times slower
/// on short slices that are nearly sorted in small steps, where insertion barely moves
/// anything. `&str`, as large as a `u128` but slower to compare, sorts 1.05–1.15 times slower
/// with it on such input and few distinct keys, so 16-byte elements do without it.
const SMALL_SORT_MAX_SIZE: usize = 8;

/// See [`SMALL_SORT_MAX_SIZE`]. Measured against 8 and 16: the same on random input, and up to
/// 1.5 times faster on short nearly sorted slices.
const SMALL_SORT_RUN: usize = 4;

/// Sorts `v`, whose first `sorted` elements are already sorted, by inserting the others into
/// that prefix two at a time.
pub(super) fn extend_run<T: Ord>(v: &mut [T], sorted: usize) {
    let len = v.len();
    let mut i = sorted.max(1);
    while i + 1 < len {
        insert_pair(&mut v[..i + 2]);
        i += 2;
    }
    if i < len {
        insert_last(v);
    }
}

/// Inserts the last two elements of `v` into the sorted rest of it, which is not empty.
///
/// The larger of the two is carried left past the greater elements of the prefix, which move
/// two slots right, and the smaller one carries on from there, the elements it passes moving
/// one slot: each prefix element moves once, however many of the two pass it.
fn insert_pair<T: Ord>(v: &mut [T]) {
    let i = v.len() - 2;
    // Only a second element strictly smaller than the first counts as out of order, so equal
    // elements keep their order.
    let swapped = v[i + 1] < v[i];
    let (s, l) = if swapped { (i + 1, i) } else { (i, i + 1) };
    if v[s] >= v[i - 1] {
        // Neither goes before the last element of the prefix.
        if swapped {
            v.swap(i, i + 1);
        }
        return;
    }
    let base = v.as_mut_ptr();
    // SAFETY: every offset is at most `i + 1`, within `v`. Both elements are read out, leaving
    // `v[i]` and `v[i + 1]` vacant, and `hole` names the vacant slots: `slot` and, while
    // `large` is out, `slot + 1`. Each shift moves the element just below the vacant slots into
    // the higher of them, which makes its own slot the lower one. If a comparison panics,
    // `hole` is dropped and writes the elements read out into the vacant slots, so `v` holds
    // every element exactly once whenever the block is left.
    unsafe {
        let large = ManuallyDrop::new(ptr::read(base.add(l)));
        let small = ManuallyDrop::new(ptr::read(base.add(s)));
        let mut hole = PairHole {
            small: &*small,
            large: &*large,
            slot: base.add(i),
            large_out: true,
        };
        let mut j = i;
        while j > 0 && *large < *base.add(j - 1) {
            ptr::copy_nonoverlapping(base.add(j - 1), base.add(j + 1), 1);
            j -= 1;
            hole.slot = base.add(j);
        }
        ptr::copy_nonoverlapping(&*large, base.add(j + 1), 1);
        hole.large_out = false;
        if j == i {
            // `small < v[i - 1]` is known from the check above.
            ptr::copy_nonoverlapping(base.add(i - 1), base.add(i), 1);
            j -= 1;
            hole.slot = base.add(j);
        }
        while j > 0 && *small < *base.add(j - 1) {
            ptr::copy_nonoverlapping(base.add(j - 1), base.add(j), 1);
            j -= 1;
            hole.slot = base.add(j);
        }
        // Dropping `hole` writes `small` into the last vacant slot.
    }
}

/// Inserts the last element of `v` into the sorted rest of it.
fn insert_last<T: Ord>(v: &mut [T]) {
    let i = v.len() - 1;
    if i == 0 || v[i] >= v[i - 1] {
        return;
    }
    let base = v.as_mut_ptr();
    // SAFETY: as in `insert_pair`, with one element read out and one vacant slot.
    unsafe {
        let last = ManuallyDrop::new(ptr::read(base.add(i)));
        let mut hole = PairHole {
            small: &*last,
            large: &*last,
            slot: base.add(i - 1),
            large_out: false,
        };
        ptr::copy_nonoverlapping(base.add(i - 1), base.add(i), 1);
        let mut j = i - 1;
        while j > 0 && *last < *base.add(j - 1) {
            ptr::copy_nonoverlapping(base.add(j - 1), base.add(j), 1);
            j -= 1;
            hole.slot = base.add(j);
        }
    }
}

/// The elements read out of a slice during an insertion and the vacant slots they go into:
/// dropping it, when the insertion is done or a comparison panics, writes them back.
struct PairHole<T> {
    small: *const T,
    large: *const T,
    slot: *mut T,
    large_out: bool,
}

impl<T> Drop for PairHole<T> {
    fn drop(&mut self) {
        // SAFETY: `slot`, and `slot + 1` if `large_out`, are vacant slots of the slice, and
        // `small` and `large` point to the elements read out of it.
        unsafe {
            ptr::copy_nonoverlapping(self.small, self.slot, 1);
            if self.large_out {
                ptr::copy_nonoverlapping(self.large, self.slot.add(1), 1);
            }
        }
    }
}

/// Returns how many elements at the start of `v` satisfy `before`, which holds for a prefix of
/// `v` and for none of the rest. Probes 1, 2, 4, ... elements from the start until `before`
/// fails, then binary searches the last gap: an answer `k` costs about `2·log₂(k + 1) + 1`
/// calls. The result never exceeds `v.len()`, whatever `before` answers.
#[inline]
fn gallop_from_start<T>(v: &[T], before: impl Fn(&T) -> bool) -> usize {
    let n = v.len();
    // The answer lies in `lo..=hi`.
    let (mut lo, mut hi) = (0, n);
    let mut probe = 0;
    while probe < n {
        if before(&v[probe]) {
            lo = probe + 1;
            probe = 2 * probe + 1;
        } else {
            hi = probe;
            break;
        }
    }
    binary_search(v, before, lo, hi)
}

/// Like [`gallop_from_start`], but probes 1, 2, 4, ... elements from the end: an answer `k`
/// costs about `2·log₂(n − k + 1) + 1` calls.
#[inline]
fn gallop_from_end<T>(v: &[T], before: impl Fn(&T) -> bool) -> usize {
    let n = v.len();
    let (mut lo, mut hi) = (0, n);
    let mut step = 1;
    while step <= n {
        let probe = n - step;
        if before(&v[probe]) {
            lo = probe + 1;
            break;
        }
        hi = probe;
        step *= 2;
    }
    binary_search(v, before, lo, hi)
}

/// Finds where `before` starts to fail in `v[lo..hi]`, given that it holds for `v[..lo]` and
/// fails for `v[hi..]`.
#[inline]
fn binary_search<T>(v: &[T], before: impl Fn(&T) -> bool, mut lo: usize, mut hi: usize) -> usize {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if before(&v[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The largest element size, in bytes, for which a merge advances its pointers by selecting
/// between the pointer and the one after it (a conditional select) instead of adding the
/// condition to it (a flag set and an add): one instruction less on the chain from each
/// comparison to the next load, which is what bounds the merge. For larger elements the
/// compiler made worse code of it: strings 1.07–1.1 times and 64-byte elements 1.04–1.4
/// times slower on input made of runs.
const SELECT_ADVANCE_MAX_SIZE: usize = 8;

/// The largest element size, in bytes, for which a merge from both ends checks for a run that
/// ran out once per batch of steps instead of at every step. Measured: 1–2% faster on input
/// made of runs for integers (with the select above, the loop shrinks from 51 to 30
/// instructions per step of two elements); strings and 16-byte boxed strings, whose
/// comparisons are calls, 1.03–1.08 times slower: with the batch's end to keep as well, the
/// compiler spilled the pointer on the chain from one comparison to the next load in a way
/// the processor could not forward.
const BATCH_MAX_SIZE: usize = 8;

/// Returns `p.add(1)` if `cond` and `p` otherwise: see [`SELECT_ADVANCE_MAX_SIZE`].
///
/// # Safety
///
/// `p.add(1)` must be in bounds of the allocation `p` points into (one past the end is fine).
#[inline(always)]
unsafe fn advance<T>(p: *const T, cond: bool) -> *const T {
    // SAFETY: as the caller guarantees.
    unsafe {
        if mem::size_of::<T>() <= SELECT_ADVANCE_MAX_SIZE {
            hint::select_unpredictable(cond, p.add(1), p)
        } else {
            p.add(usize::from(cond))
        }
    }
}

/// Returns `p.sub(1)` if `cond` and `p` otherwise: see [`SELECT_ADVANCE_MAX_SIZE`].
///
/// # Safety
///
/// `p.sub(1)` must be in bounds of the allocation `p` points into.
#[inline(always)]
unsafe fn retreat<T>(p: *const T, cond: bool) -> *const T {
    // SAFETY: as the caller guarantees.
    unsafe {
        if mem::size_of::<T>() <= SELECT_ADVANCE_MAX_SIZE {
            hint::select_unpredictable(cond, p.sub(1), p)
        } else {
            p.sub(usize::from(cond))
        }
    }
}

/// Merges the sorted runs `v[..mid]` and `v[mid..]`, where `0 < mid < v.len()`.
///
/// `min_gallop` is the streak that sets off galloping, adapted by every merge: it drops while
/// galloping pays off and rises when it does not, and once above [`MIN_GALLOP`] merges of
/// runs that are neither lopsided nor short run as two merges at once (see [`Merge::split`]
/// and [`FOUR_CHAIN_MIN`]).
pub(super) fn merge<T: Ord>(v: &mut [T], mid: usize, buf: &mut Buffer<T>, min_gallop: &mut usize) {
    // The elements of the left run that are not greater than the first of the right run are
    // already in place, and so are those of the right run not smaller than the last of the
    // left run. Both searches compare elements in place, so a panic leaves `v` untouched.
    let first = &v[mid];
    let skip = gallop_from_start(&v[..mid], |x| x <= first);
    if skip == mid {
        return;
    }
    let v = &mut v[skip..];
    let mid = mid - skip;
    let last = &v[mid - 1];
    let keep = gallop_from_end(&v[mid..], |x| x < last);
    if keep == 0 {
        // Only an `Ord` that is not a total order gets here.
        return;
    }
    let v = &mut v[..mid + keep];
    let buf = buf.get();
    // SAFETY: `buf` has room for the whole slice being sorted, which `v` is part of, and
    // does not overlap it; `0 < mid < v.len()`.
    unsafe {
        if mid.min(keep) * LOPSIDED < mid.max(keep) {
            if mid <= keep {
                merge_low(v, mid, buf, min_gallop);
            } else {
                merge_high(v, mid, buf, min_gallop);
            }
        } else {
            merge_through_buffer(v, mid, buf, min_gallop);
        }
    }
}

/// Merges the sorted runs `v[..mid]` and `v[mid..]` through `buf`, from both ends at once,
/// galloping when one run keeps winning, and as two merges at once while neither does (see
/// [`Merge::split`]). The first element of the right run should be smaller than the first of
/// the left run, and the last of the left run greater than the last of the right run: they
/// are placed first and last without comparisons.
///
/// # Safety
///
/// `0 < mid < v.len()`, and `buf` is valid for writes and reads of `v.len()` elements and does
/// not overlap `v`.
unsafe fn merge_through_buffer<T: Ord>(
    v: &mut [T],
    mid: usize,
    buf: *mut T,
    min_gallop: &mut usize,
) {
    let len = v.len();
    let dst = v.as_mut_ptr();
    // SAFETY: `v` is copied out whole into `buf`, which then holds every element exactly once
    // while the merge writes bitwise copies of them into `v`; `restore` copies `buf` back over
    // `v` if a comparison panics, which loses nothing because comparisons read only `buf`,
    // which is never written here, and only elements whose copy is not written yet (see
    // [`Merge`]), so `buf` holds the latest state of every element. The two placements below
    // take the first element of the right run and the last of the left one, which exist
    // since `0 < mid < len`, and leave `m` with as many slots as elements.
    unsafe {
        ptr::copy_nonoverlapping(dst, buf, len);
        let restore = Restore { src: buf, dst, len };
        let a = buf.cast_const();
        let mut m = Merge {
            a,
            a_end: a.add(mid),
            b: a.add(mid),
            b_end: a.add(len),
            out: dst,
            out_end: dst.add(len),
        };
        ptr::copy_nonoverlapping(m.b, m.out, 1);
        m.b = m.b.add(1);
        m.out = m.out.add(1);
        m.a_end = m.a_end.sub(1);
        m.out_end = m.out_end.sub(1);
        ptr::copy_nonoverlapping(m.a_end, m.out_end, 1);

        let mut gallop = *min_gallop;
        if gallop > MIN_GALLOP && m.rest_a() >= FOUR_CHAIN_MIN && m.rest_b() >= FOUR_CHAIN_MIN {
            // Galloping has failed to pay off in this sort: merge as two merges at once
            // until an end takes from the same run `gallop` times in a row, then gallop.
            let (mut lower, mut upper) = m.split();
            loop {
                // Both halves take `n` steps, with no run running out: a batch. Whether one
                // of their ends took from the same run throughout is checked at its end, so
                // the batch is at most `gallop` steps, the streak that sets off galloping.
                let n = lower.safe().min(upper.safe());
                if n == 0 {
                    break;
                }
                let n = n.min(gallop);
                let (mut lower_b, mut lower_a, mut upper_b, mut upper_a) = (0usize, 0, 0, 0);
                for _ in 0..n {
                    let (take_b, take_a) = lower.step();
                    lower_b += usize::from(take_b);
                    lower_a += usize::from(take_a);
                    let (take_b, take_a) = upper.step();
                    upper_b += usize::from(take_b);
                    upper_a += usize::from(take_a);
                }
                if n == gallop
                    && (streak(lower_b, n)
                        || streak(lower_a, n)
                        || streak(upper_b, n)
                        || streak(upper_a, n))
                {
                    break;
                }
            }
            // The galloping merge finishes both halves, the lower one first so that
            // `gallop` adapts in order.
            for half in [lower, upper] {
                half.finish(&mut gallop);
            }
        } else {
            m.finish(&mut gallop);
        }
        *min_gallop = gallop;
        mem::forget(restore);
    }
}

/// Whether an end that took `count` elements from one run in `n` steps took every element of
/// the batch from the same run.
#[inline(always)]
fn streak(count: usize, n: usize) -> bool {
    count == 0 || count == n
}

/// Merges whose runs both have at least this many elements left after the two known
/// placements are merged as two merges at once (see [`Merge::split`]), once galloping has
/// failed to pay off in an earlier merge of the same sort (`min_gallop` above [`MIN_GALLOP`]).
/// Splitting every merge long enough instead costs a binary search per merge for nothing on
/// input whose runs interleave in blocks, where the split is lopsided and galloping does the
/// work: 1.03–1.08 times slower on such input and on nearly sorted input, and no faster on
/// input made of runs, where galloping fails in the first merges anyway. 16 and 64 measured
/// the same as 32.
const FOUR_CHAIN_MIN: usize = 32;

/// A merge of two runs copied out into a buffer, in progress: what is left of the runs,
/// `a..a_end` and `b..b_end`, and the slots of the slice not yet written, `out..out_end`,
/// exactly as many as elements are left.
///
/// Every step takes an element from the start or the end of what is left of a run and writes
/// it to the matching end of the slots, so every element is written exactly once, to a slot of
/// its own, whatever `Ord` answers; and it reads only elements that are left, so no element is
/// compared after a copy of it was written. `a <= a_end` and `b <= b_end` always hold.
struct Merge<T> {
    a: *const T,
    a_end: *const T,
    b: *const T,
    b_end: *const T,
    out: *mut T,
    out_end: *mut T,
}

impl<T: Ord> Merge<T> {
    /// How many elements of the left run are left.
    #[inline(always)]
    unsafe fn rest_a(&self) -> usize {
        // SAFETY: `a..a_end` is a range of the buffer.
        unsafe { self.a_end.offset_from_unsigned(self.a) }
    }

    /// How many elements of the right run are left.
    #[inline(always)]
    unsafe fn rest_b(&self) -> usize {
        // SAFETY: `b..b_end` is a range of the buffer.
        unsafe { self.b_end.offset_from_unsigned(self.b) }
    }

    /// How many steps can be taken in a row without checking for a run that ran out: a step
    /// reads two elements of each run and takes two, both possibly from the same run, so
    /// `min(rest_a, rest_b) / 2` steps never read past a run.
    #[inline(always)]
    unsafe fn safe(&self) -> usize {
        // SAFETY: as for `rest_a` and `rest_b`.
        unsafe { self.rest_a().min(self.rest_b()) / 2 }
    }

    /// Takes one element at the front and one at the back, and returns whether the front took
    /// from the right run and whether the back took from the left run.
    ///
    /// # Safety
    ///
    /// Both runs must have at least two elements left.
    #[inline(always)]
    unsafe fn step(&mut self) -> (bool, bool) {
        // SAFETY: both runs have at least two elements left, so the four elements read are
        // left, the front and the back take distinct ones, and the pointers they advance
        // stay in bounds. Exactly one run advances at each end, whatever `Ord` answers, so as
        // many elements as slots are left afterwards.
        unsafe {
            // Front: the right element goes first only if it is strictly smaller.
            let take_b = *self.b < *self.a;
            let src = hint::select_unpredictable(take_b, self.b, self.a);
            ptr::copy_nonoverlapping(src, self.out, 1);
            self.out = self.out.add(1);
            self.b = advance(self.b, take_b);
            self.a = advance(self.a, !take_b);

            // Back: the left element goes last only if it is strictly greater.
            let (last_a, last_b) = (self.a_end.sub(1), self.b_end.sub(1));
            let take_a = *last_b < *last_a;
            self.out_end = self.out_end.sub(1);
            let src = hint::select_unpredictable(take_a, last_a, last_b);
            ptr::copy_nonoverlapping(src, self.out_end, 1);
            self.a_end = retreat(self.a_end, take_a);
            self.b_end = retreat(self.b_end, !take_a);
            (take_b, take_a)
        }
    }

    /// Splits the merge in two at the middle of its result: a binary search finds how many
    /// elements of each run belong to the lower half of the merged result (co-ranking), and
    /// the two halves are independent merges, each from both ends, so that four chains of
    /// comparisons overlap instead of two. The merge is bound by the chain from each
    /// comparison to the next load, not by instructions: measured on input made of runs of
    /// random content, two merges at once are 1.25–1.4 times faster for integers and strings
    /// and 1.1 times for 64-byte elements.
    ///
    /// The `k = (rest_a + rest_b) / 2` smallest elements, in the stable order, are `a[..i]`
    /// and `b[..k - i]`, where `a[m]` is among them exactly if at most `k - m - 1` elements of
    /// the right run are smaller than it, that is if `b[k - m - 1] >= a[m]`. The search runs
    /// over `k - rest_b..=min(k, rest_a)`, where both indices exist. The split keeps the merge
    /// stable: `a[i - 1] <= b[k - i]` and `b[k - i - 1] < a[i]` where those exist, so nothing
    /// in the lower half goes after anything in the upper one.
    ///
    /// # Safety
    ///
    /// The invariants of `Merge` must hold.
    unsafe fn split(self) -> (Self, Self) {
        // SAFETY: `m < hi <= min(k, rest_a)` and `m >= lo >= k - rest_b` keep `a[m]` and
        // `b[k - m - 1]` within what is left of the runs, whatever `Ord` answers, and
        // `i <= rest_a` and `k - i <= rest_b` keep both halves within them, each with as many
        // slots as elements.
        unsafe {
            let (rest_a, rest_b) = (self.rest_a(), self.rest_b());
            let k = (rest_a + rest_b) / 2;
            let (mut lo, mut hi) = (k.saturating_sub(rest_b), k.min(rest_a));
            while lo < hi {
                let m = lo + (hi - lo) / 2;
                if *self.b.add(k - m - 1) >= *self.a.add(m) {
                    lo = m + 1;
                } else {
                    hi = m;
                }
            }
            let (i, j) = (lo, k - lo);
            let lower = Merge {
                a: self.a,
                a_end: self.a.add(i),
                b: self.b,
                b_end: self.b.add(j),
                out: self.out,
                out_end: self.out.add(k),
            };
            let upper = Merge {
                a: self.a.add(i),
                a_end: self.a_end,
                b: self.b.add(j),
                b_end: self.b_end,
                out: self.out.add(k),
                out_end: self.out_end,
            };
            (lower, upper)
        }
    }

    /// Finishes the merge from both ends at once, galloping when one run keeps winning:
    /// `min_gallop` is the streak that sets off galloping, adapted on the way.
    ///
    /// # Safety
    ///
    /// The invariants of `Merge` must hold.
    #[inline(never)]
    unsafe fn finish(self, min_gallop: &mut usize) {
        let mut m = self;
        // SAFETY: every step takes an element from the start or the end of what is left of a
        // run and writes it to the matching end of `out..out_end`, so every element is
        // written exactly once, to a slot of its own, whatever `Ord` answers. A step reads
        // only elements that are left: the single-element steps run only while each run has
        // at least two elements left (in batches of `safe()` for small elements, checked at
        // every step otherwise), galloping starts only while both runs have elements left
        // and is bounded by what is left, and every other step checks for an empty run
        // first.
        unsafe {
            let mut gallop = *min_gallop;
            'merge: loop {
                // Consecutive wins of each run at the front and at the back.
                let (mut front_a, mut front_b, mut back_a, mut back_b) = (0usize, 0, 0, 0);
                loop {
                    if mem::size_of::<T>() <= BATCH_MAX_SIZE {
                        // `safe()` steps in a row never read past a run: they make a batch,
                        // checked for a run that ran out only at its end, and for a streak at
                        // every step.
                        let safe = m.safe();
                        if safe == 0 {
                            break 'merge;
                        }
                        let batch_end = m.out.add(safe);
                        loop {
                            let (take_b, take_a) = m.step();
                            front_b = (front_b + 1) * usize::from(take_b);
                            front_a = (front_a + 1) * usize::from(!take_b);
                            back_a = (back_a + 1) * usize::from(take_a);
                            back_b = (back_b + 1) * usize::from(!take_a);
                            if (front_a | front_b | back_a | back_b) >= gallop || m.out == batch_end
                            {
                                break;
                            }
                        }
                    } else {
                        // Larger elements check at every step: with a batch to keep track of
                        // as well, the compiler made worse code of the loop (see
                        // `BATCH_MAX_SIZE`).
                        if m.rest_a() < 2 || m.rest_b() < 2 {
                            break 'merge;
                        }
                        let (take_b, take_a) = m.step();
                        front_b = (front_b + 1) * usize::from(take_b);
                        front_a = (front_a + 1) * usize::from(!take_b);
                        back_a = (back_a + 1) * usize::from(take_a);
                        back_b = (back_b + 1) * usize::from(!take_a);
                    }
                    if (front_a | front_b | back_a | back_b) < gallop {
                        // No streak yet.
                        continue;
                    }
                    // A run may have run out in the last step, if both ends took from it.
                    if m.a == m.a_end || m.b == m.b_end {
                        break 'merge;
                    }
                    if (front_a | front_b) >= gallop {
                        break;
                    }
                    // A streak at the back: alternately move the elements of the left run that
                    // are greater than the last of the right run, and those of the right run
                    // that are not smaller than the last of the left run, as blocks.
                    loop {
                        gallop -= usize::from(gallop > 1);
                        let last_b = m.b_end.sub(1);
                        let rest = slice::from_raw_parts(m.a, m.a_end.offset_from_unsigned(m.a));
                        let count_a = rest.len() - gallop_from_end(rest, |x| x <= &*last_b);
                        m.a_end = m.a_end.sub(count_a);
                        m.out_end = m.out_end.sub(count_a);
                        ptr::copy_nonoverlapping(m.a_end, m.out_end, count_a);
                        if m.a == m.a_end {
                            break 'merge;
                        }
                        m.b_end = m.b_end.sub(1);
                        m.out_end = m.out_end.sub(1);
                        ptr::copy_nonoverlapping(m.b_end, m.out_end, 1);
                        if m.b == m.b_end {
                            break 'merge;
                        }

                        let last_a = m.a_end.sub(1);
                        let rest = slice::from_raw_parts(m.b, m.b_end.offset_from_unsigned(m.b));
                        let count_b = rest.len() - gallop_from_end(rest, |x| x < &*last_a);
                        m.b_end = m.b_end.sub(count_b);
                        m.out_end = m.out_end.sub(count_b);
                        ptr::copy_nonoverlapping(m.b_end, m.out_end, count_b);
                        if m.b == m.b_end {
                            break 'merge;
                        }
                        m.a_end = m.a_end.sub(1);
                        m.out_end = m.out_end.sub(1);
                        ptr::copy_nonoverlapping(m.a_end, m.out_end, 1);
                        if m.a == m.a_end {
                            break 'merge;
                        }
                        if count_a < MIN_GALLOP && count_b < MIN_GALLOP {
                            break;
                        }
                    }
                    gallop += 2;
                    continue 'merge;
                }
                // Gallop at the front: alternately move the elements of the left run that are not
                // greater than the first of the right run, and those of the right run that are
                // smaller than the first of the left run, as blocks.
                loop {
                    gallop -= usize::from(gallop > 1);
                    let first_b = m.b;
                    let rest = slice::from_raw_parts(m.a, m.a_end.offset_from_unsigned(m.a));
                    let count_a = gallop_from_start(rest, |x| x <= &*first_b);
                    ptr::copy_nonoverlapping(m.a, m.out, count_a);
                    m.a = m.a.add(count_a);
                    m.out = m.out.add(count_a);
                    if m.a == m.a_end {
                        break 'merge;
                    }
                    ptr::copy_nonoverlapping(m.b, m.out, 1);
                    m.b = m.b.add(1);
                    m.out = m.out.add(1);
                    if m.b == m.b_end {
                        break 'merge;
                    }

                    let first_a = m.a;
                    let rest = slice::from_raw_parts(m.b, m.b_end.offset_from_unsigned(m.b));
                    let count_b = gallop_from_start(rest, |x| x < &*first_a);
                    ptr::copy_nonoverlapping(m.b, m.out, count_b);
                    m.b = m.b.add(count_b);
                    m.out = m.out.add(count_b);
                    if m.b == m.b_end {
                        break 'merge;
                    }
                    ptr::copy_nonoverlapping(m.a, m.out, 1);
                    m.a = m.a.add(1);
                    m.out = m.out.add(1);
                    if m.a == m.a_end {
                        break 'merge;
                    }
                    if count_a < MIN_GALLOP && count_b < MIN_GALLOP {
                        break;
                    }
                }
                gallop += 2;
            }
            *min_gallop = gallop;

            // A run is empty, or has one element left, which a binary search places among what is
            // left of the other one.
            let rest_a = m.a_end.offset_from_unsigned(m.a);
            let rest_b = m.b_end.offset_from_unsigned(m.b);
            if rest_a == 1 && rest_b > 0 {
                let rest = slice::from_raw_parts(m.b, rest_b);
                let k = binary_search(rest, |x| x < &*m.a, 0, rest_b);
                ptr::copy_nonoverlapping(m.b, m.out, k);
                ptr::copy_nonoverlapping(m.a, m.out.add(k), 1);
                ptr::copy_nonoverlapping(m.b.add(k), m.out.add(k + 1), rest_b - k);
            } else if rest_b == 1 && rest_a > 0 {
                let rest = slice::from_raw_parts(m.a, rest_a);
                let k = binary_search(rest, |x| x <= &*m.b, 0, rest_a);
                ptr::copy_nonoverlapping(m.a, m.out, k);
                ptr::copy_nonoverlapping(m.b, m.out.add(k), 1);
                ptr::copy_nonoverlapping(m.a.add(k), m.out.add(k + 1), rest_a - k);
            } else {
                ptr::copy_nonoverlapping(m.a, m.out, rest_a);
                ptr::copy_nonoverlapping(m.b, m.out.add(rest_a), rest_b);
            }
        }
    }
}

/// Merges the sorted runs `v[..mid]` and `v[mid..]`, the left one no longer than the right one,
/// by copying the left run into `buf` and merging from the front, galloping when one run keeps
/// winning. The first element of the right run should be smaller than the first of the left
/// run, and the last of the left run greater than the last of the right run: the first is
/// placed first without a comparison.
///
/// # Safety
///
/// `0 < mid < v.len()`, and `buf` is valid for writes and reads of `mid` elements and does not
/// overlap `v`.
unsafe fn merge_low<T: Ord>(v: &mut [T], mid: usize, buf: *mut T, min_gallop: &mut usize) {
    let len = v.len();
    let base = v.as_mut_ptr();
    // SAFETY: the left run is copied out into `buf`. What is left of it is `gap.src..gap.end`,
    // and what is left of the right run `b..b_end`, still in `v`. The slots of `v` not yet
    // written are `gap.dst..b`, exactly as many as are left of the left run: every step writes
    // the element it takes to `gap.dst` and advances it, so every element is written exactly
    // once, to a slot of its own, whatever `Ord` answers, and an element of the right run is
    // never overwritten before it is taken. If a comparison panics, dropping `gap` copies the
    // rest of the left run into the vacant slots. A step reads only elements that are left:
    // every step checks for an empty run first, and galloping is bounded by what is left.
    unsafe {
        ptr::copy_nonoverlapping(base, buf, mid);
        let mut gap = Gap {
            src: buf,
            end: buf.add(mid),
            dst: base,
        };
        let mut b = base.add(mid).cast_const();
        let b_end = base.add(len).cast_const();

        ptr::copy_nonoverlapping(b, gap.dst, 1);
        b = b.add(1);
        gap.dst = gap.dst.add(1);
        if b == b_end {
            return;
        }

        let mut gallop = *min_gallop;
        'merge: loop {
            // Consecutive wins of each run.
            let (mut wins_a, mut wins_b) = (0usize, 0);
            loop {
                // The right element goes first only if it is strictly smaller.
                let take_b = *b < *gap.src;
                let src = hint::select_unpredictable(take_b, b, gap.src.cast_const());
                ptr::copy_nonoverlapping(src, gap.dst, 1);
                gap.dst = gap.dst.add(1);
                b = advance(b, take_b);
                gap.src = advance(gap.src, !take_b).cast_mut();
                wins_b = (wins_b + 1) * usize::from(take_b);
                wins_a = (wins_a + 1) * usize::from(!take_b);
                if gap.src == gap.end || b == b_end {
                    break 'merge;
                }
                if (wins_a | wins_b) >= gallop {
                    break;
                }
            }
            loop {
                gallop -= usize::from(gallop > 1);
                let rest = slice::from_raw_parts(gap.src, gap.end.offset_from_unsigned(gap.src));
                let count_a = gallop_from_start(rest, |x| x <= &*b);
                ptr::copy_nonoverlapping(gap.src, gap.dst, count_a);
                gap.src = gap.src.add(count_a);
                gap.dst = gap.dst.add(count_a);
                if gap.src == gap.end {
                    break 'merge;
                }
                ptr::copy_nonoverlapping(b, gap.dst, 1);
                b = b.add(1);
                gap.dst = gap.dst.add(1);
                if b == b_end {
                    break 'merge;
                }

                let first_a = gap.src;
                let rest = slice::from_raw_parts(b, b_end.offset_from_unsigned(b));
                let count_b = gallop_from_start(rest, |x| x < &*first_a);
                // The vacant slots lie just before `b`: the block may overlap them.
                ptr::copy(b, gap.dst, count_b);
                b = b.add(count_b);
                gap.dst = gap.dst.add(count_b);
                if b == b_end {
                    break 'merge;
                }
                ptr::copy_nonoverlapping(gap.src, gap.dst, 1);
                gap.src = gap.src.add(1);
                gap.dst = gap.dst.add(1);
                if gap.src == gap.end {
                    break 'merge;
                }
                if count_a < MIN_GALLOP && count_b < MIN_GALLOP {
                    break;
                }
            }
            gallop += 2;
        }
        *min_gallop = gallop;
        // Dropping `gap` copies what is left of the left run into the vacant slots.
    }
}

/// Merges the sorted runs `v[..mid]` and `v[mid..]`, the right one shorter than the left one,
/// by copying the right run into `buf` and merging from the back: the mirror image of
/// [`merge_low`]. The last element of the left run is placed last without a comparison.
///
/// # Safety
///
/// `0 < mid < v.len()`, and `buf` is valid for writes and reads of `v.len() - mid` elements and
/// does not overlap `v`.
unsafe fn merge_high<T: Ord>(v: &mut [T], mid: usize, buf: *mut T, min_gallop: &mut usize) {
    let len = v.len();
    let base = v.as_mut_ptr();
    // SAFETY: as in `merge_low`, mirrored. What is left of the right run is `gap.src..gap.end`
    // in `buf`, and what is left of the left run `base..gap.dst`, still in `v`. The slots of
    // `v` not yet written are `gap.dst..out`, as many as are left of the right run: every step
    // moves `out` down onto the slot it writes.
    unsafe {
        ptr::copy_nonoverlapping(base.add(mid), buf, len - mid);
        let mut gap = Gap {
            src: buf,
            end: buf.add(len - mid),
            dst: base.add(mid),
        };
        let mut out = base.add(len);

        gap.dst = gap.dst.sub(1);
        out = out.sub(1);
        ptr::copy_nonoverlapping(gap.dst, out, 1);
        if gap.dst == base {
            return;
        }

        let mut gallop = *min_gallop;
        'merge: loop {
            let (mut wins_a, mut wins_b) = (0usize, 0);
            loop {
                // The left element goes last only if it is strictly greater.
                let (last_a, last_b) = (gap.dst.sub(1), gap.end.sub(1));
                let take_a = *last_b < *last_a;
                let src = hint::select_unpredictable(take_a, last_a, last_b);
                out = out.sub(1);
                ptr::copy_nonoverlapping(src, out, 1);
                gap.dst = retreat(gap.dst, take_a).cast_mut();
                gap.end = retreat(gap.end, !take_a).cast_mut();
                wins_a = (wins_a + 1) * usize::from(take_a);
                wins_b = (wins_b + 1) * usize::from(!take_a);
                if gap.src == gap.end || gap.dst == base {
                    break 'merge;
                }
                if (wins_a | wins_b) >= gallop {
                    break;
                }
            }
            loop {
                gallop -= usize::from(gallop > 1);
                let last_b = gap.end.sub(1);
                let rest = slice::from_raw_parts(base, gap.dst.offset_from_unsigned(base));
                let count_a = rest.len() - gallop_from_end(rest, |x| x <= &*last_b);
                gap.dst = gap.dst.sub(count_a);
                out = out.sub(count_a);
                // The vacant slots lie just after the left run: the block may overlap them.
                ptr::copy(gap.dst, out, count_a);
                if gap.dst == base {
                    break 'merge;
                }
                gap.end = gap.end.sub(1);
                out = out.sub(1);
                ptr::copy_nonoverlapping(gap.end, out, 1);
                if gap.src == gap.end {
                    break 'merge;
                }

                let last_a = gap.dst.sub(1);
                let rest = slice::from_raw_parts(gap.src, gap.end.offset_from_unsigned(gap.src));
                let count_b = rest.len() - gallop_from_end(rest, |x| x < &*last_a);
                gap.end = gap.end.sub(count_b);
                out = out.sub(count_b);
                ptr::copy_nonoverlapping(gap.end, out, count_b);
                if gap.src == gap.end {
                    break 'merge;
                }
                gap.dst = gap.dst.sub(1);
                out = out.sub(1);
                ptr::copy_nonoverlapping(gap.dst, out, 1);
                if gap.dst == base {
                    break 'merge;
                }
                if count_a < MIN_GALLOP && count_b < MIN_GALLOP {
                    break;
                }
            }
            gallop += 2;
        }
        *min_gallop = gallop;
        // Dropping `gap` copies what is left of the right run into the vacant slots.
    }
}

/// What is left of a run copied out of a slice, `src..end`, and where the slots of the slice
/// that are still vacant start. Dropping it, when the merge is done or a comparison panics,
/// copies that rest into those slots.
struct Gap<T> {
    src: *mut T,
    end: *mut T,
    dst: *mut T,
}

impl<T> Drop for Gap<T> {
    fn drop(&mut self) {
        // SAFETY: `src..end` holds the elements not yet merged, which do not overlap the slice,
        // and the slice has exactly that many vacant slots from `dst` on.
        unsafe {
            let len = self.end.offset_from_unsigned(self.src);
            ptr::copy_nonoverlapping(self.src, self.dst, len);
        }
    }
}

/// A copy of a slice that is written back over it when dropped, unless forgotten.
///
/// Dropping it while unwinding from a panicking comparison is what keeps
/// [`merge_through_buffer`] panic-safe: the slice gets back every element exactly once, as it
/// was before the merge.
struct Restore<T> {
    src: *const T,
    dst: *mut T,
    len: usize,
}

impl<T> Drop for Restore<T> {
    fn drop(&mut self) {
        // SAFETY: `src` holds `len` elements, the slice's contents before the merge, and does
        // not overlap the slice at `dst`, whose current bits are only copies of them.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, self.len) }
    }
}
