use core::mem::{self, MaybeUninit};
use core::{ptr, slice};

use super::insertion;

/// Sorts a slice using merge sort.
///
/// Merge sort splits the slice into two halves, sorts each half recursively, and merges the
/// two sorted halves into one sorted slice. Every element takes part in one merge per level of
/// recursion, and there are about `log₂ n` levels, so the sort makes `O(n log n)` comparisons
/// whatever the input.
///
/// Five refinements keep the algorithm as cheap as merge sort can be:
///
/// * **Short runs sorted without branches.** Halves of at most 8 elements are not split
///   further. A run is first scanned for order, which leaves a sorted run alone after
///   `len − 1` comparisons and on other input stops at the first descent, usually after one
///   or two. Otherwise its two halves, of at most 4 elements each, are sorted straight into the
///   buffer by fixed sequences of comparisons that select where each element goes instead of
///   branching on the outcome (1, 3 and 5 comparisons for 2, 3 and 4 elements), and merged
///   back. Short runs took up to 60% of the time when they were insertion sorted; together
///   with the stack buffer below, this makes the whole sort 1.2–2.8 times faster for integers
///   (most for small slices) and 1.02–1.9 times for strings and large elements.
/// * **Ordered halves are not merged.** If the first element of the right half is not smaller
///   than the last element of the left half, the halves are already in order, and one
///   comparison replaces the merge: already sorted input costs exactly `n − 1` comparisons and
///   is never moved.
/// * **Reversed halves are swapped.** If the last element of the right half is smaller than
///   the first element of the left half, the whole right half belongs before the left half,
///   and the two trade places as blocks with no further comparison. Reverse-sorted input costs
///   about `2.6n` comparisons instead of `n·log₂ n`.
/// * **Bidirectional, branchless merging.** A merge proceeds from both ends at once: the front
///   takes the smaller of the two first elements, the back the larger of the two last ones.
///   The two halves form two independent chains of comparisons that the processor overlaps,
///   and each step copies from an address chosen by the comparison instead of branching on it.
///   On random input this is 1.2–1.6 times faster than the classic merge, which copies only
///   the left half out and merges from the front. The merge takes a fixed number of steps, so
///   once an end has used up its half it compares the last element the other end took from
///   that half once more. A merge of up to 16 elements therefore decides the whole order
///   before it copies anything; a longer one of elements of at most 8 bytes copies those at
///   most four elements again after its last comparison, and one of larger elements stops as
///   soon as a half has run out, so a comparison that mutates its arguments through interior
///   mutability never leaves a stale copy behind. Against merges that compare stale copies,
///   sound only for types without interior mutability, this costs integers 1.05–1.08 times on
///   random input and 64-byte elements 1.07–1.09 times; strings are within 1.06 either way.
/// * **No allocation for small slices.** A buffer of up to 4 KiB lives on the stack, so a
///   slice that fits in it (512 `u64`s, say) is sorted without touching the allocator.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order. Every comparison that decides a
///   placement lets an element overtake an earlier one only if it is strictly smaller, and
///   reversed halves are swapped only when every element of the right half is strictly smaller
///   than every element of the left half.
/// * **Not in place**: a buffer of `n` elements, on the stack if it fits in 4 KiB and allocated
///   once otherwise, plus `O(log n)` stack for the recursion.
/// * **Adaptive** to order at the scale of runs and halves: sorted runs and halves cost one
///   comparison per element and are never moved, reversed halves are swapped without
///   comparisons.
///
/// A merge of `m` elements makes at most `2⌊m/2⌋` comparisons, after the two that rule out
/// ordered and reversed halves: exactly that many for elements of at most 8 bytes, while a
/// merge of larger elements stops once a half has run out (see above); it moves every element
/// twice: out to the buffer and back.
///
/// | Case                     | Comparisons                      | Moves                     |
/// |--------------------------|----------------------------------|---------------------------|
/// | Best (already sorted)    | `n − 1`                          | `0`                       |
/// | Reverse sorted           | `≈ 2.6n`                         | `O(n log n)`              |
/// | Average (random)         | `≈ n·log₂ n − 0.35n`             | `≈ 2n·log₂(n/8) + 2n`     |
/// | Worst                    | `≤ n·log₂ n + O(n)`              | `≤ 2n·⌈log₂(n/8)⌉ + 2n`   |
///
/// For `n = 1000`: 999 comparisons on sorted input, about 2,590 on reverse-sorted input and
/// about 9,620 on random input. The textbook merge sort makes about 8,700 on random input,
/// since the checks cost a comparison or two per merge and a bidirectional merge does not stop
/// when one half runs out, but 4,932 on sorted and 5,044 on reverse-sorted input.
///
/// Prefer [`slice::sort`], a far more refined stable sort, for real workloads.
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
/// use a::sort::merge;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// merge(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
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
/// merge(&mut cards);
/// let order: String = cards.iter().map(|c| c.1).collect();
/// assert_eq!(order, "dgjbeihacf");
/// ```
pub fn merge<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    let size = mem::size_of::<T>();
    if size > 0 && len <= STACK_BYTES / size {
        let mut stack = StackBuffer::<T> {
            _align: [],
            bytes: [MaybeUninit::uninit(); STACK_BYTES],
        };
        sort(arr, stack.bytes.as_mut_ptr().cast::<T>());
        return;
    }
    if len <= RUN {
        // Elements too large for even a short run to fit on the stack: sorting in place beats
        // allocating.
        insertion(arr);
        return;
    }
    // The buffer is only raw storage: its length stays 0, so it never drops what passes
    // through it.
    let mut buf: Vec<T> = Vec::with_capacity(len);
    sort(arr, buf.as_mut_ptr());
}

/// The longest run that is sorted as a whole instead of being split further: two halves of at
/// most 4 elements, each sorted by [`sort_at_most_4`].
const RUN: usize = 8;

/// The size of the buffer kept on the stack, in bytes.
const STACK_BYTES: usize = 4096;

/// Room for `STACK_BYTES` bytes, aligned for `T`.
#[repr(C)]
struct StackBuffer<T> {
    _align: [T; 0],
    bytes: [MaybeUninit<u8>; STACK_BYTES],
}

/// Sorts `v`, using `buf`, which has room for `v.len()` elements and does not overlap `v`.
pub(super) fn sort<T: Ord>(v: &mut [T], buf: *mut T) {
    let len = v.len();
    if len <= RUN {
        // SAFETY: `buf` has room for `len` elements and does not overlap `v`.
        unsafe { sort_run::<T, true>(v, buf) };
        return;
    }
    let mid = len / 2;
    let (left, right) = v.split_at_mut(mid);
    sort(left, buf);
    sort(right, buf);
    // SAFETY: `buf` has room for `len` elements and does not overlap `v`, and `v.len() >= 2`.
    unsafe { merge_halves(v, buf) }
}

/// The longest slice [`sort_pingpong`] takes: its descents must fit in a `u64`.
pub(super) const PINGPONG_MAX: usize = 64;

/// Returns the descents of `v`, at most [`PINGPONG_MAX`] elements: bit `i` is set if
/// `v[i] < v[i - 1]`, for `0 < i < v.len()`, and bit 0 is clear. Costs `v.len() − 1`
/// comparisons, made in place, with no branch on their outcome.
pub(super) fn descents<T: Ord>(v: &[T]) -> u64 {
    debug_assert!(v.len() <= PINGPONG_MAX);
    let mut mask = 0u64;
    for i in 1..v.len() {
        mask |= u64::from(v[i] < v[i - 1]) << i;
    }
    mask
}

/// Sorts `v`, at most [`PINGPONG_MAX`] elements with the `descents` that [`descents`] returns
/// for it, as [`sort`] does, but with no level copied out before it is merged: every merge
/// reads the halves where the level below left them, in `v` or in `buf`, and writes the
/// other place, so from 9 elements on each level moves every element once instead of twice.
/// The descents say which runs, and which longer stretches of the recursion, are sorted
/// already: those are left alone. Halves that turn out in order after sorting are not merged
/// either, which one comparison tells.
///
/// # Safety
///
/// `v.len() <= PINGPONG_MAX`, and `buf` is valid for writes and reads of `v.len()` elements
/// and does not overlap `v`.
pub(super) unsafe fn sort_pingpong<T: Ord>(v: &mut [T], buf: *mut T, descents: u64) {
    debug_assert!(v.len() <= PINGPONG_MAX);
    // SAFETY: as the caller guarantees.
    unsafe { pingpong::<T, true>(v.as_mut_ptr(), v.len(), buf, descents) }
}

/// Sorts the `len` elements at `v`, leaving them in `v` if `IN_V` and otherwise in `buf`.
/// Bit `i` of `descents` says whether `v[i] < v[i - 1]`, for `0 < i < len`; bit 0 is clear.
///
/// Wherever the sorted elements are left, they are the live copies: the ones every comparison
/// since they were last moved has seen. When they are left in `buf`, the bits left behind in
/// `v` are stale, and the caller takes care of `v` from then on, as the `IN_V` branch below
/// does for each half it has sorted into `buf`.
///
/// # Safety
///
/// `v` holds `len <= PINGPONG_MAX` elements and `buf` is valid for writes and reads of `len`
/// elements and does not overlap `v`.
unsafe fn pingpong<T: Ord, const IN_V: bool>(v: *mut T, len: usize, buf: *mut T, descents: u64) {
    // SAFETY: a stretch without a descent is sorted and stays where it is, copied to `buf`
    // whole if that is where it is wanted. A run of at most `RUN` elements is sorted by
    // `sort_run`, which compares `v` in place and merges through `buf`, or through `tmp` when
    // its result is wanted in `buf`, with `restore` putting the live copies back into `v`
    // if a comparison panics. Both halves of a longer slice are sorted into the other place
    // than the slice is wanted and merged into it by `merge_from_buffer`, which leaves its
    // destination holding every element exactly once whatever `Ord` does; while the halves
    // are in `buf`, `restore` puts them back into `v` if a comparison panics, and the halves
    // are checked for order under it. Two halves sorted already are left in `v`, where
    // `merge_halves` keeps every element exactly once as well. So `v` holds every element
    // exactly once whenever this returns or unwinds, and holds the live copies unless it
    // returns with them in `buf`. The halves' descents are the bits below `mid`, and the bits
    // from `mid + 1` on shifted down.
    unsafe {
        if descents == 0 {
            if !IN_V {
                ptr::copy_nonoverlapping(v, buf, len);
            }
            return;
        }
        if len <= RUN {
            if IN_V {
                sort_run::<T, false>(slice::from_raw_parts_mut(v, len), buf);
                return;
            }
            let mid = len / 2;
            if len <= 4 {
                sort_at_most_4(v, len, buf);
                return;
            }
            let mut tmp = MaybeUninit::<[T; RUN]>::uninit();
            let tmp = tmp.as_mut_ptr().cast::<T>();
            sort_at_most_4(v, mid, tmp);
            sort_at_most_4(v.add(mid), len - mid, tmp.add(mid));
            // `tmp` now holds the live copies, until the merge has written them into `buf`.
            let restore = Restore {
                src: tmp,
                dst: v,
                len,
            };
            merge_from_buffer(tmp, buf, len);
            mem::forget(restore);
            return;
        }
        let mid = len / 2;
        let (left, right) = (descents & ((1u64 << mid) - 1), (descents >> mid) & !1);
        if IN_V {
            if left == 0 && right == 0 {
                // Both halves are sorted and only their boundary is not: merge sort's checks
                // for halves in order or reversed, and its merge out of `buf`, do the rest.
                merge_halves(slice::from_raw_parts_mut(v, len), buf);
                return;
            }
            pingpong::<T, false>(v, mid, buf, left);
            let mut restore = Restore {
                src: buf,
                dst: v,
                len: mid,
            };
            pingpong::<T, false>(v.add(mid), len - mid, buf.add(mid), right);
            restore.len = len;
            if *buf.add(mid) >= *buf.add(mid - 1) {
                // In order already: dropping `restore` copies the halves back as they are.
                drop(restore);
                return;
            }
            mem::forget(restore);
            merge_from_buffer(buf, v, len);
        } else {
            pingpong::<T, true>(v, mid, buf, left);
            pingpong::<T, true>(v.add(mid), len - mid, buf.add(mid), right);
            // Compares `v` in place.
            if *v.add(mid) >= *v.add(mid - 1) {
                ptr::copy_nonoverlapping(v, buf, len);
                return;
            }
            merge_from_buffer(v, buf, len);
        }
    }
}

/// Sorts a run of at most [`RUN`] elements.
///
/// If `SCAN`, the run is first checked for order, up to the first descent: a sorted run is
/// left alone after `len − 1` comparisons.
///
/// # Safety
///
/// `v.len() <= RUN`, and `buf` is valid for writes and reads of `v.len()` elements and does
/// not overlap `v`.
pub(super) unsafe fn sort_run<T: Ord, const SCAN: bool>(v: &mut [T], buf: *mut T) {
    // Compares elements in place, so a panic leaves `v` untouched.
    if SCAN && v.windows(2).all(|w| w[1] >= w[0]) {
        return;
    }
    let len = v.len();
    let mid = len / 2;
    let src = v.as_mut_ptr();
    // SAFETY: `len <= 8`, so each half has at most 4 elements. Both halves are sorted from `v`
    // into `buf`, reading `v` only, so `v` is untouched if a comparison panics; whatever `Ord`
    // answers, `buf` then holds every element of `v` exactly once, which is what
    // `merge_from_buffer` needs.
    unsafe {
        if len <= 4 {
            sort_at_most_4(src, len, buf);
            ptr::copy_nonoverlapping(buf, src, len);
            return;
        }
        sort_at_most_4(src, mid, buf);
        sort_at_most_4(src.add(mid), len - mid, buf.add(mid));
        merge_from_buffer(buf, src, len);
    }
}

/// Merges the sorted halves `v[..v.len() / 2]` and `v[v.len() / 2..]` into one sorted slice.
///
/// # Safety
///
/// `v.len() >= 2`, and `buf` is valid for writes and reads of `v.len()` elements and does not
/// overlap `v`.
pub(super) unsafe fn merge_halves<T: Ord>(v: &mut [T], buf: *mut T) {
    let len = v.len();
    let mid = len / 2;
    // Both checks compare elements in place, so a panic leaves `v` untouched.
    if v[mid] >= v[mid - 1] {
        return;
    }
    if v[len - 1] < v[0] {
        // Every element of the right half is smaller than every element of the left half.
        let dst = v.as_mut_ptr();
        // SAFETY: all offsets are within `v` and `buf`. The left half is copied out, the right
        // half moved to the front (`ptr::copy` allows the overlap), and the left half copied in
        // behind it. Nothing in between can panic.
        unsafe {
            ptr::copy_nonoverlapping(dst, buf, mid);
            ptr::copy(dst.add(mid), dst, len - mid);
            ptr::copy_nonoverlapping(buf, dst.add(len - mid), mid);
        }
        return;
    }
    let dst = v.as_mut_ptr();
    // SAFETY: `v` is copied out whole into `buf`, which then holds every element exactly once,
    // as `merge_from_buffer` needs, and has room for `len` elements.
    unsafe {
        ptr::copy_nonoverlapping(dst, buf, len);
        merge_from_buffer(buf, dst, len);
    }
}

/// Merges of at most this many elements are made by [`merge_gather`], longer ones by
/// [`merge_tracked`] or [`merge_checked`]: measured for integers, the gather is the cheaper
/// up to 16 elements and the dearer from 32 on.
const GATHER_MAX: usize = 16;

/// The largest element size, in bytes, merged by [`merge_tracked`] beyond [`GATHER_MAX`]
/// elements; larger elements are merged by [`merge_checked`]. Measured on merge sort against
/// the merge that compares stale copies: tracking costs integers 1.06–1.08 times on random
/// input and the checked merge would cost them about as much once per merge, mispredicted;
/// for strings and 64-byte elements, whose comparisons and copies dwarf that branch, the
/// checked merge costs 1.02–1.05 times, while tracking made the compiler spill the loop
/// around the comparison call (strings 1.15 times slower) or turn the selects into branches
/// (64-byte elements 1.25 times slower).
const TRACKED_MAX_SIZE: usize = 8;

/// Merges the sorted halves `src[..len / 2]` and `src[len / 2..len]` into `dst[..len]`, from
/// both ends at once with a fixed number of steps and no branch on a comparison.
///
/// `src` is only read. Whatever `Ord` does, `dst[..len]` afterwards holds every element of
/// `src[..len]` exactly once, as the live copies: the merged elements, or if a comparison
/// panicked or the merge turned out inconsistent (which only an `Ord` that is not a total
/// order can cause), plain copies of `src`.
///
/// # Safety
///
/// `len >= 2`; `src` holds `len` elements, `dst` is valid for writes of `len` elements, and
/// the two do not overlap.
unsafe fn merge_from_buffer<T: Ord>(src: *const T, dst: *mut T, len: usize) {
    // SAFETY: as the caller guarantees; each shape documents its own argument.
    unsafe {
        if len <= GATHER_MAX {
            merge_gather(src, dst, len);
        } else if mem::size_of::<T>() <= TRACKED_MAX_SIZE {
            merge_tracked(src, dst, len);
        } else {
            merge_checked(src, dst, len);
        }
    }
}

/// [`merge_from_buffer`] for elements larger than [`TRACKED_MAX_SIZE`] bytes: the merge
/// stops as soon as a half has run out and copies what is left of the other, so it never
/// compares an element whose copy is written already and needs no tracking. The first
/// `mid / 2` steps need no check: after `k` steps both ends together have taken at most `2k`
/// elements of a half that holds at least `mid`. A merge of `m` elements makes at most
/// `2⌊m/2⌋` comparisons, fewer when a half runs out early.
///
/// # Safety
///
/// As [`merge_from_buffer`].
unsafe fn merge_checked<T: Ord>(src: *const T, dst: *mut T, len: usize) {
    let mid = len / 2;
    // SAFETY: `src` holds every element exactly once and is only read, while the merge writes
    // copies of them into `dst[out..out_end)`, the slots not yet written, as many as elements
    // are left; `restore` copies `src` over `dst` if a comparison panics. Every step takes
    // one element at the front and one at the back from halves that have elements left, so
    // the reads stay within what is left of each half and every element is written exactly
    // once, whatever `Ord` answers: after the first `mid / 2` steps (safe by the argument
    // above) each step checks, and the loop ends as soon as a half has run out; what is left
    // then fills exactly the slots left. Stability: the right element goes first only if it
    // is strictly smaller, the left one last only if it is strictly greater.
    unsafe {
        let restore = Restore { src, dst, len };
        let (mut left, mut right) = (0, mid);
        let (mut left_end, mut right_end) = (mid, len);
        let (mut out, mut out_end) = (0, len);
        for _ in 0..mid / 2 {
            let take_right = *src.add(right) < *src.add(left);
            ptr::copy_nonoverlapping(
                src.add(if take_right { right } else { left }),
                dst.add(out),
                1,
            );
            out += 1;
            right += usize::from(take_right);
            left += usize::from(!take_right);

            let (last_left, last_right) = (left_end - 1, right_end - 1);
            let take_left = *src.add(last_right) < *src.add(last_left);
            out_end -= 1;
            ptr::copy_nonoverlapping(
                src.add(if take_left { last_left } else { last_right }),
                dst.add(out_end),
                1,
            );
            left_end -= usize::from(take_left);
            right_end -= usize::from(!take_left);
        }
        while left < left_end && right < right_end {
            let take_right = *src.add(right) < *src.add(left);
            ptr::copy_nonoverlapping(
                src.add(if take_right { right } else { left }),
                dst.add(out),
                1,
            );
            out += 1;
            right += usize::from(take_right);
            left += usize::from(!take_right);
            if left == left_end || right == right_end {
                break;
            }
            let (last_left, last_right) = (left_end - 1, right_end - 1);
            let take_left = *src.add(last_right) < *src.add(last_left);
            out_end -= 1;
            ptr::copy_nonoverlapping(
                src.add(if take_left { last_left } else { last_right }),
                dst.add(out_end),
                1,
            );
            left_end -= usize::from(take_left);
            right_end -= usize::from(!take_left);
        }
        // What is left of a half (of both, if the loop ended on the back's turn with both
        // run out) goes in order into the slots left, which it fills exactly.
        let rest_left = left_end - left;
        debug_assert_eq!(out + rest_left + (right_end - right), out_end);
        ptr::copy_nonoverlapping(src.add(left), dst.add(out), rest_left);
        ptr::copy_nonoverlapping(src.add(right), dst.add(out + rest_left), right_end - right);
        mem::forget(restore);
    }
}

/// [`merge_from_buffer`] for elements of at most [`TRACKED_MAX_SIZE`] bytes, with the slot of
/// the last element each end took from each half tracked, and the four elements copied again
/// once every comparison is made: a fixed number of steps with no branch on a comparison.
///
/// # Safety
///
/// As [`merge_from_buffer`].
unsafe fn merge_tracked<T: Ord>(src: *const T, dst: *mut T, len: usize) {
    let mid = len / 2;
    // SAFETY: `src` holds every element exactly once and is only read, while the merge writes
    // bitwise copies of them into `dst`; `restore` copies `src` over `dst` if a comparison
    // panics or if the merge turns out inconsistent, so `dst` holds every element exactly
    // once whenever this returns or unwinds.
    //
    // The front writes `dst[i]` for `i < mid` and the back `dst[len - 1 - i]`, the middle slot
    // of an odd length is written last, so every slot is written exactly once. Reads stay
    // within `src` whatever `Ord` answers: before step `i` the front has advanced `i` times in
    // all, so `left <= i < mid` and `right <= mid + i < len`, and likewise the back keeps
    // `left_end >= mid - i >= 1` and `right_end >= len - i > mid`.
    //
    // With a total order the front takes the `mid` smallest elements, the back the `mid`
    // largest, and the two meet exactly: the merge is consistent. If not, some element was
    // taken twice and another not at all, and `dst` gets back the contents of `src`.
    //
    // An end compares an element whose copy is already written only once its half has run
    // out (`left == left_end` at the front, say): the element is then the last one the other
    // end took from that half, `src[left_end]`, and stays so, since either end taking from
    // that half again would cross the other and make the merge inconsistent. A comparison may
    // mutate its arguments, through interior mutability, which would leave such a copy stale:
    // so the slot of the last element each end took from each half is tracked, and once every
    // comparison is made the current state of each of those four elements is copied over its
    // copy. The four copies are made whatever happened: the slot of an end that took nothing
    // from a half is `spare`, a slot of its own, and the index of a half nothing was taken
    // from is clamped into `src`. Nothing is read from `spare` and nothing is dropped from it.
    unsafe {
        let restore = Restore { src, dst, len };
        // The next element of each half at the front, and one past it at the back.
        let (mut left, mut right) = (0, mid);
        let (mut left_end, mut right_end) = (mid, len);
        // The slot of the last element each end took from each half, as an index into `dst`,
        // `usize::MAX` while an end has taken nothing from a half, kept up to date by masking
        // with the comparison's answer, which the compiler lowers to conditional selects.
        let (mut front_left_i, mut front_right_i) = (usize::MAX, usize::MAX);
        let (mut back_left_i, mut back_right_i) = (usize::MAX, usize::MAX);
        for i in 0..mid {
            // Front: the right element goes first only if it is strictly smaller.
            let take_right = *src.add(right) < *src.add(left);
            let from = src.add(if take_right { right } else { left });
            ptr::copy_nonoverlapping(from, dst.add(i), 1);
            let mask = usize::from(take_right).wrapping_neg();
            front_right_i ^= (front_right_i ^ i) & mask;
            front_left_i ^= (front_left_i ^ i) & !mask;
            right += usize::from(take_right);
            left += usize::from(!take_right);

            // Back: the left element goes last only if it is strictly greater.
            let (last_left, last_right) = (left_end - 1, right_end - 1);
            let take_left = *src.add(last_right) < *src.add(last_left);
            let from = src.add(if take_left { last_left } else { last_right });
            ptr::copy_nonoverlapping(from, dst.add(len - 1 - i), 1);
            let mask = usize::from(take_left).wrapping_neg();
            back_left_i ^= (back_left_i ^ (len - 1 - i)) & mask;
            back_right_i ^= (back_right_i ^ (len - 1 - i)) & !mask;
            left_end -= usize::from(take_left);
            right_end -= usize::from(!take_left);
        }
        // The last element each end took from each half, as it is now, over its copy; an end
        // that took nothing from a half writes `spare`, a slot of its own that is never read.
        let mut spare = MaybeUninit::<T>::uninit();
        let spare = spare.as_mut_ptr();
        let slot = |k: usize| if k == usize::MAX { spare } else { dst.add(k) };
        ptr::copy_nonoverlapping(src.add(left.max(1) - 1), slot(front_left_i), 1);
        ptr::copy_nonoverlapping(src.add(right - 1), slot(front_right_i), 1);
        ptr::copy_nonoverlapping(src.add(left_end), slot(back_left_i), 1);
        ptr::copy_nonoverlapping(src.add(right_end.min(len - 1)), slot(back_right_i), 1);
        if len % 2 == 1 {
            // One element is left, in one half: with a total order this needs no comparison,
            // and otherwise both elements compared are ones not yet copied.
            let take_left =
                left < left_end && !(right < right_end && *src.add(right) < *src.add(left));
            let from = src.add(if take_left { left } else { right });
            ptr::copy_nonoverlapping(from, dst.add(mid), 1);
            left += usize::from(take_left);
            right += usize::from(!take_left);
        }
        if left == left_end && right == right_end {
            mem::forget(restore);
        }
    }
}

/// [`merge_from_buffer`] for `len <= GATHER_MAX`, deciding the whole order first, as indices
/// into `src`, and copying afterwards.
///
/// # Safety
///
/// As [`merge_from_buffer`].
unsafe fn merge_gather<T: Ord>(src: *const T, dst: *mut T, len: usize) {
    debug_assert!(len <= GATHER_MAX);
    let mid = len / 2;
    // SAFETY: the comparisons only decide `order`, the index in `src` of the element for each
    // slot; nothing is copied until every comparison is made, so every comparison is of the
    // only live copy of its arguments. The reads stay within `src` and the merge is
    // consistent with a total order by the argument in `merge_tracked`; `order` is then a
    // permutation and the elements are gathered into `dst`, and otherwise, or if a comparison
    // panics, `restore` copies `src` over `dst`.
    unsafe {
        let restore = Restore { src, dst, len };
        let mut order = MaybeUninit::<[u8; GATHER_MAX]>::uninit();
        let order = order.as_mut_ptr().cast::<u8>();
        let (mut left, mut right) = (0, mid);
        let (mut left_end, mut right_end) = (mid, len);
        for i in 0..mid {
            let take_right = *src.add(right) < *src.add(left);
            *order.add(i) = (if take_right { right } else { left }) as u8;
            right += usize::from(take_right);
            left += usize::from(!take_right);

            let (last_left, last_right) = (left_end - 1, right_end - 1);
            let take_left = *src.add(last_right) < *src.add(last_left);
            *order.add(len - 1 - i) = (if take_left { last_left } else { last_right }) as u8;
            left_end -= usize::from(take_left);
            right_end -= usize::from(!take_left);
        }
        if len % 2 == 1 {
            let take_left =
                left < left_end && !(right < right_end && *src.add(right) < *src.add(left));
            *order.add(mid) = (if take_left { left } else { right }) as u8;
            left += usize::from(take_left);
            right += usize::from(!take_left);
        }
        if left == left_end && right == right_end {
            for k in 0..len {
                ptr::copy_nonoverlapping(src.add(usize::from(*order.add(k))), dst.add(k), 1);
            }
            mem::forget(restore);
        }
    }
}

/// Sorts the `len <= 4` elements at `src` stably into `dst`, with 0, 1, 3 or 5 comparisons.
///
/// Each comparison only chooses which element is copied where, and the choices are arranged so
/// that `dst` receives every element of `src` exactly once whatever `Ord` answers. `src` is
/// only read, and nothing is copied before the last comparison.
///
/// # Safety
///
/// `src` is valid for reads and `dst` for writes of `len` elements, and they do not overlap.
#[inline(always)]
unsafe fn sort_at_most_4<T: Ord>(src: *const T, len: usize, dst: *mut T) {
    // SAFETY: every offset is below `len`, as the caller guarantees room for.
    unsafe {
        match len {
            0 => {}
            1 => ptr::copy_nonoverlapping(src, dst, 1),
            2 => {
                let swap = *src.add(1) < *src;
                ptr::copy_nonoverlapping(src.add(usize::from(swap)), dst, 1);
                ptr::copy_nonoverlapping(src.add(usize::from(!swap)), dst.add(1), 1);
            }
            3 => {
                // Order the first two, then place the third: before `hi` if strictly smaller,
                // and before `lo` too if strictly smaller than that. Ignoring the second answer
                // when the first is "no" keeps the three destinations distinct.
                let swap = *src.add(1) < *src;
                let (lo, hi) = (src.add(usize::from(swap)), src.add(usize::from(!swap)));
                let last = src.add(2);
                let before_hi = *last < *hi;
                let before_lo = before_hi & (*last < *lo);
                let first = if before_lo { last } else { lo };
                let second = if before_lo {
                    lo
                } else if before_hi {
                    last
                } else {
                    hi
                };
                let third = if before_hi { hi } else { last };
                ptr::copy_nonoverlapping(first, dst, 1);
                ptr::copy_nonoverlapping(second, dst.add(1), 1);
                ptr::copy_nonoverlapping(third, dst.add(2), 1);
            }
            _ => {
                // The standard library's stable four-element sort: order both pairs, find the
                // minimum and the maximum among the pairs' ends, and order the two left over.
                let swap_left = *src.add(1) < *src;
                let swap_right = *src.add(3) < *src.add(2);
                let a = src.add(usize::from(swap_left));
                let b = src.add(usize::from(!swap_left));
                let c = src.add(2 + usize::from(swap_right));
                let d = src.add(2 + usize::from(!swap_right));
                let c_first = *c < *a;
                let b_last = *d < *b;
                let min = if c_first { c } else { a };
                let max = if b_last { b } else { d };
                let unknown_left = if c_first {
                    a
                } else if b_last {
                    c
                } else {
                    b
                };
                let unknown_right = if b_last {
                    d
                } else if c_first {
                    b
                } else {
                    c
                };
                let swap_middle = *unknown_right < *unknown_left;
                let lo = if swap_middle {
                    unknown_right
                } else {
                    unknown_left
                };
                let hi = if swap_middle {
                    unknown_left
                } else {
                    unknown_right
                };
                ptr::copy_nonoverlapping(min, dst, 1);
                ptr::copy_nonoverlapping(lo, dst.add(1), 1);
                ptr::copy_nonoverlapping(hi, dst.add(2), 1);
                ptr::copy_nonoverlapping(max, dst.add(3), 1);
            }
        }
    }
}

/// A copy of a slice that is written back over it when dropped, unless forgotten.
///
/// Dropping it, normally or while unwinding from a panicking comparison, is what keeps
/// [`merge_from_buffer`] panic-safe and safe against an [`Ord`] that is not a total order: the
/// slice gets back every element exactly once, as it was before the merge.
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
