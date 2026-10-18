use core::mem::{self, MaybeUninit};
use core::ptr;

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
///   the left half out and merges from the front.
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
/// A merge of `m` elements makes exactly `2⌊m/2⌋` comparisons, after the two that rule out
/// ordered and reversed halves, and moves every element twice: out to the buffer and back.
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
fn sort<T: Ord>(v: &mut [T], buf: *mut T) {
    let len = v.len();
    if len <= RUN {
        // SAFETY: `buf` has room for `len` elements and does not overlap `v`.
        unsafe { sort_run(v, buf) };
        return;
    }
    let mid = len / 2;
    let (left, right) = v.split_at_mut(mid);
    sort(left, buf);
    sort(right, buf);
    // SAFETY: `buf` has room for `len` elements and does not overlap `v`, and `v.len() >= 2`.
    unsafe { merge_halves(v, buf) }
}

/// Sorts a run of at most [`RUN`] elements.
///
/// # Safety
///
/// `v.len() <= RUN`, and `buf` is valid for writes and reads of `v.len()` elements and does
/// not overlap `v`.
unsafe fn sort_run<T: Ord>(v: &mut [T], buf: *mut T) {
    // Compares elements in place, so a panic leaves `v` untouched.
    if v.windows(2).all(|w| w[1] >= w[0]) {
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
unsafe fn merge_halves<T: Ord>(v: &mut [T], buf: *mut T) {
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

/// Merges the sorted halves `buf[..len / 2]` and `buf[len / 2..len]` into `dst[..len]`.
///
/// # Safety
///
/// `len >= 2`; `buf` holds `len` elements, every element of the slice `dst[..len]` exactly
/// once, and does not overlap it.
unsafe fn merge_from_buffer<T: Ord>(buf: *mut T, dst: *mut T, len: usize) {
    let mid = len / 2;
    // SAFETY: `buf` holds every element exactly once while the merge writes bitwise copies of
    // them into `dst`, and `restore` copies `buf` over `dst` if a comparison panics or if the
    // merge turns out inconsistent. Comparisons read only `buf`, which is never written here.
    //
    // The front writes `dst[i]` for `i < mid` and the back `dst[len - 1 - i]`, the middle slot
    // of an odd length is written last, so every slot is written exactly once. Reads stay
    // within `buf` whatever `Ord` answers: before step `i` the front has advanced `i` times in
    // all, so `left <= i < mid` and `right <= mid + i < len`, and likewise the back keeps
    // `left_end >= mid - i >= 1` and `right_end >= len - i > mid`.
    //
    // With a total order the front takes the `mid` smallest elements, the back the `mid`
    // largest, and the two meet exactly: the merge is consistent. If not, some element was
    // taken twice and another not at all, and `dst` gets back the contents of `buf`.
    unsafe {
        let restore = Restore { src: buf, dst, len };
        // The next element of each half at the front, and one past it at the back.
        let (mut left, mut right) = (0, mid);
        let (mut left_end, mut right_end) = (mid, len);
        for i in 0..mid {
            // Front: the right element goes first only if it is strictly smaller.
            let take_right = *buf.add(right) < *buf.add(left);
            let src = buf.add(if take_right { right } else { left });
            ptr::copy_nonoverlapping(src, dst.add(i), 1);
            right += usize::from(take_right);
            left += usize::from(!take_right);

            // Back: the left element goes last only if it is strictly greater. Selecting
            // between the indices of the two elements just compared, rather than between
            // `left_end` and `right_end` before subtracting 1, keeps an addition off the path
            // from the comparison to the load: 10% faster for integers.
            let (last_left, last_right) = (left_end - 1, right_end - 1);
            let take_left = *buf.add(last_right) < *buf.add(last_left);
            let src = buf.add(if take_left { last_left } else { last_right });
            ptr::copy_nonoverlapping(src, dst.add(len - 1 - i), 1);
            left_end -= usize::from(take_left);
            right_end -= usize::from(!take_left);
        }
        if len % 2 == 1 {
            // One element is left, in one half: with a total order this needs no comparison.
            let take_left =
                left < left_end && !(right < right_end && *buf.add(right) < *buf.add(left));
            let src = buf.add(if take_left { left } else { right });
            ptr::copy_nonoverlapping(src, dst.add(mid), 1);
            left += usize::from(take_left);
            right += usize::from(!take_left);
        }
        if left == left_end && right == right_end {
            mem::forget(restore);
        }
    }
}

/// Sorts the `len <= 4` elements at `src` stably into `dst`, with 0, 1, 3 or 5 comparisons.
///
/// Each comparison only chooses which element is copied where, and the choices are arranged so
/// that `dst` receives every element of `src` exactly once whatever `Ord` answers. `src` is
/// only read.
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
