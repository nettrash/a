use core::mem::ManuallyDrop;
use core::ptr;

use super::heap;

/// Sorts a slice in place using intro sort.
///
/// Intro sort (introspective sort), designed by David Musser in 1997, is a quick sort that
/// watches its own progress. Quick sort partitions the slice around a pivot and sorts the two
/// parts the same way; with good pivots each level of partitioning halves the parts and the
/// sort makes `O(n log n)` comparisons, but no cheap pivot rule is good on every input, and
/// bad pivots make it quadratic. Intro sort counts the levels: a part still being partitioned
/// after `2·⌊log₂ n⌋` levels is handed to heap sort, which is `O(n log n)` on every input, so
/// the whole sort is too, while typical input never reaches the limit and keeps quick sort's
/// speed. Short parts are left to insertion sort.
///
/// Five refinements keep the algorithm as cheap as intro sort can be, and its comparisons the
/// same as Hoare's:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or strictly descending run; if it covers the whole slice, the slice is left
///   alone or reversed. This costs `n − 1` comparisons on such input and usually two or three
///   on any other.
/// * **Branchless block partitioning.** Hoare's scheme scans from both ends for elements on
///   the wrong side of the pivot and swaps them; each scan stops at an unpredictable element,
///   a mispredicted branch. Here each end compares a block of 64 elements first, recording
///   the misplaced ones by offset without branching, and then exchanges as many misplaced
///   pairs as both blocks have (Edelkamp and Weiss's BlockQuicksort). It makes the same
///   comparisons as Hoare's scheme and exchanges each pair in a cycle of two moves instead of
///   a swap's three. On random input this is 1.8–2.7 times faster than Hoare's scans for
///   integers, 1.7–2.2 times for strings and 1.4–2.1 times for 64-byte elements. The block
///   scans are unrolled by 8, another 1.03–1.08 times on random input and 1.1–1.25 times on
///   nearly sorted input, where the scans are nearly all the work.
/// * **Equal keys split evenly.** As in Hoare's and Musser's scheme, an element equal to the
///   pivot stops the scans on both sides, so equal keys are spread over both parts: input with
///   few distinct keys, even all equal but one, costs `O(n log n)` like any other, without
///   ever reaching the depth limit.
/// * **Pivots away from the ends, from a ninther on long parts.** The pivot is the median of
///   the elements at a quarter, half and three quarters of the part, and from 128 elements on
///   the median of three medians of three spread over it (Tukey's ninther), which is closer to
///   the true median. Musser samples the first, middle and last elements instead, which on
///   sorted input with a few elements appended keeps choosing the second smallest element as
///   the pivot until the depth limit gives up: quartiles are 1.2–4 times faster there.
/// * **Insertion sort while the part is fresh.** Parts of at most 24 elements are insertion
///   sorted as soon as partitioning leaves them, while they are still in the cache. Musser
///   leaves them all for one insertion pass over the whole slice at the end, which makes the
///   same comparisons but reads the whole slice once more: 1.05–1.15 times slower for strings.
///
/// The recursion goes into the smaller part only and loops on the larger one, so the stack
/// never holds more than `log₂ n` frames.
///
/// # Properties
///
/// * **Unstable**: equal elements may be reordered.
/// * **In place**: no allocation, `O(log n)` stack.
/// * **Adaptive** to sorted and reverse-sorted input; nearly sorted input is cheap too, since
///   Hoare's scheme moves only misplaced elements.
///
/// | Case                              | Comparisons                     | Moves              |
/// |-----------------------------------|---------------------------------|--------------------|
/// | Best (sorted, reverse, all equal) | `n − 1`                         | `0` (`n` reversed) |
/// | Average (random)                  | `≈ (1.1–1.19)·n·log₂ n`         | `O(n log n)`       |
/// | Worst                             | `O(n log n)`                    | `O(n log n)`       |
///
/// For `n = 1000`: 999 comparisons on sorted or reverse-sorted input, about 6,800 with 10
/// random pairs swapped and about 11,800 on random input. Insertion into parts of up to 24
/// elements costs more comparisons than partitioning them further would, but less time. With
/// `k` distinct keys the cost stays about `0.85·n·log₂ n`: unlike this crate's
/// [`quick`](crate::sort::quick) sort, intro sort does not take all copies of the pivot out at
/// once. McIlroy's adversary, which answers comparisons so as to make every pivot bad, drives
/// the partitioning alone to 9.1 million comparisons for `n = 10,000`; the heap sort fallback
/// keeps the whole sort at about 470,000.
///
/// Against this crate's [`quick`](crate::sort::quick) sort, which also falls back to heap sort
/// but partitions elements of up to 64 bytes by Lomuto's scheme without branching, sorts short
/// parts by sorting networks and takes all copies of a pivot out at once: 1.4–1.5 times slower
/// on random integers, 1.06–1.13 times on random strings and 1.5–3.8 times with few distinct
/// keys; but 1.2–1.65 times faster on random 64-byte elements, which block partitioning moves
/// only when misplaced instead of twice each, and 1.3–3.6 times faster on nearly sorted input.
/// Against [`slice::sort_unstable`]: 1.6–1.8 times slower on random integers and within 1.07
/// times on random strings, but 1.08–1.55 times faster on random 64-byte elements and 1.5–3.7
/// times on nearly sorted input of 100 elements or more. Against [`heap`](fn@heap) sort, which
/// it falls back to, it is 1.75–2.4 times faster on random integers, 2.1–6.7 times on strings
/// and 1.5–4.4 times on 64-byte elements.
///
/// Prefer [`slice::sort_unstable`] for real workloads.
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
/// use a::sort::intro;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// intro(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Equal keys split evenly between the parts.
/// let mut v: Vec<u8> = (0..1000).map(|i| (i * 7 % 3) as u8).collect();
/// intro(&mut v);
/// assert!(v.windows(2).all(|w| w[0] <= w[1]));
///
/// let mut words = vec!["pear", "apple", "fig"];
/// intro(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn intro<T: Ord>(arr: &mut [T]) {
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
    sort(arr, 2 * len.ilog2());
}

/// Parts of at most this many elements are sorted by insertion instead of partitioned.
///
/// Measured against 16, Musser's choice: as fast on random input, and 1.05–1.45 times faster
/// on nearly sorted input, few distinct keys and short sawtooth slices. Against 32: 1.03–1.06
/// times faster on random strings of up to 1000 elements, but up to 1.1 times slower with few
/// distinct keys for 64-byte elements. For elements of at most 8 bytes, this crate's merge
/// sort's branchless short-run sort would be 1.1–1.4 times faster than insertion on random
/// slices of up to 100 elements, but 1.1–1.4 times slower on nearly sorted input and with
/// few distinct keys, whose short parts insertion sort barely has to move.
const SHORT: usize = 24;

/// From this length on, the pivot is the median of three medians of three (Tukey's ninther)
/// instead of the median of three.
///
/// Measured against the median of three throughout: 1.04–1.13 times faster on sawtooth input
/// of 100,000 elements or more, and up to 1.08 times on random input.
const NINTHER: usize = 128;

/// How many elements each side of [`partition_blocks`] compares before moving any; at most 256,
/// so that an offset into a block fits in a byte.
const BLOCK: usize = 64;

/// How many comparisons [`scan_block`] makes per round of its loop.
const SCAN_UNROLL: usize = 8;

/// Sorts `v`, falling back to heap sort once `limit` more levels of partitioning have not
/// finished the job.
fn sort<T: Ord>(mut v: &mut [T], mut limit: u32) {
    loop {
        if v.len() <= SHORT {
            insertion_sort(v);
            return;
        }
        if limit == 0 {
            heap(v);
            return;
        }
        limit -= 1;

        let pivot = choose_pivot(v);
        v.swap(0, pivot);
        let mid = partition(v);
        let (left, rest) = v.split_at_mut(mid);
        let right = &mut rest[1..];
        // Recursing into the smaller part only keeps the stack within `log₂ n` frames.
        if left.len() < right.len() {
            sort(left, limit);
            v = right;
        } else {
            sort(right, limit);
            v = left;
        }
    }
}

/// Returns the index of the median of the elements at a quarter, half and three quarters of
/// `v`, or from [`NINTHER`] elements on of the median of the medians of three groups of three
/// spread over `v`.
///
/// Musser takes the first, middle and last elements. Swapping the pivot to the front then moves
/// the first element, the smallest on sorted stretches, into the middle of the part, where the
/// next level's sample finds it again: on sorted input with a few random elements appended, the
/// pivot is the second smallest element level after level until the depth limit gives up.
/// Quartiles leave the ends alone and are 1.2–4 times faster on such input, and no different on
/// any other.
fn choose_pivot<T: Ord>(v: &[T]) -> usize {
    let len = v.len();
    if len < NINTHER {
        return median_of_3(v, len / 4, len / 2, len / 4 * 3);
    }
    let step = len / 8;
    let (a, b, c) = (len / 4, len / 2, len / 4 * 3);
    let a = median_of_3(v, a - step, a, a + step);
    let b = median_of_3(v, b - step, b, b + step);
    let c = median_of_3(v, c - step, c, c + step);
    median_of_3(v, a, b, c)
}

/// Returns whichever of `a`, `b` and `c` indexes the median of the three elements, with two or
/// three comparisons.
fn median_of_3<T: Ord>(v: &[T], a: usize, b: usize, c: usize) -> usize {
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

/// Partitions `v` around the pivot `v[0]` and moves the pivot between the two parts. Returns
/// its index `mid`: no element of `v[..mid]` is greater than the pivot, and no element of
/// `v[mid + 1..]` smaller.
fn partition<T: Ord>(v: &mut [T]) -> usize {
    let (pivot, rest) = v.split_first_mut().expect("`v` is not empty");
    let mid = partition_blocks(rest, pivot);
    v.swap(0, mid);
    mid
}

/// Partitions `v` around `pivot`, which is not in `v`, by Hoare's scheme without branching on
/// the comparisons (Edelkamp and Weiss's BlockQuicksort). Returns `mid`: no element of
/// `v[..mid]` is greater than the pivot, and no element of `v[mid..]` smaller.
///
/// Hoare's scheme scans from the left for an element that must not stay on the left, from the
/// right for one that must not stay on the right, and swaps the two; each scan stops at an
/// unpredictable element, a mispredicted branch. Here each side instead compares a whole block
/// of [`BLOCK`] elements first, writing the offset of every element and advancing a count by
/// whether it is misplaced, so the outcome selects an address instead of a branch; then as many
/// misplaced pairs as both blocks have are exchanged. A block whose misplaced elements have all
/// been exchanged is replaced by the next one; the last blocks share what is left.
///
/// Elements equal to the pivot count as misplaced on both sides, as in Hoare's and Musser's
/// scheme, so they are spread over both parts: all-equal input splits evenly instead of
/// degrading.
fn partition_blocks<T: Ord>(v: &mut [T], pivot: &T) -> usize {
    let base = v.as_mut_ptr();
    let mut offsets_l = [0u8; BLOCK];
    let mut offsets_r = [0u8; BLOCK];
    // `v[l..r]` is not partitioned yet; the left block is `v[l..l + block_l]` and the right
    // block `v[r - block_r..r]`, which do not overlap. `offsets_l[start_l..end_l]` lists the
    // misplaced elements of the left block not yet exchanged, and likewise on the right.
    let (mut l, mut r) = (0, v.len());
    let (mut block_l, mut block_r) = (BLOCK, BLOCK);
    let (mut start_l, mut end_l) = (0, 0);
    let (mut start_r, mut end_r) = (0, 0);
    // SAFETY: every element read or moved is at `l + offset` with `offset < block_l`, or at
    // `r - 1 - offset` with `offset < block_r`, and `block_l + block_r <= r - l`, so it lies
    // inside `v` and the two blocks do not overlap, whatever `Ord` answers: the comparisons only
    // decide how far the counts advance, and a count never exceeds its block's size. Elements
    // are compared only while every element is in `v`, and the exchanges that move them cannot
    // panic, so a panicking comparison leaves every element in `v` exactly once. `pivot` is not
    // in `v`.
    unsafe {
        loop {
            let last = r - l <= 2 * BLOCK;
            if last {
                // Share what is left between the blocks; a block with misplaced elements left
                // keeps its size.
                let mut rest = r - l;
                if start_l < end_l || start_r < end_r {
                    rest -= BLOCK;
                }
                if start_l < end_l {
                    block_r = rest;
                } else if start_r < end_r {
                    block_l = rest;
                } else {
                    block_l = rest / 2;
                    block_r = rest - block_l;
                }
            }
            if start_l == end_l {
                start_l = 0;
                end_l = scan_block(&mut offsets_l, block_l, move |i| *base.add(l + i) >= *pivot);
            }
            if start_r == end_r {
                start_r = 0;
                end_r = scan_block(&mut offsets_r, block_r, move |i| {
                    *base.add(r - 1 - i) <= *pivot
                });
            }

            let count = (end_l - start_l).min(end_r - start_r);
            if count > 0 {
                let left = |k: usize| base.add(l + *offsets_l.get_unchecked(start_l + k) as usize);
                let right =
                    |k: usize| base.add(r - 1 - *offsets_r.get_unchecked(start_r + k) as usize);
                // Exchange the pairs in one cycle, two moves per pair instead of a swap's
                // three: hold the first left element out, then move each right element into
                // its left partner's slot and the next left element into the slot it left.
                let held = ManuallyDrop::new(ptr::read(left(0)));
                ptr::copy_nonoverlapping(right(0), left(0), 1);
                for k in 1..count {
                    ptr::copy_nonoverlapping(left(k), right(k - 1), 1);
                    ptr::copy_nonoverlapping(right(k), left(k), 1);
                }
                ptr::copy_nonoverlapping(&*held, right(count - 1), 1);
                start_l += count;
                start_r += count;
            }

            if start_l == end_l {
                l += block_l;
            }
            if start_r == end_r {
                r -= block_r;
            }
            if last {
                break;
            }
        }

        // At most one block has misplaced elements left; everything else is partitioned.
        // Move those elements to the far side of what is left, the last one first.
        if start_l < end_l {
            while start_l < end_l {
                end_l -= 1;
                r -= 1;
                ptr::swap(
                    base.add(l + *offsets_l.get_unchecked(end_l) as usize),
                    base.add(r),
                );
            }
            r
        } else {
            while start_r < end_r {
                end_r -= 1;
                ptr::swap(
                    base.add(l),
                    base.add(r - 1 - *offsets_r.get_unchecked(end_r) as usize),
                );
                l += 1;
            }
            l
        }
    }
}

/// Writes the offsets `i < block` for which `misplaced(i)` holds into `offsets`, in order, and
/// returns how many there are. Every offset is written and the count advanced by the answer,
/// so the answer selects an address instead of a branch.
///
/// The loop is unrolled by [`SCAN_UNROLL`]: measured against no unrolling, 1.03–1.08 times faster
/// on random input and 1.1–1.25 times on nearly sorted input, where nearly nothing is
/// exchanged and the scans are all the work; against unrolling by 4, up to 1.03 times.
#[inline(always)]
fn scan_block(offsets: &mut [u8; BLOCK], block: usize, misplaced: impl Fn(usize) -> bool) -> usize {
    assert!(block <= BLOCK);
    let mut count = 0;
    let mut i = 0;
    // SAFETY: `count <= i < block <= BLOCK`, so every write is inside `offsets`. Checking each
    // index instead made the whole sort up to 1.1 times slower.
    unsafe {
        while i + SCAN_UNROLL <= block {
            for k in 0..SCAN_UNROLL {
                // `i + k < BLOCK <= 256` fits in a byte.
                *offsets.get_unchecked_mut(count) = (i + k) as u8;
                count += usize::from(misplaced(i + k));
            }
            i += SCAN_UNROLL;
        }
        while i < block {
            *offsets.get_unchecked_mut(count) = i as u8;
            count += usize::from(misplaced(i));
            i += 1;
        }
    }
    count
}

/// Sorts `v` by insertion sort: each element is carried left past the greater ones before it.
fn insertion_sort<T: Ord>(v: &mut [T]) {
    for i in 1..v.len() {
        insert_last(&mut v[..=i]);
    }
}

/// Inserts the last element of `v` into the sorted rest of it.
fn insert_last<T: Ord>(v: &mut [T]) {
    let i = v.len() - 1;
    if v[i] >= v[i - 1] {
        return;
    }
    let base = v.as_mut_ptr();
    // SAFETY: every offset is below `v.len()`. The last element is read out, and `hole.dst` is
    // always the one vacant slot: each step moves the element before it into it. Comparisons
    // read the held element and elements in place; if one panics, dropping `hole` writes the
    // held element into the vacant slot, so `v` holds every element exactly once whenever the
    // block is left.
    unsafe {
        let last = ManuallyDrop::new(ptr::read(base.add(i)));
        ptr::copy_nonoverlapping(base.add(i - 1), base.add(i), 1);
        let mut hole = Hole {
            src: &*last,
            dst: base.add(i - 1),
        };
        while hole.dst > base && *last < *hole.dst.sub(1) {
            ptr::copy_nonoverlapping(hole.dst.sub(1), hole.dst, 1);
            hole.dst = hole.dst.sub(1);
        }
        // Dropping `hole` writes the held element into the vacant slot.
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
