use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};
use core::slice;

use super::counting::{
    Count, STACK_BYTES, Stack, bump, counters, exclusive_scan, in_order, in_order_at_first,
    key_slots, min_max, places4, range,
};
use super::key::IntegerKey;
use super::key::sealed::Bits;

/// Sorts a slice of integers, `char`s or `bool`s in place using pigeonhole sort.
///
/// Pigeonhole sort gives every possible key from the smallest to the largest a hole of its own,
/// and puts each element into the hole of its key: reading the holes out in order sorts the
/// slice, in `O(n + k)` time for `k = max − min + 1` keys. Here the holes are the slice itself:
/// the keys are counted, which gives every hole its place, and then each element out of place
/// is taken out and moved straight into the next free place of its hole, the element found there
/// taken out in turn, and so on around the cycle until an element lands in the place the first
/// one left. Every element is moved at most once, in place, with no buffer: `O(k)` memory.
///
/// Equal values cannot be told apart, so [`counting`](fn@super::counting) sort, which rewrites
/// the slice from the counts, sorts them with less work: 2.2–7 times faster on random values
/// and 1.4–2.25 times on nearly sorted ones. Pigeonhole sort is for [`pigeonhole_by_key`],
/// which moves elements of any type in place.
///
/// Seven refinements keep the algorithm as cheap as pigeonhole sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is scanned for one ascending
///   or one non-increasing run, 16 neighbours at a time; if it covers the whole slice, it is
///   left alone or reversed.
/// * **Large slices are placed by swaps.** Following a cycle, each step waits for the one
///   before it, and once the slice no longer fits in the caches, for memory. From 32,768
///   elements on (more for elements of 72 bytes or more), unless the input is mostly in order,
///   the free places of every hole are passed over instead, and the element at each swapped
///   with the next free place of its own hole, where it stays: the target of a swap depends on
///   its own key only, so the processor overlaps the swaps' waits for memory, though every
///   element moves twice. 1.1–4.3 times faster on random keys from 100,000 elements on. Two
///   holes count as one more element towards that length, as their counts take room in the
///   caches too (1.2–1.4 times faster from 20,000 elements on with four keys per element). The
///   counts become the starts of the holes in place, a hole ending where the next one starts,
///   and the holes with free places are listed without a branch on whether they have any:
///   1.35–1.4 times faster on random keys in a range of about `n` from 40,000 elements on, and
///   1.5–1.65 times in a range of `4n`, most of whose holes are empty.
/// * **A few distinct keys are counted in eight tables and placed by swaps.** With at most 256
///   keys and at least 16 elements per key, the keys are counted into eight tables, one after
///   the other into the next, so that the increments of a key repeating do not wait on each
///   other; with at most 16, they are placed by swaps from any length, as each step of a cycle
///   would wait for the free place the step before it just handed out in the same few holes:
///   2–3 times faster with four distinct keys up to 10,000 elements, 1.5 times beyond.
/// * **Input mostly in order walks the places, not the holes.** When at most one key in 16 is
///   less than the one before it, or the range has two keys per element or more, the cycles
///   start from each place in order, with no loop per hole, whose end mispredicts where holes
///   hold a few elements each: 3.5–4.5 times faster on nearly sorted input from 100 elements on,
///   1.05–1.15 times on random keys in a range of `4n` below 1000 elements. A hole's next free
///   place is moved on only by the cycles, which take the later of it and the walk's place.
///   When the first keys are mostly in order, they are also counted four at a time, all four
///   counts read before any is written back, so that a run of equal keys waits for memory once
///   per four: twice as fast on nearly sorted input with sixteen keys per hole.
/// * **Holes are counted in `u16` up to 65,535 elements,** and the next free place of each hole
///   needs no clearing; a range of more than 32 keys per element skips its empty holes eight
///   at a time: 1.1–2.2 times faster on a few elements over a wide range.
/// * **Counts on the stack.** Up to 4 KiB of counts need no allocation.
///
/// # Properties
///
/// * **Stable** in effect: equal values cannot be told apart.
/// * **In place**: `O(k)` counts, on the stack up to 4 KiB and otherwise allocated once.
/// * **Not adaptive**, except to sorted and reverse-sorted input, and to elements already in
///   their holes, which are passed over.
///
/// Against [`slice::sort_unstable`], for `u32` from 1000 elements on: 1.15–2.2 times faster on
/// random keys in a range of about `n` keys and 1.5–3.3 times of `n / 16`, 4–7 times on nearly
/// sorted input, 5 times on sorted input; from 1.25 times faster to 1.2 times slower on a range
/// of `4n` keys, where most holes are empty, and about as fast to 1.17 times slower with four
/// distinct keys, which the standard library's partitioning gains from. Against the textbook
/// pigeonhole sort (a vector per hole, the elements pushed in and read out): 2–6 times faster on
/// random keys and 160–340 times on sorted input.
///
/// # Panics
///
/// Panics if the range is too large to give every key a hole: if `max − min + 1` exceeds both
/// 65,536 and four times the length of the slice, whatever the order of the slice. The slice is
/// then left as it was. Any range of a one- or two-byte type is allowed.
///
/// # Examples
///
/// ```
/// use a::sort::pigeonhole;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// pigeonhole(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// let mut letters: Vec<char> = "pigeonhole".chars().collect();
/// pigeonhole(&mut letters);
/// assert_eq!(letters.iter().collect::<String>(), "eeghilnoop");
/// ```
///
/// A range too large for the slice panics:
///
/// ```should_panic
/// use a::sort::pigeonhole;
///
/// let mut v = [0u32, 1_000_000];
/// pigeonhole(&mut v);
/// ```
pub fn pigeonhole<T: IntegerKey>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    // Sorted input has nothing to move, but its range is checked all the same: the limit does
    // not depend on the order of the elements.
    if in_order(arr, |x, y| y < x) {
        range(arr[0], arr[len - 1], len, SORT);
        return;
    }
    // Equal values cannot be told apart, so a non-increasing slice only needs reversing.
    if in_order(arr, |x, y| y > x) {
        range(arr[len - 1], arr[0], len, SORT);
        arr.reverse();
        return;
    }
    let (min, max) = min_max(arr);
    let k = range(min, max, len, SORT);
    let none = NonNull::<()>::dangling().as_ptr();
    let keys = arr.as_mut_ptr();
    // SAFETY: `arr` holds `len` keys from `min`, within `k` of it; the elements are zero-sized.
    unsafe {
        if len <= u16::MAX as usize {
            place::<T, T, (), u16, false>(keys, none, len, min, k, |_| min);
        } else if len <= u32::MAX as usize {
            place::<T, T, (), u32, false>(keys, none, len, min, k, |_| min);
        } else {
            place::<T, T, (), usize, false>(keys, none, len, min, k, |_| min);
        }
    }
}

/// Sorts a slice in place by an integer, `char` or `bool` key using pigeonhole sort. The sort is
/// not stable: elements with equal keys may change their order.
///
/// The elements are moved into the holes of their keys as [`pigeonhole`] moves values, with the
/// same refinements, and with no buffer for them: each along the cycles of the permutation that
/// sorts them, at most once, or by swaps for large slices and few keys. The keys are kept in a
/// buffer of their own as they are read; `key` is called exactly once per element of a slice of
/// two or more elements that are not zero-sized, and once the last key is read no code of the
/// caller runs.
///
/// Two refinements are its own:
///
/// * **Holes move instead of keys.** As the keys are counted, each is rewritten in their buffer
///   as its hole, in the fewest bytes that hold every hole of the range (one for 256 keys, two
///   for 65,536), and the holes move with the elements: 1.2–1.8 times faster on random keys of
///   16- to 72-byte elements at 10,000 elements, as the keys and elements of the slice then fit
///   the caches.
/// * **Non-increasing input only needs reversing.** The sort is not stable, so a slice whose
///   keys never increase is reversed, as one whose keys never decrease is left alone.
///
/// Together, against the first version of this sort (cycles only, the keys moving with their
/// elements): 1.2–3.3 times faster on random keys of up to 72-byte elements from 100,000
/// elements on (1.0–1.5 times for 128-byte ones), 1.4–2.2 times on a few elements over a wide
/// range, and as fast below 10,000 elements; nearly sorted input from 100,000 elements on,
/// which took 1.05–2 times as long, has since become 3–4 times faster than it was with the
/// first version, walked by places.
///
/// # Properties
///
/// * **Not stable**: elements with equal keys may change their order.
/// * **In place** for the elements: a buffer of `n` keys, rewritten as holes, and `O(k)` counts,
///   each on the stack up to 4 KiB and otherwise allocated once.
/// * **Not adaptive**, except to sorted and non-increasing input, and to elements already in
///   their holes, which are passed over.
///
/// Against [`slice::sort_unstable_by_key`], from 100 elements on: 1.05–2.8 times faster on
/// random `u64` keys of 16-byte elements up to 10,000 elements and 1.35–1.45 times beyond,
/// 1.7–5 times for 72-byte elements and 1.5–4.6 times for 128-byte ones, 2–16 times on nearly
/// sorted input of up to 72-byte elements; but 1.2–2.1 times slower on sorted and
/// reverse-sorted input, which the standard library recognises in one comparison per element,
/// and 1.1–1.3 times slower with four distinct keys of 16-byte elements. Against
/// [`counting_by_key`](fn@super::counting_by_key), which is stable and moves the elements
/// through a buffer: 1.15–3.1 times slower on random keys, but 1.2–2.35 times faster on nearly
/// sorted input of 72- and 128-byte elements, with no buffer for them.
///
/// # Panics
///
/// Panics if the range of the keys is too large, as [`pigeonhole`] does, leaving the slice as it
/// was. If `key` panics, the panic is propagated and the slice is left as it was: no element has
/// been moved before the last key is read.
///
/// # Examples
///
/// ```
/// use a::sort::pigeonhole_by_key;
///
/// let mut v = [(3, 'a'), (1, 'b'), (0, 'c'), (2, 'd')];
/// pigeonhole_by_key(&mut v, |&(k, _)| k);
/// assert_eq!(v, [(0, 'c'), (1, 'b'), (2, 'd'), (3, 'a')]);
///
/// // Words by length; words of equal length may come in any order.
/// let mut words = vec!["pear", "fig", "banana", "apple"];
/// pigeonhole_by_key(&mut words, |w| w.len());
/// assert_eq!(words.iter().map(|w| w.len()).collect::<Vec<_>>(), [3, 4, 5, 6]);
/// ```
pub fn pigeonhole_by_key<T, K, F>(arr: &mut [T], mut key: F)
where
    K: IntegerKey,
    F: FnMut(&T) -> K,
{
    let len = arr.len();
    if len < 2 || mem::size_of::<T>() == 0 {
        return;
    }
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let slots = key_slots::<K>(len, &mut stack, &mut heap);
    let first = key(&arr[0]);
    slots[0].write(first);
    let (mut min, mut max, mut prev) = (first, first, first);
    let (mut descent, mut ascent) = (false, false);
    for (slot, x) in slots[1..].iter_mut().zip(&arr[1..]) {
        let x = key(x);
        slot.write(x);
        min = min.min(x);
        max = max.max(x);
        descent |= x < prev;
        ascent |= x > prev;
        prev = x;
    }
    let k = range(min, max, len, SORT);
    if !descent {
        return;
    }
    // Not stable: a non-increasing slice only needs reversing.
    if !ascent {
        arr.reverse();
        return;
    }
    let (keys, elems) = (slots.as_mut_ptr().cast::<K>(), arr.as_mut_ptr());
    // SAFETY: all `len` keys have been written, from `min`, within `k` of it, one per element of
    // `arr`. A type of `b` bits has at most `2^b` keys, so the holes fit in a type no larger than
    // the keys'.
    unsafe {
        if k <= 1 << 8 {
            by_holes::<K, u8, T>(keys, elems, len, min, k);
        } else if k <= 1 << 16 {
            by_holes::<K, u16, T>(keys, elems, len, min, k);
        } else if k as u64 <= 1 << 32 {
            by_holes::<K, u32, T>(keys, elems, len, min, k);
        } else {
            by_holes::<K, usize, T>(keys, elems, len, min, k);
        }
    }
}

/// Places the elements by the holes of their keys, as `H`, which [`place`] writes over the keys
/// as it counts them.
///
/// # Safety
///
/// As for [`place`]; `H` must hold every hole below `k`, and be no larger than `K`.
#[inline(always)]
unsafe fn by_holes<K: IntegerKey, H: IntegerKey + Bits, T>(
    keys: *mut K,
    elems: *mut T,
    len: usize,
    min: K,
    k: usize,
) {
    debug_assert!(mem::size_of::<H>() <= mem::size_of::<K>());
    // SAFETY: guaranteed by the caller.
    unsafe {
        if len <= u16::MAX as usize {
            place::<K, H, T, u16, true>(keys, elems, len, min, k, H::of);
        } else if len <= u32::MAX as usize {
            place::<K, H, T, u32, true>(keys, elems, len, min, k, H::of);
        } else {
            place::<K, H, T, usize, true>(keys, elems, len, min, k, H::of);
        }
    }
}

/// The name a range too large panics with.
const SORT: &str = "pigeonhole sort";

/// Elements are placed by [`swaps`] rather than along [`cycles`] from this many on, which no
/// longer fit in the processor's fast caches...
const SWAP_MIN_LEN: usize = 1 << 15;

/// ...times one for every this many bytes of an element and its key (elements of 72 bytes from
/// twice as many, of 128 bytes from three times as many), as swapping moves each element twice.
const SWAP_BYTES_PER_STEP: usize = 40;

/// ...where this many holes count as one more element, as their counters take room in the
/// caches too: a range of four keys per element is placed by swaps from 16,384 elements on,
/// 1.2–1.4 times faster from 20,000 elements on than along the cycles.
const SWAP_HOLES_PER_ELEMENT: usize = 2;

/// ...when more than one in this many keys is less than the one before it: input mostly in
/// order is placed faster along the cycles, which pass over the elements already in place.
const SWAP_DESCENTS_PART: usize = 16;

/// Elements are also placed by swaps, at any length, when the range has at most this many
/// keys...
const SWAP_MAX_RANGE: usize = 16;

/// ...and the slice at least this many elements per key: following a cycle then waits for the
/// next free place of a hole that the step before it, most likely in the same hole, has just
/// handed out, while the target of a swap depends on its own key only. 2–3 times faster with
/// four distinct keys from 100 elements on; 16 elements over 16 keys took 1.5 times as long.
const SWAP_MIN_PER_KEY: usize = 16;

/// [`swaps`] swaps an element with its target without checking that they differ while the
/// element and its hole take at most this many bytes.
const SWAP_ALWAYS_BYTES: usize = 16;

/// The cycles are walked by places rather than by holes when the input is mostly in order (see
/// [`SWAP_DESCENTS_PART`]), or the range has at least this many keys per element, most holes
/// being empty (1.05–1.2 times faster with four keys per element).
const POSITION_MIN_PER_ELEMENT: usize = 2;

/// A range of more than this many keys per element skips empty holes eight at a time.
const SPARSE_PER_ELEMENT: usize = 32;

/// Moves the `len` keys at `keys`, and their elements at `elems` with them, into the holes of
/// the `k` keys from `min`, in place.
///
/// The keys are counted per hole first, and the counts turned into the place where each hole
/// starts and ends. If `NARROW`, the keys are rewritten as they are counted, from the start of
/// their buffer, as their holes of type `H` (`hole(h)`), which then move instead of the keys:
/// moving a byte or two instead of the key made random keys of 16-byte elements up to 1.9 times
/// faster.
///
/// # Safety
///
/// `keys` and `elems` must hold `len` initialized keys and elements, apart from each other, and
/// every key must lie from `min` to less than `k` above it. If `NARROW`, `H` must hold every
/// hole and be no larger than `K`; otherwise `H` must be `K`.
unsafe fn place<K: IntegerKey, H: IntegerKey, E, C: Count, const NARROW: bool>(
    keys: *mut K,
    elems: *mut E,
    len: usize,
    min: K,
    k: usize,
    hole: impl Fn(usize) -> H,
) {
    let tables = if k <= TABLES_MAX_RANGE && len >= TABLES_MIN_PER_KEY * k {
        TABLES
    } else {
        1
    };
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    // The count of each hole, which becomes its end.
    let ends = counters::<C>(tables * k, &mut stack, &mut heap).as_mut_ptr();
    let holes = keys.cast::<H>();
    let sparse = k > SPARSE_PER_ELEMENT * len;
    // SAFETY: guaranteed by the caller: every key's hole is below `k`, so its counters are in
    // `ends` and `next`. The hole of key `i` is written over bytes before the end of key `i`,
    // once it and the keys before it have been read; the buffer is aligned for `K`, and so for
    // `H`, which is no larger. `next` is written for every hole that holds keys before it is
    // read.
    unsafe {
        let descents = if tables == TABLES {
            count::<K, H, C, NARROW, TABLES>(keys, len, min, k, ends, &hole)
        } else {
            // Four keys at a time, their counters all read before any is written back, so that
            // a run of equal keys waits for memory once per four keys, when the first keys are
            // mostly in order: twice as fast on nearly sorted input with sixteen keys per hole,
            // but on random keys 1.02–1.05 times as long as one at a time. Written out here:
            // through `count`, nearly sorted input of 10,000 16-byte elements took 1.1 times as
            // long.
            let mut descents = 0;
            let mut prev = 0;
            if in_order_at_first(slice::from_raw_parts(keys, len)) {
                let mut i = 0;
                while i + 4 <= len {
                    let h = [0, 1, 2, 3].map(|j| (*keys.add(i + j)).offset(min));
                    for (j, &h) in h.iter().enumerate() {
                        if NARROW {
                            holes.add(i + j).write(hole(h));
                        }
                        descents += usize::from(h < prev);
                        prev = h;
                    }
                    let p = places4(h, h.map(|h| (*ends.add(h)).get()));
                    for j in 0..4 {
                        *ends.add(h[j]) = C::ZERO.add(p[j] + 1);
                    }
                    i += 4;
                }
                for i in i..len {
                    let h = (*keys.add(i)).offset(min);
                    if NARROW {
                        holes.add(i).write(hole(h));
                    }
                    descents += usize::from(h < prev);
                    prev = h;
                    bump(ends.add(h));
                }
            } else {
                for i in 0..len {
                    let h = (*keys.add(i)).offset(min);
                    if NARROW {
                        holes.add(i).write(hole(h));
                    }
                    descents += usize::from(h < prev);
                    prev = h;
                    bump(ends.add(h));
                }
            }
            descents
        };
        // The key whose hole is 0: `min`, or the hole 0 itself.
        let base = if NARROW {
            hole(0)
        } else {
            ptr::read((&raw const min).cast::<H>())
        };
        let ordered = descents * SWAP_DESCENTS_PART <= len;
        let by_swaps = !ordered
            && (by_swaps::<H, E>(len, k) || (k <= SWAP_MAX_RANGE && len >= SWAP_MIN_PER_KEY * k));
        if by_swaps && !sparse {
            // The counts become the starts of the holes in place, the next free place of each,
            // and a hole ends where the next one starts: no second array of counters.
            exclusive_scan(slice::from_raw_parts_mut(ends, k));
            return swaps::<H, E, C, true>(holes, elems, base, k, len, ends, ends);
        }
        let mut next_stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
        let mut next_heap = Vec::new();
        // The next free place of each hole; never read for an empty one, so it needs no
        // clearing.
        let next = key_slots::<C>(k, &mut next_stack, &mut next_heap)
            .as_mut_ptr()
            .cast::<C>();
        let mut sum = 0;
        if sparse {
            each_hole(ends, k, |h, count| {
                *next.add(h) = C::ZERO.add(sum);
                sum += count;
                *ends.add(h) = C::ZERO.add(sum);
            });
        } else {
            // Four holes at a time, summed among themselves before the sum of those before
            // them is added: one addition per four holes waits for the one before it.
            let mut h = 0;
            while h + 4 <= k {
                let c = [0, 1, 2, 3].map(|j| (*ends.add(h + j)).get());
                let (ab, abc) = (c[0] + c[1], c[0] + c[1] + c[2]);
                let starts = [sum, sum + c[0], sum + ab, sum + abc];
                sum += abc + c[3];
                for j in 0..4 {
                    *next.add(h + j) = C::ZERO.add(starts[j]);
                    *ends.add(h + j) = C::ZERO.add(starts[j] + c[j]);
                }
                h += 4;
            }
            for h in h..k {
                *next.add(h) = C::ZERO.add(sum);
                sum += (*ends.add(h)).get();
                *ends.add(h) = C::ZERO.add(sum);
            }
        }
        if by_swaps {
            // Only a sparse range is left to place by swaps here.
            swaps::<H, E, C, false>(holes, elems, base, k, len, ends, next);
        } else {
            let walk = if sparse {
                Walk::Sparse
            } else if ordered || k >= POSITION_MIN_PER_ELEMENT * len {
                Walk::Places(len)
            } else {
                Walk::Holes
            };
            cycles(holes, elems, base, k, ends, next, walk);
        }
    }
}

/// Keys are counted into eight tables, one key after the other into the next, while the range
/// has at most this many keys...
const TABLES_MAX_RANGE: usize = 256;

/// ...and the slice has at least this many elements per key: a run of equal keys, or a few
/// keys repeating, then makes eight chains of increments at once instead of one.
const TABLES_MIN_PER_KEY: usize = 16;

const TABLES: usize = 8;

/// Counts the holes of the `len` keys at `keys`, all within `k` of `min`, into `N` tables of
/// `k` counters at `ends`, interleaved (the key at `i` into counter `h·N + i mod N`), and adds
/// them up into the first `k` counters. If `NARROW`, writes the hole of each key over the keys
/// as it goes, as `hole(h)`. Returns how many keys have a hole less than the one before.
///
/// # Safety
///
/// As for [`place`], with `ends` holding `N·k` zeroed counters.
#[inline(always)]
unsafe fn count<K: IntegerKey, H, C: Count, const NARROW: bool, const N: usize>(
    keys: *mut K,
    len: usize,
    min: K,
    k: usize,
    ends: *mut C,
    hole: &impl Fn(usize) -> H,
) -> usize {
    let holes = keys.cast::<H>();
    let mut descents = 0;
    let mut prev = 0;
    // SAFETY: guaranteed by the caller; the hole of key `i` is written once it and the keys
    // before it have been read, as in `place`.
    unsafe {
        let mut i = 0;
        while i + N <= len {
            for j in 0..N {
                let h = (*keys.add(i + j)).offset(min);
                if NARROW {
                    holes.add(i + j).write(hole(h));
                }
                descents += usize::from(h < prev);
                prev = h;
                bump(ends.add(h * N + j));
            }
            i += N;
        }
        for i in i..len {
            let h = (*keys.add(i)).offset(min);
            if NARROW {
                holes.add(i).write(hole(h));
            }
            descents += usize::from(h < prev);
            prev = h;
            bump(ends.add(h * N));
        }
        if N > 1 {
            for h in 0..k {
                let mut sum = 0;
                for t in 0..N {
                    sum += (*ends.add(h * N + t)).get();
                }
                // `h <= h·N`: the counters of the holes after `h` are still to be read.
                *ends.add(h) = C::ZERO.add(sum);
            }
        }
    }
    descents
}

/// Whether `len` elements of type `E` with keys of type `K`, in a range of `k` keys, are placed
/// by [`swaps`].
fn by_swaps<K, E>(len: usize, k: usize) -> bool {
    let bytes = mem::size_of::<E>() + mem::size_of::<K>();
    len.max(k / SWAP_HOLES_PER_ELEMENT) >= SWAP_MIN_LEN * (bytes / SWAP_BYTES_PER_STEP).max(1)
}

/// Calls `f` with each hole whose counter is not zero, in order, and its counter, skipping eight
/// zeros at a time with one check: for ranges of many more keys than elements. Dense ranges
/// walk every hole in a plain loop instead: a branch on whether a hole is empty mispredicts
/// where about a third of them are (keys spread over a range of about `n`: twice as slow), and
/// the walk through a closure made nearly sorted input 10% slower than the loop.
///
/// # Safety
///
/// `counters` must hold `k` counters, and `f` may only write the counter of the hole it is
/// given.
#[inline(always)]
unsafe fn each_hole<C: Count>(counters: *mut C, k: usize, mut f: impl FnMut(usize, usize)) {
    // SAFETY: guaranteed by the caller; every counter read is below `k`, and a block is read
    // by value, so that `f` may write its counters.
    unsafe {
        let mut h = 0;
        while h + 8 <= k {
            let block = ptr::read(counters.add(h).cast::<[C; 8]>());
            if block.iter().any(|&c| c != C::ZERO) {
                for (j, c) in block.iter().enumerate() {
                    if *c != C::ZERO {
                        f(h + j, c.get());
                    }
                }
            }
            h += 8;
        }
        while h < k {
            let c = (*counters.add(h)).get();
            if c != 0 {
                f(h, c);
            }
            h += 1;
        }
    }
}

/// Fills the holes one after the other: each element out of the place of its hole is taken
/// out and put in the next free place of its own hole, the element there taken out in turn, and
/// so on around the cycle until an element belongs where the first was. Each element is read
/// and written once, but each step waits for the one before it, which waits for memory once the
/// slice no longer fits in the caches. The cycles start from the holes or the places in order,
/// as `walk` says.
///
/// # Safety
///
/// As for [`place`], with `next` the start of every hole that holds keys, and `ends` the end
/// of every hole: for [`Walk::Sparse`], of every hole that holds keys, and zero for the others.
unsafe fn cycles<H: IntegerKey, E, C: Count>(
    holes: *mut H,
    elems: *mut E,
    base: H,
    k: usize,
    ends: *mut C,
    next: *mut C,
    walk: Walk,
) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        match walk {
            Walk::Sparse => each_hole(ends, k, |h, end| fill(holes, elems, base, next, h, end)),
            Walk::Places(len) => walk_places(holes, elems, base, len, ends, next),
            Walk::Holes => {
                for h in 0..k {
                    fill(holes, elems, base, next, h, (*ends.add(h)).get());
                }
            }
        }
    }
}

/// How [`cycles`] walks the holes.
#[derive(Clone, Copy)]
enum Walk {
    /// Hole by hole, skipping eight empty ones at a time: for a sparse range.
    Sparse,
    /// Hole by hole, for random input in a dense range.
    Holes,
    /// Place by place, over the given number of places (see [`walk_places`]): for input mostly
    /// in order, or a range of several keys per element.
    Places(usize),
}

/// Fills the holes as [`cycles`] does, walking the places in order instead of the holes: an
/// element in the place of its hole stays, and one out of it starts a cycle, which ends when an
/// element belongs where it started. Walking the holes, each with its own loop over its free
/// places, mispredicts the end of each loop where the holes hold a few elements each and most
/// are in place, as in nearly sorted input (3–3.5 times as slow from 100 elements on), and goes
/// through every empty hole; walking the places has no such loop, and passes over no hole, but
/// looks at the places that cycles filled, which, in random order, the walk of the holes passes
/// over: 1.1–1.2 times as slow on random keys in a range of about `n`, 1.65 times with sixteen
/// elements per hole.
///
/// # Safety
///
/// As for [`cycles`], for a range that is not sparse: `ends` holds the end of every hole, the
/// start of the one after it, and `next` the start of every hole that holds keys.
#[inline(always)]
unsafe fn walk_places<H: IntegerKey, E, C: Count>(
    holes: *mut H,
    elems: *mut E,
    base: H,
    len: usize,
    ends: *mut C,
    next: *mut C,
) {
    // SAFETY: guaranteed by the caller. Every place before `i` holds its element for good, and
    // the next free place of a hole whose places include `i` or come after it is the later of
    // its `next` and `i`: an element that stays where it is does not move `next` on, and a cycle
    // puts an element at the next free place of its hole and moves `next` past it, so every
    // place from the later of the two on is free, and every place of the hole before `i` is
    // not. A cycle from `i` takes out the element there, moves each element it takes out to the
    // next free place of its hole, which is after `i` until the hole of `i` comes, and that one
    // takes `i`: every element moves at most once, and is written before anything else is read
    // or written over it, with no code in between that can panic.
    unsafe {
        for i in 0..len {
            let mut x = *holes.add(i);
            let mut d = x.offset(base);
            let start = if d == 0 { 0 } else { (*ends.add(d - 1)).get() };
            if start <= i && i < (*ends.add(d)).get() {
                // In its hole, for good. Its hole's `next` is left behind rather than moved past
                // it: a run of elements in one hole made each move wait for the one before it,
                // through memory (sixteen keys per hole, nearly sorted: 1.4 times as slow).
                continue;
            }
            let mut cur = ptr::read(elems.add(i).cast::<MaybeUninit<E>>());
            loop {
                let n = next.add(d);
                let j = (*n).get().max(i);
                *n = C::ZERO.add(j + 1);
                if j == i {
                    break;
                }
                let y = *holes.add(j);
                let taken = ptr::read(elems.add(j).cast::<MaybeUninit<E>>());
                *holes.add(j) = x;
                ptr::write(elems.add(j).cast::<MaybeUninit<E>>(), cur);
                x = y;
                cur = taken;
                d = x.offset(base);
            }
            *holes.add(i) = x;
            ptr::write(elems.add(i).cast::<MaybeUninit<E>>(), cur);
        }
    }
}

/// Fills hole `h`, which ends at `end`, as [`cycles`] does.
///
/// # Safety
///
/// As for [`cycles`].
#[inline(always)]
unsafe fn fill<H: IntegerKey, E, C: Count>(
    holes: *mut H,
    elems: *mut E,
    base: H,
    next: *mut C,
    h: usize,
    end: usize,
) {
    // SAFETY: guaranteed by the caller; the places of each hole are handed out once each, from
    // its start to its end, and the element taken out at each is written to one before anything
    // else, with no code in between that can panic.
    unsafe {
        let mut i = (*next.add(h)).get();
        while i < end {
            let mut x = *holes.add(i);
            let mut d = x.offset(base);
            if d != h {
                let mut cur = ptr::read(elems.add(i).cast::<MaybeUninit<E>>());
                loop {
                    let j = bump(next.add(d));
                    let y = *holes.add(j);
                    let taken = ptr::read(elems.add(j).cast::<MaybeUninit<E>>());
                    *holes.add(j) = x;
                    ptr::write(elems.add(j).cast::<MaybeUninit<E>>(), cur);
                    x = y;
                    cur = taken;
                    d = x.offset(base);
                    if d == h {
                        break;
                    }
                }
                *holes.add(i) = x;
                ptr::write(elems.add(i).cast::<MaybeUninit<E>>(), cur);
            }
            i += 1;
        }
    }
}

/// Fills the holes by passes over the free places of every hole that has any: the element at
/// each is swapped with the one at the next free place of its own hole, where it stays. Every
/// step places one element for good, and its target depends on its own key only, not on the
/// step before it, so the processor overlaps the steps' waits for memory: from about 32,768
/// elements on, up to 3 times faster than [`cycles`], though every element moves twice.
///
/// If `STARTS`, `next` holds the start of every hole, the end of the one before it (the last
/// one ends at `total`), and `ends` is not read; otherwise the range is sparse, and `ends` holds
/// the end of every hole that holds keys, and zero for the others.
///
/// # Safety
///
/// As for [`cycles`], with the `total` elements in the holes.
unsafe fn swaps<H: IntegerKey, E, C: Count, const STARTS: bool>(
    holes: *mut H,
    elems: *mut E,
    base: H,
    k: usize,
    total: usize,
    ends: *mut C,
    next: *mut C,
) {
    // The holes with free places, and where each ends: at most one per hole and one per
    // element, and one more written past them; on the stack up to 4 KiB.
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let open = key_slots::<(usize, usize)>(k.min(total) + 1, &mut stack, &mut heap)
        .as_mut_ptr()
        .cast::<(usize, usize)>();
    let mut len = 0;
    // SAFETY: guaranteed by the caller; a hole's free places run from its next free place to
    // its end, every element's hole has one as long as it is not placed, and a swap of two
    // places, apart or the same, leaves both elements whole. A hole with free places holds an
    // element, so `open` has room for every one, and for the hole written after them; its
    // first `len` entries are written.
    unsafe {
        // Every hole is written at the end of the list, which only takes it if it has free
        // places: a branch on that mispredicts where many holes are empty, about a quarter of
        // them full with keys spread over a range of four per element (1.6 times as slow).
        let mut add = |h: usize, end: usize| {
            open.add(len).write((h, end));
            len += usize::from((*next.add(h)).get() < end);
        };
        if STARTS {
            // `next` holds the start of every hole, where the hole before it ends.
            for h in 1..k {
                add(h - 1, (*next.add(h)).get());
            }
            add(k - 1, total);
        } else {
            each_hole(ends, k, add);
        }
        while len > 0 {
            let mut kept = 0;
            for o in 0..len {
                let (h, end) = *open.add(o);
                let mut p = (*next.add(h)).get();
                while p < end {
                    let t = bump(next.add((*holes.add(p)).offset(base)));
                    // Elements of up to 16 bytes with their holes are swapped even with
                    // themselves, which costs less than a branch that mispredicts.
                    if mem::size_of::<E>() + mem::size_of::<H>() <= SWAP_ALWAYS_BYTES || t != p {
                        ptr::swap(holes.add(p), holes.add(t));
                        ptr::swap(elems.add(p), elems.add(t));
                    }
                    p += 1;
                }
                if (*next.add(h)).get() < end {
                    *open.add(kept) = (h, end);
                    kept += 1;
                }
            }
            len = kept;
        }
    }
}
