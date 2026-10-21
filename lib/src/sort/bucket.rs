use core::hint::select_unpredictable;
use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};
use core::slice;

use super::counting::{Count, STACK_BYTES, Stack, bump, counters, in_order, key_slots, min_max};
use super::key::BucketKey;
use super::key::sealed::{Bits, Bucket};
use super::tim::Buffer;

/// Sorts a slice of integers, `char`s, `bool`s or floats in place using bucket sort.
///
/// Bucket sort does not compare elements to place them: it cuts the range from the smallest to
/// the largest key into buckets of equal width, moves every element into the bucket its key
/// falls in, sorts each bucket, and reads the buckets out in order. CLRS's version sorts `n`
/// reals spread uniformly over `[0, 1)` into `n` buckets of about one element each, sorted by
/// insertion, in `O(n)` expected time. Here a level cuts its range into one bucket per element,
/// but at most 4096, so that the counts stay in the fastest caches, and moves the elements into
/// their buckets with a stable counting pass; runs of buckets of at most 16 elements are sorted
/// by one insertion sort, and larger buckets are cut again over their own range. Floats are
/// sorted as [`f64::total_cmp`] orders them, and cut by value at the first level; integers, and
/// floats below it, by their bits, which bounds the depth by the width of the key: skewed input
/// cannot make it quadratic, as it makes the textbook version.
///
/// Eight refinements keep the algorithm as cheap as bucket sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or one non-increasing run, 16 neighbours at a time with no exit inside a block;
///   if it covers the whole slice, the slice is left alone or reversed.
/// * **Keys are read once, as bits that order as they do.** Signed integers and floats are
///   rewritten in place, in one pass that also finds their range, as unsigned integers with
///   the sign flipped (every bit, for a negative float), and back at the end; every comparison
///   is then one of integers.
/// * **Floats crowded at zero are cut by their bits.** Values spread over many orders of
///   magnitude crowd in the buckets nearest zero when cut by value, which then only peels off
///   the largest ones level after level; their bits, which grow with the exponent, spread them.
///   When the three buckets nearest zero hold more than half the keys, the level is counted
///   again by bits and moved by the cut with the smaller largest bucket: 2.6–3 times faster on
///   floats of random exponents, 3–5 times on powers of two.
/// * **Runs of keys in one bucket do not wait on each other.** A count depends on the one
///   before it in the same bucket, so a sorted stretch or a few distinct keys make a chain of
///   increments, each waiting for the last to reach memory. Keys are counted and placed four at
///   a time, all four counters read before any is written back, and a level with at least four
///   keys per bucket counts into four interleaved tables, from 16 into eight: 1.8–3 times faster
///   on powers of two, 1.3–2 times with four distinct keys, 1.7 times on nearly sorted input.
/// * **Buckets are found once.** A level of at most 2048 keys keeps the bucket of each key as
///   it counts it, on the stack, for the pass that moves it.
/// * **A level of one bucket per key is sorted as it is moved.** When there are fewer keys in
///   the range than buckets, each bucket holds equal keys, and no insertion follows: 1.5 times
///   faster on keys below `n`.
/// * **Insertion without a branch for keys that move.** Keys out of small buckets that move
///   mostly move one place, about half of them, which a branch mispredicts: each chunk of 32 is
///   sorted with a branch if at most one key of the chunk before moved, and otherwise by
///   writing the larger and the smaller of a key and its neighbour without one, two places for
///   keys of up to 8 bytes that often move further: 1.2–1.8 times faster on uniform keys.
/// * **No allocation for small sorts.** Up to 16 elements are sorted by insertion at once, and
///   a buffer and counts of up to 4 KiB each live on the stack.
///
/// Together, against the first version of this sort (the same levels with one count per bucket
/// and a branching insertion sort): 1.2–1.8 times faster on random and uniform integers, 1.3–1.8
/// times on uniform floats and 2.6–3.2 times on `f64`s of random exponents, 1.3–2 times with
/// four distinct keys, 1.7–1.9 times on nearly sorted input, and 1.8–5 times on powers of two.
/// Against the textbook version (`n` buckets as vectors, each sorted by insertion,
/// concatenated): 6.8–11 times faster on random and uniform `u64`s, 4.9–8 times on uniform
/// `f64`s, 3–4.4 times with four distinct keys, and 55–180 times on sorted and reverse-sorted
/// input; on skewed input the textbook version is quadratic (170 times as long on 10 000 `f64`s
/// of random exponents).
///
/// # Properties
///
/// * **Stable** in effect: keys equal in this order have equal bits, and cannot be told apart.
/// * **Not in place**: a buffer of `n` keys and the counts of a level, each on the stack up to
///   4 KiB and otherwise allocated.
/// * **Not adaptive**, except to sorted and reverse-sorted input, to keys in a range smaller
///   than the slice, and to nearly sorted input, whose keys mostly stay in place.
///
/// | Case                              | Levels per element                  | Time     |
/// |-----------------------------------|-------------------------------------|----------|
/// | Sorted, reverse-sorted, equal     | none: one scan, reversing           | `O(n)`   |
/// | Keys spread evenly over the range | one or two, buckets of a few keys   | `O(n)`   |
/// | Any keys of `w` bits              | at most `⌈w / 4⌉`, plus one (floats) | `O(n·w)` |
///
/// A level divides the range of its keys by its number of buckets, at least 17 as it has more
/// than 16 keys, and up to 4096: while an element's bucket holds more than 4096 keys, a level
/// takes 12 bits off its range, so 64-bit keys need at most six levels of that size.
///
/// Against [`slice::sort_unstable`] (`sort_unstable_by(f64::total_cmp)` for floats), from 100
/// elements on: 1.5–3 times faster on random and uniform integers, 2–3.3 times on uniform
/// floats, 1.4–3 times on `f64`s of random exponents, 2.2–5 times on keys below `n`, 1.3–4.5
/// times on nearly sorted input, 1.7–5 times on sorted input and 1.3–2.9 times on
/// reverse-sorted input. Its weaknesses are keys the equal-width buckets do not spread, where
/// the standard library gains from few distinct values: powers of two take 1.4–3.4 times as
/// long, four distinct keys 1.2–1.6 times as long from 1000 elements on, and `f32`s of random
/// exponents, which are mostly zeros and infinities, up to 1.8 times as long; 16 elements take
/// about as long. For integers, [`radix`](fn@super::radix) sort,
/// which does not depend on how the keys are spread, is faster on large slices of random keys.
///
/// # Examples
///
/// ```
/// use a::sort::bucket;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// bucket(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// let mut v = [0.42, -1.5, 0.0, f64::INFINITY, -0.0, 0.17];
/// bucket(&mut v);
/// assert_eq!(v, [-1.5, -0.0, 0.0, 0.17, 0.42, f64::INFINITY]);
/// ```
pub fn bucket<T: BucketKey>(arr: &mut [T]) {
    const {
        assert!(
            mem::size_of::<T>() == mem::size_of::<T::Bits>()
                && mem::align_of::<T>() == mem::align_of::<T::Bits>()
        )
    };
    let len = arr.len();
    if len < 2 {
        return;
    }
    if in_order(arr, |x, y| y.ordered() < x.ordered()) {
        return;
    }
    // Keys equal in this order have the same bits, so a non-increasing slice only needs
    // reversing.
    if in_order(arr, |x, y| y.ordered() > x.ordered()) {
        arr.reverse();
        return;
    }
    // The keys are converted in place, and their range found on the way, which spares the
    // first level a pass to find it.
    let mut bounds = None;
    // For a slice short enough to be sorted by insertion at once: how many keys are less than
    // the one before them, counted on the way (see `small_mode`).
    let mut descents = None;
    if !T::IDENTITY && len <= LEAF {
        let (mut prev, mut n) = (arr[0].ordered(), 0);
        for x in arr.iter_mut() {
            let bits = x.ordered();
            n += usize::from(bits < prev);
            prev = bits;
            // SAFETY: as below.
            unsafe { ptr::from_mut(x).cast::<T::Bits>().write(bits) };
        }
        descents = Some(n);
    } else if !T::IDENTITY {
        let first = arr[0].ordered();
        let (mut min, mut max) = (first, first);
        for x in arr.iter_mut() {
            let bits = x.ordered();
            min = min.min(bits);
            max = max.max(bits);
            // SAFETY: `T::Bits` has the size and alignment of `T`, and every bit pattern of
            // both is a valid value (`T` is an integer or a float, as `char` and `bool` are
            // `IDENTITY`).
            unsafe { ptr::from_mut(x).cast::<T::Bits>().write(bits) };
        }
        bounds = Some((min, max));
    }
    let p = arr.as_mut_ptr();
    let keys = p.cast::<T::Bits>();
    let none = NonNull::<()>::dangling().as_ptr();
    let v = Area { keys, elems: none };
    if len <= LEAF {
        // As `sort` would, without setting up the scratch space it does not need.
        // SAFETY: `v` holds `len` keys; the elements are zero-sized.
        unsafe { insertion(v, len, &mut small_mode(descents, len)) };
    } else {
        let mut buf = Buffer::<T::Bits>::new(len);
        let scratch = Area {
            keys: buf.get(),
            elems: none,
        };
        // SAFETY: `v` holds `len` keys, and `scratch` has room for `len` keys, apart from them;
        // the elements are zero-sized.
        unsafe {
            if len <= u32::MAX as usize {
                sort::<T, (), u32, false>(v, scratch, len, 0, &mut Mode::One, bounds);
            } else {
                sort::<T, (), usize, false>(v, scratch, len, 0, &mut Mode::One, bounds);
            }
        }
    }
    if !T::IDENTITY {
        for i in 0..len {
            // SAFETY: as above; the slot holds the bits of a key.
            unsafe {
                let bits = keys.add(i).read();
                p.add(i).write(T::from_ordered(bits));
            }
        }
    }
}

/// Sorts a slice in place by an integer, `char`, `bool` or float key using bucket sort. The sort
/// is stable: elements with equal keys keep their order.
///
/// The keys are bucketed as [`bucket`] buckets values, with the same refinements, and the
/// elements move with them. `key` is called exactly once per element of a slice of two or more
/// elements that are not zero-sized (a shorter slice, or one of zero-sized elements, is left as
/// it is without calling it): the keys are kept in a buffer of their own as they are read, and
/// once the last key is read no code of the caller runs.
///
/// Four refinements are its own:
///
/// * **Sorted and non-increasing input is recognised**, in the pass that reads the keys: a
///   sorted slice is left alone, and a non-increasing one reversed, and then each run of equal
///   keys reversed back, which keeps their order: 3–8 times faster on descending input with
///   duplicates.
/// * **Keys are read by a loop fit for the size of the elements.** Elements of up to 16 bytes
///   by one that tracks both orders and never leaves early, which the compiler vectorizes;
///   larger ones by loops that leave at the first key that ends the order they follow (1.2–1.4
///   times faster on sorted input of 32-byte elements).
/// * **Large elements move once.** A part of the slice of elements of at least 64 bytes that
///   takes at least 64 KiB, and has at most 8192 elements, is sorted by the indices of its
///   elements, which then move once each, along the cycles of that permutation, with no buffer
///   for them: 1.1–1.25 times faster for 72- and 128-byte elements, 1.5–2 times on nearly
///   sorted input.
/// * **Elements over 16 bytes that keys pass often are moved by position.** When more than one
///   in eight keys of a run of small buckets is less than the one before it, the positions of
///   its elements are sorted with the keys, and the elements then moved once each.
///
/// Together, against the first version of this sort: 1.3–1.4 times faster on random `u64` keys
/// of 16-byte elements from 100 to 10 000 elements, 2–2.7 times on random `f64` keys, 1.2–1.5
/// times with four distinct keys, 1.6 times on nearly sorted input and 1.7–2.4 times on powers
/// of two; but nearly sorted slices of 17 to 48 elements take 1.05–1.13 times as long (10–20
/// ns more), which none of four changes measured removed without costing more elsewhere.
///
/// # Properties
///
/// * **Stable**: elements with equal keys keep their order.
/// * **Not in place**: buffers of `n` keys and `n` elements (or `2n` `u32` indices), plus the
///   counts of a level; each on the stack up to 4 KiB and otherwise allocated.
/// * **Not adaptive**, as [`bucket`].
///
/// Against [`slice::sort_by_key`], from 100 elements on: 2.1–3 times faster on random `u64`
/// keys of 16-byte elements and 2.4–4.5 times of 72-byte elements, 1.8–3.2 times on random
/// `f64` keys, 2.6–5.5 times on keys below `n`, and about as fast on sorted and reverse-sorted
/// input. It is weaker where the standard library's merges of runs and its few distinct values
/// gain: powers of two take up to 2.1 times as long, four distinct keys up to 1.35 times, and
/// 16 elements about 1.1 times as long.
///
/// # Panics
///
/// If `key` panics, the panic is propagated and the slice is left as it was: no element has been
/// moved before the last key is read.
///
/// # Examples
///
/// ```
/// use a::sort::bucket_by_key;
///
/// let mut v = [(3, 'a'), (1, 'b'), (3, 'c'), (0, 'd'), (1, 'e')];
/// bucket_by_key(&mut v, |&(k, _)| k);
/// assert_eq!(v, [(0, 'd'), (1, 'b'), (1, 'e'), (3, 'a'), (3, 'c')]);
///
/// // Cities by temperature; equal temperatures keep their order.
/// let mut cities = vec![("Oslo", 4.5), ("Cairo", 31.0), ("Lima", 19.25), ("Kyiv", 4.5)];
/// bucket_by_key(&mut cities, |&(_, t)| t);
/// let names: Vec<&str> = cities.iter().map(|c| c.0).collect();
/// assert_eq!(names, ["Oslo", "Kyiv", "Lima", "Cairo"]);
/// ```
pub fn bucket_by_key<T, K, F>(arr: &mut [T], key: F)
where
    K: BucketKey,
    F: FnMut(&T) -> K,
{
    let len = arr.len();
    if len < 2 || mem::size_of::<T>() == 0 {
        return;
    }
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let slots = key_slots::<K::Bits>(len, &mut stack, &mut heap);
    let descents = match read_keys(arr, slots, key) {
        Order::Ascending => return,
        // Reversing a strictly descending slice keeps the order of equal keys, as there are
        // none.
        Order::Descending => {
            arr.reverse();
            return;
        }
        // Reversing a non-increasing slice reverses each run of equal keys, which is then
        // reversed back.
        Order::NonIncreasing => {
            arr.reverse();
            // SAFETY: all `len` keys have been written.
            let keys = unsafe { slice::from_raw_parts(slots.as_ptr().cast::<K::Bits>(), len) };
            // The runs of equal keys now lie in the opposite order, the first one last: `end`
            // is where the elements of the next run, in the order of the keys, end.
            let mut end = len;
            for run in keys.chunk_by(|x, y| x == y) {
                arr[end - run.len()..end].reverse();
                end -= run.len();
            }
            return;
        }
        Order::Neither(descents) => descents,
    };
    if len <= LEAF {
        let v = Area {
            keys: slots.as_mut_ptr().cast::<K::Bits>(),
            elems: arr.as_mut_ptr(),
        };
        // As `sort` would, without setting up the scratch space it does not need.
        // SAFETY: `v` holds the `len` keys, all written, and their elements.
        unsafe { insertion(v, len, &mut small_mode(descents, len)) };
        return;
    }
    let mut scratch_stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut scratch_heap = Vec::new();
    let scratch_keys = key_slots::<K::Bits>(len, &mut scratch_stack, &mut scratch_heap);
    // Sorted by moving each element once, the elements need no buffer.
    if cycles::<T>(len) {
        // SAFETY: all `len` keys have been written.
        let keys = unsafe { slice::from_raw_parts(slots.as_ptr().cast::<K::Bits>(), len) };
        let (min, max) = min_max(keys);
        if min != max {
            // SAFETY: `slots` holds the `len` keys of the elements of `arr`, from `min` to
            // `max`, and `scratch_keys` has room for as many; `len` is greater than `LEAF` and
            // at most `CYCLE_MAX_LEN`, which fits in `u32`.
            unsafe {
                by_index::<K, T, u32>(
                    slots.as_mut_ptr().cast::<K::Bits>(),
                    arr.as_mut_ptr(),
                    scratch_keys.as_mut_ptr().cast::<K::Bits>(),
                    len,
                    0,
                    (min, max),
                    &mut Mode::One,
                );
            }
        }
        return;
    }
    let mut buf = Buffer::<T>::new(len);
    let v = Area {
        keys: slots.as_mut_ptr().cast::<K::Bits>(),
        elems: arr.as_mut_ptr(),
    };
    let scratch = Area {
        keys: scratch_keys.as_mut_ptr().cast::<K::Bits>(),
        elems: buf.get(),
    };
    // SAFETY: `v` holds the `len` keys, all written, and their elements; `scratch` has room for
    // `len` of each, apart from them.
    unsafe {
        if len <= u32::MAX as usize {
            sort::<K, T, u32, false>(v, scratch, len, 0, &mut Mode::One, None);
        } else {
            sort::<K, T, usize, false>(v, scratch, len, 0, &mut Mode::One, None);
        }
    }
}

/// How the keys of a slice are ordered.
enum Order {
    /// Each key is at least the one before it.
    Ascending,
    /// Each key is less than the one before it.
    Descending,
    /// Each key is at most the one before it, and some are equal.
    NonIncreasing,
    /// None of these, with how many keys are less than the one before them, if counted.
    Neither(Option<usize>),
}

/// Elements of at most this many bytes have their keys read by one loop that tracks both
/// orders and never leaves early, which the compiler vectorizes; larger ones by loops that
/// leave at the first key that ends the order they follow. The vectorized loop gathers the keys
/// of larger elements with slow interleaved loads: on 32-byte elements, sorted input took 1.4
/// times as long as with the loops that leave early, which on 16-byte ones took 1.2–1.4 times as
/// long as the vectorized loop.
const VECTOR_READ_MAX_BYTES: usize = 16;

/// Writes the key of each element of `arr` to `slots`, calling `key` exactly once per element
/// in order, and returns how the keys are ordered.
#[inline(always)]
fn read_keys<T, K: Bucket, F: FnMut(&T) -> K>(
    arr: &[T],
    slots: &mut [MaybeUninit<K::Bits>],
    mut key: F,
) -> Order {
    let first = key(&arr[0]).ordered();
    slots[0].write(first);
    let mut prev = first;
    let mut rest = slots[1..].iter_mut().zip(&arr[1..]);
    if mem::size_of::<T>() <= VECTOR_READ_MAX_BYTES {
        let (mut descents, mut no_descent) = (0, false);
        if arr.len() <= LEAF {
            // Counted for `small_mode`; summing them made sorted input of 1000 elements 5%
            // slower than the flag below.
            for (slot, x) in rest {
                let x = key(x).ordered();
                slot.write(x);
                descents += usize::from(x < prev);
                no_descent |= x >= prev;
                prev = x;
            }
        } else {
            let mut descent = false;
            for (slot, x) in rest {
                let x = key(x).ordered();
                slot.write(x);
                descent |= x < prev;
                no_descent |= x >= prev;
                prev = x;
            }
            descents = usize::from(descent);
        }
        if descents == 0 {
            return Order::Ascending;
        }
        if !no_descent {
            return Order::Descending;
        }
        // SAFETY: all the keys have been written.
        let keys = unsafe { slice::from_raw_parts(slots.as_ptr().cast::<K::Bits>(), arr.len()) };
        // Checked apart from the loop, which a third order would slow down; most input leaves
        // at the first block of keys.
        return if in_order(keys, |x, y| y > x) {
            Order::NonIncreasing
        } else {
            Order::Neither((arr.len() <= LEAF).then_some(descents))
        };
    }
    // The last key before the first descent.
    let mut top = None;
    for (slot, x) in rest.by_ref() {
        let x = key(x).ordered();
        slot.write(x);
        if x < prev {
            top = Some(prev);
            prev = x;
            break;
        }
        prev = x;
    }
    let Some(top) = top else {
        return Order::Ascending;
    };
    // Only a slice whose keys before the first descent are all equal can be non-increasing;
    // those are equal neighbours if there are two of them.
    if top == first {
        let (mut non_increasing, mut equal) = (true, rest.len() + 2 < arr.len());
        if !equal {
            for (slot, x) in rest.by_ref() {
                let x = key(x).ordered();
                slot.write(x);
                if x >= prev {
                    (non_increasing, equal) = (x == prev, x == prev);
                    prev = x;
                    break;
                }
                prev = x;
            }
        }
        // Past the first equal neighbours, a loop that does not tell them apart.
        if equal {
            for (slot, x) in rest.by_ref() {
                let x = key(x).ordered();
                slot.write(x);
                if x > prev {
                    non_increasing = false;
                    break;
                }
                prev = x;
            }
        }
        if non_increasing {
            return if equal {
                Order::NonIncreasing
            } else {
                Order::Descending
            };
        }
    }
    for (slot, x) in rest {
        slot.write(key(x).ordered());
    }
    Order::Neither(None)
}

/// Elements of at least this many bytes are not moved with their keys when a part of the slice
/// is sorted in place: the keys are sorted with the indices of their elements, and the elements
/// are then moved once each, along the cycles of that permutation, with no buffer for them...
const CYCLE_MIN_BYTES: usize = 64;

/// ...when the part takes at least this many bytes, below which moving the elements with their
/// keys costs less than sorting the indices...
const CYCLE_MIN_PART_BYTES: usize = 64 << 10;

/// ...and has at most this many elements, beyond which walking the cycles misses the caches.
/// Twice as many made inputs of four distinct keys 1.33 times as slow; just above it, keys
/// below `n` of 72-byte elements take 1.07 times as long as they did before the cycle walk.
const CYCLE_MAX_LEN: usize = 1 << 13;

/// Whether a part of `len` elements of type `E` sorted in place is sorted by [`by_index`].
#[inline(always)]
fn cycles<E>(len: usize) -> bool {
    let size = mem::size_of::<E>();
    size >= CYCLE_MIN_BYTES
        && len > LEAF
        && len * size >= CYCLE_MIN_PART_BYTES
        && len <= CYCLE_MAX_LEN
}

/// Sorts the `len` keys at `keys` in place with the indices of their elements at `elems`, with
/// `scratch` as room for keys, then moves each element once, along the cycles of that
/// permutation. No buffer for the elements is needed.
///
/// # Safety
///
/// `keys` must hold `len` initialized keys, from the first to the second of `range` and not all
/// equal, of the `len` initialized elements at `elems`, and `scratch` room for `len` keys, apart
/// from them; `len` must be greater than [`LEAF`] and fit in `u32`.
#[inline(never)]
unsafe fn by_index<K: Bucket, E, C: Count>(
    keys: *mut K::Bits,
    elems: *mut E,
    scratch: *mut K::Bits,
    len: usize,
    depth: u32,
    range: (K::Bits, K::Bits),
    mode: &mut Mode,
) {
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let index = key_slots::<u32>(2 * len, &mut stack, &mut heap);
    let (index, index_scratch) = index.split_at_mut(len);
    for (i, slot) in (0..).zip(index.iter_mut()) {
        slot.write(i);
    }
    let index = index.as_mut_ptr().cast::<u32>();
    let v = Area { keys, elems: index };
    let scratch = Area {
        keys: scratch,
        elems: index_scratch.as_mut_ptr().cast::<u32>(),
    };
    // SAFETY: guaranteed by the caller; `v` holds the `len` keys and their indices, and
    // `scratch` has room for `len` of each, apart from them.
    unsafe { sort::<K, u32, C, false>(v, scratch, len, depth, mode, Some(range)) };
    let base = elems;
    // SAFETY: the sort permutes the indices, so they are `0..len`, each once: place `i` takes
    // the element at `index[i]`. Each cycle of the permutation is walked once: its first
    // element is read out, every place on it is filled from the next, which leaves that one
    // the gap, and the first is written into the last gap; a walked place is marked with its
    // own index, so no walk starts from it again. No code in between can panic.
    unsafe {
        for start in 0..len {
            let mut from = *index.add(start) as usize;
            if from == start {
                continue;
            }
            let first = ptr::read(base.add(start));
            let mut gap = start;
            loop {
                *index.add(gap) = gap as u32;
                if from == start {
                    break;
                }
                ptr::copy_nonoverlapping(base.add(from), base.add(gap), 1);
                gap = from;
                from = *index.add(gap) as usize;
            }
            base.add(gap).write(first);
        }
    }
}

/// Buckets of at most this many elements are sorted by insertion, together with the small
/// buckets next to them; larger ones are bucketed again.
const LEAF: usize = 16;

/// A level cuts its range into one bucket per this many elements...
const LOAD: usize = 1;

/// ...but into at most this many, so that the counts stay in the processor's fast caches; the
/// larger buckets that leaves are bucketed again.
const MAX_BUCKETS: usize = 1 << 12;

/// The first this many levels cut the range of float keys by value; deeper ones cut it by the
/// keys' bits, which bounds the depth for any input. Below the first level the range of a
/// bucket is narrow, and its keys' bits are close to linear in their values: cutting by bits
/// spreads them as well, and its map is cheaper. With two levels by value, float inputs took
/// 1.3% longer in geometric mean, uniform and exponential ones 3–4%.
const FLOAT_LEVELS: u32 = 1;

/// A level that cuts float keys by value is cut by their bits instead, if that leaves a smaller
/// largest bucket, when the bucket nearest zero and its two neighbours hold more than one in
/// this many of its keys: values spread over many orders of magnitude crowd there by value,
/// and cutting them so again only peels off the largest ones, while their bits, which grow
/// with the exponent, spread them. Checked once the keys are counted, before anything moves,
/// from three buckets; the keys are counted a second time, by bits, only when it holds, and
/// moved by whichever cut leaves the smaller largest bucket.
const CROWD: usize = 2;

/// ...unless the range ends at zero, or spans less than this factor between its ends of one
/// sign: the bits of such a range are close to linear in the values, and a crowd at zero most
/// likely zeros, one key whatever the cut, so that counting again would only cost. Measured
/// over float inputs from 100 to 1 000 000 keys, this made the geometric mean 1.3% faster: input
/// of 55–60% zeros 25% faster up to 10 000 keys, but 35% slower at 100 000, where the cut by
/// bits, which leaves the zeros alone in their bucket, saves a level.
const FEW_MAGNITUDES: f64 = 16.0;

/// A level with at least this many elements per bucket counts its keys into `FEW_TABLES`
/// tables, one key after the other into the next...
///
/// A count depends on the one before it in the same bucket: on a run of keys in one bucket, as
/// in a sorted stretch or among few distinct keys, each increment waits for the one before it
/// to reach memory. With `N` tables, `N` such chains run at once. Clearing and adding up the
/// tables costs little once there are this many elements per counter. Keys that never share a
/// bucket with the key before them only pay for it: 256 values spread evenly over the range of
/// 10 000–20 000 keys took 5–11% longer than with one table.
const FEW_LOAD: usize = 4;

/// See [`FEW_LOAD`]: two tables broke the chains of skewed input too little (geometric keys
/// took 1.2–1.3 times as long), and more cost more to clear at four elements per bucket.
const FEW_TABLES: usize = 4;

/// ...and with at least this many, into `MANY_TABLES`.
const MANY_LOAD: usize = 16;

/// See [`MANY_LOAD`]: at that many elements per bucket, clearing and adding up eight tables
/// costs less than the chains four leave; from four elements per bucket on, it cost random keys
/// 10% at 20 000 elements.
const MANY_TABLES: usize = 8;

/// A level of at most this many elements keeps the bucket of each key, found as the keys are
/// counted, for the moves that follow, so that the map is evaluated once per key. The buckets
/// are kept on the stack: kept in a fresh allocation, which a large level would need, they
/// cost more in page faults than they save.
const CACHE_MAX: usize = STACK_BYTES / mem::size_of::<u16>();

const _: () = assert!(
    MAX_BUCKETS <= 1 << 16,
    "a bucket must fit in the cached `u16`"
);

/// Where [`distribute`] finds the bucket of each key.
trait Place<B, E>: Copy {
    /// The bucket of the key at `i` of `src`.
    ///
    /// # Safety
    ///
    /// The key at `i` must be initialized.
    unsafe fn bucket(self, src: Area<B, E>, i: usize) -> usize;
}

/// The bucket of each key, mapped again.
#[derive(Clone, Copy)]
struct Mapped<I>(I);

impl<B: Bits, E, I: Index<B>> Place<B, E> for Mapped<I> {
    #[inline(always)]
    unsafe fn bucket(self, src: Area<B, E>, i: usize) -> usize {
        // SAFETY: guaranteed by the caller.
        self.0.of(unsafe { src.key(i) })
    }
}

/// The bucket of each key, kept as it was counted: one `u16` per key.
#[derive(Clone, Copy)]
struct Cached(*const u16);

impl<B, E> Place<B, E> for Cached {
    #[inline(always)]
    unsafe fn bucket(self, _: Area<B, E>, i: usize) -> usize {
        // SAFETY: a `Cached` is only made by `partition`, once `count` has written the bucket
        // of every key of the level.
        usize::from(unsafe { *self.0.add(i) })
    }
}

/// How many buckets a level of `len` elements cuts its range into.
fn buckets(len: usize) -> usize {
    (len / LOAD).clamp(2, MAX_BUCKETS)
}

/// Where the keys and their elements of a part of the slice, or of the scratch space, are.
struct Area<B, E> {
    keys: *mut B,
    elems: *mut E,
}

impl<B, E> Clone for Area<B, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, E> Copy for Area<B, E> {}

impl<B: Copy, E> Area<B, E> {
    /// The same keys, with the elements at `elems`.
    #[inline(always)]
    fn with<F>(self, elems: *mut F) -> Area<B, F> {
        Area {
            keys: self.keys,
            elems,
        }
    }

    /// The area from position `i` on.
    ///
    /// # Safety
    ///
    /// `i` must be within the area, or just past it.
    #[inline(always)]
    unsafe fn at(self, i: usize) -> Self {
        // SAFETY: guaranteed by the caller.
        unsafe {
            Self {
                keys: self.keys.add(i),
                elems: self.elems.add(i),
            }
        }
    }

    /// # Safety
    ///
    /// The key at `i` must be initialized.
    #[inline(always)]
    unsafe fn key(self, i: usize) -> B {
        // SAFETY: guaranteed by the caller.
        unsafe { *self.keys.add(i) }
    }

    /// Moves the key and the element at `i` to `j` of `dst`.
    ///
    /// # Safety
    ///
    /// Both positions must be within their areas, and the one at `i` initialized.
    #[inline(always)]
    unsafe fn move_to(self, i: usize, dst: Self, j: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe {
            dst.keys.add(j).write(*self.keys.add(i));
            ptr::copy(self.elems.add(i), dst.elems.add(j), 1);
        }
    }

    /// Moves the first `len` keys and elements to `dst`.
    ///
    /// # Safety
    ///
    /// Both areas must hold `len` positions, not overlapping, those of `self` initialized.
    #[inline(always)]
    unsafe fn move_all(self, dst: Self, len: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe {
            ptr::copy_nonoverlapping(self.keys, dst.keys, len);
            ptr::copy_nonoverlapping(self.elems, dst.elems, len);
        }
    }
}

/// Maps a key to the bucket it falls in. The map must be monotonic: a key never falls in a
/// bucket before that of a smaller key.
trait Index<B>: Copy {
    /// Whether every bucket holds a single key, so that a level is sorted once its keys are in
    /// their buckets.
    const ONE_KEY: bool = false;
    /// The bucket `key` falls in: a pure function of `key`, which gives the same bucket every
    /// time, as [`count`] and [`distribute`] may each evaluate it, and must agree. The float map
    /// does on every target whose float arithmetic is exact IEEE 754 (not x87 with its excess
    /// precision).
    fn of(self, key: B) -> usize;
    /// How many buckets the keys fall in.
    fn buckets(self) -> usize;
    /// For a cut by value that a cut by bits replaces when its keys crowd near zero (see
    /// [`CROWD`]): the bucket nearest zero, and the smallest and the largest key.
    #[inline(always)]
    fn crowd(self) -> Option<(usize, B, B)> {
        None
    }
}

/// One bucket per key from `min`: for a range of fewer keys than buckets.
#[derive(Clone, Copy)]
struct Direct<B> {
    min: B,
    buckets: usize,
}

impl<B: Bits> Index<B> for Direct<B> {
    const ONE_KEY: bool = true;
    #[inline(always)]
    fn of(self, key: B) -> usize {
        key.wrapping_sub(self.min).low64() as usize
    }
    #[inline(always)]
    fn buckets(self) -> usize {
        self.buckets
    }
}

/// Buckets of equal width over the keys' bits from `min`: `⌊(key − min) · m / 2^64⌋`, with the
/// difference shifted right by `shift` first when it may not fit in 64 bits.
#[derive(Clone, Copy)]
struct Linear<B> {
    min: B,
    shift: u32,
    m: u64,
    buckets: usize,
}

impl<B: Bits> Index<B> for Linear<B> {
    #[inline(always)]
    fn of(self, key: B) -> usize {
        let d = key.wrapping_sub(self.min);
        let d = if B::BITS > 64 { d.shr(self.shift) } else { d };
        ((u128::from(d.low64()) * u128::from(self.m)) >> 64) as usize
    }
    #[inline(always)]
    fn buckets(self) -> usize {
        self.buckets
    }
}

/// Buckets of equal width over the values of float keys from `lo`: `⌊(value − lo) · scale⌋`,
/// and the largest values in the last bucket. The map of `f32` keys is computed in `f32`, which
/// is exact for them and cheaper than converting each to `f64`.
struct Float<K: Bucket> {
    lo: f64,
    scale: f64,
    lo32: f32,
    scale32: f32,
    last: usize,
    /// The bucket nearest zero, unless the range spans few orders of magnitude (see
    /// [`FEW_MAGNITUDES`]).
    zero: Option<usize>,
    min: K::Bits,
    max: K::Bits,
}

impl<K: Bucket> Clone for Float<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: Bucket> Copy for Float<K> {}

impl<K: Bucket> Float<K> {
    /// Whether the keys are `f32`s, whose map is computed in `f32`.
    const NARROW: bool = <K::Bits as Bits>::BITS == 32;

    /// The map of `buckets` buckets from `min` to `max`, if both are finite and the width of
    /// the range is a finite, positive float (in `f32` for `f32` keys: a range of subnormals
    /// only is too narrow for that, and its bits are linear in the values anyway).
    fn new(min: K::Bits, max: K::Bits, buckets: usize) -> Option<Self> {
        let (lo, hi) = (K::value(min), K::value(max));
        let mut map = Self {
            lo,
            scale: 0.0,
            lo32: 0.0,
            scale32: 0.0,
            last: buckets - 1,
            zero: None,
            min,
            max,
        };
        if Self::NARROW {
            let (lo, hi) = (K::value32(min), K::value32(max));
            let width = hi - lo;
            map.lo32 = lo;
            map.scale32 = buckets as f32 / width;
            if !(lo.is_finite() && width.is_finite() && width > 0.0 && map.scale32.is_finite()) {
                return None;
            }
        } else {
            let width = hi - lo;
            map.scale = buckets as f64 / width;
            if !(lo.is_finite() && width.is_finite() && width > 0.0 && map.scale.is_finite()) {
                return None;
            }
        }
        let few_magnitudes = lo == 0.0
            || hi == 0.0
            || (lo > 0.0 && hi <= lo * FEW_MAGNITUDES)
            || (hi < 0.0 && lo >= hi * FEW_MAGNITUDES);
        if !few_magnitudes {
            map.zero = Some(map.at(0.0));
        }
        Some(map)
    }

    /// The bucket of `value`: the first or the last for a value outside the range.
    #[inline(always)]
    fn at(self, value: f64) -> usize {
        // Rounding keeps the map monotonic, and a cast saturates, a negative product to 0.
        if Self::NARROW {
            (((value as f32 - self.lo32) * self.scale32) as usize).min(self.last)
        } else {
            (((value - self.lo) * self.scale) as usize).min(self.last)
        }
    }
}

impl<K: Bucket> Index<K::Bits> for Float<K> {
    #[inline(always)]
    fn of(self, key: K::Bits) -> usize {
        // Every key is finite and at least `lo`, so the product is not negative; as in `at`.
        if Self::NARROW {
            (((K::value32(key) - self.lo32) * self.scale32) as usize).min(self.last)
        } else {
            self.at(K::value(key))
        }
    }
    #[inline(always)]
    fn buckets(self) -> usize {
        self.last + 1
    }
    #[inline(always)]
    fn crowd(self) -> Option<(usize, K::Bits, K::Bits)> {
        self.zero.map(|zero| (zero, self.min, self.max))
    }
}

/// Sorts the `len` keys of `src`, and their elements with them: in place if not `INTO`, with
/// `dst` as scratch space, and into `dst` if `INTO`. `mode` is how the insertion sorts move
/// keys, carried from one to the next (see [`sweep`]); `bounds` is the smallest and the largest
/// key, if the caller found them already.
///
/// Never inlined: inlined into its other instance, it spilled the registers of the hot loops
/// and made some inputs up to 10% slower.
///
/// # Safety
///
/// `src` must hold `len` initialized keys and elements, and `dst` room for as many, apart from
/// them; `bounds`, if given, must be the smallest and the largest of the keys.
#[inline(never)]
unsafe fn sort<K: Bucket, E, C: Count, const INTO: bool>(
    src: Area<K::Bits, E>,
    dst: Area<K::Bits, E>,
    len: usize,
    depth: u32,
    mode: &mut Mode,
    bounds: Option<(K::Bits, K::Bits)>,
) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        if len <= LEAF {
            if INTO {
                insertion_into(src, dst, len, mode);
            } else {
                insertion(src, len, mode);
            }
            return;
        }
        let (min, max) = match bounds {
            Some(bounds) => bounds,
            None => min_max(slice::from_raw_parts(src.keys, len)),
        };
        if min == max {
            if INTO {
                src.move_all(dst, len);
            }
            return;
        }
        // `len` is at most `CYCLE_MAX_LEN` there, which fits in `u32`.
        if !INTO && cycles::<E>(len) {
            let (keys, elems) = (src.keys, src.elems);
            return by_index::<K, E, C>(keys, elems, dst.keys, len, depth, (min, max), mode);
        }
        if K::FLOAT
            && depth < FLOAT_LEVELS
            && let Some(map) = Float::<K>::new(min, max, buckets(len))
        {
            level::<K, E, C, _, INTO>(src, dst, len, map, depth, mode, usize::MAX);
            return;
        }
        by_bits::<K, E, C, INTO>(src, dst, len, (min, max), depth, mode, usize::MAX);
    }
}

/// A level of [`sort`] that cuts the range from the smallest to the largest key by the keys'
/// bits: into one bucket per key if there are fewer keys than buckets. With `most` at most
/// `len`, it gives up, with nothing moved, if its largest bucket holds `most` keys or more,
/// and returns whether it did not.
///
/// # Safety
///
/// As for [`sort`], with `range` the smallest and the largest key, not equal.
#[inline(always)]
unsafe fn by_bits<K: Bucket, E, C: Count, const INTO: bool>(
    src: Area<K::Bits, E>,
    dst: Area<K::Bits, E>,
    len: usize,
    (min, max): (K::Bits, K::Bits),
    depth: u32,
    mode: &mut Mode,
    most: usize,
) -> bool {
    let buckets = buckets(len);
    let span = max.wrapping_sub(min);
    let width = <K::Bits as Bits>::BITS - span.leading_zeros();
    // SAFETY: guaranteed by the caller; each map sends `min` to its first bucket and `max` to
    // its last.
    unsafe {
        if width < 64 && span.low64() < buckets as u64 {
            let map = Direct {
                min,
                buckets: span.low64() as usize + 1,
            };
            level::<K, E, C, _, INTO>(src, dst, len, map, depth, mode, most)
        } else {
            let shift = width.saturating_sub(64);
            // At least `buckets`, so `m` is less than 2^64.
            let span = span.shr(shift).low64();
            let m = ((buckets as u128) << 64) / (u128::from(span) + 1);
            let map = Linear {
                min,
                shift,
                m: m as u64,
                buckets,
            };
            level::<K, E, C, _, INTO>(src, dst, len, map, depth, mode, most)
        }
    }
}

/// [`by_bits`] for a level cut by value whose keys crowd near zero, the largest of the buckets
/// there holding `most` keys: it is cut by bits only if that leaves a smaller largest bucket,
/// and so are the levels below it. Out of line, so that the counters of both cuts are not in
/// the frame of every [`sort`].
///
/// # Safety
///
/// As for [`by_bits`].
#[cold]
#[inline(never)]
unsafe fn by_bits_instead<K: Bucket, E, C: Count, const INTO: bool>(
    src: Area<K::Bits, E>,
    dst: Area<K::Bits, E>,
    len: usize,
    range: (K::Bits, K::Bits),
    mode: &mut Mode,
    most: usize,
) -> bool {
    // SAFETY: guaranteed by the caller.
    unsafe { by_bits::<K, E, C, INTO>(src, dst, len, range, FLOAT_LEVELS, mode, most) }
}

/// One level of [`sort`]: moves the keys and elements of `src` into their buckets in `dst`, then
/// sorts the buckets where `sort` leaves them, and returns `true`. A level whose largest bucket
/// holds `most` keys or more gives up, with nothing moved, and returns `false`; a level cut by
/// value whose keys crowd near zero is cut by bits instead, if that is better.
///
/// # Safety
///
/// As for [`sort`]; `map` must map every key of `src` below its `buckets`.
#[inline(always)]
unsafe fn level<K: Bucket, E, C: Count, I: Index<K::Bits>, const INTO: bool>(
    src: Area<K::Bits, E>,
    dst: Area<K::Bits, E>,
    len: usize,
    map: I,
    depth: u32,
    mode: &mut Mode,
    most: usize,
) -> bool {
    let buckets = map.buckets();
    let tables = if len >= MANY_LOAD * buckets {
        MANY_TABLES
    } else if len >= FEW_LOAD * buckets {
        FEW_TABLES
    } else {
        1
    };
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    // An allocation while elements lie in both `src` and `dst` cannot unwind: allocation
    // failure aborts on stable Rust (as in `by_index`).
    let counts = counters::<C>(tables * buckets, &mut stack, &mut heap);
    // SAFETY: guaranteed by the caller.
    unsafe {
        if let Some(most) = partition(src, dst, len, map, counts, tables, most) {
            // Nothing has moved.
            let Some((_, min, max)) = map.crowd() else {
                return false;
            };
            if by_bits_instead::<K, E, C, INTO>(src, dst, len, (min, max), mode, most) {
                return true;
            }
            // The cut by value stays, with the starts it counted.
            scatter(src, dst, len, map, &mut counts[..buckets]);
        }
        let ends = &counts[..buckets];
        // Every bucket holds one key, its elements in the order they came in: the level is
        // sorted.
        if I::ONE_KEY {
            if !INTO {
                dst.move_all(src, len);
            }
            return true;
        }
        // Runs of small buckets are sorted by one insertion sort: no key moves out of its
        // bucket, as every key before it is smaller.
        let (mut start, mut flushed) = (0, 0);
        for end in ends.iter() {
            let end = end.get();
            if end - start > LEAF {
                let (to, n) = (src.at(start), end - start);
                if INTO {
                    insertion(dst.at(flushed), start - flushed, mode);
                    sort::<K, E, C, false>(dst.at(start), to, n, depth + 1, mode, None);
                } else {
                    insertion_into(dst.at(flushed), src.at(flushed), start - flushed, mode);
                    sort::<K, E, C, true>(dst.at(start), to, n, depth + 1, mode, None);
                }
                flushed = end;
            }
            start = end;
        }
        if INTO {
            insertion(dst.at(flushed), len - flushed, mode);
        } else {
            insertion_into(dst.at(flushed), src.at(flushed), len - flushed, mode);
        }
    }
    true
}

/// Moves the keys and elements of `src` into their buckets in `dst`, stably, counting them into
/// `tables` tables first, and leaves in the first `counts.len() / tables` counters the end of
/// each bucket. Gives up before moving any, with the starts of the buckets left in those
/// counters, if [`gives_up`] says so, and returns what it says.
///
/// Never inlined, so that the buckets it keeps on the stack are not part of the frames of the
/// levels below.
///
/// # Safety
///
/// As for [`level`], with `counts` zeroed and `tables` counters per bucket, `tables` being 1,
/// `FEW_TABLES` or `MANY_TABLES`.
#[inline(never)]
unsafe fn partition<B: Bits, E, C: Count, I: Index<B>>(
    src: Area<B, E>,
    dst: Area<B, E>,
    len: usize,
    map: I,
    counts: &mut [C],
    tables: usize,
    most: usize,
) -> Option<usize> {
    let buckets = counts.len() / tables;
    // SAFETY: guaranteed by the caller; `idx` has room for the bucket of every key.
    unsafe {
        if len <= CACHE_MAX {
            let mut idx = [MaybeUninit::<u16>::uninit(); CACHE_MAX];
            let idx = idx.as_mut_ptr().cast::<u16>();
            match tables {
                MANY_TABLES => count::<_, _, _, _, MANY_TABLES>(src, len, map, counts, idx),
                FEW_TABLES => count::<_, _, _, _, FEW_TABLES>(src, len, map, counts, idx),
                _ => count::<_, _, _, _, 1>(src, len, map, counts, idx),
            }
            let starts = &mut counts[..buckets];
            if let Some(most) = gives_up(map, starts, len, most) {
                return Some(most);
            }
            distribute(src, dst, len, Cached(idx), starts);
        } else {
            let idx = ptr::null_mut();
            match tables {
                MANY_TABLES => count::<_, _, _, _, MANY_TABLES>(src, len, map, counts, idx),
                FEW_TABLES => count::<_, _, _, _, FEW_TABLES>(src, len, map, counts, idx),
                _ => count::<_, _, _, _, 1>(src, len, map, counts, idx),
            }
            let starts = &mut counts[..buckets];
            if let Some(most) = gives_up(map, starts, len, most) {
                return Some(most);
            }
            distribute(src, dst, len, Mapped(map), starts);
        }
    }
    None
}

/// Whether a level whose `len` keys `map` counted, with `starts` where each bucket starts,
/// gives up before moving any: if `map` cuts by value and its keys crowd near zero (see
/// [`CROWD`]), with the most keys any of the buckets there holds; and if its largest bucket
/// holds `most` keys or more, with nothing to tell.
#[inline(always)]
fn gives_up<B, C: Count, I: Index<B>>(
    map: I,
    starts: &[C],
    len: usize,
    most: usize,
) -> Option<usize> {
    let buckets = starts.len();
    let start = |b: usize| if b < buckets { starts[b].get() } else { len };
    if let Some((zero, _, _)) = map.crowd() {
        let (lo, hi) = (zero.saturating_sub(1), (zero + 2).min(buckets));
        if start(hi) - start(lo) > len / CROWD {
            let crowd = (lo..hi).map(|b| start(b + 1) - start(b)).max();
            return Some(crowd.unwrap_or(0));
        }
    }
    if most <= len && (0..buckets).any(|b| start(b + 1) - start(b) >= most) {
        return Some(0);
    }
    None
}

/// Moves the keys and elements of `src` into their buckets in `dst` by `map` from `starts`, as
/// [`distribute`], for a level whose cached buckets are gone: one cut by value that stays after
/// its keys were counted by bits as well.
///
/// # Safety
///
/// As for [`distribute`].
#[cold]
#[inline(never)]
unsafe fn scatter<B: Bits, E, C: Count, I: Index<B>>(
    src: Area<B, E>,
    dst: Area<B, E>,
    len: usize,
    map: I,
    starts: &mut [C],
) {
    // SAFETY: guaranteed by the caller.
    unsafe { distribute(src, dst, len, Mapped(map), starts) }
}

/// The places four keys in a row take, from their buckets `b` and the next free place `next`
/// of each bucket before them: a key takes the place after those taken by the keys before it
/// in the same bucket.
///
/// The four next places are all read before any is written back, so a run of keys in one
/// bucket waits for memory once per four keys, not once per key; a chain through memory costs
/// about 5 cycles, and a sorted stretch or a few distinct keys made the loops 2-3 times slower.
#[inline(always)]
fn places(b: [usize; 4], next: [usize; 4]) -> [usize; 4] {
    [
        next[0],
        next[1] + usize::from(b[1] == b[0]),
        next[2] + usize::from(b[2] == b[0]) + usize::from(b[2] == b[1]),
        next[3] + usize::from(b[3] == b[0]) + usize::from(b[3] == b[1]) + usize::from(b[3] == b[2]),
    ]
}

/// Counts the keys of `src` that fall in each bucket into `N` tables, interleaved, the key at
/// `i` into table `i mod N`, and leaves in the first `counts.len() / N` counters where each
/// bucket starts. If `idx` is not null, the bucket of the key at `i` is also written to it at
/// `i`.
///
/// With one table, the keys are counted four at a time, as [`places`] hands out places.
///
/// # Safety
///
/// As for [`level`], with `counts` zeroed and `N` counters per bucket, and `idx` null or valid
/// for `len` writes.
#[inline(always)]
unsafe fn count<B: Bits, E, C: Count, I: Index<B>, const N: usize>(
    src: Area<B, E>,
    len: usize,
    map: I,
    counts: &mut [C],
    idx: *mut u16,
) {
    let buckets = counts.len() / N;
    let c = counts.as_mut_ptr();
    let cache = !idx.is_null();
    // SAFETY: guaranteed by the caller: every bucket is below `buckets`, so its counters are
    // below `N * buckets`; a bucket fits in a `u16`, as `MAX_BUCKETS <= 2^16`.
    unsafe {
        let mut i = 0;
        if N == 1 {
            while i + 4 <= len {
                let b = [0, 1, 2, 3].map(|j| map.of(src.key(i + j)));
                if cache {
                    for (j, &b) in b.iter().enumerate() {
                        idx.add(i + j).write(b as u16);
                    }
                }
                let counted = places(b, b.map(|b| (*c.add(b)).get()));
                for j in 0..4 {
                    *c.add(b[j]) = C::ZERO.add(counted[j] + 1);
                }
                i += 4;
            }
        } else {
            while i + 8 <= len {
                for j in 0..8 {
                    let b = map.of(src.key(i + j));
                    if cache {
                        idx.add(i + j).write(b as u16);
                    }
                    bump(c.add(b * N + j % N));
                }
                i += 8;
            }
        }
        for i in i..len {
            let b = map.of(src.key(i));
            debug_assert!(b < buckets);
            if cache {
                idx.add(i).write(b as u16);
            }
            bump(c.add(b * N));
        }
        let mut sum = 0;
        for b in 0..buckets {
            let mut count = 0;
            for t in 0..N {
                count += (*c.add(b * N + t)).get();
            }
            // `b <= b * N`: the counters of the buckets after `b` are still to be read.
            *c.add(b) = C::ZERO.add(sum);
            sum += count;
        }
    }
}

/// Moves the keys and elements of `src` into their buckets in `dst`, stably, four at a time as
/// [`places`] hands out places, and turns the start of each bucket in `starts` into its end.
///
/// # Safety
///
/// As for [`level`], with `starts` the starts [`count`] leaves, and `place` finding the
/// buckets it counted.
#[inline(always)]
unsafe fn distribute<B: Bits, E, C: Count, P: Place<B, E>>(
    src: Area<B, E>,
    dst: Area<B, E>,
    len: usize,
    place: P,
    starts: &mut [C],
) {
    let c = starts.as_mut_ptr();
    // SAFETY: guaranteed by the caller: every bucket is below `starts.len()`, and the places
    // of each bucket are handed out once each, in order, all below `len`, as by one key at a
    // time.
    unsafe {
        let mut i = 0;
        while i + 4 <= len {
            let b = [0, 1, 2, 3].map(|j| place.bucket(src, i + j));
            let p = places(b, b.map(|b| (*c.add(b)).get()));
            debug_assert!(p.iter().all(|&p| p < len));
            for j in 0..4 {
                *c.add(b[j]) = C::ZERO.add(p[j] + 1);
            }
            for (j, &p) in p.iter().enumerate() {
                src.move_to(i + j, dst, p);
            }
            i += 4;
        }
        for i in i..len {
            let pos = bump(c.add(place.bucket(src, i)));
            debug_assert!(pos < len);
            src.move_to(i, dst, pos);
        }
    }
}

/// Elements of at most this many bytes are moved by [`sweep`] without a branch on whether they
/// move a place; larger ones only when they do, as the moves a select makes cost more than the
/// branches it saves.
const SELECT_BYTES: usize = 16;

/// [`sweep`] chooses how to move keys for each chunk of this many...
const CHUNK: usize = 32;

/// ...branching on whether a key moves when at most this many in a chunk's worth moved in the
/// chunk before: then the branch is well predicted, and cheaper than moving without one.
const FEW_MOVED: usize = 1;

/// Elements too large for [`sweep`] to move without a branch are moved once each by [`gather`]
/// when more than one in this many of their keys is less than the one before it: the branch
/// then mispredicts often enough to pay for the three passes of sorting their positions and
/// fetching them. With one in 32, keys below the length at 5000 elements (about one in 20 out
/// of order) took 1.15 times as long as when moved by the branch.
const GATHER_PART: usize = 8;

/// [`gather`] reads the positions this many at a time into a buffer on the stack.
const GATHER_BLOCK: usize = 256;

/// Elements of at most this many bytes are moved two places without a branch...
const TWO_BYTES: usize = 8;

/// ...when more than one in this many keys of the chunk before moved more than one place. For
/// keys that mostly move one place at most, the third write and select cost more than the
/// branches they save.
const FAR_PART: usize = 8;

/// How [`sweep`] moves the keys of a chunk. It is carried from one sweep to the next, as the
/// buckets of one input tend to be alike.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// With a branch on whether a key moves: for keys mostly in order or equal.
    Branch,
    /// One place without a branch, and further with one: for keys spread over the buckets.
    One,
    /// Two places without a branch, and further with one: for buckets of several keys each.
    Two,
}

/// How the insertion sort of a slice of at most [`LEAF`] elements starts, from how many of its
/// keys are less than the one before them, where a pass already read them all (it is not
/// counted for keys that need no conversion, where a pass of its own cost more than it saved,
/// nor for elements over 16 bytes, which always branch): with a branch when at most one in
/// eight is, as in a slice mostly in order. Moving without one cost such slices of 16 elements
/// up to 1.35 times as long; one in four took the branch too often on four distinct keys of 8
/// elements (1.08 times as long).
fn small_mode(descents: Option<usize>, len: usize) -> Mode {
    match descents {
        Some(d) if d * SMALL_DESCENTS_PART <= len => Mode::Branch,
        _ => Mode::One,
    }
}

/// See [`small_mode`].
const SMALL_DESCENTS_PART: usize = 8;

/// Sorts the `len` keys of `v`, and their elements with them, by insertion. Stable.
///
/// # Safety
///
/// `v` must hold `len` initialized keys and elements.
#[inline(always)]
unsafe fn insertion<B: Bits, E>(v: Area<B, E>, len: usize, mode: &mut Mode) {
    // SAFETY: guaranteed by the caller.
    unsafe { sweep::<B, E, true>(v, v, len, mode) }
}

/// Moves the `len` keys of `src`, and their elements with them, into `dst`, sorted by
/// insertion. Stable.
///
/// Elements too large for [`sweep`] to move without a branch are, when the keys are out of
/// order often enough to make the branch mispredict, moved once each by [`gather`] instead.
///
/// # Safety
///
/// `src` must hold `len` initialized keys and elements, and `dst` room for as many, apart from
/// them.
#[inline(always)]
unsafe fn insertion_into<B: Bits, E>(
    src: Area<B, E>,
    dst: Area<B, E>,
    len: usize,
    mode: &mut Mode,
) {
    if mem::size_of::<E>() > SELECT_BYTES
        && *mode != Mode::Branch
        && mem::size_of::<E>() >= 2 * mem::size_of::<usize>()
        && dst.elems.cast::<usize>().is_aligned()
        // SAFETY: guaranteed by the caller.
        && unsafe { descents(src.keys, len) } * GATHER_PART > len
    {
        // SAFETY: guaranteed by the caller, and checked above.
        return unsafe { gather(src, dst, len, mode) };
    }
    // SAFETY: guaranteed by the caller.
    unsafe { sweep::<B, E, false>(src, dst, len, mode) }
}

/// [`insertion_into`] for elements that are moved once each: their positions are sorted with
/// the keys, in the room of the elements in `dst`, and the elements then fetched from them.
///
/// The places are filled a block of [`GATHER_BLOCK`] at a time from the last block, the
/// positions of a block copied to the stack before its elements are written. Read straight
/// from the room the elements are written to, the positions made each read wait on the writes
/// before it: for elements of 24, 40 or 48 bytes, the whole sort took up to 1.7 times as long.
/// Out of line, so that the block is not in the frames of the levels that recurse.
///
/// # Safety
///
/// As for [`insertion_into`], with elements of at least `2 · size_of::<usize>()` bytes and
/// `dst.elems` aligned for `usize`.
#[inline(never)]
unsafe fn gather<B: Bits, E>(src: Area<B, E>, dst: Area<B, E>, len: usize, mode: &mut Mode) {
    let ix = dst.elems.cast::<usize>();
    let mut block = [MaybeUninit::<usize>::uninit(); GATHER_BLOCK];
    // SAFETY: guaranteed by the caller; the room for `len` elements in `dst` holds `2·len`
    // positions, aligned, the first `len` apart from the last `len`. They are sorted with the
    // keys, a permutation of `0..len` whatever the keys, and then each element is moved once,
    // into the place of its position. The places of a block, from `start` on, only cover
    // positions from `start` on: those of the block, copied out by then, and those of the
    // blocks after it, read already.
    unsafe {
        let ids = ix.add(len);
        for i in 0..len {
            ids.add(i).write(i);
        }
        sweep::<B, usize, false>(src.with(ids), dst.with(ix), len, mode);
        let mut end = len;
        while end > 0 {
            let start = end.saturating_sub(GATHER_BLOCK);
            let block = &mut block[..end - start];
            ptr::copy_nonoverlapping(ix.add(start), block.as_mut_ptr().cast(), block.len());
            for (j, i) in (start..).zip(block.iter()) {
                ptr::copy_nonoverlapping(src.elems.add(i.assume_init()), dst.elems.add(j), 1);
            }
            end = start;
        }
    }
}

/// How many of the `len` keys from `keys` are less than the one before them.
///
/// # Safety
///
/// `keys` must hold `len` initialized keys.
#[inline(always)]
unsafe fn descents<B: Bits>(keys: *const B, len: usize) -> usize {
    // SAFETY: guaranteed by the caller.
    let keys = unsafe { slice::from_raw_parts(keys, len) };
    // A plain loop: as an iterator chain, it was left out of line in some instances.
    let mut n = 0;
    for i in 1..len {
        n += usize::from(keys[i] < keys[i - 1]);
    }
    n
}

/// Insertion sort of the `len` keys of `src`, and their elements, into `dst`: `src` itself if
/// `IN_PLACE`, and otherwise apart from it.
///
/// The keys come out of small buckets, so a key that moves at all mostly moves one place. With
/// keys spread over the buckets about half the keys that share a bucket move, and branching on
/// it mispredicts: a step then writes the larger of the key and the one before it at `i` and
/// the smaller at `i - 1` without a branch, and only branches on whether the key goes further
/// down; or does so for two places (see [`Mode`]). With keys in order, or equal, that costs
/// more than a branch that is always right: a chunk after one in which few keys moved branches
/// instead. `mode` is how the sweep starts, and is left as the next one should start.
///
/// # Safety
///
/// As for [`insertion_into`], or [`insertion`] with `src` and `dst` the same if `IN_PLACE`.
#[inline(always)]
unsafe fn sweep<B: Bits, E, const IN_PLACE: bool>(
    src: Area<B, E>,
    dst: Area<B, E>,
    len: usize,
    mode: &mut Mode,
) {
    if len == 0 {
        return;
    }
    let select = mem::size_of::<E>() <= SELECT_BYTES;
    // SAFETY: guaranteed by the caller. Every position read or written is below `i < len`; the
    // first `i` places of `dst` hold the first `i` keys and elements, sorted. A step writes a
    // permutation of the element at `i` of `src`, read before anything is written over it, and
    // those of `dst` it moves up, with no code in between that can panic.
    unsafe {
        if !IN_PLACE {
            src.move_to(0, dst, 0);
        }
        let mut i = 1;
        let mut now = if select { *mode } else { Mode::Branch };
        while i < len {
            let end = if select { (i + CHUNK).min(len) } else { len };
            let chunk = end - i;
            // How many keys moved, and how many of them more than one place.
            let (mut moved, mut far) = (0, 0);
            match now {
                Mode::Branch => {
                    while i < end {
                        let k = src.key(i);
                        if dst.key(i - 1) <= k {
                            if !IN_PLACE {
                                src.move_to(i, dst, i);
                            }
                        } else {
                            moved += 1;
                            let e = ptr::read(src.elems.add(i).cast::<MaybeUninit<E>>());
                            let mut j = i;
                            loop {
                                dst.move_to(j - 1, dst, j);
                                j -= 1;
                                if j == 0 || dst.key(j - 1) <= k {
                                    break;
                                }
                            }
                            far += usize::from(j + 1 < i);
                            dst.keys.add(j).write(k);
                            ptr::copy_nonoverlapping(e.as_ptr(), dst.elems.add(j), 1);
                        }
                        i += 1;
                    }
                }
                Mode::One => {
                    // The largest key so far, at `i - 1`, and the key before it, at `i - 2`
                    // (or one no key is less than).
                    let mut last = dst.key(i - 1);
                    let mut before = if i >= 2 { dst.key(i - 2) } else { B::ZERO };
                    while i < end {
                        let k = src.key(i);
                        let up = last > k;
                        moved += usize::from(up);
                        let tmp;
                        let e = if IN_PLACE {
                            tmp = ptr::read(src.elems.add(i).cast::<MaybeUninit<E>>());
                            tmp.as_ptr()
                        } else {
                            src.elems.add(i).cast_const()
                        };
                        let p1 = dst.elems.add(i - 1).cast_const();
                        ptr::copy(select_unpredictable(up, p1, e), dst.elems.add(i), 1);
                        // Onto itself if `k` stays.
                        ptr::copy(select_unpredictable(up, e, p1), dst.elems.add(i - 1), 1);
                        dst.keys.add(i).write(select_unpredictable(up, last, k));
                        dst.keys.add(i - 1).write(select_unpredictable(up, k, last));
                        if up & (before > k) {
                            far += 1;
                            let mut j = i - 1;
                            loop {
                                dst.move_to(j - 1, dst, j);
                                j -= 1;
                                if j == 0 || dst.key(j - 1) <= k {
                                    break;
                                }
                            }
                            dst.keys.add(j).write(k);
                            ptr::copy_nonoverlapping(e, dst.elems.add(j), 1);
                            before = dst.key(i - 1);
                        } else {
                            before = select_unpredictable(up, k, last);
                            last = select_unpredictable(up, last, k);
                        }
                        i += 1;
                    }
                }
                Mode::Two => {
                    // As for `Mode::One`.
                    let mut last = dst.key(i - 1);
                    let mut before = if i >= 2 { dst.key(i - 2) } else { B::ZERO };
                    while i < end {
                        let k = src.key(i);
                        let up = last > k;
                        let up2 = before > k;
                        moved += usize::from(up);
                        far += usize::from(up2);
                        let tmp;
                        let e = if IN_PLACE {
                            tmp = ptr::read(src.elems.add(i).cast::<MaybeUninit<E>>());
                            tmp.as_ptr()
                        } else {
                            src.elems.add(i).cast_const()
                        };
                        // At `i = 1` the places `i - 2` and `i - 1` are the same: then
                        // `before` is no key, `up2` false, and what is written at `i - 1` is
                        // written last for the keys and first for the elements.
                        let i2 = i.saturating_sub(2);
                        let p1 = dst.elems.add(i - 1).cast_const();
                        let p2 = dst.elems.add(i2).cast_const();
                        let mid = select_unpredictable(up, k, last);
                        dst.keys.add(i2).write(select_unpredictable(up2, k, before));
                        dst.keys
                            .add(i - 1)
                            .write(select_unpredictable(up2, before, mid));
                        dst.keys.add(i).write(select_unpredictable(up, last, k));
                        ptr::copy(select_unpredictable(up, p1, e), dst.elems.add(i), 1);
                        let pm = select_unpredictable(up, e, p1);
                        ptr::copy(select_unpredictable(up2, p2, pm), dst.elems.add(i - 1), 1);
                        ptr::copy(select_unpredictable(up2, e, p2), dst.elems.add(i2), 1);
                        // At `i = 2` this reads place 0, written with `k` just above if
                        // `up2`: not greater, as `k` goes no further.
                        if up2 & (dst.key(i.saturating_sub(3)) > k) {
                            let mut j = i2;
                            loop {
                                dst.move_to(j - 1, dst, j);
                                j -= 1;
                                if j == 0 || dst.key(j - 1) <= k {
                                    break;
                                }
                            }
                            dst.keys.add(j).write(k);
                            ptr::copy_nonoverlapping(e, dst.elems.add(j), 1);
                        } else {
                            before = select_unpredictable(up2, before, mid);
                            last = select_unpredictable(up, last, k);
                        }
                        i += 1;
                    }
                }
            }
            // For large elements it only tells `insertion_into` whether to move them once.
            *mode = if moved * CHUNK <= FEW_MOVED * chunk {
                Mode::Branch
            } else if mem::size_of::<E>() <= TWO_BYTES && far * FAR_PART > chunk {
                Mode::Two
            } else {
                Mode::One
            };
            now = if select { *mode } else { Mode::Branch };
        }
    }
}
