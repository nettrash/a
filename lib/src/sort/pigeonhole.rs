use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};

use super::counting::{
    Count, STACK_BYTES, Stack, bump, counters, in_order, key_slots, min_max, range,
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
/// the slice from the counts, sorts them with less work: 2.5–12 times faster. Pigeonhole sort
/// is for [`pigeonhole_by_key`], which moves elements of any type in place.
///
/// Four refinements keep the algorithm as cheap as pigeonhole sort can be:
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
///   element moves twice. 1.1–4.3 times faster on random keys from 100,000 elements on.
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
/// Against [`slice::sort_unstable`], for `u32` from 1000 elements on: 1.6–3 times faster on
/// random keys in a range of about `n` or `n / 16` keys, 3–5 times on nearly sorted input,
/// 2.5–4.9 times on sorted and reverse-sorted input; but up to 1.15 times slower on a range of
/// `n` keys from 100,000 elements on, 1.4–1.8 times slower on a range of `4n` keys from 10,000
/// elements on, where most holes are empty, and 1.4–2.9 times slower with four distinct keys,
/// which the standard library's partitioning gains from. Against the textbook pigeonhole sort
/// (a vector per hole, the elements pushed in and read out): 2–6 times faster on random keys
/// and 160–340 times on sorted input.
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
/// sorts them, at most once, or by swaps for large slices. The keys are kept in a buffer of
/// their own as they are read; `key` is called exactly once per element of a slice of two or
/// more elements that are not zero-sized, and once the last key is read no code of the caller
/// runs.
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
/// range, and as fast below 10,000 elements; but nearly sorted input from 100,000 elements on
/// takes 1.05–2 times as long.
///
/// # Properties
///
/// * **Not stable**: elements with equal keys may change their order.
/// * **In place** for the elements: a buffer of `n` keys, rewritten as holes, and `O(k)` counts,
///   each on the stack up to 4 KiB and otherwise allocated once.
/// * **Not adaptive**, except to sorted and non-increasing input, and to elements already in
///   their holes, which are passed over.
///
/// Against [`slice::sort_unstable_by_key`], from 100 elements on: 1.6–2.4 times faster on
/// random `u64` keys of 16-byte elements up to 10,000 elements and 1.1–1.2 times beyond,
/// 1.8–4.9 times for 72-byte elements and 1.5–4.6 times for 128-byte ones, 1.7–14 times on
/// nearly sorted input of up to 72-byte elements; but 1.2–2.1 times slower on sorted and
/// reverse-sorted input, which the standard library recognises in one comparison per element,
/// and 1.5–2.5 times slower with four distinct keys of 16-byte elements. Against
/// [`counting_by_key`](fn@super::counting_by_key), which is stable and moves the elements
/// through a buffer: 1.2–3.4 times slower on random keys, but 1.3–2.2 times faster on nearly
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

/// ...when more than one in this many keys is less than the one before it: input mostly in
/// order is placed faster along the cycles, which pass over the elements already in place.
const SWAP_DESCENTS_PART: usize = 16;

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
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    // The count of each hole, which becomes its end.
    let ends = counters::<C>(k, &mut stack, &mut heap).as_mut_ptr();
    let mut next_stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut next_heap = Vec::new();
    // The next free place of each hole; never read for an empty one, so it needs no clearing.
    let next = key_slots::<C>(k, &mut next_stack, &mut next_heap)
        .as_mut_ptr()
        .cast::<C>();
    let holes = keys.cast::<H>();
    let sparse = k > SPARSE_PER_ELEMENT * len;
    // SAFETY: guaranteed by the caller: every key's hole is below `k`, so its counters are in
    // `ends` and `next`. The hole of key `i` is written over bytes before the end of key `i`,
    // once it and the keys before it have been read; the buffer is aligned for `K`, and so for
    // `H`, which is no larger. `next` is written for every hole that holds keys before it is
    // read.
    unsafe {
        let mut descents = 0;
        let mut prev = 0;
        for i in 0..len {
            let h = (*keys.add(i)).offset(min);
            if NARROW {
                holes.add(i).write(hole(h));
            }
            descents += usize::from(h < prev);
            prev = h;
            bump(ends.add(h));
        }
        let mut sum = 0;
        if sparse {
            each_hole(ends, k, |h, count| {
                *next.add(h) = C::ZERO.add(sum);
                sum += count;
                *ends.add(h) = C::ZERO.add(sum);
            });
        } else {
            for h in 0..k {
                *next.add(h) = C::ZERO.add(sum);
                sum += (*ends.add(h)).get();
                *ends.add(h) = C::ZERO.add(sum);
            }
        }
        // The key whose hole is 0: `min`, or the hole 0 itself.
        let base = if NARROW {
            hole(0)
        } else {
            ptr::read((&raw const min).cast::<H>())
        };
        if descents * SWAP_DESCENTS_PART > len && by_swaps::<H, E>(len) {
            swaps(holes, elems, base, k, ends, next, sparse);
        } else {
            cycles(holes, elems, base, k, ends, next, sparse);
        }
    }
}

/// Whether `len` elements of type `E` with keys of type `K` are placed by [`swaps`].
fn by_swaps<K, E>(len: usize) -> bool {
    let bytes = mem::size_of::<E>() + mem::size_of::<K>();
    len >= SWAP_MIN_LEN * (bytes / SWAP_BYTES_PER_STEP).max(1)
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
/// slice no longer fits in the caches.
///
/// # Safety
///
/// As for [`place`], with `ends` the end and `next` the start of every hole that holds keys,
/// and the end of every other hole zero.
unsafe fn cycles<H: IntegerKey, E, C: Count>(
    holes: *mut H,
    elems: *mut E,
    base: H,
    k: usize,
    ends: *mut C,
    next: *mut C,
    sparse: bool,
) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        if sparse {
            each_hole(ends, k, |h, end| fill(holes, elems, base, next, h, end));
        } else {
            for h in 0..k {
                fill(holes, elems, base, next, h, (*ends.add(h)).get());
            }
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
/// # Safety
///
/// As for [`cycles`].
unsafe fn swaps<H: IntegerKey, E, C: Count>(
    holes: *mut H,
    elems: *mut E,
    base: H,
    k: usize,
    ends: *mut C,
    next: *mut C,
    sparse: bool,
) {
    // The holes with free places, and where each ends: at most `k`, on the stack up to 4 KiB.
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let open = key_slots::<(usize, usize)>(k, &mut stack, &mut heap)
        .as_mut_ptr()
        .cast::<(usize, usize)>();
    let mut len = 0;
    // SAFETY: guaranteed by the caller; a hole's free places run from its next free place to
    // its end, every element's hole has one as long as it is not placed, and a swap of two
    // places, apart or the same, leaves both elements whole. `open` has room for every hole,
    // and its first `len` entries are written.
    unsafe {
        let mut add = |h: usize, end: usize| {
            if (*next.add(h)).get() < end {
                open.add(len).write((h, end));
                len += 1;
            }
        };
        if sparse {
            each_hole(ends, k, add);
        } else {
            for h in 0..k {
                add(h, (*ends.add(h)).get());
            }
        }
        while len > 0 {
            let mut kept = 0;
            for o in 0..len {
                let (h, end) = *open.add(o);
                let mut p = (*next.add(h)).get();
                while p < end {
                    let t = bump(next.add((*holes.add(p)).offset(base)));
                    if t != p {
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
