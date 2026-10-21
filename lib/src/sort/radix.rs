use core::mem::{self, MaybeUninit};
use core::{ptr, slice};

use super::counting::{
    Count, STACK_BYTES, Stack, bump, counters, in_order, key_slots, min_max, rewrite, scatter,
};
use super::key::IntegerKey;
use super::key::sealed::Bits;
use super::tim::Buffer;

/// Sorts a slice of integers, `char`s or `bool`s in place using LSD radix sort.
///
/// LSD radix sort, used on punched-card sorters since the 1920s, does not compare elements with
/// each other. It reads every key as a number of a few digits and sorts the slice by each digit
/// in turn, from the least significant to the most, with a stable counting pass: count the
/// values of the digit, turn the counts into the place where each value's elements begin, and
/// move every element, in order, to the next place of its digit in a buffer. As each pass keeps
/// the order the earlier ones made among equal digits, the slice ends up sorted by the whole
/// key. For `w`-bit keys and `r`-bit digits that is `⌈w / r⌉` passes of `O(n + 2^r)` each,
/// whatever the order of the input and however many distinct keys there are: unlike
/// [`counting`](fn@super::counting) sort, any range of keys can be sorted. Signed integers are
/// read with their sign bit flipped, so that they order as their bits do.
///
/// Five refinements keep the algorithm as cheap as LSD radix sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or one non-increasing run, 16 neighbours at a time with no exit inside a block,
///   which the compiler vectorizes; if it covers the whole slice, the slice is left alone or
///   reversed, and nothing is counted. Random input leaves the scan after the first block.
/// * **Only the bits in which keys differ are sorted, in digits sized by the slice.** One pass
///   ORs together every key's difference from the first; bits outside the span from the lowest
///   to the highest set bit of the result are the same in every key and need no pass. The span
///   is cut into the fewest digits of equal width of at most `log₂ n` bits (at least 8 from 64
///   elements on, at most 12), the digit size CLRS derives for radix sort: wide digits take
///   fewer passes, but their counts must stay small next to `n` and in the fastest caches.
///   Against digits of one byte with every pass of a byte all keys share skipped, this is
///   1.1–1.25 times faster on random integers from 10,000 elements on and 1.75–3.4 times on
///   integers below `n` (100,000 of them take two passes of 9 bits instead of three of 8).
/// * **One digit is a counting sort.** When a single digit, starting at bit 0, holds every bit
///   in which the values differ, values with the same digit are equal, and the pass is this
///   crate's [`counting`](fn@super::counting) sort, which rewrites the slice from the counts
///   instead of moving it, with eight count tables for few distinct values: 2.4–5.4 times
///   faster with four distinct values.
/// * **Every count in one pass, and no copy back.** The counts of all digits are made in one
///   pass over the slice, with the loop over the digits unrolled for up to 8 of them; a digit
///   all keys share is skipped; and the passes move the slice into the buffer and back
///   alternately, with one copy at the end only if the passes were odd in number.
/// * **Counts on the stack.** Up to 4 KiB of counts and of buffer need no allocation; counts
///   are `u32` up to `u32::MAX` elements.
///
/// Together, against the textbook LSD radix sort (one pass per byte, counted and moved through
/// a buffer that is copied back each time): 1.15–1.8 times faster on random integers from 100
/// elements on, 1.7–8.8 times on 24-bit integers, 2.6–16 times on integers below `n`, 12–35
/// times with four distinct values, and 58–190 times on sorted, reverse-sorted and all-equal
/// input.
///
/// # Properties
///
/// * **Stable** in effect: equal values cannot be told apart.
/// * **Not in place**: a buffer of `n` values and the counts of every digit, each on the stack up
///   to 4 KiB and otherwise allocated once.
/// * **Not adaptive**, except to sorted and reverse-sorted input, and to keys that differ in few
///   bits.
///
/// | Case                          | Passes over the slice                        | Time       |
/// |-------------------------------|----------------------------------------------|------------|
/// | Sorted, reverse-sorted, equal | one scan, reversing                          | `O(n)`     |
/// | Keys differing in `b` bits    | scan, bits, counts, `⌈b / r⌉` `r`-bit moves  | `O(n·b/r)` |
///
/// The scan of sorted input makes `n − 1` comparisons; random input leaves it after a block or
/// two. For `n = 1,000,000` random `u32`s: three passes of 11 bits; random `u64`s: six.
///
/// Against [`slice::sort_unstable`], from 1000 elements on: 2.3–4.8 times faster on random
/// `u32`s and 1.2–2.5 times on random `u64`s, 2.8–6.1 times on 24-bit integers, 3.5–6.7 times on
/// integers below `n`, 2.3–3.6 times with four distinct values, 1.6–2.9 times on nearly sorted
/// input, 2.3–4.7 times on sorted and all-equal input and 23–98 times on reverse-sorted input.
/// Short slices of wide keys are its weakness: 100 random `u32`s sort 1.4 times slower and 100
/// random `u64`s 2.8 times slower, and 16 elements up to 4 times slower, as every pass costs its
/// counts however few elements it moves. For keys in a range of about `n` values,
/// [`counting`](fn@super::counting) sort, one pass and no buffer, is 1.1–2.1 times faster.
///
/// # Examples
///
/// ```
/// use a::sort::radix;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// radix(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Any range of keys, signed ones too.
/// let mut v = [i64::MAX, -1, i64::MIN, 0, 1 << 40];
/// radix(&mut v);
/// assert_eq!(v, [i64::MIN, -1, 0, 1 << 40, i64::MAX]);
/// ```
pub fn radix<T: IntegerKey>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    if in_order(arr, |x, y| y < x) {
        return;
    }
    // Equal values cannot be told apart, so a non-increasing slice only needs reversing.
    if in_order(arr, |x, y| y > x) {
        arr.reverse();
        return;
    }
    let first = arr[0].bits();
    let varying = arr.iter().fold(<T::Bits as Bits>::ZERO, |acc, x| {
        acc.or(x.bits().xor(first))
    });
    let plan = Plan::new(varying, len);
    if plan.digits == 1 && plan.low == 0 {
        // One digit holds every bit in which the values differ, so values with the same digit
        // are equal: its pass is a counting sort, which rewrites the slice from the counts. The
        // digit starts at bit 0, so the values span at most one digit's range.
        let (min, max) = min_max(arr);
        let k = max.offset(min) + 1;
        if len <= u32::MAX as usize {
            rewrite::<T, u32>(arr, min, k);
        } else {
            rewrite::<T, usize>(arr, min, k);
        }
    } else if len <= u32::MAX as usize {
        sort_values::<T, u32>(arr, &plan);
    } else {
        sort_values::<T, usize>(arr, &plan);
    }
}

/// Sorts a slice in place by an integer, `char` or `bool` key using LSD radix sort. The sort is
/// stable: elements with equal keys keep their order.
///
/// The keys are sorted digit by digit as [`radix`] sorts values, and every pass moves the
/// elements, stably, with them. `key` is called exactly once per element: the keys are kept in a
/// buffer of their own as they are read, the passes read them from there, and once the last key
/// is read no code of the caller runs.
///
/// Five refinements keep the algorithm as cheap as LSD radix sort can be:
///
/// * **Sorted and strictly descending input is recognised**, in the same pass that reads the
///   keys and finds the bits in which they differ: the slice is then left alone or reversed.
///   Input that is descending but not strictly is sorted like any other, as reversing it would
///   swap equal keys.
/// * **Only the bits in which keys differ are sorted, in digits sized by the slice**, as in
///   [`radix`].
/// * **One digit is a counting sort.** When a single digit holds every bit in which the keys
///   differ, its pass is this crate's [`counting_by_key`](fn@super::counting_by_key), which
///   places eight segments of the slice side by side when there are few distinct keys:
///   1.3–1.8 times faster with four distinct keys for 16-byte elements.
/// * **Large elements are moved once.** When the passes would move at least 96 bytes of every
///   element (two passes of 64-byte elements, four of 24-byte ones) and the slice takes at most
///   16 MiB, the keys are sorted with the indices of their elements, and the elements are then
///   gathered in that order into a buffer: 1.25 times faster for 32-byte elements and up to 1.6
///   times for 72-byte ones. Beyond 16 MiB the gathering misses the cache, and moving the
///   elements in every pass is faster.
/// * **Buffers and counts on the stack.** Up to 4 KiB each of keys, elements and counts need no
///   allocation.
///
/// Together, against the textbook LSD radix sort (one pass per byte, calling `key` in each,
/// counted and moved through a buffer that is copied back each time), for 16-byte elements from
/// 1000 elements on: as fast to 1.2 times faster on random `u32` keys, 1.75–6 times on keys of 24
/// bits or keys below `n`, 6–14 times with four distinct keys, 1.5–5 times on nearly sorted
/// input and about 40 times on sorted input; for 72-byte elements 1.45–2.4 times faster on
/// random keys.
///
/// # Properties
///
/// * **Stable**: elements with equal keys keep their order.
/// * **Not in place**: buffers of `n` keys, `n` elements and, when the elements are gathered,
///   `2n` keys with their indices, plus the counts; each on the stack up to 4 KiB and otherwise
///   allocated once. Sorted input over 4 KiB of keys still allocates the keys, which are kept as
///   they are read.
/// * **Not adaptive**, except to sorted and strictly descending input, and to keys that differ
///   in few bits.
///
/// Against [`slice::sort_by_key`], from 1000 elements on: 2.8–4.3 times faster on random `u32`
/// keys of 16-byte elements, 3–5.2 times for 32-byte and 3.2–6.7 times for 72-byte elements,
/// 1.3–2 times on random `u64` keys; 3.4–6.8 times on keys of 24 bits or keys below `n`,
/// 1.4–1.8 times with four distinct keys, 1.9–3.4 times on reverse-sorted and nearly sorted
/// input, and as fast on sorted input. On 100 elements it is between 3.4 times faster and 2
/// times slower (random `u64` keys), and on 16 elements up to 7 times slower.
///
/// # Panics
///
/// If `key` panics, the panic is propagated and the slice is left as it was: no element has been
/// moved before the last key is read.
///
/// # Examples
///
/// ```
/// use a::sort::radix_by_key;
///
/// let mut v = [(3, 'a'), (1, 'b'), (3, 'c'), (0, 'd'), (1, 'e')];
/// radix_by_key(&mut v, |&(k, _)| k);
/// assert_eq!(v, [(0, 'd'), (1, 'b'), (1, 'e'), (3, 'a'), (3, 'c')]);
///
/// // Any range of keys: people by birth date, in seconds since 1970.
/// let mut people = vec![
///     ("Alan", -1_810_166_400i64),
///     ("Ada", -4_849_516_800),
///     ("Grace", -2_002_752_000),
/// ];
/// radix_by_key(&mut people, |&(_, born)| born);
/// let names: Vec<&str> = people.iter().map(|p| p.0).collect();
/// assert_eq!(names, ["Ada", "Grace", "Alan"]);
/// ```
pub fn radix_by_key<T, K, F>(arr: &mut [T], mut key: F)
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
    let slots = key_slots::<K::Bits>(len, &mut stack, &mut heap);
    let first = key(&arr[0]).bits();
    slots[0].write(first);
    let mut varying = <K::Bits as Bits>::ZERO;
    let mut prev = first;
    let (mut descent, mut no_descent) = (false, false);
    for (slot, x) in slots[1..].iter_mut().zip(&arr[1..]) {
        let x = key(x).bits();
        slot.write(x);
        varying = varying.or(x.xor(first));
        descent |= x < prev;
        no_descent |= x >= prev;
        prev = x;
    }
    if !descent {
        return;
    }
    // Strictly descending: reversing it keeps the order of equal keys, as there are none.
    if !no_descent {
        arr.reverse();
        return;
    }
    let plan = Plan::new(varying, len);
    if plan.digits == 1 {
        // One digit holds every bit in which the keys differ: its pass is a stable counting
        // sort by the digit.
        // SAFETY: all `len` keys have been written.
        let keys = unsafe { slice::from_raw_parts_mut(slots.as_mut_ptr().cast::<K::Bits>(), len) };
        let (mut min, mut max) = (plan.mask(), 0);
        for k in keys.iter_mut() {
            let d = k.digit(plan.low, plan.mask());
            min = min.min(d);
            max = max.max(d);
            *k = Bits::of(d);
        }
        let (min, k) = (<K::Bits as Bits>::of(min), max - min + 1);
        if len <= u32::MAX as usize {
            scatter::<T, K::Bits, u32>(arr, keys, min, k);
        } else {
            scatter::<T, K::Bits, usize>(arr, keys, min, k);
        }
    } else if len <= u32::MAX as usize {
        sort_by_keys::<T, K::Bits, u32>(arr, slots, &plan);
    } else {
        sort_by_keys::<T, K::Bits, usize>(arr, slots, &plan);
    }
}

/// How the keys are cut into digits: `digits` digits of `width` bits each, the lowest starting
/// at bit `low`. They cover every bit in which two keys differ, and no more.
struct Plan {
    low: u32,
    width: u32,
    digits: usize,
}

/// Digits have at most `log₂ n` bits, but no fewer than [`MIN_WIDTH`] from [`MIN_WIDTH_LEN`]
/// elements on, and no more than [`MAX_WIDTH`].
const MAX_WIDTH: u32 = 12;
const MIN_WIDTH: u32 = 8;
const MIN_WIDTH_LEN: usize = 64;
const SMALLEST_WIDTH: u32 = 4;

impl Plan {
    /// Plans the digits for `len` keys whose bits differ from the first key's in `varying`.
    fn new<B: Bits>(varying: B, len: usize) -> Self {
        let low = varying.trailing_zeros();
        let span = B::BITS - varying.leading_zeros() - low;
        let mut max = len.ilog2().clamp(SMALLEST_WIDTH, MAX_WIDTH);
        if len >= MIN_WIDTH_LEN {
            max = max.max(MIN_WIDTH);
        }
        let digits = span.div_ceil(max);
        Self {
            low,
            width: span.div_ceil(digits),
            digits: digits as usize,
        }
    }

    fn radix(&self) -> usize {
        1 << self.width
    }

    fn mask(&self) -> usize {
        self.radix() - 1
    }

    fn shift(&self, d: usize) -> u32 {
        self.low + d as u32 * self.width
    }
}

/// Counts the digits of the keys of `v` for every digit of `plan` in one pass, digit `d` into
/// `counts[d·radix..(d + 1)·radix]`.
fn histogram<T: IntegerKey, C: Count>(v: &[T], plan: &Plan, counts: &mut [C]) {
    match plan.digits {
        1 => count_digits::<T, C>(v, plan, 1, counts),
        2 => count_digits::<T, C>(v, plan, 2, counts),
        3 => count_digits::<T, C>(v, plan, 3, counts),
        4 => count_digits::<T, C>(v, plan, 4, counts),
        5 => count_digits::<T, C>(v, plan, 5, counts),
        6 => count_digits::<T, C>(v, plan, 6, counts),
        7 => count_digits::<T, C>(v, plan, 7, counts),
        8 => count_digits::<T, C>(v, plan, 8, counts),
        digits => count_digits::<T, C>(v, plan, digits, counts),
    }
}

/// [`histogram`] for `digits` digits, a constant in every arm of its `match` but the last: the
/// loop over the digits is then unrolled.
#[inline(always)]
fn count_digits<T: IntegerKey, C: Count>(v: &[T], plan: &Plan, digits: usize, counts: &mut [C]) {
    debug_assert!(counts.len() == digits * plan.radix());
    let c = counts.as_mut_ptr();
    let mask = plan.mask();
    for x in v {
        let bits = x.bits();
        for d in 0..digits {
            // SAFETY: `d < digits` and the digit is at most `mask`, so the counter is in
            // `counts`.
            unsafe { bump(c.add(d * plan.radix() + bits.digit(plan.shift(d), mask))) };
        }
    }
}

/// Turns the counts of one digit into the place where the elements with each value of it
/// begin. Returns whether all the elements have the same value, which leaves nothing to do.
fn places<C: Count>(counts: &mut [C], len: usize) -> bool {
    let mut sum = 0;
    for c in counts.iter_mut() {
        let count = c.get();
        if count == len {
            return true;
        }
        *c = C::ZERO.add(sum);
        sum += count;
    }
    false
}

/// Sorts `arr` digit by digit, from the least significant, each pass stably moving the values
/// from one of `arr` and a buffer into the other.
fn sort_values<T: IntegerKey, C: Count>(arr: &mut [T], plan: &Plan) {
    let len = arr.len();
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let counts = counters::<C>(plan.digits * plan.radix(), &mut stack, &mut heap);
    histogram(arr, plan, counts);
    let mut buf = Buffer::<T>::new(len);
    let mut src = arr.as_mut_ptr();
    let mut dst = buf.get();
    let mask = plan.mask();
    for (d, counts) in counts.chunks_exact_mut(plan.radix()).enumerate() {
        if places(counts, len) {
            continue;
        }
        let shift = plan.shift(d);
        let c = counts.as_mut_ptr();
        for i in 0..len {
            // SAFETY: `src` holds `len` values, initialized; the digit is at most `mask`, so the
            // counter is in `counts`, and the places of each value of the digit are handed out
            // once each, all below `len`.
            unsafe {
                let x = *src.add(i);
                let pos = bump(c.add(x.bits().digit(shift, mask)));
                dst.add(pos).write(x);
            }
        }
        mem::swap(&mut src, &mut dst);
    }
    if src != arr.as_mut_ptr() {
        // SAFETY: the last pass left the `len` sorted values in the buffer.
        unsafe { ptr::copy_nonoverlapping(src, arr.as_mut_ptr(), len) };
    }
}

/// The elements are not moved in every pass, but gathered once after their keys have been sorted
/// with their indices, when the passes would move at least this many bytes per element...
const GATHER_MIN_PASS_BYTES: usize = 96;

/// ...and the slice takes at most this many bytes, beyond which gathering misses the cache...
const GATHER_MAX_BYTES: usize = 16 << 20;

/// ...and has at least as many elements as make the digits at least [`MIN_WIDTH`] bits wide.
const GATHER_MIN_LEN: usize = MIN_WIDTH_LEN;

/// Sorts `arr` by `keys`, the bits of each element's key, digit by digit from the least
/// significant.
fn sort_by_keys<T, B: Bits + IntegerKey, C: Count>(
    arr: &mut [T],
    keys: &mut [MaybeUninit<B>],
    plan: &Plan,
) {
    let len = arr.len();
    // SAFETY: all `len` keys have been written.
    let keys = unsafe { slice::from_raw_parts_mut(keys.as_mut_ptr().cast::<B>(), len) };
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let counts = counters::<C>(plan.digits * plan.radix(), &mut stack, &mut heap);
    histogram(keys, plan, counts);
    let size = mem::size_of::<T>();
    if plan.digits * size >= GATHER_MIN_PASS_BYTES
        && len >= GATHER_MIN_LEN
        && len.saturating_mul(size) <= GATHER_MAX_BYTES
    {
        gather(arr, keys, plan, counts);
    } else {
        move_with_keys(arr, keys, plan, counts);
    }
}

/// Moves the elements of `arr` together with their keys in every pass.
fn move_with_keys<T, B: Bits + IntegerKey, C: Count>(
    arr: &mut [T],
    keys: &mut [B],
    plan: &Plan,
    counts: &mut [C],
) {
    let len = arr.len();
    let mut buf = Buffer::<T>::new(len);
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let key_buf = key_slots::<B>(len, &mut stack, &mut heap);
    let (mut src, mut dst) = (arr.as_mut_ptr(), buf.get());
    let (mut ks, mut kd) = (keys.as_mut_ptr(), key_buf.as_mut_ptr().cast::<B>());
    let mask = plan.mask();
    let last = (0..plan.digits)
        .rev()
        .find(|&d| {
            // A digit whose value is the same for all keys takes no pass.
            let c = &counts[d * plan.radix()..(d + 1) * plan.radix()];
            !c.contains(&C::ZERO.add(len))
        })
        .unwrap_or(0);
    for (d, counts) in counts.chunks_exact_mut(plan.radix()).enumerate() {
        if places(counts, len) {
            continue;
        }
        let shift = plan.shift(d);
        let c = counts.as_mut_ptr();
        let move_keys = d != last;
        for i in 0..len {
            // SAFETY: `src` and `ks` hold `len` elements and keys; the digit is at most `mask`,
            // so the counter is in `counts`, and the places of each value of the digit are
            // handed out once each, all below `len`. The last pass needs no keys moved.
            unsafe {
                let k = *ks.add(i);
                let pos = bump(c.add(k.digit(shift, mask)));
                if move_keys {
                    kd.add(pos).write(k);
                }
                ptr::copy_nonoverlapping(src.add(i), dst.add(pos), 1);
            }
        }
        mem::swap(&mut src, &mut dst);
        mem::swap(&mut ks, &mut kd);
    }
    if src != arr.as_mut_ptr() {
        // SAFETY: the last pass left the `len` sorted elements in the buffer, and `arr` holds
        // only stale bitwise copies of them.
        unsafe { ptr::copy_nonoverlapping(src, arr.as_mut_ptr(), len) };
    }
}

/// Sorts each key with the index of its element, then gathers the elements in that order into
/// a buffer and copies them back: one move of each element instead of one per pass.
fn gather<T, B: Bits + IntegerKey, C: Count>(
    arr: &mut [T],
    keys: &[B],
    plan: &Plan,
    counts: &mut [C],
) {
    let len = arr.len();
    let mut tagged: Vec<(B, u32)> = keys.iter().zip(0..).map(|(&k, i)| (k, i)).collect();
    let mut other: Vec<(B, u32)> = Vec::with_capacity(len);
    let (mut src, mut dst) = (tagged.as_mut_ptr(), other.as_mut_ptr());
    let mask = plan.mask();
    for (d, counts) in counts.chunks_exact_mut(plan.radix()).enumerate() {
        if places(counts, len) {
            continue;
        }
        let shift = plan.shift(d);
        let c = counts.as_mut_ptr();
        for i in 0..len {
            // SAFETY: as in `move_with_keys`.
            unsafe {
                let e = *src.add(i);
                let pos = bump(c.add(e.0.digit(shift, mask)));
                dst.add(pos).write(e);
            }
        }
        mem::swap(&mut src, &mut dst);
    }
    let mut buf = Buffer::<T>::new(len);
    let out = buf.get();
    let base = arr.as_ptr();
    for i in 0..len {
        // SAFETY: the indices are a permutation of `0..len`.
        unsafe { ptr::copy_nonoverlapping(base.add((*src.add(i)).1 as usize), out.add(i), 1) };
    }
    // SAFETY: `out` holds every element of `arr` exactly once.
    unsafe { ptr::copy_nonoverlapping(out, arr.as_mut_ptr(), len) };
}
