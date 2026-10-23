use core::mem::{self, MaybeUninit};
use core::{ptr, slice};

use super::counting::{
    Count, STACK_BYTES, Stack, bump, counters, exclusive_scan, in_order, key_slots, min_max,
    places4, rewrite, scatter,
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
/// Seven refinements keep the algorithm as cheap as LSD radix sort can be:
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
/// * **Short slices count their digits side by side.** Every pass costs the sum of its counts,
///   however few elements it moves: 256 of them for a digit of 8 bits. Below 80 elements per
///   8-bit digit (640 for 64-bit keys), the counts of up to eight such digits are kept in rows
///   of eight lanes, one lane per digit, 4 KiB in all, and turned into places by one running
///   sum per lane for all digits at once, which the compiler vectorizes, the first half of the
///   rows summed upwards and the second downwards from `n`: fewer passes than digits of
///   `log₂ n` bits take, for about the cost of one sum. This pays when the `s` passes saved
///   against digits of 6 bits make `n·s²` at least 144; otherwise, from 24 elements on, digits
///   of 6 bits are taken, whose 64 counts are summed while the pass before waits on memory. In
///   rows of lanes, signed values are sorted as their bits, the sign bit flipped once before the
///   passes and once after them rather than in every pass. 1.15–1.8 times faster on 16 to 256
///   random `u64`s, 1.1–1.45 times on 32 to 256 random `u32`s.
/// * **One digit is a counting sort.** When a single digit, starting at bit 0, holds every bit
///   in which the values differ, values with the same digit are equal, and the pass is this
///   crate's [`counting`](fn@super::counting) sort, which rewrites the slice from the counts
///   instead of moving it, with eight count tables for few distinct values: 2.4–5.4 times
///   faster with four distinct values.
/// * **Every count in one pass, and no copy back.** The counts of all digits are made in one
///   pass over the slice, with the loop over the digits unrolled for up to 8 of them, and for
///   the 13 and 16 digits of 64-bit keys on fewer than 64 elements; a digit all keys share,
///   which the count of one key's digit tells, is skipped; and the passes move the slice into
///   the buffer and back alternately, with one copy at the end only if the passes were odd in
///   number. The counts of a digit are turned into places four at a time, one addition per
///   four waiting for the one before it: 1.15–1.2 times faster on 100 random keys.
/// * **Runs of a digit do not wait on each other.** Incrementing a count waits for the
///   increment before it to reach memory, so keys sharing a digit one after the other make a
///   chain. The counts are made walking four quarters of the slice side by side, as they do
///   not depend on the order of the keys (1.6 times faster on nearly sorted input, whose high
///   digits keys share with their neighbours), and every pass hands out places four keys at a
///   time, all four counts read before any is written back (1.7–2.2 times faster with four
///   distinct wide keys, which share every digit with a quarter of the others).
/// * **Counts on the stack.** Up to 4 KiB of counts and of buffer need no allocation; counts
///   are `u16` up to 65,535 elements, which halves their memory, and `u32` up to `u32::MAX`.
///
/// Together, against the textbook LSD radix sort (one pass per byte, counted and moved through
/// a buffer that is copied back each time): 1.15–1.8 times faster on random integers from 1000
/// elements on (1.9–2.3 times at 100 elements and 2.7–3.4 times at 16 against one that
/// allocates its buffer), 1.7–8.8 times on 24-bit integers, 2.6–16 times on integers below `n`, 12–35
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
/// Against [`slice::sort_unstable`], from 1000 elements on: 1.8–5 times faster on random
/// `u32`s and up to 2.3 times on random `u64`s (as fast at 1000 elements on the same input over
/// and over, which the standard library's branches learn), 2.3–6 times on 24-bit integers,
/// 3–7.5 times on integers below `n`, 2.3–3.6 times with four distinct small values, 1.1–5
/// times on nearly sorted input, 2.3–5 times on sorted and all-equal input and 3–98 times on
/// reverse-sorted input; and 1.15–1.75 times faster on 100 to 256 random `u32`s. Its
/// weaknesses are short slices of wide keys, as every pass costs its counts however few
/// elements it moves: 100 random `u64`s sort 1.4 times slower on fresh input (2.3 times on the
/// same input over and over, which the standard library's branches learn), 16 to 32 elements
/// 1.6 (`u32`) to 3.2 (`u64`) times slower; and a few distinct values spread over wide keys,
/// which take every pass and which the standard library partitions away: four distinct random
/// `u32`s take 2–2.9 times as long, `u64`s 2.8–5 times.
/// For keys in a range of about `n` values, [`counting`](fn@super::counting) sort, one pass and
/// no buffer, is 1.07–1.8 times faster.
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
    let (plan, lanes) = Plan::choose(varying, len);
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
    } else if lanes == LANES {
        sort_values_in_lanes(arr, &plan);
    } else if len <= u16::MAX as usize {
        sort_values::<T, u16, 1>(arr, &plan);
    } else if len <= u32::MAX as usize {
        sort_values::<T, u32, 1>(arr, &plan);
    } else {
        sort_values::<T, usize, 1>(arr, &plan);
    }
}

/// [`sort_values`] with the counts in rows of [`LANES`] lanes, of `arr` as the bits of its
/// values: a signed integer's sign bit is flipped once before the passes and once after them,
/// not in every pass (1.05–1.15 times faster on 16 to
/// 48 random `i64`s), and every type is sorted
/// as the unsigned integer of its width.
///
/// Out of line: inlined into [`radix`], it made the passes of one table per digit slower.
#[inline(never)]
fn sort_values_in_lanes<T: IntegerKey>(arr: &mut [T], plan: &Plan) {
    const {
        assert!(
            mem::size_of::<T>() == mem::size_of::<T::Bits>()
                && mem::align_of::<T>() == mem::align_of::<T::Bits>()
        )
    };
    // The bits of every value are the value itself, read as `T::Bits`, with the same bits
    // flipped: none, or a signed integer's sign bit.
    // SAFETY: `T` and `T::Bits` have the same size, and neither has padding.
    let flip = unsafe { mem::transmute_copy::<T, T::Bits>(&arr[0]) }.xor(arr[0].bits());
    // SAFETY: `T` and `T::Bits` have the same size and alignment, and every value of `T` is a
    // valid unsigned integer `T::Bits`. Every bits written back below are those of a value of
    // `T`, and nothing reads `arr` as `T` before they are.
    let bits = unsafe { slice::from_raw_parts_mut(arr.as_mut_ptr().cast::<T::Bits>(), arr.len()) };
    let flipped = flip != <T::Bits as Bits>::ZERO;
    if flipped {
        for b in bits.iter_mut() {
            *b = b.xor(flip);
        }
    }
    sort_values::<T::Bits, u16, LANES>(bits, plan);
    if flipped {
        for b in bits.iter_mut() {
            *b = b.xor(flip);
        }
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
/// Seven refinements keep the algorithm as cheap as LSD radix sort can be:
///
/// * **Sorted and strictly descending input is recognised**, in the same pass that reads the
///   keys and finds the bits in which they differ: the slice is then left alone or reversed.
///   Input that is descending but not strictly is sorted like any other, as reversing it would
///   swap equal keys.
/// * **Only the bits in which keys differ are sorted, in digits sized by the slice**, as in
///   [`radix`].
/// * **Short slices count their digits side by side**, in rows of eight lanes, as in [`radix`]:
///   1.15–1.6 times faster on 16 to 256 random `u64` keys of 16-byte elements, 1.35–1.5 times
///   for 72-byte elements, 1.1–1.45 times on 48 to 256 random `u32` keys.
/// * **Runs of a digit do not wait on each other**, as in [`radix`]: the counts walk four
///   quarters side by side, and every pass places four elements at a time (1.7–2 times faster
///   with four distinct wide keys, 1.25–1.35 times on nearly sorted input, 1.15–1.4 times on 16
///   to 100 random keys).
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
///   allocation; counts are `u16` up to 65,535 elements.
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
/// Against [`slice::sort_by_key`], from 1000 elements on: 1.7–4.3 times faster on random `u32`
/// keys of 16-byte elements, 2.6–6 times for 72-byte elements, up to 2 times on random `u64`
/// keys; 3–7 times on keys of 24 bits or keys below `n`, 1.4–1.8 times with four distinct
/// small keys, 2.4–4 times on nearly sorted input, and as fast on sorted input; but four
/// distinct random keys take 1.3–2.6 times as long (`u32`) and 2.2–4.4 times (`u64`). On 100
/// random `u64` keys it is 1.4 times faster for 72-byte elements and 1.15 times slower for
/// 16-byte ones (2.2 times on the same input over and over), and on 16 elements up to 2.5
/// times slower on fresh input (6.5 times on the same input over and over).
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
    let (plan, lanes) = Plan::choose(varying, len);
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
        scatter(arr, keys, min, k);
    } else if len <= u16::MAX as usize {
        if lanes == LANES {
            sort_by_keys::<T, K::Bits, u16, LANES>(arr, slots, &plan);
        } else {
            sort_by_keys::<T, K::Bits, u16, 1>(arr, slots, &plan);
        }
    } else if len <= u32::MAX as usize {
        sort_by_keys::<T, K::Bits, u32, 1>(arr, slots, &plan);
    } else {
        sort_by_keys::<T, K::Bits, usize, 1>(arr, slots, &plan);
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

/// Short slices count their digits in rows of [`LANES`] lanes, one per digit, when their keys
/// take at most [`LANES`] digits of [`LANE_WIDTH`] bits...
const LANES: usize = 8;
const LANE_WIDTH: u32 = 8;

/// ...and the slice has at least [`LANES_MIN_LEN`] elements and fewer than this many per such
/// digit...
const LANE_LEN_PER_DIGIT: usize = 80;
const LANES_MIN_LEN: usize = 16;

/// ...and the `s` passes those digits save against digits of [`NARROW_WIDTH`] bits make `n·s²`
/// at least this many.
const LANE_GAIN: usize = 144;

/// Otherwise, short slices of at least [`NARROW_WIDTH_LEN`] elements, and fewer than
/// [`NARROW_LEN_PER_DIGIT`] per digit of `log₂ n` bits, take digits of this many bits, counted in
/// tables of their own: more passes, but each summing so few counts that the sum runs alongside
/// the pass before.
const NARROW_WIDTH: u32 = 6;
const NARROW_WIDTH_LEN: usize = 24;
const NARROW_LEN_PER_DIGIT: usize = 48;

impl Plan {
    /// Plans digits of at most `max` bits for keys whose bits differ from the first key's in
    /// `varying`.
    fn new<B: Bits>(varying: B, max: u32) -> Self {
        let low = varying.trailing_zeros();
        let span = B::BITS - varying.leading_zeros() - low;
        let digits = span.div_ceil(max);
        Self {
            low,
            width: span.div_ceil(digits),
            digits: digits as usize,
        }
    }

    /// Plans the digits for `len` keys whose bits differ from the first key's in `varying`,
    /// and how many lanes their counts have: [`LANES`] for digits side by side in rows, or 1 for
    /// one table per digit.
    fn choose<B: Bits>(varying: B, len: usize) -> (Self, usize) {
        let mut max = len.ilog2().clamp(SMALLEST_WIDTH, MAX_WIDTH);
        if len >= MIN_WIDTH_LEN {
            max = max.max(MIN_WIDTH);
        }
        let plan = Self::new(varying, max);
        if plan.digits == 1 || len < LANES_MIN_LEN {
            return (plan, 1);
        }
        let lanes = Self::new(varying, LANE_WIDTH);
        if lanes.digits > LANES || len >= LANE_LEN_PER_DIGIT * lanes.digits {
            return (plan, 1);
        }
        let narrow = Self::new(varying, NARROW_WIDTH);
        let saved = narrow.digits - lanes.digits;
        if len * saved * saved >= LANE_GAIN {
            (lanes, LANES)
        } else if len >= NARROW_WIDTH_LEN && len < NARROW_LEN_PER_DIGIT * plan.digits {
            (narrow, 1)
        } else {
            (plan, 1)
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

/// Where the counter of value `v` of digit `d` is in the counts: in the table of digit `d`, one
/// of `plan.radix()` counters, for `S = 1`; at lane `d` of the row of value `v`, one of `S`
/// counters, otherwise.
#[inline(always)]
fn slot<const S: usize>(plan: &Plan, d: usize, v: usize) -> usize {
    if S == 1 {
        d * plan.radix() + v
    } else {
        v * S + d
    }
}

/// How many counters the counts of `plan` take with `S` lanes.
fn counts_len<const S: usize>(plan: &Plan) -> usize {
    if S == 1 {
        plan.digits * plan.radix()
    } else {
        plan.radix() * S
    }
}

/// Counts the digits of the keys of `v` for every digit of `plan` in one pass, value `v` of
/// digit `d` into `counts[slot::<S>(plan, d, v)]`.
fn histogram<T: IntegerKey, C: Count, const S: usize>(v: &[T], plan: &Plan, counts: &mut [C]) {
    debug_assert!(S == 1 || plan.digits <= S);
    match plan.digits {
        1 => count_digits::<T, C, S>(v, plan, 1, counts),
        2 => count_digits::<T, C, S>(v, plan, 2, counts),
        3 => count_digits::<T, C, S>(v, plan, 3, counts),
        4 => count_digits::<T, C, S>(v, plan, 4, counts),
        5 => count_digits::<T, C, S>(v, plan, 5, counts),
        6 => count_digits::<T, C, S>(v, plan, 6, counts),
        7 => count_digits::<T, C, S>(v, plan, 7, counts),
        8 => count_digits::<T, C, S>(v, plan, 8, counts),
        13 => count_digits::<T, C, S>(v, plan, 13, counts),
        16 => count_digits::<T, C, S>(v, plan, 16, counts),
        digits => count_digits::<T, C, S>(v, plan, digits, counts),
    }
}

/// [`histogram`] for `digits` digits, a constant in every arm of its `match` but the last: the
/// loop over the digits is then unrolled.
#[inline(always)]
fn count_digits<T: IntegerKey, C: Count, const S: usize>(
    v: &[T],
    plan: &Plan,
    digits: usize,
    counts: &mut [C],
) {
    debug_assert!(counts.len() == counts_len::<S>(plan));
    let c = counts.as_mut_ptr();
    let mask = plan.mask();
    // The slice is walked as four quarters side by side: counts do not depend on the order of
    // the keys, and keys next to each other in the slice share their high digits when it is
    // nearly sorted, which would make each increment wait for the one before it.
    let quarter = v.len() / 4;
    let (a, rest) = v.split_at(quarter);
    let (b, rest) = rest.split_at(quarter);
    let (d, rest) = rest.split_at(quarter);
    let (e, rest) = rest.split_at(quarter);
    for (((a, b), d), e) in a.iter().zip(b).zip(d).zip(e) {
        for x in [a, b, d, e] {
            // SAFETY: as below.
            unsafe { count_key::<T, C, S>(x, plan, digits, mask, c) };
        }
    }
    for x in rest {
        // SAFETY: every digit is at most `mask`, so its counter is in `counts`.
        unsafe { count_key::<T, C, S>(x, plan, digits, mask, c) };
    }
}

/// Counts the `digits` digits of `x` into the counts at `c`.
///
/// # Safety
///
/// `c` must hold the [`counts_len`] counters of `plan`.
#[inline(always)]
unsafe fn count_key<T: IntegerKey, C: Count, const S: usize>(
    x: &T,
    plan: &Plan,
    digits: usize,
    mask: usize,
    c: *mut C,
) {
    let bits = x.bits();
    for d in 0..digits {
        // SAFETY: `d < digits` and the digit is at most `mask`, so the counter is in the
        // counts.
        unsafe { bump(c.add(slot::<S>(plan, d, bits.digit(plan.shift(d), mask)))) };
    }
}

/// Turns the counts of one digit into the place where the elements with each value of it
/// begin, unless all the elements have the same value, `any` being the value of one of them:
/// returns whether they do, which leaves nothing to do.
fn places<C: Count>(counts: &mut [C], len: usize, any: usize) -> bool {
    if counts[any].get() == len {
        return true;
    }
    table_places(counts);
    false
}

/// Turns the counts of one digit, in a table of their own, into the place where the elements
/// with each value of it begin.
fn table_places<C: Count>(counts: &mut [C]) {
    // A running sum for the 16 counts of 4-bit digits: summing them four at a time made 16
    // random `u64`s 1.2 times as slow.
    if counts.len() <= 16 {
        let mut sum = 0;
        for c in counts.iter_mut() {
            let count = c.get();
            *c = C::ZERO.add(sum);
            sum += count;
        }
    } else {
        exclusive_scan(counts);
    }
}

/// Returns the digits of `plan` all `len` keys share, bit `d` for digit `d`, from their counts in
/// rows of `S > 1` lanes, `first` being one of the keys: the count of its digit is then `len`.
/// Those digits take no pass.
fn shared_lanes<B: Bits, C: Count, const S: usize>(
    counts: &[C],
    plan: &Plan,
    len: usize,
    first: B,
) -> u64 {
    let mask = plan.mask();
    (0..plan.digits).fold(0, |shared, d| {
        let count = counts[slot::<S>(plan, d, first.digit(plan.shift(d), mask))].get();
        shared | (u64::from(count == len) << d)
    })
}

/// Turns the counts of every digit, side by side in rows of `S > 1` lanes, into the places where
/// the elements with each value of it begin, for `len` elements: `S` running sums at once, one
/// per lane, which the compiler vectorizes.
///
/// The first half of the rows is summed upwards from the first row, and the second half
/// downwards from the last, its places being `len` less the counts from there on: two sums
/// side by side, each waiting on its own addition per row, where one running sum waits on all.
fn lane_places<C: Count, const S: usize>(counts: &mut [C], len: usize) {
    // A row for every value of a digit: a power of two of them, at least 2.
    let rows = counts.as_chunks_mut::<S>().0;
    let (low, high) = rows.split_at_mut(rows.len() / 2);
    debug_assert!(low.len() == high.len());
    // The sums are kept as counters, so that they are added as wide as the counts.
    let (mut up, mut down) = ([C::ZERO; S], [C::ZERO.add(len); S]);
    for (a, b) in low.iter_mut().zip(high.iter_mut().rev()) {
        let (x, y) = (*a, *b);
        *a = up;
        for l in 0..S {
            up[l] = up[l].add(x[l].get());
            // The lanes beyond the last digit count nothing and stay at `len`; a digit's
            // counts add up to `len`, so the others stay at least 0.
            down[l] = C::ZERO.add(down[l].get() - y[l].get());
        }
        *b = down;
    }
}

/// Whether all `len` keys share digit `d`, `any` being its value in one of them: in rows of
/// lanes, whether `d` is in `shared`; with one table per digit, whether the count of `any` is
/// `len`, before the table becomes places.
#[inline(always)]
fn is_shared<C: Count, const S: usize>(
    counts: &[C],
    plan: &Plan,
    d: usize,
    shared: u64,
    len: usize,
    any: usize,
) -> bool {
    if S == 1 {
        counts[slot::<S>(plan, d, any)].get() == len
    } else {
        shared & (1 << d) != 0
    }
}

/// Prepares the pass of digit `d`, unless all keys share it ([`is_shared`]), which leaves
/// nothing to do: returns where its places are, the place of value `v` at `v·S` from there.
/// With one table per digit, the counts are turned into places only now, so that summing them
/// runs alongside the pass before, which waits on memory; with rows of lanes, they all are
/// already.
#[inline(always)]
fn pass_places<C: Count, const S: usize>(
    counts: &mut [C],
    plan: &Plan,
    d: usize,
    shared: u64,
    len: usize,
    any: usize,
) -> Option<*mut C> {
    if is_shared::<C, S>(counts, plan, d, shared, len, any) {
        return None;
    }
    if S == 1 {
        let radix = plan.radix();
        let table = &mut counts[d * radix..][..radix];
        table_places(table);
        return Some(table.as_mut_ptr());
    }
    // SAFETY: lane `d < S` of the first row is in `counts`.
    Some(unsafe { counts.as_mut_ptr().add(d) })
}

/// Sorts `arr` digit by digit, from the least significant, each pass stably moving the values
/// from one of `arr` and a buffer into the other.
fn sort_values<T: IntegerKey, C: Count, const S: usize>(arr: &mut [T], plan: &Plan) {
    let len = arr.len();
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let counts = counters::<C>(counts_len::<S>(plan), &mut stack, &mut heap);
    histogram::<T, C, S>(arr, plan, counts);
    let first = arr[0].bits();
    let mut shared = 0;
    if S > 1 {
        shared = shared_lanes::<T::Bits, C, S>(counts, plan, len, first);
        lane_places::<C, S>(counts, len);
    }
    let mut buf = Buffer::<T>::new(len);
    let mut src = arr.as_mut_ptr();
    let mut dst = buf.get();
    let mask = plan.mask();
    if S == 1 {
        for (d, counts) in counts.chunks_exact_mut(plan.radix()).enumerate() {
            let shift = plan.shift(d);
            if places(counts, len, first.digit(shift, mask)) {
                continue;
            }
            // SAFETY: `src` holds `len` values, and the counts are the places of the digit.
            unsafe { pass_values::<T, C, 1>(src, dst, len, counts.as_mut_ptr(), shift, mask) };
            mem::swap(&mut src, &mut dst);
        }
    } else {
        for d in 0..plan.digits {
            if shared & (1 << d) != 0 {
                continue;
            }
            // SAFETY: lane `d < S` of the first row is in `counts`, and the places of the
            // digit are there, every `S` counters.
            unsafe {
                let c = counts.as_mut_ptr().add(d);
                pass_values::<T, C, S>(src, dst, len, c, plan.shift(d), mask);
            }
            mem::swap(&mut src, &mut dst);
        }
    }
    if src != arr.as_mut_ptr() {
        // SAFETY: the last pass left the `len` sorted values in the buffer.
        unsafe { ptr::copy_nonoverlapping(src, arr.as_mut_ptr(), len) };
    }
}

/// Moves the `len` values at `src` to their places by the digit at `shift` into `dst`; the
/// place of a value whose digit is `v` is the counter at `c + v·S`, which is then incremented.
///
/// # Safety
///
/// `src` must hold `len` values, `dst` room for them, and the counters at `c` the places of
/// every value of the digit, each handing out places below `len`, once each.
#[inline(always)]
unsafe fn pass_values<T: IntegerKey, C: Count, const S: usize>(
    src: *const T,
    dst: *mut T,
    len: usize,
    c: *mut C,
    shift: u32,
    mask: usize,
) {
    let mut i = 0;
    while i + 4 <= len {
        // SAFETY: as below, four values at a time, their places handed out as one at a time
        // would.
        unsafe {
            let x = [0, 1, 2, 3].map(|j| *src.add(i + j));
            let b = x.map(|x| x.bits().digit(shift, mask) * S);
            let p = places4(b, b.map(|b| (*c.add(b)).get()));
            for j in 0..4 {
                *c.add(b[j]) = C::ZERO.add(p[j] + 1);
            }
            for j in 0..4 {
                dst.add(p[j]).write(x[j]);
            }
        }
        i += 4;
    }
    for i in i..len {
        // SAFETY: the digit is at most `mask`, so its counter is at `c`, and its place is below
        // `len`, handed out once.
        unsafe {
            let x = *src.add(i);
            let pos = bump(c.add(x.bits().digit(shift, mask) * S));
            dst.add(pos).write(x);
        }
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
fn sort_by_keys<T, B: Bits + IntegerKey, C: Count, const S: usize>(
    arr: &mut [T],
    keys: &mut [MaybeUninit<B>],
    plan: &Plan,
) {
    let len = arr.len();
    // SAFETY: all `len` keys have been written.
    let keys = unsafe { slice::from_raw_parts_mut(keys.as_mut_ptr().cast::<B>(), len) };
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let counts = counters::<C>(counts_len::<S>(plan), &mut stack, &mut heap);
    histogram::<B, C, S>(keys, plan, counts);
    // In rows of lanes, the digits all keys share, which take no pass, are found from the counts
    // before they become places; one table per digit is checked just before its pass.
    let mut shared = 0;
    if S > 1 {
        shared = shared_lanes::<B, C, S>(counts, plan, len, keys[0]);
        lane_places::<C, S>(counts, len);
    }
    let size = mem::size_of::<T>();
    if plan.digits * size >= GATHER_MIN_PASS_BYTES
        && len >= GATHER_MIN_LEN
        && len.saturating_mul(size) <= GATHER_MAX_BYTES
    {
        gather::<T, B, C, S>(arr, keys, plan, counts, shared);
    } else {
        move_with_keys::<T, B, C, S>(arr, keys, plan, counts, shared);
    }
}

/// Moves the elements of `arr` together with their keys in every pass.
fn move_with_keys<T, B: Bits + IntegerKey, C: Count, const S: usize>(
    arr: &mut [T],
    keys: &mut [B],
    plan: &Plan,
    counts: &mut [C],
    shared: u64,
) {
    let len = arr.len();
    let mut buf = Buffer::<T>::new(len);
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let key_buf = key_slots::<B>(len, &mut stack, &mut heap);
    let (mut src, mut dst) = (arr.as_mut_ptr(), buf.get());
    let (mut ks, mut kd) = (keys.as_mut_ptr(), key_buf.as_mut_ptr().cast::<B>());
    let mask = plan.mask();
    let first = keys[0];
    // The last pass needs no keys moved.
    let last = (0..plan.digits)
        .rev()
        .find(|&d| {
            !is_shared::<C, S>(
                counts,
                plan,
                d,
                shared,
                len,
                first.digit(plan.shift(d), mask),
            )
        })
        .unwrap_or(0);
    for d in 0..plan.digits {
        let any = first.digit(plan.shift(d), mask);
        let Some(c) = pass_places::<C, S>(counts, plan, d, shared, len, any) else {
            continue;
        };
        let shift = plan.shift(d);
        let move_keys = d != last;
        let mut i = 0;
        while i + 4 <= len {
            // SAFETY: as below, four at a time, their places handed out as one at a time would.
            unsafe {
                let k = [0, 1, 2, 3].map(|j| *ks.add(i + j));
                let b = k.map(|k| k.digit(shift, mask) * S);
                let p = places4(b, b.map(|b| (*c.add(b)).get()));
                for j in 0..4 {
                    *c.add(b[j]) = C::ZERO.add(p[j] + 1);
                }
                for j in 0..4 {
                    if move_keys {
                        kd.add(p[j]).write(k[j]);
                    }
                    ptr::copy_nonoverlapping(src.add(i + j), dst.add(p[j]), 1);
                }
            }
            i += 4;
        }
        for i in i..len {
            // SAFETY: `src` and `ks` hold `len` elements and keys; the digit is at most `mask`,
            // so the counter is in `counts`, and the places of each value of the digit are
            // handed out once each, all below `len`.
            unsafe {
                let k = *ks.add(i);
                let pos = bump(c.add(k.digit(shift, mask) * S));
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
fn gather<T, B: Bits + IntegerKey, C: Count, const S: usize>(
    arr: &mut [T],
    keys: &[B],
    plan: &Plan,
    counts: &mut [C],
    shared: u64,
) {
    let len = arr.len();
    let mut tagged: Vec<(B, u32)> = keys.iter().zip(0..).map(|(&k, i)| (k, i)).collect();
    let mut other: Vec<(B, u32)> = Vec::with_capacity(len);
    let (mut src, mut dst) = (tagged.as_mut_ptr(), other.as_mut_ptr());
    let mask = plan.mask();
    let first = keys[0];
    for d in 0..plan.digits {
        let any = first.digit(plan.shift(d), mask);
        let Some(c) = pass_places::<C, S>(counts, plan, d, shared, len, any) else {
            continue;
        };
        let shift = plan.shift(d);
        let mut i = 0;
        while i + 4 <= len {
            // SAFETY: as in `move_with_keys`.
            unsafe {
                let e = [0, 1, 2, 3].map(|j| *src.add(i + j));
                let b = e.map(|e| e.0.digit(shift, mask) * S);
                let p = places4(b, b.map(|b| (*c.add(b)).get()));
                for j in 0..4 {
                    *c.add(b[j]) = C::ZERO.add(p[j] + 1);
                }
                for j in 0..4 {
                    dst.add(p[j]).write(e[j]);
                }
            }
            i += 4;
        }
        for i in i..len {
            // SAFETY: as in `move_with_keys`.
            unsafe {
                let e = *src.add(i);
                let pos = bump(c.add(e.0.digit(shift, mask) * S));
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
