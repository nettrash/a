use core::mem::{self, MaybeUninit};
use core::{ptr, slice};

use super::tim::Buffer;

/// Sorts a slice of integers, `char`s or `bool`s in place using counting sort.
///
/// Counting sort, described by Harold Seward in 1954, does not compare elements with each other.
/// It finds the smallest and the largest value, counts how many times each of the `k` values in
/// between occurs, and rewrites the slice from the counts: so many copies of the smallest value,
/// then so many of the next, and so on. That takes `O(n + k)` time and `O(k)` memory whatever the
/// order of the input, which beats every comparison sort when `k` is not much larger than `n`:
/// bytes, small integers, values in a known range.
///
/// Equal values of these types cannot be told apart, so rewriting them from counts sorts the
/// slice exactly as moving them would. [`counting_by_key`] sorts any type by such a key, moving
/// the elements and keeping equal keys in order.
///
/// Four refinements keep the algorithm as cheap as counting sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or one non-increasing run, 16 neighbours at a time with no exit inside a block,
///   which the compiler vectorizes; if it covers the whole slice, the slice is left alone or
///   reversed, and nothing is counted. Random input leaves the scan after the first block.
/// * **Eight count tables for small ranges.** Incrementing a count waits for the increment
///   before it to reach memory, so a run of equal values, or a few values repeating, makes a
///   chain of dependent increments. With at most 2048 values and at least 16 elements per value,
///   the slice is counted into eight tables, one element after the other into the next, and the
///   tables are added up: eight chains at once, 3–5 times faster with a few distinct values and
///   on sorted runs of bytes.
/// * **Values written ahead of their counts.** Each value is written 4 times (8 for one- and
///   two-byte types, in one store) whatever its count, and only a count above that takes a
///   branch; the copies beyond the count are overwritten by the next value. Branching on every
///   count mispredicts when the range has about as many values as the slice has elements, where
///   counts of 0, 1 and 2 are all common: this is 2.5–5 times faster there. A range with 32 or more
///   values per element is rewritten instead by skipping blocks of 8 empty counts at once.
/// * **Counts on the stack.** Up to 4 KiB of counts (1024 values, or 128 values in eight tables)
///   need no allocation; counts are `u32` up to `u32::MAX` elements, halving the memory larger
///   ranges take.
///
/// Together, against the textbook counting sort (scan for the range, count, branch on every
/// count to rewrite): 3–5.5 times faster on random integers with about as many distinct values
/// as elements, 1.9–3.1 times with four times as many and 1.05–1.45 times with sixteen times
/// fewer; 1.7–3.1 times faster with four distinct values, 3.1–4.7 times on nearly sorted input,
/// and 9–80 times on sorted, reverse-sorted and all-equal input. Random bytes cost the same as
/// with the textbook sort from 10,000 elements on; 16 random bytes are 1.4 times slower, as
/// writing ahead does not pay for a range 15 times longer than the slice.
///
/// # Properties
///
/// * **Stable** in effect: equal values cannot be told apart.
/// * **Not in place**: `O(k)` counts, on the stack up to 4 KiB and otherwise allocated once.
/// * **Not adaptive**, except to sorted and reverse-sorted input.
///
/// | Case                          | Passes over the slice                           | Time       |
/// |-------------------------------|-------------------------------------------------|------------|
/// | Sorted, reverse-sorted, equal | one scan (`n − 1` comparisons), reversing        | `O(n)`     |
/// | Any other input               | a block or two of the scan, range, count, write | `O(n + k)` |
///
/// Against [`slice::sort_unstable`], for `u32` and `u64` from 100 elements on: 3.6–9 times faster
/// on random integers with about as many distinct values as elements, 2.6–9.5 times with
/// sixteen times fewer, 1.7–3.8 times with four times as many, 2.5–4 times with four distinct
/// values; 2–10 times faster on nearly sorted input, 1.6–4.8 times on sorted and all-equal input,
/// and 12–99 times on reverse-sorted input; random bytes sort 9–15 times faster from 1000
/// elements on. On 16 elements it is up to 2 times slower on sorted, nearly sorted and all-equal
/// input, and this crate's [`drift`](fn@super::drift) sort, which insertion sorts short slices,
/// is up to twice as fast on most input.
///
/// # Panics
///
/// Panics if the range is too large to count: if `max − min + 1` exceeds both 65,536 and four
/// times the length of the slice, whatever the order of the slice. The slice is then left as it
/// was. Any range of a one- or two-byte type is allowed.
///
/// # Examples
///
/// ```
/// use a::sort::counting;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// counting(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Any integer type, signed ones too, as long as the range is small.
/// let mut v = [i64::MIN + 2, i64::MIN, i64::MIN + 1];
/// counting(&mut v);
/// assert_eq!(v, [i64::MIN, i64::MIN + 1, i64::MIN + 2]);
///
/// let mut letters: Vec<char> = "counting".chars().collect();
/// counting(&mut letters);
/// assert_eq!(letters.iter().collect::<String>(), "cginnotu");
/// ```
///
/// A range too large for the slice panics:
///
/// ```should_panic
/// use a::sort::counting;
///
/// let mut v = [0u32, 1_000_000];
/// counting(&mut v);
/// ```
pub fn counting<T: CountingKey>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    // Sorted input has nothing to count, but its range is checked all the same: the limit does
    // not depend on the order of the elements.
    if in_order(arr, |x, y| y < x) {
        range(arr[0], arr[len - 1], len);
        return;
    }
    // Equal values cannot be told apart, so a non-increasing slice only needs reversing.
    if in_order(arr, |x, y| y > x) {
        range(arr[len - 1], arr[0], len);
        arr.reverse();
        return;
    }
    let (min, max) = min_max(arr);
    let k = range(min, max, len);
    if len <= u32::MAX as usize {
        rewrite::<T, u32>(arr, min, k);
    } else {
        rewrite::<T, usize>(arr, min, k);
    }
}

/// Sorts a slice in place by an integer, `char` or `bool` key using counting sort. The sort is
/// stable: elements with equal keys keep their order.
///
/// This is counting sort as it is used inside radix sort: the keys are counted, the counts are
/// turned into the place where each key's elements begin, and every element is moved, in order,
/// to the next place of its key in a buffer of `n` elements, which is then copied back. It takes
/// `O(n + k)` time for `k = max − min + 1` possible keys, and `O(n + k)` memory.
///
/// `key` is called exactly once per element: the keys are kept in a buffer of their own as they
/// are read. That costs `n` keys of memory and a few percent on input whose elements carry their
/// keys in order, but it makes the sort as fast as possible on random keys, lets the counts and
/// places be trusted without checking them, and leaves the slice untouched if `key` panics.
///
/// Four refinements keep the algorithm as cheap as counting sort can be:
///
/// * **Sorted and strictly descending input is recognised**, in the same pass that reads the keys
///   and finds the range: the slice is then left alone or reversed. Input that is descending but
///   not strictly is sorted like any other, as reversing it would swap equal keys.
/// * **Eight segments for small ranges.** Placing an element waits for the place handed out
///   before it to the same key, so a few keys repeating make a chain of dependent increments.
///   With at most 256 keys and at least 16 elements per key, the slice is cut into eight
///   segments, each with its own places for every key after those of the segments before it,
///   and the segments are counted and placed side by side: eight chains at once, 1.6–2.3 times
///   faster with four distinct keys and 2.4–2.9 times on runs of bytes. More keys than 256 gain
///   nothing and lose up to 1.25 times to the extra places.
/// * **Keys read once.** Counting and placing read the keys from their buffer instead of from
///   the elements: 1.05–1.25 times faster than calling `key` again on random keys with about as
///   many distinct keys as elements, or four times as many.
/// * **Buffers on the stack.** Up to 4 KiB each of elements, keys and places (counts are `u32`
///   up to `u32::MAX` elements) need no allocation.
///
/// Together, against the textbook counting sort (scan for the range, count, place, copy back,
/// calling `key` in each pass): 1.05–1.6 times faster on random keys, 1.35–2.2 times with four
/// distinct keys, 2.6–3.1 times on runs of bytes and 6–20 times on sorted and all-equal input;
/// but 1.02–1.15 times slower on reverse-sorted input with repeated keys, on nearly sorted
/// input, on random bytes from 10,000 elements on and on 1000–10,000 random keys with sixteen
/// times fewer keys than elements, where keeping the keys costs more than it saves.
///
/// # Properties
///
/// * **Stable**: elements with equal keys keep their order.
/// * **Not in place**: a buffer of `n` elements, one of `n` keys and `O(k)` places, each on the
///   stack up to 4 KiB and otherwise allocated once. Sorted input over 4 KiB of keys still
///   allocates the keys, which are kept as they are read.
/// * **Not adaptive**, except to sorted and strictly descending input.
///
/// | Case                         | Calls to `key` | Passes over the slice                | Time       |
/// |------------------------------|----------------|--------------------------------------|------------|
/// | Sorted, strictly descending  | `n`            | one, reversing                       | `O(n)`     |
/// | Any other input              | `n`            | read keys, count, place, copy back   | `O(n + k)` |
///
/// Against [`slice::sort_by_key`], with 16-byte elements from 100 elements on: 3–7.6 times faster
/// on random `u32` keys with sixteen times fewer distinct keys than elements, 2.9–7 times with
/// about as many, 1.85–4.2 times with four times as many; 1.6–2.3 times faster with four
/// distinct keys, 1.9–5.7 times on reverse-sorted input and 1–9 times on nearly sorted input;
/// as fast on sorted and all-equal input with `u32` keys, and 1.3–1.7 times faster with `u8`
/// keys from 1000 elements on, where random bytes sort 4.4–8 times faster. On 16 elements it is
/// between 1.9 times faster and 2 times slower.
///
/// # Panics
///
/// Panics if the range of keys is too large to count, as [`counting`] does; the slice is then
/// left as it was. If `key` panics, the panic is propagated and the slice is left as it was: no
/// element has been moved before the last key is read.
///
/// # Examples
///
/// ```
/// use a::sort::counting_by_key;
///
/// let mut v = [(3, 'a'), (1, 'b'), (3, 'c'), (0, 'd'), (1, 'e')];
/// counting_by_key(&mut v, |&(k, _)| k);
/// assert_eq!(v, [(0, 'd'), (1, 'b'), (1, 'e'), (3, 'a'), (3, 'c')]);
///
/// // Words by length; equal lengths keep their order.
/// let mut words = vec!["pear", "fig", "banana", "kiwi", "apple", "lime"];
/// counting_by_key(&mut words, |w| w.len());
/// assert_eq!(words, ["fig", "pear", "kiwi", "lime", "apple", "banana"]);
/// ```
pub fn counting_by_key<T, K, F>(arr: &mut [T], mut key: F)
where
    K: CountingKey,
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
    let (mut descent, mut no_descent) = (false, false);
    for (slot, x) in slots[1..].iter_mut().zip(&arr[1..]) {
        let x = key(x);
        slot.write(x);
        min = min.min(x);
        max = max.max(x);
        descent |= x < prev;
        no_descent |= x >= prev;
        prev = x;
    }
    // SAFETY: all `len` slots have been written.
    let keys = unsafe { slice::from_raw_parts(slots.as_ptr().cast::<K>(), len) };
    let k = range(min, max, len);
    if !descent {
        return;
    }
    // Strictly descending: reversing it keeps the order of equal keys, as there are none.
    if !no_descent {
        arr.reverse();
        return;
    }
    if len <= u32::MAX as usize {
        scatter::<T, K, u32>(arr, keys, min, k);
    } else {
        scatter::<T, K, usize>(arr, keys, min, k);
    }
}

/// A type counting sort can count: a primitive integer, `char` or `bool`.
///
/// It is sealed: it cannot be implemented outside this crate.
pub trait CountingKey: sealed::Key {}

mod sealed {
    /// What counting sort needs to know about a key type: its order as an offset from the
    /// smallest key.
    pub trait Key: Copy + Ord {
        /// `self − min`, for `min <= self`.
        fn distance(self, min: Self) -> u128;
        /// `self − min`, for `min <= self` and a distance that fits in `usize`.
        fn offset(self, min: Self) -> usize;
        /// `min + offset`, or any value of the type if that is not one.
        fn nth(min: Self, offset: usize) -> Self;
    }
}

macro_rules! integer_key {
    ($($t:ty => $u:ty),* $(,)?) => {$(
        impl sealed::Key for $t {
            #[inline(always)]
            fn distance(self, min: Self) -> u128 {
                (self as $u).wrapping_sub(min as $u) as u128
            }
            #[inline(always)]
            fn offset(self, min: Self) -> usize {
                (self as $u).wrapping_sub(min as $u) as usize
            }
            #[inline(always)]
            fn nth(min: Self, offset: usize) -> Self {
                (min as $u).wrapping_add(offset as $u) as $t
            }
        }
        impl CountingKey for $t {}
    )*};
}

integer_key! {
    u8 => u8, u16 => u16, u32 => u32, u64 => u64, u128 => u128, usize => usize,
    i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128, isize => usize,
}

impl sealed::Key for char {
    #[inline(always)]
    fn distance(self, min: Self) -> u128 {
        u128::from(self as u32 - min as u32)
    }
    #[inline(always)]
    fn offset(self, min: Self) -> usize {
        (self as u32).wrapping_sub(min as u32) as usize
    }
    #[inline(always)]
    fn nth(min: Self, offset: usize) -> Self {
        // An offset into the surrogates, which no element has, gives `min`.
        char::from_u32((min as u32).wrapping_add(offset as u32)).unwrap_or(min)
    }
}
impl CountingKey for char {}

impl sealed::Key for bool {
    #[inline(always)]
    fn distance(self, min: Self) -> u128 {
        u128::from(self as u8 - min as u8)
    }
    #[inline(always)]
    fn offset(self, min: Self) -> usize {
        usize::from(self).wrapping_sub(usize::from(min))
    }
    #[inline(always)]
    fn nth(min: Self, offset: usize) -> Self {
        usize::from(min) + offset != 0
    }
}
impl CountingKey for bool {}

/// Every range of at most this many values is allowed, whatever the length of the slice.
const RANGE_MIN: usize = 1 << 16;

/// Beyond [`RANGE_MIN`], a range is allowed if it has at most this many values per element.
const RANGE_PER_ELEMENT: usize = 4;

/// Returns the number of values from `min` to `max`, or panics if there are too many to count
/// for `len` elements.
fn range<K: CountingKey>(min: K, max: K, len: usize) -> usize {
    let limit = RANGE_MIN.max(len.saturating_mul(RANGE_PER_ELEMENT));
    let distance = max.distance(min);
    match usize::try_from(distance) {
        Ok(d) if d < limit => d + 1,
        _ => panic!(
            "counting sort: {} possible values are too many for {len} elements (at most {limit})",
            distance.saturating_add(1)
        ),
    }
}

/// Whether no two neighbours `x, y` of `v` are `out(x, y)`. The pairs are checked in blocks of
/// 16 with no exit inside a block, which the compiler vectorizes; random input leaves after the
/// first block.
#[inline(always)]
fn in_order<T: CountingKey>(v: &[T], out: impl Fn(T, T) -> bool) -> bool {
    const BLOCK: usize = 16;
    let mut i = 0;
    while i + BLOCK < v.len() {
        let b = &v[i..i + BLOCK + 1];
        let mut any = false;
        for j in 0..BLOCK {
            any |= out(b[j], b[j + 1]);
        }
        if any {
            return false;
        }
        i += BLOCK;
    }
    v[i..].windows(2).all(|w| !out(w[0], w[1]))
}

/// Returns the smallest and the largest value of a nonempty slice.
#[inline(always)]
fn min_max<T: CountingKey>(v: &[T]) -> (T, T) {
    let first = v[0];
    v.iter()
        .fold((first, first), |(min, max), &x| (min.min(x), max.max(x)))
}

/// A counter of elements: `u32` while the slice is short enough, which halves the memory the
/// counts take, and `usize` otherwise.
///
/// # Safety
///
/// All-zero bytes must be a valid zero.
unsafe trait Count: Copy + Eq {
    const ZERO: Self;
    fn add(self, n: usize) -> Self;
    fn get(self) -> usize;
}

unsafe impl Count for u32 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn add(self, n: usize) -> Self {
        self + n as u32
    }
    #[inline(always)]
    fn get(self) -> usize {
        self as usize
    }
}

unsafe impl Count for usize {
    const ZERO: Self = 0;
    #[inline(always)]
    fn add(self, n: usize) -> Self {
        self + n
    }
    #[inline(always)]
    fn get(self) -> usize {
        self
    }
}

/// Room for counters on the stack: as much as the other sorts of this crate keep their buffers
/// in.
const STACK_BYTES: usize = 4096;

#[repr(C, align(16))]
struct Stack([MaybeUninit<u8>; STACK_BYTES]);

/// Returns `len` zeroed counters: in `stack` if they fit, and otherwise in `heap`, allocated.
#[inline(always)]
fn counters<'a, C: Count>(len: usize, stack: &'a mut Stack, heap: &'a mut Vec<C>) -> &'a mut [C] {
    if len <= STACK_BYTES / mem::size_of::<C>() {
        let p = stack.0.as_mut_ptr().cast::<C>();
        // SAFETY: the stack has room for `len` counters and is aligned for them (`C` is `u32`
        // or `usize`), and all-zero bytes are a valid zero.
        unsafe {
            ptr::write_bytes(p, 0, len);
            slice::from_raw_parts_mut(p, len)
        }
    } else {
        *heap = vec![C::ZERO; len];
        heap
    }
}

/// Returns room for `len` keys: in `stack` if they fit, and otherwise in `heap`, allocated.
#[inline(always)]
fn key_slots<'a, K: CountingKey>(
    len: usize,
    stack: &'a mut Stack,
    heap: &'a mut Vec<K>,
) -> &'a mut [MaybeUninit<K>] {
    const { assert!(mem::align_of::<K>() <= mem::align_of::<Stack>()) };
    // Every key type has a nonzero size.
    if len <= STACK_BYTES / mem::size_of::<K>() {
        // SAFETY: the stack has room for `len` keys and is aligned for them.
        unsafe { slice::from_raw_parts_mut(stack.0.as_mut_ptr().cast(), len) }
    } else {
        // Its length stays 0: the keys are `Copy`, and nothing reads them through it.
        *heap = Vec::with_capacity(len);
        &mut heap.spare_capacity_mut()[..len]
    }
}

/// Counts the values of `arr` into eight tables, one value after the other into the next
/// table, while the range has at most this many values, so that the eight fit in the
/// processor's fastest cache...
const TABLES_MAX_RANGE: usize = 2048;

/// ...and the slice has at least this many elements per value, so that clearing and adding up
/// the tables costs little.
const TABLES_MIN_PER_VALUE: usize = 16;

const TABLES: usize = 8;

/// The counts are rewritten into the slice by a loop that skips empty blocks of counts when the
/// range has at least this many values per element.
const SPARSE_PER_ELEMENT: usize = 32;

/// Counts the values of `arr`, all within the `k` values from `min`, and rewrites `arr` from
/// the counts.
fn rewrite<T: CountingKey, C: Count>(arr: &mut [T], min: T, k: usize) {
    let len = arr.len();
    let tables = if k <= TABLES_MAX_RANGE && len >= TABLES_MIN_PER_VALUE * k {
        TABLES
    } else {
        1
    };
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let counts = counters::<C>(tables * k, &mut stack, &mut heap);
    if tables == TABLES {
        count::<T, C, TABLES>(arr, min, counts, k);
    } else {
        count::<T, C, 1>(arr, min, counts, k);
    }
    let counts = &counts[..k];
    if k >= SPARSE_PER_ELEMENT.saturating_mul(len) {
        write_sparse(arr, min, counts);
    } else if mem::size_of::<T>() <= 2 {
        write_ahead::<T, C, 8>(arr, min, counts);
    } else {
        write_ahead::<T, C, 4>(arr, min, counts);
    }
}

/// Counts the values of `v`, all within the `k` values from `min`, into `N` tables of `k`
/// counters, value `i` into table `i mod N`, and adds them up into the first. `N` divides 8.
///
/// A count depends on the one before it: with one table, a run of equal values waits for each
/// increment to reach memory before the next can read it. With `N` tables, `N` such chains run
/// at once.
///
/// The slice is walked 8 elements at a time whatever `N` is: left to the compiler, the loop for
/// one table was not unrolled, and counted 1.4 times slower.
#[inline(always)]
fn count<T: CountingKey, C: Count, const N: usize>(v: &[T], min: T, counts: &mut [C], k: usize) {
    const STEP: usize = 8;
    debug_assert!(STEP.is_multiple_of(N) && counts.len() == N * k);
    let c = counts.as_mut_ptr();
    let (chunks, rest) = v.as_chunks::<STEP>();
    for chunk in chunks {
        for (j, x) in chunk.iter().enumerate() {
            // SAFETY: `min <= x <= max`, so `x.offset(min) < k` and the counter is in table
            // `j mod N < N`.
            unsafe { bump(c.add(j % N * k + x.offset(min))) };
        }
    }
    for x in rest {
        // SAFETY: as above, in table 0.
        unsafe { bump(c.add(x.offset(min))) };
    }
    if N > 1 {
        let (first, rest) = counts.split_at_mut(k);
        for table in rest.chunks_exact(k) {
            for (sum, &c) in first.iter_mut().zip(table) {
                *sum = sum.add(c.get());
            }
        }
    }
}

/// Increments the counter at `c` and returns its value before.
///
/// # Safety
///
/// `c` must be valid for reads and writes.
#[inline(always)]
unsafe fn bump<C: Count>(c: *mut C) -> usize {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let before = *c;
        *c = before.add(1);
        before.get()
    }
}

/// Rewrites `v` from `counts`, `counts[o]` copies of `min + o` after each other.
///
/// Every value is first written `W` times without looking at its count, and only a count
/// greater than `W` takes a branch. Whatever was written beyond the count is overwritten by the
/// values after it. Branching on every count instead mispredicts on a range of about as many
/// values as elements, where counts of 0, 1 and 2 are all common.
#[inline(always)]
fn write_ahead<T: CountingKey, C: Count, const W: usize>(v: &mut [T], min: T, counts: &[C]) {
    let len = v.len();
    let base = v.as_mut_ptr();
    let mut pos = 0;
    let mut o = 0;
    while o < counts.len() && pos + W <= len {
        let c = counts[o].get();
        let x = T::nth(min, o);
        // SAFETY: `pos + W <= len`, and the counts add up to `len`, so `pos + c <= len`.
        unsafe {
            for j in 0..W {
                base.add(pos + j).write(x);
            }
            if c > W {
                slice::from_raw_parts_mut(base.add(pos + W), c - W).fill(x);
            }
        }
        pos += c;
        o += 1;
    }
    for (o, &c) in counts.iter().enumerate().skip(o) {
        let c = c.get();
        v[pos..pos + c].fill(T::nth(min, o));
        pos += c;
    }
}

/// Rewrites `v` from `counts` like [`write_ahead`], for a range with many more values than
/// elements: blocks of 8 empty counts are skipped with one test.
#[inline(always)]
fn write_sparse<T: CountingKey, C: Count>(v: &mut [T], min: T, counts: &[C]) {
    const BLOCK: usize = 8;
    let mut pos = 0;
    let (blocks, rest) = counts.as_chunks::<BLOCK>();
    let mut o = 0;
    for block in blocks {
        // Not `any`, which stops at the first nonzero count and is not vectorized.
        if block.iter().fold(false, |any, &c| any | (c != C::ZERO)) {
            for (j, &c) in block.iter().enumerate() {
                let c = c.get();
                v[pos..pos + c].fill(T::nth(min, o + j));
                pos += c;
            }
        }
        o += BLOCK;
    }
    for (j, &c) in rest.iter().enumerate() {
        let c = c.get();
        v[pos..pos + c].fill(T::nth(min, o + j));
        pos += c;
    }
}

/// The slice is placed as this many segments, each with its own places, while the range has
/// at most this many keys...
const SEGMENTS: usize = 8;

const SEGMENTS_MAX_RANGE: usize = 256;

/// ...and the slice has at least this many elements per key.
const SEGMENTS_MIN_PER_KEY: usize = 16;

/// Places every element of `arr` by its key in `keys`, all within the `k` keys from `min`, into
/// a buffer at the place the counts give it, and copies the buffer back.
fn scatter<T, K: CountingKey, C: Count>(arr: &mut [T], keys: &[K], min: K, k: usize) {
    let segments = if k <= SEGMENTS_MAX_RANGE && arr.len() >= SEGMENTS_MIN_PER_KEY * k {
        SEGMENTS
    } else {
        1
    };
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let next = counters::<C>(segments * k, &mut stack, &mut heap);
    if segments == SEGMENTS {
        place::<T, K, C, SEGMENTS>(arr, keys, min, k, next);
    } else {
        place::<T, K, C, 1>(arr, keys, min, k, next);
    }
}

/// Splits `arr` into `S` segments, the last one taking the remainder, and stably places its
/// elements by their keys in `keys` into a buffer, then copies the buffer back.
///
/// Every segment has its own place for each key, after the places of the same key in the
/// segments before it, so the order of equal keys is kept. The segments are walked side by
/// side: placing an element waits for the place handed out before it with the same key and in
/// the same segment, and with `S` segments `S` such chains run at once.
///
/// `next` holds `S·k` zeroed counters, which become the next place of each key in each segment.
#[inline(always)]
fn place<T, K: CountingKey, C: Count, const S: usize>(
    arr: &mut [T],
    keys: &[K],
    min: K,
    k: usize,
    next: &mut [C],
) {
    let len = arr.len();
    let seg = len / S;
    let tail = S * seg;
    let next = next.as_mut_ptr();
    let keys = keys.as_ptr();
    // Segment `s` is `arr[s·seg..(s + 1)·seg]`, and the last one also takes `arr[S·seg..]`.
    for i in 0..seg {
        for s in 0..S {
            // SAFETY: `s·seg + i < S·seg <= len`. Every key lies between `min` and the largest
            // key, so its offset is less than `k` and the counter is in `next`.
            unsafe { bump(next.add(s * k + (*keys.add(s * seg + i)).offset(min))) };
        }
    }
    for i in tail..len {
        // SAFETY: as above, in the last segment.
        unsafe { bump(next.add((S - 1) * k + (*keys.add(i)).offset(min))) };
    }
    let mut sum = 0;
    for o in 0..k {
        for s in 0..S {
            // SAFETY: `s·k + o < S·k`.
            unsafe {
                let c = next.add(s * k + o);
                let count = (*c).get();
                *c = C::ZERO.add(sum);
                sum += count;
            }
        }
    }
    let mut buf = Buffer::<T>::new(len);
    let dst = buf.get();
    let base = arr.as_ptr();
    for i in 0..seg {
        for s in 0..S {
            // SAFETY: as in the count above. The keys are the ones counted, so the places of
            // each key in each segment are handed out exactly once each, and all lie below
            // `len`.
            unsafe {
                let j = s * seg + i;
                let pos = bump(next.add(s * k + (*keys.add(j)).offset(min)));
                ptr::copy_nonoverlapping(base.add(j), dst.add(pos), 1);
            }
        }
    }
    for i in tail..len {
        // SAFETY: as above, in the last segment.
        unsafe {
            let pos = bump(next.add((S - 1) * k + (*keys.add(i)).offset(min)));
            ptr::copy_nonoverlapping(base.add(i), dst.add(pos), 1);
        }
    }
    // SAFETY: the `len` places hold each element of `arr` exactly once, and `dst` does not
    // overlap `arr`.
    unsafe { ptr::copy_nonoverlapping(dst, arr.as_mut_ptr(), len) };
}
