use core::marker::PhantomData;
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
/// reals spread uniformly over `[0, 1)` into `n` buckets, each holding about one element, in
/// `O(n)` expected time. Floats are sorted as [`f64::total_cmp`] orders them.
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
    if !T::IDENTITY {
        for x in arr.iter_mut() {
            let bits = x.ordered();
            // SAFETY: `T::Bits` has the size and alignment of `T`, and every bit pattern of
            // both is a valid value (`T` is an integer or a float, as `char` and `bool` are
            // `IDENTITY`).
            unsafe { ptr::from_mut(x).cast::<T::Bits>().write(bits) };
        }
    }
    let p = arr.as_mut_ptr();
    let keys = p.cast::<T::Bits>();
    let mut buf = Buffer::<T::Bits>::new(len);
    let none = NonNull::<()>::dangling().as_ptr();
    let v = Area { keys, elems: none };
    let scratch = Area {
        keys: buf.get(),
        elems: none,
    };
    // SAFETY: `v` holds `len` keys, and `scratch` has room for `len` keys, apart from them; the
    // elements are zero-sized.
    unsafe {
        if len <= u32::MAX as usize {
            sort::<T, (), u32, false>(v, scratch, len, 0);
        } else {
            sort::<T, (), usize, false>(v, scratch, len, 0);
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
/// The keys are bucketed as [`bucket`] buckets values, and the elements move with them. `key`
/// is called exactly once per element: the keys are kept in a buffer of their own as they are
/// read, and once the last key is read no code of the caller runs.
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
pub fn bucket_by_key<T, K, F>(arr: &mut [T], mut key: F)
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
    let first = key(&arr[0]).ordered();
    slots[0].write(first);
    let mut prev = first;
    let (mut descent, mut no_descent) = (false, false);
    for (slot, x) in slots[1..].iter_mut().zip(&arr[1..]) {
        let x = key(x).ordered();
        slot.write(x);
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
    let mut scratch_stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut scratch_heap = Vec::new();
    let scratch_keys = key_slots::<K::Bits>(len, &mut scratch_stack, &mut scratch_heap);
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
            sort::<K, T, u32, false>(v, scratch, len, 0);
        } else {
            sort::<K, T, usize, false>(v, scratch, len, 0);
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
/// keys' bits, which bounds the depth for any input.
const FLOAT_LEVELS: u32 = 2;

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
    fn of(self, key: B) -> usize;
}

/// One bucket per key from `min`: for a range of fewer keys than buckets.
#[derive(Clone, Copy)]
struct Direct<B> {
    min: B,
}

impl<B: Bits> Index<B> for Direct<B> {
    #[inline(always)]
    fn of(self, key: B) -> usize {
        key.wrapping_sub(self.min).low64() as usize
    }
}

/// Buckets of equal width over the keys' bits from `min`: `⌊(key − min) · m / 2^64⌋`, with the
/// difference shifted right by `shift` first when it may not fit in 64 bits.
#[derive(Clone, Copy)]
struct Linear<B> {
    min: B,
    shift: u32,
    m: u64,
}

impl<B: Bits> Index<B> for Linear<B> {
    #[inline(always)]
    fn of(self, key: B) -> usize {
        let d = key.wrapping_sub(self.min);
        let d = if B::BITS > 64 { d.shr(self.shift) } else { d };
        ((u128::from(d.low64()) * u128::from(self.m)) >> 64) as usize
    }
}

/// Buckets of equal width over the values of float keys from `lo`: `⌊(value − lo) · scale⌋`,
/// and the largest values in the last bucket.
struct Float<K> {
    lo: f64,
    scale: f64,
    last: usize,
    key: PhantomData<K>,
}

impl<K> Clone for Float<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for Float<K> {}

impl<K: Bucket> Float<K> {
    /// The map of `buckets` buckets from `min` to `max`, if both are finite and the width of
    /// the range is a finite, positive float.
    fn new(min: K::Bits, max: K::Bits, buckets: usize) -> Option<Self> {
        let (lo, hi) = (K::value(min), K::value(max));
        let width = hi - lo;
        let scale = buckets as f64 / width;
        (lo.is_finite() && width.is_finite() && width > 0.0 && scale.is_finite()).then_some(Self {
            lo,
            scale,
            last: buckets - 1,
            key: PhantomData,
        })
    }
}

impl<K: Bucket> Index<K::Bits> for Float<K> {
    #[inline(always)]
    fn of(self, key: K::Bits) -> usize {
        // Every key is finite and at least `lo`, so the product is not negative; rounding keeps
        // the map monotonic, and a cast saturates.
        (((K::value(key) - self.lo) * self.scale) as usize).min(self.last)
    }
}

/// Sorts the `len` keys of `src`, and their elements with them: in place if not `INTO`, with
/// `dst` as scratch space, and into `dst` if `INTO`.
///
/// # Safety
///
/// `src` must hold `len` initialized keys and elements, and `dst` room for as many, apart from
/// them.
unsafe fn sort<K: Bucket, E, C: Count, const INTO: bool>(
    src: Area<K::Bits, E>,
    dst: Area<K::Bits, E>,
    len: usize,
    depth: u32,
) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        if len <= LEAF {
            if INTO {
                insertion_into(src, dst, len);
            } else {
                insertion(src, len);
            }
            return;
        }
        let (min, max) = min_max(slice::from_raw_parts(src.keys, len));
        if min == max {
            if INTO {
                src.move_all(dst, len);
            }
            return;
        }
        let buckets = buckets(len);
        if K::FLOAT
            && depth < FLOAT_LEVELS
            && let Some(map) = Float::<K>::new(min, max, buckets)
        {
            return level::<K, E, C, _, INTO>(src, dst, len, map, buckets, depth);
        }
        let span = max.wrapping_sub(min);
        let width = <K::Bits as Bits>::BITS - span.leading_zeros();
        if width < 64 && span.low64() < buckets as u64 {
            let map = Direct { min };
            level::<K, E, C, _, INTO>(src, dst, len, map, span.low64() as usize + 1, depth);
        } else {
            let shift = width.saturating_sub(64);
            // At least `buckets`, so `m` is less than 2^64.
            let span = span.shr(shift).low64();
            let m = ((buckets as u128) << 64) / (u128::from(span) + 1);
            let map = Linear {
                min,
                shift,
                m: m as u64,
            };
            level::<K, E, C, _, INTO>(src, dst, len, map, buckets, depth);
        }
    }
}

/// One level of [`sort`]: moves the keys and elements of `src` into their buckets in `dst`, then
/// sorts the buckets where `sort` leaves them.
///
/// # Safety
///
/// As for [`sort`]; `map` must map every key of `src` below `buckets`.
#[inline(always)]
unsafe fn level<K: Bucket, E, C: Count, I: Index<K::Bits>, const INTO: bool>(
    src: Area<K::Bits, E>,
    dst: Area<K::Bits, E>,
    len: usize,
    map: I,
    buckets: usize,
    depth: u32,
) {
    let mut stack = Stack([MaybeUninit::uninit(); STACK_BYTES]);
    let mut heap = Vec::new();
    let ends = counters::<C>(buckets, &mut stack, &mut heap);
    // SAFETY: guaranteed by the caller.
    unsafe {
        distribute(src, dst, len, map, ends);
        // Runs of small buckets are sorted by one insertion sort: no key moves out of its
        // bucket, as every key before it is smaller.
        let (mut start, mut flushed) = (0, 0);
        for end in ends.iter() {
            let end = end.get();
            if end - start > LEAF {
                if INTO {
                    insertion(dst.at(flushed), start - flushed);
                    sort::<K, E, C, false>(dst.at(start), src.at(start), end - start, depth + 1);
                } else {
                    insertion_into(dst.at(flushed), src.at(flushed), start - flushed);
                    sort::<K, E, C, true>(dst.at(start), src.at(start), end - start, depth + 1);
                }
                flushed = end;
            }
            start = end;
        }
        if INTO {
            insertion(dst.at(flushed), len - flushed);
        } else {
            insertion_into(dst.at(flushed), src.at(flushed), len - flushed);
        }
    }
}

/// Moves the keys and elements of `src` into their buckets in `dst`, stably, and leaves in
/// `ends` the end of each bucket.
///
/// # Safety
///
/// As for [`level`], with `ends` zeroed and one counter per bucket.
#[inline(always)]
unsafe fn distribute<B: Bits, E, C: Count, I: Index<B>>(
    src: Area<B, E>,
    dst: Area<B, E>,
    len: usize,
    map: I,
    ends: &mut [C],
) {
    let buckets = ends.len();
    let c = ends.as_mut_ptr();
    // SAFETY: guaranteed by the caller: every bucket is below `buckets`, and the places of
    // each bucket are handed out once each, all below `len`. `ends` is only used through `c`.
    unsafe {
        for i in 0..len {
            bump(c.add(map.of(src.key(i))));
        }
        let mut sum = 0;
        for b in 0..buckets {
            let count = (*c.add(b)).get();
            *c.add(b) = C::ZERO.add(sum);
            sum += count;
        }
        for i in 0..len {
            let pos = bump(c.add(map.of(src.key(i))));
            src.move_to(i, dst, pos);
        }
    }
}

/// Sorts the `len` keys of `v`, and their elements with them, by insertion. Stable.
///
/// # Safety
///
/// `v` must hold `len` initialized keys and elements.
#[inline(always)]
unsafe fn insertion<B: Bits, E>(v: Area<B, E>, len: usize) {
    // SAFETY: guaranteed by the caller; the element at `i` is read out, the ones before it move
    // up a place, and it is written back into the gap, with no code in between that can panic.
    unsafe {
        for i in 1..len {
            let k = v.key(i);
            if v.key(i - 1) <= k {
                continue;
            }
            let e = ptr::read(v.elems.add(i));
            let mut j = i;
            loop {
                v.move_to(j - 1, v, j);
                j -= 1;
                if j == 0 || v.key(j - 1) <= k {
                    break;
                }
            }
            v.keys.add(j).write(k);
            v.elems.add(j).write(e);
        }
    }
}

/// Moves the `len` keys of `src`, and their elements with them, into `dst`, sorted by
/// insertion. Stable.
///
/// # Safety
///
/// `src` must hold `len` initialized keys and elements, and `dst` room for as many, apart from
/// them.
#[inline(always)]
unsafe fn insertion_into<B: Bits, E>(src: Area<B, E>, dst: Area<B, E>, len: usize) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        for i in 0..len {
            let k = src.key(i);
            let mut j = i;
            while j > 0 && dst.key(j - 1) > k {
                dst.move_to(j - 1, dst, j);
                j -= 1;
            }
            src.move_to(i, dst, j);
        }
    }
}
