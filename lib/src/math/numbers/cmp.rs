//! Numbers compared by value: `==` and `!=`, `<`, `<=`, `>` and `>=`, `cmp`, `max`, `min` and
//! `clamp`, with each other and with the primitive integers on the right, and a hash that agrees
//! with `==`.
//!
//! Two numbers are equal when they are the same integer, whatever their byte orders and however
//! their magnitudes are stored: zero bytes above the most significant non-zero byte count for
//! nothing, an empty vector is zero, and so is a zero magnitude with `negative` set. The derived
//! comparisons would compare the stored bytes instead: `[1]` big-endian and `[1]` little-endian
//! would differ in their byte orders, and `[1, 0]` big-endian (256) would be below `[2]`.
//!
//! Magnitudes of the same length in the same byte order are compared as byte strings, which the
//! standard library does with `memcmp`: big-endian, the first byte that differs is the most
//! significant one, so lexicographic order is numeric order. Otherwise they are compared limb by
//! limb from the top, each read in its own byte order.

use super::add::{read, read_small, significant_len};
use super::{ByteOrder, Number};
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

/// The significant bytes of a number's magnitude, without the zero bytes above them, in the
/// number's byte order, and its sign, never negative for zero.
#[derive(Clone, Copy)]
pub(super) struct Magnitude<'a> {
    /// The significant bytes: the last ones of the vector big-endian, the first little-endian.
    pub(super) bytes: &'a [u8],
    /// Whether the bytes are big-endian.
    pub(super) big_endian: bool,
    /// Whether the number is below zero.
    pub(super) negative: bool,
}

impl<'a> Magnitude<'a> {
    /// The magnitude and sign of `n`.
    #[inline(always)]
    pub(super) fn of(n: &'a Number) -> Self {
        let v = n.value.as_slice();
        let (bytes, big_endian) = if n.byte_order == ByteOrder::BigEndian {
            (&v[v.len() - significant_len::<true>(v)..], true)
        } else {
            (&v[..significant_len::<false>(v)], false)
        };
        Magnitude {
            bytes,
            big_endian,
            negative: n.negative && !bytes.is_empty(),
        }
    }

    /// The magnitude, which must have at most 16 bytes, as a `u128`.
    #[inline(always)]
    pub(super) fn small(self) -> u128 {
        debug_assert!(self.bytes.len() <= 16);
        let order = if self.big_endian {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        };
        // SAFETY: the magnitude has at most 16 bytes, as the caller guarantees.
        unsafe { read_small(self.bytes, order) }
    }
}

/// Limb `k` of the magnitude `v` in the layout `BE`, the least significant limb being limb 0.
///
/// # Safety
///
/// `v` must hold at least `8 * k + 8` bytes.
#[inline(always)]
unsafe fn limb<const BE: bool>(v: &[u8], k: usize) -> u64 {
    // SAFETY: as the caller guarantees, `[8k, 8k + 8)` lies inside `v`, and so does
    // `[n - 8k - 8, n - 8k)`.
    unsafe {
        if BE {
            u64::from_be_bytes(read(v.as_ptr().add(v.len() - 8 * k - 8)))
        } else {
            u64::from_le_bytes(read(v.as_ptr().add(8 * k)))
        }
    }
}

/// The `v.len() % 8` most significant bytes of the magnitude `v` in the layout `BE`, those above
/// its whole limbs, as a number.
#[inline(always)]
fn top<const BE: bool>(v: &[u8]) -> u64 {
    let r = v.len() % 8;
    let bytes = if BE { &v[..r] } else { &v[v.len() - r..] };
    if BE {
        bytes.iter().fold(0, |t, &b| (t << 8) | u64::from(b))
    } else {
        bytes.iter().rev().fold(0, |t, &b| (t << 8) | u64::from(b))
    }
}

/// Compares the magnitudes `a` and `b`, of the same length, laid out as `ABE` and `BBE` say, from
/// the most significant limb down.
#[inline(always)]
fn cmp_limbs<const ABE: bool, const BBE: bool>(a: &[u8], b: &[u8]) -> Ordering {
    debug_assert_eq!(a.len(), b.len());
    let ordering = top::<ABE>(a).cmp(&top::<BBE>(b));
    if ordering != Ordering::Equal {
        return ordering;
    }
    let mut k = a.len() / 8;
    if ABE == BBE && k > 4 {
        // The same layout: the top limbs, where random magnitudes differ, then the whole
        // magnitudes compared for equality as byte strings, which the standard library does with
        // vector instructions; only magnitudes that differ below their top limbs are scanned
        // further. Measured on an Apple M4 Max, equal magnitudes of 1 KiB took 107 cycles so
        // against 199 limb by limb, and those of 16 KiB 1,233 against 2,801.
        for _ in 0..4 {
            k -= 1;
            // SAFETY: both hold `a.len()` bytes, at least `8k + 8`.
            let (x, y) = unsafe { (limb::<ABE>(a, k), limb::<BBE>(b, k)) };
            if x != y {
                return x.cmp(&y);
            }
        }
        if a == b {
            return Ordering::Equal;
        }
    }
    while k > 0 {
        k -= 1;
        // SAFETY: both hold `a.len()` bytes, at least `8k + 8`.
        let (x, y) = unsafe { (limb::<ABE>(a, k), limb::<BBE>(b, k)) };
        if x != y {
            return x.cmp(&y);
        }
    }
    Ordering::Equal
}

/// Whether the magnitudes `a` and `b`, of the same length, laid out as `ABE` and `BBE` say (not
/// both the same), are equal.
#[inline(always)]
fn eq_limbs<const ABE: bool, const BBE: bool>(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut differ = top::<ABE>(a) ^ top::<BBE>(b);
    for k in 0..a.len() / 8 {
        // SAFETY: both hold `a.len()` bytes, at least `8k + 8`.
        differ |= unsafe { limb::<ABE>(a, k) ^ limb::<BBE>(b, k) };
    }
    differ == 0
}

/// Compares the magnitudes of `a` and `b`, whatever their byte orders.
#[inline(always)]
pub(super) fn cmp_magnitudes(a: Magnitude, b: Magnitude) -> Ordering {
    let (x, y) = (a.bytes, b.bytes);
    match x.len().cmp(&y.len()) {
        Ordering::Equal => {}
        unequal => return unequal,
    }
    match (a.big_endian, b.big_endian) {
        (true, true) => x.cmp(y),
        (false, false) => cmp_limbs::<false, false>(x, y),
        (true, false) => cmp_limbs::<true, false>(x, y),
        (false, true) => cmp_limbs::<false, true>(x, y),
    }
}

/// Compares two signed values, `negative_a` and `negative_b` false for zero, by the order of
/// their magnitudes, which `magnitudes` gives.
#[inline(always)]
fn signed(negative_a: bool, negative_b: bool, magnitudes: impl FnOnce() -> Ordering) -> Ordering {
    match (negative_a, negative_b) {
        (false, true) => Ordering::Greater,
        (true, false) => Ordering::Less,
        (false, false) => magnitudes(),
        (true, true) => magnitudes().reverse(),
    }
}

/// The signs and magnitudes of `a` and `b` when both vectors hold at most 16 bytes, read as
/// `u128`s without looking for their significant bytes; signs are not set for zero.
#[inline(always)]
fn small_pair(a: &Number, b: &Number) -> Option<((bool, u128), (bool, u128))> {
    if a.value.len() > 16 || b.value.len() > 16 {
        return None;
    }
    // SAFETY: both vectors hold at most 16 bytes.
    let (x, y) = unsafe {
        (
            read_small(&a.value, a.byte_order),
            read_small(&b.value, b.byte_order),
        )
    };
    Some(((a.negative && x != 0, x), (b.negative && y != 0, y)))
}

impl PartialEq for Number {
    /// Whether the numbers are the same integer, whatever their byte orders and however their
    /// magnitudes are stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let big = Number::new_from_u32(0x0102_0304, ByteOrder::BigEndian);
    /// let little = Number::new_from_u32(0x0102_0304, ByteOrder::LittleEndian);
    /// assert_eq!(big, little);
    ///
    /// // Zero bytes above the most significant one count for nothing, and zero has no sign.
    /// assert_eq!(Number::new_signed(vec![0], true, ByteOrder::BigEndian), Number::default());
    /// assert_ne!(Number::new_from_i8(-1, ByteOrder::BigEndian), Number::new_from_i8(1, ByteOrder::BigEndian));
    /// ```
    #[inline]
    fn eq(&self, other: &Number) -> bool {
        if let Some((x, y)) = small_pair(self, other) {
            return x == y;
        }
        let (a, b) = (Magnitude::of(self), Magnitude::of(other));
        let (x, y) = (a.bytes, b.bytes);
        if a.negative != b.negative || x.len() != y.len() {
            return false;
        }
        match (a.big_endian, b.big_endian) {
            (true, true) | (false, false) => x == y,
            (true, false) => eq_limbs::<true, false>(x, y),
            (false, true) => eq_limbs::<false, true>(x, y),
        }
    }
}

impl Eq for Number {}

impl PartialOrd for Number {
    /// Compares the numbers by value, as [`Ord::cmp`] does.
    #[inline]
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Number {
    /// Compares the numbers by value: every negative number is below zero, and zero below every
    /// positive one, whatever their byte orders and however their magnitudes are stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut v = vec![
    ///     Number::new_from_i32(300, ByteOrder::BigEndian),
    ///     Number::new_from_i32(-5, ByteOrder::LittleEndian),
    ///     Number::new_from_i32(7, ByteOrder::LittleEndian),
    /// ];
    /// v.sort();
    /// assert_eq!(v, [-5, 7, 300]);
    /// assert!(Number::new_from_i32(-300, ByteOrder::BigEndian) < -5);
    /// ```
    #[inline]
    fn cmp(&self, other: &Number) -> Ordering {
        if let Some(((na, x), (nb, y))) = small_pair(self, other) {
            return signed(na, nb, || x.cmp(&y));
        }
        let (a, b) = (Magnitude::of(self), Magnitude::of(other));
        signed(a.negative, b.negative, || cmp_magnitudes(a, b))
    }
}

impl Hash for Number {
    /// Hashes the number by value, so that numbers that are equal hash alike, whatever their byte
    /// orders and however their magnitudes are stored: the sign, the significant length, then the
    /// significant bytes of the magnitude, the least significant first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    /// use std::collections::HashSet;
    ///
    /// let mut seen = HashSet::new();
    /// seen.insert(Number::new_from_u64(1 << 40, ByteOrder::BigEndian));
    /// assert!(seen.contains(&Number::new(vec![0, 0, 0, 0, 0, 1, 0], ByteOrder::LittleEndian)));
    /// ```
    fn hash<H: Hasher>(&self, state: &mut H) {
        let m = Magnitude::of(self);
        let v = m.bytes;
        state.write_u8(u8::from(m.negative));
        state.write_usize(v.len());
        // The magnitude little-endian, in chunks of `HASH_CHUNK` bytes from the least significant
        // one, so that both byte orders hash the same bytes in the same calls: a big-endian
        // magnitude is turned around a chunk at a time in a buffer.
        if !m.big_endian {
            for chunk in v.chunks(HASH_CHUNK) {
                state.write(chunk);
            }
            return;
        }
        let mut buffer = [0u8; HASH_CHUNK];
        for chunk in v.rchunks(HASH_CHUNK) {
            let out = &mut buffer[..chunk.len()];
            for (o, &b) in out.iter_mut().zip(chunk.iter().rev()) {
                *o = b;
            }
            state.write(out);
        }
    }
}

/// The bytes of a magnitude hashed in one call to the hasher: measured on an Apple M4 Max with
/// the standard library's SipHash, a magnitude of 1 KiB took 835 cycles so against 1,976 as
/// a `u64` a limb.
const HASH_CHUNK: usize = 256;

/// Compares the number `n` with a primitive integer, given by its sign `negative`, never set for
/// zero, and its magnitude `m`.
#[inline(always)]
pub(super) fn cmp_primitive(n: &Number, negative: bool, m: u128) -> Ordering {
    let a = Magnitude::of(n);
    signed(a.negative, negative, || {
        if a.bytes.len() > 16 {
            Ordering::Greater
        } else {
            a.small().cmp(&m)
        }
    })
}

/// Comparisons of numbers with a primitive integer type, the number on the left.
///
/// Not the other way round: `impl PartialEq<Number> for i32` would leave `i32` two types to be
/// compared with, and code that leaves the other type to inference, such as `v == vec![]` for a
/// `Vec<i32>`, would no longer compile wherever this crate is linked.
macro_rules! compare_with {
    ($($t:ident),* $(,)?) => {$(
        impl PartialEq<$t> for Number {
            /// Whether the number is the integer `other`.
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                let (negative, m) = parts!(*other, $t);
                cmp_primitive(self, negative, m) == Ordering::Equal
            }
        }

        impl PartialOrd<$t> for Number {
            /// Compares the number with the integer `other` by value.
            #[inline]
            fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                let (negative, m) = parts!(*other, $t);
                Some(cmp_primitive(self, negative, m))
            }
        }
    )*};
}

/// The sign, never set for zero, and the magnitude of a value of a primitive integer type.
macro_rules! parts {
    ($v:expr, u8) => {
        (false, u128::from($v))
    };
    ($v:expr, u16) => {
        (false, u128::from($v))
    };
    ($v:expr, u32) => {
        (false, u128::from($v))
    };
    ($v:expr, u64) => {
        (false, u128::from($v))
    };
    ($v:expr, u128) => {
        (false, $v)
    };
    ($v:expr, usize) => {
        (false, $v as u128)
    };
    ($v:expr, $t:ident) => {{
        let v = $v;
        (v < 0, v.unsigned_abs() as u128)
    }};
}
pub(super) use parts;

compare_with!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
