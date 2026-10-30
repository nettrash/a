//! Bit shifts of [`Number`]s: `<<`, `>>`, `<<=` and `>>=` by a `usize` number of bits.
//!
//! The shifts follow the rules of Rust's primitive signed integers, with no bits lost at the
//! top: `a << k` is `a · 2^k`, and `a >> k` is `a / 2^k` rounded towards negative infinity,
//! the arithmetic shift of a two's-complement integer, so `-7 >> 1` is `-4` and `-1 >> k` is
//! `-1` for every `k`. On a sign and a magnitude, a shift moves the bits of the magnitude and
//! keeps the sign, except that a negative number whose shifted-out bits are not all zero has
//! one added to the magnitude of its quotient. That can carry: `-0x1FF >> 1` is `-0x100`, one
//! byte longer than `0x1FF >> 1`.
//!
//! A shift by `k = 8q + r` bits moves the bytes of the magnitude by `q` positions and their
//! bits by `r`. Everything about the result but its bytes is planned first: the significant
//! length `l` of the magnitude, the length of the result (`l + q` bytes for `<<`, or one more
//! when the bits shifted out of the top byte are not zero; `l - q` for `>>`, or one fewer when
//! nothing is left of the top byte), and, for a negative `>>`, whether one is added. Only then
//! is a vector chosen for the result, as the other operators choose one: the vector of the
//! operand taken by value, or of the target of `<<=` or `>>=`, when its capacity has room for
//! the result, and otherwise one new vector of exactly the result's length. A right shift is
//! never longer than its operand, so `>>` and `>>=` on an owned number allocate nothing (but
//! for zero in a vector without capacity, which needs one byte).
//!
//! Operands of at most 16 bytes, shifted left by fewer than 128 bits or right by any amount,
//! take a short path in registers: the magnitude is read into a `u128` as the addition reads
//! it, shifted (left, as two `u64`s into four, by a branch on the amount below 64 that is the
//! same for every call with that amount), and the result, at most 32 bytes, written in the
//! operand's byte order. `<<=` of 17 to 32 bytes by fewer than 64 bits does the same with four
//! limbs. The longer ones are handled out of line, in one pass over the bytes, in place in
//! either byte order; in place, every call the out-of-line functions make is their last step,
//! with its arguments in registers, so that they keep no stack frame:
//!
//! - A shift by whole bytes moves them, inline up to 64 bytes (all of them read before any is
//!   written) and with `memmove` beyond. Big-endian in place it moves nothing: `>>` drops the
//!   bytes at the end of the vector, and `<<` writes zeros after them.
//! - A shift with bits is done by kernels chosen at compile time: on 64-bit little-endian
//!   `aarch64` with NEON, sixteen bytes at a time in byte lanes, each byte of the result made
//!   from the byte at its position and its neighbour, in the order of memory, so that one
//!   kernel serves both byte orders with no byte reversal, each block loaded once and stored on
//!   16 bytes from 64 bytes up; elsewhere and under Miri, portable ones on `u64`s that carry the
//!   bits shifted out of one group of eight bytes into the next.
//!   In place they keep every byte read before a write reaches it: `<<` little-endian runs
//!   from the most significant byte down, as the bytes move up; the others run upwards.
//! - Zeros below the result of `<<` are written inline up to 64 bytes, with `memset` beyond.
//! - A negative quotient rounded down has one added to its eight least significant bytes,
//!   which are computed from the input before the kernel writes over it, so that they are not
//!   read back from stores still on their way; the bytes above are carried into only when
//!   those eight are all 0xFF.
//!
//! Some facts about the result are coin tosses for random numbers: whether the bits shifted out
//! of the top byte by `<<` are zero (the result is one byte longer or not), whether the top
//! byte is left with anything by `>>` (one byte shorter or not), and whether a bit shifted out
//! of a negative number is set (it is rounded or not). None of them is decided by a branch: the
//! byte above the top is written where the result's top byte is written anyway when it is not
//! needed, the kernels count their blocks from the input's length rather than the result's,
//! and the one of the rounding is an addition of zero or one.

// The kernels that shift the bits of long magnitudes, under the name `kernels`: NEON on the
// 64-bit little-endian aarch64 targets that the addition's assembly runs on (the ones whose
// operating system allows unaligned accesses), where NEON is enabled, as it is by default; the
// portable ones elsewhere and under Miri, which runs no SIMD of this kind.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    target_feature = "neon",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
))]
#[path = "shift/aarch64.rs"]
mod kernels;
#[cfg(not(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    target_feature = "neon",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
)))]
#[path = "shift/portable.rs"]
mod kernels;

use super::add::{SMALL, read, read_small, significant_len, write, write_small};
use super::{ByteOrder, Number};
use core::hint::select_unpredictable;
use core::ops::{Shl, ShlAssign, Shr, ShrAssign};
use core::ptr;
use std::alloc::{Layout, alloc};

impl Shl<usize> for Number {
    type Output = Number;

    /// Returns `self · 2^rhs`, in the byte order of `self`, in the shortest form, as
    /// `self <<= rhs` does.
    ///
    /// The result is built in the vector of `self` when its capacity has room for it, which
    /// allocates nothing, and otherwise in one new vector of exactly its length.
    ///
    /// # Panics
    ///
    /// When the result is too long to be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut value = Vec::with_capacity(4);
    /// value.extend([0x81, 0x02]);
    /// let a = Number::new(value, ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    ///
    /// // 0x81_02 << 9 = 0x01_02_04_00, in the vector of `a`.
    /// let shifted = a << 9;
    /// assert_eq!(shifted.value, [0x01, 0x02, 0x04, 0x00]);
    /// assert_eq!(shifted.value.as_ptr(), pointer);
    /// ```
    #[inline]
    fn shl(mut self, rhs: usize) -> Number {
        self <<= rhs;
        self
    }
}

impl Shl<usize> for &Number {
    type Output = Number;

    /// Returns `self · 2^rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The result is built in one new vector of exactly its length (one byte for zero), the
    /// only allocation; `self` is left as it was. It may have any sign and length, and need not
    /// be in the shortest form.
    ///
    /// # Panics
    ///
    /// When the result is too long to be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -3 << 8 = -0x03_00.
    /// let a = Number::new_from_i32(-3, ByteOrder::LittleEndian);
    /// let shifted = &a << 8;
    /// assert_eq!(shifted.value, [0x00, 0x03]);
    /// assert!(shifted.is_negative());
    ///
    /// // Zero stays zero, however far it is shifted.
    /// let zero = Number::new_from_u8(0, ByteOrder::BigEndian);
    /// assert_eq!((&zero << usize::MAX).value, [0]);
    /// ```
    #[inline]
    fn shl(self, rhs: usize) -> Number {
        if short_shl(self, rhs) {
            // SAFETY: the vector holds at most 16 bytes, and the shift is below 128 bits.
            let s = unsafe { small_shl(self, rhs) };
            return small_number(self.byte_order, s);
        }
        shl_ref(self, rhs)
    }
}

impl ShlAssign<usize> for Number {
    /// Multiplies `self` by `2^rhs`, keeping its byte order, in the shortest form.
    ///
    /// The result is built in the vector of `self` when its capacity has room for it, which
    /// allocates nothing, and otherwise in one new vector of exactly its length, which replaces
    /// it.
    ///
    /// # Panics
    ///
    /// When the result is too long to be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u8(0xFF, ByteOrder::LittleEndian);
    /// n <<= 4;
    /// assert_eq!(n.value, [0xF0, 0x0F]);
    /// n <<= 16;
    /// assert_eq!(n.value, [0x00, 0x00, 0xF0, 0x0F]);
    /// ```
    #[inline]
    fn shl_assign(&mut self, rhs: usize) {
        if short_shl(self, rhs) {
            // SAFETY: the vector holds at most 16 bytes, and the shift is below 128 bits.
            let s = unsafe { small_shl(self, rhs) };
            self.negative = s.negative;
            put_small(&mut self.value, self.byte_order, s);
            return;
        }
        shl_assign_long(self, rhs);
    }
}

impl Shr<usize> for Number {
    type Output = Number;

    /// Returns `self / 2^rhs` rounded towards negative infinity, in the byte order of `self`,
    /// in the shortest form, as `self >>= rhs` does.
    ///
    /// The result is built in the vector of `self`, which always has room for it unless it is
    /// an empty vector without capacity; nothing is allocated then.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u32(0x01_02_03, ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    ///
    /// // 0x01_02_03 >> 9 = 0x81, in the vector of `a`.
    /// let shifted = a >> 9;
    /// assert_eq!(shifted.value, [0x81]);
    /// assert_eq!(shifted.value.as_ptr(), pointer);
    /// ```
    #[inline]
    fn shr(mut self, rhs: usize) -> Number {
        self >>= rhs;
        self
    }
}

impl Shr<usize> for &Number {
    type Output = Number;

    /// Returns `self / 2^rhs` rounded towards negative infinity, in the byte order of `self`,
    /// in the shortest form.
    ///
    /// The result is built in one new vector of exactly its length (one byte for zero), the
    /// only allocation; `self` is left as it was. It may have any sign and length, and need not
    /// be in the shortest form.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // As for i32: -7 >> 1 = -4, 7 >> 1000 = 0 and -7 >> 1000 = -1.
    /// let a = Number::new_from_i32(-7, ByteOrder::BigEndian);
    /// let shifted = &a >> 1;
    /// assert_eq!(shifted.value, [4]);
    /// assert!(shifted.is_negative());
    /// let shifted = &a >> 1000;
    /// assert_eq!(shifted.value, [1]);
    /// assert!(shifted.is_negative());
    /// let b = Number::new_from_i32(7, ByteOrder::BigEndian);
    /// assert_eq!((&b >> 1000).value, [0]);
    /// ```
    #[inline]
    fn shr(self, rhs: usize) -> Number {
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let s = unsafe { small_shr(self, rhs) };
            return small_number(self.byte_order, s);
        }
        shr_ref(self, rhs)
    }
}

impl ShrAssign<usize> for Number {
    /// Divides `self` by `2^rhs`, rounding towards negative infinity, keeping its byte order, in
    /// the shortest form.
    ///
    /// The result is built in the vector of `self`, which always has room for it unless it is
    /// an empty vector without capacity; nothing is allocated then.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -0x01_FF >> 1 = -0x01_00: the bit shifted out rounds the quotient down, which carries.
    /// let mut n = Number::new_from_i32(-0x01FF, ByteOrder::LittleEndian);
    /// n >>= 1;
    /// assert_eq!(n.value, [0x00, 0x01]);
    /// assert!(n.is_negative());
    /// n >>= 9;
    /// assert_eq!(n.value, [1]);
    /// assert!(n.is_negative());
    /// ```
    #[inline]
    fn shr_assign(&mut self, rhs: usize) {
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let s = unsafe { small_shr(self, rhs) };
            self.negative = s.negative;
            put_small(&mut self.value, self.byte_order, s);
            return;
        }
        shr_assign_long(self, rhs);
    }
}

// Small numbers

/// The longest left shift, in bits, that the short path takes: a magnitude of 16 bytes shifted
/// by less fits in 32.
const SMALL_SHIFT: usize = 128;

/// Whether `a << k` takes the short path: the vector holds at most [`SMALL`] bytes and `k` is
/// below [`SMALL_SHIFT`].
#[inline(always)]
fn short_shl(a: &Number, k: usize) -> bool {
    a.value.len() <= SMALL && k < SMALL_SHIFT
}

/// A result of the short path, as it computes it in registers.
#[derive(Clone, Copy)]
struct Small {
    /// The 16 least significant bytes of the magnitude.
    lo: u128,
    /// The 16 bytes above them.
    hi: u128,
    /// The length of the result in the shortest form, from 1 to 32: one byte for zero.
    len: usize,
    /// Whether the result is negative; never for zero.
    negative: bool,
}

/// `a << k` for a vector of at most 16 bytes and `k` below 128: the magnitude as 32 bytes.
///
/// The halves are shifted as `u64`s, by `k` or `k - 64` bits as `k` is below 64 or not, which
/// is the same for every call with the same amount: two shifts and an `or` each, where a `u128`
/// shift by any amount is a select on the amount besides.
///
/// # Safety
///
/// The vector of `a` must hold at most [`SMALL`] bytes, and `k` must be below 128.
#[inline(always)]
unsafe fn small_shl(a: &Number, k: usize) -> Small {
    // SAFETY: as the caller guarantees.
    let x = unsafe { read_small(&a.value, a.byte_order) };
    let (x0, x1) = (x as u64, (x >> 64) as u64);
    // `y >> (64 - s)` for `s` below 64, in two steps, so that `s = 0` moves every bit out.
    let up = |y: u64, s: usize| (y >> 1) >> (63 - s);
    let (l0, l1, h0, h1) = if k < 64 {
        (x0 << k, x1 << k | up(x0, k), up(x1, k), 0)
    } else {
        let k = k - 64;
        (0, x0 << k, x1 << k | up(x0, k), up(x1, k))
    };
    let bits = 128 - x.leading_zeros() as usize;
    Small {
        lo: u128::from(l0) | u128::from(l1) << 64,
        hi: u128::from(h0) | u128::from(h1) << 64,
        len: if x == 0 { 1 } else { (bits + k).div_ceil(8) },
        negative: a.negative && x != 0,
    }
}

/// `a >> k` for a vector of at most 16 bytes: the quotient, rounded down, in 16 bytes.
///
/// # Safety
///
/// The vector of `a` must hold at most [`SMALL`] bytes.
#[inline(always)]
unsafe fn small_shr(a: &Number, k: usize) -> Small {
    // SAFETY: as the caller guarantees.
    let x = unsafe { read_small(&a.value, a.byte_order) };
    let negative = a.negative && x != 0;
    // The quotient and the bits shifted out; every bit for a shift by 128 or more.
    let (quotient, rest) = if k < 128 {
        (x >> k, x & ((1 << k) - 1))
    } else {
        (0, x)
    };
    // A negative quotient is rounded down, away from zero, when a bit that is not zero is
    // shifted out. It does not overflow: the quotient is below 2^127 when a bit is shifted out.
    let magnitude = quotient + u128::from(negative && rest != 0);
    Small {
        lo: magnitude,
        hi: 0,
        len: SMALL - (magnitude | 1).leading_zeros() as usize / 8,
        negative,
    }
}

/// Writes the small result `s` over the first `s.len` bytes at `dst`, in the layout of `order`:
/// up to 16 bytes as [`write_small`] writes a sum; from 17, the 16 least significant bytes whole
/// and the ones above them, from 1 to 16, again as [`write_small`] writes them.
///
/// # Safety
///
/// `dst` must be valid for writing `s.len` bytes.
#[inline(always)]
unsafe fn write_small_result(dst: *mut u8, order: ByteOrder, s: Small) {
    let len = s.len;
    // SAFETY: every write lies inside the `len` bytes at `dst`: what `write_small` writes for
    // `len <= 16`, and otherwise `[0, 16)` and the `len - 16` bytes after them, or big-endian
    // the `len - 16` bytes before the last 16.
    unsafe {
        if len <= SMALL {
            write_small(dst, order, s.lo as u64, (s.lo >> 64) as u64, len);
            return;
        }
        let (h0, h1) = (s.hi as u64, (s.hi >> 64) as u64);
        if order == ByteOrder::BigEndian {
            write(dst.add(len - 16), s.lo.to_be_bytes());
            write_small(dst, order, h0, h1, len - SMALL);
        } else {
            write(dst, s.lo.to_le_bytes());
            write_small(dst.add(SMALL), order, h0, h1, len - SMALL);
        }
    }
}

/// Writes the small result `s` as the whole value of `v`, in its vector when it has room, in a
/// new one otherwise.
#[inline(always)]
fn put_small(v: &mut Vec<u8>, order: ByteOrder, s: Small) {
    if v.capacity() >= s.len {
        // SAFETY: the vector has room for the `s.len` bytes, all of which are written before the
        // length is set.
        unsafe {
            write_small_result(v.as_mut_ptr(), order, s);
            v.set_len(s.len);
        }
    } else {
        replace_small(v, order, s.lo, s.hi, s.len);
    }
}

/// Replaces `v` with one new vector of exactly the small result's length, holding it, the result
/// given by its fields. Out of line and cold, and called last, with its arguments in registers,
/// so that the short path keeps no stack frame for it.
#[cold]
#[inline(never)]
fn replace_small(v: &mut Vec<u8>, order: ByteOrder, lo: u128, hi: u128, len: usize) {
    // The sign is the caller's to set; the bytes do not depend on it.
    let s = Small {
        lo,
        hi,
        len,
        negative: false,
    };
    *v = new_small(order, s);
}

/// The small result `s` in one new vector of exactly its length.
#[inline(always)]
fn new_small(order: ByteOrder, s: Small) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len);
    // SAFETY: the new vector has room for the `s.len` bytes, all of which are written before the
    // length is set.
    unsafe {
        write_small_result(v.as_mut_ptr(), order, s);
        v.set_len(s.len);
    }
    v
}

/// The small result `s` as a number in one new vector, in the byte order `order`. Out of line,
/// so that the short path of a borrowed operand saves no registers for the allocation.
#[inline(never)]
fn small_number(order: ByteOrder, s: Small) -> Number {
    Number {
        value: new_small(order, s),
        negative: s.negative,
        byte_order: order,
    }
}

// The general path: out of line, so that the short path saves no registers for it.

/// `&a << k` for the operands the short path leaves.
#[inline(never)]
fn shl_ref(a: &Number, k: usize) -> Number {
    let value = match a.byte_order {
        ByteOrder::BigEndian => shl_new::<true>(&a.value, k),
        ByteOrder::LittleEndian => shl_new::<false>(&a.value, k),
    };
    let zero = matches!(value.as_slice(), [0]);
    Number {
        value,
        negative: a.negative && !zero,
        byte_order: a.byte_order,
    }
}

/// `a <<= k` for the operands the short path leaves.
#[inline(never)]
fn shl_assign_long(a: &mut Number, k: usize) {
    match a.byte_order {
        ByteOrder::BigEndian => shl_in::<true>(a, k),
        ByteOrder::LittleEndian => shl_in::<false>(a, k),
    }
}

/// `&a >> k` for the operands the short path leaves.
#[inline(never)]
fn shr_ref(a: &Number, k: usize) -> Number {
    if k / 8 >= a.value.len() {
        let negative = shifted_away(a);
        return Number {
            value: vec![u8::from(negative)],
            negative,
            byte_order: a.byte_order,
        };
    }
    let (value, negative) = match a.byte_order {
        ByteOrder::BigEndian => shr_new::<true>(&a.value, a.negative, k),
        ByteOrder::LittleEndian => shr_new::<false>(&a.value, a.negative, k),
    };
    Number {
        value,
        negative,
        byte_order: a.byte_order,
    }
}

/// `a >>= k` for the operands the short path leaves.
#[inline(never)]
fn shr_assign_long(a: &mut Number, k: usize) {
    if k / 8 >= a.value.len() {
        a.negative = shifted_away(a);
        // SAFETY: the vector holds more than 16 bytes, and has room for the one, written before
        // the length is set.
        unsafe {
            *a.value.as_mut_ptr() = u8::from(a.negative);
            a.value.set_len(1);
        }
        return;
    }
    a.negative = match a.byte_order {
        ByteOrder::BigEndian => shr_in::<true>(&mut a.value, a.negative, k),
        ByteOrder::LittleEndian => shr_in::<false>(&mut a.value, a.negative, k),
    };
}

/// Whether `a >> k` is -1 when the shift moves out every byte of the vector of `a`, and not 0:
/// whether `a` is negative and not zero. Its first byte decides for a number in the shortest
/// form, in either byte order.
#[inline(always)]
fn shifted_away(a: &Number) -> bool {
    a.negative && a.value.iter().any(|&b| b != 0)
}

/// Makes `v` zero, `[0]`: in its vector when it has room for the byte, and otherwise in one new
/// vector of one byte. Out of line and cold.
#[cold]
#[inline(never)]
fn set_zero(v: &mut Vec<u8>) {
    if v.capacity() == 0 {
        *v = vec![0];
    } else {
        v.clear();
        v.push(0);
    }
}

// Left shifts

/// The magnitude `x << k`, planned: `x` has `l` significant bytes, not zero, and the result
/// `len` bytes, its bytes at positions `q..len` and zeros below them.
#[derive(Clone, Copy)]
struct Left {
    /// The significant length of `x`.
    l: usize,
    /// The whole bytes of the shift.
    q: usize,
    /// The bits of the shift besides them, below 8.
    r: u32,
    /// The length of the result: `l + q`, and one more when the top `r` bits of the top byte
    /// are not all zero.
    len: usize,
}

impl Left {
    /// Plans the shift by `k` of the magnitude of `l` significant bytes, not zero, whose most
    /// significant byte is `top`.
    ///
    /// The length cannot overflow: `l` is at most `isize::MAX`, a vector's length, and `q` at
    /// most `usize::MAX / 8`, so `l + q + 1` stays below `usize::MAX`. It may exceed what can be
    /// allocated, which the allocation reports.
    #[inline(always)]
    fn plan(l: usize, top: u8, k: usize) -> Left {
        let (q, r) = (k / 8, (k % 8) as u32);
        let extra = (u32::from(top) << r) >> 8 != 0;
        Left {
            l,
            q,
            r,
            len: l + q + usize::from(extra),
        }
    }
}

/// The significant length of the magnitude in `v` in the layout `BE` and its most significant
/// byte, which is zero only for zero: one read when the vector's most significant byte is not
/// zero, as in the shortest form, and otherwise as [`significant_len`] finds it.
#[inline(always)]
fn top_of<const BE: bool>(v: &[u8]) -> (usize, u8) {
    let most = if BE { v.first() } else { v.last() };
    match most {
        Some(&top) if top != 0 => (v.len(), top),
        _ => {
            let l = significant_len::<BE>(v);
            // SAFETY: position `l - 1` is a significant byte when `l > 0`.
            let top = if l == 0 {
                0
            } else {
                unsafe { get::<BE>(end::<BE>(v), l - 1) }
            };
            (l, top)
        }
    }
}

/// `x << k` for the magnitude `x` in the bytes of `v` in the layout `BE`, in one new vector of
/// exactly its length.
#[inline(always)]
fn shl_new<const BE: bool>(v: &[u8], k: usize) -> Vec<u8> {
    let (l, top) = top_of::<BE>(v);
    if l == 0 {
        return vec![0];
    }
    // SAFETY: `end` is the least significant end of `v`, of `l` significant bytes.
    unsafe { shl_from::<BE>(end::<BE>(v), Left::plan(l, top, k)) }
}

/// `x << k`, planned as `plan`, in one new vector of exactly its length, for the magnitude `x`
/// at the least significant end `src` in the layout `BE`.
///
/// # Safety
///
/// The `plan.l` significant bytes at `src` must be valid for reads.
#[inline(never)]
unsafe fn shl_from<const BE: bool>(src: *const u8, plan: Left) -> Vec<u8> {
    let mut out = allocate(plan.len);
    let dst = out.as_mut_ptr();
    // SAFETY: the new vector has room for the result, whose bytes are all written before the
    // length is set, apart from `x`.
    unsafe {
        shl_into::<BE>(if BE { dst.add(plan.len) } else { dst }, src, plan);
        out.set_len(plan.len);
    }
    out
}

/// `a <<= k` for the magnitude of `a` in the layout `BE`, in the vector of `a` when its capacity
/// has room for the result, and otherwise in one new vector of exactly its length, which replaces
/// it.
///
/// Every call it makes is its last step, with its arguments in registers, so that a shift of bits
/// in place keeps no stack frame and saves no registers: zero, a new vector, and a shift of whole
/// bytes out of line, and the zeros below a long result by `memset`. The vector's length is set
/// first, as nothing reads it before the bytes are written.
#[inline(always)]
fn shl_in<const BE: bool>(a: &mut Number, k: usize) {
    if a.value.len() <= MEDIUM && k < 64 {
        // SAFETY: the vector holds 17 to 32 bytes, as the short path leaves the shorter ones.
        return unsafe { medium_shl::<BE>(a, k) };
    }
    let v = &mut a.value;
    let (l, top) = top_of::<BE>(v);
    if l == 0 {
        a.negative = false;
        return set_zero(v);
    }
    let (len, capacity) = (v.len(), v.capacity());
    let plan = Left::plan(l, top, k);
    if capacity < plan.len {
        return shl_grow::<BE>(v, plan.l, plan.q, plan.r, plan.len);
    }
    let p = v.as_mut_ptr();
    // SAFETY: the `len` bytes at `p` are the vector's, of `l` significant bytes, and its capacity
    // has room for the result, whose bytes are all written before the vector is read again.
    // Big-endian, the input ends at `p + len` and the result at `p + plan.len`, as their least
    // significant bytes come last, and the result ends at most one byte further: `plan.len` is
    // `l + q` or one more, and its bytes from position `q` up are `l` or one more.
    unsafe {
        let src: *const u8 = if BE { p.add(len) } else { p };
        let dst = if BE { p.add(plan.len) } else { p };
        v.set_len(plan.len);
        if plan.r == 0 {
            return shl_bytes::<BE>(dst, src, plan.l, plan.q);
        }
        shl_bits_into::<BE>(dst, src, plan);
    }
}

/// The longest vector, in bytes, whose left shift by fewer than 64 bits [`shl_in`] computes in
/// registers.
const MEDIUM: usize = 32;

/// `a <<= k` for a vector of 17 to 32 bytes and `k` below 64, in registers: the magnitude read
/// as four limbs, as the short path reads two, shifted into five, and the result, at most 40
/// bytes, written as sixteen, sixteen and the rest, or fewer. Measured on an Apple M4 Max, `<<=`
/// by 13 bits of 17 to 32 bytes took 10 to 13 cycles so, against 16 to 17 by the kernels, whose
/// cost for one or two blocks is in their edges.
///
/// # Safety
///
/// The vector of `a` must hold 17 to 32 bytes, and `k` must be below 64.
#[inline(always)]
unsafe fn medium_shl<const BE: bool>(a: &mut Number, k: usize) {
    let v = &mut a.value;
    let (p, n) = (v.as_ptr(), v.len());
    // SAFETY: the 16 bytes at the least significant end and the `n - 16` from 1 to 16 above them
    // lie in the vector.
    let (lo, hi) = unsafe {
        let above = core::slice::from_raw_parts(if BE { p } else { p.add(16) }, n - 16);
        if BE {
            (
                u128::from_be_bytes(read(p.add(n - 16))),
                read_small(above, ByteOrder::BigEndian),
            )
        } else {
            (
                u128::from_le_bytes(read(p)),
                read_small(above, ByteOrder::LittleEndian),
            )
        }
    };
    let bits = if hi != 0 {
        256 - hi.leading_zeros()
    } else {
        128 - lo.leading_zeros()
    } as usize;
    if bits == 0 {
        a.negative = false;
        return set_zero(v);
    }
    let len = (bits + k).div_ceil(8);
    let (x0, x1, x2, x3) = (lo as u64, (lo >> 64) as u64, hi as u64, (hi >> 64) as u64);
    // `y >> (64 - k)` in two steps, so that `k = 0` moves every bit out.
    let up = |y: u64| (y >> 1) >> (63 - k);
    let y = [
        x0 << k,
        x1 << k | up(x0),
        x2 << k | up(x1),
        x3 << k | up(x2),
        up(x3),
    ];
    if v.capacity() < len {
        return medium_grow::<BE>(v, y[0], y[1], y[2], y[3], y[4], len);
    }
    // SAFETY: the vector has room for the `len` bytes, all of which are written before it is
    // read again.
    unsafe {
        v.set_len(len);
        write_medium::<BE>(v.as_mut_ptr(), y, len);
    }
}

/// [`medium_shl`] for a vector whose capacity has no room for the result, the limbs `y0` to `y4` of
/// `len` bytes: one new vector of exactly that length replaces it. Out of line and cold, and its
/// arguments in registers, so that the caller keeps no stack frame for it.
#[cold]
#[inline(never)]
fn medium_grow<const BE: bool>(
    v: &mut Vec<u8>,
    y0: u64,
    y1: u64,
    y2: u64,
    y3: u64,
    y4: u64,
    len: usize,
) {
    let y = [y0, y1, y2, y3, y4];
    let mut out = allocate(len);
    // SAFETY: the new vector has room for the `len` bytes, all of which are written before the
    // length is set.
    unsafe {
        write_medium::<BE>(out.as_mut_ptr(), y, len);
        out.set_len(len);
    }
    *v = out;
}

/// Writes the `len` bytes of the limbs `y`, least significant first, at `dst` in the layout `BE`:
/// up to 16 as the addition writes a short sum, and otherwise the low 16 whole and the bytes
/// above them, up to 16 again as a short sum or 16 whole and up to 8 more.
///
/// # Safety
///
/// `len` must be from 1 to 40 and hold the significant bytes of `y`, and `dst` be valid for
/// writing `len` bytes.
#[inline(always)]
unsafe fn write_medium<const BE: bool>(dst: *mut u8, y: [u64; 5], len: usize) {
    let order = order::<BE>();
    let pair = |l: u64, h: u64| u128::from(l) | u128::from(h) << 64;
    // SAFETY: as the caller guarantees: positions `0..16`, `16..32` and the rest, each written
    // in its place in the layout.
    unsafe {
        if len <= SMALL {
            return write_small(dst, order, y[0], y[1], len);
        }
        let low = pair(y[0], y[1]);
        if BE {
            write(dst.add(len - 16), low.to_be_bytes());
        } else {
            write(dst, low.to_le_bytes());
        }
        let rest = len - SMALL;
        if rest <= SMALL {
            let at = if BE { dst } else { dst.add(SMALL) };
            return write_small(at, order, y[2], y[3], rest);
        }
        let middle = pair(y[2], y[3]);
        if BE {
            write(dst.add(len - 32), middle.to_be_bytes());
            write_small(dst, order, y[4], 0, len - 32);
        } else {
            write(dst.add(SMALL), middle.to_le_bytes());
            write_small(dst.add(32), order, y[4], 0, len - 32);
        }
    }
}

/// [`shl_in`] for a vector whose capacity has no room for the result: one new vector of exactly
/// its length, planned as the fields of a [`Left`], replaces it. Out of line and cold.
#[cold]
#[inline(never)]
fn shl_grow<const BE: bool>(v: &mut Vec<u8>, l: usize, q: usize, r: u32, len: usize) {
    let src: *const u8 = end::<BE>(v);
    // SAFETY: `src` is the least significant end of `v`, of `l` significant bytes.
    *v = unsafe { shl_from::<BE>(src, Left { l, q, r, len }) };
}

/// [`shl_into`] for a shift of whole bytes, `x << 8q` of `l` bytes: out of line, as its moves
/// and zeros of more than 64 bytes call `memmove` and `memset`.
///
/// # Safety
///
/// As for [`shl_into`].
#[inline(never)]
unsafe fn shl_bytes<const BE: bool>(dst: *mut u8, src: *const u8, l: usize, q: usize) {
    // SAFETY: as the caller guarantees.
    unsafe {
        shl_into::<BE>(
            dst,
            src,
            Left {
                l,
                q,
                r: 0,
                len: l + q,
            },
        )
    }
}

/// Writes `x << k`, planned as `plan`, at the least significant end `dst` in the layout `BE`,
/// from the magnitude `x` at the least significant end `src` in the same layout, which may be
/// in the same vector.
///
/// The bytes are written from the most significant down, each group of eight read before a
/// write reaches it: the result's bytes lie at or above the positions of the bytes they come
/// from, little-endian at higher addresses; big-endian the result ends at most one byte further
/// than the input, and its groups of eight are written over bytes of the input that have been
/// read.
///
/// # Safety
///
/// The `plan.l` significant bytes at `src` must be valid for reads and the `plan.len` bytes at
/// `dst` for writes. If they overlap, `dst` must be `src` little-endian, and big-endian at most
/// `src + 1`.
#[inline(always)]
unsafe fn shl_into<const BE: bool>(dst: *mut u8, src: *const u8, plan: Left) {
    let Left { l, q, r, .. } = plan;
    // SAFETY: every position read is below `l` and every position written below `len`; the
    // order of the accesses is as described above.
    unsafe {
        if r == 0 {
            // Whole bytes: a move of the magnitude, then the zeros below it. Big-endian in place
            // the magnitude stays where it is, unless it has zero bytes above it.
            let (from, to) = (at::<BE>(src, 0, l), at_mut::<BE>(dst, q, l));
            if !ptr::eq(from, to) {
                move_bytes(to, from, l);
            }
            zero::<BE>(dst, q);
        } else {
            shl_bits_into::<BE>(dst, src, plan);
        }
    }
}

/// [`shl_into`] for a shift with bits, `r` from 1 to 7: the zeros below the result last.
///
/// # Safety
///
/// As for [`shl_into`].
#[inline(always)]
unsafe fn shl_bits_into<const BE: bool>(dst: *mut u8, src: *const u8, plan: Left) {
    let Left { l, q, r, len } = plan;
    // SAFETY: as for `shl_into`.
    unsafe {
        if l < SMALL {
            // Up to 15 bytes, shifted in a register, which they fill with the bits out of the
            // top. They are all read before any is written.
            let x = read_small(slice::<BE>(src, 0, l), order::<BE>()) << r;
            let n = len - q;
            write_small(
                at_mut::<BE>(dst, q, n),
                order::<BE>(),
                x as u64,
                (x >> 64) as u64,
                n,
            );
        } else {
            // The result's bytes at positions `q + l - 1` and `q + l`: the top byte shifted, with
            // the bits of the byte below it, and the bits shifted out of the top. Read before
            // the kernel writes over them in place.
            let (top, below) = (get::<BE>(src, l - 1), get::<BE>(src, l - 2));
            let high = (top << r) | (below >> (8 - r));
            let out = top >> (8 - r);
            kernels::shl_bits::<BE>(dst, src, l, q, r);
            // The bits out of the top go to position `q + l` when they are not all zero, and
            // otherwise position `q + l - 1` gets its own byte again, so that no branch decides
            // on them: for random numbers they are a coin toss for some shifts.
            let extra = out != 0;
            set::<BE>(
                dst,
                q + l - 1 + usize::from(extra),
                select_unpredictable(extra, out, high),
            );
        }
        zero::<BE>(dst, q);
    }
}

// Right shifts

/// The magnitude `x >> k` of a number of `l` significant bytes, rounded down when it is
/// negative, planned for the quotients that [`short_right`] leaves out of the registers.
#[derive(Clone, Copy)]
struct Right {
    /// The significant length of `x`.
    l: usize,
    /// The whole bytes of the shift.
    q: usize,
    /// The bits of the shift besides them, below 8.
    r: u32,
    /// The length of `x >> k`, not rounded: `l - q`, or one fewer when the top byte shifted by
    /// `r` bits is zero. At least 16.
    len: usize,
    /// Whether one is added to `x >> k`: the number is negative and a bit shifted out is not
    /// zero.
    round: bool,
}

impl Right {
    /// Plans the shift by `k` of the magnitude of `l` significant bytes at the least
    /// significant end `src` in the layout `BE`, whose most significant byte is `top`,
    /// `negative` if the number is, for which [`short_right`] does not hold.
    ///
    /// # Safety
    ///
    /// The `l` significant bytes at `src` must be valid for reads.
    #[inline(always)]
    unsafe fn plan<const BE: bool>(
        src: *const u8,
        l: usize,
        top: u8,
        k: usize,
        negative: bool,
    ) -> Right {
        let (q, r) = (k / 8, (k % 8) as u32);
        // SAFETY: `q < l`, as `short_right` does not hold.
        unsafe {
            Right {
                l,
                q,
                r,
                len: l - q - usize::from(top >> r == 0),
                round: negative && shifted_out::<BE>(src, l, q, r),
            }
        }
    }
}

/// Whether `x >> k`, for `x` of `l` significant bytes, is computed in registers: the bytes from
/// position `k / 8` up are at most 16. Longer quotients go to the kernels, which beat two
/// registers of 128 bits from 17 bytes up.
#[inline(always)]
fn short_right(l: usize, k: usize) -> bool {
    l.saturating_sub(k / 8) <= SMALL
}

/// The magnitude `x >> k`, where `x` has `l` significant bytes at the least significant end
/// `src` in the layout `BE`, with the sign `negative`, when [`short_right`] says so: the
/// quotient, rounded down, in registers.
///
/// # Safety
///
/// The `l` significant bytes at `src` must be valid for reads, and `short_right(l, k)` must hold.
#[inline(always)]
unsafe fn short_quotient<const BE: bool>(
    src: *const u8,
    l: usize,
    k: usize,
    negative: bool,
) -> Small {
    let (q, r) = (k / 8, (k % 8) as u32);
    let negative = negative && l != 0;
    // SAFETY: positions `q..l`, at most 16, are significant bytes, and so are the positions below
    // `q` that `shifted_out` reads when `q < l`.
    let (x, round) = unsafe {
        if q < l {
            let x = read_small(slice::<BE>(src, q, l - q), order::<BE>());
            (x, negative && shifted_out::<BE>(src, l, q, r))
        } else {
            (0, negative)
        }
    };
    // Rounded down, 16 bytes of 0xFF shifted by whole bytes carry into a 17th.
    let (magnitude, carry) = (x >> r).overflowing_add(u128::from(round));
    Small {
        lo: magnitude,
        hi: u128::from(carry),
        len: if carry {
            SMALL + 1
        } else {
            SMALL - (magnitude | 1).leading_zeros() as usize / 8
        },
        negative,
    }
}

/// `&x >> k` for the magnitude `x` in the bytes of `v` in the layout `BE` with the sign
/// `negative`, in one new vector of exactly its length; returns it and the sign of the result.
#[inline(always)]
fn shr_new<const BE: bool>(v: &[u8], negative: bool, k: usize) -> (Vec<u8>, bool) {
    let (l, top) = top_of::<BE>(v);
    let src = end::<BE>(v);
    // SAFETY: `src` is the least significant end of `v`, of `l` significant bytes.
    unsafe {
        if short_right(l, k) {
            let s = short_quotient::<BE>(src, l, k, negative);
            return (new_small(order::<BE>(), s), s.negative);
        }
        let plan = Right::plan::<BE>(src, l, top, k, negative);
        // Rounded down, the quotient is one byte longer when all its bytes are 0xFF. Whether it
        // is rounded is a coin toss when few bits are shifted out, and whether its bytes are all
        // 0xFF is not, for random numbers: that is asked first.
        let carry = negative && all_ones::<BE>(src, plan) && plan.round;
        let len = plan.len + usize::from(carry);
        let mut out = Vec::<u8>::with_capacity(len);
        let dst = out.as_mut_ptr();
        let dst = if BE { dst.add(len) } else { dst };
        if carry {
            ptr::write_bytes(at_mut::<BE>(dst, 0, len), 0, len);
            set::<BE>(dst, plan.len, 1);
        } else {
            shr_into::<BE>(dst, src, plan);
            if negative {
                // Without a branch on the rounding, a coin toss when few bits are shifted out.
                round_up::<BE>(dst, plan.len, quotient_low::<BE>(src, plan), plan.round);
            }
        }
        out.set_len(len);
        (out, negative)
    }
}

/// `x >>= k` for the magnitude `x` in `v` in the layout `BE` with the sign `negative`, in the
/// vector of `v`; returns the sign of the result.
#[inline(always)]
fn shr_in<const BE: bool>(v: &mut Vec<u8>, negative: bool, k: usize) -> bool {
    let (l, top) = top_of::<BE>(v);
    let len = v.len();
    let p = v.as_mut_ptr();
    // SAFETY: the `len` bytes at `p` are the vector's, of `l` significant bytes; the result is
    // at most `l` bytes long, one for zero, which the vector has room for unless it has no
    // capacity, which `put_small` sees to.
    unsafe {
        let src: *const u8 = if BE { p.add(len) } else { p };
        if short_right(l, k) {
            let s = short_quotient::<BE>(src, l, k, negative);
            put_small(v, order::<BE>(), s);
            return s.negative;
        }
        let plan = Right::plan::<BE>(src, l, top, k, negative);
        // Big-endian, the result starts at the start of the vector, and is written from its
        // least significant byte up, at the end; a group of eight is read at most eight
        // positions before it is written, which keeps every group read before it is written
        // over when the input ends where the vector does. Zero bytes above the most significant
        // one move the input's end away, and the significant bytes are moved to the start of
        // the vector first.
        let src = if BE && len != l {
            ptr::copy(p.add(len - l), p, l);
            p.add(l)
        } else {
            src
        };
        let dst = if BE { p.add(plan.len) } else { p };
        // The quotient's eight least significant bytes, read before they are written over. For
        // a negative number, whether it is rounded down is a coin toss when few bits are shifted
        // out, so the one is added without a branch, a zero when it is not rounded.
        let low = if negative {
            quotient_low::<BE>(src, plan)
        } else {
            0
        };
        shr_into::<BE>(dst, src, plan);
        let mut n = plan.len;
        if negative && round_up::<BE>(dst, n, low, plan.round) {
            // Every byte was 0xFF and is now zero: the quotient is 256^n, one byte longer, which
            // the vector has room for (it is at most `l - q + 1` bytes, `q >= 1` when the top
            // byte is 0xFF after the shift).
            n += 1;
            if BE {
                *p.add(n - 1) = 0;
                *p = 1;
            } else {
                *p.add(n - 1) = 1;
            }
        }
        v.set_len(n);
    }
    negative && l != 0
}

/// Writes the `plan.len` bytes of `x >> k`, not rounded, at the least significant end `dst` in
/// the layout `BE`, from the magnitude `x` at the least significant end `src` in the same
/// layout, which may be in the same vector.
///
/// The bytes are written from the least significant up, each group of eight read before a write
/// reaches it: the result's bytes lie at or below the positions of the bytes they come from,
/// little-endian at lower addresses; big-endian the input ends at the end of the vector and the
/// result `q` or `q + 1` bytes before.
///
/// # Safety
///
/// The `plan.l` significant bytes at `src` must be valid for reads and the `plan.len` bytes at
/// `dst` for writes. If they overlap, `dst` must be `src` little-endian, and big-endian
/// `src - plan.l + plan.len`, with the input's significant bytes at the start of the vector.
#[inline(always)]
unsafe fn shr_into<const BE: bool>(dst: *mut u8, src: *const u8, plan: Right) {
    let Right { l, q, r, len, .. } = plan;
    // SAFETY: as the caller guarantees.
    unsafe {
        if r == 0 {
            let (from, to) = (at::<BE>(src, q, len), at_mut::<BE>(dst, 0, len));
            // Big-endian in place, a shift by whole bytes leaves the quotient where it is.
            if !ptr::eq(from, to) {
                move_bytes(to, from, len);
            }
        } else {
            // The byte above the top of the quotient, position `q + len`: the top byte of `x`
            // when the shift leaves nothing of it, and above `x` otherwise. Chosen without a
            // branch, as for random numbers it is a coin toss for some shifts.
            let top = get::<BE>(src, l - 1);
            let above = top & 0u8.wrapping_sub(u8::from(q + len < l));
            kernels::shr_bits::<BE>(dst, src, l, q, r, len, above);
        }
    }
}

/// Whether a bit shifted out by `>> (8q + r)` is not zero: one of positions `0..q`, or of the
/// `r` low bits of position `q`, of the magnitude of `l > q` significant bytes at the least
/// significant end `src`. Eight positions at a time, so that it stops after one read for
/// random numbers.
///
/// # Safety
///
/// The `l` significant bytes at `src` must be valid for reads, and `q < l`.
#[inline(always)]
unsafe fn shifted_out<const BE: bool>(src: *const u8, l: usize, q: usize, r: u32) -> bool {
    let mut p = 0;
    // SAFETY: every read is of positions below `l`: `p + 8 <= q < l`, then `p + 8 <= l`, then
    // positions up to `q`.
    unsafe {
        while q - p >= 8 {
            if load::<BE>(src, p) != 0 {
                return true;
            }
            p += 8;
        }
        if l - p >= 8 {
            // The fewer than eight positions left and the `r` bits, in one group: below 64 bits.
            let bits = 8 * (q - p) as u32 + r;
            return load::<BE>(src, p) & ((1 << bits) - 1) != 0;
        }
        while p < q {
            if get::<BE>(src, p) != 0 {
                return true;
            }
            p += 1;
        }
        get::<BE>(src, q) & ((1 << r) - 1) != 0
    }
}

/// Whether the `plan.len` bytes of `x >> k`, before it is rounded, are all 0xFF, for the
/// magnitude `x` at the least significant end `src` in the layout `BE`: then the rounded
/// quotient is one byte longer. For random numbers it stops after the first group of eight.
///
/// # Safety
///
/// The `plan.l` significant bytes at `src` must be valid for reads.
unsafe fn all_ones<const BE: bool>(src: *const u8, plan: Right) -> bool {
    let Right { l, q, r, len, .. } = plan;
    // The byte of the quotient at position `p`; the bits from above the magnitude are zero.
    // SAFETY: positions `q + p` and `q + p + 1` are read only below `l`.
    let byte = |p: usize| unsafe {
        let above = if q + p + 1 < l {
            get::<BE>(src, q + p + 1)
        } else {
            0
        };
        ((u16::from(get::<BE>(src, q + p)) | (u16::from(above) << 8)) >> r) as u8
    };
    let mut p = 0;
    // SAFETY: positions `q + p..q + p + 8` lie below `q + len <= l`.
    unsafe {
        while p + 8 <= len && q + p + 8 < l {
            let above = u64::from(get::<BE>(src, q + p + 8));
            // `<< (64 - r)` in two steps: for `r == 0` every bit moves out.
            let v = (load::<BE>(src, q + p) >> r) | ((above << 1) << (63 - r));
            if v != u64::MAX {
                return false;
            }
            p += 8;
        }
    }
    (p..len).all(|p| byte(p) == 0xFF)
}

/// Positions `0..8` of `x >> k`, not rounded, for the magnitude `x` at the least significant end
/// `src` in the layout `BE`, planned as `plan`: from positions `q..q + 9` of `x`, all below `l`
/// as the quotient is at least 16 bytes long.
///
/// # Safety
///
/// The `plan.l` significant bytes at `src` must be valid for reads.
#[inline(always)]
unsafe fn quotient_low<const BE: bool>(src: *const u8, plan: Right) -> u64 {
    let Right { q, r, .. } = plan;
    // SAFETY: `q + 9 <= l`, as the quotient's `l - q` or `l - q - 1` bytes are at least 16.
    unsafe {
        let above = u64::from(get::<BE>(src, q + 8));
        // `<< (64 - r)` in two steps: for `r == 0` every bit moves out.
        (load::<BE>(src, q) >> r) | ((above << 1) << (63 - r))
    }
}

/// Adds `round` (one or zero) to the quotient of `n >= 16` bytes at the least significant end
/// `dst` in the layout `BE`, whose eight least significant bytes were `low` when they were
/// written: they are written again plus `round`, and only when that carries are the bytes above
/// read and carried into, which a random number does not need. Reading back the bytes just
/// written would wait for the stores of the kernel, which the core cannot forward to a load of
/// another width or place. Returns whether it carries out of the top, all bytes then zero.
///
/// # Safety
///
/// The `n` bytes must be valid for reads and writes, and positions `0..8` hold `low`.
#[inline(always)]
unsafe fn round_up<const BE: bool>(dst: *mut u8, n: usize, low: u64, round: bool) -> bool {
    let (sum, carry) = low.overflowing_add(u64::from(round));
    // SAFETY: as the caller guarantees.
    unsafe {
        store::<BE>(dst, 0, sum);
        carry && increment::<BE>(dst, 8, n)
    }
}

/// Adds one to positions `p..n` of the magnitude at the least significant end `dst` in the
/// layout `BE`, eight bytes at a time while the carry goes on; returns whether it carries out of
/// the top, all those bytes then zero. Out of line and cold: it runs only when the eight bytes
/// below carry out, all 0xFF.
///
/// # Safety
///
/// The bytes must be valid for reads and writes.
#[cold]
#[inline(never)]
unsafe fn increment<const BE: bool>(dst: *mut u8, mut p: usize, n: usize) -> bool {
    // SAFETY: every access is of positions below `n`.
    unsafe {
        while n - p >= 8 {
            let v = load::<BE>(dst, p).wrapping_add(1);
            store::<BE>(dst, p, v);
            if v != 0 {
                return false;
            }
            p += 8;
        }
        while p < n {
            let v = get::<BE>(dst, p).wrapping_add(1);
            set::<BE>(dst, p, v);
            if v != 0 {
                return false;
            }
            p += 1;
        }
    }
    true
}

// Memory

/// One new vector of capacity `len`, empty, `len > 0`.
///
/// # Panics
///
/// When it cannot be allocated, rather than aborting as a failed allocation does otherwise: a
/// left shift can ask for any length, and a shift too long for the memory is a mistake of the
/// caller's, not a lack of memory.
fn allocate(len: usize) -> Vec<u8> {
    let Ok(layout) = Layout::array::<u8>(len) else {
        too_long(len)
    };
    // SAFETY: the layout is not empty, as `len > 0`.
    let p = unsafe { alloc(layout) };
    if p.is_null() {
        too_long(len);
    }
    // SAFETY: `p` was allocated by the global allocator with the layout of `len` bytes, which is
    // the capacity given, and the vector is empty.
    unsafe { Vec::from_raw_parts(p, 0, len) }
}

/// Panics for a left shift whose result of `len` bytes cannot be allocated.
#[cold]
#[inline(never)]
fn too_long(len: usize) -> ! {
    panic!("a shifted number of {len} bytes cannot be allocated")
}

/// Writes zeros over positions `0..q` at the least significant end `dst` in the layout `BE`.
///
/// # Safety
///
/// The `q` bytes must be valid for writes.
#[inline(always)]
unsafe fn zero<const BE: bool>(dst: *mut u8, q: usize) {
    // SAFETY: as the caller guarantees.
    unsafe { zero_bytes(at_mut::<BE>(dst, 0, q), q) };
}

/// Writes zeros over the `n` bytes at `dst`: up to 64 bytes inline, in stores of 16 bytes or
/// fewer that overlap, and with `memset` beyond, whose call costs more than a short run of
/// zeros.
///
/// # Safety
///
/// The `n` bytes must be valid for writes.
#[inline(always)]
unsafe fn zero_bytes(dst: *mut u8, n: usize) {
    // SAFETY: every write lies in the `n` bytes at `dst`.
    unsafe {
        if n == 0 {
            // Shifts by fewer than eight bits, first.
        } else if n > 64 {
            ptr::write_bytes(dst, 0, n);
        } else if n >= 16 {
            write(dst, [0u8; 16]);
            write(dst.add(n - 16), [0u8; 16]);
            if n > 32 {
                write(dst.add(16), [0u8; 16]);
                write(dst.add(n - 32), [0u8; 16]);
            }
        } else if n >= 8 {
            write(dst, [0u8; 8]);
            write(dst.add(n - 8), [0u8; 8]);
        } else if n >= 4 {
            write(dst, [0u8; 4]);
            write(dst.add(n - 4), [0u8; 4]);
        } else {
            *dst = 0;
            *dst.add(n / 2) = 0;
            *dst.add(n - 1) = 0;
        }
    }
}

/// Copies the `n` bytes at `src` to `dst`, which may overlap them: up to 64 bytes inline, every
/// byte read before any is written (the reads and the writes overlap where the length is not a
/// multiple of their width, and agree there), and with `memmove` beyond.
///
/// # Safety
///
/// The `n` bytes at `src` must be valid for reads and the `n` at `dst` for writes.
#[inline(always)]
unsafe fn move_bytes(dst: *mut u8, src: *const u8, n: usize) {
    // SAFETY: every read lies in the `n` bytes at `src` and every write in the `n` at `dst`.
    unsafe {
        if n > 64 {
            ptr::copy(src, dst, n);
        } else if n > 32 {
            let a: [u8; 16] = read(src);
            let b: [u8; 16] = read(src.add(16));
            let c: [u8; 16] = read(src.add(n - 32));
            let d: [u8; 16] = read(src.add(n - 16));
            write(dst, a);
            write(dst.add(16), b);
            write(dst.add(n - 32), c);
            write(dst.add(n - 16), d);
        } else if n >= 16 {
            let a: [u8; 16] = read(src);
            let b: [u8; 16] = read(src.add(n - 16));
            write(dst, a);
            write(dst.add(n - 16), b);
        } else if n >= 8 {
            let a: [u8; 8] = read(src);
            let b: [u8; 8] = read(src.add(n - 8));
            write(dst, a);
            write(dst.add(n - 8), b);
        } else if n >= 4 {
            let a: [u8; 4] = read(src);
            let b: [u8; 4] = read(src.add(n - 4));
            write(dst, a);
            write(dst.add(n - 4), b);
        } else if n > 0 {
            let (a, b, c) = (*src, *src.add(n / 2), *src.add(n - 1));
            *dst = a;
            *dst.add(n / 2) = b;
            *dst.add(n - 1) = c;
        }
    }
}

// Positions

/// The byte order of the layout `BE`.
#[inline(always)]
fn order<const BE: bool>() -> ByteOrder {
    if BE {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    }
}

/// Positions `p..p + n` of the magnitude at the least significant end `end` in the layout `BE`,
/// as a slice of its bytes in that layout.
///
/// # Safety
///
/// The `n` bytes must be valid for reads, and not written while the slice is alive.
#[inline(always)]
unsafe fn slice<'a, const BE: bool>(end: *const u8, p: usize, n: usize) -> &'a [u8] {
    // SAFETY: as the caller guarantees.
    unsafe { &*ptr::slice_from_raw_parts(at::<BE>(end, p, n), n) }
}

/// The least significant end of the magnitude in `v` in the layout `BE`: the address of its
/// first byte little-endian, and the address one past its last byte big-endian.
#[inline(always)]
fn end<const BE: bool>(v: &[u8]) -> *const u8 {
    if BE { v.as_ptr_range().end } else { v.as_ptr() }
}

/// The lowest address of positions `p..p + n` of the magnitude at the least significant end
/// `end` in the layout `BE`: `end + p` little-endian, `end - p - n` big-endian.
///
/// # Safety
///
/// The address must lie in the allocation of `end`.
#[inline(always)]
unsafe fn at<const BE: bool>(end: *const u8, p: usize, n: usize) -> *const u8 {
    // SAFETY: as the caller guarantees.
    unsafe { if BE { end.sub(p + n) } else { end.add(p) } }
}

/// [`at`] for writing.
///
/// # Safety
///
/// As for [`at`].
#[inline(always)]
unsafe fn at_mut<const BE: bool>(end: *mut u8, p: usize, n: usize) -> *mut u8 {
    // SAFETY: as the caller guarantees.
    unsafe { if BE { end.sub(p + n) } else { end.add(p) } }
}

/// Reads position `p` of the magnitude at the least significant end `end` in the layout `BE`.
///
/// # Safety
///
/// The byte must be valid for reading.
#[inline(always)]
unsafe fn get<const BE: bool>(end: *const u8, p: usize) -> u8 {
    // SAFETY: as the caller guarantees.
    unsafe { *at::<BE>(end, p, 1) }
}

/// Writes position `p` of the magnitude at the least significant end `end` in the layout `BE`.
///
/// # Safety
///
/// The byte must be valid for writing.
#[inline(always)]
unsafe fn set<const BE: bool>(end: *mut u8, p: usize, v: u8) {
    // SAFETY: as the caller guarantees.
    unsafe { *at_mut::<BE>(end, p, 1) = v };
}

/// Reads positions `p..p + 8` of the magnitude at the least significant end `end` in the
/// layout `BE` as a `u64` whose least significant byte is position `p`.
///
/// # Safety
///
/// The eight bytes must be valid for reading; they need no alignment.
#[inline(always)]
unsafe fn load<const BE: bool>(end: *const u8, p: usize) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        let bytes = read(at::<BE>(end, p, 8));
        if BE {
            u64::from_be_bytes(bytes)
        } else {
            u64::from_le_bytes(bytes)
        }
    }
}

/// Writes positions `p..p + 8` of the magnitude at the least significant end `end` in the
/// layout `BE`, as [`load`] reads them.
///
/// # Safety
///
/// The eight bytes must be valid for writing; they need no alignment.
#[inline(always)]
unsafe fn store<const BE: bool>(end: *mut u8, p: usize, v: u64) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let bytes = if BE { v.to_be_bytes() } else { v.to_le_bytes() };
        write(at_mut::<BE>(end, p, 8), bytes);
    }
}
