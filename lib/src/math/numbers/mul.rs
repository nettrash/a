//! Multiplication of [`Number`]s: `*` and `*=`, with either operand owned or borrowed.
//!
//! The magnitude of a product is the product of the magnitudes, and it is negative when exactly
//! one operand is and neither is zero. A zero magnitude is zero whatever `negative` says, and
//! zero bytes above the most significant one count for nothing.
//!
//! Operands whose vectors both hold at most 16 bytes take a short path in registers: their
//! magnitudes are read into `u128`s as the addition reads them, multiplied into four limbs, and
//! the product, at most 32 bytes, is written in the byte order of the left operand. Up to 32 bytes
//! each, the general path does the same with four limbs an operand and a product of eight, and up
//! to 48 with six and a product of twelve, in registers too.
//!
//! Longer operands are read into limbs of eight bytes, native `u64`s, the least significant first,
//! in working memory: a buffer in the operator's own frame when the operands have at most
//! [`MEDIUM_LIMBS`] limbs together, which the schoolbook kernels multiply directly; otherwise a
//! buffer on the stack of [`STACK_LIMBS`] limbs, or one on the heap past it, for the operands, the
//! product and the scratch of Karatsuba's and Toom-Cook's methods (see [`limbs`]), each starting on
//! 16 bytes. The kernels are inline assembly on 64-bit little-endian `aarch64` and, with BMI2, on
//! `x86_64`, and portable Rust elsewhere and under Miri. A borrowed operand multiplied by itself
//! (`&a * &a`) is squared, which needs about half the products of limbs.
//!
//! The product's length in the shortest form is then known, and a vector is chosen for it as for
//! a sum: the vector of an operand taken by value, or of the target of `*=`, when it has room for
//! the product (the left operand's first), and otherwise one new vector of exactly its length.
//! The product's bytes are written there, in the byte order of the left operand, the bytes above
//! the whole limbs as the eight that end at the top, over the limb below.

// The kernels of the schoolbook method, one set per target under the name `kernels`, chosen as
// those of the addition are: assembly on 64-bit little-endian aarch64 where the system allows
// unaligned accesses and on x86_64, and portable Rust elsewhere and under Miri.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
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
#[path = "mul/aarch64.rs"]
mod kernels;
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "mul/x86_64.rs"]
mod kernels;
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_pointer_width = "64",
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
    ),
    all(target_arch = "x86_64", not(miri))
)))]
#[path = "mul/portable.rs"]
mod kernels;
pub(super) mod limbs;
mod toom;

use super::add::{SMALL, read, read_small, significant_len, small, write, write_small};
use super::{ByteOrder, Number};
use core::mem::{self, MaybeUninit};
use core::ops::{Mul, MulAssign};
use core::ptr;

impl Mul<&Number> for &Number {
    type Output = Number;

    /// Returns `self * rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The product is built in one new vector of exactly its length (one byte for zero); both
    /// operands are left as they were. A number multiplied by itself (`&a * &a`) is squared, each
    /// product of two different limbs computed once. The operands may have any signs, byte orders
    /// and lengths, and need not be in the shortest form.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x01_00 * -0x01_01 = -0x01_01_00, in the byte order of the left operand.
    /// let a = Number::new_from_i32(0x0100, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(-0x0101, ByteOrder::LittleEndian);
    /// let product = &a * &b;
    /// assert_eq!(product.value, [0x01, 0x01, 0x00]);
    /// assert!(product.is_negative());
    /// assert!(product.byte_order() == ByteOrder::BigEndian);
    ///
    /// // Anything times zero is zero, one zero byte and not negative.
    /// let zero = &b * &Number::new_from_u8(0, ByteOrder::BigEndian);
    /// assert_eq!(zero.value, [0]);
    /// assert!(!zero.is_negative());
    /// ```
    fn mul(self, rhs: &Number) -> Number {
        if small(self, rhs) {
            // SAFETY: both vectors hold at most 16 bytes.
            let p = unsafe { small_product(self, rhs) };
            return small_number(self.byte_order, p);
        }
        ref_times_ref(self, rhs)
    }
}

impl Mul<&Number> for Number {
    type Output = Number;

    /// Returns `self * rhs`, in the byte order of `self`, in the shortest form, as `self *= rhs`
    /// does.
    ///
    /// The product is built in the vector of `self` when its capacity has room for it, and
    /// otherwise in one new vector of exactly its length. `rhs` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut value = Vec::with_capacity(4);
    /// value.extend([0xFF, 0xFF]);
    /// let a = Number::new(value, ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    ///
    /// // 0xFF_FF * 0xFF_FF = 0xFF_FE_00_01, in the vector of `a`.
    /// let product = a * &Number::new_from_u16(0xFFFF, ByteOrder::LittleEndian);
    /// assert_eq!(product.value, [0xFF, 0xFE, 0x00, 0x01]);
    /// assert_eq!(product.value.as_ptr(), pointer);
    /// ```
    fn mul(mut self, rhs: &Number) -> Number {
        self *= rhs;
        self
    }
}

impl Mul<Number> for &Number {
    type Output = Number;

    /// Returns `self * rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The product is built in the vector of `rhs` when its capacity has room for it, also when
    /// `rhs` is in the other byte order; otherwise it gets one new vector of exactly its length.
    /// `self` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u16(0x0102, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(4);
    /// value.extend([0x03]);
    /// let b = Number::new(value, ByteOrder::LittleEndian);
    /// let pointer = b.value.as_ptr();
    ///
    /// // 0x01_02 * 3 = 0x03_06, big-endian, in the vector of `b`.
    /// let product = &a * b;
    /// assert_eq!(product.value, [0x03, 0x06]);
    /// assert_eq!(product.value.as_ptr(), pointer);
    /// ```
    fn mul(self, mut rhs: Number) -> Number {
        if small(self, &rhs) {
            // SAFETY: both vectors hold at most 16 bytes.
            let p = unsafe { small_product(self, &rhs) };
            let order = self.byte_order;
            put_small(&mut rhs.value, order, p);
            return Number {
                value: rhs.value,
                negative: p.negative,
                byte_order: order,
            };
        }
        // `rhs` is dropped on return, with the vector it holds then, if any.
        ref_times_value(self, &mut rhs)
    }
}

impl Mul<Number> for Number {
    type Output = Number;

    /// Returns `self * rhs`, in the byte order of `self`, in the shortest form, as `self *= rhs`
    /// does.
    ///
    /// The product is built in the vector of `self` when its capacity has room for it, or else
    /// in the vector of `rhs` when that has room, also when `rhs` is in the other byte order;
    /// otherwise it gets one new vector of exactly its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // -0x01_00 * -0x02_FF = 0x02_FF_00.
    /// let a = Number::new_from_i32(-0x0100, ByteOrder::LittleEndian);
    /// let b = Number::new_from_i32(-0x02FF, ByteOrder::BigEndian);
    /// let product = a * b;
    /// assert_eq!(product.value, [0x00, 0xFF, 0x02]);
    /// assert!(!product.is_negative());
    /// ```
    fn mul(mut self, rhs: Number) -> Number {
        self *= rhs;
        self
    }
}

impl MulAssign<&Number> for Number {
    /// Multiplies `self` by `rhs`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The product is built in the vector of `self` when its capacity has room for it, and
    /// otherwise in one new vector of exactly its length, which replaces it. `rhs` is left as it
    /// was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u8(0x10, ByteOrder::LittleEndian);
    /// n *= &Number::new_from_u16(0x0101, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x10, 0x10]);
    /// n *= &Number::new_from_i8(-2, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x20, 0x20]);
    /// assert!(n.is_negative());
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: &Number) {
        if small(self, rhs) {
            // SAFETY: both vectors hold at most 16 bytes.
            let p = unsafe { small_product(self, rhs) };
            put_small(&mut self.value, self.byte_order, p);
            self.negative = p.negative;
            return;
        }
        mul_assign_ref(self, rhs);
    }
}

impl MulAssign<Number> for Number {
    /// Multiplies `self` by `rhs`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The product is built in the vector of `self` when its capacity has room for it, or else
    /// in the vector of `rhs` when that has room, also when `rhs` is in the other byte order;
    /// otherwise it gets one new vector of exactly its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // `n` has no room for 0xFF * 0x01_02 = 0x01_00_FE, but the vector of the right operand
    /// // has.
    /// let mut n = Number::new_from_u8(0xFF, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(3);
    /// value.extend([1, 2]);
    /// let b = Number::new(value, ByteOrder::BigEndian);
    /// let pointer = b.value.as_ptr();
    /// n *= b;
    /// assert_eq!(n.value, [0x01, 0x00, 0xFE]);
    /// assert_eq!(n.value.as_ptr(), pointer);
    /// ```
    #[inline]
    fn mul_assign(&mut self, mut rhs: Number) {
        if small(self, &rhs) {
            // SAFETY: both vectors hold at most 16 bytes.
            let p = unsafe { small_product(self, &rhs) };
            if self.value.capacity() < p.len && rhs.value.capacity() >= p.len {
                // The vector of `self` goes to `rhs`, and is dropped with it.
                mem::swap(&mut self.value, &mut rhs.value);
            }
            put_small(&mut self.value, self.byte_order, p);
            self.negative = p.negative;
            return;
        }
        // `rhs` is dropped on return, with the vector it holds then.
        mul_assign_value(self, &mut rhs);
    }
}

// Small numbers

/// The product of two small operands, as the small path computes it in registers.
#[derive(Clone, Copy)]
struct Small {
    /// The four limbs of the magnitude, the least significant first.
    limbs: [u64; 4],
    /// The length of the product in the shortest form, from 1 to 32: one byte for zero.
    len: usize,
    /// Whether the product is negative; never for zero.
    negative: bool,
}

/// The product of the small operands `a` and `b`.
///
/// # Safety
///
/// Both vectors must hold at most [`SMALL`] bytes.
#[inline(always)]
unsafe fn small_product(a: &Number, b: &Number) -> Small {
    // SAFETY: as the caller guarantees.
    let (x, y) = unsafe {
        (
            read_small(&a.value, a.byte_order),
            read_small(&b.value, b.byte_order),
        )
    };
    let (x0, x1) = (x as u64, (x >> 64) as u64);
    let (y0, y1) = (y as u64, (y >> 64) as u64);
    let p00 = u128::from(x0) * u128::from(y0);
    let p01 = u128::from(x0) * u128::from(y1);
    let p10 = u128::from(x1) * u128::from(y0);
    let p11 = u128::from(x1) * u128::from(y1);
    // Each column is at most three limbs' worth below 2^66, a `u128`.
    let c1 = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
    let c2 = (c1 >> 64) + (p01 >> 64) + (p10 >> 64) + u128::from(p11 as u64);
    let c3 = (c2 >> 64) + (p11 >> 64);
    let limbs = [p00 as u64, c1 as u64, c2 as u64, c3 as u64];
    let zeros = if limbs[3] != 0 {
        limbs[3].leading_zeros() / 8
    } else if limbs[2] != 0 {
        8 + limbs[2].leading_zeros() / 8
    } else if limbs[1] != 0 {
        16 + limbs[1].leading_zeros() / 8
    } else {
        24 + (limbs[0] | 1).leading_zeros() / 8
    };
    Small {
        limbs,
        len: 32 - zeros as usize,
        negative: a.negative != b.negative && x != 0 && y != 0,
    }
}

/// Writes the small product `p` over the first `p.len` bytes at `dst`, in the layout of `order`.
/// Up to 16 bytes as [`write_small`] writes a sum; from 17, the 16 least significant bytes and the
/// 16 most significant ones, which overlap and agree where they do.
///
/// # Safety
///
/// `dst` must be valid for writing `p.len` bytes.
#[inline(always)]
unsafe fn write_small_product(dst: *mut u8, order: ByteOrder, p: Small) {
    let [l0, l1, l2, l3] = p.limbs;
    let len = p.len;
    // SAFETY: every write lies inside the `len` bytes at `dst`: `[0, 16)` and `[len - 16, len)`
    // with `len > 16`, or what `write_small` writes for `len <= 16`.
    unsafe {
        if len <= SMALL {
            write_small(dst, order, l0, l1, len);
            return;
        }
        let low = u128::from(l0) | (u128::from(l1) << 64);
        let high = u128::from(l2) | (u128::from(l3) << 64);
        // Positions `len - 16..len`: the 256 bits shifted down by `8 (len - 16)`, 8 to 128.
        let k = 8 * (len - 16) as u32;
        let top = low.checked_shr(k).unwrap_or(0) | high << (128 - k);
        if order == ByteOrder::BigEndian {
            write(dst.add(len - 16), low.to_be_bytes());
            write(dst, top.to_be_bytes());
        } else {
            write(dst, low.to_le_bytes());
            write(dst.add(len - 16), top.to_le_bytes());
        }
    }
}

/// Writes the small product `p` as the whole value of `v`, in its vector when it has room, in a
/// new one otherwise.
#[inline(always)]
fn put_small(v: &mut Vec<u8>, order: ByteOrder, p: Small) {
    if v.capacity() >= p.len {
        // SAFETY: the vector has room for the `p.len` bytes, all of which are written before the
        // length is set.
        unsafe {
            write_small_product(v.as_mut_ptr(), order, p);
            v.set_len(p.len);
        }
    } else {
        replace_small(v, order, p);
    }
}

/// Replaces `v` with one new vector of exactly the small product's length, holding it. Out of
/// line and cold, so that the small path saves no registers for it.
#[cold]
#[inline(never)]
fn replace_small(v: &mut Vec<u8>, order: ByteOrder, p: Small) {
    *v = new_small(order, p);
}

/// The small product `p` in one new vector of exactly its length.
#[inline(always)]
fn new_small(order: ByteOrder, p: Small) -> Vec<u8> {
    let mut v = Vec::with_capacity(p.len);
    // SAFETY: the new vector has room for the `p.len` bytes, all of which are written before the
    // length is set.
    unsafe {
        write_small_product(v.as_mut_ptr(), order, p);
        v.set_len(p.len);
    }
    v
}

/// The small product `p` as a number in one new vector, in the byte order `order`. Out of line,
/// so that the small path of `&a * &b` saves no registers for the allocation.
#[inline(never)]
fn small_number(order: ByteOrder, p: Small) -> Number {
    Number {
        value: new_small(order, p),
        negative: p.negative,
        byte_order: order,
    }
}

// Operands of up to 32 bytes

/// The longest vector, in bytes, of the operands that the general path multiplies in registers,
/// four limbs each, as the short path does with two.
pub(super) const WIDE: usize = 32;

/// Whether `a * b` is computed in registers as four limbs each: both vectors hold at most
/// [`WIDE`] bytes.
#[inline(always)]
fn wide(a: &Number, b: &Number) -> bool {
    a.value.len() <= WIDE && b.value.len() <= WIDE
}

/// The product of two operands of up to 32 bytes, computed in registers.
#[derive(Clone, Copy)]
pub(super) struct Wide {
    /// The eight limbs of the magnitude, the least significant first.
    pub(super) limbs: [u64; 8],
    /// The length of the product in the shortest form, from 1 to 64: one byte for zero.
    pub(super) len: usize,
    /// Whether the product is negative; never for zero.
    pub(super) negative: bool,
}

/// The magnitude in the at most 32 bytes of `v`, in the layout of `order`, as four limbs: up to 16
/// bytes as the addition reads them; from 17, the 16 least significant bytes and the 16 most
/// significant ones, which overlap and agree where they do.
///
/// # Safety
///
/// `v` must hold at most [`WIDE`] bytes.
#[inline(always)]
pub(super) unsafe fn read_wide(v: &[u8], order: ByteOrder) -> [u64; 4] {
    let (p, n) = (v.as_ptr(), v.len());
    // SAFETY: every read lies inside the `n` bytes of `v`: `[0, 16)` and `[n - 16, n)` with
    // `n > 16`, or what `read_small` reads.
    unsafe {
        if n <= SMALL {
            let x = read_small(v, order);
            return [x as u64, (x >> 64) as u64, 0, 0];
        }
        let (low, top) = if order == ByteOrder::BigEndian {
            (
                u128::from_be_bytes(read(p.add(n - 16))),
                u128::from_be_bytes(read(p)),
            )
        } else {
            (
                u128::from_le_bytes(read(p)),
                u128::from_le_bytes(read(p.add(n - 16))),
            )
        };
        // `top` holds positions `n - 16..n`; those from 16 up are the high half, a shift by
        // `8 (32 - n)` down, below 128.
        let high = top >> (8 * (WIDE - n));
        [
            low as u64,
            (low >> 64) as u64,
            high as u64,
            (high >> 64) as u64,
        ]
    }
}

/// The product of the operands `a` and `b` of up to 32 bytes: the schoolbook method on four limbs
/// each, which the compiler unrolls and keeps in registers.
///
/// # Safety
///
/// Both vectors must hold at most [`WIDE`] bytes.
#[inline(always)]
unsafe fn wide_product(a: &Number, b: &Number) -> Wide {
    // SAFETY: as the caller guarantees.
    let (x, y) = unsafe {
        (
            read_wide(&a.value, a.byte_order),
            read_wide(&b.value, b.byte_order),
        )
    };
    let mut r = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let t = u128::from(x[i]) * u128::from(y[j]) + u128::from(r[i + j]) + u128::from(carry);
            r[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        r[i + 4] = carry;
    }
    let top = r.iter().rposition(|&l| l != 0);
    let len = match top {
        Some(t) => 8 * t + 8 - r[t].leading_zeros() as usize / 8,
        None => 1,
    };
    let nonzero = |v: [u64; 4]| v.iter().any(|&l| l != 0);
    Wide {
        limbs: r,
        len,
        negative: a.negative != b.negative && nonzero(x) && nonzero(y),
    }
}

/// Writes the product `w` over the first `w.len` bytes at `dst`, in the layout of `order`: up to
/// 32 bytes as the short path writes them; from 33, the 32 least significant bytes and the 32 most
/// significant ones, which overlap and agree where they do.
///
/// # Safety
///
/// `dst` must be valid for writing `w.len` bytes.
#[inline(always)]
pub(super) unsafe fn write_wide(dst: *mut u8, order: ByteOrder, w: &Wide) {
    let (l, len) = (w.limbs, w.len);
    // SAFETY: every write lies inside the `len` bytes at `dst`.
    unsafe {
        if len <= 2 * SMALL {
            let p = Small {
                limbs: [l[0], l[1], l[2], l[3]],
                len,
                negative: w.negative,
            };
            write_small_product(dst, order, p);
            return;
        }
        // Positions `len - 32..len`: the limbs shifted down by `8 (len - 32)` bits, 8 to 256.
        let k = 8 * (len - 32);
        let (q, b) = (k / 64, (k % 64) as u32);
        let limb = |i: usize| if i < 8 { l[i] } else { 0 };
        let mut t = [0u64; 4];
        for (i, v) in t.iter_mut().enumerate() {
            *v = if b == 0 {
                limb(i + q)
            } else {
                (limb(i + q) >> b) | (limb(i + q + 1) << (64 - b))
            };
        }
        let u = |a: u64, b: u64| u128::from(a) | (u128::from(b) << 64);
        let (l01, l23, t01, t23) = (u(l[0], l[1]), u(l[2], l[3]), u(t[0], t[1]), u(t[2], t[3]));
        if order == ByteOrder::BigEndian {
            write(dst.add(len - 16), l01.to_be_bytes());
            write(dst.add(len - 32), l23.to_be_bytes());
            write(dst.add(16), t01.to_be_bytes());
            write(dst, t23.to_be_bytes());
        } else {
            write(dst, l01.to_le_bytes());
            write(dst.add(16), l23.to_le_bytes());
            write(dst.add(len - 32), t01.to_le_bytes());
            write(dst.add(len - 16), t23.to_le_bytes());
        }
    }
}

/// Writes the product `w` as the whole value of `v`, in its vector when it has room, in a new one
/// otherwise.
#[inline(always)]
pub(super) fn put_wide(v: &mut Vec<u8>, order: ByteOrder, w: &Wide) {
    if v.capacity() < w.len {
        *v = Vec::with_capacity(w.len);
    }
    // SAFETY: the vector has room for the `w.len` bytes, all of which are written before the
    // length is set.
    unsafe {
        write_wide(v.as_mut_ptr(), order, w);
        v.set_len(w.len);
    }
}

// Operands of up to 48 bytes

/// The longest vector, in bytes, of the operands that the general path multiplies in registers as
/// six limbs each, past [`WIDE`]. Measured on an Apple M4 Max, `a *= &b` of 33 to 48 bytes took
/// 53 to 60 cycles so, against 70 to 91 through limbs in memory. Eight limbs each, up to 64 bytes,
/// did not pay: the compiler's code for the 64 limb products took about twice the instructions.
const WIDEST: usize = 48;

/// Whether `a * b` is computed in registers as six limbs each: both vectors hold at most
/// [`WIDEST`] bytes.
#[inline(always)]
fn widest(a: &Number, b: &Number) -> bool {
    a.value.len() <= WIDEST && b.value.len() <= WIDEST
}

/// The product of two operands of up to 48 bytes, computed in registers.
#[derive(Clone, Copy)]
struct Widest {
    /// The twelve limbs of the magnitude, the least significant first.
    limbs: [u64; 12],
    /// The length of the product in the shortest form, from 1 to 96: one byte for zero.
    len: usize,
    /// Whether the product is negative; never for zero.
    negative: bool,
}

/// The magnitude in the at most 48 bytes of `v`, in the layout of `order`, as six limbs: up to 32
/// bytes as [`read_wide`] reads them; from 33, the 32 least significant bytes and the 16 most
/// significant ones, which overlap and agree where they do.
///
/// # Safety
///
/// `v` must hold at most [`WIDEST`] bytes.
#[inline(always)]
unsafe fn read_widest(v: &[u8], order: ByteOrder) -> [u64; 6] {
    let (p, n) = (v.as_ptr(), v.len());
    // SAFETY: every read lies inside the `n` bytes of `v`: `[0, 32)` and `[n - 16, n)` with
    // `n > 32`, or what `read_wide` reads.
    unsafe {
        if n <= WIDE {
            let [l0, l1, l2, l3] = read_wide(v, order);
            return [l0, l1, l2, l3, 0, 0];
        }
        let (low, middle, top) = if order == ByteOrder::BigEndian {
            (
                u128::from_be_bytes(read(p.add(n - 16))),
                u128::from_be_bytes(read(p.add(n - 32))),
                u128::from_be_bytes(read(p)),
            )
        } else {
            (
                u128::from_le_bytes(read(p)),
                u128::from_le_bytes(read(p.add(16))),
                u128::from_le_bytes(read(p.add(n - 16))),
            )
        };
        // `top` holds positions `n - 16..n`; those from 32 up are the high part, a shift by
        // `8 (48 - n)` down, below 128.
        let high = top >> (8 * (WIDEST - n));
        [
            low as u64,
            (low >> 64) as u64,
            middle as u64,
            (middle >> 64) as u64,
            high as u64,
            (high >> 64) as u64,
        ]
    }
}

/// The product of the operands `a` and `b` of up to 48 bytes: the schoolbook method on six limbs
/// each, which the compiler unrolls and keeps in registers.
///
/// # Safety
///
/// Both vectors must hold at most [`WIDEST`] bytes.
#[inline(always)]
unsafe fn widest_product(a: &Number, b: &Number) -> Widest {
    // SAFETY: as the caller guarantees.
    let (x, y) = unsafe {
        (
            read_widest(&a.value, a.byte_order),
            read_widest(&b.value, b.byte_order),
        )
    };
    let mut r = [0u64; 12];
    for i in 0..6 {
        let mut carry = 0u64;
        for j in 0..6 {
            let t = u128::from(x[i]) * u128::from(y[j]) + u128::from(r[i + j]) + u128::from(carry);
            r[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        r[i + 6] = carry;
    }
    let top = r.iter().rposition(|&l| l != 0);
    let len = match top {
        Some(t) => 8 * t + 8 - r[t].leading_zeros() as usize / 8,
        None => 1,
    };
    let nonzero = |v: [u64; 6]| v.iter().any(|&l| l != 0);
    Widest {
        limbs: r,
        len,
        negative: a.negative != b.negative && nonzero(x) && nonzero(y),
    }
}

/// Writes the product `w` over the first `w.len` bytes at `dst`, in the layout of `order`: up to
/// 64 bytes as [`write_wide`] writes them; from 65, the 64 least significant bytes and the 32 most
/// significant ones, which overlap and agree where they do.
///
/// # Safety
///
/// `dst` must be valid for writing `w.len` bytes.
#[inline(always)]
unsafe fn write_widest(dst: *mut u8, order: ByteOrder, w: &Widest) {
    let (l, len) = (w.limbs, w.len);
    // SAFETY: every write lies inside the `len` bytes at `dst`.
    unsafe {
        if len <= 4 * SMALL {
            let mut limbs = [0; 8];
            limbs.copy_from_slice(&l[..8]);
            let wide = Wide {
                limbs,
                len,
                negative: w.negative,
            };
            write_wide(dst, order, &wide);
            return;
        }
        // Positions `len - 32..len`: the limbs shifted down by `8 (len - 32)` bits, 264 to 512.
        let k = 8 * (len - 32);
        let (q, b) = (k / 64, (k % 64) as u32);
        let limb = |i: usize| if i < 12 { l[i] } else { 0 };
        let mut t = [0u64; 4];
        for (i, v) in t.iter_mut().enumerate() {
            *v = if b == 0 {
                limb(i + q)
            } else {
                (limb(i + q) >> b) | (limb(i + q + 1) << (64 - b))
            };
        }
        let u = |a: u64, b: u64| u128::from(a) | (u128::from(b) << 64);
        let low = [u(l[0], l[1]), u(l[2], l[3]), u(l[4], l[5]), u(l[6], l[7])];
        let (t01, t23) = (u(t[0], t[1]), u(t[2], t[3]));
        if order == ByteOrder::BigEndian {
            for (i, v) in low.iter().enumerate() {
                write(dst.add(len - 16 * i - 16), v.to_be_bytes());
            }
            write(dst.add(16), t01.to_be_bytes());
            write(dst, t23.to_be_bytes());
        } else {
            for (i, v) in low.iter().enumerate() {
                write(dst.add(16 * i), v.to_le_bytes());
            }
            write(dst.add(len - 32), t01.to_le_bytes());
            write(dst.add(len - 16), t23.to_le_bytes());
        }
    }
}

/// Writes the product `w` as the whole value of `v`, in its vector when it has room, in a new one
/// otherwise.
#[inline(always)]
fn put_widest(v: &mut Vec<u8>, order: ByteOrder, w: &Widest) {
    if v.capacity() < w.len {
        *v = Vec::with_capacity(w.len);
    }
    // SAFETY: the vector has room for the `w.len` bytes, all of which are written before the
    // length is set.
    unsafe {
        write_widest(v.as_mut_ptr(), order, w);
        v.set_len(w.len);
    }
}

// The general path: operands of more than 16 bytes, out of line, so that the small path saves no
// registers for it.

/// An operand of the general path: its vector, its byte order and its significant length.
#[derive(Clone, Copy)]
pub(super) struct Operand {
    pub(super) ptr: *const u8,
    pub(super) len: usize,
    pub(super) big_endian: bool,
    pub(super) significant: usize,
}

impl Operand {
    pub(super) fn of(n: &Number) -> Self {
        let big_endian = n.byte_order == ByteOrder::BigEndian;
        let significant = if big_endian {
            significant_len::<true>(&n.value)
        } else {
            significant_len::<false>(&n.value)
        };
        Operand {
            ptr: n.value.as_ptr(),
            len: n.value.len(),
            big_endian,
            significant,
        }
    }
}

/// `a * b` for borrowed operands, of which at least one has more than 16 bytes.
#[inline(never)]
fn ref_times_ref(a: &Number, b: &Number) -> Number {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: both vectors hold at most 32 bytes.
        let w = unsafe { wide_product(a, b) };
        let mut value = Vec::new();
        put_wide(&mut value, order, &w);
        return Number {
            value,
            negative: w.negative,
            byte_order: order,
        };
    }
    if widest(a, b) {
        // SAFETY: both vectors hold at most 48 bytes.
        let w = unsafe { widest_product(a, b) };
        let mut value = Vec::new();
        put_widest(&mut value, order, &w);
        return Number {
            value,
            negative: w.negative,
            byte_order: order,
        };
    }
    let (x, y) = (Operand::of(a), Operand::of(b));
    let negative = a.negative != b.negative;
    let square = ptr::eq(a, b);
    // SAFETY: the operands are those of `a` and `b`, which are borrowed until the product is
    // computed.
    unsafe {
        product(x, y, square, negative, |p| {
            let mut v = Vec::with_capacity(p.len);
            p.write(v.as_mut_ptr(), order);
            v.set_len(p.len);
            Number {
                value: v,
                negative: p.negative,
                byte_order: order,
            }
        })
    }
}

/// `a * b` for a borrowed `a` and an owned `b`, of which at least one has more than 16 bytes.
/// The product takes the vector of `b` when it is built there, leaving `b` an empty one; the
/// caller drops `b` either way.
#[inline(never)]
fn ref_times_value(a: &Number, b: &mut Number) -> Number {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: both vectors hold at most 32 bytes.
        let w = unsafe { wide_product(a, b) };
        put_wide(&mut b.value, order, &w);
        return Number {
            value: mem::take(&mut b.value),
            negative: w.negative,
            byte_order: order,
        };
    }
    if widest(a, b) {
        // SAFETY: both vectors hold at most 48 bytes.
        let w = unsafe { widest_product(a, b) };
        put_widest(&mut b.value, order, &w);
        return Number {
            value: mem::take(&mut b.value),
            negative: w.negative,
            byte_order: order,
        };
    }
    let (x, y) = (Operand::of(a), Operand::of(b));
    let negative = a.negative != b.negative;
    // SAFETY: the operands are those of `a` and `b`; `b`'s vector is written only once both are
    // read into limbs.
    unsafe {
        product(x, y, false, negative, |p| {
            p.put_in(&mut b.value, order);
            Number {
                value: mem::take(&mut b.value),
                negative: p.negative,
                byte_order: order,
            }
        })
    }
}

/// `a *= b` for a borrowed `b`, where at least one of them has more than 16 bytes.
#[inline(never)]
fn mul_assign_ref(a: &mut Number, b: &Number) {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: both vectors hold at most 32 bytes.
        let w = unsafe { wide_product(a, b) };
        put_wide(&mut a.value, order, &w);
        a.negative = w.negative;
        return;
    }
    if widest(a, b) {
        // SAFETY: both vectors hold at most 48 bytes.
        let w = unsafe { widest_product(a, b) };
        put_widest(&mut a.value, order, &w);
        a.negative = w.negative;
        return;
    }
    let (x, y) = (Operand::of(a), Operand::of(b));
    let negative = a.negative != b.negative;
    // SAFETY: the operands are those of `a` and `b`; `a`'s vector is written only once both are
    // read into limbs.
    unsafe {
        product(x, y, false, negative, |p| {
            p.put_in(&mut a.value, order);
            a.negative = p.negative;
        });
    }
}

/// `a *= b` for an owned `b`, where at least one of them has more than 16 bytes. When the
/// product is built in the vector of `b`, the vector of `a` goes to `b`; the caller drops `b`
/// either way.
#[inline(never)]
fn mul_assign_value(a: &mut Number, b: &mut Number) {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: both vectors hold at most 32 bytes.
        let w = unsafe { wide_product(a, b) };
        if a.value.capacity() < w.len && b.value.capacity() >= w.len {
            mem::swap(&mut a.value, &mut b.value);
        }
        put_wide(&mut a.value, order, &w);
        a.negative = w.negative;
        return;
    }
    if widest(a, b) {
        // SAFETY: both vectors hold at most 48 bytes.
        let w = unsafe { widest_product(a, b) };
        if a.value.capacity() < w.len && b.value.capacity() >= w.len {
            mem::swap(&mut a.value, &mut b.value);
        }
        put_widest(&mut a.value, order, &w);
        a.negative = w.negative;
        return;
    }
    let (x, y) = (Operand::of(a), Operand::of(b));
    let negative = a.negative != b.negative;
    // SAFETY: as in `mul_assign_ref`.
    unsafe {
        product(x, y, false, negative, |p| {
            if a.value.capacity() < p.len && b.value.capacity() >= p.len {
                mem::swap(&mut a.value, &mut b.value);
            }
            p.put_in(&mut a.value, order);
            a.negative = p.negative;
        });
    }
}

/// A product computed in limbs, before it is written out.
pub(super) struct Product {
    /// The limbs of the magnitude, the least significant first: `limbs` of them, or none for
    /// zero.
    pub(super) r: *const u64,
    /// The length of the product in the shortest form: one byte for zero.
    pub(super) len: usize,
    /// Whether the product is negative; never for zero.
    pub(super) negative: bool,
}

impl Product {
    /// Writes the `len` bytes of the product at `dst`, in the layout of `order`.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writing `self.len` bytes, apart from the limbs.
    pub(super) unsafe fn write(&self, dst: *mut u8, order: ByteOrder) {
        // SAFETY: as the caller guarantees; the limbs hold at least `len` bytes.
        unsafe {
            if self.r.is_null() {
                *dst = 0;
            } else if self.len < 8 {
                // Operands with zero bytes above their most significant ones can have a short
                // product.
                write_small(dst, order, *self.r, 0, self.len);
            } else if order == ByteOrder::BigEndian {
                write_limbs::<true>(dst, self.r, self.len);
            } else {
                write_limbs::<false>(dst, self.r, self.len);
            }
        }
    }

    /// Writes the product as the whole value of `v`, in its vector when it has room, and
    /// otherwise in one new vector of exactly its length, which replaces it.
    ///
    /// # Safety
    ///
    /// As for [`Product::write`]; the bytes of `v` are not needed any more.
    #[inline(always)]
    pub(super) unsafe fn put_in(&self, v: &mut Vec<u8>, order: ByteOrder) {
        if v.capacity() < self.len {
            *v = Vec::with_capacity(self.len);
        }
        // SAFETY: the vector has room for the `len` bytes, all of which are written before the
        // length is set.
        unsafe {
            self.write(v.as_mut_ptr(), order);
            v.set_len(self.len);
        }
    }
}

/// The limbs of both operands together up to which a product is computed in a buffer on the stack
/// of the operator's own function, inline: below Karatsuba's method, so that it needs no scratch.
const MEDIUM_LIMBS: usize = 16;

/// The limbs of working memory on the stack for longer products: those whose operands and scratch
/// fit take no allocation besides the vector of the result.
pub(super) const STACK_LIMBS: usize = 1024;

/// Two limbs of working memory, aligned to 16 bytes: the kernels load and store limbs in pairs,
/// and pairs that straddle 16 bytes cost them up to a few percent (see [`compute`]).
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Pair([MaybeUninit<u64>; 2]);

/// An uninitialised pair of limbs.
const PAIR: Pair = Pair([MaybeUninit::uninit(); 2]);

/// `n` rounded up to an even number of limbs, so that a buffer after `n` limbs of an aligned one
/// is aligned too.
#[inline(always)]
const fn even(n: usize) -> usize {
    (n + 1) & !1
}

/// Computes the product of the operands `x` and `y` (with `square`, of `x` by itself) and hands
/// it to `place`, which writes it where it belongs; returns what `place` returns.
///
/// # Safety
///
/// The operands must describe vectors that stay valid for reads until `place` is called, and
/// `y` must be `x` when `square` is set.
#[inline(always)]
unsafe fn product<R>(
    x: Operand,
    y: Operand,
    square: bool,
    negative: bool,
    place: impl FnOnce(&Product) -> R,
) -> R {
    let (la, lb) = (x.significant, y.significant);
    if la == 0 || lb == 0 {
        return place(&Product {
            r: ptr::null(),
            len: 1,
            negative: false,
        });
    }
    let (nx, ny) = (la.div_ceil(8), lb.div_ceil(8));
    // SAFETY: each buffer holds the limbs that `compute` uses: the operands, the product and, for
    // the longer products, the scratch.
    unsafe {
        if nx + ny <= MEDIUM_LIMBS {
            // The operands, each in an even number of limbs, and the product.
            let mut s = [PAIR; MEDIUM_LIMBS + 1];
            return compute(s.as_mut_ptr().cast(), x, y, square, negative, place);
        }
        let (long, short) = if nx >= ny { (nx, ny) } else { (ny, nx) };
        let itch = if square {
            limbs::sqr_itch(nx)
        } else {
            limbs::mul_itch(long, short)
        };
        let operands = if square {
            even(nx)
        } else {
            even(nx) + even(ny)
        };
        let need = operands + even(nx + ny) + itch;
        let compute = |s: *mut MaybeUninit<u64>| compute(s.cast(), x, y, square, negative, place);
        if need <= STACK_LIMBS {
            on_stack(compute)
        } else {
            on_heap(need, compute)
        }
    }
}

/// [`product`] in the working memory at `s`: the operands read into limbs, then the product of
/// the limbs after them, its length in the shortest form, and `place`.
///
/// The limbs are placed for the kernels, which load and store them in pairs, and `s` is aligned
/// to 16 bytes as the buffers of [`product`] are. The kernels take an odd limb first and the pairs
/// after it, so an operand of an odd number of limbs starts 8 bytes past 16, and its pairs on 16;
/// the product and the scratch start on 16 bytes. Pairs that straddle 16 bytes, just after
/// `to_limbs` has stored the operand a limb at a time, measured on an Apple M4 Max: 913 cycles
/// for `a *= &b` of 17 limbs each against 389, as the loads waited for the stores; with the
/// scratch 8 bytes off, Toom-Cook products of 1024 and 2048 limbs took 0.6 and 1.3 % longer.
///
/// # Safety
///
/// As for [`product`], with the operands not zero, and `s` valid for the operands' limbs, the
/// product's and the scratch the product needs, each rounded up to an even number.
#[inline(always)]
unsafe fn compute<R>(
    s: *mut u64,
    x: Operand,
    y: Operand,
    square: bool,
    negative: bool,
    place: impl FnOnce(&Product) -> R,
) -> R {
    let (la, lb) = (x.significant, y.significant);
    let (nx, ny) = (la.div_ceil(8), lb.div_ceil(8));
    // SAFETY: as the caller guarantees.
    unsafe {
        let (a, b) = (s.add(nx & 1), s.add(even(nx) + (ny & 1)));
        to_limbs(x, a);
        let operands = if square {
            even(nx)
        } else {
            even(nx) + even(ny)
        };
        let r = s.add(operands);
        let t = r.add(even(nx + ny));
        // Short operands go straight to the kernels, without the dispatch of the longer ones.
        let medium = nx + ny <= MEDIUM_LIMBS;
        if square {
            if medium {
                kernels::sqr_basecase(r, a, nx);
            } else {
                limbs::sqr(r, a, nx, t);
            }
        } else {
            to_limbs(y, b);
            let ((p, np), (q, nq)) = if nx >= ny {
                ((a, nx), (b, ny))
            } else {
                ((b, ny), (a, nx))
            };
            if medium {
                kernels::mul_basecase(r, p, np, q, nq);
            } else {
                limbs::mul(r, p, np, q, nq, t);
            }
        }
        // The product of magnitudes of `la` and `lb` significant bytes has `la + lb` of them, or
        // one fewer.
        let top = la + lb - 1;
        let byte = (*r.add(top / 8) >> (8 * (top % 8))) as u8;
        let len = if byte == 0 { top } else { top + 1 };
        place(&Product { r, len, negative })
    }
}

/// Limbs of the buffer of [`on_stack`] past the [`STACK_LIMBS`] it hands out, 2 KiB.
const STACK_PAD: usize = 256;

/// Runs `f` on [`STACK_LIMBS`] limbs of working memory on the stack, aligned to 16 bytes. Out of
/// line, so that the small path's frame does not hold them.
///
/// The buffer is the bottom of the frame, and the working memory its first limbs. A buffer of
/// exactly 8 KiB put them a multiple of 4 KiB below the caller's frame, and loads whose addresses
/// agree with those of earlier stores in their low 12 bits wait for those stores: measured on an
/// Apple M4 Max, `a *= &b` of 129 to 136 bytes took up to 2.5 times as long, some sizes from 65
/// bytes on 1.3 to 2.3 times, depending on the build. 2 KiB more of buffer keep them apart.
#[inline(never)]
pub(super) fn on_stack<R>(f: impl FnOnce(*mut MaybeUninit<u64>) -> R) -> R {
    let mut s = [PAIR; (STACK_LIMBS + STACK_PAD) / 2];
    f(s.as_mut_ptr().cast())
}

/// Runs `f` on `need` limbs of working memory on the heap, aligned to 16 bytes, freed on return.
#[cold]
#[inline(never)]
pub(super) fn on_heap<R>(need: usize, f: impl FnOnce(*mut MaybeUninit<u64>) -> R) -> R {
    let mut s = Vec::<Pair>::with_capacity(need.div_ceil(2));
    f(s.as_mut_ptr().cast())
}

/// Reads the significant bytes of the operand `x` into `x.significant.div_ceil(8)` limbs at
/// `out`, the least significant first.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `out` for writing its limbs.
#[inline(always)]
pub(super) unsafe fn to_limbs(x: Operand, out: *mut u64) {
    let l = x.significant;
    let (full, rem) = (l / 8, l % 8);
    // SAFETY: every read lies inside the significant bytes: big-endian the last `l` of the
    // vector, little-endian the first `l`.
    unsafe {
        // The `rem` bytes above the whole limbs: from 8 bytes up, the eight that end at the top,
        // shifted down to them; below, as the addition reads a short magnitude.
        let shift = 8 * (8 - rem) as u32;
        if x.big_endian {
            let end = x.ptr.add(x.len);
            for i in 0..full {
                *out.add(i) = u64::from_be_bytes(read(end.sub(8 * i + 8)));
            }
            if rem > 0 {
                *out.add(full) = if l >= 8 {
                    u64::from_be_bytes(read(end.sub(l))) >> shift
                } else {
                    read_small(
                        core::slice::from_raw_parts(end.sub(l), rem),
                        ByteOrder::BigEndian,
                    ) as u64
                };
            }
        } else {
            for i in 0..full {
                *out.add(i) = u64::from_le_bytes(read(x.ptr.add(8 * i)));
            }
            if rem > 0 {
                *out.add(full) = if l >= 8 {
                    u64::from_le_bytes(read(x.ptr.add(l - 8))) >> shift
                } else {
                    read_small(
                        core::slice::from_raw_parts(x.ptr, rem),
                        ByteOrder::LittleEndian,
                    ) as u64
                };
            }
        }
    }
}

/// Writes the `len` least significant bytes of the limbs at `r` to the `len` bytes at `dst`, in
/// the layout `BE`: the whole limbs, then the bytes above them as the eight bytes that end at the
/// top, which agree with the limb below where they overlap it.
///
/// # Safety
///
/// `len >= 8`; the limbs must hold `len` bytes, and `dst` be valid for writing them.
#[inline(always)]
unsafe fn write_limbs<const BE: bool>(dst: *mut u8, r: *const u64, len: usize) {
    let (full, rem) = (len / 8, len % 8);
    // SAFETY: as the caller guarantees: limb `i` goes to `[8i, 8i + 8)`, big-endian to the eight
    // bytes that end at `len - 8i`, and the `rem` bytes of the top limb below the rest.
    unsafe {
        for i in 0..full {
            let v = *r.add(i);
            if BE {
                write(dst.add(len - 8 * i - 8), v.to_be_bytes());
            } else {
                write(dst.add(8 * i), v.to_le_bytes());
            }
        }
        if rem > 0 {
            // Positions `len - 8..len`: the top limb's `rem` bytes over the limb below.
            let k = 8 * rem as u32;
            let v = (*r.add(full - 1) >> k) | (*r.add(full) << (64 - k));
            if BE {
                write(dst, v.to_be_bytes());
            } else {
                write(dst.add(len - 8), v.to_le_bytes());
            }
        }
    }
}
