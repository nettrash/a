//! Exact division and divisibility: [`Number::div_exact`] and [`Number::is_multiple_of`].
//!
//! When the divisor is known to divide the dividend, the quotient can be computed from the low
//! end, by Hensel's method (see [`bdiv_q`]): each limb of the quotient is exact, with no estimate
//! to correct, and the limbs of the partial remainder above the quotient's are never needed, so a
//! quotient as long as the divisor takes about half the work of a division.

use super::super::Number;
use super::super::add::{read_small, small};
use super::super::mul::{Operand, Product, STACK_LIMBS, on_heap, on_stack};
use super::limbs::{DC_DIV, bdiv_q, binvert_limb, rshift};
use super::{
    TRUNCATED, Want, div_u128, divide, operands, register_division, result, small_division,
    small_number, wide, wide_number,
};
use core::mem::MaybeUninit;

/// From this many limb products of Hensel's method, an exact division by a divisor long enough
/// for divide and conquer is left to the ordinary division, which that makes faster than the
/// quadratic method: measured on an Apple M4 Max, a quotient of 512 limbs by a divisor of 512
/// (131,072 products) took 156,842 cycles so against 173,462, one of 768 by 768 356,035 against
/// 339,764, and one of 2048 by 256 605,302 against 454,124.
const HENSEL_LIMIT: usize = 1 << 18;

/// Whether an exact quotient of `qn` limbs by a divisor of `dn`, at least two, is computed by
/// Hensel's method: below the divisors that divide and conquer divides, always, as both methods
/// are quadratic there and Hensel's does half the work; beyond, up to [`HENSEL_LIMIT`] limb
/// products: `qn m - m^2 / 2` for `m = min(qn, dn)`, as the divisor's limbs above the quotient's
/// are never needed.
fn by_hensel(qn: usize, dn: usize) -> bool {
    let m = qn.min(dn);
    dn < DC_DIV || qn * m - m * m / 2 < HENSEL_LIMIT
}

impl Number {
    /// Returns `self / rhs` for a `rhs` that divides `self` exactly, faster than `/` does: in the
    /// byte order of `self`, in the shortest form, in one new vector of exactly its length.
    ///
    /// The quotient is computed from the low end, by Hensel's method, whose limbs need no
    /// estimates and whose partial remainders need no limbs above the quotient's: for a quotient
    /// as long as the divisor, about half the work of a division. When `rhs` does not divide
    /// `self`, the result is a number, but not a meaningful one; [`is_multiple_of`] tells.
    ///
    /// [`is_multiple_of`]: Self::is_multiple_of
    ///
    /// # Panics
    ///
    /// When `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x0123_4567_89AB_CDEF * -7 = -0x07F6_E5D4_C3B2_A189, divided back by 7.
    /// let a = Number::new_from_i64(-0x07F6_E5D4_C3B2_A189, ByteOrder::BigEndian);
    /// let q = a.div_exact(&Number::new_from_i32(7, ByteOrder::LittleEndian));
    /// assert_eq!(q.value, [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]);
    /// assert!(q.is_negative());
    /// ```
    #[track_caller]
    pub fn div_exact(&self, rhs: &Number) -> Number {
        if small(self, rhs) {
            let (q, _) = small_division::<TRUNCATED, false>(self, rhs);
            return small_number(self.byte_order, q);
        }
        exact_quotient(self, rhs)
    }

    /// Returns whether `self` is a multiple of `rhs`: `self = rhs q` for an integer `q`, which is
    /// to say that the remainder of `self` by `rhs` is zero. As for `u32::is_multiple_of`, zero is
    /// a multiple of everything, and only zero is a multiple of zero. Signs do not count.
    ///
    /// Allocates nothing; a dividend shorter than the divisor, or with fewer zero bits at the low
    /// end, is told apart without dividing.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i64(-1_000_000_000_007 * 13, ByteOrder::BigEndian);
    /// assert!(a.is_multiple_of(&Number::new_from_u8(13, ByteOrder::LittleEndian)));
    /// assert!(!a.is_multiple_of(&Number::new_from_u8(12, ByteOrder::LittleEndian)));
    ///
    /// let zero = Number::new_from_u8(0, ByteOrder::BigEndian);
    /// assert!(zero.is_multiple_of(&zero));
    /// assert!(!a.is_multiple_of(&zero));
    /// ```
    pub fn is_multiple_of(&self, rhs: &Number) -> bool {
        if small(self, rhs) {
            // SAFETY: both vectors hold at most 16 bytes.
            let (x, y) = unsafe {
                (
                    read_small(&self.value, self.byte_order),
                    read_small(&rhs.value, rhs.byte_order),
                )
            };
            return if y == 0 {
                x == 0
            } else {
                div_u128(x, y).1 == 0
            };
        }
        multiple(self, rhs)
    }
}

/// `a.div_exact(b)` for operands of which at least one has more than 16 bytes.
#[inline(never)]
#[track_caller]
fn exact_quotient(a: &Number, b: &Number) -> Number {
    let order = a.byte_order;
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 16 bytes.
        let (q, _) = unsafe { register_division::<TRUNCATED, false>(a, b) };
        return wide_number(order, &q);
    }
    let (x, y) = operands::<false>(a, b);
    let (lx, ny) = (x.significant.div_ceil(8), y.significant.div_ceil(8));
    let negative = (a.negative && x.significant != 0) != b.negative;
    let place = |q: &Product| {
        let mut v = Vec::with_capacity(q.len);
        // SAFETY: the new vector has room for the `len` bytes, all of which are written before
        // the length is set.
        unsafe {
            q.write(v.as_mut_ptr(), order);
            v.set_len(q.len);
        }
        Number {
            value: v,
            negative: q.negative,
            byte_order: order,
        }
    };
    if ny == 1 || lx < ny || !by_hensel(lx - ny + 1, ny) {
        // A divisor of one limb folds the limbs in about as fast as Hensel's method goes, and a
        // long one divides faster by divide and conquer.
        // SAFETY: the operands are those of `a` and `b`, borrowed until the quotient is written.
        return unsafe {
            divide::<TRUNCATED, _>(x, y, a.negative, b.negative, Want::Quotient, |q, _| {
                place(q)
            })
        };
    }
    // SAFETY: as above.
    unsafe { hensel(x, y, negative, place) }
}

/// The quotient of the operand `x` by the operand `y` of at least two limbs, assumed exact, by
/// Hensel's method, handed to `place`. Both are shifted right by the zero bits at the low end of
/// `y`, which `x` has too when the division is exact, so that the divisor is odd; only the
/// quotient's `qn` low limbs of the dividend, and as many of the divisor, are read.
///
/// # Safety
///
/// The operands must describe vectors that stay valid for reads until `place` is called, `y` of
/// at least two limbs and `x` at least as long.
#[inline(always)]
unsafe fn hensel<R>(
    x: Operand,
    y: Operand,
    negative: bool,
    place: impl FnOnce(&Product) -> R,
) -> R {
    let (lx, ny) = (x.significant.div_ceil(8), y.significant.div_ceil(8));
    let qn = lx - ny + 1;
    // SAFETY: as the caller guarantees.
    let zeros = unsafe { trailing_zeros(y) };
    let (skip, bits) = (zeros / 64, (zeros % 64) as u32);
    // The divisor's limbs after the shift, of which the quotient's `qn` are needed.
    let dn = (8 * y.significant - zeros).div_ceil(64).min(qn);
    let need = 1 + (qn + 1) + (dn + 1) + qn;
    let compute = |s: *mut MaybeUninit<u64>| {
        let s: *mut u64 = s.cast();
        // SAFETY: the working memory, on 16 bytes, holds a limb to place the dividend, its
        // `qn + 1` limbs, the divisor's `dn + 1` and the quotient's `qn`.
        unsafe {
            // Step `i` subtracts a row of `qn - i` limbs from the dividend's limbs from `i`, of
            // which the kernel takes an odd one first and then pairs: from limb `qn` modulo 2,
            // whatever `i`, which this puts on 16 bytes. Off by 8, a quotient of 257 limbs by a
            // divisor of 256 took 48,100 cycles against 38,000, on an Apple M4 Max.
            let n = s.add(qn & 1);
            let d = n.add(qn + 1);
            let q = d.add(dn + 1);
            limbs_from(x, skip, qn + 1, n);
            limbs_from(y, skip, dn + 1, d);
            if bits != 0 {
                rshift(n, qn + 1, bits);
                rshift(d, dn + 1, bits);
            }
            bdiv_q(q, n, qn, d, dn, binvert_limb(*d));
            place(&result(q, qn, negative))
        }
    };
    if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// `a.is_multiple_of(b)` for operands of which at least one has more than 16 bytes.
#[inline(never)]
fn multiple(a: &Number, b: &Number) -> bool {
    let (x, y) = (Operand::of(a), Operand::of(b));
    if y.significant == 0 || x.significant == 0 {
        return x.significant == 0;
    }
    if x.significant < y.significant {
        return false;
    }
    // SAFETY: both operands are not zero, and borrowed.
    if unsafe { trailing_zeros(x) < trailing_zeros(y) } {
        return false;
    }
    if wide(a, b) {
        // SAFETY: the vectors hold at most 64 and 16 bytes, and the divisor is not zero.
        let (_, r) = unsafe { register_division::<TRUNCATED, true>(a, b) };
        return r.limbs == [0; 8];
    }
    // SAFETY: the operands are those of `a` and `b`, borrowed, and the divisor is not zero.
    unsafe { divide::<TRUNCATED, _>(x, y, false, false, Want::Remainder, |_, r| r.r.is_null()) }
}

/// The number of zero bits at the low end of the magnitude of the operand `x`, not zero.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and its magnitude not zero.
unsafe fn trailing_zeros(x: Operand) -> usize {
    let mut i = 0;
    loop {
        // SAFETY: the magnitude is not zero, so a limb that is not zero comes before its end.
        let l = unsafe { super::limb(x, i) };
        if l != 0 {
            return 64 * i + l.trailing_zeros() as usize;
        }
        i += 1;
    }
}

/// Reads limbs `from..from + n` of the magnitude of the operand `x`, zero above it, into the `n`
/// limbs at `out`.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `out` for writing `n` limbs.
#[inline(always)]
unsafe fn limbs_from(x: Operand, from: usize, n: usize, out: *mut u64) {
    // SAFETY: as the caller guarantees.
    unsafe { super::shifted_limbs(x, from, from + n, 0, out) }
}
