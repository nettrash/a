//! Integer roots of [`Number`]s: [`Number::isqrt`], [`Number::isqrt_rem`] and
//! [`Number::nth_root`].
//!
//! A root is truncated toward zero, as GMP's `mpz_sqrt` and `mpz_root` and Rust's `isqrt` truncate
//! it: the square root of a number is the largest integer whose square is not above it, and the
//! `n`-th root of a negative number for an odd `n` is the root of its magnitude, negated. A zero
//! magnitude is zero whatever `negative` says, and zero bytes above the most significant one count
//! for nothing.
//!
//! Magnitudes whose vectors hold at most 16 bytes are rooted in registers: as a `u128`, from a
//! floating-point estimate made exact (see [`sqrt::sqrt_rem_u128`] and [`small_root`]). Longer
//! ones are read into limbs of eight bytes, native `u64`s, the least significant first, in
//! working memory on the stack, or on the heap for the longest: square roots by Zimmermann's
//! Karatsuba square root (see [`sqrt`]), other roots by Newton's method at doubling precision
//! (see [`nth`]). The results are written in the byte order of `self`, in the shortest form.

mod nth;
mod sqrt;

use super::add::{SMALL, read_small, write_small};
use super::mul::{Operand, Product, STACK_LIMBS, WIDE, on_heap, on_stack, read_wide, to_limbs};
use super::{ByteOrder, Number};
use core::mem::MaybeUninit;

pub(super) use nth::quotient;
pub(super) use sqrt::sqrt_rem_u128;

impl Number {
    /// Returns the square root of `self` rounded down, `floor(sqrt(self))`: the largest integer
    /// whose square is not above `self`, in the byte order of `self`, in the shortest form, in one
    /// new vector.
    ///
    /// Up to 16 bytes the root is computed in registers; longer magnitudes by Zimmermann's
    /// Karatsuba square root, in the time of a product of their limbs, up to a constant.
    ///
    /// # Panics
    ///
    /// When `self` is negative, with the message of `i64::isqrt`; a negative zero is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_u64(1_000_000_000_000, ByteOrder::BigEndian);
    /// assert_eq!(n.isqrt().value, Number::new_from_u32(1_000_000, ByteOrder::BigEndian).value);
    ///
    /// // 99 is not a square: its root is rounded down.
    /// assert_eq!(Number::new_from_u8(99, ByteOrder::LittleEndian).isqrt().value, [9]);
    /// ```
    #[track_caller]
    pub fn isqrt(&self) -> Number {
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let x = unsafe { read_small(&self.value, self.byte_order) };
            if self.negative && x != 0 {
                negative_square_root();
            }
            let (s, _) = sqrt_rem_u128(x);
            return small_number(self.byte_order, u128::from(s), false);
        }
        if self.value.len() <= WIDE {
            return wide_square_root::<false>(self).0;
        }
        long_square_root::<false>(self).0
    }

    /// Returns the square root of `self` rounded down and the remainder: `(s, r)` with
    /// `s = floor(sqrt(self))` and `r = self - s^2`, which is at most `2s`, both in the byte order
    /// of `self`, in the shortest form, in one new vector each.
    ///
    /// # Panics
    ///
    /// When `self` is negative, with the message of `i64::isqrt`; a negative zero is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 1000 = 31^2 + 39.
    /// let (s, r) = Number::new_from_u16(1000, ByteOrder::BigEndian).isqrt_rem();
    /// assert_eq!(s.value, [31]);
    /// assert_eq!(r.value, [39]);
    /// ```
    #[track_caller]
    pub fn isqrt_rem(&self) -> (Number, Number) {
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let x = unsafe { read_small(&self.value, self.byte_order) };
            if self.negative && x != 0 {
                negative_square_root();
            }
            let (s, r) = sqrt_rem_u128(x);
            let order = self.byte_order;
            return (
                small_number(order, u128::from(s), false),
                small_number(order, r, false),
            );
        }
        if self.value.len() <= WIDE {
            return wide_square_root::<true>(self);
        }
        long_square_root::<true>(self)
    }

    /// Returns the `n`-th root of `self` truncated toward zero: for a `self` that is not negative,
    /// the largest integer whose `n`-th power is not above `self`, and for a negative `self` and an
    /// odd `n`, the root of its magnitude negated, as GMP's `mpz_root` and num-integer's
    /// `Roots::nth_root` give it. The root is in the byte order of `self`, in the shortest form,
    /// in one new vector.
    ///
    /// `n = 1` gives `self` and `n = 2` its square root, as [`isqrt`](Self::isqrt) does. Up to 16
    /// bytes the root is computed in registers; for longer magnitudes, by Newton's method at
    /// doubling precision, as GMP's `mpn_rootrem` does.
    ///
    /// # Panics
    ///
    /// When `n` is 0, with the message `root degree n must be at least 1`, and when `self` is
    /// negative and `n` even, with the message `even roots of a negative are imaginary`, as
    /// num-integer's `Roots::nth_root` panics; a negative zero is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 10^18 = (10^6)^3, and the cube root of -999,999 is -99.
    /// let n = Number::new_from_u64(1_000_000_000_000_000_000, ByteOrder::BigEndian);
    /// assert_eq!(n.nth_root(3).value, Number::new_from_u32(1_000_000, ByteOrder::BigEndian).value);
    /// let m = Number::new_from_i32(-999_999, ByteOrder::LittleEndian);
    /// let root = m.nth_root(3);
    /// assert_eq!(root.value, [99]);
    /// assert!(root.is_negative());
    /// ```
    #[track_caller]
    pub fn nth_root(&self, n: u32) -> Number {
        if n == 0 {
            zero_degree();
        }
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let x = unsafe { read_small(&self.value, self.byte_order) };
            let negative = self.negative && x != 0;
            if negative && n.is_multiple_of(2) {
                imaginary_root();
            }
            return small_number(self.byte_order, small_root(x, n), negative);
        }
        long_root(self, n)
    }
}

/// Panics as `i64::isqrt` does for a negative argument.
#[cold]
#[inline(never)]
#[track_caller]
fn negative_square_root() -> ! {
    panic!("argument of integer square root cannot be negative")
}

/// Panics as num-integer's `Roots::nth_root` does for a root of degree 0.
#[cold]
#[inline(never)]
#[track_caller]
fn zero_degree() -> ! {
    panic!("root degree n must be at least 1")
}

/// Panics as num-integer's `Roots::nth_root` does for an even root of a negative number.
#[cold]
#[inline(never)]
#[track_caller]
fn imaginary_root() -> ! {
    panic!("even roots of a negative are imaginary")
}

// Small numbers

/// The magnitude `v` of at most 16 bytes, its sign `negative` (dropped for zero) and the byte order
/// `order` as a number in one new vector of exactly its length.
pub(super) fn small_number(order: ByteOrder, v: u128, negative: bool) -> Number {
    let len = 16 - (v | 1).leading_zeros() as usize / 8;
    let mut value = Vec::with_capacity(len);
    // SAFETY: the new vector has room for the `len` bytes, from 1 to 16, all of which are written
    // before the length is set.
    unsafe {
        write_small(value.as_mut_ptr(), order, v as u64, (v >> 64) as u64, len);
        value.set_len(len);
    }
    Number {
        value,
        negative: negative && v != 0,
        byte_order: order,
    }
}

/// The `n`-th root of `x`, truncated: the largest `s` with `s^n <= x`.
///
/// For `n >= 3` the root is below `2^43`. Its estimate in double precision, `x^(1/n)` (the cube
/// root for `n = 3`, which the system computes faster), is off by far less than one; the root is
/// its floor, or one less or one more, which two powers computed exactly side by side tell, and
/// steps of one in the rare other cases.
pub(super) fn small_root(x: u128, n: u32) -> u128 {
    match n {
        1 => return x,
        2 => return u128::from(sqrt_rem_u128(x).0),
        _ => {}
    }
    let bits = 128 - x.leading_zeros();
    if bits <= n {
        // `1 <= x < 2^n`: the root is 1, or 0 for 0.
        return u128::from(x != 0);
    }
    let estimate = if bits.div_ceil(n) <= ROUGH_BITS {
        let top = (x << x.leading_zeros() >> 64) as u64;
        rough_root(top, u64::from(bits), n)
    } else {
        let xf = ((x >> 64) as u64) as f64 * 18_446_744_073_709_551_616.0 + (x as u64) as f64;
        if n == 3 {
            xf.cbrt()
        } else {
            xf.powf(1.0 / f64::from(n))
        }
    };
    // At least 1, as `x >= 2^n`, and below `2^43`.
    let s = u128::from((estimate as u64).max(1));
    let (low, high) = (power_at_most(s, n, x), power_at_most(s + 1, n, x));
    if low && !high {
        return s;
    }
    small_root_steps(x, n, s, low)
}

/// Whether `s^n <= x`, for `s` below `2^44`.
#[inline(always)]
fn power_at_most(s: u128, n: u32, x: u128) -> bool {
    power_u128(s, n).is_some_and(|p| p <= x)
}

/// [`small_root`] from an estimate `s` that is not the root: one less while `s^n > x`, when `low`
/// says `s^n <= x` is false, and otherwise one more while `(s + 1)^n <= x`.
#[cold]
#[inline(never)]
fn small_root_steps(x: u128, n: u32, mut s: u128, low: bool) -> u128 {
    let below = |s: u128| power_at_most(s, n, x);
    if !low {
        s -= 1;
        while !below(s) {
            s -= 1;
        }
        return s;
    }
    while below(s + 1) {
        s += 1;
    }
    s
}

/// The roots of at most this many bits are estimated by [`rough_root`].
pub(super) const ROUGH_BITS: u32 = 18;

/// An estimate of `T^(1/n)` for a `T` of `bits` bits whose top 64 are `top`, for a root of at
/// most [`ROUGH_BITS`] bits: `2^(log2(T) / n)` with `log2` and `2^x` from polynomials on `[0, 1)`
/// fitted at Chebyshev points (errors of `2.2 10^-6` and `1.1 10^-7`), so within a part in a
/// million of the root, below 0.3 for a root of 18 bits, in about a quarter of the time of the
/// system's `powf`, whose latency was about 80 cycles on an Apple M4 Max.
pub(super) fn rough_root(top: u64, bits: u64, n: u32) -> f64 {
    // `T = 2^(bits - 1) (1 + f)`.
    let f = (top >> 11) as f64 * (1.0 / 4_503_599_627_370_496.0) - 1.0;
    let horner = |c: &[f64], x: f64| c.iter().fold(0.0, |acc, &c| acc * x + c);
    let log1p = horner(
        &[
            -0.025_123_203_286_102_475,
            0.119_298_237_706_177_92,
            -0.274_623_257_617_108_6,
            0.455_527_088_061_034_7,
            -0.717_557_872_422_149_5,
            1.442_475_314_822_074_8,
            2.123_740_891_505_907_2e-6,
        ],
        f,
    );
    let l = ((bits - 1) as f64 + log1p) / f64::from(n);
    let i = l.floor();
    let exp = horner(
        &[
            0.001_895_107_290_998_152_7,
            0.008_946_214_666_338_551,
            0.055_863_282_659_265_955,
            0.240_140_770_091_823_9,
            0.693_154_620_003_301_9,
            0.999_999_895_763_136,
        ],
        l - i,
    );
    // `2^i` for `0 <= i <= ROUGH_BITS`, from its exponent bits.
    exp * f64::from_bits((i as u64 + 1023) << 52)
}

/// `s^n`, or `None` above `u128::MAX`, by squares and products: unchecked when the bits of `s`
/// show that the power fits, `s^n < 2^(bits n) <= 2^128`, `None` at once when they show it does
/// not, `s^n >= 2^((bits - 1) n) >= 2^128`, and checked at each step in between.
#[inline(always)]
fn power_u128(s: u128, n: u32) -> Option<u128> {
    let bits = u64::from(128 - s.leading_zeros());
    let n64 = u64::from(n);
    if s.is_power_of_two() {
        // `2^((bits - 1) n)`: a shift.
        let shift = (bits - 1) * n64;
        return if shift < 128 { Some(1 << shift) } else { None };
    }
    if bits * n64 <= 128 {
        Some(s.wrapping_pow(n))
    } else if bits.saturating_sub(1) * n64 >= 128 {
        None
    } else {
        s.checked_pow(n)
    }
}

// Numbers of up to 32 bytes

/// The square root of `a` and, with `REM`, the remainder, for a vector of 17 to 32 bytes: in
/// registers, by [`sqrt::sqrt_rem_wide`] for a magnitude of at least `2^128`. Without `REM`, the
/// remainder returned is an empty number.
#[inline(never)]
#[track_caller]
fn wide_square_root<const REM: bool>(a: &Number) -> (Number, Number) {
    let order = a.byte_order;
    // SAFETY: the vector holds at most 32 bytes.
    let x = unsafe { read_wide(&a.value, a.byte_order) };
    if a.negative && x != [0; 4] {
        negative_square_root();
    }
    if x[2] == 0 && x[3] == 0 {
        let (s, r) = sqrt_rem_u128(u128::from(x[1]) << 64 | u128::from(x[0]));
        let rem = if REM {
            small_number(order, r, false)
        } else {
            empty(order)
        };
        return (small_number(order, u128::from(s), false), rem);
    }
    let (s, r) = sqrt::sqrt_rem_wide::<REM>(x);
    let rem = if REM {
        // SAFETY: the array holds the remainder's three limbs.
        unsafe { limbs_number(r.as_ptr(), 3, false, order) }
    } else {
        empty(order)
    };
    (small_number(order, s, false), rem)
}

// Long numbers

/// The square root of `a` and, with `REM`, the remainder, for a vector of more than 16 bytes:
/// in registers up to 16 significant bytes, and otherwise by [`sqrt::sqrtrem`] on its limbs.
/// Without `REM`, the remainder returned is an empty number.
#[inline(never)]
#[track_caller]
fn long_square_root<const REM: bool>(a: &Number) -> (Number, Number) {
    let x = Operand::of(a);
    let order = a.byte_order;
    if a.negative && x.significant != 0 {
        negative_square_root();
    }
    if x.significant <= SMALL {
        // SAFETY: the operand is that of `a`.
        let (s, r) = sqrt_rem_u128(unsafe { small_value(x) });
        let rem = if REM {
            small_number(order, r, false)
        } else {
            empty(order)
        };
        return (small_number(order, u128::from(s), false), rem);
    }
    let an = x.significant.div_ceil(8);
    let tn = an.div_ceil(2);
    // The magnitude, the root and the remainder, each in an even number of limbs, then the working
    // memory of `sqrtrem`.
    let need =
        even(an) + even(tn + 1) + if REM { even(tn + 1) } else { 0 } + sqrt::sqrtrem_itch(an);
    let compute = |w: *mut MaybeUninit<u64>| {
        let w: *mut u64 = w.cast();
        // SAFETY: the working memory holds the `need` limbs counted above; the operand is that of
        // `a`, borrowed until the results are written.
        unsafe {
            let (m, s) = (w, w.add(even(an)));
            let r = s.add(even(tn + 1));
            let t = if REM { r.add(even(tn + 1)) } else { r };
            to_limbs(x, m);
            let rn = sqrt::sqrtrem::<REM>(s, r, m, an, t);
            let root = limbs_number(s, tn, false, order);
            let rem = if REM {
                limbs_number(r, rn, false, order)
            } else {
                empty(order)
            };
            (root, rem)
        }
    };
    if need <= LOCAL_LIMBS {
        let mut local = [PAIR; LOCAL_LIMBS / 2];
        compute(local.as_mut_ptr().cast())
    } else if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// The limbs of working memory up to which a root is computed in a buffer of its function's own
/// frame, instead of the larger one of `on_stack`.
const LOCAL_LIMBS: usize = 64;

/// Two limbs of working memory, aligned to 16 bytes, as the kernels read them in pairs.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Pair([MaybeUninit<u64>; 2]);

/// An uninitialised pair of limbs.
const PAIR: Pair = Pair([MaybeUninit::uninit(); 2]);

/// The `n`-th root of `a` for a vector of more than 16 bytes and `n >= 1`: in registers up to 16
/// significant bytes, 1 for magnitudes of at most `n` bits, and otherwise by [`nth::root`] on its
/// limbs.
#[inline(never)]
#[track_caller]
fn long_root(a: &Number, n: u32) -> Number {
    let x = Operand::of(a);
    let order = a.byte_order;
    let negative = a.negative && x.significant != 0;
    if negative && n.is_multiple_of(2) {
        imaginary_root();
    }
    if x.significant <= SMALL {
        // SAFETY: the operand is that of `a`.
        let v = unsafe { small_value(x) };
        return small_number(order, small_root(v, n), negative);
    }
    if n == 2 {
        // Not negative, or the even root would have panicked.
        return a.isqrt();
    }
    let an = x.significant.div_ceil(8);
    if 8 * x.significant as u64 <= u64::from(n) {
        // A magnitude of at most `n` bits, not zero, is below `2^n`: its root is 1.
        return small_number(order, 1, negative);
    }
    if n == 1 {
        let compute = |w: *mut MaybeUninit<u64>| {
            let w: *mut u64 = w.cast();
            // SAFETY: the working memory holds the operand's limbs; the operand is that of `a`.
            unsafe {
                to_limbs(x, w);
                limbs_number(w, an, negative, order)
            }
        };
        return if an <= STACK_LIMBS {
            on_stack(compute)
        } else {
            on_heap(an, compute)
        };
    }
    // The magnitude, then the working memory of `nth::root`, which starts with the root.
    let need = even(an) + nth::root_itch(an, n);
    let compute = |w: *mut MaybeUninit<u64>| {
        let w: *mut u64 = w.cast();
        // SAFETY: the working memory holds the `need` limbs counted above; the operand is that of
        // `a`, borrowed until the root is written.
        unsafe {
            let (m, r) = (w, w.add(even(an)));
            to_limbs(x, m);
            let sn = nth::root(m, an, n, r);
            limbs_number(r, sn, negative, order)
        }
    };
    if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// The magnitude of the operand `x` of at most 16 significant bytes, as a `u128`.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
unsafe fn small_value(x: Operand) -> u128 {
    let mut l = [0u64; 2];
    // SAFETY: the significant bytes fill at most two limbs.
    unsafe { to_limbs(x, l.as_mut_ptr()) };
    u128::from(l[0]) | (u128::from(l[1]) << 64)
}

/// A number with an empty vector, which allocates nothing: the remainder `long_square_root` does
/// not compute.
fn empty(order: ByteOrder) -> Number {
    Number {
        value: Vec::new(),
        negative: false,
        byte_order: order,
    }
}

/// `n` rounded up to an even number of limbs, so that a buffer after `n` limbs of one that starts
/// on 16 bytes starts on 16 bytes too.
const fn even(n: usize) -> usize {
    (n + 1) & !1
}

/// The `n` limbs at `r` as a number in one new vector of exactly its length, in the byte order
/// `order`, in the shortest form: without the zero limbs at the top, and zero as `[0]`, never
/// negative.
///
/// # Safety
///
/// The limbs must be valid for reads.
pub(super) unsafe fn limbs_number(
    r: *const u64,
    n: usize,
    negative: bool,
    order: ByteOrder,
) -> Number {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut top = n;
        while top > 0 && *r.add(top - 1) == 0 {
            top -= 1;
        }
        let product = if top == 0 {
            Product {
                r: core::ptr::null(),
                len: 1,
                negative: false,
            }
        } else {
            let len = 8 * top - (*r.add(top - 1)).leading_zeros() as usize / 8;
            Product { r, len, negative }
        };
        let mut v = Vec::with_capacity(product.len);
        product.write(v.as_mut_ptr(), order);
        v.set_len(product.len);
        Number {
            value: v,
            negative: product.negative,
            byte_order: order,
        }
    }
}

// Shared limb helpers

/// Sets the `n` limbs at `dst` to the `n` limbs at `src` shifted left by `s < 64` bits, and returns
/// the bits shifted out of the top one.
///
/// # Safety
///
/// The limbs at `src` must be valid for reads and those at `dst` for writes; `dst` may be `src`,
/// or above it.
pub(super) unsafe fn lshift(dst: *mut u64, src: *const u64, n: usize, s: u32) -> u64 {
    // SAFETY: as the caller guarantees; from the top down, each limb of `src` is read before the
    // limb of `dst` at or above it is written.
    unsafe {
        if s == 0 {
            core::ptr::copy(src, dst, n);
            return 0;
        }
        let out = *src.add(n - 1) >> (64 - s);
        for i in (1..n).rev() {
            *dst.add(i) = (*src.add(i) << s) | (*src.add(i - 1) >> (64 - s));
        }
        *dst = *src << s;
        out
    }
}

/// Sets the `n` limbs at `dst` to the `n` limbs at `src` shifted right by `0 < s < 64` bits.
///
/// # Safety
///
/// The limbs at `src` must be valid for reads and those at `dst` for writes; `dst` may be `src`,
/// or below it.
pub(super) unsafe fn rshift(dst: *mut u64, src: *const u64, n: usize, s: u32) {
    debug_assert!(0 < s && s < 64);
    // SAFETY: as the caller guarantees; from the bottom up, each limb of `src` is read before the
    // limb of `dst` at or below it is written.
    unsafe {
        for i in 0..n - 1 {
            *dst.add(i) = (*src.add(i) >> s) | (*src.add(i + 1) << (64 - s));
        }
        *dst.add(n - 1) = *src.add(n - 1) >> s;
    }
}

/// Sets the `n` limbs at `r` to the `n` limbs at `x` times `y`, and returns the limb above them.
///
/// # Safety
///
/// The limbs at `x` must be valid for reads and those at `r` for writes; `r` may be `x`.
pub(super) unsafe fn mul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let mut carry = 0;
    // SAFETY: as the caller guarantees; limb `i` of `x` is read before limb `i` of `r` is written.
    unsafe {
        for i in 0..n {
            let p = u128::from(*x.add(i)) * u128::from(y) + u128::from(carry);
            *r.add(i) = p as u64;
            carry = (p >> 64) as u64;
        }
    }
    carry
}

/// Adds the `n` limbs at `x` times `y` to the `n` limbs at `r`, and returns the limb that carries
/// out of them.
///
/// # Safety
///
/// The limbs at `x` must be valid for reads and those at `r` for reads and writes, apart.
pub(super) unsafe fn addmul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let mut carry = 0;
    // SAFETY: as the caller guarantees; `x y + r + carry` fits a `u128`.
    unsafe {
        for i in 0..n {
            let p =
                u128::from(*x.add(i)) * u128::from(y) + u128::from(*r.add(i)) + u128::from(carry);
            *r.add(i) = p as u64;
            carry = (p >> 64) as u64;
        }
    }
    carry
}
