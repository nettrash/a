//! The sums and products of fractions on limbs, for [`Rational`](crate::math::rational::Rational):
//! the four parts read into limbs once, Knuth's reductions by greatest common divisors carried out
//! on them in working memory, and the two parts of the result written once, into the vectors of
//! the operand assigned to where they have room.
//!
//! On numbers, each of the half dozen steps of a sum (two greatest common divisors, two or three
//! exact divisions, three or four products and an addition) reads its operands' bytes into limbs,
//! allocates its result and writes it back as bytes; here those passes over the parts happen once
//! for the whole operation. A part of one limb takes the steps for one limb: a remainder by it, the
//! greatest common divisor of two limbs, a product and an exact division by a limb, each one pass
//! over the longer number. Parts of up to two limbs each take the same steps in registers.
//!
//! The operands are fractions in lowest terms with positive denominators, numerators of any sign,
//! not zero (the callers take zero apart); a denominator may be given as 1 by `None`.

use super::super::add::read_small;
use super::super::div::limbs::binvert_limb;
use super::super::div::{Powers, REM_1_BYTES, REM_1_BYTES_WIDE, REM_1_MAX};
use super::super::div::{divide_2by1, reciprocal, rem_1 as rem_powers};
use super::super::mul::limbs::{add_1, add_n, cmp_n, mul, mul_itch, sub_1, sub_n};
use super::super::mul::{Operand, Product, to_limbs};
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
use super::limbs::{
    divexact, divexact_itch, gcd_u64, gcd_u128, lshift, normalized, tdiv_itch, tdiv_qr,
};
use super::{gcd_itch, gcd_limbs, strip, working};
use core::cmp::Ordering;
use core::mem::MaybeUninit;
use core::ptr;

/// A fraction given by its parts: the numerator, not zero, and the denominator, positive, or 1 for
/// `None`.
#[derive(Clone, Copy)]
pub(crate) struct Fraction<'a> {
    /// The numerator.
    pub(crate) num: &'a Number,
    /// The denominator, or 1.
    pub(crate) den: Option<&'a Number>,
}

/// A part read for the operation: its vector's operand and its sign.
#[derive(Clone, Copy)]
struct Part {
    op: Operand,
    negative: bool,
}

impl Part {
    #[inline(always)]
    fn of(n: &Number) -> Part {
        Part {
            op: Operand::of(n),
            negative: n.negative,
        }
    }

    /// Its number of limbs.
    #[inline(always)]
    fn limbs(self) -> usize {
        self.op.significant.div_ceil(8)
    }
}

/// The operands of an operation, read: two numerators and two denominators, 1 for `None`.
#[derive(Clone, Copy)]
struct Operands {
    xn: Part,
    xd: Option<Part>,
    yn: Part,
    yd: Option<Part>,
}

impl Operands {
    #[inline(always)]
    fn of(x: Fraction, y: Fraction) -> Operands {
        Operands {
            xn: Part::of(x.num),
            xd: x.den.map(Part::of),
            yn: Part::of(y.num),
            yd: y.den.map(Part::of),
        }
    }

    /// The limbs of the four parts together.
    #[inline(always)]
    fn limbs(self) -> usize {
        let d = |p: Option<Part>| p.map_or(1, Part::limbs);
        self.xn.limbs() + d(self.xd) + self.yn.limbs() + d(self.yd)
    }
}

/// The working memory that the values of an operation take, for parts of `s` limbs together:
/// every value is at most `s + 1` limbs, and there are at most ten of them, each rounded up to an
/// even number.
#[inline(always)]
fn values_need(s: usize) -> usize {
    12 * (s + 4)
}

/// `x + y`, or `x - y` when `subtract` is set, in lowest terms, both parts in the byte order
/// `order`, in new vectors.
pub(crate) fn sum_new(
    x: Fraction,
    y: Fraction,
    subtract: bool,
    order: ByteOrder,
) -> (Number, Number) {
    let ops = Operands::of(x, y);
    // SAFETY: the parts are borrowed until the result is written into new vectors.
    unsafe {
        if let Some(p) = wide_operands(ops)
            && let Some(w) = sum_wide(p, subtract)
        {
            let (num, den) = w.parts();
            return (new_number(&num, order), new_number(&den, order));
        }
        run(ops, |v| {
            let (num, den) = sum(v, ops, subtract);
            (new_number(&num, order), new_number(&den, order))
        })
    }
}

/// `t = t + y` (`t = y + t` when `left` is not set), or with `subtract` the difference, in lowest
/// terms, in the byte order `order`, in the vectors of `t` where they have room.
pub(crate) fn sum_assign(
    tn: &mut Number,
    td: &mut Number,
    y: Fraction,
    left: bool,
    subtract: bool,
    order: ByteOrder,
) {
    let t = Fraction {
        num: tn,
        den: Some(td),
    };
    let ops = if left {
        Operands::of(t, y)
    } else {
        Operands::of(y, t)
    };
    // SAFETY: the parts of `t` are read into limbs before its vectors are written, once the
    // result is computed; `y` is borrowed throughout.
    unsafe {
        if let Some(p) = wide_operands(ops)
            && let Some(w) = sum_wide(p, subtract)
        {
            let (num, den) = w.parts();
            put(tn, &num, order);
            put(td, &den, order);
            return;
        }
        run(ops, |v| {
            let (num, den) = sum(v, ops, subtract);
            put(tn, &num, order);
            put(td, &den, order);
        })
    }
}

/// `x · y`, or `x / y` when `divide` is set (`y` not zero), in lowest terms, both parts in the byte
/// order `order`, in new vectors.
pub(crate) fn product_new(
    x: Fraction,
    y: Fraction,
    divide: bool,
    order: ByteOrder,
) -> (Number, Number) {
    let ops = Operands::of(x, y);
    // SAFETY: as for `sum_new`.
    unsafe {
        if let Some(p) = wide_operands(ops) {
            // The result is bound: its parts point into it.
            let w = product_wide(p, divide);
            let (num, den) = w.parts();
            return (new_number(&num, order), new_number(&den, order));
        }
        run(ops, |v| {
            let (num, den) = product(v, ops, divide);
            (new_number(&num, order), new_number(&den, order))
        })
    }
}

/// `t = t · y` (`t = y · t` when `left` is not set), or with `divide` the quotient, in lowest terms,
/// in the byte order `order`, in the vectors of `t` where they have room.
pub(crate) fn product_assign(
    tn: &mut Number,
    td: &mut Number,
    y: Fraction,
    left: bool,
    divide: bool,
    order: ByteOrder,
) {
    let t = Fraction {
        num: tn,
        den: Some(td),
    };
    let ops = if left {
        Operands::of(t, y)
    } else {
        Operands::of(y, t)
    };
    // SAFETY: as for `sum_assign`.
    unsafe {
        if let Some(p) = wide_operands(ops) {
            // The result is bound: its parts point into it.
            let w = product_wide(p, divide);
            let (num, den) = w.parts();
            put(tn, &num, order);
            put(td, &den, order);
            return;
        }
        run(ops, |v| {
            let (num, den) = product(v, ops, divide);
            put(tn, &num, order);
            put(td, &den, order);
        })
    }
}

/// A value in limbs: the pointer and the number of limbs, without zero limbs at the top; zero has
/// none.
#[derive(Clone, Copy)]
struct Limbs {
    p: *const u64,
    n: usize,
}

/// A signed value in limbs.
#[derive(Clone, Copy)]
struct Signed {
    m: Limbs,
    negative: bool,
}

/// The limb 1, for the denominators given as `None`.
static ONE: u64 = 1;

impl Limbs {
    #[inline(always)]
    fn one() -> Limbs {
        Limbs { p: &ONE, n: 1 }
    }

    #[inline(always)]
    fn is_one(self) -> bool {
        // SAFETY: a value of one limb holds it.
        self.n == 1 && unsafe { *self.p } == 1
    }

    /// Whether the values are equal.
    #[inline(always)]
    fn eq(self, o: Limbs) -> bool {
        // SAFETY: both hold their limbs.
        self.n == o.n && unsafe { cmp_n(self.p, o.p, self.n) } == Ordering::Equal
    }

    /// The order of the values.
    #[inline(always)]
    fn cmp(self, o: Limbs) -> Ordering {
        if self.n != o.n {
            return self.n.cmp(&o.n);
        }
        // SAFETY: both hold their limbs.
        unsafe { cmp_n(self.p, o.p, self.n) }
    }

    /// The value as a product to write: its length in bytes, one for zero.
    #[inline(always)]
    fn product(self, negative: bool) -> Product {
        if self.n == 0 {
            return Product {
                r: ptr::null(),
                len: 1,
                negative: false,
            };
        }
        // SAFETY: the top limb, not zero, is held.
        let top = unsafe { *self.p.add(self.n - 1) };
        Product {
            r: self.p,
            len: 8 * self.n - top.leading_zeros() as usize / 8,
            negative,
        }
    }
}

/// The working memory of the values: limbs handed out from the start, each value rounded up to an
/// even number, so that every value starts on 16 bytes as the kernels like.
struct Values {
    p: *mut u64,
    used: usize,
    #[cfg(debug_assertions)]
    len: usize,
}

impl Values {
    /// Room for `n` limbs.
    ///
    /// # Safety
    ///
    /// The working memory must hold them (as planned by [`values_need`]).
    #[inline(always)]
    unsafe fn take(&mut self, n: usize) -> *mut u64 {
        let n = (n + 1) & !1;
        #[cfg(debug_assertions)]
        assert!(
            self.used + n <= self.len,
            "values past their working memory"
        );
        // SAFETY: as the caller guarantees.
        let r = unsafe { self.p.add(self.used) };
        self.used += n;
        r
    }
}

/// Reads the operands into limbs in working memory of their own and runs `f` on them: the values
/// read come first in it, in the order `xn, xd, yn, yd`.
///
/// # Safety
///
/// The operands' vectors must stay valid for reads until `f` has computed its result.
#[inline(always)]
unsafe fn run<R>(ops: Operands, f: impl FnOnce(&mut Values) -> R) -> R {
    let need = values_need(ops.limbs());
    let go = |p: *mut u64| {
        f(&mut Values {
            p,
            used: 0,
            #[cfg(debug_assertions)]
            len: need,
        })
    };
    if need <= INLINE_LIMBS {
        // Short operands take a buffer in the operation's own frame, not the 10 KiB that
        // `working` sets aside.
        let mut buf = Inline([MaybeUninit::uninit(); INLINE_LIMBS]);
        return go(buf.0.as_mut_ptr().cast());
    }
    working(need, |s| go(s.take(need)))
}

/// The limbs of working memory in the frame of an operation: values of parts of up to 12 limbs
/// together.
const INLINE_LIMBS: usize = 192;

/// The buffer of [`INLINE_LIMBS`], aligned to 16 bytes as the kernels like.
#[repr(C, align(16))]
struct Inline([MaybeUninit<u64>; INLINE_LIMBS]);

/// Reads a part into limbs taken from `v`.
///
/// # Safety
///
/// The part's vector must be valid for reads, and `v` hold its limbs.
#[inline(always)]
unsafe fn read(v: &mut Values, p: Part) -> Signed {
    let n = p.limbs();
    // SAFETY: as the caller guarantees.
    unsafe {
        let r = v.take(n);
        if n > 0 {
            to_limbs(p.op, r);
        }
        Signed {
            m: Limbs { p: r, n },
            negative: p.negative && n > 0,
        }
    }
}

/// Reads a denominator, 1 for `None`.
///
/// # Safety
///
/// As for [`read`].
#[inline(always)]
unsafe fn read_den(v: &mut Values, p: Option<Part>) -> Limbs {
    match p {
        // SAFETY: as the caller guarantees.
        Some(p) => unsafe { read(v, p).m },
        None => Limbs::one(),
    }
}

/// Writes `x` as the whole value of `n`, in its vector when it has room.
///
/// # Safety
///
/// The limbs of `x` must be valid for reads, apart from the vector of `n`.
#[inline(always)]
unsafe fn put(n: &mut Number, x: &Signed, order: ByteOrder) {
    let p = x.m.product(x.negative);
    if n.value.capacity() < p.len {
        // A vector with room to grow: a rational assigned to again and again, as a sum of many
        // terms is, takes a new one every eighth of its length rather than every byte or two.
        n.value = Vec::with_capacity(p.len + p.len / 8 + 8);
    }
    // SAFETY: as the caller guarantees; the vector has room.
    unsafe { p.put_in(&mut n.value, order) };
    n.negative = p.negative;
    n.byte_order = order;
}

/// `x` as a new number, in one vector of exactly its length.
///
/// # Safety
///
/// The limbs of `x` must be valid for reads.
#[inline(always)]
unsafe fn new_number(x: &Signed, order: ByteOrder) -> Number {
    let p = x.m.product(x.negative);
    let mut v = Vec::with_capacity(p.len);
    // SAFETY: the vector has room for the `len` bytes, all written before the length is set.
    unsafe {
        p.write(v.as_mut_ptr(), order);
        v.set_len(p.len);
    }
    Number {
        value: v,
        negative: p.negative,
        byte_order: order,
    }
}

/// The value 1, positive.
#[inline(always)]
fn one() -> Signed {
    Signed {
        m: Limbs::one(),
        negative: false,
    }
}

/// Zero over 1.
#[inline(always)]
fn zero_over_one() -> (Signed, Signed) {
    (
        Signed {
            m: Limbs {
                p: ptr::null(),
                n: 0,
            },
            negative: false,
        },
        one(),
    )
}

/// A positive value.
#[inline(always)]
fn positive(m: Limbs) -> Signed {
    Signed { m, negative: false }
}

// The steps

/// `x ± y` as Knuth's section 4.5.1 reduces it: with `g = gcd(xd, yd)`, `s = xd/g` and `t = yd/g`,
/// the numerator `xn·t ± yn·s` can share a factor only with `g`, by `g₂`, and the result is
/// `((xn·t ± yn·s)/g₂) / (s·(yd/g₂))`.
///
/// # Safety
///
/// The operands' vectors must be valid for reads, and `v` the working memory [`run`] planned.
unsafe fn sum(v: &mut Values, ops: Operands, subtract: bool) -> (Signed, Signed) {
    // SAFETY: every value is taken from `v`, which holds them, and the kernels get the working
    // memory they plan.
    unsafe {
        let xn = read(v, ops.xn);
        let xd = read_den(v, ops.xd);
        let mut yn = read(v, ops.yn);
        let yd = read_den(v, ops.yd);
        if subtract {
            yn.negative = !yn.negative && yn.m.n > 0;
        }
        if xn.m.n == 0 {
            return (yn, positive(yd));
        }
        if yn.m.n == 0 {
            return (xn, positive(xd));
        }
        let (x1, y1) = (xd.is_one(), yd.is_one());
        if x1 && y1 {
            return (signed_sum(v, xn, yn), one());
        }
        if y1 {
            // `(xn ± yn·xd)/xd`, prime to `xd` as `xn` is.
            let p = times(v, yn, xd);
            return (signed_sum(v, xn, p), positive(xd));
        }
        if x1 {
            let p = times(v, xn, yd);
            return (signed_sum(v, p, yn), positive(yd));
        }
        if xd.eq(yd) {
            // `(xn ± yn)/d`, reduced by the divisors of `d`.
            let num = signed_sum(v, xn, yn);
            if num.m.n == 0 {
                return zero_over_one();
            }
            let g = gcd(v, num.m, xd);
            if g.is_one() {
                return (num, positive(xd));
            }
            let q = exact(v, num.m, g);
            return (
                Signed {
                    m: q,
                    negative: num.negative,
                },
                positive(exact(v, xd, g)),
            );
        }
        if xd.n == 1 || yd.n == 1 {
            return sum_by_limb(v, xn, xd, yn, yd);
        }
        let g = gcd(v, xd, yd);
        if g.is_one() {
            let a = times(v, xn, yd);
            let b = times(v, yn, xd);
            let num = signed_sum(v, a, b);
            let den = mul_limbs(v, xd, yd);
            return (num, positive(den));
        }
        let s = exact(v, xd, g);
        let t = exact(v, yd, g);
        let a = times(v, xn, t);
        let b = times(v, yn, s);
        let num = signed_sum(v, a, b);
        if num.m.n == 0 {
            return zero_over_one();
        }
        let g2 = gcd(v, num.m, g);
        if g2.is_one() {
            return (num, positive(mul_limbs(v, s, yd)));
        }
        let q = exact(v, num.m, g2);
        let u = exact(v, yd, g2);
        (
            Signed {
                m: q,
                negative: num.negative,
            },
            positive(mul_limbs(v, s, u)),
        )
    }
}

/// [`sum`] when one denominator is a limb `d`, at least 2: both remainders by `d`, of the other
/// denominator for `g` and of the numerator for `g₂` (modulo `g`, which divides `d`), take the
/// powers of `B` modulo `d` computed once.
///
/// # Safety
///
/// As for [`sum`], with the values read and the denominators not 1.
unsafe fn sum_by_limb(
    v: &mut Values,
    xn: Signed,
    xd: Limbs,
    yn: Signed,
    yd: Limbs,
) -> (Signed, Signed) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let (d, e) = if yd.n == 1 { (*yd.p, xd) } else { (*xd.p, yd) };
        let mut by = ByLimb::new(d);
        let g = gcd_u64(by.rem(e), d);
        if g == 1 {
            let a = times(v, xn, yd);
            let b = times(v, yn, xd);
            let num = signed_sum(v, a, b);
            let den = mul_limbs(v, xd, yd);
            return (num, positive(den));
        }
        let gl = from_wide(v, u128::from(g));
        let s = exact(v, xd, gl);
        let t = exact(v, yd, gl);
        let a = times(v, xn, t);
        let b = times(v, yn, s);
        let num = signed_sum(v, a, b);
        if num.m.n == 0 {
            return zero_over_one();
        }
        let g2 = gcd_u64(by.rem(num.m) % g, g);
        if g2 == 1 {
            return (num, positive(mul_limbs(v, s, yd)));
        }
        let gl2 = from_wide(v, u128::from(g2));
        let q = exact(v, num.m, gl2);
        let u = exact(v, yd, gl2);
        (
            Signed {
                m: q,
                negative: num.negative,
            },
            positive(mul_limbs(v, s, u)),
        )
    }
}

/// `x · y`, or `x / y` with `divide`: `((xn/g₁)·(yn/g₂)) / ((xd/g₂)·(yd/g₁))` with
/// `g₁ = gcd(xn, yd)` and `g₂ = gcd(yn, xd)`, in lowest terms as it is; a quotient is the product
/// by the reciprocal, `yd/|yn|` of the sign of `yn`.
///
/// # Safety
///
/// As for [`sum`]; with `divide`, `yn` is not zero.
unsafe fn product(v: &mut Values, ops: Operands, divide: bool) -> (Signed, Signed) {
    // SAFETY: as in `sum`.
    unsafe {
        let xn = read(v, ops.xn);
        let xd = read_den(v, ops.xd);
        let yn0 = read(v, ops.yn);
        let yd0 = read_den(v, ops.yd);
        if xn.m.n == 0 || yn0.m.n == 0 {
            return zero_over_one();
        }
        let (yn, yd) = if divide { (yd0, yn0.m) } else { (yn0.m, yd0) };
        let negative = xn.negative != yn0.negative;
        let (a, e) = if yd.is_one() {
            (xn.m, yd)
        } else {
            let g1 = gcd(v, xn.m, yd);
            if g1.is_one() {
                (xn.m, yd)
            } else {
                (exact(v, xn.m, g1), exact(v, yd, g1))
            }
        };
        let (b, c) = if xd.is_one() {
            (yn, xd)
        } else {
            let g2 = gcd(v, yn, xd);
            if g2.is_one() {
                (yn, xd)
            } else {
                (exact(v, yn, g2), exact(v, xd, g2))
            }
        };
        let num = mul_limbs(v, a, b);
        let den = mul_limbs(v, c, e);
        (Signed { m: num, negative }, positive(den))
    }
}

// The kernels on values

/// `x · m` for a signed `x` and a positive `m`.
///
/// # Safety
///
/// As for [`mul_limbs`].
#[inline(always)]
unsafe fn times(v: &mut Values, x: Signed, m: Limbs) -> Signed {
    if m.is_one() {
        return x;
    }
    Signed {
        // SAFETY: as the caller guarantees.
        m: unsafe { mul_limbs(v, x.m, m) },
        negative: x.negative,
    }
}

/// The product of two values not zero, in new limbs from `v`.
///
/// # Safety
///
/// The limbs of the values must be valid for reads, and `v` hold the product's.
unsafe fn mul_limbs(v: &mut Values, x: Limbs, y: Limbs) -> Limbs {
    if x.is_one() {
        return y;
    }
    if y.is_one() {
        return x;
    }
    let (x, y) = if x.n >= y.n { (x, y) } else { (y, x) };
    // SAFETY: as the caller guarantees; the product's `x.n + y.n` limbs are taken from `v`, and
    // the multiplication gets the working memory it plans.
    unsafe {
        let r = v.take(x.n + y.n);
        if y.n == 1 && x.n <= SHORT_MUL {
            // By one limb, inline: the multiplication's dispatch costs more than the product.
            let m = *y.p;
            let mut carry = 0u64;
            for i in 0..x.n {
                let p = u128::from(*x.p.add(i)) * u128::from(m) + u128::from(carry);
                *r.add(i) = p as u64;
                carry = (p >> 64) as u64;
            }
            *r.add(x.n) = carry;
            let n = x.n + 1;
            return Limbs {
                p: r,
                n: if carry == 0 { n - 1 } else { n },
            };
        }
        if x.n == 2 {
            // Two limbs by two.
            let (a0, a1, b0, b1) = (*x.p, *x.p.add(1), *y.p, *y.p.add(1));
            let p00 = u128::from(a0) * u128::from(b0);
            let p01 = u128::from(a0) * u128::from(b1);
            let p10 = u128::from(a1) * u128::from(b0);
            let p11 = u128::from(a1) * u128::from(b1);
            let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
            let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
            *r = p00 as u64;
            *r.add(1) = mid as u64;
            *r.add(2) = high as u64;
            *r.add(3) = (high >> 64) as u64;
            return Limbs {
                p: r,
                n: if high >> 64 == 0 { 3 } else { 4 },
            };
        }
        let itch = mul_itch(x.n, y.n);
        if itch == 0 {
            mul(r, x.p, x.n, y.p, y.n, Scratch::none());
        } else {
            working(itch, |s| mul(r, x.p, x.n, y.p, y.n, s));
        }
        let n = x.n + y.n;
        Limbs {
            p: r,
            n: if *r.add(n - 1) == 0 { n - 1 } else { n },
        }
    }
}

/// The longest values that [`mul_limbs`] multiplies by a limb inline.
const SHORT_MUL: usize = 8;

/// `x + y` of signed values not zero, in new limbs from `v`; zero when they cancel.
///
/// # Safety
///
/// The limbs of the values must be valid for reads, and `v` hold the longer one's and one more.
unsafe fn signed_sum(v: &mut Values, x: Signed, y: Signed) -> Signed {
    // SAFETY: as the caller guarantees; the result takes the longer value's limbs and one more.
    unsafe {
        if x.negative == y.negative {
            let (l, s) = if x.m.n >= y.m.n {
                (x.m, y.m)
            } else {
                (y.m, x.m)
            };
            let r = v.take(l.n + 1);
            let carry = add_n(r, l.p, s.p, s.n);
            ptr::copy_nonoverlapping(l.p.add(s.n), r.add(s.n), l.n - s.n);
            let top = add_1(r.add(s.n), l.n - s.n, u64::from(carry));
            *r.add(l.n) = top;
            return Signed {
                m: Limbs {
                    p: r,
                    n: l.n + top as usize,
                },
                negative: x.negative,
            };
        }
        let (l, s, negative) = match x.m.cmp(y.m) {
            Ordering::Equal => {
                return Signed {
                    m: Limbs {
                        p: ptr::null(),
                        n: 0,
                    },
                    negative: false,
                };
            }
            Ordering::Greater => (x.m, y.m, x.negative),
            Ordering::Less => (y.m, x.m, y.negative),
        };
        let r = v.take(l.n);
        let borrow = sub_n(r, l.p, s.p, s.n);
        ptr::copy_nonoverlapping(l.p.add(s.n), r.add(s.n), l.n - s.n);
        sub_1(r.add(s.n), l.n - s.n, u64::from(borrow));
        Signed {
            m: Limbs {
                p: r,
                n: normalized(r, l.n),
            },
            negative,
        }
    }
}

/// The greatest common divisor of two positive values, in new limbs from `v` (or one of them).
///
/// # Safety
///
/// The limbs of the values must be valid for reads, and `v` hold the shorter one's.
unsafe fn gcd(v: &mut Values, x: Limbs, y: Limbs) -> Limbs {
    // SAFETY: as the caller guarantees.
    unsafe {
        if x.n <= 2 && y.n <= 2 {
            let g = gcd_u128(wide(x), wide(y));
            return from_wide(v, g);
        }
        if x.n == 1 || y.n == 1 {
            let (l, d) = if y.n == 1 { (x, *y.p) } else { (y, *x.p) };
            let g = gcd_u64(rem_limb(l, d), d);
            return from_wide(v, u128::from(g));
        }
        let (long, short) = if x.n >= y.n { (x.n, y.n) } else { (y.n, x.n) };
        let g = v.take(short);
        // Both numbers with a limb to spare, the result shifted with a limb above it and its zero
        // limbs, and the scratch of the first division and of the steps, as `gcd_general` plans.
        let need = (x.n + 1) + (y.n + 1) + (long + 2) + gcd_itch(long, short);
        let n = working(need, |s| {
            let (pa, s) = s.split(x.n + 1);
            let (pb, s) = s.split(y.n + 1);
            let (r, t) = s.split(long + 2);
            ptr::copy_nonoverlapping(x.p, pa, x.n);
            ptr::copy_nonoverlapping(y.p, pb, y.n);
            let (na, za) = strip(pa, x.n);
            let (nb, zb) = strip(pb, y.n);
            let k = za.min(zb);
            let (pa, na, pb, nb) = if na >= nb {
                (pa, na, pb, nb)
            } else {
                (pb, nb, pa, na)
            };
            let (zl, zbits) = (k / 64, (k % 64) as u32);
            let gp = r.add(zl);
            let gn = gcd_limbs(gp, pa, na, pb, nb, t, gcd_itch(long, short));
            ptr::write_bytes(r, 0, zl);
            *gp.add(gn) = lshift(gp, gp, gn, zbits);
            let n = normalized(r, zl + gn + 1);
            ptr::copy_nonoverlapping(r, g, n);
            n
        });
        Limbs { p: g, n }
    }
}

/// The value of at most two limbs as a `u128`.
///
/// # Safety
///
/// The value's limbs must be valid for reads.
#[inline(always)]
unsafe fn wide(x: Limbs) -> u128 {
    // SAFETY: as the caller guarantees.
    unsafe {
        match x.n {
            0 => 0,
            1 => u128::from(*x.p),
            _ => u128::from(*x.p) | (u128::from(*x.p.add(1)) << 64),
        }
    }
}

/// `g`, not zero, as a value in new limbs from `v`, or the limb 1.
///
/// # Safety
///
/// `v` must hold two limbs.
#[inline(always)]
unsafe fn from_wide(v: &mut Values, g: u128) -> Limbs {
    if g == 1 {
        return Limbs::one();
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        let r = v.take(2);
        *r = g as u64;
        *r.add(1) = (g >> 64) as u64;
        Limbs {
            p: r,
            n: if g >> 64 == 0 { 1 } else { 2 },
        }
    }
}

/// Remainders by a limb `d`, not zero, of values of any length: long ones (from [`REM_1_BYTES`],
/// [`REM_1_BYTES_WIDE`] for `d` above [`REM_1_MAX`]) by the powers of `B` modulo `d`, computed on
/// the first and kept for the others, on their limbs read as little-endian bytes; shorter ones,
/// and every value on a big-endian target, by the division by limbs with `d`'s reciprocal.
struct ByLimb {
    d: u64,
    #[cfg(target_endian = "little")]
    powers: Option<Powers>,
}

impl ByLimb {
    #[inline(always)]
    fn new(d: u64) -> ByLimb {
        ByLimb {
            d,
            #[cfg(target_endian = "little")]
            powers: None,
        }
    }

    /// The remainder of the value `x`.
    ///
    /// # Safety
    ///
    /// The value's limbs must be valid for reads.
    unsafe fn rem(&mut self, x: Limbs) -> u64 {
        let d = self.d;
        if d == 1 || x.n == 0 {
            return 0;
        }
        // SAFETY: as the caller guarantees.
        unsafe {
            #[cfg(target_endian = "little")]
            {
                let bytes = 8 * x.n;
                let long = if d <= REM_1_MAX {
                    REM_1_BYTES
                } else {
                    REM_1_BYTES_WIDE
                };
                if bytes >= long {
                    let top = *x.p.add(x.n - 1);
                    let op = Operand {
                        ptr: x.p.cast(),
                        len: bytes,
                        big_endian: false,
                        significant: bytes - top.leading_zeros() as usize / 8,
                    };
                    let p = self.powers.get_or_insert_with(|| Powers::of(d));
                    return rem_powers(op, p);
                }
            }
            let s = d.leading_zeros();
            let dn = d << s;
            let inv = reciprocal(dn);
            let n = x.n;
            // The value shifted left by `s`, a limb at a time from the top: the bits above the
            // top limb, below `2^s <= dn`, start the remainder.
            let mut r = if s == 0 {
                0
            } else {
                *x.p.add(n - 1) >> (64 - s)
            };
            for i in (0..n).rev() {
                let lo = if s == 0 {
                    *x.p.add(i)
                } else if i > 0 {
                    (*x.p.add(i) << s) | (*x.p.add(i - 1) >> (64 - s))
                } else {
                    *x.p << s
                };
                r = divide_2by1(r, lo, dn, inv).1;
            }
            r >> s
        }
    }
}

/// The remainder of the value `x` by the limb `d`, not zero.
///
/// # Safety
///
/// The value's limbs must be valid for reads.
#[inline(always)]
unsafe fn rem_limb(x: Limbs, d: u64) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe { ByLimb::new(d).rem(x) }
}

/// `x / d` for a value `d` that divides the value `x` exactly, in new limbs from `v`.
///
/// # Safety
///
/// The limbs of the values must be valid for reads, and `v` hold `x.n` limbs; `d` divides `x`.
unsafe fn exact(v: &mut Values, x: Limbs, d: Limbs) -> Limbs {
    if d.is_one() {
        return x;
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        if d.n == 1 {
            return exact_1(v, x, *d.p);
        }
        let qn = x.n - d.n + 1;
        let q = v.take(qn);
        let need = divexact_itch(x.n, d.n);
        working(need, |t| divexact(q, x.p, x.n, d.p, d.n, t, need));
        Limbs {
            p: q,
            n: normalized(q, qn),
        }
    }
}

/// `x / d` for a limb `d`, at least 2, that divides the value `x` exactly, by Hensel's method a limb
/// at a time from the low end: `x` shifted right by the zero bits of `d`, each limb of the quotient
/// the one that makes the low limb of the partial remainder zero, times the inverse of the odd part
/// of `d` modulo `B`, and the high limb of its product with `d` borrowed from the next.
///
/// A step waits for a multiplication, the high limb of another and two additions, about 7.8 cycles
/// a limb on an Apple M4 Max. The high half of the quotient by the division from the top end, with
/// `d`'s reciprocal, in a second chain beside the first, measured no faster: 7.9 cycles a limb for
/// 1,001 limbs, 7.75 for 24 against 3.3 when calls overlap.
///
/// # Safety
///
/// As for [`exact`].
unsafe fn exact_1(v: &mut Values, x: Limbs, d: u64) -> Limbs {
    let z = d.trailing_zeros();
    let d = d >> z;
    let inv = binvert_limb(d);
    let n = x.n;
    // SAFETY: as the caller guarantees.
    unsafe {
        let q = v.take(n);
        let mut c = 0u64;
        for i in 0..n {
            let lo = if z == 0 {
                *x.p.add(i)
            } else {
                let hi = if i + 1 < n {
                    *x.p.add(i + 1) << (64 - z)
                } else {
                    0
                };
                (*x.p.add(i) >> z) | hi
            };
            let (s, b) = lo.overflowing_sub(c);
            let qi = s.wrapping_mul(inv);
            *q.add(i) = qi;
            c = ((u128::from(qi) * u128::from(d)) >> 64) as u64 + u64::from(b);
        }
        Limbs {
            p: q,
            n: normalized(q, n),
        }
    }
}

/// The greatest common divisor of two limbs, either of which may be zero, as `Number::gcd` takes
/// it: the binary method, after a remainder when one is longer than the other by 16 bits or more.
#[inline(always)]
pub(crate) fn gcd_words(x: u64, y: u64) -> u64 {
    gcd_u64(x, y)
}

/// The inverse of the odd limb `d` modulo `2^64`.
#[inline(always)]
pub(crate) fn inverse_word(d: u64) -> u64 {
    binvert_limb(d)
}

/// Divides `num` and `den`, positive or negative and not zero, in the shortest form and of more
/// than 16 bytes together, by their greatest common divisor, in place: the parts read into limbs
/// once, the gcd computed on them, and when it is not 1 both quotients written back into their own
/// vectors, which hold them.
pub(crate) fn reduce(num: &mut Number, den: &mut Number) {
    let (xn, xd) = (Part::of(num), Part::of(den));
    let need = values_need(xn.limbs() + xd.limbs());
    // SAFETY: both parts are read into limbs before either vector is written; the quotients are
    // no longer than the parts, so they fit the vectors.
    unsafe {
        working(need, |s| {
            let mut v = Values {
                p: s.ptr(),
                used: 0,
                #[cfg(debug_assertions)]
                len: need,
            };
            let _ = s.take(need);
            let a = read(&mut v, xn);
            let b = read(&mut v, xd);
            let g = gcd(&mut v, a.m, b.m);
            if g.is_one() {
                return;
            }
            let p = exact(&mut v, a.m, g);
            let q = exact(&mut v, b.m, g);
            let (on, od) = (num.byte_order, den.byte_order);
            put(
                num,
                &Signed {
                    m: p,
                    negative: a.negative,
                },
                on,
            );
            put(
                den,
                &Signed {
                    m: q,
                    negative: b.negative,
                },
                od,
            );
        });
    }
}

/// The greatest common divisor of two numbers of up to two limbs, either of which may be zero, as
/// `Number::gcd` takes it: the binary method on two limbs, then on one.
#[inline(always)]
pub(crate) fn gcd_wide(x: u128, y: u128) -> u128 {
    gcd_u128(x, y)
}

/// The nearest integer to `num/den`, for a positive `den` of at most [`NEAREST_LIMBS`] limbs that
/// does not divide `num`: halves away from zero, or to the even integer with `even`. The quotient
/// and the remainder of the magnitudes on limbs, the remainder doubled against the denominator, and
/// the quotient, one more when it rounds up, written once, of the sign of `num`, in the byte order
/// `order`.
pub(crate) fn nearest(num: &Number, den: &Number, even: bool, order: ByteOrder) -> Number {
    let (xn, xd) = (Part::of(num), Part::of(den));
    let (nn, dn) = (xn.limbs(), xd.limbs());
    debug_assert!((1..=NEAREST_LIMBS).contains(&dn));
    let need = 2 * (nn + 4) + 2 * (dn + 4) + tdiv_itch(nn.max(dn), dn);
    // SAFETY: both parts are borrowed until the result is written into a new vector; every value
    // is taken from the working memory planned for it.
    unsafe {
        working(need, |s| {
            let mut v = Values {
                p: s.ptr(),
                used: 0,
                #[cfg(debug_assertions)]
                len: need,
            };
            let _ = s.take(need);
            let a = read(&mut v, xn);
            let d = read(&mut v, xd).m;
            let (q, qn, r) = if a.m.n < d.n {
                let q = v.take(2);
                *q = 0;
                (q, 1, a.m)
            } else {
                let qn = a.m.n - d.n + 1;
                let q = v.take(qn + 1);
                let n = a.m.p.cast_mut();
                let itch = tdiv_itch(a.m.n, d.n);
                working(itch, |t| tdiv_qr(q, n, a.m.n, d.p, d.n, t));
                (
                    q,
                    qn,
                    Limbs {
                        p: n,
                        n: normalized(n, d.n),
                    },
                )
            };
            // `2r` against `d`.
            let up = if r.n == 0 {
                false
            } else {
                let r2 = v.take(r.n + 1);
                *r2.add(r.n) = lshift(r2, r.p, r.n, 1);
                let twice = Limbs {
                    p: r2,
                    n: normalized(r2, r.n + 1),
                };
                match twice.cmp(d) {
                    Ordering::Greater => true,
                    Ordering::Less => false,
                    Ordering::Equal => !even || *q & 1 == 1,
                }
            };
            *q.add(qn) = if up { add_1(q, qn, 1) } else { 0 };
            let m = Limbs {
                p: q,
                n: normalized(q, qn + 1),
            };
            new_number(
                &Signed {
                    m,
                    negative: a.negative && m.n > 0,
                },
                order,
            )
        })
    }
}

/// The longest denominators of [`nearest`]: from the divisors that the division divides by divide
/// and conquer, the numbers' own division and remainder do as well.
pub(crate) const NEAREST_LIMBS: usize = 40;

// Parts of up to two limbs

/// The magnitude of a part of at most 16 significant bytes.
///
/// # Safety
///
/// The part's vector must be valid for reads.
#[inline(always)]
unsafe fn wide_part(p: Part) -> u128 {
    let op = p.op;
    let order = if op.big_endian {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    // SAFETY: as the caller guarantees: the significant bytes are the last `significant` of the
    // vector big-endian and the first little-endian.
    unsafe {
        let start = if op.big_endian {
            op.ptr.add(op.len - op.significant)
        } else {
            op.ptr
        };
        read_small(core::slice::from_raw_parts(start, op.significant), order)
    }
}

/// The parts of both operands as `u128`s, when each has at most 16 bytes: the magnitudes of the
/// numerators with their signs (not set for zero) and the denominators.
///
/// # Safety
///
/// The operands' vectors must be valid for reads.
#[inline(always)]
unsafe fn wide_operands(ops: Operands) -> Option<[(bool, u128); 4]> {
    let fits = |p: Option<Part>| p.is_none_or(|p| p.op.significant <= 16);
    if !(fits(Some(ops.xn)) && fits(ops.xd) && fits(Some(ops.yn)) && fits(ops.yd)) {
        return None;
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        let num = |p: Part| {
            let m = wide_part(p);
            (p.negative && m != 0, m)
        };
        let den = |p: Option<Part>| (false, p.map_or(1, |p| wide_part(p)));
        Some([num(ops.xn), den(ops.xd), num(ops.yn), den(ops.yd)])
    }
}

/// A result of the paths on two limbs: the sign and the limbs of the numerator, at most 257 bits,
/// and of the denominator, at most 256, each with its number of limbs without zero limbs at the
/// top.
struct WideResult {
    negative: bool,
    num: [u64; 5],
    nn: usize,
    den: [u64; 4],
    nd: usize,
}

impl WideResult {
    /// The parts as values over the result's limbs.
    #[inline(always)]
    fn parts(&self) -> (Signed, Signed) {
        (
            Signed {
                m: Limbs {
                    p: self.num.as_ptr(),
                    n: self.nn,
                },
                negative: self.negative && self.nn > 0,
            },
            positive(Limbs {
                p: self.den.as_ptr(),
                n: self.nd,
            }),
        )
    }
}

/// The product of two `u128`s.
#[inline(always)]
fn mul_wide(a: u128, b: u128) -> [u64; 4] {
    let (a0, a1, b0, b1) = (a as u64, (a >> 64) as u64, b as u64, (b >> 64) as u64);
    let p00 = u128::from(a0) * u128::from(b0);
    let p01 = u128::from(a0) * u128::from(b1);
    let p10 = u128::from(a1) * u128::from(b0);
    let p11 = u128::from(a1) * u128::from(b1);
    let mid = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
    let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    [p00 as u64, mid as u64, high as u64, (high >> 64) as u64]
}

/// The number of limbs of `x` without zero limbs at the top.
#[inline(always)]
fn len_of<const N: usize>(x: &[u64; N]) -> usize {
    let mut n = N;
    while n > 0 && x[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// `±p ± q` for magnitudes of four limbs: the sign, not set for zero, and five limbs.
#[inline(always)]
fn signed_wide(np: bool, p: [u64; 4], nq: bool, q: [u64; 4]) -> (bool, [u64; 5]) {
    let mut r = [0u64; 5];
    if np == nq {
        let mut carry = false;
        for i in 0..4 {
            let (s, c1) = p[i].overflowing_add(q[i]);
            let (s, c2) = s.overflowing_add(u64::from(carry));
            r[i] = s;
            carry = c1 | c2;
        }
        r[4] = u64::from(carry);
        return (np, r);
    }
    let (l, s, negative) = match p.iter().rev().cmp(q.iter().rev()) {
        Ordering::Equal => return (false, r),
        Ordering::Greater => (p, q, np),
        Ordering::Less => (q, p, nq),
    };
    let mut borrow = false;
    for i in 0..4 {
        let (d, b1) = l[i].overflowing_sub(s[i]);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        r[i] = d;
        borrow = b1 | b2;
    }
    (negative, r)
}

/// `x / g` and `y / g` for a `g` that divides both, by Hensel's method: the quotients of `x` and
/// `y` shifted right by the zero bits of `g`, times the inverse of its odd part modulo `2^128`.
#[inline(always)]
fn exact_wide(x: u128, y: u128, g: u128) -> (u128, u128) {
    let z = g.trailing_zeros();
    let odd = g >> z;
    let v = u128::from(binvert_limb(odd as u64));
    let v = v.wrapping_mul(2u128.wrapping_sub(odd.wrapping_mul(v)));
    ((x >> z).wrapping_mul(v), (y >> z).wrapping_mul(v))
}

/// The result `num/den` of four limbs each over a positive denominator.
#[inline(always)]
fn wide_result(negative: bool, num: [u64; 5], den: [u64; 4]) -> WideResult {
    let nn = len_of(&num);
    WideResult {
        negative: negative && nn > 0,
        num,
        nn,
        den,
        nd: len_of(&den),
    }
}

/// `n/1` for a numerator of at most 128 bits.
#[inline(always)]
fn over(negative: bool, n: u128, d: u128) -> WideResult {
    wide_result(
        negative,
        [n as u64, (n >> 64) as u64, 0, 0, 0],
        [d as u64, (d >> 64) as u64, 0, 0],
    )
}

/// [`sum`] on parts of up to two limbs, in registers: `None` in the rare case that the numerator
/// shares a divisor of more than one limb with the denominators', which the limbs take.
///
/// The gcds are those of two `u128`s, the exact divisions Hensel's modulo `2^128`, the products and
/// the numerator in five limbs, and its remainder by `g` the division by a limb. Measured on an
/// Apple M4 Max, `&a + &b` of random parts of two limbs took 659 cycles so against 763 on limbs,
/// and with denominators sharing a limb 846 against 904; `&a * &b` 1,121 against 1,202.
#[inline(always)]
fn sum_wide(parts: [(bool, u128); 4], subtract: bool) -> Option<WideResult> {
    let [(na, a), (_, b), (nc, c), (_, d)] = parts;
    let nc = (nc ^ subtract) && c != 0;
    if a == 0 {
        return Some(over(nc, c, d));
    }
    if c == 0 {
        return Some(over(na, a, b));
    }
    let g = gcd_u128(b, d);
    if g == 1 {
        let (negative, num) = signed_wide(na, mul_wide(a, d), nc, mul_wide(c, b));
        return Some(wide_result(negative, num, mul_wide(b, d)));
    }
    if g >> 64 != 0 {
        return None;
    }
    let g = g as u64;
    let (s, t) = exact_wide(b, d, u128::from(g));
    let (negative, num) = signed_wide(na, mul_wide(a, t), nc, mul_wide(c, s));
    let nn = len_of(&num);
    if nn == 0 {
        return Some(over(false, 0, 1));
    }
    // SAFETY: the limbs are the numerator's, `nn` of them.
    let r = unsafe {
        rem_limb(
            Limbs {
                p: num.as_ptr(),
                n: nn,
            },
            g,
        )
    } % g;
    let g2 = gcd_u64(r, g);
    if g2 == 1 {
        return Some(wide_result(negative, num, mul_wide(s, d)));
    }
    // The numerator by `g2`, exactly, a limb at a time from the low end.
    let z = g2.trailing_zeros();
    let odd = g2 >> z;
    let inv = binvert_limb(odd);
    let mut q = [0u64; 5];
    let mut c = 0u64;
    for i in 0..5 {
        let hi = if i < 4 && z > 0 {
            num[i + 1] << (64 - z)
        } else {
            0
        };
        let lo = (num[i] >> z) | hi;
        let (s, b) = lo.overflowing_sub(c);
        let qi = s.wrapping_mul(inv);
        q[i] = qi;
        c = ((u128::from(qi) * u128::from(odd)) >> 64) as u64 + u64::from(b);
    }
    Some(wide_result(negative, q, mul_wide(s, d / u128::from(g2))))
}

/// [`product`] on parts of up to two limbs, in registers.
#[inline(always)]
fn product_wide(parts: [(bool, u128); 4], divide: bool) -> WideResult {
    let [(na, a), (_, b), (nc, c0), (_, d0)] = parts;
    let (c, d) = if divide { (d0, c0) } else { (c0, d0) };
    if a == 0 || c == 0 {
        return over(false, 0, 1);
    }
    let negative = na != nc;
    let (a, d) = if d == 1 {
        (a, d)
    } else {
        let g1 = gcd_u128(a, d);
        if g1 == 1 {
            (a, d)
        } else {
            exact_wide(a, d, g1)
        }
    };
    let (c, b) = if b == 1 {
        (c, b)
    } else {
        let g2 = gcd_u128(c, b);
        if g2 == 1 {
            (c, b)
        } else {
            exact_wide(c, b, g2)
        }
    };
    let num = mul_wide(a, c);
    wide_result(
        negative,
        [num[0], num[1], num[2], num[3], 0],
        mul_wide(b, d),
    )
}
