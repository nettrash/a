//! The evaluation of a polynomial over the integers at a dyadic point in fixed point, for the
//! signs and the Newton steps of the real roots' refinement: Horner's rule on vectors of limbs
//! in two's complement, every value held to a unit (a power of `2^64`) chosen from an a-priori
//! bound of its magnitude, so that a step is one product of limbs, the low limbs dropped, and one
//! addition, with no allocation, no normalisation and no rounding beyond the dropped limbs, each
//! drop counted in the bound of the error.
//!
//! The point is `x = ±X·2^(−64f)`. With `A = Σ|aᵢ|` and `m = max(1, |x|)`, the partial value
//! `Vᵢ = Σ_{j≥i} aⱼ·x^(j−i)` of Horner's rule is at most `A·m^(n−i)` in magnitude, and the
//! derivative's `Dᵢ` at most `n·A·m^(n−i)`; each is held in `W` limbs of the unit `2^(64u)` with
//! `u = ⌈(log2 bound + 3)/64⌉ − W`, so that it fits with two bits to spare and carries at least
//! `64(W − 1) − 3` bits relative to its bound. A step `V ← ⌊V·X·2^(−64k)⌋ + aᵢ` drops `k` limbs,
//! an error below one unit; the errors of the earlier steps are carried through `|x|` and the
//! change of unit (in double precision, rounded up at the end), so that the value is certain to
//! within the bound `err` units. When `W` holds every bit of the exact value (all units at most
//! `−(n−i)·f`) nothing nonzero is dropped and the value is exact.

use crate::math::float::{BigComplex, BigFloat, Round};
use crate::math::numbers::nat;
use crate::math::numbers::{ByteOrder, Number, from_limbs, limbs_of};
use crate::math::rational::Rational;
use core::cmp::Ordering;

/// The least width and degree of an evaluation by rectangular splitting, by measurement.
pub(crate) const RECT_W: usize = 8;
pub(crate) const RECT_N: usize = 8;

/// The prime `2^61 − 1` of the test for an exact zero.
const P61: u64 = (1 << 61) - 1;

/// `a·b mod 2^61 − 1`, for `a, b < 2^61`.
#[inline]
fn mulmod61(a: u64, b: u64) -> u64 {
    let t = u128::from(a) * u128::from(b);
    let r = (t as u64 & P61) + (t >> 61) as u64;
    let r = (r & P61) + (r >> 61);
    if r >= P61 { r - P61 } else { r }
}

/// The magnitude `m` (limbs) modulo `2^61 − 1`.
fn mod61(m: &[u64]) -> u64 {
    // 2^64 ≡ 8.
    let mut r = 0u64;
    for &l in m.iter().rev() {
        let lo = (l & P61) + (l >> 61);
        r = mulmod61(r, 8) + lo;
        r = (r & P61) + (r >> 61);
        if r >= P61 {
            r -= P61;
        }
    }
    r
}

/// `2^k` for an integer `k`, `0` below the normal range and infinity above it.
#[inline]
fn pow2f(k: i64) -> f64 {
    if k < -1022 {
        0.0
    } else if k > 1023 {
        f64::INFINITY
    } else {
        f64::from_bits(((k + 1023) as u64) << 52)
    }
}

/// `⌈t/1024⌉`.
#[inline]
fn ceil1024(t: i64) -> i64 {
    (t + 1023).div_euclid(1024)
}

/// The limbs without their zeros at the top.
fn trim(v: &mut Vec<u64>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// The number of bits of a magnitude.
fn bits_of(m: &[u64]) -> u64 {
    m.iter()
        .rposition(|&l| l != 0)
        .map_or(0, |k| k as u64 * 64 + 64 - u64::from(m[k].leading_zeros()))
}

/// An upper bound of `16·log2(m)` for the magnitude `m ≥ 1` (limbs), and `(t, b)` with
/// `m < (t + 1)·2^(b − 64)`, `t` its top 64 bits.
fn log16_up(m: &[u64]) -> (i64, u64, u64) {
    let b = bits_of(m);
    let k = ((b - 1) / 64) as usize;
    let s = (b - 1) % 64;
    // The top 64 bits, the bit b − 1 at the top.
    let t = if s == 63 || k == 0 {
        m[k] << (63 - s)
    } else {
        (m[k] << (63 - s)) | (m[k - 1] >> (s + 1))
    };
    // log2(t + 1) + (b − 64) < log2(t·(1 + 2^−52)) + b − 64; the double's log2 is good to a few
    // units in its last place, so one sixteenth of a bit covers every rounding.
    let l = ((t as f64) * (1.0 + 1e-15)).log2() + (b as f64 - 64.0);
    ((16.0 * l).ceil() as i64 + 1, t, b)
}

/// A dyadic point `±X·2^(−64f)`.
#[derive(Clone, Debug)]
pub(crate) struct Pt {
    pub(crate) neg: bool,
    /// The magnitude `X`, without zeros at the top (empty for 0).
    pub(crate) mag: Vec<u64>,
    /// The limbs of the fraction.
    pub(crate) f: usize,
}

impl Pt {
    /// `m·2^e`, exactly.
    pub(crate) fn dyadic(m: &Number, e: i64) -> Pt {
        let neg = m.is_negative();
        let mut mag = limbs_of(m);
        trim(&mut mag);
        let f = if e >= 0 {
            0
        } else {
            e.unsigned_abs().div_ceil(64) as usize
        };
        let sh = (e + 64 * f as i64) as u64;
        shl_vec(&mut mag, sh);
        Pt { neg, mag, f }
    }

    /// The float `x` (finite), exactly.
    pub(crate) fn of_float(x: &BigFloat) -> Pt {
        match x.mantissa_exponent() {
            Some((m, e)) => Pt::dyadic(&m, e),
            None => Pt {
                neg: false,
                mag: Vec::new(),
                f: 0,
            },
        }
    }

    /// The signed integer `±X` and `f`: the point is it times `2^(−64f)`.
    pub(crate) fn of_number(x: &Number, f: usize) -> Pt {
        let mut mag = limbs_of(x);
        trim(&mut mag);
        Pt {
            neg: x.is_negative(),
            mag,
            f,
        }
    }

    /// `±X`.
    pub(crate) fn number(&self) -> Number {
        let n = from_limbs(&self.mag, ByteOrder::LittleEndian);
        if self.neg { -n } else { n }
    }

    /// The point as a rational.
    pub(crate) fn rational(&self) -> Rational {
        Rational::new(self.number(), Number::from(1u8) << (64 * self.f))
    }

    /// The point rounded to nearest at `prec` bits.
    pub(crate) fn float(&self, prec: u64, round: Round) -> BigFloat {
        BigFloat::from_parts(&self.number(), -64 * self.f as i64, prec, round).0
    }

    /// The comparison of the point with the rational `r`.
    pub(crate) fn cmp_rational(&self, r: &Rational) -> Ordering {
        // ±X·d against n·2^(64f).
        let lhs = &self.number() * r.denom();
        let rhs = r.numer() << (64 * self.f);
        lhs.cmp(&rhs)
    }

    /// The point cut to its top `w + 1` limbs (keeping the integer part): for evaluations to
    /// about `w` limbs that need no rigour.
    pub(crate) fn top(&self, w: usize) -> Pt {
        let t = self.f.min(self.mag.len().saturating_sub(w + 1));
        let mut mag = self.mag[t..].to_vec();
        trim(&mut mag);
        Pt {
            neg: self.neg,
            mag,
            f: self.f - t,
        }
    }

    /// The number of bits of `X`.
    pub(crate) fn mag_bits(&self) -> u64 {
        bits_of(&self.mag)
    }

    /// An upper bound of `log2|x|`, rounded up to an integer (`i64::MIN` for 0).
    pub(crate) fn log2_up(&self) -> i64 {
        if self.mag.is_empty() {
            return i64::MIN;
        }
        bits_of(&self.mag) as i64 - 64 * self.f as i64
    }
}

/// `v <<= s` bits on a vector of limbs, growing it.
fn shl_vec(v: &mut Vec<u64>, s: u64) {
    if v.is_empty() || s == 0 {
        return;
    }
    let q = (s / 64) as usize;
    let r = (s % 64) as u32;
    if r != 0 {
        let mut carry = 0u64;
        for l in v.iter_mut() {
            let x = *l;
            *l = (x << r) | carry;
            carry = x >> (64 - r);
        }
        if carry != 0 {
            v.push(carry);
        }
    }
    if q > 0 {
        v.splice(0..0, core::iter::repeat_n(0u64, q));
    }
}

/// A value of the evaluation: `v·2^(64u)` in two's complement, certain to within `err` units.
#[derive(Clone, Debug)]
pub(crate) struct Val {
    pub(crate) v: Vec<u64>,
    pub(crate) u: i64,
    /// An upper bound of the error, in units `2^(64u)`.
    pub(crate) err: f64,
    /// Whether no bit was dropped: the value is exact.
    pub(crate) exact: bool,
}

impl Val {
    fn negative(&self) -> bool {
        (self.v[self.v.len() - 1] as i64) < 0
    }

    /// The magnitude of the limbs, rounded down to a double.
    fn mag_down(&self) -> f64 {
        let mut m = self.v.clone();
        if self.negative() {
            negate(&mut m);
        }
        let b = bits_of(&m);
        if b == 0 {
            return 0.0;
        }
        let (_, t, b) = log16_up(&m);
        // m ≥ t·2^(b − 64), and the double of t is within 2^11 of it.
        ((t >> 11) as f64) * pow2f(b as i64 - 53)
    }

    /// The sign of the value when it is certain: the magnitude above the bound of the error, or
    /// the value exact.
    pub(crate) fn sign(&self) -> Option<i32> {
        let zero = self.v.iter().all(|&l| l == 0);
        if self.exact {
            return Some(if zero {
                0
            } else if self.negative() {
                -1
            } else {
                1
            });
        }
        if zero || !self.err.is_finite() {
            return None;
        }
        let e = self.err;
        let big = if e < 9.0e18 {
            // Compared exactly with the integer above the bound.
            let t = e.ceil() as u64;
            let mut m = self.v.clone();
            if self.negative() {
                negate(&mut m);
            }
            m[1..].iter().any(|&l| l != 0) || m[0] > t
        } else {
            self.mag_down() > e
        };
        if !big {
            return None;
        }
        Some(if self.negative() { -1 } else { 1 })
    }

    /// The value as a signed integer of units.
    pub(crate) fn number(&self) -> Number {
        let mut m = self.v.clone();
        let neg = self.negative();
        if neg {
            negate(&mut m);
        }
        let n = from_limbs(&m, ByteOrder::LittleEndian);
        if neg { -n } else { n }
    }

    /// The number of bits of the magnitude.
    pub(crate) fn bits(&self) -> u64 {
        let mut m = self.v.clone();
        if self.negative() {
            negate(&mut m);
        }
        bits_of(&m)
    }
}

/// The negation in two's complement, in place.
fn negate(x: &mut [u64]) {
    let mut carry = true;
    for l in x.iter_mut() {
        let (s, c) = (!*l).overflowing_add(u64::from(carry));
        *l = s;
        carry = c;
    }
}

/// `dst += src` with `src` in two's complement sign-extended to the length of `dst`, wrapping.
#[inline]
fn add_sext(dst: &mut [u64], src: &[u64]) {
    let ext = match src.last() {
        Some(&t) if (t as i64) < 0 => u64::MAX,
        _ => 0,
    };
    let mut carry = 0u64;
    for (j, x) in dst.iter_mut().enumerate() {
        let y = if j < src.len() { src[j] } else { ext };
        let (s, c1) = x.overflowing_add(y);
        let (s, c2) = s.overflowing_add(carry);
        *x = s;
        carry = u64::from(c1 | c2);
        if j >= src.len() && carry == 0 && ext == 0 {
            break;
        }
    }
}

/// Up to these lengths of both operands a product is a schoolbook on the stack.
const SMALL: usize = 8;

/// `r = a·b` for `r` of zeros, as long as both.
#[inline(always)]
fn school(r: &mut [u64], a: &[u64], b: &[u64]) {
    for (j, &bj) in b.iter().enumerate() {
        let mut carry = 0u64;
        for (i, &ai) in a.iter().enumerate() {
            let t = u128::from(ai) * u128::from(bj) + u128::from(r[i + j]) + u128::from(carry);
            r[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        r[j + a.len()] = carry;
    }
}

/// `v ← ⌊v·X·2^(−64k)⌋` on the `w` limbs of `v` (two's complement), the product in `p` (or on
/// the stack): whether a nonzero limb was dropped.
#[inline]
fn mul_shift(v: &mut [u64], x: &[u64], k: usize, p: &mut Vec<u64>, s: &mut Vec<u64>) -> bool {
    let w = v.len();
    let len = w + x.len();
    if w <= SMALL && x.len() <= SMALL {
        let mut q = [0u64; 2 * SMALL];
        school(&mut q[..len], v, x);
        return take(v, x, k, &mut q[..len]);
    }
    if p.len() != len {
        p.resize(len, 0);
    }
    nat::mul_to(p, v, x, s);
    take(v, x, k, p)
}

/// The signed product `p` of `v` and `X` (read as unsigned) put right, and its limbs from `k` up
/// into `v`: whether a nonzero limb was dropped.
#[inline(always)]
fn take(v: &mut [u64], x: &[u64], k: usize, p: &mut [u64]) -> bool {
    let w = v.len();
    let len = p.len();
    if (v[w - 1] as i64) < 0 {
        // v was read as v + 2^(64w): take X·2^(64w) off again.
        let mut borrow = false;
        for (r, &y) in p[w..].iter_mut().zip(x) {
            let (d, b1) = r.overflowing_sub(y);
            let (d, b2) = d.overflowing_sub(u64::from(borrow));
            *r = d;
            borrow = b1 | b2;
        }
    }
    let dropped = p[..k.min(len)].iter().any(|&l| l != 0);
    let ext = if (p[len - 1] as i64) < 0 { u64::MAX } else { 0 };
    for (j, y) in v.iter_mut().enumerate() {
        *y = if k + j < len { p[k + j] } else { ext };
    }
    dropped
}

/// A polynomial over the integers prepared for evaluations in fixed point.
#[derive(Clone, Debug)]
pub(crate) struct Fixed {
    /// The degree, at least 1.
    n: usize,
    /// Limbs per coefficient.
    w: usize,
    /// The coefficients of `f(x)` and of `f(−x)` in two's complement, `w` limbs each.
    c: [Vec<u64>; 2],
    /// An upper bound of `16·log2 Σ|aᵢ|`.
    la16: i64,
    /// An upper bound of `16·log2 n`.
    ln16: i64,
    /// The coefficients of `f(x)` modulo `2^61 − 1`.
    r61: Vec<u64>,
    /// The magnitudes of the coefficients (limbs, without zeros at the top), their signs in
    /// `f(x)`, and upper bounds of them as doubles, for rectangular splitting.
    mag: Vec<Vec<u64>>,
    neg: Vec<bool>,
    abs: Vec<f64>,
}

impl Fixed {
    /// The polynomial of the coefficients `c` (constant first, a degree at least 1).
    pub(crate) fn new(c: &[Number]) -> Fixed {
        let n = c.len() - 1;
        let limbs: Vec<Vec<u64>> = c
            .iter()
            .map(|a| {
                let mut l = limbs_of(a);
                trim(&mut l);
                l
            })
            .collect();
        let bits = limbs.iter().map(|l| bits_of(l)).max().unwrap_or(0);
        let w = (bits + 1).div_ceil(64).max(1) as usize;
        let mut pos = vec![0u64; w * (n + 1)];
        let mut neg = vec![0u64; w * (n + 1)];
        let mut sum = Number::from(0u8);
        let mut r61 = Vec::with_capacity(n + 1);
        for (i, (l, a)) in limbs.iter().zip(c).enumerate() {
            let x = &mut pos[i * w..(i + 1) * w];
            x[..l.len()].copy_from_slice(l);
            let y = &mut neg[i * w..(i + 1) * w];
            y[..l.len()].copy_from_slice(l);
            // f(−x) has the odd coefficients negated.
            if a.is_negative() {
                negate(x);
            }
            if a.is_negative() != (i % 2 == 1) {
                negate(y);
            }
            sum += &a.abs();
            let r = mod61(l);
            r61.push(if a.is_negative() && r != 0 {
                P61 - r
            } else {
                r
            });
        }
        let la16 = log16_up(&limbs_of(&sum)).0;
        let ln16 = if n <= 1 { 0 } else { log16_up(&[n as u64]).0 };
        let abs = limbs
            .iter()
            .map(|l| {
                if l.is_empty() {
                    0.0
                } else {
                    let (_, t, b) = log16_up(l);
                    (t as f64 + 4096.0) * pow2f(b as i64 - 64)
                }
            })
            .collect();
        Fixed {
            n,
            w,
            c: [pos, neg],
            la16,
            ln16,
            r61,
            neg: c.iter().map(Number::is_negative).collect(),
            mag: limbs,
            abs,
        }
    }

    /// The degree.
    pub(crate) fn degree(&self) -> usize {
        self.n
    }

    /// The coefficient `i` of `f(±x)`.
    #[inline]
    fn coeff(&self, neg: bool, i: usize) -> &[u64] {
        &self.c[usize::from(neg)][i * self.w..(i + 1) * self.w]
    }

    /// The limbs `W` that make the evaluation at `x` exact.
    pub(crate) fn exact_width(&self, x: &Pt) -> usize {
        let lx = self.lx16(x);
        let n = self.n as i64;
        (ceil1024(self.la16 + n * lx + 48) + n * x.f as i64).max(1) as usize
    }

    /// `16·log2 max(1, |x|)`, rounded up.
    fn lx16(&self, x: &Pt) -> i64 {
        if bits_of(&x.mag) == 0 || x.log2_up() <= 0 {
            return 0;
        }
        log16_up(&x.mag).0 - 1024 * x.f as i64
    }

    /// Whether `f(x) ≡ 0` modulo `2^61 − 1` (always so when `f(x) = 0`).
    pub(crate) fn zero_mod_p(&self, x: &Pt) -> bool {
        // x = ±X·2^(−64f) and 2^61 ≡ 1, so 2^(−64f) ≡ 2^(61 − (64f mod 61)).
        let mut xm = mod61(&x.mag);
        let e = (64 * x.f as u64) % 61;
        xm = mulmod61(xm, if e == 0 { 1 } else { 1u64 << (61 - e) });
        if x.neg && xm != 0 {
            xm = P61 - xm;
        }
        let mut r = 0u64;
        for &a in self.r61.iter().rev() {
            r = mulmod61(r, xm) + a;
            if r >= P61 {
                r -= P61;
            }
        }
        r == 0
    }

    /// `f(x)` in `w` limbs (see the module) and, for `wd > 0`, an approximation of `f'(x)` in
    /// `wd` limbs, from the top limbs of `X` only (for Newton's method: its bound of the error is
    /// infinite).
    pub(crate) fn eval(&self, x: &Pt, w: usize, wd: usize) -> (Val, Option<Val>) {
        let n = self.n;
        let w = w.max(1);
        let deriv = wd > 0;
        if bits_of(&x.mag) == 0 {
            return self.at_zero(w, deriv);
        }
        let neg = x.neg;
        let xs = &x.mag[..];
        let f = x.f;
        let lx = self.lx16(x);
        // |x| < xt·2^xe.
        let (_, t, b) = log16_up(xs);
        let xt = t as f64 + 4096.0;
        let xe = b as i64 - 64 - 64 * f as i64;
        let wi = w as i64;
        let unit_v = |i: usize| ceil1024(self.la16 + (n - i) as i64 * lx + 48) - wi;
        let wdi = wd as i64;
        let unit_d = |i: usize| ceil1024(self.la16 + self.ln16 + (n - i) as i64 * lx + 48) - wdi;
        // The derivative's product with the top limbs of X only.
        let t = f.min(xs.len().saturating_sub(wd + 1));
        let (xd, fd) = (&xs[t..], f - t);
        let mut p = Vec::with_capacity(w + xs.len());
        let mut s = Vec::new();
        // V = aₙ.
        let mut v = vec![0u64; w];
        let mut u = unit_v(n);
        let mut exact = true;
        let mut e = 0.0f64;
        if self.add_coeff(&mut v, neg, n, u) {
            exact = false;
            e += 1.0;
        }
        // While the error stays below 2^(64W − 3) units, every value is below 2^(64(W + u) − 2)
        // in magnitude (its bound is below 2^(64(W + u) − 3)), so that none wraps round.
        let lim = pow2f(64 * wi - 3);
        let mut bad = false;
        let mut d = vec![0u64; wd];
        let mut ud = unit_d(n);
        for i in (0..n).rev() {
            let un = unit_v(i);
            if deriv {
                // D ← D·x + V, with the old V.
                let udn = unit_d(i);
                let k = (udn - ud) as usize + fd;
                mul_shift(&mut d, xd, k, &mut p, &mut s);
                // V's limbs from the unit udn up.
                let sh = (udn - u) as usize;
                if sh < w {
                    add_sext(&mut d, &v[sh..]);
                } else if (v[w - 1] as i64) < 0 {
                    add_sext(&mut d, &[u64::MAX]);
                }
                ud = udn;
            }
            let k = (un - u) as usize + f;
            let mut step = 0.0;
            if mul_shift(&mut v, xs, k, &mut p, &mut s) {
                exact = false;
                step += 1.0;
            }
            if self.add_coeff(&mut v, neg, i, un) {
                exact = false;
                step += 1.0;
            }
            e = e * xt * pow2f(xe - 64 * (un - u)) + step;
            u = un;
            // Below this bound no value can have wrapped round (see `lim`).
            if e >= lim {
                bad = true;
            }
        }
        if bad {
            e = f64::INFINITY;
        }
        // The roundings of the bounds themselves: relatively below 2n·2^−53.
        let grow = 1.0 + (n as f64 + 2.0) * pow2f(-50);
        let mut val = Val {
            v,
            u,
            err: e * grow + 1.0,
            exact,
        };
        let mut dv = deriv.then_some(Val {
            v: d,
            u: ud,
            err: f64::INFINITY,
            exact: false,
        });
        if neg {
            // f'(x) = −(d/dy) f(−y) at y = |x|.
            if let Some(dv) = dv.as_mut() {
                negate(&mut dv.v);
            }
        }
        if val.exact {
            val.err = 0.0;
        }
        (val, dv)
    }

    /// `v += aᵢ` (of `f(±x)`) at the unit `u`: whether nonzero bits of `aᵢ` fell below it.
    #[inline]
    fn add_coeff(&self, v: &mut [u64], neg: bool, i: usize, u: i64) -> bool {
        let a = self.coeff(neg, i);
        if u <= 0 {
            let o = u.unsigned_abs() as usize;
            // Past the top only for a zero coefficient: its bound puts it below the top.
            if o < v.len() {
                add_sext(&mut v[o..], a);
            } else {
                debug_assert!(a.iter().all(|&l| l == 0));
            }
            false
        } else {
            let s = u as usize;
            let dropped = a[..s.min(a.len())].iter().any(|&l| l != 0);
            if s < a.len() {
                add_sext(v, &a[s..]);
            } else if (a[a.len() - 1] as i64) < 0 {
                add_sext(v, &[u64::MAX]);
            }
            dropped
        }
    }

    /// The evaluation at 0: `a₀` and `a₁`, exactly.
    fn at_zero(&self, w: usize, deriv: bool) -> (Val, Option<Val>) {
        let exact = |a: &[u64]| {
            let mut v = vec![0u64; w.max(self.w)];
            add_sext(&mut v, a);
            Val {
                v,
                u: 0,
                err: 0.0,
                exact: true,
            }
        };
        let dv = deriv.then(|| exact(self.coeff(false, 1)));
        (exact(self.coeff(false, 0)), dv)
    }

    /// `f(x)` in `w` limbs: by rectangular splitting from `RECT_W` limbs and the degree
    /// `RECT_N`, by Horner's rule below.
    pub(crate) fn value(&self, x: &Pt, w: usize) -> Val {
        if w >= RECT_W
            && self.n >= RECT_N
            && let Some(v) = self.rect(x, w)
        {
            return v;
        }
        self.eval(x, w, 0).0
    }

    /// `f(x)` in `w` limbs by rectangular splitting (Paterson and Stockmeyer's, for small
    /// coefficients): the powers `x^j`, `j ≤ m ≈ √(n + 1)`, each to `w` limbs of its bound
    /// `max(1, |x|)^j`; the blocks `Bₖ = Σ_{j<m} a_{km+j}·x^j` by products of a coefficient's few
    /// limbs with a power's; Horner's rule in `x^m` on the blocks. `√n` full products and `n`
    /// short ones, against Horner's `n` full products; the errors counted as there. `None` where
    /// the units would not fit (never for `w` beyond the bits of `x^m`).
    fn rect(&self, x: &Pt, w: usize) -> Option<Val> {
        let n = self.n;
        if bits_of(&x.mag) == 0 {
            return None;
        }
        let neg = x.neg;
        let xs = &x.mag[..];
        let f = x.f;
        let lx = self.lx16(x);
        let (_, t, b) = log16_up(xs);
        let xt = t as f64 + 4096.0;
        let xe = b as i64 - 64 - 64 * f as i64;
        let wi = w as i64;
        let lim = pow2f(64 * wi - 3);
        let m = (n + 1).isqrt().max(2);
        let blocks = (n + m) / m;
        let mut p = Vec::with_capacity(2 * w + 2);
        let mut s = Vec::new();
        // The powers, x^j at the unit pu[j], certain to pe[j] units.
        let pu: Vec<i64> = (0..=m).map(|j| ceil1024(j as i64 * lx + 48) - wi).collect();
        let mut pw: Vec<Vec<u64>> = Vec::with_capacity(m + 1);
        let mut pe = vec![0.0f64; m + 1];
        pw.push(Vec::new());
        // x itself: X at the unit −f, moved to pu[1].
        let sh = pu[1] + f as i64;
        let mut p1 = vec![0u64; w];
        if sh >= 0 {
            let sh = sh as usize;
            pe[1] = f64::from(u8::from(xs[..sh.min(xs.len())].iter().any(|&l| l != 0)));
            for (j, y) in p1.iter_mut().enumerate() {
                *y = xs.get(sh + j).copied().unwrap_or(0);
            }
        } else {
            let o = sh.unsigned_abs() as usize;
            if o + xs.len() > w {
                return None;
            }
            p1[o..o + xs.len()].copy_from_slice(xs);
        }
        pw.push(p1);
        for j in 2..=m {
            let k = pu[j] - pu[j - 1] + f as i64;
            if k < 0 {
                return None;
            }
            let mut q = pw[j - 1].clone();
            let dropped = mul_shift(&mut q, xs, k as usize, &mut p, &mut s);
            pe[j] = pe[j - 1] * xt * pow2f(xe - 64 * (pu[j] - pu[j - 1]))
                + f64::from(u8::from(dropped));
            pw.push(q);
        }
        if pe[m] >= lim {
            return None;
        }
        // The blocks, at the unit bu, the bound of each A·max(1, |x|)^(m−1).
        let bu = ceil1024(self.la16 + (m as i64 - 1) * lx + 48) - wi;
        let vu = |k: usize| (ceil1024(self.la16 + (n - k * m) as i64 * lx + 48) - wi).max(bu);
        let mut prod = Vec::new();
        let mut sb = Vec::new();
        let mut block = |k: usize, out: &mut [u64]| -> f64 {
            out.fill(0);
            let mut e = 0.0;
            for j in 0..m {
                let i = k * m + j;
                if i > n {
                    break;
                }
                let a = &self.mag[i];
                if a.is_empty() {
                    continue;
                }
                if j == 0 {
                    if self.add_coeff(out, neg, i, bu) {
                        e += 1.0;
                    }
                    continue;
                }
                // |a|·x^j at the unit pu[j], moved to bu.
                let len = a.len() + w;
                prod.clear();
                prod.resize(len, 0);
                nat::mul_to(&mut prod, a, &pw[j], &mut sb);
                let sh = (bu - pu[j]) as usize;
                if prod[..sh.min(len)].iter().any(|&l| l != 0) {
                    e += 1.0;
                }
                let lo = sh.min(len);
                let hi = (sh + w).min(len);
                let minus = self.neg[i] != (neg && i % 2 == 1);
                let mut carry = false;
                for (jj, r) in out.iter_mut().enumerate() {
                    let y = if lo + jj < hi { prod[lo + jj] } else { 0 };
                    if minus {
                        let (d, b1) = r.overflowing_sub(y);
                        let (d, b2) = d.overflowing_sub(u64::from(carry));
                        *r = d;
                        carry = b1 | b2;
                    } else {
                        let (d, c1) = r.overflowing_add(y);
                        let (d, c2) = d.overflowing_add(u64::from(carry));
                        *r = d;
                        carry = c1 | c2;
                    }
                }
                e += self.abs[i] * pe[j] * pow2f(64 * (pu[j] - bu));
            }
            e
        };
        // Horner's rule in x^m on the blocks.
        let mut v = vec![0u64; w];
        let mut u = vu(blocks - 1);
        let mut e = block(blocks - 1, &mut v);
        let mut bk = vec![0u64; w];
        let pm = pw[m].clone();
        let pmu = pu[m];
        // |x^m| < 2^(m·lx/16), at least 1 where |x| < 1 (lx = 0).
        let lxm = m as f64 * lx as f64 / 16.0;
        for k in (0..blocks - 1).rev() {
            let un = vu(k);
            let d = un - u - pmu;
            if d < 0 {
                return None;
            }
            let dropped = mul_shift(&mut v, &pm, d as usize, &mut p, &mut s);
            let g = (lxm + 64.0 * (u - un) as f64).exp2() + pe[m] * pow2f(64 * (pmu + u - un));
            e = e * g + f64::from(u8::from(dropped)) + pe[m] * pow2f(64 * (wi + u + pmu - un) - 2);
            let eb = block(k, &mut bk);
            let sh = (un - bu) as usize;
            if bk[..sh.min(w)].iter().any(|&l| l != 0) {
                e += 1.0;
            }
            if sh < w {
                add_sext(&mut v, &bk[sh..]);
            } else if (bk[w - 1] as i64) < 0 {
                add_sext(&mut v, &[u64::MAX]);
            }
            e += eb * pow2f(64 * (bu - un));
            u = un;
            if e >= lim {
                return None;
            }
        }
        if !e.is_finite() || e >= lim {
            return None;
        }
        let grow = 1.0 + (n as f64 + 2.0) * pow2f(-50);
        Some(Val {
            v,
            u,
            err: e * grow + 1.0,
            exact: false,
        })
    }

    /// The sign of `f(x)`, certain: in fixed point at growing widths, and exactly when the value
    /// is zero modulo `2^61 − 1` or the widths reach the exact one.
    pub(crate) fn sign(&self, x: &Pt) -> i32 {
        let exact = self.exact_width(x);
        let zero = self.zero_mod_p(x);
        let mut w = (x.f + 2 + (self.n.ilog2() as usize) / 64).min(exact);
        loop {
            if zero {
                // Most likely a root: exactly.
                w = exact;
            }
            let v = if w == exact {
                self.eval(x, w, 0).0
            } else {
                self.value(x, w)
            };
            if let Some(s) = v.sign() {
                return s;
            }
            debug_assert!(w < exact, "an exact evaluation decides the sign");
            w = (2 * w).min(exact);
        }
    }
}

/// An upper bound `m·2^e` of a magnitude, `m` in `[1, 2)` (or 0), each operation rounded up.
#[derive(Clone, Copy, Debug)]
struct Bd {
    m: f64,
    e: i64,
}

/// One more than the relative rounding of a step of `Bd`.
const UP: f64 = 1.0 + 4.0 * f64::EPSILON;

impl Bd {
    const ZERO: Bd = Bd { m: 0.0, e: 0 };

    /// `m·2^e` for a finite `m ≥ 0` (normal or 0).
    fn new(m: f64, e: i64) -> Bd {
        if m == 0.0 {
            return Bd::ZERO;
        }
        let b = m.to_bits();
        let x = ((b >> 52) & 0x7ff) as i64 - 1023;
        Bd {
            m: f64::from_bits((b & !(0x7ff << 52)) | (1023 << 52)),
            e: e + x,
        }
    }

    /// An upper bound of the magnitude `l·2^(64·s)` (limbs).
    fn of(l: &[u64], s: i64) -> Bd {
        if bits_of(l) == 0 {
            return Bd::ZERO;
        }
        let (_, t, b) = log16_up(l);
        Bd::new(t as f64 + 4096.0, b as i64 - 64 + 64 * s)
    }

    fn mul(self, o: Bd) -> Bd {
        if self.m == 0.0 || o.m == 0.0 {
            return Bd::ZERO;
        }
        Bd::new(self.m * o.m * UP, self.e + o.e)
    }

    fn add(self, o: Bd) -> Bd {
        if self.m == 0.0 {
            return o;
        }
        if o.m == 0.0 {
            return self;
        }
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = b.e - a.e;
        // Below the normal range the smaller is less than a unit of the larger's last place.
        let t = if d < -1000 {
            a.m * f64::EPSILON
        } else {
            b.m * pow2f(d)
        };
        Bd::new((a.m + t) * UP, a.e)
    }

    /// The least unit `u` (of `2^64`) with the bound below `2^(64(w + u) − 3)`.
    fn unit(self, w: i64) -> i64 {
        if self.m == 0.0 {
            return i64::MIN / 4;
        }
        // The bound is below 2^(e + 1).
        (self.e + 1 + 3 + 63).div_euclid(64) - w
    }
}

/// A complex value of the evaluation: the parts `v·2^(64u)` in two's complement, each certain
/// to within `err` units.
#[derive(Clone, Debug)]
pub(crate) struct CVal {
    pub(crate) re: Vec<u64>,
    pub(crate) im: Vec<u64>,
    pub(crate) u: i64,
    pub(crate) err: f64,
}

impl CVal {
    /// The parts as exact floats.
    pub(crate) fn float(&self) -> BigComplex {
        let part = |v: &[u64]| {
            let val = Val {
                v: v.to_vec(),
                u: self.u,
                err: 0.0,
                exact: true,
            };
            let n = val.number();
            if n.is_zero() {
                BigFloat::zero(64)
            } else {
                BigFloat::from_parts(&n, 64 * self.u, n.bits().max(1), Round::Nearest).0
            }
        };
        BigComplex::new(part(&self.re), part(&self.im))
    }

    /// An upper bound of the error of the complex value, `√2·err` units, at `prec` bits.
    pub(crate) fn error(&self, prec: u64) -> BigFloat {
        if !self.err.is_finite() {
            return BigFloat::infinity(prec);
        }
        let e = BigFloat::from_f64_round(self.err * 1.5, prec, Round::Up).0;
        let p = BigFloat::from_parts(&Number::from(1u8), 64 * self.u, prec, Round::Nearest).0;
        e.mul_round(&p, prec, Round::Up).0
    }
}

/// The magnitude of the part `p` aligned to `f` limbs of fraction (`f ≥ p.f`).
fn aligned(p: &Pt, f: usize) -> Vec<u64> {
    if p.mag.is_empty() {
        return Vec::new();
    }
    let mut v = vec![0u64; f - p.f];
    v.extend_from_slice(&p.mag);
    v
}

/// `r ← r ± ⌊a·|x|·2^(−64k)⌋` for the signed `a` (`w` limbs): whether a nonzero limb was
/// dropped. `t` is working room.
#[allow(clippy::too_many_arguments)]
#[inline]
fn add_prod(
    r: &mut [u64],
    a: &[u64],
    x: &[u64],
    minus: bool,
    k: usize,
    t: &mut Vec<u64>,
    p: &mut Vec<u64>,
    s: &mut Vec<u64>,
) -> bool {
    t.clear();
    t.extend_from_slice(a);
    let dropped = mul_shift(t, x, k, p, s);
    if minus {
        let mut borrow = false;
        for (y, &z) in r.iter_mut().zip(t.iter()) {
            let (d, b1) = y.overflowing_sub(z);
            let (d, b2) = d.overflowing_sub(u64::from(borrow));
            *y = d;
            borrow = b1 | b2;
        }
    } else {
        let mut carry = false;
        for (y, &z) in r.iter_mut().zip(t.iter()) {
            let (d, c1) = y.overflowing_add(z);
            let (d, c2) = d.overflowing_add(u64::from(carry));
            *y = d;
            carry = c1 | c2;
        }
    }
    dropped
}

/// `dst += src·2^(64·sh)` (limbs moved up for `sh > 0`, the low ones dropped for `sh < 0`),
/// `src` in two's complement: whether a nonzero limb was dropped.
fn add_moved(dst: &mut [u64], src: &[u64], sh: i64) -> bool {
    let w = src.len();
    if sh <= 0 {
        let k = sh.unsigned_abs() as usize;
        let dropped = src[..k.min(w)].iter().any(|&l| l != 0);
        if k < w {
            add_sext(dst, &src[k..]);
        } else if (src[w - 1] as i64) < 0 {
            add_sext(dst, &[u64::MAX]);
        }
        dropped
    } else {
        let k = sh as usize;
        if k < dst.len() {
            add_sext(&mut dst[k..], src);
        }
        false
    }
}

impl Fixed {
    /// `f(z)` and `f'(z)` at the complex point `re + i·im` in `w` limbs, each certain to within
    /// its bound: Horner's rule on the parts, with the bounds `sᵢ = Σ_{j≥i} |aⱼ||z|^(j−i)` and
    /// `tᵢ = Σ_{j>i} (j−i)|aⱼ||z|^(j−i−1)` carried in double precision step by step (rounded
    /// up) for the units, and each part's error carried through `|Re z| + |Im z|`.
    pub(crate) fn ceval(&self, re: &Pt, im: &Pt, w: usize) -> (CVal, CVal) {
        let n = self.n;
        let w = w.max(1);
        let wi = w as i64;
        let f = re.f.max(im.f);
        let xr = aligned(re, f);
        let xi = aligned(im, f);
        let real = xi.is_empty();
        let (sr, si) = (re.neg, im.neg);
        let lim = pow2f(64 * wi - 3);
        let mag = |i: usize| Bd::of(&self.mag[i], 0);
        if n == 0 {
            let mut v = vec![0u64; w.max(self.w)];
            add_sext(&mut v, self.coeff(false, 0));
            let z = vec![0u64; v.len()];
            let zero = CVal {
                re: z.clone(),
                im: z.clone(),
                u: 0,
                err: 0.0,
            };
            return (
                CVal {
                    re: v,
                    im: z,
                    u: 0,
                    err: 0.0,
                },
                zero,
            );
        }
        if xr.is_empty() && real {
            let (v, d) = self.at_zero(w, true);
            let d = d.expect("the derivative");
            let zero = |x: &Val| vec![0u64; x.v.len()];
            return (
                CVal {
                    im: zero(&v),
                    re: v.v,
                    u: 0,
                    err: 0.0,
                },
                CVal {
                    im: zero(&d),
                    re: d.v,
                    u: 0,
                    err: 0.0,
                },
            );
        }
        // |z| ≤ |Re z| + |Im z|.
        let bz = Bd::of(&xr, -(f as i64)).add(Bd::of(&xi, -(f as i64)));
        let mut sb = mag(n);
        let mut tb = Bd::ZERO;
        let mut u = sb.unit(wi);
        let mut vr = vec![0u64; w];
        let mut vi = vec![0u64; w];
        let mut e = f64::from(u8::from(self.add_coeff(&mut vr, false, n, u)));
        let mut dr = vec![0u64; w];
        let mut di = vec![0u64; w];
        let mut ud = u;
        let mut ed = 0.0f64;
        let (mut t, mut p, mut s) = (Vec::new(), Vec::new(), Vec::new());
        let (mut nr, mut ni) = (vec![0u64; w], vec![0u64; w]);
        let fi = f as i64;
        // The growth of a part's error in a product: (|Re z| + |Im z|)·2^(64·(old − new unit)).
        let grow = |du: i64| bz.m * pow2f(bz.e + 64 * du);
        let mut bad = false;
        let cmul = |ar: &[u64],
                    ai: &[u64],
                    k: usize,
                    nr: &mut Vec<u64>,
                    ni: &mut Vec<u64>,
                    t: &mut Vec<u64>,
                    p: &mut Vec<u64>,
                    s: &mut Vec<u64>|
         -> f64 {
            nr.fill(0);
            ni.fill(0);
            let mut drops = 0.0;
            // (ar + i·ai)(xr + i·xi) = (ar·xr − ai·xi) + i·(ar·xi + ai·xr).
            if !xr.is_empty() {
                drops += f64::from(u8::from(add_prod(nr, ar, &xr, sr, k, t, p, s)));
                drops += f64::from(u8::from(add_prod(ni, ai, &xr, sr, k, t, p, s)));
            }
            if !real {
                drops += f64::from(u8::from(add_prod(nr, ai, &xi, !si, k, t, p, s)));
                drops += f64::from(u8::from(add_prod(ni, ar, &xi, si, k, t, p, s)));
            }
            // Each part has two of the four floors.
            drops.min(2.0)
        };
        for i in (0..n).rev() {
            let tn = tb.mul(bz).add(sb);
            let sn = sb.mul(bz).add(mag(i));
            let un = sn.unit(wi).max(u - fi);
            let udn = tn.unit(wi).max(ud - fi);
            // D ← D·z + V, with the old V.
            let dd = cmul(
                &dr,
                &di,
                (udn - ud + fi) as usize,
                &mut nr,
                &mut ni,
                &mut t,
                &mut p,
                &mut s,
            );
            core::mem::swap(&mut dr, &mut nr);
            core::mem::swap(&mut di, &mut ni);
            let mv = add_moved(&mut dr, &vr, u - udn) | add_moved(&mut di, &vi, u - udn);
            ed = ed * grow(ud - udn) + dd + e * pow2f(64 * (u - udn)) + f64::from(u8::from(mv));
            ud = udn;
            // V ← V·z + aᵢ.
            let dv = cmul(
                &vr,
                &vi,
                (un - u + fi) as usize,
                &mut nr,
                &mut ni,
                &mut t,
                &mut p,
                &mut s,
            );
            core::mem::swap(&mut vr, &mut nr);
            core::mem::swap(&mut vi, &mut ni);
            let tr = self.add_coeff(&mut vr, false, i, un);
            e = e * grow(u - un) + dv + f64::from(u8::from(tr));
            u = un;
            sb = sn;
            tb = tn;
            if e >= lim || ed >= lim {
                bad = true;
            }
        }
        let g = 1.0 + (n as f64 + 2.0) * pow2f(-50);
        let fix = |x: f64| if bad { f64::INFINITY } else { x * g + 1.0 };
        (
            CVal {
                re: vr,
                im: vi,
                u,
                err: fix(e),
            },
            CVal {
                re: dr,
                im: di,
                u: ud,
                err: fix(ed),
            },
        )
    }
}
