//! A proof that a basis is reduced, by interval arithmetic: the Cholesky factorisation of its
//! exact Gram matrix in [`BigFloat`] intervals, every lower end rounded down and every upper end
//! up, so that each interval holds the exact Gram–Schmidt quantity it stands for. A condition
//! whose interval lies on the right side of its bound holds; one whose interval lies on the wrong
//! side fails; one whose interval holds the bound is not decided at that precision.
//!
//! The widths grow with the conditioning of the basis, which for a reduced one is bounded
//! (Nguyen and Stehlé), so a precision of a few bits for each dimension decides almost every
//! condition; the ones left, such as a coefficient of exactly `η`, are for the exact test. The
//! intervals are tried in doubles ([`certify_f64`]), then in balls of double-doubles
//! ([`certify_dd`]), then in `BigFloat`s. The double-doubles decided the reduced knapsack bases
//! of 50 to 200 rows and the uniform ones of 8 to 10 bits of 50 to 200 rows that were measured,
//! but not q-ary bases `[[I, A], [0, q·I]]` of 150 and 200 rows with `q` of 30 bits (Gram
//! entries of 40 bits), which the audit found to go on to the `BigFloat`s: how many rows they
//! decide depends on the conditioning, not on the size alone.

use super::exact::Insertion;
use crate::math::float::{BigFloat, Round};
use crate::math::numbers::Number;

/// What the intervals tell of a basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// Every condition holds.
    Reduced,
    /// A condition fails.
    NotReduced,
    /// Some condition is not decided at this precision.
    Unsure,
}

/// A closed interval `[lo, hi]` of floats.
#[derive(Clone, Debug)]
struct Iv {
    lo: BigFloat,
    hi: BigFloat,
}

fn down(x: (BigFloat, core::cmp::Ordering)) -> BigFloat {
    x.0
}

impl Iv {
    fn of_number(n: &Number, p: u64) -> Iv {
        Iv {
            lo: BigFloat::from_number(n, p, Round::Down).0,
            hi: BigFloat::from_number(n, p, Round::Up).0,
        }
    }

    fn of_f64(x: f64) -> Iv {
        // 53 bits hold an f64 exactly.
        let v = BigFloat::from_f64_round(x, 53, Round::Nearest).0;
        Iv {
            lo: v.clone(),
            hi: v,
        }
    }

    fn add(&self, o: &Iv, p: u64) -> Iv {
        Iv {
            lo: down(self.lo.add_round(&o.lo, p, Round::Down)),
            hi: down(self.hi.add_round(&o.hi, p, Round::Up)),
        }
    }

    fn sub(&self, o: &Iv, p: u64) -> Iv {
        Iv {
            lo: down(self.lo.sub_round(&o.hi, p, Round::Down)),
            hi: down(self.hi.sub_round(&o.lo, p, Round::Up)),
        }
    }

    fn mul(&self, o: &Iv, p: u64) -> Iv {
        let lo = |x: &BigFloat, y: &BigFloat| down(x.mul_round(y, p, Round::Down));
        let hi = |x: &BigFloat, y: &BigFloat| down(x.mul_round(y, p, Round::Up));
        let (a1, a2, b1, b2) = (&self.lo, &self.hi, &o.lo, &o.hi);
        let (ap, an) = (*a1 >= 0.0, *a2 <= 0.0);
        let (bp, bn) = (*b1 >= 0.0, *b2 <= 0.0);
        match (ap, an, bp, bn) {
            (true, _, true, _) => Iv {
                lo: lo(a1, b1),
                hi: hi(a2, b2),
            },
            (true, _, _, true) => Iv {
                lo: lo(a2, b1),
                hi: hi(a1, b2),
            },
            (_, true, true, _) => Iv {
                lo: lo(a1, b2),
                hi: hi(a2, b1),
            },
            (_, true, _, true) => Iv {
                lo: lo(a2, b2),
                hi: hi(a1, b1),
            },
            (true, _, _, _) => Iv {
                lo: lo(a2, b1),
                hi: hi(a2, b2),
            },
            (_, true, _, _) => Iv {
                lo: lo(a1, b2),
                hi: hi(a1, b1),
            },
            (_, _, true, _) => Iv {
                lo: lo(a1, b2),
                hi: hi(a2, b2),
            },
            (_, _, _, true) => Iv {
                lo: lo(a2, b1),
                hi: hi(a1, b1),
            },
            _ => {
                let (l1, l2) = (lo(a1, b2), lo(a2, b1));
                let (h1, h2) = (hi(a1, b1), hi(a2, b2));
                Iv {
                    lo: if l1 < l2 { l1 } else { l2 },
                    hi: if h1 > h2 { h1 } else { h2 },
                }
            }
        }
    }

    /// `self / o` for `o.lo > 0`.
    fn div_pos(&self, o: &Iv, p: u64) -> Iv {
        let lo = |x: &BigFloat, y: &BigFloat| down(x.div_round(y, p, Round::Down));
        let hi = |x: &BigFloat, y: &BigFloat| down(x.div_round(y, p, Round::Up));
        Iv {
            lo: if self.lo >= 0.0 {
                lo(&self.lo, &o.hi)
            } else {
                lo(&self.lo, &o.lo)
            },
            hi: if self.hi >= 0.0 {
                hi(&self.hi, &o.lo)
            } else {
                hi(&self.hi, &o.hi)
            },
        }
    }

    /// `self²`.
    fn sqr(&self, p: u64) -> Iv {
        let (a, b) = (self.lo.abs(), self.hi.abs());
        let (small, large) = if a < b { (a, b) } else { (b, a) };
        let straddles = self.lo < 0.0 && self.hi > 0.0;
        Iv {
            lo: if straddles {
                BigFloat::zero(p)
            } else {
                down(small.mul_round(&small, p, Round::Down))
            },
            hi: down(large.mul_round(&large, p, Round::Up)),
        }
    }

    /// Whether the interval is `≥ 0` throughout (`Some(true)`), `< 0` throughout
    /// (`Some(false)`), or neither.
    fn sign(&self) -> Option<bool> {
        if self.lo >= 0.0 {
            Some(true)
        } else if self.hi < 0.0 {
            Some(false)
        } else {
            None
        }
    }
}

/// The operations the certificate does on intervals, each rounded outwards.
trait Interval: Sized + Clone {
    fn of_number(n: &Number, p: u64) -> Option<Self>;
    fn of_f64(x: f64) -> Self;
    fn add(&self, o: &Self, p: u64) -> Self;
    fn sub(&self, o: &Self, p: u64) -> Self;
    fn mul(&self, o: &Self, p: u64) -> Self;
    /// `self / o` for `o` above zero throughout.
    fn div_pos(&self, o: &Self, p: u64) -> Self;
    fn sqr(&self, p: u64) -> Self;
    /// `≥ 0` throughout (`Some(true)`), `< 0` throughout (`Some(false)`), or neither.
    fn sign(&self) -> Option<bool>;
    /// Whether the upper end is at most zero.
    fn hi_le_zero(&self) -> bool;
    /// Whether the lower end is at most zero.
    fn lo_le_zero(&self) -> bool;
}

impl Interval for Iv {
    fn of_number(n: &Number, p: u64) -> Option<Iv> {
        Some(Iv::of_number(n, p))
    }
    fn of_f64(x: f64) -> Iv {
        Iv::of_f64(x)
    }
    fn add(&self, o: &Iv, p: u64) -> Iv {
        Iv::add(self, o, p)
    }
    fn sub(&self, o: &Iv, p: u64) -> Iv {
        Iv::sub(self, o, p)
    }
    fn mul(&self, o: &Iv, p: u64) -> Iv {
        Iv::mul(self, o, p)
    }
    fn div_pos(&self, o: &Iv, p: u64) -> Iv {
        Iv::div_pos(self, o, p)
    }
    fn sqr(&self, p: u64) -> Iv {
        Iv::sqr(self, p)
    }
    fn sign(&self) -> Option<bool> {
        Iv::sign(self)
    }
    fn hi_le_zero(&self) -> bool {
        self.hi <= 0.0
    }
    fn lo_le_zero(&self) -> bool {
        self.lo <= 0.0
    }
}

/// An interval of doubles: each operation rounds to nearest and then steps a unit in the last
/// place outwards, which holds the exact result, as rounding to nearest errs by half a unit at
/// most. A NaN or an infinity anywhere makes the interval undecided, the whole line.
#[derive(Clone, Copy, Debug)]
struct Iv64 {
    lo: f64,
    hi: f64,
}

impl Iv64 {
    fn of(lo: f64, hi: f64) -> Iv64 {
        if lo.is_nan() || hi.is_nan() {
            return Iv64 {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            };
        }
        Iv64 {
            lo: lo.next_down(),
            hi: hi.next_up(),
        }
    }

    fn hull(c: [f64; 4]) -> Iv64 {
        let lo = c.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if c.iter().any(|x| x.is_nan()) {
            return Iv64::of(f64::NAN, f64::NAN);
        }
        Iv64::of(lo, hi)
    }
}

impl Interval for Iv64 {
    /// The doubles below and above `n`, when its magnitude is below `2^500` (so that the products
    /// of the certificate stay finite); `None` beyond.
    fn of_number(n: &Number, _: u64) -> Option<Iv64> {
        if n.bits() > 500 {
            return None;
        }
        if n.bits() <= 53 {
            // Exactly a double.
            let x = i64::try_from(n).ok()? as f64;
            return Some(Iv64 { lo: x, hi: x });
        }
        let lo = BigFloat::from_number(n, 53, Round::Down).0.to_f64();
        let hi = BigFloat::from_number(n, 53, Round::Up).0.to_f64();
        Some(Iv64 { lo, hi })
    }
    fn of_f64(x: f64) -> Iv64 {
        Iv64 { lo: x, hi: x }
    }
    fn add(&self, o: &Iv64, _: u64) -> Iv64 {
        Iv64::of(self.lo + o.lo, self.hi + o.hi)
    }
    fn sub(&self, o: &Iv64, _: u64) -> Iv64 {
        Iv64::of(self.lo - o.hi, self.hi - o.lo)
    }
    fn mul(&self, o: &Iv64, _: u64) -> Iv64 {
        Iv64::hull([
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        ])
    }
    fn div_pos(&self, o: &Iv64, _: u64) -> Iv64 {
        Iv64::hull([
            self.lo / o.lo,
            self.lo / o.hi,
            self.hi / o.lo,
            self.hi / o.hi,
        ])
    }
    fn sqr(&self, _: u64) -> Iv64 {
        let (a, b) = (self.lo * self.lo, self.hi * self.hi);
        let top = a.max(b);
        if self.lo < 0.0 && self.hi > 0.0 {
            Iv64::of(0.0, top).max0()
        } else {
            Iv64::of(a.min(b), top).max0()
        }
    }
    fn sign(&self) -> Option<bool> {
        if self.lo >= 0.0 {
            Some(true)
        } else if self.hi < 0.0 {
            Some(false)
        } else {
            None
        }
    }
    fn hi_le_zero(&self) -> bool {
        self.hi <= 0.0
    }
    fn lo_le_zero(&self) -> bool {
        self.lo <= 0.0
    }
}

impl Iv64 {
    /// A square is not negative: the lower end raised to zero.
    fn max0(self) -> Iv64 {
        Iv64 {
            lo: self.lo.max(0.0),
            hi: self.hi,
        }
    }
}

/// `u`, the unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;

/// `a + b` exactly as `s + t` (Knuth's TwoSum; no condition on the operands).
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a1 = s - b;
    let b1 = s - a1;
    (s, (a - a1) + (b - b1))
}

/// `a·b` exactly as `p + e` (with a fused multiply-add), barring underflow.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// A real `x` in a ball: a centre `hi + lo` (a double-double, the exact sum of two doubles) and a
/// radius `r`, `|x − (hi + lo)| ≤ r`. The centre's operations are the double-word algorithms of
/// Joldes, Muller and Popescu (*Tight and rigorous error bounds for basic building blocks of
/// double-word arithmetic*, ACM TOMS 44, 2017), every renormalisation by TwoSum, each of
/// relative error below `5u²` for a sum or a product, and below `30u²` for the quotient here (a
/// first quotient and the exact part of its remainder, the rest of it rounded four times); the radius takes
/// those errors at `32u²` and `128u²`, with the propagated radii, in doubles enlarged by `1 + 16u`
/// against their own rounding. Centres outside `[2⁻⁴⁰⁰, 2⁴⁰⁰]` in magnitude (other than 0), where
/// the exact transformations could underflow or overflow, or anything not finite, give an
/// infinite radius: undecided.
#[derive(Clone, Copy, Debug)]
struct IvDd {
    hi: f64,
    lo: f64,
    r: f64,
}

/// `2⁻⁴⁰⁰` and `2⁴⁰⁰`, the magnitudes a centre may have.
const DD_MIN: f64 = 3.872_591_914_849_318e-121;
const DD_MAX: f64 = 2.582_249_878_086_908_6e120;

impl IvDd {
    /// The ball of the centre `(hi, lo)` (any two doubles, renormalised) and the radius `r`, the
    /// range checked.
    #[inline(always)]
    fn ball(hi: f64, lo: f64, r: f64) -> IvDd {
        let (hi, lo) = two_sum(hi, lo);
        let a = hi.abs();
        let bad = !(hi.is_finite() && lo.is_finite() && r.is_finite())
            || a > DD_MAX
            || (a != 0.0 && a < DD_MIN);
        IvDd {
            hi,
            lo,
            r: if bad { f64::INFINITY } else { r },
        }
    }

    /// An upper bound of `|hi + lo|`.
    #[inline(always)]
    fn mag(&self) -> f64 {
        (self.hi.abs() + self.lo.abs()) * (1.0 + 4.0 * U)
    }

    /// The exact lower end is above zero.
    #[inline(always)]
    fn positive(&self) -> bool {
        self.hi > 0.0 && (self.r + self.lo.abs()) * (1.0 + 4.0 * U) < self.hi * (1.0 - 4.0 * U)
    }

    /// The exact upper end is below zero.
    #[inline(always)]
    fn negative(&self) -> bool {
        self.hi < 0.0 && (self.r + self.lo.abs()) * (1.0 + 4.0 * U) < -self.hi * (1.0 - 4.0 * U)
    }

    /// The exact value is zero.
    #[inline(always)]
    fn zero(&self) -> bool {
        self.hi == 0.0 && self.lo == 0.0 && self.r == 0.0
    }
}

/// The centre of `x + y`, double-word plus double-word (AccurateDWPlusDW).
#[inline(always)]
fn dd_add(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (sh, sl) = two_sum(xh, yh);
    let (th, tl) = two_sum(xl, yl);
    let c = sl + th;
    let (vh, vl) = two_sum(sh, c);
    let w = tl + vl;
    two_sum(vh, w)
}

/// The centre of `x·y`, double-word times double-word (DWTimesDW with fused multiply-adds).
#[inline(always)]
fn dd_mul(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (ch, cl1) = two_prod(xh, yh);
    let tl0 = xl * yl;
    let tl1 = xh.mul_add(yl, tl0);
    let cl2 = xl.mul_add(yh, tl1);
    let cl3 = cl1 + cl2;
    two_sum(ch, cl3)
}

/// The centre of `x / y`: `q₁ = xh / yh`, then the remainder `x − q₁·y` (its first part exact)
/// divided by `yh` for the second word.
#[inline(always)]
fn dd_div(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let q1 = xh / yh;
    let (ph, pl) = two_prod(q1, yh);
    let r = (((xh - ph) - pl) + xl) - q1 * yl;
    let q2 = r / yh;
    two_sum(q1, q2)
}

impl Interval for IvDd {
    /// The integer `n` of magnitude below `2^400` (so that the products stay in range): the
    /// nearest double, the nearest double to the rest, and the rest of that as the radius.
    fn of_number(n: &Number, _: u64) -> Option<IvDd> {
        if n.bits() > 400 {
            return None;
        }
        if n.bits() <= 53 {
            let x = i64::try_from(n).ok()? as f64;
            return Some(IvDd {
                hi: x,
                lo: 0.0,
                r: 0.0,
            });
        }
        let (m, e) = BigFloat::from_number(n, 53, Round::Nearest)
            .0
            .mantissa_exponent()?;
        // The nearest double as an integer (`e ≥ 0`, as `n` has more than 53 bits).
        let h = &m << usize::try_from(e).ok()?;
        let rest = n - &h;
        let hi = BigFloat::from_number(&h, 53, Round::Nearest).0.to_f64();
        if rest.is_zero() {
            return Some(IvDd {
                hi,
                lo: 0.0,
                r: 0.0,
            });
        }
        let lo_down = BigFloat::from_number(&rest, 53, Round::Down).0.to_f64();
        let lo_up = BigFloat::from_number(&rest, 53, Round::Up).0.to_f64();
        let lo = 0.5 * (lo_down + lo_up);
        Some(IvDd::ball(hi, lo, (lo_up - lo_down) * (1.0 + 4.0 * U)))
    }
    fn of_f64(x: f64) -> IvDd {
        IvDd::ball(x, 0.0, 0.0)
    }
    fn add(&self, o: &IvDd, _: u64) -> IvDd {
        let (h, l) = dd_add(self.hi, self.lo, o.hi, o.lo);
        let e = 32.0 * U * U * (h.abs() + l.abs());
        IvDd::ball(h, l, (self.r + o.r + e) * (1.0 + 16.0 * U))
    }
    fn sub(&self, o: &IvDd, _: u64) -> IvDd {
        let (h, l) = dd_add(self.hi, self.lo, -o.hi, -o.lo);
        let e = 32.0 * U * U * (h.abs() + l.abs());
        IvDd::ball(h, l, (self.r + o.r + e) * (1.0 + 16.0 * U))
    }
    fn mul(&self, o: &IvDd, _: u64) -> IvDd {
        let (h, l) = dd_mul(self.hi, self.lo, o.hi, o.lo);
        let e = 32.0 * U * U * (h.abs() + l.abs());
        let r = self.mag() * o.r + o.mag() * self.r + self.r * o.r + e;
        IvDd::ball(h, l, r * (1.0 + 16.0 * U))
    }
    fn div_pos(&self, o: &IvDd, _: u64) -> IvDd {
        if !o.positive() {
            return IvDd::ball(f64::NAN, 0.0, f64::INFINITY);
        }
        let (h, l) = dd_div(self.hi, self.lo, o.hi, o.lo);
        let q = (h.abs() + l.abs()) * (1.0 + 4.0 * U);
        // |x/y − c_x/c_y| ≤ (r_x + |c_x/c_y|·r_y) / (c_y − r_y), c_y − r_y bounded below.
        let low = (o.hi - o.lo.abs()) * (1.0 - 4.0 * U) - o.r * (1.0 + 4.0 * U);
        if low.is_nan() || low <= 0.0 {
            return IvDd::ball(f64::NAN, 0.0, f64::INFINITY);
        }
        let prop = (self.r + q * (1.0 + 256.0 * U * U) * o.r) / low;
        let e = 128.0 * U * U * q;
        IvDd::ball(h, l, (prop + e) * (1.0 + 16.0 * U))
    }
    fn sqr(&self, p: u64) -> IvDd {
        self.mul(self, p)
    }
    fn sign(&self) -> Option<bool> {
        if self.zero() || self.positive() {
            Some(true)
        } else if self.negative() {
            Some(false)
        } else {
            None
        }
    }
    fn hi_le_zero(&self) -> bool {
        self.zero() || self.negative()
    }
    fn lo_le_zero(&self) -> bool {
        !self.positive()
    }
}

/// The same in balls of double-doubles ([`IvDd`]): [`Verdict::Unsure`] when an entry of `g`
/// reaches `2^400` or the balls leave a condition open; otherwise as certain as [`certify`], at
/// about the cost of a Cholesky factorisation in double-doubles.
pub(super) fn certify_dd(g: &[Vec<Number>], delta: f64, eta: f64, ins: Insertion) -> Verdict {
    match certify_inner::<IvDd>(g, delta, eta, ins, 106) {
        Ok(v) | Err(v) => v,
    }
}

/// Folds a condition into the verdict so far: `Some(false)` ends it.
fn note(unsure: &mut bool, holds: Option<bool>) -> Result<(), Verdict> {
    match holds {
        Some(true) => Ok(()),
        Some(false) => Err(Verdict::NotReduced),
        None => {
            *unsure = true;
            Ok(())
        }
    }
}

/// Whether the rows whose exact Gram matrix has the lower triangle `g` are linearly
/// independent, with `|μᵢⱼ| ≤ η` and the exchange condition with `δ` at every place `ins`
/// allows, as intervals of precision `p` show it.
pub(super) fn certify(g: &[Vec<Number>], delta: f64, eta: f64, ins: Insertion, p: u64) -> Verdict {
    match certify_inner::<Iv>(g, delta, eta, ins, p) {
        Ok(v) | Err(v) => v,
    }
}

/// The same in intervals of doubles: [`Verdict::Unsure`] when an entry of `g` reaches `2^500`,
/// or the doubles leave a condition open; otherwise as certain as [`certify`], at a fraction of
/// its cost.
pub(super) fn certify_f64(g: &[Vec<Number>], delta: f64, eta: f64, ins: Insertion) -> Verdict {
    match certify_inner::<Iv64>(g, delta, eta, ins, 53) {
        Ok(v) | Err(v) => v,
    }
}

fn certify_inner<I: Interval>(
    g: &[Vec<Number>],
    delta: f64,
    eta: f64,
    ins: Insertion,
    p: u64,
) -> Result<Verdict, Verdict> {
    let n = g.len();
    let (delta, eta) = (I::of_f64(delta), I::of_f64(eta));
    let mut r: Vec<Vec<I>> = Vec::with_capacity(n);
    let mut mu: Vec<Vec<I>> = Vec::with_capacity(n);
    let mut unsure = false;
    for i in 0..n {
        let mut ri: Vec<I> = Vec::with_capacity(i + 1);
        let mut mi: Vec<I> = Vec::with_capacity(i);
        for j in 0..=i {
            let Some(mut t) = I::of_number(&g[i][j], p) else {
                return Err(Verdict::Unsure);
            };
            for k in 0..j {
                let mjk = if j < i { &mu[j][k] } else { &mi[k] };
                t = t.sub(&mjk.mul(&ri[k], p), p);
            }
            if j < i {
                let m = t.div_pos(&r[j][j], p);
                // |μ| ≤ η: η − μ ≥ 0 and η + μ ≥ 0.
                note(&mut unsure, eta.sub(&m, p).sign())?;
                note(&mut unsure, eta.add(&m, p).sign())?;
                ri.push(t);
                mi.push(m);
            } else {
                // The exact r_ii is never negative: an upper end at or below zero is a zero
                // vector, a dependency; a lower end there leaves nothing to divide by.
                if t.hi_le_zero() {
                    return Err(Verdict::NotReduced);
                }
                if t.lo_le_zero() {
                    return Err(Verdict::Unsure);
                }
                ri.push(t);
            }
        }
        // |πⱼ(bᵢ)|² = r_ii + Σ_{j ≤ l < i} μ²_il·r_ll ≥ δ·r_jj at each place allowed.
        let mut s = ri[i].clone();
        for j in (0..i).rev() {
            s = s.add(&mi[j].sqr(p).mul(&r[j][j], p), p);
            if ins.allowed(j, i, 0) {
                note(&mut unsure, s.sub(&delta.mul(&r[j][j], p), p).sign())?;
            }
            if ins == Insertion::Lll {
                break;
            }
        }
        r.push(ri);
        mu.push(mi);
    }
    Ok(if unsure {
        Verdict::Unsure
    } else {
        Verdict::Reduced
    })
}
