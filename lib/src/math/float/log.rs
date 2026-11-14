//! The logarithm: `ln`, `log2`, `log10` and `ln_1p`, correctly rounded.
//!
//! The kernel writes `x = 2^e u` with `2/3 <= u < 4/3` and takes `ln x = e ln 2 + ln u`, `ln u` by
//! Newton's iteration on the exponential, `y ← y + L(u e^-y - 1)` with `L(v) = v - v²/2 + v³/3`,
//! which multiplies the bits that are right by four, from the `f64` logarithm and at a precision
//! four times larger at each step, so that the steps together cost about a third more than the
//! last; only the last step's error counts, and it is bounded from the `v` it finds.

use super::ball::{Ball, Rad, round_near, ziv_fast};
use super::consts::ln2_approx;
use super::exp::exp_fix;
use super::fix::{self, SFix, Scratch};
use super::limbs;
use super::ziv::Approx;

use super::{BigFloat, Kind, Round};
use crate::math::numbers::Number;
use core::cmp::Ordering;

/// `ln u` for the fixed-point `u` (`n` limbs of fraction, `W = 64 n`) in about `[2/3, 4/3]`: the
/// signed fixed-point `y` and a bound of `|ln u - y|`.
///
/// The last step computes `v = u e^(-y) - 1` from `e^(-y)` within a relative `ε` (the product and
/// the scaling by `2^k` truncated, two units), so `|v - v_t| <= δ_v = (1 + |v|) ε + 2u` for the
/// exact `v_t = u e^(-y) - 1`, and `ln u = y + ln(1 + v_t)` exactly. With `|v| <= 1/4` and `δ_v <=
/// 1/16`, `|ln(1 + v_t) - ln(1 + v)| <= 2 δ_v`, `|ln(1 + v) - L(v)| <= |v|^4 / (4 (1 - |v|)) <=
/// |v|^4 / 2`, and `L(v)` computed with three truncations and a division by 3, five units: `|ln u -
/// y'| <= 2 δ_v + 5u + |v|^4 / 2`.
pub(crate) fn ln_fix(u: &[u64], n: usize) -> (SFix, Rad) {
    if n <= super::small_fix::SMALL {
        let (neg, y, e) = super::small_fix::ln_small(u, n);
        let mut m = y.to_vec();
        limbs::trim(&mut m);
        return (SFix { neg, m }, e);
    }
    let w = 64 * n as u64;
    let one = {
        let mut v = vec![0u64; n + 1];
        v[n] = 1;
        v
    };
    let d = SFix {
        neg: false,
        m: u.to_vec(),
    }
    .sub(&SFix { neg: false, m: one });
    if d.is_zero() {
        return (SFix::zero(), Rad::ZERO);
    }
    // The levels of the iteration, in limbs: four times as many at each.
    let mut levels = vec![n];
    while *levels.last().expect("a level") > 3 {
        let l = *levels.last().expect("a level");
        levels.push(l.div_ceil(4) + 1);
    }
    levels.reverse();
    let n0 = levels[0];
    // The seed: the `f64` logarithm, or `d` itself when it is too small for an `f64`.
    let df = d.to_f64(n);
    let mut y = if df.abs() > 1e-300 {
        let y0 = df.ln_1p();
        let bits = y0.abs().to_bits();
        let (mant, e) = if bits >> 52 == 0 {
            (bits << 1, -1075i128)
        } else {
            (
                (bits & ((1 << 52) - 1)) | 1 << 52,
                ((bits >> 52) as i128) - 1075,
            )
        };
        let (m, _) = fix::of_mag(&[mant], e, n0);
        SFix { neg: y0 < 0.0, m }
    } else {
        SFix {
            neg: d.neg,
            m: fix::shr(&d.m, 64 * (n - n0) as u64),
        }
    };
    let mut scratch = Scratch::default();
    let mut prev = n0;
    let mut bound = Rad::int(1);
    for &l in &levels {
        // `y` from the level before, exactly.
        if l > prev {
            y.m = limbs::shl(&y.m, 64 * (l - prev) as u64);
        }
        prev = l;
        let wl = 64 * l as u64;
        let ul = Rad::pow2(-i128::from(wl));
        let uj = fix::shr(u, 64 * (n - l) as u64);
        let (e, k, eps) = exp_fix(!y.neg && !y.is_zero(), &y.m, l);
        let mut p = fix::mul(&uj, &e, l, l + 2, &mut scratch);
        if k < 0 {
            p = fix::shr(&p, k.unsigned_abs());
        } else if k > 0 {
            p = limbs::shl(&p, k as u64);
        }
        let mut onel = vec![0u64; l + 1];
        onel[l] = 1;
        let v = SFix { neg: false, m: p }.sub(&SFix {
            neg: false,
            m: onel,
        });
        let v2 = fix::sqr(&v.m, l, l + 1, &mut scratch);
        let mut v3 = fix::mul(&v2, &v.m, l, l + 1, &mut scratch);
        fix::div1(&mut v3, 3);
        let half = fix::shr(&v2, 1);
        let poly = v
            .sub(&SFix {
                neg: false,
                m: half,
            })
            .add(&SFix { neg: v.neg, m: v3 });
        y = y.add(&poly);
        if l == n {
            let vb = fix::bits(&v.m);
            let vmag = if vb == 0 {
                Rad::ZERO
            } else {
                Rad::pow2(vb as i128 - i128::from(wl))
            };
            let dv = eps.mul(Rad::int(1).add(vmag)).add(ul.shl(1));
            if !vmag.le(Rad::pow2(-2)) || !dv.le(Rad::pow2(-4)) {
                return (y, Rad::int(1));
            }
            let rem = vmag.mul(vmag).mul(vmag).mul(vmag).shl(-1);
            bound = dv.shl(1).add(ul.mul(Rad::int(5))).add(rem);
        }
    }
    debug_assert!(w == 64 * n as u64);
    (y, bound)
}

/// `ln 2` in fixed point with `W + g` bits of fraction (`n` limbs and `g` bits), and the bound of
/// its error.
pub(crate) fn ln2_fix(n: usize, g: u64) -> (Vec<u64>, Rad) {
    let w = 64 * n as u64;
    let a = ln2_approx(w + g + 8);
    let (lf, dropped) = fix::of_mag(&a.m, a.lsb + i128::from(g), n);
    let mut d = Rad::pow2(a.err);
    if dropped {
        d = d.add(Rad::pow2(-i128::from(w + g)));
    }
    (lf, d)
}

/// The natural logarithm of the ball `x` with an error about `2^-w` of the result, `None` when the
/// ball holds a number not above zero.
///
/// With `x = 2^e u`, `ln x = e ln 2 + ln u`: for `e = 0` the result is `ln u`, near `u - 1`, and
/// the fixed point takes as many more bits as `u - 1` has leading zeros; otherwise `|ln x| >= ln 2 -
/// ln(3/2) > 1/4`. The midpoint truncated to the fixed point is within `u` below, and `ln` within
/// `u / (2/3) < 2u`; a radius `ρ` relative to the midpoint changes `ln` by at most `ρ / (1 - ρ) <=
/// 2ρ` for `ρ <= 1/2`; `e ln 2` is within `|e| δ_L + u`.
pub(crate) fn ln_ball(x: &Ball, w: u64) -> Option<Ball> {
    let (neg, m, lsb, err) = ln_raw(x, w)?;
    Some(Ball::of_limbs(neg, &m, lsb, err))
}

/// [`ln_ball`] in the raw: the sign, the magnitude `M`, the exponent of its last bit and the
/// bound of the error, for an approximation without the rounding of a ball's midpoint.
pub(crate) fn ln_raw(x: &Ball, w: u64) -> Option<(bool, Vec<u64>, i128, Rad)> {
    let of_ball = |b: Ball| -> Option<(bool, Vec<u64>, i128, Rad)> {
        let (neg, m, lsb) = b.mid.parts()?;
        Some((neg, m.to_vec(), lsb, b.rad))
    };
    let lower = x.lower()?;
    if x.mid.is_sign_negative() {
        return None;
    }
    let (_, m, lsb) = x.mid.parts()?;
    let ex = x.mid.exponent().expect("finite");
    let top = m[m.len() - 1];
    // `u = x 2^-e`, in `[2/3, 4/3)` about.
    let (e, run) = if top < 0xAAAA_AAAA_AAAA_AAAA {
        // `u = 2 u0` in `[1, 4/3)`: `u - 1` has as many leading zeros as the mantissa below its top.
        (ex - 1, leading_run(m, false))
    } else {
        (ex, leading_run(m, true))
    };
    let extra = if e == 0 { run } else { 0 };
    if extra > w + 8 {
        // `|ln(1 + d) - d| <= d² / (2 (1 - |d|)) <= d²` for `d = x - 1`, `|d| < 2^-(w + 8)`.
        let d = x
            .mid
            .sub_round(&BigFloat::from(1), x.mid.prec + 2, Round::Nearest)
            .0;
        let (_, hd) = super::ball::mag_bounds(&d);
        let mut err = hd.mul(hd);
        if !x.rad.is_zero() {
            let rho = x.rad.div(lower);
            if !rho.le(Rad::pow2(-1)) {
                return None;
            }
            err = err.add(rho.shl(1));
        }
        return of_ball(Ball::new(d, err));
    }
    if w + extra >= AGM_BITS
        && let Some(l) = ln_agm(x, w + 8, extra)
    {
        return of_ball(l);
    }
    let n = (w + extra + 16).div_ceil(64) as usize;
    let wb = 64 * n as u64;
    let u = Rad::pow2(-i128::from(wb));
    let (uf, dropped) = fix::of_mag(m, lsb - i128::from(e), n);
    let (mut y, mut err) = ln_fix(&uf, n);
    if dropped {
        err = err.add(u.shl(1));
    }
    if e != 0 {
        let ea = e.unsigned_abs();
        let g = 64 - (ea + 1).leading_zeros() as u64 + 4;
        let (lf, dl) = ln2_fix(n, g);
        let el = fix::shr(&fix::mul1(&lf, ea), g);
        y = y.add(&SFix { neg: e < 0, m: el });
        err = err.add(dl.mul(Rad::int(ea))).add(u);
    }
    if !x.rad.is_zero() {
        let rho = x.rad.div(lower);
        if !rho.le(Rad::pow2(-1)) {
            return None;
        }
        err = err.add(rho.shl(1));
    }
    let mut m = y.m;
    limbs::trim(&mut m);
    Some((y.neg, m, -i128::from(wb), err))
}

/// The logarithm takes the AGM from this many bits on.
pub(crate) const AGM_BITS: u64 = 5_000;

/// `ln x` by the arithmetic-geometric mean, as MPFR's `mpfr_log`: for `s = x 2^m >= 2^(W/2 + 8)`,
/// `ln x = π / (2 AGM(1, 4/s)) - m ln 2`, the AGM and every step in balls at `W` bits and more,
/// `extra` more for the cancellation of a result near zero.
///
/// The error of the formula: `AGM(1, k') = π / (2 K(k))` for `k'² + k² = 1`, and by DLMF 19.12.1
/// `K(k) = Σ_(j >= 0) ((1/2)_j / j!)² k'^(2j) (ln(1/k') + d(j))` with `0 < d(j) <= ln 4`, so for `k'
/// = 4/s` the first term is `ln s` and every other one lies in `(0, k'^(2j) ln s]`: `0 <= π / (2
/// AGM(1, 4/s)) - ln s <= ln(s) k'² / (1 - k'²)`, below `2^-(W + 8)` for `s >= 2^(W/2 + 8)`.
/// The AGM of positive `a >= b` lies between `b_j` and `a_j` at every step, so the last pair
/// gives a ball of it: their mean, its radius half their distance and their radii.
fn ln_agm(x: &Ball, w: u64, extra: u64) -> Option<Ball> {
    let ex = x.mid.exponent()?;
    // The precision of the steps: the result is `ln s - m ln 2` with `ln s` about `W/2 ln 2`,
    // and the AGM loses a bit or two at each of its `log2 W` steps.
    let lw = 64 - w.leading_zeros() as u64;
    let wb = w + extra + 2 * lw + 16;
    let m = (wb / 2 + 8) as i64 - ex + 1;
    // `4/s = 2^(2 - m) / x`.
    let four = Ball::exact(
        BigFloat::from_parts(
            &crate::math::numbers::Number::from(1),
            2 - m,
            1,
            Round::Nearest,
        )
        .0,
    );
    let mut b = four.div(x, wb)?;
    let mut a = Ball::exact(BigFloat::from(1));
    let tol = Rad::pow2(-i128::from(wb));
    for _ in 0..4 * lw + 64 {
        let d = a.sub(&b, wb);
        let (_, hd) = super::ball::mag_bounds(&d.mid);
        if d.mid.is_zero() || hd.le(tol) {
            let mut g = a.add(&b, wb).shl(-1);
            g.rad = g.rad.add(hd.shl(-1)).add(d.rad);
            let pi = super::consts::pi_approx(wb + 8);
            let pi = Ball::of_limbs(false, &pi.m, pi.lsb, Rad::pow2(pi.err));
            let mut l = pi.div(&g.shl(1), wb)?;
            // The formula's own error: `k'² <= 2^-(W + 12)`, `ln s < 2^bits(W)`.
            let lb = i128::from(64 - wb.leading_zeros());
            l.rad = l.rad.add(Rad::pow2(lb - i128::from(wb) - 11));
            let ml = ln2_ball(wb + 64).mul_float(&BigFloat::from(m), wb + 64);
            return Some(l.sub(&ml, wb));
        }
        let a2 = a.add(&b, wb).shl(-1);
        b = a.mul(&b, wb).sqrt(wb)?;
        a = a2;
    }
    None
}

/// The length of the run of bits equal to `ones` below the top bit of the mantissa `m` (the top
/// bit set): up to its length, which means all of them.
fn leading_run(m: &[u64], ones: bool) -> u64 {
    let mut count = 0u64;
    let mut first = true;
    for &limb in m.iter().rev() {
        let mut x = if ones { !limb } else { limb };
        let mut avail = 64;
        if first {
            x &= !(1u64 << 63);
            // The top bit itself is not counted.
            x <<= 1;
            avail = 63;
            first = false;
            let lz = (x.leading_zeros() as u64).min(avail);
            count += lz;
            if lz < avail {
                return count;
            }
            continue;
        }
        let lz = (x.leading_zeros() as u64).min(avail);
        count += lz;
        if lz < avail {
            return count;
        }
    }
    count
}

/// The ball of `ln 10` to `w` bits, from the cached constant (within `2^(1 - (w + 8))`).
pub(crate) fn ln10_ball(w: u64) -> Ball {
    let a = super::consts::ln10_approx(w + 8);
    Ball::of_limbs(false, &a.m, a.lsb, Rad::pow2(a.err))
}

/// The ball of `ln 2` to `w` bits.
pub(crate) fn ln2_ball(w: u64) -> Ball {
    let a = ln2_approx(w + 8);
    Ball::of_limbs(false, &a.m, a.lsb, Rad::pow2(a.err))
}

/// A result that is exact.
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// `x - 1` exactly, for a finite `x` in `[1/2, 2]`.
fn minus_one(x: &BigFloat) -> BigFloat {
    x.sub_round(&BigFloat::from(1), x.prec + 2, Round::Nearest)
        .0
}

/// The logarithm in base `base` (the ball of `ln base`) of a positive finite `x` that is not 1:
/// `ln x / ln base`, near 1 by its rounding when `x - 1` is too small for the loop.
fn log_of(
    x: &BigFloat,
    prec: u64,
    round: Round,
    base: Option<fn(u64) -> Ball>,
) -> (BigFloat, Ordering) {
    let e = x.exponent().expect("finite");
    // `|x - 1| < 2^-run` for the run of zeros (ones) below the top bit of a mantissa in `[1, 2)`
    // (`[1/2, 1)`): only a long one can be near enough for the rounding near `x - 1`.
    let near = base.is_none()
        && (0..=1).contains(&e)
        && x.parts()
            .is_some_and(|(_, m, _)| leading_run(m, e == 0) > prec);
    if near {
        let d = minus_one(x);
        let ed = i128::from(d.exponent().expect("x is not 1"));
        // `ln(1 + d) = d - d²/2 + …`, below `d`, within `d²` of it for `|d| < 1/2`.
        if ed < -(prec as i128) - 2
            && let Some(r) = round_near(&d, Ordering::Less, 2 * ed, prec, round)
        {
            return r;
        }
    }
    let xb = Ball::exact(x.clone());
    ziv_fast(prec, round, |w| {
        if base.is_none() {
            return match ln_raw(&xb, w + 4) {
                Some((negative, m, lsb, rad)) => {
                    let err = rad.exp2_ceil().max(lsb - 128);
                    if m.is_empty() || err >= lsb + i128::from(limbs::bit_len(&m)) {
                        super::ball::undecided()
                    } else {
                        Approx {
                            negative,
                            m,
                            lsb,
                            err,
                        }
                    }
                }
                None => super::ball::undecided(),
            };
        }
        let l = ln_ball(&xb, w + 4).expect("x > 0");
        match base {
            None => unreachable!(),
            Some(b) => match l.div(&b(w + 8), w + 4) {
                Some(q) => q.approx(),
                None => Ball::new(BigFloat::zero(1), Rad::int(1)).approx(),
            },
        }
    })
}

/// The integer `k` rounded, as a result.
fn integer(k: i64, prec: u64, round: Round) -> (BigFloat, Ordering) {
    BigFloat::from_parts(&Number::from(k), 0, prec, round)
}

impl BigFloat {
    /// The special values of the logarithms: NaN for NaN, `-∞` and numbers below zero; `-∞` for
    /// `±0`; `+∞` for `+∞`; `+0` for 1.
    fn log_special(&self, prec: u64) -> Option<(BigFloat, Ordering)> {
        match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } | Kind::Finite { negative: true, .. } => {
                Some(exact(BigFloat::nan(prec)))
            }
            Kind::Inf { negative: false } => Some(exact(BigFloat::infinity(prec))),
            Kind::Zero { .. } => Some(exact(BigFloat::neg_infinity(prec))),
            Kind::Finite { exp, mant, .. } => {
                if *exp == 1
                    && mant[mant.len() - 1] == 1 << 63
                    && mant[..mant.len() - 1].iter().all(|&l| l == 0)
                {
                    Some(exact(BigFloat::zero(prec)))
                } else {
                    None
                }
            }
        }
    }

    /// The natural logarithm rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    ///
    /// `ln 1 = +0` exactly; `ln ±0 = -∞`; `ln +∞ = +∞`; NaN below zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::from(2).ln_round(53, Round::Nearest);
    /// assert_eq!(l, BigFloat::from(core::f64::consts::LN_2));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn ln_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = self.log_special(prec) {
            return r;
        }
        log_of(self, prec, round, None)
    }

    /// The natural logarithm to nearest at the precision of `self`.
    pub fn ln(&self) -> BigFloat {
        self.ln_round(self.prec, Round::Nearest).0
    }

    /// The logarithm in base 2 rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: exact for a power of two, as are its special values, those of
    /// [`ln_round`](Self::ln_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::from(0.125).log2_round(10, Round::Up);
    /// assert_eq!(l, BigFloat::from(-3));
    /// assert_eq!(o, Ordering::Equal);
    /// ```
    #[track_caller]
    pub fn log2_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = self.log_special(prec) {
            return r;
        }
        let (_, m, _) = self.parts().expect("finite");
        let power = m[m.len() - 1] == 1 << 63 && m[..m.len() - 1].iter().all(|&l| l == 0);
        if power {
            return integer(self.exponent().expect("finite") - 1, prec, round);
        }
        log_of(self, prec, round, Some(ln2_ball))
    }

    /// The logarithm in base 2 to nearest at the precision of `self`.
    pub fn log2(&self) -> BigFloat {
        self.log2_round(self.prec, Round::Nearest).0
    }

    /// The logarithm in base 10 rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: exact for a power of ten, as are its special values, those of
    /// [`ln_round`](Self::ln_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::from(1000).log10_round(8, Round::Down);
    /// assert_eq!(l, BigFloat::from(3));
    /// assert_eq!(o, Ordering::Equal);
    /// ```
    #[track_caller]
    pub fn log10_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = self.log_special(prec) {
            return r;
        }
        if let Some(k) = power_of_ten(self) {
            return integer(k, prec, round);
        }
        log_of(self, prec, round, Some(ln10_ball))
    }

    /// The logarithm in base 10 to nearest at the precision of `self`.
    pub fn log10(&self) -> BigFloat {
        self.log10_round(self.prec, Round::Nearest).0
    }

    /// `ln(1 + self)` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, accurate for `self` near zero, as MPFR's `mpfr_log1p`.
    ///
    /// `±0` for `±0`; `-∞` for `-1`; `+∞` for `+∞`; NaN below `-1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use core::cmp::Ordering;
    ///
    /// // ln(1 + 2^-100) is just below 2^-100.
    /// let (x, _) = BigFloat::from_parts(&Number::from(1), -100, 53, Round::Nearest);
    /// let (l, o) = x.ln_1p_round(53, Round::Down);
    /// assert!(l < x);
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn ln_1p_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } => exact(BigFloat::nan(prec)),
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Zero { negative } => exact(BigFloat::special(
                prec,
                Kind::Zero {
                    negative: *negative,
                },
            )),
            Kind::Finite { .. } => {
                let minus_one = BigFloat::from(-1);
                match self.partial_cmp(&minus_one).expect("not NaN") {
                    Ordering::Less => return exact(BigFloat::nan(prec)),
                    Ordering::Equal => return exact(BigFloat::neg_infinity(prec)),
                    Ordering::Greater => {}
                }
                let ex = i128::from(self.exponent().expect("finite"));
                // `ln(1 + x) = x - x²/2 + …`, below `x`, within `x²` of it for `|x| < 1/2`.
                if ex < -(prec as i128) - 2
                    && let Some(r) = round_near(self, Ordering::Less, 2 * ex, prec, round)
                {
                    return r;
                }
                let x = Ball::exact(self.clone());
                let one = Ball::exact(BigFloat::from(1));
                ziv_fast(prec, round, |w| {
                    // `1 + x` to `w + max(0, -e) + 8` bits: within a relative `2^-(w - e + 8)`,
                    // which changes the logarithm by about `|x| 2^-(w + 8)`.
                    let wu = w + (-ex).max(0) as u64 + 8;
                    let u = one.add(&x, wu);
                    match ln_ball(&u, w + 4) {
                        Some(l) => l.approx(),
                        None => Ball::new(BigFloat::zero(1), Rad::int(1)).approx(),
                    }
                })
            }
        }
    }

    /// `ln(1 + self)` to nearest at the precision of `self`.
    pub fn ln_1p(&self) -> BigFloat {
        self.ln_1p_round(self.prec, Round::Nearest).0
    }
}

/// `k` when `x = 10^k` for an integer `k >= 1`.
fn power_of_ten(x: &BigFloat) -> Option<i64> {
    let (m, e) = x.mantissa_exponent()?;
    if m.is_negative() {
        return None;
    }
    let mut l = limbs::of_number(&m);
    let z = limbs::trailing_zeros(&l);
    l = limbs::shr(&l, z);
    let k = e + z as i64;
    if k < 1 {
        return None;
    }
    // `5^k` has `floor(k log2 5) + 1` bits, `log2 5 = 2.3219…`.
    let bits = limbs::bit_len(&l);
    let lo = (k as u128 * 23219 / 10000) as u64;
    if bits < lo || bits > lo + 2 {
        return None;
    }
    let five = super::exp::pow_u64(&Number::from(5), k as u64);
    (limbs::of_number(&five) == l).then_some(k)
}
