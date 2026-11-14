//! The power `x^y`, correctly rounded, with the special values of C99's Annex F and IEEE 754 as
//! MPFR has them, its exact results found first.

use super::ball::{Ball, round_near, ziv_or};
use super::exp::{Range, exp_ball, integer_parity, out_of_range, pow_u64};
use super::limbs;
use super::log::ln_ball;
use super::round::round_mag;
use super::{BigFloat, Kind, Round};
use crate::math::numbers::ByteOrder;
use core::cmp::Ordering;

/// A result that is exact.
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// A float that is finite and not zero as `±M·2^e` with `M` odd: its sign, `M` and `e`.
fn odd_parts(x: &BigFloat) -> (bool, Vec<u64>, i128) {
    let (negative, m, lsb) = x.parts().expect("finite");
    let mut v = m.to_vec();
    limbs::trim(&mut v);
    let z = limbs::trailing_zeros(&v);
    (negative, limbs::shr(&v, z), lsb + i128::from(z))
}

/// `±2^e` rounded, or beyond the range when `e` does not fit.
fn power_of_two(
    negative: bool,
    e: Option<i128>,
    up: bool,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    match e {
        Some(e) => round_mag(negative, &[1], e, false, prec, round),
        None => out_of_range(
            if up { Range::Over } else { Range::Under },
            negative,
            prec,
            round,
        ),
    }
}

/// `x^y` when it is exactly a number of at most `prec + 1` bits (so that its rounding is the
/// rounding of an exact number, a midpoint included), for `x` and `y` finite and not zero, `x >
/// 0` unless `y` is an integer; `negative` is the sign of the result.
///
/// With `x = M 2^e`, `M` odd, and `y = c / 2^t`, `c` odd: for `t = 0`, `x^c` is exact when `M = 1`
/// or `c > 0` and `M^c` fits; for `t > 0`, `x^y` is rational only when `x` is a `2^t`-th power
/// (`c` is odd), so only when `M` is a perfect square `t` times over and `2^t` divides `e`, and
/// then it is `(M^(1/2^t))^c 2^(c e / 2^t)`, which fits as above. As `M >= 3` would need `2^t`
/// of its bits at least, `t` is at most the length of `M` in bits.
fn exact_power(
    x: &BigFloat,
    y: &BigFloat,
    negative: bool,
    prec: u64,
    round: Round,
) -> Option<(BigFloat, Ordering)> {
    let (_, mut mx, mut ex) = odd_parts(x);
    let (ny, my, ey) = odd_parts(y);
    // `t`, the power of two below `y`'s odd mantissa.
    let t = if ey < 0 { (-ey) as u64 } else { 0 };
    if t > 0 {
        if mx != [1] && t > limbs::bit_len(&mx) {
            return None;
        }
        for _ in 0..t {
            if ex % 2 != 0 {
                return None;
            }
            ex /= 2;
            if mx != [1] {
                let n = limbs::to_number(&mx, false, ByteOrder::LittleEndian);
                let (s, r) = n.isqrt_rem();
                if !r.is_zero() {
                    return None;
                }
                mx = limbs::of_number(&s);
            }
        }
    }
    // Now `x^y = (M 2^e)^c'` for the integer `c' = ±c·2^max(ey, 0)`.
    if mx == [1] {
        // `±2^(e c')`, `±1` for `e = 0`.
        if ex == 0 {
            return Some(exact(
                BigFloat::from(if negative { -1 } else { 1 }).with_prec(prec),
            ));
        }
        let c = c_value(&my, ey.max(0));
        let e = c
            .and_then(|c| ex.checked_mul(c))
            .map(|e| if ny { -e } else { e });
        let up = (ex > 0) != ny;
        return Some(power_of_two(negative, e, up, prec, round));
    }
    if ny {
        return None;
    }
    let c = c_value(&my, ey.max(0))?;
    let c = u64::try_from(c).ok()?;
    let bx = limbs::bit_len(&mx);
    // `M^c` has at least `(bx - 1) c + 1` bits.
    if (bx - 1).checked_mul(c)? > prec {
        return None;
    }
    let p = pow_u64(&limbs::to_number(&mx, false, ByteOrder::LittleEndian), c);
    let l = limbs::of_number(&p);
    if limbs::bit_len(&l) > prec + 1 {
        return None;
    }
    let e = ex.checked_mul(i128::from(c))?;
    Some(round_mag(negative, &l, e, false, prec, round))
}

/// `M·2^s` as an `i128` for an odd `M`, when it fits in 100 bits.
fn c_value(m: &[u64], s: i128) -> Option<i128> {
    let bits = i128::from(limbs::bit_len(m));
    if bits + s > 100 {
        return None;
    }
    let mut v: i128 = 0;
    for (i, &l) in m.iter().enumerate().take(2) {
        v |= i128::from(l) << (64 * i);
    }
    Some(v << s)
}

/// Bounds `[2^lo, 2^hi]` of `|ln |x||` for a finite `|x| != 1` that is not zero: from the exponent
/// `e` of `x` when `|x|` lies outside `[1/2, 2)`, `|ln x|` between `min(|e|, |e - 1|) ln 2` and
/// `max(|e|, |e - 1|) ln 2`; from `d = |x| - 1` inside, `|d|/2 <= |ln(1 + d)| <= 2|d|`.
fn ln_bounds(x: &BigFloat) -> (i128, i128) {
    let e = x.exponent().expect("finite");
    if e == 0 || e == 1 {
        let d = x
            .abs()
            .sub_round(&BigFloat::from(1), x.prec + 2, Round::Nearest)
            .0;
        let ed = i128::from(d.exponent().expect("|x| is not 1"));
        (ed - 2, ed + 1)
    } else {
        let (a, b) = (e.unsigned_abs(), (e - 1).unsigned_abs());
        let (lo, hi) = (a.min(b), a.max(b));
        let bits = |v: u64| i128::from(64 - v.leading_zeros());
        (bits(lo) - 2, bits(hi))
    }
}

impl BigFloat {
    /// `self^y` rounded to `prec` bits in the mode `round`, and the direction of the rounding,
    /// as MPFR's `mpfr_pow`.
    ///
    /// The special values are those of C99's Annex F and IEEE 754: `x^±0 = 1` and `1^y = 1` for
    /// any `x` and `y`, NaN included; NaN for a finite `x < 0` and a finite `y` that is not an
    /// integer; `(-1)^±∞ = 1`; `x^±∞` and `±0^y`, `±∞^y` by the magnitude of `x` and the sign and
    /// the parity of `y`. A result that is exact (`4^0.5 = 2`, `2^-3`, `(-3)^3`) is found before
    /// any approximation, and one far beyond the exponent range overflows or underflows at once.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let four = BigFloat::from(4);
    /// assert_eq!(four.pow_round(&BigFloat::from(0.5), 10, Round::Up), (BigFloat::from(2), Ordering::Equal));
    /// assert_eq!(BigFloat::from(-2).pow_round(&BigFloat::from(-3), 10, Round::Up), (BigFloat::from(-0.125), Ordering::Equal));
    /// let (r, o) = BigFloat::from(2).pow_round(&BigFloat::from(0.5), 53, Round::Nearest);
    /// assert_eq!(r, BigFloat::from(core::f64::consts::SQRT_2));
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn pow_round(&self, y: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let one = || exact(BigFloat::from(1).with_prec(prec));
        if y.is_zero() {
            return one();
        }
        let x_one = self.is_normal() && *self == 1;
        if x_one {
            return one();
        }
        if self.is_nan() || y.is_nan() {
            return exact(BigFloat::nan(prec));
        }
        let ny = y.is_sign_negative();
        if y.is_infinite() {
            if self.is_normal() && *self == -1 {
                return one();
            }
            // `|x| < 1` (zeros included) or `|x| > 1`.
            let small = self.is_zero() || (self.is_normal() && self.abs() < 1);
            return if small != ny {
                exact(BigFloat::zero(prec))
            } else {
                exact(BigFloat::infinity(prec))
            };
        }
        // `y` finite, not zero: is it an odd integer?
        let parity = integer_parity(y);
        let odd = parity == Some(true);
        match &self.kind {
            Kind::Zero { negative } => {
                let neg = *negative && odd;
                return if ny {
                    exact(BigFloat::special(prec, Kind::Inf { negative: neg }))
                } else {
                    exact(BigFloat::special(prec, Kind::Zero { negative: neg }))
                };
            }
            Kind::Inf { negative } => {
                let neg = *negative && odd;
                return if ny {
                    exact(BigFloat::special(prec, Kind::Zero { negative: neg }))
                } else {
                    exact(BigFloat::special(prec, Kind::Inf { negative: neg }))
                };
            }
            _ => {}
        }
        let nx = self.is_sign_negative();
        if nx && parity.is_none() {
            return exact(BigFloat::nan(prec));
        }
        let negative = nx && odd;
        if let Some(r) = exact_power(self, y, negative, prec, round) {
            return r;
        }
        // `|t| = |y ln|x||` within `[2^(ey - 1 + lo), 2^(ey + hi)]`, of the sign of `y` and of
        // `ln |x|`.
        let ax = self.abs();
        let (lo, hi) = ln_bounds(&ax);
        let ey = i128::from(y.exponent().expect("finite"));
        let up = (ax > 1) != ny;
        if ey - 1 + lo >= 63 {
            return out_of_range(
                if up { Range::Over } else { Range::Under },
                negative,
                prec,
                round,
            );
        }
        // `x^y = ±e^t` within `2|t| < 2^(ey + hi + 1)` of `±1`.
        if ey + hi + 1 < -(prec as i128) - 1 {
            let side = if up != negative {
                Ordering::Greater
            } else {
                Ordering::Less
            };
            let unit = BigFloat::from(if negative { -1 } else { 1 });
            if let Some(r) = round_near(&unit, side, ey + hi + 1, prec, round) {
                return r;
            }
        }
        // `t` to `w + max(0, ey + hi)` bits gives `|t|` units of `2^-w` at most.
        let more = (ey + hi).max(0) as u64;
        let xb = Ball::exact(ax);
        ziv_or(prec, round, |w| {
            let wl = w + more + 8;
            let l = ln_ball(&xb, wl).expect("|x| > 0");
            let t = l.mul_float(y, wl);
            match exp_ball(&t, w) {
                Ok(raw) => Ok(raw.approx(negative)),
                Err(r) => Err(out_of_range(r, negative, prec, round)),
            }
        })
    }

    /// `self^y` to nearest at the precision of `self`.
    pub fn pow(&self, y: &BigFloat) -> BigFloat {
        self.pow_round(y, self.prec, Round::Nearest).0
    }
}
