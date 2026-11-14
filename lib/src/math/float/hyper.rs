//! The hyperbolic functions and their inverses, correctly rounded: `sinh`, `cosh`, `tanh` from the
//! exponential, without cancellation (`sinh x = (u + u/(u + 1))/2` and `tanh x = u/(u + 2)` for
//! `u = e^|x| - 1` or `e^2|x| - 1`), and `asinh`, `acosh`, `atanh` from the logarithm (through
//! `ln(1 + z)` near zero).

use super::ball::{Ball, Rad, round_near, undecided, ziv_fast, ziv_or};
use super::exp::{Range, Raw, exp_ball, expo, out_of_range};
use super::log::{ln_ball, ln2_ball};

use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// The rounding of an approximation when it decides it.
fn decide(a: &super::ziv::Approx, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    super::ball::fast_round(a, prec, round).or_else(|| super::ziv::round_approx(a, prec, round))
}

/// A result that is exact.
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// The side away from zero for a number of the sign `negative`.
#[inline]
fn away(negative: bool) -> Ordering {
    if negative {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// The ball of `e^a - 1` for a finite `a > 0` below `2^62`, to `w` bits about: from `e^a` with
/// as many more bits as `a` has zeros after the point, the cancellation of the subtraction.
fn expm1_ball(a: &Ball, w: u64) -> Option<Ball> {
    let ex = expo(&a.mid);
    let more = (-ex).max(0) as u64;
    let raw = exp_ball(a, w + more + 8).ok()?;
    raw.ball()
        .map(|e| e.sub(&Ball::exact(BigFloat::from(1)), w + 8))
}

/// `ln(1 + z)` for the ball `z >= 0` (or `z > -1`), to `w` bits about: `1 + z` to as many more
/// bits as `z` has zeros after the point.
fn ln1p_ball(z: &Ball, w: u64) -> Option<Ball> {
    let ez = z.mid.exponent().unwrap_or(0);
    let more = (-ez).max(0) as u64;
    let u = Ball::exact(BigFloat::from(1)).add(z, w + more + 8);
    ln_ball(&u, w + 4)
}

/// `e^|x| / 2` for a large `|x|` (the other term below `2^-(w + 16)` of it): the raw exponential
/// halved, its error bound widened by that term.
fn half_exp(a: &Ball, w: u64) -> Result<Raw, Range> {
    let mut raw = exp_ball(a, w)?;
    raw.lsb -= 1;
    raw.rel = raw.rel.add(Rad::pow2(-i128::from(w) - 16));
    Ok(raw)
}

impl BigFloat {
    /// The hyperbolic sine rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding: `±0` and `±∞` for themselves.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let (s, _) = BigFloat::from(1).sinh_round(53, Round::Nearest);
    /// assert_eq!(s, BigFloat::from(1.1752011936438014));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sinh_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let negative = match &self.kind {
            Kind::Nan | Kind::Inf { .. } | Kind::Zero { .. } => {
                return exact(self.with_prec(prec));
            }
            Kind::Finite { negative, .. } => *negative,
        };
        let ex = expo(self);
        if ex >= 63 {
            return out_of_range(Range::Over, negative, prec, round);
        }
        // `sinh x = x + x³/6 + …`, beyond `x`, within `|x|³/5 < 2^(3e - 2)` of it.
        if 2 * ex < -(prec as i64)
            && let Some(r) = round_near(self, away(negative), 3 * i128::from(ex) - 2, prec, round)
        {
            return r;
        }
        let a = Ball::exact(self.abs());
        let big = self.abs();
        ziv_or(prec, round, |w| {
            if big >= w + 16 {
                return match half_exp(&a, w) {
                    Ok(raw) => Ok(raw.approx(negative)),
                    Err(r) => Err(out_of_range(r, negative, prec, round)),
                };
            }
            let Some(u) = expm1_ball(&a, w + 4) else {
                return Ok(undecided());
            };
            let v = u.add(&Ball::exact(BigFloat::from(1)), w + 8);
            let Some(q) = u.div(&v, w + 8) else {
                return Ok(undecided());
            };
            let s = u.add(&q, w + 8).shl(-1);
            Ok(if negative { s.neg() } else { s }.approx())
        })
    }

    /// The hyperbolic sine to nearest at the precision of `self`.
    pub fn sinh(&self) -> BigFloat {
        self.sinh_round(self.prec, Round::Nearest).0
    }

    /// The hyperbolic sine and cosine, each rounded to `prec` bits in the mode `round`, and the
    /// directions of the roundings: as [`sinh_round`](Self::sinh_round) and
    /// [`cosh_round`](Self::cosh_round), from one exponential at each working precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let x = BigFloat::from(0.75);
    /// let (s, c) = x.sinh_cosh_round(80, Round::Up);
    /// assert_eq!(s, x.sinh_round(80, Round::Up));
    /// assert_eq!(c, x.cosh_round(80, Round::Up));
    /// ```
    #[track_caller]
    pub fn sinh_cosh_round(
        &self,
        prec: u64,
        round: Round,
    ) -> ((BigFloat, Ordering), (BigFloat, Ordering)) {
        let prec = BigFloat::check_prec(prec);
        let joint = self.is_normal() && {
            let ex = expo(self);
            ex <= 40 && 2 * ex >= -(prec as i64) && self.abs() < prec + 16
        };
        if !joint {
            return (self.sinh_round(prec, round), self.cosh_round(prec, round));
        }
        let negative = self.is_sign_negative();
        let a = Ball::exact(self.abs());
        let one = Ball::exact(BigFloat::from(1));
        let more = (-expo(self)).max(0) as u64;
        let (mut sh, mut ch) = (None, None);
        let mut w = prec + super::ziv::guard(prec);
        loop {
            if let Ok(raw) = exp_ball(&a, w + more + 12)
                && let Some(e) = raw.ball()
            {
                let wb = w + 8;
                if sh.is_none() {
                    let u = e.sub(&one, wb);
                    if let Some(q) = u.div(&u.add(&one, wb), wb) {
                        let s = u.add(&q, wb).shl(-1);
                        let s = if negative { s.neg() } else { s };
                        sh = decide(&s.approx(), prec, round);
                    }
                }
                if ch.is_none()
                    && let Some(inv) = one.div(&e, wb)
                {
                    ch = decide(&e.add(&inv, wb).shl(-1).approx(), prec, round);
                }
            }
            if let (Some(s), Some(c)) = (&sh, &ch) {
                return (s.clone(), c.clone());
            }
            w += (w / 2).max(64);
        }
    }

    /// The hyperbolic cosine rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: 1 for `±0`, `+∞` for `±∞`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let (c, _) = BigFloat::from(1).cosh_round(53, Round::Nearest);
    /// assert_eq!(c, BigFloat::from(1.5430806348152437));
    /// ```
    #[track_caller]
    pub fn cosh_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan => return exact(BigFloat::nan(prec)),
            Kind::Inf { .. } => return exact(BigFloat::infinity(prec)),
            Kind::Zero { .. } => return exact(BigFloat::from(1).with_prec(prec)),
            Kind::Finite { .. } => {}
        }
        let ex = expo(self);
        if ex >= 63 {
            return out_of_range(Range::Over, false, prec, round);
        }
        // `cosh x = 1 + x²/2 + …`, above 1 by less than `x² < 2^(2e)`.
        if 2 * ex < -(prec as i64) - 2
            && let Some(r) = round_near(
                &BigFloat::from(1),
                Ordering::Greater,
                2 * i128::from(ex),
                prec,
                round,
            )
        {
            return r;
        }
        let a = Ball::exact(self.abs());
        let big = self.abs();
        ziv_or(prec, round, |w| {
            if big >= w + 16 {
                return match half_exp(&a, w) {
                    Ok(raw) => Ok(raw.approx(false)),
                    Err(r) => Err(out_of_range(r, false, prec, round)),
                };
            }
            let raw = match exp_ball(&a, w + 4) {
                Ok(raw) => raw,
                Err(r) => return Err(out_of_range(r, false, prec, round)),
            };
            let Some(e) = raw.ball() else {
                return Ok(undecided());
            };
            let Some(inv) = Ball::exact(BigFloat::from(1)).div(&e, w + 8) else {
                return Ok(undecided());
            };
            Ok(e.add(&inv, w + 8).shl(-1).approx())
        })
    }

    /// The hyperbolic cosine to nearest at the precision of `self`.
    pub fn cosh(&self) -> BigFloat {
        self.cosh_round(self.prec, Round::Nearest).0
    }

    /// The hyperbolic tangent rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: `±0` for `±0`, `±1` for `±∞`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (t, o) = BigFloat::from(100).tanh_round(53, Round::Down);
    /// assert_eq!(t, BigFloat::from(1.0 - f64::EPSILON / 2.0));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn tanh_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let negative = match &self.kind {
            Kind::Nan => return exact(BigFloat::nan(prec)),
            Kind::Inf { negative } => {
                return exact(BigFloat::from(if *negative { -1 } else { 1 }).with_prec(prec));
            }
            Kind::Zero { .. } => return exact(self.with_prec(prec)),
            Kind::Finite { negative, .. } => *negative,
        };
        let ex = expo(self);
        let toward = away(!negative);
        // `tanh x = x - x³/3 + …`, toward zero from `x`, within `|x|³/3 < 2^(3e - 1)`.
        if 2 * ex < -(prec as i64)
            && let Some(r) = round_near(self, toward, 3 * i128::from(ex) - 1, prec, round)
        {
            return r;
        }
        // For `|x| >= 2^(e - 1)`, `1 - tanh |x| < 2 e^(-2|x|) <= 2^(1 - 2^e)`.
        if ex >= 1 && (ex >= 62 || (1u64 << ex) >= prec + 4) {
            let delta = if ex >= 62 {
                -(1i128 << 62)
            } else {
                1 - (1i128 << ex)
            };
            let unit = BigFloat::from(if negative { -1 } else { 1 });
            if let Some(r) = round_near(&unit, toward, delta, prec, round) {
                return r;
            }
        }
        let two_a = Ball::exact(self.abs()).shl(1);
        ziv_fast(prec, round, |w| {
            let Some(u) = expm1_ball(&two_a, w + 4) else {
                return undecided();
            };
            let d = u.add(&Ball::exact(BigFloat::from(2)), w + 8);
            match u.div(&d, w + 8) {
                Some(t) => if negative { t.neg() } else { t }.approx(),
                None => undecided(),
            }
        })
    }

    /// The hyperbolic tangent to nearest at the precision of `self`.
    pub fn tanh(&self) -> BigFloat {
        self.tanh_round(self.prec, Round::Nearest).0
    }

    /// The inverse hyperbolic sine rounded to `prec` bits in the mode `round`, and the direction
    /// of the rounding: `±0` and `±∞` for themselves.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let (s, _) = BigFloat::from(1).asinh_round(53, Round::Nearest);
    /// assert_eq!(s, BigFloat::from(0.881373587019543));
    /// ```
    #[track_caller]
    pub fn asinh_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let negative = match &self.kind {
            Kind::Nan | Kind::Inf { .. } | Kind::Zero { .. } => {
                return exact(self.with_prec(prec));
            }
            Kind::Finite { negative, .. } => *negative,
        };
        let ex = expo(self);
        // `asinh x = x - x³/6 + …`, toward zero from `x`, within `|x|³/6`.
        if 2 * ex < -(prec as i64)
            && let Some(r) = round_near(self, away(!negative), 3 * i128::from(ex) - 2, prec, round)
        {
            return r;
        }
        let a = Ball::exact(self.abs());
        let one = Ball::exact(BigFloat::from(1));
        ziv_fast(prec, round, |w| {
            let wb = w + 8;
            let r = if ex > (w / 2 + 8) as i64 {
                // `asinh a = ln a + ln 2 + δ`, `0 < δ < 1/(4a²) <= 2^(-2e)`.
                let Some(l) = ln_ball(&a, wb) else {
                    return undecided();
                };
                let mut s = l.add(&ln2_ball(wb), wb);
                s.rad = s.rad.add(Rad::pow2(-2 * i128::from(ex)));
                s
            } else if ex >= 1 {
                // `ln(a + sqrt(a² + 1))`, `a >= 1`.
                let Some(r) = a.mul(&a, wb).add(&one, wb).sqrt(wb) else {
                    return undecided();
                };
                match ln_ball(&a.add(&r, wb), wb) {
                    Some(l) => l,
                    None => return undecided(),
                }
            } else {
                // `ln(1 + a + a²/(1 + sqrt(1 + a²)))`, the terms of the same sign.
                let a2 = a.mul(&a, wb);
                let Some(r) = a2.add(&one, wb).sqrt(wb) else {
                    return undecided();
                };
                let Some(q) = a2.div(&r.add(&one, wb), wb) else {
                    return undecided();
                };
                match ln1p_ball(&a.add(&q, wb), wb) {
                    Some(l) => l,
                    None => return undecided(),
                }
            };
            if negative { r.neg() } else { r }.approx()
        })
    }

    /// The inverse hyperbolic sine to nearest at the precision of `self`.
    pub fn asinh(&self) -> BigFloat {
        self.asinh_round(self.prec, Round::Nearest).0
    }

    /// The inverse hyperbolic cosine rounded to `prec` bits in the mode `round`, and the
    /// direction of the rounding: `+0` for 1, `+∞` for `+∞`, NaN below 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let (c, _) = BigFloat::from(2).acosh_round(53, Round::Nearest);
    /// assert_eq!(c, BigFloat::from(1.3169578969248168));
    /// ```
    #[track_caller]
    pub fn acosh_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } | Kind::Zero { .. } => {
                return exact(BigFloat::nan(prec));
            }
            Kind::Inf { negative: false } => return exact(BigFloat::infinity(prec)),
            Kind::Finite { .. } => {}
        }
        match self.partial_cmp(&BigFloat::from(1)).expect("not NaN") {
            Ordering::Less => return exact(BigFloat::nan(prec)),
            Ordering::Equal => return exact(BigFloat::zero(prec)),
            Ordering::Greater => {}
        }
        let ex = expo(self);
        let x = Ball::exact(self.clone());
        let one = Ball::exact(BigFloat::from(1));
        // `x - 1`, exactly, for `x < 2`.
        let d = (ex == 1).then(|| {
            Ball::exact(
                self.sub_round(&BigFloat::from(1), self.prec + 2, Round::Nearest)
                    .0,
            )
        });
        ziv_fast(prec, round, |w| {
            let wb = w + 8;
            let r = if ex > (w / 2 + 8) as i64 {
                // `acosh x = ln x + ln 2 - δ`, `0 < δ <= 1/x² <= 2^(2 - 2e)`.
                let Some(l) = ln_ball(&x, wb) else {
                    return undecided();
                };
                let mut s = l.add(&ln2_ball(wb), wb);
                s.rad = s.rad.add(Rad::pow2(3 - 2 * i128::from(ex)));
                s
            } else if let Some(d) = &d {
                // `ln(1 + d + sqrt(d (d + 2)))`.
                let Some(r) = d
                    .mul(&d.add(&Ball::exact(BigFloat::from(2)), wb), wb)
                    .sqrt(wb)
                else {
                    return undecided();
                };
                match ln1p_ball(&d.add(&r, wb), wb) {
                    Some(l) => l,
                    None => return undecided(),
                }
            } else {
                // `ln(x + sqrt(x² - 1))`, `x >= 2`.
                let Some(r) = x.mul(&x, wb).sub(&one, wb).sqrt(wb) else {
                    return undecided();
                };
                match ln_ball(&x.add(&r, wb), wb) {
                    Some(l) => l,
                    None => return undecided(),
                }
            };
            r.approx()
        })
    }

    /// The inverse hyperbolic cosine to nearest at the precision of `self`.
    pub fn acosh(&self) -> BigFloat {
        self.acosh_round(self.prec, Round::Nearest).0
    }

    /// The inverse hyperbolic tangent rounded to `prec` bits in the mode `round`, and the
    /// direction of the rounding: `±0` for `±0`, `±∞` for `±1`, NaN beyond.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let (t, _) = BigFloat::from(0.5).atanh_round(53, Round::Nearest);
    /// assert_eq!(t, BigFloat::from(0.5493061443340549));
    /// ```
    #[track_caller]
    pub fn atanh_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let negative = match &self.kind {
            Kind::Nan | Kind::Inf { .. } => return exact(BigFloat::nan(prec)),
            Kind::Zero { .. } => return exact(self.with_prec(prec)),
            Kind::Finite { negative, .. } => *negative,
        };
        match self.abs().partial_cmp(&BigFloat::from(1)).expect("not NaN") {
            Ordering::Greater => return exact(BigFloat::nan(prec)),
            Ordering::Equal => {
                return exact(BigFloat::special(prec, Kind::Inf { negative }));
            }
            Ordering::Less => {}
        }
        let ex = expo(self);
        // `atanh x = x + x³/3 + …`, beyond `x`, within `|x|³/2 < 2^(3e - 1)` for `|x| < 1/2`.
        if 2 * ex < -(prec as i64)
            && let Some(r) = round_near(self, away(negative), 3 * i128::from(ex) - 1, prec, round)
        {
            return r;
        }
        let a = Ball::exact(self.abs());
        let one = Ball::exact(BigFloat::from(1));
        // `1 - |x|`, exactly when `|x| >= 1/2`, which is where it cancels.
        let pa = self.prec;
        ziv_fast(prec, round, |w| {
            let wb = w + 8;
            let c = one.sub(&a, wb.max(pa + 2));
            let Some(z) = a.shl(1).div(&c, wb) else {
                return undecided();
            };
            match ln1p_ball(&z, wb) {
                Some(l) => {
                    let h = l.shl(-1);
                    if negative { h.neg() } else { h }.approx()
                }
                None => undecided(),
            }
        })
    }

    /// The inverse hyperbolic tangent to nearest at the precision of `self`.
    pub fn atanh(&self) -> BigFloat {
        self.atanh_round(self.prec, Round::Nearest).0
    }
}
