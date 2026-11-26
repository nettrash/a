//! Powers, each part correctly rounded, with MPC's special values and signs of zeros (`mpc_pow`,
//! `mpc_pow_fr`, `mpc_pow_si`).
//!
//! `z^w = e^(w ln z)`, MPC's cases first: `z^0 = 1 ± 0i`; a number with an infinite or NaN part,
//! and a zero `z`, as `e^(w ln z)` of the special values; `1^w = 1 ± 0i`; a real `z` with a real
//! `w` that is an integer, or a `z` not below zero, as the real `x^y` ([`BigFloat::pow_round`])
//! and a zero of the sign MPC's table gives; the powers that are real or imaginary (`(-1)^(n +
//! ti)`, `(-x)^(n + 1/2)`, `(±i)^n`, `(±i)^(n + ti)`, `(t ± ti)^(2n)`) with that part an exact zero.
//!
//! A real `w = m·2^e` (`m` odd) gives an exact power when `z = (c + di)·2^k` has `-e` successive
//! square roots among the Gaussian integers (the dyadic square root of `c + di` exists exactly
//! when `c² + d²` and then `(c + √(c² + d²))/2` are squares) and, for `m < 0`, when `c² + d²` is a
//! power of two: then `z^w` is computed exactly, by squarings and products of Gaussian integers, as
//! long as its parts have at most twice the precision and 64 bits more (a longer exact part is not
//! a number of the precision or a midpoint, which Ziv's rounding decides), and rounded.
//!
//! Otherwise `t = w·ln z` at the working precision (`ln|z|` as [`BigComplex::ln_round`] has it,
//! the argument by [`atan2`](super::real::atan2)) and `e^t`'s parts as `e^(Re t)` times `cos(Im
//! t)` and `sin(Im t)`, Ziv's rounding on each part; the bound of `t` is absolute, so that a large
//! `t` takes more bits. For a tiny `t` the real part is `1 + (e^(Re t) - 1)cos(Im t) - 2 sin²(Im
//! t/2)`, rounded beside 1.

use super::BigComplex;
use super::real::{
    self, Part, Step, add, check, exactly, mul, of, one, pair, set, shl, sin_cos, sqr, step, sub,
    top, zero_of,
};
use crate::math::float::exp::integer_parity;
use crate::math::float::round::round_mag;
use crate::math::float::sincos::Fx;
use crate::math::float::ziv::Approx;
use crate::math::float::{BigFloat, Round, limbs};
use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

type Parts = (Part, Part);

impl BigComplex {
    /// `self^w`, principal branch (`e^(w ln self)`), each part rounded to `prec` bits in the mode
    /// `round`, and the directions of the roundings, as MPC's `mpc_pow`, its exact cases and its
    /// signs of zero parts: `z^0 = 1 ± 0i` (`0^0 = 1 + 0i`); `1^w = 1 ± 0i`; a real `z` and a real
    /// `w` give the real `x^y` of [`BigFloat::pow_round`] when `w` is an integer or `z` is not
    /// below zero; the special values otherwise those of `e^(w ln z)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// let half = BigComplex::from(BigFloat::from(0.5));
    /// let (r, o) = z.pow_round(&half, 53, Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::from(2), BigFloat::from(1))); // exact
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    ///
    /// let i = BigComplex::i(53);
    /// let (r, _) = i.pow_round(&i, 53, Round::Nearest); // i^i = e^(-π/2)
    /// assert_eq!(*r.re(), BigFloat::from(0.2078795763507619));
    /// assert!(r.im().is_zero());
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn pow_round(
        &self,
        w: &BigComplex,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.pow_round_parts(w, (prec, prec), (round, round))
    }

    /// [`pow_round`](BigComplex::pow_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (r, o) = z.pow_round_parts(&BigComplex::from(3), (2, 2), (Round::Up, Round::Down));
    /// assert_eq!(r, BigComplex::new(BigFloat::from(-2), BigFloat::from(2))); // (1 + i)³
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    /// ```
    #[track_caller]
    pub fn pow_round_parts(
        &self,
        w: &BigComplex,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        pair(pow(self, w, check(prec), round))
    }

    /// `self^w`, to nearest at the precisions of `self`'s parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(0), BigFloat::from(2));
    /// assert_eq!(z.pow(&BigComplex::from(2)), BigComplex::from(-4));
    /// ```
    pub fn pow(&self, w: &BigComplex) -> BigComplex {
        self.pow_round_parts(w, self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }

    /// `self^x` for a real `x`, as MPC's `mpc_pow_fr`: [`pow_round`](BigComplex::pow_round) of `x
    /// + 0i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(-4), BigFloat::zero(53));
    /// let (r, _) = z.pow_real_round(&BigFloat::from(0.5), 53, Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::zero(53), BigFloat::from(2)));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn pow_real_round(
        &self,
        x: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.pow_real_round_parts(x, (prec, prec), (round, round))
    }

    /// [`pow_real_round`](BigComplex::pow_real_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(2), BigFloat::from(2));
    /// let (r, o) = z.pow_real_round_parts(&BigFloat::from(-1), (53, 53), (Round::Nearest, Round::Nearest));
    /// assert_eq!(r, BigComplex::new(BigFloat::from(0.25), BigFloat::from(-0.25)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    /// ```
    #[track_caller]
    pub fn pow_real_round_parts(
        &self,
        x: &BigFloat,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let w = BigComplex::new(x.clone(), BigFloat::zero(1));
        self.pow_round_parts(&w, prec, round)
    }

    /// `self^x` for a real `x`, to nearest at the precisions of `self`'s parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(0.0), BigFloat::from(-9.0));
    /// assert_eq!(*z.pow_real(&BigFloat::from(0.5)).re(), BigFloat::from(4.5).sqrt());
    /// ```
    pub fn pow_real(&self, x: &BigFloat) -> BigComplex {
        self.pow_real_round_parts(x, self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }

    /// `self^n` for an integer `n`, as MPC's `mpc_pow_si`: `self` rounded for `n = 1`, `1/self`
    /// ([`div_round`](BigComplex::div_round)) for `n = -1`, the square
    /// ([`sqr_round`](BigComplex::sqr_round)) for `n = 2`; otherwise, for a number with no zero
    /// part, by binary powering at a working precision (the exact power when that does not
    /// decide), and [`pow_round`](BigComplex::pow_round) of `n + 0i` for the others.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// let (r, o) = z.powi_round(5, 64, Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::from(41), BigFloat::from(-38)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn powi_round(
        &self,
        n: i64,
        prec: u64,
        round: Round,
    ) -> (BigComplex, (Ordering, Ordering)) {
        self.powi_round_parts(n, (prec, prec), (round, round))
    }

    /// [`powi_round`](BigComplex::powi_round) with a precision and a mode for each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (r, o) = z.powi_round_parts(-2, (53, 53), (Round::Nearest, Round::Nearest));
    /// assert_eq!(*r.im(), BigFloat::from(-0.5)); // (1 + i)^-2 = -i/2
    /// assert!(r.re().is_zero());
    /// ```
    #[track_caller]
    pub fn powi_round_parts(
        &self,
        n: i64,
        prec: (u64, u64),
        round: (Round, Round),
    ) -> (BigComplex, (Ordering, Ordering)) {
        let prec = check(prec);
        let ordinary = self.is_finite() && !self.re.is_zero() && !self.im.is_zero();
        if ordinary {
            match n {
                1 => {
                    return pair((
                        set(&self.re, prec.0, round.0),
                        set(&self.im, prec.1, round.1),
                    ));
                }
                -1 => return BigComplex::from(1).div_round_parts(self, prec, round),
                2 => return self.sqr_round_parts(prec, round),
                _ => {}
            }
        }
        let w = BigComplex::new(BigFloat::from(n), BigFloat::zero(1));
        let ramified = self.re.abs() == self.im.abs() && n % 2 == 0;
        if ordinary && !ramified && n.unsigned_abs() >= 3 {
            return pair(powi(self, &w, n, prec, round));
        }
        pair(pow(self, &w, prec, round))
    }

    /// `self^n`, to nearest at the precisions of `self`'s parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let i = BigComplex::i(10);
    /// assert_eq!(i.powi(3), BigComplex::new(BigFloat::neg_zero(10), BigFloat::from(-1)));
    /// ```
    pub fn powi(&self, n: i64) -> BigComplex {
        self.powi_round_parts(n, self.prec(), (Round::Nearest, Round::Nearest))
            .0
    }
}

/// `e^(w ln z)` with MPC's rounding of the special values: `ln z` and the product to 2 bits, to
/// nearest.
fn special(z: &BigComplex, w: &BigComplex, p: (u64, u64), r: (Round, Round)) -> Parts {
    let n = (Round::Nearest, Round::Nearest);
    let (u, _) = z.ln_round_parts((2, 2), n);
    let (u, _) = u.mul_round_parts(w, (2, 2), n);
    let (e, o) = u.exp_round_parts(p, r);
    let (a, b) = e.into_parts();
    ((a, o.0), (b, o.1))
}

/// The sign of `|z|² - 1` (0 for a NaN, 1 for an infinity), MPC's `cx1`: exactly, from
/// [`norm_m1`](super::real::norm_m1), whose sign is always known (its sum with the smaller square
/// is exact unless that square lies far below).
fn norm_vs_one(z: &BigComplex) -> Ordering {
    if z.is_infinite() {
        return Ordering::Greater;
    }
    if z.is_nan() {
        return Ordering::Equal;
    }
    let f = |x: &BigFloat| {
        if x.is_zero() {
            real::zero()
        } else {
            real::abs(of(x))
        }
    };
    let d = real::norm_m1(&f(&z.re), &f(&z.im), 64);
    if d.m.is_empty() {
        return Ordering::Equal;
    }
    if d.negative {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// MPC's sign of a zero imaginary part of a real power: `+` for `|z| > 1` with `Im w`'s sign, `-`
/// below 1 with the sign opposite to `Im w`'s, and at `|z| = 1` the sign of `Im z` times that of
/// `Re w`.
fn sign_zi(z: &BigComplex, w: &BigComplex) -> bool {
    match norm_vs_one(z) {
        Ordering::Less => !w.im.is_sign_negative(),
        Ordering::Equal => z.im.is_sign_negative() != w.re.is_sign_negative(),
        Ordering::Greater => w.im.is_sign_negative(),
    }
}

/// MPC's `ymod4` of `fix_sign` for an integer `y`: `|y| mod 4` (for a `y` below zero it takes
/// `4 - (y mod 4)` of the two's complement bits), `None` for a `y` that is not an integer.
fn mod4(y: &BigFloat) -> Option<u8> {
    if y.is_zero() {
        return Some(0);
    }
    let (negative, m, lsb) = y.parts()?;
    let tz = i128::from(limbs::trailing_zeros(m));
    if lsb + tz < 0 {
        return None;
    }
    let v = if lsb >= 2 {
        0
    } else if lsb >= 0 {
        ((m[0] << lsb) & 3) as u8
    } else {
        (limbs::shr(m, (-lsb) as u64).first().copied().unwrap_or(0) & 3) as u8
    };
    let _ = negative;
    Some(v)
}

/// MPC's `fix_sign`: the sign of a zero part of `(±0 + ai)^y` for an integer `y`.
fn fix_sign(re: &mut BigFloat, im: &mut BigFloat, sign_eps: bool, sign_a: bool, y: &BigFloat) {
    let Some(m) = mod4(y) else { return };
    if re.is_zero() {
        if (m == 3 && !sign_eps) || (m == 1 && sign_eps) {
            *re = -&*re;
        }
    } else if im.is_zero() && ((m == 0 && sign_a == sign_eps) || (m == 2 && sign_a != sign_eps)) {
        *im = -&*im;
    }
}

/// MPC's `fix_sign_real`: the sign bit of the zero imaginary part of a real power of a real `z`.
fn sign_real(s1: bool, s2: bool, s3: bool, s4: bool, cmp: Ordering, even: bool) -> bool {
    let (le, ge, lt) = (
        cmp != Ordering::Greater,
        cmp != Ordering::Less,
        cmp == Ordering::Less,
    );
    if !s1 {
        if s2 == s3 {
            false
        } else if !s4 {
            le
        } else {
            !lt
        }
    } else if even {
        if !s4 { le } else { ge }
    } else if !s4 {
        ge
    } else {
        le
    }
}

/// Whether `y·2^k` is an odd integer (MPC's `is_odd`).
fn is_odd(y: &BigFloat, k: i128) -> bool {
    let Some((_, m, lsb)) = y.parts() else {
        return false;
    };
    let tz = i128::from(limbs::trailing_zeros(m));
    lsb + tz + k == 0
}

/// Whether `y` is an integer (zero included).
fn integer(y: &BigFloat) -> bool {
    y.is_zero() || (y.is_finite() && integer_parity(y).is_some())
}

/// `z^w`.
fn pow(z: &BigComplex, w: &BigComplex, p: (u64, u64), r: (Round, Round)) -> Parts {
    let (x, y) = (&z.re, &z.im);
    let x_real = y.is_zero();
    let y_real = w.im.is_zero();
    if y_real && w.re.is_zero() {
        if x_real && x.is_zero() {
            return (
                set(&BigFloat::from(1), p.0, r.0),
                exactly(zero_of(p.1, false)),
            );
        }
        let neg = r.1 == Round::Down || sign_zi(z, w);
        return (
            set(&BigFloat::from(1), p.0, r.0),
            exactly(zero_of(p.1, neg)),
        );
    }
    if !z.is_finite() || !w.is_finite() {
        return special(z, w, p, r);
    }
    let mut z_real = false;
    let mut z_imag = false;
    let mut ramified = false;
    let one = BigFloat::from(1);
    if x_real {
        if x.is_zero() {
            return special(z, w, p, r);
        }
        if *x == one {
            let neg = r.1 == Round::Down || w.re.is_sign_negative() != y.is_sign_negative();
            return (set(&one, p.0, r.0), exactly(zero_of(p.1, neg)));
        }
        let s1 = x.is_sign_negative();
        if y_real && (integer(&w.re) || !s1) {
            let even = s1 && mod4(&w.re).is_some_and(|m| m % 2 == 0);
            let re = x.pow_round(&w.re, p.0, r.0);
            let cmp = x.abs().partial_cmp(&one).expect("finite");
            let s = sign_real(
                s1,
                y.is_sign_negative(),
                w.re.is_sign_negative(),
                w.im.is_sign_negative(),
                cmp,
                even,
            );
            return (re, exactly(zero_of(p.1, s)));
        }
        if *x == -1 && integer(&w.re) {
            z_real = true;
        }
        if x.is_sign_negative() && is_odd(&w.re, 1) && (y_real || *x == -1) {
            z_imag = true;
        }
    } else {
        let unit = x.is_zero() && y.abs() == one;
        if (unit || (x.is_zero() && y_real)) && integer(&w.re) {
            if is_odd(&w.re, 0) {
                z_imag = true;
            } else {
                z_real = true;
            }
        } else if x.abs() == y.abs() && y_real && integer(&w.re) && !is_odd(&w.re, 0) {
            ramified = true;
            if is_odd(&w.re, -1) {
                z_imag = true;
            } else {
                z_real = true;
            }
        }
    }
    let x_imag = x.is_zero();
    // Exact powers first (a real `w`).
    let mut tries = y_real.then_some(2 * p.0.max(p.1) + 64);
    let mut exact = |r: (Round, Round)| -> Option<Parts> {
        let m = tries?;
        match exact_power(z, &w.re, m) {
            Ok(Some(e)) => Some((e.0.round(p.0, r.0), e.1.round(p.1, r.1))),
            Ok(None) => {
                tries = None;
                None
            }
            Err(()) => {
                tries = Some(2 * m);
                None
            }
        }
    };
    if let Some(e) = exact(r) {
        return e;
    }
    let need = (!z_imag, !z_real);
    let (re, im) = match general(z, w, p, r, need, &mut exact) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut re = re.unwrap_or((zero_of(p.0, false), Ordering::Equal));
    let mut im = im.unwrap_or((zero_of(p.1, false), Ordering::Equal));
    if z_real {
        if !ramified {
            if y_real && (x_real || x_imag) {
                im = exactly(zero_of(p.1, false));
                fix_sign(
                    &mut re.0,
                    &mut im.0,
                    x.is_sign_negative(),
                    y.is_sign_negative(),
                    &w.re,
                );
            } else {
                let neg = r.1 == Round::Down || sign_zi(z, w);
                im = exactly(zero_of(p.1, neg));
            }
        }
    } else if z_imag && !ramified && y_real && x_imag {
        re = exactly(zero_of(p.0, false));
        fix_sign(&mut re.0, &mut im.0, x.is_sign_negative(), false, &w.re);
    }
    (re, im)
}

/// An exact part: a signed zero or `±M·2^lsb`.
struct ExactPart {
    negative: bool,
    m: Vec<u64>,
    lsb: i128,
}

impl ExactPart {
    fn of(n: &Number, negative_zero: bool, lsb: i128) -> ExactPart {
        let mut m = limbs::of_number(&n.abs());
        limbs::trim(&mut m);
        ExactPart {
            negative: if n.is_zero() {
                negative_zero
            } else {
                n.is_negative()
            },
            m,
            lsb,
        }
    }

    fn round(&self, prec: u64, round: Round) -> Part {
        if self.m.is_empty() {
            return exactly(zero_of(prec, self.negative));
        }
        round_mag(self.negative, &self.m, self.lsb, false, prec, round)
    }
}

/// The integer of a finite float times `2^-e`, `e` its lowest exponent, and that exponent: `(0,
/// 0)` for a zero.
fn int_of(x: &BigFloat) -> (Number, i128) {
    match x.parts() {
        None => (Number::from(0), 0),
        Some((negative, m, lsb)) => {
            let n = limbs::to_number(m, negative, ByteOrder::LittleEndian);
            (n, lsb)
        }
    }
}

/// The exponent of 2 in a Gaussian integer `a + bi` not zero (the lesser of the parts').
fn twos(a: &Number, b: &Number) -> u64 {
    let ta = a.trailing_zeros().unwrap_or(u64::MAX);
    let tb = b.trailing_zeros().unwrap_or(u64::MAX);
    ta.min(tb)
}

/// The exact `z^y` of a real `y`, MPC's `mpc_pow_exact` (see the [module](self)), as long as its
/// Gaussian integers have at most `maxprec` bits besides the spread of the parts' exponents:
/// `Ok(None)` when it is not a dyadic number, `Err(())` when it takes more bits (try again with a
/// larger `maxprec`).
fn exact_power(
    z: &BigComplex,
    y: &BigFloat,
    maxprec: u64,
) -> Result<Option<(ExactPart, ExactPart)>, ()> {
    let Some(r) = exact_inner(z, y, maxprec) else {
        return Err(());
    };
    Ok(r)
}

/// [`exact_power`]: `None` for more bits than `maxprec`.
fn exact_inner(
    z: &BigComplex,
    y: &BigFloat,
    maxprec: u64,
) -> Option<Option<(ExactPart, ExactPart)>> {
    macro_rules! no {
        ($e:expr) => {
            match $e {
                Some(v) => v,
                None => return Some(None),
            }
        };
    }
    let (my, ly) = int_of(y);
    let ty = no!(my.trailing_zeros());
    let mut my = &my >> ty as usize;
    let mut ey = ly + i128::from(ty);
    let (mut c, lc) = int_of(&z.re);
    let (mut d, ld) = int_of(&z.im);
    let (mut sc, mut sd) = (z.re.is_sign_negative(), z.im.is_sign_negative());
    // `z = (c + di)·2^ec`.
    let mut ec = if z.re.is_zero() {
        ld
    } else if z.im.is_zero() {
        lc
    } else {
        lc.min(ld)
    };
    if !z.re.is_zero() && lc > ec {
        c = &c << (lc - ec) as usize;
    }
    if !z.im.is_zero() && ld > ec {
        d = &d << (ld - ec) as usize;
    }
    // The spread of the exponents comes on top of the bits allowed.
    let maxprec = maxprec + (lc - ld).unsigned_abs().min(u128::from(u64::MAX / 4)) as u64;
    if c.bits().max(d.bits()) > maxprec {
        return None;
    }
    let t = twos(&c, &d);
    c = &c >> t as usize;
    d = &d >> t as usize;
    ec += i128::from(t);
    while ey < 0 {
        if ec & 1 != 0 {
            c = &c << 1;
            d = &d << 1;
            ec -= 1;
        }
        let (a, b) = no!(square_root(&c, &d, sd));
        // The root's real part is `+0` or above zero; the imaginary part keeps `d`'s sign.
        sc = false;
        c = a;
        d = b;
        ec /= 2;
        ey += 1;
    }
    if my.is_negative() {
        // `1/(c + di) = (c - di)/(c² + d²)`, dyadic when `c² + d²` is a power of 2.
        let n = &c * &c + &d * &d;
        let tn = no!(n.trailing_zeros());
        if n.bits() != tn + 1 {
            return Some(None);
        }
        d = -d;
        sd = !sd;
        ec = -ec - i128::from(tn);
        my = -my;
    }
    // `(c + di)^my` by squarings and products from the top bit, then `2^ey` squarings.
    let bits = my.bits();
    let (mut a, mut b) = (c.clone(), d.clone());
    let (mut sa, mut sb) = (sc, sd);
    let mut ed = ec;
    let mut k = bits - 1;
    let ybits = limbs::of_number(&my);
    let bit = |i: u64| -> bool { limbs::bit(&ybits, i) };
    let reduce = |a: &mut Number, b: &mut Number, ed: &mut i128| {
        if a.is_zero() && b.is_zero() {
            return;
        }
        let t = twos(a, b);
        *a = &*a >> t as usize;
        *b = &*b >> t as usize;
        *ed += i128::from(t);
    };
    while k > 0 {
        k -= 1;
        let u = &a * &b;
        let na = &a * &a - &b * &b;
        b = &u << 1;
        a = na;
        sb = sa != sb;
        sa = a.is_negative();
        ed *= 2;
        if ed.unsigned_abs() > 1 << 80 {
            // Far beyond the exponent range: the general rounding over- or underflows.
            return None;
        }
        if bit(k) {
            // MPC's signs of a zero part of the product (its `ta`, `tb`).
            let ta = (sa == sc) && (sb != sd);
            let tb = (sb == sc) && (sa != sd);
            let na = &a * &c - &b * &d;
            let nb = &b * &c + &a * &d;
            a = na;
            b = nb;
            if !a.is_zero() {
                sa = a.is_negative();
            } else {
                sa = !ta;
            }
            if !b.is_zero() {
                sb = b.is_negative();
            } else {
                sa = !tb;
            }
            ed += ec;
        }
        reduce(&mut a, &mut b, &mut ed);
        if a.bits().max(b.bits()) > maxprec {
            return None;
        }
    }
    while ey > 0 {
        if ed == 0 && b.is_zero() && a == 1 {
            // `1^(2^ey) = 1`, and the squarings of `1 + 0i` keep the signs as they are.
            break;
        }
        ey -= 1;
        let u = &a * &b;
        let na = &a * &a - &b * &b;
        b = &u << 1;
        a = na;
        sb = sa != sb;
        sa = a.is_negative();
        ed *= 2;
        if ed.unsigned_abs() > 1 << 80 {
            return None;
        }
        reduce(&mut a, &mut b, &mut ed);
        if a.bits().max(b.bits()) > maxprec {
            return None;
        }
    }
    Some(Some((ExactPart::of(&a, sa, ed), ExactPart::of(&b, sb, ed))))
}

/// The square root `a + bi` of the Gaussian integer `c + di` among the Gaussian integers, MPC's
/// `mpc_perfect_square_p`: for `d = 0` (`c < 0` here) `±√(-c)·i`, the sign of `d`'s zero; else `a =
/// √((c + √(c² + d²))/2)`, `b = d/(2a)`, when those are integers.
fn square_root(c: &Number, d: &Number, sd: bool) -> Option<(Number, Number)> {
    if d.is_zero() {
        if !c.is_negative() {
            // A real `c >= 0` is the real power's case, treated before.
            return None;
        }
        let (s, rem) = (-c).isqrt_rem();
        if !rem.is_zero() {
            return None;
        }
        return Some((Number::from(0), if sd { -s } else { s }));
    }
    if d.trailing_zeros().unwrap_or(0) == 0 {
        return None;
    }
    let (s, rem) = (c * c + d * d).isqrt_rem();
    if !rem.is_zero() {
        return None;
    }
    let t = c + &s;
    if t.trailing_zeros().unwrap_or(u64::MAX) == 0 {
        return None;
    }
    let (a, rem) = (&t >> 1).isqrt_rem();
    if !rem.is_zero() || a.is_zero() {
        return None;
    }
    let (b, rem) = (d >> 1).div_rem(&a);
    if !rem.is_zero() {
        return None;
    }
    Some((a, b))
}

/// `ln z` at the working precision, `z` finite and not zero: `ln|z|` and the argument.
fn ln_fx(z: &BigComplex, w: u64) -> Option<(Fx, Fx)> {
    let (x, y) = (&z.re, &z.im);
    let fx = if x.is_zero() { real::zero() } else { of(x) };
    let fy = if y.is_zero() { real::zero() } else { of(y) };
    let (ax, ay) = (real::abs(fx.clone()), real::abs(fy.clone()));
    let k = ax.exponent().max(ay.exponent());
    let la = if (k == 0 || k == 1) && !(ax.m.is_empty() && ay.m.is_empty()) {
        let d = real::norm_m1(&ax, &ay, w + 16);
        if d.m.is_empty() && d.err.0 == f64::NEG_INFINITY {
            real::zero()
        } else {
            shl(real::ln1p(&d, w + 8)?, -1)
        }
    } else {
        let n = add(&sqr(&ax, w + 8), &sqr(&ay, w + 8), w + 8);
        shl(real::ln(&n, w + 8)?, -1)
    };
    let th = if y.is_zero() {
        if x.is_sign_negative() {
            let mut p = real::pi(w + 8);
            p.negative = y.is_sign_negative();
            p
        } else {
            real::zero()
        }
    } else if x.is_zero() {
        let mut p = shl(real::pi(w + 8), -1);
        p.negative = y.is_sign_negative();
        p
    } else {
        real::atan2(&fy, &fx, w + 8)?
    };
    Some((la, th))
}

/// `(a + bi)(c + di)` at the working precision.
fn cmul(a: &(Fx, Fx), b: &(Fx, Fx), w: u64) -> (Fx, Fx) {
    let re = sub(&mul(&a.0, &b.0, w), &mul(&a.1, &b.1, w), w);
    let im = add(&mul(&a.0, &b.1, w), &mul(&a.1, &b.0, w), w);
    (re, im)
}

/// `z^n` for an ordinary `z` (finite, no zero part, not of equal parts for an even `n`) and
/// `|n| >= 3`, as MPC's `mpc_pow_usi`: by squarings and products from the top bit of `|n|` at the
/// working precision and `bits(n) + 16` more ([`fast_powi`] while the exponents stay far inside the
/// range, otherwise the values of [`real`] with each step's bound carried along), and `1/z^|n| = conj(z^|n|)/|z^|n||²` for `n < 0`; after the first
/// attempt that does not decide, the exact power ([`exact_power`]) with twice the bits each time,
/// as [`pow`] does: `z^n` for `n > 0` is a dyadic number, which Ziv's rounding alone would not
/// reach when a part is one of the precision.
fn powi(z: &BigComplex, wz: &BigComplex, n: i64, p: (u64, u64), r: (Round, Round)) -> Parts {
    let mut tries = Some(2 * p.0.max(p.1) + 64);
    let mut found: Option<Parts> = None;
    let mut first = true;
    let base = (of(&z.re), of(&z.im));
    let m = n.unsigned_abs();
    let bits = 64 - u64::from(m.leading_zeros());
    // The products of the parts stay far inside the exponent range: `|z|^k` for `k <= |n|`.
    let ez =
        z.re.exponent()
            .expect("finite")
            .abs()
            .max(z.im.exponent().expect("finite").abs());
    let fast = (i128::from(ez) + 2) * i128::from(m) < 1 << 60;
    let (re, im) = real::ziv2(p, r, |w, ask| {
        if !first
            && found.is_none()
            && let Some(t) = tries
        {
            match exact_power(z, &wz.re, t) {
                Ok(Some(e)) => found = Some((e.0.round(p.0, r.0), e.1.round(p.1, r.1))),
                Ok(None) => tries = None,
                Err(()) => tries = Some(2 * t),
            }
        }
        first = false;
        if let Some(f) = &found {
            return [Some(Err(f.0.clone())), Some(Err(f.1.clone()))];
        }
        let w2 = w + bits + 16;
        if fast {
            return fast_powi(z, n, w2).map_or(
                [Some(Ok(real::undecided())), Some(Ok(real::undecided()))],
                |(a, b)| [ask[0].then_some(Ok(a)), ask[1].then_some(Ok(b))],
            );
        }
        let mut acc = base.clone();
        for k in (0..bits - 1).rev() {
            acc = cmul(&acc, &acc, w2);
            if (m >> k) & 1 == 1 {
                acc = cmul(&acc, &base, w2);
            }
        }
        if n < 0 {
            let nrm = add(&sqr(&acc.0, w2), &sqr(&acc.1, w2), w2);
            match (real::div(&acc.0, &nrm, w2), real::div(&acc.1, &nrm, w2)) {
                (Some(a), Some(b)) => acc = (a, real::neg(b)),
                _ => return [Some(Ok(real::undecided())), Some(Ok(real::undecided()))],
            }
        }
        let out = [step(&acc.0), step(&acc.1)];
        [
            ask[0].then(|| out[0].clone()),
            ask[1].then(|| out[1].clone()),
        ]
    });
    (re, im)
}

/// `z^n` by the products of [`BigComplex`] at `w` bits, each part rounded to nearest (MPC's
/// `mpc_pow_usi`), with a bound in the norm: a product of numbers within `εa|a|` and `εb|b|` rounded
/// so is within `(εa + εb + εa εb + δ(1 + εa)(1 + εb))|ab|`, `δ = 2^(2 - w)` (each part within half
/// its last place of the product's part, so within `2^-w|ab|`, and `√2` for the norm), so that `z^k`
/// by binary powering is within `(2k - 1)δ·1.1` (by induction: a square of `z^k` gives `(4k - 1)δ`, a
/// product by `z` `2kδ`) and `1/z^m` within `(2m + 1)δ·1.1` more: `2^(bits(m) + 5 - w)` in all. A
/// part's error is then below `2^(e + 1 + bits(m) + 5 - w)` for `|z^n| < 2^(e + 1)`, `e` the larger
/// exponent of the parts. `None` for a part that came out zero or beyond the range.
fn fast_powi(z: &BigComplex, n: i64, w: u64) -> Option<(Approx, Approx)> {
    let m = n.unsigned_abs();
    let bits = 64 - u64::from(m.leading_zeros());
    let nr = (Round::Nearest, Round::Nearest);
    // The products into one number of `w` bits: the first square of `z` itself, exactly.
    let mut acc = BigComplex::zero(w);
    for k in (0..bits - 1).rev() {
        if k == bits - 2 {
            acc.set_sqr(z, Round::Nearest);
        } else {
            acc.sqr_assign_round(Round::Nearest);
        }
        if (m >> k) & 1 == 1 {
            acc.mul_assign_round(z, Round::Nearest);
        }
    }
    if n < 0 {
        acc = BigComplex::from(1).div_round_parts(&acc, (w, w), nr).0;
    }
    let (Some(er), Some(ei)) = (acc.re.exponent(), acc.im.exponent()) else {
        return None;
    };
    let err = i128::from(er.max(ei)) + 1 + i128::from(bits) + 5 - i128::from(w);
    let part = |x: &BigFloat| -> Approx {
        let (negative, m, lsb) = x.parts().expect("finite and not zero");
        Approx {
            negative,
            m: m.to_vec(),
            lsb,
            err,
        }
    };
    Some((part(&acc.re), part(&acc.im)))
}

/// The parts of `e^(w ln z)` that `need` asks for, by Ziv's rounding.
fn general(
    z: &BigComplex,
    w: &BigComplex,
    p: (u64, u64),
    r: (Round, Round),
    need: (bool, bool),
    exact: &mut dyn FnMut((Round, Round)) -> Option<Parts>,
) -> Result<(Option<Part>, Option<Part>), Parts> {
    let a = if w.re.is_zero() {
        real::zero()
    } else {
        of(&w.re)
    };
    let b = if w.im.is_zero() {
        real::zero()
    } else {
        of(&w.im)
    };
    let (pr, pi) = (if need.0 { p.0 } else { 1 }, if need.1 { p.1 } else { 1 });
    let mut found: Option<Parts> = None;
    let mut first = true;
    let (re, im) = real::ziv2((pr, pi), r, |w, ask| {
        let mut out: [Option<Step>; 2] = [None, None];
        // After a first attempt, the exact power with twice the bits each time (MPC's loop).
        if !first && found.is_none() {
            found = exact(r);
        }
        first = false;
        if found.is_some() {
            return [
                Some(Err(exactly(zero_of(1, false)))),
                Some(Err(exactly(zero_of(1, false)))),
            ];
        }
        let ask = [ask[0] && need.0, ask[1] && need.1];
        for (i, n) in [need.0, need.1].into_iter().enumerate() {
            if !n {
                out[i] = Some(Err(exactly(zero_of(1, false))));
            }
        }
        let Some((la, th)) = ln_fx(z, w + 16) else {
            for (i, o) in out.iter_mut().enumerate() {
                if ask[i] {
                    *o = Some(Ok(real::undecided()));
                }
            }
            return out;
        };
        let w2 = w + 16;
        // `t = (a + bi)(la + th·i)`.
        let tr = sub(&mul(&a, &la, w2), &mul(&b, &th, w2), w2);
        let ti = add(&mul(&a, &th, w2), &mul(&b, &la, w2), w2);
        let tiny = 2 * top(&tr).max(top(&ti)) < -i128::from(p.0) - 8;
        if tiny && ask[0] {
            let em = real::expm1(&tr, w).unwrap_or_else(|_| real::useless());
            let (sh, _) = sin_cos(&shl(ti.clone(), -1), w);
            let (_, c) = sin_cos(&ti, w);
            let omc = shl(sqr(&sh, w), 1);
            let d = sub(&mul(&em, &c, w), &omc, w);
            out[0] = Some(real::lead(&one(), &d, w, p.0, r.0));
        }
        let (s, c) = sin_cos(&ti, w2);
        match real::exp(&tr, w2) {
            Ok(e) => {
                if ask[0] && out[0].is_none() {
                    out[0] = Some(step(&mul(&e, &c, w)));
                }
                if ask[1] {
                    out[1] = Some(step(&mul(&e, &s, w)));
                }
            }
            Err(range) => {
                for (i, f) in [&c, &s].into_iter().enumerate() {
                    if ask[i] && out[i].is_none() {
                        out[i] = Some(if real::sign_known(f) {
                            real::range_step(range, f.negative, [p.0, p.1][i], [r.0, r.1][i])
                        } else {
                            Ok(real::undecided())
                        });
                    }
                }
            }
        }
        out
    });
    if let Some(f) = found {
        return Err(f);
    }
    Ok((need.0.then_some(re), need.1.then_some(im)))
}
