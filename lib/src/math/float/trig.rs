//! The sine, the cosine, the tangent and their reciprocals, correctly rounded, with MPFR's special
//! values: each is NaN for NaN and the infinities; at `±0` the sine, the tangent are `±0`, the
//! cosine and the secant 1, the cosecant and the cotangent `±∞`, all exact.
//!
//! The argument is reduced modulo `π/2`, `x = kπ/2 + r`, `|r| <= π/4`, with π to the bits that
//! the exponent of `x` and the precision need, and again with more when `r` comes out so small
//! that the cancellation leaves too few of them (see [`reduce`]); the kernels of
//! [`sincos`](super::sincos) give `sin r` and `cos r` with bounds of their errors, and Ziv's
//! rounding decides. A small argument, whose function lies within less than half an ulp of the
//! argument (or of 1), is rounded directly by the side it lies on (see [`near`]).

use super::arith::Mag;
use super::consts::pi_approx;
use super::limbs;
use super::round::round_mag;
use super::short;
use super::sincos::{self, Fx, Lg};
use super::ziv::{Approx, guard, round_approx};
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// A result that is exact: `x` and [`Ordering::Equal`].
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// The value `±M·2^lsb` (`m` not zero) moved by less than `2^err` toward zero (`away` false) or
/// away from it, rounded to `prec` bits in the mode `round`, when that is decided: when `2^err` is
/// at most a unit of the scale `2^k` at which `M·2^lsb` has two bits more than the precision, the
/// value lies strictly between `M'·2^k` and the next or the last unit, `M' = M·2^(lsb - k)`, and
/// rounds as such a value does, sticky. `None` otherwise.
pub(super) fn near(
    negative: bool,
    m: &[u64],
    lsb: i128,
    away: bool,
    err: i128,
    prec: u64,
    round: Round,
) -> Option<(BigFloat, Ordering)> {
    let e = lsb + i128::from(limbs::bit_len(m));
    let k = lsb.min(e - i128::from(prec) - 2);
    if err > k {
        return None;
    }
    let mut big = limbs::shl(m, (lsb - k) as u64);
    if !away {
        big = limbs::sub(&big, &[1]);
    }
    Some(round_mag(negative, &big, k, true, prec, round))
}

/// Whether the bits `lo..=hi` of `m` are neither all zeros nor all ones (bits beyond `m` count as
/// zeros); `false` for an empty range.
fn mixed(m: &[u64], lo: i128, hi: i128) -> bool {
    let bit = |i: i128| -> bool { i >= 0 && limbs::bit(m, i as u64) };
    hi > lo && {
        let first = bit(lo);
        (lo + 1..=hi).any(|i| bit(i) != first)
    }
}

/// The correct rounding of the value that `a` approximates when the test is cheap: with `M =
/// m·2^lsb` of the exponent `e`, the breakpoints of every mode at `prec` bits are multiples of
/// `2^h`, `h = e - prec - 1`. When the bits `err + 1..h - 1` of `M` are neither all zeros nor all
/// ones, `M mod 2^h` lies in `(2^err, 2^h - 2^err)`, so the whole interval `M ± 2^err` lies
/// strictly between two consecutive multiples of `2^h`, and the value rounds as `floor(M/2^h)`
/// with a sticky bit. `None` when the test fails.
fn round_cheap(a: &Approx, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    if a.m.is_empty() {
        return None;
    }
    let e = a.lsb + i128::from(limbs::bit_len(&a.m));
    let h = e - i128::from(prec) - 1;
    if !mixed(&a.m, a.err + 1 - a.lsb, h - 1 - a.lsb) {
        return None;
    }
    let q = if h >= a.lsb {
        limbs::shr(&a.m, (h - a.lsb) as u64)
    } else {
        limbs::shl(&a.m, (a.lsb - h) as u64)
    };
    Some(round_mag(a.negative, &q, h, true, prec, round))
}

/// Ziv's rounding as [`ziv`] does it, each attempt first by [`round_cheap`] and then by
/// [`round_approx`].
pub(super) fn ziv_fast(
    prec: u64,
    round: Round,
    mut approx: impl FnMut(u64) -> Approx,
) -> (BigFloat, Ordering) {
    let mut w = prec + guard(prec);
    loop {
        let a = approx(w);
        if let Some(r) = round_cheap(&a, prec, round).or_else(|| round_approx(&a, prec, round)) {
            return r;
        }
        w += (w / 2).max(64);
    }
}

/// The argument reduced modulo `π/2`: `|x| = kπ/2 + r` with `r = ±R·2^-g`, `|r| <= π/4` (or `r =
/// |x|` for `|x| < 1`), `q = k mod 4`, and the bound of the error of `r` in units of `2^-g`.
pub(super) struct Reduced {
    /// `k mod 4`.
    pub(super) q: u8,
    /// Whether `r` is below zero.
    pub(super) negative: bool,
    /// `|r|·2^g`.
    pub(super) r: Vec<u64>,
    /// The bits after the point of `r`.
    pub(super) g: i128,
    /// The bound of the error of `r`, in units of `2^-g`.
    pub(super) err: Lg,
}

/// `|x| = M·2^lsb` reduced modulo `π/2` with `r` to about `w` bits relatively.
///
/// For `|x| < 1`, `r = |x|` exactly. Otherwise with `π ≈ P·2^lp`, `|π - P·2^lp| <= 2^ep`
/// ([`pi_approx`] at `W` bits after the point), `N = floor(|x|·2^(1 - lp))` and `k, R` the
/// quotient and remainder of `N` by `P`, `k` rounded to nearest (`R` replaced by `P - R`, of the
/// other sign, when `2R > P`): `|x| - kπ/2 = R·2^(lp - 1) - k(π/2 - P·2^(lp - 1)) + (|x| -
/// N·2^(lp - 1))`, an error of at most `k·2^(ep - 1)`, plus `2^(lp - 1)` when `N` dropped bits.
/// `W = w + e + 16` for the exponent `e` of `x` leaves `w + 16` bits of `r` when it is near
/// `π/4`; when the remainder comes out smaller, the bits lost to the cancellation are known from
/// it, and the reduction is done again with that many more (π is irrational: `r` is not zero).
/// The cost grows with the exponent of `x`, the length of π it needs.
pub(super) fn reduce(m: &[u64], lsb: i128, w: u64) -> Reduced {
    let e = lsb + i128::from(limbs::bit_len(m));
    if e <= 0 {
        return Reduced {
            q: 0,
            negative: false,
            r: m.to_vec(),
            g: -lsb,
            err: Lg::ZERO,
        };
    }
    let mut extra: u64 = 0;
    loop {
        let wp = w + e as u64 + 16 + extra;
        let pi = pi_approx(wp);
        let lp = pi.lsb;
        let s = lsb + 1 - lp;
        let (n, cut) = if s >= 0 {
            (limbs::shl(m, s as u64), false)
        } else {
            let d = s.unsigned_abs() as u64;
            (limbs::shr(m, d), !limbs::zero_below(m, d))
        };
        let (k, rem) = sincos::div_rem(&n, &pi.m);
        let up = limbs::cmp(&limbs::shl(&rem, 1), &pi.m) == Ordering::Greater;
        let (r, negative) = if up {
            (limbs::sub(&pi.m, &rem), true)
        } else {
            (rem, false)
        };
        let q = (k.first().copied().unwrap_or(0).wrapping_add(u64::from(up)) & 3) as u8;
        let g = 1 - lp;
        // `k·2^(ep - 1)` in units of `2^-g = 2^(lp - 1)`: `k·2^(ep - lp)`.
        let kbits = i128::from(limbs::bit_len(&k)) + i128::from(up);
        let mut err = if kbits == 0 {
            Lg::ZERO
        } else {
            Lg::pow2(kbits + pi.err - lp)
        };
        if cut {
            err = err.add(Lg(0.0));
        }
        if !r.is_empty() {
            let rel = limbs::bit_len(&r) as f64 - err.0;
            if rel >= (w + 8) as f64 {
                return Reduced {
                    q,
                    negative,
                    r,
                    g,
                    err,
                };
            }
            extra += ((w + 40) as f64 - rel).max(64.0) as u64;
        } else {
            extra += extra.max(64);
        }
    }
}

/// What the functions take of the reduced argument: `sin r`, `cos r` or both, and the reduction.
pub(super) struct Eval {
    pub(super) q: u8,
    pub(super) negative: bool,
    pub(super) sin: Option<Fx>,
    pub(super) cos: Option<Fx>,
}

/// The fixed-point limbs of the kernels for a sine of `r` of the exponent `er` to `w` bits
/// relatively, or a cosine to `w` bits: `w + 24` bits when the series takes `r` as it is, and the
/// `2K + 8` bits more that the doublings lose otherwise.
pub(super) fn kernel_limbs(w: u64, er: i128) -> usize {
    let f1 = w + 24;
    if f1.div_ceil(64) as usize <= short::MAX_LIMBS {
        return f1.div_ceil(64) as usize;
    }
    let k = sincos::halvings(f1.div_ceil(64) * 64);
    let f = if er <= -k { f1 } else { f1 + 2 * k as u64 + 8 };
    f.div_ceil(64) as usize
}

/// `|x| = M·2^lsb` reduced, and `sin r`, `cos r` as `want(q)` asks for them, each with the
/// error of the reduction added, to about `w` bits relatively.
pub(super) fn eval(m: &[u64], lsb: i128, w: u64, want: impl Fn(u8) -> (bool, bool)) -> Eval {
    let red = reduce(m, lsb, w + 8);
    let er = i128::from(limbs::bit_len(&red.r)) - red.g;
    let fl = kernel_limbs(w, er);
    let (ws, wc) = want(red.q);
    let (mut s, mut c) = sincos::sin_cos(&red.r, red.g, fl, ws, wc);
    let dr = red.err.scale(-red.g);
    for f in [&mut s, &mut c].into_iter().flatten() {
        f.err = f.err.add(dr.scale(-f.lsb));
    }
    if let Some(s) = &mut s {
        s.negative = red.negative;
    }
    Eval {
        q: red.q,
        negative: red.negative,
        sin: s,
        cos: c,
    }
}

/// The [`Approx`] of an [`Fx`] of the sign `negative`.
///
/// A bound that is not a finite number (which no step should give), or that is not below the
/// value, decides nothing.
pub(super) fn approx(negative: bool, f: &Fx) -> Approx {
    let bound = f.abs_err();
    if bound.0.is_nan() || bound.0 == f64::INFINITY || bound.0 >= f.exponent() as f64 {
        return undecided();
    }
    let err = bound.ceil().unwrap_or(f.lsb - 2).max(f.lsb - 2);
    Approx {
        negative,
        m: f.m.clone(),
        lsb: f.lsb,
        err,
    }
}

/// An approximation that decides nothing, for a step whose bound is too wide.
fn undecided() -> Approx {
    Approx {
        negative: false,
        m: Vec::new(),
        lsb: 0,
        err: 0,
    }
}

/// The exact `1`, as an [`Fx`].
fn one() -> Fx {
    Fx::exact(vec![1], 0)
}

/// The circular functions, the one each method computes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum F {
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,
}

impl F {
    /// Whether the function is odd, `f(-x) = -f(x)` (otherwise even).
    fn odd(self) -> bool {
        matches!(self, F::Sin | F::Tan | F::Csc | F::Cot)
    }

    /// Of `|x| = kπ/2 + r`, `k mod 4 = q`: which of `sin r` and `cos r` the function needs.
    fn want(self, q: u8) -> (bool, bool) {
        match self {
            F::Tan | F::Cot => (true, true),
            F::Sin | F::Csc => (q & 1 == 0, q & 1 == 1),
            F::Cos | F::Sec => (q & 1 == 1, q & 1 == 0),
        }
    }

    /// The approximation of the function of `|x|` from `sin r`, `cos r`, to about `w` bits.
    fn of(self, e: &Eval, w: u64) -> Approx {
        let (q, rneg) = (e.q, e.negative);
        let s = e.sin.as_ref();
        let c = e.cos.as_ref();
        // `sin(|x|)` is `sin r, cos r, -sin r, -cos r` for `q = 0..3`; `cos(|x|)` is `cos r, -sin
        // r, -cos r, sin r`; `tan(|x|)` is `tan r` for even `q` and `-cot r` for odd.
        let quot = |a: &Fx, b: &Fx, negative: bool| match sincos::quotient(a, b, w + 16) {
            Some(f) => approx(negative, &f),
            None => undecided(),
        };
        match self {
            F::Sin => {
                if q & 1 == 0 {
                    approx((q >= 2) != rneg, s.expect("sine"))
                } else {
                    approx(q >= 2, c.expect("cosine"))
                }
            }
            F::Cos => {
                if q & 1 == 0 {
                    approx(q >= 2, c.expect("cosine"))
                } else {
                    approx((q == 1) != rneg, s.expect("sine"))
                }
            }
            F::Csc => {
                if q & 1 == 0 {
                    quot(&one(), s.expect("sine"), (q >= 2) != rneg)
                } else {
                    quot(&one(), c.expect("cosine"), q >= 2)
                }
            }
            F::Sec => {
                if q & 1 == 0 {
                    quot(&one(), c.expect("cosine"), q >= 2)
                } else {
                    quot(&one(), s.expect("sine"), (q == 1) != rneg)
                }
            }
            F::Tan => {
                let (s, c) = (s.expect("sine"), c.expect("cosine"));
                if q & 1 == 0 {
                    quot(s, c, rneg)
                } else {
                    quot(c, s, !rneg)
                }
            }
            F::Cot => {
                let (s, c) = (s.expect("sine"), c.expect("cosine"));
                if q & 1 == 0 {
                    quot(c, s, rneg)
                } else {
                    quot(s, c, !rneg)
                }
            }
        }
    }

    /// The value at `±0`.
    fn at_zero(self, negative: bool, prec: u64) -> BigFloat {
        match self {
            F::Sin | F::Tan => BigFloat::special(prec, Kind::Zero { negative }),
            F::Cos | F::Sec => BigFloat::from(1).prec_round(prec, Round::Nearest).0,
            F::Csc | F::Cot => BigFloat::special(prec, Kind::Inf { negative }),
        }
    }

    /// The direct rounding of a small argument `x = ±M·2^lsb`, `|x| < 2^e`, when the function lies
    /// within less than half an ulp of the argument, of its reciprocal or of 1:
    ///
    /// - `0 < x - sin x < x³/6 < 2^(3e - 2)`, and `0 < tan x - x < 0.36x³ < 2^(3e - 1)` for `|x| <
    ///   1/2`;
    /// - `0 < 1 - cos x < x²/2 < 2^(2e - 1)`, and `0 < sec x - 1 < 0.52x² < 2^(2e)` for `|x| < 1/2`;
    /// - for `x = ±2^(e - 1)`, whose reciprocal `±2^(1 - e)` is exact: `0 < csc x - 1/x < 0.17x <
    ///   2^(e - 2)` and `0 < 1/x - cot x < 0.34x < 2^(e - 1)` for `x > 0`.
    fn small(self, x: Mag<'_>, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
        let bits = limbs::bit_len(x.m);
        let e = x.lsb + i128::from(bits);
        let p = i128::from(prec);
        // A cheap test first: each case needs at least `2e <= 1 - p` or so.
        if e > -(p / 2) + 2 || e > -1 {
            return None;
        }
        let neg = x.negative;
        let power = limbs::trailing_zeros(x.m) == bits - 1;
        match self {
            F::Sin => near(neg, x.m, x.lsb, false, 3 * e - 2, prec, round),
            F::Tan => near(neg, x.m, x.lsb, true, 3 * e - 1, prec, round),
            F::Cos => near(false, &[1], 0, false, 2 * e - 1, prec, round),
            F::Sec => near(false, &[1], 0, true, 2 * e, prec, round),
            F::Csc if power => near(neg, &[1], 1 - e, true, e - 2, prec, round),
            F::Cot if power => near(neg, &[1], 1 - e, false, e - 1, prec, round),
            _ => None,
        }
    }

    /// The function rounded.
    #[track_caller]
    fn round(self, x: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &x.kind {
            Kind::Nan | Kind::Inf { .. } => exact(BigFloat::nan(prec)),
            Kind::Zero { negative } => exact(self.at_zero(*negative, prec)),
            Kind::Finite { .. } => {
                let mag = x.mag().expect("finite");
                // `f(x) = ±f(|x|)`: the mode for `f(|x|)` swapped when it is turned over.
                let flip = self.odd() && mag.negative;
                let r = super::arith::round_for(round, flip);
                let pos = Mag {
                    negative: false,
                    ..mag
                };
                let (mut y, o) = match self.small(pos, prec, r) {
                    Some(v) => v,
                    None => ziv_fast(prec, r, |w| {
                        let e = eval(pos.m, pos.lsb, w, |q| self.want(q));
                        self.of(&e, w)
                    }),
                };
                if flip {
                    let neg = y.is_sign_negative();
                    y.set_sign(!neg);
                    (y, o.reverse())
                } else {
                    (y, o)
                }
            }
        }
    }
}

macro_rules! plain {
    ($plain:ident, $round:ident, $name:literal) => {
        #[doc = concat!("The ", $name, ", to nearest at the precision of `self`.")]
        pub fn $plain(&self) -> BigFloat {
            self.$round(self.prec, Round::Nearest).0
        }
    };
}

impl BigFloat {
    /// The sine rounded to `prec` bits in the mode `round`, and the direction of the rounding:
    /// NaN for NaN and the infinities, `±0` for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (s, o) = BigFloat::from(1).sin_round(53, Round::Nearest);
    /// assert_eq!(s, BigFloat::from(1f64.sin()));
    /// assert_eq!(o, Ordering::Less);
    /// assert!(BigFloat::infinity(53).sin().is_nan());
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sin_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        F::Sin.round(self, prec, round)
    }

    /// The cosine rounded to `prec` bits in the mode `round`, and the direction of the rounding:
    /// NaN for NaN and the infinities, 1 for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use core::cmp::Ordering;
    ///
    /// // cos(2^100), with π to more than 100 bits for the reduction.
    /// let x = BigFloat::from_parts(&Number::from(1), 100, 1, Round::Nearest).0;
    /// let (c, o) = x.cos_round(24, Round::Up);
    /// assert_eq!(o, Ordering::Greater);
    /// assert_eq!(c, x.cos_round(200, Round::Up).0.prec_round(24, Round::Up).0);
    /// assert_eq!(BigFloat::zero(8).cos(), BigFloat::from(1));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn cos_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        F::Cos.round(self, prec, round)
    }

    /// The tangent rounded to `prec` bits in the mode `round`, and the direction of the rounding:
    /// NaN for NaN and the infinities, `±0` for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// // Near π/2 the tangent is large: π/2 to 53 bits is below π/2 by about 2^-54.
    /// let half_pi = BigFloat::from(core::f64::consts::FRAC_PI_2);
    /// let t = half_pi.tan_round(30, Round::Nearest).0;
    /// assert!(t > BigFloat::from(1e16) && t < BigFloat::from(2e16));
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn tan_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        F::Tan.round(self, prec, round)
    }

    /// The secant `1/cos x` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding: NaN for NaN and the infinities, 1 for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (s, o) = BigFloat::from(1).sec_round(10, Round::Down);
    /// assert_eq!((s, o), (BigFloat::from(1.849609375), Ordering::Less)); // 1/cos 1 = 1.8508…
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sec_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        F::Sec.round(self, prec, round)
    }

    /// The cosecant `1/sin x` rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: NaN for NaN and the infinities, `±∞` for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// assert!(BigFloat::neg_zero(53).csc().is_infinite());
    /// let (c, o) = BigFloat::from(1).csc_round(10, Round::Up);
    /// assert_eq!((c, o), (BigFloat::from(1.189453125), Ordering::Greater)); // 1.1883…
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn csc_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        F::Csc.round(self, prec, round)
    }

    /// The cotangent `1/tan x` rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: NaN for NaN and the infinities, `±∞` for `±0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (c, o) = BigFloat::from(1).cot_round(10, Round::TowardZero);
    /// assert_eq!((c, o), (BigFloat::from(0.6416015625), Ordering::Less)); // 0.6420…
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn cot_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        F::Cot.round(self, prec, round)
    }

    plain!(sin, sin_round, "sine");
    plain!(cos, cos_round, "cosine");
    plain!(tan, tan_round, "tangent");
    plain!(sec, sec_round, "secant");
    plain!(csc, csc_round, "cosecant");
    plain!(cot, cot_round, "cotangent");

    /// The sine and the cosine, both rounded to `prec` bits in the mode `round`, each with the
    /// direction of its rounding, from one reduction of the argument: as [`sin_round`] and
    /// [`cos_round`] give them.
    ///
    /// [`sin_round`]: BigFloat::sin_round
    /// [`cos_round`]: BigFloat::cos_round
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// let x = BigFloat::from(3);
    /// let ((s, _), (c, _)) = x.sin_cos_round(100, Round::Down);
    /// assert_eq!(s, x.sin_round(100, Round::Down).0);
    /// assert_eq!(c, x.cos_round(100, Round::Down).0);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sin_cos_round(
        &self,
        prec: u64,
        round: Round,
    ) -> ((BigFloat, Ordering), (BigFloat, Ordering)) {
        let prec = BigFloat::check_prec(prec);
        let Some(mag) = self.mag() else {
            return (
                F::Sin.round(self, prec, round),
                F::Cos.round(self, prec, round),
            );
        };
        let flip = mag.negative;
        let rs = super::arith::round_for(round, flip);
        let pos = Mag {
            negative: false,
            ..mag
        };
        let mut sin = F::Sin.small(pos, prec, rs);
        let mut cos = F::Cos.small(pos, prec, round);
        let mut w = prec + guard(prec);
        while sin.is_none() || cos.is_none() {
            let e = eval(pos.m, pos.lsb, w, |_| (true, true));
            if sin.is_none() {
                let a = F::Sin.of(&e, w);
                sin = round_cheap(&a, prec, rs).or_else(|| round_approx(&a, prec, rs));
            }
            if cos.is_none() {
                let a = F::Cos.of(&e, w);
                cos = round_cheap(&a, prec, round).or_else(|| round_approx(&a, prec, round));
            }
            w += (w / 2).max(64);
        }
        let (mut s, os) = sin.expect("rounded");
        let os = if flip {
            let neg = s.is_sign_negative();
            s.set_sign(!neg);
            os.reverse()
        } else {
            os
        };
        ((s, os), cos.expect("rounded"))
    }

    /// The sine and the cosine, to nearest at the precision of `self`, from one reduction.
    pub fn sin_cos(&self) -> (BigFloat, BigFloat) {
        let ((s, _), (c, _)) = self.sin_cos_round(self.prec, Round::Nearest);
        (s, c)
    }
}
