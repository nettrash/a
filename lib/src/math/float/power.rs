//! Roots and powers, each correctly rounded in every mode with its ternary: the reciprocal square
//! root, roots of any degree, the cube root, integer powers (of a machine integer or a
//! [`Number`]) and rational powers, with MPFR's special values (`mpfr_rec_sqrt`,
//! `mpfr_rootn_ui`, `mpfr_cbrt`, `mpfr_pow_si`, `mpfr_pow_z`).
//!
//! A result that is exact, or halfway between two numbers of the precision, is found first and
//! computed exactly: a root from the integer root of the mantissa (exact or not, its floor and
//! remainder decide the rounding), a power of a short odd part as the exact power. Otherwise the
//! value is irrational or needs more bits than the precision and one: Ziv's strategy then ends,
//! on powers computed with every product cut to a working length, whose error is bounded below.

use super::general::{Op, decided};
use super::limbs::{self, Work};
use super::power_fixed;
use super::rec_sqrt;
use super::root_small;
use super::round::{overflow, round_into, round_mag, seal, underflow};
use super::small;
use super::ziv::{Approx, guard, round_approx};
use super::{BigFloat, Kind, Round};
use crate::math::numbers::{ByteOrder, Number, nat};
use crate::math::rational::Rational;
use core::cmp::Ordering;

/// A positive integer magnitude and the exponent of its lowest bit: `m·2^lsb`, `m` aligned at its
/// top bit, its lowest limb not zero.
struct Int {
    m: Vec<u64>,
    lsb: i128,
}

impl Int {
    /// The magnitude of a finite number that is not zero.
    fn of(x: Op<'_>) -> Int {
        let x = x.strip();
        Int {
            m: x.m.to_vec(),
            lsb: x.exp - 64 * x.m.len() as i128,
        }
    }

    /// The odd part and the exponent: `m'·2^k` with `m'` odd.
    fn odd(&self) -> (Vec<u64>, i128) {
        let z = limbs::trailing_zeros(&self.m);
        (limbs::shr(&self.m, z), self.lsb + i128::from(z))
    }
}

/// Whether the magnitude is 1.
#[inline]
fn is_one(m: &[u64]) -> bool {
    m == [1]
}

/// `m^n` exactly, `n ≥ 1`, by squarings from the top bit of `n`.
fn pow_exact(m: &[u64], n: u64) -> Vec<u64> {
    let mut acc = m.to_vec();
    let mut t = Vec::new();
    let mut s = Vec::new();
    for i in (0..63 - n.leading_zeros()).rev() {
        nat::sqr_into(&mut t, &acc, &mut s);
        limbs::trim(&mut t);
        core::mem::swap(&mut acc, &mut t);
        if n >> i & 1 == 1 {
            nat::mul_into(&mut t, &acc, m, &mut s);
            limbs::trim(&mut t);
            core::mem::swap(&mut acc, &mut t);
        }
    }
    acc
}

/// `⌈log2 v⌉` for `v ≥ 1`.
#[inline]
fn ceil_log2(v: u64) -> u64 {
    u64::from(64 - (v - 1).leading_zeros())
}

// Powers cut to a working length

/// A positive value `v` known from below: `c ≤ v < c·(1 + ε)^k` for the cut value `c = m·2^lsb`
/// (`m` of at most `wl` limbs, its top limb not zero) and `ε = 2^(64 - 64wl)`, the relative error
/// of a cut: a product of more limbs keeps its top `wl`, and drops less than `B^(len - wl)` units
/// of a value of at least `B^(len - 1)`.
struct Low {
    m: Vec<u64>,
    lsb: i128,
}

/// `m` cut to its top `wl` limbs, `lsb` moved by the limbs dropped.
#[inline]
fn cut(m: &mut Vec<u64>, lsb: &mut i128, wl: usize) {
    limbs::trim(m);
    if m.len() > wl {
        let d = m.len() - wl;
        m.drain(..d);
        *lsb += 64 * d as i128;
    }
}

/// `b^n` cut to `wl` limbs, for the base `b` cut too: from the top bit of `n` (limbs, the least
/// significant first, not zero), a square and, for each set bit, a product by `b`.
///
/// With `k` the exponent of the error of the base (`b ≤ v < b(1 + ε)^kb`), the error of the power
/// is at most `(kb + 2)·n`: a square takes `k` to `2k + 1`, a product by the base to `k + kb + 1`,
/// so that from `k ≤ (kb + 2)·n' - 1` for the bits `n'` of `n` read so far, the next bit gives
/// `2k + 1 + kb + 1 ≤ (kb + 2)(2n' + 1) - 1`.
fn pow_low(b: &Low, n: &[u64], wl: usize) -> Low {
    let bits = limbs::bit_len(n);
    let mut acc = b.m.clone();
    let mut lsb = b.lsb;
    let mut t = Vec::new();
    let mut s = Vec::new();
    for i in (0..bits - 1).rev() {
        nat::sqr_into(&mut t, &acc, &mut s);
        lsb *= 2;
        cut(&mut t, &mut lsb, wl);
        core::mem::swap(&mut acc, &mut t);
        if limbs::bit(n, i) {
            nat::mul_into(&mut t, &acc, &b.m, &mut s);
            lsb += b.lsb;
            cut(&mut t, &mut lsb, wl);
            core::mem::swap(&mut acc, &mut t);
        }
    }
    Low { m: acc, lsb }
}

/// The approximation of `v` that `l` gives, of the sign `negative`, for a power by `n` of `bits`
/// bits of a base whose error exponent is `kb`: `v - c < c·((1 + ε)^k - 1) ≤ 2kε·c` for `kε ≤ 1`,
/// `k ≤ (kb + 2)·n < (kb + 2)·2^bits`, and `c < 2^(bit_len(m) + lsb)`.
fn approx_of(l: Low, negative: bool, bits: u64, kb: u64, wl: usize) -> Approx {
    let top = l.lsb + i128::from(limbs::bit_len(&l.m));
    let err = top + 64 - 64 * wl as i128 + i128::from(bits) + 1 + i128::from(ceil_log2(kb + 2));
    Approx {
        negative,
        m: l.m,
        lsb: l.lsb,
        err,
    }
}

/// `1/x` cut to `wl` limbs, from below, for the magnitude `x`: `floor(2^K / m)`, `K = 64(wl +
/// len) - 1`, of `wl` limbs, below `1/x` by less than one unit of its lowest bit, one `ε`.
fn reciprocal_low(x: &Int, wl: usize) -> Low {
    let dn = x.m.len();
    // A normalized divisor: the mantissa shifted to its top bit.
    let lz = u64::from(x.m[dn - 1].leading_zeros());
    let mut d = limbs::shl(&x.m, lz);
    d.resize(dn, 0);
    let mut num = vec![0u64; wl + dn];
    num[wl + dn - 1] = 1 << 63;
    let mut q = vec![0u64; wl];
    let mut s = Vec::new();
    if nat::div_qr(&mut q, &mut num, &d, &mut s) {
        // A divisor that is a power of two: the quotient `2^(64wl)`.
        q.push(1);
    }
    // `2^K / (m·2^lz)`, so `1/x = q·2^-(K - lz + lsb)` from below.
    let k = 64 * (wl + dn) as i128 - 1;
    let mut lsb = -(k - i128::from(lz) + x.lsb);
    cut(&mut q, &mut lsb, wl);
    Low { m: q, lsb }
}

/// Where the power `|x|^y` lies for `log2 |y| ≈ ly` and the sign `ys` of `y`: `Some` overflow (true)
/// or underflow (false) when it is certainly beyond the exponent range, `None` when its exponent
/// is within about `2^62` of zero, so that every exponent of the computation fits.
fn beyond(x: Op<'_>, ly: f64, y_negative: bool) -> Option<bool> {
    // `|log2 |x|| ≤ |e| + 1` for `x = 0.M·2^e`: nothing to compute when that keeps it in reach.
    if ly + ((x.exp.unsigned_abs() + 1) as f64).log2() <= 62.0 {
        return None;
    }
    let (l2, lneg) = log2_abs(x);
    let l2 = l2?;
    // `log2 |y log2 x|`, its sign that of the result's exponent.
    let t = ly + l2;
    let up = lneg == y_negative;
    if t > 62.01 { Some(up) } else { None }
}

/// `log2 |log2 |x||` and whether `log2 |x|` is negative, for a finite `x` that is not zero; `None`
/// for `|x| = 1`. Good to a few units of `2^-40` relative, which [`beyond`]'s margin absorbs.
fn log2_abs(x: Op<'_>) -> (Option<f64>, bool) {
    let m = x.m;
    let top = |v: &[u64]| -> Option<(f64, i128)> {
        // The value of the limbs `v` as `f·2^e` with `f` in `[1/2, 1)`.
        let mut v = v.to_vec();
        limbs::trim(&mut v);
        if v.is_empty() {
            return None;
        }
        let bits = limbs::bit_len(&v);
        let hi = limbs::extract(&v, i128::from(bits) - 64);
        Some((hi as f64 / 18_446_744_073_709_551_616.0, i128::from(bits)))
    };
    let len = 64 * m.len() as i128;
    if x.exp == 1 || x.exp == 0 {
        // Near 1: `|x| = 1 + t` or `1 - t`, `t` from the mantissa less `2^(len - 1)` (or `2^len`
        // less the mantissa), exactly, as a float with an exponent of its own.
        let (t, below_one) = if x.exp == 1 {
            let mut v = m.to_vec();
            let k = v.len() - 1;
            v[k] &= !(1 << 63);
            (top(&v), false)
        } else {
            // `2^len - m`: the two's complement.
            let mut v: Vec<u64> = m.iter().map(|&l| !l).collect();
            let carried = nat::add_assign(&mut v, &[1]);
            if carried {
                return (None, false);
            }
            (top(&v), true)
        };
        let Some((f, e)) = t else {
            return (None, false);
        };
        // `t = f·2^(e - len + exp)`, then `log2 |log2(1 ± t)|` from `ln_1p`, or from `t/ln 2` when
        // `t` is below the range of `f64`.
        let et = e - len + x.exp;
        let l = if et > -900 {
            let tv = f * 2f64.powi(et as i32);
            let v = if below_one { (-tv).ln_1p() } else { tv.ln_1p() };
            (v.abs() / core::f64::consts::LN_2).log2()
        } else {
            f.log2() + et as f64 - core::f64::consts::LN_2.log2()
        };
        return (Some(l), below_one);
    }
    let (f, _) = top(m).expect("not zero");
    let l = (x.exp - 1) as f64 + (2.0 * f).log2();
    (Some(l.abs().log2()), l < 0.0)
}

impl BigFloat {
    /// `self^n` rounded to `prec` bits in the mode `round`, and the direction of the rounding, as
    /// MPFR's `mpfr_pow_si`: `x^0 = 1` for every `x`, NaN included; `(±0)^n` is `±0` for an odd
    /// `n > 0`, `+0` for an even one, `±∞` for an odd `n < 0` and `+∞` for an even one; `(±∞)^n`
    /// likewise turned over. A result beyond the exponent range overflows or underflows by the
    /// mode, found from the exponents without computing the power.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(3).powi_round(5, 53, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(243), Ordering::Equal));
    /// let (x, o) = BigFloat::from(3).powi_round(-1, 53, Round::Nearest);
    /// assert_eq!(x, BigFloat::from(1.0 / 3.0));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn powi_round(&self, n: i64, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let mag = [n.unsigned_abs()];
        let mag: &[u64] = if n == 0 { &[] } else { &mag };
        pow_int(self, n < 0, mag, prec, round)
    }

    /// `self^n` to nearest at the precision of `self`.
    pub fn powi(&self, n: i64) -> BigFloat {
        self.powi_round(n, self.prec, Round::Nearest).0
    }

    /// `self^n` for an integer `n` of any size, rounded to `prec` bits in the mode `round`, and the
    /// direction of the rounding, as MPFR's `mpfr_pow_z`, with the special values of
    /// [`powi_round`](BigFloat::powi_round).
    #[track_caller]
    pub fn pow_number_round(&self, n: &Number, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let mag = limbs::of_number(n);
        pow_int(self, n.is_negative(), &mag, prec, round)
    }

    /// `self^n` to nearest at the precision of `self`.
    pub fn pow_number(&self, n: &Number) -> BigFloat {
        self.pow_number_round(n, self.prec, Round::Nearest).0
    }

    /// `1/sqrt(self)` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as MPFR's `mpfr_rec_sqrt`: `+∞` for `±0`, `+0` for `+∞`, NaN below zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    ///
    /// assert_eq!(BigFloat::from(0.25).rec_sqrt_round(10, Round::Nearest).0, BigFloat::from(2));
    /// assert!(BigFloat::neg_zero(10).rec_sqrt().is_infinite());
    /// ```
    #[track_caller]
    pub fn rec_sqrt_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let kind = match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } | Kind::Finite { negative: true, .. } => {
                Kind::Nan
            }
            Kind::Zero { .. } => Kind::Inf { negative: false },
            Kind::Inf { negative: false } => Kind::Zero { negative: false },
            Kind::Finite { .. } => return rec_sqrt(self.op().expect("finite"), prec, round),
        };
        (BigFloat { prec, kind }, Ordering::Equal)
    }

    /// `1/sqrt(self)` to nearest at the precision of `self`.
    pub fn rec_sqrt(&self) -> BigFloat {
        self.rec_sqrt_round(self.prec, Round::Nearest).0
    }

    /// The `n`-th root rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as MPFR's `mpfr_rootn_ui` and IEEE 754's `rootn`: NaN for `n = 0`; for an odd `n`
    /// the root of a negative number is negative (`-∞` for `-∞`), for an even one NaN; the root of
    /// a zero is the zero for an odd `n` and `+0` for an even one.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(-27).nth_root_round(3, 20, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(-3), Ordering::Equal));
    /// assert!(BigFloat::from(-2).nth_root_round(4, 20, Round::Nearest).0.is_nan());
    /// ```
    #[track_caller]
    pub fn nth_root_round(&self, n: u32, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let odd = n % 2 == 1;
        let kind = match &self.kind {
            _ if n == 0 => Kind::Nan,
            Kind::Nan => Kind::Nan,
            Kind::Zero { negative } => Kind::Zero {
                negative: *negative && odd,
            },
            Kind::Inf { negative: false } => Kind::Inf { negative: false },
            Kind::Inf { negative: true } => {
                if odd {
                    Kind::Inf { negative: true }
                } else {
                    Kind::Nan
                }
            }
            Kind::Finite { negative: true, .. } if !odd => Kind::Nan,
            Kind::Finite { .. } => {
                let x = self.op().expect("finite");
                return match n {
                    1 => self.prec_round(prec, round),
                    2 => self.sqrt_round(prec, round),
                    _ => root(x, u64::from(n), prec, round),
                };
            }
        };
        (BigFloat { prec, kind }, Ordering::Equal)
    }

    /// The `n`-th root to nearest at the precision of `self`.
    pub fn nth_root(&self, n: u32) -> BigFloat {
        self.nth_root_round(n, self.prec, Round::Nearest).0
    }

    /// The cube root rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as MPFR's `mpfr_cbrt`: of the sign of `self`, `-0` for `-0`.
    #[track_caller]
    pub fn cbrt_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        self.nth_root_round(3, prec, round)
    }

    /// The cube root to nearest at the precision of `self`.
    pub fn cbrt(&self) -> BigFloat {
        self.cbrt_round(self.prec, Round::Nearest).0
    }

    /// `self^(p/q)` for the rational `r = p/q` in lowest terms, rounded to `prec` bits in the mode
    /// `round`, and the direction of the rounding: the `q`-th root of `self`, as
    /// [`nth_root_round`](BigFloat::nth_root_round) takes it, to the power `p`, as
    /// [`pow_number_round`](BigFloat::pow_number_round) takes it, rounded once. A negative number
    /// has a power for an odd `q` (negative for an odd `p`) and NaN for an even one; an integer
    /// `r` is [`pow_number_round`](BigFloat::pow_number_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::rational::Rational;
    /// use core::cmp::Ordering;
    ///
    /// let two_thirds = Rational::from(2) / 3u32;
    /// let (x, o) = BigFloat::from(-8).pow_rational_round(&two_thirds, 20, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(4), Ordering::Equal));
    /// ```
    ///
    /// # Panics
    ///
    /// When `q` is `2^64` or more.
    #[track_caller]
    pub fn pow_rational_round(
        &self,
        r: &Rational,
        prec: u64,
        round: Round,
    ) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let p = r.numer();
        let qm = limbs::of_number(r.denom());
        if qm == [1] {
            return self.pow_number_round(p, prec, round);
        }
        assert!(qm.len() == 1, "a denominator of 2^64 or more");
        let q = qm[0];
        let pm = limbs::of_number(p);
        let p_odd = pm[0] & 1 == 1;
        let q_odd = q & 1 == 1;
        let p_negative = p.is_negative();
        // The root as `nth_root_round` takes it, then the power by `p` as `pow_number_round`.
        let rooted = |negative: bool| -> Kind {
            // The root's sign: negative for an odd `q`, and the power's for an odd `p`.
            let negative = negative && q_odd;
            let negative = negative && p_odd;
            if p_negative {
                Kind::Inf { negative }
            } else {
                Kind::Zero { negative }
            }
        };
        let kind = match &self.kind {
            Kind::Nan => Kind::Nan,
            Kind::Zero { negative } => rooted(*negative),
            Kind::Inf { negative } => {
                if *negative && !q_odd {
                    Kind::Nan
                } else {
                    let negative = *negative && p_odd;
                    if p_negative {
                        Kind::Zero { negative }
                    } else {
                        Kind::Inf { negative }
                    }
                }
            }
            Kind::Finite { negative: true, .. } if !q_odd => Kind::Nan,
            Kind::Finite { .. } => {
                return pow_rational(self.op().expect("finite"), p, &pm, q, prec, round);
            }
        };
        (BigFloat { prec, kind }, Ordering::Equal)
    }

    /// `self^r` to nearest at the precision of `self`.
    pub fn pow_rational(&self, r: &Rational) -> BigFloat {
        self.pow_rational_round(r, self.prec, Round::Nearest).0
    }
}

/// The result of a kind and an ordering at the precision `prec`.
#[inline]
fn done(kind: Kind, o: Ordering, prec: u64) -> (BigFloat, Ordering) {
    (BigFloat { prec, kind }, o)
}

/// `x^n` for `n = ±mag` (`mag` empty for zero), with the special values of `mpfr_pow_z`.
fn pow_int(
    x: &BigFloat,
    n_negative: bool,
    mag: &[u64],
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    if mag.is_empty() {
        return round_mag(false, &[1], 0, false, prec, round);
    }
    let odd = mag[0] & 1 == 1;
    let kind = match &x.kind {
        Kind::Nan => Kind::Nan,
        Kind::Zero { negative } | Kind::Inf { negative } => {
            let negative = *negative && odd;
            if x.is_zero() != n_negative {
                Kind::Zero { negative }
            } else {
                Kind::Inf { negative }
            }
        }
        Kind::Finite { negative, .. } => {
            let negative = *negative && odd;
            return pow_finite(
                x.op().expect("finite"),
                negative,
                n_negative,
                mag,
                prec,
                round,
            );
        }
    };
    done(kind, Ordering::Equal, prec)
}

/// `|x|^(±n)` of the sign `negative`, for a finite `x` that is not zero and `n ≥ 1` in limbs.
fn pow_finite(
    x: Op<'_>,
    negative: bool,
    n_negative: bool,
    n: &[u64],
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    if n == [1] || (!n_negative && n == [2]) {
        // `x`, `1/x` or `x^2`, rounded once.
        let ax = BigFloat {
            prec: 64 * x.m.len() as u64,
            kind: Kind::Finite {
                negative,
                exp: x.exp as i64,
                mant: x.m.to_vec(),
            },
        };
        return if n_negative {
            BigFloat::from(1).div_round(&ax, prec, round)
        } else if n == [2] {
            ax.mul_round(&ax, prec, round)
        } else {
            ax.prec_round(prec, round)
        };
    }
    if prec <= 64 && n.len() == 1 && x.m.len() <= 2 {
        // In registers first, when the power is neither a power of two nor short enough to be
        // exact or halfway: the odd part of `b` bits to the power `n` has `n(b - 1) + 1` or more.
        let w = small::wide(x.m);
        let b = 128 - u64::from(w.trailing_zeros());
        let exactly =
            b == 1 || (!n_negative && n[0].checked_mul(b - 1).is_none_or(|lo| lo < prec + 1));
        let ly = (n[0] as f64).log2();
        if !exactly
            && beyond(x, ly, n_negative).is_none()
            && let Some(r) = pow_small(x, negative, n_negative, n[0], prec, round)
        {
            let (kind, o) = seal(r, prec, round, Vec::new());
            return done(kind, o, prec);
        }
    }
    let bits = limbs::bit_len(n);
    // The bits of the odd part of the mantissa, which is aligned at its top bit.
    let b = 64 * x.m.len() as u64 - limbs::trailing_zeros(x.m);
    if n.len() == 1 && b >= 2 && (n_negative || n[0].checked_mul(b - 1).is_none_or(|lo| lo > prec))
    {
        // Neither a number of the precision nor halfway: on limbs of a fixed width first.
        let ly = (n[0] as f64).log2();
        if beyond(x, ly, n_negative).is_none()
            && let Some((kind, o)) = power_fixed::pow(x, negative, n_negative, n[0], prec, round)
        {
            return done(kind, o, prec);
        }
    }
    let xi = Int::of(x);
    let (odd, k) = xi.odd();
    if is_one(&odd) {
        // `±2^(kn)`, exactly, or beyond the range: `kn` saturates far past it.
        let kn = if bits > 100 {
            if k == 0 {
                0
            } else {
                i128::MAX / 2 * k.signum()
            }
        } else {
            let nv = u128::from(n[0]) | n.get(1).map_or(0, |&h| u128::from(h) << 64);
            k.checked_mul(nv as i128)
                .unwrap_or(i128::MAX / 2 * k.signum())
        };
        let kn = if n_negative { -kn } else { kn };
        let kn = kn.clamp(-(1 << 100), 1 << 100);
        return round_mag(negative, &[1], kn, false, prec, round);
    }
    if !n_negative && n.len() == 1 {
        // An odd part of `b ≥ 2` bits to the power `n` has at least `n(b - 1) + 1` bits: up to
        // one more than the precision it may be exact or halfway, and is computed exactly.
        let b = limbs::bit_len(&odd);
        let nv = n[0];
        // Also exactly when the exact power is short: a few products of a few limbs.
        let cheap = prec > 64 && nv.checked_mul(b).is_some_and(|bits| bits <= 4 * prec + 256);
        if let Some(lo) = nv.checked_mul(b - 1)
            && (lo < prec + 1 || cheap)
        {
            let p = pow_exact(&odd, nv);
            let lsb = k * i128::from(nv);
            let (kind, o) = round_into(negative, &p, lsb, false, prec, round, Vec::new());
            return done(kind, o, prec);
        }
    }
    // Inexact: neither a number of the precision nor halfway between two.
    let ly = (bits - 1) as f64 + (limbs::extract(n, i128::from(bits) - 53) as f64).log2() - 52.0;
    if let Some(up) = beyond(x, ly, n_negative) {
        return range(negative, up, prec, round);
    }
    if prec <= 64
        && n.len() == 1
        && x.m.len() > 2
        && let Some(r) = pow_small(x, negative, n_negative, n[0], prec, round)
    {
        let (kind, o) = seal(r, prec, round, Vec::new());
        return done(kind, o, prec);
    }
    // Ziv's strategy on products cut to `wl` limbs, which grow by half each time.
    let mut wl = (prec + 1 + bits + 4 + 12).div_ceil(64).max(2) as usize;
    loop {
        if let Some((kind, o)) = limbs::with_work(|w| {
            power_fixed::pow_vec(x, negative, n_negative, n, wl, prec, round, w)
        }) {
            return done(kind, o, prec);
        }
        if n_negative && wl > 4 * x.m.len() + 8 {
            // A divisor that the cut keeps a power of two: the general reciprocal.
            return pow_general(&xi, negative, n_negative, n, prec, round);
        }
        wl += wl / 2 + 1;
    }
}

/// `|x|^(±n)` by Ziv's strategy on powers cut to whole limbs from the general reciprocal, for the
/// divisors that [`power_fixed::pow_vec`] cannot take.
fn pow_general(
    xi: &Int,
    negative: bool,
    n_negative: bool,
    n: &[u64],
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let bits = limbs::bit_len(n);
    let kb: u64 = 1;
    let mut w = prec + guard(prec) + bits + 8;
    loop {
        let wl = (w / 64 + 2) as usize;
        let base = if n_negative {
            reciprocal_low(xi, wl)
        } else {
            let mut m = xi.m.clone();
            let mut lsb = xi.lsb;
            cut(&mut m, &mut lsb, wl);
            Low { m, lsb }
        };
        let l = pow_low(&base, n, wl);
        let a = approx_of(l, negative, bits, kb, wl);
        if let Some(r) = round_approx(&a, prec, round) {
            return r;
        }
        w += (w / 2).max(64);
    }
}

/// `|x|^(±n)` of the sign `negative` to `prec ≤ 64` bits in registers, when that decides it:
/// the power by squarings on 128 bits, every product cut to its top 128 bits from below, a relative
/// error of `ε = 2^-127` each, so that `c ≤ v < c(1 + ε)^k` with `k ≤ (kb + 2)·n` as for
/// [`pow_low`] (`kb = 1` for the base `x` cut to 128 bits or the reciprocal `floor(2^191 / m)` of a
/// mantissa of one limb, 0 for an exact base), and `c(1 + ε)^k < c + 4k` units of the lowest bit
/// of `c` for `kε ≤ 1/2`. The result when the rounding of `c` and of `c + 4k`, each with a sticky
/// bit, agree, as the rounding is monotone; `None` otherwise. `v` is neither a number of the
/// precision nor halfway, so no end is the value itself at a boundary.
fn pow_small(
    x: Op<'_>,
    negative: bool,
    n_negative: bool,
    n: u64,
    prec: u64,
    round: Round,
) -> Option<super::round::Rounded> {
    let (base, eb, kb) = if n_negative {
        let [m] = *x.m else {
            return None;
        };
        // `1/x = 2^191/m·2^(-127 - ex)` for `m > 2^63` (a power of two is exact), of 128 bits.
        let v = nat::invert_limb(m);
        let (q1, r1) = nat::div_2by1(1 << 63, 0, m, v);
        let (q0, _) = nat::div_2by1(r1, 0, m, v);
        (u128::from(q1) << 64 | u128::from(q0), 1 - x.exp, 1u64)
    } else {
        let cut = x.m.len() > 2;
        let top = limbs::extract(x.m, 64 * x.m.len() as i128 - 128);
        let hi = x.m[x.m.len() - 1];
        let b = if x.m.len() == 1 {
            u128::from(hi) << 64
        } else {
            u128::from(hi) << 64 | u128::from(top)
        };
        (b, x.exp, u64::from(cut))
    };
    debug_assert!(base >> 127 == 1);
    let mut acc = base;
    let mut e = eb;
    // The product cut to its top 128 bits, aligned at the top bit.
    let top = |a: u128, b: u128, e: i128| -> (u128, i128) {
        let (hi, lo) = small::wide_mul(a, b);
        if hi >> 127 == 1 {
            (hi, e)
        } else {
            (hi << 1 | lo >> 127, e - 1)
        }
    };
    for i in (0..63 - n.leading_zeros()).rev() {
        (acc, e) = top(acc, acc, 2 * e);
        if n >> i & 1 == 1 {
            (acc, e) = top(acc, base, e + eb);
        }
    }
    let k = u128::from(kb + 2) * u128::from(n);
    let (hi_end, over) = acc.overflowing_add(4 * k);
    if over || k >= 1 << 120 {
        return None;
    }
    let lo = small::round_top(negative, acc, e, true, prec, round);
    let hi = small::round_top(negative, hi_end, e, true, prec, round);
    (lo.m == hi.m && lo.exp == hi.exp && lo.magnitude == hi.magnitude).then_some(lo)
}

/// A result certainly beyond the exponent range: an overflow (`up`) or an underflow, of the sign
/// `negative`, by the mode.
fn range(negative: bool, up: bool, prec: u64, round: Round) -> (BigFloat, Ordering) {
    if up {
        overflow(negative, prec, round)
    } else {
        // Far below half the smallest number: zero, or the smallest one rounding away.
        underflow(
            negative,
            i128::from(BigFloat::EXP_MIN) - 4,
            false,
            prec,
            round,
        )
    }
}

/// `1/sqrt(x)` for a positive `x`: `2^(-j)` exactly for `x = 4^j`; otherwise irrational, the floor
/// of `sqrt(2^E / M)` for the mantissa `x = M·2^l` and an `E` of the parity of `l` that gives the
/// root `64(n + 1)` bits, as `floor(sqrt(floor(2^E / M))) = floor(sqrt(2^E / M))`; never exact,
/// so a sticky bit.
fn rec_sqrt(x: Op<'_>, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let x = x.strip();
    if x.m.len() == 1 && x.m[0].is_power_of_two() {
        // `2^k`, `k = exp - 1`: exact for an even `k`, otherwise the general path.
        if x.exp & 1 == 1 {
            return round_mag(false, &[1], -(x.exp - 1) / 2, false, prec, round);
        }
    } else if x.m.len() <= 2 && prec <= 127 {
        let r = rec_sqrt::small(x.m, x.exp, prec, round);
        let (kind, o) = seal(r, prec, round, Vec::new());
        return done(kind, o, prec);
    } else if let Some((kind, o)) = limbs::with_work(|w| {
        if prec < rec_sqrt::NEWTON {
            rec_sqrt::via_sqrt(x, prec, round, w)
        } else {
            rec_sqrt::newton(x, prec, round, w)
        }
    }) {
        return done(kind, o, prec);
    }
    let xi = Int::of(x);
    let qn = prec.div_ceil(64) + 1;
    let bm = limbs::bit_len(&xi.m);
    let base = 128 * qn + bm + 2;
    let e = base + ((i128::from(base) + xi.lsb) & 1) as u64;
    limbs::with_work(|w| {
        // The divisor normalized: `M·2^lz`, so the dividend is `2^(E + lz)`.
        let dn = xi.m.len();
        let lz = u64::from(xi.m[dn - 1].leading_zeros());
        let d = &mut w.c;
        d.clear();
        d.extend_from_slice(&limbs::shl(&xi.m, lz));
        d.resize(dn, 0);
        let ne = e + lz;
        let nl = (ne / 64 + 1) as usize;
        let num = &mut w.a;
        num.clear();
        num.resize(nl, 0);
        num[nl - 1] = 1 << (ne % 64);
        let q = &mut w.b;
        q.clear();
        q.resize(nl - dn, 0);
        let high = nat::div_qr(q, num, d, &mut w.scratch);
        if high {
            q.push(1);
        }
        limbs::trim(q);
        let mut s = Vec::new();
        nat::sqrt_rem(&mut s, None, q, &mut w.scratch);
        limbs::trim(&mut s);
        let lsb = -(i128::from(e) + xi.lsb) / 2;
        let (kind, o) = round_into(false, &s, lsb, true, prec, round, Vec::new());
        done(kind, o, prec)
    })
}

/// Up to this degree, or up to this many bits of radicand, `n·64(n_prec + 1)`, a root is
/// computed as the integer root; beyond, by Newton's method on floats. Measured on an Apple M4 Max
/// (loaded): the root of degree 100 of 53 bits took 4.0 µs as an integer root (12,800 bits) and
/// 1.5 µs on floats, the fifth root of 1000 bits 1.2 µs against 5.3, the cube root of 10,000 bits
/// 40 µs against 41.
const INTEGER_ROOT: u64 = 1 << 12;

/// The degree up to which a root is the integer root whatever its length.
const INTEGER_DEGREE: u64 = 16;

/// The `n`-th root, `n ≥ 3`, of the finite `x`, negative only for an odd `n`.
fn root(x: Op<'_>, n: u64, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let negative = x.negative;
    if n <= root_small::NMAX as u64
        && prec <= 127
        && let Some(r) = root_small::root(x, n, prec, round)
    {
        let (kind, o) = seal(r, prec, round, Vec::new());
        return done(kind, o, prec);
    }
    let qn = prec.div_ceil(64) + 1;
    if n <= INTEGER_DEGREE || n.saturating_mul(64 * qn) <= INTEGER_ROOT {
        return root_int(x, n, negative, prec, round);
    }
    root_float(x, n, negative, prec, round)
}

/// The `n`-th root of `|x|` of the sign `negative` as an integer root: `S = floor((M·2^t)^(1/n))`
/// for the mantissa `x = M·2^l` and `t ≡ l (mod n)` that gives `S` at least `prec + 16` bits
/// and 80 (a negative `t` drops bits of `M`, as `floor(floor(y)^(1/n)) = floor(y^(1/n))`), then
/// `S·2^((l - t)/n)` rounded, the remainder and the bits dropped for the sticky bit, computed only
/// when the bits of `S` below its round bit are all zero.
fn root_int(x: Op<'_>, n: u64, negative: bool, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let x = x.strip();
    let ni = i128::from(n);
    let bm = 64 * x.m.len() as i128;
    let xlsb = x.exp - bm;
    // Two limbs at least, which the test of the rounding reads.
    let t0 = ni * i128::from((prec + 16).max(80)) - bm;
    let t = t0 + (xlsb - t0).rem_euclid(ni);
    limbs::with_work(|w| {
        let Work {
            a: r,
            b: s,
            c: p,
            scratch,
        } = w;
        // `M·2^t`, or `floor(M·2^t)` and whether bits are dropped, into `r`.
        r.clear();
        let tail = if t >= 0 {
            let (k, b) = ((t / 64) as usize, (t % 64) as u32);
            r.resize(k, 0);
            if b == 0 {
                r.extend_from_slice(x.m);
            } else {
                let mut carry = 0u64;
                for &v in x.m {
                    r.push(v << b | carry);
                    carry = v >> (64 - b);
                }
                r.push(carry);
            }
            false
        } else {
            let sh = (-t) as u64;
            let (k, b) = ((sh / 64) as usize, (sh % 64) as u32);
            let m = &x.m[k.min(x.m.len())..];
            if b == 0 {
                r.extend_from_slice(m);
            } else {
                r.extend(
                    (0..m.len()).map(|i| m[i] >> b | m.get(i + 1).map_or(0, |&h| h << (64 - b))),
                );
            }
            !limbs::zero_below(x.m, sh.min(bm as u64))
        };
        limbs::trim(r);
        nat::root(s, r, n as u32, scratch);
        let lsb = (xlsb - t) / ni;
        let sticky = decided(s, 0, 0, prec) || tail || {
            // `S^n` against the radicand.
            pow_into(p, s, n, scratch);
            p != r
        };
        let (kind, o) = round_into(negative, s, lsb, sticky, prec, round, Vec::new());
        done(kind, o, prec)
    })
}

/// `m^n` exactly into `out`, without zero limbs at the top, `n ≥ 1`, by squarings from the top bit
/// of `n`; `scratch` is working memory.
fn pow_into(out: &mut Vec<u64>, m: &[u64], n: u64, scratch: &mut Vec<u64>) {
    let mut t = Vec::new();
    out.clear();
    out.extend_from_slice(m);
    for i in (0..63 - n.leading_zeros()).rev() {
        nat::sqr_into(&mut t, out, scratch);
        limbs::trim(&mut t);
        core::mem::swap(out, &mut t);
        if n >> i & 1 == 1 {
            nat::mul_into(&mut t, out, m, scratch);
            limbs::trim(&mut t);
            core::mem::swap(out, &mut t);
        }
    }
}

/// The `n`-th root of `|x|` of the sign `negative` by Newton's method on floats, for a large `n`:
/// the exact cases first (`x = m^n·2^(kn)` for an odd `m`, which for `m ≥ 3` needs at least
/// `n·log2 3` bits), then Ziv's strategy on a root `y` of `z = 0.M·2^r`, `r = e mod n`, certified
/// by powers rounded outward: when `y_lo^n ≤ z ≤ y_hi^n` for the powers of `y(1 ∓ 2^(4 - w))`
/// rounded up and down, the root lies between them.
fn root_float(x: Op<'_>, n: u64, negative: bool, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let xi = Int::of(x);
    let (odd, k) = xi.odd();
    let ni = i128::from(n);
    if is_one(&odd) {
        if k.rem_euclid(ni) == 0 {
            return round_mag(negative, &[1], k / ni, false, prec, round);
        }
    } else if limbs::bit_len(&odd) as f64 >= 1.58 * n as f64 {
        let r = int_root(&odd, n);
        if k.rem_euclid(ni) == 0 && pow_exact(&r, n) == odd {
            return round_mag(negative, &r, k / ni, false, prec, round);
        }
    }
    // `x = 0.M·2^e`, `e = n·qe + re`: the root is `2^qe` times that of `z = 0.M·2^re`, in [1/2, 2).
    let qe = x.exp.div_euclid(ni);
    let re = x.exp.rem_euclid(ni);
    let z = BigFloat {
        prec: 64 * x.m.len() as u64,
        kind: Kind::Finite {
            negative: false,
            exp: re as i64,
            mant: x.m.to_vec(),
        },
    };
    let top =
        limbs::extract(x.m, 64 * x.m.len() as i128 - 64) as f64 / 18_446_744_073_709_551_616.0;
    let lz = re as f64 + top.log2();
    let mut y = BigFloat::from((lz / n as f64).exp2());
    let mut w = prec + guard(prec) + 8;
    let nb = [n];
    loop {
        // Newton's steps, each doubling the bits right: `y + y·(z/y^n - 1)/n`.
        let wp = w + 20;
        let mut p = wp;
        let mut steps = Vec::new();
        while p > 40 {
            steps.push(p);
            p = p / 2 + 8;
        }
        for &p in steps.iter().rev() {
            let p = p.max(y.prec());
            let t = power(&y, &nb, p + 10, Round::Nearest);
            let u = z.div_round(&t, p + 10, Round::Nearest).0;
            let v = u.sub_round(&BigFloat::from(1), p + 10, Round::Nearest).0;
            let d = y.mul_round(&v, p + 10, Round::Nearest).0;
            let d = d.div_round(&BigFloat::from(n), p + 10, Round::Nearest).0;
            y = y.add_round(&d, p, Round::Nearest).0;
        }
        let ey = y.exponent().expect("a root") as i128;
        let delta = BigFloat::from_parts(
            &Number::new_from_u8(1, ByteOrder::LittleEndian),
            (ey + 4 - w as i128) as i64,
            1,
            Round::Nearest,
        )
        .0;
        let lo = y.sub_round(&delta, wp + 10, Round::Down).0;
        let hi = y.add_round(&delta, wp + 10, Round::Up).0;
        let up = power(&lo, &nb, wp + 10, Round::Up);
        let down = power(&hi, &nb, wp + 10, Round::Down);
        if up <= z && down >= z {
            let (_, m, lsb) = y.parts().expect("a root");
            let a = Approx {
                negative,
                m: m.to_vec(),
                lsb: lsb + qe,
                err: ey + 5 - w as i128 + qe,
            };
            let mut a = a;
            limbs::trim(&mut a.m);
            if let Some(r) = round_approx(&a, prec, round) {
                return r;
            }
        }
        w += (w / 2).max(64);
    }
}

/// `y^n` for a positive `y` by squarings, each product rounded to `p` bits in the mode `round`:
/// rounded up (down) at each step, a bound of the power from above (below).
fn power(y: &BigFloat, n: &[u64], p: u64, round: Round) -> BigFloat {
    let bits = limbs::bit_len(n);
    let mut acc = y.prec_round(p, round).0;
    for i in (0..bits - 1).rev() {
        acc = acc.mul_round(&acc, p, round).0;
        if limbs::bit(n, i) {
            acc = acc.mul_round(y, p, round).0;
        }
    }
    acc
}

/// `x^(p/q)` for a finite `x`, negative only for an odd `q`, `q ≥ 2`, `p/q` in lowest terms: the
/// root is exact or the power irrational (from `ap + bq = 1`, `x^(1/q) = (x^(p/q))^a·x^b`). An
/// exact root is raised to the power, rounded once; otherwise Ziv's strategy on a root rounded
/// toward zero (away from zero for `p < 0`) to the working precision, a relative error below
/// `2^(1 - w) = 2ε` for `w = 64(wl - 1)`, then the power cut as for integer powers: the base's
/// error exponent is 2 (3 with the reciprocal).
fn pow_rational(
    x: Op<'_>,
    p: &Number,
    pm: &[u64],
    q: u64,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let p_odd = pm[0] & 1 == 1;
    let negative = x.negative && p_odd;
    let xi = Int::of(x);
    let (odd, k) = xi.odd();
    let qi = i128::from(q);
    // The exact root, if there is one.
    let exact = if is_one(&odd) {
        (k.rem_euclid(qi) == 0).then(|| (vec![1u64], k / qi))
    } else if k.rem_euclid(qi) == 0 && limbs::bit_len(&odd) as f64 >= 1.58 * q as f64 {
        let r = int_root(&odd, q);
        (pow_exact(&r, q) == odd).then_some((r, k / qi))
    } else {
        None
    };
    if let Some((r, e)) = exact {
        let bits = limbs::bit_len(&r);
        let (mut root, _) = round_mag(false, &r, e, false, bits, Round::Nearest);
        root.set_sign(x.negative);
        return root.pow_number_round(p, prec, round);
    }
    let bits = limbs::bit_len(pm);
    let ly = (bits - 1) as f64 + (limbs::extract(pm, i128::from(bits) - 53) as f64).log2()
        - 52.0
        - (q as f64).log2();
    if let Some(up) = beyond(x, ly, p.is_negative()) {
        return range(negative, up, prec, round);
    }
    let ax = BigFloat {
        prec: 64 * x.m.len() as u64,
        kind: Kind::Finite {
            negative: false,
            exp: x.exp as i64,
            mant: x.m.to_vec(),
        },
    };
    let mut w = prec + guard(prec) + bits + 8;
    loop {
        let wl = (w / 64 + 2) as usize;
        let wb = 64 * (wl as u64 - 1);
        let (base, kb) = if p.is_negative() {
            let (r, _) = root_of(&ax, q, wb, Round::Up);
            let ri = Int::of(r.op().expect("a root"));
            (reciprocal_low(&ri, wl), 3)
        } else {
            let (r, _) = root_of(&ax, q, wb, Round::TowardZero);
            let ri = Int::of(r.op().expect("a root"));
            let (mut m, mut lsb) = (ri.m, ri.lsb);
            cut(&mut m, &mut lsb, wl);
            (Low { m, lsb }, 2)
        };
        let l = pow_low(&base, pm, wl);
        let a = approx_of(l, negative, bits, kb, wl);
        if let Some(r) = round_approx(&a, prec, round) {
            return r;
        }
        w += (w / 2).max(64);
    }
}

/// The `q`-th root, `q ≥ 2`, of a positive `x`, rounded.
fn root_of(x: &BigFloat, q: u64, prec: u64, round: Round) -> (BigFloat, Ordering) {
    if q == 2 {
        x.sqrt_round(prec, round)
    } else {
        root(x.op().expect("finite"), q, prec, round)
    }
}

/// `floor(m^(1/q))` for `q ≥ 2` and `m` not zero, without zero limbs at the top.
///
/// # Panics
///
/// For `q ≥ 2^32`, a degree whose perfect powers need more bits than a float has in practice.
fn int_root(m: &[u64], q: u64) -> Vec<u64> {
    assert!(
        q <= u64::from(u32::MAX),
        "a root of degree 2^32 or more of a perfect power"
    );
    let mut r = Vec::new();
    let mut scratch = Vec::new();
    if q == 2 {
        // Four zero limbs below, for at least three, and the root shifted back by two.
        let mut a = vec![0u64; 4];
        a.extend_from_slice(m);
        nat::sqrt_rem(&mut r, None, &a, &mut scratch);
        r.drain(..2);
        limbs::trim(&mut r);
    } else {
        nat::root(&mut r, m, q as u32, &mut scratch);
    }
    r
}
