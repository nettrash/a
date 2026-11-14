//! The exponential: `exp`, `exp2`, `exp10` and `expm1`, correctly rounded.
//!
//! The kernel reduces `x = k ln 2 + r`, `0 <= r < 1`, then `r/2^s`, sums the Taylor series of the
//! exponential of that by rectangular splitting (Smith's concurrent sums, with the coefficients
//! of a block as one-limb integers over a one-limb denominator), squares the sum `s` times and
//! multiplies by `2^k`; all in fixed point, with the bound of the error that each step adds.

use super::ball::{Ball, Rad, round_near, undecided, ziv_fast, ziv_or};
use super::consts::ln2_approx;
use super::fix::{self, SFix, Scratch};
use super::limbs;
use super::round::{overflow, underflow};
use super::ziv::Approx;
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// `floor(2^63 / ln 2)`.
pub(crate) const INV_LN2: u64 = 0xb8aa_3b29_5c17_f0bb;

/// The exponential of a fixed-point number `x` (signed, `W = 64 n` bits of fraction), `|x| <
/// 2^62`: `(y, k, ε)` with `e^x = y·2^(k - W)·(1 + θ)`, `|θ| <= ε`, `y` of `n + 1` limbs in
/// `[2^W, 3·2^W)`.
///
/// The error: let `u = 2^-W`. The reduction `r = x - k L` with `L` within `δ_L` of `ln 2` is
/// within `δ_r = |k| δ_L + u` of `x - k ln 2`; `t = floor(r 2^W / 2^s)` is within `δ_r/2^s + u`
/// of `(x - k ln 2)/2^s`, so that `e^(2^s t)` is within a factor `e^(δ_r + 2^s u)`, at most
/// `1 + 2(δ_r + 2^s u)`, of `e^(x - k ln 2)`. The sum `S` of the series of `e^t` is within `E` units
/// of `u` of `e^t` (see [`taylor`]), a relative error `ε_0 <= E u` as `S >= 1`. A truncated square
/// of `y_j` with the relative error `ε_j` has `ε_(j+1) <= 2 ε_j + ε_j² + u`, so with
/// `η_j = ε_j + u`, `η_(j+1) <= 2 η_j (1 + η_j / 2)`, and while `2^s η_0 <= 2^-20` for `s <= 1000`
/// this gives `ε_s <= 2^s (E + 1) u (1 + 2^-9)`. The two factors give `θ <= (ε_s + 2(δ_r + 2^s
/// u))(1 + 2^-9)`.
pub(crate) fn exp_fix(neg: bool, x: &[u64], n: usize) -> (Vec<u64>, i64, Rad) {
    // The arrays hold `|k| L` for `W + 2 bits(k) + 5 <= 384`: four limbs only below `2^56`.
    if n <= super::small_fix::SMALL
        && (n < 4 || limbs::bit_len(fix::trimmed(x)) <= 64 * n as u64 + 56)
    {
        let (y, k, e) = super::small_fix::exp_small(neg, x, n);
        return (y[..n + 1].to_vec(), k, e);
    }
    let w = 64 * n as u64;
    let u = Rad::pow2(-i128::from(w));
    let one = {
        let mut v = vec![0u64; n + 1];
        v[n] = 1;
        v
    };
    let x = fix::trimmed(x);
    if x.is_empty() {
        return (one, 0, Rad::ZERO);
    }
    let bx = limbs::bit_len(x);
    let ex = bx as i64 - w as i64;
    debug_assert!(ex <= 62, "an argument of the exponential too large");
    // `k` near `x / ln 2`, within 4 from the top 64 bits of `|x|`.
    let mut k: i64 = if ex <= 0 {
        if neg { -1 } else { 0 }
    } else {
        let top = if bx >= 64 {
            limbs::shr(x, bx - 64)[0]
        } else {
            x[0] << (64 - bx)
        };
        let q = ((u128::from(top) * u128::from(INV_LN2)) >> (127 - ex)) as i64;
        if neg { -q } else { q }
    };
    let mut scratch = Scratch::default();
    let (r, delta_r) = if k == 0 && !neg && ex <= 0 {
        (x.to_vec(), Rad::ZERO)
    } else {
        // `L`: ln 2 with `W + g` bits of fraction, `g` enough that `|k|` units of its last bit lie
        // below `2^-W / 16`.
        let g = 64 - (k.unsigned_abs() + 8).leading_zeros() as u64 + 4;
        let a = ln2_approx(w + g + 8);
        let (lf, dropped) = fix::of_mag(&a.m, a.lsb + i128::from(g), n);
        let mut delta_l = Rad::pow2(a.err);
        if dropped {
            delta_l = delta_l.add(Rad::pow2(-i128::from(w + g)));
        }
        let xs = SFix { neg, m: x.to_vec() };
        loop {
            let kl = fix::shr(&fix::mul1(&lf, k.unsigned_abs()), g);
            let kl = SFix { neg: k < 0, m: kl };
            let r = xs.sub(&kl);
            if r.neg && !r.is_zero() {
                k -= 1;
                continue;
            }
            if fix::bits(&r.m) > w {
                k += 1;
                continue;
            }
            let dr = delta_l.mul(Rad::int(k.unsigned_abs())).add(u);
            break (r.m, dr);
        }
    };
    let r = fix::trimmed(&r);
    if r.is_empty() {
        return (one, k, Rad::int(2).mul(delta_r).add(u));
    }
    let er = limbs::bit_len(r) as i64 - w as i64;
    let burst = n >= BB_LIMBS;
    let s = (if burst { S_BB } else { s_target(w) } + er).max(0) as u64;
    let t = fix::shr(r, s);
    // `t < 2^-tau`.
    let tau = (s as i64 - er) as u64;
    let (mut y, e) = if burst {
        bit_burst(&t, tau, n, &mut scratch)
    } else {
        taylor(&t, tau, n, &mut scratch)
    };
    let mut tmp = Vec::with_capacity(n + 2);
    for _ in 0..s {
        fix::mul_to(&mut tmp, &y, &y, n, n + 1, &mut scratch);
        core::mem::swap(&mut y, &mut tmp);
    }
    let slack = Rad::up(513, -9);
    let eps_s = Rad::int(e + 1).mul(u).shl(i128::from(s)).mul(slack);
    if !eps_s.le(Rad::pow2(-20)) {
        // Outside the bound's assumption: an error of 1 never decides.
        return (y, k, Rad::int(1));
    }
    let eps_in = delta_r.add(u.shl(i128::from(s))).shl(1);
    (y, k, eps_s.add(eps_in).mul(slack))
}

/// The squarings of the bit burst.
const S_BB: i64 = 40;

/// The least `s` the kernel reduces its argument below `2^-s` by squarings, for `W` bits: about
/// `0.3 sqrt(W)`, which balances the squarings against the blocks of the series.
#[inline]
pub(crate) fn s_target(w: u64) -> i64 {
    ((3 * isqrt(w) + 5) / 10) as i64
}

/// `floor(sqrt(n))`.
#[inline]
pub(crate) fn isqrt(n: u64) -> u64 {
    let mut r = (n as f64).sqrt() as u64;
    while r * r > n {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

/// The number of terms `N` of the series of `e^t` for `t < 2^-tau` to `W` bits: the least with
/// `2 t^(N+1) / (N+1)! <= 2^-W`, the bound of the terms left out (each at most half the one
/// before while `t <= 1`), from a lower bound of `log2 (N+1)!`.
pub(crate) fn terms(tau: u64, w: u64) -> u64 {
    // In sixteenths of a bit.
    let need = 16 * (w + 1);
    let mut have = 0u64;
    let mut i = 0u64;
    loop {
        i += 1;
        have += 16 * tau + fix::log2_16(i);
        if have >= need {
            return i - 1;
        }
    }
}

/// The sum `S` of the series `Σ_(i <= N) t^i / i!` for the fixed-point `t < 2^-tau <= 1` (`n` limbs),
/// by rectangular splitting, and the bound `E` of its error in units of `u = 2^-W`, the terms
/// left out included.
///
/// The terms are taken in blocks `[a, b)` from the top, `Q = (a+1)…b` below `2^64` and the block
/// no longer than the powers `t^1..t^m` computed: `S_a = (Σ_(a <= i < b) c_i t^(i-a) + t^(b-a)
/// S_b) / Q` with the integers `c_i = (i+1)…b`. With `S_b < 2` (it is at most `1/(1 - t/2)`),
/// `t^j` within `j` units (each product truncated, `t < 1`), the error of a block is at most
/// `(E_b + 2(b-a) + 1 + Σ c_i (i-a)) / Q + 1`, and as `c_i / Q <= 1/(i-a)!`, `Q >= (b-a)!`,
/// `Σ j/j! = e`, at most `ceil(E_b / Q) + 8`. The terms left out add one unit.
pub(crate) fn taylor(t: &[u64], tau: u64, n: usize, scratch: &mut Scratch) -> (Vec<u64>, u64) {
    let w = 64 * n as u64;
    let big_n = terms(tau, w);
    // The longest block whose denominator fits a limb at the top, and the powers it needs.
    let lq = (64 * 16 / fix::log2_16(big_n + 2).max(16)).max(1);
    let m = (isqrt(big_n) + 1).min(lq).max(1) as usize;
    // `t^j` at `pw[j n..(j + 1) n]`.
    let mut pw = vec![0u64; (m + 1) * n];
    let k = t.len().min(n);
    pw[n..n + k].copy_from_slice(&t[..k]);
    let mut tmp = Vec::with_capacity(n + 2);
    for j in 2..=m {
        let (lo, hi) = pw.split_at_mut(j * n);
        fix::mul_to(&mut tmp, &lo[(j - 1) * n..], &lo[n..2 * n], n, n, scratch);
        hi[..n].copy_from_slice(&tmp);
    }
    let pow = |j: usize| &pw[j * n..(j + 1) * n];
    let mut sum: Vec<u64> = vec![0u64; n + 2];
    let mut acc: Vec<u64> = Vec::with_capacity(n + 2);
    let mut first = true;
    let mut err = 0u64;
    let mut b = big_n + 1;
    while b > 0 {
        let mut a = b;
        let mut q: u64 = 1;
        while a > 0 && ((b - a) as usize) < m {
            match q.checked_mul(a) {
                Some(v) => {
                    q = v;
                    a -= 1;
                }
                None => break,
            }
        }
        let len = (b - a) as usize;
        if first {
            acc.clear();
            acc.resize(n + 2, 0);
            first = false;
        } else {
            fix::mul_to(&mut acc, pow(len), &sum[..n + 1], n, n + 2, scratch);
        }
        // `c_i` from the top: `c_(b-1) = b`, `c_i = c_(i+1) (i+1)`.
        let mut c: u64 = 1;
        let mut i = b;
        while i > a {
            i -= 1;
            c *= i + 1;
            if i == a {
                fix::add_at(&mut acc, n, c);
            } else {
                fix::addmul1(&mut acc, pow((i - a) as usize), c);
            }
        }
        fix::div1(&mut acc, q);
        err = err.div_ceil(q) + 8;
        core::mem::swap(&mut sum, &mut acc);
        b = a;
    }
    sum.truncate(n + 1);
    (sum, err + 1)
}

/// Where an exponential lies beyond the exponent range for any mode: far above or far below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Range {
    Over,
    Under,
}

/// An exponential in the raw: `±M·2^lsb` with a relative error bound, its exponent unbounded.
pub(crate) struct Raw {
    pub(crate) m: Vec<u64>,
    pub(crate) lsb: i128,
    pub(crate) rel: Rad,
}

impl Raw {
    /// The approximation of the value of the sign `negative` for Ziv's loop.
    pub(crate) fn approx(self, negative: bool) -> Approx {
        let mut m = self.m;
        limbs::trim(&mut m);
        let bits = limbs::bit_len(&m) as i128;
        // `|v| < 2^(lsb + bits)`.
        let err = self
            .rel
            .shl(self.lsb + bits)
            .exp2_ceil()
            .max(self.lsb - 128);
        if err >= self.lsb + bits {
            return undecided();
        }
        Approx {
            negative,
            m,
            lsb: self.lsb,
            err,
        }
    }

    /// The ball, when its exponent lies in the range.
    pub(crate) fn ball(&self) -> Option<Ball> {
        let mut m = self.m.clone();
        limbs::trim(&mut m);
        let bits = limbs::bit_len(&m) as i128;
        let e = self.lsb + bits;
        if e > i128::from(BigFloat::EXP_MAX) - 2 || e < i128::from(BigFloat::EXP_MIN) + 2 {
            return None;
        }
        let rad = self.rel.shl(e);
        Some(Ball::of_limbs(false, &m, self.lsb, rad))
    }
}

/// The exponential of the ball `x` with a relative error about `2^-w`, or where it lies beyond
/// the exponent range: `|x| >= 2^62` is, as `e^(2^62) > 2^(2^62)`.
///
/// The midpoint is put in fixed point with `W >= w + s + 16` bits, a truncation below `2^-W`,
/// which multiplies the result by at most `e^u <= 1 + 2u`; the radius `ρ <= 1/2` by at most
/// `e^ρ <= 1 + 2ρ`.
pub(crate) fn exp_ball(x: &Ball, w: u64) -> Result<Raw, Range> {
    let w = w.max(8);
    if x.mid.is_zero() {
        // `e^θ` for `|θ| <= ρ`.
        return Ok(Raw {
            m: vec![1],
            lsb: 0,
            rel: x.rad.shl(1),
        });
    }
    let (negative, m, lsb) = x.mid.parts().expect("finite");
    let ex = x.mid.exponent().expect("finite");
    if ex >= 63 {
        return Err(if negative { Range::Under } else { Range::Over });
    }
    let n = (w + s_target(w + 64) as u64 + 24).div_ceil(64) as usize;
    let wb = 64 * n as i128;
    let (xf, dropped) = fix::of_mag(m, lsb, n);
    let (y, k, mut rel) = exp_fix(negative, &xf, n);
    if dropped {
        rel = rel.add(Rad::pow2(1 - wb));
    }
    if !x.rad.le(Rad::int(1)) {
        return Ok(Raw {
            m: y,
            lsb: i128::from(k) - wb,
            rel: Rad::int(1),
        });
    }
    if !x.rad.is_zero() {
        rel = rel.add(x.rad.shl(1)).add(rel.mul(x.rad.shl(1)));
    }
    Ok(Raw {
        m: y,
        lsb: i128::from(k) - wb,
        rel,
    })
}

/// The result of an exponential beyond the range for the sign `negative`.
pub(crate) fn out_of_range(
    r: Range,
    negative: bool,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    match r {
        Range::Over => overflow(negative, prec, round),
        Range::Under => underflow(
            negative,
            i128::from(BigFloat::EXP_MIN) - 2,
            false,
            prec,
            round,
        ),
    }
}

/// A result that is exact.
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// The exponent bound of `|x|`, `|x| < 2^e`, of a finite float that is not zero.
#[inline]
pub(crate) fn expo(x: &BigFloat) -> i64 {
    x.exponent().expect("finite")
}

impl BigFloat {
    /// `e^self` rounded to `prec` bits in the mode `round`, and the direction of the rounding.
    ///
    /// `e^0 = 1` exactly; `e^+∞ = +∞`, `e^-∞ = +0`; a result beyond the exponent range overflows
    /// or underflows by the mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (e, o) = BigFloat::from(1).exp_round(53, Round::Nearest);
    /// assert_eq!(e, BigFloat::from(core::f64::consts::E));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn exp_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan => exact(BigFloat::nan(prec)),
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Inf { negative: true } => exact(BigFloat::zero(prec)),
            Kind::Zero { .. } => exact(BigFloat::from(1).with_prec(prec)),
            Kind::Finite { negative, .. } => {
                let ex = expo(self);
                if ex >= 63 {
                    let r = if *negative { Range::Under } else { Range::Over };
                    return out_of_range(r, false, prec, round);
                }
                // `e^x = 1 + x + …`, within `2|x| < 2^(ex + 1)` of 1, on the side of `x`.
                if ex < -(prec as i64) - 4 {
                    let side = if *negative {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                    if let Some(r) =
                        round_near(&BigFloat::from(1), side, i128::from(ex) + 1, prec, round)
                    {
                        return r;
                    }
                }
                let x = Ball::exact(self.clone());
                ziv_fast(prec, round, |w| match exp_ball(&x, w) {
                    Ok(raw) => raw.approx(false),
                    Err(_) => unreachable!("an argument below 2^62"),
                })
            }
        }
    }

    /// `e^self` to nearest at the precision of `self`.
    pub fn exp(&self) -> BigFloat {
        self.exp_round(self.prec, Round::Nearest).0
    }
}

/// The integer and fractional parts of a finite `x`, `|x| < 2^63`: `n = floor(x)` and `x - n` in
/// `[0, 1)`, exactly.
pub(crate) fn floor_split(x: &BigFloat) -> (i64, BigFloat) {
    let (negative, m, lsb) = match x.parts() {
        Some(p) => p,
        None => return (0, BigFloat::zero(x.prec)),
    };
    let ip = if lsb >= 0 {
        limbs::shl(m, lsb as u64)
    } else {
        limbs::shr(m, (-lsb) as u64)
    };
    debug_assert!(limbs::bit_len(&ip) <= 63);
    let mut n = ip.first().copied().unwrap_or(0) as i64;
    let frac_zero = lsb >= 0 || limbs::zero_below(m, (-lsb) as u64);
    if negative {
        n = -n;
        if !frac_zero {
            n -= 1;
        }
    }
    let f = if frac_zero {
        BigFloat::zero(x.prec)
    } else {
        // `x - n` lies in `(0, 1)` with its last bit that of `x`: `-lsb` bits at most.
        let bits = ((-lsb) as u64).max(x.prec).max(1);
        let (f, o) = x.sub_round(&BigFloat::from(n), bits, Round::Nearest);
        debug_assert_eq!(o, Ordering::Equal);
        f
    };
    (n, f)
}

/// Whether a finite float that is not zero is an integer, and if so whether it is odd.
pub(crate) fn integer_parity(x: &BigFloat) -> Option<bool> {
    let (_, m, lsb) = x.parts()?;
    let tz = i128::from(limbs::trailing_zeros(m));
    let e = lsb + tz;
    if e < 0 { None } else { Some(e == 0) }
}

impl BigFloat {
    /// `2^self` rounded to `prec` bits in the mode `round`, and the direction of the rounding:
    /// exact for an integer; `2^+∞ = +∞`, `2^-∞ = +0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// assert_eq!(BigFloat::from(-3).exp2_round(5, Round::Up), (BigFloat::from(0.125), Ordering::Equal));
    /// let (r, o) = BigFloat::from(0.5).exp2_round(53, Round::Nearest);
    /// assert_eq!(r, BigFloat::from(core::f64::consts::SQRT_2));
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    #[track_caller]
    pub fn exp2_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan => exact(BigFloat::nan(prec)),
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Inf { negative: true } => exact(BigFloat::zero(prec)),
            Kind::Zero { .. } => exact(BigFloat::from(1).with_prec(prec)),
            Kind::Finite { negative, .. } => {
                let ex = expo(self);
                if ex >= 64 {
                    let r = if *negative { Range::Under } else { Range::Over };
                    return out_of_range(r, false, prec, round);
                }
                if integer_parity(self).is_some() {
                    let (n, _) = floor_split(self);
                    return super::round::round_mag(false, &[1], i128::from(n), false, prec, round);
                }
                // `2^x = 1 + x ln 2 + …`, within `|x|` of 1 on the side of `x`.
                if ex < -(prec as i64) - 2 {
                    let side = if *negative {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                    if let Some(r) =
                        round_near(&BigFloat::from(1), side, i128::from(ex), prec, round)
                    {
                        return r;
                    }
                }
                let (n, f) = floor_split(self);
                let f = Ball::exact(f);
                ziv_fast(prec, round, |w| {
                    let t = super::log::ln2_ball(w + 8).mul(&f, w + 8);
                    match exp_ball(&t, w) {
                        Ok(mut raw) => {
                            raw.lsb += i128::from(n);
                            raw.approx(false)
                        }
                        Err(_) => unreachable!("an argument below 1"),
                    }
                })
            }
        }
    }

    /// `2^self` to nearest at the precision of `self`.
    pub fn exp2(&self) -> BigFloat {
        self.exp2_round(self.prec, Round::Nearest).0
    }

    /// `10^self` rounded to `prec` bits in the mode `round`, and the direction of the rounding:
    /// exact for an integer `n >= 0` with `5^n` of at most `prec` bits; `10^+∞ = +∞`,
    /// `10^-∞ = +0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// assert_eq!(BigFloat::from(3).exp10_round(10, Round::Down), (BigFloat::from(1000), Ordering::Equal));
    /// let (r, o) = BigFloat::from(-1).exp10_round(53, Round::Nearest);
    /// assert_eq!(r, BigFloat::from(0.1));
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    #[track_caller]
    pub fn exp10_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan => exact(BigFloat::nan(prec)),
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Inf { negative: true } => exact(BigFloat::zero(prec)),
            Kind::Zero { .. } => exact(BigFloat::from(1).with_prec(prec)),
            Kind::Finite { negative, .. } => {
                let ex = expo(self);
                if ex >= 62 {
                    let r = if *negative { Range::Under } else { Range::Over };
                    return out_of_range(r, false, prec, round);
                }
                if !*negative && integer_parity(self).is_some() {
                    // `10^n = 5^n 2^n`, `5^n` of more than `2n` bits: exact or a midpoint only for
                    // `n <= (prec + 1) / 2`, which the rounding of the integer decides.
                    let (n, _) = floor_split(self);
                    if (n as u64) <= prec.div_ceil(2) {
                        let five = pow_u64(&crate::math::numbers::Number::from(5), n as u64);
                        let l = limbs::of_number(&five);
                        if limbs::bit_len(&l) <= prec + 1 {
                            return super::round::round_mag(
                                false,
                                &l,
                                i128::from(n),
                                false,
                                prec,
                                round,
                            );
                        }
                    }
                }
                // `10^x = 1 + x ln 10 + …`, within `3|x|` of 1 on the side of `x`.
                if ex < -(prec as i64) - 4 {
                    let side = if *negative {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                    if let Some(r) =
                        round_near(&BigFloat::from(1), side, i128::from(ex) + 2, prec, round)
                    {
                        return r;
                    }
                }
                let x = self.clone();
                let more = ex.max(0) as u64;
                ziv_or(prec, round, |w| {
                    let t = super::log::ln10_ball(w + more + 8).mul_float(&x, w + more + 8);
                    match exp_ball(&t, w) {
                        Ok(raw) => Ok(raw.approx(false)),
                        // `|x ln 10| >= 2^62`: far beyond the range.
                        Err(r) => Err(out_of_range(r, false, prec, round)),
                    }
                })
            }
        }
    }

    /// `10^self` to nearest at the precision of `self`.
    pub fn exp10(&self) -> BigFloat {
        self.exp10_round(self.prec, Round::Nearest).0
    }

    /// `e^self - 1` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, accurate for `self` near zero, as MPFR's `mpfr_expm1`.
    ///
    /// `±0` for `±0`; `+∞` for `+∞`; `-1` for `-∞`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use core::cmp::Ordering;
    ///
    /// // e^(2^-100) - 1 is just above 2^-100.
    /// let (x, _) = BigFloat::from_parts(&Number::from(1), -100, 53, Round::Nearest);
    /// let (r, o) = x.exp_m1_round(53, Round::Up);
    /// assert!(r > x);
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    #[track_caller]
    pub fn exp_m1_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan => exact(BigFloat::nan(prec)),
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Inf { negative: true } => exact(BigFloat::from(-1).with_prec(prec)),
            Kind::Zero { negative } => exact(BigFloat::special(
                prec,
                Kind::Zero {
                    negative: *negative,
                },
            )),
            Kind::Finite { negative, .. } => {
                let ex = expo(self);
                let negative = *negative;
                if ex >= 63 && !negative {
                    return out_of_range(Range::Over, false, prec, round);
                }
                // `e^x - 1` for `x <= -(prec + 4)`: above -1 by `e^x < 2^-(prec + 4)`.
                if negative
                    && (ex >= 63 || *self <= -(prec as i64) - 4)
                    && let Some(r) = round_near(
                        &BigFloat::from(-1),
                        Ordering::Greater,
                        -(prec as i128) - 4,
                        prec,
                        round,
                    )
                {
                    return r;
                }
                // `e^x - 1 = x + x²/2 + …`, above `x`, within `x²` of it for `|x| < 1/2`.
                if ex < -(prec as i64) - 2
                    && let Some(r) =
                        round_near(self, Ordering::Greater, 2 * i128::from(ex), prec, round)
                {
                    return r;
                }
                let x = Ball::exact(self.clone());
                let one = Ball::exact(BigFloat::from(1));
                let more = (-ex).max(0) as u64;
                ziv_fast(prec, round, |w| {
                    let raw = match exp_ball(&x, w + more + 4) {
                        Ok(raw) => raw,
                        Err(_) => unreachable!("an argument below 2^62"),
                    };
                    if !negative && ex >= 1 && self >= &BigFloat::from(w + 16) {
                        // `e^x - 1 = e^x (1 - e^-x)`, `e^-x < 2^-(w + 16)`.
                        let mut raw = raw;
                        raw.rel = raw.rel.add(Rad::pow2(-i128::from(w) - 16));
                        return raw.approx(false);
                    }
                    match raw.ball() {
                        Some(e) => e.sub(&one, w + 4).approx(),
                        None => undecided(),
                    }
                })
            }
        }
    }

    /// `e^self - 1` to nearest at the precision of `self`.
    pub fn exp_m1(&self) -> BigFloat {
        self.exp_m1_round(self.prec, Round::Nearest).0
    }
}

/// `b^n` for a `u64` exponent, by squarings.
pub(crate) fn pow_u64(b: &crate::math::numbers::Number, n: u64) -> crate::math::numbers::Number {
    if let Ok(k) = u32::try_from(n) {
        return b.pow(k);
    }
    let half = pow_u64(b, n / 2);
    let sq = &half * &half;
    if n % 2 == 1 { &sq * b } else { sq }
}

/// The kernel takes the bit-burst sum of [`bit_burst`] from this many limbs of fraction on.
pub(crate) const BB_LIMBS: usize = 176;

/// `(P, Q, T)` of the binary splitting of `Σ_(a < k <= b) Π_(a < i <= k) p / (i 2^r)`: `P =
/// p^(b-a)`, `Q = (a+1)…b` and the sum `T / (Q 2^(r (b-a)))`, exactly.
fn bsplit(p: &[u64], r: u64, a: u64, b: u64, need_p: bool) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    if b - a <= 12 {
        // The terms one by one: `(P, Q, T) ⊕ (p, k, p) = (P p, Q k, T k 2^r + P p)`.
        let mut pp = vec![1u64];
        let mut q = vec![1u64];
        let mut t: Vec<u64> = Vec::new();
        for k in a + 1..=b {
            let np = limbs::mul(&pp, p);
            let mut tk = fix::mul1(&t, k);
            limbs::trim(&mut tk);
            t = limbs::add(&limbs::shl(&tk, r), &np);
            pp = np;
            q = fix::mul1(&q, k);
            limbs::trim(&mut q);
        }
        return (pp, q, t);
    }
    let m = a + (b - a) / 2;
    let (p1, q1, t1) = bsplit(p, r, a, m, true);
    let (p2, q2, t2) = bsplit(p, r, m, b, need_p);
    let q = limbs::mul(&q1, &q2);
    let t = limbs::add(
        &limbs::shl(&limbs::mul(&t1, &q2), r * (b - m)),
        &limbs::mul(&p1, &t2),
    );
    let pp = if need_p {
        limbs::mul(&p1, &p2)
    } else {
        Vec::new()
    };
    (pp, q, t)
}

/// `e^t` for the fixed-point `t < 2^-tau` of `n` limbs (`W = 64 n`) by the bit burst: `t` cut
/// into pieces `t_j = p_j / 2^(r_j)` of the bits from `2^-r_(j-1)` to `2^-r_j`, `r_j` twice
/// `r_(j-1)`, each `e^(t_j)` the sum of its series to the term `N_j` by binary splitting, exactly,
/// then one division, the product of the pieces truncated at each step. Returns the value, of `n
/// + 1` limbs, and the bound `E` of its error in units of `2^-W`.
///
/// A piece's sum is exact before its division, which truncates by one unit, and the terms left
/// out, at most `2 t_j^(N+1)/(N+1)!`, add one: a relative `2u` as `e^(t_j) >= 1`. The product of
/// `C` pieces with `C - 1` truncated products is within a relative `3Cu (1 + 2^-40)`, and as `e^t
/// < 3`, `E = 9C + 1` units bound its error.
pub(crate) fn bit_burst(t: &[u64], tau: u64, n: usize, scratch: &mut Scratch) -> (Vec<u64>, u64) {
    let w = 64 * n as u64;
    let mut y: Vec<u64> = {
        let mut v = vec![0u64; n + 1];
        v[n] = 1;
        v
    };
    let mut pieces = 0u64;
    let mut lo = tau;
    while lo < w {
        let hi = if lo < 16 { lo + 16 } else { 2 * lo }.min(w);
        // `p = floor(t 2^hi) mod 2^(hi - lo)`.
        let mut p = fix::shr(t, w - hi);
        let keep = hi - lo;
        let k = (keep / 64) as usize;
        if p.len() > k {
            let rem = keep % 64;
            p.truncate(k + usize::from(rem != 0));
            if rem != 0 {
                p[k] &= (1u64 << rem) - 1;
            }
        }
        limbs::trim(&mut p);
        let prev = lo;
        lo = hi;
        if p.is_empty() {
            continue;
        }
        // `t_j < 2^-prev`: the terms to `N` with `2 t_j^(N+1)/(N+1)! <= 2^-W`.
        let big_n = terms(prev.max(1), w).max(1);
        let (_, q, tt) = bsplit(&p, hi, 0, big_n, false);
        // `1 + T / (Q 2^(hi N))` in fixed point.
        let shift = i128::from(w) - i128::from(hi) * i128::from(big_n);
        let (num, den) = if shift >= 0 {
            (limbs::shl(&tt, shift as u64), q)
        } else {
            (tt, limbs::shl(&q, (-shift) as u64))
        };
        let nn = limbs::to_number(&num, false, crate::math::numbers::ByteOrder::LittleEndian);
        let dd = limbs::to_number(&den, false, crate::math::numbers::ByteOrder::LittleEndian);
        let mut s = limbs::of_number(&(&nn / &dd));
        s.resize(n + 1, 0);
        fix::add_at(&mut s, n, 1);
        y = fix::mul(&y, &s, n, n + 1, scratch);
        pieces += 1;
    }
    (y, 9 * pieces.max(1) + 1)
}
