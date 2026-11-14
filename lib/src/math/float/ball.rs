//! Balls for the functions that can only be approximated: a midpoint, a float of a working
//! precision, and a radius that bounds the distance from it to the value meant, as in Arb's ball
//! arithmetic, with every rounding of a midpoint added to the radius. A function built from these
//! operations has a sound error bound by construction, which [`Ball::approx`] hands to Ziv's loop.
//!
//! The radii are [`Rad`]s, numbers of 32 bits and an exponent rounded up at each step, so that a
//! bound loses at most a few parts in `2^31` at each operation, against a bit at each when it is a
//! power of two.

use super::limbs;
use super::round::round_mag;
use super::ziv::Approx;
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// The exponent that stands for "no error": below any that can occur.
pub(crate) const EXACT: i128 = i128::MIN / 4;

/// A non-negative real `m·2^e`, `m` of at most 32 bits (`2^31 ≤ m < 2^32` unless it is zero): an
/// upper bound wherever it is made by a function whose name ends in `up` and a lower one in `down`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rad {
    pub(crate) m: u64,
    pub(crate) e: i128,
}

impl Rad {
    /// Zero.
    pub(crate) const ZERO: Rad = Rad { m: 0, e: 0 };

    /// `2^e`, exactly.
    #[inline]
    pub(crate) fn pow2(e: i128) -> Rad {
        Rad {
            m: 1 << 31,
            e: e - 31,
        }
    }

    /// `m·2^e` rounded up to 32 bits.
    #[inline]
    pub(crate) fn up(m: u128, e: i128) -> Rad {
        if m == 0 {
            return Rad::ZERO;
        }
        let bits = 128 - m.leading_zeros();
        if bits <= 32 {
            let s = 32 - bits;
            return Rad {
                m: (m << s) as u64,
                e: e - i128::from(s),
            };
        }
        let s = bits - 32;
        let mut q = (m >> s) as u64;
        let mut e = e + i128::from(s);
        if m & ((1u128 << s) - 1) != 0 {
            q += 1;
            if q >> 32 != 0 {
                q >>= 1;
                e += 1;
            }
        }
        Rad { m: q, e }
    }

    /// `m·2^e` rounded down to 32 bits.
    #[inline]
    pub(crate) fn down(m: u128, e: i128) -> Rad {
        if m == 0 {
            return Rad::ZERO;
        }
        let bits = 128 - m.leading_zeros();
        if bits <= 32 {
            let s = 32 - bits;
            Rad {
                m: (m << s) as u64,
                e: e - i128::from(s),
            }
        } else {
            let s = bits - 32;
            Rad {
                m: (m >> s) as u64,
                e: e + i128::from(s),
            }
        }
    }

    /// The integer `n`, rounded up.
    #[inline]
    pub(crate) fn int(n: u64) -> Rad {
        Rad::up(u128::from(n), 0)
    }

    /// Whether it is zero.
    #[inline]
    pub(crate) fn is_zero(self) -> bool {
        self.m == 0
    }

    /// An upper bound of the sum of two upper bounds.
    pub(crate) fn add(self, o: Rad) -> Rad {
        if self.m == 0 {
            return o;
        }
        if o.m == 0 {
            return self;
        }
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = a.e - b.e;
        if d >= 64 {
            // `b < 2^(b.e + 32) <= 2^(a.e - 32)`: one unit of `2^(a.e - 32)` covers it.
            Rad::up((u128::from(a.m) << 32) + 1, a.e - 32)
        } else {
            Rad::up((u128::from(a.m) << d) + u128::from(b.m), b.e)
        }
    }

    /// A lower bound of `self - o` for a lower bound `self` and an upper bound `o`, `None` when it
    /// is not above zero.
    pub(crate) fn sub_down(self, o: Rad) -> Option<Rad> {
        if self.m == 0 {
            return None;
        }
        if o.m == 0 {
            return Some(self);
        }
        // Both normalized: `m·2^e` lies in `[2^(e + 31), 2^(e + 32))`.
        if o.e > self.e {
            return None;
        }
        let d = self.e - o.e;
        if d >= 64 {
            // `o < 2^(self.e - 32)`.
            return Some(Rad::down((u128::from(self.m) << 32) - 1, self.e - 32));
        }
        let a = u128::from(self.m) << d;
        let b = u128::from(o.m);
        if a <= b {
            None
        } else {
            Some(Rad::down(a - b, o.e))
        }
    }

    /// An upper bound of the product of two upper bounds (or a lower bound of two lower bounds
    /// with [`mul_down`](Self::mul_down)).
    #[inline]
    pub(crate) fn mul(self, o: Rad) -> Rad {
        Rad::up(u128::from(self.m) * u128::from(o.m), self.e + o.e)
    }

    /// A lower bound of the product of two lower bounds.
    #[inline]
    pub(crate) fn mul_down(self, o: Rad) -> Rad {
        Rad::down(u128::from(self.m) * u128::from(o.m), self.e + o.e)
    }

    /// An upper bound of `self / o` for an upper bound `self` and a lower bound `o`, not zero.
    #[inline]
    pub(crate) fn div(self, o: Rad) -> Rad {
        debug_assert!(o.m != 0);
        if self.m == 0 {
            return Rad::ZERO;
        }
        let n = u128::from(self.m) << 64;
        let d = u128::from(o.m);
        Rad::up(n.div_ceil(d), self.e - o.e - 64)
    }

    /// `self·2^k`, exactly.
    #[inline]
    pub(crate) fn shl(self, k: i128) -> Rad {
        if self.m == 0 {
            self
        } else {
            Rad {
                m: self.m,
                e: self.e + k,
            }
        }
    }

    /// The least `E` with `self <= 2^E`, [`EXACT`] for zero.
    #[inline]
    pub(crate) fn exp2_ceil(self) -> i128 {
        if self.m == 0 {
            EXACT
        } else {
            let bits = 64 - self.m.leading_zeros();
            let pow = self.m.is_power_of_two();
            self.e + i128::from(bits) - i128::from(pow)
        }
    }

    /// Whether `self <= o`.
    #[inline]
    pub(crate) fn le(self, o: Rad) -> bool {
        if self.m == 0 {
            return true;
        }
        if o.m == 0 {
            return false;
        }
        if self.e != o.e {
            // Both normalized.
            return self.e < o.e;
        }
        self.m <= o.m
    }
}

/// Bounds of the magnitude of a float that is finite and not zero: `(lower, upper)`, from the top
/// 32 bits of its mantissa.
#[inline]
pub(crate) fn mag_bounds(x: &BigFloat) -> (Rad, Rad) {
    match &x.kind {
        Kind::Finite { exp, mant, .. } => {
            let top = mant[mant.len() - 1] >> 32;
            let e = i128::from(*exp) - 32;
            let exact =
                mant[mant.len() - 1] << 32 == 0 && mant[..mant.len() - 1].iter().all(|&l| l == 0);
            let hi = if exact {
                Rad { m: top, e }
            } else {
                Rad::up(u128::from(top) + 1, e)
            };
            (Rad { m: top, e }, hi)
        }
        _ => (Rad::ZERO, Rad::ZERO),
    }
}

/// A real known to lie within `rad` of the float `mid`.
#[derive(Clone, Debug)]
pub(crate) struct Ball {
    pub(crate) mid: BigFloat,
    pub(crate) rad: Rad,
}

/// The error bound of a midpoint rounded to nearest at `w` bits with the direction `o`: half its
/// last place when it is not exact; a bound that never decides when the rounding left the
/// exponent range (which the functions' arguments exclude).
#[inline]
fn half_ulp(x: &BigFloat, o: Ordering, w: u64) -> Rad {
    match (o, x.exponent()) {
        (Ordering::Equal, _) => Rad::ZERO,
        (_, None) => Rad::pow2(1 << 70),
        (_, Some(e)) => Rad::pow2(i128::from(e) - i128::from(w) - 1),
    }
}

impl Ball {
    /// The float `x`, exactly.
    #[inline]
    pub(crate) fn exact(x: BigFloat) -> Ball {
        Ball {
            mid: x,
            rad: Rad::ZERO,
        }
    }

    /// The ball of a midpoint and a radius.
    #[inline]
    pub(crate) fn new(mid: BigFloat, rad: Rad) -> Ball {
        Ball { mid, rad }
    }

    /// The number `±M·2^lsb` of the limbs `m` exactly (zero for no limbs), with the radius `rad`.
    pub(crate) fn of_limbs(negative: bool, m: &[u64], lsb: i128, rad: Rad) -> Ball {
        let mut v = m.to_vec();
        limbs::trim(&mut v);
        if v.is_empty() {
            return Ball::new(BigFloat::zero(1), rad);
        }
        let bits = limbs::bit_len(&v);
        let z = limbs::trailing_zeros(&v);
        let prec = (bits - z).max(1);
        let (x, _) = round_mag(negative, &v, lsb, false, prec, Round::Nearest);
        Ball::new(x, rad)
    }

    /// A lower bound of the magnitude of the value, `None` when the ball holds zero.
    #[inline]
    pub(crate) fn lower(&self) -> Option<Rad> {
        if !self.mid.is_normal() {
            return None;
        }
        mag_bounds(&self.mid).0.sub_down(self.rad)
    }

    /// `-self`.
    pub(crate) fn neg(mut self) -> Ball {
        let n = self.mid.is_sign_negative();
        self.mid.set_sign(!n);
        self
    }

    /// `self·2^k`, exactly; `k` small enough for the exponent range.
    pub(crate) fn shl(&self, k: i64) -> Ball {
        let mut mid = self.mid.clone();
        if let Kind::Finite { exp, .. } = &mut mid.kind {
            *exp += k;
            debug_assert!((BigFloat::EXP_MIN..=BigFloat::EXP_MAX).contains(exp));
        }
        Ball::new(mid, self.rad.shl(i128::from(k)))
    }

    /// `self + o`, the midpoint to `w` bits.
    pub(crate) fn add(&self, o: &Ball, w: u64) -> Ball {
        let (mid, t) = self.mid.add_round(&o.mid, w, Round::Nearest);
        let rad = self.rad.add(o.rad).add(half_ulp(&mid, t, w));
        Ball::new(mid, rad)
    }

    /// `self - o`, the midpoint to `w` bits.
    pub(crate) fn sub(&self, o: &Ball, w: u64) -> Ball {
        let (mid, t) = self.mid.sub_round(&o.mid, w, Round::Nearest);
        let rad = self.rad.add(o.rad).add(half_ulp(&mid, t, w));
        Ball::new(mid, rad)
    }

    /// `self·o`, the midpoint to `w` bits: `|ab - AB| <= |A| rb + |B| ra + ra rb`.
    pub(crate) fn mul(&self, o: &Ball, w: u64) -> Ball {
        let (mid, t) = self.mid.mul_round(&o.mid, w, Round::Nearest);
        let (_, ha) = mag_bounds(&self.mid);
        let (_, hb) = mag_bounds(&o.mid);
        let rad = ha
            .mul(o.rad)
            .add(hb.mul(self.rad))
            .add(self.rad.mul(o.rad))
            .add(half_ulp(&mid, t, w));
        Ball::new(mid, rad)
    }

    /// `self·x` for a float `x` taken exactly, the midpoint to `w` bits.
    pub(crate) fn mul_float(&self, x: &BigFloat, w: u64) -> Ball {
        let (mid, t) = self.mid.mul_round(x, w, Round::Nearest);
        let (_, hx) = mag_bounds(x);
        let rad = hx.mul(self.rad).add(half_ulp(&mid, t, w));
        Ball::new(mid, rad)
    }

    /// `self / o`, the midpoint to `w` bits, `None` when `o` may be zero:
    /// `|a/b - A/B| = |B δa - A δb| / |bB| <= (|B| ra + |A| rb) / (|B| (|B| - rb))`.
    pub(crate) fn div(&self, o: &Ball, w: u64) -> Option<Ball> {
        let lb = o.lower()?;
        let (mid, t) = self.mid.div_round(&o.mid, w, Round::Nearest);
        let (lo_b, hb) = mag_bounds(&o.mid);
        let (_, ha) = mag_bounds(&self.mid);
        let num = hb.mul(self.rad).add(ha.mul(o.rad));
        let rad = if num.is_zero() {
            Rad::ZERO
        } else {
            num.div(lo_b.mul_down(lb))
        };
        let rad = rad.add(half_ulp(&mid, t, w));
        Some(Ball::new(mid, rad))
    }

    /// The square root, the midpoint to `w` bits rounded toward zero, `None` when the ball holds
    /// a number below zero: `|sqrt(a) - sqrt(A)| = |δ| / (sqrt(a) + sqrt(A)) <= r / S` for the
    /// midpoint `S <= sqrt(A)`.
    pub(crate) fn sqrt(&self, w: u64) -> Option<Ball> {
        if self.mid.is_zero() && self.rad.is_zero() {
            return Some(self.clone());
        }
        if self.mid.is_sign_negative() {
            return None;
        }
        self.lower()?;
        let (mid, t) = self.mid.sqrt_round(w, Round::TowardZero);
        let (lo, _) = mag_bounds(&mid);
        let mut rad = if self.rad.is_zero() {
            Rad::ZERO
        } else {
            self.rad.div(lo)
        };
        if t != Ordering::Equal {
            rad = rad.add(Rad::pow2(
                i128::from(mid.exponent().expect("finite")) - i128::from(w),
            ));
        }
        Some(Ball::new(mid, rad))
    }

    /// The approximation for Ziv's loop: the midpoint and the radius, as a power of two at most
    /// a few limbs below the midpoint's last bit (a larger bound is as sound). A ball whose
    /// midpoint is not finite and not zero gives an empty magnitude, which never decides.
    pub(crate) fn approx(&self) -> Approx {
        match self.mid.parts() {
            Some((negative, m, lsb)) => {
                let err = self.rad.exp2_ceil().max(lsb - 128);
                if err >= lsb + 64 * m.len() as i128 {
                    // An error as large as the value never decides.
                    return undecided();
                }
                Approx {
                    negative,
                    m: m.to_vec(),
                    lsb,
                    err,
                }
            }
            None => Approx {
                negative: false,
                m: Vec::new(),
                lsb: 0,
                err: 0,
            },
        }
    }
}

/// The correct rounding of a value `v` that lies strictly between the float `x` (finite, not zero)
/// and its neighbour on the side `side` (the direction of `v - x` as a signed number), within
/// `2^delta` of `x`, when that is close enough to decide it: `x` widened to `prec + 2` bits or its
/// own if more, `v` within one unit of its last bit; `None` otherwise.
///
/// This is MPFR's rounding of a result near its argument (`mpfr_round_near_x`): `e^x` near 1,
/// `sinh x` near `x`.
pub(crate) fn round_near(
    x: &BigFloat,
    side: Ordering,
    delta: i128,
    prec: u64,
    round: Round,
) -> Option<(BigFloat, Ordering)> {
    let (negative, m, lsb) = x.parts()?;
    let mut v = m.to_vec();
    limbs::trim(&mut v);
    let z = limbs::trailing_zeros(&v);
    let v = limbs::shr(&v, z);
    let lsb = lsb + i128::from(z);
    let bits = limbs::bit_len(&v);
    let want = (prec + 2).max(bits);
    let sh = want - bits;
    let lsb = lsb - i128::from(sh);
    if delta >= lsb {
        return None;
    }
    let v = limbs::shl(&v, sh);
    // Away from zero when the signed side agrees with the sign.
    let away = (side == Ordering::Greater) != negative;
    let m = if away { v } else { limbs::sub(&v, &[1]) };
    Some(round_mag(negative, &m, lsb, true, prec, round))
}

/// Ziv's loop, as [`ziv`](super::ziv::ziv), for an approximation that may find the result
/// itself (beyond the exponent range, say): `Err(result)` ends the loop with it.
pub(crate) fn ziv_or(
    prec: u64,
    round: Round,
    mut approx: impl FnMut(u64) -> Result<Approx, (BigFloat, Ordering)>,
) -> (BigFloat, Ordering) {
    let mut w = prec + super::ziv::guard(prec);
    loop {
        match approx(w) {
            Err(r) => return r,
            Ok(a) => {
                if let Some(r) = fast_round(&a, prec, round) {
                    return r;
                }
                if let Some(r) = super::ziv::round_approx(&a, prec, round) {
                    return r;
                }
            }
        }
        w += (w / 2).max(64);
    }
}

/// The rounding of an approximation `A = ±M·2^lsb` within `2^err` of its value, when no boundary
/// of the rounding lies within `2^err` of `A`: then every number of the interval, the value
/// included, rounds as `A`, in the same direction (`A` is not a boundary). The boundaries are the
/// multiples of `2^q`, `q = e - prec - 2` for `2^(e-1) <= A < 2^e`: the numbers of the precision
/// and the midpoints between them in the binade below `A`'s too, so that a power of two is one.
/// With `k = q - lsb` and `j = err - lsb`, `A mod 2^q` is the low `k` bits of `M`, and they lie
/// further than `2^j` from 0 and from `2^k` when their bits from `max(j + 1, 0)` to `k - 1` are
/// neither all zeros nor all ones. `None` when that does not hold.
pub(crate) fn fast_round(a: &Approx, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    if a.m.is_empty() {
        return None;
    }
    let bits = i128::from(limbs::bit_len(&a.m));
    let e = a.lsb + bits;
    let q = e - i128::from(prec) - 2;
    let k = q - a.lsb;
    let lo = (a.err - a.lsb + 1).max(0);
    if lo >= k || k > bits {
        return None;
    }
    // Bits `lo..k` of `M`, `k <= bits`: not all zeros and not all ones.
    let (lo, k) = (lo as u64, k as u64);
    let mut zeros = true;
    let mut ones = true;
    let mut i = lo;
    while i < k && (zeros || ones) {
        let limb = (i / 64) as usize;
        let off = i % 64;
        let take = (64 - off).min(k - i);
        let mask = if take == 64 {
            u64::MAX
        } else {
            (1u64 << take) - 1
        };
        let v = (a.m[limb] >> off) & mask;
        zeros &= v == 0;
        ones &= v == mask;
        i += take;
    }
    if zeros || ones {
        return None;
    }
    Some(round_mag(a.negative, &a.m, a.lsb, false, prec, round))
}

/// A ball that never decides: the approximation of a step that failed.
#[inline]
pub(crate) fn undecided() -> Approx {
    Approx {
        negative: false,
        m: Vec::new(),
        lsb: 0,
        err: 0,
    }
}

/// Ziv's loop with the quick decision of [`fast_round`] first.
#[inline]
pub(crate) fn ziv_fast(
    prec: u64,
    round: Round,
    mut approx: impl FnMut(u64) -> Approx,
) -> (BigFloat, Ordering) {
    ziv_or(prec, round, |w| Ok(approx(w)))
}
