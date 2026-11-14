//! Comparisons of floats with the primitive integers and floats, [`Number`]s and [`Rational`]s,
//! by value and exactly: `x == 1`, `x < 0.5`, `x >= n`, and `n >= x` for a `Number` or a
//! `Rational` (not for the primitives, whose comparisons with a second type would leave the
//! types of literals in code such as `v == vec![]` ambiguous). NaN, of a float or of an `f64`,
//! is unordered and equal to nothing; `-0 == +0`.

use super::limbs;
use super::{BigFloat, Kind};
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use core::cmp::Ordering;

/// The order of the magnitudes `|x| = 0.mant·2^exp` and `M·2^lsb` (`M` not zero, without zero
/// limbs at the top): by the exponents, then by the mantissas aligned at their top bits.
fn cmp_mag(exp: i64, mant: &[u64], m: &[u64], lsb: i128) -> Ordering {
    let bits = limbs::bit_len(m);
    let ey = lsb + i128::from(bits);
    match i128::from(exp).cmp(&ey) {
        Ordering::Equal => {}
        o => return o,
    }
    // `M` at the top of its limbs, as the mantissa is.
    let shift = 64 * m.len() as u64 - bits;
    let top = |i: usize| -> u64 {
        // Limb `i` from the top of `M << shift`.
        let k = m.len() - 1 - i;
        if shift == 0 {
            m[k]
        } else {
            let below = if k > 0 { m[k - 1] >> (64 - shift) } else { 0 };
            m[k] << shift | below
        }
    };
    let (na, nb) = (mant.len(), m.len());
    for i in 0..na.max(nb) {
        let x = if i < na { mant[na - 1 - i] } else { 0 };
        let y = if i < nb { top(i) } else { 0 };
        if x != y {
            return x.cmp(&y);
        }
    }
    Ordering::Equal
}

/// The order of the float `x` to `±M·2^lsb` (`M` without zero limbs at the top, zero when
/// empty), `None` when `x` is NaN.
fn cmp_parts(x: &BigFloat, negative: bool, m: &[u64], lsb: i128) -> Option<Ordering> {
    let y_negative = negative && !m.is_empty();
    Some(match &x.kind {
        Kind::Nan => return None,
        Kind::Inf { negative } => {
            if *negative {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        Kind::Zero { .. } => {
            if m.is_empty() {
                Ordering::Equal
            } else if y_negative {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        Kind::Finite {
            negative: xn,
            exp,
            mant,
        } => {
            if m.is_empty() || *xn != y_negative {
                if *xn {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else {
                let o = cmp_mag(*exp, mant, m, lsb);
                if *xn { o.reverse() } else { o }
            }
        }
    })
}

/// The order of `x` to the integer `±m`.
#[inline]
fn cmp_u128(x: &BigFloat, negative: bool, m: u128) -> Option<Ordering> {
    let v = [m as u64, (m >> 64) as u64];
    let n = if m >> 64 != 0 {
        2
    } else if m != 0 {
        1
    } else {
        0
    };
    cmp_parts(x, negative, &v[..n], 0)
}

/// The order of `x` to the `f64` `y`.
fn cmp_f64(x: &BigFloat, y: f64) -> Option<Ordering> {
    if y.is_nan() {
        return None;
    }
    if y.is_infinite() {
        if x.is_nan() {
            return None;
        }
        let o = match x.kind {
            Kind::Inf { negative } if negative == (y < 0.0) => Ordering::Equal,
            _ => {
                if y > 0.0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
        };
        return Some(o);
    }
    let bits = y.to_bits();
    let field = (bits >> 52 & 0x7ff) as i128;
    let frac = bits & ((1 << 52) - 1);
    let (m, e) = if field == 0 {
        (frac, -1074)
    } else {
        (frac | 1 << 52, field - 1075)
    };
    let v = [m];
    cmp_parts(x, y < 0.0, if m == 0 { &v[..0] } else { &v }, e)
}

/// The order of `x` to the rational `r`: by the bits of `|x|·d` and `|n|`, then exactly.
fn cmp_rational(x: &BigFloat, r: &Rational) -> Option<Ordering> {
    if r.denom().bits() == 1 {
        return x.partial_cmp(r.numer());
    }
    let (xn, m, lsb) = match x.parts() {
        Some(p) => p,
        None => {
            // NaN, an infinity, a zero: as against any number of the rational's sign.
            return cmp_parts(x, r.is_negative(), &[1], 0);
        }
    };
    let rn = r.is_negative();
    if xn != rn {
        return Some(if xn {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }
    // `|x|·d = M·d·2^lsb` against `|n|`, both not zero.
    let n = limbs::of_number(r.numer());
    let d = limbs::of_number(r.denom());
    let (bm, bd, bn) = (
        i128::from(limbs::bit_len(m)),
        i128::from(limbs::bit_len(&d)),
        i128::from(limbs::bit_len(&n)),
    );
    let o = if bm + bd - 2 + lsb >= bn {
        // `|x|·d ≥ 2^(bm + bd - 2 + lsb) ≥ 2^bn > |n|`.
        Ordering::Greater
    } else if bm + bd + lsb < bn {
        // `|x|·d < 2^(bm + bd + lsb) ≤ 2^(bn - 1) ≤ |n|`.
        Ordering::Less
    } else {
        let p = limbs::mul(m, &d);
        if lsb >= 0 {
            limbs::cmp(&limbs::shl(&p, lsb as u64), &n)
        } else {
            limbs::cmp(&p, &limbs::shl(&n, (-lsb) as u64))
        }
    };
    Some(if xn { o.reverse() } else { o })
}

impl PartialEq<Number> for BigFloat {
    /// Whether the float is the integer `n`.
    #[inline]
    fn eq(&self, n: &Number) -> bool {
        self.partial_cmp(n) == Some(Ordering::Equal)
    }
}

impl PartialOrd<Number> for BigFloat {
    /// Compares the float with the integer `n` by value, `None` for NaN.
    fn partial_cmp(&self, n: &Number) -> Option<Ordering> {
        cmp_parts(self, n.is_negative(), &limbs::of_number(n), 0)
    }
}

impl PartialEq<BigFloat> for Number {
    /// Whether the integer is the float `x`.
    #[inline]
    fn eq(&self, x: &BigFloat) -> bool {
        x == self
    }
}

impl PartialOrd<BigFloat> for Number {
    /// Compares the integer with the float `x` by value, `None` for NaN.
    #[inline]
    fn partial_cmp(&self, x: &BigFloat) -> Option<Ordering> {
        x.partial_cmp(self).map(Ordering::reverse)
    }
}

impl PartialEq<Rational> for BigFloat {
    /// Whether the float is the rational `r`.
    #[inline]
    fn eq(&self, r: &Rational) -> bool {
        self.partial_cmp(r) == Some(Ordering::Equal)
    }
}

impl PartialOrd<Rational> for BigFloat {
    /// Compares the float with the rational `r` by value, exactly, `None` for NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use a::math::rational::Rational;
    ///
    /// let third = Rational::from(1) / 3u32;
    /// assert!(BigFloat::from(1.0 / 3.0) < third);
    /// assert!(BigFloat::from(0.5) == Rational::from(1) / 2u32);
    /// ```
    #[inline]
    fn partial_cmp(&self, r: &Rational) -> Option<Ordering> {
        cmp_rational(self, r)
    }
}

impl PartialEq<BigFloat> for Rational {
    /// Whether the rational is the float `x`.
    #[inline]
    fn eq(&self, x: &BigFloat) -> bool {
        x == self
    }
}

impl PartialOrd<BigFloat> for Rational {
    /// Compares the rational with the float `x` by value, `None` for NaN.
    #[inline]
    fn partial_cmp(&self, x: &BigFloat) -> Option<Ordering> {
        x.partial_cmp(self).map(Ordering::reverse)
    }
}

macro_rules! float_cmp {
    ($($t:ty),*) => {$(
        impl PartialEq<$t> for BigFloat {
            #[doc = concat!("Whether the float is the `", stringify!($t), "` `y`, exactly: never for NaN.")]
            #[inline]
            fn eq(&self, y: &$t) -> bool {
                cmp_f64(self, f64::from(*y)) == Some(Ordering::Equal)
            }
        }

        impl PartialOrd<$t> for BigFloat {
            #[doc = concat!("Compares the float with the `", stringify!($t), "` `y` by value, exactly, `None` when either is NaN.")]
            #[inline]
            fn partial_cmp(&self, y: &$t) -> Option<Ordering> {
                cmp_f64(self, f64::from(*y))
            }
        }

    )*};
}

float_cmp!(f32, f64);

macro_rules! int_cmp {
    ($($t:ty),*) => {$(
        impl PartialEq<$t> for BigFloat {
            #[doc = concat!("Whether the float is the `", stringify!($t), "` `v`.")]
            #[inline]
            fn eq(&self, v: &$t) -> bool {
                self.partial_cmp(v) == Some(Ordering::Equal)
            }
        }

        impl PartialOrd<$t> for BigFloat {
            #[doc = concat!("Compares the float with the `", stringify!($t), "` `v` by value, `None` for NaN.")]
            #[allow(unused_comparisons)]
            #[inline]
            fn partial_cmp(&self, v: &$t) -> Option<Ordering> {
                cmp_u128(self, *v < 0, (*v as i128).unsigned_abs())
            }
        }

    )*};
}

int_cmp!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, usize);

impl PartialEq<u128> for BigFloat {
    /// Whether the float is the `u128` `v`.
    #[inline]
    fn eq(&self, v: &u128) -> bool {
        self.partial_cmp(v) == Some(Ordering::Equal)
    }
}

impl PartialOrd<u128> for BigFloat {
    /// Compares the float with the `u128` `v` by value, `None` for NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// let x = BigFloat::from(2.5);
    /// assert!(x > 2 && x < 3u128 && x == 2.5 && x != 2.4999);
    /// assert!(BigFloat::nan(10) != 0);
    /// ```
    #[inline]
    fn partial_cmp(&self, v: &u128) -> Option<Ordering> {
        cmp_u128(self, false, *v)
    }
}

macro_rules! by_reference {
    ($($t:ty),*) => {$(
        impl PartialEq<&$t> for BigFloat {
            #[doc = concat!("Whether the float is the borrowed `", stringify!($t), "`, as for one by value.")]
            #[inline]
            fn eq(&self, other: &&$t) -> bool {
                self == *other
            }
        }

        impl PartialOrd<&$t> for BigFloat {
            #[doc = concat!("Compares the float with a borrowed `", stringify!($t), "`, as with one by value: `x < &n`.")]
            #[inline]
            fn partial_cmp(&self, other: &&$t) -> Option<Ordering> {
                self.partial_cmp(*other)
            }
        }
    )*};
}

by_reference!(Number, Rational);
