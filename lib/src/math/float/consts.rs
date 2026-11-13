//! The constants, correctly rounded to any precision: π from the crate's table of its bits and
//! Chudnovsky's series beyond (the series of the partition numbers), and `ln 2` from its series
//! `Σ 1/(k 2^k)`.

use super::limbs;
use super::ziv::{Approx, ziv};
use super::{BigFloat, Round};
use crate::math::numbers::pi;
use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

/// The exponent of a bound `2^e` of `n`: `e = bits(n)`.
#[inline]
fn bound(n: u64) -> i128 {
    i128::from(64 - n.leading_zeros())
}

/// π to `w` bits after the point: an [`Approx`] with an error below `2^(3 - w)`, from `floor(π 2^w)`
/// less at most 4, for the functions that need π inside their working precision.
pub(crate) fn pi_approx(w: u64) -> Approx {
    // π lies in `[P, P + 5)·2^-w`.
    let p = pi(w);
    Approx {
        negative: false,
        m: limbs::of_number(&p),
        lsb: -i128::from(w),
        err: 3 - i128::from(w),
    }
}

/// `ln 2` to `w` bits after the point: an [`Approx`] with an error of at most `w + 1` units of
/// `2^-w`, for the functions that need it inside their working precision.
pub(crate) fn ln2_approx(w: u64) -> Approx {
    // `Σ_(k ≤ w) floor(2^(w - k) / k)`: each term within 1 below, the tail past `w` below 1, so
    // `ln 2·2^w` lies within `w + 1` above the sum.
    let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
    let mut x = &one << w as usize;
    let mut sum = Number::new_from_u8(0, ByteOrder::LittleEndian);
    for k in 1..=w {
        x >>= 1;
        if x.is_zero() {
            break;
        }
        sum = &sum + &(&x / &Number::new_from_u64(k, ByteOrder::LittleEndian));
    }
    Approx {
        negative: false,
        m: limbs::of_number(&sum),
        lsb: -i128::from(w),
        err: bound(w + 1) - i128::from(w),
    }
}

impl BigFloat {
    /// π rounded to `prec` bits in the mode `round`, and the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (p, o) = BigFloat::pi_round(53, Round::Nearest);
    /// assert_eq!(p, BigFloat::from(core::f64::consts::PI));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn pi_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        ziv(prec, round, pi_approx)
    }

    /// π to nearest at `prec` bits.
    #[track_caller]
    pub fn pi(prec: u64) -> BigFloat {
        Self::pi_round(prec, Round::Nearest).0
    }

    /// `ln 2` rounded to `prec` bits in the mode `round`, and the direction of the rounding.
    #[track_caller]
    pub fn ln2_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        ziv(prec, round, ln2_approx)
    }

    /// `ln 2` to nearest at `prec` bits.
    #[track_caller]
    pub fn ln2(prec: u64) -> BigFloat {
        Self::ln2_round(prec, Round::Nearest).0
    }
}
