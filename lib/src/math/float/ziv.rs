//! Ziv's strategy for the correct rounding of a value that can only be approximated: compute it to
//! a working precision with a bound of the error, round both ends of the interval the bound gives,
//! and accept the result when they agree and it lies outside the interval, so that its direction
//! is known too; otherwise compute again at a larger working precision.
//!
//! A function that never has an exact result of the precision (a transcendental one, away from
//! the arguments where it is exact, which it must treat first) always ends; one whose result may
//! be exact must find that case itself, as the interval of an exact result always holds it.

use super::limbs;
use super::round::round_mag;
use super::{BigFloat, Round};
use core::cmp::Ordering;

/// An approximation `±m·2^lsb` of a value `v` that is not zero, with `|v - (±m·2^lsb)| ≤ 2^err`.
#[derive(Clone, Debug)]
pub(crate) struct Approx {
    /// The sign of `v`.
    pub(crate) negative: bool,
    /// The magnitude, without zero limbs at the top.
    pub(crate) m: Vec<u64>,
    /// The exponent of the magnitude's lowest bit.
    pub(crate) lsb: i128,
    /// The exponent of the bound of the error.
    pub(crate) err: i128,
}

/// The magnitude order of a signed direction of a rounding of a number of the sign `negative`.
#[inline]
fn magnitude(negative: bool, o: Ordering) -> Ordering {
    if negative { o.reverse() } else { o }
}

/// The correct rounding of the value that `a` approximates, to `prec` bits in the mode `round`,
/// and its direction, when the approximation decides them: when both ends of
/// `[|m·2^lsb| - 2^err, |m·2^lsb| + 2^err]`, which does not hold zero, round to the same number
/// and that number lies outside the interval. `None` otherwise.
pub(crate) fn round_approx(a: &Approx, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    if a.m.is_empty() {
        return None;
    }
    let s = a.lsb.min(a.err);
    let m = limbs::shl(&a.m, (a.lsb - s) as u64);
    let e = limbs::shl(&[1], (a.err - s) as u64);
    if limbs::cmp(&m, &e) != Ordering::Greater {
        return None;
    }
    let lo = limbs::sub(&m, &e);
    let hi = limbs::add(&m, &e);
    let (rl, ol) = round_mag(a.negative, &lo, s, false, prec, round);
    let (rh, oh) = round_mag(a.negative, &hi, s, false, prec, round);
    if rl.kind != rh.kind {
        return None;
    }
    if magnitude(a.negative, ol) == Ordering::Less {
        // Below the lower end, so below the value.
        Some((rl, ol))
    } else if magnitude(a.negative, oh) == Ordering::Greater {
        Some((rh, oh))
    } else {
        None
    }
}

/// The guard bits of the first working precision for a result of `prec` bits: about
/// `2 log2(prec) + 16`, which makes a second attempt rare.
#[inline]
pub(crate) fn guard(prec: u64) -> u64 {
    2 * u64::from(64 - prec.leading_zeros()) + 16
}

/// The correct rounding to `prec` bits in the mode `round`, and its direction, of the value that
/// `approx(w)` approximates at the working precision `w`: tried at `prec` and [`guard`] bits
/// more, then at a working precision half as large again each time until it decides.
///
/// `approx(w)` should give an error about `2^-w` of the value's magnitude, so that the attempts
/// decide; any error bound is sound.
pub(crate) fn ziv(
    prec: u64,
    round: Round,
    mut approx: impl FnMut(u64) -> Approx,
) -> (BigFloat, Ordering) {
    let mut w = prec + guard(prec);
    loop {
        if let Some(r) = round_approx(&approx(w), prec, round) {
            return r;
        }
        w += (w / 2).max(64);
    }
}
