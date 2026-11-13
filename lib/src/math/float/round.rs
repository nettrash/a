//! The one rounding that every operation ends in: of an exact magnitude `M·2^lsb`, or of one
//! known only to lie strictly between `M·2^lsb` and `(M + 1)·2^lsb` (the sticky bit), to a
//! precision in a mode, with the exponent range applied.

use super::limbs;
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// Whether a number of the sign `negative` whose magnitude is to be rounded is rounded away from
/// zero in the mode `round` when it is not exact (the nearest aside).
#[inline]
pub(crate) fn away(negative: bool, round: Round) -> bool {
    match round {
        Round::Nearest | Round::TowardZero => false,
        Round::Up => !negative,
        Round::Down => negative,
    }
}

/// The direction of a rounding from the direction `magnitude` of the magnitude's: reversed for a
/// negative number.
#[inline]
pub(crate) fn signed(negative: bool, magnitude: Ordering) -> Ordering {
    if negative {
        magnitude.reverse()
    } else {
        magnitude
    }
}

/// The number `±M·2^lsb` for the magnitude `m` (not zero, without zero limbs at the top), or with
/// `sticky` a number strictly between `±M·2^lsb` and `±(M + 1)·2^lsb`, rounded to `prec` bits in
/// the mode `round`, and the direction of the rounding. With `sticky`, `M` must have more than
/// `prec` bits, so that the bits it drops hold the rounding bit.
///
/// An exponent past [`BigFloat::EXP_MAX`] after the rounding gives an infinity or the largest
/// finite number, and one below [`BigFloat::EXP_MIN`] zero or the smallest positive number, by
/// the mode.
pub(crate) fn round_mag(
    negative: bool,
    m: &[u64],
    lsb: i128,
    sticky: bool,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    debug_assert!(!m.is_empty() && m.last() != Some(&0));
    let bits = limbs::bit_len(m);
    debug_assert!(
        !sticky || bits > prec,
        "a sticky bit below a magnitude too short"
    );
    // The exponent of the exact value, `2^(e - 1) ≤ |x| < 2^e`.
    let e0 = lsb + i128::from(bits);
    let n = prec.div_ceil(64);
    let (mant, exp, magnitude) = if bits <= prec {
        (limbs::shl(m, 64 * n - bits), e0, Ordering::Equal)
    } else {
        let drop = bits - prec;
        let half = limbs::bit(m, drop - 1);
        let rest = sticky || !limbs::zero_below(m, drop - 1);
        let mut t = limbs::shr(m, drop);
        let up = match round {
            Round::Nearest => half && (rest || t[0] & 1 == 1),
            _ => away(negative, round) && (half || rest),
        };
        let mut exp = e0;
        if up {
            limbs::increment(&mut t);
            if limbs::bit_len(&t) > prec {
                t = limbs::shr(&t, 1);
                exp += 1;
            }
        }
        let magnitude = if !(half || rest) {
            Ordering::Equal
        } else if up {
            Ordering::Greater
        } else {
            Ordering::Less
        };
        (limbs::shl(&t, 64 * n - prec), exp, magnitude)
    };
    if exp > i128::from(BigFloat::EXP_MAX) {
        return overflow(negative, prec, round);
    }
    if exp < i128::from(BigFloat::EXP_MIN) {
        // The exact value is `2^(e0 - 1)` when its magnitude is a power of two, exactly.
        let power = !sticky && limbs::trailing_zeros(m) == bits - 1;
        return underflow(negative, e0, power, prec, round);
    }
    debug_assert_eq!(mant.len() as u64, n);
    (
        BigFloat {
            prec,
            kind: Kind::Finite {
                negative,
                exp: exp as i64,
                mant,
            },
        },
        signed(negative, magnitude),
    )
}

/// The result of a number of the sign `negative` too large for the exponent range: an infinity,
/// or the largest finite number of the precision when the mode rounds toward zero.
pub(crate) fn overflow(negative: bool, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let to_infinity = match round {
        Round::Nearest => true,
        _ => away(negative, round),
    };
    if to_infinity {
        (
            BigFloat {
                prec,
                kind: Kind::Inf { negative },
            },
            signed(negative, Ordering::Greater),
        )
    } else {
        (largest(negative, prec), signed(negative, Ordering::Less))
    }
}

/// `±(1 - 2^-prec)·2^EXP_MAX`, the largest finite magnitude of the precision.
pub(crate) fn largest(negative: bool, prec: u64) -> BigFloat {
    let n = prec.div_ceil(64) as usize;
    let mut mant = vec![u64::MAX; n];
    mant[0] <<= 64 * n as u64 - prec;
    BigFloat {
        prec,
        kind: Kind::Finite {
            negative,
            exp: BigFloat::EXP_MAX,
            mant,
        },
    }
}

/// `±2^(EXP_MIN - 1)`, the smallest positive magnitude.
pub(crate) fn smallest(negative: bool, prec: u64) -> BigFloat {
    let n = prec.div_ceil(64) as usize;
    let mut mant = vec![0u64; n];
    mant[n - 1] = 1 << 63;
    BigFloat {
        prec,
        kind: Kind::Finite {
            negative,
            exp: BigFloat::EXP_MIN,
            mant,
        },
    }
}

/// The result of a number of the sign `negative` too small for the exponent range, whose exact
/// value has the exponent `e0` (`2^(e0 - 1) ≤ |x| < 2^e0`) and is `2^(e0 - 1)` when `power`:
/// the smallest positive magnitude or a zero of the sign. To nearest, the zero for magnitudes up
/// to half the smallest, the half itself included (its even neighbour).
pub(crate) fn underflow(
    negative: bool,
    e0: i128,
    power: bool,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let min = i128::from(BigFloat::EXP_MIN);
    let to_smallest = match round {
        // An underflow has `e0 < EXP_MIN`: past the half for `e0 = EXP_MIN - 1` unless it is the
        // half itself, `2^(EXP_MIN - 2)`.
        Round::Nearest => e0 == min - 1 && !power,
        _ => away(negative, round),
    };
    if to_smallest {
        (
            smallest(negative, prec),
            signed(negative, Ordering::Greater),
        )
    } else {
        (
            BigFloat {
                prec,
                kind: Kind::Zero { negative },
            },
            signed(negative, Ordering::Less),
        )
    }
}
