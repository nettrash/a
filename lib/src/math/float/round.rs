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
    let (kind, o) = round_into(negative, m, lsb, sticky, prec, round, Vec::new());
    (BigFloat { prec, kind }, o)
}

/// [`round_mag`] into the vector `out`, whose limbs it replaces with the mantissa (and which it
/// drops for a result that is not finite): the value and the direction of the rounding.
///
/// The top `64n` bits of `M`, `n = ceil(prec / 64)`, are copied in one pass, shifted so that the
/// top bit of `M` is the top bit of the top limb; the bits below the precision are cleared after
/// the round bit and whether anything lies below it are read from `M` (and `sticky`), and one unit
/// of the last place is added when the mode and those bits call for it.
pub(crate) fn round_into(
    negative: bool,
    m: &[u64],
    lsb: i128,
    sticky: bool,
    prec: u64,
    round: Round,
    mut out: Vec<u64>,
) -> (Kind, Ordering) {
    debug_assert!(!m.is_empty() && m.last() != Some(&0));
    let bits = limbs::bit_len(m);
    debug_assert!(
        !sticky || bits > prec,
        "a sticky bit below a magnitude too short"
    );
    let n = prec.div_ceil(64) as usize;
    let dropped = dropped(m, bits, prec, sticky);
    out.clear();
    out.reserve(n);
    // Limb `i` of the window is bits `start + 64i..` of `M`.
    let start = i128::from(bits) - 64 * n as i128;
    if start >= 0 {
        let (k, b) = ((start / 64) as usize, (start % 64) as u32);
        if b == 0 {
            out.extend_from_slice(&m[k..k + n]);
        } else {
            out.extend(
                m[k..]
                    .windows(2)
                    .take(n)
                    .map(|w| w[0] >> b | w[1] << (64 - b)),
            );
            if out.len() < n {
                out.push(m[m.len() - 1] >> b);
            }
        }
    } else {
        let s = (-start) as u64;
        let (k, b) = ((s / 64) as usize, (s % 64) as u32);
        out.resize(k, 0);
        if b == 0 {
            out.extend_from_slice(m);
        } else {
            let mut carry = 0u64;
            for &x in m {
                out.push(x << b | carry);
                carry = x >> (64 - b);
            }
            if out.len() < n {
                out.push(carry);
            }
        }
    }
    debug_assert_eq!(out.len(), n);
    finish(negative, out, lsb + i128::from(bits), dropped, prec, round)
}

/// [`round_into`] of the magnitude in the vector `v` itself, whose limbs become the mantissa: the
/// window is shifted down in place, without a copy. The magnitude must have at least `64n - 63`
/// bits, `n = ceil(prec / 64)`, as the sums' windows do.
pub(crate) fn round_vec(
    negative: bool,
    mut v: Vec<u64>,
    lsb: i128,
    sticky: bool,
    prec: u64,
    round: Round,
) -> (Kind, Ordering) {
    debug_assert!(!v.is_empty() && v.last() != Some(&0));
    let bits = limbs::bit_len(&v);
    debug_assert!(!sticky || bits > prec);
    let n = prec.div_ceil(64) as usize;
    let start = i128::from(bits) - 64 * n as i128;
    if start < 0 {
        let m = core::mem::take(&mut v);
        return round_into(negative, &m, lsb, sticky, prec, round, v);
    }
    let dropped = dropped(&v, bits, prec, sticky);
    let (k, b) = ((start / 64) as usize, (start % 64) as u32);
    if b == 0 {
        v.copy_within(k..k + n, 0);
    } else {
        let len = v.len();
        for i in 0..n {
            let j = k + i;
            let hi = if j + 1 < len { v[j + 1] << (64 - b) } else { 0 };
            v[i] = v[j] >> b | hi;
        }
    }
    v.truncate(n);
    finish(negative, v, lsb + i128::from(bits), dropped, prec, round)
}

/// The bits a rounding to `prec` drops from the magnitude `m` of `bits` bits (with `sticky` for
/// bits below it): `None` when it keeps them all, otherwise the round bit and whether anything
/// lies below it.
#[inline]
fn dropped(m: &[u64], bits: u64, prec: u64, sticky: bool) -> Option<(bool, bool)> {
    if bits <= prec {
        None
    } else {
        let drop = bits - prec;
        Some((
            limbs::bit(m, drop - 1),
            sticky || !limbs::zero_below(m, drop - 1),
        ))
    }
}

/// The rounding of the window `out` of the top `64n` bits of a magnitude of the exponent `e0`
/// (`2^(e0 - 1) ≤ |x| < 2^e0`), given the bits it drops: the bits below the precision cleared,
/// one unit added when the mode and those bits call for it, and the exponent range applied.
#[inline]
fn finish(
    negative: bool,
    mut out: Vec<u64>,
    e0: i128,
    dropped: Option<(bool, bool)>,
    prec: u64,
    round: Round,
) -> (Kind, Ordering) {
    let n = out.len();
    let mut exp = e0;
    let magnitude = match dropped {
        None => Ordering::Equal,
        Some((half, rest)) => {
            let sh = 64 * n as u64 - prec;
            let unit = 1u64 << sh;
            out[0] &= unit.wrapping_neg();
            let up = match round {
                Round::Nearest => half && (rest || out[0] & unit != 0),
                _ => away(negative, round) && (half || rest),
            };
            if up {
                let (x, mut carry) = out[0].overflowing_add(unit);
                out[0] = x;
                let mut i = 1;
                while carry && i < n {
                    (out[i], carry) = out[i].overflowing_add(1);
                    i += 1;
                }
                if carry {
                    out[n - 1] = 1 << 63;
                    exp += 1;
                }
            }
            if !(half || rest) {
                Ordering::Equal
            } else if up {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
    };
    if exp > i128::from(BigFloat::EXP_MAX) {
        return overflow(negative, prec, round).map_kind();
    }
    if exp < i128::from(BigFloat::EXP_MIN) {
        // The exact value is `2^(e0 - 1)` when it is exact and a power of two.
        let power = magnitude == Ordering::Equal
            && out[n - 1] == 1 << 63
            && out[..n - 1].iter().all(|&v| v == 0);
        return underflow(negative, e0, power, prec, round).map_kind();
    }
    (
        Kind::Finite {
            negative,
            exp: exp as i64,
            mant: out,
        },
        signed(negative, magnitude),
    )
}

/// A result of at most two limbs rounded in registers: `±0.m·2^exp` with the mantissa `m`
/// aligned at the top of 128 bits and the bits below the precision zero, the exponent `e0` of the
/// exact value (`exp` or one less, when the rounding carried), and the direction of the rounding
/// of the magnitude.
#[derive(Clone, Copy)]
pub(crate) struct Rounded {
    pub(crate) negative: bool,
    pub(crate) m: u128,
    pub(crate) exp: i128,
    pub(crate) e0: i128,
    pub(crate) magnitude: Ordering,
}

/// The number that `r` describes, at the precision `prec` (at most 128) with the exponent range
/// applied, its mantissa in `out`: the value and the direction of the rounding.
#[inline]
pub(crate) fn seal(r: Rounded, prec: u64, round: Round, mut out: Vec<u64>) -> (Kind, Ordering) {
    debug_assert!(prec <= 128 && r.m >> 127 == 1);
    if r.exp > i128::from(BigFloat::EXP_MAX) {
        return overflow(r.negative, prec, round).map_kind();
    }
    if r.exp < i128::from(BigFloat::EXP_MIN) {
        let power = r.magnitude == Ordering::Equal && r.m == 1 << 127;
        return underflow(r.negative, r.e0, power, prec, round).map_kind();
    }
    out.clear();
    if prec <= 64 {
        out.push((r.m >> 64) as u64);
    } else {
        out.push(r.m as u64);
        out.push((r.m >> 64) as u64);
    }
    (
        Kind::Finite {
            negative: r.negative,
            exp: r.exp as i64,
            mant: out,
        },
        signed(r.negative, r.magnitude),
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

/// The value and direction of a [`BigFloat`] result, without its precision.
trait MapKind {
    fn map_kind(self) -> (Kind, Ordering);
}

impl MapKind for (BigFloat, Ordering) {
    #[inline]
    fn map_kind(self) -> (Kind, Ordering) {
        (self.0.kind, self.1)
    }
}
