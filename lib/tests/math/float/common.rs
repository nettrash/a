//! What the tests of floats share: random floats of every precision, and the reference of the
//! correct rounding, an exact rational rounded with integers alone.

use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// Scales a randomised test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Small deterministic xorshift generator.
pub(super) fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A random number below `n`.
pub(super) fn below(state: &mut u64, n: u64) -> u64 {
    xorshift(state) % n
}

/// The four modes.
pub(super) const MODES: [Round; 4] = [Round::Nearest, Round::TowardZero, Round::Up, Round::Down];

/// A random mode.
pub(super) fn mode(state: &mut u64) -> Round {
    MODES[below(state, 4) as usize]
}

/// The number 1 in little-endian bytes.
pub(super) fn one() -> Number {
    Number::new_from_u8(1, ByteOrder::LittleEndian)
}

/// `2^k`.
pub(super) fn pow2(k: u64) -> Number {
    one() << k as usize
}

/// A random integer of exactly `bits` bits, its bits in long runs of zeros and ones.
pub(super) fn mantissa(state: &mut u64, bits: u64) -> Number {
    let mut m = Number::new_from_u8(0, ByteOrder::LittleEndian);
    let mut i = 0;
    while i < bits {
        let run = (1 + below(state, 24)).min(bits - i);
        let kind = below(state, 3);
        for j in 0..run {
            let set = match kind {
                0 => false,
                1 => true,
                _ => xorshift(state) & 1 == 1,
            };
            if set {
                m.set_bit(i + j, true);
            }
        }
        i += run;
    }
    m.set_bit(bits - 1, true);
    m
}

/// A random finite float that is not zero: of a random precision up to `max_prec`, its mantissa
/// all of it, and an exponent `e` of `0.1…·2^e` in `-exps..=exps`.
pub(super) fn float(state: &mut u64, max_prec: u64, exps: i64) -> BigFloat {
    let prec = 1 + below(state, max_prec);
    let mut m = mantissa(state, prec);
    if xorshift(state) & 1 == 1 {
        m = -m;
    }
    let e = below(state, 2 * exps as u64 + 1) as i64 - exps;
    let (x, o) = BigFloat::from_parts(&m, e - prec as i64, prec, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

/// The rational a finite float is.
pub(super) fn rational(x: &BigFloat) -> Rational {
    Rational::try_from(x).expect("a finite float")
}

/// The rational `q`, not zero, rounded to `prec` bits in the mode `round`, and the direction, by
/// integers alone: `M = floor(|q|·2^t)` with `2^(p-1) ≤ M < 2^p`, the remainder deciding.
pub(super) fn reference(q: &Rational, prec: u64, round: Round) -> (BigFloat, Ordering) {
    assert!(!q.is_zero());
    let negative = q.is_negative();
    let n = q.numer().abs();
    let d = q.denom().clone();
    let (lo, hi) = (pow2(prec - 1), pow2(prec));
    let mut t = prec as i64 - (n.bits() as i64 - d.bits() as i64);
    // `floor(|q|·2^t)`, twice the remainder and the divisor it is over.
    let scaled = |t: i64| -> (Number, Number, Number) {
        let (a, b) = if t >= 0 {
            (&n << t as usize, d.clone())
        } else {
            (n.clone(), &d << (-t) as usize)
        };
        let (m, r) = a.div_rem(&b);
        (m, &r << 1usize, b)
    };
    let (mut m, r2, b) = loop {
        let (m, r2, b) = scaled(t);
        if m >= hi {
            t -= 1;
        } else if m < lo {
            t += 1;
        } else {
            break (m, r2, b);
        }
    };
    let inexact = !r2.is_zero();
    let up = match round {
        Round::Nearest => r2 > b || r2 == b && m.bit(0),
        Round::TowardZero => false,
        Round::Up => inexact && !negative,
        Round::Down => inexact && negative,
    };
    if up {
        m = &m + &one();
        if m == hi {
            m = lo;
            t -= 1;
        }
    }
    let signed = if negative { -m } else { m };
    let (x, o) = BigFloat::from_parts(&signed, -t, prec, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    let magnitude = if !inexact {
        Ordering::Equal
    } else if up {
        Ordering::Greater
    } else {
        Ordering::Less
    };
    (
        x,
        if negative {
            magnitude.reverse()
        } else {
            magnitude
        },
    )
}

/// Asserts that `got` is `want`: the same value, the same sign of zero, the same precision and
/// the same direction of rounding.
#[track_caller]
pub(super) fn same(got: &(BigFloat, Ordering), want: &(BigFloat, Ordering), what: &str) {
    let ok = got.0.prec() == want.0.prec()
        && got.1 == want.1
        && (got.0.is_nan() && want.0.is_nan()
            || got.0 == want.0 && got.0.is_sign_negative() == want.0.is_sign_negative());
    assert!(
        ok,
        "{what}: got {:?} {:?}, want {:?} {:?}",
        got.0, got.1, want.0, want.1
    );
}
