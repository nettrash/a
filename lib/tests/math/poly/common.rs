//! What the tests of polynomials share: random coefficients of every size, sign and byte order,
//! and an oracle of plain vectors of numbers to check the polynomials against.

use a::math::numbers::{ByteOrder, Number};
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

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

/// A random integer of up to `max` bytes, often with long runs of zero and `0xFF` bytes, of either
/// sign and byte order.
pub(super) fn number(state: &mut u64, max: usize) -> Number {
    let len = below(state, max as u64 + 1) as usize;
    let mut bytes = Vec::with_capacity(len);
    while bytes.len() < len {
        let run = (1 + below(state, 12) as usize).min(len - bytes.len());
        match below(state, 4) {
            0 => bytes.resize(bytes.len() + run, 0),
            1 => bytes.resize(bytes.len() + run, 0xFF),
            _ => bytes.extend((0..run).map(|_| xorshift(state) as u8)),
        }
    }
    let negative = below(state, 2) == 1;
    if below(state, 2) == 1 {
        bytes.reverse();
        Number::new_signed(bytes, negative, ByteOrder::BigEndian)
    } else {
        Number::new_signed(bytes, negative, ByteOrder::LittleEndian)
    }
}

/// A random vector of `len` integers of up to `max` bytes, its top one not zero when `len > 0`.
pub(super) fn coeffs(state: &mut u64, len: usize, max: usize) -> Vec<Number> {
    let mut c: Vec<Number> = (0..len).map(|_| number(state, max)).collect();
    if let Some(top) = c.last_mut() {
        while top.is_zero() {
            *top = number(state, max.max(1));
        }
    }
    c
}

/// A random polynomial over the integers of up to `len` coefficients of up to `max` bytes.
pub(super) fn poly(state: &mut u64, len: usize, max: usize) -> Polynomial<Number> {
    let n = below(state, len as u64 + 1) as usize;
    Polynomial::new(coeffs(state, n, max))
}

/// A random rational of a numerator and a denominator of up to `max` bytes.
pub(super) fn rational(state: &mut u64, max: usize) -> Rational {
    let num = number(state, max);
    let mut den = number(state, max);
    while den.is_zero() {
        den = number(state, max.max(1));
    }
    Rational::new(num, den)
}

/// A random polynomial over the rationals of up to `len` coefficients.
pub(super) fn rpoly(state: &mut u64, len: usize, max: usize) -> Polynomial<Rational> {
    let n = below(state, len as u64 + 1) as usize;
    Polynomial::new((0..n).map(|_| rational(state, max)).collect())
}

/// The integers `c`.
pub(super) fn ints(c: &[i64]) -> Polynomial<Number> {
    Polynomial::new(c.iter().map(|&x| Number::from(x)).collect())
}

/// The rationals `c`.
pub(super) fn rats(c: &[i64]) -> Polynomial<Rational> {
    Polynomial::new(c.iter().map(|&x| Rational::from(x)).collect())
}

/// The oracle's product: the schoolbook product of the vectors `a` and `b`, without zeros at the
/// top.
pub(super) fn oracle_mul(a: &[Number], b: &[Number]) -> Vec<Number> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![Number::from(0); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            c[i + j] = &c[i + j] + &(x * y);
        }
    }
    trim(c)
}

/// `c` without zeros at the top.
pub(super) fn trim(mut c: Vec<Number>) -> Vec<Number> {
    while c.last().is_some_and(Number::is_zero) {
        c.pop();
    }
    c
}

/// The oracle's residues of `c` modulo `n`, in `0..n`, without zeros at the top.
pub(super) fn oracle_mod(c: &[Number], n: &Number) -> Vec<Number> {
    trim(c.iter().map(|x| x.rem_euclid(n)).collect())
}

/// The value of `c` at `x`, as the sum of each `cᵢ·xⁱ`.
pub(super) fn oracle_eval(c: &[Number], x: &Number) -> Number {
    let mut v = Number::from(0);
    let mut p = Number::from(1);
    for c in c {
        v = &v + &(c * &p);
        p = &p * x;
    }
    v
}

/// The moduli the tests take: small primes, primes at the top of a word, of two and three limbs,
/// and composite ones of one and of several limbs.
pub(super) fn moduli() -> Vec<Number> {
    let p = |s: &str| Number::from_str_radix(s, 10, ByteOrder::LittleEndian).unwrap();
    vec![
        Number::from(2),
        Number::from(3),
        Number::from(7),
        Number::from(65537),
        Number::from(1_000_000_007u64),
        // 2^64 - 59, the largest prime below 2^64.
        Number::from(18_446_744_073_709_551_557u64),
        // 2^64 + 13, a prime of two limbs.
        p("18446744073709551629"),
        // 2^127 - 1, a Mersenne prime.
        p("170141183460469231731687303715884105727"),
        // A prime of three limbs: 2^130 - 5.
        p("1361129467683753853853498429727072845819"),
        // Composite: 2^64 - 1, 2^128, 10^30.
        Number::from(u64::MAX),
        p("340282366920938463463374607431768211456"),
        p("1000000000000000000000000000000"),
    ]
}

/// The moduli of [`moduli`] that are prime.
pub(super) fn primes() -> Vec<Number> {
    moduli().into_iter().take(9).collect()
}

/// A random polynomial of up to `len` coefficients modulo `n`.
pub(super) fn mpoly(state: &mut u64, len: usize, n: &Number) -> ModPolynomial {
    let k = below(state, len as u64 + 1) as usize;
    let bytes = n.to_str_radix(16).len() / 2 + 2;
    ModPolynomial::new(&coeffs(state, k, bytes), n)
}
