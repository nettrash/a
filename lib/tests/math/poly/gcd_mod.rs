//! Greatest common divisors, Bézout's cofactors, resultants and discriminants of `ModPolynomial`s,
//! of one word and of several limbs, below and above the crossovers to the half gcd: against
//! Euclid's algorithm on the core's division, Bézout's identity, and the Sylvester determinant.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;

/// The oracle's monic gcd: Euclid's algorithm on the core's schoolbook division.
fn euclid(a: &ModPolynomial, b: &ModPolynomial) -> ModPolynomial {
    let (mut a, mut b) = (a.clone(), b.clone());
    while !b.is_zero() {
        let r = a.div_rem(&b).1;
        a = b;
        b = r;
    }
    a.to_monic()
}

/// The oracle's resultant: the Sylvester determinant by Gaussian elimination modulo the prime `p`.
fn sylvester(a: &ModPolynomial, b: &ModPolynomial) -> Number {
    let (Some(m), Some(n)) = (a.degree(), b.degree()) else {
        return Number::from(0);
    };
    sylvester_formal(a, m, b, n)
}

/// The Sylvester determinant of `a` and `b` taken of the formal degrees `m` and `n`, at least
/// their degrees.
fn sylvester_formal(a: &ModPolynomial, m: usize, b: &ModPolynomial, n: usize) -> Number {
    let p = a.modulus().clone();
    if m == 0 && n == 0 {
        return Number::from(1);
    }
    let size = m + n;
    let mut ac = a.coeffs();
    ac.resize(m + 1, Number::from(0));
    let mut bc = b.coeffs();
    bc.resize(n + 1, Number::from(0));
    let mut mat = vec![vec![Number::from(0); size]; size];
    for i in 0..n {
        for (j, c) in ac.iter().rev().enumerate() {
            mat[i][i + j] = c.clone();
        }
    }
    for i in 0..m {
        for (j, c) in bc.iter().rev().enumerate() {
            mat[n + i][i + j] = c.clone();
        }
    }
    let mut det = Number::from(1);
    for col in 0..size {
        let Some(piv) = (col..size).find(|&r| !mat[r][col].is_zero()) else {
            return Number::from(0);
        };
        if piv != col {
            mat.swap(piv, col);
            det = (&p - &det).rem_euclid(&p);
        }
        det = (&det * &mat[col][col]).rem_euclid(&p);
        let inv = mat[col][col].mod_inverse(&p).unwrap();
        for r in col + 1..size {
            if mat[r][col].is_zero() {
                continue;
            }
            let f = (&mat[r][col] * &inv).rem_euclid(&p);
            let pivot = mat[col].clone();
            for (x, y) in mat[r][col..].iter_mut().zip(&pivot[col..]) {
                *x = (&*x - &(&f * y)).rem_euclid(&p);
            }
        }
    }
    det
}

/// A random polynomial of exactly `len` coefficients modulo `n` (`len = 0`: zero).
fn exact(state: &mut u64, len: usize, n: &Number) -> ModPolynomial {
    let bytes = n.to_str_radix(16).len() / 2 + 2;
    let mut c = coeffs(state, len, bytes);
    if len > 0 {
        // A unit at the top for a prime n; retry the top for the others.
        loop {
            let t = c[len - 1].rem_euclid(n);
            if !t.is_zero() && t.gcd(n) == 1 {
                break;
            }
            c[len - 1] = number(state, bytes);
        }
    }
    ModPolynomial::new(&c, n)
}

/// Checks gcd, xgcd and lcm of `a` and `b` against the oracle.
fn check(a: &ModPolynomial, b: &ModPolynomial) {
    let g = a.gcd(b);
    assert_eq!(g, euclid(a, b), "gcd of {a} and {b}");
    let (g2, s, t) = a.xgcd(b);
    assert_eq!(g2, g);
    assert_eq!(&(&s * a) + &(&t * b), g, "Bézout for {a} and {b}");
    if let Some(dg) = g.degree() {
        if let Some(db) = b.degree()
            && db > dg
        {
            assert!(s.degree().is_none_or(|d| d < db - dg), "deg s");
        }
        if let Some(da) = a.degree()
            && da > dg
        {
            assert!(t.degree().is_none_or(|d| d < da - dg), "deg t");
        }
    }
}

#[test]
fn small_cases() {
    let n = Number::from(7);
    let z = ModPolynomial::zero(&n);
    let one = ModPolynomial::one(&n);
    let a = ModPolynomial::from_u64(&[2, 3, 1], 7);
    assert_eq!(z.gcd(&z), z);
    assert_eq!(z.xgcd(&z), (z.clone(), z.clone(), z.clone()));
    assert_eq!(a.gcd(&z), a);
    assert_eq!(z.gcd(&a), a);
    let c = ModPolynomial::from_u64(&[3], 7);
    assert_eq!(a.gcd(&c), one);
    let (g, s, t) = c.xgcd(&a);
    assert_eq!(g, one);
    assert_eq!(&(&s * &c) + &(&t * &a), one);
    assert_eq!(z.lcm(&a), z);
    assert_eq!(a.lcm(&c), a);
    // Resultants of zeros and constants.
    assert_eq!(z.resultant(&a), Number::from(0));
    assert_eq!(c.resultant(&a), Number::from(9 % 7));
    assert_eq!(a.resultant(&c), Number::from(2));
    assert_eq!(c.resultant(&c), Number::from(1));
    assert_eq!(z.discriminant(), Number::from(0));
    assert_eq!(c.discriminant(), Number::from(0));
    assert_eq!(
        ModPolynomial::from_u64(&[5, 3], 7).discriminant(),
        Number::from(1)
    );
}

#[test]
fn gcd_and_xgcd_agree_with_euclid() {
    let mut st = 0x9E37_79B9_7F4A_7C15;
    for n in primes() {
        for _ in 0..size(30, 3) {
            let lg = below(&mut st, 8) as usize;
            let g = exact(&mut st, lg, &n);
            let (la, lb) = (below(&mut st, 12) as usize, below(&mut st, 12) as usize);
            let a = &exact(&mut st, la, &n) * &g;
            let b = &exact(&mut st, lb, &n) * &g;
            check(&a, &b);
            check(&b, &a);
            check(&a, &a);
        }
    }
}

#[test]
fn half_gcd_degrees() {
    if cfg!(miri) {
        // The half gcd's degrees are out of Miri's reach; the kernels are covered above.
        return;
    }
    // Above the crossovers to the half gcd (1500 and 3000 on words, 80 and 250 on limbs): a common
    // factor of a random degree, and coprime pairs, for word primes below and above 2^63 and for
    // two- and three-limb primes.
    let mut st = 0x1234_5678_9ABC_DEF1;
    let primes = primes();
    for (i, n) in primes.iter().enumerate().skip(3) {
        let len = if n.bits() > 64 {
            600 + below(&mut st, 300) as usize
        } else {
            2600 + below(&mut st, 400) as usize
        };
        let lg = below(&mut st, len as u64 / 2) as usize + 1;
        let g = exact(&mut st, lg, n);
        let a = &exact(&mut st, len - lg + 1, n) * &g;
        let b = &exact(&mut st, len - lg - (i % 3), n) * &g;
        check(&a, &b);
        let c = exact(&mut st, len, n);
        let d = exact(&mut st, len - 1 - i, n);
        check(&c, &d);
    }
}

#[test]
fn word_gcd_past_the_half_gcd_crossover() {
    if cfg!(miri) {
        // The half gcd's degrees are out of Miri's reach; the kernels are covered above.
        return;
    }
    // gcd alone at a degree above the crossover (3000), against Euclid's algorithm.
    let mut st = 0x0A0B_0C0D_0E0F_1011;
    for n in [
        Number::from(1_000_000_007u64),
        Number::from(18_446_744_073_709_551_557u64),
    ] {
        let g = exact(&mut st, 1500, &n);
        let a = &exact(&mut st, 3000, &n) * &g;
        let b = &exact(&mut st, 2990, &n) * &g;
        assert_eq!(a.gcd(&b), euclid(&a, &b));
    }
}

#[test]
fn word_gcds_past_the_crossovers_of_small_and_full_words() {
    if cfg!(miri) {
        // The half gcd's degrees are out of Miri's reach; the kernels are covered above.
        return;
    }
    // The crossovers depend on the size of the modulus: past them for 17 and 30 bits (gcd from
    // 1000 and 1900, xgcd from 600 and 950) and for a modulus above 2^63 (2000 and 850), the gcd,
    // xgcd and resultant against Euclid's algorithm, Bézout's identity and a coprime pair.
    let mut st = 0x1122_3344_5566_7788;
    for (n, lg, la) in [
        (Number::from(65_537u64), 400, 1400),
        (Number::from((1u64 << 30) - 35), 700, 1300),
        (Number::from(18_446_744_073_709_551_557u64), 300, 1000),
    ] {
        let g = exact(&mut st, lg, &n);
        let a = &exact(&mut st, la, &n) * &g;
        let b = &exact(&mut st, la - 10, &n) * &g;
        check(&a, &b);
        assert!(a.resultant(&b).is_zero());
        let (c, d) = (exact(&mut st, la + lg, &n), exact(&mut st, la + lg - 7, &n));
        if c.gcd(&d).degree() == Some(0) {
            assert!(!c.resultant(&d).is_zero());
        }
        check(&c, &d);
    }
}

#[test]
fn resultants_agree_with_sylvester() {
    let mut st = 0x0F0F_1234_ABCD_9876;
    for n in primes() {
        for _ in 0..size(25, 2) {
            let a = {
                let l = below(&mut st, 9) as usize;
                exact(&mut st, l, &n)
            };
            let b = {
                let l = below(&mut st, 9) as usize;
                exact(&mut st, l, &n)
            };
            assert_eq!(a.resultant(&b), sylvester(&a, &b), "res({a}, {b})");
            // A common factor makes it 0.
            let g = exact(&mut st, 2, &n);
            assert!((&a * &g).resultant(&(&b * &g)).is_zero());
        }
    }
}

#[test]
fn resultant_identities_at_half_gcd_degrees() {
    if cfg!(miri) {
        // The half gcd's degrees are out of Miri's reach; the kernels are covered above.
        return;
    }
    // res(a, b·c) = res(a, b)·res(a, c), and res(a, b) = (-1)^(deg a deg b) res(b, a).
    let mut st = 0x5555_AAAA_3333_CCCC;
    for n in primes().iter().skip(3) {
        // Past the crossovers: 3000 on words, 250 on limbs.
        let (la, lb, lc) = if n.bits() > 64 {
            (400, 250, 200)
        } else {
            (4200, 2200, 2000)
        };
        for _ in 0..size(1, 0) {
            let l = la + below(&mut st, 100) as usize;
            let a = exact(&mut st, l, n);
            let l = lb + below(&mut st, 100) as usize;
            let b = exact(&mut st, l, n);
            let l = lc + below(&mut st, 100) as usize;
            let c = exact(&mut st, l, n);
            let rb = a.resultant(&b);
            let rc = a.resultant(&c);
            let rbc = a.resultant(&(&b * &c));
            assert_eq!(rbc, (&rb * &rc).rem_euclid(n));
            let (da, db) = (a.degree().unwrap(), b.degree().unwrap());
            let rba = b.resultant(&a);
            let flip = if da * db % 2 == 1 && !rba.is_zero() {
                n - &rba
            } else {
                rba
            };
            assert_eq!(rb, flip);
            // A shared root makes it 0.
            let g = exact(&mut st, 2, n);
            assert!((&a * &g).resultant(&(&c * &g)).is_zero());
        }
    }
}

#[test]
fn discriminants() {
    let mut st = 0x7777_1111_2222_3333;
    for n in primes() {
        for _ in 0..size(20, 2) {
            // disc(f) = (-1)^(d(d-1)/2) res(f, f') / lc(f), and disc(g²h) = 0.
            let f = {
                let l = 2 + below(&mut st, 8) as usize;
                exact(&mut st, l, &n)
            };
            let d = f.degree().unwrap();
            let lc = f.leading().unwrap();
            // The Sylvester determinant of f and f' of the formal degree d - 1.
            let r = sylvester_formal(&f, d, &f.derivative(), d - 1);
            let mut want = (&r * &lc.mod_inverse(&n).unwrap()).rem_euclid(&n);
            if d * (d - 1) / 2 % 2 == 1 {
                want = (&n - &want).rem_euclid(&n);
            }
            assert_eq!(f.discriminant(), want, "disc({f})");
            let g = exact(&mut st, 2, &n);
            assert!((&(&g * &g) * &f).discriminant().is_zero());
        }
    }
}

#[test]
fn composite_moduli_where_the_divisions_are_by_units() {
    // A monic divisor of the other: one division, without a remainder.
    let mut st = 0xABCD_0123_4567_89EF;
    for n in moduli().iter().skip(9) {
        let mut g = exact(&mut st, 5, n);
        g = g.to_monic();
        let c = exact(&mut st, 4, n);
        let a = &g * &c;
        assert_eq!(a.gcd(&g), g);
        let (h, s, t) = g.xgcd(&a);
        assert_eq!(h, g);
        assert_eq!(&(&s * &g) + &(&t * &a), g);
    }
}

#[test]
#[should_panic(expected = "the leading coefficient is not a unit modulo n")]
fn a_non_unit_leading_coefficient_panics() {
    // Modulo 6, 2x + 1 has a leading coefficient that is not a unit.
    let a = ModPolynomial::from_u64(&[1, 0, 1], 6);
    let b = ModPolynomial::from_u64(&[1, 2], 6);
    let _ = a.gcd(&b);
}

#[test]
#[should_panic(expected = "the leading coefficient is not a unit modulo n")]
fn a_non_unit_leading_coefficient_of_several_limbs_panics() {
    // Modulo 2^128 the pseudo-remainder steps take no inverse; the product of the leading
    // coefficients, inverted once, still finds the even one.
    let n = &Number::from(1) << 128usize;
    let a = ModPolynomial::new(&[1.into(), 3.into(), 0.into(), 1.into()], &n);
    let b = ModPolynomial::new(&[1.into(), 2.into(), 2.into()], &n);
    let _ = a.gcd(&b);
}

#[test]
fn a_non_unit_divisor_always_panics() {
    // Modulo 2^128 a divisor of an even leading coefficient, which the pseudo-remainder steps do
    // not invert: whatever the rest of the sequence, the gcd must refuse it.
    let n = &Number::from(1) << 128usize;
    let mut st = 0x2BAD_0DD0_1234_5678;
    for _ in 0..size(20, 2) {
        let la = 3 + below(&mut st, 10) as usize;
        let mut a = coeffs(&mut st, la, 17);
        a[la - 1] = Number::from(1);
        let lb = 2 + below(&mut st, (la - 2) as u64) as usize;
        let mut b = coeffs(&mut st, lb, 17);
        b[lb - 1] = Number::from(2 * (1 + below(&mut st, 1000)));
        let (a, b) = (ModPolynomial::new(&a, &n), ModPolynomial::new(&b, &n));
        let r = std::panic::catch_unwind(|| a.gcd(&b));
        assert!(
            r.is_err(),
            "gcd modulo 2^128 of {a} and {b} by an even leading coefficient"
        );
    }
}

#[test]
#[should_panic(expected = "the polynomials have different moduli")]
fn gcd_of_different_moduli_panics() {
    let _ = ModPolynomial::from_u64(&[1, 1], 5).gcd(&ModPolynomial::from_u64(&[1], 7));
}
