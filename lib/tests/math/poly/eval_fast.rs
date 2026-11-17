//! The fast paths of evaluation and interpolation: Montgomery's products modulo odd moduli of 2
//! to 16 limbs (and the division beyond and for even ones), Montgomery's products modulo odd
//! words from 2^62, the trees on the polynomials' products (cyclic middle products, products of
//! monic polynomials) past their crossovers, integer products of linear factors on words, and
//! the weights of the rational interpolation.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

/// `2^bits - c`.
fn pow2_minus(bits: usize, c: u64) -> Number {
    (Number::from(1u8) << bits) - Number::from(c)
}

/// Checks `eval_many` against `eval` (the core's Horner's rule, by division) at `count` points
/// for a polynomial of `len` coefficients modulo `n`, every `step`-th point.
fn check_eval(state: &mut u64, n: &Number, len: usize, count: usize, step: usize) {
    let p = mpoly(state, len, n);
    let bytes = n.to_str_radix(16).len() / 2 + 2;
    let xs: Vec<Number> = (0..count).map(|_| number(state, bytes)).collect();
    let v = p.eval_many(&xs);
    assert_eq!(v.len(), count);
    for i in (0..count).step_by(step.max(1)) {
        assert_eq!(p.eval(&xs[i]), v[i], "n = {n}, len = {len}, point {i}");
    }
}

/// Checks `from_roots` against the product of the linear factors.
fn check_roots(state: &mut u64, n: &Number, count: usize) {
    let bytes = n.to_str_radix(16).len() / 2 + 2;
    let r: Vec<Number> = (0..count).map(|_| number(state, bytes)).collect();
    let p = ModPolynomial::from_roots(&r, n);
    let mut q = ModPolynomial::one(n);
    for x in &r {
        q = &q * &ModPolynomial::new(&[-x, Number::from(1)], n);
    }
    assert_eq!(p, q, "n = {n}, {count} roots");
}

#[test]
fn several_limbs_by_montgomery() {
    // Odd moduli of 2 to 17 limbs (17: the division), 2^(64k) - 1, and even ones.
    let mut s = 0x5eed_0001u64;
    let mut ns: Vec<Number> = (2..=17).map(|k| pow2_minus(64 * k, 1)).collect();
    ns.extend([
        Number::from(1u8) << 128usize,
        pow2_minus(128, 0) + Number::from(2),
        Number::from(10).pow(30),
    ]);
    for n in &ns {
        for &(len, count) in &[(1, 2), (2, 3), (3, 5), (4, 4), (9, 7), (30, 41)] {
            let (len, count) = (size(len, len.min(5)), size(count, count.min(6)));
            check_eval(&mut s, n, len, count, 1);
        }
        for &count in &[1, 2, 5, 17, 40] {
            check_roots(&mut s, n, size(count, count.min(6)));
        }
    }
}

#[test]
fn several_limbs_past_the_crossovers() {
    // The trees modulo several limbs past Horner's rule (960 points of 2 limbs, 224 of 9) and
    // Newton's form (52 points); under Miri from a dozen.
    let mut s = 0x5eed_0002u64;
    for (n, len) in [
        (pow2_minus(127, 1), size(1100, 20)),
        (pow2_minus(130, 5), size(520, 20)),
        (pow2_minus(521, 1), size(300, 14)),
        (pow2_minus(1279, 1), size(60, 14)),
        (Number::from(1u8) << 192usize, size(60, 14)),
    ] {
        check_eval(&mut s, &n, len, len + 3, 13);
    }
    for (n, count) in [
        (pow2_minus(127, 1), size(70, 14)),
        (pow2_minus(255, 19), size(64, 12)),
        (pow2_minus(521, 1), size(60, 10)),
        (pow2_minus(607, 1), size(50, 10)),
        (pow2_minus(1279, 1), size(20, 9)),
    ] {
        let bytes = n.to_str_radix(16).len() / 2 + 2;
        let mut seen = std::collections::HashSet::new();
        let mut xs = Vec::new();
        while xs.len() < count {
            let x = number(&mut s, bytes);
            if seen.insert(x.rem_euclid(&n)) {
                xs.push(x);
            }
        }
        let ys: Vec<Number> = (0..count).map(|_| number(&mut s, bytes)).collect();
        let p = ModPolynomial::interpolate(&xs, &ys, &n);
        let v = p.eval_many(&xs);
        for (y, v) in ys.iter().zip(&v) {
            assert_eq!(&y.rem_euclid(&n), v, "n = {n}, {count} points");
        }
        assert_eq!(ModPolynomial::interpolate_newton(&xs, &ys, &n), p);
        assert_eq!(ModPolynomial::interpolate_lagrange(&xs, &ys, &n), p);
        // Newton's form on both sides of its crossover agrees with the tree.
        let half = count / 2;
        let q = ModPolynomial::interpolate(&xs[..half], &ys[..half], &n);
        assert_eq!(
            ModPolynomial::interpolate_newton(&xs[..half], &ys[..half], &n),
            q
        );
        check_roots(&mut s, &n, size(70, 12));
    }
}

#[test]
fn words_by_montgomery_and_the_reciprocal() {
    // Odd words from 2^62 (Montgomery's products) and even ones (Shoup's, the reciprocal), with
    // few steps or points (no constants of the points) and more.
    let mut s = 0x5eed_0003u64;
    for n in [
        (1u64 << 62) + 1,
        (1 << 62) + 2,
        (1 << 63) - 25,
        (1 << 63) - 24,
        1 << 63,
        (1 << 63) + 29,
        (1 << 63) + 2,
        u64::MAX - 58,
        u64::MAX - 1,
        u64::MAX,
    ] {
        for &(len, count) in &[(1, 1), (2, 9), (3, 3), (4, 4), (5, 13), (40, 37), (150, 9)] {
            let (len, count) = (size(len, len.min(8)), size(count, count.min(9)));
            let c: Vec<u64> = (0..len).map(|_| xorshift(&mut s)).collect();
            let p = ModPolynomial::from_u64(&c, n);
            let xs: Vec<u64> = (0..count).map(|_| xorshift(&mut s)).collect();
            let v = p.eval_many_u64(&xs);
            for (x, y) in xs.iter().zip(&v) {
                assert_eq!(p.eval(&Number::from(*x)), Number::from(*y), "n = {n}");
            }
        }
    }
    // Lagrange's form on Horner's blocks for primes from 2^63, odd and with an even neighbour
    // excluded (interpolation needs units).
    for p in [(1u64 << 63) + 29, u64::MAX - 58] {
        let m = size(100, 10);
        let xs: Vec<u64> = (0..m as u64).map(|i| i * 3_000_017 + 5).collect();
        let ys: Vec<u64> = (0..m).map(|_| xorshift(&mut s) % p).collect();
        let q = ModPolynomial::interpolate_u64(&xs, &ys, p);
        assert_eq!(q.eval_many_u64(&xs), ys, "p = {p}");
        let pn = Number::from(p);
        let (bx, by): (Vec<Number>, Vec<Number>) = (
            xs.iter().map(|&x| Number::from(x)).collect(),
            ys.iter().map(|&y| Number::from(y)).collect(),
        );
        assert_eq!(ModPolynomial::interpolate_newton(&bx, &by, &pn), q);
    }
}

#[test]
fn word_trees_past_the_crossovers() {
    // Horner's rule up to 1152, 1024 and 448 points; the trees beyond, of lengths that are not
    // powers of two, by the transforms' cyclic middle products and the monic products.
    let mut s = 0x5eed_0004u64;
    for p in [
        65_537u64,
        (1 << 30) - 35,
        (1 << 32) + 15,
        (1 << 50) - 27,
        (1 << 62) - 57,
        (1 << 63) - 25,
        u64::MAX - 58,
        u64::MAX,
    ] {
        for &(len, count) in &[(1153, 1300), (1500, 1201), (460, 2000), (2100, 700)] {
            let (len, count) = (size(len, 21), size(count, 26));
            let c: Vec<u64> = (0..len).map(|_| xorshift(&mut s) % p).collect();
            let f = ModPolynomial::from_u64(&c, p);
            let xs: Vec<u64> = (0..count).map(|_| xorshift(&mut s)).collect();
            let v = f.eval_many_u64(&xs);
            for i in (0..count).step_by(size(41, 1)) {
                assert_eq!(Number::from(v[i]), f.eval(&Number::from(xs[i])), "p = {p}");
            }
        }
        // from_roots against the product of blocks of linear factors.
        let r: Vec<u64> = (0..size(333, 19)).map(|_| xorshift(&mut s) % p).collect();
        let f = ModPolynomial::from_roots_u64(&r, p);
        let mut g = ModPolynomial::from_u64(&[1], p);
        for chunk in r.chunks(7) {
            let mut h = ModPolynomial::from_u64(&[1], p);
            for &x in chunk {
                h = &h * &ModPolynomial::from_u64(&[(p - x % p) % p, 1], p);
            }
            g = &g * &h;
        }
        assert_eq!(f, g, "p = {p}");
    }
    // Interpolation by the tree, primes only.
    for p in [65_537u64, (1 << 50) - 27, (1 << 63) - 25, u64::MAX - 58] {
        let m = size(700, 24);
        let xs: Vec<u64> = (0..m as u64).map(|i| i * 11 + 2).collect();
        let ys: Vec<u64> = (0..m).map(|_| xorshift(&mut s) % p).collect();
        let q = ModPolynomial::interpolate_u64(&xs, &ys, p);
        assert!(q.length() <= m);
        assert_eq!(q.eval_many_u64(&xs), ys, "p = {p}");
    }
}

#[test]
#[should_panic(expected = "the point 4 is repeated")]
fn newton_names_a_late_repeated_point() {
    // x₄ = x₁: the difference of the third column.
    let n = pow2_minus(127, 1);
    let xs: Vec<Number> = [1, 2, 3, 4, 2].into_iter().map(Number::from).collect();
    let ys: Vec<Number> = (0..5).map(Number::from).collect();
    let _ = ModPolynomial::interpolate_newton(&xs, &ys, &n);
}

#[test]
fn integer_roots_on_words() {
    // Roots of up to 64 bits by the two's complement words, and a larger one by the limbs.
    let mut s = 0x5eed_0005u64;
    let edge = [
        Number::from(u64::MAX),
        -Number::from(u64::MAX),
        Number::from(0),
        Number::from(1),
        Number::from(-1),
        Number::from(i64::MIN),
        Number::from(1u64 << 63),
    ];
    for count in [1usize, 2, 3, 7, 16, 17, 40] {
        let count = size(count, count.min(7));
        for wide in [false, true] {
            let mut r: Vec<Number> = (0..count)
                .map(|i| {
                    if i % 3 == 0 {
                        edge[i % edge.len()].clone()
                    } else {
                        let bytes = 1 + (xorshift(&mut s) % 8) as usize;
                        number(&mut s, bytes)
                    }
                })
                .collect();
            if wide {
                r.push(-(Number::from(u64::MAX) + Number::from(1)));
            }
            let p = Polynomial::from_roots(&r);
            let mut q = vec![Number::from(1)];
            for x in &r {
                q = oracle_mul(&q, &[-x, Number::from(1)]);
            }
            assert_eq!(p.coeffs(), &trim(q)[..], "{count} roots, wide = {wide}");
        }
    }
}

#[test]
fn rational_interpolation_weights() {
    // Integer points whose differences fill a word after a few (near 2^61) and many (small).
    let mut s = 0x5eed_0006u64;
    for (base, step) in [
        (1i64 << 61, (1i64 << 58) + 3),
        (-5, 1),
        (-(1i64 << 61), 7_777_777),
    ] {
        let m = size(40, 6);
        let xs: Vec<Rational> = (0..m as i64)
            .map(|i| {
                Rational::from(
                    Number::from(base)
                        + Number::from(i)
                            * Number::from(step)
                            * Number::from(if i % 2 == 0 { 1 } else { -1 }),
                )
            })
            .collect();
        let ys: Vec<Rational> = (0..m).map(|_| rational(&mut s, 6)).collect();
        let p = Polynomial::interpolate(&xs, &ys);
        assert!(p.length() <= m);
        assert_eq!(p.eval_many(&xs), ys, "base = {base}");
        if m <= 12 {
            assert_eq!(Polynomial::interpolate_newton(&xs, &ys), p);
        }
    }
}
