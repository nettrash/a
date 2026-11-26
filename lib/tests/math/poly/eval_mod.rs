//! Evaluation at many points, products of linear factors and interpolation of `ModPolynomial`s.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;

/// `count` random residues-to-be of either sign, larger than the modulus.
fn points(state: &mut u64, count: usize, n: &Number) -> Vec<Number> {
    let bytes = n.to_str_radix(16).len() / 2 + 2;
    (0..count).map(|_| number(state, bytes)).collect()
}

/// `count` distinct points modulo the prime `n`, when it has that many.
fn distinct(state: &mut u64, count: usize, n: &Number) -> Vec<Number> {
    let mut seen = std::collections::HashSet::new();
    let mut v = Vec::new();
    while v.len() < count {
        let x = number(state, n.to_str_radix(16).len() / 2 + 2);
        if seen.insert(x.rem_euclid(n)) {
            v.push(x);
        }
    }
    v
}

#[test]
fn eval_many_matches_eval() {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for n in moduli() {
        for &(len, count) in &[
            (0, 3),
            (1, 5),
            (3, 0),
            (5, 9),
            (40, 200),
            (130, 300),
            (300, 700),
        ] {
            let (len, count) = (size(len, len.min(20)), size(count, count.min(30)));
            let p = mpoly(&mut s, len, &n);
            let xs = points(&mut s, count, &n);
            let v = p.eval_many(&xs);
            assert_eq!(v.len(), xs.len());
            for (x, y) in xs.iter().zip(&v) {
                assert_eq!(&p.eval(x), y, "n = {n}, len = {}", p.length());
            }
        }
    }
}

#[test]
fn eval_many_words() {
    let mut s = 7u64;
    for &n in &[
        2u64,
        3,
        1 << 29,
        (1 << 30) - 35,
        (1 << 30) + 3,
        1_000_000_007,
        (1 << 62) - 57,
        (1 << 63) - 25,
        u64::MAX - 58,
        u64::MAX,
    ] {
        for &(len, count) in &[(1, 3), (17, 50), (40, 333), (200, 450), (97, 1000)] {
            let (len, count) = (size(len, len.min(20)), size(count, count.min(30)));
            let c: Vec<u64> = (0..len).map(|_| xorshift(&mut s)).collect();
            let p = ModPolynomial::from_u64(&c, n);
            let xs: Vec<u64> = (0..count).map(|_| xorshift(&mut s)).collect();
            let v = p.eval_many_u64(&xs);
            for (x, y) in xs.iter().zip(&v) {
                assert_eq!(p.eval(&Number::from(*x)), Number::from(*y), "n = {n}");
            }
        }
    }
}

#[test]
fn from_roots_is_the_product() {
    let mut s = 11u64;
    for n in moduli() {
        for &count in &[0, 1, 2, 3, 7, 30, 70] {
            let count = size(count, count.min(10));
            let r = points(&mut s, count, &n);
            let p = ModPolynomial::from_roots(&r, &n);
            let mut q = ModPolynomial::one(&n);
            for x in &r {
                q = &q * &ModPolynomial::new(&[-x, Number::from(1)], &n);
            }
            assert_eq!(p, q, "n = {n}, {count} roots");
            assert!(p.eval_many(&r).iter().all(Number::is_zero));
        }
    }
}

#[test]
fn interpolate_round_trips() {
    let mut s = 13u64;
    for n in primes().into_iter().skip(3) {
        for &count in &[0, 1, 2, 3, 10, 49, 120, 300] {
            let count = size(count, count.min(12));
            let xs = distinct(&mut s, count, &n);
            let ys = points(&mut s, count, &n);
            let p = ModPolynomial::interpolate(&xs, &ys, &n);
            assert!(p.length() <= count);
            let v = p.eval_many(&xs);
            for (y, v) in ys.iter().zip(&v) {
                assert_eq!(&y.rem_euclid(&n), v, "n = {n}, {count} points");
            }
            assert_eq!(ModPolynomial::interpolate_newton(&xs, &ys, &n), p);
            if count <= 120 {
                assert_eq!(ModPolynomial::interpolate_lagrange(&xs, &ys, &n), p);
            }
        }
    }
}

#[test]
fn edge_cases() {
    let n = Number::from(7);
    // No roots, no points, the zero polynomial, no evaluation points.
    assert!(ModPolynomial::from_roots(&[], &n).is_one());
    assert!(ModPolynomial::interpolate(&[], &[], &n).is_zero());
    assert!(ModPolynomial::interpolate_newton(&[], &[], &n).is_zero());
    assert!(ModPolynomial::interpolate_lagrange(&[], &[], &n).is_zero());
    let z = ModPolynomial::zero(&n);
    let xs: Vec<Number> = (0..5).map(Number::from).collect();
    assert!(z.eval_many(&xs).iter().all(Number::is_zero));
    assert!(
        ModPolynomial::from_u64(&[3, 1], 7)
            .eval_many(&[])
            .is_empty()
    );
    // A constant.
    let c = ModPolynomial::from_u64(&[4], 7);
    assert_eq!(c.eval_many_u64(&[0, 1, 100]), [4, 4, 4]);
    let one = ModPolynomial::interpolate(&[Number::from(3)], &[Number::from(-1)], &n);
    assert_eq!(one, ModPolynomial::from_u64(&[6], 7));
    // p = 2 and 3: every point.
    for p in [2u64, 3] {
        let xs: Vec<u64> = (0..p).collect();
        for code in 0..p.pow(p as u32) {
            let ys: Vec<u64> = (0..p).map(|i| code / p.pow(i as u32) % p).collect();
            let f = ModPolynomial::interpolate_u64(&xs, &ys, p);
            assert_eq!(f.eval_many_u64(&xs), ys);
            let pn = Number::from(p);
            let (bx, by): (Vec<Number>, Vec<Number>) = (
                xs.iter().map(|&x| Number::from(x)).collect(),
                ys.iter().map(|&y| Number::from(y)).collect(),
            );
            assert_eq!(ModPolynomial::interpolate_newton(&bx, &by, &pn), f);
            assert_eq!(ModPolynomial::interpolate_lagrange(&bx, &by, &pn), f);
        }
        let r = ModPolynomial::from_roots_u64(&xs, p);
        // x^p - x
        let mut want = vec![0u64; p as usize + 1];
        want[1] = p - 1;
        want[p as usize] = 1;
        assert_eq!(r, ModPolynomial::from_u64(&want, p));
    }
    // A composite modulus whose differences of the points are units.
    let n15 = Number::from(15);
    let xs: Vec<Number> = [0, 1, 2].into_iter().map(Number::from).collect();
    let ys: Vec<Number> = [4, 9, 1].into_iter().map(Number::from).collect();
    let f = ModPolynomial::interpolate(&xs, &ys, &n15);
    assert_eq!(f.eval_many(&xs), ys);
    // The words and the numbers agree.
    let mut s = 3u64;
    let p = (1u64 << 61) - 1;
    let m = size(200, 12) as u64;
    let xs: Vec<u64> = (0..m).map(|i| i * 7919 + 3).collect();
    let ys: Vec<u64> = (0..m).map(|_| xorshift(&mut s) % p).collect();
    let (bx, by): (Vec<Number>, Vec<Number>) = (
        xs.iter().map(|&x| Number::from(x)).collect(),
        ys.iter().map(|&y| Number::from(y)).collect(),
    );
    let pn = Number::from(p);
    assert_eq!(
        ModPolynomial::interpolate_u64(&xs, &ys, p),
        ModPolynomial::interpolate(&bx, &by, &pn)
    );
    assert_eq!(
        ModPolynomial::from_roots_u64(&xs, p),
        ModPolynomial::from_roots(&bx, &pn)
    );
}

#[test]
#[should_panic(expected = "distinct modulo n")]
fn repeated_point_modulo_n() {
    let xs = [Number::from(1), Number::from(8)];
    let _ = ModPolynomial::interpolate(&xs, &[Number::from(0), Number::from(1)], &Number::from(7));
}

#[test]
#[should_panic(expected = "distinct modulo n")]
fn repeated_point_newton() {
    let xs: Vec<Number> = [1, 2, 3, 15].into_iter().map(Number::from).collect();
    let ys: Vec<Number> = (0..4).map(Number::from).collect();
    let _ = ModPolynomial::interpolate_newton(&xs, &ys, &Number::from(7));
}

#[test]
#[should_panic(expected = "distinct modulo n")]
fn repeated_point_lagrange() {
    let xs: Vec<Number> = [1, 9].into_iter().map(Number::from).collect();
    let ys: Vec<Number> = (0..2).map(Number::from).collect();
    let _ = ModPolynomial::interpolate_lagrange(&xs, &ys, &Number::from(4));
}

#[test]
#[should_panic(expected = "distinct modulo n")]
fn non_unit_difference_tree() {
    // Modulo 2^64 - 1 = 3·5·17·257·641·65537·6700417, the points 0 and 3 differ by a non-unit;
    // enough points for the tree.
    let n = Number::from(u64::MAX);
    let xs: Vec<Number> = (0..100)
        .map(|i| Number::from(if i == 1 { 3 } else { 2 * i + 1 }))
        .collect();
    let mut xs = xs;
    xs[0] = Number::from(0);
    let ys: Vec<Number> = (0..100).map(Number::from).collect();
    let _ = ModPolynomial::interpolate(&xs, &ys, &n);
}

#[test]
#[should_panic(expected = "below 2^64")]
fn words_need_a_word_modulus() {
    let n = Number::from(u64::MAX) + Number::from(2);
    let _ = ModPolynomial::one(&n).eval_many_u64(&[1]);
}

#[test]
#[should_panic(expected = "as many values")]
fn lengths_differ_modulo_n() {
    let _ = ModPolynomial::interpolate_u64(&[1, 2], &[3], 7);
}

#[test]
fn large_degrees() {
    let mut s = 21u64;
    let p = (1u64 << 50) - 27;
    for &(len, count) in &[(5000, 5000), (3000, 9000), (200, 30000)] {
        let (len, count) = (size(len, 40), size(count, 50));
        let c: Vec<u64> = (0..len).map(|_| xorshift(&mut s) % p).collect();
        let f = ModPolynomial::from_u64(&c, p);
        let xs: Vec<u64> = (0..count).map(|_| xorshift(&mut s)).collect();
        let v = f.eval_many_u64(&xs);
        for i in (0..count).step_by(count / 17 + 1) {
            assert_eq!(Number::from(v[i]), f.eval(&Number::from(xs[i])));
        }
    }
    let m = size(4000, 60);
    let xs: Vec<u64> = (0..m as u64).map(|i| i * i + 1).collect();
    let ys: Vec<u64> = (0..m).map(|_| xorshift(&mut s) % p).collect();
    let f = ModPolynomial::interpolate_u64(&xs, &ys, p);
    assert_eq!(f.eval_many_u64(&xs), ys);
    let r = ModPolynomial::from_roots_u64(&xs, p);
    assert_eq!(r.degree(), Some(m));
    assert!(r.eval_many_u64(&xs).iter().all(|&v| v == 0));
}

#[test]
fn word_trees() {
    // The trees modulo a word past their crossovers (Miri included: products by Kronecker
    // substitution, the interpolation tree).
    let mut s = 51u64;
    let p = (1u64 << 50) - 27;
    let r: Vec<u64> = (0..200).map(|_| xorshift(&mut s) % p).collect();
    let f = ModPolynomial::from_roots_u64(&r, p);
    let mut g = ModPolynomial::from_u64(&[1], p);
    for chunk in r.chunks(50) {
        let mut h = ModPolynomial::from_u64(&[1], p);
        for &x in chunk {
            h = &h * &ModPolynomial::from_u64(&[p - x, 1], p);
        }
        g = &g * &h;
    }
    assert_eq!(f, g);
    let m = 460;
    let xs: Vec<u64> = (0..m as u64).map(|i| 3 * i + 7).collect();
    let ys: Vec<u64> = (0..m).map(|_| xorshift(&mut s) % p).collect();
    let q = ModPolynomial::interpolate_u64(&xs, &ys, p);
    for i in (0..m).step_by(37) {
        assert_eq!(q.eval(&Number::from(xs[i])), Number::from(ys[i]));
    }
    if !cfg!(miri) {
        assert_eq!(q.eval_many_u64(&xs), ys);
        let big = 2100;
        let c: Vec<u64> = (0..big).map(|_| xorshift(&mut s) % p).collect();
        let f = ModPolynomial::from_u64(&c, p);
        let xs: Vec<u64> = (0..big).map(|_| xorshift(&mut s)).collect();
        let v = f.eval_many_u64(&xs);
        for i in (0..big).step_by(97) {
            assert_eq!(Number::from(v[i]), f.eval(&Number::from(xs[i])));
        }
    }
}

#[test]
#[should_panic(expected = "at least 2")]
fn word_modulus_of_one() {
    let _ = ModPolynomial::from_roots_u64(&[], 1);
}

#[test]
fn words_near_the_top() {
    // Moduli just below 2^62, 2^63 and 2^64, where Shoup's products leave their result in 0..2n
    // most often and the reciprocal takes over.
    let mut s = 61u64;
    for p in [(1u64 << 62) - 57, (1 << 63) - 25, u64::MAX - 58] {
        let r: Vec<u64> = (0..size(150, 20)).map(|_| xorshift(&mut s) % p).collect();
        let f = ModPolynomial::from_roots_u64(&r, p);
        let mut g = ModPolynomial::from_u64(&[1], p);
        for &x in &r {
            g = &g * &ModPolynomial::from_u64(&[p - x, 1], p);
        }
        assert_eq!(f, g, "p = {p}");
        let xs: Vec<u64> = (0..r.len() as u64).map(|i| i * 1_000_003 + 11).collect();
        let ys: Vec<u64> = (0..r.len()).map(|_| xorshift(&mut s) % p).collect();
        let q = ModPolynomial::interpolate_u64(&xs, &ys, p);
        assert_eq!(q.eval_many_u64(&xs), ys, "p = {p}");
        let pn = Number::from(p);
        let (bx, by): (Vec<Number>, Vec<Number>) = (
            xs.iter().map(|&x| Number::from(x)).collect(),
            ys.iter().map(|&y| Number::from(y)).collect(),
        );
        assert_eq!(ModPolynomial::interpolate_newton(&bx, &by, &pn), q);
        assert_eq!(ModPolynomial::interpolate_lagrange(&bx, &by, &pn), q);
    }
}
