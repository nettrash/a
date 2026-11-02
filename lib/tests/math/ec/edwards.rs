use super::common::{Rng, add_e, n, size, times};
use a::math::ec::{Curve, CurveError, Edwards, Point};
use a::math::numbers::{ByteOrder, Number};

/// Every point of `a x^2 + y^2 = 1 + d x^2 y^2` modulo the small prime `p`.
fn all_points(a: i64, d: i64, p: i64) -> Vec<(Number, Number)> {
    let mut points = Vec::new();
    for x in 0..p {
        for y in 0..p {
            if (a * x * x + y * y - 1 - d * x * x * y * y).rem_euclid(p) == 0 {
                points.push((n(x), n(y)));
            }
        }
    }
    points
}

/// Sums and products of every pair and every point of a small curve, against the textbook.
fn exhaustive(a: i64, d: i64, p: i64) {
    let curve = Edwards::new(&n(a), &n(d), &n(p)).unwrap();
    let (ra, rd, rp) = (n(a).rem_euclid(&n(p)), n(d), n(p));
    let points = all_points(a, d, p);
    let zero = (n(0), n(1));
    for u in &points {
        let pu = curve.point(&u.0, &u.1).unwrap();
        for v in &points {
            let pv = curve.point(&v.0, &v.1).unwrap();
            let want = add_e(u, v, &ra, &rd, &rp);
            assert_eq!((&pu + &pv).coordinates(), Some(want));
            assert_eq!(pu == pv, u == v);
        }
        assert_eq!(pu.double().coordinates(), Some(add_e(u, u, &ra, &rd, &rp)));
        let mut multiple = zero.clone();
        for k in 0..points.len() as i64 + 2 {
            assert_eq!((&pu * &n(k)).coordinates(), Some(multiple.clone()));
            assert_eq!(pu.prepare(8).mul(&n(k)), &pu * &n(k));
            multiple = add_e(&multiple, u, &ra, &rd, &rp);
        }
        assert_eq!(pu.is_identity(), *u == zero);
    }
}

#[test]
fn small_curves_exhaustively() {
    // With `a = -1`, whose formulas are their own, and `a = 1`, `a = 4`.
    let curves: &[(i64, i64, i64)] = if cfg!(miri) {
        &[(1, 2, 13)]
    } else {
        &[
            (1, 2, 13),
            (-1, 2, 13),
            (-1, 2, 29),
            (4, 3, 17),
            (1, 5, 37),
            (-1, 2, 37),
        ]
    };
    for &(a, d, p) in curves {
        exhaustive(a, d, p);
    }
}

#[test]
fn every_point_lifts() {
    // Primes 5 and 1 modulo 8 and 3 modulo 4, the three ways to the square root of the ratio
    // `x^2 = (y^2 - 1) / (d y^2 - a)`.
    for &(a, d, p) in &[(1, 2, 13), (-1, 2, 29), (4, 3, 17), (-1, 3, 41), (1, 2, 19)] {
        let curve = Edwards::new(&n(a), &n(d), &n(p)).unwrap();
        let points = all_points(a, d, p);
        let mut lifted = 0;
        for y in 0..p {
            for odd in [false, true] {
                if let Some(q) = curve.lift(&n(y), odd) {
                    let (qx, qy) = q.coordinates().unwrap();
                    assert_eq!((qx.bit(0), &qy), (odd, &n(y)));
                    assert!(points.contains(&(qx, qy)));
                    lifted += 1;
                }
            }
        }
        // Each point once: `x` and `-x` have different parities, and `0` is even.
        assert_eq!(lifted, points.len());
    }
}

/// A complete curve modulo `p`, with `a = -1` or not.
fn random_curve(p: &Number, minus_one: bool, r: &mut Rng) -> (Edwards, Number, Number) {
    loop {
        let (a, d) = if minus_one {
            (p - 1u8, r.below(p))
        } else {
            let s = r.below(p);
            ((&s * &s).rem_euclid(p), r.below(p))
        };
        if let Ok(curve) = Edwards::new(&a, &d, p) {
            return (curve, a, d);
        }
    }
}

#[test]
fn fields_of_every_shape() {
    let mut r = Rng(0xFEDC_BA98_7654_3210);
    let mut primes = vec![
        (Number::new_from_u8(1, ByteOrder::BigEndian) << 255usize) - 19u8,
        (Number::new_from_u8(1, ByteOrder::BigEndian) << 521usize) - 1u8,
    ];
    let limbs: &[usize] = if cfg!(miri) {
        &[1, 4]
    } else {
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13]
    };
    for &l in limbs {
        // 1 modulo 4, so that -1 is a square.
        let mut v = r.number(8 * l);
        v.set_bit(8 * 8 * l as u64 - 1, true);
        let mut c = v.next_prime();
        while !c.bit(0) || c.bit(1) {
            c = (&c + 1u8).next_prime();
        }
        primes.push(c);
    }
    for p in &primes {
        if cfg!(miri) && p.bits() > 256 {
            continue;
        }
        for minus_one in [true, false] {
            if minus_one && p.bit(1) {
                continue;
            }
            let (curve, a, d) = random_curve(p, minus_one, &mut r);
            let (y, g) = loop {
                let y = r.below(p);
                if let Some(g) = curve.lift(&y, false) {
                    break (y, g);
                }
            };
            let u = g.coordinates().unwrap();
            assert_eq!(u.1, y);
            assert!(!u.0.bit(0));
            let add = |s: &(Number, Number), t: &(Number, Number)| add_e(s, t, &a, &d, p);
            let k = r.number(p.value.len().min(size(4096, 2)));
            let want = times(&u, &k, (n(0), n(1)), add);
            let product = &g * &k;
            assert_eq!(product.coordinates(), Some(want.clone()));
            assert_eq!((&product + &g).coordinates(), Some(add(&want, &u)));
            assert_eq!(&(&product + &g) - &g, product);
            assert_eq!(product.double().coordinates(), Some(add(&want, &want)));
            assert_eq!(
                curve.point_from_bytes(&product.compress()).unwrap(),
                product
            );
            assert_eq!(
                curve.point_from_bytes(&product.to_bytes()).unwrap(),
                product
            );
            let m = r.number(6);
            assert_eq!(g.prepare(50).mul(&m), &g * &m);
            assert_eq!(
                curve.multi_mul(&[(&k, &g), (&m, &product)]),
                &product + &(&product * &m)
            );
            // Straus's method, and Pippenger's from 64 terms.
            let count = if cfg!(miri) { 40 } else { 70 };
            let many: Vec<Number> = (0..count).map(|i| n(i * 3 + 1)).collect();
            let terms: Vec<(&Number, &Point<'_, Edwards>)> = many.iter().map(|s| (s, &g)).collect();
            assert_eq!(
                curve.multi_mul(&terms),
                &g * &n(count * (count - 1) / 2 * 3 + count)
            );
        }
    }
}

#[test]
fn curves_are_refused() {
    let check = |a, d, p, e| assert_eq!(Edwards::new(&n(a), &n(d), &n(p)).err(), Some(e));
    check(1, 2, 15, CurveError::NotPrime);
    check(1, 2, 3, CurveError::NotPrime);
    check(0, 2, 13, CurveError::Singular);
    check(1, 0, 13, CurveError::Singular);
    check(2, 2, 13, CurveError::Singular);
    // `a` not a square, then `d` a square.
    check(2, 5, 13, CurveError::NotComplete);
    check(1, 3, 13, CurveError::NotComplete);
}

#[test]
fn encodings() {
    let curve = Edwards::new(&n(1), &n(2), &n(13)).unwrap();
    let o = curve.identity();
    assert_eq!(o.compress(), [1]);
    assert_eq!(o.to_bytes(), [0, 1]);
    assert_eq!(curve.point_from_bytes(&[1]).unwrap(), o);
    assert_eq!(curve.point_from_bytes(&[0, 1]).unwrap(), o);
    assert!(curve.point_from_bytes(&[]).is_none());
    assert!(curve.point_from_bytes(&[0, 1, 2]).is_none());
    // `y` not below the prime, in both forms.
    assert!(curve.point_from_bytes(&[14]).is_none());
    assert!(curve.point_from_bytes(&[0, 14]).is_none());
    // Off the curve.
    assert!(curve.point_from_bytes(&[2, 2]).is_none());
    // `x = 0` has no odd root: the top bit set on the neutral element.
    assert!(curve.point_from_bytes(&[0x81]).is_none());
    for u in all_points(1, 2, 13) {
        let p = curve.point(&u.0, &u.1).unwrap();
        assert_eq!(curve.point_from_bytes(&p.compress()).unwrap(), p);
    }
    // A `y` with no point at all.
    let none = (0..13).find(|&y| curve.lift(&n(y), false).is_none());
    if let Some(y) = none {
        assert!(curve.point_from_bytes(&[y as u8]).is_none());
    }
}

#[test]
fn coefficients_and_invariants() {
    let p = n(13);
    let curve = Edwards::new(&n(1), &n(2), &p).unwrap();
    assert_eq!((curve.a(), curve.d()), (&n(1), &n(2)));
    // a d (a - d)^4 / 16 = 2 / 3 modulo 13.
    let sixteen = n(16).mod_inverse(&p).unwrap();
    assert_eq!(curve.discriminant(), (n(2) * sixteen).rem_euclid(&p));
    // 16 (1 + 28 + 4)^3 / 2 = 8 33^3.
    assert_eq!(curve.j_invariant(), (n(8) * n(33 * 33 * 33)).rem_euclid(&p));
    assert!(core::ptr::eq(curve.identity().curve(), &curve));
    assert_eq!(curve.modulus(), &p);
}
