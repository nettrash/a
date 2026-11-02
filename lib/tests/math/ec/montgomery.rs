use super::common::{Affine, Rng, add_m, hex, n, size, times};
use a::math::ec::{Curve, CurveError, Edwards, Montgomery, Point, Weierstrass};
use a::math::numbers::{ByteOrder, Number};

/// Every point of `B y^2 = x^3 + A x^2 + x` modulo the small prime `p`.
fn all_points(a: i64, b: i64, p: i64) -> Vec<Affine> {
    let mut points = vec![None];
    for x in 0..p {
        for y in 0..p {
            if (b * y * y - (x * x * x + a * x * x + x)).rem_euclid(p) == 0 {
                points.push(Some((n(x), n(y))));
            }
        }
    }
    points
}

/// The curve's point for the reference's.
fn on<'c>(curve: &'c Montgomery, p: &Affine) -> Point<'c, Montgomery> {
    match p {
        Some((x, y)) => curve.point(x, y).unwrap(),
        None => curve.identity(),
    }
}

/// Sums and products of every pair and every point of a small curve, against the textbook.
fn exhaustive(a: i64, b: i64, p: i64) {
    let curve = Montgomery::new(&n(a), &n(b), &n(p)).unwrap();
    let (ra, rb, rp) = (n(a), n(b), n(p));
    let points = all_points(a, b, p);
    for u in &points {
        let pu = on(&curve, u);
        for v in &points {
            let want = add_m(u, v, &ra, &rb, &rp);
            assert_eq!((&pu + &on(&curve, v)).coordinates(), want);
        }
        assert_eq!(pu.double().coordinates(), add_m(u, u, &ra, &rb, &rp));
        // The ladder, with its cases of `k p` and `(k + 1) p` at infinity and points of order two.
        let mut multiple = None;
        for k in 0..points.len() as i64 + 2 {
            assert_eq!((&pu * &n(k)).coordinates(), multiple, "{k} {u:?}");
            if k > 0 {
                let x = multiple.as_ref().map_or(n(0), |m| m.0.clone());
                if let Some((ux, _)) = u {
                    assert_eq!(curve.x_mul(ux, &n(k)), x);
                }
            }
            multiple = add_m(&multiple, u, &ra, &rb, &rp);
        }
    }
}

#[test]
fn small_curves_exhaustively() {
    let curves: &[(i64, i64, i64)] = if cfg!(miri) {
        &[(3, 1, 7)]
    } else {
        &[(6, 1, 101), (6, 3, 101), (3, 2, 37), (5, 1, 47), (2, 1, 5)]
    };
    for &(a, b, p) in curves {
        if (a * a - 4) % p != 0 {
            exhaustive(a, b, p);
        }
    }
}

#[test]
fn every_point_lifts() {
    // Primes 5 and 1 modulo 8 and 3 modulo 4, the three ways to the square root of `y^2`, with `B`
    // one and not.
    for &(a, b, p) in &[
        (6, 3, 101),
        (3, 2, 37),
        (5, 1, 47),
        (5, 2, 47),
        (3, 1, 41),
        (5, 3, 41),
    ] {
        let curve = Montgomery::new(&n(a), &n(b), &n(p)).unwrap();
        let points = all_points(a, b, p);
        let mut lifted = 0;
        for x in 0..p {
            for odd in [false, true] {
                if let Some(q) = curve.lift(&n(x), odd) {
                    let (qx, qy) = q.coordinates().unwrap();
                    assert_eq!((&qx, qy.bit(0)), (&n(x), odd));
                    assert!(points.contains(&Some((qx, qy))));
                    lifted += 1;
                }
            }
        }
        // Each point once: `y` and `-y` have different parities, and `0` is even.
        assert_eq!(lifted, points.len() - 1);
    }
}

/// A curve modulo `p` and a point on it.
fn random_curve(p: &Number, r: &mut Rng) -> (Montgomery, Number, Number, Number) {
    loop {
        let (a, b) = (r.below(p), r.below(p));
        let Ok(curve) = Montgomery::new(&a, &b, p) else {
            continue;
        };
        for _ in 0..200 {
            let x = r.below(p);
            if curve.lift(&x, false).is_some() {
                return (curve, a, b, x);
            }
        }
    }
}

#[test]
fn fields_of_every_shape() {
    let mut r = Rng(0xABCD_EF01_2345_6789);
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
        let mut v = r.number(8 * l);
        v.set_bit(8 * 8 * l as u64 - 1, true);
        primes.push(v.next_prime());
    }
    for p in &primes {
        if cfg!(miri) && p.bits() > 256 {
            continue;
        }
        let (curve, a, b, x) = random_curve(p, &mut r);
        let g = curve.lift(&x, true).unwrap();
        let u = g.coordinates();
        let add = |s: &Affine, t: &Affine| add_m(s, t, &a, &b, p);
        let k = r.number(p.value.len().min(size(4096, 2)));
        let want = times(&u, &k, None, add);
        let product = &g * &k;
        assert_eq!(product.coordinates(), want);
        assert_eq!((-&product).double(), -product.double());
        assert_eq!((&product + &g).coordinates(), add(&want, &u));
        assert_eq!(&(&product + &g) - &g, product);
        assert_eq!(
            curve.x_mul(&x, &k),
            want.as_ref().map_or(n(0), |w| w.0.clone()).rem_euclid(p)
        );
        assert_eq!(
            curve.point_from_bytes(&product.compress()).unwrap(),
            product
        );
        assert_eq!(
            curve.point_from_bytes(&product.to_bytes()).unwrap(),
            product
        );
        assert_eq!(
            curve.multi_mul(&[(&k, &g), (&k, &product)]),
            &product + &(&product * &k)
        );
        let m = r.number(6);
        assert_eq!(g.prepare(50).mul(&m), &g * &m);

        // The isomorphic Weierstrass curve, whose group is the same.
        let w = curve.to_weierstrass();
        assert_eq!(w.j_invariant(), curve.j_invariant());
        let image = curve.map_to_weierstrass(&w, &g);
        assert_eq!(curve.map_to_weierstrass(&w, &product), &image * &k);
        assert_eq!(
            curve.map_to_weierstrass(&w, &curve.identity()),
            w.identity()
        );
    }
}

#[test]
fn curves_are_refused() {
    let check = |a, b, p, e| assert_eq!(Montgomery::new(&n(a), &n(b), &n(p)).err(), Some(e));
    check(6, 1, 15, CurveError::NotPrime);
    check(6, 1, 3, CurveError::NotPrime);
    check(6, 0, 101, CurveError::Singular);
    check(2, 1, 101, CurveError::Singular);
    check(99, 1, 101, CurveError::Singular);
}

#[test]
fn coefficients_and_invariants() {
    let p = n(101);
    let curve = Montgomery::new(&n(6), &n(3), &p).unwrap();
    assert_eq!((curve.a(), curve.b()), (&n(6), &n(3)));
    let b6 = n(3 * 3 * 3 * 3 * 3 * 3).mod_inverse(&p).unwrap();
    assert_eq!(curve.discriminant(), (n(16 * 32) * b6).rem_euclid(&p));
    let want = (n(256 * 33 * 33 * 33) * n(32).mod_inverse(&p).unwrap()).rem_euclid(&p);
    assert_eq!(curve.j_invariant(), want);
    // Coefficients in the other byte order come back in the prime's.
    let le = |x| Number::new_from_u32(x, ByteOrder::LittleEndian);
    let curve = Montgomery::new(&le(6 + 3 * 101), &le(3 + 5 * 101), &p).unwrap();
    for c in [
        curve.a(),
        curve.b(),
        &curve.discriminant(),
        &curve.j_invariant(),
    ] {
        assert_eq!(c.byte_order, ByteOrder::BigEndian);
    }
    assert_eq!((curve.a(), curve.b()), (&n(6), &n(3)));
}

#[test]
#[should_panic(expected = "the target is not this curve's Weierstrass form")]
fn the_weierstrass_target_is_checked() {
    let curve = Montgomery::new(&n(6), &n(1), &n(101)).unwrap();
    let other = Weierstrass::new(&n(2), &n(3), &n(101)).unwrap();
    curve.map_to_weierstrass(&other, &curve.identity());
}

#[test]
#[should_panic(expected = "the target is not this curve's Edwards form")]
fn the_edwards_target_is_checked() {
    // `9 x^2 + y^2 = 1 + 7 x^2 y^2` is this curve's; (0, 0) goes to (0, -1), on every one.
    let curve = Montgomery::new(&n(3), &n(2), &n(13)).unwrap();
    let other = Edwards::new(&n(1), &n(8), &n(13)).unwrap();
    curve.map_to_edwards(&other, &curve.point(&n(0), &n(0)).unwrap());
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn mapping_a_point_of_another_curve_panics() {
    let curve = Montgomery::new(&n(3), &n(2), &n(13)).unwrap();
    let edwards = curve.to_edwards().unwrap();
    let other = Montgomery::new(&n(3), &n(5), &n(13)).unwrap();
    curve.map_to_edwards(&edwards, &other.point(&n(0), &n(0)).unwrap());
}

#[test]
fn the_point_of_order_two() {
    let curve = Montgomery::new(&n(6), &n(1), &n(101)).unwrap();
    let t = curve.point(&n(0), &n(0)).unwrap();
    assert_eq!(t.double(), curve.identity());
    assert!((&t * &n(2)).is_identity());
    assert_eq!(&t * &n(3), t);
    assert_eq!(&t + &t, curve.identity());
}

#[test]
fn the_edwards_form() {
    // `a = (A + 2) / B`, `d = (A - 2) / B`: complete here, with 8 a square and 4 ... chosen below.
    let p = hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF43");
    let mut r = Rng(0x0123_4567_89AB_CDEF);
    let (curve, edwards) = loop {
        let (a, b) = (r.below(&p), r.below(&p));
        let Ok(c) = Montgomery::new(&a, &b, &p) else {
            continue;
        };
        if let Ok(e) = c.to_edwards() {
            break (c, e);
        }
    };
    let x = loop {
        let x = r.below(&p);
        if curve.lift(&x, false).is_some() {
            break x;
        }
    };
    let g = curve.lift(&x, false).unwrap();
    let k = r.number(size(32, 2));
    let image = curve.map_to_edwards(&edwards, &g);
    assert_eq!(curve.map_to_edwards(&edwards, &(&g * &k)), &image * &k);
    assert_eq!(
        curve.map_to_edwards(&edwards, &curve.identity()),
        edwards.identity()
    );
    let t = curve.point(&n(0), &n(0)).unwrap();
    let e = curve.map_to_edwards(&edwards, &t);
    assert_eq!(e.coordinates(), Some((n(0), &p - 1u8)));
    let m = edwards.to_montgomery();
    assert_eq!(m.j_invariant(), curve.j_invariant());
    assert_eq!(
        edwards.map_to_montgomery(&m, &edwards.identity()),
        m.identity()
    );
    assert_eq!(
        edwards.map_to_montgomery(&m, &e),
        m.point(&n(0), &n(0)).unwrap()
    );
    let back = edwards.map_to_montgomery(&m, &image);
    assert_eq!(edwards.map_to_montgomery(&m, &(&image * &k)), &back * &k);
    assert_eq!(edwards.j_invariant(), curve.j_invariant());
}
