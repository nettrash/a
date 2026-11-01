use super::common::{Affine, Rng, add_w, hex, n, size, times};
use a::math::ec::{Curve, CurveError, Domain, NotOnCurve, Point, Weierstrass};
use a::math::numbers::{ByteOrder, Number};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Every point of the curve modulo the small prime `p`, the one at infinity first.
fn all_points(a: i64, b: i64, p: i64) -> Vec<Affine> {
    let mut points = vec![None];
    for x in 0..p {
        for y in 0..p {
            if (y * y - (x * x * x + a * x + b)).rem_euclid(p) == 0 {
                points.push(Some((n(x), n(y))));
            }
        }
    }
    points
}

/// The curve's point for the reference's.
fn on<'c>(curve: &'c Weierstrass, p: &Affine) -> Point<'c, Weierstrass> {
    match p {
        Some((x, y)) => curve.point(x, y).unwrap(),
        None => curve.identity(),
    }
}

/// Sums and products of every pair and every point of a small curve, against the textbook.
fn exhaustive(a: i64, b: i64, p: i64) {
    let curve = Weierstrass::new(&n(a), &n(b), &n(p)).unwrap();
    let (ra, rp) = (n(a).rem_euclid(&n(p)), n(p));
    let points = all_points(a, b, p);
    for u in &points {
        let pu = on(&curve, u);
        assert_eq!(pu.coordinates(), *u);
        for v in &points {
            let pv = on(&curve, v);
            let want = add_w(u, v, &ra, &rp);
            assert_eq!((&pu + &pv).coordinates(), want);
            assert_eq!(pu == pv, u == v);
        }
        assert_eq!(pu.double().coordinates(), add_w(u, u, &ra, &rp));
        assert_eq!((-&pu).coordinates(), neg(u, &rp));
        // Every multiple up to the group's order, which reaches the identity in the table.
        let mut multiple = None;
        for k in 0..points.len() as i64 + 2 {
            assert_eq!((&pu * &n(k)).coordinates(), multiple);
            multiple = add_w(&multiple, u, &ra, &rp);
        }
    }
}

/// `-p` in the reference.
fn neg(p: &Affine, m: &Number) -> Affine {
    p.as_ref().map(|(x, y)| (x.clone(), (-y).rem_euclid(m)))
}

#[test]
fn small_curves_exhaustively() {
    // a = 0, a = -3, and a general a, whose doublings differ; curves with points of small order.
    let primes: &[(i64, i64, i64)] = if cfg!(miri) {
        &[(2, 3, 7)]
    } else {
        &[
            (2, 3, 97),
            (0, 7, 67),
            (-3, 5, 61),
            (1, 0, 53),
            (0, 1, 31),
            (1, 1, 5),
        ]
    };
    for &(a, b, p) in primes {
        exhaustive(a, b, p);
    }
}

#[test]
fn the_group_of_the_documented_curve() {
    assert_eq!(all_points(2, 3, 97).len(), 100);
}

/// A curve modulo `p` with random coefficients and a point on it.
fn random_curve(p: &Number, r: &mut Rng) -> (Weierstrass, Number, Number) {
    loop {
        let (a, b) = (r.below(p), r.below(p));
        let Ok(curve) = Weierstrass::new(&a, &b, p) else {
            continue;
        };
        for _ in 0..200 {
            let x = r.below(p);
            if curve.lift(&x, false).is_some() {
                return (curve, a, x);
            }
        }
    }
}

/// Checks the operations of a random curve modulo `p` against the textbook.
fn check_field(p: &Number, r: &mut Rng) {
    let (curve, a, x) = random_curve(p, r);
    let g = curve.lift(&x, false).unwrap();
    let u = g.coordinates();
    assert!(!u.as_ref().unwrap().1.bit(0));
    assert_eq!(u.as_ref().unwrap().0, x);
    let add = |s: &Affine, t: &Affine| add_w(s, t, &a, p);
    let k = r.number(p.value.len().min(size(4096, 2)));
    let want = times(&u, &k, None, add);
    let product = &g * &k;
    assert_eq!(product.coordinates(), want);
    let sum = &product + &g;
    assert_eq!(sum.coordinates(), add(&want, &u));
    assert_eq!((&sum - &g), product);
    assert_eq!(product.double().coordinates(), add(&want, &want));
    assert_eq!(&product + &product, product.double());
    assert_eq!(&product + &-&product, curve.identity());
    // Through Jacobian forms with `Z` other than 1.
    let again = &(&sum + &sum) - &(&product + &product);
    assert_eq!(again, g.double());
    assert_eq!(
        curve.point_from_bytes(&product.compress()).unwrap(),
        product
    );
    assert_eq!(
        curve.point_from_bytes(&product.to_bytes()).unwrap(),
        product
    );
    let (m, table) = (r.number(8), g.prepare(70));
    assert_eq!(table.mul(&m), &g * &m);
    assert_eq!(table.mul(&(&m + 1u8)), &(&g * &m) + &g);
    assert_eq!(
        curve.multi_mul(&[(&m, &g), (&k, &product)]),
        &(&g * &m) + &(&product * &k)
    );
}

#[test]
fn fields_of_every_length() {
    // One to nine limbs in arrays, then longer ones on numbers.
    let mut r = Rng(0x1234_5678_9ABC_DEF1);
    let limbs: &[usize] = if cfg!(miri) {
        &[1, 3, 4, 6, 9, 10]
    } else {
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20]
    };
    for &l in limbs {
        for _ in 0..size(4, 1) {
            let mut v = r.number(8 * l);
            v.set_bit(8 * 8 * l as u64 - 1, true);
            check_field(&v.next_prime(), &mut r);
        }
    }
}

#[test]
fn primes_just_below_a_power_of_two() {
    let mut r = Rng(0x0F1E_2D3C_4B5A_6978);
    let shapes: &[(usize, u64)] = if cfg!(miri) {
        &[(256, 189), (521, 1)]
    } else {
        &[
            (255, 19),
            (256, 189),
            (251, 9),
            (256, 0x1_0000_03D1),
            (384, 317),
            (383, 187),
            (521, 1),
            (512, 569),
        ]
    };
    for &(k, c) in shapes {
        let p = (Number::new_from_u8(1, ByteOrder::BigEndian) << k) - c;
        assert!(p.is_probable_prime());
        for _ in 0..size(4, 1) {
            check_field(&p, &mut r);
        }
    }
}

#[test]
fn every_operand_order_and_ownership() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let p = curve.point(&n(3), &n(6)).unwrap();
    let q = p.double();
    let sum = &p + &q;
    assert_eq!(p.clone() + q.clone(), sum);
    assert_eq!(p.clone() + &q, sum);
    assert_eq!(&p + q.clone(), sum);
    assert_eq!(sum.clone() - &q, p);
    assert_eq!(&sum - q.clone(), p);
    let mut s = p.clone();
    s += &q;
    assert_eq!(s, sum);
    s -= q.clone();
    assert_eq!(s, p);
    s += q.clone();
    s -= &q;
    assert_eq!(s, p);
    s *= &n(3);
    assert_eq!(s, sum);
    s *= n(2);
    assert_eq!(s, &sum * &n(2));
    assert_eq!(&n(3) * &p, sum);
    assert_eq!(n(3) * &p, sum);
    assert_eq!(&n(3) * p.clone(), sum);
    assert_eq!(n(3) * p.clone(), sum);
    assert_eq!(p.clone() * n(3), sum);
    assert_eq!(p.clone() * &n(3), sum);
    assert_eq!(&p * n(3), sum);
    assert_eq!(-p.clone(), -&p);
    // Negative, zero and long scalars.
    assert_eq!(&p * &n(-1), -&p);
    assert_eq!(&p * &n(-7), -(&p * &n(7)));
    assert!((&p * &n(0)).is_identity());
    assert_eq!(&p * &(n(5) << 1000usize), curve.identity());
    assert_eq!(&p * &((n(5) << 1000usize) + 2u8), q);
}

#[test]
fn the_coordinates_come_in_the_byte_order_of_the_prime() {
    for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
        let m = |x| Number::new_from_u32(x, order);
        let curve = Weierstrass::new(&m(2), &m(3), &m(97)).unwrap();
        let (x, y) = curve.point(&m(3), &m(6)).unwrap().coordinates().unwrap();
        assert_eq!((x.byte_order, y.byte_order), (order, order));
        assert_eq!(curve.modulus().byte_order, order);
        assert_eq!(curve.a().byte_order, order);
    }
    // Operands in the other order, and coordinates to be reduced.
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let x = Number::new_from_u32(3 + 97, ByteOrder::LittleEndian);
    let y = Number::new_from_i32(6 - 97, ByteOrder::LittleEndian);
    assert_eq!(
        curve.point(&x, &y).unwrap(),
        curve.point(&n(3), &n(6)).unwrap()
    );
}

#[test]
fn curves_are_refused() {
    let check = |a, b, p, e| assert_eq!(Weierstrass::new(&n(a), &n(b), &n(p)).err(), Some(e));
    check(1, 1, 15, CurveError::NotPrime);
    check(1, 1, 3, CurveError::NotPrime);
    check(1, 1, 2, CurveError::NotPrime);
    check(1, 1, 0, CurveError::NotPrime);
    check(1, 1, -7, CurveError::NotPrime);
    // 4 a^3 + 27 b^2 = 0: a double root.
    check(-3, 2, 97, CurveError::Singular);
    check(0, 0, 97, CurveError::Singular);
}

#[test]
fn points_off_the_curve_are_refused() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    assert_eq!(curve.point(&n(3), &n(7)).err(), Some(NotOnCurve));
    assert_eq!(NotOnCurve.to_string(), "the point is not on the curve");
    assert_eq!(CurveError::Singular.to_string(), "the curve is singular");
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn adding_across_curves_panics() {
    let c1 = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let c2 = Weierstrass::new(&n(2), &n(4), &n(97)).unwrap();
    let p = c1.point(&n(3), &n(6)).unwrap();
    let _ = &p + &c2.identity();
}

#[test]
fn two_values_of_one_curve_mix() {
    let c1 = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let c2 = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let p = c1.point(&n(3), &n(6)).unwrap();
    let q = c2.point(&n(3), &n(6)).unwrap();
    assert_eq!(&p + &q, p.double());
}

#[test]
fn encodings() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let p = curve.point(&n(3), &n(6)).unwrap();
    assert_eq!(p.compress(), [2, 3]);
    assert_eq!(p.to_bytes(), [4, 3, 6]);
    assert_eq!(curve.identity().compress(), [0]);
    assert_eq!(curve.identity().to_bytes(), [0]);
    assert_eq!(curve.point_from_bytes(&[0]).unwrap(), curve.identity());
    assert_eq!(curve.point_from_bytes(&[2, 3]).unwrap(), p);
    assert_eq!(curve.point_from_bytes(&[3, 3]).unwrap(), -&p);
    assert_eq!(curve.point_from_bytes(&[4, 3, 6]).unwrap(), p);
    assert!(curve.point_from_bytes(&[]).is_none());
    assert!(curve.point_from_bytes(&[1, 3]).is_none());
    assert!(curve.point_from_bytes(&[2]).is_none());
    assert!(curve.point_from_bytes(&[2, 3, 3]).is_none());
    assert!(curve.point_from_bytes(&[0, 0]).is_none());
    assert!(curve.point_from_bytes(&[4, 3, 7]).is_none());
    // Coordinates that are not below the prime.
    assert!(curve.point_from_bytes(&[2, 100]).is_none());
    assert!(curve.point_from_bytes(&[4, 100, 6]).is_none());
    // An `x` with no point: the right side is not a square.
    let none = (0..97)
        .find(|&x| curve.lift(&n(x), false).is_none())
        .unwrap();
    assert!(curve.point_from_bytes(&[2, none as u8]).is_none());
    // A point with `y = 0` has no odd root.
    let points = all_points(1, 0, 53);
    let c = Weierstrass::new(&n(1), &n(0), &n(53)).unwrap();
    let flat = points
        .iter()
        .flatten()
        .find(|(_, y)| y.is_zero())
        .map(|(x, _)| x)
        .unwrap();
    assert!(c.lift(flat, true).is_none());
    assert!(c.lift(flat, false).is_some());
}

#[test]
fn fields_of_prime_three_modulo_four_and_one_modulo_eight() {
    // The square roots of the decoding, with `p - 1` divisible by a large power of two.
    let mut r = Rng(0x5DEE_CE66_D1CE_4E5B);
    for p in [
        "3B",
        "61",
        "101",
        "C1",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000000000001",
    ] {
        let p = hex(p);
        assert!(p.is_probable_prime());
        for _ in 0..size(3, 1) {
            let (curve, _, x) = random_curve(&p, &mut r);
            let u = curve.lift(&x, false).unwrap();
            assert_eq!(curve.lift(&x, true), Some(-&u));
            assert_eq!(curve.point_from_bytes(&u.compress()).unwrap(), u);
        }
    }
}

#[test]
fn invariants() {
    let p = n(97);
    let curve = Weierstrass::new(&n(2), &n(3), &p).unwrap();
    // -16 (32 + 243) = -4400, and 1728 32 / 275.
    assert_eq!(curve.discriminant(), n(-4400).rem_euclid(&p));
    let want = (n(1728 * 32) * n(275).mod_inverse(&p).unwrap()).rem_euclid(&p);
    assert_eq!(curve.j_invariant(), want);
    assert!(
        Weierstrass::new(&n(0), &n(5), &p)
            .unwrap()
            .j_invariant()
            .is_zero()
    );
    assert_eq!(
        Weierstrass::new(&n(5), &n(0), &p).unwrap().j_invariant(),
        n(1728).rem_euclid(&p)
    );
    assert_eq!(curve.a(), &n(2));
    assert_eq!(curve.b(), &n(3));
}

#[test]
fn many_products_at_once() {
    // Straus's method and Pippenger's, with zero, negative and repeated terms.
    let p = (Number::new_from_u8(1, ByteOrder::BigEndian) << 256usize) - 189u8;
    let mut r = Rng(0x7777_1234_ABCD_EF01);
    let (curve, _, x) = random_curve(&p, &mut r);
    let g = curve.lift(&x, false).unwrap();
    for len in [0usize, 1, 2, 3, 31, 32, 33, 100] {
        if cfg!(miri) && len > 33 {
            continue;
        }
        let scalars: Vec<Number> = (0..len)
            .map(|i| match i % 5 {
                0 => n(0),
                1 => -r.number(32),
                2 => r.number(4),
                _ => r.number(32),
            })
            .collect();
        let points: Vec<Point<'_, Weierstrass>> = (0..len)
            .map(|i| {
                if i % 7 == 3 {
                    g.clone()
                } else {
                    &g * &r.number(8)
                }
            })
            .collect();
        let terms: Vec<(&Number, &Point<'_, Weierstrass>)> =
            scalars.iter().zip(points.iter()).collect();
        let want = terms.iter().fold(curve.identity(), |s, (k, p)| s + *p * *k);
        assert_eq!(curve.multi_mul(&terms), want);
    }
}

#[test]
fn many_coordinates_at_once() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let p = curve.point(&n(3), &n(6)).unwrap();
    let points = [p.double(), curve.identity(), p.clone(), &p + &p.double()];
    let each: Vec<_> = points.iter().map(Point::coordinates).collect();
    assert_eq!(Point::coordinates_of(&points), each);
    assert!(Point::<Weierstrass>::coordinates_of(&[]).is_empty());
}

#[test]
fn text_and_hashes() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let p = curve.point(&n(3), &n(6)).unwrap();
    assert_eq!(p.to_string(), "(3, 6)");
    assert_eq!(format!("{p:?}"), "(3, 6)");
    assert_eq!(curve.identity().to_string(), "infinity");
    let hash = |q: &Point<'_, Weierstrass>| {
        let mut h = DefaultHasher::new();
        q.hash(&mut h);
        h.finish()
    };
    // The same point by two ways, in different projective forms.
    assert_eq!(hash(&(&p + &p)), hash(&p.double()));
    assert_eq!(hash(&(&p * &n(6))), hash(&p));
    let mut q = curve.identity();
    q.clone_from(&p);
    assert_eq!(q, p);
    assert!(core::ptr::eq(p.curve(), &curve));
}

#[test]
fn prepared_points() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let p = curve.point(&n(3), &n(6)).unwrap();
    for bits in [1, 4, 8, 64, 128, 200, 400] {
        let t = p.prepare(bits);
        for k in -40i64..40 {
            assert_eq!(t.mul(&n(k)), &p * &n(k));
        }
        // Longer than the table.
        assert_eq!(t.mul(&(n(1) << 500usize)), &p * &(n(1) << 500usize));
    }
    assert!(curve.identity().prepare(16).mul(&n(9)).is_identity());
}

#[test]
fn domains() {
    let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
    let d = Domain::new(curve, &n(3), &n(6), &n(5), &n(20)).unwrap();
    let g = d.generator();
    assert_eq!(d.order(), &n(5));
    assert_eq!(d.cofactor(), &n(20));
    for k in -12i64..12 {
        assert_eq!(d.mul_generator(&n(k)), &g * &n(k));
        assert_eq!(d.mul_add(&n(k), &n(3), &g), &g * &n(k + 3));
    }
    assert!(d.in_subgroup(&g));
    assert!(d.clear_cofactor(&g).is_identity());
    // A point outside the generator's subgroup.
    let outside = (0..97)
        .filter_map(|x| d.lift(&n(x), false))
        .find(|p| !d.in_subgroup(p))
        .unwrap();
    assert!(d.in_subgroup(&d.clear_cofactor(&outside)));

    let check = |x, y, order, e| {
        let c = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
        assert_eq!(
            Domain::new(c, &n(x), &n(y), &n(order), &n(1)).err(),
            Some(e)
        );
    };
    check(3, 7, 5, CurveError::NotOnCurve);
    check(3, 6, 4, CurveError::WrongOrder);
}
