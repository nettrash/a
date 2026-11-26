//! Curves over binary fields, `a::math::ec::binary`: on small fields against the textbook's
//! affine group law and every point found by search, on the 18 curves of SEC 2 against
//! OpenSSL 4.1's EC2M, from `data/binary_openssl.txt`, which `data/binary_gen.c` writes, and on
//! Koblitz curves over large fields, whose τ-adic forms take the widest windows, against PARI's
//! numbers of points (`data/binary_card641.gp`), the ladder and prepared tables.

use super::common::{Rng, size};
use a::math::ec::binary::named;
use a::math::ec::{BinaryCurve, BinaryDomain, BinaryPoint, CurveError, NotOnCurve};
use a::math::field::{BinaryField, Clmul, F2m};
use a::math::numbers::{ByteOrder, Number};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A point as the reference holds it: `None` at infinity.
type Affine<'f> = Option<(F2m<'f>, F2m<'f>)>;

/// The number `x`.
fn n(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::BigEndian)
}

/// The number in hexadecimal, most significant digit first.
fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::BigEndian).unwrap()
}

/// The bytes in hexadecimal.
fn bytes(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// The bytes in hexadecimal, upper case.
fn hex_of(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02X}")).collect()
}

/// The words of the hexadecimal number, the least significant first.
fn words(s: &str) -> Vec<u64> {
    let digits = s.trim_start_matches('0');
    if digits.is_empty() {
        return vec![0];
    }
    digits
        .as_bytes()
        .rchunks(16)
        .map(|c| u64::from_str_radix(core::str::from_utf8(c).unwrap(), 16).unwrap())
        .collect()
}

/// The element of `f` with the bits of `x`.
fn el(f: &BinaryField, x: u64) -> F2m<'_> {
    f.element(&[x])
}

/// The sum on `y² + x·y = x³ + a·x² + b` by the textbook's chord and tangent.
fn add_ref<'f>(a: &F2m<'f>, p: &Affine<'f>, q: &Affine<'f>) -> Affine<'f> {
    let (Some((x1, y1)), Some((x2, y2))) = (p, q) else {
        return p.clone().or(q.clone());
    };
    let lambda = if x1 == x2 {
        // The same point or its opposite (x, x + y); (0, √b) is its own.
        if y1 != y2 || x1.is_zero() {
            return None;
        }
        x1 + &(y1 / x1)
    } else {
        &(y1 + y2) / &(x1 + x2)
    };
    let x3 = &(&(&lambda.square() + &lambda) + &(x1 + x2)) + a;
    let y3 = &(&(&lambda * &(x1 + &x3)) + &x3) + y1;
    Some((x3, y3))
}

/// `k·p` by doublings and additions of the textbook's law, for any sign of `k`.
fn mul_ref<'f>(a: &F2m<'f>, p: &Affine<'f>, k: &Number) -> Affine<'f> {
    let mut r = None;
    for i in (0..k.bits()).rev() {
        r = add_ref(a, &r, &r);
        if k.abs().bit(i) {
            r = add_ref(a, &r, p);
        }
    }
    if k.is_negative() {
        r.map(|(x, y)| (x.clone(), &x + &y))
    } else {
        r
    }
}

/// The curve's point for the reference's.
fn on<'c>(curve: &'c BinaryCurve, p: &Affine<'_>) -> BinaryPoint<'c> {
    match p {
        Some((x, y)) => curve.point(x, y).unwrap(),
        None => curve.identity(),
    }
}

/// Every point of `curve`, by trying every `(x, y)`, the point at infinity first.
fn all_points(curve: &BinaryCurve) -> Vec<Affine<'_>> {
    let f = curve.field();
    let q = 1u64 << f.degree();
    let (a, b) = (curve.a(), curve.b());
    let mut points = vec![None];
    for x in 0..q {
        let x = el(f, x);
        let right = &(&(&x + &a) * &x.square()) + &b;
        for y in 0..q {
            let y = el(f, y);
            if &(&y + &x) * &y == right {
                points.push(Some((x.clone(), y)));
            }
        }
    }
    points
}

/// The small fields of the tests: `GF(2ᵐ)` for `m` from 1 to 8, odd and even.
fn small_fields() -> Vec<BinaryField> {
    [
        &[1, 0][..],
        &[2, 1, 0],
        &[3, 1, 0],
        &[4, 1, 0],
        &[5, 2, 0],
        &[6, 1, 0],
        &[7, 1, 0],
        &[8, 4, 3, 1, 0],
    ]
    .iter()
    .map(|e| BinaryField::new(e))
    .collect()
}

/// The curves of a field the tests take: both Koblitz curves, and others with `a` and `b` of
/// several kinds.
fn small_curves(f: &BinaryField) -> Vec<BinaryCurve> {
    let q = 1u64 << f.degree();
    let mut pairs = vec![(0, 1), (1, 1)];
    for (a, b) in [(0, 2), (1, 3), (2, 1), (3, 5), (5, 7), (6, 3)] {
        if a < q && b < q && !pairs.contains(&(a, b)) {
            pairs.push((a, b));
        }
    }
    pairs
        .into_iter()
        .map(|(a, b)| BinaryCurve::new(&el(f, a), &el(f, b)).unwrap())
        .collect()
}

#[test]
fn small_curves_have_exactly_the_points_of_their_equation() {
    for f in small_fields().into_iter().take(size(7, 3)) {
        for curve in small_curves(&f) {
            let points = all_points(&curve);
            let q = 1u64 << f.degree();
            let mut found = 0;
            for x in 0..q {
                for y in 0..q {
                    let (x, y) = (el(&f, x), el(&f, y));
                    let on_curve = points.contains(&Some((x.clone(), y.clone())));
                    assert_eq!(curve.point(&x, &y).is_ok(), on_curve);
                    if !on_curve {
                        assert_eq!(curve.point(&x, &y).unwrap_err(), NotOnCurve);
                    }
                    found += usize::from(on_curve);
                }
            }
            assert_eq!(found + 1, points.len());
            // Hasse's bound, and the order of every point divides the number of points: by the
            // textbook's law, by the ladder and by a prepared table, none of which reduces the
            // scalar. Not by `*`: on a Koblitz curve it reduces N modulo τᵐ − 1, to 0, and returns
            // the point at infinity without computing anything.
            let count = n(points.len() as i64);
            assert!((points.len() as i64 - q as i64 - 1).pow(2) <= 4 * q as i64);
            let a = curve.a();
            for p in &points {
                assert!(mul_ref(&a, p, &count).is_none());
                assert!(on(&curve, p).prepare(16).mul(&count).is_identity());
                if let Some((x, _)) = p {
                    assert_eq!(curve.x_mul(x, &count), None);
                }
            }
            assert_eq!(
                curve.is_koblitz(),
                curve.b().is_one() && curve.a().words()[0] < 2
            );
            if curve.is_koblitz() {
                assert_eq!(curve.koblitz_order(), Some(count));
            } else {
                assert_eq!(curve.koblitz_order(), None);
            }
        }
    }
}

#[test]
fn small_curves_add_double_and_negate_as_the_textbook_does() {
    for f in small_fields().into_iter().take(size(7, 4)) {
        for curve in small_curves(&f) {
            let a = curve.a();
            let points = all_points(&curve);
            // Pairs of all points up to GF(2⁴), and of the first 16 points beyond.
            let some = &points[..points.len().min(16)];
            for p in some {
                let pp = on(&curve, p);
                assert_eq!(pp.double(), on(&curve, &add_ref(&a, p, p)));
                let neg = p.as_ref().map(|(x, y)| (x.clone(), x + y));
                assert_eq!(-&pp, on(&curve, &neg));
                assert_eq!(-pp.clone(), on(&curve, &neg));
                assert_eq!(pp.coordinates(), p.clone());
                for q in some {
                    let qq = on(&curve, q);
                    let sum = on(&curve, &add_ref(&a, p, q));
                    assert_eq!(&pp + &qq, sum);
                    assert_eq!((&pp + &qq).coordinates(), add_ref(&a, p, q));
                    assert_eq!(&sum - &qq, pp);
                    // Projective inputs, with Z not 1: p + q held as `(p + q + q) - q`.
                    let projective = &(&pp + &qq.double()) - &qq;
                    assert_eq!(projective, sum);
                    assert_eq!(&projective - &qq, pp);
                    assert_eq!(&projective + &pp, &sum + &pp);
                    assert_eq!(projective.double(), sum.double());
                    assert_eq!(&projective - &pp + &projective, &pp + &qq + &qq);
                    let mut s = pp.clone();
                    s += &qq;
                    s -= qq.clone();
                    assert_eq!(s, pp);
                }
            }
        }
    }
}

#[test]
fn small_curves_multiply_by_every_scalar_in_every_way() {
    for f in small_fields().into_iter().take(size(8, 3)) {
        for curve in small_curves(&f) {
            let a = curve.a();
            let points = all_points(&curve);
            let count = points.len() as i64;
            let long = &(&n(count) << 90usize) + &n(5);
            // Every scalar from −N − 2 to N + 2 for the curves of up to 64 points, and beyond them
            // those near 0, ±N and ±2N, and a few others.
            let mut scalars: Vec<Number> = if count <= 64 {
                (-count - 2..=count + 2).map(n).collect()
            } else {
                let mut s: Vec<i64> = (-6..=6).collect();
                for c in [count, 2 * count] {
                    s.extend((-2..=2).flat_map(|d| [c + d, -c - d]));
                }
                s.extend((1..20).map(|i| (i * 7919 % (3 * count)) - count));
                s.into_iter().map(n).collect()
            };
            scalars.extend([long.clone(), -&long, &long * &long]);
            let points_taken = if f.degree() <= 5 { 12 } else { 5 };
            for p in points.iter().take(size(points_taken, 3)) {
                let pp = on(&curve, p);
                let prepared = pp.prepare(8);
                let x = p.as_ref().map(|(x, _)| x.clone());
                for (i, k) in scalars.iter().enumerate() {
                    // N·P = O for every point (the test above), so k·P = (k mod N)·P.
                    let want = on(&curve, &mul_ref(&a, p, &k.rem_euclid(&n(count))));
                    assert_eq!(&pp * k, want, "{k} times {pp}");
                    assert_eq!(curve.multi_mul(&[(k, &pp)]), want);
                    assert_eq!(prepared.mul(k), want);
                    if let Some(x) = &x {
                        let wx = want.coordinates().map(|c| c.0);
                        assert_eq!(curve.x_mul(x, k), wx, "{k} times x = {x:?}");
                    }
                    if i % 8 == 0 {
                        assert_eq!(k * &pp, want);
                        assert_eq!(pp.clone() * k.clone(), want);
                        let mut q = pp.clone();
                        q *= k;
                        assert_eq!(q, want);
                    }
                }
            }
        }
    }
}

#[test]
fn small_curves_sum_many_products_by_straus_and_by_pippenger() {
    let mut rng = Rng(0x5eed_b1a7);
    for f in small_fields().into_iter().skip(2).take(size(6, 2)) {
        for curve in small_curves(&f) {
            let points: Vec<BinaryPoint<'_>> =
                all_points(&curve).iter().map(|p| on(&curve, p)).collect();
            for terms in [0, 1, 2, 7, 63, 64, 70] {
                let chosen: Vec<(Number, BinaryPoint<'_>)> = (0..terms)
                    .map(|_| {
                        let k = n((rng.next() % 4000) as i64 - 2000);
                        (k, points[rng.next() as usize % points.len()].clone())
                    })
                    .collect();
                let refs: Vec<(&Number, &BinaryPoint<'_>)> =
                    chosen.iter().map(|(k, p)| (k, p)).collect();
                let want = chosen
                    .iter()
                    .fold(curve.identity(), |acc, (k, p)| acc + p * k);
                assert_eq!(curve.multi_mul(&refs), want, "{terms} terms");
            }
        }
    }
}

#[test]
fn small_curves_encode_and_lift_every_point() {
    for f in small_fields().into_iter().take(size(8, 3)) {
        let len = (f.degree() as usize).div_ceil(8);
        let curves = if f.degree() >= 7 { 3 } else { 8 };
        for curve in small_curves(&f).into_iter().take(curves) {
            let points = all_points(&curve);
            for p in &points {
                let pp = on(&curve, p);
                for bytes in [pp.compress(), pp.to_bytes(), pp.to_bytes_hybrid()] {
                    assert_eq!(curve.point_from_bytes(&bytes).unwrap(), pp);
                }
                match p {
                    None => {
                        assert_eq!(pp.compress(), [0]);
                        assert_eq!(pp.to_bytes(), [0]);
                    }
                    Some((x, y)) => {
                        let bit = !x.is_zero() && (y / x).words()[0] & 1 == 1;
                        let c = pp.compress();
                        assert_eq!((c[0], c.len()), (2 + u8::from(bit), 1 + len));
                        let h = pp.to_bytes_hybrid();
                        assert_eq!((h[0], &h[1..]), (6 + u8::from(bit), &pp.to_bytes()[1..]));
                        assert_eq!(pp.to_bytes()[0], 4);
                        assert_eq!(curve.lift(x, bit).unwrap(), pp);
                    }
                }
            }
            // Every x with each bit: a point exactly when one has that x, the other point of
            // that x with the other bit, and (0, √b) with either.
            for x in 0..1u64 << f.degree() {
                let x = el(&f, x);
                let with_x: Vec<_> = points.iter().flatten().filter(|(px, _)| *px == x).collect();
                for bit in [false, true] {
                    let lifted = curve.lift(&x, bit);
                    let mut encoding = vec![2 + u8::from(bit)];
                    encoding.extend((0..len).rev().map(|i| (x.words()[0] >> (8 * i)) as u8));
                    let decoded = curve.point_from_bytes(&encoding);
                    assert_eq!(lifted, decoded);
                    match lifted {
                        None => assert!(with_x.is_empty()),
                        Some(q) => {
                            let (qx, qy) = q.coordinates().unwrap();
                            assert_eq!(qx, x);
                            assert!(with_x.iter().any(|(_, y)| *y == qy));
                            if !x.is_zero() {
                                assert_eq!((&qy / &qx).words()[0] & 1 == 1, bit);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn small_curves_are_associative_on_every_triple() {
    let f = BinaryField::new(&[4, 1, 0]);
    for curve in [
        BinaryCurve::new(&el(&f, 3), &el(&f, 5)).unwrap(),
        BinaryCurve::new(&f.zero(), &f.one()).unwrap(),
    ] {
        let mut points: Vec<_> = all_points(&curve).iter().map(|p| on(&curve, p)).collect();
        points.truncate(size(16, 4));
        for p in &points {
            for q in &points {
                assert_eq!(p + q, q + p);
                for r in &points {
                    assert_eq!(&(p + q) + r, p + &(q + r));
                }
            }
        }
    }
}

#[test]
fn x_mul_takes_points_of_the_quadratic_twist() {
    // For an odd degree the trace of 1 is 1, and `a + 1` makes the twist.
    let fields = [BinaryField::new(&[5, 2, 0]), BinaryField::new(&[7, 1, 0])];
    for f in fields.iter().take(size(2, 1)) {
        for curve in small_curves(f).into_iter().take(size(8, 2)) {
            let twist = BinaryCurve::new(&(&curve.a() + &f.one()), &curve.b()).unwrap();
            let on_twist = all_points(&twist);
            let count = n(on_twist.len() as i64);
            for p in on_twist.iter().flatten() {
                assert!(p.0.is_zero() || curve.lift(&p.0, false).is_none());
                let tp = twist.point(&p.0, &p.1).unwrap();
                for k in [
                    n(1),
                    n(2),
                    n(3),
                    n(7),
                    n(-11),
                    count.clone(),
                    &count + &n(1),
                ] {
                    let want = (&tp * &k).coordinates().map(|c| c.0);
                    let got = curve.x_mul(&p.0, &k);
                    assert_eq!(
                        got.map(|x| x.words().to_vec()),
                        want.map(|x| x.words().to_vec())
                    );
                }
            }
        }
    }
}

#[test]
fn curves_and_domains_refuse_invalid_parameters() {
    let f = BinaryField::new(&[5, 2, 0]);
    assert_eq!(
        BinaryCurve::new(&f.one(), &f.zero()).unwrap_err(),
        CurveError::Singular
    );
    let curve = || BinaryCurve::new(&f.one(), &f.one()).unwrap();
    // (x³, x⁴ + x² + x + 1) has order 11 on this curve of 22 points.
    let (x, y) = (el(&f, 0b1000), el(&f, 0b10111));
    let d = BinaryDomain::new(curve(), &x, &y, &n(11), &n(2)).unwrap();
    assert_eq!((d.order(), d.cofactor()), (&n(11), &n(2)));
    assert_eq!(d.generator().coordinates(), Some((x.clone(), y.clone())));
    assert!(BinaryDomain::new(curve(), &x, &y, &n(22), &n(1)).is_ok());
    for wrong in [n(10), n(0), n(-11), n(1)] {
        assert_eq!(
            BinaryDomain::new(curve(), &x, &y, &wrong, &n(2)).err(),
            Some(CurveError::WrongOrder)
        );
    }
    assert_eq!(
        BinaryDomain::new(curve(), &x, &x, &n(11), &n(2)).err(),
        Some(CurveError::NotOnCurve)
    );
    // The domain's own operations.
    let g = d.generator();
    for k in -30..30 {
        let k = n(k);
        assert_eq!(d.mul_generator(&k), &g * &k);
        assert_eq!(d.mul_add(&k, &n(3), &g), &g * &(&k + &n(3)));
    }
    let t = d.point(&f.zero(), &f.one()).unwrap();
    assert!(!d.in_subgroup(&t));
    assert!(d.in_subgroup(&g) && d.in_subgroup(&d.identity()));
    assert!(d.clear_cofactor(&t).is_identity());
    let p6 = d.lift(&el(&f, 0b110), false).unwrap(); // of order 22
    assert!(!d.in_subgroup(&p6));
    assert!(d.in_subgroup(&d.clear_cofactor(&p6)));
    // The invariants, and the field.
    assert_eq!(d.discriminant(), f.one());
    assert_eq!(d.j_invariant(), f.one());
    let c = BinaryCurve::new(&el(&f, 3), &el(&f, 6)).unwrap();
    assert_eq!(&c.j_invariant() * &c.b(), f.one());
    assert_eq!(c.field(), &f);
    assert_eq!((c.a(), c.b()), (el(&f, 3), el(&f, 6)));
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_coordinate_of_another_field_panics() {
    let f = BinaryField::new(&[5, 2, 0]);
    let g = BinaryField::new(&[7, 1, 0]);
    let curve = BinaryCurve::new(&f.one(), &f.one()).unwrap();
    let _ = curve.point(&g.zero(), &g.one());
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn coefficients_of_two_fields_panic() {
    let f = BinaryField::new(&[5, 2, 0]);
    let g = BinaryField::new(&[5, 3, 0]);
    let _ = BinaryCurve::new(&f.one(), &g.one());
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn points_of_different_curves_panic() {
    let _ = named::sect113r1().generator() + named::sect113r2().generator();
}

#[test]
fn equal_curves_share_their_points() {
    let f = BinaryField::new(&[5, 2, 0]);
    let c1 = BinaryCurve::new(&f.one(), &f.one()).unwrap();
    let g = BinaryField::new(&[5, 2, 0]);
    let c2 = BinaryCurve::new(&g.one(), &g.one()).unwrap();
    let p = c1.lift(&el(&f, 0b110), true).unwrap();
    let q = c2.lift(&el(&g, 0b110), true).unwrap();
    assert_eq!(p, q);
    assert_eq!((&p + &q).coordinates(), p.double().coordinates());
    assert_eq!(c2.multi_mul(&[(&n(1), &q)]), p);
    // Hashes follow equality, whatever form the points are held in.
    let hash = |p: &BinaryPoint<'_>| {
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        h.finish()
    };
    let r = &(&p + &q.double()) - &q.double();
    assert_eq!(hash(&r), hash(&p));
    assert_ne!(hash(&c1.identity()), hash(&p));
    assert_eq!(format!("{}", c1.identity()), "infinity");
    let (x, y) = p.coordinates().unwrap();
    assert_eq!(format!("{p:?}"), format!("({x:?}, {y:?})"));
}

#[test]
fn coordinates_of_shares_one_inversion_among_points() {
    let curve = named::sect113r1();
    let g = curve.generator();
    let points: Vec<_> = (0..6).map(|k| &g * &n(k)).collect();
    let all = BinaryPoint::coordinates_of(&points);
    for (p, c) in points.iter().zip(&all) {
        assert_eq!(&p.coordinates(), c);
    }
    assert!(all[0].is_none());
    assert!(BinaryPoint::coordinates_of(&[]).is_empty());
}

// The named curves.

/// The data OpenSSL gave for the curve, the lines of its section.
fn section<'d>(data: &'d str, name: &str) -> Vec<Vec<&'d str>> {
    let mut lines = data.lines().map(|l| l.split(' ').collect::<Vec<_>>());
    let mut out = Vec::new();
    for l in lines.by_ref() {
        if l == ["curve", name] {
            break;
        }
    }
    for l in lines {
        if l[0] == "curve" {
            break;
        }
        out.push(l);
    }
    assert!(!out.is_empty(), "no data for {name}");
    out
}

/// The point of the encoding, which must be one.
fn point<'c>(curve: &'c BinaryDomain, s: &str) -> BinaryPoint<'c> {
    curve
        .point_from_bytes(&bytes(s))
        .unwrap_or_else(|| panic!("{s} is a point"))
}

/// Checks a curve's parameters, products, sums and encodings against OpenSSL's.
fn check_against_openssl(curve: &BinaryDomain, data: &str, name: &str) {
    let g = curve.generator();
    let (gx, gy) = g.coordinates().unwrap();
    let mut counts = [0; 8];
    for l in section(data, name) {
        match l[0] {
            "modulus" => {
                let e: Vec<u32> = l[1..]
                    .iter()
                    .map(|e| e.parse::<i32>().unwrap())
                    .filter(|&e| e >= 0)
                    .map(|e| e as u32)
                    .collect();
                assert_eq!(curve.field().modulus(), e);
            }
            "a" => assert_eq!(curve.a(), curve.field().element(&words(l[1]))),
            "b" => assert_eq!(curve.b(), curve.field().element(&words(l[1]))),
            "gx" => assert_eq!(gx, curve.field().element(&words(l[1]))),
            "gy" => assert_eq!(gy, curve.field().element(&words(l[1]))),
            "order" => assert_eq!(curve.order(), &hex(l[1])),
            "cofactor" => assert_eq!(curve.cofactor(), &hex(l[1])),
            "mulg" => {
                let (k, want) = (hex(l[1]), l[2]);
                assert_eq!(hex_of(&curve.mul_generator(&k).compress()), want);
                let p = &g * &k;
                assert_eq!(hex_of(&p.compress()), want, "{name}: {k} G");
                assert_eq!(curve.mul_generator(&-&k), -p);
                counts[0] += 1;
            }
            "mulp" => {
                let (p, k) = (point(curve, l[1]), hex(l[2]));
                let r = &p * &k;
                assert_eq!(hex_of(&r.compress()), l[3], "{name}: {k} P");
                // The ladder on the point of any order, once a curve.
                if counts[1] == 1 {
                    let (x, _) = p.coordinates().unwrap();
                    assert_eq!(curve.x_mul(&x, &k), r.coordinates().map(|c| c.0));
                }
                counts[1] += 1;
            }
            "add" => {
                let (p, q) = (point(curve, l[1]), point(curve, l[2]));
                let r = &p + &q;
                assert_eq!(hex_of(&r.compress()), l[3], "{name}: P + Q");
                assert_eq!(&r - &q, p);
                counts[2] += 1;
            }
            "enc" => {
                let u = bytes(l[1]);
                let p = curve.point_from_bytes(&u).unwrap();
                assert_eq!(p.to_bytes(), u);
                let len = (u.len() - 1) / 2;
                let mut compressed = bytes(l[2]);
                compressed.extend(&u[1..=len]);
                assert_eq!(p.compress(), compressed);
                let mut hybrid = bytes(l[3]);
                hybrid.extend(&u[1..]);
                assert_eq!(p.to_bytes_hybrid(), hybrid);
                assert_eq!(curve.point_from_bytes(&compressed).unwrap(), p);
                assert_eq!(curve.point_from_bytes(&hybrid).unwrap(), p);
                counts[3] += 1;
            }
            "check" => {
                let (u, verdicts) = (bytes(l[1]), l[2].as_bytes());
                let p = curve.point_from_bytes(&u).unwrap();
                let m = curve.field().degree() as usize;
                let len = m.div_ceil(8);
                let mut changes: Vec<Vec<u8>> = Vec::new();
                let mut c = u.clone();
                *c.last_mut().unwrap() ^= 1;
                changes.push(c);
                for at in [1, 1 + len] {
                    let mut c = u.clone();
                    c[at] |= 1 << (m % 8);
                    changes.push(c);
                }
                changes.push(u[..u.len() - 1].to_vec());
                let mut c = u.clone();
                c.push(0);
                changes.push(c);
                for tag in [1, 5, 8, 0xFF, 2, 3, 6, 7] {
                    let mut c = u.clone();
                    c[0] = tag;
                    changes.push(c);
                }
                for tag in [2, 3] {
                    let mut c = u[..=len].to_vec();
                    c[0] = tag;
                    changes.push(c);
                }
                assert_eq!(changes.len(), verdicts.len());
                for (c, &v) in changes.iter().zip(verdicts) {
                    let decoded = curve.point_from_bytes(c);
                    assert_eq!(decoded.is_some(), v == b'1', "{name}: {}", hex_of(c));
                    if let Some(q) = decoded {
                        assert!(q == p || q == -&p);
                    }
                }
                counts[4] += 1;
            }
            "nox" => {
                let x = bytes(l[1]);
                for (tag, &v) in [2u8, 3].iter().zip(l[2].as_bytes()) {
                    let mut c = vec![*tag];
                    c.extend(&x);
                    assert_eq!(curve.point_from_bytes(&c).is_some(), v == b'1');
                }
                counts[5] += 1;
            }
            "zero" => {
                let len = curve.field().degree().div_ceil(8) as usize;
                let t = curve
                    .point(&curve.field().zero(), &curve.b().sqrt())
                    .unwrap();
                let mut cases = Vec::new();
                for tag in [2u8, 3] {
                    let mut c = vec![0; 1 + len];
                    c[0] = tag;
                    cases.push(c);
                }
                cases.extend([vec![0], vec![0, 0], vec![]]);
                for tag in [6, 7] {
                    let mut c = t.to_bytes();
                    c[0] = tag;
                    cases.push(c);
                }
                for (c, &v) in cases.iter().zip(l[1].as_bytes()) {
                    let decoded = curve.point_from_bytes(c);
                    assert_eq!(decoded.is_some(), v == b'1', "{name}: {}", hex_of(c));
                    if let Some(q) = decoded {
                        assert!(q == t || q.is_identity());
                    }
                }
                assert!(t.double().is_identity());
                counts[6] += 1;
            }
            other => panic!("unknown line {other}"),
        }
    }
    assert!(counts[..7].iter().all(|&c| c > 0), "{name}: {counts:?}");
}

/// The data of `binary_gen.c 1`.
const OPENSSL: &str = include_str!("data/binary_openssl.txt");

/// A test of the named curve against OpenSSL's data for it; under Miri only sect113r1's.
macro_rules! against_openssl {
    ($($test:ident: $name:ident),* $(,)?) => {$(
        #[test]
        fn $test() {
            if cfg!(miri) && stringify!($name) != "sect113r1" {
                return;
            }
            check_against_openssl(named::$name(), OPENSSL, stringify!($name));
        }
    )*};
}

against_openssl!(
    sect113r1_matches_openssl: sect113r1,
    sect113r2_matches_openssl: sect113r2,
    sect131r1_matches_openssl: sect131r1,
    sect131r2_matches_openssl: sect131r2,
    sect163k1_matches_openssl: sect163k1,
    sect163r1_matches_openssl: sect163r1,
    sect163r2_matches_openssl: sect163r2,
    sect193r1_matches_openssl: sect193r1,
    sect193r2_matches_openssl: sect193r2,
    sect233k1_matches_openssl: sect233k1,
    sect233r1_matches_openssl: sect233r1,
    sect239k1_matches_openssl: sect239k1,
    sect283k1_matches_openssl: sect283k1,
    sect283r1_matches_openssl: sect283r1,
    sect409k1_matches_openssl: sect409k1,
    sect409r1_matches_openssl: sect409r1,
    sect571k1_matches_openssl: sect571k1,
    sect571r1_matches_openssl: sect571r1,
);

/// A larger sweep of `binary_gen.c`, run by hand: `binary_gen 12 > sweep.txt`, then
/// `A_BINARY_SWEEP=sweep.txt cargo test --release --test math -- --ignored ec::binary::`. It takes
/// minutes with a slow field.
#[test]
#[ignore]
fn named_curves_match_a_larger_openssl_sweep() {
    let path = std::env::var("A_BINARY_SWEEP").expect("A_BINARY_SWEEP names the sweep's file");
    let data = std::fs::read_to_string(path).unwrap();
    for name in named::NAMES {
        check_against_openssl(named::by_name(name).unwrap(), &data, name);
    }
}

#[test]
fn named_curves_have_the_number_of_points_their_cofactors_say() {
    for name in named::NAMES.iter().take(size(18, 1)) {
        let curve = named::by_name(name).unwrap();
        let (order, h) = (curve.order(), curve.cofactor());
        let m = curve.field().degree() as usize;
        assert!(order.is_probable_prime(), "{name}");
        assert_eq!(order.gcd(h), n(1));
        assert_eq!(name.ends_with("k1"), curve.is_koblitz(), "{name}");
        // The generator has the prime order n, so n divides the number of points N, which
        // Hasse's theorem puts within 2·2^(m/2) of 2^m + 1: with n > 4·2^(m/2) only one multiple
        // of n is there, and it must be N = h·n.
        let points = order * h;
        let q1 = &(&n(1) << m) + &n(1);
        let distance = (&points - &q1).abs();
        assert!(&distance * &distance <= &n(1) << (m + 2), "{name}");
        assert!(order * order > &n(16) << m, "{name}");
        assert!((&curve.generator() * order).is_identity());
        assert!(!curve.generator().is_identity());
        if curve.is_koblitz() {
            assert_eq!(curve.koblitz_order().unwrap(), points, "{name}");
        }
    }
}

#[test]
fn named_curves_have_their_nist_names() {
    if cfg!(miri) {
        assert!(core::ptr::eq(
            named::by_name("K-163").unwrap(),
            named::sect163k1()
        ));
        return;
    }
    let pairs = [
        ("K-163", named::sect163k1(), named::k163()),
        ("B-163", named::sect163r2(), named::b163()),
        ("K-233", named::sect233k1(), named::k233()),
        ("B-233", named::sect233r1(), named::b233()),
        ("K-283", named::sect283k1(), named::k283()),
        ("B-283", named::sect283r1(), named::b283()),
        ("K-409", named::sect409k1(), named::k409()),
        ("B-409", named::sect409r1(), named::b409()),
        ("K-571", named::sect571k1(), named::k571()),
        ("B-571", named::sect571r1(), named::b571()),
    ];
    for (name, sec, nist) in pairs {
        assert!(core::ptr::eq(sec, nist));
        assert!(core::ptr::eq(named::by_name(name).unwrap(), sec));
        assert!(core::ptr::eq(
            named::by_name(&name.to_lowercase()).unwrap(),
            sec
        ));
    }
    for name in named::NAMES {
        let curve = named::by_name(name).unwrap();
        assert!(core::ptr::eq(
            named::by_name(&name.to_uppercase()).unwrap(),
            curve
        ));
    }
    assert!(named::by_name("sect163k2").is_none());
    assert!(named::by_name("P-256").is_none());
    assert!(named::by_name("").is_none());
}

/// Checks the group's laws and the several ways to compute products on a named curve, with
/// seeded scalars and points.
fn group_laws(curve: &BinaryDomain, rounds: usize, seed: u64) {
    let mut rng = Rng(seed);
    let g = curve.generator();
    let order = curve.order();
    let len = curve.field().degree().div_ceil(8) as usize;
    // The point of order 2.
    let t = curve.lift(&curve.field().zero(), false).unwrap();
    let random_point = |rng: &mut Rng| loop {
        let x = rng.number(len) % &(&n(1) << curve.field().degree() as usize);
        let mut words = vec![0u64; len.div_ceil(8)];
        for i in 0..x.bits() {
            if x.bit(i) {
                words[i as usize / 64] |= 1 << (i % 64);
            }
        }
        if let Some(p) = curve.lift(&curve.field().element(&words), rng.next() & 1 == 1) {
            break p;
        }
    };
    for _ in 0..rounds {
        let (a, b) = (rng.below(order), rng.below(order));
        let (pa, pb) = (curve.mul_generator(&a), curve.mul_generator(&b));
        assert_eq!(curve.mul_generator(&(&a + &b)), &pa + &pb);
        assert_eq!(&pa * &b, &pb * &a);
        assert_eq!(curve.mul_add(&a, &b, &pb), &pa + &(&pb * &b));
        // A point of any order, the cofactor's part included.
        let p = random_point(&mut rng);
        let q = random_point(&mut rng);
        assert_eq!(&(&p + &q) + &pa, &p + &(&q + &pa));
        let (x, _) = p.coordinates().unwrap();
        let prepared = p.prepare(16 * len);
        // h·n·P = O by the ladder and by the table, which do not reduce the scalar: on a Koblitz
        // curve `*` reduces h·n, the number of points, to 0 modulo τᵐ − 1 and computes nothing.
        let points = order * curve.cofactor();
        assert_eq!(curve.x_mul(&x, &points), None);
        assert!(prepared.mul(&points).is_identity());
        // P is in the subgroup when n·P = O, by the ladder and the table; h·P always is, and
        // h·P + T, for T of order 2, never is.
        let in_subgroup = curve.x_mul(&x, order).is_none();
        assert_eq!(prepared.mul(order).is_identity(), in_subgroup);
        assert_eq!(curve.in_subgroup(&p), in_subgroup);
        let hp = curve.clear_cofactor(&p);
        let (hx, _) = hp.coordinates().unwrap();
        assert!(curve.in_subgroup(&hp) && curve.x_mul(&hx, order).is_none());
        assert!(!curve.in_subgroup(&(&hp + &t)));
        let k = rng.number(2 * len);
        let want = &p * &k;
        assert_eq!(curve.x_mul(&x, &k), want.coordinates().map(|c| c.0));
        assert_eq!(prepared.mul(&k), want);
        // (h·n + k)·P = k·P, the first by `*` (on a random curve the non-adjacent form of the longer
        // scalar), the second by the table.
        assert_eq!(&p * &(&points + &k), prepared.mul(&k));
        assert_eq!(curve.multi_mul(&[(&k, &p), (&a, &g)]), &want + &pa);
    }
    // Pippenger's method, from 64 terms, against Straus's.
    let terms: Vec<(Number, BinaryPoint<'_>)> = (0..64)
        .map(|i| (rng.number(8) - &n(1 << 62), &g * &n(i + 2)))
        .collect();
    let refs: Vec<(&Number, &BinaryPoint<'_>)> = terms.iter().map(|(k, p)| (k, p)).collect();
    let straus = curve.multi_mul(&refs[..32]) + curve.multi_mul(&refs[32..]);
    assert_eq!(curve.multi_mul(&refs), straus);
    assert!(t.double().is_identity() && !t.is_identity());
    assert!(!curve.in_subgroup(&t));
    assert_eq!(&t * order, t);
}

#[test]
fn named_curves_obey_the_group_laws() {
    let names = ["sect113r1", "sect131r2", "sect163k1", "sect163r2"];
    for name in names.iter().take(size(4, 1)) {
        group_laws(named::by_name(name).unwrap(), 1, 0xb1a5);
    }
}

/// The laws on every curve, with more rounds: minutes with a slow field.
#[test]
#[ignore]
fn named_curves_obey_the_group_laws_in_a_longer_sweep() {
    for (i, name) in named::NAMES.iter().enumerate() {
        group_laws(named::by_name(name).unwrap(), 8, 0x5eed + i as u64);
    }
}

// The widest windows: of the non-adjacent forms for scalars of 1,280 bits and more, and of the
// τ-adic forms for Koblitz curves whose numbers of points have 640 bits and more, which no
// named curve has.

/// A random point of `curve`, of any order, from a random `x` lifted.
fn any_point<'c>(curve: &'c BinaryCurve, rng: &mut Rng) -> BinaryPoint<'c> {
    let m = curve.field().degree() as usize;
    loop {
        let mut words: Vec<u64> = (0..m.div_ceil(64)).map(|_| rng.next()).collect();
        if !m.is_multiple_of(64) {
            *words.last_mut().unwrap() &= (1 << (m % 64)) - 1;
        }
        if let Some(p) = curve.lift(&curve.field().element(&words), rng.next() & 1 == 1) {
            return p;
        }
    }
}

/// A random number of exactly `bits` bits.
fn random_bits(rng: &mut Rng, bits: usize) -> Number {
    let top = &n(1) << (bits - 1);
    &top + &(rng.number(bits.div_ceil(8)) % &top)
}

#[test]
fn long_scalars_take_the_widest_non_adjacent_forms() {
    // Widths 7 from 1,280 bits and 8 from 3,072: on curves that are not Koblitz curves `*` takes
    // the non-adjacent form of the whole scalar, checked against the ladder and against `*` on the
    // scalar reduced modulo the number of points h·n, of width 4.
    let mut rng = Rng(0x5ca1_ab1e);
    for name in ["sect113r1", "sect163r2"] {
        let curve = named::by_name(name).unwrap();
        let points = curve.order() * curve.cofactor();
        let g = curve.generator();
        let p = any_point(curve, &mut rng);
        let (x, _) = p.coordinates().unwrap();
        for bits in [1279, 1280, 2000, 3071, 3072, 4000] {
            for k in [random_bits(&mut rng, bits), -random_bits(&mut rng, bits)] {
                let want = &p * &k.rem_euclid(&points);
                assert_eq!(&p * &k, want, "{name}, {bits} bits");
                assert_eq!(curve.x_mul(&x, &k), want.coordinates().map(|c| c.0));
            }
        }
        // Straus's method takes each term's own width.
        let (k1, k2) = (random_bits(&mut rng, 3500), random_bits(&mut rng, 1400));
        let want = &p * &k1.rem_euclid(&points) + &g * &k2.rem_euclid(curve.order());
        assert_eq!(curve.multi_mul(&[(&k1, &p), (&k2, &g)]), want);
    }
}

/// Fields for Koblitz curves with wide τ-adic windows: `x⁶⁴¹ + x¹¹ + 1`, `x¹²⁷⁹ + x²¹⁶ + 1`,
/// `x²²⁸¹ + x⁷¹⁵ + 1` and `x³²¹⁷ + x⁶⁷ + 1`, irreducible (PARI's `polisirreducible`). The numbers
/// of points have 641 or 642, 1,279 or 1,280, 2,281 or 2,282 and 3,217 or 3,218 bits, for `a = 1`
/// and `a = 0`, so the widths are 6, 6 or 7, 7 and 8.
const WIDE: [&[u32]; 4] = [
    &[641, 11, 0],
    &[1279, 216, 0],
    &[2281, 715, 0],
    &[3217, 67, 0],
];

/// The number of points of `y² + x·y = x³ + a·x² + 1` over `GF(2⁶⁴¹)`, for `a = 0` and `a = 1`, by
/// PARI's `ellcard` (`data/binary_card641.gp`).
const CARD641: [&str; 2] = [CARD641_A0, CARD641_A1];

/// See [`CARD641`].
const CARD641_A0: &str = concat!(
    "1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFFDE414FE5C2FEC031256BD5FF6A8FAACB75A2F7B40BF81D88",
    "3A1AB61E8BF13961CBCE67BFA046D825C",
);

/// See [`CARD641`].
const CARD641_A1: &str = concat!(
    "2000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000021BEB01A3D013FCEDA942A00957055348A5D084BF407E277",
    "C5E549E1740EC69E343198405FB927DA6",
);

/// Checks products on the Koblitz curve with the coefficient `a` over the field `e` against a
/// prepared table and the ladder, neither of which takes the τ-adic form: short scalars always,
/// long ones and a table as long as `m` with `long`.
fn wide_koblitz(e: &[u32], a: u64, long: bool, rng: &mut Rng) {
    let f = BinaryField::new(e);
    let curve = BinaryCurve::new(&f.element(&[a]), &f.one()).unwrap();
    let m = e[0] as usize;
    let count = curve.koblitz_order().unwrap();
    let bits = count.bits() as usize;
    assert!(bits == m || bits == m + 1);
    let distance = (&count - &(&(&n(1) << m) + &n(1))).abs();
    assert!(&distance * &distance <= &n(1) << (m + 2), "Hasse's bound");
    let p = any_point(&curve, rng);
    let (x, _) = p.coordinates().unwrap();
    // A short scalar has a short τ-adic form, but over the whole digit set of the width.
    let table = p.prepare(64);
    for k in [
        n(1),
        n(3),
        n(-7),
        random_bits(rng, 40),
        -random_bits(rng, 64),
    ] {
        assert_eq!(&p * &k, table.mul(&k), "m = {m}, a = {a}, k = {k}");
        let shifted = &(&count * &random_bits(rng, 20)) + &k;
        assert_eq!(&p * &shifted, table.mul(&k));
    }
    // The points of order 2 and, for a = 0, of order 4, with k = N − 1.
    let t = curve.lift(&f.zero(), false).unwrap();
    assert_eq!(&t * &(&count - &n(1)), t);
    if a == 0 {
        let u = curve.point(&f.one(), &f.zero()).unwrap();
        assert_eq!(u.double(), t);
        assert_eq!(&u * &(&count - &n(1)), -&u);
        assert_eq!(&u * &n(-5), -&u);
    }
    if long {
        // N·P = O by the ladder, which does not reduce the scalar as `*` does.
        assert_eq!(curve.x_mul(&x, &count), None);
        for bits in [m / 2, m, 3 * m] {
            let k = random_bits(rng, bits);
            let want = &p * &k;
            assert_eq!(curve.x_mul(&x, &k), want.coordinates().map(|c| c.0));
            assert_eq!(&p * &-&k, -&want);
        }
        let k = random_bits(rng, m);
        assert_eq!(p.prepare(m).mul(&k), &p * &k);
    }
}

#[test]
fn koblitz_curves_over_large_fields_take_wide_tau_adic_windows() {
    let mut rng = Rng(0x7a0_7a0);
    // Width 6, with PARI's numbers of points.
    for a in [0, 1] {
        let f = BinaryField::new(WIDE[0]);
        let curve = BinaryCurve::new(&f.element(&[a]), &f.one()).unwrap();
        assert_eq!(curve.koblitz_order().unwrap(), hex(CARD641[a as usize]));
        wide_koblitz(WIDE[0], a, true, &mut rng);
    }
    // Widths 7 and 8, with short scalars: the long ones in the sweep below.
    wide_koblitz(WIDE[1], 0, false, &mut rng);
    wide_koblitz(WIDE[3], 0, false, &mut rng);
}

/// All four fields, both curves, long scalars: about a minute in a debug build, so ignored.
#[test]
#[ignore]
fn koblitz_curves_over_large_fields_in_a_longer_sweep() {
    let mut rng = Rng(0x05ee_d7a0);
    for e in WIDE {
        for a in [0, 1] {
            wide_koblitz(e, a, true, &mut rng);
        }
    }
}

#[test]
fn koblitz_curves_with_cofactor_four_multiply_their_points_of_order_four() {
    // On y² + x·y = x³ + 1, (1, 0) has order 4: 2·(1, 0) = (0, 1), and −(1, 0) = (1, 1). OpenSSL
    // 4.1's EC_POINT_mul gives (1, 0) for (#E − 1)·(1, 0) on sect283k1, where −(1, 0) is right.
    for name in [
        "sect233k1",
        "sect239k1",
        "sect283k1",
        "sect409k1",
        "sect571k1",
    ] {
        let curve = named::by_name(name).unwrap();
        let f = curve.field();
        let u = curve.point(&f.one(), &f.zero()).unwrap();
        let minus = curve.point(&f.one(), &f.one()).unwrap();
        let t = curve.point(&f.zero(), &f.one()).unwrap();
        let count = curve.koblitz_order().unwrap();
        assert_eq!(curve.cofactor(), &n(4));
        // n is odd: n·(1, 0) is (1, 0) or its opposite, by n modulo 4.
        let order = curve.order().clone();
        let by_order = if order.rem_euclid(&n(4)) == n(1) {
            &u
        } else {
            &minus
        };
        for (k, want) in [
            (&count - &n(1), &minus),
            (count.clone() + n(2), &t),
            (&count * &n(3) + n(5), &u),
            (order, by_order),
        ] {
            assert_eq!(&u * &k, *want, "{name}: {k}");
            assert_eq!(curve.x_mul(&f.one(), &k), want.coordinates().map(|c| c.0));
            assert_eq!(u.prepare(k.bits() as usize).mul(&k), *want);
        }
        assert!((&u * &count).is_identity() && curve.x_mul(&f.one(), &count).is_none());
        assert_eq!(&u * &n(4), curve.identity());
    }
}

/// Moduli of at most 256 bits, whose products by scalars run on inline words, with the second
/// term close to `xᵐ` (many small chunks for the sparse reduction, so that Barrett's takes them
/// under the instruction), 8 bits below it (Barrett's under the instruction, the sparse one in
/// chunks of 8 bits under the portable kernel), many terms, and the standards' pentanomial of
/// `sect163k1` (the sparse one in chunks of 64 bits under the portable kernel and `PCLMULQDQ`,
/// Barrett's under `PMULL`) and trinomial of `sect233k1` (the sparse one under every kernel): of 1
/// to 4 words, `m` a multiple of 64 among them.
const NARROW: [&[u32]; 10] = [
    &[61, 59, 9, 2, 0],
    &[127, 126, 0],
    &[153, 152, 0],
    &[233, 232, 208, 207, 0],
    &[256, 255, 58, 16, 0],
    &[233, 225, 96, 12, 0],
    &[256, 248, 214, 81, 0],
    &[
        127, 62, 61, 60, 58, 57, 55, 54, 52, 51, 49, 48, 47, 45, 44, 43, 42, 41, 40, 39, 36, 32,
        31, 26, 24, 23, 22, 21, 20, 19, 17, 10, 9, 8, 7, 6, 5, 3, 2, 1, 0,
    ],
    &[163, 7, 6, 3, 0],
    &[233, 74, 0],
];

#[test]
fn fields_of_every_reduction_multiply_as_the_textbook_does() {
    // Each field under the CPU's kernel and the portable one (whose squares spread the bits and
    // whose costs choose the reductions otherwise), on both Koblitz curves (τ-adic products) and a
    // random one: k·P, x_mul and multi_mul against the textbook's affine law.
    let mut rng = Rng(0x0b1a_5ed0);
    for e in NARROW.iter().take(size(10, 2)) {
        let m = e[0] as usize;
        for f in [
            BinaryField::new(e),
            BinaryField::with_clmul(e, Clmul::Portable),
        ] {
            let random: Vec<u64> = (0..m.div_ceil(64)).map(|_| rng.next()).collect();
            let coefficients = [
                (f.zero(), f.one()),
                (f.one(), f.one()),
                (f.element(&random), f.element(&[0x1234_5677, 9])),
            ];
            for (a, b) in coefficients.iter().take(size(3, 1)) {
                let curve = BinaryCurve::new(a, b).unwrap();
                let p = any_point(&curve, &mut rng);
                let q = any_point(&curve, &mut rng);
                let (pa, qa) = (p.coordinates(), q.coordinates());
                let (x, _) = pa.clone().unwrap();
                let k = random_bits(&mut rng, m);
                let l = random_bits(&mut rng, m + 5);
                for k in [k.clone(), -&k, n(7)].iter().take(size(3, 1)) {
                    let want = mul_ref(a, &pa, k);
                    assert_eq!(
                        &p * k,
                        on(&curve, &want),
                        "{e:?}, {:?}, {k} times",
                        f.clmul()
                    );
                    assert_eq!(curve.x_mul(&x, k), want.map(|c| c.0), "{e:?}, x_mul");
                }
                let want = add_ref(a, &mul_ref(a, &pa, &k), &mul_ref(a, &qa, &l));
                assert_eq!(
                    curve.multi_mul(&[(&k, &p), (&l, &q)]),
                    on(&curve, &want),
                    "{e:?}"
                );
            }
        }
    }
}

/// Four products `k·P` on a curve over the field of the exponents `e`, as bytes.
fn products_over(e: &[u32]) -> Vec<Vec<u8>> {
    let f = BinaryField::new(e);
    let c = BinaryCurve::new(&f.one(), &f.element(&[0x1234_5677, 9, 77, 1])).unwrap();
    let k = Number::from(0xffff_ffff_ffff_fff1u64);
    let mut out = Vec::new();
    let mut x = 5u64;
    while out.len() < 4 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        if let Some(p) = c.lift(&f.element(&[x, x >> 3, x ^ 99, x >> 40]), false) {
            out.push((&p * &k).to_bytes());
        }
    }
    out
}

/// The narrow arithmetic keeps the reduction it chose for the last field of a thread (the sparse
/// one or Barrett's), keyed by the degree, the other exponents and the kernel: two fields of one
/// degree taken in turn on one thread must each get their own, as on a fresh thread.
#[test]
fn two_fields_of_one_degree_in_turn_on_one_thread_keep_their_own_reductions() {
    const SPARSE: &[u32] = &[233, 74, 0];
    const BARRETT: &[u32] = &[233, 232, 208, 207, 0];
    let fresh = |e: &'static [u32]| std::thread::spawn(move || products_over(e)).join().unwrap();
    let (a, b) = (fresh(SPARSE), fresh(BARRETT));
    for _ in 0..2 {
        assert_eq!(
            products_over(SPARSE),
            a,
            "x²³³ + x⁷⁴ + 1 after the other field"
        );
        assert_eq!(
            products_over(BARRETT),
            b,
            "x²³³ + x²³² + x²⁰⁸ + x²⁰⁷ + 1 after the other field"
        );
    }
}
