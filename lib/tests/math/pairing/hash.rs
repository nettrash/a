//! Hashing to BLS12-381's `G1` and `G2`, RFC 9380 section 8.8: every vector of appendix J.9 and
//! J.10, the parameters, the searches for `Z`, the isogenies of appendix E.2 and E.3 shown to be
//! isogenies, the simplified SWU map over `F_p²` (in full on a small field), the cofactor
//! clearings against `h_eff` and appendix G.3, and the panics.

use super::common::{Rng, g1_from_x, g2_from_x, num};
use a::hash::{Sha256, expand_message_xmd};
use a::math::ec::hash::SimplifiedSwu;
use a::math::ec::{Curve, Weierstrass};
use a::math::field::{Fp, Fp2, PrimeField, Tower};
use a::math::numbers::{ByteOrder, Number};
use a::math::pairing::hash::{
    self, G1Suite, G2Suite, IsogenyMapFp2, SimplifiedSwuFp2, hash_to_field_fp2,
};
use a::math::pairing::{G1, G2, PairingCurve};
use a::math::poly::ModPolynomial;
use std::collections::HashMap;

/// Scales a test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Appendix J.9 and J.10, from `data/h2c_bls12381_vectors.txt` (see `data/gen_h2c_bls12381.py`).
const VECTORS: &str = include_str!("data/h2c_bls12381_vectors.txt");

/// One vector: its suite, tag, message and values, each one or two numbers.
struct Vector {
    suite: String,
    dst: String,
    msg: String,
    values: HashMap<String, Vec<Number>>,
}

/// The vectors of appendix J.9 and J.10.
fn vectors() -> Vec<Vector> {
    parse(VECTORS)
}

/// The vectors of a data file.
fn parse(text: &str) -> Vec<Vector> {
    let (mut suite, mut dst) = (String::new(), String::new());
    let mut out: Vec<Vector> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        match k {
            "suite" => suite = v.to_string(),
            "dst" => dst = v.to_string(),
            "msg" => out.push(Vector {
                suite: suite.clone(),
                dst: dst.clone(),
                msg: v.to_string(),
                values: HashMap::new(),
            }),
            _ => {
                let nums = v.split(' ').map(num).collect();
                out.last_mut().unwrap().values.insert(k.to_string(), nums);
            }
        }
    }
    out
}

/// gnark-crypto's vectors for BN254's `G1`, from `data/h2c_bn254_vectors.txt` (see
/// `data/gen_h2c_bn254.py`).
const BN254_VECTORS: &str = include_str!("data/h2c_bn254_vectors.txt");

/// The coordinates of a point of `G1` as numbers.
fn xy1(p: &G1<'_>) -> Vec<Number> {
    let (x, y) = p.coordinates().expect("an affine point");
    vec![x.value(), y.value()]
}

/// The coordinates of a point of `G2` as numbers, `x₀, x₁, y₀, y₁`.
fn xy2(q: &G2<'_>) -> Vec<Number> {
    let (x, y) = q.coordinates().expect("an affine point");
    vec![
        x.c0().value(),
        x.c1().value(),
        y.c0().value(),
        y.c1().value(),
    ]
}

/// The element of `F_p²` as its two numbers.
fn pair(u: &Fp2<'_>) -> Vec<Number> {
    vec![u.c0().value(), u.c1().value()]
}

/// The named point of a vector, its coordinates one after another.
fn point(v: &Vector, name: &str) -> Vec<Number> {
    let mut out = v.values[&format!("{name}.x")].clone();
    out.extend(v.values[&format!("{name}.y")].clone());
    out
}

#[test]
fn every_appendix_j_vector_of_bls12_381_is_reproduced() {
    let all = vectors();
    assert_eq!(all.len(), 20);
    let (g1, g2) = (hash::bls12_381_g1(), hash::bls12_381_g2());
    let mut counts = [0; 4];
    for v in &all {
        let (msg, dst) = (v.msg.as_bytes(), v.dst.as_bytes());
        let what = format!("{} {:.12}", v.suite, v.msg);
        let at = |k: &str| v.values[k].clone();
        if v.suite == g1.ro_id() || v.suite == g1.nu_id() {
            let ro = v.suite == g1.ro_id();
            let u = g1.hash_to_field(msg, dst, if ro { 2 } else { 1 });
            assert_eq!(vec![u[0].value()], at("u0"), "{what}");
            let p = if ro {
                assert_eq!(vec![u[1].value()], at("u1"), "{what}");
                let (q0, q1) = (g1.map_to_curve(&u[0]), g1.map_to_curve(&u[1]));
                assert_eq!(xy1(&q0), point(v, "Q0"), "{what}");
                assert_eq!(xy1(&q1), point(v, "Q1"), "{what}");
                let p = g1.hash_to_curve(msg, dst);
                assert_eq!(p, g1.clear_cofactor(&(q0 + q1)), "{what}");
                p
            } else {
                let q = g1.map_to_curve(&u[0]);
                assert_eq!(xy1(&q), point(v, "Q"), "{what}");
                let p = g1.encode_to_curve(msg, dst);
                assert_eq!(p, g1.clear_cofactor(&q), "{what}");
                p
            };
            assert_eq!(xy1(&p), point(v, "P"), "{what}");
            assert!(p.is_in_subgroup(), "{what}");
            assert!((&p * g1.curve().order()).is_identity(), "{what}");
            counts[usize::from(!ro)] += 1;
        } else {
            assert!(v.suite == g2.ro_id() || v.suite == g2.nu_id());
            let ro = v.suite == g2.ro_id();
            let u = g2.hash_to_field(msg, dst, if ro { 2 } else { 1 });
            assert_eq!(pair(&u[0]), at("u0"), "{what}");
            let p = if ro {
                assert_eq!(pair(&u[1]), at("u1"), "{what}");
                let (q0, q1) = (g2.map_to_curve(&u[0]), g2.map_to_curve(&u[1]));
                assert_eq!(xy2(&q0), point(v, "Q0"), "{what}");
                assert_eq!(xy2(&q1), point(v, "Q1"), "{what}");
                let p = g2.hash_to_curve(msg, dst);
                assert_eq!(p, g2.clear_cofactor(&(q0 + q1)), "{what}");
                p
            } else {
                let q = g2.map_to_curve(&u[0]);
                assert_eq!(xy2(&q), point(v, "Q"), "{what}");
                let p = g2.encode_to_curve(msg, dst);
                assert_eq!(p, g2.clear_cofactor(&q), "{what}");
                p
            };
            assert_eq!(xy2(&p), point(v, "P"), "{what}");
            assert!(p.is_in_subgroup(), "{what}");
            assert!((&p * g2.curve().order()).is_identity(), "{what}");
            counts[2 + usize::from(!ro)] += 1;
        }
    }
    assert_eq!(counts, [5; 4]);
}

#[test]
fn bn254s_g1_reproduces_gnark_cryptos_vectors() {
    // RFC 9380 has no suite for BN254; gnark-crypto's HashToG1 and EncodeToG1 (the
    // Shallue–van de Woestijne map, Z = 1) and their published vectors are the reference.
    let s = hash::bn254_g1();
    let all = parse(BN254_VECTORS);
    assert_eq!(all.len(), 10);
    let mut counts = [0; 2];
    for v in &all {
        let (msg, dst) = (v.msg.as_bytes(), v.dst.as_bytes());
        let what = format!("{} {:.12}", v.suite, v.msg);
        let at = |k: &str| v.values[k].clone();
        let ro = v.suite == s.ro_id();
        assert!(ro || v.suite == s.nu_id());
        let u = s.hash_to_field(msg, dst, if ro { 2 } else { 1 });
        assert_eq!(vec![u[0].value()], at("u0"), "{what}");
        let p = if ro {
            assert_eq!(vec![u[1].value()], at("u1"), "{what}");
            let (q0, q1) = (s.map_to_curve(&u[0]), s.map_to_curve(&u[1]));
            assert_eq!(xy1(&q0), point(v, "Q0"), "{what}");
            assert_eq!(xy1(&q1), point(v, "Q1"), "{what}");
            let p = s.hash_to_curve(msg, dst);
            assert_eq!(p, q0 + q1, "{what}");
            p
        } else {
            let q = s.map_to_curve(&u[0]);
            assert_eq!(xy1(&q), point(v, "Q"), "{what}");
            let p = s.encode_to_curve(msg, dst);
            assert_eq!(p, q, "{what}");
            p
        };
        assert_eq!(xy1(&p), point(v, "P"), "{what}");
        assert!(p.is_in_subgroup(), "{what}");
        counts[usize::from(!ro)] += 1;
    }
    assert_eq!(counts, [5, 5]);
}

#[test]
fn bn254s_g1_suite_has_its_parameters() {
    let s = hash::bn254_g1();
    let c = PairingCurve::bn254();
    assert!(core::ptr::eq(s.curve(), c));
    assert_eq!(
        (s.ro_id(), s.nu_id()),
        (
            "BN254G1_XMD:SHA-256_SVDW_RO_",
            "BN254G1_XMD:SHA-256_SVDW_NU_"
        )
    );
    assert_eq!((s.k(), s.l()), (128, 48));
    assert_eq!(s.h_eff(), &Number::from(1u8));
    assert_eq!(s.z(), &s.field().one());
    assert!(s.sswu().is_none() && s.isogeny().is_none());
    let m = s.svdw().unwrap();
    assert_eq!((m.a(), m.b()), (&s.field().zero(), &c.b()));
    // The Z of appendix H.1 for y² = x³ + 3.
    assert_eq!(
        a::math::ec::hash::ShallueVanDeWoestijne::find_z(&s.field().zero(), &c.b()),
        s.field().one()
    );
    // Random messages hash to points of G1, the whole of E(F_p).
    let mut rng = Rng(0x0254_0254_u64);
    for i in 0..size(4, 1) {
        let msg: Vec<u8> = (0..i * 13).map(|_| rng.next() as u8).collect();
        let p = s.hash_to_curve(&msg, b"A-TEST-V01-with-BN254G1_XMD:SHA-256_SVDW_RO_");
        assert!(p.is_in_subgroup() && !p.is_identity());
        assert_ne!(
            p,
            s.encode_to_curve(&msg, b"A-TEST-V01-with-BN254G1_XMD:SHA-256_SVDW_RO_")
        );
        assert_eq!(s.clear_cofactor(&p), p);
    }
}

/// `A'` and `B'` of the curve isogenous to BLS12-381's `E` (section 8.8.1).
fn g1_iso_curve() -> (Number, Number) {
    (
        num(
            "144698a3b8e9433d693a02c96d4982b0ea985383ee66a8d8e8981aefd881ac98936f8da0e0f97f5cf428082d584c1d",
        ),
        num(
            "12e2908d11688030018b12e8753eee3b2016c1f0f24f4070a0b9c14fcef35ef55a23215a316ceaa5d1cc48e98e172be0",
        ),
    )
}

/// `A' = 240·I` and `B' = 1012·(1 + I)` of the curve isogenous to the twist (section 8.8.2).
fn g2_iso_curve(t: &'static Tower<'static>) -> (Fp2<'static>, Fp2<'static>) {
    let n = |x: i64| Number::from(x);
    (
        Fp2::from_numbers(t, &n(0), &n(240)),
        Fp2::from_numbers(t, &n(1012), &n(1012)),
    )
}

#[test]
fn the_suites_have_the_parameters_of_section_8_8() {
    let c = PairingCurve::bls12_381();
    let g1 = hash::bls12_381_g1();
    assert_eq!(
        (g1.ro_id(), g1.nu_id()),
        (
            "BLS12381G1_XMD:SHA-256_SSWU_RO_",
            "BLS12381G1_XMD:SHA-256_SSWU_NU_"
        )
    );
    assert!(core::ptr::eq(g1.curve(), c));
    assert_eq!((g1.k(), g1.l()), (128, 64));
    assert_eq!(g1.h_eff(), &num("d201000000010001"));
    assert_eq!(g1.h_eff(), &(Number::from(1u8) - c.x()));
    assert_eq!(g1.z(), &g1.field().from_u64(11));
    let (a, b) = g1_iso_curve();
    assert_eq!(g1.sswu().unwrap().a().value(), a);
    assert_eq!(g1.sswu().unwrap().b().value(), b);
    assert_eq!(g1.isogeny().unwrap().degree(), 11);
    assert!(g1.svdw().is_none());

    let g2 = hash::bls12_381_g2();
    assert_eq!(
        (g2.ro_id(), g2.nu_id()),
        (
            "BLS12381G2_XMD:SHA-256_SSWU_RO_",
            "BLS12381G2_XMD:SHA-256_SSWU_NU_"
        )
    );
    assert!(core::ptr::eq(g2.curve(), c));
    assert!(core::ptr::eq(g2.tower(), c.tower()));
    assert_eq!((g2.k(), g2.l()), (128, 64));
    assert_eq!(
        g2.h_eff(),
        &num(
            "bc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe1329c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a359894c0adebbf6b4e8020005aaa95551"
        )
    );
    let t = g2.tower();
    assert_eq!(
        g2.z(),
        &Fp2::from_numbers(t, &Number::from(-2), &Number::from(-1))
    );
    let (a, b) = g2_iso_curve(t);
    assert_eq!((g2.sswu().a(), g2.sswu().b()), (&a, &b));
    assert_eq!(g2.isogeny().degree(), 3);
    // The tower's u is the RFC's I: u² = −1.
    assert_eq!(t.beta(), -t.base().one());
}

#[test]
fn the_searches_of_appendix_h_give_the_suites_z() {
    // Over F_p from 1, −1, 2, …: Z = 11 for G1's E'.
    let f = PrimeField::bls12_381_base();
    let (a, b) = g1_iso_curve();
    assert_eq!(
        SimplifiedSwu::find_z(&f.element(&a), &f.element(&b)),
        f.from_u64(11)
    );
    // Over F_p² from I, −I, 1 + I, −(1 + I), …: Z = −(2 + I) for G2's E'.
    let t = Tower::bls12_381();
    let (a, b) = g2_iso_curve(t);
    assert_eq!(
        SimplifiedSwuFp2::find_z(&a, &b),
        Fp2::from_numbers(t, &Number::from(-2), &Number::from(-1))
    );
}

#[test]
fn hash_to_field_into_fp2_reduces_pieces_of_l_bytes_two_at_a_time() {
    let s = hash::bls12_381_g2();
    let t = s.tower();
    let p = t.base().modulus();
    let dst = b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_RO_";
    for count in 0..4 {
        let bytes = expand_message_xmd::<Sha256>(b"m", dst, 128 * count);
        let e: Vec<Number> = bytes
            .chunks(64)
            .map(|c| Number::new(c.to_vec(), ByteOrder::BigEndian).rem_euclid(p))
            .collect();
        let got = s.hash_to_field(b"m", dst, count);
        assert_eq!(got.len(), count);
        for (i, u) in got.iter().enumerate() {
            assert_eq!(pair(u), vec![e[2 * i].clone(), e[2 * i + 1].clone()]);
        }
        assert_eq!(hash_to_field_fp2::<Sha256>(t, b"m", dst, count, 128), got);
    }
    // 63 elements are 8064 bytes; 64 would be 8192, beyond 255 blocks of SHA-256.
    assert_eq!(s.hash_to_field(b"m", b"D", 63).len(), 63);
    let r = std::panic::catch_unwind(|| s.hash_to_field(b"m", b"D", 64));
    let e = r.unwrap_err();
    assert!(e.downcast_ref::<String>().unwrap().contains("255 blocks"));
    let r = std::panic::catch_unwind(|| s.hash_to_field(b"m", b"D", usize::MAX));
    let e = r.unwrap_err();
    assert!(
        e.downcast_ref::<String>()
            .unwrap()
            .contains("too many to expand")
    );
    // G1's elements are 64 bytes of F_p each, 127 of them at most.
    let g1 = hash::bls12_381_g1();
    assert_eq!(g1.hash_to_field(b"m", b"D", 127).len(), 127);
    assert!(std::panic::catch_unwind(|| g1.hash_to_field(b"m", b"D", 128)).is_err());
}

/// The sum on `y² = x³ + a·x + b` over `F_p²`, in affine coordinates (`None` at infinity).
fn add2<'t>(
    p: &Option<(Fp2<'t>, Fp2<'t>)>,
    q: &Option<(Fp2<'t>, Fp2<'t>)>,
    a: &Fp2<'t>,
) -> Option<(Fp2<'t>, Fp2<'t>)> {
    let (Some((x1, y1)), Some((x2, y2))) = (p, q) else {
        return p.clone().or(q.clone());
    };
    let t = x1.tower();
    let lambda = if x1 == x2 {
        if (y1 + y2).is_zero() {
            return None;
        }
        let three = Fp2::from_numbers(t, &Number::from(3u8), &Number::default());
        &(&(&three * &x1.square()) + a) / &y1.double()
    } else {
        &(y2 - y1) / &(x2 - x1)
    };
    let x3 = &(&lambda.square() - x1) - x2;
    let y3 = &(&lambda * &(x1 - &x3)) - y1;
    Some((x3, y3))
}

#[test]
fn the_11_isogeny_maps_e_prime_onto_bls12_381_as_a_homomorphism() {
    // E' over F_p, as a curve of a::math::ec; its images are points of G1's curve.
    let s = hash::bls12_381_g1();
    let c = s.curve();
    let f = s.field();
    let (a, b) = g1_iso_curve();
    let ep = Weierstrass::new(&a, &b, f.modulus()).unwrap();
    let iso = s.isogeny().unwrap();
    let image = |p: &a::math::ec::Point<'_, Weierstrass>| -> G1<'static> {
        match p.coordinates() {
            None => c.g1_identity(),
            Some((x, y)) => match iso.map(&f.element(&x), &f.element(&y)) {
                None => c.g1_identity(),
                Some((x, y)) => G1::new_on_curve(c, x, y).expect("on BLS12-381"),
            },
        }
    };
    let mut rng = Rng(0x1111_0b15_u64);
    let mut points = Vec::new();
    while points.len() < size(5, 2) {
        let x = rng.fp(c).value();
        if let Some(p) = ep.lift(&x, rng.next() % 2 == 1) {
            points.push(p);
        }
    }
    for p in &points {
        for q in &points {
            assert_eq!(image(&(p + q)), image(p) + image(q));
        }
        assert_eq!(image(&p.double()), image(p).double());
        assert_eq!(image(&-p), -image(p));
        let k = rng.number(2);
        assert_eq!(image(&(p * &k)), &image(p) * &k);
    }
    // The kernel: x_den is the square of a polynomial of degree 5 (and y_den its cube), and its
    // roots in F_p, if any, go to the point at infinity.
    let coeffs = |c: &[Fp<'_>]| c.iter().map(Fp::value).collect::<Vec<_>>();
    let x_den = ModPolynomial::new(&coeffs(iso.x_den()), f.modulus());
    let y_den = ModPolynomial::new(&coeffs(iso.y_den()), f.modulus());
    let (_, factors) = x_den.factor();
    assert!(factors.iter().all(|(_, m)| *m == 2), "x_den is a square");
    let (_, factors) = y_den.factor();
    assert!(factors.iter().all(|(_, m)| *m == 3), "y_den is a cube");
    let roots = x_den.roots();
    assert_eq!(roots.len(), 5);
    for (r, _) in &roots {
        assert_eq!(iso.map(&f.element(r), &f.one()), None);
    }
}

/// The elements `u` that the simplified SWU map takes to a point with the abscissa `x0`: from
/// `x0 = x1(u)`, a quadratic in `t = Z·u²`, and from `x0 = Z·u²·x1(u)`, another.
fn sswu_preimages<'f>(map: &SimplifiedSwu<'f>, x0: &Fp<'f>) -> Vec<Fp<'f>> {
    let f = x0.field();
    let (a, b, z) = (map.a(), map.b(), map.z());
    let m = -(b / a);
    let two = f.from_u64(2);
    let mut ts = Vec::new();
    // x0 = m·(1 + 1/(t² + t)): t² + t − 1/c = 0 with c = x0/m − 1.
    let c = x0 / &m - f.one();
    if let Some(s) = c
        .inverse()
        .and_then(|i| (f.one() + f.from_u64(4) * &i).sqrt())
    {
        ts.push((-f.one() + &s) / &two);
        ts.push((-f.one() - &s) / &two);
    }
    // x0 = t·m·(1 + 1/(t² + t)) = m·(t + 1/(t + 1)): m·t² + (m − x0)·t + (m − x0) = 0.
    let (qa, qb) = (m.clone(), &m - x0);
    if let Some(s) = (&qb * &qb - f.from_u64(4) * &qa * &qb).sqrt() {
        ts.push((-&qb + &s) / &(&two * &qa));
        ts.push((-&qb - &s) / &(&two * &qa));
    }
    let mut out = Vec::new();
    for t in ts {
        if let Some(r) = (&t / z).sqrt() {
            for u in [r.clone(), -r] {
                if &map.map(&u).0 == x0 && !out.contains(&u) {
                    out.push(u);
                }
            }
        }
    }
    out
}

#[test]
fn g1s_map_reaches_the_kernel_of_the_11_isogeny_and_gives_the_identity() {
    // E'(F_p) has points of order 11 in the isogeny's kernel: x_den has five roots in F_p, and
    // the simplified SWU map reaches the points over two of them. Their u go to the point at
    // infinity, as section 6.6.3 requires, and the hash of such an element is cleared from it.
    let s = hash::bls12_381_g1();
    let f = s.field();
    let iso = s.isogeny().unwrap();
    let swu = s.sswu().unwrap();
    let coeffs: Vec<Number> = iso.x_den().iter().map(Fp::value).collect();
    let x_den = ModPolynomial::new(&coeffs, f.modulus());
    let mut reached = 0;
    for (r, _) in x_den.roots() {
        let x0 = f.element(&r);
        let g = (x0.square() + swu.a()) * &x0 + swu.b();
        assert_eq!(g.legendre(), 1, "a rational point of the kernel");
        for u in sswu_preimages(swu, &x0) {
            assert!(s.map_to_curve(&u).is_identity());
            let other = s.map_to_curve(&f.from_u64(7));
            assert_eq!(
                s.clear_cofactor(&(s.map_to_curve(&u) + &other)),
                s.clear_cofactor(&other)
            );
            reached += 1;
        }
    }
    assert_eq!(reached, 16);
}

#[test]
fn the_simplified_swu_map_over_fp2_follows_section_6_6_2() {
    // On BLS12-381's E' over F_p², at random elements: on the curve, x is x1 or x2 by the
    // definition, y has the sign of u, and −u gives (x, −y).
    let s = hash::bls12_381_g2();
    let c = s.curve();
    let t = s.tower();
    let map = s.sswu();
    let (a, b, z) = (map.a(), map.b(), map.z());
    let g = |x: &Fp2<'static>| &(&(&x.square() * x) + &(a * x)) + b;
    let mut rng = Rng(0x5550_f9f2_u64);
    for i in 0..size(40, 4) {
        let u = if i == 0 { Fp2::zero(t) } else { rng.fp2(c) };
        let (x, y) = map.map(&u);
        assert_eq!(y.square(), g(&x));
        let zu2 = z * &u.square();
        let den = &zu2.square() + &zu2;
        let x1 = match den.inverse() {
            None => b / &(z * a),
            Some(inv) => &(-(b / a)) * &(&Fp2::one(t) + &inv),
        };
        let want = if g(&x1).is_square() {
            x1.clone()
        } else {
            &zu2 * &x1
        };
        assert_eq!(x, want);
        assert_eq!(y.sgn0(), u.sgn0());
        if !u.is_zero() {
            assert_eq!(map.map(&-&u), (x, -&y));
        }
    }
    // −1 is a square of F_p², as every element of F_p is, so −1/Z is not, as Z is not: u = 0 is
    // the only input with Z²u⁴ + Z·u² = 0.
    assert!((-Fp2::one(t)).is_square());
    assert!(!(-(Fp2::one(t) / z)).is_square());
    {
        let u = Fp2::zero(t);
        let (x, y) = map.map(&u);
        assert_eq!(x, b / &(z * a));
        assert_eq!(y.sgn0(), u.sgn0());
        let q = s.map_to_curve(&u);
        let (ix, iy) = s.isogeny().map(&x, &y).unwrap();
        assert_eq!(q.coordinates(), Some((ix, iy)));
    }
}

#[test]
fn the_simplified_swu_map_over_fp2_follows_section_6_6_2_on_a_small_field_in_full() {
    // F_103² = F_103[u]/(u² + 1): every element u, two random curves, and Z checked against the
    // four criteria by brute force.
    let f = PrimeField::new(&Number::from(103u8));
    let t = Tower::new(
        &f,
        &Number::from(-1),
        (&Number::from(2u8), &Number::from(1u8)),
    );
    let t = &t;
    let all: Vec<Fp2<'_>> = (0..103u64)
        .flat_map(|c1| (0..103u64).map(move |c0| (c0, c1)))
        .map(|(c0, c1)| Fp2::new(t, f.from_u64(c0), f.from_u64(c1)))
        .collect();
    let squares: std::collections::HashSet<Vec<u64>> = all
        .iter()
        .map(|x| {
            let s = x.square();
            vec![
                s.c0().value().try_into().unwrap(),
                s.c1().value().try_into().unwrap(),
            ]
        })
        .collect();
    let key = |x: &Fp2<'_>| -> Vec<u64> {
        vec![
            x.c0().value().try_into().unwrap(),
            x.c1().value().try_into().unwrap(),
        ]
    };
    let mut rng = Rng(0x0103_0103_u64);
    let mut built = 0;
    while built < size(2, 1) {
        let a = Fp2::new(
            t,
            f.from_u64(rng.next() % 103),
            f.from_u64(rng.next() % 103),
        );
        let b = Fp2::new(
            t,
            f.from_u64(rng.next() % 103),
            f.from_u64(rng.next() % 103),
        );
        let four = Fp2::from_base(t, f.from_u64(4));
        let disc =
            &(&four * &(&a.square() * &a)) + &(&Fp2::from_base(t, f.from_u64(27)) * &b.square());
        if a.is_zero() || b.is_zero() || disc.is_zero() {
            continue;
        }
        // The criteria by brute force, for the first candidates of appendix H.2.
        let valid = |z: &Fp2<'_>| -> bool {
            !squares.contains(&key(z))
                && *z != -Fp2::one(t)
                && all.iter().all(|x| !(cubic2(x, &a, &b) - z).is_zero())
                && squares.contains(&key(&cubic2(&(&b / &(z * &a)), &a, &b)))
        };
        let mut expected = None;
        'search: for c0 in 0..103u64 {
            let c = Fp2::new(t, f.from_u64(c0), f.one());
            for z in [c.clone(), -c] {
                let ok = valid(&z);
                let r = std::panic::catch_unwind(|| {
                    let _ = SimplifiedSwuFp2::new(&a, &b, &z);
                });
                assert_eq!(r.is_ok(), ok, "Z = {z}");
                if ok {
                    expected = Some(z);
                    break 'search;
                }
            }
        }
        let z = expected.expect("a Z");
        assert_eq!(SimplifiedSwuFp2::find_z(&a, &b), z);
        let map = SimplifiedSwuFp2::new(&a, &b, &z);
        for u in &all {
            let (x, y) = map.map(u);
            assert_eq!(y.square(), cubic2(&x, &a, &b), "u = {u}");
            if !y.is_zero() {
                assert_eq!(y.sgn0(), u.sgn0(), "u = {u}");
            }
        }
        built += 1;
    }
}

#[test]
fn g1s_cofactor_clearing_is_the_product_by_h_eff() {
    let s = hash::bls12_381_g1();
    let c = s.curve();
    let mut rng = Rng(0x0001_cafe_u64);
    for _ in 0..size(4, 1) {
        let p = g1_from_x(c, rng.fp(c));
        let cleared = s.clear_cofactor(&p);
        assert_eq!(cleared, &p * s.h_eff());
        assert!(cleared.is_in_subgroup());
    }
}

#[test]
fn g2s_cofactor_clearing_is_the_product_by_h_eff_and_appendix_g3() {
    // Appendix G.3 with ψ from its constants c1 = 1/(1 + I)^((p − 1)/3), c2 = 1/(1 + I)^((p −
    // 1)/2), and ψ² from 1/2^((p − 1)/3), computed here; and the product by h_eff.
    let s = hash::bls12_381_g2();
    let c = s.curve();
    let t = s.tower();
    let p = t.base().modulus();
    let one_i = Fp2::from_numbers(t, &Number::from(1u8), &Number::from(1u8));
    let c1 = one_i.pow(&((p - 1u8) / 3u8)).inverse().unwrap();
    let c2 = one_i.pow(&((p - 1u8) / 2u8)).inverse().unwrap();
    let d1 = Fp2::from_base(t, t.base().from_u64(2))
        .pow(&((p - 1u8) / 3u8))
        .inverse()
        .unwrap();
    let psi = |q: &G2<'static>| match q.coordinates() {
        None => c.g2_identity(),
        Some((x, y)) => G2::new_on_curve(c, &c1 * &x.conjugate(), &c2 * &y.conjugate()).unwrap(),
    };
    let psi2 = |q: &G2<'static>| match q.coordinates() {
        None => c.g2_identity(),
        Some((x, y)) => G2::new_on_curve(c, &d1 * &x, -y).unwrap(),
    };
    let x = c.x();
    let mut rng = Rng(0x0002_cafe_u64);
    for _ in 0..size(3, 1) {
        let q = g2_from_x(c, rng.fp2(c));
        assert!(!q.is_in_subgroup());
        assert_eq!(psi(&q), q.psi());
        assert_eq!(psi2(&q), psi(&psi(&q)));
        // Appendix G.3, step by step.
        let t1 = &q * x;
        let t2 = psi(&q);
        let t3 = psi2(&q.double());
        let t3 = &t3 - &t2;
        let t2 = &(&t1 + &t2) * x;
        let t3 = &t3 + &t2;
        let t3 = &t3 - &t1;
        let want = &t3 - &q;
        let cleared = s.clear_cofactor(&q);
        assert_eq!(cleared, want);
        assert_eq!(cleared, &q * s.h_eff());
        assert!(cleared.is_in_subgroup());
    }
}

/// Checks the hashes of random messages: in the group, not the identity, repeated, changed by the
/// tag, `encode_to_curve` apart from `hash_to_curve`.
fn check_random_g1(s: &G1Suite, rng: &mut Rng, count: usize) {
    for i in 0..count {
        let msg: Vec<u8> = (0..i * 11).map(|_| rng.next() as u8).collect();
        let p = s.hash_to_curve(
            &msg,
            b"A-TEST-V01-CS01-with-BLS12381G1_XMD:SHA-256_SSWU_RO_",
        );
        assert!(p.is_in_subgroup() && !p.is_identity());
        assert_eq!(
            p,
            s.hash_to_curve(
                &msg,
                b"A-TEST-V01-CS01-with-BLS12381G1_XMD:SHA-256_SSWU_RO_"
            )
        );
        assert_ne!(p, s.hash_to_curve(&msg, b"ANOTHER-TAG"));
        let q = s.encode_to_curve(
            &msg,
            b"A-TEST-V01-CS01-with-BLS12381G1_XMD:SHA-256_SSWU_RO_",
        );
        assert!(q.is_in_subgroup());
        assert_ne!(p, q);
    }
}

/// As [`check_random_g1`] for `G2`.
fn check_random_g2(s: &G2Suite, rng: &mut Rng, count: usize) {
    for i in 0..count {
        let msg: Vec<u8> = (0..i * 11).map(|_| rng.next() as u8).collect();
        let p = s.hash_to_curve(
            &msg,
            b"A-TEST-V01-CS01-with-BLS12381G2_XMD:SHA-256_SSWU_RO_",
        );
        assert!(p.is_in_subgroup() && !p.is_identity());
        assert_eq!(
            p,
            s.hash_to_curve(
                &msg,
                b"A-TEST-V01-CS01-with-BLS12381G2_XMD:SHA-256_SSWU_RO_"
            )
        );
        assert_ne!(p, s.hash_to_curve(&msg, b"ANOTHER-TAG"));
        let q = s.encode_to_curve(
            &msg,
            b"A-TEST-V01-CS01-with-BLS12381G2_XMD:SHA-256_SSWU_RO_",
        );
        assert!(q.is_in_subgroup());
        assert_ne!(p, q);
    }
}

#[test]
fn random_messages_hash_into_g1_and_g2() {
    let mut rng = Rng(0xb15_0001_u64);
    check_random_g1(hash::bls12_381_g1(), &mut rng, size(4, 1));
    check_random_g2(hash::bls12_381_g2(), &mut rng, size(3, 1));
}

#[test]
fn hash_to_field_fp2_works_on_bn254s_tower() {
    // L = ceil((254 + 128) / 8) = 48 on BN254's F_p².
    let t = Tower::bn254();
    let p = t.base().modulus();
    let u = hash_to_field_fp2::<Sha256>(t, b"x", b"DST", 2, 128);
    let bytes = expand_message_xmd::<Sha256>(b"x", b"DST", 2 * 2 * 48);
    let e: Vec<Number> = bytes
        .chunks(48)
        .map(|c| Number::new(c.to_vec(), ByteOrder::BigEndian).rem_euclid(p))
        .collect();
    assert_eq!(pair(&u[0]), e[..2].to_vec());
    assert_eq!(pair(&u[1]), e[2..].to_vec());
}

/// `x³ + a·x + b` over `F_p²`.
fn cubic2<'t>(x: &Fp2<'t>, a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
    &(&(&x.square() * x) + &(a * x)) + b
}

/// BLS12-381's `E'` of `G2` and a few elements.
fn g2_parts() -> (Fp2<'static>, Fp2<'static>, &'static Tower<'static>) {
    let t = Tower::bls12_381();
    let (a, b) = g2_iso_curve(t);
    (a, b, t)
}

#[test]
#[should_panic(expected = "Z does not meet the criteria")]
fn a_square_z_is_refused_by_the_simplified_swu_map_over_fp2() {
    let (a, b, t) = g2_parts();
    let _ = SimplifiedSwuFp2::new(
        &a,
        &b,
        &Fp2::from_numbers(t, &Number::from(4u8), &Number::default()),
    );
}

#[test]
#[should_panic(expected = "Z does not meet the criteria")]
fn z_of_minus_one_is_refused_by_the_simplified_swu_map_over_fp2() {
    let (a, b, t) = g2_parts();
    let _ = SimplifiedSwuFp2::new(&a, &b, &-Fp2::one(t));
}

#[test]
#[should_panic(expected = "needs A ≠ 0 and B ≠ 0")]
fn a_zero_a_is_refused_by_the_simplified_swu_map_over_fp2() {
    let (_, b, t) = g2_parts();
    let _ = SimplifiedSwuFp2::new(&Fp2::zero(t), &b, &-Fp2::one(t));
}

#[test]
#[should_panic(expected = "needs A ≠ 0 and B ≠ 0")]
fn a_zero_b_is_refused_by_find_z_over_fp2() {
    let (a, _, t) = g2_parts();
    let _ = SimplifiedSwuFp2::find_z(&a, &Fp2::zero(t));
}

#[test]
#[should_panic(expected = "the curve is singular")]
fn a_singular_curve_is_refused_by_the_simplified_swu_map_over_fp2() {
    // 4·(−3)³ + 27·2² = 0.
    let t = Tower::bls12_381();
    let n = |x: i64| Fp2::from_numbers(t, &Number::from(x), &Number::default());
    let _ = SimplifiedSwuFp2::find_z(&n(-3), &n(2));
}

#[test]
#[should_panic(expected = "no coefficients")]
fn an_isogeny_map_over_fp2_needs_coefficients() {
    let t = Tower::bls12_381();
    let _ = IsogenyMapFp2::new(&[Fp2::one(t)], &[Fp2::one(t)], &[], &[Fp2::one(t)]);
}

#[test]
#[should_panic(expected = "leading coefficient of the map is 0")]
fn an_isogeny_map_over_fp2_needs_nonzero_leading_coefficients() {
    let t = Tower::bls12_381();
    let one = [Fp2::one(t)];
    let lead_zero = [Fp2::one(t), Fp2::zero(t)];
    let _ = IsogenyMapFp2::new(&one, &lead_zero, &one, &one);
}

#[test]
#[should_panic(expected = "elements of different towers")]
fn g2s_map_refuses_an_element_of_another_tower() {
    let u = Fp2::one(Tower::bn254());
    let _ = hash::bls12_381_g2().map_to_curve(&u);
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn g1s_map_refuses_an_element_of_another_field() {
    let u = PrimeField::bn254_base().from_u64(5);
    let _ = hash::bls12_381_g1().map_to_curve(&u);
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn g1s_clear_cofactor_refuses_a_point_of_bn254() {
    let _ = hash::bls12_381_g1().clear_cofactor(&PairingCurve::bn254().g1_generator());
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn g2s_clear_cofactor_refuses_a_point_of_bn254() {
    let _ = hash::bls12_381_g2().clear_cofactor(&PairingCurve::bn254().g2_generator());
}

#[test]
#[should_panic(expected = "the domain separation tag must not be empty")]
fn hashing_to_g2_refuses_an_empty_tag() {
    let _ = hash::bls12_381_g2().hash_to_curve(b"abc", b"");
}

#[test]
fn the_3_isogeny_maps_e_prime_onto_the_twist_as_a_homomorphism() {
    let s = hash::bls12_381_g2();
    let c = s.curve();
    let t = s.tower();
    let (a, b) = g2_iso_curve(t);
    let iso = s.isogeny();
    let image = |p: &Option<(Fp2<'static>, Fp2<'static>)>| -> G2<'static> {
        match p {
            None => c.g2_identity(),
            Some((x, y)) => match iso.map(x, y) {
                None => c.g2_identity(),
                Some((x, y)) => G2::new_on_curve(c, x, y).expect("on the twist"),
            },
        }
    };
    let mut rng = Rng(0x0003_0b15_u64);
    let mut points = Vec::new();
    while points.len() < size(5, 2) {
        let x = rng.fp2(c);
        if let Some(y) = (&(&(&x.square() * &x) + &(&a * &x)) + &b).sqrt() {
            points.push(Some((x, y)));
        }
    }
    for p in &points {
        for q in &points {
            assert_eq!(image(&add2(p, q, &a)), image(p) + image(q));
        }
        assert_eq!(image(&add2(p, p, &a)), image(p).double());
        let neg = p.as_ref().map(|(x, y)| (x.clone(), -y));
        assert_eq!(image(&neg), -image(p));
        assert_eq!(image(&add2(p, &neg, &a)), c.g2_identity());
    }
    // x_den = (x − x0)²: x0 goes to the point at infinity, and g'(x0) is not a square, so the
    // kernel's points are not over F_p² and no point the map reaches is exceptional.
    let k21 = Fp2::new(
        t,
        t.base().from_u64(12),
        t.base().element(&num(
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa9f",
        )),
    );
    let half = Fp2::from_base(t, t.base().from_u64(2).inverse().unwrap());
    let x0 = -(&k21 * &half);
    assert_eq!(iso.map(&x0, &Fp2::one(t)), None);
    let g = &(&(&x0.square() * &x0) + &(&a * &x0)) + &b;
    assert!(!g.is_square());
}
