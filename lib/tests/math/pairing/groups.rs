//! `G1` and `G2`: the group law, the parameters, the subgroup checks against arkworks and against
//! the order, with points of every prime order that divides the cofactors, the cofactor clearings
//! and the endomorphism `ψ`.

use super::common::{Fnv, Rng, curves, g1_from_x, g2_from_x, num, reference};
use a::math::field::Fp2;
use a::math::numbers::Number;
use a::math::pairing::{Family, G1, G2, PairingCurve, PointError, Twist};

#[test]
fn the_curves_have_their_published_parameters() {
    let bn = PairingCurve::bn254();
    assert_eq!(bn.name(), "BN254");
    assert_eq!((bn.family(), bn.twist()), (Family::Bn, Twist::D));
    assert_eq!(bn.x(), &Number::from(4_965_661_367_192_848_881_u64));
    assert_eq!(
        bn.base_field().modulus(),
        &num("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47")
    );
    assert_eq!(
        bn.order(),
        &num("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001")
    );
    assert_eq!(bn.g1_cofactor(), &Number::from(1u8));
    assert_eq!(
        bn.g2_cofactor(),
        &num("30644e72e131a029b85045b68181585e06ceecda572a2489345f2299c0f9fa8d")
    );
    assert_eq!(bn.b().value(), Number::from(3u8));
    // b' = 3/(9 + u), as arkworks has it.
    assert_eq!(
        bn.twist_b(),
        Fp2::from_numbers(
            bn.tower(),
            &Number::from_str_radix(
                "19485874751759354771024239261021720505790618469301721065564631296452457478373",
                10,
                a::math::numbers::ByteOrder::BigEndian
            )
            .unwrap(),
            &Number::from_str_radix(
                "266929791119991161246907387137283842545076965332900288569378510910307636690",
                10,
                a::math::numbers::ByteOrder::BigEndian
            )
            .unwrap()
        )
    );
    assert_eq!(bn.miller_loop_scalar(), &(bn.x() * 6u8 + 2u8));

    let bls = PairingCurve::bls12_381();
    assert_eq!(bls.name(), "BLS12-381");
    assert_eq!((bls.family(), bls.twist()), (Family::Bls12, Twist::M));
    assert_eq!(bls.x(), &-num("d201000000010000"));
    assert_eq!(bls.g1_cofactor(), &num("396c8c005555e1568c00aaab0000aaab"));
    assert_eq!(
        bls.g2_cofactor(),
        &num(
            "5d543a95414e7f1091d50792876a202cd91de4547085abaa68a205b2e5a7ddfa628f1cb4d9e82ef21537e293a6691ae1616ec6e786f0c70cf1c38e31c7238e5"
        )
    );
    assert_eq!(bls.b().value(), Number::from(4u8));
    assert_eq!(
        bls.twist_b(),
        Fp2::from_numbers(bls.tower(), &Number::from(4u8), &Number::from(4u8))
    );
    assert_eq!(bls.miller_loop_scalar(), bls.x());
    assert_eq!(format!("{bls:?}"), "BLS12-381");
    // The orders of the groups: #E(F_p) = h₁ r = p + 1 − t with the trace t = 6x² + 1 (BN) or
    // x + 1 (BLS12), and #E'(F_p²) = h₂ r, which random points of the curves are killed by.
    for (_, c) in curves() {
        let p = c.base_field().modulus();
        let x = c.x();
        let t = match c.family() {
            Family::Bn => x * x * 6u8 + 1u8,
            Family::Bls12 => x + 1u8,
        };
        assert_eq!(c.g1_cofactor() * c.order(), p + 1u8 - &t);
        let mut r = Rng(17);
        let s = g1_from_x(c, r.fp(c));
        assert!((&s * &(c.g1_cofactor() * c.order())).is_identity());
        let s = g2_from_x(c, r.fp2(c));
        assert!((&s * &(c.g2_cofactor() * c.order())).is_identity());
        assert!(!(&s * c.order()).is_identity());
    }
}

#[test]
fn the_group_law_holds_on_g1_and_g2() {
    for (_, c) in curves() {
        let mut r = Rng(11);
        let g1 = c.g1_generator();
        let g2 = c.g2_generator();
        let (a, b) = (r.number(2), r.number(2));
        let (p, q) = (&g1 * &a, &g1 * &b);
        let (s, t) = (&g2 * &a, &g2 * &b);
        let o1 = c.g1_identity();
        let o2 = c.g2_identity();
        // Identity, negation, commutativity, associativity, distributivity.
        assert_eq!(&p + &o1, p);
        assert_eq!(&o2 + &s, s);
        assert!((&p - &p).is_identity());
        assert!((&s + &-&s).is_identity());
        assert_eq!(&p + &q, &q + &p);
        assert_eq!(&s + &t, &t + &s);
        assert_eq!(&(&p + &q) + &g1, &p + &(&q + &g1));
        assert_eq!(&(&s + &t) + &g2, &s + &(&t + &g2));
        assert_eq!(&g1 * &(&a + &b), &p + &q);
        assert_eq!(&g2 * &(&a + &b), &s + &t);
        assert_eq!(&p * &b, &q * &a);
        assert_eq!(p.double(), &p + &p);
        assert_eq!(s.double(), &s + &s);
        assert!(o1.double().is_identity());
        assert!(o2.coordinates().is_none());
        // Products by small, negative, zero and large scalars.
        let mut acc1 = c.g1_identity();
        let mut acc2 = c.g2_identity();
        for k in 0..20u32 {
            assert_eq!(&g1 * &Number::from(k), acc1, "k = {k}");
            assert_eq!(&g2 * &Number::from(k), acc2, "k = {k}");
            assert_eq!(&g1 * &-Number::from(k), -&acc1, "k = {k}");
            acc1 += &g1;
            acc2 += &g2;
        }
        assert!((&g1 * c.order()).is_identity());
        assert!((&g2 * c.order()).is_identity());
        assert_eq!(&g1 * &(c.order() + 1u8), g1);
        assert_eq!(&g2 * &(c.order() - 1u8), -&g2);
        let huge = r.number(40); // 2560 bits
        assert_eq!(&g1 * &huge, &g1 * &huge.rem_euclid(c.order()));
        assert_eq!(&g2 * &huge, &g2 * &huge.rem_euclid(c.order()));
        assert!((&o1 * &huge).is_identity());
        // The operator forms and assignments.
        let mut x = p.clone();
        x -= &q;
        x += q.clone();
        x *= &b;
        assert_eq!(x, &a * &(&g1 * &b));
        let mut y = s.clone();
        y -= t.clone();
        y *= Number::from(2u8);
        assert_eq!(y, Number::from(2u8) * (&s - &t));
        // Equality across projective forms, and hashes by the affine point.
        use std::collections::HashSet;
        let twice = p.double();
        let sum = &p + &p;
        #[allow(
            clippy::mutable_key_type,
            reason = "the field may hold a cache filled once (`OnceLock`); the hash reads only the value"
        )]
        let set: HashSet<G1<'_>> = [twice, sum, q.clone()].into_iter().collect();
        assert_eq!(set.len(), 2);
        assert_eq!(o1.to_string(), "infinity");
        assert!(g1.to_string().starts_with('('));
    }
}

#[test]
fn subgroup_checks_and_cofactor_clearings_match_arkworks() {
    let cases = if cfg!(miri) { 1 } else { 8 };
    for (name, c) in curves() {
        let mut r = Rng(3);
        let (mut s1, mut c1, mut m1, mut s2, mut c2, mut m2) = (
            Fnv::new(),
            Fnv::new(),
            Fnv::new(),
            Fnv::new(),
            Fnv::new(),
            Fnv::new(),
        );
        for _ in 0..cases {
            let x = r.fp(c);
            let p = g1_from_x(c, x);
            let x = r.fp2(c);
            let q = g2_from_x(c, x);
            let (pi, qi) = (p.is_in_subgroup(), q.is_in_subgroup());
            // The subgroup checks by endomorphisms against the order.
            assert_eq!(pi, (&p * c.order()).is_identity(), "{name}");
            assert_eq!(qi, (&q * c.order()).is_identity(), "{name}");
            s1.eat(&[u8::from(pi)]);
            s2.eat(&[u8::from(qi)]);
            // Every clearing lands in the subgroup.
            for x in [p.clear_cofactor(), p.mul_by_cofactor()] {
                assert!(
                    x.is_in_subgroup() && (&x * c.order()).is_identity(),
                    "{name}"
                );
            }
            for x in [q.clear_cofactor(), q.mul_by_cofactor()] {
                assert!(
                    x.is_in_subgroup() && (&x * c.order()).is_identity(),
                    "{name}"
                );
            }
            match c.family() {
                Family::Bls12 => {
                    c1.eat(&p.clear_cofactor().to_zcash_compressed());
                    m1.eat(&p.mul_by_cofactor().to_zcash_compressed());
                    c2.eat(&q.clear_cofactor().to_zcash_compressed());
                    m2.eat(&q.mul_by_cofactor().to_zcash_compressed());
                }
                Family::Bn => m2.eat(&q.mul_by_cofactor().to_ethereum()),
            }
        }
        if cfg!(miri) {
            continue;
        }
        assert_eq!(s1.0, reference(&format!("{name}_g1_subgroup")), "{name}");
        assert_eq!(s2.0, reference(&format!("{name}_g2_subgroup")), "{name}");
        assert_eq!(
            m2.0,
            reference(&format!("{name}_g2_mul_by_cofactor")),
            "{name}"
        );
        if c.family() == Family::Bls12 {
            assert_eq!(c1.0, reference("bls_g1_clear_cofactor"));
            assert_eq!(m1.0, reference("bls_g1_mul_by_cofactor"));
            // arkworks clears G2 of BLS12-381 by Budroni and Pintore's map, as RFC 9380 does.
            assert_eq!(c2.0, reference("bls_g2_clear_cofactor"));
        }
    }
}

/// The prime powers whose product is the cofactor, from PARI's `factor`.
fn cofactor_primes(c: &PairingCurve, group: u8) -> Vec<(Number, u32)> {
    let d =
        |s: &str| Number::from_str_radix(s, 10, a::math::numbers::ByteOrder::BigEndian).unwrap();
    match (c.family(), group) {
        (Family::Bn, 1) => vec![],
        (Family::Bn, _) => vec![
            (d("10069"), 1),
            (d("5864401"), 1),
            (d("1875725156269"), 1),
            (
                d("197620364512881247228717050342013327560683201906968909"),
                1,
            ),
        ],
        (Family::Bls12, 1) => vec![
            (d("3"), 1),
            (d("11"), 2),
            (d("10177"), 2),
            (d("859267"), 2),
            (d("52437899"), 2),
        ],
        (Family::Bls12, _) => vec![
            (d("13"), 2),
            (d("23"), 2),
            (d("2713"), 1),
            (d("11953"), 1),
            (d("262069"), 1),
            (
                d(
                    "402096035359507321594726366720466575392706800671181159425656785868777272553337714697862511267018014931937703598282857976535744623203249",
                ),
                1,
            ),
        ],
    }
}

#[test]
fn subgroup_checks_reject_a_component_of_every_prime_order_dividing_the_cofactors() {
    if cfg!(miri) {
        return;
    }
    for (name, c) in curves() {
        let mut r = Rng(12);
        for group in [1u8, 2] {
            let h = if group == 1 {
                c.g1_cofactor()
            } else {
                c.g2_cofactor()
            };
            let primes = cofactor_primes(c, group);
            let mut product = Number::from(1u8);
            for (l, e) in &primes {
                product *= l.pow(*e);
            }
            assert_eq!(&product, h, "{name} G{group}: the factorisation");
            let n = h * c.order();
            // A component of each prime order ℓ: [n/ℓʲ] R for the least j that leaves one (the
            // ℓ-part may be Z/ℓ × Z/ℓ, as on E(F_p) of BLS12-381, where [n/ℓ] kills it), times ℓ
            // until its order is ℓ. The planes E[ℓ] of the squared primes are the business of the
            // test of the characteristic polynomials.
            for (l, e) in &primes {
                let mut found = false;
                for _ in 0..8 {
                    if group == 1 {
                        let x = r.fp(c);
                        let s = g1_from_x(c, x);
                        let Some(mut t) = (1..=*e)
                            .map(|j| &s * &(&n / &l.pow(j)))
                            .find(|t| !t.is_identity())
                        else {
                            continue;
                        };
                        while !(&t * l).is_identity() {
                            t = &t * l;
                        }
                        let p = &c.g1_generator() + &t;
                        assert!(!t.is_in_subgroup(), "{name} G1 ℓ = {l}");
                        assert!(!p.is_in_subgroup(), "{name} G1 ℓ = {l}");
                        assert!(t.clear_cofactor().is_identity(), "{name} G1 ℓ = {l}");
                        assert_eq!(p.clear_cofactor(), c.g1_generator().clear_cofactor());
                        let (x, y) = p.coordinates().unwrap();
                        assert_eq!(G1::new(c, x, y), Err(PointError::NotInSubgroup));
                    } else {
                        let x = r.fp2(c);
                        let s = g2_from_x(c, x);
                        let Some(mut t) = (1..=*e)
                            .map(|j| &s * &(&n / &l.pow(j)))
                            .find(|t| !t.is_identity())
                        else {
                            continue;
                        };
                        while !(&t * l).is_identity() {
                            t = &t * l;
                        }
                        let q = &c.g2_generator() + &t;
                        assert!(!t.is_in_subgroup(), "{name} G2 ℓ = {l}");
                        assert!(!q.is_in_subgroup(), "{name} G2 ℓ = {l}");
                        assert!(t.clear_cofactor().is_identity(), "{name} G2 ℓ = {l}");
                        assert_eq!(q.clear_cofactor(), c.g2_generator().clear_cofactor());
                        let (x, y) = q.coordinates().unwrap();
                        assert_eq!(G2::new(c, x, y), Err(PointError::NotInSubgroup));
                    }
                    found = true;
                    break;
                }
                assert!(found, "{name} G{group}: no component of order {l} drawn");
            }
        }
    }
}

#[test]
fn psi_is_the_frobenius_map_on_the_twist() {
    for (_, c) in curves() {
        let p = c.base_field().modulus();
        let t = p + 1u8 - c.g1_cofactor() * c.order();
        let mut r = Rng(13);
        // On G2, ψ is the product by p.
        let q = &c.g2_generator() * &r.number(4);
        assert_eq!(q.psi(), &q * &p.rem_euclid(c.order()));
        // On the whole twist, ψ² − [t] ψ + [p] = 0, the characteristic polynomial of the Frobenius
        // map of E over F_p.
        let x = r.fp2(c);
        let s = g2_from_x(c, x);
        assert!(!s.is_in_subgroup());
        let lhs = &(&s.psi().psi() - &(&s.psi() * &t)) + &(&s * p);
        assert!(lhs.is_identity());
        // ψ¹² is the identity, as the twelfth power of the Frobenius map is on E(F_p¹²); ψ⁶ is not.
        let mut y = s.clone();
        for i in 1..=12 {
            y = y.psi();
            assert_eq!(y == s, i == 12, "ψ^{i}");
        }
    }
}

#[test]
fn the_cofactor_clearings_are_the_products_by_their_scalars() {
    for (_, c) in curves() {
        let mut r = Rng(14);
        let x = c.x();
        let p = c.base_field().modulus();
        // G1: RFC 9380's h_eff = 1 − x (BLS12-381), the identity map (BN254).
        let s = g1_from_x(c, r.fp(c));
        match c.family() {
            Family::Bls12 => {
                assert_eq!(s.clear_cofactor(), &s * &num("d201000000010001"));
            }
            Family::Bn => assert_eq!(s.clear_cofactor(), s),
        }
        let q2 = g2_from_x(c, r.fp2(c));
        let g2 = &c.g2_generator() * &r.number(4);
        match c.family() {
            Family::Bls12 => {
                // RFC 9380 §8.8.2: h_eff of G2, on every point of the twist.
                let h_eff = num(
                    "bc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe1329c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a359894c0adebbf6b4e8020005aaa95551",
                );
                assert_eq!(q2.clear_cofactor(), &q2 * &h_eff);
                assert_eq!(g2.clear_cofactor(), &g2 * &h_eff);
            }
            Family::Bn => {
                // Fuentes-Castañeda et al.: on G2, x + 3xp + xp² + p³.
                let k = (x + x * 3u8 * p + x * p * p + p.pow(3)).rem_euclid(c.order());
                assert_eq!(g2.clear_cofactor(), &g2 * &k);
                assert!(!k.is_zero());
                assert!(!q2.clear_cofactor().is_identity());
            }
        }
    }
}

#[test]
fn points_from_coordinates_are_checked() {
    for (_, c) in curves() {
        let (x, y) = c.g1_generator().coordinates().unwrap();
        assert_eq!(G1::new(c, x.clone(), y.clone()).unwrap(), c.g1_generator());
        assert_eq!(
            G1::new(c, x.clone(), &y + &c.base_field().one()),
            Err(PointError::NotOnCurve)
        );
        assert_eq!(G1::new_on_curve(c, x, -&y).unwrap(), -c.g1_generator());
        let (x, y) = c.g2_generator().coordinates().unwrap();
        assert_eq!(G2::new(c, x.clone(), y.clone()).unwrap(), c.g2_generator());
        assert_eq!(
            G2::new(c, x.clone(), y.double()),
            Err(PointError::NotOnCurve)
        );
        assert!(G2::new_on_curve(c, x, y).unwrap().is_in_subgroup());
        let mut r = Rng(15);
        let s = g2_from_x(c, r.fp2(c));
        let (x, y) = s.coordinates().unwrap();
        assert_eq!(
            G2::new(c, x.clone(), y.clone()),
            Err(PointError::NotInSubgroup)
        );
        assert_eq!(G2::new_on_curve(c, x, y).unwrap(), s);
        assert_eq!(
            G1::new(c, c.base_field().zero(), c.base_field().zero()),
            Err(PointError::NotOnCurve)
        );
    }
    // BN254's E(F_p) has prime order: every point is in G1.
    let c = PairingCurve::bn254();
    let mut r = Rng(16);
    let s = g1_from_x(c, r.fp(c));
    assert!(s.is_in_subgroup() && (&s * c.order()).is_identity());
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn points_of_different_curves_do_not_add() {
    let _ = PairingCurve::bn254().g1_generator() + PairingCurve::bls12_381().g1_generator();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn coordinates_of_another_field_panic() {
    let c = PairingCurve::bn254();
    let f = PairingCurve::bls12_381().base_field();
    let _ = G1::new(c, f.one(), f.from_u64(2));
}

#[test]
#[should_panic(expected = "elements of different towers")]
fn twist_coordinates_of_another_tower_panic() {
    let c = PairingCurve::bn254();
    let (x, y) = PairingCurve::bls12_381()
        .g2_generator()
        .coordinates()
        .unwrap();
    let _ = G2::new_on_curve(c, x, y);
}

/// Why the tests by endomorphisms are exact, not only right on the points tried: on the twist,
/// `ψ² − [t] ψ + [p] = 0`, so on the points of a prime order `ℓ` dividing the cofactor `ψ − [k]`
/// is invertible unless `k² − tk + p ≡ 0 (mod ℓ)`, and then no point with a component of order `ℓ`
/// passes `ψ(Q) = [k] Q`; on `E(F_p)`, `φ² + φ + 1 = 0` and the same holds for `φ − [−x²]` unless
/// `x⁴ − x² + 1 ≡ 0 (mod ℓ)`. None of the primes of the cofactors is such an exception.
#[test]
fn the_subgroup_checks_are_exact_by_the_characteristic_polynomials_of_the_endomorphisms() {
    for (_, c) in curves() {
        let p = c.base_field().modulus();
        let x = c.x();
        let t = p + 1u8 - c.g1_cofactor() * c.order();
        let k = match c.family() {
            Family::Bn => x * x * 6u8,
            Family::Bls12 => x.clone(),
        };
        // Points of G2 pass: k ≡ p (mod r).
        assert_eq!(k.rem_euclid(c.order()), p.rem_euclid(c.order()));
        for (l, _) in cofactor_primes(c, 2) {
            assert!(!(&k * &k - &t * &k + p).rem_euclid(&l).is_zero(), "ℓ = {l}");
        }
        if c.family() == Family::Bls12 {
            let lambda = -(x * x);
            assert!(
                (&lambda * &lambda + &lambda + 1u8)
                    .rem_euclid(c.order())
                    .is_zero()
            );
            for (l, _) in cofactor_primes(c, 1) {
                let v = (&lambda * &lambda + &lambda + 1u8).rem_euclid(&l);
                assert!(!v.is_zero(), "ℓ = {l}");
            }
            // φ(x, y) = (βx, y) for a cube root of unity β: the three points (βⁱx, y) lie on a
            // horizontal line, so φ² + φ + 1 = 0 on every point; the test is φ(P) = [−x²] P.
            let f = c.base_field();
            let root = (-f.from_u64(3)).sqrt().unwrap();
            let half = f.from_u64(2).inverse().unwrap();
            let mut r = Rng(18);
            let s = g1_from_x(c, r.fp(c));
            let (sx, sy) = s.coordinates().unwrap();
            for beta in [(&root - &f.one()) * &half, (-&root - &f.one()) * &half] {
                assert!(beta.pow_u64(3).is_one());
                let phi = G1::new_on_curve(c, &sx * &beta, sy.clone()).unwrap();
                let phi2 = G1::new_on_curve(c, &sx * &beta.square(), sy.clone()).unwrap();
                assert!((&(&phi2 + &phi) + &s).is_identity());
            }
        }
    }
}
