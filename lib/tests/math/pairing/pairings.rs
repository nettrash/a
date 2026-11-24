//! The optimal ate pairing: bilinearity, non-degeneracy, the identities, the product of pairings,
//! the two halves, and `GT`.

use super::common::{Rng, curves, g1_from_x, g2_from_x};
use a::math::field::{Fp2, Fp6, Fp12};
use a::math::numbers::Number;
use a::math::pairing::{G1, G2, Gt, PairingCurve, pairing};

#[test]
fn the_pairing_is_bilinear() {
    for (_, c) in curves() {
        let mut r = Rng(21);
        let (g1, g2) = (c.g1_generator(), c.g2_generator());
        let e = c.pairing(&g1, &g2);
        let (a, b) = (r.number(4), r.number(4));
        // e(aP, bQ) = e(P, Q)^(ab) = e(abP, Q) = e(P, abQ).
        let ab = &a * &b;
        let eab = c.pairing(&(&g1 * &a), &(&g2 * &b));
        assert_eq!(eab, e.pow(&ab));
        assert_eq!(eab, c.pairing(&(&g1 * &ab), &g2));
        // In each argument.
        let (p1, p2) = (&g1 * &r.number(2), &g1 * &r.number(2));
        let q = &g2 * &r.number(2);
        assert_eq!(
            c.pairing(&(&p1 + &p2), &q),
            &c.pairing(&p1, &q) * &c.pairing(&p2, &q)
        );
        let (q1, q2) = (&g2 * &r.number(2), &g2 * &r.number(2));
        assert_eq!(
            c.pairing(&p1, &(&q1 + &q2)),
            &c.pairing(&p1, &q1) * &c.pairing(&p1, &q2)
        );
    }
}

#[test]
fn the_pairing_is_non_degenerate_and_respects_inverses_and_identities() {
    for (_, c) in curves() {
        let (g1, g2) = (c.g1_generator(), c.g2_generator());
        let e = c.pairing(&g1, &g2);
        // Of order r exactly, r prime.
        assert!(!e.is_identity());
        assert!(e.pow(c.order()).is_identity());
        assert!(e.value().cyclotomic_pow(c.order()).is_one());
        assert_eq!(Gt::from_fp12(c, e.value().clone()), Some(e.clone()));
        // e(P, −Q) = e(−P, Q) = e(P, Q)⁻¹.
        assert_eq!(c.pairing(&g1, &-&g2), e.inverse());
        assert_eq!(c.pairing(&-&g1, &g2), e.inverse());
        assert_eq!(c.pairing(&-&g1, &-&g2), e);
        // The points at infinity.
        assert!(c.pairing(&c.g1_identity(), &g2).is_identity());
        assert!(c.pairing(&g1, &c.g2_identity()).is_identity());
        assert!(c.pairing(&c.g1_identity(), &c.g2_identity()).is_identity());
        assert!(c.pairing(&(&g1 * c.order()), &g2).is_identity());
        // The free function and the generator of GT.
        assert_eq!(pairing(&g1, &g2), e);
        assert_eq!(c.gt_generator(), e);
        assert!(c.gt_identity().is_identity());
    }
}

#[test]
fn a_product_of_pairings_takes_one_final_exponentiation() {
    for (_, c) in curves() {
        let mut r = Rng(22);
        let points: Vec<(G1<'_>, G2<'_>)> = (0..3)
            .map(|_| {
                (
                    &c.g1_generator() * &r.number(2),
                    &c.g2_generator() * &r.number(2),
                )
            })
            .collect();
        let mut pairs: Vec<(&G1<'_>, &G2<'_>)> = points.iter().map(|(p, q)| (p, q)).collect();
        let mut want = c.gt_identity();
        for (p, q) in &pairs {
            want *= c.pairing(p, q);
        }
        assert_eq!(c.multi_pairing(&pairs), want);
        // Pairs with a point at infinity contribute 1.
        let (o1, o2) = (c.g1_identity(), c.g2_identity());
        pairs.push((&o1, &points[0].1));
        pairs.push((&points[1].0, &o2));
        assert_eq!(c.multi_pairing(&pairs), want);
        assert!(c.multi_pairing(&[]).is_identity());
        // The halves.
        let f = c.miller_loop(&pairs);
        assert_eq!(c.final_exponentiation(&f), want);
        assert!(c.miller_loop(&[]).is_one());
        // A pairing check of e(aP, Q) e(−P, aQ) = 1.
        let a = r.number(4);
        let (p, q) = (c.g1_generator(), c.g2_generator());
        let (ap, aq, mp) = (&p * &a, &q * &a, -&p);
        assert!(c.multi_pairing(&[(&ap, &q), (&mp, &aq)]).is_identity());
    }
}

#[test]
fn the_final_exponentiation_maps_every_nonzero_element_into_gt() {
    for (_, c) in curves() {
        let t = c.tower();
        let x = Fp12::new(
            Fp6::new(t.xi(), Fp2::one(t), t.xi().square()),
            Fp6::new(Fp2::zero(t), t.xi(), Fp2::one(t)),
        );
        let e = c.final_exponentiation(&x);
        assert!(e.pow(c.order()).is_identity());
        assert!(Gt::from_fp12(c, e.value().clone()).is_some());
        // An element outside GT is refused, also one of the cyclotomic subgroup of another order.
        assert!(Gt::from_fp12(c, x.clone()).is_none());
        let easy = &x.conjugate() / &x;
        let easy = &easy.frobenius(2) * &easy;
        assert!(easy.is_cyclotomic());
        assert!(Gt::from_fp12(c, easy).is_none());
        assert!(Gt::from_fp12(c, Fp12::zero(t)).is_none());
        // Elements of F_p⁶ vanish: (p⁶ − 1) divides the exponent.
        let y = Fp12::from_fp6(Fp6::new(t.xi(), Fp2::one(t), Fp2::zero(t)));
        assert!(c.final_exponentiation(&y).is_identity());
    }
}

#[test]
fn points_outside_the_subgroups_give_a_value_of_gt_without_panicking() {
    for (_, c) in curves() {
        let mut r = Rng(23);
        let p = g1_from_x(c, r.fp(c));
        let q = g2_from_x(c, r.fp2(c));
        assert!(!q.is_in_subgroup());
        let e = c.pairing(&p, &q);
        assert!(e.pow(c.order()).is_identity());
        // Points of small order on the twist: T + T + … reaches infinity inside the loop.
        let n = c.g2_cofactor() * c.order();
        let small = &q
            * &(&n
                / &Number::from(match c.family() {
                    a::math::pairing::Family::Bn => 10069u32,
                    a::math::pairing::Family::Bls12 => 13u32,
                }));
        let e = c.pairing(&c.g1_generator(), &small);
        assert!(e.pow(c.order()).is_identity());
    }
}

#[test]
fn gt_is_a_group_of_order_r() {
    for (_, c) in curves() {
        let mut r = Rng(24);
        let g = c.gt_generator();
        let (a, b) = (r.number(4), r.number(4));
        let (ga, gb) = (g.pow(&a), g.pow(&b));
        assert_eq!(&ga * &gb, g.pow(&(&a + &b)));
        assert_eq!(&ga / &gb, g.pow(&(&a - &b)));
        assert_eq!(&ga * &ga.inverse(), c.gt_identity());
        assert_eq!(ga.square(), &ga * &ga);
        assert_eq!(g.pow(&-&a), ga.inverse());
        assert_eq!(g.pow(&(c.order() + &a)), ga);
        assert!(g.pow(&Number::from(0u8)).is_identity());
        let mut x = ga.clone();
        x *= &gb;
        x /= gb.clone();
        assert_eq!(x, ga);
        use std::collections::HashSet;
        #[allow(
            clippy::mutable_key_type,
            reason = "the field may hold a cache filled once (`OnceLock`); the hash reads only the value"
        )]
        let set: HashSet<Gt<'_>> = [ga.clone(), ga.clone(), gb].into_iter().collect();
        assert_eq!(set.len(), 2);
        assert!(!ga.to_string().is_empty());
        assert_eq!(ga.curve().name(), c.name());
    }
}

#[test]
#[should_panic(expected = "the points are on different curves")]
fn a_pairing_of_points_of_different_curves_panics() {
    let _ = pairing(
        &PairingCurve::bn254().g1_generator(),
        &PairingCurve::bls12_381().g2_generator(),
    );
}

#[test]
#[should_panic(expected = "the value of a Miller loop is not 0")]
fn the_final_exponentiation_of_zero_panics() {
    let c = PairingCurve::bn254();
    let _ = c.final_exponentiation(&Fp12::zero(c.tower()));
}
