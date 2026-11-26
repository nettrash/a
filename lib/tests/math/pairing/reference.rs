//! The pairings against arkworks, the `bls12_381` crate, py_ecc, RELIC and the IETF draft on
//! pairing-friendly curves.

use super::common::{Fnv, Rng, curves, reference, unhex};
use a::math::field::Fp12;
use a::math::numbers::{ByteOrder, Number};
use a::math::pairing::{Family, G1, G2, PairingCurve};

/// The encoding the references hash: ZCash's compressed for BLS12-381, Ethereum's for BN254.
fn enc1(p: &G1<'_>) -> Vec<u8> {
    match p.curve().family() {
        Family::Bls12 => p.to_zcash_compressed(),
        Family::Bn => p.to_ethereum(),
    }
}

fn enc2(q: &G2<'_>) -> Vec<u8> {
    match q.curve().family() {
        Family::Bls12 => q.to_zcash_compressed(),
        Family::Bn => q.to_ethereum(),
    }
}

#[test]
fn pairings_of_seeded_multiples_match_arkworks_the_bls12_381_crate_and_py_ecc() {
    let cases = if cfg!(miri) { 1 } else { 6 };
    for (name, c) in curves() {
        let mut r = Rng(2);
        let (mut h1, mut h2, mut he, mut hm, mut hpy) =
            (Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new());
        let mut points = Vec::new();
        for i in 0..cases {
            let a = r.number(4);
            let b = r.number(4);
            let p = &c.g1_generator() * &a;
            let q = &c.g2_generator() * &b;
            h1.eat(&enc1(&p));
            h2.eat(&enc2(&q));
            let e = c.pairing(&p, &q);
            he.fp12(e.value());
            if i < 2 {
                hpy.fp12(e.value());
            }
            points.push((p, q));
        }
        if cfg!(miri) {
            continue;
        }
        let pairs: Vec<(&G1<'_>, &G2<'_>)> = points.iter().map(|(p, q)| (p, q)).collect();
        hm.fp12(c.multi_pairing(&pairs).value());
        assert_eq!(h1.0, reference(&format!("{name}_g1_mul")), "{name} G1");
        assert_eq!(h2.0, reference(&format!("{name}_g2_mul")), "{name} G2");
        assert_eq!(
            he.0,
            reference(&format!("{name}_pairing")),
            "{name} pairing"
        );
        assert_eq!(
            hm.0,
            reference(&format!("{name}_multi_pairing")),
            "{name} multi"
        );
        // py_ecc raises to (p¹² − 1)/r alone, and does not conjugate for BLS12-381's negative x:
        // its values, raised to −3 and to 2x(6x² + 3x + 1), are these (gen_py.py).
        assert_eq!(
            hpy.0,
            reference(&format!("{name}_pairing_py_ecc")),
            "{name} py_ecc"
        );
    }
}

/// `e(g₁, g₂)` on BLS12-381 as Diego Aranha computed it with RELIC, in the test of the
/// `bls12_381` crate `test_pairing_result_against_relic`: the twelve coefficients in the tower's
/// order.
const RELIC: [&str; 12] = [
    "1250EBD871FC0A92A7B2D83168D0D727272D441BEFA15C503DD8E90CE98DB3E7B6D194F60839C508A84305AACA1789B6",
    "089A1C5B46E5110B86750EC6A532348868A84045483C92B7AF5AF689452EAFABF1A8943E50439F1D59882A98EAA0170F",
    "1368BB445C7C2D209703F239689CE34C0378A68E72A6B3B216DA0E22A5031B54DDFF57309396B38C881C4C849EC23E87",
    "193502B86EDB8857C273FA075A50512937E0794E1E65A7617C90D8BD66065B1FFFE51D7A579973B1315021EC3C19934F",
    "01B2F522473D171391125BA84DC4007CFBF2F8DA752F7C74185203FCCA589AC719C34DFFBBAAD8431DAD1C1FB597AAA5",
    "018107154F25A764BD3C79937A45B84546DA634B8F6BE14A8061E55CCEBA478B23F7DACAA35C8CA78BEAE9624045B4B6",
    "19F26337D205FB469CD6BD15C3D5A04DC88784FBB3D0B2DBDEA54D43B2B73F2CBB12D58386A8703E0F948226E47EE89D",
    "06FBA23EB7C5AF0D9F80940CA771B6FFD5857BAAF222EB95A7D2809D61BFE02E1BFD1B68FF02F0B8102AE1C2D5D5AB1A",
    "11B8B424CD48BF38FCEF68083B0B0EC5C81A93B330EE1A677D0D15FF7B984E8978EF48881E32FAC91B93B47333E2BA57",
    "03350F55A7AEFCD3C31B4FCB6CE5771CC6A0E9786AB5973320C806AD360829107BA810C5A09FFDD9BE2291A0C25A99A2",
    "04C581234D086A9902249B64728FFD21A189E87935A954051C7CDBA7B3872629A4FAFC05066245CB9108F0242D0FE3EF",
    "0F41E58663BF08CF068672CBD01A7EC73BACA4D72CA93544DEFF686BFD6DF543D48EAA24AFE47E1EFDE449383B676631",
];

/// `e(BP, BP')` on BLS12-381 in appendix B of the IETF draft on pairing-friendly curves
/// (draft-irtf-cfrg-pairing-friendly-curves-11), `e_0, …, e_11`, which are the tower's
/// coefficients of `f_x(P)^((p¹² − 1)/r)` (gen_py.py checks the order against py_ecc).
const IETF: [&str; 12] = [
    "11619b45f61edfe3b47a15fac19442526ff489dcda25e59121d9931438907dfd448299a87dde3a649bdba96e84d54558",
    "153ce14a76a53e205ba8f275ef1137c56a566f638b52d34ba3bf3bf22f277d70f76316218c0dfd583a394b8448d2be7f",
    "095668fb4a02fe930ed44767834c915b283b1c6ca98c047bd4c272e9ac3f3ba6ff0b05a93e59c71fba77bce995f04692",
    "16deedaa683124fe7260085184d88f7d036b86f53bb5b7f1fc5e248814782065413e7d958d17960109ea006b2afdeb5f",
    "09c92cf02f3cd3d2f9d34bc44eee0dd50314ed44ca5d30ce6a9ec0539be7a86b121edc61839ccc908c4bdde256cd6048",
    "111061f398efc2a97ff825b04d21089e24fd8b93a47e41e60eae7e9b2a38d54fa4dedced0811c34ce528781ab9e929c7",
    "01ecfcf31c86257ab00b4709c33f1c9c4e007659dd5ffc4a735192167ce197058cfb4c94225e7f1b6c26ad9ba68f63bc",
    "08890726743a1f94a8193a166800b7787744a8ad8e2f9365db76863e894b7a11d83f90d873567e9d645ccf725b32d26f",
    "0e61c752414ca5dfd258e9606bac08daec29b3e2c57062669556954fb227d3f1260eedf25446a086b0844bcd43646c10",
    "0fe63f185f56dd29150fc498bbeea78969e7e783043620db33f75a05a0a2ce5c442beaff9da195ff15164c00ab66bdde",
    "10900338a92ed0b47af211636f7cfdec717b7ee43900eee9b5fc24f0000c5874d4801372db478987691c566a8c474978",
    "1454814f3085f0e6602247671bc408bbce2007201536818c901dbd4d2095dd86c1ec8b888e59611f60a301af7776be3d",
];

fn fp12_of(c: &PairingCurve, hex: &[&str; 12]) -> Fp12<'static> {
    let n: Vec<Number> = hex
        .iter()
        .map(|h| Number::new(unhex(h), ByteOrder::BigEndian))
        .collect();
    Fp12::from_coefficients(c.tower(), &n)
}

#[test]
fn the_pairing_of_the_generators_of_bls12_381_is_relics() {
    let c = PairingCurve::bls12_381();
    let e = c.pairing(&c.g1_generator(), &c.g2_generator());
    assert_eq!(*e.value(), fp12_of(c, &RELIC));
    assert_eq!(e, c.gt_generator());
}

#[test]
fn the_pairing_of_the_generators_of_bls12_381_is_the_cube_of_the_ietf_drafts() {
    // The draft's input points are the generators, as the curve has them.
    let c = PairingCurve::bls12_381();
    let (x, y) = c.g1_generator().coordinates().unwrap();
    assert_eq!(
        (x.value(), y.value()),
        (
            Number::new(
                unhex(
                    "17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
                ),
                ByteOrder::BigEndian
            ),
            Number::new(
                unhex(
                    "08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"
                ),
                ByteOrder::BigEndian
            )
        )
    );
    let e = c.pairing(&c.g1_generator(), &c.g2_generator());
    let draft = fp12_of(c, &IETF);
    assert_eq!(*e.value(), draft.cyclotomic_pow(&Number::from(3u8)));
    assert_eq!(*e.value(), draft.pow(&Number::from(3u8)));
    // The draft's value is itself in GT, and our pairing raised to 1/3 modulo r is it.
    let third = Number::from(3u8)
        .mod_inverse(c.order())
        .expect("3 is prime to r");
    assert_eq!(*e.pow(&third).value(), draft);
}

#[test]
fn the_documented_recipe_gives_py_eccs_values() {
    // py_ecc gives e^(1/c) on BN254 and e^(−1/3) on BLS12-381 (it loops over |x| without
    // conjugating): e.pow(c⁻¹ mod r), inverted on BLS12-381, is its value exactly.
    let cases = if cfg!(miri) { 1 } else { 2 };
    for (name, c) in curves() {
        let x = c.x();
        let factor = match c.family() {
            Family::Bn => x * 2u8 * (x * x * 6u8 + x * 3u8 + 1u8),
            Family::Bls12 => Number::from(3u8),
        };
        let inv = factor.mod_inverse(c.order()).expect("c is prime to r");
        let mut r = Rng(2);
        let mut h = Fnv::new();
        for _ in 0..cases {
            let a = r.number(4);
            let b = r.number(4);
            let e = c.pairing(&(&c.g1_generator() * &a), &(&c.g2_generator() * &b));
            let theirs = match c.family() {
                Family::Bn => e.pow(&inv),
                Family::Bls12 => e.pow(&inv).inverse(),
            };
            h.fp12(theirs.value());
        }
        if !cfg!(miri) {
            assert_eq!(
                h.0,
                reference(&format!("{name}_pairing_py_ecc_raw")),
                "{name}"
            );
        }
    }
}

#[test]
fn the_factor_of_the_final_exponentiation_is_prime_to_r() {
    // BN254: c = 2x(6x² + 3x + 1); BLS12-381: c = 3. Both prime to r, so e^c is a pairing too.
    for (_, curve) in curves() {
        let x = curve.x();
        let c = match curve.family() {
            Family::Bn => x * 2u8 * (x * x * 6u8 + x * 3u8 + 1u8),
            Family::Bls12 => Number::from(3u8),
        };
        assert_eq!(c.gcd(curve.order()), Number::from(1u8));
        // c (p⁴ − p² + 1)/r is what the chains raise to: check it against the polynomials of FKR
        // and HHT in x, p.
        let p = curve.base_field().modulus();
        let r = curve.order();
        let d = (p.pow(4) - p.pow(2) + 1u8) / r;
        match curve.family() {
            Family::Bn => {
                let x2 = x * x;
                let x3 = &x2 * x;
                let l0 = &x3 * 12u8 + &x2 * 12u8 + x * 6u8 + 1u8;
                let l1 = &x3 * 12u8 + &x2 * 6u8 + x * 4u8;
                let l2 = &x3 * 12u8 + &x2 * 6u8 + x * 6u8;
                let l3 = &x3 * 12u8 + &x2 * 6u8 + x * 4u8 - 1u8;
                assert_eq!(l0 + l1 * p + l2 * p.pow(2) + l3 * p.pow(3), &c * &d);
            }
            Family::Bls12 => {
                let xm1 = x - 1u8;
                let lhs = &xm1 * &xm1 * (x + p) * (x * x + p * p - 1u8) + 3u8;
                assert_eq!(lhs, &c * &d);
            }
        }
    }
}
