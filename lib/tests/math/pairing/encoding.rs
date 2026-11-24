//! The ZCash encodings of BLS12-381 against the `bls12_381` crate (its test vectors of the first
//! thousand multiples, and its verdicts on corrupted encodings), and the Ethereum encodings of
//! BN254.

use super::common::{Fnv, Rng, g2_from_x, hex, unhex};
use a::math::pairing::{G1, G2, PairingCurve, PointError};

const CASES: &str = include_str!("data/zcash_cases.txt");

/// The digest and the length of the `bls12_381` crate's file of encodings `name`.
fn zcash_reference(name: &str) -> (u64, usize) {
    let key = format!("zcash_{name}");
    let line = include_str!("data/pairing_ref.txt")
        .lines()
        .find(|l| l.split(' ').next() == Some(key.as_str()))
        .unwrap();
    let mut it = line.split(' ').skip(1);
    let d = u64::from_str_radix(it.next().unwrap(), 16).unwrap();
    (d, it.next().unwrap().parse().unwrap())
}

/// Encodes `[i] G` for `i` in `0..count` in the four forms, decoding every `step`-th back, and
/// compares the digests of all of them with those of the crate's files.
fn thousand_multiples(count: usize, step: usize) {
    let c = PairingCurve::bls12_381();
    let mut h = [Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new()];
    let mut len = [0usize; 4];
    let (mut p, mut q) = (c.g1_identity(), c.g2_identity());
    for i in 0..count {
        let e = [
            p.to_zcash_compressed(),
            p.to_zcash_uncompressed(),
            q.to_zcash_compressed(),
            q.to_zcash_uncompressed(),
        ];
        for k in 0..4 {
            h[k].eat(&e[k]);
            len[k] += e[k].len();
        }
        if i % step == 0 {
            assert_eq!(G1::from_zcash(c, &e[0]).unwrap(), p, "{i}");
            assert_eq!(G1::from_zcash(c, &e[1]).unwrap(), p, "{i}");
            assert_eq!(G2::from_zcash(c, &e[2]).unwrap(), q, "{i}");
            assert_eq!(G2::from_zcash(c, &e[3]).unwrap(), q, "{i}");
        }
        p += c.g1_generator();
        q += c.g2_generator();
    }
    if count == 1000 {
        for (k, name) in [
            "g1_compressed",
            "g1_uncompressed",
            "g2_compressed",
            "g2_uncompressed",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!((h[k].0, len[k]), zcash_reference(name), "{name}");
        }
    }
}

#[test]
fn zcash_encodings_of_the_first_thousand_multiples_are_the_bls12_381_crates() {
    if cfg!(miri) {
        thousand_multiples(3, 1);
    } else {
        thousand_multiples(1000, 50);
    }
}

/// Every one of the 4000 encodings decoded back: about 20 s in the debug profile.
#[test]
#[ignore]
fn zcash_encodings_of_the_first_thousand_multiples_all_decode() {
    thousand_multiples(1000, 1);
}

#[test]
fn zcash_decoding_agrees_with_the_bls12_381_crate_on_corrupted_encodings() {
    let c = PairingCurve::bls12_381();
    let (mut ok, mut err) = (0, 0);
    for line in CASES.lines().filter(|l| !l.starts_with('#')) {
        let mut f = line.split(' ');
        let (group, input, want) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let bytes = unhex(input);
        let got = if group == "g1" {
            G1::from_zcash(c, &bytes).map(|p| hex(&p.to_zcash_compressed()))
        } else {
            G2::from_zcash(c, &bytes).map(|q| hex(&q.to_zcash_compressed()))
        };
        match got {
            Ok(h) => {
                assert_eq!(h, want, "{group} {input}");
                ok += 1;
            }
            Err(e) => {
                assert_eq!(want, "err", "{group} {input}: {e}");
                err += 1;
            }
        }
    }
    assert_eq!((ok, err), (22, 101));
}

#[test]
fn zcash_decoding_names_what_is_wrong() {
    let c = PairingCurve::bls12_381();
    let g = c.g1_generator().to_zcash_compressed();
    assert_eq!(G1::from_zcash(c, &g[..47]), Err(PointError::Length));
    assert_eq!(G1::from_zcash(c, &[]), Err(PointError::Length));
    assert_eq!(G2::from_zcash(c, &g), Err(PointError::Length));
    // The compression flag must agree with the length.
    let mut b = g.clone();
    b[0] &= 0x7f;
    assert_eq!(G1::from_zcash(c, &b), Err(PointError::Flags));
    let mut u = c.g1_generator().to_zcash_uncompressed();
    u[0] |= 0x80;
    assert_eq!(G1::from_zcash(c, &u), Err(PointError::Flags));
    // The sort flag only on a compressed finite point.
    let mut u = c.g1_generator().to_zcash_uncompressed();
    u[0] |= 0x20;
    assert_eq!(G1::from_zcash(c, &u), Err(PointError::Flags));
    // x = p.
    let mut b = vec![0u8; 48];
    b.copy_from_slice(&super::common::be(c.base_field().modulus(), 48));
    b[0] |= 0x80;
    assert_eq!(G1::from_zcash(c, &b), Err(PointError::NotInField));
    // x = 0 gives (0, ±2), of order 3; x = 1 gives x³ + 4 = 5, not a square modulo p.
    let mut b = vec![0u8; 48];
    b[0] = 0x80;
    assert_eq!(G1::from_zcash(c, &b), Err(PointError::NotInSubgroup));
    b[47] = 1;
    assert_eq!(G1::from_zcash(c, &b), Err(PointError::NotOnCurve));
    // A point of the twist outside G2.
    let mut r = Rng(31);
    let s = g2_from_x(c, r.fp2(c));
    assert_eq!(
        G2::from_zcash(c, &s.to_zcash_compressed()),
        Err(PointError::NotInSubgroup)
    );
    assert_eq!(
        G2::from_zcash(c, &s.to_zcash_uncompressed()),
        Err(PointError::NotInSubgroup)
    );
    // Not on the curve: y + 1.
    let mut u = c.g2_generator().to_zcash_uncompressed();
    u[191] ^= 1;
    assert_eq!(G2::from_zcash(c, &u), Err(PointError::NotOnCurve));
    // The points at infinity.
    let o = c.g2_identity().to_zcash_compressed();
    assert_eq!(o[0], 0xc0);
    assert!(o[1..].iter().all(|&x| x == 0));
    assert!(G2::from_zcash(c, &o).unwrap().is_identity());
    let o = c.g1_identity().to_zcash_uncompressed();
    assert_eq!((o[0], o.len()), (0x40, 96));
    assert!(G1::from_zcash(c, &o).unwrap().is_identity());
    // −P sets the sort flag the other way.
    let (p, m) = (c.g2_generator(), -c.g2_generator());
    assert_eq!(
        p.to_zcash_compressed()[0] ^ m.to_zcash_compressed()[0],
        0x20
    );
}

#[test]
fn ethereum_encodings_round_trip_and_reject_invalid_points() {
    let c = PairingCurve::bn254();
    let mut r = Rng(32);
    let p = &c.g1_generator() * &r.number(4);
    let q = &c.g2_generator() * &r.number(4);
    let (bp, bq) = (p.to_ethereum(), q.to_ethereum());
    assert_eq!((bp.len(), bq.len()), (64, 128));
    assert_eq!(G1::from_ethereum(c, &bp).unwrap(), p);
    assert_eq!(G2::from_ethereum(c, &bq).unwrap(), q);
    // The coefficient of u first: x₁ ‖ x₀ ‖ y₁ ‖ y₀.
    let (x, _) = q.coordinates().unwrap();
    assert_eq!(bq[..32], super::common::be(&x.c1().value(), 32)[..]);
    assert_eq!(bq[32..64], super::common::be(&x.c0().value(), 32)[..]);
    assert!(c.g1_identity().to_ethereum().iter().all(|&x| x == 0));
    assert!(G2::from_ethereum(c, &[0; 128]).unwrap().is_identity());
    assert_eq!(G1::from_ethereum(c, &bp[1..]), Err(PointError::Length));
    assert_eq!(G2::from_ethereum(c, &bp), Err(PointError::Length));
    // x = p.
    let mut b = bp.clone();
    b[..32].copy_from_slice(&super::common::be(c.base_field().modulus(), 32));
    assert_eq!(G1::from_ethereum(c, &b), Err(PointError::NotInField));
    let mut b = bq.clone();
    b[96..].copy_from_slice(&super::common::be(c.base_field().modulus(), 32));
    assert_eq!(G2::from_ethereum(c, &b), Err(PointError::NotInField));
    // Not on the curve; (0, 1) is not the point at infinity.
    let mut b = bp.clone();
    b[63] ^= 1;
    assert_eq!(G1::from_ethereum(c, &b), Err(PointError::NotOnCurve));
    let mut b = vec![0u8; 64];
    b[63] = 1;
    assert_eq!(G1::from_ethereum(c, &b), Err(PointError::NotOnCurve));
    // A point of the twist outside G2.
    let s = g2_from_x(c, r.fp2(c));
    assert_eq!(
        G2::from_ethereum(c, &s.to_ethereum()),
        Err(PointError::NotInSubgroup)
    );
}

#[test]
#[should_panic(expected = "the ZCash encoding is that of BLS12-381")]
fn the_zcash_encoding_is_not_bn254s() {
    let _ = PairingCurve::bn254().g1_generator().to_zcash_compressed();
}

#[test]
#[should_panic(expected = "the Ethereum encoding of EIP-196 and EIP-197 is that of BN254")]
fn the_ethereum_encoding_is_not_bls12_381s() {
    let c = PairingCurve::bls12_381();
    let _ = G2::from_ethereum(c, &[0; 192]);
}
