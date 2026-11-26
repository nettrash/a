//! Ethereum's precompiles for BN254 against the tests of go-ethereum (the vectors of EIP-196 and
//! EIP-197), and their failures.

use super::common::{Rng, be, g2_from_x, hex, unhex};
use a::math::pairing::ethereum::{ec_add, ec_mul, ec_pairing};
use a::math::pairing::{PairingCurve, PointError};

const VECTORS: &str = include_str!("data/eth_vectors.txt");

#[test]
fn the_precompiles_pass_go_ethereums_tests() {
    let mut counts = [0usize; 3];
    for line in VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        let (kind, name, input, want) = (f[0], f[1], f[2], f[3]);
        let input = if input == "-" {
            Vec::new()
        } else {
            unhex(input)
        };
        let got = match kind {
            "add" => {
                counts[0] += 1;
                hex(&ec_add(&input).unwrap())
            }
            "mul" => {
                counts[1] += 1;
                hex(&ec_mul(&input).unwrap())
            }
            _ => {
                counts[2] += 1;
                // The ten-pair products take a few seconds in the debug profile under Miri.
                if cfg!(miri) && input.len() > 2 * 192 {
                    continue;
                }
                hex(&ec_pairing(&input).unwrap())
            }
        };
        assert_eq!(got, want, "{kind} {name}");
    }
    assert_eq!(counts, [16, 19, 14]);
}

#[test]
fn the_precompiles_pad_short_inputs_and_ignore_the_rest() {
    let g = PairingCurve::bn254().g1_generator().to_ethereum();
    let mut input = g.clone();
    input.extend_from_slice(&g);
    let sum = ec_add(&input).unwrap();
    // One point: the other is read as zeros, the point at infinity.
    assert_eq!(ec_add(&g).unwrap().to_vec(), g);
    assert_eq!(ec_add(&[]).unwrap(), [0; 64]);
    input.extend_from_slice(&[0xff; 7]);
    assert_eq!(ec_add(&input).unwrap(), sum);
    // A scalar of 31 bytes is padded on the right: 2 · 256.
    let mut m = g.clone();
    m.extend_from_slice(&[0; 30]);
    m.push(2);
    let mut by512 = g.clone();
    by512.extend_from_slice(&[0; 30]);
    by512.extend_from_slice(&[2, 0]);
    assert_eq!(ec_mul(&m).unwrap(), ec_mul(&by512).unwrap());
    // The scalar r gives the point at infinity, r + 1 the point.
    let c = PairingCurve::bn254();
    let mut m = g.clone();
    m.extend_from_slice(&be(c.order(), 32));
    assert_eq!(ec_mul(&m).unwrap(), [0; 64]);
    let mut m = g.clone();
    m.extend_from_slice(&be(&(c.order() + 1u8), 32));
    assert_eq!(ec_mul(&m).unwrap().to_vec(), g);
}

#[test]
fn the_precompiles_fail_on_invalid_inputs() {
    let c = PairingCurve::bn254();
    let g1 = c.g1_generator().to_ethereum();
    let g2 = c.g2_generator().to_ethereum();
    // A coordinate equal to p; a point off the curve.
    let mut bad = g1.clone();
    bad[..32].copy_from_slice(&be(c.base_field().modulus(), 32));
    assert_eq!(ec_add(&bad), Err(PointError::NotInField));
    assert_eq!(ec_mul(&bad), Err(PointError::NotInField));
    let mut off = g1.clone();
    off[63] = 3;
    let mut input = g1.clone();
    input.extend_from_slice(&off);
    assert_eq!(ec_add(&input), Err(PointError::NotOnCurve));
    // ecPairing: lengths that are not multiples of 192, and invalid points of G2.
    let mut pair = g1.clone();
    pair.extend_from_slice(&g2);
    assert_eq!(ec_pairing(&pair[..191]), Err(PointError::Length));
    assert_eq!(ec_pairing(&[0; 64]), Err(PointError::Length));
    let mut r = Rng(41);
    let s = g2_from_x(c, r.fp2(c)).to_ethereum();
    let mut input = g1.clone();
    input.extend_from_slice(&s);
    assert_eq!(ec_pairing(&input), Err(PointError::NotInSubgroup));
    let mut input = g1.clone();
    let mut big = g2.clone();
    big[..32].copy_from_slice(&be(c.base_field().modulus(), 32));
    input.extend_from_slice(&big);
    assert_eq!(ec_pairing(&input), Err(PointError::NotInField));
    // e(P, Q) alone is not 1; e(P, Q) e(−P, Q) is.
    assert_eq!(ec_pairing(&pair).unwrap(), [0; 32]);
    let mut check = pair.clone();
    check.extend_from_slice(&(-c.g1_generator()).to_ethereum());
    check.extend_from_slice(&g2);
    let mut one = [0u8; 32];
    one[31] = 1;
    assert_eq!(ec_pairing(&check).unwrap(), one);
}
