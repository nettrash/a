//! Residues and products modulo moduli of several limbs: the division by the normalized modulus
//! with the reciprocal of its top limbs, against the integers' remainder, for moduli whose top
//! limbs are all ones, that need no shift, and of up to ten limbs; the quotient limb at its
//! largest by dividends just below multiples of the modulus.

use super::common::*;
use a::math::numbers::{ByteOrder, Number};
use a::math::poly::ModPolynomial;

/// `2^bits`.
fn pow2(bits: usize) -> Number {
    Number::from(1u8) << bits
}

/// The moduli of several limbs: no shift (top bit set), a shift of one, top limbs all ones, a
/// power of 2, of 2 to 10 limbs.
pub(super) fn limb_moduli() -> Vec<Number> {
    let one = Number::from(1u8);
    let mut v = vec![
        &pow2(64) + &Number::from(13u8),
        &pow2(128) - &one,
        &pow2(127) - &one,
        &pow2(128) - &Number::from(159u8),
        &pow2(64) + &one,
        &pow2(192) - &one,
        &pow2(191) + &Number::from(5u8),
        // Top two limbs all ones, the third zero.
        &(&pow2(192) - &pow2(64)) + &one,
        &pow2(130) - &Number::from(5u8),
        &pow2(255) - &Number::from(19u8),
        &pow2(256) - &Number::from(189u8),
        pow2(200),
    ];
    if !cfg!(miri) {
        v.extend([
            &pow2(512) - &one,
            &pow2(521) - &one,
            &pow2(640) + &Number::from(7u8),
            &(&pow2(576) - &pow2(512)) + &Number::from(3u8),
        ]);
    }
    v
}

/// Dividends that make the quotient limbs largest: multiples of `n` less one and less small
/// numbers, by `B - 1`, `B^2 - 1` and random factors.
fn hard_dividends(s: &mut u64, n: &Number) -> Vec<Number> {
    let one = Number::from(1u8);
    let mut v = Vec::new();
    for m in [
        &pow2(64) - &one,
        &pow2(128) - &one,
        n - &one,
        n.clone(),
        &pow2(n.bits() as usize) - &one,
    ] {
        let p = n * &m;
        v.push(&p - &one);
        v.push(&p + &one);
        v.push(p.clone());
        v.push(&p - &(n >> 1));
    }
    for _ in 0..size(40, 6) {
        let m = number(s, 3 * (n.bits() as usize).div_ceil(8));
        v.push(&(n * &m.abs()) - &one);
    }
    v
}

#[test]
fn residues_of_any_integer_modulo_several_limbs() {
    let mut s = 0x5DEE_CE66_D1CE_4E5B;
    for n in limb_moduli() {
        let bytes = (n.bits() as usize).div_ceil(8);
        let mut xs: Vec<Number> = (0..size(200, 12))
            .map(|_| number(&mut s, 3 * bytes + 4))
            .collect();
        xs.extend(hard_dividends(&mut s, &n));
        for x in xs {
            let want = x.rem_euclid(&n);
            let p = ModPolynomial::new(&[x.clone(), Number::from(1u8)], &n);
            assert_eq!(p.coeff(0), want, "{x} mod {n}");
        }
    }
}

#[test]
fn products_of_residues_modulo_several_limbs() {
    let mut s = 0x2545_F491_4F6C_DD1D;
    for n in limb_moduli() {
        let bytes = (n.bits() as usize).div_ceil(8) + 1;
        let top = &n - &Number::from(1u8);
        for (la, lb) in [
            (1, 1),
            (1, 3),
            (2, 2),
            (3, 5),
            (6, 6),
            (7, 9),
            (12, 20),
            (33, 31),
        ] {
            if la * lb > size(1000, 40) {
                continue;
            }
            // All maximal residues, then random ones.
            for round in 0..3 {
                let make = |s: &mut u64, len: usize| -> Vec<Number> {
                    (0..len)
                        .map(|_| {
                            if round == 0 {
                                top.clone()
                            } else {
                                number(s, bytes)
                            }
                        })
                        .collect()
                };
                let a = ModPolynomial::new(&make(&mut s, la), &n);
                let b = ModPolynomial::new(&make(&mut s, lb), &n);
                let want = oracle_mod(&oracle_mul(&a.coeffs(), &b.coeffs()), &n);
                assert_eq!((&a * &b).coeffs(), want, "n = {n}, {la} by {lb}");
                let sq = oracle_mod(&oracle_mul(&a.coeffs(), &a.coeffs()), &n);
                assert_eq!((&a * &a.clone()).coeffs(), sq, "n = {n}, square of {la}");
            }
        }
    }
}

#[test]
fn residues_in_both_byte_orders() {
    let mut s = 0x1234_5678_9ABC_DEF1;
    for n in limb_moduli() {
        let bytes = (n.bits() as usize).div_ceil(8);
        for _ in 0..size(50, 4) {
            let x = number(&mut s, 2 * bytes + 3);
            let mut be = x.clone();
            be.value = x.value.iter().rev().copied().collect();
            be.byte_order = match x.byte_order {
                ByteOrder::BigEndian => ByteOrder::LittleEndian,
                ByteOrder::LittleEndian => ByteOrder::BigEndian,
            };
            let want = x.rem_euclid(&n);
            let p = ModPolynomial::new(&[x.clone(), be, Number::from(1u8)], &n);
            assert_eq!(p.coeff(0), want);
            assert_eq!(p.coeff(1), want);
        }
    }
}

#[test]
fn powers_and_compositions_modulo_f_of_several_limbs() {
    // Past the crossovers of the transforms on pieces and of the remainders by their ready
    // transforms (49 coefficients and more in f): against products and remainders one at a time.
    let mut s = 0x6A09_E667_F3BC_C908;
    let moduli = limb_moduli();
    for n in moduli.iter().take(size(12, 3)) {
        for lf in [size(50, 5), size(97, 7)] {
            let bytes = (n.bits() as usize).div_ceil(8) + 1;
            let mut fc: Vec<Number> = (0..lf).map(|_| number(&mut s, bytes)).collect();
            fc[lf - 1] = Number::from(1);
            let f = ModPolynomial::new(&fc, n);
            let a = ModPolynomial::new(
                &(0..2 * lf)
                    .map(|_| number(&mut s, bytes))
                    .collect::<Vec<_>>(),
                n,
            );
            let e = Number::from(size(1000, 13) as u64);
            // a^e mod f by squares and products reduced one at a time.
            let mut want = ModPolynomial::one(n);
            let base = &a % &f;
            for i in (0..e.bits()).rev() {
                want = &(&want * &want) % &f;
                if e.bit(i) {
                    want = &(&want * &base) % &f;
                }
            }
            assert_eq!(a.pow_mod(&e, &f), want, "n = {n}, lf = {lf}");
            let g = ModPolynomial::new(
                &(0..lf + 3)
                    .map(|_| number(&mut s, bytes))
                    .collect::<Vec<_>>(),
                n,
            );
            let p = ModPolynomial::new(
                &(0..9).map(|_| number(&mut s, bytes)).collect::<Vec<_>>(),
                n,
            );
            assert_eq!(
                p.compose_mod(&g, &f),
                &p.compose(&g) % &f,
                "n = {n}, lf = {lf}"
            );
        }
    }
}

#[test]
fn newton_division_and_series_modulo_several_limbs() {
    // Past the schoolbook division (128 coefficients for 2 to 4 limbs, 32 beyond) and the
    // transforms on pieces: q d + r = a with deg r < deg d, and f g = 1 mod x^n.
    let mut s = 0xBB67_AE85_84CA_A73B;
    for n in limb_moduli().iter().take(size(12, 2)) {
        let bytes = (n.bits() as usize).div_ceil(8) + 1;
        for (la, ld) in [
            (size(300, 30), size(140, 12)),
            (size(420, 26), size(180, 14)),
        ] {
            let mut dc: Vec<Number> = (0..ld).map(|_| number(&mut s, bytes)).collect();
            dc[ld - 1] = Number::from(1);
            let d = ModPolynomial::new(&dc, n);
            let a = ModPolynomial::new(
                &(0..la).map(|_| number(&mut s, bytes)).collect::<Vec<_>>(),
                n,
            );
            let (q, r) = a.div_rem(&d);
            assert_eq!(&(&q * &d) + &r, a, "n = {n}, {la} by {ld}");
            assert!(r.length() < d.length());
            let mut fc = dc.clone();
            fc[0] = Number::from(1);
            let f = ModPolynomial::new(&fc, n);
            let g = f.inv_series(la);
            let one = (&f * &g).coeffs();
            assert_eq!(one[0], Number::from(1));
            assert!(
                one[1..la].iter().all(Number::is_zero),
                "n = {n}, series {la}"
            );
        }
    }
}
