//! The constants at every width, the smallest widths (one limb, and zero limbs, which hold only
//! zero), the larger ones, and what can be computed in a constant.

use super::common::outcome;
use a::math::numbers::Number;
use a::math::uint::{Int, Uint};
use std::collections::HashSet;

fn constants<const N: usize>() {
    let bits = 64 * N as u32;
    assert_eq!(Uint::<N>::BITS, bits);
    assert_eq!(Int::<N>::BITS, bits);
    let full = Number::from(1u64) << bits as usize;
    let half = Number::from(1u64) << (bits - 1) as usize;
    assert_eq!(Number::from(Uint::<N>::MAX), &full - Number::from(1u64));
    assert_eq!(Number::from(Uint::<N>::MIN), Number::from(0u64));
    assert_eq!(Number::from(Uint::<N>::ONE), Number::from(1u64));
    assert_eq!(Number::from(Int::<N>::MAX), &half - Number::from(1u64));
    assert_eq!(Number::from(Int::<N>::MIN), -&half);
    assert_eq!(Number::from(Int::<N>::MINUS_ONE), Number::from(-1i64));
    assert_eq!(Uint::<N>::default(), Uint::ZERO);
    assert_eq!(Int::<N>::default(), Int::ZERO);
    assert_eq!(Uint::<N>::MAX.count_ones(), bits);
    assert_eq!(Int::<N>::MIN.leading_zeros(), 0);
    assert_eq!(Int::<N>::MAX.leading_zeros(), 1);
    type I<const N: usize> = Int<N>;
    assert!(I::<N>::MIN < I::<N>::MINUS_ONE && I::<N>::MINUS_ONE < I::<N>::ZERO);
    assert!(I::<N>::ZERO < I::<N>::MAX);
    assert!(Uint::<N>::ZERO < Uint::<N>::ONE && Uint::<N>::ONE < Uint::<N>::MAX);
    // Hashes agree with equality.
    let set: HashSet<Uint<N>> = [Uint::ONE, Uint::from_u64(1), Uint::MAX]
        .into_iter()
        .collect();
    assert_eq!(set.len(), 2);
    let set: HashSet<Int<N>> = [Int::MIN, Int::from_limbs(*Int::<N>::MIN.as_limbs())]
        .into_iter()
        .collect();
    assert_eq!(set.len(), 1);
}

#[test]
fn the_constants_are_right_at_every_width() {
    constants::<1>();
    constants::<2>();
    constants::<3>();
    constants::<4>();
    constants::<5>();
    constants::<6>();
    constants::<7>();
    constants::<8>();
    constants::<9>();
    constants::<16>();
    constants::<32>();
    constants::<64>();
}

#[test]
fn zero_limbs_hold_only_zero() {
    type Z = Uint<0>;
    assert_eq!(Z::BITS, 0);
    assert_eq!(Z::ZERO, Z::MAX);
    assert_eq!(Z::from_u64(0), Z::ZERO);
    assert!(outcome(|| Z::from_u64(1)).is_err());
    assert!(outcome(|| Int::<0>::from_i64(-1)).is_err());
    assert_eq!(Z::ZERO.overflowing_add(Z::ZERO), (Z::ZERO, false));
    assert_eq!(Z::ZERO.wrapping_mul(Z::ZERO), Z::ZERO);
    assert_eq!(Z::ZERO.bits(), 0);
    assert_eq!(Z::ZERO.to_string(), "0");
    assert_eq!(Z::ZERO.to_f64(), 0.0);
    assert_eq!("0".parse::<Z>(), Ok(Z::ZERO));
    assert!("1".parse::<Z>().is_err());
    assert_eq!("-0".parse::<Int<0>>(), Ok(Int::<0>::ZERO));
    assert_eq!(Z::try_from(Number::from(0u64)), Ok(Z::ZERO));
    assert!(Z::try_from(Number::from(1u64)).is_err());
    assert_eq!(Number::from(Z::ZERO), Number::from(0u64));
    assert_eq!(Number::from(Int::<0>::ZERO), Number::from(0u64));
    assert_eq!(Z::from_le_slice(&[0, 0]), Some(Z::ZERO));
    assert_eq!(Z::from_le_slice(&[1]), None);
    assert_eq!(Uint::<3>::MAX.resize::<0>(), Z::ZERO);
    assert_eq!(Z::ZERO.resize::<2>(), Uint::ZERO);
    assert_eq!(Z::ZERO.overflowing_shl(0), (Z::ZERO, true));
    assert_eq!(Z::ZERO.unbounded_shr(0), Z::ZERO);
}

#[test]
fn the_large_widths_work_as_the_small_ones() {
    // 4096 bits: a power, a square root and a division that undoes a product.
    let three = Uint::<64>::from_u64(3);
    let p = three.pow(2000);
    assert_eq!(Number::from(p), Number::from(3u64).pow(2000));
    assert_eq!(p.isqrt(), three.pow(1000));
    let q = Uint::<64>::from_u64(7).pow(1100) + Uint::from_u64(12345);
    let (d, r) = p.div_rem(q);
    assert_eq!(d * q + r, p);
    assert!(r < q);
    assert_eq!(Number::from(d), Number::from(p) / Number::from(q));
    assert_eq!(p.ilog(three), 2000);
    assert_eq!(p.ilog10(), 954);
    let m = Int::<64>::MIN;
    assert_eq!(m.to_string().len(), 1234);
    assert_eq!(m.to_string().parse::<Int<64>>(), Ok(m));
}

#[test]
fn methods_compute_in_constants() {
    const X: Uint<4> = Uint::from_u64(10).pow(70);
    const Q: (Uint<4>, Uint<4>) = X.div_rem(Uint::from_u128(u128::MAX));
    const S: Uint<4> = X.isqrt();
    const L: u32 = X.ilog10();
    const B: [u8; 32] = X.to_be_bytes();
    const F: f64 = X.to_f64();
    const M: Int<4> = Int::from_i64(-7).pow(3).wrapping_mul(Int::MAX);
    const R: Int<4> = Int::from_i64(-7).rem_euclid(Int::from_i64(3));
    const H: Option<Uint<4>> = Uint::from_be_slice(&[1, 2, 3]);
    assert_eq!(Number::from(X), Number::from(10u64).pow(70));
    assert_eq!(Q.0 * Uint::from_u128(u128::MAX) + Q.1, X);
    assert_eq!(S, Uint::from_u64(10).pow(35));
    assert_eq!(L, 70);
    assert_eq!(Uint::from_be_bytes(B), X);
    assert_eq!(F, 1e70);
    assert_eq!(M, Int::from_i64(-343).wrapping_mul(Int::MAX));
    assert_eq!(R, Int::from_i64(2));
    assert_eq!(H, Some(Uint::from_u64(0x010203)));
}

#[test]
fn bytes_of_every_length_and_order() {
    let x = Uint::<2>::from_u128(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
    let le: [u8; 16] = x.to_le_bytes();
    assert_eq!(
        le,
        0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128.to_le_bytes()
    );
    let be: [u8; 16] = x.to_be_bytes();
    assert_eq!(
        be,
        0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10u128.to_be_bytes()
    );
    assert!(outcome(|| x.write_le_bytes(&mut [0u8; 15])).is_err());
    assert!(outcome(|| x.write_be_bytes(&mut [0u8; 17])).is_err());
    assert_eq!(Uint::<2>::from_be_slice(&[]), Some(Uint::ZERO));
    assert_eq!(Int::<2>::from_be_slice(&[]), Some(Int::ZERO));
    assert_eq!(Int::<2>::from_be_slice(&[0x80]), Some(Int::from_i64(-128)));
    assert_eq!(
        Int::<2>::from_be_slice(&[0, 0x80]),
        Some(Int::from_i64(128))
    );
    // Twenty bytes, big-endian: −129 sign-extended over them fits; with a byte beyond the low
    // sixteen that is not the sign's fill, or a top bit within them against the sign, it does
    // not.
    let mut long = vec![0xFF; 20];
    long[19] = 0x7F;
    assert_eq!(Int::<2>::from_be_slice(&long), Some(Int::from_i64(-129)));
    long[0] = 0xFE;
    assert_eq!(Int::<2>::from_be_slice(&long), None);
    long[0] = 0xFF;
    long[3] = 0x7F;
    assert_eq!(Int::<2>::from_be_slice(&long), None);
    long[3] = 0xFF;
    long[4] = 0x7F;
    assert_eq!(Int::<2>::from_be_slice(&long), None);
    long[4] = 0xFF;
    long[19] = 0xFE;
    assert_eq!(Int::<2>::from_be_slice(&long), Some(Int::from_i64(-2)));
    assert_eq!(
        Int::<1>::from_le_slice(&(-5i64).to_le_bytes()),
        Some(Int::from_i64(-5))
    );
    assert_eq!(
        Int::<1>::from_le_slice(&(-5i128).to_le_bytes()),
        Some(Int::from_i64(-5))
    );
    assert_eq!(
        Int::<1>::from_le_slice(&(i64::MIN as i128 - 1).to_le_bytes()),
        None
    );
    assert_eq!(
        Int::<1>::from_le_slice(&(i64::MAX as i128 + 1).to_le_bytes()),
        None
    );
}
