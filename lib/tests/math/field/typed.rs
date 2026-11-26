#![allow(
    clippy::op_ref,
    reason = "the operators on borrowed operands are tested"
)]
// The typed prime fields against the runtime ones of the same primes (`PrimeField`, `Fp`) and
// against `Number`: every operation on seeded and edge inputs, for the named fields and for
// declared primes of 1 to 9 limbs that cover the kernels (no-carry, with carries, P-256, P-384,
// the numbers' kernel from seven limbs) and the methods of square roots (p ≡ 3 mod 4, 5 mod 8,
// Tonelli and Shanks's with s from 4 to 32).

use crate::allocations::count;
use a::math::field::named::{
    Bls12_381Base, Bls12_381Scalar, Bn254Base, Bn254Scalar, Curve25519Base, Curve25519Scalar,
    Goldilocks, P256Base, P256Scalar, P384Base, P384Scalar, Secp256k1Base, Secp256k1Scalar,
};
use a::math::field::{PrimeField, TypedFp, TypedPrime, limbs_from_hex};
use a::math::numbers::{ByteOrder, Number};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

a::typed_prime! {
    /// 2^61 - 1: one limb, p = 3 mod 4.
    M61, 1, "1fffffffffffffff"
}
a::typed_prime! {
    /// 65537: one limb, s = 16.
    F4, 1, "10001"
}
a::typed_prime! {
    /// 2^127 - 1: two limbs.
    M127, 2, "7fffffffffffffffffffffffffffffff"
}
a::typed_prime! {
    /// P-192's prime: three limbs, no spare bit.
    P192, 3, "fffffffffffffffffffffffffffffffeffffffffffffffff"
}
a::typed_prime! {
    /// A prime of 190 bits, p = 5 mod 8.
    T190, 3, "224dbdcbaa39b07fc2be6b0536223d8d033779161f2e59d5"
}
a::typed_prime! {
    /// A prime of 300 bits, five limbs, s = 4.
    T300, 5, "860892f902bd23f0824128b2f330c5c7fd0a6a3a4506513270e269e0d37f2a74de452e6b671"
}
a::typed_prime! {
    /// A prime of 448 bits, seven limbs, p = 5 mod 8.
    T448, 7, "860bb83956aeeb42207c9f6ca01235b86a643531b7daea11369ee14508ad794c24fd4172e5c69b8ec1d6023d7c13b2677bf2a7f582b85e85"
}
a::typed_prime! {
    /// A prime of 512 bits, eight limbs, s = 4.
    T512, 8, "a0f4661034ad3c4a12277c638d9be6bd35aab4840284f2906ba5736c107b1a519628ee8df90f202bb5944b503e84ed92b5b2f9eff82030504b06f39c4cdf3ff1"
}
a::typed_prime! {
    /// A prime of 576 bits, nine full limbs.
    T576, 9, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff89f"
}
a::typed_prime! {
    /// 2^521 - 1, nine limbs.
    M521, 9, "1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn number(&mut self, limbs: usize) -> Number {
        let b: Vec<u8> = (0..limbs).flat_map(|_| self.next().to_le_bytes()).collect();
        Number::new(b, ByteOrder::LittleEndian)
    }
}

fn hash<T: Hash>(x: &T) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

/// Every operation of `TypedFp<P, N>` against the runtime field, `per` seeded pairs and the edges;
/// the number of checks.
fn agree<P: TypedPrime<N>, const N: usize>(seed: u64, per: usize) -> u64 {
    type T<P, const N: usize> = TypedFp<P, N>;
    let p = T::<P, N>::modulus();
    assert!(T::<P, N>::modulus_is_prime(), "{} is prime", P::NAME);
    let f = PrimeField::new(&p);
    let one = Number::from(1u8);
    let mut rng = Rng(seed);
    let mut values: Vec<Number> = vec![
        Number::from(0u8),
        one.clone(),
        &p - &one,
        &p - &Number::from(2u8),
        &(&p - &one) >> 1usize,
        &(&p + &one) >> 1usize,
        (&one << (64 * N)).rem_euclid(&p),
    ];
    for _ in 0..per {
        values.push(rng.number(N + 1).rem_euclid(&p));
    }
    let mut checks = 0u64;
    let n = values.len();
    for i in 0..n {
        let (xv, yv) = (&values[i], &values[(i * 7 + 3) % n]);
        let (x, y) = (T::<P, N>::from_number(xv), T::<P, N>::from_number(yv));
        let (fx, fy) = (f.element(xv), f.element(yv));
        assert_eq!(x.value(), *xv, "{}", P::NAME);
        assert_eq!((x + y).value(), (&fx + &fy).value(), "{} +", P::NAME);
        assert_eq!((x - y).value(), (&fx - &fy).value(), "{} -", P::NAME);
        assert_eq!((x * y).value(), (&fx * &fy).value(), "{} *", P::NAME);
        assert_eq!((&x * &y), (x * &y));
        assert_eq!((&x + y), (x + &y));
        assert_eq!((-x).value(), (-&fx).value(), "{} neg", P::NAME);
        assert_eq!(-&x, -x);
        assert_eq!(
            x.square().value(),
            fx.square().value(),
            "{} square",
            P::NAME
        );
        assert_eq!(
            x.double().value(),
            fx.double().value(),
            "{} double",
            P::NAME
        );
        let mut a = x;
        a += y;
        a -= x;
        a *= x;
        assert_eq!(a, y * x);
        a *= &y;
        a += &x;
        a -= &y;
        assert_eq!(a, y * x * y + x - y);
        match (x.inverse(), fx.inverse()) {
            (Some(i), Some(fi)) => {
                assert_eq!(i.value(), fi.value(), "{} inverse", P::NAME);
                assert_eq!((y / x).value(), (&fy / &fx).value());
                let mut d = y;
                d /= x;
                assert_eq!(d * x, y);
            }
            (None, None) => assert!(x.is_zero()),
            _ => panic!("{} inverse of {xv}", P::NAME),
        }
        let e = rng.number(N);
        assert_eq!(x.pow(&e).value(), fx.pow(&e).value(), "{} pow", P::NAME);
        let long = rng.number(N + 2);
        assert_eq!(
            x.pow(&long).value(),
            fx.pow(&long).value(),
            "{} long pow",
            P::NAME
        );
        if !x.is_zero() {
            assert_eq!(
                x.pow(&-&e).value(),
                fx.pow(&-&e).value(),
                "{} negative pow",
                P::NAME
            );
        }
        let w = rng.next();
        assert_eq!(
            x.pow_u64(w).value(),
            fx.pow_u64(w).value(),
            "{} pow_u64",
            P::NAME
        );
        assert_eq!(x.legendre(), fx.legendre(), "{} legendre", P::NAME);
        assert_eq!(x.is_square(), fx.is_square());
        assert_eq!(x.sgn0(), fx.sgn0());
        match (x.sqrt(), fx.sqrt()) {
            (Some(r), Some(fr)) => {
                assert_eq!(r.value(), fr.value(), "{} sqrt", P::NAME);
                assert_eq!(r.square(), x);
            }
            (None, None) => {}
            _ => panic!("{} sqrt of {xv}", P::NAME),
        }
        let s = x.square();
        let r = s.sqrt().expect("a square");
        assert_eq!(r.square(), s);
        assert_eq!(r.value(), f.element(&s.value()).sqrt().unwrap().value());
        // Encodings and conversions.
        assert_eq!(x.to_bytes_be(), fx.to_bytes_be(), "{} bytes", P::NAME);
        assert_eq!(x.to_bytes_le(), fx.to_bytes_le());
        assert_eq!(T::<P, N>::from_bytes_be(&fx.to_bytes_be()), Some(x));
        assert_eq!(T::<P, N>::from_bytes_le(&fx.to_bytes_le()), Some(x));
        assert_eq!(x.to_fp(&f), fx);
        assert_eq!(T::<P, N>::from_fp(&fx), x);
        assert_eq!(
            T::<P, N>::from_number(&(xv - &(&p * &Number::from(3u8)))),
            x
        );
        assert_eq!(T::<P, N>::from_number(&(xv + &(&p << 70usize))), x);
        assert_eq!(hash(&x), hash(&T::<P, N>::from_number(xv)));
        assert_eq!(format!("{x}"), format!("{xv}"));
        assert_eq!(x.is_zero(), fx.is_zero());
        assert_eq!(x.is_one(), fx.is_one());
        checks += 40;
    }
    // Batch inversion, zeros included.
    let xs: Vec<T<P, N>> = values.iter().map(T::<P, N>::from_number).collect();
    let inv = T::<P, N>::batch_invert(&xs);
    for (x, i) in xs.iter().zip(&inv) {
        assert_eq!(*i, x.inverse().unwrap_or(T::<P, N>::ZERO));
        checks += 1;
    }
    // Words, the constants and the encodings that are not elements.
    for w in [0u64, 1, 2, 3, 0xffff_ffff, u64::MAX] {
        assert_eq!(T::<P, N>::from_u64(w).value(), f.from_u64(w).value());
        checks += 1;
    }
    assert_eq!(T::<P, N>::ZERO.value(), Number::from(0u8));
    assert_eq!(T::<P, N>::ONE.value(), one);
    assert_eq!(T::<P, N>::default(), T::<P, N>::ZERO);
    let mut pb = p.value.clone();
    if p.byte_order == ByteOrder::BigEndian {
        pb.reverse();
    }
    pb.resize(T::<P, N>::byte_len(), 0);
    assert_eq!(T::<P, N>::from_bytes_le(&pb), None, "p itself");
    assert_eq!(T::<P, N>::from_bytes_le(&pb[1..]), None, "a short encoding");
    assert_eq!(T::<P, N>::byte_len(), f.byte_len());
    assert_eq!(u64::from(T::<P, N>::bits()), p.bits());
    assert_eq!(T::<P, N>::modulus_limbs().to_vec(), {
        let mut l: Vec<u64> = pb
            .chunks(8)
            .map(|c| {
                let mut w = [0u8; 8];
                w[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(w)
            })
            .collect();
        l.resize(N, 0);
        l
    });
    checks + 8
}

#[test]
fn the_named_typed_fields_agree_with_the_runtime_ones() {
    let per = if cfg!(miri) { 1 } else { 24 };
    let mut n = 0;
    n += agree::<Bn254Base, 4>(1, per);
    n += agree::<Bn254Scalar, 4>(2, per);
    n += agree::<Bls12_381Base, 6>(3, per);
    n += agree::<Bls12_381Scalar, 4>(4, per);
    n += agree::<Secp256k1Base, 4>(5, per);
    n += agree::<Secp256k1Scalar, 4>(6, per);
    n += agree::<P256Base, 4>(7, per);
    n += agree::<P256Scalar, 4>(8, per);
    n += agree::<P384Base, 6>(9, per);
    n += agree::<P384Scalar, 6>(10, per);
    n += agree::<Curve25519Base, 4>(11, per);
    n += agree::<Curve25519Scalar, 4>(12, per);
    n += agree::<Goldilocks, 1>(13, per);
    assert!(n > 0);
}

#[test]
fn declared_typed_fields_of_one_to_nine_limbs_agree_with_the_runtime_ones() {
    let per = if cfg!(miri) { 1 } else { 16 };
    let mut n = 0;
    n += agree::<M61, 1>(21, per);
    n += agree::<F4, 1>(22, per);
    n += agree::<M127, 2>(23, per);
    n += agree::<P192, 3>(24, per);
    n += agree::<T190, 3>(25, per);
    if !cfg!(miri) {
        n += agree::<T300, 5>(26, per);
        n += agree::<T448, 7>(27, per);
        n += agree::<T512, 8>(28, per);
        n += agree::<T576, 9>(29, per);
        n += agree::<M521, 9>(30, per);
    }
    assert!(n > 0);
}

#[test]
fn typed_arithmetic_does_not_allocate() {
    type F = TypedFp<Bls12_381Base, 6>;
    let (x, y) = (F::from_u64(12345), F::from_u64(678));
    let (r, n) = count(|| {
        let mut a = x * y + x - y;
        a = a.square().double();
        a *= x;
        a += y;
        a -= x;
        (
            -a,
            a == y,
            a.is_zero(),
            a.is_one(),
            F::from_u64(7),
            hash(&a),
        )
    });
    assert_eq!(n, 0);
    assert!(!r.1);
}

#[test]
fn limbs_from_hex_reads_the_digits() {
    const A: [u64; 2] = limbs_from_hex("0x1_0000_0000_0000_0003");
    assert_eq!(A, [3, 1]);
    const B: [u64; 1] = limbs_from_hex("FFFFffff00000001");
    assert_eq!(B, [0xffff_ffff_0000_0001]);
    assert_eq!(
        <Goldilocks as TypedPrime<1>>::MODULUS,
        [0xffff_ffff_0000_0001]
    );
    assert_eq!(<Goldilocks as TypedPrime<1>>::NAME, "Goldilocks");
}

#[test]
#[should_panic(expected = "another prime")]
fn a_runtime_field_of_another_prime_panics() {
    let _ = TypedFp::<Bn254Base, 4>::ONE.to_fp(PrimeField::bn254_scalar());
}

#[test]
#[should_panic(expected = "division by zero")]
fn typed_division_by_zero_panics() {
    let _ = TypedFp::<Goldilocks, 1>::ONE / TypedFp::<Goldilocks, 1>::ZERO;
}

#[test]
fn elements_of_two_to_the_k_minus_c_compare_and_hash_by_their_value() {
    // 2^255 - 19 and secp256k1's prime take the folds by 2^256 mod p on aarch64, whose elements are
    // any four limbs congruent to the value: equality, hashing, the encodings and the predicates go
    // by the value.
    fn check<P: TypedPrime<4>>() {
        type T<P> = TypedFp<P, 4>;
        let p = T::<P>::modulus();
        let top = T::<P>::from_number(&(&p - &Number::from(1u8)));
        let zero = top + T::<P>::ONE;
        assert_eq!(zero, T::<P>::ZERO);
        assert!(zero.is_zero());
        assert_eq!(hash(&zero), hash(&T::<P>::ZERO));
        assert_eq!(zero.value(), Number::from(0u8));
        assert_eq!(zero.to_bytes_be(), vec![0u8; T::<P>::byte_len()]);
        let one = top * top;
        assert!(one.is_one());
        assert_eq!(one, T::<P>::ONE);
        assert_eq!(hash(&one), hash(&T::<P>::ONE));
        let two = -top - top;
        assert_eq!(two, T::<P>::from_u64(2));
        assert_eq!(two.inverse().unwrap() * two, T::<P>::ONE);
        assert_eq!(
            (top.double() + T::<P>::from_u64(2)).value(),
            Number::from(0u8)
        );
        assert_eq!(T::<P>::from_u64(4).sqrt(), Some(T::<P>::from_u64(2)));
    }
    check::<Curve25519Base>();
    check::<Secp256k1Base>();
    check::<Bn254Base>();
}
