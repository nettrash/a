//! The named prime fields as [`TypedPrime`](super::TypedPrime)s: the parameter types, one for
//! each of [`PrimeField`](super::PrimeField)'s named fields but P-521's, and the element types,
//! [`TypedFp`] of each.
//!
//! # Examples
//!
//! ```
//! use a::math::field::named::{Bls12_381BaseFp, GoldilocksFp};
//!
//! let x = GoldilocksFp::from_u64(u64::MAX); // 2^64 - 1 = 2^32 - 2 modulo the prime
//! assert_eq!(x, GoldilocksFp::from_u64((1 << 32) - 2));
//! assert_eq!(Bls12_381BaseFp::bits(), 381);
//! ```

use super::TypedFp;

crate::typed_prime! {
    /// BN254's (alt_bn128's) base prime, `p` of [`PrimeField::bn254_base`](super::PrimeField::bn254_base).
    pub Bn254Base, 4, "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
}
crate::typed_prime! {
    /// BN254's group order, of [`PrimeField::bn254_scalar`](super::PrimeField::bn254_scalar).
    pub Bn254Scalar, 4, "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
}
crate::typed_prime! {
    /// BLS12-381's base prime, of [`PrimeField::bls12_381_base`](super::PrimeField::bls12_381_base).
    pub Bls12_381Base, 6, "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
}
crate::typed_prime! {
    /// BLS12-381's group order, of
    /// [`PrimeField::bls12_381_scalar`](super::PrimeField::bls12_381_scalar).
    pub Bls12_381Scalar, 4, "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
}
crate::typed_prime! {
    /// secp256k1's base prime `2^256 − 2^32 − 977`, of
    /// [`PrimeField::secp256k1_base`](super::PrimeField::secp256k1_base).
    pub Secp256k1Base, 4, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
}
crate::typed_prime! {
    /// secp256k1's group order, of
    /// [`PrimeField::secp256k1_scalar`](super::PrimeField::secp256k1_scalar).
    pub Secp256k1Scalar, 4, "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
}
crate::typed_prime! {
    /// P-256's base prime, of [`PrimeField::p256_base`](super::PrimeField::p256_base): its
    /// products take P-256's reduction on aarch64, as the runtime field's.
    pub P256Base, 4, "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"
}
crate::typed_prime! {
    /// P-256's group order, of [`PrimeField::p256_scalar`](super::PrimeField::p256_scalar).
    pub P256Scalar, 4, "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"
}
crate::typed_prime! {
    /// P-384's base prime, of [`PrimeField::p384_base`](super::PrimeField::p384_base): its
    /// products take P-384's reduction on aarch64, as the runtime field's.
    pub P384Base, 6, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff"
}
crate::typed_prime! {
    /// P-384's group order, of [`PrimeField::p384_scalar`](super::PrimeField::p384_scalar).
    pub P384Scalar, 6, "ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973"
}
crate::typed_prime! {
    /// `2^255 − 19`, Curve25519's base prime, of
    /// [`PrimeField::curve25519_base`](super::PrimeField::curve25519_base).
    pub Curve25519Base, 4, "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed"
}
crate::typed_prime! {
    /// `2^252 + 27742317777372353535851937790883648493`, the order of Ed25519's group, of
    /// [`PrimeField::curve25519_scalar`](super::PrimeField::curve25519_scalar).
    pub Curve25519Scalar, 4, "1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed"
}
crate::typed_prime! {
    /// The Goldilocks prime `2^64 − 2^32 + 1`.
    pub Goldilocks, 1, "ffffffff00000001"
}

/// An element of BN254's base field.
pub type Bn254BaseFp = TypedFp<Bn254Base, 4>;
/// An element of BN254's scalar field.
pub type Bn254ScalarFp = TypedFp<Bn254Scalar, 4>;
/// An element of BLS12-381's base field.
pub type Bls12_381BaseFp = TypedFp<Bls12_381Base, 6>;
/// An element of BLS12-381's scalar field.
pub type Bls12_381ScalarFp = TypedFp<Bls12_381Scalar, 4>;
/// An element of secp256k1's base field.
pub type Secp256k1BaseFp = TypedFp<Secp256k1Base, 4>;
/// An element of secp256k1's scalar field.
pub type Secp256k1ScalarFp = TypedFp<Secp256k1Scalar, 4>;
/// An element of P-256's base field.
pub type P256BaseFp = TypedFp<P256Base, 4>;
/// An element of P-256's scalar field.
pub type P256ScalarFp = TypedFp<P256Scalar, 4>;
/// An element of P-384's base field.
pub type P384BaseFp = TypedFp<P384Base, 6>;
/// An element of P-384's scalar field.
pub type P384ScalarFp = TypedFp<P384Scalar, 6>;
/// An element of `GF(2^255 − 19)`.
pub type Curve25519BaseFp = TypedFp<Curve25519Base, 4>;
/// An element of the field of Ed25519's group order.
pub type Curve25519ScalarFp = TypedFp<Curve25519Scalar, 4>;
/// An element of the Goldilocks field.
pub type GoldilocksFp = TypedFp<Goldilocks, 1>;
