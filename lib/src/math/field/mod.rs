//! Finite fields: prime fields `GF(p)`, their extensions `GF(pᵏ)` and the towers
//! `F_p² → F_p⁶ → F_p¹²` of pairings, and binary fields `GF(2ᵐ)`, with `GF(2⁸)` and `GF(2¹²⁸)` as
//! AES and GCM use them.
//!
//! * [`PrimeField`] and [`Fp`]: `GF(p)` for a prime of any length, in a form of arithmetic chosen
//!   from the shape and the length of `p`, the forms of the fields of
//!   [`a::math::ec`](crate::math::ec) among them; square roots by a method chosen from `p`
//!   (a power, Atkin's, Tonelli and Shanks's, or Cipolla and Lehmer's), the Legendre symbol,
//!   inversion in batches by Montgomery's trick, encodings of a fixed length, and the fields of
//!   BN254, BLS12-381 and the named curves ready made.
//! * [`ExtensionField`] and [`Fpk`]: `GF(pᵏ) = GF(p)[x]/(g)` for a monic irreducible `g`, given
//!   and checked by Ben-Or's test, or the smallest of its degree; with the Frobenius map, norms,
//!   traces, square roots, multiplicative orders and primitive elements.
//! * [`Tower`], [`Fp2`], [`Fp6`] and [`Fp12`]: the towers `F_p² = F_p[u]/(u² − β)`,
//!   `F_p⁶ = F_p²[v]/(v³ − ξ)` and `F_p¹² = F_p⁶[w]/(w² − v)` that
//!   [`a::math::pairing`](crate::math::pairing) computes in, BN254's and BLS12-381's ready made
//!   ([`Tower::bn254`], [`Tower::bls12_381`]).
//! * [`BinaryField`] and [`F2m`]: `GF(2ᵐ) = GF(2)[x]/(f)` for an irreducible `f`, checked by
//!   Rabin's test, in the polynomial basis of SEC 1, multiplied by the CPU's carry-less product
//!   (`PMULL` or `PCLMULQDQ`, see [`Clmul`]) or a portable kernel; and [`NormalBasis`] and
//!   [`NormalElement`], the Gaussian normal bases of FIPS 186-4, with [`BasisConversion`] between
//!   a normal and a polynomial basis.
//! * [`Gf2_8`] and [`Gf2_128`]: AES's and GCM's fields as small `Copy` types, with the AES S-box
//!   and GCM's GHASH.
//!
//! A field is an object built once, such as [`PrimeField::new`], [`ExtensionField::new`],
//! [`Tower::new`] or [`BinaryField::new`], which checks what it is given and chooses the form of
//! its arithmetic; its elements borrow it, as the points of [`a::math::ec`](crate::math::ec)
//! borrow their curve, and combine with `+`, `-`, `*` and `/`. Elements of different fields do not
//! mix: an operation on two of them panics.
//!
//! Like the rest of the library this is not constant-time: it is for mathematics and for
//! verifying with public data, not for secret keys.
//!
//! # Examples
//!
//! ```
//! use a::math::field::PrimeField;
//! use a::math::numbers::Number;
//!
//! let f = PrimeField::new(&Number::from(101u8));
//! let x = f.from_u64(7);
//! let y = &x * &x + &x; // 56
//! assert_eq!(y.value(), Number::from(56u8));
//! assert_eq!(&y / &x * &x, y);
//! let r = f.from_u64(5).sqrt().unwrap(); // 45² = 2025 = 20·101 + 5
//! assert_eq!(&r * &r, f.from_u64(5));
//! ```

mod binary;
mod binary_clmul;
mod binary_inv;
mod binary_normal;
mod binary_poly;
mod ext;
mod ext_arith;
mod ext_find;
mod ext_maps;
mod ext_roots;
mod ext_zech;
mod gf2_128;
#[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
mod gf2_128_pmull;
mod gf2_8;
pub mod named;
mod prime;
mod prime_chains;
mod prime_form;
mod prime_inv;
mod prime_sqrt;
mod prime_wide;
mod tower;
mod tower_fast;
mod tower_fp12;
mod tower_fp2;
mod tower_fp6;
mod typed;

pub use binary::{BinaryField, Clmul, F2m};
pub use binary_normal::{BasisConversion, NormalBasis, NormalElement};
pub use ext::{ExtensionField, Fpk};
pub use gf2_8::Gf2_8;
pub use gf2_128::Gf2_128;
pub use prime::{Fp, PrimeField};
pub use tower::Tower;
pub use tower_fp2::Fp2;
pub use tower_fp6::Fp6;
pub use tower_fp12::Fp12;
pub(crate) use typed::TypedKernel;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
pub(crate) use typed::TypedSpecial;
pub use typed::{TypedFp, TypedPrime, limbs_from_hex};
