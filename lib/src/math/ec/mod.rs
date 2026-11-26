//! Elliptic curves over prime fields, with coordinates and scalars that are
//! [`Number`](crate::math::numbers::Number)s, and in [`binary`] over binary fields, with
//! coordinates that are [`F2m`](crate::math::field::F2m) elements; and in [`hash`], hashing to
//! curves as RFC 9380 specifies.
//!
//! * [`Weierstrass`], [`Montgomery`] and [`Edwards`]: the three forms of curve, each built from its
//!   coefficients and a prime of any length. What they share is the [`Curve`] trait: the points
//!   from their coordinates or their bytes, the discriminant, the `j`-invariant and the sum of many
//!   products.
//! * [`Point`]: a point of a curve, which adds with `+` and `+=`, subtracts with `-` and `-=`,
//!   negates with `-` and multiplies by a number with `*` and `*=`.
//! * [`Domain`]: a curve with a generator, its order and its cofactor; [`named`] holds those in
//!   common use: secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519.
//! * [`binary`]: curves over binary fields, `y² + x·y = x³ + a·x² + b` over `GF(2ᵐ)`, as
//!   [`BinaryCurve`], [`BinaryPoint`] and [`BinaryDomain`], with the 18 curves of SEC 2 in
//!   [`binary::named`].
//! * [`hash`]: hashing to curves as RFC 9380 specifies, its twelve suites for P-256, P-384, P-521,
//!   secp256k1, curve25519 and edwards25519 ready made, with
//!   [`hash_to_field`](hash::hash_to_field) and the maps they are made of, for other curves too:
//!   Shallue–van de Woestijne, simplified SWU, Elligator 2 and isogenies.
//!
//! **Nothing here is constant-time**: these are calculations on curves, and the time they take
//! depends on the scalars, as with [`pow_mod`](crate::math::numbers::Number::pow_mod), so they are
//! not meant for secret scalars such as private keys.

pub mod binary;
mod curve;
mod domain;
mod edwards;
pub(crate) mod field;
mod group;
pub mod hash;
mod montgomery;
pub mod named;
mod point;
mod weierstrass;

pub use binary::{BinaryCurve, BinaryDomain, BinaryPoint, BinaryPrepared};
pub use curve::{Curve, CurveError, NotOnCurve};
pub use domain::Domain;
pub use edwards::Edwards;
pub use montgomery::Montgomery;
pub use point::{Point, Prepared};
pub use weierstrass::Weierstrass;
