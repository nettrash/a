//! Elliptic curves over prime fields, with coordinates and scalars that are
//! [`Number`](crate::math::numbers::Number)s.
//!
//! * [`Weierstrass`], [`Montgomery`] and [`Edwards`]: the three forms of curve, each built from its
//!   coefficients and a prime of any length. What they share is the [`Curve`] trait: the points
//!   from their coordinates or their bytes, the discriminant, the `j`-invariant and the sum of many
//!   products.
//! * [`Point`]: a point of a curve, which adds with `+` and `+=`, subtracts with `-` and `-=`,
//!   negates with `-` and multiplies by a number with `*` and `*=`.
//! * [`Domain`]: a curve with a generator, its order and its cofactor; [`named`] holds those in
//!   common use: secp256k1, P-256, P-384, P-521, Curve25519 and Ed25519.
//!
//! **Nothing here is constant-time**: these are calculations on curves, and the time they take
//! depends on the scalars, as with [`pow_mod`](crate::math::numbers::Number::pow_mod), so they are
//! not meant for secret scalars such as private keys.

mod curve;
mod domain;
mod edwards;
mod field;
mod group;
mod montgomery;
pub mod named;
mod point;
mod weierstrass;

pub use curve::{Curve, CurveError, NotOnCurve};
pub use domain::Domain;
pub use edwards::Edwards;
pub use montgomery::Montgomery;
pub use point::{Point, Prepared};
pub use weierstrass::Weierstrass;
