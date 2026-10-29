//! Integers of any size: a [`Number`] holds the sign of an integer and the bytes of its
//! absolute value, in the [`ByteOrder`] it was created with.
//!
//! Numbers add with `+` and `+=` and subtract with `-` and `-=`, each operand owned or
//! borrowed, whatever their signs and byte orders: the result is in the shortest form and in the
//! byte order of the left operand, and it is built in the vector of an operand taken by value,
//! or of the target of `+=` or `-=`, whenever that vector has room for it (the left operand's
//! first).

mod add;
mod byte_order;
mod number;
mod sub;

pub use byte_order::ByteOrder;
pub use number::Number;
