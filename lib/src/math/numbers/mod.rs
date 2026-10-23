//! Integers of any size: a [`Number`] holds the sign of an integer and the bytes of its
//! absolute value, in the [`ByteOrder`] it was created with.

mod byte_order;
mod number;

pub use byte_order::ByteOrder;
pub use number::Number;
