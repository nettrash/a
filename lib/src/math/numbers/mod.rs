//! Integers of any size: a [`Number`] holds the sign of an integer and the bytes of its
//! absolute value, in the [`ByteOrder`] it was created with.
//!
//! Numbers add with `+` and `+=`, subtract with `-` and `-=` and multiply with `*` and `*=`,
//! each operand owned or borrowed, whatever their signs and byte orders: the result is in the
//! shortest form and in the byte order of the left operand, and it is built in the vector of an
//! operand taken by value, or of the target of `+=`, `-=` or `*=`, whenever that vector has room
//! for it (the left operand's first).
//!
//! They divide with `/` and `/=`, truncated toward zero, and take the remainder with `%` and `%=`,
//! which has the sign of the dividend, as Rust's signed integers do, and dividing by zero panics;
//! [`Number::div_rem`] gives both, and [`Number::div_euclid`] and [`Number::rem_euclid`] Euclid's
//! division, whose remainder is never negative. A [`Divisor`] prepares a number to divide many
//! others by, [`Number::div_exact`] divides by a number known to divide exactly, and
//! [`Number::is_multiple_of`] tells whether it does.
//!
//! They shift by a `usize` number of bits with `<<`, `>>`, `<<=` and `>>=`, owned or borrowed, as
//! Rust's signed integers do but with no bits lost at the top: `a << k` is `a · 2^k`, and `a >> k`
//! is `a / 2^k` rounded towards negative infinity. The result keeps the byte order of the number
//! shifted, and is built in its vector, when it is owned, whenever that has room for it.
//!
//! They are written as text in decimal by `Display`, in hexadecimal, octal and binary by the other
//! formatting traits, and in any radix from 2 to 36 by [`Number::to_str_radix`], and read by
//! [`Number::from_str_radix`]: the sign and the digits of the magnitude.
//!
//! They combine bit by bit with `&`, `|`, `^`, `&=`, `|=` and `^=`, owned or borrowed, and invert
//! with `!`, as if they were in two's complement with infinitely many bits, as Rust's signed
//! integers are on their width: `!a` is `-a - 1`. The result is in the shortest form and in the
//! byte order of the left operand, built in an operand's vector as a sum is. [`Number::bits`],
//! [`Number::count_ones`] and [`Number::trailing_zeros`] read the bits of the magnitude, and
//! [`Number::bit`] and [`Number::set_bit`] one bit in two's complement.
//!
//! They compare and hash by value, whatever their byte orders, with `==`, `<`, `Ord` and `Hash`,
//! also with a primitive integer on the right; they clone, print with `Debug`, default to zero,
//! negate with `-`, and give [`Number::abs`], [`Number::signum`], [`Number::is_zero`] and
//! [`Number::is_positive`]. They convert from every primitive integer with `From`, back with
//! `TryFrom` ([`TryFromNumberError`] when out of range), to the nearest float with
//! [`Number::to_f64`] and [`Number::to_f32`], and from decimal text with `str::parse`; iterators of
//! them add up with `Sum` and multiply with `Product`. The arithmetic operators take a primitive
//! integer on either side, and [`Number::div_floor`] and [`Number::rem_floor`] round the quotient
//! toward negative infinity.
//!
//! [`Number::gcd`] gives their greatest common divisor, [`Number::lcm`] their least common
//! multiple, [`Number::gcd_ext`] the greatest common divisor with GMP's cofactors, and
//! [`Number::mod_inverse`] an inverse modulo a number, by Lehmer's method and Möller's half gcd.
//!
//! They have integer roots, truncated toward zero: [`Number::isqrt`] and [`Number::isqrt_rem`],
//! the square root rounded down and the remainder, and [`Number::nth_root`], the root of any
//! degree. [`Number::factorial`] and [`Number::binomial`] build factorials and binomial
//! coefficients.
//!
//! They are raised to a `u32` power by [`Number::pow`], as `i32::pow` does but exactly, and to the
//! power of a number modulo another by [`Number::pow_mod`], whose result is Euclid's remainder, never
//! negative. [`Number::is_probable_prime`] tells whether a number is prime by the Baillie–PSW test,
//! exact below `2^64`, and [`Number::next_prime`] gives the next probable prime.

mod add;
mod bits;
mod byte_order;
mod cmp;
mod convert;
mod div;
mod divisor;
mod factorial;
mod gcd;
mod iter;
mod mul;
mod number;
mod pow;
mod prim;
mod prime;
mod radix;
mod root;
mod shift;
mod sign;
mod sub;

pub use byte_order::ByteOrder;
pub use convert::TryFromNumberError;
pub use divisor::Divisor;
pub use number::Number;
pub use radix::{NumberErrorKind, ParseNumberError};
