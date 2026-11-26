//! Fixed-width integers: [`Uint<N>`], an unsigned integer of `N` 64-bit limbs (`64·N` bits), and
//! [`Int<N>`], a signed one of the same width in two's complement, as `i64` is to `u64`. Both live
//! on the stack and are `Copy`.
//!
//! They behave as a `u{64·N}` and an `i{64·N}` would, method for method where the method makes
//! sense: the operators `+ - * / %`, `<< >>`, `& | ^ !` (and unary `-` for `Int`) with owned and
//! borrowed operands and their assignment forms, `Sum` and `Product`; the `checked_*`,
//! `wrapping_*`, `overflowing_*` and `saturating_*` forms of the arithmetic; `pow`, `isqrt`, the
//! integer logarithms, the bit counts, rotations and byte orders. As for the primitive integers
//! under the default profiles, the arithmetic operators and the shifts panic on overflow when
//! `debug_assertions` are on and wrap when they are off; division and remainder by zero always
//! panic, and so do `Int::MIN / -1` and `Int::MIN % -1`. The panic and error messages are those
//! of the primitives in current Rust (1.100); some of them changed between releases (`pow`'s
//! overflow and `TryFromIntError`'s text differ in Rust 1.88, for one).
//!
//! The arithmetic is the schoolbook one, as in the other fixed-width libraries (ruint,
//! crypto-bigint, bnum): sums limb by limb with the carry, products in `O(N²)` limb products, and
//! quotients by Knuth's algorithm D (*The Art of Computer Programming*, vol. 2, 4.3.1), each
//! quotient digit the quotient of three limbs by the divisor's top two, taken with its reciprocal
//! (Möller and Granlund, *Improved division by invariant integers*, 2011), and for a divisor of one
//! limb by a loop of 2-by-1 divisions with its reciprocal; square roots by Zimmermann's Karatsuba
//! square root, about the cost of one such division.
//!
//! They convert to and from the primitive integers (`From` where every width holds the value,
//! `TryFrom` with [`TryFromIntError`] elsewhere), to and from [`Number`](crate::math::numbers::Number)
//! with `From` and `TryFrom`, between widths with `resize` and `try_resize`, and to the nearest
//! `f64` and `f32`. They are written in decimal by `Display`, in hexadecimal, octal and binary by
//! the other formatting traits, honouring the formatter's flags as the primitive integers do, in
//! any radix from 2 to 36 by `to_str_radix`, and read by `from_str_radix` and `str::parse`, with
//! [`ParseIntError`] and the standard library's [`IntErrorKind`](core::num::IntErrorKind).
//! [`Montgomery`] is the arithmetic modulo an odd number of the same width in Montgomery form.
//!
//! Nothing here is constant-time: the time of an operation can depend on the values.

/// The value of an overflowing operation `$e`, a pair `(value, overflowed)`, as the operators of
/// the primitive integers give it under the default profiles: a panic with `$msg` on overflow when
/// debug assertions are on, the wrapped value when they are off.
macro_rules! debug_checked {
    ($e:expr, $msg:literal) => {{
        let (value, overflow) = $e;
        if overflow && cfg!(debug_assertions) {
            panic!($msg)
        }
        value
    }};
}

mod convert;
mod limbs;
mod mont;
mod number;
mod ops;
mod signed;
mod text;
mod unsigned;

pub use convert::TryFromIntError;
pub use mont::Montgomery;
pub use signed::Int;
pub use text::ParseIntError;
pub use unsigned::Uint;
