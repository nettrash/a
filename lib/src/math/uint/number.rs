//! Conversions between the fixed-width integers and [`Number`].
//!
//! A number is read by its significant bytes, in its own byte order, whatever zero bytes it holds
//! above them and whether its zero is marked negative (a `Number` built by hand may have either):
//! as the comparisons of numbers, a conversion goes by the value.

use super::{Int, Uint};
use crate::math::numbers::{ByteOrder, Magnitude, Number, TryFromNumberError};

/// The little-endian bytes of `limbs`.
/// The number of the limbs `l`, the low one first.
pub(super) fn number_of_limbs(l: &[u64]) -> Number {
    Number::new(bytes_of(l), ByteOrder::LittleEndian)
}

fn bytes_of(limbs: &[u64]) -> Vec<u8> {
    limbs.iter().flat_map(|l| l.to_le_bytes()).collect()
}

/// The magnitude of `m` as `N` limbs, or `None` when it needs more.
fn limbs_of<const N: usize>(m: &Magnitude) -> Option<[u64; N]> {
    let bytes = m.bytes;
    if bytes.len() > 8 * N {
        return None;
    }
    let mut limbs = [0u64; N];
    let n = bytes.len();
    for i in 0..n {
        // The i-th byte from the least significant.
        let b = if m.big_endian {
            bytes[n - 1 - i]
        } else {
            bytes[i]
        };
        limbs[i / 8] |= u64::from(b) << (8 * (i % 8));
    }
    Some(limbs)
}

impl<const N: usize> From<Uint<N>> for Number {
    /// The same integer as a number, little-endian.
    fn from(x: Uint<N>) -> Number {
        Number::new(bytes_of(x.as_limbs()), ByteOrder::LittleEndian)
    }
}

impl<const N: usize> From<&Uint<N>> for Number {
    /// The same integer as a number, little-endian.
    fn from(x: &Uint<N>) -> Number {
        Number::from(*x)
    }
}

impl<const N: usize> TryFrom<&Number> for Uint<N> {
    type Error = TryFromNumberError;

    /// The same integer, or an error when it is negative or needs more than `N` limbs.
    fn try_from(n: &Number) -> Result<Self, Self::Error> {
        let m = Magnitude::of(n);
        if m.negative {
            return Err(TryFromNumberError::new());
        }
        limbs_of::<N>(&m)
            .map(Uint::from_limbs)
            .ok_or(TryFromNumberError::new())
    }
}

impl<const N: usize> TryFrom<Number> for Uint<N> {
    type Error = TryFromNumberError;

    /// The same integer, or an error when it is negative or needs more than `N` limbs.
    fn try_from(n: Number) -> Result<Self, Self::Error> {
        Uint::try_from(&n)
    }
}

impl<const N: usize> From<Int<N>> for Number {
    /// The same integer as a number, little-endian.
    fn from(x: Int<N>) -> Number {
        Number::new_signed(
            bytes_of(x.unsigned_abs().as_limbs()),
            x.is_negative(),
            ByteOrder::LittleEndian,
        )
    }
}

impl<const N: usize> From<&Int<N>> for Number {
    /// The same integer as a number, little-endian.
    fn from(x: &Int<N>) -> Number {
        Number::from(*x)
    }
}

impl<const N: usize> TryFrom<&Number> for Int<N> {
    type Error = TryFromNumberError;

    /// The same integer, or an error when it is outside `[MIN, MAX]`.
    fn try_from(n: &Number) -> Result<Self, Self::Error> {
        let m = Magnitude::of(n);
        let abs = limbs_of::<N>(&m)
            .map(Uint::from_limbs)
            .ok_or(TryFromNumberError::new())?;
        // At most 2^(64N − 1) for a negative number, one less for a positive one.
        let top = abs.bit(Self::BITS.wrapping_sub(1));
        let fits = !top || (m.negative && abs.trailing_zeros() == Self::BITS.wrapping_sub(1));
        if fits {
            Ok(Int::from_sign_magnitude(m.negative, abs))
        } else {
            Err(TryFromNumberError::new())
        }
    }
}

impl<const N: usize> TryFrom<Number> for Int<N> {
    type Error = TryFromNumberError;

    /// The same integer, or an error when it is outside `[MIN, MAX]`.
    fn try_from(n: Number) -> Result<Self, Self::Error> {
        Int::try_from(&n)
    }
}
