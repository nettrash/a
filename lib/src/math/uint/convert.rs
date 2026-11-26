//! Conversions of the fixed-width integers: with the primitive integers, between widths, between
//! [`Uint`] and [`Int`], and with the floats.
//!
//! `From` is implemented where every width of one limb or more holds every value of the source,
//! as the standard library implements it only for conversions that cannot fail: into [`Uint`]
//! from `bool`, `u8`, `u16`, `u32`, `u64` and `usize`, into [`Int`] from `bool`, `i8` to `i64`,
//! `isize`, and `u8` to `u32`. The rest is `TryFrom`, with [`TryFromIntError`]: `u128` and the
//! signed types into `Uint`, `u64`, `usize`, `u128` and `i128` into `Int`, and both into every
//! primitive integer. (A width of zero limbs holds only zero, and its `From` panics for anything
//! else, as [`Uint::from_u64`] does.)

use super::{Int, Uint};
use core::fmt;
use core::num::IntErrorKind;

/// The error of a conversion of a fixed-width integer, or into one, that cannot hold the value:
/// the counterpart of the standard library's `TryFromIntError`. It tells whether the value was
/// above the target's range or below it, and says so in the words of current Rust's
/// `TryFromIntError` (Rust 1.100): "number too large to fit in target type" or "number too small
/// to fit in target type". Older releases of Rust, 1.88 among them, wrote "out of range integral
/// type conversion attempted" for both.
///
/// # Examples
///
/// ```
/// use a::math::uint::{Int, Uint};
/// use core::num::IntErrorKind;
///
/// let e = u8::try_from(Uint::<2>::from_u64(256)).unwrap_err();
/// assert_eq!(*e.kind(), IntErrorKind::PosOverflow);
/// assert_eq!(e.to_string(), "number too large to fit in target type");
/// let e = Uint::<2>::try_from(-1i32).unwrap_err();
/// assert_eq!(*e.kind(), IntErrorKind::NegOverflow);
/// assert_eq!(e.to_string(), "number too small to fit in target type");
/// assert!(Uint::<2>::try_from(Int::<2>::MINUS_ONE).is_err());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TryFromIntError {
    /// Whether the value was below the target's range (else above it). A `bool` rather than an
    /// `IntErrorKind`, which is neither `Copy` nor `Hash` in Rust 1.88.
    below: bool,
}

impl TryFromIntError {
    /// The value was above the target's range.
    const LARGE: Self = TryFromIntError { below: false };
    /// The value was below the target's range.
    const SMALL: Self = TryFromIntError { below: true };

    /// The error for a value of the sign `negative` out of range: below it if negative, above
    /// it if not.
    #[inline]
    const fn of_sign(negative: bool) -> Self {
        if negative { Self::SMALL } else { Self::LARGE }
    }

    /// Which way the value was out of range: [`IntErrorKind::PosOverflow`] above the target's
    /// range, [`IntErrorKind::NegOverflow`] below it (a negative value into an unsigned type
    /// included).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    /// use core::num::IntErrorKind;
    ///
    /// assert_eq!(*i8::try_from(Int::<2>::from_i64(-129)).unwrap_err().kind(), IntErrorKind::NegOverflow);
    /// assert_eq!(*i8::try_from(Int::<2>::from_i64(128)).unwrap_err().kind(), IntErrorKind::PosOverflow);
    /// ```
    #[inline]
    pub const fn kind(&self) -> &IntErrorKind {
        if self.below {
            &IntErrorKind::NegOverflow
        } else {
            &IntErrorKind::PosOverflow
        }
    }
}

impl fmt::Debug for TryFromIntError {
    /// Writes `TryFromIntError(PosOverflow)` or `TryFromIntError(NegOverflow)`, as the standard
    /// library's error does in current Rust.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TryFromIntError").field(self.kind()).finish()
    }
}

impl fmt::Display for TryFromIntError {
    /// Writes which way the value was out of range, in the words of current Rust's
    /// `TryFromIntError`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.below {
            "number too small to fit in target type"
        } else {
            "number too large to fit in target type"
        }
        .fmt(f)
    }
}

impl std::error::Error for TryFromIntError {}

impl<const N: usize> Uint<N> {
    /// The value as a `u128` if it fits.
    #[inline]
    const fn to_u128(self) -> Option<u128> {
        if self.bits() > 128 {
            return None;
        }
        let l = self.as_limbs();
        Some(match N {
            0 => 0,
            1 => l[0] as u128,
            _ => l[0] as u128 | (l[1] as u128) << 64,
        })
    }

    /// The same value in `M` limbs, the high limbs dropped or zeros added, as `as` converts
    /// between the unsigned primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<2>::from_limbs([1, 2]);
    /// assert_eq!(x.resize::<3>(), Uint::from_limbs([1, 2, 0]));
    /// assert_eq!(x.resize::<1>(), Uint::from_u64(1));
    /// ```
    #[inline]
    pub const fn resize<const M: usize>(self) -> Uint<M> {
        let mut limbs = [0; M];
        let l = self.as_limbs();
        let mut i = 0;
        while i < M && i < N {
            limbs[i] = l[i];
            i += 1;
        }
        Uint::from_limbs(limbs)
    }

    /// The same value in `M` limbs, or `None` if it needs more.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(5).try_resize::<1>(), Some(Uint::from_u64(5)));
    /// assert_eq!(Uint::<2>::from_limbs([1, 2]).try_resize::<1>(), None);
    /// ```
    #[inline]
    pub const fn try_resize<const M: usize>(self) -> Option<Uint<M>> {
        if self.bits() > 64 * M as u32 {
            None
        } else {
            Some(self.resize())
        }
    }

    /// The nearest `f64`, ties to even, as `as` converts the primitive integers; infinity when the
    /// value rounds beyond the largest `f64`.
    ///
    /// The top 64 bits, with a bit set at the bottom when any bit below them is (which can only
    /// matter when the top bits are halfway between two `f64`s), are converted by the hardware,
    /// and scaled by a power of two, which is exact: one rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.to_f64(), 2f64.powi(128));
    /// // 2^64 + 2^11 + 1 is above the halfway point 2^64 + 2^11 by the bit below the top 64.
    /// let x = Uint::<2>::from_limbs([(1 << 11) + 1, 1]);
    /// assert_eq!(x.to_f64(), 2f64.powi(64) + 2f64.powi(12));
    /// assert_eq!(Uint::<16>::MAX.to_f64(), f64::INFINITY);
    /// ```
    pub const fn to_f64(self) -> f64 {
        let bits = self.bits();
        if bits <= 64 {
            return if N == 0 {
                0.0
            } else {
                self.as_limbs()[0] as f64
            };
        }
        if bits > 1024 {
            return f64::INFINITY;
        }
        let (top, shift) = self.top_bits(bits);
        // 2^shift, from 2^1 to 2^960, exactly; a value rounding up to 2^1024 overflows to infinity
        // in the product, as it must.
        top as f64 * f64::from_bits(((shift + 1023) as u64) << 52)
    }

    /// The nearest `f32`, ties to even, as `as` converts the primitive integers; infinity when the
    /// value rounds beyond the largest `f32`. It is rounded once, from the value itself: rounding
    /// [`to_f64`](Self::to_f64) to an `f32` would round twice.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(1 << 40).to_f32(), 2f32.powi(40));
    /// assert_eq!(Uint::<2>::MAX.to_f32(), f32::INFINITY);
    /// ```
    pub const fn to_f32(self) -> f32 {
        let bits = self.bits();
        if bits <= 64 {
            return if N == 0 {
                0.0
            } else {
                self.as_limbs()[0] as f32
            };
        }
        if bits > 128 {
            return f32::INFINITY;
        }
        let (top, shift) = self.top_bits(bits);
        top as f32 * f32::from_bits((shift + 127) << 23)
    }

    /// For `bits > 64` significant bits: the top 64 bits, the lowest of them or'ed with every bit
    /// below them, and the number of bits below, `bits − 64`.
    #[inline]
    const fn top_bits(self, bits: u32) -> (u64, u32) {
        let shift = bits - 64;
        // The 64 bits from `shift` up straddle limbs i and i + 1 (which exists for s > 0, as the
        // top bit is at 64i + s + 63); the sticky bit is any set bit below them.
        let (i, s) = ((shift / 64) as usize, shift % 64);
        let l = self.as_limbs();
        let top = if s == 0 {
            l[i]
        } else {
            (l[i] >> s) | (l[i + 1] << (64 - s))
        };
        let mut sticky = s != 0 && l[i] & ((1u64 << s) - 1) != 0;
        let mut j = 0;
        while !sticky && j < i {
            sticky = l[j] != 0;
            j += 1;
        }
        (top | sticky as u64, shift)
    }

    /// The value of `x` truncated toward zero, as `as` converts a float to a primitive integer:
    /// held to `[0, MAX]`, and zero for a NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_f64(1e30), Uint::from_u128(1_000_000_000_000_000_019_884_624_838_656));
    /// assert_eq!(Uint::<2>::from_f64(2.9), Uint::from_u64(2));
    /// assert_eq!(Uint::<2>::from_f64(-5.0), Uint::ZERO);
    /// assert_eq!(Uint::<2>::from_f64(1e300), Uint::MAX);
    /// assert_eq!(Uint::<2>::from_f64(f64::NAN), Uint::ZERO);
    /// ```
    pub const fn from_f64(x: f64) -> Self {
        // Below 1, negative or a NaN.
        if x.is_nan() || x < 1.0 {
            return Self::ZERO;
        }
        let b = x.to_bits();
        let exp = ((b >> 52) & 0x7FF) as u32;
        // x ≥ 1 is infinite or normal, in [2^e, 2^(e+1)) with e = exp − 1023 ≥ 0.
        let e = exp - 1023;
        if exp == 0x7FF || e >= Self::BITS {
            return Self::MAX;
        }
        let mantissa = (b & ((1 << 52) - 1)) | (1 << 52);
        Self::from_mantissa(mantissa, e as i32 - 52)
    }

    /// The value of `x` truncated toward zero, as `as` converts a float to a primitive integer:
    /// held to `[0, MAX]`, and zero for a NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_f32(3.5), Uint::from_u64(3));
    /// assert_eq!(Uint::<2>::from_f32(f32::INFINITY), Uint::MAX);
    /// ```
    pub const fn from_f32(x: f32) -> Self {
        if x.is_nan() || x < 1.0 {
            return Self::ZERO;
        }
        let b = x.to_bits();
        let exp = (b >> 23) & 0xFF;
        let e = exp - 127;
        if exp == 0xFF || e >= Self::BITS {
            return Self::MAX;
        }
        let mantissa = ((b & ((1 << 23) - 1)) | (1 << 23)) as u64;
        Self::from_mantissa(mantissa, e as i32 - 23)
    }

    /// `⌊m · 2^k⌋` for `m < 2⁵³` and `k` with the product below `2^BITS`.
    #[inline]
    const fn from_mantissa(m: u64, k: i32) -> Self {
        if k <= 0 {
            Self::from_u64(m >> -k)
        } else {
            Self::from_u64(m).unbounded_shl(k as u32)
        }
    }
}

impl<const N: usize> Int<N> {
    /// The value as an `i128` if it fits.
    #[inline]
    const fn to_i128(self) -> Option<i128> {
        let m = self.unsigned_abs();
        if self.is_negative() {
            match m.to_u128() {
                Some(m) if m <= 1 << 127 => Some((m as i128).wrapping_neg()),
                _ => None,
            }
        } else {
            match m.to_u128() {
                Some(m) if m < 1 << 127 => Some(m as i128),
                _ => None,
            }
        }
    }

    /// The same value in `M` limbs, the high limbs dropped or the sign copied into new ones, as
    /// `as` converts between the signed primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<1>::from_i64(-2).resize::<2>(), Int::from_i64(-2));
    /// assert_eq!(Int::<2>::from_limbs([5, 1]).resize::<1>(), Int::from_i64(5));
    /// ```
    #[inline]
    pub const fn resize<const M: usize>(self) -> Int<M> {
        let fill = if self.is_negative() { u64::MAX } else { 0 };
        let mut limbs = [fill; M];
        let l = self.as_limbs();
        let mut i = 0;
        while i < M && i < N {
            limbs[i] = l[i];
            i += 1;
        }
        Int::from_limbs(limbs)
    }

    /// The same value in `M` limbs, or `None` if it is out of their range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-5).try_resize::<1>(), Some(Int::from_i64(-5)));
    /// assert_eq!(Int::<2>::from_limbs([1 << 63, 0]).try_resize::<1>(), None);
    /// ```
    #[inline]
    pub const fn try_resize<const M: usize>(self) -> Option<Int<M>> {
        let r = self.resize::<M>();
        // It fits exactly when sign-extending it back gives the value again.
        let back = r.resize::<N>();
        let (a, b) = (back.as_limbs(), self.as_limbs());
        let mut i = 0;
        while i < N {
            if a[i] != b[i] {
                return None;
            }
            i += 1;
        }
        Some(r)
    }

    /// The nearest `f64`, ties to even, as `as` converts the primitive integers; an infinity when
    /// the value rounds beyond the largest `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.to_f64(), -(2f64.powi(127)));
    /// assert_eq!(Int::<2>::from_i64(-3).to_f64(), -3.0);
    /// ```
    #[inline]
    pub const fn to_f64(self) -> f64 {
        let m = self.unsigned_abs().to_f64();
        if self.is_negative() { -m } else { m }
    }

    /// The nearest `f32`, ties to even, rounded once, as `as` converts the primitive integers; an
    /// infinity when the value rounds beyond the largest `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.to_f32(), -(2f32.powi(127)));
    /// assert_eq!(Int::<3>::MIN.to_f32(), f32::NEG_INFINITY);
    /// ```
    #[inline]
    pub const fn to_f32(self) -> f32 {
        let m = self.unsigned_abs().to_f32();
        if self.is_negative() { -m } else { m }
    }

    /// The value of `x` truncated toward zero, as `as` converts a float to a primitive integer:
    /// held to `[MIN, MAX]`, and zero for a NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_f64(-2.9), Int::from_i64(-2));
    /// assert_eq!(Int::<2>::from_f64(-1e300), Int::MIN);
    /// assert_eq!(Int::<2>::from_f64(f64::NAN), Int::ZERO);
    /// ```
    pub const fn from_f64(x: f64) -> Self {
        let negative = x < 0.0;
        let m = Uint::<N>::from_f64(if negative { -x } else { x });
        Self::saturate(negative, m)
    }

    /// The value of `x` truncated toward zero, as `as` converts a float to a primitive integer:
    /// held to `[MIN, MAX]`, and zero for a NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_f32(-3.5), Int::from_i64(-3));
    /// assert_eq!(Int::<2>::from_f32(f32::INFINITY), Int::MAX);
    /// ```
    pub const fn from_f32(x: f32) -> Self {
        let negative = x < 0.0;
        let m = Uint::<N>::from_f32(if negative { -x } else { x });
        Self::saturate(negative, m)
    }

    /// The integer of the sign and the magnitude `m`, held to `[MIN, MAX]`.
    #[inline]
    const fn saturate(negative: bool, m: Uint<N>) -> Self {
        let top = m.bit(Self::BITS - 1);
        if negative {
            // Up to 2^(64N − 1), which is MIN itself.
            if top {
                Self::MIN
            } else {
                m.wrapping_neg().cast_signed()
            }
        } else if top {
            Self::MAX
        } else {
            m.cast_signed()
        }
    }
}

impl<const N: usize> TryFrom<Int<N>> for Uint<N> {
    type Error = TryFromIntError;

    /// The same value, or an error when it is negative.
    #[inline]
    fn try_from(x: Int<N>) -> Result<Self, TryFromIntError> {
        if x.is_negative() {
            Err(TryFromIntError::SMALL)
        } else {
            Ok(x.cast_unsigned())
        }
    }
}

impl<const N: usize> TryFrom<Uint<N>> for Int<N> {
    type Error = TryFromIntError;

    /// The same value, or an error when it is above `Int::MAX`.
    #[inline]
    fn try_from(x: Uint<N>) -> Result<Self, TryFromIntError> {
        let y = x.cast_signed();
        if y.is_negative() {
            Err(TryFromIntError::LARGE)
        } else {
            Ok(y)
        }
    }
}

impl<const N: usize> From<bool> for Uint<N> {
    /// 1 for `true`, 0 for `false`.
    ///
    /// # Panics
    ///
    /// For `true` if `N` is 0.
    #[inline]
    #[track_caller]
    fn from(b: bool) -> Self {
        Uint::from_u64(b as u64)
    }
}

impl<const N: usize> From<bool> for Int<N> {
    /// 1 for `true`, 0 for `false`.
    ///
    /// # Panics
    ///
    /// For `true` if `N` is 0.
    #[inline]
    #[track_caller]
    fn from(b: bool) -> Self {
        Int::from_i64(b as i64)
    }
}

/// `From` a primitive integer that every width of one limb holds, through `u64` or `i64`.
macro_rules! from_small {
    ($t:ident $via:ident: $($p:ty),*) => {$(
        impl<const N: usize> From<$p> for $t<N> {
            #[doc = concat!("The integer `x`, as [`", stringify!($t), "::", stringify!($via), "`] gives it.")]
            ///
            /// # Panics
            ///
            /// If `N` is 0 and `x` is not.
            #[inline]
            #[track_caller]
            fn from(x: $p) -> Self {
                $t::$via(x as _)
            }
        }
    )*};
}

from_small!(Uint from_u64: u8, u16, u32, usize);
from_small!(Int from_i64: i8, i16, i32, isize, u8, u16, u32);

impl<const N: usize> From<u64> for Uint<N> {
    /// The integer `x`, as [`Uint::from_u64`] gives it.
    #[inline]
    #[track_caller]
    fn from(x: u64) -> Self {
        Self::from_u64(x)
    }
}

impl<const N: usize> From<i64> for Int<N> {
    /// The integer `x`, as [`Int::from_i64`] gives it.
    #[inline]
    #[track_caller]
    fn from(x: i64) -> Self {
        Self::from_i64(x)
    }
}

impl<const N: usize> TryFrom<u128> for Uint<N> {
    type Error = TryFromIntError;

    /// The integer `x`, or an error when it does not fit (only for `N ≤ 1`).
    #[inline]
    fn try_from(x: u128) -> Result<Self, TryFromIntError> {
        if 128 - x.leading_zeros() > Self::BITS {
            Err(TryFromIntError::LARGE)
        } else {
            Ok(Uint::from_u128(x))
        }
    }
}

/// `TryFrom` a signed primitive integer into `Uint`, through `i128`.
macro_rules! uint_from_signed {
    ($($p:ty),*) => {$(
        impl<const N: usize> TryFrom<$p> for Uint<N> {
            type Error = TryFromIntError;

            /// The integer `x`, or an error when it is negative or does not fit.
            #[inline]
            fn try_from(x: $p) -> Result<Self, TryFromIntError> {
                if x < 0 {
                    Err(TryFromIntError::SMALL)
                } else {
                    Uint::try_from(x as u128)
                }
            }
        }
    )*};
}

uint_from_signed!(i8, i16, i32, i64, i128, isize);

/// `TryFrom` a primitive integer into `Int` that a narrow width may not hold, through `i128` or
/// `u128`.
macro_rules! int_from_wide {
    ($($p:ty),*) => {$(
        impl<const N: usize> TryFrom<$p> for Int<N> {
            type Error = TryFromIntError;

            /// The integer `x`, or an error when it is out of range.
            #[inline]
            fn try_from(x: $p) -> Result<Self, TryFromIntError> {
                // As a sign and a magnitude below 2^128.
                let (negative, m) = if (x as i128) < 0 && <$p>::MIN != 0 {
                    (true, (x as i128).unsigned_abs())
                } else {
                    (false, x as u128)
                };
                let m = Uint::<N>::try_from(m).map_err(|_| TryFromIntError::of_sign(negative))?;
                let top = m.bit(Self::BITS.wrapping_sub(1));
                let fits = if negative {
                    !top || m.trailing_zeros() == Self::BITS - 1
                } else {
                    !top
                };
                if fits {
                    Ok(Int::from_sign_magnitude(negative, m))
                } else {
                    Err(TryFromIntError::of_sign(negative))
                }
            }
        }
    )*};
}

int_from_wide!(u64, usize, u128, i128);

/// `TryFrom` a fixed-width integer into each primitive integer, through `u128` or `i128`.
macro_rules! into_primitive {
    ($($p:ty),*) => {$(
        impl<const N: usize> TryFrom<Uint<N>> for $p {
            type Error = TryFromIntError;

            #[doc = concat!("The value as a `", stringify!($p), "`, or an error when it is out of the type's range.")]
            #[inline]
            fn try_from(x: Uint<N>) -> Result<$p, TryFromIntError> {
                x.to_u128()
                    .and_then(|v| <$p>::try_from(v).ok())
                    .ok_or(TryFromIntError::LARGE)
            }
        }

        impl<const N: usize> TryFrom<Int<N>> for $p {
            type Error = TryFromIntError;

            #[doc = concat!("The value as a `", stringify!($p), "`, or an error when it is out of the type's range.")]
            #[inline]
            fn try_from(x: Int<N>) -> Result<$p, TryFromIntError> {
                // Out of range on the side of its sign.
                x.to_i128()
                    .and_then(|v| <$p>::try_from(v).ok())
                    .ok_or(TryFromIntError::of_sign(x.is_negative()))
            }
        }
    )*};
}

into_primitive!(u8, u16, u32, u64, usize, i8, i16, i32, i64, i128, isize);

impl<const N: usize> TryFrom<Uint<N>> for u128 {
    type Error = TryFromIntError;

    /// The value as a `u128`, or an error when it needs more than 128 bits.
    #[inline]
    fn try_from(x: Uint<N>) -> Result<u128, TryFromIntError> {
        x.to_u128().ok_or(TryFromIntError::LARGE)
    }
}

impl<const N: usize> TryFrom<Int<N>> for u128 {
    type Error = TryFromIntError;

    /// The value as a `u128`, or an error when it is negative or needs more than 128 bits.
    #[inline]
    fn try_from(x: Int<N>) -> Result<u128, TryFromIntError> {
        if x.is_negative() {
            Err(TryFromIntError::SMALL)
        } else {
            x.cast_unsigned().to_u128().ok_or(TryFromIntError::LARGE)
        }
    }
}
