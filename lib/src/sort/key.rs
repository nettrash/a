//! The key types of the sorts that do not compare elements: [`counting`](super::counting),
//! [`radix`](super::radix) and [`bucket`](super::bucket) sort.

/// A key the sorts that do not compare elements can sort by: a primitive integer, `char` or
/// `bool`.
///
/// It is sealed: it cannot be implemented outside this crate.
pub trait IntegerKey: sealed::Key {}

/// A key bucket sort can sort by: a primitive integer, `char`, `bool`, `f32` or `f64`.
///
/// Floats are ordered as [`f64::total_cmp`] orders them: negative NaNs, −∞, the negative
/// numbers, −0, +0, the positive numbers, +∞, positive NaNs.
///
/// It is sealed: it cannot be implemented outside this crate.
pub trait BucketKey: sealed::Bucket {}

pub(super) mod sealed {
    /// What the sorts need to know about a key type.
    pub trait Key: Copy + Ord {
        /// The key as an unsigned integer of the same width that orders as the key does; as a
        /// key itself, it is its own bits.
        type Bits: Bits + super::IntegerKey;
        /// The key as [`Self::Bits`]: its bits, with the sign bit flipped for a signed type.
        fn bits(self) -> Self::Bits;
        /// `self − min`, for `min <= self`.
        fn distance(self, min: Self) -> u128;
        /// `self − min`, for `min <= self` and a distance that fits in `usize`.
        fn offset(self, min: Self) -> usize;
        /// `min + offset`, or any value of the type if that is not one.
        fn nth(min: Self, offset: usize) -> Self;
    }

    /// An unsigned integer, read as digits of a few bits.
    pub trait Bits: Copy + Ord {
        const ZERO: Self;
        /// The width in bits.
        const BITS: u32;
        fn or(self, other: Self) -> Self;
        fn xor(self, other: Self) -> Self;
        fn leading_zeros(self) -> u32;
        fn trailing_zeros(self) -> u32;
        /// The bits from `shift` on, masked by `mask`.
        fn digit(self, shift: u32, mask: usize) -> usize;
        /// `n`, which is less than `2^BITS`.
        fn of(n: usize) -> Self;
        fn wrapping_sub(self, other: Self) -> Self;
        fn shr(self, shift: u32) -> Self;
        /// The low 64 bits.
        fn low64(self) -> u64;
    }

    /// What bucket sort needs to know about a key type.
    pub trait Bucket: Copy {
        /// The key as an unsigned integer of the same size and alignment that orders as the key
        /// does.
        type Bits: Bits + super::IntegerKey;
        /// Whether a key's bytes, read as [`Self::Bits`], already are [`Self::ordered`].
        const IDENTITY: bool;
        /// Whether the key is a float, whose buckets are cut by value.
        const FLOAT: bool;
        /// The key as [`Self::Bits`].
        fn ordered(self) -> Self::Bits;
        /// The key whose [`Self::ordered`] bits are `bits`.
        fn from_ordered(bits: Self::Bits) -> Self;
        /// The value of the float whose [`Self::ordered`] bits are `bits`; `0.0` for other keys.
        fn value(bits: Self::Bits) -> f64;
    }
}

macro_rules! bits {
    ($($u:ty),*) => {$(
        impl sealed::Bits for $u {
            const ZERO: Self = 0;
            const BITS: u32 = <$u>::BITS;
            #[inline(always)]
            fn or(self, other: Self) -> Self {
                self | other
            }
            #[inline(always)]
            fn xor(self, other: Self) -> Self {
                self ^ other
            }
            #[inline(always)]
            fn leading_zeros(self) -> u32 {
                <$u>::leading_zeros(self)
            }
            #[inline(always)]
            fn trailing_zeros(self) -> u32 {
                <$u>::trailing_zeros(self)
            }
            #[inline(always)]
            fn digit(self, shift: u32, mask: usize) -> usize {
                (self >> shift) as usize & mask
            }
            #[inline(always)]
            fn of(n: usize) -> Self {
                n as $u
            }
            #[inline(always)]
            fn wrapping_sub(self, other: Self) -> Self {
                <$u>::wrapping_sub(self, other)
            }
            #[inline(always)]
            fn shr(self, shift: u32) -> Self {
                self >> shift
            }
            #[inline(always)]
            fn low64(self) -> u64 {
                self as u64
            }
        }
    )*};
}

bits!(u8, u16, u32, u64, u128, usize);

macro_rules! integer_key {
    ($($t:ty => $u:ty, $sign:expr);* $(;)?) => {$(
        impl sealed::Key for $t {
            type Bits = $u;
            #[inline(always)]
            fn bits(self) -> $u {
                (self as $u) ^ $sign
            }
            #[inline(always)]
            fn distance(self, min: Self) -> u128 {
                (self as $u).wrapping_sub(min as $u) as u128
            }
            #[inline(always)]
            fn offset(self, min: Self) -> usize {
                (self as $u).wrapping_sub(min as $u) as usize
            }
            #[inline(always)]
            fn nth(min: Self, offset: usize) -> Self {
                (min as $u).wrapping_add(offset as $u) as $t
            }
        }
        impl IntegerKey for $t {}
        impl sealed::Bucket for $t {
            type Bits = $u;
            const IDENTITY: bool = {
                let sign: $u = $sign;
                sign == 0
            };
            const FLOAT: bool = false;
            #[inline(always)]
            fn ordered(self) -> $u {
                sealed::Key::bits(self)
            }
            #[inline(always)]
            fn from_ordered(bits: $u) -> Self {
                (bits ^ $sign) as $t
            }
            #[inline(always)]
            fn value(_: $u) -> f64 {
                0.0
            }
        }
        impl BucketKey for $t {}
    )*};
}

integer_key! {
    u8 => u8, 0; u16 => u16, 0; u32 => u32, 0; u64 => u64, 0; u128 => u128, 0;
    usize => usize, 0;
    i8 => u8, 1 << 7; i16 => u16, 1 << 15; i32 => u32, 1 << 31; i64 => u64, 1 << 63;
    i128 => u128, 1 << 127; isize => usize, 1 << (usize::BITS - 1);
}

impl sealed::Key for char {
    type Bits = u32;
    #[inline(always)]
    fn bits(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn distance(self, min: Self) -> u128 {
        u128::from(self as u32 - min as u32)
    }
    #[inline(always)]
    fn offset(self, min: Self) -> usize {
        (self as u32).wrapping_sub(min as u32) as usize
    }
    #[inline(always)]
    fn nth(min: Self, offset: usize) -> Self {
        // An offset into the surrogates, which no element has, gives `min`.
        char::from_u32((min as u32).wrapping_add(offset as u32)).unwrap_or(min)
    }
}
impl IntegerKey for char {}

impl sealed::Bucket for char {
    type Bits = u32;
    const IDENTITY: bool = true;
    const FLOAT: bool = false;
    #[inline(always)]
    fn ordered(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn from_ordered(bits: u32) -> Self {
        char::from_u32(bits).unwrap_or_default()
    }
    #[inline(always)]
    fn value(_: u32) -> f64 {
        0.0
    }
}
impl BucketKey for char {}

impl sealed::Key for bool {
    type Bits = u8;
    #[inline(always)]
    fn bits(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn distance(self, min: Self) -> u128 {
        u128::from(self as u8 - min as u8)
    }
    #[inline(always)]
    fn offset(self, min: Self) -> usize {
        usize::from(self).wrapping_sub(usize::from(min))
    }
    #[inline(always)]
    fn nth(min: Self, offset: usize) -> Self {
        usize::from(min) + offset != 0
    }
}
impl IntegerKey for bool {}

impl sealed::Bucket for bool {
    type Bits = u8;
    const IDENTITY: bool = true;
    const FLOAT: bool = false;
    #[inline(always)]
    fn ordered(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn from_ordered(bits: u8) -> Self {
        bits != 0
    }
    #[inline(always)]
    fn value(_: u8) -> f64 {
        0.0
    }
}
impl BucketKey for bool {}

macro_rules! float_key {
    ($($f:ty => $u:ty, $i:ty);* $(;)?) => {$(
        impl sealed::Bucket for $f {
            type Bits = $u;
            const IDENTITY: bool = false;
            const FLOAT: bool = true;
            /// The bits with the sign bit flipped for a positive float and every bit flipped for
            /// a negative one, which order as [`f64::total_cmp`] does.
            #[inline(always)]
            fn ordered(self) -> $u {
                let u = self.to_bits();
                u ^ ((((u as $i) >> (<$u>::BITS - 1)) as $u) | (1 << (<$u>::BITS - 1)))
            }
            #[inline(always)]
            fn from_ordered(bits: $u) -> Self {
                let flip = ((((!bits) as $i) >> (<$u>::BITS - 1)) as $u) | (1 << (<$u>::BITS - 1));
                <$f>::from_bits(bits ^ flip)
            }
            #[inline(always)]
            fn value(bits: $u) -> f64 {
                f64::from(<Self as sealed::Bucket>::from_ordered(bits))
            }
        }
        impl BucketKey for $f {}
    )*};
}

float_key! { f32 => u32, i32; f64 => u64, i64 }
