//! The key types of the sorts that do not compare elements: [`counting`](super::counting) and
//! [`radix`](super::radix) sort.

/// A key the sorts that do not compare elements can sort by: a primitive integer, `char` or
/// `bool`.
///
/// It is sealed: it cannot be implemented outside this crate.
pub trait IntegerKey: sealed::Key {}

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
