//! The operators of the fixed-width integers: `+ - * / %`, `& | ^ !`, `<< >>` by any primitive
//! integer, unary `-` for [`Int`], each with owned and borrowed operands and its assignment form,
//! and `Sum` and `Product`.
//!
//! They behave as those of the primitive integers under the default profiles: `+`, `-`, `*`,
//! unary `-` and the shifts panic on overflow when `debug_assertions` are on (as `overflow-checks`
//! is by default) and wrap when they are off, `cfg!(debug_assertions)` in this crate deciding;
//! `/` and `%` by zero always panic, as do `Int::MIN / −1` and `Int::MIN % −1`. A shift by an
//! amount outside `0..BITS` panics with debug assertions, and without them shifts by the amount
//! modulo `BITS` (Euclid's remainder, so `−1` is `BITS − 1`): for a width that is a power of two
//! this is the primitives' masking of the amount's low bits. The panic messages are those of the
//! primitives in current Rust (1.100): "attempt to add with overflow" and so on, and "attempt to
//! exponentiate with overflow" for `pow`, which Rust 1.88 still wrote as "attempt to multiply
//! with overflow".
//!
//! As a shift is implemented for every primitive integer type of amount, the type of an amount
//! written as a bare literal is only settled at the end, by the fallback to `i32`: a method called
//! on the result before that needs the literal's type, `(x << 3u32).count_ones()`, where
//! `x << 3` alone, or compared or assigned, is fine.

use super::{Int, Uint};
use core::iter::{Product, Sum};
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

impl<const N: usize> Uint<N> {
    #[inline]
    #[track_caller]
    fn op_add(self, rhs: Self) -> Self {
        debug_checked!(self.overflowing_add(rhs), "attempt to add with overflow")
    }

    #[inline]
    #[track_caller]
    fn op_sub(self, rhs: Self) -> Self {
        debug_checked!(
            self.overflowing_sub(rhs),
            "attempt to subtract with overflow"
        )
    }

    #[inline]
    #[track_caller]
    fn op_mul(self, rhs: Self) -> Self {
        debug_checked!(
            self.overflowing_mul(rhs),
            "attempt to multiply with overflow"
        )
    }

    #[inline]
    #[track_caller]
    fn op_div(self, rhs: Self) -> Self {
        self.div_rem(rhs).0
    }

    #[inline]
    #[track_caller]
    fn op_rem(self, rhs: Self) -> Self {
        self.rem_nonzero(rhs)
    }

    #[inline]
    fn op_and(self, rhs: Self) -> Self {
        let (a, b) = (self.as_limbs(), rhs.as_limbs());
        Uint::from_limbs(core::array::from_fn(|i| a[i] & b[i]))
    }

    #[inline]
    fn op_or(self, rhs: Self) -> Self {
        let (a, b) = (self.as_limbs(), rhs.as_limbs());
        Uint::from_limbs(core::array::from_fn(|i| a[i] | b[i]))
    }

    #[inline]
    fn op_xor(self, rhs: Self) -> Self {
        let (a, b) = (self.as_limbs(), rhs.as_limbs());
        Uint::from_limbs(core::array::from_fn(|i| a[i] ^ b[i]))
    }

    #[inline]
    #[track_caller]
    fn op_shl(self, (k, out): (u32, bool)) -> Self {
        if out && cfg!(debug_assertions) {
            panic!("attempt to shift left with overflow");
        }
        self.wrapping_shl(k)
    }

    #[inline]
    #[track_caller]
    fn op_shr(self, (k, out): (u32, bool)) -> Self {
        if out && cfg!(debug_assertions) {
            panic!("attempt to shift right with overflow");
        }
        self.wrapping_shr(k)
    }
}

impl<const N: usize> Int<N> {
    #[inline]
    #[track_caller]
    fn op_add(self, rhs: Self) -> Self {
        debug_checked!(self.overflowing_add(rhs), "attempt to add with overflow")
    }

    #[inline]
    #[track_caller]
    fn op_sub(self, rhs: Self) -> Self {
        debug_checked!(
            self.overflowing_sub(rhs),
            "attempt to subtract with overflow"
        )
    }

    #[inline]
    #[track_caller]
    fn op_mul(self, rhs: Self) -> Self {
        debug_checked!(
            self.overflowing_mul(rhs),
            "attempt to multiply with overflow"
        )
    }

    #[inline]
    #[track_caller]
    fn op_div(self, rhs: Self) -> Self {
        self.div_rem(rhs).0
    }

    #[inline]
    #[track_caller]
    fn op_rem(self, rhs: Self) -> Self {
        match self.overflowing_rem(rhs) {
            (r, false) => r,
            _ => panic!("attempt to calculate the remainder with overflow"),
        }
    }

    #[inline]
    fn op_and(self, rhs: Self) -> Self {
        Int::from_limbs((self.cast_unsigned() & rhs.cast_unsigned()).into_limbs())
    }

    #[inline]
    fn op_or(self, rhs: Self) -> Self {
        Int::from_limbs((self.cast_unsigned() | rhs.cast_unsigned()).into_limbs())
    }

    #[inline]
    fn op_xor(self, rhs: Self) -> Self {
        Int::from_limbs((self.cast_unsigned() ^ rhs.cast_unsigned()).into_limbs())
    }

    #[inline]
    #[track_caller]
    fn op_shl(self, (k, out): (u32, bool)) -> Self {
        if out && cfg!(debug_assertions) {
            panic!("attempt to shift left with overflow");
        }
        self.wrapping_shl(k)
    }

    #[inline]
    #[track_caller]
    fn op_shr(self, (k, out): (u32, bool)) -> Self {
        if out && cfg!(debug_assertions) {
            panic!("attempt to shift right with overflow");
        }
        self.wrapping_shr(k)
    }
}

/// A binary operator of a fixed-width type, with owned and borrowed operands on both sides, and
/// its assignment form with an owned or a borrowed right-hand side.
macro_rules! binary {
    ($t:ident, $op:ident, $method:ident, $assign:ident, $assign_method:ident, $imp:ident) => {
        impl<const N: usize> $op for $t<N> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: Self) -> Self {
                self.$imp(rhs)
            }
        }
        impl<const N: usize> $op<&$t<N>> for $t<N> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &Self) -> Self {
                self.$imp(*rhs)
            }
        }
        impl<const N: usize> $op<$t<N>> for &$t<N> {
            type Output = $t<N>;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: $t<N>) -> $t<N> {
                (*self).$imp(rhs)
            }
        }
        impl<const N: usize> $op<&$t<N>> for &$t<N> {
            type Output = $t<N>;
            #[inline]
            #[track_caller]
            fn $method(self, rhs: &$t<N>) -> $t<N> {
                (*self).$imp(*rhs)
            }
        }
        impl<const N: usize> $assign for $t<N> {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: Self) {
                *self = (*self).$imp(rhs);
            }
        }
        impl<const N: usize> $assign<&$t<N>> for $t<N> {
            #[inline]
            #[track_caller]
            fn $assign_method(&mut self, rhs: &Self) {
                *self = (*self).$imp(*rhs);
            }
        }
    };
}

/// All the binary operators of a fixed-width type.
macro_rules! binaries {
    ($t:ident) => {
        binary!($t, Add, add, AddAssign, add_assign, op_add);
        binary!($t, Sub, sub, SubAssign, sub_assign, op_sub);
        binary!($t, Mul, mul, MulAssign, mul_assign, op_mul);
        binary!($t, Div, div, DivAssign, div_assign, op_div);
        binary!($t, Rem, rem, RemAssign, rem_assign, op_rem);
        binary!($t, BitAnd, bitand, BitAndAssign, bitand_assign, op_and);
        binary!($t, BitOr, bitor, BitOrAssign, bitor_assign, op_or);
        binary!($t, BitXor, bitxor, BitXorAssign, bitxor_assign, op_xor);
    };
}

binaries!(Uint);
binaries!(Int);

/// A shift amount of a primitive integer type, reduced for a width of `bits` bits.
trait Amount: Copy {
    /// The amount modulo `bits` (Euclid's remainder) and whether it is outside `0..bits`.
    fn reduce(self, bits: u32) -> (u32, bool);
}

macro_rules! amount_unsigned {
    ($($p:ty),*) => {$(
        impl Amount for $p {
            #[inline]
            fn reduce(self, bits: u32) -> (u32, bool) {
                if bits == 0 {
                    return (0, true);
                }
                let x = self as u128;
                ((x % bits as u128) as u32, x >= bits as u128)
            }
        }
    )*};
}

macro_rules! amount_signed {
    ($($p:ty),*) => {$(
        impl Amount for $p {
            #[inline]
            fn reduce(self, bits: u32) -> (u32, bool) {
                if bits == 0 {
                    return (0, true);
                }
                let x = self as i128;
                (x.rem_euclid(bits as i128) as u32, x < 0 || x >= bits as i128)
            }
        }
    )*};
}

amount_unsigned!(u8, u16, u32, u64, u128, usize);
amount_signed!(i8, i16, i32, i64, i128, isize);

/// The shifts of a fixed-width type by the primitive integer types, owned or borrowed on both
/// sides, and their assignment forms.
macro_rules! shifts {
    ($t:ident: $($p:ty),*) => {$(
        impl<const N: usize> Shl<$p> for $t<N> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn shl(self, rhs: $p) -> Self {
                self.op_shl(rhs.reduce(Self::BITS))
            }
        }
        impl<const N: usize> Shl<&$p> for $t<N> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn shl(self, rhs: &$p) -> Self {
                self.op_shl(rhs.reduce(Self::BITS))
            }
        }
        impl<const N: usize> Shl<$p> for &$t<N> {
            type Output = $t<N>;
            #[inline]
            #[track_caller]
            fn shl(self, rhs: $p) -> $t<N> {
                self.op_shl(rhs.reduce($t::<N>::BITS))
            }
        }
        impl<const N: usize> Shl<&$p> for &$t<N> {
            type Output = $t<N>;
            #[inline]
            #[track_caller]
            fn shl(self, rhs: &$p) -> $t<N> {
                self.op_shl(rhs.reduce($t::<N>::BITS))
            }
        }
        impl<const N: usize> ShlAssign<$p> for $t<N> {
            #[inline]
            #[track_caller]
            fn shl_assign(&mut self, rhs: $p) {
                *self = self.op_shl(rhs.reduce(Self::BITS));
            }
        }
        impl<const N: usize> ShlAssign<&$p> for $t<N> {
            #[inline]
            #[track_caller]
            fn shl_assign(&mut self, rhs: &$p) {
                *self = self.op_shl(rhs.reduce(Self::BITS));
            }
        }
        impl<const N: usize> Shr<$p> for $t<N> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn shr(self, rhs: $p) -> Self {
                self.op_shr(rhs.reduce(Self::BITS))
            }
        }
        impl<const N: usize> Shr<&$p> for $t<N> {
            type Output = Self;
            #[inline]
            #[track_caller]
            fn shr(self, rhs: &$p) -> Self {
                self.op_shr(rhs.reduce(Self::BITS))
            }
        }
        impl<const N: usize> Shr<$p> for &$t<N> {
            type Output = $t<N>;
            #[inline]
            #[track_caller]
            fn shr(self, rhs: $p) -> $t<N> {
                self.op_shr(rhs.reduce($t::<N>::BITS))
            }
        }
        impl<const N: usize> Shr<&$p> for &$t<N> {
            type Output = $t<N>;
            #[inline]
            #[track_caller]
            fn shr(self, rhs: &$p) -> $t<N> {
                self.op_shr(rhs.reduce($t::<N>::BITS))
            }
        }
        impl<const N: usize> ShrAssign<$p> for $t<N> {
            #[inline]
            #[track_caller]
            fn shr_assign(&mut self, rhs: $p) {
                *self = self.op_shr(rhs.reduce(Self::BITS));
            }
        }
        impl<const N: usize> ShrAssign<&$p> for $t<N> {
            #[inline]
            #[track_caller]
            fn shr_assign(&mut self, rhs: &$p) {
                *self = self.op_shr(rhs.reduce(Self::BITS));
            }
        }
    )*};
}

shifts!(Uint: u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
shifts!(Int: u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

impl<const N: usize> Not for Uint<N> {
    type Output = Self;
    /// Every bit inverted.
    #[inline]
    fn not(self) -> Self {
        self.not_bits()
    }
}

impl<const N: usize> Not for &Uint<N> {
    type Output = Uint<N>;
    /// Every bit inverted.
    #[inline]
    fn not(self) -> Uint<N> {
        self.not_bits()
    }
}

impl<const N: usize> Not for Int<N> {
    type Output = Self;
    /// Every bit of the two's complement inverted: `!x = −x − 1`.
    #[inline]
    fn not(self) -> Self {
        self.cast_unsigned().not_bits().cast_signed()
    }
}

impl<const N: usize> Not for &Int<N> {
    type Output = Int<N>;
    /// Every bit of the two's complement inverted: `!x = −x − 1`.
    #[inline]
    fn not(self) -> Int<N> {
        !*self
    }
}

impl<const N: usize> Neg for Int<N> {
    type Output = Self;
    /// `−self`; `−MIN` panics when debug assertions are on and gives `MIN` when they are off.
    #[inline]
    #[track_caller]
    fn neg(self) -> Self {
        debug_checked!(self.overflowing_neg(), "attempt to negate with overflow")
    }
}

impl<const N: usize> Neg for &Int<N> {
    type Output = Int<N>;
    /// `−self`, as for an owned operand.
    #[inline]
    #[track_caller]
    fn neg(self) -> Int<N> {
        -*self
    }
}

/// `Sum` and `Product` of a fixed-width type over owned and borrowed items, by the operators, so
/// overflowing as they do.
macro_rules! folds {
    ($t:ident) => {
        impl<const N: usize> Sum for $t<N> {
            #[track_caller]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                let mut acc = Self::ZERO;
                for x in iter {
                    acc = acc.op_add(x);
                }
                acc
            }
        }
        impl<'a, const N: usize> Sum<&'a $t<N>> for $t<N> {
            #[track_caller]
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                let mut acc = Self::ZERO;
                for x in iter {
                    acc = acc.op_add(*x);
                }
                acc
            }
        }
        impl<const N: usize> Product for $t<N> {
            #[track_caller]
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                let mut acc = Self::ONE;
                for x in iter {
                    acc = acc.op_mul(x);
                }
                acc
            }
        }
        impl<'a, const N: usize> Product<&'a $t<N>> for $t<N> {
            #[track_caller]
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                let mut acc = Self::ONE;
                for x in iter {
                    acc = acc.op_mul(*x);
                }
                acc
            }
        }
    };
}

folds!(Uint);
folds!(Int);
