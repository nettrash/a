//! The signed fixed-width integer [`Int<N>`], two's complement over [`Uint<N>`]: its constants,
//! its arithmetic in the forms of the primitive integers (checked, wrapping, overflowing,
//! saturating), powers, roots, logarithms, bits and bytes.
//!
//! Sums, differences, products and bit operations are those of the unsigned bits, as two's
//! complement makes them; quotients, remainders, roots and logarithms work on the absolute values
//! and restore the sign, truncating toward zero as the primitives do.

use super::Uint;
use core::cmp::Ordering;
use core::fmt;

/// A signed integer of `N` 64-bit limbs in two's complement, as `i64` is: from `−2^(64·N−1)` to
/// `2^(64·N−1) − 1`. It behaves as an `i{64·N}` would: see the [module](super) for the operators
/// and their overflow behaviour.
///
/// # Examples
///
/// ```
/// use a::math::uint::Int;
///
/// let m = Int::<4>::from_i64(-5);
/// assert!(m.is_negative());
/// assert!(m < Int::ZERO);
/// assert_eq!(m.cast_unsigned().as_limbs(), &[u64::MAX - 4, u64::MAX, u64::MAX, u64::MAX]);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Int<const N: usize> {
    /// The bits, as an unsigned integer of the same width.
    bits: Uint<N>,
}

impl<const N: usize> Int<N> {
    /// The width in bits, `64·N`.
    pub const BITS: u32 = 64 * N as u32;
    /// Zero.
    pub const ZERO: Self = Int { bits: Uint::ZERO };
    /// One.
    pub const ONE: Self = Int { bits: Uint::ONE };
    /// Minus one.
    pub const MINUS_ONE: Self = Int { bits: Uint::MAX };
    /// The smallest value, `−2^(64·N−1)`.
    pub const MIN: Self = {
        let mut limbs = [0; N];
        limbs[N - 1] = 1 << 63;
        Int {
            bits: Uint::from_limbs(limbs),
        }
    };
    /// The largest value, `2^(64·N−1) − 1`.
    pub const MAX: Self = {
        let mut limbs = [u64::MAX; N];
        limbs[N - 1] = u64::MAX >> 1;
        Int {
            bits: Uint::from_limbs(limbs),
        }
    };

    /// The integer of the two's-complement limbs `limbs`, the low one first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_limbs([u64::MAX; 2]), Int::MINUS_ONE);
    /// ```
    #[inline]
    pub const fn from_limbs(limbs: [u64; N]) -> Self {
        Int {
            bits: Uint::from_limbs(limbs),
        }
    }

    /// The two's-complement limbs, the low one first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-1).as_limbs(), &[u64::MAX; 2]);
    /// ```
    #[inline]
    pub const fn as_limbs(&self) -> &[u64; N] {
        self.bits.as_limbs()
    }

    /// The two's-complement limbs, the low one first, to change in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let mut x = Int::<2>::ZERO;
    /// x.as_limbs_mut()[1] = 1 << 63;
    /// assert_eq!(x, Int::MIN);
    /// ```
    #[inline]
    pub const fn as_limbs_mut(&mut self) -> &mut [u64; N] {
        self.bits.as_limbs_mut()
    }

    /// The two's-complement limbs, the low one first, by value.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.into_limbs(), [0, 1 << 63]);
    /// ```
    #[inline]
    pub const fn into_limbs(self) -> [u64; N] {
        self.bits.into_limbs()
    }

    /// The integer `x`.
    ///
    /// # Panics
    ///
    /// If `N` is 0 and `x` is not.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-2).as_limbs(), &[u64::MAX - 1, u64::MAX]);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn from_i64(x: i64) -> Self {
        if N == 0 {
            assert!(x == 0, "the value does not fit in zero limbs");
        }
        let fill = if x < 0 { u64::MAX } else { 0 };
        let mut limbs = [fill; N];
        if N > 0 {
            limbs[0] = x as u64;
        }
        Int {
            bits: Uint::from_limbs(limbs),
        }
    }

    /// The integer `x`. `TryFrom<i128>` is the form that fails instead of panicking.
    ///
    /// # Panics
    ///
    /// If `x` does not fit: if `N` is 1 and `x` is outside the range of `i64`, or `N` is 0 and `x`
    /// is not 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// const X: Int<2> = Int::from_i128(i128::MIN);
    /// assert_eq!(X, Int::MIN);
    /// assert_eq!(Int::<1>::from_i128(-7), Int::from_i64(-7));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn from_i128(x: i128) -> Self {
        let fill = if x < 0 { u64::MAX } else { 0 };
        let mut limbs = [fill; N];
        if N >= 2 {
            limbs[0] = x as u64;
            limbs[1] = (x >> 64) as u64;
        } else if N == 1 {
            assert!(x as i64 as i128 == x, "the value does not fit in one limb");
            limbs[0] = x as u64;
        } else {
            assert!(x == 0, "the value does not fit in zero limbs");
        }
        Int {
            bits: Uint::from_limbs(limbs),
        }
    }

    /// The same bits as an unsigned integer, as `i64::cast_unsigned` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MINUS_ONE.cast_unsigned(), Uint::MAX);
    /// ```
    #[inline]
    pub const fn cast_unsigned(self) -> Uint<N> {
        self.bits
    }

    /// Whether it is below zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert!(Int::<3>::MIN.is_negative());
    /// assert!(!Int::<3>::ZERO.is_negative());
    /// ```
    #[inline]
    pub const fn is_negative(&self) -> bool {
        N > 0 && self.bits.as_limbs()[N - 1] >> 63 == 1
    }

    /// Whether it is above zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert!(Int::<3>::ONE.is_positive());
    /// assert!(!Int::<3>::ZERO.is_positive());
    /// assert!(!Int::<3>::MINUS_ONE.is_positive());
    /// ```
    #[inline]
    pub const fn is_positive(&self) -> bool {
        !self.is_negative() && !self.bits.is_zero()
    }

    /// Whether it is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert!(Int::<3>::ZERO.is_zero());
    /// assert!(!Int::<3>::MIN.is_zero());
    /// ```
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.bits.is_zero()
    }

    /// The sign: −1 below zero, 0 for zero, 1 above.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.signum(), Int::MINUS_ONE);
    /// assert_eq!(Int::<2>::ZERO.signum(), Int::ZERO);
    /// assert_eq!(Int::<2>::MAX.signum(), Int::ONE);
    /// ```
    #[inline]
    pub const fn signum(self) -> Self {
        if self.is_negative() {
            Self::MINUS_ONE
        } else if self.bits.is_zero() {
            Self::ZERO
        } else {
            Self::ONE
        }
    }

    /// The order of `self` and `rhs`, as `Ord::cmp` gives it, in a `const fn`.
    #[inline]
    pub(super) const fn compare(&self, rhs: &Self) -> Ordering {
        match (self.is_negative(), rhs.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            // Of one sign, two's complement orders as the unsigned bits do.
            _ => self.bits.compare(&rhs.bits),
        }
    }

    /// Whether it is `MIN`, without naming the constant (which needs a limb).
    #[inline]
    const fn is_min(&self) -> bool {
        self.is_negative() && self.bits.trailing_zeros() == Self::BITS - 1
    }

    /// The integer of the sign `negative` and the magnitude `m`, wrapped to the width.
    #[inline]
    pub(super) const fn from_sign_magnitude(negative: bool, m: Uint<N>) -> Self {
        Int {
            bits: if negative { m.wrapping_neg() } else { m },
        }
    }

    // ---------------------------------------------------------------------------------------
    // Absolute values and negation.

    /// `|self|` as an unsigned integer, which holds it even for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MIN.unsigned_abs(), Uint::ONE << 127);
    /// assert_eq!(Int::<2>::from_i64(-5).unsigned_abs(), Uint::from_u64(5));
    /// ```
    #[inline]
    pub const fn unsigned_abs(self) -> Uint<N> {
        if self.is_negative() {
            self.bits.wrapping_neg()
        } else {
            self.bits
        }
    }

    /// `|self|` wrapped to the width, and whether it wrapped (only for `MIN`, whose absolute value
    /// is `MIN` again).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-5).overflowing_abs(), (Int::from_i64(5), false));
    /// assert_eq!(Int::<2>::MIN.overflowing_abs(), (Int::MIN, true));
    /// ```
    #[inline]
    pub const fn overflowing_abs(self) -> (Self, bool) {
        (
            Int {
                bits: self.unsigned_abs(),
            },
            self.is_min(),
        )
    }

    /// `|self|`, or `None` for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-5).checked_abs(), Some(Int::from_i64(5)));
    /// assert_eq!(Int::<2>::MIN.checked_abs(), None);
    /// ```
    #[inline]
    pub const fn checked_abs(self) -> Option<Self> {
        match self.overflowing_abs() {
            (a, false) => Some(a),
            _ => None,
        }
    }

    /// `|self|` wrapped to the width: `MIN` for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_abs(), Int::MIN);
    /// ```
    #[inline]
    pub const fn wrapping_abs(self) -> Self {
        self.overflowing_abs().0
    }

    /// `|self|`, or `MAX` for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.saturating_abs(), Int::MAX);
    /// ```
    #[inline]
    pub const fn saturating_abs(self) -> Self {
        match self.overflowing_abs() {
            (a, false) => a,
            _ => Self::MAX,
        }
    }

    /// `|self|`. As an operator, `abs` of `MIN` panics when debug assertions are on, and gives
    /// `MIN` when they are off.
    ///
    /// # Panics
    ///
    /// With debug assertions, for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-5).abs(), Int::from_i64(5));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn abs(self) -> Self {
        debug_checked!(self.overflowing_abs(), "attempt to negate with overflow")
    }

    /// `|self − rhs|` as an unsigned integer, which always holds it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MIN.abs_diff(Int::MAX), Uint::MAX);
    /// assert_eq!(Int::<2>::from_i64(-3).abs_diff(Int::from_i64(4)), Uint::from_u64(7));
    /// ```
    #[inline]
    pub const fn abs_diff(self, rhs: Self) -> Uint<N> {
        // The difference of the bits modulo 2^(64N) is the true difference, below 2^(64N).
        if self.compare(&rhs).is_lt() {
            rhs.bits.wrapping_sub(self.bits)
        } else {
            self.bits.wrapping_sub(rhs.bits)
        }
    }

    /// `−self` wrapped to the width, and whether it wrapped (only for `MIN`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.overflowing_neg(), (Int::MINUS_ONE, false));
    /// assert_eq!(Int::<2>::MIN.overflowing_neg(), (Int::MIN, true));
    /// ```
    #[inline]
    pub const fn overflowing_neg(self) -> (Self, bool) {
        (
            Int {
                bits: self.bits.wrapping_neg(),
            },
            self.is_min(),
        )
    }

    /// `−self`, or `None` for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.checked_neg(), Some(Int::MINUS_ONE));
    /// assert_eq!(Int::<2>::MIN.checked_neg(), None);
    /// ```
    #[inline]
    pub const fn checked_neg(self) -> Option<Self> {
        match self.overflowing_neg() {
            (n, false) => Some(n),
            _ => None,
        }
    }

    /// `−self` wrapped to the width: `MIN` for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_neg(), Int::MIN);
    /// ```
    #[inline]
    pub const fn wrapping_neg(self) -> Self {
        self.overflowing_neg().0
    }

    /// `−self`, or `MAX` for `MIN`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.saturating_neg(), Int::MAX);
    /// ```
    #[inline]
    pub const fn saturating_neg(self) -> Self {
        match self.overflowing_neg() {
            (n, false) => n,
            _ => Self::MAX,
        }
    }

    // ---------------------------------------------------------------------------------------
    // Sums and differences.

    /// `self + rhs` wrapped to the width, and whether the true sum is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.overflowing_add(Int::ONE), (Int::MIN, true));
    /// assert_eq!(Int::<2>::MINUS_ONE.overflowing_add(Int::ONE), (Int::ZERO, false));
    /// ```
    #[inline]
    pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (bits, _) = self.bits.overflowing_add(rhs.bits);
        let sum = Int { bits };
        let a = self.is_negative();
        let overflow = a == rhs.is_negative() && sum.is_negative() != a;
        (sum, overflow)
    }

    /// `self − rhs` wrapped to the width, and whether the true difference is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.overflowing_sub(Int::ONE), (Int::MAX, true));
    /// ```
    #[inline]
    pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (bits, _) = self.bits.overflowing_sub(rhs.bits);
        let diff = Int { bits };
        let a = self.is_negative();
        let overflow = a != rhs.is_negative() && diff.is_negative() != a;
        (diff, overflow)
    }

    /// `self + rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.checked_add(Int::MAX), Some(Int::MINUS_ONE));
    /// assert_eq!(Int::<2>::MAX.checked_add(Int::ONE), None);
    /// ```
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.overflowing_add(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// `self + rhs` wrapped to the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.wrapping_add(Int::ONE), Int::MIN);
    /// ```
    #[inline]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }

    /// `self + rhs`, held to `[MIN, MAX]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.saturating_add(Int::ONE), Int::MAX);
    /// assert_eq!(Int::<2>::MIN.saturating_add(Int::MINUS_ONE), Int::MIN);
    /// ```
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        match self.overflowing_add(rhs) {
            (s, false) => s,
            // Both operands have the sign of the overflow.
            _ if rhs.is_negative() => Self::MIN,
            _ => Self::MAX,
        }
    }

    /// `self − rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ZERO.checked_sub(Int::MAX), Some(Int::MIN + Int::ONE));
    /// assert_eq!(Int::<2>::MIN.checked_sub(Int::ONE), None);
    /// ```
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.overflowing_sub(rhs) {
            (d, false) => Some(d),
            _ => None,
        }
    }

    /// `self − rhs` wrapped to the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_sub(Int::ONE), Int::MAX);
    /// ```
    #[inline]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    /// `self − rhs`, held to `[MIN, MAX]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.saturating_sub(Int::ONE), Int::MIN);
    /// assert_eq!(Int::<2>::MAX.saturating_sub(Int::MINUS_ONE), Int::MAX);
    /// ```
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        match self.overflowing_sub(rhs) {
            (d, false) => d,
            _ if rhs.is_negative() => Self::MAX,
            _ => Self::MIN,
        }
    }

    /// `self + rhs` for an unsigned `rhs`, wrapped to the width, and whether the true sum is out
    /// of range, as `i64::overflowing_add_unsigned` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MIN.overflowing_add_unsigned(Uint::MAX), (Int::MAX, false));
    /// assert_eq!(Int::<2>::ZERO.overflowing_add_unsigned(Uint::MAX), (Int::MINUS_ONE, true));
    /// ```
    #[inline]
    pub const fn overflowing_add_unsigned(self, rhs: Uint<N>) -> (Self, bool) {
        // As signed, an rhs with its top bit set is 2^(64N) too small, which turns the overflow
        // of the signed sum over.
        let r = Int { bits: rhs };
        let (sum, overflow) = self.overflowing_add(r);
        (sum, overflow != r.is_negative())
    }

    /// `self + rhs` for an unsigned `rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MINUS_ONE.checked_add_unsigned(Uint::ONE), Some(Int::ZERO));
    /// assert_eq!(Int::<2>::MAX.checked_add_unsigned(Uint::ONE), None);
    /// ```
    #[inline]
    pub const fn checked_add_unsigned(self, rhs: Uint<N>) -> Option<Self> {
        match self.overflowing_add_unsigned(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// `self + rhs` for an unsigned `rhs`, wrapped to the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MAX.wrapping_add_unsigned(Uint::ONE), Int::MIN);
    /// ```
    #[inline]
    pub const fn wrapping_add_unsigned(self, rhs: Uint<N>) -> Self {
        self.overflowing_add_unsigned(rhs).0
    }

    /// `self + rhs` for an unsigned `rhs`, or `MAX` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::ZERO.saturating_add_unsigned(Uint::MAX), Int::MAX);
    /// ```
    #[inline]
    pub const fn saturating_add_unsigned(self, rhs: Uint<N>) -> Self {
        match self.overflowing_add_unsigned(rhs) {
            (s, false) => s,
            _ => Self::MAX,
        }
    }

    /// `self − rhs` for an unsigned `rhs`, wrapped to the width, and whether the true difference
    /// is out of range, as `i64::overflowing_sub_unsigned` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MAX.overflowing_sub_unsigned(Uint::MAX), (Int::MIN, false));
    /// assert_eq!(Int::<2>::ZERO.overflowing_sub_unsigned(Uint::MAX), (Int::ONE, true));
    /// ```
    #[inline]
    pub const fn overflowing_sub_unsigned(self, rhs: Uint<N>) -> (Self, bool) {
        let r = Int { bits: rhs };
        let (diff, overflow) = self.overflowing_sub(r);
        (diff, overflow != r.is_negative())
    }

    /// `self − rhs` for an unsigned `rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::ZERO.checked_sub_unsigned(Uint::ONE), Some(Int::MINUS_ONE));
    /// assert_eq!(Int::<2>::MIN.checked_sub_unsigned(Uint::ONE), None);
    /// ```
    #[inline]
    pub const fn checked_sub_unsigned(self, rhs: Uint<N>) -> Option<Self> {
        match self.overflowing_sub_unsigned(rhs) {
            (d, false) => Some(d),
            _ => None,
        }
    }

    /// `self − rhs` for an unsigned `rhs`, wrapped to the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_sub_unsigned(Uint::ONE), Int::MAX);
    /// ```
    #[inline]
    pub const fn wrapping_sub_unsigned(self, rhs: Uint<N>) -> Self {
        self.overflowing_sub_unsigned(rhs).0
    }

    /// `self − rhs` for an unsigned `rhs`, or `MIN` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Int::<2>::ZERO.saturating_sub_unsigned(Uint::MAX), Int::MIN);
    /// ```
    #[inline]
    pub const fn saturating_sub_unsigned(self, rhs: Uint<N>) -> Self {
        match self.overflowing_sub_unsigned(rhs) {
            (d, false) => d,
            _ => Self::MIN,
        }
    }

    // ---------------------------------------------------------------------------------------
    // Products.

    /// `self · rhs` wrapped to the width, and whether the true product is out of range: the
    /// product of the absolute values, by the schoolbook method, with the sign restored.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let x = Int::<2>::ONE << 126u32;
    /// assert_eq!(x.overflowing_mul(Int::from_i64(-2)), (Int::MIN, false));
    /// assert_eq!(x.overflowing_mul(Int::from_i64(2)), (Int::MIN, true));
    /// ```
    #[inline]
    pub const fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        let negative = self.is_negative() != rhs.is_negative();
        let (m, wrapped) = self.unsigned_abs().overflowing_mul(rhs.unsigned_abs());
        // The magnitude may be up to 2^(64N − 1) for a negative product, one less for a positive.
        let top = m.bit(Self::BITS - 1);
        let overflow = wrapped
            || if negative {
                top && m.trailing_zeros() != Self::BITS - 1
            } else {
                top
            };
        (Self::from_sign_magnitude(negative, m), overflow)
    }

    /// `self · rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.checked_mul(Int::ONE), Some(Int::MIN));
    /// assert_eq!(Int::<2>::MIN.checked_mul(Int::MINUS_ONE), None);
    /// ```
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        match self.overflowing_mul(rhs) {
            (p, false) => Some(p),
            _ => None,
        }
    }

    /// `self · rhs` wrapped to the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_mul(Int::MINUS_ONE), Int::MIN);
    /// ```
    #[inline]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        Int {
            bits: self.bits.wrapping_mul(rhs.bits),
        }
    }

    /// `self · rhs`, held to `[MIN, MAX]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.saturating_mul(Int::MINUS_ONE), Int::MAX);
    /// assert_eq!(Int::<2>::MIN.saturating_mul(Int::from_i64(2)), Int::MIN);
    /// ```
    #[inline]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        match self.overflowing_mul(rhs) {
            (p, false) => p,
            _ if self.is_negative() != rhs.is_negative() => Self::MIN,
            _ => Self::MAX,
        }
    }

    // ---------------------------------------------------------------------------------------
    // Quotients and remainders.

    /// The quotient truncated toward zero and the remainder with the sign of `self`, wrapped:
    /// `MIN / −1` gives `(MIN, 0)`. The caller has checked that `rhs` is not zero.
    #[inline]
    const fn div_rem_wrapping(self, rhs: Self) -> (Self, Self) {
        let (q, r) = self.unsigned_abs().div_rem(rhs.unsigned_abs());
        (
            Self::from_sign_magnitude(self.is_negative() != rhs.is_negative(), q),
            Self::from_sign_magnitude(self.is_negative(), r),
        )
    }

    /// Whether `self / rhs` overflows: `MIN / −1`.
    #[inline]
    const fn div_overflows(&self, rhs: &Self) -> bool {
        self.is_min() && rhs.bits.count_zeros() == 0
    }

    /// The quotient truncated toward zero and the remainder with the sign of `self`, `(self / rhs,
    /// self % rhs)`, computed together.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero, or for `MIN / −1`, whose quotient is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let (q, r) = Int::<2>::from_i64(-7).div_rem(Int::from_i64(2));
    /// assert_eq!((q, r), (Int::from_i64(-3), Int::from_i64(-1)));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn div_rem(self, rhs: Self) -> (Self, Self) {
        if rhs.is_zero() {
            panic!("attempt to divide by zero");
        }
        if self.div_overflows(&rhs) {
            panic!("attempt to divide with overflow");
        }
        self.div_rem_wrapping(rhs)
    }

    /// `self / rhs` truncated toward zero, wrapped to the width, and whether it wrapped (only
    /// `MIN / −1`, which gives `MIN`).
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).overflowing_div(Int::from_i64(2)), (Int::from_i64(-3), false));
    /// assert_eq!(Int::<2>::MIN.overflowing_div(Int::MINUS_ONE), (Int::MIN, true));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_div(self, rhs: Self) -> (Self, bool) {
        if rhs.is_zero() {
            panic!("attempt to divide by zero");
        }
        (self.div_rem_wrapping(rhs).0, self.div_overflows(&rhs))
    }

    /// `self / rhs` truncated toward zero, or `None` if `rhs` is zero or for `MIN / −1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(7).checked_div(Int::from_i64(-2)), Some(Int::from_i64(-3)));
    /// assert_eq!(Int::<2>::MIN.checked_div(Int::MINUS_ONE), None);
    /// assert_eq!(Int::<2>::ONE.checked_div(Int::ZERO), None);
    /// ```
    #[inline]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() || self.div_overflows(&rhs) {
            None
        } else {
            Some(self.div_rem_wrapping(rhs).0)
        }
    }

    /// `self / rhs` truncated toward zero, wrapped: `MIN / −1` gives `MIN`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_div(Int::MINUS_ONE), Int::MIN);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_div(self, rhs: Self) -> Self {
        self.overflowing_div(rhs).0
    }

    /// `self / rhs` truncated toward zero, `MAX` for `MIN / −1`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.saturating_div(Int::MINUS_ONE), Int::MAX);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        match self.overflowing_div(rhs) {
            (q, false) => q,
            _ => Self::MAX,
        }
    }

    /// `self % rhs`, with the sign of `self`, wrapped, and whether it wrapped (only `MIN % −1`,
    /// which gives 0).
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).overflowing_rem(Int::from_i64(2)), (Int::MINUS_ONE, false));
    /// assert_eq!(Int::<2>::MIN.overflowing_rem(Int::MINUS_ONE), (Int::ZERO, true));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
        if rhs.is_zero() {
            panic!("attempt to calculate the remainder with a divisor of zero");
        }
        (self.div_rem_wrapping(rhs).1, self.div_overflows(&rhs))
    }

    /// `self % rhs`, or `None` if `rhs` is zero or for `MIN % −1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).checked_rem(Int::from_i64(2)), Some(Int::MINUS_ONE));
    /// assert_eq!(Int::<2>::MIN.checked_rem(Int::MINUS_ONE), None);
    /// ```
    #[inline]
    pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() || self.div_overflows(&rhs) {
            None
        } else {
            Some(self.div_rem_wrapping(rhs).1)
        }
    }

    /// `self % rhs`, wrapped: `MIN % −1` gives 0.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_rem(Int::MINUS_ONE), Int::ZERO);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_rem(self, rhs: Self) -> Self {
        self.overflowing_rem(rhs).0
    }

    /// Euclid's quotient and remainder, wrapped: the remainder `r` is never negative, `0 ≤ r <
    /// |rhs|`, and `self = q·rhs + r`. The caller has checked that `rhs` is not zero.
    #[inline]
    const fn div_rem_euclid_wrapping(self, rhs: Self) -> (Self, Self) {
        let (q, r) = self.div_rem_wrapping(rhs);
        if r.is_negative() {
            // r + |rhs|, and the quotient one further from zero, away from rhs's sign. For
            // rhs = MIN the sum of the bits is still right: r − MIN, below 2^(64N − 1).
            let r = Int {
                bits: r.bits.wrapping_add(rhs.unsigned_abs()),
            };
            let q = if rhs.is_negative() {
                q.wrapping_add(Self::ONE)
            } else {
                q.wrapping_sub(Self::ONE)
            };
            (q, r)
        } else {
            (q, r)
        }
    }

    /// Euclid's quotient: `q` with `self = q·rhs + r` and `0 ≤ r < |rhs|`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero, or for `MIN / −1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).div_euclid(Int::from_i64(2)), Int::from_i64(-4));
    /// assert_eq!(Int::<2>::from_i64(-7).div_euclid(Int::from_i64(-2)), Int::from_i64(4));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn div_euclid(self, rhs: Self) -> Self {
        if rhs.is_zero() {
            panic!("attempt to divide by zero");
        }
        if self.div_overflows(&rhs) {
            panic!("attempt to divide with overflow");
        }
        self.div_rem_euclid_wrapping(rhs).0
    }

    /// Euclid's quotient, or `None` if `rhs` is zero or for `MIN / −1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).checked_div_euclid(Int::from_i64(2)), Some(Int::from_i64(-4)));
    /// assert_eq!(Int::<2>::MIN.checked_div_euclid(Int::MINUS_ONE), None);
    /// ```
    #[inline]
    pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() || self.div_overflows(&rhs) {
            None
        } else {
            Some(self.div_rem_euclid_wrapping(rhs).0)
        }
    }

    /// Euclid's quotient, wrapped: `MIN / −1` gives `MIN`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_div_euclid(Int::MINUS_ONE), Int::MIN);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_div_euclid(self, rhs: Self) -> Self {
        self.overflowing_div_euclid(rhs).0
    }

    /// Euclid's quotient, wrapped, and whether it wrapped (only `MIN / −1`).
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.overflowing_div_euclid(Int::MINUS_ONE), (Int::MIN, true));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
        if rhs.is_zero() {
            panic!("attempt to divide by zero");
        }
        (
            self.div_rem_euclid_wrapping(rhs).0,
            self.div_overflows(&rhs),
        )
    }

    /// Euclid's remainder: `0 ≤ r < |rhs|` with `self = q·rhs + r`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero, or for `MIN % −1`, as `i64::rem_euclid` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).rem_euclid(Int::from_i64(2)), Int::ONE);
    /// assert_eq!(Int::<2>::from_i64(-7).rem_euclid(Int::from_i64(-2)), Int::ONE);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn rem_euclid(self, rhs: Self) -> Self {
        if rhs.is_zero() {
            panic!("attempt to calculate the remainder with a divisor of zero");
        }
        if self.div_overflows(&rhs) {
            panic!("attempt to calculate the remainder with overflow");
        }
        self.div_rem_euclid_wrapping(rhs).1
    }

    /// Euclid's remainder, or `None` if `rhs` is zero or for `MIN % −1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-7).checked_rem_euclid(Int::from_i64(2)), Some(Int::ONE));
    /// assert_eq!(Int::<2>::MIN.checked_rem_euclid(Int::MINUS_ONE), None);
    /// ```
    #[inline]
    pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() || self.div_overflows(&rhs) {
            None
        } else {
            Some(self.div_rem_euclid_wrapping(rhs).1)
        }
    }

    /// Euclid's remainder, wrapped: `MIN % −1` gives 0.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.wrapping_rem_euclid(Int::MINUS_ONE), Int::ZERO);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_rem_euclid(self, rhs: Self) -> Self {
        self.overflowing_rem_euclid(rhs).0
    }

    /// Euclid's remainder, wrapped, and whether it wrapped (only `MIN % −1`).
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.overflowing_rem_euclid(Int::MINUS_ONE), (Int::ZERO, true));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
        if rhs.is_zero() {
            panic!("attempt to calculate the remainder with a divisor of zero");
        }
        (
            self.div_rem_euclid_wrapping(rhs).1,
            self.div_overflows(&rhs),
        )
    }

    /// `(self + rhs) / 2` rounded toward zero, without overflow, as `i64::midpoint` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.midpoint(Int::MAX), Int::ZERO);
    /// assert_eq!(Int::<2>::from_i64(-3).midpoint(Int::ZERO), Int::from_i64(-1));
    /// ```
    #[inline]
    pub const fn midpoint(self, rhs: Self) -> Self {
        // ⌊(a + b)/2⌋ = (a & b) + ((a ^ b) >> 1), the shift arithmetic; then one up for a negative
        // odd sum, to round toward zero.
        let a = self.bits.into_limbs();
        let b = rhs.bits.into_limbs();
        let mut and = [0; N];
        let mut xor = [0; N];
        let mut i = 0;
        while i < N {
            and[i] = a[i] & b[i];
            xor[i] = a[i] ^ b[i];
            i += 1;
        }
        let half = Int::from_limbs(xor).unbounded_shr(1);
        let floor = Int::from_limbs(and).wrapping_add(half);
        if floor.is_negative() && N > 0 && xor[0] & 1 == 1 {
            floor.wrapping_add(Self::ONE)
        } else {
            floor
        }
    }

    // ---------------------------------------------------------------------------------------
    // Shifts.

    /// `self << (rhs mod BITS)` and whether `rhs ≥ BITS`, as `i64::overflowing_shl` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.overflowing_shl(127), (Int::MIN, false));
    /// assert_eq!(Int::<2>::ONE.overflowing_shl(129), (Int::from_i64(2), true));
    /// ```
    #[inline]
    pub const fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
        let (bits, o) = self.bits.overflowing_shl(rhs);
        (Int { bits }, o)
    }

    /// `self << rhs`, or `None` if `rhs ≥ BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MINUS_ONE.checked_shl(127), Some(Int::MIN));
    /// assert_eq!(Int::<2>::ONE.checked_shl(128), None);
    /// ```
    #[inline]
    pub const fn checked_shl(self, rhs: u32) -> Option<Self> {
        match self.overflowing_shl(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// `self << (rhs mod BITS)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.wrapping_shl(130), Int::from_i64(4));
    /// ```
    #[inline]
    pub const fn wrapping_shl(self, rhs: u32) -> Self {
        self.overflowing_shl(rhs).0
    }

    /// `self << rhs`, zero if `rhs ≥ BITS`, as `i64::unbounded_shl` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MINUS_ONE.unbounded_shl(128), Int::ZERO);
    /// ```
    #[inline]
    pub const fn unbounded_shl(self, rhs: u32) -> Self {
        Int {
            bits: self.bits.unbounded_shl(rhs),
        }
    }

    /// The arithmetic shift `self >> (rhs mod BITS)`, which copies the sign bit in at the top, and
    /// whether `rhs ≥ BITS`, as `i64::overflowing_shr` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.overflowing_shr(127), (Int::MINUS_ONE, false));
    /// assert_eq!(Int::<2>::from_i64(-8).overflowing_shr(130), (Int::from_i64(-2), true));
    /// ```
    #[inline]
    pub const fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
        if Self::BITS == 0 {
            return (self, true);
        }
        (self.shr_bounded(rhs % Self::BITS), rhs >= Self::BITS)
    }

    /// The arithmetic shift by `k < BITS`.
    #[inline]
    const fn shr_bounded(self, k: u32) -> Self {
        let fill = if self.is_negative() { u64::MAX } else { 0 };
        Int::from_limbs(super::limbs::shr(self.bits.as_limbs(), k, fill))
    }

    /// The arithmetic shift `self >> rhs`, or `None` if `rhs ≥ BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-8).checked_shr(2), Some(Int::from_i64(-2)));
    /// assert_eq!(Int::<2>::ONE.checked_shr(128), None);
    /// ```
    #[inline]
    pub const fn checked_shr(self, rhs: u32) -> Option<Self> {
        match self.overflowing_shr(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// The arithmetic shift `self >> (rhs mod BITS)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-8).wrapping_shr(130), Int::from_i64(-2));
    /// ```
    #[inline]
    pub const fn wrapping_shr(self, rhs: u32) -> Self {
        self.overflowing_shr(rhs).0
    }

    /// The arithmetic shift `self >> rhs`, which for `rhs ≥ BITS` is 0 for a number at least zero
    /// and −1 for a negative one, as `i64::unbounded_shr` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-8).unbounded_shr(1000), Int::MINUS_ONE);
    /// assert_eq!(Int::<2>::MAX.unbounded_shr(1000), Int::ZERO);
    /// ```
    #[inline]
    pub const fn unbounded_shr(self, rhs: u32) -> Self {
        if rhs >= Self::BITS {
            if self.is_negative() {
                Self::MINUS_ONE
            } else {
                Self::ZERO
            }
        } else {
            self.shr_bounded(rhs)
        }
    }

    // ---------------------------------------------------------------------------------------
    // Powers, roots and logarithms.

    /// `self^exp` wrapped to the width, and whether the true power is out of range, by squaring
    /// and multiplying from the low bit of `exp` up. `0⁰ = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let m2 = Int::<2>::from_i64(-2);
    /// assert_eq!(m2.overflowing_pow(127), (Int::MIN, false));
    /// assert_eq!(m2.overflowing_pow(128), (Int::ZERO, true));
    /// ```
    pub const fn overflowing_pow(self, exp: u32) -> (Self, bool) {
        if exp == 0 {
            return (Self::ONE, false);
        }
        // For |base| ≥ 2 every partial product divides the power, and no partial product is
        // exactly MAX + 1 unless the power is beyond MIN, so the flag is exact.
        let mut base = self;
        let mut acc = Self::ONE;
        let mut overflow = false;
        let mut e = exp;
        loop {
            if e & 1 == 1 {
                let (p, o) = acc.overflowing_mul(base);
                acc = p;
                overflow |= o;
                if e == 1 {
                    return (acc, overflow);
                }
            }
            e >>= 1;
            let (s, o) = base.overflowing_mul(base);
            base = s;
            overflow |= o;
        }
    }

    /// `self^exp`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(2).checked_pow(126), Some(Int::ONE << 126));
    /// assert_eq!(Int::<2>::from_i64(2).checked_pow(127), None);
    /// ```
    #[inline]
    pub const fn checked_pow(self, exp: u32) -> Option<Self> {
        match self.overflowing_pow(exp) {
            (p, false) => Some(p),
            _ => None,
        }
    }

    /// `self^exp` wrapped to the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(2).wrapping_pow(127), Int::MIN);
    /// ```
    #[inline]
    pub const fn wrapping_pow(self, exp: u32) -> Self {
        self.overflowing_pow(exp).0
    }

    /// `self^exp`, held to `[MIN, MAX]`: `MIN` when the true power is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-3).saturating_pow(81), Int::MIN);
    /// assert_eq!(Int::<2>::from_i64(-3).saturating_pow(82), Int::MAX);
    /// ```
    #[inline]
    pub const fn saturating_pow(self, exp: u32) -> Self {
        match self.overflowing_pow(exp) {
            (p, false) => p,
            _ if self.is_negative() && exp & 1 == 1 => Self::MIN,
            _ => Self::MAX,
        }
    }

    /// `self^exp`. As an operator, it panics on overflow when debug assertions are on, and wraps
    /// when they are off.
    ///
    /// # Panics
    ///
    /// With debug assertions, if the power overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-10).pow(3), Int::from_i64(-1000));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn pow(self, exp: u32) -> Self {
        debug_checked!(
            self.overflowing_pow(exp),
            "attempt to exponentiate with overflow"
        )
    }

    /// The square root rounded down, or `None` for a negative number.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(99).checked_isqrt(), Some(Int::from_i64(9)));
    /// assert_eq!(Int::<2>::MINUS_ONE.checked_isqrt(), None);
    /// ```
    #[inline]
    pub const fn checked_isqrt(self) -> Option<Self> {
        if self.is_negative() {
            None
        } else {
            Some(Int {
                bits: self.bits.isqrt(),
            })
        }
    }

    /// The square root rounded down.
    ///
    /// # Panics
    ///
    /// If `self` is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.isqrt(), Int::from_i128(13_043_817_825_332_782_212));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn isqrt(self) -> Self {
        match self.checked_isqrt() {
            Some(r) => r,
            None => panic!("argument of integer square root cannot be negative"),
        }
    }

    /// The base-2 logarithm rounded down, or `None` if `self ≤ 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.checked_ilog2(), Some(126));
    /// assert_eq!(Int::<2>::MINUS_ONE.checked_ilog2(), None);
    /// ```
    #[inline]
    pub const fn checked_ilog2(self) -> Option<u32> {
        if self.is_negative() {
            None
        } else {
            self.bits.checked_ilog2()
        }
    }

    /// The base-2 logarithm rounded down.
    ///
    /// # Panics
    ///
    /// If `self ≤ 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(1024).ilog2(), 10);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn ilog2(self) -> u32 {
        match self.checked_ilog2() {
            Some(l) => l,
            None => panic!("argument of integer logarithm must be positive"),
        }
    }

    /// The base-10 logarithm rounded down, or `None` if `self ≤ 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.checked_ilog10(), Some(38));
    /// assert_eq!(Int::<2>::ZERO.checked_ilog10(), None);
    /// ```
    #[inline]
    pub const fn checked_ilog10(self) -> Option<u32> {
        if self.is_negative() {
            None
        } else {
            self.bits.checked_ilog10()
        }
    }

    /// The base-10 logarithm rounded down.
    ///
    /// # Panics
    ///
    /// If `self ≤ 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(1000).ilog10(), 3);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn ilog10(self) -> u32 {
        match self.checked_ilog10() {
            Some(l) => l,
            None => panic!("argument of integer logarithm must be positive"),
        }
    }

    /// The logarithm to `base` rounded down, or `None` if `self ≤ 0` or `base < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(80).checked_ilog(Int::from_i64(3)), Some(3));
    /// assert_eq!(Int::<2>::from_i64(80).checked_ilog(Int::from_i64(-3)), None);
    /// ```
    #[inline]
    pub const fn checked_ilog(self, base: Self) -> Option<u32> {
        if self.is_negative() || base.is_negative() {
            None
        } else {
            self.bits.checked_ilog(base.bits)
        }
    }

    /// The logarithm to `base` rounded down.
    ///
    /// # Panics
    ///
    /// If `self ≤ 0` or `base < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.ilog(Int::from_i64(3)), 80);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn ilog(self, base: Self) -> u32 {
        if base.is_negative() || base.bits.compare(&Uint::ONE).is_le() {
            panic!("base of integer logarithm must be at least 2");
        }
        match self.checked_ilog(base) {
            Some(l) => l,
            None => panic!("argument of integer logarithm must be positive"),
        }
    }

    // ---------------------------------------------------------------------------------------
    // Bits and bytes, those of the two's complement.

    /// The number of set bits of the two's complement.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MINUS_ONE.count_ones(), 128);
    /// assert_eq!(Int::<2>::MIN.count_ones(), 1);
    /// ```
    #[inline]
    pub const fn count_ones(self) -> u32 {
        self.bits.count_ones()
    }

    /// The number of clear bits of the two's complement.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.count_zeros(), 1);
    /// ```
    #[inline]
    pub const fn count_zeros(self) -> u32 {
        self.bits.count_zeros()
    }

    /// The number of clear bits above the highest set bit: 0 for a negative number.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.leading_zeros(), 127);
    /// assert_eq!(Int::<2>::MINUS_ONE.leading_zeros(), 0);
    /// ```
    #[inline]
    pub const fn leading_zeros(self) -> u32 {
        self.bits.leading_zeros()
    }

    /// The number of clear bits below the lowest set bit: `BITS` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.trailing_zeros(), 127);
    /// ```
    #[inline]
    pub const fn trailing_zeros(self) -> u32 {
        self.bits.trailing_zeros()
    }

    /// The number of set bits above the highest clear bit.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(-4).leading_ones(), 126);
    /// ```
    #[inline]
    pub const fn leading_ones(self) -> u32 {
        self.bits.leading_ones()
    }

    /// The number of set bits below the lowest clear bit.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MAX.trailing_ones(), 127);
    /// ```
    #[inline]
    pub const fn trailing_ones(self) -> u32 {
        self.bits.trailing_ones()
    }

    /// The bits rotated left by `n mod BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::MIN.rotate_left(1), Int::ONE);
    /// ```
    #[inline]
    pub const fn rotate_left(self, n: u32) -> Self {
        Int {
            bits: self.bits.rotate_left(n),
        }
    }

    /// The bits rotated right by `n mod BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.rotate_right(1), Int::MIN);
    /// ```
    #[inline]
    pub const fn rotate_right(self, n: u32) -> Self {
        Int {
            bits: self.bits.rotate_right(n),
        }
    }

    /// The bytes of the two's complement in the opposite order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_i64(0x80).swap_bytes(), Int::MIN);
    /// ```
    #[inline]
    pub const fn swap_bytes(self) -> Self {
        Int {
            bits: self.bits.swap_bytes(),
        }
    }

    /// The bits of the two's complement in the opposite order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::ONE.reverse_bits(), Int::MIN);
    /// ```
    #[inline]
    pub const fn reverse_bits(self) -> Self {
        Int {
            bits: self.bits.reverse_bits(),
        }
    }

    /// The `B = 8·N` bytes of the two's complement, the least significant first. A length other
    /// than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let bytes: [u8; 16] = Int::<2>::from_i64(-2).to_le_bytes();
    /// assert_eq!(bytes[..2], [0xFE, 0xFF]);
    /// ```
    #[inline]
    pub const fn to_le_bytes<const B: usize>(self) -> [u8; B] {
        self.bits.to_le_bytes()
    }

    /// The `B = 8·N` bytes of the two's complement, the most significant first. A length other
    /// than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let bytes: [u8; 16] = Int::<2>::MIN.to_be_bytes();
    /// assert_eq!(bytes[..2], [0x80, 0]);
    /// ```
    #[inline]
    pub const fn to_be_bytes<const B: usize>(self) -> [u8; B] {
        self.bits.to_be_bytes()
    }

    /// The integer of the `B = 8·N` two's-complement bytes `bytes`, the least significant first.
    /// A length other than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<2>::from_le_bytes([0xFF; 16]), Int::MINUS_ONE);
    /// ```
    #[inline]
    pub const fn from_le_bytes<const B: usize>(bytes: [u8; B]) -> Self {
        Int {
            bits: Uint::from_le_bytes(bytes),
        }
    }

    /// The integer of the `B = 8·N` two's-complement bytes `bytes`, the most significant first.
    /// A length other than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let mut bytes = [0u8; 16];
    /// bytes[0] = 0x80;
    /// assert_eq!(Int::<2>::from_be_bytes(bytes), Int::MIN);
    /// ```
    #[inline]
    pub const fn from_be_bytes<const B: usize>(bytes: [u8; B]) -> Self {
        Int {
            bits: Uint::from_be_bytes(bytes),
        }
    }

    /// Writes the `8·N` bytes of the two's complement, the least significant first, into `out`.
    ///
    /// # Panics
    ///
    /// If `out` is not `8·N` bytes long.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let mut out = [0u8; 8];
    /// Int::<1>::from_i64(-2).write_le_bytes(&mut out);
    /// assert_eq!(out, (-2i64).to_le_bytes());
    /// ```
    #[inline]
    #[track_caller]
    pub const fn write_le_bytes(self, out: &mut [u8]) {
        self.bits.write_le_bytes(out)
    }

    /// Writes the `8·N` bytes of the two's complement, the most significant first, into `out`.
    ///
    /// # Panics
    ///
    /// If `out` is not `8·N` bytes long.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let mut out = [0u8; 8];
    /// Int::<1>::from_i64(-2).write_be_bytes(&mut out);
    /// assert_eq!(out, (-2i64).to_be_bytes());
    /// ```
    #[inline]
    #[track_caller]
    pub const fn write_be_bytes(self, out: &mut [u8]) {
        self.bits.write_be_bytes(out)
    }

    /// The integer whose two's complement in `8·k` bits is the `k` bytes `bytes`, the least
    /// significant first, of any length (sign-extended from the last byte), or `None` if it does
    /// not fit; no bytes at all are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<1>::from_le_slice(&[0xFE]), Some(Int::from_i64(-2)));
    /// assert_eq!(Int::<1>::from_le_slice(&[0xFF; 12]), Some(Int::MINUS_ONE));
    /// assert_eq!(Int::<1>::from_le_slice(&[0, 0, 0, 0, 0, 0, 0, 0x80, 0xFF]), Some(Int::MIN));
    /// assert_eq!(Int::<1>::from_le_slice(&[0, 0, 0, 0, 0, 0, 0, 0x80, 0]), None);
    /// ```
    pub const fn from_le_slice(bytes: &[u8]) -> Option<Self> {
        let len = bytes.len();
        let negative = len > 0 && bytes[len - 1] >> 7 == 1;
        Self::from_slice_with(bytes, negative, false)
    }

    /// The integer whose two's complement in `8·k` bits is the `k` bytes `bytes`, the most
    /// significant first, of any length (sign-extended from the first byte), or `None` if it does
    /// not fit; no bytes at all are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// assert_eq!(Int::<1>::from_be_slice(&[0xFF, 0xFE]), Some(Int::from_i64(-2)));
    /// assert_eq!(Int::<1>::from_be_slice(&[0, 0x80, 0, 0, 0, 0, 0, 0, 0]), None);
    /// ```
    pub const fn from_be_slice(bytes: &[u8]) -> Option<Self> {
        let negative = !bytes.is_empty() && bytes[0] >> 7 == 1;
        Self::from_slice_with(bytes, negative, true)
    }

    /// The integer of the two's-complement bytes `bytes` in either order, sign-extended: the bytes
    /// beyond the first `8·N` must all be the fill of the sign, and the sign of the value read must
    /// be the sign of the bytes.
    const fn from_slice_with(bytes: &[u8], negative: bool, big_endian: bool) -> Option<Self> {
        let fill = if negative { u64::MAX } else { 0 };
        let mut limbs = [fill; N];
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            let b = if big_endian {
                bytes[len - 1 - i]
            } else {
                bytes[i]
            };
            if i < 8 * N {
                let shift = 8 * (i % 8);
                limbs[i / 8] = (limbs[i / 8] & !(0xFF << shift)) | ((b as u64) << shift);
            } else if b != fill as u8 {
                return None;
            }
            i += 1;
        }
        let x = Int::from_limbs(limbs);
        if x.is_negative() == negative {
            Some(x)
        } else {
            None
        }
    }
}

impl<const N: usize> Uint<N> {
    /// The same bits as a signed integer, as `u64::cast_signed` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Uint::<2>::MAX.cast_signed(), Int::MINUS_ONE);
    /// ```
    #[inline]
    pub const fn cast_signed(self) -> Int<N> {
        Int { bits: self }
    }
}

impl<const N: usize> Default for Int<N> {
    /// Zero.
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> Ord for Int<N> {
    /// Compares the values: a negative one is below every other.
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

impl<const N: usize> PartialOrd for Int<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> fmt::Debug for Int<N> {
    /// Writes the sign and the absolute value in hexadecimal, `-0x5`, with the formatter's width,
    /// fill, alignment and the flags `+` and `0` applied as `{:#x}` applies them to a sign and a
    /// magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Int;
    ///
    /// let x = Int::<2>::from_i64(-5);
    /// assert_eq!(format!("{x:?}"), "-0x5");
    /// assert_eq!(format!("[{x:>6?}] [{x:06?}] [{:+?}]", Int::<2>::ONE), "[  -0x5] [-0x005] [+0x1]");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::text::debug_hex(f, self.is_negative(), self.unsigned_abs())
    }
}
