//! The unsigned fixed-width integer [`Uint<N>`]: its constants and limbs, its arithmetic in the
//! forms of the primitive integers (checked, wrapping, overflowing, saturating), the limb-level
//! helpers, powers, roots, logarithms, bits and bytes.

use super::limbs;
use super::signed::Int;
use core::cmp::Ordering;
use core::fmt;

/// An unsigned integer of `N` 64-bit limbs, `64·N` bits, on the stack.
///
/// The limbs are kept the low one first, so [`from_limbs`](Self::from_limbs) of `[1, 2]` is
/// `1 + 2·2⁶⁴`. It behaves as a `u{64·N}` would: see the [module](super) for the operators and
/// their overflow behaviour.
///
/// # Examples
///
/// ```
/// use a::math::uint::Uint;
///
/// let a = Uint::<4>::from_u64(u64::MAX);
/// let (sum, carry) = a.overflowing_add(Uint::ONE);
/// assert_eq!(sum.as_limbs(), &[0, 1, 0, 0]);
/// assert!(!carry);
/// assert_eq!(Uint::<4>::BITS, 256);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Uint<const N: usize> {
    /// The limbs, the low one first.
    limbs: [u64; N],
}

impl<const N: usize> Uint<N> {
    /// The width in bits, `64·N`.
    pub const BITS: u32 = 64 * N as u32;
    /// Zero.
    pub const ZERO: Self = Uint { limbs: [0; N] };
    /// One.
    pub const ONE: Self = {
        let mut limbs = [0; N];
        limbs[0] = 1;
        Uint { limbs }
    };
    /// The largest value, `2^(64·N) − 1`.
    pub const MAX: Self = Uint {
        limbs: [u64::MAX; N],
    };
    /// The smallest value, zero, as `u64::MIN` is.
    pub const MIN: Self = Self::ZERO;

    /// The integer of the limbs `limbs`, the low one first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_limbs([5, 0]), Uint::from_u64(5));
    /// ```
    #[inline]
    pub const fn from_limbs(limbs: [u64; N]) -> Self {
        Uint { limbs }
    }

    /// The limbs, the low one first.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).as_limbs(), &[7, 0]);
    /// ```
    #[inline]
    pub const fn as_limbs(&self) -> &[u64; N] {
        &self.limbs
    }

    /// The limbs, the low one first, to change in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let mut x = Uint::<2>::ZERO;
    /// x.as_limbs_mut()[1] = 1;
    /// assert_eq!(x, Uint::ONE << 64);
    /// ```
    #[inline]
    pub const fn as_limbs_mut(&mut self) -> &mut [u64; N] {
        &mut self.limbs
    }

    /// The limbs, the low one first, by value.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.into_limbs(), [u64::MAX; 2]);
    /// ```
    #[inline]
    pub const fn into_limbs(self) -> [u64; N] {
        self.limbs
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
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<3>::from_u64(9).as_limbs(), &[9, 0, 0]);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn from_u64(x: u64) -> Self {
        let mut limbs = [0; N];
        if N == 0 {
            assert!(x == 0, "the value does not fit in zero limbs");
        } else {
            limbs[0] = x;
        }
        Uint { limbs }
    }

    /// The integer `x`. `TryFrom<u128>` is the form that fails instead of panicking.
    ///
    /// # Panics
    ///
    /// If `x` does not fit: if `N` is 1 and `x ≥ 2⁶⁴`, or `N` is 0 and `x` is not 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// const X: Uint<2> = Uint::from_u128(1 << 100);
    /// assert_eq!(X.as_limbs(), &[0, 1 << 36]);
    /// assert_eq!(Uint::<1>::from_u128(7), Uint::from_u64(7));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn from_u128(x: u128) -> Self {
        let mut limbs = [0; N];
        let (low, high) = (x as u64, (x >> 64) as u64);
        if N >= 2 {
            limbs[0] = low;
            limbs[1] = high;
        } else if N == 1 {
            assert!(high == 0, "the value does not fit in one limb");
            limbs[0] = low;
        } else {
            assert!(x == 0, "the value does not fit in zero limbs");
        }
        Uint { limbs }
    }

    /// Whether it is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert!(Uint::<4>::ZERO.is_zero());
    /// assert!(!Uint::<4>::ONE.is_zero());
    /// ```
    #[inline]
    pub const fn is_zero(&self) -> bool {
        let mut i = 0;
        while i < N {
            if self.limbs[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// The number of significant bits: 0 for zero, else one more than the index of the highest
    /// set bit.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ZERO.bits(), 0);
    /// assert_eq!(Uint::<2>::from_limbs([0, 1]).bits(), 65);
    /// ```
    #[inline]
    pub const fn bits(&self) -> u32 {
        let mut i = N;
        while i > 0 {
            i -= 1;
            if self.limbs[i] != 0 {
                return 64 * i as u32 + (64 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Whether bit `i` (bit 0 the lowest) is set; `false` for `i ≥ BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<2>::from_limbs([0, 2]);
    /// assert!(x.bit(65));
    /// assert!(!x.bit(64));
    /// assert!(!x.bit(1000));
    /// ```
    #[inline]
    pub const fn bit(&self, i: u32) -> bool {
        if i >= Self::BITS {
            return false;
        }
        (self.limbs[(i / 64) as usize] >> (i % 64)) & 1 == 1
    }

    /// The order of `self` and `rhs`, as `Ord::cmp` gives it, in a `const fn`.
    #[inline]
    pub(super) const fn compare(&self, rhs: &Self) -> Ordering {
        limbs::compare(&self.limbs, &rhs.limbs)
    }

    // ---------------------------------------------------------------------------------------
    // Sums and differences.

    /// `self + rhs` modulo `2^(64·N)` and whether it wrapped.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.overflowing_add(Uint::ONE), (Uint::ZERO, true));
    /// ```
    #[inline]
    pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.carrying_add(rhs, false)
    }

    /// `self − rhs` modulo `2^(64·N)` and whether it wrapped (whether `rhs > self`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ZERO.overflowing_sub(Uint::ONE), (Uint::MAX, true));
    /// ```
    #[inline]
    pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        self.borrowing_sub(rhs, false)
    }

    /// `self + rhs + carry` modulo `2^(64·N)` and the carry out, as the unstable
    /// `u64::carrying_add` does: the step of a sum of integers of several `Uint`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.carrying_add(Uint::ZERO, true), (Uint::ZERO, true));
    /// assert_eq!(Uint::<2>::ONE.carrying_add(Uint::ONE, true), (Uint::from_u64(3), false));
    /// ```
    #[inline]
    pub const fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool) {
        let mut limbs = [0; N];
        let mut carry = carry;
        let mut i = 0;
        while i < N {
            let (s, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (s, c2) = s.overflowing_add(carry as u64);
            limbs[i] = s;
            carry = c1 | c2;
            i += 1;
        }
        (Uint { limbs }, carry)
    }

    /// `self − rhs − borrow` modulo `2^(64·N)` and the borrow out, as the unstable
    /// `u64::borrowing_sub` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ZERO.borrowing_sub(Uint::ZERO, true), (Uint::MAX, true));
    /// assert_eq!(Uint::<2>::ONE.borrowing_sub(Uint::ZERO, true), (Uint::ZERO, false));
    /// ```
    #[inline]
    pub const fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
        let mut limbs = [0; N];
        let mut borrow = borrow;
        let mut i = 0;
        while i < N {
            let (d, b1) = self.limbs[i].overflowing_sub(rhs.limbs[i]);
            let (d, b2) = d.overflowing_sub(borrow as u64);
            limbs[i] = d;
            borrow = b1 | b2;
            i += 1;
        }
        (Uint { limbs }, borrow)
    }

    /// `self + rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.checked_add(Uint::ONE), Some(Uint::from_u64(2)));
    /// assert_eq!(Uint::<2>::MAX.checked_add(Uint::ONE), None);
    /// ```
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.overflowing_add(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// `self + rhs` modulo `2^(64·N)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.wrapping_add(Uint::from_u64(2)), Uint::ONE);
    /// ```
    #[inline]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }

    /// `self + rhs`, or [`MAX`](Self::MAX) if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.saturating_add(Uint::ONE), Uint::MAX);
    /// ```
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        match self.overflowing_add(rhs) {
            (s, false) => s,
            _ => Self::MAX,
        }
    }

    /// `self − rhs`, or `None` if `rhs > self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.checked_sub(Uint::ONE), Some(Uint::ZERO));
    /// assert_eq!(Uint::<2>::ZERO.checked_sub(Uint::ONE), None);
    /// ```
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        match self.overflowing_sub(rhs) {
            (d, false) => Some(d),
            _ => None,
        }
    }

    /// `self − rhs` modulo `2^(64·N)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ZERO.wrapping_sub(Uint::ONE), Uint::MAX);
    /// ```
    #[inline]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    /// `self − rhs`, or zero if `rhs > self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.saturating_sub(Uint::MAX), Uint::ZERO);
    /// ```
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        match self.overflowing_sub(rhs) {
            (d, false) => d,
            _ => Self::ZERO,
        }
    }

    /// `self + rhs` for a signed `rhs`, modulo `2^(64·N)`, and whether the true sum is out of
    /// range, as `u64::overflowing_add_signed` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// let one = Uint::<2>::ONE;
    /// assert_eq!(one.overflowing_add_signed(Int::from_i64(-1)), (Uint::ZERO, false));
    /// assert_eq!(one.overflowing_add_signed(Int::from_i64(-2)), (Uint::MAX, true));
    /// ```
    #[inline]
    pub const fn overflowing_add_signed(self, rhs: Int<N>) -> (Self, bool) {
        // The sum of the bits wraps exactly when the true sum is out of range, but for a negative
        // rhs, whose bits are 2^(64N) too many: then it is the other way round.
        let (sum, carry) = self.overflowing_add(rhs.cast_unsigned());
        (sum, carry != rhs.is_negative())
    }

    /// `self + rhs` for a signed `rhs`, or `None` if the true sum is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Uint::<2>::ONE.checked_add_signed(Int::from_i64(-1)), Some(Uint::ZERO));
    /// assert_eq!(Uint::<2>::ZERO.checked_add_signed(Int::from_i64(-1)), None);
    /// ```
    #[inline]
    pub const fn checked_add_signed(self, rhs: Int<N>) -> Option<Self> {
        match self.overflowing_add_signed(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// `self + rhs` for a signed `rhs`, modulo `2^(64·N)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Uint::<2>::ZERO.wrapping_add_signed(Int::from_i64(-1)), Uint::MAX);
    /// ```
    #[inline]
    pub const fn wrapping_add_signed(self, rhs: Int<N>) -> Self {
        self.overflowing_add_signed(rhs).0
    }

    /// `self + rhs` for a signed `rhs`, held to the range: zero below it, [`MAX`](Self::MAX)
    /// above.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Int, Uint};
    ///
    /// assert_eq!(Uint::<2>::ONE.saturating_add_signed(Int::from_i64(-5)), Uint::ZERO);
    /// assert_eq!(Uint::<2>::MAX.saturating_add_signed(Int::ONE), Uint::MAX);
    /// ```
    #[inline]
    pub const fn saturating_add_signed(self, rhs: Int<N>) -> Self {
        match self.overflowing_add_signed(rhs) {
            (s, false) => s,
            _ if rhs.is_negative() => Self::ZERO,
            _ => Self::MAX,
        }
    }

    /// `−self` modulo `2^(64·N)`, and whether `self` is not zero (whether the true negation is out
    /// of range), as `u64::overflowing_neg` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.overflowing_neg(), (Uint::MAX, true));
    /// assert_eq!(Uint::<2>::ZERO.overflowing_neg(), (Uint::ZERO, false));
    /// ```
    #[inline]
    pub const fn overflowing_neg(self) -> (Self, bool) {
        Self::ZERO.overflowing_sub(self)
    }

    /// `−self`, which is `Some` only for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ZERO.checked_neg(), Some(Uint::ZERO));
    /// assert_eq!(Uint::<2>::ONE.checked_neg(), None);
    /// ```
    #[inline]
    pub const fn checked_neg(self) -> Option<Self> {
        match self.overflowing_neg() {
            (n, false) => Some(n),
            _ => None,
        }
    }

    /// `−self` modulo `2^(64·N)`: `2^(64·N) − self`, and zero for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.wrapping_neg(), Uint::MAX);
    /// ```
    #[inline]
    pub const fn wrapping_neg(self) -> Self {
        self.overflowing_neg().0
    }

    // ---------------------------------------------------------------------------------------
    // Products.

    /// The full product `self · rhs`, as its low and high halves: `self · rhs = low + high ·
    /// 2^(64·N)`, by the schoolbook method, `N²` limb products.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let (low, high) = Uint::<2>::MAX.widening_mul(Uint::MAX);
    /// assert_eq!(low, Uint::ONE);
    /// assert_eq!(high, Uint::MAX.overflowing_sub(Uint::ONE).0);
    /// ```
    #[inline]
    pub const fn widening_mul(self, rhs: Self) -> (Self, Self) {
        self.carrying_mul(rhs, Self::ZERO)
    }

    /// The full `self · rhs + carry`, as its low and high halves, as the unstable
    /// `u64::carrying_mul` does; it never overflows the two halves.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let (low, high) = Uint::<2>::MAX.carrying_mul(Uint::MAX, Uint::MAX);
    /// assert_eq!((low, high), (Uint::ZERO, Uint::MAX));
    /// ```
    pub const fn carrying_mul(self, rhs: Self, carry: Self) -> (Self, Self) {
        let mut low = carry.limbs;
        let mut high = [0; N];
        let mut i = 0;
        while i < N {
            let a = self.limbs[i];
            let mut c: u64 = 0;
            let mut j = 0;
            while j < N {
                let k = i + j;
                if k < N {
                    (low[k], c) = limbs::mac(low[k], a, rhs.limbs[j], c);
                } else {
                    (high[k - N], c) = limbs::mac(high[k - N], a, rhs.limbs[j], c);
                }
                j += 1;
            }
            // The limb at i + N is not written by a row before; the carry of the addend `carry`
            // went into row 0 with the low limbs.
            high[i] = c;
            i += 1;
        }
        (Uint { limbs: low }, Uint { limbs: high })
    }

    /// The full product of `self` and one limb `rhs`: `self · rhs = low + high · 2^(64·N)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.widening_mul_u64(2), (Uint::MAX - Uint::ONE, 1));
    /// ```
    #[inline]
    pub const fn widening_mul_u64(self, rhs: u64) -> (Self, u64) {
        let (limbs, high) = limbs::mul_add_1(&self.limbs, rhs, 0);
        (Uint { limbs }, high)
    }

    /// `self · rhs` modulo `2^(64·N)` and whether it wrapped, by the schoolbook method over the
    /// limbs that land below the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<2>::ONE << 64u32;
    /// assert_eq!(x.overflowing_mul(x), (Uint::ZERO, true));
    /// assert_eq!(x.overflowing_mul(Uint::from_u64(3)), (x * Uint::from_u64(3), false));
    /// ```
    #[inline]
    pub const fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        let (limbs, overflow) = limbs::mul_low(&self.limbs, &rhs.limbs);
        (Uint { limbs }, overflow)
    }

    /// `self · rhs`, or `None` if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.checked_mul(Uint::ONE), Some(Uint::MAX));
    /// assert_eq!(Uint::<2>::MAX.checked_mul(Uint::from_u64(2)), None);
    /// ```
    #[inline]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        match self.overflowing_mul(rhs) {
            (p, false) => Some(p),
            _ => None,
        }
    }

    /// `self · rhs` modulo `2^(64·N)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.wrapping_mul(Uint::MAX), Uint::ONE);
    /// ```
    #[inline]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        self.overflowing_mul(rhs).0
    }

    /// `self · rhs`, or [`MAX`](Self::MAX) if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.saturating_mul(Uint::from_u64(2)), Uint::MAX);
    /// ```
    #[inline]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        match self.overflowing_mul(rhs) {
            (p, false) => p,
            _ => Self::MAX,
        }
    }

    // ---------------------------------------------------------------------------------------
    // Quotients and remainders.

    /// The quotient and the remainder of `self` by `rhs`, `(self / rhs, self % rhs)`, computed
    /// together: by Knuth's algorithm D, or by the plain loop for a divisor of one limb.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let (q, r) = Uint::<2>::MAX.div_rem(Uint::from_u64(10));
    /// assert_eq!(q * Uint::from_u64(10) + r, Uint::MAX);
    /// assert_eq!(r, Uint::from_u64(5));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn div_rem(self, rhs: Self) -> (Self, Self) {
        if rhs.is_zero() {
            panic!("attempt to divide by zero");
        }
        let (q, r) = limbs::div_rem(&self.limbs, &rhs.limbs);
        (Uint { limbs: q }, Uint { limbs: r })
    }

    /// The quotient and the remainder of `self` by one limb `rhs`, by the plain loop.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let (q, r) = Uint::<2>::MAX.div_rem_u64(10);
    /// assert_eq!(q, Uint::MAX / Uint::from_u64(10));
    /// assert_eq!(r, 5);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn div_rem_u64(self, rhs: u64) -> (Self, u64) {
        if rhs == 0 {
            panic!("attempt to divide by zero");
        }
        let (q, r) = limbs::div_rem_1(&self.limbs, rhs);
        (Uint { limbs: q }, r)
    }

    /// `self / rhs`, or `None` if `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).checked_div(Uint::from_u64(2)), Some(Uint::from_u64(3)));
    /// assert_eq!(Uint::<2>::ONE.checked_div(Uint::ZERO), None);
    /// ```
    #[inline]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() {
            None
        } else {
            Some(self.div_rem(rhs).0)
        }
    }

    /// `self / rhs`, which never wraps for unsigned integers.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).wrapping_div(Uint::from_u64(2)), Uint::from_u64(3));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_div(self, rhs: Self) -> Self {
        self.div_rem(rhs).0
    }

    /// `self / rhs` and `false`: an unsigned quotient never overflows.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).overflowing_div(Uint::from_u64(2)), (Uint::from_u64(3), false));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_div(self, rhs: Self) -> (Self, bool) {
        (self.div_rem(rhs).0, false)
    }

    /// `self / rhs`, which never saturates for unsigned integers.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.saturating_div(Uint::MAX), Uint::ONE);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        self.div_rem(rhs).0
    }

    /// `self % rhs`, or `None` if `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).checked_rem(Uint::from_u64(2)), Some(Uint::ONE));
    /// assert_eq!(Uint::<2>::ONE.checked_rem(Uint::ZERO), None);
    /// ```
    #[inline]
    pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() {
            None
        } else {
            Some(self.div_rem(rhs).1)
        }
    }

    /// `self % rhs`, which never wraps for unsigned integers.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).wrapping_rem(Uint::from_u64(2)), Uint::ONE);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_rem(self, rhs: Self) -> Self {
        self.rem_nonzero(rhs)
    }

    /// `self % rhs` and `false`: an unsigned remainder never overflows.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).overflowing_rem(Uint::from_u64(2)), (Uint::ONE, false));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
        (self.rem_nonzero(rhs), false)
    }

    /// `self % rhs`, panicking with the remainder's message if `rhs` is zero.
    #[inline]
    #[track_caller]
    pub(super) const fn rem_nonzero(self, rhs: Self) -> Self {
        if rhs.is_zero() {
            panic!("attempt to calculate the remainder with a divisor of zero");
        }
        Uint {
            limbs: limbs::div_rem(&self.limbs, &rhs.limbs).1,
        }
    }

    /// Euclid's quotient, which for unsigned integers is `self / rhs`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).div_euclid(Uint::from_u64(2)), Uint::from_u64(3));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn div_euclid(self, rhs: Self) -> Self {
        self.div_rem(rhs).0
    }

    /// Euclid's quotient, or `None` if `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.checked_div_euclid(Uint::ZERO), None);
    /// ```
    #[inline]
    pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
        self.checked_div(rhs)
    }

    /// Euclid's quotient, which never wraps for unsigned integers.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).wrapping_div_euclid(Uint::from_u64(2)), Uint::from_u64(3));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_div_euclid(self, rhs: Self) -> Self {
        self.div_rem(rhs).0
    }

    /// Euclid's quotient and `false`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let seven = Uint::<2>::from_u64(7);
    /// assert_eq!(seven.overflowing_div_euclid(Uint::from_u64(2)), (Uint::from_u64(3), false));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
        (self.div_rem(rhs).0, false)
    }

    /// Euclid's remainder, which for unsigned integers is `self % rhs`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).rem_euclid(Uint::from_u64(2)), Uint::ONE);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn rem_euclid(self, rhs: Self) -> Self {
        self.rem_nonzero(rhs)
    }

    /// Euclid's remainder, or `None` if `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.checked_rem_euclid(Uint::ZERO), None);
    /// ```
    #[inline]
    pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
        self.checked_rem(rhs)
    }

    /// Euclid's remainder, which never wraps for unsigned integers.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).wrapping_rem_euclid(Uint::from_u64(2)), Uint::ONE);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn wrapping_rem_euclid(self, rhs: Self) -> Self {
        self.rem_nonzero(rhs)
    }

    /// Euclid's remainder and `false`.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).overflowing_rem_euclid(Uint::from_u64(2)), (Uint::ONE, false));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
        (self.rem_nonzero(rhs), false)
    }

    /// `self / rhs` rounded up.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).div_ceil(Uint::from_u64(2)), Uint::from_u64(4));
    /// assert_eq!(Uint::<2>::from_u64(8).div_ceil(Uint::from_u64(2)), Uint::from_u64(4));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn div_ceil(self, rhs: Self) -> Self {
        let (q, r) = self.div_rem(rhs);
        if r.is_zero() {
            q
        } else {
            // q < MAX, as rhs ≥ 2 here.
            q.wrapping_add(Self::ONE)
        }
    }

    /// The smallest multiple of `rhs` at least `self`. As an operator, it panics on overflow when
    /// debug assertions are on, and wraps when they are off.
    ///
    /// # Panics
    ///
    /// If `rhs` is zero, or (with debug assertions) if the multiple overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).next_multiple_of(Uint::from_u64(4)), Uint::from_u64(8));
    /// assert_eq!(Uint::<2>::from_u64(8).next_multiple_of(Uint::from_u64(4)), Uint::from_u64(8));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn next_multiple_of(self, rhs: Self) -> Self {
        let r = self.rem_nonzero(rhs);
        if r.is_zero() {
            self
        } else {
            debug_checked!(
                self.overflowing_add(rhs.wrapping_sub(r)),
                "attempt to add with overflow"
            )
        }
    }

    /// The smallest multiple of `rhs` at least `self`, or `None` if `rhs` is zero or the multiple
    /// overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(7).checked_next_multiple_of(Uint::from_u64(4)), Some(Uint::from_u64(8)));
    /// assert_eq!(Uint::<2>::MAX.checked_next_multiple_of(Uint::from_u64(2)), None);
    /// ```
    #[inline]
    pub const fn checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() {
            return None;
        }
        let r = self.rem_nonzero(rhs);
        if r.is_zero() {
            Some(self)
        } else {
            self.checked_add(rhs.wrapping_sub(r))
        }
    }

    /// Whether `self` is a multiple of `rhs`: of zero only zero is.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert!(Uint::<2>::from_u64(8).is_multiple_of(Uint::from_u64(4)));
    /// assert!(!Uint::<2>::from_u64(7).is_multiple_of(Uint::from_u64(4)));
    /// assert!(Uint::<2>::ZERO.is_multiple_of(Uint::ZERO));
    /// ```
    #[inline]
    pub const fn is_multiple_of(self, rhs: Self) -> bool {
        if rhs.is_zero() {
            self.is_zero()
        } else {
            self.rem_nonzero(rhs).is_zero()
        }
    }

    /// `|self − rhs|`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.abs_diff(Uint::MAX), Uint::MAX - Uint::ONE);
    /// ```
    #[inline]
    pub const fn abs_diff(self, rhs: Self) -> Self {
        if self.compare(&rhs).is_lt() {
            rhs.wrapping_sub(self)
        } else {
            self.wrapping_sub(rhs)
        }
    }

    /// `(self + rhs) / 2` rounded down, without overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.midpoint(Uint::ONE), Uint::ONE << 127);
    /// assert_eq!(Uint::<2>::ZERO.midpoint(Uint::ONE), Uint::ZERO);
    /// ```
    #[inline]
    pub const fn midpoint(self, rhs: Self) -> Self {
        // a + b = 2(a & b) + (a ^ b).
        let mut limbs = [0; N];
        let mut i = 0;
        while i < N {
            limbs[i] = self.limbs[i] ^ rhs.limbs[i];
            i += 1;
        }
        let half = Uint { limbs }.unbounded_shr(1);
        i = 0;
        while i < N {
            limbs[i] = self.limbs[i] & rhs.limbs[i];
            i += 1;
        }
        Uint { limbs }.wrapping_add(half)
    }

    // ---------------------------------------------------------------------------------------
    // Shifts.

    /// `self << (rhs mod BITS)` and whether `rhs ≥ BITS`, as `u64::overflowing_shl` does: the
    /// shift amount is reduced modulo the width (which, for a width that is a power of two, is
    /// the primitives' masking of its low bits), and the bits shifted out are lost either way.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.overflowing_shl(127), (Uint::ONE << 127, false));
    /// assert_eq!(Uint::<2>::ONE.overflowing_shl(129), (Uint::from_u64(2), true));
    /// assert_eq!(Uint::<3>::ONE.overflowing_shl(193), (Uint::from_u64(2), true));
    /// ```
    #[inline]
    pub const fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
        if Self::BITS == 0 {
            return (self, true);
        }
        let k = rhs % Self::BITS;
        (
            Uint {
                limbs: limbs::shl(&self.limbs, k),
            },
            rhs >= Self::BITS,
        )
    }

    /// `self << rhs`, or `None` if `rhs ≥ BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.checked_shl(64), Some(Uint::from_limbs([0, 1])));
    /// assert_eq!(Uint::<2>::ONE.checked_shl(128), None);
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
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.wrapping_shl(130), Uint::from_u64(4));
    /// ```
    #[inline]
    pub const fn wrapping_shl(self, rhs: u32) -> Self {
        self.overflowing_shl(rhs).0
    }

    /// `self << rhs`, zero if `rhs ≥ BITS`, as `u64::unbounded_shl` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.unbounded_shl(127), Uint::ONE << 127);
    /// assert_eq!(Uint::<2>::ONE.unbounded_shl(128), Uint::ZERO);
    /// ```
    #[inline]
    pub const fn unbounded_shl(self, rhs: u32) -> Self {
        if rhs >= Self::BITS {
            Self::ZERO
        } else {
            Uint {
                limbs: limbs::shl(&self.limbs, rhs),
            }
        }
    }

    /// `self >> (rhs mod BITS)` and whether `rhs ≥ BITS`, as `u64::overflowing_shr` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.overflowing_shr(127), (Uint::ONE, false));
    /// assert_eq!(Uint::<2>::MAX.overflowing_shr(255), (Uint::ONE, true));
    /// ```
    #[inline]
    pub const fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
        if Self::BITS == 0 {
            return (self, true);
        }
        let k = rhs % Self::BITS;
        (
            Uint {
                limbs: limbs::shr(&self.limbs, k, 0),
            },
            rhs >= Self::BITS,
        )
    }

    /// `self >> rhs`, or `None` if `rhs ≥ BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.checked_shr(127), Some(Uint::ONE));
    /// assert_eq!(Uint::<2>::MAX.checked_shr(128), None);
    /// ```
    #[inline]
    pub const fn checked_shr(self, rhs: u32) -> Option<Self> {
        match self.overflowing_shr(rhs) {
            (s, false) => Some(s),
            _ => None,
        }
    }

    /// `self >> (rhs mod BITS)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.wrapping_shr(255), Uint::ONE);
    /// ```
    #[inline]
    pub const fn wrapping_shr(self, rhs: u32) -> Self {
        self.overflowing_shr(rhs).0
    }

    /// `self >> rhs`, zero if `rhs ≥ BITS`, as `u64::unbounded_shr` does.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.unbounded_shr(127), Uint::ONE);
    /// assert_eq!(Uint::<2>::MAX.unbounded_shr(128), Uint::ZERO);
    /// ```
    #[inline]
    pub const fn unbounded_shr(self, rhs: u32) -> Self {
        if rhs >= Self::BITS {
            Self::ZERO
        } else {
            Uint {
                limbs: limbs::shr(&self.limbs, rhs, 0),
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Powers, roots and logarithms.

    /// `self^exp` modulo `2^(64·N)` and whether the true power is out of range, by squaring and
    /// multiplying from the low bit of `exp` up. `0⁰ = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let three = Uint::<2>::from_u64(3);
    /// assert_eq!(three.overflowing_pow(80), (Uint::from_u128(3u128.pow(80)), false));
    /// assert_eq!(three.overflowing_pow(81).1, true);
    /// ```
    pub const fn overflowing_pow(self, exp: u32) -> (Self, bool) {
        if exp == 0 {
            return (Self::ONE, false);
        }
        // Once a square or a product wraps, the true power is out of range (for a base of at least
        // 2 every partial product divides it), so the flag is exact.
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
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(2).checked_pow(127), Some(Uint::ONE << 127));
    /// assert_eq!(Uint::<2>::from_u64(2).checked_pow(128), None);
    /// ```
    #[inline]
    pub const fn checked_pow(self, exp: u32) -> Option<Self> {
        match self.overflowing_pow(exp) {
            (p, false) => Some(p),
            _ => None,
        }
    }

    /// `self^exp` modulo `2^(64·N)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(2).wrapping_pow(128), Uint::ZERO);
    /// assert_eq!(Uint::<2>::MAX.wrapping_pow(3), Uint::MAX);
    /// ```
    #[inline]
    pub const fn wrapping_pow(self, exp: u32) -> Self {
        self.overflowing_pow(exp).0
    }

    /// `self^exp`, or [`MAX`](Self::MAX) if it overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(2).saturating_pow(128), Uint::MAX);
    /// ```
    #[inline]
    pub const fn saturating_pow(self, exp: u32) -> Self {
        match self.overflowing_pow(exp) {
            (p, false) => p,
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
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<4>::from_u64(10).pow(30), Uint::from_u128(10u128.pow(30)));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn pow(self, exp: u32) -> Self {
        debug_checked!(
            self.overflowing_pow(exp),
            "attempt to exponentiate with overflow"
        )
    }

    /// The square root rounded down, by Newton's iteration from a power of two above the root,
    /// which decreases to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.isqrt(), Uint::from_u64(u64::MAX));
    /// assert_eq!(Uint::<2>::from_u64(99).isqrt(), Uint::from_u64(9));
    /// ```
    pub const fn isqrt(self) -> Self {
        let bits = self.bits();
        if bits <= 1 {
            return self;
        }
        // 2^⌈bits/2⌉ is above the root; from above, x ↦ ⌊(x + ⌊n/x⌋)/2⌋ decreases strictly until
        // it reaches ⌊√n⌋ (Brent and Zimmermann, Modern Computer Arithmetic, 1.5.1). The sum is
        // below 2^(bits/2 + 2), which fits.
        let mut x = Self::ONE.unbounded_shl(bits.div_ceil(2));
        loop {
            let y = x.wrapping_add(self.div_rem(x).0).unbounded_shr(1);
            if !y.compare(&x).is_lt() {
                return x;
            }
            x = y;
        }
    }

    /// The base-2 logarithm rounded down: the index of the highest set bit.
    ///
    /// # Panics
    ///
    /// If `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.ilog2(), 127);
    /// assert_eq!(Uint::<2>::ONE.ilog2(), 0);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn ilog2(self) -> u32 {
        match self.checked_ilog2() {
            Some(l) => l,
            None => panic!("argument of integer logarithm must be positive"),
        }
    }

    /// The base-2 logarithm rounded down, or `None` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(1024).checked_ilog2(), Some(10));
    /// assert_eq!(Uint::<2>::ZERO.checked_ilog2(), None);
    /// ```
    #[inline]
    pub const fn checked_ilog2(self) -> Option<u32> {
        match self.bits() {
            0 => None,
            b => Some(b - 1),
        }
    }

    /// The base-10 logarithm rounded down: the number of decimal digits less one.
    ///
    /// # Panics
    ///
    /// If `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.ilog10(), 38);
    /// assert_eq!(Uint::<2>::from_u64(999).ilog10(), 2);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn ilog10(self) -> u32 {
        match self.checked_ilog10() {
            Some(l) => l,
            None => panic!("argument of integer logarithm must be positive"),
        }
    }

    /// The base-10 logarithm rounded down, or `None` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(1000).checked_ilog10(), Some(3));
    /// assert_eq!(Uint::<2>::ZERO.checked_ilog10(), None);
    /// ```
    #[inline]
    pub const fn checked_ilog10(self) -> Option<u32> {
        self.checked_ilog_u64(10)
    }

    /// The logarithm to `base` rounded down: the largest `k` with `base^k ≤ self`.
    ///
    /// # Panics
    ///
    /// If `self` is zero or `base` is below 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.ilog(Uint::from_u64(3)), 80);
    /// assert_eq!(Uint::<2>::MAX.ilog(Uint::ONE << 64), 1);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn ilog(self, base: Self) -> u32 {
        if base.compare(&Self::ONE).is_le() {
            panic!("base of integer logarithm must be at least 2");
        }
        match self.checked_ilog(base) {
            Some(l) => l,
            None => panic!("argument of integer logarithm must be positive"),
        }
    }

    /// The logarithm to `base` rounded down, or `None` if `self` is zero or `base` is below 2.
    ///
    /// A base of one limb divides by its largest power that fits in a limb while it can, `O(N²)`
    /// in all; a longer base divides by the base itself, at most `BITS / 64` times.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(80).checked_ilog(Uint::from_u64(3)), Some(3));
    /// assert_eq!(Uint::<2>::ZERO.checked_ilog(Uint::from_u64(3)), None);
    /// assert_eq!(Uint::<2>::from_u64(5).checked_ilog(Uint::ONE), None);
    /// ```
    pub const fn checked_ilog(self, base: Self) -> Option<u32> {
        if self.is_zero() || base.compare(&Self::ONE).is_le() {
            return None;
        }
        if limbs::significant(&base.limbs) == 1 {
            return self.checked_ilog_u64(base.limbs[0]);
        }
        let mut x = self;
        let mut k = 0;
        while !x.compare(&base).is_lt() {
            x = x.div_rem(base).0;
            k += 1;
        }
        Some(k)
    }

    /// The logarithm to a base `2 ≤ b < 2⁶⁴` rounded down, or `None` for zero: the quotients by
    /// `B = b^j`, the largest power of `b` that fits in a limb, counted `j` at a time, then the
    /// last limb's logarithm; `⌊log_b ⌊x/B⌋⌋ = ⌊log_b x⌋ − j` as `⌊⌊x/B⌋/c⌋ = ⌊x/(Bc)⌋`.
    const fn checked_ilog_u64(self, b: u64) -> Option<u32> {
        if self.is_zero() {
            return None;
        }
        let mut big = b;
        let mut j = 1;
        while let Some(next) = big.checked_mul(b) {
            big = next;
            j += 1;
        }
        let mut x = self;
        let mut k = 0;
        while limbs::significant(&x.limbs) > 1 || x.limbs[0] >= big {
            x = x.div_rem_u64(big).0;
            k += j;
        }
        Some(k + x.limbs[0].ilog(b))
    }

    // ---------------------------------------------------------------------------------------
    // Powers of two.

    /// Whether it is a power of two: exactly one bit set.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert!((Uint::<2>::ONE << 100u32).is_power_of_two());
    /// assert!(!Uint::<2>::ZERO.is_power_of_two());
    /// assert!(!Uint::<2>::from_u64(6).is_power_of_two());
    /// ```
    #[inline]
    pub const fn is_power_of_two(self) -> bool {
        self.count_ones() == 1
    }

    /// The smallest power of two at least `self`, or `None` if it is beyond the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(5).checked_next_power_of_two(), Some(Uint::from_u64(8)));
    /// assert_eq!(Uint::<2>::ZERO.checked_next_power_of_two(), Some(Uint::ONE));
    /// assert_eq!(((Uint::<2>::ONE << 127u32) + Uint::ONE).checked_next_power_of_two(), None);
    /// ```
    #[inline]
    pub const fn checked_next_power_of_two(self) -> Option<Self> {
        match self.overflowing_next_power_of_two() {
            (p, false) => Some(p),
            _ => None,
        }
    }

    /// The smallest power of two at least `self`. As the primitives', it panics when that is
    /// beyond the width and debug assertions are on, and gives zero when they are off.
    ///
    /// # Panics
    ///
    /// With debug assertions, if the power is beyond the width.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(8).next_power_of_two(), Uint::from_u64(8));
    /// assert_eq!(Uint::<2>::from_u64(9).next_power_of_two(), Uint::from_u64(16));
    /// ```
    #[inline]
    #[track_caller]
    pub const fn next_power_of_two(self) -> Self {
        debug_checked!(
            self.overflowing_next_power_of_two(),
            "attempt to add with overflow"
        )
    }

    /// The smallest power of two at least `self`, zero when it is beyond the width, and whether
    /// it was.
    const fn overflowing_next_power_of_two(self) -> (Self, bool) {
        if self.compare(&Self::ONE).is_le() {
            return (Self::ONE, false);
        }
        let k = self.wrapping_sub(Self::ONE).bits();
        if k >= Self::BITS {
            (Self::ZERO, true)
        } else {
            (Self::ONE.unbounded_shl(k), false)
        }
    }

    // ---------------------------------------------------------------------------------------
    // Bits.

    /// The number of set bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::MAX.count_ones(), 128);
    /// assert_eq!(Uint::<2>::from_u64(5).count_ones(), 2);
    /// ```
    #[inline]
    pub const fn count_ones(self) -> u32 {
        let mut c = 0;
        let mut i = 0;
        while i < N {
            c += self.limbs[i].count_ones();
            i += 1;
        }
        c
    }

    /// The number of clear bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(5).count_zeros(), 126);
    /// ```
    #[inline]
    pub const fn count_zeros(self) -> u32 {
        Self::BITS - self.count_ones()
    }

    /// The number of clear bits above the highest set bit: `BITS` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.leading_zeros(), 127);
    /// assert_eq!(Uint::<2>::ZERO.leading_zeros(), 128);
    /// ```
    #[inline]
    pub const fn leading_zeros(self) -> u32 {
        Self::BITS - self.bits()
    }

    /// The number of clear bits below the lowest set bit: `BITS` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!((Uint::<2>::ONE << 100u32).trailing_zeros(), 100);
    /// assert_eq!(Uint::<2>::ZERO.trailing_zeros(), 128);
    /// ```
    #[inline]
    pub const fn trailing_zeros(self) -> u32 {
        let mut i = 0;
        while i < N {
            if self.limbs[i] != 0 {
                return 64 * i as u32 + self.limbs[i].trailing_zeros();
            }
            i += 1;
        }
        Self::BITS
    }

    /// The number of set bits above the highest clear bit: `BITS` for `MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!((Uint::<2>::MAX << 100u32).leading_ones(), 28);
    /// assert_eq!(Uint::<2>::MAX.leading_ones(), 128);
    /// ```
    #[inline]
    pub const fn leading_ones(self) -> u32 {
        self.not_bits().leading_zeros()
    }

    /// The number of set bits below the lowest clear bit: `BITS` for `MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(0b0111).trailing_ones(), 3);
    /// assert_eq!(Uint::<2>::MAX.trailing_ones(), 128);
    /// ```
    #[inline]
    pub const fn trailing_ones(self) -> u32 {
        self.not_bits().trailing_zeros()
    }

    /// Every bit inverted, `!self`, in a `const fn`.
    #[inline]
    pub(super) const fn not_bits(self) -> Self {
        let mut limbs = [0; N];
        let mut i = 0;
        while i < N {
            limbs[i] = !self.limbs[i];
            i += 1;
        }
        Uint { limbs }
    }

    /// The bits rotated left by `n mod BITS`: those shifted out at the top come in at the
    /// bottom.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<2>::from_limbs([1, 1 << 63]);
    /// assert_eq!(x.rotate_left(1), Uint::from_limbs([3, 0]));
    /// assert_eq!(x.rotate_left(129), x.rotate_left(1));
    /// ```
    #[inline]
    pub const fn rotate_left(self, n: u32) -> Self {
        if Self::BITS == 0 {
            return self;
        }
        let k = n % Self::BITS;
        if k == 0 {
            return self;
        }
        let high = limbs::shl(&self.limbs, k);
        let low = limbs::shr(&self.limbs, Self::BITS - k, 0);
        let mut limbs = [0; N];
        let mut i = 0;
        while i < N {
            limbs[i] = high[i] | low[i];
            i += 1;
        }
        Uint { limbs }
    }

    /// The bits rotated right by `n mod BITS`: those shifted out at the bottom come in at the
    /// top.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::from_u64(3).rotate_right(1), Uint::from_limbs([1, 1 << 63]));
    /// ```
    #[inline]
    pub const fn rotate_right(self, n: u32) -> Self {
        if Self::BITS == 0 {
            return self;
        }
        self.rotate_left(Self::BITS - n % Self::BITS)
    }

    /// The bytes in the opposite order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<2>::from_u64(0x12);
    /// assert_eq!(x.swap_bytes(), Uint::from_limbs([0, 0x12 << 56]));
    /// ```
    #[inline]
    pub const fn swap_bytes(self) -> Self {
        let mut limbs = [0; N];
        let mut i = 0;
        while i < N {
            limbs[i] = self.limbs[N - 1 - i].swap_bytes();
            i += 1;
        }
        Uint { limbs }
    }

    /// The bits in the opposite order: bit `i` becomes bit `BITS − 1 − i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<2>::ONE.reverse_bits(), Uint::ONE << 127);
    /// ```
    #[inline]
    pub const fn reverse_bits(self) -> Self {
        let mut limbs = [0; N];
        let mut i = 0;
        while i < N {
            limbs[i] = self.limbs[N - 1 - i].reverse_bits();
            i += 1;
        }
        Uint { limbs }
    }

    // ---------------------------------------------------------------------------------------
    // Bytes.

    /// The `B = 8·N` bytes, the least significant first, as `u64::to_le_bytes` gives them. The
    /// length is a const parameter, inferred from the context, as `[u8; 8·N]` cannot be named on
    /// stable Rust; generic code over `N` can use [`write_le_bytes`](Self::write_le_bytes).
    ///
    /// A length other than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let bytes: [u8; 16] = Uint::<2>::from_u64(0x0102).to_le_bytes();
    /// assert_eq!(bytes[..3], [2, 1, 0]);
    /// ```
    ///
    /// ```compile_fail
    /// use a::math::uint::Uint;
    ///
    /// let bytes: [u8; 15] = Uint::<2>::ONE.to_le_bytes();
    /// ```
    #[inline]
    pub const fn to_le_bytes<const B: usize>(self) -> [u8; B] {
        const { assert!(B == 8 * N, "the length of the bytes must be 8·N") };
        let mut out = [0; B];
        let mut i = 0;
        while i < B {
            out[i] = (self.limbs[i / 8] >> (8 * (i % 8))) as u8;
            i += 1;
        }
        out
    }

    /// The `B = 8·N` bytes, the most significant first, as `u64::to_be_bytes` gives them.
    ///
    /// A length other than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let bytes: [u8; 16] = Uint::<2>::from_u64(0x0102).to_be_bytes();
    /// assert_eq!(bytes[13..], [0, 1, 2]);
    /// ```
    #[inline]
    pub const fn to_be_bytes<const B: usize>(self) -> [u8; B] {
        const { assert!(B == 8 * N, "the length of the bytes must be 8·N") };
        let mut out = [0; B];
        let mut i = 0;
        while i < B {
            out[B - 1 - i] = (self.limbs[i / 8] >> (8 * (i % 8))) as u8;
            i += 1;
        }
        out
    }

    /// The integer of the `B = 8·N` bytes `bytes`, the least significant first.
    ///
    /// A length other than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let mut bytes = [0u8; 16];
    /// bytes[8] = 1;
    /// assert_eq!(Uint::<2>::from_le_bytes(bytes), Uint::ONE << 64);
    /// ```
    #[inline]
    pub const fn from_le_bytes<const B: usize>(bytes: [u8; B]) -> Self {
        const { assert!(B == 8 * N, "the length of the bytes must be 8·N") };
        let mut limbs = [0; N];
        let mut i = 0;
        while i < B {
            limbs[i / 8] |= (bytes[i] as u64) << (8 * (i % 8));
            i += 1;
        }
        Uint { limbs }
    }

    /// The integer of the `B = 8·N` bytes `bytes`, the most significant first.
    ///
    /// A length other than `8·N` is an error at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let mut bytes = [0u8; 16];
    /// bytes[15] = 7;
    /// assert_eq!(Uint::<2>::from_be_bytes(bytes), Uint::from_u64(7));
    /// ```
    #[inline]
    pub const fn from_be_bytes<const B: usize>(bytes: [u8; B]) -> Self {
        const { assert!(B == 8 * N, "the length of the bytes must be 8·N") };
        let mut limbs = [0; N];
        let mut i = 0;
        while i < B {
            limbs[i / 8] |= (bytes[B - 1 - i] as u64) << (8 * (i % 8));
            i += 1;
        }
        Uint { limbs }
    }

    /// Writes the `8·N` bytes, the least significant first, into `out`.
    ///
    /// # Panics
    ///
    /// If `out` is not `8·N` bytes long.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let mut out = [0xFFu8; 16];
    /// Uint::<2>::from_u64(0x0102).write_le_bytes(&mut out);
    /// assert_eq!(out[..3], [2, 1, 0]);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn write_le_bytes(self, out: &mut [u8]) {
        assert!(out.len() == 8 * N, "the length of the bytes must be 8·N");
        let mut i = 0;
        while i < out.len() {
            out[i] = (self.limbs[i / 8] >> (8 * (i % 8))) as u8;
            i += 1;
        }
    }

    /// Writes the `8·N` bytes, the most significant first, into `out`.
    ///
    /// # Panics
    ///
    /// If `out` is not `8·N` bytes long.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let mut out = [0xFFu8; 16];
    /// Uint::<2>::from_u64(0x0102).write_be_bytes(&mut out);
    /// assert_eq!(out[13..], [0, 1, 2]);
    /// ```
    #[inline]
    #[track_caller]
    pub const fn write_be_bytes(self, out: &mut [u8]) {
        assert!(out.len() == 8 * N, "the length of the bytes must be 8·N");
        let len = out.len();
        let mut i = 0;
        while i < len {
            out[len - 1 - i] = (self.limbs[i / 8] >> (8 * (i % 8))) as u8;
            i += 1;
        }
    }

    /// The integer of the bytes `bytes`, the least significant first, of any length, or `None` if
    /// it does not fit: if a byte beyond the first `8·N` is not zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<1>::from_le_slice(&[1, 2]), Some(Uint::from_u64(0x0201)));
    /// assert_eq!(Uint::<1>::from_le_slice(&[0; 9]), Some(Uint::ZERO));
    /// assert_eq!(Uint::<1>::from_le_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 1]), None);
    /// ```
    pub const fn from_le_slice(bytes: &[u8]) -> Option<Self> {
        let mut limbs = [0; N];
        let mut i = 0;
        while i < bytes.len() {
            if i < 8 * N {
                limbs[i / 8] |= (bytes[i] as u64) << (8 * (i % 8));
            } else if bytes[i] != 0 {
                return None;
            }
            i += 1;
        }
        Some(Uint { limbs })
    }

    /// The integer of the bytes `bytes`, the most significant first, of any length, or `None` if
    /// it does not fit: if a byte before the last `8·N` is not zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<1>::from_be_slice(&[1, 2]), Some(Uint::from_u64(0x0102)));
    /// assert_eq!(Uint::<1>::from_be_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0]), None);
    /// ```
    pub const fn from_be_slice(bytes: &[u8]) -> Option<Self> {
        let mut limbs = [0; N];
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            let b = bytes[len - 1 - i];
            if i < 8 * N {
                limbs[i / 8] |= (b as u64) << (8 * (i % 8));
            } else if b != 0 {
                return None;
            }
            i += 1;
        }
        Some(Uint { limbs })
    }
}

impl<const N: usize> Default for Uint<N> {
    /// Zero.
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> Ord for Uint<N> {
    /// Compares the values, from the high limb down.
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

impl<const N: usize> PartialOrd for Uint<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> fmt::Debug for Uint<N> {
    /// Writes the value in hexadecimal, `0x` and the significant digits, as `{:#x}` writes it,
    /// with the formatter's width, fill, alignment and the flags `+` and `0` applied as they are
    /// to `{:#x}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let x = Uint::<2>::from_u64(255);
    /// assert_eq!(format!("{x:?}"), "0xff");
    /// assert_eq!(format!("[{x:>8?}] [{x:<8?}] [{x:08?}]"), "[    0xff] [0xff    ] [0x0000ff]");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::text::debug_hex(f, false, *self)
    }
}
