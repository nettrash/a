//! `GF(2⁸)` as AES uses it (FIPS 197, §4): `GF(2)[x] / (x⁸ + x⁴ + x³ + x + 1)`, an element a byte,
//! bit `i` the coefficient of `xⁱ`.
//!
//! Products, inverses and powers go through tables of logarithms and powers of the generator
//! `x + 1` (`{03}`), built at compile time: a product is two lookups of the logarithms and one of
//! the power of their sum. The AES S-box is the inverse followed by the affine map of FIPS 197,
//! §5.1.1. Not constant-time: the lookups depend on the values.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// `x·a` modulo AES's polynomial (FIPS 197, §4.2.1, `xtime`).
const fn xtime(a: u8) -> u8 {
    (a << 1) ^ if a & 0x80 != 0 { 0x1b } else { 0 }
}

/// The powers of `{03}`, twice over so that a sum of two logarithms needs no reduction, and the
/// logarithms to that base.
const TABLES: ([u8; 512], [u8; 256]) = {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut x: u8 = 1;
    let mut i = 0;
    while i < 255 {
        exp[i] = x;
        exp[i + 255] = x;
        log[x as usize] = i as u8;
        x ^= xtime(x);
        i += 1;
    }
    (exp, log)
};

/// The powers of `{03}`.
const EXP: [u8; 512] = TABLES.0;
/// The logarithms to the base `{03}`; that of 0 is unused.
const LOG: [u8; 256] = TABLES.1;

/// An element of `GF(2⁸)` with AES's polynomial `x⁸ + x⁴ + x³ + x + 1`, held as a byte in the
/// convention of FIPS 197: bit `i` the coefficient of `xⁱ`, so `{57}` is `x⁶ + x⁴ + x² + x + 1`.
///
/// `+` and `-` are the exclusive or; `*`, `/`, [`inverse`](Gf2_8::inverse) and
/// [`pow`](Gf2_8::pow) are by tables of logarithms. Not constant-time.
///
/// # Examples
///
/// ```
/// use a::math::field::Gf2_8;
///
/// let a = Gf2_8::new(0x57);
/// assert_eq!(a + Gf2_8::new(0x83), Gf2_8::new(0xd4)); // FIPS 197, 4.1
/// assert_eq!(a * Gf2_8::new(0x83), Gf2_8::new(0xc1)); // FIPS 197, 4.2
/// assert_eq!(a * a.inverse().unwrap(), Gf2_8::ONE);
/// assert_eq!(Gf2_8::new(0x53).aes_sbox(), Gf2_8::new(0xed)); // FIPS 197, 5.1.1
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Gf2_8(u8);

impl Gf2_8 {
    /// 0.
    pub const ZERO: Gf2_8 = Gf2_8(0);
    /// 1, `{01}`.
    pub const ONE: Gf2_8 = Gf2_8(1);

    /// The element of the byte `b`, bit `i` the coefficient of `xⁱ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// assert_eq!(Gf2_8::new(0x1b).to_byte(), 0x1b);
    /// ```
    #[must_use]
    pub const fn new(b: u8) -> Gf2_8 {
        Gf2_8(b)
    }

    /// The byte, bit `i` the coefficient of `xⁱ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// assert_eq!(Gf2_8::ONE.to_byte(), 1);
    /// ```
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        self.0
    }

    /// `x·a`, FIPS 197's `xtime`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// // FIPS 197, 4.2.1: xtime({57}) = {ae}, xtime({ae}) = {47}.
    /// assert_eq!(Gf2_8::new(0x57).xtime(), Gf2_8::new(0xae));
    /// assert_eq!(Gf2_8::new(0xae).xtime(), Gf2_8::new(0x47));
    /// ```
    #[must_use]
    pub const fn xtime(self) -> Gf2_8 {
        Gf2_8(xtime(self.0))
    }

    /// The inverse, `None` for 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// assert_eq!(Gf2_8::new(0x53).inverse(), Some(Gf2_8::new(0xca)));
    /// assert_eq!(Gf2_8::ZERO.inverse(), None);
    /// ```
    #[must_use]
    pub fn inverse(self) -> Option<Gf2_8> {
        (self.0 != 0).then(|| Gf2_8(EXP[255 - LOG[self.0 as usize] as usize]))
    }

    /// The `e`-th power; `0⁰ = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// let g = Gf2_8::new(3);
    /// assert_eq!(g.pow(255), Gf2_8::ONE);
    /// assert_eq!(g.pow(2), g * g);
    /// assert_eq!(Gf2_8::ZERO.pow(0), Gf2_8::ONE);
    /// ```
    #[must_use]
    pub fn pow(self, e: u64) -> Gf2_8 {
        if self.0 == 0 {
            return if e == 0 { Gf2_8::ONE } else { Gf2_8::ZERO };
        }
        let l = u64::from(LOG[self.0 as usize]) * (e % 255) % 255;
        Gf2_8(EXP[l as usize])
    }

    /// The AES S-box (FIPS 197, §5.1.1): the inverse (0 for 0), then the affine map
    /// `bᵢ' = bᵢ + b_{i+4} + b_{i+5} + b_{i+6} + b_{i+7} + cᵢ` (indices mod 8, `c = {63}`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// assert_eq!(Gf2_8::ZERO.aes_sbox(), Gf2_8::new(0x63));
    /// assert_eq!(Gf2_8::new(0x53).aes_sbox(), Gf2_8::new(0xed));
    /// ```
    #[must_use]
    pub fn aes_sbox(self) -> Gf2_8 {
        let b = self.inverse().unwrap_or(Gf2_8::ZERO).0;
        Gf2_8(b ^ b.rotate_left(1) ^ b.rotate_left(2) ^ b.rotate_left(3) ^ b.rotate_left(4) ^ 0x63)
    }

    /// The inverse of the AES S-box (FIPS 197, §5.3.2): the inverse of the affine map, then the
    /// inverse in the field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_8;
    ///
    /// assert_eq!(Gf2_8::new(0xed).aes_inverse_sbox(), Gf2_8::new(0x53));
    /// for b in 0..=255u8 {
    ///     assert_eq!(Gf2_8::new(b).aes_sbox().aes_inverse_sbox(), Gf2_8::new(b));
    /// }
    /// ```
    #[must_use]
    pub fn aes_inverse_sbox(self) -> Gf2_8 {
        let s = self.0;
        let b = s.rotate_left(1) ^ s.rotate_left(3) ^ s.rotate_left(6) ^ 0x05;
        Gf2_8(b).inverse().unwrap_or(Gf2_8::ZERO)
    }
}

impl From<u8> for Gf2_8 {
    fn from(b: u8) -> Gf2_8 {
        Gf2_8(b)
    }
}

impl From<Gf2_8> for u8 {
    fn from(a: Gf2_8) -> u8 {
        a.0
    }
}

impl fmt::Debug for Gf2_8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2_8({{{:02x}}})", self.0)
    }
}

impl Neg for Gf2_8 {
    type Output = Gf2_8;
    fn neg(self) -> Gf2_8 {
        self
    }
}

/// `a + b`, which is `a − b`: the coefficients' sums modulo 2.
fn sum(a: Gf2_8, b: Gf2_8) -> Gf2_8 {
    Gf2_8(a.0 ^ b.0)
}

impl Add for Gf2_8 {
    type Output = Gf2_8;
    fn add(self, rhs: Gf2_8) -> Gf2_8 {
        sum(self, rhs)
    }
}

impl Sub for Gf2_8 {
    type Output = Gf2_8;
    fn sub(self, rhs: Gf2_8) -> Gf2_8 {
        sum(self, rhs)
    }
}

/// `a·b` by the tables: the power of the sum of the logarithms.
fn product(a: Gf2_8, b: Gf2_8) -> Gf2_8 {
    if a.0 == 0 || b.0 == 0 {
        return Gf2_8::ZERO;
    }
    Gf2_8(EXP[LOG[a.0 as usize] as usize + LOG[b.0 as usize] as usize])
}

impl Mul for Gf2_8 {
    type Output = Gf2_8;
    fn mul(self, rhs: Gf2_8) -> Gf2_8 {
        product(self, rhs)
    }
}

impl Div for Gf2_8 {
    type Output = Gf2_8;
    /// # Panics
    ///
    /// Panics if `rhs` is 0.
    #[track_caller]
    fn div(self, rhs: Gf2_8) -> Gf2_8 {
        product(self, rhs.inverse().expect("division by zero"))
    }
}

/// The assignments of the operators.
macro_rules! assign {
    ($tr:ident, $m:ident, $op:tt) => {
        impl $tr for Gf2_8 {
            #[track_caller]
            fn $m(&mut self, rhs: Gf2_8) {
                *self = *self $op rhs;
            }
        }
    };
}

assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);
