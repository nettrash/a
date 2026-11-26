//! `GF(2¹²⁸)` as GCM uses it (McGrew and Viega, "The Galois/Counter Mode of Operation", 2005;
//! NIST SP 800-38D, 2007): `GF(2)[x] / (x¹²⁸ + x⁷ + x² + x + 1)`, and GHASH.
//!
//! GCM writes an element as a block of 16 bytes in a bit-reflected order: the first bit of the
//! block, the most significant bit of its first byte, is the coefficient of `x⁰`, and the last bit
//! that of `x¹²⁷`. [`Gf2_128`] holds the polynomial in the natural order instead (bit `i` of a
//! `u128` the coefficient of `xⁱ`, as [`BinaryField`](super::BinaryField) holds its elements), and
//! converts at [`from_bytes`](Gf2_128::from_bytes) and [`to_bytes`](Gf2_128::to_bytes) by
//! reversing the bits. A product is the carry-less product of two 128-bit polynomials by
//! Karatsuba's method, three products of words by the CPU's instruction where it has one
//! ([`Clmul::detect`]), then the reduction by `x¹²⁸ = x⁷ + x² + x + 1`, two folds of the top half.
//! A square spreads the bits, a zero after each, and reduces the same way, with no products. The
//! inverse is Itoh and Tsujii's `a⁻¹ = (a^(2¹²⁷ − 1))²`, 12 products and 127 squares.
//!
//! Not constant-time in its use of branches on zero; not an implementation of GCM, only of its
//! field and of the GHASH function.

use super::binary::Clmul;
use super::binary_clmul as clmul;
#[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
use super::gf2_128_pmull as pmull;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// An element of `GF(2¹²⁸)` with GCM's polynomial `x¹²⁸ + x⁷ + x² + x + 1`.
///
/// Bytes are in GCM's block order ([`from_bytes`](Gf2_128::from_bytes)); the polynomial, bit `i`
/// the coefficient of `xⁱ`, is [`from_polynomial`](Gf2_128::from_polynomial). `+` and `-` are the
/// exclusive or, `*` and `/` the field's product and quotient. Not constant-time.
///
/// # Examples
///
/// ```
/// use a::math::field::Gf2_128;
///
/// // GCM's block 80 00 … 00 is the polynomial 1.
/// let mut one = [0u8; 16];
/// one[0] = 0x80;
/// assert_eq!(Gf2_128::from_bytes(one), Gf2_128::ONE);
/// // x¹²⁷ · x = x¹²⁸ = x⁷ + x² + x + 1.
/// let x = Gf2_128::from_polynomial(2);
/// assert_eq!(Gf2_128::from_polynomial(1 << 127) * x, Gf2_128::from_polynomial(0x87));
/// assert_eq!(x * x.inverse().unwrap(), Gf2_128::ONE);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Gf2_128(u128);

impl Gf2_128 {
    /// 0.
    pub const ZERO: Gf2_128 = Gf2_128(0);
    /// 1.
    pub const ONE: Gf2_128 = Gf2_128(1);

    /// The element of a GCM block: the most significant bit of `b[0]` is the coefficient of `x⁰`,
    /// the least significant bit of `b[15]` that of `x¹²⁷`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// let mut b = [0u8; 16];
    /// b[0] = 0x40; // the second bit: x
    /// assert_eq!(Gf2_128::from_bytes(b).polynomial(), 2);
    /// assert_eq!(Gf2_128::from_bytes(b).to_bytes(), b);
    /// ```
    #[must_use]
    pub const fn from_bytes(b: [u8; 16]) -> Gf2_128 {
        Gf2_128(u128::from_be_bytes(b).reverse_bits())
    }

    /// The GCM block of the element, the inverse of [`from_bytes`](Gf2_128::from_bytes).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// // SP 800-38D's R = 11100001 ‖ 0¹²⁰ is x⁷ + x² + x + 1.
    /// let r = Gf2_128::from_polynomial(0x87).to_bytes();
    /// assert_eq!(r[0], 0xe1);
    /// assert!(r[1..].iter().all(|&b| b == 0));
    /// ```
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0.reverse_bits().to_be_bytes()
    }

    /// The element of the polynomial whose coefficient of `xⁱ` is bit `i` of `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// assert_eq!(Gf2_128::from_polynomial(1), Gf2_128::ONE);
    /// ```
    #[must_use]
    pub const fn from_polynomial(p: u128) -> Gf2_128 {
        Gf2_128(p)
    }

    /// The polynomial: bit `i` is the coefficient of `xⁱ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// assert_eq!(Gf2_128::ONE.polynomial(), 1);
    /// ```
    #[must_use]
    pub const fn polynomial(self) -> u128 {
        self.0
    }

    /// Whether it is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// assert!(Gf2_128::ZERO.is_zero() && !Gf2_128::ONE.is_zero());
    /// ```
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The product by the carry-less multiplier `kind`: `*` is this with [`Clmul::detect`]; the
    /// others give the same product, for comparisons.
    ///
    /// # Panics
    ///
    /// Panics if the CPU cannot run `kind`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Clmul, Gf2_128};
    ///
    /// let a = Gf2_128::from_polynomial(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
    /// let b = Gf2_128::from_polynomial(u128::MAX);
    /// assert_eq!(a.mul_with(b, Clmul::Portable), a * b);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn mul_with(self, rhs: Gf2_128, kind: Clmul) -> Gf2_128 {
        let a = [self.0 as u64, (self.0 >> 64) as u64];
        let b = [rhs.0 as u64, (rhs.0 >> 64) as u64];
        let mut p = [0u64; 4];
        let mut s = [0u64; 4];
        clmul::mul(kind, &a, &b, &mut p, &mut s);
        reduce(p)
    }

    /// The square, by spreading the bits (`(Σ aᵢxⁱ)² = Σ aᵢx²ⁱ`) and reducing: no products.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// let x = Gf2_128::from_polynomial(1 << 64);
    /// assert_eq!(x.square(), Gf2_128::from_polynomial(0x87)); // x¹²⁸
    /// ```
    #[must_use]
    pub fn square(self) -> Gf2_128 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
        return Gf2_128(pmull::sqr(self.0));
        #[allow(unreachable_code)]
        let mut p = [0u64; 4];
        clmul::square(&[self.0 as u64, (self.0 >> 64) as u64], &mut p);
        reduce(p)
    }

    /// The `e`-th power; `0⁰ = 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// let a = Gf2_128::from_polynomial(0xdead_beef);
    /// assert_eq!(a.pow(u128::MAX), Gf2_128::ONE); // the group has 2¹²⁸ − 1 elements
    /// assert_eq!(a.pow(3), a * a * a);
    /// ```
    #[must_use]
    pub fn pow(self, e: u128) -> Gf2_128 {
        let mut r = Gf2_128::ONE;
        for i in (0..u128::BITS - e.leading_zeros()).rev() {
            r = r.square();
            if e >> i & 1 == 1 {
                r *= self;
            }
        }
        r
    }

    /// The inverse by Itoh and Tsujii's method, `(a^(2¹²⁷ − 1))²`, 12 products and 127 squares, or
    /// `None` for 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// let a = Gf2_128::from_polynomial(0x1234_5678_9abc_def0);
    /// assert_eq!(a * a.inverse().unwrap(), Gf2_128::ONE);
    /// assert_eq!(Gf2_128::ZERO.inverse(), None);
    /// ```
    #[must_use]
    pub fn inverse(self) -> Option<Gf2_128> {
        if self.0 == 0 {
            return None;
        }
        // b = a^(2ᵏ − 1) for k the bits of 127 read from the top: b ← b^(2ᵏ)·b doubles k, and
        // b ← b²·a adds one.
        let mut b = self;
        let mut k = 1u32;
        for bit in (0..6).rev() {
            let mut t = b;
            for _ in 0..k {
                t = t.square();
            }
            b = t * b;
            k *= 2;
            if 127u32 >> bit & 1 == 1 {
                b = b.square() * self;
                k += 1;
            }
        }
        debug_assert_eq!(k, 127);
        Some(b.square())
    }

    /// GHASH of NIST SP 800-38D (§6.4), `GHASH_H(X)` for `X` a whole number of blocks:
    /// `Y₀ = 0`, `Yᵢ = (Yᵢ₋₁ + Xᵢ)·H`.
    ///
    /// # Panics
    ///
    /// Panics if the length of `x` is not a multiple of 16.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// let h = Gf2_128::from_polynomial(0xfeed);
    /// let x = [7u8; 32];
    /// let block = Gf2_128::from_bytes([7u8; 16]);
    /// assert_eq!(Gf2_128::ghash(h, &x), (block * h + block) * h);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn ghash(h: Gf2_128, x: &[u8]) -> Gf2_128 {
        assert!(
            x.len().is_multiple_of(16),
            "GHASH takes whole blocks of 16 bytes"
        );
        absorb(Gf2_128::ZERO, h, x)
    }

    /// GCM's `GHASH(H, A, C)` (McGrew and Viega, §2.3), over the additional authenticated data
    /// `a` and the ciphertext `c`, each padded with zeros to a whole number of blocks, then the
    /// block of their lengths in bits, `len(A) ‖ len(C)`, two 64-bit big-endian integers. GCM's tag
    /// is this plus the encryption of the first counter block.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::Gf2_128;
    ///
    /// // McGrew and Viega, test case 2: K = 0¹²⁸, IV = 0⁹⁶, P = 0¹²⁸.
    /// let hex = |s: &str| -> [u8; 16] {
    ///     core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
    /// };
    /// let h = Gf2_128::from_bytes(hex("66e94bd4ef8a2c3b884cfa59ca342b2e"));
    /// let c = hex("0388dace60b6a392f328c2b971b2fe78");
    /// let g = Gf2_128::gcm_ghash(h, &[], &c);
    /// assert_eq!(g.to_bytes(), hex("f38cbb1ad69223dcc3457ae5b6b0f885"));
    /// ```
    #[must_use]
    pub fn gcm_ghash(h: Gf2_128, a: &[u8], c: &[u8]) -> Gf2_128 {
        let y = absorb(Gf2_128::ZERO, h, a);
        let y = absorb(y, h, c);
        let mut lengths = [0u8; 16];
        lengths[..8].copy_from_slice(&((a.len() as u64) * 8).to_be_bytes());
        lengths[8..].copy_from_slice(&((c.len() as u64) * 8).to_be_bytes());
        (y + Gf2_128::from_bytes(lengths)) * h
    }
}

/// The 256-bit polynomial `p` (low word first) modulo `x¹²⁸ + x⁷ + x² + x + 1`: the top half times
/// `x⁷ + x² + x + 1`, whose 7 bits above `x¹²⁷` fold once more.
fn reduce(p: [u64; 4]) -> Gf2_128 {
    let lo = u128::from(p[0]) | (u128::from(p[1]) << 64);
    let hi = u128::from(p[2]) | (u128::from(p[3]) << 64);
    let over = (hi >> 127) ^ (hi >> 126) ^ (hi >> 121);
    let fold = |v: u128| v ^ (v << 1) ^ (v << 2) ^ (v << 7);
    Gf2_128(lo ^ fold(hi) ^ fold(over))
}

/// `y` after GHASH's steps over `x`, its last block padded with zeros.
fn absorb(mut y: Gf2_128, h: Gf2_128, x: &[u8]) -> Gf2_128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
    {
        let whole = x.len() & !15;
        y = Gf2_128(pmull::ghash(y.0, h.0, &x[..whole]));
        if whole == x.len() {
            return y;
        }
        let mut block = [0u8; 16];
        block[..x.len() - whole].copy_from_slice(&x[whole..]);
        return (y + Gf2_128::from_bytes(block)) * h;
    }
    #[allow(unreachable_code)]
    for chunk in x.chunks(16) {
        let mut block = [0u8; 16];
        block[..chunk.len()].copy_from_slice(chunk);
        y = (y + Gf2_128::from_bytes(block)) * h;
    }
    y
}

impl fmt::Debug for Gf2_128 {
    /// The GCM block, in hexadecimal.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf2_128(")?;
        for b in self.to_bytes() {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl Neg for Gf2_128 {
    type Output = Gf2_128;
    fn neg(self) -> Gf2_128 {
        self
    }
}

/// `a + b`, which is `a − b`: the coefficients' sums modulo 2.
fn sum(a: Gf2_128, b: Gf2_128) -> Gf2_128 {
    Gf2_128(a.0 ^ b.0)
}

impl Add for Gf2_128 {
    type Output = Gf2_128;
    fn add(self, rhs: Gf2_128) -> Gf2_128 {
        sum(self, rhs)
    }
}

impl Sub for Gf2_128 {
    type Output = Gf2_128;
    fn sub(self, rhs: Gf2_128) -> Gf2_128 {
        sum(self, rhs)
    }
}

impl Mul for Gf2_128 {
    type Output = Gf2_128;
    #[inline]
    fn mul(self, rhs: Gf2_128) -> Gf2_128 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
        return Gf2_128(pmull::mul(self.0, rhs.0));
        #[allow(unreachable_code)]
        self.mul_with(rhs, Clmul::detect())
    }
}

impl Div for Gf2_128 {
    type Output = Gf2_128;
    /// # Panics
    ///
    /// Panics if `rhs` is 0.
    #[track_caller]
    fn div(self, rhs: Gf2_128) -> Gf2_128 {
        let inv = rhs.inverse().expect("division by zero");
        Mul::mul(self, inv)
    }
}

/// The assignments of the operators.
macro_rules! assign {
    ($tr:ident, $m:ident, $op:tt) => {
        impl $tr for Gf2_128 {
            #[track_caller]
            fn $m(&mut self, rhs: Gf2_128) {
                *self = *self $op rhs;
            }
        }
    };
}

assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);
