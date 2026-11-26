//! Prime fields `GF(p)` and their elements, in the arithmetic of the curves of
//! [`a::math::ec`](crate::math::ec).
//!
//! [`PrimeField::new`] chooses the form of the arithmetic once, from the prime, as a curve does
//! (see `prime_form`): Montgomery's form for any odd prime of one to nine limbs, with the
//! products of rows below seven limbs and of the multiplication's kernels from seven; the primes
//! `2^k - c` with a small `c` of four, six and nine limbs (secp256k1's, Curve25519's), whose
//! products fold the high half back by a product by `c`; the NIST primes of P-256 and P-384,
//! Montgomery's form with their own products on aarch64; P-521's `2^521 - 1` in nine limbs of 58
//! bits; Montgomery's form of any length beyond nine limbs, in a vector of limbs; and bits for
//! `GF(2)`. Each operation matches the form once; the assignments write only the form's limbs.
//!
//! An element holds its limbs inline up to nine limbs, so that sums, differences, products,
//! negation, squares and doubling, with their assignments, comparisons, hashes, `zero`, `one`,
//! `from_u64` and `from_bytes_be`/`_le` allocate nothing. What does allocate, in a field of up to
//! nine limbs: an inversion (`inverse`, `/`, `/=`) twice, the value as a `Number` and the inverse
//! [`Number::mod_inverse`] returns; the Legendre symbol (`legendre`, `is_square`) once, the value,
//! and beyond six limbs [`Number::jacobi`] may take working memory too; a reduced decoding
//! (`from_bytes_*_reduced`) three times, the bytes as a number, its remainder and the remainder's
//! limbs; powers and square roots their tables of powers; `element`, `value`, `to_bytes_*`,
//! `batch_invert` and the formatting their numbers and vectors. `GF(2)` inverts and takes symbols
//! without allocating; the elements of longer fields are vectors of limbs, which the assignments
//! reuse and the other operations allocate.
//!
//! Inversion is by Lehmer's extended Euclid of [`Number::mod_inverse`] (see `prime_inv`), the
//! Legendre symbol by the binary method and Lehmer's steps of [`Number::jacobi`], both on the
//! value, and square
//! roots by the method that fits `p` (see `prime_sqrt`): one power for `p = 3 (mod 4)`, Atkin's
//! for `p = 5 (mod 8)`, Tonelli and Shanks's otherwise, and Cipolla and Lehmer's when `p - 1` has
//! many factors of two.
//!
//! Like the rest of the library this is not constant-time: the time of an inversion, a square
//! root, a power or a comparison depends on the values.

use super::prime_form::{
    Arith, Canon, Form, Value, canonical, canonical_value, legendre, with_form, with_limbs,
};
use super::prime_sqrt::{Plan, sqrt};
use crate::math::ec::field::{Field, Prime, limbs_of, number_of};
use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::OnceLock;

/// The prime field `GF(p)`: the integers modulo a prime `p`, its arithmetic chosen once, when it is
/// built, for the shape and the length of `p`.
///
/// # Examples
///
/// ```
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// let f = PrimeField::new(&Number::from(13u8));
/// assert_eq!(f.modulus(), &Number::from(13u8));
/// assert_eq!((f.from_u64(9) + f.from_u64(8)).value(), Number::from(4u8));
/// ```
#[derive(Clone)]
pub struct PrimeField {
    /// The prime.
    p: Number,
    /// The arithmetic.
    form: Form,
    /// The limbs of `p`, the low one first.
    pub(super) limbs: Vec<u64>,
    /// The limbs of `(p - 1) / 2`, as many as `p`'s: the larger of two roots is above it.
    half: Vec<u64>,
    /// What square roots need, computed with the field: at most one power, against the several
    /// of the test of primality.
    sqrt: Plan,
}

impl fmt::Debug for PrimeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrimeField")
            .field("p", &format_args!("{}", self.p))
            .field("form", &self.form.name())
            .finish()
    }
}

impl PrimeField {
    /// The field of the integers modulo `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a prime (by [`Number::is_probable_prime`], which no composite is known
    /// to pass).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(1_000_000_007u64));
    /// assert_eq!((f.from_u64(1_000_000_006) * f.from_u64(2)).value(), Number::from(1_000_000_005u64));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(p: &Number) -> PrimeField {
        assert!(*p >= 2u8 && p.is_probable_prime(), "{p} is not a prime");
        let limbs = limbs_of(p);
        let mut half = limbs_of(&((p - &Number::from(1u8)) >> 1usize));
        half.resize(limbs.len(), 0);
        let form = Form::new(p);
        let sqrt = with_form!(&form, f => Plan::new(f, p));
        PrimeField {
            p: p.clone(),
            form,
            limbs,
            half,
            sqrt,
        }
    }

    /// The prime `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!(f.modulus(), &Number::from(13u8));
    /// ```
    #[must_use]
    pub fn modulus(&self) -> &Number {
        &self.p
    }

    /// The characteristic of the field, `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(7u8));
    /// assert_eq!(f.characteristic(), &Number::from(7u8));
    /// ```
    #[must_use]
    pub fn characteristic(&self) -> &Number {
        &self.p
    }

    /// The number of elements of the field, `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(7u8));
    /// assert_eq!(f.order(), &Number::from(7u8));
    /// ```
    #[must_use]
    pub fn order(&self) -> &Number {
        &self.p
    }

    /// The number of bits of `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// assert_eq!(PrimeField::bn254_base().bits(), 254);
    /// assert_eq!(PrimeField::new(&Number::from(2u8)).bits(), 2);
    /// ```
    #[must_use]
    pub fn bits(&self) -> u64 {
        self.p.bits()
    }

    /// The number of bytes of an element's encoding, the bytes of `p`: `bits().div_ceil(8)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::bls12_381_base().byte_len(), 48);
    /// assert_eq!(PrimeField::p521_base().byte_len(), 66);
    /// ```
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.p.bits().div_ceil(8) as usize
    }

    /// The element congruent to `x`, of any sign and size.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!(f.element(&Number::from(-1i8)).value(), Number::from(12u8));
    /// assert_eq!(f.element(&(Number::from(1u8) << 100usize)), f.from_u64(3)); // 2^100 = 3 (mod 13)
    /// ```
    #[must_use]
    pub fn element(&self, x: &Number) -> Fp<'_> {
        let v = with_form!(&self.form, f => f.put_number(x));
        Fp { f: self, v }
    }

    /// The element congruent to `x`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!(f.from_u64(100).value(), Number::from(9u8));
    /// ```
    #[must_use]
    pub fn from_u64(&self, x: u64) -> Fp<'_> {
        let mut c = Canon::zeros(self.limbs.len());
        // A prime of two limbs or more is above every word.
        c.limbs_mut()[0] = if self.limbs.len() == 1 {
            x % self.limbs[0]
        } else {
            x
        };
        self.of_limbs(&c)
    }

    /// 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::p256_base();
    /// assert!(f.zero().is_zero());
    /// assert_eq!(&f.zero() + &f.one(), f.one());
    /// ```
    #[must_use]
    pub fn zero(&self) -> Fp<'_> {
        let v = with_form!(&self.form, f => f.put(f.zero()));
        Fp { f: self, v }
    }

    /// 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::p256_base();
    /// assert!(f.one().is_one());
    /// assert_eq!(f.one().value(), Number::from(1u8));
    /// ```
    #[must_use]
    pub fn one(&self) -> Fp<'_> {
        let v = with_form!(&self.form, f => f.put(f.one()));
        Fp { f: self, v }
    }

    /// The element whose value has the big-endian encoding `bytes`, of exactly
    /// [`byte_len`](Self::byte_len) bytes, or `None` when the length differs or the value is not
    /// below `p` (SEC 1's and ZCash's field encodings).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(65_521u32));
    /// assert_eq!(f.from_bytes_be(&[0x01, 0x02]).unwrap().value(), Number::from(258u16));
    /// assert!(f.from_bytes_be(&[0xff, 0xf1]).is_none()); // p itself
    /// assert!(f.from_bytes_be(&[0x01]).is_none()); // too short
    /// ```
    #[must_use]
    pub fn from_bytes_be(&self, bytes: &[u8]) -> Option<Fp<'_>> {
        self.decode(bytes, ByteOrder::BigEndian)
    }

    /// The element whose value has the little-endian encoding `bytes`, of exactly
    /// [`byte_len`](Self::byte_len) bytes, or `None` when the length differs or the value is not
    /// below `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(65_521u32));
    /// assert_eq!(f.from_bytes_le(&[0x02, 0x01]).unwrap().value(), Number::from(258u16));
    /// assert!(f.from_bytes_le(&[0xf1, 0xff]).is_none());
    /// ```
    #[must_use]
    pub fn from_bytes_le(&self, bytes: &[u8]) -> Option<Fp<'_>> {
        self.decode(bytes, ByteOrder::LittleEndian)
    }

    /// The element congruent to the number with the big-endian bytes `bytes`, of any length: an
    /// empty slice is 0. RFC 9380's `OS2IP(tv) mod p` of `hash_to_field`. It reduces a `Number`:
    /// three allocations for a field of up to nine limbs, none for an empty slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(251u8));
    /// assert_eq!(f.from_bytes_be_reduced(&[1, 0]).value(), Number::from(5u8)); // 256 - 251
    /// ```
    #[must_use]
    pub fn from_bytes_be_reduced(&self, bytes: &[u8]) -> Fp<'_> {
        self.reduce_bytes(bytes, ByteOrder::BigEndian)
    }

    /// The element congruent to the number with the little-endian bytes `bytes`, of any length:
    /// an empty slice is 0. It reduces a `Number`: three allocations for a field of up to nine
    /// limbs, none for an empty slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(251u8));
    /// assert_eq!(f.from_bytes_le_reduced(&[0, 1]).value(), Number::from(5u8));
    /// ```
    #[must_use]
    pub fn from_bytes_le_reduced(&self, bytes: &[u8]) -> Fp<'_> {
        self.reduce_bytes(bytes, ByteOrder::LittleEndian)
    }

    /// The inverses of `xs`, by Montgomery's trick: one inversion and three products for each
    /// nonzero element. A zero stays zero, as it does in arkworks' `batch_inversion`.
    ///
    /// # Panics
    ///
    /// Panics if an element is not of this field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(17u8));
    /// let xs = [f.from_u64(2), f.zero(), f.from_u64(5)];
    /// let inv = f.batch_invert(&xs);
    /// assert_eq!(&inv[0] * &xs[0], f.one());
    /// assert!(inv[1].is_zero());
    /// assert_eq!(&inv[2] * &xs[2], f.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn batch_invert<'f>(&'f self, xs: &[Fp<'f>]) -> Vec<Fp<'f>> {
        for x in xs {
            self.check(x);
        }
        let values = with_form!(&self.form, f => invert_all(f, xs));
        values.into_iter().map(|v| Fp { f: self, v }).collect()
    }

    /// The base field of BN254 (alt_bn128), the prime
    /// `21888242871839275222246405745257275088696311157297823662689037894645226208583`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bn254_base();
    /// assert_eq!(f.bits(), 254);
    /// assert_eq!(f.form(), "Montgomery, 4 limbs");
    /// ```
    #[must_use]
    pub fn bn254_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named("21888242871839275222246405745257275088696311157297823662689037894645226208583")
        })
    }

    /// The scalar field of BN254, the order of its groups,
    /// `21888242871839275222246405745257275088548364400416034343698204186575808495617`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::bn254_scalar();
    /// assert_eq!((f.modulus() - &Number::from(1u8)).trailing_zeros(), Some(28));
    /// assert_eq!(f.sqrt_method(), "Tonelli-Shanks");
    /// ```
    #[must_use]
    pub fn bn254_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named("21888242871839275222246405745257275088548364400416034343698204186575808495617")
        })
    }

    /// The base field of BLS12-381, the prime `0x1a0111ea…ffffaaab` of 381 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bls12_381_base();
    /// assert_eq!(f.bits(), 381);
    /// assert_eq!(f.form(), "Montgomery, 6 limbs");
    /// ```
    #[must_use]
    pub fn bls12_381_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex(
                "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
            )
        })
    }

    /// The scalar field of BLS12-381, the order of its groups, `0x73eda753…00000001` of 255 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::bls12_381_scalar();
    /// assert_eq!(f.bits(), 255);
    /// assert_eq!((f.modulus() - &Number::from(1u8)).trailing_zeros(), Some(32));
    /// ```
    #[must_use]
    pub fn bls12_381_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001")
        })
    }

    /// The base field of secp256k1 (SEC 2), the prime `2^256 - 2^32 - 977`, in the form of the
    /// primes `2^k - c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::secp256k1_base();
    /// let p = (Number::from(1u8) << 256usize) - (Number::from(1u8) << 32usize) - Number::from(977u16);
    /// assert_eq!(f.modulus(), &p);
    /// ```
    #[must_use]
    pub fn secp256k1_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f")
        })
    }

    /// The scalar field of secp256k1, the order of its group,
    /// `0xffffffff…bfd25e8cd0364141` of 256 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::named;
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::secp256k1_scalar().modulus(), named::secp256k1().order());
    /// ```
    #[must_use]
    pub fn secp256k1_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141")
        })
    }

    /// The base field of P-256 (FIPS 186), the prime `2^256 - 2^224 + 2^192 + 2^96 - 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::{Curve, named};
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::p256_base();
    /// assert_eq!(f.modulus(), named::p256().modulus());
    /// assert_eq!(f.form(), "P-256");
    /// ```
    #[must_use]
    pub fn p256_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex("ffffffff00000001000000000000000000000000ffffffffffffffffffffffff")
        })
    }

    /// The scalar field of P-256, the order of its group, `0xffffffff00000000…fc632551`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::named;
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::p256_scalar().modulus(), named::p256().order());
    /// ```
    #[must_use]
    pub fn p256_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551")
        })
    }

    /// The base field of P-384 (FIPS 186), the prime `2^384 - 2^128 - 2^96 + 2^32 - 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::{Curve, named};
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::p384_base();
    /// assert_eq!(f.modulus(), named::p384().modulus());
    /// assert_eq!(f.form(), "P-384");
    /// ```
    #[must_use]
    pub fn p384_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex(
                "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff",
            )
        })
    }

    /// The scalar field of P-384, the order of its group, `0xffffffff…ccc52973`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::named;
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::p384_scalar().modulus(), named::p384().order());
    /// ```
    #[must_use]
    pub fn p384_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex(
                "ffffffffffffffffffffffffffffffffffffffffffffffffc7634d81f4372ddf581a0db248b0a77aecec196accc52973",
            )
        })
    }

    /// The base field of P-521 (FIPS 186), the prime `2^521 - 1`, in limbs of 58 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::p521_base();
    /// assert_eq!(f.modulus(), &((Number::from(1u8) << 521usize) - Number::from(1u8)));
    /// assert_eq!(f.form(), "P-521, limbs of 58 bits");
    /// ```
    #[must_use]
    pub fn p521_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| PrimeField::new(&((Number::from(1u8) << 521usize) - Number::from(1u8))))
    }

    /// The scalar field of P-521, the order of its group,
    /// `0x1ff…fa51868783bf2f966b7fcc0148f709a5d03bb5c9b8899c47aebb6fb71e91386409` of 521 bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::named;
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::p521_scalar().modulus(), named::p521().order());
    /// assert_eq!(PrimeField::p521_scalar().form(), "Montgomery, 9 limbs");
    /// ```
    #[must_use]
    pub fn p521_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex(
                "1fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa51868783bf2f966b7fcc0148f709a5d03bb5c9b8899c47aebb6fb71e91386409",
            )
        })
    }

    /// The base field of Curve25519 and Ed25519 (RFC 7748, RFC 8032), the prime `2^255 - 19`, in
    /// the form of the primes `2^k - c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::curve25519_base();
    /// assert_eq!(f.modulus(), &((Number::from(1u8) << 255usize) - Number::from(19u8)));
    /// assert_eq!(f.form(), "2^k - c, 4 limbs");
    /// ```
    #[must_use]
    pub fn curve25519_base() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| PrimeField::new(&((Number::from(1u8) << 255usize) - Number::from(19u8))))
    }

    /// The scalar field of Curve25519 and Ed25519, the order of their prime subgroup,
    /// `2^252 + 27742317777372353535851937790883648493`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::named;
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::curve25519_scalar().modulus(), named::ed25519().order());
    /// ```
    #[must_use]
    pub fn curve25519_scalar() -> &'static PrimeField {
        static F: OnceLock<PrimeField> = OnceLock::new();
        F.get_or_init(|| {
            named_hex("1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed")
        })
    }

    /// The name of the method of this field's square roots: `"p = 3 mod 4"`, `"Atkin"`,
    /// `"Tonelli-Shanks"`, `"Cipolla"`, or `"GF(2)"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::bls12_381_base().sqrt_method(), "p = 3 mod 4");
    /// assert_eq!(PrimeField::curve25519_base().sqrt_method(), "Atkin");
    /// assert_eq!(PrimeField::bls12_381_scalar().sqrt_method(), "Tonelli-Shanks"); // 2^32 | p - 1
    /// ```
    #[must_use]
    pub fn sqrt_method(&self) -> &'static str {
        self.sqrt.name()
    }

    /// The name of the form of this field's arithmetic, such as `"Montgomery, 4 limbs"`,
    /// `"2^k - c, 4 limbs"`, `"P-256"`, `"P-521, limbs of 58 bits"` or
    /// `"Montgomery, more than 9 limbs"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// assert_eq!(PrimeField::bn254_base().form(), "Montgomery, 4 limbs");
    /// assert_eq!(PrimeField::secp256k1_base().form(), "2^k - c, 4 limbs");
    /// assert_eq!(PrimeField::p521_base().form(), "P-521, limbs of 58 bits");
    /// ```
    #[must_use]
    pub fn form(&self) -> &'static str {
        self.form.name()
    }

    /// The form of the arithmetic, for the towers.
    #[inline(always)]
    pub(super) fn form_ref(&self) -> &Form {
        &self.form
    }

    /// The element of the number below `p` with the limbs `c`.
    fn of_limbs(&self, c: &Canon) -> Fp<'_> {
        let v = with_form!(&self.form, f => f.put_canon(c.limbs()));
        Fp { f: self, v }
    }

    /// The element whose value has the encoding `bytes` of exactly `byte_len` bytes, if below `p`.
    fn decode(&self, bytes: &[u8], order: ByteOrder) -> Option<Fp<'_>> {
        if bytes.len() != self.byte_len() {
            return None;
        }
        let mut c = Canon::zeros(self.limbs.len());
        let l = c.limbs_mut();
        // Eight bytes a limb, from the least significant end; the top limb may take fewer.
        let n = bytes.len();
        for (j, limb) in l.iter_mut().enumerate() {
            let (lo, hi) = (8 * j, (8 * j + 8).min(n));
            if lo >= hi {
                break;
            }
            let mut w = [0u8; 8];
            *limb = match order {
                ByteOrder::LittleEndian => {
                    w[..hi - lo].copy_from_slice(&bytes[lo..hi]);
                    u64::from_le_bytes(w)
                }
                ByteOrder::BigEndian => {
                    w[8 - (hi - lo)..].copy_from_slice(&bytes[n - hi..n - lo]);
                    u64::from_be_bytes(w)
                }
            };
        }
        (compare(c.limbs(), &self.limbs) == Ordering::Less).then(|| self.of_limbs(&c))
    }

    /// The element congruent to the number with the bytes `bytes` in the order `order`.
    fn reduce_bytes(&self, bytes: &[u8], order: ByteOrder) -> Fp<'_> {
        if bytes.is_empty() {
            return self.zero();
        }
        self.element(&Number::new(bytes.to_vec(), order))
    }

    /// Panics unless `x` is an element of this field: one comparison of pointers on the way of
    /// the operations, the primes compared only for two fields built apart.
    #[track_caller]
    #[inline(always)]
    pub(super) fn check(&self, x: &Fp<'_>) {
        if !core::ptr::eq(self, x.f) {
            self.check_apart(x.f);
        }
    }

    /// Panics unless `other`, a field built apart from this one, has the same prime.
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn check_apart(&self, other: &PrimeField) {
        if self.p != other.p {
            different_fields();
        }
    }

    /// Whether `other` is this field, or one of the same prime.
    #[inline]
    fn same_field(&self, other: &PrimeField) -> bool {
        core::ptr::eq(self, other) || self.p == other.p
    }
}

/// The inverses of `xs` in the form `f`, zeros left as they are, by Montgomery's trick:
/// `prefix[i]`, the product of the nonzero elements before `i`, one inversion of the product of
/// them all, and two products back for each.
fn invert_all<F: Arith>(f: &F, xs: &[Fp<'_>]) -> Vec<Value> {
    let mut prefix = Vec::with_capacity(xs.len());
    let mut acc = f.one();
    for x in xs {
        let e = f.get(&x.v);
        prefix.push(acc.clone());
        if !f.is_zero(e) {
            acc = f.mul(&acc, e);
        }
    }
    let mut inv = f
        .invert_value(&acc)
        .expect("a product of nonzero elements is nonzero");
    let mut out = vec![f.put(f.zero()); xs.len()];
    for i in (0..xs.len()).rev() {
        let e = f.get(&xs[i].v);
        if f.is_zero(e) {
            continue;
        }
        out[i] = f.put(f.mul(&inv, &prefix[i]));
        inv = f.mul(&inv, e);
    }
    out
}

/// The panic of an operation on elements of different fields.
#[cold]
#[inline(never)]
#[track_caller]
fn different_fields() -> ! {
    panic!("elements of different fields")
}

/// The order of the numbers with the limbs `a` and `b`, as many of each, the low limb first.
fn compare(a: &[u64], b: &[u64]) -> Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// A named field from its prime in decimal.
fn named(decimal: &str) -> PrimeField {
    PrimeField::new(
        &Number::from_str_radix(decimal, 10, ByteOrder::LittleEndian).expect("a decimal prime"),
    )
}

/// A named field from its prime in hexadecimal.
fn named_hex(hex: &str) -> PrimeField {
    PrimeField::new(
        &Number::from_str_radix(hex, 16, ByteOrder::LittleEndian).expect("a hexadecimal prime"),
    )
}

/// An element of a [`PrimeField`], which it borrows, holding its limbs inline in the field's form
/// for a prime of up to nine limbs and a [`Number`] beyond. `+`, `-`, `*` and `/` take every
/// combination of owned and borrowed operands, with their assignments, and `-x` negates; two
/// elements are equal when they are of the same prime and have the same value, and hash by their
/// value.
///
/// Up to nine limbs, `+`, `-`, `*`, `-x`, [`square`](Fp::square), [`double`](Fp::double), their
/// assignments, `==`, hashing, [`is_zero`](Fp::is_zero), [`is_one`](Fp::is_one) and
/// [`sgn0`](Fp::sgn0) allocate nothing; [`inverse`](Fp::inverse), `/` and `/=` allocate twice,
/// [`legendre`](Fp::legendre) and [`is_square`](Fp::is_square) once up to six limbs, and
/// [`pow`](Fp::pow), [`pow_u64`](Fp::pow_u64) and [`sqrt`](Fp::sqrt) keep their tables of powers
/// in vectors.
///
/// # Panics
///
/// The operators panic when the operands are of fields of different primes ("elements of
/// different fields"; two fields built from the same prime mix), and `/` and `/=` when the
/// divisor is 0 ("division by zero").
///
/// # Examples
///
/// ```
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// let f = PrimeField::bn254_base();
/// let x = f.from_u64(3);
/// assert_eq!((&x - &x).is_zero(), true);
/// assert_eq!(x.pow(&(f.modulus() - &Number::from(1u8))), f.one()); // Fermat
/// assert_eq!(f.from_u64(4).legendre(), 1);
/// ```
#[derive(Clone)]
pub struct Fp<'f> {
    f: &'f PrimeField,
    /// The element, in the field's form.
    v: Value,
}

impl<'f> Fp<'f> {
    /// The field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// let x = f.from_u64(5);
    /// assert!(core::ptr::eq(x.field(), &f));
    /// ```
    #[must_use]
    pub fn field(&self) -> &'f PrimeField {
        self.f
    }

    /// The value, in `[0, p)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!((f.from_u64(7) * f.from_u64(8)).value(), Number::from(4u8)); // 56 = 4·13 + 4
    /// ```
    #[must_use]
    pub fn value(&self) -> Number {
        number_of(self.canonical().limbs(), ByteOrder::LittleEndian)
    }

    /// Whether it is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::secp256k1_base();
    /// assert!(f.element(f.modulus()).is_zero());
    /// assert!(!f.one().is_zero());
    /// ```
    #[must_use]
    pub fn is_zero(&self) -> bool {
        with_form!(&self.f.form, f => f.is_zero(f.get(&self.v)))
    }

    /// Whether it is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::p521_base();
    /// assert!((f.from_u64(3) / f.from_u64(3)).is_one());
    /// assert!(!f.zero().is_one());
    /// ```
    #[must_use]
    pub fn is_one(&self) -> bool {
        with_form!(&self.f.form, f => f.same(f.get(&self.v), &f.one()))
    }

    /// The square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!(f.from_u64(5).square(), f.from_u64(12)); // 25 = 13 + 12
    /// ```
    #[must_use]
    pub fn square(&self) -> Fp<'f> {
        self.with(with_form!(&self.f.form, f => f.put(f.sqr(f.get(&self.v)))))
    }

    /// Squares the element in place: [`square`](Fp::square) without making a new element, so that
    /// only the form's limbs are written (as `*=` does).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// let mut x = f.from_u64(5);
    /// x.square_in_place();
    /// assert_eq!(x, f.from_u64(12)); // 25 = 13 + 12
    /// ```
    pub fn square_in_place(&mut self) {
        with_form!(&self.f.form, f => f.update(&mut self.v, |x| f.sqr(x)));
    }

    /// Doubles the element in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// let mut x = f.from_u64(9);
    /// x.double_in_place();
    /// assert_eq!(x, f.from_u64(5)); // 18 = 13 + 5
    /// ```
    pub fn double_in_place(&mut self) {
        with_form!(&self.f.form, f => f.update(&mut self.v, |x| f.twice(x)));
    }

    /// Negates the element in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// let mut x = f.from_u64(3);
    /// x.neg_in_place();
    /// assert_eq!(x, f.from_u64(10));
    /// ```
    pub fn neg_in_place(&mut self) {
        with_form!(&self.f.form, f => f.update(&mut self.v, |x| f.neg(x)));
    }

    /// Twice the element.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!(f.from_u64(9).double(), f.from_u64(5));
    /// ```
    #[must_use]
    pub fn double(&self) -> Fp<'f> {
        self.with(with_form!(&self.f.form, f => f.put(f.twice(f.get(&self.v)))))
    }

    /// The `e`-th power; a negative `e` is a power of the inverse. An exponent much longer than
    /// `p` is first reduced modulo `p - 1`, by Fermat's little theorem.
    ///
    /// # Panics
    ///
    /// Panics if `e` is negative and the element is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::bls12_381_base();
    /// let x = f.from_u64(7);
    /// assert_eq!(x.pow(&Number::from(3u8)), f.from_u64(343));
    /// assert_eq!(x.pow(&Number::from(-1i8)), x.inverse().unwrap());
    /// assert_eq!(x.pow(f.modulus()), x); // Fermat
    /// ```
    #[track_caller]
    #[must_use]
    pub fn pow(&self, e: &Number) -> Fp<'f> {
        if e.is_negative() {
            let inv = self.inverse().expect("0 has no inverse");
            return inv.pow(&-e);
        }
        if e.bits() > self.f.bits() + 64 {
            // `x^e = x^(((e - 1) mod (p - 1)) + 1)` for every `x`, 0 included, as `e >= 1`.
            let one = Number::from(1u8);
            return self.pow(&(((e - &one) % (&self.f.p - &one)) + &one));
        }
        let v = with_limbs(
            e,
            |limbs| with_form!(&self.f.form, f => f.put(f.power(f.get(&self.v), limbs))),
        );
        self.with(v)
    }

    /// The `e`-th power.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bls12_381_scalar();
    /// let x = f.from_u64(3);
    /// assert_eq!(x.pow_u64(4), f.from_u64(81));
    /// assert_eq!(x.pow_u64(0), f.one());
    /// ```
    #[must_use]
    pub fn pow_u64(&self, e: u64) -> Fp<'f> {
        self.with(with_form!(&self.f.form, f => f.put(f.power(f.get(&self.v), &[e]))))
    }

    /// The inverse, or `None` for 0, by Lehmer's extended Euclid of [`Number::mod_inverse`] on the
    /// value: two allocations, the value and its inverse, in a field of up to nine limbs (none in
    /// `GF(2)`). `/` and `/=` take it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(13u8));
    /// assert_eq!(f.from_u64(5).inverse(), Some(f.from_u64(8))); // 40 = 3·13 + 1
    /// assert_eq!(f.zero().inverse(), None);
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Option<Fp<'f>> {
        let v = with_form!(&self.f.form, f => f.invert_value(f.get(&self.v)).map(|r| f.put(r)))?;
        Some(self.with(v))
    }

    /// A square root, the smaller of the two in `[0, p)` (0 for 0, and 1 for 1 in `GF(2)`, where it
    /// is the only one), or `None` if the element is not a square. The method is the field's
    /// [`sqrt_method`](PrimeField::sqrt_method).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(41u8)); // 41 = 1 + 5·2^3: Tonelli and Shanks
    /// assert_eq!(f.from_u64(2).sqrt(), Some(f.from_u64(17))); // 17² = 289 = 7·41 + 2; 24 is the other
    /// assert_eq!(f.from_u64(3).sqrt(), None);
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Option<Fp<'f>> {
        let plan = &self.f.sqrt;
        let half = &self.f.half;
        let v = with_form!(&self.f.form, f => {
            let r = sqrt(f, plan, f.get(&self.v))?;
            if compare(canonical(f, &r).limbs(), half) == Ordering::Greater {
                f.put(f.neg(&r))
            } else {
                f.put(r)
            }
        });
        Some(self.with(v))
    }

    /// The Legendre symbol: 1 for a nonzero square, −1 for a non-square, 0 for 0, by
    /// [`Number::jacobi`] on the value: one allocation, the value, in a field of up to six limbs,
    /// and beyond that whatever working memory `Number::jacobi` takes.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(7u8));
    /// let symbols: Vec<i32> = (0..7).map(|x| f.from_u64(x).legendre()).collect();
    /// assert_eq!(symbols, [0, 1, 1, -1, 1, -1, -1]);
    /// ```
    #[must_use]
    pub fn legendre(&self) -> i32 {
        with_form!(&self.f.form, f => {
            let e = f.get(&self.v);
            if matches!(self.f.form, Form::Two(_)) {
                i32::from(!f.is_zero(e))
            } else {
                legendre(f, e)
            }
        })
    }

    /// Whether the element is a square, 0 included: RFC 9380's `is_square`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(7u8));
    /// let squares: Vec<u64> = (0..7).filter(|&x| f.from_u64(x).is_square()).collect();
    /// assert_eq!(squares, [0, 1, 2, 4]);
    /// ```
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.legendre() >= 0
    }

    /// The sign of the element in RFC 9380 (section 4.1) for a prime field: whether its value is
    /// odd. `sgn0(x)` of the RFC is `u8::from(x.sgn0())`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(7u8));
    /// assert!(f.from_u64(3).sgn0());
    /// assert!(!(-f.from_u64(3)).sgn0()); // 4
    /// assert!(!f.zero().sgn0());
    /// ```
    #[must_use]
    pub fn sgn0(&self) -> bool {
        self.canonical().limbs()[0] & 1 == 1
    }

    /// The big-endian encoding of the value, in exactly [`byte_len`](PrimeField::byte_len) bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bn254_base();
    /// let x = f.from_u64(0x0102);
    /// let bytes = x.to_bytes_be();
    /// assert_eq!(bytes.len(), 32);
    /// assert_eq!(&bytes[30..], &[1, 2]);
    /// assert_eq!(f.from_bytes_be(&bytes), Some(x));
    /// ```
    #[must_use]
    pub fn to_bytes_be(&self) -> Vec<u8> {
        let c = self.canonical();
        let l = c.limbs();
        let mut b = vec![0u8; self.f.byte_len()];
        // Eight bytes a limb from the least significant end; the top limb's low bytes in front.
        let mut chunks = b.rchunks_exact_mut(8);
        for (dst, w) in (&mut chunks).zip(l) {
            dst.copy_from_slice(&w.to_be_bytes());
        }
        let top = chunks.into_remainder();
        if let Some(w) = l.last().filter(|_| !top.is_empty()) {
            let n = top.len();
            top.copy_from_slice(&w.to_be_bytes()[8 - n..]);
        }
        b
    }

    /// The little-endian encoding of the value, in exactly [`byte_len`](PrimeField::byte_len)
    /// bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bls12_381_base();
    /// let x = f.from_u64(0x0102);
    /// let bytes = x.to_bytes_le();
    /// assert_eq!(bytes.len(), 48);
    /// assert_eq!(&bytes[..2], &[2, 1]);
    /// assert_eq!(f.from_bytes_le(&bytes), Some(x));
    /// ```
    #[must_use]
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let c = self.canonical();
        let l = c.limbs();
        let mut b = vec![0u8; self.f.byte_len()];
        let (chunks, top) = b.as_chunks_mut::<8>();
        for (dst, w) in chunks.iter_mut().zip(l) {
            *dst = w.to_le_bytes();
        }
        if let Some(w) = l.last().filter(|_| !top.is_empty()) {
            let n = top.len();
            top.copy_from_slice(&w.to_le_bytes()[..n]);
        }
        b
    }

    /// The limbs of the value.
    fn canonical(&self) -> Canon {
        with_form!(&self.f.form, f => canonical_value(f, &self.v))
    }

    /// The element of this field held in `v`.
    #[inline(always)]
    fn with(&self, v: Value) -> Fp<'f> {
        Fp { f: self.f, v }
    }

    /// The element of `f` whose Montgomery residue `xR mod p` (`R = 2^(64 n)`, fully reduced) has
    /// the limbs `limbs`, the low one first, as the curves' fields hold their elements: for a
    /// field in Montgomery's form of four or six limbs (BN254's and BLS12-381's base fields and
    /// the like); `None` for another form, another number of limbs, or limbs not below `p`.
    #[inline]
    pub(crate) fn from_montgomery(f: &'f PrimeField, limbs: &[u64]) -> Option<Fp<'f>> {
        fn below(x: &[u64], m: &[u64]) -> bool {
            compare(x, m) == Ordering::Less
        }
        match &f.form {
            Form::Odd(Prime::Mont4(m)) if limbs.len() == 4 && below(limbs, &f.limbs) => {
                Some(Fp::from_raw(f, m.put(m.load(limbs))))
            }
            Form::Odd(Prime::Mont6(m)) if limbs.len() == 6 && below(limbs, &f.limbs) => {
                Some(Fp::from_raw(f, m.put(m.load(limbs))))
            }
            _ => None,
        }
    }

    /// The limbs of the element's Montgomery residue, the low one first, for a field in
    /// Montgomery's form of four or six limbs ([`from_montgomery`](Fp::from_montgomery)'s
    /// inverse); `None` for another form.
    #[inline]
    pub(crate) fn montgomery(&self) -> Option<&[u64]> {
        match &self.f.form {
            Form::Odd(Prime::Mont4(m)) => Some(&m.get(&self.v)[..]),
            Form::Odd(Prime::Mont6(m)) => Some(&m.get(&self.v)[..]),
            _ => None,
        }
    }

    /// The element as the field's form holds it, for the arithmetic of the towers.
    #[inline(always)]
    pub(super) fn raw(&self) -> &Value {
        &self.v
    }

    /// The element of `f` that its form holds as `v`.
    #[inline(always)]
    pub(super) fn from_raw(f: &'f PrimeField, v: Value) -> Fp<'f> {
        Fp { f, v }
    }

    /// The element as its form holds it, to change in place.
    #[inline(always)]
    pub(super) fn raw_mut(&mut self) -> &mut Value {
        &mut self.v
    }
}

impl PartialEq for Fp<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.f.same_field(other.f)
            && with_form!(&self.f.form, f => f.same(f.get(&self.v), f.get(&other.v)))
    }
}

impl Eq for Fp<'_> {}

impl Hash for Fp<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        // The value's limbs, which equal elements share whatever their representation.
        self.canonical().limbs().hash(h);
    }
}

impl fmt::Debug for Fp<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value(), self.f.p)
    }
}

impl fmt::Display for Fp<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value(), f)
    }
}

impl<'f> Neg for &Fp<'f> {
    type Output = Fp<'f>;
    fn neg(self) -> Fp<'f> {
        self.with(with_form!(&self.f.form, f => f.put(f.neg(f.get(&self.v)))))
    }
}

impl<'f> Neg for Fp<'f> {
    type Output = Fp<'f>;
    fn neg(self) -> Fp<'f> {
        -&self
    }
}

/// The binary operators for the four combinations of owned and borrowed operands, and their
/// assignments, from the one on two borrows.
macro_rules! binary {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $op:ident, $into:ident) => {
        impl<'f> $tr<&Fp<'f>> for &Fp<'f> {
            type Output = Fp<'f>;
            #[track_caller]
            #[inline]
            fn $m(self, rhs: &Fp<'f>) -> Fp<'f> {
                self.f.check(rhs);
                self.with($op(self.f, &self.v, &rhs.v))
            }
        }
        impl<'f> $tr<Fp<'f>> for &Fp<'f> {
            type Output = Fp<'f>;
            #[track_caller]
            fn $m(self, rhs: Fp<'f>) -> Fp<'f> {
                self.$m(&rhs)
            }
        }
        impl<'f> $tr<&Fp<'f>> for Fp<'f> {
            type Output = Fp<'f>;
            #[track_caller]
            fn $m(self, rhs: &Fp<'f>) -> Fp<'f> {
                (&self).$m(rhs)
            }
        }
        impl<'f> $tr<Fp<'f>> for Fp<'f> {
            type Output = Fp<'f>;
            #[track_caller]
            fn $m(self, rhs: Fp<'f>) -> Fp<'f> {
                (&self).$m(&rhs)
            }
        }
        impl<'f> $atr<&Fp<'f>> for Fp<'f> {
            #[track_caller]
            #[inline]
            fn $am(&mut self, rhs: &Fp<'f>) {
                self.f.check(rhs);
                $into(self.f, &mut self.v, &rhs.v);
            }
        }
        impl<'f> $atr<Fp<'f>> for Fp<'f> {
            #[track_caller]
            #[inline]
            fn $am(&mut self, rhs: Fp<'f>) {
                self.$am(&rhs);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, sum, sum_into);
binary!(Sub, sub, SubAssign, sub_assign, difference, difference_into);
binary!(Mul, mul, MulAssign, mul_assign, product, product_into);
binary!(Div, div, DivAssign, div_assign, quotient, quotient_into);

/// `a + b` in the field `k`.
fn sum(k: &PrimeField, a: &Value, b: &Value) -> Value {
    with_form!(&k.form, f => f.put(f.add(f.get(a), f.get(b))))
}

/// `a - b` in the field `k`.
fn difference(k: &PrimeField, a: &Value, b: &Value) -> Value {
    with_form!(&k.form, f => f.put(f.sub(f.get(a), f.get(b))))
}

/// `a b` in the field `k`.
fn product(k: &PrimeField, a: &Value, b: &Value) -> Value {
    with_form!(&k.form, f => f.put(f.mul(f.get(a), f.get(b))))
}

/// `a += b` in the field `k`, in place.
fn sum_into(k: &PrimeField, a: &mut Value, b: &Value) {
    with_form!(&k.form, f => f.add_into(a, b))
}

/// `a -= b` in the field `k`, in place.
fn difference_into(k: &PrimeField, a: &mut Value, b: &Value) {
    with_form!(&k.form, f => f.sub_into(a, b))
}

/// `a *= b` in the field `k`, in place.
fn product_into(k: &PrimeField, a: &mut Value, b: &Value) {
    with_form!(&k.form, f => f.mul_into(a, b))
}

/// `a /= b` in the field `k`, in place.
#[track_caller]
fn quotient_into(k: &PrimeField, a: &mut Value, b: &Value) {
    *a = quotient(k, a, b);
}

/// `a / b` in the field `k`, as `a` times the inverse of `b`.
#[track_caller]
fn quotient(k: &PrimeField, a: &Value, b: &Value) -> Value {
    with_form!(&k.form, f => {
        let Some(inv) = f.invert_value(f.get(b)) else {
            division_by_zero()
        };
        f.put(f.mul(f.get(a), &inv))
    })
}

/// The panic of a division by zero.
#[cold]
#[inline(never)]
#[track_caller]
fn division_by_zero() -> ! {
    panic!("division by zero")
}
