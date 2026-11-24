//! Hashing to the curves of [`a::math::ec`](crate::math::ec), as RFC 9380 specifies it: byte
//! strings to field elements, field elements to points, and the suites of section 8 for the named
//! curves.
//!
//! * [`hash_to_field`] (section 5.2): `count` elements of a [`PrimeField`] from a message and a
//!   domain separation tag, each the reduction of `L = ceil((ceil(log2 p) + k) / 8)` bytes of
//!   [`expand_message_xmd`] modulo `p`, so that its bias is at most `2^-k`.
//! * [`sgn0`] (section 4.1), the "sign" by which the maps choose between the two square roots.
//! * The maps of section 6: [`ShallueVanDeWoestijne`], for any curve in short Weierstrass form
//!   (section 6.6.1); [`SimplifiedSwu`], the simplified Shallue–van de Woestijne–Ulas map
//!   to a curve `y² = x³ + A·x + B` with `A·B ≠ 0` (Brier et al., "Efficient indifferentiable
//!   hashing into ordinary elliptic curves", CRYPTO 2010; Wahby and Boneh, "Fast and simple
//!   constant-time hashing to the BLS12-381 elliptic curve", CHES 2019), with an
//!   [`IsogenyMap`] onto a curve with `A·B = 0` (section 6.6.3); [`Elligator2`], to a curve
//!   `K·t² = s³ + J·s² + s` (Bernstein, Hamburg, Krasnova and Lange, "Elligator", CCS 2013),
//!   with the rational map of RFC 7748 onto edwards25519 (section 6.8.2); and the searches of
//!   appendix H for their constants `Z`.
//! * [`Suite`]: the suites of section 8 for the named curves of [`named`], each with
//!   [`hash_to_curve`](Suite::hash_to_curve), its `_RO_` suite, and
//!   [`encode_to_curve`](Suite::encode_to_curve), its `_NU_` suite: [`p256`], [`p384`], [`p521`],
//!   [`secp256k1`], [`curve25519`] and [`edwards25519`].
//!
//! The maps are the straightforward ones of section 6, with an inversion and a square root or
//! two for each element, on [`Fp`], not the straight-line forms of appendices F and G, which are
//! written for constant time and compute the same points. RFC 9380's suites for BLS12-381's
//! groups, and a hashing to BN254's `G1`, are in
//! [`a::math::pairing::hash`](crate::math::pairing::hash), with the groups of the pairings.
//!
//! **Nothing here is constant-time**: the maps branch on whether an element is a square, as the
//! rest of the library branches on its data. They are for public messages, such as those of a
//! signature's verification, not for hashing secrets an attacker could time.
//!
//! # Examples
//!
//! ```
//! use a::math::ec::hash;
//!
//! // RFC 9380, appendix J.1.1: P256_XMD:SHA-256_SSWU_RO_ of the message "abc".
//! let suite = hash::p256();
//! assert_eq!(suite.ro_id(), "P256_XMD:SHA-256_SSWU_RO_");
//! let p = suite.hash_to_curve(b"abc", b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_");
//! let (x, _) = p.coordinates().unwrap();
//! assert_eq!(x.to_str_radix(16), "bb8b87485551aa43ed54f009230450b492fead5f1cc91658775dac4a3388a0f");
//! assert!(suite.domain().in_subgroup(&p));
//! ```

use super::{Curve, Domain, Edwards, Montgomery, Point, Weierstrass, named};
use crate::hash::{Digest, Sha256, Sha384, Sha512, expand_message_xmd};
use crate::math::field::{Fp, PrimeField};
use crate::math::numbers::{ByteOrder, Number};
use crate::math::poly::ModPolynomial;
use std::sync::OnceLock;

/// An `expand_message` of RFC 9380, section 5.3: `(msg, DST, len_in_bytes) -> uniform_bytes`.
type Expander = fn(&[u8], &[u8], usize) -> Vec<u8>;

/// RFC 9380's `sgn0` for a prime field (section 4.1): whether `x`, as an integer in `[0, p)`, is
/// odd, which the RFC calls negative. 0 is not negative.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash::sgn0;
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// let f = PrimeField::new(&Number::from(13u8));
/// assert!(!sgn0(&f.zero()));
/// assert!(sgn0(&f.from_u64(5)));
/// assert!(!sgn0(&-f.from_u64(5))); // 8
/// ```
#[must_use]
pub fn sgn0(x: &Fp<'_>) -> bool {
    x.value().bit(0)
}

/// Whether `x` is a square, 0 included: RFC 9380's `is_square`.
fn is_square(x: &Fp<'_>) -> bool {
    x.legendre() >= 0
}

/// The inverse of `x`, or 0 for 0: RFC 9380's `inv0`.
fn inv0<'f>(x: &Fp<'f>) -> Fp<'f> {
    x.inverse().unwrap_or_else(|| x.field().zero())
}

/// `x` or `-x`, whichever has the sign `negative`; 0 for 0.
fn with_sign<'f>(x: Fp<'f>, negative: bool) -> Fp<'f> {
    if sgn0(&x) == negative { x } else { -x }
}

/// The element `v` of `field`, of either sign.
fn small<'f>(field: &'f PrimeField, v: i64) -> Fp<'f> {
    field.element(&Number::new_from_i64(v, ByteOrder::LittleEndian))
}

/// The element of `field` given in hexadecimal.
pub(crate) fn hex<'f>(field: &'f PrimeField, s: &str) -> Fp<'f> {
    let n = Number::from_str_radix(s, 16, ByteOrder::LittleEndian)
        .unwrap_or_else(|_| unreachable!("the constants are valid hexadecimal"));
    field.element(&n)
}

/// RFC 9380's `L` for `field` and the security level `k`: `ceil((ceil(log2 p) + k) / 8)`.
pub(crate) fn element_bytes(field: &PrimeField, k: u32) -> usize {
    let p = field.modulus();
    // ceil(log2 p) is the length of p in bits, but for p = 2, the one prime that is a power of 2.
    let log = if *p == 2u8 { 1 } else { p.bits() };
    // A modulus of 2^64 bits would not fit the memory.
    usize::try_from((log + u64::from(k)).div_ceil(8)).unwrap_or(usize::MAX)
}

/// `count` elements of `field`, `l` bytes of `expand` each.
#[track_caller]
fn to_field<'f>(
    field: &'f PrimeField,
    expand: Expander,
    l: usize,
    msg: &[u8],
    dst: &[u8],
    count: usize,
) -> Vec<Fp<'f>> {
    let len = count
        .checked_mul(l)
        .unwrap_or_else(|| panic!("{count} elements of {l} bytes are too many to expand"));
    let bytes = expand(msg, dst, len);
    bytes
        .chunks_exact(l)
        .map(|tv| field.element(&Number::new(tv.to_vec(), ByteOrder::BigEndian)))
        .collect()
}

/// `count` elements of `field` hashed from `msg` with the domain separation tag `dst`, by RFC
/// 9380's `hash_to_field` (section 5.2) with `expand_message_xmd` on the hash `H`, for the security
/// level of `k` bits.
///
/// Each element is `OS2IP(tv) mod p` for its `L = ceil((ceil(log2 p) + k) / 8)` bytes `tv` of
/// [`expand_message_xmd`]`::<H>(msg, dst, count · L)`; the `k` extra bits keep the bias of the
/// reduction below `2^-k`. This is the RFC's function for `m = 1`, the prime fields; the RFC names
/// the hash and `k` of each suite (SHA-256 and 128 for P-256), and a [`Suite`]'s
/// [`hash_to_field`](Suite::hash_to_field) takes its own.
///
/// # Panics
///
/// Panics as [`expand_message_xmd`] does: for an empty `dst`, and when `count · L` is above 65535
/// bytes or 255 blocks of the hash (170 elements of P-256 with SHA-256).
///
/// # Examples
///
/// ```
/// use a::hash::Sha256;
/// use a::math::ec::hash::{self, hash_to_field};
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// // P-256's two elements of the message "abc" in appendix J.1.1, as the suite gives them.
/// let suite = hash::p256();
/// let dst = b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_";
/// let u = hash_to_field::<Sha256>(suite.field(), b"abc", dst, 2, 128);
/// assert_eq!(u, suite.hash_to_field(b"abc", dst, 2));
/// assert_eq!(
///     u[0].value().to_str_radix(16),
///     "afe47f2ea2b10465cc26ac403194dfb68b7f5ee865cda61e9f3e07a537220af1"
/// );
///
/// // Any prime field: here 101, with L = ceil((7 + 128) / 8) = 17 bytes each.
/// let f = PrimeField::new(&Number::from(101u8));
/// let v = hash_to_field::<Sha256>(&f, b"msg", b"MY-APP-V1", 3, 128);
/// assert_eq!(v.len(), 3);
/// ```
#[track_caller]
#[must_use]
pub fn hash_to_field<'f, H: Digest>(
    field: &'f PrimeField,
    msg: &[u8],
    dst: &[u8],
    count: usize,
    k: u32,
) -> Vec<Fp<'f>> {
    to_field(
        field,
        expand_message_xmd::<H>,
        element_bytes(field, k),
        msg,
        dst,
        count,
    )
}

/// Panics unless the field's characteristic is above 3, as short Weierstrass and Montgomery
/// curves need.
#[track_caller]
fn check_large(field: &PrimeField) {
    assert!(
        *field.modulus() > 3u8,
        "the maps of RFC 9380 here are for prime fields above 3"
    );
}

/// The Shallue–van de Woestijne map of RFC 9380, section 6.6.1, from a prime field to any curve
/// `y² = x³ + A·x + B`, `A` or `B` zero included, with its constant `Z` (Shallue and van de
/// Woestijne, *Construction of rational points on elliptic curves over finite fields*, ANTS 2006,
/// in the parameterisation of Wahby, *An explicit, generic parameterization for the
/// Shallue–van de Woestijne map*, 2019).
///
/// For `u`, with `t = u²·g(Z)`, `x1 = −Z/2 − c3·u·(1 − t)/((1 − t)(1 + t))` and
/// `x2 = −Z/2 + c3·u·(1 − t)/((1 − t)(1 + t))` for `c3 = √(−g(Z)·(3Z² + 4A))` of `sgn0(c3) = 0`,
/// and `x3 = Z + c4·((1 + t)²/((1 − t)(1 + t)))²` for `c4 = −4·g(Z)/(3Z² + 4A)`, inverses by
/// `inv0`; `x` is the first of `x1`, `x2`, `x3` whose `g` is a square (one of them always is), and
/// `y` is its square root with `sgn0(y) = sgn0(u)`. That is an inversion, two Legendre symbols at
/// most, a square root and a few products: more than the simplified SWU map takes, which RFC 9380
/// prefers where it applies.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash::ShallueVanDeWoestijne;
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// // y² = x³ + 3 over F_103, which has A = 0.
/// let f = PrimeField::new(&Number::from(103u8));
/// let (a, b) = (f.zero(), f.from_u64(3));
/// let map = ShallueVanDeWoestijne::new(&a, &b, &ShallueVanDeWoestijne::find_z(&a, &b));
/// for u in 0..103 {
///     let (x, y) = map.map(&f.from_u64(u));
///     assert_eq!(y.square(), &x * &x * &x + &b);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct ShallueVanDeWoestijne<'f> {
    /// `A`.
    a: Fp<'f>,
    /// `B`.
    b: Fp<'f>,
    /// `Z`.
    z: Fp<'f>,
    /// `g(Z)`.
    gz: Fp<'f>,
    /// `−Z/2`.
    c2: Fp<'f>,
    /// `√(−g(Z)·(3Z² + 4A))` with `sgn0 = 0`.
    c3: Fp<'f>,
    /// `−4·g(Z)/(3Z² + 4A)`.
    c4: Fp<'f>,
}

impl<'f> ShallueVanDeWoestijne<'f> {
    /// The map to `y² = x³ + a·x + b` with the constant `z`.
    ///
    /// # Panics
    ///
    /// Panics if the elements are of different fields or of a field of 2 or 3 elements, if the
    /// curve is singular, and if `z` is 0 or does not meet the four criteria of section 6.6.1: `g(z) ≠ 0`,
    /// `h(z) = −(3z² + 4a)/(4·g(z))` is neither 0 nor a non-square, and `g(z)` or `g(−z/2)` is a
    /// square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::ShallueVanDeWoestijne;
    /// use a::math::field::PrimeField;
    ///
    /// // BN254's G1, y² = x³ + 3, with the Z = 1 of appendix H.1.
    /// let f = PrimeField::bn254_base();
    /// let map = ShallueVanDeWoestijne::new(&f.zero(), &f.from_u64(3), &f.one());
    /// assert_eq!(map.z(), &f.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(a: &Fp<'f>, b: &Fp<'f>, z: &Fp<'f>) -> ShallueVanDeWoestijne<'f> {
        let field = a.field();
        check_large(field);
        check_nonsingular(a, b);
        assert!(
            svdw_z_valid(a, b, z),
            "Z does not meet the criteria of RFC 9380, section 6.6.1"
        );
        let gz = (z.square() + a) * z + b;
        let k = small(field, 3) * z.square() + small(field, 4) * a;
        let c3 = (-(&gz * &k))
            .sqrt()
            .unwrap_or_else(|| unreachable!("−g(Z)·(3Z² + 4A) is a square by criterion 3"));
        let c4 = small(field, -4) * &gz / &k;
        ShallueVanDeWoestijne {
            a: a.clone(),
            b: b.clone(),
            z: z.clone(),
            c2: -(z / &small(field, 2)),
            c3: with_sign(c3, false),
            c4,
            gz,
        }
    }

    /// The `Z` of RFC 9380, appendix H.1, for `y² = x³ + a·x + b`: the first of 1, −1, 2, −2, …
    /// that meets the four criteria of section 6.6.1. It is 1 for BN254's `y² = x³ + 3`.
    ///
    /// # Panics
    ///
    /// Panics, as [`new`](ShallueVanDeWoestijne::new) does, for elements of different fields or
    /// of a field of 2 or 3 elements and for a singular curve, and if no element meets the
    /// criteria.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::ShallueVanDeWoestijne;
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bn254_base();
    /// assert_eq!(ShallueVanDeWoestijne::find_z(&f.zero(), &f.from_u64(3)), f.one());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn find_z(a: &Fp<'f>, b: &Fp<'f>) -> Fp<'f> {
        let field = a.field();
        check_large(field);
        check_nonsingular(a, b);
        let half = field.modulus() >> 1usize;
        let mut ctr = Number::new_from_u8(1, ByteOrder::LittleEndian);
        while ctr <= half {
            let c = field.element(&ctr);
            for z in [c.clone(), -c] {
                if svdw_z_valid(a, b, &z) {
                    return z;
                }
            }
            ctr += 1u8;
        }
        panic!("no element of the field meets the criteria of RFC 9380, section 6.6.1")
    }

    /// `A`.
    #[must_use]
    pub fn a(&self) -> &Fp<'f> {
        &self.a
    }

    /// `B`.
    #[must_use]
    pub fn b(&self) -> &Fp<'f> {
        &self.b
    }

    /// `Z`.
    #[must_use]
    pub fn z(&self) -> &Fp<'f> {
        &self.z
    }

    /// `g(x) = x³ + A·x + B`.
    fn g(&self, x: &Fp<'f>) -> Fp<'f> {
        (x.square() + &self.a) * x + &self.b
    }

    /// The point `(x, y)` that `u` maps to, as section 6.6.1 computes it.
    ///
    /// # Panics
    ///
    /// Panics if `u` is of another field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::{ShallueVanDeWoestijne, sgn0};
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::bn254_base();
    /// let map = ShallueVanDeWoestijne::new(&f.zero(), &f.from_u64(3), &f.one());
    /// let u = f.from_u64(12345);
    /// let (x, y) = map.map(&u);
    /// assert_eq!(y.square(), &x * &x * &x + f.from_u64(3));
    /// assert_eq!(sgn0(&y), sgn0(&u));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn map(&self, u: &Fp<'f>) -> (Fp<'f>, Fp<'f>) {
        let one = self.a.field().one();
        let tv1 = u.square() * &self.gz;
        let tv2 = &one + &tv1;
        let tv1 = &one - &tv1;
        let tv3 = inv0(&(&tv1 * &tv2));
        let tv5 = u * &tv1 * &tv3 * &self.c3;
        let x1 = &self.c2 - &tv5;
        let x2 = &self.c2 + &tv5;
        let x3 = &self.z + &self.c4 * &(tv2.square() * &tv3).square();
        let (x, y) = [x1, x2]
            .into_iter()
            .find_map(|x| self.g(&x).sqrt().map(|y| (x, y)))
            .unwrap_or_else(|| {
                let y = self.g(&x3).sqrt().unwrap_or_else(|| {
                    unreachable!("g(x3) is a square when g(x1) and g(x2) are not")
                });
                (x3, y)
            });
        (x, with_sign(y, sgn0(u)))
    }
}

/// Whether `z` meets the criteria of RFC 9380, section 6.6.1, for `y² = x³ + a·x + b`.
fn svdw_z_valid<'f>(a: &Fp<'f>, b: &Fp<'f>, z: &Fp<'f>) -> bool {
    // Section 6.6.1 asks for a nonzero Z before its four criteria, which 0 can meet.
    if z.is_zero() {
        return false;
    }
    let field = a.field();
    let g = |x: &Fp<'f>| (x.square() + a) * x + b;
    let gz = g(z);
    if gz.is_zero() {
        return false;
    }
    let h = -(small(field, 3) * z.square() + small(field, 4) * a) / (small(field, 4) * &gz);
    if h.is_zero() || !is_square(&h) {
        return false;
    }
    is_square(&gz) || is_square(&g(&-(z / &small(field, 2))))
}

/// The simplified Shallue–van de Woestijne–Ulas map of RFC 9380, section 6.6.2, from a prime
/// field to the curve `y² = x³ + A·x + B` with `A·B ≠ 0`, with its constant `Z` (Brier, Coron,
/// Icart, Madore, Randriam and Tibouchi, CRYPTO 2010; Wahby and Boneh, CHES 2019).
///
/// For `u`, `x1 = (−B/A)·(1 + 1/(Z²u⁴ + Z·u²))`, or `B/(Z·A)` where that denominator is 0; `x` is
/// `x1` if `g(x1) = x1³ + A·x1 + B` is a square and `x2 = Z·u²·x1` otherwise, as `g(x2) = Z³u⁶·g(x1)`
/// is then a square; `y` is the square root of `g(x)` with `sgn0(y) = sgn0(u)`. That is an
/// inversion, one or two square roots and a few products.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash::SimplifiedSwu;
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// // y² = x³ + 3x + 5 over F_103, with the Z appendix H.2 finds.
/// let f = PrimeField::new(&Number::from(103u8));
/// let (a, b) = (f.from_u64(3), f.from_u64(5));
/// let z = SimplifiedSwu::find_z(&a, &b);
/// let map = SimplifiedSwu::new(&a, &b, &z);
/// for u in 0..103 {
///     let (x, y) = map.map(&f.from_u64(u));
///     assert_eq!(y.square(), &x * &x * &x + &a * &x + &b);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct SimplifiedSwu<'f> {
    /// `A`.
    a: Fp<'f>,
    /// `B`.
    b: Fp<'f>,
    /// `Z`.
    z: Fp<'f>,
    /// `−B / A`.
    minus_b_over_a: Fp<'f>,
    /// `B / (Z·A)`, `x1` in the exceptional cases.
    exceptional: Fp<'f>,
}

impl<'f> SimplifiedSwu<'f> {
    /// The map to `y² = x³ + a·x + b` with the constant `z`.
    ///
    /// # Panics
    ///
    /// Panics if the elements are of different fields or of a field of 2 or 3 elements, if `a` or
    /// `b` is 0 (map to an isogenous curve then, as section 6.6.3 does), if the curve is singular,
    /// and if `z` does not meet the four criteria of section 6.6.2: `z` is not a square, `z ≠ −1`,
    /// `x³ + a·x + b − z` is irreducible, and `g(b / (z·a))` is a square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::SimplifiedSwu;
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// // y² = x³ + 3x + 5 over F_103, with Z = −13, the first that meets the criteria.
    /// let f = PrimeField::new(&Number::from(103u8));
    /// let z = -f.from_u64(13);
    /// let map = SimplifiedSwu::new(&f.from_u64(3), &f.from_u64(5), &z);
    /// assert_eq!(map.z(), &z);
    /// assert_eq!(SimplifiedSwu::find_z(&f.from_u64(3), &f.from_u64(5)), z);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(a: &Fp<'f>, b: &Fp<'f>, z: &Fp<'f>) -> SimplifiedSwu<'f> {
        let field = a.field();
        check_large(field);
        assert!(
            !a.is_zero() && !b.is_zero(),
            "the simplified SWU map needs A ≠ 0 and B ≠ 0; map to an isogenous curve (RFC 9380, section 6.6.3)"
        );
        check_nonsingular(a, b);
        assert!(
            sswu_z_valid(a, b, z),
            "Z does not meet the criteria of RFC 9380, section 6.6.2"
        );
        let inv_a = a.inverse().unwrap_or_else(|| unreachable!("A ≠ 0"));
        let exceptional = b * &inv_a / z;
        SimplifiedSwu {
            a: a.clone(),
            b: b.clone(),
            z: z.clone(),
            minus_b_over_a: -(b * &inv_a),
            exceptional,
        }
    }

    /// The `Z` of RFC 9380, appendix H.2, for the curve `y² = x³ + a·x + b`: the first of 1, −1,
    /// 2, −2, … that meets the four criteria of section 6.6.2. It is −10 for P-256, −12 for
    /// P-384, −4 for P-521 and −11 for the curve isogenous to secp256k1 of section 8.7.
    ///
    /// # Panics
    ///
    /// Panics, as [`new`](SimplifiedSwu::new) does, if `a` and `b` are of different fields or of a
    /// field of 2 or 3 elements, if `a` or `b` is 0 and if the curve is singular; and if no element
    /// meets the criteria, which can happen only in very small fields (it was seen up to `p = 89`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::{Curve, hash::SimplifiedSwu, named};
    /// use a::math::field::PrimeField;
    ///
    /// let curve = named::p256();
    /// let f = PrimeField::new(curve.modulus());
    /// let z = SimplifiedSwu::find_z(&f.element(curve.a()), &f.element(curve.b()));
    /// assert_eq!(z, -f.from_u64(10));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn find_z(a: &Fp<'f>, b: &Fp<'f>) -> Fp<'f> {
        let field = a.field();
        check_large(field);
        assert!(
            !a.is_zero() && !b.is_zero(),
            "the simplified SWU map needs A ≠ 0 and B ≠ 0"
        );
        check_nonsingular(a, b);
        let half = field.modulus() >> 1usize;
        let mut ctr = Number::new_from_u8(1, ByteOrder::LittleEndian);
        while ctr <= half {
            let c = field.element(&ctr);
            for z in [c.clone(), -c] {
                if sswu_z_valid(a, b, &z) {
                    return z;
                }
            }
            ctr += 1u8;
        }
        panic!("no element of the field meets the criteria of RFC 9380, section 6.6.2")
    }

    /// `A`.
    #[must_use]
    pub fn a(&self) -> &Fp<'f> {
        &self.a
    }

    /// `B`.
    #[must_use]
    pub fn b(&self) -> &Fp<'f> {
        &self.b
    }

    /// `Z`.
    #[must_use]
    pub fn z(&self) -> &Fp<'f> {
        &self.z
    }

    /// `g(x) = x³ + A·x + B`.
    fn g(&self, x: &Fp<'f>) -> Fp<'f> {
        (x.square() + &self.a) * x + &self.b
    }

    /// The point `(x, y)` that `u` maps to, as section 6.6.2 computes it.
    ///
    /// # Panics
    ///
    /// Panics if `u` is of another field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::{SimplifiedSwu, sgn0};
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(103u8));
    /// let map = SimplifiedSwu::new(&f.from_u64(3), &f.from_u64(5), &-f.from_u64(13));
    /// let (x, y) = map.map(&f.from_u64(7));
    /// // u and −u give the same x, and y the sign of u.
    /// let (x2, y2) = map.map(&-f.from_u64(7));
    /// assert_eq!((x2, y2), (x, -y.clone()));
    /// assert_eq!(sgn0(&y), sgn0(&f.from_u64(7)));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn map(&self, u: &Fp<'f>) -> (Fp<'f>, Fp<'f>) {
        let zu2 = &self.z * &u.square();
        let tv1 = inv0(&(zu2.square() + &zu2));
        let x1 = if tv1.is_zero() {
            self.exceptional.clone()
        } else {
            &self.minus_b_over_a * &(tv1 + self.a.field().one())
        };
        let (x, y) = match self.g(&x1).sqrt() {
            Some(y) => (x1, y),
            None => {
                let x2 = zu2 * x1;
                let y = self.g(&x2).sqrt().unwrap_or_else(|| {
                    unreachable!("g(x2) = Z³u⁶·g(x1) is a square when g(x1) is not")
                });
                (x2, y)
            }
        };
        (x, with_sign(y, sgn0(u)))
    }
}

/// Panics if `y² = x³ + a·x + b` is singular: `4·a³ + 27·b² = 0` (and for elements of different
/// fields).
#[track_caller]
fn check_nonsingular<'f>(a: &Fp<'f>, b: &Fp<'f>) {
    let field = a.field();
    let disc = small(field, 4) * a * a * a + small(field, 27) * b * b;
    assert!(!disc.is_zero(), "the curve is singular");
}

/// Whether `z` meets the criteria of RFC 9380, section 6.6.2, for `y² = x³ + a·x + b`.
fn sswu_z_valid<'f>(a: &Fp<'f>, b: &Fp<'f>, z: &Fp<'f>) -> bool {
    let field = a.field();
    if is_square(z) || z == &-field.one() {
        return false;
    }
    let g = |x: &Fp<'f>| -> bool { is_square(&((x.square() + a) * x + b)) };
    if !g(&(b / &(z * a))) {
        return false;
    }
    // g(x) − Z, a cubic, is irreducible.
    let p = field.modulus();
    ModPolynomial::new(
        &[
            (b - z).value(),
            a.value(),
            Number::default(),
            Number::new_from_u8(1, ByteOrder::LittleEndian),
        ],
        p,
    )
    .is_irreducible()
}

/// The Elligator 2 map of RFC 9380, section 6.7.1, from a prime field to the Montgomery curve
/// `K·t² = s³ + J·s² + s`, with its constant `Z` (Bernstein, Hamburg, Krasnova and Lange, CCS
/// 2013).
///
/// For `u`, `x1 = −(J/K) / (1 + Z·u²)`, or `−J/K` where that denominator is 0; `x` is `x1` with
/// `sgn0(y) = 1` if `g(x1) = x1³ + (J/K)·x1² + x1/K²` is a square, otherwise `x2 = −x1 − J/K` with
/// `sgn0(y) = 0`; the point is `(x·K, y·K)`. That is an inversion, one or two square roots and a
/// few products.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash::Elligator2;
/// use a::math::field::PrimeField;
/// use a::math::numbers::Number;
///
/// // 3·t² = s³ + 4·s² + s over F_101: (4² − 4) / 3² is not a square.
/// let f = PrimeField::new(&Number::from(101u8));
/// let (j, k) = (f.from_u64(4), f.from_u64(3));
/// let map = Elligator2::new(&j, &k, &Elligator2::find_z(&f));
/// for u in 0..101 {
///     let (s, t) = map.map(&f.from_u64(u));
///     assert_eq!(&k * &t.square(), &s * &s * &s + &j * &s * &s + &s);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Elligator2<'f> {
    /// `J`.
    j: Fp<'f>,
    /// `K`.
    k: Fp<'f>,
    /// `Z`.
    z: Fp<'f>,
    /// `J / K`.
    j_over_k: Fp<'f>,
    /// `1 / K²`.
    inv_k2: Fp<'f>,
}

impl<'f> Elligator2<'f> {
    /// The map to `k·t² = s³ + j·s² + s` with the constant `z`.
    ///
    /// # Panics
    ///
    /// Panics if the elements are of different fields or of a field of 2 or 3 elements, if `j`
    /// or `k` is 0, if `(j² − 4) / k²` is 0 or a square (section 6.7.1's precondition: the curve
    /// is not singular and has a single point of order 2), and if `z` is a square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::Elligator2;
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(101u8));
    /// let map = Elligator2::new(&f.from_u64(4), &f.from_u64(3), &f.from_u64(2));
    /// assert_eq!(map.z(), &f.from_u64(2));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(j: &Fp<'f>, k: &Fp<'f>, z: &Fp<'f>) -> Elligator2<'f> {
        let field = j.field();
        check_large(field);
        assert!(
            k.field().modulus() == field.modulus() && z.field().modulus() == field.modulus(),
            "elements of different fields"
        );
        assert!(
            !j.is_zero() && !k.is_zero(),
            "Elligator 2 needs J ≠ 0 and K ≠ 0"
        );
        let inv_k = k.inverse().unwrap_or_else(|| unreachable!("K ≠ 0"));
        let inv_k2 = inv_k.square();
        let d = (j.square() - small(field, 4)) * &inv_k2;
        assert!(
            !d.is_zero() && d.legendre() == -1,
            "Elligator 2 needs (J² − 4) / K² to be neither 0 nor a square (RFC 9380, section 6.7.1)"
        );
        assert!(
            !is_square(z),
            "Z must not be a square (RFC 9380, section 6.7.1)"
        );
        Elligator2 {
            j: j.clone(),
            k: k.clone(),
            z: z.clone(),
            j_over_k: j * &inv_k,
            inv_k2,
        }
    }

    /// The `Z` of RFC 9380, appendix H.3: the first of 1, −1, 2, −2, … that is not a square in
    /// `field`. It is 2 for curve25519's field.
    ///
    /// # Panics
    ///
    /// Panics for the field of 2 elements, where every element is a square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::{Curve, hash::Elligator2, named};
    /// use a::math::field::PrimeField;
    ///
    /// let f = PrimeField::new(named::curve25519().modulus());
    /// assert_eq!(Elligator2::find_z(&f), f.from_u64(2));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn find_z(field: &'f PrimeField) -> Fp<'f> {
        assert!(
            *field.modulus() > 2u8,
            "every element of the field of 2 elements is a square"
        );
        // 1 is a square; half the elements are not, and some c in [1, (p − 1)/2] or its negative
        // is one of them.
        let mut ctr = Number::new_from_u8(1, ByteOrder::LittleEndian);
        loop {
            let c = field.element(&ctr);
            for z in [c.clone(), -c] {
                if !is_square(&z) {
                    return z;
                }
            }
            ctr += 1u8;
        }
    }

    /// `J`.
    #[must_use]
    pub fn j(&self) -> &Fp<'f> {
        &self.j
    }

    /// `K`.
    #[must_use]
    pub fn k(&self) -> &Fp<'f> {
        &self.k
    }

    /// `Z`.
    #[must_use]
    pub fn z(&self) -> &Fp<'f> {
        &self.z
    }

    /// `x³ + (J/K)·x² + x/K²`.
    fn g(&self, x: &Fp<'f>) -> Fp<'f> {
        ((x + &self.j_over_k) * x + &self.inv_k2) * x
    }

    /// The point `(s, t)` that `u` maps to, as section 6.7.1 computes it.
    ///
    /// # Panics
    ///
    /// Panics if `u` is of another field.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::Elligator2;
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// let f = PrimeField::new(&Number::from(101u8));
    /// let map = Elligator2::new(&f.from_u64(4), &f.from_u64(3), &f.from_u64(2));
    /// // u and −u give the same point.
    /// assert_eq!(map.map(&f.from_u64(9)), map.map(&-f.from_u64(9)));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn map(&self, u: &Fp<'f>) -> (Fp<'f>, Fp<'f>) {
        let field = self.j.field();
        let mut x1 = -(&self.j_over_k * &inv0(&(&self.z * &u.square() + field.one())));
        if x1.is_zero() {
            x1 = -self.j_over_k.clone();
        }
        let gx1 = self.g(&x1);
        let (x, y) = match gx1.sqrt() {
            Some(y) => (x1, with_sign(y, true)),
            None => {
                let x2 = -x1 - &self.j_over_k;
                let y = self.g(&x2).sqrt().unwrap_or_else(|| {
                    unreachable!("g(x2) is a square when g(x1) is not, as Z is not")
                });
                (x2, with_sign(y, false))
            }
        };
        (x * &self.k, y * &self.k)
    }
}

/// A rational map `(x, y) → (x_num(x) / x_den(x), y · y_num(x) / y_den(x))` between curves in
/// short Weierstrass form, the shape in which RFC 9380, appendix E, gives its isogenies, such as
/// the 3-isogeny from the curve `E'` of section 8.7 onto secp256k1.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
/// use a::math::numbers::{ByteOrder, Number};
///
/// // secp256k1's suite maps to E' and then by its 3-isogeny.
/// let suite = hash::secp256k1();
/// let iso = suite.isogeny().unwrap();
/// let f = suite.field();
/// assert_eq!(iso.degree(), 3);
/// // x_den = (x − x0)²: x0 is the abscissa of the kernel, which goes to the point at infinity.
/// let x0 = "89291c84de3e11f1041da6957255eed5fc964a4df050df221d6ad4ce6ab9c5a5";
/// let x0 = f.element(&Number::from_str_radix(x0, 16, ByteOrder::BigEndian).unwrap());
/// assert_eq!(iso.map(&x0, &f.one()), None);
/// ```
#[derive(Clone, Debug)]
pub struct IsogenyMap<'f> {
    /// The coefficients of `x_num`, the constant first.
    x_num: Vec<Fp<'f>>,
    /// The coefficients of `x_den`, the constant first.
    x_den: Vec<Fp<'f>>,
    /// The coefficients of `y_num`, the constant first.
    y_num: Vec<Fp<'f>>,
    /// The coefficients of `y_den`, the constant first.
    y_den: Vec<Fp<'f>>,
}

impl<'f> IsogenyMap<'f> {
    /// The map with the four polynomials given by their coefficients, the constant first: the
    /// `k_(1,i)`, `k_(2,i)`, `k_(3,i)` and `k_(4,i)` of RFC 9380, appendix E, with the leading
    /// coefficients 1 of `x_den` and `y_den` that the RFC leaves out written in.
    ///
    /// # Panics
    ///
    /// Panics if a polynomial has no coefficients, if its leading one is 0, or if they are of
    /// different fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash::IsogenyMap;
    /// use a::math::field::PrimeField;
    /// use a::math::numbers::Number;
    ///
    /// // The identity map, x / 1 and y · 1 / 1.
    /// let f = PrimeField::new(&Number::from(101u8));
    /// let (zero, one) = (f.zero(), f.one());
    /// let id = IsogenyMap::new(&[zero, one.clone()], &[one.clone()], &[one.clone()], &[one]);
    /// assert_eq!(id.map(&f.from_u64(4), &f.from_u64(7)), Some((f.from_u64(4), f.from_u64(7))));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(
        x_num: &[Fp<'f>],
        x_den: &[Fp<'f>],
        y_num: &[Fp<'f>],
        y_den: &[Fp<'f>],
    ) -> IsogenyMap<'f> {
        let all = [x_num, x_den, y_num, y_den];
        let first = all
            .iter()
            .find_map(|c| c.first())
            .expect("a polynomial of the map has no coefficients");
        for c in all {
            let lead = c
                .last()
                .expect("a polynomial of the map has no coefficients");
            assert!(!lead.is_zero(), "a leading coefficient of the map is 0");
            for e in c {
                assert!(
                    e.field().modulus() == first.field().modulus(),
                    "elements of different fields"
                );
            }
        }
        IsogenyMap {
            x_num: x_num.to_vec(),
            x_den: x_den.to_vec(),
            y_num: y_num.to_vec(),
            y_den: y_den.to_vec(),
        }
    }

    /// The degree of the map, that of `x_num`: the degree of the isogeny.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.x_num.len() - 1
    }

    /// The coefficients of `x_num`, the constant first.
    #[must_use]
    pub fn x_num(&self) -> &[Fp<'f>] {
        &self.x_num
    }

    /// The coefficients of `x_den`, the constant first.
    #[must_use]
    pub fn x_den(&self) -> &[Fp<'f>] {
        &self.x_den
    }

    /// The coefficients of `y_num`, the constant first.
    #[must_use]
    pub fn y_num(&self) -> &[Fp<'f>] {
        &self.y_num
    }

    /// The coefficients of `y_den`, the constant first.
    #[must_use]
    pub fn y_den(&self) -> &[Fp<'f>] {
        &self.y_den
    }

    /// The image of the point `(x, y)`, or `None`, the point at infinity, where `x_den` or `y_den`
    /// is 0, as section 6.6.3 requires for the exceptional cases.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is of another field.
    #[track_caller]
    #[must_use]
    pub fn map(&self, x: &Fp<'f>, y: &Fp<'f>) -> Option<(Fp<'f>, Fp<'f>)> {
        let horner = |c: &[Fp<'f>]| {
            let mut acc = c[c.len() - 1].clone();
            for k in c[..c.len() - 1].iter().rev() {
                acc = acc * x + k;
            }
            acc
        };
        let (xd, yd) = (horner(&self.x_den), horner(&self.y_den));
        // One inversion for both denominators; it is None exactly when one of them is 0.
        let inv = (&xd * &yd).inverse()?;
        let xn = horner(&self.x_num);
        let yn = horner(&self.y_num);
        Some((xn * &yd * &inv, y * &yn * &xd * &inv))
    }
}

/// How a suite maps an element to its curve.
#[derive(Debug)]
enum Mapping {
    /// The simplified SWU map to the curve itself.
    Sswu(SimplifiedSwu<'static>),
    /// The simplified SWU map to an isogenous curve, then the isogeny.
    SswuIsogeny(SimplifiedSwu<'static>, IsogenyMap<'static>),
    /// Elligator 2 to the Montgomery curve itself.
    Elligator2(Elligator2<'static>),
    /// Elligator 2 to curve25519, then RFC 7748's map to edwards25519 with the constant
    /// `c1 = sqrt(−486664)`.
    Elligator2Edwards(Elligator2<'static>, Fp<'static>),
}

/// The suites of RFC 9380, section 8, for one of the [named] curves: `hash_to_curve` (the
/// suite whose ID ends in `_RO_`, a random oracle to the curve's prime-order subgroup) and
/// `encode_to_curve` (the `_NU_` suite, cheaper and not uniform), with their `hash_to_field`,
/// `map_to_curve` and `clear_cofactor`.
///
/// | Function | Suites | `k` | `L` | Map | `h_eff` |
/// | --- | --- | ---: | ---: | --- | ---: |
/// | [`p256`] | `P256_XMD:SHA-256_SSWU_RO_`, `_NU_` | 128 | 48 | simplified SWU, `Z = −10` | 1 |
/// | [`p384`] | `P384_XMD:SHA-384_SSWU_RO_`, `_NU_` | 192 | 72 | simplified SWU, `Z = −12` | 1 |
/// | [`p521`] | `P521_XMD:SHA-512_SSWU_RO_`, `_NU_` | 256 | 98 | simplified SWU, `Z = −4` | 1 |
/// | [`secp256k1`] | `secp256k1_XMD:SHA-256_SSWU_RO_`, `_NU_` | 128 | 48 | simplified SWU to `E'`, `Z = −11`, and the 3-isogeny | 1 |
/// | [`curve25519`] | `curve25519_XMD:SHA-512_ELL2_RO_`, `_NU_` | 128 | 48 | Elligator 2, `Z = 2` | 8 |
/// | [`edwards25519`] | `edwards25519_XMD:SHA-512_ELL2_RO_`, `_NU_` | 128 | 48 | Elligator 2 to curve25519 and RFC 7748's map | 8 |
///
/// The domain separation tag is the application's (section 3.1 says how to choose one: it must
/// not be empty, and should name the application, its version and the suite).
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
///
/// let suite = hash::edwards25519();
/// assert_eq!(suite.nu_id(), "edwards25519_XMD:SHA-512_ELL2_NU_");
/// let dst = b"MY-APP-V01-CS01-with-edwards25519_XMD:SHA-512_ELL2_RO_";
/// let p = suite.hash_to_curve(b"a message", dst);
/// assert!(suite.domain().in_subgroup(&p));
/// // hash_to_curve is the cofactor times the sum of the two maps.
/// let u = suite.hash_to_field(b"a message", dst, 2);
/// let sum = suite.map_to_curve(&u[0]) + suite.map_to_curve(&u[1]);
/// assert_eq!(p, suite.clear_cofactor(&sum));
/// ```
pub struct Suite<C: Curve + 'static> {
    /// The ID of the `hash_to_curve` suite.
    ro: &'static str,
    /// The ID of the `encode_to_curve` suite.
    nu: &'static str,
    /// The curve and its group.
    domain: &'static Domain<C>,
    /// The curve's field.
    field: &'static PrimeField,
    /// `expand_message_xmd` with the suite's hash.
    expand: Expander,
    /// The security level in bits.
    k: u32,
    /// The bytes of each element of `hash_to_field`.
    l: usize,
    /// The map.
    mapping: Mapping,
    /// The multiple that clears the cofactor.
    h_eff: Number,
}

impl<C: Curve + 'static> core::fmt::Debug for Suite<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Suite")
            .field("ro", &self.ro)
            .field("nu", &self.nu)
            .field("k", &self.k)
            .field("l", &self.l)
            .field("h_eff", &self.h_eff)
            .finish_non_exhaustive()
    }
}

impl<C: Curve + 'static> Suite<C> {
    /// The suite on `domain`, over `field`, its prime's.
    fn new(
        ids: [&'static str; 2],
        domain: &'static Domain<C>,
        field: &'static PrimeField,
        expand: Expander,
        k: u32,
        mapping: Mapping,
    ) -> Suite<C> {
        Suite {
            ro: ids[0],
            nu: ids[1],
            domain,
            field,
            expand,
            k,
            l: element_bytes(field, k),
            mapping,
            // h_eff is the cofactor on these curves.
            h_eff: domain.cofactor().clone(),
        }
    }

    /// The ID of the suite of [`hash_to_curve`](Suite::hash_to_curve), as section 8 names it:
    /// `P256_XMD:SHA-256_SSWU_RO_` and the like.
    #[must_use]
    pub fn ro_id(&self) -> &'static str {
        self.ro
    }

    /// The ID of the suite of [`encode_to_curve`](Suite::encode_to_curve), as section 8 names it:
    /// `P256_XMD:SHA-256_SSWU_NU_` and the like.
    #[must_use]
    pub fn nu_id(&self) -> &'static str {
        self.nu
    }

    /// The curve, with its generator, order and cofactor: one of the [named] curves.
    #[must_use]
    pub fn domain(&self) -> &'static Domain<C> {
        self.domain
    }

    /// The field of the curve's coordinates, in which [`hash_to_field`](Suite::hash_to_field)
    /// gives its elements.
    #[must_use]
    pub fn field(&self) -> &'static PrimeField {
        self.field
    }

    /// The security level `k` of the suite, in bits.
    #[must_use]
    pub fn k(&self) -> u32 {
        self.k
    }

    /// The length `L` in bytes of the uniform string behind each element of
    /// [`hash_to_field`](Suite::hash_to_field): `ceil((ceil(log2 p) + k) / 8)`.
    #[must_use]
    pub fn l(&self) -> usize {
        self.l
    }

    /// The scalar `h_eff` of [`clear_cofactor`](Suite::clear_cofactor): the curve's cofactor
    /// here, 1 or 8.
    #[must_use]
    pub fn h_eff(&self) -> &Number {
        &self.h_eff
    }

    /// The constant `Z` of the suite's map.
    #[must_use]
    pub fn z(&self) -> &Fp<'static> {
        match &self.mapping {
            Mapping::Sswu(m) | Mapping::SswuIsogeny(m, _) => m.z(),
            Mapping::Elligator2(m) | Mapping::Elligator2Edwards(m, _) => m.z(),
        }
    }

    /// The isogeny from the curve the map reaches onto the suite's curve, on secp256k1 alone.
    #[must_use]
    pub fn isogeny(&self) -> Option<&IsogenyMap<'static>> {
        match &self.mapping {
            Mapping::SswuIsogeny(_, iso) => Some(iso),
            _ => None,
        }
    }

    /// `count` elements of the curve's field from `msg` and the domain separation tag `dst`, by
    /// [`hash_to_field`] with the suite's hash and `k`.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty or `count` is too large for `expand_message_xmd`.
    #[track_caller]
    #[must_use]
    pub fn hash_to_field(&self, msg: &[u8], dst: &[u8], count: usize) -> Vec<Fp<'static>> {
        to_field(self.field, self.expand, self.l, msg, dst, count)
    }

    /// The point of the curve that `u` maps to, by the suite's map (and isogeny or rational
    /// map), before the cofactor is cleared. The exceptional cases of section 6 give the points
    /// the RFC fixes for them.
    ///
    /// # Panics
    ///
    /// Panics if `u` is not an element of the suite's field.
    #[track_caller]
    #[must_use]
    pub fn map_to_curve(&self, u: &Fp<'_>) -> Point<'static, C> {
        let curve: &'static C = self.domain;
        let xy = match &self.mapping {
            Mapping::Sswu(m) => Some(m.map(u)),
            Mapping::SswuIsogeny(m, iso) => {
                let (x, y) = m.map(u);
                iso.map(&x, &y)
            }
            Mapping::Elligator2(m) => Some(m.map(u)),
            Mapping::Elligator2Edwards(m, c1) => {
                let (s, t) = m.map(u);
                Some(montgomery_to_edwards25519(c1, &s, &t))
            }
        };
        match xy {
            None => curve.identity(),
            Some((x, y)) => curve
                .point(&x.value(), &y.value())
                .unwrap_or_else(|_| unreachable!("the maps give points of the curve")),
        }
    }

    /// `h_eff` times `p`, a point of the curve's prime-order subgroup (section 7).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a point of the suite's curve.
    #[track_caller]
    #[must_use]
    pub fn clear_cofactor<'c>(&self, p: &Point<'c, C>) -> Point<'c, C> {
        p.check_curve(self.domain);
        if self.h_eff == 1u8 {
            p.clone()
        } else {
            p * &self.h_eff
        }
    }

    /// `hash_to_curve` of section 3, the suite whose ID ends in `_RO_`: the cofactor cleared
    /// from the sum of the points that the two elements of
    /// [`hash_to_field`](Suite::hash_to_field)`(msg, dst, 2)` map to. Its output is close to
    /// uniform in the prime-order subgroup.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash;
    ///
    /// // Appendix J.8.1: secp256k1_XMD:SHA-256_SSWU_RO_ of the empty message.
    /// let suite = hash::secp256k1();
    /// let p = suite.hash_to_curve(b"", b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_");
    /// let (x, y) = p.coordinates().unwrap();
    /// assert_eq!(x.to_str_radix(16), "c1cae290e291aee617ebaef1be6d73861479c48b841eaba9b7b5852ddfeb1346");
    /// assert_eq!(y.to_str_radix(16), "64fa678e07ae116126f08b022a94af6de15985c996c3a91b64c406a960e51067");
    /// ```
    #[track_caller]
    #[must_use]
    pub fn hash_to_curve(&self, msg: &[u8], dst: &[u8]) -> Point<'static, C> {
        let u = self.hash_to_field(msg, dst, 2);
        let q = self.map_to_curve(&u[0]) + self.map_to_curve(&u[1]);
        self.clear_cofactor(&q)
    }

    /// `encode_to_curve` of section 3, the suite whose ID ends in `_NU_`: the cofactor cleared
    /// from the point that the one element of [`hash_to_field`](Suite::hash_to_field)`(msg, dst,
    /// 1)` maps to: one map where [`hash_to_curve`](Suite::hash_to_curve) takes two, and an
    /// output that reaches only a part of the subgroup, not uniformly.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::hash;
    ///
    /// // Appendix J.4.2: curve25519_XMD:SHA-512_ELL2_NU_ of the empty message.
    /// let suite = hash::curve25519();
    /// let p = suite.encode_to_curve(b"", b"QUUX-V01-CS02-with-curve25519_XMD:SHA-512_ELL2_NU_");
    /// let (x, _) = p.coordinates().unwrap();
    /// assert_eq!(x.to_str_radix(16), "1bb913f0c9daefa0b3375378ffa534bda5526c97391952a7789eb976edfe4d08");
    /// ```
    #[track_caller]
    #[must_use]
    pub fn encode_to_curve(&self, msg: &[u8], dst: &[u8]) -> Point<'static, C> {
        let u = self.hash_to_field(msg, dst, 1);
        self.clear_cofactor(&self.map_to_curve(&u[0]))
    }
}

/// RFC 7748's rational map from curve25519 to edwards25519 as RFC 9380 fixes it (section 8.5
/// and appendix G.2.2): `(v, w) = (c1·s/t, (s − 1)/(s + 1))`, with `c1 = sqrt(−486664)` of
/// `sgn0(c1) = 0`, and the identity `(0, 1)` where `t = 0` or `s = −1`.
fn montgomery_to_edwards25519<'f>(c1: &Fp<'f>, s: &Fp<'f>, t: &Fp<'f>) -> (Fp<'f>, Fp<'f>) {
    let field = s.field();
    let one = field.one();
    let s1 = s + &one;
    match (t * &s1).inverse() {
        None => (field.zero(), one),
        Some(inv) => {
            let v = c1 * s * &s1 * &inv;
            let w = (s - &one) * t * &inv;
            (v, w)
        }
    }
}

/// The field of a named curve's prime, kept in `cell`.
fn field_of<C: Curve>(
    cell: &'static OnceLock<PrimeField>,
    domain: &Domain<C>,
) -> &'static PrimeField {
    cell.get_or_init(|| PrimeField::new(domain.modulus()))
}

/// A suite on a named Weierstrass curve by the simplified SWU map to the curve itself.
fn sswu_suite(
    ids: [&'static str; 2],
    domain: &'static Domain<Weierstrass>,
    cell: &'static OnceLock<PrimeField>,
    expand: Expander,
    k: u32,
    z: i64,
) -> Suite<Weierstrass> {
    let field = field_of(cell, domain);
    let (a, b) = (field.element(domain.a()), field.element(domain.b()));
    let map = SimplifiedSwu::new(&a, &b, &small(field, z));
    Suite::new(ids, domain, field, expand, k, Mapping::Sswu(map))
}

/// The suites of P-256: `P256_XMD:SHA-256_SSWU_RO_` and `P256_XMD:SHA-256_SSWU_NU_` (RFC 9380,
/// section 8.2), on [`named::p256`]. The first call builds them.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
///
/// let suite = hash::p256();
/// assert_eq!((suite.k(), suite.l()), (128, 48));
/// assert_eq!(suite.z(), &-suite.field().from_u64(10));
/// ```
#[must_use]
pub fn p256() -> &'static Suite<Weierstrass> {
    static FIELD: OnceLock<PrimeField> = OnceLock::new();
    static SUITE: OnceLock<Suite<Weierstrass>> = OnceLock::new();
    SUITE.get_or_init(|| {
        sswu_suite(
            ["P256_XMD:SHA-256_SSWU_RO_", "P256_XMD:SHA-256_SSWU_NU_"],
            named::p256(),
            &FIELD,
            expand_message_xmd::<Sha256>,
            128,
            -10,
        )
    })
}

/// The suites of P-384: `P384_XMD:SHA-384_SSWU_RO_` and `P384_XMD:SHA-384_SSWU_NU_` (RFC 9380,
/// section 8.3), on [`named::p384`]. The first call builds them.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
///
/// let suite = hash::p384();
/// assert_eq!((suite.k(), suite.l()), (192, 72));
/// ```
#[must_use]
pub fn p384() -> &'static Suite<Weierstrass> {
    static FIELD: OnceLock<PrimeField> = OnceLock::new();
    static SUITE: OnceLock<Suite<Weierstrass>> = OnceLock::new();
    SUITE.get_or_init(|| {
        sswu_suite(
            ["P384_XMD:SHA-384_SSWU_RO_", "P384_XMD:SHA-384_SSWU_NU_"],
            named::p384(),
            &FIELD,
            expand_message_xmd::<Sha384>,
            192,
            -12,
        )
    })
}

/// The suites of P-521: `P521_XMD:SHA-512_SSWU_RO_` and `P521_XMD:SHA-512_SSWU_NU_` (RFC 9380,
/// section 8.4), on [`named::p521`]. The first call builds them.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
///
/// let suite = hash::p521();
/// assert_eq!((suite.k(), suite.l()), (256, 98));
/// ```
#[must_use]
pub fn p521() -> &'static Suite<Weierstrass> {
    static FIELD: OnceLock<PrimeField> = OnceLock::new();
    static SUITE: OnceLock<Suite<Weierstrass>> = OnceLock::new();
    SUITE.get_or_init(|| {
        sswu_suite(
            ["P521_XMD:SHA-512_SSWU_RO_", "P521_XMD:SHA-512_SSWU_NU_"],
            named::p521(),
            &FIELD,
            expand_message_xmd::<Sha512>,
            256,
            -4,
        )
    })
}

/// The suites of secp256k1: `secp256k1_XMD:SHA-256_SSWU_RO_` and
/// `secp256k1_XMD:SHA-256_SSWU_NU_` (RFC 9380, section 8.7), on [`named::secp256k1`]: the
/// simplified SWU map to the isogenous curve
/// `E': y'² = x'³ + A'·x' + 1771`, `A' = 0x3f8731ab…a444533`, with `Z = −11`, then the 3-isogeny
/// of appendix E.1 onto secp256k1 (whose `A = 0` the map cannot take directly). The first call
/// builds them.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
///
/// let suite = hash::secp256k1();
/// assert_eq!(suite.z(), &-suite.field().from_u64(11));
/// assert_eq!(suite.isogeny().map(|i| i.degree()), Some(3));
/// ```
#[must_use]
pub fn secp256k1() -> &'static Suite<Weierstrass> {
    static FIELD: OnceLock<PrimeField> = OnceLock::new();
    static SUITE: OnceLock<Suite<Weierstrass>> = OnceLock::new();
    SUITE.get_or_init(|| {
        let domain = named::secp256k1();
        let f = field_of(&FIELD, domain);
        let h = |s| hex(f, s);
        let map = SimplifiedSwu::new(
            &h("3f8731abdd661adca08a5558f0f5d272e953d363cb6f0e5d405447c01a444533"),
            &small(f, 1771),
            &small(f, -11),
        );
        let x_num = [
            h("8e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38daaaaa8c7"),
            h("07d3d4c80bc321d5b9f315cea7fd44c5d595d2fc0bf63b92dfff1044f17c6581"),
            h("534c328d23f234e6e2a413deca25caece4506144037c40314ecbd0b53d9dd262"),
            h("8e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38e38daaaaa88c"),
        ];
        let x_den = [
            h("d35771193d94918a9ca34ccbb7b640dd86cd409542f8487d9fe6b745781eb49b"),
            h("edadc6f64383dc1df7c4b2d51b54225406d36b641f5e41bbc52a56612a8c6d14"),
            f.one(),
        ];
        let y_num = [
            h("4bda12f684bda12f684bda12f684bda12f684bda12f684bda12f684b8e38e23c"),
            h("c75e0c32d5cb7c0fa9d0a54b12a0a6d5647ab046d686da6fdffc90fc201d71a3"),
            h("29a6194691f91a73715209ef6512e576722830a201be2018a765e85a9ecee931"),
            h("2f684bda12f684bda12f684bda12f684bda12f684bda12f684bda12f38e38d84"),
        ];
        let y_den = [
            h("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffff93b"),
            h("7a06534bb8bdb49fd5e9e6632722c2989467c1bfc8e8d978dfb425d2685c2573"),
            h("6484aa716545ca2cf3a70c3fa8fe337e0a3d21162f0d6299a7bf8192bfd2a76f"),
            f.one(),
        ];
        let iso = IsogenyMap::new(&x_num, &x_den, &y_num, &y_den);
        Suite::new(
            [
                "secp256k1_XMD:SHA-256_SSWU_RO_",
                "secp256k1_XMD:SHA-256_SSWU_NU_",
            ],
            domain,
            f,
            expand_message_xmd::<Sha256>,
            128,
            Mapping::SswuIsogeny(map, iso),
        )
    })
}

/// Curve25519's field, which its suites and edwards25519's share.
static FIELD25519: OnceLock<PrimeField> = OnceLock::new();

/// Elligator 2 to curve25519, `t² = s³ + 486662·s² + s`, with `Z = 2`.
fn elligator2_25519() -> Elligator2<'static> {
    let domain = named::curve25519();
    let f = field_of(&FIELD25519, domain);
    Elligator2::new(&f.element(domain.a()), &f.element(domain.b()), &small(f, 2))
}

/// The suites of curve25519: `curve25519_XMD:SHA-512_ELL2_RO_` and
/// `curve25519_XMD:SHA-512_ELL2_NU_` (RFC 9380, section 8.5), on [`named::curve25519`]: Elligator
/// 2 with `Z = 2`, and the cofactor 8 cleared. The first call builds them.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
/// use a::math::numbers::Number;
///
/// let suite = hash::curve25519();
/// assert_eq!(suite.h_eff(), &Number::from(8u8));
/// assert_eq!(suite.z(), &suite.field().from_u64(2));
/// ```
#[must_use]
pub fn curve25519() -> &'static Suite<Montgomery> {
    static SUITE: OnceLock<Suite<Montgomery>> = OnceLock::new();
    SUITE.get_or_init(|| {
        let domain = named::curve25519();
        let map = elligator2_25519();
        Suite::new(
            [
                "curve25519_XMD:SHA-512_ELL2_RO_",
                "curve25519_XMD:SHA-512_ELL2_NU_",
            ],
            domain,
            field_of(&FIELD25519, domain),
            expand_message_xmd::<Sha512>,
            128,
            Mapping::Elligator2(map),
        )
    })
}

/// The suites of edwards25519: `edwards25519_XMD:SHA-512_ELL2_RO_` and
/// `edwards25519_XMD:SHA-512_ELL2_NU_` (RFC 9380, section 8.5), on [`named::ed25519`]: Elligator
/// 2 to curve25519, then RFC 7748's rational map `(v, w) = (c1·s/t, (s − 1)/(s + 1))` with
/// `c1 = sqrt(−486664)` of `sgn0(c1) = 0`, and the cofactor 8 cleared. That sign takes RFC 7748's
/// base point `(9, V)`, with the `V` of its erratum 4730, to edwards25519's base point; the
/// generator of [`named::curve25519`] has the `V` printed before that erratum, `−V`, and goes to
/// the negative of edwards25519's. The first call builds them.
///
/// # Examples
///
/// ```
/// use a::math::ec::hash;
///
/// let suite = hash::edwards25519();
/// assert_eq!(suite.ro_id(), "edwards25519_XMD:SHA-512_ELL2_RO_");
/// assert_eq!(suite.l(), 48);
/// ```
#[must_use]
pub fn edwards25519() -> &'static Suite<Edwards> {
    static SUITE: OnceLock<Suite<Edwards>> = OnceLock::new();
    SUITE.get_or_init(|| {
        let domain = named::ed25519();
        let f = field_of(&FIELD25519, domain);
        let c1 = small(f, -486664)
            .sqrt()
            .unwrap_or_else(|| unreachable!("−486664 is a square modulo 2^255 − 19"));
        let c1 = with_sign(c1, false);
        Suite::new(
            [
                "edwards25519_XMD:SHA-512_ELL2_RO_",
                "edwards25519_XMD:SHA-512_ELL2_NU_",
            ],
            domain,
            f,
            expand_message_xmd::<Sha512>,
            128,
            Mapping::Elligator2Edwards(elligator2_25519(), c1),
        )
    })
}
