//! Elliptic curves over binary fields: `y² + x·y = x³ + a·x² + b` over `GF(2ᵐ)`, `b ≠ 0`, with
//! the curves of SEC 2 in [`named`].
//!
//! Every ordinary elliptic curve over `GF(2ᵐ)`, one whose `j`-invariant is not 0, has an equation of
//! this form; the supersingular ones, `y² + c·y = x³ + …`, are not covered. The coordinates are
//! elements of a [`BinaryField`] in its polynomial basis, as SEC 1 and the curves of SEC 2 and
//! FIPS 186 give them.
//!
//! * [`BinaryCurve`]: a curve, built from its coefficients `a` and `b`, with the points from their
//!   coordinates, from `x` and one bit of `y`, or from their encodings in SEC 1; the sum of many
//!   products; and the `x`-coordinate of a product from `x` alone, by the Montgomery ladder of
//!   López and Dahab.
//! * [`BinaryPoint`]: a point of a curve, which it borrows, held in López–Dahab coordinates, with
//!   the operators `+ - *` and their assignments, and the encodings of SEC 1: compressed,
//!   uncompressed and hybrid.
//! * [`BinaryDomain`]: a curve with a generator, its order and its cofactor, as the curves of
//!   [`named`] are.
//!
//! Products by a scalar take the width-`w` non-adjacent form and mixed additions; on Koblitz
//! curves, `a ∈ {0, 1}` and `b = 1`, the τ-adic non-adjacent form of Solinas, whose Frobenius maps,
//! three squares each, take the place of the doublings.
//!
//! **Nothing here is constant-time**: the times of the operations depend on the scalars and on the
//! coordinates, so they are not meant for secret scalars such as private keys.
//!
//! # Examples
//!
//! ```
//! use a::math::ec::binary::named;
//! use a::math::numbers::Number;
//!
//! let curve = named::sect163k1(); // NIST's K-163
//! let g = curve.generator();
//! let k = Number::from(0x1234_5678_u32);
//! let p = curve.mul_generator(&k);
//! assert_eq!(p, &g * &k);
//! assert!((&g * curve.order()).is_identity());
//! // SEC 1's compressed form, decompressed by the half-trace.
//! let bytes = p.compress();
//! assert_eq!(bytes.len(), 22);
//! assert_eq!(curve.point_from_bytes(&bytes).unwrap(), p);
//! ```

mod arith;
mod domain;
mod koblitz;
mod mul;
pub mod named;
mod point;

pub use domain::BinaryDomain;
pub use point::{BinaryPoint, BinaryPrepared};

use super::{CurveError, NotOnCurve};
use crate::math::field::{BinaryField, F2m};
use crate::math::numbers::Number;
use arith::{Affine, Ld, Ops};
use koblitz::Frobenius;
use std::sync::OnceLock;

/// From this many terms [`BinaryCurve::multi_mul`] takes Pippenger's method instead of Straus's,
/// as the curves over prime fields do.
const PIPPENGER: usize = 64;

/// The forms of SEC 1's encodings of a point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// `2` or `3` by `ỹ`, then `x`.
    Compressed,
    /// `4`, then `x` and `y`.
    Uncompressed,
    /// `6` or `7` by `ỹ`, then `x` and `y`.
    Hybrid,
}

/// The curve `y² + x·y = x³ + a·x² + b` over a binary field `GF(2ᵐ)`, `b ≠ 0`.
///
/// Its points, [`BinaryPoint`]s, are held in López–Dahab coordinates, `x = X / Z`, `y = Y / Z²`,
/// so that the group's operations take no inversion. The coefficients and coordinates are
/// [`F2m`] elements of the curve's field; an element of another field equal to it is accepted, and
/// one of a different field panics, as the field's own operations do.
///
/// # Examples
///
/// ```
/// use a::math::ec::BinaryCurve;
/// use a::math::field::BinaryField;
/// use a::math::numbers::Number;
///
/// // y² + x·y = x³ + x² + 1 over GF(2⁵) = GF(2)[x] / (x⁵ + x² + 1), a Koblitz curve.
/// let f = BinaryField::new(&[5, 2, 0]);
/// let curve = BinaryCurve::new(&f.one(), &f.one()).unwrap();
/// assert!(curve.is_koblitz());
/// let n = curve.koblitz_order().unwrap(); // 2⁵ + 1 − V₅ = 33 − 11
/// assert_eq!(n, 22u8);
/// let p = curve.lift(&f.element(&[0b110]), false).unwrap(); // x² + x
/// assert!(!(&p * &Number::from(11u8)).is_identity());
/// assert!((&p * &n).is_identity());
/// assert_eq!(-&p, &p * &(&n - 1u8));
/// ```
#[derive(Clone, Debug)]
pub struct BinaryCurve {
    /// The field.
    field: BinaryField,
    /// `a`, as words.
    a: Vec<u64>,
    /// `b`, as words.
    b: Vec<u64>,
    /// `Some(a == 1)` when the curve is a Koblitz curve: `a ∈ {0, 1}` and `b = 1`.
    koblitz: Option<bool>,
    /// What its products need, once one is asked for.
    frobenius: OnceLock<Frobenius>,
    /// An element of trace 1, for the quadratic equations of an even degree, once one is needed.
    trace_one: OnceLock<Vec<u64>>,
}

impl BinaryCurve {
    /// The curve `y² + x·y = x³ + a·x² + b` over the field of `a` and `b`.
    ///
    /// # Errors
    ///
    /// [`CurveError::Singular`] when `b = 0`: the discriminant of the curve is `b`.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` are elements of different fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::{BinaryCurve, CurveError};
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[163, 7, 6, 3, 0]);
    /// assert!(BinaryCurve::new(&f.one(), &f.element(&[0x1234])).is_ok());
    /// assert_eq!(BinaryCurve::new(&f.one(), &f.zero()).unwrap_err(), CurveError::Singular);
    /// ```
    #[track_caller]
    pub fn new(a: &F2m<'_>, b: &F2m<'_>) -> Result<BinaryCurve, CurveError> {
        assert!(a.field() == b.field(), "elements of different fields");
        if b.is_zero() {
            return Err(CurveError::Singular);
        }
        let koblitz = (b.is_one() && (a.is_zero() || a.is_one())).then(|| a.is_one());
        Ok(BinaryCurve {
            field: a.field().clone(),
            a: a.words().to_vec(),
            b: b.words().to_vec(),
            koblitz,
            frobenius: OnceLock::new(),
            trace_one: OnceLock::new(),
        })
    }

    /// The field of the curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::BinaryCurve;
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[7, 1, 0]);
    /// let curve = BinaryCurve::new(&f.element(&[3]), &f.element(&[5])).unwrap();
    /// assert_eq!(curve.field(), &f);
    /// assert_eq!((curve.a().words(), curve.b().words()), (&[3][..], &[5][..]));
    /// ```
    pub fn field(&self) -> &BinaryField {
        &self.field
    }

    /// The coefficient `a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let b163 = named::b163(); // a = 1
    /// assert!(b163.a().is_one() && !b163.b().is_one());
    /// assert_eq!(b163.discriminant(), b163.b());
    /// ```
    pub fn a(&self) -> F2m<'_> {
        self.field.element(&self.a)
    }

    /// The coefficient `b`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let b163 = named::b163(); // a = 1
    /// assert!(b163.a().is_one() && !b163.b().is_one());
    /// assert_eq!(b163.discriminant(), b163.b());
    /// ```
    pub fn b(&self) -> F2m<'_> {
        self.field.element(&self.b)
    }

    /// The discriminant of the curve, `b`, not zero as the curve is not singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let b163 = named::b163(); // a = 1
    /// assert!(b163.a().is_one() && !b163.b().is_one());
    /// assert_eq!(b163.discriminant(), b163.b());
    /// ```
    pub fn discriminant(&self) -> F2m<'_> {
        self.b()
    }

    /// The `j`-invariant `1 / b`: two curves with the same one are isomorphic over the algebraic
    /// closure of the field, and over the field itself when the traces of their `a`s are equal too.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::BinaryCurve;
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[7, 1, 0]);
    /// let curve = BinaryCurve::new(&f.element(&[3]), &f.element(&[5])).unwrap();
    /// assert_eq!(curve.discriminant(), f.element(&[5]));
    /// assert_eq!(&curve.j_invariant() * &curve.b(), f.one());
    /// ```
    pub fn j_invariant(&self) -> F2m<'_> {
        self.b()
            .inverse()
            .unwrap_or_else(|| unreachable!("b is not zero"))
    }

    /// Whether this is a Koblitz curve, `a ∈ {0, 1}` and `b = 1`, defined over `GF(2)`: its
    /// products take the τ-adic non-adjacent form, and its number of points is known in closed
    /// form, [`koblitz_order`](BinaryCurve::koblitz_order).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// assert!(named::sect233k1().is_koblitz());
    /// assert!(!named::sect233r1().is_koblitz());
    /// ```
    pub fn is_koblitz(&self) -> bool {
        self.koblitz.is_some()
    }

    /// The number of points of a Koblitz curve over `GF(2ᵐ)`, the point at infinity included:
    /// `2ᵐ + 1 − V_m` for the Lucas sequence `V_0 = 2`, `V_1 = μ`, `V_(k+1) = μ·V_k − 2·V_(k−1)`,
    /// `μ = 1` for `a = 1` and `−1` for `a = 0`. `None` for a curve that is not a Koblitz curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let k163 = named::sect163k1();
    /// assert_eq!(k163.koblitz_order().unwrap(), k163.order() * k163.cofactor());
    /// assert_eq!(named::sect163r2().koblitz_order(), None);
    /// ```
    pub fn koblitz_order(&self) -> Option<Number> {
        self.frobenius().map(|f| f.points().clone())
    }

    /// The neutral element of the curve's group, the point at infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect113r1();
    /// let o = curve.identity();
    /// assert!(o.is_identity() && o.coordinates().is_none());
    /// assert_eq!(&curve.generator() + &o, curve.generator());
    /// assert_eq!(o.compress(), [0]);
    /// ```
    pub fn identity(&self) -> BinaryPoint<'_> {
        BinaryPoint::from_ld(self, self.ops().identity())
    }

    /// The point `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`NotOnCurve`] when it doesn't satisfy the curve's equation.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is an element of another field than the curve's.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::BinaryCurve;
    /// use a::math::field::BinaryField;
    ///
    /// let f = BinaryField::new(&[5, 2, 0]);
    /// let curve = BinaryCurve::new(&f.one(), &f.one()).unwrap();
    /// // x = 0 has the one point (0, √b), of order 2.
    /// let t = curve.point(&f.zero(), &f.one()).unwrap();
    /// assert!(t.double().is_identity());
    /// assert!(curve.point(&f.zero(), &f.zero()).is_err());
    /// ```
    #[track_caller]
    pub fn point(&self, x: &F2m<'_>, y: &F2m<'_>) -> Result<BinaryPoint<'_>, NotOnCurve> {
        let (x, y) = (self.adopt(x), self.adopt(y));
        let ops = self.ops();
        if !ops.satisfied(&x, &y) {
            return Err(NotOnCurve);
        }
        Ok(BinaryPoint::from_ld(self, ops.of_affine(&Affine { x, y })))
    }

    /// The point with the coordinate `x` whose `ỹ`, the lowest bit of `y / x` in the polynomial
    /// basis, is `odd`, as SEC 1 (§2.3.4) decompresses a point, or `None` when there is no point
    /// with that `x`. With `z² + z = x + a + b / x²`, solved by the half-trace for an odd `m` and
    /// by IEEE 1363's method for an even one, `y = x·z`. For `x = 0` it is the one point
    /// `(0, √b)`, whatever `odd` is, as SEC 1 and OpenSSL decompress it, though its `ỹ` is 0.
    ///
    /// # Panics
    ///
    /// Panics if `x` is an element of another field than the curve's.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect233r1();
    /// let g = curve.generator();
    /// let (x, y) = g.coordinates().unwrap();
    /// let odd = (&y / &x).words()[0] & 1 == 1;
    /// assert_eq!(curve.lift(&x, odd).unwrap(), g);
    /// assert_eq!(curve.lift(&x, !odd).unwrap(), -&g);
    /// ```
    #[track_caller]
    pub fn lift(&self, x: &F2m<'_>, odd: bool) -> Option<BinaryPoint<'_>> {
        let x = self.adopt(x);
        self.lift_ld(x, odd).map(|p| BinaryPoint::from_ld(self, p))
    }

    /// The point in the bytes of [`BinaryPoint::compress`], [`BinaryPoint::to_bytes`] or
    /// [`BinaryPoint::to_bytes_hybrid`], SEC 1's encodings, or `None` when they are not the
    /// encoding of a point of this curve: a wrong length or first byte, a coordinate with bits at
    /// or above `m`, a point off the curve, a compressed `x` with no point, or a hybrid form whose
    /// `ỹ` is not that of its `y`, which is 0 for `x = 0`. A compressed `x = 0` is the point
    /// `(0, √b)` with either first byte, as in [`lift`](BinaryCurve::lift).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect113r1();
    /// let g = curve.generator();
    /// for bytes in [g.compress(), g.to_bytes(), g.to_bytes_hybrid()] {
    ///     assert_eq!(curve.point_from_bytes(&bytes).unwrap(), g);
    /// }
    /// assert!(curve.point_from_bytes(&[0]).unwrap().is_identity());
    /// assert!(curve.point_from_bytes(&[5, 1, 2]).is_none());
    /// ```
    pub fn point_from_bytes(&self, bytes: &[u8]) -> Option<BinaryPoint<'_>> {
        self.decode(bytes).map(|p| BinaryPoint::from_ld(self, p))
    }

    /// The sum of `k·p` over the terms: Straus's method for a few, the expansions read together
    /// with one run of doublings (of Frobenius maps on a Koblitz curve) for all of them, and from
    /// 64 terms Pippenger's, which sorts them into buckets for each window of digits.
    ///
    /// # Panics
    ///
    /// Panics if a point is not on this curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let curve = named::sect131r1();
    /// let g = curve.generator();
    /// let p = g.double();
    /// let (a, b) = (Number::from(1_000_003u32), -Number::from(77u8));
    /// assert_eq!(curve.multi_mul(&[(&a, &g), (&b, &p)]), &g * &a + &p * &b);
    /// ```
    #[track_caller]
    pub fn multi_mul<'c>(&'c self, terms: &[(&Number, &BinaryPoint<'c>)]) -> BinaryPoint<'c> {
        for t in terms {
            t.1.check_curve(self);
        }
        let ops = self.ops();
        let r = if terms.len() >= PIPPENGER {
            let signed: Vec<(Vec<u64>, Ld<'c>)> = terms
                .iter()
                .map(|&(k, p)| {
                    let q = if k.is_negative() {
                        ops.neg(&p.p)
                    } else {
                        p.p.clone()
                    };
                    (mul::magnitude(k), q)
                })
                .collect();
            mul::pippenger(&ops, &signed)
        } else if let Some(frobenius) = self.frobenius() {
            let expansions: Vec<_> = terms
                .iter()
                .map(|&(k, p)| frobenius.expansion(&ops, &p.p, k))
                .collect();
            mul::horner(&ops, &expansions, mul::Step::Frobenius)
        } else {
            let expansions: Vec<_> = terms
                .iter()
                .map(|&(k, p)| {
                    let q = if k.is_negative() {
                        ops.neg(&p.p)
                    } else {
                        p.p.clone()
                    };
                    mul::wnaf_expansion(&ops, &q, &mul::magnitude(k))
                })
                .collect();
            mul::horner(&ops, &expansions, mul::Step::Double)
        };
        BinaryPoint::from_ld(self, r)
    }

    /// `x` of `k·P` from `x` of `P` alone, by the Montgomery ladder of López and Dahab, 6M + 5S a
    /// bit of `k`, or `None` when `k·P` is the point at infinity. `x` need not be on this curve: it
    /// may be on the curve's quadratic twist, as each `x` but 0 is on exactly one of the two, and
    /// the ladder needs neither `y` nor `a`. `k` counts by magnitude, as `x` is the same for `k·P`
    /// and `−k·P`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is an element of another field than the curve's.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let curve = named::sect163r2();
    /// let g = curve.generator();
    /// let k = Number::from(987_654_321u32);
    /// let (x, _) = g.coordinates().unwrap();
    /// assert_eq!(curve.x_mul(&x, &k), (&g * &k).coordinates().map(|c| c.0));
    /// assert_eq!(curve.x_mul(&x, curve.order()), None);
    /// ```
    #[track_caller]
    pub fn x_mul(&self, x: &F2m<'_>, k: &Number) -> Option<F2m<'_>> {
        let x = self.adopt(x);
        mul::x_ladder(&self.ops(), &x, &mul::magnitude(k))
    }

    /// The arithmetic of the curve's points.
    fn ops(&self) -> Ops<'_> {
        Ops::new(&self.field, &self.a, &self.b)
    }

    /// What the products of a Koblitz curve need, `None` for another curve.
    fn frobenius(&self) -> Option<&Frobenius> {
        let a_is_one = self.koblitz?;
        Some(
            self.frobenius
                .get_or_init(|| Frobenius::new(self.field.degree(), a_is_one)),
        )
    }

    /// `k·p`, by the τ-adic form on a Koblitz curve and the non-adjacent form otherwise.
    fn product<'c>(&'c self, p: &Ld<'c>, k: &Number) -> Ld<'c> {
        let ops = self.ops();
        if let Some(frobenius) = self.frobenius() {
            return frobenius.mul(&ops, p, k);
        }
        let r = mul::mul(&ops, p, &mul::magnitude(k));
        if k.is_negative() { ops.neg(&r) } else { r }
    }

    /// Whether the two are the same curve.
    fn same_curve(&self, other: &BinaryCurve) -> bool {
        core::ptr::eq(self, other)
            || (self.field == other.field && self.a == other.a && self.b == other.b)
    }

    /// The element `x` of an equal field, as an element of the curve's.
    #[track_caller]
    fn adopt(&self, x: &F2m<'_>) -> F2m<'_> {
        assert!(x.field() == &self.field, "elements of different fields");
        self.field.element(x.words())
    }

    /// A `z` with `z² + z = c`, or `None` when there is none, which is when the trace of `c` is 1.
    /// For an odd degree the half-trace of `c` solves it; for an even one, IEEE 1363-2000's
    /// method (A.4.7) with an element `t` of trace 1: `z₀ = 0`, `w₀ = c`, then `m − 1` times
    /// `z ← z² + w²·t` and `w ← w² + c`, which leaves `w` the trace of `c` and, when it is 0,
    /// `z² + z = c`.
    fn solve_quadratic<'c>(&'c self, c: &F2m<'c>) -> Option<F2m<'c>> {
        let m = self.field.degree();
        if m % 2 == 1 {
            // `z² + z = c + Tr(c)` for the half-trace `z`.
            let z = c.half_trace();
            return (&(&z.square() + &z) == c).then_some(z);
        }
        let t = self.field.element(self.trace_one());
        let mut z = self.field.zero();
        let mut w = c.clone();
        for _ in 1..m {
            z = &z.square() + &(&w.square() * &t);
            w = &w.square() + c;
        }
        w.is_zero().then_some(z)
    }

    /// An element of trace 1, the first power `xⁱ` that has it: there is one, as the trace is a
    /// linear map onto `GF(2)` and the powers below `m` a basis.
    fn trace_one(&self) -> &[u64] {
        self.trace_one.get_or_init(|| {
            let m = self.field.degree();
            (0..m)
                .map(|i| {
                    let mut words = vec![0u64; (m as usize).div_ceil(64)];
                    words[i as usize / 64] = 1 << (i % 64);
                    words
                })
                .find(|w| self.field.element(w).trace() == 1)
                .unwrap_or_else(|| unreachable!("the trace is not zero on a basis"))
        })
    }

    /// The point with the coordinate `x` and the bit `ỹ = odd`, if there is one.
    fn lift_ld<'c>(&'c self, x: F2m<'c>, odd: bool) -> Option<Ld<'c>> {
        let ops = self.ops();
        if x.is_zero() {
            // The point (0, √b): SEC 1 does not look at `ỹ` for it.
            let y = ops.b.sqrt();
            return Some(ops.of_affine(&Affine { x, y }));
        }
        let inverse = x
            .square()
            .inverse()
            .unwrap_or_else(|| unreachable!("x is not zero"));
        let beta = &(&x + &ops.a) + &(&ops.b * &inverse);
        let mut z = self.solve_quadratic(&beta)?;
        if (z.words()[0] & 1 == 1) != odd {
            z += self.field.one();
        }
        let y = &x * &z;
        Some(ops.of_affine(&Affine { x, y }))
    }

    /// The bytes an element takes in SEC 1, `⌈m / 8⌉`.
    fn element_bytes(&self) -> usize {
        (self.field.degree() as usize).div_ceil(8)
    }

    /// The element in the big-endian bytes, `None` when it has bits at or above `m`.
    fn element_of(&self, bytes: &[u8]) -> Option<F2m<'_>> {
        let m = self.field.degree() as usize;
        let mut words = vec![0u64; m.div_ceil(64)];
        for (i, &byte) in bytes.iter().rev().enumerate() {
            for j in 0..8 {
                if byte >> j & 1 == 1 {
                    let at = 8 * i + j;
                    if at >= m {
                        return None;
                    }
                    words[at / 64] |= 1 << (at % 64);
                }
            }
        }
        Some(self.field.element(&words))
    }

    /// The bytes of `x` in big-endian order, `⌈m / 8⌉` of them.
    fn bytes_of(&self, x: &F2m<'_>) -> Vec<u8> {
        let len = self.element_bytes();
        let mut out = vec![0; len];
        for (i, w) in x.words().iter().enumerate() {
            for (j, b) in w.to_le_bytes().iter().enumerate() {
                let at = 8 * i + j;
                if at < len {
                    out[len - 1 - at] = *b;
                }
            }
        }
        out
    }

    /// `ỹ`: the lowest bit of `y / x`, 0 for `x = 0`.
    fn y_bit(&self, p: &Affine<'_>) -> bool {
        match p.x.inverse() {
            Some(i) => (&p.y * &i).words()[0] & 1 == 1,
            None => false,
        }
    }

    /// The encoding of SEC 1 (§2.3.3) of the affine point `p`, a single zero byte for the point at
    /// infinity.
    fn encode(&self, p: Option<&Affine<'_>>, form: Form) -> Vec<u8> {
        let Some(p) = p else {
            return vec![0];
        };
        let tag = match form {
            Form::Compressed => 2 + u8::from(self.y_bit(p)),
            Form::Uncompressed => 4,
            Form::Hybrid => 6 + u8::from(self.y_bit(p)),
        };
        let mut out = vec![tag];
        out.extend(self.bytes_of(&p.x));
        if form != Form::Compressed {
            out.extend(self.bytes_of(&p.y));
        }
        out
    }

    /// The point in the encoding of SEC 1 (§2.3.4), if the bytes are one of a point of the curve.
    fn decode(&self, bytes: &[u8]) -> Option<Ld<'_>> {
        let len = self.element_bytes();
        let (&tag, rest) = bytes.split_first()?;
        match tag {
            0 if rest.is_empty() => Some(self.ops().identity()),
            2 | 3 if rest.len() == len => self.lift_ld(self.element_of(rest)?, tag == 3),
            4 | 6 | 7 if rest.len() == 2 * len => {
                let p = Affine {
                    x: self.element_of(&rest[..len])?,
                    y: self.element_of(&rest[len..])?,
                };
                let ops = self.ops();
                let hybrid_fits = tag == 4 || self.y_bit(&p) == (tag == 7);
                (hybrid_fits && ops.satisfied(&p.x, &p.y)).then(|| ops.of_affine(&p))
            }
            _ => None,
        }
    }
}
