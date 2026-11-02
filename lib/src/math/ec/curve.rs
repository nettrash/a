//! What every form of curve offers, and the operations on stored points each form implements.

use super::field::{Base, limbs_of};
use super::group::Table;
use super::point::Point;
use crate::math::numbers::{ByteOrder, Number};
use core::fmt;

/// Why a curve can't be built from its parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveError {
    /// The modulus is not a prime above 3.
    NotPrime,
    /// The curve has a singular point: its discriminant is zero modulo the prime.
    Singular,
    /// The twisted Edwards addition law is not complete: `a` must be a square and `d` not.
    NotComplete,
    /// The generator of a [`Domain`](super::Domain) is not on its curve.
    NotOnCurve,
    /// The order given for a [`Domain`](super::Domain) is not a multiple of its generator's.
    WrongOrder,
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CurveError::NotPrime => "the modulus is not a prime above 3",
            CurveError::Singular => "the curve is singular",
            CurveError::NotComplete => {
                "the addition law is not complete: a must be a square and d not"
            }
            CurveError::NotOnCurve => "the generator is not on the curve",
            CurveError::WrongOrder => "the order times the generator is not the neutral element",
        })
    }
}

impl core::error::Error for CurveError {}

/// The error of [`Curve::point`] for coordinates that don't satisfy the curve's equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotOnCurve;

impl fmt::Display for NotOnCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the point is not on the curve")
    }
}

impl core::error::Error for NotOnCurve {}

/// The operations on points held as the limbs of their coordinates in the field's form, which each
/// form compiles for each field. Not nameable outside the crate, which seals [`Curve`].
pub trait Arithmetic: Sized {
    /// The coordinates a point is stored with.
    const COORDINATES: usize;

    /// The field.
    fn base(&self) -> &Base;
    /// Whether the two are the same curve.
    fn same_curve(&self, other: &Self) -> bool;
    /// Whether `(x, y)` satisfies the curve's equation.
    fn satisfied(&self, x: &Number, y: &Number) -> bool;
    /// The point with the given coordinate, whose other coordinate has the parity `odd`.
    fn lift_limbs(&self, coordinate: &Number, odd: bool) -> Option<Vec<u64>>;
    /// The bytes of the point with the affine coordinates `xy`.
    fn encode(&self, xy: Option<(&Number, &Number)>, compressed: bool) -> Vec<u8>;
    /// The point the bytes stand for.
    fn decode(&self, bytes: &[u8]) -> Option<Vec<u64>>;

    /// The neutral element.
    fn identity_limbs(&self) -> Vec<u64>;
    /// The point `(x, y)`.
    fn affine_limbs(&self, x: &Number, y: &Number) -> Vec<u64>;
    /// Whether `p` is the neutral element.
    fn is_identity_limbs(&self, p: &[u64]) -> bool;
    /// `p += q`.
    fn add_limbs(&self, p: &mut [u64], q: &[u64]);
    /// `p -= q`.
    fn sub_limbs(&self, p: &mut [u64], q: &[u64]);
    /// `p = 2 p`.
    fn double_limbs(&self, p: &mut [u64]);
    /// `p = -p`.
    fn neg_limbs(&self, p: &mut [u64]);
    /// `p = k p`, for the magnitude `k` with the sign `negative`.
    fn mul_limbs(&self, p: &mut [u64], k: &[u64], negative: bool);
    /// `r` is the sum of the terms, each a magnitude, its sign and a point.
    fn multi_limbs(&self, r: &mut [u64], terms: &[(&[u64], bool, &[u64])]);
    /// Whether `p` and `q` are the same point.
    fn equal_limbs(&self, p: &[u64], q: &[u64]) -> bool;
    /// The affine coordinates of each point, `None` for one at infinity.
    fn coordinates_limbs(&self, p: &[&[u64]]) -> Vec<Option<(Number, Number)>>;
    /// The table of `p` for scalars of up to `bits` bits.
    fn prepare_limbs(&self, p: &[u64], bits: usize) -> Table;
    /// `r = k p` from the table of `p`.
    fn prepared_limbs(&self, r: &mut [u64], table: &Table, k: &[u64], negative: bool);
}

/// Implements the operations of [`Arithmetic`] that all forms write the same way, given the macro
/// that binds a curve's group.
macro_rules! arithmetic {
    ($with:ident) => {
        fn identity_limbs(&self) -> Vec<u64> {
            $with!(self, g => {
                let mut r = vec![0; Self::COORDINATES * g.field().limbs()];
                g.store(&g.identity(), &mut r);
                r
            })
        }

        fn affine_limbs(&self, x: &Number, y: &Number) -> Vec<u64> {
            $with!(self, g => {
                let f = g.field();
                let mut r = vec![0; Self::COORDINATES * f.limbs()];
                g.store(&g.of_affine(&f.enter(x), &f.enter(y)), &mut r);
                r
            })
        }

        fn is_identity_limbs(&self, p: &[u64]) -> bool {
            $with!(self, g => g.is_identity(&g.load(p)))
        }

        fn add_limbs(&self, p: &mut [u64], q: &[u64]) {
            $with!(self, g => {
                let s = g.add(&g.load(p), &g.load(q));
                g.store(&s, p)
            })
        }

        fn sub_limbs(&self, p: &mut [u64], q: &[u64]) {
            $with!(self, g => {
                let s = g.add(&g.load(p), &g.neg(&g.load(q)));
                g.store(&s, p)
            })
        }

        fn double_limbs(&self, p: &mut [u64]) {
            $with!(self, g => {
                let s = g.double(&g.load(p));
                g.store(&s, p)
            })
        }

        fn neg_limbs(&self, p: &mut [u64]) {
            $with!(self, g => {
                let s = g.neg(&g.load(p));
                g.store(&s, p)
            })
        }

        fn mul_limbs(&self, p: &mut [u64], k: &[u64], negative: bool) {
            $with!(self, g => {
                let s = Group::mul(&g, &g.load(p), k, negative);
                g.store(&s, p)
            })
        }

        fn multi_limbs(&self, r: &mut [u64], terms: &[(&[u64], bool, &[u64])]) {
            $with!(self, g => {
                let loaded: Vec<_> = terms
                    .iter()
                    .map(|&(k, negative, p)| (k, negative, g.load(p)))
                    .collect();
                g.store(&$crate::math::ec::group::multi_mul(&g, &loaded), r)
            })
        }

        fn equal_limbs(&self, p: &[u64], q: &[u64]) -> bool {
            $with!(self, g => g.equal(&g.load(p), &g.load(q)))
        }

        fn coordinates_limbs(&self, p: &[&[u64]]) -> Vec<Option<(Number, Number)>> {
            $with!(self, g => {
                let f = g.field();
                let loaded: Vec<_> = p.iter().map(|q| g.load(q)).collect();
                g.affine_all(&loaded)
                    .iter()
                    .map(|a| a.as_ref().map(|(x, y)| (f.leave(x), f.leave(y))))
                    .collect()
            })
        }

        fn prepare_limbs(&self, p: &[u64], bits: usize) -> $crate::math::ec::group::Table {
            $with!(self, g => $crate::math::ec::group::prepare(&g, &g.load(p), bits))
        }

        fn prepared_limbs(
            &self,
            r: &mut [u64],
            table: &$crate::math::ec::group::Table,
            k: &[u64],
            negative: bool,
        ) {
            $with!(self, g => {
                g.store(&$crate::math::ec::group::prepared_mul(&g, table, k, negative), r)
            })
        }
    };
}
pub(super) use arithmetic;

/// An elliptic curve over a prime field: [`Weierstrass`](super::Weierstrass),
/// [`Montgomery`](super::Montgomery) or [`Edwards`](super::Edwards). The trait is sealed.
///
/// Numbers come back in the byte order of the curve's prime, reduced below it.
pub trait Curve: Arithmetic {
    /// The discriminant of the curve modulo `p`, not zero as the curve is not singular.
    fn discriminant(&self) -> Number;

    /// The `j`-invariant modulo `p`: two curves with the same one are isomorphic over the
    /// algebraic closure of the field.
    fn j_invariant(&self) -> Number;

    /// The prime of the field.
    fn modulus(&self) -> &Number {
        &self.base().p
    }

    /// The neutral element of the curve's group: the point at infinity, or `(0, 1)` on an Edwards
    /// curve.
    fn identity(&self) -> Point<'_, Self> {
        Point::from_limbs(self, self.identity_limbs())
    }

    /// The point `(x, y)`, reduced modulo `p`.
    ///
    /// # Errors
    ///
    /// [`NotOnCurve`] when it doesn't satisfy the curve's equation.
    fn point(&self, x: &Number, y: &Number) -> Result<Point<'_, Self>, NotOnCurve> {
        if !self.satisfied(x, y) {
            return Err(NotOnCurve);
        }
        Ok(Point::from_limbs(self, self.affine_limbs(x, y)))
    }

    /// The point with the given coordinate, reduced modulo `p`, whose other coordinate has the
    /// parity `odd`: `x` is given and `y` computed on Weierstrass and Montgomery curves, and the
    /// other way round on Edwards curves, or `None` when there is no such point.
    fn lift(&self, coordinate: &Number, odd: bool) -> Option<Point<'_, Self>> {
        self.lift_limbs(coordinate, odd)
            .map(|l| Point::from_limbs(self, l))
    }

    /// The point in the bytes of [`Point::compress`] or [`Point::to_bytes`], or `None` when they
    /// are not the encoding of a point of this curve.
    fn point_from_bytes(&self, bytes: &[u8]) -> Option<Point<'_, Self>> {
        self.decode(bytes).map(|l| Point::from_limbs(self, l))
    }

    /// The sum of `k p` over the terms: Straus's method for a few, Pippenger's for many, much
    /// faster than as many products.
    ///
    /// # Panics
    ///
    /// Panics if a point is not on this curve.
    fn multi_mul<'c>(&'c self, terms: &[(&Number, &Point<'c, Self>)]) -> Point<'c, Self> {
        let scalars: Vec<Vec<u64>> = terms.iter().map(|t| limbs_of(t.0)).collect();
        let mut stored = Vec::with_capacity(terms.len());
        for (k, t) in scalars.iter().zip(terms.iter()) {
            t.1.check_curve(self);
            stored.push((&k[..], t.0.is_negative(), &t.1.limbs[..]));
        }
        let mut r = self.identity_limbs();
        self.multi_limbs(&mut r, &stored);
        Point::from_limbs(self, r)
    }
}

/// `y` or `-y`, whichever has the parity `odd`, or `None` for `y = 0` and an odd parity.
pub(super) fn choose_root<F: super::field::Field>(f: &F, y: F::E, odd: bool) -> Option<F::E> {
    if f.leave(&y).bit(0) == odd {
        return Some(y);
    }
    let y = f.neg(&y);
    (f.leave(&y).bit(0) == odd).then_some(y)
}

/// The `len` bytes of `x` in big-endian order.
pub(super) fn fixed_be(x: &Number, len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    for (i, l) in limbs_of(x).iter().enumerate() {
        for (j, b) in l.to_le_bytes().iter().enumerate() {
            let at = 8 * i + j;
            if at < len {
                out[len - 1 - at] = *b;
            }
        }
    }
    out
}

/// The encoding of SEC 1: `0` for the point at infinity, `2` or `3`, by the parity of `y`, then
/// `x`, or `4` then `x` and `y`, in big-endian order and the `len` bytes of the field.
pub(super) fn sec1_encode(len: usize, xy: Option<(&Number, &Number)>, compressed: bool) -> Vec<u8> {
    let Some((x, y)) = xy else {
        return vec![0];
    };
    let mut out = Vec::with_capacity(1 + 2 * len);
    if compressed {
        out.push(2 + u8::from(y.bit(0)));
        out.extend_from_slice(&fixed_be(x, len));
    } else {
        out.push(4);
        out.extend_from_slice(&fixed_be(x, len));
        out.extend_from_slice(&fixed_be(y, len));
    }
    out
}

/// The point in the encoding of SEC 1, on `curve`, if the bytes are one of a point on it.
pub(super) fn sec1_decode<C: Arithmetic>(curve: &C, bytes: &[u8]) -> Option<Vec<u64>> {
    let len = curve.base().bytes();
    let (&tag, rest) = bytes.split_first()?;
    let order = curve.base().p.byte_order;
    let number = |b: &[u8]| {
        let mut n = Number::new(b.to_vec(), ByteOrder::BigEndian);
        if order == ByteOrder::LittleEndian {
            n.value.reverse();
            n.byte_order = order;
        }
        (n < curve.base().p).then_some(n)
    };
    match tag {
        0 if rest.is_empty() => Some(curve.identity_limbs()),
        2 | 3 if rest.len() == len => curve.lift_limbs(&number(rest)?, tag == 3),
        4 if rest.len() == 2 * len => {
            let (x, y) = (number(&rest[..len])?, number(&rest[len..])?);
            curve.satisfied(&x, &y).then(|| curve.affine_limbs(&x, &y))
        }
        _ => None,
    }
}
