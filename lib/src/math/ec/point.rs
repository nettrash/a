//! Points of a curve.

use super::curve::Curve;
use super::field::limbs_of;
use super::group::Table;
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A point of the curve `C`, which it borrows: always on the curve, as every way to get one
/// checks.
///
/// It is held in a projective form in the field's own representation, so the group's operations
/// take no inversion, and [`coordinates`](Point::coordinates) makes the one inversion they need
/// when they are asked for. Every operator takes the points owned or borrowed; `+=`, `-=` and
/// `*=`, and the operators on an owned left operand, keep its vector.
///
/// Scalars are numbers of any sign and length. **Nothing here is constant-time**: the times of the
/// additions and doublings, and of a product, depend on the bits of the scalar, so it is not meant
/// for secret scalars, such as private keys, where whoever can time it could learn them.
///
/// # Panics
///
/// The operators panic for two points of different curves.
///
/// # Examples
///
/// ```
/// use a::math::ec::{Curve, Weierstrass};
/// use a::math::numbers::{ByteOrder, Number};
///
/// let n = |x| Number::new_from_u32(x, ByteOrder::BigEndian);
/// let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
/// let p = curve.point(&n(3), &n(6)).unwrap();
///
/// // The point has order 5.
/// assert_eq!(&p * &n(2) + &p * &n(3), curve.identity());
/// assert!((&p * &n(5)).is_identity());
/// assert_eq!(-&p, &p * &n(4));
/// ```
pub struct Point<'c, C: Curve> {
    /// The curve.
    curve: &'c C,
    /// The coordinates in the field's form, one after another.
    pub(super) limbs: Vec<u64>,
}

impl<'c, C: Curve> Point<'c, C> {
    /// The point on `curve` with the stored coordinates `limbs`.
    #[inline]
    pub(super) fn from_limbs(curve: &'c C, limbs: Vec<u64>) -> Self {
        Point { curve, limbs }
    }

    /// Panics unless the point is on `curve`.
    #[inline(always)]
    #[track_caller]
    pub(super) fn check_curve(&self, curve: &C) {
        if !core::ptr::eq(self.curve, curve) && !self.curve.same_curve(curve) {
            different_curves();
        }
    }

    /// The curve of the point.
    #[inline]
    pub fn curve(&self) -> &'c C {
        self.curve
    }

    /// Whether this is the neutral element of the curve's group.
    pub fn is_identity(&self) -> bool {
        self.curve.is_identity_limbs(&self.limbs)
    }

    /// The affine coordinates `(x, y)`, below the prime, or `None` for the point at infinity. An
    /// Edwards curve has none: its neutral element is `(0, 1)`.
    ///
    /// This is the one inversion that the group's operations leave for the end. For many points,
    /// [`coordinates_of`](Point::coordinates_of) shares it.
    pub fn coordinates(&self) -> Option<(Number, Number)> {
        self.curve
            .coordinates_limbs(&[&self.limbs])
            .pop()
            .unwrap_or(None)
    }

    /// The affine coordinates of each of the points, as
    /// [`coordinates`](Point::coordinates) gives them, with one inversion for all of them and
    /// about three products for each.
    ///
    /// # Panics
    ///
    /// Panics for points of different curves.
    pub fn coordinates_of(points: &[Point<'c, C>]) -> Vec<Option<(Number, Number)>> {
        let Some(first) = points.first() else {
            return Vec::new();
        };
        let stored: Vec<&[u64]> = points
            .iter()
            .map(|p| {
                p.check_curve(first.curve);
                &p.limbs[..]
            })
            .collect();
        first.curve.coordinates_limbs(&stored)
    }

    /// `2 p`, which costs less than `p + p`.
    pub fn double(&self) -> Self {
        let mut r = self.clone();
        self.curve.double_limbs(&mut r.limbs);
        r
    }

    /// The point in its shortest encoding, which [`Curve::point_from_bytes`] reads: SEC 1's compressed
    /// form on Weierstrass and Montgomery curves, a byte for the parity of `y` and then `x` in
    /// big-endian order, a single zero byte for the point at infinity; on Edwards curves that of
    /// RFC 8032, `y` in little-endian order, with the parity of `x` in the top bit.
    pub fn compress(&self) -> Vec<u8> {
        let xy = self.coordinates();
        self.curve.encode(xy.as_ref().map(|(x, y)| (x, y)), true)
    }

    /// The point in the encoding with both coordinates, which [`Curve::point_from_bytes`] reads: SEC 1's
    /// uncompressed form on Weierstrass and Montgomery curves, on Edwards curves `x` and then `y`
    /// in little-endian order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let xy = self.coordinates();
        self.curve.encode(xy.as_ref().map(|(x, y)| (x, y)), false)
    }

    /// The multiples of this point computed once for all, for scalars of up to `bits` bits, as a
    /// [`Divisor`](crate::math::numbers::Divisor) is for divisions. Measured on an Apple M4 Max on
    /// P-256, building the table for 256 bits took the time of 30 products, and each product
    /// afterwards an eighth of the time of one of [`Mul`].
    pub fn prepare(&self, bits: usize) -> Prepared<'c, C> {
        Prepared {
            curve: self.curve,
            table: self.curve.prepare_limbs(&self.limbs, bits),
            point: self.limbs.clone(),
        }
    }
}

/// Panics for points of different curves.
#[cold]
#[inline(never)]
#[track_caller]
fn different_curves() -> ! {
    panic!("the points are on different curves")
}

/// The multiples of a point computed once, from [`Point::prepare`].
pub struct Prepared<'c, C: Curve> {
    /// The curve.
    curve: &'c C,
    /// The multiples.
    table: Table,
    /// The point, for scalars longer than the table.
    point: Vec<u64>,
}

impl<'c, C: Curve> Prepared<'c, C> {
    /// `k` times the point, from the table when `k` is not longer than what it was built for.
    pub fn mul(&self, k: &Number) -> Point<'c, C> {
        let scalar = limbs_of(k);
        let mut limbs = self.point.clone();
        if k.bits() as usize <= self.table.bits {
            self.curve
                .prepared_limbs(&mut limbs, &self.table, &scalar, k.is_negative());
        } else {
            self.curve.mul_limbs(&mut limbs, &scalar, k.is_negative());
        }
        Point::from_limbs(self.curve, limbs)
    }
}

impl<C: Curve> Clone for Point<'_, C> {
    fn clone(&self) -> Self {
        Point {
            curve: self.curve,
            limbs: self.limbs.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.curve = source.curve;
        self.limbs.clone_from(&source.limbs);
    }
}

/// Two points are equal when they are the same point, in whatever projective form each is held.
/// This needs no inversion.
impl<C: Curve> PartialEq for Point<'_, C> {
    #[track_caller]
    fn eq(&self, other: &Self) -> bool {
        other.check_curve(self.curve);
        self.curve.equal_limbs(&self.limbs, &other.limbs)
    }
}

impl<C: Curve> Eq for Point<'_, C> {}

/// By the affine coordinates, the same for equal points: each hash takes an inversion.
impl<C: Curve> Hash for Point<'_, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coordinates().hash(state);
    }
}

impl<C: Curve> fmt::Debug for Point<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `(x, y)` in decimal, or `infinity`.
impl<C: Curve> fmt::Display for Point<'_, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.coordinates() {
            Some((x, y)) => write!(f, "({x}, {y})"),
            None => f.write_str("infinity"),
        }
    }
}

impl<'c, C: Curve> Neg for &Point<'c, C> {
    type Output = Point<'c, C>;

    fn neg(self) -> Point<'c, C> {
        -self.clone()
    }
}

impl<'c, C: Curve> Neg for Point<'c, C> {
    type Output = Point<'c, C>;

    fn neg(mut self) -> Point<'c, C> {
        self.curve.neg_limbs(&mut self.limbs);
        self
    }
}

/// Implements a binary operator for every mix of owned and borrowed operands, from its assignment
/// on the left operand's coordinates, reusing an owned left operand's vector.
macro_rules! operators {
    ($trait:ident, $method:ident, $assign:ident, $assign_method:ident, $limbs:ident) => {
        impl<'c, C: Curve> $assign<&Point<'c, C>> for Point<'c, C> {
            #[track_caller]
            fn $assign_method(&mut self, rhs: &Point<'c, C>) {
                rhs.check_curve(self.curve);
                self.curve.$limbs(&mut self.limbs, &rhs.limbs);
            }
        }

        impl<'c, C: Curve> $assign<Point<'c, C>> for Point<'c, C> {
            #[track_caller]
            fn $assign_method(&mut self, rhs: Point<'c, C>) {
                self.$assign_method(&rhs);
            }
        }

        impl<'c, C: Curve> $trait<&Point<'c, C>> for Point<'c, C> {
            type Output = Point<'c, C>;

            #[track_caller]
            fn $method(mut self, rhs: &Point<'c, C>) -> Point<'c, C> {
                self.$assign_method(rhs);
                self
            }
        }

        impl<'c, C: Curve> $trait<Point<'c, C>> for Point<'c, C> {
            type Output = Point<'c, C>;

            #[track_caller]
            fn $method(self, rhs: Point<'c, C>) -> Point<'c, C> {
                self.$method(&rhs)
            }
        }

        impl<'c, C: Curve> $trait<&Point<'c, C>> for &Point<'c, C> {
            type Output = Point<'c, C>;

            #[track_caller]
            fn $method(self, rhs: &Point<'c, C>) -> Point<'c, C> {
                self.clone().$method(rhs)
            }
        }

        impl<'c, C: Curve> $trait<Point<'c, C>> for &Point<'c, C> {
            type Output = Point<'c, C>;

            #[track_caller]
            fn $method(self, rhs: Point<'c, C>) -> Point<'c, C> {
                self.clone().$method(&rhs)
            }
        }
    };
}
operators!(Add, add, AddAssign, add_assign, add_limbs);
operators!(Sub, sub, SubAssign, sub_assign, sub_limbs);

impl<C: Curve> MulAssign<&Number> for Point<'_, C> {
    fn mul_assign(&mut self, k: &Number) {
        self.curve
            .mul_limbs(&mut self.limbs, &limbs_of(k), k.is_negative());
    }
}

impl<C: Curve> MulAssign<Number> for Point<'_, C> {
    fn mul_assign(&mut self, k: Number) {
        *self *= &k;
    }
}

impl<'c, C: Curve> Mul<&Number> for Point<'c, C> {
    type Output = Point<'c, C>;

    fn mul(mut self, k: &Number) -> Point<'c, C> {
        self *= k;
        self
    }
}

impl<'c, C: Curve> Mul<Number> for Point<'c, C> {
    type Output = Point<'c, C>;

    fn mul(self, k: Number) -> Point<'c, C> {
        self * &k
    }
}

impl<'c, C: Curve> Mul<&Number> for &Point<'c, C> {
    type Output = Point<'c, C>;

    fn mul(self, k: &Number) -> Point<'c, C> {
        self.clone() * k
    }
}

impl<'c, C: Curve> Mul<Number> for &Point<'c, C> {
    type Output = Point<'c, C>;

    fn mul(self, k: Number) -> Point<'c, C> {
        self.clone() * &k
    }
}

impl<'c, C: Curve> Mul<&Point<'c, C>> for &Number {
    type Output = Point<'c, C>;

    fn mul(self, p: &Point<'c, C>) -> Point<'c, C> {
        p * self
    }
}

impl<'c, C: Curve> Mul<Point<'c, C>> for &Number {
    type Output = Point<'c, C>;

    fn mul(self, p: Point<'c, C>) -> Point<'c, C> {
        p * self
    }
}

impl<'c, C: Curve> Mul<&Point<'c, C>> for Number {
    type Output = Point<'c, C>;

    fn mul(self, p: &Point<'c, C>) -> Point<'c, C> {
        p * &self
    }
}

impl<'c, C: Curve> Mul<Point<'c, C>> for Number {
    type Output = Point<'c, C>;

    fn mul(self, p: Point<'c, C>) -> Point<'c, C> {
        p * &self
    }
}
