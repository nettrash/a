//! Points of a binary curve.

use super::arith::Ld;
use super::mul::{self, Table};
use super::{BinaryCurve, Form};
use crate::math::field::F2m;
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A point of a [`BinaryCurve`], which it borrows: always on the curve, as every way to get one
/// checks.
///
/// It is held in López–Dahab coordinates, `x = X / Z`, `y = Y / Z²`, so that the group's
/// operations take no inversion, and [`coordinates`](BinaryPoint::coordinates) makes the one they
/// need when they are asked for. Every operator takes the points owned or borrowed.
///
/// Scalars are numbers of any sign and length. **Nothing here is constant-time**: the times of
/// the additions, doublings and products depend on the bits of the scalar and on the
/// coordinates, so it is not meant for secret scalars, such as private keys.
///
/// # Panics
///
/// The operators panic for two points of different curves.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
/// use a::math::numbers::Number;
///
/// let curve = named::sect113r2();
/// let g = curve.generator();
/// let three = Number::from(3u8);
/// assert_eq!(&g * &three, &g + &g.double());
/// assert_eq!(&(&g * &three) - &g, g.double());
/// assert!((&g + &-&g).is_identity());
/// let (x, y) = g.coordinates().unwrap();
/// assert_eq!((-&g).coordinates().unwrap(), (x.clone(), &x + &y));
/// ```
pub struct BinaryPoint<'c> {
    /// The curve.
    curve: &'c BinaryCurve,
    /// The coordinates.
    pub(super) p: Ld<'c>,
}

impl<'c> BinaryPoint<'c> {
    /// The point of `curve` with the López–Dahab coordinates `p`.
    pub(super) fn from_ld(curve: &'c BinaryCurve, p: Ld<'c>) -> Self {
        BinaryPoint { curve, p }
    }

    /// Whether the point is on `curve`: that value, or one of the same curve.
    pub(super) fn belongs_to(&self, curve: &BinaryCurve) -> bool {
        self.curve.same_curve(curve)
    }

    /// Panics unless the point is on `curve`.
    #[track_caller]
    pub(super) fn check_curve(&self, curve: &BinaryCurve) {
        assert!(self.belongs_to(curve), "the points are on different curves");
    }

    /// The curve of the point.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect131r1();
    /// assert!(core::ptr::eq(curve.generator().curve(), &**curve));
    /// ```
    pub fn curve(&self) -> &'c BinaryCurve {
        self.curve
    }

    /// Whether this is the point at infinity, the neutral element of the curve's group.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect131r1();
    /// let g = curve.generator();
    /// assert!(!g.is_identity());
    /// assert!((&g * curve.order()).is_identity());
    /// ```
    pub fn is_identity(&self) -> bool {
        self.p.z.is_zero()
    }

    /// The affine coordinates `(x, y)`, elements of the curve's field, or `None` for the point at
    /// infinity. This is the one inversion that the group's operations leave for the end; for many
    /// points [`coordinates_of`](BinaryPoint::coordinates_of) shares it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect113r2();
    /// let (x, y) = curve.generator().double().coordinates().unwrap();
    /// assert_eq!(curve.point(&x, &y).unwrap(), curve.generator().double());
    /// assert_eq!(curve.identity().coordinates(), None);
    /// ```
    pub fn coordinates(&self) -> Option<(F2m<'c>, F2m<'c>)> {
        self.curve.ops().affine(&self.p).map(|a| (a.x, a.y))
    }

    /// The affine coordinates of each of the points, as [`coordinates`](BinaryPoint::coordinates)
    /// gives them, with one inversion for all of them and five products for each.
    ///
    /// # Panics
    ///
    /// Panics for points of different curves.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::BinaryPoint;
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect131r2();
    /// let g = curve.generator();
    /// let points = [g.clone(), curve.identity(), g.double()];
    /// let all = BinaryPoint::coordinates_of(&points);
    /// assert_eq!(all[0], g.coordinates());
    /// assert_eq!(all[1], None);
    /// assert_eq!(all[2], g.double().coordinates());
    /// ```
    #[track_caller]
    pub fn coordinates_of(points: &[BinaryPoint<'c>]) -> Vec<Option<(F2m<'c>, F2m<'c>)>> {
        let Some(first) = points.first() else {
            return Vec::new();
        };
        let stored: Vec<Ld<'c>> = points
            .iter()
            .map(|p| {
                p.check_curve(first.curve);
                p.p.clone()
            })
            .collect();
        first
            .curve
            .ops()
            .affine_all(&stored)
            .into_iter()
            .map(|a| a.map(|a| (a.x, a.y)))
            .collect()
    }

    /// `2·p`, which costs less than `p + p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let g = named::sect163r1().generator();
    /// assert_eq!(g.double(), &g + &g);
    /// ```
    pub fn double(&self) -> Self {
        BinaryPoint::from_ld(self.curve, self.curve.ops().double(&self.p))
    }

    /// The point in SEC 1's compressed form, which [`BinaryCurve::point_from_bytes`] reads: `2` or
    /// `3` by `ỹ`, the lowest bit of `y / x` (0 for `x = 0`), then `x` in big-endian order in
    /// `⌈m / 8⌉` bytes; a single zero byte for the point at infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// // SEC 2 gives sect113r1's generator compressed as 03 009D73616F35F4AB1407D73562C10F.
    /// let bytes = named::sect113r1().generator().compress();
    /// assert_eq!(bytes[..3], [0x03, 0x00, 0x9D]);
    /// assert_eq!(bytes.len(), 16);
    /// ```
    pub fn compress(&self) -> Vec<u8> {
        self.encoded(Form::Compressed)
    }

    /// The point in SEC 1's uncompressed form, which [`BinaryCurve::point_from_bytes`] reads: `4`,
    /// then `x` and `y` in big-endian order in `⌈m / 8⌉` bytes each; a single zero byte for the
    /// point at infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let bytes = named::sect113r1().generator().to_bytes();
    /// assert_eq!((bytes[0], bytes.len()), (4, 31));
    /// assert_eq!(bytes[16..18], [0x00, 0xA5]); // y = 00A52830277958EE84D1315ED31886
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encoded(Form::Uncompressed)
    }

    /// The point in SEC 1's hybrid form (ANSI X9.62), which [`BinaryCurve::point_from_bytes`]
    /// reads: `6` or `7` by `ỹ`, then `x` and `y`; a single zero byte for the point at infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let g = named::sect113r1().generator();
    /// let (hybrid, compressed) = (g.to_bytes_hybrid(), g.compress());
    /// assert_eq!(hybrid[0], compressed[0] + 4);
    /// assert_eq!(hybrid[1..], g.to_bytes()[1..]);
    /// ```
    pub fn to_bytes_hybrid(&self) -> Vec<u8> {
        self.encoded(Form::Hybrid)
    }

    /// The point in the encoding `form`.
    fn encoded(&self, form: Form) -> Vec<u8> {
        let a = self.curve.ops().affine(&self.p);
        self.curve.encode(a.as_ref(), form)
    }

    /// The multiples of this point computed once for all, for scalars of up to `bits` bits: its
    /// multiples by each digit of a signed window at each position, so that a product takes
    /// additions only, about one for each window of the scalar.
    ///
    /// # Panics
    ///
    /// When the table is too long to allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let curve = named::sect113r1();
    /// let g = curve.generator();
    /// let prepared = g.prepare(64);
    /// let k = Number::from(0xDEAD_BEEF_u32);
    /// assert_eq!(prepared.mul(&k), &g * &k);
    /// assert_eq!(prepared.mul(&-&k), -(&g * &k));
    /// ```
    pub fn prepare(&self, bits: usize) -> BinaryPrepared<'c> {
        BinaryPrepared {
            curve: self.curve,
            table: mul::prepare(&self.curve.ops(), &self.p, bits),
            point: self.p.clone(),
        }
    }
}

/// The multiples of a point computed once, from [`BinaryPoint::prepare`]: for each window of the
/// scalars it was built for, the point's multiples by each digit, so that a product takes one
/// addition a window and no doubling.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
/// use a::math::numbers::Number;
///
/// let curve = named::sect163r2();
/// let g = curve.generator();
/// let prepared = g.prepare(163);
/// for k in [0u64, 1, 2, 0x1234_5678_9ABC_DEF0] {
///     let k = Number::from(k);
///     assert_eq!(prepared.mul(&k), &g * &k);
/// }
/// assert!(prepared.mul(curve.order()).is_identity());
/// ```
pub struct BinaryPrepared<'c> {
    /// The curve.
    curve: &'c BinaryCurve,
    /// The multiples.
    table: Table,
    /// The point, for scalars longer than the table.
    point: Ld<'c>,
}

impl<'c> BinaryPrepared<'c> {
    /// `k` times the point, from the table when `k` is not longer than what it was built for, and
    /// as [`Mul`] computes it otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let curve = named::sect131r2();
    /// let p = curve.generator().double();
    /// let prepared = p.prepare(40);
    /// let short = Number::from(0xFF_FFFF_FFFF_u64); // 40 bits: from the table
    /// let long = Number::from(u64::MAX); // 64 bits: by the product
    /// assert_eq!(prepared.mul(&short), &p * &short);
    /// assert_eq!(prepared.mul(&long), &p * &long);
    /// ```
    pub fn mul(&self, k: &Number) -> BinaryPoint<'c> {
        // In 64 bits: on 32-bit targets `as usize` would cut the bits of a long scalar short.
        if k.bits() <= self.table.bits as u64 {
            let ops = self.curve.ops();
            let r = mul::prepared_mul(&ops, &self.table, &mul::magnitude(k));
            let r = if k.is_negative() { ops.neg(&r) } else { r };
            BinaryPoint::from_ld(self.curve, r)
        } else {
            BinaryPoint::from_ld(self.curve, self.curve.product(&self.point, k))
        }
    }
}

impl Clone for BinaryPoint<'_> {
    fn clone(&self) -> Self {
        BinaryPoint {
            curve: self.curve,
            p: self.p.clone(),
        }
    }
}

/// Two points are equal when they are the same point, in whatever projective form each is held.
/// This needs no inversion.
impl PartialEq for BinaryPoint<'_> {
    #[track_caller]
    fn eq(&self, other: &Self) -> bool {
        other.check_curve(self.curve);
        self.curve.ops().equal(&self.p, &other.p)
    }
}

impl Eq for BinaryPoint<'_> {}

/// By the affine coordinates, the same for equal points: each hash takes an inversion.
impl Hash for BinaryPoint<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.coordinates() {
            Some((x, y)) => {
                1u8.hash(state);
                x.words().hash(state);
                y.words().hash(state);
            }
            None => 0u8.hash(state),
        }
    }
}

impl fmt::Debug for BinaryPoint<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `(x, y)` with each coordinate in hexadecimal, its bits those of the polynomial basis, or
/// `infinity`.
impl fmt::Display for BinaryPoint<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.coordinates() {
            Some((x, y)) => write!(f, "({x:?}, {y:?})"),
            None => f.write_str("infinity"),
        }
    }
}

impl<'c> Neg for &BinaryPoint<'c> {
    type Output = BinaryPoint<'c>;

    fn neg(self) -> BinaryPoint<'c> {
        BinaryPoint::from_ld(self.curve, self.curve.ops().neg(&self.p))
    }
}

impl<'c> Neg for BinaryPoint<'c> {
    type Output = BinaryPoint<'c>;

    fn neg(self) -> BinaryPoint<'c> {
        -&self
    }
}

impl<'c> AddAssign<&BinaryPoint<'c>> for BinaryPoint<'c> {
    #[track_caller]
    fn add_assign(&mut self, rhs: &BinaryPoint<'c>) {
        rhs.check_curve(self.curve);
        self.p = self.curve.ops().add(&self.p, &rhs.p);
    }
}

impl<'c> SubAssign<&BinaryPoint<'c>> for BinaryPoint<'c> {
    #[track_caller]
    fn sub_assign(&mut self, rhs: &BinaryPoint<'c>) {
        rhs.check_curve(self.curve);
        let ops = self.curve.ops();
        self.p = ops.add(&self.p, &ops.neg(&rhs.p));
    }
}

/// Implements a binary operator of points for every mix of owned and borrowed operands, and its
/// assignment from an owned right operand, from its assignment from a borrowed one.
macro_rules! operators {
    ($trait:ident, $method:ident, $assign:ident, $assign_method:ident) => {
        impl<'c> $assign<BinaryPoint<'c>> for BinaryPoint<'c> {
            #[track_caller]
            fn $assign_method(&mut self, rhs: BinaryPoint<'c>) {
                self.$assign_method(&rhs);
            }
        }

        impl<'c> $trait<&BinaryPoint<'c>> for BinaryPoint<'c> {
            type Output = BinaryPoint<'c>;

            #[track_caller]
            fn $method(mut self, rhs: &BinaryPoint<'c>) -> BinaryPoint<'c> {
                self.$assign_method(rhs);
                self
            }
        }

        impl<'c> $trait<BinaryPoint<'c>> for BinaryPoint<'c> {
            type Output = BinaryPoint<'c>;

            #[track_caller]
            fn $method(mut self, rhs: BinaryPoint<'c>) -> BinaryPoint<'c> {
                self.$assign_method(&rhs);
                self
            }
        }

        impl<'c> $trait<&BinaryPoint<'c>> for &BinaryPoint<'c> {
            type Output = BinaryPoint<'c>;

            #[track_caller]
            fn $method(self, rhs: &BinaryPoint<'c>) -> BinaryPoint<'c> {
                self.clone().$method(rhs)
            }
        }

        impl<'c> $trait<BinaryPoint<'c>> for &BinaryPoint<'c> {
            type Output = BinaryPoint<'c>;

            #[track_caller]
            fn $method(self, rhs: BinaryPoint<'c>) -> BinaryPoint<'c> {
                self.clone().$method(&rhs)
            }
        }
    };
}
operators!(Add, add, AddAssign, add_assign);
operators!(Sub, sub, SubAssign, sub_assign);

impl MulAssign<&Number> for BinaryPoint<'_> {
    fn mul_assign(&mut self, k: &Number) {
        self.p = self.curve.product(&self.p, k);
    }
}

impl MulAssign<Number> for BinaryPoint<'_> {
    fn mul_assign(&mut self, k: Number) {
        *self *= &k;
    }
}

impl<'c> Mul<&Number> for BinaryPoint<'c> {
    type Output = BinaryPoint<'c>;

    fn mul(mut self, k: &Number) -> BinaryPoint<'c> {
        self *= k;
        self
    }
}

impl<'c> Mul<Number> for BinaryPoint<'c> {
    type Output = BinaryPoint<'c>;

    fn mul(mut self, k: Number) -> BinaryPoint<'c> {
        self *= &k;
        self
    }
}

impl<'c> Mul<&Number> for &BinaryPoint<'c> {
    type Output = BinaryPoint<'c>;

    fn mul(self, k: &Number) -> BinaryPoint<'c> {
        BinaryPoint::from_ld(self.curve, self.curve.product(&self.p, k))
    }
}

impl<'c> Mul<Number> for &BinaryPoint<'c> {
    type Output = BinaryPoint<'c>;

    fn mul(self, k: Number) -> BinaryPoint<'c> {
        self * &k
    }
}

impl<'c> Mul<&BinaryPoint<'c>> for &Number {
    type Output = BinaryPoint<'c>;

    fn mul(self, p: &BinaryPoint<'c>) -> BinaryPoint<'c> {
        p * self
    }
}

impl<'c> Mul<BinaryPoint<'c>> for &Number {
    type Output = BinaryPoint<'c>;

    fn mul(self, p: BinaryPoint<'c>) -> BinaryPoint<'c> {
        p * self
    }
}

impl<'c> Mul<&BinaryPoint<'c>> for Number {
    type Output = BinaryPoint<'c>;

    fn mul(self, p: &BinaryPoint<'c>) -> BinaryPoint<'c> {
        p * &self
    }
}

impl<'c> Mul<BinaryPoint<'c>> for Number {
    type Output = BinaryPoint<'c>;

    fn mul(self, p: BinaryPoint<'c>) -> BinaryPoint<'c> {
        p * &self
    }
}
