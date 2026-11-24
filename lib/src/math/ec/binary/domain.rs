//! Binary curves with a generator.

use super::arith::{Affine, Ld};
use super::mul::{self, Table};
use super::{BinaryCurve, BinaryPoint};
use crate::math::ec::CurveError;
use crate::math::field::F2m;
use crate::math::numbers::Number;
use core::ops::Deref;
use std::sync::OnceLock;

/// A binary curve with a generator of a subgroup, the subgroup's order and its cofactor: the
/// domain parameters that the [named curves](super::named) are, and that any curve can be given.
/// It dereferences to its curve.
///
/// It keeps the multiples of the generator once [`mul_generator`](BinaryDomain::mul_generator)
/// has first needed them.
///
/// # Examples
///
/// ```
/// use a::math::ec::{BinaryCurve, BinaryDomain};
/// use a::math::field::BinaryField;
/// use a::math::numbers::Number;
///
/// // y² + x·y = x³ + x² + 1 over GF(2⁵) has 22 points; (x³, x⁴ + x² + x + 1) has order 11.
/// let f = BinaryField::new(&[5, 2, 0]);
/// let curve = BinaryCurve::new(&f.one(), &f.one()).unwrap();
/// let (x, y) = (f.element(&[0b1000]), f.element(&[0b10111]));
/// let domain = BinaryDomain::new(curve, &x, &y, &Number::from(11u8), &Number::from(2u8)).unwrap();
/// let p = domain.mul_generator(&Number::from(3u8));
/// assert_eq!(p, &domain.generator() * &Number::from(3u8));
/// assert!(domain.in_subgroup(&p));
/// ```
pub struct BinaryDomain {
    /// The curve.
    curve: BinaryCurve,
    /// The generator's `x`, as words.
    x: Vec<u64>,
    /// The generator's `y`, as words.
    y: Vec<u64>,
    /// The order of the generator.
    order: Number,
    /// The curve's number of points over the generator's order.
    cofactor: Number,
    /// The multiples of the generator, built when first needed.
    table: OnceLock<Table>,
}

impl BinaryDomain {
    /// `curve` with the generator `(x, y)` of the given order and cofactor.
    ///
    /// # Errors
    ///
    /// [`CurveError::NotOnCurve`] when `(x, y)` is not on the curve; [`CurveError::WrongOrder`]
    /// when `order` is not positive or `order` times the generator is not the point at infinity.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is an element of another field than the curve's.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::{BinaryCurve, BinaryDomain, CurveError};
    /// use a::math::field::BinaryField;
    /// use a::math::numbers::Number;
    ///
    /// let f = BinaryField::new(&[5, 2, 0]);
    /// let curve = || BinaryCurve::new(&f.one(), &f.one()).unwrap();
    /// let (x, y) = (f.element(&[0b1000]), f.element(&[0b10111]));
    /// let (eleven, two) = (Number::from(11u8), Number::from(2u8));
    /// assert!(BinaryDomain::new(curve(), &x, &y, &eleven, &two).is_ok());
    /// let wrong = BinaryDomain::new(curve(), &x, &y, &Number::from(7u8), &two);
    /// assert_eq!(wrong.err(), Some(CurveError::WrongOrder));
    /// let off = BinaryDomain::new(curve(), &x, &x, &eleven, &two);
    /// assert_eq!(off.err(), Some(CurveError::NotOnCurve));
    /// ```
    #[track_caller]
    pub fn new(
        curve: BinaryCurve,
        x: &F2m<'_>,
        y: &F2m<'_>,
        order: &Number,
        cofactor: &Number,
    ) -> Result<BinaryDomain, CurveError> {
        {
            let g = curve.point(x, y).map_err(|_| CurveError::NotOnCurve)?;
            if !order.is_positive() || !(&g * order).is_identity() {
                return Err(CurveError::WrongOrder);
            }
        }
        Ok(BinaryDomain {
            x: x.words().to_vec(),
            y: y.words().to_vec(),
            curve,
            order: order.clone(),
            cofactor: cofactor.clone(),
            table: OnceLock::new(),
        })
    }

    /// The generator in López–Dahab coordinates.
    fn generator_ld(&self) -> Ld<'_> {
        let f = self.curve.field();
        self.curve.ops().of_affine(&Affine {
            x: f.element(&self.x),
            y: f.element(&self.y),
        })
    }

    /// The generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect163r2();
    /// let g = curve.generator();
    /// assert!((&g * curve.order()).is_identity());
    /// assert_eq!(curve.cofactor(), &a::math::numbers::Number::from(2u8));
    /// assert_eq!(curve.order().bits(), 163);
    /// ```
    pub fn generator(&self) -> BinaryPoint<'_> {
        BinaryPoint::from_ld(&self.curve, self.generator_ld())
    }

    /// The order of the generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// // K-233's order: 2²³¹ + 0x69D5BB915BCD46EFB1AD5F173ABDF, a prime, of 232 bits.
    /// let k233 = named::k233();
    /// let n = k233.order();
    /// assert!(n.is_probable_prime());
    /// assert_eq!(n.bits(), 232);
    /// assert!((&k233.generator() * n).is_identity());
    /// ```
    pub fn order(&self) -> &Number {
        &self.order
    }

    /// The number of points of the curve over the order of the generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let k233 = named::k233();
    /// assert_eq!(k233.cofactor(), &Number::from(4u8));
    /// // h·n is the number of points of the curve.
    /// assert_eq!(k233.koblitz_order().unwrap(), k233.order() * k233.cofactor());
    /// ```
    pub fn cofactor(&self) -> &Number {
        &self.cofactor
    }

    /// `k` times the generator, `k` of any sign and length reduced modulo the order, from a table
    /// of the generator's multiples, which takes additions only. The first call builds the table.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let curve = named::sect193r1();
    /// let k = Number::from(1u8) << 300usize; // longer than the order
    /// assert_eq!(curve.mul_generator(&k), &curve.generator() * &k);
    /// assert_eq!(curve.mul_generator(&-&k), -(&curve.generator() * &k));
    /// ```
    pub fn mul_generator(&self, k: &Number) -> BinaryPoint<'_> {
        let table = self.table.get_or_init(|| {
            // On 32-bit targets an order of `2^32` bits or more does not fit a `usize`, nor its
            // table the memory: `prepare` saturates to `usize::MAX` bits and fails to allocate.
            let bits = usize::try_from(self.order.bits()).unwrap_or(usize::MAX);
            mul::prepare(&self.curve.ops(), &self.generator_ld(), bits)
        });
        let k = k.rem_euclid(&self.order);
        let r = mul::prepared_mul(&self.curve.ops(), table, &mul::magnitude(&k));
        BinaryPoint::from_ld(&self.curve, r)
    }

    /// `a` times the generator plus `b` times `point`, as a signature's verification needs.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    /// use a::math::numbers::Number;
    ///
    /// let curve = named::sect193r2();
    /// let g = curve.generator();
    /// let (a, b) = (Number::from(5u8), Number::from(7u8));
    /// assert_eq!(curve.mul_add(&a, &b, &g), &g * &Number::from(12u8));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `point` is not on this curve.
    #[track_caller]
    pub fn mul_add<'c>(
        &'c self,
        a: &Number,
        b: &Number,
        point: &BinaryPoint<'c>,
    ) -> BinaryPoint<'c> {
        point.check_curve(&self.curve);
        self.mul_generator(a) + point * b
    }

    /// Whether `point` is on this curve and `order` times it is the point at infinity. When the
    /// order is a prime that does not divide the cofactor, as on every
    /// [named curve](super::named), those are the points of the generator's subgroup.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect113r1();
    /// // (0, √b) has order 2, and the cofactor is 2.
    /// let t = curve.lift(&curve.field().zero(), false).unwrap();
    /// assert!(!curve.in_subgroup(&t));
    /// assert!(curve.in_subgroup(&curve.generator()));
    /// assert!(curve.in_subgroup(&curve.clear_cofactor(&(&t + &curve.generator()))));
    /// ```
    pub fn in_subgroup(&self, point: &BinaryPoint<'_>) -> bool {
        point.belongs_to(&self.curve) && (point * &self.order).is_identity()
    }

    /// `point` times the cofactor, which lands in the generator's subgroup when the order is
    /// prime to the cofactor.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::binary::named;
    ///
    /// let curve = named::sect131r1();
    /// let t = curve.lift(&curve.field().zero(), false).unwrap();
    /// assert!(curve.clear_cofactor(&t).is_identity());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `point` is not on this curve.
    #[track_caller]
    pub fn clear_cofactor<'c>(&self, point: &BinaryPoint<'c>) -> BinaryPoint<'c> {
        point.check_curve(&self.curve);
        point * &self.cofactor
    }
}

impl Deref for BinaryDomain {
    type Target = BinaryCurve;

    fn deref(&self) -> &BinaryCurve {
        &self.curve
    }
}
