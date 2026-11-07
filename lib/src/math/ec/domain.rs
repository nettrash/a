//! Curves with a generator.

use super::curve::{Curve, CurveError};
use super::field::limbs_of;
use super::group::Table;
use super::point::Point;
use crate::math::numbers::Number;
use core::ops::Deref;
use std::sync::OnceLock;

/// A curve with a generator of a subgroup, the subgroup's order and its cofactor: the domain
/// parameters that the [named curves](super::named) are, and that any curve can be given. It
/// dereferences to its curve.
///
/// It keeps the multiples of the generator once [`mul_generator`](Domain::mul_generator) has first
/// needed them, as a [`Divisor`](crate::math::numbers::Divisor) keeps what it prepares.
///
/// # Examples
///
/// ```
/// use a::math::ec::{Curve, Domain, Weierstrass};
/// use a::math::numbers::{ByteOrder, Number};
///
/// let n = |x| Number::new_from_u32(x, ByteOrder::BigEndian);
/// let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
/// // (3, 6) has order 5, and the curve 100 points.
/// let domain = Domain::new(curve, &n(3), &n(6), &n(5), &n(20)).unwrap();
/// let p = domain.mul_generator(&n(2));
/// assert_eq!(p, domain.generator().double());
/// assert!(domain.in_subgroup(&p));
/// ```
pub struct Domain<C: Curve> {
    /// The curve.
    curve: C,
    /// The generator's stored coordinates.
    generator: Vec<u64>,
    /// The order of the generator.
    order: Number,
    /// The curve's order over the generator's.
    cofactor: Number,
    /// The multiples of the generator, built when first needed.
    table: OnceLock<Table>,
}

impl<C: Curve> Domain<C> {
    /// `curve` with the generator `(x, y)` of the given order and cofactor.
    ///
    /// # Errors
    ///
    /// [`CurveError::NotOnCurve`] when `(x, y)` is not on the curve; [`CurveError::WrongOrder`]
    /// when `order` is not positive or `order` times it is not the neutral element.
    pub fn new(
        curve: C,
        x: &Number,
        y: &Number,
        order: &Number,
        cofactor: &Number,
    ) -> Result<Domain<C>, CurveError> {
        let generator = {
            let g = curve.point(x, y).map_err(|_| CurveError::NotOnCurve)?;
            if !order.is_positive() || !(&g * order).is_identity() {
                return Err(CurveError::WrongOrder);
            }
            g.limbs.clone()
        };
        Ok(Domain {
            curve,
            generator,
            order: order.clone(),
            cofactor: cofactor.clone(),
            table: OnceLock::new(),
        })
    }

    /// The generator.
    pub fn generator(&self) -> Point<'_, C> {
        Point::from_limbs(&self.curve, self.generator.clone())
    }

    /// The order of the generator.
    pub fn order(&self) -> &Number {
        &self.order
    }

    /// The number of points of the curve over the order of the generator.
    pub fn cofactor(&self) -> &Number {
        &self.cofactor
    }

    /// `k` times the generator, from a table of its multiples, several times faster than a product
    /// of any other point. The first call builds the table.
    pub fn mul_generator(&self, k: &Number) -> Point<'_, C> {
        let table = self.table();
        let k = k.rem_euclid(&self.order);
        let mut limbs = vec![0; self.generator.len()];
        self.curve
            .prepared_limbs(&mut limbs, table, &limbs_of(&k), false);
        Point::from_limbs(&self.curve, limbs)
    }

    /// `a` times the generator plus `b` times `point`, as a signature's verification needs.
    ///
    /// # Panics
    ///
    /// Panics if `point` is not on this curve.
    pub fn mul_add<'c>(&'c self, a: &Number, b: &Number, point: &Point<'c, C>) -> Point<'c, C> {
        point.check_curve(&self.curve);
        self.mul_generator(a) + point * b
    }

    /// Whether `point` is on this curve and `order` times it is the neutral element. When the order
    /// is coprime to the cofactor, as on every [named curve](super::named), those are the points
    /// of the generator's subgroup. Otherwise more can be: on `y^2 = x^3 - x` modulo 5, with the
    /// generator `(0, 0)` of order 2 and the cofactor 4, so is `(1, 0)`.
    pub fn in_subgroup(&self, point: &Point<'_, C>) -> bool {
        point.belongs_to(&self.curve) && (point * &self.order).is_identity()
    }

    /// `point` times the cofactor, which lands in the generator's subgroup when the order is
    /// coprime to the cofactor.
    ///
    /// # Panics
    ///
    /// Panics if `point` is not on this curve.
    pub fn clear_cofactor<'c>(&self, point: &Point<'c, C>) -> Point<'c, C> {
        point.check_curve(&self.curve);
        point * &self.cofactor
    }

    /// The multiples of the generator.
    fn table(&self) -> &Table {
        self.table.get_or_init(|| {
            // On 32-bit targets an order of `2^32` bits or more does not fit a `usize`, nor its
            // table the memory: `prepare` refuses `usize::MAX` bits.
            let bits = usize::try_from(self.order.bits()).unwrap_or(usize::MAX);
            self.curve.prepare_limbs(&self.generator, bits)
        })
    }
}

impl<C: Curve> Deref for Domain<C> {
    type Target = C;

    fn deref(&self) -> &C {
        &self.curve
    }
}
