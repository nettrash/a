//! Interpolation and approximation of a function of one real variable.
//!
//! Five piecewise cubics through given knots share one representation, the Hermite form: on every
//! interval the polynomial is the cubic that takes the value `y[i]` with slope `m[i]` at one knot
//! and `y[i+1]` with slope `m[i+1]` at the next. They differ only in how the slopes are chosen,
//! and so in how smooth and how faithful to the shape of the data they are:
//!
//! | Constructor | Slopes from | Smoothness | Overshoots |
//! | --- | --- | --- | --- |
//! | [`Spline::cubic`] | a tridiagonal solve | `C²` | yes |
//! | [`Spline::cubic_clamped`] | the same, the end slopes given | `C²` | yes |
//! | [`Spline::cubic_not_a_knot`] | the same, the third derivative continued | `C²` but at two knots | yes |
//! | [`Spline::akima`] | Akima's weights of the four nearest secants | `C¹` | little |
//! | [`Spline::pchip`] | Fritsch and Carlson's monotone slopes | `C¹` | never between monotone knots |
//!
//! A spline answers [`evaluate`](Spline::evaluate), [`derivative`](Spline::derivative),
//! [`second_derivative`](Spline::second_derivative) and [`integral`](Spline::integral) between its
//! first and last knot, and `None` outside them: it does not extrapolate.
//!
//! Where the function is smooth across the whole interval one polynomial is better, and the
//! polynomial that agrees with the function at the Chebyshev nodes is within a small factor of the
//! best one there is: [`chebyshev`] fits it and evaluates it by Clenshaw's recurrence, in time
//! linear in the degree. It is a polynomial, so outside its interval it is evaluated as one, which
//! is an extrapolation and usually a poor one. [`barycentric_weights`] and
//! [`barycentric_interpolate`] evaluate the polynomial through values given at any distinct
//! nodes, in the second barycentric form, which is what to use when the values are data rather
//! than a function to be sampled. Whether that polynomial is any good depends on the nodes, not
//! on the formula: at Chebyshev nodes it converges, at equally spaced ones it does not (Runge's
//! phenomenon).
//!
//! Every constructor panics on input of the wrong shape (values and knots of different lengths,
//! too few knots, knots that do not increase) and returns `None` on knots or values that are not
//! finite.

use super::Real;
use crate::math::fft::{Dct, DctKind};
use std::sync::OnceLock;

/// The number of nodes up to which [`chebyshev`] sums its cosine transform directly, `n²`
/// cosines, rather than planning the fast transform of `a::math::fft`.
// Measured on an Apple M4 Max, the whole of `chebyshev` on a cheap function: the transform
// takes 2.2 to 2.5 times the time of the direct sum at 4 to 8 nodes, 1.03 to 1.16 at 10 and 12,
// and 0.48 to 0.79 from 13 on, in `f64` and `f32` (at the odd sizes 9 and 11 already 0.84 to
// 0.90; sweep 4 to 96 nodes).
const DIRECT_TRANSFORM: usize = 12;

/// The piecewise cubic through the knots `(x[i], y[i])`, held in Hermite form. Build one with
/// [`Spline::cubic`], [`Spline::cubic_clamped`], [`Spline::cubic_not_a_knot`], [`Spline::akima`]
/// or [`Spline::pchip`], and ask it for [`evaluate`](Self::evaluate),
/// [`derivative`](Self::derivative), [`second_derivative`](Self::second_derivative) and
/// [`integral`](Self::integral).
///
/// # Examples
///
/// ```
/// use a::math::numerics::Spline;
///
/// // The sine through 33 knots on [0, 1]. Between the knots the value is good to nine digits and
/// // the slope to seven; the natural condition at the ends costs two of the four digits that a
/// // spline of degree three otherwise earns, which is what `Spline::cubic_not_a_knot` is for.
/// let x: Vec<f64> = (0..33).map(|k| f64::from(k) / 32.0).collect();
/// let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();
/// let s = Spline::cubic(&x, &y).unwrap();
/// assert!((s.evaluate(0.731).unwrap() - 0.731f64.sin()).abs() < 1e-8);
/// assert!((s.derivative(0.731).unwrap() - 0.731f64.cos()).abs() < 1e-6);
/// // Outside the knots there is no answer.
/// assert_eq!(s.evaluate(1.5), None);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Spline<T> {
    x: Vec<T>,
    y: Vec<T>,
    m: Vec<T>,
}

impl<T: Real> Spline<T> {
    /// The knots, between which [`evaluate`](Self::evaluate) is defined.
    pub fn knots(&self) -> &[T] {
        &self.x
    }

    /// The values at the knots.
    pub fn values(&self) -> &[T] {
        &self.y
    }

    /// The slopes at the knots, which with the values are the Hermite form of the spline.
    pub fn slopes(&self) -> &[T] {
        &self.m
    }

    /// The interval that holds `t`, the last one that begins at or before it, so that the right
    /// end of the interval belongs to it; found by bisection of the knots.
    fn interval(&self, t: T) -> Option<usize> {
        let n = self.x.len();
        if n < 2 || !(t >= self.x[0] && t <= self.x[n - 1]) {
            return None;
        }
        // The number of knots at or before `t` is at least one, since `t` is not below the first.
        let at_or_before = self.x.partition_point(|&k| k <= t);
        Some((at_or_before - 1).min(n - 2))
    }

    /// The interval `i`, its length, and where `t` lies in it as a fraction of that length.
    fn locate(&self, t: T) -> Option<(usize, T, T)> {
        let i = self.interval(t)?;
        let h = self.x[i + 1] - self.x[i];
        Some((i, h, (t - self.x[i]) / h))
    }

    /// The value of the spline at `t`, or `None` when `t` is outside the knots. At a knot it is
    /// the value given there, exactly, and on an interval between two equal values with zero
    /// slopes it is that value, exactly.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// let s = Spline::cubic(&[0.0f64, 1.0, 2.0], &[0.0, 1.0, 0.0]).unwrap();
    /// assert_eq!(s.evaluate(1.0), Some(1.0));
    /// assert!((s.evaluate(0.5).unwrap() - 0.6875).abs() < 1e-15);
    /// assert_eq!(s.evaluate(-0.1), None);
    /// ```
    pub fn evaluate(&self, t: T) -> Option<T> {
        let (i, h, u) = self.locate(t)?;
        if u == T::ONE {
            return Some(self.y[i + 1]);
        }
        let v = T::ONE - u;
        // The Hermite basis h00·y[i] + h01·y[i+1] + h10·h·m[i] + h11·h·m[i+1], with h00 + h01 = 1
        // used to carry the values as the left one plus the difference: a flat run then stays
        // flat to the last bit, and a large common offset is not rounded into the rest.
        let h01 = u * u * (T::from_f64(3.0) - T::TWO * u);
        let h10 = u * v * v;
        let h11 = -(u * u * v);
        Some(
            self.y[i]
                + (self.y[i + 1] - self.y[i]) * h01
                + h * (h10 * self.m[i] + h11 * self.m[i + 1]),
        )
    }

    /// The derivative of the spline at `t`, or `None` when `t` is outside the knots.
    ///
    /// The two values enter only through their difference, the secant of the interval, so that
    /// data with a large common offset lose nothing to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // Through two knots the natural spline is the secant, whatever the offset of the data.
    /// let s = Spline::cubic(&[0.0f64, 1.0], &[101_325.0, 101_325.01]).unwrap();
    /// assert_eq!(s.derivative(0.5), Some(s.slopes()[0]));
    /// ```
    pub fn derivative(&self, t: T) -> Option<T> {
        let (i, h, u) = self.locate(t)?;
        let v = T::ONE - u;
        let secant = (self.y[i + 1] - self.y[i]) / h;
        // The derivatives of the Hermite basis with respect to u: −6uv and 6uv for the values,
        // v(v − 2u) and u(3u − 2) for the slopes, which carry the length of the interval already.
        Some(
            T::from_f64(6.0) * u * v * secant
                + v * (v - T::TWO * u) * self.m[i]
                + u * (T::from_f64(3.0) * u - T::TWO) * self.m[i + 1],
        )
    }

    /// The second derivative of the spline at `t`, or `None` when `t` is outside the knots. At a
    /// knot it is the one of the interval to the right (of the last interval at the last knot),
    /// which matters for the `C¹` splines, whose second derivative jumps there.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // The natural spline has no curvature at its ends, and a parabola's not-a-knot spline is
    /// // the parabola, of curvature 2 everywhere.
    /// let x = [0.0f64, 0.4, 1.0, 1.7, 2.0];
    /// let natural = Spline::cubic(&x, &[0.0, 1.0, 0.5, 0.2, 0.9]).unwrap();
    /// assert!(natural.second_derivative(0.0).unwrap().abs() < 1e-14);
    /// let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    /// let parabola = Spline::cubic_not_a_knot(&x, &y).unwrap();
    /// assert!((parabola.second_derivative(1.3).unwrap() - 2.0).abs() < 1e-13);
    /// ```
    pub fn second_derivative(&self, t: T) -> Option<T> {
        let (i, h, u) = self.locate(t)?;
        let secant = (self.y[i + 1] - self.y[i]) / h;
        // The second derivatives of the Hermite basis: 12u − 6 and 6 − 12u for the values,
        // 6u − 4 and 6u − 2 for the slopes, over the square of the interval.
        let six = T::from_f64(6.0);
        Some(
            ((six - T::from_f64(12.0) * u) * secant
                + (six * u - T::from_f64(4.0)) * self.m[i]
                + (six * u - T::TWO) * self.m[i + 1])
                / h,
        )
    }

    /// The integral of the spline from `a` to `b`, negative when `b` is below `a`, or `None`
    /// when either is outside the knots. Each cubic is integrated exactly, the interval holding
    /// `a` and the one holding `b` in part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // The not-a-knot spline of a cubic is the cubic, so its integral is exact: the integral of
    /// // x³ from 0.5 to 2.5 is (2.5⁴ − 0.5⁴)/4 = 9.75.
    /// let x = [0.0f64, 0.7, 1.5, 2.0, 3.0];
    /// let y: Vec<f64> = x.iter().map(|t| t * t * t).collect();
    /// let s = Spline::cubic_not_a_knot(&x, &y).unwrap();
    /// assert!((s.integral(0.5, 2.5).unwrap() - 9.75).abs() < 1e-13);
    /// assert!((s.integral(2.5, 0.5).unwrap() + 9.75).abs() < 1e-13);
    /// assert_eq!(s.integral(0.5, 3.5), None);
    /// ```
    pub fn integral(&self, a: T, b: T) -> Option<T> {
        let (ia, _, ua) = self.locate(a)?;
        let (ib, _, ub) = self.locate(b)?;
        if a > b {
            return self.integral(b, a).map(|v| -v);
        }
        if ia == ib {
            return Some(self.primitive(ia, ub) - self.primitive(ia, ua));
        }
        // The rest of the interval of a, the whole intervals between, and the start of b's.
        let mut sum = self.primitive(ia, T::ONE) - self.primitive(ia, ua);
        for i in ia + 1..ib {
            sum += self.primitive(i, T::ONE);
        }
        Some(sum + self.primitive(ib, ub))
    }

    /// The integral of the cubic of interval `i` from its left knot to the fraction `u` of its
    /// length. The values enter as the left one and the difference, so that a large common
    /// offset does not cancel.
    fn primitive(&self, i: usize, u: T) -> T {
        let h = self.x[i + 1] - self.x[i];
        let (u2, u3) = (u * u, u * u * u);
        let u4 = u2 * u2;
        let quarter = T::from_f64(0.25);
        let third = T::ONE / T::from_f64(3.0);
        // ∫h00 = u − u³ + u⁴/2 and ∫h01 = u³ − u⁴/2 sum to u; ∫h10 = u²/2 − 2u³/3 + u⁴/4 and
        // ∫h11 = u⁴/4 − u³/3.
        let values = self.y[i] * u + (self.y[i + 1] - self.y[i]) * (u3 - u4 * T::HALF);
        let left = u2 * T::HALF - T::TWO * third * u3 + quarter * u4;
        let right = quarter * u4 - third * u3;
        h * (values + h * (left * self.m[i] + right * self.m[i + 1]))
    }

    /// The spline of `y` at the strictly increasing knots `x` whose second derivative vanishes at
    /// both ends, the natural spline. It is the smoothest of the five and the one to reach for
    /// when nothing is known about the ends. Through two knots it is the straight line.
    ///
    /// Returns `None` when a knot or a value is not finite.
    ///
    /// # Panics
    ///
    /// If `x` and `y` differ in length, if there are fewer than two of them, or if `x` is not
    /// strictly increasing.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // Three knots give a parabola at the middle and straight ends; the natural condition
    /// // leaves the slopes below.
    /// let s = Spline::cubic(&[0.0f64, 1.0, 2.0], &[0.0, 1.0, 0.0]).unwrap();
    /// assert_eq!(s.slopes(), &[1.5, 0.0, -1.5]);
    /// assert!((s.evaluate(0.5).unwrap() - 0.6875).abs() < 1e-15);
    /// assert_eq!(Spline::cubic(&[0.0, 1.0], &[0.0, f64::NAN]), None);
    /// ```
    #[track_caller]
    pub fn cubic(x: &[T], y: &[T]) -> Option<Self> {
        Self::build(x, y, End::Natural, T::ZERO, T::ZERO)
    }

    /// The spline of `y` at the knots `x` whose slope at the first knot is `left` and at the last
    /// `right` — the clamped spline, better than the natural one where the end slopes are known,
    /// since the two extra conditions are worth two knots of accuracy. Through two knots it is
    /// the cubic with those two slopes.
    ///
    /// Returns `None` when a knot, a value or one of the two slopes is not finite.
    ///
    /// # Panics
    ///
    /// As [`Spline::cubic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // The sine on [0, π] through nine knots, clamped to its own slopes, 1 and −1, at the ends.
    /// let x: Vec<f64> = (0..9).map(|k| f64::from(k) * core::f64::consts::PI / 8.0).collect();
    /// let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();
    /// let s = Spline::cubic_clamped(&x, &y, 1.0, -1.0).unwrap();
    /// assert_eq!(&s.slopes()[..1], &[1.0]);
    /// assert!((s.evaluate(0.5).unwrap() - 0.5f64.sin()).abs() < 1e-4);
    ///
    /// // Two knots and two slopes: the Hermite cubic.
    /// let s = Spline::cubic_clamped(&[0.0, 2.0], &[1.0, 5.0], 9.0, -9.0).unwrap();
    /// assert_eq!(s.slopes(), &[9.0, -9.0]);
    /// assert_eq!(s.evaluate(1.0), Some(7.5));
    /// ```
    #[track_caller]
    pub fn cubic_clamped(x: &[T], y: &[T], left: T, right: T) -> Option<Self> {
        Self::build(x, y, End::Clamped, left, right)
    }

    /// The spline of `y` at the knots `x` whose third derivative is continued across the second
    /// and penultimate knots — the not-a-knot spline, the usual choice when the end behaviour of
    /// the function is unknown. Through three knots the two conditions coincide and it is the
    /// parabola through them; through four it is the cubic through them; through two, the line.
    ///
    /// The conditions make one cubic of the first two intervals and one of the last two, so the
    /// spline is found as the cubic spline of the knots without the second and the penultimate,
    /// through all the points: the slopes at its knots solve a tridiagonal system whose end rows
    /// say that the end cubics pass through the two points left out, by elimination with scaled
    /// partial pivoting, and through four knots the slopes are those of divided differences. The
    /// slopes are then as accurate as the values determine them, within a small multiple of the
    /// change one unit in the last place of the values makes, for any spacing of the knots — the
    /// system of all the slopes with the conditions combined into its first and last rows, as
    /// C. de Boor and SciPy write it, loses as many digits as a short second or penultimate
    /// interval is shorter than its neighbours.
    ///
    /// Returns `None` when a knot or a value is not finite.
    ///
    /// # Panics
    ///
    /// As [`Spline::cubic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // The exponential through nine knots on [0, 1]: the fourth power of the step, 0.125, is
    /// // the accuracy to look for, since the third derivative no longer has a knot to jump at.
    /// let x: Vec<f64> = (0..9).map(|k| f64::from(k) / 8.0).collect();
    /// let y: Vec<f64> = x.iter().map(|&t| t.exp()).collect();
    /// let s = Spline::cubic_not_a_knot(&x, &y).unwrap();
    /// assert!((s.evaluate(0.65).unwrap() - 0.65f64.exp()).abs() < 1e-6);
    ///
    /// // Three points of x² give x² itself.
    /// let s = Spline::cubic_not_a_knot(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap();
    /// assert_eq!(s.slopes(), &[0.0, 2.0, 4.0]);
    /// assert_eq!(s.evaluate(1.5), Some(2.25));
    /// ```
    #[track_caller]
    pub fn cubic_not_a_knot(x: &[T], y: &[T]) -> Option<Self> {
        Self::build(x, y, End::NotAKnot, T::ZERO, T::ZERO)
    }

    /// Akima's spline of `y` at the knots `x` (H. Akima, *J. ACM* 17, 1970): the slope at a knot
    /// is a weighted mean of the two secants that meet at it, each weighted by how far apart the
    /// two secants beyond the other one are, so that a knot in the middle of a straight run takes
    /// the slope of that run exactly; where neither side decides, the plain mean. At the ends the
    /// secants are continued by Akima's own rule, two more on each side, each the previous one
    /// plus the change between the last two (`2·d[n−2] − d[n−3]` and so on). Nothing is solved,
    /// so it costs what the secants cost, and it swings far less than the cubic spline near a
    /// jump; it is only `C¹`. Through two knots it is the line.
    ///
    /// The weights are normalised before they multiply a secant, so that no product of two
    /// secants is formed and data far from 1 in size neither overflow nor underflow. A weight is
    /// zero only when the two secants are equal; SciPy's `Akima1DInterpolator` takes a weight
    /// below `10⁻⁹` of the largest as zero, so the two part where straight runs carry rounding:
    /// at a corner between two such runs the slope here follows the rounding, anywhere between
    /// the two secants (0.667 for the secants 0 and 1 with noise of `10⁻¹³`), and SciPy's is their
    /// mean.
    ///
    /// Returns `None` when a knot or a value is not finite, or a secant overflows.
    ///
    /// # Panics
    ///
    /// As [`Spline::cubic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // A step: Akima follows the flat parts exactly rather than ringing around the jump.
    /// let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    /// let s = Spline::akima(&x, &y).unwrap();
    /// assert_eq!(s.evaluate(0.5), Some(0.0));
    /// assert_eq!(s.evaluate(4.5), Some(1.0));
    ///
    /// // A parabola keeps its slopes at both ends: the end rule continues it.
    /// let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    /// assert_eq!(Spline::akima(&x, &y).unwrap().slopes(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    /// ```
    #[track_caller]
    pub fn akima(x: &[T], y: &[T]) -> Option<Self> {
        if !check(x, y) {
            return None;
        }
        let n = x.len();
        if n == 2 {
            return Self::secant(x, y);
        }
        // The secants d[k + 2] of the interval beginning at x[k], continued two past each end by
        // Akima's rule, so that every knot has the four secants nearest it.
        let mut d = vec![T::ZERO; n + 3];
        for k in 0..n - 1 {
            d[k + 2] = (y[k + 1] - y[k]) / (x[k + 1] - x[k]);
        }
        d[1] = T::TWO * d[2] - d[3];
        d[0] = T::TWO * d[1] - d[2];
        d[n + 1] = T::TWO * d[n] - d[n - 1];
        d[n + 2] = T::TWO * d[n + 1] - d[n];
        let m = (0..n).map(|i| akima_slope(&d[i..i + 4])).collect();
        Self::finished(x, y, m)
    }

    /// The shape-preserving spline of `y` at the knots `x`, PCHIP (F. N. Fritsch and R. E.
    /// Carlson, *SIAM J. Numer. Anal.* 17, 1980, with the slopes of F. N. Fritsch and J. Butland,
    /// *SIAM J. Sci. Stat. Comput.* 5, 1984, as SLATEC's `PCHIM` and SciPy's
    /// `PchipInterpolator` take them). At an interior knot the slope is zero where the data turn
    /// or a secant vanishes, and otherwise the harmonic mean of the two secants weighted by the
    /// lengths of the intervals, which lies between them; at an end it is the one-sided slope of
    /// the parabola through the three nearest knots, set to zero if it points against the end
    /// secant and limited to three times that secant if the data turn at the next knot. Between
    /// two knots of monotone data the spline is monotone too, so it never leaves their range;
    /// through two knots it is the line.
    ///
    /// The harmonic mean is formed from the reciprocals of the secants, never their product, so
    /// that data far from 1 in size neither overflow nor underflow.
    ///
    /// Returns `None` when a knot or a value is not finite, or a secant overflows.
    ///
    /// # Panics
    ///
    /// As [`Spline::cubic`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Spline;
    ///
    /// // A peak: the monotone spline never rises above the knot that peaks.
    /// let s = Spline::pchip(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 0.0, 0.0]).unwrap();
    /// assert!(s.evaluate(0.5).unwrap() <= 1.0);
    /// assert!(s.evaluate(1.5).unwrap() >= 0.0);
    /// // The three-point slope at the left end, 2, rather than nought.
    /// assert_eq!(s.slopes(), &[2.0, 0.0, 0.0, 0.0]);
    /// ```
    #[track_caller]
    pub fn pchip(x: &[T], y: &[T]) -> Option<Self> {
        if !check(x, y) {
            return None;
        }
        let n = x.len();
        if n == 2 {
            return Self::secant(x, y);
        }
        let h: Vec<T> = (0..n - 1).map(|k| x[k + 1] - x[k]).collect();
        let d: Vec<T> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / h[k]).collect();
        let mut m = vec![T::ZERO; n];
        for i in 1..n - 1 {
            let (dl, dr) = (d[i - 1], d[i]);
            if sign(dl) * sign(dr) <= 0 {
                continue;
            }
            // 1/m = (w1/dl + w2/dr)/(w1 + w2), w1 = 2·hr + hl and w2 = hr + 2·hl: the weights
            // normalised first, so that nothing larger than a secant's reciprocal is formed.
            let (hl, hr) = (h[i - 1], h[i]);
            let (w1, w2) = (T::TWO * hr + hl, hr + T::TWO * hl);
            let total = w1 + w2;
            m[i] = T::ONE / ((w1 / total) / dl + (w2 / total) / dr);
        }
        m[0] = pchip_end(h[0], h[1], d[0], d[1]);
        m[n - 1] = pchip_end(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
        Self::finished(x, y, m)
    }

    /// The straight line through two knots, which every constructor but the clamped one gives
    /// for them.
    fn secant(x: &[T], y: &[T]) -> Option<Self> {
        let d = (y[1] - y[0]) / (x[1] - x[0]);
        Self::finished(x, y, vec![d, d])
    }

    /// The spline of the knots and slopes, or `None` when a slope is not finite.
    fn finished(x: &[T], y: &[T], m: Vec<T>) -> Option<Self> {
        m.iter().all(|s| s.is_finite()).then(|| Self {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }

    /// The cubic spline in Hermite form, from the slopes its end condition asks for. The interior
    /// slopes come from the tridiagonal system that makes the second derivative continuous. The
    /// natural and clamped conditions are eliminated into its first and last rows, which leaves
    /// the system diagonally dominant; the not-a-knot condition is not, and is solved as a spline
    /// of fewer knots (see [`not_a_knot`]).
    #[track_caller]
    fn build(x: &[T], y: &[T], end: End, left: T, right: T) -> Option<Self> {
        if !check(x, y) || !left.is_finite() || !right.is_finite() {
            return None;
        }
        let n = x.len();
        if n == 2 {
            return match end {
                End::Clamped => Self::finished(x, y, vec![left, right]),
                End::Natural | End::NotAKnot => Self::secant(x, y),
            };
        }
        let h: Vec<T> = (0..n - 1).map(|k| x[k + 1] - x[k]).collect();
        let d: Vec<T> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / h[k]).collect();
        if end == End::NotAKnot {
            return Self::finished(x, y, not_a_knot(x, y, &h, &d)?);
        }
        let mut m = vec![T::ZERO; n];
        // The rows of the interior knots x[1] .. x[n-2]: the second derivative is continuous
        // there, which is one linear relation among the three slopes that meet at the knot.
        let k = n - 2;
        let mut lower = vec![T::ZERO; k];
        let mut centre = vec![T::ZERO; k];
        let mut upper = vec![T::ZERO; k];
        let mut rhs = vec![T::ZERO; k];
        for i in 0..k {
            // The second derivative is continuous at x[i+1], which weighs the slope of the
            // interval on the far side of a knot by the length of the one on the near side. The
            // pairing looks backwards and is not: multiplying the row by 1/(h[i]·h[i+1]) leaves
            // 1/h[i+1] on both sides of the diagonal, so the system is symmetrisable, and it is
            // this one that makes the second derivative continuous.
            lower[i] = h[i + 1];
            centre[i] = T::TWO * (h[i] + h[i + 1]);
            upper[i] = h[i];
            rhs[i] = T::from_f64(3.0) * (h[i + 1] * d[i] + h[i] * d[i + 1]);
        }
        if end == End::Natural {
            natural_rows(&mut lower, &mut centre, &mut upper, &mut rhs, &h, &d);
        } else {
            // The end slopes are given, so their terms move to the right-hand side: the first
            // row carries h[1]·m[0] and the last h[n−3]·m[n−1].
            rhs[0] -= lower[0] * left;
            rhs[k - 1] -= upper[k - 1] * right;
            m[0] = left;
            m[n - 1] = right;
        }
        solve_tridiagonal(&lower, &mut centre, &upper, &mut rhs, &mut m[1..n - 1])?;
        if end == End::Natural {
            // The end slopes themselves, from the two relations that eliminated them.
            m[0] = (T::from_f64(3.0) * d[0] - m[1]) * T::HALF;
            m[n - 1] = (T::from_f64(3.0) * d[k] - m[n - 2]) * T::HALF;
        }
        Self::finished(x, y, m)
    }
}

/// How the two end rows of the spline's tridiagonal system are filled.
#[derive(Debug, Clone, Copy, PartialEq)]
enum End {
    Natural,
    Clamped,
    NotAKnot,
}

/// The arguments every constructor checks: `false` when a knot or a value is not finite.
///
/// # Panics
///
/// On knots and values that are not equally long, fewer than two knots, or finite knots that
/// are not strictly increasing.
#[track_caller]
fn check<T: Real>(x: &[T], y: &[T]) -> bool {
    assert_eq!(
        x.len(),
        y.len(),
        "the knots and the values differ in length"
    );
    assert!(x.len() >= 2, "a spline needs at least two knots");
    if !x.iter().chain(y).all(|v| v.is_finite()) {
        return false;
    }
    assert!(
        x.windows(2).all(|w| w[0] < w[1]),
        "the knots must be strictly increasing"
    );
    true
}

/// The sign of `v` as −1, 0 or 1; unlike `signum`, zero has the sign 0.
fn sign<T: Real>(v: T) -> i32 {
    if v > T::ZERO {
        1
    } else if v < T::ZERO {
        -1
    } else {
        0
    }
}

/// Akima's slope at a knot, from the four secants `d[0..4]` nearest it, of which `d[1]` ends and
/// `d[2]` begins at the knot: `(|d3 − d2|·d1 + |d1 − d0|·d2)/(|d3 − d2| + |d1 − d0|)`, the mean of
/// `d1` and `d2` when both weights vanish. The weights are scaled by the larger before they are
/// normalised, so that they lie in `[0, 1]` and multiply a secant without overflow or underflow;
/// mirrored data give the mirrored slope to the last bit, since the two terms are formed alike.
fn akima_slope<T: Real>(d: &[T]) -> T {
    let (w1, w2) = ((d[3] - d[2]).abs(), (d[1] - d[0]).abs());
    let big = w1.max(w2);
    if big == T::ZERO {
        return (d[1] + d[2]) * T::HALF;
    }
    let (w1, w2) = (w1 / big, w2 / big);
    let total = w1 + w2;
    (w1 / total) * d[1] + (w2 / total) * d[2]
}

/// The slope at an end of PCHIP, from the end interval (length `h0`, secant `d0`) and the next one
/// (`h1`, `d1`): the one-sided derivative of the parabola through the three knots,
/// `((2·h0 + h1)·d0 − h0·d1)/(h0 + h1)`, written as `d0 + (d0 − d1)·h0/(h0 + h1)` so that no
/// product overflows; zero when it points against `d0` (or `d0` is zero), and `3·d0` when the
/// data turn at the next knot and it is steeper than that, which is the limit that keeps the
/// end interval monotone.
fn pchip_end<T: Real>(h0: T, h1: T, d0: T, d1: T) -> T {
    let s = d0 + (d0 - d1) * (h0 / (h0 + h1));
    let three = T::from_f64(3.0);
    if sign(s) != sign(d0) {
        T::ZERO
    } else if sign(d0) != sign(d1) && s.abs() > three * d0.abs() {
        three * d0
    } else {
        s
    }
}

/// The Thomas algorithm: a tridiagonal solve in time linear in the size of the system, without
/// row interchanges, for the diagonally dominant systems of the natural and clamped splines.
/// `lower` and `upper` are the off-diagonals, `centre` the diagonal (destroyed), `rhs` the
/// right-hand side (destroyed) and `x` where the answer is written.
///
/// Returns `None` on a zero pivot, which is a singular system.
fn solve_tridiagonal<T: Real>(
    lower: &[T],
    centre: &mut [T],
    upper: &[T],
    rhs: &mut [T],
    x: &mut [T],
) -> Option<()> {
    let n = centre.len();
    if n == 0 {
        return Some(());
    }
    let mut c = vec![T::ZERO; n];
    let mut d = vec![T::ZERO; n];
    if centre[0] == T::ZERO {
        return None;
    }
    c[0] = upper[0] / centre[0];
    d[0] = rhs[0] / centre[0];
    for i in 1..n {
        let pivot = centre[i] - lower[i] * c[i - 1];
        if pivot == T::ZERO {
            return None;
        }
        c[i] = if i + 1 < n { upper[i] / pivot } else { T::ZERO };
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / pivot;
    }
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    Some(())
}

/// The two end rows of the natural spline. A second derivative that vanishes at an end relates
/// its two slopes — `2·m[0] + m[1] = 3·d[0]` and `m[n−2] + 2·m[n−1] = 3·d[n−2]` — and those two
/// relations are what eliminate the end slopes from the first and last rows.
fn natural_rows<T: Real>(
    lower: &mut [T],
    centre: &mut [T],
    upper: &mut [T],
    rhs: &mut [T],
    h: &[T],
    d: &[T],
) {
    let k = centre.len();
    // The first row multiplies m[0] by the length of the interval after it, and the last row
    // multiplies m[n−1] by the length of the interval before it.
    let (h0, hl) = (h[1], h[k - 1]);
    // m[0] = (3·d[0] − m[1])/2 and m[n−1] = (3·d[n−2] − m[n−2])/2, whose terms leave the two
    // rows with nothing multiplying an end slope.
    centre[0] -= h0 * T::HALF;
    rhs[0] -= h0 * (T::from_f64(3.0) * d[0]) * T::HALF;
    centre[k - 1] -= hl * T::HALF;
    rhs[k - 1] -= hl * (T::from_f64(3.0) * d[k]) * T::HALF;
    lower[0] = T::ZERO;
    upper[k - 1] = T::ZERO;
}

/// The slopes of the not-a-knot spline through the knots `x` and values `y`, from the lengths
/// `h` and the secants `d` of its intervals (at least two of them), or `None` on a singular
/// system.
///
/// Continuing the third derivative across the second and the penultimate knot makes one cubic
/// span the first two intervals and one the last two, so the spline is the cubic spline of the
/// knots without `x[1]` and `x[n−2]` that also passes through those two points. Through three
/// knots the two conditions say the same, and leave the parabola; through four, one cubic spans
/// all three intervals, and [`cubic_through_four`] gives its slopes. From five knots on the
/// unknowns are the slopes at the knots that remain, `m[0]`, `m[2]`, …, `m[n−3]`, `m[n−1]`: at
/// each interior one the second derivative is continuous, the usual tridiagonal row, and the first
/// and last rows say that the end cubics pass through `y[1]` and `y[n−2]`. For an end cubic over
/// `[x[0], x[2]]` that is `h[1]·m[0] − h[0]·m[2] = h[1]·(1 + 2u)·d[0] − h[0]·(1 + 2v)·d[1]`, with
/// `u = h[0]/(h[0] + h[1])` and `v = h[1]/(h[0] + h[1])`. The system is solved by
/// [`solve_tridiagonal_scaled`], and the slopes at `x[1]` and `x[n−2]` are those of the end cubics
/// there.
///
/// It is the spline of C. de Boor's *A Practical Guide to Splines* (1978, chapter IV) and of
/// SciPy's `CubicSpline`, which write the system in all `n` slopes instead, the two conditions
/// combined with the rows of the second and penultimate knots. Those rows lose as many digits as a
/// short second (or penultimate) interval is shorter than its neighbours — the square of that for
/// four knots — where the data themselves lose far fewer; written as here, the slopes are as
/// accurate as the values determine them, whatever the spacing.
fn not_a_knot<T: Real>(x: &[T], y: &[T], h: &[T], d: &[T]) -> Option<Vec<T>> {
    let n = x.len();
    if n == 3 {
        // The parabola through the three knots, whose slopes at the ends and the middle are
        // these; each is a secant corrected by the change of the secants.
        let s = h[0] + h[1];
        return Some(vec![
            d[0] + (d[0] - d[1]) * (h[0] / s),
            d[0] + (d[1] - d[0]) * (h[0] / s),
            d[1] + (d[1] - d[0]) * (h[1] / s),
        ]);
    }
    if n == 4 {
        return Some(cubic_through_four(x, y, d));
    }
    // The k knots of the reduced spline, x[0], x[2], …, x[n−3], x[n−1], and its pieces.
    let k = n - 2;
    let knot = |i: usize| {
        if i == 0 {
            0
        } else if i == k - 1 {
            n - 1
        } else {
            i + 1
        }
    };
    // The pieces' lengths and secants: those of the first and the last piece, each over two
    // intervals; the others are intervals of the spline, whose `h` and `d` they are, formed by
    // the same operations.
    let first = x[2] - x[0];
    let last = x[n - 1] - x[n - 3];
    let (first_secant, last_secant) = ((y[2] - y[0]) / first, (y[n - 1] - y[n - 3]) / last);
    let length = |i: usize| {
        if i == 0 {
            first
        } else if i == k - 2 {
            last
        } else {
            h[i + 1]
        }
    };
    let secant = |i: usize| {
        if i == 0 {
            first_secant
        } else if i == k - 2 {
            last_secant
        } else {
            d[i + 1]
        }
    };
    let three = T::from_f64(3.0);
    let mut lower = vec![T::ZERO; k - 1];
    let mut diag = vec![T::ZERO; k];
    let mut upper = vec![T::ZERO; k - 1];
    let mut rhs = vec![T::ZERO; k];
    for i in 1..k - 1 {
        lower[i - 1] = length(i);
        diag[i] = T::TWO * (length(i - 1) + length(i));
        upper[i] = length(i - 1);
        rhs[i] = three * (length(i) * secant(i - 1) + length(i - 1) * secant(i));
    }
    // The end cubics through y[1] and y[n−2], each row multiplied by the length of its piece so
    // that its entries are lengths like those of the others.
    let (u, v) = (h[0] / first, h[1] / first);
    diag[0] = h[1];
    upper[0] = -h[0];
    rhs[0] = h[1] * (T::ONE + T::TWO * u) * d[0] - h[0] * (T::ONE + T::TWO * v) * d[1];
    let (u, v) = (h[n - 3] / last, h[n - 2] / last);
    lower[k - 2] = h[n - 2];
    diag[k - 1] = -h[n - 3];
    rhs[k - 1] =
        h[n - 2] * (T::ONE + T::TWO * u) * d[n - 3] - h[n - 3] * (T::ONE + T::TWO * v) * d[n - 2];
    let reduced = solve_tridiagonal_scaled(lower, diag, upper, rhs)?;
    let mut m = vec![T::ZERO; n];
    for (i, &s) in reduced.iter().enumerate() {
        m[knot(i)] = s;
    }
    m[1] = inner_slope(h[0], h[1], first, first_secant, m[0], m[2]);
    m[n - 2] = inner_slope(h[n - 3], h[n - 2], last, last_secant, m[n - 3], m[n - 1]);
    Some(m)
}

/// The slope, at the point a length `left` from its start and `right` from its end, of the cubic
/// of a piece `length` long with secant `secant` and the end slopes `start` and `end`: the
/// derivative of the Hermite form, `6uv·secant + v(v − 2u)·start + u(u − 2v)·end` with `u` and
/// `v` the two lengths as fractions of the piece.
fn inner_slope<T: Real>(left: T, right: T, length: T, secant: T, start: T, end: T) -> T {
    let (u, v) = (left / length, right / length);
    T::from_f64(6.0) * u * v * secant + v * (v - T::TWO * u) * start + u * (u - T::TWO * v) * end
}

/// The slopes at its four knots of the cubic through four points, from the divided differences
/// taken from each knot outwards, nearest knots first: `p′(x[k]) = f[k, a] + f[k, a, b]·(x[k] −
/// x[a]) + f[0, 1, 2, 3]·(x[k] − x[a])·(x[k] − x[b])`, with `a` the knot nearest to `x[k]` and `b`
/// the next (the lower index first between two as near). A short interval then enters the slopes
/// at its ends through its secant, which is what the data determine there, rather than through a
/// difference of two rows that both carry it; `d` holds the secants of the three intervals.
fn cubic_through_four<T: Real>(x: &[T], y: &[T], d: &[T]) -> Vec<T> {
    let first = |i: usize, j: usize| {
        let (a, b) = (i.min(j), i.max(j));
        if b == a + 1 {
            d[a]
        } else {
            (y[b] - y[a]) / (x[b] - x[a])
        }
    };
    let second = |i: usize, j: usize, k: usize| {
        let mut s = [i, j, k];
        s.sort_unstable();
        (first(s[1], s[2]) - first(s[0], s[1])) / (x[s[2]] - x[s[0]])
    };
    let third = (second(1, 2, 3) - second(0, 1, 2)) / (x[3] - x[0]);
    (0..4)
        .map(|k| {
            let mut others: Vec<usize> = (0..4).filter(|&j| j != k).collect();
            // A stable sort: of two knots as far from x[k], the one of lower index comes first.
            others.sort_by(|&i, &j| {
                (x[i] - x[k])
                    .abs()
                    .partial_cmp(&(x[j] - x[k]).abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            let (a, b) = (others[0], others[1]);
            first(k, a) + second(k, a, b) * (x[k] - x[a]) + third * (x[k] - x[a]) * (x[k] - x[b])
        })
        .collect()
}

/// The solution of the tridiagonal system with sub-diagonal `lower` (`lower[i]` in row `i + 1`),
/// diagonal `diag` and super-diagonal `upper` (`upper[i]` in row `i`), by Gaussian elimination
/// with scaled partial pivoting: at each step the pivot is the row whose entry in the column is
/// the larger beside the largest entry of its own row as given (a row left over from an
/// interchange keeps the size of the row it came from), and an interchange fills in a second
/// super-diagonal (G. H. Golub and C. F. Van Loan, *Matrix Computations*, sections 3.4 and 4.3).
///
/// Plain partial pivoting compares the entries themselves, which for the spline's rows are
/// lengths: where a long interval follows short ones it takes the row of the next knot as the
/// pivot for the slope of this one, in which that slope is only an off-diagonal term, and each
/// such step doubles the error of the slopes before it (1.5·10⁵ eps over knots whose intervals
/// range from 10⁻³ to 10³). Measured against the row sizes, the diagonally dominant rows of the
/// interior knots never trade places and only the two end rows can, as they need to.
///
/// Returns `None` on a singular system.
fn solve_tridiagonal_scaled<T: Real>(
    mut lower: Vec<T>,
    mut diag: Vec<T>,
    mut upper: Vec<T>,
    mut rhs: Vec<T>,
) -> Option<Vec<T>> {
    let n = diag.len();
    // The size of a row as it is given: the largest of its entries.
    let size_of = |i: usize, diag: &[T], lower: &[T], upper: &[T]| {
        let mut s = diag[i].abs();
        if i > 0 {
            s = s.max(lower[i - 1].abs());
        }
        if i + 1 < n {
            s = s.max(upper[i].abs());
        }
        s
    };
    if (0..n).any(|i| size_of(i, &diag, &lower, &upper) == T::ZERO) {
        return None;
    }
    // The sizes of rows i and i + 1, kept as the elimination goes: row i + 1 is still as given
    // when its turn comes, unless an interchange put the old row i, and its size, there.
    let mut size_i = size_of(0, &diag, &lower, &upper);
    // The second super-diagonal, filled in by the interchanges.
    let mut fill = vec![T::ZERO; n.saturating_sub(2)];
    for i in 0..n - 1 {
        let size_next = size_of(i + 1, &diag, &lower, &upper);
        let mut carried = size_next;
        if diag[i].abs() / size_i >= lower[i].abs() / size_next {
            if diag[i] == T::ZERO {
                return None;
            }
            let f = lower[i] / diag[i];
            let (above, pivot_rhs) = (upper[i], rhs[i]);
            diag[i + 1] -= f * above;
            rhs[i + 1] -= f * pivot_rhs;
        } else {
            // Rows i and i + 1 trade places: the old row i + 1 is the pivot row, and what is left
            // of the old row i after eliminating with it is the new row i + 1.
            let f = diag[i] / lower[i];
            diag[i] = lower[i];
            let below = diag[i + 1];
            diag[i + 1] = upper[i] - f * below;
            if i + 2 < n {
                fill[i] = upper[i + 1];
                upper[i + 1] = -f * upper[i + 1];
            }
            upper[i] = below;
            let pivot_rhs = rhs[i + 1];
            rhs[i + 1] = rhs[i] - f * pivot_rhs;
            rhs[i] = pivot_rhs;
            carried = size_i;
        }
        lower[i] = T::ZERO;
        size_i = carried;
    }
    if diag[n - 1] == T::ZERO {
        return None;
    }
    // The back substitution in place of the right-hand side: each element is read before its
    // solution is written there.
    let x = &mut rhs;
    x[n - 1] /= diag[n - 1];
    if n >= 2 {
        x[n - 2] = (x[n - 2] - upper[n - 2] * x[n - 1]) / diag[n - 2];
    }
    for i in (0..n.saturating_sub(2)).rev() {
        x[i] = (x[i] - upper[i] * x[i + 1] - fill[i] * x[i + 2]) / diag[i];
    }
    Some(rhs)
}

/// The `count` nodes of Chebyshev interpolation on `[a, b]`: the zeros of the Chebyshev
/// polynomial `T[count]`, mapped onto the interval and returned in increasing order — strictly
/// so wherever the floats of the interval hold them apart: on an interval only a few floats wide
/// (50 nodes on `[1e6, 1e6 + 1]` in `f32`, where the floats are `1/16` apart) neighbours can round
/// to the same float, and [`barycentric_weights`] then panics. The polynomial of degree
/// `count − 1` through these nodes is within a logarithmic factor of the best there is.
///
/// # Panics
///
/// If `count` is zero, or `a` and `b` are not finite with `a` below `b`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::chebyshev_nodes;
///
/// // The three nodes of a quadratic on [-1, 1] are the zeros of T₃, in increasing order.
/// let x = chebyshev_nodes(3, -1.0, 1.0);
/// assert!((x[0] + (core::f64::consts::FRAC_PI_6).cos()).abs() < 1e-15);
/// assert!((x[1]).abs() < 1e-15);
/// assert!(x[0] < x[1] && x[1] < x[2]);
/// ```
#[track_caller]
pub fn chebyshev_nodes<T: Real>(count: usize, a: T, b: T) -> Vec<T> {
    assert!(
        count > 0,
        "a Chebyshev interpolation needs at least one node"
    );
    assert!(
        a.is_finite() && b.is_finite() && a < b,
        "the interval must be finite and increasing"
    );
    let half = (b - a) * T::HALF;
    let mid = a + half;
    let n = T::from_f64(count as f64);
    (0..count)
        .map(|k| {
            let k = T::from_f64(k as f64);
            // The zero −cos θ of T[count] is mapped onto the interval, which reverses the order
            // the zeros come in and so leaves them increasing.
            mid - half * (T::PI * (T::TWO * k + T::ONE) / (T::TWO * n)).cos()
        })
        .collect()
}

/// The barycentric weights of the nodes `x`, proportional to `1 / Π_{j≠k}(x[k] − x[j])`, which
/// [`barycentric_interpolate`] needs. The classical node sets have closed forms, but this general
/// product is what arbitrary data need; it costs a pass over the pairs of nodes.
///
/// The weights matter only up to a common factor, and the products themselves leave the range of
/// the type for a hundred or so nodes on an interval much shorter or longer than 4 (J.-P. Berrut
/// and L. N. Trefethen, *SIAM Review* 46, 2004, scale the differences by `4/(max x − min x)` for
/// that reason). Here each difference is split exactly into a significand in `[1, 2)` and a power
/// of two, the significands are multiplied and the powers added, so no product can overflow or
/// underflow for any number of nodes; the weights are returned divided by the power of two of
/// the largest, which leaves them at most 1 in size. A weight too small to be represented beside
/// the largest one is negligible in the formula and comes out as zero.
///
/// # Panics
///
/// If `x` is empty, holds a node that is not finite, or holds two equal nodes.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{barycentric_interpolate, barycentric_weights, chebyshev_nodes};
///
/// // The weights of three equally spaced nodes alternate, the middle one twice the others.
/// let w = barycentric_weights(&[0.0, 1.0, 2.0]);
/// assert_eq!(w[0], w[2]);
/// assert_eq!(w[1], -2.0 * w[0]);
///
/// // 200 Chebyshev nodes on [0, 1000], where the plain products would overflow: the
/// // interpolant of cos(t/300) is right to rounding.
/// let x = chebyshev_nodes(200, 0.0f64, 1000.0);
/// let y: Vec<f64> = x.iter().map(|t| (t / 300.0).cos()).collect();
/// let w = barycentric_weights(&x);
/// let t = 402.419;
/// assert!((barycentric_interpolate(&x, &y, &w, t) - (t / 300.0).cos()).abs() < 1e-14);
/// ```
#[track_caller]
pub fn barycentric_weights<T: Real>(x: &[T]) -> Vec<T> {
    assert!(!x.is_empty(), "there must be at least one node");
    assert!(x.iter().all(|v| v.is_finite()), "the nodes must be finite");
    let n = x.len();
    let mut product = vec![T::ONE; n];
    let mut exponent = vec![0i64; n];
    for k in 0..n {
        let (mut p, mut e) = (T::ONE, 0i64);
        for j in 0..n {
            if j == k {
                continue;
            }
            let mut difference = x[k] - x[j];
            let mut doubled = 0;
            if difference.is_infinite() {
                // Nodes near both ends of the range: the difference of the halves, which are
                // exact there, and one more power of two.
                difference = x[k] * T::HALF - x[j] * T::HALF;
                doubled = 1;
            }
            assert!(difference != T::ZERO, "two nodes are equal");
            let (m, f) = split(difference);
            // Both significands lie in [1, 2), so the product lies in [1, 4) and one halving
            // at most brings it back.
            p *= m;
            e += f + doubled;
            if p.abs() >= T::TWO {
                p *= T::HALF;
                e += 1;
            }
        }
        product[k] = p;
        exponent[k] = e;
    }
    // w[k] = 2^(−e[k]) / p[k] with 1/p[k] in (1/2, 1]; divided by 2^(−min e), the weight of the
    // smallest product keeps its power 0 and the others have negative ones.
    let lowest = exponent.iter().copied().min().unwrap_or(0);
    product
        .iter()
        .zip(&exponent)
        .map(|(&p, &e)| scale_by_power_of_two(T::ONE / p, lowest - e))
        .collect()
}

/// `v = m·2^e` with `1 ≤ |m| < 2`, for a finite `v` other than zero; exact, since the significand
/// keeps the bits of `v` (an `f32` is held exactly by an `f64`, and a subnormal `f64` is brought
/// into the normal range first).
fn split<T: Real>(v: T) -> (T, i64) {
    let mut f = v.to_f64();
    let mut e = 0i64;
    if f.abs() < f64::MIN_POSITIVE {
        f *= 2f64.powi(64);
        e -= 64;
    }
    let bits = f.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i64;
    e += biased - 1023;
    // The sign and the fraction of `f` with the exponent of 1: the significand in [1, 2).
    let m = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1023_u64 << 52));
    (T::from_f64(m), e)
}

/// `v·2^k` for `k ≤ 0`, in steps the type can hold, so that a result that underflows does so
/// only once and not through an intermediate power of two that is zero.
fn scale_by_power_of_two<T: Real>(mut v: T, mut k: i64) -> T {
    while k < 0 && v != T::ZERO {
        let step = k.max(-64);
        v *= T::TWO.powi(step as i32);
        k -= step;
    }
    v
}

/// The value at `t` of the polynomial of degree below `x.len()` that takes the values `y` at the
/// nodes `x`, by the second form of the barycentric formula,
/// `p(t) = Σ w[k]·y[k]/(t − x[k]) ÷ Σ w[k]/(t − x[k])`.
///
/// `w` must be the weights of `x`, from [`barycentric_weights`]: the formula is one pass once
/// they are known, which is why they are kept apart so that a node set can serve many data sets.
/// At a node, and so close to one that its term overflows, the value is the datum there. The
/// differences `t − x[k]` are measured in units of the span of the nodes, so that nodes across
/// the whole range of the type lose no digits to terms below the smallest normal number.
///
/// # Panics
///
/// If `x` is empty, or `y` or `w` is not as long as `x`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{barycentric_interpolate, barycentric_weights};
///
/// // The polynomial through (0, 1), (1, 2) and (2, 5) is 1 + x², which at 3 is 10.
/// let x = [0.0f64, 1.0, 2.0];
/// let y = [1.0f64, 2.0, 5.0];
/// let w = barycentric_weights(&x);
/// assert!((barycentric_interpolate(&x, &y, &w, 3.0) - 10.0).abs() < 1e-12);
/// // A subnormal away from the node 0 the value is the datum there.
/// assert_eq!(barycentric_interpolate(&x, &y, &w, 5e-324), 1.0);
/// ```
#[track_caller]
pub fn barycentric_interpolate<T: Real>(x: &[T], y: &[T], w: &[T], t: T) -> T {
    assert_eq!(
        x.len(),
        y.len(),
        "the nodes and the values differ in length"
    );
    assert!(!x.is_empty(), "there must be at least one node");
    assert_eq!(
        w.len(),
        x.len(),
        "the weights differ in length from the nodes"
    );
    // The terms may share any factor: the differences are taken in units of the power of two of
    // half the span of the nodes. That leaves ordinary terms as they are, bit for bit, and keeps
    // them above the smallest normal float for nodes that span the range of the type, where they
    // would lose their digits; a difference that overflows is taken of the halves.
    let (lo, hi) = x
        .iter()
        .fold((x[0], x[0]), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let half = hi * T::HALF - lo * T::HALF;
    let e = if half > T::ZERO && half.is_finite() {
        split(half).1 as i32
    } else {
        0
    };
    let unit = T::TWO.powi(e / 2) * T::TWO.powi(e - e / 2);
    let mut numerator = T::ZERO;
    let mut denominator = T::ZERO;
    for k in 0..x.len() {
        let difference = t - x[k];
        if difference == T::ZERO {
            return y[k];
        }
        let difference = if difference.is_infinite() && t.is_finite() {
            (t * T::HALF - x[k] * T::HALF) / unit * T::TWO
        } else {
            difference / unit
        };
        let term = w[k] / difference;
        if term.is_infinite() {
            // `t` is so close to the node that the polynomial is its datum to the last bit.
            return y[k];
        }
        numerator += term * y[k];
        denominator += term;
    }
    numerator / denominator
}

/// The Chebyshev series `Σ c[k]·T[k](u(t))` of a function on `[a, b]`, where `u` maps `[a, b]`
/// onto `[-1, 1]`. Build one with [`chebyshev`].
///
/// It is a polynomial: [`evaluate`](Self::evaluate), [`derivative`](Self::derivative) and
/// [`integral`](Self::integral) answer anywhere, and outside `[a, b]` that is the polynomial
/// continued, an extrapolation, not an approximation of the function.
///
/// # Examples
///
/// ```
/// use a::math::numerics::chebyshev;
///
/// let c = chebyshev(|x: f64| x.exp(), 0.0, 1.0, 12).unwrap();
/// assert!((c.evaluate(0.5) - 0.5f64.exp()).abs() < 1e-13);
/// // Just outside the interval the polynomial still follows the function; far out it does not.
/// assert!((c.evaluate(1.1) - 1.1f64.exp()).abs() < 1e-10);
/// assert!((c.evaluate(10.0) - 10.0f64.exp()).abs() > 1.0);
/// ```
#[derive(Clone)]
pub struct Chebyshev<T> {
    a: T,
    b: T,
    c: Vec<T>,
    /// The series of the derivative, formed on the first call of `derivative` and kept: it is
    /// the same for every point, and forming it again for each was most of the cost of one.
    derivative: OnceLock<Vec<T>>,
    /// The series of the integral and its value at `a`, kept the same way.
    integral: OnceLock<(Vec<T>, T)>,
}

impl<T: core::fmt::Debug> core::fmt::Debug for Chebyshev<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Chebyshev")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("c", &self.c)
            .finish()
    }
}

/// Two fits are equal when their intervals and coefficients are; the series kept for the
/// derivative and the integral follow from those.
impl<T: PartialEq> PartialEq for Chebyshev<T> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }
}

impl<T: Real> Chebyshev<T> {
    /// The coefficients of the series, the constant term first.
    pub fn coefficients(&self) -> &[T] {
        &self.c
    }

    /// The degree of the series.
    pub fn degree(&self) -> usize {
        self.c.len().saturating_sub(1)
    }

    /// The interval the series was fitted to.
    pub fn interval(&self) -> (T, T) {
        (self.a, self.b)
    }

    /// The value at `t`, by Clenshaw's backward recurrence.
    pub fn evaluate(&self, t: T) -> T {
        clenshaw(&self.c, map_in(t, self.a, self.b))
    }

    /// The derivative at `t`, from the coefficients of the derivative series and Clenshaw again.
    ///
    /// If `p = Σ c[k]·T[k]` and `p′ = Σ d[k]·T[k]`, then `d[k−1] = d[k+1] + 2k·c[k]`, run
    /// downwards from two vanishing coefficients at the top, with the constant term halved.
    pub fn derivative(&self, t: T) -> T {
        let n = self.c.len();
        if n < 2 {
            return T::ZERO;
        }
        let d = self.derivative.get_or_init(|| {
            let mut d = vec![T::ZERO; n];
            // The top coefficient of the derivative series vanishes, and with it the one above.
            d[n - 2] = T::from_f64(2.0 * (n - 1) as f64) * self.c[n - 1];
            for k in (1..n - 1).rev() {
                d[k - 1] = d[k + 1] + T::from_f64(2.0 * k as f64) * self.c[k];
            }
            d[0] *= T::HALF;
            d
        });
        clenshaw(d, map_in(t, self.a, self.b)) / ((self.b - self.a) * T::HALF)
    }

    /// The integral from `a` to `t`, term by term.
    pub fn integral(&self, t: T) -> T {
        let n = self.c.len();
        if n == 0 {
            return T::ZERO;
        }
        let (e, at_a) = self.integral.get_or_init(|| {
            // ∫T[0] = T[1], ∫T[1] = T[0]/4 + T[2]/4 and, for k at least 2,
            // ∫T[k] = (T[k+1]/(k+1) − T[k−1]/(k−1))/2. The constant term is settled below.
            let mut e = vec![T::ZERO; n + 2];
            e[1] += self.c[0];
            if n > 1 {
                e[2] += self.c[1] * T::from_f64(0.25);
                e[0] += self.c[1] * T::from_f64(0.25);
            }
            for k in 2..n {
                let kf = T::from_f64(k as f64);
                e[k + 1] += self.c[k] * T::HALF / (kf + T::ONE);
                e[k - 1] -= self.c[k] * T::HALF / (kf - T::ONE);
            }
            let at_a = clenshaw(&e, -T::ONE);
            (e, at_a)
        });
        let scale = (self.b - self.a) * T::HALF;
        (clenshaw(e, map_in(t, self.a, self.b)) - *at_a) * scale
    }
}

/// The Chebyshev series of `f` on `[a, b]` of the given `degree`, fitted at the `degree + 1`
/// Chebyshev nodes of [`chebyshev_nodes`], which makes it the interpolating polynomial of that
/// degree — the construction of `numpy.polynomial.chebyshev.Chebyshev.interpolate`. The
/// coefficients are the discrete cosine transform of the values, summed directly: the fit costs
/// `degree + 1` evaluations of `f` and `(degree + 1)²` cosines (`10⁸` at degree `10⁴`). The degree
/// is the caller's: nothing here chooses it or estimates the error; the size of the last few
/// coefficients is the usual guide. Degree 0 is the constant `f((a + b)/2)`.
///
/// Returns `None` when `a` and `b` are not finite with `a` below `b`, or a value of `f` is not
/// finite.
///
/// # Panics
///
/// If `degree` is `usize::MAX`: `degree + 1` nodes cannot be counted.
///
/// # Examples
///
/// ```
/// use a::math::numerics::chebyshev;
///
/// // Runge's function, which a polynomial through equally spaced nodes cannot follow at all. Its
/// // poles at ±0.2i bound the error of the best polynomial of degree n at ρ⁻ⁿ for ρ ≈ 1.22, so
/// // the degree has to be high: 60 buys six digits, where 30 buys barely three.
/// let c = chebyshev(|x: f64| 1.0 / (1.0 + 25.0 * x * x), -1.0, 1.0, 60).unwrap();
/// assert!((c.evaluate(0.9) - 1.0 / (1.0 + 25.0 * 0.81)).abs() < 1e-5);
///
/// // The integral of the fit is the integral of the function, to the fit's accuracy.
/// let c = chebyshev(|x: f64| x.exp(), 0.0, 1.0, 14).unwrap();
/// assert!((c.integral(1.0) - (core::f64::consts::E - 1.0)).abs() < 1e-14);
///
/// // A reversed interval has no fit.
/// assert!(chebyshev(|x: f64| x, 1.0, 0.0, 3).is_none());
/// ```
#[track_caller]
pub fn chebyshev<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    degree: usize,
) -> Option<Chebyshev<T>> {
    let count = degree
        .checked_add(1)
        .expect("degree + 1 nodes must be countable");
    if !(a.is_finite() && b.is_finite() && a < b) {
        return None;
    }
    let nodes = chebyshev_nodes(count, a, b);
    let mut values = Vec::with_capacity(count);
    for &node in &nodes {
        let value = f(node);
        if !value.is_finite() {
            return None;
        }
        values.push(value);
    }
    // The discrete cosine transform at the Gauss nodes: the orthogonality of the cosines there
    // gives the constant coefficient as the mean of the values and every other one as twice its
    // projection. The nodes come from −cos θ, and T[k](−u) is (−1)^k·T[k](u), which is what the
    // parity of k carries below.
    let n = T::from_f64(count as f64);
    let c = if count <= DIRECT_TRANSFORM {
        // Summed directly, the angle k·θ_j = π·k·(2j + 1)/(2n) reduced modulo 2π in integers
        // before the cosine is taken.
        let period = 4 * count;
        let mut c = Vec::with_capacity(count);
        for k in 0..count {
            let mut sum = T::ZERO;
            for (j, &value) in values.iter().enumerate() {
                let turns = (k * (2 * j + 1)) % period;
                let t = (T::PI * T::from_f64(turns as f64) / (T::TWO * n)).cos();
                sum += value * if k % 2 == 0 { t } else { -t };
            }
            c.push(if k == 0 { sum / n } else { T::TWO * sum / n });
        }
        c
    } else {
        // The DCT-II of `a::math::fft`, `Y[k] = 2·Σ_j v_j·cos(k·θ_j)`, in `O(n log n)`.
        Dct::<T>::new(count, DctKind::II).transform(&mut values);
        values
            .iter()
            .enumerate()
            .map(|(k, &y)| {
                let y = if k % 2 == 0 { y } else { -y };
                if k == 0 { y / (T::TWO * n) } else { y / n }
            })
            .collect()
    };
    Some(Chebyshev {
        a,
        b,
        c,
        derivative: OnceLock::new(),
        integral: OnceLock::new(),
    })
}

/// `t` mapped from `[a, b]` onto `[-1, 1]`.
fn map_in<T: Real>(t: T, a: T, b: T) -> T {
    ((t - a) / (b - a) - T::HALF) * T::TWO
}

/// The value at `u` of the Chebyshev series with coefficients `c`, by Clenshaw's recurrence:
/// `b[k−1] = 2u·b[k] − b[k+1] + c[k]`, and the value `u·b[0] − b[1] + c[0]`.
fn clenshaw<T: Real>(c: &[T], u: T) -> T {
    let two_u = T::TWO * u;
    let mut b1 = T::ZERO;
    let mut b2 = T::ZERO;
    for k in (1..c.len()).rev() {
        let b = two_u * b1 - b2 + c[k];
        b2 = b1;
        b1 = b;
    }
    u * b1 - b2 + c[0]
}
