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
//! | [`Spline::akima`] | Akima's five-point weights | `C¹` | little |
//! | [`Spline::pchip`] | the Fritsch and Carlson limiter | `C¹` | never between monotone knots |
//!
//! Where the function is smooth across the whole interval one polynomial is better, and the one
//! that keeps the worst error small is best approximated by the polynomial that agrees with the
//! function at the Chebyshev nodes: [`chebyshev`] fits it, and evaluates it by Clenshaw's
//! recurrence in time linear in the degree. [`barycentric_weights`] and
//! [`barycentric_interpolate`] evaluate the same interpolant from the values at the nodes
//! themselves, which is what to use when the values are data rather than a function to be sampled.
//!
//! Every constructor panics on knots that are not strictly increasing, and an interpolation
//! outside the knots returns `None` rather than extrapolating.

use super::Real;

/// The piecewise cubic through the knots `(x[i], y[i])`, held in Hermite form. Build one with
/// [`Spline::cubic`], [`Spline::cubic_clamped`], [`Spline::cubic_not_a_knot`], [`Spline::akima`]
/// or [`Spline::pchip`], and ask it for [`evaluate`](Self::evaluate) and
/// [`derivative`](Self::derivative).
///
/// # Examples
///
/// ```
/// use a::math::numerics::Spline;
///
/// // The sine through 33 knots on [0, 1]. Between the knots the value is good to nine digits and
/// // the slope to seven; the natural condition at the ends costs two of the four digits that a
/// // spline of degree three otherwise earns, which is what [`Spline::cubic_not_a_knot`] is for.
/// let x: Vec<f64> = (0..33).map(|k| f64::from(k) / 32.0).collect();
/// let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();
/// let s = Spline::cubic(&x, &y).unwrap();
/// assert!((s.evaluate(0.731).unwrap() - 0.731f64.sin()).abs() < 1e-8);
/// assert!((s.derivative(0.731).unwrap() - 0.731f64.cos()).abs() < 1e-6);
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
    /// end of the interval belongs to it.
    fn interval(&self, t: T) -> Option<usize> {
        let n = self.x.len();
        if n < 2 || !(t >= self.x[0] && t <= self.x[n - 1]) {
            return None;
        }
        let mut i = 0;
        while i + 2 < n && self.x[i + 1] <= t {
            i += 1;
        }
        Some(i)
    }

    /// The value of the spline at `t`, or `None` when `t` is outside the knots.
    pub fn evaluate(&self, t: T) -> Option<T> {
        let i = self.interval(t)?;
        let h = self.x[i + 1] - self.x[i];
        let u = (t - self.x[i]) / h;
        let v = T::ONE - u;
        // The Hermite basis: h00 carries the left value, h10 the left slope times the interval,
        // h01 the right value and h11 the right slope times the interval.
        let h00 = (T::TWO * u + T::ONE) * v * v;
        let h10 = u * v * v;
        let h01 = u * u * (T::from_f64(3.0) - T::TWO * u);
        let h11 = u * u * (u - T::ONE);
        Some(
            h00 * self.y[i]
                + h10 * (self.m[i] * h)
                + h01 * self.y[i + 1]
                + h11 * (self.m[i + 1] * h),
        )
    }

    /// The derivative of the spline at `t`, or `None` when `t` is outside the knots.
    pub fn derivative(&self, t: T) -> Option<T> {
        let i = self.interval(t)?;
        let h = self.x[i + 1] - self.x[i];
        let u = (t - self.x[i]) / h;
        let v = T::ONE - u;
        let dh00 = -T::from_f64(6.0) * u * v;
        let dh10 = v * (v - T::TWO * u);
        let dh01 = T::from_f64(6.0) * u * v;
        let dh11 = u * (T::from_f64(3.0) * u - T::TWO);
        Some(
            (dh00 * self.y[i]
                + dh10 * (self.m[i] * h)
                + dh01 * self.y[i + 1]
                + dh11 * (self.m[i + 1] * h))
                / h,
        )
    }

    /// The spline of `y` at the strictly increasing knots `x` whose second derivative vanishes at
    /// both ends, the natural spline. It is the smoothest of the five and the one to reach for
    /// when nothing is known about the ends.
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
    /// ```
    pub fn cubic(x: &[T], y: &[T]) -> Option<Self> {
        Self::build(x, y, End::Natural, T::ZERO, T::ZERO)
    }

    /// The spline of `y` at the knots `x` whose slope at the first knot is `left` and at the last
    /// `right` — the clamped spline, better than the natural one where the end slopes are known,
    /// since the two extra conditions are worth two knots of accuracy.
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
    /// ```
    pub fn cubic_clamped(x: &[T], y: &[T], left: T, right: T) -> Option<Self> {
        Self::build(x, y, End::Clamped, left, right)
    }

    /// The spline of `y` at the knots `x` whose third derivative is continued across the second
    /// and penultimate knots — the not-a-knot spline, the usual choice when the end behaviour of
    /// the function is unknown.
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
    /// ```
    pub fn cubic_not_a_knot(x: &[T], y: &[T]) -> Option<Self> {
        Self::build(x, y, End::NotAKnot, T::ZERO, T::ZERO)
    }

    /// Akima's spline of `y` at the knots `x`: the slope at a knot is a weighted mean of the two
    /// secants that meet at it, weighted by how far the two further secants are from them, so that
    /// a knot in the middle of a straight run takes the slope of that run exactly. Nothing is
    /// solved, so it costs what the secants cost, and it swings far less than the cubic spline
    /// near a jump; it is only `C¹`.
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
    /// ```
    pub fn akima(x: &[T], y: &[T]) -> Option<Self> {
        check(x, y);
        let n = x.len();
        let mut m = vec![T::ZERO; n];
        if n == 2 {
            let d = (y[1] - y[0]) / (x[1] - x[0]);
            m[0] = d;
            m[1] = d;
            return Some(Self {
                x: x.to_vec(),
                y: y.to_vec(),
                m,
            });
        }
        // The secants d[k] of the interval beginning at x[k], extended two past each end by the
        // reflection Akima's own end condition asks for, so that every knot has five of them.
        let mut d = vec![T::ZERO; n + 4];
        for k in 0..n - 1 {
            d[k + 2] = (y[k + 1] - y[k]) / (x[k + 1] - x[k]);
        }
        d[1] = T::TWO * d[2] - d[3];
        d[0] = T::TWO * d[1] - d[2];
        d[n + 2] = T::TWO * d[n + 1] - d[n];
        d[n + 3] = T::TWO * d[n + 2] - d[n + 1];
        for i in 0..n {
            m[i] = akima_slope(&d[i..i + 5]);
        }
        Some(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }

    /// The shape-preserving spline of `y` at the knots `x`, whose slopes are the harmonic mean of
    /// the secants on either side and zero at a turning point: between two knots it never leaves
    /// their range, so a peak in the data is a peak in the spline and a flat run stays flat.
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
    /// ```
    pub fn pchip(x: &[T], y: &[T]) -> Option<Self> {
        check(x, y);
        let n = x.len();
        let mut m = vec![T::ZERO; n];
        if n == 2 {
            let d = (y[1] - y[0]) / (x[1] - x[0]);
            m[0] = d;
            m[1] = d;
            return Some(Self {
                x: x.to_vec(),
                y: y.to_vec(),
                m,
            });
        }
        let d: Vec<T> = (0..n - 1)
            .map(|k| (y[k + 1] - y[k]) / (x[k + 1] - x[k]))
            .collect();
        let h: Vec<T> = (0..n - 1).map(|k| x[k + 1] - x[k]).collect();
        for i in 1..n - 1 {
            // The harmonic mean of the two secants, weighted by the lengths of the intervals on
            // either side (Witten's form), which lies between them and so cannot overshoot; zero
            // where the data turn.
            m[i] = if d[i - 1] * d[i] <= T::ZERO {
                T::ZERO
            } else {
                let (hl, hr) = (h[i - 1], h[i]);
                let (dl, dr) = (d[i - 1], d[i]);
                T::from_f64(3.0) * (hl + hr) * dl * dr
                    / ((hl + T::TWO * hr) * dr + (T::TWO * hl + hr) * dl)
            };
        }
        m[0] = pchip_end(&h, &d, true);
        m[n - 1] = pchip_end(&h, &d, false);
        Some(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }

    /// The cubic spline in Hermite form, from the slopes its end condition asks for. The interior
    /// slopes come from the tridiagonal system that makes the second derivative continuous, and
    /// the two end rows carry the end condition itself with the end slopes eliminated.
    fn build(x: &[T], y: &[T], end: End, left: T, right: T) -> Option<Self> {
        check(x, y);
        let n = x.len();
        let mut m = vec![T::ZERO; n];
        if n == 2 {
            let d = (y[1] - y[0]) / (x[1] - x[0]);
            m[0] = d;
            m[1] = d;
            return Some(Self {
                x: x.to_vec(),
                y: y.to_vec(),
                m,
            });
        }
        let h: Vec<T> = (0..n - 1).map(|k| x[k + 1] - x[k]).collect();
        let d: Vec<T> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / h[k]).collect();
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
            // 1/h[i+1] on both sides of the diagonal, so the system is symmetrizable, and it is
            // this one that makes the second derivative continuous.
            lower[i] = h[i + 1];
            centre[i] = T::TWO * (h[i] + h[i + 1]);
            upper[i] = h[i];
            rhs[i] = T::from_f64(3.0) * (h[i + 1] * d[i] + h[i] * d[i + 1]);
        }
        // The two ratios the not-a-knot relations are written with.
        let (r, sl) = (
            (h[0] / h[1]) * (h[0] / h[1]),
            (h[k - 1] / h[k]) * (h[k - 1] / h[k]),
        );
        match end {
            End::Natural => natural_rows(&mut lower, &mut centre, &mut upper, &mut rhs, &h, &d),
            End::NotAKnot => {
                not_a_knot_rows(&mut lower, &mut centre, &mut upper, &mut rhs, &h, &d, r, sl);
            }
            End::Clamped => {
                // The end slopes are given, so their terms move to the right-hand side: the first
                // row carries h[0]·m[0] and the last h[n−2]·m[n−1].
                rhs[0] -= lower[0] * left;
                rhs[k - 1] -= upper[k - 1] * right;
                m[0] = left;
                m[n - 1] = right;
            }
        }
        solve_tridiagonal(&lower, &mut centre, &upper, &mut rhs, &mut m[1..n - 1])?;
        // The end slopes themselves, from the two relations that eliminated them.
        match end {
            End::Natural => {
                m[0] = (T::from_f64(3.0) * d[0] - m[1]) * T::HALF;
                m[n - 1] = (T::from_f64(3.0) * d[k] - m[n - 2]) * T::HALF;
            }
            End::NotAKnot => {
                m[0] = T::TWO * d[0] - T::TWO * r * d[1] - (T::ONE - r) * m[1] + r * m[2];
                let inv = T::ONE / sl;
                m[n - 1] = inv
                    * (m[n - 3] + (T::ONE - sl) * m[n - 2] - T::TWO * d[k - 1]
                        + T::TWO * sl * d[k]);
            }
            End::Clamped => {}
        }
        Some(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }
}

/// How the two end rows of the spline's tridiagonal system are filled.
#[derive(Debug, Clone, Copy, PartialEq)]
enum End {
    Natural,
    Clamped,
    NotAKnot,
}

/// The arguments every constructor checks.
///
/// # Panics
///
/// Always, on knots that are not equally long, fewer than two, or not strictly increasing.
#[track_caller]
fn check<T: Real>(x: &[T], y: &[T]) {
    assert_eq!(
        x.len(),
        y.len(),
        "the knots and the values differ in length"
    );
    assert!(x.len() >= 2, "a spline needs at least two knots");
    assert!(
        x.windows(2).all(|w| w[0] < w[1]),
        "the knots must be strictly increasing"
    );
}

/// Akima's slope at a knot, from the five secants `d[0..5]` of which `d[1]` ends and `d[2]`
/// begins at the knot: each is weighted by the difference of the two secants beyond the other, so
/// that a secant agreeing with its neighbour decides the slope on its own. Where no secant
/// distinguishes the two sides the mean of the two is used.
fn akima_slope<T: Real>(d: &[T]) -> T {
    // Akima's own weights: the secant on one side of the knot is weighted by how far the two
    // secants on the other side are from each other, so that a straight run decides the slope.
    let (w1, w2) = ((d[3] - d[2]).abs(), (d[0] - d[1]).abs());
    if w1 + w2 == T::ZERO {
        (d[1] + d[2]) * T::HALF
    } else {
        (w1 * d[1] + w2 * d[2]) / (w1 + w2)
    }
}

/// The slope at an end of the monotone spline: the one-sided derivative of the quadratic through
/// the three nearest knots, kept only when it points the way the nearest secant does — Fritsch
/// and Carlson's safeguard, which puts zero where the data turn and twice the secant where the
/// quadratic would overshoot it.
fn pchip_end<T: Real>(h: &[T], d: &[T], at_left: bool) -> T {
    let (hl, hr, dl, dr) = if at_left {
        (h[0], h[1], d[0], d[1])
    } else {
        (
            h[d.len() - 1],
            h[d.len() - 2],
            d[d.len() - 1],
            d[d.len() - 2],
        )
    };
    if dl == T::ZERO || dl * dr <= T::ZERO {
        return T::ZERO;
    }
    let s = ((T::TWO * hl + hr) * dl - hl * dr) / (hl + hr);
    if s * dl <= T::ZERO {
        T::ZERO
    } else if s.signum() != dl.signum() {
        T::TWO * dl
    } else {
        s
    }
}

/// The Thomas algorithm: a tridiagonal solve in time linear in the size of the system. `lower`
/// and `upper` are the off-diagonals, `centre` the diagonal (destroyed), `rhs` the right-hand
/// side (replaced by the answer) and `x` where the answer is written.
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

/// The two end rows of the not-a-knot spline. The third derivative of the cubic on the interval
/// `i` is `6·(m[i] + m[i+1] − 2·d[i])/h[i]²`, so continuing it across the second knot gives
/// `h[1]²·(m[0] + m[1] − 2·d[0]) = h[0]²·(m[1] + m[2] − 2·d[1])`, and similarly at the other end;
/// each relation eliminates one end slope.
#[allow(clippy::too_many_arguments)]
fn not_a_knot_rows<T: Real>(
    lower: &mut [T],
    centre: &mut [T],
    upper: &mut [T],
    rhs: &mut [T],
    h: &[T],
    d: &[T],
    r: T,
    sl: T,
) {
    let k = centre.len();
    // The first knot: m[0] = 2·d[0] − 2·r·d[1] − (1 − r)·m[1] + r·m[2], which stands in the
    // first row with the coefficient h[1] that the interior rows give the slope of the end knot.
    centre[0] -= h[1] * (T::ONE - r);
    upper[0] += h[1] * r;
    rhs[0] -= h[1] * (T::TWO * d[0] - T::TWO * r * d[1]);
    lower[0] = T::ZERO;
    // The penultimate knot: with s = (h[n−3]/h[n−2])², the continued third derivative gives
    // m[n−1] = m[n−3]/s + (1 − s)·m[n−2]/s − 2·d[n−3]/s + 2·d[n−2], whose terms fold into the
    // last row, where the coefficient of m[n−1] is h[n−2] itself.
    let before = h[k - 1];
    let inv = T::ONE / sl;
    lower[k - 1] += before * inv;
    centre[k - 1] += before * inv * (T::ONE - sl);
    rhs[k - 1] += before * (T::TWO * inv * d[k - 1] - T::TWO * d[k]);
    upper[k - 1] = T::ZERO;
}

/// The `count` nodes of Chebyshev interpolation on `[a, b]`: the zeros of the Chebyshev
/// polynomial `T[count]`, mapped onto the interval and returned in increasing order. The
/// polynomial of degree `count − 1` through these nodes is within a logarithmic factor of the
/// best there is.
///
/// # Panics
///
/// If `degree` is zero or `a` is not less than `b`.
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
pub fn chebyshev_nodes<T: Real>(count: usize, a: T, b: T) -> Vec<T> {
    assert!(
        count > 0,
        "a Chebyshev interpolation needs at least one node"
    );
    assert!(a < b, "the interval must be increasing");
    let half = (b - a) * T::HALF;
    let mid = a + half;
    (0..count)
        .map(|k| {
            let k = T::from_f64(k as f64);
            let n = T::from_f64(count as f64);
            // The zero −cos θ of T[count] is mapped onto the interval, which reverses the order
            // the zeros come in and so leaves them increasing.
            mid - half * (T::PI * (T::TWO * k + T::ONE) / (T::TWO * n)).cos()
        })
        .collect()
}

/// The barycentric weights of the nodes `x`, `w[k] = 1 / Π_{j≠k}(x[k] − x[j])`, which
/// [`barycentric_interpolate`] needs. The classical node sets have closed forms, but this general
/// product is what arbitrary data need; it costs a pass over the pairs of nodes.
///
/// The weights matter only up to a common factor, so they can be scaled freely to keep them in
/// range for a large number of nodes.
///
/// # Panics
///
/// If `x` is empty or holds two equal nodes.
///
/// # Examples
///
/// ```
/// use a::math::numerics::barycentric_weights;
///
/// // The weights of three equally spaced nodes, defined up to a common factor.
/// let w = barycentric_weights(&[0.0, 1.0, 2.0]);
/// assert_eq!(w[1], -1.0);
/// assert_eq!(w[0], w[2]);
/// ```
pub fn barycentric_weights<T: Real>(x: &[T]) -> Vec<T> {
    assert!(!x.is_empty(), "there must be at least one node");
    let n = x.len();
    let mut w = vec![T::ONE; n];
    for k in 0..n {
        for j in 0..n {
            if j != k {
                let difference = x[k] - x[j];
                assert!(difference != T::ZERO, "two nodes are equal");
                w[k] *= difference;
            }
        }
        w[k] = w[k].recip();
    }
    w
}

/// The value at `t` of the polynomial of degree below `x.len()` that takes the values `y` at the
/// nodes `x`, by the second form of the barycentric formula,
/// `p(t) = Σ w[k]·y[k]/(t − x[k]) ÷ Σ w[k]/(t − x[k])`.
///
/// `w` must be the weights of `x`, from [`barycentric_weights`]: the formula is one pass once
/// they are known, which is why they are kept apart so that a node set can serve many data sets.
/// A `w` of zero length weights every node equally.
///
/// # Panics
///
/// If `x` and `y` differ in length, if `x` is empty, or if `w` is neither empty nor as long as
/// they.
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
/// ```
pub fn barycentric_interpolate<T: Real>(x: &[T], y: &[T], w: &[T], t: T) -> T {
    assert_eq!(
        x.len(),
        y.len(),
        "the nodes and the values differ in length"
    );
    assert!(!x.is_empty(), "there must be at least one node");
    assert!(
        w.is_empty() || w.len() == x.len(),
        "the weights differ in length"
    );
    let mut numerator = T::ZERO;
    let mut denominator = T::ZERO;
    for k in 0..x.len() {
        let weight = if w.is_empty() { T::ONE } else { w[k] };
        let difference = t - x[k];
        if difference == T::ZERO {
            return y[k];
        }
        let term = weight / difference;
        numerator += term * y[k];
        denominator += term;
    }
    numerator / denominator
}

/// The Chebyshev series `Σ c[k]·T[k](u(t))` of a function on `[a, b]`, where `u` maps `[a, b]`
/// onto `[-1, 1]`. Build one with [`chebyshev`].
///
/// # Examples
///
/// ```
/// use a::math::numerics::chebyshev;
///
/// let c = chebyshev(|x: f64| x.exp(), 0.0, 1.0, 12).unwrap();
/// assert!((c.evaluate(0.5) - 0.5f64.exp()).abs() < 1e-13);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Chebyshev<T> {
    a: T,
    b: T,
    c: Vec<T>,
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
        let mut d = vec![T::ZERO; n];
        // The top coefficient of the derivative series vanishes, and with it the one above.
        d[n - 2] = T::from_f64(2.0 * (n - 1) as f64) * self.c[n - 1];
        for k in (1..n - 1).rev() {
            d[k - 1] = d[k + 1] + T::from_f64(2.0 * k as f64) * self.c[k];
        }
        d[0] *= T::HALF;
        clenshaw(&d, map_in(t, self.a, self.b)) / ((self.b - self.a) * T::HALF)
    }

    /// The integral from `a` to `t`, term by term.
    pub fn integral(&self, t: T) -> T {
        let n = self.c.len();
        if n == 0 {
            return T::ZERO;
        }
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
        let scale = (self.b - self.a) * T::HALF;
        (clenshaw(&e, map_in(t, self.a, self.b)) - clenshaw(&e, -T::ONE)) * scale
    }
}

/// The Chebyshev series of `f` on `[a, b]` of the given `degree`, fitted at the `degree + 1`
/// Chebyshev nodes of [`chebyshev_nodes`], which makes it the interpolating polynomial of that
/// degree. The coefficients are the discrete cosine transform of the values, so `degree + 1`
/// evaluations of `f` are all it costs.
///
/// Returns `None` when a value of `f` is not finite.
///
/// # Panics
///
/// If `degree` is zero or `a` is not less than `b`.
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
/// ```
pub fn chebyshev<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    degree: usize,
) -> Option<Chebyshev<T>> {
    let nodes = chebyshev_nodes(degree + 1, a, b);
    let mut values = Vec::with_capacity(degree + 1);
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
    let n = T::from_f64((degree + 1) as f64);
    let mut c = Vec::with_capacity(degree + 1);
    for k in 0..=degree {
        let kf = T::from_f64(k as f64);
        let mut sum = T::ZERO;
        for (j, &value) in values.iter().enumerate() {
            let jf = T::from_f64(j as f64);
            let theta = T::PI * (T::TWO * jf + T::ONE) / (T::TWO * n);
            let t = (kf * theta).cos();
            sum += value * if k % 2 == 0 { t } else { -t };
        }
        c.push(if k == 0 { sum / n } else { T::TWO * sum / n });
    }
    Some(Chebyshev { a, b, c })
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
