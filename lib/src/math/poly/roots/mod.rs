//! Roots of polynomials: in double precision for `f64` and `Complex<f64>` coefficients, certified
//! to any precision for integer and rational ones, and the isolation and refinement of real roots.

mod aberth;
mod certified;
mod dd;
mod fixed;
mod flat;
mod real;
mod yun;

use super::Polynomial;
use crate::math::complex::Complex;
use crate::math::float::{BigComplex, BigFloat};
use crate::math::matrix::Matrix;
use crate::math::numbers::Number;
use crate::math::rational::Rational;

/// A complex root of a polynomial over the integers or the rationals, certified: the root lies in
/// the closed disc of the [`radius`](ComplexRoot::radius) about the
/// [`value`](ComplexRoot::value), which holds no other root of the polynomial, and is a root of
/// the [`multiplicity`](ComplexRoot::multiplicity).
///
/// # Examples
///
/// ```
/// use a::math::numbers::Number;
/// use a::math::poly::Polynomial;
///
/// // x² + 1: ±i.
/// let p = Polynomial::new(vec![Number::from(1), Number::from(0), Number::from(1)]);
/// let r = p.complex_roots(64);
/// assert_eq!(r.len(), 2);
/// assert!(r[0].value().re().abs() <= *r[0].radius());
/// assert_eq!(r[0].multiplicity(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct ComplexRoot {
    pub(crate) value: BigComplex,
    pub(crate) radius: BigFloat,
    pub(crate) multiplicity: u32,
}

impl ComplexRoot {
    /// The centre of the disc: an approximation of the root of at least the precision asked for
    /// in each part, the imaginary part an exact zero for a root proven real, and the parts of a
    /// pair of conjugate roots proven so exact conjugates.
    pub fn value(&self) -> &BigComplex {
        &self.value
    }

    /// The radius of the disc, an upper bound of the distance from the value to the root.
    pub fn radius(&self) -> &BigFloat {
        &self.radius
    }

    /// The multiplicity of the root.
    pub fn multiplicity(&self) -> u32 {
        self.multiplicity
    }
}

/// The companion matrix of `p` (not zero): ones under the diagonal and `−aᵢ/aₙ` in the last
/// column, so that its characteristic polynomial is `p/aₙ`.
fn companion<T: Clone>(c: &[T], zero: T, one: T, entry: impl Fn(&T, &T) -> T) -> Matrix<T> {
    let n = c
        .len()
        .checked_sub(1)
        .expect("the zero polynomial has no companion matrix");
    let lead = &c[n];
    Matrix::from_fn(n, n, |i, j| {
        if j + 1 == n {
            entry(&c[i], lead)
        } else if i == j + 1 {
            one.clone()
        } else {
            zero.clone()
        }
    })
}

/// Panics unless every coefficient is finite.
#[track_caller]
fn finite<T: Copy>(c: &[T], parts: impl Fn(T) -> (f64, f64)) {
    assert!(
        c.iter().all(|&x| {
            let (r, i) = parts(x);
            r.is_finite() && i.is_finite()
        }),
        "a coefficient is not finite"
    );
}

impl Polynomial<f64> {
    /// The complex roots, each as often as its multiplicity, ordered by their real and then
    /// their imaginary parts: those that are not real in pairs of exact conjugates, the one of
    /// the negative imaginary part first. A constant has none.
    ///
    /// They are found by the Aberth–Ehrlich iteration in double precision, from starting points
    /// on circles of the radii of the Newton polygon of the coefficients' moduli, each stopped
    /// when the polynomial's value is within the rounding error of Horner's rule; those whose
    /// inclusion discs then meet (near a multiple root, or where the polynomial is ill
    /// conditioned) are polished by the same iteration in double-double arithmetic, and a
    /// cluster of them as close as a multiple root's at that precision is that multiple root.
    /// The roots are as accurate as the conditioning allows: each is a root of a polynomial
    /// whose coefficients are those of `p` changed by a few units in their last place, and each
    /// root of `p` has one of them near it.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero or a coefficient is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    ///
    /// // x² − 3x + 2 = (x − 1)(x − 2)
    /// let r = Polynomial::new(vec![2.0, -3.0, 1.0]).roots();
    /// assert!((r[0].re - 1.0).abs() < 1e-15 && r[0].im == 0.0);
    /// assert!((r[1].re - 2.0).abs() < 1e-15 && r[1].im == 0.0);
    ///
    /// // x² + 1 = (x − i)(x + i)
    /// let r = Polynomial::new(vec![1.0, 0.0, 1.0]).roots();
    /// assert!(r[0].re.abs() < 1e-15 && (r[0].im + 1.0).abs() < 1e-15);
    /// assert_eq!(r[1], r[0].conj());
    /// ```
    #[track_caller]
    pub fn roots(&self) -> Vec<Complex<f64>> {
        assert!(
            !self.c.is_empty(),
            "the zero polynomial has every number as a root"
        );
        finite(&self.c, |x| (x, 0.0));
        aberth::roots(&self.c)
    }

    /// The companion matrix: ones under the diagonal and `−aᵢ/aₙ` in the last column, the
    /// eigenvalues of which are the roots; `0 × 0` for a constant.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::poly::Polynomial;
    ///
    /// let c = Polynomial::new(vec![2.0, -3.0, 1.0]).companion();
    /// assert_eq!(c, Matrix::from_rows(&[[0.0, -2.0], [1.0, 3.0]]));
    /// ```
    #[track_caller]
    pub fn companion(&self) -> Matrix<f64> {
        companion(&self.c, 0.0, 1.0, |a, l| -a / l)
    }
}

impl Polynomial<Complex<f64>> {
    /// The complex roots, each as often as its multiplicity, ordered by their real and then
    /// their imaginary parts, by the Aberth–Ehrlich iteration in double precision as for
    /// [`Polynomial<f64>::roots`]. A constant has none.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero or a coefficient is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::poly::Polynomial;
    ///
    /// // x − i
    /// let p = Polynomial::new(vec![-Complex::I, Complex::ONE]);
    /// assert_eq!(p.roots(), [Complex::I]);
    /// ```
    #[track_caller]
    pub fn roots(&self) -> Vec<Complex<f64>> {
        assert!(
            !self.c.is_empty(),
            "the zero polynomial has every number as a root"
        );
        finite(&self.c, |x| (x.re, x.im));
        aberth::roots(&self.c)
    }

    /// The companion matrix: ones under the diagonal and `−aᵢ/aₙ` in the last column; `0 × 0`
    /// for a constant.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::poly::Polynomial;
    ///
    /// let p = Polynomial::new(vec![-Complex::I, Complex::ONE]);
    /// assert_eq!(p.companion()[(0, 0)], Complex::I);
    /// ```
    #[track_caller]
    pub fn companion(&self) -> Matrix<Complex<f64>> {
        companion(&self.c, Complex::ZERO, Complex::ONE, |a, l| -*a / *l)
    }
}

impl Polynomial<Rational> {
    /// The companion matrix, exactly: ones under the diagonal and `−aᵢ/aₙ` in the last column;
    /// `0 × 0` for a constant.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // 2x² + 1
    /// let p = Polynomial::new(vec![Rational::from(1), Rational::from(0), Rational::from(2)]);
    /// let c = p.companion();
    /// assert_eq!(c[(0, 1)], Rational::new((-1).into(), 2.into()));
    /// assert_eq!(c[(1, 0)], Rational::from(1));
    /// ```
    #[track_caller]
    pub fn companion(&self) -> Matrix<Rational> {
        companion(&self.c, Rational::default(), Rational::from(1), |a, l| {
            -(a / l)
        })
    }

    /// The complex roots, certified, to `prec` bits: see [`Polynomial<Number>::complex_roots`].
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero or `prec` is 0 or above `BigFloat::MAX_PREC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // x² − 1/4: ±1/2.
    /// let p = Polynomial::new(vec![Rational::new((-1).into(), 4.into()), 0.into(), 1.into()]);
    /// let r = p.complex_roots(53);
    /// assert_eq!(r[0].value().re().to_f64(), -0.5);
    /// assert_eq!(r[1].value().re().to_f64(), 0.5);
    /// ```
    #[track_caller]
    pub fn complex_roots(&self, prec: u64) -> Vec<ComplexRoot> {
        self.to_primitive().1.complex_roots(prec)
    }

    /// Isolating intervals of the real roots: see [`Polynomial<Number>::real_root_intervals`].
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // (x − 1/3)(x² − 2): −√2, 1/3, √2.
    /// let third = Rational::new(1.into(), 3.into());
    /// let p = Polynomial::new(vec![-third.clone(), 1.into()])
    ///     * Polynomial::new(vec![Rational::from(-2), 0.into(), 1.into()]);
    /// let r = p.real_root_intervals();
    /// assert_eq!(r.len(), 3);
    /// assert!(r[1].0 <= third && third <= r[1].1);
    /// assert!(&r[2].0 * &r[2].0 < Rational::from(2) && &r[2].1 * &r[2].1 > Rational::from(2));
    /// ```
    #[track_caller]
    pub fn real_root_intervals(&self) -> Vec<(Rational, Rational, u32)> {
        self.to_primitive().1.real_root_intervals()
    }

    /// The isolating intervals by Descartes's rule of signs: see
    /// [`Polynomial<Number>::real_root_intervals_descartes`].
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = Polynomial::new(vec![Rational::from(-2), 0.into(), 1.into()]);
    /// assert_eq!(p.real_root_intervals_descartes().len(), 2);
    /// ```
    #[track_caller]
    pub fn real_root_intervals_descartes(&self) -> Vec<(Rational, Rational, u32)> {
        self.to_primitive().1.real_root_intervals_descartes()
    }

    /// The isolating intervals by continued fractions: see
    /// [`Polynomial<Number>::real_root_intervals_cf`].
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = Polynomial::new(vec![Rational::from(-2), 0.into(), 1.into()]);
    /// assert_eq!(p.real_root_intervals_cf().len(), 2);
    /// ```
    #[track_caller]
    pub fn real_root_intervals_cf(&self) -> Vec<(Rational, Rational, u32)> {
        self.to_primitive().1.real_root_intervals_cf()
    }

    /// The real roots, correctly rounded: see [`Polynomial<Number>::real_roots`].
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero or `prec` is 0 or above `BigFloat::MAX_PREC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = Polynomial::new(vec![Rational::from(-2), 0.into(), 1.into()]);
    /// let r = p.real_roots(53);
    /// assert_eq!(r[1].0.to_f64(), 2f64.sqrt());
    /// ```
    #[track_caller]
    pub fn real_roots(&self, prec: u64) -> Vec<(BigFloat, u32)> {
        self.to_primitive().1.real_roots(prec)
    }

    /// The number of distinct real roots: see [`Polynomial<Number>::count_real_roots`].
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// let p = Polynomial::new(vec![Rational::from(1), 0.into(), 1.into()]);
    /// assert_eq!(p.count_real_roots(), 0);
    /// ```
    #[track_caller]
    pub fn count_real_roots(&self) -> usize {
        self.to_primitive().1.count_real_roots()
    }
}

impl Polynomial<Number> {
    /// Panics if the polynomial is zero.
    #[track_caller]
    fn not_zero(&self) {
        assert!(
            !self.c.is_empty(),
            "the zero polynomial has every number as a root"
        );
    }

    /// The complex roots, certified, to `prec` bits: for each distinct root a [`ComplexRoot`],
    /// the root within the radius of the value, the discs pairwise disjoint, with the root's
    /// multiplicity; the radius at most `2^(1−prec)` times the value's modulus. The real roots
    /// come first, increasing, with an exact zero imaginary part; then the others by their real
    /// parts, each with a positive imaginary part before its conjugate. A constant has none.
    ///
    /// The polynomial is split into square-free factors by Yun's algorithm. The roots of each
    /// factor are approximated by the Aberth–Ehrlich iteration, in double precision when the
    /// coefficients fit and at doubling precisions after, until discs about them of the radius
    /// `d·|f(z)|/|f'(z)|` (with the errors of the evaluation bounded, rounded up) are disjoint:
    /// then each holds one root. Each disc is refined by Newton's method at doubling precisions,
    /// a new one accepted only inside the old. The polynomial and its derivative are evaluated
    /// by Horner's rule in fixed point on limbs, each part of a value held to a power of `2^64`
    /// chosen from a bound of its size and each truncation counted in the bound of its error, or
    /// in [`BigComplex`] at a point whose parts are of very different sizes.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero or `prec` is 0 or above `BigFloat::MAX_PREC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    ///
    /// // (x − 2)²(x² + 2): 2 twice, ±i√2.
    /// let p = Polynomial::new([8, -8, 6, -4, 1].map(Number::from).to_vec());
    /// let r = p.complex_roots(100);
    /// assert_eq!(r.len(), 3);
    /// assert_eq!(r[0].multiplicity(), 2);
    /// assert!(r[0].value().im().is_zero());
    /// let sqrt2 = BigFloat::from(2).sqrt_round(100, Round::Nearest).0;
    /// let err = (r[1].value().im() - &sqrt2).abs();
    /// assert!(err <= *r[1].radius());
    /// ```
    #[track_caller]
    pub fn complex_roots(&self, prec: u64) -> Vec<ComplexRoot> {
        self.not_zero();
        BigFloat::check_prec(prec);
        certified::complex_roots(self, prec)
    }

    /// Isolating intervals of the real roots, increasing, with their multiplicities: `(a, a, m)`
    /// for a rational root `a`, `(a, b, m)` with `a < b` for a root that is the only one of the
    /// polynomial in the open interval `(a, b)`, whose ends are not roots. No two overlap. Each
    /// square-free factor is isolated, the positive and the negative roots apart, by continued
    /// fractions ([`real_root_intervals_cf`](Self::real_root_intervals_cf)) where the bound
    /// of the roots is large (roots far apart in a long interval, as Wilkinson's), else by
    /// Descartes's rule of signs
    /// ([`real_root_intervals_descartes`](Self::real_root_intervals_descartes)), whose nodes
    /// deep in the bisection (very close roots, as Mignotte's) go on by continued fractions.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    /// use a::math::rational::Rational;
    ///
    /// // (x − 1)²(x² − 2): −√2, 1 twice, √2.
    /// let p = Polynomial::new([-2, 4, -1, -2, 1].map(Number::from).to_vec());
    /// let r = p.real_root_intervals();
    /// assert_eq!(r.len(), 3);
    /// assert_eq!(r[1], (Rational::from(1), Rational::from(1), 2));
    /// assert!(&r[2].0 * &r[2].0 < Rational::from(2));
    /// ```
    #[track_caller]
    pub fn real_root_intervals(&self) -> Vec<(Rational, Rational, u32)> {
        self.not_zero();
        real::intervals(self, real::DEFAULT)
    }

    /// The isolating intervals of [`real_root_intervals`](Self::real_root_intervals) by
    /// Descartes's rule of signs: the bisection of Vincent, Collins and Akritas, each interval
    /// `(c/2^k, (c+1)/2^k)·2^s` tested by the sign variations of the Taylor shift
    /// `(x+1)ⁿ·q(1/(x+1))` of its polynomial, the shift stopped at the second variation.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    ///
    /// // x³ − x: −1, 0, 1, exactly.
    /// let p = Polynomial::new([0, -1, 0, 1].map(Number::from).to_vec());
    /// let r = p.real_root_intervals_descartes();
    /// assert!(r.iter().all(|(a, b, m)| a == b && *m == 1));
    /// ```
    #[track_caller]
    pub fn real_root_intervals_descartes(&self) -> Vec<(Rational, Rational, u32)> {
        self.not_zero();
        real::intervals(self, real::Method::Descartes)
    }

    /// The isolating intervals of [`real_root_intervals`](Self::real_root_intervals) by
    /// continued fractions: Vincent's theorem as Akritas and Strzeboński's VAS applies it, each
    /// polynomial shifted by the local-max-quadratic lower bound of its positive roots, then split
    /// into `q(x + 1)` and `(x+1)ⁿ·q(1/(x+1))`.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    ///
    /// let p = Polynomial::new([-2, 0, 1].map(Number::from).to_vec());
    /// let r = p.real_root_intervals_cf();
    /// assert_eq!(r.len(), 2);
    /// ```
    #[track_caller]
    pub fn real_root_intervals_cf(&self) -> Vec<(Rational, Rational, u32)> {
        self.not_zero();
        real::intervals(self, real::Method::ContinuedFractions)
    }

    /// The real roots, increasing, each rounded to nearest (ties to even) at `prec` bits,
    /// correctly, with its multiplicity. The roots are refined from their isolating intervals by
    /// Newton's method at doubling precisions, safeguarded by bisection, on dyadic points, the
    /// polynomial evaluated by Horner's rule in fixed point on limbs (by rectangular splitting
    /// from about 500 bits), and the rounding proven by the signs of the factor at the two ends
    /// of the interval of the reals that round to the value.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero or `prec` is 0 or above `BigFloat::MAX_PREC`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    ///
    /// // x² − 2
    /// let p = Polynomial::new([-2, 0, 1].map(Number::from).to_vec());
    /// let r = p.real_roots(200);
    /// assert_eq!(r[1].0, BigFloat::from(2).sqrt_round(200, Round::Nearest).0);
    /// assert_eq!(r[0].0, -r[1].0.clone());
    /// ```
    #[track_caller]
    pub fn real_roots(&self, prec: u64) -> Vec<(BigFloat, u32)> {
        self.not_zero();
        BigFloat::check_prec(prec);
        real::real_roots(self, prec)
    }

    /// The number of distinct real roots, as PARI's `polsturm`: for each square-free factor by
    /// Sturm's sequence when it collapses at once (the first remainder of a low degree, as for
    /// `xⁿ` plus a polynomial of a low degree), else by the isolation's method, counting its
    /// intervals.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    /// use a::math::poly::Polynomial;
    ///
    /// // (x − 1)²(x² − 2)
    /// let p = Polynomial::new([-2, 4, -1, -2, 1].map(Number::from).to_vec());
    /// assert_eq!(p.count_real_roots(), 3);
    /// ```
    #[track_caller]
    pub fn count_real_roots(&self) -> usize {
        self.not_zero();
        real::count(self, real::DEFAULT)
    }
}
