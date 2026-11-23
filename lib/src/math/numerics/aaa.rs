//! Rational approximation by the AAA algorithm, in barycentric form.
//!
//! The adaptive Antoulas–Anderson algorithm of Y. Nakatsukasa, O. Sete and L. N. Trefethen
//! (*SIAM J. Sci. Comput.* 40, 2018, A1494–A1522) builds a rational function of type
//! `(m − 1, m − 1)` that approximates sampled data,
//!
//! `r(x) = (Σ w_j·f_j/(x − z_j)) / (Σ w_j/(x − z_j))`,
//!
//! from `m` of the samples, the support points `z_j`, where it takes the data values `f_j`
//! exactly. They are chosen greedily: each step adds the sample where the current approximation
//! is worst, and the weights `w_j` are the right singular vector of the smallest singular value of
//! the Loewner matrix `(f_i − f_j)/(x_i − z_j)` over the remaining samples, which minimises the
//! linearised error `‖f·d − n‖` with `‖w‖ = 1`. It stops when the error on the samples is at most
//! `rtol·max|f|`, or at the term limit, which [`Aaa::converged`] reports. Its poles, residues and
//! zeros are found from the generalised eigenvalue problem of the barycentric form, reduced to a
//! standard one, and refined until each is a zero to rounding, which is checked; and the
//! Froissart doublets — spurious poles with negligible residue that a tight
//! tolerance or noisy data leave — are removed as in the paper's clean-up step.
//!
//! The samples may be real ([`aaa`]) or lie anywhere in the complex plane ([`aaa_complex`]),
//! where AAA is at its best: a rational function follows poles and branch cuts that no
//! polynomial can.

use super::Real;
use crate::math::complex::{Complex, Float};
use crate::math::matrix::{Matrix, Scalar};

/// The options of [`aaa_with`] and [`aaa_complex_with`].
#[derive(Debug, Clone, Copy)]
pub struct AaaOptions<T> {
    /// The relative tolerance: the iteration stops when the largest error on the samples is at
    /// most `rtol` times the largest value. Zero runs to the term limit.
    pub rtol: T,
    /// The largest number of support points, at least 1. No more than half the samples (rounded
    /// up) are ever used, since beyond that the weights are no longer determined by the data.
    pub max_terms: usize,
    /// Whether the Froissart doublets are removed after the iteration.
    pub clean_up: bool,
    /// A pole is a Froissart doublet when its residue divided by its distance from the nearest
    /// sample is below `clean_up_tol` times the geometric mean of the nonzero `|f|`.
    pub clean_up_tol: T,
}

impl<T: Real> Default for AaaOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> AaaOptions<T> {
    /// The default options, SciPy's: `rtol` `ε^(3/4)` (`1.9e−12` for `f64`, `6.4e−6` for `f32`),
    /// 100 terms, the clean-up on, with `clean_up_tol` `1e−13` for `f64` and the same number of
    /// units in the last place, `5.4e−5`, for `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::AaaOptions;
    ///
    /// let o = AaaOptions::<f64>::new().with_rtol(1e-13).with_max_terms(50).with_clean_up(false);
    /// assert_eq!((o.rtol, o.max_terms, o.clean_up), (1e-13, 50, false));
    /// ```
    pub fn new() -> Self {
        Self {
            rtol: T::EPSILON.powf(T::from_f64(0.75)),
            max_terms: 100,
            clean_up: true,
            clean_up_tol: T::EPSILON * T::from_f64(1e-13 / f64::EPSILON),
        }
    }

    /// Sets the relative tolerance.
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the largest number of support points.
    pub fn with_max_terms(mut self, max_terms: usize) -> Self {
        self.max_terms = max_terms;
        self
    }

    /// Turns the removal of Froissart doublets on or off.
    pub fn with_clean_up(mut self, clean_up: bool) -> Self {
        self.clean_up = clean_up;
        self
    }

    /// Sets the threshold of the clean-up.
    pub fn with_clean_up_tol(mut self, clean_up_tol: T) -> Self {
        self.clean_up_tol = clean_up_tol;
        self
    }
}

/// A rational approximation in barycentric form, from [`aaa`] (real samples, `S` = `f32` or
/// `f64`) or [`aaa_complex`] (complex samples, `S` = `Complex<f32>` or `Complex<f64>`).
///
/// # Examples
///
/// ```
/// use a::math::numerics::aaa;
///
/// // Runge's function is itself rational, of type (0, 2): three support points find it, and
/// // its poles ±i/5.
/// let x: Vec<f64> = (0..1000).map(|k| -1.0 + 2.0 * f64::from(k) / 999.0).collect();
/// let f: Vec<f64> = x.iter().map(|t| 1.0 / (1.0 + 25.0 * t * t)).collect();
/// let r = aaa(&x, &f).unwrap();
/// assert!(r.converged() && r.degree() == 2);
/// assert!((r.evaluate(0.3) - 1.0 / (1.0 + 25.0 * 0.09)).abs() < 1e-14);
/// for p in r.poles() {
///     assert!(p.re.abs() < 1e-13 && (p.im.abs() - 0.2).abs() < 1e-13);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Aaa<S: Scalar> {
    points: Vec<S>,
    values: Vec<S>,
    /// The support values divided by `2^exponent`, which leaves the largest sample between 1 and
    /// 2 in size: what the weights were found for, and what every formula here uses.
    scaled: Vec<S>,
    weights: Vec<S>,
    exponent: i32,
    errors: Vec<S::Real>,
    error: S::Real,
    converged: bool,
    doublets: usize,
}

impl<S: Scalar> Aaa<S> {
    /// The value of the approximation at `x`: the data value at a support point, exactly, and
    /// also where `x` is so close to one that its term overflows; at an infinite `x` the limit
    /// `Σ w_j·f_j / Σ w_j`; NaN at NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::aaa;
    ///
    /// let x: Vec<f64> = (0..200).map(|k| f64::from(k) / 199.0).collect();
    /// let f: Vec<f64> = x.iter().map(|t| t.exp()).collect();
    /// let r = aaa(&x, &f).unwrap();
    /// let z = r.support_points()[0];
    /// assert_eq!(r.evaluate(z), r.support_values()[0]);
    /// assert!((r.evaluate(0.123) - 0.123f64.exp()).abs() < 1e-12);
    /// ```
    pub fn evaluate(&self, x: S) -> S {
        match barycentric(x, &self.points, &self.scaled, &self.weights) {
            At::Support(j) => self.values[j],
            At::Value(v) => times_two_to(v, self.exponent),
        }
    }

    /// The value of the approximation at a point `z` of the complex plane, which for real
    /// samples continues the approximation off the real line.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::numerics::aaa;
    ///
    /// // exp on [-1, 1], continued to the imaginary axis: e^(i/2) = cos ½ + i·sin ½.
    /// let x: Vec<f64> = (0..400).map(|k| -1.0 + 2.0 * f64::from(k) / 399.0).collect();
    /// let f: Vec<f64> = x.iter().map(|t| t.exp()).collect();
    /// let r = aaa(&x, &f).unwrap();
    /// let v = r.evaluate_complex(Complex::new(0.0, 0.5));
    /// assert!((v - Complex::new(0.5f64.cos(), 0.5f64.sin())).abs() < 1e-10);
    /// ```
    pub fn evaluate_complex(&self, z: Complex<S::Real>) -> Complex<S::Real> {
        if z.is_nan() {
            return z;
        }
        let back = |v: Complex<S::Real>| {
            Complex::new(
                real_times_two_to(v.re, self.exponent),
                real_times_two_to(v.im, self.exponent),
            )
        };
        let terms = self.points.iter().zip(&self.scaled).zip(&self.weights);
        if !z.is_finite() {
            let (n, d) = terms.fold((Complex::ZERO, Complex::ZERO), |(n, d), ((_, &f), &w)| {
                (n + w.to_complex() * f.to_complex(), d + w.to_complex())
            });
            return back(n / d);
        }
        let (mut n, mut d) = (Complex::ZERO, Complex::ZERO);
        for (j, ((&p, &f), &w)) in terms.enumerate() {
            let difference = z - p.to_complex();
            if difference == Complex::ZERO {
                return self.values[j].to_complex();
            }
            let term = w.to_complex() / difference;
            if !term.is_finite() {
                return self.values[j].to_complex();
            }
            n += term * f.to_complex();
            d += term;
        }
        back(n / d)
    }

    /// The support points `z_j`, the samples where the approximation takes the data values, in
    /// the order the iteration chose them.
    pub fn support_points(&self) -> &[S] {
        &self.points
    }

    /// The data values `f_j` at the support points.
    pub fn support_values(&self) -> &[S] {
        &self.values
    }

    /// The barycentric weights `w_j`, of 2-norm 1 (scaled column by column when the Loewner
    /// matrix was ill-conditioned; see [`aaa_with`]).
    pub fn weights(&self) -> &[S] {
        &self.weights
    }

    /// The degree `m − 1` of numerator and denominator, for `m` support points.
    pub fn degree(&self) -> usize {
        self.points.len().saturating_sub(1)
    }

    /// The largest error `|f − r|` on the samples of this approximation, as computed: `r` is
    /// evaluated in the arithmetic of the type, so the figure carries the rounding of that
    /// evaluation (as SciPy's does), which next to a tolerance can tip the balance — of 300 cases
    /// a review recomputed exactly, one called converged had an error of 1.004 times the
    /// tolerance.
    pub fn error(&self) -> S::Real {
        self.error
    }

    /// The largest error on the samples after each step of the iteration (before any clean-up).
    pub fn errors(&self) -> &[S::Real] {
        &self.errors
    }

    /// Whether the error on the samples, as [`error`](Self::error) computes it, is within the
    /// tolerance: `false` when the term limit was reached first (noisy data, a tolerance below the
    /// attainable accuracy), or the clean-up raised the error above it. When the limit was reached, the approximation is the step of
    /// smallest error, which on noisy data is not the last one (SciPy keeps the last).
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The number of Froissart doublets the clean-up found.
    pub fn doublets(&self) -> usize {
        self.doublets
    }

    /// The poles: the zeros of the denominator `d(x) = Σ w_j/(x − z_j)`, in no particular order
    /// and complex in general, those of real samples real or in exact conjugate pairs; `m − 1` of
    /// them for `m` support points, fewer when the degree drops — a pole numerically at infinity
    /// (beyond about `1/(16·m·ε)` times the distance of the data from the shift used) is left
    /// out — or when a pole cannot be found to rounding (see below).
    ///
    /// The generalised eigenvalue problem of the paper, of the pencil `([0, wᵀ; 1, diag z],
    /// diag(0, 1, …, 1))`, has two infinite eigenvalues. With a shift `s` away from the data,
    /// `t_j = 1/(z_j − s)` and `τ = Σ w_j·t_j`, the finite ones are `s + 1/μ` for the eigenvalues
    /// `μ` of `G = diag(t) − t·(w∘t)ᵀ/τ` other than its `μ = 0` (eigenvector all ones), and that
    /// one is removed exactly by restricting `G` to the invariant subspace `wᵀv = 0`, through the
    /// orthonormal basis a Householder reflection gives. The standard eigenproblem of the
    /// `(m − 1) × (m − 1)` matrix that leaves is solved by [`Matrix::eigenvalues`], and the shift is
    /// the one of four candidates that keeps `|τ|` largest against `Σ|w_j·t_j|`.
    ///
    /// Those eigenvalues are estimates only: the reduction divides by `τ`, which is small at every
    /// shift when the denominator is small away from the samples (for an approximation of an
    /// entire function, whose poles are spurious), and then loses up to ten digits. They are
    /// refined together by the Aberth–Ehrlich iteration on the polynomial `d(x)·Π(x − z_j)`, which
    /// has the same zeros and no pole at a support point, and polished by Newton's method on it, a
    /// step kept only while it lowers the backward error. What is then not a zero to rounding, or
    /// is a zero found twice, is started again, up to six times: the poles found are held fixed,
    /// and the others iterated together freely in the complex plane — so that a conjugate pair
    /// the estimates gave where the denominator has two real zeros can part, and two real
    /// estimates can meet as a pair — from where they were, then from points on circles round the
    /// data. All of this is done in `f64` for `f32` samples too, which are `f64` numbers exactly,
    /// and the poles are rounded to `f32` at the end.
    ///
    /// What is returned is checked. Every pole has a backward error of at most `8ε` (the `ε` of
    /// `f64`, before the rounding to `f32`): relative changes of the weights and of the pole `x`
    /// itself of at most that size make it an exact zero of `d`, the backward error being
    /// `|F| / (|w_j| + |x − z_j|·Σ_{i≠j}|w_i/(x − z_i)| + |x|·|F′|)` for
    /// `F = (x − z_j)·d(x)`, written about the support point `z_j` of the largest term so that it
    /// is finite at and next to one — also for a pole closer to a support point than the floats
    /// there are apart (data that blow up at a sample). A pole is given once: two within twice the
    /// distance that bound allows of each other, where the slope of `d(x)·Π(x − z_j)` points the
    /// same way at both and no other pole is near enough to turn it, are one found twice; two
    /// close poles at which it points opposite ways (a pole numerically double) are both given.
    /// A pole that no restart brings within the bound is left out, and fewer than `m − 1` are
    /// returned. In the tens of thousands of approximations measured none was left out, and every
    /// backward error was below `3ε`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::aaa;
    ///
    /// // tan(πx/2) sampled at 1000 midpoints of [-1, 1] has its poles at ±1.
    /// let x: Vec<f64> = (0..1000).map(|k| -1.0 + (2.0 * f64::from(k) + 1.0) / 1000.0).collect();
    /// let f: Vec<f64> = x.iter().map(|t| (core::f64::consts::FRAC_PI_2 * t).tan()).collect();
    /// let r = aaa(&x, &f).unwrap();
    /// let near: Vec<_> = r.poles().into_iter().filter(|p| p.abs() < 1.5).collect();
    /// assert_eq!(near.len(), 2);
    /// for p in near {
    ///     assert!((p.re.abs() - 1.0).abs() < 1e-12 && p.im == 0.0);
    /// }
    /// ```
    pub fn poles(&self) -> Vec<Complex<S::Real>> {
        let w: Vec<Complex<f64>> = self.weights.iter().map(|&a| wide(a)).collect();
        roots(&self.points, &w)
    }

    /// The residues of the poles, in the order of [`poles`](Self::poles): `n(p)/d′(p)` for the
    /// numerator `n` and the denominator `d` of the barycentric form.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::aaa;
    ///
    /// // 1/(x − 2) + 1/(x + 3) sampled on [-1, 1]: residue 1 at each pole.
    /// let x: Vec<f64> = (0..500).map(|k| -1.0 + 2.0 * f64::from(k) / 499.0).collect();
    /// let f: Vec<f64> = x.iter().map(|t| 1.0 / (t - 2.0) + 1.0 / (t + 3.0)).collect();
    /// let r = aaa(&x, &f).unwrap();
    /// for (p, res) in r.poles().iter().zip(r.residues()) {
    ///     assert!((p.re - 2.0).abs() < 1e-10 || (p.re + 3.0).abs() < 1e-10);
    ///     assert!((res.re - 1.0).abs() < 1e-9 && res.im.abs() < 1e-9);
    /// }
    /// ```
    pub fn residues(&self) -> Vec<Complex<S::Real>> {
        self.poles()
            .into_iter()
            .map(|p| {
                let r = residue(p, &self.points, &self.scaled, &self.weights);
                Complex::new(
                    real_times_two_to(r.re, self.exponent),
                    real_times_two_to(r.im, self.exponent),
                )
            })
            .collect()
    }

    /// The zeros: the support points whose value is zero, and the zeros of the numerator
    /// `Σ w_j·f_j/(x − z_j)` over the others, found and checked as the poles are (see
    /// [`poles`](Self::poles)): each a zero of the numerator with a backward error of at most `8ε`,
    /// each given once, real or in exact conjugate pairs for real samples, and one that cannot be
    /// found to rounding left out. A zero of the data at a sample, where the value computed is a
    /// rounding error rather than 0 (`sin(kπx)` at `x = j/k`), lies a fraction of a float from that
    /// support point and is found there. Empty when every support value is zero, since the
    /// approximation is then zero everywhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::aaa;
    ///
    /// // (x² − 1/4)/(x + 2) on [-1, 1] vanishes at ±1/2.
    /// let x: Vec<f64> = (0..500).map(|k| -1.0 + 2.0 * f64::from(k) / 499.0).collect();
    /// let f: Vec<f64> = x.iter().map(|t| (t * t - 0.25) / (t + 2.0)).collect();
    /// let r = aaa(&x, &f).unwrap();
    /// let mut zeros: Vec<f64> = r.zeros().iter().filter(|z| z.abs() < 1.0).map(|z| z.re).collect();
    /// zeros.sort_by(f64::total_cmp);
    /// assert!((zeros[0] + 0.5).abs() < 1e-12 && (zeros[1] - 0.5).abs() < 1e-12);
    /// ```
    pub fn zeros(&self) -> Vec<Complex<S::Real>> {
        let mut out = Vec::new();
        let mut points = Vec::new();
        let mut products = Vec::new();
        for ((&z, &f), &w) in self.points.iter().zip(&self.scaled).zip(&self.weights) {
            if f == S::zero() {
                out.push(z.to_complex());
            } else {
                points.push(z);
                // Exact for f32 samples, and as rounded before for f64 ones.
                products.push(wide(w) * wide(f));
            }
        }
        if points.is_empty() {
            return Vec::new();
        }
        out.extend(roots(&points, &products));
        out
    }
}

/// The AAA rational approximation of the real samples `f` at the points `x`, with the default
/// options ([`AaaOptions::new`]).
///
/// See [`aaa_with`].
///
/// # Panics
///
/// If `x` and `f` differ in length.
///
/// # Examples
///
/// ```
/// use a::math::numerics::aaa;
///
/// // The gamma function on [-1.5, 1.5], through its poles at 0 and -1: ten terms, and the poles
/// // to ten digits.
/// let x: Vec<f64> = (0..100).map(|k| -1.5 + 3.0 * f64::from(k) / 99.0).collect();
/// # let gamma = |t: f64| -> f64 {
/// #     // Γ(t) by the recurrence from Γ on [1, 2], where a Lanczos sum of g = 7 serves.
/// #     fn lanczos(z: f64) -> f64 {
/// #         let g = 7.0;
/// #         let c = [0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8,
/// #             771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905,
/// #             -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7];
/// #         let z = z - 1.0;
/// #         let mut s = c[0];
/// #         for (i, &ci) in c.iter().enumerate().skip(1) { s += ci / (z + i as f64); }
/// #         let t = z + g + 0.5;
/// #         (2.0 * core::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * s
/// #     }
/// #     let (mut z, mut scale) = (t, 1.0);
/// #     while z < 1.0 { scale /= z; z += 1.0; }
/// #     scale * lanczos(z)
/// # };
/// let f: Vec<f64> = x.iter().map(|&t| gamma(t)).collect();
/// let r = aaa(&x, &f).unwrap();
/// assert!(r.converged());
/// let poles = r.poles();
/// for want in [0.0, -1.0] {
///     assert!(poles.iter().any(|p| (p.re - want).abs() < 1e-9 && p.im.abs() < 1e-9));
/// }
/// ```
#[track_caller]
pub fn aaa<T: Real + Scalar<Real = T>>(x: &[T], f: &[T]) -> Option<Aaa<T>> {
    aaa_with(x, f, &AaaOptions::new())
}

/// [`aaa`] with the options `opts`.
///
/// The samples are prepared as SciPy's `scipy.interpolate.AAA` prepares them, so that the two
/// choose the same support points: a sample whose point or value is not finite is left out, the
/// points are sorted, and of repeated points the first given is kept. The iteration is the
/// paper's, as SciPy runs it: it starts from the mean of the values, takes the first of equally
/// bad samples, sums the singular vectors of a smallest singular value that is repeated, and once
/// the Loewner matrix's condition number has passed `1/(3ε)` scales its columns to unit norm
/// before the singular value decomposition, from that step on. Samples equally bad but for the
/// last bit — symmetric data — are a tie that the rounding of the platform breaks (the libm's
/// `hypot` in a complex error, the order of the sums in the decompositions), so where two codes or
/// two platforms break it differently they follow different paths from there. At most `⌈M/2⌉`
/// support points are used for `M` samples, since with more the weights would not be determined
/// by the data. The clean-up then finds the doublets by `opts.clean_up_tol`, removes the support
/// point nearest to each, and solves for the weights once more on the remaining samples (one
/// pass, as in the paper); the error is measured again after it, and can be larger: the new
/// weights are as accurate as that least-squares problem allows, which past the attainable
/// accuracy is not the error the iteration reached (`|x|` at 300 points, no tolerance, 50 terms:
/// 1.4e−15 after it on aarch64, 1.4e−13 on x86_64). When the term limit is reached before the
/// tolerance, the step of smallest error is the one returned (SciPy returns the last).
///
/// Before anything is computed the values are divided by the power of two that leaves the
/// largest of them between 1 and 2 in size, and what is reported in the units of the values
/// (evaluations, residues, errors) is multiplied back: so the approximation of `2^k·f` is that of
/// `f`, the same support points, weights, poles and zeros, bit for bit, for every `k` that keeps
/// the values normal numbers and the Loewner matrix finite. The squares the 2-norms sum are
/// scaled the same way.
///
/// Returns `None` when no sample is left, `rtol` or `clean_up_tol` is negative or not finite, or
/// `max_terms` is zero.
///
/// # Panics
///
/// If `x` and `f` differ in length.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{AaaOptions, aaa_with};
///
/// // Noisy data cannot be approximated to rounding: the term limit is reached and said so.
/// let x: Vec<f64> = (0..300).map(|k| f64::from(k) / 299.0).collect();
/// let mut seed = 1u64;
/// let mut noise = || {
///     seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
///     (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
/// };
/// let f: Vec<f64> = x.iter().map(|t| t.exp() + 1e-6 * noise()).collect();
/// let opts = AaaOptions::new().with_max_terms(20);
/// let r = aaa_with(&x, &f, &opts).unwrap();
/// assert!(!r.converged() && r.support_points().len() <= 20);
/// assert!(r.error() < 1e-5);
/// ```
#[track_caller]
pub fn aaa_with<T: Real + Scalar<Real = T>>(
    x: &[T],
    f: &[T],
    opts: &AaaOptions<T>,
) -> Option<Aaa<T>> {
    run(x, f, opts)
}

/// The AAA rational approximation of complex samples `f` at points `z` of the complex plane, with
/// the default options.
///
/// See [`aaa_with`]; the points are sorted by real part, then imaginary part.
///
/// # Panics
///
/// If `z` and `f` differ in length.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::numerics::aaa_complex;
///
/// // tan(πz/2) on a spiral of 1000 points winding round the origin (the paper's example): the
/// // poles at ±1, among the points, to rounding, and those at ±3, beyond them, to six digits.
/// let z: Vec<Complex<f64>> = (0..1000)
///     .map(|k| {
///         let t = f64::from(k) / 999.0;
///         Complex::new(-0.5 + t, 15.0 * core::f64::consts::PI * t).exp()
///     })
///     .collect();
/// let tan = |w: Complex<f64>| {
///     let e = (w * Complex::new(0.0, core::f64::consts::PI)).exp();
///     (e - 1.0) / ((e + 1.0) * Complex::I)
/// };
/// let f: Vec<Complex<f64>> = z.iter().map(|&w| tan(w)).collect();
/// let r = aaa_complex(&z, &f).unwrap();
/// assert!(r.converged());
/// let poles = r.poles();
/// for (want, tol) in [(-3.0, 1e-5), (-1.0, 1e-12), (1.0, 1e-12), (3.0, 1e-5)] {
///     assert!(poles.iter().any(|p| (p - Complex::new(want, 0.0)).abs() < tol));
/// }
/// ```
#[track_caller]
pub fn aaa_complex<T: Real>(z: &[Complex<T>], f: &[Complex<T>]) -> Option<Aaa<Complex<T>>>
where
    Complex<T>: Scalar<Real = T>,
{
    aaa_complex_with(z, f, &AaaOptions::new())
}

/// [`aaa_complex`] with the options `opts`; see [`aaa_with`].
///
/// # Panics
///
/// If `z` and `f` differ in length.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::numerics::{AaaOptions, aaa_complex_with};
///
/// // log(2 + z⁴)/(1 − 16z⁴) on the unit circle, run to 50 terms with no tolerance: past the
/// // attainable accuracy the steps depend on the last bits of the decompositions, and so do the
/// // doublets they leave (on aarch64 three in a debug build and none in an optimised one, on
/// // x86_64 one in a debug build); the clean-up removes those it finds, and the error stays at
/// // rounding.
/// let z: Vec<Complex<f64>> = (0..1000)
///     .map(|k| Complex::from_polar(1.0, 2.0 * core::f64::consts::PI * f64::from(k) / 1000.0))
///     .collect();
/// let f: Vec<Complex<f64>> = z
///     .iter()
///     .map(|&w| {
///         let w4 = w * w * w * w;
///         (w4 + 2.0).ln() / (1.0 - 16.0 * w4)
///     })
///     .collect();
/// let opts = AaaOptions::new().with_rtol(0.0).with_max_terms(50);
/// let r = aaa_complex_with(&z, &f, &opts).unwrap();
/// assert!(r.doublets() <= 3 && r.error() < 1e-13);
/// ```
#[track_caller]
pub fn aaa_complex_with<T: Real>(
    z: &[Complex<T>],
    f: &[Complex<T>],
    opts: &AaaOptions<T>,
) -> Option<Aaa<Complex<T>>>
where
    Complex<T>: Scalar<Real = T>,
{
    run(z, f, opts)
}

/// The order SciPy's `numpy.unique` sorts the points by: the real part, then the imaginary part.
fn by_value<S: Scalar>(a: &S, b: &S) -> core::cmp::Ordering {
    let key = |v: &S| (v.re(), v.im());
    key(a)
        .partial_cmp(&key(b))
        .unwrap_or(core::cmp::Ordering::Equal)
}

/// The largest `|x|` of the iterator, NaN if one of them is NaN.
fn largest<R: Float>(values: impl Iterator<Item = R>) -> R {
    let mut big = R::ZERO;
    for v in values {
        if v.is_nan() {
            return R::NAN;
        }
        big = big.max(v);
    }
    big
}

/// The AAA iteration, then the clean-up.
#[track_caller]
fn run<S: Scalar>(x: &[S], f: &[S], opts: &AaaOptions<S::Real>) -> Option<Aaa<S>> {
    assert_eq!(
        x.len(),
        f.len(),
        "the points and the values differ in length"
    );
    let zero = S::Real::ZERO;
    let valid = |t: S::Real| t >= zero && t.is_finite();
    if !valid(opts.rtol) || !valid(opts.clean_up_tol) || opts.max_terms == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && f[i].is_finite())
        .collect();
    order.sort_by(|&a, &b| by_value(&x[a], &x[b]));
    order.dedup_by(|later, kept| x[*later] == x[*kept]);
    if order.is_empty() {
        return None;
    }
    let z: Vec<S> = order.iter().map(|&i| x[i]).collect();
    let original: Vec<S> = order.iter().map(|&i| f[i]).collect();
    // The values as the iteration sees them: divided by the power of two that leaves the largest
    // part between 1 and 2, exactly, so that nothing below depends on their size.
    let exponent = power_of(original.iter().copied());
    let v: Vec<S> = original
        .iter()
        .map(|&t| times_two_to(t, -exponent))
        .collect();
    let samples = z.len();
    let atol = opts.rtol * largest(v.iter().map(|t| t.abs()));
    let limit = opts.max_terms.min(samples.div_ceil(2));

    let count = S::Real::from_f64(samples as f64);
    let mean = v.iter().fold(S::zero(), |s, &t| s + t) / S::from_real(count);
    let mut r = vec![mean; samples];
    let mut free = vec![true; samples];
    let mut support: Vec<usize> = Vec::with_capacity(limit);
    // The Cauchy matrix 1/(x_i − z_j) and the Loewner matrix (f_i − f_j)/(x_i − z_j), a column
    // for each support point.
    let mut cauchy: Vec<Vec<S>> = Vec::with_capacity(limit);
    let mut loewner: Vec<Vec<S>> = Vec::with_capacity(limit);
    let mut ill = false;
    let mut errors: Vec<S::Real> = Vec::with_capacity(limit);
    let mut weights: Vec<S> = Vec::new();
    // The best step so far, kept for when the tolerance is never met: its number of support
    // points, its weights and its error (NaN counting as the worst).
    let mut best: Option<(usize, Vec<S>, S::Real)> = None;
    let mut converged = false;
    for _ in 0..limit {
        // The next support point: the free sample of largest error, the first of equals, a NaN
        // error counting as the largest (as numpy's argmax has it).
        let mut next = None;
        let mut worst = -S::Real::ONE;
        for i in (0..samples).filter(|&i| free[i]) {
            let e = (v[i] - r[i]).abs();
            if e.is_nan() {
                next = Some(i);
                break;
            }
            if e > worst {
                worst = e;
                next = Some(i);
            }
        }
        let j = next.expect("a free sample remains below the term limit");
        support.push(j);
        free[j] = false;
        let column: Vec<S> = z.iter().map(|&t| S::one() / (t - z[j])).collect();
        loewner.push(
            column
                .iter()
                .zip(&v)
                .map(|(&c, &t)| (t - v[j]) * c)
                .collect(),
        );
        cauchy.push(column);
        weights = smallest_singular_vector(&loewner, &free, &mut ill);
        approximate(&mut r, &cauchy, &weights, &support, &v);
        let error = largest(v.iter().zip(&r).map(|(&a, &b)| (a - b).abs()));
        errors.push(error);
        if error <= atol {
            converged = true;
            break;
        }
        if best.as_ref().is_none_or(|b| b.2.is_nan() || error < b.2) {
            best = Some((support.len(), weights.clone(), error));
        }
    }
    if !converged {
        // The term limit was reached: the best approximation of the iteration is returned, which
        // on noisy data is not the last one.
        if let Some((terms, w, _)) = best {
            support.truncate(terms);
            weights = w;
        }
    }
    // Support points of zero weight take no part in the approximation.
    let keep: Vec<usize> = (0..support.len())
        .filter(|&k| weights[k] != S::zero())
        .collect();
    let mut chosen: Vec<usize> = keep.iter().map(|&k| support[k]).collect();
    let mut w: Vec<S> = keep.iter().map(|&k| weights[k]).collect();
    // The largest error on the samples of the approximation with the support points `chosen`.
    let error_of = |chosen: &[usize], w: &[S]| {
        let points: Vec<S> = chosen.iter().map(|&i| z[i]).collect();
        let values: Vec<S> = chosen.iter().map(|&i| v[i]).collect();
        largest(z.iter().zip(&v).map(|(&t, &a)| {
            let value = match barycentric(t, &points, &values, w) {
                At::Support(j) => values[j],
                At::Value(value) => value,
            };
            (a - value).abs()
        }))
    };
    let mut error = error_of(&chosen, &w);
    let mut doublets = 0;
    if opts.clean_up
        && chosen.len() > 1
        && let Some((n, kept, cleaned)) = clean_up(&z, &v, &chosen, &w, opts.clean_up_tol)
    {
        doublets = n;
        (chosen, w) = (kept, cleaned);
        error = error_of(&chosen, &w);
    }
    Some(Aaa {
        points: chosen.iter().map(|&i| z[i]).collect(),
        values: chosen.iter().map(|&i| original[i]).collect(),
        scaled: chosen.iter().map(|&i| v[i]).collect(),
        weights: w,
        exponent,
        errors: errors
            .iter()
            .map(|&e| real_times_two_to(e, exponent))
            .collect(),
        error: real_times_two_to(error, exponent),
        converged: error <= atol,
        doublets,
    })
}

/// The exponent `e` of the power of two that leaves the largest real or imaginary part of the
/// values between 1 and 2 in size, `2^e ≤ max < 2^(e+1)`; 0 when every value is zero.
fn power_of<S: Scalar>(values: impl Iterator<Item = S>) -> i32 {
    let big = largest(values.map(|t| t.re().abs().max(t.im().abs())));
    if !(big > S::Real::ZERO && big.is_finite()) {
        return 0;
    }
    // The exponent of the f64 that holds the value exactly; a subnormal f64 is brought into the
    // normal range first.
    let (mut f, mut e) = (big.to_f64(), 0i32);
    if f < f64::MIN_POSITIVE {
        f *= 2f64.powi(64);
        e -= 64;
    }
    e + ((f.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `v·2^k`, exactly whenever the result is a normal number: by two factors, each a power of two
/// the type holds for every `k` that [`power_of`] gives or its negative.
fn times_two_to<S: Scalar>(v: S, k: i32) -> S {
    let half = k / 2;
    v.scale(S::Real::TWO.powi(half))
        .scale(S::Real::TWO.powi(k - half))
}

/// [`times_two_to`] for a real number.
fn real_times_two_to<R: Float>(v: R, k: i32) -> R {
    let half = k / 2;
    v * R::TWO.powi(half) * R::TWO.powi(k - half)
}

/// The 2-norm of the values, the squares summed after the values are divided by the power of two
/// of the largest, so that none overflows or underflows.
fn norm2<S: Scalar>(values: &[S]) -> S::Real {
    let k = power_of(values.iter().copied());
    let sum = values
        .iter()
        .map(|&a| times_two_to(a, -k).abs2())
        .fold(S::Real::ZERO, |s, a| s + a);
    real_times_two_to(sum.sqrt(), k)
}

/// The weights of the current step: the right singular vector of the smallest singular value
/// of the Loewner matrix on the free samples (several summed, over the square root of their
/// number, when the smallest is repeated), the matrix padded with zero rows when there are fewer
/// free samples than columns. `ill` turns on, for good, the scaling of the columns to unit norm,
/// when the condition number has passed `1/(3ε)`.
fn smallest_singular_vector<S: Scalar>(
    loewner: &[Vec<S>],
    free: &[bool],
    ill: &mut bool,
) -> Vec<S> {
    let cols = loewner.len();
    let rows: Vec<usize> = (0..free.len()).filter(|&i| free[i]).collect();
    let height = rows.len().max(cols);
    let mut data = Vec::with_capacity(height * cols);
    for column in loewner {
        data.extend(rows.iter().map(|&i| column[i]));
        data.resize(data.len() + height - rows.len(), S::zero());
    }
    let matrix = Matrix::from_column_major(height, cols, data);
    let limit = S::Real::ONE / (S::Real::from_f64(3.0) * S::Real::EPSILON);
    let mut svd = matrix.svd();
    if !*ill {
        let s = svd.singular_values();
        let (big, small) = (s[0], s[s.len() - 1]);
        *ill = if small == S::Real::ZERO {
            big > S::Real::ZERO
        } else {
            big / small > limit
        };
    }
    let mut norms = vec![S::Real::ONE; cols];
    if *ill {
        for (k, column) in loewner.iter().enumerate() {
            let entries: Vec<S> = rows.iter().map(|&i| column[i]).collect();
            let n = norm2(&entries);
            if n > S::Real::ZERO && n.is_finite() {
                norms[k] = n;
            }
        }
        let scaled: Vec<S> = matrix
            .as_slice()
            .iter()
            .enumerate()
            .map(|(e, &a)| a.scale(S::Real::ONE / norms[e / height]))
            .collect();
        svd = Matrix::from_column_major(height, cols, scaled).svd();
    }
    let s = svd.singular_values();
    let least = s[s.len() - 1];
    let ties: Vec<usize> = (0..s.len()).filter(|&k| s[k] == least).collect();
    let v = svd.v();
    let root = S::Real::from_f64(ties.len() as f64).sqrt();
    (0..cols)
        .map(|r| {
            let sum = ties.iter().fold(S::zero(), |acc, &k| acc + v.column(k)[r]);
            sum.scale(S::Real::ONE / (root * norms[r]))
        })
        .collect()
}

/// The approximation at every sample, from the Cauchy columns of the support points: `n/d`
/// over the terms of nonzero weight, and the data value where `d` is not finite, which is at the
/// support points themselves.
fn approximate<S: Scalar>(r: &mut [S], cauchy: &[Vec<S>], w: &[S], support: &[usize], v: &[S]) {
    let terms: Vec<usize> = (0..w.len()).filter(|&k| w[k] != S::zero()).collect();
    for (i, ri) in r.iter_mut().enumerate() {
        let (mut n, mut d) = (S::zero(), S::zero());
        for &k in &terms {
            let c = cauchy[k][i] * w[k];
            n += c * v[support[k]];
            d += c;
        }
        *ri = if d.is_finite() { n / d } else { v[i] };
    }
}

/// What the barycentric form gives at a point: the support point it is (or is so close to that
/// its term overflows), whose datum is the value, or the value of the formula.
enum At<S> {
    Support(usize),
    Value(S),
}

/// The barycentric form at `x`, from the support values `values` as the weights were found for.
fn barycentric<S: Scalar>(x: S, points: &[S], values: &[S], weights: &[S]) -> At<S> {
    if x.re().is_nan() || x.im().is_nan() {
        return At::Value(x);
    }
    if !x.is_finite() {
        let (n, d) = points
            .iter()
            .zip(values)
            .zip(weights)
            .fold((S::zero(), S::zero()), |(n, d), ((_, &f), &w)| {
                (n + w * f, d + w)
            });
        return At::Value(n / d);
    }
    let (mut n, mut d) = (S::zero(), S::zero());
    for (j, ((&z, &f), &w)) in points.iter().zip(values).zip(weights).enumerate() {
        let difference = x - z;
        if difference == S::zero() {
            return At::Support(j);
        }
        let term = w / difference;
        if !term.is_finite() {
            return At::Support(j);
        }
        n += term * f;
        d += term;
    }
    At::Value(n / d)
}

/// The residue of the barycentric form at the pole `p`, `n(p)/d′(p)`.
fn residue<S: Scalar>(
    p: Complex<S::Real>,
    points: &[S],
    values: &[S],
    weights: &[S],
) -> Complex<S::Real> {
    let (mut n, mut slope) = (Complex::ZERO, Complex::ZERO);
    for ((&z, &f), &w) in points.iter().zip(values).zip(weights) {
        let c = Complex::ONE / (p - z.to_complex());
        n += w.to_complex() * f.to_complex() * c;
        slope -= w.to_complex() * c * c;
    }
    n / slope
}

/// A value of the type of the samples as a complex `f64`, exactly.
fn wide<S: Scalar>(v: S) -> Complex<f64> {
    Complex::new(v.re().to_f64(), v.im().to_f64())
}

/// The zeros of `Σ w_j/(x − z_j)` (every `w_j` nonzero) at the support points `z` of the type of
/// the samples, the coefficients `w` given in `f64`: by [`roots_in`], in `f64` whatever that type,
/// since the reduction can lose more digits than `f32` holds (see [`refine`]); the samples are
/// `f64` numbers exactly, and the zeros are rounded once to their type at the end.
fn roots<S: Scalar>(z: &[S], w: &[Complex<f64>]) -> Vec<Complex<S::Real>> {
    let found = if S::IS_COMPLEX {
        let z: Vec<Complex<f64>> = z.iter().map(|&p| wide(p)).collect();
        roots_in(&z, w)
    } else {
        let z: Vec<f64> = z.iter().map(|p| p.re().to_f64()).collect();
        let w: Vec<f64> = w.iter().map(|c| c.re).collect();
        roots_in(&z, &w)
    };
    let narrow = |v: f64| S::Real::from_f64(v);
    found
        .into_iter()
        .map(|p| Complex::new(narrow(p.re), narrow(p.im)))
        .collect()
}

/// The zeros of `Σ w_j/(x − z_j)` (every `w_j` nonzero), by the reduction described at
/// [`Aaa::poles`], refined by [`refine`].
fn roots_in<S: Scalar<Real = f64>>(z: &[S], w: &[S]) -> Vec<Complex<f64>> {
    let m = z.len();
    if m < 2 {
        return Vec::new();
    }
    // The zeros do not change when every coefficient is multiplied by one number: they are
    // divided by the power of two of the largest, so that no sum below overflows or underflows,
    // whatever their size.
    let k = power_of(w.iter().copied());
    let w: Vec<S> = w.iter().map(|&a| times_two_to(a, -k)).collect();
    let w = &w[..];
    let real = |v: f64| S::Real::from_f64(v);
    // Four shifts outside the data, the one that keeps |τ| largest against Σ|w_j·t_j| taken.
    let candidates: Vec<S> = if S::IS_COMPLEX {
        let centre = z.iter().fold(S::zero(), |s, &t| s + t) / S::from_real(real(m as f64));
        let radius = largest(z.iter().map(|&t| (t - centre).abs()));
        let radius = if radius > S::Real::ZERO {
            radius
        } else {
            S::Real::ONE
        };
        [(1.37, 0.7), (1.93, 2.9), (2.71, 4.4), (3.3, 5.6)]
            .iter()
            .map(|&(f, angle)| {
                let (sin, cos) = real(angle).sin_cos();
                centre + S::from_parts(radius * real(f) * cos, radius * real(f) * sin)
            })
            .collect()
    } else {
        let lo = z.iter().map(|t| t.re()).fold(S::Real::INFINITY, Float::min);
        let hi = z
            .iter()
            .map(|t| t.re())
            .fold(-S::Real::INFINITY, Float::max);
        let centre = (lo + hi) * S::Real::HALF;
        let half = (hi - lo) * S::Real::HALF;
        let half = if half > S::Real::ZERO {
            half
        } else {
            centre.abs().max(S::Real::ONE)
        };
        [1.37, -1.61, 2.93, -3.31]
            .iter()
            .map(|&f| S::from_real(centre + half * real(f)))
            .collect()
    };
    let mut best: Option<(S::Real, S, Vec<S>, S)> = None;
    for s in candidates {
        let t: Vec<S> = z.iter().map(|&p| S::one() / (p - s)).collect();
        if t.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let tau = w
            .iter()
            .zip(&t)
            .fold(S::zero(), |acc, (&a, &b)| acc + a * b);
        let total = w
            .iter()
            .zip(&t)
            .map(|(&a, &b)| (a * b).abs())
            .sum::<S::Real>();
        let rho = tau.abs() / total;
        if best.as_ref().is_none_or(|b| rho > b.0) {
            best = Some((rho, s, t, tau));
        }
    }
    let Some((_, shift, t, tau)) = best else {
        return Vec::new();
    };
    // The Householder reflection H with H·(conj(w)/‖w‖) along e₁: its columns 2..m are an
    // orthonormal basis of the vectors v with wᵀv = 0.
    let norm = norm2(w);
    let mut u: Vec<S> = w
        .iter()
        .map(|a| a.conj().scale(S::Real::ONE / norm))
        .collect();
    let head = u[0].abs();
    let alpha = if head > S::Real::ZERO {
        -u[0].scale(S::Real::ONE / head)
    } else {
        -S::one()
    };
    u[0] -= alpha;
    let uu = u.iter().map(|a| a.abs2()).sum::<S::Real>();
    let two = S::Real::TWO / uu;
    let basis = |row: usize, col: usize| {
        // H e_(col+1) = e_(col+1) − 2·u·conj(u_(col+1))/(u*u)
        let e = if row == col + 1 { S::one() } else { S::zero() };
        e - (u[row] * u[col + 1].conj()).scale(two)
    };
    let k = m - 1;
    // a = (w∘t)ᵀN, then GN = diag(t)·N − t·a/τ, then K = N*·GN.
    let wt: Vec<S> = w.iter().zip(&t).map(|(&a, &b)| a * b).collect();
    let a: Vec<S> = (0..k)
        .map(|c| (0..m).fold(S::zero(), |acc, j| acc + wt[j] * basis(j, c)))
        .collect();
    let gn: Vec<S> = (0..k)
        .flat_map(|c| {
            let (t, a) = (&t, &a);
            (0..m).map(move |j| t[j] * basis(j, c) - t[j] * a[c] / tau)
        })
        .collect();
    let mut kdata = Vec::with_capacity(k * k);
    for c in 0..k {
        for rrow in 0..k {
            let mut acc = S::zero();
            for j in 0..m {
                acc += basis(j, rrow).conj() * gn[j + c * m];
            }
            kdata.push(acc);
        }
    }
    let mu = Matrix::from_column_major(k, k, kdata).eigenvalues();
    let biggest = largest(t.iter().map(|v| v.abs()));
    let floor = real(16.0) * real(m as f64) * S::Real::EPSILON * biggest;
    let shift = shift.to_complex();
    let zc: Vec<Complex<S::Real>> = z.iter().map(|p| p.to_complex()).collect();
    let wc: Vec<Complex<S::Real>> = w.iter().map(|p| p.to_complex()).collect();
    let estimates: Vec<Complex<S::Real>> = mu
        .into_iter()
        .filter(|m| m.abs() > floor)
        .map(|m| shift + m.recip())
        .collect();
    refine(estimates, &zc, &wc, !S::IS_COMPLEX)
}

/// The most sweeps of [`aberth`] from the eigenvalue estimates; from them a few are enough, from
/// estimates that have crossed into the neighbours' basins a few dozen.
const SWEEPS: usize = 60;

/// The most restarts of the zeros [`refine`] has not found.
const RESTARTS: usize = 6;

/// The most sweeps of [`aberth`] in a restart, whose starts are farther from the zeros.
const RESTART_SWEEPS: usize = 200;

/// The zeros of `d(x) = Σ c_i/(x − z_i)` from the estimates of the eigenvalue problem: only those
/// [`gather`] certifies, each a zero to rounding and each once, and for real coefficients
/// (`real`) real or in exact conjugate pairs — as many as the estimates unless the iteration
/// below cannot bring some of them to rounding, which are then left out.
///
/// The reduction of [`roots`] divides by `τ = −d(s)`, which is small at every shift `s` outside
/// the data when the approximation's denominator is (an approximation of an entire function,
/// whose poles are spurious and far from the samples): the estimates are then off by many times
/// what the data determine, often by more than the distance between neighbouring zeros. They are
/// refined together by the Aberth–Ehrlich iteration ([`aberth`]), which keeps them apart and
/// converges from such starts, and each is then polished by [`polish`].
///
/// With real coefficients the zeros are real or in conjugate pairs, and the eigenvalues of the
/// real matrix come so: in that first iteration a real estimate stays real and the lower zero of
/// a pair is the conjugate of the upper. A pair close to the axis can come out as two real
/// numbers, which no iteration that keeps them real can reach: two real estimates that did not
/// settle are tried as a pair, started between them, kept if it lands nearer a zero than either.
///
/// What is then not a zero to rounding, or is a zero found twice, is started again, up to
/// [`RESTARTS`] times: the zeros certified are held fixed, so that the iteration keeps the others
/// away from them, and the others are iterated freely in the complex plane — free of the symmetry,
/// so that a pair can part into two real zeros and two real estimates can meet as a pair — first
/// from where they were, moved a little off it, then from points on circles round the data and the
/// zeros found, larger at each restart.
fn refine<R: Float>(
    estimates: Vec<Complex<R>>,
    z: &[Complex<R>],
    c: &[Complex<R>],
    real: bool,
) -> Vec<Complex<R>> {
    let n = estimates.len();
    // The partner (the lower root) of each upper root of a conjugate pair, and which roots follow
    // a partner rather than being iterated.
    let mut partner: Vec<Option<usize>> = vec![None; n];
    let mut follows = vec![false; n];
    if real {
        for k in 0..n {
            if estimates[k].im > R::ZERO {
                let pair = (0..n).find(|&l| !follows[l] && estimates[l] == estimates[k].conj());
                if let Some(l) = pair {
                    partner[k] = Some(l);
                    follows[l] = true;
                }
            }
        }
    }
    let mut on_axis: Vec<bool> = estimates.iter().map(|e| real && e.im == R::ZERO).collect();
    let rounding = rounding_of();
    let residual = |p: Complex<R>| newton_step(p, z, c).1;
    let mut x = estimates;
    let mut done = follows.clone();
    aberth(&mut x, z, c, &partner, &on_axis, &mut done, SWEEPS);
    for k in (0..n).filter(|&k| !follows[k]) {
        x[k] = polish(x[k], z, c);
        if on_axis[k] {
            x[k].im = R::ZERO;
        }
        if let Some(l) = partner[k] {
            x[l] = x[k].conj();
        }
    }
    if real {
        let mut stray: Vec<usize> = (0..n)
            .filter(|&k| on_axis[k] && residual(x[k]) > rounding)
            .collect();
        stray.sort_by(|&a, &b| {
            x[a].re
                .partial_cmp(&x[b].re)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for i in (1..stray.len()).step_by(2) {
            let (a, b) = (stray[i - 1], stray[i]);
            let mid = (x[a].re + x[b].re) * R::HALF;
            let half = ((x[b].re - x[a].re) * R::HALF).max(R::EPSILON.sqrt() * mid.abs());
            let mut y = x.clone();
            y[a] = Complex::new(mid, half.max(R::MIN_POSITIVE));
            y[b] = y[a].conj();
            let mut pairs = partner.clone();
            pairs[a] = Some(b);
            let mut axis = on_axis.clone();
            (axis[a], axis[b]) = (false, false);
            let mut fixed = vec![true; n];
            fixed[a] = false;
            aberth(&mut y, z, c, &pairs, &axis, &mut fixed, SWEEPS);
            y[a] = polish(y[a], z, c);
            if residual(y[a]) < residual(x[a]).min(residual(x[b])) {
                (x[a], x[b]) = (y[a], y[a].conj());
                partner = pairs;
                on_axis = axis;
            }
        }
    }
    let (mut found, mut left) = gather(&x, z, c, real, rounding);
    let count = R::from_f64(z.len() as f64);
    let centre = z.iter().fold(Complex::ZERO, |s, &p| s + p) / count;
    let free = vec![None; n];
    let off = vec![false; n];
    for attempt in 0..RESTARTS {
        if found.len() >= n {
            break;
        }
        let fixed = found.len();
        let size = largest(z.iter().chain(&found).map(|&p| (p - centre).abs()));
        let mut y = found.clone();
        for i in 0..n - fixed {
            // Steps of the golden angle: no two starts, and no start and the conjugate of
            // another, on one ray.
            let turn = R::from_f64(2.399_963_229_728_653 * (1 + i + 5 * attempt) as f64);
            y.push(if attempt == 0 && i < left.len() {
                left[i] + Complex::from_polar(size * R::from_f64(1e-3), turn)
            } else {
                let radius = size * R::from_f64(1.0 + 0.5 * attempt as f64);
                centre + Complex::from_polar(radius, turn)
            });
        }
        let mut done: Vec<bool> = (0..n).map(|k| k < fixed).collect();
        aberth(&mut y, z, c, &free, &off, &mut done, RESTART_SWEEPS);
        for p in &mut y[fixed..] {
            *p = polish(*p, z, c);
        }
        (found, left) = gather(&y, z, c, real, rounding);
    }
    found
}

/// The residual of [`newton_step`] at most which a point counts as a zero of `Σ c_i/(x − z_i)`:
/// `8ε`. The residual is, to first order, the distance from the zero in units of how far relative
/// changes of `ε` in the coefficients and in the point move it; at the zeros found it is below
/// `3ε` in every case measured (tens of thousands of approximations, up to a hundred support
/// points). `4·(m + 1)·ε`, a bound on the rounding of the sum, was too loose: a conjugate pair
/// 200 of those units from the two real zeros it stood for passed it, at `109ε`.
fn rounding_of<R: Float>() -> R {
    R::from_f64(8.0) * R::EPSILON
}

/// The zeros of `d(x) = Σ c_i/(x − z_i)` among the points `x`, and the points that are not.
///
/// A point is a zero when it is finite and the residual of [`newton_step`] there — the size of
/// the relative changes of the `c_i` and of the point itself that make it an exact zero — is at
/// most `rounding`. Of two zeros that are one zero found twice ([`twice`]) the one of smaller
/// residual is kept, and the other goes with the points that are not zeros. With real
/// coefficients (`real`) a zero off the axis is given with its conjugate, a zero to the same
/// residual, and a zero found within the rounding of that conjugate is taken to be it and not
/// counted again; a zero off the axis without one, whose real part is a zero too and within its
/// radius of it (where the residual stays below `rounding`, to first order), is put on the axis.
fn gather<R: Float>(
    x: &[Complex<R>],
    z: &[Complex<R>],
    c: &[Complex<R>],
    real: bool,
    rounding: R,
) -> (Vec<Complex<R>>, Vec<Complex<R>>) {
    // (zero, residual, spread)
    let mut zeros: Vec<(Complex<R>, R, R)> = Vec::new();
    let mut left = Vec::new();
    for &p in x {
        let (_, residual, spread) = newton_step(p, z, c);
        if p.is_finite() && residual <= rounding {
            zeros.push((p, residual, spread));
        } else {
            left.push(p);
        }
    }
    zeros.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let all: Vec<Complex<R>> = zeros.iter().map(|q| q.0).collect();
    let mut kept: Vec<(Complex<R>, R, R)> = Vec::new();
    for q in zeros {
        if kept
            .iter()
            .any(|k| twice(k.0, q.0, rounding * k.2.min(q.2), &all, z, c))
        {
            left.push(q.0);
        } else {
            kept.push(q);
        }
    }
    if !real {
        return (kept.into_iter().map(|k| k.0).collect(), left);
    }
    let mut out = Vec::with_capacity(kept.len() + 1);
    let mut taken = vec![false; kept.len()];
    for i in 0..kept.len() {
        if taken[i] {
            continue;
        }
        taken[i] = true;
        let (p, _, spread) = kept[i];
        if p.im == R::ZERO {
            out.push(p);
            continue;
        }
        let mirror = p.conj();
        let partner = (i + 1..kept.len())
            .filter(|&l| !taken[l])
            .map(|l| (l, (kept[l].0 - mirror).abs()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        match partner {
            Some((l, distance)) if distance <= R::TWO * rounding * spread.min(kept[l].2) => {
                taken[l] = true;
            }
            _ => {
                let axis = Complex::new(p.re, R::ZERO);
                if p.im.abs() <= rounding * spread && newton_step(axis, z, c).1 <= rounding {
                    out.push(axis);
                    continue;
                }
            }
        }
        out.push(p);
        out.push(mirror);
    }
    (out, left)
}

/// Whether the zeros `a` and `b` of `d(x) = Σ c_i/(x − z_i)` are one zero found twice, where
/// `near` is the distance within which a point is a zero to rounding (the radius of [`gather`],
/// for the smaller spread) and `all` holds every zero found.
///
/// Two zeros farther apart than twice that are two. Closer, the slope of the polynomial
/// `p(x) = d(x)·Π(x − z_i)` decides — it points the same way at two points beside one simple
/// zero, and opposite ways at two zeros that close (numerically double), which are both kept —
/// but only where the other zeros are far enough away not to turn it between `a` and `b`
/// (`|a − b|·Σ 1/|a − x_k|` below a half over the others): within a cluster of zeros nothing
/// tells, and both are kept, as the simultaneous iteration that found them keeps its estimates
/// apart. At a zero `p′ = F′·Π_{i≠j}(x − z_i)` for `F = (x − z_j)·d` written about the support
/// point `z_j` of `a`'s largest term, so the directions are compared through `F′` and the
/// ratios `(a − z_i)/(b − z_i)`.
fn twice<R: Float>(
    a: Complex<R>,
    b: Complex<R>,
    near: R,
    all: &[Complex<R>],
    z: &[Complex<R>],
    c: &[Complex<R>],
) -> bool {
    use core::cmp::Ordering::{Equal, Less};
    let gap = (a - b).abs();
    if !matches!(gap.partial_cmp(&(R::TWO * near)), Some(Less | Equal)) {
        return false;
    }
    let crowd = all
        .iter()
        .filter(|&&x| x != a && x != b)
        .map(|&x| gap / (a - x).abs())
        .fold(R::ZERO, |s, t| s + t);
    if !matches!(crowd.partial_cmp(&R::HALF), Some(Less)) {
        return false;
    }
    let j = largest_term(a, z, c);
    let mut turn = slope_about(a, z, c, j) * slope_about(b, z, c, j).conj();
    for (i, &p) in z.iter().enumerate() {
        if i != j {
            // The direction only, kept at size 1.
            turn *= (a - p) * (b - p).conj();
            let size = turn.abs();
            if !(size > R::ZERO && size.is_finite()) {
                return false;
            }
            turn /= size;
        }
    }
    turn.re > R::ZERO
}

/// The index of the largest term `|c_i/(x − z_i)|` of the sum at `x`, the first support point
/// equal to `x` if there is one.
fn largest_term<R: Float>(x: Complex<R>, z: &[Complex<R>], c: &[Complex<R>]) -> usize {
    let mut j = 0;
    let mut big = -R::ONE;
    for (i, (&p, &q)) in z.iter().zip(c).enumerate() {
        let distance = (x - p).abs();
        if distance == R::ZERO {
            return i;
        }
        let size = q.abs() / distance;
        if size > big {
            big = size;
            j = i;
        }
    }
    j
}

/// The derivative of `F = (x − z_j)·d(x) = c_j + (x − z_j)·Σ_{i≠j} c_i/(x − z_i)` at `x`.
fn slope_about<R: Float>(
    x: Complex<R>,
    z: &[Complex<R>],
    c: &[Complex<R>],
    j: usize,
) -> Complex<R> {
    let y = x - z[j];
    let (mut rest, mut bend) = (Complex::ZERO, Complex::ZERO);
    for (i, (&p, &q)) in z.iter().zip(c).enumerate() {
        if i != j {
            let difference = x - p;
            let t = q / difference;
            rest += t;
            bend += t / difference;
        }
    }
    rest - y * bend
}

/// Sweeps of the Aberth–Ehrlich iteration (O. Aberth, *Math. Comp.* 27, 1973) on the polynomial
/// `p(x) = d(x)·Π(x − z_i)` over the roots `x` not `done`, each updated in place,
/// `x_k ← x_k − N_k/(1 − N_k·Σ_{l≠k} 1/(x_k − x_l))` with the Newton step `N_k` of
/// [`newton_step`], until none moves or after `sweeps` sweeps. The roots that are `done` stay
/// where they are and keep the others away. A root is done when its relative residual is at most
/// `ε` or its correction is below the rounding of `x_k`. A root `on_axis` stays real, and the
/// `partner` of a root is set to its conjugate.
fn aberth<R: Float>(
    x: &mut [Complex<R>],
    z: &[Complex<R>],
    c: &[Complex<R>],
    partner: &[Option<usize>],
    on_axis: &[bool],
    done: &mut [bool],
    sweeps: usize,
) {
    for _ in 0..sweeps {
        let mut moved = false;
        for k in 0..x.len() {
            if done[k] {
                continue;
            }
            let (step, residual, _) = newton_step(x[k], z, c);
            if residual <= R::EPSILON {
                done[k] = true;
                continue;
            }
            let mut repulsion = Complex::ZERO;
            for (l, &other) in x.iter().enumerate() {
                if l != k {
                    repulsion += (x[k] - other).recip();
                }
            }
            let correction = step / (Complex::ONE - step * repulsion);
            if !correction.is_finite() {
                done[k] = true;
                continue;
            }
            let mut next = x[k] - correction;
            if on_axis[k] {
                next.im = R::ZERO;
            }
            if next == x[k] || correction.abs() <= R::EPSILON * x[k].abs() {
                done[k] = true;
            }
            moved |= next != x[k];
            x[k] = next;
            if let Some(l) = partner[k] {
                x[l] = next.conj();
            }
        }
        if !moved {
            break;
        }
    }
}

/// The most Newton steps [`polish`] takes; after [`refine`]'s iteration one or two are left.
const POLISH_STEPS: usize = 6;

/// The zero of `d(x) = Σ c_i/(x − z_i)` near the estimate `x`, by Newton's method on the
/// polynomial `p(x) = d(x)·Π(x − z_i)` that has the same zeros (see [`newton_step`]). A step is
/// kept only while it lowers the relative residual of [`newton_step`], the size of the relative
/// changes of the `c_i` and of `x` that would make `x` an exact zero: about `ε` at a zero to
/// rounding, and not lowered by a step away from one.
///
/// Newton's method on `d` itself goes wrong where a zero lies next to a support point `z_j` of
/// small `c_j`, closer to it than the error of the estimate: `d` has a pole at `z_j` beside its
/// zero, every step from the far side of the zero moves away from both, and `|d|` still falls
/// there, to the size of the other terms. The polynomial has no pole: near `z_j` it is nearly
/// linear, and one step lands on the zero.
fn polish<R: Float>(mut x: Complex<R>, z: &[Complex<R>], c: &[Complex<R>]) -> Complex<R> {
    let (mut step, mut residual, _) = newton_step(x, z, c);
    for _ in 0..POLISH_STEPS {
        let next = x - step;
        if !next.is_finite() || next == x {
            break;
        }
        let (s, r, _) = newton_step(next, z, c);
        if r < residual {
            (x, step, residual) = (next, s, r);
        } else {
            break;
        }
    }
    x
}

/// The Newton step `p(x)/p′(x)` of the polynomial `p(x) = d(x)·Π(x − z_i)`, for
/// `d(x) = Σ c_i/(x − z_i)`, the relative residual of `x` as a zero, and its spread, the change
/// of `x` that changes the residual by 1 to first order.
///
/// Both are written about the support point `z_j` of the largest term, `y = x − z_j`, so that
/// they stay finite at and next to it: with `t_i = c_i/(x − z_i)` and `r = Σ_{i≠j} t_i`,
/// `F = y·d = c_j + y·r` and `y·d·p′/p = Σ_i y·(d − t_i)/(x − z_i)
/// = r + Σ_{i≠j} (c_j + y·(r − t_i))/(x − z_i)`, their ratio the step. The sum `d − t_j`, where
/// `t_j` alone is large, is `r`, summed without it.
///
/// The residual is `|F| / D` with `D = |c_j| + |y|·Σ_{i≠j}|t_i| + |x|·|F′|`: the size of the
/// relative changes of the coefficients and of `x` itself that make `x` an exact zero, about `ε`
/// at the float nearest a zero. Without the term in `x` it would be about 1 at every float near a
/// zero that lies closer to a support point than the floats there are apart (the sine's zero at a
/// sample, where the sine is `1e−16`), and the float nearest it would look no better than a
/// point far off. The spread is `D/|F′|`.
fn newton_step<R: Float>(x: Complex<R>, z: &[Complex<R>], c: &[Complex<R>]) -> (Complex<R>, R, R) {
    let j = largest_term(x, z, c);
    let y = x - z[j];
    let (mut rest, mut total) = (Complex::ZERO, R::ZERO);
    for (i, (&p, &q)) in z.iter().zip(c).enumerate() {
        if i != j {
            let t = q / (x - p);
            rest += t;
            total += t.abs();
        }
    }
    let scaled = c[j] + y * rest;
    let mut slope = rest;
    // Σ_{i≠j} t_i/(x − z_i), for the derivative (y·d)′ = r − y·Σ_{i≠j} t_i/(x − z_i).
    let mut bend = Complex::ZERO;
    for (i, (&p, &q)) in z.iter().zip(c).enumerate() {
        if i != j {
            let difference = x - p;
            let t = q / difference;
            slope += (c[j] + y * (rest - t)) / difference;
            bend += t / difference;
        }
    }
    let derivative = rest - y * bend;
    let size = c[j].abs() + y.abs() * total + x.abs() * derivative.abs();
    (scaled / slope, scaled.abs() / size, size / derivative.abs())
}

/// The clean-up of the approximation whose support points are the samples `chosen` of `z` and
/// `v`: the poles whose residue over their distance from the nearest sample is below `tol` times
/// the geometric mean of the nonzero `|f|` are doublets; the support point nearest each is
/// removed, and the weights solved for again on the samples that are not support points, by the
/// right singular vector of the smallest singular value. Returns the number of doublets, the
/// samples kept as support points and their weights; `None` when there is none to remove (or
/// every support point would go).
fn clean_up<S: Scalar>(
    z: &[S],
    v: &[S],
    chosen: &[usize],
    weights: &[S],
    tol: S::Real,
) -> Option<(usize, Vec<usize>, Vec<S>)> {
    let points: Vec<S> = chosen.iter().map(|&i| z[i]).collect();
    let values: Vec<S> = chosen.iter().map(|&i| v[i]).collect();
    let (points, values) = (&points[..], &values[..]);
    let (mut sum, mut count) = (S::Real::ZERO, 0usize);
    for f in v {
        let a = f.abs();
        if a > S::Real::ZERO {
            sum += a.ln();
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let mean = (sum / S::Real::from_f64(count as f64)).exp();
    let w: Vec<Complex<f64>> = weights.iter().map(|&a| wide(a)).collect();
    let poles = roots(points, &w);
    let mut drop = vec![false; points.len()];
    let mut found = 0;
    for p in poles {
        let res = residue(p, points, values, weights).abs();
        let distance = smallest(z.iter().map(|t| (p - t.to_complex()).abs()));
        if res / distance < tol * mean {
            found += 1;
            let mut nearest = 0;
            let mut best = S::Real::INFINITY;
            for (j, q) in points.iter().enumerate() {
                let d = (p - q.to_complex()).abs();
                if d < best {
                    best = d;
                    nearest = j;
                }
            }
            drop[nearest] = true;
        }
    }
    let left: Vec<usize> = (0..points.len()).filter(|&j| !drop[j]).collect();
    if found == 0 || left.is_empty() {
        return None;
    }
    let kept: Vec<usize> = left.iter().map(|&j| chosen[j]).collect();
    let mut free = vec![true; z.len()];
    for &i in &kept {
        free[i] = false;
    }
    let loewner: Vec<Vec<S>> = kept
        .iter()
        .map(|&j| {
            let (p, q) = (z[j], v[j]);
            z.iter().zip(v).map(|(&t, &f)| (f - q) / (t - p)).collect()
        })
        .collect();
    let rows: Vec<usize> = (0..z.len()).filter(|&i| free[i]).collect();
    let cols = left.len();
    let height = rows.len().max(cols);
    let mut data = Vec::with_capacity(height * cols);
    for column in &loewner {
        data.extend(rows.iter().map(|&i| column[i]));
        data.resize(data.len() + height - rows.len(), S::zero());
    }
    let svd = Matrix::from_column_major(height, cols, data).svd();
    let w: Vec<S> = svd.v().column(cols - 1).to_vec();
    let keep: Vec<usize> = (0..cols).filter(|&k| w[k] != S::zero()).collect();
    Some((
        found,
        keep.iter().map(|&k| kept[k]).collect(),
        keep.iter().map(|&k| w[k]).collect(),
    ))
}

/// The smallest of the values, infinity for none.
fn smallest<R: Float>(values: impl Iterator<Item = R>) -> R {
    values.fold(R::INFINITY, |m, x| m.min(x))
}
