//! What the two stiff methods share: the Jacobian of the right-hand side, from the caller's
//! function or by finite differences, and the tolerance their Newton iterations are held to.
//!
//! Both methods solve their implicit equations by a simplified Newton iteration, whose matrix is
//! formed from an approximation `J ≈ ∂f/∂y` that is kept for as many steps as it serves: the
//! iteration converges with any `J` near enough the true Jacobian, more slowly the further it is,
//! and the methods watch the rate of convergence to decide when a new one is worth its cost.

use super::control::{all_finite, scaled_squares};
use crate::math::matrix::Matrix;
use crate::math::numerics::Real;

/// The Jacobian argument of a stiff method that has none of its own: a function pointer that is
/// never called, the Jacobian being formed by finite differences instead. It is the default
/// second type parameter of [`RadauIia`](super::RadauIia) and [`Bdf`](super::Bdf).
///
/// # Examples
///
/// ```
/// use a::math::numerics::{Bdf, NoJacobian, RadauIia};
///
/// // The types `RadauIia::new()` and `Bdf::new()` have, written out.
/// let radau: RadauIia<f64, NoJacobian<f64>> = RadauIia::new();
/// let bdf: Bdf<f64, NoJacobian<f64>> = Bdf::new();
/// # let _ = (radau, bdf);
/// ```
pub type NoJacobian<T> = fn(T, &Matrix<T>, &mut Matrix<T>);

/// The Jacobian of a stiff method: the caller's function, or forward differences whose
/// increments adapt to the problem, column by column, from one Jacobian to the next.
///
/// The differences are those of MATLAB's `numjac`, which Shampine and Reichelt describe with the
/// suite of `ode15s`[^sr] after Salane[^salane]: column `j` steps `yⱼ` by `δ = σⱼ·s·max(|yⱼ|, atol)`,
/// `s` the sign of `fⱼ` (so the state moves as the solution does), from a factor `σⱼ` that starts
/// at `√ε`. When the largest change of `f` in the column is lost in the rounding of `f` (below
/// `ε^(7/8)` of it), the column is evaluated again with ten times the step, and the better of the
/// two kept; then `σⱼ` is raised tenfold if the change was below `ε^(3/4)` of `f`, and lowered
/// tenfold if above `ε^(1/4)`, for the next time. A fixed rule cannot serve every scaling: the
/// `√(ε·max(10⁻⁵, |yⱼ|))` of RADAU5 steps a component of size 10⁻¹³ (Robertson's `y₂`) by
/// `5·10⁻¹¹`, and `√ε·max(|yⱼ|, atol)` alone loses a component passing through zero in the rounding
/// of the others.
///
/// [^sr]: L. F. Shampine, M. W. Reichelt, "The MATLAB ODE Suite", *SIAM Journal on Scientific
///     Computing* 18 (1997), 1–22, section 5.
///
/// [^salane]: D. E. Salane, "Adaptive routines for forming Jacobians numerically", Sandia
///     National Laboratories report SAND86-1319, 1986.
#[derive(Debug, Clone)]
pub(super) struct Jacobian<T> {
    /// The factors `σⱼ` of the increments, one for each column.
    factor: Vec<T>,
}

impl<T: Real> Jacobian<T> {
    /// The increments of a state of `n` components, before any Jacobian was formed.
    pub fn new(n: usize) -> Self {
        Jacobian {
            factor: vec![T::EPSILON.sqrt(); n],
        }
    }

    /// Writes the Jacobian `∂f/∂y` at `(t, y)` to the `n × n` matrix `out`: by `jacobian` when
    /// the caller gave one, which is called with `out` zeroed, and otherwise by the forward
    /// differences of the type from `fy = f(t, y)`, `n` evaluations of `f` and one more for each
    /// column whose difference was lost in rounding. Whether every entry is finite. `work` and
    /// `column` are states of the shape of `y`, overwritten.
    #[allow(clippy::too_many_arguments)]
    pub fn form<F, J>(
        &mut self,
        f: &mut F,
        jacobian: Option<&mut J>,
        t: T,
        y: &Matrix<T>,
        fy: &Matrix<T>,
        atol: T,
        out: &mut Matrix<T>,
        work: &mut Matrix<T>,
        column: &mut Matrix<T>,
    ) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
        J: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        for v in out.as_mut_slice() {
            *v = T::ZERO;
        }
        if let Some(jacobian) = jacobian {
            jacobian(t, y, out);
            return all_finite(out.as_slice());
        }
        let n = y.as_slice().len();
        let reject = T::EPSILON.powf(T::from_f64(0.875));
        let small = T::EPSILON.powf(T::from_f64(0.75));
        let big = T::EPSILON.powf(T::from_f64(0.25));
        let (least, most) = (T::from_f64(1e3) * T::EPSILON, big);
        work.as_mut_slice().copy_from_slice(y.as_slice());
        for j in 0..n {
            let yj = y.as_slice()[j];
            let size = yj.abs().max(atol);
            let size = if size > T::ZERO { size } else { T::ONE };
            let scale = if fy.as_slice()[j] >= T::ZERO {
                size
            } else {
                -size
            };
            // The step really taken, which rounding may change, and never nought.
            let mut factor = self.factor[j];
            let mut delta = (yj + factor * scale) - yj;
            while delta == T::ZERO && factor < most {
                factor *= T::from_f64(10.0);
                delta = (yj + factor * scale) - yj;
            }
            let (mut change, mut size_f) = self.column(f, t, j, yj + delta, work, fy, column);
            let quotients = |out: &mut Matrix<T>, column: &Matrix<T>, delta: T| {
                for (i, (&c, &f0)) in column.as_slice().iter().zip(fy.as_slice()).enumerate() {
                    out[(i, j)] = (c - f0) / delta;
                }
            };
            quotients(out, column, delta);
            if change < reject * size_f && factor < most {
                // The difference is in the rounding of f: try a step ten times longer, and keep
                // whichever difference stands out more from its f.
                let tried = (factor * T::from_f64(10.0)).min(most);
                let delta2 = (yj + tried * scale) - yj;
                let (change2, size2) = self.column(f, t, j, yj + delta2, work, fy, column);
                if change * size2 < change2 * size_f {
                    quotients(out, column, delta2);
                    (change, size_f, factor) = (change2, size2, tried);
                }
            }
            if change < small * size_f {
                factor *= T::from_f64(10.0);
            } else if change > big * size_f {
                factor *= T::from_f64(0.1);
            }
            self.factor[j] = factor.max(least).min(most);
        }
        all_finite(out.as_slice())
    }

    /// `f` with component `j` of the state set to `yj`, into `column`, and the largest change of
    /// `f` it made with the size of `f` in that row, `max(|f₀|, |f|)`. `work` holds the state with
    /// that component restored when it returns.
    #[allow(clippy::too_many_arguments)]
    fn column<F>(
        &self,
        f: &mut F,
        t: T,
        j: usize,
        yj: T,
        work: &mut Matrix<T>,
        fy: &Matrix<T>,
        column: &mut Matrix<T>,
    ) -> (T, T)
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let restore = work.as_slice()[j];
        work.as_mut_slice()[j] = yj;
        f(t, work, column);
        work.as_mut_slice()[j] = restore;
        let (mut change, mut size) = (T::ZERO, T::ZERO);
        for (&new, &old) in column.as_slice().iter().zip(fy.as_slice()) {
            let d = (new - old).abs();
            // A change that is not a number marks the column, as the largest of all.
            if d > change || d.is_nan() {
                change = d;
                size = old.abs().max(new.abs());
                if d.is_nan() {
                    break;
                }
            }
        }
        (change, size)
    }
}

/// The rounding of the state `y` in the norm of the tolerances, `10·ε·ρ` with `ρ` the size of `y`
/// in that norm: a Newton correction below it changes the iterate only in its last bits, and no
/// further iterate can do better. It is the floor of [`newton_tolerance`].
pub(super) fn rounding_floor<T: Real>(y: &[T], rtol: T, atol: T) -> T {
    let size = if y.is_empty() {
        T::ZERO
    } else {
        (scaled_squares(y, y, y, rtol, atol) / T::from_f64(y.len() as f64)).sqrt()
    };
    T::from_f64(10.0) * T::EPSILON * size
}

/// The tolerance of a Newton iteration, in the norm the step is judged by: the iteration stops
/// when its error, estimated from the rate of convergence, is below this fraction of the
/// tolerances.
///
/// It is the rule of RADAU5, which SciPy's BDF uses too, `max(10·ε/rtol, min(0.03, √rtol))`: a
/// few hundredths of the tolerance, less at stringent tolerances, and never below what the
/// rounding of the state allows. The first term is written here as `10·ε·ρ` (the
/// [`rounding_floor`]), with `ρ` the size of the state `y` in the norm of the tolerances, which
/// is `1/rtol` when `atol` is negligible — the published rule — and which still means the
/// rounding of the state when `rtol` is zero; a purely absolute tolerance takes `0.03` for the
/// second term.
pub(super) fn newton_tolerance<T: Real>(y: &[T], rtol: T, atol: T) -> T {
    let cap = T::from_f64(0.03);
    let wanted = if rtol > T::ZERO {
        cap.min(rtol.sqrt())
    } else {
        cap
    };
    rounding_floor(y, rtol, atol).max(wanted)
}
