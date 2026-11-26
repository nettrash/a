//! Linear programming by the simplex method, primal and dual.
//!
//! A linear programme asks for the point that makes a linear function as small, or as large, as
//! it can be among the points that satisfy linear inequalities, linear equations and bounds on
//! each variable:
//!
//! ```text
//! minimise    cᵀx
//! subject to  A_le·x ≤ b_le,   A_eq·x = b_eq,   l ≤ x ≤ u.
//! ```
//!
//! A [`LinearProgram`] is built from the costs, then given its constraints and bounds; a variable
//! that is given no bounds is `x ≥ 0`, as in most statements of the problem, and either bound may
//! be infinite. [`LinearProgram::solve`] answers with an [`LpResult`]: how the method ended
//! ([`LpStatus`]), the point, the objective there, the slacks of the inequalities, the dual values
//! and the number of pivots.
//!
//! The method, what each status has been checked to mean, the tolerances, the sign convention
//! of the dual values and the limits are set out on [`LinearProgram`].
//!
//! # Examples
//!
//! ```
//! use a::math::matrix::Matrix;
//! use a::math::numerics::{LinearProgram, LpStatus};
//!
//! // Maximise 3x + 5y subject to x ≤ 4, 2y ≤ 12 and 3x + 2y ≤ 18, with x and y at least zero.
//! let a = Matrix::from_rows(&[[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]);
//! let r = LinearProgram::maximize(&[3.0_f64, 5.0])
//!     .subject_to_le(&a, &[4.0, 12.0, 18.0])
//!     .solve()
//!     .unwrap();
//! assert_eq!(r.status, LpStatus::Optimal);
//! assert!((r.x[0] - 2.0).abs() < 1e-12 && (r.x[1] - 6.0).abs() < 1e-12);
//! assert!((r.objective - 36.0).abs() < 1e-12);
//! // The first inequality is slack by 2 and worth nothing; a unit more of the second is worth
//! // 3/2 and of the third 1.
//! assert!((r.slack[0] - 2.0).abs() < 1e-12);
//! assert!(r.dual_le[0] == 0.0 && (r.dual_le[1] - 1.5).abs() < 1e-12);
//! assert!((r.dual_le[2] - 1.0).abs() < 1e-12);
//! ```

use super::{DEFAULT_MAX_ITER, Real};
use crate::math::complex::Float;
use crate::math::matrix::{Lu, Matrix, Scalar};

mod dual;
mod factor;

/// The number of rows from which [`LpOptions::new`] has the dual simplex method solve a
/// programme: [`LpOptions::dual_threshold`].
// Measured on an Apple M4 Max, each programme solved by either method alone: on the random
// programmes of the scoreboard's kind (dense and at density 0.1, 1 to 96 rows, 2 to 600
// variables) the dual method is the faster from 3 rows; on the 600 random programmes of the
// tests (a fifth unbounded, a quarter infeasible) only from about 20. Against 32, the threshold
// 20 makes the battery's programmes of 20 to 23 and 24 to 31 rows 0.76 and 0.75 of their time
// (geometric means, twice; 16 makes those of 16 to 19 rows 1.03), the netlib-like assignment and
// transportation problems of 25 and 30 rows 0.47 and 0.56, and the random ones of 20 to 28 rows
// 0.10 to 0.23 (sweep 1 to 96 rows; 16, 20 and 24 against 32). Unbounded programmes, which the
// primal method finishes after the dual method's phase one, are the exception: the battery's of
// 20 to 31 rows take 1.08 to 1.22 times as long (and the netlib-like ones of 80 and 110 rows,
// dual already, 1.21 times the primal method's time).
const DUAL_THRESHOLD: usize = 20;

/// The magnitude from which a bound, in the units it was given in, is taken by the dual method
/// to stand for an infinite one, as HiGHS takes it: a variable placed at such a bound would put
/// its size into every row it meets.
const INFINITE_BOUND: f64 = 1e20;

/// How many changes of basis the product form absorbs before the basis is factorised afresh and
/// the basic variables are computed again from the data.
const REFACTOR_EVERY: usize = 20;

/// The fraction of a bound's tolerance by which Harris's ratio test relaxes the bound in its first
/// pass. A basic variable may end a step that far outside its bound, so it has to stay well
/// inside the tolerance the final point is checked to; the error a value carries beyond the
/// tolerance is not added, as a step it allowed would then break the bound by more than the
/// tolerance once the basis is factorised again and the value recomputed.
const HARRIS: f64 = 0.1;

/// How many units in the last place two quantities may differ by and still be taken for equal:
/// the tie of two ratios under Bland's rule and the rounding a certificate's margin has to clear.
const ROUNDING: f64 = 64.0;

/// The factor by which a bound on the error of a reduced cost, which is worked out to first order
/// from quantities that carry rounding of their own, is taken larger than it comes out.
const ERROR_SAFETY: f64 = 2.0;

/// How many times what the refinement of the multipliers changed a reduced cost pricing takes for
/// the error still in it. Pricing only proposes: the variable it chooses is checked against a
/// bound on the error of its reduced cost before it enters, and every answer against such bounds
/// for all the variables, so this estimate decides how much work is done, not what is right.
const PRICING_SAFETY: f64 = 4.0;

/// How many times the rows, then the columns, are scaled to their largest element.
const SCALE_SWEEPS: usize = 2;

/// How many times phase one may resume after phase two has begun — the point, recomputed from
/// the data when the basis is factorised, breaking a bound by more than its allowance — before
/// the method gives up. Over some fifty thousand solves of random and adversarial programmes in
/// `f64` it resumed at most once in any; it resumes at every factorisation only where the
/// rounding of a step is larger than the tolerance of the bounds (the cubes of Klee and Minty in
/// tens and hundreds without the scaling, beyond nine dimensions), and there the arithmetic
/// cannot keep the point feasible.
const RESUMPTIONS: usize = 10;

/// The magnitude of `x`, named through [`Float`]: the elements of a matrix have an absolute value
/// of their own, and both are in scope where a float is also a matrix element.
fn mag<T: Real>(x: T) -> T {
    Float::abs(x)
}

/// Whether `x` is finite, named through [`Float`] for the same reason.
fn finite<T: Real>(x: T) -> bool {
    Float::is_finite(x)
}

/// The inner product of two vectors of the same length.
fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut sum = T::ZERO;
    for (x, y) in a.iter().zip(b) {
        sum += *x * *y;
    }
    sum
}

/// `ε^(4/7)`: about `1.1·10⁻⁹` for `f64` and `1.1·10⁻⁴` for `f32`. It is the default tolerance and
/// the largest size, relative to the larger of one and the largest element of the entering
/// column, below which an element of that column is not pivoted on. The rounding of a solve with
/// a basis of moderate condition stays well below it, and what a well-scaled problem takes for a
/// nonzero stays well above it.
fn default_tolerance<T: Real>() -> T {
    T::EPSILON.powf(T::from_f64(4.0 / 7.0))
}

/// How a linear programme ended.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{LinearProgram, LpStatus};
///
/// // x ≤ 1 and x ≥ 2 cannot both hold.
/// let a = Matrix::from_rows(&[[1.0], [-1.0]]);
/// let r = LinearProgram::minimize(&[1.0])
///     .subject_to_le(&a, &[1.0, -2.0])
///     .solve()
///     .unwrap();
/// assert_eq!(r.status, LpStatus::Infeasible);
///
/// // Nothing stops x from growing, and the objective with it.
/// let r = LinearProgram::maximize(&[1.0]).solve().unwrap();
/// assert_eq!(r.status, LpStatus::Unbounded);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LpStatus {
    /// The point is optimal: it is feasible within the tolerance, and the dual values prove, to
    /// the tolerance, that no feasible point has a better objective.
    Optimal,
    /// No point satisfies the constraints and the bounds together.
    Infeasible,
    /// There are feasible points, and among them the objective improves without limit.
    Unbounded,
    /// The method used the pivots it was allowed before it reached any of the other three; the
    /// result holds the basic solution it had reached.
    IterationLimit,
}

/// The options of the simplex method: its tolerance, how many pivots it may take, when Bland's
/// rule takes over from Dantzig's, whether the problem is scaled, and from how many rows the dual
/// method solves it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::LpOptions;
///
/// let opts = LpOptions::<f64>::new().with_tol(1e-10).with_max_iter(500);
/// assert_eq!(opts.tol, 1e-10);
/// assert_eq!(opts.max_iter, 500);
/// assert_eq!(opts.stall_limit, 100);
/// assert!(opts.scaling);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LpOptions<T> {
    /// The tolerance, on the scaled problem. A bound holds when it is broken by no more than
    /// `tol · max(1, |bound|)` (for the slack of a row, `tol` times the size of the row's terms
    /// when that is larger), a row when its residual is no more than `tol` times the sum of the
    /// magnitudes of its terms (or `tol` itself, when that sum is below one), and a reduced cost
    /// `c_j − a_jᵀy` is taken for zero within `tol · (|c_j| + |a_jᵀy|)`, together with a bound on
    /// the error it carries. A basic variable is held to its bounds no more tightly than the
    /// error its own value carries, which the basis gives it; [`LinearProgram`] says more.
    pub tol: T,
    /// The largest number of pivots, a move of a variable from one of its bounds to the other
    /// counting as one. In the dual method an iteration — one change of basis, with whatever
    /// moves to the other bound its ratio test makes — counts as one, and it may take half of
    /// them; the primal method, which finishes from where it stops, the rest.
    pub max_iter: usize,
    /// How many pivots in a row may leave the objective where it was before Bland's rule takes
    /// over from Dantzig's, until the objective moves again. Zero is Bland's rule throughout;
    /// `usize::MAX` is Dantzig's rule throughout, which can cycle. It governs the primal method;
    /// the dual method perturbs its costs instead.
    pub stall_limit: usize,
    /// Whether the rows and the columns are scaled, by powers of two, until the largest element
    /// of each is near one, and the right-hand sides and the bounds together by the power of two
    /// that brings the median right-hand side near one.
    pub scaling: bool,
    /// The number of rows from which the dual simplex method solves the programme, the primal
    /// method checking its answer and finishing from it; a programme of fewer rows is solved by
    /// the primal method alone. Zero is the dual method for every programme; `usize::MAX` the
    /// primal method for every one.
    pub dual_threshold: usize,
}

impl<T: Real> Default for LpOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> LpOptions<T> {
    /// The defaults: `tol` `ε^(4/7)` (about `10⁻⁹` for `f64` and `10⁻⁴` for `f32`), `max_iter`
    /// 10 000, `stall_limit` 100, scaling on, and `dual_threshold` 20.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LpOptions;
    ///
    /// let opts = LpOptions::<f64>::new();
    /// assert!(opts.tol > 1e-9 && opts.tol < 1.2e-9);
    /// assert_eq!(opts.max_iter, 10_000);
    /// let single = LpOptions::<f32>::new();
    /// assert!(single.tol > 1e-4 && single.tol < 1.2e-4);
    /// assert_eq!(opts.dual_threshold, 20);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            tol: default_tolerance(),
            max_iter: 100 * DEFAULT_MAX_ITER,
            stall_limit: DEFAULT_MAX_ITER,
            scaling: true,
            dual_threshold: DUAL_THRESHOLD,
        }
    }

    /// Sets [`tol`](Self::tol).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LpOptions;
    ///
    /// assert_eq!(LpOptions::<f64>::new().with_tol(1e-7).tol, 1e-7);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }

    /// Sets [`max_iter`](Self::max_iter).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LpOptions;
    ///
    /// assert_eq!(LpOptions::<f64>::new().with_max_iter(50).max_iter, 50);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets [`stall_limit`](Self::stall_limit).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LpOptions;
    ///
    /// // Bland's rule from the first pivot.
    /// assert_eq!(LpOptions::<f64>::new().with_stall_limit(0).stall_limit, 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_stall_limit(mut self, stall_limit: usize) -> Self {
        self.stall_limit = stall_limit;
        self
    }

    /// Sets [`scaling`](Self::scaling).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LpOptions;
    ///
    /// assert!(!LpOptions::<f64>::new().with_scaling(false).scaling);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_scaling(mut self, scaling: bool) -> Self {
        self.scaling = scaling;
        self
    }

    /// Sets [`dual_threshold`](Self::dual_threshold).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LpOptions;
    ///
    /// // The dual simplex method for every programme.
    /// assert_eq!(LpOptions::<f64>::new().with_dual_threshold(0).dual_threshold, 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_dual_threshold(mut self, dual_threshold: usize) -> Self {
        self.dual_threshold = dual_threshold;
        self
    }
}

/// What the simplex method found: how it ended, the point, and the dual values.
///
/// The point, the slacks and the objective always describe the basic solution the method stopped
/// at, and the dual values are the multipliers of that basis; [`status`](Self::status) says what
/// they are worth. When it is [`LpStatus::Optimal`] they are a solution and the proof of its
/// optimality. When it is [`LpStatus::Unbounded`] the point is feasible but nothing is optimal;
/// when it is [`LpStatus::Infeasible`] the point is where phase one stopped, no pivot reducing
/// the amount by which it breaks the bounds and the rows, and feasible it is not; when it is
/// [`LpStatus::IterationLimit`] the point is as far as the method got. In those three the dual
/// values certify nothing.
///
/// The signs of the dual values are set out on [`LinearProgram`]: each is the rate of change of
/// the optimal objective with the right-hand side of its row, or with the bound of its variable.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{LinearProgram, LpStatus};
///
/// // Minimise 2x + 3y subject to x + y = 4 and x ≤ 3, with x and y at least zero.
/// let r = LinearProgram::minimize(&[2.0_f64, 3.0])
///     .subject_to_eq(&Matrix::from_rows(&[[1.0, 1.0]]), &[4.0])
///     .subject_to_le(&Matrix::from_rows(&[[1.0, 0.0]]), &[3.0])
///     .solve()
///     .unwrap();
/// assert_eq!(r.status, LpStatus::Optimal);
/// assert!((r.x[0] - 3.0).abs() < 1e-12 && (r.x[1] - 1.0).abs() < 1e-12);
/// assert!((r.objective - 9.0).abs() < 1e-12);
/// // A unit more on the right of the equation costs 3; a unit more of x allowed saves 1.
/// assert!((r.dual_eq[0] - 3.0).abs() < 1e-12 && (r.dual_le[0] + 1.0).abs() < 1e-12);
/// // Both variables are strictly inside their bounds, so their reduced costs are zero, and the
/// // objective is the dual objective.
/// assert_eq!(r.reduced_costs, [0.0, 0.0]);
/// assert!((r.objective - (4.0 * r.dual_eq[0] + 3.0 * r.dual_le[0])).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LpResult<T> {
    /// How the method ended.
    pub status: LpStatus,
    /// The point, one value for each variable: inside its bounds exactly when the status is
    /// [`LpStatus::Optimal`] or [`LpStatus::Unbounded`].
    pub x: Vec<T>,
    /// The objective `cᵀx` at the point: the minimum of a minimisation and the maximum of a
    /// maximisation when the status is [`LpStatus::Optimal`].
    pub objective: T,
    /// `b_le − A_le·x`, one value for each inequality, in the order they were given: what is left
    /// of each. It is negative where the point breaks an inequality.
    pub slack: Vec<T>,
    /// The multipliers of the inequalities: not positive for a minimisation, not negative for a
    /// maximisation, and zero where the slack is basic.
    pub dual_le: Vec<T>,
    /// The multipliers of the equations, of either sign.
    pub dual_eq: Vec<T>,
    /// The reduced costs `c − A_leᵀ·y_le − A_eqᵀ·y_eq`, one for each variable, zero for a basic
    /// variable: the multipliers of the bounds.
    pub reduced_costs: Vec<T>,
    /// The number of pivots: the changes of basis and the moves of a variable from one of its
    /// bounds to the other, an iteration of the dual method counting as one, of both methods
    /// together.
    pub pivots: usize,
}

/// A linear programme: a linear objective to minimise or maximise, subject to linear inequalities
/// `A_le·x ≤ b_le`, linear equations `A_eq·x = b_eq` and bounds `l ≤ x ≤ u`.
///
/// It is built from the costs by [`minimize`](Self::minimize) or [`maximize`](Self::maximize),
/// given its constraints by [`subject_to_le`](Self::subject_to_le) and
/// [`subject_to_eq`](Self::subject_to_eq), which may each be called more than once, and its
/// bounds by [`bounds`](Self::bounds); without bounds every variable is `x ≥ 0`. An inequality
/// `a·x ≥ b` is given as `−a·x ≤ −b`. The programme keeps a copy of what it is given.
///
/// # The method
///
/// The primal simplex method in two phases, in the revised form, on bounded variables; and for a
/// programme of at least [`LpOptions::dual_threshold`] rows, 20 by default, the dual simplex
/// method first, whose answer the primal method checks, and finishes from where it is not
/// proven (see the dual method, below). The primal method:
///
/// * **The working problem.** Every row is given a logical variable, `A·x + z = b`, with
///   `z ≥ 0` for an inequality and `z = 0` for an equation, so that the logical variables are a
///   basis to start from whatever the rank of the rows: an equation that repeats others never
///   makes the basis singular, its logical variable simply stays in it. The rows and the columns
///   are scaled by powers of two until the largest element of each is near one, and the
///   right-hand sides and the bounds together by the power of two that brings the median
///   right-hand side near one; a scale factor may be any power of two whose reciprocal is a
///   normal number too. That changes no digit of the data, makes one tolerance mean the same in
///   every row, and solves a programme whose numbers are all small, or all large, as the same
///   programme in moderate numbers would be solved.
/// * **Bounded variables.** A variable out of the basis sits at one of its bounds, or at zero
///   when zero is strictly inside them, and the ratio test lets the entering variable run to its
///   bound without a change of basis; no bound is turned into a row. The method starts from the
///   basis of the logical variables with every other variable as near zero as its bounds allow,
///   so a large bound standing for an infinite one never puts its size into the arithmetic.
/// * **Phase one** makes the point feasible. The basic variables that break a bound are given the
///   cost of the amount they break it by, and the simplex method minimises the sum; a variable
///   that reaches the bound it broke is blocked there and counts as feasible from then on. This is
///   the method of artificial variables with each artificial variable merged into the logical
///   variable of its row. When the sum cannot be reduced and is not zero, the programme has no
///   feasible point. **Phase two** minimises the objective from the feasible basis.
/// * **The basis** is factorised by [`Matrix::lu`]; a change of basis is absorbed as one factor of
///   the product form of the inverse (Dantzig and Orchard-Hays, 1954), and the basis is factorised
///   afresh, and the basic variables computed again from the data, every twenty changes and before
///   any answer is given.
/// * **Pricing** is Dantzig's rule: the variable whose reduced cost is the most favourable enters.
///   The simplex method can cycle on a degenerate vertex under that rule, so once
///   [`LpOptions::stall_limit`] pivots in a row have left the objective where it was, Bland's rule
///   (the eligible variable of the smallest index enters, and of the tied rows the variable of the
///   smallest index leaves) takes over until the objective moves again; under Bland's rule the
///   method cannot cycle (Bland, 1977).
/// * **The multipliers** `y` of the basis, the solution of `Bᵀ·y = c_B`, are improved by a step of
///   iterative refinement, and the residual left, `c_B − Bᵀ·y`, bounds their error: the reduced
///   cost of the multipliers of the basis itself differs from the one `y` gives by
///   `(B⁻¹·a_j)ᵀ·(c_B − Bᵀ·y)`, so the column of a variable as the basis sees it bounds the error
///   of its reduced cost. Pricing, which has no such column for every variable, proposes on an
///   estimate of that error, from what the refinement changed the reduced cost by; before the
///   variable it proposes enters, its column gives the bound, and a reduced cost within it is
///   rounding, not a direction of descent: the column is passed over until the basis changes.
///   The reduced cost is also computed again from the column, `c_q − c_Bᵀ·B⁻¹·a_q`, and has to
///   agree in sign.
/// * **The ratio test** is Harris's, in two passes (Harris, 1973): the first finds how far the
///   step may go when every bound is relaxed by a tenth of its tolerance, the second takes, of
///   the rows that block within that step, the one with the largest pivot. An element of the
///   entering column is not pivoted on when it is below the smaller of the tolerance and
///   `ε^(4/7)`, times the larger of one and the column's largest element.
///
/// The dual method works on the same scaled problem and keeps the reduced costs of the signs
/// their bounds ask for while it brings the basic variables inside theirs, one leaving the basis
/// at a time (the parts are those of Koberstein, *The dual simplex method, techniques for a fast
/// and stable implementation*, 2005):
///
/// * **The basis** is factorised by a sparse LU decomposition: pivots by Markowitz's rule with
///   threshold pivoting (Markowitz, 1957; Suhl and Suhl, 1990), the columns and rows of one
///   element, such as those of the logical variables, first, and what is left factorised dense
///   by [`Matrix::lu`] once it fills more than three tenths of its square. Changes of basis are
///   factors of the product form, and the basis is factorised afresh every 64 of them. The
///   matrix is kept by columns and by rows, and a row of `B⁻¹·A` is computed by whichever the
///   density of the row of `B⁻¹` suits.
/// * **Pricing** is by dual steepest edge (Forrest and Goldfarb, 1992): the basic variable that
///   breaks a bound by the most against the norm of its row of `B⁻¹`.
/// * **The ratio test** is the bound-flipping one (Fourer, 1994), with Harris's tolerance on the
///   reduced costs: a variable with two finite bounds whose reduced cost would change sign moves
///   to its other bound instead, as long as the leaving variable is still beyond its bound by
///   more than its tolerance, and only then does a variable enter.
/// * **Phase one** makes the reduced costs feasible where the bounds cannot, by the same method
///   on Fourer's subproblem: right-hand sides zero and the bounds `[0, 1]`, `[−1, 0]` and
///   `[−1000, 1000]` for a variable with only a lower, only an upper and no bound. Where its
///   optimum leaves a reduced cost infeasible, the programme is infeasible or unbounded and the
///   primal method decides which.
/// * **Degeneracy**: after twenty steps in a row that leave the dual objective where it was, the
///   costs are perturbed by small random amounts towards feasibility; they are restored at the
///   end. A bound of `10²⁰` or more as it was given, or beyond `1/tol` after the scaling, is
///   taken for infinite: a variable placed at it would put its size into every row it meets.
///
/// The basis the dual method ends at is handed to the primal method, factorised afresh by
/// [`Matrix::lu`] with its basic variables computed from the data, and judged by the certificates
/// below; where they fail, the primal method pivots on from that basis, and where it cannot, it
/// starts again from its own start. Where the dual method found a row of `B⁻¹` along which the
/// dual objective grows without end, that row alone is the candidate certificate of
/// infeasibility. So every answer is checked as the primal method's are, whichever method found
/// it.
///
/// # What an answer means
///
/// Every status but [`LpStatus::IterationLimit`] is checked against the data before it is
/// returned, on a basis factorised afresh:
///
/// * [`LpStatus::Optimal`]: the point, as it is reported (each variable inside its bounds),
///   satisfies every row within the tolerance; and no reduced cost of a variable out of the basis
///   has the sign that would improve the objective by more than the tolerance and the bound on
///   its error, which its own column gives (a variable that pricing passed over and that this
///   finds is entered instead, and the method goes on), nor is a basic one further from zero
///   than the residual of the multipliers allows. That makes the multipliers a feasible point of
///   the dual programme, to the tolerance, so by the duality of linear programming no feasible
///   point has an objective below the dual objective `bᵀy + Σ_j d_j·β_j`, where `β_j` is the
///   bound the sign of `d_j` prices. The objective at the point is also checked to equal that
///   dual objective within what the rows' residuals, held to the tolerance, are worth at `y`: a
///   check that the two were computed consistently, which the first two parts already imply up
///   to rounding.
/// * [`LpStatus::Infeasible`]: the multipliers of phase one — of the costs `±1` of the basic
///   variables that break a bound, or of that one alone in whose row the dual method found its
///   ray — are a certificate of Farkas's lemma: a combination of the rows whose value, over the
///   box of the bounds, cannot reach the value the right-hand sides ask for. No reduced cost of phase one may be favourable by more than
///   its tolerance and its error bound, as for an optimum, and the margin by which the value
///   falls short has to be larger than half the tolerance and than what the error of the
///   multipliers, their residual times the basic part of the point, could make of it: large
///   multipliers carry a large error, and prove nothing by a margin that error could make.
/// * [`LpStatus::Unbounded`]: the point is feasible, and from it runs a ray that keeps every row
///   and every bound and along which the objective improves for ever.
///
/// A programme whose answer cannot be so verified — a basis with a column that is, to rounding,
/// a combination of the others, a certificate that fails, a point that the arithmetic cannot keep
/// feasible (see the limits below) — gives `None` rather than a status that might be wrong.
///
/// The tolerance is [`LpOptions::tol`], `ε^(4/7)` by default, which is about `10⁻⁹` for `f64` and
/// `10⁻⁴` for `f32`, applied to the scaled problem and relative to the size of what it measures:
///
/// * a bound `l` or `u` holds when it is broken by no more than `tol · max(1, |bound|)`; the bound
///   `z ≥ 0` of the slack of a row, by no more than `tol` times the size of the row's terms,
///   `|b_i| + Σ_j |a_ij·x_j|`, when that is larger;
/// * a basic variable is held to a bound no more tightly than the error its value carries, which
///   is worked out each time the basis is factorised: the basic variables are the solution of
///   `B·x_B = b − N·x_N`, so the residual `r` of the rows at the point, and the rounding `γ·t` it
///   is computed with (`t` the sizes of the rows' terms, `γ = (n + 2)·ε`), bound their errors by
///   `|B⁻¹|·(|r| + γ·t)`, taken twice over. That follows the coupling of the basis: a large
///   right-hand side, slack or value loosens only the variables the basis computes from it, not
///   the rows and bounds of the others. The ratio test keeps to the tolerance alone, and a point
///   called feasible is reported only after every variable has been put inside its bounds and
///   every row checked at the point so reported;
/// * a row holds when its residual is no more than `tol` times the size of its terms (or `tol`,
///   when that is below one);
/// * a reduced cost `d_j = c_j − a_jᵀy` is taken for zero when it is no more than
///   `tol · (|c_j| + |a_jᵀy|)`, relative to the two numbers it is the difference of, together with
///   the bound on its error: what the residual of the multipliers makes of it through its
///   column, and the rounding of its own sum, `(m + 2)·ε` times the sum of the magnitudes of its
///   terms. Multipliers that are large in rows its variable does not meet do not enter either.
///
/// # Single precision
///
/// A programme in `f32` is solved in `f64`: its numbers, converted exactly, by the method above
/// in `f64` arithmetic under its own options (the tolerance as given, about `10⁻⁴` by default),
/// every certificate checked in `f64`. The point and the dual values are rounded to `f32` at the
/// end, and the objective and the slacks computed again, in `f32`, from the point so rounded. As
/// the method in `f32` arithmetic would, it gives `None` where a number of the scaled working
/// problem, or the size of a row's terms at the point (on the working problem), is beyond the
/// range of `f32`; a value of the answer beyond it is infinite. In `f32` arithmetic the bounds on the errors of
/// the reduced costs, through the columns of `B⁻¹·A`, grow with the rows until, at a few hundred,
/// every reduced cost can pass for rounding, and a basis far from the optimum for an optimal one.
///
/// # The dual values
///
/// [`LpResult::dual_le`] and [`LpResult::dual_eq`] hold a multiplier `y` for every row and
/// [`LpResult::reduced_costs`] holds `r = c − A_leᵀ·y_le − A_eqᵀ·y_eq`. Each is the rate at which
/// the optimal objective changes with the right-hand side of its row, or with the bound its
/// variable sits at — the convention of `scipy.optimize.linprog`. So, for a minimisation,
/// `y_le ≤ 0` (to relax an inequality can only lower the minimum), `r ≥ 0` for a variable at its
/// lower bound and `r ≤ 0` for one at its upper bound; for a maximisation every one of these signs
/// is reversed. A multiplier is zero where its row is slack and a reduced cost is zero, to its
/// tolerance, where its variable is strictly between its bounds (complementary slackness), and the
/// objective equals `b_leᵀ·y_le + b_eqᵀ·y_eq + rᵀ·x` (strong duality).
///
/// # Limits
///
/// The matrices are given dense. The primal method's factorisation is dense too: a pivot costs
/// `O(m² + m·n)` operations for `m` rows and `n` variables, and the basis is factorised afresh,
/// at `O(m³)`, every twenty pivots and before every answer (and the magnitudes of its inverse,
/// for the errors of the basic variables, cost as much again), which suits a few dozen rows.
/// Dantzig's rule can take a number of pivots exponential in the size of the problem (the cubes
/// of Klee and Minty), and Bland's is slower still on most problems. The dual method takes
/// about one to two pivots a row on the random programmes it was measured on, each costing
/// solves with the sparse factors and the elements of `A`, `O(m² + nnz(A))` at worst; its
/// answer is checked once on a dense factorisation, at `O(m³)`. There is no presolve. The tolerances on the bounds
/// and the rows have a floor of one after the scaling, so where the right-hand sides span many
/// orders of magnitude, the ones far below the median are held only to `tol` times the median;
/// and a programme that is infeasible, or unbounded, by less than the tolerance may be reported
/// as feasible, or bounded. In particular a reduced cost within its tolerance is taken for zero
/// even when the bound it points to is infinite: a programme unbounded only along rays whose
/// slope is within the tolerance of the costs' terms is reported optimal (rays of a relative
/// slope of `2.8·10⁻¹⁰` have been seen to be, at the default tolerance), and where the
/// multipliers are large the optimum reported can lie above the exact one by about the
/// tolerance times the terms of those reduced costs (`2.3·10⁻⁸` relative beside multipliers of
/// `10⁴`, about `2·10⁻¹¹` of the objective's terms). Turning the scaling off leaves the data as
/// they are and the tolerances in their units, which for rows in very small units are no longer
/// small: such a programme may then be judged by tolerances it was not written for, and get a
/// status or an optimum that is right only for them, as any solver with tolerances in those units
/// would. Where the data are so large that the rounding of a step is more than the tolerance of
/// the bounds it moves variables towards (the cubes of Klee and Minty in tens and hundreds, in
/// ten dimensions or more, without the scaling, whose steps reach `10¹⁸` and beyond), the point
/// computed afresh at a factorisation can break a bound that phase two was keeping; phase one
/// then resumes, and after the tenth time the answer is `None`.
///
/// On an ill-conditioned programme the point can move much further than the tolerance. Two
/// equations that differ by `δ` in one coefficient fix a variable only to the rows' tolerance
/// divided by `δ`, so once `δ` is no larger than that tolerance the answer is the optimum of a
/// programme within the tolerance of the one given, which can lie well below the exact optimum;
/// and reduced costs made of terms as large as the multipliers, about `1/δ`, are known only to the
/// rounding of those terms. The dual values still prove a bound below every feasible objective,
/// so the optimum reported is not above the exact one by more than that rounding.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{LinearProgram, LpStatus};
///
/// // Minimise −x + 4y subject to −3x + y ≤ 6 and x + 2y ≤ 4, with x free and y ≥ −3.
/// let a = Matrix::from_rows(&[[-3.0, 1.0], [1.0, 2.0]]);
/// let r = LinearProgram::minimize(&[-1.0, 4.0])
///     .subject_to_le(&a, &[6.0, 4.0])
///     .bounds(&[f64::NEG_INFINITY, -3.0], &[f64::INFINITY, f64::INFINITY])
///     .solve()
///     .unwrap();
/// assert_eq!(r.status, LpStatus::Optimal);
/// assert!((r.x[0] - 10.0).abs() < 1e-12 && (r.x[1] + 3.0).abs() < 1e-12);
/// assert!((r.objective + 22.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LinearProgram<T> {
    /// The costs, as they were given.
    cost: Vec<T>,
    /// Whether the objective is to be made as large as it can be.
    maximize: bool,
    /// The rows of the inequalities, one after another.
    le_rows: Vec<T>,
    /// Their right-hand sides.
    le_rhs: Vec<T>,
    /// The rows of the equations, one after another.
    eq_rows: Vec<T>,
    /// Their right-hand sides.
    eq_rhs: Vec<T>,
    /// The lower bounds.
    lower: Vec<T>,
    /// The upper bounds.
    upper: Vec<T>,
}

impl<T: Real + Scalar> LinearProgram<T> {
    /// The programme that minimises `cᵀx` over `x ≥ 0`, one variable for each cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{LinearProgram, LpStatus};
    ///
    /// // With nothing else to hold it, the minimum of x + 2y over x, y ≥ 0 is at the origin.
    /// let r = LinearProgram::minimize(&[1.0, 2.0]).solve().unwrap();
    /// assert_eq!(r.status, LpStatus::Optimal);
    /// assert_eq!(r.x, [0.0, 0.0]);
    /// assert_eq!(r.objective, 0.0);
    /// ```
    #[must_use]
    pub fn minimize(c: &[T]) -> Self {
        Self {
            cost: c.to_vec(),
            maximize: false,
            le_rows: Vec::new(),
            le_rhs: Vec::new(),
            eq_rows: Vec::new(),
            eq_rhs: Vec::new(),
            lower: vec![T::ZERO; c.len()],
            upper: vec![T::INFINITY; c.len()],
        }
    }

    /// The programme that maximises `cᵀx` over `x ≥ 0`, one variable for each cost.
    ///
    /// The result reports the maximum itself, and dual values that are the rates of change of the
    /// maximum: their signs are the reverse of those of a minimisation.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{LinearProgram, LpStatus};
    ///
    /// // The largest x + y with x + 2y ≤ 4 and 3x + y ≤ 6 is 14/5, at (8/5, 6/5).
    /// let a = Matrix::from_rows(&[[1.0, 2.0], [3.0, 1.0]]);
    /// let r = LinearProgram::maximize(&[1.0_f64, 1.0])
    ///     .subject_to_le(&a, &[4.0, 6.0])
    ///     .solve()
    ///     .unwrap();
    /// assert_eq!(r.status, LpStatus::Optimal);
    /// assert!((r.objective - 2.8).abs() < 1e-12);
    /// // To relax either inequality raises the maximum.
    /// assert!(r.dual_le[0] > 0.0 && r.dual_le[1] > 0.0);
    /// ```
    #[must_use]
    pub fn maximize(c: &[T]) -> Self {
        Self {
            maximize: true,
            ..Self::minimize(c)
        }
    }

    /// Adds the inequalities `a·x ≤ b`, one for each row of `a`, after those already there. A
    /// matrix of no rows adds nothing.
    ///
    /// # Panics
    ///
    /// Panics if `a` has rows and not one column for each variable, or if `b` has not one element
    /// for each row of `a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::LinearProgram;
    ///
    /// // Two calls add up: x ≤ 3, then y ≤ 2.
    /// let r = LinearProgram::maximize(&[1.0, 1.0])
    ///     .subject_to_le(&Matrix::from_rows(&[[1.0, 0.0]]), &[3.0])
    ///     .subject_to_le(&Matrix::from_rows(&[[0.0, 1.0]]), &[2.0])
    ///     .solve()
    ///     .unwrap();
    /// assert_eq!(r.x, [3.0, 2.0]);
    /// assert_eq!(r.slack, [0.0, 0.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn subject_to_le(mut self, a: &Matrix<T>, b: &[T]) -> Self {
        append_rows(&mut self.le_rows, &mut self.le_rhs, a, b, self.cost.len());
        self
    }

    /// Adds the equations `a·x = b`, one for each row of `a`, after those already there. A matrix
    /// of no rows adds nothing. The rows need not be independent: an equation that follows from
    /// the others is carried along, and one that contradicts them makes the programme infeasible.
    ///
    /// # Panics
    ///
    /// Panics if `a` has rows and not one column for each variable, or if `b` has not one element
    /// for each row of `a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{LinearProgram, LpStatus};
    ///
    /// // x + y = 2 twice over, the second time doubled: the minimum of x − y is at (0, 2).
    /// let a = Matrix::from_rows(&[[1.0, 1.0], [2.0, 2.0]]);
    /// let r = LinearProgram::minimize(&[1.0, -1.0])
    ///     .subject_to_eq(&a, &[2.0, 4.0])
    ///     .solve()
    ///     .unwrap();
    /// assert_eq!(r.status, LpStatus::Optimal);
    /// assert_eq!(r.x, [0.0, 2.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn subject_to_eq(mut self, a: &Matrix<T>, b: &[T]) -> Self {
        append_rows(&mut self.eq_rows, &mut self.eq_rhs, a, b, self.cost.len());
        self
    }

    /// Sets the bounds `lower[j] ≤ x[j] ≤ upper[j]` of every variable, in place of `x ≥ 0`. A
    /// lower bound may be `−∞` and an upper bound `+∞`; a variable with both is free, and one
    /// whose bounds are equal is fixed.
    ///
    /// # Panics
    ///
    /// Panics if `lower` or `upper` has not one element for each variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LinearProgram;
    ///
    /// // Each variable goes to the bound its cost prefers.
    /// let r = LinearProgram::minimize(&[1.0, -1.0, 0.0])
    ///     .bounds(&[-2.0, 0.0, 5.0], &[3.0, 7.0, 5.0])
    ///     .solve()
    ///     .unwrap();
    /// assert_eq!(r.x, [-2.0, 7.0, 5.0]);
    /// assert_eq!(r.objective, -9.0);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn bounds(mut self, lower: &[T], upper: &[T]) -> Self {
        let n = self.cost.len();
        assert_eq!(
            lower.len(),
            n,
            "there is not one lower bound for each variable"
        );
        assert_eq!(
            upper.len(),
            n,
            "there is not one upper bound for each variable"
        );
        self.lower = lower.to_vec();
        self.upper = upper.to_vec();
        self
    }

    /// Solves the programme by the simplex method, with the default options: see
    /// [`LinearProgram`] for the method and for what each status has been checked to mean.
    ///
    /// Returns `None` when a cost, a coefficient or a right-hand side is not finite, when a bound
    /// is NaN, a lower bound `+∞` or an upper bound `−∞`, and when the answer the method reached
    /// could not be verified.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{LinearProgram, LpStatus};
    ///
    /// // The cheapest way to buy 8 units of one nutrient and 10 of another from two foods.
    /// let a = Matrix::from_rows(&[[-2.0, -1.0], [-1.0, -3.0]]);
    /// let r = LinearProgram::minimize(&[3.0_f64, 4.0])
    ///     .subject_to_le(&a, &[-8.0, -10.0])
    ///     .solve()
    ///     .unwrap();
    /// assert_eq!(r.status, LpStatus::Optimal);
    /// assert!((r.x[0] - 2.8).abs() < 1e-12 && (r.x[1] - 2.4).abs() < 1e-12);
    /// assert!((r.objective - 18.0).abs() < 1e-12);
    ///
    /// // A NaN in the data is refused.
    /// assert!(LinearProgram::minimize(&[f64::NAN]).solve().is_none());
    /// ```
    #[must_use]
    pub fn solve(&self) -> Option<LpResult<T>> {
        self.solve_with(&LpOptions::new())
    }

    /// [`solve`](Self::solve) under `opts`. Returns `None` also when the tolerance is not a
    /// positive finite number.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{LinearProgram, LpOptions, LpStatus};
    ///
    /// let a = Matrix::from_rows(&[[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]]);
    /// let lp = LinearProgram::maximize(&[3.0_f64, 5.0]).subject_to_le(&a, &[4.0, 12.0, 18.0]);
    ///
    /// // Bland's rule throughout reaches the same vertex.
    /// let r = lp.solve_with(&LpOptions::new().with_stall_limit(0)).unwrap();
    /// assert_eq!(r.status, LpStatus::Optimal);
    /// assert!((r.objective - 36.0).abs() < 1e-12);
    ///
    /// // One pivot is not enough, and the result says so.
    /// let r = lp.solve_with(&LpOptions::new().with_max_iter(1)).unwrap();
    /// assert_eq!(r.status, LpStatus::IterationLimit);
    /// assert_eq!(r.pivots, 1);
    ///
    /// assert!(lp.solve_with(&LpOptions::new().with_tol(0.0)).is_none());
    /// ```
    #[must_use]
    pub fn solve_with(&self, opts: &LpOptions<T>) -> Option<LpResult<T>> {
        if T::MANTISSA_DIGITS < f64::MANTISSA_DIGITS {
            self.solve_in_f64(opts)
        } else {
            self.solve_in_place(opts)
        }
    }

    /// [`solve_with`](Self::solve_with) for a programme in a precision narrower than `f64`: the
    /// programme in `f64`, each number converted exactly, solved under the same options, the
    /// point and the dual values rounded at the end, and the objective and the slacks computed
    /// again from the point so rounded, as for an answer computed in that precision. In `f32`
    /// arithmetic the bounds on the errors of the reduced costs, through the columns of `B⁻¹·A`,
    /// grow with the rows until every reduced cost can pass for rounding, and a basis far from
    /// the optimum for an optimal one (on random programmes of 220 rows, optima of −184 and
    /// −420 where the optimum is −527 and −533); in `f64` they stay small. As the method in the
    /// precision of the programme would, it gives `None` where a number of the scaled working
    /// problem, or the size of a row's terms at the point as the method measures it on the
    /// working problem, is beyond the range of that precision; a value of the answer beyond it is
    /// infinite, as the arithmetic of that precision makes it.
    fn solve_in_f64(&self, opts: &LpOptions<T>) -> Option<LpResult<T>> {
        // The working problem in the precision of the programme, as the method in that precision
        // would have it: where a scaled number of it is beyond the range, there is no answer.
        let working = self.working(opts)?;
        let wide = |v: &[T]| -> Vec<f64> { v.iter().map(|x| x.to_f64()).collect() };
        let programme = LinearProgram::<f64> {
            cost: wide(&self.cost),
            maximize: self.maximize,
            le_rows: wide(&self.le_rows),
            le_rhs: wide(&self.le_rhs),
            eq_rows: wide(&self.eq_rows),
            eq_rhs: wide(&self.eq_rhs),
            lower: wide(&self.lower),
            upper: wide(&self.upper),
        };
        let options = LpOptions::<f64> {
            tol: opts.tol.to_f64(),
            max_iter: opts.max_iter,
            stall_limit: opts.stall_limit,
            scaling: opts.scaling,
            dual_threshold: opts.dual_threshold,
        };
        let answer = programme.solve_in_place(&options)?;
        let narrow = |v: &[f64]| -> Vec<T> { v.iter().map(|x| T::from_f64(*x)).collect() };
        // The point rounded to the precision stays inside the bounds, which it can state; beyond
        // its range, as the method's report makes it there, it is infinite.
        let x = narrow(&answer.x);
        let n = self.cost.len();
        let objective = dot(&self.cost, &x);
        let slack: Vec<T> = (0..self.le_rhs.len())
            .map(|i| self.le_rhs[i] - dot(&self.le_rows[i * n..(i + 1) * n], &x))
            .collect();
        // Nor is there one where the sizes of the rows' terms at the point, as the method
        // measures them on the working problem, are beyond the range in that precision.
        let Working {
            a,
            b,
            column_scale,
            primal,
            ..
        } = &working;
        let scaled: Vec<T> = answer
            .x
            .iter()
            .zip(column_scale)
            .map(|(v, cs)| T::from_f64(*v * primal.to_f64() / cs.to_f64()))
            .collect();
        let mut sizes: Vec<T> = b.iter().map(|v| mag(*v)).collect();
        for (j, &v) in scaled.iter().enumerate() {
            if v != T::ZERO {
                for (size, c) in sizes.iter_mut().zip(a.column(j)) {
                    *size += mag(*c * v);
                }
            }
        }
        if !sizes.iter().all(|v| finite(*v)) {
            return None;
        }
        // Any value beyond the range is infinite, as arithmetic in that precision makes it.
        Some(LpResult {
            status: answer.status,
            x,
            objective,
            slack,
            dual_le: narrow(&answer.dual_le),
            dual_eq: narrow(&answer.dual_eq),
            reduced_costs: narrow(&answer.reduced_costs),
            pivots: answer.pivots,
        })
    }

    /// The working problem of the programme under `opts`, as the method solves it: the rows of
    /// both kinds as one matrix, the inequalities first, scaled; the right-hand sides, the costs
    /// of the minimisation and the bounds of every variable, logical ones included, scaled with
    /// them; and the scale factors. `None` when a scaled number is not finite or a bound NaN.
    fn working(&self, opts: &LpOptions<T>) -> Option<Working<T>> {
        let n = self.cost.len();
        let (m_le, m_eq) = (self.le_rhs.len(), self.eq_rhs.len());
        let m = m_le + m_eq;
        // The rows of both kinds as one matrix, the inequalities first.
        let mut a = Matrix::from_fn(m, n, |i, j| {
            if i < m_le {
                self.le_rows[i * n + j]
            } else {
                self.eq_rows[(i - m_le) * n + j]
            }
        });
        let (row_scale, column_scale) = if opts.scaling {
            scales(&a)
        } else {
            (vec![T::ONE; m], vec![T::ONE; n])
        };
        for (j, cs) in column_scale.iter().enumerate() {
            for (v, rs) in a.column_mut(j).iter_mut().zip(&row_scale) {
                *v = *v * *rs * *cs;
            }
        }
        let mut b: Vec<T> = self
            .le_rhs
            .iter()
            .chain(&self.eq_rhs)
            .zip(&row_scale)
            .map(|(v, rs)| *v * *rs)
            .collect();

        // The working problem is a minimisation over the structural variables and one logical
        // variable for each row: not negative for an inequality, zero for an equation.
        let sign = if self.maximize { -T::ONE } else { T::ONE };
        let mut cost: Vec<T> = self
            .cost
            .iter()
            .zip(&column_scale)
            .map(|(c, cs)| sign * *c * *cs)
            .collect();
        cost.resize(n + m, T::ZERO);
        let scaled = |bounds: &[T]| -> Vec<T> {
            let mut out: Vec<T> = bounds
                .iter()
                .zip(&column_scale)
                .map(|(v, cs)| *v / *cs)
                .collect();
            out.resize(n + m, T::ZERO);
            out
        };
        let mut lower = scaled(&self.lower);
        let mut upper = scaled(&self.upper);
        let primal = if opts.scaling {
            primal_scale(&b, &lower[..n], &upper[..n])
        } else {
            T::ONE
        };
        for v in b.iter_mut().chain(&mut lower[..n]).chain(&mut upper[..n]) {
            *v *= primal;
        }
        for u in &mut upper[n..n + m_le] {
            *u = T::INFINITY;
        }
        let all_finite = a.as_slice().iter().all(|v| finite(*v))
            && b.iter().chain(&cost).all(|v| finite(*v))
            && lower
                .iter()
                .zip(&upper)
                .all(|(l, u)| !l.is_nan() && !u.is_nan());
        all_finite.then_some(Working {
            a,
            b,
            cost,
            lower,
            upper,
            row_scale,
            column_scale,
            primal,
        })
    }

    /// [`solve_with`](Self::solve_with) in the arithmetic of `T`.
    fn solve_in_place(&self, opts: &LpOptions<T>) -> Option<LpResult<T>> {
        let n = self.cost.len();
        let (m_le, m_eq) = (self.le_rhs.len(), self.eq_rhs.len());
        let m = m_le + m_eq;
        if !finite(opts.tol) || opts.tol <= T::ZERO {
            return None;
        }
        let data = [
            &self.cost,
            &self.le_rows,
            &self.le_rhs,
            &self.eq_rows,
            &self.eq_rhs,
        ];
        if data.iter().any(|part| part.iter().any(|v| !finite(*v))) {
            return None;
        }
        let lower_ok = self.lower.iter().all(|l| *l < T::INFINITY);
        let upper_ok = self.upper.iter().all(|u| *u > -T::INFINITY);
        if !lower_ok || !upper_ok {
            return None;
        }

        let Working {
            a,
            b,
            cost,
            lower,
            upper,
            row_scale,
            column_scale,
            primal,
        } = self.working(opts)?;

        // A variable whose bounds cross leaves nothing to search: there is no feasible point.
        let crossed = self.lower.iter().zip(&self.upper).any(|(l, u)| l > u);
        let mut engine = Engine::new(a, b, cost, lower, upper, opts)?;
        let status = if crossed {
            LpStatus::Infeasible
        } else if m >= opts.dual_threshold {
            // The bounds the dual method works to: those that stand for infinite ones, by their
            // size as they were given, made infinite.
            let huge = T::from_f64(INFINITE_BOUND);
            let mut dual_lower = engine.lower.clone();
            let mut dual_upper = engine.upper.clone();
            for j in 0..n {
                if self.lower[j] <= -huge {
                    dual_lower[j] = -T::INFINITY;
                }
                if self.upper[j] >= huge {
                    dual_upper[j] = T::INFINITY;
                }
            }
            engine.run_dual(&dual_lower, &dual_upper)?
        } else {
            engine.run()?
        };
        Some(self.report(&engine, status, &row_scale, &column_scale, primal))
    }

    /// The result in the units the programme was given in, from the state the method stopped in;
    /// `primal` is the factor the right-hand sides and the bounds were multiplied by, after the
    /// rows and the columns were scaled.
    fn report(
        &self,
        engine: &Engine<T>,
        status: LpStatus,
        row_scale: &[T],
        column_scale: &[T],
        primal: T,
    ) -> LpResult<T> {
        let n = self.cost.len();
        let m_le = self.le_rhs.len();
        let sign = if self.maximize { -T::ONE } else { T::ONE };
        // A point that is claimed feasible is feasible within the tolerance; the little by which
        // a basic variable may still be outside a bound, within the tolerance or the error its
        // value carries, is taken off, as the certificate took it off before it checked the rows.
        let clamp = matches!(status, LpStatus::Optimal | LpStatus::Unbounded);
        let x: Vec<T> = (0..n)
            .map(|j| match engine.place[j] {
                Place::Lower | Place::Fixed => self.lower[j],
                Place::Upper => self.upper[j],
                Place::Zero => T::ZERO,
                Place::Basic => {
                    let v = engine.x[j] * column_scale[j] / primal;
                    if clamp {
                        v.max(self.lower[j]).min(self.upper[j])
                    } else {
                        v
                    }
                }
            })
            .collect();
        let objective = dot(&self.cost, &x);
        let slack: Vec<T> = (0..m_le)
            .map(|i| self.le_rhs[i] - dot(&self.le_rows[i * n..(i + 1) * n], &x))
            .collect();

        // The multipliers of the basis for the objective itself, whatever phase the method
        // stopped in, refined as the certificate of an optimum had them. Where the logical
        // variable of a row is basic the multiplier is zero, and where a variable is basic so is
        // its reduced cost: both are set to that, not to the rounding of a solve.
        let duals = engine.duals(false);
        let dual: Vec<T> = (0..engine.rows)
            .map(|i| {
                if engine.place[n + i] == Place::Basic {
                    T::ZERO
                } else {
                    sign * row_scale[i] * duals.y[i]
                }
            })
            .collect();
        let reduced_costs: Vec<T> = (0..n)
            .map(|j| {
                if engine.place[j] == Place::Basic {
                    T::ZERO
                } else {
                    sign * engine.priced(j, &duals).d / column_scale[j]
                }
            })
            .collect();
        LpResult {
            status,
            x,
            objective,
            slack,
            dual_le: dual[..m_le].to_vec(),
            dual_eq: dual[m_le..].to_vec(),
            reduced_costs,
            pivots: engine.pivots,
        }
    }
}

/// The working problem of a programme, scaled, and its scale factors: see
/// [`LinearProgram::working`].
struct Working<T> {
    /// `A`, `m × n`: the rows of the inequalities, then those of the equations, scaled.
    a: Matrix<T>,
    /// The right-hand sides, scaled.
    b: Vec<T>,
    /// The costs of the minimisation, `n + m` of them, zero for the logical variables.
    cost: Vec<T>,
    /// The lower bounds of the `n + m` variables, scaled.
    lower: Vec<T>,
    /// The upper bounds of the `n + m` variables, scaled.
    upper: Vec<T>,
    /// The factors the rows were multiplied by.
    row_scale: Vec<T>,
    /// The factors the columns were multiplied by.
    column_scale: Vec<T>,
    /// The factor the right-hand sides and the bounds were multiplied by, after the rows and
    /// the columns were scaled.
    primal: T,
}

/// Appends the rows of `a` and their right-hand sides `b` to the rows and right-hand sides of one
/// kind of constraint of a programme of `n` variables.
#[track_caller]
fn append_rows<T: Real>(rows: &mut Vec<T>, rhs: &mut Vec<T>, a: &Matrix<T>, b: &[T], n: usize) {
    assert_eq!(
        a.rows(),
        b.len(),
        "there is not one right-hand side for each row of the constraints"
    );
    if a.rows() == 0 {
        return;
    }
    assert_eq!(
        a.cols(),
        n,
        "the constraints have not one column for each variable"
    );
    for i in 0..a.rows() {
        rows.extend(a.row(i).copied());
    }
    rhs.extend_from_slice(b);
}

/// The largest exponent `e` for which `2^e` and `2^−e` are both finite normal numbers of `T`: 1022
/// for `f64` and 126 for `f32`. A scale factor is a power of two within that range, so that it
/// changes no digit of what it multiplies, however small or large the data are.
fn exponent_limit<T: Real>() -> f64 {
    T::MAX.log2().to_f64().floor() - 1.0
}

/// The power of two nearest to `v`, which is positive, in the logarithm.
fn power_of_two<T: Real>(v: T) -> T {
    let limit = exponent_limit::<T>();
    let exponent = v.log2().to_f64().round().clamp(-limit, limit);
    T::TWO.powi(exponent as i32)
}

/// The power of two the right-hand sides and the bounds are multiplied by so that the typical
/// right-hand side is near one: the reciprocal of the median of the magnitudes of the right-hand
/// sides that are not zero, or of the finite bounds that are not zero when every right-hand side
/// is. It multiplies the point by the same factor and leaves the dual values as they are, so a
/// programme whose data are all small, or all large, is solved as the same programme in moderate
/// numbers would be. A median, unlike a mean, is not drawn away by a few large numbers, and the
/// bounds are left out because a large one often stands for an infinite one.
fn primal_scale<T: Real>(rhs: &[T], lower: &[T], upper: &[T]) -> T {
    // The binary logarithms of the magnitudes that are neither zero nor infinite.
    let exponents = |values: &[T], out: &mut Vec<f64>| {
        for v in values {
            let v = mag(*v);
            if v > T::ZERO && finite(v) {
                out.push(v.log2().to_f64());
            }
        }
    };
    let mut found = Vec::new();
    exponents(rhs, &mut found);
    if found.is_empty() {
        exponents(lower, &mut found);
        exponents(upper, &mut found);
    }
    if found.is_empty() {
        return T::ONE;
    }
    found.sort_by(f64::total_cmp);
    let median = found[(found.len() - 1) / 2];
    let limit = exponent_limit::<T>();
    let exponent = (-median).round().clamp(-limit, limit);
    T::TWO.powi(exponent as i32)
}

/// The factors, powers of two, the rows and the columns of `a` are multiplied by so that the
/// largest element of each is near one: the rows by their largest, then the columns by theirs,
/// twice over. A row or a column of zeros is left as it is.
fn scales<T: Real>(a: &Matrix<T>) -> (Vec<T>, Vec<T>) {
    let (m, n) = a.shape();
    let mut row = vec![T::ONE; m];
    let mut column = vec![T::ONE; n];
    for _ in 0..SCALE_SWEEPS {
        let mut largest = vec![T::ZERO; m];
        for (j, cs) in column.iter().enumerate() {
            for (big, v) in largest.iter_mut().zip(a.column(j)) {
                *big = big.max(mag(*v) * *cs);
            }
        }
        for (rs, big) in row.iter_mut().zip(&largest) {
            *rs = if *big > T::ZERO {
                T::ONE / power_of_two(*big)
            } else {
                T::ONE
            };
        }
        for (j, cs) in column.iter_mut().enumerate() {
            let mut big = T::ZERO;
            for (v, rs) in a.column(j).iter().zip(&row) {
                big = big.max(mag(*v) * *rs);
            }
            *cs = if big > T::ZERO {
                T::ONE / power_of_two(big)
            } else {
                T::ONE
            };
        }
    }
    (row, column)
}

/// Where a variable of the working problem is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// In the basis.
    Basic,
    /// Out of the basis, at its lower bound.
    Lower,
    /// Out of the basis, at its upper bound.
    Upper,
    /// Out of the basis at zero, strictly inside its bounds (either of which may be infinite):
    /// it may move either way.
    Zero,
    /// Out of the basis with its two bounds equal: it never enters.
    Fixed,
}

/// What the ratio test allows the entering variable.
enum Step<T> {
    /// Nothing stops it: the step is as long as one likes.
    Unbounded,
    /// It reaches its own bound, the one it moves towards, first, after a step of this length.
    Flip(T),
    /// The basic variable of this row reaches a bound first, after a step of this length.
    Pivot(usize, T),
}

/// The basis `B` in the product form: the LU decomposition `P·B₀ = L·U` of the basis as it was
/// when it was last factorised, and one elementary factor for each change since,
/// `B = B₀·E₁·E₂⋯`, where `Eₖ` is the identity with one column replaced by the entering column as
/// the basis before it saw it.
struct Basis<T> {
    /// `L`, with ones on its diagonal.
    lower: Matrix<T>,
    /// `U`.
    upper: Matrix<T>,
    /// Row `i` of `P·B₀` is row `permutation[i]` of `B₀`.
    permutation: Vec<usize>,
    /// The elementary factors: the row each replaces, and the column it puts there.
    updates: Vec<(usize, Vec<T>)>,
    /// `|B₀⁻¹|`, the magnitudes of the elements of the inverse of `B₀`, column by column, from
    /// which [`bound_solve`](Self::bound_solve) bounds the errors of the basic variables.
    magnitudes: Matrix<T>,
    /// The decomposition itself, for the blocked solves of [`invert_blocked`](Self::invert_blocked);
    /// `None` for the identity.
    lu: Option<Lu<T>>,
}

impl<T: Real> Basis<T> {
    /// The number of changes of basis since the basis was factorised.
    fn updates(&self) -> usize {
        self.updates.len()
    }

    /// Records that the basic variable of `row` has been replaced by the one whose column is
    /// `B·column`.
    fn replace(&mut self, row: usize, column: &[T]) {
        self.updates.push((row, column.to_vec()));
    }

    /// Writes the solution `v` of `B·v = rhs` to `out`.
    fn solve(&self, rhs: &[T], out: &mut [T]) {
        let m = out.len();
        for (v, &p) in out.iter_mut().zip(&self.permutation) {
            *v = rhs[p];
        }
        // `L`, a column at a time.
        for k in 0..m {
            let pivot = out[k];
            if pivot != T::ZERO {
                let column = &self.lower.column(k)[k + 1..];
                for (v, l) in out[k + 1..].iter_mut().zip(column) {
                    *v -= *l * pivot;
                }
            }
        }
        // `U`, a column at a time from the last.
        for k in (0..m).rev() {
            let column = self.upper.column(k);
            let pivot = out[k] / column[k];
            out[k] = pivot;
            if pivot != T::ZERO {
                for (v, u) in out[..k].iter_mut().zip(column) {
                    *v -= *u * pivot;
                }
            }
        }
        // The elementary factors, the oldest first: `E⁻¹` divides the element of its row by the
        // pivot and takes that multiple of the column off the others.
        for (row, column) in &self.updates {
            let pivot = out[*row] / column[*row];
            if pivot != T::ZERO {
                for (v, c) in out.iter_mut().zip(column) {
                    *v -= *c * pivot;
                }
            }
            out[*row] = pivot;
        }
    }

    /// Writes the solution `v` of `Bᵀ·v = rhs` to `out`; `rhs` is used as working memory.
    fn solve_transposed(&self, rhs: &mut [T], out: &mut [T]) {
        let m = rhs.len();
        // The elementary factors, the newest first: `E⁻ᵀ` changes only the element of its row.
        for (row, column) in self.updates.iter().rev() {
            let others = dot(rhs, column) - rhs[*row] * column[*row];
            rhs[*row] = (rhs[*row] - others) / column[*row];
        }
        // `Uᵀ`, which is lower triangular: a row of it is a column of `U`.
        for k in 0..m {
            let column = self.upper.column(k);
            rhs[k] = (rhs[k] - dot(&rhs[..k], &column[..k])) / column[k];
        }
        // `Lᵀ`, which is upper triangular with ones on its diagonal.
        for k in (0..m).rev() {
            let column = &self.lower.column(k)[k + 1..];
            rhs[k] -= dot(&rhs[k + 1..], column);
        }
        for (v, &p) in rhs.iter().zip(&self.permutation) {
            out[p] = *v;
        }
    }

    /// Writes `|B⁻¹|·w` to `out`, for `w` of no negative element, on a basis just factorised:
    /// what errors of the magnitudes `w` in a right-hand side can make of each element of the
    /// solution. `O(m²)`, like a solve.
    fn bound_solve(&self, w: &[T], out: &mut [T]) {
        debug_assert!(self.updates.is_empty(), "the magnitudes are those of B₀");
        out.fill(T::ZERO);
        for (i, v) in w.iter().enumerate() {
            if *v != T::ZERO {
                for (o, c) in out.iter_mut().zip(self.magnitudes.column(i)) {
                    *o += *c * *v;
                }
            }
        }
    }
}

impl<T: Real + Scalar> Basis<T> {
    /// The basis of the logical variables of `m` rows, which is the identity.
    fn identity(m: usize) -> Self {
        Self {
            lower: Matrix::identity(m),
            upper: Matrix::identity(m),
            permutation: (0..m).collect(),
            updates: Vec::new(),
            magnitudes: Matrix::identity(m),
            lu: None,
        }
    }

    /// Computes [`magnitudes`](Self::magnitudes), `|B₀⁻¹|`, on a basis just factorised. Where
    /// `unit[i]` is `Some(k)`, column `k` of the basis is the unit vector of row `i`, the column
    /// of a logical variable, and column `i` of the inverse is the unit vector of position `k`;
    /// every other column of the inverse is a solve with the factors. That is `O(m²)` for each
    /// row whose logical variable is not basic: at most one factorisation's work again.
    fn invert(&mut self, unit: &[Option<usize>]) {
        let m = unit.len();
        let mut magnitudes = Matrix::zeros(m, m);
        let mut e = vec![T::ZERO; m];
        let mut solution = vec![T::ZERO; m];
        for (i, position) in unit.iter().enumerate() {
            let column = magnitudes.column_mut(i);
            match position {
                Some(k) => column[*k] = T::ONE,
                None => {
                    e[i] = T::ONE;
                    self.solve(&e, &mut solution);
                    e[i] = T::ZERO;
                    for (c, v) in column.iter_mut().zip(&solution) {
                        *c = mag(*v);
                    }
                }
            }
        }
        self.magnitudes = magnitudes;
    }

    /// [`invert`](Self::invert) by one blocked solve of all the columns of the inverse it needs,
    /// `B₀·X = [e_i …]` for the rows `i` whose logical variable is not basic, with the same
    /// factors: the same magnitudes but for the order of the rounding, at the speed of products
    /// of blocks. The dual simplex method's bases, of hundreds of rows, are checked so.
    fn invert_blocked(&mut self, unit: &[Option<usize>]) {
        let m = unit.len();
        let rows: Vec<usize> = (0..m).filter(|&i| unit[i].is_none()).collect();
        let solved = match &self.lu {
            Some(lu) if !rows.is_empty() => {
                let rhs =
                    Matrix::from_fn(
                        m,
                        rows.len(),
                        |r, t| {
                            if r == rows[t] { T::ONE } else { T::ZERO }
                        },
                    );
                lu.solve(&rhs)
            }
            _ => None,
        };
        let Some(solved) = solved else {
            self.invert(unit);
            return;
        };
        let mut magnitudes = Matrix::zeros(m, m);
        for (i, position) in unit.iter().enumerate() {
            if let Some(k) = position {
                magnitudes.column_mut(i)[*k] = T::ONE;
            }
        }
        for (t, &i) in rows.iter().enumerate() {
            for (c, v) in magnitudes.column_mut(i).iter_mut().zip(solved.column(t)) {
                *c = mag(*v);
            }
        }
        self.magnitudes = magnitudes;
    }

    /// The basis with the given columns, by [`Matrix::lu`], or `None` when it is singular to
    /// working precision: a pivot that is not finite, or one that is to the largest element of
    /// its own column of the basis as rounding is to one. Elimination with partial pivoting
    /// leaves the columns in their order, so a pivot that small says that its column is, to
    /// rounding, a combination of the columns before it; measured against its own column rather
    /// than against the other pivots, the test does not depend on the scales of the columns.
    fn factor(columns: &Matrix<T>) -> Option<Self> {
        let m = columns.rows();
        let lu = columns.lu();
        let upper = lu.u();
        let rounding = T::EPSILON * T::from_f64(m as f64);
        for k in 0..m {
            let pivot = mag(upper.column(k)[k]);
            let largest = columns
                .column(k)
                .iter()
                .fold(T::ZERO, |big, v| big.max(mag(*v)));
            if !finite(pivot) || pivot <= largest * rounding {
                return None;
            }
        }
        Some(Self {
            lower: lu.l(),
            upper,
            permutation: lu.permutation().to_vec(),
            updates: Vec::new(),
            magnitudes: Matrix::zeros(0, 0),
            lu: Some(lu),
        })
    }
}

/// The multipliers of a basis for the costs of a phase: the solution `y` of `Bᵀ·y = c_B`, improved
/// by one step of iterative refinement, and what is known of its accuracy.
struct Duals<T> {
    /// Whether the costs are those of phase one, the marks of the broken bounds, rather than the
    /// objective's.
    phase_one: bool,
    /// `y`.
    y: Vec<T>,
    /// The correction the refinement made, `B⁻ᵀ·(c_B − Bᵀ·y₀)` for the first solution `y₀`: the
    /// error `y₀` had, and so a measure of the error the refined `y` may still have. Pricing takes
    /// it for one.
    correction: Vec<T>,
    /// For each position `k` of the basis, a bound on the residual `|c_B − Bᵀ·y|_k` of the refined
    /// `y`: the residual as computed and the rounding it was computed with. The multipliers of the
    /// basis itself differ from `y` by `B⁻ᵀ` times the residual, so the reduced cost of variable
    /// `j` under them differs from the one under `y` by at most `Σ_k |(B⁻¹·a_j)_k|·residual_k`.
    residual: Vec<T>,
}

/// A reduced cost and what it is made of.
#[derive(Debug, Clone, Copy)]
struct Priced<T> {
    /// `d_j = c_j − a_jᵀ·y`, with `c_j` the cost of the phase.
    d: T,
    /// `|c_j| + |a_jᵀ·y|`, the size of the two numbers `d_j` is the difference of: the tolerance
    /// is relative to it.
    scale: T,
    /// `|c_j| + Σ_i |a_ij·y_i|`, the size of the terms `d_j` is computed from: its rounding is
    /// relative to it.
    terms: T,
    /// `a_jᵀ·e`, with `e` the correction of the refinement: what the refinement changed `d_j` by.
    change: T,
}

/// What a certificate makes of the state the method stopped in.
enum Verdict<T> {
    /// The status is proven.
    Proven,
    /// It is not proven, and the method cannot go on.
    Fails,
    /// It is not proven, because this variable can still improve the objective of the phase in
    /// this direction by more than its reduced cost's error: pricing passed it over on an estimate
    /// of that error. It is to enter the basis next.
    Enter(usize, T, Priced<T>),
}

/// The state of the simplex method on the working problem: minimise `costᵀ·x` subject to
/// `A·x_s + x_l = b` and `lower ≤ x ≤ upper`, where `x` is the `n` structural variables `x_s`
/// followed by the `m` logical variables `x_l`, one for each row.
struct Engine<T> {
    /// `m`, the number of rows.
    rows: usize,
    /// `n`, the number of structural variables.
    columns: usize,
    /// `A`, `m × n`, scaled.
    a: Matrix<T>,
    /// `|b_i| + Σ_j |a_ij·x_j|`, the size of the terms of each row, at the point where the basis
    /// was last factorised ([`measure`](Self::measure)).
    activity: Vec<T>,
    /// For each variable that was basic when the basis was last factorised, a bound on the error
    /// of the value the factorisation gave it, which no bound of it is held more tightly than:
    /// what the residual of the rows and its rounding make of that value through the basis,
    /// `|B⁻¹|·(|r| + γ·t)` ([`measure`](Self::measure)). It is the error of the variable's own
    /// value: large numbers in rows the basis does not couple it to do not enter it. Zero for the
    /// other variables, which sit exactly where they are put; a variable that enters the basis
    /// gets its floor at the next factorisation, and until then is held to the tolerance alone.
    floor: Vec<T>,
    /// `b`.
    b: Vec<T>,
    /// The costs of phase two, zero for the logical variables.
    cost: Vec<T>,
    /// The lower bounds.
    lower: Vec<T>,
    /// The upper bounds.
    upper: Vec<T>,
    /// The point.
    x: Vec<T>,
    /// Where each variable is.
    place: Vec<Place>,
    /// For a basic variable that breaks a bound by more than the tolerance, which: `-1` when it
    /// is below its lower bound and `1` when it is above its upper bound. While it is so marked
    /// its cost is that mark, the bound it breaks is the only one it has, and it is the far side
    /// of it: phase one brings the variable back to that bound and no further. `0` otherwise.
    broken: Vec<i8>,
    /// The basic variable of each row of the basis.
    basis: Vec<usize>,
    /// The basis, factorised.
    factor: Basis<T>,
    /// The variables pricing passes over: those whose reduced cost turned out, on their column,
    /// to be no more than its error, and those along which the ratio test found no bound and no
    /// ray could be certified. Cleared at every change of basis, of phase and of factors.
    banned: Vec<bool>,
    /// The tolerance.
    tol: T,
    /// The size, relative to the larger of one and the largest element of the entering column,
    /// below which an element of that column is not pivoted on.
    small: T,
    /// The largest number of pivots.
    max_iter: usize,
    /// The number of pivots without progress after which Bland's rule takes over.
    stall_limit: usize,
    /// The number of pivots taken.
    pivots: usize,
    /// The number of pivots in a row that have left the objective where it was.
    stalled: usize,
    /// Whether the magnitudes of the inverse of the basis are computed by one blocked solve
    /// ([`Basis::invert_blocked`]), as they are once the dual simplex method has run.
    blocked: bool,
}

impl<T: Real> Engine<T> {
    /// The size the tolerances on the bounds of variable `j` are relative to: the larger of one
    /// and `|bound|` and, for the logical variable of a row, the size of the row's terms at the
    /// point, [`activity`](Self::activity). A logical variable is computed as the difference of
    /// the row's right-hand side and its terms, so it is known only to a precision relative to
    /// them, however small the bound it is held to.
    fn scale_of(&self, j: usize, bound: T) -> T {
        let scale = mag(bound).max(T::ONE);
        if j < self.columns {
            scale
        } else {
            scale.max(self.activity[j - self.columns])
        }
    }

    /// By how much variable `j` may be outside `bound` and still count as inside: the tolerance
    /// times [`scale_of`](Self::scale_of), and never less than the [`floor`](Self::floor) of the
    /// variable, the error its value may carry.
    fn allowance(&self, j: usize, bound: T) -> T {
        (self.tol * self.scale_of(j, bound)).max(self.floor[j])
    }

    /// Computes, at the point and on a basis just factorised, the sizes of the rows' terms,
    /// `|b_i| + Σ_j |a_ij·x_j|`, and the [`floor`](Self::floor) of every variable; `None` when
    /// either is not finite, which only numbers near the end of the floating-point range give.
    ///
    /// The basic variables are what the basis makes of the rest of the point: the exact basic
    /// solution `x_B` satisfies `B·x_B = b − N·x_N`, so the computed one differs from it by
    /// `B⁻¹·r`, where `r = b − A·x_s − x_l` is the residual of the rows at the point. The residual
    /// is computed with a rounding of at most `γ·t_i` in row `i`, `t_i` the sum of the magnitudes
    /// of its terms and `γ = (n + 2)·ε` for a sum of as many terms; so the error of basic variable
    /// `k` is at most `(|B⁻¹|·(|r| + γ·t))_k` ([`Basis::bound_solve`]), which [`ERROR_SAFETY`]
    /// enlarges for what this first-order bound leaves out. It follows the coupling of the basis:
    /// a basic logical variable whose row meets no basic structural one is known to the rounding
    /// of its own row, however large the numbers of other rows, and a variable that the basis
    /// computes from large numbers through an ill-conditioned basis is known only as well as that
    /// allows.
    fn measure(&mut self) -> Option<()> {
        let (m, n) = (self.rows, self.columns);
        let mut residual = self.b.clone();
        let mut activity: Vec<T> = self.b.iter().map(|v| mag(*v)).collect();
        for j in 0..n {
            let x = self.x[j];
            if x != T::ZERO {
                for ((r, s), c) in residual.iter_mut().zip(&mut activity).zip(self.a.column(j)) {
                    let term = *c * x;
                    *r -= term;
                    *s += mag(term);
                }
            }
        }
        let gamma = T::EPSILON * T::from_f64((n + 2) as f64);
        let errors: Vec<T> = residual
            .iter()
            .zip(&activity)
            .zip(&self.x[n..])
            .map(|((r, s), z)| mag(*r - *z) + gamma * (*s + mag(*z)))
            .collect();
        let mut through = vec![T::ZERO; m];
        self.factor.bound_solve(&errors, &mut through);
        self.floor.fill(T::ZERO);
        for (&j, f) in self.basis.iter().zip(&through) {
            self.floor[j] = T::from_f64(ERROR_SAFETY) * *f;
        }
        let measured = activity.iter().chain(&self.floor).all(|v| finite(*v));
        self.activity = activity;
        measured.then_some(())
    }

    /// `γ`, a bound on the relative rounding of a sum of as many products as a column of
    /// `[A I]` has elements, with one term more: `(m + 2)·ε`.
    fn gamma(&self) -> T {
        T::EPSILON * T::from_f64((self.rows + 2) as f64)
    }

    /// The inner product of column `j` of `[A I]` with `v`.
    fn against(&self, j: usize, v: &[T]) -> T {
        if j < self.columns {
            dot(self.a.column(j), v)
        } else {
            v[j - self.columns]
        }
    }

    /// The inner product of column `j` of `[A I]` with `v`, and the sum of the magnitudes of its
    /// terms.
    fn against_with_size(&self, j: usize, v: &[T]) -> (T, T) {
        if j < self.columns {
            let mut sum = T::ZERO;
            let mut size = T::ZERO;
            for (a, x) in self.a.column(j).iter().zip(v) {
                let term = *a * *x;
                sum += term;
                size += mag(term);
            }
            (sum, size)
        } else {
            let x = v[j - self.columns];
            (x, mag(x))
        }
    }

    /// Adds `scale` times column `j` of `[A I]` to `v`.
    fn add_column(&self, j: usize, scale: T, v: &mut [T]) {
        if j < self.columns {
            for (v, c) in v.iter_mut().zip(self.a.column(j)) {
                *v += *c * scale;
            }
        } else {
            v[j - self.columns] += scale;
        }
    }

    /// Writes `B⁻¹` times column `j` of `[A I]`, the column as the basis sees it, to `out`.
    fn column_of(&self, j: usize, out: &mut [T]) {
        let mut entering = vec![T::ZERO; self.rows];
        self.add_column(j, T::ONE, &mut entering);
        self.factor.solve(&entering, out);
    }

    /// The cost of variable `j` in the phase: in phase one the mark of the bound it breaks, which
    /// is zero for a variable that breaks none and for every variable out of the basis; in phase
    /// two its cost.
    fn phase_cost(&self, j: usize, phase_one: bool) -> T {
        if phase_one {
            T::from_f64(f64::from(self.broken[j]))
        } else {
            self.cost[j]
        }
    }

    /// The multipliers of the basis for the costs of the phase, the solution of `Bᵀ·y = c_B`, with
    /// one step of iterative refinement, the correction it made and a bound on the residual left.
    fn duals(&self, phase_one: bool) -> Duals<T> {
        let costs: Vec<T> = self
            .basis
            .iter()
            .map(|&j| self.phase_cost(j, phase_one))
            .collect();
        let mut work = costs.clone();
        let mut y = vec![T::ZERO; self.rows];
        self.factor.solve_transposed(&mut work, &mut y);
        for ((w, &j), c) in work.iter_mut().zip(&self.basis).zip(&costs) {
            *w = *c - self.against(j, &y);
        }
        let mut correction = vec![T::ZERO; self.rows];
        self.factor.solve_transposed(&mut work, &mut correction);
        for (v, e) in y.iter_mut().zip(&correction) {
            *v += *e;
        }
        let gamma = self.gamma();
        let residual = self
            .basis
            .iter()
            .zip(&costs)
            .map(|(&j, &c)| {
                let (sum, size) = self.against_with_size(j, &y);
                mag(c - sum) + gamma * (mag(c) + size)
            })
            .collect();
        Duals {
            phase_one,
            y,
            correction,
            residual,
        }
    }

    /// The reduced cost of variable `j` under `duals`, with the size of its terms and what the
    /// refinement changed it by.
    fn priced(&self, j: usize, duals: &Duals<T>) -> Priced<T> {
        let cost = self.phase_cost(j, duals.phase_one);
        let (sum, size, change) = if j < self.columns {
            let mut sum = T::ZERO;
            let mut size = T::ZERO;
            let mut change = T::ZERO;
            let column = self.a.column(j);
            for ((a, y), e) in column.iter().zip(&duals.y).zip(&duals.correction) {
                let term = *a * *y;
                sum += term;
                size += mag(term);
                change += *a * *e;
            }
            (sum, size, change)
        } else {
            let i = j - self.columns;
            (duals.y[i], mag(duals.y[i]), duals.correction[i])
        };
        Priced {
            d: cost - sum,
            scale: mag(cost) + mag(sum),
            terms: mag(cost) + size,
            change,
        }
    }

    /// The error pricing takes a reduced cost to carry: the rounding of its own sum and
    /// [`PRICING_SAFETY`] times what the refinement changed it by.
    fn estimate(&self, priced: &Priced<T>) -> T {
        self.gamma() * priced.terms + T::from_f64(PRICING_SAFETY) * mag(priced.change)
    }

    /// A bound on the error of a reduced cost, given the column `column = B⁻¹·a_j` of its
    /// variable: what the residual of the multipliers makes of it through that column,
    /// `Σ_k |column_k|·residual_k`, and the rounding of its own sum, [`ERROR_SAFETY`] times over.
    /// It is the error in the variable's own terms: multipliers that are large, or wrong, in rows
    /// the column does not reach do not enter it.
    fn error_bound(&self, priced: &Priced<T>, column: &[T], duals: &Duals<T>) -> T {
        let mut through = T::ZERO;
        for (c, r) in column.iter().zip(&duals.residual) {
            through += mag(*c) * *r;
        }
        T::from_f64(ERROR_SAFETY) * (through + self.gamma() * priced.terms)
    }

    /// The bounds the ratio test holds variable `j` to: its own, or, while it breaks one, that
    /// one from the far side.
    fn working_bounds(&self, j: usize) -> (T, T) {
        match self.broken[j] {
            0 => (self.lower[j], self.upper[j]),
            mark if mark < 0 => (-T::INFINITY, self.lower[j]),
            _ => (self.upper[j], T::INFINITY),
        }
    }

    /// Marks the basic variables that break a bound by more than the tolerance, and no others.
    fn classify(&mut self) {
        for k in 0..self.rows {
            let j = self.basis[k];
            let x = self.x[j];
            self.broken[j] = if x < self.lower[j] - self.allowance(j, self.lower[j]) {
                -1
            } else if x > self.upper[j] + self.allowance(j, self.upper[j]) {
                1
            } else {
                0
            };
        }
    }

    /// Whether the point still breaks a bound by more than the tolerance, which is whether the
    /// next pivot belongs to phase one. A marked variable that has come back to its bound is
    /// unmarked on the way, and once none breaks a bound by more than the tolerance all are.
    fn phase_one(&mut self) -> bool {
        let mut marked = false;
        let mut infeasible = false;
        for &j in &self.basis {
            let mark = self.broken[j];
            if mark == 0 {
                continue;
            }
            let (excess, bound) = if mark < 0 {
                (self.lower[j] - self.x[j], self.lower[j])
            } else {
                (self.x[j] - self.upper[j], self.upper[j])
            };
            if excess <= T::ZERO {
                self.broken[j] = 0;
            } else {
                marked = true;
                infeasible |= excess > self.allowance(j, bound);
            }
        }
        if marked && !infeasible {
            for &j in &self.basis {
                self.broken[j] = 0;
            }
        }
        infeasible
    }

    /// The direction (`1` up, `−1` down) in which variable `j`, out of the basis at `place`, would
    /// improve the objective of the phase given its reduced cost `d`, when `d` is beyond
    /// `threshold`; `None` when it would not, or not by more than that.
    fn favourable(place: Place, d: T, threshold: T) -> Option<T> {
        match place {
            Place::Lower | Place::Zero if d < -threshold => Some(T::ONE),
            Place::Upper | Place::Zero if d > threshold => Some(-T::ONE),
            _ => None,
        }
    }

    /// The variable to enter the basis, the direction it moves in and its reduced cost; or `None`
    /// when no variable's reduced cost is favourable by more than the tolerance and the
    /// [`estimate`](Self::estimate) of its error. Dantzig's rule takes the largest reduced cost,
    /// the first of equals; Bland's the first that is eligible.
    fn price(&self, duals: &Duals<T>, bland: bool) -> Option<(usize, T, Priced<T>)> {
        let mut best: Option<(usize, T, Priced<T>)> = None;
        for j in 0..self.columns + self.rows {
            let place = self.place[j];
            if matches!(place, Place::Basic | Place::Fixed) || self.banned[j] {
                continue;
            }
            let priced = self.priced(j, duals);
            let threshold = self.tol * priced.scale + self.estimate(&priced);
            let Some(direction) = Self::favourable(place, priced.d, threshold) else {
                continue;
            };
            if bland {
                return Some((j, direction, priced));
            }
            if best
                .as_ref()
                .is_none_or(|(_, _, other)| mag(priced.d) > mag(other.d))
            {
                best = Some((j, direction, priced));
            }
        }
        best
    }

    /// How far the basic variable of row `i` is from the bound it moves towards, and that bound,
    /// when the entering variable moves in the direction `sigma` and `a` is the element of its
    /// column in that row; `None` when the element is too small to pivot on or the variable has
    /// no bound that way. The distance is negative when the variable is already past the bound.
    fn room(&self, i: usize, sigma: T, a: T, small: T) -> Option<(T, T)> {
        if mag(a) <= small {
            return None;
        }
        let j = self.basis[i];
        let (low, high) = self.working_bounds(j);
        // The basic variable changes by `−sigma·a` for each unit of the step.
        let (bound, distance) = if sigma * a > T::ZERO {
            (low, self.x[j] - low)
        } else {
            (high, high - self.x[j])
        };
        finite(bound).then_some((distance, bound))
    }

    /// The ratio test: how far variable `q` can move in the direction `sigma`, `column` being its
    /// column as the basis sees it, and what stops it.
    ///
    /// The first pass finds the longest step that takes no basic variable past its bound by more
    /// than a slack: a tenth of its tolerance under Dantzig's rule (Harris's test), rounding
    /// alone under Bland's. The second takes, of the rows whose own ratio is within that step,
    /// the one with the largest pivot under Dantzig's rule, and the one whose variable has the
    /// smallest index under Bland's. The step is then the ratio of that row, or zero when its
    /// variable is already past its bound.
    fn ratio(&self, q: usize, sigma: T, column: &[T], bland: bool) -> Step<T> {
        let largest = column.iter().fold(T::ONE, |big, a| big.max(mag(*a)));
        let small = self.small * largest;
        let mut limit = T::INFINITY;
        for (i, &a) in column.iter().enumerate() {
            if let Some((distance, bound)) = self.room(i, sigma, a, small) {
                let j = self.basis[i];
                let room = if bland {
                    T::EPSILON * T::from_f64(ROUNDING) * self.scale_of(j, bound)
                } else {
                    T::from_f64(HARRIS) * self.tol * self.scale_of(j, bound)
                };
                let relaxed = (distance + room).max(T::ZERO);
                limit = limit.min(relaxed / mag(a));
            }
        }
        // The entering variable's own way to its other bound.
        let flip = if sigma > T::ZERO {
            self.upper[q] - self.x[q]
        } else {
            self.x[q] - self.lower[q]
        };
        let mut leaving: Option<(usize, T, T)> = None;
        for (i, &a) in column.iter().enumerate() {
            let Some((distance, _)) = self.room(i, sigma, a, small) else {
                continue;
            };
            let ratio = distance / mag(a);
            if ratio > limit {
                continue;
            }
            let better = match leaving {
                None => true,
                Some((row, _, _)) if bland => self.basis[i] < self.basis[row],
                Some((row, _, _)) => {
                    let pivot = mag(column[row]);
                    mag(a) > pivot || (mag(a) == pivot && self.basis[i] < self.basis[row])
                }
            };
            if better {
                leaving = Some((i, ratio, mag(a)));
            }
        }
        match leaving {
            Some((row, ratio, _)) => {
                let step = ratio.max(T::ZERO);
                if flip <= step {
                    Step::Flip(flip)
                } else {
                    Step::Pivot(row, step)
                }
            }
            None if finite(flip) => Step::Flip(flip),
            None => Step::Unbounded,
        }
    }

    /// Whether the entering variable `q`, priced as `priced`, descends: its reduced cost is beyond
    /// `threshold`, the tolerance and the [`error_bound`](Self::error_bound) its column
    /// `column = B⁻¹·a_q` gives, and the reduced cost worked out the other way,
    /// `c_q − c_Bᵀ·column`, has the same sign. A reduced cost within its error is rounding, not
    /// a direction of descent, and a step along its column would carry the point to wherever a
    /// bound stops it, however far from the data that is.
    fn descends(
        &self,
        q: usize,
        priced: &Priced<T>,
        threshold: T,
        column: &[T],
        phase_one: bool,
    ) -> bool {
        if mag(priced.d) <= threshold {
            return false;
        }
        let mut again = self.phase_cost(q, phase_one);
        for (&j, &a) in self.basis.iter().zip(column) {
            again -= self.phase_cost(j, phase_one) * a;
        }
        again * priced.d > T::ZERO
    }

    /// The size of the objective of the phase at the point: the sum of the amounts by which the
    /// marked variables break their bounds in phase one, `Σ|c_j·x_j|` in phase two. A pivot
    /// that changes the objective by no more than the tolerance times this has not moved it.
    fn objective_size(&self, phase_one: bool) -> T {
        let mut size = T::ZERO;
        if phase_one {
            for &j in &self.basis {
                size += match self.broken[j] {
                    0 => T::ZERO,
                    mark if mark < 0 => (self.lower[j] - self.x[j]).max(T::ZERO),
                    _ => (self.x[j] - self.upper[j]).max(T::ZERO),
                };
            }
        } else {
            for (c, x) in self.cost.iter().zip(&self.x) {
                size += mag(*c * *x);
            }
        }
        size
    }

    /// Moves the entering variable `q` by `step` in the direction `sigma`, and the basic
    /// variables with it.
    fn advance(&mut self, q: usize, sigma: T, column: &[T], step: T) {
        if step > T::ZERO {
            for (&j, &a) in self.basis.iter().zip(column) {
                self.x[j] -= sigma * a * step;
            }
            self.x[q] += sigma * step;
        }
    }

    /// The point as it is reported, when it is feasible, and the sizes of the rows' terms there;
    /// `None` when it is not.
    ///
    /// Each structural variable has to be inside its bounds within its
    /// [`allowance`](Self::allowance) — the tolerance, or the error its value carries when that
    /// is larger — and is then put inside them, as the result reports it. Each row is checked at
    /// that point: what it leaves of its right-hand side, `z_i = b_i − Σ_j a_ij·x_j`, has to be
    /// inside the bounds of its logical variable within `tol` times the larger of one and the
    /// size of its terms, `|b_i| + Σ_j |a_ij·x_j|`. The point returned holds the structural
    /// variables so placed and the logical variables `z_i`. The rows are checked where the
    /// point is reported, not where it was computed, so that taking the error of a value off it
    /// cannot leave a row broken by more than the tolerance.
    fn primal_check(&self) -> Option<(Vec<T>, Vec<T>)> {
        let (m, n) = (self.rows, self.columns);
        let mut point = Vec::with_capacity(n + m);
        for j in 0..n {
            let x = self.x[j];
            let inside = x >= self.lower[j] - self.allowance(j, self.lower[j])
                && x <= self.upper[j] + self.allowance(j, self.upper[j]);
            if !inside {
                return None;
            }
            point.push(x.max(self.lower[j]).min(self.upper[j]));
        }
        let mut logical = self.b.clone();
        let mut size: Vec<T> = self.b.iter().map(|v| mag(*v)).collect();
        for (j, &x) in point.iter().enumerate() {
            if x == T::ZERO {
                continue;
            }
            for ((z, s), c) in logical.iter_mut().zip(&mut size).zip(self.a.column(j)) {
                let term = *c * x;
                *z -= term;
                *s += mag(term);
            }
        }
        for (i, (z, s)) in logical.iter().zip(&size).enumerate() {
            let room = self.tol * s.max(T::ONE);
            if *z < self.lower[n + i] - room || *z > self.upper[n + i] + room {
                return None;
            }
        }
        point.extend_from_slice(&logical);
        Some((point, size))
    }

    /// What the reduced costs of the variables out of the basis say of the basis, for a
    /// certificate: [`Verdict::Enter`] for the first variable whose reduced cost is favourable
    /// by more than the tolerance and its [`error_bound`](Self::error_bound), or
    /// [`Verdict::Fails`] when that variable has been passed over since the basis was
    /// factorised; [`Verdict::Proven`] when there is none.
    fn every_reduced_cost(&self, duals: &Duals<T>) -> Verdict<T> {
        let mut column = vec![T::ZERO; self.rows];
        for j in 0..self.columns + self.rows {
            let place = self.place[j];
            if matches!(place, Place::Basic | Place::Fixed) {
                continue;
            }
            let priced = self.priced(j, duals);
            let tolerance = self.tol * priced.scale;
            if Self::favourable(place, priced.d, tolerance).is_none() {
                continue;
            }
            self.column_of(j, &mut column);
            let threshold = tolerance + self.error_bound(&priced, &column, duals);
            if let Some(sigma) = Self::favourable(place, priced.d, threshold) {
                return if self.banned[j] {
                    Verdict::Fails
                } else {
                    Verdict::Enter(j, sigma, priced)
                };
            }
        }
        Verdict::Proven
    }

    /// Whether the point, as it is reported, and the multipliers `duals` of phase two prove each
    /// other optimal, on a basis factorised afresh.
    ///
    /// * The point is feasible ([`primal_check`](Self::primal_check)).
    /// * No reduced cost is favourable by more than the tolerance and its error bound
    ///   ([`every_reduced_cost`](Self::every_reduced_cost)), and the reduced costs of the basic
    ///   variables are zero to the residual of the multipliers.
    ///
    /// The two make the multipliers a feasible point of the dual programme, to the tolerance, so
    /// by weak duality no feasible point has an objective below the dual objective
    /// `bᵀ·y + Σ_j d_j·β_j`, where `β_j` is the bound `d_j` prices — the variable's own bound,
    /// where the sign of `d_j` is the one its bound asks for, and none where `d_j` is taken for
    /// zero — and the objective at the point is that dual objective up to what the rows'
    /// residuals and the reduced costs taken for zero are worth. The last test, that the two
    /// objectives agree within what the rows' residuals, held to the tolerance, are worth at `y`,
    /// what the reduced costs taken for zero are worth at the point, and the rounding of the
    /// sums, is a check that the two sides were computed consistently: the first two imply it,
    /// up to rounding.
    fn certify_optimal(&self, duals: &Duals<T>) -> Verdict<T> {
        let Some((point, sizes)) = self.primal_check() else {
            return Verdict::Fails;
        };
        match self.every_reduced_cost(duals) {
            Verdict::Proven => {}
            other => return other,
        }
        let y = &duals.y;
        let gamma = self.gamma();
        let mut primal = T::ZERO;
        let mut dual = T::ZERO;
        // The sum of the magnitudes of the terms of both sides, for their rounding.
        let mut size = T::ZERO;
        // What the rows' residuals may be worth at the multipliers: each row holds to the
        // tolerance times the larger of one and the size of its terms.
        let mut rows = T::ZERO;
        for ((v, b), s) in y.iter().zip(&self.b).zip(&sizes) {
            dual += *v * *b;
            size += mag(*v * *b);
            rows += mag(*v) * s.max(T::ONE);
        }
        for (c, x) in self.cost.iter().zip(&point) {
            primal += *c * *x;
            size += mag(*c * *x);
        }
        // What the reduced costs taken for zero are worth at the point.
        let mut excluded = T::ZERO;
        for (k, &j) in self.basis.iter().enumerate() {
            let priced = self.priced(j, duals);
            let zero = self.tol * priced.scale
                + T::from_f64(ERROR_SAFETY) * (duals.residual[k] + gamma * priced.terms);
            if mag(priced.d) > zero {
                return Verdict::Fails;
            }
            excluded += mag(priced.d * point[j]);
        }
        // A variable out of the basis sits exactly at the bound it prices.
        for j in 0..self.columns + self.rows {
            let place = self.place[j];
            if place == Place::Basic {
                continue;
            }
            let priced = self.priced(j, duals);
            let x = self.x[j];
            let prices_its_bound = match place {
                Place::Lower => priced.d >= T::ZERO,
                Place::Upper => priced.d <= T::ZERO,
                Place::Fixed => true,
                _ => false,
            };
            if prices_its_bound {
                dual += priced.d * x;
                size += priced.terms * mag(x);
            } else {
                excluded += mag(priced.d * x);
            }
        }
        let rounding = (T::EPSILON * T::from_f64(ROUNDING) + gamma) * size;
        if mag(primal - dual) <= self.tol * rows + excluded + rounding {
            Verdict::Proven
        } else {
            Verdict::Fails
        }
    }

    /// Whether the multipliers `duals` of phase one prove that no point is feasible, by Farkas's
    /// lemma, on a basis factorised afresh.
    ///
    /// Every point of the rows has `yᵀ·[A I]·x = yᵀ·b`; over the box of the bounds the left side
    /// is at most the sum of `g_j·u_j` where `g_j = yᵀ·[A I]_j` is positive and `g_j·l_j` where it
    /// is negative, and when that sum is below `yᵀ·b` no point of the box is a point of the
    /// rows. For the multipliers of the basis, `g_j` of a basic variable is its mark, `−1` where
    /// it is below its lower bound, `1` where it is above its upper bound and `0` otherwise, so
    /// it meets only the finite bound it breaks. A variable out of the basis sits at the bound
    /// its `g_j` meets, or `g_j` is taken for zero: no reduced cost of phase one may be
    /// favourable by more than the tolerance and its error bound
    /// ([`every_reduced_cost`](Self::every_reduced_cost)).
    ///
    /// The difference of the two sides, the margin, is then the sum of the amounts by which the
    /// marked variables break their bounds. It is asked to be at least half the tolerance and to
    /// be larger than what the error of the multipliers can make of it, `Σ_k |x_B,k|·residual_k`
    /// (the margin of the multipliers of the basis differs from that of `y` by the residual
    /// times `B⁻¹·(b − N·x_N)`, which is the basic part of the point), with the rounding of the
    /// sums: multipliers that are large, and so carry a large error, prove nothing by a margin
    /// that error could make.
    fn certify_infeasible(&self, duals: &Duals<T>) -> Verdict<T> {
        match self.every_reduced_cost(duals) {
            Verdict::Proven => {}
            other => return other,
        }
        let y = &duals.y;
        let mut asked = T::ZERO;
        let mut most = T::ZERO;
        let mut size = T::ZERO;
        for (v, b) in y.iter().zip(&self.b) {
            asked += *v * *b;
            size += mag(*v * *b);
        }
        let mut error = T::ZERO;
        for (k, &j) in self.basis.iter().enumerate() {
            match self.broken[j] {
                0 => {}
                mark if mark < 0 => {
                    most -= self.lower[j];
                    size += mag(self.lower[j]);
                }
                _ => {
                    most += self.upper[j];
                    size += mag(self.upper[j]);
                }
            }
            error += mag(self.x[j]) * duals.residual[k];
        }
        for j in 0..self.columns + self.rows {
            if self.place[j] == Place::Basic {
                continue;
            }
            // In phase one a variable out of the basis has no cost: `g_j = −d_j`.
            let priced = self.priced(j, duals);
            let x = self.x[j];
            most -= priced.d * x;
            size += priced.terms * mag(x);
        }
        let margin = asked - most;
        let doubt = T::from_f64(ERROR_SAFETY) * error
            + (T::EPSILON * T::from_f64(ROUNDING) + self.gamma()) * size;
        if margin > self.tol * T::HALF && margin > doubt {
            Verdict::Proven
        } else {
            Verdict::Fails
        }
    }

    /// Whether the programme is unbounded: the point is feasible, and the ray along which
    /// variable `q` moves in the direction `sigma`, the basic variables following by `column`,
    /// keeps every row (`[A I]·ray = 0`), runs into no finite bound, and improves the objective
    /// by more than half of `threshold`, the tolerance and the error bound of `q`'s reduced cost,
    /// for each unit that `q` moves. A component of the ray that would run into a bound has to
    /// be one the ratio test took for zero, and is made zero before the rows are checked.
    fn certify_unbounded(&self, q: usize, sigma: T, column: &[T], threshold: T) -> bool {
        if self.primal_check().is_none() {
            return false;
        }
        let (m, n) = (self.rows, self.columns);
        let mut ray = vec![T::ZERO; n + m];
        ray[q] = sigma;
        for (&j, &a) in self.basis.iter().zip(column) {
            ray[j] = -sigma * a;
        }
        let largest = ray.iter().fold(T::ONE, |big, v| big.max(mag(*v)));
        let small = self.small * largest;
        for (j, r) in ray.iter_mut().enumerate() {
            let blocked =
                (*r > T::ZERO && finite(self.upper[j])) || (*r < T::ZERO && finite(self.lower[j]));
            if blocked {
                if mag(*r) > small {
                    return false;
                }
                *r = T::ZERO;
            }
        }
        let mut image = ray[n..].to_vec();
        let mut slope = T::ZERO;
        let mut size = T::ZERO;
        for (j, &r) in ray.iter().enumerate() {
            if r == T::ZERO {
                continue;
            }
            if j < n {
                self.add_column(j, r, &mut image);
            }
            slope += self.cost[j] * r;
            size += mag(self.cost[j] * r);
        }
        let allowed = T::TWO * self.tol * largest;
        image.iter().all(|v| mag(*v) <= allowed)
            && slope < -threshold * T::HALF
            && -slope > T::EPSILON * T::from_f64(ROUNDING) * size
    }
}

impl<T: Real + Scalar> Engine<T> {
    /// The method at its start: the logical variables basic, every structural variable at zero
    /// when that is inside its bounds and at the bound nearer to zero otherwise. `None` when the
    /// sizes of the rows' terms at that point are not finite.
    fn new(
        a: Matrix<T>,
        b: Vec<T>,
        cost: Vec<T>,
        lower: Vec<T>,
        upper: Vec<T>,
        opts: &LpOptions<T>,
    ) -> Option<Self> {
        let (m, n) = a.shape();
        let mut engine = Self {
            rows: m,
            columns: n,
            a,
            activity: vec![T::ZERO; m],
            floor: vec![T::ZERO; n + m],
            b,
            cost,
            lower,
            upper,
            x: vec![T::ZERO; n + m],
            place: vec![Place::Basic; n + m],
            broken: vec![0; n + m],
            basis: (n..n + m).collect(),
            factor: Basis::identity(m),
            banned: vec![false; n + m],
            tol: opts.tol,
            small: opts.tol.min(default_tolerance()),
            max_iter: opts.max_iter,
            stall_limit: opts.stall_limit,
            pivots: 0,
            stalled: 0,
            blocked: false,
        };
        engine.start()?;
        Some(engine)
    }

    /// Puts the method at its start, on factors of the identity, whatever state it is in but for
    /// the pivots it has counted: the logical variables basic, every structural variable at zero
    /// when that is inside its bounds and at the bound nearer to zero otherwise. `None` when the
    /// sizes of the rows' terms at that point are not finite.
    fn start(&mut self) -> Option<()> {
        let n = self.columns;
        for j in 0..n {
            // As near the origin as the bounds allow: at zero when it is inside them, else at the
            // bound nearer to it. A start at a bound far from the data, such as a large number
            // standing for an infinite one, would put that number into every row.
            let (lower, upper) = (self.lower[j], self.upper[j]);
            (self.place[j], self.x[j]) = if lower == upper {
                (Place::Fixed, lower)
            } else if lower >= T::ZERO {
                (Place::Lower, lower)
            } else if upper <= T::ZERO {
                (Place::Upper, upper)
            } else {
                (Place::Zero, T::ZERO)
            };
        }
        self.place[n..].fill(Place::Basic);
        self.broken.fill(0);
        for (k, j) in self.basis.iter_mut().enumerate() {
            *j = n + k;
        }
        self.banned.fill(false);
        self.stalled = 0;
        let mut logical = self.b.clone();
        for j in 0..n {
            self.add_column(j, -self.x[j], &mut logical);
        }
        self.x[n..].copy_from_slice(&logical);
        self.measure()?;
        self.classify();
        Some(())
    }

    /// Puts the method at the basis `basis`, the variables out of it at `place`: the end of the
    /// dual simplex method. The basis is factorised afresh and the basic variables computed from
    /// the data, as at any factorisation; `None` when it is singular to working precision, or
    /// when `basis` and `place` do not describe a basis and places the method can stand at: one
    /// variable for each row, no variable twice, exactly those placed in the basis, and every
    /// other at a finite bound, at both when they are equal, or at zero strictly inside them.
    fn adopt(&mut self, basis: &[usize], place: &[Place]) -> Option<()> {
        let total = self.columns + self.rows;
        if basis.len() != self.rows || place.len() != total {
            return None;
        }
        let mut seen = vec![false; total];
        for &j in basis {
            if j >= total || seen[j] || place[j] != Place::Basic {
                return None;
            }
            seen[j] = true;
        }
        for (j, &p) in place.iter().enumerate() {
            let (lower, upper) = (self.lower[j], self.upper[j]);
            let valid = match p {
                Place::Basic => seen[j],
                Place::Lower => finite(lower) && lower != upper,
                Place::Upper => finite(upper) && lower != upper,
                Place::Fixed => lower == upper,
                Place::Zero => lower < T::ZERO && upper > T::ZERO,
            };
            if !valid {
                return None;
            }
        }
        for (j, &p) in place.iter().enumerate() {
            self.place[j] = p;
            self.x[j] = match p {
                Place::Lower | Place::Fixed => self.lower[j],
                Place::Upper => self.upper[j],
                Place::Zero | Place::Basic => T::ZERO,
            };
        }
        self.basis.copy_from_slice(basis);
        self.broken.fill(0);
        self.stalled = 0;
        self.refactor()
    }

    /// Whether the row of basis position `r`, which the dual simplex method found a ray of the
    /// dual programme in, proves that no point is feasible: the certificate of
    /// [`certify_infeasible`](Self::certify_infeasible) for the multipliers of phase one that
    /// mark only the basic variable of that row. A combination of rows is a certificate whatever
    /// the marks are, and the dual simplex method finds it in a single row; the marks of the
    /// other basic variables that break a bound are put back when it fails.
    fn certify_ray(&mut self, r: usize) -> bool {
        let j = self.basis[r];
        if self.broken[j] == 0 {
            return false;
        }
        let marks = self.broken.clone();
        for &k in &self.basis {
            if k != j {
                self.broken[k] = 0;
            }
        }
        let duals = self.duals(true);
        let proven = duals.y.iter().all(|v| finite(*v))
            && matches!(self.certify_infeasible(&duals), Verdict::Proven);
        if !proven {
            self.broken = marks;
        }
        proven
    }

    /// Runs the dual simplex method, then the primal method from the basis it ends at, which
    /// checks its answer by the certificates of [`run`](Self::run) and pivots on where they fail;
    /// and when that gives no answer, the primal method from its start. The pivots of all of
    /// them count towards the limit. `lower` and `upper` are the bounds the dual method works to:
    /// the working problem's, but for those that stand for infinite ones, made infinite; the
    /// primal method keeps to the working problem's own.
    fn run_dual(&mut self, lower: &[T], upper: &[T]) -> Option<LpStatus> {
        let finish = dual::solve(
            &self.a,
            &self.b,
            &self.cost,
            lower,
            upper,
            self.tol,
            self.max_iter,
        );
        self.pivots = finish.pivots;
        self.blocked = true;
        if self.adopt(&finish.basis, &finish.place).is_some() {
            if let dual::Outcome::Infeasible(r) = finish.outcome
                && self.certify_ray(r)
            {
                return Some(LpStatus::Infeasible);
            }
            if let Some(status) = self.run() {
                return Some(status);
            }
        }
        self.factor = Basis::identity(self.rows);
        self.start()?;
        self.run()
    }

    /// Factorises the basis afresh, computes the basic variables again from the data — the
    /// solution of `B·x_B = b − N·x_N`, with one step of iterative refinement — and marks those
    /// that break a bound. `None` when the basis is singular to working precision or the
    /// solution is not finite.
    fn refactor(&mut self) -> Option<()> {
        let (m, n) = (self.rows, self.columns);
        let columns = Matrix::from_fn(m, m, |i, k| {
            let j = self.basis[k];
            if j < n {
                self.a.column(j)[i]
            } else if j - n == i {
                T::ONE
            } else {
                T::ZERO
            }
        });
        self.factor = Basis::factor(&columns)?;
        let mut unit = vec![None; m];
        for (k, &j) in self.basis.iter().enumerate() {
            if j >= n {
                unit[j - n] = Some(k);
            }
        }
        if self.blocked {
            self.factor.invert_blocked(&unit);
        } else {
            self.factor.invert(&unit);
        }
        let mut rhs = self.b.clone();
        for j in 0..n + m {
            if self.place[j] != Place::Basic && self.x[j] != T::ZERO {
                self.add_column(j, -self.x[j], &mut rhs);
            }
        }
        let mut values = vec![T::ZERO; m];
        self.factor.solve(&rhs, &mut values);
        let mut residual = rhs;
        for (&j, v) in self.basis.iter().zip(&values) {
            self.add_column(j, -*v, &mut residual);
        }
        let mut correction = vec![T::ZERO; m];
        self.factor.solve(&residual, &mut correction);
        for ((&j, v), c) in self.basis.iter().zip(&values).zip(&correction) {
            let value = *v + *c;
            if !finite(value) {
                return None;
            }
            self.x[j] = value;
        }
        self.measure()?;
        self.classify();
        // A column passed over on the old factors may price otherwise on the new ones.
        self.banned.fill(false);
        Some(())
    }

    /// Runs the simplex method to one of the four statuses, each but the iteration limit checked
    /// by its certificate on a basis factorised afresh; `None` when a certificate fails, the
    /// basis becomes singular to working precision, or phase two loses the feasibility phase one
    /// found more than [`RESUMPTIONS`] times.
    fn run(&mut self) -> Option<LpStatus> {
        let mut column = vec![T::ZERO; self.rows];
        let mut was_phase_one = false;
        // Whether phase two has begun, and how often phase one has had to resume since.
        let mut feasible_once = false;
        let mut resumed = 0;
        loop {
            if self.factor.updates() >= REFACTOR_EVERY {
                self.refactor()?;
            }
            let phase_one = self.phase_one();
            if phase_one != was_phase_one {
                if phase_one && feasible_once {
                    // The point recomputed from the data breaks a bound that phase two was
                    // to keep: a step the ratio test could not see, through an element of the
                    // entering column below the pivot tolerance, or one that the rounding of a
                    // step through large numbers moved. Phase one restores it; when that keeps
                    // happening, the arithmetic cannot hold the point feasible, and no answer
                    // it reaches could be verified.
                    resumed += 1;
                    if resumed > RESUMPTIONS {
                        return None;
                    }
                }
                // A column passed over in one phase may be what the other needs.
                self.stalled = 0;
                self.banned.fill(false);
                was_phase_one = phase_one;
            }
            feasible_once |= !phase_one;
            let duals = self.duals(phase_one);
            if !duals.y.iter().all(|v| finite(*v)) {
                // Only a basis close to singular gives multipliers that are not finite; the
                // factors of a fresh one are checked for that.
                if self.factor.updates() > 0 {
                    self.refactor()?;
                    continue;
                }
                return None;
            }
            let bland = self.stalled >= self.stall_limit;
            let (q, sigma, priced) = match self.price(&duals, bland) {
                Some(found) => found,
                None => {
                    // Nothing is left to enter. An answer is only given on a basis factorised
                    // afresh, whose basic variables may differ by rounding and so price
                    // otherwise.
                    if self.factor.updates() > 0 {
                        self.refactor()?;
                        continue;
                    }
                    let verdict = if phase_one {
                        self.certify_infeasible(&duals)
                    } else {
                        self.certify_optimal(&duals)
                    };
                    match verdict {
                        Verdict::Proven if phase_one => return Some(LpStatus::Infeasible),
                        Verdict::Proven => return Some(LpStatus::Optimal),
                        Verdict::Fails => return None,
                        Verdict::Enter(q, sigma, priced) => (q, sigma, priced),
                    }
                }
            };
            if self.pivots >= self.max_iter {
                return Some(LpStatus::IterationLimit);
            }
            self.column_of(q, &mut column);
            let threshold = self.tol * priced.scale + self.error_bound(&priced, &column, &duals);
            if !self.descends(q, &priced, threshold, &column, phase_one) {
                self.banned[q] = true;
                continue;
            }
            let step = match self.ratio(q, sigma, &column, bland) {
                Step::Unbounded => {
                    if self.factor.updates() > 0 {
                        self.refactor()?;
                        continue;
                    }
                    if !phase_one && self.certify_unbounded(q, sigma, &column, threshold) {
                        return Some(LpStatus::Unbounded);
                    }
                    // Neither a pivot nor a ray: the column is passed over until the basis
                    // changes. If nothing else can enter, the certificate of the answer then
                    // reached sees this column too, and fails.
                    self.banned[q] = true;
                    continue;
                }
                Step::Flip(step) => {
                    self.advance(q, sigma, &column, step);
                    (self.x[q], self.place[q]) = if sigma > T::ZERO {
                        (self.upper[q], Place::Upper)
                    } else {
                        (self.lower[q], Place::Lower)
                    };
                    step
                }
                Step::Pivot(row, step) => {
                    let leaving = self.basis[row];
                    let (low, high) = self.working_bounds(leaving);
                    let target = if sigma * column[row] > T::ZERO {
                        low
                    } else {
                        high
                    };
                    self.advance(q, sigma, &column, step);
                    // The leaving variable goes to its bound exactly. When the step was cut to
                    // zero because it was already past the bound, that moves it by less than the
                    // slack of the ratio test; the next factorisation computes the basic
                    // variables again from the data and so removes the inconsistency.
                    self.x[leaving] = target;
                    self.broken[leaving] = 0;
                    self.place[leaving] = if self.lower[leaving] == self.upper[leaving] {
                        Place::Fixed
                    } else if target == self.lower[leaving] {
                        Place::Lower
                    } else {
                        Place::Upper
                    };
                    self.basis[row] = q;
                    self.place[q] = Place::Basic;
                    self.factor.replace(row, &column);
                    self.banned.fill(false);
                    step
                }
            };
            self.pivots += 1;
            // A pivot that changes the objective by no more than the tolerance, relative to the
            // size of the objective's terms, has left it where it was: enough of them in a row
            // and Bland's rule takes over.
            if step * mag(priced.d) <= self.tol * self.objective_size(phase_one) {
                self.stalled = self.stalled.saturating_add(1);
            } else {
                self.stalled = 0;
            }
        }
    }
}
