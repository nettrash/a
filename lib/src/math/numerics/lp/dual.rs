//! The dual simplex method, on the working problem of [`LinearProgram`](super::LinearProgram):
//! minimise `costᵀ·x` subject to `A·x_s + x_l = b` and `lower ≤ x ≤ upper`.
//!
//! It keeps the reduced costs of the variables out of the basis of the signs their bounds ask
//! for (dual feasibility) and works towards a basic point inside the bounds, one basic variable
//! that breaks a bound leaving the basis at a time. The parts are those of Koberstein's thesis
//! ("The dual simplex method, techniques for a fast and stable implementation", Paderborn, 2005):
//!
//! * **The basis** is factorised by the sparse LU decomposition of [`Factor`], with the changes of
//!   basis in the product form, and factorised afresh every [`REFACTOR`] changes.
//! * **Pricing** is by dual steepest edge (Forrest and Goldfarb, 1992, their dual algorithm 1):
//!   of the basic variables that break a bound, the one whose excess, squared, is the largest
//!   against the squared norm of its row of `B⁻¹`, the norms kept up to date at the cost of one
//!   more solve each change of basis, from one for every row at the basis of the logical
//!   variables.
//! * **The ratio test** is the bound-flipping one (Fourer, 1994): along the dual ray, a variable
//!   with two finite bounds whose reduced cost changes sign moves to its other bound, as long as
//!   the dual objective still rises, and only where it stops rising does a variable enter. Its
//!   passes are Harris's (1973): each takes the variables that block within the step a tolerance
//!   of the reduced costs allows, and the entering one is, of the last such group, the one of the
//!   largest pivot.
//! * **Phase one** makes the reduced costs feasible where the bounds alone cannot, by the
//!   subproblem of Fourer (1994) as Koberstein sets it out: the same method on the programme
//!   whose right-hand sides are zero and whose bounds are `[0, 0]` for a variable with two finite
//!   bounds, `[0, 1]` for one with only a lower bound, `[−1, 0]` for one with only an upper bound
//!   and `[−1000, 1000]` for a free one, whose optimum is the least sum of dual infeasibilities.
//! * **Degeneracy**: after [`DEGENERATE`] steps in a row that leave the dual objective where it
//!   was, the costs of the variables out of the basis are perturbed by small random amounts
//!   towards feasibility, and they are restored at the end.
//!
//! The method only proposes: the basis it ends at is handed to the primal method, which
//! factorises it afresh, densely, and checks it by its certificates, pivoting on from it where
//! they fail.

use super::factor::{Factor, dot, dot_gather};
use super::{Place, Real, finite, mag};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};

/// How many changes of basis the product form absorbs before the basis is factorised afresh.
// Measured on an Apple M4 Max, on the scoreboard's programmes (11 to 550 rows, densities 1, 0.1
// and 0.02), the netlib-like set and the battery's of 32 rows and more: against 64, 96 and 128
// make the dense ones of 110 to 550 rows 0.94 to 0.98 of their time and leave the rest within
// 1.4 %; 32 and 48 are up to 16 % slower, 192 and 256 make those of density 0.1 up to 6 % (sweep
// 32 to 256). With `ETA_FILL` 3 and `ROW_PRICE` 0.3 the three make every kind of programme 0.92
// to 1.01 of its time (geometric means, twice; 0.973 and 0.978 over 201 programmes).
const REFACTOR: usize = 96;

/// How many times the elements of the factors the elementary factors of the product form may hold
/// before the basis is factorised afresh, whatever their number: a sparse basis whose changes are
/// dense is factorised more often.
// Measured on an Apple M4 Max on the same programmes and random ones of density 0.1: against 2,
// 3 makes the netlib-like very sparse, boxed, free-variable and infeasible ones and the random
// ones 0.96 to 0.99 of their time, no kind more than 1.1 % slower; 1 and 1.5 are up to 8 %
// slower, 4 and 6 make the scoreboard's very sparse one of 550 rows 3 and 5 % slower (sweep 1
// to 6).
const ETA_FILL: f64 = 3.0;

/// The density of `A` from which a row of `B⁻¹·A` is computed by dense columns.
// Measured on an Apple M4 Max on random programmes of 110 and 220 rows: the dense columns
// overtake the rows of `A` between the densities 0.40 and 0.45 (sweep 0.2 to 0.8).
const DENSE_PRICE: f64 = 0.4;

/// The density of a row of `B⁻¹` below which its row of `B⁻¹·A` is computed row by row of `A`.
// Measured on an Apple M4 Max, against 0.1: 0.3 makes the netlib-like transportation, assignment
// and network problems 0.95, 0.95 and 0.98 of their time and random ones of density 0.1 and 12
// to 96 rows 0.96 to 0.99, no kind more than 0.4 % slower; above it random programmes of
// densities 0.15 to 0.35 and the scoreboard's of 0.1 and 0.02 turn slower, by 3 to 6 % at 0.5
// and up to 19 % at 0.7 (sweep 0.05 to 0.7).
const ROW_PRICE: f64 = 0.3;

/// How many steps in a row may leave the dual objective where it was before the costs are
/// perturbed.
// Measured on an Apple M4 Max: 5, 10 and 50 change the times of the scoreboard's, the
// netlib-like and the battery's programmes by at most 1 % (geometric means of each kind), and
// their pivots by less than 1 % (sweep 5 to 50).
const DEGENERATE: usize = 20;

/// The size of the perturbation of a cost, in units of the tolerance times `1 + |c_j|`: a random
/// amount between one and two times this.
// Measured on an Apple M4 Max: 10, 100 and 1000 change the times of the scoreboard's, the
// netlib-like and the battery's programmes by less than 0.3 %, and their pivots, the primal
// method's after the costs are restored among them, by less than 0.3 % (sweep 10 to 1000).
const PERTURB: f64 = 50.0;

/// How many times its tolerance a reduced cost of the wrong sign found at a factorisation has to
/// be beyond it before its variable, if it has two bounds, moves to the other rather than has its
/// cost shifted.
// Measured on an Apple M4 Max: 10 and 100 change the times of the same programmes, the boxed
// ones among them, by less than 0.3 % (sweep 10 to 1000).
const FLIP: f64 = 1000.0;

/// How many times the costs may be perturbed.
const PERTURBATIONS: usize = 10;

/// The bounds of a free variable in the subproblem of phase one.
const FREE_BOUND: f64 = 1000.0;

/// How the dual simplex method ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Outcome {
    /// The basic point is inside the bounds, to the method's tolerance, and the reduced costs
    /// are of the signs the bounds ask for under the perturbed costs (those of the costs as given
    /// are left to the primal method to check).
    Optimal,
    /// The basic variable of this basis position breaks a bound, and no variable out of the basis
    /// can bring it back: its row of `B⁻¹` is a ray of the dual programme along which the dual
    /// objective grows for ever, and so a certificate that no point is feasible.
    Infeasible(usize),
    /// Phase one could not make the reduced costs feasible: the dual programme has no feasible
    /// point, so the programme is infeasible or unbounded.
    DualInfeasible,
    /// The pivots allowed are used up.
    Limit,
    /// A basis turned out singular to working precision, or the solves disagreed.
    Failed,
}

/// Where the dual simplex method stopped: how, the basis and the places of the variables out of
/// it, and the pivots it took.
pub(super) struct Finish {
    pub(super) outcome: Outcome,
    pub(super) basis: Vec<usize>,
    pub(super) place: Vec<Place>,
    pub(super) pivots: usize,
}

/// A matrix compressed by columns, or by rows.
struct Compressed<T> {
    start: Vec<usize>,
    index: Vec<usize>,
    value: Vec<T>,
}

/// A variable the ratio test considers: its index, the magnitude of its element in the pivot
/// row, how far its reduced cost is inside the sign its place asks for, and its ratios, plain and
/// with Harris's tolerance.
#[derive(Debug, Clone, Copy)]
struct Candidate<T> {
    j: usize,
    a: T,
    room: T,
    ratio: T,
    harris: T,
}

/// The place of a variable as a small number: 0 in the basis, 1 at its lower bound, 2 at its
/// upper bound, 3 at zero and 4 fixed.
#[inline(always)]
fn code(place: Place) -> u8 {
    match place {
        Place::Basic => 0,
        Place::Lower => 1,
        Place::Upper => 2,
        Place::Zero => 3,
        Place::Fixed => 4,
    }
}

/// What a step of the method did.
enum Progress {
    /// No basic variable breaks a bound.
    Done,
    /// A change of basis.
    Pivoted,
    /// The basis was factorised afresh: the step is to be taken again.
    Again,
}

/// The kinds of bounds a variable has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Fixed,
    Boxed,
    LowerOnly,
    UpperOnly,
    Free,
}

struct Dual<'a, T> {
    m: usize,
    n: usize,
    a: &'a Matrix<T>,
    cols: Compressed<T>,
    rows: Compressed<T>,
    /// Whether rows of `B⁻¹·A` are computed by dense columns.
    dense_price: bool,
    /// `‖a_j‖²` of each column of `[A I]`.
    norm2: Vec<T>,
    /// `‖a_j‖₁` of each column of `[A I]`.
    norm1: Vec<T>,
    b: &'a [T],
    /// The right-hand sides of the phase: zero in phase one, `b` in phase two.
    rhs: Vec<T>,
    /// The costs as given.
    original: &'a [T],
    /// The costs worked with: perturbed and shifted.
    cost: Vec<T>,
    /// The bounds as the method takes them: an infinite one for a bound beyond `1/tol`, which
    /// stands for one.
    lo: Vec<T>,
    up: Vec<T>,
    /// The bounds of the phase.
    lower: Vec<T>,
    upper: Vec<T>,
    x: Vec<T>,
    d: Vec<T>,
    place: Vec<Place>,
    basis: Vec<usize>,
    /// The squared norms of the rows of `B⁻¹`, by basis position.
    weight: Vec<T>,
    /// The rounding a reduced cost carries, at the last factorisation: `γ` times the size of its
    /// terms, `|c_j| + Σ_i |a_ij·y_i|`.
    rounding: Vec<T>,
    /// How many times the costs have been perturbed.
    perturbations: usize,
    /// `|b_i| + Σ_j |a_ij·x_j|`, by row, at the last factorisation.
    activity: Vec<T>,
    factor: Factor<T>,
    tol: T,
    pivot_tol: T,
    /// How far, relative to the larger of one and itself, the pivot as the entering column gives
    /// it may differ from the pivot as the row gives it before the basis is factorised afresh:
    /// `10⁻⁶` in `f64`, ten times `√ε` in `f32`.
    agree: T,
    pivots: usize,
    max_iter: usize,
    degenerate: usize,
    seed: u64,
    // Working memory.
    rho: Vec<T>,
    row: Vec<T>,
    column: Vec<T>,
    tau: Vec<T>,
    flip: Vec<T>,
    candidates: Vec<Candidate<T>>,
    /// The variables whose elements of the pivot row may not be zero, when it was computed by the
    /// rows of `A` ([`listed`](Self::listed)): the structural variables of the rows its row of
    /// `B⁻¹` meets and the logical variables of those rows. The ratio test and the update of
    /// the reduced costs look at these alone.
    nonzero: Vec<usize>,
    /// Whether [`nonzero`](Self::nonzero) lists the pivot row's elements.
    listed: bool,
    /// For each structural variable, whether it is in [`nonzero`](Self::nonzero) yet.
    seen: Vec<bool>,
    flips: Vec<usize>,
    bstart: Vec<usize>,
    bindex: Vec<usize>,
    bvalue: Vec<T>,
}

/// Runs the dual simplex method on the working problem from the basis of the logical variables.
pub(super) fn solve<T: Real + Scalar>(
    a: &Matrix<T>,
    b: &[T],
    cost: &[T],
    lower: &[T],
    upper: &[T],
    tol: T,
    max_iter: usize,
) -> Finish {
    // Half the pivots allowed, so that the primal method has the other half to finish from where
    // the dual method stops, or to start again.
    let mut dual = Dual::new(a, b, cost, lower, upper, tol, max_iter - max_iter / 2);
    let outcome = dual.run();
    Finish {
        outcome,
        basis: dual.basis,
        place: dual.place,
        pivots: dual.pivots,
    }
}

impl<'a, T: Real + Scalar> Dual<'a, T> {
    fn new(
        a: &'a Matrix<T>,
        b: &'a [T],
        cost: &'a [T],
        lower: &[T],
        upper: &[T],
        tol: T,
        max_iter: usize,
    ) -> Self {
        let (m, n) = a.shape();
        // The matrix by columns and by rows.
        let mut cols = Compressed {
            start: Vec::with_capacity(n + 1),
            index: Vec::new(),
            value: Vec::new(),
        };
        cols.start.push(0);
        let mut count = vec![0usize; m + 1];
        let mut norm2 = Vec::with_capacity(n + m);
        let mut norm1 = Vec::with_capacity(n + m);
        for j in 0..n {
            let mut sum = T::ZERO;
            let mut size = T::ZERO;
            for (i, &v) in a.column(j).iter().enumerate() {
                if v != T::ZERO {
                    cols.index.push(i);
                    cols.value.push(v);
                    count[i + 1] += 1;
                    sum += v * v;
                    size += mag(v);
                }
            }
            cols.start.push(cols.index.len());
            norm2.push(sum);
            norm1.push(size);
        }
        norm2.resize(n + m, T::ONE);
        norm1.resize(n + m, T::ONE);
        for i in 0..m {
            count[i + 1] += count[i];
        }
        let nnz = cols.index.len();
        let mut rows = Compressed {
            start: count.clone(),
            index: vec![0; nnz],
            value: vec![T::ZERO; nnz],
        };
        for j in 0..n {
            for p in cols.start[j]..cols.start[j + 1] {
                let i = cols.index[p];
                let at = count[i];
                count[i] += 1;
                rows.index[at] = j;
                rows.value[at] = cols.value[p];
            }
        }
        let density = if m * n > 0 {
            nnz as f64 / (m * n) as f64
        } else {
            0.0
        };
        let limit = T::ONE / tol;
        let lo: Vec<T> = lower
            .iter()
            .map(|&v| if v < -limit { -T::INFINITY } else { v })
            .collect();
        let up: Vec<T> = upper
            .iter()
            .map(|&v| if v > limit { T::INFINITY } else { v })
            .collect();
        Self {
            m,
            n,
            a,
            cols,
            rows,
            dense_price: density >= DENSE_PRICE,
            norm2,
            norm1,
            b,
            rhs: b.to_vec(),
            original: cost,
            cost: cost.to_vec(),
            lower: lo.clone(),
            upper: up.clone(),
            lo,
            up,
            x: vec![T::ZERO; n + m],
            d: vec![T::ZERO; n + m],
            place: (0..n + m)
                .map(|j| if j < n { Place::Lower } else { Place::Basic })
                .collect(),
            basis: (n..n + m).collect(),
            weight: vec![T::ONE; m],
            rounding: vec![T::ZERO; n + m],
            perturbations: 0,
            activity: vec![T::ZERO; m],
            factor: Factor::new(),
            tol,
            pivot_tol: Float::sqrt(T::EPSILON),
            agree: T::from_f64(1e-6).max(T::from_f64(10.0) * Float::sqrt(T::EPSILON)),
            pivots: 0,
            max_iter,
            degenerate: 0,
            seed: 0x9E37_79B9_7F4A_7C15,
            rho: vec![T::ZERO; m],
            row: vec![T::ZERO; n + m],
            column: vec![T::ZERO; m],
            tau: vec![T::ZERO; m],
            flip: vec![T::ZERO; m],
            candidates: Vec::new(),
            nonzero: Vec::new(),
            listed: false,
            seen: vec![false; n],
            flips: Vec::new(),
            bstart: Vec::new(),
            bindex: Vec::new(),
            bvalue: Vec::new(),
        }
    }

    /// The kind of the bounds of variable `j` in the phase.
    fn kind(&self, j: usize) -> Kind {
        let (l, u) = (self.lower[j], self.upper[j]);
        match (finite(l), finite(u)) {
            (true, true) if l == u => Kind::Fixed,
            (true, true) => Kind::Boxed,
            (true, false) => Kind::LowerOnly,
            (false, true) => Kind::UpperOnly,
            (false, false) => Kind::Free,
        }
    }

    /// The tolerance on the bound `bound` of variable `j`: half the primal method's, relative to
    /// the larger of one, the bound and, for a logical variable, the size of its row's terms.
    fn primal_tol(&self, j: usize, bound: T) -> T {
        let scale = mag(bound).max(T::ONE);
        let scale = if j < self.n {
            scale
        } else {
            scale.max(self.activity[j - self.n])
        };
        T::HALF * self.tol * scale
    }

    /// The tolerance on the reduced cost of variable `j`: half the primal method's, relative to
    /// the two numbers it is the difference of, `|c_j|` and `|a_jᵀ·y| = |c_j − d_j|`, and the
    /// rounding of its terms.
    fn dual_tol(&self, j: usize) -> T {
        let c = self.cost[j];
        T::HALF * self.tol * (mag(c) + mag(c - self.d[j])) + self.rounding[j]
    }

    /// How far the reduced cost of variable `j`, out of the basis at its place, is inside the
    /// sign its place asks for: negative when it is outside.
    fn room(&self, j: usize) -> T {
        let d = self.d[j];
        match self.place[j] {
            Place::Lower => d,
            Place::Upper => -d,
            Place::Zero => -mag(d),
            Place::Fixed | Place::Basic => T::INFINITY,
        }
    }

    /// The value a variable out of the basis has at its place.
    fn value_at(&self, j: usize) -> T {
        match self.place[j] {
            Place::Lower | Place::Fixed => self.lower[j],
            Place::Upper => self.upper[j],
            Place::Zero | Place::Basic => T::ZERO,
        }
    }

    /// The place, out of the basis, that the sign of its reduced cost asks for, in the phase.
    fn place_by_sign(&self, j: usize) -> Place {
        match self.kind(j) {
            Kind::Fixed => Place::Fixed,
            Kind::Boxed => {
                let (d, l, u) = (self.d[j], self.lower[j], self.upper[j]);
                if d > T::ZERO {
                    Place::Lower
                } else if d < T::ZERO {
                    Place::Upper
                } else if l < T::ZERO && u > T::ZERO {
                    // Either bound would do: zero, as near the origin as the primal method
                    // starts, keeps a large bound out of the arithmetic.
                    Place::Zero
                } else if mag(l) <= mag(u) {
                    Place::Lower
                } else {
                    Place::Upper
                }
            }
            Kind::LowerOnly => Place::Lower,
            Kind::UpperOnly => Place::Upper,
            Kind::Free => Place::Zero,
        }
    }

    /// Whether some variable out of the basis has a reduced cost of the wrong sign for its place
    /// by more than its tolerance.
    fn dual_infeasible(&self) -> bool {
        (0..self.n + self.m)
            .any(|j| self.place[j] != Place::Basic && self.room(j) < -self.dual_tol(j))
    }

    /// The whole method: phase one where it is needed, then phase two.
    fn run(&mut self) -> Outcome {
        // The basis of the logical variables, and the reduced costs it gives, from which the
        // variables out of it take their places.
        if self.factorize().is_err() {
            return Outcome::Failed;
        }
        self.compute_duals();
        for j in 0..self.n + self.m {
            if self.place[j] != Place::Basic {
                self.place[j] = self.place_by_sign(j);
            }
        }
        if self.dual_infeasible() {
            // Phase one: the subproblem of the least sum of dual infeasibilities.
            for j in 0..self.n + self.m {
                let (l, u) = match self.kind(j) {
                    Kind::Fixed | Kind::Boxed => (T::ZERO, T::ZERO),
                    Kind::LowerOnly => (T::ZERO, T::ONE),
                    Kind::UpperOnly => (-T::ONE, T::ZERO),
                    Kind::Free => (-T::from_f64(FREE_BOUND), T::from_f64(FREE_BOUND)),
                };
                self.lower[j] = l;
                self.upper[j] = u;
            }
            self.rhs.fill(T::ZERO);
            for j in 0..self.n + self.m {
                if self.place[j] != Place::Basic {
                    self.place[j] = self.place_by_sign(j);
                }
            }
            self.compute_primals();
            let ended = self.phase();
            // Back to the programme's bounds and right-hand sides, the variables out of the basis
            // at the places their reduced costs ask for there, however phase one ended.
            self.lower.copy_from_slice(&self.lo);
            self.upper.copy_from_slice(&self.up);
            self.rhs.copy_from_slice(self.b);
            for j in 0..self.n + self.m {
                if self.place[j] != Place::Basic {
                    self.place[j] = self.place_by_sign(j);
                }
            }
            match ended {
                Ok(()) => {}
                // The subproblem always has a feasible point, zero.
                Err(Outcome::Infeasible(_)) => return Outcome::Failed,
                Err(other) => return other,
            }
            if self.dual_infeasible() {
                return Outcome::DualInfeasible;
            }
        }
        self.compute_primals();
        if let Err(outcome) = self.phase() {
            return outcome;
        }
        // The costs as given, and the reduced costs they give.
        if self.perturbations > 0 || self.cost != self.original {
            self.cost.copy_from_slice(self.original);
            self.compute_duals();
        }
        Outcome::Optimal
    }

    /// The method under the bounds, right-hand sides and costs of the phase, until no basic
    /// variable breaks a bound on a basis just factorised.
    fn phase(&mut self) -> Result<(), Outcome> {
        loop {
            match self.step()? {
                Progress::Pivoted | Progress::Again => {}
                Progress::Done => {
                    if self.factor.updates() == 0 {
                        return Ok(());
                    }
                    self.refactor().map_err(|()| Outcome::Failed)?;
                }
            }
        }
    }

    /// Factorises the basis afresh and computes the basic variables and the reduced costs again,
    /// putting right the reduced costs that came out of the wrong sign.
    fn refactor(&mut self) -> Result<(), ()> {
        self.factorize()?;
        self.compute_primals();
        self.compute_duals();
        // A reduced cost of the wrong sign: a variable with two bounds goes to the other when it
        // is well beyond its tolerance, any other has its cost shifted so that its reduced cost
        // is zero.
        let mut flipped = false;
        for j in 0..self.n + self.m {
            if self.place[j] == Place::Basic {
                continue;
            }
            let (room, tol) = (self.room(j), self.dual_tol(j));
            if room >= -tol {
                continue;
            }
            if self.kind(j) == Kind::Boxed && room < -T::from_f64(FLIP) * tol {
                self.place[j] = if self.d[j] >= T::ZERO {
                    Place::Lower
                } else {
                    Place::Upper
                };
                flipped = true;
            } else {
                self.cost[j] -= self.d[j];
                self.d[j] = T::ZERO;
            }
        }
        if flipped {
            self.compute_primals();
        }
        Ok(())
    }

    /// Factorises the basis afresh.
    fn factorize(&mut self) -> Result<(), ()> {
        let (m, n) = (self.m, self.n);
        self.bstart.clear();
        self.bindex.clear();
        self.bvalue.clear();
        self.bstart.push(0);
        for &j in &self.basis {
            if j < n {
                let (s, e) = (self.cols.start[j], self.cols.start[j + 1]);
                self.bindex.extend_from_slice(&self.cols.index[s..e]);
                self.bvalue.extend_from_slice(&self.cols.value[s..e]);
            } else {
                self.bindex.push(j - n);
                self.bvalue.push(T::ONE);
            }
            self.bstart.push(self.bindex.len());
        }
        self.factor
            .factor(m, &self.bstart, &self.bindex, &self.bvalue)
            .map_err(|_| ())
    }

    /// The basic variables from the data, `x_B = B⁻¹·(rhs − N·x_N)`, and the sizes of the rows'
    /// terms there.
    fn compute_primals(&mut self) {
        let (m, n) = (self.m, self.n);
        let mut r = core::mem::take(&mut self.column);
        r.copy_from_slice(&self.rhs);
        for j in 0..n + m {
            if self.place[j] == Place::Basic {
                continue;
            }
            let v = self.value_at(j);
            self.x[j] = v;
            if v != T::ZERO {
                if j < n {
                    for p in self.cols.start[j]..self.cols.start[j + 1] {
                        r[self.cols.index[p]] -= self.cols.value[p] * v;
                    }
                } else {
                    r[j - n] -= v;
                }
            }
        }
        self.factor.ftran(&mut r);
        for (k, &j) in self.basis.iter().enumerate() {
            self.x[j] = r[k];
        }
        let activity = &mut self.activity;
        for (s, v) in activity.iter_mut().zip(&self.rhs) {
            *s = mag(*v);
        }
        for j in 0..n {
            let v = self.x[j];
            if v != T::ZERO {
                for p in self.cols.start[j]..self.cols.start[j + 1] {
                    activity[self.cols.index[p]] += mag(self.cols.value[p] * v);
                }
            }
        }
        self.column = r;
    }

    /// The multipliers `y = B⁻ᵀ·c_B` and the reduced costs `d = c − [A I]ᵀ·y` of the variables
    /// out of the basis; zero for the basic ones.
    fn compute_duals(&mut self) {
        let (m, n) = (self.m, self.n);
        let mut y = core::mem::take(&mut self.rho);
        for (v, &j) in y.iter_mut().zip(&self.basis) {
            *v = self.cost[j];
        }
        self.factor.btran(&mut y);
        let mut row = core::mem::take(&mut self.row);
        self.price(&y, &mut row);
        for j in 0..n + m {
            self.d[j] = if self.place[j] == Place::Basic {
                T::ZERO
            } else {
                self.cost[j] - row[j]
            };
        }
        // The rounding, against `|c_j| + ‖a_j‖₁·‖y‖∞`: a multiplier that is small by
        // cancellation carries the rounding of the larger ones it came from.
        let gamma = T::from_f64(4.0 * (m + 2) as f64) * T::EPSILON;
        let largest = y.iter().fold(T::ZERO, |big, v| big.max(mag(*v)));
        for j in 0..n + m {
            self.rounding[j] = gamma * (mag(self.cost[j]) + self.norm1[j] * largest);
        }
        self.row = row;
        self.rho = y;
    }

    /// `out_j = a_jᵀ·v` for every variable out of the basis, `a_j` the column of `[A I]`; the
    /// elements of basic variables are left as they come.
    fn price(&self, v: &[T], out: &mut [T]) {
        let (m, n) = (self.m, self.n);
        out[n..n + m].copy_from_slice(v);
        if !self.dense_price {
            let nonzero = v.iter().filter(|x| **x != T::ZERO).count();
            if (nonzero as f64) < ROW_PRICE * m as f64 {
                out[..n].fill(T::ZERO);
                for (i, &vi) in v.iter().enumerate() {
                    if vi != T::ZERO {
                        for p in self.rows.start[i]..self.rows.start[i + 1] {
                            out[self.rows.index[p]] += self.rows.value[p] * vi;
                        }
                    }
                }
                return;
            }
        }
        self.price_by_columns(v, out);
    }

    /// The pivot row `out_j = a_jᵀ·ρ`, as [`price`](Self::price) computes it, listing in
    /// [`nonzero`](Self::nonzero) the variables whose elements may not be zero when it goes by
    /// the rows of `A`; [`listed`](Self::listed) says whether it did.
    fn price_row(&mut self, v: &[T], out: &mut [T]) {
        let (m, n) = (self.m, self.n);
        self.listed = false;
        if !self.dense_price {
            let nonzero = v.iter().filter(|x| **x != T::ZERO).count();
            if (nonzero as f64) < ROW_PRICE * m as f64 {
                self.listed = true;
                self.nonzero.clear();
                for (i, &vi) in v.iter().enumerate() {
                    if vi == T::ZERO {
                        continue;
                    }
                    out[n + i] = vi;
                    self.nonzero.push(n + i);
                    for p in self.rows.start[i]..self.rows.start[i + 1] {
                        let j = self.rows.index[p];
                        if !self.seen[j] {
                            self.seen[j] = true;
                            self.nonzero.push(j);
                            out[j] = T::ZERO;
                        }
                        out[j] += self.rows.value[p] * vi;
                    }
                }
                for &j in &self.nonzero {
                    if j < n {
                        self.seen[j] = false;
                    }
                }
                return;
            }
        }
        out[n..n + m].copy_from_slice(v);
        self.price_by_columns(v, out);
    }

    /// The structural part of [`price`](Self::price) by columns: dense ones, or compressed.
    fn price_by_columns(&self, v: &[T], out: &mut [T]) {
        let n = self.n;
        if self.dense_price {
            for (j, o) in out[..n].iter_mut().enumerate() {
                if self.place[j] != Place::Basic {
                    *o = dot(self.a.column(j), v);
                }
            }
        } else {
            for (j, o) in out[..n].iter_mut().enumerate() {
                if self.place[j] != Place::Basic {
                    let (s, e) = (self.cols.start[j], self.cols.start[j + 1]);
                    *o = dot_gather(&self.cols.value[s..e], &self.cols.index[s..e], v);
                }
            }
        }
    }

    /// Adds `scale` times column `j` of `[A I]` to `v`.
    fn add_column(&self, j: usize, scale: T, v: &mut [T]) {
        if j < self.n {
            for p in self.cols.start[j]..self.cols.start[j + 1] {
                v[self.cols.index[p]] += self.cols.value[p] * scale;
            }
        } else {
            v[j - self.n] += scale;
        }
    }

    /// The basis position to leave the basis, by dual steepest edge: of the basic variables that
    /// break a bound by more than its tolerance, the one of the largest squared excess against the
    /// weight of its position.
    fn choose_row(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut merit = T::ZERO;
        for (k, &j) in self.basis.iter().enumerate() {
            let x = self.x[j];
            let (l, u) = (self.lower[j], self.upper[j]);
            let excess = if x < l - self.primal_tol(j, l) {
                l - x
            } else if x > u + self.primal_tol(j, u) {
                x - u
            } else {
                continue;
            };
            let value = excess * excess / self.weight[k];
            if value > merit {
                merit = value;
                best = Some(k);
            }
        }
        best
    }

    /// The bound-flipping ratio test with Harris's tolerance, for the row of basis position `r`
    /// whose variable is to go to `target`, `s` being `−1` when it goes up to a lower bound and
    /// `1` when it goes down to an upper one, and `slope` is how far it is from it. Fills `flips`
    /// with the variables that move to their other bound and returns the entering variable;
    /// `None` when nothing stops the dual ray, the variables that move to their other bounds along
    /// it in `flips`. A group of variables is passed only when the leaving variable is still
    /// beyond its bound by more than `margin`, its tolerance, once they have moved.
    fn choose_column(&mut self, s: T, slope: T, margin: T) -> Option<usize> {
        let mut candidates = core::mem::take(&mut self.candidates);
        candidates.clear();
        // One sweep: the largest element of the row among the variables that can move, for the
        // pivot tolerance, and those whose reduced costs the step drives towards the wrong sign,
        // with how far each is from it.
        let mut largest = T::ONE;
        let list = core::mem::take(&mut self.nonzero);
        let count = if self.listed {
            list.len()
        } else {
            self.n + self.m
        };
        // Without branches on the data, which a pivot row's signs and the basis's places make
        // unpredictable: every variable is written, and kept when it is a candidate.
        let dummy = Candidate {
            j: 0,
            a: T::ZERO,
            room: T::ZERO,
            ratio: T::ZERO,
            harris: T::ZERO,
        };
        candidates.resize(count, dummy);
        let mut kept = 0;
        {
            let this = &*self;
            let mut visit = |j: usize| {
                let code = code(this.place[j]);
                let (lower, upper, zero) = (code == 1, code == 2, code == 3);
                let a = this.row[j];
                let size = mag(a);
                largest = largest.max(if lower | upper | zero { size } else { T::ZERO });
                let sa = s * a;
                let (up, down) = (sa > T::ZERO, sa < T::ZERO);
                let eligible = (lower & up) | (upper & down) | (zero & (up | down));
                let d = this.d[j];
                let room = if lower | (zero & up) { d } else { -d };
                candidates[kept] = Candidate {
                    j,
                    a: size,
                    room,
                    ratio: T::ZERO,
                    harris: T::ZERO,
                };
                kept += usize::from(eligible);
            };
            if this.listed {
                list.iter().for_each(|&j| visit(j));
            } else {
                (0..count).for_each(visit);
            }
        }
        candidates.truncate(kept);
        self.nonzero = list;
        // The elements too small to pivot on go; the others get their ratios, as they are and
        // with Harris's tolerance.
        let small = self.pivot_tol * largest;
        let mut kept = 0;
        for i in 0..candidates.len() {
            let c = candidates[i];
            if c.a <= small {
                continue;
            }
            candidates[kept] = Candidate {
                ratio: c.room.max(T::ZERO) / c.a,
                harris: (c.room + self.dual_tol(c.j)).max(T::ZERO) / c.a,
                ..c
            };
            kept += 1;
        }
        candidates.truncate(kept);
        self.flips.clear();
        let mut slope = slope;
        let entering = loop {
            if candidates.is_empty() {
                break None;
            }
            // Harris's bound on the step: no reduced cost beyond its tolerance.
            let bound = candidates.iter().fold(T::INFINITY, |b, c| b.min(c.harris));
            // The group within it: pass it whole when every member has two bounds and the
            // leaving variable is still beyond its bound once they have moved.
            let mut decrease = T::ZERO;
            let mut boxed = true;
            let mut best: Option<(usize, T)> = None;
            for c in &candidates {
                if c.ratio > bound {
                    continue;
                }
                // A variable at zero inside its bounds enters rather than moves to a bound.
                if self.kind(c.j) == Kind::Boxed && self.place[c.j] != Place::Zero {
                    decrease += c.a * (self.upper[c.j] - self.lower[c.j]);
                } else {
                    boxed = false;
                }
                if best.is_none_or(|(_, b)| c.a > b) {
                    best = Some((c.j, c.a));
                }
            }
            if boxed && slope > decrease + margin {
                slope -= decrease;
                let flips = &mut self.flips;
                candidates.retain(|c| {
                    let within = c.ratio <= bound;
                    if within {
                        flips.push(c.j);
                    }
                    !within
                });
                continue;
            }
            break best.map(|(j, _)| j);
        };
        self.candidates = candidates;
        entering
    }

    /// Moves the variables of `flips` to their other bounds, and the basic variables with them.
    fn apply_flips(&mut self) {
        if self.flips.is_empty() {
            return;
        }
        let mut f = core::mem::take(&mut self.flip);
        self.flip_vector(&mut f);
        self.factor.ftran(&mut f);
        self.commit_flips(&f);
        self.flip = f;
    }

    /// Writes `Σ_j a_j·Δ_j` over the variables of `flips` to `f`, `Δ_j` the move of `x_j` to its
    /// other bound.
    fn flip_vector(&self, f: &mut [T]) {
        f.fill(T::ZERO);
        for &j in &self.flips {
            let delta = self.upper[j] - self.lower[j];
            let delta = if self.place[j] == Place::Lower {
                delta
            } else {
                -delta
            };
            self.add_column(j, delta, f);
        }
    }

    /// Moves the variables of `flips` to their other bounds, `f` being `B⁻¹·Σ_j a_j·Δ_j`, what the
    /// basic variables lose by it.
    fn commit_flips(&mut self, f: &[T]) {
        for &j in &self.flips {
            let (l, u) = (self.lower[j], self.upper[j]);
            if self.place[j] == Place::Lower {
                self.place[j] = Place::Upper;
                self.x[j] = u;
            } else {
                self.place[j] = Place::Lower;
                self.x[j] = l;
            }
        }
        for (k, &j) in self.basis.iter().enumerate() {
            self.x[j] -= f[k];
        }
    }

    /// One step of the method.
    fn step(&mut self) -> Result<Progress, Outcome> {
        let (m, n) = (self.m, self.n);
        let Some(r) = self.choose_row() else {
            return Ok(Progress::Done);
        };
        if self.pivots >= self.max_iter {
            return Err(Outcome::Limit);
        }
        let leaving = self.basis[r];
        let xr = self.x[leaving];
        let (s, target) = if xr < self.lower[leaving] {
            (-T::ONE, self.lower[leaving])
        } else {
            (T::ONE, self.upper[leaving])
        };
        // The row of `B⁻¹` and its row of `B⁻¹·[A I]`.
        let mut rho = core::mem::take(&mut self.rho);
        rho.fill(T::ZERO);
        rho[r] = T::ONE;
        self.factor.btran(&mut rho);
        let mut row = core::mem::take(&mut self.row);
        self.price_row(&rho, &mut row);
        self.row = row;
        let margin = self.primal_tol(leaving, target);
        let Some(q) = self.choose_column(s, mag(xr - target), margin) else {
            self.rho = rho;
            self.apply_flips();
            return Err(Outcome::Infeasible(r));
        };
        // The entering column as the basis sees it, what the moves to the other bounds do to the
        // basic variables, and `τ = B⁻¹·ρ` for the weights, in one pass over the factors.
        let mut column = core::mem::take(&mut self.column);
        column.fill(T::ZERO);
        self.add_column(q, T::ONE, &mut column);
        let mut tau = core::mem::take(&mut self.tau);
        tau.copy_from_slice(&rho);
        let mut f = core::mem::take(&mut self.flip);
        let flipping = !self.flips.is_empty();
        if flipping {
            self.flip_vector(&mut f);
            self.factor.ftran_many(&mut [&mut column, &mut tau, &mut f]);
        } else {
            self.factor.ftran_many(&mut [&mut column, &mut tau]);
        }
        // A check that the column agrees with the row.
        let (from_column, from_row) = (column[r], self.row[q]);
        let disagree = mag(from_column - from_row) > self.agree * mag(from_column).max(T::ONE);
        if disagree || mag(from_column) <= T::ZERO {
            self.column = column;
            self.tau = tau;
            self.flip = f;
            self.rho = rho;
            if self.factor.updates() > 0 {
                self.refactor().map_err(|()| Outcome::Failed)?;
                return Ok(Progress::Again);
            }
            return Err(Outcome::Failed);
        }
        if flipping {
            self.commit_flips(&f);
        }
        self.flip = f;
        // The squared norm of the row of `B⁻¹`, for the weights.
        let mut norm = T::ZERO;
        for v in &rho {
            norm += *v * *v;
        }
        // The primal step.
        let pivot = column[r];
        let theta_p = (self.x[leaving] - target) / pivot;
        if theta_p != T::ZERO {
            for (k, &j) in self.basis.iter().enumerate() {
                self.x[j] -= theta_p * column[k];
            }
        }
        self.x[q] += theta_p;
        self.x[leaving] = target;
        // The dual step. A reduced cost of the entering variable of the wrong sign, which
        // Harris's tolerance lets through, is made zero by a shift of its cost, so that the step
        // never goes backwards.
        let room = match self.place[q] {
            Place::Lower => self.d[q],
            Place::Upper => -self.d[q],
            _ => T::ZERO,
        };
        if room < T::ZERO {
            self.cost[q] -= self.d[q];
            self.d[q] = T::ZERO;
        }
        let theta_d = self.d[q] / self.row[q];
        if theta_d != T::ZERO {
            if self.listed {
                for &j in &self.nonzero {
                    if self.place[j] != Place::Basic {
                        self.d[j] -= theta_d * self.row[j];
                    }
                }
            } else {
                for j in 0..n + m {
                    if self.place[j] != Place::Basic {
                        self.d[j] -= theta_d * self.row[j];
                    }
                }
            }
        }
        self.d[q] = T::ZERO;
        self.d[leaving] = -theta_d;
        // The weights.
        let wr = norm;
        for k in 0..m {
            if k == r {
                continue;
            }
            let ratio = column[k] / pivot;
            if ratio == T::ZERO {
                continue;
            }
            let w = self.weight[k] + ratio * (ratio * wr - T::TWO * tau[k]);
            let floor = ratio * ratio / self.norm2[leaving];
            self.weight[k] = w.max(floor);
        }
        self.weight[r] = (wr / (pivot * pivot)).max(T::MIN_POSITIVE);
        // The change of basis.
        self.basis[r] = q;
        self.place[q] = Place::Basic;
        self.place[leaving] = if self.lower[leaving] == self.upper[leaving] {
            Place::Fixed
        } else if target == self.lower[leaving] {
            Place::Lower
        } else {
            Place::Upper
        };
        self.pivots += 1;
        self.rho = rho;
        self.tau = tau;
        let filled = self.factor.eta_elements() as f64
            > ETA_FILL * self.factor.elements().max(self.m) as f64;
        if self.factor.updates() + 1 >= REFACTOR || filled {
            self.column = column;
            self.refactor().map_err(|()| Outcome::Failed)?;
        } else {
            self.factor.update(r, &column);
            self.column = column;
        }
        // Degeneracy: a step that leaves the dual objective where it was.
        if mag(theta_d) <= self.tol {
            self.degenerate += 1;
            if self.degenerate >= DEGENERATE && self.perturbations < PERTURBATIONS {
                self.perturb();
                self.degenerate = 0;
            }
        } else {
            self.degenerate = 0;
        }
        Ok(Progress::Pivoted)
    }

    /// Perturbs the cost of every variable out of the basis by a small random amount towards the
    /// sign the programme's bounds ask of its reduced cost: up for one with only a lower bound,
    /// down for one with only an upper bound, and for one with two the way its place asks. A free
    /// variable, whose reduced cost has to be zero, and a fixed one are left alone; so, in phase
    /// one, is one with two bounds, which the subproblem fixes.
    fn perturb(&mut self) {
        self.perturbations += 1;
        for j in 0..self.n + self.m {
            let (lo, up) = (finite(self.lo[j]), finite(self.up[j]));
            let sign = match (self.place[j], lo, up) {
                (Place::Basic | Place::Fixed | Place::Zero, _, _) => continue,
                (_, true, false) => T::ONE,
                (_, false, true) => -T::ONE,
                (Place::Lower, true, true) if self.lo[j] != self.up[j] => T::ONE,
                (Place::Upper, true, true) if self.lo[j] != self.up[j] => -T::ONE,
                _ => continue,
            };
            // SplitMix64.
            self.seed = self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 11) as f64 / (1u64 << 53) as f64;
            let size =
                T::from_f64(PERTURB * (1.0 + u)) * self.tol * (T::ONE + mag(self.original[j]));
            self.cost[j] += sign * size;
            self.d[j] += sign * size;
        }
    }
}
