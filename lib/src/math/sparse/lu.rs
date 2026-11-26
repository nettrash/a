//! The sparse LU decomposition with partial pivoting.
//!
//! [`SparseLu`] factors a square sparse matrix as `P·A·Q = L·U`. `Q` is a permutation of the
//! columns, taken beforehand from the pattern alone so that the factors fill in little; `P` is the
//! permutation of the rows that partial pivoting chooses while the elimination runs; `L` is lower
//! triangular with ones on its diagonal, and `U` is upper triangular.
//!
//! # The method
//!
//! The elimination is the left-looking one of Gilbert and Peierls, *Sparse partial pivoting in
//! time proportional to arithmetic operations* (SIAM J. Sci. Stat. Comput. 9, 1988). Step `j`
//! takes column `q[j]` of `A` and solves a triangular system with the columns of `L` found so
//! far. The part of the solution that lies in the rows already pivotal is column `j` of `U`; the
//! rest, divided by the pivot chosen among it, is column `j` of `L`.
//!
//! What makes this a sparse method is that the solve never visits a row the solution is zero in.
//! A row of the solution can be nonzero only if it is a nonzero of the column of `A`, or if `L`
//! has an element in that row in a column whose own pivotal row is nonzero. So the nonzeros are
//! the rows *reachable* from those of the column of `A` in the graph that joins the pivotal row
//! of every column of `L` to each row of that column. A depth-first search finds them, and the
//! reverse of the order in which it finishes them is one in which every row comes after the rows
//! it depends on. The arithmetic runs over that list alone, so a step costs what its
//! floating-point operations cost, and the whole factorisation takes time proportional to the
//! operations plus the order and the nonzeros of `A`, never to the square of the order.
//!
//! The search is shortened by the symmetric pruning of Eisenstat and Liu, *Exploiting structural
//! symmetry in a sparse partial pivoting code* (SIAM J. Sci. Comput. 14, 1993). Once `U` has an
//! element `(k, j)` and `L` one in the pivotal row of step `j` in column `k`, every row of column
//! `k` of `L` that is not pivotal yet is in column `j` of `L` too, so the search can reach it
//! through `j` and need not follow it from `k`. Column `k` is then rearranged with its pivotal
//! rows first, and the search reads no further than those. The rows reached are the same, and
//! so are the factors; only the searching is less.
//!
//! # Pivoting
//!
//! The pivot of a column is chosen among the rows that are not pivotal yet, by threshold partial
//! pivoting. The size of an element is `|re| + |im|`, as in the dense
//! [`Lu`](crate::math::matrix::Lu) and in LAPACK. The diagonal element — the one in row `q[j]`
//! of `A` — is kept when it is nonzero and at least [`pivot_threshold`] times the largest; else
//! the largest is taken, and of several of that size the one in the lowest row of `A`. With the
//! threshold of 1, the default, this is partial pivoting with ties going to the diagonal, and no
//! element of `L` is larger than 1 in absolute value — than √2 in a complex matrix, whose sizes
//! are within that factor of the absolute values. A smaller threshold lets `L` have elements up
//! to its reciprocal (times √2), and in exchange keeps the pivots on the diagonal, where a
//! fill-reducing ordering of a matrix with a symmetric pattern expects them.
//!
//! # Strategies
//!
//! [`SparseLu::new`] chooses the ordering and the pivoting by the pattern, by the rule UMFPACK
//! uses to choose between its symmetric and unsymmetric strategies (T. A. Davis, *Algorithm 832:
//! UMFPACK V4.3, an unsymmetric-pattern multifrontal method*, ACM Trans. Math. Softw. 30, 2004).
//! A pattern that is nearly symmetric — at least half of the elements off the diagonal have their
//! mirror image stored, where UMFPACK 6.3 (SuiteSparse 7.14) asks three tenths
//! (`STRATEGY_THRESH_SYM`, one half in its earlier versions) — with at least nine tenths of its
//! diagonal stored (`STRATEGY_THRESH_NNZDIAG`) takes the *symmetric*
//! strategy: the columns in the order [`amd`](super::amd) gives the pattern of `A + Aᵀ`, and the
//! pivots chosen to keep the diagonal that order expects. The diagonal element is the pivot when
//! it is nonzero and at least 10⁻⁴ of the largest element of its column (a tenth of UMFPACK's
//! `SYM_PIVOT_TOLERANCE`, 0.001); otherwise the pivot is an element of at least 0.1 of the largest
//! (UMFPACK's `PIVOT_TOLERANCE`): of those the largest in a row whose own column is eliminated
//! already, if there is one, and the largest of all if not. A row whose column is eliminated is
//! one whose diagonal an earlier step passed over, which no later column can take on its
//! diagonal, so taking it leaves the later diagonals in place; the row of the column that took
//! this column's diagonal is one, and a pivot there is what UMFPACK makes by taking the mirror
//! of an off-diagonal pivot as the next diagonal, its likeness of a pivot of two rows and two
//! columns. UMFPACK chooses among its acceptable rows by their sparsity, which a left-looking
//! elimination does not know, and it refines the solutions (two steps by default), which this
//! does not. Other patterns take the *unsymmetric* strategy: [`Ordering::Colamd`] and partial
//! pivoting.
//!
//! The order of the symmetric strategy bounds the fill only while the pivots stay on the diagonal,
//! and diagonal pivots of a ten-thousandth of their column make multipliers of up to ten thousand.
//! So [`SparseLu::new`] watches its symmetric elimination: after every step it compares the
//! elements of the factors so far with what the elimination with every pivot on the diagonal makes
//! in as many steps — the pattern of the Cholesky factor of `A + Aᵀ` in that order, counted by the
//! elimination tree and the row and column counts of Gilbert, Ng and Peyton (*An efficient
//! algorithm to compute row and column counts for sparse Cholesky factorization*, SIAM J. Matrix
//! Anal. Appl. 15, 1994) once a pivot has left the diagonal — and the largest element of `U` with
//! the largest of `A`. When the elements are more than twice the prediction and the elements of `A`
//! more, or an element of `U` is more than 16 times the largest of `A`, the elimination stops and
//! `A` is factored again, from the start, by the unsymmetric strategy: the result is then that of
//! [`LuStrategy::Unsymmetric`], element for element. So it is when the symmetric elimination ends
//! with factors that are singular — a pivot of zero, an element that is not finite, or a condition
//! beyond the bound of the test of singularity to working precision, which its growth raises —
//! unless the pattern alone makes the matrix singular: the verdict on such a matrix is that of
//! partial pivoting. The tests are functions of the pattern, the values and the order of the
//! operations alone, so a matrix always takes the same path. On grids whose diagonal is weak —
//! shifted Laplacians and Helmholtz operators, and `A − σI` for a shift inside the spectrum — the
//! symmetric elimination keeps its pivots on the diagonal and its fill, but grows by hundreds to
//! tens of thousands, and the refactorisation is what keeps the accuracy of partial pivoting; on
//! the matrices of grids and meshes with a strong diagonal the symmetric strategy runs to its end.
//!
//! # Singular matrices
//!
//! A column in which no row is left to choose, or none but zeros, has no pivot: the matrix is
//! singular. The elimination goes on with a pivot of zero, so that `P·A·Q = L·U` holds for a
//! singular matrix as well. A matrix with an element that is not finite, or one whose
//! elimination overflows, has no factors to trust.
//!
//! A matrix can also be singular by its pattern alone: when no `n` of its nonzeros lie in
//! distinct rows and columns, every product in the expansion of its determinant has a zero in
//! it, whatever the elements are. The elimination of such a matrix need not meet a pivot of zero
//! — rounding can leave one of the size of the rounding error instead — so the pattern is
//! checked as well. The pivots that are elements of `A` are already as many nonzeros in
//! distinct rows and columns, and as a rule they are all of them; the remaining columns are
//! matched to rows by the augmenting paths of Hopcroft and Karp, *An n^5/2 algorithm for maximum
//! matchings in bipartite graphs* (SIAM J. Comput. 2, 1973), and a matrix for which they find
//! no complete matching is reported singular.
//!
//! Most singular matrices never show a pivot of zero, though. Rounding makes every pivot a
//! little wrong, so the one that should be zero comes out of the size of the rounding error,
//! or as exactly zero, by the accident of the order of the columns: the integers 1 to 9 in rows
//! of three end with a pivot of about `10⁻¹⁶` in four of the six orders of their columns, and
//! with zero in the other two. A pivot that small cannot be told from zero, and a solution
//! divided by it is noise. What the factors can tell is how near the matrix is to a singular
//! one, and that is the reciprocal of its condition number: as LAPACK's `xGESVX` and SuperLU's
//! `xGSSVX` define it, a matrix whose reciprocal condition number is below `ε`, the machine
//! epsilon of its type, is *singular to working precision*, and a solution of it has no digit
//! that the bound on its error vouches for. The condition is taken of the matrix with its rows
//! and then its columns scaled to a largest absolute value of 1, as LAPACK's `xGEEQU` scales
//! them, so that rows of very different sizes, and a diagonal matrix such as that of `10⁻²⁰`
//! and 1, are not mistaken for a matrix near a singular one. One pass of that scaling undoes a
//! scaling of the rows; a scaling of the columns it undoes only in part (see the limits below).
//!
//! The condition is estimated, once, when the matrix is factored, by the method of
//! [`rcond`](SparseLu::rcond), from a few solves with the factors. The estimate is a lower bound
//! for the norm of the inverse, found by a search that starts from three vectors: one of equal
//! elements, one of alternating signs and one of pseudo-random signs and sizes, the same for
//! every matrix. It is made from the factors, so it depends on the order of the columns and on
//! the rounding, as the pivots do, though far less. The third start vector is there because the
//! first two can both be orthogonal to the left null vector of a singular matrix (that of a
//! matrix whose rows satisfy `r_i + r_{i+1} = r_{i+2} + r_{i+3}` is), and the search then sees
//! nothing of the singular direction.
//!
//! Two refinements make the test hold for the factors as they are computed. The bound `ε`
//! presumes that the factors are those of a matrix within about `ε` of `A`; by Wilkinson's
//! analysis of Gaussian elimination they are within about `ρ·ε`, `ρ` the
//! [growth](SparseLu::growth) of the elements, which partial pivoting keeps within a few units
//! on all but contrived matrices and a pivot threshold below 1 lets grow. So the bound is
//! `τ = ε·max(1, ρ)`. And the elimination pivots on the matrix as it is, and is backward stable
//! in its norm rather than in the norm of the scaled matrix: a small row eliminated against
//! large ones can leave an exactly singular matrix several `τ` from a singular one in the
//! scaled norm, while the unscaled norm shows it within `τ`. So a matrix is singular to working
//! precision when the smaller of its two reciprocal condition numbers, scaled and unscaled, is
//! below `τ`, unless the scaled one is 16 times `τ` or more, which says that the matrix is well
//! conditioned and merely badly scaled.
//!
//! [`is_singular`](SparseLu::is_singular) reports each of these, and the solves of a singular
//! matrix return `None`.
//!
//! # Supernodes
//!
//! The elimination begins a column at a time, which is the faster while the factors stay sparse.
//! In a matrix of order 64 or more, once a step has made 96 elements of `L` and `U` together in
//! its column (the pivot not counted), the steps after it are those of Demmel, Eisenstat,
//! Gilbert, Li and Liu, *A supernodal approach to sparse partial pivoting* (SIAM J. Matrix Anal.
//! Appl. 20, 1999): the columns of `L` that share their rows are held together as dense blocks
//! while it runs, the columns of the matrix are taken sixteen at a time, and most of the
//! arithmetic is that of dense triangular solves and products; the columns made before become
//! blocks of one column each. The search, its pruning and the choice of the pivots are those
//! above, made over the blocks instead of the columns, and a pivot is chosen from the whole
//! column once every update of it is in: the factors have the same elements in the same places
//! and are those of the same pivots, but for the rounding of the sums, which the blocks group
//! otherwise, and through it, rarely, the choice between two elements within rounding of each
//! other. A matrix of order below 64, and one whose factors never have a column of 96 elements —
//! a tridiagonal matrix of any order — is eliminated a column at a time to the end.
//!
//! # What is not done
//!
//! The matrix is not permuted to block triangular form first, its rows and columns are not
//! scaled for the elimination (only for the test of its condition), and a solution is not
//! refined. The elimination is sequential.
//!
//! The limits measured, of the test of singularity and of pivoting on unscaled rows and
//! columns among them, are listed in the documentation of [`SparseLu`].
//!
//! [`pivot_threshold`]: SparseLuOptions::pivot_threshold

use super::amd::amd_of;
use super::graph::Graph;
use super::{Ordering, SparseMatrix};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use watch::Watch;

mod supernodes;
mod watch;

/// The order from which the elimination can go by supernodes and panels ([`supernodes`]) rather
/// than a column at a time.
// Measured on an Apple M4 Max (`SparseLu::new`, orders 144 to 1728; sweep 64 to 2048): below
// order 97 no column reaches [`DENSE_COLUMN`] elements, so every value up to 97 is the same, and
// above it the small matrices that fill need the supernodes: a random one of order 200 took 1.28
// times as long at 256, one of order 500 1.73 times at 512, an elasticity mesh of order 648 1.84
// times at 1024.
const SUPERNODAL: usize = 64;

/// The elements of a column of `L` and `U` together from which the rest of the elimination goes
/// by supernodes: the column method is the faster while the factors stay this sparse.
// Measured on an Apple M4 Max (`SparseLu::new`; sweep 8 to 256 on 2-D and 3-D grids, elasticity
// meshes, circuits, random and tridiagonal matrices of orders 400 to 100 000, `f64` and
// `Complex<f64>`, both strategies): 96 against 32 takes 4 to 24 % off the 2-D grids and circuits of
// orders 400 to 2000, which the column method finishes the faster, and costs the 3-D grids and
// meshes of orders 600 to 2000 at most 1 %; from 128 on these lose 2 to 44 %. Orders from 10 000
// on move by less than 1 % between 8 and 128.
const DENSE_COLUMN: usize = 96;

/// The fraction of the largest element of its column down to which the symmetric strategy keeps
/// the diagonal element as the pivot: a tenth of UMFPACK's `SYM_PIVOT_TOLERANCE` (0.001, KLU's
/// default too).
// The owner's choice, with the growth watch (`watch::GROWTH`) at 16, from the quiet sweep on an
// Apple M4 Max (sweep 10⁻⁴ to 0.1, the watch at 64): `SparseLu::new` on the weak-diagonal 5-point
// grids of 100² and 200² (with and without convection) as fast from 10⁻⁴ to 10⁻³ (within 0.3 %),
// and up to 2 % slower at 0.003, 5 % at 0.01 and 31 % at 0.03 and 0.1, the pivots leaving the
// diagonal earlier and the elimination running further before the watch stops it; the
// scoreboard's convection–diffusion, upwind and circuit matrices the same at every value. On the
// 450 random systems of the review every value refuses master's 64; the nearly symmetric ones are
// refactored 116 times at 10⁻⁴, 104 at 10⁻³, 64 at 0.01 and 33 at 0.1, with the worst backward
// error per family 23 to 63, 34 to 197, 34 to 388 and 5 to 96 `ε·‖A‖∞·‖x‖∞`. Measured again with
// both of the owner's values: master's 64 refused, 135 refactored, worst backward errors 0.8 to
// 23 (master 0.8 to 14.5) and forward errors at most master's per family; `SparseLu::new` gives
// the decompositions of 10⁻³ and 64, bit for bit, on the 27 weak-diagonal grids of 50² to 200²
// (in 0.87 to 1.01 of their time, two alternated runs), on the scoreboard's matrices and on the
// shifted Laplacians of the shift-and-invert timings.
const SYMMETRIC_DIAGONAL: f64 = 1e-4;

/// The fraction of the largest element of its column that an element off the diagonal must reach
/// to be the pivot under the symmetric strategy: UMFPACK's `PIVOT_TOLERANCE`.
// Measured on an Apple M4 Max (the matrices of [`SYMMETRIC_DIAGONAL`]; sweep 0.01 to 1): the time
// is the same at every value (within 0.5 %); on the 450 systems every value refuses master's 64,
// and the worst backward error per nearly symmetric family is 103 `ε·‖A‖∞·‖x‖∞` at 0.01 and 0.03,
// 197 at 0.1, 397 at 0.3 and 486 at 1.
const SYMMETRIC_OFF_DIAGONAL: f64 = 0.1;

/// The step of a row that no column has taken as its pivot yet, and the row of a search that has
/// found none.
const NONE: usize = usize::MAX;

/// How often the estimate of [`SparseLu::rcond`] may move to a better unit vector, as in LAPACK.
const ESTIMATE_CYCLES: usize = 5;

/// The vectors the searches of the estimate of [`SparseLu::rcond`] start from.
const STARTS: usize = 3;

/// How far above the bound `τ` of the test of singularity to working precision the reciprocal
/// condition number of the scaled matrix may be while that of the matrix as it is still
/// decides. The elimination pivots on the unscaled matrix and is backward stable in its norm,
/// not always in the scaled one: a small row eliminated against large ones can leave an exactly
/// singular matrix as far as about ten `τ` from a singular one in the scaled norm, while the
/// unscaled norm still shows it within `τ` (seen in 44 of 400 000 random singular matrices of
/// review round 1, the farthest at 11·`τ`).
const UNSCALED_RANGE: f64 = 16.0;

/// Where the pseudo-random start vector of the estimate of the condition begins: any fixed value
/// would do, and a fixed one gives every matrix the same estimate, and the same verdict, each time
/// it is factored.
const RANDOM_START: u64 = 0x9E37_79B9_7F4A_7C15;

/// The next 64 bits of the SplitMix64 generator of Steele, Lea and Flood (*Fast splittable
/// pseudorandom number generators*, OOPSLA 2014), whose state is `state`.
fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// How [`SparseLu`] chooses the order of the columns and the pivot threshold: by the pattern of
/// the matrix, which is the default and what [`SparseLu::new`] does, by one of the two strategies
/// it chooses between, or by the [`SparseLuOptions`]' own ordering and threshold.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{LuStrategy, SparseLu, SparseLuOptions, SparseMatrix};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 5], &[-1.0; 4]);
/// let auto = SparseLu::with_options(&a, &SparseLuOptions::new()).unwrap();
/// assert_eq!(auto, SparseLu::new(&a));
/// // A symmetric pattern takes the symmetric strategy.
/// let symmetric = SparseLuOptions::new().with_strategy(LuStrategy::Symmetric);
/// assert_eq!(auto, SparseLu::with_options(&a, &symmetric).unwrap());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LuStrategy {
    /// [`Symmetric`](Self::Symmetric) when the pattern is nearly symmetric — at least half of
    /// the elements off the diagonal have their mirror image stored — and at least nine tenths of
    /// the diagonal is stored, [`Unsymmetric`](Self::Unsymmetric) otherwise: UMFPACK's rule,
    /// with the symmetry threshold of its earlier versions (UMFPACK 6.3 has three tenths). The
    /// symmetric elimination is watched, and when its factors hold more than twice the elements
    /// that its order predicts with every pivot on the diagonal, and as many as `A` more, or an
    /// element of `U` grows beyond 16 times the largest of `A`, it stops and the matrix is
    /// factored again from the start by the unsymmetric strategy; so it is when it ends with
    /// factors that are singular, unless the pattern alone made them so (see
    /// [`SparseLu::new`]). The default, and what [`SparseLu::new`] does; [`SparseLuOptions`]
    /// whose ordering or threshold is not its default factor as [`Explicit`](Self::Explicit)
    /// instead.
    #[default]
    Auto,
    /// The columns in the order [`amd`](super::amd) gives the pattern of `A + Aᵀ`, the diagonal
    /// element the pivot when it is nonzero and at least 10⁻⁴ of the largest element of its
    /// column (UMFPACK's symmetric strategy has 0.001), and otherwise an element of at least 0.1
    /// of the largest, the largest such in a row whose own column is eliminated already if there
    /// is one: multipliers up to 10 000 on the diagonal and 10 off it (`√2` times those for
    /// complex elements). Unlike [`Auto`](Self::Auto), it runs to its end whatever it fills and
    /// however much it grows; [`growth`](SparseLu::growth) and [`nnz`](SparseLu::nnz) say what
    /// it did.
    Symmetric,
    /// The columns by [`Ordering::Colamd`] and partial pivoting (the threshold 1): multipliers
    /// up to 1 (`√2`).
    Unsymmetric,
    /// The options' own [`ordering`](SparseLuOptions::ordering) and
    /// [`pivot_threshold`](SparseLuOptions::pivot_threshold), which the other strategies leave
    /// aside. [`SparseLuOptions::with_ordering`] and [`SparseLuOptions::with_pivot_threshold`]
    /// choose it.
    Explicit,
}

/// How [`SparseLu`] orders the columns of a matrix and how it chooses its pivots: a
/// [`LuStrategy`], and the ordering and threshold that [`LuStrategy::Explicit`] uses.
///
/// The strategy the options factor by is [`strategy`](Self::strategy), with one exception:
/// [`LuStrategy::Auto`] holds only while [`ordering`](Self::ordering) is [`Ordering::Colamd`]
/// and [`pivot_threshold`](Self::pivot_threshold) is 1, their defaults, and options whose
/// ordering or threshold is anything else factor as [`LuStrategy::Explicit`]. So options built
/// field by field, or by struct update syntax from [`SparseLuOptions::new`], get the ordering and
/// the threshold they name; a field set to its default cannot be told from one left alone, and
/// [`with_ordering`](Self::with_ordering) or [`with_pivot_threshold`](Self::with_pivot_threshold)
/// make the strategy `Explicit` whatever the value. [`LuStrategy::Symmetric`] and
/// [`LuStrategy::Unsymmetric`] leave both fields aside.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Ordering, SparseLu, SparseLuOptions, SparseMatrix};
///
/// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 3.0), (1, 1, 4.0)]);
/// // Partial pivoting takes the 3 of the first column, and swaps the rows.
/// let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
/// assert_eq!(lu.permutation(), [1, 0]);
/// // The diagonal 2 is at least half of it, so a threshold of one half keeps the rows in place.
/// let options = SparseLuOptions::new()
///     .with_ordering(Ordering::Natural)
///     .with_pivot_threshold(0.5);
/// let lu = SparseLu::with_options(&a, &options).unwrap();
/// assert_eq!(lu.permutation(), [0, 1]);
/// assert_eq!(lu.det(), 5.0);
/// // Options written field by field get the ordering they name, not the automatic choice.
/// let by_field = SparseLuOptions { ordering: Ordering::Natural, ..SparseLuOptions::new() };
/// let natural = SparseLu::with_ordering(&a, &Ordering::Natural);
/// assert_eq!(SparseLu::with_options(&a, &by_field).unwrap(), natural);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SparseLuOptions {
    /// How the ordering and the threshold are chosen: [`LuStrategy::Auto`] unless set, and
    /// [`LuStrategy::Explicit`] once an ordering or a threshold is set by
    /// [`with_ordering`](Self::with_ordering) or
    /// [`with_pivot_threshold`](Self::with_pivot_threshold). `Auto` with an ordering or a threshold
    /// that is not the default factors as `Explicit`.
    pub strategy: LuStrategy,
    /// The ordering of the columns under [`LuStrategy::Explicit`], and under
    /// [`LuStrategy::Auto`] when it or the threshold is not its default: [`Ordering::Colamd`]
    /// unless set.
    pub ordering: Ordering,
    /// The fraction of the largest element of its column the diagonal element must reach to be
    /// taken as the pivot, in `(0, 1]`, under [`LuStrategy::Explicit`] and under
    /// [`LuStrategy::Auto`] when it or the ordering is not its default: 1 unless set, which is
    /// partial pivoting. [`SparseLu::with_options`] checks its range whatever the strategy.
    pub pivot_threshold: f64,
}

impl Default for SparseLuOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseLuOptions {
    /// The defaults: the strategy [`LuStrategy::Auto`], which is what [`SparseLu::new`] does;
    /// for [`LuStrategy::Explicit`], the columns ordered by [`Ordering::Colamd`] and a pivot
    /// threshold of 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{LuStrategy, Ordering, SparseLuOptions};
    ///
    /// let options = SparseLuOptions::new();
    /// assert_eq!(options.strategy, LuStrategy::Auto);
    /// assert_eq!(options.ordering, Ordering::Colamd);
    /// assert_eq!(options.pivot_threshold, 1.0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategy: LuStrategy::Auto,
            ordering: Ordering::Colamd,
            pivot_threshold: 1.0,
        }
    }

    /// Sets the strategy, leaving the ordering and the threshold as they are.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{LuStrategy, SparseLuOptions};
    ///
    /// let options = SparseLuOptions::new().with_strategy(LuStrategy::Unsymmetric);
    /// assert_eq!(options.strategy, LuStrategy::Unsymmetric);
    /// ```
    #[must_use]
    pub fn with_strategy(mut self, strategy: LuStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the ordering of the columns, and the strategy to [`LuStrategy::Explicit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{LuStrategy, Ordering, SparseLuOptions};
    ///
    /// let options = SparseLuOptions::new().with_ordering(Ordering::Given(vec![1, 0]));
    /// assert_eq!(options.ordering, Ordering::Given(vec![1, 0]));
    /// assert_eq!(options.strategy, LuStrategy::Explicit);
    /// ```
    #[must_use]
    pub fn with_ordering(mut self, ordering: Ordering) -> Self {
        self.ordering = ordering;
        self.strategy = LuStrategy::Explicit;
        self
    }

    /// Sets the pivot threshold, and the strategy to [`LuStrategy::Explicit`]: the diagonal
    /// element of a column is its pivot when it is at least this fraction of the largest element
    /// the column offers. It must lie in `(0, 1]`, which [`SparseLu::with_options`] checks.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{LuStrategy, SparseLuOptions};
    ///
    /// let options = SparseLuOptions::new().with_pivot_threshold(0.1);
    /// assert_eq!(options.pivot_threshold, 0.1);
    /// assert_eq!(options.strategy, LuStrategy::Explicit);
    /// ```
    #[must_use]
    pub fn with_pivot_threshold(mut self, pivot_threshold: f64) -> Self {
        self.pivot_threshold = pivot_threshold;
        self.strategy = LuStrategy::Explicit;
        self
    }

    /// The strategy these options factor by: [`strategy`](Self::strategy), except that
    /// [`LuStrategy::Auto`] is [`LuStrategy::Explicit`] once the ordering or the threshold is
    /// not its default.
    fn chosen_strategy(&self) -> LuStrategy {
        let defaults = self.ordering == Ordering::Colamd && self.pivot_threshold == 1.0;
        if self.strategy == LuStrategy::Auto && !defaults {
            LuStrategy::Explicit
        } else {
            self.strategy
        }
    }
}

/// The LU decomposition `P·A·Q = L·U` of a square sparse matrix, by the left-looking elimination
/// of Gilbert and Peierls with threshold partial pivoting: `Q` a permutation of the columns that
/// keeps the factors sparse, `P` the permutation of the rows that pivoting chose, `L` lower
/// triangular with ones on its diagonal and `U` upper triangular. The factors are held by
/// columns.
///
/// The decomposition exists for every square matrix. [`is_singular`](Self::is_singular) reports
/// one that is singular — a pivot of zero, a pattern that is singular, or an estimated condition
/// number beyond the precision of the type, a decision near a bound whose rare errors are
/// listed below — and its solves return `None`.
///
/// # Known limits
///
/// Measured in the tests and in the campaigns of the reviews of this decomposition. `10^±d`
/// stands for rows or columns multiplied by factors between `10⁻ᵈ` and `10ᵈ`; the matrices
/// scaled were diagonally dominant on a scrambled diagonal, of orders up to 31, factored with
/// the default threshold and random or COLAMD orders of their columns, 2 000 for each type,
/// kind of scaling and range.
///
/// - Singular to working precision is decided from an estimate, against the bound `τ` that
///   [`is_singular`](Self::is_singular) defines. Of 6.3 million factorisations of exactly
///   singular matrices in the campaigns of review round 2, 7 were called regular, all with a
///   pivot threshold of `10⁻⁴`, the estimates exact and the factors 1 to 5 `τ` from a singular
///   matrix: the bound has no factor of the order `n`, which the backward error of an
///   elimination carries. None was missed at the default threshold or at 0.1. A matrix whose
///   condition number is near `1/ε` gives solutions with few correct digits whichever way it is
///   decided.
/// - Rows of very different sizes never make a matrix reported singular (tested to `10^±10` in
///   single and `10^±20` in double precision). Partial pivoting on unscaled rows becomes
///   unstable when their sizes differ by far more than `1/ε`, though, and then a solution can be
///   wrong while neither
///   [`is_singular`](Self::is_singular) nor [`growth`](Self::growth) shows it: no
///   solution wrong by more than 0.1 up to `10^±4` in single and `10^±12` in double precision;
///   1 and 6 of 2 000 in `f32` at `10^±5` and `10^±6`, 1 (`f64`) and 6 (`Complex<f64>`) at
///   `10^±15`, more beyond.
/// - Columns of very different sizes can make a well-conditioned matrix reported singular,
///   because one pass of the scaling, rows first, does not undo them: none up to `10^±3` in
///   single and `10^±6` in double precision, 0.1 % at `10^±8` in double, 5 to 6 % at `10^±4` in
///   single and 8 to 9 % at `10^±10` in double precision, about 70 % at the widest ranges tried.
///   Their factors would as a rule have solved them to `10⁻⁶`; the answer `None` is a refusal,
///   not a wrong result. Factoring the matrix with its columns brought to like sizes, and
///   solving for the correspondingly scaled unknowns, avoids it. A solution accepted was never
///   wrong by more than `10⁻⁵` relative to its largest element, whatever the range.
/// - With rows and columns both scaled, the same: nothing reported and no solution wrong by more
///   than 0.1 up to `10^±3` in single and `10^±6` in double precision; beyond, refusals as for
///   columns, and in some of the ranges one of the 2 000 matrices accepted with a solution wrong
///   by more than 0.1. (A solution of all ones measures the problem rather than the elimination
///   here: its components in the small columns are not determined by `b` to the precision of
///   the type.)
/// - A matrix refused for the growth of its elimination need not be near a singular one. The
///   matrix of Wilkinson with ones on its diagonal and in its last column and −1 below the
///   diagonal, condition number `n` and growth `2^(n−1)`, is reported from order 20 in single
///   and 48 in double precision, and its [`det`](Self::det) is then 0, not `2^(n−1)`.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
///
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 1, 2.0), (0, 2, 1.0),
///     (1, 0, 4.0), (1, 1, 1.0),
///     (2, 0, 2.0), (2, 2, 3.0),
/// ]);
/// let lu = SparseLu::new(&a);
/// assert!(!lu.is_singular());
/// let x = lu.solve(&Matrix::from_rows(&[[3.0], [5.0], [5.0]])).unwrap();
/// for i in 0..3 {
///     assert!((x[(i, 0)] - 1.0f64).abs() < 1e-14);
/// }
/// assert!((lu.det() + 26.0f64).abs() < 1e-12);
///
/// // In the order the matrix is in, the zero on the diagonal makes the first step swap two rows.
/// let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
/// assert_eq!(lu.permutation(), [1, 0, 2]);
/// assert_eq!(&(&lu.p() * &a) * &lu.q(), &lu.l() * &lu.u());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SparseLu<T: Scalar> {
    /// The order of the matrix.
    n: usize,
    /// Where each column of `L` begins in `l_rows` and `l_values`, of length `n + 1`.
    l_ptrs: Vec<usize>,
    /// The rows of the elements of `L` below its diagonal, as rows of `P·A·Q`, in no order within
    /// a column. The diagonal of ones is not stored.
    l_rows: Vec<usize>,
    /// The elements of `L` below its diagonal.
    l_values: Vec<T>,
    /// Where each column of `U` begins in `u_rows` and `u_values`, of length `n + 1`.
    u_ptrs: Vec<usize>,
    /// The rows of the elements of `U` above its diagonal, in no order within a column.
    u_rows: Vec<usize>,
    /// The elements of `U` above its diagonal.
    u_values: Vec<T>,
    /// The diagonal of `U`: the pivots.
    pivots: Vec<T>,
    /// Row `i` of `P·A·Q` is row `row_perm[i]` of `A`.
    row_perm: Vec<usize>,
    /// Column `j` of `P·A·Q` is column `col_perm[j]` of `A`.
    col_perm: Vec<usize>,
    /// Whether the two permutations together are an odd one.
    odd: bool,
    /// Whether the matrix is exactly singular: a column had no pivot, or the pattern has no
    /// `n` nonzeros in distinct rows and columns.
    deficient: bool,
    /// Whether every element of the factors is finite.
    finite: bool,
    /// Whether the matrix, though no pivot is zero, is singular to working precision by its
    /// estimated condition: see [`SparseLu::is_regular_to_working_precision`]. Only asked of a
    /// matrix with finite factors that isn't `deficient`.
    near_singular: bool,
    /// The largest absolute value of an element of `A`.
    largest_a: T::Real,
    /// The largest absolute value of an element of `U`.
    largest_u: T::Real,
    /// The largest sum of the absolute values of a column of `A`: its 1-norm.
    norm_one: T::Real,
}

/// Whether [`SparseLu::new`] takes the symmetric strategy for the square `a`: at least half of the
/// stored elements off the diagonal have their mirror image stored, and at least nine tenths of
/// the diagonal is stored (UMFPACK's test, with its thresholds 0.3 and 0.9 — here 0.5 and 0.9).
fn symmetric_strategy<T: Scalar>(a: &SparseMatrix<T>, t: &SparseMatrix<T>) -> bool {
    let n = a.rows();
    if n == 0 {
        return false;
    }
    let (ptrs, cols) = (a.row_pointers(), a.col_indices());
    let (tptrs, tcols) = (t.row_pointers(), t.col_indices());
    let (mut diagonal, mut off, mut matched) = (0usize, 0usize, 0usize);
    for i in 0..n {
        // Row i of A and row i of Aᵀ (column i of A), both ascending: a merge.
        let (row, mirror) = (&cols[ptrs[i]..ptrs[i + 1]], &tcols[tptrs[i]..tptrs[i + 1]]);
        let mut q = 0;
        for &j in row {
            if j == i {
                diagonal += 1;
                continue;
            }
            off += 1;
            while q < mirror.len() && mirror[q] < j {
                q += 1;
            }
            if q < mirror.len() && mirror[q] == j {
                matched += 1;
            }
        }
    }
    // The thresholds (UMFPACK 6.3 asks 0.3 for the symmetry): measured on an Apple M4 Max,
    // `SparseLu::new` by either strategy on 2-D and 3-D grids with a fraction of their edges
    // stored one way only. With a strong diagonal the symmetric strategy is the faster at every
    // symmetry from 0.2 to 0.7 (0.31 to 0.74 of the time), but with a weak one (a fifth of it or
    // less) at 0.3 and 0.4 it is stopped by its watch and costs 3 to 10 % more than partial
    // pivoting alone, so the symmetry asked stays at a half. With 5 % of the diagonal missing it
    // is 0.81 of the time on a symmetric grid and 2.1 times at a symmetry of 0.6, from 10 % on
    // 1.03 to 1.5 times: no value of the diagonal fraction is the better for both.
    2 * matched >= off && 10 * diagonal >= 9 * n
}

impl<T: Scalar> SparseLu<T> {
    /// The decomposition of the square `a` by the strategy that suits its pattern, chosen by
    /// UMFPACK's rule. Every square matrix has one; one that is singular,
    /// or singular to working precision by the estimate of its condition that
    /// [`is_singular`](Self::is_singular) describes, is reported by it and refused by
    /// [`solve`](Self::solve).
    ///
    /// It takes time proportional to the floating-point operations of the elimination plus the
    /// nonzeros of `a`, and memory proportional to the nonzeros of the factors. The estimate of
    /// the condition that tells a matrix singular to working precision adds at most seventeen
    /// solves with the factors, each in time proportional to their nonzeros, and seventeen more
    /// for a matrix whose condition is close to the bound. When some pivot is
    /// not an element of `a` itself, the check of the pattern for structural singularity adds a
    /// matching that costs `O(√n·(n + nnz(a)))` at worst and is as a rule far cheaper.
    ///
    /// The strategy:
    ///
    /// - **Symmetric**, when the pattern is nearly symmetric — at least half of the elements off
    ///   the diagonal have their mirror image stored — and at least nine tenths of the diagonal
    ///   is stored: the columns in the order [`amd`](super::amd) gives the pattern of `A + Aᵀ`,
    ///   the diagonal element the pivot as long as it is 10⁻⁴ of the largest element of its
    ///   column (a tenth of UMFPACK's `SYM_PIVOT_TOLERANCE`), and otherwise an element of at least
    ///   a tenth of the largest (its `PIVOT_TOLERANCE`), from a row whose diagonal an earlier step
    ///   passed over if there is one. The multipliers are then at most 10 000 on the diagonal and
    ///   10 off it (`√2` times those for complex elements). On the matrices of grids and meshes
    ///   with a strong diagonal this fills in 0.4 to 0.85 of what COLAMD with partial pivoting
    ///   does: 0.5 to 0.7 on two-dimensional grids, 0.4 to 0.5 on three-dimensional seven-point
    ///   ones of 10³ to 30³ points, 0.7 to 0.85 on three-dimensional elasticity meshes (three
    ///   unknowns a point) of 648 to 12 288 unknowns.
    /// - **Unsymmetric** otherwise: the columns by [`Ordering::Colamd`] and partial pivoting,
    ///   the threshold 1, every multiplier at most 1 (`√2`).
    ///
    /// The symmetric elimination is watched: when its factors hold more than twice the elements
    /// that its order predicts with every pivot on the diagonal, and as many as `a` more, or an
    /// element of `U` grows beyond 16 times the largest of `a`, it stops, and `a` is factored again
    /// from the start by the unsymmetric strategy, whose decomposition is then the result. So it is
    /// when the symmetric factors are singular, unless the pattern of `a` alone makes them so: the
    /// verdict on a singular matrix, and its time, are those of partial pivoting, plus the
    /// symmetric elimination. A diagonal that is weak, as in shifted Laplacians, Helmholtz
    /// operators and `A − σI` for a shift inside the spectrum, keeps its pivots on the diagonal but
    /// grows the elements by hundreds to tens of thousands, and takes that second factorisation:
    /// its time is that of partial pivoting and the part of the symmetric one made before it
    /// stopped.
    ///
    /// This is [`LuStrategy::Auto`]; [`with_options`](Self::with_options) gives either strategy
    /// without the watch, or any ordering and threshold; [`with_ordering`](Self::with_ordering)
    /// any ordering with partial pivoting.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 5], &[-1.0; 4]);
    /// let lu = SparseLu::new(&a);
    /// let x = lu.solve_vec(&[1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    /// for xi in x {
    ///     assert!((xi - 1.0f64).abs() < 1e-14);
    /// }
    /// ```
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Self {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        Self::by_strategy(a, LuStrategy::Auto)
    }

    /// The decomposition of the square `a` by one of the three strategies that choose their own
    /// ordering and threshold.
    #[track_caller]
    fn by_strategy(a: &SparseMatrix<T>, strategy: LuStrategy) -> Self {
        // The transpose is the matrix by columns, which the elimination reads too.
        let by_columns = a.transpose();
        let symmetric = match strategy {
            LuStrategy::Auto => symmetric_strategy(a, &by_columns),
            LuStrategy::Symmetric => true,
            LuStrategy::Unsymmetric | LuStrategy::Explicit => false,
        };
        if symmetric {
            // `amd(a)`, with the graph of `A + Aᵀ` kept for the watch.
            let graph = Graph::of_symmetric_pattern(a);
            let order = amd_of(&graph);
            let rule = Rule::symmetric();
            if strategy != LuStrategy::Auto {
                return Self::eliminate(a, &by_columns, order, rule, None)
                    .expect("an elimination without a watch runs to its end");
            }
            let mut watch = Watch::new(&graph, a.nnz());
            if let Some(lu) = Self::eliminate(a, &by_columns, order, rule, Some(&mut watch)) {
                // A matrix the symmetric elimination finds singular is left to partial pivoting
                // to judge, unless its pattern decided it: a pivot of zero, an overflow or an
                // estimated condition beyond the bound, which its growth raises, can be those of
                // the pivots rather than of the matrix.
                let by_pattern = lu.deficient && lu.pivots.iter().all(|p| !p.is_zero());
                if !lu.is_singular() || by_pattern {
                    return lu;
                }
            }
            // The symmetric elimination filled or grew beyond its watch, or found the matrix
            // singular: it is factored again, from the start, by the unsymmetric strategy.
        }
        let columns = Ordering::Colamd.columns(a);
        Self::eliminate(a, &by_columns, columns, Rule::partial(1.0), None)
            .expect("an elimination without a watch runs to its end")
    }

    /// The decomposition of the square `a` with its columns in the order `ordering` gives them,
    /// and its pivots chosen by partial pivoting.
    ///
    /// [`Ordering::Natural`] leaves the columns where they are, [`Ordering::Colamd`] orders them
    /// from the pattern of `a` itself, [`Ordering::Given`] takes a permutation `q` with
    /// `q[new] = old`, and the symmetric orderings ([`Ordering::Amd`],
    /// [`Ordering::ReverseCuthillMckee`], [`Ordering::NestedDissection`]) are applied to the
    /// pattern of `Aᴴ·A`, which is formed for them: a row of `a` with `m` nonzeros puts `m²` in
    /// it, so a dense row makes it, and the time and memory of the ordering, of the order of
    /// `n²`.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, or if a given permutation isn't one of its columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// // An arrow pointing up and left: in its own order the first step fills the whole matrix,
    /// // with the first column eliminated last nothing fills.
    /// let n = 6;
    /// let mut entries = vec![(0, 0, 8.0)];
    /// for i in 1..n {
    ///     entries.extend([(i, i, 8.0), (0, i, 1.0), (i, 0, 1.0)]);
    /// }
    /// let a = SparseMatrix::from_coo(n, n, &entries);
    /// let natural = SparseLu::with_ordering(&a, &Ordering::Natural);
    /// let last: Vec<usize> = (1..n).chain([0]).collect();
    /// let reordered = SparseLu::with_ordering(&a, &Ordering::Given(last));
    /// assert_eq!(natural.nnz(), n * n);
    /// assert_eq!(reordered.nnz(), a.nnz());
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_ordering(a: &SparseMatrix<T>, ordering: &Ordering) -> Self {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        Self::factor(a, ordering.columns(a), 1.0)
    }

    /// The decomposition of the square `a` by the strategy of `options` — with its column
    /// ordering and pivot threshold under [`LuStrategy::Explicit`], and under
    /// [`LuStrategy::Auto`] when either of them is not its default (see [`SparseLuOptions`]) —
    /// or `None` if the threshold isn't in `(0, 1]`. The default options give what
    /// [`new`](Self::new) gives.
    ///
    /// With a threshold `t` below 1 the diagonal element of a column is its pivot whenever it is
    /// nonzero and at least `t` times the largest element the column offers, so the elements of
    /// `L` can be as large as `1/t` (`√2/t` in a complex matrix, whose elements are sized by
    /// `|re| + |im|`). That keeps the pivots where a symmetric ordering put them
    /// — pass it as [`Ordering::Given`] — at the price of an elimination that can grow more:
    /// [`growth`](Self::growth) and [`rcond`](Self::rcond) tell whether it did, and a
    /// decomposition whose growth makes its factors useless is reported by
    /// [`is_singular`](Self::is_singular), whatever the condition of the matrix. The threshold is
    /// rounded to the real type of the elements, so one below its range (`10⁻⁶⁰` for `f32`)
    /// becomes 0 and keeps any diagonal element that is not zero.
    ///
    /// The shape of the input is checked before the threshold: a given permutation that isn't
    /// one of the columns panics whatever the threshold is.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, or if a given permutation isn't one of its columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseLuOptions, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 4.0), (1, 1, 2.0)]);
    /// let natural = SparseLuOptions::new().with_ordering(Ordering::Natural);
    /// // A quarter of the largest is enough for a threshold of 0.25, not for one of 0.5.
    /// let kept = SparseLu::with_options(&a, &natural.clone().with_pivot_threshold(0.25)).unwrap();
    /// assert_eq!(kept.permutation(), [0, 1]);
    /// let swapped = SparseLu::with_options(&a, &natural.clone().with_pivot_threshold(0.5)).unwrap();
    /// assert_eq!(swapped.permutation(), [1, 0]);
    /// assert!(SparseLu::with_options(&a, &natural.with_pivot_threshold(0.0)).is_none());
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_options(a: &SparseMatrix<T>, options: &SparseLuOptions) -> Option<Self> {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        // A wrong shape panics before a wrong number gives `None`; the other orderings are
        // computed from `a` and have no shape to get wrong.
        if let Ordering::Given(q) = &options.ordering {
            assert_permutation(q, a.cols());
        }
        let threshold = options.pivot_threshold;
        // Written so that a threshold that is not a number fails it.
        if !(threshold > 0.0 && threshold <= 1.0) {
            return None;
        }
        if options.chosen_strategy() != LuStrategy::Explicit {
            return Some(Self::by_strategy(a, options.strategy));
        }
        let columns = options.ordering.columns(a);
        Some(Self::factor(a, columns, threshold))
    }

    /// Factors `a` with its columns in the order `col_perm`, `col_perm[new] = old`, and threshold
    /// partial pivoting with the threshold `threshold`.
    #[track_caller]
    fn factor(a: &SparseMatrix<T>, col_perm: Vec<usize>, threshold: f64) -> Self {
        Self::eliminate(a, &a.transpose(), col_perm, Rule::partial(threshold), None)
            .expect("an elimination without a watch runs to its end")
    }

    /// Factors `a`, whose transpose is `by_columns`, with its columns in the order `col_perm` and
    /// its pivots by `rule`; `None` if `watch` stopped the elimination.
    #[track_caller]
    fn eliminate(
        a: &SparseMatrix<T>,
        by_columns: &SparseMatrix<T>,
        col_perm: Vec<usize>,
        rule: Rule,
        mut watch: Option<&mut Watch<'_>>,
    ) -> Option<Self> {
        let n = a.rows();
        assert_permutation(&col_perm, n);
        // The matrix by columns: the rows of its transpose.
        let starts = by_columns.row_pointers();
        let rows = by_columns.col_indices();
        let values = by_columns.values();
        let mut largest_a = T::Real::ZERO;
        let mut norm_one = T::Real::ZERO;
        for column in 0..n {
            let mut sum = T::Real::ZERO;
            for x in &values[starts[column]..starts[column + 1]] {
                let size = x.abs();
                largest_a = largest_a.max(size);
                sum += size;
            }
            norm_one = norm_one.max(sum);
        }
        // The step of each column of `A`, which the choice of a pivot off the diagonal asks.
        let mut step_of_column = Vec::new();
        if rule.off < 1.0 {
            step_of_column = vec![0; n];
            for (step, &column) in col_perm.iter().enumerate() {
                step_of_column[column] = step;
            }
        }
        let pivoting = Pivoting {
            diagonal: T::Real::from_f64(rule.diagonal),
            off: T::Real::from_f64(rule.off),
            step_of_column: &step_of_column,
        };
        // A column at a time while the columns stay sparse, then by supernodes.
        let supernodal = n >= SUPERNODAL;
        let mut work = Elimination::new(n, a.nnz());
        let mut off_diagonal = false;
        let mut j = 0;
        while j < n {
            let column = col_perm[j];
            let (begin, end) = (starts[column], starts[column + 1]);
            work.step(j, column, &rows[begin..end], &values[begin..end], &pivoting);
            if let Some(watch) = watch.as_deref_mut() {
                off_diagonal |= work.row_perm[j] != column;
                let made = work.l_rows.len() + work.u_rows.len() + j + 1;
                if watch.exceeded(&col_perm, j, off_diagonal, made, work.largest_u, largest_a) {
                    return None;
                }
            }
            j += 1;
            let elements =
                work.l_ptrs[j] - work.l_ptrs[j - 1] + work.u_ptrs[j] - work.u_ptrs[j - 1];
            if supernodal && elements >= DENSE_COLUMN {
                break;
            }
        }
        let mut work = if j < n {
            let watched = watch.map(|watch| (watch, off_diagonal, largest_a));
            let columns = supernodes::Columns {
                col_perm: &col_perm,
                starts,
                rows,
                values,
            };
            supernodes::eliminate(&columns, &pivoting, work, j, watched)?
        } else {
            work.into()
        };
        // The rows of `L` are known by their place in `P·A` only now that every row has one.
        for row in &mut work.l_rows {
            *row = work.step_of_row[*row];
        }
        // A pivot that rounding alone kept from zero must not hide a matrix whose pattern makes it
        // singular. A matrix already found singular needs no more evidence.
        let deficient = work.deficient
            || (work.finite && !has_perfect_matching(a, by_columns, &work.row_perm, &col_perm));
        let odd = is_odd(&work.row_perm) != is_odd(&col_perm);
        let mut lu = Self {
            n,
            l_ptrs: work.l_ptrs,
            l_rows: work.l_rows,
            l_values: work.l_values,
            u_ptrs: work.u_ptrs,
            u_rows: work.u_rows,
            u_values: work.u_values,
            pivots: work.pivots,
            row_perm: work.row_perm,
            col_perm,
            odd,
            deficient,
            finite: work.finite,
            near_singular: false,
            largest_a,
            largest_u: work.largest_u,
            norm_one,
        };
        // Every pivot is finite and none is zero, but that alone says little: see the module's
        // documentation.
        if n > 0 && !deficient && lu.finite {
            lu.near_singular = !lu.is_regular_to_working_precision(a, by_columns);
        }
        Some(lu)
    }

    /// Whether the matrix, with factors that are finite and no pivot of zero, is regular to
    /// working precision. With `τ = ε·max(1, ρ)`, `ρ` the [growth](Self::growth) of the
    /// elimination, it is singular when the smaller of two reciprocal condition numbers in the
    /// 1-norm is below `τ` — that of `B = D_r·A·D_c` and that of `A` as it is — unless the one of
    /// `B` is [`UNSCALED_RANGE`] times `τ` or more. `D_r` scales each row of `A` to a largest
    /// absolute value of 1 and `D_c` then each column of `D_r·A`, as LAPACK's `xGEEQU` does;
    /// `by_columns` is the transpose of `a`. `‖B⁻¹‖₁ = ‖D_c⁻¹·A⁻¹·D_r⁻¹‖₁` is estimated with the
    /// factors, and so is `‖A⁻¹‖₁`, though only when the first leaves the answer open.
    ///
    /// The scaled condition is what keeps rows of very different sizes (and columns, as far as
    /// one pass of the scaling undoes them) from being taken for a matrix near a singular one;
    /// the unscaled one is the condition in the norm in
    /// which the elimination, which pivots on `A` itself, is backward stable, and it catches the
    /// singular matrices that the scaled norm, in which a small row eliminated against large
    /// ones is not, lets through; the range keeps it from refusing a matrix that the scaling
    /// shows to be well conditioned. The factors are those of a matrix within about `ρ·ε` of
    /// `A`, relative to its size (Wilkinson's bound on the backward error of Gaussian
    /// elimination), so that a matrix nearer than that to a singular one cannot be told from
    /// it; growth beyond a few units needs a pivot threshold below 1 or a contrived matrix.
    ///
    /// The estimates of the norms are lower bounds, so a matrix well within the test passes but
    /// for rounding. An inverse that overflows, which only elements near the ends of the range
    /// of the type allow in a matrix that would pass, fails it.
    fn is_regular_to_working_precision(
        &self,
        a: &SparseMatrix<T>,
        by_columns: &SparseMatrix<T>,
    ) -> bool {
        let n = self.n;
        let zero = T::Real::ZERO;
        let tiny = T::Real::MIN_POSITIVE;
        // `D_r⁻¹` and `D_c⁻¹`: the largest absolute value of each row of `A`, then of each
        // column of `D_r·A`. No row or column is empty in a matrix with a complete matching; a
        // largest value below the smallest normal number is raised to it, as `xGEEQU` does, so
        // that it can be divided by.
        let mut row_scale = vec![zero; n];
        for (i, scale) in row_scale.iter_mut().enumerate() {
            *scale = a
                .row(i)
                .fold(zero, |most, (_, x)| most.max(x.abs()))
                .max(tiny);
        }
        let starts = by_columns.row_pointers();
        let rows = by_columns.col_indices();
        let values = by_columns.values();
        let mut column_scale = vec![zero; n];
        let mut norm = zero;
        for (j, scale) in column_scale.iter_mut().enumerate() {
            let column = starts[j]..starts[j + 1];
            let scaled = |at: usize| values[at].abs() / row_scale[rows[at]];
            let most = column
                .clone()
                .fold(zero, |most, at| most.max(scaled(at)))
                .max(tiny);
            *scale = most;
            norm = norm.max(column.fold(zero, |sum, at| sum + scaled(at) / most));
        }
        // `1/τ`, and the reciprocals of the two reciprocal condition numbers, compared with it;
        // an estimate that is not a number, which only an overflow in the solves can make,
        // fails.
        let growth = (self.largest_u / self.largest_a).max(T::Real::ONE);
        let limit = T::Real::ONE / (T::Real::EPSILON * growth);
        let scaled = norm * self.inverse_norm_estimate((&row_scale, &column_scale));
        if scaled.is_nan() || scaled > limit {
            return false;
        }
        if scaled * T::Real::from_f64(UNSCALED_RANGE) <= limit {
            return true;
        }
        // Within the range where the unscaled condition decides: `‖A‖₁·‖A⁻¹‖₁` as `rcond`
        // estimates it.
        let inner = vec![self.norm_one; n];
        let outer = vec![T::Real::ONE; n];
        self.inverse_norm_estimate((&inner, &outer)) <= limit
    }

    /// Whether the matrix is singular to working precision, which is when any of these holds:
    ///
    /// - a column was left without a pivot, because no row remained for it or none but zeros;
    /// - the pattern of the matrix has no `n` nonzeros in distinct rows and columns, which makes
    ///   it singular whatever its elements are, though rounding kept every pivot from zero;
    /// - the elimination met an element that is not finite, in the matrix or by overflow, and
    ///   has no factors to trust;
    /// - with `τ = ε·max(1, ρ)`, `ε` the machine epsilon of the type and `ρ` the
    ///   [growth](Self::growth) of the elimination: the smaller of the estimated reciprocal
    ///   condition numbers in the 1-norm of the matrix as it is and of the matrix with its rows
    ///   and then its columns scaled to a largest absolute value of 1 is below `τ`, and the
    ///   scaled one is below `16·τ`.
    ///
    /// The last is what catches a matrix that is singular in exact arithmetic but whose
    /// elimination ends with a pivot of the size of the rounding error instead of zero, as most
    /// do. Whether a pivot comes out as exactly zero depends on the order of the columns and on
    /// the rounding; the condition of the matrix does not, but its estimate is made from the
    /// factors and is an estimate: a lower bound for the norm of the inverse, from at most
    /// seventeen solves, by a search started from a vector of equal elements, one of alternating
    /// signs and one of pseudo-random signs and sizes, so that a singular direction is not hidden
    /// from all three. So the verdict, too, depends on the order of the columns and on the
    /// rounding; the exactly singular matrices found regular since the third start vector was
    /// added all had factors near the bound, and they are counted, with the refusals of regular
    /// matrices, under the [known limits](SparseLu#known-limits). For a stable elimination of a
    /// matrix whose rows and columns are of like sizes it is the test by which LAPACK's
    /// `xGESVX` and SuperLU's `xGSSVX` define singular to working precision, the reciprocal
    /// condition number below `ε`: the bound on the error of a solution is then as large as the
    /// solution. A pivot threshold below 1 can let the elimination grow, and then the test
    /// refuses sooner, as the factors are those of a matrix within about `ρ·ε` of `A` rather
    /// than `ε`: when `ρ·ε` nears 1 the factors can tell nothing from a singular matrix, and a
    /// well-conditioned matrix is reported too, as its solutions by those factors are noise.
    /// That takes growth of the order of `1/ε`, which a threshold of `10⁻⁸` in single precision
    /// often brings; refactor with a larger threshold. The scaling leaves rows that are merely of
    /// different sizes alone, and diagonal matrices: `diag(10⁻²⁰, 1)` is regular, though its
    /// [`rcond`](Self::rcond) is `10⁻²⁰`, because its scaled condition is 1. Columns of different
    /// sizes it undoes only in part: a well-conditioned matrix whose columns differ in size by
    /// more than about `1/ε` can be reported, though its factors would solve it (the rates are
    /// among the [known limits](SparseLu#known-limits)); bring the columns to like sizes before
    /// factoring it. The unscaled condition, in whose norm the elimination is backward stable,
    /// catches the singular matrices whose small rows the scaling would let pass. A matrix well
    /// within the test is never reported, the estimates of the norms of the inverses being lower
    /// bounds.
    ///
    /// This differs from the dense [`Lu`](crate::math::matrix::Lu), which reports a pivot of
    /// exactly zero alone.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// // The second row is twice the first.
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 4.0)]);
    /// assert!(SparseLu::new(&a).is_singular());
    /// // Two columns that share their only row: singular whatever the elements are.
    /// let b = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 5.0)]);
    /// assert!(SparseLu::new(&b).is_singular());
    /// assert!(!SparseLu::new(&SparseMatrix::<f64>::identity(2)).is_singular());
    /// // The integers 1 to 9: singular, though in this order of the columns the last pivot is
    /// // about 10⁻¹⁶ rather than zero.
    /// let c = SparseMatrix::from_fn(3, 3, |i, j| (3 * i + j + 1) as f64);
    /// let lu = SparseLu::with_ordering(&c, &Ordering::Natural);
    /// assert!(lu.u().diagonal().iter().all(|&p| p != 0.0));
    /// assert!(lu.is_singular() && lu.det() == 0.0 && lu.solve_vec(&[1.0, 0.0, 0.0]).is_none());
    /// // Rows of very different sizes are not a reason.
    /// let d = SparseMatrix::from_diagonal(&[1e-20, 1.0]);
    /// assert!(!SparseLu::new(&d).is_singular());
    /// ```
    #[must_use]
    pub fn is_singular(&self) -> bool {
        self.deficient || !self.finite || self.near_singular
    }

    /// The unit lower triangular factor `L`, with its diagonal of ones stored: `P·A·Q = L·U`.
    /// An element that the elimination made exactly zero is left out of the result, as of every
    /// [`SparseMatrix`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 4.0), (1, 1, 3.0)]);
    /// let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    /// let l = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (1, 0, 0.5), (1, 1, 1.0)]);
    /// assert_eq!(lu.l(), l);
    /// ```
    #[must_use]
    pub fn l(&self) -> SparseMatrix<T> {
        let ones = vec![T::one(); self.n];
        by_rows(self.n, &self.l_ptrs, &self.l_rows, &self.l_values, &ones)
    }

    /// The upper triangular factor `U`: `P·A·Q = L·U`. An element that the elimination made
    /// exactly zero is left out of the result, a pivot of zero among them.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 4.0), (1, 1, 3.0)]);
    /// let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    /// let u = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (0, 1, 3.0), (1, 1, -0.5)]);
    /// assert_eq!(lu.u(), u);
    /// ```
    #[must_use]
    pub fn u(&self) -> SparseMatrix<T> {
        by_rows(
            self.n,
            &self.u_ptrs,
            &self.u_rows,
            &self.u_values,
            &self.pivots,
        )
    }

    /// The permutation matrix `P` of the rows: row `i` of `P·A` is row
    /// [`permutation()[i]`](Self::permutation) of `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 1.0), (1, 0, 1.0)]);
    /// let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    /// assert_eq!(lu.p(), a);
    /// assert_eq!(&lu.p() * &a, SparseMatrix::identity(2));
    /// ```
    #[must_use]
    pub fn p(&self) -> SparseMatrix<T> {
        let n = self.n;
        SparseMatrix::from_csr(
            n,
            n,
            (0..=n).collect(),
            self.row_perm.clone(),
            vec![T::one(); n],
        )
    }

    /// The permutation matrix `Q` of the columns: column `j` of `A·Q` is column
    /// [`column_permutation()[j]`](Self::column_permutation) of `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)]);
    /// let lu = SparseLu::with_ordering(&a, &Ordering::Given(vec![1, 0]));
    /// let swapped = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 3.0)]);
    /// assert_eq!(&a * &lu.q(), swapped);
    /// ```
    #[must_use]
    pub fn q(&self) -> SparseMatrix<T> {
        let n = self.n;
        // Row `old` of `Q` has its one in the column `new` with `col_perm[new] = old`.
        let mut inverse = vec![0; n];
        for (new, &old) in self.col_perm.iter().enumerate() {
            inverse[old] = new;
        }
        SparseMatrix::from_csr(n, n, (0..=n).collect(), inverse, vec![T::one(); n])
    }

    /// The permutation of the rows: row `i` of `P·A·Q` is row `permutation()[i]` of `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// // A cyclic shift: every pivot is found off the diagonal.
    /// let a = SparseMatrix::from_coo(3, 3, &[(0, 2, 1.0), (1, 0, 1.0), (2, 1, 1.0)]);
    /// let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    /// assert_eq!(lu.permutation(), [1, 2, 0]);
    /// ```
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.row_perm
    }

    /// The permutation of the columns: column `j` of `P·A·Q` is column `column_permutation()[j]`
    /// of `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::<f64>::identity(3);
    /// let lu = SparseLu::with_ordering(&a, &Ordering::Given(vec![2, 0, 1]));
    /// assert_eq!(lu.column_permutation(), [2, 0, 1]);
    /// // The pivots stay on the diagonal, so the rows follow the columns.
    /// assert_eq!(lu.permutation(), [2, 0, 1]);
    /// ```
    #[must_use]
    pub fn column_permutation(&self) -> &[usize] {
        &self.col_perm
    }

    /// The solution `X` of `A·X = B`, or `None` if the matrix is singular — to working precision
    /// included, as [`is_singular`](Self::is_singular) tells — or the solution isn't finite.
    /// `B` may hold several right-hand sides as its columns; each takes time proportional to the
    /// nonzeros of the factors.
    ///
    /// # Panics
    ///
    /// Panics if `b` hasn't as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 2.0), (1, 0, 4.0), (1, 1, 1.0)]);
    /// let lu = SparseLu::new(&a);
    /// let b = Matrix::from_rows(&[[2.0, 4.0], [5.0, 6.0]]);
    /// let x = lu.solve(&b).unwrap();
    /// assert_eq!(x, Matrix::from_rows(&[[1.0, 1.0], [1.0, 2.0]]));
    ///
    /// let singular = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (1, 0, 1.0)]);
    /// assert!(SparseLu::new(&singular).solve(&b).is_none());
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Option<Matrix<T>> {
        self.solve_columns(System::Direct, b)
    }

    /// The solution `x` of `A·x = b` for one right-hand side, or `None` if the matrix is singular
    /// — to working precision included, as [`is_singular`](Self::is_singular) tells — or the
    /// solution isn't finite.
    ///
    /// # Panics
    ///
    /// Panics if `b` isn't as long as the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 2.0), (1, 0, 4.0), (1, 1, 1.0)]);
    /// assert_eq!(SparseLu::new(&a).solve_vec(&[2.0, 5.0]), Some(vec![1.0, 1.0]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_vec(&self, b: &[T]) -> Option<Vec<T>> {
        self.solve_one(System::Direct, b)
    }

    /// The solution `X` of `Aᵀ·X = B`, from the same factors, or `None` if the matrix is singular
    /// — to working precision included, as [`is_singular`](Self::is_singular) tells — or the
    /// solution isn't finite.
    ///
    /// # Panics
    ///
    /// Panics if `b` hasn't as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 2.0), (1, 0, 4.0), (1, 1, 1.0)]);
    /// let lu = SparseLu::new(&a);
    /// // The transpose has the rows (0, 4) and (2, 1).
    /// let x = lu.solve_transpose(&Matrix::from_rows(&[[4.0], [3.0]])).unwrap();
    /// assert_eq!(x, Matrix::from_rows(&[[1.0], [1.0]]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_transpose(&self, b: &Matrix<T>) -> Option<Matrix<T>> {
        self.solve_columns(System::Transpose, b)
    }

    /// The solution `x` of `Aᵀ·x = b` for one right-hand side, or `None` if the matrix is
    /// singular — to working precision included, as [`is_singular`](Self::is_singular) tells —
    /// or the solution isn't finite.
    ///
    /// # Panics
    ///
    /// Panics if `b` isn't as long as the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 2.0), (1, 0, 4.0), (1, 1, 1.0)]);
    /// assert_eq!(SparseLu::new(&a).solve_transpose_vec(&[4.0, 3.0]), Some(vec![1.0, 1.0]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_transpose_vec(&self, b: &[T]) -> Option<Vec<T>> {
        self.solve_one(System::Transpose, b)
    }

    /// The solution `X` of `Aᴴ·X = B`, from the same factors, or `None` if the matrix is singular
    /// — to working precision included, as [`is_singular`](Self::is_singular) tells — or the
    /// solution isn't finite. For a real matrix it is
    /// [`solve_transpose`](Self::solve_transpose).
    ///
    /// # Panics
    ///
    /// Panics if `b` hasn't as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let i = Complex::new(0.0, 1.0);
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, i), (0, 1, Complex::new(1.0, 0.0)), (1, 1, i)]);
    /// let lu = SparseLu::new(&a);
    /// // The conjugate transpose has the rows (−i, 0) and (1, −i).
    /// let b = Matrix::from_rows(&[[-i], [Complex::new(1.0, -1.0)]]);
    /// let x = lu.solve_conjugate_transpose(&b).unwrap();
    /// assert_eq!(a.conjugate_transpose().mul_matrix(&x), b);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_conjugate_transpose(&self, b: &Matrix<T>) -> Option<Matrix<T>> {
        self.solve_columns(System::ConjugateTranspose, b)
    }

    /// The solution `x` of `Aᴴ·x = b` for one right-hand side, or `None` if the matrix is
    /// singular — to working precision included, as [`is_singular`](Self::is_singular) tells —
    /// or the solution isn't finite.
    ///
    /// # Panics
    ///
    /// Panics if `b` isn't as long as the order of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(0.0, 2.0))]);
    /// let x = SparseLu::new(&a).solve_conjugate_transpose_vec(&[Complex::new(4.0, 0.0)]);
    /// // −2i·x = 4 has the solution 2i.
    /// assert_eq!(x, Some(vec![Complex::new(0.0, 2.0)]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_conjugate_transpose_vec(&self, b: &[T]) -> Option<Vec<T>> {
        self.solve_one(System::ConjugateTranspose, b)
    }

    /// The determinant of the matrix: the product of the pivots, negated when the permutations
    /// of the rows and of the columns together are an odd one. It is zero for every matrix that
    /// [`is_singular`](Self::is_singular) reports, singular to working precision included, 1 for
    /// the `0 × 0` matrix, and not a number when the matrix or its elimination wasn't finite. The
    /// product of the pivots of a matrix singular to working precision is as a rule made of
    /// rounding errors, but not always: a matrix refused for the growth of its elimination can
    /// have exact pivots and a determinant far from zero, and `det` is zero for it all the same
    /// (Wilkinson's matrix of growth `2^(n−1)` and determinant `2^(n−1)`, refused from order 20
    /// in single and 48 in double precision, is one). The product of many pivots overflows or
    /// underflows easily, and there is no logarithmic form of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// // An odd permutation of the columns of a diagonal matrix.
    /// let a = SparseMatrix::from_coo(3, 3, &[(0, 1, 2.0), (1, 0, 3.0), (2, 2, 5.0)]);
    /// assert_eq!(SparseLu::new(&a).det(), -30.0);
    /// let singular = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (1, 0, 1.0)]);
    /// assert_eq!(SparseLu::new(&singular).det(), 0.0);
    /// ```
    #[must_use]
    pub fn det(&self) -> T {
        if !self.finite {
            return T::from_real(T::Real::NAN);
        }
        // Asked first: a pivot of zero times one that overflowed would not be zero.
        if self.deficient || self.near_singular {
            return T::zero();
        }
        let mut product = T::one();
        for &pivot in &self.pivots {
            product *= pivot;
        }
        if self.odd { -product } else { product }
    }

    /// The number of elements the two factors hold together, the diagonal of ones of `L` not
    /// counted: what a column ordering is judged on. It counts the elements the elimination
    /// reaches, including the few it may leave exactly zero, which [`l`](Self::l) and
    /// [`u`](Self::u) drop.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseMatrix};
    ///
    /// // A tridiagonal matrix that needs no pivoting fills nothing in when it is left in its order.
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 4], &[4.0; 5], &[-1.0; 4]);
    /// assert_eq!(SparseLu::with_ordering(&a, &Ordering::Natural).nnz(), a.nnz());
    /// ```
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.l_rows.len() + self.u_rows.len() + self.n
    }

    /// How much the elements grew in the elimination: the largest absolute value of an element of
    /// `U` over the largest of `A`. Partial pivoting keeps it small on all but contrived
    /// matrices; a large value, which a small pivot threshold makes more likely, says that the
    /// factors are those of a matrix as far from `A` as the growth is large, in units of the
    /// rounding error, and [`is_singular`](Self::is_singular) asks that much more distance from
    /// a singular matrix. It is 1 for a matrix of zeros and not a number when the elimination
    /// wasn't finite. Being taken over the whole matrix, it can miss growth among the small
    /// elements of a matrix whose rows or columns differ in size by many orders of magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLu, SparseLuOptions, SparseMatrix};
    ///
    /// let a = SparseMatrix::<f64>::from_coo(2, 2, &[(0, 0, 0.01), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]);
    /// let natural = SparseLuOptions::new().with_ordering(Ordering::Natural);
    /// let pivoted = SparseLu::with_options(&a, &natural).unwrap();
    /// assert_eq!(pivoted.growth(), 1.0);
    /// // Kept on the diagonal, the pivot of 0.01 makes an element of 99.
    /// let kept = SparseLu::with_options(&a, &natural.with_pivot_threshold(0.001)).unwrap();
    /// assert!((kept.growth() - 99.0f64).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn growth(&self) -> T::Real {
        if !self.finite {
            T::Real::NAN
        } else if self.largest_a > T::Real::ZERO {
            self.largest_u / self.largest_a
        } else {
            T::Real::ONE
        }
    }

    /// An estimate of the reciprocal of the condition number in the 1-norm,
    /// `1/(‖A‖₁·‖A⁻¹‖₁)`, of the matrix as it is: near 1 for a well-conditioned matrix, 0 for one
    /// [`is_singular`](Self::is_singular) reports — those whose estimated condition it finds
    /// beyond the precision of the type among them — and 1 for the `0 × 0` matrix. It is not
    /// scaled, so
    /// a matrix whose rows or columns are of very different sizes has a small one without being
    /// singular (`diag(10⁻²⁰, 1)` has `10⁻²⁰`), and a reciprocal below the smallest number of
    /// the type comes out as 0.
    ///
    /// `‖A⁻¹‖₁` is estimated by Hager's method as Higham refined it for LAPACK (*FORTRAN codes
    /// for estimating the one-norm of a real or complex matrix*, ACM Trans. Math. Softw. 14,
    /// 1988), with one more step of its search from the vector of alternating signs, which
    /// keeps it from stopping far short of the norm of a matrix near one with two equal rows,
    /// and a search step from a vector of pseudo-random signs and sizes, which keeps it from
    /// missing the large direction of the inverse of a matrix near one whose left null vector is
    /// orthogonal to both fixed vectors: from a few solves with `A` and with `Aᴴ`, at most
    /// seventeen, each in time proportional to the nonzeros of the factors, on vectors scaled by
    /// `‖A‖₁` so that the solves overflow only where the condition number does (the `1 × 1`
    /// matrix of the smallest positive `f64` has 1). The estimate of the norm is a lower bound,
    /// so the result is never smaller than the true value but for rounding; it is not always
    /// close: on 920 000 random matrices it was up to 22 times larger than the true value, before
    /// the two steps after the search were added, which can only bring it closer.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLu, SparseMatrix};
    ///
    /// let a = SparseMatrix::<f64>::from_diagonal(&[1.0, 0.5, 0.001]);
    /// assert!((SparseLu::new(&a).rcond() - 0.001f64).abs() < 1e-15);
    /// let singular = SparseMatrix::from_diagonal(&[1.0, 0.0]);
    /// assert_eq!(SparseLu::new(&singular).rcond(), 0.0);
    /// // Perfectly conditioned, though its inverse is beyond the range of `f64`.
    /// let tiny = SparseMatrix::from_diagonal(&[f64::from_bits(1)]);
    /// assert_eq!(SparseLu::new(&tiny).rcond(), 1.0);
    /// ```
    #[must_use]
    pub fn rcond(&self) -> T::Real {
        let n = self.n;
        if n == 0 {
            return T::Real::ONE;
        }
        if self.is_singular() {
            return T::Real::ZERO;
        }
        // The estimate of `‖‖A‖₁·A⁻¹‖₁`, which is the condition number itself.
        let inner = vec![self.norm_one; n];
        let outer = vec![T::Real::ONE; n];
        let reciprocal = T::Real::ONE / self.inverse_norm_estimate((&inner, &outer));
        // An estimate that overflowed, or is not a number, is that of a matrix nothing can be
        // solved with; and rounding alone could put the reciprocal above 1.
        if reciprocal.is_finite() {
            reciprocal.min(T::Real::ONE)
        } else {
            T::Real::ZERO
        }
    }

    /// A lower bound for `‖D_o·A⁻¹·D_i‖₁` that is close to it as a rule, `D_i` and `D_o` the
    /// diagonal matrices of `inner` and `outer` in `(inner, outer)`, of a matrix whose factors
    /// are finite, with no pivot of zero, and of an order above 0: the largest `‖D_o·A⁻¹·D_i·x‖₁`
    /// over a few `x` with `‖x‖₁ = 1`, each chosen from the gradient of the norm at the one
    /// before, and one more whose signs alternate, which catches the matrices the search can
    /// miss, followed by one step of the search from it; then the same from a vector of
    /// pseudo-random signs and sizes. At most seventeen solves.
    ///
    /// The two steps after the alternating vector are not LAPACK's. The search starts from a
    /// vector of equal elements, which has no part along a left null vector whose elements sum
    /// to zero, such as that of a matrix with two equal rows. Near such a singular matrix — two
    /// rows that the scaling makes equal are enough — the search then sees nothing of the one
    /// large direction of the inverse and can stop hundreds of times short of its norm, and the
    /// alternating vector, whose part along it can be small, recovers only some of that; a step
    /// from it recovers the rest (two solves). A left null vector can be orthogonal to the
    /// alternating vector as well — `(…, 1, 1, −1, −1, …)`, that of a matrix with rows
    /// `r_i + r_{i+1} = r_{i+2} + r_{i+3}`, is, wherever it begins — and a matrix of elements
    /// `±1` then leaves the search to the ties that rounding breaks: review round 2 found such
    /// singular matrices estimated up to `10¹⁴` times short and called regular. The
    /// pseudo-random vector, the same for every matrix so that a verdict can be repeated, has a
    /// part along every vector of that kind (checked to the order `10⁶`: at least `1.4·10⁻⁶` of
    /// its elements of sizes 1 to 2), and a step from it finds the large direction (three
    /// solves).
    fn inverse_norm_estimate(&self, scales: (&[T::Real], &[T::Real])) -> T::Real {
        let n = self.n;
        let zero = T::Real::ZERO;
        let norm = |v: &[T]| v.iter().fold(zero, |sum, x| sum + x.abs());
        let order = T::Real::from_f64(n as f64);
        // The three searches are independent until the first ends its first step, so they run
        // side by side, three vectors to a solve, which reads the factors once for all three:
        // the vector of equal elements, the one of alternating signs that grow from 1 to 2,
        // whose 1-norm is 3n/2, and the one of pseudo-random signs and sizes in `[1, 2)`, the
        // same for every matrix. Each vector gets the arithmetic it would get alone.
        let mut x = vec![T::zero(); STARTS * n];
        let mut v = vec![T::zero(); STARTS * n];
        let mut work = vec![T::zero(); STARTS * n];
        let (equal, rest) = x.split_at_mut(n);
        let (alternating, random) = rest.split_at_mut(n);
        equal.fill(T::from_real(T::Real::ONE / order));
        for (i, element) in alternating.iter_mut().enumerate() {
            let size = T::Real::ONE + T::Real::from_f64(i as f64) / (order - T::Real::ONE);
            *element = T::from_real(if i % 2 == 0 { size } else { -size });
        }
        let mut state = RANDOM_START;
        let mut total = zero;
        for element in random.iter_mut() {
            let bits = split_mix(&mut state);
            // The size from the 53 high bits, the sign from the lowest.
            let size = T::Real::ONE + T::Real::from_f64((bits >> 11) as f64 / (1u64 << 53) as f64);
            total += size;
            *element = T::from_real(if bits & 1 == 0 { size } else { -size });
        }
        self.scaled_solve_starts(System::Direct, scales, &mut x, &mut v, &mut work);
        let mut estimate = norm(&v[..n]);
        if n == 1 {
            return estimate;
        }
        let alternating = T::Real::TWO * norm(&v[n..2 * n]) / (T::Real::from_f64(3.0) * order);
        let random = norm(&v[2 * n..]) / total;
        // The first step of each search: to the unit vector where its norm rises fastest.
        let steepest = self.steepest_starts(scales, &mut x, &mut v, &mut work);
        for (start, &(place, _)) in steepest.iter().enumerate() {
            let x = &mut x[start * n..(start + 1) * n];
            x.fill(T::zero());
            x[place] = T::one();
        }
        self.scaled_solve_starts(System::Direct, scales, &mut x, &mut v, &mut work);
        let after_alternating = norm(&v[n..2 * n]);
        let after_random = norm(&v[2 * n..]);
        // The search from the vector of equal elements goes on alone while it rises.
        let (x, v, work) = (&mut x[..n], &mut v[..n], &mut work[..n]);
        let next = norm(v);
        // Not `next <= estimate`, as the search alone has it: one that is not a number goes on.
        if next > estimate || next.is_nan() || estimate.is_nan() {
            estimate = next;
            let mut last = steepest[0].0;
            for _ in 1..ESTIMATE_CYCLES {
                let (steepest, largest) = self.steepest(scales, x, v, work);
                // The vector tried last is still as steep as any: there is nothing better to try.
                if v[last].abs() >= largest {
                    break;
                }
                last = steepest;
                x.fill(T::zero());
                x[steepest] = T::one();
                self.scaled_solve(System::Direct, scales, x, v, work);
                let next = norm(v);
                if next <= estimate {
                    break;
                }
                estimate = next;
            }
        }
        estimate
            .max(alternating)
            .max(after_alternating)
            .max(random)
            .max(after_random)
    }

    /// [`steepest`](Self::steepest) for the [`STARTS`] vectors of `v`, one after the other,
    /// at once.
    fn steepest_starts(
        &self,
        scales: (&[T::Real], &[T::Real]),
        x: &mut [T],
        v: &mut [T],
        work: &mut [T],
    ) -> [(usize, T::Real); STARTS] {
        let zero = T::Real::ZERO;
        for (sign, &element) in x.iter_mut().zip(v.iter()) {
            let size = element.abs();
            *sign = if size > zero {
                T::from_parts(element.re() / size, element.im() / size)
            } else {
                T::one()
            };
        }
        self.scaled_solve_starts(System::ConjugateTranspose, scales, x, v, work);
        core::array::from_fn(|start| {
            let mut steepest = 0;
            let mut largest = -T::Real::ONE;
            for (i, element) in v[start * self.n..(start + 1) * self.n].iter().enumerate() {
                let size = element.abs();
                if size > largest {
                    largest = size;
                    steepest = i;
                }
            }
            (steepest, largest)
        })
    }

    /// [`scaled_solve`](Self::scaled_solve) for the [`STARTS`] vectors of `x`, one after the
    /// other, at once; `work` is as long as they are.
    fn scaled_solve_starts(
        &self,
        system: System,
        (inner, outer): (&[T::Real], &[T::Real]),
        x: &mut [T],
        v: &mut [T],
        work: &mut [T],
    ) {
        let n = self.n;
        let (first, then) = match system {
            System::Direct => (inner, outer),
            System::Transpose | System::ConjugateTranspose => (outer, inner),
        };
        for x in x.chunks_exact_mut(n) {
            for (element, &scale) in x.iter_mut().zip(first) {
                *element = element.scale(scale);
            }
        }
        self.solve_many_into::<STARTS>(system, x, v, work);
        for v in v.chunks_exact_mut(n) {
            for (element, &scale) in v.iter_mut().zip(then) {
                *element = element.scale(scale);
            }
        }
    }

    /// One step of the search of [`inverse_norm_estimate`](Self::inverse_norm_estimate): with
    /// `v = C·x` on entry, `C = D_o·A⁻¹·D_i`, the gradient of the norm at `x`,
    /// `Cᴴ·sign(C·x)`, left in `v`, and the place and the size of its largest element, the
    /// first of that size: the unit vector at that place is where the norm rises fastest. `x`
    /// and `work` are working memory.
    fn steepest(
        &self,
        scales: (&[T::Real], &[T::Real]),
        x: &mut [T],
        v: &mut [T],
        work: &mut [T],
    ) -> (usize, T::Real) {
        let zero = T::Real::ZERO;
        for (sign, &element) in x.iter_mut().zip(v.iter()) {
            let size = element.abs();
            *sign = if size > zero {
                T::from_parts(element.re() / size, element.im() / size)
            } else {
                T::one()
            };
        }
        self.scaled_solve(System::ConjugateTranspose, scales, x, v, work);
        let mut steepest = 0;
        let mut largest = -T::Real::ONE;
        for (i, element) in v.iter().enumerate() {
            let size = element.abs();
            if size > largest {
                largest = size;
                steepest = i;
            }
        }
        (steepest, largest)
    }

    /// Writes to `v` the product `D_o·A⁻¹·D_i·x` for [`System::Direct`], and its adjoint
    /// `D_i·A⁻ᴴ·D_o·x` for [`System::ConjugateTranspose`], `D_i` and `D_o` the diagonal matrices
    /// of `inner` and `outer` in `scales`. `x` is scaled in place and `work` is working memory;
    /// all are as long as the order of the matrix.
    fn scaled_solve(
        &self,
        system: System,
        (inner, outer): (&[T::Real], &[T::Real]),
        x: &mut [T],
        v: &mut [T],
        work: &mut [T],
    ) {
        let (first, then) = match system {
            System::Direct => (inner, outer),
            System::Transpose | System::ConjugateTranspose => (outer, inner),
        };
        for (element, &scale) in x.iter_mut().zip(first) {
            *element = element.scale(scale);
        }
        self.solve_into(system, x, v, work);
        for (element, &scale) in v.iter_mut().zip(then) {
            *element = element.scale(scale);
        }
    }

    /// The solutions of one of the three systems for the columns of `b`, or `None` if the matrix
    /// is singular or a solution isn't finite.
    #[track_caller]
    fn solve_columns(&self, system: System, b: &Matrix<T>) -> Option<Matrix<T>> {
        assert_eq!(
            b.rows(),
            self.n,
            "the right-hand side has not as many rows as the matrix"
        );
        if self.is_singular() {
            return None;
        }
        let mut x = Matrix::zeros(self.n, b.cols());
        let mut work = vec![T::zero(); self.n];
        for c in 0..b.cols() {
            self.solve_into(system, b.column(c), x.column_mut(c), &mut work);
        }
        x.as_slice().iter().all(|v| v.is_finite()).then_some(x)
    }

    /// The solution of one of the three systems for the right-hand side `b`, or `None` if the
    /// matrix is singular or the solution isn't finite.
    #[track_caller]
    fn solve_one(&self, system: System, b: &[T]) -> Option<Vec<T>> {
        assert_eq!(
            b.len(),
            self.n,
            "the right-hand side is not as long as the order of the matrix"
        );
        if self.is_singular() {
            return None;
        }
        let mut x = vec![T::zero(); self.n];
        let mut work = vec![T::zero(); self.n];
        self.solve_into(system, b, &mut x, &mut work);
        x.iter().all(|v| v.is_finite()).then_some(x)
    }

    /// Writes to `x` the solution of one of the three systems for the right-hand side `b`, with
    /// `work` as working memory; the three are as long as the order of the matrix, which isn't
    /// singular.
    fn solve_into(&self, system: System, b: &[T], x: &mut [T], work: &mut [T]) {
        let n = self.n;
        match system {
            System::Direct => {
                // `A = Pᵀ·L·U·Qᵀ`: `L·y = P·b`, then `U·z = y`, then `x = Q·z`.
                for (w, &old) in work.iter_mut().zip(&self.row_perm) {
                    *w = b[old];
                }
                // Both triangles by columns, as they are held: an element of the solution, once
                // known, is taken off the ones that depend on it.
                for j in 0..n {
                    let y = work[j];
                    if y.is_zero() {
                        continue;
                    }
                    for at in self.l_ptrs[j]..self.l_ptrs[j + 1] {
                        let i = self.l_rows[at];
                        work[i] = work[i].mul_sub(self.l_values[at], y);
                    }
                }
                for j in (0..n).rev() {
                    let z = work[j] / self.pivots[j];
                    work[j] = z;
                    if z.is_zero() {
                        continue;
                    }
                    for at in self.u_ptrs[j]..self.u_ptrs[j + 1] {
                        let i = self.u_rows[at];
                        work[i] = work[i].mul_sub(self.u_values[at], z);
                    }
                }
                for (&w, &old) in work.iter().zip(&self.col_perm) {
                    x[old] = w;
                }
            }
            System::Transpose | System::ConjugateTranspose => {
                // `Aᵀ = Q·Uᵀ·Lᵀ·P`: `Uᵀ·y = Qᵀ·b`, then `Lᵀ·z = y`, then `x = Pᵀ·z`. A column of
                // a factor is a row of its transpose, so each element of the solution is one
                // inner product with a column.
                let conjugate = T::IS_COMPLEX && system == System::ConjugateTranspose;
                let of = |element: T| {
                    if conjugate { element.conj() } else { element }
                };
                for (w, &old) in work.iter_mut().zip(&self.col_perm) {
                    *w = b[old];
                }
                for j in 0..n {
                    let mut sum = work[j];
                    for at in self.u_ptrs[j]..self.u_ptrs[j + 1] {
                        sum = sum.mul_sub(of(self.u_values[at]), work[self.u_rows[at]]);
                    }
                    work[j] = sum / of(self.pivots[j]);
                }
                for j in (0..n).rev() {
                    let mut sum = work[j];
                    for at in self.l_ptrs[j]..self.l_ptrs[j + 1] {
                        sum = sum.mul_sub(of(self.l_values[at]), work[self.l_rows[at]]);
                    }
                    work[j] = sum;
                }
                for (&w, &old) in work.iter().zip(&self.row_perm) {
                    x[old] = w;
                }
            }
        }
    }
}

impl<T: Scalar> SparseLu<T> {
    /// [`solve_into`](Self::solve_into) for `K` right-hand sides at once, one after the other in
    /// `b` and in `x`, with `work` as long as they are: one pass over the factors for all of
    /// them, each solution getting the arithmetic, operation for operation, that it gets alone.
    fn solve_many_into<const K: usize>(
        &self,
        system: System,
        b: &[T],
        x: &mut [T],
        work: &mut [T],
    ) {
        let n = self.n;
        debug_assert!(b.len() == K * n && x.len() == K * n && work.len() == K * n);
        // `work` holds the `K` elements of a row side by side.
        let gather = |work: &mut [T], perm: &[usize]| {
            for (i, &old) in perm.iter().enumerate() {
                for k in 0..K {
                    work[i * K + k] = b[k * n + old];
                }
            }
        };
        let scatter = |work: &[T], perm: &[usize], x: &mut [T]| {
            for (i, &old) in perm.iter().enumerate() {
                for k in 0..K {
                    x[k * n + old] = work[i * K + k];
                }
            }
        };
        match system {
            System::Direct => {
                gather(work, &self.row_perm);
                for j in 0..n {
                    let y: [T; K] = core::array::from_fn(|k| work[j * K + k]);
                    let nonzero = y.iter().filter(|y| !y.is_zero()).count();
                    if nonzero == 0 {
                        continue;
                    }
                    let (rows, values) = (
                        &self.l_rows[self.l_ptrs[j]..self.l_ptrs[j + 1]],
                        &self.l_values[self.l_ptrs[j]..self.l_ptrs[j + 1]],
                    );
                    for (&i, &l) in rows.iter().zip(values) {
                        let row = &mut work[i * K..(i + 1) * K];
                        for k in 0..K {
                            // A zero is skipped, as the solve of one vector skips it.
                            if nonzero == K || !y[k].is_zero() {
                                row[k] = row[k].mul_sub(l, y[k]);
                            }
                        }
                    }
                }
                for j in (0..n).rev() {
                    let z: [T; K] = core::array::from_fn(|k| work[j * K + k] / self.pivots[j]);
                    work[j * K..(j + 1) * K].copy_from_slice(&z);
                    let nonzero = z.iter().filter(|z| !z.is_zero()).count();
                    if nonzero == 0 {
                        continue;
                    }
                    let (rows, values) = (
                        &self.u_rows[self.u_ptrs[j]..self.u_ptrs[j + 1]],
                        &self.u_values[self.u_ptrs[j]..self.u_ptrs[j + 1]],
                    );
                    for (&i, &u) in rows.iter().zip(values) {
                        let row = &mut work[i * K..(i + 1) * K];
                        for k in 0..K {
                            if nonzero == K || !z[k].is_zero() {
                                row[k] = row[k].mul_sub(u, z[k]);
                            }
                        }
                    }
                }
                scatter(work, &self.col_perm, x);
            }
            System::Transpose | System::ConjugateTranspose => {
                let conjugate = T::IS_COMPLEX && system == System::ConjugateTranspose;
                let of = |element: T| {
                    if conjugate { element.conj() } else { element }
                };
                gather(work, &self.col_perm);
                for j in 0..n {
                    let mut sum: [T; K] = core::array::from_fn(|k| work[j * K + k]);
                    for at in self.u_ptrs[j]..self.u_ptrs[j + 1] {
                        let (u, i) = (of(self.u_values[at]), self.u_rows[at]);
                        for k in 0..K {
                            sum[k] = sum[k].mul_sub(u, work[i * K + k]);
                        }
                    }
                    let pivot = of(self.pivots[j]);
                    for k in 0..K {
                        work[j * K + k] = sum[k] / pivot;
                    }
                }
                for j in (0..n).rev() {
                    let mut sum: [T; K] = core::array::from_fn(|k| work[j * K + k]);
                    for at in self.l_ptrs[j]..self.l_ptrs[j + 1] {
                        let (l, i) = (of(self.l_values[at]), self.l_rows[at]);
                        for k in 0..K {
                            sum[k] = sum[k].mul_sub(l, work[i * K + k]);
                        }
                    }
                    work[j * K..(j + 1) * K].copy_from_slice(&sum);
                }
                scatter(work, &self.row_perm, x);
            }
        }
    }
}

/// Which system a solve is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum System {
    /// `A·x = b`.
    Direct,
    /// `Aᵀ·x = b`.
    Transpose,
    /// `Aᴴ·x = b`.
    ConjugateTranspose,
}

/// What an elimination leaves: the factors, with the rows of `L` still rows of `A`, and what it
/// found of them.
struct Eliminated<T: Scalar> {
    /// Where each column of `L` begins in `l_rows` and `l_values`, and where the last ends.
    l_ptrs: Vec<usize>,
    /// The rows of `A` of the elements of `L` below its diagonal.
    l_rows: Vec<usize>,
    /// The elements of `L` below its diagonal.
    l_values: Vec<T>,
    /// Where each column of `U` begins in `u_rows` and `u_values`, and where the last ends.
    u_ptrs: Vec<usize>,
    /// The steps of the pivotal rows of the elements of `U` above its diagonal.
    u_rows: Vec<usize>,
    /// The elements of `U` above its diagonal.
    u_values: Vec<T>,
    /// The pivots: the diagonal of `U`.
    pivots: Vec<T>,
    /// The step at which each row of `A` became the pivotal one.
    step_of_row: Vec<usize>,
    /// The pivotal row of each step.
    row_perm: Vec<usize>,
    /// Whether a column was left without a pivot.
    deficient: bool,
    /// Whether every element of the factors is finite.
    finite: bool,
    /// The largest absolute value of an element of `U`.
    largest_u: T::Real,
}

impl<T: Scalar> From<Elimination<T>> for Eliminated<T> {
    fn from(work: Elimination<T>) -> Self {
        Self {
            l_ptrs: work.l_ptrs,
            l_rows: work.l_rows,
            l_values: work.l_values,
            u_ptrs: work.u_ptrs,
            u_rows: work.u_rows,
            u_values: work.u_values,
            pivots: work.pivots,
            step_of_row: work.step_of_row,
            row_perm: work.row_perm,
            deficient: work.deficient,
            finite: work.finite,
            largest_u: work.largest_u,
        }
    }
}

/// How the pivots of an elimination are chosen, as the strategy gives it, before its fractions are
/// rounded to the real type of the elements.
#[derive(Clone, Copy, Debug)]
struct Rule {
    /// The fraction of the largest element of its column the diagonal element must reach to be
    /// the pivot.
    diagonal: f64,
    /// Below 1, the fraction of the largest an element off the diagonal must reach to be the
    /// pivot, which [`Pivoting::off_diagonal`] then chooses among; at 1, the largest element is
    /// the pivot, the lowest row of several.
    off: f64,
}

impl Rule {
    /// Threshold partial pivoting with the threshold `threshold`.
    fn partial(threshold: f64) -> Self {
        Self {
            diagonal: threshold,
            off: 1.0,
        }
    }

    /// The rule of the symmetric strategy: the diagonal down to [`SYMMETRIC_DIAGONAL`], else an
    /// element of at least [`SYMMETRIC_OFF_DIAGONAL`] of the largest.
    fn symmetric() -> Self {
        Self {
            diagonal: SYMMETRIC_DIAGONAL,
            off: SYMMETRIC_OFF_DIAGONAL,
        }
    }
}

/// A [`Rule`] in the real type `R` of the elements, with what its choice off the diagonal reads.
pub(super) struct Pivoting<'p, R> {
    /// The fraction of the largest element of its column the diagonal element must reach.
    pub(super) diagonal: R,
    /// The fraction of the largest an element off the diagonal must reach, below 1; at 1 the
    /// largest is taken.
    pub(super) off: R,
    /// The step of each column of `A` when `off` is below 1, else empty.
    step_of_column: &'p [usize],
}

impl<R: Float> Pivoting<'_, R> {
    /// The pivotal row of step `j` when its diagonal element is not taken: `lowest`, the lowest
    /// row of the largest size `largest`, if `off` is 1; else, of the rows in `free` — the rows
    /// of the column that are not pivotal yet, with the sizes of their elements — whose element
    /// is at least `off` times the largest, one whose own column of `A` is eliminated already,
    /// if there is one, the largest of those (the lowest row of several); else `lowest`.
    ///
    /// A row whose column is eliminated already is one whose diagonal element some earlier step
    /// passed over, and it can be the pivot of no later column on the diagonal: taking it leaves
    /// the diagonal of every later column where the ordering expects its pivot, and the pivot it
    /// takes, of the row of the column of step `j`, is what a pivot of two rows and two columns
    /// on the diagonal would give.
    pub(super) fn off_diagonal(
        &self,
        j: usize,
        largest: R,
        lowest: usize,
        free: impl Iterator<Item = (usize, R)>,
    ) -> usize {
        if self.off >= R::ONE {
            return lowest;
        }
        let bar = self.off * largest;
        let mut best = NONE;
        let mut best_size = R::ZERO;
        for (row, size) in free {
            if size >= bar
                && self.step_of_column[row] < j
                && (best == NONE || size > best_size || (size == best_size && row < best))
            {
                best = row;
                best_size = size;
            }
        }
        if best == NONE { lowest } else { best }
    }
}

/// The elimination while it runs: the factors as far as they are found, and the working memory of
/// a step. Until the end, a row is known by its number in `A`: where it goes in `P·A` is decided
/// only when a column takes it as its pivot.
struct Elimination<T: Scalar> {
    /// Where each column of `L` found so far begins, and where the last ends.
    l_ptrs: Vec<usize>,
    /// The rows of `A` of the elements of `L` below its diagonal.
    l_rows: Vec<usize>,
    /// The elements of `L` below its diagonal.
    l_values: Vec<T>,
    /// Where each column of `U` found so far begins, and where the last ends.
    u_ptrs: Vec<usize>,
    /// The rows of `U` of its elements above the diagonal: the steps of their pivotal rows.
    u_rows: Vec<usize>,
    /// The elements of `U` above its diagonal.
    u_values: Vec<T>,
    /// The pivots of the steps so far: the diagonal of `U`.
    pivots: Vec<T>,
    /// The step at which each row of `A` became the pivotal one, or [`NONE`].
    step_of_row: Vec<usize>,
    /// The pivotal row of each step so far.
    row_perm: Vec<usize>,
    /// Where the part of each column of `L` ends that a search follows: the whole column until
    /// it is pruned, then the rows that were pivotal by then, moved to its front.
    followed: Vec<usize>,
    /// Whether each column of `L` has been pruned.
    pruned: Vec<bool>,
    /// The column being eliminated, by rows of `A`; zero between the steps.
    x: Vec<T>,
    /// The step in which a search last reached each row.
    reached: Vec<usize>,
    /// The rows a search is in the middle of, the deepest last.
    path: Vec<usize>,
    /// For each row on the path, where in its column of `L` the search goes on from.
    resume: Vec<usize>,
    /// From the position a search returns to the end: the rows it reached, each after every row
    /// it depends on.
    order: Vec<usize>,
    /// No row before this one is without a step.
    free: usize,
    /// Whether a column was left without a pivot.
    deficient: bool,
    /// Whether every element of the factors so far is finite.
    finite: bool,
    /// The largest absolute value of an element of `U` so far.
    largest_u: T::Real,
}

impl<T: Scalar> Elimination<T> {
    /// The elimination of a matrix of order `n` with `nnz` elements, before its first step.
    fn new(n: usize, nnz: usize) -> Self {
        // The factors hold at least the elements of the matrix, shared between the two of them
        // and the pivots.
        let room = nnz / 2;
        let mut l_ptrs = Vec::with_capacity(n + 1);
        let mut u_ptrs = Vec::with_capacity(n + 1);
        l_ptrs.push(0);
        u_ptrs.push(0);
        Self {
            l_ptrs,
            l_rows: Vec::with_capacity(room),
            l_values: Vec::with_capacity(room),
            u_ptrs,
            u_rows: Vec::with_capacity(room),
            u_values: Vec::with_capacity(room),
            pivots: Vec::with_capacity(n),
            step_of_row: vec![NONE; n],
            row_perm: Vec::with_capacity(n),
            followed: Vec::with_capacity(n),
            pruned: Vec::with_capacity(n),
            x: vec![T::zero(); n],
            reached: vec![NONE; n],
            path: vec![0; n],
            resume: vec![0; n],
            order: vec![0; n],
            free: 0,
            deficient: false,
            finite: true,
            largest_u: T::Real::ZERO,
        }
    }

    /// Step `j` of the elimination: column `j` of the factors from the column of `A` whose
    /// elements are `values`, in the rows `rows`. `diagonal` is the row whose element is on the
    /// diagonal of `A` in that column, and `pivoting` says what it must reach to be the pivot and
    /// which row is the pivot when it does not.
    fn step(
        &mut self,
        j: usize,
        diagonal: usize,
        rows: &[usize],
        values: &[T],
        pivoting: &Pivoting<'_, T::Real>,
    ) {
        let top = self.reach(rows, j);
        for (&i, &value) in rows.iter().zip(values) {
            self.x[i] = value;
        }
        // The triangular solve with the columns of `L` so far, over the rows reached alone and in
        // an order that has every row settled before it is used.
        let n = self.x.len();
        debug_assert!(self.l_rows.iter().all(|&row| row < n));
        let x = &mut self.x;
        for &i in &self.order[top..] {
            let k = self.step_of_row[i];
            if k == NONE {
                continue;
            }
            let u = x[i];
            let (begin, end) = (self.l_ptrs[k], self.l_ptrs[k + 1]);
            for (&row, &l) in self.l_rows[begin..end]
                .iter()
                .zip(&self.l_values[begin..end])
            {
                // SAFETY: every row of `L` is a row of the matrix, below `n`, the length of
                // `x`: the rows come from the column of `A` and from earlier columns of `L`.
                unsafe {
                    let target = x.get_unchecked_mut(row);
                    *target = target.mul_sub(l, u);
                }
            }
        }
        // The largest of the rows that are not pivotal yet, the lowest row of that size. A size
        // that is not a number is never the largest.
        let zero = T::Real::ZERO;
        let mut largest = -T::Real::ONE;
        let mut lowest = NONE;
        for &i in &self.order[top..] {
            if self.step_of_row[i] != NONE {
                continue;
            }
            let size = self.x[i].abs1();
            if size > largest || (size == largest && i < lowest) {
                largest = size;
                lowest = i;
            }
        }
        let diagonal_is_free = self.step_of_row[diagonal] == NONE;
        let pivot_row = if largest > zero {
            // A row of `x` outside the reach is zero, so the diagonal passes only from within.
            let size = self.x[diagonal].abs1();
            if diagonal_is_free && size > zero && size >= pivoting.diagonal * largest {
                diagonal
            } else {
                let (x, step_of_row) = (&self.x, &self.step_of_row);
                let free = self.order[top..]
                    .iter()
                    .filter(|&&i| step_of_row[i] == NONE)
                    .map(|&i| (i, x[i].abs1()));
                pivoting.off_diagonal(j, largest, lowest, free)
            }
        } else {
            // Nothing to divide by: the matrix is singular. Some row must still take the step,
            // and any that is not pivotal will do, since all that is left of the column in such
            // rows is zero.
            self.deficient = true;
            if diagonal_is_free {
                diagonal
            } else if lowest != NONE {
                lowest
            } else {
                while self.step_of_row[self.free] != NONE {
                    self.free += 1;
                }
                self.free
            }
        };
        let pivot = self.x[pivot_row];
        // As the dense decomposition does: by the reciprocal, unless that overflows.
        let reciprocal = T::one() / pivot;
        let multiply = reciprocal.is_finite();
        let mut finite = pivot.is_finite();
        let mut largest_u = self.largest_u.max(pivot.abs());
        for &i in &self.order[top..] {
            let value = self.x[i];
            self.x[i] = T::zero();
            let k = self.step_of_row[i];
            if k != NONE {
                finite &= value.is_finite();
                largest_u = largest_u.max(value.abs());
                self.u_rows.push(k);
                self.u_values.push(value);
            } else if i != pivot_row {
                // A row the search reached is kept though its element be zero: the pruning
                // counts on finding it. A zero is not divided, so that a pivot of zero leaves it.
                let element = if value.is_zero() {
                    value
                } else if multiply {
                    value * reciprocal
                } else {
                    value / pivot
                };
                finite &= element.is_finite();
                self.l_rows.push(i);
                self.l_values.push(element);
            }
        }
        self.finite &= finite;
        self.largest_u = largest_u;
        self.pivots.push(pivot);
        self.step_of_row[pivot_row] = j;
        self.row_perm.push(pivot_row);
        self.u_ptrs.push(self.u_rows.len());
        self.l_ptrs.push(self.l_rows.len());
        self.followed.push(self.l_rows.len());
        self.pruned.push(false);
        self.prune(j, pivot_row);
    }

    /// The rows in which the solution of the triangular system with the columns of `L` so far
    /// can be nonzero, for a right-hand side that is nonzero in `roots`: those reachable from
    /// `roots` along the columns of `L`, by a depth-first search that follows a pivotal row to
    /// the rows of its column. They are left in `order` from the position returned to the end,
    /// in the reverse of the order the search finished them in, which has every row after the
    /// rows it depends on. `stamp` marks the rows of this search and differs from that of every
    /// search before.
    ///
    /// The search keeps its own stack, so a chain of dependencies as long as the matrix costs
    /// memory and no recursion.
    fn reach(&mut self, roots: &[usize], stamp: usize) -> usize {
        let mut top = self.order.len();
        for &root in roots {
            if self.reached[root] == stamp {
                continue;
            }
            self.reached[root] = stamp;
            self.path[0] = root;
            self.resume[0] = self.first_followed(root);
            let mut depth = 0;
            loop {
                let row = self.path[depth];
                let end = self.end_followed(row);
                let mut at = self.resume[depth];
                let mut next = NONE;
                while at < end {
                    let candidate = self.l_rows[at];
                    at += 1;
                    if self.reached[candidate] != stamp {
                        next = candidate;
                        break;
                    }
                }
                self.resume[depth] = at;
                if next == NONE {
                    // Everything this row leads to is in the list: the row goes before it.
                    top -= 1;
                    self.order[top] = row;
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                } else {
                    self.reached[next] = stamp;
                    depth += 1;
                    self.path[depth] = next;
                    self.resume[depth] = self.first_followed(next);
                }
            }
        }
        top
    }

    /// Where the rows a search follows from `row` begin in `l_rows`.
    fn first_followed(&self, row: usize) -> usize {
        match self.step_of_row[row] {
            NONE => 0,
            k => self.l_ptrs[k],
        }
    }

    /// Where the rows a search follows from `row` end in `l_rows`: none are followed from a row
    /// that is not pivotal.
    fn end_followed(&self, row: usize) -> usize {
        match self.step_of_row[row] {
            NONE => 0,
            k => self.followed[k],
        }
    }

    /// The symmetric pruning of Eisenstat and Liu after step `j`, whose pivotal row is
    /// `pivot_row`. For a column `k` with an element in row `k` of column `j` of `U`, the search
    /// of this step went through the whole of column `k` of `L`, so each of its rows that is
    /// still not pivotal is now in column `j` of `L`. If `pivot_row` is in column `k` as well,
    /// a later search that comes to `k` gets to all those rows through `j`, and need follow from
    /// `k` only the rows that are pivotal by now: they are moved to the front of the column.
    ///
    /// The scan of a column costs no more than the elimination of column `j` by it did, and a
    /// column is pruned once.
    fn prune(&mut self, j: usize, pivot_row: usize) {
        for at in self.u_ptrs[j]..self.u_ptrs[j + 1] {
            let k = self.u_rows[at];
            if self.pruned[k] {
                continue;
            }
            let (begin, end) = (self.l_ptrs[k], self.l_ptrs[k + 1]);
            if !self.l_rows[begin..end].contains(&pivot_row) {
                continue;
            }
            let mut kept = begin;
            for at in begin..end {
                if self.step_of_row[self.l_rows[at]] != NONE {
                    self.l_rows.swap(kept, at);
                    self.l_values.swap(kept, at);
                    kept += 1;
                }
            }
            self.followed[k] = kept;
            self.pruned[k] = true;
        }
    }
}

/// A triangular factor held by columns without its diagonal — column `j` in
/// `ptrs[j]..ptrs[j + 1]` of `rows` and `values`, its rows in any order — as a sparse matrix,
/// with `diagonal` on the diagonal.
fn by_rows<T: Scalar>(
    n: usize,
    ptrs: &[usize],
    rows: &[usize],
    values: &[T],
    diagonal: &[T],
) -> SparseMatrix<T> {
    // A counting sort by row. The columns are gone through in ascending order, so each row comes
    // out in ascending order too, the diagonal in its place: column `j` is the first with
    // anything in row `j` of an upper triangle, and the last of a lower one.
    let mut row_ptrs = vec![0; n + 1];
    for &i in rows {
        row_ptrs[i + 1] += 1;
    }
    for i in 0..n {
        row_ptrs[i + 1] += row_ptrs[i] + 1;
    }
    let mut next = row_ptrs.clone();
    let mut col_indices = vec![0; row_ptrs[n]];
    let mut elements = vec![T::zero(); row_ptrs[n]];
    for j in 0..n {
        col_indices[next[j]] = j;
        elements[next[j]] = diagonal[j];
        next[j] += 1;
        for at in ptrs[j]..ptrs[j + 1] {
            let i = rows[at];
            col_indices[next[i]] = j;
            elements[next[i]] = values[at];
            next[i] += 1;
        }
    }
    SparseMatrix::from_csr(n, n, row_ptrs, col_indices, elements)
}

/// Checks that `p` is a permutation of `0..n`, which an ordering of the columns must be.
///
/// # Panics
///
/// Panics if it isn't.
#[track_caller]
fn assert_permutation(p: &[usize], n: usize) {
    assert_eq!(
        p.len(),
        n,
        "the column ordering isn't a permutation of the columns"
    );
    let mut seen = vec![false; n];
    for &old in p {
        assert!(
            old < n && !seen[old],
            "the column ordering isn't a permutation of the columns"
        );
        seen[old] = true;
    }
}

/// Whether the pattern of the square `a` holds `n` nonzeros in distinct rows and columns: a
/// perfect matching of its columns to its rows, without which the matrix is singular whatever
/// its elements are. `by_columns` is the transpose of `a`, whose rows are the columns of `a`, and
/// `row_perm`, `col_perm` are the pivotal rows and the columns of the elimination, step by step.
///
/// The pivots that are elements of `a` are a matching already, and as a rule a complete one.
/// What they leave is matched first to a free row of the column, if it has one, and then by
/// augmenting paths in the phases of Hopcroft and Karp: a breadth-first search from the
/// unmatched columns along alternating paths finds the length of the shortest augmenting path,
/// and depth-first searches confined to its layers find a maximal set of such paths without a
/// vertex in common, all of which are applied. A phase costs `O(n + nnz)`, and `O(√n)` phases
/// reach a maximum matching; a phase that finds no augmenting path at all proves the matrix
/// structurally singular.
fn has_perfect_matching<T: Scalar>(
    a: &SparseMatrix<T>,
    by_columns: &SparseMatrix<T>,
    row_perm: &[usize],
    col_perm: &[usize],
) -> bool {
    let n = a.rows();
    let mut row_of_column = vec![NONE; n];
    let mut column_of_row = vec![NONE; n];
    let mut matched = 0;
    for (&row, &column) in row_perm.iter().zip(col_perm) {
        if a.find(row, column).is_some() {
            row_of_column[column] = row;
            column_of_row[row] = column;
            matched += 1;
        }
    }
    if matched == n {
        return true;
    }
    let starts = by_columns.row_pointers();
    let rows = by_columns.col_indices();
    for column in 0..n {
        if row_of_column[column] != NONE {
            continue;
        }
        let free = rows[starts[column]..starts[column + 1]]
            .iter()
            .find(|&&row| column_of_row[row] == NONE);
        if let Some(&row) = free {
            row_of_column[column] = row;
            column_of_row[row] = column;
            matched += 1;
        }
    }
    // The layer of each column in the search of a phase; `NONE` for one outside the layers, or
    // one that a path of the phase has used or found to lead nowhere.
    let mut layer = vec![NONE; n];
    // Where the depth-first search goes on in each column's rows.
    let mut next = vec![0; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    let mut stack: Vec<usize> = Vec::new();
    while matched < n {
        // The layers: the unmatched columns, then the columns matched to their rows, and so on,
        // up to the first layer with a column that has an unmatched row.
        layer.fill(NONE);
        queue.clear();
        for (column, &row) in row_of_column.iter().enumerate() {
            if row == NONE {
                layer[column] = 0;
                queue.push(column);
            }
        }
        let mut shortest = NONE;
        let mut head = 0;
        while head < queue.len() {
            let column = queue[head];
            head += 1;
            let depth = layer[column];
            if depth >= shortest {
                break;
            }
            for &row in &rows[starts[column]..starts[column + 1]] {
                let owner = column_of_row[row];
                if owner == NONE {
                    shortest = depth;
                } else if layer[owner] == NONE {
                    layer[owner] = depth + 1;
                    queue.push(owner);
                }
            }
        }
        if shortest == NONE {
            // No unmatched column has an augmenting path: the matching is a maximum one, and
            // smaller than the order.
            return false;
        }
        next.copy_from_slice(&starts[..n]);
        for start in 0..n {
            if row_of_column[start] != NONE || layer[start] != 0 {
                continue;
            }
            stack.clear();
            stack.push(start);
            while let Some(&column) = stack.last() {
                let depth = layer[column];
                let end = starts[column + 1];
                let mut advanced = false;
                while next[column] < end {
                    let row = rows[next[column]];
                    next[column] += 1;
                    let owner = column_of_row[row];
                    if owner == NONE {
                        if depth == shortest {
                            // Each column of the path takes the row it went on by, the last
                            // one the free row, and no other path of the phase may use them.
                            for &on_path in &stack {
                                let taken = rows[next[on_path] - 1];
                                row_of_column[on_path] = taken;
                                column_of_row[taken] = on_path;
                                layer[on_path] = NONE;
                            }
                            matched += 1;
                            stack.clear();
                            advanced = true;
                            break;
                        }
                    } else if depth < shortest && layer[owner] == depth + 1 {
                        stack.push(owner);
                        advanced = true;
                        break;
                    }
                }
                if !advanced {
                    // Nothing on from here reaches a free row in this phase.
                    layer[column] = NONE;
                    stack.pop();
                }
            }
        }
    }
    true
}

/// Whether the permutation `p` is an odd one: it is as many transpositions as its cycles have
/// elements beyond their first.
fn is_odd(p: &[usize]) -> bool {
    let mut seen = vec![false; p.len()];
    let mut odd = false;
    for start in 0..p.len() {
        let mut i = p[start];
        seen[start] = true;
        while !seen[i] {
            seen[i] = true;
            odd = !odd;
            i = p[i];
        }
    }
    odd
}
