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
//! # What is not done
//!
//! The columns are eliminated one at a time: there are no supernodes and no dense blocks, the
//! matrix is not permuted to block triangular form first, its rows and columns are not scaled
//! for the elimination (only for the test of its condition), and a solution is not refined. The
//! elimination is sequential.
//!
//! The limits measured, of the test of singularity and of pivoting on unscaled rows and
//! columns among them, are listed in the documentation of [`SparseLu`].
//!
//! [`pivot_threshold`]: SparseLuOptions::pivot_threshold

use super::{Ordering, SparseMatrix};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};

/// The step of a row that no column has taken as its pivot yet, and the row of a search that has
/// found none.
const NONE: usize = usize::MAX;

/// How often the estimate of [`SparseLu::rcond`] may move to a better unit vector, as in LAPACK.
const ESTIMATE_CYCLES: usize = 5;

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

/// How [`SparseLu`] orders the columns of a matrix and how it chooses its pivots.
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
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SparseLuOptions {
    /// The ordering of the columns: [`Ordering::Colamd`] unless set.
    pub ordering: Ordering,
    /// The fraction of the largest element of its column the diagonal element must reach to be
    /// taken as the pivot, in `(0, 1]`: 1 unless set, which is partial pivoting.
    pub pivot_threshold: f64,
}

impl Default for SparseLuOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseLuOptions {
    /// The defaults: the columns ordered by [`Ordering::Colamd`], and a pivot threshold of 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLuOptions};
    ///
    /// let options = SparseLuOptions::new();
    /// assert_eq!(options.ordering, Ordering::Colamd);
    /// assert_eq!(options.pivot_threshold, 1.0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            ordering: Ordering::Colamd,
            pivot_threshold: 1.0,
        }
    }

    /// Sets the ordering of the columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLuOptions};
    ///
    /// let options = SparseLuOptions::new().with_ordering(Ordering::Given(vec![1, 0]));
    /// assert_eq!(options.ordering, Ordering::Given(vec![1, 0]));
    /// ```
    #[must_use]
    pub fn with_ordering(mut self, ordering: Ordering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Sets the pivot threshold: the diagonal element of a column is its pivot when it is at
    /// least this fraction of the largest element the column offers. It must lie in `(0, 1]`,
    /// which [`SparseLu::with_options`] checks.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseLuOptions;
    ///
    /// let options = SparseLuOptions::new().with_pivot_threshold(0.1);
    /// assert_eq!(options.pivot_threshold, 0.1);
    /// ```
    #[must_use]
    pub fn with_pivot_threshold(mut self, pivot_threshold: f64) -> Self {
        self.pivot_threshold = pivot_threshold;
        self
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

impl<T: Scalar> SparseLu<T> {
    /// The decomposition of the square `a`, with its columns ordered by [`Ordering::Colamd`] and
    /// its pivots chosen by partial pivoting. Every square matrix has one; one that is singular,
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
        Self::with_ordering(a, &Ordering::Colamd)
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
        Self::factor(a, ordering.columns(a), T::Real::ONE)
    }

    /// The decomposition of the square `a` with the column ordering and the pivot threshold of
    /// `options`, or `None` if the threshold isn't in `(0, 1]`.
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
        let columns = options.ordering.columns(a);
        Some(Self::factor(a, columns, T::Real::from_f64(threshold)))
    }

    /// Factors `a` with its columns in the order `col_perm`, `col_perm[new] = old`.
    #[track_caller]
    fn factor(a: &SparseMatrix<T>, col_perm: Vec<usize>, threshold: T::Real) -> Self {
        let n = a.rows();
        assert_permutation(&col_perm, n);
        // The matrix by columns: the rows of its transpose.
        let by_columns = a.transpose();
        let starts = by_columns.row_pointers();
        let rows = by_columns.col_indices();
        let values = by_columns.values();
        let mut work = Elimination::new(n, a.nnz());
        let mut largest_a = T::Real::ZERO;
        let mut norm_one = T::Real::ZERO;
        for (j, &column) in col_perm.iter().enumerate() {
            let (begin, end) = (starts[column], starts[column + 1]);
            let mut sum = T::Real::ZERO;
            for x in &values[begin..end] {
                let size = x.abs();
                largest_a = largest_a.max(size);
                sum += size;
            }
            norm_one = norm_one.max(sum);
            work.step(j, column, &rows[begin..end], &values[begin..end], threshold);
        }
        // The rows of `L` are known by their place in `P·A` only now that every row has one.
        for row in &mut work.l_rows {
            *row = work.step_of_row[*row];
        }
        // A pivot that rounding alone kept from zero must not hide a matrix whose pattern makes it
        // singular. A matrix already found singular needs no more evidence.
        let deficient = work.deficient
            || (work.finite && !has_perfect_matching(a, &by_columns, &work.row_perm, &col_perm));
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
            lu.near_singular = !lu.is_regular_to_working_precision(a, &by_columns);
        }
        lu
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
        let mut x = vec![T::from_real(T::Real::ONE / order); n];
        let mut v = vec![T::zero(); n];
        let mut work = vec![T::zero(); n];
        self.scaled_solve(System::Direct, scales, &mut x, &mut v, &mut work);
        let mut estimate = norm(&v);
        if n == 1 {
            return estimate;
        }
        let mut last = NONE;
        for _ in 0..ESTIMATE_CYCLES {
            let (steepest, largest) = self.steepest(scales, &mut x, &mut v, &mut work);
            // The vector tried last is still as steep as any: there is nothing better to try.
            if last != NONE && v[last].abs() >= largest {
                break;
            }
            last = steepest;
            x.fill(T::zero());
            x[steepest] = T::one();
            self.scaled_solve(System::Direct, scales, &mut x, &mut v, &mut work);
            let next = norm(&v);
            if next <= estimate {
                break;
            }
            estimate = next;
        }
        // Elements of alternating signs that grow from 1 to 2, whose 1-norm is 3n/2.
        for (i, element) in x.iter_mut().enumerate() {
            let size = T::Real::ONE + T::Real::from_f64(i as f64) / (order - T::Real::ONE);
            *element = T::from_real(if i % 2 == 0 { size } else { -size });
        }
        self.scaled_solve(System::Direct, scales, &mut x, &mut v, &mut work);
        let alternating = T::Real::TWO * norm(&v) / (T::Real::from_f64(3.0) * order);
        let (steepest, _) = self.steepest(scales, &mut x, &mut v, &mut work);
        x.fill(T::zero());
        x[steepest] = T::one();
        self.scaled_solve(System::Direct, scales, &mut x, &mut v, &mut work);
        let estimate = estimate.max(alternating).max(norm(&v));
        // Elements of pseudo-random signs and sizes in `[1, 2)`, the same for every matrix.
        let mut state = RANDOM_START;
        let mut total = zero;
        for element in x.iter_mut() {
            let bits = split_mix(&mut state);
            // The size from the 53 high bits, the sign from the lowest.
            let size = T::Real::ONE + T::Real::from_f64((bits >> 11) as f64 / (1u64 << 53) as f64);
            total += size;
            *element = T::from_real(if bits & 1 == 0 { size } else { -size });
        }
        self.scaled_solve(System::Direct, scales, &mut x, &mut v, &mut work);
        let random = norm(&v) / total;
        let (steepest, _) = self.steepest(scales, &mut x, &mut v, &mut work);
        x.fill(T::zero());
        x[steepest] = T::one();
        self.scaled_solve(System::Direct, scales, &mut x, &mut v, &mut work);
        estimate.max(random).max(norm(&v))
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
    /// diagonal of `A` in that column, and `threshold` the fraction of the largest element it
    /// must reach to be the pivot.
    fn step(
        &mut self,
        j: usize,
        diagonal: usize,
        rows: &[usize],
        values: &[T],
        threshold: T::Real,
    ) {
        let top = self.reach(rows, j);
        for (&i, &value) in rows.iter().zip(values) {
            self.x[i] = value;
        }
        // The triangular solve with the columns of `L` so far, over the rows reached alone and in
        // an order that has every row settled before it is used.
        for &i in &self.order[top..] {
            let k = self.step_of_row[i];
            if k == NONE {
                continue;
            }
            let u = self.x[i];
            for at in self.l_ptrs[k]..self.l_ptrs[k + 1] {
                let row = self.l_rows[at];
                self.x[row] = self.x[row].mul_sub(self.l_values[at], u);
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
            if diagonal_is_free && size > zero && size >= threshold * largest {
                diagonal
            } else {
                lowest
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
