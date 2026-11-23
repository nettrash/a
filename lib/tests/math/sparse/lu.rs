//! The sparse LU decomposition: the factors against the matrix they came from and against a plain
//! elimination on dense columns, the pivoting rule, the solves with the matrix, its transpose and
//! its conjugate transpose against the dense decomposition and against SciPy, determinants,
//! singular matrices and matrices that are not finite, and the shapes at the edges.

use super::common::{Num, Rng, assert_csr, each_type, magnitude, random_sparse, size};
use a::math::complex::{Complex, Float};
use a::math::matrix::{Matrix, Norm};
use a::math::sparse::{Ordering, SparseLu, SparseLuOptions, SparseMatrix};

/// A system that SciPy and NumPy have solved: `data/gen_lu.py` writes them to
/// `data/lu_reference.rs`.
struct Reference {
    /// What the matrix is.
    name: &'static str,
    /// Its order.
    n: usize,
    /// Whether it has elements that are not real.
    complex: bool,
    /// Its elements, as `(row, column, real part, imaginary part)`.
    entries: &'static [(usize, usize, f64, f64)],
    /// A right-hand side `b`, as `(real part, imaginary part)`.
    rhs: &'static [(f64, f64)],
    /// The solution of `A·x = b`, by `scipy.sparse.linalg.spsolve`.
    x: &'static [(f64, f64)],
    /// The solution of `Aᵀ·x = b`, by `scipy.sparse.linalg.splu`.
    xt: &'static [(f64, f64)],
    /// The solution of `Aᴴ·x = b`, by `scipy.sparse.linalg.splu`.
    xh: &'static [(f64, f64)],
    /// The determinant, by `numpy.linalg.det`.
    det: (f64, f64),
    /// The reciprocal of the condition number in the 1-norm, by `numpy.linalg.cond`.
    rcond: f64,
    /// The rows in the order SuperLU pivots them into with the columns left in place, partial
    /// pivoting and no scaling: `perm[new] = old`.
    perm: &'static [usize],
    /// The elements of `L` below its diagonal and of `U` that the elimination with those pivots
    /// reaches, by a symbolic elimination in which nothing cancels.
    fill: usize,
}

/// A thin random pattern and its structural rank by `scipy.sparse.csgraph.structural_rank`,
/// from `data/gen_lu.py`.
struct Structural {
    /// The order.
    n: usize,
    /// The elements, `(row, column, value)`.
    entries: &'static [(usize, usize, f64)],
    /// The largest number of nonzeros in distinct rows and columns.
    rank: usize,
}

include!("data/lu_reference.rs");

/// An exactly singular matrix of signs that the estimate of the condition once missed, with the
/// column order it was missed with: `data/gen_lu_singular_cases.py` writes them to
/// `data/lu_singular_cases.rs`.
struct SignCase {
    /// Where it was found: its number, its family in the campaign of review round 2, the type it
    /// was missed in, and its order.
    name: &'static str,
    /// The column order, `q[new] = old`.
    order: &'static [usize],
    /// The rows, an element a character: `+` for 1, `-` for −1 and `.` for 0.
    rows: &'static [&'static str],
}

include!("data/lu_singular_cases.rs");

/// The element `re + im·i` of the type, the real part alone for a real type.
fn element<T: Num>(re: f64, im: f64) -> T {
    T::from_parts(
        <T::Real as Float>::from_f64(re),
        <T::Real as Float>::from_f64(im),
    )
}

/// A column of elements given as pairs of parts.
fn column<T: Num>(pairs: &[(f64, f64)]) -> Matrix<T> {
    Matrix::from_fn(pairs.len(), 1, |i, _| element(pairs[i].0, pairs[i].1))
}

/// The matrix of a reference case.
fn reference_matrix<T: Num>(case: &Reference) -> SparseMatrix<T> {
    let entries: Vec<(usize, usize, T)> = case
        .entries
        .iter()
        .map(|&(i, j, re, im)| (i, j, element(re, im)))
        .collect();
    SparseMatrix::from_coo(case.n, case.n, &entries)
}

/// The structural rank of a square matrix: the largest number of its nonzeros that lie in
/// distinct rows and columns, by Kuhn's augmenting paths, one search per column. Slow, and
/// plainly right, which is what an oracle should be.
fn structural_rank<T: Num>(a: &SparseMatrix<T>) -> usize {
    fn augment(
        column: usize,
        columns: &[Vec<usize>],
        owner: &mut [usize],
        seen: &mut [bool],
    ) -> bool {
        for &row in &columns[column] {
            if !seen[row] {
                seen[row] = true;
                if owner[row] == usize::MAX || augment(owner[row], columns, owner, seen) {
                    owner[row] = column;
                    return true;
                }
            }
        }
        false
    }
    let n = a.rows();
    let t = a.transpose();
    let columns: Vec<Vec<usize>> = (0..n).map(|j| t.row(j).map(|(i, _)| i).collect()).collect();
    let mut owner = vec![usize::MAX; n];
    (0..n)
        .filter(|&column| augment(column, &columns, &mut owner, &mut vec![false; n]))
        .count()
}

/// A random permutation of `0..n`.
fn shuffled(r: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, r.below(i + 1));
    }
    p
}

/// Whether `p` holds every index below `n` once.
fn is_permutation(p: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    p.len() == n
        && p.iter()
            .all(|&i| i < n && !core::mem::replace(&mut seen[i], true))
}

/// Whether the permutation is an odd one, by counting its inversions.
fn is_odd(p: &[usize]) -> bool {
    let mut inversions = 0;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            if p[i] > p[j] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 1
}

/// The largest absolute value of an element of a dense matrix.
fn largest<T: Num>(a: &Matrix<T>) -> f64 {
    a.as_slice()
        .iter()
        .fold(0.0, |most, &x| most.max(magnitude(x)))
}

/// The largest absolute value of an element of `L` below its diagonal.
fn largest_multiplier<T: Num>(lu: &SparseLu<T>) -> f64 {
    let l = lu.l();
    let mut most = 0.0f64;
    for i in 0..l.rows() {
        for (j, x) in l.row(i) {
            if j < i {
                most = most.max(magnitude(x));
            }
        }
    }
    most
}

/// A matrix that is safely nonsingular and still needs its rows exchanged: a random pattern of
/// about `per_row` elements a row, over a scrambled diagonal of larger ones, drawn again until
/// its condition number in the 1-norm is at most `limit`.
fn well_conditioned<T: Num>(r: &mut Rng, n: usize, per_row: usize, limit: f64) -> SparseMatrix<T> {
    for _ in 0..1000 {
        let scramble = shuffled(r, n);
        let mut entries = Vec::new();
        for _ in 0..n * per_row {
            entries.push((r.below(n), r.below(n), T::random(r)));
        }
        for (i, &j) in scramble.iter().enumerate() {
            let sign = if r.below(2) == 0 { 1.0 } else { -1.0 };
            entries.push((i, j, T::real(sign * (2.0 + r.unit().abs()))));
        }
        let a = SparseMatrix::from_coo(n, n, &entries);
        if a.to_matrix().cond(Norm::One).to_f64() <= limit {
            return a;
        }
    }
    panic!("no matrix of order {n} with a condition number below {limit} was found");
}

/// The five-point convection-diffusion operator on a `k × k` grid numbered row by row: 4 on the
/// diagonal, `−1 ∓ cx` to the west and the east and `−1 ∓ cy` to the south and the north. With
/// `cx = cy = 0` it is the Laplacian; past 1 the matrix is no longer diagonally dominant.
fn convection_diffusion<T: Num>(k: usize, cx: f64, cy: f64) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    for i in 0..k {
        for j in 0..k {
            let me = i * k + j;
            entries.push((me, me, T::real(4.0)));
            if i > 0 {
                entries.push((me, me - k, T::real(-1.0 - cy)));
            }
            if j > 0 {
                entries.push((me, me - 1, T::real(-1.0 - cx)));
            }
            if j + 1 < k {
                entries.push((me, me + 1, T::real(-1.0 + cx)));
            }
            if i + 1 < k {
                entries.push((me, me + k, T::real(-1.0 + cy)));
            }
        }
    }
    SparseMatrix::from_coo(k * k, k * k, &entries)
}

/// The nested dissection of a `k × k` grid numbered row by row, from its geometry: the longer
/// side is halved, the halves are ordered, and the line between them comes last.
fn dissection(k: usize) -> Vec<usize> {
    fn part(k: usize, rows: (usize, usize), cols: (usize, usize), out: &mut Vec<usize>) {
        let (height, width) = (rows.1 - rows.0, cols.1 - cols.0);
        if height == 0 || width == 0 {
            return;
        }
        if height * width <= 4 {
            for i in rows.0..rows.1 {
                out.extend((cols.0..cols.1).map(|j| i * k + j));
            }
        } else if height >= width {
            let middle = rows.0 + height / 2;
            part(k, (rows.0, middle), cols, out);
            part(k, (middle + 1, rows.1), cols, out);
            out.extend((cols.0..cols.1).map(|j| middle * k + j));
        } else {
            let middle = cols.0 + width / 2;
            part(k, rows, (cols.0, middle), out);
            part(k, rows, (middle + 1, cols.1), out);
            out.extend((rows.0..rows.1).map(|i| i * k + middle));
        }
    }
    let mut out = Vec::with_capacity(k * k);
    part(k, (0, k), (0, k), &mut out);
    out
}

/// Checks what holds of every decomposition — the permutations are permutations, the factors are
/// well-formed sparse matrices and triangular, `L` has ones on its diagonal, the permutation
/// matrices say what the permutations say — and returns the largest element of `P·A·Q − L·U`
/// in units of `n·ε·max|L|·max|U|`, which rounding keeps of the order of 1.
fn factor_error<T: Num>(lu: &SparseLu<T>, a: &SparseMatrix<T>) -> f64 {
    let n = a.rows();
    let (p, q) = (lu.permutation(), lu.column_permutation());
    assert!(is_permutation(p, n), "the rows aren't permuted: {p:?}");
    assert!(is_permutation(q, n), "the columns aren't permuted: {q:?}");
    let (l, u) = (lu.l(), lu.u());
    for factor in [&l, &u, &lu.p(), &lu.q()] {
        assert_csr(factor);
        assert_eq!((factor.rows(), factor.cols()), (n, n));
    }
    for i in 0..n {
        assert!(l.row(i).all(|(j, _)| j <= i), "L isn't lower triangular");
        assert!(u.row(i).all(|(j, _)| j >= i), "U isn't upper triangular");
        assert!(l[(i, i)] == T::one(), "the diagonal of L isn't 1");
    }
    let permuted = Matrix::from_fn(n, n, |i, j| a[(p[i], q[j])]);
    assert!(
        (&(&lu.p() * a) * &lu.q()).to_matrix() == permuted,
        "the permutation matrices disagree with the permutations"
    );
    let (l, u) = (l.to_matrix(), u.to_matrix());
    let scale = (largest(&l) * largest(&u)).max(f64::MIN_POSITIVE);
    largest(&(&permuted - &(&l * &u))) / (scale * n.max(1) as f64 * T::EPS)
}

/// The backward error of `x` as a solution of `m·x = b`: the largest element of the residual
/// over `‖m‖∞·max|x| + max|b|`, in units of `n·ε`.
fn backward_error<T: Num>(m: &Matrix<T>, x: &Matrix<T>, b: &Matrix<T>) -> f64 {
    let scale = m.norm(Norm::Infinity).to_f64() * largest(x) + largest(b);
    if scale == 0.0 {
        return 0.0;
    }
    largest(&(&(m * x) - b)) / (scale * m.rows().max(1) as f64 * T::EPS)
}

/// What a plain elimination on dense columns makes of a matrix when it takes the pivots the
/// sparse one took: the oracle for the pattern and the elements of the factors, and the judge
/// of each pivot.
struct Followed<T> {
    /// The unit lower triangular factor, its rows in the pivoted order.
    l: Matrix<T>,
    /// The upper triangular factor.
    u: Matrix<T>,
    /// The elements the elimination reaches: below the diagonal of `L`, above that of `U`, and
    /// the diagonal once.
    nnz: usize,
    /// Whether a column had no pivot.
    deficient: bool,
    /// How many pivots were not the ones this arithmetic would have chosen first.
    different: usize,
    /// By how much the worst of those breaks the rule beyond what rounding explains, relative
    /// to the largest element of its column: 0 when every pivot is one the rule allows.
    miss: f64,
    /// Whether a pivot was at the level of the rounding of its column, where two arithmetics
    /// may disagree about what is exactly zero.
    fragile: bool,
}

/// The left-looking elimination of `A·Q` on dense columns, with the pattern carried as flags and
/// no search at all, taking at step `j` the pivotal row `p[j]`. Each choice is measured against
/// the rule in this arithmetic: the largest size `|re| + |im|` among the rows that are not
/// pivotal, the lowest row of that size, unless the diagonal row reaches `threshold` times it.
fn follow<T: Num>(a: &SparseMatrix<T>, p: &[usize], q: &[usize], threshold: f64) -> Followed<T> {
    const NONE: usize = usize::MAX;
    let n = a.rows();
    let dense = a.to_matrix();
    let threshold = <T::Real as Float>::from_f64(threshold).to_f64();
    let size_of = |x: T| x.abs1().to_f64();
    let mut step_of_row = vec![NONE; n];
    let mut l_columns: Vec<Vec<(usize, T)>> = Vec::new();
    let mut u: Matrix<T> = Matrix::zeros(n, n);
    let mut out = Followed {
        l: Matrix::identity(n),
        u: Matrix::zeros(n, n),
        nnz: n,
        deficient: false,
        different: 0,
        miss: 0.0,
        fragile: false,
    };
    let mut fragility = 0.0f64;
    let mut free = 0;
    for j in 0..n {
        let col = q[j];
        let mut x: Vec<T> = (0..n).map(|i| dense[(i, col)]).collect();
        let mut pattern: Vec<bool> = (0..n).map(|i| a.find(i, col).is_some()).collect();
        // What each element was summed from, in size: the scale its rounding is on.
        let mut scale: Vec<f64> = x.iter().map(|&v| size_of(v)).collect();
        for k in 0..j {
            let pivotal = p[k];
            if !pattern[pivotal] {
                continue;
            }
            let multiplier = x[pivotal];
            for &(i, l) in &l_columns[k] {
                pattern[i] = true;
                scale[i] += size_of(l) * size_of(multiplier);
                x[i] = x[i].mul_sub(l, multiplier);
            }
        }
        let column_scale = scale.iter().cloned().fold(0.0, f64::max);
        // The rounding of this column, and that of the columns of `L` it was built with: a pivot
        // small beside its column leaves multipliers with as large a relative error as the
        // ratio says.
        let noise = 256.0 * n as f64 * T::EPS * column_scale * fragility.max(1.0);
        let candidate = |i: usize| pattern[i] && step_of_row[i] == NONE;
        let mut most = -1.0f64;
        let mut lowest = NONE;
        for i in (0..n).filter(|&i| candidate(i)) {
            if size_of(x[i]) > most {
                most = size_of(x[i]);
                lowest = i;
            }
        }
        let diagonal_free = step_of_row[col] == NONE;
        let on_diagonal = if pattern[col] { size_of(x[col]) } else { 0.0 };
        let own = if most > 0.0 {
            if diagonal_free && on_diagonal > 0.0 && on_diagonal >= threshold * most {
                col
            } else {
                lowest
            }
        } else if diagonal_free {
            col
        } else if lowest != NONE {
            lowest
        } else {
            while step_of_row[free] != NONE {
                free += 1;
            }
            free
        };
        let row = p[j];
        assert!(step_of_row[row] == NONE, "a row was pivotal twice");
        if row != own {
            out.different += 1;
            if most > 0.0 {
                let taken = if pattern[row] { size_of(x[row]) } else { 0.0 };
                let over = |violation: f64| (violation - noise).max(0.0) / most.max(noise);
                if row == col {
                    // Kept on the diagonal: it must reach the threshold.
                    out.miss = out.miss.max(over(threshold * most - taken));
                } else {
                    // Taken as the largest: it must be that, and the diagonal must fall short.
                    out.miss = out.miss.max(over(most - taken));
                    if diagonal_free && on_diagonal > 0.0 {
                        out.miss = out.miss.max(over(on_diagonal - threshold * most));
                    }
                }
            }
        }
        if most <= 0.0 {
            out.deficient = true;
        }
        let pivot = x[row];
        if size_of(pivot) <= noise {
            out.fragile = true;
        }
        fragility = fragility.max(if size_of(pivot) > 0.0 {
            column_scale / size_of(pivot)
        } else {
            f64::INFINITY
        });
        let reciprocal = T::one() / pivot;
        let mut l_column = Vec::new();
        for i in (0..n).filter(|&i| pattern[i]) {
            if step_of_row[i] != NONE {
                u[(step_of_row[i], j)] = x[i];
                out.nnz += 1;
            } else if i != row {
                // A pivot of zero is one the sparse code found among zeros alone: what this
                // arithmetic has left in the other rows is rounding, and goes the same way.
                let l = if x[i].is_zero() || pivot.is_zero() {
                    T::zero()
                } else if reciprocal.is_finite() {
                    x[i] * reciprocal
                } else {
                    x[i] / pivot
                };
                l_column.push((i, l));
                out.nnz += 1;
            }
        }
        u[(j, j)] = pivot;
        step_of_row[row] = j;
        l_columns.push(l_column);
    }
    for (k, l_column) in l_columns.iter().enumerate() {
        for &(i, l) in l_column {
            out.l[(step_of_row[i], k)] = l;
        }
    }
    out.u = u;
    out
}

/// The options with the columns left in place and the given pivot threshold.
fn natural(threshold: f64) -> SparseLuOptions {
    SparseLuOptions::new()
        .with_ordering(Ordering::Natural)
        .with_pivot_threshold(threshold)
}

/// Every kind of ordering, the given one a random permutation.
fn orderings(r: &mut Rng, n: usize) -> Vec<Ordering> {
    vec![
        Ordering::Natural,
        Ordering::ReverseCuthillMckee,
        Ordering::Amd,
        Ordering::Colamd,
        Ordering::NestedDissection,
        Ordering::Given(shuffled(r, n)),
    ]
}

fn factors_reproduce<T: Num>() {
    let mut r = Rng(0x5EED_0001);
    for _ in 0..size(60, 3) {
        let n = 1 + r.below(size(48, 7));
        // Any matrix at all, singular ones among them: the factors are those of the matrix
        // whether or not a pivot was found in every column.
        let per_row = 1 + r.below(5);
        let a: SparseMatrix<T> = random_sparse(&mut r, n, n, n * per_row);
        for lu in [
            SparseLu::new(&a),
            SparseLu::with_ordering(&a, &Ordering::Natural),
        ] {
            let error = factor_error(&lu, &a);
            assert!(error < 4.0, "P·A·Q differs from L·U by {error} units");
        }
    }
}

#[test]
fn the_factors_reproduce_the_permuted_matrix() {
    each_type!(factors_reproduce);
}

fn every_ordering_factors<T: Num>() {
    let mut r = Rng(0x5EED_0002);
    for _ in 0..size(8, 1) {
        let n = 2 + r.below(size(36, 6));
        let a: SparseMatrix<T> = well_conditioned(&mut r, n, 2, 200.0);
        let dense = a.to_matrix();
        let b = Matrix::from_fn(n, 2, |_, _| T::random(&mut r));
        for ordering in orderings(&mut r, n) {
            let lu = SparseLu::with_ordering(&a, &ordering);
            if let Ordering::Given(q) = &ordering {
                assert_eq!(lu.column_permutation(), q);
            }
            if ordering == Ordering::Natural {
                assert!(lu.column_permutation().iter().copied().eq(0..n));
            }
            let error = factor_error(&lu, &a);
            assert!(
                error < 4.0,
                "{ordering:?}: P·A·Q differs from L·U by {error}"
            );
            assert!(!lu.is_singular(), "{ordering:?} found the matrix singular");
            let x = lu.solve(&b).unwrap();
            let error = backward_error(&dense, &x, &b);
            assert!(error < 8.0, "{ordering:?}: a backward error of {error}");
            // The determinant is that of the matrix whatever the order of the elimination.
            let (det, expected) = (lu.det(), dense.det());
            assert!(
                magnitude(det - expected) <= 1e3 * n as f64 * T::EPS * magnitude(expected),
                "{ordering:?}: the determinant {det:?} isn't {expected:?}"
            );
        }
    }
}

#[test]
fn every_ordering_gives_a_decomposition_of_the_same_matrix() {
    each_type!(every_ordering_factors);
}

fn agrees_with_dense_elimination<T: Num>() {
    let mut r = Rng(0x5EED_0003);
    for trial in 0..size(40, 2) {
        let n = 1 + r.below(size(40, 7));
        // Well-conditioned matrices, whose pivots are far from ties, and any matrix at all.
        let a: SparseMatrix<T> = if trial % 2 == 0 {
            let per_row = 1 + r.below(3);
            well_conditioned(&mut r, n, per_row, 100.0)
        } else {
            let per_row = 1 + r.below(4);
            random_sparse(&mut r, n, n, n * per_row)
        };
        let ordering = if r.below(2) == 0 {
            Ordering::Natural
        } else {
            Ordering::Given(shuffled(&mut r, n))
        };
        let threshold = [1.0, 1.0, 0.5, 0.1, 0.01][r.below(5)];
        let options = SparseLuOptions::new()
            .with_ordering(ordering)
            .with_pivot_threshold(threshold);
        let lu = SparseLu::with_options(&a, &options).unwrap();
        let oracle = follow(&a, lu.permutation(), lu.column_permutation(), threshold);
        // The search reaches the rows the flags reach, pruned or not.
        assert_eq!(lu.nnz(), oracle.nnz, "the factors hold other elements");
        // Each pivot is one the rule allows.
        assert!(
            oracle.miss == 0.0,
            "a pivot misses the rule by {} of the largest element",
            oracle.miss
        );
        if !oracle.fragile {
            assert_eq!(lu.is_singular(), oracle.deficient);
        }
        // And where no pivot is in doubt, the factors are the same but for the order of the
        // sums. How far that moves them depends on the conditioning of the leading blocks, so
        // only the well-conditioned matrices are held to it, and in double precision.
        if trial % 2 == 0 && oracle.different == 0 && T::EPS < 1e-10 {
            let (l, u) = (lu.l().to_matrix(), lu.u().to_matrix());
            let difference = largest(&(&l - &oracle.l)).max(largest(&(&u - &oracle.u)));
            assert!(
                difference <= 1e-9 * largest(&oracle.u).max(1.0),
                "the factors differ from those of the dense elimination by {difference}"
            );
        }
    }
}

#[test]
fn the_factors_are_those_of_a_plain_elimination_with_the_same_pivots() {
    each_type!(agrees_with_dense_elimination);
}

#[test]
fn the_pivot_is_the_largest_element_and_ties_go_to_the_diagonal_then_the_lowest_row() {
    // The largest element of the column, wherever it is.
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, 1.0),
            (1, 0, -3.0),
            (2, 0, 2.0),
            (0, 1, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.permutation()[0], 1);
    // A tie with the diagonal goes to the diagonal, though a lower row has the same size.
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 1, 2.0),
            (1, 1, -2.0),
            (2, 1, 2.0),
            (0, 0, 1.0),
            (1, 2, 1.0),
            (2, 2, 3.0),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Given(vec![1, 0, 2]));
    assert_eq!(lu.permutation()[0], 1);
    // A tie the diagonal is not part of goes to the lowest row.
    let a = SparseMatrix::from_coo(
        4,
        4,
        &[
            (0, 0, 1.0),
            (1, 0, 0.0),
            (2, 0, -5.0),
            (3, 0, 5.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
            (3, 3, 1.0),
            (0, 3, 1.0),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.permutation()[0], 2);
    // The diagonal is the row with the number of the column, wherever the ordering puts the
    // column: here column 2 goes first, and its three equal elements leave the choice to row 2.
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 2, 4.0),
            (1, 2, -4.0),
            (2, 2, 4.0),
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 1, 1.0),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Given(vec![2, 0, 1]));
    assert_eq!(lu.permutation()[0], 2);
    // A diagonal row that an earlier column has taken is no longer a choice.
    let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (1, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.permutation(), [1, 0]);
    assert!(factor_error(&lu, &a) < 4.0);
}

#[test]
fn the_size_of_a_complex_element_is_the_sum_of_its_parts() {
    // 1 + i has the size 2 and the absolute value 1.41: it is preferred to 1.5, as in LAPACK and
    // in the dense decomposition, and the multiplier it leaves is larger than 1.
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, Complex::new(1.5, 0.0)),
            (1, 0, Complex::new(1.0, 1.0)),
            (0, 1, Complex::new(1.0, 0.0)),
            (1, 1, Complex::new(0.0, 2.0)),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.permutation(), [1, 0]);
    assert_eq!(a.to_matrix().lu().permutation(), [1, 0]);
    let multiplier: f64 = largest_multiplier(&lu);
    assert!((multiplier - 1.5 / 2f64.sqrt()).abs() < 1e-15);
    assert!(factor_error(&lu, &a) < 4.0);
}

fn multipliers_are_bounded<T: Num>() {
    let mut r = Rng(0x5EED_0004);
    // Sizes are `|re| + |im|`, within a factor √2 of the absolute value of a complex element.
    let slack = if T::IS_COMPLEX { 2f64.sqrt() } else { 1.0 };
    for _ in 0..size(30, 2) {
        let n = 2 + r.below(size(40, 6));
        let per_row = 2 + r.below(4);
        let a: SparseMatrix<T> = random_sparse(&mut r, n, n, n * per_row);
        for threshold in [1.0, 0.5, 0.125, 0.01] {
            let lu = SparseLu::with_options(&a, &natural(threshold)).unwrap();
            let multiplier = largest_multiplier(&lu);
            assert!(
                multiplier * threshold <= slack * (1.0 + 8.0 * T::EPS),
                "a multiplier of {multiplier} under the threshold {threshold}"
            );
        }
    }
}

#[test]
fn no_multiplier_exceeds_the_reciprocal_of_the_pivot_threshold() {
    each_type!(multipliers_are_bounded);
}

fn threshold_keeps_the_diagonal<T: Num>() {
    // Blocks with 1 on the diagonal and 2 off it: partial pivoting exchanges the rows of each,
    // and a threshold the diagonal reaches leaves them where they are.
    let blocks = size(8, 2);
    let n = 2 * blocks;
    let mut entries = Vec::new();
    for b in 0..blocks {
        let (i, j) = (2 * b, 2 * b + 1);
        entries.extend([
            (i, i, T::real(1.0)),
            (i, j, T::real(2.0)),
            (j, i, T::real(2.0)),
            (j, j, T::real(1.0)),
        ]);
    }
    let a = SparseMatrix::from_coo(n, n, &entries);
    let identity: Vec<usize> = (0..n).collect();
    let swapped: Vec<usize> = (0..n).map(|i| i ^ 1).collect();
    for (threshold, expected) in [
        (1.0, &swapped),
        (0.6, &swapped),
        (0.5, &identity),
        (0.4, &identity),
    ] {
        let lu = SparseLu::with_options(&a, &natural(threshold)).unwrap();
        assert_eq!(lu.permutation(), expected.as_slice(), "at {threshold}");
        assert!(factor_error(&lu, &a) < 4.0);
        assert!(
            magnitude(lu.det() - T::real((-3.0f64).powi(blocks as i32)))
                <= 1e-4 * 3f64.powi(blocks as i32)
        );
    }
    // Whatever symmetric ordering the columns are given, the rows follow it when the diagonal
    // is kept: the pattern eliminated is the one the ordering was made for.
    let mut r = Rng(0x5EED_0005);
    let q = shuffled(&mut r, n);
    let options = SparseLuOptions::new()
        .with_ordering(Ordering::Given(q.clone()))
        .with_pivot_threshold(0.5);
    let lu = SparseLu::with_options(&a, &options).unwrap();
    assert_eq!(lu.permutation(), q);
    assert_eq!(lu.column_permutation(), q);
}

#[test]
fn a_pivot_threshold_keeps_the_pivots_on_the_diagonal() {
    each_type!(threshold_keeps_the_diagonal);
}

fn threshold_saves_fill<T: Num>() {
    // Convection so strong that partial pivoting leaves the diagonal, in the nested dissection
    // order of the grid: taking the rows from elsewhere fills what the ordering kept empty.
    let k = size(12, 5);
    let a: SparseMatrix<T> = convection_diffusion(k, 1.5, 3.5);
    let dense = a.to_matrix();
    let ordering = Ordering::Given(dissection(k));
    let partial = SparseLu::with_ordering(&a, &ordering);
    let options = SparseLuOptions::new()
        .with_ordering(ordering)
        .with_pivot_threshold(0.25);
    let kept = SparseLu::with_options(&a, &options).unwrap();
    assert_ne!(partial.permutation(), partial.column_permutation());
    assert_eq!(kept.permutation(), kept.column_permutation());
    assert!(
        kept.nnz() < partial.nnz(),
        "{} elements with the threshold, {} without",
        kept.nnz(),
        partial.nnz()
    );
    let b = Matrix::from_fn(k * k, 1, |i, _| T::real(1.0 + (i % 7) as f64));
    for lu in [&partial, &kept] {
        assert!(factor_error(lu, &a) < 4.0);
        let x = lu.solve(&b).unwrap();
        let growth = lu.growth().to_f64().max(1.0) * largest_multiplier(lu).max(1.0);
        assert!(backward_error(&dense, &x, &b) < 8.0 * growth);
    }
}

#[test]
fn a_pivot_threshold_keeps_the_fill_of_a_symmetric_ordering() {
    each_type!(threshold_saves_fill);
}

#[test]
fn a_pivot_threshold_outside_its_range_gives_none() {
    let a = SparseMatrix::<f64>::identity(3);
    for bad in [
        0.0,
        -0.5,
        1.000001,
        2.0,
        f64::NAN,
        f64::INFINITY,
        -f64::INFINITY,
    ] {
        let options = SparseLuOptions::new().with_pivot_threshold(bad);
        assert!(
            SparseLu::with_options(&a, &options).is_none(),
            "a threshold of {bad} was accepted"
        );
    }
    for good in [1.0, 0.5, 1e-3, f64::MIN_POSITIVE] {
        let options = SparseLuOptions::new().with_pivot_threshold(good);
        let lu = SparseLu::with_options(&a, &options).expect("a threshold in range");
        assert_eq!(lu.det(), 1.0);
    }
    // A threshold too small for the type of the elements keeps any diagonal that is not zero,
    // and still never a zero one.
    let b = SparseMatrix::<f32>::from_coo(2, 2, &[(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1e-30)]);
    let lu = SparseLu::with_options(&b, &natural(1e-300)).unwrap();
    assert!(!lu.is_singular());
    assert_eq!(lu.permutation(), [1, 0]);
}

#[test]
fn the_default_options_are_colamd_and_partial_pivoting() {
    let options = SparseLuOptions::new();
    assert_eq!(options, SparseLuOptions::default());
    assert_eq!(options.ordering, Ordering::Colamd);
    assert_eq!(options.pivot_threshold, 1.0);
    let mut r = Rng(0x5EED_0006);
    let a: SparseMatrix<f64> = well_conditioned(&mut r, size(20, 5), 2, 100.0);
    // The three ways to ask for the defaults give one decomposition.
    let lu = SparseLu::new(&a);
    assert_eq!(lu, SparseLu::with_ordering(&a, &Ordering::Colamd));
    assert_eq!(lu, SparseLu::with_options(&a, &options).unwrap());
    assert_eq!(lu.clone(), lu);
}

fn agrees_with_the_dense_decomposition<T: Num>() {
    let mut r = Rng(0x5EED_0007);
    for _ in 0..size(20, 2) {
        let n = 1 + r.below(size(40, 6));
        let limit = 100.0;
        let per_row = 1 + r.below(3);
        let a: SparseMatrix<T> = well_conditioned(&mut r, n, per_row, limit);
        let dense = a.to_matrix();
        let b = Matrix::from_fn(n, 3, |_, _| T::random(&mut r));
        let expected = dense.lu().solve(&b).unwrap();
        let x = SparseLu::new(&a).solve(&b).unwrap();
        let error = largest(&(&x - &expected)) / largest(&expected);
        assert!(
            error <= 4.0 * limit * n as f64 * T::EPS,
            "the solutions differ by {error}"
        );
        assert!(backward_error(&dense, &x, &b) < 8.0);
    }
}

#[test]
fn solutions_agree_with_the_dense_decomposition() {
    each_type!(agrees_with_the_dense_decomposition);
}

fn right_hand_sides<T: Num>() {
    let mut r = Rng(0x5EED_0008);
    let n = size(25, 5);
    let a: SparseMatrix<T> = well_conditioned(&mut r, n, 2, 100.0);
    let lu = SparseLu::new(&a);
    for columns in [0, 1, 4] {
        let b = Matrix::from_fn(n, columns, |_, _| T::random(&mut r));
        let x = lu.solve(&b).unwrap();
        let xt = lu.solve_transpose(&b).unwrap();
        let xh = lu.solve_conjugate_transpose(&b).unwrap();
        assert_eq!((x.rows(), x.cols()), (n, columns));
        assert_eq!((xt.rows(), xt.cols()), (n, columns));
        assert_eq!((xh.rows(), xh.cols()), (n, columns));
        // A column of a matrix of right-hand sides is solved as it is alone.
        for c in 0..columns {
            assert!(lu.solve_vec(b.column(c)).unwrap() == x.column(c));
            assert!(lu.solve_transpose_vec(b.column(c)).unwrap() == xt.column(c));
            assert!(lu.solve_conjugate_transpose_vec(b.column(c)).unwrap() == xh.column(c));
        }
    }
    // A right-hand side of zeros has the solution of zeros.
    let zero = lu.solve_vec(&vec![T::zero(); n]).unwrap();
    assert!(zero.iter().all(|x| x.is_zero()));
}

#[test]
fn several_right_hand_sides_are_solved_column_by_column() {
    each_type!(right_hand_sides);
}

fn transposed_solves<T: Num>() {
    let mut r = Rng(0x5EED_0009);
    for _ in 0..size(12, 2) {
        let n = 2 + r.below(size(36, 5));
        let a: SparseMatrix<T> = well_conditioned(&mut r, n, 2, 100.0);
        let dense = a.to_matrix();
        let (transpose, adjoint) = (dense.transpose(), dense.conjugate_transpose());
        let ordering = Ordering::Given(shuffled(&mut r, n));
        let lu = SparseLu::with_ordering(&a, &ordering);
        let b = Matrix::from_fn(n, 2, |_, _| T::random(&mut r));
        let xt = lu.solve_transpose(&b).unwrap();
        let xh = lu.solve_conjugate_transpose(&b).unwrap();
        assert!(backward_error(&transpose, &xt, &b) < 8.0);
        assert!(backward_error(&adjoint, &xh, &b) < 8.0);
        // The same against the factors of the transposed matrices themselves.
        let direct = SparseLu::new(&a.transpose()).solve(&b).unwrap();
        assert!(largest(&(&xt - &direct)) <= 1e3 * n as f64 * T::EPS * largest(&direct));
        let direct = SparseLu::new(&a.conjugate_transpose()).solve(&b).unwrap();
        assert!(largest(&(&xh - &direct)) <= 1e3 * n as f64 * T::EPS * largest(&direct));
        if T::IS_COMPLEX {
            // The two are different systems for a matrix that is not real: the solution of one
            // doesn't solve the other.
            assert!(backward_error(&transpose, &xh, &b) > 1e3);
            assert!(backward_error(&adjoint, &xt, &b) > 1e3);
        } else {
            assert!(xt == xh);
        }
    }
}

#[test]
fn the_transposed_systems_are_solved_from_the_same_factors() {
    each_type!(transposed_solves);
}

fn needs_pivoting<T: Num>() {
    // The neighbours of a path of even length: nothing on the diagonal at all, so every pivot
    // is found off it.
    let n = size(40, 6);
    let x_true = Matrix::from_fn(n, 1, |i, _| T::real(1.0 + (i % 5) as f64));
    for diagonal in [0.0, 1e-20] {
        let a = SparseMatrix::tridiagonal(
            &vec![T::real(1.0); n - 1],
            &vec![T::real(diagonal); n],
            &vec![T::real(1.0); n - 1],
        );
        let dense = a.to_matrix();
        let b = &dense * &x_true;
        for threshold in [1.0, 0.1] {
            let lu = SparseLu::with_options(&a, &natural(threshold)).unwrap();
            assert!(!lu.is_singular());
            assert!(lu.permutation().iter().enumerate().any(|(i, &p)| i != p));
            assert!(factor_error(&lu, &a) < 4.0);
            // No pivot was taken from the tiny diagonal: nothing grew.
            assert!(
                lu.growth().to_f64() <= 2.0,
                "the growth is {:?}",
                lu.growth()
            );
            let x = lu.solve(&b).unwrap();
            let error = largest(&(&x - &x_true));
            assert!(
                error <= 64.0 * n as f64 * T::EPS * 5.0,
                "an error of {error}"
            );
            let xt = lu.solve_transpose(&b).unwrap();
            assert!(largest(&(&xt - &x_true)) <= 64.0 * n as f64 * T::EPS * 5.0);
        }
    }
    // A cyclic shift: a permutation matrix, each pivot the only element of its column.
    let shift: Vec<(usize, usize, T)> = (0..n).map(|i| (i, (i + 1) % n, T::real(2.0))).collect();
    let a = SparseMatrix::from_coo(n, n, &shift);
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.nnz(), n);
    let b: Vec<T> = (0..n).map(|i| T::real(2.0 * i as f64)).collect();
    let x = lu.solve_vec(&b).unwrap();
    // Row i says 2·x[i + 1] = b[i].
    for i in 0..n {
        assert!(x[(i + 1) % n] == T::real(i as f64));
    }
}

#[test]
fn matrices_with_a_zero_or_a_tiny_diagonal_are_solved_by_pivoting() {
    each_type!(needs_pivoting);
}

fn block_triangular<T: Num>() {
    let mut r = Rng(0x5EED_000A);
    let (blocks, order) = (size(5, 2), size(6, 3));
    let n = blocks * order;
    // Diagonal blocks that are safely nonsingular, and elements above them here and there.
    let mut dense: Matrix<T> = Matrix::zeros(n, n);
    let mut determinant = T::one();
    for b in 0..blocks {
        let block: SparseMatrix<T> = well_conditioned(&mut r, order, 2, 50.0);
        let block = block.to_matrix();
        determinant *= block.det();
        for i in 0..order {
            for j in 0..order {
                dense[(b * order + i, b * order + j)] = block[(i, j)];
            }
        }
        for c in b + 1..blocks {
            for _ in 0..order {
                let (i, j) = (b * order + r.below(order), c * order + r.below(order));
                dense[(i, j)] = T::random(&mut r);
            }
        }
    }
    let rhs = Matrix::from_fn(n, 2, |_, _| T::random(&mut r));
    let check = |m: &Matrix<T>, expected: T, r: &mut Rng| {
        let a = SparseMatrix::from_matrix(m);
        for ordering in orderings(r, n) {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert!(!lu.is_singular());
            assert!(factor_error(&lu, &a) < 4.0);
            let x = lu.solve(&rhs).unwrap();
            assert!(backward_error(m, &x, &rhs) < 8.0);
            let det = lu.det();
            assert!(
                magnitude(det - expected) <= 1e4 * n as f64 * T::EPS * magnitude(expected),
                "{ordering:?}: the determinant {det:?} isn't the product of the blocks', {expected:?}"
            );
        }
    };
    // Block triangular as it is.
    check(&dense, determinant, &mut r);
    // Reducible: the blocks with nothing between them.
    let diagonal = Matrix::from_fn(n, n, |i, j| {
        if i / order == j / order {
            dense[(i, j)]
        } else {
            T::zero()
        }
    });
    check(&diagonal, determinant, &mut r);
    // And with its rows and columns scrambled, which hides the blocks and can only change the
    // sign of the determinant.
    let (rows, cols) = (shuffled(&mut r, n), shuffled(&mut r, n));
    let scrambled = Matrix::from_fn(n, n, |i, j| dense[(rows[i], cols[j])]);
    let sign = if is_odd(&rows) != is_odd(&cols) {
        -determinant
    } else {
        determinant
    };
    check(&scrambled, sign, &mut r);
}

#[test]
fn block_triangular_and_reducible_matrices_are_factored() {
    each_type!(block_triangular);
}

fn matches_scipy<T: Num>() {
    for case in REFERENCES.iter().take(size(REFERENCES.len(), 2)) {
        if case.complex && !T::IS_COMPLEX {
            continue;
        }
        let n = case.n;
        let a: SparseMatrix<T> = reference_matrix(case);
        assert_eq!(a.nnz(), case.entries.len());
        let cond = 1.0 / case.rcond;
        let b: Matrix<T> = column(case.rhs);
        for ordering in [
            Ordering::Natural,
            Ordering::Colamd,
            Ordering::ReverseCuthillMckee,
        ] {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert!(!lu.is_singular(), "{}", case.name);
            let solutions = [
                (lu.solve(&b), case.x, "A"),
                (lu.solve_transpose(&b), case.xt, "the transpose of A"),
                (
                    lu.solve_conjugate_transpose(&b),
                    case.xh,
                    "the conjugate transpose of A",
                ),
            ];
            for (x, expected, system) in solutions {
                let (x, expected): (Matrix<T>, Matrix<T>) = (x.unwrap(), column(expected));
                let error = largest(&(&x - &expected)) / largest(&expected);
                assert!(
                    error <= 4.0 * cond * T::EPS,
                    "{}: the solution with {system} is {error} from SciPy's",
                    case.name
                );
            }
            let det = lu.det();
            let expected: T = element(case.det.0, case.det.1);
            assert!(
                magnitude(det - expected) <= 4.0 * n as f64 * cond * T::EPS * magnitude(expected),
                "{}: the determinant {det:?} isn't NumPy's {expected:?}",
                case.name
            );
            // The estimate of the condition is a lower bound, and a close one.
            let ratio = lu.rcond().to_f64() / case.rcond;
            assert!(
                ratio > 1.0 - 64.0 * n as f64 * cond * T::EPS && ratio < 3.0,
                "{}: the estimate of the reciprocal condition is {ratio} of the true one",
                case.name
            );
        }
    }
}

#[test]
fn solutions_determinants_and_conditions_match_scipy_and_numpy() {
    each_type!(matches_scipy);
}

fn matches_superlu<T: Num>() {
    for case in REFERENCES.iter().take(size(REFERENCES.len(), 2)) {
        if case.complex && !T::IS_COMPLEX {
            continue;
        }
        let a: SparseMatrix<T> = reference_matrix(case);
        let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
        // The same columns and the same rule: the same rows, and the same reach. Which of the
        // elements reached come out exactly zero depends on the rounding, in which SuperLU and
        // this elimination differ; the elements themselves do not.
        assert_eq!(lu.permutation(), case.perm, "{}", case.name);
        assert_eq!(lu.nnz(), case.fill, "{}", case.name);
        assert!(lu.l().nnz() - case.n + lu.u().nnz() <= case.fill);
    }
}

#[test]
fn the_row_order_and_the_fill_are_those_of_superlu() {
    each_type!(matches_superlu);
}

fn determinant_signs<T: Num>() {
    let mut r = Rng(0x5EED_000B);
    for _ in 0..size(30, 3) {
        let n = 1 + r.below(size(30, 6));
        // A permutation matrix, whose determinant is the sign of the permutation, exactly.
        let sigma = shuffled(&mut r, n);
        let entries: Vec<(usize, usize, T)> = sigma
            .iter()
            .enumerate()
            .map(|(i, &j)| (i, j, T::one()))
            .collect();
        let a = SparseMatrix::from_coo(n, n, &entries);
        let expected = if is_odd(&sigma) { -T::one() } else { T::one() };
        for ordering in orderings(&mut r, n) {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert!(lu.det() == expected, "{ordering:?} of {sigma:?}");
        }
        // The same with elements whose product is known: a scaled permutation.
        let scaled = a.map(|i, _, _| T::real(if i % 2 == 0 { 2.0 } else { -0.5 }));
        let product = T::real((-1.0f64).powi((n / 2) as i32) * if n % 2 == 1 { 2.0 } else { 1.0 });
        let expected = if is_odd(&sigma) { -product } else { product };
        assert!(SparseLu::new(&scaled).det() == expected);
        assert!(
            SparseLu::with_ordering(&scaled, &Ordering::Given(shuffled(&mut r, n))).det()
                == expected
        );
    }
    // And against the dense determinant of matrices with fill.
    for _ in 0..size(12, 2) {
        let n = 2 + r.below(size(30, 5));
        let a: SparseMatrix<T> = well_conditioned(&mut r, n, 2, 100.0);
        let expected = a.to_matrix().det();
        for ordering in [Ordering::Natural, Ordering::Given(shuffled(&mut r, n))] {
            for threshold in [1.0, 0.3] {
                let options = SparseLuOptions::new()
                    .with_ordering(ordering.clone())
                    .with_pivot_threshold(threshold);
                let lu = SparseLu::with_options(&a, &options).unwrap();
                let det = lu.det();
                let growth = lu.growth().to_f64().max(1.0) * largest_multiplier(&lu).max(1.0);
                assert!(
                    magnitude(det - expected)
                        <= 1e3 * growth * n as f64 * T::EPS * magnitude(expected),
                    "the determinant {det:?} isn't the dense one, {expected:?}"
                );
            }
        }
    }
}

#[test]
fn the_determinant_carries_the_sign_of_both_permutations() {
    each_type!(determinant_signs);
}

/// Checks everything a singular matrix must answer.
fn assert_singular<T: Num>(a: &SparseMatrix<T>, what: &str) {
    let n = a.rows();
    let mut r = Rng(0x5EED_000C);
    for ordering in orderings(&mut r, n) {
        for threshold in [1.0, 0.1] {
            let options = SparseLuOptions::new()
                .with_ordering(ordering.clone())
                .with_pivot_threshold(threshold);
            let lu = SparseLu::with_options(a, &options).unwrap();
            assert!(
                lu.is_singular(),
                "{what} wasn't found singular ({ordering:?})"
            );
            assert!(
                lu.det() == T::zero(),
                "{what}: a determinant of {:?}",
                lu.det()
            );
            assert!(lu.det().re().to_f64().is_sign_positive());
            assert_eq!(lu.rcond().to_f64(), 0.0, "{what}");
            let b = Matrix::from_fn(n, 2, |i, _| T::real(i as f64));
            assert!(lu.solve(&b).is_none(), "{what} was solved");
            assert!(lu.solve_transpose(&b).is_none());
            assert!(lu.solve_conjugate_transpose(&b).is_none());
            assert!(lu.solve_vec(b.column(0)).is_none());
            assert!(lu.solve_transpose_vec(b.column(0)).is_none());
            assert!(lu.solve_conjugate_transpose_vec(b.column(0)).is_none());
            // The factors are still those of the matrix. One singular by its elements has a
            // zero on the diagonal of U; one singular by its pattern may come out of the
            // rounding with a tiny pivot instead, and is known by its pattern.
            let error = factor_error(&lu, a);
            assert!(error < 4.0, "{what}: P·A·Q differs from L·U by {error}");
            if structural_rank(a) == n {
                let u = lu.u();
                assert!(
                    (0..n).any(|i| u[(i, i)].is_zero()),
                    "{what}: no pivot of U is zero"
                );
            }
        }
    }
}

fn singular_matrices<T: Num>() {
    let n = size(7, 4);
    let mut r = Rng(0x5EED_000D);
    let full: SparseMatrix<T> = well_conditioned(&mut r, n, 3, 100.0);
    assert_singular(&SparseMatrix::<T>::zeros(n, n), "the matrix of zeros");
    assert_singular(&SparseMatrix::<T>::zeros(1, 1), "the 1 × 1 zero");
    // A column of zeros, then a row of zeros.
    let hole = 2;
    assert_singular(
        &full.map(|_, j, x| if j == hole { T::zero() } else { x }),
        "a zero column",
    );
    assert_singular(
        &full.map(|i, _, x| if i == hole { T::zero() } else { x }),
        "a zero row",
    );
    // Two columns alike, by elements that are exact in every type, so that the elimination
    // leaves exact zeros: small integers, and multipliers that are halves.
    let twice = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(2.0)),
            (0, 1, T::real(4.0)),
            (1, 0, T::real(1.0)),
            (1, 1, T::real(2.0)),
            (1, 2, T::real(1.0)),
            (2, 2, T::real(3.0)),
        ],
    );
    assert_singular(&twice, "a column twice another");
    assert_singular(&twice.transpose(), "a row twice another");
    // Singular in the last step only: the leading blocks are all nonsingular.
    let last = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(1.0)),
            (0, 2, T::real(1.0)),
            (1, 1, T::real(1.0)),
            (1, 2, T::real(1.0)),
            (2, 0, T::real(1.0)),
            (2, 1, T::real(1.0)),
            (2, 2, T::real(2.0)),
        ],
    );
    assert_singular(&last, "a matrix singular in its last pivot");
}

#[test]
fn a_singular_matrix_is_reported_and_never_solved() {
    each_type!(singular_matrices);
}

fn structurally_singular<T: Num>() {
    let mut r = Rng(0x5EED_000E);
    let n = size(12, 5);
    // Three columns that have elements in the same two rows only: no elements whatever can make
    // them independent. The other columns are full.
    let mut entries = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if j < 3 && i >= 2 {
                continue;
            }
            entries.push((i, j, T::random(&mut r)));
        }
    }
    let a = SparseMatrix::from_coo(n, n, &entries);
    assert_singular(&a, "three columns in two rows");
    assert_singular(&a.transpose(), "three rows in two columns");
    // A column with no element at all in a matrix that is otherwise a permutation: rows 1 to
    // n − 1 take the columns 2, …, n − 1, 0, and row 0 shares column 2 with row 1, so that
    // column 1 is empty.
    let mut shift: Vec<(usize, usize, T)> = (1..n).map(|i| (i, (i + 1) % n, T::one())).collect();
    shift.push((0, 2, T::one()));
    assert_singular(&SparseMatrix::from_coo(n, n, &shift), "an empty column");
}

#[test]
fn a_structurally_singular_matrix_is_reported() {
    each_type!(structurally_singular);
}

/// Checks that a matrix singular to working precision is reported and refused by every solve,
/// with a determinant and a condition estimate of zero.
fn assert_refused<T: Num>(lu: &SparseLu<T>, what: &str) {
    let n = lu.permutation().len();
    assert!(
        lu.is_singular(),
        "{what} wasn't found singular: {} rcond {:e} growth {:e} pivots {:?}",
        std::any::type_name::<T>(),
        lu.rcond().to_f64(),
        lu.growth().to_f64(),
        lu.u().diagonal()
    );
    assert!(
        lu.det() == T::zero(),
        "{what}: a determinant of {:?}",
        lu.det()
    );
    assert_eq!(lu.rcond().to_f64(), 0.0, "{what}");
    let b = Matrix::from_fn(n, 2, |i, c| T::real(if c == 0 { 1.0 } else { i as f64 }));
    assert!(lu.solve(&b).is_none(), "{what} was solved");
    assert!(lu.solve_transpose(&b).is_none(), "{what} was solved");
    assert!(
        lu.solve_conjugate_transpose(&b).is_none(),
        "{what} was solved"
    );
    assert!(lu.solve_vec(b.column(0)).is_none(), "{what} was solved");
    assert!(lu.solve_transpose_vec(b.column(1)).is_none());
    assert!(lu.solve_conjugate_transpose_vec(b.column(1)).is_none());
}

fn one_to_nine<T: Num>() {
    // The integers 1 to 9 by rows: singular, but the thirds of its elimination are not exact, so
    // in most orders of the columns the last pivot comes out of the size of the rounding error
    // rather than as zero. The draft of review round 1 called it regular in four of the six
    // orders and solved it, returning numbers of the order of 10¹⁵; the dense decomposition
    // does the same.
    let a = SparseMatrix::from_fn(3, 3, |i, j| T::real((3 * i + j + 1) as f64));
    let mut rounded = 0;
    for q in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for threshold in [1.0, 0.5] {
            let options = SparseLuOptions::new()
                .with_ordering(Ordering::Given(q.to_vec()))
                .with_pivot_threshold(threshold);
            let lu = SparseLu::with_options(&a, &options).unwrap();
            let u = lu.u();
            if (0..3).all(|i| !u[(i, i)].is_zero()) {
                rounded += 1;
            }
            assert_refused(&lu, &format!("1 to 9 in the column order {q:?}"));
            let error = factor_error(&lu, &a);
            assert!(error < 4.0, "P·A·Q differs from L·U by {error}");
        }
    }
    // The orders in which no pivot comes out as zero are the point of the test.
    assert!(rounded > 0, "every order met a pivot of zero");
    // A well-conditioned neighbour for contrast.
    let b = &a + &SparseMatrix::identity(3);
    let lu = SparseLu::new(&b);
    assert!(!lu.is_singular() && lu.rcond().to_f64() > 1e-3);
}

#[test]
fn the_integers_one_to_nine_are_singular_in_every_order_of_their_columns() {
    each_type!(one_to_nine);
}

fn dependent_rows<T: Num>() {
    // A row, or a column, that is another times 1, −1, 3, 7, 5, −3 or ½, over elements k/64
    // with |k| ≤ 1024: every product is exact in every type, so the matrix as stored is
    // singular, and in most orders its elimination ends with a pivot of the size of the
    // rounding error. The draft of review round 1 called 11 % of the real and 47 % of the
    // complex ones with two equal rows regular.
    let mut r = Rng(0x5EED_0030);
    let factors = [1.0, -1.0, 3.0, 7.0, 5.0, -3.0, 0.5];
    for trial in 0..size(140, 7) {
        let n = 3 + r.below(size(18, 3));
        let mut m = vec![vec![(0.0, 0.0); n]; n];
        let mut q = || (r.unit() * 1024.0).round() / 64.0;
        let mut values = Vec::new();
        for _ in 0..3 * n {
            values.push((q(), q()));
        }
        for &(re, im) in &values {
            let (i, j) = (r.below(n), r.below(n));
            m[i][j] = (re, im);
        }
        for (i, &j) in shuffled(&mut r, n).iter().enumerate() {
            m[i][j] = (3.0 + (r.unit() * 1024.0).round().abs() / 1024.0, 0.0);
        }
        let i = r.below(n);
        let k = (i + 1 + r.below(n - 1)) % n;
        let f = factors[trial % factors.len()];
        m[k] = m[i].iter().map(|&(re, im)| (re * f, im * f)).collect();
        let mut entries = Vec::new();
        for (row, line) in m.iter().enumerate() {
            for (col, &(re, im)) in line.iter().enumerate() {
                if re != 0.0 || im != 0.0 {
                    let at = if trial % 2 == 0 {
                        (row, col)
                    } else {
                        (col, row)
                    };
                    entries.push((at.0, at.1, element::<T>(re, im)));
                }
            }
        }
        let a = SparseMatrix::from_coo(n, n, &entries);
        for ordering in orderings(&mut r, n) {
            for threshold in [1.0, 0.1, 1e-4] {
                let options = SparseLuOptions::new()
                    .with_ordering(ordering.clone())
                    .with_pivot_threshold(threshold);
                let lu = SparseLu::with_options(&a, &options).unwrap();
                let what = format!("trial {trial} ({ordering:?}, threshold {threshold})");
                assert_refused(&lu, &what);
            }
        }
    }
}

#[test]
fn a_row_or_column_a_multiple_of_another_is_singular_whatever_the_order_and_the_threshold() {
    each_type!(dependent_rows);
}

fn rows_equal_once_scaled<T: Num>() {
    // Row 19 is 7 times row 21, so the scaling makes them equal, and the vector of equal
    // elements that the estimate of the condition starts from has no part along the left null
    // vector of the scaled matrix. The search from it stopped 400 times short of the norm of
    // the inverse, and the matrix was called regular in single precision with the columns in
    // their order and a pivot threshold of 0.01, until a step of the search from the vector of
    // alternating signs was added (found by the round 1 campaign, seed 0x9990619b3ce62a8f).
    let entries = [
        (0, 7, 3.8212890625),
        (0, 18, 6.109375),
        (1, 6, 3.4619140625),
        (1, 7, 6.4375),
        (1, 23, -14.34375),
        (2, 0, 3.853515625),
        (2, 3, 8.890625),
        (2, 4, 1.96875),
        (3, 1, -5.703125),
        (3, 3, 3.958984375),
        (4, 8, 7.40625),
        (4, 20, 7.328125),
        (4, 22, 3.8388671875),
        (4, 23, 10.234375),
        (5, 1, 3.2216796875),
        (5, 9, -0.859375),
        (6, 2, -3.5),
        (6, 8, 3.3095703125),
        (7, 2, 1.515625),
        (7, 11, 6.65625),
        (7, 18, 3.9765625),
        (8, 9, 3.3623046875),
        (8, 11, -14.921875),
        (8, 17, -2.5625),
        (9, 16, 3.4853515625),
        (9, 20, -15.796875),
        (10, 10, 4.609375),
        (10, 20, 3.982421875),
        (11, 11, 14.265625),
        (11, 15, 3.1533203125),
        (12, 9, -4.203125),
        (12, 12, 3.2919921875),
        (13, 3, 4.734375),
        (13, 20, 15.171875),
        (13, 23, 3.7978515625),
        (14, 2, 5.40625),
        (14, 17, 3.0869140625),
        (14, 23, -2.140625),
        (15, 11, 3.8271484375),
        (16, 4, 15.84375),
        (16, 10, 3.392578125),
        (17, 4, 12.765625),
        (17, 5, 9.359375),
        (17, 9, 10.1875),
        (17, 14, 3.751953125),
        (18, 2, 13.25),
        (18, 4, 3.1748046875),
        (18, 14, 1.296875),
        (18, 19, 1.34375),
        (19, 4, 12.140625),
        (19, 13, 23.1943359375),
        (19, 17, 30.734375),
        (19, 21, -34.890625),
        (20, 1, 14.96875),
        (20, 2, 3.912109375),
        (21, 4, 1.734375),
        (21, 13, 3.3134765625),
        (21, 17, 4.390625),
        (21, 21, -4.984375),
        (22, 5, -5.859375),
        (22, 9, 2.296875),
        (22, 10, -2.390625),
        (22, 19, 3.8818359375),
        (23, 5, 3.7001953125),
        (23, 15, 12.609375),
        (23, 19, 16.0),
        (23, 22, 9.3125),
    ];
    let entries: Vec<(usize, usize, T)> = entries
        .iter()
        .map(|&(i, j, x)| (i, j, T::real(x)))
        .collect();
    let a = SparseMatrix::from_coo(24, 24, &entries);
    let mut r = Rng(0x5EED_0032);
    for threshold in [1.0, 0.1, 0.01] {
        let lu = SparseLu::with_options(&a, &natural(threshold)).unwrap();
        assert_refused(&lu, &format!("the matrix, threshold {threshold}"));
    }
    for ordering in orderings(&mut r, 24) {
        let lu = SparseLu::with_ordering(&a, &ordering);
        assert_refused(&lu, &format!("the matrix, {ordering:?}"));
    }
}

#[test]
fn a_singular_matrix_with_two_rows_that_the_scaling_makes_equal_is_reported() {
    each_type!(rows_equal_once_scaled);
}

/// The matrix of a case of signs, the same in every element type.
fn sign_matrix<T: Num>(case: &SignCase) -> SparseMatrix<T> {
    let n = case.rows.len();
    let mut entries = Vec::new();
    for (i, row) in case.rows.iter().enumerate() {
        assert_eq!(row.len(), n, "{}: a row of the wrong length", case.name);
        for (j, sign) in row.bytes().enumerate() {
            match sign {
                b'+' => entries.push((i, j, T::one())),
                b'-' => entries.push((i, j, -T::one())),
                b'.' => {}
                _ => panic!("{}: an element that is not a sign", case.name),
            }
        }
    }
    SparseMatrix::from_coo(n, n, &entries)
}

fn hidden_from_the_fixed_starts<T: Num>() {
    // The 35 exactly singular matrices of signs of review round 2: four consecutive rows with
    // r_i + r_{i+1} = r_{i+2} + r_{i+3}, and in most two equal columns as well. Their left null
    // vector, (…, 1, 1, −1, −1, …), is orthogonal both to the vector of equal elements and to
    // the alternating vector that the estimate of the condition used to start from, so its
    // search saw nothing of the singular direction, and with elements of ±1 its ties went
    // whichever way the rounding took them. In the column order each was found with, every one
    // was called regular at the default threshold, in the type it was found in and as complex
    // numbers, and solved into numbers of the order of 1/ε; the pseudo-random start vector
    // sees them all.
    let mut r = Rng(0x5EED_0040);
    for case in SIGN_CASES.iter().take(size(SIGN_CASES.len(), 2)) {
        let a = sign_matrix::<T>(case);
        let n = a.rows();
        for threshold in [1.0, 0.1, 1e-4] {
            let options = SparseLuOptions::new()
                .with_ordering(Ordering::Given(case.order.to_vec()))
                .with_pivot_threshold(threshold);
            let lu = SparseLu::with_options(&a, &options).unwrap();
            let what = format!("{} in its own order, threshold {threshold}", case.name);
            assert_refused(&lu, &what);
            assert!(factor_error(&lu, &a) < 4.0, "{what}");
        }
        for ordering in orderings(&mut r, n) {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert_refused(&lu, &format!("{} ({ordering:?})", case.name));
        }
    }
}

#[test]
fn a_singular_matrix_whose_null_vector_is_orthogonal_to_the_fixed_start_vectors_is_reported() {
    each_type!(hidden_from_the_fixed_starts);
}

/// A matrix of signs of order `n` whose row `i + 3` is row `i` plus row `i + 1` minus row `i + 2`,
/// for a random `i`, made as the campaign of review round 2 made them: every element `−1`, `0`
/// or `1`, a sum beyond that range being brought back into it by moving the element of row
/// `i + 2` towards it. With `twin_columns`, the matrix has `n − 1` columns of that kind and then
/// one of them again, the columns shuffled.
fn rows_in_fours(r: &mut Rng, n: usize, twin_columns: bool) -> Vec<Vec<i32>> {
    let width = if twin_columns { n - 1 } else { n };
    let sign = |r: &mut Rng| if r.below(2) == 0 { 1 } else { -1 };
    let mut m: Vec<Vec<i32>> = (0..n)
        .map(|_| {
            (0..width)
                .map(|_| if r.below(3) == 0 { sign(r) } else { 0 })
                .collect()
        })
        .collect();
    if !twin_columns {
        for (i, &j) in shuffled(r, n).iter().enumerate() {
            m[i][j] = sign(r);
        }
    }
    let i = r.below(n - 3);
    let (first, second) = m[i..i + 4].split_at_mut(2);
    let (third, fourth) = second.split_at_mut(1);
    let pairs = first[0].iter().zip(&first[1]);
    for ((&a, &b), (c, d)) in pairs.zip(third[0].iter_mut().zip(fourth[0].iter_mut())) {
        let mut s = a + b - *c;
        // A sum above 1 has the element of row `i + 2` below 1, and one below −1 has it above
        // −1, so the element stays a sign as it moves.
        while s.abs() > 1 {
            *c += s.signum();
            s -= s.signum();
        }
        *d = s;
    }
    if !twin_columns {
        return m;
    }
    let twin = r.below(width);
    let order = shuffled(r, n);
    m.iter()
        .map(|row| {
            order
                .iter()
                .map(|&from| row[if from == width { twin } else { from }])
                .collect()
        })
        .collect()
}

fn rows_related_in_fours<T: Num>() {
    // A guard over the two families the 35 cases came from, made afresh: the campaign of
    // review round 2 found the old estimate missing 5 to 15 in 120 000 factorisations of them.
    let mut r = Rng(0x5EED_0041);
    for trial in 0..size(100, 2) {
        let n = 9 + r.below(size(22, 4));
        let m = rows_in_fours(&mut r, n, trial % 2 == 1);
        let mut entries = Vec::new();
        for (i, row) in m.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                if x != 0 {
                    entries.push((i, j, T::real(f64::from(x))));
                }
            }
        }
        let a = SparseMatrix::from_coo(n, n, &entries);
        for ordering in [
            Ordering::Natural,
            Ordering::ReverseCuthillMckee,
            Ordering::Given(shuffled(&mut r, n)),
        ] {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert_refused(&lu, &format!("trial {trial} ({ordering:?})"));
        }
    }
}

#[test]
fn matrices_of_signs_with_four_rows_related_are_singular_in_every_order() {
    each_type!(rows_related_in_fours);
}

fn hidden_by_the_scaling<T: Num>() {
    // Column 9 is 5 times column 8. With the columns in their order and partial pivoting, the
    // computed factors put the matrix, scaled, between τ and 1.4·τ from a singular one — its
    // small rows are eliminated against large ones, which the norm of the scaled matrix does
    // not forgive — while the unscaled reciprocal condition is 0.05·ε. The scaled test alone
    // called it regular in double precision (round 1 campaign, seed 0x10cc542632b4d761); the
    // unscaled one, within its range, does not.
    let entries = [
        (0, 5, 3.7412109375),
        (1, 4, 15.875),
        (1, 5, -1.765625),
        (1, 6, -14.515625),
        (2, 5, -5.46875),
        (2, 6, -3.484375),
        (2, 7, 3.2822265625),
        (2, 8, -13.765625),
        (2, 9, -68.828125),
        (3, 0, 3.4609375),
        (4, 1, -9.734375),
        (4, 4, -2.171875),
        (4, 7, 14.21875),
        (4, 10, 3.294921875),
        (5, 4, 3.1552734375),
        (6, 3, -14.109375),
        (6, 5, -5.0625),
        (6, 8, 3.3955078125),
        (6, 9, 16.9775390625),
        (7, 1, 3.107421875),
        (7, 6, 2.3125),
        (7, 8, 4.46875),
        (7, 9, 22.34375),
        (8, 0, 15.390625),
        (8, 2, -8.96875),
        (8, 6, 3.41015625),
        (8, 7, 9.34375),
        (8, 10, 1.125),
        (9, 1, 1.84375),
        (9, 2, 3.5595703125),
        (9, 7, -9.90625),
        (9, 10, 2.390625),
        (10, 3, 3.0927734375),
    ];
    let entries: Vec<(usize, usize, T)> = entries
        .iter()
        .map(|&(i, j, x)| (i, j, T::real(x)))
        .collect();
    let a = SparseMatrix::from_coo(11, 11, &entries);
    let mut r = Rng(0x5EED_0033);
    for threshold in [1.0, 0.5, 0.1] {
        let lu = SparseLu::with_options(&a, &natural(threshold)).unwrap();
        assert_refused(&lu, &format!("the matrix, threshold {threshold}"));
    }
    for ordering in orderings(&mut r, 11) {
        let lu = SparseLu::with_ordering(&a, &ordering);
        assert_refused(&lu, &format!("the matrix, {ordering:?}"));
    }
}

#[test]
fn a_singular_matrix_that_the_scaled_norm_hides_is_found_by_the_unscaled_one() {
    each_type!(hidden_by_the_scaling);
}

fn near_singular<T: Num>() {
    // [[1, 1], [1, 1 + δ]], its rows and columns scaled as the test scales them, is
    // [[1, 1], [1/(1 + δ), 1]], whose reciprocal condition in the 1-norm is δ/(4·(1 + δ)): below
    // ε for δ up to about 4ε. Its pivots are 1 and δ (or δ/(1 + δ)), neither of them zero.
    let two = |delta: f64| {
        SparseMatrix::from_coo(
            2,
            2,
            &[
                (0, 0, T::one()),
                (0, 1, T::one()),
                (1, 0, T::one()),
                (1, 1, T::real(1.0 + delta)),
            ],
        )
    };
    let eps = T::EPS;
    for (delta, singular) in [
        (eps, true),
        (2.0 * eps, true),
        (16.0 * eps, false),
        (eps.sqrt(), false),
    ] {
        let a = two(delta);
        for ordering in [Ordering::Natural, Ordering::Given(vec![1, 0])] {
            let lu = SparseLu::with_ordering(&a, &ordering);
            let u = lu.u();
            assert!((0..2).all(|i| !u[(i, i)].is_zero()));
            let what = format!("δ = {delta:e} ({ordering:?})");
            if singular {
                assert_refused(&lu, &what);
            } else {
                assert!(!lu.is_singular(), "{what} was found singular");
                assert!(
                    lu.solve_vec(&[T::real(2.0), T::real(2.0 + delta)])
                        .is_some()
                );
            }
        }
    }
    // The Hilbert matrix of order 8: its condition with the rows and columns scaled is 1.0·10¹⁰
    // in double precision, and 1.0·10⁹ once its elements are rounded to single precision
    // (scratch hilbert_check.py, with mpmath): regular in the one, singular to working precision
    // in the other.
    let n = 8;
    let h = SparseMatrix::from_fn(n, n, |i, j| T::real(1.0 / (i + j + 1) as f64));
    let lu = SparseLu::new(&h);
    if eps < 1e-10 {
        assert!(!lu.is_singular());
        let b = h.mul_vec(&vec![T::one(); n]);
        let x = lu.solve_vec(&b).unwrap();
        let error = x
            .iter()
            .fold(0.0f64, |most, &xi| most.max(magnitude(xi - T::one())));
        assert!(error < 1e-3, "an error of {error}");
    } else {
        assert_refused(&lu, "the Hilbert matrix of order 8");
    }
}

#[test]
fn a_matrix_within_rounding_of_a_singular_one_is_singular_to_working_precision() {
    each_type!(near_singular);
}

fn scaled_bound_alone<T: Num>() {
    // A singular matrix of small integers whose scaling makes its condition 2.35 times worse
    // (found by scratch/round2/n12/search2.py), with c·ε added to its element (3, 1). Measured
    // from the inverse of its factors, column by column (scratch probe n12.rs, the same in the
    // four types): for c = 200 the scaled reciprocal condition is 0.65·τ and the unscaled one
    // 1.5·τ, so the matrix is singular to working precision by the scaled bound alone, the
    // unscaled one passing it; for c = 400 they are 1.3·τ and 3.0·τ, and it is regular.
    let s = [
        [-8.0, -4.0, 0.0, 8.0, -4.0],
        [4.0, 3.0, 8.0, -16.0, 8.0],
        [2.0, -1.0, 0.0, 0.0, -1.0],
        [4.0, 1.0, -8.0, 8.0, -4.0],
        [4.0, -4.0, 0.0, 2.0, 2.0],
    ];
    let matrix = |c: f64| {
        let mut entries = Vec::new();
        for (i, row) in s.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                let x = if (i, j) == (3, 1) { x + c * T::EPS } else { x };
                if x != 0.0 {
                    entries.push((i, j, T::real(x)));
                }
            }
        }
        SparseMatrix::from_coo(5, 5, &entries)
    };
    for ordering in [Ordering::Natural, Ordering::Given(vec![4, 3, 2, 1, 0])] {
        let lu = SparseLu::with_ordering(&matrix(200.0), &ordering);
        let u = lu.u();
        assert!((0..5).all(|i| !u[(i, i)].is_zero()));
        assert_refused(&lu, &format!("c = 200 ({ordering:?})"));
        let a = matrix(400.0);
        let lu = SparseLu::with_ordering(&a, &ordering);
        assert!(!lu.is_singular(), "c = 400 ({ordering:?}) reported");
        assert!(lu.solve_vec(&a.mul_vec(&[T::one(); 5])).is_some());
    }
}

#[test]
fn a_matrix_singular_to_working_precision_by_its_scaled_condition_alone_is_reported() {
    each_type!(scaled_bound_alone);
}

fn badly_scaled<T: Num>() {
    // A diagonal matrix of 10⁻²⁰ and 1: perfectly conditioned once scaled, solved exactly.
    let d = SparseMatrix::from_diagonal(&[T::real(1e-20), T::one()]);
    let lu = SparseLu::new(&d);
    assert!(!lu.is_singular());
    assert!(lu.rcond().to_f64() <= 1.0001e-20);
    let x = lu.solve_vec(&[T::real(1e-20), T::one()]).unwrap();
    assert!(x.iter().all(|&xi| magnitude(xi - T::one()) <= 4.0 * T::EPS));
    // Matrices diagonally dominant by rows on a scrambled diagonal, their rows and their
    // columns scaled by powers of ten over twelve orders of magnitude (six in single
    // precision): plain condition numbers far beyond 1/ε, and not one of them singular.
    let decades = if T::EPS < 1e-10 { 6.0 } else { 3.0 };
    let mut r = Rng(0x5EED_0031);
    for trial in 0..size(60, 4) {
        let n = 2 + r.below(size(24, 4));
        let mut entries = Vec::new();
        let mut sums = vec![0.0f64; n];
        for _ in 0..3 * n {
            let (i, j) = (r.below(n), r.below(n));
            let x: T = T::random(&mut r);
            sums[i] += magnitude(x);
            entries.push((i, j, x));
        }
        for (i, &j) in shuffled(&mut r, n).iter().enumerate() {
            entries.push((i, j, T::real(2.0 * sums[i] + 1.0)));
        }
        let rows: Vec<f64> = (0..n).map(|_| 10f64.powf(decades * r.unit())).collect();
        let cols: Vec<f64> = (0..n).map(|_| 10f64.powf(decades * r.unit())).collect();
        let a =
            SparseMatrix::from_coo(n, n, &entries).map(|i, j, x| x * T::real(rows[i] * cols[j]));
        let lu = SparseLu::new(&a);
        assert!(!lu.is_singular(), "trial {trial} was found singular");
        let b = a.mul_vec(&vec![T::one(); n]);
        let x = lu.solve_vec(&b).unwrap();
        let m = a.to_matrix();
        let error = backward_error(
            &m,
            &Matrix::from_fn(n, 1, |i, _| x[i]),
            &Matrix::from_fn(n, 1, |i, _| b[i]),
        );
        assert!(error < 8.0, "trial {trial}: a backward error of {error}");
    }
}

#[test]
fn rows_and_columns_of_very_different_sizes_do_not_make_a_matrix_singular() {
    each_type!(badly_scaled);
}

/// `x` from `P·A·Q = L·U` by substitution with the factors as `l()` and `u()` give them, whether
/// or not the decomposition is reported singular: what its solves would have returned.
fn solve_with_factors<T: Num>(lu: &SparseLu<T>, b: &[T]) -> Vec<T> {
    let n = b.len();
    let (l, u) = (lu.l().to_matrix(), lu.u().to_matrix());
    let (p, q) = (lu.permutation(), lu.column_permutation());
    let mut y: Vec<T> = (0..n).map(|i| b[p[i]]).collect();
    for j in 0..n {
        let yj = y[j];
        for (i, yi) in y.iter_mut().enumerate().skip(j + 1) {
            *yi -= l[(i, j)] * yj;
        }
    }
    for j in (0..n).rev() {
        y[j] /= u[(j, j)];
        let yj = y[j];
        for (i, yi) in y.iter_mut().enumerate().take(j) {
            *yi -= u[(i, j)] * yj;
        }
    }
    let mut x = vec![T::zero(); n];
    for (k, &old) in q.iter().enumerate() {
        x[old] = y[k];
    }
    x
}

fn scaled_rows_or_columns<T: Num>() {
    // Matrices diagonally dominant by rows on a scrambled diagonal, as above, with their rows
    // alone or their columns alone multiplied by powers of ten, and a solution of the sizes of
    // the columns, `x_j = (1 + |u|)/c_j`. This pins what the known limits of `SparseLu` state
    // (review round 2 measured 2 000 of each a type and range). Rows alone never make a matrix
    // reported, up to 10^±10 in single and 10^±20 in double precision. Columns alone are not
    // undone by one pass of the scaling, rows first: up to 10^±3 (single) and 10^±6 (double)
    // nothing is reported, but at 10^±5 and 10^±12 about a quarter of them are, although their
    // factors would have solved them; that is a refusal, never a wrong answer, and an accepted
    // solution is accurate.
    let double = T::EPS < 1e-10;
    let (rows_far, columns_near, columns_far) = if double {
        (20.0, 6.0, 12.0)
    } else {
        (10.0, 3.0, 5.0)
    };
    let mut r = Rng(0x5EED_0042);
    let mut refused = 0;
    let trials = size(100, 3);
    for trial in 0..trials {
        let n = 2 + r.below(size(30, 4));
        let mut entries = Vec::new();
        let mut sums = vec![0.0f64; n];
        for _ in 0..3 * n {
            let (i, j) = (r.below(n), r.below(n));
            let x: T = T::random(&mut r);
            sums[i] += magnitude(x);
            entries.push((i, j, x));
        }
        for (i, &j) in shuffled(&mut r, n).iter().enumerate() {
            entries.push((i, j, T::real(2.0 * sums[i] + 1.0)));
        }
        let m = SparseMatrix::from_coo(n, n, &entries);
        let given = Ordering::Given(shuffled(&mut r, n));
        let rows: Vec<f64> = (0..n).map(|_| 10f64.powf(rows_far * r.unit())).collect();
        let by_rows = m.map(|i, _, x| x * T::real(rows[i]));
        let lu = SparseLu::with_ordering(&by_rows, &given);
        assert!(
            !lu.is_singular(),
            "trial {trial}: rows scaled by 10^±{rows_far} reported"
        );
        for (decades, may_refuse) in [(columns_near, false), (columns_far, true)] {
            let columns: Vec<f64> = (0..n).map(|_| 10f64.powf(decades * r.unit())).collect();
            let a = m.map(|_, j, x| x * T::real(columns[j]));
            let truth: Vec<T> = columns
                .iter()
                .map(|&c| T::real((1.0 + r.unit().abs()) / c))
                .collect();
            let b = a.mul_vec(&truth);
            let lu = SparseLu::with_ordering(&a, &given);
            let x = if lu.is_singular() {
                assert!(
                    may_refuse,
                    "trial {trial}: columns scaled by 10^±{decades} reported"
                );
                refused += 1;
                assert!(lu.solve_vec(&b).is_none());
                solve_with_factors(&lu, &b)
            } else {
                lu.solve_vec(&b).unwrap()
            };
            let top = truth.iter().fold(0.0f64, |most, &t| most.max(magnitude(t)));
            let error = x
                .iter()
                .zip(&truth)
                .fold(0.0f64, |most, (&xi, &ti)| most.max(magnitude(xi - ti)))
                / top;
            // At most 3·10⁻⁶ was measured, refused or not, but for a few refused in single
            // precision.
            assert!(
                error < 1e-4,
                "trial {trial}, columns by 10^±{decades}: an error of {error:e}"
            );
        }
    }
    // The limit is there: without it the documentation would have to change.
    if !cfg!(miri) {
        assert!(
            refused >= trials / 10,
            "{refused} of {trials} matrices with columns scaled by 10^±{columns_far} reported"
        );
    }
}

#[test]
fn columns_far_apart_in_size_can_be_refused_though_never_solved_wrongly_and_rows_never_are() {
    each_type!(scaled_rows_or_columns);
}

/// Wilkinson's matrix of order `n`: ones on the diagonal and in the last column, −1 below the
/// diagonal.
fn wilkinson<T: Num>(n: usize) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    for i in 0..n {
        entries.push((i, i, T::one()));
        if i + 1 < n {
            entries.push((i, n - 1, T::one()));
        }
        for j in 0..i {
            entries.push((i, j, -T::one()));
        }
    }
    SparseMatrix::from_coo(n, n, &entries)
}

fn refused_for_growth<T: Num>() {
    // Partial pivoting keeps every pivot of Wilkinson's matrix on the diagonal (the ties go to
    // it) and doubles the last column at each step: growth 2^(n−1), every element of the factors
    // exact, determinant 2^(n−1), condition number n in the 1-norm, and rows and columns of
    // largest element 1, so that the matrix is its own scaled one. The test refuses it once
    // 1/n < ε·2^(n−1), from order 20 in single and 48 in double precision, as the documentation
    // of `det` and the known limits of `SparseLu` say, and `det` is then 0.
    let first = if T::EPS < 1e-10 { 48 } else { 20 };
    for n in [first - 1, first] {
        let a = wilkinson::<T>(n);
        let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
        let growth = 2f64.powi(n as i32 - 1);
        assert_eq!(lu.permutation(), (0..n).collect::<Vec<_>>());
        assert_eq!(lu.growth().to_f64(), growth);
        assert_eq!(lu.u()[(n - 1, n - 1)], T::real(growth));
        assert_eq!(factor_error(&lu, &a), 0.0);
        if n < first {
            assert!(!lu.is_singular(), "order {n} reported");
            assert_eq!(lu.det(), T::real(growth));
            let rcond = lu.rcond().to_f64();
            assert!((rcond * n as f64 - 1.0).abs() < 1e-3, "rcond {rcond:e}");
        } else {
            assert_refused(&lu, &format!("Wilkinson's matrix of order {n}"));
        }
    }
}

#[test]
fn wilkinsons_matrix_is_refused_for_its_growth_and_its_determinant_is_then_zero() {
    each_type!(refused_for_growth);
}

fn alternating<T: Num>() {
    // Found by a search (scratch probe hager.rs) that replays the estimator on the dense
    // inverse: the gradient search stops at 0.056 of ‖A⁻¹‖₁, and the vector of alternating
    // signs that Higham's refinement adds reaches 0.78 of it. Elements k/256, exact in every
    // type.
    let entries = [
        (0, 0, 3.17578125),
        (0, 1, -2.80078125),
        (0, 2, 0.16015625),
        (1, 0, -2.80078125),
        (1, 1, 2.03125),
        (1, 2, 4.07421875),
        (2, 0, -2.765625),
        (2, 1, 2.2109375),
        (2, 2, 4.12109375),
    ];
    let entries: Vec<(usize, usize, T)> = entries
        .iter()
        .map(|&(i, j, x)| (i, j, T::real(x)))
        .collect();
    let a = SparseMatrix::from_coo(3, 3, &entries);
    let exact = 1.0 / a.to_matrix().cond(Norm::One).to_f64();
    for ordering in [Ordering::Natural, Ordering::Given(vec![2, 0, 1])] {
        let estimate = SparseLu::with_ordering(&a, &ordering).rcond().to_f64();
        let ratio = estimate / exact;
        // Without the alternating vector the estimate would be about 18 times the true value.
        assert!(
            (1.0 - 1e3 * T::EPS..1.5).contains(&ratio),
            "an estimate {ratio} times the true reciprocal condition"
        );
    }
}

#[test]
fn the_condition_estimate_uses_the_vector_of_alternating_signs_where_the_search_stops_short() {
    each_type!(alternating);
}

fn tiny_elements<T: Num>() {
    // The smallest positive number of the type: perfectly conditioned, though its inverse is
    // beyond the range of the type. The draft of review round 1 gave a reciprocal condition of
    // 0, which is reserved for singular matrices.
    let tiny = if T::EPS < 1e-10 {
        f64::from_bits(1)
    } else {
        f64::from(f32::from_bits(1))
    };
    let a = SparseMatrix::from_diagonal(&[T::real(tiny)]);
    let lu = SparseLu::new(&a);
    assert!(!lu.is_singular());
    assert_eq!(lu.rcond().to_f64(), 1.0);
    assert!(lu.solve_vec(&[T::real(tiny)]).unwrap() == [T::one()]);
    // And of order 3, the elements of a well-conditioned matrix scaled down to the underflow.
    let b = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(4.0 * tiny)),
            (0, 1, T::real(tiny)),
            (1, 1, T::real(4.0 * tiny)),
            (2, 1, T::real(tiny)),
            (2, 2, T::real(4.0 * tiny)),
        ],
    );
    let lu = SparseLu::new(&b);
    assert!(!lu.is_singular());
    assert!(lu.rcond().to_f64() > 0.1);
}

#[test]
fn a_matrix_of_tiny_elements_is_neither_singular_nor_of_zero_condition() {
    each_type!(tiny_elements);
}

#[test]
#[should_panic(expected = "column ordering")]
fn a_given_ordering_of_the_wrong_shape_panics_whatever_the_threshold() {
    // Shapes are checked before the numbers: the threshold outside (0, 1] does not turn the
    // panic into `None`.
    let a = SparseMatrix::<f64>::identity(3);
    let options = SparseLuOptions::new()
        .with_ordering(Ordering::Given(vec![0, 0, 0]))
        .with_pivot_threshold(2.0);
    let _ = SparseLu::with_options(&a, &options);
}

fn not_finite<T: Num>() {
    let mut r = Rng(0x5EED_000F);
    let n = size(9, 4);
    let a: SparseMatrix<T> = well_conditioned(&mut r, n, 3, 100.0);
    let b = Matrix::from_fn(n, 1, |_, _| T::random(&mut r));
    // One element at a time replaced by a NaN or an infinity, wherever it falls in the factors.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for target in 0..a.nnz().min(size(40, 6)) {
            let mut seen = 0;
            let poisoned = a.map(|_, _, x| {
                seen += 1;
                if seen - 1 == target { T::real(bad) } else { x }
            });
            for ordering in [Ordering::Natural, Ordering::Given(shuffled(&mut r, n))] {
                for threshold in [1.0, 0.01] {
                    let options = SparseLuOptions::new()
                        .with_ordering(ordering.clone())
                        .with_pivot_threshold(threshold);
                    let lu = SparseLu::with_options(&poisoned, &options).unwrap();
                    assert!(lu.is_singular(), "a matrix with {bad} passed for regular");
                    assert!(lu.solve(&b).is_none());
                    assert!(lu.solve_transpose(&b).is_none());
                    assert!(lu.solve_conjugate_transpose(&b).is_none());
                    assert!(lu.solve_vec(b.column(0)).is_none());
                    assert!(
                        magnitude(lu.det()).is_nan(),
                        "a determinant of {:?}",
                        lu.det()
                    );
                    assert!(lu.growth().to_f64().is_nan());
                    assert_eq!(lu.rcond().to_f64(), 0.0);
                    // The accessors still answer.
                    assert!(is_permutation(lu.permutation(), n));
                    assert!(is_permutation(lu.column_permutation(), n));
                    assert_eq!(lu.l().rows(), n);
                    assert_eq!(lu.u().rows(), n);
                    assert!(lu.nnz() >= n);
                }
            }
        }
    }
    // An elimination that overflows, of a matrix that is finite.
    let huge = <T::Real as Float>::MAX.to_f64() * 0.75;
    let overflowing = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, T::real(huge)),
            (0, 1, T::real(huge)),
            (1, 0, T::real(huge)),
            (1, 1, T::real(-huge)),
        ],
    );
    let lu = SparseLu::new(&overflowing);
    assert!(lu.is_singular());
    assert!(lu.solve_vec(&[T::one(), T::one()]).is_none());
    assert!(magnitude(lu.det()).is_nan());
}

#[test]
fn a_matrix_that_is_not_finite_is_reported_and_never_solved() {
    each_type!(not_finite);
}

fn bad_right_hand_sides<T: Num>() {
    let mut r = Rng(0x5EED_0010);
    let n = size(10, 4);
    let a: SparseMatrix<T> = well_conditioned(&mut r, n, 2, 100.0);
    let lu = SparseLu::new(&a);
    let good: Vec<T> = (0..n).map(|_| T::random(&mut r)).collect();
    assert!(lu.solve_vec(&good).is_some());
    for bad in [f64::NAN, f64::INFINITY] {
        for at in 0..n {
            let mut b = good.clone();
            b[at] = T::real(bad);
            assert!(lu.solve_vec(&b).is_none());
            assert!(lu.solve_transpose_vec(&b).is_none());
            assert!(lu.solve_conjugate_transpose_vec(&b).is_none());
            // One bad column refuses the whole matrix of right-hand sides.
            let both = Matrix::from_columns(&[good.clone(), b]);
            assert!(lu.solve(&both).is_none());
            assert!(lu.solve_transpose(&both).is_none());
            assert!(lu.solve_conjugate_transpose(&both).is_none());
        }
    }
    // A solution that overflows is not returned either.
    let tiny = SparseMatrix::from_diagonal(&[T::real(<T::Real as Float>::MIN_POSITIVE.to_f64())]);
    let lu = SparseLu::new(&tiny);
    assert!(!lu.is_singular());
    assert!(lu.solve_vec(&[T::real(0.5)]).is_some());
    assert!(lu.solve_vec(&[T::real(1e30)]).is_none());
}

#[test]
fn a_right_hand_side_or_a_solution_that_is_not_finite_gives_none() {
    each_type!(bad_right_hand_sides);
}

fn edges<T: Num>() {
    // No rows and no columns: regular, with the determinant of the empty product.
    let empty = SparseMatrix::<T>::zeros(0, 0);
    for lu in [
        SparseLu::new(&empty),
        SparseLu::with_ordering(&empty, &Ordering::Given(Vec::new())),
        SparseLu::with_options(&empty, &natural(0.5)).unwrap(),
    ] {
        assert!(!lu.is_singular());
        assert!(lu.det() == T::one());
        assert_eq!(lu.nnz(), 0);
        assert_eq!(lu.rcond().to_f64(), 1.0);
        assert_eq!(lu.growth().to_f64(), 1.0);
        assert!(lu.permutation().is_empty() && lu.column_permutation().is_empty());
        for factor in [lu.l(), lu.u(), lu.p(), lu.q()] {
            assert_eq!((factor.rows(), factor.cols(), factor.nnz()), (0, 0, 0));
        }
        assert_eq!(lu.solve_vec(&[]), Some(Vec::new()));
        assert_eq!(lu.solve_transpose_vec(&[]), Some(Vec::new()));
        assert_eq!(lu.solve_conjugate_transpose_vec(&[]), Some(Vec::new()));
        let x = lu.solve(&Matrix::zeros(0, 3)).unwrap();
        assert_eq!((x.rows(), x.cols()), (0, 3));
    }
    // One element.
    let one = SparseMatrix::from_coo(1, 1, &[(0, 0, T::real(-4.0))]);
    let lu = SparseLu::new(&one);
    assert!(!lu.is_singular());
    assert!(lu.det() == T::real(-4.0));
    assert_eq!(lu.nnz(), 1);
    assert_eq!(lu.rcond().to_f64(), 1.0);
    assert_eq!(lu.growth().to_f64(), 1.0);
    assert!(lu.l() == SparseMatrix::identity(1));
    assert!(lu.u() == one);
    assert!(lu.p() == SparseMatrix::identity(1) && lu.q() == SparseMatrix::identity(1));
    assert!(lu.solve_vec(&[T::real(2.0)]).unwrap() == [T::real(-0.5)]);
    assert!(lu.solve_transpose_vec(&[T::real(2.0)]).unwrap() == [T::real(-0.5)]);
    assert!(lu.solve_conjugate_transpose_vec(&[T::real(2.0)]).unwrap() == [T::real(-0.5)]);
    // The identity and a diagonal matrix, of any order.
    let n = size(50, 6);
    let lu = SparseLu::new(&SparseMatrix::<T>::identity(n));
    assert!(lu.det() == T::one());
    assert_eq!(lu.nnz(), n);
    // The first vector of the estimate has the elements 1/n, which are rounded, and their sum
    // with them: the estimate is 1 up to that rounding.
    let rcond = lu.rcond().to_f64();
    assert!(
        rcond <= 1.0 && 1.0 - rcond <= 4.0 * n as f64 * T::EPS,
        "{rcond}"
    );
    let b: Vec<T> = (0..n).map(|i| T::real(i as f64)).collect();
    assert!(lu.solve_vec(&b).unwrap() == b);
}

#[test]
fn the_empty_matrix_and_the_matrix_of_one_element_are_decomposed() {
    each_type!(edges);
}

#[test]
fn a_complex_one_by_one_matrix_conjugates_in_the_conjugate_transposed_solve_alone() {
    let a = SparseMatrix::from_coo(1, 1, &[(0, 0, Complex::new(0.0, 2.0f64))]);
    let lu = SparseLu::new(&a);
    let b = [Complex::new(4.0, 0.0)];
    assert_eq!(lu.solve_vec(&b).unwrap(), [Complex::new(0.0, -2.0)]);
    assert_eq!(
        lu.solve_transpose_vec(&b).unwrap(),
        [Complex::new(0.0, -2.0)]
    );
    assert_eq!(
        lu.solve_conjugate_transpose_vec(&b).unwrap(),
        [Complex::new(0.0, 2.0)]
    );
    assert_eq!(lu.det(), Complex::new(0.0, 2.0));
}

#[test]
#[should_panic(expected = "isn't square")]
fn a_matrix_that_is_not_square_panics() {
    let _ = SparseLu::new(&SparseMatrix::<f64>::zeros(2, 3));
}

#[test]
#[should_panic(expected = "isn't square")]
fn a_matrix_that_is_not_square_panics_with_an_ordering() {
    let _ = SparseLu::with_ordering(&SparseMatrix::<f64>::zeros(3, 2), &Ordering::Natural);
}

#[test]
#[should_panic(expected = "isn't square")]
fn a_matrix_that_is_not_square_panics_with_options() {
    let _ = SparseLu::with_options(&SparseMatrix::<f32>::zeros(1, 2), &SparseLuOptions::new());
}

#[test]
#[should_panic]
fn a_given_ordering_that_repeats_a_column_panics() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = SparseLu::with_ordering(&a, &Ordering::Given(vec![0, 1, 1]));
}

#[test]
#[should_panic]
fn a_given_ordering_of_the_wrong_length_panics() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = SparseLu::with_ordering(&a, &Ordering::Given(vec![0, 1]));
}

#[test]
#[should_panic(expected = "right-hand side")]
fn a_right_hand_side_with_the_wrong_number_of_rows_panics() {
    let lu = SparseLu::new(&SparseMatrix::<f64>::identity(3));
    let _ = lu.solve(&Matrix::zeros(2, 1));
}

#[test]
#[should_panic(expected = "right-hand side")]
fn a_right_hand_side_of_the_wrong_length_panics() {
    let lu = SparseLu::new(&SparseMatrix::<f64>::identity(3));
    let _ = lu.solve_vec(&[1.0; 4]);
}

#[test]
#[should_panic(expected = "right-hand side")]
fn a_transposed_right_hand_side_of_the_wrong_shape_panics() {
    let lu = SparseLu::new(&SparseMatrix::<f64>::identity(3));
    let _ = lu.solve_transpose(&Matrix::zeros(4, 1));
}

#[test]
#[should_panic(expected = "right-hand side")]
fn a_conjugate_transposed_right_hand_side_of_the_wrong_length_panics() {
    let lu = SparseLu::new(&SparseMatrix::<f64>::identity(3));
    let _ = lu.solve_conjugate_transpose_vec(&[1.0; 2]);
}

#[test]
#[should_panic(expected = "right-hand side")]
fn a_right_hand_side_of_the_wrong_shape_panics_before_singularity_is_asked() {
    let lu = SparseLu::new(&SparseMatrix::<f64>::zeros(3, 3));
    let _ = lu.solve_vec(&[1.0; 2]);
}

fn long_chain<T: Num>() {
    // Each column of L has one element, in the next row, and the last column of the matrix has
    // an element in the first row: its elimination passes through every column in turn. A
    // search that called itself would need a stack as deep as the matrix.
    let n = size(60_000, 40);
    let mut entries = Vec::with_capacity(2 * n);
    for i in 0..n {
        entries.push((i, i, T::real(2.0)));
        if i + 1 < n {
            entries.push((i + 1, i, T::real(1.0)));
        }
    }
    entries.push((0, n - 1, T::real(1.0)));
    let a = SparseMatrix::from_coo(n, n, &entries);
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert!(!lu.is_singular());
    // The last column fills in from top to bottom, and nothing else does.
    assert_eq!(lu.nnz(), a.nnz() + n - 2);
    let x: Vec<T> = (0..n).map(|i| T::real((i % 3) as f64 - 1.0)).collect();
    let b = a.mul_vec(&x);
    let solved = lu.solve_vec(&b).unwrap();
    let error = x
        .iter()
        .zip(&solved)
        .fold(0.0f64, |most, (&p, &q)| most.max(magnitude(p - q)));
    assert!(error <= 64.0 * T::EPS, "an error of {error}");
}

#[test]
fn a_chain_of_dependencies_as_long_as_the_matrix_needs_no_recursion() {
    each_type!(long_chain);
}

fn stays_sparse<T: Num>() {
    // A tridiagonal matrix that needs no pivoting has factors of the same pattern, whatever its
    // order: nothing here may cost the square of the order, in time or in memory.
    let n = size(150_000, 60);
    let a = SparseMatrix::tridiagonal(
        &vec![T::real(-1.0); n - 1],
        &vec![T::real(4.0); n],
        &vec![T::real(1.5); n - 1],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.nnz(), 3 * n - 2);
    assert!(lu.permutation().iter().copied().eq(0..n));
    let x: Vec<T> = (0..n).map(|i| T::real(1.0 + (i % 4) as f64)).collect();
    let solved = lu.solve_vec(&a.mul_vec(&x)).unwrap();
    let error = x
        .iter()
        .zip(&solved)
        .fold(0.0f64, |most, (&p, &q)| most.max(magnitude(p - q)));
    assert!(error <= 64.0 * T::EPS, "an error of {error}");
    // The permutation matrices and the factors are built in time proportional to their
    // nonzeros too.
    assert_eq!(lu.p().nnz(), n);
    assert_eq!(lu.q().nnz(), n);
    assert_eq!(lu.l().nnz() + lu.u().nnz(), lu.nnz() + n);
    // The same matrix with its rows swapped in pairs, (0 1)(2 3)…: partial pivoting swaps them
    // back, step by step, and the factors are those of the matrix in its own order.
    let pairs: Vec<usize> = (0..n).map(|i| if i ^ 1 < n { i ^ 1 } else { i }).collect();
    let mut swapped = Vec::with_capacity(a.nnz());
    for (new, &old) in pairs.iter().enumerate() {
        swapped.extend(a.row(old).map(|(j, x)| (new, j, x)));
    }
    let b = SparseMatrix::from_coo(n, n, &swapped);
    let pivoted = SparseLu::with_ordering(&b, &Ordering::Natural);
    assert!(!pivoted.is_singular());
    assert_eq!(pivoted.permutation(), pairs.as_slice());
    assert_eq!(pivoted.nnz(), 3 * n - 2);
    // `P·B` is `A` again, eliminated in the same arithmetic.
    assert!(pivoted.l() == lu.l() && pivoted.u() == lu.u());
    let solved = pivoted.solve_vec(&b.mul_vec(&x)).unwrap();
    let error = x
        .iter()
        .zip(&solved)
        .fold(0.0f64, |most, (&p, &q)| most.max(magnitude(p - q)));
    assert!(error <= 64.0 * T::EPS, "an error of {error}");
}

#[test]
fn the_factors_of_a_banded_matrix_stay_banded_at_any_order() {
    each_type!(stays_sparse);
}

fn grid_fill<T: Num>() {
    // The Laplacian of a grid in its natural order fills its band and nothing more, and the
    // pivots stay on the diagonal.
    let k = size(24, 5);
    let n = k * k;
    let a: SparseMatrix<T> = convection_diffusion(k, 0.0, 0.0);
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert!(lu.permutation().iter().copied().eq(0..n));
    assert_eq!(lu.nnz(), 2 * ((k - 1) + (n - k) * k) + n);
    assert_eq!(lu.l().nnz() + lu.u().nnz(), lu.nnz() + n);
    assert!(factor_error(&lu, &a) < 4.0);
    // The nested dissection of the grid fills far less.
    let dissected = SparseLu::with_ordering(&a, &Ordering::Given(dissection(k)));
    assert!(factor_error(&dissected, &a) < 4.0);
    // At k = 24 the dissection keeps 0.55 of the fill of the band (14 858 elements against
    // 27 118), and its advantage grows with the grid.
    if k >= 24 {
        assert!(10 * dissected.nnz() < 6 * lu.nnz());
    }
    // Convection that takes the dominance away moves rows, and the system is solved all the same.
    let a: SparseMatrix<T> = convection_diffusion(k, 1.5, 3.5);
    let dense = a.to_matrix();
    let b = Matrix::from_fn(n, 1, |i, _| T::real(((i * 7) % 11) as f64 - 5.0));
    for ordering in [Ordering::Natural, Ordering::Given(dissection(k))] {
        let lu = SparseLu::with_ordering(&a, &ordering);
        assert_ne!(lu.permutation(), lu.column_permutation());
        assert!(factor_error(&lu, &a) < 4.0);
        let x = lu.solve(&b).unwrap();
        assert!(backward_error(&dense, &x, &b) < 8.0);
        let xt = lu.solve_transpose(&b).unwrap();
        assert!(backward_error(&dense.transpose(), &xt, &b) < 8.0);
    }
}

#[test]
fn grid_problems_fill_as_their_ordering_says() {
    each_type!(grid_fill);
}

fn counts_and_growth<T: Num>() {
    let mut r = Rng(0x5EED_0011);
    for _ in 0..size(20, 2) {
        let n = 1 + r.below(size(30, 6));
        let a: SparseMatrix<T> = well_conditioned(&mut r, n, 2, 100.0);
        for threshold in [1.0, 0.05] {
            let lu = SparseLu::with_options(&a, &natural(threshold)).unwrap();
            let (l, u) = (lu.l(), lu.u());
            // Random elements cancel nowhere: every element the elimination reaches is nonzero.
            assert_eq!(l.nnz() + u.nnz(), lu.nnz() + n);
            assert!(lu.nnz() >= a.nnz());
            let expected = largest(&u.to_matrix()) / largest(&a.to_matrix());
            let growth = lu.growth().to_f64();
            assert!(
                (growth - expected).abs() <= 1e-5 * expected,
                "a growth of {growth}, not {expected}"
            );
        }
    }
    // An element the elimination makes exactly zero is still counted, and left out of the factor.
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(2.0)),
            (0, 1, T::real(2.0)),
            (1, 0, T::real(1.0)),
            (1, 1, T::real(1.0)),
            (1, 2, T::real(1.0)),
            (2, 1, T::real(1.0)),
            (2, 2, T::real(1.0)),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert!(!lu.is_singular());
    assert_eq!(lu.permutation(), [0, 2, 1]);
    // L has (1, 0) and a zero where row 1 of A met the second column; U has six elements.
    assert_eq!(lu.nnz(), 7);
    assert_eq!(lu.l().nnz() + lu.u().nnz(), lu.nnz() + 3 - 1);
    // The pivots 2, 1 and 1, and the odd permutation of the rows: 2·(1 − 1) − 2·(1 − 0) = −2.
    assert!(lu.det() == T::real(-2.0));
}

#[test]
fn the_count_of_elements_and_the_growth_are_those_of_the_factors() {
    each_type!(counts_and_growth);
}

fn condition_estimates<T: Num>() {
    let mut r = Rng(0x5EED_0012);
    let mut closest = f64::INFINITY;
    let mut farthest = 0.0f64;
    for trial in 0..size(40, 3) {
        let n = 1 + r.below(size(36, 6));
        let limit = if trial % 2 == 0 { 100.0 } else { 1e4 };
        let per_row = 1 + r.below(3);
        let a: SparseMatrix<T> = well_conditioned(&mut r, n, per_row, limit);
        let exact = 1.0 / a.to_matrix().cond(Norm::One).to_f64();
        for ordering in [Ordering::Natural, Ordering::Given(shuffled(&mut r, n))] {
            let estimate = SparseLu::with_ordering(&a, &ordering).rcond().to_f64();
            assert!(estimate > 0.0 && estimate <= 1.0);
            let ratio = estimate / exact;
            closest = closest.min(ratio);
            farthest = farthest.max(ratio);
            // Never below the true value but for the rounding of the solves, whose relative
            // error is of the order of the condition number in units of the last place.
            assert!(
                ratio >= 1.0 - 64.0 * n as f64 * T::EPS / exact,
                "an estimate {ratio} of the true reciprocal condition"
            );
            assert!(
                ratio <= 4.0,
                "an estimate {ratio} of the true reciprocal condition"
            );
        }
    }
    assert!(closest <= farthest);
    // Exact where the inverse is known: a diagonal matrix, and a triangular one whose inverse
    // has one large column.
    let d =
        SparseMatrix::from_diagonal(&[T::real(4.0), T::real(-0.5), T::real(0.25), T::real(2.0)]);
    assert_eq!(SparseLu::new(&d).rcond().to_f64(), 0.0625);
    let t = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(1.0)),
            (0, 1, T::real(-8.0)),
            (1, 1, T::real(1.0)),
            (1, 2, T::real(-8.0)),
            (2, 2, T::real(1.0)),
        ],
    );
    // The inverse has 1, 8, 64 in its last column: norms of 9 and 73.
    let estimate = SparseLu::new(&t).rcond().to_f64();
    assert!((estimate - 1.0 / (9.0 * 73.0)).abs() <= 1e-6 / (9.0 * 73.0));
}

#[test]
fn the_condition_estimate_is_a_close_lower_bound_of_the_condition_number() {
    each_type!(condition_estimates);
}

fn singular_by_pattern<T: Num>() {
    // Rows 1 and 2 have their only elements in column 0, so no elements can make the matrix
    // regular. Its elimination in this order meets no pivot of zero: 0.1/0.3 is rounded, and the
    // last pivot comes out at the level of the rounding error (1e-18 in double precision).
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(0.3)),
            (0, 1, T::real(0.1)),
            (0, 2, T::real(0.1)),
            (1, 0, T::real(0.1)),
            (2, 0, T::real(0.3)),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    let u = lu.u();
    assert!(
        (0..3).all(|i| !u[(i, i)].is_zero()),
        "a pivot is zero after all"
    );
    assert!(magnitude(u[(2, 2)]) <= 64.0 * T::EPS);
    assert!(lu.is_singular());
    assert!(lu.det() == T::zero());
    assert_eq!(lu.rcond().to_f64(), 0.0);
    assert!(lu.solve_vec(&[T::one(), T::one(), T::one()]).is_none());
    assert!(
        lu.solve_transpose_vec(&[T::one(), T::one(), T::one()])
            .is_none()
    );
    assert!(factor_error(&lu, &a) < 4.0);
}

#[test]
fn a_matrix_singular_by_its_pattern_is_reported_though_rounding_hides_its_zero_pivot() {
    each_type!(singular_by_pattern);
}

fn structural_singularity<T: Num>() {
    // Thin random patterns whose structural rank SciPy computed.
    let mut r = Rng(0x5EED_0013);
    for case in STRUCTURAL.iter().take(size(STRUCTURAL.len(), 2)) {
        let n = case.n;
        let entries: Vec<(usize, usize, T)> = case
            .entries
            .iter()
            .map(|&(i, j, x)| (i, j, T::real(x)))
            .collect();
        let a = SparseMatrix::from_coo(n, n, &entries);
        assert_eq!(a.nnz(), case.entries.len());
        assert_eq!(
            structural_rank(&a),
            case.rank,
            "the oracle disagrees with SciPy"
        );
        let dense = a.to_matrix();
        for ordering in orderings(&mut r, n) {
            for threshold in [1.0, 0.1] {
                let options = SparseLuOptions::new()
                    .with_ordering(ordering.clone())
                    .with_pivot_threshold(threshold);
                let lu = SparseLu::with_options(&a, &options).unwrap();
                assert_eq!(
                    lu.is_singular(),
                    case.rank < n,
                    "rank {} of {n}, {ordering:?}",
                    case.rank
                );
                assert!(factor_error(&lu, &a) < 4.0);
                let b = Matrix::from_fn(n, 1, |i, _| T::real(1.0 + i as f64));
                if case.rank < n {
                    assert!(lu.det() == T::zero());
                    assert!(lu.solve(&b).is_none());
                } else {
                    let x = lu.solve(&b).unwrap();
                    assert!(backward_error(&dense, &x, &b) < 8.0);
                }
            }
        }
    }
    // And many more against the oracle: a structurally singular matrix is reported whatever the
    // order of its columns, whose elimination may or may not meet a zero pivot.
    let mut singular = 0;
    for _ in 0..size(300, 12) {
        let n = 1 + r.below(size(40, 6));
        let extra = r.below(n + 1);
        let a: SparseMatrix<T> = random_sparse(&mut r, n, n, n + extra);
        let deficient = structural_rank(&a) < n;
        singular += usize::from(deficient);
        for ordering in [
            Ordering::Natural,
            Ordering::Colamd,
            Ordering::Given(shuffled(&mut r, n)),
        ] {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert!(factor_error(&lu, &a) < 4.0);
            if deficient {
                assert!(
                    lu.is_singular(),
                    "a structurally singular matrix passed ({ordering:?})"
                );
                assert!(lu.det() == T::zero());
                assert!(lu.solve_vec(&vec![T::one(); n]).is_none());
            }
        }
    }
    assert!(
        singular >= size(100, 2),
        "only {singular} patterns were structurally singular"
    );
}

#[test]
fn structural_singularity_is_found_whatever_the_column_order() {
    each_type!(structural_singularity);
}

// The defects of the draft this module replaced, from the audit `p_direct`, each on the matrices
// that showed it.

fn interchanged_rows<T: Num>() {
    // p_direct-1: [[1, 2], [3, 4]]·x = [3, 7] has x = [1, 1] and the determinant −2; partial
    // pivoting takes the 3, and the factors must follow the interchange of the rows.
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, T::real(1.0)),
            (0, 1, T::real(2.0)),
            (1, 0, T::real(3.0)),
            (1, 1, T::real(4.0)),
        ],
    );
    let tolerance = 8.0 * T::EPS;
    for lu in [
        SparseLu::new(&a),
        SparseLu::with_ordering(&a, &Ordering::Natural),
    ] {
        assert!(!lu.is_singular());
        let x = lu.solve_vec(&[T::real(3.0), T::real(7.0)]).unwrap();
        assert!(
            x.iter().all(|&v| magnitude(v - T::one()) <= tolerance),
            "{x:?}"
        );
        assert!(
            magnitude(lu.det() - T::real(-2.0)) <= 2.0 * tolerance,
            "{:?}",
            lu.det()
        );
        assert!(factor_error(&lu, &a) < 4.0);
    }
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.permutation(), [1, 0]);
    let l = Matrix::from_rows(&[[T::one(), T::zero()], [T::real(1.0 / 3.0), T::one()]]);
    let u = Matrix::from_rows(&[
        [T::real(3.0), T::real(4.0)],
        [T::zero(), T::real(2.0 / 3.0)],
    ]);
    assert!(largest(&(&lu.l().to_matrix() - &l)) <= tolerance);
    assert!(largest(&(&lu.u().to_matrix() - &u)) <= tolerance);
    // Rows 0 and 2 alike: exactly singular, and every step of the elimination is exact.
    let s = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(1.0)),
            (0, 1, T::real(2.0)),
            (0, 2, T::real(-1.0)),
            (1, 0, T::real(-2.0)),
            (1, 2, T::real(1.0)),
            (2, 0, T::real(1.0)),
            (2, 1, T::real(2.0)),
            (2, 2, T::real(-1.0)),
        ],
    );
    let mut r = Rng(0x5EED_0014);
    for ordering in orderings(&mut r, 3) {
        let lu = SparseLu::with_ordering(&s, &ordering);
        assert!(lu.is_singular(), "{ordering:?}");
        assert!(lu.det() == T::zero());
        assert!(
            lu.solve(&column(&[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]))
                .is_none()
        );
    }
}

#[test]
fn the_factors_follow_an_interchange_of_rows() {
    each_type!(interchanged_rows);
}

/// The elements of a small matrix, `(row, column, value)`.
type Entries = &'static [(usize, usize, f64)];

fn eliminated_pivot<T: Num>() {
    // p_direct-2 (a): the pivot of the second column exists only once the first is taken off
    // it. Matrices the draft called singular, with their orders and determinants.
    let cases: [(Entries, usize, f64); 3] = [
        (&[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0)], 2, -1.0),
        (
            &[
                (0, 0, 2.0),
                (0, 2, 1.0),
                (1, 1, 2.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
            ],
            3,
            -4.0,
        ),
        (
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 1.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (2, 2, 1.0),
            ],
            3,
            -1.0,
        ),
    ];
    let mut r = Rng(0x5EED_0015);
    for (entries, n, det) in cases {
        let entries: Vec<(usize, usize, T)> = entries
            .iter()
            .map(|&(i, j, x)| (i, j, T::real(x)))
            .collect();
        let a = SparseMatrix::from_coo(n, n, &entries);
        let ones = vec![T::one(); n];
        let b = a.mul_vec(&ones);
        for ordering in orderings(&mut r, n) {
            let lu = SparseLu::with_ordering(&a, &ordering);
            assert!(!lu.is_singular(), "{ordering:?}");
            assert!(
                magnitude(lu.det() - T::real(det)) <= 16.0 * T::EPS,
                "{:?}",
                lu.det()
            );
            let x = lu.solve_vec(&b).unwrap();
            assert!(
                x.iter().all(|&v| magnitude(v - T::one()) <= 32.0 * T::EPS),
                "{x:?}"
            );
        }
    }
    // (b): after the first column the second holds 1e-13 and 0.225 below the first row, so
    // partial pivoting takes 0.225, where the draft compared the untouched 0.35 + 1e-13 with it.
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, T::real(1.0)),
            (0, 1, T::real(0.7)),
            (0, 2, T::real(0.3)),
            (1, 0, T::real(0.5)),
            (1, 1, T::real(0.35 + 1e-13)),
            (1, 2, T::real(0.9)),
            (2, 0, T::real(0.25)),
            (2, 1, T::real(0.4)),
            (2, 2, T::real(0.8)),
        ],
    );
    let lu = SparseLu::with_ordering(&a, &Ordering::Natural);
    assert_eq!(lu.permutation(), [0, 2, 1]);
    assert!(largest_multiplier(&lu) <= 1.0);
    let x_true = [T::real(1.0), T::real(2.0), T::real(3.0)];
    let x = lu.solve_vec(&a.mul_vec(&x_true)).unwrap();
    // The condition number in the 1-norm is about 16.5.
    let error = x
        .iter()
        .zip(&x_true)
        .fold(0.0f64, |most, (&p, &q)| most.max(magnitude(p - q)));
    assert!(error <= 64.0 * 16.5 * T::EPS, "an error of {error}");
}

#[test]
fn the_pivot_is_chosen_among_elements_already_eliminated() {
    each_type!(eliminated_pivot);
}

fn singular_answers<T: Num>() {
    // p_direct-3: a singular matrix is refused by the solves, never with a panic, and its
    // decomposition still describes an n × n factorisation.
    let nan = T::real(f64::NAN);
    let cases: [(&str, SparseMatrix<T>); 4] = [
        (
            "the matrix of ones",
            SparseMatrix::from_coo(
                2,
                2,
                &[
                    (0, 0, T::one()),
                    (0, 1, T::one()),
                    (1, 0, T::one()),
                    (1, 1, T::one()),
                ],
            ),
        ),
        ("the 2 × 2 zero", SparseMatrix::zeros(2, 2)),
        (
            "a 3 × 3 matrix with an empty column",
            SparseMatrix::from_coo(
                3,
                3,
                &[(0, 0, T::one()), (1, 0, T::one()), (2, 2, T::one())],
            ),
        ),
        (
            "a matrix holding a NaN",
            SparseMatrix::from_coo(2, 2, &[(0, 0, nan), (1, 1, T::one())]),
        ),
    ];
    for (what, a) in cases {
        let n = a.rows();
        let lu = SparseLu::new(&a);
        assert!(lu.is_singular(), "{what}");
        let b = Matrix::from_fn(n, 1, |i, _| T::real(1.0 + i as f64));
        assert!(lu.solve(&b).is_none(), "{what}");
        assert!(lu.solve_vec(b.column(0)).is_none(), "{what}");
        assert_eq!(lu.permutation().len(), n, "{what}");
        assert_eq!(lu.column_permutation().len(), n, "{what}");
        for factor in [lu.l(), lu.u(), lu.p(), lu.q()] {
            assert_eq!((factor.rows(), factor.cols()), (n, n), "{what}");
        }
        assert_eq!(lu.p().nnz(), n);
        if what != "a matrix holding a NaN" {
            assert!(lu.det() == T::zero(), "{what}");
            assert!(factor_error(&lu, &a) < 4.0, "{what}");
        }
    }
}

#[test]
fn a_singular_decomposition_refuses_to_solve_and_keeps_its_shape() {
    each_type!(singular_answers);
}
