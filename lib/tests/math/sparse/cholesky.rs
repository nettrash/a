//! The supernodal sparse Cholesky factorisation, checked by what defines it — `P·A·Pᵀ = L·Lᴴ`
//! and the residuals of its solutions, under every ordering — against the dense factorisation,
//! against closed forms on grid Laplacians, and against NumPy on embedded matrices.

use super::common::{
    Num, Rng, assert_close, assert_csr, each_type, magnitude, norm, positive_definite, size,
};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::sparse::{Ordering, SparseCholesky, SparseMatrix};
use std::collections::BTreeSet;

/// The dense Hermitian matrix whose lower triangle is that of `a`: what the factorisations take
/// `a` to be, whatever it holds above its diagonal.
pub(super) fn hermitian<T: Num>(a: &SparseMatrix<T>) -> Matrix<T> {
    let n = a.rows();
    let mut h = Matrix::zeros(n, n);
    for i in 0..n {
        for (j, x) in a.row(i) {
            if j < i {
                h[(i, j)] = x;
                h[(j, i)] = x.conj();
            } else if j == i {
                h[(i, i)] = T::from_real(x.re());
            }
        }
    }
    h
}

/// `P·H·Pᵀ` for the permutation `p[new] = old`.
pub(super) fn permuted<T: Num>(h: &Matrix<T>, p: &[usize]) -> Matrix<T> {
    Matrix::from_fn(p.len(), p.len(), |i, j| h[(p[i], p[j])])
}

/// A random permutation of `0..n`.
pub(super) fn random_permutation(r: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, r.below(i + 1));
    }
    p
}

/// Panics unless `p` is a permutation of `0..n`.
pub(super) fn assert_permutation(p: &[usize], n: usize) {
    assert_eq!(p.len(), n, "the permutation is of the wrong length");
    let mut seen = vec![false; n];
    for &i in p {
        assert!(i < n && !seen[i], "{p:?} is not a permutation");
        seen[i] = true;
    }
}

/// The orderings a factorisation is tested under: each kind, and two random permutations.
pub(super) fn orderings(r: &mut Rng, n: usize) -> Vec<Ordering> {
    vec![
        Ordering::Natural,
        Ordering::ReverseCuthillMckee,
        Ordering::Amd,
        Ordering::Colamd,
        Ordering::NestedDissection,
        Ordering::Given(random_permutation(r, n)),
        Ordering::Given(random_permutation(r, n)),
    ]
}

/// A Hermitian positive definite matrix with a random pattern of about `pairs` pairs of
/// off-diagonal elements, random in both parts for a complex type, and a real diagonal that
/// dominates its row. Both triangles are stored.
pub(super) fn hermitian_positive_definite<T: Num>(
    r: &mut Rng,
    n: usize,
    pairs: usize,
) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    let mut sums = vec![0.0f64; n];
    if n > 1 {
        for _ in 0..pairs {
            let (i, j) = (r.below(n), r.below(n));
            if i == j {
                continue;
            }
            let x = T::random(r);
            entries.push((i, j, x));
            entries.push((j, i, x.conj()));
            sums[i] += magnitude(x);
            sums[j] += magnitude(x);
        }
    }
    for (i, &sum) in sums.iter().enumerate() {
        entries.push((i, i, T::real(sum + 0.5 + r.unit().abs())));
    }
    SparseMatrix::from_coo(n, n, &entries)
}

/// The Laplacian of a grid of `dims[0] × dims[1] × dims[2]` points with zero boundary values:
/// twice the number of dimensions on the diagonal — those of more than one point — and −1
/// between neighbours. A dimension of one point makes it the Laplacian of a grid of fewer
/// dimensions.
pub(super) fn laplacian<T: Num>(dims: [usize; 3]) -> SparseMatrix<T> {
    let index = |x: usize, y: usize, z: usize| x + dims[0] * (y + dims[1] * z);
    let diagonal = 2.0 * dims.iter().filter(|&&d| d > 1).count() as f64;
    let mut entries = Vec::new();
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let i = index(x, y, z);
                entries.push((i, i, T::real(diagonal)));
                let mut link = |j: usize| {
                    entries.push((i, j, T::real(-1.0)));
                    entries.push((j, i, T::real(-1.0)));
                };
                if x + 1 < dims[0] {
                    link(index(x + 1, y, z));
                }
                if y + 1 < dims[1] {
                    link(index(x, y + 1, z));
                }
                if z + 1 < dims[2] {
                    link(index(x, y, z + 1));
                }
            }
        }
    }
    let n = dims[0] * dims[1] * dims[2];
    SparseMatrix::from_coo(n, n, &entries)
}

/// The nested dissection ordering of the grid of [`laplacian`], found from its geometry: a box is
/// cut across its longest side by a plane of points, the two halves are ordered the same way,
/// and the plane comes after both. A box of at most two points a side is taken as it is.
pub(super) fn dissection(dims: [usize; 3]) -> Vec<usize> {
    fn points(lo: [usize; 3], hi: [usize; 3], dims: [usize; 3], out: &mut Vec<usize>) {
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    out.push(x + dims[0] * (y + dims[1] * z));
                }
            }
        }
    }
    fn dissect(lo: [usize; 3], hi: [usize; 3], dims: [usize; 3], out: &mut Vec<usize>) {
        let axis = (0..3).max_by_key(|&d| hi[d] - lo[d]).unwrap();
        let width = hi[axis] - lo[axis];
        if width <= 2 {
            points(lo, hi, dims, out);
            return;
        }
        let middle = lo[axis] + width / 2;
        let (mut left, mut right, mut plane_lo, mut plane_hi) = (hi, lo, lo, hi);
        left[axis] = middle;
        right[axis] = middle + 1;
        plane_lo[axis] = middle;
        plane_hi[axis] = middle + 1;
        dissect(lo, left, dims, out);
        dissect(right, hi, dims, out);
        points(plane_lo, plane_hi, dims, out);
    }
    let mut order = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
    dissect([0; 3], dims, dims, &mut order);
    order
}

/// The number of nonzeros of the Cholesky factor of `P·H·Pᵀ`, its diagonal counted, by the
/// elimination game on sets: eliminating a column joins every two of the rows it has below its
/// diagonal. Slow, and sure.
pub(super) fn fill<T: Num>(h: &Matrix<T>, p: &[usize]) -> usize {
    let n = p.len();
    let mut columns: Vec<BTreeSet<usize>> = (0..n)
        .map(|j| (j + 1..n).filter(|&i| !h[(p[i], p[j])].is_zero()).collect())
        .collect();
    let mut count = 0;
    for j in 0..n {
        let rows: Vec<usize> = columns[j].iter().copied().collect();
        count += rows.len() + 1;
        for (at, &k) in rows.iter().enumerate() {
            columns[k].extend(&rows[at + 1..]);
        }
    }
    count
}

/// `L·D·Lᴴ` of the sparse lower triangular `l` (`D` the identity where `d` is `None`), as a
/// dense matrix, with `|L|·|D|·|Lᴴ|` beside it, the size the backward error is measured against.
/// Each column of `l` adds its outer product, so the work is that of the factorisation and not
/// `n³`, and the sums are formed in another order than the factorisation formed them.
pub(super) fn products<T: Num>(
    l: &SparseMatrix<T>,
    d: Option<&[T::Real]>,
) -> (Matrix<T>, Matrix<f64>) {
    let n = l.rows();
    let columns = l.transpose();
    let mut product = Matrix::<T>::zeros(n, n);
    let mut bound = Matrix::<f64>::zeros(n, n);
    for j in 0..n {
        let column: Vec<(usize, T)> = columns.row(j).collect();
        let pivot = d.map_or(T::Real::ONE, |d| d[j]);
        for &(i, x) in &column {
            let scaled = x * T::from_real(pivot);
            for &(k, y) in &column {
                product[(i, k)] += scaled * y.conj();
                bound[(i, k)] += magnitude(x) * pivot.to_f64().abs() * magnitude(y);
            }
        }
    }
    (product, bound)
}

/// Checks what defines the factorisation `c` of the matrix whose lower triangle is that of `a`:
/// a permutation, a lower triangular `L` with a positive real diagonal and no more nonzeros than
/// counted, and `P·A·Pᵀ = L·Lᴴ` to `factor` roundings.
pub(super) fn assert_factorisation<T: Num>(
    a: &SparseMatrix<T>,
    c: &SparseCholesky<T>,
    factor: f64,
) {
    let n = a.rows();
    let p = c.permutation();
    assert_permutation(p, n);
    let l = c.l();
    assert_csr(&l);
    assert_eq!((l.rows(), l.cols()), (n, n));
    assert!(
        l.nnz() <= c.nnz(),
        "the factor has more nonzeros than counted"
    );
    for i in 0..n {
        let diagonal = l[(i, i)];
        assert!(
            diagonal.re() > T::Real::ZERO && diagonal.im() == T::Real::ZERO,
            "the diagonal of the factor isn't positive"
        );
        assert!(
            l.row(i).all(|(j, _)| j <= i),
            "the factor has an element above its diagonal"
        );
    }
    let (product, _) = products(&l, None);
    assert_close(&product, &permuted(&hermitian(a), p), factor);
}

/// Checks that `x` solves `H·x = b` to `factor` roundings, in the sense of the backward error.
pub(super) fn assert_solves<T: Num>(h: &Matrix<T>, x: &Matrix<T>, b: &Matrix<T>, factor: f64) {
    assert_eq!(x.shape(), b.shape());
    let residual = norm(&(&(h * x) - b));
    let scale = norm(h) * norm(x) + norm(b);
    let tolerance = factor * T::EPS * (h.rows() + 1) as f64 * scale;
    assert!(
        residual <= tolerance,
        "the residual {residual} is more than the tolerated {tolerance}"
    );
}

/// A random dense matrix of the shape.
pub(super) fn random_dense<T: Num>(r: &mut Rng, rows: usize, cols: usize) -> Matrix<T> {
    Matrix::from_fn(rows, cols, |_, _| T::random(r))
}

/// A Hermitian positive definite matrix with no zero at all: random below the diagonal, and a
/// real diagonal that dominates.
pub(super) fn full_positive_definite<T: Num>(r: &mut Rng, n: usize) -> SparseMatrix<T> {
    let below = random_dense::<T>(r, n, n);
    SparseMatrix::from_fn(n, n, |i, j| match i.cmp(&j) {
        core::cmp::Ordering::Greater => below[(i, j)],
        core::cmp::Ordering::Less => below[(j, i)].conj(),
        core::cmp::Ordering::Equal => T::real(1.5 * n as f64 + 1.0),
    })
}

/// The sizes of the random matrices, and how many pairs of elements each has off its diagonal:
/// from nothing off the diagonal to nearly full.
fn shapes() -> Vec<(usize, usize)> {
    if cfg!(miri) {
        vec![(0, 0), (1, 0), (2, 1), (7, 9), (12, 20)]
    } else {
        vec![
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (5, 0),
            (8, 6),
            (17, 30),
            (33, 40),
            (40, 400),
            (73, 110),
            (120, 150),
        ]
    }
}

fn factors_and_solves_of<T: Num>() {
    let mut r = Rng(0x5EED_2201);
    for (n, pairs) in shapes() {
        let a = hermitian_positive_definite::<T>(&mut r, n, pairs);
        let h = hermitian(&a);
        let b = random_dense::<T>(&mut r, n, 3);
        for ordering in orderings(&mut r, n) {
            let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
            assert_factorisation(&a, &c, 16.0);
            match &ordering {
                Ordering::Natural => assert!(c.permutation().iter().copied().eq(0..n)),
                Ordering::Given(p) => assert_eq!(c.permutation(), p),
                _ => {}
            }
            // The count of the nonzeros is that of the pattern, and nothing cancels here.
            assert_eq!(c.nnz(), fill(&h, c.permutation()), "{n} {ordering:?}");
            assert_eq!(c.l().nnz(), c.nnz(), "{n} {ordering:?}");
            assert!(c.supernodes() <= n && (n == 0 || c.supernodes() >= 1));
            let x = c.solve(&b);
            assert_solves(&h, &x, &b, 16.0);
            // One right-hand side at a time gives the same solutions.
            for k in 0..b.cols() {
                let one = Matrix::from_column_major(n, 1, c.solve_vec(b.column(k)));
                let of_many = Matrix::from_column_major(n, 1, x.column(k).to_vec());
                assert_close(&one, &of_many, 16.0);
            }
        }
    }
}

#[test]
fn the_factor_reproduces_the_permuted_matrix_and_solves_under_every_ordering() {
    each_type!(factors_and_solves_of);
}

fn the_default_ordering_of<T: Num>() {
    let mut r = Rng(0x5EED_2202);
    let n = size(60, 9);
    let a = hermitian_positive_definite::<T>(&mut r, n, 2 * n);
    let c = SparseCholesky::new(&a).unwrap();
    assert_factorisation(&a, &c, 16.0);
    let with_default = SparseCholesky::with_ordering(&a, &Ordering::default()).unwrap();
    assert_eq!(c.permutation(), with_default.permutation());
    assert_eq!(c.l(), with_default.l());
}

#[test]
fn new_takes_the_default_ordering() {
    each_type!(the_default_ordering_of);
}

fn dense_agreement_of<T: Num>() {
    let mut r = Rng(0x5EED_2203);
    for (n, pairs) in shapes() {
        // The matrices of the shared helper as well: real elements in every type, and a pattern
        // that repeats cells.
        for a in [
            hermitian_positive_definite::<T>(&mut r, n, pairs),
            positive_definite::<T>(&mut r, n.max(1), pairs),
        ] {
            let n = a.rows();
            let h = hermitian(&a);
            for ordering in orderings(&mut r, n) {
                let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
                let dense = permuted(&h, c.permutation()).cholesky().unwrap();
                assert_close(&c.l().to_matrix(), &dense.l(), 64.0);
                // The determinant, where it is in range, and its logarithm, which always is.
                let (det, reference) = (c.det().to_f64(), dense.det().to_f64());
                let tolerance = 64.0 * T::EPS * (n + 1) as f64;
                if reference.is_finite() {
                    assert!((det - reference).abs() <= tolerance * reference, "{n}");
                } else {
                    assert_eq!(det, reference);
                }
                let log: f64 = dense
                    .l()
                    .diagonal()
                    .iter()
                    .map(|&l| 2.0 * magnitude(l).ln())
                    .sum();
                assert!(
                    (c.log_det().to_f64() - log).abs() <= tolerance * (1.0 + log.abs()),
                    "{n}"
                );
                // The solutions against those of the dense factorisation of the matrix itself.
                let b = random_dense::<T>(&mut r, n, 2);
                assert_close(&c.solve(&b), &h.cholesky().unwrap().solve(&b), 256.0);
            }
        }
    }
}

#[test]
fn the_factor_the_determinant_and_the_solutions_agree_with_the_dense_cholesky() {
    each_type!(dense_agreement_of);
}

fn hermitian_example_of<R: Float>()
where
    Complex<R>: Num<Real = R>,
{
    // [[2, 1 − i], [1 + i, 3]] = L·Lᴴ with L = [[√2, 0], [(1 + i)/√2, √2]]: the element below
    // the diagonal is the one given, not its conjugate.
    let c = |re: f64, im: f64| Complex::new(R::from_f64(re), R::from_f64(im));
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(2.0, 0.0)),
            (1, 0, c(1.0, 1.0)),
            (1, 1, c(3.0, 0.0)),
        ],
    );
    let f = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    let l = f.l();
    let root = 2.0f64.sqrt();
    let tolerance = 4.0 * R::EPSILON.to_f64();
    let near = |x: Complex<R>, re: f64, im: f64| {
        (x.re.to_f64() - re).abs() <= tolerance && (x.im.to_f64() - im).abs() <= tolerance
    };
    assert!(near(l[(0, 0)], root, 0.0));
    assert!(near(l[(1, 0)], 1.0 / root, 1.0 / root));
    assert!(near(l[(1, 1)], root, 0.0));
    assert_eq!(l.nnz(), 3);
    // A·x = b for x = (1 + 2i, 3 − i): b = (2(1 + 2i) + (1 − i)(3 − i), (1 + i)(1 + 2i) + 3(3 − i)).
    let x = f.solve_vec(&[c(4.0, 0.0), c(8.0, 0.0)]);
    assert!(near(x[0], 1.0, 2.0) && near(x[1], 3.0, -1.0), "{x:?}");
    assert!((f.det().to_f64() - 4.0).abs() <= 8.0 * tolerance);
    // Stored with both triangles, or with a wrong upper one, it is the same matrix.
    let both = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(2.0, 0.0)),
            (0, 1, c(1.0, -1.0)),
            (1, 0, c(1.0, 1.0)),
            (1, 1, c(3.0, 0.0)),
        ],
    );
    let wrong = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(2.0, 0.0)),
            (0, 1, c(-7.0, 5.0)),
            (1, 0, c(1.0, 1.0)),
            (1, 1, c(3.0, 0.0)),
        ],
    );
    for other in [both, wrong] {
        let g = SparseCholesky::with_ordering(&other, &Ordering::Natural).unwrap();
        assert_eq!(g.l(), l);
    }
}

#[test]
fn a_complex_hermitian_matrix_keeps_its_conjugations() {
    hermitian_example_of::<f32>();
    hermitian_example_of::<f64>();
}

fn upper_triangle_of<T: Num>() {
    let mut r = Rng(0x5EED_2204);
    let n = size(40, 8);
    let full = hermitian_positive_definite::<T>(&mut r, n, 2 * n);
    let lower = full.map(|i, j, x| if j <= i { x } else { T::real(0.0) });
    // The same lower triangle under an upper one of anything at all: other elements, in other
    // cells, some of them not numbers.
    let mut entries: Vec<(usize, usize, T)> = Vec::new();
    for i in 0..n {
        entries.extend(lower.row(i).map(|(j, x)| (i, j, x)));
    }
    for k in 0..3 * n {
        let (i, j) = (r.below(n), r.below(n));
        if i < j {
            let x = match k % 5 {
                0 => T::real(f64::NAN),
                1 => T::real(f64::INFINITY),
                _ => T::random(&mut r) * T::real(1e3),
            };
            entries.push((i, j, x));
        }
    }
    let garbage = SparseMatrix::from_coo(n, n, &entries);
    assert!(garbage.nnz() > lower.nnz());
    let b = random_dense::<T>(&mut r, n, 2);
    for ordering in orderings(&mut r, n) {
        let reference = SparseCholesky::with_ordering(&full, &ordering).unwrap();
        for other in [&lower, &garbage] {
            let c = SparseCholesky::with_ordering(other, &ordering).unwrap();
            assert_eq!(c.permutation(), reference.permutation(), "{ordering:?}");
            assert_eq!(c.l(), reference.l(), "{ordering:?}");
            assert_eq!(c.solve(&b), reference.solve(&b), "{ordering:?}");
            assert_eq!(c.nnz(), reference.nnz());
            assert_eq!(c.supernodes(), reference.supernodes());
        }
    }
}

#[test]
fn what_stands_above_the_diagonal_is_not_read() {
    each_type!(upper_triangle_of);
}

fn imaginary_diagonal_of<R: Float>()
where
    Complex<R>: Num<Real = R>,
{
    // A Hermitian matrix has a real diagonal; of one that hasn't, the real part is taken.
    let c = |re: f64, im: f64| Complex::new(R::from_f64(re), R::from_f64(im));
    let clean = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(4.0, 0.0)),
            (1, 0, c(1.0, 1.0)),
            (1, 1, c(3.0, 0.0)),
        ],
    );
    let stray = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(4.0, 9.0)),
            (1, 0, c(1.0, 1.0)),
            (1, 1, c(3.0, -2.0)),
        ],
    );
    let (f, g) = (
        SparseCholesky::new(&clean).unwrap(),
        SparseCholesky::new(&stray).unwrap(),
    );
    assert_eq!(f.l(), g.l());
    // The matrix of audit p_direct-7, which the earlier implementation refused for an imaginary
    // part of rounding size on a pivot while the dense Cholesky, as LAPACK, accepted it: here
    // the two agree, with the part on either pivot, of either sign, down to the least
    // subnormal, and as large as the real part.
    let least = (R::MIN_POSITIVE * R::EPSILON).to_f64();
    for (first, last) in [
        (1e-17, 0.0),
        (-1e-17, 0.0),
        (least, 0.0),
        (0.0, 1e-17),
        (0.0, -least),
        (1.0, 0.0),
        (2.0, -3.0),
    ] {
        let a = SparseMatrix::from_coo(
            2,
            2,
            &[
                (0, 0, c(2.0, first)),
                (1, 0, c(0.5, 0.25)),
                (0, 1, c(0.5, -0.25)),
                (1, 1, c(3.0, last)),
            ],
        );
        let dense = a
            .to_matrix()
            .cholesky()
            .expect("the dense Cholesky reads the real part");
        for ordering in [Ordering::Natural, Ordering::Given(vec![1, 0])] {
            let sparse = SparseCholesky::with_ordering(&a, &ordering).expect("as does this one");
            assert_eq!(sparse.l().nnz(), 3);
            if ordering == Ordering::Natural {
                assert_close(&sparse.l().to_matrix(), &dense.l(), 4.0);
                assert!(
                    (sparse.det() - dense.det()).abs()
                        <= R::from_f64(8.0) * R::EPSILON * dense.det()
                );
            }
        }
    }
    let one = SparseMatrix::from_coo(1, 1, &[(0, 0, c(4.0, 1e-17))]);
    assert!(one.to_matrix().cholesky().is_some());
    assert_eq!(SparseCholesky::new(&one).unwrap().l()[(0, 0)], c(2.0, 0.0));
}

#[test]
fn the_imaginary_part_of_a_diagonal_element_is_left_out() {
    imaginary_diagonal_of::<f32>();
    imaginary_diagonal_of::<f64>();
}

fn solutions_of_audit_matrices_of<T: Num>() {
    let t = T::real;
    // Audit p_direct-4: the back substitution of the earlier implementation subtracted the
    // undivided value, so its solutions were right only where the factor is diagonal. For
    // [[4, 2], [2, 5]]·x = (4, 7) it gave x = (−0.25, 1.25) where x = (0.375, 1.25).
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, t(4.0)),
            (0, 1, t(2.0)),
            (1, 0, t(2.0)),
            (1, 1, t(5.0)),
        ],
    );
    let b = Matrix::from_rows(&[[t(4.0)], [t(7.0)]]);
    let x = Matrix::from_rows(&[[t(0.375)], [t(1.25)]]);
    for ordering in [
        Ordering::Natural,
        Ordering::Given(vec![1, 0]),
        Ordering::Amd,
    ] {
        let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
        assert_close(&c.solve(&b), &x, 8.0);
        assert_close(
            &Matrix::from_column_major(2, 1, c.solve_vec(&[t(4.0), t(7.0)])),
            &x,
            8.0,
        );
    }
    // The 1-D Laplacian of twenty points with b_i = i, for which the earlier implementation left
    // a residual of 71: the solution is x_i = i·(21² − i²)/6, the cubic whose second difference
    // is −i and which vanishes at 0 and 21.
    let n = 20;
    let a = SparseMatrix::tridiagonal(
        &vec![t(-1.0); n - 1],
        &vec![t(2.0); n],
        &vec![t(-1.0); n - 1],
    );
    let b: Vec<T> = (1..=n).map(|i| t(i as f64)).collect();
    let exact: Vec<f64> = (1..=n)
        .map(|i| {
            let i = i as f64;
            i * (441.0 - i * i) / 6.0
        })
        .collect();
    let largest = exact.iter().copied().fold(0.0, f64::max);
    let mut r = Rng(0x5EED_2210);
    for ordering in orderings(&mut r, n) {
        let x = SparseCholesky::with_ordering(&a, &ordering)
            .unwrap()
            .solve_vec(&b);
        // The condition number is about 180.
        for (x, &exact) in x.iter().zip(&exact) {
            assert!(
                magnitude(*x - t(exact)) <= 4096.0 * T::EPS * largest,
                "{x:?} for {exact} under {ordering:?}"
            );
        }
    }
}

#[test]
fn the_solutions_are_right_where_the_factor_is_not_diagonal() {
    each_type!(solutions_of_audit_matrices_of);
}

/// The time a factorisation of `a` in its own order and a solve with it took, after checking that
/// the factor has `nnz` nonzeros and that the solution solves the system.
fn timed(a: &SparseMatrix<f64>, nnz: usize) -> f64 {
    let n = a.rows();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
    let start = std::time::Instant::now();
    let c = SparseCholesky::with_ordering(a, &Ordering::Natural).unwrap();
    let x = c.solve_vec(&b);
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(c.nnz(), nnz);
    // The backward error in the infinity norm, which for a sum of `n` terms grows like `√n`.
    let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let residual: Vec<f64> = a.mul_vec(&x).iter().zip(&b).map(|(y, b)| y - b).collect();
    let row_sums: Vec<f64> = (0..n)
        .map(|i| a.row(i).map(|(_, v)| v.abs()).sum())
        .collect();
    let error = largest(&residual) / (largest(&row_sums) * largest(&x) + largest(&b));
    assert!(
        error <= 64.0 * f64::EPSILON * (n as f64).sqrt(),
        "the backward error is {error}"
    );
    elapsed
}

/// An arrow with its tip last: `n − 1` columns with one element under the diagonal each, all in
/// the last row, which is the root of a tree of `n − 1` leaves. Both triangles are stored, for
/// the products that check the solutions.
fn arrow(n: usize) -> SparseMatrix<f64> {
    let mut entries: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i, 2.0)).collect();
    entries.push((n - 1, n - 1, n as f64));
    entries.extend((0..n - 1).flat_map(|j| [(n - 1, j, -1.0), (j, n - 1, -1.0)]));
    SparseMatrix::from_coo(n, n, &entries)
}

#[test]
fn the_cost_grows_with_the_order_as_the_nonzeros_do() {
    // Audit p_direct-8: the earlier implementation cleared and scanned a dense vector of length
    // `n` for every column, about `n²` work whatever the nonzeros: a tridiagonal matrix of 40 000
    // rows, whose factor has 80 000 nonzeros, took almost a second. Here a tridiagonal matrix (a
    // path for a tree) and an arrow (a star) of `8·m` rows are factored and solved once, against
    // eight times for those of `m` rows: the same work for a method whose cost is that of the
    // nonzeros, so the two take about as long, where a quadratic method takes eight times as
    // long for the larger. Timing the same amount of work keeps the two exposed alike to a busy
    // machine; the least of three runs is taken, and the bound is four times the linear ratio.
    let small = size(12_500, 40);
    let large = 8 * small;
    let tridiagonal = |n: usize| {
        SparseMatrix::tridiagonal(&vec![-1.0f64; n - 1], &vec![4.0; n], &vec![-1.0; n - 1])
    };
    for (name, build) in [
        (
            "tridiagonal",
            &tridiagonal as &dyn Fn(usize) -> SparseMatrix<f64>,
        ),
        ("arrow", &arrow),
    ] {
        let (a, b) = (build(small), build(large));
        let mut times = [f64::INFINITY; 2];
        for _ in 0..if cfg!(miri) { 1 } else { 3 } {
            times[0] = times[0].min((0..8).map(|_| timed(&a, 2 * small - 1)).sum());
            times[1] = times[1].min(timed(&b, 2 * large - 1));
        }
        if !cfg!(miri) {
            let ratio = times[1] / times[0];
            assert!(
                ratio < 4.0,
                "{name}: 8 systems of {small} rows took {:.4} s and one of {large} rows {:.4} s, \
                 {ratio:.1} times as long",
                times[0],
                times[1]
            );
        }
    }
}

fn indefinite_of<T: Num>() {
    let mut r = Rng(0x5EED_2205);
    // The smallest cases, under any ordering.
    let none = |a: &SparseMatrix<T>| {
        let n = a.rows();
        for ordering in orderings(&mut Rng(7), n) {
            assert!(
                SparseCholesky::with_ordering(a, &ordering).is_none(),
                "{ordering:?}"
            );
        }
    };
    let t = T::real;
    none(&SparseMatrix::from_coo(
        2,
        2,
        &[(0, 0, t(1.0)), (1, 0, t(2.0)), (1, 1, t(1.0))],
    ));
    none(&SparseMatrix::from_diagonal(&[t(1.0), t(-1.0), t(1.0)]));
    none(&SparseMatrix::from_diagonal(&[t(-4.0)]));
    none(&SparseMatrix::zeros(3, 3));
    none(&SparseMatrix::from_coo(
        3,
        3,
        &[(1, 0, t(1.0)), (2, 1, t(1.0))],
    ));
    // A positive definite matrix shifted to either side of its smallest eigenvalue: the pivot
    // that fails can be anywhere, in a block on the diagonal or below one.
    for (n, pairs) in [(size(30, 8), 40), (size(64, 9), 400), (size(90, 10), 120)] {
        let a = hermitian_positive_definite::<T>(&mut r, n, pairs);
        let smallest = hermitian(&a).hermitian_eigenvalues()[0].to_f64();
        let shifted = |shift: f64| a.map(|i, j, x| if i == j { x - T::real(shift) } else { x });
        let (below, above) = (shifted(0.9 * smallest), shifted(1.1 * smallest));
        for ordering in orderings(&mut r, n) {
            let c = SparseCholesky::with_ordering(&below, &ordering).expect("positive definite");
            assert_factorisation(&below, &c, 64.0);
            assert!(
                SparseCholesky::with_ordering(&above, &ordering).is_none(),
                "{n} {ordering:?}"
            );
        }
        // One diagonal element negated: wherever it lies, its pivot is negative at the latest.
        for _ in 0..4 {
            let k = r.below(n);
            let negated = a.map(|i, j, x| {
                if i == k && j == k {
                    T::real(0.0) - x
                } else {
                    x
                }
            });
            for ordering in orderings(&mut r, n) {
                assert!(SparseCholesky::with_ordering(&negated, &ordering).is_none());
            }
        }
    }
}

#[test]
fn a_matrix_that_is_not_positive_definite_has_no_factorisation() {
    each_type!(indefinite_of);
}

fn semidefinite_of<T: Num>() {
    let t = T::real;
    // The Laplacian of a path with free ends is singular: its pivots are 1, …, 1 and then 0,
    // all of them exact in the natural order.
    for n in [2, 3, 10, size(50, 6)] {
        let mut diagonal = vec![t(2.0); n];
        diagonal[0] = t(1.0);
        diagonal[n - 1] = t(1.0);
        let a = SparseMatrix::tridiagonal(&vec![t(-1.0); n - 1], &diagonal, &vec![t(-1.0); n - 1]);
        assert!(
            SparseCholesky::with_ordering(&a, &Ordering::Natural).is_none(),
            "{n}"
        );
        // With one end held it is positive definite, of determinant 1.
        diagonal[n - 1] = t(2.0);
        let held =
            SparseMatrix::tridiagonal(&vec![t(-1.0); n - 1], &diagonal, &vec![t(-1.0); n - 1]);
        let c = SparseCholesky::with_ordering(&held, &Ordering::Natural).unwrap();
        assert_eq!(c.det().to_f64(), 1.0);
    }
    // The matrix of ones has rank 1: the second pivot is exactly zero in whatever order.
    for n in [2, 5, size(40, 6)] {
        let ones = SparseMatrix::from_fn(n, n, |_, _| t(1.0));
        for ordering in orderings(&mut Rng(11), n) {
            assert!(SparseCholesky::with_ordering(&ones, &ordering).is_none());
        }
    }
    // A zero on the diagonal with nothing in its row and column.
    let a = SparseMatrix::from_coo(3, 3, &[(0, 0, t(1.0)), (2, 2, t(1.0))]);
    for ordering in orderings(&mut Rng(12), 3) {
        assert!(SparseCholesky::with_ordering(&a, &ordering).is_none());
    }
}

#[test]
fn a_singular_positive_semidefinite_matrix_has_no_factorisation() {
    each_type!(semidefinite_of);
}

fn not_finite_of<T: Num>() {
    let mut r = Rng(0x5EED_2206);
    let n = size(36, 8);
    let a = hermitian_positive_definite::<T>(&mut r, n, 3 * n);
    let stored: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| a.row(i).map(move |(j, _)| (i, j)))
        .filter(|&(i, j)| j <= i)
        .collect();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for _ in 0..6 {
            // In a cell of the lower triangle, on the diagonal or off it.
            let (bi, bj) = stored[r.below(stored.len())];
            let spoiled = a.map(|i, j, x| if (i, j) == (bi, bj) { T::real(bad) } else { x });
            for ordering in orderings(&mut r, n) {
                assert!(
                    SparseCholesky::with_ordering(&spoiled, &ordering).is_none(),
                    "{bad} at ({bi}, {bj}) under {ordering:?}"
                );
            }
        }
        // And on each diagonal element of a diagonal matrix, where nothing follows the pivot.
        let mut diagonal = vec![T::real(2.0); 5];
        diagonal[3] = T::real(bad);
        assert!(SparseCholesky::new(&SparseMatrix::from_diagonal(&diagonal)).is_none());
        assert!(SparseCholesky::new(&SparseMatrix::from_diagonal(&[T::real(bad)])).is_none());
    }
}

#[test]
fn a_matrix_with_an_element_that_is_not_finite_has_no_factorisation() {
    each_type!(not_finite_of);
}

fn smallest_of<T: Num>() {
    // No rows at all: the empty product for a determinant, and solutions of no rows.
    let empty = SparseMatrix::<T>::zeros(0, 0);
    for ordering in [Ordering::Natural, Ordering::Amd, Ordering::Given(vec![])] {
        let c = SparseCholesky::with_ordering(&empty, &ordering).unwrap();
        assert_eq!(c.det().to_f64(), 1.0);
        assert_eq!(c.log_det().to_f64(), 0.0);
        assert_eq!((c.nnz(), c.supernodes()), (0, 0));
        assert!(c.permutation().is_empty());
        assert_eq!((c.l().rows(), c.l().cols(), c.l().nnz()), (0, 0, 0));
        assert_eq!(c.solve(&Matrix::zeros(0, 3)).shape(), (0, 3));
        assert_eq!(c.solve(&Matrix::zeros(0, 0)).shape(), (0, 0));
        assert!(c.solve_vec(&[]).is_empty());
    }
    // One row: the square root and a division.
    let one = SparseMatrix::from_diagonal(&[T::real(9.0)]);
    let c = SparseCholesky::new(&one).unwrap();
    assert_eq!(c.l().to_matrix(), Matrix::from_rows(&[[T::real(3.0)]]));
    assert_eq!(c.det().to_f64(), 9.0);
    // As good as the platform's logarithm, which need not be correctly rounded: Miri moves it
    // by several units in the last place on purpose (16 epsilons were seen for ln 9).
    assert!((c.log_det().to_f64() - 9.0f64.ln()).abs() <= 64.0 * T::EPS);
    assert_eq!((c.nnz(), c.supernodes()), (1, 1));
    assert_eq!(c.permutation(), [0]);
    assert_eq!(c.solve_vec(&[T::real(18.0)]), [T::real(2.0)]);
    assert_eq!(c.solve(&Matrix::zeros(1, 0)).shape(), (1, 0));
    // And the three ways for it to have no factorisation.
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(SparseCholesky::new(&SparseMatrix::from_diagonal(&[T::real(bad)])).is_none());
    }
    assert!(SparseCholesky::new(&SparseMatrix::<T>::zeros(1, 1)).is_none());
}

#[test]
fn empty_and_one_by_one_matrices() {
    each_type!(smallest_of);
}

fn right_hand_sides_of<T: Num>() {
    let mut r = Rng(0x5EED_2207);
    let n = size(50, 9);
    let a = hermitian_positive_definite::<T>(&mut r, n, 2 * n);
    let h = hermitian(&a);
    let c = SparseCholesky::new(&a).unwrap();
    // No columns, one, and more than the kernels take at a time.
    for columns in [0, 1, 2, 7, size(19, 3)] {
        let b = random_dense::<T>(&mut r, n, columns);
        let x = c.solve(&b);
        assert_eq!(x.shape(), (n, columns));
        assert_solves(&h, &x, &b, 16.0);
    }
    // A zero right-hand side has the zero solution, exactly.
    assert_eq!(c.solve(&Matrix::zeros(n, 2)), Matrix::zeros(n, 2));
    assert_eq!(c.solve_vec(&vec![T::real(0.0); n]), vec![T::real(0.0); n]);
    // The solution of A·x = A·e is e.
    let e = Matrix::from_fn(n, 1, |i, _| T::real(1.0 + (i % 3) as f64));
    assert_close(&c.solve(&(&h * &e)), &e, 64.0);
}

#[test]
fn any_number_of_right_hand_sides_is_solved() {
    each_type!(right_hand_sides_of);
}

/// A matrix with one large supernode and many small ones under it: `blocks` separate tridiagonal
/// pieces of `piece` points, each joined to all of a full border of `border` rows.
fn bordered<T: Num>(r: &mut Rng, blocks: usize, piece: usize, border: usize) -> SparseMatrix<T> {
    let inner = blocks * piece;
    let n = inner + border;
    let mut entries = Vec::new();
    let mut sums = vec![0.0f64; n];
    let mut link = |i: usize, j: usize, x: T, entries: &mut Vec<(usize, usize, T)>| {
        entries.push((i, j, x));
        entries.push((j, i, x.conj()));
        sums[i] += magnitude(x);
        sums[j] += magnitude(x);
    };
    for b in 0..blocks {
        for k in 1..piece {
            link(b * piece + k, b * piece + k - 1, T::random(r), &mut entries);
        }
        for q in 0..border {
            link(
                inner + q,
                b * piece + r.below(piece),
                T::random(r),
                &mut entries,
            );
        }
    }
    for q in 0..border {
        for s in 0..q {
            link(inner + q, inner + s, T::random(r), &mut entries);
        }
    }
    for (i, &sum) in sums.iter().enumerate() {
        entries.push((i, i, T::real(sum + 1.0)));
    }
    SparseMatrix::from_coo(n, n, &entries)
}

fn large_supernodes_of<T: Num>() {
    let mut r = Rng(0x5EED_2208);
    // Blocks beyond the sizes the dense kernels take whole: a full matrix, which is one
    // supernode, and a border under many small supernodes, which all update it.
    let full = size(150, 20);
    let a = full_positive_definite::<T>(&mut r, full);
    let c = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    assert_eq!(a.nnz(), full * full, "the matrix is full");
    assert_eq!((c.supernodes(), c.nnz()), (1, full * (full + 1) / 2));
    assert_factorisation(&a, &c, 16.0);
    let b = random_dense::<T>(&mut r, full, 3);
    assert_solves(&hermitian(&a), &c.solve(&b), &b, 16.0);

    let (blocks, piece, border) = (size(12, 3), size(9, 3), size(70, 6));
    let a = bordered::<T>(&mut r, blocks, piece, border);
    let n = a.rows();
    let h = hermitian(&a);
    let b = random_dense::<T>(&mut r, n, 2);
    for ordering in [
        Ordering::Natural,
        Ordering::Given(random_permutation(&mut r, n)),
    ] {
        let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
        assert_factorisation(&a, &c, 16.0);
        assert_eq!(c.nnz(), fill(&h, c.permutation()));
        assert_solves(&h, &c.solve(&b), &b, 16.0);
        assert_close(
            &c.l().to_matrix(),
            &permuted(&h, c.permutation()).cholesky().unwrap().l(),
            64.0,
        );
    }
}

#[test]
fn supernodes_larger_than_the_dense_kernels_take_whole() {
    each_type!(large_supernodes_of);
}

fn grids_of<T: Num>() {
    let mut r = Rng(0x5EED_2209);
    let grids = if cfg!(miri) {
        vec![[5, 1, 1], [3, 3, 1], [2, 2, 2]]
    } else {
        vec![[40, 1, 1], [13, 11, 1], [24, 24, 1], [6, 5, 4], [8, 8, 8]]
    };
    for dims in grids {
        let a = laplacian::<T>(dims);
        let n = a.rows();
        let h = hermitian(&a);
        let b = random_dense::<T>(&mut r, n, 2);
        let mut nonzeros = Vec::new();
        for ordering in [
            Ordering::Natural,
            Ordering::Given(dissection(dims)),
            Ordering::ReverseCuthillMckee,
            Ordering::Amd,
            Ordering::Given(random_permutation(&mut r, n)),
        ] {
            let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
            assert_factorisation(&a, &c, 16.0);
            assert_solves(&h, &c.solve(&b), &b, 16.0);
            assert_eq!(c.l().nnz(), c.nnz(), "{dims:?} {ordering:?}");
            nonzeros.push(c.nnz());
        }
        // In the natural order the factor fills its envelope, from the first nonzero of each
        // row to the diagonal, as every row but the first has a nonzero before its diagonal
        // (George and Liu): the nonzeros of the factor are the profile of the matrix.
        let envelope: usize = (0..n).map(|i| i - a.row(i).next().unwrap().0 + 1).sum();
        assert_eq!(nonzeros[0], envelope, "{dims:?}");
        assert_eq!(nonzeros[0], fill(&h, &(0..n).collect::<Vec<_>>()));
        assert_eq!(nonzeros[1], fill(&h, &dissection(dims)));
        // Dissection pays once the grid is large enough for the band to be wide.
        if n >= 500 {
            assert!(nonzeros[1] < nonzeros[0], "{dims:?}: {nonzeros:?}");
        }
    }
}

#[test]
fn grid_laplacians_in_one_two_and_three_dimensions() {
    each_type!(grids_of);
}

fn log_determinants_of<T: Num>() {
    // The eigenvalues of the Laplacian of a grid are sums of those of its dimensions,
    // `2 − 2·cos(k·π/(m + 1))` for `k = 1, …, m`, so the logarithm of its determinant is a sum
    // of logarithms known in closed form.
    let grids = if cfg!(miri) {
        vec![[4, 1, 1], [3, 2, 1]]
    } else {
        vec![[30, 1, 1], [17, 12, 1], [30, 30, 1], [9, 8, 7]]
    };
    for dims in grids {
        let line = |m: usize| -> Vec<f64> {
            if m == 1 {
                return vec![0.0];
            }
            (1..=m)
                .map(|k| 2.0 - 2.0 * (k as f64 * core::f64::consts::PI / (m + 1) as f64).cos())
                .collect()
        };
        let (xs, ys, zs) = (line(dims[0]), line(dims[1]), line(dims[2]));
        let mut expected = 0.0;
        for x in &xs {
            for y in &ys {
                for z in &zs {
                    expected += (x + y + z).ln();
                }
            }
        }
        let a = laplacian::<T>(dims);
        let n = a.rows();
        for ordering in [Ordering::Natural, Ordering::Given(dissection(dims))] {
            let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
            let log_det = c.log_det().to_f64();
            // Each of the `n` logarithms of the diagonal is good to a few roundings of a
            // number near 1, and so is each pivot, relatively.
            let tolerance = 64.0 * T::EPS * n as f64;
            assert!(
                (log_det - expected).abs() <= tolerance * (1.0 + expected.abs()),
                "{dims:?}: {log_det} for {expected}"
            );
            // The determinant itself is in range for the small grids only.
            if expected < 80.0 {
                let det = c.det().to_f64();
                assert!((det - expected.exp()).abs() <= tolerance * expected.exp() * n as f64);
            }
        }
    }
    // A determinant far out of range, whose logarithm is not.
    let n = size(600, 30);
    let big = SparseMatrix::from_diagonal(&vec![T::real(1e6); n]);
    let c = SparseCholesky::new(&big).unwrap();
    let expected = n as f64 * 1e6f64.ln();
    assert!((c.log_det().to_f64() - expected).abs() <= 64.0 * T::EPS * expected);
    if !cfg!(miri) {
        assert!(c.det().to_f64().is_infinite());
    }
    let tiny = SparseMatrix::from_diagonal(&vec![T::real(1e-6); n]);
    let c = SparseCholesky::new(&tiny).unwrap();
    assert!((c.log_det().to_f64() + expected).abs() <= 64.0 * T::EPS * expected);
    if !cfg!(miri) {
        assert_eq!(c.det().to_f64(), 0.0);
    }
    // A long diagonal of random elements, whose logarithm of the determinant is known to the
    // last digits: the logarithms of its elements, as the type holds them, summed in `f64`
    // with compensation. Each term of the factorisation's sum is good to a rounding or two,
    // and the sum may lose no more than two roundings (epsilons) of the total. A plain sum of
    // these 20 000 terms in `f32` loses about six, three times the tolerance; the compensated
    // one a third of one.
    let n = size(20_000, 50);
    let mut r = Rng(0x5EED_2211);
    let diagonal: Vec<T> = (0..n).map(|_| T::real(2.5 + 1.5 * r.unit())).collect();
    let c = SparseCholesky::new(&SparseMatrix::from_diagonal(&diagonal)).unwrap();
    let (mut sum, mut carry) = (0.0f64, 0.0f64);
    for d in &diagonal {
        let term = d.re().to_f64().ln();
        let next = sum + term;
        carry += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    let expected = sum + carry;
    let error = (c.log_det().to_f64() - expected).abs();
    let mut tolerance = 2.0 * T::EPS * expected + 8.0 * T::EPS * (n as f64).sqrt();
    if cfg!(miri) {
        // Miri moves each logarithm by several units in the last place on purpose, as a
        // platform's library is allowed to, and then each term carries that much.
        tolerance += 32.0 * T::EPS * n as f64;
    }
    assert!(error <= tolerance, "{error} for a tolerance of {tolerance}");
}

#[test]
fn the_logarithm_of_the_determinant_of_a_grid_laplacian_is_that_of_its_eigenvalues() {
    each_type!(log_determinants_of);
}

fn supernode_counts_of<T: Num>() {
    let t = T::real;
    let natural =
        |a: &SparseMatrix<T>| SparseCholesky::with_ordering(a, &Ordering::Natural).unwrap();
    // Nothing joins the columns of a diagonal matrix, under any ordering.
    let n = size(40, 20);
    let diagonal = SparseMatrix::from_diagonal(&vec![t(2.0); n]);
    for ordering in orderings(&mut Rng(21), n) {
        let c = SparseCholesky::with_ordering(&diagonal, &ordering).unwrap();
        assert_eq!((c.supernodes(), c.nnz()), (n, n));
    }
    // A full matrix is one supernode, under any ordering.
    let full = SparseMatrix::from_fn(n, n, |i, j| if i == j { t(n as f64) } else { t(1.0) });
    for ordering in orderings(&mut Rng(22), n) {
        let c = SparseCholesky::with_ordering(&full, &ordering).unwrap();
        assert_eq!((c.supernodes(), c.nnz()), (1, n * (n + 1) / 2));
    }
    // What follows pins the thresholds of the amalgamation as the documentation states them.
    //
    // A tridiagonal matrix in its own order: the elimination tree is a path and every column
    // but the last has two nonzeros, so the fundamental supernodes are single columns but for
    // the last two. A merge of two columns adds a zero; it is made up to 4 columns, and then up
    // to 16 while less than 80 % of the block is zeros: 16 columns hold 152 elements for 32
    // nonzeros, 17 would hold 170 for 34. So the columns go in sixteens.
    for (n, expected) in [
        (1, 1),
        (2, 1),
        (5, 1),
        (16, 1),
        (17, 2),
        (64, 4),
        (65, 5),
        (100, 7),
    ] {
        if cfg!(miri) && n > 20 {
            continue;
        }
        let a = SparseMatrix::tridiagonal(
            &vec![t(-1.0); n - 1],
            &vec![t(2.0); n],
            &vec![t(-1.0); n - 1],
        );
        let c = natural(&a);
        assert_eq!(c.supernodes(), expected, "tridiagonal of {n}");
        assert_eq!(c.nnz(), 2 * n - 1);
        assert_factorisation(&a, &c, 16.0);
    }
    // An arrow with its tip last: every column is a child of the last one. The last child joins
    // it without a zero, the next ones while the block of the columns taken in stays under 80 %
    // zeros up to 16 columns, and the rest stay as they are: n − 15 supernodes.
    for n in [3, 16, 17, 40] {
        let mut entries: Vec<(usize, usize, T)> = (0..n).map(|i| (i, i, t(n as f64))).collect();
        entries.extend((0..n - 1).map(|j| (n - 1, j, t(1.0))));
        let a = SparseMatrix::from_coo(n, n, &entries);
        let c = natural(&a);
        assert_eq!(
            c.supernodes(),
            if n <= 16 { 1 } else { n - 15 },
            "arrow of {n}"
        );
        assert_eq!(c.nnz(), 2 * n - 1);
        assert_factorisation(&a, &c, 16.0);
    }
    // Two full blocks with nothing between them are two supernodes: a supernode is merged into
    // its parent only, and neither is the other's.
    let k = size(12, 4);
    let blocks = SparseMatrix::from_fn(2 * k, 2 * k, |i, j| {
        if i == j {
            t(k as f64)
        } else if (i < k) == (j < k) {
            t(1.0)
        } else {
            t(0.0)
        }
    });
    let c = natural(&blocks);
    assert_eq!((c.supernodes(), c.nnz()), (2, k * (k + 1)));
    assert_factorisation(&blocks, &c, 16.0);
}

#[test]
fn the_supernodes_of_known_patterns() {
    each_type!(supernode_counts_of);
}

#[test]
fn the_factorisation_can_be_cloned_and_written() {
    let a = SparseMatrix::tridiagonal(&[-1.0; 5], &[2.0; 6], &[-1.0; 5]);
    let c = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    let d = c.clone();
    assert_eq!(c.l(), d.l());
    assert_eq!(c.solve_vec(&[1.0; 6]), d.solve_vec(&[1.0; 6]));
    assert_eq!(
        format!("{c:?}"),
        "SparseCholesky { order: 6, nnz: 11, supernodes: 1 }"
    );
}

#[test]
#[should_panic(expected = "the matrix isn't square")]
fn a_matrix_that_is_not_square_panics() {
    let _ = SparseCholesky::new(&SparseMatrix::<f64>::zeros(2, 3));
}

#[test]
#[should_panic(expected = "the right-hand side has not as many rows")]
fn a_right_hand_side_of_another_height_panics() {
    let c = SparseCholesky::new(&SparseMatrix::<f64>::identity(3)).unwrap();
    let _ = c.solve(&Matrix::zeros(2, 1));
}

#[test]
#[should_panic(expected = "the right-hand side has not as many elements")]
fn a_right_hand_side_vector_of_another_length_panics() {
    let c = SparseCholesky::new(&SparseMatrix::<f64>::identity(3)).unwrap();
    let _ = c.solve_vec(&[1.0; 4]);
}

#[test]
#[should_panic(expected = "the permutation repeats an index")]
fn a_given_ordering_that_is_not_a_permutation_panics() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = SparseCholesky::with_ordering(&a, &Ordering::Given(vec![0, 2, 2]));
}

/// The matrices and the solutions of `data/gen_cholesky.py`, computed with NumPy and SciPy.
mod reference {
    include!("data/cholesky_reference.rs");
}

/// The matrix of a complex reference case, from its lower triangle.
fn reference_matrix(n: usize, lower: &[(usize, usize, f64, f64)]) -> SparseMatrix<Complex<f64>> {
    let entries: Vec<(usize, usize, Complex<f64>)> = lower
        .iter()
        .map(|&(i, j, re, im)| (i, j, Complex::new(re, im)))
        .collect();
    SparseMatrix::from_coo(n, n, &entries)
}

#[test]
fn a_real_matrix_agrees_with_numpy() {
    use reference::REAL as case;
    let n = case.n;
    let entries: Vec<(usize, usize, f64)> =
        case.lower.iter().map(|&(i, j, re, _)| (i, j, re)).collect();
    let a = SparseMatrix::from_coo(n, n, &entries);
    let natural = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    // The fill of the natural order, as the factor of NumPy has it.
    assert_eq!(natural.nnz(), case.nnz_natural);
    assert_eq!(natural.l().nnz(), case.nnz_natural);
    let given =
        SparseCholesky::with_ordering(&a, &Ordering::Given(case.permutation.to_vec())).unwrap();
    assert_eq!(given.nnz(), case.nnz_permuted);
    for c in [natural, given, SparseCholesky::new(&a).unwrap()] {
        let x = c.solve_vec(case.b_re);
        for (x, reference) in x.iter().zip(case.x_re) {
            assert!(
                (x - reference).abs() <= 1e-13 * reference.abs().max(1.0),
                "{x} for {reference}"
            );
        }
        assert!((c.log_det() - case.log_det).abs() <= 1e-12 * case.log_det.abs());
        assert!((c.det() - case.det).abs() <= 1e-12 * case.det.abs());
    }
    // The factor of the natural order, element by element.
    let l = SparseCholesky::with_ordering(&a, &Ordering::Natural)
        .unwrap()
        .l();
    assert_eq!(l.nnz(), case.l_natural.len());
    for &(i, j, re, _) in case.l_natural {
        assert!(
            (l[(i, j)] - re).abs() <= 1e-13 * re.abs().max(1.0),
            "L[{i}, {j}]"
        );
    }
    // In single precision, to its own rounding.
    let single: Vec<(usize, usize, f32)> =
        entries.iter().map(|&(i, j, x)| (i, j, x as f32)).collect();
    let c = SparseCholesky::new(&SparseMatrix::from_coo(n, n, &single)).unwrap();
    let b: Vec<f32> = case.b_re.iter().map(|&x| x as f32).collect();
    for (x, reference) in c.solve_vec(&b).iter().zip(case.x_re) {
        assert!((*x as f64 - reference).abs() <= 2e-4 * reference.abs().max(1.0));
    }
}

#[test]
fn a_complex_hermitian_matrix_agrees_with_numpy() {
    use reference::COMPLEX as case;
    let n = case.n;
    let a = reference_matrix(n, case.lower);
    let b: Vec<Complex<f64>> = case
        .b_re
        .iter()
        .zip(case.b_im)
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();
    let natural = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    assert_eq!(natural.nnz(), case.nnz_natural);
    let given =
        SparseCholesky::with_ordering(&a, &Ordering::Given(case.permutation.to_vec())).unwrap();
    assert_eq!(given.nnz(), case.nnz_permuted);
    for c in [natural, given, SparseCholesky::new(&a).unwrap()] {
        let x = c.solve_vec(&b);
        for ((x, re), im) in x.iter().zip(case.x_re).zip(case.x_im) {
            let reference = Complex::new(*re, *im);
            assert!(
                (*x - reference).abs() <= 1e-13 * reference.abs().max(1.0),
                "{x:?} for {reference:?}"
            );
        }
        assert!((c.log_det() - case.log_det).abs() <= 1e-12 * case.log_det.abs());
    }
    let l = SparseCholesky::with_ordering(&a, &Ordering::Natural)
        .unwrap()
        .l();
    assert_eq!(l.nnz(), case.l_natural.len());
    for &(i, j, re, im) in case.l_natural {
        let reference = Complex::new(re, im);
        assert!(
            (l[(i, j)] - reference).abs() <= 1e-13 * reference.abs().max(1.0),
            "L[{i}, {j}]"
        );
    }
    // In single precision.
    let single: SparseMatrix<Complex<f32>> = a_single(n, case.lower);
    let c = SparseCholesky::new(&single).unwrap();
    let b: Vec<Complex<f32>> = b
        .iter()
        .map(|z| Complex::new(z.re as f32, z.im as f32))
        .collect();
    for ((x, re), im) in c.solve_vec(&b).iter().zip(case.x_re).zip(case.x_im) {
        let x = Complex::new(x.re as f64, x.im as f64);
        let reference = Complex::new(*re, *im);
        assert!((x - reference).abs() <= 2e-4 * reference.abs().max(1.0));
    }
}

/// The matrix of a reference case in single precision.
fn a_single(n: usize, lower: &[(usize, usize, f64, f64)]) -> SparseMatrix<Complex<f32>> {
    let entries: Vec<(usize, usize, Complex<f32>)> = lower
        .iter()
        .map(|&(i, j, re, im)| (i, j, Complex::new(re as f32, im as f32)))
        .collect();
    SparseMatrix::from_coo(n, n, &entries)
}
