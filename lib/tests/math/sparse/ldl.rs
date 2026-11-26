//! The supernodal sparse `L·D·Lᴴ` factorisation, checked by what defines it — `P·A·Pᵀ = L·D·Lᴴ`,
//! the residuals of its solutions and the inertia — on quasi-definite, negative definite and
//! positive definite matrices, on matrices with a zero pivot, and against NumPy and mpmath on
//! embedded matrices.

use super::cholesky::{
    assert_permutation, assert_solves, hermitian, hermitian_positive_definite, orderings, permuted,
    products, random_dense, random_permutation,
};
use super::common::{Num, Rng, assert_close, assert_csr, each_type, norm, size};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::sparse::{Ordering, SparseCholesky, SparseLdl, SparseMatrix};

/// A quasi-definite matrix `[[H, Bᴴ], [B, −C]]`: `H` of order `n` and `C` of order `m` Hermitian
/// positive definite with random patterns, `B` with about `links` random elements. Both
/// triangles are stored.
fn quasi_definite<T: Num>(r: &mut Rng, n: usize, m: usize, links: usize) -> SparseMatrix<T> {
    let h = hermitian_positive_definite::<T>(r, n, 2 * n);
    let c = hermitian_positive_definite::<T>(r, m, 2 * m);
    let mut entries = Vec::new();
    for i in 0..n {
        entries.extend(h.row(i).map(|(j, x)| (i, j, x)));
    }
    for i in 0..m {
        entries.extend(c.row(i).map(|(j, x)| (n + i, n + j, T::real(0.0) - x)));
    }
    if n > 0 && m > 0 {
        for _ in 0..links {
            let (i, j, x) = (n + r.below(m), r.below(n), T::random(r));
            entries.push((i, j, x));
            entries.push((j, i, x.conj()));
        }
    }
    SparseMatrix::from_coo(n + m, n + m, &entries)
}

/// The numbers of positive, negative and zero eigenvalues of the Hermitian `h`, by the dense
/// eigenvalues: those within rounding of zero count as zero.
fn dense_inertia<T: Num>(h: &Matrix<T>) -> (usize, usize, usize) {
    let eigenvalues = h.hermitian_eigenvalues();
    let cut = 64.0 * T::EPS * (h.rows() + 1) as f64 * norm(h);
    let positive = eigenvalues.iter().filter(|x| x.to_f64() > cut).count();
    let negative = eigenvalues.iter().filter(|x| x.to_f64() < -cut).count();
    (positive, negative, eigenvalues.len() - positive - negative)
}

/// Checks what defines the factorisation `f` of the matrix whose lower triangle is that of `a`:
/// a permutation, a lower triangular `L` with ones on its diagonal, real pivots that are finite
/// and not zero, an inertia that counts their signs, and `P·A·Pᵀ = L·D·Lᴴ` to `factor`
/// roundings of the size of `|L|·|D|·|Lᴴ|`.
fn assert_factorisation<T: Num>(a: &SparseMatrix<T>, f: &SparseLdl<T>, factor: f64) {
    let n = a.rows();
    let p = f.permutation();
    assert_permutation(p, n);
    let l = f.l();
    assert_csr(&l);
    assert_eq!((l.rows(), l.cols()), (n, n));
    assert!(
        l.nnz() <= f.nnz(),
        "the factor has more nonzeros than counted"
    );
    for i in 0..n {
        assert_eq!(
            l[(i, i)],
            T::real(1.0),
            "the diagonal of the factor isn't 1"
        );
        assert!(
            l.row(i).all(|(j, _)| j <= i),
            "the factor has an element above its diagonal"
        );
    }
    let d = f.d();
    assert_eq!(d.len(), n);
    assert!(d.iter().all(|x| x.is_finite() && *x != T::Real::ZERO));
    let positive = d.iter().filter(|&&x| x > T::Real::ZERO).count();
    assert_eq!(f.inertia(), (positive, n - positive, 0));
    // The bound of the backward error is in terms of |L|·|D|·|Lᴴ|, which is larger than the
    // matrix when pivots of both signs have let the factors grow.
    let (product, bound) = products(&l, Some(d));
    let bound = norm(&bound);
    let difference = norm(&(&product - &permuted(&hermitian(a), p)));
    let tolerance = factor * T::EPS * (n + 1) as f64 * bound;
    assert!(
        difference <= tolerance,
        "L·D·Lᴴ differs from the matrix by {difference}, more than the tolerated {tolerance}"
    );
}

/// The sizes of the two blocks of the quasi-definite matrices, and the elements that couple them.
fn shapes() -> Vec<(usize, usize, usize)> {
    if cfg!(miri) {
        vec![(1, 1, 1), (4, 2, 5), (7, 5, 12)]
    } else {
        vec![
            (1, 1, 1),
            (2, 1, 2),
            (3, 3, 4),
            (10, 4, 12),
            (20, 20, 50),
            (45, 15, 90),
            (30, 60, 150),
            (80, 40, 200),
        ]
    }
}

fn quasi_definite_of<T: Num>() {
    let mut r = Rng(0x5EED_2301);
    for (n, m, links) in shapes() {
        let a = quasi_definite::<T>(&mut r, n, m, links);
        let h = hermitian(&a);
        let b = random_dense::<T>(&mut r, n + m, 3);
        for ordering in orderings(&mut r, n + m) {
            // Quasi-definite: every permutation has a factorisation, with as many positive
            // pivots as `H` has rows and as many negative ones as `C` has.
            let f = SparseLdl::with_ordering(&a, &ordering).expect("quasi-definite");
            assert_factorisation(&a, &f, 16.0);
            assert_eq!(f.inertia(), (n, m, 0), "{n} {m} {ordering:?}");
            match &ordering {
                Ordering::Natural => assert!(f.permutation().iter().copied().eq(0..n + m)),
                Ordering::Given(p) => assert_eq!(f.permutation(), p),
                _ => {}
            }
            // The solutions: as good as the growth of the factors lets them be, which for these
            // matrices, whose blocks dominate their coupling, is little.
            let x = f.solve(&b);
            assert_solves(&h, &x, &b, 256.0);
            for k in 0..b.cols() {
                let one = Matrix::from_column_major(n + m, 1, f.solve_vec(b.column(k)));
                let of_many = Matrix::from_column_major(n + m, 1, x.column(k).to_vec());
                assert_close(&one, &of_many, 256.0);
            }
            // The same supernodes and the same count as the Cholesky factorisation of a matrix
            // of the same pattern would have.
            let order = (n + m) as f64;
            let pattern = a.map(|i, j, _| T::real(if i == j { order } else { 0.5 / order }));
            let c = SparseCholesky::with_ordering(&pattern, &ordering).unwrap();
            assert_eq!((f.nnz(), f.supernodes()), (c.nnz(), c.supernodes()));
        }
        assert_eq!(dense_inertia(&h), (n, m, 0));
    }
}

#[test]
fn a_quasi_definite_matrix_is_factored_and_solved_under_every_ordering() {
    each_type!(quasi_definite_of);
}

fn definite_of<T: Num>() {
    let mut r = Rng(0x5EED_2302);
    for (n, pairs) in [
        (1, 0),
        (2, 1),
        (9, 12),
        (size(40, 7), 70),
        (size(75, 10), 500),
    ] {
        let a = hermitian_positive_definite::<T>(&mut r, n, pairs);
        let negative = a.map(|_, _, x| T::real(0.0) - x);
        let h = hermitian(&a);
        let b = random_dense::<T>(&mut r, n, 2);
        for ordering in orderings(&mut r, n) {
            // Positive definite: the factors of the Cholesky factorisation, `L·√D`.
            let f = SparseLdl::with_ordering(&a, &ordering).unwrap();
            assert_factorisation(&a, &f, 16.0);
            assert_eq!(f.inertia(), (n, 0, 0));
            assert!(f.d().iter().all(|&d| d > T::Real::ZERO));
            let c = SparseCholesky::with_ordering(&a, &ordering).unwrap();
            assert_eq!(f.permutation(), c.permutation());
            assert_eq!((f.nnz(), f.supernodes()), (c.nnz(), c.supernodes()));
            let l = f.l().to_matrix();
            let scaled = Matrix::from_fn(n, n, |i, j| l[(i, j)] * T::from_real(f.d()[j].sqrt()));
            assert_close(&scaled, &c.l().to_matrix(), 64.0);
            // The determinant, where it is in range.
            let (det, reference) = (f.det().to_f64(), c.det().to_f64());
            if reference.is_finite() {
                assert!((det - reference).abs() <= 64.0 * T::EPS * (n + 1) as f64 * reference);
            } else {
                assert_eq!(det, reference);
            }
            assert_solves(&h, &f.solve(&b), &b, 16.0);
            assert_close(&f.solve(&b), &c.solve(&b), 256.0);

            // Negative definite: the same `L`, and `D` with the other sign.
            let g = SparseLdl::with_ordering(&negative, &ordering).unwrap();
            assert_factorisation(&negative, &g, 16.0);
            assert_eq!(g.inertia(), (0, n, 0));
            assert_eq!(g.l(), f.l());
            assert!(g.d().iter().zip(f.d()).all(|(&x, &y)| x == -y));
            assert_eq!(g.det(), if n % 2 == 0 { f.det() } else { -f.det() });
            assert_solves(&hermitian(&negative), &g.solve(&b), &b, 16.0);
            assert!(SparseCholesky::with_ordering(&negative, &ordering).is_none());
        }
    }
}

#[test]
fn definite_matrices_of_either_sign_are_factored_as_by_cholesky() {
    each_type!(definite_of);
}

fn hermitian_example_of<R: Float>()
where
    Complex<R>: Num<Real = R>,
{
    // [[1, 2 − i], [2 + i, 1]] = L·D·Lᴴ with L = [[1, 0], [2 + i, 1]] and D = (1, 1 − 5): the
    // element below the diagonal is the one given, not its conjugate.
    let c = |re: f64, im: f64| Complex::new(R::from_f64(re), R::from_f64(im));
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(1.0, 0.0)),
            (1, 0, c(2.0, 1.0)),
            (1, 1, c(1.0, 0.0)),
        ],
    );
    let f = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    assert_eq!(
        f.l().to_matrix(),
        Matrix::from_rows(&[[c(1.0, 0.0), c(0.0, 0.0)], [c(2.0, 1.0), c(1.0, 0.0)]])
    );
    assert_eq!(f.d(), [R::from_f64(1.0), R::from_f64(-4.0)]);
    assert_eq!(f.inertia(), (1, 1, 0));
    assert_eq!(f.det(), R::from_f64(-4.0));
    // A·x = b for x = (1 + 2i, 3 − i): b = (1 + 2i + (2 − i)(3 − i), (2 + i)(1 + 2i) + 3 − i).
    let x = f.solve_vec(&[c(6.0, -3.0), c(3.0, 4.0)]);
    assert_eq!(x, [c(1.0, 2.0), c(3.0, -1.0)]);
    // Stored with both triangles, or with a wrong upper one, it is the same matrix.
    let wrong = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, c(1.0, 0.0)),
            (0, 1, c(9.0, 9.0)),
            (1, 0, c(2.0, 1.0)),
            (1, 1, c(1.0, 0.0)),
        ],
    );
    let g = SparseLdl::with_ordering(&wrong, &Ordering::Natural).unwrap();
    assert_eq!((g.l(), g.d()), (f.l(), f.d()));
    // Of a diagonal element the real part is read, as the dense methods read it (audit
    // p_direct-7 found the earlier Cholesky refusing a part of rounding size): an imaginary part
    // of any size changes nothing.
    let least = (R::MIN_POSITIVE * R::EPSILON).to_f64();
    for (first, last) in [(1e-17, 0.0), (0.0, -least), (3.0, -2.0)] {
        let stray = SparseMatrix::from_coo(
            2,
            2,
            &[
                (0, 0, c(1.0, first)),
                (1, 0, c(2.0, 1.0)),
                (1, 1, c(1.0, last)),
            ],
        );
        let g = SparseLdl::with_ordering(&stray, &Ordering::Natural).unwrap();
        assert_eq!((g.l(), g.d()), (f.l(), f.d()));
    }
}

#[test]
fn a_complex_hermitian_indefinite_matrix_keeps_its_conjugations() {
    hermitian_example_of::<f32>();
    hermitian_example_of::<f64>();
}

fn zero_pivots_of<T: Num>() {
    let t = T::real;
    let natural = |a: &SparseMatrix<T>| SparseLdl::with_ordering(a, &Ordering::Natural);
    // The first pivot is zero, though the matrix is nonsingular.
    let swap = SparseMatrix::from_coo(2, 2, &[(1, 0, t(1.0))]);
    for ordering in orderings(&mut Rng(31), 2) {
        assert!(SparseLdl::with_ordering(&swap, &ordering).is_none());
    }
    let a = SparseMatrix::from_coo(2, 2, &[(1, 0, t(1.0)), (1, 1, t(2.0))]);
    assert!(natural(&a).is_none());
    // The other way round it has one, with the pivots 2 and −1/2.
    let f = SparseLdl::with_ordering(&a, &Ordering::Given(vec![1, 0])).unwrap();
    assert_eq!(f.d(), [T::Real::from_f64(2.0), T::Real::from_f64(-0.5)]);
    assert_factorisation(&a, &f, 4.0);
    // A zero on the diagonal is no obstacle where an earlier column has filled it.
    let b = SparseMatrix::from_coo(2, 2, &[(0, 0, t(2.0)), (1, 0, t(1.0))]);
    assert_eq!(
        natural(&b).unwrap().d(),
        [T::Real::from_f64(2.0), T::Real::from_f64(-0.5)]
    );
    // A pivot that the elimination makes zero, exactly: [[1, 1], [1, 1]], and the same inside a
    // larger block and in a later one.
    assert!(natural(&SparseMatrix::from_fn(2, 2, |_, _| t(1.0))).is_none());
    assert!(natural(&SparseMatrix::<T>::zeros(1, 1)).is_none());
    assert!(natural(&SparseMatrix::<T>::zeros(4, 4)).is_none());
    for n in [3, 8, size(40, 6), size(90, 7)] {
        // Ones everywhere: the second pivot is 1 − 1.
        let ones = SparseMatrix::from_fn(n, n, |_, _| t(1.0));
        for ordering in orderings(&mut Rng(32), n) {
            assert!(SparseLdl::with_ordering(&ones, &ordering).is_none());
        }
        // The identity bordered by a row of ones with n − 1 on its diagonal: the last pivot is
        // n − 1 less the n − 1 ones before it. One more on that diagonal and it is 1.
        let mut entries: Vec<(usize, usize, T)> = (0..n - 1).map(|i| (i, i, t(1.0))).collect();
        entries.extend((0..n - 1).map(|j| (n - 1, j, t(1.0))));
        entries.push((n - 1, n - 1, t((n - 1) as f64)));
        assert!(
            natural(&SparseMatrix::from_coo(n, n, &entries)).is_none(),
            "{n}"
        );
        entries.push((n - 1, n - 1, t(1.0)));
        let f = natural(&SparseMatrix::from_coo(n, n, &entries)).unwrap();
        assert_eq!(f.d()[n - 1], T::Real::from_f64(1.0));
        assert_eq!(f.inertia(), (n, 0, 0));
        // A zero in the middle of the diagonal of a diagonal matrix.
        let mut diagonal = vec![t(-3.0); n];
        diagonal[n / 2] = t(0.0);
        for ordering in orderings(&mut Rng(33), n) {
            assert!(
                SparseLdl::with_ordering(&SparseMatrix::from_diagonal(&diagonal), &ordering)
                    .is_none()
            );
        }
    }
    // A saddle-point matrix without regularisation, the constraints first: zero pivots at once.
    // With the constraints last it is the quasi-definite case in the limit, and goes through.
    let (n, m) = (size(12, 4), size(5, 2));
    let mut r = Rng(0x5EED_2303);
    let h = hermitian_positive_definite::<T>(&mut r, n, 2 * n);
    let mut first = Vec::new();
    let mut last = Vec::new();
    for i in 0..n {
        for (j, x) in h.row(i) {
            first.push((m + i, m + j, x));
            last.push((i, j, x));
        }
    }
    for i in 0..m {
        // Each constraint has a variable of its own, so that they are independent.
        for j in [i, m + r.below(n - m)] {
            let x = t(1.0 + r.unit().abs());
            first.push((m + j, i, x));
            last.push((n + i, j, x));
        }
    }
    assert!(natural(&SparseMatrix::from_coo(n + m, n + m, &first)).is_none());
    let a = SparseMatrix::from_coo(n + m, n + m, &last);
    let f = natural(&a).unwrap();
    assert_eq!(f.inertia(), (n, m, 0));
    assert_factorisation(&a, &f, 64.0);
}

#[test]
fn a_zero_pivot_gives_no_factorisation() {
    each_type!(zero_pivots_of);
}

fn inertia_of<T: Num>() {
    let mut r = Rng(0x5EED_2304);
    // Blocks of either sign on the diagonal, mixed by a random permutation: no pivot can
    // vanish, and the inertia is the sizes of the blocks.
    for (sizes, signs) in [
        (vec![3usize, 4], vec![1.0, -1.0]),
        (
            vec![size(20, 4), size(15, 3), size(8, 2)],
            vec![-1.0, 1.0, -1.0],
        ),
        (
            vec![size(30, 5), 1, size(12, 3), 2],
            vec![1.0, -1.0, -1.0, 1.0],
        ),
    ] {
        let n: usize = sizes.iter().sum();
        let mut entries = Vec::new();
        let mut offset = 0;
        for (&k, &sign) in sizes.iter().zip(&signs) {
            let block = hermitian_positive_definite::<T>(&mut r, k, 3 * k);
            for i in 0..k {
                entries.extend(
                    block
                        .row(i)
                        .map(|(j, x)| (offset + i, offset + j, x * T::real(sign))),
                );
            }
            offset += k;
        }
        let mix = random_permutation(&mut r, n);
        let mixed: Vec<(usize, usize, T)> = entries
            .iter()
            .map(|&(i, j, x)| (mix[i], mix[j], x))
            .collect();
        let a = SparseMatrix::from_coo(n, n, &mixed);
        let positive: usize = sizes
            .iter()
            .zip(&signs)
            .filter(|s| *s.1 > 0.0)
            .map(|s| *s.0)
            .sum();
        assert_eq!(dense_inertia(&hermitian(&a)), (positive, n - positive, 0));
        for ordering in orderings(&mut r, n) {
            let f = SparseLdl::with_ordering(&a, &ordering).unwrap();
            assert_factorisation(&a, &f, 16.0);
            assert_eq!(f.inertia(), (positive, n - positive, 0));
            // The determinant against the dense one, which is the product of the pivots of a
            // factorisation with pivoting.
            let (det, reference) = (f.det().to_f64(), hermitian(&a).det().re().to_f64());
            assert!(
                (det - reference).abs() <= 256.0 * T::EPS * (n + 1) as f64 * reference.abs(),
                "{det} for {reference}"
            );
            let product: f64 = f.d().iter().map(|d| d.to_f64()).product();
            assert!((det - product).abs() <= 8.0 * T::EPS * (n + 1) as f64 * product.abs());
        }
    }
}

#[test]
fn the_inertia_counts_the_signs_of_the_eigenvalues() {
    each_type!(inertia_of);
}

fn smallest_of<T: Num>() {
    let empty = SparseMatrix::<T>::zeros(0, 0);
    for ordering in [Ordering::Natural, Ordering::Amd, Ordering::Given(vec![])] {
        let f = SparseLdl::with_ordering(&empty, &ordering).unwrap();
        assert!(f.d().is_empty() && f.permutation().is_empty());
        assert_eq!(f.inertia(), (0, 0, 0));
        assert_eq!(f.det().to_f64(), 1.0);
        assert_eq!((f.nnz(), f.supernodes()), (0, 0));
        assert_eq!((f.l().rows(), f.l().nnz()), (0, 0));
        assert_eq!(f.solve(&Matrix::zeros(0, 2)).shape(), (0, 2));
        assert!(f.solve_vec(&[]).is_empty());
    }
    for (x, inertia) in [(-3.0, (0, 1, 0)), (5.0, (1, 0, 0))] {
        let f = SparseLdl::new(&SparseMatrix::from_diagonal(&[T::real(x)])).unwrap();
        assert_eq!(f.d(), [T::Real::from_f64(x)]);
        assert_eq!(f.inertia(), inertia);
        assert_eq!(f.det().to_f64(), x);
        assert_eq!(f.l().to_matrix(), Matrix::from_rows(&[[T::real(1.0)]]));
        assert_eq!((f.nnz(), f.supernodes()), (1, 1));
        assert_eq!(f.solve_vec(&[T::real(2.0 * x)]), [T::real(2.0)]);
        assert_eq!(f.solve(&Matrix::zeros(1, 0)).shape(), (1, 0));
    }
    for bad in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(SparseLdl::new(&SparseMatrix::from_diagonal(&[T::real(bad)])).is_none());
    }
}

#[test]
fn empty_and_one_by_one_matrices() {
    each_type!(smallest_of);
}

fn upper_triangle_of<T: Num>() {
    let mut r = Rng(0x5EED_2305);
    let (n, m) = (size(25, 5), size(15, 3));
    let full = quasi_definite::<T>(&mut r, n, m, 2 * n);
    let order = n + m;
    let lower = full.map(|i, j, x| if j <= i { x } else { T::real(0.0) });
    let mut entries: Vec<(usize, usize, T)> = Vec::new();
    for i in 0..order {
        entries.extend(lower.row(i).map(|(j, x)| (i, j, x)));
    }
    for k in 0..3 * order {
        let (i, j) = (r.below(order), r.below(order));
        if i < j {
            let x = match k % 5 {
                0 => T::real(f64::NAN),
                1 => T::real(f64::NEG_INFINITY),
                _ => T::random(&mut r) * T::real(1e3),
            };
            entries.push((i, j, x));
        }
    }
    let garbage = SparseMatrix::from_coo(order, order, &entries);
    assert!(garbage.nnz() > lower.nnz());
    let b = random_dense::<T>(&mut r, order, 2);
    for ordering in orderings(&mut r, order) {
        let reference = SparseLdl::with_ordering(&full, &ordering).unwrap();
        for other in [&lower, &garbage] {
            let f = SparseLdl::with_ordering(other, &ordering).unwrap();
            assert_eq!(f.permutation(), reference.permutation(), "{ordering:?}");
            assert_eq!(f.l(), reference.l(), "{ordering:?}");
            assert_eq!(f.d(), reference.d(), "{ordering:?}");
            assert_eq!(f.solve(&b), reference.solve(&b), "{ordering:?}");
        }
    }
}

#[test]
fn what_stands_above_the_diagonal_is_not_read() {
    each_type!(upper_triangle_of);
}

fn not_finite_of<T: Num>() {
    let mut r = Rng(0x5EED_2306);
    let (n, m) = (size(24, 5), size(12, 3));
    let a = quasi_definite::<T>(&mut r, n, m, 2 * n);
    let order = n + m;
    let stored: Vec<(usize, usize)> = (0..order)
        .flat_map(|i| a.row(i).map(move |(j, _)| (i, j)))
        .filter(|&(i, j)| j <= i)
        .collect();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for _ in 0..6 {
            let (bi, bj) = stored[r.below(stored.len())];
            let spoiled = a.map(|i, j, x| if (i, j) == (bi, bj) { T::real(bad) } else { x });
            for ordering in orderings(&mut r, order) {
                assert!(
                    SparseLdl::with_ordering(&spoiled, &ordering).is_none(),
                    "{bad} at ({bi}, {bj}) under {ordering:?}"
                );
            }
        }
    }
}

#[test]
fn a_matrix_with_an_element_that_is_not_finite_has_no_factorisation() {
    each_type!(not_finite_of);
}

fn large_supernodes_of<T: Num>() {
    let mut r = Rng(0x5EED_2307);
    // A full quasi-definite matrix is one supernode, wider than the dense kernel takes whole:
    // the recursion on halves, its solve and its products.
    let (n, m) = (size(90, 10), size(60, 8));
    let order = n + m;
    let h = hermitian_positive_definite::<T>(&mut r, n, n * n);
    let c = hermitian_positive_definite::<T>(&mut r, m, m * m);
    let scale = T::real(1.0 / (order as f64));
    let a = SparseMatrix::from_fn(order, order, |i, j| match (i < n, j < n) {
        (true, true) => h[(i, j)],
        (false, false) => T::real(0.0) - c[(i - n, j - n)],
        (false, true) => T::real(0.25 + ((i * 7 + j * 3) % 5) as f64) * scale,
        (true, false) => T::real(0.25 + ((j * 7 + i * 3) % 5) as f64) * scale,
    });
    let dense = hermitian(&a);
    let b = random_dense::<T>(&mut r, order, 3);
    for ordering in [
        Ordering::Natural,
        Ordering::Given(random_permutation(&mut r, order)),
    ] {
        let f = SparseLdl::with_ordering(&a, &ordering).unwrap();
        if !cfg!(miri) {
            assert_eq!(f.supernodes(), 1);
        }
        assert_factorisation(&a, &f, 16.0);
        assert_eq!(f.inertia(), (n, m, 0));
        let x = f.solve(&b);
        assert_solves(&dense, &x, &b, 256.0);
        assert_close(&x, &dense.solve(&b).unwrap(), 4096.0);
    }
    // And a negative definite full matrix, whose factors are those of the Cholesky method.
    let negative = SparseMatrix::from_fn(n, n, |i, j| T::real(0.0) - h[(i, j)]);
    let f = SparseLdl::new(&negative).unwrap();
    assert_factorisation(&negative, &f, 16.0);
    assert_eq!(f.inertia(), (0, n, 0));
}

#[test]
fn supernodes_larger_than_the_dense_kernel_takes_whole() {
    each_type!(large_supernodes_of);
}

fn right_hand_sides_of<T: Num>() {
    let mut r = Rng(0x5EED_2308);
    let (n, m) = (size(30, 5), size(20, 4));
    let a = quasi_definite::<T>(&mut r, n, m, 2 * n);
    let h = hermitian(&a);
    let f = SparseLdl::new(&a).unwrap();
    for columns in [0, 1, 2, 7, size(19, 3)] {
        let b = random_dense::<T>(&mut r, n + m, columns);
        let x = f.solve(&b);
        assert_eq!(x.shape(), (n + m, columns));
        assert_solves(&h, &x, &b, 256.0);
    }
    assert_eq!(f.solve(&Matrix::zeros(n + m, 2)), Matrix::zeros(n + m, 2));
    let e = Matrix::from_fn(n + m, 1, |i, _| T::real(1.0 + (i % 3) as f64));
    assert_close(&f.solve(&(&h * &e)), &e, 1024.0);
}

#[test]
fn any_number_of_right_hand_sides_is_solved() {
    each_type!(right_hand_sides_of);
}

#[test]
fn the_factorisation_can_be_cloned_and_written() {
    let a = SparseMatrix::from_coo(
        3,
        3,
        &[
            (0, 0, 2.0),
            (1, 1, 2.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 2, -1.0),
        ],
    );
    let f = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    let g = f.clone();
    assert_eq!((f.l(), f.d()), (g.l(), g.d()));
    assert_eq!(f.d(), [2.0, 2.0, -2.0]);
    assert_eq!(
        format!("{f:?}"),
        "SparseLdl { order: 3, nnz: 5, supernodes: 1, inertia: (2, 1, 0) }"
    );
}

#[test]
#[should_panic(expected = "the matrix isn't square")]
fn a_matrix_that_is_not_square_panics() {
    let _ = SparseLdl::new(&SparseMatrix::<f64>::zeros(3, 2));
}

#[test]
#[should_panic(expected = "the right-hand side has not as many rows")]
fn a_right_hand_side_of_another_height_panics() {
    let f = SparseLdl::new(&SparseMatrix::<f64>::identity(3)).unwrap();
    let _ = f.solve(&Matrix::zeros(4, 1));
}

#[test]
#[should_panic(expected = "the right-hand side has not as many elements")]
fn a_right_hand_side_vector_of_another_length_panics() {
    let f = SparseLdl::new(&SparseMatrix::<f64>::identity(3)).unwrap();
    let _ = f.solve_vec(&[1.0; 2]);
}

#[test]
#[should_panic(expected = "the permutation is out of range")]
fn a_given_ordering_that_is_not_a_permutation_panics() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = SparseLdl::with_ordering(&a, &Ordering::Given(vec![0, 1, 3]));
}

/// The matrices, the pivots and the solutions of `data/gen_cholesky.py`, computed with NumPy,
/// SciPy and mpmath.
mod reference {
    include!("data/ldl_reference.rs");
}

#[test]
fn a_real_saddle_point_matrix_agrees_with_numpy() {
    use reference::REAL as case;
    let n = case.n;
    let entries: Vec<(usize, usize, f64)> =
        case.lower.iter().map(|&(i, j, re, _)| (i, j, re)).collect();
    let a = SparseMatrix::from_coo(n, n, &entries);
    let natural = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    let given = SparseLdl::with_ordering(&a, &Ordering::Given(case.permutation.to_vec())).unwrap();
    // The pivots of the two orders, one by one.
    for (f, pivots) in [(&natural, case.d_natural), (&given, case.d_permuted)] {
        assert_eq!(f.d().len(), n);
        for (d, reference) in f.d().iter().zip(pivots) {
            assert!(
                (d - reference).abs() <= 1e-12 * reference.abs(),
                "{d} for {reference}"
            );
        }
        assert_eq!(f.inertia(), case.inertia);
    }
    assert_eq!(natural.nnz(), case.nnz_natural);
    assert_eq!(given.nnz(), case.nnz_permuted);
    for &(i, j, re, _) in case.l_natural {
        let x = natural.l()[(i, j)];
        assert!(
            (x - re).abs() <= 1e-12 * re.abs().max(1.0),
            "L[{i}, {j}]: {x} for {re}"
        );
    }
    assert_eq!(natural.l().nnz(), case.l_natural.len());
    for f in [natural, given, SparseLdl::new(&a).unwrap()] {
        let x = f.solve_vec(case.b_re);
        for (x, reference) in x.iter().zip(case.x_re) {
            assert!(
                (x - reference).abs() <= 1e-12 * reference.abs().max(1.0),
                "{x} for {reference}"
            );
        }
        assert!((f.det() - case.det).abs() <= 1e-11 * case.det.abs());
        assert_eq!(f.inertia(), case.inertia);
    }
}

#[test]
fn a_complex_saddle_point_matrix_agrees_with_numpy() {
    use reference::COMPLEX as case;
    let n = case.n;
    let entries: Vec<(usize, usize, Complex<f64>)> = case
        .lower
        .iter()
        .map(|&(i, j, re, im)| (i, j, Complex::new(re, im)))
        .collect();
    let a = SparseMatrix::from_coo(n, n, &entries);
    let b: Vec<Complex<f64>> = case
        .b_re
        .iter()
        .zip(case.b_im)
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();
    let natural = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    let given = SparseLdl::with_ordering(&a, &Ordering::Given(case.permutation.to_vec())).unwrap();
    for (f, pivots) in [(&natural, case.d_natural), (&given, case.d_permuted)] {
        for (d, reference) in f.d().iter().zip(pivots) {
            assert!(
                (d - reference).abs() <= 1e-12 * reference.abs(),
                "{d} for {reference}"
            );
        }
        assert_eq!(f.inertia(), case.inertia);
    }
    assert_eq!(natural.nnz(), case.nnz_natural);
    assert_eq!(given.nnz(), case.nnz_permuted);
    for &(i, j, re, im) in case.l_natural {
        let reference = Complex::new(re, im);
        let x = natural.l()[(i, j)];
        assert!(
            (x - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "L[{i}, {j}]"
        );
    }
    assert_eq!(natural.l().nnz(), case.l_natural.len());
    for f in [natural, given, SparseLdl::new(&a).unwrap()] {
        let x = f.solve_vec(&b);
        for ((x, re), im) in x.iter().zip(case.x_re).zip(case.x_im) {
            let reference = Complex::new(*re, *im);
            assert!(
                (*x - reference).abs() <= 1e-12 * reference.abs().max(1.0),
                "{x:?} for {reference:?}"
            );
        }
        assert!((f.det() - case.det).abs() <= 1e-11 * case.det.abs());
    }
    // In single precision.
    let single: Vec<(usize, usize, Complex<f32>)> = entries
        .iter()
        .map(|&(i, j, x)| (i, j, Complex::new(x.re as f32, x.im as f32)))
        .collect();
    let f = SparseLdl::new(&SparseMatrix::from_coo(n, n, &single)).unwrap();
    assert_eq!(f.inertia(), case.inertia);
    let b: Vec<Complex<f32>> = b
        .iter()
        .map(|z| Complex::new(z.re as f32, z.im as f32))
        .collect();
    for ((x, re), im) in f.solve_vec(&b).iter().zip(case.x_re).zip(case.x_im) {
        let x = Complex::new(x.re as f64, x.im as f64);
        let reference = Complex::new(*re, *im);
        assert!((x - reference).abs() <= 1e-3 * reference.abs().max(1.0));
    }
}
