//! The preconditioners of the iterative solvers: `Jacobi`, `Ilu0` and `Ic0`, each against its
//! defining property, on the four element types, and against the defects the audit found
//! (`p_iter-5`, `p_iter-6`, `p_iter-14`, `p_direct-5`).

use super::common::{Num, Rng, each_type, magnitude, size};
use super::solve::{c, problem, within};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::sparse::{
    Ic0, Ilu0, Jacobi, Preconditioner, SolveOptions, SparseMatrix, StopReason, bicgstab_with,
    cg_with, gmres_with, minres_with,
};

/// A random square matrix with about `density` off-diagonal elements, structurally unsymmetric,
/// and a diagonal of either sign that dominates its row, so that ILU(0) exists.
pub(super) fn dominant<T: Num>(r: &mut Rng, n: usize, density: usize) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    let mut sums = vec![0.0f64; n];
    for _ in 0..density {
        let (i, j) = (r.below(n), r.below(n));
        if i != j {
            let x = T::random(r);
            sums[i] += magnitude(x);
            entries.push((i, j, x));
        }
    }
    for (i, &sum) in sums.iter().enumerate() {
        let sign = if r.below(2) == 0 { 1.0 } else { -1.0 };
        let size = sum + 0.25 + r.unit().abs();
        // A complex diagonal entry of that modulus for the complex types.
        let angle = if T::IS_COMPLEX { r.unit() } else { 0.0 };
        entries.push((i, i, c::<T>(sign * size * angle.cos(), size * angle.sin())));
    }
    SparseMatrix::from_coo(n, n, &entries)
}

/// A random Hermitian matrix with complex elements off the diagonal and a positive diagonal
/// that dominates its row: positive definite, and IC(0) exists.
pub(super) fn hermitian<T: Num>(r: &mut Rng, n: usize, density: usize) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    let mut sums = vec![0.0f64; n];
    for _ in 0..density {
        let (i, j) = (r.below(n), r.below(n));
        if i != j {
            let x = T::random(r);
            sums[i] += magnitude(x);
            sums[j] += magnitude(x);
            entries.push((i, j, x));
            entries.push((j, i, x.conj()));
        }
    }
    for (i, &sum) in sums.iter().enumerate() {
        entries.push((i, i, T::real(sum + 0.25 + r.unit().abs())));
    }
    SparseMatrix::from_coo(n, n, &entries)
}

/// The dense form with every element replaced by its absolute value.
fn absolute<T: Num>(a: &SparseMatrix<T>) -> Matrix<f64> {
    let d = a.to_matrix();
    let mut m = Matrix::zeros(d.rows(), d.cols());
    for i in 0..d.rows() {
        for j in 0..d.cols() {
            m[(i, j)] = magnitude(d[(i, j)]);
        }
    }
    m
}

/// A random vector.
fn vector<T: Num>(r: &mut Rng, n: usize) -> Vec<T> {
    (0..n).map(|_| T::random(r)).collect()
}

/// `Σ conj(uᵢ)·vᵢ` in the element type.
fn inner<T: Num>(u: &[T], v: &[T]) -> T {
    u.iter()
        .zip(v)
        .fold(T::zero(), |s, (&u, &v)| s + u.conj() * v)
}

/// Asserts that `product` equals `a` on every stored element of `a` that `keep` selects, to the
/// componentwise bound of an elimination, `n·ε·(|F₁|·|F₂|)ᵢⱼ` for the factors `F₁`, `F₂`.
fn assert_on_pattern<T: Num>(
    a: &SparseMatrix<T>,
    product: &Matrix<T>,
    bound: &Matrix<f64>,
    keep: impl Fn(usize, usize) -> bool,
) {
    let n = a.rows() as f64;
    for i in 0..a.rows() {
        for (j, x) in a.row(i) {
            if keep(i, j) {
                let error = magnitude(product[(i, j)] - x);
                let allowed = 4.0 * (n + 2.0) * T::EPS * bound[(i, j)] + f64::MIN_POSITIVE;
                assert!(
                    error <= allowed,
                    "({i}, {j}): the product of the factors is {:?}, the matrix {x:?}",
                    product[(i, j)]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Jacobi
// ---------------------------------------------------------------------------------------------

#[test]
fn jacobi_divides_by_the_diagonal_for_every_element_type() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0001);
        for n in [1, 2, 7, 30] {
            let a = dominant::<T>(&mut r, n, 3 * n);
            let jacobi = Jacobi::new(&a).expect("the diagonal is full and nonzero");
            let v = vector::<T>(&mut r, n);
            let mut z = vec![T::zero(); n];
            jacobi.apply(&v, &mut z);
            let d = a.diagonal();
            for i in 0..n {
                let error = magnitude(z[i] * d[i] - v[i]);
                assert!(error <= 4.0 * T::EPS * magnitude(v[i]) + f64::MIN_POSITIVE);
            }
        }
    }
    each_type!(check);
}

#[test]
fn jacobi_is_none_for_a_missing_zero_or_non_finite_diagonal_entry() {
    let missing = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0f64), (1, 0, 1.0)]);
    assert!(Jacobi::new(&missing).is_none());
    // The audit's p_iter-14: a NaN on the diagonal made a preconditioner of NaNs.
    let nan = SparseMatrix::from_diagonal(&[1.0f64, f64::NAN]);
    assert!(Jacobi::new(&nan).is_none());
    let infinite = SparseMatrix::from_diagonal(&[1.0f64, f64::INFINITY]);
    assert!(Jacobi::new(&infinite).is_none());
    let complex =
        SparseMatrix::from_diagonal(&[Complex::new(1.0f32, 0.0), Complex::new(0.0, f32::NAN)]);
    assert!(Jacobi::new(&complex).is_none());
    // The empty matrix has an empty preconditioner.
    let empty = SparseMatrix::<f64>::zeros(0, 0);
    let jacobi = Jacobi::new(&empty).unwrap();
    jacobi.apply(&[], &mut []);
}

#[test]
#[should_panic(expected = "square")]
fn jacobi_of_a_rectangular_matrix_panics() {
    // The audit's p_iter-14: a 2 x 3 matrix used to give a preconditioner.
    let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 1.0f64), (1, 1, 2.0)]);
    let _ = Jacobi::new(&a);
}

#[test]
#[should_panic(expected = "wrong length")]
fn jacobi_applied_to_a_vector_of_the_wrong_length_panics() {
    // The audit's p_iter-14: the shorter of the two slices used to be taken.
    let jacobi = Jacobi::new(&SparseMatrix::from_diagonal(&[2.0f64, 4.0])).unwrap();
    let mut z = [0.0; 3];
    jacobi.apply(&[1.0, 2.0, 3.0], &mut z);
}

// ---------------------------------------------------------------------------------------------
// ILU(0)
// ---------------------------------------------------------------------------------------------

#[test]
fn ilu0_reproduces_every_stored_element_of_the_matrix() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0002);
        for (n, density) in [(1, 0), (2, 2), (5, 8), (12, 30), (40, 200), (60, 150)] {
            let a = dominant::<T>(&mut r, n, density);
            let ilu = Ilu0::new(&a).expect("ILU(0) of a diagonally dominant matrix exists");
            let (l, u) = (ilu.l(), ilu.u());
            assert_eq!(ilu.nnz(), a.nnz());
            // L is unit lower triangular, U upper triangular, and neither leaves the pattern.
            for i in 0..n {
                for (j, x) in l.row(i) {
                    assert!(j <= i, "L has ({i}, {j})");
                    assert!(j == i || a.find(i, j).is_some(), "L fills ({i}, {j})");
                    if j == i {
                        assert_eq!(x, T::one());
                    }
                }
                for (j, _) in u.row(i) {
                    assert!(j >= i, "U has ({i}, {j})");
                    assert!(a.find(i, j).is_some(), "U fills ({i}, {j})");
                }
            }
            let product = &l.to_matrix() * &u.to_matrix();
            let bound = &absolute(&l) * &absolute(&u);
            assert_on_pattern(&a, &product, &bound, |_, _| true);
        }
    }
    each_type!(check);
}

#[test]
fn ilu0_applies_the_inverse_of_its_factors() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0003);
        for (n, density) in [(1, 0), (6, 12), (30, 120)] {
            let a = dominant::<T>(&mut r, n, density);
            let ilu = Ilu0::new(&a).unwrap();
            let v = vector::<T>(&mut r, n);
            let mut z = vec![T::zero(); n];
            ilu.apply(&v, &mut z);
            // L·U·z = v, to the rounding of the two substitutions.
            let back = ilu.l().mul_vec(&ilu.u().mul_vec(&z));
            let scale = absolute(&ilu.l()) * absolute(&ilu.u());
            for i in 0..n {
                let size: f64 = (0..n).map(|j| scale[(i, j)] * magnitude(z[j])).sum();
                let error = magnitude(back[i] - v[i]);
                assert!(error <= 8.0 * (n as f64 + 2.0) * T::EPS * size + f64::MIN_POSITIVE);
            }
        }
    }
    each_type!(check);
}

#[test]
fn ilu0_of_a_tridiagonal_matrix_is_its_lu_factorisation() {
    fn check<T: Num>() {
        // No fill, so the incomplete factors are the complete ones and M⁻¹ is A⁻¹.
        let n = 8;
        let a = SparseMatrix::tridiagonal(
            &[c::<T>(-1.0, 0.5); 7],
            &[c::<T>(3.0, -1.0); 8],
            &[c::<T>(-1.5, 0.0); 7],
        );
        let ilu = Ilu0::new(&a).unwrap();
        let product = &ilu.l().to_matrix() * &ilu.u().to_matrix();
        let difference = &product - &a.to_matrix();
        for i in 0..n {
            for j in 0..n {
                assert!(magnitude(difference[(i, j)]) <= 32.0 * T::EPS);
            }
        }
        let x: Vec<T> = (0..n).map(|i| c::<T>(i as f64 - 3.0, 1.0)).collect();
        let b = a.mul_vec(&x);
        let mut z = vec![T::zero(); n];
        ilu.apply(&b, &mut z);
        for i in 0..n {
            assert!(magnitude(z[i] - x[i]) <= 256.0 * T::EPS * 4.0);
        }
    }
    each_type!(check);
}

#[test]
fn ilu0_is_none_without_a_diagonal_entry_or_with_a_zero_pivot() {
    let missing = SparseMatrix::from_coo(2, 2, &[(0, 1, 1.0f64), (1, 0, 1.0), (1, 1, 1.0)]);
    assert!(Ilu0::new(&missing).is_none());
    // The second pivot is 1 − 1·1 = 0.
    let singular = SparseMatrix::from_coo(
        2,
        2,
        &[(0, 0, 1.0f64), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)],
    );
    assert!(Ilu0::new(&singular).is_none());
    let singular = SparseMatrix::from_coo(
        2,
        2,
        &[(0, 0, 2.0f64), (0, 1, 1.0), (1, 0, 4.0), (1, 1, 2.0)],
    );
    assert!(Ilu0::new(&singular).is_none());
    // The audit's p_iter-14: a NaN used to give factors of NaNs.
    let nan = SparseMatrix::from_coo(
        2,
        2,
        &[(0, 0, 1.0f64), (0, 1, f64::NAN), (1, 0, 2.0), (1, 1, 1.0)],
    );
    assert!(Ilu0::new(&nan).is_none());
    let nan = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0f64), (1, 0, f64::INFINITY), (1, 1, 1.0)]);
    assert!(Ilu0::new(&nan).is_none());
    // The empty matrix factors.
    assert_eq!(
        Ilu0::new(&SparseMatrix::<f64>::zeros(0, 0)).unwrap().nnz(),
        0
    );
}

#[test]
#[should_panic(expected = "wrong length")]
fn ilu0_applied_to_a_vector_of_the_wrong_length_panics() {
    // The audit's p_iter-14: zeros used to be written and nothing said.
    let ilu = Ilu0::new(&SparseMatrix::from_diagonal(&[2.0f64, 4.0])).unwrap();
    let mut z = [0.0; 2];
    ilu.apply(&[1.0], &mut z);
}

#[test]
#[should_panic(expected = "square")]
fn ilu0_of_a_rectangular_matrix_panics() {
    let _ = Ilu0::new(&SparseMatrix::from_coo(
        2,
        3,
        &[(0, 0, 1.0f64), (1, 1, 1.0)],
    ));
}

#[test]
fn ilu0_and_ic0_take_time_in_proportion_to_their_updates() {
    // The 2-D Laplacian of 250 000 unknowns: a few dozen operations a row, a fraction of a
    // second even unoptimised. An elimination that searched rows or kept dense work vectors
    // would cost ~n² = 6·10¹⁰ here, minutes optimised, and the watchdog fails it (the review of
    // round 1 found the test only slowed by such a regression; the minute allows for a loaded
    // machine).
    let k = size(500, 6);
    let (a, _, _) = problem::<f64>(&format!("lap2d{k}"));
    let (ilu, ic, a) = within(60, move || {
        (Ilu0::new(&a).unwrap(), Ic0::new(&a).unwrap(), a)
    });
    assert_eq!(ilu.nnz(), a.nnz());
    assert_eq!(ic.nnz(), (a.nnz() + k * k) / 2);
}

// ---------------------------------------------------------------------------------------------
// IC(0)
// ---------------------------------------------------------------------------------------------

#[test]
fn ic0_reproduces_the_stored_lower_triangle() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0004);
        for (n, density) in [(1, 0), (2, 1), (5, 6), (12, 25), (40, 120), (60, 300)] {
            let a = hermitian::<T>(&mut r, n, density);
            let ic = Ic0::new(&a).expect("IC(0) of a diagonally dominant matrix exists");
            let l = ic.l();
            for i in 0..n {
                for (j, x) in l.row(i) {
                    assert!(j <= i, "L has ({i}, {j})");
                    assert!(a.find(i, j).is_some(), "L fills ({i}, {j})");
                    if j == i {
                        assert!(x.im() == T::Real::ZERO && x.re() > T::Real::ZERO);
                    }
                }
            }
            let ld = l.to_matrix();
            let product = &ld * &ld.conjugate_transpose();
            let la = absolute(&l);
            let bound = &la * &la.transpose();
            assert_on_pattern(&a, &product, &bound, |i, j| j <= i);
        }
    }
    each_type!(check);
}

#[test]
fn ic0_applies_the_inverse_of_l_times_its_conjugate_transpose() {
    fn check<T: Num>() {
        let mut r = Rng(0x5eed_0005);
        for (n, density) in [(1, 0), (6, 10), (40, 150)] {
            let a = hermitian::<T>(&mut r, n, density);
            let ic = Ic0::new(&a).unwrap();
            let l = ic.l();
            let lh = l.conjugate_transpose();
            let v = vector::<T>(&mut r, n);
            let mut z = vec![T::zero(); n];
            ic.apply(&v, &mut z);
            let back = l.mul_vec(&lh.mul_vec(&z));
            let la = absolute(&l);
            let scale = &la * &la.transpose();
            for i in 0..n {
                let size: f64 = (0..n).map(|j| scale[(i, j)] * magnitude(z[j])).sum();
                assert!(magnitude(back[i] - v[i]) <= 8.0 * (n as f64 + 2.0) * T::EPS * size);
            }
            // The audit's p_iter-6: the operator was not Hermitian, so not one cg could use.
            let u = vector::<T>(&mut r, n);
            let mut y = vec![T::zero(); n];
            ic.apply(&u, &mut y);
            let (left, right) = (inner(&u, &z), inner(&v, &y).conj());
            let size = magnitude(inner(&u, &u)).sqrt() * magnitude(inner(&z, &z)).sqrt();
            assert!(magnitude(left - right) <= 64.0 * n as f64 * T::EPS * size.max(1.0));
        }
    }
    each_type!(check);
}

#[test]
fn ic0_of_a_tridiagonal_matrix_is_its_cholesky_factor() {
    fn check<T: Num>() {
        // The audit's p_iter-5 and p_direct-5: the draft gave the diagonal √2 everywhere and the
        // subdiagonal +0.7071. A tridiagonal matrix has no fill, so IC(0) is Cholesky's factor.
        let a =
            SparseMatrix::tridiagonal(&[T::real(-1.0); 3], &[T::real(2.0); 4], &[T::real(-1.0); 3]);
        let l = Ic0::new(&a).unwrap().l();
        let diagonal = [
            2f64.sqrt(),
            1.5f64.sqrt(),
            (4.0f64 / 3.0).sqrt(),
            1.25f64.sqrt(),
        ];
        let below = [
            -(0.5f64.sqrt()),
            -((2.0f64 / 3.0).sqrt()),
            -(0.75f64.sqrt()),
        ];
        for i in 0..4 {
            assert!(magnitude(l[(i, i)] - T::real(diagonal[i])) <= 4.0 * T::EPS);
        }
        for i in 0..3 {
            assert!(magnitude(l[(i + 1, i)] - T::real(below[i])) <= 4.0 * T::EPS);
        }
        // And the dense Cholesky factor of the same matrix agrees.
        let dense = a.to_matrix().cholesky().unwrap();
        let ld = dense.l();
        for i in 0..4 {
            for j in 0..=i {
                assert!(magnitude(l[(i, j)] - ld[(i, j)]) <= 8.0 * T::EPS);
            }
        }
    }
    each_type!(check);
}

#[test]
fn ic0_of_a_full_matrix_is_its_cholesky_factor_and_solves_with_it() {
    // The audit's p_iter-5 / p_iter-6 repro: [[4, 2], [2, 5]] = L·Lᵀ with L = [[2, 0], [1, 2]];
    // the draft gave the diagonal [2, 2.236], l₁₀ = −3 and solve(4, 2) = (6.37, 1.6).
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[(0, 0, 4.0f64), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 5.0)],
    );
    let ic = Ic0::new(&a).unwrap();
    let l = ic.l();
    assert_eq!(
        (l[(0, 0)], l[(1, 0)], l[(1, 1)], l[(0, 1)]),
        (2.0, 1.0, 2.0, 0.0)
    );
    let mut z = [0.0; 2];
    ic.apply(&[4.0, 2.0], &mut z);
    assert_eq!(z, [1.0, 0.0]);
    // A complex Hermitian one: [[4, 2i], [−2i, 5]] = L·Lᴴ with L = [[2, 0], [−i, 2]].
    let i = Complex::new(0.0f64, 1.0);
    let h = SparseMatrix::from_coo(
        2,
        2,
        &[
            (0, 0, Complex::from(4.0)),
            (0, 1, i * 2.0),
            (1, 0, -i * 2.0),
            (1, 1, Complex::from(5.0)),
        ],
    );
    let l = Ic0::new(&h).unwrap().l();
    assert_eq!(
        (l[(0, 0)], l[(1, 0)], l[(1, 1)]),
        (Complex::from(2.0), -i, Complex::from(2.0))
    );
}

#[test]
fn ic0_preconditions_cg_on_the_matrices_the_audit_used() {
    // The audit's p_direct-5 / p_iter-6: with the draft's factor cg did not converge in 2000
    // iterations on the 1-D Laplacian of 200 points and in 5000 on the 2-D ones.
    let opts = SolveOptions::new().with_rtol(1e-10);
    let n = 200;
    let a = SparseMatrix::tridiagonal(&vec![-1.0f64; n - 1], &vec![2.0; n], &vec![-1.0; n - 1]);
    let b = vec![1.0; n];
    let ic = Ic0::new(&a).unwrap();
    let r = cg_with(&a, &b, &opts, Some(&ic)).unwrap();
    // The factor is exact, so one iteration solves the system.
    assert!(r.converged && r.iterations == 1);
    let small = SparseMatrix::tridiagonal(&[-1.0f64; 3], &[2.0; 4], &[-1.0; 3]);
    let r = cg_with(
        &small,
        &[1.0, 0.0, 0.0, 1.0],
        &opts,
        Some(&Ic0::new(&small).unwrap()),
    )
    .unwrap();
    assert!(r.converged && r.iterations == 1);
    // The 2-D Laplacians of 10 x 10 and 30 x 30, where the audit saw 32 and 87 iterations of
    // plain cg and no convergence in 5000 with the draft's factor.
    for k in [10, size(30, 10)] {
        let (a, b, _) = problem::<f64>(&format!("lap2d{k}"));
        let r0 = cg_with(&a, &b, &opts, None).unwrap();
        let r1 = cg_with(&a, &b, &opts, Some(&Ic0::new(&a).unwrap())).unwrap();
        assert!(r0.converged && r1.converged);
        assert!(
            r1.iterations * 2 < r0.iterations,
            "{} against {}",
            r1.iterations,
            r0.iterations
        );
    }
}

#[test]
fn ic0_is_none_for_an_indefinite_matrix_or_a_missing_diagonal_entry() {
    // The audit's p_iter-5: [[1, 2], [2, 1]] (eigenvalues 3 and −1) used to be factored.
    let a = SparseMatrix::from_coo(
        2,
        2,
        &[(0, 0, 1.0f64), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 1.0)],
    );
    assert!(Ic0::new(&a).is_none());
    assert!(Ic0::new(&SparseMatrix::from_diagonal(&[1.0f64, -1.0])).is_none());
    assert!(Ic0::new(&SparseMatrix::from_diagonal(&[1.0f64, 0.0])).is_none());
    let missing = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0f64), (1, 0, 0.5)]);
    assert!(Ic0::new(&missing).is_none());
    let nan = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0f64), (1, 0, f64::NAN), (1, 1, 1.0)]);
    assert!(Ic0::new(&nan).is_none());
    assert_eq!(
        Ic0::new(&SparseMatrix::<f64>::zeros(0, 0)).unwrap().nnz(),
        0
    );
}

#[test]
fn ic0_with_a_shift_exists_where_the_plain_one_breaks_down() {
    fn check<T: Num>() {
        // Kershaw's matrix: positive definite (eigenvalues 3 ± 2√2, each twice), and yet IC(0)
        // reaches a last pivot of −5.
        let e = |i, j, x: f64| (i, j, T::real(x));
        let a = SparseMatrix::from_coo(
            4,
            4,
            &[
                e(0, 0, 3.0),
                e(0, 1, -2.0),
                e(0, 3, 2.0),
                e(1, 0, -2.0),
                e(1, 1, 3.0),
                e(1, 2, -2.0),
                e(2, 1, -2.0),
                e(2, 2, 3.0),
                e(2, 3, -2.0),
                e(3, 0, 2.0),
                e(3, 2, -2.0),
                e(3, 3, 3.0),
            ],
        );
        assert!(a.to_matrix().cholesky().is_some());
        assert!(Ic0::new(&a).is_none());
        let alpha = T::Real::from_f64(0.5);
        let ic = Ic0::with_shift(&a, alpha).unwrap();
        // The factor of A + α·diag(A): its defining property with the diagonal enlarged.
        let shifted = a.map(|i, j, x| {
            if i == j {
                x.scale(T::Real::ONE + alpha)
            } else {
                x
            }
        });
        let ld = ic.l().to_matrix();
        let product = &ld * &ld.conjugate_transpose();
        let la = absolute(&ic.l());
        assert_on_pattern(&shifted, &product, &(&la * &la.transpose()), |i, j| j <= i);
        // A negative or not finite shift is no shift.
        assert!(Ic0::with_shift(&a, T::Real::from_f64(-0.5)).is_none());
        assert!(Ic0::with_shift(&a, T::Real::NAN).is_none());
        assert!(Ic0::with_shift(&a, T::Real::INFINITY).is_none());
        // A zero shift is the plain factorisation.
        let b =
            SparseMatrix::tridiagonal(&[T::real(-1.0); 4], &[T::real(2.0); 5], &[T::real(-1.0); 4]);
        assert_eq!(Ic0::with_shift(&b, T::Real::ZERO), Ic0::new(&b));
    }
    each_type!(check);
}

#[test]
fn ic0_reads_only_the_lower_triangle_and_the_real_part_of_the_diagonal() {
    let mut r = Rng(0x5eed_0006);
    let a = hermitian::<Complex<f64>>(&mut r, 20, 60);
    // Garbage in the strict upper triangle and in the imaginary part of the diagonal.
    let spoilt = a.map(|i, j, x| {
        if j > i {
            x * Complex::new(3.0, -7.0)
        } else if i == j {
            x + Complex::new(0.0, 1e-3)
        } else {
            x
        }
    });
    assert_eq!(Ic0::new(&a), Ic0::new(&spoilt));
}

#[test]
#[should_panic(expected = "wrong length")]
fn ic0_applied_to_a_vector_of_the_wrong_length_panics() {
    let ic = Ic0::new(&SparseMatrix::from_diagonal(&[2.0f64, 4.0])).unwrap();
    let mut z = [0.0; 1];
    ic.apply(&[1.0, 2.0], &mut z);
}

// ---------------------------------------------------------------------------------------------
// What the preconditioners do for the solvers
// ---------------------------------------------------------------------------------------------

/// The iteration counts of the four methods on `name` without and with `m`.
fn counts<T: Num>(name: &str, m: &dyn Preconditioner<T>, methods: &[&str]) -> Vec<(usize, usize)> {
    let (a, b, _) = problem::<T>(name);
    let rtol = if T::EPS < 1e-10 { 1e-10 } else { 1e-4 };
    let opts = SolveOptions::new()
        .with_rtol(T::Real::from_f64(rtol))
        .with_max_iter(20 * b.len());
    methods
        .iter()
        .map(|&method| {
            let run = |pre: Option<&dyn Preconditioner<T>>| {
                let r = match method {
                    "cg" => cg_with(&a, &b, &opts, pre),
                    "minres" => minres_with(&a, &b, &opts, pre),
                    "gmres" => gmres_with(&a, &b, &opts, pre),
                    _ => bicgstab_with(&a, &b, &opts, pre),
                }
                .unwrap();
                assert!(r.converged, "{method} on {name} ({:?})", r.reason);
                assert_eq!(r.reason, StopReason::Converged);
                r.iterations
            };
            let counts = (run(None), run(Some(m)));
            eprintln!(
                "{} {name} {method}: {} plain, {} preconditioned",
                core::any::type_name::<T>(),
                counts.0,
                counts.1
            );
            counts
        })
        .collect()
}

#[test]
fn the_preconditioners_cut_the_iterations_on_poisson_and_convection_diffusion() {
    fn check<T: Num>() {
        let real = !T::IS_COMPLEX;
        // 2-D Poisson (900 unknowns) or its magnetic counterpart: IC(0) for the Hermitian
        // methods, ILU(0) for the others. Each cuts the count by a third at least (the measured
        // counts, in the doc page, are a half to a quarter).
        let spd = match (real, cfg!(miri)) {
            (true, false) => "lap2d30",
            (true, true) => "lap2d12",
            (false, _) => "magnetic2d12",
        };
        let (a, _, _) = problem::<T>(spd);
        let ic = Ic0::new(&a).unwrap();
        let ilu = Ilu0::new(&a).unwrap();
        for (plain, with) in counts::<T>(spd, &ic, &["cg", "minres"]) {
            assert!(
                3 * with <= 2 * plain,
                "{spd}: {with} with IC(0) against {plain}"
            );
        }
        for (plain, with) in counts::<T>(spd, &ilu, &["gmres", "bicgstab"]) {
            assert!(
                3 * with <= 2 * plain,
                "{spd}: {with} with ILU(0) against {plain}"
            );
        }
        // Convection and diffusion.
        let cd = if real {
            "convdiff2d12"
        } else {
            "cconvdiff2d10"
        };
        let (a, _, _) = problem::<T>(cd);
        let ilu = Ilu0::new(&a).unwrap();
        for (plain, with) in counts::<T>(cd, &ilu, &["gmres", "bicgstab"]) {
            assert!(
                3 * with <= 2 * plain,
                "{cd}: {with} with ILU(0) against {plain}"
            );
        }
        // Rows of very different sizes, which Jacobi evens out.
        if real {
            let (a, _, _) = problem::<T>("scaled2d12");
            let jacobi = Jacobi::new(&a).unwrap();
            for (plain, with) in counts::<T>(
                "scaled2d12",
                &jacobi,
                &["cg", "minres", "gmres", "bicgstab"],
            ) {
                assert!(
                    3 * with <= 2 * plain,
                    "scaled2d12: {with} with Jacobi against {plain}"
                );
            }
        }
    }
    each_type!(check);
}
