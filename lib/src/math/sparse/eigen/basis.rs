//! What the Lanczos and the Arnoldi method share: the products of vectors, the Gram–Schmidt step
//! that keeps a basis orthonormal, the starting vectors, the dense eigensolvers of the small
//! matrices, the selection rules, and the bookkeeping and the restart decision of one run.
//!
//! A basis is held column after column in one slice, as a [`Matrix`](crate::math::matrix::Matrix)
//! holds its columns, with the length of a vector as the leading dimension.

use super::super::LinearOperator;
use crate::math::complex::{Complex, Float};
use crate::math::matrix::blas::{Op, gemm};
use crate::math::matrix::householder::norm2;
use crate::math::matrix::{Matrix, Scalar};

/// How many rows of the basis are multiplied by the small matrix of a restart at a time: the block
/// is copied out and written back, so that the basis is updated where it is.
const ROWS: usize = 256;

/// What a pass of Gram–Schmidt must leave of a vector for the vector to count as orthogonal to
/// the basis already: Daniel, Gragg, Kaufman and Stewart's `1/√2`. A pass that takes more than
/// that away was working on rounding errors, and is followed by another.
const KEPT: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// How many passes of Gram–Schmidt a vector is given before what is left of it is taken to be
/// nothing but rounding errors.
const PASSES: usize = 4;

/// The 2-norm of a vector, scaled so that its squares neither overflow nor underflow: the sum of
/// the squares in four running sums, and only where it leaves the safe range the scaled sum of
/// the dense `norm2`.
pub(super) fn norm<T: Scalar>(x: &[T]) -> T::Real {
    let mut lanes = [T::Real::ZERO; 4];
    let (chunks, chunks_rest) = x.as_chunks::<4>();
    for x4 in chunks.iter() {
        for (lane, v) in lanes.iter_mut().zip(x4) {
            *lane += v.abs2();
        }
    }
    for v in chunks_rest {
        lanes[0] += v.abs2();
    }
    let sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    // Within this range no square overflowed and none that matters underflowed.
    let low = T::Real::MIN_POSITIVE / T::Real::EPSILON;
    if sum.is_finite() && sum >= low {
        sum.sqrt()
    } else {
        norm2(x)
    }
}

/// Whether every element of the vector is finite.
pub(super) fn finite<T: Scalar>(x: &[T]) -> bool {
    x.iter().all(|v| v.is_finite())
}

/// `x / r`, by a division of every element: the reciprocal of a subnormal norm would overflow.
pub(super) fn scale_down<T: Scalar>(x: &mut [T], r: T::Real) {
    let r = T::from_real(r);
    for v in x {
        *v /= r;
    }
}

/// `ε^(2/3)` of the type: the size below which ARPACK's convergence test no longer follows a
/// Ritz value down, because nothing separates such a value from zero.
pub(super) fn eps23<R: Float>() -> R {
    R::from_f64(R::EPSILON.to_f64().powf(2.0 / 3.0))
}

/// A power of two within a factor two of the largest absolute value of `x`, one if that is zero
/// or not finite: what the small matrices of the methods are divided by before a dense
/// eigensolver sees them. The dense solvers test for negligible elements against absolute
/// thresholds near the underflow level, as LAPACK's do, and lose accuracy on a matrix whose
/// elements are all near it; a division by a power of two is exact, and so is the
/// multiplication of the eigenvalues that undoes it.
fn binary_scale<T: Scalar>(x: &[T]) -> T::Real {
    let largest = x.iter().fold(T::Real::ZERO, |m, v| m.max(v.abs()));
    if largest == T::Real::ZERO || !largest.is_finite() {
        return T::Real::ONE;
    }
    let exponent = largest.to_f64().log2().floor();
    T::Real::from_f64(exponent.exp2())
}

/// The eigenvalues and eigenvectors of a square matrix of the methods: the values, and the
/// vectors as the columns of a matrix.
pub(super) type Eigenpairs<V, M> = (Vec<V>, Matrix<M>);

/// [`Eigenpairs`] of a general matrix, complex values and vectors over the real type `R`.
pub(super) type ComplexEigenpairs<R> = Eigenpairs<Complex<R>, Complex<R>>;

/// The eigenvalues and the eigenvectors (columns of length 1) of the square matrix `h` of order
/// `m`, column after column, by the dense eigensolver (the Schur form, then back substitution),
/// divided by [`binary_scale`] first. `None` if a value is not finite.
pub(super) fn general_eigen<T: Scalar>(h: &[T], m: usize) -> Option<ComplexEigenpairs<T::Real>> {
    let p = binary_scale(h);
    let scaled: Vec<T> = h.iter().map(|&x| x / T::from_real(p)).collect();
    let e = Matrix::from_column_major(m, m, scaled).eigen();
    let values: Vec<Complex<T::Real>> = e
        .values()
        .iter()
        .map(|z| Complex::new(z.re * p, z.im * p))
        .collect();
    let finite = values.iter().all(|z| z.re.is_finite() && z.im.is_finite())
        && e.vectors()
            .as_slice()
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite());
    finite.then(|| (values, e.vectors().clone()))
}

/// The eigenvalues, ascending, and the orthonormal eigenvectors of the Hermitian matrix `h` of
/// order `m`, column after column, by the dense Hermitian eigensolver, divided by
/// [`binary_scale`] first. `None` if a value is not finite.
pub(super) fn hermitian_eigen<T: Scalar>(h: &[T], m: usize) -> Option<Eigenpairs<T::Real, T>> {
    let p = binary_scale(h);
    let scaled: Vec<T> = h.iter().map(|&x| x / T::from_real(p)).collect();
    let e = Matrix::from_column_major(m, m, scaled).hermitian_eigen();
    let values: Vec<T::Real> = e.values().iter().map(|&x| x * p).collect();
    let finite = values.iter().all(|x| x.is_finite())
        && e.vectors().as_slice().iter().all(|x| x.is_finite());
    finite.then(|| (values, e.vectors().clone()))
}

/// How many times larger in absolute value than the open values asked for a converged Ritz value
/// of shift-and-invert must be to be locked at once, its run started again in the complement
/// (see [`Step::Lock`]). Kept in the factorisation, it spoils the others, relative to their own
/// size, by about `ε` times the ratio, through the rotations of every restart; past a ratio of a
/// few, that is more than the room for rounding of the test with `A` allows in single precision.
const LOCKING: f64 = 8.0;

/// A small deterministic generator, Marsaglia's xorshift on 64 bits, for the starting vectors:
/// a run is repeatable, and nothing is asked of the system.
pub(super) struct Random(u64);

impl Random {
    /// The generator in its fixed first state.
    pub(super) fn new() -> Self {
        Random(0x9E37_79B9_7F4A_7C15)
    }

    /// The generator in another fixed state, for vectors that must not disturb the sequence of
    /// the starting vectors.
    pub(super) fn other() -> Self {
        Random(0x2545_F491_4F6C_DD1D)
    }

    /// A float uniform in `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// A vector with parts uniform in `[-1, 1)`: one with a component along every direction, as
    /// far as chance allows, which a vector of a simple pattern (all ones, a unit vector) is not.
    pub(super) fn vector<T: Scalar>(&mut self, n: usize) -> Vec<T> {
        (0..n)
            .map(|_| {
                let re = T::Real::from_f64(self.unit());
                let im = if T::IS_COMPLEX {
                    T::Real::from_f64(self.unit())
                } else {
                    T::Real::ZERO
                };
                T::from_parts(re, im)
            })
            .collect()
    }
}

/// The rows a pass of [`project`] reads at a time, so that the piece of `w` stays in the
/// first-level cache while every column streams by once.
// Measured on an Apple M4 Max (`lanczos_with` and `arnoldi_with`, two restarts, on Laplacians of
// 10⁴ and 10⁵ unknowns and a random matrix of 10⁵, bases of 20 and 40; sweep 512 to 8192): mixed,
// so kept: with a basis of 40, 2048 to 8192 rows take 8 to 10 % off; with 20, every larger block
// costs 1 to 3 %.
const BLOCK: usize = 512;

/// Takes out of `w` its components along the `count` orthonormal columns of `columns` (of the
/// length of `w`): `c = Vᴴ·w`, then `w −= V·c`, two passes over the columns, a block of rows of
/// `w` at a time. Each coefficient is summed in four running sums (the row `i` to the sum
/// `i mod 4`), the same order every time.
fn project<T: Scalar>(columns: &[T], count: usize, w: &mut [T], c: &mut [T]) {
    let n = w.len();
    debug_assert!(columns.len() >= count * n && c.len() >= count);
    let mut lanes = vec![[T::zero(); 4]; count];
    let mut start = 0;
    while start < n {
        // The blocks begin at multiples of 4: a row's sum is its index mod 4.
        let end = (start + BLOCK).min(n);
        let wb = &w[start..end];
        let column = |k: usize| &columns[k * n + start..k * n + end];
        // Four columns at a time, each piece of `w` read once for the four.
        let mut k = 0;
        while k + 4 <= count {
            let m = wb.len();
            let (v0, v1, v2, v3) = (
                &column(k)[..m],
                &column(k + 1)[..m],
                &column(k + 2)[..m],
                &column(k + 3)[..m],
            );
            let mut l = [lanes[k], lanes[k + 1], lanes[k + 2], lanes[k + 3]];
            let whole = wb.len() / 4 * 4;
            for i in (0..whole).step_by(4) {
                for q in 0..4 {
                    let x = wb[i + q];
                    l[0][q] += v0[i + q].conj() * x;
                    l[1][q] += v1[i + q].conj() * x;
                    l[2][q] += v2[i + q].conj() * x;
                    l[3][q] += v3[i + q].conj() * x;
                }
            }
            for i in whole..wb.len() {
                let x = wb[i];
                l[0][0] += v0[i].conj() * x;
                l[1][0] += v1[i].conj() * x;
                l[2][0] += v2[i].conj() * x;
                l[3][0] += v3[i].conj() * x;
            }
            lanes[k..k + 4].copy_from_slice(&l);
            k += 4;
        }
        for (k, lane) in lanes.iter_mut().enumerate().skip(k) {
            let v = column(k);
            let (vc, vc_rest) = v.as_chunks::<4>();
            let (wc, wc_rest) = wb.as_chunks::<4>();
            let mut l = *lane;
            for (v4, w4) in vc.iter().zip(wc.iter()) {
                for q in 0..4 {
                    l[q] += v4[q].conj() * w4[q];
                }
            }
            for (&v, &x) in vc_rest.iter().zip(wc_rest) {
                l[0] += v.conj() * x;
            }
            *lane = l;
        }
        start = end;
    }
    for (c, l) in c[..count].iter_mut().zip(&lanes) {
        *c = (l[0] + l[1]) + (l[2] + l[3]);
    }
    let mut start = 0;
    while start < n {
        let end = (start + BLOCK).min(n);
        let wb = &mut w[start..end];
        let column = |k: usize| &columns[k * n + start..k * n + end];
        // Four columns at a time, the element of `w` held while the four are taken out of it in
        // their order: the same arithmetic as one column after another.
        let mut k = 0;
        while k + 4 <= count {
            let m = wb.len();
            let (v0, v1, v2, v3) = (
                &column(k)[..m],
                &column(k + 1)[..m],
                &column(k + 2)[..m],
                &column(k + 3)[..m],
            );
            let (c0, c1, c2, c3) = (c[k], c[k + 1], c[k + 2], c[k + 3]);
            for (i, w) in wb.iter_mut().enumerate() {
                *w = w
                    .mul_sub(c0, v0[i])
                    .mul_sub(c1, v1[i])
                    .mul_sub(c2, v2[i])
                    .mul_sub(c3, v3[i]);
            }
            k += 4;
        }
        for (k, &ck) in c[..count].iter().enumerate().skip(k) {
            for (w, &v) in wb.iter_mut().zip(column(k)) {
                *w = w.mul_sub(ck, v);
            }
        }
        start = end;
    }
}

/// Takes out of `w` its components along the orthonormal columns of every slice of `locked` and
/// of `basis`, by classical Gram–Schmidt applied twice — the coefficients of a pass all come
/// from the same `w`, and the second pass removes what the rounding of the first left behind.
/// The coefficients along `basis`, summed over the passes, go into `coefficients`; those along
/// `locked` are dropped, which is what restricts the operator to the complement of the locked
/// vectors.
///
/// Returns the norm of what is left. When a pass after the first still takes away more than a
/// fraction `1 − 1/√2` of the vector, the vector had been reduced to rounding errors, and a
/// further pass is made; if those go on shrinking, `w` lies in the span of the columns as far as
/// the arithmetic can tell, it is set to zero and zero is returned.
pub(super) fn orthogonalise<T: Scalar>(
    locked: &[&[T]],
    basis: &[T],
    w: &mut [T],
    coefficients: &mut [T],
) -> T::Real {
    let n = w.len();
    coefficients.fill(T::zero());
    if n == 0 {
        return T::Real::ZERO;
    }
    let mut before = norm(w);
    if before == T::Real::ZERO {
        return T::Real::ZERO;
    }
    let kept = T::Real::from_f64(KEPT);
    let b = basis.len() / n;
    let widest = locked.iter().map(|l| l.len() / n).fold(b, usize::max);
    let mut c = vec![T::zero(); widest];
    for pass in 0..PASSES {
        for part in locked {
            let l = part.len() / n;
            if l > 0 {
                project(part, l, w, &mut c[..l]);
            }
        }
        if b > 0 {
            project(basis, b, w, &mut c[..b]);
            for (sum, c) in coefficients.iter_mut().zip(&c[..b]) {
                *sum += *c;
            }
        }
        let after = norm(w);
        if after == T::Real::ZERO || !after.is_finite() {
            break;
        }
        if pass > 0 && after >= kept * before {
            return after;
        }
        before = after;
    }
    if !finite(w) {
        return T::Real::NAN;
    }
    w.fill(T::zero());
    T::Real::ZERO
}

/// A unit vector orthogonal to the columns of `locked` and of `basis`, from the generator: what
/// a factorisation goes on with when its own next vector has vanished, and what a run starts
/// from. `None` if no such vector is found, which only happens when the columns span the space.
pub(super) fn fresh<T: Scalar>(
    random: &mut Random,
    n: usize,
    locked: &[&[T]],
    basis: &[T],
) -> Option<Vec<T>> {
    for _ in 0..8 {
        let mut w: Vec<T> = random.vector(n);
        if unit(locked, basis, &mut w) {
            return Some(w);
        }
    }
    None
}

/// Makes `w` a unit vector orthogonal to the columns of `locked` and of `basis`. False if
/// nothing of it is left.
pub(super) fn unit<T: Scalar>(locked: &[&[T]], basis: &[T], w: &mut [T]) -> bool {
    let n = w.len();
    let mut ignored = vec![T::zero(); basis.len() / n.max(1)];
    let length = orthogonalise(locked, basis, w, &mut ignored);
    if length == T::Real::ZERO || !length.is_finite() {
        return false;
    }
    scale_down(w, length);
    // Dividing by a tiny norm magnifies the rounding of the passes: once more, on a unit vector.
    let length = orthogonalise(locked, basis, w, &mut ignored);
    if length == T::Real::ZERO || !length.is_finite() {
        return false;
    }
    scale_down(w, length);
    true
}

/// Appends to `out` the columns of `block` (of length `n`), made orthonormal to working
/// precision: each made by [`unit`] a unit vector orthogonal to the columns of `locked`, to those
/// of `out` and to the columns of the block before it. False, with `out` as it was, if nothing is
/// left of a column.
///
/// Everything a run passes on from its factorisation goes through this: a block it locks, and
/// the Ritz vectors `V·s` it ends with. The columns of `V` are orthonormal to working precision
/// when they are made, but every restart replaces them by `V·Q`, and the rounding of the
/// product and of `Q` moves their lengths and angles by an `ε` or so each time, a drift that
/// nothing undoes. Passed on as they were, they made the vectors found drift with them, and
/// the eigenvectors of the answer: on the 1-D Laplacian of 400 points Lanczos's came back 45
/// to 92ε from length 1 after 100 to 200 restarts, and after 290 restarts on a rotation matrix
/// of the tests a locked block held a column of length `1 + 155ε` on x86_64 (`1 + 20ε` on
/// aarch64, whose dense kernels round differently), and Arnoldi returned an eigenvector `150ε`
/// from length 1. The callers take the vectors found to be orthonormal: the operator is
/// restricted to their complement by products with `Zᴴ`, and the answer is the Rayleigh–Ritz
/// step on `Zᴴ·A·Z`, whose eigenvectors `s` of length 1 give vectors `Z·s` of length 1 only for
/// an orthonormal `Z`. Gram–Schmidt moves the columns by no more than they had drifted.
pub(super) fn append_orthonormal<T: Scalar>(
    n: usize,
    locked: &[T],
    block: &[T],
    out: &mut Vec<T>,
) -> bool {
    let start = out.len();
    for column in block.chunks_exact(n) {
        let mut w = column.to_vec();
        if !unit(&[locked], out, &mut w) {
            out.truncate(start);
            return false;
        }
        out.extend_from_slice(&w);
    }
    true
}

/// The residual of the columns a restart keeps, `f⁺ = v⁺·h⁺ + f·σ` (`σ` here times `‖f‖`, as a run
/// holds the direction of its residual), into `f`, divided by `s`, the larger of `|h⁺|` and
/// `|σ|`, which is returned: what `f⁺` must be multiplied by. Formed without the division, the
/// residual of a run that keeps restarting about pairs that have converged on the operator but
/// do not pass its test — shift-and-invert of a matrix that is not normal, near an eigenvalue —
/// shrinks by a factor at every restart, down to subnormal numbers, whose direction has a few
/// bits; normalised, it made a basis that was not orthonormal, and the factorisation went to
/// pieces.
pub(super) fn restart_residual<T: Scalar>(f: &mut [T], next: &[T], below: T, sigma: T) -> T::Real {
    let s = below.abs().max(sigma.abs());
    if s > T::Real::ZERO && s.is_finite() {
        let (a, b) = (below / T::from_real(s), sigma / T::from_real(s));
        for (f, v) in f.iter_mut().zip(next) {
            *f = *v * a + *f * b;
        }
    } else {
        f.fill(T::zero());
    }
    s
}

/// Replaces the first `columns` columns of the basis `v`, of `m` columns of length `n`, by
/// `v·q[.., ..columns]` for the `m × m` matrix `q`: the change of basis of an implicit restart.
/// Rows are taken a block at a time, so that the product is one of blocks and the basis needs no
/// copy of itself.
pub(super) fn rotate<T: Scalar>(n: usize, m: usize, v: &mut [T], q: &[T], columns: usize) {
    if n == 0 || columns == 0 {
        return;
    }
    let mut block = vec![T::zero(); ROWS.min(n) * columns];
    let mut first = 0;
    while first < n {
        let rows = ROWS.min(n - first);
        gemm(
            rows,
            columns,
            m,
            T::one(),
            &v[first..],
            n,
            Op::N,
            q,
            m,
            Op::N,
            T::zero(),
            &mut block,
            rows,
        );
        for j in 0..columns {
            v[first + j * n..first + j * n + rows]
                .copy_from_slice(&block[j * rows..(j + 1) * rows]);
        }
        first += rows;
    }
}

/// `v·s` for the basis `v` of `m` columns of length `n` and the `m × columns` matrix `s`: the
/// vectors of the space the small problem picked.
pub(super) fn combine<T: Scalar>(n: usize, m: usize, v: &[T], s: &[T], columns: usize) -> Vec<T> {
    let mut x = vec![T::zero(); n * columns];
    if n > 0 && columns > 0 && m > 0 {
        gemm(
            n,
            columns,
            m,
            T::one(),
            v,
            n,
            Op::N,
            s,
            m,
            Op::N,
            T::zero(),
            &mut x,
            n,
        );
    }
    x
}

/// `v·y` in complex arithmetic for the basis `v` of `m` columns of length `n` and the complex
/// coefficients `y`: for real elements, the products with the real and the imaginary part.
pub(super) fn combine_complex<T: Scalar>(
    n: usize,
    m: usize,
    v: &[T],
    y: &[Complex<T::Real>],
) -> Vec<Complex<T::Real>> {
    if T::IS_COMPLEX {
        let s: Vec<T> = y.iter().map(|z| T::from_parts(z.re, z.im)).collect();
        combine(n, m, v, &s, 1)
            .iter()
            .map(|x| x.to_complex())
            .collect()
    } else {
        let re: Vec<T> = y.iter().map(|z| T::from_real(z.re)).collect();
        let im: Vec<T> = y.iter().map(|z| T::from_real(z.im)).collect();
        let (xr, xi) = (combine(n, m, v, &re, 1), combine(n, m, v, &im, 1));
        xr.iter()
            .zip(&xi)
            .map(|(a, b)| Complex::new(a.re(), b.re()))
            .collect()
    }
}

/// `A·x` for a complex vector `x` and an operator of the element type: for real elements, the
/// products with the real and the imaginary part.
pub(super) fn apply_complex<T: Scalar>(
    op: &dyn LinearOperator<T>,
    x: &[Complex<T::Real>],
) -> Vec<Complex<T::Real>> {
    let n = x.len();
    if T::IS_COMPLEX {
        let xt: Vec<T> = x.iter().map(|z| T::from_parts(z.re, z.im)).collect();
        let mut y = vec![T::zero(); n];
        op.apply(&xt, &mut y);
        y.iter().map(|v| v.to_complex()).collect()
    } else {
        let re: Vec<T> = x.iter().map(|z| T::from_real(z.re)).collect();
        let (mut yr, mut yi) = (vec![T::zero(); n], vec![T::zero(); n]);
        op.apply(&re, &mut yr);
        if x.iter().any(|z| z.im != T::Real::ZERO) {
            let im: Vec<T> = x.iter().map(|z| T::from_real(z.im)).collect();
            op.apply(&im, &mut yi);
        }
        yr.iter()
            .zip(&yi)
            .map(|(a, b)| Complex::new(a.re(), b.re()))
            .collect()
    }
}

/// The 2-norm of a complex vector, scaled so that its squares neither overflow nor underflow.
pub(super) fn complex_norm<R: Float>(x: &[Complex<R>]) -> R {
    let scale = x
        .iter()
        .fold(R::ZERO, |m, z| m.max(z.re.abs()).max(z.im.abs()));
    if scale == R::ZERO || !scale.is_finite() {
        return scale;
    }
    let sum = x.iter().fold(R::ZERO, |s, z| {
        let (a, b) = (z.re / scale, z.im / scale);
        s + a * a + b * b
    });
    sum.sqrt() * scale
}

/// For a unit vector `x` orthogonal to the columns of every slice of `locked`, the Rayleigh
/// quotient `λ = xᴴ·A·x` and the residual `‖P·(A·x − λ·x)‖` of the operator restricted to their
/// complement, `P` the projection onto it, from one product with `a` (two for a real operator
/// and a vector that is not real). A run that works in the complement of vectors found before
/// converges to the eigenvectors of `P·A·P`, which for an `A` that is not normal are not those
/// of `A` but its Schur vectors in the complement — together with the locked vectors they span
/// the invariant subspace, and the Rayleigh–Ritz step at the end turns them into eigenvectors;
/// for a Hermitian `A` the two residuals agree. `None` if a value is not finite.
pub(super) fn residual_with<T: Scalar>(
    a: &dyn LinearOperator<T>,
    x: &[Complex<T::Real>],
    locked: &[&[T]],
) -> Option<(Complex<T::Real>, T::Real)> {
    let n = x.len();
    let y = apply_complex(a, x);
    let zero = Complex::new(T::Real::ZERO, T::Real::ZERO);
    let lambda = x.iter().zip(&y).fold(zero, |s, (u, v)| s + u.conj() * *v);
    let mut d: Vec<Complex<T::Real>> = y.iter().zip(x).map(|(&y, &v)| y - lambda * v).collect();
    for part in locked {
        for z in part.chunks_exact(n.max(1)) {
            let c = z
                .iter()
                .zip(&d)
                .fold(zero, |s, (u, v)| s + u.to_complex().conj() * *v);
            for (di, zi) in d.iter_mut().zip(z) {
                *di -= zi.to_complex() * c;
            }
        }
    }
    let residual = complex_norm(&d);
    (lambda.re.is_finite() && lambda.im.is_finite() && residual.is_finite())
        .then_some((lambda, residual))
}

impl<T: Scalar> Run<'_, T> {
    /// Whether a block of the factorisation that split off at an exact zero may be locked: its
    /// columns `v` (of length `n`), and the coefficients of its Ritz vectors in them, one column
    /// of `coefficients` a pair. Without shift-and-invert, always: the block spans an invariant
    /// subspace of the operator, and its Ritz pairs are exact as far as the arithmetic can tell.
    /// With it, only when every pair of the block passes the test with `A`, as any pair must to
    /// converge: the block is invariant for the inverse as computed, whose products near an
    /// eigenvalue of an `A` that is not normal are accurate only to about `ε·‖A‖/δ` in the
    /// directions of the other eigenvectors, and a block that split off at that level of
    /// rounding — a restart residual or a new vector that Gram–Schmidt found to be rounding
    /// alone — is no invariant subspace of `A` to the tolerance. Locked unjudged, such blocks
    /// spoiled the complement every later run worked in: they gave pairs with residuals of the
    /// order of `‖A‖` and a final answer of nothing. A block refused stays in
    /// the factorisation, where its pairs are judged at every decision like the others.
    pub(super) fn may_lock(
        &self,
        deflated: &[T],
        v: &[T],
        coefficients: &[Vec<Complex<T::Real>>],
    ) -> bool {
        let Some(original) = self.original else {
            return true;
        };
        let n = self.n;
        let width = v.len() / n.max(1);
        coefficients.iter().all(|y| {
            let x = combine_complex(n, width, v, y);
            residual_with(original.operator, &x, &[self.locked, deflated]).is_some_and(
                |(lambda, r)| accepts_with(self.tolerance, original.norm, n, lambda.abs(), r),
            )
        })
    }
}

/// How a selection rule ranks an eigenvalue: by a score, the larger the more wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Score {
    /// The real part.
    Real,
    /// The real part, negated: the leftmost first.
    NegReal,
    /// The absolute value.
    Magnitude,
    /// The imaginary part.
    Imaginary,
    /// The imaginary part, negated.
    NegImaginary,
    /// The absolute value of the imaginary part: for a real operator, whose eigenvalues that are
    /// not real come in conjugate pairs, the two of a pair the same.
    AbsImaginary,
    /// The absolute value of the imaginary part, negated: the nearest to the real axis first.
    NegAbsImaginary,
}

impl Score {
    /// The score of `z`.
    pub(super) fn of<R: Float>(self, z: Complex<R>) -> R {
        match self {
            Score::Real => z.re,
            Score::NegReal => -z.re,
            Score::Magnitude => z.re.hypot(z.im),
            Score::Imaginary => z.im,
            Score::NegImaginary => -z.im,
            Score::AbsImaginary => z.im.abs(),
            Score::NegAbsImaginary => -z.im.abs(),
        }
    }

    /// A number whose score is `t`: what stands for a threshold where a value is needed.
    pub(super) fn point<R: Float>(self, t: R) -> Complex<R> {
        match self {
            Score::Real | Score::Magnitude => Complex::new(t, R::ZERO),
            Score::NegReal => Complex::new(-t, R::ZERO),
            Score::Imaginary | Score::AbsImaginary | Score::NegAbsImaginary => {
                Complex::new(R::ZERO, t)
            }
            Score::NegImaginary => Complex::new(R::ZERO, -t),
        }
    }
}

/// A selection rule: the most wanted by one score, or a share from each end of the real line —
/// half of them from the left, the other half, one more when the count is odd, from the right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// The largest scores.
    One(Score),
    /// Both ends of the real line.
    BothEnds,
}

/// The indices of `values` from the most wanted by `score` to the least, ties in the order of
/// the indices, in units: a value, or with `pairs` a value and its exact conjugate, which a real
/// operator's Ritz values come in, the one with the positive imaginary part first.
fn ranked<R: Float>(score: Score, values: &[Complex<R>], pairs: bool) -> Vec<Vec<usize>> {
    let m = values.len();
    let mut units: Vec<Vec<usize>> = Vec::with_capacity(m);
    let mut taken = vec![false; m];
    for i in 0..m {
        if taken[i] {
            continue;
        }
        taken[i] = true;
        let z = values[i];
        let partner = if pairs && z.im != R::ZERO {
            (i + 1..m).find(|&j| !taken[j] && values[j].re == z.re && values[j].im == -z.im)
        } else {
            None
        };
        match partner {
            Some(j) => {
                taken[j] = true;
                units.push(if z.im > R::ZERO {
                    vec![i, j]
                } else {
                    vec![j, i]
                });
            }
            None => units.push(vec![i]),
        }
    }
    units.sort_by(|a, b| {
        let (x, y) = (score.of(values[a[0]]), score.of(values[b[0]]));
        y.partial_cmp(&x).unwrap_or(core::cmp::Ordering::Equal)
    });
    units
}

/// The `count` values the rule wants most, as the parts of the selection, each a list of units
/// from its end inward: one part for a rule by one score; for [`Rule::BothEnds`] the left part,
/// from the leftmost, and the right part, from the rightmost. With `pairs` a unit is a
/// conjugate pair where there is one, which is never split: when the count would take one of
/// it, it takes both, and one more value than asked for is returned.
pub(super) fn parts<R: Float>(
    rule: Rule,
    values: &[Complex<R>],
    count: usize,
    pairs: bool,
) -> Vec<Vec<Vec<usize>>> {
    let count = count.min(values.len());
    match rule {
        Rule::One(score) => {
            let mut part = Vec::new();
            let mut taken = 0;
            for unit in ranked(score, values, pairs) {
                if taken >= count {
                    break;
                }
                taken += unit.len();
                part.push(unit);
            }
            vec![part]
        }
        Rule::BothEnds => {
            let low = count / 2;
            let ascending: Vec<usize> = ranked(Score::NegReal, values, false)
                .into_iter()
                .flatten()
                .collect();
            let high = count - low;
            let left = ascending[..low].iter().map(|&i| vec![i]).collect();
            let right = ascending
                .iter()
                .rev()
                .take(high)
                .map(|&i| vec![i])
                .collect();
            vec![left, right]
        }
    }
}

/// The indices of the `count` values the rule wants most, the most wanted first (for
/// [`Rule::BothEnds`], the left end's from the leftmost, then the right end's from the
/// rightmost); the units of [`parts`] one after the other.
pub(super) fn pick<R: Float>(
    rule: Rule,
    values: &[Complex<R>],
    count: usize,
    pairs: bool,
) -> Vec<usize> {
    parts(rule, values, count, pairs)
        .into_iter()
        .flatten()
        .flatten()
        .collect()
}

/// The `count` indices from `first` on that the rule wants most, judged among all of `values`,
/// as a set: a restart keeps them. By one score, the highest scores; for both ends, the values
/// nearest an end of the whole spectrum, the right end first on a tie, which is how
/// [`Rule::BothEnds`] shares out its count. A conjugate pair is kept as a whole.
fn kept<R: Float>(
    rule: Rule,
    values: &[Complex<R>],
    first: usize,
    count: usize,
    pairs: bool,
) -> Vec<bool> {
    let m = values.len();
    let mut keep = vec![false; m];
    let order: Vec<Vec<usize>> = match rule {
        Rule::One(score) => ranked(score, values, pairs)
            .into_iter()
            .filter(|unit| unit[0] >= first)
            .collect(),
        Rule::BothEnds => {
            let ascending: Vec<usize> = ranked(Score::NegReal, values, false)
                .into_iter()
                .flatten()
                .collect();
            let mut depth = vec![0usize; m];
            for (p, &i) in ascending.iter().enumerate() {
                depth[i] = (2 * p + 1).min(2 * (m - 1 - p));
            }
            let mut order: Vec<usize> = (first..m).collect();
            order.sort_by_key(|&i| depth[i]);
            order.into_iter().map(|i| vec![i]).collect()
        }
    };
    let mut taken = 0;
    for unit in order {
        if taken >= count {
            break;
        }
        taken += unit.len();
        for i in unit {
            keep[i] = true;
        }
    }
    keep
}

/// Why a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stop {
    /// Every Ritz pair it was asked for passed the convergence test.
    Converged,
    /// The judge it was given said that it had seen enough.
    Settled,
    /// It ran out of restarts.
    Exhausted,
    /// The operator, or the small eigenproblem, gave a value that is not finite.
    Failed,
}

/// What a run found: orthonormal vectors of length `n`, a basis of an invariant subspace — the
/// blocks the run locked and the Ritz vectors of the most wanted pairs that converged — the
/// eigenvalues of the operator on it, and the Ritz values asked for that had not converged.
pub(super) struct Outcome<T: Scalar> {
    /// The vectors, column after column, as many as there are values.
    pub(super) vectors: Vec<T>,
    /// The eigenvalues of the operator on their span: those of the locked blocks, wanted or not,
    /// and the converged Ritz values that every more wanted one asked for had passed before.
    pub(super) values: Vec<Complex<T::Real>>,
    /// Why the run ended.
    pub(super) stop: Stop,
    /// When the run ended before all it was asked for converged, the Ritz values asked for that
    /// had not: no value less wanted than one of them is known to be an eigenvalue as wanted as
    /// it seems, since an eigenvalue could lie between.
    pub(super) frontier: Vec<Complex<T::Real>>,
}

impl<T: Scalar> Outcome<T> {
    /// A run that ended without anything to show.
    pub(super) fn empty(stop: Stop) -> Self {
        Outcome {
            vectors: Vec::new(),
            values: Vec::new(),
            stop,
            frontier: Vec::new(),
        }
    }
}

/// A Ritz value of a factorisation as the judge of a run is shown it: the value and the
/// estimate of its residual.
pub(super) type Candidate<R> = (Complex<R>, R);

/// Whether the candidate of an index has converged, worked out when it is asked for.
pub(super) type Verdict<'a> = &'a mut dyn FnMut(usize) -> bool;

/// The judge of a run: shown every Ritz value of a factorisation, a way to learn whether one has
/// converged, and the size of the operator so far, it says whether the run can stop.
pub(super) type Judge<'a, R> = &'a dyn Fn(&[Candidate<R>], Verdict<'_>, R) -> bool;

/// The counters a call shares between its runs.
pub(super) struct State<R> {
    /// The restarts still allowed, counted down.
    pub(super) restarts_left: usize,
    /// The restarts made, counted up.
    pub(super) restarts: usize,
    /// The applications of the operator, counted up.
    pub(super) applications: usize,
    /// The largest `‖A·v‖` met for a unit `v`, and the largest absolute Ritz value: a lower
    /// bound of the norm of the operator, which the tests are scaled by.
    pub(super) norm: R,
    /// The generator of the starting vectors and of the vectors a broken-down factorisation
    /// goes on with.
    pub(super) random: Random,
}

/// What a factorisation's Ritz values decide.
pub(super) enum Step {
    /// The run stops: the indices of the most wanted values asked for that converged (every
    /// more wanted one asked for converged too), why, and the indices of those asked for that
    /// had not converged when it was not every one.
    Stop(Vec<usize>, Stop, Vec<usize>),
    /// The run restarts: the indices of the Ritz values to apply as shifts, those with the
    /// largest estimates first, and the number of columns of the factorisation the restart
    /// keeps, all but one for each shift; no shift when the factorisation is only to grow.
    Restart(Vec<usize>, usize),
    /// The run locks the Ritz vectors of the values of the first list and starts again, in
    /// their complement, from those of the values asked for that are still open, the second
    /// list: an explicit restart, for two cases. For shift-and-invert, values that have
    /// converged and are [`LOCKING`] times or more larger than the open ones: kept in the
    /// factorisation, a value of that size would spoil the others, the rotations of every
    /// restart mixing it into them with errors of the order of `ε` times itself. And values
    /// with an estimate of exactly zero, when nothing else is left to shift.
    Lock(Vec<usize>, Vec<usize>),
}

/// The operator `A` of a shift-and-invert call, whose eigenpairs the Ritz pairs of the operator
/// `(A − σI)⁻¹` of the call are taken to be, and an estimate of `‖A‖`.
pub(super) struct Original<'a, T: Scalar> {
    /// `A`.
    pub(super) operator: &'a dyn LinearOperator<T>,
    /// `‖A‖`, or a lower bound of it within a small factor.
    pub(super) norm: T::Real,
}

/// What one run of either method is given: the operator, the vectors already found, which it
/// stays orthogonal to, the sizes, the rule, the tolerance, and the counters it shares with the
/// runs before and after it.
pub(super) struct Run<'a, T: Scalar> {
    /// The operator the Krylov space is built with.
    pub(super) operator: &'a dyn LinearOperator<T>,
    /// The length of the vectors.
    pub(super) n: usize,
    /// The vectors found by earlier runs, orthonormal, column after column.
    pub(super) locked: &'a [T],
    /// The size of the basis, `m`.
    pub(super) basis: usize,
    /// How many Ritz pairs are asked for.
    pub(super) want: usize,
    /// Which.
    pub(super) rule: Rule,
    /// Whether the operator is real and its Ritz values that are not real come in conjugate
    /// pairs, which are kept together.
    pub(super) pairs: bool,
    /// The tolerance of the convergence test.
    pub(super) tolerance: T::Real,
    /// The counters.
    pub(super) state: &'a mut State<T::Real>,
    /// The judge, if the run has one.
    pub(super) judge: Option<Judge<'a, T::Real>>,
    /// For shift-and-invert, `A`: a Ritz pair has converged when its vector is an eigenvector of
    /// `A` to the tolerance, by its residual with `A`.
    pub(super) original: Option<&'a Original<'a, T>>,
}

/// The convergence verdicts of the values of one decision, each worked out once, when it is
/// first asked for: the locked values have converged; the others by the estimate of their
/// residual, or, for shift-and-invert, by `check`, their true residual with `A`.
struct Verdicts<'v, 'c, R> {
    /// The verdicts worked out so far.
    cache: Vec<Option<bool>>,
    /// The index of the first value that is not locked.
    first: usize,
    /// The values.
    values: &'v [Complex<R>],
    /// Their estimates.
    bounds: &'v [R],
    /// The tolerance.
    tolerance: R,
    /// The size of the operator.
    size: R,
    /// The verdict with `A`, for shift-and-invert.
    check: Option<Verdict<'c>>,
}

impl<R: Float> Verdicts<'_, '_, R> {
    /// Whether the value `i` has converged.
    fn get(&mut self, i: usize) -> bool {
        if let Some(v) = self.cache[i] {
            return v;
        }
        let v = i < self.first
            || match self.check.as_mut() {
                Some(check) => check(i),
                None => {
                    let z = self.values[i];
                    accepts(self.tolerance, self.size, z.re.hypot(z.im), self.bounds[i])
                }
            };
        self.cache[i] = Some(v);
        v
    }
}

impl<T: Scalar> Run<'_, T> {
    /// Applies the operator to `x`, into `y`, and counts it. False if the image is not finite.
    pub(super) fn apply(&mut self, x: &[T], y: &mut [T]) -> bool {
        self.operator.apply(x, y);
        self.state.applications += 1;
        // The norm is finite exactly when every element is: a NaN makes it NaN, an infinity
        // infinite, and a square that overflows is caught by the scaled sum.
        let size = norm(y);
        if !size.is_finite() {
            return false;
        }
        if size > self.state.norm {
            self.state.norm = size;
        }
        true
    }

    /// Takes in the eigenvalues of the blocks the run has locked, `values[..first]`, and the
    /// Ritz values of its factorisation, `values[first..]`, with their estimates, and decides, as
    /// ARPACK's `dsaup2` and `dnaup2` do, whether the run stops or restarts, and with which
    /// shifts. `room` says whether the factorisation may grow without a restart (it is shorter
    /// than the basis, a block having been locked); `check`, for shift-and-invert, whether a
    /// Ritz value has converged.
    ///
    /// The run stops when every value asked for has converged, when the judge says so, or when
    /// no restart is left; it then reports, at each end of the selection, the values asked for
    /// from the most wanted inward up to the first that has not converged, and those that have
    /// not. Otherwise it restarts with shifts from its factorisation alone — a locked value is
    /// never one — and keeps more than it was asked for: `want + min(converged, shifts / 2)`
    /// of the factorisation's own, so that the shifts are fewer and none of them falls on a Ritz
    /// value about to be wanted, which is what makes a restarted run stagnate (Lehoucq and
    /// Sorensen's safeguard). The shifts are applied those with the largest estimates first,
    /// which is the order that disturbs the others least. A factorisation too short to spare a
    /// shift grows instead.
    pub(super) fn decide(
        &mut self,
        values: &[Complex<T::Real>],
        bounds: &[T::Real],
        first: usize,
        room: bool,
        check: Option<Verdict<'_>>,
    ) -> Step {
        let m = values.len();
        for z in values {
            let size = z.re.hypot(z.im);
            if size > self.state.norm {
                self.state.norm = size;
            }
        }
        let mut verdicts = Verdicts {
            cache: vec![None; m],
            first,
            values,
            bounds,
            tolerance: self.tolerance,
            size: self.state.norm,
            check,
        };
        let parts = parts(self.rule, values, self.want, self.pairs);
        let asked: Vec<usize> = parts.iter().flatten().flatten().copied().collect();
        // The converged values at each end, from the most wanted inward, up to the first unit
        // that has not converged; a conjugate pair converges as a whole.
        let mut reported = Vec::with_capacity(asked.len());
        for part in &parts {
            for unit in part {
                let mut all = true;
                for &i in unit {
                    all &= verdicts.get(i);
                }
                if !all {
                    break;
                }
                reported.extend_from_slice(unit);
            }
        }
        if reported.len() == asked.len() {
            return Step::Stop(reported, Stop::Converged, Vec::new());
        }
        let missing: Vec<usize> = asked
            .iter()
            .copied()
            .filter(|&i| !verdicts.get(i))
            .collect();
        if let Some(judge) = self.judge {
            let shown: Vec<Candidate<T::Real>> = (0..m).map(|i| (values[i], bounds[i])).collect();
            if judge(&shown, &mut |i| verdicts.get(i), self.state.norm) {
                return Step::Stop(reported, Stop::Settled, Vec::new());
            }
        }
        if self.state.restarts_left == 0 {
            return Step::Stop(reported, Stop::Exhausted, missing);
        }
        if self.original.is_some() {
            let open: Vec<usize> = asked
                .iter()
                .copied()
                .filter(|&i| i >= first && !verdicts.get(i))
                .collect();
            let modulus = |i: usize| values[i].re.hypot(values[i].im);
            let largest = open.iter().fold(T::Real::ZERO, |m, &i| m.max(modulus(i)));
            let ratio = T::Real::from_f64(LOCKING);
            // The vector of a value locked must be accurate to rounding on the inverse too, not
            // only pass the test with `A`: for an `A` that is not normal, a vector orthogonal to
            // it still has a component along the eigenvector in the eigenbasis, which the
            // inverse multiplies by the value itself, and the projection removes only to the
            // accuracy of the locked vector.
            let rounding = T::Real::from_f64(ROUNDING * (self.n.max(1) as f64).sqrt());
            let lock: Vec<usize> = asked
                .iter()
                .copied()
                .filter(|&i| {
                    i >= first
                        && verdicts.get(i)
                        && modulus(i) >= ratio * largest
                        && bounds[i] <= rounding * T::Real::EPSILON * modulus(i)
                })
                .collect();
            if !lock.is_empty() && !open.is_empty() {
                return Step::Lock(lock, open);
            }
            // The values asked for that are open have converged on the operator to the rounding
            // of their own size, and no value of the factorisation is large enough to spoil them
            // through the restarts (one that was would have been locked above): no restart can
            // move their vectors, and so their residuals with `A`, any further. Near an
            // eigenvalue of an `A` that is not normal that is where the rounding of the inverse
            // leaves the pairs other than the nearest (see `arnoldi_inverse_with`); the run
            // stops there rather than spend the rest of its budget: without this, every call
            // `10⁻⁶·‖A‖` from such an eigenvalue spent all of it.
            let here = (first..m).fold(T::Real::ZERO, |l, i| l.max(modulus(i)));
            let stalled = !open.is_empty()
                && open.iter().all(|&i| {
                    bounds[i] <= T::Real::EPSILON * modulus(i) && here < ratio * modulus(i)
                });
            if stalled {
                return Step::Stop(reported, Stop::Exhausted, missing);
            }
        }
        let length = m - first;
        let want = asked.iter().filter(|&&i| i >= first).count();
        let converged = asked
            .iter()
            .filter(|&&i| i >= first && verdicts.get(i))
            .count();
        // The shifts for ARPACK's count, or, when a conjugate pair completed at its edge leaves
        // none, for one value fewer kept, down to the values asked for, which leave one.
        let mut shifts = Vec::new();
        if let Some(count) = keep(length, want, converged) {
            for count in (want..=count).rev() {
                let keep = kept(self.rule, values, first, count, self.pairs);
                shifts = (first..m)
                    .filter(|&i| !keep[i] && bounds[i] != T::Real::ZERO)
                    .collect();
                if !shifts.is_empty() {
                    break;
                }
            }
        }
        shifts.sort_by(|&i, &j| {
            bounds[j]
                .partial_cmp(&bounds[i])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        if shifts.is_empty() && !room {
            // Only Ritz values with an estimate of exactly zero are left to shift: the dense
            // eigensolver found the factorisation split there, at a coupling below the test of
            // [`negligible`](super::lanczos) but of the order of rounding. Their Ritz vectors
            // are exact as far as the factorisation can tell (for shift-and-invert, those whose
            // residual with `A` passes): they are locked, and the run starts again from the
            // values asked for that are open, rather than stop with restarts left.
            let open: Vec<usize> = asked
                .iter()
                .copied()
                .filter(|&i| i >= first && !verdicts.get(i))
                .collect();
            let exact: Vec<usize> = (first..m)
                .filter(|&i| {
                    bounds[i] == T::Real::ZERO && (self.original.is_none() || verdicts.get(i))
                })
                .collect();
            if !exact.is_empty() && !open.is_empty() {
                return Step::Lock(exact, open);
            }
            return Step::Stop(reported, Stop::Exhausted, missing);
        }
        let columns = length - shifts.len();
        Step::Restart(shifts, columns)
    }
}

/// Whether a residual (or an estimate of one) `bound` of an eigenvalue of absolute value `size`
/// of an operator of the size `operator` passes the convergence test: ARPACK's
/// `bound ≤ tol·max(ε^(2/3), |θ|)`, with its floor `ε^(2/3)` made relative to the size of the
/// operator so that the test is the same for `A` and for `c·A`. An estimate at the level of the
/// rounding of the operator's own products passes too, as nothing below that can be told from
/// zero.
pub(super) fn accepts<R: Float>(tolerance: R, operator: R, size: R, bound: R) -> bool {
    let reach = tolerance * size.max(eps23::<R>() * operator);
    bound <= reach.max(R::EPSILON * operator)
}

/// How many times `√n·ε·‖A‖` a residual with `A`, computed from a product with `A`, may be and
/// still count as converged at the level of rounding: the error of the product, which grows
/// with the length of its inner products.
const ROUNDING: f64 = 8.0;

/// Whether the residual `residual` with `A`, from a product with `A`, of an eigenpair whose
/// eigenvalue has the absolute value `size`, passes the convergence test of shift-and-invert: the
/// test of [`accepts`] made with `A` itself, `tol·max(|λ|, ε^(2/3)·‖A‖)`, with room for the
/// rounding of that product, `8·√n·ε·‖A‖`, `‖A‖` the estimate `operator` or `|λ|` if larger.
pub(super) fn accepts_with<R: Float>(
    tolerance: R,
    operator: R,
    n: usize,
    size: R,
    residual: R,
) -> bool {
    let operator = operator.max(size);
    let reach = tolerance * size.max(eps23::<R>() * operator);
    let room = R::from_f64(ROUNDING * (n.max(1) as f64).sqrt()) * R::EPSILON * operator;
    residual <= reach + room
}

/// How many Ritz values of a factorisation of `m` a restart keeps, at the least, when `want` of
/// them were asked for and `converged` of those converged; `None` when no shift would be left.
/// ARPACK's count: the wanted ones, and on top of them half the shifts that are left, up to the
/// number converged, with its special cases for a single value.
fn keep(m: usize, want: usize, converged: usize) -> Option<usize> {
    let mut keep = want;
    if keep >= m {
        return None;
    }
    keep += converged.min((m - keep) / 2);
    if keep == 1 && m >= 6 {
        keep = m / 2;
    } else if keep == 1 && m > 2 {
        keep = 2;
    }
    (keep < m).then_some(keep)
}
