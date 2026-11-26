//! One run of the implicitly restarted Arnoldi method, for any square operator.
//!
//! The run keeps an `m`-step Arnoldi factorisation `A·V = V·H + f·eₘᵀ`: `V` of `m` orthonormal
//! columns, `H = Vᴴ·A·V` upper Hessenberg, and the residual `f` orthogonal to `V`. The
//! eigenvalues of `H` are the Ritz values and `‖f‖·|yₘ|`, with `yₘ` the last component of a unit
//! eigenvector of `H`, is the norm of the residual of its Ritz pair. A restart applies the Ritz
//! values that are not wanted as exact shifts of implicitly shifted QR steps on `H`, `H ← Qᴴ·H·Q`
//! and `V ← V·Q`, and keeps the leading `k` columns, which is again a factorisation, of a Krylov
//! space whose starting vector is the old one filtered by the polynomial with the shifts as its
//! roots; it is then extended back to `m` steps. This is Sorensen's implicit restart (SIAM J.
//! Matrix Anal. Appl. 13, 1992) as Lehoucq, Sorensen and Yang's ARPACK users' guide (SIAM, 1998)
//! lays it out for `dnaupd` and `znaupd`.
//!
//! For a real operator the factorisation stays real: the Ritz values that are not real come in
//! conjugate pairs, a pair is wanted or shifted away as a whole, and a pair of shifts is applied
//! as one Francis double-shift step in real arithmetic, as `dnapps` does. Carrying the
//! factorisation in complex arithmetic instead would cost four times the work on the basis and
//! twice the memory, and the rounding would no longer keep the conjugate of a Ritz value a Ritz
//! value. For a complex operator each shift is one single-shift step with complex rotations.
//!
//! When an element of the subdiagonal of `H` is zero — a breakdown, or a QR step that made it
//! negligible — the columns above it span an invariant subspace of the operator, whose
//! eigenvalues have converged and cannot be shifted out of the basis. The run locks such a
//! block, as the Lanczos run does: its vectors leave the factorisation, which goes on with all of
//! its columns in their complement, the operator restricted to it, and its eigenvalues stand
//! beside the Ritz values in every decision. The vectors of a locked block are part of what the
//! run finds, wanted or not: the Ritz vectors found in their complement are eigenvectors of the
//! restricted operator, and only together with the block do they span an invariant subspace of
//! `A`.

use super::basis::{
    Outcome, Run, Step, Stop, accepts_with, append_orthonormal, combine, combine_complex, fresh,
    general_eigen, orthogonalise, residual_with, restart_residual, rotate, scale_down, unit,
};
use crate::math::complex::{Complex, Float};
use crate::math::matrix::{Matrix, Scalar};
use core::ops::Range;

/// The factorisation of a run.
struct Factorisation<T: Scalar> {
    /// `V`, room for `m` columns of length `n`.
    v: Vec<T>,
    /// `H`, room for `m × m`, column after column: the leading `len × len` block is in use.
    h: Vec<T>,
    /// `‖f‖`.
    beta: T::Real,
    /// The direction of the residual, `f/‖f‖`, of length 1 (zero when `f` is): see
    /// `basis::restart_residual`.
    f: Vec<T>,
    /// The number of columns in use.
    len: usize,
}

/// The longest the factorisation of a run may be: the size of the basis, or what the locked
/// vectors leave of the space.
fn capacity<T: Scalar>(run: &Run<'_, T>, deflated: &[T]) -> usize {
    let n = run.n;
    run.basis.min(n - run.locked.len() / n - deflated.len() / n)
}

/// The block of order `size` of the matrix `h` (columns of length `ld`) whose first element is
/// the diagonal element `at`, as a matrix of its own, column after column.
fn block<T: Scalar>(h: &[T], ld: usize, at: usize, size: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(size * size);
    for j in 0..size {
        let column = (at + j) * ld + at;
        out.extend_from_slice(&h[column..column + size]);
    }
    out
}

/// `h` (columns of length `ld`) made zero but for its leading block of order `size`, which is
/// `b`, column after column.
fn set_leading<T: Scalar>(h: &mut [T], ld: usize, b: &[T], size: usize) {
    h.fill(T::zero());
    for j in 0..size {
        h[j * ld..j * ld + size].copy_from_slice(&b[j * size..(j + 1) * size]);
    }
}

/// Runs the method from the unit vector `start`, which is orthogonal to the locked vectors, and
/// returns an orthonormal basis of an invariant subspace: the blocks it locked, and the span of
/// the Ritz vectors of the most wanted values asked for that converged — for a real operator,
/// the real and imaginary parts of the vector of a pair in place of the two complex ones.
pub(super) fn run<T: Scalar>(run: &mut Run<'_, T>, start: &[T]) -> Outcome<T> {
    let (n, m) = (run.n, run.basis);
    let zero = T::Real::ZERO;
    let mut fact = Factorisation {
        v: vec![T::zero(); n * m],
        h: vec![T::zero(); m * m],
        beta: zero,
        f: vec![T::zero(); n],
        len: 0,
    };
    // The blocks locked so far: their vectors, orthonormal and orthogonal to the factorisation,
    // and their eigenvalues.
    let mut deflated: Vec<T> = Vec::new();
    let mut locked_values: Vec<Complex<T::Real>> = Vec::new();
    // For shift-and-invert only an exact zero splits `H`, as for Lanczos: the products of
    // `(A − σI)⁻¹` carry errors of the order of `ε/δ` that `H` does not see.
    let deflation = if run.original.is_some() {
        zero
    } else {
        T::Real::EPSILON
    };
    fact.v[..n].copy_from_slice(start);
    let target = capacity(run, &deflated);
    if !extend(run, &mut fact, &deflated, 0, target) {
        return Outcome::empty(Stop::Failed);
    }
    // For shift-and-invert, the leading blocks up to this order were judged and refused (see
    // `Run::may_lock`); they lie above the part the restarts work on, so they do not change.
    let mut refused = 0;
    loop {
        // An element of the subdiagonal that is negligible is a zero, as in the QR algorithm,
        // and the blocks above a zero have split off: lock the largest such leading block that
        // may be locked — for shift-and-invert, one whose pairs pass the test with `A`.
        let len = fact.len;
        for i in 0..len - 1 {
            let at = i + 1 + i * m;
            let (a, c) = (fact.h[i + i * m], fact.h[i + 1 + (i + 1) * m]);
            if negligible(fact.h[at], a, c, deflation) {
                fact.h[at] = T::zero();
            }
        }
        let splits: Vec<usize> = (0..len - 1)
            .filter(|&i| fact.h[i + 1 + i * m] == T::zero())
            .map(|i| i + 1)
            .filter(|&b| b > refused)
            .collect();
        let mut locked_block = false;
        for &b in splits.iter().rev() {
            let Some((theta, y)) = general_eigen(&block(&fact.h, m, 0, b), b) else {
                return Outcome::empty(Stop::Failed);
            };
            let coefficients: Vec<Vec<Complex<T::Real>>> =
                (0..b).map(|i| y.column(i).to_vec()).collect();
            // The columns as they are, after the restarts, are orthonormal only to the drift
            // `append_orthonormal` takes out.
            if !run.may_lock(&deflated, &fact.v[..b * n], &coefficients)
                || !append_orthonormal(n, run.locked, &fact.v[..b * n], &mut deflated)
            {
                continue;
            }
            locked_values.extend(theta);
            fact.v.copy_within(b * n..len * n, 0);
            let rest = block(&fact.h, m, b, len - b);
            set_leading(&mut fact.h, m, &rest, len - b);
            fact.len = len - b;
            locked_block = true;
            break;
        }
        refused = if locked_block {
            0
        } else {
            splits.last().copied().unwrap_or(refused)
        };
        let len = fact.len;
        let h = block(&fact.h, m, 0, len);
        let Some((theta, y)) = general_eigen(&h, len) else {
            return Outcome::empty(Stop::Failed);
        };
        let mut estimates: Vec<T::Real> = (0..len)
            .map(|i| fact.beta * y[(len - 1, i)].abs())
            .collect();
        if run.pairs {
            // The two of a pair have conjugate vectors and one estimate; rounding is not let to
            // pass one of them and not the other.
            for i in 0..len {
                if theta[i].im > zero
                    && let Some(j) = (0..len).find(|&j| theta[j] == theta[i].conj())
                {
                    let b = estimates[i].max(estimates[j]);
                    estimates[i] = b;
                    estimates[j] = b;
                }
            }
        }
        let first = locked_values.len();
        let mut values = locked_values.clone();
        values.extend_from_slice(&theta);
        let mut bounds = vec![zero; first];
        bounds.extend(estimates);
        let target = capacity(run, &deflated);
        let step = {
            let (original, tolerance) = (run.original, run.tolerance);
            let (locked, deflated) = (run.locked, deflated.as_slice());
            let (v, y) = (&fact.v[..len * n], &y);
            // For shift-and-invert: the Ritz vector `V·y`, judged by its residual with `A`.
            let mut check = |i: usize| -> bool {
                let Some(original) = original else {
                    return false;
                };
                let x = combine_complex(n, len, v, y.column(i - first));
                residual_with(original.operator, &x, &[locked, deflated]).is_some_and(
                    |(lambda, r)| accepts_with(tolerance, original.norm, n, lambda.abs(), r),
                )
            };
            let check: Option<&mut dyn FnMut(usize) -> bool> = if original.is_some() {
                Some(&mut check)
            } else {
                None
            };
            run.decide(&values, &bounds, first, len < target, check)
        };
        let (shifts, keep) = match step {
            Step::Stop(reported, stop, missing) => {
                let picked: Vec<usize> = reported
                    .into_iter()
                    .filter(|&i| i >= first)
                    .map(|i| i - first)
                    .collect();
                let frontier = missing.iter().map(|&i| values[i]).collect();
                let mut vectors = deflated;
                span(
                    run,
                    &fact.v[..len * n],
                    len,
                    &y,
                    &theta,
                    &picked,
                    &mut vectors,
                );
                locked_values.extend(picked.iter().map(|&i| theta[i]));
                return Outcome {
                    vectors,
                    values: locked_values,
                    stop,
                    frontier,
                };
            }
            Step::Restart(shifts, keep) => (shifts, keep),
            Step::Lock(lock, open) => {
                // The span of the Ritz vectors `V·y` of the values to lock joins the locked
                // blocks, and the factorisation starts again, in their complement, from the sum
                // of the Ritz vectors of the values asked for that are still open: for a real
                // operator, whose open pairs come whole, a real vector.
                let picked: Vec<usize> = lock.iter().map(|&i| i - first).collect();
                span(
                    run,
                    &fact.v[..len * n],
                    len,
                    &y,
                    &theta,
                    &picked,
                    &mut deflated,
                );
                locked_values.extend(picked.iter().map(|&i| theta[i]));
                let mut c = vec![Complex::new(zero, zero); len];
                for &i in &open {
                    for (sum, &z) in c.iter_mut().zip(y.column(i - first)) {
                        *sum += z;
                    }
                }
                let x = combine_complex(n, len, &fact.v[..len * n], &c);
                let mut w: Vec<T> = x.iter().map(|z| T::from_parts(z.re, z.im)).collect();
                if !unit(&[run.locked, &deflated], &[], &mut w) {
                    match fresh(&mut run.state.random, n, &[run.locked, &deflated], &[]) {
                        Some(v) => w = v,
                        None => return Outcome::empty(Stop::Failed),
                    }
                }
                fact.h.fill(T::zero());
                fact.v[..n].copy_from_slice(&w);
                run.state.restarts_left -= 1;
                run.state.restarts += 1;
                refused = 0;
                let target = capacity(run, &deflated);
                if !extend(run, &mut fact, &deflated, 0, target) {
                    return Outcome::empty(Stop::Failed);
                }
                continue;
            }
        };
        // What the residual of the kept columns is multiplied by: `‖f‖` without a shift.
        let mut scale = fact.beta;
        if !shifts.is_empty() {
            let shifts: Vec<Complex<T::Real>> = shifts.into_iter().map(|i| values[i]).collect();
            let mut h = h;
            let mut q = vec![T::zero(); len * len];
            for i in 0..len {
                q[i + i * len] = T::one();
            }
            if run.pairs {
                apply_real(&mut h, len, &shifts, &mut q, deflation);
            } else {
                apply_complex(&mut h, len, &shifts, &mut q, deflation);
            }
            // The first `keep` columns of `V·Q` are the new basis, and the residual that goes
            // with them is `v⁺·h⁺ + f·σ`: `v⁺` the next column of `V·Q`, `h⁺` the element of the
            // new `H` under its `keep`-th diagonal one, and `σ` the element of the last row of
            // `Q` in column `keep`, the only one of that row the shifts leave nonzero among the
            // columns kept.
            rotate(n, len, &mut fact.v, &q, keep + 1);
            let below = h[keep + (keep - 1) * len];
            let sigma = q[len - 1 + (keep - 1) * len].scale(fact.beta);
            let next = &fact.v[keep * n..(keep + 1) * n];
            scale = restart_residual(&mut fact.f, next, below, sigma);
            // `H` is cut to its leading `keep × keep` block.
            set_leading(&mut fact.h, m, &block(&h, len, 0, keep), keep);
        }
        // Without a shift the factorisation only grows, from its own residual, into the room a
        // locked block left.
        let basis = &fact.v[..keep * n];
        let mut ignored = vec![T::zero(); keep];
        let length = orthogonalise(&[run.locked, &deflated], basis, &mut fact.f, &mut ignored);
        if !length.is_finite() {
            return Outcome::empty(Stop::Failed);
        }
        fact.len = keep;
        // The coupling may underflow to zero where the direction does not: the kept columns
        // are then an invariant subspace as far as the arithmetic can tell.
        let coupling = length * scale;
        if coupling > zero {
            fact.h[keep + (keep - 1) * m] = T::from_real(coupling);
            scale_down(&mut fact.f, length);
            fact.v[keep * n..(keep + 1) * n].copy_from_slice(&fact.f);
        } else {
            // What was kept is an invariant subspace: the factorisation goes on from a vector
            // of its complement, with a zero in `H` where the two parts meet.
            fact.h[keep + (keep - 1) * m] = T::zero();
            let Some(w) = fresh(&mut run.state.random, n, &[run.locked, &deflated], basis) else {
                return Outcome::empty(Stop::Failed);
            };
            fact.v[keep * n..(keep + 1) * n].copy_from_slice(&w);
        }
        run.state.restarts_left -= 1;
        run.state.restarts += 1;
        if !extend(run, &mut fact, &deflated, keep, target) {
            return Outcome::empty(Stop::Failed);
        }
    }
}

/// Extends the factorisation from `from` steps to `to`: column `from` of `V` is in place, and
/// each step applies the operator to the last column, takes out of the image its components
/// along the locked vectors (of earlier runs, `run.locked`, and of this one, `deflated`) and
/// along the whole basis by Gram–Schmidt applied twice, puts the coefficients along the basis
/// into column `j` of `H` and makes what is left the next column of `V`. A basis that reaches
/// everything the locked vectors leave has no residual. False if the operator gives a value
/// that is not finite.
fn extend<T: Scalar>(
    run: &mut Run<'_, T>,
    fact: &mut Factorisation<T>,
    deflated: &[T],
    from: usize,
    to: usize,
) -> bool {
    let (n, m) = (run.n, run.basis);
    let complete = run.locked.len() / n + deflated.len() / n + to == n;
    let mut w = vec![T::zero(); n];
    for j in from..to {
        if !run.apply(&fact.v[j * n..(j + 1) * n], &mut w) {
            return false;
        }
        let basis = &fact.v[..(j + 1) * n];
        let column = &mut fact.h[j * m..j * m + j + 1];
        let mut length = orthogonalise(&[run.locked, deflated], basis, &mut w, column);
        if !length.is_finite() {
            return false;
        }
        if j + 1 == to {
            if complete {
                length = T::Real::ZERO;
                w.fill(T::zero());
            }
            fact.beta = length;
            if length > T::Real::ZERO {
                scale_down(&mut w, length);
            }
            fact.f.copy_from_slice(&w);
        } else if length > T::Real::ZERO {
            fact.h[j + 1 + j * m] = T::from_real(length);
            scale_down(&mut w, length);
            fact.v[(j + 1) * n..(j + 2) * n].copy_from_slice(&w);
        } else {
            // The basis so far spans an invariant subspace: any unit vector orthogonal to it
            // continues the factorisation.
            fact.h[j + 1 + j * m] = T::zero();
            let Some(next) = fresh(&mut run.state.random, n, &[run.locked, deflated], basis) else {
                return false;
            };
            fact.v[(j + 1) * n..(j + 2) * n].copy_from_slice(&next);
        }
    }
    fact.len = to;
    true
}

/// Appends to `out` (the blocks the run locked) an orthonormal basis of the span of the Ritz
/// vectors `V·y` of the Ritz values `picked` of the factorisation of `len` columns `v`,
/// orthogonal to the locked vectors and to `out`. For a real operator a pair contributes the
/// real and the imaginary part of the vector of its member with the positive imaginary part,
/// which span the same real space as the two complex vectors.
fn span<T: Scalar>(
    run: &Run<'_, T>,
    v: &[T],
    len: usize,
    y: &Matrix<Complex<T::Real>>,
    values: &[Complex<T::Real>],
    picked: &[usize],
    out: &mut Vec<T>,
) {
    let n = run.n;
    let add = |s: Vec<T>, out: &mut Vec<T>| {
        append_orthonormal(n, run.locked, &combine(n, len, v, &s, 1), out);
    };
    for &i in picked {
        let column = y.column(i);
        if run.pairs {
            if values[i].im < T::Real::ZERO {
                continue;
            }
            add(column.iter().map(|z| T::from_real(z.re)).collect(), out);
            if values[i].im > T::Real::ZERO {
                add(column.iter().map(|z| T::from_real(z.im)).collect(), out);
            }
        } else {
            add(
                column.iter().map(|z| T::from_parts(z.re, z.im)).collect(),
                out,
            );
        }
    }
}

/// Whether the subdiagonal element `b` between the diagonal elements `a` and `c` of a Hessenberg
/// matrix is negligible: `|b| ≤ factor·√(|a|·|c|)`, with `factor = ε` the test of the QR
/// algorithm in the form that keeps the small eigenvalues of a graded matrix, as for Lanczos;
/// with `factor = 0` only an exact zero.
fn negligible<T: Scalar>(b: T, a: T, c: T, factor: T::Real) -> bool {
    b.abs() <= factor * (a.abs().sqrt() * c.abs().sqrt())
}

/// The unreduced diagonal blocks of rows `first..m` of the Hessenberg `h` of order `m`, as
/// `(low, high)`, after the subdiagonal elements that have become negligible are set to zero.
fn blocks<T: Scalar>(h: &mut [T], m: usize, first: usize, factor: T::Real) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut low = first;
    while low < m {
        let mut high = low;
        while high + 1 < m {
            let at = high + 1 + high * m;
            if negligible(
                h[at],
                h[high + high * m],
                h[high + 1 + (high + 1) * m],
                factor,
            ) {
                h[at] = T::zero();
                break;
            }
            high += 1;
        }
        out.push((low, high));
        low = high + 1;
    }
    out
}

/// The rotation `G = [[c, s], [−s̄, c]]`, `c` real, with `G·(f, g) = (r, 0)`: `(c, s, r)`.
/// Every quotient is a division, not a product with a reciprocal, which overflows for a
/// subnormal divisor. Where `f` and `g` are so small that they may be subnormal numbers of a few
/// bits — as the couplings of converged parts become in shift-and-invert, where only exact zeros
/// split `H` — `c` and `s` come from them divided by their largest part, so that
/// `c² + |s|² = 1` to working precision: from the numbers themselves the rotation was not
/// orthogonal, and the basis it was multiplied into lost its orthonormality.
/// Elsewhere the direct form, which kept the residuals of a random stress smaller.
fn givens<T: Scalar>(f: T, g: T) -> (T::Real, T, T) {
    let zero = T::Real::ZERO;
    if g == T::zero() {
        return (T::Real::ONE, T::zero(), f);
    }
    let t = f
        .re()
        .abs()
        .max(f.im().abs())
        .max(g.re().abs())
        .max(g.im().abs());
    let tiny = t < T::Real::MIN_POSITIVE / T::Real::EPSILON;
    let t = if tiny { t } else { T::Real::ONE };
    let (f, g) = (f / T::from_real(t), g / T::from_real(t));
    let (af, ag) = (f.abs(), g.abs());
    if af == zero {
        return (zero, g.conj() / T::from_real(ag), T::from_real(ag * t));
    }
    let r = af.hypot(ag);
    let phase = f / T::from_real(af);
    (
        af / r,
        phase * (g.conj() / T::from_real(r)),
        phase.scale(r * t),
    )
}

/// `G` of [`givens`] on the rows `i, i + 1` of the columns `columns` of `h` (order `m`).
fn rows<T: Scalar>(h: &mut [T], m: usize, i: usize, columns: Range<usize>, g: (T::Real, T)) {
    let (c, s) = (T::from_real(g.0), g.1);
    for j in columns {
        let (a, b) = (h[i + j * m], h[i + 1 + j * m]);
        h[i + j * m] = c * a + s * b;
        h[i + 1 + j * m] = c * b - s.conj() * a;
    }
}

/// `Gᴴ` of [`givens`] on the columns `i, i + 1` of the rows `range` of `h`, whose columns have the
/// length `ld`.
fn columns<T: Scalar>(h: &mut [T], ld: usize, i: usize, range: Range<usize>, g: (T::Real, T)) {
    let (c, s) = (T::from_real(g.0), g.1);
    let (left, right) = h.split_at_mut((i + 1) * ld);
    for (a, b) in left[i * ld..][range.clone()]
        .iter_mut()
        .zip(&mut right[range])
    {
        let (x, y) = (*a, *b);
        *a = x * c + y * s.conj();
        *b = y * c - x * s;
    }
}

/// The single-shift QR step with the shift `mu` on the unreduced block `low..=high` of the
/// Hessenberg `h` of order `m`, `h ← Gᴴ·h·G` rotation by rotation, the rotations multiplied
/// into `q`: the first rotation takes the first column of `h − μ·I` to a multiple of the first
/// unit vector, the others chase the element it puts below the subdiagonal out of the block.
fn single<T: Scalar>(h: &mut [T], m: usize, low: usize, high: usize, mu: T, q: &mut [T]) {
    let mut x = h[low + low * m] - mu;
    let mut z = h[low + 1 + low * m];
    for i in low..high {
        if i > low {
            x = h[i + (i - 1) * m];
            z = h[i + 1 + (i - 1) * m];
        }
        let (c, s, r) = givens(x, z);
        if i > low {
            h[i + (i - 1) * m] = r;
            h[i + 1 + (i - 1) * m] = T::zero();
        }
        rows(h, m, i, i..m, (c, s));
        columns(h, m, i, 0..(i + 2).min(high) + 1, (c, s));
        columns(q, m, i, 0..m, (c, s));
    }
}

/// The first row of the block of `h` (order `m`) that holds its last row, after the negligible
/// subdiagonal elements are set to zero: the blocks above it have split off, invariant
/// subspaces whose Ritz values are converged or cannot be shifted, which a restart leaves as
/// they are — `QR(ψ(H))` of a block triangular `H` is block diagonal, so rotating them would
/// change nothing but their rounding.
fn active<T: Scalar>(h: &mut [T], m: usize, factor: T::Real) -> usize {
    blocks(h, m, 0, factor).last().map_or(0, |b| b.0)
}

/// The single-shift steps of the shifts, each on every unreduced block of the active part of
/// `h`, for a complex operator: one step a shift.
fn apply_complex<T: Scalar>(
    h: &mut [T],
    m: usize,
    shifts: &[Complex<T::Real>],
    q: &mut [T],
    deflation: T::Real,
) {
    let first = active(h, m, deflation);
    for &mu in shifts {
        let mu = T::from_parts(mu.re, mu.im);
        for (low, high) in blocks(h, m, first, deflation) {
            if high > low {
                single(h, m, low, high, mu, q);
            }
        }
    }
    blocks(h, m, 0, deflation);
}

/// The shifts of a real operator, each on every unreduced block of the active part of `h`: a
/// real shift by a single-shift step in real arithmetic, a conjugate pair (the member with the
/// positive imaginary part stands for it; the other is passed over) by a Francis double-shift
/// step in real arithmetic.
fn apply_real<T: Scalar>(
    h: &mut [T],
    m: usize,
    shifts: &[Complex<T::Real>],
    q: &mut [T],
    deflation: T::Real,
) {
    let mut hr: Vec<T::Re> = h.iter().map(|x| real(x.re())).collect();
    let mut qr: Vec<T::Re> = q.iter().map(|x| real(x.re())).collect();
    let first = active(&mut hr, m, deflation);
    for &mu in shifts {
        if mu.im < T::Real::ZERO {
            continue;
        }
        for (low, high) in blocks(&mut hr, m, first, deflation) {
            if high == low {
                continue;
            }
            if mu.im == T::Real::ZERO {
                single(&mut hr, m, low, high, real(mu.re), &mut qr);
            } else {
                double(&mut hr, m, low, high, mu, &mut qr);
            }
        }
    }
    blocks(&mut hr, m, 0, deflation);
    for (x, r) in h.iter_mut().zip(hr) {
        *x = T::from_real(part(r));
    }
    for (x, r) in q.iter_mut().zip(qr) {
        *x = T::from_real(part(r));
    }
}

/// The real number `x` as an element of the real type `S`.
fn real<S: Scalar>(x: S::Real) -> S {
    S::from_real(x)
}

/// The real part of `x`, which for the real type `S` is all of it.
fn part<S: Scalar>(x: S) -> S::Real {
    x.re()
}

/// The real reflection `P = I − τ·u·uᵀ`, `u = (1, u₁, u₂)`, with `P·(x, y, z) = (β, 0, 0)`:
/// `(τ, u₁, u₂, β)`, the identity when `y` and `z` are zero.
fn reflector3<R: Float>(x: R, y: R, z: R) -> (R, R, R, R) {
    let scale = x.abs().max(y.abs()).max(z.abs());
    if y == R::ZERO && z == R::ZERO || scale == R::ZERO {
        return (R::ZERO, R::ZERO, R::ZERO, x);
    }
    let (xs, ys, zs) = (x / scale, y / scale, z / scale);
    let length = (xs * xs + ys * ys + zs * zs).sqrt();
    let beta = -length.copysign(xs);
    let d = xs - beta;
    ((beta - xs) / beta, ys / d, zs / d, beta * scale)
}

/// `P` of [`reflector3`], given as `(τ, u₁, u₂)`, on the rows `k..k + 3` of the columns `range`
/// of the real `h` (order `m`).
fn reflect_rows<S: Scalar>(h: &mut [S], m: usize, k: usize, range: Range<usize>, p: (S, S, S)) {
    let (tau, u1, u2) = p;
    for j in range {
        let c = j * m;
        let s = h[k + c] + u1 * h[k + 1 + c] + u2 * h[k + 2 + c];
        h[k + c] -= tau * s;
        h[k + 1 + c] -= tau * s * u1;
        h[k + 2 + c] -= tau * s * u2;
    }
}

/// `P` of [`reflector3`], given as `(τ, u₁, u₂)`, on the columns `k..k + 3` of the rows `range`
/// of the real `h`, whose columns have the length `ld`.
fn reflect_columns<S: Scalar>(h: &mut [S], ld: usize, k: usize, range: Range<usize>, p: (S, S, S)) {
    let (tau, u1, u2) = p;
    for i in range {
        let s = h[i + k * ld] + u1 * h[i + (k + 1) * ld] + u2 * h[i + (k + 2) * ld];
        h[i + k * ld] -= tau * s;
        h[i + (k + 1) * ld] -= tau * s * u1;
        h[i + (k + 2) * ld] -= tau * s * u2;
    }
}

/// Francis's double-shift step with the shifts `μ` and `μ̄` on the unreduced block `low..=high`
/// (of at least two rows) of the real Hessenberg `h` of order `m`, the reflections multiplied
/// into `q`: a reflection takes the first column of `(H − μ·I)·(H − μ̄·I) = H² − 2·Re μ·H + |μ|²·I`,
/// which has three nonzero elements, to a multiple of the first unit vector, and reflections of
/// three rows, then one rotation of two, chase the bulge it makes out of the block.
///
/// That column is formed divided by `s = |h₀₀ − Re μ| + |Im μ| + |h₁₀|`, as LAPACK's `dlahqr`
/// forms it: its elements are products of two elements of `H` or of the shift, which overflow
/// once `‖H‖²` passes the largest number of the type and underflow below the smallest, where
/// after the division no product exceeds an element of `H` or underflows unless the element
/// does. Only the direction of the column matters.
fn double<S: Scalar>(
    h: &mut [S],
    m: usize,
    low: usize,
    high: usize,
    mu: Complex<S::Real>,
    q: &mut [S],
) {
    let at = |i: usize, j: usize| i + j * m;
    let (re, im) = (S::from_real(mu.re), S::from_real(mu.im));
    let (h00, h10, h01, h11) = (
        h[at(low, low)],
        h[at(low + 1, low)],
        h[at(low, low + 1)],
        h[at(low + 1, low + 1)],
    );
    let d0 = h00 - re;
    let s = d0.abs() + im.abs() + h10.abs();
    if s == S::Real::ZERO || !s.is_finite() {
        // `h₁₀ = 0`: the block is not unreduced, and there is nothing to chase.
        return;
    }
    let s = S::from_real(s);
    let h10s = h10 / s;
    let mut x = h10s * h01 + d0 * (d0 / s) + im * (im / s);
    let mut y = h10s * (d0 + (h11 - re));
    let mut z = if high >= low + 2 {
        h10s * h[at(low + 2, low + 1)]
    } else {
        S::zero()
    };
    for k in low..high.saturating_sub(1) {
        let (tau, u1, u2, beta) = reflector3(x.re(), y.re(), z.re());
        let p = (S::from_real(tau), S::from_real(u1), S::from_real(u2));
        let first = if k > low { k - 1 } else { low };
        reflect_rows(h, m, k, first..m, p);
        if k > low {
            h[at(k, k - 1)] = S::from_real(beta);
            h[at(k + 1, k - 1)] = S::zero();
            h[at(k + 2, k - 1)] = S::zero();
        }
        reflect_columns(h, m, k, 0..(k + 3).min(high) + 1, p);
        reflect_columns(q, m, k, 0..m, p);
        x = h[at(k + 1, k)];
        y = h[at(k + 2, k)];
        if k + 3 <= high {
            z = h[at(k + 3, k)];
        }
    }
    // The last two rows: a rotation.
    let k = high - 1;
    let (c, s, r) = givens(x, y);
    if k > low {
        h[at(k, k - 1)] = r;
        h[at(k + 1, k - 1)] = S::zero();
        rows(h, m, k, k..m, (c, s));
    } else {
        rows(h, m, k, low..m, (c, s));
    }
    columns(h, m, k, 0..high + 1, (c, s));
    columns(q, m, k, 0..m, (c, s));
}
