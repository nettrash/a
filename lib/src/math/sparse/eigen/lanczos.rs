//! One run of the implicitly restarted Lanczos method, for a Hermitian operator.
//!
//! The run keeps an `m`-step Lanczos factorisation `A·V = V·T + f·eₘᵀ`: `V` of `m` orthonormal
//! columns, `T` real, symmetric and tridiagonal — real for a complex Hermitian operator too —
//! and the residual `f` orthogonal to `V`. The eigenvalues of `T` are the Ritz values, and
//! `‖f‖·|sₘ|`, with `sₘ` the last component of an eigenvector of `T`, is the norm of the
//! residual of its Ritz pair. A restart applies the Ritz values that are not wanted as the
//! shifts of implicit QR steps on `T`, which leaves a factorisation of the same form whose
//! first `k` columns hold what was wanted of the space; the factorisation is then extended back
//! to `m` steps. This is Sorensen's method (SIAM J. Matrix Anal. Appl. 13, 1992) as Lehoucq,
//! Sorensen and Yang's ARPACK users' guide (SIAM, 1998) lays it out.
//!
//! When an element of the subdiagonal of `T` is zero — the factorisation broke down and went on
//! from a fresh vector, or a QR step made the element negligible — the columns above it span an
//! invariant subspace: their eigenvalues are converged, and no shift can move them out of the
//! basis. The run locks such a block: its vectors leave the factorisation, which goes on in
//! their complement with all of its columns, and its eigenvalues stand beside the Ritz values in
//! every decision. Without it, blocks of values that cannot be shifted fill the basis of an
//! operator with few distinct eigenvalues, and the run stops with nothing left to shift.

use super::basis::{
    Outcome, Run, Step, Stop, accepts_with, append_orthonormal, combine, combine_complex, fresh,
    hermitian_eigen, orthogonalise, residual_with, restart_residual, rotate, scale_down, unit,
};
use crate::math::complex::{Complex, Float};
use crate::math::matrix::Scalar;

/// The factorisation of a run.
struct Factorisation<T: Scalar> {
    /// `V`, room for `m` columns of length `n`.
    v: Vec<T>,
    /// The diagonal of `T`.
    alpha: Vec<T::Real>,
    /// The subdiagonal of `T`, and after it the norm of the residual: `beta[j]` couples the
    /// columns `j` and `j + 1`, and `beta[len − 1]` is `‖f‖`.
    beta: Vec<T::Real>,
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

/// Runs the method from the unit vector `start`, which is orthogonal to the locked vectors, and
/// returns the blocks it locked and the Ritz vectors of the most wanted values asked for that
/// converged.
pub(super) fn run<T: Scalar>(run: &mut Run<'_, T>, start: &[T]) -> Outcome<T> {
    let (n, m) = (run.n, run.basis);
    let zero = T::Real::ZERO;
    let mut fact = Factorisation {
        v: vec![T::zero(); n * m],
        alpha: vec![zero; m],
        beta: vec![zero; m],
        f: vec![T::zero(); n],
        len: 0,
    };
    // The blocks locked so far: their vectors, orthonormal and orthogonal to the factorisation,
    // and their eigenvalues.
    let mut deflated: Vec<T> = Vec::new();
    let mut locked_values: Vec<Complex<T::Real>> = Vec::new();
    // For shift-and-invert only an exact zero splits `T`: a Ritz value of `(A − σI)⁻¹` near `σ`
    // is of the order of `1/δ`, `δ` the distance of `σ` from an eigenvalue, the products carry
    // errors of the order of `ε/δ` that `T` does not see, and an element of `T` small enough to
    // pass for negligible may still couple a Ritz vector to directions it has not shed: the
    // restarts go on refining it, as its residual with `A` shows, where a split would lock it
    // as it is.
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
            if negligible(fact.beta[i], fact.alpha[i], fact.alpha[i + 1], deflation) {
                fact.beta[i] = zero;
            }
        }
        let splits: Vec<usize> = (0..len - 1)
            .filter(|&i| fact.beta[i] == zero)
            .map(|i| i + 1)
            .filter(|&b| b > refused)
            .collect();
        let mut locked_block = false;
        for &b in splits.iter().rev() {
            let Some((theta, s)) =
                tridiagonal_eigen::<T::Re>(&fact.alpha[..b], &fact.beta[..b - 1])
            else {
                return Outcome::empty(Stop::Failed);
            };
            let coefficients: Vec<Vec<Complex<T::Real>>> = s
                .chunks_exact(b)
                .map(|column| column.iter().map(|&x| Complex::new(x, zero)).collect())
                .collect();
            // The columns as they are, after the restarts, are orthonormal only to the drift
            // `append_orthonormal` takes out.
            if !run.may_lock(&deflated, &fact.v[..b * n], &coefficients)
                || !append_orthonormal(n, run.locked, &fact.v[..b * n], &mut deflated)
            {
                continue;
            }
            locked_values.extend(theta.iter().map(|&x| Complex::new(x, zero)));
            fact.v.copy_within(b * n..len * n, 0);
            fact.alpha.copy_within(b..len, 0);
            fact.beta.copy_within(b..len, 0);
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
        let Some((theta, s)) =
            tridiagonal_eigen::<T::Re>(&fact.alpha[..len], &fact.beta[..len - 1])
        else {
            return Outcome::empty(Stop::Failed);
        };
        let first = locked_values.len();
        let mut values = locked_values.clone();
        values.extend(theta.iter().map(|&x| Complex::new(x, zero)));
        let residual = fact.beta[len - 1];
        let mut bounds = vec![zero; first];
        bounds.extend((0..len).map(|i| residual * s[len - 1 + i * len].abs()));
        let target = capacity(run, &deflated);
        let step = {
            let (original, tolerance) = (run.original, run.tolerance);
            let (locked, deflated) = (run.locked, deflated.as_slice());
            let (v, s) = (&fact.v[..len * n], &s);
            // For shift-and-invert: the Ritz vector `V·s`, judged by its residual with `A`.
            let mut check = |i: usize| -> bool {
                let Some(original) = original else {
                    return false;
                };
                let column = &s[(i - first) * len..(i - first + 1) * len];
                let y: Vec<Complex<T::Real>> =
                    column.iter().map(|&x| Complex::new(x, zero)).collect();
                let x = combine_complex(n, len, v, &y);
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
                let picked: Vec<usize> = reported.into_iter().filter(|&i| i >= first).collect();
                let coefficients: Vec<T> = picked
                    .iter()
                    .flat_map(|&i| s[(i - first) * len..(i - first + 1) * len].iter())
                    .map(|&x| T::from_real(x))
                    .collect();
                // The Ritz vectors `V·s`, made orthonormal to working precision, as the columns
                // of `V` are not after the restarts.
                let ritz = combine(n, len, &fact.v, &coefficients, picked.len());
                let mut vectors = deflated;
                for (&i, x) in picked.iter().zip(ritz.chunks_exact(n)) {
                    if append_orthonormal(n, run.locked, x, &mut vectors) {
                        locked_values.push(values[i]);
                    }
                }
                let frontier = missing.iter().map(|&i| values[i]).collect();
                return Outcome {
                    vectors,
                    values: locked_values,
                    stop,
                    frontier,
                };
            }
            Step::Restart(shifts, keep) => (shifts, keep),
            Step::Lock(lock, open) => {
                // The Ritz vectors `V·s` of the values to lock join the locked blocks, and the
                // factorisation starts again, in their complement, from the sum of the Ritz
                // vectors of the values asked for that are still open.
                let column = |i: usize| &s[(i - first) * len..(i - first + 1) * len];
                for &i in &lock {
                    let c: Vec<T> = column(i).iter().map(|&x| T::from_real(x)).collect();
                    let mut x = combine(n, len, &fact.v, &c, 1);
                    if unit(&[run.locked, &deflated], &[], &mut x) {
                        deflated.extend_from_slice(&x);
                        locked_values.push(values[i]);
                    }
                }
                let mut c = vec![T::zero(); len];
                for &i in &open {
                    for (sum, &x) in c.iter_mut().zip(column(i)) {
                        *sum += T::from_real(x);
                    }
                }
                let mut w = combine(n, len, &fact.v, &c, 1);
                if !unit(&[run.locked, &deflated], &[], &mut w) {
                    match fresh(&mut run.state.random, n, &[run.locked, &deflated], &[]) {
                        Some(v) => w = v,
                        None => return Outcome::empty(Stop::Failed),
                    }
                }
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
        let mut scale = fact.beta[len - 1];
        if !shifts.is_empty() {
            let shifts: Vec<T::Real> = shifts.into_iter().map(|i| values[i].re).collect();
            let mut q = vec![zero; len * len];
            for i in 0..len {
                q[i + i * len] = T::Real::ONE;
            }
            shift(
                &mut fact.alpha[..len],
                &mut fact.beta[..len - 1],
                &shifts,
                &mut q,
                deflation,
            );
            // The first `keep` columns of `V·Q` are the new basis, and the residual that goes
            // with them is `v⁺·β⁺ + f·σ`: `v⁺` the next column of `V·Q`, `β⁺` the element of the
            // new `T` under its `keep`-th diagonal one, and `σ` the last element of the
            // `keep`-th column of `Q`, the only one of the last row of `Q` that the
            // `len − keep` shifts leave nonzero among the columns kept.
            let q: Vec<T> = q.into_iter().map(T::from_real).collect();
            rotate(n, len, &mut fact.v, &q, keep + 1);
            let below = T::from_real(fact.beta[keep - 1]);
            let sigma = q[len - 1 + (keep - 1) * len].scale(fact.beta[len - 1]);
            let next = &fact.v[keep * n..(keep + 1) * n];
            scale = restart_residual(&mut fact.f, next, below, sigma);
        }
        // Without a shift the factorisation only grows, from its own residual, into the room a
        // locked block left.
        let basis = &fact.v[..keep * n];
        let mut ignored = vec![T::zero(); keep];
        let length = orthogonalise(&[run.locked, &deflated], basis, &mut fact.f, &mut ignored);
        if !length.is_finite() {
            return Outcome::empty(Stop::Failed);
        }
        // The coupling may underflow to zero where the direction does not: the kept columns
        // are then an invariant subspace as far as the arithmetic can tell.
        let coupling = length * scale;
        fact.beta[keep - 1] = coupling;
        fact.len = keep;
        if coupling > zero {
            scale_down(&mut fact.f, length);
            fact.v[keep * n..(keep + 1) * n].copy_from_slice(&fact.f);
        } else {
            // What was kept is an invariant subspace: the factorisation goes on from a vector
            // of its complement, with a zero in `T` where the two parts meet.
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
/// along the whole of the basis by Gram–Schmidt applied twice — the full reorthogonalisation
/// that keeps the basis orthonormal to working precision, where the three-term recurrence alone
/// loses it as soon as a Ritz pair converges — and makes what is left the next column. A basis
/// that reaches everything the locked vectors leave of the space has no residual: its last step
/// gives the exact zero, which rounding would make a small number of. False if the operator
/// gives a value that is not finite.
fn extend<T: Scalar>(
    run: &mut Run<'_, T>,
    fact: &mut Factorisation<T>,
    deflated: &[T],
    from: usize,
    to: usize,
) -> bool {
    let n = run.n;
    let complete = run.locked.len() / n + deflated.len() / n + to == n;
    let mut w = vec![T::zero(); n];
    let mut h = vec![T::zero(); to];
    for j in from..to {
        if !run.apply(&fact.v[j * n..(j + 1) * n], &mut w) {
            return false;
        }
        let basis = &fact.v[..(j + 1) * n];
        let mut length = orthogonalise(&[run.locked, deflated], basis, &mut w, &mut h[..=j]);
        if !length.is_finite() {
            return false;
        }
        // `vⱼᴴ·A·vⱼ` is real for a Hermitian operator; its imaginary part is rounding.
        fact.alpha[j] = h[j].re();
        if j + 1 == to {
            if complete {
                length = T::Real::ZERO;
                w.fill(T::zero());
            }
            fact.beta[j] = length;
            if length > T::Real::ZERO {
                scale_down(&mut w, length);
            }
            fact.f.copy_from_slice(&w);
        } else if length > T::Real::ZERO {
            fact.beta[j] = length;
            scale_down(&mut w, length);
            fact.v[(j + 1) * n..(j + 2) * n].copy_from_slice(&w);
        } else {
            // The basis so far spans an invariant subspace. Any unit vector orthogonal to it
            // continues the factorisation, with a zero in `T` where the two parts meet.
            fact.beta[j] = T::Real::ZERO;
            let Some(next) = fresh(&mut run.state.random, n, &[run.locked, deflated], basis) else {
                return false;
            };
            fact.v[(j + 1) * n..(j + 2) * n].copy_from_slice(&next);
        }
    }
    fact.len = to;
    true
}

/// The eigenvalues of a symmetric tridiagonal matrix of order `m`, ascending, and its
/// eigenvectors, `m × m` column after column.
type Eigensystem<R> = (Vec<R>, Vec<R>);

/// The eigenvalues, ascending, and the eigenvectors of the symmetric tridiagonal matrix with
/// the diagonal `alpha` and the subdiagonal `beta`, by the dense Hermitian eigensolver on the
/// real type `S`. `None` if it gives a value that is not finite.
fn tridiagonal_eigen<S: Scalar>(
    alpha: &[S::Real],
    beta: &[S::Real],
) -> Option<Eigensystem<S::Real>> {
    let m = alpha.len();
    let mut t = vec![S::zero(); m * m];
    for i in 0..m {
        t[i + i * m] = S::from_real(alpha[i]);
        if i + 1 < m {
            t[i + 1 + i * m] = S::from_real(beta[i]);
            t[i + (i + 1) * m] = S::from_real(beta[i]);
        }
    }
    let (values, vectors) = hermitian_eigen::<S>(&t, m)?;
    let vectors = vectors.as_slice().iter().map(|x| x.re()).collect();
    Some((values, vectors))
}

/// Whether the element `b` between the diagonal elements `a` and `c` of a tridiagonal matrix is
/// negligible: `|b| ≤ factor·√(|a|·|c|)`, with `factor = ε` the test of the QR algorithm in the
/// form that keeps the
/// small eigenvalues of a graded matrix accurate. The tridiagonal matrix of shift-and-invert
/// spans many orders of magnitude — `1/(λ − σ)` for an eigenvalue `λ` near `σ`, and numbers of
/// the order of `1/‖A‖` — and the form `ε·(|a| + |c|)` would split off, as converged, Ritz
/// values of its small end that are coupled to its large end by far more than their own
/// accuracy allows. The deflation a split brings is the locking of a block, whose values are
/// then never revised, so the test errs on the side of keeping a block coupled.
fn negligible<R: Float>(b: R, a: R, c: R, factor: R) -> bool {
    b.abs() <= factor * (a.abs().sqrt() * c.abs().sqrt())
}

/// Applies the shifts to the symmetric tridiagonal matrix `(alpha, beta)` by implicit QR steps,
/// `T ← Qᵀ·T·Q`, and multiplies the rotations into `q`. An element of the subdiagonal that has
/// become negligible is set to zero, and a shift is then applied to each of the blocks it
/// separates: a Ritz value that has converged ends up in a block of its own — it is deflated,
/// and the run locks it at its next decision. The blocks that had split off above the one that
/// holds the last row before the restart are left alone: `QR(ψ(T))` of a block diagonal `T` is
/// block diagonal, and rotating them would change nothing but their rounding.
fn shift<R: Float>(alpha: &mut [R], beta: &mut [R], shifts: &[R], q: &mut [R], deflation: R) {
    let m = alpha.len();
    let mut first = 0;
    for i in 0..m.saturating_sub(1) {
        if negligible(beta[i], alpha[i], alpha[i + 1], deflation) {
            beta[i] = R::ZERO;
            first = i + 1;
        }
    }
    for &mu in shifts {
        let mut low = first;
        while low < m {
            let mut high = low;
            while high + 1 < m {
                if negligible(beta[high], alpha[high], alpha[high + 1], deflation) {
                    beta[high] = R::ZERO;
                    break;
                }
                high += 1;
            }
            if high > low {
                step(alpha, beta, low, high, mu, q);
            }
            low = high + 1;
        }
    }
    for i in 0..m.saturating_sub(1) {
        if negligible(beta[i], alpha[i], alpha[i + 1], deflation) {
            beta[i] = R::ZERO;
        }
    }
}

/// One implicit QR step with the shift `mu` on the unreduced block `low..=high` of the
/// tridiagonal matrix: the rotation that takes `(α_low − μ, β_low)` to a multiple of the first
/// unit vector, then the rotations that chase the element it puts outside the three diagonals
/// down the block and out of it.
fn step<R: Float>(alpha: &mut [R], beta: &mut [R], low: usize, high: usize, mu: R, q: &mut [R]) {
    let m = alpha.len();
    let mut x = alpha[low] - mu;
    let mut z = beta[low];
    for i in low..high {
        // The rotation `G = [[c, −s], [s, c]]` of the rows and columns `i, i + 1`, with
        // `Gᵀ·(x, z) = (r, 0)`. Where `x` and `z` are so small that they may be subnormal
        // numbers of a few bits — as the couplings of converged parts become in
        // shift-and-invert, where only exact zeros split `T` — `c` and `s` come from them
        // divided by the larger, so that `c² + s² = 1` to working precision: from the numbers
        // themselves the rotation was not orthogonal, and the basis lost its orthonormality
        // Elsewhere the direct form, which over many restarts kept the
        // residuals about ten times smaller in a random stress than the divided one.
        let t = x.abs().max(z.abs());
        let (c, s, r) = if t != R::ZERO && t < R::MIN_POSITIVE / R::EPSILON {
            let (x1, z1) = (x / t, z / t);
            let r1 = x1.hypot(z1);
            (x1 / r1, z1 / r1, r1 * t)
        } else {
            let r = x.hypot(z);
            if r == R::ZERO {
                (R::ONE, R::ZERO, r)
            } else {
                (x / r, z / r, r)
            }
        };
        if i > low {
            beta[i - 1] = r;
        }
        let (a, b, d) = (alpha[i], beta[i], alpha[i + 1]);
        alpha[i] = c * (c * a + s * b) + s * (c * b + s * d);
        alpha[i + 1] = s * (s * a - c * b) + c * (c * d - s * b);
        beta[i] = c * (c * b + s * d) - s * (c * a + s * b);
        if i + 1 < high {
            // The element pushed below the subdiagonal, which the next rotation removes.
            x = beta[i];
            z = s * beta[i + 1];
            beta[i + 1] = c * beta[i + 1];
        }
        let (left, right) = q.split_at_mut((i + 1) * m);
        for (u, v) in left[i * m..].iter_mut().zip(&mut right[..m]) {
            let (p, t) = (*u, *v);
            *u = c * p + s * t;
            *v = c * t - s * p;
        }
    }
}
