//! Householder reflections, alone and in blocks, which QR, Hessenberg and bidiagonal reductions
//! are built of.

use super::blas::{Op, gemm};
use super::{Matrix, Scalar};
use crate::math::complex::Float;

/// The 2-norm of `x`, scaled so that the squares neither overflow nor underflow.
pub(crate) fn norm2<T: Scalar>(x: &[T]) -> T::Real {
    let mut scale = T::Real::ZERO;
    for v in x {
        scale = scale.max(v.re().abs()).max(v.im().abs());
    }
    if scale == T::Real::ZERO || !scale.is_finite() {
        return scale;
    }
    let r = T::Real::ONE / scale;
    let mut s = T::Real::ZERO;
    if r.is_finite() {
        for v in x {
            s += v.scale(r).abs2();
        }
    } else {
        // A subnormal scale, whose reciprocal overflows: a division for each part.
        for v in x {
            let (a, b) = (v.re() / scale, v.im() / scale);
            s += a * a + b * b;
        }
    }
    s.sqrt() * scale
}

/// The reflection `H = I - τ·v·v*`, `v = (1, x)`, with `H*·(alpha, x) = (β, 0)` and `β` real,
/// as LAPACK's `larfg`: overwrites `x` with the rest of `v` and returns `(τ, β)`. `τ` is zero,
/// and `H` the identity, when `x` is zero and `alpha` real.
pub(crate) fn reflector<T: Scalar>(alpha: T, x: &mut [T]) -> (T, T) {
    let xnorm = norm2(x);
    if xnorm == T::Real::ZERO && alpha.im() == T::Real::ZERO {
        return (T::zero(), alpha);
    }
    reflection(alpha, alpha.abs().hypot(xnorm), x)
}

/// The rest of [`reflector`] once the norm of `(alpha, x)` is known, which every reflector here
/// shares: `β = ∓norm`, `τ` and `x / (alpha - β)`, as LAPACK's `larfg`, but without overflowing
/// where the norm is near the largest float. `τ = 1 - alpha/β` rather than `(β - alpha)/β`,
/// whose difference adds two magnitudes (`alpha/β` is not positive, so nothing cancels); and
/// `alpha - β` halved when it overflows, or each element divided when `1/(alpha - β)` would.
pub(crate) fn reflection<T: Scalar>(alpha: T, norm: T::Real, x: &mut [T]) -> (T, T) {
    let beta = if alpha.re() >= T::Real::ZERO {
        -norm
    } else {
        norm
    };
    let tau = T::from_parts(T::Real::ONE - alpha.re() / beta, -alpha.im() / beta);
    let d = alpha - T::from_real(beta);
    if !d.is_finite() {
        let half = alpha.scale(T::Real::HALF) - T::from_real(beta * T::Real::HALF);
        let r = T::one() / half;
        for v in x.iter_mut() {
            *v = (*v * r).scale(T::Real::HALF);
        }
    } else {
        let r = T::one() / d;
        if r.is_finite() {
            for v in x.iter_mut() {
                *v *= r;
            }
        } else {
            for v in x.iter_mut() {
                *v /= d;
            }
        }
    }
    (tau, T::from_real(beta))
}

/// Applies `H = I - τ·v·v*` (`conj` false) or `H*` (`conj` true) from the left to the `m × n`
/// block `c`, `v` of length `m` with its first element taken as 1.
pub(crate) fn reflect_left<T: Scalar>(
    tau: T,
    conj: bool,
    v: &[T],
    m: usize,
    n: usize,
    c: &mut [T],
    ldc: usize,
) {
    if tau == T::zero() {
        return;
    }
    let tau = if conj { tau.conj() } else { tau };
    for j in 0..n {
        let col = &mut c[j * ldc..j * ldc + m];
        // `w = v*·c_j`.
        let mut w = col[0];
        for (x, v) in col[1..].iter().zip(&v[1..m]) {
            w += v.conj() * *x;
        }
        let w = tau * w;
        col[0] -= w;
        for (x, v) in col[1..].iter_mut().zip(&v[1..m]) {
            *x -= *v * w;
        }
    }
}

/// Applies `H = I - τ·v·v*` from the right to the `m × n` block `c`, `v` of length `n` with its
/// first element taken as 1.
pub(crate) fn reflect_right<T: Scalar>(
    tau: T,
    v: &[T],
    m: usize,
    n: usize,
    c: &mut [T],
    ldc: usize,
) {
    if tau == T::zero() {
        return;
    }
    // `w = c·v`, then `c -= τ·w·v*`.
    let mut w: Vec<T> = c[..m].to_vec();
    for p in 1..n {
        let x = v[p];
        for (w, c) in w.iter_mut().zip(&c[p * ldc..p * ldc + m]) {
            *w += *c * x;
        }
    }
    for w in &mut w {
        *w *= tau;
    }
    for (c, w) in c[..m].iter_mut().zip(&w) {
        *c -= *w;
    }
    for p in 1..n {
        let x = v[p].conj();
        for (c, w) in c[p * ldc..p * ldc + m].iter_mut().zip(&w) {
            *c -= *w * x;
        }
    }
}

/// A block of `k` reflections `H₁·H₂·…·H_k = I - V·T·V*`, LAPACK's compact WY form: `V` of
/// `m × k`, unit lower trapezoidal, held with its ones and zeros, and `T` of `k × k` upper
/// triangular.
pub(crate) struct Block<T> {
    /// `V`, `m × k`.
    pub v: Vec<T>,
    /// `T`, `k × k`.
    pub t: Vec<T>,
    /// Rows.
    pub m: usize,
    /// Reflections.
    pub k: usize,
}

impl<T: Scalar> Block<T> {
    /// The block of the reflections whose vectors are below the diagonal of the `m × k` block
    /// `a` (as a QR panel leaves them) and whose factors are `taus`, as LAPACK's `larft`.
    pub fn new(a: &[T], lda: usize, m: usize, taus: &[T]) -> Block<T> {
        let k = taus.len();
        let mut v = vec![T::zero(); m * k];
        for j in 0..k {
            v[j + j * m] = T::one();
            v[j * m + j + 1..(j + 1) * m].copy_from_slice(&a[j * lda + j + 1..j * lda + m]);
        }
        let mut t = vec![T::zero(); k * k];
        for j in 0..k {
            let tau = taus[j];
            t[j + j * k] = tau;
            if j == 0 || tau == T::zero() {
                continue;
            }
            // `T[0..j, j] = -τ·T[0..j, 0..j]·V[:, 0..j]*·v_j`.
            let mut w = vec![T::zero(); j];
            gemm(
                j,
                1,
                m - j,
                -tau,
                &v[j..],
                m,
                Op::C,
                &v[j * m + j..],
                m,
                Op::N,
                T::zero(),
                &mut w,
                j,
            );
            for i in 0..j {
                let mut s = T::zero();
                for (p, w) in w.iter().enumerate().skip(i) {
                    s += t[i + p * k] * *w;
                }
                t[i + j * k] = s;
            }
        }
        Block { v, t, m, k }
    }

    /// Applies the block, or its conjugate transpose when `conj` is set, from the left to the
    /// `m × n` block `c`: `c -= V·op(T)·(V*·c)`.
    pub fn apply_left(&self, conj: bool, n: usize, c: &mut [T], ldc: usize) {
        let (m, k) = (self.m, self.k);
        if n == 0 || k == 0 {
            return;
        }
        let mut w = vec![T::zero(); k * n];
        gemm(
            k,
            n,
            m,
            T::one(),
            &self.v,
            m,
            Op::C,
            c,
            ldc,
            Op::N,
            T::zero(),
            &mut w,
            k,
        );
        let mut tw = vec![T::zero(); k * n];
        let op = if conj { Op::C } else { Op::N };
        gemm(
            k,
            n,
            k,
            T::one(),
            &self.t,
            k,
            op,
            &w,
            k,
            Op::N,
            T::zero(),
            &mut tw,
            k,
        );
        gemm(
            m,
            n,
            k,
            -T::one(),
            &self.v,
            m,
            Op::N,
            &tw,
            k,
            Op::N,
            T::one(),
            c,
            ldc,
        );
    }

    /// Applies the block from the right to the `n × m` block `c`: `c -= (c·V)·T·V*`.
    pub fn apply_right(&self, n: usize, c: &mut [T], ldc: usize) {
        let (m, k) = (self.m, self.k);
        if n == 0 || k == 0 {
            return;
        }
        let mut w = vec![T::zero(); n * k];
        gemm(
            n,
            k,
            m,
            T::one(),
            c,
            ldc,
            Op::N,
            &self.v,
            m,
            Op::N,
            T::zero(),
            &mut w,
            n,
        );
        let mut wt = vec![T::zero(); n * k];
        gemm(
            n,
            k,
            k,
            T::one(),
            &w,
            n,
            Op::N,
            &self.t,
            k,
            Op::N,
            T::zero(),
            &mut wt,
            n,
        );
        gemm(
            n,
            m,
            k,
            -T::one(),
            &wt,
            n,
            Op::N,
            &self.v,
            m,
            Op::C,
            T::one(),
            c,
            ldc,
        );
    }
}

/// The `m × m` (or `m × cols`) matrix `H₁·H₂·…·H_k` applied to the first `cols` columns of the
/// identity, for the reflections below the diagonal of the `m × k` block `a`, as LAPACK's
/// `orgqr`: the explicit `Q` of a QR decomposition.
pub(crate) fn form_q<T: Scalar>(
    a: &[T],
    lda: usize,
    m: usize,
    taus: &[T],
    cols: usize,
) -> Matrix<T> {
    let mut q = Matrix::zeros(m, cols);
    for i in 0..m.min(cols) {
        q.data[i + i * m] = T::one();
    }
    let k = taus.len();
    // The blocks of `PANEL` reflections from the last: each changes only the rows and columns
    // from its first reflection on, the columns before being those of the identity still.
    let mut end = k;
    while end > 0 {
        let start = (end - 1) / PANEL * PANEL;
        let block = Block::new(&a[start + start * lda..], lda, m - start, &taus[start..end]);
        let n = cols - start;
        block.apply_left(false, n, &mut q.data[start + start * m..], m);
        end = start;
    }
    q
}

/// The width of the panels of blocked reductions.
pub(crate) const PANEL: usize = 32;
