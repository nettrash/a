//! The reduction of a matrix to a real upper bidiagonal one, `B = Q*·A·P`, in panels as LAPACK's
//! `gebrd` and `labrd`: the reflections of a panel are applied to the rest of the matrix at once,
//! by two products of blocks, and within the panel the rest is only read, once per column, by
//! a pass that computes both of its products with vectors. The reduction to triangular that
//! comes first for tall matrices, and the products of the reflections with the singular vectors,
//! by blocks.

use super::super::Scalar;
use super::super::blas::{Op, gemm};
use super::super::householder::{Block, norm2, reflection};
use crate::math::complex::Float;

/// The width of the panels of the reduction.
const NB: usize = 16;

/// The reflections applied together to the singular vectors.
const BLOCK: usize = 32;

/// The width of the panels of the reduction to triangular.
const QB: usize = 16;

/// The reduction of an `m × n` matrix, `m ≥ n`, to the real upper bidiagonal `B = Q*·A·P`:
/// `Q = H₀·H₁·…·H_{n-1}` and `P = G₀·G₁·…·G_{n-2}`, products of reflections.
pub(super) struct Bidiagonal<T> {
    /// `m × n`, the vectors of the reflections `H_j` below the diagonal.
    a: Vec<T>,
    /// The rows.
    m: usize,
    /// The columns.
    n: usize,
    /// `n × n`, the vector of `G_j` in column `j` from row `j + 1`, its leading one held.
    w: Vec<T>,
    /// The factors of `H_j`.
    tauq: Vec<T>,
    /// The factors of `G_j`.
    taup: Vec<T>,
}

/// Reduces the `m × n` column-major `a`, `m ≥ n`: the diagonal and the superdiagonal of `B`,
/// and the reflections.
pub(super) fn reduce<T: Scalar>(
    mut a: Vec<T>,
    m: usize,
    n: usize,
) -> (Vec<T::Real>, Vec<T::Real>, Bidiagonal<T>) {
    let mut d = vec![T::Real::ZERO; n];
    let mut e = vec![T::Real::ZERO; n.saturating_sub(1)];
    let mut tauq = vec![T::zero(); n];
    let mut taup = vec![T::zero(); n.saturating_sub(1)];
    let mut w = vec![T::zero(); n * n];
    let nb = NB.min(n);
    // `X` (`m × nb`) and `Y` (`n × nb`) of a panel: the reflections of the panel change the rest
    // of the matrix by `-V·Y* - X·W*`, `V` and `W` the vectors of the left and right ones.
    let mut x = vec![T::zero(); m * nb];
    let mut y = vec![T::zero(); n * nb];
    let mut p = 0;
    while p < n {
        let ib = nb.min(n - p);
        let mut panel = Panel {
            a: &mut a,
            m,
            n,
            p,
            x: &mut x,
            y: &mut y,
            w: &mut w,
        };
        for i in 0..ib {
            let (dq, tq, ep) = panel.step(i);
            d[p + i] = dq;
            tauq[p + i] = tq;
            if let Some((ev, tp)) = ep {
                e[p + i] = ev;
                taup[p + i] = tp;
            }
        }
        let q = p + ib;
        if q < n {
            // The rest: `A₂₂ -= V₂·Y₂* + X₂·W₂*`.
            let (left, right) = a.split_at_mut(q * m);
            let c = &mut right[q..];
            let minus = -T::one();
            gemm(
                m - q,
                n - q,
                ib,
                minus,
                &left[q + p * m..],
                m,
                Op::N,
                &y[q..],
                n,
                Op::C,
                T::one(),
                c,
                m,
            );
            gemm(
                m - q,
                n - q,
                ib,
                minus,
                &x[q..],
                m,
                Op::N,
                &w[q + p * n..],
                n,
                Op::C,
                T::one(),
                c,
                m,
            );
        }
        p = q;
    }
    (
        d,
        e,
        Bidiagonal {
            a,
            m,
            n,
            w,
            tauq,
            taup,
        },
    )
}

/// The state of the reduction of the panel from column `p`.
struct Panel<'a, T> {
    /// The matrix, `m × n`.
    a: &'a mut [T],
    /// The rows.
    m: usize,
    /// The columns.
    n: usize,
    /// The first column of the panel.
    p: usize,
    /// `X`, `m × nb`.
    x: &'a mut [T],
    /// `Y`, `n × nb`.
    y: &'a mut [T],
    /// The vectors of the right reflections, `n × n`.
    w: &'a mut [T],
}

impl<T: Scalar> Panel<'_, T> {
    /// Reduces column and row `c = p + i`: the diagonal element and the factor of the left
    /// reflection, and when there is one, the superdiagonal element and the factor of the right.
    #[allow(clippy::type_complexity)]
    fn step(&mut self, i: usize) -> (T::Real, T, Option<(T::Real, T)>) {
        let (m, n, p) = (self.m, self.n, self.p);
        let c = p + i;
        let ldy = n;
        let mut t = vec![T::zero(); i + 1];
        // Column `c`, rows `c..`: `-= V·conj(Y[c, ..i]) + X·conj(W[c, ..i])`.
        {
            let (left, right) = self.a.split_at_mut(c * m);
            let col = &mut right[c..m];
            if i > 0 {
                for (l, t) in t[..i].iter_mut().enumerate() {
                    *t = -self.y[c + l * ldy].conj();
                }
                gemv_n(m - c, i, &left[c + p * m..], m, &t[..i], col);
                for (l, t) in t[..i].iter_mut().enumerate() {
                    *t = -self.w[c + (p + l) * n].conj();
                }
                gemv_n(m - c, i, &self.x[c..], m, &t[..i], col);
            }
        }
        let (tauq, beta) = {
            let (head, tail) = self.a[c * m + c..(c + 1) * m].split_at_mut(1);
            reflector(head[0], tail)
        };
        self.a[c + c * m] = T::one();
        if c + 1 == n {
            return (beta.re(), tauq, None);
        }
        let nc = n - c - 1;
        let a = &*self.a;
        let v = &a[c + c * m..(c + 1) * m];
        // `t2 = V*·v` and `t3 = X*·v`, for `Y[.., i] = τ·(A*·v - Y·t2 - W·t3)`.
        let mut t2 = vec![T::zero(); i];
        let mut t3 = vec![T::zero(); i];
        gemv_c(m - c, i, &a[c + p * m..], m, v, &mut t2);
        gemv_c(m - c, i, &self.x[c..], m, v, &mut t3);
        let mut corr = vec![T::zero(); nc];
        gemv_n(nc, i, &self.y[c + 1..], ldy, &t2, &mut corr);
        gemv_n(nc, i, &self.w[c + 1 + p * n..], n, &t3, &mut corr);
        // Row `c` conjugated, all but `-Y[.., i]`: `conj(A[c, ..]) - Y·conj(V[c, ..]) -
        // W·conj(X[c, ..])`.
        let mut r: Vec<T> = (0..nc).map(|j| a[c + (c + 1 + j) * m].conj()).collect();
        for (l, t) in t[..i].iter_mut().enumerate() {
            *t = -a[c + (p + l) * m].conj();
        }
        gemv_n(nc, i, &self.y[c + 1..], ldy, &t[..i], &mut r);
        for (l, t) in t[..i].iter_mut().enumerate() {
            *t = -self.x[c + l * m].conj();
        }
        gemv_n(nc, i, &self.w[c + 1 + p * n..], n, &t[..i], &mut r);
        // One pass over the rest: `Y[.., i]`, the row, and `z = A₂₂·r` with it.
        let mut z = vec![T::zero(); m - c - 1];
        let yi = &mut self.y[c + 1 + i * ldy..c + 1 + i * ldy + nc];
        fused(
            &a[c + (c + 1) * m..],
            m,
            m - c,
            nc,
            v,
            tauq,
            &corr,
            yi,
            &mut r,
            &mut z,
        );
        let alpha = r[0];
        let (taup, beta2) = {
            let (head, tail) = r.split_at_mut(1);
            reflector(head[0], tail)
        };
        let wc = &mut self.w[c + 1 + c * n..(c + 1) * n];
        wc[0] = T::one();
        wc[1..].copy_from_slice(&r[1..]);
        // `X[.., i] = τ·(A·w - V·(Y*·w) - X·(W*·w))`, `A·w = (z - β·A[.., c + 1])/(α - β)`.
        let xi = c + 1 + i * m;
        if taup == T::zero() {
            self.x[xi..xi + m - c - 1].fill(T::zero());
        } else {
            let s = T::one() / (alpha - beta2);
            let a1 = &a[c + 1 + (c + 1) * m..(c + 2) * m];
            let mut xc: Vec<T> = z
                .iter()
                .zip(a1)
                .map(|(&z, &a)| (z - a * beta2) * s)
                .collect();
            let wc = &self.w[c + 1 + c * n..(c + 1) * n];
            let mut t4 = vec![T::zero(); i + 1];
            let mut t5 = vec![T::zero(); i];
            gemv_c(nc, i + 1, &self.y[c + 1..], ldy, wc, &mut t4);
            gemv_c(nc, i, &self.w[c + 1 + p * n..], n, wc, &mut t5);
            for t in t4.iter_mut().chain(t5.iter_mut()) {
                *t = -*t;
            }
            gemv_n(m - c - 1, i + 1, &a[c + 1 + p * m..], m, &t4, &mut xc);
            gemv_n(m - c - 1, i, &self.x[c + 1..], m, &t5, &mut xc);
            for (x, &v) in self.x[xi..xi + m - c - 1].iter_mut().zip(&xc) {
                *x = v * taup;
            }
        }
        (beta.re(), tauq, Some((beta2.re(), taup)))
    }
}

/// Reduces the `m × n` column-major `a`, `m ≥ n`, to upper triangular by Householder
/// reflections, as LAPACK's `geqrf`: `R` on and above the diagonal, the vectors below it, and the
/// factors returned. Panels of `QB` columns are reduced with the products by vectors of the
/// reduction to bidiagonal, and their reflections applied to the rest of the matrix in blocks.
pub(super) fn triangularize<T: Scalar>(a: &mut [T], m: usize, n: usize) -> Vec<T> {
    let mut taus = vec![T::zero(); n];
    let mut p = 0;
    while p < n {
        let q = (p + QB).min(n);
        for j in p..q {
            let (tau, beta) = {
                let (head, tail) = a[j * m + j..(j + 1) * m].split_at_mut(1);
                reflector(head[0], tail)
            };
            a[j + j * m] = beta;
            taus[j] = tau;
            if tau == T::zero() || j + 1 == q {
                continue;
            }
            // The rest of the panel: `c -= conj(τ)·v·(v*·c)`, two columns at a time.
            let ct = tau.conj();
            let (left, right) = a.split_at_mut((j + 1) * m);
            let v = &left[j * m + j + 1..];
            let mut cols = right[j..].chunks_mut(m).take(q - j - 1);
            loop {
                match (cols.next(), cols.next()) {
                    (Some(c0), Some(c1)) => {
                        let (s0, s1) = dot2_c(&c0[1..m - j], &c1[1..m - j], v);
                        let w0 = ct * (c0[0] + s0.conj());
                        let w1 = ct * (c1[0] + s1.conj());
                        c0[0] -= w0;
                        c1[0] -= w1;
                        let it = c0[1..m - j].iter_mut().zip(c1[1..m - j].iter_mut());
                        for ((x0, x1), &v) in it.zip(v) {
                            *x0 -= v * w0;
                            *x1 -= v * w1;
                        }
                    }
                    (Some(c0), None) => {
                        let w0 = ct * (c0[0] + dot_c(&c0[1..m - j], v).conj());
                        c0[0] -= w0;
                        for (x0, &v) in c0[1..m - j].iter_mut().zip(v) {
                            *x0 -= v * w0;
                        }
                        break;
                    }
                    _ => break,
                }
            }
        }
        if q < n {
            let block = Block::new(&a[p + p * m..], m, m - p, &taus[p..q]);
            block.apply_left(true, n - q, &mut a[p + q * m..], m);
        }
        p = q;
    }
    taus
}

/// The reflection `H = I - τ·v·v*`, `v = (1, x)`, with `H*·(alpha, x) = (β, 0)` and `β` real, as
/// [`reflector`](super::super::householder::reflector) but with the norm of `x` from one pass of
/// sums of squares when they can neither overflow nor underflow.
pub(super) fn reflector<T: Scalar>(alpha: T, x: &mut [T]) -> (T, T) {
    let xnorm = norm(x);
    if xnorm == T::Real::ZERO && alpha.im() == T::Real::ZERO {
        return (T::zero(), alpha);
    }
    reflection(alpha, alpha.abs().hypot(xnorm), x)
}

/// The 2-norm of `x`: the root of the sum of the squares in eight sums, or [`norm2`]'s scaled
/// one when the sum is out of the range where no square can have overflowed or underflowed
/// enough to matter, zero included.
fn norm<T: Scalar>(x: &[T]) -> T::Real {
    let (chunks, rest) = x.as_chunks::<8>();
    let mut s = [T::Real::ZERO; 8];
    for c in chunks {
        for l in 0..8 {
            s[l] += c[l].abs2();
        }
    }
    let mut t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for v in rest {
        t += v.abs2();
    }
    let small = T::Real::MIN_POSITIVE / T::Real::EPSILON;
    if t >= small && t <= T::Real::MAX * T::Real::EPSILON {
        t.sqrt()
    } else {
        norm2(x)
    }
}

/// `y += A·x`, `A` of `rows × cols` with the leading dimension `lda`.
fn gemv_n<T: Scalar>(rows: usize, cols: usize, a: &[T], lda: usize, x: &[T], y: &mut [T]) {
    let y = &mut y[..rows];
    let mut j = 0;
    while j + 4 <= cols {
        let (x0, x1, x2, x3) = (x[j], x[j + 1], x[j + 2], x[j + 3]);
        let a0 = &a[j * lda..j * lda + rows];
        let a1 = &a[(j + 1) * lda..(j + 1) * lda + rows];
        let a2 = &a[(j + 2) * lda..(j + 2) * lda + rows];
        let a3 = &a[(j + 3) * lda..(j + 3) * lda + rows];
        for ((((y, &a0), &a1), &a2), &a3) in y.iter_mut().zip(a0).zip(a1).zip(a2).zip(a3) {
            *y += a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3;
        }
        j += 4;
    }
    while j < cols {
        let x0 = x[j];
        for (y, &a0) in y.iter_mut().zip(&a[j * lda..j * lda + rows]) {
            *y += a0 * x0;
        }
        j += 1;
    }
}

/// `y = A*·x`, `A` of `rows × cols` with the leading dimension `lda`.
fn gemv_c<T: Scalar>(rows: usize, cols: usize, a: &[T], lda: usize, x: &[T], y: &mut [T]) {
    for (j, y) in y[..cols].iter_mut().enumerate() {
        *y = dot_c(&a[j * lda..j * lda + rows], &x[..rows]);
    }
}

/// `a*·x`, in eight sums, which the compiler keeps in vector registers.
#[inline]
fn dot_c<T: Scalar>(a: &[T], x: &[T]) -> T {
    let (xc, xr) = x.as_chunks::<8>();
    let (ac, ar) = a[..x.len()].as_chunks::<8>();
    let mut s = [T::zero(); 8];
    for (a, x) in ac.iter().zip(xc) {
        for l in 0..8 {
            s[l] += a[l].conj() * x[l];
        }
    }
    let mut t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (a, x) in ar.iter().zip(xr) {
        t += a.conj() * *x;
    }
    t
}

/// `(a*·x, b*·x)`, each in eight sums, the loads of `x` shared.
#[inline]
fn dot2_c<T: Scalar>(a: &[T], b: &[T], x: &[T]) -> (T, T) {
    let n = x.len();
    let (xc, xr) = x.as_chunks::<8>();
    let (ac, ar) = a[..n].as_chunks::<8>();
    let (bc, br) = b[..n].as_chunks::<8>();
    let (mut s, mut t) = ([T::zero(); 8], [T::zero(); 8]);
    for ((a, b), x) in ac.iter().zip(bc).zip(xc) {
        for l in 0..8 {
            s[l] += a[l].conj() * x[l];
        }
        for l in 0..8 {
            t[l] += b[l].conj() * x[l];
        }
    }
    let mut u = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    let mut w = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    for ((a, b), x) in ar.iter().zip(br).zip(xr) {
        u += a.conj() * *x;
        w += b.conj() * *x;
    }
    (u, w)
}

/// The pass over the rest of the matrix, `A` of `rows × cols` from row and column `c` (its
/// first column `c + 1`): for each column `j`, `y_j = τ·(A[.., j]*·v - corr_j)`,
/// `r_j -= y_j`, then `z += A[1.., j]·r_j`.
#[allow(clippy::too_many_arguments)]
fn fused<T: Scalar>(
    a: &[T],
    lda: usize,
    rows: usize,
    cols: usize,
    v: &[T],
    tau: T,
    corr: &[T],
    y: &mut [T],
    r: &mut [T],
    z: &mut [T],
) {
    let v = &v[..rows];
    let z = &mut z[..rows - 1];
    let mut j = 0;
    while j + 4 <= cols {
        let c0 = &a[j * lda..j * lda + rows];
        let c1 = &a[(j + 1) * lda..(j + 1) * lda + rows];
        let c2 = &a[(j + 2) * lda..(j + 2) * lda + rows];
        let c3 = &a[(j + 3) * lda..(j + 3) * lda + rows];
        let ((s0, s1), (s2, s3)) = (dot2_c(c0, c1, v), dot2_c(c2, c3, v));
        let s = [s0, s1, s2, s3];
        let mut rr = [T::zero(); 4];
        for q in 0..4 {
            let yj = tau * (s[q] - corr[j + q]);
            y[j + q] = yj;
            r[j + q] -= yj;
            rr[q] = r[j + q];
        }
        let it = z
            .iter_mut()
            .zip(&c0[1..])
            .zip(&c1[1..])
            .zip(&c2[1..])
            .zip(&c3[1..]);
        for ((((z, &a0), &a1), &a2), &a3) in it {
            *z += (a0 * rr[0] + a1 * rr[1]) + (a2 * rr[2] + a3 * rr[3]);
        }
        j += 4;
    }
    while j < cols {
        let c0 = &a[j * lda..j * lda + rows];
        let yj = tau * (dot_c(c0, v) - corr[j]);
        y[j] = yj;
        r[j] -= yj;
        let rj = r[j];
        for (z, &a0) in z.iter_mut().zip(&c0[1..]) {
            *z += a0 * rj;
        }
        j += 1;
    }
}

impl<T: Scalar> Bidiagonal<T> {
    /// `x = x·Q*`, `x` of `rows × m`.
    pub fn apply_q_right(&self, x: &mut [T], rows: usize) {
        apply_adjoint_right(&self.a, self.m, self.m, &self.tauq, x, rows);
    }

    /// `x = x·P*`, `x` of `rows × n`.
    pub fn apply_p_right(&self, x: &mut [T], rows: usize) {
        let n = self.n;
        if n > 1 {
            apply_adjoint_right(&self.w[1..], n, n - 1, &self.taup, &mut x[rows..], rows);
        }
    }
}

/// `x = x·(H₀·H₁·…·H_{k-1})*` for the reflections whose vectors are below the diagonal of the
/// `m × k` block `a` and whose factors are `taus`, `x` of `rows × m`: by blocks of `BLOCK`
/// reflections from the last, `x·(I - V·T·V*)* = x - (x·V)·T*·V*`, whose products all take their
/// left operand as it is held.
pub(super) fn apply_adjoint_right<T: Scalar>(
    a: &[T],
    lda: usize,
    m: usize,
    taus: &[T],
    x: &mut [T],
    rows: usize,
) {
    let k = taus.len();
    let mut end = k;
    while end > 0 {
        let start = (end - 1) / BLOCK * BLOCK;
        let mut block = Block::new(&a[start + start * lda..], lda, m - start, &taus[start..end]);
        let b = block.k;
        let t = block.t.clone();
        for i in 0..b {
            for j in 0..b {
                block.t[i + j * b] = t[j + i * b].conj();
            }
        }
        block.apply_right(rows, &mut x[start * rows..], rows);
        end = start;
    }
}
