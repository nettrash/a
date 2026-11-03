//! The eigenvectors of a matrix from its Schur form `A = Q·T·Q*`: those of `T` by back
//! substitution, as LAPACK's `trevc`, then `Q` times them by products of blocks. For real
//! elements the substitution stays in real arithmetic on the quasi-triangular `T`, a pair of
//! complex conjugate eigenvectors held as its real and imaginary parts. The substitution is
//! blocked: a block of eigenvectors at a time, solved a block of rows at a time, the solved
//! rows' share of the rows above one product of blocks; a vector that needs scaling against
//! overflow is solved again alone, as LAPACK does it.

use super::super::Scalar;
use super::super::blas::{Op, gemm};
use super::super::householder::norm2;
use crate::math::complex::{Complex, Float};

/// The columns of a block of the product by `Q`.
const BLOCK: usize = 64;

/// The eigenvectors solved together, and the rows of a block of their substitution.
const SOLVE: usize = 64;

/// The unit eigenvectors of `A = Q·T·Q*`, as the columns of an `n × n` matrix in the order of
/// `T`'s diagonal, given `T` (the real Schur form, with standard `2 × 2` blocks, for real
/// elements) and `Q`, column-major. A pair of complex conjugate eigenvalues of a real matrix has
/// the conjugate eigenvectors, the one of the positive imaginary part first.
pub(crate) fn eigenvectors<T: Scalar>(t: &[T], q: &[T], n: usize) -> Vec<Complex<T::Real>> {
    let zero = T::Real::ZERO;
    if t.iter().chain(q).any(|x| !x.is_finite()) {
        return vec![Complex::new(T::Real::NAN, T::Real::NAN); n * n];
    }
    let x = Triangle::new(t, n).solve();
    let v = times_q(q, &x, n);
    let column = |j: usize| &v[j * n..(j + 1) * n];
    let mut out = vec![Complex::new(zero, zero); n * n];
    let mut k = 0;
    while k < n {
        if second(t, n, k + 1) {
            let (re, im) = (column(k), column(k + 1));
            let norm = norm2(re).hypot(norm2(im));
            let r = if norm > zero {
                T::Real::ONE / norm
            } else {
                T::Real::ONE
            };
            for i in 0..n {
                let (a, b) = (re[i].re() * r, im[i].re() * r);
                out[i + k * n] = Complex::new(a, b);
                out[i + (k + 1) * n] = Complex::new(a, -b);
            }
            k += 2;
        } else {
            let norm = norm2(column(k));
            let r = if norm > zero {
                T::Real::ONE / norm
            } else {
                T::Real::ONE
            };
            for (o, v) in out[k * n..(k + 1) * n].iter_mut().zip(column(k)) {
                *o = v.scale(r).to_complex();
            }
            k += 1;
        }
    }
    out
}

/// `Q·X` for the upper triangular `X`, a block of columns at a time, each the product by the
/// columns of `Q` it needs.
fn times_q<T: Scalar>(q: &[T], x: &[T], n: usize) -> Vec<T> {
    let mut v = vec![T::zero(); n * n];
    let mut c0 = 0;
    while c0 < n {
        let c1 = (c0 + BLOCK).min(n);
        let (xs, vs) = (&x[c0 * n..], &mut v[c0 * n..]);
        gemm(
            n,
            c1 - c0,
            c1,
            T::one(),
            q,
            n,
            Op::N,
            xs,
            n,
            Op::N,
            T::zero(),
            vs,
            n,
        );
        c0 = c1;
    }
    v
}

/// Whether row `k` of the real Schur form `t` is the second of a `2 × 2` block, never for
/// complex elements.
fn second<T: Scalar>(t: &[T], n: usize, k: usize) -> bool {
    !T::IS_COMPLEX && k > 0 && k < n && t[k + (k - 1) * n] != T::zero()
}

/// The (quasi-)triangular `T` of the Schur form, with what its back substitutions share.
struct Triangle<'a, T: Scalar> {
    /// `T`, column-major.
    t: &'a [T],
    /// The order.
    n: usize,
    /// The 1-norms of the columns above the diagonal, bounding the growth of an update.
    norms: Vec<T::Real>,
    /// The unit roundoff.
    ulp: T::Real,
    /// The smallest pivot, `n` units of roundoff above underflow.
    smlnum: T::Real,
    /// The largest element a vector may reach before it is scaled down.
    bignum: T::Real,
}

impl<'a, T: Scalar> Triangle<'a, T> {
    /// The triangle `t` of order `n`.
    fn new(t: &'a [T], n: usize) -> Self {
        let ulp = T::Real::EPSILON;
        let smlnum = T::Real::MIN_POSITIVE * (T::Real::from_f64(n.max(1) as f64) / ulp);
        let norms = (0..n)
            .map(|j| {
                t[j * n..j * n + j]
                    .iter()
                    .fold(T::Real::ZERO, |s, x| s + x.abs1())
            })
            .collect();
        Triangle {
            t,
            n,
            norms,
            ulp,
            smlnum,
            bignum: (T::Real::ONE - ulp) / smlnum,
        }
    }

    /// Element `(i, j)` of `T`.
    #[inline(always)]
    fn at(&self, i: usize, j: usize) -> T {
        self.t[i + j * self.n]
    }

    /// The columns of the eigenvalue at `k`: 2 for a pair of complex ones of a real matrix.
    fn width(&self, k: usize) -> usize {
        if second(self.t, self.n, k + 1) { 2 } else { 1 }
    }

    /// The eigenvectors of `T` as LAPACK's `trevc`: in column `k` for an eigenvalue, but for a
    /// pair of complex ones of a real matrix at `k, k + 1` the real part of the first one's in
    /// column `k` and the imaginary part in column `k + 1`; upper triangular, each vector
    /// scaled to a largest element of 1, `|re| + |im|` for the complex ones of a real matrix.
    fn solve(&self) -> Vec<T> {
        let (n, zero) = (self.n, T::Real::ZERO);
        let mut x = vec![T::zero(); n * n];
        let mut redo = vec![false; n];
        let mut c0 = 0;
        while c0 < n {
            let mut c1 = (c0 + SOLVE).min(n);
            if second(self.t, n, c1) {
                c1 += 1;
            }
            let mut k = c0;
            while k < c1 {
                self.start(&mut x, k, c0);
                redo[k] = !self.rows(&mut x, k, c0, k, false);
                k += self.width(k);
            }
            // The rows above, a block at a time: the solved rows' share in one product, then
            // the block's own substitution.
            let mut hi = c0;
            while hi > 0 {
                let mut lo = hi.saturating_sub(SOLVE);
                if second(self.t, n, lo) {
                    lo -= 1;
                }
                let (rows, cols) = (hi - lo, c1 - c0);
                let mut part = vec![T::zero(); rows * cols];
                for j in 0..cols {
                    part[j * rows..(j + 1) * rows]
                        .copy_from_slice(&x[lo + (c0 + j) * n..hi + (c0 + j) * n]);
                }
                let (tb, xb) = (&self.t[lo + hi * n..], &x[hi + c0 * n..]);
                gemm(
                    rows,
                    cols,
                    c1 - hi,
                    -T::one(),
                    tb,
                    n,
                    Op::N,
                    xb,
                    n,
                    Op::N,
                    T::one(),
                    &mut part,
                    rows,
                );
                for j in 0..cols {
                    x[lo + (c0 + j) * n..hi + (c0 + j) * n]
                        .copy_from_slice(&part[j * rows..(j + 1) * rows]);
                }
                let mut k = c0;
                while k < c1 {
                    if !redo[k] {
                        redo[k] = !self.rows(&mut x, k, lo, hi, false);
                    }
                    k += self.width(k);
                }
                hi = lo;
            }
            c0 = c1;
        }
        // The vectors that needed scaling, or overflowed in the products, solved again with
        // the scaling; then each scaled to a largest element of 1.
        let mut k = 0;
        while k < n {
            let w = self.width(k);
            let cols = k * n..(k + w) * n;
            let largest = |x: &[T]| {
                (0..=k + w - 1).fold(zero, |m, i| {
                    let s = if w == 2 {
                        x[i].abs1() + x[i + n].abs1()
                    } else {
                        x[i].abs1()
                    };
                    if s.is_finite() {
                        m.max(s)
                    } else {
                        T::Real::INFINITY
                    }
                })
            };
            if redo[k] || largest(&x[cols.clone()]) > self.bignum {
                x[cols.clone()].fill(T::zero());
                self.start(&mut x, k, 0);
                self.rows(&mut x, k, 0, k, true);
            }
            let m = largest(&x[cols.clone()]);
            if m > zero && m.is_finite() {
                let r = T::Real::ONE / m;
                for v in &mut x[cols] {
                    *v = v.scale(r);
                }
            }
            k += w;
        }
        x
    }

    /// The eigenvalue at `k`: `(wr, wi)`, `wi` positive for a pair of a real matrix, and the
    /// smallest pivot of its substitution.
    fn shift(&self, k: usize) -> (T, T::Real, T::Real) {
        let lambda = self.at(k, k);
        if self.width(k) == 2 {
            let p = k + 1;
            let wr = self.at(p, p).re();
            let wi = self.at(p, k).re().abs().sqrt() * self.at(k, p).re().abs().sqrt();
            (lambda, wi, (self.ulp * (wr.abs() + wi)).max(self.smlnum))
        } else {
            (
                lambda,
                T::Real::ZERO,
                (self.ulp * lambda.abs1()).max(self.smlnum),
            )
        }
    }

    /// Sets the columns of the eigenvalue at `k` to its part on the diagonal, 1 for a single
    /// one, and in rows `from..` above it the right-hand side of the substitution, that part's
    /// share; the rows above `from` get it with the rest of the solved rows' shares.
    fn start(&self, x: &mut [T], k: usize, from: usize) {
        let n = self.n;
        if self.width(k) == 1 {
            x[k + k * n] = T::one();
            for i in from..k {
                x[i + k * n] = -self.at(i, k);
            }
            return;
        }
        // `x = re + i·im` for `λ = wr + i·wi`: `(T_b - λ·I)·(x_k, x_p) = 0` on the block.
        let p = k + 1;
        let (_, wi, _) = self.shift(k);
        let (up, down) = (self.at(k, p).re(), self.at(p, k).re());
        let (rk, ip) = if up.abs() >= down.abs() {
            (T::Real::ONE, wi / up)
        } else {
            (-wi / down, T::Real::ONE)
        };
        let (re, im) = x[k * n..].split_at_mut(n);
        re[k] = T::from_real(rk);
        re[p] = T::zero();
        im[k] = T::zero();
        im[p] = T::from_real(ip);
        for i in from..k {
            re[i] = self.at(i, k).scale(-rk);
            im[i] = self.at(i, p).scale(-ip);
        }
    }

    /// The substitution of rows `lo..hi` of the columns of the eigenvalue at `k`, the rows
    /// below already solved and their share subtracted: `(T - λ·I)·x = b` a diagonal block of
    /// `T` at a time, from the bottom, each block's solution taken from the rows above it. When
    /// the solution grows near overflow, the vector is scaled down if `careful`, or else the
    /// substitution stops and gives false.
    fn rows(&self, x: &mut [T], k: usize, lo: usize, hi: usize, careful: bool) -> bool {
        let (n, one) = (self.n, T::Real::ONE);
        let (lambda, wi, smin) = self.shift(k);
        let w = self.width(k);
        let (head, tail) = x[k * n..].split_at_mut(n);
        let (re, mut im) = (head, if w == 2 { Some(&mut tail[..n]) } else { None });
        let last = k + w - 1;
        let mut j = hi;
        while j > lo {
            let j2 = j - 1;
            let j1 = if j2 > lo && second(self.t, n, j2) {
                j2 - 1
            } else {
                j2
            };
            let nb = j2 - j1 + 1;
            if T::IS_COMPLEX {
                let mut d = self.at(j2, j2) - lambda;
                if d.abs1() < smin {
                    d = T::from_real(smin);
                }
                let xi = re[j2] / d;
                re[j2] = xi;
                let size = xi.abs1();
                if size > self.bignum {
                    if !careful {
                        return false;
                    }
                    let r = one / size;
                    for v in &mut re[..=last] {
                        *v = v.scale(r);
                    }
                }
            } else {
                let g = |i: usize, c: usize| self.at(i, c).re();
                let a = [[g(j1, j1), g(j1, j2)], [g(j2, j1), g(j2, j2)]];
                let b = match &im {
                    Some(im) => [[re[j1].re(), im[j1].re()], [re[j2].re(), im[j2].re()]],
                    None => [[re[j1].re(), T::Real::ZERO], [re[j2].re(), T::Real::ZERO]],
                };
                let (mut s, mut scale, xnorm) = small_solve(nb, w, smin, a, b, lambda.re(), wi);
                if xnorm > one && self.norms[j1].max(self.norms[j2]) > self.bignum / xnorm {
                    let r = one / xnorm;
                    for v in s.iter_mut().flatten() {
                        *v *= r;
                    }
                    scale *= r;
                }
                if scale != one {
                    if !careful {
                        return false;
                    }
                    for v in re[..=last]
                        .iter_mut()
                        .chain(im.iter_mut().flat_map(|im| im[..=last].iter_mut()))
                    {
                        *v = v.scale(scale);
                    }
                }
                for (p, row) in [j1, j2].into_iter().enumerate().take(nb) {
                    re[row] = T::from_real(s[p][0]);
                    if let Some(im) = &mut im {
                        im[row] = T::from_real(s[p][1]);
                    }
                }
            }
            for row in j1..=j2 {
                let column = &self.t[lo + row * n..j1 + row * n];
                let f = re[row];
                for (x, t) in re[lo..j1].iter_mut().zip(column) {
                    *x -= *t * f;
                }
                if let Some(im) = &mut im {
                    let f = im[row];
                    for (x, t) in im[lo..j1].iter_mut().zip(column) {
                        *x -= *t * f;
                    }
                }
            }
            j = j1;
        }
        true
    }
}

/// `(a + i·b) / (c + i·d)` by Smith's algorithm, which doesn't overflow needlessly.
fn divide<R: Float>(a: R, b: R, c: R, d: R) -> (R, R) {
    if d.abs() <= c.abs() {
        let e = d / c;
        let f = c + d * e;
        ((a + b * e) / f, (b - a * e) / f)
    } else {
        let e = c / d;
        let f = d + c * e;
        ((b + a * e) / f, (-a + b * e) / f)
    }
}

/// The solution of `(A - w·I)·X = scale·B` for the real `A` of order `na` (1 or 2) and
/// `w = wr + i·wi`, `X` and `B` with `nw` columns (1, or 2 for the real and imaginary parts), as
/// LAPACK's `laln2`: Gaussian elimination with complete pivoting, a pivot below `smin` replaced
/// by it, and `scale ≤ 1` keeping `X` from overflowing. Returns `X`, `scale` and the largest
/// row sum of `|X|`.
#[allow(clippy::type_complexity)]
fn small_solve<R: Float>(
    na: usize,
    nw: usize,
    smin: R,
    a: [[R; 2]; 2],
    b: [[R; 2]; 2],
    wr: R,
    wi: R,
) -> ([[R; 2]; 2], R, R) {
    let (zero, one) = (R::ZERO, R::ONE);
    let smlnum = R::TWO * R::MIN_POSITIVE;
    let bignum = one / smlnum;
    let smini = smin.max(smlnum);
    let mut scale = one;
    let mut x = [[zero; 2]; 2];
    if na == 1 {
        if nw == 1 {
            let mut csr = a[0][0] - wr;
            let mut cnorm = csr.abs();
            if cnorm < smini {
                csr = smini;
                cnorm = smini;
            }
            let bnorm = b[0][0].abs();
            if cnorm < one && bnorm > one && bnorm > bignum * cnorm {
                scale = one / bnorm;
            }
            x[0][0] = (b[0][0] * scale) / csr;
            return (x, scale, x[0][0].abs());
        }
        let (mut csr, mut csi) = (a[0][0] - wr, -wi);
        let mut cnorm = csr.abs() + csi.abs();
        if cnorm < smini {
            csr = smini;
            csi = zero;
            cnorm = smini;
        }
        let bnorm = b[0][0].abs() + b[0][1].abs();
        if cnorm < one && bnorm > one && bnorm > bignum * cnorm {
            scale = one / bnorm;
        }
        (x[0][0], x[0][1]) = divide(scale * b[0][0], scale * b[0][1], csr, csi);
        return (x, scale, x[0][0].abs() + x[0][1].abs());
    }
    // `C = A - wr·I`, column after column, and the order of the pivoting for each choice of
    // the first pivot.
    let cr = [a[0][0] - wr, a[1][0], a[0][1], a[1][1] - wr];
    const PIVOT: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];
    const RSWAP: [bool; 4] = [false, true, false, true];
    const ZSWAP: [bool; 4] = [false, false, true, true];
    if nw == 1 {
        let mut cmax = zero;
        let mut icmax = 0;
        for (j, c) in cr.iter().enumerate() {
            if c.abs() > cmax {
                cmax = c.abs();
                icmax = j;
            }
        }
        if cmax < smini {
            let bnorm = b[0][0].abs().max(b[1][0].abs());
            if smini < one && bnorm > one && bnorm > bignum * smini {
                scale = one / bnorm;
            }
            let temp = scale / smini;
            x[0][0] = temp * b[0][0];
            x[1][0] = temp * b[1][0];
            return (x, scale, temp * bnorm);
        }
        let p = PIVOT[icmax];
        let ur11 = cr[icmax];
        let (cr21, ur12, cr22) = (cr[p[1]], cr[p[2]], cr[p[3]]);
        let ur11r = one / ur11;
        let lr21 = ur11r * cr21;
        let mut ur22 = cr22 - ur12 * lr21;
        if ur22.abs() < smini {
            ur22 = smini;
        }
        let (br1, mut br2) = if RSWAP[icmax] {
            (b[1][0], b[0][0])
        } else {
            (b[0][0], b[1][0])
        };
        br2 -= lr21 * br1;
        let bbnd = (br1 * (ur22 * ur11r)).abs().max(br2.abs());
        if bbnd > one && ur22.abs() < one && bbnd >= bignum * ur22.abs() {
            scale = one / bbnd;
        }
        let xr2 = (br2 * scale) / ur22;
        let xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12);
        if ZSWAP[icmax] {
            x[0][0] = xr2;
            x[1][0] = xr1;
        } else {
            x[0][0] = xr1;
            x[1][0] = xr2;
        }
        let mut xnorm = xr1.abs().max(xr2.abs());
        if xnorm > one && cmax > one && xnorm > bignum / cmax {
            let temp = cmax / bignum;
            x[0][0] *= temp;
            x[1][0] *= temp;
            xnorm *= temp;
            scale *= temp;
        }
        return (x, scale, xnorm);
    }
    let ci = [-wi, zero, zero, -wi];
    let mut cmax = zero;
    let mut icmax = 0;
    for j in 0..4 {
        if cr[j].abs() + ci[j].abs() > cmax {
            cmax = cr[j].abs() + ci[j].abs();
            icmax = j;
        }
    }
    if cmax < smini {
        let bnorm = (b[0][0].abs() + b[0][1].abs()).max(b[1][0].abs() + b[1][1].abs());
        if smini < one && bnorm > one && bnorm > bignum * smini {
            scale = one / bnorm;
        }
        let temp = scale / smini;
        for (xr, br) in x.iter_mut().zip(&b) {
            xr[0] = temp * br[0];
            xr[1] = temp * br[1];
        }
        return (x, scale, temp * bnorm);
    }
    let p = PIVOT[icmax];
    let (ur11, ui11) = (cr[icmax], ci[icmax]);
    let (cr21, ci21) = (cr[p[1]], ci[p[1]]);
    let (ur12, ui12) = (cr[p[2]], ci[p[2]]);
    let (cr22, ci22) = (cr[p[3]], ci[p[3]]);
    let (ur11r, ui11r, lr21, li21, ur12s, ui12s, mut ur22, mut ui22);
    if icmax == 0 || icmax == 3 {
        // The off-diagonal elements of the pivoted `C` are real.
        if ur11.abs() > ui11.abs() {
            let temp = ui11 / ur11;
            ur11r = one / (ur11 * (one + temp * temp));
            ui11r = -temp * ur11r;
        } else {
            let temp = ur11 / ui11;
            ui11r = -one / (ui11 * (one + temp * temp));
            ur11r = -temp * ui11r;
        }
        lr21 = cr21 * ur11r;
        li21 = cr21 * ui11r;
        ur12s = ur12 * ur11r;
        ui12s = ur12 * ui11r;
        ur22 = cr22 - ur12 * lr21;
        ui22 = ci22 - ur12 * li21;
    } else {
        // The diagonal elements of the pivoted `C` are real.
        ur11r = one / ur11;
        ui11r = zero;
        lr21 = cr21 * ur11r;
        li21 = ci21 * ur11r;
        ur12s = ur12 * ur11r;
        ui12s = ui12 * ur11r;
        ur22 = cr22 - ur12 * lr21 + ui12 * li21;
        ui22 = -ur12 * li21 - ui12 * lr21;
    }
    let u22abs = ur22.abs() + ui22.abs();
    if u22abs < smini {
        ur22 = smini;
        ui22 = zero;
    }
    let (mut br1, mut bi1, mut br2, mut bi2) = if RSWAP[icmax] {
        (b[1][0], b[1][1], b[0][0], b[0][1])
    } else {
        (b[0][0], b[0][1], b[1][0], b[1][1])
    };
    br2 = br2 - lr21 * br1 + li21 * bi1;
    bi2 = bi2 - li21 * br1 - lr21 * bi1;
    let bbnd = ((br1.abs() + bi1.abs()) * (u22abs * (ur11r.abs() + ui11r.abs())))
        .max(br2.abs() + bi2.abs());
    if bbnd > one && u22abs < one && bbnd >= bignum * u22abs {
        scale = one / bbnd;
        br1 *= scale;
        bi1 *= scale;
        br2 *= scale;
        bi2 *= scale;
    }
    let (xr2, xi2) = divide(br2, bi2, ur22, ui22);
    let xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
    let xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
    if ZSWAP[icmax] {
        x = [[xr2, xi2], [xr1, xi1]];
    } else {
        x = [[xr1, xi1], [xr2, xi2]];
    }
    let mut xnorm = (xr1.abs() + xi1.abs()).max(xr2.abs() + xi2.abs());
    if xnorm > one && cmax > one && xnorm > bignum / cmax {
        let temp = cmax / bignum;
        for v in x.iter_mut().flatten() {
            *v *= temp;
        }
        xnorm *= temp;
        scale *= temp;
    }
    (x, scale, xnorm)
}
