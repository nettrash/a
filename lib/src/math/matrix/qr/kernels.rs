//! The vector kernels of the unblocked steps of the QR decomposition: a column's reflection, its
//! norm from one pass of sums of squares, and the products of a reflection's vector with a few
//! columns at a time, in sums of lanes that the compiler keeps in vectors.

use super::super::Scalar;
use super::super::householder::norm2;
use crate::math::complex::Float;

/// Whether the target fuses multiply-adds in hardware; elsewhere `mul_add` is a library call.
const FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `c + a·b`, rounded once where the target fuses multiply-adds.
#[inline(always)]
fn fma<T: Scalar>(a: T, b: T, c: T) -> T {
    if !FMA {
        c + a * b
    } else if T::IS_COMPLEX {
        let (ar, ai, br, bi) = (a.re(), a.im(), b.re(), b.im());
        let re = (-ai).mul_add(bi, ar.mul_add(br, c.re()));
        let im = ai.mul_add(br, ar.mul_add(bi, c.im()));
        T::from_parts(re, im)
    } else {
        T::from_real(a.re().mul_add(b.re(), c.re()))
    }
}

/// `c + conj(a)·b`, rounded once where the target fuses multiply-adds.
#[inline(always)]
fn fma_conj<T: Scalar>(a: T, b: T, c: T) -> T {
    if !FMA {
        c + a.conj() * b
    } else if T::IS_COMPLEX {
        let (ar, ai, br, bi) = (a.re(), a.im(), b.re(), b.im());
        let re = ai.mul_add(bi, ar.mul_add(br, c.re()));
        let im = (-ai).mul_add(br, ar.mul_add(bi, c.im()));
        T::from_parts(re, im)
    } else {
        T::from_real(a.re().mul_add(b.re(), c.re()))
    }
}

/// The sum of the lanes of an accumulator, pairwise.
#[inline(always)]
fn lanes<T: Scalar, const L: usize>(s: [T; L]) -> T {
    let mut s = s;
    let mut w = L;
    while w > 1 {
        w /= 2;
        for l in 0..w {
            let x = s[l + w];
            s[l] += x;
        }
    }
    s[0]
}

/// The whole chunks of `L` of `x`.
#[inline(always)]
fn chunks<T, const L: usize>(x: &[T]) -> &[[T; L]] {
    x.as_chunks::<L>().0
}

/// The whole chunks of `L` of `x`, to write.
#[inline(always)]
fn chunks_mut<T, const L: usize>(x: &mut [T]) -> &mut [[T; L]] {
    x.as_chunks_mut::<L>().0
}

/// `v*·a_i` for four columns `a_i` as long as `v`, in `L` lanes each.
#[inline(always)]
fn dot4<T: Scalar, const L: usize>(v: &[T], a: [&[T]; 4]) -> [T; 4] {
    let (vb, vt) = v.as_chunks::<L>();
    let body = v.len() - vt.len();
    let mut s = [[T::zero(); L]; 4];
    let it = vb
        .iter()
        .zip(chunks::<T, L>(a[0]))
        .zip(chunks::<T, L>(a[1]));
    let it = it.zip(chunks::<T, L>(a[2]).iter().zip(chunks::<T, L>(a[3])));
    for (((x, a0), a1), (a2, a3)) in it {
        for l in 0..L {
            s[0][l] = fma_conj(x[l], a0[l], s[0][l]);
            s[1][l] = fma_conj(x[l], a1[l], s[1][l]);
            s[2][l] = fma_conj(x[l], a2[l], s[2][l]);
            s[3][l] = fma_conj(x[l], a3[l], s[3][l]);
        }
    }
    let mut out = [T::zero(); 4];
    for ((s, a), out) in s.iter_mut().zip(a).zip(&mut out) {
        for (&x, &a) in vt.iter().zip(&a[body..]) {
            s[0] = fma_conj(x, a, s[0]);
        }
        *out = lanes(*s);
    }
    out
}

/// `v*·a` for a column `a` as long as `v`, in `L` lanes.
#[inline(always)]
fn dot1<T: Scalar, const L: usize>(v: &[T], a: &[T]) -> T {
    let (vb, vt) = v.as_chunks::<L>();
    let body = v.len() - vt.len();
    let mut s = [T::zero(); L];
    for (x, a) in vb.iter().zip(chunks::<T, L>(a)) {
        for l in 0..L {
            s[l] = fma_conj(x[l], a[l], s[l]);
        }
    }
    for (&x, &a) in vt.iter().zip(&a[body..]) {
        s[0] = fma_conj(x, a, s[0]);
    }
    lanes(s)
}

/// `a_i -= w_i·v` for four columns `a_i` as long as `v`, by chunks of `L`.
#[inline(always)]
fn axpy4<T: Scalar, const L: usize>(v: &[T], a: [&mut [T]; 4], w: [T; 4]) {
    let (vb, vt) = v.as_chunks::<L>();
    let body = v.len() - vt.len();
    let n = w.map(|w| -w);
    let [a0, a1, a2, a3] = a;
    {
        let it = vb
            .iter()
            .zip(chunks_mut::<T, L>(a0))
            .zip(chunks_mut::<T, L>(a1));
        let it = it.zip(
            chunks_mut::<T, L>(a2)
                .iter_mut()
                .zip(chunks_mut::<T, L>(a3)),
        );
        for (((x, a0), a1), (a2, a3)) in it {
            // All four chunks read before any is written, so that they go in vectors although
            // the columns might, for all the compiler knows, overlap.
            let (x, mut b0, mut b1, mut b2, mut b3) = (*x, *a0, *a1, *a2, *a3);
            for l in 0..L {
                b0[l] = fma(x[l], n[0], b0[l]);
                b1[l] = fma(x[l], n[1], b1[l]);
                b2[l] = fma(x[l], n[2], b2[l]);
                b3[l] = fma(x[l], n[3], b3[l]);
            }
            (*a0, *a1, *a2, *a3) = (b0, b1, b2, b3);
        }
    }
    for (a, n) in [a0, a1, a2, a3].into_iter().zip(n) {
        for (&x, a) in vt.iter().zip(&mut a[body..]) {
            *a = fma(x, n, *a);
        }
    }
}

/// `a -= w·v` for a column `a` as long as `v`, by chunks of `L`.
#[inline(always)]
fn axpy1<T: Scalar, const L: usize>(v: &[T], a: &mut [T], w: T) {
    let (vb, vt) = v.as_chunks::<L>();
    let body = v.len() - vt.len();
    let n = -w;
    for (x, a) in vb.iter().zip(chunks_mut::<T, L>(a)) {
        for l in 0..L {
            a[l] = fma(x[l], n, a[l]);
        }
    }
    for (&x, a) in vt.iter().zip(&mut a[body..]) {
        *a = fma(x, n, *a);
    }
}

/// `out[c] = v*·a_c` for the `cols` columns `a_c = a[c·lda..c·lda + v.len()]`.
pub(super) fn dots<T: Scalar>(v: &[T], a: &[T], lda: usize, cols: usize, out: &mut [T]) {
    if T::IS_COMPLEX {
        dots_by::<T, 2>(v, a, lda, cols, out);
    } else if narrow::<T>(v.len()) {
        dots_by::<T, 8>(v, a, lda, cols, out);
    } else {
        dots_by::<T, 4>(v, a, lda, cols, out);
    }
}

/// Whether the type is `f32`, whose vectors hold twice the lanes of `f64`'s, and the columns of
/// `len` long enough for twice the lanes, as many vectors of sums: shorter ones would leave more
/// of their elements to the sequential tail. At 8 lanes from any length, `f32` decompositions of
/// `32 × 32` took 2.19 µs against 1.77, `Q` of `128 × 16` 1.68 against 2.36.
#[inline(always)]
fn narrow<T: Scalar>(len: usize) -> bool {
    !T::IS_COMPLEX && core::mem::size_of::<T>() == 4 && len >= 64
}

/// [`dots`] in `L` lanes, four columns a pass.
fn dots_by<T: Scalar, const L: usize>(v: &[T], a: &[T], lda: usize, cols: usize, out: &mut [T]) {
    let len = v.len();
    if len == 0 {
        out[..cols].fill(T::zero());
        return;
    }
    let col = |c: usize| &a[c * lda..c * lda + len];
    let mut c = 0;
    while c + 4 <= cols {
        let s = dot4::<T, L>(v, [col(c), col(c + 1), col(c + 2), col(c + 3)]);
        out[c..c + 4].copy_from_slice(&s);
        c += 4;
    }
    while c < cols {
        out[c] = dot1::<T, L>(v, col(c));
        c += 1;
    }
}

/// `c ← c - s·(1, v)·((1, v)*·c)` for the `cols` columns `c = a[j·lda..j·lda + 1 + v.len()]`:
/// a reflection `I - τ·(1, v)·(1, v)*` applied with `s = τ`, or its conjugate transpose with
/// `s = conj(τ)`. Four columns a pass, each read for the products and again, from the cache,
/// for the update.
pub(super) fn reflect<T: Scalar>(v: &[T], s: T, a: &mut [T], lda: usize, cols: usize) {
    if cols == 1 && v.len() >= LONG {
        // One long column, as for a single right-hand side: more sums, which the latency of
        // the multiply-adds would otherwise hold up.
        let (h, t) = a[..v.len() + 1].split_first_mut().unwrap();
        if T::IS_COMPLEX {
            let w = s * (*h + dot1::<T, 8>(v, t));
            *h -= w;
            axpy1::<T, 8>(v, t, w);
        } else if narrow::<T>(v.len()) {
            let w = s * (*h + dot1::<T, 32>(v, t));
            *h -= w;
            axpy1::<T, 32>(v, t, w);
        } else {
            let w = s * (*h + dot1::<T, 16>(v, t));
            *h -= w;
            axpy1::<T, 16>(v, t, w);
        }
    } else if T::IS_COMPLEX && cols == 1 && v.len() >= 8 {
        // A complex multiply-add is two in a row of each part's sum, so that a shorter column
        // already waits on them in two lanes.
        reflect_by::<T, 4>(v, s, a, lda, cols);
    } else if T::IS_COMPLEX {
        reflect_by::<T, 2>(v, s, a, lda, cols);
    } else if narrow::<T>(v.len()) {
        reflect_by::<T, 8>(v, s, a, lda, cols);
    } else {
        reflect_by::<T, 4>(v, s, a, lda, cols);
    }
}

/// [`reflect`] in `L` lanes.
fn reflect_by<T: Scalar, const L: usize>(v: &[T], s: T, a: &mut [T], lda: usize, cols: usize) {
    let len = v.len() + 1;
    let mut rest = a;
    let mut c = 0;
    while c + 4 <= cols {
        let (a0, r) = rest.split_at_mut(lda);
        let (a1, r) = r.split_at_mut(lda);
        let (a2, r) = r.split_at_mut(lda);
        let (a3, r) = r.split_at_mut(lda.min(r.len()));
        rest = r;
        let (h0, t0) = a0[..len].split_first_mut().unwrap();
        let (h1, t1) = a1[..len].split_first_mut().unwrap();
        let (h2, t2) = a2[..len].split_first_mut().unwrap();
        let (h3, t3) = a3[..len].split_first_mut().unwrap();
        let u = dot4::<T, L>(v, [t0, t1, t2, t3]);
        let mut w = [T::zero(); 4];
        for ((h, u), w) in [h0, h1, h2, h3].into_iter().zip(u).zip(&mut w) {
            *w = s * (*h + u);
            *h -= *w;
        }
        axpy4::<T, L>(v, [t0, t1, t2, t3], w);
        c += 4;
    }
    while c < cols {
        let (a0, r) = rest.split_at_mut(lda.min(rest.len()));
        rest = r;
        let (h, t) = a0[..len].split_first_mut().unwrap();
        let w = s * (*h + dot1::<T, L>(v, t));
        *h -= w;
        axpy1::<T, L>(v, t, w);
        c += 1;
    }
}

/// `[v_0*·y, v_1*·y, v_2*·y, v_3*·y]` for four columns `v_i` as long as `y`.
pub(super) fn dots4<T: Scalar>(y: &[T], v: [&[T]; 4]) -> [T; 4] {
    let s = if T::IS_COMPLEX {
        dot4::<T, 2>(y, v)
    } else if narrow::<T>(y.len()) {
        dot4::<T, 8>(y, v)
    } else {
        dot4::<T, 4>(y, v)
    };
    s.map(|x| x.conj())
}

/// `y -= Σ w_i·v_i` for four columns `v_i` as long as `y`, `y` read and written once.
pub(super) fn axpys4<T: Scalar>(y: &mut [T], v: [&[T]; 4], w: [T; 4]) {
    if T::IS_COMPLEX {
        axpys4_by::<T, 2>(y, v, w);
    } else if narrow::<T>(y.len()) {
        axpys4_by::<T, 8>(y, v, w);
    } else {
        axpys4_by::<T, 4>(y, v, w);
    }
}

/// [`axpys4`] by chunks of `L`.
fn axpys4_by<T: Scalar, const L: usize>(y: &mut [T], v: [&[T]; 4], w: [T; 4]) {
    let (yb, yt) = y.as_chunks_mut::<L>();
    let body = yb.len() * L;
    let n = w.map(|w| -w);
    let it = yb
        .iter_mut()
        .zip(chunks::<T, L>(v[0]))
        .zip(chunks::<T, L>(v[1]));
    let it = it.zip(chunks::<T, L>(v[2]).iter().zip(chunks::<T, L>(v[3])));
    for (((y, v0), v1), (v2, v3)) in it {
        for l in 0..L {
            y[l] = fma(
                v3[l],
                n[3],
                fma(v2[l], n[2], fma(v1[l], n[1], fma(v0[l], n[0], y[l]))),
            );
        }
    }
    for (r, y) in yt.iter_mut().enumerate() {
        let r = body + r;
        *y = fma(
            v[3][r],
            n[3],
            fma(v[2][r], n[2], fma(v[1][r], n[1], fma(v[0][r], n[0], *y))),
        );
    }
}

/// The length of `v` from which [`reflect`] of one column sums in more lanes, 16 of `f64` (32 of
/// `f32` from 64) and 8 of a complex type: the latency of the multiply-adds, not their number, holds up the sums
/// of a long column, and a single right-hand side of the least squares has no other columns to
/// interleave with.
const LONG: usize = 32;

/// `a -= w·v` for a column `a` as long as `v`.
pub(super) fn axpy<T: Scalar>(v: &[T], w: T, a: &mut [T]) {
    if T::IS_COMPLEX {
        axpy1::<T, 4>(v, a, w);
    } else if narrow::<T>(v.len()) {
        axpy1::<T, 16>(v, a, w);
    } else {
        axpy1::<T, 8>(v, a, w);
    }
}

/// The sum of the squares of the absolute values of `x`, in eight sums.
fn squares<T: Scalar>(x: &[T]) -> T::Real {
    let (body, rest) = x.as_chunks::<8>();
    let mut s = [T::Real::ZERO; 8];
    for c in body {
        for l in 0..8 {
            s[l] += c[l].abs2();
        }
    }
    let mut t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for v in rest {
        t += v.abs2();
    }
    t
}

/// Whether a sum of squares is in the range where no square can have overflowed or underflowed
/// enough to matter, zero and NaN excluded.
#[inline(always)]
fn safe<R: Float>(t: R) -> bool {
    t >= R::MIN_POSITIVE / R::EPSILON && t <= R::MAX * R::EPSILON
}

/// The reflection `H = I - τ·v·v*`, `v = (1, x)`, with `H*·(alpha, x) = (β, 0)` and `β` real, as
/// [`reflector`](super::super::householder::reflector), but with the norm of `(alpha, x)` from
/// one pass of sums of squares where that is safe: overwrites `x` with the rest of `v` and
/// returns `(τ, β)`.
pub(super) fn reflector<T: Scalar>(alpha: T, x: &mut [T]) -> (T, T) {
    let t = squares(x);
    let norm = if safe(t) {
        let a = alpha.abs2();
        if a == T::Real::ZERO || safe(a) {
            (a + t).sqrt()
        } else {
            alpha.abs().hypot(t.sqrt())
        }
    } else {
        let xnorm = norm2(x);
        if xnorm == T::Real::ZERO && alpha.im() == T::Real::ZERO {
            return (T::zero(), alpha);
        }
        alpha.abs().hypot(xnorm)
    };
    let beta = if alpha.re() >= T::Real::ZERO {
        -norm
    } else {
        norm
    };
    let tau = T::from_parts((beta - alpha.re()) / beta, -alpha.im() / beta);
    let r = T::one() / (alpha - T::from_real(beta));
    for v in x.iter_mut() {
        *v *= r;
    }
    (tau, T::from_real(beta))
}
