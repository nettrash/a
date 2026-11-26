//! The reduction of one front to its upper triangle, by panels of Householder reflections.
//!
//! The reflections of a front are taken in panels of [`PANEL`] columns. Within a panel each
//! reflection is applied to the columns of the panel after it as soon as it is made, since the
//! next one is taken from them; the columns right of the panel take the panel's reflections all
//! together. In a front held dense ([`dense`]) they take them as one block, `I − V·T·Vᴴ`, the
//! compact WY form of Schreiber and Van Loan ("A storage-efficient WY representation for products
//! of Householder transformations", SIAM J. Sci. Stat. Comput. 10, 1989), by three products of
//! dense blocks: the crate's [`Block`]. In a front held by its staircase, whose columns have
//! lengths of their own, they take them one reflection after another, four columns at a time,
//! the panel's vectors staying in the cache from one group of columns to the next.
//!
//! Every reflection works on the rows of its column's step of the staircase and on no others,
//! in both forms: a block of reflections spans the rows from its first reflection down to the
//! step of its last, which every column right of it reaches.

use crate::math::complex::Float;
use crate::math::matrix::Scalar;
use crate::math::matrix::blas::{Op, gemm};
use crate::math::matrix::householder::{Block, reflector};
use core::ops::Range;

/// The reflections of a panel. Measured on an Apple M4 Max (`SparseQr::new` on the least squares
/// of grids of 100 and 183 points a side, the Laplacians of 100 and 183, the random tall matrix of
/// 1000 columns and the very tall band, in `f32`, `f64`, `Complex<f32>` and `Complex<f64>`; sweep
/// 4 to 64, with [`DENSE`] 32 to 512): 16 with fronts held dense from 128 reflections takes 3 to
/// 14 % off the grids, Laplacians and random matrix of 183 points a side and 1000 columns in every
/// type against 32 and 256, and leaves the rest within 0.5 %. Panels of 8 are faster still in `f64`
/// (up to 21 %) but up to 8 % slower in `Complex<f64>`; 48 and 64 lose up to 28 %.
pub(super) const PANEL: usize = 16;

/// The most columns of a front reduced without panels, each reflection applied to the columns
/// after it one at a time (by the numeric phase itself, not by [`reduce`]): the fronts of a curve
/// fit or a band have two or three. Measured on an Apple M4 Max (`SparseQr::new` on the curve
/// fits, bands, grids' least squares, Laplacians and random tall matrices of the scoreboard;
/// sweep 32 to 128): flat from 32 to 64 (within 1 %), while 96 and 128 cost the grids of 64 to
/// 183 points a side up to 4 and 7 %.
pub(super) const SMALL: usize = 64;

/// The fewest reflections of a front held dense. Below, the panels' columns at a time beat the
/// blocks. Measured on an Apple M4 Max (the matrices of [`PANEL`]; sweep 32 to 512): with panels
/// of 16, the grids and Laplacians of 183 points a side, whose largest fronts have a few hundred
/// reflections, take 4 % longer with 256 than with 128 and 8 to 10 % longer with 384; 32 to 96
/// are as fast as 128 in `f64` (within 1 %) but up to 8 % slower in `Complex<f64>`.
const DENSE: usize = 128;

/// The fewest columns right of a panel that take its reflections as a block; fewer take them one
/// at a time, which spares the triangular factor of the block. Measured on an Apple M4 Max (the
/// matrices of [`PANEL`], `f64` and `Complex<f64>`; sweep 4 to 32): flat, within 0.6 %.
const BLOCK_COLUMNS: usize = 16;

/// Whether a front of `height` rows, `width` columns and `count` reflections, `size` elements
/// in its staircase, is held dense: its column `t` at `t·height`, with the elements below its
/// step unused, so that the columns right of a panel are one block with one leading dimension.
/// So it is when it has [`DENSE`] reflections or more and its staircase fills at least half of
/// its rectangle; the rest of the rectangle is memory and nothing else, no arithmetic touches it.
/// Measured on an Apple M4 Max (the matrices of [`PANEL`], `f64` and `Complex<f64>`): a third or a
/// quarter instead of the half are as fast (within 0.4 %), two thirds or four fifths up to 48 %
/// slower (the random tall matrix, whose largest fronts then keep their staircase).
pub(super) fn dense(height: usize, width: usize, count: usize, size: usize) -> bool {
    count >= DENSE && (height as u64) * (width as u64) <= 2 * size as u64
}

/// Reduces a front to its upper triangle: reflection `t` takes column `t` from row `t` to its
/// step `end[t]` to `(β, 0, …)`, its vector (with its leading 1) is written to
/// `vectors[vector_ptr[t]..vector_ptr[t + 1]]`, which is `end[t] − t` long, and its factor to
/// `taus[t]`, and it is applied to every column after `t`.
///
/// Column `t` of the front begins at `start[t]` (`start` has an entry after the last column)
/// and holds the rows above `end[t]`, which is not below the step of any column before it.
/// `ld` is the leading dimension of a front held dense ([`dense`]), `None` for one held by its
/// staircase. `scratch` is working memory for the vectors of a block.
#[allow(clippy::too_many_arguments)]
pub(super) fn reduce<T: Scalar>(
    front: &mut [T],
    start: &[usize],
    end: &[usize],
    ld: Option<usize>,
    count: usize,
    vectors: &mut [T],
    vector_ptr: &[usize],
    taus: &mut [T],
    scratch: &mut Vec<T>,
) {
    let width = end.len();
    let mut t0 = 0;
    while t0 < count {
        let t1 = (t0 + PANEL).min(count);
        for t in t0..t1 {
            let (left, right) = front.split_at_mut(start[t + 1]);
            let column = &mut left[start[t] + t..start[t] + end[t]];
            let (tau, beta) = reflector(column[0], &mut column[1..]);
            let vector = &mut vectors[vector_ptr[t]..vector_ptr[t + 1]];
            vector[0] = T::one();
            vector[1..].copy_from_slice(&column[1..]);
            column[0] = beta;
            taus[t] = tau;
            if t + 1 < t1 && tau != T::zero() {
                apply(
                    right,
                    start[t + 1],
                    start,
                    t..t + 1,
                    vectors,
                    vector_ptr,
                    taus,
                    t + 1..t1,
                );
            }
        }
        let trailing = width - t1;
        match ld {
            _ if trailing == 0 => {}
            Some(ld) if trailing >= BLOCK_COLUMNS && t1 - t0 > 1 => {
                // The panel's vectors, zero below their steps, as `V`; its rows are those of the
                // front from `t0` down to the step of the panel's last column.
                let (kb, h) = (t1 - t0, end[t1 - 1] - t0);
                let mut v = core::mem::take(scratch);
                v.clear();
                v.resize(h * kb, T::zero());
                for j in 0..kb {
                    let x = &vectors[vector_ptr[t0 + j]..vector_ptr[t0 + j + 1]];
                    v[j * h + j..j * h + j + x.len()].copy_from_slice(x);
                }
                let t = triangle(&v, h, &taus[t0..t1]);
                let block = Block { v, t, m: h, k: kb };
                block.apply_left(true, trailing, &mut front[start[t1] + t0..], ld);
                *scratch = block.v;
            }
            _ => {
                let base = start[t1];
                let right = &mut front[base..];
                apply(
                    right,
                    base,
                    start,
                    t0..t1,
                    vectors,
                    vector_ptr,
                    taus,
                    t1..width,
                );
            }
        }
        t0 = t1;
    }
}

/// The triangular factor `T` of the block of the reflections whose vectors are the columns of
/// `v`, `m × k` with their leading ones and the zeros above them, and whose factors are `taus`:
/// `H₀·H₁·…·H_k−1 = I − V·T·Vᴴ`, as LAPACK's `larft`, column `j` of `T` above its diagonal
/// `−τ_j·T₀₀·Vᴴ·v_j` with `T₀₀` the factor of the reflections before `j`. The products `Vᴴ·v_j`
/// come from the one product of blocks `Vᴴ·V` rather than one product a column.
fn triangle<T: Scalar>(v: &[T], m: usize, taus: &[T]) -> Vec<T> {
    let k = taus.len();
    let mut gram = vec![T::zero(); k * k];
    gemm(
        k,
        k,
        m,
        T::one(),
        v,
        m,
        Op::C,
        v,
        m,
        Op::N,
        T::zero(),
        &mut gram,
        k,
    );
    let mut t = vec![T::zero(); k * k];
    for j in 0..k {
        let tau = taus[j];
        t[j + j * k] = tau;
        if tau == T::zero() {
            continue;
        }
        for i in 0..j {
            let mut sum = T::zero();
            for p in i..j {
                sum += t[i + p * k] * gram[p + j * k];
            }
            t[i + j * k] = -tau * sum;
        }
    }
    t
}

/// Applies the conjugate transposes of the reflections `ts`, in order, to the columns `us` of the
/// front: reflection `t` with the vector `vectors[vector_ptr[t]..vector_ptr[t + 1]]` to the rows
/// from `t` on that it is long. `front` is the part of the front from `base` on, which holds the
/// columns `us`, each at `start[u] − base`. Four columns at a time, each group taking every
/// reflection before the next group.
#[allow(clippy::too_many_arguments)]
fn apply<T: Scalar>(
    front: &mut [T],
    base: usize,
    start: &[usize],
    ts: Range<usize>,
    vectors: &[T],
    vector_ptr: &[usize],
    taus: &[T],
    us: Range<usize>,
) {
    let mut rest = front;
    let mut at = base;
    let mut u = us.start;
    while u < us.end {
        let group = (us.end - u).min(4);
        let skipped = core::mem::take(&mut rest).split_at_mut(start[u] - at).1;
        let (columns, after) = skipped.split_at_mut(start[u + group] - start[u]);
        rest = after;
        at = start[u + group];
        if group == 4 {
            let (c0, r) = columns.split_at_mut(start[u + 1] - start[u]);
            let (c1, r) = r.split_at_mut(start[u + 2] - start[u + 1]);
            let (c2, c3) = r.split_at_mut(start[u + 3] - start[u + 2]);
            for t in ts.clone() {
                let tau = taus[t];
                if tau == T::zero() {
                    continue;
                }
                let v = &vectors[vector_ptr[t]..vector_ptr[t + 1]];
                let rows = t..t + v.len();
                reflect4(
                    v,
                    tau.conj(),
                    [
                        &mut c0[rows.clone()],
                        &mut c1[rows.clone()],
                        &mut c2[rows.clone()],
                        &mut c3[rows],
                    ],
                );
            }
        } else {
            let mut columns = columns;
            for w in u..u + group {
                let (c, r) = core::mem::take(&mut columns).split_at_mut(start[w + 1] - start[w]);
                columns = r;
                for t in ts.clone() {
                    let tau = taus[t];
                    if tau == T::zero() {
                        continue;
                    }
                    let v = &vectors[vector_ptr[t]..vector_ptr[t + 1]];
                    reflect1(v, tau.conj(), &mut c[t..t + v.len()]);
                }
            }
        }
        u += group;
    }
}

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
fn lanes<T: Scalar, const L: usize>(mut s: [T; L]) -> T {
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

/// `c_i ← c_i − s·v·(vᴴ·c_i)` for four columns `c_i` as long as `v`: the reflection
/// `I − τ·v·vᴴ` with `s = τ`, or its conjugate transpose with `s = conj(τ)`. Each column is read
/// for its product with `v` and again, from the cache, for its update.
fn reflect4<T: Scalar>(v: &[T], s: T, c: [&mut [T]; 4]) {
    if T::IS_COMPLEX {
        reflect4_by::<T, 2>(v, s, c);
    } else {
        reflect4_by::<T, 4>(v, s, c);
    }
}

/// [`reflect4`] in sums of `L` lanes.
#[inline(always)]
fn reflect4_by<T: Scalar, const L: usize>(v: &[T], s: T, c: [&mut [T]; 4]) {
    let n = v.len();
    if n < L {
        // Too short for a chunk: each column on its own.
        for c in c {
            short(v, s, c);
        }
        return;
    }
    let [c0, c1, c2, c3] = c;
    let (c0, c1, c2, c3) = (&mut c0[..n], &mut c1[..n], &mut c2[..n], &mut c3[..n]);
    let (vb, vt) = v.as_chunks::<L>();
    let body = n - vt.len();
    let mut acc = [[T::zero(); L]; 4];
    {
        let it = vb
            .iter()
            .zip(c0.as_chunks::<L>().0)
            .zip(c1.as_chunks::<L>().0)
            .zip(c2.as_chunks::<L>().0.iter().zip(c3.as_chunks::<L>().0));
        for (((x, a0), a1), (a2, a3)) in it {
            for l in 0..L {
                acc[0][l] = fma_conj(x[l], a0[l], acc[0][l]);
                acc[1][l] = fma_conj(x[l], a1[l], acc[1][l]);
                acc[2][l] = fma_conj(x[l], a2[l], acc[2][l]);
                acc[3][l] = fma_conj(x[l], a3[l], acc[3][l]);
            }
        }
    }
    let mut w = [T::zero(); 4];
    for (i, a) in [&*c0, &*c1, &*c2, &*c3].into_iter().enumerate() {
        for (&x, &y) in vt.iter().zip(&a[body..]) {
            acc[i][0] = fma_conj(x, y, acc[i][0]);
        }
        w[i] = -(s * lanes(acc[i]));
    }
    {
        let it = vb
            .iter()
            .zip(c0.as_chunks_mut::<L>().0)
            .zip(c1.as_chunks_mut::<L>().0)
            .zip(
                c2.as_chunks_mut::<L>()
                    .0
                    .iter_mut()
                    .zip(c3.as_chunks_mut::<L>().0),
            );
        for (((x, a0), a1), (a2, a3)) in it {
            // All four chunks read before any is written, so that they stay in vectors.
            let (x, mut b0, mut b1, mut b2, mut b3) = (*x, *a0, *a1, *a2, *a3);
            for l in 0..L {
                b0[l] = fma(x[l], w[0], b0[l]);
                b1[l] = fma(x[l], w[1], b1[l]);
                b2[l] = fma(x[l], w[2], b2[l]);
                b3[l] = fma(x[l], w[3], b3[l]);
            }
            (*a0, *a1, *a2, *a3) = (b0, b1, b2, b3);
        }
    }
    for (a, w) in [c0, c1, c2, c3].into_iter().zip(w) {
        for (&x, y) in vt.iter().zip(&mut a[body..]) {
            *y = fma(x, w, *y);
        }
    }
}

/// [`reflect4`] for one column.
fn reflect1<T: Scalar>(v: &[T], s: T, c: &mut [T]) {
    if T::IS_COMPLEX {
        reflect1_by::<T, 2>(v, s, c);
    } else {
        reflect1_by::<T, 4>(v, s, c);
    }
}

/// [`reflect1`] in sums of `L` lanes.
#[inline(always)]
fn reflect1_by<T: Scalar, const L: usize>(v: &[T], s: T, c: &mut [T]) {
    let n = v.len();
    if n < L {
        short(v, s, c);
        return;
    }
    let c = &mut c[..n];
    let (vb, vt) = v.as_chunks::<L>();
    let body = n - vt.len();
    let mut acc = [T::zero(); L];
    for (x, a) in vb.iter().zip(c.as_chunks::<L>().0) {
        for l in 0..L {
            acc[l] = fma_conj(x[l], a[l], acc[l]);
        }
    }
    for (&x, &y) in vt.iter().zip(&c[body..]) {
        acc[0] = fma_conj(x, y, acc[0]);
    }
    let w = -(s * lanes(acc));
    for (x, a) in vb.iter().zip(c.as_chunks_mut::<L>().0) {
        for l in 0..L {
            a[l] = fma(x[l], w, a[l]);
        }
    }
    for (&x, y) in vt.iter().zip(&mut c[body..]) {
        *y = fma(x, w, *y);
    }
}

/// [`reflect1`] for a vector shorter than a chunk of lanes, in plain loops.
#[inline(always)]
fn short<T: Scalar>(v: &[T], s: T, c: &mut [T]) {
    let c = &mut c[..v.len()];
    let mut sum = T::zero();
    for (&x, &y) in v.iter().zip(c.iter()) {
        sum = fma_conj(x, y, sum);
    }
    let w = -(s * sum);
    for (&x, y) in v.iter().zip(c.iter_mut()) {
        *y = fma(x, w, *y);
    }
}
