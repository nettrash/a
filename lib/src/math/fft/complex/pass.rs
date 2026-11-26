//! The butterflies and the passes of the Stockham transform, written once over [`Blk`].
//!
//! A pass of radix `r` takes the length-`L` sub-transforms, `L = r·m`, of `s` interleaved
//! sequences (decimation in frequency): with `x[q + s·(p + k·m)]` for `q < s`, `p < m`, `k < r`,
//! it writes `y[q + s·(r·p + j)] = W_L^(j·p) · Σ_k x[q + s·(p + k·m)]·W_r^(j·k)`. After the
//! passes of all the factors of `n` (with `s` growing from 1 and `m` shrinking to 1) the
//! transform is in natural order, with no bit reversal. A pass with `m = 1` has no twiddles and
//! may run in place.

use super::Algo;
use super::blk::Blk;
use crate::math::complex::{Complex, Float};

/// A pass kernel: reads `x`, writes `y` (the same for an in-place last pass), for `count`
/// transforms one after the other, `stride` values apart.
pub(crate) type Kern<T> = unsafe fn(&Pass<T>, *const Complex<T>, *mut Complex<T>, usize, usize);

/// The way a pass is computed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Form {
    /// Vectors along `q` (the twiddles of a `p` shared by all lanes): any `s`.
    Q,
    /// Vectors along `p` for `s = 1`, the twiddles in lane form and the results stored by rows.
    P,
    /// The last pass (`m = 1`, no twiddles) along `q`, with no loop over `p`.
    L,
    /// A whole transform of a hard-coded radix (`s = m = 1`) for each run of a batch, the
    /// vectors across the runs, loaded and stored by rows.
    B,
}

/// A fused real kernel (see [`real_fwd`] and [`real_inv`]): `(g, tw, x, y)`.
pub(crate) type RKern<T> = unsafe fn(&[T], *const T, *const Complex<T>, *mut Complex<T>);

/// The fn pointers of the kernels for one float type and one instruction set.
pub struct Kset<T> {
    /// The fused real kernels of `n = 2m` for a length `m` of one butterfly: forward, inverse.
    pub(crate) real: fn(usize) -> Option<[RKern<T>; 2]>,
    /// The real kernels of an odd length of one butterfly: forward, inverse.
    pub(crate) real_odd: fn(usize) -> Option<[RKern<T>; 2]>,
    /// The combine passes of the odd split of a real transform by the radix, if it has them.
    pub(crate) split: fn(usize) -> Option<SKerns<T>>,
    /// The last pass of radix `r` fused with the even real transform's pass, if it has one.
    pub(crate) half_last: fn(usize) -> Option<HKern<T>>,
    /// The kernel of a pass of radix `r` in the form, forward or inverse.
    pub(crate) pass: fn(usize, Form, bool) -> Kern<T>,
    /// `x[i] *= w[i]` (or its conjugate for the inverse) for `len` values, the factors in lane
    /// form for the blocks of [`wide`](Self::wide) values and compact for the rest.
    pub(crate) pmul: [unsafe fn(*mut Complex<T>, *const T, usize); 2],
    /// The values of the widest block.
    pub(crate) wide: usize,
    /// The values of the block of a [`Form::P`] pass (1: no such pass).
    pub(crate) plane: usize,
}

/// One pass of a Stockham transform.
pub(crate) struct Pass<T> {
    /// The radix.
    pub r: usize,
    /// The sub-transforms of length `L = r·m` per interleaved sequence: `m`.
    pub m: usize,
    /// The interleaved sequences, the stride of the contiguous index.
    pub s: usize,
    /// The twiddles `W_L^(j·p)`, `1 ≤ j < r`, `p < m`: in compact form by `p` then `j` for
    /// [`Form::Q`]; for [`Form::P`] in lane form by blocks of `p` then `j`, and
    /// the `p` left over in compact form.
    pub tw: Vec<T>,
    /// For the radices 11 to 23: `cos(2π·i/r)` for `i < r`, then `sin(2π·i/r)`.
    pub g: Vec<T>,
    /// The kernels, forward and inverse (unused for a pass by a sub-plan).
    pub kern: [Kern<T>; 2],
    /// For a whole transform of a hard-coded radix, the [`Form::B`] kernels for batches.
    pub bkern: Option<[Kern<T>; 2]>,
    /// A pass of a large prime radix by its own plan, one gathered sequence at a time.
    pub sub: Option<Box<Algo<T>>>,
    /// The same pass by blocks of interleaved sequences, when it applies.
    pub batched: Option<Box<super::SubBatch<T>>>,
}

// cos and sin of 2π·i/r, i < r, for the hard-coded odd radices.
const C3: [f64; 3] = [1.0, -0.5, -0.5];
const S3: [f64; 3] = [0.0, 0.866_025_403_784_438_6, -0.866_025_403_784_438_6];
const C5: [f64; 5] = [
    1.0,
    0.309_016_994_374_947_45,
    -0.809_016_994_374_947_5,
    -0.809_016_994_374_947_5,
    0.309_016_994_374_947_45,
];
const S5: [f64; 5] = [
    0.0,
    0.951_056_516_295_153_5,
    0.587_785_252_292_473_1,
    -0.587_785_252_292_473_1,
    -0.951_056_516_295_153_5,
];
const C7: [f64; 7] = [
    1.0,
    0.623_489_801_858_733_5,
    -0.222_520_933_956_314_4,
    -0.900_968_867_902_419_1,
    -0.900_968_867_902_419_1,
    -0.222_520_933_956_314_4,
    0.623_489_801_858_733_5,
];
const S7: [f64; 7] = [
    0.0,
    0.781_831_482_468_029_8,
    0.974_927_912_181_823_6,
    0.433_883_739_117_558_1,
    -0.433_883_739_117_558_1,
    -0.974_927_912_181_823_6,
    -0.781_831_482_468_029_8,
];

/// The radices with a butterfly: hard-coded, or for the primes from 11 to 23 the general odd
/// one from the pass's table of cosines and sines.
pub(crate) const CODELETS: [usize; 10] = [2, 3, 4, 5, 7, 8, 11, 13, 17, 19];

/// The largest radix of a pass; a larger prime factor is taken by a sub-plan pass.
pub(crate) const MAX_GENERIC: usize = 19;

/// The short lengths `n = r1·r2` transformed whole by one composite butterfly (a single pass,
/// every value in registers): `(n, r1, r2)`, the inner butterflies of `r1` first (the order of
/// [`bfly`]). Measured on an Apple M4 Max (each length, one run and batches of 64, both types):
/// each takes 0.18 to 0.96 of its passes' time (`f32` 20 in batches alike); 32 = 8·4 took 1.16
/// to 1.87 times its two passes' time, so it is none; 18, 27, 30, 36 and 48 would need nested
/// composites. The order: 20 = 4·5 takes 0.95 to 0.99 of 5·4's time in `f64` (0.98 to 1.0 in
/// `f32`); the others reversed lose or stay within the noise (24 = 3·8 1.28 to 1.51 times in
/// `f32`, 16 = 8·2 or 2·8 1.03 to 1.05, 12 = 3·4 up to 1.03; 15 = 3·5 0.97 to 0.99 alone but 1.02
/// to 1.03 in the odd real splits by 15).
pub(crate) const COMPOSITE: [(usize, usize, usize); 9] = [
    (6, 3, 2),
    (9, 3, 3),
    (10, 5, 2),
    (12, 4, 3),
    (15, 5, 3),
    (16, 4, 4),
    (20, 4, 5),
    (24, 8, 3),
    (25, 5, 5),
];

/// The table of the butterfly of the radix or length `r`: the cosines and sines for the primes
/// from 11, [`composite_table`] for a composite length, none for the rest.
pub(crate) fn bfly_table<T: Float>(r: usize) -> Vec<T> {
    if composite(r) {
        composite_table::<T>(r)
    } else if r > 8 {
        let mut g: Vec<T> = (0..r)
            .map(|i| T::from_f64(super::twiddle::root(i, r).0))
            .collect();
        g.extend((0..r).map(|i| T::from_f64(-super::twiddle::root(i, r).1)));
        g
    } else {
        Vec::new()
    }
}

/// Whether `n` has a composite butterfly.
pub(crate) fn composite(n: usize) -> bool {
    COMPOSITE.iter().any(|c| c.0 == n)
}

/// The table of a composite butterfly of `n`: `W_n^e` for `e < n` in compact form.
pub(crate) fn composite_table<T: Float>(n: usize) -> Vec<T> {
    (0..n)
        .flat_map(|e| {
            let (c, s) = super::twiddle::root(e, n);
            let (c, s) = (T::from_f64(c), T::from_f64(s));
            [c, c, T::ZERO - s, s]
        })
        .collect()
}

#[inline(always)]
fn bf2<T: Float, B: Blk<T>>(a: &mut [B]) {
    let (x, y) = (a[0], a[1]);
    a[0] = x.add(y);
    a[1] = x.sub(y);
}

/// `x + w₄·y`, by [`Blk::add_rot_f`] when `FC` (the written-out butterflies outside loops).
#[inline(always)]
fn rot_add<T: Float, B: Blk<T>, const INV: bool, const FC: bool>(x: B, y: B) -> B {
    if FC {
        x.add_rot_f::<INV>(y)
    } else {
        x.add_rot::<INV>(y)
    }
}

/// `x − w₄·y`, as [`rot_add`].
#[inline(always)]
fn rot_sub<T: Float, B: Blk<T>, const INV: bool, const FC: bool>(x: B, y: B) -> B {
    if FC {
        x.sub_rot_f::<INV>(y)
    } else {
        x.sub_rot::<INV>(y)
    }
}

#[inline(always)]
fn bf4<T: Float, B: Blk<T>, const INV: bool, const FC: bool>(a: &mut [B]) {
    let t0 = a[0].add(a[2]);
    let t1 = a[0].sub(a[2]);
    let t2 = a[1].add(a[3]);
    let t3 = a[1].sub(a[3]);
    a[0] = t0.add(t2);
    a[2] = t0.sub(t2);
    a[1] = rot_add::<T, B, INV, FC>(t1, t3);
    a[3] = rot_sub::<T, B, INV, FC>(t1, t3);
}

#[inline(always)]
fn bf8<T: Float, B: Blk<T>, const INV: bool, const FC: bool>(a: &mut [B]) {
    let h = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
    let (u0, v0) = (a[0].add(a[4]), a[0].sub(a[4]));
    let (u1, v1) = (a[1].add(a[5]), a[1].sub(a[5]));
    let (u2, v2) = (a[2].add(a[6]), a[2].sub(a[6]));
    let (u3, v3) = (a[3].add(a[7]), a[3].sub(a[7]));
    // The even outputs: the DFT of u.
    let t0 = u0.add(u2);
    let t1 = u0.sub(u2);
    let t2 = u1.add(u3);
    let t3 = u1.sub(u3);
    a[0] = t0.add(t2);
    a[4] = t0.sub(t2);
    a[2] = rot_add::<T, B, INV, FC>(t1, t3);
    a[6] = rot_sub::<T, B, INV, FC>(t1, t3);
    // The odd outputs: the DFT of v_k·W_8^k, with v1·W_8 = (v1 + w₄v1)/√2 and
    // v3·W_8³ = −(v3 − w₄v3)/√2.
    let e0 = rot_add::<T, B, INV, FC>(v0, v2);
    let e1 = rot_sub::<T, B, INV, FC>(v0, v2);
    let p = rot_add::<T, B, INV, FC>(v1, v1);
    let q = rot_sub::<T, B, INV, FC>(v3, v3);
    let e2 = p.sub(q).mulr(h);
    let e3 = p.add(q).mulr(h);
    a[1] = e0.add(e2);
    a[5] = e0.sub(e2);
    a[3] = rot_add::<T, B, INV, FC>(e1, e3);
    a[7] = rot_sub::<T, B, INV, FC>(e1, e3);
}

/// The butterfly of an odd radix `r = a.len()` from the cosines `c` and sines `s` of `2π·i/r`:
/// `y_j = x₀ + Σ_k c_jk·(x_k + x_(r−k)) + w₄·Σ_k s_jk·(x_k − x_(r−k))` and `y_(r−j)` with the
/// second sum subtracted, over the pairs `k ≤ (r − 1)/2`.
#[inline(always)]
fn bf_odd<T: Float, B: Blk<T>, const INV: bool, const H: usize>(
    a: &mut [B],
    c: impl Fn(usize) -> T,
    s: impl Fn(usize) -> T,
) {
    let r = a.len();
    let h = (r - 1) / 2;
    let x0 = a[0];
    let mut t = [B::zero(); H];
    let mut u = [B::zero(); H];
    let mut y0 = x0;
    for k in 1..=h {
        t[k - 1] = a[k].add(a[r - k]);
        u[k - 1] = a[k].sub(a[r - k]);
        y0 = y0.add(t[k - 1]);
    }
    for j in 1..=h {
        let mut aa = x0;
        let mut bb = u[0].mulr(s(j % r));
        for k in 1..=h {
            aa = aa.fmar(t[k - 1], c((j * k) % r));
        }
        for k in 2..=h {
            bb = bb.fmar(u[k - 1], s((j * k) % r));
        }
        a[j] = aa.add_rot::<INV>(bb);
        a[r - j] = aa.sub_rot::<INV>(bb);
    }
    a[0] = y0;
}

/// `x·W_n^e` (its conjugate for the inverse), `g` the table of [`composite_table`]: the
/// quarter and eighth turns by rotations, the rest by the table.
#[inline(always)]
fn twiddle_mul<T: Float, B: Blk<T>, const INV: bool>(x: B, e: usize, n: usize, g: &[T]) -> B {
    let h = T::from_f64(core::f64::consts::FRAC_1_SQRT_2);
    let e = e % n;
    if e == 0 {
        x
    } else if 2 * e == n {
        B::zero().sub(x)
    } else if 4 * e == n {
        x.rot::<INV>()
    } else if 4 * e == 3 * n {
        x.rot_neg::<INV>()
    } else if 8 * e == n {
        x.add_rot::<INV>(x).mulr(h)
    } else if 8 * e == 3 * n {
        x.sub_rot::<INV>(x).mulr(T::ZERO - h)
    } else if 8 * e == 5 * n {
        x.add_rot::<INV>(x).mulr(T::ZERO - h)
    } else if 8 * e == 7 * n {
        x.sub_rot::<INV>(x).mulr(h)
    } else {
        // SAFETY: the table holds the n factors, 4 reals each.
        unsafe { x.mul_tw::<INV>(B::tw_b(g.as_ptr().add(4 * e))) }
    }
}

/// The butterfly of the length `n = R1·R2` (decimation in time): the `R2` transforms of `R1`
/// of `x[R2·n1 + n2]` over `n1`, the twiddles `W_n^(n2·k1)`, then the `R1` transforms of `R2`
/// over `n2`, written to `X[k1 + R1·k2]`; every value in registers.
#[inline(always)]
fn bf_comp<T: Float, B: Blk<T>, const R1: usize, const R2: usize, const INV: bool>(
    a: &mut [B],
    g: &[T],
) {
    let n = R1 * R2;
    let mut c = [[B::zero(); R1]; R2];
    for (n2, col) in c.iter_mut().enumerate() {
        for (n1, v) in col.iter_mut().enumerate() {
            *v = a[R2 * n1 + n2];
        }
        bfly_base::<T, B, R1, INV, false>(col, &[]);
        for (k1, v) in col.iter_mut().enumerate().skip(1) {
            *v = twiddle_mul::<T, B, INV>(*v, n2 * k1, n, g);
        }
    }
    for k1 in 0..R1 {
        let mut d = [B::zero(); R2];
        for (n2, v) in d.iter_mut().enumerate() {
            *v = c[n2][k1];
        }
        bfly_base::<T, B, R2, INV, false>(&mut d, &[]);
        for (k2, v) in d.iter().enumerate() {
            a[k1 + R1 * k2] = *v;
        }
    }
}

/// The butterfly of the radix `R`; `g` the table of [`Pass::g`] for the primes from 11 on, or
/// of [`composite_table`] for a composite length.
#[inline(always)]
fn bfly<T: Float, B: Blk<T>, const R: usize, const INV: bool>(a: &mut [B; R], g: &[T]) {
    match R {
        6 => bf_comp::<T, B, 3, 2, INV>(a, g),
        9 => bf_comp::<T, B, 3, 3, INV>(a, g),
        10 => bf_comp::<T, B, 5, 2, INV>(a, g),
        12 => bf_comp::<T, B, 4, 3, INV>(a, g),
        15 => bf_comp::<T, B, 5, 3, INV>(a, g),
        16 => bf_comp::<T, B, 4, 4, INV>(a, g),
        20 => bf_comp::<T, B, 4, 5, INV>(a, g),
        24 => bf_comp::<T, B, 8, 3, INV>(a, g),
        25 => bf_comp::<T, B, 5, 5, INV>(a, g),
        _ => bfly_base::<T, B, R, INV, true>(a, g),
    }
}

/// The butterfly of a prime radix or of 4 or 8.
#[inline(always)]
fn bfly_base<T: Float, B: Blk<T>, const R: usize, const INV: bool, const FC: bool>(
    a: &mut [B; R],
    g: &[T],
) {
    let a: &mut [B] = a;
    match R {
        11 | 13 | 17 | 19 | 23 => {
            let (c, s) = (&g[..R], &g[R..2 * R]);
            bf_odd::<T, B, INV, 11>(a, |i| c[i], |i| s[i])
        }
        2 => bf2::<T, B>(a),
        3 => bf_odd::<T, B, INV, 1>(a, |i| T::from_f64(C3[i]), |i| T::from_f64(S3[i])),
        4 => bf4::<T, B, INV, FC>(a),
        5 => bf_odd::<T, B, INV, 2>(a, |i| T::from_f64(C5[i]), |i| T::from_f64(S5[i])),
        7 => bf_odd::<T, B, INV, 3>(a, |i| T::from_f64(C7[i]), |i| T::from_f64(S7[i])),
        8 => bf8::<T, B, INV, FC>(a),
        _ => unreachable!(),
    }
}

/// The columns `q` from `q` on of one `p` with blocks of `B` while a whole block fits in `s`;
/// returns the first `q` left. `first` is `p = 0`, whose twiddles are 1.
///
/// # Safety
/// The pointers must be valid for the pass's reads and writes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn cols<T: Float, B: Blk<T>, const R: usize, const INV: bool>(
    xp: *const Complex<T>,
    yp: *mut Complex<T>,
    s: usize,
    ms: usize,
    twp: *const T,
    first: bool,
    mut q: usize,
    g: &[T],
) -> usize {
    // SAFETY: the caller's contract covers every index below.
    unsafe {
        if q + B::K > s {
            return q;
        }
        if first {
            while q + B::K <= s {
                let mut a = [B::zero(); R];
                for (k, v) in a.iter_mut().enumerate() {
                    *v = B::ld(xp.add(q + ms * k));
                }
                bfly::<T, B, R, INV>(&mut a, g);
                for (j, v) in a.iter().enumerate() {
                    v.st(yp.add(q + s * j));
                }
                q += B::K;
            }
        } else {
            let mut w = [B::tw_b(twp); R];
            for (j, w) in w.iter_mut().enumerate().skip(1) {
                *w = B::tw_b(twp.add(4 * (j - 1)));
            }
            while q + B::K <= s {
                let mut a = [B::zero(); R];
                for (k, v) in a.iter_mut().enumerate() {
                    *v = B::ld(xp.add(q + ms * k));
                }
                bfly::<T, B, R, INV>(&mut a, g);
                for j in 1..R {
                    a[j] = a[j].mul_tw::<INV>(w[j]);
                }
                for (j, v) in a.iter().enumerate() {
                    v.st(yp.add(q + s * j));
                }
                q += B::K;
            }
        }
        q
    }
}

/// A pass of the radix `R` along `q`: blocks of `W`, then of `H`, then of `N` for what is left
/// of each `s`.
///
/// # Safety
/// `x` must be valid for reading and `y` for writing the `r·m·s` values of the pass; they may
/// be the same only when `m = 1`.
#[inline(always)]
pub(crate) unsafe fn q_pass<
    T: Float,
    W: Blk<T>,
    H: Blk<T>,
    N: Blk<T>,
    const R: usize,
    const INV: bool,
>(
    d: &Pass<T>,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let (m, s) = (d.m, d.s);
    let ms = m * s;
    let tw = d.tw.as_ptr();
    for p in 0..m {
        // SAFETY: within the pass's values and its table of `m·(R − 1)` twiddles.
        unsafe {
            let xp = x.add(s * p);
            let yp = y.add(s * R * p);
            let twp = tw.add(4 * (R - 1) * p);
            let first = p == 0;
            let mut q = cols::<T, W, R, INV>(xp, yp, s, ms, twp, first, 0, &d.g);
            if H::K != W::K {
                q = cols::<T, H, R, INV>(xp, yp, s, ms, twp, first, q, &d.g);
            }
            if N::K != H::K {
                cols::<T, N, R, INV>(xp, yp, s, ms, twp, first, q, &d.g);
            }
        }
    }
}

/// The last pass of the radix `R` (`m = 1`, no twiddles): blocks of `W`, `H`, then `N` along `q`.
///
/// # Safety
/// As for [`q_pass`]; `x` and `y` may be the same.
#[inline(always)]
pub(crate) unsafe fn l_pass<
    T: Float,
    W: Blk<T>,
    H: Blk<T>,
    N: Blk<T>,
    const R: usize,
    const INV: bool,
>(
    d: &Pass<T>,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let s = d.s;
    let g = &d.g[..];
    // SAFETY: within the pass's `R·s` values.
    unsafe {
        let mut q = 0;
        while q + W::K <= s {
            let mut a = [W::zero(); R];
            for (k, v) in a.iter_mut().enumerate() {
                *v = W::ld(x.add(q + s * k));
            }
            bfly::<T, W, R, INV>(&mut a, g);
            for (j, v) in a.iter().enumerate() {
                v.st(y.add(q + s * j));
            }
            q += W::K;
        }
        if H::K != W::K {
            while q + H::K <= s {
                let mut a = [H::zero(); R];
                for (k, v) in a.iter_mut().enumerate() {
                    *v = H::ld(x.add(q + s * k));
                }
                bfly::<T, H, R, INV>(&mut a, g);
                for (j, v) in a.iter().enumerate() {
                    v.st(y.add(q + s * j));
                }
                q += H::K;
            }
        }
        while q < s {
            let mut a = [N::zero(); R];
            for (k, v) in a.iter_mut().enumerate() {
                *v = N::ld(x.add(q + s * k));
            }
            bfly::<T, N, R, INV>(&mut a, g);
            for (j, v) in a.iter().enumerate() {
                v.st(y.add(q + s * j));
            }
            q += 1;
        }
    }
}

/// A pass of the radix `R` with `s = 1` along `p`: blocks of `P` lanes, the rest with `N`.
///
/// # Safety
/// As for [`q_pass`]; `x` and `y` differ.
#[inline(always)]
pub(crate) unsafe fn p_pass<T: Float, P: Blk<T>, N: Blk<T>, const R: usize, const INV: bool>(
    d: &Pass<T>,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let m = d.m;
    let g = &d.g[..];
    let mut tw = d.tw.as_ptr();
    let mut p = 0;
    // SAFETY: within the pass's values and its twiddles, laid out as `Pass::tw` says.
    unsafe {
        while p + P::K <= m {
            let mut a = [P::zero(); R];
            for (k, v) in a.iter_mut().enumerate() {
                *v = P::ld(x.add(p + m * k));
            }
            bfly::<T, P, R, INV>(&mut a, g);
            for (j, v) in a.iter_mut().enumerate().skip(1) {
                *v = v.mul_tw::<INV>(P::tw_l(tw.add(4 * P::K * (j - 1))));
            }
            P::st_rows(&a, y.add(R * p), R);
            tw = tw.add(4 * P::K * (R - 1));
            p += P::K;
        }
        while p < m {
            let mut a = [N::zero(); R];
            for (k, v) in a.iter_mut().enumerate() {
                *v = N::ld(x.add(p + m * k));
            }
            bfly::<T, N, R, INV>(&mut a, g);
            for (j, v) in a.iter_mut().enumerate().skip(1) {
                *v = v.mul_tw::<INV>(N::tw_b(tw.add(4 * (j - 1))));
            }
            for (j, v) in a.iter().enumerate() {
                v.st(y.add(R * p + j));
            }
            tw = tw.add(4 * (R - 1));
            p += 1;
        }
    }
}

/// The transforms of the length `R` of `count` runs `stride` apart (a [`Form::B`] pass): blocks
/// of `P::K` runs with the lanes across them, the rest with `N`.
///
/// # Safety
/// `x` must be valid for reading and `y` for writing the runs; they may be the same.
#[inline(always)]
pub(crate) unsafe fn b_pass<T: Float, P: Blk<T>, N: Blk<T>, const R: usize, const INV: bool>(
    g: &[T],
    x: *const Complex<T>,
    y: *mut Complex<T>,
    count: usize,
    stride: usize,
) {
    let mut c = 0;
    // SAFETY: the caller's contract.
    unsafe {
        while c + P::K <= count {
            let mut a = [P::zero(); R];
            P::ld_rows(&mut a, x.add(c * stride), stride);
            bfly::<T, P, R, INV>(&mut a, g);
            P::st_rows(&a, y.add(c * stride), stride);
            c += P::K;
        }
        while c < count {
            let mut a = [N::zero(); R];
            N::ld_rows(&mut a, x.add(c * stride), stride);
            bfly::<T, N, R, INV>(&mut a, g);
            N::st_rows(&a, y.add(c * stride), stride);
            c += 1;
        }
    }
}

/// The kernel of the last pass of a complex transform fused with the pass over the pairs of the
/// even real transform: `(pass, x, y, tw)`, see [`half_last`].
pub(crate) type HKern<T> = unsafe fn(&Pass<T>, *const Complex<T>, *mut Complex<T>, *const T);

/// The last pass (`m = 1`, radix `R`, the `s` butterflies `x[q + s·k] → Z[q + s·j]`) of the
/// complex transform of the `M = R·s` values packing `2M` reals, fused with the real transform's
/// pass over the pairs `(k, M − k)`: the butterflies `q` and `s − q` together, whose outputs are
/// each other's partners (`Z[q + s·j]` and `Z[s − q + s·(R − 1 − j)]`), so that the spectrum's
/// `M + 1` values are written from the registers. `tw` holds the pass's factors
/// `−i·e^(−2πi·k/2M)/2` as `[re, im]` for `2k < M`.
///
/// # Safety
/// `x` must be valid for reading `M` values and `y` for writing `M + 1`; they may be the same
/// (each pair of butterflies writes the places it read).
#[inline(always)]
pub(crate) unsafe fn half_last<T: Float, B: Blk<T>, const R: usize>(
    d: &Pass<T>,
    x: *const Complex<T>,
    y: *mut Complex<T>,
    tw: *const T,
) {
    let s = d.s;
    let mm = R * s;
    let g = &d.g[..];
    let half = T::from_f64(0.5);
    // SAFETY: the caller's contract: every index below `M` (`M` for the last value).
    unsafe {
        // The pair (k, M − k), 0 < 2k < M, from a = Z[k] and b = Z[M − k].
        let pair = |a: B, b: B, k: usize| {
            let c = b.conj();
            let u = a.add(c).mulr(half);
            let t = a.sub(c).mul_tw::<false>(B::tw_c(tw.add(2 * k)));
            u.add(t).st(y.add(k));
            u.sub(t).conj().st(y.add(mm - k));
        };
        let ld = |q: usize| {
            let mut a = [B::zero(); R];
            for (k, v) in a.iter_mut().enumerate() {
                *v = B::ld(x.add(q + s * k));
            }
            bfly::<T, B, R, false>(&mut a, g);
            a
        };
        // The butterfly 0: Z[s·j] pairs with Z[s·(R − j)]; Z[0] gives X[0] and X[M].
        let a = ld(0);
        let mut z = Complex::ZERO;
        a[0].st(&mut z);
        for j in 1..R {
            if 2 * j < R {
                pair(a[j], a[R - j], s * j);
            } else if 2 * j == R {
                a[j].conj().st(y.add(s * j));
            }
        }
        // The butterflies q and s − q for 0 < q < s/2.
        for q in 1..s.div_ceil(2) {
            let (a, b) = (ld(q), ld(s - q));
            for j in 0..R {
                if 2 * j < R {
                    pair(a[j], b[R - 1 - j], q + s * j);
                } else {
                    pair(b[R - 1 - j], a[j], s - q + s * (R - 1 - j));
                }
            }
        }
        // The butterfly s/2 of an even s: Z[s/2 + s·j] pairs with Z[s/2 + s·(R − 1 − j)].
        if s.is_multiple_of(2) {
            let a = ld(s / 2);
            for j in 0..R {
                if 2 * j + 1 < R {
                    pair(a[j], a[R - 1 - j], s / 2 + s * j);
                } else if 2 * j + 1 == R {
                    a[j].conj().st(y.add(s / 2 + s * j));
                }
            }
        }
        *y = Complex::new(z.re + z.im, T::ZERO);
        *y.add(mm) = Complex::new(z.re - z.im, T::ZERO);
    }
}

/// The kernel of the forward combine pass of the odd split `n = R·m` of a real transform:
/// `(g, tw, zb, yl, m, y)`, see [`split_fwd`].
pub(crate) type SFwd<T> =
    unsafe fn(&[T], *const T, *const Complex<T>, *const Complex<T>, usize, *mut Complex<T>);

/// The kernel of the inverse combine pass of the odd split: `(g, tw, x, m, zb, ul)`, see
/// [`split_inv`].
pub(crate) type SInv<T> =
    unsafe fn(&[T], *const T, *const Complex<T>, usize, *mut Complex<T>, *mut Complex<T>);

/// The forward and inverse combine passes of the odd split by one radix.
pub(crate) type SKerns<T> = (SFwd<T>, SInv<T>);

/// The combine pass of the odd split `n = R·m` (`m` odd) of a real transform, forward: for each
/// `s < h = (m + 1)/2` the spectra `Y_a[s]` of the `R` residue sequences (those of the pair `p`
/// from the transform `Z_p` of the `(R − 1)/2` at `zb`, `m` apart, as
/// `Y_2p = (Z[s] + conj Z[m − s])/2` and `Y_2p+1 = −i·(Z[s] − conj Z[m − s])/2`; the last at
/// `yl`), twiddled by `tw` (`e^(−2πi·a·s/n)` in compact form, `a = 1..R` by `a` then `s`), then
/// the butterfly of `R` (table `g`): `X[s + q·m]` for `q ≤ (R − 1)/2` into `y`, the others by
/// conjugates at `n − s − q·m`. `y[0]`'s imaginary part is left to the caller.
///
/// # Safety
/// `zb` must be valid for reading `(R − 1)/2·m` values, `yl` `h`, `tw` `4(R − 1)h` reals and `y`
/// for writing `(n + 1)/2`, none overlapping `y`.
#[inline(always)]
pub(crate) unsafe fn split_fwd<T: Float, B: Blk<T>, const R: usize>(
    g: &[T],
    tw: *const T,
    zb: *const Complex<T>,
    yl: *const Complex<T>,
    m: usize,
    y: *mut Complex<T>,
) {
    let (h, np) = (m.div_ceil(2), (R - 1) / 2);
    let half = T::from_f64(0.5);
    // SAFETY: the caller's contract: `s`, `m − s` below `m`, the outputs below `(n + 1)/2`.
    unsafe {
        for s in 0..h {
            let sm = if s == 0 { 0 } else { m - s };
            let mut a = [B::zero(); R];
            for p in 0..np {
                let z = zb.add(p * m);
                let (u, v) = (B::ld(z.add(s)), B::ld(z.add(sm)).conj());
                a[2 * p] = u.add(v).mulr(half);
                a[2 * p + 1] = u.sub(v).mulr(half).rot::<false>();
            }
            a[R - 1] = B::ld(yl.add(s));
            for (j, v) in a.iter_mut().enumerate().skip(1) {
                *v = v.mul_tw::<false>(B::tw_b(tw.add(4 * ((j - 1) * h + s))));
            }
            bfly::<T, B, R, false>(&mut a, g);
            for (q, v) in a.iter().enumerate().take(np + 1) {
                v.st(y.add(s + q * m));
            }
            if s != 0 {
                for (q, v) in a.iter().enumerate().skip(np + 1) {
                    v.conj().st(y.add((R - q) * m - s));
                }
            }
        }
    }
}

/// The inverse of [`split_fwd`], unscaled: from the `(n + 1)/2` values of the spectrum at `x`
/// (the imaginary part of `x[0]` not read), for each `s < h` the butterfly of `R` on
/// `X[s + q·m]` (conjugates of the first half beyond it), twiddled by the conjugates of `tw`,
/// into the pairs `Z_p = U_2p + i·U_2p+1` at `zb` (all `m` of each) and the last sequence's
/// half spectrum `ul`.
///
/// # Safety
/// `x` must be valid for reading `(n + 1)/2` values, `tw` `4(R − 1)h` reals, `zb` for writing
/// `(R − 1)/2·m` values and `ul` `h`, none overlapping.
#[inline(always)]
pub(crate) unsafe fn split_inv<T: Float, B: Blk<T>, const R: usize>(
    g: &[T],
    tw: *const T,
    x: *const Complex<T>,
    m: usize,
    zb: *mut Complex<T>,
    ul: *mut Complex<T>,
) {
    let (n, h, np) = (R * m, m.div_ceil(2), (R - 1) / 2);
    // SAFETY: the caller's contract: `s + q·m ≤ (n − 1)/2` for `q ≤ (R − 1)/2` as `s < h` and
    // `m` is odd, and `n − s − q·m` in `1..=(n − 1)/2` for the others.
    unsafe {
        let x0 = Complex::new((*x).re, T::ZERO);
        for s in 0..h {
            let mut a = [B::zero(); R];
            for (q, v) in a.iter_mut().enumerate() {
                *v = if q <= np {
                    B::ld(if s + q == 0 { &x0 } else { x.add(s + q * m) })
                } else {
                    B::ld(x.add(n - s - q * m)).conj()
                };
            }
            bfly::<T, B, R, true>(&mut a, g);
            for (j, v) in a.iter_mut().enumerate().skip(1) {
                *v = v.mul_tw::<true>(B::tw_b(tw.add(4 * ((j - 1) * h + s))));
            }
            if s == 0 {
                // The sequences are real: their values at 0 too.
                for v in a.iter_mut() {
                    let mut c = Complex::ZERO;
                    v.st(&mut c);
                    c.im = T::ZERO;
                    *v = B::ld(&c);
                }
            }
            for p in 0..np {
                let (u, v) = (a[2 * p], a[2 * p + 1]);
                let z = zb.add(p * m);
                u.add_rot::<true>(v).st(z.add(s));
                if s != 0 {
                    u.sub_rot::<true>(v).conj().st(z.add(m - s));
                }
            }
            a[R - 1].st(ul.add(s));
        }
    }
}

/// The real transform of `n = 2M` reals, packed in pairs at `x` (`M` complex values), into the
/// `M + 1` values of the spectrum at `y`: the complex butterfly of `M`, then the pass over the
/// pairs `(k, M − k)` of the real transform's `Half` algorithm in registers, `tw` its factors
/// `−i·e^(−2πi·k/n)/2` in compact form for `k < (M + 1)/2`; `g` the butterfly's table.
///
/// # Safety
/// `x` must be valid for reading `M` values, `y` for writing `M + 1`, and they do not overlap;
/// `tw` holds `4·⌈M/2⌉` reals.
#[inline(always)]
pub(crate) unsafe fn real_fwd<T: Float, B: Blk<T>, const M: usize>(
    g: &[T],
    tw: *const T,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let half = T::from_f64(0.5);
    let mut a = [B::zero(); M];
    // SAFETY: the caller's contract.
    unsafe {
        for (j, v) in a.iter_mut().enumerate() {
            *v = B::ld(x.add(j));
        }
        bfly::<T, B, M, false>(&mut a, g);
        let mut k = 1;
        while 2 * k < M {
            let (p, q) = (a[k], a[M - k].conj());
            let u = p.add(q).mulr(half);
            let t = p.sub(q).mul_tw::<false>(B::tw_b(tw.add(4 * k)));
            u.add(t).st(y.add(k));
            u.sub(t).conj().st(y.add(M - k));
            k += 1;
        }
        if M.is_multiple_of(2) {
            a[M / 2].conj().st(y.add(M / 2));
        }
        a[0].st(y);
        let z = *y;
        *y = Complex::new(z.re + z.im, T::ZERO);
        *y.add(M) = Complex::new(z.re - z.im, T::ZERO);
    }
}

/// The inverse of [`real_fwd`], unscaled: the `M + 1` values at `x` into the `2M` reals at `y`,
/// packed in pairs; `tw` the factors `i·e^(2πi·k/n)` in compact form. The imaginary parts of
/// `x[0]` and `x[M]` are not read.
///
/// # Safety
/// `x` must be valid for reading `M + 1` values, `y` for writing `M`; they do not overlap; `tw`
/// as for [`real_fwd`].
#[inline(always)]
pub(crate) unsafe fn real_inv<T: Float, B: Blk<T>, const M: usize>(
    g: &[T],
    tw: *const T,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let two = T::from_f64(2.0);
    let mut a = [B::zero(); M];
    // SAFETY: the caller's contract.
    unsafe {
        let (x0, xm) = ((*x).re, (*x.add(M)).re);
        let z0 = Complex::new(x0 + xm, x0 - xm);
        a[0] = B::ld(&z0);
        let mut k = 1;
        while 2 * k < M {
            let (p, q) = (B::ld(x.add(k)), B::ld(x.add(M - k)).conj());
            let u = p.add(q);
            let t = p.sub(q).mul_tw::<false>(B::tw_b(tw.add(4 * k)));
            a[k] = u.add(t);
            a[M - k] = u.sub(t).conj();
            k += 1;
        }
        if M.is_multiple_of(2) {
            a[M / 2] = B::ld(x.add(M / 2)).conj().mulr(two);
        }
        bfly::<T, B, M, true>(&mut a, g);
        for (j, v) in a.iter().enumerate() {
            v.st(y.add(j));
        }
    }
}

/// The real transform of the `M` reals at `x` (`M` odd) into the `(M + 1)/2` values of the
/// spectrum at `y`, by the complex butterfly of `M` on the reals; `g` the butterfly's table.
///
/// # Safety
/// `x` must be valid for reading `M` reals and `y` for writing `(M + 1)/2` values.
#[inline(always)]
pub(crate) unsafe fn real_odd_fwd<T: Float, B: Blk<T>, const M: usize>(
    g: &[T],
    _tw: *const T,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let x = x.cast::<T>();
    let mut a = [B::zero(); M];
    // SAFETY: the caller's contract.
    unsafe {
        for (j, v) in a.iter_mut().enumerate() {
            let z = Complex::new(*x.add(j), T::ZERO);
            *v = B::ld(&z);
        }
        bfly::<T, B, M, false>(&mut a, g);
        for (k, v) in a.iter().take(M.div_ceil(2)).enumerate() {
            v.st(y.add(k));
        }
        (*y).im = T::ZERO;
    }
}

/// The inverse of [`real_odd_fwd`], unscaled: the `(M + 1)/2` values at `x` into the `M` reals
/// at `y`; the imaginary part of `x[0]` is not read.
///
/// # Safety
/// `x` must be valid for reading `(M + 1)/2` values and `y` for writing `M` reals.
#[inline(always)]
pub(crate) unsafe fn real_odd_inv<T: Float, B: Blk<T>, const M: usize>(
    g: &[T],
    _tw: *const T,
    x: *const Complex<T>,
    y: *mut Complex<T>,
) {
    let y = y.cast::<T>();
    let mut a = [B::zero(); M];
    // SAFETY: the caller's contract.
    unsafe {
        let z0 = Complex::new((*x).re, T::ZERO);
        a[0] = B::ld(&z0);
        for k in 1..M.div_ceil(2) {
            let v = B::ld(x.add(k));
            a[k] = v;
            a[M - k] = v.conj();
        }
        bfly::<T, B, M, true>(&mut a, g);
        for (j, v) in a.iter().enumerate() {
            let mut z = Complex::new(T::ZERO, T::ZERO);
            v.st(&mut z);
            *y.add(j) = z.re;
        }
    }
}

/// `x[i] *= w[i]` (the conjugate for the inverse) for `len` values: blocks of `W` with the
/// factors in lane form, the rest with `N` in compact form.
///
/// # Safety
/// `x` must be valid for `len` values and `w` for `4·len` reals.
#[inline(always)]
pub(crate) unsafe fn pmul<T: Float, W: Blk<T>, N: Blk<T>, const INV: bool>(
    x: *mut Complex<T>,
    w: *const T,
    len: usize,
) {
    let mut i = 0;
    // SAFETY: the caller's contract.
    unsafe {
        while i + W::K <= len {
            W::ld(x.add(i))
                .mul_tw::<INV>(W::tw_l(w.add(4 * i)))
                .st(x.add(i));
            i += W::K;
        }
        while i < len {
            N::ld(x.add(i))
                .mul_tw::<INV>(N::tw_b(w.add(4 * i)))
                .st(x.add(i));
            i += 1;
        }
    }
}
