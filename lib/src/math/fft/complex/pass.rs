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

/// The fn pointers of the kernels for one float type and one instruction set.
pub struct Kset<T> {
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

#[inline(always)]
fn bf2<T: Float, B: Blk<T>>(a: &mut [B]) {
    let (x, y) = (a[0], a[1]);
    a[0] = x.add(y);
    a[1] = x.sub(y);
}

#[inline(always)]
fn bf4<T: Float, B: Blk<T>, const INV: bool>(a: &mut [B]) {
    let t0 = a[0].add(a[2]);
    let t1 = a[0].sub(a[2]);
    let t2 = a[1].add(a[3]);
    let t3 = a[1].sub(a[3]);
    a[0] = t0.add(t2);
    a[2] = t0.sub(t2);
    a[1] = t1.add_rot::<INV>(t3);
    a[3] = t1.sub_rot::<INV>(t3);
}

#[inline(always)]
fn bf8<T: Float, B: Blk<T>, const INV: bool>(a: &mut [B]) {
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
    a[2] = t1.add_rot::<INV>(t3);
    a[6] = t1.sub_rot::<INV>(t3);
    // The odd outputs: the DFT of v_k·W_8^k, with v1·W_8 = (v1 + w₄v1)/√2 and
    // v3·W_8³ = −(v3 − w₄v3)/√2.
    let e0 = v0.add_rot::<INV>(v2);
    let e1 = v0.sub_rot::<INV>(v2);
    let p = v1.add_rot::<INV>(v1);
    let q = v3.sub_rot::<INV>(v3);
    let e2 = p.sub(q).mulr(h);
    let e3 = p.add(q).mulr(h);
    a[1] = e0.add(e2);
    a[5] = e0.sub(e2);
    a[3] = e1.add_rot::<INV>(e3);
    a[7] = e1.sub_rot::<INV>(e3);
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

/// The butterfly of the radix `R`; `g` the table of [`Pass::g`] for the primes from 11 on.
#[inline(always)]
fn bfly<T: Float, B: Blk<T>, const R: usize, const INV: bool>(a: &mut [B; R], g: &[T]) {
    let a: &mut [B] = a;
    match R {
        11 | 13 | 17 | 19 | 23 => {
            let (c, s) = (&g[..R], &g[R..2 * R]);
            bf_odd::<T, B, INV, 11>(a, |i| c[i], |i| s[i])
        }
        2 => bf2::<T, B>(a),
        3 => bf_odd::<T, B, INV, 1>(a, |i| T::from_f64(C3[i]), |i| T::from_f64(S3[i])),
        4 => bf4::<T, B, INV>(a),
        5 => bf_odd::<T, B, INV, 2>(a, |i| T::from_f64(C5[i]), |i| T::from_f64(S5[i])),
        7 => bf_odd::<T, B, INV, 3>(a, |i| T::from_f64(C7[i]), |i| T::from_f64(S7[i])),
        8 => bf8::<T, B, INV>(a),
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
