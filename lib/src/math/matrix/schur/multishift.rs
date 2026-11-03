//! The small-bulge multishift QR algorithm with aggressive early deflation of Braman, Byers and
//! Mathias, as LAPACK's `laqr0` and `laqr5`: each sweep chases a chain of many bulges, two
//! shifts each, down the diagonal, a few columns at a time, the reflections of each step
//! gathered into a small orthogonal (unitary) matrix and applied to the rest of the matrix and
//! to `Z` as products of blocks. One code serves real and complex elements: for real ones the
//! shifts come in pairs of conjugates or of reals, so that the arithmetic stays real.

use super::super::Scalar;
use super::super::blas::{Op, gemm};
use super::deflate::deflate;
use super::{short_reflector, small_qr, standardize};
use crate::math::complex::{Complex, Float};

/// Below this order the QR algorithm with one or two shifts is used, LAPACK's `nmin` of 75:
/// measured on Apple's M4, 120 (for complex elements with `Z`, 90), as Francis's double shifts
/// for real elements and the single shifts without `Z` keep up to there.
pub(super) fn nmin<T: Scalar>(want_z: bool) -> usize {
    if T::IS_COMPLEX && want_z { 90 } else { 120 }
}

/// A sweep is skipped when the deflation window deflated more than this percentage of itself.
const NIBBLE: usize = 14;

/// After this many iterations without a deflation the deflation window grows.
const KEXNW: usize = 5;

/// Every this many iterations without a deflation the shifts are exceptional ones.
const KEXSH: usize = 6;

/// Sweeps on matrices of at least this order gather their reflections into a block, which
/// pays for small ones too when the products by `Z` are long.
const ACCUMULATE: usize = 200;

/// The number of shifts of a sweep and the order of the deflation window, for a matrix of
/// order `n`, with or without `Z`: LAPACK's `iparmq` choices up to order 150, beyond it
/// `n/16` shifts, at most `iparmq`'s or, without `Z`, half of them, and windows half as
/// large again. Measured on Apple's M4, fewer shifts than `iparmq`'s (tuned for older machines)
/// pay, except where the products by `Z` are long.
fn parameters(n: usize, want_z: bool) -> (usize, usize) {
    let lapack = if n >= 6000 {
        256
    } else if n >= 3000 {
        128
    } else if n >= 590 {
        64
    } else if n >= 150 {
        10.max(n / (n as f64).log2().round() as usize)
    } else if n >= 60 {
        10
    } else if n >= 30 {
        4
    } else {
        2
    };
    let ns = if n < 150 {
        lapack
    } else {
        (n / 16)
            .min(if want_z { lapack } else { lapack / 2 })
            .max(10)
    };
    let ns = (ns - ns % 2).max(2);
    let nw = if n < 150 { ns } else { 3 * ns / 2 };
    (ns, nw)
}

/// `|re| + |im|` of a shift.
fn abs1<R: Float>(x: Complex<R>) -> R {
    x.re.abs() + x.im.abs()
}

/// The Schur form of the upper Hessenberg `h` of order `n`, as LAPACK's `laqr0`, or `laqr4`
/// when `nested`, whose deflation windows don't recurse again: the eigenvalues into `w`, `h`
/// turned into the Schur form when `want_t` is set (otherwise only the eigenvalues are right),
/// and the transformations accumulated into the columns of `z`, of order `n` too, when given.
/// Returns the number of leading eigenvalues it didn't converge to, zero on success.
pub(super) fn multishift<T: Scalar>(
    want_t: bool,
    h: &mut [T],
    n: usize,
    mut z: Option<&mut [T]>,
    w: &mut [Complex<T::Real>],
    nested: bool,
) -> usize {
    let nmin = nmin::<T>(z.is_some());
    if n < nmin {
        return small_qr(h, n, z, want_t, w);
    }
    let at = |i: usize, j: usize| i + j * n;
    let (ns, nwr) = parameters(n, z.is_some());
    let nwr = nwr.max(2).min((n - 1) / 3);
    let nsr = ns.min((n - 3) / 6);
    let nsr = (nsr - nsr % 2).max(2);
    let nwmax = (n - 1) / 3;
    let nsmax = (n - 3) / 6;
    let nsmax = nsmax - nsmax % 2;
    let accumulate = z.is_some() || n >= ACCUMULATE;
    let itmax = 30.max(2 * KEXSH) * n;
    let mut nw = nwmax;
    let mut ndfl = 1;
    let mut ndec: isize = -1;
    // The active block ends before `end`.
    let mut end = n;
    for _ in 0..itmax {
        if end == 0 {
            return 0;
        }
        let kbot = end - 1;
        let mut ktop = kbot;
        while ktop > 0 && h[at(ktop, ktop - 1)] != T::zero() {
            ktop -= 1;
        }
        // The deflation window: the recommended size, or after a while without deflations
        // larger and then gradually smaller.
        let nh = kbot - ktop + 1;
        let nwupbd = nh.min(nwmax);
        nw = if ndfl < KEXNW {
            nwupbd.min(nwr)
        } else {
            nwupbd.min(2 * nw)
        };
        if nw < nwmax {
            if nw + 1 >= nh {
                nw = nh;
            } else {
                let kwtop = kbot + 1 - nw;
                if h[at(kwtop, kwtop - 1)].abs1() > h[at(kwtop - 1, kwtop - 2)].abs1() {
                    nw += 1;
                }
            }
        }
        if ndfl < KEXNW {
            ndec = -1;
        } else if ndec >= 0 || nw >= nwupbd {
            ndec += 1;
            if (nw as isize) - ndec < 2 {
                ndec = 0;
            }
            nw -= ndec as usize;
        }
        let (ls, ld) = deflate(want_t, h, n, z.as_deref_mut(), ktop, kbot, nw, w, nested);
        end -= ld;
        // A sweep, unless the deflations were many, or the rest is small.
        if ld == 0 || (100 * ld <= nw * NIBBLE && end - ktop > nmin.min(nwmax)) {
            let kbot = end - 1;
            let mut ks = end - ls;
            let mut ns = nsmax.min(nsr).min(2.max(kbot - ktop));
            ns -= ns % 2;
            if ndfl % KEXSH == 0 {
                exceptional(h, n, ktop, kbot, ns, w);
                ks = kbot + 1 - ns;
            } else {
                if kbot + 1 - ks <= ns / 2 {
                    // Too few shifts from the window: the eigenvalues of the trailing block.
                    ks = kbot + 1 - ns;
                    let mut sub = vec![T::zero(); ns * ns];
                    for j in 0..ns {
                        for i in 0..=(j + 1).min(ns - 1) {
                            sub[i + j * ns] = h[at(ks + i, ks + j)];
                        }
                    }
                    let info = if ns > nmin && !nested {
                        multishift(false, &mut sub, ns, None, &mut w[ks..=kbot], true)
                    } else {
                        small_qr(&mut sub, ns, None, false, &mut w[ks..=kbot])
                    };
                    ks += info;
                    if ks >= kbot {
                        trailing_pair(h, n, kbot, w);
                        ks = kbot - 1;
                    }
                }
                if kbot + 1 - ks > ns {
                    // The largest first, conjugates staying together.
                    let mut last = kbot;
                    let mut sorted = false;
                    while !sorted && last > ks {
                        sorted = true;
                        for i in ks..last {
                            if abs1(w[i]) < abs1(w[i + 1]) {
                                sorted = false;
                                w.swap(i, i + 1);
                            }
                        }
                        last -= 1;
                    }
                }
                if !T::IS_COMPLEX {
                    pair_up(&mut w[ks..=kbot]);
                }
            }
            if kbot + 1 - ks == 2 {
                // Of two shifts both real, or of a complex one's, the one nearer the corner.
                let corner = h[at(kbot, kbot)].to_complex();
                if T::IS_COMPLEX || w[kbot].im == T::Real::ZERO {
                    if abs1(w[kbot] - corner) < abs1(w[kbot - 1] - corner) {
                        w[kbot - 1] = w[kbot];
                    } else {
                        w[kbot] = w[kbot - 1];
                    }
                }
            }
            ns = ns.min(kbot + 1 - ks);
            ns -= ns % 2;
            ks = kbot + 1 - ns;
            let mut s = w[ks..=kbot].to_vec();
            sweep(
                want_t,
                h,
                n,
                z.as_deref_mut(),
                ktop,
                kbot,
                &mut s,
                accumulate,
            );
        }
        ndfl = if ld > 0 { 1 } else { ndfl + 1 };
    }
    end
}

/// Exceptional shifts into `w[kbot + 1 - ns..=kbot]`, from the subdiagonal near the corner.
fn exceptional<T: Scalar>(
    h: &[T],
    n: usize,
    ktop: usize,
    kbot: usize,
    ns: usize,
    w: &mut [Complex<T::Real>],
) {
    let at = |i: usize, j: usize| i + j * n;
    let ks = kbot + 1 - ns;
    let wilk1 = T::Real::from_f64(0.75);
    if T::IS_COMPLEX {
        let mut i = kbot;
        while i > ks {
            let x = h[at(i, i)] + T::from_real(wilk1 * h[at(i, i - 1)].abs1());
            w[i] = x.to_complex();
            w[i - 1] = w[i];
            if i < ks + 3 {
                break;
            }
            i -= 2;
        }
    } else {
        let low = (ks + 1).max(ktop + 2);
        let mut i = kbot;
        while i >= low {
            let ss = h[at(i, i - 1)].abs() + h[at(i - 1, i - 2)].abs();
            let aa = wilk1 * ss + h[at(i, i)].re();
            let s = standardize(aa, ss, T::Real::from_f64(-0.4375) * ss, aa);
            w[i - 1] = Complex::new(s.rt1r, s.rt1i);
            w[i] = Complex::new(s.rt2r, s.rt2i);
            if i < low + 2 {
                break;
            }
            i -= 2;
        }
        if ks == ktop {
            w[ks + 1] = Complex::new(h[at(ks + 1, ks + 1)].re(), T::Real::ZERO);
            w[ks] = w[ks + 1];
        }
    }
}

/// The eigenvalues of the trailing `2 × 2` block into `w[kbot - 1..=kbot]`, the shifts when the
/// QR algorithm failed on a trailing block.
fn trailing_pair<T: Scalar>(h: &[T], n: usize, kbot: usize, w: &mut [Complex<T::Real>]) {
    let at = |i: usize, j: usize| i + j * n;
    let (a, b, c, d) = (
        h[at(kbot - 1, kbot - 1)],
        h[at(kbot - 1, kbot)],
        h[at(kbot, kbot - 1)],
        h[at(kbot, kbot)],
    );
    if T::IS_COMPLEX {
        let s = a.abs1() + b.abs1() + c.abs1() + d.abs1();
        if s == T::Real::ZERO {
            w[kbot - 1] = Complex::new(T::Real::ZERO, T::Real::ZERO);
            w[kbot] = w[kbot - 1];
            return;
        }
        let r = T::Real::ONE / s;
        let (a, b, c, d) = (a.scale(r), b.scale(r), c.scale(r), d.scale(r));
        let tr = (a + d).scale(T::Real::HALF);
        let det = (a - tr) * (d - tr) - b * c;
        let root = (-det).sqrt();
        w[kbot - 1] = (tr + root).scale(s).to_complex();
        w[kbot] = (tr - root).scale(s).to_complex();
    } else {
        let s = standardize(a.re(), b.re(), c.re(), d.re());
        w[kbot - 1] = Complex::new(s.rt1r, s.rt1i);
        w[kbot] = Complex::new(s.rt2r, s.rt2i);
    }
}

/// Orders real shifts so that each consecutive pair is two reals or two conjugates, those
/// already next to each other.
fn pair_up<R: Float>(s: &mut [Complex<R>]) {
    let mut i = s.len();
    while i >= 3 {
        if s[i - 1].im != -s[i - 2].im {
            s[i - 3..i].rotate_right(1);
        }
        i -= 2;
    }
}

/// A multiple of the first column of `(H - s₁·I)·(H - s₂·I)`, `H` the block of order `nb`, 2 or
/// 3, at row and column `r` of `h`, as LAPACK's `laqr1`: real for real elements, the shifts
/// being two reals or two conjugates.
fn first_column<T: Scalar>(
    h: &[T],
    n: usize,
    r: usize,
    nb: usize,
    s1: Complex<T::Real>,
    s2: Complex<T::Real>,
) -> [T; 3] {
    let at = |i: usize, j: usize| r + i + (r + j) * n;
    let zero = T::Real::ZERO;
    if T::IS_COMPLEX {
        let (s1, s2) = (T::from_parts(s1.re, s1.im), T::from_parts(s2.re, s2.im));
        let h11 = h[at(0, 0)];
        let h21 = h[at(1, 0)];
        let h31 = if nb == 3 { h[at(2, 0)] } else { T::zero() };
        let s = (h11 - s2).abs1() + h21.abs1() + h31.abs1();
        if s == zero {
            return [T::zero(); 3];
        }
        let rs = T::Real::ONE / s;
        let (h21s, h31s) = (h21.scale(rs), h31.scale(rs));
        let mut v0 = (h11 - s1) * (h11 - s2).scale(rs) + h[at(0, 1)] * h21s;
        let mut v1 = h21s * (h11 + h[at(1, 1)] - s1 - s2);
        let mut v2 = T::zero();
        if nb == 3 {
            v0 += h[at(0, 2)] * h31s;
            v1 += h[at(1, 2)] * h31s;
            v2 = h31s * (h11 + h[at(2, 2)] - s1 - s2) + h21s * h[at(2, 1)];
        }
        [v0, v1, v2]
    } else {
        let g = |i: usize, j: usize| h[at(i, j)].re();
        let (h11, h21) = (g(0, 0), g(1, 0));
        let h31 = if nb == 3 { g(2, 0) } else { zero };
        let s = (h11 - s2.re).abs() + s2.im.abs() + h21.abs() + h31.abs();
        if s == zero {
            return [T::zero(); 3];
        }
        let (h21s, h31s) = (h21 / s, h31 / s);
        let sum = s1.re + s2.re;
        let mut v0 = (h11 - s1.re) * ((h11 - s2.re) / s) - s1.im * (s2.im / s) + g(0, 1) * h21s;
        let mut v1 = h21s * (h11 + g(1, 1) - sum);
        let mut v2 = zero;
        if nb == 3 {
            v0 += g(0, 2) * h31s;
            v1 += g(1, 2) * h31s;
            v2 = h31s * (h11 + g(2, 2) - sum) + h21s * g(2, 1);
        }
        [T::from_real(v0), T::from_real(v1), T::from_real(v2)]
    }
}

/// Sets the subdiagonal `h[k + 1, k]` to zero when it is negligible both by the usual test and
/// by Ahues and Tisseur's, for the vigilant deflation inside a sweep.
#[allow(clippy::too_many_arguments)]
fn vigilant<T: Scalar>(
    h: &mut [T],
    n: usize,
    k: usize,
    ktop: usize,
    kbot: usize,
    smlnum: T::Real,
    ulp: T::Real,
) {
    let at = |i: usize, j: usize| i + j * n;
    if h[at(k + 1, k)] == T::zero() {
        return;
    }
    let mut tst1 = h[at(k, k)].abs1() + h[at(k + 1, k + 1)].abs1();
    if tst1 == T::Real::ZERO {
        for d in 1..=3 {
            if k >= ktop + d {
                tst1 += h[at(k, k - d)].abs1();
            }
        }
        for d in 2..=4 {
            if k + d <= kbot {
                tst1 += h[at(k + d, k + 1)].abs1();
            }
        }
    }
    let sub = h[at(k + 1, k)].abs1();
    if sub <= smlnum.max(ulp * tst1) {
        let up = h[at(k, k + 1)].abs1();
        let (h12, h21) = (sub.max(up), sub.min(up));
        let d = (h[at(k, k)] - h[at(k + 1, k + 1)]).abs1();
        let hk = h[at(k + 1, k + 1)].abs1();
        let (h11, h22) = (hk.max(d), hk.min(d));
        let scl = h11 + h12;
        let tst2 = h22 * (h11 / scl);
        if tst2 == T::Real::ZERO || h21 * (h12 / scl) <= smlnum.max(ulp * tst2) {
            h[at(k + 1, k)] = T::zero();
        }
    }
}

/// A reflection `I - τ·v·v*`, `v = (1, v₂, v₃)`, as `[τ, v₂, v₃]`.
type Reflection<T> = [T; 3];

/// `x - τ·(x·v)·v*` on the elements `x₁..x₃` of a row, the reflection from the right.
#[inline(always)]
fn right3<T: Scalar>(r: Reflection<T>, x1: &mut T, x2: &mut T, x3: &mut T) {
    let [tau, v2, v3] = r;
    let sum = tau * (*x1 + v2 * *x2 + v3 * *x3);
    *x1 -= sum;
    *x2 -= sum * v2.conj();
    *x3 -= sum * v3.conj();
}

/// `x - conj(τ)·v·(v*·x)` on the elements `x₁..x₃` of a column, the reflection's conjugate
/// transpose from the left.
#[inline(always)]
fn left3<T: Scalar>(r: Reflection<T>, x1: &mut T, x2: &mut T, x3: &mut T) {
    let [tau, v2, v3] = r;
    let sum = tau.conj() * (*x1 + v2.conj() * *x2 + v3.conj() * *x3);
    *x1 -= sum;
    *x2 -= sum * v2;
    *x3 -= sum * v3;
}

/// Three columns `c..c + 3` of the column-major `a` with the leading dimension `ld`, to change.
#[inline(always)]
fn columns3<T>(a: &mut [T], ld: usize, c: usize) -> (&mut [T], &mut [T], &mut [T]) {
    let (x, rest) = a[c * ld..].split_at_mut(ld);
    let (y, rest) = rest.split_at_mut(ld);
    (x, y, &mut rest[..ld])
}

/// Applies the reflection from the right to rows `rows` of columns `c..c + 3`.
#[inline(always)]
fn right3_columns<T: Scalar>(
    r: Reflection<T>,
    a: &mut [T],
    ld: usize,
    c: usize,
    rows: core::ops::Range<usize>,
) {
    let (x, y, z) = columns3(a, ld, c);
    for ((x, y), z) in x[rows.clone()]
        .iter_mut()
        .zip(&mut y[rows.clone()])
        .zip(&mut z[rows])
    {
        right3(r, x, y, z);
    }
}

/// One sweep of the small-bulge multishift QR algorithm on the active block `ktop..=kbot` of the
/// upper Hessenberg `h`, with the shifts `s`, as LAPACK's `laqr5`: chains of bulges two shifts
/// each, chased `2·(bulges)` columns at a time, the reflections of each step gathered into `U`
/// and applied to the far parts of `h` and to `z` by products of blocks when `accumulate` is
/// set.
#[allow(clippy::too_many_arguments)]
fn sweep<T: Scalar>(
    want_t: bool,
    h: &mut [T],
    n: usize,
    mut z: Option<&mut [T]>,
    ktop: usize,
    kbot: usize,
    s: &mut [Complex<T::Real>],
    accumulate: bool,
) {
    let at = |i: usize, j: usize| i + j * n;
    if s.len() < 2 || ktop >= kbot {
        return;
    }
    if !T::IS_COMPLEX {
        // Pairs from the start, an odd one out last.
        let mut i = 0;
        while i + 2 < s.len() {
            if s[i].im != -s[i + 1].im {
                s[i..i + 3].rotate_left(1);
            }
            i += 2;
        }
    }
    let ns = s.len() - s.len() % 2;
    let ulp = T::Real::EPSILON;
    let smlnum = T::Real::MIN_POSITIVE * (T::Real::from_f64(n as f64) / ulp);
    if ktop + 2 <= kbot {
        h[at(ktop + 2, ktop)] = T::zero();
    }
    let nbmps = ns / 2;
    let kdu = 4 * nbmps;
    let mut v: Vec<Reflection<T>> = vec![[T::zero(); 3]; nbmps];
    let mut u = if accumulate {
        vec![T::zero(); kdu * kdu]
    } else {
        Vec::new()
    };
    let (top, bot) = (ktop as isize, kbot as isize);
    let step = 2 * nbmps as isize;
    // Each chunk moves the chain from `incol` (perhaps before the block, the bulges not yet in)
    // `step` columns down; `U`'s row and column `c` is `h`'s `incol + 1 + c`.
    let mut incol = top - step + 1;
    while incol <= bot - 2 {
        let ndcol = incol + kdu as isize;
        let jtop = if accumulate {
            ktop.max(incol.max(0) as usize)
        } else if want_t {
            0
        } else {
            ktop
        };
        let jbot = if accumulate {
            ndcol.min(bot) as usize
        } else if want_t {
            n - 1
        } else {
            kbot
        };
        if accumulate {
            u.fill(T::zero());
            for i in 0..kdu {
                u[i + i * kdu] = T::one();
            }
        }
        let u_first = (top - incol - 1).max(0) as usize;
        for krcol in incol..=(incol + step - 1).min(bot - 2) {
            // Bulges `mtop..mbot` are in the block; bulge `m` acts on rows and columns
            // `r..r + 3`, `r = krcol + 1 + 2m`, made from column `r - 1`.
            let mtop = ((top - krcol).max(0) / 2) as usize;
            let mbot = (nbmps as isize).min((bot - krcol - 1) / 2) as usize;
            let row = |m: usize| (krcol + 1 + 2 * m as isize) as usize;
            if mbot < nbmps && row(mbot) + 1 == kbot {
                // A last bulge with room for only a 2 × 2 reflection.
                let m = mbot;
                let r = row(m);
                let (tau, v2) = if r == ktop {
                    let x = first_column(h, n, r, 2, s[2 * m], s[2 * m + 1]);
                    let mut rest = [x[1]];
                    let (tau, _) = short_reflector(x[0], &mut rest);
                    (tau, rest[0])
                } else {
                    let mut rest = [h[at(r + 1, r - 1)]];
                    let (tau, beta) = short_reflector(h[at(r, r - 1)], &mut rest);
                    h[at(r, r - 1)] = beta;
                    h[at(r + 1, r - 1)] = T::zero();
                    (tau, rest[0])
                };
                v[m] = [tau, v2, T::zero()];
                let (x, y) = h[r * n..].split_at_mut(n);
                for (x, y) in x[jtop..=kbot].iter_mut().zip(&mut y[jtop..=kbot]) {
                    let sum = tau * (*x + v2 * *y);
                    *x -= sum;
                    *y -= sum * v2.conj();
                }
                let (tc, v2c) = (tau.conj(), v2.conj());
                for j in r..=jbot {
                    let sum = tc * (h[at(r, j)] + v2c * h[at(r + 1, j)]);
                    h[at(r, j)] -= sum;
                    h[at(r + 1, j)] -= sum * v2;
                }
                if r > ktop {
                    vigilant(h, n, r - 1, ktop, kbot, smlnum, ulp);
                }
                let (target, ld, c, rows) = if accumulate {
                    (
                        Some(&mut u[..]),
                        kdu,
                        (r as isize - incol - 1) as usize,
                        u_first..kdu,
                    )
                } else {
                    (z.as_deref_mut(), n, r, 0..n)
                };
                if let Some(target) = target {
                    let (x, y) = target[c * ld..].split_at_mut(ld);
                    for (x, y) in x[rows.clone()].iter_mut().zip(&mut y[rows]) {
                        let sum = tau * (*x + v2 * *y);
                        *x -= sum;
                        *y -= sum * v2.conj();
                    }
                }
            }
            // The 3 × 3 bulges, from the bottom one up: each moved down a column, applied from
            // the right and to its next column from the left (what the deflation test needs),
            // the rest of the left side after.
            for m in (mtop..mbot).rev() {
                let r = row(m);
                if r == ktop {
                    let x = first_column(h, n, r, 3, s[2 * m], s[2 * m + 1]);
                    let mut rest = [x[1], x[2]];
                    let (tau, _) = short_reflector(x[0], &mut rest);
                    v[m] = [tau, rest[0], rest[1]];
                } else {
                    // The previous step's reflection on the row below the bulge, whose first
                    // two elements are zero.
                    let [tau, v2, v3] = v[m];
                    let sum = tau * v3 * h[at(r + 2, r + 1)];
                    h[at(r + 2, r - 1)] = -sum;
                    h[at(r + 2, r)] = -(sum * v2.conj());
                    h[at(r + 2, r + 1)] -= sum * v3.conj();
                    let mut rest = [h[at(r + 1, r - 1)], h[at(r + 2, r - 1)]];
                    let (tau, beta) = short_reflector(h[at(r, r - 1)], &mut rest);
                    let moved = [tau, rest[0], rest[1]];
                    if h[at(r + 2, r - 1)] != T::zero()
                        || h[at(r + 2, r)] != T::zero()
                        || h[at(r + 2, r + 1)] == T::zero()
                    {
                        h[at(r, r - 1)] = beta;
                        v[m] = moved;
                    } else {
                        // The bulge collapsed: a fresh one from the shifts, if its fill is
                        // negligible.
                        let x = first_column(h, n, r, 3, s[2 * m], s[2 * m + 1]);
                        let mut rest = [x[1], x[2]];
                        let (tau, _) = short_reflector(x[0], &mut rest);
                        let fresh = [tau, rest[0], rest[1]];
                        let sum =
                            tau.conj() * (h[at(r, r - 1)] + rest[0].conj() * h[at(r + 1, r - 1)]);
                        let fill =
                            (h[at(r + 1, r - 1)] - sum * rest[0]).abs1() + (sum * rest[1]).abs1();
                        let near = h[at(r - 1, r - 1)].abs1()
                            + h[at(r, r)].abs1()
                            + h[at(r + 1, r + 1)].abs1();
                        if fill > ulp * near {
                            h[at(r, r - 1)] = beta;
                            v[m] = moved;
                        } else {
                            h[at(r, r - 1)] -= sum;
                            v[m] = fresh;
                        }
                    }
                    h[at(r + 1, r - 1)] = T::zero();
                    h[at(r + 2, r - 1)] = T::zero();
                }
                let reflection = v[m];
                right3_columns(reflection, h, n, r, jtop..kbot.min(r + 2) + 1);
                let (mut x1, mut x2, mut x3) = (h[at(r, r)], h[at(r + 1, r)], h[at(r + 2, r)]);
                left3(reflection, &mut x1, &mut x2, &mut x3);
                (h[at(r, r)], h[at(r + 1, r)], h[at(r + 2, r)]) = (x1, x2, x3);
                if r > ktop {
                    vigilant(h, n, r - 1, ktop, kbot, smlnum, ulp);
                }
            }
            for m in (mtop..mbot).rev() {
                let r = row(m);
                let reflection = v[m];
                for j in ktop.max(r + 1)..=jbot {
                    let column = &mut h[j * n + r..j * n + r + 3];
                    let [x1, x2, x3] = column else { unreachable!() };
                    left3(reflection, x1, x2, x3);
                }
            }
            if accumulate {
                for m in (mtop..mbot).rev() {
                    let r = row(m);
                    let c = (r as isize - incol - 1) as usize;
                    let i2 = u_first.max(2 * m);
                    let i4 = kdu.min((krcol - incol) as usize + 2 * (mbot - 1) + 5);
                    right3_columns(v[m], &mut u, kdu, c, i2..i4);
                }
            } else if let Some(z) = z.as_deref_mut() {
                for m in (mtop..mbot).rev() {
                    right3_columns(v[m], z, n, row(m), 0..n);
                }
            }
        }
        if accumulate {
            // `U` on the far parts: `U*` on the rows of the window to their right, `U` on the
            // columns of the window above it and in `Z`.
            let (jtop, jbot) = if want_t { (0, n - 1) } else { (ktop, kbot) };
            let k1 = u_first;
            let nu = kdu - (ndcol - bot).max(0) as usize - k1;
            let h0 = (incol + 1 + k1 as isize) as usize;
            let mut uw = vec![T::zero(); nu * nu];
            let mut uh = vec![T::zero(); nu * nu];
            for j in 0..nu {
                for i in 0..nu {
                    let x = u[k1 + i + (k1 + j) * kdu];
                    uw[i + j * nu] = x;
                    uh[j + i * nu] = x.conj();
                }
            }
            let c0 = (ndcol.min(bot) + 1) as usize;
            if c0 <= jbot {
                let cols = jbot + 1 - c0;
                let mut tmp = vec![T::zero(); nu * cols];
                let src = &h[h0 + c0 * n..];
                gemm(
                    nu,
                    cols,
                    nu,
                    T::one(),
                    &uh,
                    nu,
                    Op::N,
                    src,
                    n,
                    Op::N,
                    T::zero(),
                    &mut tmp,
                    nu,
                );
                for j in 0..cols {
                    h[h0 + (c0 + j) * n..h0 + (c0 + j) * n + nu]
                        .copy_from_slice(&tmp[j * nu..(j + 1) * nu]);
                }
            }
            let r1 = ktop.max(incol.max(0) as usize);
            if jtop < r1 {
                times_u(&mut h[h0 * n..], n, jtop..r1, &uw, nu);
            }
            if let Some(z) = z.as_deref_mut() {
                times_u(&mut z[h0 * n..], n, 0..n, &uw, nu);
            }
        }
        incol += step;
    }
}

/// `A[rows, 0..k] = A[rows, 0..k]·U`, `U` of `k × k`, by a product into a copy.
fn times_u<T: Scalar>(a: &mut [T], ld: usize, rows: core::ops::Range<usize>, u: &[T], k: usize) {
    let m = rows.len();
    let mut tmp = vec![T::zero(); m * k];
    gemm(
        m,
        k,
        k,
        T::one(),
        &a[rows.start..],
        ld,
        Op::N,
        u,
        k,
        Op::N,
        T::zero(),
        &mut tmp,
        m,
    );
    for j in 0..k {
        a[j * ld + rows.start..j * ld + rows.end].copy_from_slice(&tmp[j * m..(j + 1) * m]);
    }
}
