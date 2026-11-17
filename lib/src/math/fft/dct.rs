//! Discrete cosine and sine transforms of types I to IV, unscaled as FFTW's `REDFT` and `RODFT`
//! kinds are, in `O(n log n)` by FFTW's accurate algorithms on real transforms:
//!
//! * II and III by Makhoul's reordering: the even-indexed values, then the odd-indexed ones
//!   backwards, through the real transform of `n`, with a twiddle `e^(−πi·k/2n)` on each value
//!   of the spectrum (FFTW's `reodft010e-r2hc`).
//! * IV of an even `n` through the complex transform of `n/2` of `x[2j] + i·x[n − 1 − 2j]`, with
//!   twiddles before and after; of an odd `n` through the real transform of `n` of a permutation
//!   of the values with signs (Chan and Ho, FFTW's `reodft11e-r2hc-odd`).
//! * I of an odd `n` by the split-radix step of FFTW's `reodft00e-splitradix`: the type I of the
//!   `(n + 1)/2` even-indexed values (DST-I: of the `(n − 1)/2` odd-indexed ones) and the type II
//!   of the others, recursively; of an even `n` through the real transform of the symmetric
//!   extension, `2(n − 1)` values (DST-I: `2(n + 1)`).
//! * The sine transforms by the same steps with the signs and orders that turn each into the
//!   cosine transform of its type (`DST-II(x)[k] = DCT-II((−1)^j·x[j])[n − 1 − k]`, and so on).
//!
//! The shortest lengths (up to 8, 16 for type I) are summed directly from a table of the whole
//! matrix.

use super::real::{as_complex_mut, as_reals_mut, roots, with_work};
use super::twiddle;
use super::{Fft, FftFloat, RealFft, check_batch};
use crate::math::complex::Complex;
use std::sync::Arc;

/// The type of a discrete cosine transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DctKind {
    /// DCT-I (FFTW's `REDFT00`), for `n ≥ 2`:
    /// `Y[k] = X[0] + (−1)^k·X[n−1] + 2·Σ_{j=1}^{n−2} X[j]·cos(π·jk/(n−1))`.
    I,
    /// DCT-II (`REDFT10`, "the" DCT): `Y[k] = 2·Σ X[j]·cos(π·(j + 1/2)·k/n)`.
    II,
    /// DCT-III (`REDFT01`, the inverse of DCT-II up to `2n`):
    /// `Y[k] = X[0] + 2·Σ_{j=1}^{n−1} X[j]·cos(π·j·(k + 1/2)/n)`.
    III,
    /// DCT-IV (`REDFT11`): `Y[k] = 2·Σ X[j]·cos(π·(j + 1/2)·(k + 1/2)/n)`.
    IV,
}

/// The type of a discrete sine transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DstKind {
    /// DST-I (FFTW's `RODFT00`): `Y[k] = 2·Σ X[j]·sin(π·(j + 1)·(k + 1)/(n + 1))`.
    I,
    /// DST-II (`RODFT10`): `Y[k] = 2·Σ X[j]·sin(π·(j + 1/2)·(k + 1)/n)`.
    II,
    /// DST-III (`RODFT01`, the inverse of DST-II up to `2n`):
    /// `Y[k] = (−1)^k·X[n−1] + 2·Σ_{j=0}^{n−2} X[j]·sin(π·(j + 1)·(k + 1/2)/n)`.
    III,
    /// DST-IV (`RODFT11`): `Y[k] = 2·Σ X[j]·sin(π·(j + 1/2)·(k + 1/2)/n)`.
    IV,
}

/// A discrete cosine transform of real sequences of the length `n`, planned once.
///
/// The transforms are unscaled, as FFTW's: DCT-I is its own inverse up to the factor
/// `2(n − 1)`, DCT-II and DCT-III are each other's up to `2n`, and DCT-IV is its own up to `2n`.
/// [`transform`](Self::transform) takes a batch of sequences one after the other, and
/// [`transform_with_scratch`](Self::transform_with_scratch) its working memory from the caller,
/// at least [`scratch_len`](Self::scratch_len) complex values, and then never allocates. A plan
/// is immutable and shared by its clones.
///
/// # Examples
///
/// ```
/// use a::math::fft::{Dct, DctKind};
///
/// let dct = Dct::<f64>::new(4, DctKind::II);
/// let mut x = [1.0, 1.0, 1.0, 1.0];
/// dct.transform(&mut x);
/// assert!((x[0] - 8.0).abs() < 1e-12);
/// assert!(x[1..].iter().all(|v| v.abs() < 1e-12));
///
/// let back = Dct::<f64>::new(4, DctKind::III);
/// back.transform(&mut x);
/// assert!(x.iter().all(|v| (v - 8.0).abs() < 1e-12)); // 2n·x
/// ```
#[derive(Clone)]
pub struct Dct<T> {
    n: usize,
    kind: DctKind,
    /// The values of working memory one sequence takes.
    work: usize,
    plan: Arc<Plan<T>>,
}

/// A discrete sine transform of real sequences of the length `n`, planned once; as [`Dct`]: DST-I
/// is its own inverse up to `2(n + 1)`, DST-II and DST-III each other's up to `2n`, DST-IV its own
/// up to `2n`. The same methods, [`transform_with_scratch`](Self::transform_with_scratch) among
/// them.
///
/// # Examples
///
/// ```
/// use a::math::fft::{Dst, DstKind};
///
/// let dst = Dst::<f64>::new(3, DstKind::I);
/// let mut x = [1.0, 0.0, 0.0];
/// dst.transform(&mut x);
/// dst.transform(&mut x);
/// assert!((x[0] - 8.0).abs() < 1e-12 && x[1].abs() < 1e-12 && x[2].abs() < 1e-12);
/// ```
#[derive(Clone)]
pub struct Dst<T> {
    n: usize,
    kind: DstKind,
    /// The values of working memory one sequence takes.
    work: usize,
    plan: Arc<Plan<T>>,
}

/// The longest lengths summed directly, from the table of the matrix: of types II to IV, and of
/// type I (whose fast steps cost more), measured on an Apple M4 Max.
const DIRECT: usize = 8;
const DIRECT_I: usize = 16;

/// A family of transforms: the cosines or the sines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fam {
    Cos,
    Sin,
}

/// The algorithm behind a plan of one family, type and length.
enum Plan<T> {
    /// Length 0.
    Empty,
    /// The matrix column after column, each padded to 16 entries with zeros:
    /// `y[k] = Σ_j mat[16j + k]·x[j]`, the 16 sums held in registers.
    Direct { n: usize, mat: Vec<T> },
    /// Types II and III by Makhoul's reordering and the real transform of `n`;
    /// `w[k] = e^(−πi·k/2n)` for `k ≤ n/2`.
    Makhoul {
        n: usize,
        rfft: RealFft<T>,
        w: Vec<Complex<T>>,
    },
    /// Type IV of an even `n` by the complex transform of `n/2`: `pre[j] = e^(−πi·j/n)`,
    /// `post[p] = 2·e^(−πi·(4p + 1)/4n)`.
    FourEven {
        n: usize,
        fft: Fft<T>,
        pre: Vec<Complex<T>>,
        post: Vec<Complex<T>>,
    },
    /// Type IV of an odd `n` by the real transform of `n` (Chan and Ho).
    FourOdd { n: usize, rfft: RealFft<T> },
    /// Type I of an odd `n` by the split-radix step: the type I `a` of the values of one parity
    /// and the type II `b` of the others.
    SplitI {
        n: usize,
        a: Box<Plan<T>>,
        b: Box<Plan<T>>,
    },
    /// Type I through the real transform of the symmetric extension.
    PadI { n: usize, rfft: RealFft<T> },
}

/// The entry `(k, j)` of the matrix of a transform, in `f64`.
fn entry(fam: Fam, kind: u8, n: usize, k: usize, j: usize) -> f64 {
    // cos(π·a/b) and sin(π·a/b) from e^(−2πi·a/2b).
    let cos = |a: usize, b: usize| twiddle::root(a % (2 * b), 2 * b).0;
    let sin = |a: usize, b: usize| -twiddle::root(a % (2 * b), 2 * b).1;
    match (fam, kind) {
        (Fam::Cos, 1) => {
            let c = if j == 0 || j == n - 1 { 1.0 } else { 2.0 };
            c * cos(j * k, n - 1)
        }
        (Fam::Cos, 2) => 2.0 * cos((2 * j + 1) * k, 2 * n),
        (Fam::Cos, 3) => (if j == 0 { 1.0 } else { 2.0 }) * cos(j * (2 * k + 1), 2 * n),
        (Fam::Cos, _) => 2.0 * cos((2 * j + 1) * (2 * k + 1), 4 * n),
        (Fam::Sin, 1) => 2.0 * sin((j + 1) * (k + 1), n + 1),
        (Fam::Sin, 2) => 2.0 * sin((2 * j + 1) * (k + 1), 2 * n),
        (Fam::Sin, 3) => (if j == n - 1 { 1.0 } else { 2.0 }) * sin((j + 1) * (2 * k + 1), 2 * n),
        (Fam::Sin, _) => 2.0 * sin((2 * j + 1) * (2 * k + 1), 4 * n),
    }
}

/// `x` with the sign `−` when `odd`.
#[inline(always)]
fn sign<T: FftFloat>(x: T, odd: bool) -> T {
    if odd { -x } else { x }
}

impl<T: FftFloat> Plan<T> {
    fn new(fam: Fam, kind: u8, n: usize) -> Plan<T> {
        if n == 0 {
            return Plan::Empty;
        }
        if n <= if kind == 1 { DIRECT_I } else { DIRECT } {
            let mut mat = Vec::with_capacity(16 * n);
            for j in 0..n {
                for k in 0..16 {
                    mat.push(if k < n {
                        T::from_f64(entry(fam, kind, n, k, j))
                    } else {
                        T::ZERO
                    });
                }
            }
            return Plan::Direct { n, mat };
        }
        match kind {
            2 | 3 => Plan::Makhoul {
                n,
                rfft: RealFft::new(n),
                w: roots::table(4 * n, n / 2 + 1),
            },
            4 if n.is_multiple_of(2) => Plan::FourEven {
                n,
                fft: Fft::new(n / 2),
                pre: roots::table(2 * n, n / 2),
                post: roots::affine(8 * n, 1, 4, n / 2)
                    .into_iter()
                    .map(|w: Complex<T>| w * T::TWO)
                    .collect(),
            },
            4 => Plan::FourOdd {
                n,
                rfft: RealFft::new(n),
            },
            _ if n % 2 == 1 => {
                let (la, lb) = match fam {
                    Fam::Cos => (n.div_ceil(2), n / 2),
                    Fam::Sin => (n / 2, n.div_ceil(2)),
                };
                Plan::SplitI {
                    n,
                    a: Box::new(Plan::new(fam, 1, la)),
                    b: Box::new(Plan::new(fam, 2, lb)),
                }
            }
            _ => Plan::PadI {
                n,
                rfft: RealFft::new(match fam {
                    Fam::Cos => 2 * (n - 1),
                    Fam::Sin => 2 * (n + 1),
                }),
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            Plan::Empty => 0,
            Plan::Direct { n, .. }
            | Plan::Makhoul { n, .. }
            | Plan::FourEven { n, .. }
            | Plan::FourOdd { n, .. }
            | Plan::SplitI { n, .. }
            | Plan::PadI { n, .. } => *n,
        }
    }

    /// The complex values of working memory [`run`](Self::run) takes.
    fn work_len(&self) -> usize {
        match self {
            Plan::Empty => 0,
            Plan::Direct { n, .. } => n.div_ceil(2),
            Plan::Makhoul { n, rfft, .. } | Plan::FourOdd { n, rfft } => {
                n / 2 + 1 + n.div_ceil(2) + rfft.work_len()
            }
            Plan::FourEven { n, fft, .. } => n / 2 + fft.scratch_len(),
            Plan::SplitI { a, b, .. } => {
                a.len().div_ceil(2) + b.len().div_ceil(2) + a.work_len().max(b.work_len())
            }
            Plan::PadI { rfft, .. } => rfft.len() / 2 + 1 + rfft.work_len(),
        }
    }

    /// The transform of the type `kind` of the family `fam` of `x` (or of `y` itself when `x`
    /// is `None`) into `y`.
    fn run(&self, fam: Fam, kind: u8, x: Option<&[T]>, y: &mut [T], work: &mut [Complex<T>]) {
        match self {
            Plan::Empty => {}
            Plan::Direct { n, mat } => {
                let n = *n;
                let buf = &mut as_reals_mut(&mut work[..n.div_ceil(2)])[..n];
                buf.copy_from_slice(x.unwrap_or(y));
                let mut acc = [T::ZERO; 16];
                for (&x, col) in buf.iter().zip(mat.as_chunks::<16>().0) {
                    for (a, &m) in acc.iter_mut().zip(col) {
                        *a += m * x;
                    }
                }
                y.copy_from_slice(&acc[..n]);
            }
            Plan::Makhoul { n, rfft, w } => {
                let n = *n;
                let (spec, rest) = work.split_at_mut(n / 2 + 1);
                let (v, rest) = rest.split_at_mut(n.div_ceil(2));
                let odd = fam == Fam::Sin;
                if kind == 2 {
                    let xs = x.unwrap_or(y);
                    if rfft.is_half() {
                        reorder(xs, as_reals_mut(&mut spec[..n / 2]), odd);
                        rfft.forward_packed(spec, rest);
                    } else {
                        let v = &mut as_reals_mut(v)[..n];
                        reorder(xs, v, odd);
                        rfft.forward_run(v, spec, rest);
                    }
                    makhoul_post(fam, n, spec, w, y);
                } else {
                    makhoul_pre(fam, n, x.unwrap_or(y), w, spec);
                    if rfft.is_half() {
                        rfft.inverse_packed(spec, v, rest);
                        unreorder(as_reals_mut(v), y, odd);
                    } else {
                        let v = &mut as_reals_mut(v)[..n];
                        rfft.inverse_run(spec, v, rest);
                        unreorder(v, y, odd);
                    }
                }
            }
            Plan::FourEven { n, fft, pre, post } => {
                let (buf, work) = work.split_at_mut(n / 2);
                let src = match x {
                    Some(x) => x,
                    None => {
                        let b = as_reals_mut(buf);
                        b.copy_from_slice(y);
                        b
                    }
                };
                four_even_pre(fam, pre, src, y);
                fft.forward_with_scratch(as_complex_mut(y), work);
                four_even_post(fam, *n, post, y);
            }
            Plan::FourOdd { n, rfft } => {
                let n = *n;
                let (spec, rest) = work.split_at_mut(n / 2 + 1);
                let (v, rest) = rest.split_at_mut(n.div_ceil(2));
                let v = &mut as_reals_mut(v)[..n];
                four_odd_permute(fam, x.unwrap_or(y), v);
                rfft.forward_run(v, spec, rest);
                four_odd_post(fam, n, spec, y);
            }
            Plan::SplitI { n, a, b } => {
                let n = *n;
                let (la, lb) = (a.len(), b.len());
                let (ea, rest) = work.split_at_mut(la.div_ceil(2));
                let (eb, rest) = rest.split_at_mut(lb.div_ceil(2));
                let (ea, eb) = (&mut as_reals_mut(ea)[..la], &mut as_reals_mut(eb)[..lb]);
                let xs = x.unwrap_or(y);
                // Cosines: a of the even-indexed values, b of the odd ones; sines the other way.
                let (pa, pb) = if fam == Fam::Cos { (0, 1) } else { (1, 0) };
                for (i, v) in ea.iter_mut().enumerate() {
                    *v = xs[2 * i + pa];
                }
                for (i, v) in eb.iter_mut().enumerate() {
                    *v = xs[2 * i + pb];
                }
                a.run(fam, 1, None, ea, rest);
                b.run(fam, 2, None, eb, rest);
                if fam == Fam::Cos {
                    // y[k] = A[k] + B[k], y[N − k] = A[k] − B[k] for N = n − 1, B[N/2] = 0.
                    let h = lb;
                    for k in 0..h {
                        y[k] = ea[k] + eb[k];
                        y[n - 1 - k] = ea[k] - eb[k];
                    }
                    y[h] = ea[h];
                } else {
                    // With N = n + 1, from 1: y[k] = A[k] + B[k], y[N − k] = B[k] − A[k],
                    // y[N/2] = B[N/2].
                    for k in 0..la {
                        y[k] = ea[k] + eb[k];
                        y[n - 1 - k] = eb[k] - ea[k];
                    }
                    y[la] = eb[la];
                }
            }
            Plan::PadI { n, rfft } => {
                let n = *n;
                let l = rfft.len();
                let (spec, rest) = work.split_at_mut(l / 2 + 1);
                {
                    let xs = x.unwrap_or(y);
                    let e = as_reals_mut(&mut spec[..l / 2]);
                    if fam == Fam::Cos {
                        // The even extension x[0], …, x[n − 1], x[n − 2], …, x[1].
                        e[..n].copy_from_slice(xs);
                        for j in n..l {
                            e[j] = xs[l - j];
                        }
                    } else {
                        // The odd extension 0, x[0], …, x[n − 1], 0, −x[n − 1], …, −x[0].
                        e[0] = T::ZERO;
                        e[1..=n].copy_from_slice(xs);
                        e[n + 1] = T::ZERO;
                        for j in n + 2..l {
                            e[j] = -xs[l - j - 1];
                        }
                    }
                }
                // The extension is longer than 16, above the real transform's direct sums.
                debug_assert!(rfft.is_half());
                rfft.forward_packed(spec, rest);
                if fam == Fam::Cos {
                    for (y, s) in y.iter_mut().zip(spec.iter()) {
                        *y = s.re;
                    }
                } else {
                    for (y, s) in y.iter_mut().zip(spec[1..].iter()) {
                        *y = -s.im;
                    }
                }
            }
        }
    }
}

// The passes below index from both ends of slices cut to the lengths they cover, a form the
// compiler vectorises (zipped reversed iterators it does not).

/// Makhoul's order: `v[i] = x[2i]` and `v[n − 1 − i] = ±x[2i + 1]` (`−` when `odd`).
#[inline]
fn reorder<T: FftFloat>(x: &[T], v: &mut [T], odd: bool) {
    let n = x.len();
    let (h, c) = (n / 2, n.div_ceil(2));
    let s = if odd { -T::ONE } else { T::ONE };
    let (front, back) = v[..n].split_at_mut(c);
    reorder_halves(&x[..2 * h], &mut front[..h], &mut back[..h], s);
    if c > h {
        front[h] = x[n - 1];
    }
}

/// `front[i] = x[2i]`, `back[h − 1 − i] = s·x[2i + 1]`.
#[inline]
fn reorder_halves<T: FftFloat>(x: &[T], front: &mut [T], back: &mut [T], s: T) {
    let h = front.len();
    let (x, back) = (&x[..2 * h], &mut back[..h]);
    for i in 0..h {
        front[i] = x[2 * i];
        back[h - 1 - i] = x[2 * i + 1] * s;
    }
}

/// The inverse of [`reorder`]: `y[2i] = v[i]`, `y[2i + 1] = ±v[n − 1 − i]`.
#[inline]
fn unreorder<T: FftFloat>(v: &[T], y: &mut [T], odd: bool) {
    let n = y.len();
    let (h, c) = (n / 2, n.div_ceil(2));
    let s = if odd { -T::ONE } else { T::ONE };
    let (front, back) = (&v[..h], &v[c..c + h]);
    let y2 = &mut y[..2 * h];
    for i in 0..h {
        y2[2 * i] = front[i];
        y2[2 * i + 1] = back[h - 1 - i] * s;
    }
    if c > h {
        y[n - 1] = v[h];
    }
}

/// Types II from the spectrum `v` of the reordered values: with `c = w[k]·V[k]`,
/// `y[k] = 2·Re c` and `y[n − k] = −2·Im c` (DCT-II), the outputs reversed for DST-II.
#[inline]
fn makhoul_post<T: FftFloat>(fam: Fam, n: usize, v: &[Complex<T>], w: &[Complex<T>], y: &mut [T]) {
    let two = T::TWO;
    // The pairs k, n − k for 1 ≤ k ≤ q, and the middle k = n/2 of an even n.
    let q = (n - 1) / 2;
    let (vs, ws) = (&v[1..=q], &w[1..=q]);
    let mid = n.is_multiple_of(2).then(|| (w[n / 2] * v[n / 2]).re * two);
    let y0 = two * v[0].re;
    let y = &mut y[..n];
    if fam == Fam::Cos {
        // y[1..=q] ascending, y[n − q..n] descending.
        let (lo, hi) = y.split_at_mut(q + 1);
        let off = hi.len() - q;
        let (mhi, back) = hi.split_at_mut(off);
        post_pairs(vs, ws, &mut lo[1..], &mut back[..q], two, -two);
        if let Some(m) = mid {
            mhi[0] = m;
        }
        y[0] = y0;
    } else {
        // y[0..q] ascending (−Im), y[n − 1 − q..n − 1] descending (Re).
        let (lo, hi) = y.split_at_mut(q);
        let off = hi.len() - 1 - q;
        post_pairs_rev(vs, ws, lo, &mut hi[off..off + q], -two, two);
        y[n - 1] = y0;
        if let Some(m) = mid {
            y[n / 2 - 1] = m;
        }
    }
}

/// With `c = w[k]·v[k]`: `lo[k] = a·Re c` and `back[q − 1 − k] = b·Im c`.
#[inline]
fn post_pairs<T: FftFloat>(
    v: &[Complex<T>],
    w: &[Complex<T>],
    lo: &mut [T],
    back: &mut [T],
    a: T,
    b: T,
) {
    let q = v.len();
    let (w, lo, back) = (&w[..q], &mut lo[..q], &mut back[..q]);
    for k in 0..q {
        let c = w[k] * v[k];
        lo[k] = a * c.re;
        back[q - 1 - k] = b * c.im;
    }
}

/// With `c = w[k]·v[k]`: `lo[k] = a·Im c` and `back[q − 1 − k] = b·Re c`.
#[inline]
fn post_pairs_rev<T: FftFloat>(
    v: &[Complex<T>],
    w: &[Complex<T>],
    lo: &mut [T],
    back: &mut [T],
    a: T,
    b: T,
) {
    let q = v.len();
    let (w, lo, back) = (&w[..q], &mut lo[..q], &mut back[..q]);
    for k in 0..q {
        let c = w[k] * v[k];
        lo[k] = a * c.im;
        back[q - 1 - k] = b * c.re;
    }
}

/// Types III: the spectrum `v` of the reordered result, `V[k] = conj(w[k])·(X[k] − i·X[n − k])`
/// with `X[n] = 0` (DCT-III), of the input reversed for DST-III.
#[inline]
fn makhoul_pre<T: FftFloat>(fam: Fam, n: usize, x: &[T], w: &[Complex<T>], v: &mut [Complex<T>]) {
    let h = n / 2;
    let f = |w: Complex<T>, p: T, q: T| Complex::new(w.re * p - w.im * q, -(w.re * q + w.im * p));
    let (vs, ws) = (&mut v[1..=h], &w[1..=h]);
    if fam == Fam::Cos {
        // p = x[k] for k = 1..=h, q = x[n − k].
        let (ps, qs) = (&x[1..=h], &x[n - h..n]);
        for k in 0..h {
            vs[k] = f(ws[k], ps[k], qs[h - 1 - k]);
        }
        v[0] = Complex::new(x[0], T::ZERO);
    } else {
        // p = x[n − 1 − k], q = x[k − 1].
        let (ps, qs) = (&x[n - 1 - h..n - 1], &x[..h]);
        for k in 0..h {
            vs[k] = f(ws[k], ps[h - 1 - k], qs[k]);
        }
        v[0] = Complex::new(x[n - 1], T::ZERO);
    }
}

/// Types IV of an even `n`: `z[j] = (x[2j] + i·x[n − 1 − 2j])·pre[j]` in `y` as complex values
/// (DST-IV: the input reversed).
#[inline]
fn four_even_pre<T: FftFloat>(fam: Fam, pre: &[Complex<T>], x: &[T], y: &mut [T]) {
    let n = x.len();
    let h = n / 2;
    let (pre, y, x) = (&pre[..h], &mut y[..2 * h], &x[..2 * h]);
    let cos = fam == Fam::Cos;
    for j in 0..h {
        let (a, b) = (x[2 * j], x[2 * (h - 1 - j) + 1]);
        let (a, b) = if cos { (a, b) } else { (b, a) };
        let z = Complex::new(a, b) * pre[j];
        y[2 * j] = z.re;
        y[2 * j + 1] = z.im;
    }
}

/// Types IV of an even `n` from the transform `Z` in `y`: with `W = Z[p]·post[p]`,
/// `y[2p] = Re W` and `y[n − 1 − 2p] = −Im W` (DST-IV: `+Im W`).
#[inline]
fn four_even_post<T: FftFloat>(fam: Fam, n: usize, post: &[Complex<T>], y: &mut [T]) {
    let h = n / 2;
    let s = if fam == Fam::Cos { -T::ONE } else { T::ONE };
    let z = &mut as_complex_mut(y)[..h];
    twiddle_in_place(z, &post[..h]);
    // The imaginary parts of p and h − 1 − p exchanged, with the sign.
    let (lo, hi) = z.split_at_mut(h / 2);
    let mid = hi.len() - lo.len();
    let (mhi, hi) = hi.split_at_mut(mid);
    if mid == 1 {
        mhi[0].im *= s;
    }
    let q = lo.len();
    swap_im(lo, &mut hi[..q], s);
}

/// `z[k] *= w[k]`.
#[inline]
fn twiddle_in_place<T: FftFloat>(z: &mut [Complex<T>], w: &[Complex<T>]) {
    let w = &w[..z.len()];
    for k in 0..z.len() {
        z[k] *= w[k];
    }
}

/// The imaginary parts of `lo[p]` and `hi[q − 1 − p]` exchanged, times `s`.
#[inline]
fn swap_im<T: FftFloat>(lo: &mut [Complex<T>], hi: &mut [Complex<T>], s: T) {
    let q = lo.len();
    let hi = &mut hi[..q];
    for p in 0..q {
        let t = lo[p].im;
        lo[p].im = hi[q - 1 - p].im * s;
        hi[q - 1 - p].im = t * s;
    }
}

/// Types IV of an odd `n`: the permutation with signs of FFTW's `reodft11e-r2hc-odd`.
fn four_odd_permute<T: FftFloat>(fam: Fam, x: &[T], buf: &mut [T]) {
    let n = x.len();
    let (mut i, mut m) = (0, n / 2);
    // The index m runs over n/2 + 4i modulo 4n; each quarter of it reads one stretch of x.
    let at = |m: usize| -> T {
        match fam {
            Fam::Cos => {
                if m < n {
                    x[m]
                } else if m < 2 * n {
                    -x[2 * n - m - 1]
                } else if m < 3 * n {
                    -x[m - 2 * n]
                } else {
                    x[4 * n - m - 1]
                }
            }
            Fam::Sin => {
                if m < n {
                    x[n - 1 - m]
                } else if m < 2 * n {
                    -x[m - n]
                } else if m < 3 * n {
                    -x[3 * n - 1 - m]
                } else {
                    x[m - 3 * n]
                }
            }
        }
    };
    while i < n {
        buf[i] = at(m);
        i += 1;
        m += 4;
        if m >= 4 * n {
            m -= 4 * n;
        }
    }
}

/// Types IV of an odd `n` from the spectrum `s` of the permuted values (FFTW's
/// `reodft11e-r2hc-odd`, its half-complex array read from `s`).
fn four_odd_post<T: FftFloat>(fam: Fam, n: usize, s: &[Complex<T>], y: &mut [T]) {
    let n2 = n / 2;
    let hc = |j: usize| if j <= n2 { s[j].re } else { s[n - j].im };
    let r2 = T::from_f64(core::f64::consts::SQRT_2);
    // DST-IV flips the sign of every other output: an extra `j` in each sign.
    let e = |j: usize| if fam == Fam::Sin { j } else { 0 };
    let sg = |x: T, k: usize| sign(x, k % 2 == 1);
    let mut i = 0;
    while 2 * i + 1 < n2 {
        let k = 2 * i + 1;
        let (c1, c2) = (hc(k), hc(k + 1));
        let (s2, s1) = (hc(n - (k + 1)), hc(n - k));
        let o = n - (i + 1);
        y[i] = r2 * (sg(c1, i.div_ceil(2) + e(i)) + sg(s1, i / 2 + e(i)));
        y[o] = r2 * (sg(c1, (n - i) / 2 + e(i)) - sg(s1, (n - (i + 1)) / 2 + e(i)));
        let j = n2 - (i + 1);
        y[j] = r2 * (sg(c2, (n2 - i) / 2 + e(j)) - sg(s2, (n2 - (i + 1)) / 2 + e(j)));
        y[n2 + i + 1] = r2 * (sg(c2, (n2 + i + 2) / 2 + e(j)) + sg(s2, (n2 + (i + 1)) / 2 + e(j)));
        i += 1;
    }
    if 2 * i + 1 == n2 {
        let (c, s) = (hc(n2), hc(n - n2));
        y[i] = r2 * (sg(c, i.div_ceil(2) + e(i)) + sg(s, i / 2 + e(i)));
        y[n - (i + 1)] = r2 * (sg(c, (i + 1).div_ceil(2) + e(i)) + sg(s, i.div_ceil(2) + e(i)));
    }
    y[n2] = r2 * sg(hc(0), n2.div_ceil(2) + e(n2));
}

/// The transform of each run of `n` values of `input` (or of `data` itself) into `data`, with
/// the working memory `scratch` of at least `work` values.
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn apply<T: FftFloat>(
    plan: &Plan<T>,
    work: usize,
    fam: Fam,
    kind: u8,
    input: Option<&[T]>,
    data: &mut [T],
    scratch: &mut [Complex<T>],
    what: &str,
) {
    let n = plan.len();
    if let Some(x) = input {
        check_batch(x.len(), n, what);
        assert_eq!(
            x.len(),
            data.len(),
            "{what}: input and output lengths differ"
        );
    } else {
        check_batch(data.len(), n, what);
    }
    assert!(
        scratch.len() >= work,
        "{what}: scratch of {} values, {work} needed",
        scratch.len()
    );
    if n == 0 {
        return;
    }
    let scratch = &mut scratch[..work];
    for (r, y) in data.chunks_exact_mut(n).enumerate() {
        let x = input.map(|x| &x[r * n..(r + 1) * n]);
        plan.run(fam, kind, x, y, scratch);
    }
}

fn dct_kind(kind: DctKind) -> u8 {
    match kind {
        DctKind::I => 1,
        DctKind::II => 2,
        DctKind::III => 3,
        DctKind::IV => 4,
    }
}

fn dst_kind(kind: DstKind) -> u8 {
    match kind {
        DstKind::I => 1,
        DstKind::II => 2,
        DstKind::III => 3,
        DstKind::IV => 4,
    }
}

impl<T: FftFloat> Dct<T> {
    /// The plan for the length `n` and the type `kind`; DCT-I needs `n ≥ 2` (or 0) and panics
    /// for `n = 1`.
    #[track_caller]
    pub fn new(n: usize, kind: DctKind) -> Dct<T> {
        assert!(
            !(kind == DctKind::I && n == 1),
            "Dct: DCT-I needs a length of at least 2"
        );
        let plan = Plan::new(Fam::Cos, dct_kind(kind), n);
        Dct {
            n,
            kind,
            work: plan.work_len(),
            plan: Arc::new(plan),
        }
    }

    /// The planned length.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the planned length is 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The planned type.
    #[inline]
    pub fn kind(&self) -> DctKind {
        self.kind
    }

    /// The transform of each run of `n` values of `data`, in place.
    #[track_caller]
    pub fn transform(&self, data: &mut [T]) {
        with_work(self.work, |work| self.transform_with_scratch(data, work));
    }

    /// The transform of each run of `n` values of `input` into the same place in `output`.
    #[track_caller]
    pub fn transform_to(&self, input: &[T], output: &mut [T]) {
        let k = dct_kind(self.kind);
        let (plan, work) = (&*self.plan, self.work);
        with_work(work, |s| {
            apply(plan, work, Fam::Cos, k, Some(input), output, s, "Dct")
        });
    }

    /// The number of complex values of working memory
    /// [`transform_with_scratch`](Self::transform_with_scratch) needs: what one sequence takes.
    #[inline]
    pub fn scratch_len(&self) -> usize {
        self.work
    }

    /// [`transform`](Self::transform) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values; it never allocates.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::fft::{Dct, DctKind};
    ///
    /// let plan = Dct::<f64>::new(8, DctKind::II);
    /// let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
    /// let mut x = [0.5; 24]; // three sequences
    /// let mut y = x;
    /// plan.transform_with_scratch(&mut x, &mut scratch);
    /// plan.transform(&mut y);
    /// assert_eq!(x, y);
    /// ```
    #[track_caller]
    pub fn transform_with_scratch(&self, data: &mut [T], scratch: &mut [Complex<T>]) {
        let (plan, k) = (&*self.plan, dct_kind(self.kind));
        apply(plan, self.work, Fam::Cos, k, None, data, scratch, "Dct");
    }
}

impl<T: FftFloat> Dst<T> {
    /// The plan for the length `n` and the type `kind`.
    pub fn new(n: usize, kind: DstKind) -> Dst<T> {
        let plan = Plan::new(Fam::Sin, dst_kind(kind), n);
        Dst {
            n,
            kind,
            work: plan.work_len(),
            plan: Arc::new(plan),
        }
    }

    /// The planned length.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the planned length is 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The planned type.
    #[inline]
    pub fn kind(&self) -> DstKind {
        self.kind
    }

    /// The transform of each run of `n` values of `data`, in place.
    #[track_caller]
    pub fn transform(&self, data: &mut [T]) {
        with_work(self.work, |work| self.transform_with_scratch(data, work));
    }

    /// The transform of each run of `n` values of `input` into the same place in `output`.
    #[track_caller]
    pub fn transform_to(&self, input: &[T], output: &mut [T]) {
        let k = dst_kind(self.kind);
        let (plan, work) = (&*self.plan, self.work);
        with_work(work, |s| {
            apply(plan, work, Fam::Sin, k, Some(input), output, s, "Dst")
        });
    }

    /// The number of complex values of working memory
    /// [`transform_with_scratch`](Self::transform_with_scratch) needs: what one sequence takes.
    #[inline]
    pub fn scratch_len(&self) -> usize {
        self.work
    }

    /// [`transform`](Self::transform) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values; it never allocates.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::fft::{Dst, DstKind};
    ///
    /// let plan = Dst::<f32>::new(8, DstKind::IV);
    /// let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
    /// let mut x = [0.5; 24]; // three sequences
    /// let mut y = x;
    /// plan.transform_with_scratch(&mut x, &mut scratch);
    /// plan.transform(&mut y);
    /// assert_eq!(x, y);
    /// ```
    #[track_caller]
    pub fn transform_with_scratch(&self, data: &mut [T], scratch: &mut [Complex<T>]) {
        let (plan, k) = (&*self.plan, dst_kind(self.kind));
        apply(plan, self.work, Fam::Sin, k, None, data, scratch, "Dst");
    }
}

impl<T: FftFloat> core::fmt::Debug for Dct<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Dct")
            .field("n", &self.n)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl<T: FftFloat> core::fmt::Debug for Dst<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Dst")
            .field("n", &self.n)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}
