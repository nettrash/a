//! Transforms of real sequences.
//!
//! An even length `n = 2m` takes the pairs `x[2j] + i·x[2j + 1]` through the complex transform
//! of `m` and separates the spectra of the even and odd halves in one pass over the pairs of
//! values `(k, m − k)`, vectorised. An odd length with a small factor `r`, `n = r·m`, splits
//! into the `r` sequences of the residues of the index modulo `r`: their spectra two at a time
//! from one complex transform of `m` (the last one, of an odd `r`, by the real transform of `m`
//! again), then a pass of radix `r` that only makes the first half of the spectrum. Any other
//! odd length (a prime, or one whose factors are all large) runs the complex transform of the
//! length on the reals, and the shortest odd ones are summed directly from a table.

use super::twiddle;
use super::{Fft, FftFloat, check_batch};
use crate::math::complex::Complex;
use core::any::TypeId;
use std::sync::Arc;

// The vector kernels, one set per target under the name `simd`: NEON on little-endian
// aarch64, SSE2 and AVX2 + FMA on x86_64, none elsewhere and under Miri. The portable code is
// the reference they follow and runs their remainders.
#[path = "real/portable.rs"]
pub(super) mod portable;
#[path = "real/roots.rs"]
pub(super) mod roots;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
#[path = "real/aarch64.rs"]
pub(super) mod simd;
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "real/x86_64.rs"]
pub(super) mod simd;
#[cfg(not(any(
    all(target_arch = "aarch64", target_endian = "little", not(miri)),
    all(target_arch = "x86_64", not(miri))
)))]
#[path = "real/none.rs"]
pub(super) mod simd;

pub(super) use simd::Isa;

/// The largest odd factor `r` an odd length is split by; a length whose smallest factor is
/// larger runs the complex transform of the length.
const MAX_SPLIT: usize = 11;

/// The longest odd and even lengths summed directly (the sums of the whole spectrum held in
/// registers, faster there than the transforms and their fixed costs; measured).
const ODD_DIRECT: usize = 15;
const EVEN_DIRECT: usize = 16;

/// The most values of working memory taken on the stack rather than from the heap.
const STACK_WORK: usize = 512;

/// The complex values a batch of short even lengths stages at a time: the complex transforms of
/// the runs as one batch, out of place, the pass reading them from there.
const STAGE: usize = 4096;

/// Calls `f` with `len` values of zeroed working memory: on the stack when there are few.
#[inline]
pub(super) fn with_work<T: FftFloat, R>(len: usize, f: impl FnOnce(&mut [Complex<T>]) -> R) -> R {
    if len <= STACK_WORK {
        let mut buf = [const { core::mem::MaybeUninit::<Complex<T>>::uninit() }; STACK_WORK];
        for v in &mut buf[..len] {
            v.write(Complex::ZERO);
        }
        // SAFETY: the first `len` values were initialised.
        let work = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), len) };
        f(work)
    } else {
        f(&mut vec![Complex::ZERO; len])
    }
}

/// The discrete Fourier transform of real sequences of the length `n`, planned once.
///
/// [`forward`](Self::forward) takes `n` reals and gives the `n/2 + 1` values `X[0..=n/2]` of the
/// transform, which determine the rest (`X[n − k]` is the conjugate of `X[k]`; `X[0]` is real,
/// and so is `X[n/2]` for an even `n`), as FFTW's `r2c` and NumPy's `rfft` do.
/// [`inverse`](Self::inverse) takes those `n/2 + 1` values and gives `n` reals, unscaled, so that
/// `inverse(forward(x)) = n·x`; it ignores the imaginary parts of `X[0]` and, for an even `n`, of
/// `X[n/2]`. Both take batches: several sequences one after the other, as many on either side.
/// The `_with_scratch` forms take their working memory from the caller, at least
/// [`scratch_len`](Self::scratch_len) values, and then never allocate. A plan is immutable and
/// shared by its clones.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::fft::RealFft;
///
/// let fft = RealFft::<f64>::new(4);
/// let mut spectrum = [Complex::ZERO; 3];
/// fft.forward(&[1.0, 2.0, 3.0, 4.0], &mut spectrum);
/// assert_eq!(spectrum, [Complex::new(10.0, 0.0), Complex::new(-2.0, 2.0), Complex::new(-2.0, 0.0)]);
/// let mut x = [0.0; 4];
/// fft.inverse(&spectrum, &mut x);
/// assert_eq!(x, [4.0, 8.0, 12.0, 16.0]);
/// ```
#[derive(Clone)]
pub struct RealFft<T> {
    n: usize,
    /// The values of working memory one sequence takes.
    work: usize,
    algo: Arc<Algo<T>>,
}

enum Algo<T> {
    /// `n ≤ 1`: the identity.
    Trivial,
    /// `n = 2m`: the complex transform of `m` and the pass over the pairs; `fwd[k]` is
    /// `−i·e^(−2πi·k/n)/2` and `inv[k]` is `i·e^(2πi·k/n)`, for `k ≤ (m − 1)/2`.
    Half {
        fft: Fft<T>,
        fwd: Vec<Complex<T>>,
        inv: Vec<Complex<T>>,
        isa: Isa,
    },
    /// An odd `n`: the complex transform of `n`.
    Full(Fft<T>),
    /// A short `n`: `e^(−2πi·j·k/n)` for `k ≤ n/2` at `cols[p·j + k]` (zero for the `k`
    /// beyond) and at `rows[16k + j]` (zero for `j ≥ n`), with `p = 8` up to 15 and 9 for 16,
    /// the sums of a row or a column held in registers.
    Direct {
        cols: Vec<Complex<T>>,
        rows: Vec<Complex<T>>,
    },
    /// An odd `n = r·m` by the residue sequences modulo `r`.
    Split(Split<T>),
}

/// The transform of an odd `n = r·m` for the smallest prime factor `r` of `n`: the complex
/// transform of `m` for the pairs of residue sequences, the real transform of `m` for the last
/// one, `tw[(a − 1)·h + s] = e^(−2πi·a·s/n)` for `1 ≤ a < r` and `s < h = (m + 1)/2`, and
/// `wr[j] = e^(−2πi·j/r)`.
struct Split<T> {
    n: usize,
    r: usize,
    m: usize,
    pairs: Fft<T>,
    rest: Box<RealFft<T>>,
    tw: Vec<Complex<T>>,
    wr: Vec<Complex<T>>,
}

/// The smallest prime factor of an odd `n ≥ 3`.
fn smallest_factor(n: usize) -> usize {
    let mut p = 3;
    while p * p <= n {
        if n.is_multiple_of(p) {
            return p;
        }
        p += 2;
    }
    n
}

/// `x` as complex values, two reals each.
#[inline]
pub(super) fn as_complex<T: FftFloat>(x: &[T]) -> &[Complex<T>] {
    // SAFETY: `Complex<T>` is `repr(C)` of two `T`, of the alignment of `T`.
    unsafe { core::slice::from_raw_parts(x.as_ptr().cast(), x.len() / 2) }
}

/// `x` as complex values, two reals each.
#[inline]
pub(super) fn as_complex_mut<T: FftFloat>(x: &mut [T]) -> &mut [Complex<T>] {
    // SAFETY: as in `as_complex`.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len() / 2) }
}

/// The parts of `x`, re and im one after the other.
#[inline]
pub(super) fn as_reals_mut<T: FftFloat>(x: &mut [Complex<T>]) -> &mut [T] {
    // SAFETY: as in `as_complex`.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), 2 * x.len()) }
}

/// Whether `T` is `U`.
#[inline(always)]
pub(super) fn is<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// The pass over the pairs `(k, m − k)` of [`portable::pairs`], by the vector kernel of `T` and
/// the portable code for the remainder.
///
/// # Safety
///
/// As for `portable::pairs`.
#[inline]
unsafe fn pairs<T: FftFloat>(
    isa: Isa,
    dst: *mut Complex<T>,
    src: *const Complex<T>,
    m: usize,
    tw: *const Complex<T>,
    s: T,
) {
    // SAFETY: the caller's guarantees; the casts are to the type `T` is.
    unsafe {
        let from = if is::<T, f64>() {
            simd::pairs_f64(isa, dst.cast(), src.cast(), m, tw.cast(), s.to_f64())
        } else if is::<T, f32>() {
            simd::pairs_f32(isa, dst.cast(), src.cast(), m, tw.cast(), s.to_f64() as f32)
        } else {
            1
        };
        portable::pairs(dst, src, m, tw, s, from);
    }
}

impl<T: FftFloat> RealFft<T> {
    /// The plan for the length `n`.
    pub fn new(n: usize) -> RealFft<T> {
        let direct = if n.is_multiple_of(2) {
            EVEN_DIRECT
        } else {
            ODD_DIRECT
        };
        let algo = if n <= 1 {
            Algo::Trivial
        } else if n <= direct {
            let h = n / 2 + 1;
            let hp = if n <= 15 { 8 } else { 9 };
            let jp = 16;
            let w = |j: usize, k: usize| {
                if j < n && k < h {
                    twiddle::root_t(j * k % n, n)
                } else {
                    Complex::ZERO
                }
            };
            Algo::Direct {
                cols: (0..hp * n).map(|i| w(i / hp, i % hp)).collect(),
                rows: (0..jp * h).map(|i| w(i % jp, i / jp)).collect(),
            }
        } else if n.is_multiple_of(2) {
            let m = n / 2;
            let h = m.div_ceil(2);
            let mut fwd = roots::table::<T>(n, h);
            let inv = fwd.iter().map(|w| Complex::new(w.im, w.re)).collect();
            for w in &mut fwd {
                *w = Complex::new(w.im * T::HALF, -w.re * T::HALF);
            }
            Algo::Half {
                fft: Fft::new(m),
                fwd,
                inv,
                isa: Isa::detect(),
            }
        } else {
            let r = smallest_factor(n);
            if r == n || r > MAX_SPLIT {
                Algo::Full(Fft::new(n))
            } else {
                let m = n / r;
                let h = m.div_ceil(2);
                let tw: Vec<Complex<T>> = (1..r).flat_map(|a| roots::affine(n, 0, a, h)).collect();
                Algo::Split(Split {
                    n,
                    r,
                    m,
                    pairs: Fft::new(m),
                    rest: Box::new(RealFft::new(m)),
                    tw,
                    wr: twiddle::table(r, r),
                })
            }
        };
        let work = match &algo {
            Algo::Trivial | Algo::Direct { .. } => 0,
            Algo::Half { fft, .. } => fft.scratch_len(),
            Algo::Full(fft) => n + fft.scratch_len(),
            Algo::Split(sp) => sp.work_len(),
        };
        RealFft {
            n,
            work,
            algo: Arc::new(algo),
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

    /// The length of the spectrum of one sequence, `n/2 + 1` (0 for `n = 0`).
    #[inline]
    pub fn spectrum_len(&self) -> usize {
        if self.n == 0 { 0 } else { self.n / 2 + 1 }
    }

    /// The `n/2 + 1` values of the transform of each run of `n` reals of `input`, one after the
    /// other in `output`.
    #[track_caller]
    pub fn forward(&self, input: &[T], output: &mut [Complex<T>]) {
        let (n, h) = (self.n, self.spectrum_len());
        check_batch(input.len(), n, "RealFft");
        let runs = input.len().checked_div(n).unwrap_or(0);
        assert_eq!(
            output.len(),
            runs * h,
            "RealFft: the output needs n/2 + 1 values a sequence"
        );
        if n == 0 || runs == 0 {
            return;
        }
        if let (Algo::Half { fft, .. }, false) = (&*self.algo, runs == 1 && self.work <= STACK_WORK)
        {
            // One long sequence: its complex transform straight into the output (a short one
            // takes its working memory on the stack below); several: theirs as batches, out of
            // place, into a buffer the pass reads.
            let m = n / 2;
            if runs == 1 {
                fft.forward_to(as_complex(input), &mut output[..m]);
                let p = output.as_mut_ptr();
                // SAFETY: in place, `m + 1` values.
                unsafe { self.separate(p, p) };
                return;
            }
            let g = (STAGE / m).clamp(1, runs);
            let mut tmp = vec![Complex::ZERO; g * m];
            for (xs, ys) in input.chunks(g * n).zip(output.chunks_mut(g * h)) {
                let tmp = &mut tmp[..xs.len() / 2];
                fft.forward_to(as_complex(xs), tmp);
                for (z, y) in tmp.chunks_exact(m).zip(ys.chunks_exact_mut(h)) {
                    // SAFETY: `z` holds `m` values and `y` `m + 1`, disjoint.
                    unsafe { self.separate(y.as_mut_ptr(), z.as_ptr()) };
                }
            }
            return;
        }
        with_work(self.work, |work| {
            self.forward_with_scratch(input, output, work)
        });
    }

    /// The `n` reals of the unscaled inverse transform of each run of `n/2 + 1` values of
    /// `input`, one after the other in `output`.
    #[track_caller]
    pub fn inverse(&self, input: &[Complex<T>], output: &mut [T]) {
        with_work(self.work, |work| {
            self.inverse_with_scratch(input, output, work)
        });
    }

    /// The number of values of working memory the `_with_scratch` methods need: what one
    /// sequence takes (0 for some lengths).
    #[inline]
    pub fn scratch_len(&self) -> usize {
        self.work
    }

    /// [`forward`](Self::forward) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values; it never allocates.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::fft::RealFft;
    ///
    /// let fft = RealFft::<f32>::new(6);
    /// let mut scratch = vec![Complex::ZERO; fft.scratch_len()];
    /// let mut spectrum = [Complex::ZERO; 8]; // two sequences of 6, 4 values each
    /// let x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// fft.forward_with_scratch(&x, &mut spectrum, &mut scratch);
    /// assert_eq!(spectrum[0], Complex::new(6.0, 0.0));
    /// assert!(spectrum[1..4].iter().all(|v| v.abs() < 1e-6));
    /// assert_eq!(spectrum[4], Complex::new(1.0, 0.0));
    /// ```
    #[track_caller]
    pub fn forward_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        let (n, h) = (self.n, self.spectrum_len());
        check_batch(input.len(), n, "RealFft");
        let runs = input.len().checked_div(n).unwrap_or(0);
        assert_eq!(
            output.len(),
            runs * h,
            "RealFft: the output needs n/2 + 1 values a sequence"
        );
        self.check_scratch(scratch);
        if n == 0 || runs == 0 {
            return;
        }
        let work = &mut scratch[..self.work];
        for (x, y) in input.chunks_exact(n).zip(output.chunks_exact_mut(h)) {
            self.forward_run(x, y, work);
        }
    }

    /// [`inverse`](Self::inverse) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values; it never allocates. More scratch can let the
    /// complex transforms of a batch of short sequences run together.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::fft::RealFft;
    ///
    /// let fft = RealFft::<f64>::new(4);
    /// let mut scratch = vec![Complex::ZERO; fft.scratch_len()];
    /// let spectrum = [Complex::new(4.0, 0.0), Complex::ZERO, Complex::ZERO];
    /// let mut x = [0.0; 4];
    /// fft.inverse_with_scratch(&spectrum, &mut x, &mut scratch);
    /// assert_eq!(x, [4.0; 4]);
    /// ```
    #[track_caller]
    pub fn inverse_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        scratch: &mut [Complex<T>],
    ) {
        let (n, h) = (self.n, self.spectrum_len());
        check_batch(output.len(), n, "RealFft");
        let runs = output.len().checked_div(n).unwrap_or(0);
        assert_eq!(
            input.len(),
            runs * h,
            "RealFft: the input needs n/2 + 1 values a sequence"
        );
        self.check_scratch(scratch);
        if n == 0 || runs == 0 {
            return;
        }
        if let Algo::Half { fft, .. } = &*self.algo {
            // The pass of every sequence into its place, then their complex transforms as one
            // batch, in place.
            for (x, y) in input.chunks_exact(h).zip(output.chunks_exact_mut(n)) {
                self.combine(x, as_complex_mut(y));
            }
            fft.inverse_with_scratch(as_complex_mut(output), scratch);
            return;
        }
        let work = &mut scratch[..self.work];
        for (x, y) in input.chunks_exact(h).zip(output.chunks_exact_mut(n)) {
            self.inverse_run(x, y, work);
        }
    }

    #[track_caller]
    fn check_scratch(&self, scratch: &[Complex<T>]) {
        assert!(
            scratch.len() >= self.work,
            "RealFft: scratch of {} values, {} needed",
            scratch.len(),
            self.work
        );
    }

    /// Whether the plan takes the complex transform of `n/2`, so that
    /// [`forward_packed`](Self::forward_packed) and [`inverse_packed`](Self::inverse_packed)
    /// apply.
    #[inline]
    pub(super) fn is_half(&self) -> bool {
        matches!(*self.algo, Algo::Half { .. })
    }

    /// The values of working memory [`forward_run`](Self::forward_run) and
    /// [`inverse_run`](Self::inverse_run) take.
    #[inline]
    pub(super) fn work_len(&self) -> usize {
        self.work
    }

    /// The transform of the `n` reals `x` into the `n/2 + 1` values `y`.
    pub(super) fn forward_run(&self, x: &[T], y: &mut [Complex<T>], work: &mut [Complex<T>]) {
        match &*self.algo {
            Algo::Trivial => y[0] = Complex::new(x[0], T::ZERO),
            Algo::Half { .. } => {
                let m = self.n / 2;
                y[..m].copy_from_slice(as_complex(x));
                self.forward_packed(y, work);
            }
            Algo::Full(fft) => {
                let (z, scratch) = work.split_at_mut(self.n);
                for (z, &x) in z.iter_mut().zip(x) {
                    *z = Complex::new(x, T::ZERO);
                }
                fft.forward_with_scratch(z, scratch);
                y.copy_from_slice(&z[..y.len()]);
                // X[0] is the sum of the reals: no rounding error of the transform in its
                // imaginary part.
                y[0].im = T::ZERO;
            }
            Algo::Direct { cols, .. } => {
                if self.n <= 15 {
                    direct_forward::<T, 8>(x, cols, y);
                } else {
                    direct_forward::<T, 9>(x, cols, y);
                }
            }
            Algo::Split(sp) => sp.forward(x, y, work),
        }
    }

    /// For an even `n = 2m`: the transform of the reals packed in `y[..m]` as complex values
    /// (`y[j] = x[2j] + i·x[2j + 1]`) into all of `y`, `m + 1` values.
    pub(super) fn forward_packed(&self, y: &mut [Complex<T>], work: &mut [Complex<T>]) {
        let Algo::Half { fft, .. } = &*self.algo else {
            unreachable!("an even length")
        };
        fft.forward_with_scratch(&mut y[..self.n / 2], work);
        let p = y.as_mut_ptr();
        // SAFETY: in place, `m + 1` values.
        unsafe { self.separate(p, p) };
    }

    /// For an even `n = 2m`: the spectrum `dst[..=m]` of the reals from the complex transform
    /// `src[..m]` of their pairs.
    ///
    /// # Safety
    ///
    /// `src` is valid for `m` values and `dst` for `m + 1`; they are the same or disjoint.
    unsafe fn separate(&self, dst: *mut Complex<T>, src: *const Complex<T>) {
        let Algo::Half { fwd, isa, .. } = &*self.algo else {
            unreachable!("an even length")
        };
        let m = self.n / 2;
        // SAFETY: the caller's guarantees; X[0], X[m] and X[m/2] are read before the pass and
        // written after it, which touches neither.
        unsafe {
            let z0 = *src;
            let mid = (*src.add(m / 2)).conj();
            pairs(*isa, dst, src, m, fwd.as_ptr(), T::HALF);
            *dst = Complex::new(z0.re + z0.im, T::ZERO);
            *dst.add(m) = Complex::new(z0.re - z0.im, T::ZERO);
            if m.is_multiple_of(2) {
                *dst.add(m / 2) = mid;
            }
        }
    }

    /// The unscaled inverse transform of the `n/2 + 1` values `x` into the `n` reals `y`.
    pub(super) fn inverse_run(&self, x: &[Complex<T>], y: &mut [T], work: &mut [Complex<T>]) {
        match &*self.algo {
            Algo::Trivial => y[0] = x[0].re,
            Algo::Half { .. } => self.inverse_packed(x, as_complex_mut(y), work),
            Algo::Full(fft) => {
                let n = self.n;
                let (z, scratch) = work.split_at_mut(n);
                z[0] = Complex::new(x[0].re, T::ZERO);
                for k in 1..x.len() {
                    z[k] = x[k];
                    z[n - k] = x[k].conj();
                }
                fft.inverse_with_scratch(z, scratch);
                for (y, z) in y.iter_mut().zip(z.iter()) {
                    *y = z.re;
                }
            }
            Algo::Direct { rows, .. } => {
                direct_inverse::<T, 16>(x, rows, y);
            }
            Algo::Split(sp) => sp.inverse(x, y, work),
        }
    }

    /// For an even `n = 2m`: the unscaled inverse transform of the `m + 1` values `x` into `z`,
    /// `m` values holding the reals packed in pairs.
    pub(super) fn inverse_packed(
        &self,
        x: &[Complex<T>],
        z: &mut [Complex<T>],
        work: &mut [Complex<T>],
    ) {
        let Algo::Half { fft, .. } = &*self.algo else {
            unreachable!("an even length")
        };
        self.combine(x, z);
        fft.inverse_with_scratch(&mut z[..self.n / 2], work);
    }

    /// For an even `n = 2m`: the `m` values `z` whose inverse complex transform is the inverse
    /// transform of the spectrum `x`, `m + 1` values, in pairs of reals.
    fn combine(&self, x: &[Complex<T>], z: &mut [Complex<T>]) {
        let Algo::Half { inv, isa, .. } = &*self.algo else {
            unreachable!("an even length")
        };
        let m = self.n / 2;
        let (x, z) = (&x[..=m], &mut z[..m]);
        let (a, b) = (x[0].re, x[m].re);
        z[0] = Complex::new(a + b, a - b);
        if m.is_multiple_of(2) {
            z[m / 2] = x[m / 2].conj() * T::TWO;
        }
        // SAFETY: `x` holds `m + 1` values, `z` `m` and `inv` `(m + 1)/2`; disjoint.
        unsafe { pairs(*isa, z.as_mut_ptr(), x.as_ptr(), m, inv.as_ptr(), T::ONE) };
    }
}

/// The spectrum `y` of the short `x` as the sum of the columns of the table, `P` of them a
/// row, times the reals.
#[inline]
fn direct_forward<T: FftFloat, const P: usize>(x: &[T], cols: &[Complex<T>], y: &mut [Complex<T>]) {
    let mut acc = [Complex::ZERO; P];
    for (&x, col) in x.iter().zip(cols.as_chunks::<P>().0) {
        for (a, &w) in acc.iter_mut().zip(col) {
            *a += w * x;
        }
    }
    y.copy_from_slice(&acc[..y.len()]);
    y[0].im = T::ZERO;
}

/// The `n = y.len()` reals of the inverse of the spectrum `x`:
/// `y[j] = X[0] + Σ_k c_k·Re(X[k]·e^(2πi·j·k/n))` with `c_k = 2`, but 1 for `k = n/2`; the
/// imaginary parts of `X[0]` and `X[n/2]` are not read or meet a zero of the table.
#[inline]
fn direct_inverse<T: FftFloat, const Q: usize>(x: &[Complex<T>], rows: &[Complex<T>], y: &mut [T]) {
    let n = y.len();
    let mut acc = [x[0].re; Q];
    for (k, (&xk, row)) in x.iter().zip(rows.as_chunks::<Q>().0).enumerate().skip(1) {
        let c = if 2 * k == n { T::ONE } else { T::TWO };
        let (a, b) = (xk.re * c, xk.im * c);
        for (v, w) in acc.iter_mut().zip(row) {
            *v += a * w.re + b * w.im;
        }
    }
    y.copy_from_slice(&acc[..n]);
}

/// `Σ_j t[j]·wr[(j·q) mod r]` for the `r` values `t`.
#[inline]
fn dft_row<T: FftFloat>(t: &[Complex<T>], wr: &[Complex<T>], q: usize) -> Complex<T> {
    let r = t.len();
    let mut acc = t[0];
    let mut e = 0;
    for &v in &t[1..] {
        e += q;
        if e >= r {
            e -= r;
        }
        acc += v * wr[e];
    }
    acc
}

/// `i·x`.
#[inline(always)]
fn times_i<T: FftFloat>(x: Complex<T>) -> Complex<T> {
    Complex::new(-x.im, x.re)
}

/// The `r` values `out[q] = Σ_a t[a]·e^(∓2πi·a·q/r)` (`+` when `inv`) for the `r = t.len()`
/// values `t`: the butterflies of radix 3 and 5 written out, the sums for the others.
#[inline(always)]
fn butterfly<T: FftFloat>(t: &[Complex<T>], wr: &[Complex<T>], inv: bool, out: &mut [Complex<T>]) {
    let sign = |v: T| if inv { -v } else { v };
    match t.len() {
        3 => {
            // With S = t1 + t2, D = t1 − t2 and h = −sin(2π/3):
            // X0 = t0 + S, X1,2 = t0 − S/2 ± i·h·D.
            let (s, d) = (t[1] + t[2], t[1] - t[2]);
            out[0] = t[0] + s;
            let mid = t[0] - s * T::HALF;
            let id = times_i(d) * sign(wr[1].im);
            out[1] = mid + id;
            out[2] = mid - id;
        }
        5 => {
            let (c1, c2) = (wr[1].re, wr[2].re);
            let (s1, s2) = (sign(-wr[1].im), sign(-wr[2].im));
            let (a1, b1) = (t[1] + t[4], t[1] - t[4]);
            let (a2, b2) = (t[2] + t[3], t[2] - t[3]);
            out[0] = t[0] + a1 + a2;
            let m1 = t[0] + a1 * c1 + a2 * c2;
            let m2 = t[0] + a1 * c2 + a2 * c1;
            let n1 = times_i(b1 * s1 + b2 * s2);
            let n2 = times_i(b1 * s2 - b2 * s1);
            out[1] = m1 - n1;
            out[4] = m1 + n1;
            out[2] = m2 - n2;
            out[3] = m2 + n2;
        }
        r => {
            for (q, o) in out.iter_mut().enumerate() {
                *o = dft_row(t, wr, if inv { (r - q) % r } else { q });
            }
        }
    }
}

impl<T: FftFloat> Split<T> {
    /// The values of working memory [`forward`](Self::forward) and [`inverse`](Self::inverse)
    /// take: the pairs, the spectrum and the reals of the last sequence, and the scratch.
    fn work_len(&self) -> usize {
        let h = self.m.div_ceil(2);
        (self.r - 1) / 2 * self.m + 2 * h + self.pairs.scratch_len().max(self.rest.work)
    }

    /// The transform of the `n` reals `x` into the `(n + 1)/2` values `y`.
    fn forward(&self, x: &[T], y: &mut [Complex<T>], work: &mut [Complex<T>]) {
        let (r, m) = (self.r, self.m);
        let (h, np) = (m.div_ceil(2), (r - 1) / 2);
        let (zb, work) = work.split_at_mut(np * m);
        let (yl, work) = work.split_at_mut(h);
        let (xl, work) = work.split_at_mut(h);
        // The pairs of residue sequences 2p and 2p + 1 as complex values, and their transforms.
        for (p, z) in zb.chunks_exact_mut(m).enumerate() {
            for (j, z) in z.iter_mut().enumerate() {
                *z = Complex::new(x[2 * p + r * j], x[2 * p + 1 + r * j]);
            }
        }
        self.pairs.forward_with_scratch(zb, work);
        // The last one by the real transform of m.
        let xl = &mut as_reals_mut(xl)[..m];
        for (j, v) in xl.iter_mut().enumerate() {
            *v = x[r - 1 + r * j];
        }
        self.rest.forward_run(xl, yl, work);
        // For each s < h: the spectra Y_a[s] of the residue sequences (a pair's apart as
        // Y_2p = (Z[s] + conj Z[m − s])/2, Y_2p+1 = (Z[s] − conj Z[m − s])/2i), twiddled, then
        // X[s + q·m] = Σ_a e^(−2πi·a·s/n)·Y_a[s]·e^(−2πi·a·q/r): the first half of the spectrum
        // directly for q ≤ (r − 1)/2, by conjugates at n − s − q·m for the others.
        let mut t = [Complex::ZERO; MAX_SPLIT];
        let mut o = [Complex::ZERO; MAX_SPLIT];
        let (t, o) = (&mut t[..r], &mut o[..r]);
        for s in 0..h {
            let sm = if s == 0 { 0 } else { m - s };
            for (p, z) in zb.chunks_exact(m).enumerate() {
                let (a, b) = (z[s], z[sm].conj());
                let (e, d) = ((a + b) * T::HALF, (a - b) * T::HALF);
                t[2 * p] = e;
                t[2 * p + 1] = Complex::new(d.im, -d.re);
            }
            t[r - 1] = yl[s];
            for (a, t) in t.iter_mut().enumerate().skip(1) {
                *t *= self.tw[(a - 1) * h + s];
            }
            butterfly(t, &self.wr, false, o);
            for q in 0..=np {
                y[s + q * m] = o[q];
            }
            if s != 0 {
                for q in np + 1..r {
                    y[(r - 1 - q) * m + m - s] = o[q].conj();
                }
            }
        }
        y[0].im = T::ZERO;
    }

    /// The unscaled inverse transform of the `(n + 1)/2` values `x` into the `n` reals `y`.
    fn inverse(&self, x: &[Complex<T>], y: &mut [T], work: &mut [Complex<T>]) {
        let (n, r, m) = (self.n, self.r, self.m);
        let (h, np, half) = (m.div_ceil(2), (r - 1) / 2, (n - 1) / 2);
        let (zb, work) = work.split_at_mut(np * m);
        let (ul, work) = work.split_at_mut(h);
        let (xl, work) = work.split_at_mut(h);
        // For each s < h: U_a[s] = e^(2πi·a·s/n)·Σ_q X[s + q·m]·e^(2πi·a·q/r), the inverse
        // transform of U_a being the residue sequence a; a pair's as Z = U_2p + i·U_2p+1 over
        // all of m.
        let mut t = [Complex::ZERO; MAX_SPLIT];
        let mut o = [Complex::ZERO; MAX_SPLIT];
        let (t, o) = (&mut t[..r], &mut o[..r]);
        for s in 0..h {
            for (q, t) in t.iter_mut().enumerate() {
                let k = s + q * m;
                *t = if k == 0 {
                    Complex::new(x[0].re, T::ZERO)
                } else if k <= half {
                    x[k]
                } else {
                    x[n - k].conj()
                };
            }
            butterfly(t, &self.wr, true, o);
            for (a, o) in o.iter_mut().enumerate().skip(1) {
                *o *= self.tw[(a - 1) * h + s].conj();
            }
            if s == 0 {
                for v in o.iter_mut() {
                    v.im = T::ZERO;
                }
            }
            for (p, z) in zb.chunks_exact_mut(m).enumerate() {
                let (a, b) = (o[2 * p], o[2 * p + 1]);
                z[s] = Complex::new(a.re - b.im, a.im + b.re);
                if s != 0 {
                    z[m - s] = Complex::new(a.re + b.im, b.re - a.im);
                }
            }
            ul[s] = o[r - 1];
        }
        self.pairs.inverse_with_scratch(zb, work);
        for (p, z) in zb.chunks_exact(m).enumerate() {
            for (j, z) in z.iter().enumerate() {
                y[2 * p + r * j] = z.re;
                y[2 * p + 1 + r * j] = z.im;
            }
        }
        // The last one by the real transform of m.
        let xl = &mut as_reals_mut(xl)[..m];
        self.rest.inverse_run(ul, xl, work);
        for (j, &v) in xl.iter().enumerate() {
            y[r - 1 + r * j] = v;
        }
    }
}

impl<T> core::fmt::Debug for RealFft<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RealFft")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}
