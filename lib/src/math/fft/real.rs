//! Transforms of real sequences.
//!
//! An even length `n = 2m` takes the pairs `x[2j] + i·x[2j + 1]` through the complex transform
//! of `m` and separates the spectra of the even and odd halves in one pass over the pairs of
//! values `(k, m − k)`, vectorised: for a half of one butterfly the whole transform in one
//! kernel, and for `f64` the pass fused with the complex transform's last pass. An odd length
//! with a factor `r` among `SPLIT_RADICES` (19, 17, 15, 13, 11, 9, 7, 5 and 3, the first that
//! divides it taken, so the largest, composites included), `n = r·m`, splits into the `r`
//! sequences of the residues of the index modulo `r`: their spectra two at a time from one
//! complex transform of `m` (the last one, of an odd `r`, by the real transform of `m` again),
//! then a pass of radix `r` that only makes the first half of the spectrum. The odd lengths 3 to
//! 15 take the complex butterfly on the reals; any other odd length (a prime above 15, or one
//! with no factor up to 19) runs the complex transform of the length on the reals; and `n = 2`
//! is summed directly from a table.

use super::complex::{
    HKern, Isa as ComplexIsa, RKern, SFwd, SInv, real_kernels, real_odd_kernels, split_kernels,
};
use super::twiddle;
use super::{Fft, FftFloat, check_batch, with_raw, with_work};
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

/// The radices an odd length is split by, the first that divides it (and is below it) taken: the
/// butterflies of the complex passes, composite ones too (fewer levels of the split); a length
/// none divides runs the complex transform of the length. Measured on an Apple M4 Max (odd
/// lengths 45 to 99,225, smooth and with large prime factors, both types, forward and inverse,
/// against 19, 17, 15, 13, 11, 9, 7, 5, 3): 15 and 9 before 17, 13 and 11 take 0.86 to 0.98 of
/// the time where both divide (99, 117, 153, 255, 297, 351, 459, 693, 1071, 1683, 2295), within
/// the noise elsewhere; 19 stays first, as after 15 and 9 the forward `f64` transforms of 171, 285 and
/// 4845 took 1.05 to 1.06 times. 9 before 15 would gain up to 18 % at 45 to 1215 but lose up to
/// 4 % at 225, 3375, 4725 and 10,395; the primes alone, or the smallest first, up to 1.3 times
/// at 81, 135 and 225 (and 625 and 3125 lost 10 % with 25, 5·5 in one butterfly, under load).
const SPLIT_RADICES: [usize; 9] = [19, 15, 9, 17, 13, 11, 7, 5, 3];

/// The bounds of the short lengths. The odd lengths up to `ODD_DIRECT` take the complex
/// butterfly on the reals (`Algo::Odd`), and the even ones up to `EVEN_DIRECT` the fused kernels
/// where their half is one butterfly (`Algo::Half` with `fused`); the sums of the whole spectrum
/// from a table (`Algo::Direct`) are left for the lengths in these bounds that neither takes,
/// which is `n = 2` alone.
const ODD_DIRECT: usize = 15;
const EVEN_DIRECT: usize = 16;

/// The complex values a batch of short even lengths stages at a time: the complex transforms of
/// the runs as one batch, out of place, the pass reading them from there.
const STAGE: usize = 4096;

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
        /// For an `m` of one butterfly: the complex transform and the pass in one kernel.
        fused: Option<Fused<T>>,
        /// Else, for the plans that have it: the complex transform's last pass fused with the
        /// pass over the pairs (forward).
        last: Option<HKern<T>>,
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
    /// A short odd `n` of one complex butterfly: that butterfly on the reals.
    Odd { kern: [RKern<T>; 2], g: Vec<T> },
}

/// The fused kernels of an even `n = 2m` for an `m` of one complex butterfly: the butterfly
/// and the pass over the pairs in registers, `fwd` and `inv` the pass's factors in compact form.
struct Fused<T> {
    kern: [RKern<T>; 2],
    g: Vec<T>,
    fwd: Vec<T>,
    inv: Vec<T>,
}

impl<T: FftFloat> Fused<T> {
    fn new(m: usize, fwd: &[Complex<T>], inv: &[Complex<T>]) -> Option<Fused<T>> {
        let (kern, g) = real_kernels::<T>(m, ComplexIsa::detect())?;
        let compact = |w: &[Complex<T>]| -> Vec<T> {
            w.iter()
                .flat_map(|w| [w.re, w.re, T::ZERO - w.im, w.im])
                .collect()
        };
        Some(Fused {
            kern,
            g,
            fwd: compact(fwd),
            inv: compact(inv),
        })
    }

    /// The spectrum `y` (`m + 1` values) of the `2m` reals packed at `x` (`m` values); `x` and
    /// `y` may start at the same place.
    ///
    /// # Safety
    /// `x` must be valid for reading `m` values and `y` for writing `m + 1`.
    #[inline(always)]
    unsafe fn forward(&self, x: *const Complex<T>, y: *mut Complex<T>) {
        // SAFETY: the caller's contract; the kernel loads every value before it stores one.
        unsafe { (self.kern[0])(&self.g, self.fwd.as_ptr(), x, y) }
    }

    /// The `2m` reals packed at `y` (`m` values) of the spectrum at `x` (`m + 1` values),
    /// unscaled; `x` and `y` may start at the same place.
    ///
    /// # Safety
    /// `x` must be valid for reading `m + 1` values and `y` for writing `m`.
    #[inline(always)]
    unsafe fn inverse(&self, x: *const Complex<T>, y: *mut Complex<T>) {
        // SAFETY: as for `forward`.
        unsafe { (self.kern[1])(&self.g, self.inv.as_ptr(), x, y) }
    }
}

/// The transform of an odd `n = r·m` for a radix `r` of [`SPLIT_RADICES`]: the complex
/// transform of `m` for the pairs of residue sequences, the real transform of `m` for the last
/// one, and the combine passes by the butterfly of `r` (its table `g`), with `tw` the twiddles
/// `e^(−2πi·a·s/n)` for `1 ≤ a < r` (by `a`, then `s < h = (m + 1)/2`) in compact form.
struct Split<T> {
    r: usize,
    m: usize,
    pairs: Fft<T>,
    rest: Box<RealFft<T>>,
    fwd: SFwd<T>,
    inv: SInv<T>,
    g: Vec<T>,
    tw: Vec<T>,
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
        // The even lengths up to 16 take the fused kernels rather than the direct sums:
        // measured on an Apple M4 Max, the sums take 1.45 to 3.6 times the kernels' time forward
        // and 3.6 to 6.0 times inverse (one run, both types), 1.8 to 11.8 times on batches.
        let fused_m = n.is_multiple_of(2)
            && n >= 4
            && real_kernels::<T>(n / 2, ComplexIsa::detect()).is_some();
        // The odd lengths up to 15 by the complex butterfly rather than the direct sums: the
        // sums take 1.12 to 3.5 times its time (Apple M4 Max, one run and batches, both types).
        let odd = (n % 2 == 1 && n <= ODD_DIRECT)
            .then(|| real_odd_kernels::<T>(n, ComplexIsa::detect()))
            .flatten();
        let algo = if n <= 1 {
            Algo::Trivial
        } else if let Some((kern, g)) = odd {
            Algo::Odd { kern, g }
        } else if n <= direct && !fused_m {
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
            let inv: Vec<Complex<T>> = fwd.iter().map(|w| Complex::new(w.im, w.re)).collect();
            for w in &mut fwd {
                *w = Complex::new(w.im * T::HALF, -w.re * T::HALF);
            }
            let fused = Fused::new(m, &fwd, &inv);
            let fft = Fft::new(m);
            // The fused last pass on blocks of one value: faster for `f64`, slower for `f32`,
            // whose separate pass runs four pairs a vector (it would need the pass on blocks of
            // two values, lanes q, q + 1 against s − q, s − q − 1). Measured on an Apple M4 Max
            // (the forward transform of 96 to 2²⁰ values; the inverse's time is unchanged): `f64`
            // takes 0.79 to 1.0 of the separate pass's time but at 2¹⁶ and 2¹⁹ (1.10 and 1.34, a
            // last pass of radix 8 over 2¹⁵ and 2¹⁸ values), `f32` 1.01 to 1.51 times.
            let last = if fused.is_none() && T::MANTISSA_DIGITS >= 53 {
                fft.half_last()
            } else {
                None
            };
            Algo::Half {
                fft,
                fwd,
                inv,
                isa: Isa::detect(),
                fused,
                last,
            }
        } else {
            let isa = ComplexIsa::detect();
            let split = SPLIT_RADICES
                .iter()
                .find(|&&r| r < n && n.is_multiple_of(r))
                .and_then(|&r| split_kernels::<T>(r, isa).map(|k| (r, k)));
            match split {
                None => Algo::Full(Fft::new(n)),
                Some((r, (fwd, inv, g))) => {
                    let m = n / r;
                    let h = m.div_ceil(2);
                    let tw = (1..r)
                        .flat_map(|a| roots::affine::<T>(n, 0, a, h))
                        .flat_map(|w| [w.re, w.re, -w.im, w.im])
                        .collect();
                    Algo::Split(Split {
                        r,
                        m,
                        pairs: Fft::new(m),
                        rest: Box::new(RealFft::new(m)),
                        fwd,
                        inv,
                        g,
                        tw,
                    })
                }
            }
        };
        let work = match &algo {
            Algo::Trivial | Algo::Direct { .. } | Algo::Odd { .. } => 0,
            // The fused kernels take no working memory.
            Algo::Half { fused: Some(_), .. } => 0,
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
        if let Algo::Half {
            fused: Some(fu), ..
        } = &*self.algo
        {
            let x = as_complex(input).as_ptr();
            for r in 0..runs {
                // SAFETY: run `r` of the input, `m` packed values, and of the output, `m + 1`.
                unsafe { fu.forward(x.add(r * (n / 2)), output.as_mut_ptr().add(r * h)) };
            }
            return;
        }
        if let Algo::Half { fft, .. } = &*self.algo {
            // One sequence: its complex transform straight into the output, the working memory
            // not initialised; several: theirs as batches, out of place, into a buffer the pass
            // reads.
            let m = n / 2;
            if runs == 1 {
                let (x, p) = (as_complex(input).as_ptr(), output.as_mut_ptr());
                if let Algo::Half {
                    last: Some(k), fwd, ..
                } = &*self.algo
                {
                    let tw = fwd.as_ptr().cast::<T>();
                    with_raw(fft.scratch_len(), |scr| {
                        // SAFETY: `x` holds the `m` packed values, `output` room for `m + 1`,
                        // apart; the working memory of the transform, a third buffer; `tw` the
                        // pass's factors.
                        unsafe { fft.run_half_last(x, p, scr, *k, tw) }
                    });
                    return;
                }
                with_raw(fft.scratch_len(), |scr| {
                    // SAFETY: `x` holds the `m` packed values, `output` room for `m + 1`, apart;
                    // the working memory of the transform, a third buffer; then the pass in
                    // place on the `m + 1` values.
                    unsafe {
                        fft.run_to_raw(x, p, scr, false);
                        self.separate(p, p);
                    }
                });
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
        if let Algo::Half { fused: Some(_), .. } = &*self.algo {
            // No working memory.
            return self.inverse_with_scratch(input, output, &mut []);
        }
        if let Algo::Half { fft, .. } = &*self.algo
            && input.len() == self.n / 2 + 1
            && output.len() == self.n
        {
            // One sequence: the pass into the output, then the complex transform in place with
            // the working memory not initialised.
            let z = as_complex_mut(output);
            self.combine(input, z);
            let (p, len) = (z.as_mut_ptr(), z.len());
            with_raw(fft.scratch_len(), |scr| {
                // SAFETY: `z` holds the `m` values, transformed in place; the working memory of
                // the transform, apart, `scratch_len()` values.
                unsafe { fft.run_raw(p, len, scr, fft.scratch_len(), true) }
            });
            return;
        }
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
        if let Algo::Half {
            fused: Some(fu), ..
        } = &*self.algo
        {
            let y = as_complex_mut(output).as_mut_ptr();
            for r in 0..runs {
                // SAFETY: run `r` of the input, `m + 1` values, and of the output, `m` packed.
                unsafe { fu.inverse(input.as_ptr().add(r * h), y.add(r * (n / 2))) };
            }
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
            Algo::Half {
                fused: Some(fu), ..
            } => {
                // SAFETY: `x` holds the `m` packed values, `y` room for `m + 1`.
                unsafe { fu.forward(as_complex(x).as_ptr(), y.as_mut_ptr()) }
            }
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
            Algo::Odd { kern, g } => {
                // SAFETY: `x` holds the n reals and `y` room for `(n + 1)/2` values.
                unsafe { (kern[0])(g, core::ptr::null(), x.as_ptr().cast(), y.as_mut_ptr()) }
            }
        }
    }

    /// For an even `n = 2m`: the transform of the reals packed in `y[..m]` as complex values
    /// (`y[j] = x[2j] + i·x[2j + 1]`) into all of `y`, `m + 1` values.
    pub(super) fn forward_packed(&self, y: &mut [Complex<T>], work: &mut [Complex<T>]) {
        let Algo::Half { fft, fused, .. } = &*self.algo else {
            unreachable!("an even length")
        };
        if let Some(fu) = fused {
            let p = y[..=self.n / 2].as_mut_ptr();
            // SAFETY: `y` holds the `m + 1` values, the packed reals first, in place.
            unsafe { fu.forward(p, p) };
            return;
        }
        if let Algo::Half {
            last: Some(k), fwd, ..
        } = &*self.algo
            && work.len() >= fft.scratch_len()
        {
            let p = y[..=self.n / 2].as_mut_ptr();
            // SAFETY: in place on the `m` packed values, `m + 1` written; `work` the transform's
            // working memory, apart; `fwd` the pass's factors.
            unsafe { fft.run_half_last(p, p, work.as_mut_ptr(), *k, fwd.as_ptr().cast()) };
            return;
        }
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
            Algo::Odd { kern, g } => {
                // SAFETY: `x` holds `(n + 1)/2` values and `y` room for the n reals.
                unsafe { (kern[1])(g, core::ptr::null(), x.as_ptr(), y.as_mut_ptr().cast()) }
            }
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
        let Algo::Half { fft, fused, .. } = &*self.algo else {
            unreachable!("an even length")
        };
        if let Some(fu) = fused {
            let (x, z) = (&x[..=self.n / 2], &mut z[..self.n / 2]);
            // SAFETY: `x` holds `m + 1` values and `z` room for `m`.
            unsafe { fu.inverse(x.as_ptr(), z.as_mut_ptr()) };
            return;
        }
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
        // For each s < h: the spectra of the residue sequences (a pair's apart), twiddled, and
        // the butterfly of r: the first half of the spectrum.
        // SAFETY: the pairs' `np·m` and the last sequence's `h` values, the `4(r − 1)h` reals of
        // the twiddles, and `y` of `(n + 1)/2` values, apart from each other.
        unsafe {
            (self.fwd)(
                &self.g,
                self.tw.as_ptr(),
                zb.as_ptr(),
                yl.as_ptr(),
                m,
                y.as_mut_ptr(),
            )
        };
        y[0].im = T::ZERO;
    }

    /// The unscaled inverse transform of the `(n + 1)/2` values `x` into the `n` reals `y`.
    fn inverse(&self, x: &[Complex<T>], y: &mut [T], work: &mut [Complex<T>]) {
        let (r, m) = (self.r, self.m);
        let (h, np) = (m.div_ceil(2), (r - 1) / 2);
        let (zb, work) = work.split_at_mut(np * m);
        let (ul, work) = work.split_at_mut(h);
        let (xl, work) = work.split_at_mut(h);
        // For each s < h: the butterfly of r on X[s + q·m], twiddled, into the pairs'
        // transforms Z = U_2p + i·U_2p+1 (all of m) and the last sequence's half spectrum.
        // SAFETY: `x` holds the `(n + 1)/2` values, the twiddles `4(r − 1)h` reals; the pairs'
        // `np·m` and the last sequence's `h` values, apart from each other.
        unsafe {
            (self.inv)(
                &self.g,
                self.tw.as_ptr(),
                x.as_ptr(),
                m,
                zb.as_mut_ptr(),
                ul.as_mut_ptr(),
            )
        };
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
