//! The plan of a complex transform.

use super::complex::{Algo, Isa};
use super::{FftFloat, check_batch};
use crate::math::complex::Complex;
use std::sync::Arc;

/// The discrete Fourier transform of complex sequences of the length `n`, planned once:
/// `X[k] = Σ x[j]·e^(−2πi·jk/n)` forward and `x[j] = Σ X[k]·e^(2πi·jk/n)` inverse, unscaled as in
/// FFTW, so that `inverse(forward(x)) = n·x`.
///
/// Every length is planned, in time `O(n log n)`: by passes of the radices 2, 4 and 8, 3, 5 and
/// 7 and the primes up to 19 (Stockham's form, no bit reversal), a larger prime factor by its
/// own plan, and a prime length above 19 by Rader's algorithm or by Bluestein's with a smooth
/// inner length, whichever is estimated faster. The kernels are NEON on aarch64, SSE2 or (detected
/// when the plan is made) AVX2 with FMA on x86_64, and portable Rust elsewhere; `Debug` shows the
/// plan. A plan is immutable and shared by its clones.
///
/// Each method takes a slice of any multiple of `n` values and transforms each run of `n` in
/// turn (a batch); it panics on a length that is not a multiple. The `_with_scratch` methods take
/// the working memory from the caller, at least [`scratch_len`](Self::scratch_len) values, and
/// then never allocate.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::fft::Fft;
///
/// let fft = Fft::<f32>::new(3);
/// let x = [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0), Complex::new(-1.0, 0.0)];
/// let mut y = [Complex::ZERO; 3];
/// fft.forward_to(&x, &mut y);
/// let mut z = y;
/// fft.inverse(&mut z);
/// for (z, x) in z.iter().zip(&x) {
///     assert!((*z - *x * 3.0).abs() < 1e-5);
/// }
/// ```
#[derive(Clone)]
pub struct Fft<T> {
    n: usize,
    scratch: usize,
    isa: Isa,
    algo: Arc<Algo<T>>,
}

impl<T: FftFloat> Fft<T> {
    /// The plan for the length `n` (0 and 1 give the identity).
    pub fn new(n: usize) -> Fft<T> {
        let isa = Isa::detect();
        let algo = Algo::new(n, isa);
        Fft {
            n,
            scratch: algo.scratch(),
            isa,
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

    /// The number of values of working memory the `_with_scratch` methods need: what one run
    /// takes (0 for the lengths transformed by one pass in place). More lets the short runs of
    /// a batch go through the passes together.
    pub fn scratch_len(&self) -> usize {
        self.scratch
    }

    /// The working memory the allocating methods take for `len` values: one run's, or room
    /// for the short runs of a batch to go pass by pass together.
    fn work(&self, len: usize) -> usize {
        if len > self.n {
            self.algo.batch_scratch()
        } else {
            self.scratch
        }
    }

    /// The forward transform of each run of `n` values of `data`, in place.
    #[track_caller]
    pub fn forward(&self, data: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::ZERO; self.work(data.len())];
        self.forward_with_scratch(data, &mut scratch);
    }

    /// The inverse transform of each run of `n` values of `data`, in place, unscaled.
    #[track_caller]
    pub fn inverse(&self, data: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::ZERO; self.work(data.len())];
        self.inverse_with_scratch(data, &mut scratch);
    }

    /// [`forward`](Self::forward) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values.
    #[track_caller]
    pub fn forward_with_scratch(&self, data: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.in_place(data, scratch, false);
    }

    /// [`inverse`](Self::inverse) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values.
    #[track_caller]
    pub fn inverse_with_scratch(&self, data: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.in_place(data, scratch, true);
    }

    /// The forward transform of each run of `n` values of `input` into the same place in
    /// `output`, of the same length.
    #[track_caller]
    pub fn forward_to(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        self.out_of_place(input, output, false);
    }

    /// The inverse transform of each run of `n` values of `input` into the same place in
    /// `output`, of the same length, unscaled.
    #[track_caller]
    pub fn inverse_to(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        self.out_of_place(input, output, true);
    }

    #[track_caller]
    fn in_place(&self, data: &mut [Complex<T>], scratch: &mut [Complex<T>], inverse: bool) {
        let need = self.scratch;
        if data.len() == self.n && self.n > 1 && scratch.len() >= need {
            // One run: no divisions for the batch.
            let p = data.as_mut_ptr();
            // SAFETY: `data` holds the n values, transformed in place; the scratch, a distinct
            // slice, the plan's working memory.
            unsafe { self.algo.run(p, p, scratch.as_mut_ptr(), inverse) };
            return;
        }
        check_batch(data.len(), self.n, "Fft");
        assert!(
            scratch.len() >= need,
            "Fft: scratch of {} values, {} needed",
            scratch.len(),
            need
        );
        if self.n <= 1 {
            return;
        }
        let (per, _) = self.algo.batch(scratch.len());
        let scr = scratch.as_mut_ptr();
        for runs in data.chunks_mut(per * self.n) {
            let p = runs.as_mut_ptr();
            // SAFETY: `runs` holds whole runs of n values, at most the batch the scratch has
            // room for, transformed in place; the scratch, a distinct slice.
            unsafe { self.algo.run_batch(p, p, scr, runs.len() / self.n, inverse) };
        }
    }

    #[track_caller]
    fn out_of_place(&self, input: &[Complex<T>], output: &mut [Complex<T>], inverse: bool) {
        check_batch(input.len(), self.n, "Fft");
        assert_eq!(
            input.len(),
            output.len(),
            "Fft: input and output lengths differ"
        );
        if self.n == 0 || input.is_empty() {
            return;
        }
        let mut scratch = vec![Complex::ZERO; self.work(input.len())];
        let (per, _) = self.algo.batch(scratch.len());
        let per = per * self.n;
        let scr = scratch.as_mut_ptr();
        for (x, y) in input.chunks(per).zip(output.chunks_mut(per)) {
            // SAFETY: `x` and `y` are distinct slices of the same whole runs, at most the
            // plan's batch; the scratch, the plan's working memory.
            unsafe {
                self.algo
                    .run_batch(x.as_ptr(), y.as_mut_ptr(), scr, x.len() / self.n, inverse)
            };
        }
    }
}

impl<T> core::fmt::Debug for Fft<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fft")
            .field("n", &self.n)
            .field("algorithm", &self.algo.describe())
            .field("isa", &self.isa)
            .finish_non_exhaustive()
    }
}
