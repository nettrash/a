//! The plan of a complex transform.

use super::complex::{Algo, Isa, Short};
use super::{FftFloat, check_batch, with_work};
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
    /// The direct kernel of a length of one butterfly, its pass inside `algo`.
    short: Option<Short<T>>,
}

// SAFETY: `short` points into the plan `algo` holds (kept alive by the `Arc`, never changed);
// the rest is a shared immutable plan, `Send` and `Sync` as `Algo` is.
unsafe impl<T: Send + Sync> Send for Fft<T> {}
// SAFETY: as above; nothing is mutated through a shared plan.
unsafe impl<T: Send + Sync> Sync for Fft<T> {}

impl<T: FftFloat> Fft<T> {
    /// The plan for the length `n` (0 and 1 give the identity).
    pub fn new(n: usize) -> Fft<T> {
        let isa = Isa::detect();
        let algo = Arc::new(Algo::new(n, isa));
        // The pass the direct kernel points to is in the `Arc`'s allocation, which does not move.
        let short = algo.short();
        Fft {
            n,
            scratch: algo.scratch(),
            isa,
            algo,
            short,
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
        if self.one(data, false) {
            return;
        }
        with_work(self.work(data.len()), |s| {
            self.forward_with_scratch(data, s)
        });
    }

    /// The inverse transform of each run of `n` values of `data`, in place, unscaled.
    #[track_caller]
    pub fn inverse(&self, data: &mut [Complex<T>]) {
        if self.one(data, true) {
            return;
        }
        with_work(self.work(data.len()), |s| {
            self.inverse_with_scratch(data, s)
        });
    }

    /// [`forward`](Self::forward) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values.
    #[track_caller]
    #[inline]
    pub fn forward_with_scratch(&self, data: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if !self.one(data, false) {
            self.in_place(data, scratch, false);
        }
    }

    /// [`inverse`](Self::inverse) with the working memory `scratch` of at least
    /// [`scratch_len`](Self::scratch_len) values.
    #[track_caller]
    #[inline]
    pub fn inverse_with_scratch(&self, data: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        if !self.one(data, true) {
            self.in_place(data, scratch, true);
        }
    }

    /// The forward transform of each run of `n` values of `input` into the same place in
    /// `output`, of the same length.
    #[track_caller]
    #[inline]
    pub fn forward_to(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if !self.one_to(input, output, false) {
            self.out_of_place(input, output, false);
        }
    }

    /// The inverse transform of each run of `n` values of `input` into the same place in
    /// `output`, of the same length, unscaled.
    #[track_caller]
    #[inline]
    pub fn inverse_to(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if !self.one_to(input, output, true) {
            self.out_of_place(input, output, true);
        }
    }

    /// One run of a length of one butterfly, by its kernel directly; whether it was.
    #[inline(always)]
    fn one(&self, data: &mut [Complex<T>], inverse: bool) -> bool {
        match &self.short {
            Some(short) if data.len() == self.n => {
                let p = data.as_mut_ptr();
                // SAFETY: `data` holds the run, transformed in place; the plan is alive in
                // `self`.
                unsafe { short.run(p, p, inverse) };
                true
            }
            _ => false,
        }
    }

    /// [`one`](Self::one) out of place.
    #[inline(always)]
    fn one_to(&self, input: &[Complex<T>], output: &mut [Complex<T>], inverse: bool) -> bool {
        match &self.short {
            Some(short) if input.len() == self.n && output.len() == self.n => {
                // SAFETY: distinct slices of the run; the plan is alive in `self`.
                unsafe { short.run(input.as_ptr(), output.as_mut_ptr(), inverse) };
                true
            }
            _ => false,
        }
    }

    #[track_caller]
    #[inline(never)]
    fn in_place(&self, data: &mut [Complex<T>], scratch: &mut [Complex<T>], inverse: bool) {
        let need = self.scratch;
        if data.len() != self.n || scratch.len() < need {
            check_batch(data.len(), self.n, "Fft");
            assert!(
                scratch.len() >= need,
                "Fft: scratch of {} values, {} needed",
                scratch.len(),
                need
            );
        }
        // SAFETY: `data` holds whole runs, the scratch is a distinct slice of enough values.
        unsafe {
            self.run_raw(
                data.as_mut_ptr(),
                data.len(),
                scratch.as_mut_ptr(),
                scratch.len(),
                inverse,
            )
        }
    }

    /// The values of working memory that let a call on `len` values run its short runs
    /// together: one run's, or the room of a whole batch of short runs.
    pub(crate) fn batch_work(&self, len: usize) -> usize {
        self.work(len)
    }

    /// The fused last pass for the even real transform of `2n` (see [`Algo::half_last`]).
    pub(crate) fn half_last(&self) -> Option<crate::math::fft::complex::HKern<T>> {
        if self.short.is_some() {
            return None;
        }
        self.algo.half_last(self.isa)
    }

    /// One run from `x` into `y` (`n + 1` values) with the last pass by `k`, the factors `tw`;
    /// `scr` the working memory of [`scratch_len`](Self::scratch_len) values, not initialised.
    ///
    /// # Safety
    /// `x` valid for reading `n` values, `y` for writing `n + 1` (the same as `x` or apart),
    /// `scr` apart from both; `k` from [`half_last`](Self::half_last), `tw` its factors.
    #[inline]
    pub(crate) unsafe fn run_half_last(
        &self,
        x: *const Complex<T>,
        y: *mut Complex<T>,
        scr: *mut Complex<T>,
        k: crate::math::fft::complex::HKern<T>,
        tw: *const T,
    ) {
        // SAFETY: the caller's contract.
        unsafe { self.algo.run_half_last(x, y, scr, k, tw) }
    }

    /// One run from `x` into `y`, with the working memory `scr` of
    /// [`scratch_len`](Self::scratch_len) values that need not be initialised.
    ///
    /// # Safety
    /// `x` must be valid for reading and `y` for writing `n` values, not overlapping; `scr` for
    /// reading and writing the working memory, apart from both.
    #[inline]
    pub(crate) unsafe fn run_to_raw(
        &self,
        x: *const Complex<T>,
        y: *mut Complex<T>,
        scr: *mut Complex<T>,
        inverse: bool,
    ) {
        if self.n == 0 {
            return;
        }
        // SAFETY: the caller's contract; the algorithms write their working memory before they
        // read it.
        unsafe {
            match &self.short {
                Some(short) => short.run(x, y, inverse),
                None => self.algo.run(x, y, scr, inverse),
            }
        }
    }

    /// The transform of each run of `n` values of the `len` at `data`, in place, with the
    /// working memory at `scr` of `scr_len` values (at least [`scratch_len`](Self::scratch_len);
    /// more lets the short runs of a batch go through the passes together).
    ///
    /// # Safety
    /// `len` must be a multiple of `n`; `data` must be valid for reading and writing `len`
    /// values and `scr` for `scr_len` (which need not be initialised: every value is written
    /// before it is read), not overlapping `data`.
    pub(crate) unsafe fn run_raw(
        &self,
        data: *mut Complex<T>,
        len: usize,
        scr: *mut Complex<T>,
        scr_len: usize,
        inverse: bool,
    ) {
        if self.n <= 1 || len == 0 {
            return;
        }
        // SAFETY: the caller's contract.
        unsafe {
            if len == self.n {
                match &self.short {
                    Some(short) => short.run(data, data, inverse),
                    None => self.algo.run(data, data, scr, inverse),
                }
                return;
            }
            let (per, _) = self.algo.batch(scr_len);
            let mut c = 0;
            while c < len {
                let runs = (per * self.n).min(len - c);
                let p = data.add(c);
                self.algo.run_batch(p, p, scr, runs / self.n, inverse);
                c += runs;
            }
        }
    }

    #[track_caller]
    #[inline(never)]
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
        if input.len() == self.n {
            // One run: straight from the input into the output, no division into batches.
            with_work(self.scratch, |scr| {
                // SAFETY: `input` and `output` hold the n values and are distinct slices; the
                // scratch, a third one, the plan's working memory for one run.
                unsafe {
                    self.algo.run(
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        scr.as_mut_ptr(),
                        inverse,
                    )
                }
            });
            return;
        }
        with_work(self.work(input.len()), |scratch| {
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
        });
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
