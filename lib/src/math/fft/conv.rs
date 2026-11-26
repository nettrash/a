//! Linear and circular convolution and correlation, directly for short operands and by transforms
//! for long ones.
//!
//! # The paths
//!
//! * **Scalar**: the double loop on stack copies, for at most [`SCALAR_PRODUCTS`] products.
//! * **Direct**: `c[k] = Σ_t taps[t]·x[k + t]`, with `x` the longer operand between
//!   `len(short) − 1` zeros and `taps` the shorter one reversed, by the SIMD kernels of `conv/`
//!   (NEON on aarch64, SSE2 and AVX2 with FMA on x86_64, the same steps on arrays of lanes
//!   elsewhere and under Miri; the bodies are the macros of `conv/macros.rs`): a block of outputs
//!   is kept in registers while the taps run past it, and each load of `x` serves two taps.
//!   Complex values stay interleaved: the sums of `x·re(tap)` and `x·im(tap)` are taken lane by
//!   lane and combined once a block. The outputs that every tap reaches, a multiple of
//!   [`INTERIOR`], are computed in place on the longer operand, the others on short copies
//!   between zeros. An operand reversed or conjugated (a correlation) is handled by
//!   `conv(op(u), v) = op(conv(u, op(v)))`, so the longer operand is never copied.
//! * **Transforms**: the operands padded to a length `L` of the form `2^a·3^b·5^c·7^d` (even for
//!   real data, which go through [`RealFft`]), multiplied in the frequency domain (the scale `1/L`
//!   folded in, by the kernels' `cmul`) and transformed back. When one operand is much shorter
//!   than the other, the longer is cut into blocks of `B = L − len(short) + 1` values whose
//!   convolutions are added where they overlap (overlap-add), the shorter one transformed once
//!   and the blocks transformed in batches; one block is the plain padded product.
//!
//! The path and the length `L` are chosen by a model of their costs in nanoseconds, from
//! constants measured for each type ([`Costs`]): the direct path costs about
//! `direct + copy·len(c) + mac·len(c)·len(short)`, a transform of the length `L` about
//! `call + fft·L·(a + w₃·b + w₅·c + w₇·d)` (so `fft·L·log₂ L` for a power of two), planning
//! `PLAN_SHARE·(plan_call + plan·L)` and each pass over a transform's buffers `pass·L`. The
//! bench's `calibrate` measures them; its `crossover.py` prints the crossover they give.
//!
//! The plans of the transforms are kept for the next calls by a bounded cache shared by the
//! threads (`conv/cache.rs`): at most 16 plans and 16 MiB, the least recently used evicted.

use super::{Fft, FftFloat, RealFft};
use crate::math::complex::{Complex, Float};
use core::any::TypeId;
use core::mem::MaybeUninit;

#[macro_use]
mod macros;
mod cache;

// The kernels, one set per target under the name `kernels`.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
#[path = "conv/aarch64.rs"]
mod kernels;
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "conv/x86_64.rs"]
mod kernels;
#[cfg(not(any(
    all(target_arch = "aarch64", target_endian = "little", not(miri)),
    all(target_arch = "x86_64", not(miri))
)))]
#[path = "conv/portable.rs"]
mod kernels;

use kernels::PAD;

/// Up to this many products, `len(a)·len(b)`, a linear convolution is computed by the double loop
/// (when neither length exceeds [`SCALAR_LEN`]).
const SCALAR_PRODUCTS: usize = 16;

/// The longest operand of the double loop.
const SCALAR_LEN: usize = 16;

/// The most taps of the direct path kept on the stack.
const STACK_TAPS: usize = 128;

/// The longest copy of the direct path kept on the stack.
const STACK_X: usize = 512;

/// The costs, in nanoseconds, that the choice of a path is made from: see the module's notes.
/// Measured on an Apple M4 Max by the bench's `calibrate` (`radix` the median of the four types,
/// `slow` and `odd` per type from its lengths `11·2^k`, `13·2^k`, primes and odd real lengths,
/// `mac` from sweeps of the direct path near the crossover, where its taps are longer than the
/// fit of `calibrate` sees), and again for v0.2's final transforms and kernels (below the
/// constants): `fft`, `call`, `plan`, `plan_call`, `pass`, `radix`, `slow` and `odd` are to be
/// measured again whenever the transforms change, `mac`, `copy` and `direct` whenever the kernels
/// do. Planning is charged at [`PLAN_SHARE`].
pub struct Costs {
    /// A product added into a sum by the direct kernels, per output and tap.
    mac: f64,
    /// Each value of the result of the direct path (its allocation, first writes and the copies
    /// at its ends).
    copy: f64,
    /// The fixed cost of the direct path (its allocations).
    direct: f64,
    /// A transform of the length `L` costs `fft · L · weight(L)`.
    fft: f64,
    /// The fixed cost of a call of a transform.
    call: f64,
    /// Planning a transform of the length `L` costs `plan · L + plan_call`.
    plan: f64,
    /// The fixed cost of planning a transform.
    plan_call: f64,
    /// Each value of a transform's buffers passed over outside the transforms: the copy into
    /// the padded buffer, the product of the spectra, the sum into the result.
    pass: f64,
    /// The weights of a factor 3, 5 and 7 in `weight(L)`, a factor 2 weighing 1.
    radix: [f64; 3],
    /// The weight of a factor 11 and 13, and the cost of a length with a larger prime factor
    /// relative to a padded transform of `2L` (Bluestein's or Rader's algorithm).
    slow: [f64; 3],
    /// The cost of a real transform of an odd length relative to an even one of the same weight.
    odd: f64,
}

/// The share of the cost of planning that the choice of a path charges. The plans are kept for
/// the next calls (`conv/cache.rs`), so a call of a length seen before pays none of it and a
/// first call all of it; the choice cannot depend on what is kept, or the same call would give
/// other bits after another one. Measured (the bench's `alpha.py` over sweeps of the three costs):
/// charging all of it loses up to 3.2 times (`f32`) on repeated calls in the band where the kept
/// plan makes the transforms win, charging none of it up to 2.9 times on first calls; a half
/// keeps both within about 1.6 (2.0 for repeated `f32`), the geometric means within 1.03.
const PLAN_SHARE: f64 = 0.5;

// Measured again on an Apple M4 Max for the final transforms and kernels of v0.2: the direct
// path's times (64 to 30,000 values against 5 to 384 taps) fit `mac` 0.0175, 0.0346, 0.0726 and
// 0.139 ns, `copy` 0.038, 0.056, 0.091 and 0.35 and `direct` 45, 56, 59 and 77 (`f32`, `f64`,
// `Complex<f32>`, `Complex<f64>`); the mean of a forward and an inverse transform of the smooth
// lengths from 64 to 65,536 fits `fft` 0.043, 0.080, 0.080 and 0.153 (0.78 to 0.81 of the
// constants before) with the weights of 3, 5 and 7 at about 2.25, 2.95 and 3.7, and planning
// 64 to 2048 values `plan_call` 375, 306, 315 and 210 and `plan` 1.66, 2.24, 1.72 and 2.17. For
// `f32`, `f64` and `Complex<f32>` the constants below take the direct path's fitted values and,
// for the transforms, the constants before scaled by 0.85, 0.90 and 0.87, which keeps the
// transform lengths they choose: against the constants before (the same code alternated, plans
// kept, two runs) the shapes whose path changes take 0.51 to 0.75 of their time and no shape
// measured more than 1.04 of it. `Complex<f64>` keeps the constants before. The fitted values,
// or smaller factors, move the switch further (0.49 to 0.95 of the time on the shapes they
// change) but slow others down: a lower `fft` weakens the bound that skips the search of the
// transform lengths, so that small direct convolutions pay that search (about 0.2 µs: 1.15 to
// 1.76 times at 200 × 12 in `Complex<f32>`, 1500 × 32 in `f64` and the circular `Complex<f64>`
// of 16), and the fitted weights choose lengths up to 1.15 times slower inside overlap-add
// (powers of two of 2048 in batches).
/// `f32`, real.
const COSTS_F32: Costs = Costs {
    mac: 0.0175,
    copy: 0.038,
    direct: 45.0,
    fft: 0.04675,
    call: 13.6,
    plan: 1.122,
    plan_call: 425.0,
    pass: 0.425,
    radix: [2.1, 2.4, 3.0],
    slow: [4.9, 4.4, 2.5],
    odd: 2.7,
};

/// `f64`, real.
const COSTS_F64: Costs = Costs {
    mac: 0.0346,
    copy: 0.056,
    direct: 56.0,
    fft: 0.0918,
    call: 10.8,
    plan: 1.521,
    plan_call: 360.0,
    pass: 0.54,
    radix: [2.1, 2.4, 3.0],
    slow: [5.6, 9.5, 2.3],
    odd: 1.6,
};

/// `Complex<f32>`.
const COSTS_C32: Costs = Costs {
    mac: 0.0726,
    copy: 0.091,
    direct: 59.0,
    fft: 0.087,
    call: 8.7,
    plan: 1.9662,
    plan_call: 174.0,
    pass: 0.696,
    radix: [2.1, 2.4, 3.0],
    slow: [4.8, 6.9, 1.2],
    odd: 1.0,
};

/// `Complex<f64>`.
const COSTS_C64: Costs = Costs {
    mac: 0.146,
    copy: 0.27,
    direct: 100.0,
    fft: 0.19,
    call: 10.0,
    plan: 2.15,
    plan_call: 220.0,
    pass: 1.0,
    radix: [2.1, 2.4, 3.0],
    slow: [5.9, 9.7, 1.2],
    odd: 1.0,
};

/// Whether `T` is `U`.
#[inline(always)]
fn is<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

/// The direct kernel of `T`: `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`.
///
/// # Safety
///
/// `x` must be valid for reads of `ny + nt − 1 + PAD` values, `taps` of `nt` and `y` for writes
/// of `ny`.
#[inline]
unsafe fn fir<T: FftFloat>(x: *const T, taps: *const T, nt: usize, y: *mut T, ny: usize) {
    // SAFETY: forwarded; `T` is `f32` or `f64`, the only two `FftFloat`s.
    unsafe {
        if is::<T, f32>() {
            kernels::fir_f32(x.cast(), taps.cast(), nt, y.cast(), ny)
        } else {
            kernels::fir_f64(x.cast(), taps.cast(), nt, y.cast(), ny)
        }
    }
}

/// The real direct kernel of `T` for `1 ≤ nt ≤ 4` taps.
///
/// # Safety
///
/// `x` must be valid for reads of `ny + nt − 1` values, `taps` of `nt` and `y` for writes of `ny`.
#[inline]
unsafe fn fir_few<T: FftFloat>(x: *const T, taps: *const T, nt: usize, y: *mut T, ny: usize) {
    // SAFETY: forwarded; `T` is `f32` or `f64`.
    unsafe {
        if is::<T, f32>() {
            kernels::fir_few_f32(x.cast(), taps.cast(), nt, y.cast(), ny)
        } else {
            kernels::fir_few_f64(x.cast(), taps.cast(), nt, y.cast(), ny)
        }
    }
}

/// The complex direct kernel of `T` for `1 ≤ nt ≤ 4` taps, the counts in complex values.
///
/// # Safety
///
/// As for [`fir_few`], in complex values.
#[inline]
unsafe fn firc_few<T: FftFloat>(x: *const T, taps: *const T, nt: usize, y: *mut T, ny: usize) {
    // SAFETY: forwarded; `T` is `f32` or `f64`.
    unsafe {
        if is::<T, f32>() {
            kernels::firc_few_f32(x.cast(), taps.cast(), nt, y.cast(), ny)
        } else {
            kernels::firc_few_f64(x.cast(), taps.cast(), nt, y.cast(), ny)
        }
    }
}

/// The complex direct kernel of `T` on interleaved parts, the counts in complex values.
///
/// # Safety
///
/// `x` must be valid for reads of `ny + nt − 1 + PAD` complex values, `taps` of `nt` and `y` for
/// writes of `ny`.
#[inline]
unsafe fn firc<T: FftFloat>(x: *const T, taps: *const T, nt: usize, y: *mut T, ny: usize) {
    // SAFETY: forwarded; `T` is `f32` or `f64`.
    unsafe {
        if is::<T, f32>() {
            kernels::firc_f32(x.cast(), taps.cast(), nt, y.cast(), ny)
        } else {
            kernels::firc_f64(x.cast(), taps.cast(), nt, y.cast(), ny)
        }
    }
}

/// `x[j] = x[j]·y[j]·s` for each `j` below the length of `x`, at most that of `y`.
#[inline]
fn cmul<T: FftFloat>(x: &mut [Complex<T>], y: &[Complex<T>], s: T) {
    let n = x.len();
    assert!(y.len() >= n);
    // SAFETY: `Complex` is `repr(C)`, two parts; both slices hold `n` values; `T` is `f32` or
    // `f64`, and `s` is read as the one it is.
    unsafe {
        let (xp, yp) = (x.as_mut_ptr().cast::<T>(), y.as_ptr().cast::<T>());
        if is::<T, f32>() {
            kernels::cmul_f32(xp.cast(), yp.cast(), *(&s as *const T).cast::<f32>(), n)
        } else {
            kernels::cmul_f64(xp.cast(), yp.cast(), *(&s as *const T).cast::<f64>(), n)
        }
    }
}

/// `1/n`.
#[inline]
fn recip<T: Float>(n: usize) -> T {
    T::ONE / T::from_f64(n as f64)
}

/// A sequence `v`, or `v` reversed, conjugated or both, read without being copied.
#[derive(Clone, Copy)]
pub struct Op<'a, S> {
    v: &'a [S],
    rev: bool,
    conj: bool,
}

impl<'a, S: Sample> Op<'a, S> {
    fn new(v: &'a [S]) -> Self {
        Op {
            v,
            rev: false,
            conj: false,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.v.len()
    }

    /// The sequence reversed.
    fn reversed(self) -> Self {
        Op {
            rev: !self.rev,
            ..self
        }
    }

    /// The sequence reversed and conjugated.
    fn rev_conj(self) -> Self {
        Op {
            v: self.v,
            rev: !self.rev,
            conj: !self.conj,
        }
    }

    /// The range of `dst` that the values `start..start + dst.len()` inside the sequence fall
    /// on, and the range of `v` they come from (reversed when `rev`).
    #[inline]
    fn ranges(
        &self,
        start: isize,
        len: usize,
    ) -> Option<(core::ops::Range<usize>, core::ops::Range<usize>)> {
        let n = self.v.len() as isize;
        let lo = start.max(0);
        let hi = (start + len as isize).min(n);
        if lo >= hi {
            return None;
        }
        let off = (lo - start) as usize;
        let (lo, hi) = (lo as usize, hi as usize);
        let src = if self.rev {
            self.v.len() - hi..self.v.len() - lo
        } else {
            lo..hi
        };
        Some((off..off + hi - lo, src))
    }

    /// `dst[i]` = the value `start + i` where it is inside the sequence; the rest of `dst` is
    /// left as it is.
    fn write(&self, dst: &mut [S], start: isize) {
        let Some((d, r)) = self.ranges(start, dst.len()) else {
            return;
        };
        let (dst, src) = (&mut dst[d], &self.v[r]);
        match (self.rev, self.conj) {
            (false, false) => dst.copy_from_slice(src),
            (false, true) => {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = s.conj();
                }
            }
            (true, false) => {
                for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
                    *d = *s;
                }
            }
            (true, true) => {
                for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
                    *d = s.conj();
                }
            }
        }
    }

    /// `dst[i]` = the value `start + i`, 0 past the ends.
    fn fill(&self, dst: &mut [S], start: isize) {
        match self.ranges(start, dst.len()) {
            Some((d, _)) => {
                dst[..d.start].fill(S::zero());
                dst[d.end..].fill(S::zero());
            }
            None => dst.fill(S::zero()),
        }
        self.write(dst, start);
    }
}

pub(crate) mod sealed {
    use super::Costs;
    use crate::math::complex::Complex;
    use crate::math::fft::FftFloat;

    /// The arithmetic and the transforms of a sample type.
    pub trait Sample:
        Copy + core::ops::Add<Output = Self> + core::ops::Mul<Output = Self> + 'static
    {
        /// The type of the parts.
        type Real: FftFloat;
        /// The plan of a transform.
        type Plan: Send + Sync + 'static;
        /// 0.
        fn zero() -> Self;
        /// The complex conjugate (the value itself for a real).
        fn conj(self) -> Self;
        /// The costs of the paths.
        fn costs() -> &'static Costs;
        /// Whether the values are real (and their transforms need an even length to be fast).
        fn real() -> bool;
        /// The plan of a transform of the length `n`, from the cache of plans.
        fn plan(n: usize) -> std::sync::Arc<Self::Plan>;
        /// The values of the spectrum of a sequence of the length `n`.
        fn spectrum_len(n: usize) -> usize;
        /// The forward transforms of each run of `n` in `x` into `y`.
        fn forward(plan: &Self::Plan, x: &[Self], y: &mut [Complex<Self::Real>]);
        /// The unscaled inverse transforms of each spectrum in `y` into `x`.
        fn inverse(plan: &Self::Plan, y: &[Complex<Self::Real>], x: &mut [Self]);
        /// The direct kernel: `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`.
        ///
        /// # Safety
        ///
        /// `x` must be valid for reads of `ny + nt − 1 + PAD` values, `taps` of `nt` and `y` for
        /// writes of `ny`.
        unsafe fn fir(x: *const Self, taps: *const Self, nt: usize, y: *mut Self, ny: usize);
        /// The direct kernel for `1 ≤ nt ≤ 4` taps: as [`fir`](Self::fir), `x` valid for reads
        /// of `ny + nt − 1` values only.
        ///
        /// # Safety
        ///
        /// As for [`fir`](Self::fir), without the padding.
        unsafe fn fir_few(x: *const Self, taps: *const Self, nt: usize, y: *mut Self, ny: usize);
    }
}

/// `f32`, `f64`, `Complex<f32>` or `Complex<f64>`: the values [`convolve`], [`correlate`],
/// [`circular_convolve`] and [`circular_correlate`] take. The trait is sealed.
pub trait Sample: sealed::Sample {}

impl<T: FftFloat> Sample for T {}
impl<T: FftFloat> Sample for Complex<T> {}

impl<T: FftFloat> sealed::Sample for T {
    type Real = T;
    type Plan = RealFft<T>;
    #[inline]
    fn zero() -> Self {
        T::ZERO
    }
    #[inline]
    fn conj(self) -> Self {
        self
    }
    fn costs() -> &'static Costs {
        if is::<T, f32>() {
            &COSTS_F32
        } else {
            &COSTS_F64
        }
    }
    fn real() -> bool {
        true
    }
    fn plan(n: usize) -> std::sync::Arc<RealFft<T>> {
        cache::plan(n, cache::bytes(n, size_of::<Complex<T>>()), || {
            RealFft::new(n)
        })
    }
    fn spectrum_len(n: usize) -> usize {
        n / 2 + 1
    }
    fn forward(plan: &RealFft<T>, x: &[T], y: &mut [Complex<T>]) {
        plan.forward(x, y);
    }
    fn inverse(plan: &RealFft<T>, y: &[Complex<T>], x: &mut [T]) {
        plan.inverse(y, x);
    }
    unsafe fn fir(x: *const T, taps: *const T, nt: usize, y: *mut T, ny: usize) {
        // SAFETY: the caller's.
        unsafe { fir(x, taps, nt, y, ny) }
    }
    unsafe fn fir_few(x: *const T, taps: *const T, nt: usize, y: *mut T, ny: usize) {
        // SAFETY: the caller's.
        unsafe { fir_few(x, taps, nt, y, ny) }
    }
}

impl<T: FftFloat> sealed::Sample for Complex<T> {
    type Real = T;
    type Plan = Fft<T>;
    #[inline]
    fn zero() -> Self {
        Complex::ZERO
    }
    #[inline]
    fn conj(self) -> Self {
        Complex::conj(self)
    }
    fn costs() -> &'static Costs {
        if is::<T, f32>() {
            &COSTS_C32
        } else {
            &COSTS_C64
        }
    }
    fn real() -> bool {
        false
    }
    fn plan(n: usize) -> std::sync::Arc<Fft<T>> {
        cache::plan(n, cache::bytes(n, size_of::<Complex<T>>()), || Fft::new(n))
    }
    fn spectrum_len(n: usize) -> usize {
        n
    }
    fn forward(plan: &Fft<T>, x: &[Self], y: &mut [Self]) {
        plan.forward_to(x, y);
    }
    fn inverse(plan: &Fft<T>, y: &[Self], x: &mut [Self]) {
        plan.inverse_to(y, x);
    }
    unsafe fn fir(x: *const Self, taps: *const Self, nt: usize, y: *mut Self, ny: usize) {
        // SAFETY: the caller's; `Complex` is `repr(C)`, its two parts.
        unsafe { firc::<T>(x.cast(), taps.cast(), nt, y.cast(), ny) }
    }
    unsafe fn fir_few(x: *const Self, taps: *const Self, nt: usize, y: *mut Self, ny: usize) {
        // SAFETY: as for `fir`.
        unsafe { firc_few::<T>(x.cast(), taps.cast(), nt, y.cast(), ny) }
    }
}

/// The exponents `(b, c, d)` of `3^b·5^c·7^d` the transform lengths are made of.
const SMOOTH: [(u32, u32, u32); 30] = {
    let mut v = [(0, 0, 0); 30];
    let mut i = 0;
    let mut b = 0;
    while b <= 4 {
        let mut c = 0;
        while c <= 2 {
            let mut d = 0;
            while d <= 1 {
                v[i] = (b, c, d);
                i += 1;
                d += 1;
            }
            c += 1;
        }
        b += 1;
    }
    v
};

impl Costs {
    /// The weight of `2^a·3^b·5^c·7^d` per value: `a + w₃·b + w₅·c + w₇·d`.
    #[inline]
    fn weight(&self, a: u32, (b, c, d): (u32, u32, u32)) -> f64 {
        a as f64 + self.radix[0] * b as f64 + self.radix[1] * c as f64 + self.radix[2] * d as f64
    }

    /// The least weight of a radix of a smooth length per bit of it, at most 1: the transforms
    /// cost at least `fft·L·log₂ L` times this.
    fn low_weight(&self) -> f64 {
        let r = self.radix;
        (r[0] / 3f64.log2())
            .min(r[1] / 5f64.log2())
            .min(r[2] / 7f64.log2())
            .min(1.0)
    }

    /// The cost charged for planning a transform of the length `l`: [`PLAN_SHARE`] of its cost.
    #[inline]
    fn plan_cost(&self, l: usize) -> f64 {
        PLAN_SHARE * (self.plan_call + self.plan * l as f64)
    }

    /// The cost of one transform of the length `n`, of any factors; `real` for real data.
    fn transform_cost(&self, n: usize, real: bool) -> f64 {
        let mut m = n;
        let mut w = 0.0;
        for (p, wp) in [
            (2, 1.0),
            (3, self.radix[0]),
            (5, self.radix[1]),
            (7, self.radix[2]),
            (11, self.slow[0]),
            (13, self.slow[1]),
        ] {
            while m.is_multiple_of(p) {
                m /= p;
                w += wp;
            }
        }
        if m > 1 {
            // A larger prime factor: Bluestein's algorithm, about `slow[2]` padded transforms of
            // twice the length.
            let (l, c) = self.best_len(2 * n - 1, false);
            return self.slow[2] * (c + self.pass * l as f64);
        }
        let odd = if real && n % 2 == 1 { self.odd } else { 1.0 };
        self.call + self.fft * n as f64 * w * odd
    }

    /// The length of the cheapest transform of at least `n` values, among `2^a·3^b·5^c·7^d` (with
    /// `a ≥ 1` when `even`), and its cost.
    fn best_len(&self, n: usize, even: bool) -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        for &e in &SMOOTH {
            let m = 3usize.pow(e.0) * 5usize.pow(e.1) * 7usize.pow(e.2);
            let mut a = 0;
            while (m << a) < n || (even && a == 0) {
                a += 1;
            }
            let l = m << a;
            let c = self.call + self.fft * l as f64 * self.weight(a, e);
            if c < best.1 {
                best = (l, c);
            }
        }
        best
    }
}

/// How a linear convolution is computed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Scalar,
    Direct,
    /// By transforms of the length `l` of blocks of `l − nt + 1` values of the longer operand
    /// (one block: the padded product).
    Fft {
        l: usize,
    },
}

/// The cost of the transform path with transforms of the length `l` (cost `t` each) for a
/// longer operand of `nl` values and a shorter of `nt`.
#[inline]
fn fft_cost(c: &Costs, nl: usize, nt: usize, l: usize, t: f64) -> f64 {
    let b = l + 1 - nt;
    let blocks = nl.div_ceil(b) as f64;
    // The plan, the shorter operand's transform and per block two transforms and the passes.
    c.plan_cost(l) + t + c.pass * l as f64 + blocks * (2.0 * t + 2.0 * c.pass * l as f64)
}

/// The path of a linear convolution of `nl ≥ nt ≥ 1` values, and its cost.
fn choose<S: Sample>(nl: usize, nt: usize) -> (Path, f64) {
    if nl * nt <= SCALAR_PRODUCTS && nl <= SCALAR_LEN {
        return (Path::Scalar, 0.0);
    }
    let c = S::costs();
    let len = nl + nt - 1;
    let direct = c.direct + c.copy * len as f64 + c.mac * kernels::mac_scale() * (len * nt) as f64;
    let mut best = (Path::Direct, direct);
    // Every transform path plans and makes three transform calls at least: below that, without
    // the logarithm of the bound after it, the direct path is the cheapest.
    if c.plan_cost(0) + 3.0 * c.call >= direct {
        return best;
    }
    // A lower bound of every transform path below, so that its search is skipped when the direct
    // path is cheaper anyway: each has a length `L ≥ 2nt − 1` (the padded product `L ≥ len`,
    // overlap-add `L ≥ 2nt`), plans it, transforms the shorter operand once and transforms the
    // blocks there and back, `blocks·L ≥ nl` values, each transform at least
    // `fft·L·log₂ L·low` with `low` the least weight of a radix per bit.
    let m = 2 * nt - 1;
    let lg = (m as f64).log2().max(1.0) * c.low_weight();
    let floor = c.plan_cost(m) + 3.0 * c.call + c.fft * lg * (m + 2 * nl) as f64;
    if floor >= best.1 {
        return best;
    }
    // The padded product, the cheapest length of at least `len`.
    let even = S::real();
    let (l, t) = c.best_len(len, even);
    let f = fft_cost(c, nl, nt, l, t);
    if f < best.1 {
        best = (Path::Fft { l }, f);
    }
    // Overlap-add: blocks of the length `l` from `2nt` up to `64nt`, below the padded product's.
    if nl >= 4 * nt {
        let hi = l.min(64 * nt.max(16));
        for &e in &SMOOTH {
            let m = 3usize.pow(e.0) * 5usize.pow(e.1) * 7usize.pow(e.2);
            let mut a = 0;
            while (m << a) < 2 * nt || (even && a == 0) {
                a += 1;
            }
            while (m << a) < hi {
                let l = m << a;
                let t = c.call + c.fft * l as f64 * c.weight(a, e);
                let f = fft_cost(c, nl, nt, l, t);
                if f < best.1 {
                    best = (Path::Fft { l }, f);
                }
                a += 1;
            }
        }
    }
    best
}

/// The linear convolution of the sequences `a` and `b`, of lengths at least 1.
fn linear<S: Sample>(a: Op<'_, S>, b: Op<'_, S>) -> Vec<S> {
    let len = a.len() + b.len() - 1;
    let mut c: Vec<S> = Vec::with_capacity(len);
    // SAFETY: room for the `len` values, each written once before the length is set.
    unsafe {
        linear_into(a, b, c.as_mut_ptr());
        c.set_len(len);
    }
    c
}

/// [`linear`] into the `len(a) + len(b) − 1` values at `y`.
///
/// # Safety
/// `y` must be valid for writing those values (they need not be initialised) and not overlap
/// the operands.
unsafe fn linear_into<S: Sample>(a: Op<'_, S>, b: Op<'_, S>, y: *mut S) {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let (nl, nt) = (long.len(), short.len());
    let path = choose::<S>(nl, nt).0;
    // SAFETY: the caller's contract, passed on.
    unsafe {
        match path {
            Path::Scalar => scalar(long, short, y),
            Path::Direct => direct(long, short, y),
            Path::Fft { l } => by_fft_into(long, short, l, y),
        }
    }
}

/// The most taps of the direct path for which every output is computed in place, without
/// copies: the outputs that take every tap by the kernel for few taps, the `nt − 1` at either
/// end by their sums over the values inside.
// Measured on an Apple M4 Max (1 to 4 taps, 16 to 10⁴ outputs, the four types): the kernels
// for few taps take 0.5 to 1.0 of the general path's time (1.03 and 1.04 at 19 and 11 outputs
// of 4 taps in `f32` and `Complex<f32>`); a kernel for more than 4 would hold more registers.
const FEW_TAPS: usize = 4;

/// The outputs of the direct path computed in place on the longer operand are a multiple of this,
/// a multiple of the outputs a step of every kernel computes (which then read no value past the
/// last they need).
const INTERIOR: usize = 64;

/// The linear convolution by the direct kernels: in place on `long` for the outputs to which all
/// of `short` contributes (a multiple of [`INTERIOR`] of them), and on short copies between
/// zeros for the rest at either end.
///
/// # Safety
/// As for [`linear_into`].
unsafe fn direct<S: Sample>(long: Op<'_, S>, short: Op<'_, S>, y: *mut S) {
    // conv(op(u), v) = op(conv(u, op(v))) for op a reversal, a conjugation or both: the kernels
    // read `long` as it is.
    let (rev, conj) = (long.rev, long.conj && !S::real());
    let short = Op {
        v: short.v,
        rev: short.rev ^ long.rev,
        conj: short.conj ^ long.conj,
    };
    let long = long.v;
    let (nl, nt) = (long.len(), short.len());
    let len = nl + nt - 1;
    // The taps, and the copies at the ends, on the stack when they are short.
    let mut taps_st = [MaybeUninit::<S>::uninit(); STACK_TAPS];
    let mut taps_heap = Vec::new();
    let taps = zeros(&mut taps_st, &mut taps_heap, nt);
    short.reversed().write(taps, 0);
    if nt <= FEW_TAPS {
        // SAFETY: the outputs `nt − 1 ..= nl − 1` read `long` below `nl`; the others are the
        // sums over the values inside. Each output below `len` is written once (the caller's
        // `y` has room for them), before the slice of them is made.
        unsafe {
            S::fir_few(long.as_ptr(), taps.as_ptr(), nt, y.add(nt - 1), nl + 1 - nt);
            for k in (0..nt - 1).chain(nl..len) {
                // The output k: Σ taps[t]·long[k + t − (nt − 1)] for the indices inside.
                let lo = (nt - 1).saturating_sub(k);
                let hi = (len - k).min(nt);
                let mut acc = S::zero();
                for t in lo..hi {
                    acc = acc + taps[t] * long[k + t + 1 - nt];
                }
                y.add(k).write(acc);
            }
            finish(core::slice::from_raw_parts_mut(y, len), rev, conj);
        }
        return;
    }
    let m = (nl + 1 - nt) / INTERIOR * INTERIOR;
    let ends = if m == 0 {
        [(0, len), (len, len)]
    } else {
        [(0, nt - 1), (nt - 1 + m, len)]
    };
    let mut buf_st = [MaybeUninit::<S>::uninit(); STACK_X];
    let mut buf_heap = Vec::new();
    // SAFETY: the outputs `nt − 1 .. nt − 1 + m` read `long` below `m + nt − 1 ≤ nl`, as `m` is
    // a multiple of every kernel's step; the others read copies of `ny + nt − 1 + PAD` values.
    // Each output below `len` is written once (the caller's `y` has room for them), before
    // the slice of them is made.
    unsafe {
        S::fir(long.as_ptr(), taps.as_ptr(), nt, y.add(nt - 1), m);
        for (k0, k1) in ends {
            if k1 > k0 {
                let n = k1 - k0 + nt - 1 + PAD;
                let buf = zeros(&mut buf_st, &mut buf_heap, n);
                Op::new(long).write(buf, k0 as isize + 1 - nt as isize);
                S::fir(buf.as_ptr(), taps.as_ptr(), nt, y.add(k0), k1 - k0);
            }
        }
        finish(core::slice::from_raw_parts_mut(y, len), rev, conj);
    }
}

/// The outputs `c` of [`direct`] reversed when `rev` and conjugated when `conj`.
#[inline]
fn finish<S: Sample>(c: &mut [S], rev: bool, conj: bool) {
    if rev {
        c.reverse();
    }
    if conj {
        for x in c.iter_mut() {
            *x = x.conj();
        }
    }
}

/// `n` zeros, in `stack` when they fit, else in `heap`.
fn zeros<'a, S: Sample, const N: usize>(
    stack: &'a mut [MaybeUninit<S>; N],
    heap: &'a mut Vec<S>,
    n: usize,
) -> &'a mut [S] {
    if n <= N {
        let s = &mut stack[..n];
        for v in s.iter_mut() {
            v.write(S::zero());
        }
        // SAFETY: the `n` values have just been written, and `MaybeUninit<S>` has the layout of
        // `S`.
        unsafe { &mut *(s as *mut [MaybeUninit<S>] as *mut [S]) }
    } else {
        heap.clear();
        heap.resize(n, S::zero());
        heap
    }
}

/// The linear convolution by the double loop, for `len(a) ≤ SCALAR_LEN` and
/// `len(b) ≤ SCALAR_LEN`, on copies on the stack.
///
/// # Safety
/// As for [`linear_into`].
unsafe fn scalar<S: Sample>(a: Op<'_, S>, b: Op<'_, S>, out: *mut S) {
    let (mut xs, mut ys) = (
        [MaybeUninit::<S>::uninit(); SCALAR_LEN],
        [MaybeUninit::<S>::uninit(); SCALAR_LEN],
    );
    let (mut xh, mut yh) = (Vec::new(), Vec::new());
    let x = zeros(&mut xs, &mut xh, a.len());
    let y = zeros(&mut ys, &mut yh, b.len());
    a.write(x, 0);
    b.write(y, 0);
    let len = a.len() + b.len() - 1;
    // SAFETY: the caller's contract: room for the `len` values, each written before the slice
    // of them is made.
    let c = unsafe {
        for k in 0..len {
            out.add(k).write(S::zero());
        }
        core::slice::from_raw_parts_mut(out, len)
    };
    for (i, &u) in x.iter().enumerate() {
        for (c, &v) in c[i..].iter_mut().zip(y.iter()) {
            *c = *c + u * v;
        }
    }
}

/// The values a batch of blocks of the overlap-add holds at most, unless one block is longer.
const BATCH: usize = 65536;

/// The linear convolution by transforms of the length `l ≥ len(short) + 1`: blocks of
/// `l − len(short) + 1` values of `long`, each convolved with `short` and added in.
fn by_fft<S: Sample>(long: Op<'_, S>, short: Op<'_, S>, l: usize) -> Vec<S> {
    let len = long.len() + short.len() - 1;
    let mut c: Vec<S> = Vec::with_capacity(len);
    // SAFETY: room for the `len` values, each written before the length is set.
    unsafe {
        by_fft_into(long, short, l, c.as_mut_ptr());
        c.set_len(len);
    }
    c
}

/// [`by_fft`] into the `len(long) + len(short) − 1` values at `out`.
///
/// # Safety
/// As for [`linear_into`].
unsafe fn by_fft_into<S: Sample>(long: Op<'_, S>, short: Op<'_, S>, l: usize, out: *mut S) {
    let (nl, nt) = (long.len(), short.len());
    let len = nl + nt - 1;
    let b = l + 1 - nt;
    let h = S::spectrum_len(l);
    let plan = S::plan(l);
    let scale = recip::<S::Real>(l);
    // The transform of the shorter operand, once.
    let mut x = vec![S::zero(); l];
    short.fill(&mut x, 0);
    let mut kernel = vec![Complex::ZERO; h];
    S::forward(&plan, &x, &mut kernel);
    let blocks = nl.div_ceil(b);
    if blocks == 1 {
        let mut spec = vec![Complex::ZERO; h];
        long.fill(&mut x, 0);
        S::forward(&plan, &x, &mut spec);
        cmul(&mut spec, &kernel, scale);
        S::inverse(&plan, &spec, &mut x);
        // SAFETY: the caller's contract: room for `len ≤ l` values.
        unsafe { core::ptr::copy_nonoverlapping(x.as_ptr(), out, len) };
        return;
    }
    let per = (BATCH / l).clamp(1, blocks);
    // SAFETY: the caller's contract: room for the `len` values, zeroed before the slice.
    let c = unsafe {
        for k in 0..len {
            out.add(k).write(S::zero());
        }
        core::slice::from_raw_parts_mut(out, len)
    };
    x.resize(per * l, S::zero());
    let mut spec = vec![Complex::ZERO; per * h];
    let mut j = 0;
    while j < blocks {
        let k = per.min(blocks - j);
        let (xs, ss) = (&mut x[..k * l], &mut spec[..k * h]);
        for (q, run) in xs.chunks_exact_mut(l).enumerate() {
            let (data, pad) = run.split_at_mut(b);
            long.fill(data, ((j + q) * b) as isize);
            pad.fill(S::zero());
        }
        S::forward(&plan, xs, ss);
        for run in ss.chunks_exact_mut(h) {
            cmul(run, &kernel, scale);
        }
        S::inverse(&plan, ss, xs);
        for (q, run) in xs.chunks_exact(l).enumerate() {
            let at = (j + q) * b;
            let n = l.min(len - at);
            for (c, &y) in c[at..at + n].iter_mut().zip(&run[..n]) {
                *c = *c + y;
            }
        }
        j += k;
    }
}

/// The linear convolution of `a` and `b`: the `len(a) + len(b) − 1` values
/// `c[k] = Σ a[i]·b[k − i]` (none when either is empty), as NumPy's `convolve` gives them in its
/// full mode.
///
/// Computed by the path the costs of the lengths favour: directly by SIMD kernels while the
/// shorter operand is short (up to some tens to a few hundred values against a long one), else
/// by real or complex transforms of a length `2^a·3^b·5^c·7^d`, cutting the longer operand into
/// blocks (overlap-add) when it is much longer than the other. The plans of those transforms
/// are kept for the next calls (at most 16 plans and 16 MiB of them, shared by all threads): a
/// repeated call of the same lengths skips the planning, and gives the same bits. The direct
/// path's sums are in the order of the taps; the transforms' error is a few units of rounding
/// of the type times
/// `‖a‖·‖b‖·log₂ len`. With infinities or NaNs among the values, the result may hold NaN where
/// the exact sums would be infinite.
///
/// # Examples
///
/// ```
/// use a::math::fft::convolve;
///
/// assert_eq!(convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5]), vec![0.0, 1.0, 2.5, 4.0, 1.5]);
/// ```
pub fn convolve<S: Sample>(a: &[S], b: &[S]) -> Vec<S> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    linear(Op::new(a), Op::new(b))
}

/// The linear cross-correlation of `a` and `b`: the `len(a) + len(b) − 1` values
/// `c[k + len(b) − 1] = Σ a[i + k]·conj(b[i])` for the lags `k` from `−(len(b) − 1)` to
/// `len(a) − 1`, as NumPy's `correlate` gives them in its full mode. The convolution of `a` with
/// `b` reversed and conjugated, by the same paths as [`convolve`], without a copy of `b`.
///
/// # Examples
///
/// ```
/// use a::math::fft::correlate;
///
/// assert_eq!(correlate(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5]), vec![0.5, 2.0, 3.5, 3.0, 0.0]);
/// ```
pub fn correlate<S: Sample>(a: &[S], b: &[S]) -> Vec<S> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    linear(Op::new(a), Op::new(b).rev_conj())
}

/// [`convolve`] into `out`, which must hold `len(a) + len(b) − 1` values (none when either is
/// empty): the same values, without allocating the result. Panics when `out` has another
/// length.
///
/// # Examples
///
/// ```
/// use a::math::fft::convolve_into;
///
/// let mut c = [0.0; 5];
/// convolve_into(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5], &mut c);
/// assert_eq!(c, [0.0, 1.0, 2.5, 4.0, 1.5]);
/// ```
#[track_caller]
pub fn convolve_into<S: Sample>(a: &[S], b: &[S], out: &mut [S]) {
    into(Op::new(a), Op::new(b), out, "convolve_into");
}

/// [`correlate`] into `out`, which must hold `len(a) + len(b) − 1` values (none when either is
/// empty): the same values, without allocating the result. Panics when `out` has another
/// length.
///
/// # Examples
///
/// ```
/// use a::math::fft::correlate_into;
///
/// let mut c = [0.0; 5];
/// correlate_into(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5], &mut c);
/// assert_eq!(c, [0.5, 2.0, 3.5, 3.0, 0.0]);
/// ```
#[track_caller]
pub fn correlate_into<S: Sample>(a: &[S], b: &[S], out: &mut [S]) {
    into(Op::new(a), Op::new(b).rev_conj(), out, "correlate_into");
}

#[track_caller]
fn into<S: Sample>(a: Op<'_, S>, b: Op<'_, S>, out: &mut [S], what: &str) {
    let len = if a.len() == 0 || b.len() == 0 {
        0
    } else {
        a.len() + b.len() - 1
    };
    assert_eq!(
        out.len(),
        len,
        "{what}: the output needs len(a) + len(b) − 1 values"
    );
    if len > 0 {
        // SAFETY: `out` holds the `len` values and does not overlap the operands, which are
        // shared borrows.
        unsafe { linear_into(a, b, out.as_mut_ptr()) }
    }
}

/// How a circular convolution is computed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Circular {
    Direct,
    /// Transforms of the length `n`.
    Fft,
    /// The linear convolution, folded.
    Linear,
}

/// The path of a circular convolution of the length `n ≥ 1`.
fn choose_circular<S: Sample>(n: usize) -> Circular {
    let c = S::costs();
    let direct = c.direct + c.copy * (3 * n) as f64 + c.mac * kernels::mac_scale() * (n * n) as f64;
    let mut best = (Circular::Direct, direct);
    let floor = c.plan_cost(n) + 3.0 * (c.call + c.fft * n as f64);
    if floor >= best.1 {
        return best.0;
    }
    let t = c.transform_cost(n, S::real());
    let f = c.plan_cost(n) + 3.0 * t + 3.0 * c.pass * n as f64;
    if f < best.1 {
        best = (Circular::Fft, f);
    }
    let (l, t) = c.best_len(2 * n - 1, S::real());
    let f = fft_cost(c, n, n, l, t) + c.pass * n as f64;
    if f < best.1 {
        best = (Circular::Linear, f);
    }
    best.0
}

/// The circular convolution of `a` and `b`, of the same length `n ≥ 1`.
fn circular<S: Sample>(a: &[S], b: &[S], flip: bool) -> Vec<S> {
    let n = a.len();
    // The values of b' = b, or b flipped (conj(b[(n − j) mod n])), written into `dst` in order or
    // reversed, without a copy of b'.
    let conj = |v| Op {
        v,
        rev: false,
        conj: true,
    };
    let write = |dst: &mut [S], reversed: bool| match (flip, reversed) {
        (false, false) => dst.copy_from_slice(b),
        (false, true) => Op::new(b).reversed().write(dst, 0),
        (true, false) => {
            dst[0] = b[0].conj();
            conj(&b[1..]).reversed().write(&mut dst[1..], 0);
        }
        (true, true) => {
            conj(&b[1..]).write(&mut dst[..n - 1], 0);
            dst[n - 1] = b[0].conj();
        }
    };
    let path = choose_circular::<S>(n);
    match path {
        // c[k] = Σ_t b[n − 1 − t]·x[k + t] with x = a[1..], a: x[j] = a[(j − (n − 1)) mod n].
        Circular::Direct => {
            let mut x = Vec::with_capacity(2 * n - 1 + PAD);
            x.extend_from_slice(&a[1..]);
            x.extend_from_slice(a);
            x.resize(2 * n - 1 + PAD, S::zero());
            let mut taps = vec![S::zero(); n];
            write(&mut taps, true);
            let mut c = Vec::with_capacity(n);
            // SAFETY: `x` holds `n + n − 1 + PAD` values, `taps` `n` and `c` room for `n`,
            // written before the length is set.
            unsafe {
                S::fir(x.as_ptr(), taps.as_ptr(), n, c.as_mut_ptr(), n);
                c.set_len(n);
            }
            c
        }
        Circular::Fft => {
            let plan = S::plan(n);
            let h = S::spectrum_len(n);
            let (mut fa, mut fb) = (vec![Complex::ZERO; h], vec![Complex::ZERO; h]);
            let mut x = vec![S::zero(); n];
            S::forward(&plan, a, &mut fa);
            write(&mut x, false);
            S::forward(&plan, &x, &mut fb);
            let scale = recip::<S::Real>(n);
            cmul(&mut fa, &fb, scale);
            S::inverse(&plan, &fa, &mut x);
            x
        }
        Circular::Linear => {
            let (l, _) = S::costs().best_len(2 * n - 1, S::real());
            let flipped;
            let b = if flip {
                let mut f = vec![S::zero(); n];
                write(&mut f, false);
                flipped = f;
                &flipped[..]
            } else {
                b
            };
            let mut c = by_fft(Op::new(a), Op::new(b), l);
            let (lo, hi) = c.split_at_mut(n);
            for (x, &y) in lo.iter_mut().zip(hi.iter()) {
                *x = *x + y;
            }
            c.truncate(n);
            c
        }
    }
}

/// The circular convolution of `a` and `b` of the same length `n`:
/// `c[k] = Σ a[j]·b[(k − j) mod n]`. Panics when the lengths differ.
///
/// Computed directly for short lengths, by transforms of the length `n` when its prime factors
/// are small, and otherwise (a large prime factor, as for a prime `n`) through the linear
/// convolution by transforms of a smooth length of at least `2n − 1`, folded.
///
/// # Examples
///
/// ```
/// use a::math::fft::circular_convolve;
///
/// assert_eq!(circular_convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0]), vec![3.0, 1.0, 2.0]);
/// ```
#[track_caller]
pub fn circular_convolve<S: Sample>(a: &[S], b: &[S]) -> Vec<S> {
    assert_eq!(a.len(), b.len(), "circular_convolve: the lengths differ");
    if a.is_empty() {
        return Vec::new();
    }
    circular(a, b, false)
}

/// The circular cross-correlation of `a` and `b` of the same length `n`:
/// `c[k] = Σ a[(j + k) mod n]·conj(b[j])`. Panics when the lengths differ.
///
/// # Examples
///
/// ```
/// use a::math::fft::circular_correlate;
///
/// assert_eq!(circular_correlate(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0]), vec![2.0, 3.0, 1.0]);
/// ```
#[track_caller]
pub fn circular_correlate<S: Sample>(a: &[S], b: &[S]) -> Vec<S> {
    assert_eq!(a.len(), b.len(), "circular_correlate: the lengths differ");
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }
    // c = a ⊛ b' with b'[j] = conj(b[(−j) mod n]): b[0] then b reversed, conjugated.
    circular(a, b, true)
}
