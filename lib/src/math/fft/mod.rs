//! Fast Fourier transforms of `f32`, `f64`, `Complex<f32>` and `Complex<f64>` sequences (see
//! [`Complex`](crate::math::complex::Complex)).
//!
//! A transform is planned once for its length and applied as often as needed: the plan holds the
//! factorisation, the algorithm and the tables of twiddle factors, never changes and may be shared
//! between threads. Every `forward` and `inverse` also takes several sequences of the planned
//! length one after the other in one slice, and transforms each of them.
//!
//! * [`Fft`]: the discrete Fourier transform of complex sequences of any length,
//!   `X[k] = Σ x[j]·e^(−2πi·jk/n)` forward and `x[j] = Σ X[k]·e^(2πi·jk/n)` inverse, in place or
//!   out of place. As in FFTW, the inverse is not scaled: `inverse(forward(x)) = n·x`.
//! * [`RealFft`]: the transform of real sequences, giving the `n/2 + 1` values `X[0..=n/2]` that
//!   determine the rest (`X[n − k]` is the conjugate of `X[k]`), and its inverse.
//! * [`Fft2`] and [`Fft3`]: the transforms of complex arrays of two and three dimensions, stored
//!   with the last index varying fastest.
//! * [`Dct`] and [`Dst`]: the discrete cosine and sine transforms of types I to IV, as FFTW's
//!   `REDFT` and `RODFT` kinds define them.
//! * [`convolve`], [`correlate`], [`circular_convolve`] and [`circular_correlate`]: linear and
//!   circular convolutions and correlations of real or complex sequences, computed directly or by
//!   transforms, whichever is faster for the lengths.
//!
//! # Examples
//!
//! ```
//! use a::math::complex::Complex;
//! use a::math::fft::Fft;
//!
//! let fft = Fft::<f64>::new(4);
//! let mut x = [1.0, 2.0, 3.0, 4.0].map(|re| Complex::new(re, 0.0));
//! fft.forward(&mut x);
//! assert_eq!(x[0], Complex::new(10.0, 0.0));
//! assert_eq!(x[1], Complex::new(-2.0, 2.0));
//! fft.inverse(&mut x);
//! assert_eq!(x[3], Complex::new(16.0, 0.0)); // n·x[3], unscaled
//! ```

mod complex;
mod conv;
mod dct;
mod nd;
mod plan;
mod real;
mod twiddle;

pub use conv::{
    Sample, circular_convolve, circular_correlate, convolve, convolve_into, correlate,
    correlate_into,
};
pub use dct::{Dct, DctKind, Dst, DstKind};
pub use nd::{Fft2, Fft3};
pub use plan::Fft;
pub use real::RealFft;

use crate::math::complex::Float;

pub(crate) mod sealed {
    /// The kernels of a float type; implemented for `f32` and `f64` only.
    pub trait FftFloat: super::complex::Kernels {}
    impl FftFloat for f32 {}
    impl FftFloat for f64 {}
}

/// `f32` or `f64`: the float types the transforms are planned for. The trait is sealed.
pub trait FftFloat: Float + sealed::FftFloat {}

impl FftFloat for f32 {}
impl FftFloat for f64 {}

/// The most values of working memory taken on the stack rather than from the heap.
pub(crate) const STACK_WORK: usize = 512;

/// Calls `f` with `len` values of zeroed working memory: on the stack when there are few.
#[inline]
pub(crate) fn with_work<T: FftFloat, R>(
    len: usize,
    f: impl FnOnce(&mut [crate::math::complex::Complex<T>]) -> R,
) -> R {
    if len <= STACK_WORK {
        let mut buf = [const { core::mem::MaybeUninit::<crate::math::complex::Complex<T>>::uninit() };
            STACK_WORK];
        for v in &mut buf[..len] {
            v.write(crate::math::complex::Complex::ZERO);
        }
        // SAFETY: the first `len` values were initialised.
        let work = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), len) };
        f(work)
    } else {
        f(&mut vec![crate::math::complex::Complex::ZERO; len])
    }
}

/// Calls `f` with a pointer to `len` values of uninitialised working memory (on the stack when
/// there are few); `f` must write every value before reading it.
#[inline]
pub(crate) fn with_raw<T: FftFloat, R>(
    len: usize,
    f: impl FnOnce(*mut crate::math::complex::Complex<T>) -> R,
) -> R {
    if len <= STACK_WORK {
        let mut buf = [const { core::mem::MaybeUninit::<crate::math::complex::Complex<T>>::uninit() };
            STACK_WORK];
        f(buf.as_mut_ptr().cast())
    } else {
        let mut v: Vec<crate::math::complex::Complex<T>> = Vec::with_capacity(len);
        f(v.as_mut_ptr())
    }
}

/// Panics unless `len` is a multiple of `n` (any length for `n = 0` must be 0).
#[inline]
#[track_caller]
pub(crate) fn check_batch(len: usize, n: usize, what: &str) {
    if n == 0 {
        assert!(
            len == 0,
            "{what}: data of length {len} for a transform of length 0"
        );
    } else {
        assert!(
            len.is_multiple_of(n),
            "{what}: data of length {len} is not a multiple of the length {n}"
        );
    }
}
