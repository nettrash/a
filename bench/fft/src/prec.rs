//! The float types: everything a contender needs of `f32` and `f64`.

use crate::fftw::Fftw;
use crate::vdsp::Vdsp;
use a::math::complex::Complex;
use a::math::fft::FftFloat;

pub trait P:
    FftFloat + rustfft::FftNum + rustdct::DctNum + Fftw + Vdsp + Default + std::fmt::Debug
{
    const NAME: &'static str;
    const EPS: f64;
    fn of(x: f64) -> Self;
    fn to(self) -> f64;
}

impl P for f64 {
    const NAME: &'static str = "f64";
    const EPS: f64 = f64::EPSILON;
    fn of(x: f64) -> f64 {
        x
    }
    fn to(self) -> f64 {
        self
    }
}

impl P for f32 {
    const NAME: &'static str = "f32";
    const EPS: f64 = f32::EPSILON as f64;
    fn of(x: f64) -> f32 {
        x as f32
    }
    fn to(self) -> f64 {
        self as f64
    }
}

pub type C<T> = Complex<T>;
pub type Nc<T> = rustfft::num_complex::Complex<T>;

pub fn cvec<T: P>(v: &[f64]) -> Vec<C<T>> {
    v.chunks_exact(2)
        .map(|p| C::new(T::of(p[0]), T::of(p[1])))
        .collect()
}

pub fn rvec<T: P>(v: &[f64]) -> Vec<T> {
    v.iter().map(|&x| T::of(x)).collect()
}

pub fn flat<T: P>(x: &[C<T>]) -> Vec<f64> {
    x.iter().flat_map(|z| [z.re.to(), z.im.to()]).collect()
}

pub fn flatr<T: P>(x: &[T]) -> Vec<f64> {
    x.iter().map(|&v| v.to()).collect()
}

/// `Complex<T>` and `num_complex::Complex<T>` are both `#[repr(C)] {re, im}`.
pub fn nc<T: P>(x: &mut [C<T>]) -> &mut [Nc<T>] {
    // SAFETY: identical layout and alignment; the borrow is carried over.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut Nc<T>, x.len()) }
}

pub fn ncr<T: P>(x: &[C<T>]) -> &[Nc<T>] {
    // SAFETY: as `nc`.
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const Nc<T>, x.len()) }
}

/// Pointer to the first real part of a complex buffer, as FFTW and vDSP's interleaved API take.
pub fn cp<T: P>(x: &mut [C<T>]) -> *mut T {
    x.as_mut_ptr() as *mut T
}

/// The tolerance of a same-work check at length `n`.
pub fn tol<T: P>(n: usize) -> f64 {
    40.0 * T::EPS * ((n.max(2) as f64).log2() + 1.0)
}

/// Splits interleaved complex values into separate real and imaginary arrays.
pub fn split_into<T: P>(x: &[C<T>], re: &mut [T], im: &mut [T]) {
    for (i, z) in x.iter().enumerate() {
        re[i] = z.re;
        im[i] = z.im;
    }
}

pub fn join<T: P>(re: &[T], im: &[T]) -> Vec<f64> {
    re.iter()
        .zip(im)
        .flat_map(|(a, b)| [a.to(), b.to()])
        .collect()
}
