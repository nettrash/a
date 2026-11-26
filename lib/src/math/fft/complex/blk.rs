//! Blocks of complex values, the unit the kernels compute on: one `Complex<T>` in portable Rust,
//! or a SIMD vector of `K` interleaved complex values on a target that has them. The butterflies
//! and passes are written once over the [`Blk`] trait and instantiated for each block type.
//!
//! Twiddle factors are kept expanded for the products: a factor `w = wr + wi·i` is stored as the
//! four reals `[wr, wr, −wi, wi]` ("compact" form, one factor for every lane), and a block of `K`
//! factors for `K` lanes as `[wr₀, wr₀, wr₁, wr₁, …]` followed by `[−wi₀, wi₀, −wi₁, wi₁, …]`
//! ("lane" form, `4K` reals). For `K = 1` the two forms agree.

use crate::math::complex::{Complex, Float};

/// A block of [`K`](Self::K) complex values of the type `T`.
///
/// The `INV` parameter of the rotations and twiddle products selects the direction: the
/// forward transform multiplies by `w₄ = −i` and by the twiddle `w`, the inverse by `w₄ = +i`
/// and by the conjugate of `w`.
pub(crate) trait Blk<T: Float>: Copy {
    /// The complex values of a block.
    const K: usize;
    /// An expanded twiddle factor (or `K` of them), ready for [`mul_tw`](Self::mul_tw).
    type Tw: Copy;

    /// The `K` values at `p`.
    ///
    /// # Safety
    /// `p` must be valid for reading `K` values.
    unsafe fn ld(p: *const Complex<T>) -> Self;
    /// Stores the `K` values at `p`.
    ///
    /// # Safety
    /// `p` must be valid for writing `K` values.
    unsafe fn st(self, p: *mut Complex<T>);
    /// The zero block.
    fn zero() -> Self;
    /// The complex conjugate, lane by lane.
    fn conj(self) -> Self;
    /// `self + b`.
    fn add(self, b: Self) -> Self;
    /// `self − b`.
    fn sub(self, b: Self) -> Self;
    /// `self · c` for a real `c`.
    fn mulr(self, c: T) -> Self;
    /// `self + b · c` for a real `c`.
    fn fmar(self, b: Self, c: T) -> Self;
    /// `self + w₄·b`.
    fn add_rot<const INV: bool>(self, b: Self) -> Self;
    /// `self − w₄·b`.
    fn sub_rot<const INV: bool>(self, b: Self) -> Self;
    /// [`add_rot`](Self::add_rot) by the CPU's complex addition where it has one (FCADD): for the
    /// written-out butterflies only, as inline assembly keeps the compiler from unrolling the
    /// loops of the others.
    #[inline(always)]
    fn add_rot_f<const INV: bool>(self, b: Self) -> Self {
        self.add_rot::<INV>(b)
    }
    /// [`sub_rot`](Self::sub_rot) as [`add_rot_f`](Self::add_rot_f).
    #[inline(always)]
    fn sub_rot_f<const INV: bool>(self, b: Self) -> Self {
        self.sub_rot::<INV>(b)
    }
    /// `w₄·self` alone (`zero().add_rot(self)`): a block may do it without the addition.
    #[inline(always)]
    fn rot<const INV: bool>(self) -> Self {
        Self::zero().add_rot::<INV>(self)
    }
    /// `−w₄·self` alone (`zero().sub_rot(self)`).
    #[inline(always)]
    fn rot_neg<const INV: bool>(self) -> Self {
        Self::zero().sub_rot::<INV>(self)
    }
    /// The twiddle at `p` in compact form, the same factor for every lane.
    ///
    /// # Safety
    /// `p` must be valid for reading 4 reals.
    unsafe fn tw_b(p: *const T) -> Self::Tw;
    /// The `K` twiddles at `p` in lane form.
    ///
    /// # Safety
    /// `p` must be valid for reading `4K` reals.
    unsafe fn tw_l(p: *const T) -> Self::Tw;
    /// The twiddle at `p` stored as `[wr, wi]`, the same factor for every lane.
    ///
    /// # Safety
    /// `p` must be valid for reading 2 reals.
    unsafe fn tw_c(p: *const T) -> Self::Tw;
    /// `self · w` (forward) or `self · conj(w)` (inverse), lane by lane.
    fn mul_tw<const INV: bool>(self, w: Self::Tw) -> Self;
    /// Loads `o[j]` with lane `l` from `p + l·stride + j`, for every `j` and lane `l`.
    ///
    /// # Safety
    /// Every place read must be valid.
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<T>, stride: usize);
    /// Stores lane `l` of `o[j]` at `p + l·stride + j`, for every `j` and lane `l`.
    ///
    /// # Safety
    /// Every place written must be valid.
    unsafe fn st_rows(o: &[Self], p: *mut Complex<T>, stride: usize);
}

/// One complex value in portable Rust.
#[derive(Clone, Copy)]
pub(crate) struct Sc<T>(pub Complex<T>);

impl<T: Float> Blk<T> for Sc<T> {
    const K: usize = 1;
    type Tw = (T, T);

    #[inline(always)]
    unsafe fn ld(p: *const Complex<T>) -> Self {
        // SAFETY: the caller's contract.
        Sc(unsafe { p.read() })
    }
    #[inline(always)]
    unsafe fn st(self, p: *mut Complex<T>) {
        // SAFETY: the caller's contract.
        unsafe { p.write(self.0) }
    }
    #[inline(always)]
    fn zero() -> Self {
        Sc(Complex::ZERO)
    }
    #[inline(always)]
    fn conj(self) -> Self {
        Sc(Complex::new(self.0.re, T::ZERO - self.0.im))
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        Sc(Complex::new(self.0.re + b.0.re, self.0.im + b.0.im))
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        Sc(Complex::new(self.0.re - b.0.re, self.0.im - b.0.im))
    }
    #[inline(always)]
    fn mulr(self, c: T) -> Self {
        Sc(Complex::new(self.0.re * c, self.0.im * c))
    }
    #[inline(always)]
    fn fmar(self, b: Self, c: T) -> Self {
        Sc(Complex::new(self.0.re + b.0.re * c, self.0.im + b.0.im * c))
    }
    #[inline(always)]
    fn add_rot<const INV: bool>(self, b: Self) -> Self {
        let (a, b) = (self.0, b.0);
        if INV {
            Sc(Complex::new(a.re - b.im, a.im + b.re))
        } else {
            Sc(Complex::new(a.re + b.im, a.im - b.re))
        }
    }
    #[inline(always)]
    fn sub_rot<const INV: bool>(self, b: Self) -> Self {
        let (a, b) = (self.0, b.0);
        if INV {
            Sc(Complex::new(a.re + b.im, a.im - b.re))
        } else {
            Sc(Complex::new(a.re - b.im, a.im + b.re))
        }
    }
    #[inline(always)]
    unsafe fn tw_b(p: *const T) -> (T, T) {
        // SAFETY: the caller's contract.
        unsafe { (p.read(), p.add(3).read()) }
    }
    #[inline(always)]
    unsafe fn tw_l(p: *const T) -> (T, T) {
        // SAFETY: the caller's contract; one lane, the forms agree.
        unsafe { Self::tw_b(p) }
    }
    #[inline(always)]
    unsafe fn tw_c(p: *const T) -> (T, T) {
        // SAFETY: the caller's contract.
        unsafe { (p.read(), p.add(1).read()) }
    }
    #[inline(always)]
    fn mul_tw<const INV: bool>(self, (wr, wi): (T, T)) -> Self {
        let a = self.0;
        if INV {
            Sc(Complex::new(a.re * wr + a.im * wi, a.im * wr - a.re * wi))
        } else {
            Sc(Complex::new(a.re * wr - a.im * wi, a.im * wr + a.re * wi))
        }
    }
    #[inline(always)]
    unsafe fn ld_rows(o: &mut [Self], p: *const Complex<T>, _stride: usize) {
        for (j, v) in o.iter_mut().enumerate() {
            // SAFETY: the caller's contract.
            *v = unsafe { Self::ld(p.add(j)) };
        }
    }
    #[inline(always)]
    unsafe fn st_rows(o: &[Self], p: *mut Complex<T>, _stride: usize) {
        for (j, v) in o.iter().enumerate() {
            // SAFETY: the caller's contract.
            unsafe { v.st(p.add(j)) }
        }
    }
}
