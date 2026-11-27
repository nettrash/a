//! Apple's Accelerate vDSP (macOS only). Elsewhere every setup is null, so its rows are skipped.

#[allow(unused_imports)]
use std::ffi::{c_int, c_long, c_ulong, c_void};

pub type Setup = *mut c_void;

#[repr(C)]
pub struct Split<T> {
    pub re: *mut T,
    pub im: *mut T,
}

pub const FWD: c_int = 1; // kFFTDirection_Forward, vDSP_DFT_FORWARD
pub const INV: c_int = -1;

#[cfg(target_os = "macos")]
#[allow(dead_code)]
mod ffi {
    use super::*;
    unsafe extern "C" {
        pub fn vDSP_create_fftsetup(log2n: c_ulong, radix: c_int) -> Setup;
        pub fn vDSP_create_fftsetupD(log2n: c_ulong, radix: c_int) -> Setup;
        pub fn vDSP_destroy_fftsetup(s: Setup);
        pub fn vDSP_destroy_fftsetupD(s: Setup);
        pub fn vDSP_fft_zip(s: Setup, c: *const Split<f32>, ic: c_long, log2n: c_ulong, dir: c_int);
        pub fn vDSP_fft_zipD(s: Setup, c: *const Split<f64>, ic: c_long, log2n: c_ulong, dir: c_int);
        pub fn vDSP_fftm_zip(s: Setup, c: *const Split<f32>, ic: c_long, im: c_long, log2n: c_ulong, m: c_ulong, dir: c_int);
        pub fn vDSP_fftm_zipD(s: Setup, c: *const Split<f64>, ic: c_long, im: c_long, log2n: c_ulong, m: c_ulong, dir: c_int);
        pub fn vDSP_fft_zrip(s: Setup, c: *const Split<f32>, ic: c_long, log2n: c_ulong, dir: c_int);
        pub fn vDSP_fft_zripD(s: Setup, c: *const Split<f64>, ic: c_long, log2n: c_ulong, dir: c_int);
        pub fn vDSP_fftm_zrip(s: Setup, c: *const Split<f32>, ic: c_long, im: c_long, log2n: c_ulong, m: c_ulong, dir: c_int);
        pub fn vDSP_fftm_zripD(s: Setup, c: *const Split<f64>, ic: c_long, im: c_long, log2n: c_ulong, m: c_ulong, dir: c_int);
        pub fn vDSP_fft2d_zip(s: Setup, c: *const Split<f32>, ic0: c_long, ic1: c_long, l0: c_ulong, l1: c_ulong, dir: c_int);
        pub fn vDSP_fft2d_zipD(s: Setup, c: *const Split<f64>, ic0: c_long, ic1: c_long, l0: c_ulong, l1: c_ulong, dir: c_int);
        pub fn vDSP_DFT_zop_CreateSetup(prev: Setup, n: c_ulong, dir: c_int) -> Setup;
        pub fn vDSP_DFT_zop_CreateSetupD(prev: Setup, n: c_ulong, dir: c_int) -> Setup;
        pub fn vDSP_DFT_zrop_CreateSetup(prev: Setup, n: c_ulong, dir: c_int) -> Setup;
        pub fn vDSP_DFT_zrop_CreateSetupD(prev: Setup, n: c_ulong, dir: c_int) -> Setup;
        pub fn vDSP_DFT_Execute(s: Setup, ir: *const f32, ii: *const f32, or: *mut f32, oi: *mut f32);
        pub fn vDSP_DFT_ExecuteD(s: Setup, ir: *const f64, ii: *const f64, or: *mut f64, oi: *mut f64);
        pub fn vDSP_DFT_DestroySetup(s: Setup);
        pub fn vDSP_DFT_DestroySetupD(s: Setup);
        pub fn vDSP_DFT_Interleaved_CreateSetup(prev: Setup, n: c_ulong, dir: c_int, real: bool) -> Setup;
        pub fn vDSP_DFT_Interleaved_CreateSetupD(prev: Setup, n: c_ulong, dir: c_int, real: bool) -> Setup;
        pub fn vDSP_DFT_Interleaved_Execute(s: Setup, i: *const f32, o: *mut f32);
        pub fn vDSP_DFT_Interleaved_ExecuteD(s: Setup, i: *const f64, o: *mut f64);
        pub fn vDSP_DFT_Interleaved_DestroySetup(s: Setup);
        pub fn vDSP_DFT_Interleaved_DestroySetupD(s: Setup);
        pub fn vDSP_DCT_CreateSetup(prev: Setup, n: c_ulong, kind: c_int) -> Setup;
        pub fn vDSP_DCT_Execute(s: Setup, i: *const f32, o: *mut f32);
        pub fn vDSP_conv(a: *const f32, ia: c_long, f: *const f32, if_: c_long, c: *mut f32, ic: c_long, n: c_ulong, p: c_ulong);
        pub fn vDSP_convD(a: *const f64, ia: c_long, f: *const f64, if_: c_long, c: *mut f64, ic: c_long, n: c_ulong, p: c_ulong);
        pub fn vDSP_zconv(a: *const Split<f32>, ia: c_long, f: *const Split<f32>, if_: c_long, c: *const Split<f32>, ic: c_long, n: c_ulong, p: c_ulong);
        pub fn vDSP_zconvD(a: *const Split<f64>, ia: c_long, f: *const Split<f64>, if_: c_long, c: *const Split<f64>, ic: c_long, n: c_ulong, p: c_ulong);
    }
}

/// The vDSP functions of one precision. Every function is unsafe FFI with raw pointers.
#[allow(clippy::missing_safety_doc)]
pub trait Vdsp: Copy + 'static {
    unsafe fn fft_setup(log2n: usize) -> Setup;
    unsafe fn fft_destroy(s: Setup);
    unsafe fn fft_zip(s: Setup, c: &Split<Self>, log2n: usize, dir: c_int);
    unsafe fn fftm_zip(s: Setup, c: &Split<Self>, dist: usize, log2n: usize, m: usize, dir: c_int);
    unsafe fn fft_zrip(s: Setup, c: &Split<Self>, log2n: usize, dir: c_int);
    unsafe fn fftm_zrip(s: Setup, c: &Split<Self>, dist: usize, log2n: usize, m: usize, dir: c_int);
    unsafe fn fft2d_zip(s: Setup, c: &Split<Self>, log2cols: usize, log2rows: usize, dir: c_int);
    unsafe fn dft_zop(n: usize, dir: c_int) -> Setup;
    unsafe fn dft_zrop(n: usize, dir: c_int) -> Setup;
    unsafe fn dft_exec(s: Setup, ir: *const Self, ii: *const Self, or: *mut Self, oi: *mut Self);
    unsafe fn dft_destroy(s: Setup);
    unsafe fn il_setup(n: usize, dir: c_int, real: bool) -> Setup;
    unsafe fn il_exec(s: Setup, i: *const Self, o: *mut Self);
    unsafe fn il_destroy(s: Setup);
    unsafe fn dct_setup(n: usize, kind: c_int) -> Setup;
    unsafe fn dct_exec(s: Setup, i: *const Self, o: *mut Self);
    unsafe fn conv(a: *const Self, f: *const Self, if_: isize, c: *mut Self, n: usize, p: usize);
    unsafe fn zconv(a: &Split<Self>, f: &Split<Self>, if_: isize, c: &Split<Self>, n: usize, p: usize);
}

#[cfg(target_os = "macos")]
macro_rules! impl_vdsp {
    ($t:ty, $cs:ident, $ds:ident, $zip:ident, $zipm:ident, $zrip:ident, $zripm:ident, $z2d:ident,
     $zop:ident, $zrop:ident, $exec:ident, $dd:ident, $ils:ident, $ile:ident, $ild:ident,
     $conv:ident, $zconv:ident, $dct:expr) => {
        impl Vdsp for $t {
            unsafe fn fft_setup(log2n: usize) -> Setup { unsafe { ffi::$cs(log2n as c_ulong, 0) } }
            unsafe fn fft_destroy(s: Setup) { unsafe { ffi::$ds(s) } }
            unsafe fn fft_zip(s: Setup, c: &Split<Self>, log2n: usize, dir: c_int) {
                unsafe { ffi::$zip(s, c, 1, log2n as c_ulong, dir) }
            }
            unsafe fn fftm_zip(s: Setup, c: &Split<Self>, dist: usize, log2n: usize, m: usize, dir: c_int) {
                unsafe { ffi::$zipm(s, c, 1, dist as c_long, log2n as c_ulong, m as c_ulong, dir) }
            }
            unsafe fn fft_zrip(s: Setup, c: &Split<Self>, log2n: usize, dir: c_int) {
                unsafe { ffi::$zrip(s, c, 1, log2n as c_ulong, dir) }
            }
            unsafe fn fftm_zrip(s: Setup, c: &Split<Self>, dist: usize, log2n: usize, m: usize, dir: c_int) {
                unsafe { ffi::$zripm(s, c, 1, dist as c_long, log2n as c_ulong, m as c_ulong, dir) }
            }
            unsafe fn fft2d_zip(s: Setup, c: &Split<Self>, log2cols: usize, log2rows: usize, dir: c_int) {
                unsafe { ffi::$z2d(s, c, 1, 0, log2cols as c_ulong, log2rows as c_ulong, dir) }
            }
            unsafe fn dft_zop(n: usize, dir: c_int) -> Setup {
                unsafe { ffi::$zop(std::ptr::null_mut(), n as c_ulong, dir) }
            }
            unsafe fn dft_zrop(n: usize, dir: c_int) -> Setup {
                unsafe { ffi::$zrop(std::ptr::null_mut(), n as c_ulong, dir) }
            }
            unsafe fn dft_exec(s: Setup, ir: *const Self, ii: *const Self, or: *mut Self, oi: *mut Self) {
                unsafe { ffi::$exec(s, ir, ii, or, oi) }
            }
            unsafe fn dft_destroy(s: Setup) { unsafe { ffi::$dd(s) } }
            unsafe fn il_setup(n: usize, dir: c_int, real: bool) -> Setup {
                unsafe { ffi::$ils(std::ptr::null_mut(), n as c_ulong, dir, real) }
            }
            unsafe fn il_exec(s: Setup, i: *const Self, o: *mut Self) { unsafe { ffi::$ile(s, i, o) } }
            unsafe fn il_destroy(s: Setup) { unsafe { ffi::$ild(s) } }
            unsafe fn dct_setup(n: usize, kind: c_int) -> Setup {
                let f: fn(usize, c_int) -> Setup = $dct;
                f(n, kind)
            }
            unsafe fn dct_exec(s: Setup, i: *const Self, o: *mut Self) {
                // Only reached with a non-null setup, which only `f32` makes.
                unsafe { ffi::vDSP_DCT_Execute(s, i as *const f32, o as *mut f32) }
            }
            unsafe fn conv(a: *const Self, f: *const Self, if_: isize, c: *mut Self, n: usize, p: usize) {
                unsafe { ffi::$conv(a, 1, f, if_ as c_long, c, 1, n as c_ulong, p as c_ulong) }
            }
            unsafe fn zconv(a: &Split<Self>, f: &Split<Self>, if_: isize, c: &Split<Self>, n: usize, p: usize) {
                unsafe { ffi::$zconv(a, 1, f, if_ as c_long, c, 1, n as c_ulong, p as c_ulong) }
            }
        }
    };
}

#[cfg(target_os = "macos")]
impl_vdsp!(f32, vDSP_create_fftsetup, vDSP_destroy_fftsetup, vDSP_fft_zip, vDSP_fftm_zip,
    vDSP_fft_zrip, vDSP_fftm_zrip, vDSP_fft2d_zip, vDSP_DFT_zop_CreateSetup,
    vDSP_DFT_zrop_CreateSetup, vDSP_DFT_Execute, vDSP_DFT_DestroySetup,
    vDSP_DFT_Interleaved_CreateSetup, vDSP_DFT_Interleaved_Execute,
    vDSP_DFT_Interleaved_DestroySetup, vDSP_conv, vDSP_zconv,
    |n, k| unsafe { ffi::vDSP_DCT_CreateSetup(std::ptr::null_mut(), n as c_ulong, k) });

#[cfg(target_os = "macos")]
impl_vdsp!(f64, vDSP_create_fftsetupD, vDSP_destroy_fftsetupD, vDSP_fft_zipD, vDSP_fftm_zipD,
    vDSP_fft_zripD, vDSP_fftm_zripD, vDSP_fft2d_zipD, vDSP_DFT_zop_CreateSetupD,
    vDSP_DFT_zrop_CreateSetupD, vDSP_DFT_ExecuteD, vDSP_DFT_DestroySetupD,
    vDSP_DFT_Interleaved_CreateSetupD, vDSP_DFT_Interleaved_ExecuteD,
    vDSP_DFT_Interleaved_DestroySetupD, vDSP_convD, vDSP_zconvD,
    |_, _| std::ptr::null_mut());

/// The DCT setups are destroyed by the DFT's destroy routine of `f32`.
pub unsafe fn dct_destroy(s: Setup) {
    #[cfg(target_os = "macos")]
    unsafe {
        ffi::vDSP_DFT_DestroySetup(s)
    }
    #[cfg(not(target_os = "macos"))]
    let _ = s;
}

#[cfg(not(target_os = "macos"))]
macro_rules! impl_none {
    ($t:ty) => {
        impl Vdsp for $t {
            unsafe fn fft_setup(_: usize) -> Setup { std::ptr::null_mut() }
            unsafe fn fft_destroy(_: Setup) {}
            unsafe fn fft_zip(_: Setup, _: &Split<Self>, _: usize, _: c_int) { unreachable!() }
            unsafe fn fftm_zip(_: Setup, _: &Split<Self>, _: usize, _: usize, _: usize, _: c_int) { unreachable!() }
            unsafe fn fft_zrip(_: Setup, _: &Split<Self>, _: usize, _: c_int) { unreachable!() }
            unsafe fn fftm_zrip(_: Setup, _: &Split<Self>, _: usize, _: usize, _: usize, _: c_int) { unreachable!() }
            unsafe fn fft2d_zip(_: Setup, _: &Split<Self>, _: usize, _: usize, _: c_int) { unreachable!() }
            unsafe fn dft_zop(_: usize, _: c_int) -> Setup { std::ptr::null_mut() }
            unsafe fn dft_zrop(_: usize, _: c_int) -> Setup { std::ptr::null_mut() }
            unsafe fn dft_exec(_: Setup, _: *const Self, _: *const Self, _: *mut Self, _: *mut Self) { unreachable!() }
            unsafe fn dft_destroy(_: Setup) {}
            unsafe fn il_setup(_: usize, _: c_int, _: bool) -> Setup { std::ptr::null_mut() }
            unsafe fn il_exec(_: Setup, _: *const Self, _: *mut Self) { unreachable!() }
            unsafe fn il_destroy(_: Setup) {}
            unsafe fn dct_setup(_: usize, _: c_int) -> Setup { std::ptr::null_mut() }
            unsafe fn dct_exec(_: Setup, _: *const Self, _: *mut Self) { unreachable!() }
            unsafe fn conv(_: *const Self, _: *const Self, _: isize, _: *mut Self, _: usize, _: usize) { unreachable!() }
            unsafe fn zconv(_: &Split<Self>, _: &Split<Self>, _: isize, _: &Split<Self>, _: usize, _: usize) { unreachable!() }
        }
    };
}

#[cfg(not(target_os = "macos"))]
impl_none!(f32);
#[cfg(not(target_os = "macos"))]
impl_none!(f64);

pub fn available() -> bool {
    cfg!(target_os = "macos")
}

/// A DFT setup of `n` made and destroyed (single or double precision), for the planning probe.
#[cfg(target_os = "macos")]
pub fn mix_setup(n: usize, double: bool) {
    // SAFETY: a setup made and destroyed at once.
    unsafe {
        if double {
            let s = ffi::vDSP_DFT_zop_CreateSetupD(core::ptr::null_mut(), n as _, FWD);
            ffi::vDSP_DFT_DestroySetupD(s);
        } else {
            let s = ffi::vDSP_DFT_zop_CreateSetup(core::ptr::null_mut(), n as _, FWD);
            ffi::vDSP_DFT_DestroySetup(s);
        }
    }
}

/// Without Accelerate there is no vDSP setup to make between plans: nothing.
#[cfg(not(target_os = "macos"))]
pub fn mix_setup(_n: usize, _double: bool) {}
