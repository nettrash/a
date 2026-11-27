//! The cost of planning: `Fft::new`, `RealFft::new`, `Dct::new`, `Fft2::new` against FFTW's
//! planner (ESTIMATE, and MEASURE where it is affordable; its wisdom forgotten before every
//! plan), fresh rustfft/realfft/rustdct planners and vDSP's setups. Each call makes and drops
//! one plan.

use crate::fftw;
use crate::prec::*;
use crate::race::Race;
use crate::timing::Cfg;
use crate::util::{ABuf, Out};
use a::math::fft::{Dct, DctKind, Fft, Fft2, RealFft};
use realfft::RealFftPlanner;
use rustdct::DctPlanner;
use rustfft::FftPlanner;
use std::hint::black_box;

pub const COMPLEX: &[usize] = &[16, 64, 1024, 4096, 65536, 1048576, 1000, 1009, 10007, 100003];
pub const REAL: &[usize] = &[16, 1024, 65536, 1048576, 1000, 1001, 1009];
pub const DCT: &[usize] = &[16, 1024, 65536, 1000];
pub const D2: &[(usize, usize)] = &[(64, 64), (1024, 1024)];

/// FFTW_MEASURE planning is timed up to this length.
const MEASURE_MAX: usize = 4096;

fn fftw_plan<T: P>(mk: impl Fn() -> fftw::RawPlan) {
    unsafe {
        T::forget();
        let p = mk();
        T::destroy(p);
    }
}

pub fn complex<T: P>(o: &mut Out, cfg: &Cfg, n: usize) {
    let mut fb = ABuf::<C<T>>::new(n);
    let p = cp(&mut fb);
    let mut race = Race::new();
    let us = race.ours("Fft::new", || {
        black_box(Fft::<T>::new(black_box(n)));
    });
    race.reference(
        "FFTW ESTIMATE",
        "plan_dft_1d in place, FFTW_ESTIMATE (its default heuristic plan), wisdom forgotten first",
        us,
        move || fftw_plan::<T>(|| unsafe { T::dft_1d(n, p, p, fftw::FORWARD, fftw::ESTIMATE) }),
    );
    if n <= MEASURE_MAX {
        race.reference(
            "FFTW MEASURE",
            "plan_dft_1d in place, FFTW_MEASURE (times candidate plans), wisdom forgotten first",
            us,
            move || fftw_plan::<T>(|| unsafe { T::dft_1d(n, p, p, fftw::FORWARD, fftw::MEASURE) }),
        );
    }
    race.reference("rustfft", "a fresh FftPlanner, plan_fft_forward", us, || {
        black_box(FftPlanner::<T>::new().plan_fft_forward(black_box(n)));
    });
    if n.is_power_of_two() && !unsafe { T::fft_setup(n.trailing_zeros() as usize) }.is_null() {
        race.reference(
            "vDSP fft_setup",
            "vDSP_create_fftsetup(log2 n) and destroy",
            us,
            || unsafe {
                let s = T::fft_setup(n.trailing_zeros() as usize);
                T::fft_destroy(s);
            },
        );
    }
    let ds = unsafe { T::dft_zop(n, 1) };
    if !ds.is_null() {
        unsafe { T::dft_destroy(ds) };
        race.reference("vDSP DFT split", "vDSP_DFT_zop_CreateSetup and destroy", us, || unsafe {
            T::dft_destroy(T::dft_zop(n, 1));
        });
    }
    let is = unsafe { T::il_setup(n, 1, false) };
    if !is.is_null() {
        unsafe { T::il_destroy(is) };
        race.reference(
            "vDSP DFT interleaved",
            "vDSP_DFT_Interleaved_CreateSetup and destroy",
            us,
            || unsafe { T::il_destroy(T::il_setup(n, 1, false)) },
        );
    }
    race.finish(o, cfg, "plan/complex", &format!("T={} n={n}", T::NAME), 1.0);
}

pub fn real<T: P>(o: &mut Out, cfg: &Cfg, n: usize) {
    let mut ri = ABuf::<T>::new(n);
    let mut co = ABuf::<C<T>>::new(n / 2 + 1);
    let (p, q) = (ri.ptr(), cp(&mut co));
    let mut race = Race::new();
    let us = race.ours("RealFft::new", || {
        black_box(RealFft::<T>::new(black_box(n)));
    });
    race.reference(
        "FFTW ESTIMATE",
        "plan_dft_r2c_1d out of place, FFTW_ESTIMATE, wisdom forgotten first",
        us,
        move || fftw_plan::<T>(|| unsafe { T::r2c_1d(n, p, q, fftw::ESTIMATE) }),
    );
    if n <= MEASURE_MAX {
        race.reference(
            "FFTW MEASURE",
            "plan_dft_r2c_1d out of place, FFTW_MEASURE, wisdom forgotten first",
            us,
            move || fftw_plan::<T>(|| unsafe { T::r2c_1d(n, p, q, fftw::MEASURE) }),
        );
    }
    race.reference("realfft", "a fresh RealFftPlanner, plan_fft_forward", us, || {
        black_box(RealFftPlanner::<T>::new().plan_fft_forward(black_box(n)));
    });
    // (vDSP on macOS only: elsewhere its stand-in makes no setup, nothing to time.)
    if n.is_power_of_two() && crate::vdsp::available() {
        race.reference(
            "vDSP fft_setup",
            "vDSP_create_fftsetup(log2 n) and destroy (zrip's setup)",
            us,
            || unsafe {
                let s = T::fft_setup(n.trailing_zeros() as usize);
                T::fft_destroy(s);
            },
        );
    }
    let ds = if n % 2 == 0 { unsafe { T::dft_zrop(n, 1) } } else { std::ptr::null_mut() };
    if !ds.is_null() {
        unsafe { T::dft_destroy(ds) };
        race.reference("vDSP DFT_zrop", "vDSP_DFT_zrop_CreateSetup and destroy", us, || unsafe {
            T::dft_destroy(T::dft_zrop(n, 1));
        });
    }
    race.finish(o, cfg, "plan/real", &format!("T={} n={n}", T::NAME), 1.0);
}

pub fn dct<T: P>(o: &mut Out, cfg: &Cfg, n: usize) {
    let (mut i, mut j) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
    let (p, q) = (i.ptr(), j.ptr());
    let mut race = Race::new();
    let us = race.ours("Dct::new (II)", || {
        black_box(Dct::<T>::new(black_box(n), DctKind::II));
    });
    race.reference(
        "FFTW ESTIMATE",
        "plan_r2r_1d REDFT10, FFTW_ESTIMATE, wisdom forgotten first",
        us,
        move || fftw_plan::<T>(|| unsafe { T::r2r_1d(n, p, q, fftw::REDFT10, fftw::ESTIMATE) }),
    );
    race.reference("rustdct", "a fresh DctPlanner, plan_dct2", us, || {
        black_box(DctPlanner::<T>::new().plan_dct2(black_box(n)));
    });
    let vs = unsafe { T::dct_setup(n, 2) };
    if !vs.is_null() {
        unsafe { crate::vdsp::dct_destroy(vs) };
        race.reference("vDSP DCT", "vDSP_DCT_CreateSetup(II) and destroy", us, || unsafe {
            crate::vdsp::dct_destroy(T::dct_setup(n, 2));
        });
    }
    race.finish(o, cfg, "plan/dct", &format!("T={} n={n}", T::NAME), 1.0);
}

pub fn d2<T: P>(o: &mut Out, cfg: &Cfg, r: usize, c: usize) {
    let mut fb = ABuf::<C<T>>::new(r * c);
    let p = cp(&mut fb);
    let mut race = Race::new();
    let us = race.ours("Fft2::new", || {
        black_box(Fft2::<T>::new(black_box(r), c));
    });
    race.reference(
        "FFTW ESTIMATE",
        "plan_dft_2d in place, FFTW_ESTIMATE, wisdom forgotten first",
        us,
        move || fftw_plan::<T>(|| unsafe { T::dft_2d(r, c, p, p, fftw::FORWARD, fftw::ESTIMATE) }),
    );
    race.finish(o, cfg, "plan/2d", &format!("T={} rows={r} cols={c}", T::NAME), 1.0);
}

pub fn run<T: P>(o: &mut Out, cfg: &Cfg, _part: &str) {
    for &n in COMPLEX {
        complex::<T>(o, cfg, n);
    }
    for &n in REAL {
        real::<T>(o, cfg, n);
    }
    for &n in DCT {
        dct::<T>(o, cfg, n);
    }
    for &(r, c) in D2 {
        d2::<T>(o, cfg, r, c);
    }
}
