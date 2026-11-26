//! Two and three dimensions.

use crate::complex::join_notes;
use crate::fftw::{self, Plan};
use crate::prec::*;
use crate::race::Race;
use crate::timing::Cfg;
use crate::util::{ABuf, Out, Rng, check};
use crate::vdsp::{self, Split};
use a::math::fft::{Fft2, Fft3};

pub const D2: &[(usize, usize)] = &[
    (16, 16),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (480, 640),
    (1000, 1000),
    (4096, 64),
    (64, 4096),
];
pub const D2_INV: &[(usize, usize)] = &[(64, 64), (1024, 1024), (480, 640)];

pub const D3: &[[usize; 3]] = &[
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64],
    [128, 128, 128],
    [256, 256, 256],
    [100, 100, 100],
    [8, 256, 256],
    [256, 256, 8],
];
pub const D3_INV: &[[usize; 3]] = &[[16, 16, 16], [64, 64, 64]];

pub fn d2<T: P>(o: &mut Out, cfg: &Cfg, rows: usize, cols: usize, inv: bool) {
    let n = rows * cols;
    let x0 = cvec::<T>(&Rng::new(n as u64 + 41).vec(2 * n));
    let tl = tol::<T>(n);
    let plan = Fft2::<T>::new(rows, cols);
    let go = |b: &mut [C<T>]| if inv { plan.inverse(b) } else { plan.forward(b) };
    let mut b = x0.clone();
    go(&mut b);
    let want = flat(&b);
    let what = format!("Fft2 {rows}x{cols}");
    let sign = if inv { fftw::BACKWARD } else { fftw::FORWARD };
    let mut fb = ABuf::<C<T>>::new(n);
    let fp = Plan::<T>::wrap(unsafe {
        T::dft_2d(rows, cols, cp(&mut fb), cp(&mut fb), sign, fftw::MEASURE)
    })
    .unwrap();
    fb.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fb), tl).1;
    let dir = if inv { vdsp::INV } else { vdsp::FWD };
    let (lr, lc) = (rows.trailing_zeros() as usize, cols.trailing_zeros() as usize);
    let zs = if rows.is_power_of_two() && cols.is_power_of_two() {
        unsafe { T::fft_setup(lr.max(lc)) }
    } else {
        std::ptr::null_mut()
    };
    let (mut zr, mut zi) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
    let mut znote = String::new();
    if !zs.is_null() {
        split_into(&x0, &mut zr, &mut zi);
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        unsafe { T::fft2d_zip(zs, &sp, lc, lr, dir) };
        znote = check(&what, "vDSP fft2d_zip", &want, &join(&zr, &zi), tl).1;
    }
    let mut ob = ABuf::<C<T>>::from(&x0);
    let mut race = Race::new();
    let us = race.ours(if inv { "Fft2::inverse" } else { "Fft2::forward" }, || go(&mut ob));
    let fpr = &fp;
    race.reference("FFTW", &join_notes("plan_dft_2d, FFTW_MEASURE, in place", &fnote), us, move || {
        fpr.exec()
    });
    if !zs.is_null() {
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        race.reference(
            "vDSP fft2d_zip",
            &join_notes("split complex (layout differs), in place", &znote),
            us,
            move || unsafe { T::fft2d_zip(zs, &sp, lc, lr, dir) },
        );
    }
    let group = if inv { "nd/2d-inverse" } else { "nd/2d" };
    race.finish(o, cfg, group, &format!("T={} rows={rows} cols={cols}", T::NAME), 1.0);
    unsafe { T::fft_destroy(zs) };
}

pub fn d3<T: P>(o: &mut Out, cfg: &Cfg, s: [usize; 3], inv: bool) {
    let n = s[0] * s[1] * s[2];
    let x0 = cvec::<T>(&Rng::new(n as u64 + 43).vec(2 * n));
    let tl = tol::<T>(n);
    let plan = Fft3::<T>::new(s[0], s[1], s[2]);
    let go = |b: &mut [C<T>]| if inv { plan.inverse(b) } else { plan.forward(b) };
    let mut b = x0.clone();
    go(&mut b);
    let want = flat(&b);
    let what = format!("Fft3 {s:?}");
    let sign = if inv { fftw::BACKWARD } else { fftw::FORWARD };
    let mut fb = ABuf::<C<T>>::new(n);
    let fp = Plan::<T>::wrap(unsafe { T::dft_3d(s, cp(&mut fb), cp(&mut fb), sign, fftw::MEASURE) })
        .unwrap();
    fb.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fb), tl).1;
    let mut ob = ABuf::<C<T>>::from(&x0);
    let mut race = Race::new();
    let us = race.ours(if inv { "Fft3::inverse" } else { "Fft3::forward" }, || go(&mut ob));
    let fpr = &fp;
    race.reference("FFTW", &join_notes("plan_dft_3d, FFTW_MEASURE, in place", &fnote), us, move || {
        fpr.exec()
    });
    let group = if inv { "nd/3d-inverse" } else { "nd/3d" };
    race.finish(
        o,
        cfg,
        group,
        &format!("T={} shape={}x{}x{}", T::NAME, s[0], s[1], s[2]),
        1.0,
    );
}

pub fn run<T: P>(o: &mut Out, cfg: &Cfg, _part: &str) {
    for &(r, c) in D2 {
        d2::<T>(o, cfg, r, c, false);
    }
    for &(r, c) in D2_INV {
        d2::<T>(o, cfg, r, c, true);
    }
    for &s in D3 {
        d3::<T>(o, cfg, s, false);
    }
    for &s in D3_INV {
        d3::<T>(o, cfg, s, true);
    }
}
