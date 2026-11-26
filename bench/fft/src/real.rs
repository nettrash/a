//! Real transforms: `RealFft::forward`/`inverse` (and `_with_scratch`), batches.

use crate::complex::join_notes;
use crate::fftw::{self, Plan};
use crate::prec::*;
use crate::race::Race;
use crate::timing::Cfg;
use crate::util::{ABuf, Out, Rng, check, rel};
use crate::vdsp::{self, Split};
use a::math::fft::RealFft;
use realfft::RealFftPlanner;

pub const SIZES: &[usize] = &[
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1048576, 96, 384,
    1536, 6144, 80, 640, 5120, 1000, 10000, 100000, 1000000, 6000, 30000, 8, 12, 15, 45, 135, 243,
    625, 2187, 3125, 2401, 16807, 19683, 177147, 1155, 15015, 99225, 101, 1009, 10007, 100003,
];

pub const SCRATCH: &[usize] = &[16, 64, 1024, 65536, 1000, 15, 101];

pub const BATCH: &[(usize, usize)] = &[
    (8, 2048),
    (16, 1024),
    (32, 512),
    (64, 256),
    (15, 1092),
    (60, 272),
    (100, 160),
    (1024, 64),
];

fn class(n: usize) -> &'static str {
    if n.is_power_of_two() {
        "pow2"
    } else if n % 2 == 0 {
        "even"
    } else {
        let mut m = n;
        for p in [3, 5, 7, 11] {
            while m % p == 0 {
                m /= p;
            }
        }
        if m == 1 { "odd-smooth" } else { "odd-other" }
    }
}

/// Unpacks vDSP's packed real spectrum (re[0] = X0, im[0] = X(n/2)) into `n/2 + 1` values.
fn unpack<T: P>(re: &[T], im: &[T], n: usize) -> Vec<f64> {
    let h = n / 2;
    let mut v = vec![0.0; 2 * (h + 1)];
    v[0] = re[0].to();
    v[2 * h] = im[0].to();
    for k in 1..h {
        v[2 * k] = re[k].to();
        v[2 * k + 1] = im[k].to();
    }
    v
}

/// Packs a spectrum of `n/2 + 1` values into vDSP's layout.
fn pack<T: P>(x: &[C<T>], re: &mut [T], im: &mut [T], n: usize) {
    let h = n / 2;
    re[0] = x[0].re;
    im[0] = x[h].re;
    for k in 1..h {
        re[k] = x[k].re;
        im[k] = x[k].im;
    }
}

/// The best constant `s` with `theirs·s ≈ ours`, and whether it fits to `tol`.
pub fn scaled(ours: &[f64], theirs: &[f64], tol: f64) -> (bool, f64) {
    let (mut d, mut t) = (0.0, 0.0);
    for (a, b) in ours.iter().zip(theirs) {
        d += a * b;
        t += b * b;
    }
    let s = if t == 0.0 { 1.0 } else { d / t };
    let v: Vec<f64> = theirs.iter().map(|b| b * s).collect();
    let e = rel(ours, &v);
    (e <= tol, s)
}

pub fn scale_note(what: &str, name: &str, ok: bool, s: f64) -> String {
    if !ok {
        eprintln!("CHECK FAILED: {what} vs {name} (scaled by {s})");
        return "CHECK FAILED".into();
    }
    if (s - 1.0).abs() < 1e-6 {
        String::new()
    } else {
        format!("its output = {} x ours (a convention)", crate::util::fmt_ratio(1.0 / s))
    }
}

pub fn forward<T: P>(o: &mut Out, cfg: &Cfg, n: usize, with_scratch: bool) {
    let h = n / 2 + 1;
    let x0 = rvec::<T>(&Rng::new(n as u64 + 21).vec(n));
    let tl = tol::<T>(n);
    let fft = RealFft::<T>::new(n);
    let mut y = ABuf::<C<T>>::new(h);
    fft.forward(&x0, &mut y);
    let want = flat(&y);
    let what = format!("real forward n={n}");

    let mut fi = ABuf::<T>::new(n);
    let mut fo = ABuf::<C<T>>::new(h);
    let fp = Plan::<T>::wrap(unsafe { T::r2c_1d(n, fi.ptr(), cp(&mut fo), fftw::MEASURE) }).unwrap();
    fi.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fo), tl).1;
    let (mut pi, mut po) = (ABuf::<T>::new(n), ABuf::<C<T>>::new(h));
    let mut pnote = String::new();
    let pp = if !with_scratch && fftw::patient(n) {
        let (a, b) = (pi.ptr(), cp(&mut po));
        let pp = fftw::with_patient::<T, _>(|| {
            Plan::<T>::wrap(unsafe { T::r2c_1d(n, a, b, fftw::PATIENT) }).unwrap()
        });
        pi.copy_from_slice(&x0);
        pp.exec();
        pnote = check(&what, "FFTW PATIENT", &want, &flat(&po), tl).1;
        Some(pp)
    } else {
        None
    };

    let rf = RealFftPlanner::<T>::new().plan_fft_forward(n);
    let mut rs = vec![Nc::<T>::default(); rf.get_scratch_len()];
    let mut ri = ABuf::<T>::from(&x0);
    let mut ro = ABuf::<C<T>>::new(h);
    let _ = rf.process_with_scratch(&mut ri, nc(&mut ro), &mut rs);
    let rnote = check(&what, "realfft", &want, &flat(&ro), tl).1;

    // vDSP: zrip (powers of two) and DFT_zrop on the reals split into even and odd indices,
    // the interleaved real DFT on the reals as they are.
    let even = n % 2 == 0;
    let log2n = n.trailing_zeros() as usize;
    let zs = if n.is_power_of_two() && n >= 4 {
        unsafe { T::fft_setup(log2n) }
    } else {
        std::ptr::null_mut()
    };
    let m = n / 2;
    let (mut zr, mut zi) = (ABuf::<T>::new(m.max(1)), ABuf::<T>::new(m.max(1)));
    let mut znote = String::new();
    let fill = |re: &mut [T], im: &mut [T]| {
        for j in 0..m {
            re[j] = x0[2 * j];
            im[j] = x0[2 * j + 1];
        }
    };
    if !zs.is_null() {
        fill(&mut zr, &mut zi);
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        unsafe { T::fft_zrip(zs, &sp, log2n, vdsp::FWD) };
        let (ok, s) = scaled(&want, &unpack(&zr, &zi, n), tl);
        znote = scale_note(&what, "vDSP fft_zrip", ok, s);
    }
    let ds = if even { unsafe { T::dft_zrop(n, vdsp::FWD) } } else { std::ptr::null_mut() };
    let (mut dr, mut di, mut er, mut ei) = (
        ABuf::<T>::new(m.max(1)),
        ABuf::<T>::new(m.max(1)),
        ABuf::<T>::new(m.max(1)),
        ABuf::<T>::new(m.max(1)),
    );
    let mut dnote = String::new();
    if !ds.is_null() {
        fill(&mut dr, &mut di);
        unsafe { T::dft_exec(ds, dr.ptr(), di.ptr(), er.ptr(), ei.ptr()) };
        let (ok, s) = scaled(&want, &unpack(&er, &ei, n), tl);
        dnote = scale_note(&what, "vDSP DFT_zrop", ok, s);
    }
    let is = if even { unsafe { T::il_setup(m, vdsp::FWD, true) } } else { std::ptr::null_mut() };
    let mut ii = ABuf::<T>::from(&x0);
    let mut io = ABuf::<C<T>>::new(m.max(1));
    let mut inote = String::new();
    if !is.is_null() {
        unsafe { T::il_exec(is, ii.ptr(), cp(&mut io)) };
        let re: Vec<T> = io.iter().map(|z| z.re).collect();
        let im: Vec<T> = io.iter().map(|z| z.im).collect();
        let (ok, s) = scaled(&want, &unpack(&re, &im, n), tl);
        inote = scale_note(&what, "vDSP DFT interleaved r2c", ok, s);
    }

    let xo = ABuf::<T>::from(&x0);
    let mut sc = vec![C::<T>::ZERO; fft.scratch_len()];
    let mut race = Race::new();
    let us = if with_scratch {
        race.ours("RealFft::forward_with_scratch", || {
            fft.forward_with_scratch(&xo, &mut y, &mut sc)
        })
    } else {
        race.ours("RealFft::forward", || fft.forward(&xo, &mut y))
    };
    let fpr = &fp;
    race.reference("FFTW", &join_notes("r2c, FFTW_MEASURE, out of place", &fnote), us, move || {
        fpr.exec()
    });
    if let Some(pp) = &pp {
        race.reference(
            "FFTW PATIENT",
            &join_notes("r2c, FFTW_PATIENT, out of place", &pnote),
            us,
            move || pp.exec(),
        );
    }
    race.reference(
        "realfft",
        &join_notes("process_with_scratch (input used as scratch)", &rnote),
        us,
        || {
            let _ = rf.process_with_scratch(&mut ri, nc(&mut ro), &mut rs);
        },
    );
    if !with_scratch {
        if !zs.is_null() {
            let sp = Split { re: zr.ptr(), im: zi.ptr() };
            race.reference(
                "vDSP fft_zrip",
                &join_notes(
                    "in place on the reals pre-split even/odd (vDSP_ctoz not timed), packed output",
                    &znote,
                ),
                us,
                move || unsafe { T::fft_zrip(zs, &sp, log2n, vdsp::FWD) },
            );
        }
        if !ds.is_null() {
            let (p, q, r, s) = (dr.ptr(), di.ptr(), er.ptr(), ei.ptr());
            race.reference(
                "vDSP DFT_zrop",
                &join_notes("reals pre-split even/odd (not timed), packed output, out of place", &dnote),
                us,
                move || unsafe { T::dft_exec(ds, p, q, r, s) },
            );
        }
        if !is.is_null() {
            let (p, q) = (ii.ptr(), cp(&mut io));
            race.reference(
                "vDSP DFT interleaved r2c",
                &join_notes("reals as they are, packed output, out of place", &inote),
                us,
                move || unsafe { T::il_exec(is, p, q) },
            );
        }
    }
    let group = if with_scratch {
        "real/forward-scratch".to_string()
    } else {
        format!("real/forward-{}", class(n))
    };
    race.finish(o, cfg, &group, &format!("T={} n={n}", T::NAME), 1.0);
    unsafe {
        T::fft_destroy(zs);
        T::dft_destroy(ds);
        T::il_destroy(is);
    }
}

pub fn inverse<T: P>(o: &mut Out, cfg: &Cfg, n: usize, with_scratch: bool) {
    let h = n / 2 + 1;
    let x0 = rvec::<T>(&Rng::new(n as u64 + 23).vec(n));
    let tl = tol::<T>(n);
    let fft = RealFft::<T>::new(n);
    let mut spec = vec![C::<T>::ZERO; h];
    fft.forward(&x0, &mut spec);
    let mut y = ABuf::<T>::new(n);
    fft.inverse(&spec, &mut y);
    let want = flatr(&y);
    let what = format!("real inverse n={n}");

    let mut fi = ABuf::<C<T>>::new(h);
    let mut fo = ABuf::<T>::new(n);
    let fp = Plan::<T>::wrap(unsafe { T::c2r_1d(n, cp(&mut fi), fo.ptr(), fftw::MEASURE) }).unwrap();
    fi.copy_from_slice(&spec);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flatr(&fo), tl).1;
    fi.copy_from_slice(&spec);

    let rf = RealFftPlanner::<T>::new().plan_fft_inverse(n);
    let mut rs = vec![Nc::<T>::default(); rf.get_scratch_len()];
    let mut ri = ABuf::<C<T>>::from(&spec);
    let mut ro = ABuf::<T>::new(n);
    let _ = rf.process_with_scratch(nc(&mut ri), &mut ro, &mut rs);
    let rnote = check(&what, "realfft", &want, &flatr(&ro), tl).1;

    let even = n % 2 == 0;
    let m = n / 2;
    let log2n = n.trailing_zeros() as usize;
    let zs = if n.is_power_of_two() && n >= 4 {
        unsafe { T::fft_setup(log2n) }
    } else {
        std::ptr::null_mut()
    };
    let (mut zr, mut zi) = (ABuf::<T>::new(m.max(1)), ABuf::<T>::new(m.max(1)));
    let interleave = |re: &[T], im: &[T]| -> Vec<f64> {
        (0..m).flat_map(|j| [re[j].to(), im[j].to()]).collect()
    };
    let mut znote = String::new();
    if !zs.is_null() {
        pack(&spec, &mut zr, &mut zi, n);
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        unsafe { T::fft_zrip(zs, &sp, log2n, vdsp::INV) };
        let (ok, s) = scaled(&want, &interleave(&zr, &zi), tl);
        znote = scale_note(&what, "vDSP fft_zrip", ok, s);
    }
    let ds = if even { unsafe { T::dft_zrop(n, vdsp::INV) } } else { std::ptr::null_mut() };
    let (mut dr, mut di, mut er, mut ei) = (
        ABuf::<T>::new(m.max(1)),
        ABuf::<T>::new(m.max(1)),
        ABuf::<T>::new(m.max(1)),
        ABuf::<T>::new(m.max(1)),
    );
    let mut dnote = String::new();
    if !ds.is_null() {
        pack(&spec, &mut dr, &mut di, n);
        unsafe { T::dft_exec(ds, dr.ptr(), di.ptr(), er.ptr(), ei.ptr()) };
        let (ok, s) = scaled(&want, &interleave(&er, &ei), tl);
        dnote = scale_note(&what, "vDSP DFT_zrop", ok, s);
    }
    let is = if even { unsafe { T::il_setup(m, vdsp::INV, true) } } else { std::ptr::null_mut() };
    let mut ii = ABuf::<C<T>>::new(m.max(1));
    let mut io = ABuf::<T>::new(n.max(2));
    let mut inote = String::new();
    if !is.is_null() {
        let (mut re, mut im) = (vec![T::of(0.0); m], vec![T::of(0.0); m]);
        pack(&spec, &mut re, &mut im, n);
        for j in 0..m {
            ii[j] = C::new(re[j], im[j]);
        }
        unsafe { T::il_exec(is, cp(&mut ii), io.ptr()) };
        let (ok, s) = scaled(&want, &flatr(&io[..n]), tl);
        inote = scale_note(&what, "vDSP DFT interleaved c2r", ok, s);
    }

    let so = ABuf::<C<T>>::from(&spec);
    let mut sc = vec![C::<T>::ZERO; fft.scratch_len()];
    let mut race = Race::new();
    let us = if with_scratch {
        race.ours("RealFft::inverse_with_scratch", || {
            fft.inverse_with_scratch(&so, &mut y, &mut sc)
        })
    } else {
        race.ours("RealFft::inverse", || fft.inverse(&so, &mut y))
    };
    let fpr = &fp;
    race.reference(
        "FFTW",
        &join_notes("c2r, FFTW_MEASURE, out of place; FFTW's default lets c2r destroy its input", &fnote),
        us,
        move || fpr.exec(),
    );
    race.reference(
        "realfft",
        &join_notes("process_with_scratch (input used as scratch)", &rnote),
        us,
        || {
            let _ = rf.process_with_scratch(nc(&mut ri), &mut ro, &mut rs);
        },
    );
    if !with_scratch {
        if !zs.is_null() {
            let sp = Split { re: zr.ptr(), im: zi.ptr() };
            race.reference(
                "vDSP fft_zrip",
                &join_notes("packed split input, in place, output split even/odd", &znote),
                us,
                move || unsafe { T::fft_zrip(zs, &sp, log2n, vdsp::INV) },
            );
        }
        if !ds.is_null() {
            let (p, q, r, s) = (dr.ptr(), di.ptr(), er.ptr(), ei.ptr());
            race.reference(
                "vDSP DFT_zrop",
                &join_notes("packed split input, out of place, output split even/odd", &dnote),
                us,
                move || unsafe { T::dft_exec(ds, p, q, r, s) },
            );
        }
        if !is.is_null() {
            let (p, q) = (cp(&mut ii), io.ptr());
            race.reference(
                "vDSP DFT interleaved c2r",
                &join_notes("packed input, out of place", &inote),
                us,
                move || unsafe { T::il_exec(is, p, q) },
            );
        }
    }
    let group = if with_scratch {
        "real/inverse-scratch".to_string()
    } else {
        format!("real/inverse-{}", class(n))
    };
    race.finish(o, cfg, &group, &format!("T={} n={n}", T::NAME), 1.0);
    unsafe {
        T::fft_destroy(zs);
        T::dft_destroy(ds);
        T::il_destroy(is);
    }
}

pub fn batch<T: P>(o: &mut Out, cfg: &Cfg, n: usize, count: usize) {
    let h = n / 2 + 1;
    let x0 = rvec::<T>(&Rng::new(n as u64 + 29).vec(n * count));
    let tl = tol::<T>(n);
    let fft = RealFft::<T>::new(n);
    let mut y = ABuf::<C<T>>::new(h * count);
    fft.forward(&x0, &mut y);
    let want = flat(&y);
    let what = format!("real batch n={n}x{count}");
    let mut fi = ABuf::<T>::new(n * count);
    let mut fo = ABuf::<C<T>>::new(h * count);
    let fp = Plan::<T>::wrap(unsafe { T::many_r2c(n, count, fi.ptr(), cp(&mut fo), fftw::MEASURE) })
        .unwrap();
    fi.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fo), tl).1;
    let rf = RealFftPlanner::<T>::new().plan_fft_forward(n);
    let mut rs = vec![Nc::<T>::default(); rf.get_scratch_len()];
    let mut ri = ABuf::<T>::from(&x0);
    let mut ro = ABuf::<C<T>>::new(h * count);
    let mut rbatch = |ri: &mut [T], ro: &mut [C<T>], rs: &mut [Nc<T>]| {
        for (a, b) in ri.chunks_exact_mut(n).zip(ro.chunks_exact_mut(h)) {
            let _ = rf.process_with_scratch(a, nc(b), rs);
        }
    };
    rbatch(&mut ri, &mut ro, &mut rs);
    let rnote = check(&what, "realfft", &want, &flat(&ro), tl).1;
    let m = n / 2;
    let log2n = n.trailing_zeros() as usize;
    let zs = if n.is_power_of_two() && n >= 4 {
        unsafe { T::fft_setup(log2n) }
    } else {
        std::ptr::null_mut()
    };
    let (mut zr, mut zi) = (ABuf::<T>::new(m * count), ABuf::<T>::new(m * count));
    let mut znote = String::new();
    if !zs.is_null() {
        for j in 0..m * count {
            zr[j] = x0[2 * j];
            zi[j] = x0[2 * j + 1];
        }
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        unsafe { T::fftm_zrip(zs, &sp, m, log2n, count, vdsp::FWD) };
        let mut got = Vec::new();
        for c in 0..count {
            got.extend(unpack(&zr[c * m..], &zi[c * m..], n));
        }
        let (ok, s) = scaled(&want, &got, tl);
        znote = scale_note(&what, "vDSP fftm_zrip", ok, s);
    }
    let xo = ABuf::<T>::from(&x0);
    let mut race = Race::new();
    let us = race.ours("RealFft::forward (batch)", || fft.forward(&xo, &mut y));
    let fpr = &fp;
    race.reference(
        "FFTW",
        &join_notes("plan_many_dft_r2c, FFTW_MEASURE, out of place", &fnote),
        us,
        move || fpr.exec(),
    );
    race.reference(
        "realfft",
        &join_notes("process_with_scratch per sequence", &rnote),
        us,
        || rbatch(&mut ri, &mut ro, &mut rs),
    );
    if !zs.is_null() {
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        race.reference(
            "vDSP fftm_zrip",
            &join_notes("pre-split even/odd (not timed), packed output, in place", &znote),
            us,
            move || unsafe { T::fftm_zrip(zs, &sp, m, log2n, count, vdsp::FWD) },
        );
    }
    race.finish(
        o,
        cfg,
        "real/batch",
        &format!("T={} n={n} count={count} (ns per transform)", T::NAME),
        count as f64,
    );
    unsafe { T::fft_destroy(zs) };
}

pub fn run<T: P>(o: &mut Out, cfg: &Cfg, part: &str) {
    let all = part == "all";
    if all || part == "forward" {
        for &n in SIZES {
            forward::<T>(o, cfg, n, false);
        }
        for &n in SCRATCH {
            forward::<T>(o, cfg, n, true);
        }
    }
    if all || part == "inverse" {
        for &n in SIZES {
            inverse::<T>(o, cfg, n, false);
        }
        for &n in SCRATCH {
            inverse::<T>(o, cfg, n, true);
        }
    }
    if all || part == "batch" {
        for &(n, c) in BATCH {
            batch::<T>(o, cfg, n, c);
        }
    }
}
