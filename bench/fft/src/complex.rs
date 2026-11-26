//! The complex transform: single in place, out of place, allocating, inverse, batches.

use crate::fftw::{self, Plan};
use crate::prec::*;
use crate::race::Race;
use crate::timing::Cfg;
use crate::util::{ABuf, Out, Rng, check};
use crate::vdsp::{self, Split};
use a::math::fft::Fft;
use rustfft::{FftDirection, FftPlanner};

pub const POW2: &[usize] = &[
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304,
];

pub const SMOOTH: &[usize] = &[
    6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304, 196608,
    393216, 786432, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920,
    163840, 327680, 655360, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 25,
    125, 625, 3125, 15625, 78125, 390625, 49, 343, 2401, 16807, 117649, 823543, 60, 360, 1000,
    2160, 6000, 10000, 15360, 30000, 48000, 100000, 216000, 1000000,
];

pub const PRIME: &[usize] = &[
    17, 101, 1009, 10007, 100003, 1000003, 202, 2018, 20014, 200006, 2000006,
];

pub const SMALL_ODD: &[usize] = &[3, 5, 7, 11, 13, 15, 18, 36, 100];

pub const INVERSE: &[usize] = &[
    16, 64, 1024, 4096, 65536, 1048576, 1000, 1000000, 1009, 10007, 100003,
];

pub const OOP: &[usize] = &[16, 64, 256, 1024, 4096, 65536, 1048576, 1000, 10007];

pub const ALLOC: &[usize] = &[16, 1024, 65536];

pub const BATCH: &[usize] = &[2, 4, 8, 16, 32, 64, 3, 5, 6, 10, 12, 15, 17, 128, 1024, 101];

fn class(n: usize) -> &'static str {
    if n.is_power_of_two() {
        "pow2"
    } else {
        let mut m = n;
        for p in [2, 3, 5, 7, 11, 13] {
            while m % p == 0 {
                m /= p;
            }
        }
        if m == 1 { "smooth" } else { "prime" }
    }
}

fn input<T: P>(n: usize, seed: u64) -> Vec<C<T>> {
    cvec::<T>(&Rng::new(seed).vec(2 * n))
}

/// One complex transform of length `n` in place (forward, or the inverse).
pub fn inplace<T: P>(o: &mut Out, cfg: &Cfg, group: &str, n: usize, inv: bool) {
    let x0 = input::<T>(n, n as u64);
    let tl = tol::<T>(n);
    let fft = Fft::<T>::new(n);
    let mut scratch = vec![C::<T>::ZERO; fft.scratch_len()];
    let run_ours = |f: &Fft<T>, b: &mut [C<T>], s: &mut [C<T>]| {
        if inv {
            f.inverse_with_scratch(b, s)
        } else {
            f.forward_with_scratch(b, s)
        }
    };
    let mut b = x0.clone();
    run_ours(&fft, &mut b, &mut scratch);
    let want = flat(&b);
    let what = format!("{} n={n}", if inv { "inverse" } else { "forward" });
    let sign = if inv { fftw::BACKWARD } else { fftw::FORWARD };

    // FFTW, in place.
    let mut fb = ABuf::<C<T>>::new(n);
    let fp = Plan::<T>::wrap(unsafe { T::dft_1d(n, cp(&mut fb), cp(&mut fb), sign, fftw::MEASURE) })
        .unwrap();
    fb.copy_from_slice(&x0);
    fp.exec();
    let (_, fnote) = check(&what, "FFTW", &want, &flat(&fb), tl);
    let pat = !inv && fftw::patient(n);
    let mut pb = ABuf::<C<T>>::new(n);
    let mut pnote = String::new();
    let pp = if pat {
        let p = cp(&mut pb);
        let pp = fftw::with_patient::<T, _>(|| {
            Plan::<T>::wrap(unsafe { T::dft_1d(n, p, p, sign, fftw::PATIENT) }).unwrap()
        });
        pb.copy_from_slice(&x0);
        pp.exec();
        pnote = check(&what, "FFTW PATIENT", &want, &flat(&pb), tl).1;
        Some(pp)
    } else {
        None
    };

    // rustfft, in place with scratch.
    let mut planner = FftPlanner::<T>::new();
    let rf = planner.plan_fft(n, if inv { FftDirection::Inverse } else { FftDirection::Forward });
    let mut rs = vec![Nc::<T>::default(); rf.get_inplace_scratch_len()];
    let mut rb = ABuf::<C<T>>::from(&x0);
    rf.process_with_scratch(nc(&mut rb), &mut rs);
    let (_, rnote) = check(&what, "rustfft", &want, &flat(&rb), tl);

    // vDSP: fft_zip (split, powers of two), DFT on split data, DFT on interleaved data.
    let dir = if inv { vdsp::INV } else { vdsp::FWD };
    let log2n = n.trailing_zeros() as usize;
    let zs = if n.is_power_of_two() && n >= 2 {
        unsafe { T::fft_setup(log2n) }
    } else {
        std::ptr::null_mut()
    };
    let (mut zr, mut zi) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
    let mut znote = String::new();
    if !zs.is_null() {
        split_into(&x0, &mut zr, &mut zi);
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        unsafe { T::fft_zip(zs, &sp, log2n, dir) };
        znote = check(&what, "vDSP fft_zip", &want, &join(&zr, &zi), tl).1;
    }
    let ds = unsafe { T::dft_zop(n, dir) };
    let (mut dr, mut di) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
    let mut dnote = String::new();
    if !ds.is_null() {
        split_into(&x0, &mut dr, &mut di);
        unsafe { T::dft_exec(ds, dr.ptr(), di.ptr(), dr.ptr(), di.ptr()) };
        dnote = check(&what, "vDSP DFT split", &want, &join(&dr, &di), tl).1;
    }
    let is = unsafe { T::il_setup(n, dir, false) };
    let mut ib = ABuf::<C<T>>::from(&x0);
    let mut inote = String::new();
    if !is.is_null() {
        unsafe { T::il_exec(is, cp(&mut ib), cp(&mut ib)) };
        inote = check(&what, "vDSP DFT interleaved", &want, &flat(&ib), tl).1;
    }

    let mut ob = ABuf::<C<T>>::from(&x0);
    let mut race = Race::new();
    let op = if inv { "Fft::inverse_with_scratch" } else { "Fft::forward_with_scratch" };
    // The call alone in the closure (the slices made once, the direction a copied flag), as
    // lean as the references' closures.
    let (obs, scs): (&mut [C<T>], &mut [C<T>]) = (&mut ob, &mut scratch);
    let fr = &fft;
    let us = race.ours(op, move || {
        if inv {
            fr.inverse_with_scratch(obs, scs)
        } else {
            fr.forward_with_scratch(obs, scs)
        }
    });
    let fpr = &fp;
    race.reference("FFTW", &join_notes("FFTW_MEASURE, in place", &fnote), us, move || fpr.exec());
    if let Some(pp) = &pp {
        race.reference(
            "FFTW PATIENT",
            &join_notes("FFTW_PATIENT, in place", &pnote),
            us,
            move || pp.exec(),
        );
    }
    race.reference("rustfft", &join_notes("process_with_scratch, in place", &rnote), us, || {
        rf.process_with_scratch(nc(&mut rb), &mut rs)
    });
    if !zs.is_null() {
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        race.reference(
            "vDSP fft_zip",
            &join_notes("split complex (layout differs), in place", &znote),
            us,
            move || unsafe { T::fft_zip(zs, &sp, log2n, dir) },
        );
    }
    if !ds.is_null() {
        let (p, q) = (dr.ptr(), di.ptr());
        race.reference(
            "vDSP DFT split",
            &join_notes("vDSP_DFT_Execute, split complex (layout differs), in place", &dnote),
            us,
            move || unsafe { T::dft_exec(ds, p, q, p, q) },
        );
    }
    if !is.is_null() {
        let p = cp(&mut ib);
        race.reference(
            "vDSP DFT interleaved",
            &join_notes("vDSP_DFT_Interleaved_Execute, in place", &inote),
            us,
            move || unsafe { T::il_exec(is, p, p) },
        );
    }
    race.finish(o, cfg, group, &format!("T={} n={n}", T::NAME), 1.0);
    unsafe {
        T::fft_destroy(zs);
        T::dft_destroy(ds);
        T::il_destroy(is);
    }
}

pub fn join_notes(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (_, true) => a.to_string(),
        (true, false) => b.to_string(),
        _ => format!("{a}; {b}"),
    }
}

/// One forward transform out of place.
pub fn oop<T: P>(o: &mut Out, cfg: &Cfg, n: usize) {
    let x0 = input::<T>(n, n as u64 + 7);
    let tl = tol::<T>(n);
    let fft = Fft::<T>::new(n);
    let mut y = ABuf::<C<T>>::new(n);
    fft.forward_to(&x0, &mut y);
    let want = flat(&y);
    let what = format!("forward_to n={n}");

    let (mut fi, mut fo) = (ABuf::<C<T>>::new(n), ABuf::<C<T>>::new(n));
    let fp = Plan::<T>::wrap(unsafe {
        T::dft_1d(n, cp(&mut fi), cp(&mut fo), fftw::FORWARD, fftw::MEASURE)
    })
    .unwrap();
    fi.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fo), tl).1;

    let mut planner = FftPlanner::<T>::new();
    let rf = planner.plan_fft_forward(n);
    let mut rs = vec![Nc::<T>::default(); rf.get_immutable_scratch_len()];
    let rin = ABuf::<C<T>>::from(&x0);
    let mut rout = ABuf::<C<T>>::new(n);
    rf.process_immutable_with_scratch(ncr(&rin), nc(&mut rout), &mut rs);
    let rnote = check(&what, "rustfft", &want, &flat(&rout), tl).1;
    let mut rs2 = vec![Nc::<T>::default(); rf.get_outofplace_scratch_len()];
    let mut rin2 = ABuf::<C<T>>::from(&x0);
    let mut rout2 = ABuf::<C<T>>::new(n);
    rf.process_outofplace_with_scratch(nc(&mut rin2), nc(&mut rout2), &mut rs2);
    let rnote2 = check(&what, "rustfft", &want, &flat(&rout2), tl).1;

    let ds = unsafe { T::dft_zop(n, vdsp::FWD) };
    let (mut dr, mut di, mut er, mut ei) =
        (ABuf::<T>::new(n), ABuf::<T>::new(n), ABuf::<T>::new(n), ABuf::<T>::new(n));
    let mut dnote = String::new();
    if !ds.is_null() {
        split_into(&x0, &mut dr, &mut di);
        unsafe { T::dft_exec(ds, dr.ptr(), di.ptr(), er.ptr(), ei.ptr()) };
        dnote = check(&what, "vDSP DFT split", &want, &join(&er, &ei), tl).1;
    }
    let is = unsafe { T::il_setup(n, vdsp::FWD, false) };
    let mut ii = ABuf::<C<T>>::from(&x0);
    let mut io = ABuf::<C<T>>::new(n);
    let mut inote = String::new();
    if !is.is_null() {
        unsafe { T::il_exec(is, cp(&mut ii), cp(&mut io)) };
        inote = check(&what, "vDSP DFT interleaved", &want, &flat(&io), tl).1;
    }

    let xo = ABuf::<C<T>>::from(&x0);
    let mut race = Race::new();
    let us = race.ours("Fft::forward_to", || fft.forward_to(&xo, &mut y));
    let fpr = &fp;
    race.reference("FFTW", &join_notes("FFTW_MEASURE, out of place", &fnote), us, move || {
        fpr.exec()
    });
    race.reference(
        "rustfft",
        &join_notes("process_immutable_with_scratch (input kept)", &rnote),
        us,
        || rf.process_immutable_with_scratch(ncr(&rin), nc(&mut rout), &mut rs),
    );
    race.reference(
        "rustfft (destroys input)",
        &join_notes("process_outofplace_with_scratch: input used as scratch, a weaker contract", &rnote2),
        us,
        || rf.process_outofplace_with_scratch(nc(&mut rin2), nc(&mut rout2), &mut rs2),
    );
    if !ds.is_null() {
        let (p, q, r, s) = (dr.ptr(), di.ptr(), er.ptr(), ei.ptr());
        race.reference(
            "vDSP DFT split",
            &join_notes("split complex (layout differs), out of place", &dnote),
            us,
            move || unsafe { T::dft_exec(ds, p, q, r, s) },
        );
    }
    if !is.is_null() {
        let (p, q) = (cp(&mut ii), cp(&mut io));
        race.reference(
            "vDSP DFT interleaved",
            &join_notes("out of place", &inote),
            us,
            move || unsafe { T::il_exec(is, p, q) },
        );
    }
    race.finish(o, cfg, "complex/out-of-place", &format!("T={} n={n}", T::NAME), 1.0);
    unsafe {
        T::dft_destroy(ds);
        T::il_destroy(is);
    }
}

/// The allocating `forward` against FFTW in place and rustfft's allocating `process`.
pub fn alloc<T: P>(o: &mut Out, cfg: &Cfg, n: usize) {
    let x0 = input::<T>(n, n as u64 + 11);
    let tl = tol::<T>(n);
    let fft = Fft::<T>::new(n);
    let mut b = x0.clone();
    fft.forward(&mut b);
    let want = flat(&b);
    let what = format!("forward n={n}");
    let mut fb = ABuf::<C<T>>::new(n);
    let fp = Plan::<T>::wrap(unsafe {
        T::dft_1d(n, cp(&mut fb), cp(&mut fb), fftw::FORWARD, fftw::MEASURE)
    })
    .unwrap();
    fb.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fb), tl).1;
    let rf = FftPlanner::<T>::new().plan_fft_forward(n);
    let mut rb = ABuf::<C<T>>::from(&x0);
    rf.process(nc(&mut rb));
    let rnote = check(&what, "rustfft", &want, &flat(&rb), tl).1;
    let mut ob = ABuf::<C<T>>::from(&x0);
    let mut race = Race::new();
    let us = race.ours("Fft::forward (allocating)", || fft.forward(&mut ob));
    let fpr = &fp;
    race.reference(
        "FFTW",
        &join_notes("FFTW_MEASURE, in place (plan owns no per-call memory)", &fnote),
        us,
        move || fpr.exec(),
    );
    race.reference(
        "rustfft",
        &join_notes("process (allocates its scratch each call)", &rnote),
        us,
        || rf.process(nc(&mut rb)),
    );
    race.finish(o, cfg, "complex/allocating", &format!("T={} n={n}", T::NAME), 1.0);
}

/// A batch of `count` transforms of length `n` in one call; ns per transform.
pub fn batch<T: P>(o: &mut Out, cfg: &Cfg, n: usize, count: usize) {
    let len = n * count;
    let x0 = input::<T>(len, n as u64 + 13);
    let tl = tol::<T>(n);
    let fft = Fft::<T>::new(n);
    let mut b = x0.clone();
    fft.forward(&mut b);
    let want = flat(&b);
    let what = format!("batch n={n}x{count}");
    let mut fb = ABuf::<C<T>>::new(len);
    let fp = Plan::<T>::wrap(unsafe {
        T::many_dft(n, count, cp(&mut fb), cp(&mut fb), fftw::FORWARD, fftw::MEASURE)
    })
    .unwrap();
    fb.copy_from_slice(&x0);
    fp.exec();
    let fnote = check(&what, "FFTW", &want, &flat(&fb), tl).1;
    let rf = FftPlanner::<T>::new().plan_fft_forward(n);
    let mut rs = vec![Nc::<T>::default(); rf.get_inplace_scratch_len()];
    let mut rb = ABuf::<C<T>>::from(&x0);
    rf.process_with_scratch(nc(&mut rb), &mut rs);
    let rnote = check(&what, "rustfft", &want, &flat(&rb), tl).1;
    let log2n = n.trailing_zeros() as usize;
    let zs = if n.is_power_of_two() && n >= 2 {
        unsafe { T::fft_setup(log2n) }
    } else {
        std::ptr::null_mut()
    };
    let (mut zr, mut zi) = (ABuf::<T>::new(len), ABuf::<T>::new(len));
    let mut znote = String::new();
    if !zs.is_null() {
        split_into(&x0, &mut zr, &mut zi);
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        unsafe { T::fftm_zip(zs, &sp, n, log2n, count, vdsp::FWD) };
        znote = check(&what, "vDSP fftm_zip", &want, &join(&zr, &zi), tl).1;
    }
    let mut ob = ABuf::<C<T>>::from(&x0);
    let mut race = Race::new();
    let us = race.ours("Fft::forward (batch)", || fft.forward(&mut ob));
    let fpr = &fp;
    race.reference("FFTW", &join_notes("plan_many_dft, FFTW_MEASURE, in place", &fnote), us, move || {
        fpr.exec()
    });
    race.reference(
        "rustfft",
        &join_notes("process_with_scratch on the whole buffer", &rnote),
        us,
        || rf.process_with_scratch(nc(&mut rb), &mut rs),
    );
    if !zs.is_null() {
        let sp = Split { re: zr.ptr(), im: zi.ptr() };
        race.reference(
            "vDSP fftm_zip",
            &join_notes("split complex (layout differs), in place", &znote),
            us,
            move || unsafe { T::fftm_zip(zs, &sp, n, log2n, count, vdsp::FWD) },
        );
    }
    race.finish(
        o,
        cfg,
        "complex/batch",
        &format!("T={} n={n} count={count} (ns per transform)", T::NAME),
        count as f64,
    );
    unsafe { T::fft_destroy(zs) };
}

pub fn run<T: P>(o: &mut Out, cfg: &Cfg, part: &str) {
    let all = part == "all";
    if all || part == "pow2" {
        for &n in POW2 {
            inplace::<T>(o, cfg, "complex/pow2", n, false);
        }
    }
    if all || part == "smooth" {
        for &n in SMALL_ODD.iter().chain(SMOOTH) {
            inplace::<T>(o, cfg, &format!("complex/{}", class(n)), n, false);
        }
    }
    if all || part == "prime" {
        for &n in PRIME {
            inplace::<T>(o, cfg, "complex/prime", n, false);
        }
    }
    if all || part == "inverse" {
        for &n in INVERSE {
            inplace::<T>(o, cfg, "complex/inverse", n, true);
        }
    }
    if all || part == "oop" {
        for &n in OOP {
            oop::<T>(o, cfg, n);
        }
        for &n in ALLOC {
            alloc::<T>(o, cfg, n);
        }
    }
    if all || part == "batch" {
        for &n in BATCH {
            batch::<T>(o, cfg, n, 1000);
        }
    }
}
