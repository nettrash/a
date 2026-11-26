//! fftbench: the scoreboard of `a::math::fft` against FFTW, rustfft/realfft/rustdct and vDSP.
//!
//! Usage: fftbench header
//!        fftbench <group> <type> [part]
//!   group complex (parts pow2 smooth prime inverse oop batch), real (forward inverse batch),
//!         dct (dct dst), nd, conv (conv corr circ), plan, sanity
//!   type  f32 | f64 (conv: f32 | f64 | c32 | c64)
//! Rows go to stdout (TSV without header), progress and check failures to stderr.

mod complex;
mod conv;
mod dct;
mod fftw;
mod fuzz;
mod nd;
mod plan;
mod prec;
mod race;
mod real;
mod timing;
mod util;
mod vdsp;

use a::math::complex::Complex;
use a::math::fft::Fft;
use fftw::Fftw;
use prec::P;
use timing::{Cfg, Job};
use util::{Out, Rng};

fn dispatch<T: P>(group: &str, part: &str, o: &mut Out, cfg: &Cfg) {
    match group {
        "complex" => complex::run::<T>(o, cfg, part),
        "real" => real::run::<T>(o, cfg, part),
        "dct" => dct::run::<T>(o, cfg, part),
        "nd" => nd::run::<T>(o, cfg, part),
        "plan" => plan::run::<T>(o, cfg, part),
        "conv" => conv::run::<T>(o, cfg, part),
        "sanity" => sanity::<T>(cfg),
        "patient" => patient::<T>(cfg),
        g => panic!("unknown group {g}"),
    }
}

/// Whether special values or the growth of repeated in-place transforms change the timings:
/// the same plan on zeros, on NaNs, on random data left to grow (overflowing to infinities
/// and NaNs within a sample), and out of place from fixed random data.
fn sanity<T: P>(cfg: &Cfg) {
    for n in [16usize, 1024, 65536] {
        let fft = Fft::<T>::new(n);
        let mut s1 = vec![Complex::<T>::ZERO; fft.scratch_len()];
        let mut s2 = s1.clone();
        let mut s3 = s1.clone();
        let mut z = vec![Complex::<T>::ZERO; n];
        let nan = T::of(f64::NAN);
        let mut q = vec![Complex::new(nan, nan); n];
        let x0 = prec::cvec::<T>(&Rng::new(3).vec(2 * n));
        let mut g = x0.clone();
        let mut y = vec![Complex::<T>::ZERO; n];
        let mut jobs: Vec<Job> = vec![
            Box::new(|| fft.forward_with_scratch(&mut z, &mut s1)),
            Box::new(|| fft.forward_with_scratch(&mut q, &mut s2)),
            Box::new(|| fft.forward_with_scratch(&mut g, &mut s3)),
            Box::new(|| fft.forward_to(&x0, &mut y)),
        ];
        let st = timing::run(&mut jobs, cfg);
        println!(
            "sanity T={} n={n}: zeros {:.1} ns, NaN {:.1} ns, growing {:.1} ns, out of place {:.1} ns",
            T::NAME,
            st[0].median,
            st[1].median,
            st[2].median,
            st[3].median
        );
    }
}

/// FFTW_PATIENT against FFTW_MEASURE plans (each planned with no wisdom), complex in place.
fn patient<T: P>(cfg: &Cfg) {
    for n in [64usize, 1024, 4096, 65536, 1048576, 1000, 10007] {
        let mut a = util::ABuf::<Complex<T>>::new(n);
        let mut b = util::ABuf::<Complex<T>>::new(n);
        let (pa, pb) = (prec::cp(&mut a), prec::cp(&mut b));
        let t0 = std::time::Instant::now();
        unsafe { T::forget() };
        let m = fftw::Plan::<T>::wrap(unsafe { T::dft_1d(n, pa, pa, fftw::FORWARD, fftw::MEASURE) }).unwrap();
        let tm = t0.elapsed().as_secs_f64();
        unsafe { T::forget() };
        let t1 = std::time::Instant::now();
        let p = fftw::Plan::<T>::wrap(unsafe { T::dft_1d(n, pb, pb, fftw::FORWARD, fftw::PATIENT) }).unwrap();
        let tp = t1.elapsed().as_secs_f64();
        let mut jobs: Vec<Job> = vec![Box::new(|| m.exec()), Box::new(|| p.exec())];
        let st = timing::run(&mut jobs, cfg);
        println!(
            "patient T={} n={n}: MEASURE {:.1} ns (planned in {tm:.2} s), PATIENT {:.1} ns (planned in {tp:.2} s), ratio {:.3}",
            T::NAME, st[0].median, st[1].median, st[1].median / st[0].median
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("header") {
        println!("area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes");
        return;
    }
    if args.len() < 3 {
        eprintln!("usage: fftbench header | fftbench <group> <type> [part]");
        std::process::exit(2);
    }
    let (group, ty) = (args[1].as_str(), args[2].as_str());
    if group == "convloop" {
        // fftbench convloop <la> <lb> <seconds>: f64 convolve in a loop, for a profiler.
        let la: usize = ty.parse().expect("la");
        let lb: usize = args[3].parse().expect("lb");
        let secs: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        let a: Vec<f64> = Rng::new(1).vec(la);
        let b: Vec<f64> = Rng::new(2).vec(lb);
        let t = std::time::Instant::now();
        let mut k = 0u64;
        while t.elapsed().as_secs_f64() < secs {
            for _ in 0..1000 {
                std::hint::black_box(a::math::fft::convolve(&a, &b));
            }
            k += 1000;
        }
        println!("{} calls, {:.1} ns each", k, t.elapsed().as_secs_f64() / k as f64 * 1e9);
        return;
    }
    if group == "realloop" {
        // fftbench realloop <n> <seconds>: f64 RealFft::forward in a loop, for a profiler.
        let n: usize = ty.parse().expect("n");
        let secs: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        let fft = a::math::fft::RealFft::<f64>::new(n);
        let x: Vec<f64> = Rng::new(1).vec(n);
        let mut y = vec![Complex::<f64>::ZERO; n / 2 + 1];
        let t = std::time::Instant::now();
        let mut k = 0u64;
        while t.elapsed().as_secs_f64() < secs {
            for _ in 0..100 {
                fft.forward(&x, &mut y);
            }
            k += 100;
        }
        println!("{} calls, {:.1} ns each", k, t.elapsed().as_secs_f64() / k as f64 * 1e9);
        return;
    }
    if group == "describe" {
        // fftbench describe: the plans' Debug of some lengths.
        for n in [16usize, 32, 1024, 1_000_000, 1009, 10007, 202, 25, 64, 128, 4096] {
            println!("{n}: {:?} / {:?}", Fft::<f64>::new(n), Fft::<f32>::new(n));
        }
        for n in [2usize, 3, 15, 16, 26, 28, 30, 36, 64, 66, 135, 2401] {
            let r = a::math::fft::RealFft::<f64>::new(n);
            let q = a::math::fft::RealFft::<f32>::new(n);
            println!("real {n}: scratch f64 {} f32 {}", r.scratch_len(), q.scratch_len());
        }
        for n in [8usize, 9, 16, 17] {
            let d = a::math::fft::Dct::<f64>::new(n, a::math::fft::DctKind::II);
            let e = a::math::fft::Dct::<f64>::new(n, a::math::fft::DctKind::I);
            println!("dct {n}: II {} I {}", d.scratch_len(), e.scratch_len());
        }
        for n in [16usize, 25, 6, 32, 23] {
            println!("complex {n}: scratch {}", Fft::<f64>::new(n).scratch_len());
        }
        return;
    }
    if group == "planloop" {
        // fftbench planloop <f32|f64> <n> <seconds>: plans in a loop, for a profiler.
        let n: usize = args[3].parse().expect("n");
        let secs: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        // mix: a vDSP setup of the same type made between plans (untimed); mem: a fresh buffer
        // of the size of the expanded tables (4 reals per entry) written in order instead.
        let mode = args.get(5).map(String::as_str).unwrap_or("");
        let mut k = 0u64;
        let mut ns = 0.0;
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs_f64() < secs {
            if mode == "mix" {
                vdsp::mix_setup(n, ty != "f32");
            }
            let t = std::time::Instant::now();
            match (ty, mode) {
                ("f32", "mem") => {
                    let mut v: Vec<f32> = Vec::with_capacity(4 * n);
                    v.extend((0..4 * n).map(|i| i as f32));
                    std::hint::black_box(v);
                }
                (_, "mem") => {
                    let mut v: Vec<f64> = Vec::with_capacity(4 * n);
                    v.extend((0..4 * n).map(|i| i as f64));
                    std::hint::black_box(v);
                }
                ("f32", _) => drop(std::hint::black_box(Fft::<f32>::new(n))),
                _ => drop(std::hint::black_box(Fft::<f64>::new(n))),
            }
            ns += t.elapsed().as_secs_f64();
            k += 1;
        }
        println!("{ty} {mode}: {} plans, {:.1} us each", k, ns / k as f64 * 1e6);
        return;
    }
    if group == "fuzz" {
        // fftbench fuzz <seconds> [seed]
        let secs: f64 = ty.parse().expect("seconds");
        let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
        fuzz::run(secs, seed);
        return;
    }
    let part = args.get(3).map(String::as_str).unwrap_or("all");
    let cfg = Cfg::from_env();
    eprintln!(
        "fftbench {group} {ty} {part}: FFTW {}, {} samples of >= {:.1} ms",
        f64::version(),
        cfg.samples,
        cfg.min_ns / 1e6
    );
    fftw::load_wisdom::<f64>();
    fftw::load_wisdom::<f32>();
    let mut o = Out::new();
    let t = std::time::Instant::now();
    match (group, ty) {
        ("conv", "c32") => conv::run::<Complex<f32>>(&mut o, &cfg, part),
        ("conv", "c64") => conv::run::<Complex<f64>>(&mut o, &cfg, part),
        (g, "f32") => dispatch::<f32>(g, part, &mut o, &cfg),
        (g, "f64") => dispatch::<f64>(g, part, &mut o, &cfg),
        _ => panic!("unknown type {ty}"),
    }
    if group != "plan" && group != "patient" {
        fftw::save_wisdom::<f64>();
        fftw::save_wisdom::<f32>();
    }
    eprintln!(
        "fftbench {group} {ty} {part}: {} rows in {:.1} s",
        o.rows,
        t.elapsed().as_secs_f64()
    );
}
