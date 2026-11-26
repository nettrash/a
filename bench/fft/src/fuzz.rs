//! A differential fuzz of `a::math::fft` against FFTW (and a direct sum for the convolutions):
//! random lengths, kinds and data, the relative L2 distance against the bound of the checks.

use crate::fftw::{self, Plan};
use crate::prec::*;
use crate::util::{ABuf, Rng, rel};
use a::math::fft::{
    Dct, DctKind, Dst, DstKind, Fft, Fft2, RealFft, circular_convolve, convolve, convolve_into,
    correlate,
};

struct Tally {
    cases: usize,
    fails: usize,
    worst: f64,
}

impl Tally {
    fn check(&mut self, what: &str, e: f64, bound: f64) {
        self.cases += 1;
        self.worst = self.worst.max(e / bound);
        if !(e <= bound) {
            self.fails += 1;
            eprintln!("FUZZ FAIL {what}: rel {e:.3e} > {bound:.3e}");
        }
    }
}

fn pick(r: &mut Rng, lo: usize, hi: usize) -> usize {
    lo + (((r.next() + 1.0) / 2.0) * (hi - lo + 1) as f64) as usize % (hi - lo + 1)
}

fn length(r: &mut Rng) -> usize {
    match pick(r, 0, 9) {
        0..=4 => pick(r, 1, 64),
        5..=7 => pick(r, 65, 2048),
        8 => pick(r, 2049, 20000),
        _ => 1 << pick(r, 6, 16),
    }
}

fn one<T: P>(r: &mut Rng, t: &mut Tally) {
    let n = length(r);
    let b = 30.0 * T::EPS * ((n as f64).log2() + 1.0);
    let x = cvec::<T>(&r.vec(2 * n));
    // Complex, forward and inverse, in place.
    for inv in [false, true] {
        let fft = Fft::<T>::new(n);
        let mut y = x.clone();
        if inv { fft.inverse(&mut y) } else { fft.forward(&mut y) }
        let mut fb = ABuf::<C<T>>::new(n);
        let sign = if inv { fftw::BACKWARD } else { fftw::FORWARD };
        let p = Plan::<T>::wrap(unsafe { T::dft_1d(n, cp(&mut fb), cp(&mut fb), sign, fftw::ESTIMATE) })
            .unwrap();
        fb.copy_from_slice(&x);
        p.exec();
        t.check(&format!("{} complex n={n} inv={inv}", T::NAME), rel(&flat(&fb), &flat(&y)), b);
        let mut z = vec![C::<T>::ZERO; n];
        if inv { fft.inverse_to(&x, &mut z) } else { fft.forward_to(&x, &mut z) }
        t.check(&format!("{} complex_to n={n}", T::NAME), rel(&flat(&fb), &flat(&z)), b);
    }
    // Real forward and inverse.
    let xr = rvec::<T>(&r.vec(n));
    let rf = RealFft::<T>::new(n);
    let h = n / 2 + 1;
    let mut y = vec![C::<T>::ZERO; h];
    rf.forward(&xr, &mut y);
    let (mut fi, mut fo) = (ABuf::<T>::new(n), ABuf::<C<T>>::new(h));
    let p = Plan::<T>::wrap(unsafe { T::r2c_1d(n, fi.ptr(), cp(&mut fo), fftw::ESTIMATE) }).unwrap();
    fi.copy_from_slice(&xr);
    p.exec();
    t.check(&format!("{} r2c n={n}", T::NAME), rel(&flat(&fo), &flat(&y)), b);
    let mut back = vec![T::of(0.0); n];
    rf.inverse(&y, &mut back);
    let (mut ci, mut co) = (ABuf::<C<T>>::new(h), ABuf::<T>::new(n));
    let p = Plan::<T>::wrap(unsafe { T::c2r_1d(n, cp(&mut ci), co.ptr(), fftw::ESTIMATE) }).unwrap();
    ci.copy_from_slice(&y);
    p.exec();
    t.check(&format!("{} c2r n={n}", T::NAME), rel(&flatr(&co), &flatr(&back)), b);
    // A batch of 3 against single runs.
    let x3 = rvec::<T>(&r.vec(3 * n));
    let mut y3 = vec![C::<T>::ZERO; 3 * h];
    rf.forward(&x3, &mut y3);
    for k in 0..3 {
        let mut y1 = vec![C::<T>::ZERO; h];
        rf.forward(&x3[k * n..(k + 1) * n], &mut y1);
        t.check(&format!("{} r2c batch n={n}", T::NAME), rel(&flat(&y1), &flat(&y3[k * h..(k + 1) * h])), b);
    }
    // The eight cosine and sine transforms.
    for (k, kind) in [(1u8, fftw::REDFT00), (2, fftw::REDFT10), (3, fftw::REDFT01), (4, fftw::REDFT11)] {
        if k == 1 && n < 2 {
            continue;
        }
        let d = Dct::<T>::new(n, [DctKind::I, DctKind::II, DctKind::III, DctKind::IV][k as usize - 1]);
        let mut y = vec![T::of(0.0); n];
        d.transform_to(&xr, &mut y);
        let (mut i, mut o) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
        let p = Plan::<T>::wrap(unsafe { T::r2r_1d(n, i.ptr(), o.ptr(), kind, fftw::ESTIMATE) }).unwrap();
        i.copy_from_slice(&xr);
        p.exec();
        t.check(&format!("{} DCT-{k} n={n}", T::NAME), rel(&flatr(&o), &flatr(&y)), 3.0 * b);
    }
    for (k, kind) in [(1u8, fftw::RODFT00), (2, fftw::RODFT10), (3, fftw::RODFT01), (4, fftw::RODFT11)] {
        let d = Dst::<T>::new(n, [DstKind::I, DstKind::II, DstKind::III, DstKind::IV][k as usize - 1]);
        let mut y = xr.clone();
        d.transform(&mut y);
        let (mut i, mut o) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
        let p = Plan::<T>::wrap(unsafe { T::r2r_1d(n, i.ptr(), o.ptr(), kind, fftw::ESTIMATE) }).unwrap();
        i.copy_from_slice(&xr);
        p.exec();
        t.check(&format!("{} DST-{k} n={n}", T::NAME), rel(&flatr(&o), &flatr(&y)), 3.0 * b);
    }
    // Two dimensions.
    let (rows, cols) = (pick(r, 1, 300), pick(r, 1, 300));
    let m = rows * cols;
    let x2 = cvec::<T>(&r.vec(2 * m));
    let f2 = Fft2::<T>::new(rows, cols);
    let mut y2 = x2.clone();
    f2.forward(&mut y2);
    let mut fb = ABuf::<C<T>>::new(m);
    let p = Plan::<T>::wrap(unsafe {
        T::dft_2d(rows, cols, cp(&mut fb), cp(&mut fb), fftw::FORWARD, fftw::ESTIMATE)
    })
    .unwrap();
    fb.copy_from_slice(&x2);
    p.exec();
    let b2 = 30.0 * T::EPS * ((m as f64).log2() + 1.0);
    t.check(&format!("{} 2d {rows}x{cols}", T::NAME), rel(&flat(&fb), &flat(&y2)), b2);
    // Convolutions against the direct sum in f64.
    let (la, lb) = (pick(r, 1, 3000), pick(r, 1, 300));
    let (a, bb) = (rvec::<T>(&r.vec(la)), rvec::<T>(&r.vec(lb)));
    let (af, bf) = (flatr(&a), flatr(&bb));
    let mut want = vec![0.0; la + lb - 1];
    for (i, &u) in af.iter().enumerate() {
        for (j, &v) in bf.iter().enumerate() {
            want[i + j] += u * v;
        }
    }
    let got = convolve(&a, &bb);
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = norm(&af) * norm(&bf) / norm(&want).max(f64::MIN_POSITIVE);
    let bc = 200.0 * T::EPS * (((la + lb) as f64).log2() + 1.0) * scale;
    t.check(&format!("{} conv {la}x{lb}", T::NAME), rel(&want, &flatr(&got)), bc);
    let mut into = vec![T::of(0.0); la + lb - 1];
    convolve_into(&a, &bb, &mut into);
    t.check(&format!("{} conv_into {la}x{lb}", T::NAME), rel(&flatr(&got), &flatr(&into)), 0.0);
    let cr = correlate(&a, &bb);
    let mut want = vec![0.0; la + lb - 1];
    for (i, &u) in af.iter().enumerate() {
        for (j, &v) in bf.iter().enumerate() {
            want[i + lb - 1 - j] += u * v;
        }
    }
    t.check(&format!("{} corr {la}x{lb}", T::NAME), rel(&want, &flatr(&cr)), bc);
    let nc = pick(r, 1, 400);
    let (a, bb) = (rvec::<T>(&r.vec(nc)), rvec::<T>(&r.vec(nc)));
    let (af, bf) = (flatr(&a), flatr(&bb));
    let mut want = vec![0.0; nc];
    for i in 0..nc {
        for j in 0..nc {
            want[(i + j) % nc] += af[i] * bf[j];
        }
    }
    let got = circular_convolve(&a, &bb);
    let scale = norm(&af) * norm(&bf) / norm(&want).max(f64::MIN_POSITIVE);
    let bc = 200.0 * T::EPS * ((nc as f64).log2() + 1.0) * scale;
    t.check(&format!("{} circ {nc}", T::NAME), rel(&want, &flatr(&got)), bc);
}

pub fn run(secs: f64, seed: u64) {
    let mut r = Rng::new(seed);
    let mut t = Tally { cases: 0, fails: 0, worst: 0.0 };
    let start = std::time::Instant::now();
    let mut rounds = 0;
    while start.elapsed().as_secs_f64() < secs {
        one::<f64>(&mut r, &mut t);
        one::<f32>(&mut r, &mut t);
        rounds += 1;
    }
    println!(
        "fuzz: {rounds} rounds, {} cases, {} failures, worst {:.3} of the bound",
        t.cases, t.fails, t.worst
    );
}
