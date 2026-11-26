//! Cosine and sine transforms of types I to IV.

use crate::complex::join_notes;
use crate::fftw::{self, Plan};
use crate::prec::*;
use crate::race::Race;
use crate::real::{scale_note, scaled};
use crate::timing::Cfg;
use crate::util::{ABuf, Out, Rng};
use crate::vdsp;
use a::math::fft::{Dct, DctKind, Dst, DstKind};
use rustdct::DctPlanner;
use std::sync::Arc;

pub const SIZES: &[usize] = &[
    8, 16, 64, 256, 1024, 4096, 16384, 65536, 1048576, 48, 1000, 10000, 3000, 255, 1023,
];
pub const SIZES_DCT1: &[usize] = &[9, 17, 65, 257, 1025, 4097, 16385, 65537, 64, 1000, 1024, 4096];
pub const SIZES_DST1: &[usize] = &[7, 15, 63, 255, 1023, 4095, 16383, 65535, 64, 1000, 1024, 4096];

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    C(u8),
    S(u8),
}

impl Kind {
    pub fn name(self) -> String {
        let r = ["", "I", "II", "III", "IV"];
        match self {
            Kind::C(t) => format!("DCT-{}", r[t as usize]),
            Kind::S(t) => format!("DST-{}", r[t as usize]),
        }
    }
    fn fftw(self) -> i32 {
        match self {
            Kind::C(1) => fftw::REDFT00,
            Kind::C(2) => fftw::REDFT10,
            Kind::C(3) => fftw::REDFT01,
            Kind::C(4) => fftw::REDFT11,
            Kind::S(1) => fftw::RODFT00,
            Kind::S(2) => fftw::RODFT10,
            Kind::S(3) => fftw::RODFT01,
            Kind::S(4) => fftw::RODFT11,
            _ => unreachable!(),
        }
    }
    fn fftw_name(self) -> &'static str {
        match self {
            Kind::C(1) => "REDFT00",
            Kind::C(2) => "REDFT10",
            Kind::C(3) => "REDFT01",
            Kind::C(4) => "REDFT11",
            Kind::S(1) => "RODFT00",
            Kind::S(2) => "RODFT10",
            Kind::S(3) => "RODFT01",
            Kind::S(4) => "RODFT11",
            _ => unreachable!(),
        }
    }
}

enum Ours<T> {
    C(Dct<T>),
    S(Dst<T>),
}

impl<T: P> Ours<T> {
    fn new(k: Kind, n: usize) -> Ours<T> {
        match k {
            Kind::C(t) => Ours::C(Dct::new(
                n,
                [DctKind::I, DctKind::II, DctKind::III, DctKind::IV][t as usize - 1],
            )),
            Kind::S(t) => Ours::S(Dst::new(
                n,
                [DstKind::I, DstKind::II, DstKind::III, DstKind::IV][t as usize - 1],
            )),
        }
    }
    fn to(&self, x: &[T], y: &mut [T]) {
        match self {
            Ours::C(d) => d.transform_to(x, y),
            Ours::S(d) => d.transform_to(x, y),
        }
    }
    fn ip(&self, x: &mut [T], s: &mut [C<T>]) {
        match self {
            Ours::C(d) => d.transform_with_scratch(x, s),
            Ours::S(d) => d.transform_with_scratch(x, s),
        }
    }
    fn scratch_len(&self) -> usize {
        match self {
            Ours::C(d) => d.scratch_len(),
            Ours::S(d) => d.scratch_len(),
        }
    }
}

type RdFn<T> = Box<dyn Fn(&mut [T], &mut [T])>;

/// rustdct's plan of a kind, as a closure on (buffer, scratch), and its scratch length.
fn rustdct<T: P>(k: Kind, n: usize) -> (RdFn<T>, usize) {
    let mut p = DctPlanner::<T>::new();
    match k {
        Kind::C(1) => {
            let d = p.plan_dct1(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dct1_with_scratch(b, s)), s)
        }
        Kind::C(2) => {
            let d = p.plan_dct2(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dct2_with_scratch(b, s)), s)
        }
        Kind::C(3) => {
            let d = p.plan_dct3(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dct3_with_scratch(b, s)), s)
        }
        Kind::C(4) => {
            let d = p.plan_dct4(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dct4_with_scratch(b, s)), s)
        }
        Kind::S(1) => {
            let d = p.plan_dst1(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dst1_with_scratch(b, s)), s)
        }
        Kind::S(2) => {
            let d = p.plan_dst2(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dst2_with_scratch(b, s)), s)
        }
        Kind::S(3) => {
            let d = p.plan_dst3(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dst3_with_scratch(b, s)), s)
        }
        Kind::S(4) => {
            let d = p.plan_dst4(n);
            let s = d.get_scratch_len();
            (Box::new(move |b, s| d.process_dst4_with_scratch(b, s)), s)
        }
        _ => unreachable!(),
    }
}

pub fn row<T: P>(o: &mut Out, cfg: &Cfg, k: Kind, n: usize) {
    let x0 = rvec::<T>(&Rng::new(n as u64 + 31).vec(n));
    let tl = tol::<T>(n);
    let ours = Arc::new(Ours::<T>::new(k, n));
    let mut y = ABuf::<T>::new(n);
    ours.to(&x0, &mut y);
    let want = flatr(&y);
    let what = format!("{} n={n}", k.name());

    let (mut fi, mut fo) = (ABuf::<T>::new(n), ABuf::<T>::new(n));
    let fp = Plan::<T>::wrap(unsafe { T::r2r_1d(n, fi.ptr(), fo.ptr(), k.fftw(), fftw::MEASURE) })
        .unwrap();
    fi.copy_from_slice(&x0);
    fp.exec();
    let (ok, s) = scaled(&want, &flatr(&fo), tl);
    let fnote = scale_note(&what, "FFTW", ok, s);

    let (rd, rlen) = rustdct::<T>(k, n);
    let mut rb = ABuf::<T>::from(&x0);
    let mut rs = vec![T::of(0.0); rlen];
    rd(&mut rb, &mut rs);
    let (ok, s) = scaled(&want, &flatr(&rb), tl);
    let rnote = scale_note(&what, "rustdct", ok, s);

    let vk = match k {
        Kind::C(2) => 2,
        Kind::C(3) => 3,
        Kind::C(4) => 4,
        _ => 0,
    };
    let vs = if vk != 0 { unsafe { T::dct_setup(n, vk) } } else { std::ptr::null_mut() };
    let (mut vi, mut vo) = (ABuf::<T>::from(&x0), ABuf::<T>::new(n));
    let mut vnote = String::new();
    if !vs.is_null() {
        unsafe { T::dct_exec(vs, vi.ptr(), vo.ptr()) };
        let (ok, s) = scaled(&want, &flatr(&vo), tl);
        vnote = scale_note(&what, "vDSP DCT", ok, s);
    }

    let xo = ABuf::<T>::from(&x0);
    let mut ib = ABuf::<T>::from(&x0);
    let mut sc = vec![C::<T>::ZERO; ours.scratch_len()];
    let mut race = Race::new();
    let (o1, o2) = (ours.clone(), ours.clone());
    let us_to = race.ours(&format!("{}::transform_to", if matches!(k, Kind::C(_)) { "Dct" } else { "Dst" }), move || {
        o1.to(&xo, &mut y)
    });
    let us_ip = race.ours(
        &format!("{}::transform_with_scratch", if matches!(k, Kind::C(_)) { "Dct" } else { "Dst" }),
        move || o2.ip(&mut ib, &mut sc),
    );
    let fpr = &fp;
    race.reference(
        "FFTW",
        &join_notes(&format!("r2r {}, FFTW_MEASURE, out of place", k.fftw_name()), &fnote),
        us_to,
        move || fpr.exec(),
    );
    if !vs.is_null() {
        let (p, q) = (vi.ptr(), vo.ptr());
        race.reference(
            "vDSP DCT",
            &join_notes("vDSP_DCT_Execute, out of place", &vnote),
            us_to,
            move || unsafe { T::dct_exec(vs, p, q) },
        );
    }
    race.reference(
        "rustdct",
        &join_notes("process_*_with_scratch, in place; against our in-place form", &rnote),
        us_ip,
        || rd(&mut rb, &mut rs),
    );
    let group = match k {
        Kind::C(t) => format!("dct/{}", ["", "I", "II", "III", "IV"][t as usize]),
        Kind::S(t) => format!("dst/{}", ["", "I", "II", "III", "IV"][t as usize]),
    };
    race.finish(o, cfg, &group, &format!("T={} n={n}", T::NAME), 1.0);
    unsafe { vdsp::dct_destroy(vs) };
}

pub fn run<T: P>(o: &mut Out, cfg: &Cfg, part: &str) {
    let kinds: Vec<Kind> = match part {
        "all" => vec![
            Kind::C(1),
            Kind::C(2),
            Kind::C(3),
            Kind::C(4),
            Kind::S(1),
            Kind::S(2),
            Kind::S(3),
            Kind::S(4),
        ],
        "dct" => vec![Kind::C(1), Kind::C(2), Kind::C(3), Kind::C(4)],
        "dst" => vec![Kind::S(1), Kind::S(2), Kind::S(3), Kind::S(4)],
        p => panic!("dct part {p}?"),
    };
    for k in kinds {
        let sizes = match k {
            Kind::C(1) => SIZES_DCT1,
            Kind::S(1) => SIZES_DST1,
            _ => SIZES,
        };
        for &n in sizes {
            row::<T>(o, cfg, k, n);
        }
    }
}
