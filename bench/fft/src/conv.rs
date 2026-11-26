//! Convolutions and correlations against FFTW and rustfft/realfft (padded product or
//! overlap-add, the fastest length of each kept) and vDSP's direct `conv`/`zconv`.

use crate::complex::join_notes;
use crate::fftw::{self, Plan};
use crate::prec::*;
use crate::race::Race;
use crate::timing::Cfg;
use crate::util::{ABuf, Out, Rng, check};
use crate::vdsp::{self, Split};
use a::math::fft::{
    Sample, circular_convolve, circular_correlate, convolve, convolve_into, correlate,
    correlate_into,
};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

pub const SIZES: &[(usize, usize)] = &[
    (16, 3), (16, 8), (16, 16),
    (100, 3), (100, 8), (100, 16), (100, 32), (100, 64),
    (1000, 3), (1000, 8), (1000, 16), (1000, 32), (1000, 64), (1000, 128), (1000, 256), (1000, 1000),
    (10000, 3), (10000, 8), (10000, 16), (10000, 32), (10000, 64), (10000, 128), (10000, 256),
    (10000, 1000), (10000, 10000),
    (100000, 3), (100000, 8), (100000, 16), (100000, 32), (100000, 64), (100000, 128),
    (100000, 256), (100000, 1000), (100000, 10000), (100000, 100000),
    (1000000, 3), (1000000, 8), (1000000, 16), (1000000, 32), (1000000, 64), (1000000, 128),
    (1000000, 256), (1000000, 1000), (1000000, 10000), (1000000, 100000),
];

pub const CORR: &[(usize, usize)] = &[
    (100, 8), (1000, 16), (1000, 1000), (10000, 64), (10000, 1000), (100000, 32), (100000, 10000),
    (1000000, 128),
];

pub const CIRC: &[usize] = &[16, 100, 128, 1000, 1009, 1024, 4096, 10007, 65536, 100000, 1048576];

/// vDSP's direct sums are timed only up to this many products.
const VDSP_MAX_PRODUCTS: f64 = 1e8;

/// The sample types: `T` or `Complex<T>`.
pub trait Samp: Sample + Copy + Send + Sync + 'static {
    type T: P;
    const CPLX: bool;
    fn zero() -> Self;
    fn cj(self) -> Self;
    fn make(r: &mut Rng, n: usize) -> Vec<Self>;
    fn flat(x: &[Self]) -> Vec<f64>;
    fn name() -> String;
    fn engine(lib: Lib, l: usize) -> Box<dyn Engine<Self>>;
    /// vDSP's direct convolution (`dir` = −1) or correlation (`dir` = +1) of `a` padded with
    /// `lb − 1` zeros at both ends by `f` (its first value, or its last for a convolution).
    fn vdsp(v: &mut VdspConv<Self>, dir: isize);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Lib {
    Fftw,
    Rust,
}

/// A transform of the length `L` on owned buffers: time → spectrum and back, unscaled.
pub trait Engine<S: Samp> {
    fn len(&self) -> usize;
    fn time(&mut self) -> &mut [S];
    fn spec(&mut self) -> &mut [C<S::T>];
    fn fwd(&mut self);
    fn inv(&mut self);
}

struct FftwReal<T: P> {
    l: usize,
    r: ABuf<T>,
    c: ABuf<C<T>>,
    f: Plan<T>,
    b: Plan<T>,
}

impl<T: P> Engine<T> for FftwReal<T>
where
    T: Samp<T = T>,
{
    fn len(&self) -> usize {
        self.l
    }
    fn time(&mut self) -> &mut [T] {
        &mut self.r
    }
    fn spec(&mut self) -> &mut [C<T>] {
        &mut self.c
    }
    fn fwd(&mut self) {
        self.f.exec()
    }
    fn inv(&mut self) {
        self.b.exec()
    }
}

struct RustReal<T: P> {
    l: usize,
    r: ABuf<T>,
    c: ABuf<C<T>>,
    f: Arc<dyn RealToComplex<T>>,
    b: Arc<dyn ComplexToReal<T>>,
    s: Vec<Nc<T>>,
}

impl<T: P> Engine<T> for RustReal<T>
where
    T: Samp<T = T>,
{
    fn len(&self) -> usize {
        self.l
    }
    fn time(&mut self) -> &mut [T] {
        &mut self.r
    }
    fn spec(&mut self) -> &mut [C<T>] {
        &mut self.c
    }
    fn fwd(&mut self) {
        let _ = self.f.process_with_scratch(&mut self.r, nc(&mut self.c), &mut self.s);
    }
    fn inv(&mut self) {
        let _ = self.b.process_with_scratch(nc(&mut self.c), &mut self.r, &mut self.s);
    }
}

struct FftwCplx<T: P> {
    l: usize,
    x: ABuf<C<T>>,
    f: Plan<T>,
    b: Plan<T>,
}

impl<T: P> Engine<C<T>> for FftwCplx<T>
where
    C<T>: Samp<T = T>,
{
    fn len(&self) -> usize {
        self.l
    }
    fn time(&mut self) -> &mut [C<T>] {
        &mut self.x
    }
    fn spec(&mut self) -> &mut [C<T>] {
        &mut self.x
    }
    fn fwd(&mut self) {
        self.f.exec()
    }
    fn inv(&mut self) {
        self.b.exec()
    }
}

struct RustCplx<T: P> {
    l: usize,
    x: ABuf<C<T>>,
    f: Arc<dyn rustfft::Fft<T>>,
    b: Arc<dyn rustfft::Fft<T>>,
    s: Vec<Nc<T>>,
}

impl<T: P> Engine<C<T>> for RustCplx<T>
where
    C<T>: Samp<T = T>,
{
    fn len(&self) -> usize {
        self.l
    }
    fn time(&mut self) -> &mut [C<T>] {
        &mut self.x
    }
    fn spec(&mut self) -> &mut [C<T>] {
        &mut self.x
    }
    fn fwd(&mut self) {
        self.f.process_with_scratch(nc(&mut self.x), &mut self.s)
    }
    fn inv(&mut self) {
        self.b.process_with_scratch(nc(&mut self.x), &mut self.s)
    }
}

/// vDSP's direct path on pre-padded (and, for complex values, pre-split) operands.
pub struct VdspConv<S: Samp> {
    pub la: usize,
    pub lb: usize,
    pub a: ABuf<S::T>,  // real: padded a; complex: re of padded a
    pub ai: ABuf<S::T>, // complex: im of padded a
    pub f: ABuf<S::T>,
    pub fi: ABuf<S::T>,
    pub c: ABuf<S::T>,
    pub ci: ABuf<S::T>,
}

impl<S: Samp> VdspConv<S> {
    pub fn new(a: &[S], b: &[S], conj_b: bool) -> VdspConv<S> {
        let (la, lb) = (a.len(), b.len());
        let pad = la + 2 * (lb - 1);
        let n = la + lb - 1;
        let fa = S::flat(a);
        let fb = S::flat(b);
        let w = if S::CPLX { 2 } else { 1 };
        let mut v = VdspConv {
            la,
            lb,
            a: ABuf::new(pad),
            ai: ABuf::new(if S::CPLX { pad } else { 1 }),
            f: ABuf::new(lb),
            fi: ABuf::new(if S::CPLX { lb } else { 1 }),
            c: ABuf::new(n),
            ci: ABuf::new(if S::CPLX { n } else { 1 }),
        };
        for i in 0..la {
            v.a[lb - 1 + i] = <S::T as P>::of(fa[w * i]);
            if S::CPLX {
                v.ai[lb - 1 + i] = <S::T as P>::of(fa[w * i + 1]);
            }
        }
        for i in 0..lb {
            v.f[i] = <S::T as P>::of(fb[w * i]);
            if S::CPLX {
                let im = fb[w * i + 1];
                v.fi[i] = <S::T as P>::of(if conj_b { -im } else { im });
            }
        }
        v
    }
    pub fn out(&self) -> Vec<f64> {
        if S::CPLX {
            join(&self.c, &self.ci)
        } else {
            flatr(&self.c)
        }
    }
}

impl<T: P> Samp for T
where
    T: Sample,
{
    type T = T;
    const CPLX: bool = false;
    fn zero() -> T {
        T::of(0.0)
    }
    fn cj(self) -> T {
        self
    }
    fn make(r: &mut Rng, n: usize) -> Vec<T> {
        rvec::<T>(&r.vec(n))
    }
    fn flat(x: &[T]) -> Vec<f64> {
        flatr(x)
    }
    fn name() -> String {
        T::NAME.to_string()
    }
    fn engine(lib: Lib, l: usize) -> Box<dyn Engine<T>> {
        let h = l / 2 + 1;
        let (mut r, mut c) = (ABuf::<T>::new(l), ABuf::<C<T>>::new(h));
        match lib {
            Lib::Fftw => {
                let f = Plan::wrap(unsafe { T::r2c_1d(l, r.ptr(), cp(&mut c), fftw::MEASURE) }).unwrap();
                let b = Plan::wrap(unsafe { T::c2r_1d(l, cp(&mut c), r.ptr(), fftw::MEASURE) }).unwrap();
                Box::new(FftwReal { l, r, c, f, b })
            }
            Lib::Rust => {
                let mut p = RealFftPlanner::<T>::new();
                let f = p.plan_fft_forward(l);
                let b = p.plan_fft_inverse(l);
                let s = vec![Nc::<T>::default(); f.get_scratch_len().max(b.get_scratch_len())];
                Box::new(RustReal { l, r, c, f, b, s })
            }
        }
    }
    fn vdsp(v: &mut VdspConv<T>, dir: isize) {
        let n = v.la + v.lb - 1;
        let fp = if dir < 0 {
            // SAFETY: the last tap; vDSP steps backwards from it.
            unsafe { v.f.ptr().add(v.lb - 1) }
        } else {
            v.f.ptr()
        };
        unsafe { T::conv(v.a.ptr(), fp, dir, v.c.ptr(), n, v.lb) }
    }
}

impl<T: P> Samp for C<T>
where
    C<T>: Sample,
{
    type T = T;
    const CPLX: bool = true;
    fn zero() -> C<T> {
        C::ZERO
    }
    fn cj(self) -> C<T> {
        C::new(self.re, T::of(-self.im.to()))
    }
    fn make(r: &mut Rng, n: usize) -> Vec<C<T>> {
        cvec::<T>(&r.vec(2 * n))
    }
    fn flat(x: &[C<T>]) -> Vec<f64> {
        flat(x)
    }
    fn name() -> String {
        format!("Complex<{}>", T::NAME)
    }
    fn engine(lib: Lib, l: usize) -> Box<dyn Engine<C<T>>> {
        let mut x = ABuf::<C<T>>::new(l);
        match lib {
            Lib::Fftw => {
                let f = Plan::wrap(unsafe {
                    T::dft_1d(l, cp(&mut x), cp(&mut x), fftw::FORWARD, fftw::MEASURE)
                })
                .unwrap();
                let b = Plan::wrap(unsafe {
                    T::dft_1d(l, cp(&mut x), cp(&mut x), fftw::BACKWARD, fftw::MEASURE)
                })
                .unwrap();
                Box::new(FftwCplx { l, x, f, b })
            }
            Lib::Rust => {
                let mut p = FftPlanner::<T>::new();
                let f = p.plan_fft_forward(l);
                let b = p.plan_fft_inverse(l);
                let s = vec![
                    Nc::<T>::default();
                    f.get_inplace_scratch_len().max(b.get_inplace_scratch_len())
                ];
                Box::new(RustCplx { l, x, f, b, s })
            }
        }
    }
    fn vdsp(v: &mut VdspConv<C<T>>, dir: isize) {
        let n = v.la + v.lb - 1;
        let a = Split { re: v.a.ptr(), im: v.ai.ptr() };
        let f = if dir < 0 {
            // SAFETY: the last tap of each part.
            unsafe { Split { re: v.f.ptr().add(v.lb - 1), im: v.fi.ptr().add(v.lb - 1) } }
        } else {
            Split { re: v.f.ptr(), im: v.fi.ptr() }
        };
        let c = Split { re: v.c.ptr(), im: v.ci.ptr() };
        unsafe { T::zconv(&a, &f, dir, &c, n, v.lb) }
    }
}

#[inline]
fn cmul<T: P>(a: C<T>, b: C<T>) -> C<T> {
    C::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// The linear convolution of `a` by `b` (both non-empty) into `out` through the engine:
/// a padded product when the engine is long enough, overlap-add otherwise.
pub fn conv_ref<S: Samp>(e: &mut dyn Engine<S>, sb: &mut [C<S::T>], a: &[S], b: &[S], out: &mut [S]) {
    let l = e.len();
    let (la, lb) = (a.len(), b.len());
    let scale = <S::T as P>::of(1.0 / l as f64);
    let t = e.time();
    t[..lb].copy_from_slice(b);
    t[lb..].fill(<S as Samp>::zero());
    e.fwd();
    for (w, z) in sb.iter_mut().zip(e.spec().iter()) {
        *w = C::new(z.re * scale, z.im * scale);
    }
    let blk = l - lb + 1;
    let single = blk >= la;
    if !single {
        out.fill(<S as Samp>::zero());
    }
    let mut start = 0;
    while start < la {
        let end = (start + blk).min(la);
        let t = e.time();
        t[..end - start].copy_from_slice(&a[start..end]);
        t[end - start..].fill(<S as Samp>::zero());
        e.fwd();
        for (z, w) in e.spec().iter_mut().zip(sb.iter()) {
            *z = cmul(*z, *w);
        }
        e.inv();
        let cnt = end - start + lb - 1;
        let t = e.time();
        if single {
            out.copy_from_slice(&t[..cnt]);
        } else {
            for (o, v) in out[start..start + cnt].iter_mut().zip(&t[..cnt]) {
                *o = *o + *v;
            }
        }
        start = end;
    }
}

/// The smallest `2^a·3^b·5^c·7^d` of at least `n`.
fn smooth_at_least(n: usize) -> usize {
    let mut best = n.next_power_of_two();
    let mut p7 = 1usize;
    while p7 < 2 * n {
        let mut p5 = p7;
        while p5 < 2 * n {
            let mut p3 = p5;
            while p3 < 2 * n {
                let mut v = p3;
                while v < n {
                    v *= 2;
                }
                best = best.min(v);
                p3 *= 3;
            }
            p5 *= 5;
        }
        p7 *= 7;
    }
    best
}

/// The engines of a library kept by length, dropped all at once past a memory bound.
pub struct Engines<S: Samp> {
    map: HashMap<(Lib, usize), Box<dyn Engine<S>>>,
    values: usize,
}

impl<S: Samp> Engines<S> {
    pub fn new() -> Engines<S> {
        Engines {
            map: HashMap::new(),
            values: 0,
        }
    }
    pub fn get(&mut self, lib: Lib, l: usize) -> &mut dyn Engine<S> {
        if !self.map.contains_key(&(lib, l)) {
            if self.values + l > (1 << 24) {
                self.map.clear();
                self.values = 0;
                fftw::save_wisdom::<S::T>();
            }
            self.values += l;
            self.map.insert((lib, l), S::engine(lib, l));
        }
        self.map.get_mut(&(lib, l)).unwrap().as_mut()
    }
    /// Takes an engine out (to be owned by a timed closure) and puts it back later.
    pub fn take(&mut self, lib: Lib, l: usize) -> Box<dyn Engine<S>> {
        self.get(lib, l);
        self.map.remove(&(lib, l)).unwrap()
    }
    pub fn put(&mut self, lib: Lib, l: usize, e: Box<dyn Engine<S>>) {
        self.map.insert((lib, l), e);
    }
}

/// The candidate lengths: the cheapest smooth padded length, the next power of two, and the
/// overlap-add lengths `2^k` from `2·lb` to `64·lb` below the padded length.
fn candidates(la: usize, lb: usize) -> Vec<usize> {
    let n = la + lb - 1;
    let mut v = vec![smooth_at_least(n), n.next_power_of_two()];
    if la >= 2 * lb {
        let mut l = (2 * lb).next_power_of_two();
        while l <= 64 * lb && l < v[0] {
            v.push(l);
            l *= 2;
        }
    }
    v.sort();
    v.dedup();
    v
}

fn quick_ns(f: &mut dyn FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    let mut spent = 0.0;
    let mut reps = 0;
    while reps < 3 || (spent < 3e6 && reps < 50) {
        let t = Instant::now();
        f();
        let ns = t.elapsed().as_nanos() as f64;
        best = best.min(ns);
        spent += ns;
        reps += 1;
    }
    best
}

/// Picks the fastest length of a library for `a ∗ b`; returns it and a note.
fn best_len<S: Samp>(en: &mut Engines<S>, lib: Lib, a: &[S], b: &[S], out: &mut [S]) -> (usize, String) {
    let cands = candidates(a.len(), b.len());
    let mut best = (f64::INFINITY, 0);
    for &l in &cands {
        let e = en.get(lib, l);
        let mut sb = vec![C::<S::T>::ZERO; e.spec().len()];
        let ns = quick_ns(&mut || conv_ref(e, &mut sb, a, b, out));
        if ns < best.0 {
            best = (ns, l);
        }
    }
    let n = a.len() + b.len() - 1;
    let how = if best.1 >= n { "padded product" } else { "overlap-add" };
    (best.1, format!("{how} L={} (fastest of {:?})", best.1, cands))
}

fn lib_name<S: Samp>(lib: Lib) -> &'static str {
    match (lib, S::CPLX) {
        (Lib::Fftw, _) => "FFTW",
        (Lib::Rust, false) => "realfft",
        (Lib::Rust, true) => "rustfft",
    }
}

/// `convolve` (or `correlate`) of random operands of `la` and `lb` values.
pub fn linear<S: Samp>(o: &mut Out, cfg: &Cfg, en: &mut Engines<S>, la: usize, lb: usize, corr: bool) {
    let mut rng = Rng::new((la * 7919 + lb) as u64);
    let a = S::make(&mut rng, la);
    let b = S::make(&mut rng, lb);
    let n = la + lb - 1;
    let tl = 200.0 * <S::T as P>::EPS * ((n as f64).log2() + 1.0);
    let ours = if corr { correlate(&a, &b) } else { convolve(&a, &b) };
    let want = S::flat(&ours);
    let what = format!("{} {}x{lb}", if corr { "correlate" } else { "convolve" }, la);
    // The references convolve by b reversed and conjugated for a correlation.
    let brc: Vec<S> = b.iter().rev().map(|v| v.cj()).collect();
    let bb: &[S] = if corr { &brc } else { &b };
    let mut out = vec![<S as Samp>::zero(); n];
    let mut libs = Vec::new();
    for lib in [Lib::Fftw, Lib::Rust] {
        let (l, how) = best_len(en, lib, &a, bb, &mut out);
        let e = en.get(lib, l);
        let mut sb = vec![C::<S::T>::ZERO; e.spec().len()];
        conv_ref(e, &mut sb, &a, bb, &mut out);
        let note = check(&what, lib_name::<S>(lib), &want, &S::flat(&out), tl).1;
        libs.push((lib, l, sb, join_notes(&how, &note)));
    }
    let vd = vdsp::available() && (la as f64) * (lb as f64) <= VDSP_MAX_PRODUCTS;
    let mut v = VdspConv::<S>::new(&a, &b, corr && S::CPLX);
    let dir = if corr { 1 } else { -1 };
    let mut vnote = String::new();
    if vd {
        S::vdsp(&mut v, dir);
        vnote = check(&what, "vDSP", &want, &v.out(), tl).1;
    }

    let mut engines: Vec<(Lib, usize, Box<dyn Engine<S>>)> =
        libs.iter().map(|(lib, l, _, _)| (*lib, *l, en.take(*lib, *l))).collect();
    let mut outs: Vec<Vec<S>> = libs.iter().map(|_| vec![<S as Samp>::zero(); n]).collect();
    let mut into_out = vec![<S as Samp>::zero(); n];
    let mut bufs: Vec<Vec<S>> = libs.iter().map(|_| b.clone()).collect();
    {
        let mut race = Race::new();
        let (ra, rb) = (&a, &b);
        let us = race.ours(if corr { "correlate" } else { "convolve" }, move || {
            if corr {
                black_box(correlate(ra, rb));
            } else {
                black_box(convolve(ra, rb));
            }
        });
        let io = &mut into_out;
        let us_into = race.ours(if corr { "correlate_into" } else { "convolve_into" }, move || {
            if corr {
                correlate_into(ra, rb, io);
            } else {
                convolve_into(ra, rb, io);
            }
            black_box(&io);
        });
        let into_group = format!("{}-into/{}", if corr { "correlate" } else { "convolve" }, S::name());
        let mut rows = Vec::new();
        for (((e, (lib, _, sb, note)), out), bw) in engines
            .iter_mut()
            .zip(libs.iter_mut())
            .zip(outs.iter_mut())
            .zip(bufs.iter_mut())
        {
            let e = e.2.as_mut();
            let name = lib_name::<S>(*lib);
            let glue = if corr {
                "transforms by the library, glue (b reversed and conjugated per call, spectral product, overlap-add) by the harness, output preallocated"
            } else {
                "transforms by the library, glue (spectral product, overlap-add) by the harness, output preallocated"
            };
            rows.push(race.reference(name, &join_notes(glue, note), us, move || {
                if corr {
                    for (d, s) in bw.iter_mut().zip(rb.iter().rev()) {
                        *d = s.cj();
                    }
                }
                conv_ref(e, sb, ra, bw, out)
            }));
        }
        if vd {
            let vr = &mut v;
            let what = if S::CPLX {
                if corr {
                    "vDSP_zconv, direct; operands pre-padded and pre-split, b pre-conjugated (not timed)"
                } else {
                    "vDSP_zconv, direct; operands pre-padded and pre-split (not timed)"
                }
            } else {
                "vDSP_conv, direct; a pre-padded with zeros (not timed)"
            };
            rows.push(race.reference("vDSP", &join_notes(what, &vnote), us, move || S::vdsp(vr, dir)));
        }
        for row in rows {
            race.pair(row, us_into, &into_group);
        }
        let group = format!("{}/{}", if corr { "correlate" } else { "convolve" }, S::name());
        race.finish(o, cfg, &group, &format!("T={} la={la} lb={lb}", S::name()), 1.0);
    }
    for (lib, l, e) in engines {
        en.put(lib, l, e);
    }
}

/// `circular_convolve` (or `circular_correlate`) of two random sequences of `n` values.
pub fn circular<S: Samp>(o: &mut Out, cfg: &Cfg, en: &mut Engines<S>, n: usize, corr: bool) {
    let mut rng = Rng::new((n * 31 + 5) as u64);
    let a = S::make(&mut rng, n);
    let b = S::make(&mut rng, n);
    let tl = 200.0 * <S::T as P>::EPS * ((n as f64).log2() + 1.0);
    let ours = if corr { circular_correlate(&a, &b) } else { circular_convolve(&a, &b) };
    let want = S::flat(&ours);
    let what = format!("circular {} n={n}", if corr { "correlate" } else { "convolve" });
    let circ = move |e: &mut dyn Engine<S>, sa: &mut [C<S::T>], a: &[S], b: &[S], out: &mut [S]| {
        let scale = <S::T as P>::of(1.0 / n as f64);
        e.time().copy_from_slice(a);
        e.fwd();
        sa.copy_from_slice(e.spec());
        e.time().copy_from_slice(b);
        e.fwd();
        for (z, w) in e.spec().iter_mut().zip(sa.iter()) {
            let zb = if corr { C::new(z.re, <S::T as P>::of(-z.im.to())) } else { *z };
            let p = cmul(*w, zb);
            *z = C::new(p.re * scale, p.im * scale);
        }
        e.inv();
        out.copy_from_slice(e.time());
    };
    let mut engines = Vec::new();
    for lib in [Lib::Fftw, Lib::Rust] {
        let mut e = en.take(lib, n);
        let mut sa = vec![C::<S::T>::ZERO; e.spec().len()];
        let mut out = vec![<S as Samp>::zero(); n];
        circ(e.as_mut(), &mut sa[..], &a, &b, &mut out[..]);
        let note = check(&what, lib_name::<S>(lib), &want, &S::flat(&out), tl).1;
        engines.push((lib, e, sa, out, note));
    }
    {
        let mut race = Race::new();
        let (ra, rb) = (&a, &b);
        let us = race.ours(if corr { "circular_correlate" } else { "circular_convolve" }, move || {
            if corr {
                black_box(circular_correlate(ra, rb));
            } else {
                black_box(circular_convolve(ra, rb));
            }
        });
        for (lib, e, sa, out, note) in engines.iter_mut() {
            let e = e.as_mut();
            let notes = join_notes(
                "two forward transforms of n and one inverse by the library, product by the harness, output preallocated",
                note,
            );
            race.reference(lib_name::<S>(*lib), &notes, us, move || circ(&mut *e, &mut sa[..], ra, rb, &mut out[..]));
        }
        let group = format!(
            "{}/{}",
            if corr { "circular-correlate" } else { "circular-convolve" },
            S::name()
        );
        race.finish(o, cfg, &group, &format!("T={} n={n}", S::name()), 1.0);
    }
    for (lib, e, ..) in engines {
        en.put(lib, n, e);
    }
}

pub fn run<S: Samp>(o: &mut Out, cfg: &Cfg, part: &str) {
    let mut en = Engines::<S>::new();
    let all = part == "all";
    if all || part == "conv" {
        for &(la, lb) in SIZES {
            linear::<S>(o, cfg, &mut en, la, lb, false);
        }
    }
    if all || part == "corr" {
        for &(la, lb) in CORR {
            linear::<S>(o, cfg, &mut en, la, lb, true);
        }
    }
    if all || part == "circ" {
        for &n in CIRC {
            circular::<S>(o, cfg, &mut en, n, false);
            circular::<S>(o, cfg, &mut en, n, true);
        }
    }
}
