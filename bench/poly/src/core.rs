//! Inputs, their encoding for the shims, the fingerprints of results, and the timing harness.

use a::math::complex::Complex;
use a::math::numbers::{ByteOrder, Number};
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------- the reference libraries

type Prep = unsafe extern "C" fn(*const c_char, *const u64, usize) -> *mut c_void;
type Run = unsafe extern "C" fn(*mut c_void, u64);
type Check = unsafe extern "C" fn(*mut c_void, *mut u64, usize) -> usize;
type Free = unsafe extern "C" fn(*mut c_void);
type Label = unsafe extern "C" fn(*mut c_void) -> *const c_char;

unsafe extern "C" {
    fn sf_prep(op: *const c_char, buf: *const u64, n: usize) -> *mut c_void;
    fn sf_run(h: *mut c_void, reps: u64);
    fn sf_check(h: *mut c_void, out: *mut u64, cap: usize) -> usize;
    fn sf_free(h: *mut c_void);
    fn sf_label(h: *mut c_void) -> *const c_char;
    fn sf_missing(op: *const c_char) -> *const c_char;
    fn sn_prep(op: *const c_char, buf: *const u64, n: usize) -> *mut c_void;
    fn sn_run(h: *mut c_void, reps: u64);
    fn sn_check(h: *mut c_void, out: *mut u64, cap: usize) -> usize;
    fn sn_free(h: *mut c_void);
    fn sn_label(h: *mut c_void) -> *const c_char;
    fn sp_prep(op: *const c_char, buf: *const u64, n: usize) -> *mut c_void;
    fn sp_run(h: *mut c_void, reps: u64);
    fn sp_check(h: *mut c_void, out: *mut u64, cap: usize) -> usize;
    fn sp_free(h: *mut c_void);
    fn sp_label(h: *mut c_void) -> *const c_char;
    fn sl_prep(op: *const c_char, buf: *const u64, n: usize) -> *mut c_void;
    fn sl_run(h: *mut c_void, reps: u64);
    fn sl_check(h: *mut c_void, out: *mut u64, cap: usize) -> usize;
    fn sl_free(h: *mut c_void);
    fn sl_label(h: *mut c_void) -> *const c_char;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lib {
    Flint,
    Ntl,
    Pari,
    Lapack,
}

impl Lib {
    fn fns(self) -> (Prep, Run, Check, Free, Label) {
        match self {
            Lib::Flint => (sf_prep, sf_run, sf_check, sf_free, sf_label),
            Lib::Ntl => (sn_prep, sn_run, sn_check, sn_free, sn_label),
            Lib::Pari => (sp_prep, sp_run, sp_check, sp_free, sp_label),
            Lib::Lapack => (sl_prep, sl_run, sl_check, sl_free, sl_label),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Lib::Flint => "FLINT",
            Lib::Ntl => "NTL",
            Lib::Pari => "PARI",
            Lib::Lapack => {
                if cfg!(target_os = "macos") {
                    "Accelerate"
                } else {
                    "LAPACK"
                }
            }
        }
    }
    /// Why the installed library cannot do `op` although the shim knows it (an older version
    /// without the function), or None.
    pub fn missing(self, op: &str) -> Option<String> {
        if self != Lib::Flint || !self.available() {
            return None;
        }
        let c = CString::new(op).unwrap();
        // SAFETY: a NUL-terminated name; the result is null or a static C string.
        let p = unsafe { sf_missing(c.as_ptr()) };
        if p.is_null() {
            None
        } else {
            // SAFETY: non-null, NUL-terminated.
            Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        }
    }
    pub fn available(self) -> bool {
        env!("POLYBENCH_REFS").split(',').any(|r| {
            r == self.name() || (self == Lib::Lapack && (r == "Accelerate" || r == "LAPACK"))
        })
    }
}

/// Something timed: ours or a reference.
pub trait Contender {
    /// `reps` calls, the input variants in turn.
    fn run(&mut self, reps: u64);
    /// One call on each variant, its outputs (the layouts of shims/shim.h).
    fn check(&mut self) -> Vec<u64>;
}

pub struct Shim {
    h: *mut c_void,
    run: Run,
    check: Check,
    free: Free,
    pub label: String,
}

impl Shim {
    pub fn new(lib: Lib, op: &str, buf: &[u64]) -> Option<Shim> {
        if !lib.available() {
            return None;
        }
        let (prep, run, check, free, label) = lib.fns();
        let c = CString::new(op).unwrap();
        // SAFETY: the shims read `buf` (n words) and return a handle or null.
        let h = unsafe { prep(c.as_ptr(), buf.as_ptr(), buf.len()) };
        if h.is_null() {
            return None;
        }
        // SAFETY: a live handle; the label is a static or owned C string.
        let l = unsafe { CStr::from_ptr(label(h)) }.to_string_lossy().into_owned();
        Some(Shim { h, run, check, free, label: l })
    }
}

impl Contender for Shim {
    fn run(&mut self, reps: u64) {
        // SAFETY: a live handle.
        unsafe { (self.run)(self.h, reps) }
    }
    fn check(&mut self) -> Vec<u64> {
        let mut cap = 1 << 16;
        loop {
            let mut out = vec![0u64; cap];
            // SAFETY: `out` has `cap` words; the shim writes at most `cap` and returns the count.
            let n = unsafe { (self.check)(self.h, out.as_mut_ptr(), cap) };
            if n <= cap {
                out.truncate(n);
                return out;
            }
            cap = n;
        }
    }
}

impl Drop for Shim {
    fn drop(&mut self) {
        // SAFETY: the handle is freed once.
        unsafe { (self.free)(self.h) }
    }
}

/// Ours (or a reference written in Rust): one call on a variant, and its output.
pub struct Native<S, F, G> {
    pub s: S,
    pub v: usize,
    pub one: F,
    pub out: G,
}

impl<S, F: FnMut(&S, usize), G: FnMut(&S, usize) -> Vec<u64>> Contender for Native<S, F, G> {
    fn run(&mut self, reps: u64) {
        let v = self.v as u64;
        for r in 0..reps {
            (self.one)(&self.s, (r % v) as usize);
        }
    }
    fn check(&mut self) -> Vec<u64> {
        let mut o = Vec::new();
        for i in 0..self.v {
            o.extend((self.out)(&self.s, i));
        }
        o
    }
}

pub fn native<S: 'static>(
    s: S,
    v: usize,
    one: impl FnMut(&S, usize) + 'static,
    out: impl FnMut(&S, usize) -> Vec<u64> + 'static,
) -> Box<dyn Contender> {
    Box::new(Native { s, v, one, out })
}

// ---------------------------------------------------------------- random inputs

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return self.next();
        }
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    /// A random signed integer of exactly `bits` bits (top bit set), random sign.
    pub fn int(&mut self, bits: u32) -> Big {
        let mut b = self.uint(bits);
        b.neg = self.next() & 1 == 1;
        b
    }
    /// A random non-negative integer of exactly `bits` bits.
    pub fn uint(&mut self, bits: u32) -> Big {
        let nl = bits.div_ceil(64) as usize;
        let mut l: Vec<u64> = (0..nl).map(|_| self.next()).collect();
        let top = bits - 64 * (nl as u32 - 1);
        if top < 64 {
            l[nl - 1] &= (1u64 << top) - 1;
        }
        l[nl - 1] |= 1u64 << (top - 1);
        Big { neg: false, l }
    }
    /// A random residue below the multi-limb `n`.
    pub fn below_big(&mut self, n: &Big) -> Big {
        let nl = n.l.len();
        loop {
            let mut l: Vec<u64> = (0..nl).map(|_| self.next()).collect();
            let top_bits = 64 - n.l[nl - 1].leading_zeros();
            if top_bits < 64 {
                l[nl - 1] &= (1u64 << top_bits) - 1;
            }
            let b = Big { neg: false, l };
            if b.cmp_abs(n) == std::cmp::Ordering::Less {
                return b.norm();
            }
        }
    }
}

/// A signed integer as limbs, little-endian.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Big {
    pub neg: bool,
    pub l: Vec<u64>,
}

impl Big {
    pub fn from_u64(x: u64) -> Big {
        Big { neg: false, l: vec![x] }.norm()
    }
    pub fn from_i64(x: i64) -> Big {
        Big { neg: x < 0, l: vec![x.unsigned_abs()] }.norm()
    }
    pub fn norm(mut self) -> Big {
        while self.l.last() == Some(&0) {
            self.l.pop();
        }
        if self.l.is_empty() {
            self.neg = false;
        }
        self
    }
    pub fn cmp_abs(&self, o: &Big) -> std::cmp::Ordering {
        let a = self.clone().norm();
        let b = o.clone().norm();
        a.l.len().cmp(&b.l.len()).then_with(|| a.l.iter().rev().cmp(b.l.iter().rev()))
    }
    /// 2^k - c.
    pub fn pow2_minus(k: u32, c: u64) -> Big {
        let n = Number::from(1u64) << k as usize;
        from_number(&(&n - &Number::from(c)))
    }
    pub fn to_number(&self) -> Number {
        let mut bytes = Vec::with_capacity(self.l.len() * 8);
        for w in &self.l {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        if bytes.is_empty() {
            bytes.push(0);
        }
        Number::new_signed(bytes, self.neg, ByteOrder::LittleEndian)
    }
    fn enc(&self, o: &mut Vec<u64>) {
        let b = self.clone().norm();
        o.push(b.neg as u64);
        o.push(b.l.len() as u64);
        o.extend_from_slice(&b.l);
    }
}

pub fn from_number(n: &Number) -> Big {
    let mut bytes = n.value.clone();
    if n.byte_order == ByteOrder::BigEndian {
        bytes.reverse();
    }
    let mut l = vec![0u64; bytes.len().div_ceil(8)];
    for (i, b) in bytes.iter().enumerate() {
        l[i / 8] |= (*b as u64) << (8 * (i % 8));
    }
    Big { neg: n.is_negative(), l }.norm()
}

/// An argument for the shims.
#[derive(Clone, Debug)]
pub enum Arg {
    U(u64),
    W(Vec<u64>),
    I(Big),
    IV(Vec<Big>),
    F(Vec<f64>),
    Q(Vec<(Big, Big)>),
}

impl Arg {
    fn enc(&self, o: &mut Vec<u64>) {
        match self {
            Arg::U(x) => o.extend([1, *x]),
            Arg::W(v) => {
                o.extend([2, v.len() as u64]);
                o.extend_from_slice(v);
            }
            Arg::I(b) => {
                o.push(3);
                b.enc(o);
            }
            Arg::IV(v) => {
                o.extend([4, v.len() as u64]);
                for b in v {
                    b.enc(o);
                }
            }
            Arg::F(v) => {
                o.extend([5, v.len() as u64]);
                o.extend(v.iter().map(|x| x.to_bits()));
            }
            Arg::Q(v) => {
                o.extend([6, v.len() as u64]);
                for (n, d) in v {
                    n.enc(o);
                    d.enc(o);
                }
            }
        }
    }
    pub fn w(&self) -> &[u64] {
        match self {
            Arg::W(v) => v,
            _ => panic!("not words"),
        }
    }
    pub fn iv(&self) -> &[Big] {
        match self {
            Arg::IV(v) => v,
            _ => panic!("not integers"),
        }
    }
    pub fn q(&self) -> &[(Big, Big)] {
        match self {
            Arg::Q(v) => v,
            _ => panic!("not rationals"),
        }
    }
    pub fn big(&self) -> Big {
        match self {
            Arg::I(b) => b.clone(),
            Arg::U(x) => Big::from_u64(*x),
            _ => panic!("not an integer"),
        }
    }
    pub fn u(&self) -> u64 {
        match self {
            Arg::U(x) => *x,
            _ => panic!("not a word"),
        }
    }
}

pub fn encode(shared: &[Arg], vars: &[Vec<Arg>]) -> Vec<u64> {
    let mut o = vec![shared.len() as u64];
    for a in shared {
        a.enc(&mut o);
    }
    o.push(vars.len() as u64);
    o.push(vars.first().map_or(0, |v| v.len()) as u64);
    for v in vars {
        for a in v {
            a.enc(&mut o);
        }
    }
    o
}

// ---------------------------------------------------------------- conversions to ours

pub fn zpoly(v: &[Big]) -> Polynomial<Number> {
    Polynomial::new(v.iter().map(Big::to_number).collect())
}
pub fn numbers(v: &[Big]) -> Vec<Number> {
    v.iter().map(Big::to_number).collect()
}
pub fn qpoly(v: &[(Big, Big)]) -> Polynomial<Rational> {
    Polynomial::new(v.iter().map(|(n, d)| Rational::new(n.to_number(), d.to_number())).collect())
}
pub fn rationals_int(v: &[Big]) -> Vec<Rational> {
    v.iter().map(|b| Rational::from_integer(b.to_number())).collect()
}
pub fn fpoly(v: &[f64]) -> Polynomial<f64> {
    Polynomial::new(v.to_vec())
}
pub fn cpoly(v: &[f64]) -> Polynomial<Complex<f64>> {
    Polynomial::new(v.chunks(2).map(|c| Complex { re: c[0], im: c[1] }).collect())
}

// ---------------------------------------------------------------- fingerprints

pub const FP: u64 = (1 << 61) - 1;
pub const FP_R: u64 = 0x0123_4567_89AB_CDEF;

pub fn fp_red(x: u128) -> u64 {
    let lo = (x as u64) & FP;
    let hi = (x >> 61) as u64;
    let mut s = lo + (hi & FP) + (hi >> 61);
    while s >= FP {
        s -= FP;
    }
    s
}
pub fn fp_mul(a: u64, b: u64) -> u64 {
    fp_red(a as u128 * b as u128)
}
pub fn fp_add(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= FP { s - FP } else { s }
}
pub fn fp_mix(h: u64) -> u64 {
    fp_add(fp_mul(h, h), fp_mul(h, 0x2545_F491_4F6C_DD1D % FP))
}
pub fn fp_inv(a: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u64, a, FP - 2);
    while e > 0 {
        if e & 1 == 1 {
            r = fp_mul(r, b);
        }
        b = fp_mul(b, b);
        e >>= 1;
    }
    r
}
pub fn fp_number(n: &Number) -> u64 {
    let mut h = 0u64;
    let it: Box<dyn Iterator<Item = &u8>> = match n.byte_order {
        ByteOrder::BigEndian => Box::new(n.value.iter()),
        ByteOrder::LittleEndian => Box::new(n.value.iter().rev()),
    };
    for &b in it {
        h = fp_add(fp_mul(h, 256), b as u64);
    }
    if n.is_negative() && h != 0 { FP - h } else { h }
}
pub fn fp_rational(q: &Rational) -> u64 {
    fp_mul(fp_number(q.numer()), fp_inv(fp_number(q.denom())))
}
pub fn fp_words(c: &[u64]) -> u64 {
    let mut h = 0u64;
    for &x in c.iter().rev() {
        h = fp_add(fp_mul(h, FP_R), x % FP);
    }
    h
}
pub fn fp_nums(c: &[Number]) -> u64 {
    let mut h = 0u64;
    for x in c.iter().rev() {
        h = fp_add(fp_mul(h, FP_R), fp_number(x));
    }
    h
}
pub fn out_mod(p: &ModPolynomial) -> [u64; 2] {
    let h = match p.word_coeffs() {
        Some(w) => fp_words(w),
        None => fp_nums(&p.coeffs()),
    };
    [p.length() as u64, h]
}
pub fn out_z(p: &Polynomial<Number>) -> [u64; 2] {
    [p.length() as u64, fp_nums(p.coeffs())]
}
pub fn out_q(p: &Polynomial<Rational>) -> [u64; 2] {
    let mut h = 0u64;
    for x in p.coeffs().iter().rev() {
        h = fp_add(fp_mul(h, FP_R), fp_rational(x));
    }
    [p.length() as u64, h]
}
pub fn out_roots(z: &[Complex<f64>]) -> Vec<u64> {
    let mut v: Vec<(f64, f64)> = z.iter().map(|c| (c.re, c.im)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut o = vec![v.len() as u64];
    for (r, i) in v {
        o.push(r.to_bits());
        o.push(i.to_bits());
    }
    o
}
pub fn out_fac<P>(lc: u64, fs: &[(P, u32)], fp: impl Fn(&P) -> (u64, u64)) -> Vec<u64> {
    let (mut sdeg, mut mix) = (0u64, 0u64);
    for (f, e) in fs {
        let (len, h) = fp(f);
        sdeg += *e as u64 * (len - 1);
        mix = fp_add(mix, fp_mul(*e as u64, fp_mix(h)));
    }
    vec![fs.len() as u64, sdeg, mix, lc]
}

// ---------------------------------------------------------------- comparison

#[derive(Clone)]
pub enum Cmp {
    /// The words must agree.
    Exact,
    /// As exact, but a word the reference leaves 0 is not compared (multiplicities it does not
    /// give).
    RefZeroWild,
    /// Complex roots [n, re, im, ...] per variant, matched within `tol · max(1, |z|)`.
    Roots(f64),
    /// Real values [n, x, ...] per variant, matched within `tol · max(1, |x|)`.
    Reals(f64),
    /// Float coefficient vectors [n, bits...] per variant within `tol` of the largest.
    Floats(f64),
}

fn split_counted(v: &[u64], per: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let n = v[i] as usize;
        out.push(v[i + 1..i + 1 + n * per].iter().map(|&b| f64::from_bits(b)).collect());
        i += 1 + n * per;
    }
    out
}

/// None when they agree, else a short description of the difference.
pub fn compare(cmp: &Cmp, ours: &[u64], refv: &[u64]) -> Option<String> {
    match cmp {
        Cmp::Exact => {
            if ours == refv {
                None
            } else {
                let i = ours.iter().zip(refv).position(|(a, b)| a != b).unwrap_or(ours.len().min(refv.len()));
                Some(format!("words differ at {i} of {}/{}", ours.len(), refv.len()))
            }
        }
        Cmp::RefZeroWild => {
            if ours.len() == refv.len() && ours.iter().zip(refv).all(|(a, b)| *b == 0 || a == b) {
                None
            } else {
                Some("words differ".into())
            }
        }
        Cmp::Roots(tol) | Cmp::Reals(tol) => {
            let per = if matches!(cmp, Cmp::Roots(_)) { 2 } else { 1 };
            let a = split_counted(ours, per);
            let b = split_counted(refv, per);
            if a.len() != b.len() {
                return Some("variant count".into());
            }
            let mut worst = 0f64;
            for (x, y) in a.iter().zip(&b) {
                if x.len() != y.len() {
                    return Some(format!("{} roots against {}", x.len() / per, y.len() / per));
                }
                let n = x.len() / per;
                let mut used = vec![false; n];
                for i in 0..n {
                    let (xr, xi) = (x[per * i], if per == 2 { x[per * i + 1] } else { 0.0 });
                    let mut best = f64::INFINITY;
                    let mut bj = 0;
                    for j in 0..n {
                        if used[j] {
                            continue;
                        }
                        let (yr, yi) = (y[per * j], if per == 2 { y[per * j + 1] } else { 0.0 });
                        let d = ((xr - yr).powi(2) + (xi - yi).powi(2)).sqrt();
                        if d < best {
                            best = d;
                            bj = j;
                        }
                    }
                    used[bj] = true;
                    let scale = (xr * xr + xi * xi).sqrt().max(1.0);
                    worst = worst.max(best / scale);
                }
            }
            if worst <= *tol { None } else { Some(format!("roots differ by {worst:.1e}")) }
        }
        Cmp::Floats(tol) => {
            let a = split_counted(ours, 1);
            let b = split_counted(refv, 1);
            if a.len() != b.len() {
                return Some("variant count".into());
            }
            let mut worst = 0f64;
            for (x, y) in a.iter().zip(&b) {
                if x.len() != y.len() {
                    return Some("lengths".into());
                }
                let m = x.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-300);
                for (p, q) in x.iter().zip(y) {
                    worst = worst.max((p - q).abs() / m);
                }
            }
            if worst <= *tol { None } else { Some(format!("coefficients differ by {worst:.1e}")) }
        }
    }
}

// ---------------------------------------------------------------- the timing

pub struct Ref {
    pub label: String,
    pub c: Box<dyn Contender>,
    pub note: String,
}

pub struct Case {
    pub group: &'static str,
    pub op: String,
    pub params: String,
    pub ours: Box<dyn Contender>,
    pub refs: Vec<Ref>,
    pub cmp: Cmp,
    pub notes: String,
    /// Extra notes per reference from the check (e.g. backward errors).
    pub check_note: Option<Box<dyn Fn(&[u64], &[u64]) -> String>>,
}

pub struct Out {
    pub w: Box<dyn Write>,
    pub filter: Vec<String>,
    /// Smoke runs (POLYBENCH_QUICK=maxlen): 3 samples, sizes above maxlen skipped.
    pub quick: Option<usize>,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}
fn spread(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let (q1, q3) = (s[n / 4], s[(3 * n) / 4]);
    let m = median(&mut s);
    (q3 - q1) / m
}

/// Calls for a sample of at least 1 ms, and the time of one call.
fn calibrate(c: &mut dyn Contender) -> (u64, f64) {
    let mut reps = 1u64;
    loop {
        let t = Instant::now();
        c.run(reps);
        let e = t.elapsed().as_secs_f64();
        if e >= 1.0e-3 {
            let per = e / reps as f64;
            let r = ((1.2e-3 / per).ceil() as u64).max(1);
            return (r, per);
        }
        let grow = if e > 0.0 { ((1.5e-3 / e).ceil() as u64).clamp(2, 1 << 20) } else { 1 << 10 };
        reps = reps.saturating_mul(grow);
    }
}

fn fmt_ns(x: f64) -> String {
    if x >= 100.0 { format!("{x:.0}") } else if x >= 10.0 { format!("{x:.1}") } else { format!("{x:.2}") }
}
fn fmt_ratio(r: f64) -> String {
    if r >= 100.0 {
        format!("{r:.0}")
    } else if r >= 10.0 {
        format!("{r:.1}")
    } else if r >= 1.0 {
        format!("{r:.2}")
    } else {
        format!("{r:.3}")
    }
}

impl Out {
    /// Whether a size is skipped in a smoke run.
    pub fn skip(&self, len: usize) -> bool {
        self.quick.is_some_and(|m| len > m)
    }

    pub fn wants(&self, group: &str) -> bool {
        self.filter.is_empty() || self.filter.iter().any(|f| f == group)
    }

    pub fn case(&mut self, mut c: Case) {
        // POLYBENCH_MATCH: only the cases whose "operation params" contain one of the
        // '|'-separated strings (one ending in '$': end with it).
        if let Ok(m) = std::env::var("POLYBENCH_MATCH") {
            let key = format!("{} {}", c.op, c.params);
            let hit = |p: &str| match p.strip_suffix('$') {
                Some(e) => key.ends_with(e),
                None => key.contains(p),
            };
            if !m.split('|').any(hit) {
                return;
            }
        }
        // Same work: compare every reference's outputs with ours before timing.
        let ours_out = c.ours.check();
        let mut keep = Vec::new();
        for (i, r) in c.refs.iter_mut().enumerate() {
            let ro = r.c.check();
            match compare(&c.cmp, &ours_out, &ro) {
                None => {}
                Some(d) => {
                    eprintln!("MISMATCH {} {} {} vs {}: {d}", c.group, c.op, c.params, r.label);
                    r.note = format!("MISMATCH ({d}); {}", r.note);
                }
            }
            if let Some(f) = &c.check_note {
                let n = f(&ours_out, &ro);
                if !n.is_empty() {
                    r.note = if r.note.is_empty() { n } else { format!("{}; {n}", r.note) };
                }
            }
            keep.push(i);
        }
        if let Some(secs) = std::env::var("POLYBENCH_PROFILE").ok().and_then(|v| v.parse::<f64>().ok()) {
            // Ours alone for `secs` seconds (to sample its stacks), then the next case.
            let t = Instant::now();
            while t.elapsed().as_secs_f64() < secs {
                c.ours.run(16);
            }
            eprintln!("PROFILED {} {}", c.op, c.params);
            return;
        }
        if std::env::var("POLYBENCH_CHECK").is_ok() {
            // Check only (no timing): one line per case.
            let bad = c.refs.iter().filter(|r| r.note.contains("MISMATCH")).count();
            let names: Vec<&str> = c.refs.iter().map(|r| r.label.as_str()).collect();
            let line = format!("CHECK\t{}\t{}\t{}\t{} refs, {bad} mismatched\t{}\n", c.group, c.op, c.params, c.refs.len(), names.join(" | "));
            self.emit(&line);
            return;
        }
        // Calibrate everyone (this is also the warm-up).
        let mut all: Vec<&mut Box<dyn Contender>> = Vec::new();
        all.push(&mut c.ours);
        for r in c.refs.iter_mut() {
            all.push(&mut r.c);
        }
        let mut reps = Vec::new();
        let mut round = 0.0;
        for x in all.iter_mut() {
            let (r, per) = calibrate(x.as_mut());
            reps.push(r);
            round += r as f64 * per;
        }
        // POLYBENCH_SAMPLES raises the 15 samples of the cheap cases (A/B decisions take 21).
        let base: usize = std::env::var("POLYBENCH_SAMPLES").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
        let samples = if self.quick.is_some() {
            3
        } else if round <= 0.25 {
            base
        } else if round <= 2.0 {
            ((4.0 / round).ceil() as usize).clamp(5, 15)
        } else {
            3
        };
        let k = all.len();
        let mut t: Vec<Vec<f64>> = vec![Vec::new(); k];
        for s in 0..samples {
            for j in 0..k {
                let i = (j + s) % k;
                let st = Instant::now();
                all[i].run(reps[i]);
                t[i].push(st.elapsed().as_nanos() as f64 / reps[i] as f64);
            }
        }
        let mut ours_t = t[0].clone();
        let ours_ns = median(&mut ours_t);
        let ours_sp = spread(&t[0]);
        let mut base_note = c.notes.clone();
        if samples < 15 {
            base_note = join(&base_note, &format!("{samples} samples"));
        }
        if ours_sp > 0.10 {
            base_note = join(&base_note, &format!("spread ours {:.0}%", ours_sp * 100.0));
        }
        if c.refs.is_empty() {
            let line = format!(
                "poly\t{}\t{}\t{}\t{}\t-\t-\t-\t{}\n",
                c.group, c.op, c.params, fmt_ns(ours_ns), base_note
            );
            self.emit(&line);
        }
        for (j, r) in c.refs.iter().enumerate() {
            let mut rt = t[j + 1].clone();
            let rn = median(&mut rt);
            let rsp = spread(&t[j + 1]);
            let mut note = join(&base_note, &r.note);
            if rsp > 0.10 {
                note = join(&note, &format!("spread ref {:.0}%", rsp * 100.0));
            }
            let line = format!(
                "poly\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                c.group,
                c.op,
                c.params,
                fmt_ns(ours_ns),
                r.label,
                fmt_ns(rn),
                fmt_ratio(ours_ns / rn),
                note
            );
            self.emit(&line);
        }
    }

    fn emit(&mut self, line: &str) {
        self.w.write_all(line.as_bytes()).unwrap();
        self.w.flush().unwrap();
        eprint!("{line}");
    }
}

pub fn join(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ => format!("{a}; {b}"),
    }
}

/// The references for an operation, each that supports it.
pub fn shims(libs: &[(Lib, &str, &str)], buf: &[u64]) -> Vec<Ref> {
    let mut v = Vec::new();
    for (lib, op, func) in libs {
        if let Some(s) = Shim::new(*lib, op, buf) {
            let func = if s.label == "F2x" { func.replace("Flx", "F2x") } else { func.to_string() };
            let label = if s.label.is_empty() {
                format!("{} {}", lib.name(), func)
            } else {
                format!("{} {} ({})", lib.name(), func, s.label)
            };
            v.push(Ref { label, c: Box::new(s), note: String::new() });
        }
    }
    v
}

/// The notes of the references that the installed versions cannot run (`Lib::missing`).
pub fn missing_notes(libs: &[(Lib, &str, &str)]) -> String {
    let mut n = String::new();
    for (lib, op, _) in libs {
        if let Some(m) = lib.missing(op) {
            n = join(&n, &m);
        }
    }
    n
}

/// Number of input variants for a size: fresh inputs per repetition where they matter.
pub fn variants(len: usize) -> usize {
    if len <= 64 {
        8
    } else if len <= 2000 {
        2
    } else {
        1
    }
}
