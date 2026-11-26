// Bindings to the C drivers in c/gmpflint.c and c/pari.c, and safe pool wrappers.
#![allow(dead_code)]
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};

use crate::data::Q;

pub const ONE_NONE: usize = usize::MAX;

unsafe extern "C" {
    fn gq_new(n: usize) -> *mut c_void;
    fn gq_free(p: *mut c_void);
    fn gq_set(p: *mut c_void, i: usize, nb: *const u8, nl: usize, neg: c_int, db: *const u8, dl: usize, canon: c_int);
    fn gq_set_si(p: *mut c_void, i: usize, x: c_long);
    fn gq_set_d(p: *mut c_void, i: usize, x: f64);
    fn gq_set_s(p: *mut c_void, i: usize, s: *const c_char);
    fn gq_reserve_buf(p: *mut c_void, cap: usize);
    fn gq_str(p: *mut c_void, i: usize, which: c_int, base: c_int, buf: *mut c_char, cap: usize) -> usize;
    fn gq_bin(op: c_int, a: *mut c_void, b: *mut c_void, count: usize, one: usize) -> i64;
    fn gq_un(op: c_int, a: *mut c_void, count: usize, one: usize, e: c_long) -> i64;
    fn gq_sum(a: *mut c_void, mode: c_int, count: usize);
}

#[cfg(has_flint)]
unsafe extern "C" {
    fn fq_new(n: usize) -> *mut c_void;
    fn fq_free(p: *mut c_void);
    fn fq_set(p: *mut c_void, i: usize, nb: *const u8, nl: usize, neg: c_int, db: *const u8, dl: usize, canon: c_int);
    fn fq_set_si(p: *mut c_void, i: usize, x: c_long);
    fn fq_set_d(p: *mut c_void, i: usize, x: f64);
    fn fq_set_s(p: *mut c_void, i: usize, s: *const c_char);
    fn fq_reserve_buf(p: *mut c_void, cap: usize);
    fn fq_prepare_cfrac(p: *mut c_void, i: usize);
    fn fq_str(p: *mut c_void, i: usize, which: c_int, base: c_int, buf: *mut c_char, cap: usize) -> usize;
    fn fq_bin(op: c_int, a: *mut c_void, b: *mut c_void, count: usize, one: usize) -> i64;
    fn fq_recon2(a: *mut c_void, b: *mut c_void, c: *mut c_void, count: usize, one: usize) -> i64;
    fn fq_un(op: c_int, a: *mut c_void, count: usize, one: usize, e: c_long) -> i64;
    fn fq_sum(a: *mut c_void, mode: c_int, count: usize);
    fn fl_seq(a: *mut c_void, kind: c_int, n0: u64, step: u64, count: usize);
    fn fl_bvec(a: *mut c_void, kind: c_int, n: u64, count: usize);
    fn fm_new(k: usize, n: usize, m: usize) -> *mut c_void;
    fn fm_free(p: *mut c_void);
    fn fm_set(p: *mut c_void, which: c_int, t: usize, i: usize, j: usize, nb: *const u8, nl: usize, neg: c_int, db: *const u8, dl: usize);
    fn fm_run(p: *mut c_void, op: c_int, count: usize, one: usize) -> i64;
    fn fm_str(p: *mut c_void, op: c_int, i: usize, j: usize, buf: *mut c_char, cap: usize) -> usize;
}

#[cfg(has_pari)]
unsafe extern "C" {
    fn pa_init();
    fn pp_new(n: usize) -> *mut c_void;
    fn pp_free(p: *mut c_void);
    fn pp_set(p: *mut c_void, i: usize, nb: *const u8, nl: usize, neg: c_int, db: *const u8, dl: usize);
    fn pp_run(op: c_int, a: *mut c_void, b: *mut c_void, count: usize, one: usize) -> i64;
    fn pp_seq(a: *mut c_void, kind: c_int, n0: u64, step: u64, count: usize);
    fn pp_bvec(a: *mut c_void, m: c_long, count: usize);
    fn pp_str(p: *mut c_void, idx: c_long, buf: *mut c_char, cap: usize) -> usize;
    fn pm_new(k: usize, n: usize, m: usize) -> *mut c_void;
    fn pm_free(p: *mut c_void);
    fn pm_begin(p: *mut c_void);
    fn pm_entry(which: c_int, i: usize, j: usize, nb: *const u8, nl: usize, neg: c_int, db: *const u8, dl: usize);
    fn pm_end(p: *mut c_void, t: usize);
    fn pm_run(p: *mut c_void, op: c_int, count: usize, one: usize) -> i64;
    fn pm_str(p: *mut c_void, op: c_int, i: usize, j: usize, buf: *mut c_char, cap: usize) -> usize;
}

fn take(buf: &[u8], len: usize) -> String {
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

const CAP: usize = 1 << 26;

thread_local! {
    static BUF: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0u8; CAP]);
}

fn with_buf(f: impl FnOnce(*mut c_char, usize) -> usize) -> String {
    BUF.with(|b| {
        let mut b = b.borrow_mut();
        let l = f(b.as_mut_ptr() as *mut c_char, CAP);
        take(&b, l)
    })
}

// ---------------------------------------------------------------------------------------
// GMP
// ---------------------------------------------------------------------------------------

pub struct Gmp(*mut c_void, pub usize);

impl Gmp {
    pub fn new(qs: &[Q], canon: bool) -> Gmp {
        let p = unsafe { gq_new(qs.len()) };
        for (i, q) in qs.iter().enumerate() {
            unsafe { gq_set(p, i, q.num.as_ptr(), q.num.len(), q.neg as c_int, q.den.as_ptr(), q.den.len(), canon as c_int) };
        }
        Gmp(p, qs.len())
    }
    pub fn set_si(&self, i: usize, x: i64) {
        unsafe { gq_set_si(self.0, i, x as c_long) }
    }
    pub fn set_d(&self, i: usize, x: f64) {
        unsafe { gq_set_d(self.0, i, x) }
    }
    pub fn set_s(&self, i: usize, s: &str) {
        let c = CString::new(s).unwrap();
        unsafe { gq_set_s(self.0, i, c.as_ptr()) }
    }
    pub fn reserve(&self, cap: usize) {
        unsafe { gq_reserve_buf(self.0, cap) }
    }
    /// which: 0 item, 1 result, 2 integer result.
    pub fn str(&self, i: usize, which: i32, base: i32) -> String {
        with_buf(|b, c| unsafe { gq_str(self.0, i, which, base, b, c) })
    }
    pub fn bin(&self, op: i32, b: &Gmp, count: usize) -> i64 {
        unsafe { gq_bin(op, self.0, b.0, count, ONE_NONE) }
    }
    pub fn bin_one(&self, op: i32, b: &Gmp, i: usize) -> i64 {
        unsafe { gq_bin(op, self.0, b.0, 1, i) }
    }
    pub fn un(&self, op: i32, count: usize, e: i64) -> i64 {
        unsafe { gq_un(op, self.0, count, ONE_NONE, e as c_long) }
    }
    pub fn un_one(&self, op: i32, i: usize, e: i64) -> i64 {
        unsafe { gq_un(op, self.0, 1, i, e as c_long) }
    }
    pub fn sum(&self, mode: i32, count: usize) {
        unsafe { gq_sum(self.0, mode, count) }
    }
}

impl Drop for Gmp {
    fn drop(&mut self) {
        unsafe { gq_free(self.0) }
    }
}

// ---------------------------------------------------------------------------------------
// FLINT
// ---------------------------------------------------------------------------------------

#[cfg(has_flint)]
pub struct Flint(*mut c_void, pub usize);

#[cfg(has_flint)]
impl Flint {
    pub fn new(qs: &[Q], canon: bool) -> Flint {
        let p = unsafe { fq_new(qs.len()) };
        for (i, q) in qs.iter().enumerate() {
            unsafe { fq_set(p, i, q.num.as_ptr(), q.num.len(), q.neg as c_int, q.den.as_ptr(), q.den.len(), canon as c_int) };
        }
        Flint(p, qs.len())
    }
    pub fn set_si(&self, i: usize, x: i64) {
        unsafe { fq_set_si(self.0, i, x as c_long) }
    }
    pub fn set_d(&self, i: usize, x: f64) {
        unsafe { fq_set_d(self.0, i, x) }
    }
    pub fn set_s(&self, i: usize, s: &str) {
        let c = CString::new(s).unwrap();
        unsafe { fq_set_s(self.0, i, c.as_ptr()) }
    }
    pub fn reserve(&self, cap: usize) {
        unsafe { fq_reserve_buf(self.0, cap) }
    }
    pub fn prepare_cfrac(&self, i: usize) {
        unsafe { fq_prepare_cfrac(self.0, i) }
    }
    pub fn str(&self, i: usize, which: i32, base: i32) -> String {
        with_buf(|b, c| unsafe { fq_str(self.0, i, which, base, b, c) })
    }
    pub fn bin(&self, op: i32, b: &Flint, count: usize) -> i64 {
        unsafe { fq_bin(op, self.0, b.0, count, ONE_NONE) }
    }
    pub fn bin_one(&self, op: i32, b: &Flint, i: usize) -> i64 {
        unsafe { fq_bin(op, self.0, b.0, 1, i) }
    }
    pub fn recon2(&self, b: &Flint, c: &Flint, count: usize) -> i64 {
        unsafe { fq_recon2(self.0, b.0, c.0, count, ONE_NONE) }
    }
    pub fn recon2_one(&self, b: &Flint, c: &Flint, i: usize) -> i64 {
        unsafe { fq_recon2(self.0, b.0, c.0, 1, i) }
    }
    pub fn un(&self, op: i32, count: usize, e: i64) -> i64 {
        unsafe { fq_un(op, self.0, count, ONE_NONE, e as c_long) }
    }
    pub fn un_one(&self, op: i32, i: usize, e: i64) -> i64 {
        unsafe { fq_un(op, self.0, 1, i, e as c_long) }
    }
    pub fn sum(&self, mode: i32, count: usize) {
        unsafe { fq_sum(self.0, mode, count) }
    }
    pub fn seq(&self, kind: i32, n0: u64, step: u64, count: usize) {
        unsafe { fl_seq(self.0, kind, n0, step, count) }
    }
    pub fn bvec(&self, kind: i32, n: u64, count: usize) {
        unsafe { fl_bvec(self.0, kind, n, count) }
    }
}

#[cfg(has_flint)]
impl Drop for Flint {
    fn drop(&mut self) {
        unsafe { fq_free(self.0) }
    }
}

#[cfg(has_flint)]
pub struct FlintMat(*mut c_void);

#[cfg(has_flint)]
impl FlintMat {
    pub fn new(k: usize, n: usize, m: usize) -> FlintMat {
        FlintMat(unsafe { fm_new(k, n, m) })
    }
    pub fn set(&self, which: i32, t: usize, i: usize, j: usize, q: &Q) {
        unsafe { fm_set(self.0, which, t, i, j, q.num.as_ptr(), q.num.len(), q.neg as c_int, q.den.as_ptr(), q.den.len()) }
    }
    pub fn run(&self, op: i32, count: usize) -> i64 {
        unsafe { fm_run(self.0, op, count, ONE_NONE) }
    }
    pub fn run_one(&self, op: i32, t: usize) -> i64 {
        unsafe { fm_run(self.0, op, 1, t) }
    }
    pub fn str(&self, op: i32, i: usize, j: usize) -> String {
        with_buf(|b, c| unsafe { fm_str(self.0, op, i, j, b, c) })
    }
}

#[cfg(has_flint)]
impl Drop for FlintMat {
    fn drop(&mut self) {
        unsafe { fm_free(self.0) }
    }
}

// ---------------------------------------------------------------------------------------
// PARI
// ---------------------------------------------------------------------------------------

#[cfg(has_pari)]
pub fn pari_init() {
    unsafe { pa_init() }
}

#[cfg(has_pari)]
pub struct Pari(*mut c_void, pub usize);

#[cfg(has_pari)]
impl Pari {
    pub fn new(qs: &[Q]) -> Pari {
        pari_init();
        let p = unsafe { pp_new(qs.len().max(1)) };
        for (i, q) in qs.iter().enumerate() {
            unsafe { pp_set(p, i, q.num.as_ptr(), q.num.len(), q.neg as c_int, q.den.as_ptr(), q.den.len()) };
        }
        Pari(p, qs.len())
    }
    pub fn run(&self, op: i32, b: &Pari, count: usize) -> i64 {
        unsafe { pp_run(op, self.0, b.0, count, ONE_NONE) }
    }
    pub fn run_one(&self, op: i32, b: &Pari, i: usize) -> i64 {
        unsafe { pp_run(op, self.0, b.0, 1, i) }
    }
    pub fn seq(&self, kind: i32, n0: u64, step: u64, count: usize) {
        unsafe { pp_seq(self.0, kind, n0, step, count) }
    }
    pub fn bvec(&self, m: i64, count: usize) {
        unsafe { pp_bvec(self.0, m as c_long, count) }
    }
    /// The last result, or its entry idx (1-based) of a vector.
    pub fn last(&self, idx: i64) -> String {
        with_buf(|b, c| unsafe { pp_str(self.0, idx as c_long, b, c) })
    }
}

#[cfg(has_pari)]
impl Drop for Pari {
    fn drop(&mut self) {
        unsafe { pp_free(self.0) }
    }
}

#[cfg(has_pari)]
pub struct PariMat(*mut c_void);

#[cfg(has_pari)]
impl PariMat {
    pub fn new(k: usize, n: usize, m: usize) -> PariMat {
        pari_init();
        PariMat(unsafe { pm_new(k, n, m) })
    }
    /// Sets pair t from entry functions of A (n x n) and B (n x m).
    pub fn set(&self, t: usize, n: usize, m: usize, a: impl Fn(usize, usize) -> Q, b: impl Fn(usize, usize) -> Q) {
        unsafe {
            pm_begin(self.0);
            for i in 0..n {
                for j in 0..n {
                    let q = a(i, j);
                    pm_entry(0, i, j, q.num.as_ptr(), q.num.len(), q.neg as c_int, q.den.as_ptr(), q.den.len());
                }
                for j in 0..m {
                    let q = b(i, j);
                    pm_entry(1, i, j, q.num.as_ptr(), q.num.len(), q.neg as c_int, q.den.as_ptr(), q.den.len());
                }
            }
            pm_end(self.0, t);
        }
    }
    pub fn run(&self, op: i32, count: usize) -> i64 {
        unsafe { pm_run(self.0, op, count, ONE_NONE) }
    }
    pub fn run_one(&self, op: i32, t: usize) -> i64 {
        unsafe { pm_run(self.0, op, 1, t) }
    }
    pub fn str(&self, op: i32, i: usize, j: usize) -> String {
        with_buf(|b, c| unsafe { pm_str(self.0, op, i, j, b, c) })
    }
}

#[cfg(has_pari)]
impl Drop for PariMat {
    fn drop(&mut self) {
        unsafe { pm_free(self.0) }
    }
}

#[allow(unused)]
fn _unused(_: &CStr) {}
