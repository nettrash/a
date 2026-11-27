//! The references through their C shims (`csrc/`), each behind `has_<lib>`.
#![allow(dead_code)]
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};

/// Takes a string the shims returned (malloc'd) and frees it.
fn take(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the shims return NUL-terminated strings from malloc, freed once here.
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { libc_free(p as *mut c_void) };
    if s.starts_with("ERROR") {
        crate::timing::ref_failed(&s);
    }
    Some(s)
}

unsafe extern "C" {
    #[link_name = "free"]
    fn libc_free(p: *mut c_void);
}

fn cs(s: &str) -> CString {
    CString::new(s).unwrap()
}

macro_rules! handle {
    ($name:ident, $free:ident) => {
        pub struct $name(pub *mut c_void);
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: the handle came from the shim's constructor and is freed once.
                unsafe { $free(self.0) }
            }
        }
    };
}

// ---------------- FLINT ----------------
#[cfg(has_flint)]
pub mod flint {
    use super::*;
    unsafe extern "C" {
        fn fl_zmat_new(t: *const c_char, r: c_long, c: c_long) -> *mut c_void;
        fn fl_zmat_free(a: *mut c_void);
        fn fl_qmat_new(t: *const c_char, r: c_long, c: c_long) -> *mut c_void;
        fn fl_qmat_free(a: *mut c_void);
        fn fl_lll(a: *const c_void, d: f64, e: f64, u: c_int, w: c_int) -> *mut c_char;
        fn fl_lll_is_reduced(a: *const c_void, d: f64, e: f64, prec: c_long) -> c_int;
        fn fl_hnf(a: *const c_void, u: c_int, w: c_int) -> *mut c_char;
        fn fl_snf(a: *const c_void, uv: c_int, w: c_int) -> *mut c_char;
        fn fl_det(a: *const c_void, w: c_int) -> *mut c_char;
        fn fl_det_bareiss(a: *const c_void, w: c_int) -> *mut c_char;
        fn fl_solve(a: *const c_void, b: *const c_void, w: c_int) -> *mut c_char;
        fn fl_solve_fflu(a: *const c_void, b: *const c_void, w: c_int) -> *mut c_char;
        fn fl_inv(a: *const c_void, rational: c_int, w: c_int) -> *mut c_char;
        fn fl_mul(a: *const c_void, b: *const c_void, w: c_int) -> *mut c_char;
        fn fl_qsolve(a: *const c_void, b: *const c_void, w: c_int) -> *mut c_char;
        fn fl_qdet(a: *const c_void, w: c_int) -> *mut c_char;
        fn fl_qinv(a: *const c_void, w: c_int) -> *mut c_char;
        fn fl_nmat_new(w: *const u64, r: c_long, c: c_long, p: u64) -> *mut c_void;
        fn fl_nmat_free(a: *mut c_void);
        fn fl_nmat_op(op: c_int, a: *const c_void, b: *const c_void, arg: u64, w: c_int) -> u64;
        fn fl_version() -> *const c_char;
        fn fl_has_snf_transform() -> c_int;
    }
    handle!(Z, fl_zmat_free);
    handle!(Q, fl_qmat_free);
    handle!(N, fl_nmat_free);

    pub fn version() -> String {
        // SAFETY: a static string.
        unsafe { CStr::from_ptr(fl_version()) }.to_string_lossy().into_owned()
    }
    pub fn z(text: &str, r: usize, c: usize) -> Z {
        let t = cs(text);
        // SAFETY: a valid string of r·c integers.
        Z(unsafe { fl_zmat_new(t.as_ptr(), r as c_long, c as c_long) })
    }
    pub fn q(text: &str, r: usize, c: usize) -> Q {
        let t = cs(text);
        // SAFETY: a valid string of r·c rationals.
        Q(unsafe { fl_qmat_new(t.as_ptr(), r as c_long, c as c_long) })
    }
    pub fn n(words: &[u64], r: usize, c: usize, p: u64) -> N {
        assert_eq!(words.len(), r * c);
        // SAFETY: r·c words.
        N(unsafe { fl_nmat_new(words.as_ptr(), r as c_long, c as c_long, p) })
    }
    // SAFETY (all below): the handles are live for the call.
    pub fn lll(a: &Z, delta: f64, eta: f64, with_u: bool, want: bool) -> Option<String> {
        take(unsafe { fl_lll(a.0, delta, eta, with_u as c_int, want as c_int) })
    }
    pub fn lll_is_reduced(a: &Z, delta: f64, eta: f64, prec: usize) -> bool {
        unsafe { fl_lll_is_reduced(a.0, delta, eta, prec as c_long) != 0 }
    }
    pub fn hnf(a: &Z, with_u: bool, want: bool) -> Option<String> {
        take(unsafe { fl_hnf(a.0, with_u as c_int, want as c_int) })
    }
    /// Whether this FLINT has fmpz_mat_snf_transform (3.5 and later).
    pub fn has_snf_transform() -> bool {
        // SAFETY: no arguments.
        unsafe { fl_has_snf_transform() != 0 }
    }
    pub fn snf(a: &Z, with_uv: bool, want: bool) -> Option<String> {
        take(unsafe { fl_snf(a.0, with_uv as c_int, want as c_int) })
    }
    pub fn det(a: &Z, want: bool) -> Option<String> {
        take(unsafe { fl_det(a.0, want as c_int) })
    }
    pub fn det_bareiss(a: &Z, want: bool) -> Option<String> {
        take(unsafe { fl_det_bareiss(a.0, want as c_int) })
    }
    pub fn solve(a: &Z, b: &Z, want: bool) -> Option<String> {
        take(unsafe { fl_solve(a.0, b.0, want as c_int) })
    }
    pub fn solve_fflu(a: &Z, b: &Z, want: bool) -> Option<String> {
        take(unsafe { fl_solve_fflu(a.0, b.0, want as c_int) })
    }
    pub fn inv(a: &Z, rational: bool, want: bool) -> Option<String> {
        take(unsafe { fl_inv(a.0, rational as c_int, want as c_int) })
    }
    pub fn mul(a: &Z, b: &Z, want: bool) -> Option<String> {
        take(unsafe { fl_mul(a.0, b.0, want as c_int) })
    }
    pub fn qsolve(a: &Q, b: &Q, want: bool) -> Option<String> {
        take(unsafe { fl_qsolve(a.0, b.0, want as c_int) })
    }
    pub fn qdet(a: &Q, want: bool) -> Option<String> {
        take(unsafe { fl_qdet(a.0, want as c_int) })
    }
    pub fn qinv(a: &Q, want: bool) -> Option<String> {
        take(unsafe { fl_qinv(a.0, want as c_int) })
    }
    pub fn nop(op: i32, a: &N, b: Option<&N>, want: bool) -> u64 {
        let bp = b.map_or(std::ptr::null(), |b| b.0 as *const c_void);
        unsafe { fl_nmat_op(op, a.0, bp, 0, want as c_int) }
    }
    /// `fmpz_mat_get_nmod_mat` of a FLINT integer matrix.
    pub fn from_z(a: &Z, p: u64, want: bool) -> u64 {
        unsafe { fl_nmat_op(12, a.0, std::ptr::null(), p, want as c_int) }
    }
}

// ---------------- fplll ----------------
#[cfg(has_fplll)]
pub mod fplll {
    use super::*;
    unsafe extern "C" {
        fn fp_new(t: *const c_char, r: c_long, c: c_long) -> *mut c_void;
        fn fp_free(a: *mut c_void);
        fn fp_lll(a: *const c_void, d: f64, e: f64, m: c_int, u: c_int, w: c_int) -> *mut c_char;
        fn fp_bkz(a: *const c_void, b: c_int, d: f64, v: c_int, u: c_int, w: c_int) -> *mut c_char;
        fn fp_svp(a: *const c_void, m: c_int, w: c_int) -> *mut c_char;
        fn fp_cvp(a: *const c_void, t: *const c_void, m: c_int, w: c_int) -> *mut c_char;
        fn fp_version() -> *const c_char;
    }
    handle!(M, fp_free);
    pub fn version() -> String {
        // SAFETY: a static string.
        unsafe { CStr::from_ptr(fp_version()) }.to_string_lossy().into_owned()
    }
    pub fn m(text: &str, r: usize, c: usize) -> M {
        let t = cs(text);
        // SAFETY: r·c integers.
        M(unsafe { fp_new(t.as_ptr(), r as c_long, c as c_long) })
    }
    pub const WRAPPER: i32 = 0;
    pub const PROVED: i32 = 1;
    pub const HEURISTIC: i32 = 2;
    pub const FAST: i32 = 3;
    // SAFETY (below): live handles.
    pub fn lll(a: &M, delta: f64, eta: f64, method: i32, with_u: bool, want: bool) -> Option<String> {
        take(unsafe { fp_lll(a.0, delta, eta, method, with_u as c_int, want as c_int) })
    }
    pub fn bkz(a: &M, beta: usize, delta: f64, variant: i32, with_u: bool, want: bool) -> Option<String> {
        take(unsafe { fp_bkz(a.0, beta as c_int, delta, variant, with_u as c_int, want as c_int) })
    }
    pub fn svp(a: &M, method: i32, want: bool) -> Option<String> {
        take(unsafe { fp_svp(a.0, method, want as c_int) })
    }
    pub fn cvp(a: &M, t: &M, method: i32, want: bool) -> Option<String> {
        take(unsafe { fp_cvp(a.0, t.0, method, want as c_int) })
    }
}

// ---------------- NTL ----------------
#[cfg(has_ntl)]
pub mod ntl {
    use super::*;
    unsafe extern "C" {
        fn nt_new(t: *const c_char, r: c_long, c: c_long) -> *mut c_void;
        fn nt_free(a: *mut c_void);
        fn nt_lll(a: *const c_void, k: c_int, d: f64, deep: c_long, u: c_int, w: c_int) -> *mut c_char;
        fn nt_bkz(a: *const c_void, k: c_int, b: c_long, d: f64, u: c_int, w: c_int) -> *mut c_char;
        fn nt_hnf(a: *const c_void, w: c_int) -> *mut c_char;
        fn nt_det(a: *const c_void, w: c_int) -> *mut c_char;
        fn nt_sp_nbits() -> c_long;
        fn nt_version() -> *const c_char;
        fn nt_zp_new(w: *const u64, r: c_long, c: c_long, p: u64) -> *mut c_void;
        fn nt_zp_free(a: *mut c_void);
        fn nt_zp_op(op: c_int, a: *mut c_void, b: *mut c_void, w: c_int) -> u64;
        fn nt_gf2_new(w: *const u64, r: c_long, c: c_long) -> *mut c_void;
        fn nt_gf2_free(a: *mut c_void);
        fn nt_gf2_op(op: c_int, a: *const c_void, b: *const c_void, w: c_int) -> u64;
        fn nt_gf2_transpose(a: *const c_void) -> *mut c_void;
    }
    handle!(M, nt_free);
    handle!(Zp, nt_zp_free);
    handle!(G, nt_gf2_free);
    pub fn m(text: &str, r: usize, c: usize) -> M {
        let t = cs(text);
        // SAFETY: r·c integers.
        M(unsafe { nt_new(t.as_ptr(), r as c_long, c as c_long) })
    }
    pub fn version() -> String {
        // SAFETY: a static string.
        unsafe { CStr::from_ptr(nt_version()) }.to_string_lossy().into_owned()
    }
    pub fn sp_nbits() -> u32 {
        // SAFETY: a constant.
        unsafe { nt_sp_nbits() as u32 }
    }
    pub const LLL_FP: i32 = 0;
    pub const G_LLL_FP: i32 = 1;
    pub const LLL_EXACT: i32 = 2;
    pub const LLL_XD: i32 = 3;
    pub const G_LLL_XD: i32 = 4;
    pub const BKZ_FP: i32 = 0;
    pub const BKZ_XD: i32 = 2;
    /// LLL_FP and BKZ_FP stop the process ("numbers too big") for entries of about 500 bits and
    /// more: the XD variants (an extended exponent) are NTL's own advice there.
    pub fn needs_xd(max_bits: u64) -> bool {
        max_bits > 400
    }
    // SAFETY (below): live handles.
    pub fn lll(a: &M, kind: i32, delta: f64, deep: usize, with_u: bool, want: bool) -> Option<String> {
        take(unsafe { nt_lll(a.0, kind, delta, deep as c_long, with_u as c_int, want as c_int) })
    }
    pub fn bkz(a: &M, kind: i32, beta: usize, delta: f64, with_u: bool, want: bool) -> Option<String> {
        take(unsafe { nt_bkz(a.0, kind, beta as c_long, delta, with_u as c_int, want as c_int) })
    }
    pub fn hnf(a: &M, want: bool) -> Option<String> {
        take(unsafe { nt_hnf(a.0, want as c_int) })
    }
    pub fn det(a: &M, want: bool) -> Option<String> {
        take(unsafe { nt_det(a.0, want as c_int) })
    }
    pub fn zp(words: &[u64], r: usize, c: usize, p: u64) -> Zp {
        // SAFETY: r·c words.
        Zp(unsafe { nt_zp_new(words.as_ptr(), r as c_long, c as c_long, p) })
    }
    pub fn zp_op(op: i32, a: &Zp, b: Option<&Zp>, want: bool) -> u64 {
        let bp = b.map_or(std::ptr::null_mut(), |b| b.0);
        unsafe { nt_zp_op(op, a.0, bp, want as c_int) }
    }
    pub fn gf2(words: &[u64], r: usize, c: usize) -> G {
        // SAFETY: r·⌈c/64⌉ words.
        G(unsafe { nt_gf2_new(words.as_ptr(), r as c_long, c as c_long) })
    }
    pub fn gf2_transpose(a: &G) -> G {
        G(unsafe { nt_gf2_transpose(a.0) })
    }
    pub fn gf2_op(op: i32, a: &G, b: Option<&G>, want: bool) -> u64 {
        let bp = b.map_or(std::ptr::null(), |b| b.0 as *const c_void);
        unsafe { nt_gf2_op(op, a.0, bp, want as c_int) }
    }
}

// ---------------- M4RI ----------------
#[cfg(has_m4ri)]
pub mod m4ri {
    use super::*;
    unsafe extern "C" {
        fn m4_new(w: *const u64, r: c_long, c: c_long) -> *mut c_void;
        fn m4_free(a: *mut c_void);
        fn m4_op(op: c_int, a: *const c_void, b: *const c_void, arg: c_int, w: c_int) -> u64;
        fn m4_rowop(op: c_int, a: *mut c_void, x: c_long, y: c_long);
    }
    handle!(M, m4_free);
    pub fn m(words: &[u64], r: usize, c: usize) -> M {
        // SAFETY: r·⌈c/64⌉ words.
        M(unsafe { m4_new(words.as_ptr(), r as c_long, c as c_long) })
    }
    pub fn rowop(op: i32, a: &mut M, x: usize, y: usize) {
        // SAFETY: a live handle, rows inside the matrix.
        unsafe { m4_rowop(op, a.0, x as c_long, y as c_long) }
    }
    pub fn op(op: i32, a: &M, b: Option<&M>, arg: i32, want: bool) -> u64 {
        let bp = b.map_or(std::ptr::null(), |b| b.0 as *const c_void);
        // SAFETY: live handles.
        unsafe { m4_op(op, a.0, bp, arg, want as c_int) }
    }
}

// ---------------- PARI ----------------
#[cfg(has_pari)]
pub mod pari {
    use super::*;
    unsafe extern "C" {
        fn pa_init();
        fn pa_zmat_new(t: *const c_char, r: c_long, c: c_long, tr: c_int) -> *mut c_void;
        fn pa_qmat_new(t: *const c_char, r: c_long, c: c_long) -> *mut c_void;
        fn pa_reals_new(m: *const c_char, e: *const c_long, p: *const c_long, n: c_long) -> *mut c_void;
        fn pa_free(a: *mut c_void);
        fn pa_qflll(a: *const c_void, f: c_long, w: c_int) -> *mut c_char;
        fn pa_mathnf(a: *const c_void, f: c_long, w: c_int) -> *mut c_char;
        fn pa_matsnf(a: *const c_void, f: c_long, w: c_int) -> *mut c_char;
        fn pa_det(a: *const c_void, w: c_int) -> *mut c_char;
        fn pa_solve(a: *const c_void, b: *const c_void, w: c_int) -> *mut c_char;
        fn pa_inv(a: *const c_void, w: c_int) -> *mut c_char;
        fn pa_mul(a: *const c_void, b: *const c_void, w: c_int) -> *mut c_char;
        fn pa_lindep(a: *const c_void, w: c_int) -> *mut c_char;
        fn pa_svp(a: *const c_void, f: c_long, w: c_int) -> *mut c_char;
        fn pa_short(a: *const c_void, b: *const c_char, w: c_int) -> *mut c_char;
        fn pa_cvp(a: *const c_void, y: *const c_void, w: c_int) -> *mut c_char;
        fn pa_version() -> *const c_char;
    }
    handle!(G, pa_free);
    pub fn init() {
        // SAFETY: idempotent initialisation of libpari, single-threaded harness.
        unsafe { pa_init() }
    }
    pub fn version() -> String {
        // SAFETY: a static string.
        unsafe { CStr::from_ptr(pa_version()) }.to_string_lossy().into_owned()
    }
    /// A matrix from the crate's rows; `transpose` makes the rows PARI's columns (lattices).
    pub fn z(text: &str, r: usize, c: usize, transpose: bool) -> G {
        init();
        let t = cs(text);
        // SAFETY: r·c integers.
        G(unsafe { pa_zmat_new(t.as_ptr(), r as c_long, c as c_long, transpose as c_int) })
    }
    pub fn q(text: &str, r: usize, c: usize) -> G {
        init();
        let t = cs(text);
        // SAFETY: r·c rationals.
        G(unsafe { pa_qmat_new(t.as_ptr(), r as c_long, c as c_long) })
    }
    pub fn reals(mants: &[String], exps: &[i64], precs: &[u64]) -> G {
        init();
        let t = cs(&mants.join(" "));
        let e: Vec<c_long> = exps.iter().map(|&x| x as c_long).collect();
        let p: Vec<c_long> = precs.iter().map(|&x| x as c_long).collect();
        // SAFETY: n mantissas, exponents and precisions.
        G(unsafe { pa_reals_new(t.as_ptr(), e.as_ptr(), p.as_ptr(), mants.len() as c_long) })
    }
    // SAFETY (below): live handles.
    pub fn qflll(a: &G, flag: i64, want: bool) -> Option<String> {
        take(unsafe { pa_qflll(a.0, flag as c_long, want as c_int) })
    }
    pub fn mathnf(a: &G, flag: i64, want: bool) -> Option<String> {
        take(unsafe { pa_mathnf(a.0, flag as c_long, want as c_int) })
    }
    pub fn matsnf(a: &G, flag: i64, want: bool) -> Option<String> {
        take(unsafe { pa_matsnf(a.0, flag as c_long, want as c_int) })
    }
    pub fn det(a: &G, want: bool) -> Option<String> {
        take(unsafe { pa_det(a.0, want as c_int) })
    }
    pub fn solve(a: &G, b: &G, want: bool) -> Option<String> {
        take(unsafe { pa_solve(a.0, b.0, want as c_int) })
    }
    pub fn inv(a: &G, want: bool) -> Option<String> {
        take(unsafe { pa_inv(a.0, want as c_int) })
    }
    pub fn mul(a: &G, b: &G, want: bool) -> Option<String> {
        take(unsafe { pa_mul(a.0, b.0, want as c_int) })
    }
    pub fn lindep(a: &G, want: bool) -> Option<String> {
        take(unsafe { pa_lindep(a.0, want as c_int) })
    }
    pub fn svp(a: &G, flag: i64, want: bool) -> Option<String> {
        take(unsafe { pa_svp(a.0, flag as c_long, want as c_int) })
    }
    pub fn short(a: &G, bound: &str, want: bool) -> Option<String> {
        let b = cs(bound);
        take(unsafe { pa_short(a.0, b.as_ptr(), want as c_int) })
    }
    pub fn cvp(a: &G, y: &G, want: bool) -> Option<String> {
        take(unsafe { pa_cvp(a.0, y.0, want as c_int) })
    }
}
