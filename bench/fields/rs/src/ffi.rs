//! The C shims (csrc/), each family behind one small wrapper type.

#![allow(dead_code)]

use crate::harness::{Op, UNSUPPORTED};
use std::ffi::c_void;

type PfNew = unsafe extern "C" fn(*const u64, usize, *const u64, usize, *const u64, usize) -> *mut c_void;
type ExtNew =
    unsafe extern "C" fn(*const u64, usize, *const u64, usize, *const u64, usize, *const u64, usize) -> *mut c_void;
type BinNew = unsafe extern "C" fn(*const u32, usize, *const u64, usize, u64) -> *mut c_void;
type Out = unsafe extern "C" fn(*mut c_void, i32, usize, *mut u64) -> i32;
type Run = unsafe extern "C" fn(*mut c_void, i32, u64);
type Free = unsafe extern "C" fn(*mut c_void);

/// A handle of a C reference, with its entry points.
pub struct CRef {
    pub name: &'static str,
    h: *mut c_void,
    out: Out,
    run: Run,
    free: Free,
    /// Limbs of one output element.
    width: usize,
}

impl CRef {
    /// The output of `op` on input `i`: the limbs and the returned flag or symbol, `None` if the
    /// reference does not have the operation.
    pub fn out(&self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let mut v = vec![0u64; self.width];
        // SAFETY: the handle is live and `v` has the width the shim writes.
        let r = unsafe { (self.out)(self.h, op as i32, i, v.as_mut_ptr()) };
        if r == UNSUPPORTED { None } else { Some((v, r)) }
    }

    /// `iters` operations of the timed loop.
    pub fn run(&mut self, op: Op, iters: u64) {
        // SAFETY: the handle is live.
        unsafe { (self.run)(self.h, op as i32, iters) }
    }
}

impl Drop for CRef {
    fn drop(&mut self) {
        // SAFETY: made by the matching `new`, freed once.
        unsafe { (self.free)(self.h) }
    }
}

#[cfg(has_flint)]
unsafe extern "C" {
    fn flint_fmpz_mod_pf_new(p: *const u64, n: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn flint_fmpz_mod_pf_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn flint_fmpz_mod_pf_run(h: *mut c_void, op: i32, iters: u64);
    fn flint_fmpz_mod_pf_free(h: *mut c_void);
    fn flint_nmod_pf_new(p: *const u64, n: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn flint_nmod_pf_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn flint_nmod_pf_run(h: *mut c_void, op: i32, iters: u64);
    fn flint_nmod_pf_free(h: *mut c_void);
    fn flint_mpn_mod_pf_new(p: *const u64, n: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn flint_mpn_mod_pf_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn flint_mpn_mod_pf_run(h: *mut c_void, op: i32, iters: u64);
    fn flint_mpn_mod_pf_free(h: *mut c_void);
    fn flint_fq_ext_new(p: *const u64, n: usize, g: *const u64, k: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn flint_fq_ext_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn flint_fq_ext_run(h: *mut c_void, op: i32, iters: u64);
    fn flint_fq_ext_free(h: *mut c_void);
    fn flint_fq_nmod_ext_new(p: *const u64, n: usize, g: *const u64, k: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn flint_fq_nmod_ext_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn flint_fq_nmod_ext_run(h: *mut c_void, op: i32, iters: u64);
    fn flint_fq_nmod_ext_free(h: *mut c_void);
    fn flint_fq_zech_ext_new(p: *const u64, n: usize, g: *const u64, k: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn flint_fq_zech_ext_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn flint_fq_zech_ext_run(h: *mut c_void, op: i32, iters: u64);
    fn flint_fq_zech_ext_free(h: *mut c_void);
}

#[cfg(has_gmp)]
unsafe extern "C" {
    fn gmp_pf_new(p: *const u64, n: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn gmp_pf_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn gmp_pf_run(h: *mut c_void, op: i32, iters: u64);
    fn gmp_pf_free(h: *mut c_void);
}

#[cfg(has_ntl)]
unsafe extern "C" {
    fn ntl_pf_new(p: *const u64, n: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn ntl_pf_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn ntl_pf_run(h: *mut c_void, op: i32, iters: u64);
    fn ntl_pf_free(h: *mut c_void);
    fn ntl_ext_new(p: *const u64, n: usize, g: *const u64, k: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn ntl_ext_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn ntl_ext_run(h: *mut c_void, op: i32, iters: u64);
    fn ntl_ext_free(h: *mut c_void);
    fn ntl_sp_ext_new(p: *const u64, n: usize, g: *const u64, k: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn ntl_sp_ext_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn ntl_sp_ext_run(h: *mut c_void, op: i32, iters: u64);
    fn ntl_sp_ext_free(h: *mut c_void);
    fn ntl_bin_new(exps: *const u32, nexp: usize, xs: *const u64, c: usize, e: u64) -> *mut c_void;
    fn ntl_bin_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn ntl_bin_run(h: *mut c_void, op: i32, iters: u64);
    fn ntl_bin_free(h: *mut c_void);
    pub fn ntl_gf2x_irred(exps: *const u32, nexp: usize, reps: u64) -> i64;
}

#[cfg(has_ossl)]
unsafe extern "C" {
    fn ossl_version() -> *const std::ffi::c_char;
    fn ossl_pf_new(p: *const u64, n: usize, xs: *const u64, c: usize, e: *const u64, en: usize) -> *mut c_void;
    fn ossl_pf_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn ossl_pf_run(h: *mut c_void, op: i32, iters: u64);
    fn ossl_pf_free(h: *mut c_void);
    fn ossl_bin_new(exps: *const u32, nexp: usize, xs: *const u64, c: usize, e: u64) -> *mut c_void;
    fn ossl_bin_out(h: *mut c_void, op: i32, i: usize, out: *mut u64) -> i32;
    fn ossl_bin_run(h: *mut c_void, op: i32, iters: u64);
    fn ossl_bin_free(h: *mut c_void);
    pub fn ossl_gcm_keys(key: *const u8, iv: *const u8, h: *mut u8, ej0: *mut u8) -> i32;
    pub fn ossl_gcm_tag(key: *const u8, iv: *const u8, aad: *const u8, len: usize, tag: *mut u8) -> i32;
    pub fn ossl_gcm_new(key: *const u8, iv: *const u8) -> *mut c_void;
    pub fn ossl_gcm_aad(h: *mut c_void, aad: *const u8, len: usize, reps: u64);
    pub fn ossl_gcm_free(h: *mut c_void);
}

/// The OpenSSL version string, if it is linked.
pub fn openssl_version() -> Option<String> {
    #[cfg(has_ossl)]
    {
        // SAFETY: OpenSSL returns a static NUL-terminated string.
        let s = unsafe { std::ffi::CStr::from_ptr(ossl_version()) };
        return Some(s.to_string_lossy().into_owned());
    }
    #[allow(unreachable_code)]
    None
}

/// The prime-field references that are built: (label, new, out, run, free).
pub fn pf_refs() -> Vec<(&'static str, PfNew, Out, Run, Free)> {
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, PfNew, Out, Run, Free)> = vec![];
    #[cfg(has_flint)]
    {
        v.push(("FLINT fmpz_mod", flint_fmpz_mod_pf_new, flint_fmpz_mod_pf_out, flint_fmpz_mod_pf_run, flint_fmpz_mod_pf_free));
        v.push(("FLINT nmod", flint_nmod_pf_new, flint_nmod_pf_out, flint_nmod_pf_run, flint_nmod_pf_free));
        v.push(("FLINT mpn_mod", flint_mpn_mod_pf_new, flint_mpn_mod_pf_out, flint_mpn_mod_pf_run, flint_mpn_mod_pf_free));
    }
    #[cfg(has_ntl)]
    v.push(("NTL ZZ_p", ntl_pf_new, ntl_pf_out, ntl_pf_run, ntl_pf_free));
    #[cfg(has_gmp)]
    v.push(("GMP mpz", gmp_pf_new, gmp_pf_out, gmp_pf_run, gmp_pf_free));
    #[cfg(has_ossl)]
    v.push(("OpenSSL BN", ossl_pf_new, ossl_pf_out, ossl_pf_run, ossl_pf_free));
    v
}

/// The extension-field references that are built.
pub fn ext_refs() -> Vec<(&'static str, ExtNew, Out, Run, Free)> {
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, ExtNew, Out, Run, Free)> = vec![];
    #[cfg(has_flint)]
    {
        v.push(("FLINT fq_zech", flint_fq_zech_ext_new, flint_fq_zech_ext_out, flint_fq_zech_ext_run, flint_fq_zech_ext_free));
        v.push(("FLINT fq_nmod", flint_fq_nmod_ext_new, flint_fq_nmod_ext_out, flint_fq_nmod_ext_run, flint_fq_nmod_ext_free));
        v.push(("FLINT fq", flint_fq_ext_new, flint_fq_ext_out, flint_fq_ext_run, flint_fq_ext_free));
    }
    #[cfg(has_ntl)]
    {
        v.push(("NTL zz_pE", ntl_sp_ext_new, ntl_sp_ext_out, ntl_sp_ext_run, ntl_sp_ext_free));
        v.push(("NTL ZZ_pE", ntl_ext_new, ntl_ext_out, ntl_ext_run, ntl_ext_free));
    }
    v
}

/// The binary-field references that are built.
pub fn bin_refs() -> Vec<(&'static str, BinNew, Out, Run, Free)> {
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, BinNew, Out, Run, Free)> = vec![];
    #[cfg(has_ntl)]
    v.push(("NTL GF2E", ntl_bin_new, ntl_bin_out, ntl_bin_run, ntl_bin_free));
    #[cfg(has_ossl)]
    v.push(("OpenSSL BN_GF2m", ossl_bin_new, ossl_bin_out, ossl_bin_run, ossl_bin_free));
    v
}

/// A prime-field reference on the prime with the limbs `p`, inputs `xs` (`count` of `p.len()`
/// limbs) and the exponent `e`; `None` if it does not take this prime.
pub fn pf_new(r: &(&'static str, PfNew, Out, Run, Free), p: &[u64], xs: &[u64], count: usize, e: &[u64]) -> Option<CRef> {
    // SAFETY: the slices are as long as the lengths given.
    let h = unsafe { (r.1)(p.as_ptr(), p.len(), xs.as_ptr(), count, e.as_ptr(), e.len()) };
    if h.is_null() {
        return None;
    }
    Some(CRef { name: r.0, h, out: r.2, run: r.3, free: r.4, width: p.len() })
}

/// An extension-field reference: `g` the `k + 1` coefficients of the modulus, `xs` `count`
/// elements of `k` coefficients, each of `n` limbs.
#[allow(clippy::too_many_arguments)]
pub fn ext_new(
    r: &(&'static str, ExtNew, Out, Run, Free),
    p: &[u64],
    g: &[u64],
    k: usize,
    xs: &[u64],
    count: usize,
    e: &[u64],
) -> Option<CRef> {
    // SAFETY: the slices are as long as the lengths given.
    let h = unsafe { (r.1)(p.as_ptr(), p.len(), g.as_ptr(), k, xs.as_ptr(), count, e.as_ptr(), e.len()) };
    if h.is_null() {
        return None;
    }
    Some(CRef { name: r.0, h, out: r.2, run: r.3, free: r.4, width: k * p.len() })
}

/// A binary-field reference: `exps` the exponents of the modulus in decreasing order, `xs`
/// `count` elements of `w` words.
pub fn bin_new(r: &(&'static str, BinNew, Out, Run, Free), exps: &[u32], xs: &[u64], count: usize, w: usize, e: u64) -> Option<CRef> {
    // SAFETY: the slices are as long as the lengths given.
    let h = unsafe { (r.1)(exps.as_ptr(), exps.len(), xs.as_ptr(), count, e) };
    if h.is_null() {
        return None;
    }
    Some(CRef { name: r.0, h, out: r.2, run: r.3, free: r.4, width: w })
}
