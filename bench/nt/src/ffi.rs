//! The C shims (csrc/), each behind the cfg flag build.rs sets when its library is found.
#![allow(dead_code)]
use std::ffi::{CStr, c_char, c_void};

pub type H = *mut c_void;

#[cfg(has_gmp)]
unsafe extern "C" {
    pub fn gmp_pool(limbs: *const u64, off: *const u64, n: usize) -> H;
    pub fn gmp_op(op: i32, a: H, b: H, i0: usize, k: usize) -> u64;
    pub fn gmp_seq(op: i32, ns: *const u64, n: usize, i0: usize, k: usize) -> u64;
    pub fn gmp_version_str() -> *const c_char;
}
pub mod g {
    pub const PRIME: i32 = 1;
    pub const NEXT: i32 = 2;
    pub const PREV: i32 = 3;
    pub const KRON: i32 = 4;
    pub const JACOBI: i32 = 5;
    pub const PPOW: i32 = 6;
    pub const FIB: i32 = 1;
    pub const LUC: i32 = 2;
    pub const FIB2L: i32 = 3;
    pub const PRIMORIAL: i32 = 4;
}

#[cfg(has_flint)]
unsafe extern "C" {
    pub fn flint_setup();
    pub fn flint_pool(limbs: *const u64, off: *const u64, n: usize) -> H;
    pub fn flint_op(op: i32, a: H, b: H, i0: usize, k: usize, arg: u64, b1: u64, b2: u64) -> u64;
    pub fn flint_facpool(p: H, exps: *const u32, goff: *const u64, ng: usize) -> H;
    pub fn flint_facop(op: i32, q: H, i0: usize, k: usize, arg: u64) -> u64;
    pub fn flint_nop(op: i32, xs: *const u64, n: usize, i0: usize, k: usize) -> u64;
    pub fn flint_version_str() -> *const c_char;
}
pub mod f {
    pub const BPSW: i32 = 1;
    pub const NEXT: i32 = 2;
    pub const FACTOR: i32 = 3;
    pub const QSIEVE: i32 = 4;
    pub const TRIAL: i32 = 5;
    pub const RHO: i32 = 6;
    pub const ECM: i32 = 7;
    pub const PP1: i32 = 8;
    pub const PPOW: i32 = 9;
    pub const CRT2: i32 = 10;
    pub const MCRT: i32 = 11;
    pub const KRON: i32 = 12;
    pub const SQRTMOD: i32 = 13;
    pub const LUCCHAIN: i32 = 14;
    pub const LUCCHAIN1: i32 = 15;
    pub const DIVISORS: i32 = 16;
    pub const FACPHI: i32 = 17;
    pub const FACMU: i32 = 18;
    pub const JACOBI: i32 = 19;
    pub const ISPRIME: i32 = 20;
    pub const FF_PHI: i32 = 1;
    pub const FF_MU: i32 = 2;
    pub const FF_SIGMA: i32 = 3;
    pub const N_ISPRIME: i32 = 1;
    pub const N_NEXT: i32 = 2;
    pub const N_FACTOR: i32 = 3;
    pub const N_PRIMROOT: i32 = 4;
    pub const N_PHI: i32 = 5;
    pub const N_MU: i32 = 6;
    pub const N_SQFREE: i32 = 7;
    pub const N_SQRTMOD: i32 = 8;
    pub const N_DLOG: i32 = 9;
    pub const N_PRIMEPI: i32 = 10;
    pub const N_NTH: i32 = 11;
    pub const N_FIB: i32 = 12;
    pub const N_PRIMORIAL: i32 = 13;
    pub const N_PART: i32 = 14;
    pub const N_PART_CLEAN: i32 = 15;
    pub const N_SQRTMODN: i32 = 16;
}

#[cfg(has_pari)]
unsafe extern "C" {
    pub fn pari_setup();
    pub fn pari_pool(limbs: *const u64, off: *const u64, n: usize) -> H;
    pub fn pari_op(op: i32, a: H, b: H, i0: usize, k: usize, arg: u64) -> u64;
    pub fn pari_facpool(p: H, exps: *const u32, goff: *const u64, ng: usize) -> H;
    pub fn pari_facop(op: i32, q: H, b: H, i0: usize, k: usize, arg: u64) -> u64;
    pub fn pari_version_str() -> *const c_char;
}
pub mod p {
    pub const ISPSEUDO: i32 = 1;
    pub const NEXT: i32 = 2;
    pub const PREC: i32 = 3;
    pub const FACTOR: i32 = 4;
    pub const MPQS: i32 = 5;
    pub const RHO: i32 = 6;
    pub const ISPOWER: i32 = 7;
    pub const KRON: i32 = 8;
    pub const FPSQRT: i32 = 9;
    pub const CORN: i32 = 10;
    pub const CHINESE: i32 = 11;
    pub const ZVCHINESE: i32 = 12;
    pub const ZNORDER: i32 = 13;
    pub const PRIMROOT: i32 = 14;
    pub const ZNLOG: i32 = 15;
    pub const FPLOG: i32 = 16;
    pub const PHI: i32 = 17;
    pub const MU: i32 = 18;
    pub const SQFREE: i32 = 19;
    pub const NUMBPART: i32 = 20;
    pub const PRIMEPI: i32 = 21;
    pub const PRIME: i32 = 22;
    pub const ZNSQRT: i32 = 23;
    pub const SIGMA1: i32 = 24;
    pub const NUMDIV: i32 = 25;
    pub const F_PHI: i32 = 1;
    pub const F_NUMDIV: i32 = 2;
    pub const F_SIGMA: i32 = 3;
    pub const F_MU: i32 = 4;
    pub const F_DIVISORS: i32 = 5;
    pub const F_RADICAL: i32 = 6;
    pub const F_ZNSQRT: i32 = 7;
    pub const F_VALUE: i32 = 8;
}

#[cfg(has_ecm)]
unsafe extern "C" {
    pub fn ecm_op(op: i32, a: H, i0: usize, k: usize, arg: u64, b1: f64, b2: f64) -> u64;
    pub fn ecm_version_str() -> *const c_char;
}
pub mod e {
    pub const ECM: i32 = 1;
    pub const PM1: i32 = 2;
    pub const PP1: i32 = 3;
}

#[cfg(has_primesieve)]
unsafe extern "C" {
    pub fn ps_setup();
    pub fn ps_count(a: u64, b: u64) -> u64;
    pub fn ps_iter_sum(a: u64, b: u64) -> u64;
    pub fn ps_first(a: u64, k: u64) -> u64;
    pub fn pc_pi(x: u64) -> u64;
    pub fn pc_nth(n: u64) -> u64;
    pub fn ps_version_str() -> *const c_char;
    pub fn pc_version_str() -> *const c_char;
}

pub fn cstr(p: *const c_char) -> String {
    // SAFETY: the shims return static NUL-terminated strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// The references found and their versions, for the run's header.
pub fn versions() -> Vec<String> {
    #[allow(unused_mut)]
    let mut v = vec![];
    #[cfg(has_gmp)]
    v.push(format!("GMP {}", cstr(unsafe { gmp_version_str() })));
    #[cfg(has_flint)]
    v.push(format!("FLINT {}", cstr(unsafe { flint_version_str() })));
    #[cfg(has_pari)]
    v.push(format!("PARI {}", cstr(unsafe { pari_version_str() })));
    #[cfg(has_ecm)]
    v.push(format!("GMP-ECM {}", cstr(unsafe { ecm_version_str() })));
    #[cfg(has_primesieve)]
    {
        v.push(format!("primesieve {}", cstr(unsafe { ps_version_str() })));
        v.push(format!("primecount {}", cstr(unsafe { pc_version_str() })));
    }
    v
}

/// One-time set-up of the libraries: one thread each.
pub fn setup() {
    #[cfg(has_flint)]
    unsafe {
        flint_setup()
    };
    #[cfg(has_pari)]
    unsafe {
        pari_setup()
    };
    #[cfg(has_primesieve)]
    unsafe {
        ps_setup()
    };
}
