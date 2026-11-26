//! Apple Accelerate's Sparse Solvers (csrc/accel.c); macOS only.

#![allow(dead_code)]
use std::ffi::{c_long, c_void};

pub type H = *mut c_void;

unsafe extern "C" {
    pub fn ac_single_threaded() -> i32;
    pub fn ac_mul(m: i32, n: i32, cs: *mut c_long, ri: *mut i32, x: *mut f64, transpose: i32, k: i32, xx: *mut f64, yy: *mut f64);
    pub fn ac_mul_f32(m: i32, n: i32, cs: *mut c_long, ri: *mut i32, x: *mut f32, transpose: i32, xx: *mut f32, yy: *mut f32);
    pub fn ac_from_coo(m: i32, n: i32, nnz: c_long, ri: *const i32, ci: *const i32, x: *const f64) -> c_long;
    pub fn ac_factor(ty: i32, m: i32, n: i32, cs: *mut c_long, ri: *mut i32, x: *mut f64, kind_sym: i32, order: i32, perm: *mut i32) -> H;
    pub fn ac_factor_default(ty: i32, m: i32, n: i32, cs: *mut c_long, ri: *mut i32, x: *mut f64, kind_sym: i32) -> H;
    pub fn ac_factor_size(h: H) -> f64;
    pub fn ac_solve(h: H, k: i32, b: *mut f64, x: *mut f64);
    pub fn ac_free(h: H);
    pub fn ac_iterative(method: i32, variant: i32, nvec: i32, n: i32, cs: *mut c_long, ri: *mut i32, x_: *mut f64, sym: i32, b: *mut f64, x: *mut f64, rtol: f64, maxit: i32, precond: i32) -> i32;
    pub fn ac_iterative_count(method: i32, variant: i32, nvec: i32, n: i32, cs: *mut c_long, ri: *mut i32, x_: *mut f64, sym: i32, b: *mut f64, x: *mut f64, rtol: f64, maxit: i32, precond: i32, products: *mut c_long) -> i32;
}

// SparseFactorization_t values (Solve.h).
pub const CHOLESKY: i32 = 0;
pub const LDLT: i32 = 1;
pub const LDLT_UNPIVOTED: i32 = 2;
pub const QR: i32 = 40;
pub const CHOLESKY_ATA: i32 = 41;

/// The value of SparseFactorizationLU and its variants, read from the header at build time would
/// be better; these are the values of macOS 26's Solve.h.
pub const LU_TPP: i32 = lu_tpp();
const fn lu_tpp() -> i32 {
    // SparseFactorizationLU = 80, Unpivoted 81, SPP 82, TPP 83 (Solve.h, macOS 26).
    83
}
