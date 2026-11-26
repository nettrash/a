//! The SuiteSparse shim (csrc/ss.c) and the int32 compressed-column form it reads.

#![allow(dead_code)]
use crate::mats::Csr;
use std::ffi::c_void;

/// Compressed sparse columns with int32 indices (int64 copies for SPQR).
pub struct Csc32 {
    pub m: i32,
    pub n: i32,
    pub p: Vec<i32>,
    pub i: Vec<i32>,
    pub x: Vec<f64>,
}

impl Csc32 {
    /// The compressed columns of `a` (its transpose's rows).
    pub fn of(a: &Csr) -> Csc32 {
        let t = a.transpose();
        Csc32::of_rows_as_cols(&t, a.nrows, a.ncols)
    }
    /// The arrays of `t` (rows) read as the columns of its transpose: no copy of the layout.
    pub fn of_rows_as_cols(t: &Csr, m: usize, n: usize) -> Csc32 {
        Csc32 {
            m: m as i32,
            n: n as i32,
            p: t.ptr.iter().map(|&v| v as i32).collect(),
            i: t.idx.iter().map(|&v| v as i32).collect(),
            x: t.val.clone(),
        }
    }
    /// The CSR arrays of `a` read as the CSC arrays of Aᵀ.
    pub fn of_transpose(a: &Csr) -> Csc32 {
        Csc32::of_rows_as_cols(a, a.ncols, a.nrows)
    }
    pub fn nnz(&self) -> usize {
        self.i.len()
    }
    pub fn p64(&self) -> Vec<i64> {
        self.p.iter().map(|&v| v as i64).collect()
    }
    pub fn i64(&self) -> Vec<i64> {
        self.i.iter().map(|&v| v as i64).collect()
    }
    pub fn plong(&self) -> Vec<std::ffi::c_long> {
        self.p.iter().map(|&v| v as std::ffi::c_long).collect()
    }
}

pub type H = *mut c_void;

unsafe extern "C" {
    pub fn ss_sp_nrow(h: H) -> i32;
    pub fn ss_sp_ncol(h: H) -> i32;
    pub fn ss_sp_nnz(h: H) -> i64;
    pub fn ss_sp_copy(h: H, p: *mut i32, i: *mut i32, x: *mut f64);
    pub fn ss_sp_sorted(h: H) -> i32;
    pub fn ss_sp_free(h: H);

    pub fn ss_sdmult(nrow: i32, ncol: i32, p: *const i32, i: *const i32, x: *const f64, transpose: i32, k: i32, xx: *mut f64, yy: *mut f64) -> i32;
    pub fn ss_ssmult_h(am: i32, an: i32, ap: *const i32, ai: *const i32, ax: *const f64, bm: i32, bn: i32, bp: *const i32, bi: *const i32, bx: *const f64) -> H;
    pub fn ss_add_h(m: i32, n: i32, ap: *const i32, ai: *const i32, ax: *const f64, bp: *const i32, bi: *const i32, bx: *const f64) -> H;
    pub fn ss_transpose_h(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64) -> H;
    pub fn ss_triplet_h(m: i32, n: i32, nnz: i64, ti: *const i32, tj: *const i32, tx: *const f64) -> H;
    pub fn ss_submatrix_h(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64, rset: *mut i32, rsize: i64, cset: *mut i32, csize: i64) -> H;
    pub fn ss_band_h(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64, k1: i64, k2: i64) -> H;
    pub fn ss_norm(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64, norm: i32) -> f64;
    pub fn ss_to_dense(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64, out: *mut f64) -> i32;
    pub fn ss_from_dense_h(m: i32, n: i32, x: *mut f64) -> H;
    pub fn ss_symmetry(n: i32, p: *const i32, i: *const i32, x: *const f64) -> i32;

    pub fn ss_cs_nnz(h: H) -> i64;
    pub fn ss_cs_copy(h: H, p: *mut i32, i: *mut i32, x: *mut f64);
    pub fn ss_cs_free(h: H);
    pub fn ss_cs_multiply_h(am: i32, an: i32, ap: *const i32, ai: *const i32, ax: *const f64, bm: i32, bn: i32, bp: *const i32, bi: *const i32, bx: *const f64) -> H;
    pub fn ss_cs_add_h(m: i32, n: i32, ap: *const i32, ai: *const i32, ax: *const f64, bp: *const i32, bi: *const i32, bx: *const f64) -> H;
    pub fn ss_cs_transpose_h(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64) -> H;
    pub fn ss_cs_compress_h(m: i32, n: i32, nnz: i64, ti: *const i32, tj: *const i32, tx: *const f64, sort: i32) -> H;
    pub fn ss_cs_permute_h(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64, pinv: *const i32, q: *const i32) -> H;
    pub fn ss_cs_etree(n: i32, p: *const i32, i: *const i32, parent: *mut i32) -> i64;
    pub fn ss_cs_post(n: i32, parent: *const i32, post: *mut i32) -> i64;
    pub fn ss_cs_lu(n: i32, p: *const i32, i: *const i32, x: *const f64, order: i32, tol: f64) -> H;
    pub fn ss_cs_lu_nnz(h: H) -> i64;
    pub fn ss_cs_lu_solve(h: H, b: *const f64, x: *mut f64, work: *mut f64);
    pub fn ss_cs_lu_free(h: H);
    pub fn ss_cs_chol(n: i32, p: *const i32, i: *const i32, x: *const f64, order: i32) -> H;
    pub fn ss_cs_chol_nnz(h: H) -> i64;
    pub fn ss_cs_chol_solve(h: H, b: *const f64, x: *mut f64, work: *mut f64);
    pub fn ss_cs_chol_free(h: H);
    pub fn ss_cs_qr(m: i32, n: i32, p: *const i32, i: *const i32, x: *const f64, order: i32) -> H;
    pub fn ss_cs_qr_nnz(h: H) -> i64;
    pub fn ss_cs_qr_solve(h: H, b: *const f64, x: *mut f64, work: *mut f64);
    pub fn ss_cs_qr_m2(h: H) -> i32;
    pub fn ss_cs_qr_free(h: H);

    pub fn ss_amd(n: i32, p: *const i32, i: *const i32, perm: *mut i32) -> f64;
    pub fn ss_camd(n: i32, p: *const i32, i: *const i32, perm: *mut i32) -> f64;
    pub fn ss_colamd(m: i32, n: i32, p: *const i32, i: *const i32, perm: *mut i32) -> i32;
    pub fn ss_symamd(n: i32, p: *const i32, i: *const i32, perm: *mut i32) -> i32;
    pub fn ss_metis(n: i32, p: *const i32, i: *const i32, perm: *mut i32, postorder: i32) -> i32;
    pub fn ss_nesdis(n: i32, p: *const i32, i: *const i32, perm: *mut i32, cparent: *mut i32, cmember: *mut i32) -> i64;
    pub fn ss_etree(n: i32, p: *const i32, i: *const i32, parent: *mut i32) -> i32;
    pub fn ss_postorder(n: i32, parent: *const i32, post: *mut i32) -> i64;

    pub fn ss_chol(n: i32, p: *const i32, i: *const i32, x: *const f64, mode: i32, ll: i32, ordering: i32, perm: *const i32) -> H;
    pub fn ss_chol_lnz(h: H) -> f64;
    pub fn ss_chol_is_super(h: H) -> i32;
    pub fn ss_chol_nsuper(h: H) -> i64;
    pub fn ss_chol_solve(h: H, n: i32, k: i32, b: *mut f64, x: *mut f64) -> i32;
    pub fn ss_chol_solve2(h: H, n: i32, k: i32, b: *mut f64, x: *mut f64) -> i32;
    pub fn ss_chol_free(h: H);

    pub fn ss_ldl(n: i32, p: *const i32, i: *const i32, x: *const f64, perm: *const i32) -> H;
    pub fn ss_ldl_ok(h: H) -> i32;
    pub fn ss_ldl_lnz(h: H) -> i64;
    pub fn ss_ldl_solve(h: H, b: *const f64, x: *mut f64);
    pub fn ss_ldl_free(h: H);

    pub fn ss_umf(n: i32, p: *const i32, i: *const i32, x: *const f64, ordering: i32, strategy: i32, pivot_tol: f64, scale: i32, qinit: *const i32) -> H;
    pub fn ss_umf_nnz(h: H) -> i64;
    pub fn ss_umf_solve(h: H, sys: i32, b: *const f64, x: *mut f64) -> i32;
    pub fn ss_umf_rcond(h: H) -> f64;
    pub fn ss_umf_free(h: H);

    pub fn ss_klu(n: i32, p: *const i32, i: *const i32, x: *const f64, ordering: i32, btf: i32, tol: f64, scale: i32) -> H;
    pub fn ss_klu_nnz(h: H) -> i64;
    pub fn ss_klu_solve(h: H, sys: i32, b: *const f64, x: *mut f64) -> i32;
    pub fn ss_klu_rcond(h: H, p: *const i32, i: *const i32, x: *const f64) -> f64;
    pub fn ss_klu_free(h: H);

    pub fn ss_spqr(m: i64, n: i64, p: *const i64, i: *const i64, x: *const f64, ordering: i32, tol: f64) -> H;
    pub fn ss_spqr_solve(h: H, m: i64, n: i64, k: i32, b: *mut f64, x: *mut f64) -> i32;
    pub fn ss_spqr_free(h: H);
    pub fn ss_spqr_nnzR() -> f64;
    pub fn ss_spqr_nnzH() -> f64;
    pub fn ss_spqr_r(m: i64, n: i64, p: *const i64, i: *const i64, x: *const f64, rp: *mut *mut i64, ri: *mut *mut i64, rx: *mut *mut f64) -> i64;
    pub fn ss_free(p: *mut c_void);
}

/// Copies a CHOLMOD result (compressed columns) back into rows of `Csr` form, for comparison:
/// the columns of the result are the rows of its transpose.
pub fn cholmod_result(h: H) -> (usize, usize, Vec<i32>, Vec<i32>, Vec<f64>) {
    unsafe {
        let m = ss_sp_nrow(h) as usize;
        let n = ss_sp_ncol(h) as usize;
        let nnz = ss_sp_nnz(h) as usize;
        let mut p = vec![0i32; n + 1];
        let mut i = vec![0i32; nnz];
        let mut x = vec![0f64; nnz];
        ss_sp_copy(h, p.as_mut_ptr(), i.as_mut_ptr(), x.as_mut_ptr());
        (m, n, p, i, x)
    }
}

pub fn cs_result(h: H, n: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    unsafe {
        let nnz = ss_cs_nnz(h) as usize;
        let mut p = vec![0i32; n + 1];
        let mut i = vec![0i32; nnz];
        let mut x = vec![0f64; nnz];
        ss_cs_copy(h, p.as_mut_ptr(), i.as_mut_ptr(), x.as_mut_ptr());
        (p, i, x)
    }
}

/// The dense column-major matrix of compressed columns (for small checks).
pub fn csc_to_ours(m: usize, n: usize, p: &[i32], i: &[i32], x: &[f64]) -> a::math::sparse::SparseMatrix<f64> {
    let mut t = Vec::with_capacity(i.len());
    for j in 0..n {
        for k in p[j] as usize..p[j + 1] as usize {
            t.push((i[k] as usize, j, x[k]));
        }
    }
    a::math::sparse::SparseMatrix::from_coo(m, n, &t)
}
