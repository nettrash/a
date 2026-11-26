//! The same matrix in the forms of the Rust references.

#![allow(dead_code)]
use crate::mats::Csr;
use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseColMat, SymbolicSparseRowMat};

pub fn faer_csr(a: &Csr) -> SparseRowMat<usize, f64> {
    SparseRowMat::new(
        SymbolicSparseRowMat::new_checked(a.nrows, a.ncols, a.ptr.clone(), None, a.idx.clone()),
        a.val.clone(),
    )
}

pub fn faer_csr32(a: &Csr) -> SparseRowMat<u32, f64> {
    SparseRowMat::new(
        SymbolicSparseRowMat::new_checked(
            a.nrows,
            a.ncols,
            a.ptr.iter().map(|&v| v as u32).collect(),
            None,
            a.idx.iter().map(|&v| v as u32).collect(),
        ),
        a.val.clone(),
    )
}

/// The compressed columns of `a`.
pub fn faer_csc(a: &Csr) -> SparseColMat<usize, f64> {
    let t = a.transpose();
    SparseColMat::new(
        SymbolicSparseColMat::new_checked(a.nrows, a.ncols, t.ptr, None, t.idx),
        t.val,
    )
}

/// The lower triangle of a symmetric `a`, in compressed columns (= the upper triangle by rows).
pub fn faer_csc_lower(a: &Csr) -> SparseColMat<usize, f64> {
    let u = a.upper();
    SparseColMat::new(SymbolicSparseColMat::new_checked(a.nrows, a.ncols, u.ptr, None, u.idx), u.val)
}

pub fn sprs_csr(a: &Csr) -> sprs::CsMat<f64> {
    sprs::CsMat::new((a.nrows, a.ncols), a.ptr.clone(), a.idx.clone(), a.val.clone())
}

pub fn sprs_csc(a: &Csr) -> sprs::CsMat<f64> {
    let t = a.transpose();
    sprs::CsMat::new_csc((a.nrows, a.ncols), t.ptr, t.idx, t.val)
}

pub fn na_csr(a: &Csr) -> nalgebra_sparse::CsrMatrix<f64> {
    nalgebra_sparse::CsrMatrix::try_from_csr_data(a.nrows, a.ncols, a.ptr.clone(), a.idx.clone(), a.val.clone()).unwrap()
}

pub fn na_csc(a: &Csr) -> nalgebra_sparse::CscMatrix<f64> {
    let t = a.transpose();
    nalgebra_sparse::CscMatrix::try_from_csc_data(a.nrows, a.ncols, t.ptr, t.idx, t.val).unwrap()
}

pub fn from_faer_csc(c: &SparseColMat<usize, f64>) -> Csr {
    // The columns of c are the rows of its transpose.
    let (nr, nc) = (c.nrows(), c.ncols());
    let sym = c.symbolic();
    let ptr = sym.col_ptr().to_vec();
    let idx = sym.row_idx().to_vec();
    let t = Csr { nrows: nc, ncols: nr, ptr, idx, val: c.val().to_vec() };
    // Sort and sum through ours: faer's results are sorted, but the check must not assume it.
    let mut tr = Vec::with_capacity(t.nnz());
    for j in 0..nc {
        for k in t.ptr[j]..t.ptr[j + 1] {
            tr.push((t.idx[k], j, t.val[k]));
        }
    }
    Csr::from_ours(&a::math::sparse::SparseMatrix::from_coo(nr, nc, &tr))
}

pub fn from_sprs(m: &sprs::CsMat<f64>) -> Csr {
    let mut tr = Vec::with_capacity(m.nnz());
    for (v, (i, j)) in m.iter() {
        tr.push((i, j, *v));
    }
    Csr::from_ours(&a::math::sparse::SparseMatrix::from_coo(m.rows(), m.cols(), &tr))
}

pub fn from_na(m: &nalgebra_sparse::CsrMatrix<f64>) -> Csr {
    let mut tr = Vec::with_capacity(m.nnz());
    for (i, j, v) in m.triplet_iter() {
        tr.push((i, j, *v));
    }
    Csr::from_ours(&a::math::sparse::SparseMatrix::from_coo(m.nrows(), m.ncols(), &tr))
}

/// The largest difference between two matrices of the same shape, relative to the largest
/// element (the patterns may differ by explicit zeros).
pub fn mat_diff(a: &Csr, b: &Csr) -> f64 {
    assert_eq!((a.nrows, a.ncols), (b.nrows, b.ncols), "shapes differ");
    let d = &a.ours() - &b.ours();
    let scale = a.val.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    d.values().iter().fold(0.0f64, |m, v| m.max(v.abs())) / scale
}

pub fn vec_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}
