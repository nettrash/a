//! Storage and operations: products, sums, transposes, conversions, sub-matrices, norms.

use crate::conv::*;
use crate::mats::{self, Csr};
use crate::ss::{self, Csc32};
use crate::timing::{Contender, keep, note, part, race, rows};
use a::math::matrix::{Matrix, Norm};
use a::math::sparse::{CooMatrix, CscMatrix, SparseMatrix, symmetric_permutation};
use faer::{Accum, MatMut, MatRef, Par};
use faer::sparse::linalg::matmul::{dense_sparse_matmul, sparse_dense_matmul, sparse_sparse_matmul};
use std::ffi::c_long;

const G: &str = "storage";

fn params(name: &str, a: &Csr) -> String {
    format!("{name} n={} nnz={}", a.nrows, a.nnz())
}

fn check(label: &str, err: f64, tol: f64) {
    if !(err <= tol) {
        panic!("{label}: outputs differ by {err:e} (tolerance {tol:e})");
    }
}

pub fn run(quick: bool) {
    if part("spmv") {
        let spmv_set: Vec<&str> = if quick {
            vec!["lap2d-100", "randunsym-100000"]
        } else {
            vec!["lap2d-100", "lap2d-1000", "lap3d-100", "elast3d-30", "randunsym-1000000", "circuit-1000000"]
        };
        for name in &spmv_set {
            spmv(name, &mats::by_name(name));
        }
        let tset: Vec<&str> = if quick { vec!["randunsym-100000"] } else { vec!["randunsym-1000000", "elast3d-30", "lap2d-1000"] };
        for name in &tset {
            spmv_t(name, &mats::by_name(name));
        }
        csc_mul(if quick { "randunsym-100000" } else { "randunsym-1000000" });
        spmv_f32_c64(if quick { "lap2d-300" } else { "lap2d-1000" });
    }
    if part("spmm") {
        for name in if quick { vec!["lap2d-316"] } else { vec!["lap2d-316", "randunsym-100000"] } {
            let a = mats::by_name(name);
            for k in [4usize, 16, 64] {
                spmm(name, &a, k);
            }
        }
        dense_times_sparse("lap2d-316", 16);
    }
    if part("spgemm") {
        let gset: Vec<&str> = if quick { vec!["lap2d-316"] } else { vec!["lap2d-316", "lap3d-46", "randunsym-100000", "elast3d-16"] };
        for name in &gset {
            spgemm(name, &mats::by_name(name));
        }
        for name in if quick { vec!["randunsym-100000"] } else { vec!["randunsym-100000", "randunsym-1000000"] } {
            add(name, &mats::by_name(name));
        }
        scale(if quick { "randunsym-100000" } else { "randunsym-1000000" });
    }
    if part("transpose") {
        for name in if quick { vec!["randunsym-100000"] } else { vec!["randunsym-1000000", "lap3d-100", "elast3d-30"] } {
            transpose(name, &mats::by_name(name));
        }
    }
    if part("assembly") {
        for k in if quick { vec![300usize] } else { vec![300, 1000] } {
            assembly_fe(k);
        }
        assembly_random(if quick { 100_000 } else { 1_000_000 }, if quick { 1_000_000 } else { 10_000_000 });
    }
    if part("misc") {
        for name in if quick { vec!["lap2d-316"] } else { vec!["lap2d-1000", "elast3d-30"] } {
            permute(name, &mats::by_name(name));
        }
        submatrices(if quick { "lap2d-316" } else { "lap2d-1000" });
        triangles(if quick { "randunsym-100000" } else { "randunsym-1000000" });
        norms(if quick { "randunsym-100000" } else { "randunsym-1000000" });
        dense_conv(2000, 20);
        symmetry(if quick { "lap3d-30" } else { "lap3d-100" });
        from_raw(if quick { "lap2d-316" } else { "lap2d-1000" });
    }
}

/// The contenders for y = A·x on a matrix held by rows (ours, faer, sprs, nalgebra) or by
/// columns (CHOLMOD, Accelerate: their CSC of A, scattered; and, given our CSR arrays as the CSC
/// of Aᵀ, Aᵀᵀ·x by dot products).
fn spmv(name: &str, a: &Csr) {
    note(&format!("spmv {name}"));
    let n = a.nrows;
    let x = mats::rand_vec(a.ncols, 1);
    let ours = a.ours();
    let want = ours.mul_vec(&x);
    let fa = faer_csr(a);
    let fa32 = faer_csr32(a);
    let sp = sprs_csr(a);
    let na = na_csr(a);
    let csc = Csc32::of(a);
    let cst = Csc32::of_transpose(a);
    // Every reference is checked against ours before it is timed.
    let mut y = vec![0.0; n];
    sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, fa.as_ref(), MatRef::from_column_major_slice(&x, a.ncols, 1), 1.0, Par::Seq);
    check("faer spmv", vec_diff(&want, &y), 1e-13);
    let mut y2 = vec![0.0; n];
    unsafe { ss::ss_sdmult(csc.m, csc.n, csc.p.as_ptr(), csc.i.as_ptr(), csc.x.as_ptr(), 0, 1, x.as_ptr() as *mut f64, y2.as_mut_ptr()) };
    check("cholmod spmv", vec_diff(&want, &y2), 1e-13);
    let mut c: Vec<Contender> = Vec::new();
    {
        let mut y = vec![0.0; n];
        let ours = &ours;
        let x = &x;
        c.push(Contender::new("ours", move || {
            ours.mul_vec_into(x, &mut y);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (fa, x) = (&fa, &x);
        c.push(Contender::new("faer 0.24 (usize)", move || {
            sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, fa.as_ref(), MatRef::from_column_major_slice(x, x.len(), 1), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (fa, x) = (&fa32, &x);
        c.push(Contender::new("faer 0.24 (u32)", move || {
            sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, fa.as_ref(), MatRef::from_column_major_slice(x, x.len(), 1), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (sp, x) = (&sp, &x);
        c.push(Contender::new("sprs 0.11", move || {
            y.iter_mut().for_each(|v| *v = 0.0);
            sprs::prod::mul_acc_mat_vec_csr(sp.view(), &x[..], &mut y[..]);
            keep(&y);
        }));
    }
    {
        let mut y = nalgebra::DMatrix::<f64>::zeros(n, 1);
        let xm = nalgebra::DMatrix::<f64>::from_column_slice(a.ncols, 1, &x);
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11", move || {
            nalgebra_sparse::ops::serial::spmm_csr_dense(0.0, &mut y, 1.0, nalgebra_sparse::ops::Op::NoOp(na), nalgebra_sparse::ops::Op::NoOp(&xm));
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (s, x) = (&csc, &x);
        c.push(Contender::new("CHOLMOD sdmult (CSC, A·x)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 0, 1, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (s, x) = (&cst, &x);
        c.push(Contender::new("CHOLMOD sdmult (CSC of Aᵀ, transposed)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 1, 1, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    #[cfg(have_accelerate)]
    {
        let mut y = vec![0.0; n];
        let mut csl: Vec<c_long> = csc.plong();
        let mut ri = csc.i.clone();
        let mut xv = csc.x.clone();
        let mut xx = x.clone();
        let (m_, n_) = (csc.m, csc.n);
        c.push(Contender::new("Accelerate SparseMultiply (CSC)", move || {
            unsafe { crate::accel::ac_mul(m_, n_, csl.as_mut_ptr(), ri.as_mut_ptr(), xv.as_mut_ptr(), 0, 1, xx.as_mut_ptr(), y.as_mut_ptr()) };
            keep(&y);
        }));
        let mut y = vec![0.0; n];
        let mut csl: Vec<c_long> = cst.plong();
        let mut ri = cst.i.clone();
        let mut xv = cst.x.clone();
        let mut xx = x.clone();
        let (m_, n_) = (cst.m, cst.n);
        c.push(Contender::new("Accelerate SparseMultiply (CSC of Aᵀ, transposed)", move || {
            unsafe { crate::accel::ac_mul(m_, n_, csl.as_mut_ptr(), ri.as_mut_ptr(), xv.as_mut_ptr(), 1, 1, xx.as_mut_ptr(), y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    let _ = c_long::default();
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[3] = "y zeroed then y += A·x (sprs accumulates)".into();
    notes[5] = "int32 indices".into();
    notes[6] = "int32 indices; our CSR arrays read as CSC of Aᵀ".into();
    if st.len() > 7 {
        notes[7] = "int32 indices; single-threaded BLAS setting; may use AMX".into();
        notes[8] = "int32 indices; CSR arrays as CSC of Aᵀ with the transpose attribute".into();
    }
    rows(G, "mul_vec (A·x)", &params(name, a), &st, &notes);
}

/// y = Aᵀ·x without forming Aᵀ.
fn spmv_t(name: &str, a: &Csr) {
    note(&format!("spmv_t {name}"));
    let (m, n) = (a.nrows, a.ncols);
    let x = mats::rand_vec(m, 2);
    let ours = a.ours();
    let want = ours.mul_vec_transpose(&x);
    let fa = faer_csr(a);
    let sp = sprs_csr(a);
    let na = na_csr(a);
    let cst = Csc32::of_transpose(a); // CSC of Aᵀ = our arrays
    let csc = Csc32::of(a);
    let mut y = vec![0.0; n];
    sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, fa.as_ref().transpose(), MatRef::from_column_major_slice(&x, m, 1), 1.0, Par::Seq);
    check("faer spmv_t", vec_diff(&want, &y), 1e-13);
    let mut c: Vec<Contender> = Vec::new();
    {
        let mut y = vec![0.0; n];
        let (ours, x) = (&ours, &x);
        c.push(Contender::new("ours", move || {
            ours.mul_vec_transpose_into(x, &mut y);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (fa, x) = (&fa, &x);
        c.push(Contender::new("faer 0.24 (transpose view, CSC scatter)", move || {
            sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, fa.as_ref().transpose(), MatRef::from_column_major_slice(x, m, 1), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (sp, x) = (&sp, &x);
        c.push(Contender::new("sprs 0.11 (transpose view)", move || {
            y.iter_mut().for_each(|v| *v = 0.0);
            sprs::prod::mul_acc_mat_vec_csc(sp.transpose_view(), &x[..], &mut y[..]);
            keep(&y);
        }));
    }
    {
        let mut y = nalgebra::DMatrix::<f64>::zeros(n, 1);
        let xm = nalgebra::DMatrix::<f64>::from_column_slice(m, 1, &x);
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11 (Op::Transpose)", move || {
            nalgebra_sparse::ops::serial::spmm_csr_dense(0.0, &mut y, 1.0, nalgebra_sparse::ops::Op::Transpose(na), nalgebra_sparse::ops::Op::NoOp(&xm));
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (s, x) = (&cst, &x);
        c.push(Contender::new("CHOLMOD sdmult (our arrays as CSC of Aᵀ, scatter)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 0, 1, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (s, x) = (&csc, &x);
        c.push(Contender::new("CHOLMOD sdmult (CSC of A, Aᵀ·x by dots)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 1, 1, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    #[cfg(have_accelerate)]
    {
        let mut y = vec![0.0; n];
        let mut csl: Vec<c_long> = cst.plong();
        let mut ri = cst.i.clone();
        let mut xv = cst.x.clone();
        let mut xx = x.clone();
        let (m_, n_) = (cst.m, cst.n);
        c.push(Contender::new("Accelerate SparseMultiply (our arrays as CSC of Aᵀ)", move || {
            unsafe { crate::accel::ac_mul(m_, n_, csl.as_mut_ptr(), ri.as_mut_ptr(), xv.as_mut_ptr(), 0, 1, xx.as_mut_ptr(), y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[2] = "y zeroed then accumulated".into();
    notes[5] = "int32 indices; needs the CSC of A, a second copy of the matrix: the same as our CSR mul_vec".into();
    rows(G, "mul_vec_transpose (Aᵀ·x)", &params(name, a), &st, &notes);
}

/// CscMatrix::mul_vec: y = A·x from compressed columns (a scatter), against the column-oriented
/// references.
fn csc_mul(name: &str) {
    note(&format!("csc mul {name}"));
    let a = mats::by_name(name);
    let n = a.nrows;
    let x = mats::rand_vec(a.ncols, 3);
    let oc: CscMatrix<f64> = a.ours().to_csc();
    let want = a.ours().mul_vec(&x);
    check("csc mul", vec_diff(&want, &oc.mul_vec(&x)), 1e-13);
    let fc = faer_csc(&a);
    let csc = Csc32::of(&a);
    let mut c: Vec<Contender> = Vec::new();
    {
        let mut y = vec![0.0; n];
        let (oc, x) = (&oc, &x);
        c.push(Contender::new("ours", move || {
            oc.mul_vec_into(x, &mut y);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (fc, x) = (&fc, &x);
        c.push(Contender::new("faer 0.24 (CSC)", move || {
            sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, fc.as_ref(), MatRef::from_column_major_slice(x, x.len(), 1), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; n];
        let (s, x) = (&csc, &x);
        c.push(Contender::new("CHOLMOD sdmult (CSC)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 0, 1, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    #[cfg(have_accelerate)]
    {
        let mut y = vec![0.0; n];
        let mut csl: Vec<c_long> = csc.plong();
        let mut ri = csc.i.clone();
        let mut xv = csc.x.clone();
        let mut xx = x.clone();
        let (m_, n_) = (csc.m, csc.n);
        c.push(Contender::new("Accelerate SparseMultiply (CSC)", move || {
            unsafe { crate::accel::ac_mul(m_, n_, csl.as_mut_ptr(), ri.as_mut_ptr(), xv.as_mut_ptr(), 0, 1, xx.as_mut_ptr(), y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[2] = "int32 indices".into();
    rows(G, "CscMatrix::mul_vec (A·x by columns)", &params(name, &a), &st, &notes);
}

/// SpMV in f32 and in Complex<f64>.
fn spmv_f32_c64(name: &str) {
    note(&format!("spmv f32/c64 {name}"));
    use a::math::complex::Complex;
    let a = mats::by_name(name);
    let n = a.nrows;
    // f32
    {
        let o32: SparseMatrix<f32> = SparseMatrix::from_csr(n, n, a.ptr.clone(), a.idx.clone(), a.val.iter().map(|&v| v as f32).collect());
        let x: Vec<f32> = mats::rand_vec(n, 4).into_iter().map(|v| v as f32).collect();
        let f32m = faer::sparse::SparseRowMat::<usize, f32>::new(
            faer::sparse::SymbolicSparseRowMat::new_checked(n, n, a.ptr.clone(), None, a.idx.clone()),
            a.val.iter().map(|&v| v as f32).collect(),
        );
        let sp = sprs::CsMat::new((n, n), a.ptr.clone(), a.idx.clone(), a.val.iter().map(|&v| v as f32).collect::<Vec<f32>>());
        let mut c: Vec<Contender> = Vec::new();
        {
            let mut y = vec![0.0f32; n];
            let (o, x) = (&o32, &x);
            c.push(Contender::new("ours", move || {
                o.mul_vec_into(x, &mut y);
                keep(&y);
            }));
        }
        {
            let mut y = vec![0.0f32; n];
            let (f, x) = (&f32m, &x);
            c.push(Contender::new("faer 0.24 (usize)", move || {
                sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, f.as_ref(), MatRef::from_column_major_slice(x, n, 1), 1.0f32, Par::Seq);
                keep(&y);
            }));
        }
        {
            let mut y = vec![0.0f32; n];
            let (sp, x) = (&sp, &x);
            c.push(Contender::new("sprs 0.11", move || {
                y.iter_mut().for_each(|v| *v = 0.0);
                sprs::prod::mul_acc_mat_vec_csr(sp.view(), &x[..], &mut y[..]);
                keep(&y);
            }));
        }
        #[cfg(have_accelerate)]
        {
            let cst = Csc32::of_transpose(&a);
            let mut y = vec![0.0f32; n];
            let mut csl: Vec<c_long> = cst.plong();
            let mut ri = cst.i.clone();
            let mut xv: Vec<f32> = cst.x.iter().map(|&v| v as f32).collect();
            let mut xx = x.clone();
            c.push(Contender::new("Accelerate SparseMultiply (CSC of Aᵀ, transposed)", move || {
                unsafe { crate::accel::ac_mul_f32(n as i32, n as i32, csl.as_mut_ptr(), ri.as_mut_ptr(), xv.as_mut_ptr(), 1, xx.as_mut_ptr(), y.as_mut_ptr()) };
                keep(&y);
            }));
        }
        let st = race(&mut c);
        rows(G, "mul_vec (A·x) f32", &params(name, &a), &st, &vec![String::new(); st.len()]);
    }
    // Complex<f64>
    {
        let vals: Vec<Complex<f64>> = a.val.iter().enumerate().map(|(k, &v)| Complex { re: v, im: 0.25 * ((k % 7) as f64 - 3.0) }).collect();
        let oc: SparseMatrix<Complex<f64>> = SparseMatrix::from_csr(n, n, a.ptr.clone(), a.idx.clone(), vals.clone());
        let xr = mats::rand_vec(2 * n, 5);
        let x: Vec<Complex<f64>> = (0..n).map(|i| Complex { re: xr[2 * i], im: xr[2 * i + 1] }).collect();
        let fv: Vec<faer::c64> = vals.iter().map(|z| faer::c64::new(z.re, z.im)).collect();
        let fm = faer::sparse::SparseRowMat::<usize, faer::c64>::new(faer::sparse::SymbolicSparseRowMat::new_checked(n, n, a.ptr.clone(), None, a.idx.clone()), fv);
        let fx: Vec<faer::c64> = x.iter().map(|z| faer::c64::new(z.re, z.im)).collect();
        let sv: Vec<num_complex::Complex64> = vals.iter().map(|z| num_complex::Complex64::new(z.re, z.im)).collect();
        let sp = sprs::CsMat::new((n, n), a.ptr.clone(), a.idx.clone(), sv);
        let sx: Vec<num_complex::Complex64> = x.iter().map(|z| num_complex::Complex64::new(z.re, z.im)).collect();
        // Check faer against ours.
        let want = oc.mul_vec(&x);
        let mut fy = vec![faer::c64::new(0.0, 0.0); n];
        sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut fy, n, 1), Accum::Replace, fm.as_ref(), MatRef::from_column_major_slice(&fx, n, 1), faer::c64::new(1.0, 0.0), Par::Seq);
        let err = want.iter().zip(&fy).map(|(w, f)| (w.re - f.re).abs() + (w.im - f.im).abs()).fold(0.0, f64::max);
        check("faer c64", err, 1e-12);
        let mut c: Vec<Contender> = Vec::new();
        {
            let mut y = vec![Complex { re: 0.0, im: 0.0 }; n];
            let (o, x) = (&oc, &x);
            c.push(Contender::new("ours", move || {
                o.mul_vec_into(x, &mut y);
                keep(&y);
            }));
        }
        {
            let mut y = vec![faer::c64::new(0.0, 0.0); n];
            let (f, x) = (&fm, &fx);
            c.push(Contender::new("faer 0.24 (usize)", move || {
                sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, n, 1), Accum::Replace, f.as_ref(), MatRef::from_column_major_slice(x, n, 1), faer::c64::new(1.0, 0.0), Par::Seq);
                keep(&y);
            }));
        }
        {
            let mut y = vec![num_complex::Complex64::new(0.0, 0.0); n];
            let (sp, x) = (&sp, &sx);
            c.push(Contender::new("sprs 0.11", move || {
                y.iter_mut().for_each(|v| *v = num_complex::Complex64::new(0.0, 0.0));
                sprs::prod::mul_acc_mat_vec_csr(sp.view(), &x[..], &mut y[..]);
                keep(&y);
            }));
        }
        let st = race(&mut c);
        rows(G, "mul_vec (A·x) Complex<f64>", &params(name, &a), &st, &vec![String::new(); st.len()]);
    }
}

/// Y = A·X for a dense X of k columns.
fn spmm(name: &str, a: &Csr, k: usize) {
    note(&format!("spmm {name} k={k}"));
    let (m, n) = (a.nrows, a.ncols);
    let xs = mats::rand_vec(n * k, 6);
    let xm = Matrix::from_column_major(n, k, xs.clone());
    let ours = a.ours();
    let want = ours.mul_matrix(&xm);
    let fa = faer_csr(a);
    let na = na_csr(a);
    let sp = sprs_csr(a);
    let csc = Csc32::of(a);
    let cst = Csc32::of_transpose(a);
    let mut y = vec![0.0; m * k];
    sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, m, k), Accum::Replace, fa.as_ref(), MatRef::from_column_major_slice(&xs, n, k), 1.0, Par::Seq);
    check("faer spmm", vec_diff(want.as_slice(), &y), 1e-13);
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, x) = (&ours, &xm);
        c.push(Contender::new("ours", move || keep(o.mul_matrix(x))));
    }
    {
        let mut y = vec![0.0; m * k];
        let (fa, x) = (&fa, &xs);
        c.push(Contender::new("faer 0.24", move || {
            sparse_dense_matmul(MatMut::from_column_major_slice_mut(&mut y, m, k), Accum::Replace, fa.as_ref(), MatRef::from_column_major_slice(x, n, k), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    {
        let mut y = nalgebra::DMatrix::<f64>::zeros(m, k);
        let xn = nalgebra::DMatrix::<f64>::from_column_slice(n, k, &xs);
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11", move || {
            nalgebra_sparse::ops::serial::spmm_csr_dense(0.0, &mut y, 1.0, nalgebra_sparse::ops::Op::NoOp(na), nalgebra_sparse::ops::Op::NoOp(&xn));
            keep(&y);
        }));
    }
    {
        // sprs: CSR × dense (row-major ndarray, its fast layout for a CSR left factor).
        let xa = ndarray::Array2::from_shape_fn((n, k), |(i, j)| xs[j * n + i]);
        let sp = &sp;
        c.push(Contender::new("sprs 0.11 (row-major dense)", move || keep(sp * &xa)));
    }
    {
        let mut y = vec![0.0; m * k];
        let (s, x) = (&csc, &xs);
        c.push(Contender::new("CHOLMOD sdmult (CSC)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 0, k as i32, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; m * k];
        let (s, x) = (&cst, &xs);
        c.push(Contender::new("CHOLMOD sdmult (CSC of Aᵀ, transposed)", move || {
            unsafe { ss::ss_sdmult(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 1, k as i32, x.as_ptr() as *mut f64, y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    #[cfg(have_accelerate)]
    {
        let mut y = vec![0.0; m * k];
        let mut csl: Vec<c_long> = csc.plong();
        let mut ri = csc.i.clone();
        let mut xv = csc.x.clone();
        let mut xx = xs.clone();
        let (m_, n_) = (csc.m, csc.n);
        c.push(Contender::new("Accelerate SparseMultiply (CSC)", move || {
            unsafe { crate::accel::ac_mul(m_, n_, csl.as_mut_ptr(), ri.as_mut_ptr(), xv.as_mut_ptr(), 0, k as i32, xx.as_mut_ptr(), y.as_mut_ptr()) };
            keep(&y);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "ours allocates its result, the references write into a given one (except sprs)".into();
    notes[3] = "allocates its result".into();
    rows(G, "mul_matrix (A·X dense)", &format!("{} k={k}", params(name, a)), &st, &notes);
}

/// Y = X·A for a dense X of k rows.
fn dense_times_sparse(name: &str, k: usize) {
    note(&format!("dense×sparse {name} k={k}"));
    let a = mats::by_name(name);
    let (m, n) = (a.nrows, a.ncols);
    let xs = mats::rand_vec(k * m, 7);
    let xm = Matrix::from_column_major(k, m, xs.clone());
    let ours = a.ours();
    let want = &xm * &ours;
    let fa = faer_csr(&a);
    let fc = faer_csc(&a);
    let mut y = vec![0.0; k * n];
    dense_sparse_matmul(MatMut::from_column_major_slice_mut(&mut y, k, n), Accum::Replace, MatRef::from_column_major_slice(&xs, k, m), fa.as_ref(), 1.0, Par::Seq);
    check("faer dense×sparse", vec_diff(want.as_slice(), &y), 1e-13);
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, x) = (&ours, &xm);
        c.push(Contender::new("ours", move || keep(x * o)));
    }
    {
        let mut y = vec![0.0; k * n];
        let (fa, x) = (&fa, &xs);
        c.push(Contender::new("faer 0.24 (CSR right factor)", move || {
            dense_sparse_matmul(MatMut::from_column_major_slice_mut(&mut y, k, n), Accum::Replace, MatRef::from_column_major_slice(x, k, m), fa.as_ref(), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    {
        let mut y = vec![0.0; k * n];
        let (fc, x) = (&fc, &xs);
        c.push(Contender::new("faer 0.24 (CSC right factor)", move || {
            dense_sparse_matmul(MatMut::from_column_major_slice_mut(&mut y, k, n), Accum::Replace, MatRef::from_column_major_slice(x, k, m), fc.as_ref(), 1.0, Par::Seq);
            keep(&y);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "ours allocates its result".into();
    rows(G, "dense × sparse (X·A)", &format!("{} k={k}", params(name, &a)), &st, &notes);
}

/// C = A·A (sparse × sparse).
fn spgemm(name: &str, a: &Csr) {
    note(&format!("spgemm {name}"));
    let ours = a.ours();
    let want = Csr::from_ours(&(&ours * &ours));
    let fc = faer_csc(a);
    let sp = sprs_csr(a);
    let na = na_csr(a);
    let csc = Csc32::of(a);
    // Checks.
    let fr = sparse_sparse_matmul(fc.as_ref(), fc.as_ref(), 1.0, Par::Seq).unwrap();
    check("faer spgemm", mat_diff(&want, &from_faer_csc(&fr)), 1e-13);
    check("sprs spgemm", mat_diff(&want, &from_sprs(&(&sp * &sp))), 1e-13);
    check("nalgebra spgemm", mat_diff(&want, &from_na(&(&na * &na))), 1e-13);
    let h = unsafe { ss::ss_ssmult_h(csc.m, csc.n, csc.p.as_ptr(), csc.i.as_ptr(), csc.x.as_ptr(), csc.m, csc.n, csc.p.as_ptr(), csc.i.as_ptr(), csc.x.as_ptr()) };
    let (mm, nn, p, i, x) = ss::cholmod_result(h);
    unsafe { ss::ss_sp_free(h) };
    check("cholmod spgemm", mat_diff(&want, &Csr::from_ours(&ss::csc_to_ours(mm, nn, &p, &i, &x))), 1e-13);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &ours;
        c.push(Contender::new("ours", move || keep(o * o)));
    }
    {
        let fc = &fc;
        c.push(Contender::new("faer 0.24 (symbolic + numeric)", move || keep(sparse_sparse_matmul(fc.as_ref(), fc.as_ref(), 1.0, Par::Seq).unwrap())));
    }
    {
        let sp = &sp;
        c.push(Contender::new("sprs 0.11", move || keep(sp * sp)));
    }
    {
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11", move || keep(na * na)));
    }
    {
        let s = &csc;
        c.push(Contender::new("CHOLMOD ssmult (sorted)", move || unsafe {
            let h = ss::ss_ssmult_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr());
            ss::ss_sp_free(h);
        }));
    }
    {
        let s = &csc;
        c.push(Contender::new("CSparse cs_multiply (unsorted)", move || unsafe {
            let h = ss::ss_cs_multiply_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr());
            ss::ss_cs_free(h);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = format!("nnz(A·A)={}", want.nnz());
    notes[4] = "int32 indices".into();
    notes[5] = "int32; leaves columns unsorted and explicit zeros (less work)".into();
    rows(G, "sparse × sparse (A·A)", &params(name, a), &st, &notes);
}

/// C = A + Aᵀ (two different patterns).
fn add(name: &str, a: &Csr) {
    note(&format!("add {name}"));
    let b = a.transpose();
    let (oa, ob) = (a.ours(), b.ours());
    let want = Csr::from_ours(&(&oa + &ob));
    let (sa, sb) = (sprs_csr(a), sprs_csr(&b));
    let (na_, nb) = (na_csr(a), na_csr(&b));
    let (fa, fb) = (faer_csc(a), faer_csc(&b));
    let (ca, cb) = (Csc32::of(a), Csc32::of(&b));
    check("sprs add", mat_diff(&want, &from_sprs(&(&sa + &sb))), 1e-14);
    check("nalgebra add", mat_diff(&want, &from_na(&(&na_ + &nb))), 1e-14);
    check("faer add", mat_diff(&want, &from_faer_csc(&faer::sparse::ops::add(fa.as_ref(), fb.as_ref()).unwrap())), 1e-14);
    let mut c: Vec<Contender> = Vec::new();
    {
        let (x, y) = (&oa, &ob);
        c.push(Contender::new("ours", move || keep(x + y)));
    }
    {
        let (x, y) = (&fa, &fb);
        c.push(Contender::new("faer 0.24 (ops::add)", move || keep(faer::sparse::ops::add(x.as_ref(), y.as_ref()).unwrap())));
    }
    {
        let (x, y) = (&sa, &sb);
        c.push(Contender::new("sprs 0.11", move || keep(x + y)));
    }
    {
        let (x, y) = (&na_, &nb);
        c.push(Contender::new("nalgebra-sparse 0.11", move || keep(x + y)));
    }
    {
        let (x, y) = (&ca, &cb);
        c.push(Contender::new("CHOLMOD add (sorted)", move || unsafe {
            let h = ss::ss_add_h(x.m, x.n, x.p.as_ptr(), x.i.as_ptr(), x.x.as_ptr(), y.p.as_ptr(), y.i.as_ptr(), y.x.as_ptr());
            ss::ss_sp_free(h);
        }));
    }
    {
        let (x, y) = (&ca, &cb);
        c.push(Contender::new("CSparse cs_add (unsorted)", move || unsafe {
            let h = ss::ss_cs_add_h(x.m, x.n, x.p.as_ptr(), x.i.as_ptr(), x.x.as_ptr(), y.p.as_ptr(), y.i.as_ptr(), y.x.as_ptr());
            ss::ss_cs_free(h);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = format!("B = Aᵀ; nnz(A+B)={}", want.nnz());
    notes[5] = "unsorted result (less work)".into();
    rows(G, "add (A + B)", &params(name, a), &st, &notes);
}

fn scale(name: &str) {
    note(&format!("scale {name}"));
    let a = mats::by_name(name);
    let o = a.ours();
    let sp = sprs_csr(&a);
    let na = na_csr(&a);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o * 1.5)));
    }
    {
        let sp = &sp;
        c.push(Contender::new("sprs 0.11 (map)", move || keep(sp.map(|v| v * 1.5))));
    }
    {
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11", move || keep(na * 1.5)));
    }
    let st = race(&mut c);
    rows(G, "scale (A·α)", &params(name, &a), &st, &vec![String::new(); st.len()]);
}

/// Aᵀ (and the conversion to compressed columns, which is the same work).
fn transpose(name: &str, a: &Csr) {
    note(&format!("transpose {name}"));
    let o = a.ours();
    let want = a.transpose();
    let fa = faer_csr(a);
    let sp = sprs_csr(a);
    let na = na_csr(a);
    let cs = Csc32::of_transpose(a);
    check("faer transpose", mat_diff(&want, &from_faer_csc(&fa.as_ref().to_col_major().unwrap()).transpose()), 0.0);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o.transpose())));
    }
    {
        let fa = &fa;
        c.push(Contender::new("faer 0.24 (to_col_major)", move || keep(fa.as_ref().to_col_major().unwrap())));
    }
    {
        let sp = &sp;
        c.push(Contender::new("sprs 0.11 (to_other_storage)", move || keep(sp.to_other_storage())));
    }
    {
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11 (transpose)", move || keep(na.transpose())));
    }
    {
        let s = &cs;
        c.push(Contender::new("CHOLMOD transpose", move || unsafe {
            let h = ss::ss_transpose_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr());
            ss::ss_sp_free(h);
        }));
    }
    {
        let s = &cs;
        c.push(Contender::new("CSparse cs_transpose", move || unsafe {
            let h = ss::ss_cs_transpose_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr());
            ss::ss_cs_free(h);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "= to_csc() and CscMatrix::to_csr(), the same counting sort".into();
    notes[4] = "int32 indices".into();
    notes[5] = "int32 indices".into();
    rows(G, "transpose (Aᵀ)", &params(name, a), &st, &notes);
}

/// Assembly from the triplets of a finite-element mesh: four repeats of every interior cell.
fn assembly_fe(k: usize) {
    note(&format!("assembly fe {k}"));
    let (n, ri, ci, v) = mats::fe_triplets(k);
    assembly(&format!("FE assembly {k}×{k} Q1, {} triplets", ri.len()), n, ri, ci, v);
}

fn assembly_random(n: usize, nnz: usize) {
    note(&format!("assembly random {n} {nnz}"));
    let (ri, ci, v) = mats::random_triplets(n, nnz, 9);
    assembly(&format!("random triplets n={n}, {nnz} triplets"), n, ri, ci, v);
}

fn assembly(params: &str, n: usize, ri: Vec<usize>, ci: Vec<usize>, v: Vec<f64>) {
    let coo = CooMatrix::from_triplets(n, n, ri.clone(), ci.clone(), v.clone());
    let want = Csr::from_ours(&coo.to_csr());
    let tup: Vec<(usize, usize, f64)> = (0..v.len()).map(|k| (ri[k], ci[k], v[k])).collect();
    let tri = sprs::TriMat::from_triplets((n, n), ri.clone(), ci.clone(), v.clone());
    let nac = nalgebra_sparse::CooMatrix::try_from_triplets(n, n, ri.clone(), ci.clone(), v.clone()).unwrap();
    let ftrip: Vec<faer::sparse::Triplet<usize, usize, f64>> = (0..v.len()).map(|k| faer::sparse::Triplet::new(ri[k], ci[k], v[k])).collect();
    let ri32: Vec<i32> = ri.iter().map(|&x| x as i32).collect();
    let ci32: Vec<i32> = ci.iter().map(|&x| x as i32).collect();
    check("sprs assembly", mat_diff(&want, &from_sprs(&tri.to_csr())), 1e-15);
    check("nalgebra assembly", mat_diff(&want, &from_na(&nalgebra_sparse::CsrMatrix::from(&nac))), 1e-15);
    let f = faer::sparse::SparseRowMat::<usize, f64>::try_new_from_triplets(n, n, &ftrip).unwrap();
    check("faer assembly", mat_diff(&want, &from_faer_csc(&f.as_ref().to_col_major().unwrap())), 1e-15);
    let mut c: Vec<Contender> = Vec::new();
    {
        let coo = &coo;
        c.push(Contender::new("ours", move || keep(coo.to_csr())));
    }
    {
        let ftrip = &ftrip;
        c.push(Contender::new("faer 0.24 (try_new_from_triplets)", move || keep(faer::sparse::SparseRowMat::<usize, f64>::try_new_from_triplets(n, n, ftrip).unwrap())));
    }
    {
        let tri = &tri;
        c.push(Contender::new("sprs 0.11 (TriMat::to_csr)", move || keep(tri.to_csr::<usize>())));
    }
    {
        let nac = &nac;
        c.push(Contender::new("nalgebra-sparse 0.11 (CsrMatrix::from(&coo))", move || keep(nalgebra_sparse::CsrMatrix::from(nac))));
    }
    {
        let (r, cc, v) = (&ri32, &ci32, &v);
        c.push(Contender::new("CHOLMOD triplet_to_sparse", move || unsafe {
            let h = ss::ss_triplet_h(n as i32, n as i32, v.len() as i64, r.as_ptr(), cc.as_ptr(), v.as_ptr());
            ss::ss_sp_free(h);
        }));
    }
    {
        let (r, cc, v) = (&ri32, &ci32, &v);
        c.push(Contender::new("CSparse cs_compress + cs_dupl (unsorted)", move || unsafe {
            let h = ss::ss_cs_compress_h(n as i32, n as i32, v.len() as i64, r.as_ptr(), cc.as_ptr(), v.as_ptr(), 0);
            ss::ss_cs_free(h);
        }));
    }
    {
        let (r, cc, v) = (&ri32, &ci32, &v);
        c.push(Contender::new("CSparse cs_compress + cs_dupl + sort", move || unsafe {
            let h = ss::ss_cs_compress_h(n as i32, n as i32, v.len() as i64, r.as_ptr(), cc.as_ptr(), v.as_ptr(), 1);
            ss::ss_cs_free(h);
        }));
    }
    #[cfg(have_accelerate)]
    {
        let (r, cc, v) = (&ri32, &ci32, &v);
        c.push(Contender::new("Accelerate SparseConvertFromCoordinate", move || unsafe {
            keep(crate::accel::ac_from_coo(n as i32, n as i32, v.len() as c_long, r.as_ptr(), cc.as_ptr(), v.as_ptr()));
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = format!("CooMatrix::to_csr; nnz={}", want.nnz());
    notes[4] = "int32; sums repeats, keeps a zero sum".into();
    notes[5] = "int32; columns left unsorted".into();
    notes[6] = "int32; sorted by a double transpose".into();
    if st.len() > 7 {
        notes[7] = "int32".into();
    }
    rows(G, "assembly (triplets → CSR)", params, &st, &notes);
    // from_coo on a slice of tuples, against the same references (the fastest of them matters).
    let mut c: Vec<Contender> = Vec::new();
    {
        let tup = &tup;
        c.push(Contender::new("ours", move || keep(SparseMatrix::from_coo(n, n, tup))));
    }
    {
        let ftrip = &ftrip;
        c.push(Contender::new("faer 0.24 (try_new_from_triplets)", move || keep(faer::sparse::SparseRowMat::<usize, f64>::try_new_from_triplets(n, n, ftrip).unwrap())));
    }
    {
        let tri = &tri;
        c.push(Contender::new("sprs 0.11 (TriMat::to_csr)", move || keep(tri.to_csr::<usize>())));
    }
    {
        let (r, cc, v) = (&ri32, &ci32, &v);
        c.push(Contender::new("CHOLMOD triplet_to_sparse", move || unsafe {
            let h = ss::ss_triplet_h(n as i32, n as i32, v.len() as i64, r.as_ptr(), cc.as_ptr(), v.as_ptr());
            ss::ss_sp_free(h);
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "SparseMatrix::from_coo(&[(i, j, x)])".into();
    rows(G, "from_coo (tuples → CSR)", params, &st, &notes);
}

/// P·A·Pᵀ for a random permutation.
fn permute(name: &str, a: &Csr) {
    note(&format!("permute {name}"));
    let n = a.nrows;
    let p = mats::random_perm(n, 10);
    let o = a.ours();
    let want = Csr::from_ours(&o.permute(&p, &p));
    let mut pinv = vec![0i32; n];
    for (k, &v) in p.iter().enumerate() {
        pinv[v] = k as i32;
    }
    let p32: Vec<i32> = p.iter().map(|&v| v as i32).collect();
    let csc = Csc32::of(a);
    // CHOLMOD's submatrix with both sets the permutation: C = A(p, p).
    let h = unsafe {
        let mut r = p32.clone();
        let mut cc = p32.clone();
        ss::ss_submatrix_h(csc.m, csc.n, csc.p.as_ptr(), csc.i.as_ptr(), csc.x.as_ptr(), r.as_mut_ptr(), n as i64, cc.as_mut_ptr(), n as i64)
    };
    let (mm, nn, pp, ii, xx) = ss::cholmod_result(h);
    unsafe { ss::ss_sp_free(h) };
    check("cholmod permute", mat_diff(&want, &Csr::from_ours(&ss::csc_to_ours(mm, nn, &pp, &ii, &xx))), 0.0);
    let sp = sprs_csr(a);
    let sperm = sprs::PermOwned::new(p.clone());
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, p) = (&o, &p);
        c.push(Contender::new("ours", move || keep(o.permute(p, p))));
    }
    {
        let (s, pinv, q) = (&csc, &pinv, &p32);
        c.push(Contender::new("CSparse cs_permute (unsorted)", move || unsafe {
            let h = ss::ss_cs_permute_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), pinv.as_ptr(), q.as_ptr());
            ss::ss_cs_free(h);
        }));
    }
    {
        let (s, p) = (&csc, &p32);
        let mut r = p.clone();
        let mut cc = p.clone();
        c.push(Contender::new("CHOLMOD submatrix (A(p,p), sorted)", move || unsafe {
            let h = ss::ss_submatrix_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), r.as_mut_ptr(), n as i64, cc.as_mut_ptr(), n as i64);
            ss::ss_sp_free(h);
        }));
    }
    {
        let (sp, pm) = (&sp, &sperm);
        c.push(Contender::new("sprs 0.11 (transform_mat_papt)", move || keep(sprs::transform_mat_papt(sp.view(), pm.view()))));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[1] = "int32; columns left unsorted".into();
    notes[2] = "int32".into();
    rows(G, "permute (P·A·Pᵀ)", &params(name, a), &st, &notes);
    // symmetric_permutation of the ordering module: the same work.
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, p) = (&o, &p);
        c.push(Contender::new("ours", move || keep(symmetric_permutation(o, p))));
    }
    {
        let (s, pinv, q) = (&csc, &pinv, &p32);
        c.push(Contender::new("CSparse cs_permute (unsorted)", move || unsafe {
            let h = ss::ss_cs_permute_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), pinv.as_ptr(), q.as_ptr());
            ss::ss_cs_free(h);
        }));
    }
    let st = race(&mut c);
    rows("ordering", "symmetric_permutation (P·A·Pᵀ)", &params(name, a), &st, &vec![String::new(); st.len()]);
}

/// submatrix (two ranges) and select (two lists).
fn submatrices(name: &str) {
    note(&format!("submatrix {name}"));
    let a = mats::by_name(name);
    let n = a.nrows;
    let o = a.ours();
    let csc = Csc32::of(&a);
    let (r0, r1) = (n / 4, 3 * n / 4);
    let want = Csr::from_ours(&o.submatrix(r0..r1, r0..r1));
    let mut rs: Vec<i32> = (r0 as i32..r1 as i32).collect();
    let h = unsafe {
        let mut cs = rs.clone();
        ss::ss_submatrix_h(csc.m, csc.n, csc.p.as_ptr(), csc.i.as_ptr(), csc.x.as_ptr(), rs.as_mut_ptr(), rs.len() as i64, cs.as_mut_ptr(), rs.len() as i64)
    };
    let (mm, nn, pp, ii, xx) = ss::cholmod_result(h);
    unsafe { ss::ss_sp_free(h) };
    check("cholmod submatrix", mat_diff(&want, &Csr::from_ours(&ss::csc_to_ours(mm, nn, &pp, &ii, &xx))), 0.0);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o.submatrix(r0..r1, r0..r1))));
    }
    {
        let s = &csc;
        let mut r = rs.clone();
        let mut cc = rs.clone();
        c.push(Contender::new("CHOLMOD submatrix (index lists)", move || unsafe {
            let h = ss::ss_submatrix_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), r.as_mut_ptr(), r.len() as i64, cc.as_mut_ptr(), cc.len() as i64);
            ss::ss_sp_free(h);
        }));
    }
    let st = race(&mut c);
    rows(G, "submatrix (ranges, middle half)", &params(name, &a), &st, &vec![String::new(); st.len()]);
    // select: random lists of half the rows and columns, unsorted, no repeats.
    let mut rows_ = mats::random_perm(n, 12);
    rows_.truncate(n / 2);
    let mut cols_ = mats::random_perm(n, 13);
    cols_.truncate(n / 2);
    let want = Csr::from_ours(&o.select(&rows_, &cols_));
    let r32: Vec<i32> = rows_.iter().map(|&v| v as i32).collect();
    let c32: Vec<i32> = cols_.iter().map(|&v| v as i32).collect();
    let h = unsafe {
        let (mut r, mut cc) = (r32.clone(), c32.clone());
        ss::ss_submatrix_h(csc.m, csc.n, csc.p.as_ptr(), csc.i.as_ptr(), csc.x.as_ptr(), r.as_mut_ptr(), r.len() as i64, cc.as_mut_ptr(), cc.len() as i64)
    };
    let (mm, nn, pp, ii, xx) = ss::cholmod_result(h);
    unsafe { ss::ss_sp_free(h) };
    check("cholmod select", mat_diff(&want, &Csr::from_ours(&ss::csc_to_ours(mm, nn, &pp, &ii, &xx))), 0.0);
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, r, cc) = (&o, &rows_, &cols_);
        c.push(Contender::new("ours", move || keep(o.select(r, cc))));
    }
    {
        let s = &csc;
        let (mut r, mut cc) = (r32.clone(), c32.clone());
        c.push(Contender::new("CHOLMOD submatrix (index lists)", move || unsafe {
            let h = ss::ss_submatrix_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), r.as_mut_ptr(), r.len() as i64, cc.as_mut_ptr(), cc.len() as i64);
            ss::ss_sp_free(h);
        }));
    }
    let st = race(&mut c);
    rows(G, "select (random half of rows and columns)", &params(name, &a), &st, &vec![String::new(); st.len()]);
}

fn triangles(name: &str) {
    note(&format!("triangles {name}"));
    let a = mats::by_name(name);
    let n = a.nrows as i64;
    let o = a.ours();
    let cst = Csc32::of_transpose(&a); // CSC of Aᵀ: its lower band is A's upper triangle
    for (op, upper) in [("upper_triangle", true), ("lower_triangle", false)] {
        let want = if upper { a.upper() } else { a.lower() };
        let (k1, k2) = if upper { (-n, 0) } else { (0, n) };
        let h = unsafe { ss::ss_band_h(cst.m, cst.n, cst.p.as_ptr(), cst.i.as_ptr(), cst.x.as_ptr(), k1, k2) };
        let (mm, nn, pp, ii, xx) = ss::cholmod_result(h);
        unsafe { ss::ss_sp_free(h) };
        let got = Csr::from_ours(&ss::csc_to_ours(mm, nn, &pp, &ii, &xx)).transpose();
        check("cholmod band", mat_diff(&want, &got), 0.0);
        let mut c: Vec<Contender> = Vec::new();
        {
            let o = &o;
            c.push(Contender::new("ours", move || keep(if upper { o.upper_triangle() } else { o.lower_triangle() })));
        }
        {
            let s = &cst;
            c.push(Contender::new("CHOLMOD band", move || unsafe {
                let h = ss::ss_band_h(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), k1, k2);
                ss::ss_sp_free(h);
            }));
        }
        let st = race(&mut c);
        rows(G, op, &params(name, &a), &st, &vec![String::new(); st.len()]);
    }
}

fn norms(name: &str) {
    note(&format!("norms {name}"));
    let a = mats::by_name(name);
    let o = a.ours();
    let cs = Csc32::of(&a);
    let w1 = o.norm(Norm::One);
    let wi = o.norm(Norm::Infinity);
    let c1 = unsafe { ss::ss_norm(cs.m, cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 1) };
    let ci = unsafe { ss::ss_norm(cs.m, cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 0) };
    check("norm 1", ((w1 - c1) / w1).abs(), 1e-14);
    check("norm inf", ((wi - ci) / wi).abs(), 1e-14);
    for (op, nrm, ch) in [("norm(One)", Norm::One, 1), ("norm(Infinity)", Norm::Infinity, 0)] {
        let mut c: Vec<Contender> = Vec::new();
        {
            let o = &o;
            c.push(Contender::new("ours", move || keep(o.norm(nrm))));
        }
        {
            let s = &cs;
            c.push(Contender::new("CHOLMOD norm_sparse", move || keep(unsafe { ss::ss_norm(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), ch) })));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = "ours compensated (Neumaier) sums".into();
        notes[1] = "plain sums".into();
        rows(G, op, &params(name, &a), &st, &notes);
    }
    // Frobenius and Max: against a plain pass over the values (the work any library does).
    for (op, nrm) in [("norm(Frobenius)", Norm::Frobenius), ("norm(Max)", Norm::Max)] {
        let mut c: Vec<Contender> = Vec::new();
        {
            let o = &o;
            c.push(Contender::new("ours", move || keep(o.norm(nrm))));
        }
        {
            let sp = sprs_csr(&a);
            let fro = matches!(nrm, Norm::Frobenius);
            c.push(Contender::new("sprs 0.11 (data().iter() fold)", move || {
                keep(if fro { sp.data().iter().map(|v| v * v).sum::<f64>().sqrt() } else { sp.data().iter().fold(0.0f64, |m, v| m.max(v.abs())) })
            }));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = "ours scaled and compensated (Frobenius)".into();
        notes[1] = "an unscaled, uncompensated fold: less work than ours".into();
        rows(G, op, &params(name, &a), &st, &notes);
    }
}

fn dense_conv(n: usize, per_row: usize) {
    note("dense conversions");
    let a = mats::random_unsym(n, per_row, 21);
    let o = a.ours();
    let d = o.to_matrix();
    let cs = Csc32::of(&a);
    let fa = faer_csr(&a);
    let sp = sprs_csr(&a);
    let na = na_csr(&a);
    let mut out = vec![0.0; n * n];
    unsafe { ss::ss_to_dense(cs.m, cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), out.as_mut_ptr()) };
    check("cholmod to_dense", vec_diff(d.as_slice(), &out), 0.0);
    let pn = format!("randunsym n={n} nnz={}", a.nnz());
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o.to_matrix())));
    }
    {
        let s = &cs;
        let mut out = vec![0.0; n * n];
        c.push(Contender::new("CHOLMOD sparse_to_dense (+copy out)", move || unsafe {
            ss::ss_to_dense(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), out.as_mut_ptr());
        }));
    }
    {
        let fa = &fa;
        c.push(Contender::new("faer 0.24 (to_dense)", move || keep(fa.to_dense())));
    }
    {
        let sp = &sp;
        c.push(Contender::new("sprs 0.11 (to_dense)", move || keep(sp.to_dense())));
    }
    {
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11 (DMatrix::from)", move || keep(nalgebra::DMatrix::from(na))));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[1] = "includes a copy of the dense result out of CHOLMOD's".into();
    rows(G, "to_matrix (sparse → dense)", &pn, &st, &notes);
    let mut c: Vec<Contender> = Vec::new();
    {
        let d = &d;
        c.push(Contender::new("ours", move || keep(SparseMatrix::from_matrix(d))));
    }
    {
        let mut dd = d.as_slice().to_vec();
        c.push(Contender::new("CHOLMOD dense_to_sparse", move || unsafe {
            let h = ss::ss_from_dense_h(n as i32, n as i32, dd.as_mut_ptr());
            ss::ss_sp_free(h);
        }));
    }
    {
        let dn = nalgebra::DMatrix::from_column_slice(n, n, d.as_slice());
        c.push(Contender::new("nalgebra-sparse 0.11 (CsrMatrix::from(&DMatrix))", move || keep(nalgebra_sparse::CsrMatrix::from(&dn))));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "column-major dense input to rows".into();
    notes[1] = "column-major to columns: the natural direction".into();
    rows(G, "from_matrix (dense → sparse)", &pn, &st, &notes);
}

fn symmetry(name: &str) {
    note(&format!("symmetry {name}"));
    let a = mats::by_name(name);
    let o = a.ours();
    let cs = Csc32::of(&a);
    assert!(o.is_symmetric());
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o.is_symmetric())));
    }
    {
        let s = &cs;
        c.push(Contender::new("CHOLMOD symmetry (option 1)", move || keep(unsafe { ss::ss_symmetry(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr()) })));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[1] = "also counts the matched entries and the diagonal".into();
    rows(G, "is_symmetric", &params(name, &a), &st, &notes);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o.is_symmetric_pattern())));
    }
    {
        let s = &cs;
        c.push(Contender::new("CHOLMOD symmetry (option 1)", move || keep(unsafe { ss::ss_symmetry(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr()) })));
    }
    let st = race(&mut c);
    rows(G, "is_symmetric_pattern", &params(name, &a), &st, &notes);
    // diagonal()
    let sp = sprs_csr(&a);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(o.diagonal())));
    }
    {
        let sp = &sp;
        c.push(Contender::new("sprs 0.11 (diag)", move || keep(sp.diag())));
    }
    let st = race(&mut c);
    rows(G, "diagonal", &params(name, &a), &st, &vec![String::new(); st.len()]);
}

/// from_csr: the arrays checked and taken (a clone of them in every contender).
fn from_raw(name: &str) {
    note(&format!("from_csr {name}"));
    let a = mats::by_name(name);
    let (m, n) = (a.nrows, a.ncols);
    let mut c: Vec<Contender> = Vec::new();
    {
        let a = &a;
        c.push(Contender::new("ours", move || keep(SparseMatrix::from_csr(m, n, a.ptr.clone(), a.idx.clone(), a.val.clone()))));
    }
    {
        let a = &a;
        c.push(Contender::new("faer 0.24 (new_checked)", move || {
            keep(faer::sparse::SparseRowMat::new(faer::sparse::SymbolicSparseRowMat::new_checked(m, n, a.ptr.clone(), None, a.idx.clone()), a.val.clone()))
        }));
    }
    {
        let a = &a;
        c.push(Contender::new("sprs 0.11 (CsMat::new)", move || keep(sprs::CsMat::new((m, n), a.ptr.clone(), a.idx.clone(), a.val.clone()))));
    }
    {
        let a = &a;
        c.push(Contender::new("nalgebra-sparse 0.11 (try_from_csr_data)", move || {
            keep(nalgebra_sparse::CsrMatrix::try_from_csr_data(m, n, a.ptr.clone(), a.idx.clone(), a.val.clone()).unwrap())
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "every contender clones the three arrays, then validates; ours also checks for stored zeros".into();
    rows(G, "from_csr (validated arrays)", &params(name, &a), &st, &notes);
}
