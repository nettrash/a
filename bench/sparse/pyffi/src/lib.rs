//! A C ABI over a::math::sparse for py/scipy_ref.py: each `t_*` function runs one operation
//! `reps` times and returns the nanoseconds per operation, measured with `Instant` here.
#![allow(clippy::missing_safety_doc)]

use a::math::matrix::Norm;
use a::math::sparse::{
    ArnoldiWhich, CooMatrix, EigenOptions, Jacobi, LanczosWhich, Preconditioner, SolveOptions, SparseLu, SparseMatrix,
    arnoldi_shift_invert_with, arnoldi_with, bicgstab_with, cg_with, connected_components, gmres_with, lanczos_shift_invert_with,
    lanczos_with, minres_with, read_matrix_market, reverse_cuthill_mckee, write_matrix_market, write_matrix_market_with,
    MatrixMarketOptions, MatrixMarketSymmetry,
};
use std::hint::black_box;
use std::time::Instant;

type M = SparseMatrix<f64>;

unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n) } }
}

fn per(t: Instant, reps: u64) -> f64 {
    t.elapsed().as_nanos() as f64 / reps as f64
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ours_new(nrows: u64, ncols: u64, nnz: u64, ptr: *const u64, idx: *const u64, val: *const f64) -> *mut M {
    let (p, i, v) = unsafe { (sl(ptr, nrows as usize + 1), sl(idx, nnz as usize), sl(val, nnz as usize)) };
    let m = SparseMatrix::from_csr(
        nrows as usize,
        ncols as usize,
        p.iter().map(|&x| x as usize).collect(),
        i.iter().map(|&x| x as usize).collect(),
        v.to_vec(),
    );
    Box::into_raw(Box::new(m))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ours_free(h: *mut M) {
    drop(unsafe { Box::from_raw(h) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ours_nnz(h: *const M) -> u64 {
    unsafe { (*h).nnz() as u64 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_spmv(h: *const M, x: *const f64, y: *mut f64, transpose: i32, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let (xl, yl) = if transpose != 0 { (a.rows(), a.cols()) } else { (a.cols(), a.rows()) };
    let x = unsafe { sl(x, xl) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, yl) };
    let t = Instant::now();
    for _ in 0..reps {
        if transpose != 0 { a.mul_vec_transpose_into(x, y) } else { a.mul_vec_into(x, y) }
        black_box(&*y);
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_spgemm(h: *const M, g: *const M, reps: u64, out_nnz: *mut u64) -> f64 {
    let (a, b) = unsafe { (&*h, &*g) };
    let t = Instant::now();
    let mut nnz = 0;
    for _ in 0..reps {
        let c = a * b;
        nnz = c.nnz();
        black_box(c);
    }
    let r = per(t, reps);
    unsafe { *out_nnz = nnz as u64 };
    r
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_add(h: *const M, g: *const M, reps: u64) -> f64 {
    let (a, b) = unsafe { (&*h, &*g) };
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a + b);
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_transpose(h: *const M, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a.transpose());
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_coo(n: u64, nnz: u64, ri: *const u64, ci: *const u64, v: *const f64, reps: u64) -> f64 {
    let (r, c, x) = unsafe { (sl(ri, nnz as usize), sl(ci, nnz as usize), sl(v, nnz as usize)) };
    let coo = CooMatrix::from_triplets(n as usize, n as usize, r.iter().map(|&x| x as usize).collect(), c.iter().map(|&x| x as usize).collect(), x.to_vec());
    let t = Instant::now();
    for _ in 0..reps {
        black_box(coo.to_csr());
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_permute(h: *const M, p: *const u64, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let p: Vec<usize> = unsafe { sl(p, a.rows()) }.iter().map(|&x| x as usize).collect();
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a.permute(&p, &p));
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_submatrix(h: *const M, r0: u64, r1: u64, c0: u64, c1: u64, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a.submatrix(r0 as usize..r1 as usize, c0 as usize..c1 as usize));
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_select(h: *const M, rows: *const u64, nr: u64, cols: *const u64, nc: u64, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let r: Vec<usize> = unsafe { sl(rows, nr as usize) }.iter().map(|&x| x as usize).collect();
    let c: Vec<usize> = unsafe { sl(cols, nc as usize) }.iter().map(|&x| x as usize).collect();
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a.select(&r, &c));
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_triangle(h: *const M, upper: i32, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    for _ in 0..reps {
        black_box(if upper != 0 { a.upper_triangle() } else { a.lower_triangle() });
    }
    per(t, reps)
}

/// kind 0: one, 1: infinity, 2: Frobenius, 3: max.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_norm(h: *const M, kind: i32, reps: u64, out: *mut f64) -> f64 {
    let a = unsafe { &*h };
    let nrm = match kind {
        0 => Norm::One,
        1 => Norm::Infinity,
        2 => Norm::Frobenius,
        _ => Norm::Max,
    };
    let t = Instant::now();
    let mut v = 0.0;
    for _ in 0..reps {
        v = black_box(a.norm(nrm));
    }
    let r = per(t, reps);
    unsafe { *out = v };
    r
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_diagonal(h: *const M, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a.diagonal());
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_scale(h: *const M, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    for _ in 0..reps {
        black_box(a * 1.5);
    }
    per(t, reps)
}

/// Reads a Matrix Market file (whole file read into memory inside the timing, as scipy does
/// from a path); csr != 0: into compressed rows.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_mmread(path: *const u8, len: u64, csr: i32, reps: u64) -> f64 {
    let p = std::str::from_utf8(unsafe { sl(path, len as usize) }).unwrap().to_string();
    let t = Instant::now();
    for _ in 0..reps {
        let f = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&p).unwrap());
        if csr != 0 {
            black_box(SparseMatrix::<f64>::read_matrix_market(f).unwrap());
        } else {
            black_box(read_matrix_market::<f64>(f).unwrap());
        }
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_mmwrite(h: *const M, path: *const u8, len: u64, sym: i32, reps: u64) -> f64 {
    let a = unsafe { &*h };
    let p = std::str::from_utf8(unsafe { sl(path, len as usize) }).unwrap().to_string();
    let opts = MatrixMarketOptions::new().with_symmetry(MatrixMarketSymmetry::Symmetric);
    let t = Instant::now();
    for _ in 0..reps {
        let f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&p).unwrap());
        if sym != 0 {
            write_matrix_market_with(a, f, &opts).unwrap();
        } else {
            write_matrix_market(a, f).unwrap();
        }
    }
    per(t, reps)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_rcm(h: *const M, reps: u64, out: *mut u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    let mut p = Vec::new();
    for _ in 0..reps {
        p = black_box(reverse_cuthill_mckee(a));
    }
    let r = per(t, reps);
    for (k, &v) in p.iter().enumerate() {
        unsafe { *out.add(k) = v as u64 };
    }
    r
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_components(h: *const M, reps: u64, out_count: *mut u64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    let mut c = Vec::new();
    for _ in 0..reps {
        c = black_box(connected_components(a));
    }
    let r = per(t, reps);
    unsafe { *out_count = c.iter().max().map_or(0, |m| m + 1) as u64 };
    r
}

/// SparseLu::new (COLAMD, partial pivoting); out[0] = nnz(L+U), out[1] = 1 if singular.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_lu(h: *const M, reps: u64, out: *mut f64) -> f64 {
    let a = unsafe { &*h };
    let t = Instant::now();
    let mut nnz = 0;
    let mut sing = false;
    for _ in 0..reps {
        let f = SparseLu::new(a);
        nnz = f.nnz();
        sing = f.is_singular();
        black_box(f);
    }
    let r = per(t, reps);
    unsafe {
        *out = nnz as f64;
        *out.add(1) = sing as i32 as f64;
    }
    r
}

/// A factor kept for the solve rows.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ours_lu_new(h: *const M) -> *mut SparseLu<f64> {
    Box::into_raw(Box::new(SparseLu::new(unsafe { &*h })))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ours_lu_free(f: *mut SparseLu<f64>) {
    drop(unsafe { Box::from_raw(f) });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_lu_solve(f: *const SparseLu<f64>, n: u64, b: *const f64, x: *mut f64, reps: u64) -> f64 {
    let f = unsafe { &*f };
    let b = unsafe { sl(b, n as usize) };
    let t = Instant::now();
    let mut s = Vec::new();
    for _ in 0..reps {
        s = black_box(f.solve_vec(b).unwrap());
    }
    let r = per(t, reps);
    unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), x, n as usize) };
    r
}

/// Krylov solve. method 0 cg, 1 minres, 2 gmres(restart), 3 bicgstab; precond 0 none, 1 Jacobi.
/// out: [iterations, matvecs, converged, ‖r‖ reported], x the solution.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_krylov(h: *const M, method: i32, precond: i32, rtol: f64, maxit: u64, restart: u64, b: *const f64, x: *mut f64, reps: u64, out: *mut f64) -> f64 {
    let a = unsafe { &*h };
    let n = a.rows();
    let b = unsafe { sl(b, n) };
    let jac = Jacobi::new(a).unwrap();
    let pre: Option<&dyn Preconditioner<f64>> = if precond == 1 { Some(&jac) } else { None };
    let opts = SolveOptions::new().with_rtol(rtol).with_max_iter(maxit as usize).with_restart(restart as usize);
    let run = || match method {
        0 => cg_with(a, b, &opts, pre),
        1 => minres_with(a, b, &opts, pre),
        2 => gmres_with(a, b, &opts, pre),
        _ => bicgstab_with(a, b, &opts, pre),
    }
    .unwrap();
    let t = Instant::now();
    let mut res = None;
    for _ in 0..reps {
        res = Some(black_box(run()));
    }
    let r = per(t, reps);
    let s = res.unwrap();
    unsafe {
        *out = s.iterations as f64;
        *out.add(1) = s.matvecs as f64;
        *out.add(2) = s.converged as i32 as f64;
        *out.add(3) = s.residual;
        std::ptr::copy_nonoverlapping(s.x.as_ptr(), x, n);
    }
    r
}

/// Eigenvalues. method 0 lanczos, 1 arnoldi, 2 lanczos_shift_invert, 3 arnoldi_shift_invert.
/// which (lanczos): 0 LA, 1 SA, 2 LM; (arnoldi): 0 LM, 1 LR, 2 SR. vals: k reals (lanczos) or
/// 2k (re, im) (arnoldi). out: [converged, restarts, applications].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn t_eigen(h: *const M, method: i32, which: i32, k: u64, basis: u64, tol: f64, sigma: f64, max_restarts: u64, v0: *const f64, reps: u64, vals: *mut f64, out: *mut f64) -> f64 {
    let a = unsafe { &*h };
    let n = a.rows();
    let start = unsafe { sl(v0, n) }.to_vec();
    let opts = EigenOptions::new().with_basis(basis as usize).with_tolerance(tol).with_max_restarts(max_restarts as usize).with_start(start);
    let k = k as usize;
    let lw = match which {
        0 => LanczosWhich::LargestAlgebraic,
        1 => LanczosWhich::SmallestAlgebraic,
        _ => LanczosWhich::LargestMagnitude,
    };
    let aw = match which {
        0 => ArnoldiWhich::LargestMagnitude,
        1 => ArnoldiWhich::LargestReal,
        _ => ArnoldiWhich::SmallestReal,
    };
    let t = Instant::now();
    let mut stats = (0usize, 0usize, 0usize);
    let mut values: Vec<f64> = Vec::new();
    for _ in 0..reps {
        match method {
            0 | 2 => {
                let Some(r) = (if method == 0 { lanczos_with(a, k, lw, &opts) } else { lanczos_shift_invert_with(a, sigma, k, &opts) }) else {
                    unsafe { *out = -1.0 };
                    return -1.0;
                };
                stats = (r.converged, r.restarts, r.applications);
                values = r.values.clone();
                black_box(r);
            }
            _ => {
                let Some(r) = (if method == 1 { arnoldi_with(a, k, aw, &opts) } else { arnoldi_shift_invert_with(a, sigma, k, &opts) }) else {
                    unsafe { *out = -1.0 };
                    return -1.0;
                };
                stats = (r.converged, r.restarts, r.applications);
                values = r.values.iter().flat_map(|z| [z.re, z.im]).collect();
                black_box(r);
            }
        }
    }
    let r = per(t, reps);
    unsafe {
        for (j, v) in values.iter().enumerate() {
            *vals.add(j) = *v;
        }
        *out = stats.0 as f64;
        *out.add(1) = stats.1 as f64;
        *out.add(2) = stats.2 as f64;
    }
    r
}
