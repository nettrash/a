//! Sparse QR for least squares: factorisation and solve.

use crate::mats::{self, Csr};
use crate::ss::{self, Csc32};
use crate::timing::{Contender, keep, note, race, rows, selected};
use a::math::sparse::{SparseMatrix, SparseQr};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SparseColMat;
use faer::sparse::linalg::qr::{QrRef, QrSymbolicParams, SymbolicQr, factorize_symbolic_qr};
use faer::{Conj, MatMut, Par};

const G: &str = "qr";

pub struct FaerQr {
    sym: SymbolicQr<usize>,
    idx: Vec<usize>,
    val: Vec<f64>,
}

pub fn faer_qr(a: &SparseColMat<usize, f64>) -> FaerQr {
    let sym = factorize_symbolic_qr(a.symbolic(), QrSymbolicParams::default()).unwrap();
    let mut idx = vec![0usize; sym.len_idx()];
    let mut val = vec![0.0f64; sym.len_val()];
    let req = sym.factorize_numeric_qr_scratch::<f64>(Par::Seq, Default::default());
    let mut buf = MemBuffer::new(req);
    sym.factorize_numeric_qr(&mut idx, &mut val, a.as_ref(), Par::Seq, MemStack::new(&mut buf), Default::default());
    FaerQr { sym, idx, val }
}

impl FaerQr {
    /// The least-squares solution: rhs of m rows in, the first n rows out.
    pub fn solve(&self, rhs: &mut [f64]) {
        let m = rhs.len();
        let req = self.sym.solve_in_place_scratch::<f64>(1, Par::Seq);
        let mut buf = MemBuffer::new(req);
        let qr = unsafe { QrRef::new_unchecked(&self.sym, &self.idx, &self.val) };
        qr.solve_in_place_with_conj(Conj::No, MatMut::from_column_major_slice_mut(rhs, m, 1), Par::Seq, MemStack::new(&mut buf));
    }
}

/// ‖Aᴴ·(A·x − b)‖ / (‖A‖²·‖x‖ + ‖A‖·‖b‖): the optimality of a least-squares solution.
fn ls_err(a: &SparseMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let r: Vec<f64> = a.mul_vec(x).iter().zip(b).map(|(p, q)| p - q).collect();
    let g = a.mul_vec_transpose(&r);
    let na = a.norm(a::math::matrix::Norm::Frobenius);
    let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    g.iter().map(|v| v * v).sum::<f64>().sqrt() / (na * na * nx + na * nb)
}

pub fn run(quick: bool) {
    let set: Vec<&str> = if quick {
        vec!["curve-5000", "gridls-31", "randtall-1000"]
    } else {
        vec![
            "curve-5000", "curve-50000", "tallband-500", "tallband-5000", "gridls-31", "gridls-100", "gridls-183", "randtall-1000",
            "randtall-3000", "lap2d-100",
        ]
    };
    for name in set.iter().filter(|n| selected(n)) {
        qr(name, &mats::by_name(name));
    }
}

fn qr(name: &str, a: &Csr) {
    note(&format!("qr {name}"));
    let (m, n) = (a.nrows, a.ncols);
    let o = a.ours();
    let b = mats::rand_vec(m, 61);
    let cs = Csc32::of(a);
    let (p64, i64_) = (cs.p64(), cs.i64());
    let fc = crate::conv::faer_csc(a);
    let pp = format!("{name} {m}×{n} nnz={}", a.nnz());
    let f = SparseQr::new(&o);
    let x = f.solve_vec(&b).expect("ours refused");
    let e_ours = ls_err(&o, &x, &b);
    assert!(e_ours < 1e-12, "ours ls error {e_ours:e}");
    let nnz_ours = f.nnz();
    let nnz_r = f.r().nnz();
    // SPQR.
    let h = unsafe { ss::ss_spqr(m as i64, n as i64, p64.as_ptr(), i64_.as_ptr(), cs.x.as_ptr(), 2, -2.0) };
    assert!(!h.is_null());
    let (nr, nh) = unsafe { (ss::ss_spqr_nnzR(), ss::ss_spqr_nnzH()) };
    let mut bb = b.clone();
    let mut xs = vec![0.0; n];
    unsafe { ss::ss_spqr_solve(h, m as i64, n as i64, 1, bb.as_mut_ptr(), xs.as_mut_ptr()) };
    assert!(ls_err(&o, &xs, &b) < 1e-12, "spqr ls error");
    unsafe { ss::ss_spqr_free(h) };
    // faer's (simplicial) QR of the curve and the tall band takes time quadratic in m (7 s for the
    // curve at m = 10⁴, minutes for the band at m = 10⁵): left out above
    // that.
    let use_faer = !((name.starts_with("curve") || name.starts_with("tallband")) && m > 20_000);
    let fq = if use_faer { faer_qr(&fc) } else { faer_qr(&crate::conv::faer_csc(&mats::curve(10))) };
    if use_faer {
        let mut xf = b.clone();
        fq.solve(&mut xf);
        assert!(ls_err(&o, &xf[..n], &b) < 1e-12, "faer ls error {:e}", ls_err(&o, &xf[..n], &b));
    }
    let hc = unsafe { ss::ss_cs_qr(cs.m, cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 3) };
    let m2 = unsafe { ss::ss_cs_qr_m2(hc) } as usize;
    let mut w = vec![0.0; m2.max(m)];
    unsafe { ss::ss_cs_qr_solve(hc, b.as_ptr(), xs.as_mut_ptr(), w.as_mut_ptr()) };
    assert!(ls_err(&o, &xs, &b) < 1e-12, "cs_qr ls error");
    let nnz_cs = unsafe { ss::ss_cs_qr_nnz(hc) };
    unsafe { ss::ss_cs_qr_free(hc) };
    #[cfg(have_accelerate)]
    let mut acin = (cs.plong(), cs.i.clone(), cs.x.clone());
    #[cfg(have_accelerate)]
    {
        let ha = unsafe { crate::accel::ac_factor_default(crate::accel::QR, cs.m, cs.n, acin.0.as_mut_ptr(), acin.1.as_mut_ptr(), acin.2.as_mut_ptr(), 0) };
        assert!(!ha.is_null(), "accelerate qr");
        let mut bb = b.clone();
        let mut xa = vec![0.0; n];
        unsafe { crate::accel::ac_solve(ha, 1, bb.as_mut_ptr(), xa.as_mut_ptr()) };
        assert!(ls_err(&o, &xa, &b) < 1e-12, "accelerate ls error {:e}", ls_err(&o, &xa, &b));
        unsafe { crate::accel::ac_free(ha) };
    }
    let fills = format!("ours nnz(R) {nnz_r}, R+H {nnz_ours}; SPQR R {nr:.0}, H {nh:.0}; CSparse V+R {nnz_cs}");
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(SparseQr::new(o))));
    }
    {
        let (p, i, x) = (&p64, &i64_, &cs.x);
        c.push(Contender::new("SPQR 4.3 (COLAMD)", move || unsafe { ss::ss_spqr_free(ss::ss_spqr(m as i64, n as i64, p.as_ptr(), i.as_ptr(), x.as_ptr(), 2, -2.0)) }));
    }
    {
        let (p, i, x) = (&p64, &i64_, &cs.x);
        c.push(Contender::new("SPQR 4.3 (default ordering)", move || unsafe { ss::ss_spqr_free(ss::ss_spqr(m as i64, n as i64, p.as_ptr(), i.as_ptr(), x.as_ptr(), 7, -2.0)) }));
    }
    if use_faer {
        let fc = &fc;
        c.push(Contender::new("faer 0.24 Qr (COLAMD, auto)", move || keep(faer_qr(fc))));
    }
    if m <= 120_000 && !name.starts_with("gridls-183") {
        let s = &cs;
        c.push(Contender::new("CSparse cs_qr (AMD of AᵀA)", move || unsafe { ss::ss_cs_qr_free(ss::ss_cs_qr(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 3)) }));
    }
    #[cfg(have_accelerate)]
    {
        let (mut p, mut i, mut x) = (cs.plong(), cs.i.clone(), cs.x.clone());
        let (m_, n_) = (cs.m, cs.n);
        c.push(Contender::new("Accelerate SparseFactor QR", move || unsafe {
            crate::accel::ac_free(crate::accel::ac_factor_default(crate::accel::QR, m_, n_, p.as_mut_ptr(), i.as_mut_ptr(), x.as_mut_ptr(), 0))
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = format!("ours includes a condition estimate (6 solves with R); {fills}");
    notes[1] = "int64; rank detection by Heath's tolerance; BLAS-3 fronts, OpenBLAS 1 thread".into();
    if st.last().unwrap().name.starts_with("Accelerate") {
        *notes.last_mut().unwrap() = "single-threaded BLAS setting; may use AMX".into();
    }
    rows(G, "SparseQr::new", &pp, &st, &notes);
    // Least-squares solve.
    let h = unsafe { ss::ss_spqr(m as i64, n as i64, p64.as_ptr(), i64_.as_ptr(), cs.x.as_ptr(), 2, -2.0) };
    let mut c: Vec<Contender> = Vec::new();
    {
        let (f, b) = (&f, &b);
        c.push(Contender::new("ours", move || keep(f.solve_vec(b))));
    }
    {
        let mut bb = b.clone();
        let mut x = vec![0.0; n];
        c.push(Contender::new("SPQR 4.3 (qmult Qᵀb + solve R)", move || unsafe {
            ss::ss_spqr_solve(h, m as i64, n as i64, 1, bb.as_mut_ptr(), x.as_mut_ptr());
        }));
    }
    if use_faer {
        let fq = &fq;
        let b0 = b.clone();
        let mut x = b.clone();
        c.push(Contender::new("faer 0.24 solve_in_place", move || {
            x.copy_from_slice(&b0);
            fq.solve(&mut x);
            keep(&x);
        }));
    }
    #[cfg(have_accelerate)]
    {
        let ha = unsafe { crate::accel::ac_factor_default(crate::accel::QR, cs.m, cs.n, acin.0.as_mut_ptr(), acin.1.as_mut_ptr(), acin.2.as_mut_ptr(), 0) };
        let mut bb = b.clone();
        let mut x = vec![0.0; n];
        c.push(Contender::new("Accelerate SparseSolve (QR)", move || unsafe {
            crate::accel::ac_solve(ha, 1, bb.as_mut_ptr(), x.as_mut_ptr());
        }));
    }
    let st = race(&mut c);
    unsafe { ss::ss_spqr_free(h) };
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "least squares min ‖A·x − b‖".into();
    rows(G, "SparseQr::solve_vec", &pp, &st, &notes);
}
