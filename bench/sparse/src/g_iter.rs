//! Krylov methods with preconditioners: time to the same tolerance, iterations in the notes.

use crate::mats::{self, Csr};
use crate::timing::{Contender, keep, note, race, rows, selected};
use a::math::sparse::{
    Ic0, Ilu0, Jacobi, Preconditioner, SolveOptions, SparseMatrix, bicgstab_with, cg_with, gmres_with,
};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::bicgstab::{BicgParams, bicgstab, bicgstab_scratch};
use faer::matrix_free::conjugate_gradient::{CgParams, conjugate_gradient, conjugate_gradient_scratch};
use faer::matrix_free::{IdentityPrecond, InitialGuessStatus};
use faer::{Col, MatMut, MatRef, Par};
#[allow(unused_imports)]
use std::ffi::c_long;

const G: &str = "iterative";
pub const RTOL: f64 = 1e-8;

fn true_rel_resid(a: &SparseMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.mul_vec(x);
    let r: f64 = ax.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
    r / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn shifted(a: &Csr, sigma: f64) -> Csr {
    let mut o = a.ours();
    o = &o - &(&SparseMatrix::identity(a.nrows) * sigma);
    Csr::from_ours(&o)
}

pub fn run(quick: bool) {
    let spd: Vec<&str> = if quick { vec!["lap2d-100"] } else { vec!["lap2d-316", "lap2d-1000", "lap3d-46", "lap3d-100"] };
    for name in spd.iter().filter(|n| selected(n)) {
        cg_rows(name, &mats::by_name(name));
    }
    let uns: Vec<&str> = if quick { vec!["upwind2d-100"] } else { vec!["upwind2d-316", "upwind2d-1000", "convdiff3d-46"] };
    for name in uns.iter().filter(|n| selected(n)) {
        unsym_rows(name, &mats::by_name(name));
    }
    let ind: Vec<&str> = if quick { vec!["lap2d-100"] } else { vec!["lap2d-316", "lap3d-46"] };
    for name in ind.iter().filter(|n| selected(n)) {
        let a = shifted(&mats::by_name(name), 0.5);
        minres_rows(&format!("{name} − 0.5·I"), &a);
    }
}

fn params(name: &str, a: &Csr) -> String {
    format!("{name} n={} nnz={} rtol={RTOL:e}", a.nrows, a.nnz())
}

// faer 0.24 leaves its residual uninitialised with InitialGuessStatus::Zero (its CG and BiCGSTAB
// then return garbage), so both start from a zero guess given as MaybeNonZero: one product more.
fn faer_cg(fa: &faer::sparse::SparseRowMat<usize, f64>, b: &[f64], dinv: Option<&faer::diag::Diag<f64>>) -> (Vec<f64>, usize) {
    let n = b.len();
    let mut x = vec![0.0; n];
    let params = CgParams { initial_guess: InitialGuessStatus::MaybeNonZero, abs_tolerance: 0.0, rel_tolerance: RTOL, max_iters: 100_000, ..Default::default() };
    let info = match dinv {
        Some(d) => {
            let req = conjugate_gradient_scratch(d.as_ref(), fa.as_ref(), 1, Par::Seq);
            let mut buf = MemBuffer::new(req);
            conjugate_gradient(MatMut::from_column_major_slice_mut(&mut x, n, 1), d.as_ref(), fa.as_ref(), MatRef::from_column_major_slice(b, n, 1), params, |_| {}, Par::Seq, MemStack::new(&mut buf))
        }
        None => {
            let id = IdentityPrecond { dim: n };
            let req = conjugate_gradient_scratch(id, fa.as_ref(), 1, Par::Seq);
            let mut buf = MemBuffer::new(req);
            conjugate_gradient(MatMut::from_column_major_slice_mut(&mut x, n, 1), IdentityPrecond { dim: n }, fa.as_ref(), MatRef::from_column_major_slice(b, n, 1), params, |_| {}, Par::Seq, MemStack::new(&mut buf))
        }
    };
    let it = info.map(|i| i.iter_count).unwrap_or(usize::MAX);
    (x, it)
}

fn faer_bicgstab(fa: &faer::sparse::SparseRowMat<usize, f64>, b: &[f64], dinv: Option<&faer::diag::Diag<f64>>) -> (Vec<f64>, usize) {
    let n = b.len();
    let mut x = vec![0.0; n];
    let params = BicgParams { initial_guess: InitialGuessStatus::MaybeNonZero, abs_tolerance: 0.0, rel_tolerance: RTOL, max_iters: 100_000, ..Default::default() };
    let id = || IdentityPrecond { dim: n };
    let info = match dinv {
        Some(d) => {
            let req = bicgstab_scratch(id(), d.as_ref(), fa.as_ref(), 1, Par::Seq);
            let mut buf = MemBuffer::new(req);
            bicgstab(MatMut::from_column_major_slice_mut(&mut x, n, 1), id(), d.as_ref(), fa.as_ref(), MatRef::from_column_major_slice(b, n, 1), params, |_| {}, Par::Seq, MemStack::new(&mut buf))
        }
        None => {
            let req = bicgstab_scratch(id(), id(), fa.as_ref(), 1, Par::Seq);
            let mut buf = MemBuffer::new(req);
            bicgstab(MatMut::from_column_major_slice_mut(&mut x, n, 1), id(), id(), fa.as_ref(), MatRef::from_column_major_slice(b, n, 1), params, |_| {}, Par::Seq, MemStack::new(&mut buf))
        }
    };
    let it = info.map(|i| i.iter_count).unwrap_or(usize::MAX);
    (x, it)
}

fn inv_diag(a: &Csr) -> faer::diag::Diag<f64> {
    let d = a.ours().diagonal();
    Col::from_fn(a.nrows, |i| 1.0 / d[i]).into_diagonal()
}

#[cfg(have_accelerate)]
struct AcIn {
    p: Vec<c_long>,
    i: Vec<i32>,
    x: Vec<f64>,
    sym: i32,
}
#[cfg(have_accelerate)]
impl AcIn {
    /// The whole matrix in compressed columns, or (sym) its lower triangle, as Accelerate's CG
    /// wants a symmetric matrix.
    fn of(a: &Csr, sym: bool) -> AcIn {
        let c = if sym {
            let u = a.upper();
            crate::ss::Csc32::of_rows_as_cols(&u, a.nrows, a.ncols)
        } else {
            crate::ss::Csc32::of(a)
        };
        AcIn { p: c.plong(), i: c.i, x: c.x, sym: sym as i32 }
    }
    /// Returns (status, products) of a run through a counting operator.
    fn count(&mut self, method: i32, nvec: i32, n: usize, b: &[f64], precond: i32) -> (i32, c_long) {
        let mut bb = b.to_vec();
        let mut x = vec![0.0; n];
        let mut prods: c_long = 0;
        let st = unsafe {
            crate::accel::ac_iterative_count(method, 1, nvec, n as i32, self.p.as_mut_ptr(), self.i.as_mut_ptr(), self.x.as_mut_ptr(), self.sym, bb.as_mut_ptr(), x.as_mut_ptr(), RTOL, 100_000, precond, &mut prods)
        };
        (st, prods)
    }
    /// One timed solve; returns (status, x).
    fn solve(&mut self, method: i32, nvec: i32, n: usize, b: &[f64], precond: i32) -> (i32, Vec<f64>) {
        let mut bb = b.to_vec();
        let mut x = vec![0.0; n];
        let st = unsafe {
            crate::accel::ac_iterative(method, 1, nvec, n as i32, self.p.as_mut_ptr(), self.i.as_mut_ptr(), self.x.as_mut_ptr(), self.sym, bb.as_mut_ptr(), x.as_mut_ptr(), RTOL, 100_000, precond)
        };
        (st, x)
    }
}

fn cg_rows(name: &str, a: &Csr) {
    note(&format!("cg {name}"));
    let n = a.nrows;
    let o = a.ours();
    let b = mats::rand_vec(n, 71);
    let fa = crate::conv::faer_csr(a);
    let dinv = inv_diag(a);
    let jac = Jacobi::new(&o).unwrap();
    let ic = Ic0::new(&o).unwrap();
    let opts = SolveOptions::new().with_rtol(RTOL).with_max_iter(100_000);
    let pp = params(name, a);
    for (label, pre) in [("none", None), ("Jacobi", Some(&jac as &dyn Preconditioner<f64>)), ("IC(0)", Some(&ic as &dyn Preconditioner<f64>))] {
        let r = cg_with(&o, &b, &opts, pre).unwrap();
        assert!(r.converged, "ours cg {label} did not converge");
        let tr = true_rel_resid(&o, &r.x, &b);
        let mut c: Vec<Contender> = Vec::new();
        {
            let (o, b, opts) = (&o, &b, &opts);
            c.push(Contender::new("ours", move || keep(cg_with(o, b, opts, pre).unwrap())));
        }
        let mut notes = vec![format!("ours {} iterations, true residual {tr:.1e}", r.iterations)];
        if label != "IC(0)" {
            let d = if label == "Jacobi" { Some(&dinv) } else { None };
            let (xf, itf) = faer_cg(&fa, &b, d);
            let trf = true_rel_resid(&o, &xf, &b);
            assert!(trf < 1e-6, "faer cg {label}: true residual {trf:e}");
            {
                let (fa, b) = (&fa, &b);
                c.push(Contender::new(format!("faer 0.24 conjugate_gradient ({label})"), move || keep(faer_cg(fa, b, d))));
            }
            notes.push(format!("{itf} iterations, true residual {trf:.1e}"));
            #[cfg(have_accelerate)]
            {
                let mut ai = AcIn::of(a, true);
                let pc = if label == "Jacobi" { 2 } else { 0 };
                let (_, prods) = ai.count(0, 0, n, &b, pc);
                let (st, xa) = ai.solve(0, 0, n, &b, pc);
                let tra = true_rel_resid(&o, &xa, &b);
                assert!(st == 0 && tra < 1e-6, "accelerate cg status {st}, residual {tra:e}");
                let bb = b.clone();
                c.push(Contender::new(format!("Accelerate SparseSolve CG ({label})"), move || keep(ai.solve(0, 0, n, &bb, pc))));
                notes.push(format!("{prods} products, true residual {tra:.1e}; lower triangle, symmetric kind; stops at ‖r‖ < rtol·‖r₀‖"));
            }
        }
        if c.len() < 2 {
            continue; // IC(0): no reference with an incomplete Cholesky in process (SciPy has none).
        }
        let st = race(&mut c);
        rows(G, &format!("cg_with ({label})"), &pp, &st, &notes);
    }
    // Preconditioner construction: no reference (none of them offers IC(0)/ILU(0)); time only ours.
}

fn unsym_rows(name: &str, a: &Csr) {
    note(&format!("gmres/bicgstab {name}"));
    let n = a.nrows;
    let o = a.ours();
    let b = mats::rand_vec(n, 72);
    let fa = crate::conv::faer_csr(a);
    let dinv = inv_diag(a);
    let jac = Jacobi::new(&o).unwrap();
    let ilu = Ilu0::new(&o).unwrap();
    let pp = params(name, a);
    let opts = SolveOptions::new().with_rtol(RTOL).with_max_iter(100_000).with_restart(30);
    for (label, pre) in [("none", None), ("Jacobi", Some(&jac as &dyn Preconditioner<f64>)), ("ILU(0)", Some(&ilu as &dyn Preconditioner<f64>))] {
        // BiCGSTAB
        let r = bicgstab_with(&o, &b, &opts, pre).unwrap();
        assert!(r.converged, "ours bicgstab {label}");
        let mut c: Vec<Contender> = Vec::new();
        {
            let (o, b, opts) = (&o, &b, &opts);
            c.push(Contender::new("ours", move || keep(bicgstab_with(o, b, opts, pre).unwrap())));
        }
        let mut notes = vec![format!("ours {} iterations ({} products), true residual {:.1e}", r.iterations, r.matvecs, true_rel_resid(&o, &r.x, &b))];
        if label != "ILU(0)" {
            let d = if label == "Jacobi" { Some(&dinv) } else { None };
            let (xf, itf) = faer_bicgstab(&fa, &b, d);
            let trf = true_rel_resid(&o, &xf, &b);
            assert!(trf < 1e-6, "faer bicgstab {label}: true residual {trf:e}");
            notes.push(format!("{itf} iterations, true residual {trf:.1e}"));
            let (fa, b) = (&fa, &b);
            c.push(Contender::new(format!("faer 0.24 bicgstab ({label}, right)"), move || keep(faer_bicgstab(fa, b, d))));
            let st = race(&mut c);
            rows(G, &format!("bicgstab_with ({label})"), &pp, &st, &notes);
        }
        // GMRES(30)
        let r = gmres_with(&o, &b, &opts, pre).unwrap();
        assert!(r.converged, "ours gmres {label}");
        #[allow(unused_mut)]
        let mut c: Vec<Contender> = Vec::new();
        {
            let (o, b, opts) = (&o, &b, &opts);
            c.push(Contender::new("ours", move || keep(gmres_with(o, b, opts, pre).unwrap())));
        }
        #[allow(unused_mut)]
        let mut notes = vec![format!("restart 30; ours {} iterations ({} products), true residual {:.1e}", r.iterations, r.matvecs, true_rel_resid(&o, &r.x, &b))];
        #[cfg(have_accelerate)]
        if label != "ILU(0)" {
            let mut ai = AcIn::of(a, false);
            let pc = if label == "Jacobi" { 2 } else { 0 };
            let (_, prods) = ai.count(1, 30, n, &b, pc);
            let (st, xa) = ai.solve(1, 30, n, &b, pc);
            let tra = true_rel_resid(&o, &xa, &b);
            assert!(st == 0 && tra < 1e-6, "accelerate gmres status {st}, residual {tra:e}");
            let bb = b.clone();
            c.push(Contender::new(format!("Accelerate SparseSolve GMRES(30) ({label})"), move || keep(ai.solve(1, 30, n, &bb, pc))));
            notes.push(format!("{prods} products, true residual {tra:.1e}"));
        }
        if c.len() >= 2 {
            let st = race(&mut c);
            rows(G, &format!("gmres_with ({label})"), &pp, &st, &notes);
        }
    }
}

fn minres_rows(name: &str, a: &Csr) {
    // In process no reference has MINRES (faer has none); the SciPy rows come from py/scipy_ref.py.
    let _ = (name, a);
}
