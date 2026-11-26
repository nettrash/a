//! Sparse LU with partial pivoting: factorisation and solves.

use crate::g_chol::resid;
use crate::mats::{self, Csr};
use crate::ss::{self, Csc32};
use crate::timing::{Contender, keep, note, race, rows, selected};
use a::math::sparse::{Ordering, SparseLu, SparseLuOptions, amd};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SparseColMat;
use faer::sparse::linalg::lu::{LuRef, LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu};
use faer::{Conj, MatMut, Par};
#[allow(unused_imports)]
use std::ffi::c_long;

const G: &str = "lu";

pub struct FaerLu {
    sym: SymbolicLu<usize>,
    num: NumericLu<usize, f64>,
}

pub fn faer_lu(a: &SparseColMat<usize, f64>) -> Option<FaerLu> {
    let sym = factorize_symbolic_lu(a.symbolic(), LuSymbolicParams::default()).ok()?;
    let mut num = NumericLu::new();
    let req = sym.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default());
    let mut buf = MemBuffer::new(req);
    sym.factorize_numeric_lu(&mut num, a.as_ref(), Par::Seq, MemStack::new(&mut buf), Default::default()).ok()?;
    Some(FaerLu { sym, num })
}

impl FaerLu {
    pub fn solve(&self, x: &mut [f64], transpose: bool) {
        let n = x.len();
        let req = self.sym.solve_in_place_scratch::<f64>(1, Par::Seq);
        let mut buf = MemBuffer::new(req);
        let lu = LuRef::new_unchecked(&self.sym, &self.num);
        let rhs = MatMut::from_column_major_slice_mut(x, n, 1);
        if transpose {
            lu.solve_transpose_in_place_with_conj(Conj::No, rhs, Par::Seq, MemStack::new(&mut buf));
        } else {
            lu.solve_in_place_with_conj(Conj::No, rhs, Par::Seq, MemStack::new(&mut buf));
        }
    }
}

fn faer_csc_of(a: &Csr) -> SparseColMat<usize, f64> {
    crate::conv::faer_csc(a)
}

pub fn run(quick: bool) {
    let set: Vec<&str> = if quick {
        vec!["convdiff2d-100", "circuit-10000", "tridiag-100000"]
    } else {
        vec![
            "convdiff2d-31", "convdiff2d-100", "convdiff2d-316", "upwind2d-316", "lap2d-316", "convdiff3d-10", "convdiff3d-22",
            "circuit-3000", "circuit-10000", "tridiag-100000", "tridiag-1000000",
        ]
    };
    for name in set.iter().filter(|n| selected(n)) {
        lu(name, &mats::by_name(name));
    }
}

fn params(name: &str, a: &Csr) -> String {
    format!("{name} n={} nnz={}", a.nrows, a.nnz())
}

fn lu(name: &str, a: &Csr) {
    note(&format!("lu {name}"));
    let n = a.nrows;
    let o = a.ours();
    let b = mats::rand_vec(n, 51);
    let cs = Csc32::of(a);
    let fc = faer_csc_of(a);
    let pp = params(name, a);
    let f = SparseLu::new(&o);
    assert!(!f.is_singular(), "ours reports singular");
    let x = f.solve_vec(&b).unwrap();
    let r_ours = resid(&o, &x, &b);
    assert!(r_ours < 1e-10, "ours residual {r_ours:e}");
    // The COLAMD order of the columns, as the given order of the second row (SparseLu::new takes
    // AMD of A + Aᵀ instead for a nearly symmetric pattern since the release speed pass).
    let q_colamd = Ordering::Colamd.columns(&o);
    let q32: Vec<i32> = q_colamd.iter().map(|&v| v as i32).collect();
    let nnz_ours = f.nnz();
    // References: check that each solves.
    let mut fill = Vec::new();
    let hu = unsafe { ss::ss_umf(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), -1, -1, -1.0, -1, std::ptr::null()) };
    assert!(!hu.is_null());
    let mut xu = vec![0.0; n];
    unsafe { ss::ss_umf_solve(hu, 0, b.as_ptr(), xu.as_mut_ptr()) };
    assert!(resid(&o, &xu, &b) < 1e-10, "umfpack residual");
    fill.push(format!("UMFPACK {}", unsafe { ss::ss_umf_nnz(hu) }));
    unsafe { ss::ss_umf_free(hu) };
    let hk = unsafe { ss::ss_klu(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 0, 1, -1.0, -1) };
    assert!(!hk.is_null());
    unsafe { ss::ss_klu_solve(hk, 0, b.as_ptr(), xu.as_mut_ptr()) };
    assert!(resid(&o, &xu, &b) < 1e-10, "klu residual {:e}", resid(&o, &xu, &b));
    fill.push(format!("KLU {}", unsafe { ss::ss_klu_nnz(hk) }));
    unsafe { ss::ss_klu_free(hk) };
    let hc = unsafe { ss::ss_cs_lu(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 2, 1.0) };
    let mut work = vec![0.0; n];
    unsafe { ss::ss_cs_lu_solve(hc, b.as_ptr(), xu.as_mut_ptr(), work.as_mut_ptr()) };
    assert!(resid(&o, &xu, &b) < 1e-10, "cs_lu residual");
    fill.push(format!("CSparse {}", unsafe { ss::ss_cs_lu_nnz(hc) }));
    unsafe { ss::ss_cs_lu_free(hc) };
    let fl = faer_lu(&fc).expect("faer lu");
    let mut xf = b.clone();
    fl.solve(&mut xf, false);
    assert!(resid(&o, &xf, &b) < 1e-10, "faer lu residual");
    #[cfg(have_accelerate)]
    let mut ac_in = (cs.plong(), cs.i.clone(), cs.x.clone());
    #[cfg(have_accelerate)]
    {
        let ha = unsafe { crate::accel::ac_factor_default(80, cs.m, cs.n, ac_in.0.as_mut_ptr(), ac_in.1.as_mut_ptr(), ac_in.2.as_mut_ptr(), 0) };
        assert!(!ha.is_null(), "accelerate lu");
        let mut bb = b.clone();
        unsafe { crate::accel::ac_solve(ha, 1, bb.as_mut_ptr(), xu.as_mut_ptr()) };
        assert!(resid(&o, &xu, &b) < 1e-10, "accelerate lu residual {:e}", resid(&o, &xu, &b));
        fill.push(format!("Accelerate factor bytes {:.0}", unsafe { crate::accel::ac_factor_size(ha) }));
        unsafe { crate::accel::ac_free(ha) };
    }
    let fills = format!("nnz(L+U): ours {nnz_ours}, {}", fill.join(", "));
    // 1. Factorisation, each with its own defaults.
    {
        let mut c: Vec<Contender> = Vec::new();
        {
            let o = &o;
            c.push(Contender::new("ours", move || keep(SparseLu::new(o))));
        }
        {
            let s = &cs;
            c.push(Contender::new("UMFPACK 6.3 (defaults)", move || unsafe {
                ss::ss_umf_free(ss::ss_umf(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), -1, -1, -1.0, -1, std::ptr::null()))
            }));
        }
        {
            let s = &cs;
            c.push(Contender::new("UMFPACK 6.3 (unsymmetric, COLAMD, pivot 1, no scaling)", move || unsafe {
                ss::ss_umf_free(ss::ss_umf(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 1, 1, 1.0, 0, std::ptr::null()))
            }));
        }
        {
            let s = &cs;
            c.push(Contender::new("KLU 2.3 (defaults: BTF, AMD)", move || unsafe { ss::ss_klu_free(ss::ss_klu(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 0, 1, -1.0, -1)) }));
        }
        {
            let s = &cs;
            c.push(Contender::new("KLU 2.3 (COLAMD, pivot 1, no BTF) + condest", move || unsafe {
                let h = ss::ss_klu(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 1, 0, 1.0, 0);
                keep(ss::ss_klu_rcond(h, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr()));
                ss::ss_klu_free(h)
            }));
        }
        {
            let s = &cs;
            c.push(Contender::new("CSparse cs_lu (AMD of AᵀA, pivot 1)", move || unsafe { ss::ss_cs_lu_free(ss::ss_cs_lu(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 2, 1.0)) }));
        }
        {
            let fc = &fc;
            c.push(Contender::new("faer 0.24 Lu (COLAMD, auto)", move || keep(faer_lu(fc).unwrap())));
        }
        #[cfg(have_accelerate)]
        {
            let (mut p, mut i, mut x) = (cs.plong(), cs.i.clone(), cs.x.clone());
            let (m_, n_) = (cs.m, cs.n);
            c.push(Contender::new("Accelerate SparseFactor LU (default)", move || unsafe {
                crate::accel::ac_free(crate::accel::ac_factor_default(80, m_, n_, p.as_mut_ptr(), i.as_mut_ptr(), x.as_mut_ptr(), 0))
            }));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = format!("automatic strategy (AMD of A+Aᵀ and threshold 0.1 for a nearly symmetric pattern, else COLAMD and partial pivoting) + condition estimate; {fills}");
        notes[1] = "threshold 0.1, row scaling, auto strategy; BLAS-3 fronts (OpenBLAS 1 thread)".into();
        notes[3] = "threshold 0.001, row scaling; no BLAS".into();
        notes[4] = "the same work as ours: pivoting rule and condition estimate".into();
        notes[5] = "Gilbert-Peierls as ours, no pruning".into();
        if st.len() > 7 {
            notes[7] = "single-threaded BLAS setting; may use AMX".into();
        }
        rows(G, "SparseLu::new", &pp, &st, &notes);
    }
    // 2. The same column order (ours COLAMD) given to UMFPACK.
    {
        let given = SparseLuOptions::new().with_ordering(Ordering::Given(q_colamd.clone()));
        let mut c: Vec<Contender> = Vec::new();
        {
            let (o, g) = (&o, &given);
            c.push(Contender::new("ours", move || keep(SparseLu::with_options(o, g).unwrap())));
        }
        {
            let (s, q) = (&cs, &q32);
            c.push(Contender::new("UMFPACK 6.3 (given Q, unsymmetric, pivot 1, no scaling)", move || unsafe {
                ss::ss_umf_free(ss::ss_umf(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 2, 1, 1.0, 0, q.as_ptr()))
            }));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = "our COLAMD column order given".into();
        notes[1] = "UMFPACK still refines the order within its fronts".into();
        rows(G, "SparseLu::with_options (same given column order)", &pp, &st, &notes);
    }
    // 2b. The symmetric strategy UMFPACK and KLU choose by default for a nearly symmetric
    // pattern, through our API: AMD of A + Aᵀ as the column order, the pivot threshold 0.1.
    {
        let sym_opts = SparseLuOptions::new().with_ordering(Ordering::Given(amd(&o))).with_pivot_threshold(0.1);
        let fs = SparseLu::with_options(&o, &sym_opts).unwrap();
        let ok = !fs.is_singular() && fs.solve_vec(&b).map_or(false, |x| resid(&o, &x, &b) < 1e-10);
        let mut c: Vec<Contender> = Vec::new();
        {
            let (o, g) = (&o, &sym_opts);
            c.push(Contender::new("ours", move || keep(SparseLu::with_options(o, g).unwrap())));
        }
        {
            let s = &cs;
            c.push(Contender::new("UMFPACK 6.3 (defaults)", move || unsafe {
                ss::ss_umf_free(ss::ss_umf(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), -1, -1, -1.0, -1, std::ptr::null()))
            }));
        }
        {
            let s = &cs;
            c.push(Contender::new("KLU 2.3 (defaults: BTF, AMD)", move || unsafe { ss::ss_klu_free(ss::ss_klu(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), 0, 1, -1.0, -1)) }));
        }
        #[cfg(have_accelerate)]
        {
            let (mut p, mut i, mut x) = (cs.plong(), cs.i.clone(), cs.x.clone());
            let (m_, n_) = (cs.m, cs.n);
            c.push(Contender::new("Accelerate SparseFactor LU (default)", move || unsafe {
                crate::accel::ac_free(crate::accel::ac_factor_default(80, m_, n_, p.as_mut_ptr(), i.as_mut_ptr(), x.as_mut_ptr(), 0))
            }));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = format!("ours with_options(Given(amd(A)), pivot 0.1): nnz(L+U) {}, solves {}; {fills}", fs.nnz(), if ok { "to 1e-10" } else { "NOT to 1e-10" });
        rows(G, "SparseLu::with_options (AMD of A+Aᵀ, threshold 0.1)", &pp, &st, &notes);
    }
    // 3. Solves.
    for (op, tr) in [("SparseLu::solve_vec", false), ("SparseLu::solve_transpose_vec", true)] {
        let hu = unsafe { ss::ss_umf(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), -1, -1, -1.0, -1, std::ptr::null()) };
        let hk = unsafe { ss::ss_klu(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 0, 1, -1.0, -1) };
        let mut c: Vec<Contender> = Vec::new();
        {
            let (f, b) = (&f, &b);
            c.push(Contender::new("ours", move || keep(if tr { f.solve_transpose_vec(b) } else { f.solve_vec(b) })));
        }
        {
            let b = b.clone();
            let mut x = vec![0.0; n];
            c.push(Contender::new("UMFPACK 6.3", move || unsafe {
                ss::ss_umf_solve(hu, tr as i32, b.as_ptr(), x.as_mut_ptr());
            }));
        }
        {
            let b = b.clone();
            let mut x = vec![0.0; n];
            c.push(Contender::new("KLU 2.3", move || unsafe {
                ss::ss_klu_solve(hk, tr as i32, b.as_ptr(), x.as_mut_ptr());
            }));
        }
        {
            let fl = &fl;
            let b0 = b.clone();
            let mut x = b.clone();
            c.push(Contender::new("faer 0.24", move || {
                x.copy_from_slice(&b0);
                fl.solve(&mut x, tr);
                keep(&x);
            }));
        }
        if !tr {
            let hc = unsafe { ss::ss_cs_lu(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), cs.x.as_ptr(), 2, 1.0) };
            let b = b.clone();
            let mut x = vec![0.0; n];
            let mut w = vec![0.0; n];
            c.push(Contender::new("CSparse cs_lu solve", move || unsafe {
                ss::ss_cs_lu_solve(hc, b.as_ptr(), x.as_mut_ptr(), w.as_mut_ptr());
            }));
        }
        #[cfg(have_accelerate)]
        if !tr {
            let ha = unsafe { crate::accel::ac_factor_default(80, cs.m, cs.n, ac_in.0.as_mut_ptr(), ac_in.1.as_mut_ptr(), ac_in.2.as_mut_ptr(), 0) };
            let mut bb = b.clone();
            let mut x = vec![0.0; n];
            c.push(Contender::new("Accelerate SparseSolve (LU)", move || unsafe {
                crate::accel::ac_solve(ha, 1, bb.as_mut_ptr(), x.as_mut_ptr());
            }));
        }
        let st = race(&mut c);
        unsafe {
            ss::ss_umf_free(hu);
            ss::ss_klu_free(hk);
        }
        let mut notes = vec![String::new(); st.len()];
        notes[0] = format!("each with its own factors; {fills}");
        notes[1] = "UMFPACK's solve includes 2 steps of iterative refinement by default".into();
        rows(G, op, &pp, &st, &notes);
    }
}
