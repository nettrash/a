//! Sparse Cholesky and L·D·Lᴴ: factorisation (with each one's own ordering, and with one given
//! ordering for all), solves of one and of sixteen right-hand sides.

use crate::mats::{self, Csr};
use crate::ss::{self, Csc32};
use crate::timing::{Contender, fits, keep, note, race, rows, selected};
use a::math::matrix::Matrix;
use a::math::sparse::{Ordering, SparseCholesky, SparseLdl, SparseMatrix, amd};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LdltRef, LltRef, SymbolicCholesky, SymmetricOrdering, factorize_symbolic_cholesky,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Conj, MatMut, Par, Side};
use std::ffi::c_long;

const G: &str = "cholesky";

/// The relative residual ‖A·x − b‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞).
pub fn resid(a: &SparseMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.mul_vec(x);
    let r = ax.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
    let na = a.norm(a::math::matrix::Norm::Infinity);
    let nx = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let nb = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    r / (na * nx + nb)
}

fn check_resid(label: &str, a: &SparseMatrix<f64>, x: &[f64], b: &[f64]) {
    let r = resid(a, x, b);
    if !(r < 1e-12) {
        panic!("{label}: residual {r:e}");
    }
}

pub struct FaerChol {
    pub sym: SymbolicCholesky<usize>,
    pub val: Vec<f64>,
    pub ldlt: bool,
}

pub fn faer_lower(a: &Csr) -> SparseColMat<usize, f64> {
    // CSC of the lower triangle = the rows of the upper triangle.
    let u = a.upper();
    SparseColMat::new(SymbolicSparseColMat::new_checked(a.nrows, a.ncols, u.ptr, None, u.idx), u.val)
}

/// faer's sparse L·Lᴴ (ldlt = false) or L·D·Lᴴ, symbolic and numeric; `mode` is the supernodal
/// threshold; `perm` a given ordering (forward and inverse arrays).
pub fn faer_factor(al: &SparseColMat<usize, f64>, ldlt: bool, mode: SupernodalThreshold, perm: Option<(&[usize], &[usize])>) -> Option<FaerChol> {
    let ord = match perm {
        None => SymmetricOrdering::Amd,
        Some((f, i)) => SymmetricOrdering::Custom(faer::perm::PermRef::new_checked(f, i, f.len())),
    };
    let sym = factorize_symbolic_cholesky(al.symbolic(), Side::Lower, ord, CholeskySymbolicParams { supernodal_flop_ratio_threshold: mode, ..Default::default() }).ok()?;
    let mut val = vec![0.0f64; sym.len_val()];
    let ok = if ldlt {
        let req = sym.factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default());
        let mut buf = MemBuffer::new(req);
        sym.factorize_numeric_ldlt(&mut val, al.as_ref(), Side::Lower, Default::default(), Par::Seq, MemStack::new(&mut buf), Default::default()).is_ok()
    } else {
        let req = sym.factorize_numeric_llt_scratch::<f64>(Par::Seq, Default::default());
        let mut buf = MemBuffer::new(req);
        sym.factorize_numeric_llt(&mut val, al.as_ref(), Side::Lower, Default::default(), Par::Seq, MemStack::new(&mut buf), Default::default()).is_ok()
    };
    ok.then_some(FaerChol { sym, val, ldlt })
}

/// The number of values faer's factor would hold with the given ordering (forward and inverse
/// arrays), from its symbolic analysis alone: the `nnz()` of `faer_factor` with that ordering.
fn faer_len_val(al: &SparseColMat<usize, f64>, perm: (&[usize], &[usize])) -> usize {
    let ord = SymmetricOrdering::Custom(faer::perm::PermRef::new_checked(perm.0, perm.1, perm.0.len()));
    let params = CholeskySymbolicParams { supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO, ..Default::default() };
    factorize_symbolic_cholesky(al.symbolic(), Side::Lower, ord, params).map_or(usize::MAX, |s| s.len_val())
}

impl FaerChol {
    pub fn solve(&self, x: &mut [f64], n: usize, k: usize) {
        let req = self.sym.solve_in_place_scratch::<f64>(k, Par::Seq);
        let mut buf = MemBuffer::new(req);
        let rhs = MatMut::from_column_major_slice_mut(x, n, k);
        if self.ldlt {
            LdltRef::new(&self.sym, &self.val).solve_in_place_with_conj(Conj::No, rhs, Par::Seq, MemStack::new(&mut buf));
        } else {
            LltRef::new(&self.sym, &self.val).solve_in_place_with_conj(Conj::No, rhs, Par::Seq, MemStack::new(&mut buf));
        }
    }
    pub fn nnz(&self) -> usize {
        // The values stored: for the supernodal form the dense blocks, zeros included.
        self.val.len()
    }
    pub fn is_supernodal(&self) -> bool {
        matches!(self.sym.raw(), faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Supernodal(_))
    }
}

/// The CHOLMOD and Accelerate inputs of a symmetric matrix.
pub struct SymInputs {
    pub up: Csc32,    // CHOLMOD: the upper triangle in compressed columns (stype 1)
    pub low: Csc32,   // Accelerate: the lower triangle in compressed columns
    pub low_cs: Vec<c_long>,
}

impl SymInputs {
    pub fn of(a: &Csr) -> SymInputs {
        let l = a.lower(); // rows of the lower triangle = columns of the upper
        let u = a.upper(); // rows of the upper triangle = columns of the lower
        let up = Csc32::of_rows_as_cols(&l, a.nrows, a.ncols);
        let low = Csc32::of_rows_as_cols(&u, a.nrows, a.ncols);
        let low_cs = low.plong();
        SymInputs { up, low, low_cs }
    }
}

pub fn run(quick: bool) {
    let set: Vec<&str> = if quick {
        vec!["lap2d-31", "lap2d-100", "lap3d-10", "tridiag-100000"]
    } else {
        vec![
            "lap2d-31", "lap2d-100", "lap2d-316", "lap2d-1000", "lap3d-10", "lap3d-22", "lap3d-46", "elast3d-10", "elast3d-16",
            "randspd-10000", "tridiag-1000000", "arrow-100000",
        ]
    };
    for name in set.iter().filter(|n| selected(n)) {
        let a = mats::by_name(name);
        if fits(G, &CHOL_OPS, &params(name, &a), peak_gb(name)) {
            cholesky(name, &a, quick);
        }
    }
    let lset: Vec<&str> = if quick { vec!["saddle-50", "lap2d-100"] } else { vec!["saddle-100", "saddle-283", "lap2d-316", "lap3d-22", "elast3d-10"] };
    for name in lset.iter().filter(|n| selected(n)) {
        let a = mats::by_name(name);
        if fits(G, &["SparseLdl::new", "SparseLdl::solve_vec"], &params(name, &a), peak_gb(name)) {
            ldl(name, &a);
        }
    }
}

const CHOL_OPS: [&str; 4] = [
    "SparseCholesky::new (own ordering)",
    "SparseCholesky::with_ordering (same given order)",
    "SparseCholesky::solve_vec",
    "SparseCholesky::solve (16 columns)",
];

/// The peak memory (GB, rounded up) of the command of one matrix of this group (`BENCH_ONLY`),
/// measured on Linux (aarch64, Ubuntu 26.04: lap2d-1000 6.2 GB, lap3d-46 3.4 GB, tridiag-1000000
/// 2.2 GB, the rest below 0.4 GB); 0 for the small ones. A machine without a quarter more than
/// this skips the matrix (`fits`).
fn peak_gb(name: &str) -> f64 {
    match name {
        "lap2d-1000" => 7.0,
        "lap3d-46" => 4.0,
        "tridiag-1000000" => 3.0,
        _ => 0.0,
    }
}

fn params(name: &str, a: &Csr) -> String {
    format!("{name} n={} nnz={}", a.nrows, a.nnz())
}

fn chol_ss(s: &SymInputs, mode: i32, ll: i32, ordering: i32, perm: Option<&[i32]>) -> ss::H {
    unsafe { ss::ss_chol(s.up.n, s.up.p.as_ptr(), s.up.i.as_ptr(), s.up.x.as_ptr(), mode, ll, ordering, perm.map_or(std::ptr::null(), |p| p.as_ptr())) }
}

#[cfg(have_accelerate)]
fn chol_ac(s: &mut SymInputs, ty: i32, order: i32, perm: Option<&mut [i32]>) -> crate::accel::H {
    let n = s.low.n;
    unsafe {
        crate::accel::ac_factor(ty, n, n, s.low_cs.as_mut_ptr(), s.low.i.as_mut_ptr(), s.low.x.as_mut_ptr(), 1, order, perm.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr()))
    }
}

fn cholesky(name: &str, a: &Csr, _quick: bool) {
    note(&format!("cholesky {name}"));
    let n = a.nrows;
    let o = a.ours();
    let b = mats::rand_vec(n, 31);
    #[allow(unused_mut)]
    let mut si = SymInputs::of(a);
    let al = faer_lower(a);
    let pp = params(name, a);
    // Ours, and the checks.
    let f = SparseCholesky::new(&o).expect("not positive definite");
    check_resid("ours", &o, &f.solve_vec(&b), &b);
    let p = f.permutation().to_vec();
    let lnz_ours = f.nnz();
    let p32: Vec<i32> = p.iter().map(|&v| v as i32).collect();
    let mut pinv = vec![0usize; n];
    for (k, &v) in p.iter().enumerate() {
        pinv[v] = k;
    }
    let mut x = vec![0.0; n];
    let mut bb = b.clone();
    let hs = chol_ss(&si, 2, 1, 2, None);
    assert!(!hs.is_null());
    unsafe { ss::ss_chol_solve(hs, n as i32, 1, bb.as_mut_ptr(), x.as_mut_ptr()) };
    check_resid("cholmod", &o, &x, &b);
    let lnz_ss = unsafe { ss::ss_chol_lnz(hs) };
    let ff = faer_factor(&al, false, SupernodalThreshold::AUTO, None).unwrap();
    let mut xf = b.clone();
    ff.solve(&mut xf, n, 1);
    check_resid("faer", &o, &xf, &b);
    // faer with our ordering: the right convention is the one that gives our fill. Compared by
    // the symbolic analyses alone: the numeric factor of the wrong convention can be far larger
    // than the right one (more than the 16 GB of a GitHub runner for lap2d-1000).
    let (fill_p, fill_pinv) = (faer_len_val(&al, (&p, &pinv)), faer_len_val(&al, (&pinv, &p)));
    note(&format!("faer's factor values with our order as its forward / inverse array: {fill_p} / {fill_pinv}"));
    let faer_fwd_is_p = fill_p <= fill_pinv;
    let (ffw, finv) = if faer_fwd_is_p { (p.clone(), pinv.clone()) } else { (pinv.clone(), p.clone()) };
    #[cfg(have_accelerate)]
    let ac_user_is_p = {
        let ha = chol_ac(&mut si, crate::accel::CHOLESKY, 0, None);
        assert!(!ha.is_null());
        let mut x = vec![0.0; n];
        let mut bb = b.clone();
        unsafe { crate::accel::ac_solve(ha, 1, bb.as_mut_ptr(), x.as_mut_ptr()) };
        check_resid("accelerate", &o, &x, &b);
        unsafe { crate::accel::ac_free(ha) };
        let mut q1 = p32.clone();
        let mut q2: Vec<i32> = pinv.iter().map(|&v| v as i32).collect();
        let h1 = chol_ac(&mut si, crate::accel::CHOLESKY, 1, Some(&mut q1));
        let h2 = chol_ac(&mut si, crate::accel::CHOLESKY, 1, Some(&mut q2));
        let (s1, s2) = unsafe { (crate::accel::ac_factor_size(h1), crate::accel::ac_factor_size(h2)) };
        unsafe {
            crate::accel::ac_free(h1);
            crate::accel::ac_free(h2);
        }
        s1 <= s2
    };
    let notes_fill = format!("nnz(L): ours {lnz_ours}, CHOLMOD AMD {lnz_ss:.0}; ours {} supernodes", f.supernodes());
    unsafe { ss::ss_chol_free(hs) };

    // 1. Each with its own default ordering (AMD for all of them).
    {
        let mut c: Vec<Contender> = Vec::new();
        {
            let o = &o;
            c.push(Contender::new("ours", move || keep(SparseCholesky::new(o).unwrap())));
        }
        {
            let s = &si;
            c.push(Contender::new("CHOLMOD 5.3 supernodal (AMD)", move || unsafe { ss::ss_chol_free(chol_ss(s, 2, 1, 2, None)) }));
        }
        {
            let s = &si;
            c.push(Contender::new("CHOLMOD 5.3 simplicial LLᵀ (AMD)", move || unsafe { ss::ss_chol_free(chol_ss(s, 0, 1, 2, None)) }));
        }
        {
            let s = &si;
            c.push(Contender::new("CHOLMOD 5.3 default (auto, AMD/METIS)", move || unsafe { ss::ss_chol_free(chol_ss(s, 1, 1, -1, None)) }));
        }
        {
            let al = &al;
            c.push(Contender::new("faer 0.24 Llt (AMD, auto)", move || keep(faer_factor(al, false, SupernodalThreshold::AUTO, None).unwrap())));
        }
        if n <= 120_000 && !name.starts_with("lap3d-46") {
            let s = &si;
            c.push(Contender::new("CSparse cs_chol (AMD, up-looking)", move || unsafe { ss::ss_cs_chol_free(ss::ss_cs_chol(s.up.n, s.up.p.as_ptr(), s.up.i.as_ptr(), s.up.x.as_ptr(), 1)) }));
        }
        #[cfg(have_accelerate)]
        {
            let mut s = SymInputs::of(a);
            c.push(Contender::new("Accelerate SparseFactor Cholesky (default order)", move || unsafe { crate::accel::ac_free(chol_ac(&mut s, crate::accel::CHOLESKY, 0, None)) }));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = format!("analysis+factorisation; {notes_fill}");
        notes[1] = "OpenBLAS 0.3.34 single thread; int32".into();
        notes[3] = "may pick METIS".into();
        if let Some(l) = notes.last_mut() {
            if st.last().unwrap().name.starts_with("Accelerate") {
                *l = "single-threaded BLAS setting; may use AMX".into();
            }
        }
        rows(G, "SparseCholesky::new (own ordering)", &pp, &st, &notes);
    }
    // 2. Our AMD ordering given to every one: the numeric work is the same.
    {
        let given = Ordering::Given(p.clone());
        let mut c: Vec<Contender> = Vec::new();
        {
            let (o, g) = (&o, &given);
            c.push(Contender::new("ours", move || keep(SparseCholesky::with_ordering(o, g).unwrap())));
        }
        {
            let (s, q) = (&si, &p32);
            c.push(Contender::new("CHOLMOD 5.3 supernodal (given)", move || unsafe { ss::ss_chol_free(chol_ss(s, 2, 1, 1, Some(q))) }));
        }
        {
            let (s, q) = (&si, &p32);
            c.push(Contender::new("CHOLMOD 5.3 simplicial LLᵀ (given)", move || unsafe { ss::ss_chol_free(chol_ss(s, 0, 1, 1, Some(q))) }));
        }
        {
            let (al, fw, inv) = (&al, &ffw, &finv);
            c.push(Contender::new("faer 0.24 supernodal (given)", move || keep(faer_factor(al, false, SupernodalThreshold::FORCE_SUPERNODAL, Some((fw, inv))).unwrap())));
        }
        {
            let (al, fw, inv) = (&al, &ffw, &finv);
            c.push(Contender::new("faer 0.24 simplicial (given)", move || keep(faer_factor(al, false, SupernodalThreshold::FORCE_SIMPLICIAL, Some((fw, inv))).unwrap())));
        }
        #[cfg(have_accelerate)]
        {
            let mut s = SymInputs::of(a);
            let mut q: Vec<i32> = if ac_user_is_p { p32.clone() } else { pinv.iter().map(|&v| v as i32).collect() };
            c.push(Contender::new("Accelerate SparseFactor Cholesky (given)", move || unsafe { crate::accel::ac_free(chol_ac(&mut s, crate::accel::CHOLESKY, 1, Some(&mut q))) }));
        }
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = format!("our AMD order given to all; analysis+factorisation; {notes_fill}");
        rows(G, "SparseCholesky::with_ordering (same given order)", &pp, &st, &notes);
    }
    // 3. Solves of one and of sixteen right-hand sides.
    for k in [1usize, 16] {
        let bk = mats::rand_vec(n * k, 32);
        let bm = Matrix::from_column_major(n, k, bk.clone());
        let hs = chol_ss(&si, 2, 1, 2, None);
        let ff = faer_factor(&al, false, SupernodalThreshold::AUTO, None).unwrap();
        let mut c: Vec<Contender> = Vec::new();
        if k == 1 {
            let (f, b) = (&f, &b);
            c.push(Contender::new("ours", move || keep(f.solve_vec(b))));
        } else {
            let (f, b) = (&f, &bm);
            c.push(Contender::new("ours", move || keep(f.solve(b))));
        }
        {
            let mut bb = bk.clone();
            let mut x = vec![0.0; n * k];
            c.push(Contender::new("CHOLMOD 5.3 supernodal solve", move || unsafe {
                ss::ss_chol_solve(hs, n as i32, k as i32, bb.as_mut_ptr(), x.as_mut_ptr());
            }));
        }
        {
            let mut bb = bk.clone();
            let mut x = vec![0.0; n * k];
            c.push(Contender::new("CHOLMOD 5.3 supernodal solve2 (reused workspace)", move || unsafe {
                ss::ss_chol_solve2(hs, n as i32, k as i32, bb.as_mut_ptr(), x.as_mut_ptr());
            }));
        }
        {
            let ff = &ff;
            let b0 = bk.clone();
            let mut x = bk.clone();
            c.push(Contender::new("faer 0.24 solve_in_place", move || {
                x.copy_from_slice(&b0);
                ff.solve(&mut x, n, k);
                keep(&x);
            }));
        }
        #[cfg(have_accelerate)]
        {
            let ha = chol_ac(&mut si, crate::accel::CHOLESKY, 0, None);
            let mut bb = bk.clone();
            let mut x = vec![0.0; n * k];
            c.push(Contender::new("Accelerate SparseSolve", move || unsafe {
                crate::accel::ac_solve(ha, k as i32, bb.as_mut_ptr(), x.as_mut_ptr());
            }));
        }
        let st = race(&mut c);
        unsafe { ss::ss_chol_free(hs) };
        let mut notes = vec![String::new(); st.len()];
        notes[0] = format!("each its own AMD factor; {notes_fill}");
        // By name: the Accelerate row exists on macOS only.
        if let Some(i) = st.iter().position(|s| s.name.starts_with("faer")) {
            notes[i] = "includes the copy of b into x".into();
        }
        rows(G, if k == 1 { "SparseCholesky::solve_vec" } else { "SparseCholesky::solve (16 columns)" }, &pp, &st, &notes);
    }
    let _ = amd::<f64>;
}

fn ldl(name: &str, a: &Csr) {
    note(&format!("ldl {name}"));
    let n = a.nrows;
    let o = a.ours();
    let b = mats::rand_vec(n, 41);
    let al = faer_lower(a);
    #[allow(unused_mut)]
    let mut si = SymInputs::of(a);
    let pp = params(name, a);
    let f = SparseLdl::new(&o).expect("no L·D·Lᴴ");
    check_resid("ours ldl", &o, &f.solve_vec(&b), &b);
    let (pos, neg, _) = f.inertia();
    let p32: Vec<i32> = f.permutation().iter().map(|&v| v as i32).collect();
    // CHOLMOD simplicial L·D·Lᵀ.
    let hs = chol_ss(&si, 0, 0, 2, None);
    assert!(!hs.is_null(), "cholmod ldl failed");
    let mut x = vec![0.0; n];
    let mut bb = b.clone();
    unsafe { ss::ss_chol_solve(hs, n as i32, 1, bb.as_mut_ptr(), x.as_mut_ptr()) };
    check_resid("cholmod ldl", &o, &x, &b);
    unsafe { ss::ss_chol_free(hs) };
    // LDL with our ordering: with a permutation it reads the upper triangle of P·A·Pᵀ, so it is
    // given the whole (symmetric) matrix.
    let full = Csc32::of(a);
    let hl = unsafe { ss::ss_ldl(full.n, full.p.as_ptr(), full.i.as_ptr(), full.x.as_ptr(), p32.as_ptr()) };
    assert!(unsafe { ss::ss_ldl_ok(hl) } == 1);
    unsafe { ss::ss_ldl_solve(hl, b.as_ptr(), x.as_mut_ptr()) };
    check_resid("LDL", &o, &x, &b);
    unsafe { ss::ss_ldl_free(hl) };
    let ff = faer_factor(&al, true, SupernodalThreshold::AUTO, None).unwrap();
    let mut xf = b.clone();
    ff.solve(&mut xf, n, 1);
    check_resid("faer ldlt", &o, &xf, &b);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(SparseLdl::new(o).unwrap())));
    }
    {
        let s = &si;
        c.push(Contender::new("CHOLMOD 5.3 simplicial LDLᵀ (AMD)", move || unsafe { ss::ss_chol_free(chol_ss(s, 0, 0, 2, None)) }));
    }
    {
        let (s, q) = (&full, &p32);
        c.push(Contender::new("SuiteSparse LDL 3.3 (our AMD order given)", move || unsafe { ss::ss_ldl_free(ss::ss_ldl(s.n, s.p.as_ptr(), s.i.as_ptr(), s.x.as_ptr(), q.as_ptr())) }));
    }
    {
        let al = &al;
        c.push(Contender::new("faer 0.24 Ldlt (AMD, auto)", move || keep(faer_factor(al, true, SupernodalThreshold::AUTO, None).unwrap())));
    }
    #[cfg(have_accelerate)]
    {
        let mut s = SymInputs::of(a);
        let mut s2 = SymInputs::of(a);
        c.push(Contender::new("Accelerate SparseFactor LDLTUnpivoted", move || unsafe { crate::accel::ac_free(chol_ac(&mut s, crate::accel::LDLT_UNPIVOTED, 0, None)) }));
        c.push(Contender::new("Accelerate SparseFactor LDLT (pivoted, default)", move || unsafe {
            let n = s2.low.n;
            crate::accel::ac_free(crate::accel::ac_factor_default(crate::accel::LDLT, n, n, s2.low_cs.as_mut_ptr(), s2.low.i.as_mut_ptr(), s2.low.x.as_mut_ptr(), 1))
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = format!("inertia ({pos}, {neg}); nnz(L) ours {}", f.nnz());
    notes[1] = "no supernodal LDLᵀ in CHOLMOD".into();
    notes[2] = "up-looking, no supernodes, ordering not timed".into();
    if let Some(i) = st.iter().position(|s| s.name.starts_with("Accelerate SparseFactor LDLT (pivoted")) {
        notes[i] = "Bunch-Kaufman-type pivoting: more work, more robust".into();
    }
    rows(G, "SparseLdl::new", &pp, &st, &notes);
    // Solve.
    let hs = chol_ss(&si, 0, 0, 2, None);
    let mut c: Vec<Contender> = Vec::new();
    {
        let (f, b) = (&f, &b);
        c.push(Contender::new("ours", move || keep(f.solve_vec(b))));
    }
    {
        let mut bb = b.clone();
        let mut x = vec![0.0; n];
        c.push(Contender::new("CHOLMOD 5.3 simplicial LDLᵀ solve2", move || unsafe {
            ss::ss_chol_solve2(hs, n as i32, 1, bb.as_mut_ptr(), x.as_mut_ptr());
        }));
    }
    {
        let ff = &ff;
        let b0 = b.clone();
        let mut x = b.clone();
        c.push(Contender::new("faer 0.24 Ldlt solve_in_place", move || {
            x.copy_from_slice(&b0);
            ff.solve(&mut x, n, 1);
            keep(&x);
        }));
    }
    let st = race(&mut c);
    unsafe { ss::ss_chol_free(hs) };
    rows(G, "SparseLdl::solve_vec", &pp, &st, &vec![String::new(); st.len()]);
}
