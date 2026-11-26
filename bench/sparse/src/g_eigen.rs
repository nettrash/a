//! Lanczos and Arnoldi: a few eigenvalues of a large matrix, against faer's Krylov–Schur.
//! (ARPACK through SciPy is in py/scipy_ref.py.)

use crate::mats::{self, Csr};
use crate::timing::{Contender, keep, note, race, rows, selected};
use a::math::matrix::Norm;
use a::math::sparse::{ArnoldiWhich, EigenOptions, LanczosWhich, SparseMatrix, arnoldi_with, lanczos_with};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{PartialEigenParams, partial_eigen, partial_eigen_scratch, partial_self_adjoint_eigen, partial_self_adjoint_eigen_scratch};
use faer::{Col, Mat, Par, c64};

const G: &str = "eigen";
pub const TOL: f64 = 1e-10;
pub const K: usize = 6;
pub const BASIS: usize = 20;

pub fn run(quick: bool) {
    // Anisotropic grids: the isotropic ones have multiple eigenvalues, of which a method may
    // return one copy or all, so that two methods asked for k need not return the same k.
    // (The largest eigenvalues of grids of 10⁶ unknowns are so clustered that a basis of 20
    // takes thousands of restarts, minutes for every method: the grids stop at 10⁵.)
    let sym: Vec<&str> = if quick { vec!["ani2d-100"] } else { vec!["ani2d-316", "ani3d-22", "randspd-100000"] };
    for name in sym.iter().filter(|n| selected(n)) {
        lanczos_rows(name, &mats::by_name(name));
    }
    // cd2d: mild convection, its eigenvalues in closed form (the strongly convective upwind2d is
    // so far from normal that its computed eigenvalues are uncertain in the third digit).
    let uns: Vec<&str> = if quick { vec!["cd2d-100"] } else { vec!["cd2d-316", "circuit-100000", "randunsym-100000"] };
    for name in uns.iter().filter(|n| selected(n)) {
        arnoldi_rows(name, &mats::by_name(name));
    }
}

fn faer_sym(fa: &faer::sparse::SparseRowMat<usize, f64>, n: usize, tol: f64) -> (Vec<f64>, usize, Mat<f64>) {
    let v0 = Col::<f64>::from_fn(n, |i| 1.0 + ((i * 7919) % 101) as f64 / 101.0);
    let mut vecs = Mat::<f64>::zeros(n, K);
    let mut vals = vec![0.0; K];
    let params = PartialEigenParams { max_dim: BASIS, ..Default::default() };
    let req = partial_self_adjoint_eigen_scratch(&fa.as_ref(), K, Par::Seq, params);
    let mut buf = MemBuffer::new(req);
    let info = partial_self_adjoint_eigen(vecs.as_mut(), &mut vals, &fa.as_ref(), v0.as_ref(), tol, Par::Seq, MemStack::new(&mut buf), params);
    (vals, info.n_converged_eigen, vecs)
}

fn faer_gen(fa: &faer::sparse::SparseRowMat<usize, f64>, n: usize, tol: f64) -> (Vec<c64>, usize, Mat<c64>) {
    let v0 = Col::<f64>::from_fn(n, |i| 1.0 + ((i * 7919) % 101) as f64 / 101.0);
    let mut vecs = Mat::<c64>::zeros(n, K);
    let mut vals = vec![c64::new(0.0, 0.0); K];
    let params = PartialEigenParams { max_dim: BASIS, ..Default::default() };
    let req = partial_eigen_scratch(&fa.as_ref(), K, Par::Seq, params);
    let mut buf = MemBuffer::new(req);
    let info = partial_eigen(vecs.as_mut(), &mut vals, &fa.as_ref(), v0.as_ref(), tol, Par::Seq, MemStack::new(&mut buf), params);
    (vals, info.n_converged_eigen, vecs)
}

/// The eigenvalues of the anisotropic grids, descending (the closed form), or None.
fn exact_ani(name: &str) -> Option<Vec<f64>> {
    let (fam, k) = name.split_once('-')?;
    let k: usize = k.parse().ok()?;
    let c: Vec<f64> = (1..=k).map(|i| (i as f64 * std::f64::consts::PI / (k + 1) as f64).cos()).collect();
    let mut v = Vec::new();
    match fam {
        "ani2d" => {
            let w = 1.37;
            for a in &c {
                for b in &c {
                    v.push((2.0 + 2.0 * w) - 2.0 * a - 2.0 * w * b);
                }
            }
        }
        "cd2d" => {
            let (gx, gy) = (2.0 * (1.0f64 - 0.05 * 0.05).sqrt(), 2.0 * (1.0f64 - 0.03 * 0.03).sqrt());
            for a in &c {
                for b in &c {
                    v.push(4.0 + gx * a + gy * b);
                }
            }
        }
        "ani3d" => {
            let (wx, wy, wz) = (1.71, 1.37, 1.0);
            for a in &c {
                for b in &c {
                    for d in &c {
                        v.push(2.0 * (wx + wy + wz) - 2.0 * wx * a - 2.0 * wy * b - 2.0 * wz * d);
                    }
                }
            }
        }
        _ => return None,
    }
    v.sort_by(|x, y| y.partial_cmp(x).unwrap());
    Some(v)
}

fn sym_resid(a: &SparseMatrix<f64>, lam: f64, v: &[f64]) -> f64 {
    let av = a.mul_vec(v);
    let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    av.iter().zip(v).map(|(p, q)| (p - lam * q).powi(2)).sum::<f64>().sqrt() / nv
}

fn lanczos_rows(name: &str, a: &Csr) {
    note(&format!("lanczos {name}"));
    let n = a.nrows;
    let o = a.ours();
    let na = (o.norm(Norm::One) * o.norm(Norm::Infinity)).sqrt();
    let fa = crate::conv::faer_csr(a);
    let opts = EigenOptions::new().with_basis(BASIS).with_tolerance(TOL).with_max_restarts(1000);
    let r = lanczos_with(&o, K, LanczosWhich::LargestAlgebraic, &opts).unwrap();
    let mut worst = 0.0f64;
    for j in 0..r.converged {
        let v: Vec<f64> = (0..n).map(|i| r.vectors[(i, j)]).collect();
        worst = worst.max(sym_resid(&o, r.values[j], &v) / na);
    }
    let (fv, fconv, fvec) = faer_sym(&fa, n, TOL * na);
    let mut fworst = 0.0f64;
    for j in 0..fconv.min(K) {
        let v: Vec<f64> = (0..n).map(|i| fvec[(i, j)]).collect();
        fworst = fworst.max(sym_resid(&o, fv[j], &v) / na);
    }
    let mut ov = r.values.clone();
    ov.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let mut fs = fv.clone();
    fs.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let agree = ov.iter().zip(&fs).map(|(x, y)| (x - y).abs() / na).fold(0.0, f64::max);
    let exact = exact_ani(name).map(|ex| {
        let d = |v: &[f64]| v.iter().zip(&ex).map(|(x, y)| (x - y).abs() / na).fold(0.0, f64::max);
        let (eo, ef) = (d(&ov), d(&fs));
        format!("; against the closed form: ours {eo:.1e}, faer {ef:.1e}{}", if ef > 1e-8 { " — faer returned a wrong set" } else { "" })
    }).unwrap_or_default();
    let pp = format!("{name} n={n} nnz={} k={K} basis={BASIS} tol={TOL:e} which=LA", a.nnz());
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, opts) = (&o, &opts);
        c.push(Contender::new("ours", move || keep(lanczos_with(o, K, LanczosWhich::LargestAlgebraic, opts))));
    }
    {
        let fa = &fa;
        c.push(Contender::new("faer 0.24 partial_self_adjoint_eigen (Krylov–Schur)", move || keep(faer_sym(fa, n, TOL * na))));
    }
    let st = race(&mut c);
    let notes = vec![
        format!("LargestAlgebraic; ours {} converged, {} restarts, {} products, worst residual/‖A‖ {worst:.1e}", r.converged, r.restarts, r.applications),
        format!("largest magnitude (= largest algebraic here); {fconv} converged, worst residual/‖A‖ {fworst:.1e}; eigenvalues agree to {agree:.1e}·‖A‖{exact}; ours reorthogonalises fully and checks for missed copies"),
    ];
    rows(G, "lanczos_with", &pp, &st, &notes);
}

fn arnoldi_rows(name: &str, a: &Csr) {
    note(&format!("arnoldi {name}"));
    let n = a.nrows;
    let o = a.ours();
    let na = (o.norm(Norm::One) * o.norm(Norm::Infinity)).sqrt();
    let fa = crate::conv::faer_csr(a);
    let opts = EigenOptions::new().with_basis(BASIS).with_tolerance(TOL).with_max_restarts(1000);
    let r = arnoldi_with(&o, K, ArnoldiWhich::LargestMagnitude, &opts).unwrap();
    let (fv, fconv, _) = faer_gen(&fa, n, TOL * na);
    let mut om: Vec<f64> = r.values.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).collect();
    om.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let mut fm: Vec<f64> = fv.iter().map(|z| z.norm()).collect();
    fm.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let agree = om.iter().zip(&fm).map(|(x, y)| (x - y).abs() / na).fold(0.0, f64::max);
    let exact = exact_ani(name).map(|ex| {
        let d = |v: &[f64]| v.iter().zip(&ex).map(|(x, y)| (x - y).abs() / na).fold(0.0, f64::max);
        let (eo, ef) = (d(&om[..K.min(om.len())]), d(&fm));
        format!("; |λ| against the closed form: ours {eo:.1e}, faer {ef:.1e}{}", if ef > 1e-8 { " — faer returned a wrong set" } else { "" })
    }).unwrap_or_default();
    let pp = format!("{name} n={n} nnz={} k={K} basis={BASIS} tol={TOL:e} which=LM", a.nnz());
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, opts) = (&o, &opts);
        c.push(Contender::new("ours", move || keep(arnoldi_with(o, K, ArnoldiWhich::LargestMagnitude, opts))));
    }
    {
        let fa = &fa;
        c.push(Contender::new("faer 0.24 partial_eigen (Krylov–Schur)", move || keep(faer_gen(fa, n, TOL * na))));
    }
    let st = race(&mut c);
    let notes = vec![
        format!("LargestMagnitude; ours {} converged, {} restarts, {} products", r.converged, r.restarts, r.applications),
        format!("{fconv} converged; |λ| agree to {agree:.1e}·‖A‖{exact}"),
    ];
    rows(G, "arnoldi_with", &pp, &st, &notes);
}
