//! argmin 0.11 drivers (Vec<f64> backend).

use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::brent::BrentRoot;

struct F1<'a>(&'a dyn Fn(f64) -> f64);
impl CostFunction for F1<'_> {
    type Param = f64;
    type Output = f64;
    fn cost(&self, x: &f64) -> Result<f64, Error> {
        Ok((self.0)(*x))
    }
}

pub fn brent_root(lo: f64, hi: f64, tol: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let solver = BrentRoot::new(lo, hi, tol);
    let r = Executor::new(F1(f), solver)
        .configure(|s| s.max_iters(2000))
        .run()
        .unwrap();
    *r.state().get_best_param().unwrap_or(&f64::NAN)
}

use argmin::core::Gradient;
use argmin::solver::brent::BrentOpt;
use argmin::solver::goldensectionsearch::GoldenSectionSearch;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::{BFGS, LBFGS};
use std::cell::Cell;

pub fn golden(lo: f64, hi: f64, init: f64, tol: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let solver = GoldenSectionSearch::new(lo, hi).unwrap().with_tolerance(tol).unwrap();
    let r = Executor::new(F1(f), solver)
        .configure(|s| s.param(init).max_iters(5000))
        .run()
        .unwrap();
    *r.state().get_best_param().unwrap_or(&f64::NAN)
}

pub fn brent_opt(lo: f64, hi: f64, eps: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let solver = BrentOpt::new(lo, hi).set_tolerance(eps, 1e-15);
    let r = Executor::new(F1(f), solver)
        .configure(|s| s.max_iters(5000))
        .run()
        .unwrap();
    *r.state().get_best_param().unwrap_or(&f64::NAN)
}

/// Several variables: the problems of opt.rs, with evaluation counters.
pub struct Pn<'a> {
    pub prob: i32,
    pub lam: &'a [f64],
    pub fe: Cell<usize>,
    pub ge: Cell<usize>,
}
impl CostFunction for Pn<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Error> {
        self.fe.set(self.fe.get() + 1);
        Ok(crate::opt::mn_f(self.prob, self.lam, x))
    }
}
impl Gradient for Pn<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        self.ge.set(self.ge.get() + 1);
        let mut g = vec![0.0; x.len()];
        crate::opt::mn_g(self.prob, self.lam, x, &mut g);
        Ok(g)
    }
}

fn xerr(x: &[f64], sol: &[f64]) -> f64 {
    x.iter().zip(sol).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
}

/// Returns (x, iterations, f evals, g evals).
pub fn bfgs(prob: i32, lam: &[f64], x0: &[f64], gtol: f64) -> (Vec<f64>, u64, usize, usize) {
    let n = x0.len();
    let p = Pn { prob, lam, fe: Cell::new(0), ge: Cell::new(0) };
    let ls = MoreThuenteLineSearch::new();
    let solver = BFGS::new(ls).with_tolerance_grad(gtol).unwrap().with_tolerance_cost(0.0).unwrap();
    let h0: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let r = Executor::new(p, solver)
        .configure(|s| s.param(x0.to_vec()).inv_hessian(h0).max_iters(10_000))
        .run();
    match r {
        Ok(r) => {
            let it = r.state().get_iter();
            let x = r.state().get_best_param().cloned().unwrap_or_default();
            let p = r.problem().problem.as_ref().unwrap();
            (x, it, p.fe.get(), p.ge.get())
        }
        Err(_) => (vec![f64::NAN; n], 0, 0, 0),
    }
}
pub fn bfgs_note(prob: i32, lam: &[f64], x0: &[f64], gtol: f64, sol: &[f64]) -> String {
    let (x, it, fe, ge) = bfgs(prob, lam, x0, gtol);
    format!("{it} iterations, {fe} f, {ge} g, |x−x*|∞ {:.1e}", xerr(&x, sol))
}

pub fn nm(prob: i32, lam: &[f64], x0: &[f64], step: f64) -> (Vec<f64>, u64, usize) {
    let n = x0.len();
    let p = Pn { prob, lam, fe: Cell::new(0), ge: Cell::new(0) };
    let mut simplex = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        simplex.push(v);
    }
    let solver = NelderMead::new(simplex).with_sd_tolerance(1e-16).unwrap();
    let r = Executor::new(p, solver).configure(|s| s.max_iters(100_000)).run();
    match r {
        Ok(r) => {
            let it = r.state().get_iter();
            let x = r.state().get_best_param().cloned().unwrap_or_default();
            let p = r.problem().problem.as_ref().unwrap();
            (x, it, p.fe.get())
        }
        Err(_) => (vec![f64::NAN; n], 0, 0),
    }
}
pub fn nm_note(prob: i32, lam: &[f64], x0: &[f64], step: f64, sol: &[f64]) -> String {
    let (x, it, fe) = nm(prob, lam, x0, step);
    format!("{it} iterations, {fe} f, |x−x*|∞ {:.1e}", xerr(&x, sol))
}

/// L-BFGS (unbounded) on the problems of lbfgsb (prob as in opt.rs; memory m).
pub fn lbfgs(prob: i32, lam: &[f64], x0: &[f64], m: usize, gtol: f64) -> (Vec<f64>, u64, usize, usize) {
    let n = x0.len();
    let p = Pn { prob, lam, fe: Cell::new(0), ge: Cell::new(0) };
    let ls = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(ls, m).with_tolerance_grad(gtol).unwrap().with_tolerance_cost(0.0).unwrap();
    let r = Executor::new(p, solver).configure(|s| s.param(x0.to_vec()).max_iters(100_000)).run();
    match r {
        Ok(r) => {
            let it = r.state().get_iter();
            let x = r.state().get_best_param().cloned().unwrap_or_default();
            let p = r.problem().problem.as_ref().unwrap();
            (x, it, p.fe.get(), p.ge.get())
        }
        Err(_) => (vec![f64::NAN; n], 0, 0, 0),
    }
}
