//! Optimisation: one variable, several variables, least squares; against GSL, Boost and argmin.

use crate::harness::{Cmp, bb};
use a::math::matrix::Matrix;
use a::math::numerics::{
    OptOptions, bfgs, conjugate_gradient, gauss_newton, golden_section_with, levenberg_marquardt,
    minimize_with, nelder_mead,
};
use std::cell::Cell;

unsafe extern "C" {
    fn probe_gsl_min(kind: i32, prob: i32, p: f64, a: f64, b: f64, guess: f64, epsrel: f64, evals: *mut i64) -> f64;
    fn probe_boost_min(prob: i32, p: f64, a: f64, b: f64, bits: i32, evals: *mut i64) -> f64;
    fn probe_gsl_multimin(
        kind: i32,
        prob: i32,
        n: i32,
        x: *mut f64,
        gtol: f64,
        step: f64,
        maxiter: i32,
        fe: *mut i64,
        ge: *mut i64,
        iters: *mut i32,
    ) -> i32;
    fn probe_gsl_fit(
        trs: i32,
        model: i32,
        m: i32,
        p: i32,
        x: *const f64,
        y: *const f64,
        b: *mut f64,
        xtol: f64,
        gtol: f64,
        maxiter: i32,
        fe: *mut i64,
        je: *mut i64,
        iters: *mut i32,
    ) -> i32;
}

pub fn min1(prob: i32, p: f64, x: f64) -> f64 {
    match prob {
        0 => {
            let d = x - p;
            d * d + 1.0
        }
        1 => x * x.ln() - p * x,
        _ => {
            let d = x - p;
            let d2 = d * d;
            d2 * d2
        }
    }
}
fn min1_exact(prob: i32, p: f64) -> f64 {
    match prob {
        1 => (p - 1.0).exp(),
        _ => p,
    }
}

pub fn mn_f(prob: i32, lam: &[f64], x: &[f64]) -> f64 {
    let n = x.len();
    match prob {
        0 => {
            let mut s = 0.0;
            for i in 0..n - 1 {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                s += 100.0 * a * a + b * b;
            }
            s
        }
        1 => {
            let (a, b, c, d) = (x[0] + 10.0 * x[1], x[2] - x[3], x[1] - 2.0 * x[2], x[0] - x[3]);
            a * a + 5.0 * b * b + c * c * c * c + 10.0 * d * d * d * d
        }
        2 => {
            let (a, b, c, d, e, f) = (x[1] - x[0] * x[0], 1.0 - x[0], x[3] - x[2] * x[2], 1.0 - x[2], x[1] - 1.0, x[3] - 1.0);
            100.0 * a * a + b * b + 90.0 * c * c + d * d + 10.1 * (e * e + f * f) + 19.8 * e * f
        }
        3 => {
            let a = 1.5 - x[0] + x[0] * x[1];
            let b = 2.25 - x[0] + x[0] * x[1] * x[1];
            let c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
            a * a + b * b + c * c
        }
        6 => {
            let mut s = 0.0;
            for i in (0..n).step_by(2) {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                s += 100.0 * a * a + b * b;
            }
            s
        }
        _ => {
            let mut s = 0.0;
            for i in 0..n {
                s += 0.5 * lam[i] * x[i] * x[i];
            }
            s
        }
    }
}
pub fn mn_g(prob: i32, lam: &[f64], x: &[f64], g: &mut [f64]) {
    let n = x.len();
    match prob {
        0 => {
            g.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..n - 1 {
                let a = x[i + 1] - x[i] * x[i];
                g[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
                g[i + 1] += 200.0 * a;
            }
        }
        1 => {
            let (a, b, c, d) = (x[0] + 10.0 * x[1], x[2] - x[3], x[1] - 2.0 * x[2], x[0] - x[3]);
            g[0] = 2.0 * a + 40.0 * d * d * d;
            g[1] = 20.0 * a + 4.0 * c * c * c;
            g[2] = 10.0 * b - 8.0 * c * c * c;
            g[3] = -10.0 * b - 40.0 * d * d * d;
        }
        2 => {
            let (a, c, e, f) = (x[1] - x[0] * x[0], x[3] - x[2] * x[2], x[1] - 1.0, x[3] - 1.0);
            g[0] = -400.0 * x[0] * a - 2.0 * (1.0 - x[0]);
            g[1] = 200.0 * a + 20.2 * e + 19.8 * f;
            g[2] = -360.0 * x[2] * c - 2.0 * (1.0 - x[2]);
            g[3] = 180.0 * c + 20.2 * f + 19.8 * e;
        }
        3 => {
            let y = x[1];
            let a = 1.5 - x[0] + x[0] * y;
            let b = 2.25 - x[0] + x[0] * y * y;
            let c = 2.625 - x[0] + x[0] * y * y * y;
            g[0] = 2.0 * a * (y - 1.0) + 2.0 * b * (y * y - 1.0) + 2.0 * c * (y * y * y - 1.0);
            g[1] = 2.0 * a * x[0] + 4.0 * b * x[0] * y + 6.0 * c * x[0] * y * y;
        }
        6 => {
            for i in (0..n).step_by(2) {
                let a = x[i + 1] - x[i] * x[i];
                g[i] = -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
                g[i + 1] = 200.0 * a;
            }
        }
        _ => {
            for i in 0..n {
                g[i] = lam[i] * x[i];
            }
        }
    }
}

fn mn_start(prob: i32, n: usize) -> Vec<f64> {
    match prob {
        0 => (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect(),
        1 => vec![3.0, -1.0, 0.0, 1.0],
        2 => vec![-3.0, -1.0, -3.0, -1.0],
        3 => vec![1.0, 1.0],
        _ => vec![1.0; n],
    }
}
fn mn_solution(prob: i32, n: usize) -> Vec<f64> {
    match prob {
        0 => vec![1.0; n],
        1 => vec![0.0; 4],
        2 => vec![1.0; 4],
        3 => vec![3.0, 0.5],
        _ => vec![0.0; n],
    }
}
fn lambdas(n: usize) -> Vec<f64> {
    (0..n).map(|i| 10f64.powf(4.0 * i as f64 / (n - 1).max(1) as f64)).collect()
}

fn col(v: Vec<f64>) -> Matrix<f64> {
    let n = v.len();
    Matrix::from_column_major(n, 1, v)
}

pub fn run() {
    one_dim();
    several();
    fits();
}

fn one_dim() {
    const N: usize = 64;
    let opts = OptOptions::<f64>::new(); // rtol 4√ε
    let rtol = opts.rtol;
    for (prob, name, a, b, plo, phi) in [
        (0, "(x−p)²+1 on [−2,4], p∈[0,2]", -2.0, 4.0, 0.0, 2.0),
        (1, "x·ln x − p·x on [0.1,3], p∈[0.5,1.5]", 0.1, 3.0, 0.5, 1.5),
        (2, "(x−p)⁴ on [−2,4], p∈[0,2] (flat)", -2.0, 4.0, 0.0, 2.0),
    ] {
        let ps: Vec<f64> = (0..N).map(|i| plo + (phi - plo) * ((i * 37 % N) as f64 + 0.5) / N as f64).collect();
        let guess = 1.0;
        let stat = |mut run: Box<dyn FnMut(f64) -> (f64, f64) + '_>| -> String {
            let (mut ev, mut err) = (0.0, 0.0f64);
            for &p in &ps {
                let (x, e) = run(p);
                ev += e;
                let r = min1_exact(prob, p);
                err = err.max(((x - r) / r.abs().max(1e-300)).abs());
            }
            format!("{:.1} evals, max rel err of x {err:.1e}", ev / N as f64)
        };
        let params = format!("{name}; 64 instances; rtol 4√ε");
        let cnt = Cell::new(0usize);
        // golden_section
        {
            let on = stat(Box::new(|p| {
                cnt.set(0);
                let r = golden_section_with(|x| { cnt.set(cnt.get() + 1); min1(prob, p, x) }, a, b, &opts).unwrap();
                (r.x, cnt.get() as f64)
            }));
            let gn = stat(Box::new(|p| {
                let mut e = 0i64;
                let x = unsafe { probe_gsl_min(0, prob, p, a, b, guess, rtol, &mut e) };
                (x, e as f64)
            }));
            let an = stat(Box::new(|p| {
                cnt.set(0);
                let x = crate::argmin_probe::golden(a, b, guess, rtol / 2.0, &|x| { cnt.set(cnt.get() + 1); min1(prob, p, x) });
                (x, cnt.get() as f64)
            }));
            let mut k = (0usize, 0usize, 0usize);
            let mut f0 = || { let p = ps[k.0 % N]; k.0 += 1; bb(golden_section_with(|x| min1(prob, p, x), a, b, &opts)); };
            let mut k1 = 0usize;
            let mut f1 = || { let p = ps[k1 % N]; k1 += 1; let mut e = 0; bb(unsafe { probe_gsl_min(0, prob, p, a, b, guess, rtol, &mut e) }); };
            let mut k2 = 0usize;
            let mut f2 = || { let p = ps[k2 % N]; k2 += 1; bb(crate::argmin_probe::golden(a, b, guess, rtol / 2.0, &|x| min1(prob, p, x))); };
            Cmp { group: "opt", op: "golden_section", params: &params }.run(
                &mut [&mut f0, &mut f1, &mut f2],
                &["GSL 2.8 min goldensection", "argmin 0.11 GoldenSectionSearch"],
                &on,
                &[format!("{gn}; guess 1, test_interval(0, 4√ε)"), format!("{an}; tolerance 2√ε (its relative test on |x₁|+|x₂|)")],
                1.0,
            );
            let _ = &mut k;
        }
        // minimize (Brent)
        {
            let on = stat(Box::new(|p| {
                cnt.set(0);
                let r = minimize_with(|x| { cnt.set(cnt.get() + 1); min1(prob, p, x) }, a, b, &opts).unwrap();
                (r.x, cnt.get() as f64)
            }));
            let g1 = stat(Box::new(|p| { let mut e = 0i64; let x = unsafe { probe_gsl_min(1, prob, p, a, b, guess, rtol, &mut e) }; (x, e as f64) }));
            let bn = stat(Box::new(|p| { let mut e = 0i64; let x = unsafe { probe_boost_min(prob, p, a, b, 25, &mut e) }; (x, e as f64) }));
            let an = stat(Box::new(|p| {
                cnt.set(0);
                let x = crate::argmin_probe::brent_opt(a, b, rtol, &|x| { cnt.set(cnt.get() + 1); min1(prob, p, x) });
                (x, cnt.get() as f64)
            }));
            let mut k0 = 0usize;
            let mut f0 = || { let p = ps[k0 % N]; k0 += 1; bb(minimize_with(|x| min1(prob, p, x), a, b, &opts)); };
            let mut k1 = 0usize;
            let mut f1 = || { let p = ps[k1 % N]; k1 += 1; let mut e = 0; bb(unsafe { probe_gsl_min(1, prob, p, a, b, guess, rtol, &mut e) }); };
            let mut k3 = 0usize;
            let mut f3 = || { let p = ps[k3 % N]; k3 += 1; let mut e = 0; bb(unsafe { probe_boost_min(prob, p, a, b, 25, &mut e) }); };
            let mut k4 = 0usize;
            let mut f4 = || { let p = ps[k4 % N]; k4 += 1; bb(crate::argmin_probe::brent_opt(a, b, rtol, &|x| min1(prob, p, x))); };
            Cmp { group: "opt", op: "minimize", params: &params }.run(
                &mut [&mut f0, &mut f1, &mut f3, &mut f4],
                &["GSL 2.8 min brent", "Boost 1.92 brent_find_minima", "argmin 0.11 BrentOpt"],
                &on,
                &[
                    format!("{g1}; guess 1, test_interval(0, 4√ε)"),
                    format!("{bn}; bits 25 (tolerance 2⁻²⁴ ≈ 4√ε)"),
                    format!("{an}; eps 4√ε, t 1e-15"),
                ],
                1.0,
            );
        }
    }
}

fn several() {
    let cases: [(i32, usize, &str); 7] = [
        (0, 2, "Rosenbrock, n = 2"),
        (0, 10, "chained Rosenbrock, n = 10"),
        (0, 50, "chained Rosenbrock, n = 50"),
        (1, 4, "Powell singular, n = 4"),
        (2, 4, "Wood, n = 4"),
        (3, 2, "Beale, n = 2"),
        (4, 10, "quadratic, condition 10⁴, n = 10"),
    ];
    let gtol = f64::EPSILON.sqrt();
    for (prob, n, name) in cases {
        let lam = lambdas(n);
        let x0 = col(mn_start(prob, n));
        let sol = mn_solution(prob, n);
        let xerr = |x: &[f64]| x.iter().zip(&sol).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let fcount = Cell::new(0usize);
        let gcount = Cell::new(0usize);
        let f = |x: &Matrix<f64>| { fcount.set(fcount.get() + 1); mn_f(prob, &lam, x.as_slice()) };
        let g = |x: &Matrix<f64>| {
            gcount.set(gcount.get() + 1);
            let mut v = vec![0.0; n];
            mn_g(prob, &lam, x.as_slice(), &mut v);
            col(v)
        };
        let fq = |x: &Matrix<f64>| mn_f(prob, &lam, x.as_slice());
        let gq = |x: &Matrix<f64>| {
            let mut v = vec![0.0; n];
            mn_g(prob, &lam, x.as_slice(), &mut v);
            col(v)
        };
        let gsl = |kind: i32, step: f64, tol: f64| -> String {
            let mut x = mn_start(prob, n);
            let (mut fe, mut ge, mut it) = (0i64, 0i64, 0i32);
            let st = unsafe { probe_gsl_multimin(kind, prob, n as i32, x.as_mut_ptr(), tol, step, 100_000, &mut fe, &mut ge, &mut it) };
            format!(
                "{it} iterations, {fe} f, {ge} g, {}, |x−x*|∞ {:.1e}",
                if st == 0 { "converged".to_string() } else { format!("status {st} (stopped before the test: does less)") },
                xerr(&x)
            )
        };
        let gsl_t = |kind: i32, step: f64, tol: f64| {
            move || {
                let mut x = mn_start(prob, n);
                let (mut fe, mut ge, mut it) = (0i64, 0i64, 0i32);
                bb(unsafe { probe_gsl_multimin(kind, prob, n as i32, x.as_mut_ptr(), tol, step, 100_000, &mut fe, &mut ge, &mut it) });
            }
        };
        let params = format!("{name}; standard start; gtol √ε");
        // bfgs
        {
            fcount.set(0);
            gcount.set(0);
            let r = bfgs(f, g, &x0).unwrap();
            let on = format!(
                "{} iterations, {} f, {} g, {}converged, |x−x*|∞ {:.1e}; ‖g‖∞ ≤ √ε",
                r.iterations, fcount.get(), gcount.get(), if r.converged { "" } else { "not " }, xerr(r.x.as_slice())
            );
            let an = crate::argmin_probe::bfgs_note(prob, &lam, &mn_start(prob, n), gtol, &sol);
            let mut a = || { bb(bfgs(fq, gq, &x0)); };
            let mut g0 = gsl_t(0, 0.0, gtol);
            let x0v = mn_start(prob, n);
            let mut am = || { bb(crate::argmin_probe::bfgs(prob, &lam, &x0v, gtol)); };
            Cmp { group: "opt", op: "bfgs", params: &params }.run(
                &mut [&mut a, &mut g0, &mut am],
                &["GSL 2.8 multimin vector_bfgs2", "argmin 0.11 BFGS (More–Thuente)"],
                &on,
                &[format!("{}; ‖g‖₂ < √ε", gsl(0, 0.0, gtol)), format!("{an}; ‖g‖₂ < √ε")],
                1.0,
            );
        }
        // conjugate_gradient
        {
            fcount.set(0);
            gcount.set(0);
            let r = conjugate_gradient(f, g, &x0).unwrap();
            let on = format!(
                "{} iterations, {} f, {} g, {}converged, |x−x*|∞ {:.1e}; ‖g‖∞ ≤ √ε",
                r.iterations, fcount.get(), gcount.get(), if r.converged { "" } else { "not " }, xerr(r.x.as_slice())
            );
            let mut a = || { bb(conjugate_gradient(fq, gq, &x0)); };
            let mut g1 = gsl_t(1, 0.0, gtol);
            let mut g2 = gsl_t(2, 0.0, gtol);
            Cmp { group: "opt", op: "conjugate_gradient", params: &params }.run(
                &mut [&mut a, &mut g1, &mut g2],
                &["GSL 2.8 multimin conjugate_pr", "GSL 2.8 multimin conjugate_fr"],
                &on,
                &[
                    format!("{}; ‖g‖₂ < √ε", gsl(1, 0.0, gtol)),
                    format!("{}; ‖g‖₂ < √ε (argmin's NonlinearConjugateGradient has no gradient test, so no row)", gsl(2, 0.0, gtol)),
                ],
                1.0,
            );
        }
        // nelder_mead (small problems)
        if n <= 4 {
            let step = 0.5;
            fcount.set(0);
            let r = nelder_mead(f, &x0, step).unwrap();
            let on = format!(
                "{} iterations, {} f, {}converged, |x−x*|∞ {:.1e}; size ≤ 4√ε·|x| and spread ≤ 4√ε·max(|f|,1)",
                r.iterations, fcount.get(), if r.converged { "" } else { "not " }, xerr(r.x.as_slice())
            );
            let ntol = 4.0 * gtol;
            let an = crate::argmin_probe::nm_note(prob, &lam, &mn_start(prob, n), step, &sol);
            let mut a = || { bb(nelder_mead(fq, &x0, step)); };
            let mut g3 = gsl_t(3, step, ntol);
            let x0v = mn_start(prob, n);
            let mut am = || { bb(crate::argmin_probe::nm(prob, &lam, &x0v, step)); };
            let p2 = format!("{name}; standard start; first step 0.5");
            Cmp { group: "opt", op: "nelder_mead", params: &p2 }.run(
                &mut [&mut a, &mut g3, &mut am],
                &["GSL 2.8 multimin nmsimplex2", "argmin 0.11 NelderMead"],
                &on,
                &[format!("{}; simplex size < 4√ε", gsl(3, step, ntol)), format!("{an}; sd of the values < 1e-16")],
                1.0,
            );
        }
    }
}

struct Fit {
    name: &'static str,
    model: i32,
    x: Vec<f64>,
    y: Vec<f64>,
    start: Vec<f64>,
    certified: Vec<f64>,
}

fn model(m: i32, b: &[f64], x: f64, d: &mut [f64]) -> f64 {
    match m {
        0 => {
            let e = (-b[1] * x).exp();
            d[0] = 1.0 - e;
            d[1] = b[0] * x * e;
            b[0] * (1.0 - e)
        }
        1 => {
            let (x2, x3) = (x * x, x * x * x);
            let num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3;
            let den = 1.0 + b[4] * x + b[5] * x2 + b[6] * x3;
            let d2 = den * den;
            d[0] = 1.0 / den;
            d[1] = x / den;
            d[2] = x2 / den;
            d[3] = x3 / den;
            d[4] = -num * x / d2;
            d[5] = -num * x2 / d2;
            d[6] = -num * x3 / d2;
            num / den
        }
        2 => {
            let num = x * x + x * b[1];
            let den = x * x + x * b[2] + b[3];
            d[0] = num / den;
            d[1] = b[0] * x / den;
            d[2] = -b[0] * num * x / (den * den);
            d[3] = -b[0] * num / (den * den);
            b[0] * num / den
        }
        3 => {
            let (e4, e5) = ((-x * b[3]).exp(), (-x * b[4]).exp());
            d[0] = 1.0;
            d[1] = e4;
            d[2] = e5;
            d[3] = -b[1] * x * e4;
            d[4] = -b[2] * x * e5;
            b[0] + b[1] * e4 + b[2] * e5
        }
        _ => {
            let e = (-b[1] * x).exp();
            d[0] = e;
            d[1] = -b[0] * x * e;
            d[2] = 1.0;
            b[0] * e + b[2]
        }
    }
}

fn fits_list() -> Vec<Fit> {
    let d = &crate::nist::DATA;
    let mut v = vec![];
    for (name, model, x, y, starts, cert) in d.iter() {
        for (si, s) in starts.iter().enumerate() {
            // The hard first starts of MGH09, MGH17 and BoxBOD fail or wander for most codes.
            if si == 0 && matches!(*name, "MGH09" | "MGH17" | "BoxBOD") {
                continue;
            }
            v.push(Fit {
                name: if si == 0 { Box::leak(format!("{name}, start 1").into_boxed_str()) } else { Box::leak(format!("{name}, start 2").into_boxed_str()) },
                model: *model,
                x: x.to_vec(),
                y: y.to_vec(),
                start: s.to_vec(),
                certified: cert.to_vec(),
            });
        }
    }
    // Synthetic: 2·e^(−1.5t) + 0.05 with a deterministic perturbation, 1000 points on [0, 4].
    let m = 1000;
    let x: Vec<f64> = (0..m).map(|i| 4.0 * i as f64 / (m - 1) as f64).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, t)| 2.0 * (-1.5 * t).exp() + 0.05 + 1e-3 * ((i as f64) * 0.7).sin()).collect();
    v.push(Fit { name: "a·e^(−λt)+c, 1000 points (synthetic)", model: 4, x, y, start: vec![1.0, 1.0, 0.0], certified: vec![] });
    v
}

/// (name, model, x, y, start, certified) of every fit, for py/refs.py.
pub fn fits_export() -> Vec<(String, i32, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    fits_list().into_iter().map(|f| (f.name.to_string(), f.model, f.x, f.y, f.start, f.certified)).collect()
}

fn fits() {
    for fit in fits_list() {
        let (m, p) = (fit.x.len(), fit.start.len());
        let digits = |b: &[f64]| -> f64 {
            if fit.certified.is_empty() {
                return f64::NAN;
            }
            b.iter()
                .zip(&fit.certified)
                .map(|(b, c)| {
                    let e = ((b - c) / c).abs();
                    if e == 0.0 { 16.0 } else { -e.log10() }
                })
                .fold(16.0f64, f64::min)
                .max(0.0)
        };
        let fe = Cell::new(0usize);
        let je = Cell::new(0usize);
        let res = |b: &Matrix<f64>| {
            fe.set(fe.get() + 1);
            let mut d = [0.0; 8];
            col(fit.x.iter().zip(&fit.y).map(|(x, y)| model(fit.model, b.as_slice(), *x, &mut d) - y).collect())
        };
        let jac = |b: &Matrix<f64>| {
            je.set(je.get() + 1);
            let mut j = vec![0.0; m * p];
            let mut d = [0.0; 8];
            for (i, x) in fit.x.iter().enumerate() {
                model(fit.model, b.as_slice(), *x, &mut d);
                for k in 0..p {
                    j[k * m + i] = d[k];
                }
            }
            Matrix::from_column_major(m, p, j)
        };
        let x0 = col(fit.start.clone());
        let (xtol, gtol) = (1e-8, f64::EPSILON.cbrt());
        let gsl = |trs: i32| -> (String, bool) {
            let mut b = fit.start.clone();
            let (mut f, mut j, mut it) = (0i64, 0i64, 0i32);
            let st = unsafe { probe_gsl_fit(trs, fit.model, m as i32, p as i32, fit.x.as_ptr(), fit.y.as_ptr(), b.as_mut_ptr(), xtol, gtol, 500, &mut f, &mut j, &mut it) };
            (format!("{it} iterations, {f} f, {j} J, {}, {:.1} digits; xtol 1e-8, gtol ε^(1/3)", if st == 0 { "converged".into() } else { format!("status {st}") }, digits(&b)), st == 0)
        };
        let gsl_t = |trs: i32| {
            let fit = &fit;
            move || {
                let mut b = fit.start.clone();
                let (mut f, mut j, mut it) = (0i64, 0i64, 0i32);
                bb(unsafe { probe_gsl_fit(trs, fit.model, m as i32, p as i32, fit.x.as_ptr(), fit.y.as_ptr(), b.as_mut_ptr(), xtol, gtol, 500, &mut f, &mut j, &mut it) });
            }
        };
        let params = format!("{}; m = {m}, p = {p}", fit.name);
        for (op, lm) in [("levenberg_marquardt", true), ("gauss_newton", false)] {
            fe.set(0);
            je.set(0);
            let r = if lm { levenberg_marquardt(res, jac, &x0) } else { gauss_newton(res, jac, &x0) };
            let Some(r) = r else { continue };
            let on = format!(
                "{} iterations, {} f, {} J, {}converged, {:.1} digits",
                r.iterations, fe.get(), je.get(), if r.converged { "" } else { "not " }, digits(r.x.as_slice())
            );
            if !r.converged {
                crate::harness::Row {
                    group: "fit".into(),
                    op: op.into(),
                    params: params.clone(),
                    ours: f64::NAN,
                    reference: "none (ours does not converge from this start: no same-work comparison)".into(),
                    refns: None,
                    notes: format!("ours: {on}; GSL lm: {}", gsl(0).0),
                }
                .print();
                continue;
            }
            let mut a = || {
                if lm { bb(levenberg_marquardt(res, jac, &x0)); } else { bb(gauss_newton(res, jac, &x0)); }
            };
            let mut g0 = gsl_t(0);
            let mut g1 = gsl_t(1);
            let mut g2 = gsl_t(2);
            let tt = Cmp { group: "fit", op, params: &params }.run(
                &mut [&mut a, &mut g0, &mut g1, &mut g2],
                &["GSL 2.8 multifit_nlinear trust lm", "GSL 2.8 multifit_nlinear trust lmaccel", "GSL 2.8 multifit_nlinear trust dogleg"],
                &on,
                &[gsl(0).0, gsl(1).0, gsl(2).0],
                1.0,
            );
            if lm {
                crate::harness::py_rows("fit", op, &params, &format!("fit:{}", fit.name), tt[0].median, &on, None);
            }
        }
    }
}
