//! The comparisons whose references are Python's (HiGHS through SciPy, SciPy's L-BFGS-B,
//! cubature and AAA), plus `export`, which writes the inputs py/refs.py reads.

use crate::harness::{Cmp, Lcg, Row, bb, measure, py_rows};
use a::math::complex::Complex;
use a::math::matrix::Matrix;
use a::math::numerics::{LbfgsbOptions, LinearProgram, LpOptions, QuadOptions, aaa, aaa_complex, cubature, lbfgsb_with};
use std::io::Write;

// ---------------------------------------------------------------------------------------------
// Linear programmes
// ---------------------------------------------------------------------------------------------

pub struct Lp {
    pub name: String,
    pub mle: usize,
    pub meq: usize,
    pub n: usize,
    pub c: Vec<f64>,
    pub ale: Vec<f64>,
    pub ble: Vec<f64>,
    pub aeq: Vec<f64>,
    pub beq: Vec<f64>,
    pub lo: Vec<f64>,
    pub up: Vec<f64>,
}

/// A feasible, bounded programme: rows through a random point x* with random slacks, every
/// variable in [0, 10] but a tenth in [0, ∞) whose costs are non-negative.
fn make_lp(family: &str, mle: usize, meq: usize, n: usize, density: f64, seed: u64) -> Lp {
    let mut r = Lcg(seed);
    let entry = |r: &mut Lcg| if r.next() < density { 2.0 * r.next() - 1.0 } else { 0.0 };
    let xs: Vec<f64> = (0..n).map(|_| r.next()).collect();
    let mut ale = vec![0.0; mle * n];
    for v in ale.iter_mut() {
        *v = entry(&mut r);
    }
    let mut aeq = vec![0.0; meq * n];
    for v in aeq.iter_mut() {
        *v = entry(&mut r);
    }
    let ble: Vec<f64> = (0..mle).map(|i| (0..n).map(|j| ale[i * n + j] * xs[j]).sum::<f64>() + r.next()).collect();
    let beq: Vec<f64> = (0..meq).map(|i| (0..n).map(|j| aeq[i * n + j] * xs[j]).sum::<f64>()).collect();
    let mut lo = vec![0.0; n];
    let mut up = vec![10.0; n];
    let mut c = vec![0.0; n];
    for j in 0..n {
        if j % 10 == 9 {
            up[j] = f64::INFINITY;
            c[j] = r.next();
        } else {
            c[j] = 2.0 * r.next() - 1.0;
        }
        lo[j] = 0.0;
    }
    Lp { name: format!("{family}_{mle}x{meq}x{n}"), mle, meq, n, c, ale, ble, aeq, beq, lo, up }
}

fn lps() -> Vec<Lp> {
    let mut v = vec![];
    for (i, &(mle, meq, n)) in [(10usize, 0usize, 20usize), (30, 5, 60), (100, 10, 200), (200, 20, 400), (300, 30, 600)].iter().enumerate() {
        v.push(make_lp("dense", mle, meq, n, 1.0, 100 + i as u64));
        if n >= 60 {
            v.push(make_lp("sparse10", mle, meq, n, 0.1, 200 + i as u64));
        }
    }
    // Rows added by the lp area (2026-11-25): larger, and sparser.
    v.push(make_lp("dense", 500, 50, 1000, 1.0, 105));
    v.push(make_lp("sparse10", 500, 50, 1000, 0.1, 205));
    v.push(make_lp("sparse02", 200, 20, 400, 0.02, 303));
    v.push(make_lp("sparse02", 500, 50, 1000, 0.02, 305));
    v
}

fn solve_lp(p: &Lp) -> Option<a::math::numerics::LpResult<f64>> {
    let mut lp = LinearProgram::minimize(&p.c).subject_to_le(&Matrix::from_row_major(p.mle, p.n, p.ale.clone()), &p.ble);
    if p.meq > 0 {
        lp = lp.subject_to_eq(&Matrix::from_row_major(p.meq, p.n, p.aeq.clone()), &p.beq);
    }
    lp.bounds(&p.lo, &p.up).solve()
}

pub fn lp() {
    for p in lps() {
        // The matrices are built outside the timed call: LinearProgram takes them by reference.
        let ale = Matrix::from_row_major(p.mle, p.n, p.ale.clone());
        let aeq = Matrix::from_row_major(p.meq.max(1), p.n, if p.meq > 0 { p.aeq.clone() } else { vec![0.0; p.n] });
        let build = || {
            let mut lp = LinearProgram::minimize(&p.c).subject_to_le(&ale, &p.ble);
            if p.meq > 0 {
                lp = lp.subject_to_eq(&aeq, &p.beq);
            }
            lp.bounds(&p.lo, &p.up)
        };
        let r = solve_lp(&p);
        let note = match &r {
            Some(r) => format!("{:?}, objective {:.12}, {} pivots", r.status, r.objective, r.pivots),
            None => "None".into(),
        };
        let lp = build();
        let mut f = || {
            bb(lp.solve());
        };
        let t = measure(&mut [&mut f]);
        let params = format!(
            "{} rows ≤, {} rows =, {} variables, density {}; feasible, bounded",
            p.mle,
            p.meq,
            p.n,
            if p.name.starts_with("dense") { "1" } else if p.name.starts_with("sparse02") { "0.02" } else { "0.1" }
        );
        let method = if p.mle + p.meq >= LpOptions::<f64>::new().dual_threshold {
            "dual simplex (sparse LU, dual steepest edge, bound flipping), basis certified by the primal method"
        } else {
            "dense two-phase primal simplex"
        };
        py_rows("lp", "LinearProgram::solve", &params, &format!("lp:{}", p.name), t[0].median, &format!("{note}; {method}, default options"), None);
    }
}

// ---------------------------------------------------------------------------------------------
// L-BFGS-B
// ---------------------------------------------------------------------------------------------

/// The extended (separable) Rosenbrock function of Moré, Garbow and Hillstrom, n even.
fn rosen_f(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in (0..x.len()).step_by(2) {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s += 100.0 * a * a + b * b;
    }
    s
}
fn rosen_g(x: &[f64], g: &mut [f64]) {
    for i in (0..x.len()).step_by(2) {
        let a = x[i + 1] - x[i] * x[i];
        g[i] = -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
        g[i + 1] = 200.0 * a;
    }
}

pub fn lbfgsb() {
    for (prob, n, bounded) in [(0, 1000usize, false), (0, 10000, false), (0, 100000, false), (0, 1000, true), (0, 10000, true), (5, 1000, true), (5, 100000, true)] {
        let lam: Vec<f64> = (0..n).map(|i| 10f64.powf(3.0 * i as f64 / (n - 1) as f64)).collect();
        let cc: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64 + 1.0).sin()).collect();
        let fv = |x: &[f64]| -> f64 {
            if prob == 0 {
                rosen_f(x)
            } else {
                x.iter().zip(&lam).zip(&cc).map(|((x, l), c)| 0.5 * l * (x - c) * (x - c)).sum()
            }
        };
        let gv = |x: &[f64], g: &mut [f64]| {
            if prob == 0 {
                rosen_g(x, g)
            } else {
                for i in 0..x.len() {
                    g[i] = lam[i] * (x[i] - cc[i]);
                }
            }
        };
        let x0v: Vec<f64> = if prob == 0 {
            if bounded { vec![0.5; n] } else { (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect() }
        } else {
            vec![0.0; n]
        };
        let (lo, hi) = if prob == 0 {
            if bounded { (0.0, 0.8) } else { (f64::NEG_INFINITY, f64::INFINITY) }
        } else {
            (-1.0, 1.0)
        };
        let col = |v: Vec<f64>| Matrix::from_column_major(n, 1, v);
        let x0 = col(x0v.clone());
        let lower = col(vec![lo; n]);
        let upper = col(vec![hi; n]);
        let f = |x: &Matrix<f64>| fv(x.as_slice());
        let g = |x: &Matrix<f64>| {
            let mut v = vec![0.0; n];
            gv(x.as_slice(), &mut v);
            col(v)
        };
        let o = LbfgsbOptions::new();
        let r = lbfgsb_with(f, g, &x0, &lower, &upper, &o).unwrap();
        let on = format!("{} iterations, {} f+g, f {:.10}, {:?}", r.iterations, r.evaluations, r.value, r.stop);
        // ours cost of one f+g, at the start
        let mut fg = || {
            bb(f(&x0));
            bb(g(&x0));
        };
        let pe = measure(&mut [&mut fg])[0].median;
        let name = if prob == 0 {
            if bounded { format!("extended Rosenbrock in [0, 0.8]ⁿ from 0.5, n = {n}") } else { format!("extended Rosenbrock from (−1.2, 1, …), n = {n}") }
        } else {
            format!("quadratic of condition 10³ in [−1, 1]ⁿ (a third of the bounds active), n = {n}")
        };
        let params = format!("{name}; m = 10, pgtol 1e-5, ftol 1e7·ε");
        let mut a = || {
            bb(lbfgsb_with(f, g, &x0, &lower, &upper, &o));
        };
        let tt = if !bounded {
            let (x, it, fe, ge) = crate::argmin_probe::lbfgs(6, &[], &x0v, 10, 1e-5);
            let mut am = || {
                bb(crate::argmin_probe::lbfgs(6, &[], &x0v, 10, 1e-5));
            };
            Cmp { group: "lbfgsb", op: "lbfgsb", params: &params }.run(
                &mut [&mut a, &mut am],
                &["argmin 0.11 L-BFGS (unbounded, More–Thuente)"],
                &on,
                &[format!("{it} iterations, {fe} f, {ge} g, f {:.10}; ‖g‖₂ < 1e-5 (another stopping test)", rosen_f(&x))],
                1.0,
            )
        } else {
            measure(&mut [&mut a])
        };
        py_rows("lbfgsb", "lbfgsb", &params, &format!("lbfgsb:{prob}:{n}:{}", bounded as u8), tt[0].median, &format!("{on}; one f+g {pe:.0} ns"), Some(pe));
    }
}

// ---------------------------------------------------------------------------------------------
// Cubature
// ---------------------------------------------------------------------------------------------

fn genz(fam: &str, d: usize, x: &[f64]) -> f64 {
    match fam {
        "oscillatory" => (2.0 * std::f64::consts::PI * 0.3 + 1.5 * x.iter().sum::<f64>()).cos(),
        "product_peak" => x.iter().map(|t| 1.0 / (5f64.powi(-2) + (t - 0.4) * (t - 0.4))).product(),
        "gaussian" => (-9.0 * x.iter().map(|t| (t - 0.45) * (t - 0.45)).sum::<f64>()).exp(),
        _ => (1.0 + x.iter().sum::<f64>()).powi(-(d as i32 + 1)),
    }
}

pub fn cub() {
    for (fam, d) in [("oscillatory", 3usize), ("product_peak", 2), ("product_peak", 3), ("gaussian", 4), ("corner_peak", 3), ("gaussian", 5)] {
        let a = vec![0.0; d];
        let b = vec![1.0; d];
        let o = QuadOptions::new().with_epsabs(0.0).with_epsrel(1e-6).with_max_subintervals(100_000).with_max_evaluations(100_000_000);
        let r = cubature(|x: &[f64]| genz(fam, d, x), &a, &b, &o).unwrap();
        let mut f = || {
            bb(cubature(|x: &[f64]| genz(fam, d, x), &a, &b, &o));
        };
        let t = measure(&mut [&mut f]);
        let params = format!("Genz {fam}, d = {d}; epsrel 1e-6, epsabs 0");
        let on = format!("{} evals, {:.12} ± {:.1e}, {}converged; Genz–Malik degree 7/5", r.evaluations, r.value, r.error, if r.converged { "" } else { "not " });
        py_rows("quad", "cubature", &params, &format!("cub:{fam}:{d}"), t[0].median, &on, None);
    }
}

// ---------------------------------------------------------------------------------------------
// AAA
// ---------------------------------------------------------------------------------------------

fn linspace(a: f64, b: f64, m: usize) -> Vec<f64> {
    let step = (b - a) / (m - 1) as f64;
    let mut v: Vec<f64> = (0..m).map(|i| a + i as f64 * step).collect();
    v[m - 1] = b;
    v
}

pub fn aaa_group() {
    let m = 1000;
    let lin = linspace(-1.0, 1.0, m);
    let mid: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5) / m as f64 * 2.0 - 1.0).collect();
    let sq = linspace(0.0, 1.0, m);
    let g = linspace(-1.5, 1.5, 100);
    let cases: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        ("exp", lin.clone(), lin.iter().map(|x| x.exp()).collect()),
        ("tan", mid.clone(), mid.iter().map(|x| (std::f64::consts::PI * x / 2.0).tan()).collect()),
        ("abs", lin.clone(), lin.iter().map(|x| x.abs()).collect()),
        ("sqrt", sq.clone(), sq.iter().map(|x| x.sqrt()).collect()),
        ("runge", lin.clone(), lin.iter().map(|x| 1.0 / (1.0 + 25.0 * x * x)).collect()),
        ("gamma", g.clone(), g.iter().map(|x| a::math::special::gamma(*x)).collect()),
    ];
    let mut rng = Lcg(4);
    let pts: Vec<f64> = (0..10_000).map(|_| -0.99 + 1.98 * rng.next()).collect();
    for (name, x, y) in &cases {
        let r = aaa(x, y).unwrap();
        let mut f = || {
            bb(aaa(x, y));
        };
        let t = measure(&mut [&mut f]);
        let params = format!("{name}, {} samples; defaults (rtol ε^(3/4), clean-up on)", x.len());
        let on = format!("{} support points, error {:.1e}", r.support_points().len(), r.error());
        Row { group: "aaa".into(), op: "aaa".into(), params: params.clone(), ours: t[0].median, reference: "none (C)".into(), refns: None, notes: on.clone() }.print();
        py_rows("aaa", "aaa", &params, &format!("aaa:{name}:build"), t[0].median, &on, None);
        let pp: Vec<f64> = if *name == "sqrt" { pts.iter().map(|t| (t + 1.0) / 2.0).collect() } else { pts.clone() };
        let mut e = || {
            let mut acc = 0.0;
            for &p in &pp {
                acc += r.evaluate(p);
            }
            bb(acc);
        };
        let t = measure(&mut [&mut e]);
        let p2 = format!("{name}; 10⁴ random points; ns per point");
        py_rows("aaa", "Aaa::evaluate", &p2, &format!("aaa:{name}:eval"), t[0].median / pp.len() as f64, &format!("{} terms", r.support_points().len()), None);
        if matches!(*name, "abs" | "tan" | "gamma") {
            let mut fp = || { bb(r.poles()); };
            let mut fr = || { bb(r.residues()); };
            let mut fz = || { bb(r.zeros()); };
            let t = measure(&mut [&mut fp, &mut fr, &mut fz]);
            for (i, what) in ["poles", "residues", "zeros"].iter().enumerate() {
                let p3 = format!("{name}, {} terms", r.support_points().len());
                py_rows("aaa", &format!("Aaa::{what}"), &p3, &format!("aaa:{name}:{what}"), t[i].median, "", None);
            }
        }
    }
    // complex: log(2 + z⁴)/(1 − 16 z⁴) on the unit circle
    let z: Vec<Complex<f64>> = (0..m).map(|k| Complex::from_polar(1.0, 2.0 * std::f64::consts::PI * k as f64 / m as f64)).collect();
    let fz: Vec<Complex<f64>> = z
        .iter()
        .map(|&z| {
            let z4 = z.powi(4);
            (Complex { re: 2.0, im: 0.0 } + z4).ln() / (Complex { re: 1.0, im: 0.0 } - Complex { re: 16.0, im: 0.0 } * z4)
        })
        .collect();
    let r = aaa_complex(&z, &fz).unwrap();
    let mut f = || {
        bb(aaa_complex(&z, &fz));
    };
    let t = measure(&mut [&mut f]);
    let params = "log(2+z⁴)/(1−16z⁴), 1000 points of the unit circle; defaults".to_string();
    let on = format!("{} support points", r.support_points().len());
    py_rows("aaa", "aaa_complex", &params, "aaa:circle:build", t[0].median, &on, None);
    let mut rng = Lcg(4);
    let zp: Vec<Complex<f64>> = (0..10_000).map(|_| Complex::from_polar(1.0, 2.0 * std::f64::consts::PI * rng.next())).collect();
    let mut e = || {
        let mut acc = 0.0;
        for &p in &zp {
            acc += r.evaluate(p).re;
        }
        bb(acc);
    };
    let t = measure(&mut [&mut e]);
    py_rows("aaa", "Aaa::evaluate (complex)", "circle; 10⁴ random points; ns per point", "aaa:circle:eval", t[0].median / zp.len() as f64, &on, None);
}

// ---------------------------------------------------------------------------------------------
// Export of the inputs for py/refs.py
// ---------------------------------------------------------------------------------------------

pub fn export(dir: &str) {
    std::fs::create_dir_all(dir).unwrap();
    for p in lps() {
        let mut f = std::fs::File::create(format!("{dir}/lp_{}.txt", p.name)).unwrap();
        writeln!(f, "{} {} {}", p.mle, p.meq, p.n).unwrap();
        let w = |f: &mut std::fs::File, v: &[f64]| {
            let s: Vec<String> = v.iter().map(|x| if x.is_infinite() { if *x > 0.0 { "inf".into() } else { "-inf".into() } } else { format!("{x:e}") }).collect();
            writeln!(f, "{}", s.join(" ")).unwrap();
        };
        w(&mut f, &p.c);
        w(&mut f, &p.ale);
        w(&mut f, &p.ble);
        w(&mut f, &p.aeq);
        w(&mut f, &p.beq);
        w(&mut f, &p.lo);
        w(&mut f, &p.up);
    }
    // ODE references and the linear problem's matrix
    let mut f = std::fs::File::create(format!("{dir}/ode_ref.txt")).unwrap();
    for prob in [0, 1, 2, 3, 10, 11, 12, 13, 14] {
        let r = crate::ode::reference_pub(prob);
        let s: Vec<String> = r.iter().map(|x| format!("{x:e}")).collect();
        writeln!(f, "{prob} {}", s.join(" ")).unwrap();
    }
    let mut f = std::fs::File::create(format!("{dir}/ode_lin_A.txt")).unwrap();
    let a = crate::ode::lin_a_pub();
    let s: Vec<String> = a.iter().map(|x| format!("{x:e}")).collect();
    writeln!(f, "{}", s.join(" ")).unwrap();
    // fits
    let mut out = String::from("[");
    for (i, fit) in crate::opt::fits_export().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let arr = |v: &[f64]| format!("[{}]", v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(","));
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"model\":{},\"x\":{},\"y\":{},\"start\":{},\"certified\":{}}}",
            fit.0, fit.1, arr(&fit.2), arr(&fit.3), arr(&fit.4), arr(&fit.5)
        ));
    }
    out.push(']');
    std::fs::write(format!("{dir}/fits.json"), out).unwrap();
}
