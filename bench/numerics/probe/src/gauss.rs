//! Gaussian rules: the nodes and weights by order, against GSL's glfixed (Legendre) and
//! gsl_integration_fixed (Golub–Welsch for every family).

use crate::harness::{Cmp, bb};
use a::math::numerics::{NodesWeights, hermite, hermite_scaled, jacobi, laguerre, laguerre_scaled, legendre};

unsafe extern "C" {
    fn probe_gsl_gauss(t: i32, n: i32, alpha: f64, beta: f64, x: *mut f64, w: *mut f64);
}

fn ulps(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    let u = if b == 0.0 { f64::MIN_POSITIVE } else { (b.abs() * f64::EPSILON).max(f64::MIN_POSITIVE * f64::EPSILON) };
    ((a - b) / u).abs()
}

/// Largest node error in ulps of the node (relative to the largest |node| for nodes near 0) and
/// largest weight error relative to the largest weight in units of ε, of the reference against
/// ours (ours are the correctly rounded values, the crate's tests check them against mpmath).
fn compare(ours: &NodesWeights<f64>, x: &[f64], w: &[f64]) -> String {
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(w.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut ourp: Vec<(f64, f64)> = ours.nodes.iter().copied().zip(ours.weights.iter().copied()).collect();
    ourp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if pairs.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        return "reference has NaN/∞ entries".into();
    }
    let xmax = ourp.iter().fold(0.0f64, |m, p| m.max(p.0.abs()));
    let wmax = ourp.iter().fold(0.0f64, |m, p| m.max(p.1.abs()));
    let (mut ex, mut ew, mut ewr) = (0.0f64, 0.0f64, 0.0f64);
    for (r, o) in pairs.iter().zip(&ourp) {
        ex = ex.max((r.0 - o.0).abs() / (xmax * f64::EPSILON));
        ew = ew.max((r.1 - o.1).abs() / (wmax * f64::EPSILON));
        if o.1 > 1e-280 {
            ewr = ewr.max(ulps(r.1, o.1));
        }
    }
    format!(
        "nodes within {ex:.1} ε·max|x|, weights within {ew:.1} ε·max w ({ewr:.2e} ulps of the weight at worst) of ours{}",
        if ew > 1000.0 || ex > 1000.0 { " (does less: weights off by more than 1000 ε·max w)" } else { "" }
    )
}

pub fn run() {
    // (name, ours, gsl type, alpha, beta, sizes)
    let sizes = [5usize, 20, 100, 101, 1000, 10_000];
    for n in sizes {
        let reps_heavy = n >= 10_000;
        // Legendre
        {
            let o = legendre::<f64>(n);
            let mut x = vec![0.0; n];
            let mut w = vec![0.0; n];
            unsafe { probe_gsl_gauss(0, n as i32, 0.0, 0.0, x.as_mut_ptr(), w.as_mut_ptr()) };
            let n0 = compare(&o, &x, &w);
            unsafe { probe_gsl_gauss(1, n as i32, 0.0, 0.0, x.as_mut_ptr(), w.as_mut_ptr()) };
            let n1 = compare(&o, &x, &w);
            let mut a = || {
                bb(legendre::<f64>(n));
            };
            let mut g0 = || {
                unsafe { probe_gsl_gauss(0, n as i32, 0.0, 0.0, x.as_mut_ptr(), w.as_mut_ptr()) };
                bb(&w);
            };
            let mut x1 = vec![0.0; n];
            let mut w1 = vec![0.0; n];
            let mut g1 = || {
                unsafe { probe_gsl_gauss(1, n as i32, 0.0, 0.0, x1.as_mut_ptr(), w1.as_mut_ptr()) };
                bb(&w1);
            };
            let table = matches!(n, 2..=20 | 32 | 64 | 96 | 100 | 128 | 256 | 512 | 1024);
            let params = format!("n = {n}");
            let ours_note = if n <= 100 { "Golub–Welsch + Newton polish in double-double" } else { "asymptotic guesses + Newton, polished in double-double" };
            let tt = Cmp { group: "gauss", op: "legendre", params: &params }.run(
                &mut [&mut a, &mut g0, &mut g1],
                &["GSL 2.8 glfixed_table_alloc", "GSL 2.8 integration_fixed legendre"],
                ours_note,
                &[
                    format!("{}; {}", n0, if table { "precomputed table: a copy, no computation (does less)" } else { "Newton on the recurrence" }),
                    format!("{n1}; Golub–Welsch (implicit QL, first components)"),
                ],
                1.0,
            );
            crate::harness::py_rows("gauss", "legendre", &params, &format!("gauss:legendre:{n}"), tt[0].median, ours_note, None);
        }
        if reps_heavy {
            // GSL's Golub–Welsch at 10⁴ points takes seconds per call; Legendre above covers it.
        }
        for (op, t, al, be) in [("hermite", 2, 0.0, 0.0), ("laguerre", 3, 0.0, 0.0), ("jacobi", 4, 0.5, 1.5)] {
            if n >= 10_000 && t == 4 {
                continue; // ours is O(n³) Golub–Welsch for Jacobi at every order: skip 10⁴
            }
            let ours = |n: usize| -> NodesWeights<f64> {
                match t {
                    2 => hermite(n),
                    3 => laguerre(n),
                    _ => jacobi(n, al, be),
                }
            };
            let o = ours(n);
            let mut x = vec![0.0; n];
            let mut w = vec![0.0; n];
            unsafe { probe_gsl_gauss(t, n as i32, al, be, x.as_mut_ptr(), w.as_mut_ptr()) };
            let note = compare(&o, &x, &w);
            let mut a = || {
                bb(ours(n));
            };
            let mut g = || {
                unsafe { probe_gsl_gauss(t, n as i32, al, be, x.as_mut_ptr(), w.as_mut_ptr()) };
                bb(&w);
            };
            let params = if t == 4 { format!("n = {n}; α = 0.5, β = 1.5") } else { format!("n = {n}") };
            let on = if t == 4 {
                "Golub–Welsch at every order (O(n³)), Newton polish"
            } else if n <= 100 {
                "Golub–Welsch + Newton polish in double-double"
            } else {
                "asymptotic guesses + Newton, polished in double-double"
            };
            let tt = Cmp { group: "gauss", op, params: &params }.run(
                &mut [&mut a, &mut g],
                &[match t {
                    2 => "GSL 2.8 integration_fixed hermite",
                    3 => "GSL 2.8 integration_fixed laguerre",
                    _ => "GSL 2.8 integration_fixed jacobi",
                }],
                on,
                &[format!("{note}; Golub–Welsch (implicit QL, first components)")],
                1.0,
            );
            crate::harness::py_rows("gauss", op, &params, &format!("gauss:{op}:{n}"), tt[0].median, on, None);
        }
        if n == 100 || n == 1000 {
            for (op, t) in [("hermite_scaled", 2), ("laguerre_scaled", 3)] {
                let mut a = || {
                    bb(if t == 2 { hermite_scaled::<f64>(n) } else { laguerre_scaled::<f64>(n) });
                };
                let mut b = || {
                    bb(if t == 2 { hermite::<f64>(n) } else { laguerre::<f64>(n) });
                };
                let t2 = crate::harness::measure(&mut [&mut a, &mut b]);
                crate::harness::Row {
                    group: "gauss".into(),
                    op: op.into(),
                    params: format!("n = {n}"),
                    ours: t2[0].median,
                    reference: "none".into(),
                    refns: None,
                    notes: format!("no reference has the scaled weights; ours unscaled rule {:.0} ns", t2[1].median),
                }
                .print();
            }
        }
    }
}
