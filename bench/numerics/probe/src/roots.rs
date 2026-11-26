//! Roots of one variable: bisect, brent, secant, newton, halley against GSL, Boost.Math, the
//! `roots` crate and argmin.

use crate::harness::{Cmp, bb};
use a::math::numerics::{
    RootOptions, bisect_with, brent_with, halley_with, newton_with, secant_with, toms748_with,
};
use std::cell::Cell;

unsafe extern "C" {
    fn probe_gsl_root_bracket(
        kind: i32,
        prob: i32,
        p: f64,
        lo: f64,
        hi: f64,
        epsrel: f64,
        evals: *mut i64,
    ) -> f64;
    fn probe_gsl_root_fdf(
        kind: i32,
        prob: i32,
        p: f64,
        x0: f64,
        epsrel: f64,
        evals: *mut i64,
        devals: *mut i64,
    ) -> f64;
    fn probe_boost_root_bracket(
        kind: i32,
        prob: i32,
        p: f64,
        lo: f64,
        hi: f64,
        bits: i32,
        evals: *mut i64,
    ) -> f64;
    fn probe_boost_root_open(
        kind: i32,
        prob: i32,
        p: f64,
        x0: f64,
        lo: f64,
        hi: f64,
        digits: i32,
        evals: *mut i64,
    ) -> f64;
}

#[inline]
fn f(prob: i32, p: f64, x: f64) -> f64 {
    match prob {
        0 => x * x - p,
        1 => x.cos() - p * x,
        2 => {
            let d = x - p;
            d * d * d
        }
        _ => (x - p).cbrt(),
    }
}
#[inline]
fn df(prob: i32, p: f64, x: f64) -> f64 {
    match prob {
        0 => 2.0 * x,
        1 => -x.sin() - p,
        2 => {
            let d = x - p;
            3.0 * d * d
        }
        _ => {
            let c = (x - p).cbrt();
            1.0 / (3.0 * c * c)
        }
    }
}
#[inline]
fn d2f(prob: i32, p: f64, x: f64) -> f64 {
    match prob {
        0 => 2.0,
        1 => -x.cos(),
        2 => 6.0 * (x - p),
        _ => {
            let c = (x - p).cbrt();
            -2.0 / (9.0 * c.powi(5))
        }
    }
}
fn exact(prob: i32, p: f64) -> f64 {
    match prob {
        0 => p.sqrt(),
        1 => {
            // cos x = p x by Newton in f64 to convergence.
            let mut x: f64 = 0.7;
            for _ in 0..60 {
                x -= (x.cos() - p * x) / (-x.sin() - p);
            }
            x
        }
        _ => p,
    }
}

struct Problem {
    prob: i32,
    name: &'static str,
    lo: f64,
    hi: f64,
    plo: f64,
    phi: f64,
    x0: f64,
}

const PROBLEMS: [Problem; 4] = [
    Problem { prob: 0, name: "x²−p on [1,2], p∈[1.2,3.8]", lo: 1.0, hi: 2.0, plo: 1.2, phi: 3.8, x0: 1.5 },
    Problem { prob: 1, name: "cos x−p·x on [0,2], p∈[0.3,2]", lo: 0.0, hi: 2.0, plo: 0.3, phi: 2.0, x0: 0.6 },
    Problem { prob: 2, name: "(x−p)³ on [0,3], p∈[0.8,1.2] (triple root)", lo: 0.0, hi: 3.0, plo: 0.8, phi: 1.2, x0: 2.0 },
    Problem { prob: 3, name: "∛(x−p) on [−1,3], p∈[0.2,1.8] (infinite slope)", lo: -1.0, hi: 3.0, plo: 0.2, phi: 1.8, x0: 2.0 },
];

const N: usize = 64;

fn params(pr: &Problem) -> Vec<f64> {
    (0..N)
        .map(|i| pr.plo + (pr.phi - pr.plo) * ((i * 37 % N) as f64 + 0.5) / N as f64)
        .collect()
}

/// Mean evaluations and the largest error relative to the root over the instances.
fn stats(ps: &[f64], prob: i32, mut run: impl FnMut(f64) -> (f64, f64)) -> (f64, f64) {
    let mut ev = 0.0;
    let mut err: f64 = 0.0;
    for &p in ps {
        let (x, e) = run(p);
        ev += e;
        let r = exact(prob, p);
        err = err.max(((x - r) / r).abs());
    }
    (ev / ps.len() as f64, err)
}

fn note(ev: f64, err: f64) -> String {
    format!("{ev:.1} evals, max rel err {err:.1e}")
}

pub fn run() {
    let eps = f64::EPSILON;
    let opts = RootOptions::<f64>::new(); // xtol 0, rtol 4ε
    for pr in &PROBLEMS {
        let ps = params(pr);
        let (lo, hi, prob) = (pr.lo, pr.hi, pr.prob);
        let params_s = format!("{}; 64 instances; rtol 4ε", pr.name);

        // ---- bisect
        {
            let cnt = Cell::new(0usize);
            let (ev, err) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = bisect_with(lo, hi, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, &opts)
                .unwrap();
                (x, cnt.get() as f64)
            });
            let ours_n = note(ev, err);
            let (gev, gerr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_gsl_root_bracket(0, prob, p, lo, hi, 4.0 * eps, &mut e) };
                (x, e as f64)
            });
            let (bev, berr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_boost_root_bracket(0, prob, p, lo, hi, 53, &mut e) };
                (x, e as f64)
            });
            let mut k = 0usize;
            let mut a = || {
                let p = ps[k % N];
                k += 1;
                bb(bisect_with(lo, hi, |x| f(prob, p, x), &opts));
            };
            let mut k2 = 0usize;
            let mut g = || {
                let p = ps[k2 % N];
                k2 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_gsl_root_bracket(0, prob, p, lo, hi, 4.0 * eps, &mut e) });
            };
            let mut k3 = 0usize;
            let mut b = || {
                let p = ps[k3 % N];
                k3 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_boost_root_bracket(0, prob, p, lo, hi, 53, &mut e) });
            };
            Cmp { group: "roots", op: "bisect", params: &params_s }.run(
                &mut [&mut a, &mut g, &mut b],
                &["GSL 2.8 bisection", "Boost 1.92 tools::bisect"],
                &ours_n,
                &[
                    format!("{}; test_interval(0, 4ε)", note(gev, gerr)),
                    format!("{}; eps_tolerance(53)", note(bev, berr)),
                ],
                1.0,
            );
        }

        // ---- brent
        {
            let cnt = Cell::new(0usize);
            let (ev, err) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = brent_with(lo, hi, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, &opts)
                .unwrap();
                (x, cnt.get() as f64)
            });
            let ours_n = note(ev, err);
            let (gev, gerr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_gsl_root_bracket(1, prob, p, lo, hi, 4.0 * eps, &mut e) };
                (x, e as f64)
            });
            let (bev, berr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_boost_root_bracket(1, prob, p, lo, hi, 53, &mut e) };
                (x, e as f64)
            });
            // roots crate: absolute eps on x and on |f|.
            let reps = 4.0 * eps * hi.abs().max(1.0);
            let (rev, rerr) = stats(&ps, prob, |p| {
                cnt.set(0);
                let mut c = roots::SimpleConvergency { eps: reps, max_iter: 2000 };
                let x = roots::find_root_brent(lo, hi, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, &mut c)
                .unwrap_or(f64::NAN);
                (x, cnt.get() as f64)
            });
            let (aev, aerr) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = crate::argmin_probe::brent_root(lo, hi, 4.0 * eps, &|x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                });
                (x, cnt.get() as f64)
            });
            let mut k = 0usize;
            let mut a = || {
                let p = ps[k % N];
                k += 1;
                bb(brent_with(lo, hi, |x| f(prob, p, x), &opts));
            };
            let mut k2 = 0usize;
            let mut g = || {
                let p = ps[k2 % N];
                k2 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_gsl_root_bracket(1, prob, p, lo, hi, 4.0 * eps, &mut e) });
            };
            let mut k3 = 0usize;
            let mut b = || {
                let p = ps[k3 % N];
                k3 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_boost_root_bracket(1, prob, p, lo, hi, 53, &mut e) });
            };
            let mut k4 = 0usize;
            let mut r = || {
                let p = ps[k4 % N];
                k4 += 1;
                let mut c = roots::SimpleConvergency { eps: reps, max_iter: 2000 };
                bb(roots::find_root_brent(lo, hi, |x| f(prob, p, x), &mut c).ok());
            };
            let mut k5 = 0usize;
            let mut am = || {
                let p = ps[k5 % N];
                k5 += 1;
                bb(crate::argmin_probe::brent_root(lo, hi, 4.0 * eps, &|x| f(prob, p, x)));
            };
            Cmp { group: "roots", op: "brent", params: &params_s }.run(
                &mut [&mut a, &mut g, &mut b, &mut r, &mut am],
                &[
                    "GSL 2.8 brent",
                    "Boost 1.92 toms748_solve",
                    "roots 0.0.8 find_root_brent",
                    "argmin 0.11 BrentRoot",
                ],
                &ours_n,
                &[
                    format!("{}; test_interval(0, 4ε)", note(gev, gerr)),
                    format!("{}; TOMS 748 (another algorithm), eps_tolerance(53)", note(bev, berr)),
                    format!("{}; stops on |x₁−x₂| or |f| below an absolute 4ε·max(|b|,1): does less", note(rev, rerr)),
                    format!("{}; tol 4ε (absolute+relative per argmin), Executor overhead", note(aev, aerr)),
                ],
                1.0,
            );
        }

        // ---- toms748 (Algorithm 748, the algorithm of Boost's toms748_solve)
        {
            let cnt = Cell::new(0usize);
            let (ev, err) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = toms748_with(lo, hi, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, &opts)
                .unwrap();
                (x, cnt.get() as f64)
            });
            let ours_n = note(ev, err);
            let (bev, berr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_boost_root_bracket(1, prob, p, lo, hi, 53, &mut e) };
                (x, e as f64)
            });
            let mut k = 0usize;
            let mut a = || {
                let p = ps[k % N];
                k += 1;
                bb(toms748_with(lo, hi, |x| f(prob, p, x), &opts));
            };
            let mut k3 = 0usize;
            let mut b = || {
                let p = ps[k3 % N];
                k3 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_boost_root_bracket(1, prob, p, lo, hi, 53, &mut e) });
            };
            Cmp { group: "roots", op: "toms748", params: &params_s }.run(
                &mut [&mut a, &mut b],
                &["Boost 1.92 toms748_solve"],
                &ours_n,
                &[format!("{}; the same algorithm, eps_tolerance(53)", note(bev, berr))],
                1.0,
            );
        }

        if prob > 1 {
            continue; // open methods on the simple roots only
        }
        let x0 = pr.x0;
        // ---- newton
        {
            let cnt = Cell::new(0usize);
            let (ev, err) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = newton_with(x0, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, |x| df(prob, p, x), &opts)
                .unwrap();
                (x, cnt.get() as f64)
            });
            let ours_n = format!("{} (f; as many f′)", note(ev, err));
            let (gev, gerr) = stats(&ps, prob, |p| {
                let (mut e, mut d) = (0i64, 0i64);
                let x = unsafe { probe_gsl_root_fdf(0, prob, p, x0, 4.0 * eps, &mut e, &mut d) };
                (x, e as f64)
            });
            let (bev, berr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_boost_root_open(0, prob, p, x0, lo, hi, 51, &mut e) };
                (x, e as f64)
            });
            let reps = 4.0 * eps * 2.0;
            let (rev, rerr) = stats(&ps, prob, |p| {
                cnt.set(0);
                let mut c = roots::SimpleConvergency { eps: reps, max_iter: 200 };
                let x = roots::find_root_newton_raphson(x0, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, |x| df(prob, p, x), &mut c)
                .unwrap_or(f64::NAN);
                (x, cnt.get() as f64)
            });
            let mut k = 0usize;
            let mut a = || {
                let p = ps[k % N];
                k += 1;
                bb(newton_with(x0, |x| f(prob, p, x), |x| df(prob, p, x), &opts));
            };
            let mut k2 = 0usize;
            let mut g = || {
                let p = ps[k2 % N];
                k2 += 1;
                let (mut e, mut d) = (0i64, 0i64);
                bb(unsafe { probe_gsl_root_fdf(0, prob, p, x0, 4.0 * eps, &mut e, &mut d) });
            };
            let mut k3 = 0usize;
            let mut b = || {
                let p = ps[k3 % N];
                k3 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_boost_root_open(0, prob, p, x0, lo, hi, 51, &mut e) });
            };
            let mut k4 = 0usize;
            let mut r = || {
                let p = ps[k4 % N];
                k4 += 1;
                let mut c = roots::SimpleConvergency { eps: reps, max_iter: 200 };
                bb(roots::find_root_newton_raphson(x0, |x| f(prob, p, x), |x| df(prob, p, x), &mut c).ok());
            };
            let ps_s = format!("{}; 64 instances; x₀ = {x0}; rtol 4ε", pr.name);
            Cmp { group: "roots", op: "newton", params: &ps_s }.run(
                &mut [&mut a, &mut g, &mut b, &mut r],
                &[
                    "GSL 2.8 fdfsolver newton",
                    "Boost 1.92 newton_raphson_iterate",
                    "roots 0.0.8 find_root_newton_raphson",
                ],
                &ours_n,
                &[
                    format!("{}; test_delta(0, 4ε)", note(gev, gerr)),
                    format!("{}; digits 51 (step ≤ 4ε·x), bracketed [lo, hi]", note(bev, berr)),
                    format!("{}; absolute eps 8ε on step or |f|", note(rev, rerr)),
                ],
                1.0,
            );
        }
        // ---- secant
        {
            let cnt = Cell::new(0usize);
            let x1 = x0 + 0.1;
            let (ev, err) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = secant_with(x0, x1, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, &opts)
                .unwrap();
                (x, cnt.get() as f64)
            });
            let ours_n = note(ev, err);
            let (gev, gerr) = stats(&ps, prob, |p| {
                let (mut e, mut d) = (0i64, 0i64);
                let x = unsafe { probe_gsl_root_fdf(1, prob, p, x0, 4.0 * eps, &mut e, &mut d) };
                (x, (e + d) as f64)
            });
            let reps = 4.0 * eps * 2.0;
            let (rev, rerr) = stats(&ps, prob, |p| {
                cnt.set(0);
                let mut c = roots::SimpleConvergency { eps: reps, max_iter: 200 };
                let x = roots::find_root_secant(x0, x1, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, &mut c)
                .unwrap_or(f64::NAN);
                (x, cnt.get() as f64)
            });
            let mut k = 0usize;
            let mut a = || {
                let p = ps[k % N];
                k += 1;
                bb(secant_with(x0, x1, |x| f(prob, p, x), &opts));
            };
            let mut k2 = 0usize;
            let mut g = || {
                let p = ps[k2 % N];
                k2 += 1;
                let (mut e, mut d) = (0i64, 0i64);
                bb(unsafe { probe_gsl_root_fdf(1, prob, p, x0, 4.0 * eps, &mut e, &mut d) });
            };
            let mut k4 = 0usize;
            let mut r = || {
                let p = ps[k4 % N];
                k4 += 1;
                let mut c = roots::SimpleConvergency { eps: reps, max_iter: 200 };
                bb(roots::find_root_secant(x0, x1, |x| f(prob, p, x), &mut c).ok());
            };
            let ps_s = format!("{}; 64 instances; x₀ = {x0}, x₁ = x₀+0.1; rtol 4ε", pr.name);
            Cmp { group: "roots", op: "secant", params: &ps_s }.run(
                &mut [&mut a, &mut g, &mut r],
                &["GSL 2.8 fdfsolver secant", "roots 0.0.8 find_root_secant"],
                &ours_n,
                &[
                    format!("{}; one f′ at the start (counted); test_delta(0, 4ε)", note(gev, gerr)),
                    format!("{}; absolute eps 8ε on step or |f|", note(rev, rerr)),
                ],
                1.0,
            );
        }
        // ---- halley
        {
            let cnt = Cell::new(0usize);
            let (ev, err) = stats(&ps, prob, |p| {
                cnt.set(0);
                let x = halley_with(x0, |x| {
                    cnt.set(cnt.get() + 1);
                    f(prob, p, x)
                }, |x| df(prob, p, x), |x| d2f(prob, p, x), &opts)
                .unwrap();
                (x, cnt.get() as f64)
            });
            let ours_n = format!("{} (f; as many f′, f″)", note(ev, err));
            let (bev, berr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_boost_root_open(1, prob, p, x0, lo, hi, 51, &mut e) };
                (x, e as f64)
            });
            let (sev, serr) = stats(&ps, prob, |p| {
                let mut e = 0i64;
                let x = unsafe { probe_boost_root_open(2, prob, p, x0, lo, hi, 51, &mut e) };
                (x, e as f64)
            });
            let mut k = 0usize;
            let mut a = || {
                let p = ps[k % N];
                k += 1;
                bb(halley_with(x0, |x| f(prob, p, x), |x| df(prob, p, x), |x| d2f(prob, p, x), &opts));
            };
            let mut k3 = 0usize;
            let mut b = || {
                let p = ps[k3 % N];
                k3 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_boost_root_open(1, prob, p, x0, lo, hi, 51, &mut e) });
            };
            let mut k4 = 0usize;
            let mut s = || {
                let p = ps[k4 % N];
                k4 += 1;
                let mut e = 0i64;
                bb(unsafe { probe_boost_root_open(2, prob, p, x0, lo, hi, 51, &mut e) });
            };
            let ps_s = format!("{}; 64 instances; x₀ = {x0}; rtol 4ε", pr.name);
            Cmp { group: "roots", op: "halley", params: &ps_s }.run(
                &mut [&mut a, &mut b, &mut s],
                &["Boost 1.92 halley_iterate", "Boost 1.92 schroder_iterate"],
                &ours_n,
                &[
                    format!("{}; digits 51, bracketed", note(bev, berr)),
                    format!("{}; Schröder's method (same derivatives), digits 51", note(sev, serr)),
                ],
                1.0,
            );
        }
    }
}
