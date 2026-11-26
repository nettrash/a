//! Quadrature of one variable, Monte Carlo and the Sobol' sequence, against GSL and Boost.

use crate::harness::{Cmp, bb};
use a::math::numerics::{
    MonteCarloOptions, QuadOptions, Sobol, clenshaw_curtis, monte_carlo, quad, quad_g7k15,
    quad_half_infinite, quad_infinite, quasi_monte_carlo_with, tanh_sinh,
};

unsafe extern "C" {
    fn probe_gsl_quad(
        kind: i32,
        k: i32,
        p: f64,
        a: f64,
        b: f64,
        epsabs: f64,
        epsrel: f64,
        evals: *mut i64,
        err: *mut f64,
        status: *mut i32,
    ) -> f64;
    fn probe_boost_quad(
        kind: i32,
        k: i32,
        p: f64,
        a: f64,
        b: f64,
        tol: f64,
        evals: *mut i64,
        err: *mut f64,
    ) -> f64;
    fn probe_gsl_monte(kind: i32, m: i32, d: i32, n: usize, err: *mut f64) -> f64;
    fn probe_gsl_sobol(d: i32, n: usize, acc: *mut f64);
    fn probe_gsl_rqmc(m: i32, d: i32, n: usize, r: i32, err: *mut f64) -> f64;
}

#[inline]
pub fn integrand(k: i32, p: f64, x: f64) -> f64 {
    match k {
        0 => {
            let u = p * x;
            4.0 / (1.0 + u * u)
        }
        1 => (p * x).ln(),
        2 => 1.0 / (p * x).sqrt(),
        3 => (p * x).powf(-0.9),
        4 => (100.0 * p * x).cos(),
        5 => (-p * x).exp(),
        6 => (-p * x * x).exp(),
        7 => (p * x).powf(-1.1),
        8 => {
            let u = p * x;
            1.0 / (1.0 + u * u)
        }
        _ => {
            let d = x - 0.3;
            1.0 / (d * d + 1e-4 * p)
        }
    }
}

fn exact(k: i32, p: f64) -> f64 {
    use std::f64::consts::PI;
    match k {
        0 => 4.0 * p.atan() / p,
        1 => p.ln() - 1.0,
        2 => 2.0 / p.sqrt(),
        3 => 10.0 * p.powf(-0.9),
        4 => (100.0 * p).sin() / (100.0 * p),
        5 => 1.0 / p,
        6 => (PI / p).sqrt(),
        7 => 10.0 * p.powf(-1.1),
        8 => PI / p,
        _ => {
            let c: f64 = 1e-4 * p;
            let s = c.sqrt();
            ((0.7 / s).atan() + (0.3 / s).atan()) / s
        }
    }
}

const NP: usize = 16;

fn ps() -> Vec<f64> {
    (0..NP).map(|i| 1.0 + 0.1 * ((i * 7 % NP) as f64 + 0.5) / NP as f64).collect()
}

/// Mean evaluations, max relative error, converged count over the instances.
fn st(k: i32, ps: &[f64], mut run: impl FnMut(f64) -> (f64, f64, bool)) -> String {
    let (mut ev, mut err, mut conv) = (0.0, 0.0f64, 0);
    for &p in ps {
        let (v, e, c) = run(p);
        ev += e;
        let x = exact(k, p);
        let r = ((v - x) / x).abs();
        err = if r.is_nan() { f64::INFINITY } else { err.max(r) };
        conv += c as usize;
    }
    format!(
        "{:.0} evals, max rel err {err:.1e}{}{}",
        ev / ps.len() as f64,
        if conv == ps.len() {
            String::new()
        } else {
            format!(", {conv}/{} converged", ps.len())
        },
        // 10⁴ times the requested 10⁻¹² (10⁻⁶ for the loose row) is not the same work.
        if err > 1e-8 { " (does less: misses the tolerance)" } else { "" }
    )
}

// A timed closure cycling through the instances.
fn cyc<'a>(ps: &'a [f64], mut g: impl FnMut(f64) -> (f64, f64, bool) + 'a) -> impl FnMut() + 'a {
    let mut i = 0usize;
    move || {
        let p = ps[i % ps.len()];
        i += 1;
        bb(g(p));
    }
}

/// One finite interval, the integrand fixed at compile time (`K`), as a user's closure is: every
/// side inlines it (ours and Boost's templates; GSL calls it through its function pointer).
fn finite_case<const K: i32>(name: &str) {
    let k = K;
    let ps = ps();
    let tol = 1e-12;
    let opts = QuadOptions::<f64>::new(); // epsabs = epsrel = 1e-12
    let gsl = |kind: i32, k: i32, a: f64, b: f64| {
        move |p: f64| {
            let (mut e, mut er, mut s) = (0i64, 0.0, 0i32);
            let v = unsafe { probe_gsl_quad(kind, k, p, a, b, tol, tol, &mut e, &mut er, &mut s) };
            (v, e as f64, s == 0)
        }
    };
    let boost = |kind: i32, k: i32, a: f64, b: f64| {
        move |p: f64| {
            let (mut e, mut er) = (0i64, 0.0);
            let v = unsafe { probe_boost_quad(kind, k, p, a, b, tol, &mut e, &mut er) };
            (v, e as f64, v.is_finite())
        }
    };
    let ours_q = |k: i32, a: f64, b: f64| {
        move |p: f64| {
            let r = quad(|x: f64| integrand(K, p, x), a, b).unwrap();
            (r.value, r.evaluations as f64, r.converged)
        }
    };
        let (a, b) = (0.0, 1.0);
        let params = format!("{name}; 16 instances p∈[1,1.1]; epsabs = epsrel = 1e-12");
        let singular = matches!(k, 1 | 2 | 3);
        // quad (QAGS on G10–K21) against every integrator.
        {
            let mut names = vec!["GSL 2.8 qags", "GSL 2.8 cquad", "Boost 1.92 gauss_kronrod<21>", "Boost 1.92 tanh_sinh"];
            let mut notes = vec![
                format!("{}; QAGS (qk21, ε-algorithm)", st(k, &ps, gsl(0, k, a, b))),
                format!("{}; doubly adaptive Clenshaw–Curtis", st(k, &ps, gsl(3, k, a, b))),
                format!("{}; adaptive bisection, no extrapolation, tol rel 1e-12", st(k, &ps, boost(1, k, a, b))),
                format!("{}; tol 1e-12", st(k, &ps, boost(2, k, a, b))),
            ];
            let mut f0 = cyc(&ps, ours_q(k, a, b));
            let mut f1 = cyc(&ps, gsl(0, k, a, b));
            let mut f2 = cyc(&ps, gsl(3, k, a, b));
            let mut f3 = cyc(&ps, boost(1, k, a, b));
            let mut f4 = cyc(&ps, boost(2, k, a, b));
            let mut f5 = cyc(&ps, gsl(4, k, a, b));
            let mut fs: Vec<&mut dyn FnMut()> = vec![&mut f0, &mut f1, &mut f2, &mut f3, &mut f4];
            if k == 0 {
                names.push("GSL 2.8 qng");
                notes.push(format!("{}; non-adaptive 21/43/87-point", st(k, &ps, gsl(4, k, a, b))));
                fs.push(&mut f5);
            }
            Cmp { group: "quad", op: "quad", params: &params }.run(
                &mut fs,
                &names,
                &st(k, &ps, ours_q(k, a, b)),
                &notes,
                1.0,
            );
        }
        // quad_g7k15
        {
            let ours = |p: f64| {
                let r = quad_g7k15(|x: f64| integrand(K, p, x), a, b, &opts).unwrap();
                (r.value, r.evaluations as f64, r.converged)
            };
            let mut f0 = cyc(&ps, ours);
            let mut f1 = cyc(&ps, gsl(1, k, a, b));
            let mut f2 = cyc(&ps, boost(0, k, a, b));
            Cmp { group: "quad", op: "quad_g7k15", params: &params }.run(
                &mut [&mut f0, &mut f1, &mut f2],
                &["GSL 2.8 qag GAUSS15", "Boost 1.92 gauss_kronrod<15>"],
                &st(k, &ps, ours),
                &[
                    format!("{}; QAG (no extrapolation)", st(k, &ps, gsl(1, k, a, b))),
                    format!("{}; no extrapolation, tol rel 1e-12", st(k, &ps, boost(0, k, a, b))),
                ],
                1.0,
            );
        }
        // clenshaw_curtis (finite at the ends only)
        if !singular {
            let ours = |p: f64| {
                let r = clenshaw_curtis(|x: f64| integrand(K, p, x), a, b, &opts).unwrap();
                (r.value, r.evaluations as f64, r.converged)
            };
            let mut f0 = cyc(&ps, ours);
            let mut f1 = cyc(&ps, gsl(3, k, a, b));
            let mut f2 = cyc(&ps, gsl(7, k, a, b));
            Cmp { group: "quad", op: "clenshaw_curtis", params: &params }.run(
                &mut [&mut f0, &mut f1, &mut f2],
                &["GSL 2.8 cquad", "GSL 2.8 romberg"],
                &st(k, &ps, ours),
                &[
                    format!("{}; doubly adaptive Clenshaw–Curtis (Gonnet)", st(k, &ps, gsl(3, k, a, b))),
                    format!("{}; Romberg (another rule)", st(k, &ps, gsl(7, k, a, b))),
                ],
                1.0,
            );
        }
        // tanh_sinh
        {
            let ours = |p: f64| {
                let r = tanh_sinh(|x: f64| integrand(K, p, x), a, b, &opts).unwrap();
                (r.value, r.evaluations as f64, r.converged)
            };
            let mut f0 = cyc(&ps, ours);
            let mut f1 = cyc(&ps, boost(2, k, a, b));
            Cmp { group: "quad", op: "tanh_sinh", params: &params }.run(
                &mut [&mut f0, &mut f1],
                &["Boost 1.92 tanh_sinh"],
                &st(k, &ps, ours),
                &[format!("{}; tol 1e-12", st(k, &ps, boost(2, k, a, b)))],
                1.0,
            );
        }
    }

/// One infinite interval, the integrand fixed at compile time.
fn infinite_case<const K: i32>(a: f64, b: f64, name: &str, op: &str) {
    let k = K;
    let ps = ps();
    let tol = 1e-12;
    let opts = QuadOptions::<f64>::new(); // epsabs = epsrel = 1e-12
    let gsl = |kind: i32, k: i32, a: f64, b: f64| {
        move |p: f64| {
            let (mut e, mut er, mut s) = (0i64, 0.0, 0i32);
            let v = unsafe { probe_gsl_quad(kind, k, p, a, b, tol, tol, &mut e, &mut er, &mut s) };
            (v, e as f64, s == 0)
        }
    };
    let boost = |kind: i32, k: i32, a: f64, b: f64| {
        move |p: f64| {
            let (mut e, mut er) = (0i64, 0.0);
            let v = unsafe { probe_boost_quad(kind, k, p, a, b, tol, &mut e, &mut er) };
            (v, e as f64, v.is_finite())
        }
    };
    let ours_q = |k: i32, a: f64, b: f64| {
        move |p: f64| {
            let r = quad(|x: f64| integrand(K, p, x), a, b).unwrap();
            (r.value, r.evaluations as f64, r.converged)
        }
    };
        let params = format!("{name}; 16 instances p∈[1,1.1]; epsabs = epsrel = 1e-12");
        let half = b.is_infinite() && a.is_finite();
        let ours = |p: f64| {
            let r = if half {
                quad_half_infinite(|x: f64| integrand(K, p, x), a, &opts).unwrap()
            } else {
                quad_infinite(|x: f64| integrand(K, p, x), &opts).unwrap()
            };
            (r.value, r.evaluations as f64, r.converged)
        };
        let gk = if half { gsl(5, k, a, b) } else { gsl(6, k, a, b) };
        let gname = if half { "GSL 2.8 qagiu" } else { "GSL 2.8 qagi" };
        let dek = if half { boost(3, k, a, b) } else { boost(4, k, a, b) };
        let dname = if half { "Boost 1.92 exp_sinh" } else { "Boost 1.92 sinh_sinh" };
        {
            let mut f0 = cyc(&ps, ours);
            let mut f1 = cyc(&ps, gk);
            let mut f2 = cyc(&ps, boost(1, k, a, b));
            let mut f3 = cyc(&ps, dek);
            Cmp { group: "quad", op, params: &params }.run(
                &mut [&mut f0, &mut f1, &mut f2, &mut f3],
                &[gname, "Boost 1.92 gauss_kronrod<21>", dname],
                &st(k, &ps, ours),
                &[
                    format!("{}; QUADPACK QAGI (qk15 on the mapped interval)", st(k, &ps, gk)),
                    format!("{}; mapped interval, tol rel 1e-12", st(k, &ps, boost(1, k, a, b))),
                    format!("{}; tol 1e-12", st(k, &ps, dek)),
                ],
                1.0,
            );
        }
        {
            let ours = |p: f64| {
                let r = tanh_sinh(|x: f64| integrand(K, p, x), a, b, &opts).unwrap();
                (r.value, r.evaluations as f64, r.converged)
            };
            let mut f0 = cyc(&ps, ours);
            let mut f1 = cyc(&ps, dek);
            Cmp { group: "quad", op: "tanh_sinh", params: &params }.run(
                &mut [&mut f0, &mut f1],
                &[dname],
                &st(k, &ps, ours),
                &[format!("{}; tol 1e-12", st(k, &ps, dek))],
                1.0,
            );
        }
    }

pub fn run() {
    let ps = ps();
    let tol = 1e-12;
    let opts = QuadOptions::<f64>::new(); // epsabs = epsrel = 1e-12
    let inf = f64::INFINITY;
    // (k, a, b, name)
    let finite: [(i32, &str); 6] = [
        (0, "4/(1+(px)²) on [0,1] (smooth)"),
        (4, "cos(100px) on [0,1] (oscillatory)"),
        (9, "1/((x−0.3)²+10⁻⁴p) on [0,1] (peak)"),
        (1, "ln(px) on [0,1] (log end)"),
        (2, "(px)^(−1/2) on [0,1] (end singularity)"),
        (3, "(px)^(−0.9) on [0,1] (strong end singularity)"),
    ];
    let gsl = |kind: i32, k: i32, a: f64, b: f64| {
        move |p: f64| {
            let (mut e, mut er, mut s) = (0i64, 0.0, 0i32);
            let v = unsafe { probe_gsl_quad(kind, k, p, a, b, tol, tol, &mut e, &mut er, &mut s) };
            (v, e as f64, s == 0)
        }
    };
    let boost = |kind: i32, k: i32, a: f64, b: f64| {
        move |p: f64| {
            let (mut e, mut er) = (0i64, 0.0);
            let v = unsafe { probe_boost_quad(kind, k, p, a, b, tol, &mut e, &mut er) };
            (v, e as f64, v.is_finite())
        }
    };
    let ours_q = |k: i32, a: f64, b: f64| {
        move |p: f64| {
            let r = quad(|x: f64| integrand(k, p, x), a, b).unwrap();
            (r.value, r.evaluations as f64, r.converged)
        }
    };
    for (k, name) in finite {
        match k {
            0 => finite_case::<0>(name),
            1 => finite_case::<1>(name),
            2 => finite_case::<2>(name),
            3 => finite_case::<3>(name),
            4 => finite_case::<4>(name),
            _ => finite_case::<9>(name),
        }
    }
    // Infinite intervals.
    let infinite: [(i32, f64, f64, &str, &str); 4] = [
        (5, 0.0, inf, "e^(−px) on [0,∞)", "quad_half_infinite"),
        (7, 1.0, inf, "(px)^(−1.1) on [1,∞) (slow tail)", "quad_half_infinite"),
        (6, -inf, inf, "e^(−px²) on ℝ", "quad_infinite"),
        (8, -inf, inf, "1/(1+(px)²) on ℝ (slow tails)", "quad_infinite"),
    ];
    for (k, a, b, name, op) in infinite {
        match k {
            5 => infinite_case::<5>(a, b, name, op),
            6 => infinite_case::<6>(a, b, name, op),
            7 => infinite_case::<7>(a, b, name, op),
            _ => infinite_case::<8>(a, b, name, op),
        }
    }
    // quad with a loose tolerance, where the per-call overhead shows.
    {
        let k = 0;
        let o6 = QuadOptions::<f64>::new().with_epsabs(1e-6).with_epsrel(1e-6);
        let ours = |p: f64| {
            let r = a::math::numerics::quad_g10k21(|x: f64| integrand(k, p, x), 0.0, 1.0, &o6).unwrap();
            (r.value, r.evaluations as f64, r.converged)
        };
        let g = move |p: f64| {
            let (mut e, mut er, mut s) = (0i64, 0.0, 0i32);
            let v = unsafe { probe_gsl_quad(0, k, p, 0.0, 1.0, 1e-6, 1e-6, &mut e, &mut er, &mut s) };
            (v, e as f64, s == 0)
        };
        let mut f0 = cyc(&ps, ours);
        let mut f1 = cyc(&ps, g);
        Cmp { group: "quad", op: "quad_g10k21", params: "4/(1+(px)²) on [0,1]; 16 instances; epsabs = epsrel = 1e-6 (one panel)" }.run(
            &mut [&mut f0, &mut f1],
            &["GSL 2.8 qags"],
            &st(k, &ps, ours),
            &[st(k, &ps, g)],
            1.0,
        );
    }

    // ---- Monte Carlo
    let fams: [(i32, &str); 2] = [(0, "∏(xᵢ+½)"), (1, "exp(−Σxᵢ²)")];
    for (m, fname) in fams {
        for d in [2usize, 6, 10] {
            let n = 1usize << 16;
            let f = |x: &[f64]| -> f64 {
                match m {
                    0 => x.iter().map(|t| t + 0.5).product(),
                    _ => (-x.iter().map(|t| t * t).sum::<f64>()).exp(),
                }
            };
            let lo = vec![0.0; d];
            let hi = vec![1.0; d];
            let r = monte_carlo(f, &lo, &hi, n).unwrap();
            let mut ge = 0.0;
            let gv = unsafe { probe_gsl_monte(0, m, d as i32, n, &mut ge) };
            let mut a0 = || {
                bb(monte_carlo(f, &lo, &hi, n));
            };
            let mut g0 = || {
                let mut e = 0.0;
                bb(unsafe { probe_gsl_monte(0, m, d as i32, n, &mut e) });
            };
            let params = format!("{fname} over [0,1]^{d}; n = 2^16 samples; ns per sample");
            Cmp { group: "montecarlo", op: "monte_carlo", params: &params }.run(
                &mut [&mut a0, &mut g0],
                &["GSL 2.8 monte_plain (mt19937)"],
                &format!("{:.6} ± {:.1e} (xoshiro256**)", r.value, r.error),
                &[format!("{gv:.6} ± {ge:.1e}")],
                n as f64,
            );
            // quasi-Monte Carlo, 8 shifted replications
            let qo = MonteCarloOptions::new().with_samples(n).with_replications(8);
            let q = quasi_monte_carlo_with(f, &lo, &hi, &qo).unwrap();
            let mut qe = 0.0;
            let qv = unsafe { probe_gsl_rqmc(m, d as i32, n / 8, 8, &mut qe) };
            let mut a1 = || {
                bb(quasi_monte_carlo_with(f, &lo, &hi, &qo));
            };
            let mut g1 = || {
                let mut e = 0.0;
                bb(unsafe { probe_gsl_rqmc(m, d as i32, n / 8, 8, &mut e) });
            };
            let params = format!("{fname} over [0,1]^{d}; 8 shifted replications of 2^13 Sobol' points; ns per sample");
            Cmp { group: "montecarlo", op: "quasi_monte_carlo", params: &params }.run(
                &mut [&mut a1, &mut g1],
                &["GSL 2.8 qrng_sobol + the same shifted estimator (C driver)"],
                &format!("{:.8} ± {:.1e}; compensated sums", q.value, q.error),
                &[format!("{qv:.8} ± {qe:.1e}; plain sums")],
                n as f64,
            );
        }
    }
    // ---- Sobol' points
    for d in [1usize, 2, 10, 40] {
        let n = 1usize << 16;
        let mut acc = vec![0.0; d];
        let mut a0 = || {
            let mut s = Sobol::new(d).unwrap();
            let mut x = vec![0.0; d];
            let mut sum = 0.0;
            for _ in 0..n {
                s.next_point(&mut x);
                sum += x[0];
            }
            bb(sum);
        };
        let mut g0 = || {
            unsafe { probe_gsl_sobol(d as i32, n, acc.as_mut_ptr()) };
            bb(acc[0]);
        };
        let params = format!("dimension {d}; 2^16 points; ns per point");
        Cmp { group: "montecarlo", op: "Sobol::next_point", params: &params }.run(
            &mut [&mut a0, &mut g0],
            &["GSL 2.8 qrng_sobol"],
            "Joe–Kuo directions, Gray code, 32 bits",
            &["Bratley–Fox directions, Gray code; d ≤ 40".into()],
            n as f64,
        );
    }
}
