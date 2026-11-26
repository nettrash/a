//! Interpolation (splines, Chebyshev, barycentric) and differentiation against GSL and Boost.

use crate::harness::{Cmp, Lcg, Row, bb, measure};
use a::math::numerics::{
    Spline, barycentric_interpolate, barycentric_weights, chebyshev, chebyshev_nodes, derivative,
    second_derivative,
};
use std::ffi::c_void;

unsafe extern "C" {
    fn probe_gsl_spline_build(t: i32, n: i32, x: *const f64, y: *const f64) -> *mut c_void;
    fn probe_gsl_spline_free(h: *mut c_void);
    fn probe_gsl_spline_eval(h: *mut c_void, t: *const f64, m: i32, what: i32, accel: i32) -> f64;
    fn probe_gsl_spline_integ(h: *mut c_void, a: f64, b: f64) -> f64;
    fn probe_gsl_cheb_build(order: i32, a: f64, b: f64) -> *mut c_void;
    fn probe_gsl_cheb_free(h: *mut c_void);
    fn probe_gsl_cheb_eval(h: *mut c_void, t: *const f64, m: i32) -> f64;
    fn probe_gsl_cheb_deriv_eval(h: *mut c_void, t: *const f64, m: i32) -> f64;
    fn probe_boost_interp_build(t: i32, n: i32, x: *const f64, y: *const f64, dy: *const f64) -> *mut c_void;
    fn probe_boost_interp_free(t: i32, h: *mut c_void);
    fn probe_boost_interp_eval(t: i32, h: *mut c_void, x: *const f64, m: i32, what: i32) -> f64;
    fn probe_gsl_deriv(kind: i32, k: i32, x: f64, h: f64, abserr: *mut f64, evals: *mut i64) -> f64;
    fn probe_boost_deriv(kind: i32, k: i32, x: f64, err: *mut f64, evals: *mut i64) -> f64;
}

pub fn knots(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 + 0.3 * (1.7 * i as f64).sin()).collect();
    let y: Vec<f64> = x.iter().map(|t| (0.3 * t).sin() + 0.2 * (1.1 * t).cos()).collect();
    (x, y)
}

fn points(x: &[f64], m: usize, sorted: bool) -> Vec<f64> {
    let (a, b) = (x[0], x[x.len() - 1]);
    let mut r = Lcg(12345);
    let mut t: Vec<f64> = (0..m).map(|_| a + (b - a) * r.next()).collect();
    if sorted {
        t.sort_by(|p, q| p.partial_cmp(q).unwrap());
    }
    t
}

pub fn cheb_f(x: f64) -> f64 {
    x.exp() * (3.0 * x).cos()
}

pub fn run() {
    splines();
    cheb();
    bary();
    diff();
}

fn splines() {
    for n in [10usize, 100, 10_000, 1_000_000] {
        let (x, y) = knots(n);
        let params = format!("n = {n} non-uniform knots");
        // construction
        {
            let mut a = || { bb(Spline::cubic(&x, &y)); };
            let mut g = || unsafe {
                let h = probe_gsl_spline_build(0, n as i32, x.as_ptr(), y.as_ptr());
                probe_gsl_spline_free(bb(h));
            };
            let tt = Cmp { group: "interp", op: "Spline::cubic (build)", params: &params }.run(
                &mut [&mut a, &mut g],
                &["GSL 2.8 spline cspline (alloc+init+free)"],
                "natural end conditions",
                &["natural end conditions".into()],
                1.0,
            );
            crate::harness::py_rows("interp", "Spline::cubic (build)", &params, &format!("spline:cubic:{n}:build"), tt[0].median, "", None);
        }
        {
            let mut a = || { bb(Spline::cubic_not_a_knot(&x, &y)); };
            let t = measure(&mut [&mut a]);
            Row { group: "interp".into(), op: "Spline::cubic_not_a_knot (build)".into(), params: params.clone(), ours: t[0].median, reference: "none (no C reference has not-a-knot)".into(), refns: None, notes: String::new() }.print();
            crate::harness::py_rows("interp", "Spline::cubic_not_a_knot (build)", &params, &format!("spline:not_a_knot:{n}:build"), t[0].median, "", None);
            let mut c = || { bb(Spline::cubic_clamped(&x, &y, 0.3, 0.0)); };
            let t = measure(&mut [&mut c]);
            Row { group: "interp".into(), op: "Spline::cubic_clamped (build)".into(), params: params.clone(), ours: t[0].median, reference: "none (no C reference has clamped ends)".into(), refns: None, notes: String::new() }.print();
            crate::harness::py_rows("interp", "Spline::cubic_clamped (build)", &params, &format!("spline:clamped:{n}:build"), t[0].median, "", None);
        }
        if n >= 4 {
            let mut a = || { bb(Spline::akima(&x, &y)); };
            let mut g = || unsafe {
                let h = probe_gsl_spline_build(1, n as i32, x.as_ptr(), y.as_ptr());
                probe_gsl_spline_free(bb(h));
            };
            let mut b = || unsafe {
                let h = probe_boost_interp_build(0, n as i32, x.as_ptr(), y.as_ptr(), std::ptr::null());
                probe_boost_interp_free(0, bb(h));
            };
            let tt = Cmp { group: "interp", op: "Spline::akima (build)", params: &params }.run(
                &mut [&mut a, &mut g, &mut b],
                &["GSL 2.8 spline akima", "Boost 1.92 makima"],
                "Akima 1970",
                &["Akima 1970 (same)".into(), "modified Akima (other weights; copies the data in)".into()],
                1.0,
            );
            crate::harness::py_rows("interp", "Spline::akima (build)", &params, &format!("spline:akima:{n}:build"), tt[0].median, "", None);
            let mut a = || { bb(Spline::pchip(&x, &y)); };
            let mut b = || unsafe {
                let h = probe_boost_interp_build(1, n as i32, x.as_ptr(), y.as_ptr(), std::ptr::null());
                probe_boost_interp_free(1, bb(h));
            };
            let mut g = || unsafe {
                let h = probe_gsl_spline_build(2, n as i32, x.as_ptr(), y.as_ptr());
                probe_gsl_spline_free(bb(h));
            };
            let tt = Cmp { group: "interp", op: "Spline::pchip (build)", params: &params }.run(
                &mut [&mut a, &mut b, &mut g],
                &["Boost 1.92 pchip", "GSL 2.8 spline steffen"],
                "Fritsch–Butland slopes (SciPy's PchipInterpolator rules)",
                &["PCHIP (copies the data in)".into(), "Steffen's monotone method (another rule)".into()],
                1.0,
            );
            crate::harness::py_rows("interp", "Spline::pchip (build)", &params, &format!("spline:pchip:{n}:build"), tt[0].median, "", None);
        }
        // evaluation per point
        if n >= 100 {
            let m = 10_000;
            for sorted in [false, true] {
                if sorted && n != 10_000 {
                    continue;
                }
                let t = points(&x, m, sorted);
                let s = Spline::cubic(&x, &y).unwrap();
                let gh = unsafe { probe_gsl_spline_build(0, n as i32, x.as_ptr(), y.as_ptr()) };
                // agreement
                let mut dmax = 0.0f64;
                for &p in t.iter().take(200) {
                    let g = unsafe { probe_gsl_spline_eval(gh, &p, 1, 0, 1) };
                    dmax = dmax.max((s.evaluate(p).unwrap() - g).abs());
                }
                let mut a = || {
                    let mut acc = 0.0;
                    for &p in &t {
                        acc += s.evaluate(p).unwrap_or(0.0);
                    }
                    bb(acc);
                };
                let mut g1 = || { bb(unsafe { probe_gsl_spline_eval(gh, t.as_ptr(), m as i32, 0, 1) }); };
                let mut g0 = || { bb(unsafe { probe_gsl_spline_eval(gh, t.as_ptr(), m as i32, 0, 0) }); };
                let p2 = format!("{params}; 10⁴ {} points; ns per point", if sorted { "sorted" } else { "random" });
                let tt = Cmp { group: "interp", op: "Spline::evaluate", params: &p2 }.run(
                    &mut [&mut a, &mut g1, &mut g0],
                    &["GSL 2.8 gsl_spline_eval (with accel)", "GSL 2.8 gsl_spline_eval (no accel)"],
                    &format!("natural cubic; agrees with GSL to {dmax:.1e}"),
                    &["interval cache + binary search".into(), "binary search".into()],
                    m as f64,
                );
                if !sorted {
                    crate::harness::py_rows("interp", "Spline::evaluate", &p2, &format!("spline:cubic:{n}:eval"), tt[0].median / m as f64, "10⁴ random points (SciPy: 10⁶ in one call)", None);
                }
                if !sorted {
                    // derivative
                    let mut a = || {
                        let mut acc = 0.0;
                        for &p in &t {
                            acc += s.derivative(p).unwrap_or(0.0);
                        }
                        bb(acc);
                    };
                    let mut g = || { bb(unsafe { probe_gsl_spline_eval(gh, t.as_ptr(), m as i32, 1, 1) }); };
                    let p3 = format!("{params}; 10⁴ random points; ns per point");
                    Cmp { group: "interp", op: "Spline::derivative", params: &p3 }.run(
                        &mut [&mut a, &mut g],
                        &["GSL 2.8 gsl_spline_eval_deriv (with accel)"],
                        "",
                        &[String::new()],
                        m as f64,
                    );
                    // pchip evaluation vs Boost
                    let sp = Spline::pchip(&x, &y).unwrap();
                    let bh = unsafe { probe_boost_interp_build(1, n as i32, x.as_ptr(), y.as_ptr(), std::ptr::null()) };
                    let mut dmax = 0.0f64;
                    for &p in t.iter().take(200) {
                        let b = unsafe { probe_boost_interp_eval(1, bh, &p, 1, 0) };
                        dmax = dmax.max((sp.evaluate(p).unwrap() - b).abs());
                    }
                    let mut a = || {
                        let mut acc = 0.0;
                        for &p in &t {
                            acc += sp.evaluate(p).unwrap_or(0.0);
                        }
                        bb(acc);
                    };
                    let mut b = || { bb(unsafe { probe_boost_interp_eval(1, bh, t.as_ptr(), m as i32, 0) }); };
                    let tt = Cmp { group: "interp", op: "Spline::evaluate (pchip)", params: &p3 }.run(
                        &mut [&mut a, &mut b],
                        &["Boost 1.92 pchip operator()"],
                        &format!("agrees with Boost to {dmax:.1e}"),
                        &[String::new()],
                        m as f64,
                    );
                    crate::harness::py_rows("interp", "Spline::evaluate (pchip)", &p3, &format!("spline:pchip:{n}:eval"), tt[0].median / m as f64, "10⁴ random points (SciPy: 10⁶ in one call)", None);
                    unsafe { probe_boost_interp_free(1, bh) };
                    // integral over the whole range
                    let (lo, hi) = (x[0], x[n - 1]);
                    let gi = unsafe { probe_gsl_spline_integ(gh, lo, hi) };
                    let oi = s.integral(lo, hi).unwrap();
                    let mut a = || { bb(s.integral(lo, hi)); };
                    let mut g = || { bb(unsafe { probe_gsl_spline_integ(gh, lo, hi) }); };
                    Cmp { group: "interp", op: "Spline::integral", params: &format!("{params}; over the whole range") }.run(
                        &mut [&mut a, &mut g],
                        &["GSL 2.8 gsl_spline_eval_integ"],
                        &format!("differs from GSL by {:.1e} relative", ((oi - gi) / gi).abs()),
                        &[String::new()],
                        1.0,
                    );
                }
                unsafe { probe_gsl_spline_free(gh) };
            }
        }
    }
}

fn cheb() {
    let (a, b) = (-1.0, 2.0);
    for deg in [10usize, 100, 1000] {
        let params = format!("e^x·cos 3x on [−1, 2]; degree {deg}");
        let c = chebyshev(cheb_f, a, b, deg).unwrap();
        let gh = unsafe { probe_gsl_cheb_build(deg as i32, a, b) };
        let pts = {
            let mut r = Lcg(99);
            (0..10_000).map(|_| a + (b - a) * r.next()).collect::<Vec<f64>>()
        };
        let mut dmax = 0.0f64;
        for &p in pts.iter().take(500) {
            let g = unsafe { probe_gsl_cheb_eval(gh, &p, 1) };
            dmax = dmax.max((c.evaluate(p) - g).abs());
        }
        let mut fa = || { bb(chebyshev(cheb_f, a, b, deg)); };
        let mut fg = || unsafe {
            let h = probe_gsl_cheb_build(deg as i32, a, b);
            probe_gsl_cheb_free(bb(h));
        };
        let tt = Cmp { group: "interp", op: "chebyshev (build)", params: &params }.run(
            &mut [&mut fa, &mut fg],
            &["GSL 2.8 cheb_init"],
            &format!("deg+1 samples at the first-kind points; values agree with GSL to {dmax:.1e}"),
            &["the same points; O(n²) cosine sums".into()],
            1.0,
        );
        crate::harness::py_rows("interp", "chebyshev (build)", &params, &format!("cheb:{deg}:build"), tt[0].median, "", None);
        if deg <= 100 {
            let m = pts.len();
            let mut fa = || {
                let mut acc = 0.0;
                for &p in &pts {
                    acc += c.evaluate(p);
                }
                bb(acc);
            };
            let mut fg = || { bb(unsafe { probe_gsl_cheb_eval(gh, pts.as_ptr(), m as i32) }); };
            let pe = format!("{params}; 10⁴ random points; ns per point");
            let tt = Cmp { group: "interp", op: "Chebyshev::evaluate", params: &pe }.run(
                &mut [&mut fa, &mut fg],
                &["GSL 2.8 cheb_eval"],
                "Clenshaw",
                &["Clenshaw".into()],
                m as f64,
            );
            crate::harness::py_rows("interp", "Chebyshev::evaluate", &pe, &format!("cheb:{deg}:eval"), tt[0].median / m as f64, "", None);
            let mut fa = || {
                let mut acc = 0.0;
                for &p in &pts {
                    acc += c.derivative(p);
                }
                bb(acc);
            };
            let mut fg = || { bb(unsafe { probe_gsl_cheb_deriv_eval(gh, pts.as_ptr(), m as i32) }); };
            Cmp { group: "interp", op: "Chebyshev::derivative", params: &format!("{params}; 10⁴ random points; ns per point") }.run(
                &mut [&mut fa, &mut fg],
                &["GSL 2.8 cheb_calc_deriv once + cheb_eval"],
                "",
                &["the derivative series formed once per 10⁴ points".into()],
                m as f64,
            );
            let mut fa = || {
                let mut acc = 0.0;
                for &p in &pts {
                    acc += c.integral(p);
                }
                bb(acc);
            };
            let t = measure(&mut [&mut fa]);
            Row { group: "interp".into(), op: "Chebyshev::integral".into(), params: format!("{params}; 10⁴ random points; ns per point"), ours: t[0].median / m as f64, reference: "none (GSL's cheb_calc_integ + cheb_eval would be the same as the derivative row)".into(), refns: None, notes: String::new() }.print();
        }
        unsafe { probe_gsl_cheb_free(gh) };
    }
    for n in [100usize, 10_000] {
        let mut fa = || { bb(chebyshev_nodes(n, -1.0, 2.0)); };
        let t = measure(&mut [&mut fa]);
        Row { group: "interp".into(), op: "chebyshev_nodes".into(), params: format!("n = {n}"), ours: t[0].median, reference: "none".into(), refns: None, notes: "n cosines".into() }.print();
    }
}

fn bary() {
    for n in [100usize, 1000] {
        let x: Vec<f64> = chebyshev_nodes(n, 0.0, 1.0);
        let y: Vec<f64> = x.iter().map(|t: &f64| (3.0 * t).cos()).collect();
        let w = barycentric_weights(&x);
        let mut fa = || { bb(barycentric_weights(&x)); };
        let t = measure(&mut [&mut fa]);
        Row { group: "interp".into(), op: "barycentric_weights".into(), params: format!("n = {n} Chebyshev points"), ours: t[0].median, reference: "none (C)".into(), refns: None, notes: "O(n²), scaled products".into() }.print();
        crate::harness::py_rows("interp", "barycentric_weights", &format!("n = {n} Chebyshev points"), &format!("bary:{n}:weights"), t[0].median, "O(n²), scaled products", None);
        let pts = {
            let mut r = Lcg(5);
            (0..10_000).map(|_| r.next()).collect::<Vec<f64>>()
        };
        let mut fa = || {
            let mut acc = 0.0;
            for &p in &pts {
                acc += barycentric_interpolate(&x, &y, &w, p);
            }
            bb(acc);
        };
        let t = measure(&mut [&mut fa]);
        Row { group: "interp".into(), op: "barycentric_interpolate".into(), params: format!("n = {n}; 10⁴ random points; ns per point"), ours: t[0].median / pts.len() as f64, reference: "none (C)".into(), refns: None, notes: String::new() }.print();
        crate::harness::py_rows("interp", "barycentric_interpolate", &format!("n = {n}; 10⁴ random points; ns per point"), &format!("bary:{n}:eval"), t[0].median / pts.len() as f64, "", None);
    }
}

fn exact_d(k: i32, x: f64) -> f64 {
    match k {
        0 => x.cos(),
        1 => x.exp(),
        2 => 1.0 / x,
        3 => 2.5 * x.powf(1.5),
        _ => 1.0 / (1.0 + x * x),
    }
}
fn dfun(k: i32, x: f64) -> f64 {
    match k {
        0 => x.sin(),
        1 => x.exp(),
        2 => x.ln(),
        3 => x.powf(2.5),
        _ => x.atan(),
    }
}

fn diff() {
    for (k, name, lo, hi) in [
        (0, "sin", 0.5, 1.5),
        (1, "exp", 0.0, 1.0),
        (2, "ln", 1.0, 2.0),
        (3, "x^2.5", 1.0, 2.0),
        (4, "atan", 0.0, 2.0),
    ] {
        let n = 64;
        let xs: Vec<f64> = (0..n).map(|i| lo + (hi - lo) * ((i * 37 % n) as f64 + 0.5) / n as f64).collect();
        let st = |mut f: Box<dyn FnMut(f64) -> (f64, f64) + '_>| {
            let (mut ev, mut err) = (0.0, 0.0f64);
            for &x in &xs {
                let (d, e) = f(x);
                ev += e;
                let r = exact_d(k, x);
                err = err.max(((d - r) / r).abs());
            }
            format!("{:.1} evals, max rel err {err:.1e}", ev / n as f64)
        };
        let cnt = std::cell::Cell::new(0usize);
        let on = st(Box::new(|x| {
            cnt.set(0);
            let d = derivative(|t| { cnt.set(cnt.get() + 1); dfun(k, t) }, x).unwrap_or(f64::NAN);
            (d, cnt.get() as f64)
        }));
        let gn = st(Box::new(|x| { let (mut e, mut ev) = (0.0, 0i64); let d = unsafe { probe_gsl_deriv(0, k, x, 1e-3, &mut e, &mut ev) }; (d, ev as f64) }));
        let b6 = st(Box::new(|x| { let (mut e, mut ev) = (0.0, 0i64); let d = unsafe { probe_boost_deriv(0, k, x, &mut e, &mut ev) }; (d, ev as f64) }));
        let b8 = st(Box::new(|x| { let (mut e, mut ev) = (0.0, 0i64); let d = unsafe { probe_boost_deriv(1, k, x, &mut e, &mut ev) }; (d, ev as f64) }));
        let bc = st(Box::new(|x| { let (mut e, mut ev) = (0.0, 0i64); let d = unsafe { probe_boost_deriv(2, k, x, &mut e, &mut ev) }; (d, ev as f64) }));
        let mut i0 = 0usize;
        let mut fa = || { let x = xs[i0 % n]; i0 += 1; bb(derivative(|t| dfun(k, t), x)); };
        let mut i1 = 0usize;
        let mut fg = || { let x = xs[i1 % n]; i1 += 1; let (mut e, mut ev) = (0.0, 0i64); bb(unsafe { probe_gsl_deriv(0, k, x, 1e-3, &mut e, &mut ev) }); };
        let mut i2 = 0usize;
        let mut f6 = || { let x = xs[i2 % n]; i2 += 1; let (mut e, mut ev) = (0.0, 0i64); bb(unsafe { probe_boost_deriv(0, k, x, &mut e, &mut ev) }); };
        let mut i3 = 0usize;
        let mut f8 = || { let x = xs[i3 % n]; i3 += 1; let (mut e, mut ev) = (0.0, 0i64); bb(unsafe { probe_boost_deriv(1, k, x, &mut e, &mut ev) }); };
        let mut i4 = 0usize;
        let mut fc = || { let x = xs[i4 % n]; i4 += 1; let (mut e, mut ev) = (0.0, 0i64); bb(unsafe { probe_boost_deriv(2, k, x, &mut e, &mut ev) }); };
        let params = format!("{name}′ at 64 points of [{lo}, {hi}]; default tolerance √ε");
        Cmp { group: "diff", op: "derivative", params: &params }.run(
            &mut [&mut fa, &mut fg, &mut f6, &mut f8, &mut fc],
            &["GSL 2.8 deriv_central (h = 10⁻³)", "Boost 1.92 finite_difference_derivative<6>", "Boost 1.92 finite_difference_derivative<8>", "Boost 1.92 complex_step_derivative"],
            &format!("{on}; Richardson extrapolation to its tolerance"),
            &[
                format!("{gn}; 5-point rule with an error estimate (does less: lower accuracy)"),
                format!("{b6}; fixed stencil with an error estimate (does less)"),
                format!("{b8}; fixed stencil with an error estimate (does less)"),
                format!("{bc}; needs f analytic on complex arguments (another method)"),
            ],
            1.0,
        );
        let mut i0 = 0usize;
        let mut fa = || { let x = xs[i0 % n]; i0 += 1; bb(second_derivative(|t| dfun(k, t), x)); };
        let t = measure(&mut [&mut fa]);
        Row { group: "diff".into(), op: "second_derivative".into(), params: format!("{name}″ at 64 points of [{lo}, {hi}]"), ours: t[0].median, reference: "none (GSL and Boost have no second derivative)".into(), refns: None, notes: String::new() }.print();
    }
}
