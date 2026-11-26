//! The battery of one-dimensional integrals of `data/battery.txt` (`data/gen_quad.py`): smooth,
//! oscillatory, peaked, singular at an end, singular or broken inside, wide, infinite and
//! divergent, with the exact value of each from mpmath and what `scipy.integrate.quad` (QUADPACK)
//! returns for it at the same tolerances. Every rule is held to the exact values, and `quad` to
//! SciPy's evaluation counts.

use a::math::numerics::{QuadOptions, QuadResult, clenshaw_curtis, quad, quad_g7k15, tanh_sinh};

const DATA: &str = include_str!("data/battery.txt");

/// The integrand of a row, by its name in the data.
pub(super) fn integrand(name: &str) -> fn(f64) -> f64 {
    match name {
        "exp" => |x| x.exp(),
        "four_over_one_plus_x2" => |x| 4.0 / (1.0 + x * x),
        "sin" => |x| x.sin(),
        "x10" => |x| x.powi(10),
        "runge" => |x| 1.0 / (1.0 + 25.0 * x * x),
        "gauss3" => |x| (-x * x).exp(),
        "log1p" => |x| x.ln_1p(),
        "x_exp_x" => |x| x * x.exp(),
        "poly5" => |x| x.powi(5) - 3.0 * x * x,
        "sqrt" => |x| x.sqrt(),
        "sin30" => |x| (30.0 * x).sin(),
        "cos200" => |x| (200.0 * x).cos(),
        "x_sin50" => |x| x * (50.0 * x).sin(),
        "exp_cos1000" => |x| (-x).exp() * (1000.0 * x).cos(),
        "sinc" => |x| x.sin() / x,
        "lorentz" => |x| 1.0 / (1e-4 + (x - 0.3) * (x - 0.3)),
        "gauss_peak" => |x| (-100.0 * (x - 0.5) * (x - 0.5)).exp(),
        "gauss13" => |x| (-(x - 13.0) * (x - 13.0)).exp(),
        "sech2" => |x| {
            let c = (10.0 * (x - 0.2)).cosh();
            1.0 / (c * c)
        },
        "inv_sqrt" => |x| 1.0 / x.sqrt(),
        "log" => |x| x.ln(),
        "x_m09" => |x| x.powf(-0.9),
        "inv_sqrt_right" => |x| 1.0 / (1.0 - x).sqrt(),
        "log_log" => |x| x.ln() * (1.0 - x).ln(),
        "arcsine" => |x| 1.0 / (x * (1.0 - x)).sqrt(),
        "cos_over_sqrt" => |x| x.cos() / x.sqrt(),
        "x_m09_right" => |x| (1.0 - x).powf(-0.9),
        "log_right" => |x| (1.0 - x).ln(),
        "kink" => |x| (x - 1.0 / 3.0).abs(),
        "inner_sqrt" => |x| (x - 0.4).abs().powf(-0.5),
        "step" => |x| if x > 0.7 { 1.0 } else { 0.0 },
        "floor4" => |x| (4.0 * x).floor(),
        "sqrt_abs" => |x| (x - 0.3).abs().sqrt(),
        "exp_decay50" => |x| (-x).exp(),
        "inv_x_wide" => |x| 1.0 / x,
        "exp_half" => |x| (-x).exp(),
        "gauss_line" => |x| (-x * x).exp(),
        "cauchy_line" => |x| 1.0 / (1.0 + x * x),
        "exp_cos_half" => |x| (-x).exp() * x.cos(),
        "power_tail" => |x| x.powf(-1.1),
        "sqrt_cauchy_half" => |x| 1.0 / ((1.0 + x) * x.sqrt()),
        "exp_lower" => |x| x.exp(),
        "x2_exp_half" => |x| x * x * (-x).exp(),
        "abs_tail_line" => |x| (1.0 + x.abs()).powf(-1.5),
        "normal30_line" => {
            |x| (-(x - 30.0) * (x - 30.0) / 2.0).exp() / (2.0 * core::f64::consts::PI).sqrt()
        }
        "gauss30_half" => |x| (-(x - 30.0) * (x - 30.0)).exp(),
        "inv_x_tail" | "inv_x_zero" => |x| 1.0 / x,
        _ => panic!("no integrand {name}"),
    }
}

/// A row of the battery.
pub(super) struct Row {
    pub(super) name: &'static str,
    pub(super) a: f64,
    pub(super) b: f64,
    /// NaN for a divergent integral.
    pub(super) exact: f64,
    scipy_value: f64,
    scipy_evaluations: usize,
    scipy_converged: bool,
}

pub(super) fn rows() -> Vec<Row> {
    DATA.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&'static str> = l.split_whitespace().collect();
            let num = |s: &str| s.parse::<f64>().unwrap();
            Row {
                name: f[0],
                a: num(f[1]),
                b: num(f[2]),
                exact: num(f[3]),
                scipy_value: num(f[4]),
                scipy_evaluations: f[6].parse().unwrap(),
                scipy_converged: f[7] == "0",
            }
        })
        .collect()
}

/// Whether a converged result keeps its promise: its true error within its error estimate (and
/// a few ulps of the value for the rounding of the comparison itself).
fn honest(r: &QuadResult<f64>, exact: f64) -> bool {
    (r.value - exact).abs() <= r.error + 4.0 * f64::EPSILON * exact.abs()
}

#[test]
fn every_rule_is_honest_on_the_battery() {
    // No converged result has a true error above its own estimate, and no divergent integral is
    // reported converged — by any of the four rules, at the default tolerances and at loose ones.
    // (Measured at the default ones: the largest true error over estimate among the converged
    // results is 0.54 for quad, 0.29 for quad_g7k15, 0.16 for clenshaw_curtis and 0.66 for
    // tanh_sinh; at 1e-6, 0.17, 0.34, 0.46 and 0.66.)
    let mut failures = Vec::new();
    for opts in [
        QuadOptions::new(),
        QuadOptions::new().with_epsabs(1e-6).with_epsrel(1e-6),
    ] {
        for row in rows() {
            let f = integrand(row.name);
            let results = [
                ("quad_g10k21", quad_g10k21_with(f, &row, &opts)),
                ("quad_g7k15", quad_g7k15(f, row.a, row.b, &opts)),
                ("clenshaw_curtis", clenshaw_curtis(f, row.a, row.b, &opts)),
                ("tanh_sinh", tanh_sinh(f, row.a, row.b, &opts)),
            ];
            for (rule, r) in results {
                let Some(r) = r else { continue };
                if !r.converged {
                    continue;
                }
                if row.exact.is_nan() {
                    failures.push(format!(
                        "{rule} reported the divergent {} converged at {} ({:e})",
                        row.name, r.value, opts.epsabs
                    ));
                } else if !honest(&r, row.exact) {
                    failures.push(format!(
                        "{rule} on {} at {:e}: error {:e} beyond its estimate {:e}",
                        row.name,
                        opts.epsabs,
                        (r.value - row.exact).abs(),
                        r.error
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn quad_g10k21_with(
    f: fn(f64) -> f64,
    row: &Row,
    opts: &QuadOptions<f64>,
) -> Option<QuadResult<f64>> {
    a::math::numerics::quad_g10k21(f, row.a, row.b, opts)
}

#[test]
fn quad_does_what_scipy_integrate_quad_does() {
    // quad is QUADPACK's QAGS (QAGI on an infinite interval) step for step: on every row of the
    // battery it converges where SciPy converges, to the exact value within its error estimate,
    // and with SciPy's number of evaluations. Measured: the same count on 46 of the 48 rows. The
    // exceptions: (1 − x)^(−0.9), converged within its estimate after 1533 evaluations where SciPy
    // stops at its test for round-off after 3507 — the two part in the epsilon algorithm, which
    // magnifies the last bits of the sequence it extrapolates, and these differ: QUADPACK keeps
    // the sum of its intervals as a running sum, quad sums it afresh (review round 1, B1); and
    // floor4, below. The counts rest on comparisons such roundings tip, and on the integrand's
    // elementary functions, which differ by an ulp from one mathematical library to another: the
    // test asks for 45 of the 48 and for every count within half again of SciPy's — or below,
    // converged and within its estimate, where SciPy gave up.
    //
    // floor4 jumps from 0 to 1 at 1/4, the centre of an interval QAGS bisects, where the value is
    // the right side's. The left half's nearest node, 5.4·10⁻⁴ from the jump, sees 0, and nothing
    // tells a jump at 1/4 from a rise inside that gap; QUADPACK takes the right half's node for
    // the centre's value, quad does not (review round 2, B4: the same exemption let the tail of a
    // peak across the centre go unseen) and bisects towards 1/4 until the most the gap could
    // hold, its width times the value 1, is below the tolerance: 29 more bisections, 1365
    // evaluations where SciPy takes 147. That count is held to converging within its estimate.
    let rows = rows();
    let mut same = 0;
    for row in &rows {
        let jump_at_a_bisection_point = row.name == "floor4";
        let r = quad(integrand(row.name), row.a, row.b).unwrap();
        if row.scipy_converged {
            assert!(r.converged, "{} did not converge where SciPy did", row.name);
            assert!(honest(&r, row.exact), "{}", row.name);
            assert!(
                (r.value - row.scipy_value).abs() <= 2.0 * r.error,
                "{}: {} against SciPy's {}",
                row.name,
                r.value,
                row.scipy_value
            );
        }
        let (got, want) = (r.evaluations as f64, row.scipy_evaluations as f64);
        let sooner = !row.scipy_converged && r.converged && honest(&r, row.exact);
        if jump_at_a_bisection_point {
            assert!(r.converged && honest(&r, row.exact) && r.evaluations <= 1500);
            continue;
        }
        assert!(
            got <= 1.5 * want && (want <= 1.5 * got || sooner),
            "{}: {} evaluations, SciPy {}",
            row.name,
            r.evaluations,
            row.scipy_evaluations
        );
        same += usize::from(r.evaluations == row.scipy_evaluations);
    }
    assert_eq!(rows.len(), 48);
    assert!(same >= 45, "SciPy's count on only {same} of the 48 rows");
}

#[test]
fn each_rule_converges_where_its_method_reaches() {
    // At the default tolerances, every rule converges on every row but those its method cannot
    // reach, listed with the reason; and what converges is within 10⁻¹¹ of the exact value
    // relatively (10⁻¹² is the tolerance).
    let cannot: [(&str, &[&str]); 3] = [
        // 1000 oscillations need more than 200 panels of 15 points; the strong singularity at a
        // nonzero end is beyond QAGS within 200 subintervals, as it is for SciPy.
        ("quad_g7k15", &["exp_cos1000", "x_m09_right"]),
        // The rule evaluates at the ends: an integrand infinite there is beyond it, and an
        // inner singularity is reached only slowly by bisection.
        (
            "clenshaw_curtis",
            &[
                "inv_sqrt",
                "log",
                "x_m09",
                "inv_sqrt_right",
                "log_log",
                "arcsine",
                "cos_over_sqrt",
                "x_m09_right",
                "log_right",
                "inner_sqrt",
            ],
        ),
        // A singularity at a nonzero end is approached only to an ulp (the mass beyond is in
        // the error estimate); the inner kinks, jumps and singularities are not what the rule is
        // for; and an algebraic tail on both sides decays too slowly for sinh-sinh.
        (
            "tanh_sinh",
            &[
                "inv_sqrt_right",
                "arcsine",
                "x_m09_right",
                "kink",
                "inner_sqrt",
                "step",
                "floor4",
                "sqrt_abs",
                "abs_tail_line",
            ],
        ),
    ];
    let opts = QuadOptions::new();
    for row in rows().iter().filter(|r| !r.exact.is_nan()) {
        let f = integrand(row.name);
        let infinite = !row.a.is_finite() || !row.b.is_finite();
        let results = [
            ("quad_g7k15", quad_g7k15(f, row.a, row.b, &opts)),
            ("clenshaw_curtis", clenshaw_curtis(f, row.a, row.b, &opts)),
            ("tanh_sinh", tanh_sinh(f, row.a, row.b, &opts)),
        ];
        for (rule, r) in results {
            let skip = cannot
                .iter()
                .any(|(n, l)| *n == rule && l.contains(&row.name));
            if rule == "clenshaw_curtis" && infinite {
                assert!(r.is_none(), "clenshaw_curtis took an infinite interval");
                continue;
            }
            let r = r.unwrap();
            if skip {
                continue;
            }
            assert!(r.converged, "{rule} did not converge on {}", row.name);
            let rel = (r.value - row.exact).abs() / row.exact.abs().max(1.0);
            assert!(rel < 1e-11, "{rule} on {}: {rel:e}", row.name);
        }
    }
}

#[test]
fn every_rule_integrates_the_battery_in_f32() {
    // The integrands evaluated in f64 and rounded to f32, at the f32 default tolerance 1e-5:
    // what converges is honest to f32 rounding and within 10⁻⁴ of the exact value.
    let opts = QuadOptions::<f32>::new();
    let mut converged = 0;
    let mut failures = Vec::new();
    for row in rows().iter().filter(|r| !r.exact.is_nan()) {
        let g = integrand(row.name);
        let f = |x: f32| g(f64::from(x)) as f32;
        let (a, b) = (row.a as f32, row.b as f32);
        let results = [
            ("quad", quad(f, a, b)),
            ("quad_g7k15", quad_g7k15(f, a, b, &opts)),
            ("clenshaw_curtis", clenshaw_curtis(f, a, b, &opts)),
            ("tanh_sinh", tanh_sinh(f, a, b, &opts)),
        ];
        for (rule, r) in results {
            let Some(r) = r else { continue };
            if !r.converged {
                continue;
            }
            converged += 1;
            let err = (f64::from(r.value) - row.exact).abs();
            let slack = 4.0 * f64::from(f32::EPSILON) * row.exact.abs();
            if err > f64::from(r.error) + slack || err > 1e-4 * row.exact.abs().max(1.0) {
                failures.push(format!(
                    "{rule} on {} in f32: error {err:e}, estimate {:e}",
                    row.name, r.error
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(converged >= 120, "only {converged} f32 results converged");
}
