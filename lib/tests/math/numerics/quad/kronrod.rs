//! The adaptive Kronrod rules, QAGS and QAGI: the tables of the rules against mpmath, the
//! singular ends the bisection homes in on, the infinite intervals, divergent integrals, and the
//! regressions of the audit (n_quad-4, -6 and -10).

use a::math::numerics::{
    QuadOptions, QuadResult, Real, quad, quad_g7k15, quad_g10k21, quad_half_infinite, quad_infinite,
};
use std::cell::Cell;

/// An integrand of the tables of cases.
type F = fn(f64) -> f64;

/// The nodes and weights of the two pairs at 40 digits, from `data/gen_kronrod.py`.
const REFERENCE: &str = include_str!("data/kronrod_ref.txt");
/// The source of the rules, whose literals are the tables.
const SOURCE: &str = include_str!("../../../../src/math/numerics/quad/kronrod.rs");

type Rule = fn(fn(f64) -> f64, f64, f64, &QuadOptions<f64>) -> Option<QuadResult<f64>>;

/// The two adaptive Kronrod rules.
const RULES: [(&str, Rule); 2] = [
    ("quad_g7k15", |f, a, b, o| quad_g7k15(f, a, b, o)),
    ("quad_g10k21", |f, a, b, o| quad_g10k21(f, a, b, o)),
];

/// One node of a table: the abscissa, its Kronrod weight and its Gauss weight if it has one.
type Node = (f64, f64, Option<f64>);

/// The reference table `name`, each number parsed to the nearest `f64`.
fn reference(name: &str) -> Vec<Node> {
    REFERENCE
        .lines()
        .filter(|l| l.starts_with(name) && l[name.len()..].starts_with(' '))
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let g = (f[3] != "-").then(|| f[3].parse().unwrap());
            (f[1].parse().unwrap(), f[2].parse().unwrap(), g)
        })
        .collect()
}

/// The table `name` as the source writes it, `(u, wk, None)` and `(u, wk, Some(wg))`.
fn literals(name: &str) -> Vec<Node> {
    let start = SOURCE
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("no table {name} in the source"));
    let body = &SOURCE[start..];
    let body = &body[body.find("= [").unwrap() + 3..body.find("];").unwrap()];
    let text: String = body
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '_')
        .collect();
    // The entries are the parenthesised groups at the outer level.
    let mut entries = Vec::new();
    let (mut depth, mut from) = (0, 0);
    for (i, c) in text.char_indices() {
        match c {
            '(' => {
                if depth == 0 {
                    from = i + 1;
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    entries.push(&text[from..i]);
                }
            }
            _ => {}
        }
    }
    entries
        .iter()
        .map(|e| {
            let parts: Vec<&str> = e.splitn(3, ',').collect();
            let wg = parts[2].trim_end_matches(',');
            let wg = (wg != "None").then(|| {
                wg.trim_start_matches("Some(")
                    .trim_end_matches(')')
                    .parse()
                    .unwrap()
            });
            (parts[0].parse().unwrap(), parts[1].parse().unwrap(), wg)
        })
        .collect()
}

#[test]
fn the_kronrod_tables_are_the_rules_to_the_last_bit() {
    // gen_kronrod.py computes the pairs from the Stieltjes polynomial in mpmath at 60 digits and
    // checks that each integrates every monomial to degree 3n + 1. Every literal of the source
    // is the correctly rounded f64 of the true value, and the Gauss weights sit on the nodes of
    // the embedded Gauss rule (every other node, the centre only for G7).
    for (name, points) in [("G7K15", 15usize), ("G10K21", 21)] {
        let want = reference(name);
        let got = literals(name);
        assert_eq!(got.len(), want.len(), "{name}: the number of nodes");
        assert_eq!(2 * want.len() - 1, points);
        assert_eq!(got, want, "{name}: the tables differ");
        let gauss = want.iter().filter(|n| n.2.is_some()).count();
        assert_eq!(gauss, (points - 1) / 4 + usize::from(name == "G7K15"));
    }
}

/// The points of one application of a rule to `[-1, 1]`, and its estimate for `f`.
fn one_panel(rule: Rule, f: fn(f64) -> f64) -> QuadResult<f64> {
    let opts = QuadOptions::new().with_max_subintervals(1);
    rule(f, -1.0, 1.0, &opts).unwrap()
}

#[test]
fn the_rules_evaluate_at_the_nodes_of_their_tables_with_their_weights() {
    // Through the public functions: one panel on [−1, 1] evaluates the integrand at exactly the
    // 15 or 21 nodes ±u of the table, and the estimate of an integrand that is 1 at one node
    // and 0 at the others is that node's Kronrod weight, exactly.
    use std::cell::RefCell;
    thread_local! {
        static SEEN: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
        static NODE: RefCell<f64> = const { RefCell::new(0.0) };
    }
    for ((name, rule), table) in RULES.iter().zip(["G7K15", "G10K21"]) {
        let want = reference(table);
        SEEN.with(|s| s.borrow_mut().clear());
        let r = one_panel(*rule, |x| {
            SEEN.with(|s| s.borrow_mut().push(x));
            1.0
        });
        let mut seen = SEEN.with(|s| s.borrow().clone());
        seen.sort_by(f64::total_cmp);
        let mut nodes: Vec<f64> = want.iter().flat_map(|n| [-n.0, n.0]).collect();
        nodes.sort_by(f64::total_cmp);
        nodes.dedup();
        assert_eq!(seen, nodes, "{name}: the points evaluated");
        assert_eq!(r.evaluations, nodes.len());
        // The weights sum to 2.
        assert!(
            (r.value - 2.0).abs() <= 4.0 * f64::EPSILON,
            "{name}: {}",
            r.value
        );
        for &(u, wk, _) in &want {
            NODE.with(|n| *n.borrow_mut() = u);
            let r = one_panel(*rule, |x| {
                if NODE.with(|n| x == *n.borrow()) {
                    1.0
                } else {
                    0.0
                }
            });
            assert_eq!(r.value, wk, "{name}: the weight of the node {u}");
        }
    }
}

#[test]
fn one_panel_has_the_degree_of_its_kronrod_rule() {
    // G7–K15 integrates the monomials to degree 3·7 + 1 = 22 exactly, and G10–K21 to 31; the
    // next even degree is out of reach (by 7e-8 and 2e-11 relatively).
    for ((name, rule), degree) in RULES.iter().zip([22i32, 30]) {
        let f: [fn(f64) -> f64; 2] = if degree == 22 {
            [|x| x.powi(22), |x| x.powi(24)]
        } else {
            [|x| x.powi(30), |x| x.powi(32)]
        };
        let exact = one_panel(*rule, f[0]).value;
        let want = 2.0 / (f64::from(degree) + 1.0);
        assert!(
            (exact - want).abs() <= 8.0 * f64::EPSILON * want,
            "{name}: x^{degree}"
        );
        let beyond = one_panel(*rule, f[1]).value;
        let want = 2.0 / (f64::from(degree) + 3.0);
        assert!(
            (beyond - want).abs() > 1e-12 * want,
            "{name}: x^{}",
            degree + 2
        );
    }
}

/// Whether a result keeps its promise: converged only within its error estimate (and the
/// rounding of the comparison).
fn honest(r: &QuadResult<f64>, exact: f64) -> bool {
    !r.converged || (r.value - exact).abs() <= r.error + 4.0 * f64::EPSILON * exact.abs()
}

#[test]
fn a_singular_end_that_is_not_zero_is_reached_and_never_evaluated() {
    // Audit n_quad-6: the old rule clamped its nodes away from the ends, and on ∫₀¹ (1 − x)^(−½)
    // reported 1.99999998725, error 2e-12, converged, for 2. QAGS reaches it like the end 0.
    let cases: [(&str, F, f64, f64, f64, f64); 6] = [
        (
            "(1 − x)^(−½)",
            |x| 1.0 / (1.0 - x).sqrt(),
            0.0,
            1.0,
            2.0,
            1e-12,
        ),
        ("x^(−½)", |x| 1.0 / x.sqrt(), 0.0, 1.0, 2.0, 1e-12),
        (
            "(x − 1)^(−½) on [1, 2]",
            |x| 1.0 / (x - 1.0).sqrt(),
            1.0,
            2.0,
            2.0,
            1e-12,
        ),
        (
            "(1 − x)^(−0.856)",
            |x| (1.0 - x).powf(-0.856_093_710_216_098_6),
            0.0,
            1.0,
            1.0 / (1.0 - 0.856_093_710_216_098_6),
            1e-8,
        ),
        (
            "|x − ½|^(−½)",
            |x| 1.0 / (x - 0.5).abs().sqrt(),
            0.0,
            1.0,
            2.0 * core::f64::consts::SQRT_2,
            1e-11,
        ),
        ("ln(1 − x)", |x| (1.0 - x).ln(), 0.0, 1.0, -1.0, 1e-12),
    ];
    thread_local! {
        static AT_END: Cell<bool> = const { Cell::new(false) };
    }
    for (what, f, a, b, exact, tol) in cases {
        for (name, rule) in RULES {
            let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
            let r = rule(f, a, b, &opts).unwrap();
            assert!(honest(&r, exact), "{name} on {what}: {r:?}, exact {exact}");
            if tol == 1e-12 && what != "|x − ½|^(−½)" {
                assert!(r.converged, "{name} on {what}: {r:?}");
            }
        }
    }
    // Not one node at an end, however close the bisection comes: an integrand that records an
    // evaluation at an end.
    AT_END.with(|c| c.set(false));
    let watch = |x: f64| {
        if x == 1.0 || x == 0.0 {
            AT_END.with(|c| c.set(true));
        }
        1.0 / ((1.0 - x) * x).sqrt()
    };
    let r = quad(watch, 0.0, 1.0).unwrap();
    assert!(!AT_END.with(Cell::get), "an end was evaluated");
    assert!(honest(&r, core::f64::consts::PI) && r.converged, "{r:?}");
    // The same in f32, where the old rule claimed convergence 3.3e-4 off.
    let r = quad(|x: f32| 1.0 / (1.0 - x).sqrt(), 0.0, 1.0).unwrap();
    let err = (f64::from(r.value) - 2.0).abs();
    assert!(!r.converged || err <= f64::from(r.error) + 1e-6, "{r:?}");
}

/// A deterministic sequence of exponents in `(−0.9, 2)`, the golden-ratio sequence.
fn exponents(count: usize) -> impl Iterator<Item = f64> {
    (1..=count).map(|k| {
        let u = (k as f64 * 0.618_033_988_749_895) % 1.0;
        -0.9 + 2.9 * u
    })
}

#[test]
fn the_error_estimate_holds_on_families_of_singular_ends() {
    // The audit found 34 and 32 of 400 cases of (1 − x)^p converged but wrong at 1e-12; now no
    // converged result is beyond its estimate, at either end, at tight and loose tolerances
    // (measured: 0.39 and 0.36 of it at most here, 0.92 on 400 exponents).
    let count = if cfg!(miri) { 4 } else { 150 };
    for tol in [1e-12, 1e-8] {
        let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
        for p in exponents(count) {
            let exact = 1.0 / (p + 1.0);
            let right = move |x: f64| (1.0 - x).powf(p);
            let left = move |x: f64| x.powf(p);
            for (name, r) in [
                ("g7k15 right", quad_g7k15(right, 0.0, 1.0, &opts)),
                ("g10k21 right", quad_g10k21(right, 0.0, 1.0, &opts)),
                ("g7k15 left", quad_g7k15(left, 0.0, 1.0, &opts)),
                ("g10k21 left", quad_g10k21(left, 0.0, 1.0, &opts)),
            ] {
                let r = r.unwrap();
                if r.converged {
                    let err = (r.value - exact).abs();
                    assert!(
                        err <= r.error + 4.0 * f64::EPSILON * exact,
                        "{name} p = {p} at {tol:e}: error {err:e}, estimate {:e}",
                        r.error
                    );
                }
            }
        }
    }
}

#[test]
fn the_infinite_intervals_see_their_whole_tail() {
    // Audit n_quad-4: the old maps stopped at x ≈ 4.5e15, and the lost tail was in no error
    // estimate: ∫₁^∞ x^(−1.1) dx came out 9.76, converged, and the divergent ∫₁^∞ dx/x 37.3.
    let loose = QuadOptions::new().with_epsabs(1e-6).with_epsrel(1e-6);
    let defaults = QuadOptions::new();
    let pi = core::f64::consts::PI;
    let cases: [(&str, F, f64, f64, f64); 9] = [
        (
            "x^(−1.1) on [1, ∞)",
            |x| x.powf(-1.1),
            1.0,
            f64::INFINITY,
            10.0,
        ),
        (
            "(1 + |x|)^(−1.5) on the line",
            |x| (1.0 + x.abs()).powf(-1.5),
            f64::NEG_INFINITY,
            f64::INFINITY,
            4.0,
        ),
        (
            "1/(√x (1 + x)) on [0, ∞)",
            |x| 1.0 / (x.sqrt() * (1.0 + x)),
            0.0,
            f64::INFINITY,
            pi,
        ),
        (
            "e^(−x) cos x on [0, ∞)",
            |x| (-x).exp() * x.cos(),
            0.0,
            f64::INFINITY,
            0.5,
        ),
        ("eˣ on (−∞, 0]", |x| x.exp(), f64::NEG_INFINITY, 0.0, 1.0),
        (
            "1/(1 + x²) on (−∞, 0]",
            |x| 1.0 / (1.0 + x * x),
            f64::NEG_INFINITY,
            0.0,
            pi / 2.0,
        ),
        (
            "(1 + x)^(−1.01) − ... on [0, ∞)",
            |x| (1.0 + x).powf(-1.01),
            0.0,
            f64::INFINITY,
            100.0,
        ),
        (
            "e^(−x²) on the line",
            |x| (-x * x).exp(),
            f64::NEG_INFINITY,
            f64::INFINITY,
            pi.sqrt(),
        ),
        (
            "sin x / x on [0, ∞)",
            |x| if x == 0.0 { 1.0 } else { x.sin() / x },
            0.0,
            f64::INFINITY,
            pi / 2.0,
        ),
    ];
    for (what, f, a, b, exact) in cases {
        for opts in [&defaults, &loose] {
            for (name, rule) in RULES {
                let r = rule(f, a, b, opts).unwrap();
                assert!(
                    honest(&r, exact),
                    "{name} on {what} at {:e}: {r:?}",
                    opts.epsabs
                );
            }
        }
        // What converges at the defaults: all but the slowly decaying oscillation, which QAGI
        // (QUADPACK's, SciPy's) does not reach, and the tail that falls as x^(−1.01).
        if !what.starts_with("sin") && !what.contains("1.01") {
            let r = quad(f, a, b).unwrap();
            assert!(r.converged, "{what}: {r:?}");
            assert!(
                (r.value - exact).abs() <= 1e-12 * exact.abs().max(1.0),
                "{what}"
            );
        }
    }
    // An integral that diverges is reported, at every tolerance: ∫₁^∞ dx/x, ∫ eˣ over the line.
    for opts in [&defaults, &loose] {
        let r = quad_half_infinite(|x: f64| 1.0 / x, 1.0, opts).unwrap();
        assert!(!r.converged, "∫₁^∞ dx/x: {r:?}");
        let r = quad_infinite(|x: f64| 1.0 / (1.0 + x.abs()), opts).unwrap();
        assert!(!r.converged, "∫ dx/(1 + |x|): {r:?}");
    }
    // And in f32, where the old rule gave 8.22 for 10, converged.
    let r = quad_half_infinite(|x: f32| x.powf(-1.1), 1.0, &QuadOptions::new()).unwrap();
    let err = (f64::from(r.value) - 10.0).abs();
    assert!(!r.converged || err <= f64::from(r.error) + 1e-5, "{r:?}");
}

#[test]
fn a_narrow_feature_far_out_is_not_passed_off_as_a_converged_nothing() {
    // Audit n_quad-10: the normal density about 30 over the line came out 1.35e-13, converged,
    // after the first panel. And at an absolute tolerance of 1e-6, QUADPACK's QAGI (and SciPy's
    // quad) return 5.47e-7 with the error estimate 9.9e-7 for ∫₀^∞ e^(−(x−30)²) dx = √π: an
    // estimate larger than the result is not accepted however loose the tolerance.
    let normal =
        |x: f64| (-(x - 30.0) * (x - 30.0) / 2.0).exp() / (2.0 * core::f64::consts::PI).sqrt();
    let bump = |x: f64| (-(x - 30.0) * (x - 30.0)).exp();
    let sqrt_pi = core::f64::consts::PI.sqrt();
    for tol in [1e-12, 1e-8, 1e-6, 1e-4] {
        let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
        for (name, rule) in RULES {
            let r = rule(normal, f64::NEG_INFINITY, f64::INFINITY, &opts).unwrap();
            assert!(r.converged && honest(&r, 1.0), "{name} at {tol:e}: {r:?}");
            let r = rule(bump, 0.0, f64::INFINITY, &opts).unwrap();
            assert!(
                r.converged && honest(&r, sqrt_pi),
                "{name} at {tol:e}: {r:?}"
            );
        }
    }
}

/// An integrand of a family of cases, borrowed.
type Shape<'a> = &'a dyn Fn(f64) -> f64;

/// One of the two Kronrod rules (`quad_g7k15` if `g7`) on `f` over `[a, b]` in the type `T`, at
/// the tolerance `tol`; the result in `f64`, and whether any value of the integrand the rule
/// evaluated, in `T`, was other than zero.
fn watched<T: Real>(
    g7: bool,
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
) -> (QuadResult<f64>, bool) {
    let seen = Cell::new(false);
    let g = |x: T| {
        let v = T::from_f64(f(x.to_f64()));
        if v != T::from_f64(0.0) {
            seen.set(true);
        }
        v
    };
    let (a, b, tol) = (T::from_f64(a), T::from_f64(b), T::from_f64(tol));
    let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
    let r = if g7 {
        quad_g7k15(g, a, b, &opts)
    } else {
        quad_g10k21(g, a, b, &opts)
    }
    .unwrap();
    let r = QuadResult {
        value: r.value.to_f64(),
        error: r.error.to_f64(),
        evaluations: r.evaluations,
        converged: r.converged,
    };
    (r, seen.get())
}

/// Whether a result in the type `T` keeps its promise: converged only within its error
/// estimate, and the rounding of `T` on the exact value.
fn honest_in<T: Real>(r: &QuadResult<f64>, exact: f64) -> bool {
    !r.converged || (r.value - exact).abs() <= r.error + 4.0 * T::EPSILON.to_f64() * exact.abs()
}

#[test]
fn a_bump_that_a_bisected_interval_saw_is_never_summed_away() {
    // Review round 1, B1. QUADPACK keeps the totals of the errors as running sums, errsum +
    // erro12 − errmax, and they cancel: on ∫₀^∞ e^(−(x−40)²) dx = √π the first panel of QAGI
    // saw the bump at one node, x = 38.3, with an error estimate of 86, its halves saw nothing
    // (1e-56 each), and 86 + 2e-56 − 86 = 0 was taken for the error of the whole: 8.3e-56, error
    // 0, converged (scipy.integrate.quad alike). The totals are summed afresh now, and a sample
    // that the halves of its interval do not account for stays in their error until a node beside
    // it sees a tenth of it. Every centre from 35 to 60, on the half-line and on the line, and the normal
    // density: found, converged and within the estimate, at every tolerance. On the round-1 code
    // 1048 of these 1224 runs in f64, and 332 of the 360 in f32 that saw the bump, were
    // converged beyond their estimate.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let inv = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
    let step = if cfg!(miri) { 12.5 } else { 0.5 };
    let centres = (0..)
        .map(|k| 35.0 + step * f64::from(k))
        .take_while(|c| *c <= 60.0);
    for c in centres {
        let bump = move |x: f64| (-(x - c) * (x - c)).exp();
        let normal = move |x: f64| (-(x - c) * (x - c) / 2.0).exp() * inv;
        let (inf, minf) = (f64::INFINITY, f64::NEG_INFINITY);
        let cases: [(&str, Shape, f64, f64, f64); 3] = [
            ("e^(−(x−c)²) on [0, ∞)", &bump, 0.0, inf, sqrt_pi),
            ("e^(−(x−c)²) on the line", &bump, minf, inf, sqrt_pi),
            ("the N(c, 1) density on the line", &normal, minf, inf, 1.0),
        ];
        for (what, f, a, b, exact) in cases {
            for tol in [1e-12, 1e-9, 1e-6, 1e-3] {
                for g7 in [true, false] {
                    let (r, seen) = watched::<f64>(g7, f, a, b, tol);
                    assert!(
                        seen && r.converged && honest_in::<f64>(&r, exact),
                        "g7 {g7}, c = {c}, {what} at {tol:e}: {r:?}"
                    );
                }
            }
            // In f32 the first panel sees the bump only while e^(−(38.3−c)²) is above half the
            // smallest subnormal f32: from c = 48.5 on (52.75 for the density) every value it
            // evaluates rounds to 0, and nothing tells the rule that anything is there.
            for tol in [1e-5, 1e-3] {
                for g7 in [true, false] {
                    let (r, seen) = watched::<f32>(g7, f, a, b, tol);
                    if seen {
                        assert!(
                            honest_in::<f32>(&r, exact),
                            "f32, g7 {g7}, c = {c}, {what} at {tol:e}: {r:?}"
                        );
                    } else {
                        assert_eq!((r.value, r.error), (0.0, 0.0), "f32, c = {c}, {what}");
                    }
                }
            }
        }
    }
}

#[test]
fn bumps_away_from_the_origin_are_found_or_unseen_but_never_lost() {
    // Review round 1, B1: Gaussian bumps of random centre and width, on [0, ∞) (centre 6w to
    // 100, width w from 0.05 to 5) and on the line (centre −50 to 50, width 0.1 to 10). A
    // converged result is within its error estimate unless no node of the rule saw the bump at
    // all, every value it evaluated exactly 0; the result is then 0 with the error 0, the limit
    // of every rule that samples (scipy.integrate.quad alike), documented. On the round-1 code,
    // 192 of the 432 runs on [0, ∞) that saw the bump, and 160 of the 736 on the line, were
    // converged beyond their estimate.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let count = if cfg!(miri) { 2 } else { 100 };
    let (mut seen_runs, mut seen_converged) = (0usize, 0usize);
    for k in 1..=count {
        let u = (k as f64 * 0.618_033_988_749_895) % 1.0;
        let v = (k as f64 * 0.754_877_666_246_693) % 1.0;
        let w = 0.05 * 100f64.powf(v);
        let c = 6.0 * w + u * (100.0 - 6.0 * w);
        let on_half = move |x: f64| (-((x - c) / w) * ((x - c) / w)).exp();
        let (w2, c2) = (2.0 * w, -50.0 + 100.0 * u);
        let on_line = move |x: f64| (-0.5 * ((x - c2) / w2) * ((x - c2) / w2)).exp();
        let cases: [(Shape, f64, f64); 2] = [
            (&on_half, 0.0, w * sqrt_pi),
            (
                &on_line,
                f64::NEG_INFINITY,
                w2 * (2.0 * core::f64::consts::PI).sqrt(),
            ),
        ];
        for (f, a, exact) in cases {
            for tol in [1e-12, 1e-9, 1e-6, 1e-3] {
                for g7 in [true, false] {
                    let (r, seen) = watched::<f64>(g7, f, a, f64::INFINITY, tol);
                    let at = format!("k = {k}, from {a}, g7 {g7} at {tol:e}: {r:?}");
                    if seen {
                        assert!(honest_in::<f64>(&r, exact), "{at}, exact {exact}");
                        seen_runs += 1;
                        seen_converged += usize::from(r.converged);
                    } else {
                        assert_eq!((r.value, r.error), (0.0, 0.0), "{at}");
                    }
                }
            }
        }
    }
    // And a bump seen is integrated, not merely reported: measured 1146 of 1168.
    assert!(
        seen_converged * 100 >= 95 * seen_runs,
        "{seen_converged} of {seen_runs} converged"
    );
}

#[test]
fn a_vast_budget_is_spent_as_needed_not_reserved() {
    // Review round 1: QAGS reserved its lists for the whole subinterval limit up front, and with
    // budgets of usize::MAX ∫₀¹ √x dx aborted the process ("memory allocation of
    // 3513665537849438408 bytes failed"). The lists grow with the bisection now; every rule
    // takes such a budget and stops where its method does.
    let vast = QuadOptions::new()
        .with_max_subintervals(usize::MAX)
        .with_max_evaluations(usize::MAX);
    for (name, rule) in RULES {
        let r = rule(|x| x.sqrt(), 0.0, 1.0, &vast).unwrap();
        assert!(r.converged && honest(&r, 2.0 / 3.0), "{name}: {r:?}");
        let r = rule(|x| 1.0 / x, 0.0, 1.0, &vast).unwrap();
        assert!(!r.converged, "{name}: ∫₀¹ dx/x {r:?}");
    }
    let r = a::math::numerics::clenshaw_curtis(|x: f64| x.sqrt(), 0.0, 1.0, &vast).unwrap();
    assert!(honest(&r, 2.0 / 3.0), "clenshaw_curtis: {r:?}");
    let r = a::math::numerics::tanh_sinh(|x: f64| x.sqrt(), 0.0, 1.0, &vast).unwrap();
    assert!(r.converged && honest(&r, 2.0 / 3.0), "tanh_sinh: {r:?}");
    let f = |x: &[f64]| x[0] * x[1];
    let r = a::math::numerics::cubature(f, &[0.0; 2], &[1.0; 2], &vast).unwrap();
    assert!(r.converged && honest(&r, 0.25), "cubature: {r:?}");
}

#[test]
fn an_integral_that_cancels_to_zero_still_converges() {
    // The guard above must not refuse an integral that is 0 by cancellation, whose error is at
    // the round-off floor of the rule rather than below the result.
    let two_pi = 2.0 * core::f64::consts::PI;
    let cases: [(&str, F, f64, f64); 4] = [
        ("sin on [0, 2π]", |x| x.sin(), 0.0, two_pi),
        ("x³ on [−1, 1]", |x| x * x * x, -1.0, 1.0),
        (
            "sin 30x · e^(−x²) on the line",
            |x| (30.0 * x).sin() * (-x * x).exp(),
            f64::NEG_INFINITY,
            f64::INFINITY,
        ),
        ("x eˣ on (−∞, 1]", |x| x * x.exp(), f64::NEG_INFINITY, 1.0),
    ];
    for (what, f, a, b) in cases {
        for (name, rule) in RULES {
            let r = rule(f, a, b, &QuadOptions::new()).unwrap();
            assert!(r.converged, "{name} on {what}: {r:?}");
            assert!(
                r.value.abs() <= r.error.max(1e-15),
                "{name} on {what}: {r:?}"
            );
        }
    }
}

#[test]
fn a_divergent_integral_or_round_off_ends_unconverged_and_within_the_budget() {
    // ∫₀¹ dx/x and ∫₋₁¹ dx/x² diverge; an integrand with noise of 1e-13 cannot be integrated
    // to 1e-15, and QUADPACK's round-off detection ends the bisection early.
    let noisy = |x: f64| {
        let bits = (x.to_bits() ^ 0x9e37_79b9_7f4a_7c15).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x + 1e-13 * ((bits >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
    };
    let tight = QuadOptions::new().with_epsabs(1e-15).with_epsrel(1e-15);
    for (name, rule) in RULES {
        let r = rule(|x| 1.0 / x, 0.0, 1.0, &QuadOptions::new()).unwrap();
        assert!(!r.converged, "{name}: ∫₀¹ dx/x {r:?}");
        let r = rule(|x| 1.0 / (x * x), -1.0, 1.0, &QuadOptions::new()).unwrap();
        assert!(!r.converged, "{name}: ∫₋₁¹ dx/x² {r:?}");
        let r = rule(noisy, 0.0, 1.0, &tight).unwrap();
        assert!(!r.converged, "{name}: noise {r:?}");
        assert!((r.value - 0.5).abs() < 1e-12, "{name}: noise {r:?}");
        assert!(r.evaluations < 100_000, "{name}: {}", r.evaluations);
    }
}

#[test]
fn a_first_panel_that_saw_only_the_tail_of_a_peak_is_not_accepted() {
    // Review round 2, R2-B2. Every node of QAGI's first panel on [−3, ∞) sees the tail of a
    // Lorentzian of width 1.3·10⁻⁴ about 77.2, and K and G agree on it to a half: the panel was
    // accepted at 10⁻⁹ as 1.0·10⁻⁹ ± 5.2·10⁻¹⁰ for 4.13·10⁻⁴ (scipy.integrate.quad alike). An
    // estimate whose error is a tenth of it or more is no longer accepted; the run goes on and
    // ends within its estimate or unconverged.
    let (c, w) = (77.21891235287957, 0.00013152740782116954);
    let f = move |x: f64| 1.0 / (1.0 + ((x - c) / w) * ((x - c) / w));
    // data/gen_review2.py
    let exact = 4.1320532250357147e-4;
    for tol in [1e-9, 1e-6] {
        let o = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
        let r = quad_half_infinite(f, -3.0, &o).unwrap();
        assert!(honest(&r, exact) && r.evaluations > 15, "at {tol:e}: {r:?}");
    }
}

#[test]
fn a_peak_across_the_point_of_a_bisection_is_found_or_reported() {
    // Review round 2, R2-B4. A sample at the centre of a bisected interval lies at an end of both
    // halves, and either half used to account for it: for a peak of width 0.001 about −0.0021 on
    // [−1, 1] the left half's nodes find the peak, and the right half's nearest node, 0.0022 in,
    // sees 10⁻⁸ where the centre saw 0.012 — the tail between, 2.6·10⁻⁶, was in no estimate, and
    // quad_g7k15 came out 1.5·10⁻³ off relatively with the error 2.8·10⁻¹⁴. Each half now
    // accounts for the centre itself. Peaks within four widths of the centre of [−1, 1]: the old
    // code had 20 to 122 of 243 runs per rule and width beyond their estimates in f64, and 6 to
    // 48 in f32; now none.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let step = if cfg!(miri) { 16 } else { 1 };
    let mut failures = Vec::new();
    for g7 in [true, false] {
        for w in [3e-4, 1e-3, 3e-3] {
            for k in (-40..=40).step_by(step) {
                let c = f64::from(k) * w / 10.0;
                let f = move |x: f64| (-((x - c) / w) * ((x - c) / w)).exp();
                for tol in [1e-12, 1e-10, 1e-8] {
                    let (r, _) = watched::<f64>(g7, f, -1.0, 1.0, tol);
                    if !honest_in::<f64>(&r, w * sqrt_pi) {
                        failures.push(format!("g7 {g7}, c = {c:e}, w = {w}, tol {tol:e}: {r:?}"));
                    }
                }
                for tol in [1e-5, 1e-3] {
                    let (r, _) = watched::<f32>(g7, f, -1.0, 1.0, tol);
                    if !honest_in::<f32>(&r, w * sqrt_pi) {
                        failures.push(format!(
                            "f32 g7 {g7}, c = {c:e}, w = {w}, tol {tol:e}: {r:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // A jump exactly at the centre is no exception, and costs bisections towards it, not
    // convergence: what the gap beside the jump could hide halves with each.
    let step_at_0 = |x: f64| if x <= 0.0 { 1.0 } else { 0.0 };
    for (name, rule) in RULES {
        let r = rule(step_at_0, -1.0, 1.0, &QuadOptions::new()).unwrap();
        assert!(r.converged && honest(&r, 1.0), "{name}: {r:?}");
    }
}

#[test]
fn every_feature_an_interval_saw_is_kept_not_only_the_heaviest() {
    // Review round 2. An interval kept one witness, the heaviest: the half of [0, 1] in QAGI's
    // variable that held both of two bumps of width 0.83 about 28.5 and 56.9 (the review's
    // bumps.csv id 971) had the evidence of both, kept the first bump's, and lost the second,
    // which a node had seen at 0.09 of its height: 1.463 ± 3.4·10⁻⁵ at 10⁻³ for 2.926.
    let (c, w) = (28.46139236549884, 0.8254181151991499);
    let f = move |x: f64| {
        let (t, u) = ((x - c) / w, (x - 2.0 * c) / w);
        (-t * t).exp() + (-u * u).exp()
    };
    // data/gen_review2.py
    let exact = 2.926_031_033_783_812_2;
    for tol in [1e-3, 1e-6, 1e-9] {
        for g7 in [true, false] {
            let (r, _) = watched::<f64>(g7, f, 0.0, f64::INFINITY, tol);
            assert!(
                r.converged && honest(&r, exact),
                "g7 {g7} at {tol:e}: {r:?}"
            );
        }
    }
}

#[test]
fn every_value_a_bisected_interval_saw_is_offered_to_its_halves() {
    // Round 3. A bisected interval offered its halves as evidence only its largest sample on each
    // side of its centre. On [0, 1] with peaks of width 3·10⁻⁴ about 0.0643 and 0.2499 (the
    // second 0.895 high), [0, 0.5] saw the first at 0.30 of its height and the second at 0.84 at
    // its centre, the largest sample of both its sides: the first peak was offered to no half,
    // the nodes of [0, 0.25] missed it, and quad_g7k15 returned 4.76·10⁻⁴ ± 7.8·10⁻¹⁴ at 10⁻¹²
    // for 1.008·10⁻³. Every sample of a side that the half's nodes do not account for is offered
    // now, the heaviest four kept.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    let cases = [
        // c₁, c₂, h₂, w, tolerances
        (
            0.064_287_671_022_247_71,
            0.249_926_448_514_596_17,
            0.895_439_320_960_384,
            3e-4,
            &[1e-12, 1e-8][..],
        ),
        (
            0.395_425_573_342_116_53,
            0.352_871_454_029_552_1,
            0.271_296_648_691_980_7,
            1e-3,
            &[1e-8][..],
        ),
    ];
    for (c1, c2, h2, w, tols) in cases {
        let f = move |x: f64| {
            let (t, u) = ((x - c1) / w, (x - c2) / w);
            (-t * t).exp() + h2 * (-u * u).exp()
        };
        for &tol in tols {
            let opts = QuadOptions::new().with_epsabs(tol).with_epsrel(tol);
            let r = quad_g7k15(f, 0.0, 1.0, &opts).unwrap();
            assert!(
                honest(&r, w * sqrt_pi * (1.0 + h2)),
                "peaks about {c1} and {c2} at {tol:e}: {r:?}"
            );
        }
    }
    // The confirmation's sweep of pairs of peaks on [0, 1], both in one half or one in each: at
    // 10⁻¹² and 10⁻⁸ no converged run whose nodes saw both peaks above 10⁻³ of their heights is
    // beyond its estimate, where the code of round 2 had 3 (quad_g7k15, up to 6.8·10⁹ times).
    // At 10⁻⁴ some are, at most as many as the documentation states (its Limits: evidence that
    // shrinks below the tolerance before a node finds the peak; 24 of 142 converged runs that
    // saw both peaks for quad_g7k15 and 33 of 246 for quad_g10k21 when this was written).
    let (mut seen, mut failures) = (0, Vec::new());
    for (g7, documented) in [(true, 24), (false, 33)] {
        let name = if g7 { "quad_g7k15" } else { "quad_g10k21" };
        let on_unit = |f: &dyn Fn(f64) -> f64, o: &QuadOptions<f64>| {
            if g7 {
                quad_g7k15(f, 0.0, 1.0, o)
            } else {
                quad_g10k21(f, 0.0, 1.0, o)
            }
            .unwrap()
        };
        let mut loose = 0;
        for split in [false, true] {
            for w in [1e-3, 3e-4] {
                for tol in [1e-12, 1e-8] {
                    let (s, beyond) = super::pair_sweep(&on_unit, w, split, tol);
                    seen += s;
                    failures.extend(beyond.into_iter().map(|b| format!("{name}: {b}")));
                }
                loose += super::pair_sweep(&on_unit, w, split, 1e-4).1.len();
            }
        }
        assert!(
            loose <= documented,
            "{name}: {loose} runs beyond their estimates at 1e-4, the documentation says {documented}"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(seen > 1000, "only {seen} runs saw both peaks");
}

/// The widths, in units in the last place of the larger end, from which the first panel's
/// outermost nodes no longer round onto the ends: `2/(1 − u)` for the outermost node `u` of the
/// rule in `f64`, where the centre's own rounding takes half an ulp of the room, and `1/(1 − u)`
/// in `f32`, whose nodes are formed in `f64` and rounded once.
const NO_END: [(bool, u32, u32); 2] = [(true, 234, 118), (false, 460, 231)];

#[test]
fn no_end_is_evaluated_on_an_interval_as_wide_as_the_documentation_says() {
    // Review round 2, R2-B5. The documentation said 256 ulps for both rules; the first panel of
    // G10–K21 evaluated an end on intervals up to 459 ulps wide in f64. From the widths of
    // NO_END on, no end is evaluated; below, a node may round onto an end or, where the centre
    // itself rounds onto one ([1, 1 + ε]), beyond it, and an integrand infinite or undefined
    // there gives a result that is not a finite number, not converged.
    let widths: u64 = if cfg!(miri) { 40 } else { 600 };
    let f64_ends: [f64; 4] = [1.0, 1.5, 1e6, -3.0];
    for (g7, wide, _) in NO_END {
        for x in f64_ends {
            for k in 1..=widths {
                let b = f64::from_bits(if x > 0.0 {
                    x.to_bits() + k
                } else {
                    x.to_bits() - k
                });
                let (lo, hi) = (x.min(b), x.max(b));
                let ulp =
                    f64::from_bits(hi.abs().max(lo.abs()).to_bits() + 1) - hi.abs().max(lo.abs());
                let k_ulps = ((hi - lo) / ulp).round() as u64;
                let end = Cell::new(false);
                let g = |t: f64| {
                    end.set(end.get() || t <= lo || t >= hi);
                    1.0 / (t - lo).min(hi - t).sqrt()
                };
                let opts = QuadOptions::new();
                let r = if g7 {
                    quad_g7k15(g, lo, hi, &opts)
                } else {
                    quad_g10k21(g, lo, hi, &opts)
                }
                .unwrap();
                if k_ulps >= u64::from(wide) {
                    assert!(
                        !end.get(),
                        "g7 {g7} on [{lo}, {hi}], {k_ulps} ulps, evaluated at or beyond an end"
                    );
                } else if end.get() {
                    assert!(
                        !r.converged && !r.value.is_finite(),
                        "g7 {g7} on [{lo}, {hi}]: {r:?}"
                    );
                }
            }
        }
    }
    let f32_ends: [f32; 3] = [1.0, 1e6, -3.0];
    for (g7, _, wide) in NO_END {
        for x in f32_ends {
            for k in 1..=(widths / 2) as u32 {
                let b = f32::from_bits(if x > 0.0 {
                    x.to_bits() + k
                } else {
                    x.to_bits() - k
                });
                let (lo, hi) = (x.min(b), x.max(b));
                let ulp =
                    f32::from_bits(hi.abs().max(lo.abs()).to_bits() + 1) - hi.abs().max(lo.abs());
                let k_ulps = ((hi - lo) / ulp).round() as u32;
                let end = Cell::new(false);
                let g = |t: f32| {
                    end.set(end.get() || t <= lo || t >= hi);
                    1.0 / (t - lo).min(hi - t).sqrt()
                };
                let opts = QuadOptions::new();
                let r = if g7 {
                    quad_g7k15(g, lo, hi, &opts)
                } else {
                    quad_g10k21(g, lo, hi, &opts)
                }
                .unwrap();
                if k_ulps >= wide {
                    assert!(
                        !end.get(),
                        "f32 g7 {g7} on [{lo}, {hi}], {k_ulps} ulps, evaluated at or beyond an end"
                    );
                } else if end.get() {
                    assert!(
                        !r.converged && !r.value.is_finite(),
                        "f32 g7 {g7} on [{lo}, {hi}]: {r:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn an_integral_too_small_for_normal_numbers_keeps_an_error_estimate() {
    // Review round 2. QUADPACK drops its floor of 50ε ∫|f| on the error where ∫|f| is below
    // tiny/(50ε), and ∫₀^(10⁻³¹⁰) dx came out a subnormal ulp off with the error 0, converged;
    // ∫₀^(10⁻³⁰⁰) dx 1.7·10⁻³¹⁶ off with the estimate 1.1·10⁻³¹⁶. The floor is now 50ε times
    // ∫|f| or the smallest normal number, whichever is larger.
    for b in [1e-310, 3e-320, 1e-300, 2.5e-308] {
        for (name, rule) in RULES {
            let r = rule(|_| 1.0, 0.0, b, &QuadOptions::new()).unwrap();
            assert!(
                r.converged && (r.value - b).abs() <= r.error,
                "{name} on [0, {b:e}]: {r:?}"
            );
        }
    }
}
