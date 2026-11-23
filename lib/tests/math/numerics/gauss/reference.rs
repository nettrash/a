//! The Gaussian rules against the reference nodes and weights of `data/gauss_ref.txt`, computed
//! with mpmath at 60 digits by `data/gen_gauss.py` (Newton's method on the recurrence, each zero's
//! index proved by a Sturm count, the weights as Christoffel numbers): every node and weight in
//! units in the last place of the type, `f64` and `f32`, for orders 1 to 2000 (and in the ignored
//! test, which is for a release build, 5000 and 10,000).

use a::math::numerics::{
    NodesWeights, hermite, hermite_scaled, jacobi, laguerre, laguerre_scaled, legendre,
};
use std::collections::BTreeMap;

const DATA: &str = include_str!("data/gauss_ref.txt");

/// One line of the data: the rule, the index of the node, the node, its weight and (Hermite,
/// Laguerre) its scaled weight.
struct Row {
    family: String,
    alpha: f64,
    beta: f64,
    n: usize,
    k: usize,
    node: f64,
    weight: f64,
    scaled: Option<f64>,
}

fn rows() -> Vec<Row> {
    DATA.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let num = |s: &str| s.parse::<f64>().unwrap();
            if f[0] == "jacobi" {
                Row {
                    family: f[0].into(),
                    alpha: num(f[1]),
                    beta: num(f[2]),
                    n: f[3].parse().unwrap(),
                    k: f[4].parse().unwrap(),
                    node: num(f[5]),
                    weight: num(f[6]),
                    scaled: None,
                }
            } else {
                Row {
                    family: f[0].into(),
                    alpha: 0.0,
                    beta: 0.0,
                    n: f[1].parse().unwrap(),
                    k: f[2].parse().unwrap(),
                    node: num(f[3]),
                    weight: num(f[4]),
                    scaled: f.get(5).map(|s| num(s)),
                }
            }
        })
        .collect()
}

/// The distance from `got` to the correctly rounded `want` in units of the last place of an
/// `f64` at `want` (of the smallest subnormal below the normal range).
fn ulps(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    let unit = if want.abs() < f64::MIN_POSITIVE {
        f64::from_bits(1)
    } else {
        2f64.powi(want.abs().log2().floor() as i32 - 52)
    };
    (got - want).abs() / unit
}

/// The same for `f32`, against the `f64` reference rounded to `f32`.
fn ulps32(got: f32, want: f64) -> f64 {
    let want = want as f32;
    if got == want {
        return 0.0;
    }
    let unit = if want.abs() < f32::MIN_POSITIVE {
        f32::from_bits(1) as f64
    } else {
        2f64.powi((want.abs() as f64).log2().floor() as i32 - 23)
    };
    (got as f64 - want as f64).abs() / unit
}

/// The rule of a row in `f64`, its scaled form where there is one, and in `f32`.
type Rules = (
    NodesWeights<f64>,
    Option<NodesWeights<f64>>,
    NodesWeights<f32>,
);

fn rules(r: &Row) -> Rules {
    match r.family.as_str() {
        "legendre" => (legendre(r.n), None, legendre(r.n)),
        "hermite" => (hermite(r.n), Some(hermite_scaled(r.n)), hermite(r.n)),
        "laguerre" => (laguerre(r.n), Some(laguerre_scaled(r.n)), laguerre(r.n)),
        _ => (
            jacobi(r.n, r.alpha, r.beta),
            None,
            jacobi(r.n, r.alpha as f32, r.beta as f32),
        ),
    }
}

/// The largest error the tests allow for the weights of a row, in ulps: none for Legendre,
/// Hermite and Laguerre, whose weights are double-double Christoffel numbers rounded once, and
/// a little more for the total weight of Jacobi's rules, which goes through `B(α+1, β+1)` (α + 1
/// rounded, 3 ulps measured at α = 0.3) and through logarithms where it leaves the range
/// (`α = β = 600`, 31 measured).
fn weight_bound(r: &Row) -> f64 {
    match (r.family.as_str(), r.alpha) {
        ("jacobi", a) if a >= 600.0 => 64.0,
        ("jacobi", _) => 4.0,
        _ => 0.0,
    }
}

/// Checks every row of orders in `orders`; returns the largest errors seen (nodes, weights,
/// scaled weights, `f32` nodes, `f32` weights).
fn check(orders: impl Fn(usize) -> bool) -> [f64; 5] {
    let mut cache: BTreeMap<String, Rules> = BTreeMap::new();
    let mut worst = [0.0f64; 5];
    for r in rows().iter().filter(|r| orders(r.n)) {
        let key = format!("{} {} {} {}", r.family, r.alpha, r.beta, r.n);
        let (rule, scaled, rule32) = cache.entry(key.clone()).or_insert_with(|| rules(r));
        let at = format!("{key} node {}", r.k);
        // The nodes: correctly rounded, the middle node of a symmetric rule exactly +0.
        let x = rule.nodes[r.k];
        if r.node.abs() < 1e-30 {
            assert!(
                x == 0.0 && x.is_sign_positive(),
                "{at}: the middle node is {x:e}"
            );
        } else {
            let e = ulps(x, r.node);
            assert!(e == 0.0, "{at}: node {x:e} is {e} ulps from {:e}", r.node);
            worst[0] = worst[0].max(e);
        }
        // The weights, a subnormal number or zero where the reference is below the normal
        // range.
        let w = rule.weights[r.k];
        assert!(w.is_finite() && w >= 0.0, "{at}: weight {w:e}");
        let e = ulps(w, r.weight);
        assert!(
            e <= weight_bound(r),
            "{at}: weight {w:e} is {e} ulps from {:e}",
            r.weight
        );
        worst[1] = worst[1].max(e);
        if let (Some(s), Some(want)) = (scaled.as_ref(), r.scaled) {
            // The weight times e^(x²) or eˣ, the exponential of a double-double argument:
            // within 2 ulps measured.
            let e = ulps(s.weights[r.k], want);
            assert!(e <= 2.0, "{at}: scaled weight {e} ulps from {want:e}");
            assert_eq!(s.nodes[r.k], x, "{at}: the scaled rule has other nodes");
            worst[2] = worst[2].max(e);
        }
        // f32, where the parameters are exact in f32 (0.3 and 0.999 are not).
        let exact32 = (r.alpha as f32) as f64 == r.alpha && (r.beta as f32) as f64 == r.beta;
        if exact32 {
            let (x32, w32) = (rule32.nodes[r.k], rule32.weights[r.k]);
            // The f64 rule rounded once: correctly rounded in f32 too, measured on every row.
            if r.node.abs() >= 1e-30 {
                let e = ulps32(x32, r.node);
                assert!(e == 0.0, "{at}: f32 node {x32:e} is {e} ulps off");
                worst[3] = worst[3].max(e);
            }
            let e = ulps32(w32, r.weight);
            assert!(
                e <= weight_bound(r).min(1.0),
                "{at}: f32 weight {w32:e} is {e} ulps off"
            );
            worst[4] = worst[4].max(e);
        }
    }
    worst
}

#[test]
fn every_node_and_weight_up_to_two_thousand_points_is_correctly_rounded() {
    let worst = check(|n| n <= 2000);
    // Every node of every rule and every plain weight of Legendre, Hermite and Laguerre to the
    // last bit, in f64 and f32; the scaled weights within 2 ulps (2 measured).
    assert!(worst[0] == 0.0 && worst[3] == 0.0, "{worst:?}");
}

#[test]
fn the_small_weights_of_the_large_rules_are_right_or_zero_but_never_noise() {
    // The weights of the outermost nodes of Hermite's and Laguerre's rules of 1000 and 2000
    // points are far below the range of an f64: the data holds their true values, which round
    // to zero, and the rules must give zero and not the noise of an eigenvector's components.
    let mut checked = 0;
    for r in rows().iter().filter(|r| r.n >= 1000 && r.n <= 2000) {
        if r.family != "hermite" && r.family != "laguerre" {
            continue;
        }
        let rule = if r.family == "hermite" {
            hermite::<f64>(r.n)
        } else {
            laguerre::<f64>(r.n)
        };
        let w = rule.weights[r.k];
        // A reference that parses to zero is below half the smallest subnormal.
        if r.weight == 0.0 {
            assert_eq!(w, 0.0, "{} {} node {}", r.family, r.n, r.k);
            checked += 1;
        }
    }
    assert!(
        checked >= 8,
        "only {checked} underflowing weights were looked at"
    );
}

#[test]
#[ignore = "five and ten thousand points take minutes in a debug build; run with --release"]
fn the_rules_of_five_and_ten_thousand_points_are_correctly_rounded() {
    let worst = check(|n| n >= 5000);
    assert!(worst[0] == 0.0, "{worst:?}");
}

#[test]
#[ignore = "twenty thousand points take minutes in a debug build; run with --release"]
fn the_largest_orders_cost_newtons_time_not_an_eigenproblems() {
    // Review round 1: nothing pinned the cost of the Newton path above GOLUB_WELSCH_LARGEST,
    // which would have passed every other test with the switch moved to 10⁵. At 20,000 points
    // the dense eigenproblem needs 3.2 GB for its matrix and O(n³) work, hours; Newton's method
    // on the recurrence, O(n²) work and O(n) memory, took 12, 10.5 and 23.6 seconds for the three
    // rules in a release build on the loaded machine of the review.
    let start = std::time::Instant::now();
    let n = 20_000;
    let sizes = [
        legendre::<f64>(n).nodes.len(),
        hermite::<f64>(n).nodes.len(),
        laguerre::<f64>(n).nodes.len(),
    ];
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(sizes, [n; 3]);
    assert!(
        seconds < 300.0,
        "{seconds} s for the three rules of {n} points"
    );
}
