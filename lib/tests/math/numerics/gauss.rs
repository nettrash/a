//! The Gaussian rules of `a::math::numerics`: their nodes and weights against mpmath (in
//! `gauss/reference.rs`), the degree of polynomial each integrates exactly, the total weight each
//! is defined to have, their symmetries, the rules of large order and their small weights, and
//! `f32`.

mod reference;

use a::math::numerics::{
    GOLUB_WELSCH_LARGEST, hermite, hermite_scaled, jacobi, laguerre, laguerre_scaled, legendre,
};

const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
const SQRT_PI: f64 = 1.772_453_850_905_516;

/// A sum of `f64`s by Neumaier's compensated summation, so that a sum rule tests the weights and
/// not the order of the additions.
fn sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    s + c
}

#[test]
fn legendre_is_exact_up_to_twice_its_nodes_minus_one() {
    for n in (1..=40usize).chain([59, 60, 61, 64, 99, 100, 101, 128]) {
        let rule = legendre::<f64>(n);
        // Only the even moments are nonzero; the odd ones vanish by the symmetry of the nodes.
        for degree in 0..2 * n {
            let want = if degree % 2 == 0 {
                2.0 / (degree as f64 + 1.0)
            } else {
                0.0
            };
            let got = sum((0..n).map(|i| rule.weights[i] * rule.nodes[i].powi(degree as i32)));
            assert!(
                (got - want).abs() < 4e-16 * want.max(1.0),
                "{n} points miss x^{degree}: {got} against {want}"
            );
        }
    }
}

#[test]
fn legendre_nodes_and_weights_are_what_the_smallest_rules_are() {
    let one = legendre::<f64>(1);
    assert_eq!(one.nodes, [0.0]);
    assert_eq!(one.weights, [2.0]);
    let two = legendre::<f64>(2);
    // 1/√3 correctly rounded (the quotient of the rounded root is an ulp off).
    let inv3: f64 = "0.577350269189625764509148780502".parse().unwrap();
    assert_eq!(two.nodes, [-inv3, inv3]);
    assert_eq!(two.weights, [1.0, 1.0]);
    let three = legendre::<f64>(3);
    assert_eq!(three.nodes[1], 0.0);
    assert!((three.nodes[2] - 0.6f64.sqrt()).abs() <= f64::EPSILON);
    assert!((three.weights[1] - 8.0 / 9.0).abs() <= f64::EPSILON);
    let four = legendre::<f64>(4);
    let (nodes, weights) = four.as_slices();
    assert_eq!((nodes.len(), weights.len()), (4, 4));
}

#[test]
fn the_nodes_are_ordered_symmetric_and_inside_the_interval() {
    for n in [1usize, 2, 3, 9, 20, 60, 61, 100, 101, 257, 1000] {
        // Gauss–Jacobi is Golub and Welsch's at every order, O(n³): up to 257 points here.
        let rules = if n <= 257 {
            vec![legendre::<f64>(n), jacobi::<f64>(n, 0.75, 0.75)]
        } else {
            vec![legendre::<f64>(n)]
        };
        for rule in rules {
            assert_eq!(rule.nodes.len(), n);
            assert_eq!(rule.weights.len(), n);
            assert!(
                rule.nodes.windows(2).all(|w| w[0] < w[1]),
                "{n}: not ascending"
            );
            assert!(
                rule.weights.iter().all(|&w| w > 0.0),
                "{n}: a weight is not positive"
            );
            assert!(
                rule.nodes.iter().all(|&x| x > -1.0 && x < 1.0),
                "{n}: a node has escaped"
            );
            // Symmetric to the last bit, the middle node +0.
            for k in 0..n {
                assert_eq!(rule.nodes[k], -rule.nodes[n - 1 - k], "{n}: not symmetric");
                assert_eq!(rule.weights[k], rule.weights[n - 1 - k]);
            }
            if n % 2 == 1 {
                let mid = rule.nodes[n / 2];
                assert!(
                    mid == 0.0 && mid.is_sign_positive(),
                    "{n}: middle node {mid:e}"
                );
            }
        }
        let total = sum(legendre::<f64>(n).weights);
        assert!(
            (total - 2.0).abs() <= 2.0 * f64::EPSILON,
            "{n}: the weights sum to {total}"
        );
    }
}

#[test]
fn legendre_integrates_smooth_functions_to_the_last_digit() {
    // ∫₋₁¹ eˣ dx = e − 1/e, and a 12-point rule is far more than enough.
    let rule = legendre::<f64>(12);
    let got = rule.integrate(|x: f64| x.exp());
    let want = core::f64::consts::E - 1.0 / core::f64::consts::E;
    assert!((got - want).abs() < 1e-15);
    // ∫₋₁¹ cos(πx/2) dx = 4/π.
    let got = rule.integrate(|x: f64| (FRAC_PI_2 * x).cos());
    assert!((got - 4.0 / core::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn laguerre_weights_the_exponential_decay_exactly_to_degree_two_n_minus_one() {
    for n in [1usize, 2, 5, 12, 40, 59, 60, 61, 99, 100, 101, 128] {
        let rule = laguerre::<f64>(n);
        let total = sum(rule.weights.iter().copied());
        assert!(
            (total - 1.0).abs() <= 2.0 * f64::EPSILON,
            "{n}: the total weight should be 1, is {total}"
        );
        assert!(
            rule.nodes.iter().all(|&x| x > 0.0),
            "{n}: a node is not positive"
        );
        assert!(
            rule.nodes.windows(2).all(|w| w[0] < w[1]),
            "{n}: not ascending"
        );
        // ∫₀^∞ xᵏ e⁻ˣ dx / k! = 1, exactly for k below 2n; xᵏ/k! is formed as a product so that
        // it neither overflows nor underflows, and the highest moments depend on the smallest
        // weights of all.
        for k in 0..2 * n {
            let got = sum((0..n).map(|i| {
                let mut t = rule.weights[i];
                for j in 1..=k {
                    t *= rule.nodes[i] / j as f64;
                }
                t
            }));
            assert!(
                (got - 1.0).abs() < 1e-13,
                "{n} points miss x^{k}/{k}!: {got}"
            );
        }
    }
}

#[test]
fn hermite_weights_the_gaussian_exactly_to_degree_two_n_minus_one() {
    for n in [1usize, 3, 8, 30, 59, 60, 61, 99, 100, 101, 128] {
        let rule = hermite::<f64>(n);
        let total = sum(rule.weights.iter().copied());
        assert!(
            (total - SQRT_PI).abs() <= 4.0 * f64::EPSILON,
            "{n}: the total weight should be √π, is {total}"
        );
        assert!(
            rule.nodes.windows(2).all(|w| w[0] < w[1]),
            "{n}: not ascending"
        );
        // ∫ x²ᵏ e^(−x²) dx = Γ(k + ½), exact for 2k below 2n; x²ᵏ/Γ(k + ½) is formed as the
        // product of x²/(j − ½) and 1/√π.
        for k in 0..n {
            let got = sum((0..n).map(|i| {
                let mut t = rule.weights[i] / SQRT_PI;
                let x2 = rule.nodes[i] * rule.nodes[i];
                for j in 1..=k {
                    t *= x2 / (j as f64 - 0.5);
                }
                t
            }));
            assert!(
                (got - 1.0).abs() < 1e-13,
                "{n} points miss x^{}/Γ({k} + ½): {got}",
                2 * k
            );
        }
    }
}

/// The moments `mₖ = ∫ xᵏ (1 − x)^α (1 + x)^β dx` by their recurrence
/// `(k + 2 + α + β) mₖ₊₁ = (β − α) mₖ + k mₖ₋₁`, which integrating `xᵏ (1 − x²) ω′` by parts
/// gives, from `m₀ = 2^(α+β+1) B(α+1, β+1)`; no term of it cancels another.
fn jacobi_moments(count: usize, alpha: f64, beta: f64) -> Vec<f64> {
    let mut m =
        vec![2f64.powf(alpha + beta + 1.0) * a::math::special::beta(alpha + 1.0, beta + 1.0)];
    m.push((beta - alpha) / (alpha + beta + 2.0) * m[0]);
    for k in 1..count {
        let next = ((beta - alpha) * m[k] + k as f64 * m[k - 1]) / (k as f64 + 2.0 + alpha + beta);
        m.push(next);
    }
    m.truncate(count.max(1));
    m
}

#[test]
fn jacobi_is_exact_against_its_own_weight_for_a_range_of_exponents() {
    let pairs = [
        (1.5, 0.25),
        (-0.5, -0.5),
        (-0.9, 0.5),
        (0.5, -0.75),
        (2.0, 3.0),
        (-0.999, 4.0),
        (7.5, 0.0),
    ];
    for (alpha, beta) in pairs {
        for n in [1usize, 2, 5, 10, 25] {
            let rule = jacobi::<f64>(n, alpha, beta);
            assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]));
            assert!(rule.nodes.iter().all(|&x| x > -1.0 && x < 1.0));
            let moments = jacobi_moments(2 * n, alpha, beta);
            for (k, &want) in moments.iter().enumerate() {
                let got = sum((0..n).map(|i| rule.weights[i] * rule.nodes[i].powi(k as i32)));
                assert!(
                    (got - want).abs() < 1e-14 * moments[0],
                    "({alpha}, {beta}), {n} points, x^{k}: {got} against {want}"
                );
            }
        }
    }
}

#[test]
fn jacobi_reduces_to_legendre_and_to_chebyshev() {
    // α = β = 0 is Legendre, to the last bit.
    for n in [7usize, 64, 150] {
        assert_eq!(jacobi::<f64>(n, 0.0, 0.0), legendre::<f64>(n));
    }
    // α = β = −1/2 is Chebyshev of the first kind: π/n at the zeros of Tₙ.
    let n = 9;
    let cheb = jacobi::<f64>(n, -0.5, -0.5);
    for k in 0..n {
        let want = core::f64::consts::PI / n as f64;
        assert!((cheb.weights[k] - want).abs() <= 2.0 * f64::EPSILON * want);
        let node = (core::f64::consts::PI * (2 * k + 1) as f64 / (2 * n) as f64).cos();
        assert!(
            (cheb.nodes[n - 1 - k] - node).abs() <= 2.0 * f64::EPSILON,
            "node {k}"
        );
    }
    // α = β = 1/2 is Chebyshev of the second kind: the area of the half disc is π/2.
    let total = sum(jacobi::<f64>(6, 0.5, 0.5).weights);
    assert!((total - FRAC_PI_2).abs() <= 2.0 * f64::EPSILON);
}

#[test]
#[should_panic(expected = "number of nodes is zero")]
fn a_rule_with_no_nodes_panics() {
    let _ = legendre::<f64>(0);
}

#[test]
#[should_panic(expected = "greater than −1")]
fn a_jacobi_exponent_of_minus_one_panics() {
    let _ = jacobi::<f64>(3, -1.0, 0.0);
}

#[test]
#[should_panic(expected = "greater than −1")]
fn a_jacobi_exponent_that_is_not_a_number_panics() {
    let _ = jacobi::<f64>(3, 0.0, f64::NAN);
}

#[test]
#[should_panic(expected = "finite and greater than −1")]
fn an_infinite_jacobi_exponent_panics() {
    // Review round 1: α = ∞ passed the test α > −1 and gave NaN nodes and weights silently.
    let _ = jacobi::<f64>(3, f64::INFINITY, 0.0);
}

#[test]
fn jacobi_weights_that_no_digit_is_known_of_are_not_a_number() {
    // Review round 1: for α + β near 2·10³⁰ the logarithms of the total weight cancel
    // completely, and jacobi::<f32>(10, 1e30, 1e30) had weights summing to 1.0, where the true
    // total 2^(α+β+1) B(α+1, β+1) is 1.8e-15. The weights lose about (α + β)/4 ulps; from
    // α + β = 2⁵³ on, where that is half of them, they are NaN. Below, the total weight is right
    // to that loss (mpmath: 1.7724538509e-6 for α = β = 10¹², measured 9.5e-5 off relatively).
    for (a, b) in [(1e30, 1e30), (1e16, 0.0), (2f64.powi(52), 2f64.powi(52))] {
        let rule = jacobi::<f64>(10, a, b);
        assert!(rule.weights.iter().all(|w| w.is_nan()), "({a}, {b})");
    }
    let rule = jacobi::<f32>(10, 1e30, 1e30);
    assert!(rule.weights.iter().all(|w| w.is_nan()));
    let total = sum(jacobi::<f64>(10, 1e12, 1e12).weights);
    let want = 1.772_453_851e-6;
    assert!((total / want - 1.0).abs() < 2e-4, "{total}");
    let total = sum(jacobi::<f64>(10, 1e15, 1e15).weights);
    assert!(
        total.is_finite() && (total / 5.604_991_216e-8 - 1.0).abs() < 0.25,
        "{total}"
    );
}

#[test]
fn hermite_and_laguerre_above_the_eigensolvers_switch_integrate_correctly() {
    // Audit n_quad-5: from 113 points the old weights, taken from divide and conquer's
    // eigenvectors, were noise below about 10⁻³⁵: laguerre(113) gave 7.2e18 for ∫ e^(x/2) e⁻ˣ dx
    // = 2, and hermite(113) 1.0000003 for the moment x⁵⁰ / Γ(25.5).
    for n in [112usize, 113, 128, 200] {
        let l = laguerre::<f64>(n);
        let got = l.integrate(|x: f64| (0.5 * x).exp());
        assert!(
            (got - 2.0).abs() < 1e-13,
            "laguerre({n}): ∫ e^(x/2) e⁻ˣ = {got}"
        );
        let mut moment = 0.0;
        for (x, w) in l.nodes.iter().zip(&l.weights) {
            let mut t = *w;
            for j in 1..=20 {
                t *= x / j as f64;
            }
            moment += t;
        }
        assert!(
            (moment - 1.0).abs() < 1e-13,
            "laguerre({n}): x²⁰/20! = {moment}"
        );
        let h = hermite::<f64>(n);
        let got = sum((0..n).map(|i| {
            let mut t = h.weights[i] / SQRT_PI;
            for j in 1..=25 {
                t *= h.nodes[i] * h.nodes[i] / (j as f64 - 0.5);
            }
            t
        }));
        assert!(
            (got - 1.0).abs() < 1e-13,
            "hermite({n}): x⁵⁰/Γ(25.5) = {got}"
        );
    }
    // The weight of the largest node of hermite(200) is 2.229e-163 (gen_gauss.py), which the old
    // rule gave as 0.
    let w = *hermite::<f64>(200).weights.last().unwrap();
    let want: f64 = "2.22909349628062775773978368313e-163".parse().unwrap();
    assert!((w - want).abs() <= f64::EPSILON * want, "{w:e}");
}

#[test]
fn the_large_rules_keep_their_sum_rules_and_never_give_a_weight_that_is_not_a_number() {
    let orders: &[usize] = if cfg!(miri) {
        &[150]
    } else {
        &[500, 1000, 2000]
    };
    for &n in orders {
        let mut rules = vec![
            ("legendre", legendre::<f64>(n), 2.0),
            ("hermite", hermite::<f64>(n), SQRT_PI),
            ("laguerre", laguerre::<f64>(n), 1.0),
        ];
        // Gauss–Jacobi is Golub and Welsch's at every order, O(n³): 500 points at most here.
        if n <= 500 {
            rules.push((
                "jacobi",
                jacobi::<f64>(n, 0.5, -0.25),
                jacobi_moments(1, 0.5, -0.25)[0],
            ));
        }
        for (name, rule, total) in rules {
            assert!(
                rule.weights.iter().all(|w| w.is_finite() && *w >= 0.0),
                "{name}({n}) has a weight that is not a number or is negative"
            );
            assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]), "{name}({n})");
            let s = sum(rule.weights.iter().copied());
            assert!(
                (s - total).abs() <= 8.0 * f64::EPSILON * total,
                "{name}({n}): the weights sum to {s}, not {total}"
            );
        }
        // Towards the outermost nodes the weights of Hermite's and Laguerre's rules fall
        // monotonically, through the subnormal numbers to zero.
        let h = hermite::<f64>(n);
        assert!(h.weights[n / 2..].windows(2).all(|w| w[1] <= w[0]));
        let l = laguerre::<f64>(n);
        let peak = (0..n)
            .max_by(|&i, &j| l.weights[i].total_cmp(&l.weights[j]))
            .unwrap();
        assert!(l.weights[peak..].windows(2).all(|w| w[1] <= w[0]));
    }
}

#[test]
fn the_scaled_rules_integrate_what_the_plain_ones_cannot() {
    // ∫ sech x dx over the line is π: the integrand needs the outermost nodes of a large rule,
    // whose plain weights underflow.
    let pi = core::f64::consts::PI;
    // The rule of n points reaches out to about √(2n), and the tail beyond it is 4e^(−√(2n)):
    // 4e-6 at 100 points, 7e-15 at 600.
    for n in [600usize, 800] {
        let rule = hermite_scaled::<f64>(n);
        let got = rule.integrate(|x: f64| 1.0 / x.cosh());
        assert!((got - pi).abs() < 1e-14, "hermite_scaled({n}): {got}");
        assert!(rule.weights.iter().all(|w| w.is_normal()));
    }
    // ∫₀^∞ e^(−x/2) dx = 2 by the scaled Laguerre rule: exact once the integrand is written
    // against the weight.
    let rule = laguerre_scaled::<f64>(300);
    let got = rule.integrate(|x: f64| (-0.5 * x).exp());
    assert!((got - 2.0).abs() < 1e-13, "laguerre_scaled: {got}");
    assert!(rule.weights.iter().all(|w| w.is_normal()));
    // The scaled weights are the plain ones times e^(x²) and eˣ where both are in range.
    let (h, hs) = (hermite::<f64>(60), hermite_scaled::<f64>(60));
    for i in 0..60 {
        let want = h.weights[i] * (h.nodes[i] * h.nodes[i]).exp();
        assert!((hs.weights[i] - want).abs() <= 1e-13 * want);
    }
}

#[test]
fn jacobi_with_large_exponents_gives_numbers() {
    // Audit n_quad-16: 2^1201 overflowed before B(601, 601) = 2.3e-363 underflowed, and every
    // weight was NaN; the total weight is 2^1201 B(601, 601) = 0.072314939600975038 (mpmath).
    let rule = jacobi::<f64>(3, 600.0, 600.0);
    assert!(rule.weights.iter().all(|w| w.is_finite() && *w > 0.0));
    let total = sum(rule.weights.iter().copied());
    assert!(
        (total / 0.072_314_939_600_975_04 - 1.0).abs() < 1e-13,
        "{total}"
    );
    assert_eq!(rule.nodes[1], 0.0);
    // A total weight beyond the range of an f64, 2^2001/2001 for α = 2000, β = 0: the weights
    // overflow honestly to infinity rather than to NaN.
    let big = jacobi::<f64>(4, 2000.0, 0.0);
    assert!(big.weights.iter().all(|w| !w.is_nan()));
}

#[test]
fn the_two_methods_meet_at_the_switch() {
    // Golub–Welsch below and at the switch, Newton from the asymptotic guesses above it: the
    // rules on either side are of the same accuracy (the data comparisons say to the last bit),
    // and their sums and highest moments are continuous.
    let m = GOLUB_WELSCH_LARGEST;
    for n in [m - 1, m, m + 1, m + 2] {
        let l = legendre::<f64>(n);
        let h = hermite::<f64>(n);
        let g = laguerre::<f64>(n);
        assert!((sum(l.weights.iter().copied()) - 2.0).abs() <= 2.0 * f64::EPSILON);
        assert!((sum(h.weights.iter().copied()) - SQRT_PI).abs() <= 4.0 * f64::EPSILON);
        assert!((sum(g.weights.iter().copied()) - 1.0).abs() <= 2.0 * f64::EPSILON);
        let top = sum((0..n).map(|i| l.weights[i] * l.nodes[i].powi(2 * n as i32 - 2)));
        let want = 2.0 / (2.0 * n as f64 - 1.0);
        assert!((top - want).abs() < 1e-15, "legendre({n}) x^(2n−2): {top}");
    }
}

#[test]
fn an_f32_rule_is_the_f64_rule_rounded() {
    // Every rule is computed in f64 and rounded once to f32; the weights of f32 underflow from
    // 52 (Hermite) and 27 (Laguerre) points, to subnormal numbers and then to zero.
    for n in [1usize, 5, 26, 27, 51, 52, 60, 61, 100, 101, 300] {
        let pairs = [
            (legendre::<f32>(n), legendre::<f64>(n)),
            (hermite::<f32>(n), hermite::<f64>(n)),
            (laguerre::<f32>(n), laguerre::<f64>(n)),
            (jacobi::<f32>(n, 1.5, -0.25), jacobi::<f64>(n, 1.5, -0.25)),
        ];
        for (single, double) in pairs {
            for i in 0..n {
                assert_eq!(single.nodes[i], double.nodes[i] as f32);
                assert_eq!(single.weights[i], double.weights[i] as f32);
            }
        }
    }
    let h = hermite::<f32>(61);
    assert!(h.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    assert_eq!(h.weights[0], 0.0);
}

#[test]
fn legendre_end_weights_lose_nothing_at_large_orders() {
    // Audit n_quad-15: the old weights 2/((1 − x²)Pₙ′²) at the rounded node lost n² ulps at the
    // outermost nodes (3.4e4 at 1000 points). The weight of the outermost node of 1000 points,
    // from gen_gauss.py, correctly rounded:
    let w = legendre::<f64>(1000).weights[0];
    let want: f64 = "7.41333841643207151747683163123e-6".parse().unwrap();
    assert_eq!(w, want);
}
