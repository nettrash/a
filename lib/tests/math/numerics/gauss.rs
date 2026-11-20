//! The Gaussian rules of [`a::math::numerics::gauss`]: their nodes and weights, the degree of
//! polynomial each integrates exactly, and the total weight each is defined to have.

use a::math::numerics::{hermite, jacobi, laguerre, legendre};

const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
const SQRT_PI: f64 = 1.772_453_850_905_516_7;

#[test]
fn legendre_is_exact_up_to_twice_its_nodes_minus_one() {
    for n in 1..12usize {
        let rule = legendre::<f64>(n);
        // Only the even moments are nonzero, so only they say anything about the rule.
        for degree in (0..(2 * n)).step_by(2) {
            let want = 2.0 / (degree as f64 + 1.0);
            let got = rule.integrate(|x: f64| x.powi(degree as i32));
            assert!(
                (got - want).abs() < 1e-13 * want.max(1.0),
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
    let inv3 = 1.0 / 3.0f64.sqrt();
    assert!((two.nodes[0] + inv3).abs() < 1e-15);
    assert!((two.nodes[1] - inv3).abs() < 1e-15);
    assert!((two.weights[0] - 1.0).abs() < 1e-15);
    assert!((two.weights[1] - 1.0).abs() < 1e-15);
    let four = legendre::<f64>(4);
    let (nodes, weights) = four.as_slices();
    assert_eq!(nodes.len(), 4);
    assert_eq!(weights.len(), 4);
}

#[test]
fn the_nodes_are_ordered_symmetric_and_inside_the_interval() {
    for n in [1usize, 2, 3, 9, 20, 60, 100] {
        let rule = legendre::<f64>(n);
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
        let total: f64 = rule.weights.iter().sum();
        assert!(
            (total - 2.0).abs() < 1e-13,
            "{n}: the weights sum to {total}"
        );
        for k in 0..n / 2 {
            assert!(
                (rule.nodes[k] + rule.nodes[n - 1 - k]).abs() < 1e-13,
                "{n}: not symmetric"
            );
            assert!((rule.weights[k] - rule.weights[n - 1 - k]).abs() < 1e-13);
        }
    }
}

#[test]
fn legendre_integrates_smooth_functions_to_the_last_digit() {
    // int from -1 to 1 of e^x dx = e - 1/e, and a 12-point rule is far more than enough.
    let rule = legendre::<f64>(12);
    let got = rule.integrate(|x: f64| x.exp());
    let want = core::f64::consts::E - 1.0 / core::f64::consts::E;
    assert!((got - want).abs() < 1e-15);
    // int from -1 to 1 of cos(pi x / 2) dx = 4 / pi.
    let got = rule.integrate(|x: f64| (FRAC_PI_2 * x).cos());
    assert!((got - 4.0 / core::f64::consts::PI).abs() < 1e-14);
}

#[test]
fn laguerre_weights_the_exponential_decay() {
    for n in [1usize, 2, 5, 12, 40] {
        let rule = laguerre::<f64>(n);
        let total: f64 = rule.weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "{n}: the total weight should be 1, is {total}"
        );
        assert!(
            rule.nodes.iter().all(|&x| x >= 0.0),
            "{n}: a node is negative"
        );
        assert!(
            rule.nodes.windows(2).all(|w| w[0] < w[1]),
            "{n}: not ascending"
        );
        // int from 0 to infinity of x^k e^-x dx = k!, exactly for k below 2n.
        for k in 0..2 * n {
            let mut factorial = 1.0;
            for m in 1..=k {
                factorial *= m as f64;
            }
            let got = rule.integrate(|x: f64| x.powi(k as i32));
            assert!(
                (got - factorial).abs() < 1e-8 * factorial.max(1.0),
                "{n} points miss x^{k}: {got} against {factorial}"
            );
        }
    }
}

#[test]
fn hermite_weights_the_gaussian() {
    for n in [1usize, 3, 8, 30] {
        let rule = hermite::<f64>(n);
        let total: f64 = rule.weights.iter().sum();
        assert!(
            (total - SQRT_PI).abs() < 1e-12,
            "{n}: the total weight should be sqrt(pi), is {total}"
        );
        assert!(
            rule.nodes.windows(2).all(|w| w[0] < w[1]),
            "{n}: not ascending"
        );
        // int of x^2k e^-x^2 dx = (2k)! sqrt(pi) / (4^k k!), exact for 2k below 2n.
        for k in 0..n {
            let mut want = SQRT_PI;
            for m in 1..=k {
                want *= (2 * m * (2 * m - 1)) as f64 / (4 * m) as f64;
            }
            let got = rule.integrate(|x: f64| x.powi(2 * k as i32));
            assert!(
                (got - want).abs() < 1e-10 * want.max(1.0),
                "{n} points miss x^{}: {got} against {want}",
                2 * k
            );
        }
    }
}

#[test]
fn jacobi_reduces_to_the_other_three() {
    // alpha = beta = 0 is Legendre.
    let plain = jacobi::<f64>(7, 0.0, 0.0);
    let rule = legendre::<f64>(7);
    for (a, b) in plain.nodes.iter().zip(&rule.nodes) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in plain.weights.iter().zip(&rule.weights) {
        assert!((a - b).abs() < 1e-12);
    }
    // alpha = beta = -1/2 is Chebyshev of the first kind: pi/n on the zeros of T_n.
    let n = 9;
    let cheb = jacobi::<f64>(n, -0.5, -0.5);
    let total: f64 = cheb.weights.iter().sum();
    assert!(
        (total - core::f64::consts::PI).abs() < 1e-12,
        "Chebyshev weights sum to {total}"
    );
    for k in 0..n {
        let want = core::f64::consts::PI / n as f64;
        assert!(
            (cheb.weights[k] - want).abs() < 1e-11,
            "weight {k} is {}",
            cheb.weights[k]
        );
        let node = ((core::f64::consts::PI * (2 * k + 1) as f64) / (2 * n) as f64).cos();
        assert!((cheb.nodes[n - 1 - k] - node).abs() < 1e-13, "node {k}");
    }
    // alpha = beta = 1/2 is Chebyshev of the second kind, against the weight sqrt(1 - x^2).
    let second = jacobi::<f64>(6, 0.5, 0.5);
    let total: f64 = second.weights.iter().sum();
    assert!(
        (total - core::f64::consts::FRAC_PI_2).abs() < 1e-11,
        "the area of the half circle should be pi/2, is {total}"
    );
}

#[test]
fn jacobi_is_exact_against_its_own_weight() {
    let (alpha, beta) = (1.5, 0.25);
    let rule = jacobi::<f64>(10, alpha, beta);
    // The moments of (1-x)^alpha (1+x)^beta, from x = 2t - 1 and the binomial expansion of
    // (2t - 1)^k, which leave a sum of beta functions.
    for k in 0..12usize {
        let want: f64 = (0..=k)
            .map(|j| {
                let sign = if (k - j) % 2 == 0 { 1.0 } else { -1.0 };
                sign * binom(k, j) as f64
                    * 2.0f64.powi(j as i32)
                    * beta_fn(j as f64 + beta + 1.0, alpha + 1.0)
            })
            .sum();
        let want = want * 2.0f64.powf(alpha + beta + 1.0);
        let got = rule.integrate(|x: f64| x.powi(k as i32));
        assert!(
            (got - want).abs() < 1e-10 * want.abs().max(1.0),
            "x^{k}: {got} against {want}"
        );
    }
}

/// The binomial coefficient, by the multiplicative formula.
fn binom(n: usize, k: usize) -> usize {
    let k = k.min(n - k);
    let mut acc = 1usize;
    for i in 0..k {
        acc = acc * (n - i) / (i + 1);
    }
    acc
}

/// Euler's beta function, gamma(a) gamma(b) / gamma(a + b).
fn beta_fn(a: f64, b: f64) -> f64 {
    a::math::special::gamma(a) * a::math::special::gamma(b) / a::math::special::gamma(a + b)
}

#[test]
#[should_panic(expected = "number of nodes is zero")]
fn a_rule_with_no_nodes_panics() {
    let _ = legendre::<f64>(0);
}
