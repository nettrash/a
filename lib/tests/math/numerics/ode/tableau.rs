//! The coefficients of the explicit methods, proved by the order conditions: for every rooted
//! tree `t` up to the order claimed, `Σᵢ bᵢ·Φᵢ(t) = 1/γ(t)` (Butcher's conditions, Hairer, Nørsett
//! and Wanner, section II.2), for the weights, for the embedded formulas and, with `θ^|t|` on the
//! right, for the continuous extensions — and the failure of the same conditions one order up,
//! so that the order is the one claimed and not more. The DOP853 numbers were checked the same
//! way at 60 digits from the published decimals (`scratch/order_conditions.py` of the task); in
//! `f64` the residuals are a few units in the last place of the terms they cancel.

use a::math::numerics::{ButcherTableau, DormandPrince5, DormandPrince853, Rk4};

/// The rooted trees up to some order, each as the indices of its subtrees in the same list, in
/// non-decreasing order, so that each tree is listed once.
struct Trees {
    children: Vec<Vec<usize>>,
    order: Vec<usize>,
}

impl Trees {
    fn up_to(max: usize) -> Self {
        let mut trees = Trees {
            children: vec![Vec::new()],
            order: vec![1],
        };
        for n in 2..=max {
            let smaller = trees.children.len();
            let mut found = Vec::new();
            extend(
                n - 1,
                0,
                &mut Vec::new(),
                &trees.order[..smaller],
                &mut found,
            );
            for children in found {
                trees.children.push(children);
                trees.order.push(n);
            }
        }
        trees
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    /// `γ(t) = |t| · Π γ(subtrees)`.
    fn gamma(&self, t: usize) -> f64 {
        let mut g = self.order[t] as f64;
        for &c in &self.children[t] {
            g *= self.gamma(c);
        }
        g
    }

    /// The elementary weights `Φᵢ(t)` of every tree, for the stage coefficients `a` of `s`
    /// stages: `Φᵢ(•) = 1` and `Φᵢ([t₁ … tₘ]) = Π_k Σⱼ aᵢⱼ·Φⱼ(t_k)`.
    fn weights(&self, a: &[&[f64]], s: usize) -> Vec<Vec<f64>> {
        let mut phi: Vec<Vec<f64>> = Vec::with_capacity(self.len());
        for t in 0..self.len() {
            let mut v = vec![1.0; s];
            for &c in &self.children[t] {
                for (i, vi) in v.iter_mut().enumerate() {
                    let row = a[i];
                    let sum: f64 = row.iter().zip(&phi[c]).map(|(x, p)| x * p).sum();
                    *vi *= sum;
                }
            }
            phi.push(v);
        }
        phi
    }
}

/// Every multiset of the trees `order` lists, given as non-decreasing indices from `min`, whose
/// orders add up to `rest`, appended to `current` and collected in `out`.
fn extend(
    rest: usize,
    min: usize,
    current: &mut Vec<usize>,
    order: &[usize],
    out: &mut Vec<Vec<usize>>,
) {
    if rest == 0 {
        out.push(current.clone());
        return;
    }
    for (idx, &o) in order.iter().enumerate().skip(min) {
        if o <= rest {
            current.push(idx);
            extend(rest - o, idx, current, order, out);
            current.pop();
        }
    }
}

/// The largest residual `|Σᵢ wᵢ·Φᵢ(t) − scale(t)/γ(t)|` over the trees of order `orders`, against
/// the size `Σᵢ |wᵢ·Φᵢ(t)|` of the terms it cancels.
fn residual(
    trees: &Trees,
    phi: &[Vec<f64>],
    w: &[f64],
    orders: core::ops::RangeInclusive<usize>,
    scale: impl Fn(usize) -> f64,
) -> (f64, f64) {
    let (mut worst, mut size) = (0.0f64, 0.0f64);
    for (t, weights) in phi.iter().enumerate() {
        if !orders.contains(&trees.order[t]) {
            continue;
        }
        let lhs: f64 = w.iter().zip(weights).map(|(x, p)| x * p).sum();
        let terms: f64 = w.iter().zip(weights).map(|(x, p)| (x * p).abs()).sum();
        worst = worst.max((lhs - scale(trees.order[t]) / trees.gamma(t)).abs());
        size = size.max(terms);
    }
    (worst, size)
}

/// The stage coefficients as `s` full rows, the rows of the tableau padded with zeros.
fn rows(tableau: &ButcherTableau, s: usize) -> Vec<Vec<f64>> {
    (0..s)
        .map(|i| {
            let mut r = tableau.a[i].to_vec();
            r.resize(s, 0.0);
            r
        })
        .collect()
}

fn as_slices(rows: &[Vec<f64>]) -> Vec<&[f64]> {
    rows.iter().map(|r| r.as_slice()).collect()
}

#[test]
fn the_rooted_trees_are_counted_right() {
    // 1, 1, 2, 4, 9, 20, 48, 115, 286: OEIS A000081.
    let trees = Trees::up_to(9);
    let mut count = [0usize; 10];
    for &o in &trees.order {
        count[o] += 1;
    }
    assert_eq!(&count[1..], &[1, 1, 2, 4, 9, 20, 48, 115, 286]);
    // γ of the bushy tree of order n is n, of the tall one n!.
    assert_eq!(trees.gamma(0), 1.0);
    assert_eq!(trees.gamma(1), 2.0);
}

#[test]
fn every_row_of_the_stage_coefficients_sums_to_its_abscissa() {
    let tableaux = [
        Rk4::<f64>::new().tableau(),
        DormandPrince5::<f64>::new().tableau(),
        DormandPrince853::<f64>::new().tableau(),
    ];
    for tableau in &tableaux {
        assert_eq!(tableau.a.len(), tableau.c.len());
        for (i, (row, &c)) in tableau.a.iter().zip(tableau.c).enumerate() {
            assert_eq!(
                row.len(),
                i,
                "row {i} of an explicit method has {i} entries"
            );
            let sum: f64 = row.iter().sum();
            let size: f64 = row.iter().map(|x| x.abs()).sum();
            assert!(
                (sum - c).abs() <= 8.0 * f64::EPSILON * size.max(1.0),
                "row {i}: {sum} against {c}"
            );
        }
    }
}

#[test]
fn the_classical_method_is_of_the_fourth_order_and_no_more() {
    let tableau = Rk4::<f64>::new().tableau();
    let trees = Trees::up_to(5);
    let a = rows(&tableau, 4);
    let phi = trees.weights(&as_slices(&a), 4);
    let (worst, size) = residual(&trees, &phi, tableau.b, 1..=4, |_| 1.0);
    assert!(worst <= 4.0 * f64::EPSILON * size, "{worst}");
    let (fifth, _) = residual(&trees, &phi, tableau.b, 5..=5, |_| 1.0);
    assert!(fifth > 1e-3, "{fifth}");
}

#[test]
fn the_dormand_prince_pair_has_orders_five_and_four() {
    let tableau = DormandPrince5::<f64>::new().tableau();
    let trees = Trees::up_to(6);
    let a = rows(&tableau, 7);
    let phi = trees.weights(&as_slices(&a), 7);
    // The weights propagated: order 5, all 17 trees, and not order 6.
    let (worst, size) = residual(&trees, &phi, tableau.b, 1..=5, |_| 1.0);
    assert!(worst <= 8.0 * f64::EPSILON * size, "{worst}");
    let (sixth, _) = residual(&trees, &phi, tableau.b, 6..=6, |_| 1.0);
    assert!(sixth > 1e-4, "{sixth}");
    // The embedded formula b − e: order 4 and not 5.
    let embedded: Vec<f64> = tableau
        .b
        .iter()
        .zip(tableau.e[0])
        .map(|(b, e)| b - e)
        .collect();
    let (worst, size) = residual(&trees, &phi, &embedded, 1..=4, |_| 1.0);
    assert!(worst <= 8.0 * f64::EPSILON * size, "{worst}");
    let (fifth, _) = residual(&trees, &phi, &embedded, 5..=5, |_| 1.0);
    assert!(fifth > 1e-4, "{fifth}");
    // Its seventh stage is the derivative at the end of the step, which the next step begins
    // with: the row of a is the weights.
    assert_eq!(tableau.a[6], &tableau.b[..6]);
    assert_eq!(tableau.b[6], 0.0);
}

/// The weights `bᵢ(θ)` of DOPRI5's continuous extension, `y₀ + θ·(r₁ + (1 − θ)·(r₂ + θ·(r₃ +
/// (1 − θ)·r₄)))` with `r₁ = h·Σbᵢkᵢ`, `r₂ = h·k₁ − r₁`, `r₃ = r₁ − h·k₇ − r₂`, `r₄ = h·Σdᵢkᵢ`.
fn dopri5_dense(tableau: &ButcherTableau, theta: f64) -> Vec<f64> {
    let s = 1.0 - theta;
    (0..7)
        .map(|i| {
            let r1 = tableau.b[i];
            let r2 = if i == 0 { 1.0 } else { 0.0 } - r1;
            let r3 = r1 - if i == 6 { 1.0 } else { 0.0 } - r2;
            let r4 = tableau.d[0][i];
            theta * (r1 + s * (r2 + theta * (r3 + s * r4)))
        })
        .collect()
}

#[test]
fn the_continuous_extension_of_the_pair_is_of_the_fourth_order() {
    let tableau = DormandPrince5::<f64>::new().tableau();
    let trees = Trees::up_to(5);
    let a = rows(&tableau, 7);
    let phi = trees.weights(&as_slices(&a), 7);
    for theta in [0.0, 0.1, 1.0 / 3.0, 0.5, 0.75, 0.9, 1.0] {
        let w = dopri5_dense(&tableau, theta);
        let (worst, size) = residual(&trees, &phi, &w, 1..=4, |o| theta.powi(o as i32));
        assert!(
            worst <= 16.0 * f64::EPSILON * size.max(1.0),
            "θ = {theta}: {worst}"
        );
        if theta > 0.0 && theta < 1.0 {
            let (fifth, _) = residual(&trees, &phi, &w, 5..=5, |o| theta.powi(o as i32));
            assert!(fifth > 1e-5, "θ = {theta}: {fifth}");
        }
    }
    // At θ = 1 it is the step itself.
    let end = dopri5_dense(&tableau, 1.0);
    for (x, b) in end.iter().zip(tableau.b) {
        assert!((x - b).abs() < 1e-15);
    }
}

#[test]
fn the_method_of_order_eight_has_its_three_orders() {
    let tableau = DormandPrince853::<f64>::new().tableau();
    let trees = Trees::up_to(9);
    // The twelve stages of the step.
    let a = rows(&tableau, 12);
    let phi = trees.weights(&as_slices(&a), 12);
    // The weights: order 8, all 200 trees, and not order 9.
    let (worst, size) = residual(&trees, &phi, tableau.b, 1..=8, |_| 1.0);
    assert!(worst <= 64.0 * f64::EPSILON * size, "{worst} of {size}");
    let (ninth, _) = residual(&trees, &phi, tableau.b, 9..=9, |_| 1.0);
    assert!(ninth > 1e-6, "{ninth}");
    // The first error estimate is the difference of two formulas of order 5 at least: it
    // annihilates every tree up to order 5, and not those of order 6.
    let zero = |_| 0.0;
    let (worst, size) = residual(&trees, &phi, tableau.e[0], 1..=5, zero);
    assert!(worst <= 64.0 * f64::EPSILON * size, "{worst} of {size}");
    let (sixth, _) = residual(&trees, &phi, tableau.e[0], 6..=6, zero);
    assert!(sixth > 1e-5, "{sixth}");
    // The second is the difference of the weights and a formula of order 3.
    let third: Vec<f64> = tableau
        .b
        .iter()
        .zip(tableau.e[1])
        .map(|(b, e)| b - e)
        .collect();
    let (worst, size) = residual(&trees, &phi, &third, 1..=3, |_| 1.0);
    assert!(worst <= 64.0 * f64::EPSILON * size, "{worst} of {size}");
    let (fourth, _) = residual(&trees, &phi, &third, 4..=4, |_| 1.0);
    assert!(fourth > 1e-3, "{fourth}");
    // The third-order formula lives on the first, ninth and twelfth stages only.
    for (i, x) in third.iter().enumerate() {
        assert_eq!(*x == 0.0, ![0, 8, 11].contains(&i), "stage {i}");
    }
}

/// The weights `bᵢ(θ)` of DOP853's continuous extension over its sixteen stages, as `contd8`
/// evaluates it: `y₀ + θ·(r₁ + (1 − θ)·(r₂ + θ·(r₃ + (1 − θ)·(r₄ + θ·(r₅ + (1 − θ)·(r₆ +
/// θ·r₇))))))`, with `r₁ … r₃` as for DOPRI5 (the derivative at the end being the thirteenth
/// stage) and `r₄ … r₇ = h·Σ dᵢkᵢ` from the four rows of `d`.
fn dop853_dense(tableau: &ButcherTableau, theta: f64) -> Vec<f64> {
    let s = 1.0 - theta;
    (0..16)
        .map(|i| {
            let r1 = if i < 12 { tableau.b[i] } else { 0.0 };
            let r2 = if i == 0 { 1.0 } else { 0.0 } - r1;
            let r3 = r1 - if i == 12 { 1.0 } else { 0.0 } - r2;
            let d = |k: usize| tableau.d[k][i];
            let inner = d(0) + theta * (d(1) + s * (d(2) + theta * d(3)));
            theta * (r1 + s * (r2 + theta * (r3 + s * inner)))
        })
        .collect()
}

#[test]
fn the_continuous_extension_of_order_eight_is_of_the_seventh_order() {
    let tableau = DormandPrince853::<f64>::new().tableau();
    let trees = Trees::up_to(8);
    let a = rows(&tableau, 16);
    let phi = trees.weights(&as_slices(&a), 16);
    for theta in [0.0, 0.1, 1.0 / 7.0, 1.0 / 3.0, 0.5, 0.8, 0.9, 1.0] {
        let w = dop853_dense(&tableau, theta);
        let (worst, size) = residual(&trees, &phi, &w, 1..=7, |o| theta.powi(o as i32));
        assert!(
            worst <= 256.0 * f64::EPSILON * size.max(1.0),
            "θ = {theta}: {worst} of {size}"
        );
        if theta > 0.0 && theta < 1.0 {
            let (eighth, _) = residual(&trees, &phi, &w, 8..=8, |o| theta.powi(o as i32));
            assert!(eighth > 1e-7, "θ = {theta}: {eighth}");
        }
    }
    // The extension ends on the step, which does not use the last four stages.
    let end = dop853_dense(&tableau, 1.0);
    for (i, x) in end.iter().enumerate() {
        let b = if i < 12 { tableau.b[i] } else { 0.0 };
        assert!((x - b).abs() < 1e-13, "stage {i}: {x} against {b}");
    }
    // The thirteenth stage is the derivative at the end of the step.
    assert_eq!(tableau.a[12], tableau.b);
}
