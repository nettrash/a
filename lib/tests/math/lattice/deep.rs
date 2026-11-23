//! LLL with deep insertions: `a::math::lattice::deep_lll`. The results are checked exactly
//! (`U·B = B'`, `U` unimodular, `is_lll_reduced`) and against the definition of the deep
//! exchange condition in rationals; compared with `lll_fp` (deep insertions find shorter first
//! vectors on knapsack bases) and with NTL's `LLL_FP` with deep insertions
//! (data/deep_reference.txt, from data/gen_deep_reference.py); and bases built to fail the deep
//! condition by a hair drive it through every stage, the exact reduction with deep insertions at
//! the end.

use super::common::{Rng, basis, check_reduction, knapsack, random_basis};
use super::l2::{DEEP_35, DEEP_200, DEEP_500, ONE_MINUS, keeps, matrix_of, with_marker};
use super::lll::{
    dependent_basis, first_norm2, is_t_up_to_signs, log2_ratio, norm2, reduced_by_definition,
    unique_basis,
};
use a::math::lattice::{DeepLllOptions, LllOptions, deep_lll, is_lll_reduced, lll_fp};
use a::math::matrix::Matrix;
use a::math::numbers::Number;

/// The places `i < k` where a depth allows an insertion (besides `k − 1`, always).
fn allowed(depth: Option<usize>) -> impl Fn(usize, usize) -> bool {
    move |i, k| depth.is_none_or(|t| i < t || k - i <= t)
}

#[test]
fn results_meet_the_deep_exchange_condition() {
    let mut rng = Rng(51);
    for (n, bits) in [(2, 20), (4, 30), (6, 24), (8, 40), (10, 30)] {
        let bases = [
            knapsack(&mut rng, n, bits),
            random_basis(&mut rng, n, n, bits),
            dependent_basis(&mut rng, n + 1, n.max(2) - 1, n, bits, 5),
        ];
        for b in &bases {
            for depth in [None, Some(0), Some(1), Some(3)] {
                for (delta, eta) in [(0.99, 0.51), (0.75, 0.5), (0.6, 0.7)] {
                    let o = DeepLllOptions::new()
                        .with_delta(delta)
                        .with_eta(eta)
                        .with_depth(depth);
                    let r = deep_lll(b, &o);
                    check_reduction(b, &r);
                    assert!(is_lll_reduced(r.basis(), delta, eta));
                    assert!(
                        reduced_by_definition(r.basis(), delta, eta, allowed(depth)),
                        "n = {n}, depth = {depth:?}, δ = {delta}, η = {eta}: {:?}",
                        r.basis()
                    );
                }
            }
        }
    }
}

#[test]
fn deep_insertions_beat_lll_on_knapsack_bases() {
    // Insertions anywhere and at depth 5: on 16 knapsack bases of 20 numbers of 40 bits the
    // first vector is never longer than LLL's, and shorter in 16 of the 32 runs (fplll's δ).
    let (mut wins, mut losses, mut sum) = (0, 0, 0.0);
    for seed in 0..16 {
        let b = knapsack(&mut Rng(1000 + seed), 20, 40);
        let l = first_norm2(&lll_fp(&b, &LllOptions::new()));
        for depth in [None, Some(5)] {
            let d = first_norm2(&deep_lll(&b, &DeepLllOptions::new().with_depth(depth)));
            wins += usize::from(d < l);
            losses += usize::from(d > l);
            sum += log2_ratio(&d, &l);
        }
    }
    assert!(wins >= 8, "{wins} wins");
    assert!(losses <= 2, "{losses} losses");
    assert!(sum < -0.5, "Σ log₂ ratio {sum}");
}

#[test]
fn depth_zero_is_lll() {
    // Some(0) allows LLL's exchange alone: the same steps as `lll_fp`, the same result.
    let mut rng = Rng(61);
    for (n, bits) in [(6, 30), (12, 50), (16, 64)] {
        let b = knapsack(&mut rng, n, bits);
        let d = deep_lll(&b, &DeepLllOptions::new().with_depth(Some(0)));
        check_reduction(&b, &d);
        assert_eq!(d.basis(), lll_fp(&b, &LllOptions::new()).basis());
    }
}

#[test]
fn first_vectors_compare_with_ntl() {
    // NTL's LLL_FP(B, 0.99, deep) inserts at i ≤ deep or k − i ≤ deep, as depth Some(deep) does.
    // Floating-point reductions of the same kind need not give the same vector; on average ours
    // are as short.
    let (mut sum_all, mut sum5, mut sum_lll, mut same, mut count) = (0.0, 0.0, 0.0, 0, 0);
    for line in include_str!("data/deep_reference.txt").lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let (n, bits, seed): (usize, u64, u64) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        );
        let ntl: Vec<Number> = f[3..6].iter().map(|x| x.parse().unwrap()).collect();
        let b = knapsack(&mut Rng(seed), n, bits);
        let all = first_norm2(&deep_lll(&b, &DeepLllOptions::new()));
        let five = first_norm2(&deep_lll(&b, &DeepLllOptions::new().with_depth(Some(5))));
        let plain = first_norm2(&lll_fp(&b, &LllOptions::new()));
        sum_all += log2_ratio(&all, &ntl[2]);
        sum5 += log2_ratio(&five, &ntl[1]);
        sum_lll += log2_ratio(&plain, &ntl[0]);
        same += usize::from(all == ntl[2]) + usize::from(five == ntl[1]);
        count += 1;
    }
    assert_eq!(count, 24);
    let c = count as f64;
    assert!(sum_all / c <= 0.15 && sum5 / c <= 0.15 && sum_lll / c <= 0.15);
    assert!(same >= count, "{same} of {}", 2 * count);
}

#[test]
fn eta_of_one_half_is_exact() {
    // The integral reduction with deep insertions: size-reduced to 1/2 exactly, also where a
    // coefficient is exactly 1/2 and floating point computes it as more.
    let mut rng = Rng(71);
    let tie = basis(&[
        [3324, -4192, -3024],
        [3856, 4428, 2434],
        [800326, -4937672, 7720461],
    ]);
    for b in [
        tie,
        basis(&[[3, 0, 0], [1, 6, 0], [1, 3, 6]]),
        knapsack(&mut rng, 12, 40),
        random_basis(&mut rng, 6, 6, 30),
        dependent_basis(&mut rng, 7, 4, 6, 20, 4),
    ] {
        for depth in [None, Some(2)] {
            let o = DeepLllOptions::new().with_eta(0.5).with_depth(depth);
            let r = deep_lll(&b, &o);
            check_reduction(&b, &r);
            assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
            assert!(reduced_by_definition(r.basis(), 0.99, 0.5, allowed(depth)));
        }
    }
    // A unique reduced basis is found exactly.
    let (b, t) = unique_basis(&mut rng, 6, 2);
    let r = deep_lll(&b, &DeepLllOptions::new().with_eta(0.5));
    assert!(is_t_up_to_signs(r.basis(), &t, 2));
}

#[test]
fn bases_that_fail_the_deep_condition_by_a_hair_take_every_path() {
    // LLL-reduced bases whose last row belongs in front, by 2 in δ·A² for A = 2^k, δ = 1 − 2⁻⁵³:
    // the 53-bit run (no margin at this δ) does not see it; the intervals prove the failure
    // (k = 35) or leave it to the exact test (k = 200, 500); a BigFloat run, with δ̄ = δ + 2⁻⁵⁵,
    // inserts the row in front. With η = 1/2 the exact reduction with deep insertions does it.
    for rows in [DEEP_35, DEEP_200, DEEP_500] {
        let b = matrix_of(rows);
        assert!(is_lll_reduced(&b, ONE_MINUS, 0.51));
        assert!(!reduced_by_definition(&b, ONE_MINUS, 0.51, allowed(None)));
        for eta in [0.51, 0.5] {
            let o = DeepLllOptions::new().with_delta(ONE_MINUS).with_eta(eta);
            let r = deep_lll(&b, &o);
            check_reduction(&b, &r);
            assert!(reduced_by_definition(
                r.basis(),
                ONE_MINUS,
                eta,
                allowed(None)
            ));
            assert_eq!(norm2(r.basis(), 0), norm2(&b, 2));
        }
        // With a marker (coefficient 0.7 on the first row): kept by the floating-point runs at
        // η = 0.9, so a BigFloat run did the insertion; taken down to −0.3 by the exact reduction.
        let (bm, marker) = with_marker(&b);
        let o = DeepLllOptions::new().with_delta(ONE_MINUS).with_eta(0.9);
        let r = deep_lll(&bm, &o);
        check_reduction(&bm, &r);
        assert!(reduced_by_definition(
            r.basis(),
            ONE_MINUS,
            0.9,
            allowed(None)
        ));
        assert_eq!(norm2(r.basis(), 0), norm2(&b, 2));
        assert!(keeps(r.basis(), &marker), "{:?}", r.basis());
        let e = deep_lll(&bm, &o.with_eta(0.5));
        assert!(reduced_by_definition(
            e.basis(),
            ONE_MINUS,
            0.5,
            allowed(None)
        ));
        assert_eq!(norm2(e.basis(), 0), norm2(&b, 2));
        assert!(!keeps(e.basis(), &marker));
    }
}

#[test]
fn dependent_rows_and_empty_shapes() {
    let o = DeepLllOptions::new();
    for (n, m) in [(0, 0), (0, 2), (2, 0), (3, 3)] {
        let b = Matrix::<Number>::zeros(n, m);
        let r = deep_lll(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(r.rank(), 0);
    }
    for (b, rank) in [
        (basis(&[[3, 0], [5, 0]]), 1),
        (basis(&[[1, 2], [2, 4], [0, 0], [3, 6]]), 1),
        (basis(&[[4, 2], [6, 3], [2, 0]]), 2),
        (basis(&[[5]]), 1),
    ] {
        for eta in [0.51, 0.5] {
            let r = deep_lll(&b, &o.with_eta(eta));
            check_reduction(&b, &r);
            assert_eq!(r.rank(), rank);
            assert!(reduced_by_definition(r.basis(), 0.99, eta, allowed(None)));
        }
    }
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1)")]
fn delta_one_panics() {
    deep_lll(
        &basis(&[[1, 0], [0, 1]]),
        &DeepLllOptions::new().with_delta(1.0),
    );
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1)")]
fn delta_below_a_quarter_panics() {
    deep_lll(
        &basis(&[[1, 0], [0, 1]]),
        &DeepLllOptions::new().with_delta(0.2),
    );
}

#[test]
#[should_panic(expected = "η must be in [1/2, √δ)")]
fn eta_below_one_half_panics() {
    deep_lll(
        &basis(&[[1, 0], [0, 1]]),
        &DeepLllOptions::new().with_eta(0.49),
    );
}

#[test]
#[should_panic(expected = "η must be in [1/2, √δ)")]
fn eta_beyond_the_root_of_delta_panics() {
    deep_lll(
        &basis(&[[1, 0], [0, 1]]),
        &DeepLllOptions::new().with_delta(0.5).with_eta(0.71),
    );
}
