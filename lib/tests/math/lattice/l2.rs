//! The L² algorithm: `a::math::lattice::lll_fp`. Every result is checked exactly (`U·B = B'`,
//! `U` unimodular, `is_lll_reduced`), compared with `lll` and with fplll and FLINT
//! (data/lll_reference.txt), and bases built to fail Lovász's condition by a hair drive it through
//! every stage: the 53-bit run, the proof by intervals, the exact test, the runs in `BigFloat`
//! and the exact reduction at the end.

use super::common::{Rng, basis, check_reduction, knapsack, random_basis};
use super::lll::{
    dependent_basis, first_norm2, is_t_up_to_signs, log2_ratio, norm2, reduced_by_definition,
    reference_cases, unique_basis,
};
use a::math::lattice::{DeepLllOptions, LllOptions, deep_lll, is_lll_reduced, lll, lll_fp};
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use std::time::Instant;

/// `1 − 2⁻⁵³`, the largest float below 1: with it the floating-point tests have no margin.
pub const ONE_MINUS: f64 = 1.0 - f64::EPSILON / 2.0;

// Bases from data/gen_l2_boundary.py: with A = 2^k and δ = 1 − 2⁻⁵³, the last row has the squared
// norm δ·A² − 2. LOVASZ_k: b₀ = (A, 0, 0), b₁ short in its first entry, Lovász's condition failing
// by 2. DEEP_k: three rows, LLL-reduced, the deep exchange condition of b₂ at place 0 failing by 2.
// Floats of fewer than about 2k bits cannot see either.
pub const LOVASZ_35: &[&[&str]] = &[
    &["34359738368", "0", "0"],
    &["19", "24694698458", "23890656935"],
];
pub const LOVASZ_200: &[&[&str]] = &[
    &[
        "1606938044258990275541962092341162602522202993782792835301376",
        "0",
        "0",
    ],
    &[
        "235",
        "1501876910461756888128050663601755493937124395590578398000175",
        "571502951793563517837097627446086659980547269892169203728986",
    ],
];
pub const LOVASZ_500: &[&[&str]] = &[
    &[
        "3273390607896141870013189696827599152216642046043064789483291368096133796404674554883270092325904157150886684127560071009217256545885393053328527589376",
        "0",
        "0",
    ],
    &[
        "773",
        "3226504822261369124410255865962737509944045573214011476828212845974961075445381091349961893841570390050191823957193790738120075326769807686756981131754",
        "552044113986194816433639568268421619247979579526016585840501878985992040414785879825203604096935378544652066374906974928913920590367388828150253695999",
    ],
];
pub const DEEP_35: &[&[&str]] = &[
    &["34359738368", "0", "0", "0", "0"],
    &["0", "34140458024", "3875660799", "0", "0"],
    &["19", "0", "0", "24694698458", "23890656935"],
];
pub const DEEP_200: &[&[&str]] = &[
    &[
        "1606938044258990275541962092341162602522202993782792835301376",
        "0",
        "0",
        "0",
        "0",
    ],
    &[
        "0",
        "1222974934679336284586859838181182993579560178407617260200488",
        "1042392530303715164521247205960310696873751186040723536163487",
        "0",
        "0",
    ],
    &[
        "235",
        "0",
        "0",
        "1501876910461756888128050663601755493937124395590578398000175",
        "571502951793563517837097627446086659980547269892169203728986",
    ],
];
pub const DEEP_500: &[&[&str]] = &[
    &[
        "3273390607896141870013189696827599152216642046043064789483291368096133796404674554883270092325904157150886684127560071009217256545885393053328527589376",
        "0",
        "0",
        "0",
        "0",
    ],
    &[
        "0",
        "2729052098091939599411149950196879997307779716896997874635144249543194817331125532025180999707954382616402195320657839804249071850640160161985194546259",
        "1807584221485310402285182025202893279532807754482188527231420314292008325264160090247431115469479213964304906318623362170383967742178700052266724669884",
        "0",
        "0",
    ],
    &[
        "773",
        "0",
        "0",
        "3226504822261369124410255865962737509944045573214011476828212845974961075445381091349961893841570390050191823957193790738120075326769807686756981131754",
        "552044113986194816433639568268421619247979579526016585840501878985992040414785879825203604096935378544652066374906974928913920590367388828150253695999",
    ],
];

/// The matrix of rows of decimal strings.
pub fn matrix_of(rows: &[&[&str]]) -> Matrix<Number> {
    Matrix::from_fn(rows.len(), rows[0].len(), |i, j| {
        rows[i][j].parse().unwrap()
    })
}

#[test]
fn random_knapsack_and_dependent_bases_are_reduced() {
    let mut rng = Rng(19);
    for (n, bits) in [
        (2, 10),
        (3, 30),
        (5, 20),
        (8, 60),
        (12, 16),
        (16, 40),
        (24, 30),
    ] {
        let r = random_basis(&mut rng, n, n, bits);
        let k = knapsack(&mut rng, n, bits * 3);
        let d = dependent_basis(&mut rng, n + 2, n.max(3) - 1, n, bits, 6);
        for b in [r, k, d] {
            for (delta, eta) in [(0.99, 0.51), (0.75, 0.6), (0.3, 0.52)] {
                let r = lll_fp(&b, &LllOptions::new().with_delta(delta).with_eta(eta));
                check_reduction(&b, &r);
                assert!(
                    is_lll_reduced(r.basis(), delta, eta),
                    "n = {n}, bits = {bits}, δ = {delta}, η = {eta}"
                );
            }
        }
    }
}

#[test]
fn parameters_near_their_limits() {
    let mut rng = Rng(29);
    for (delta, eta) in [
        (0.999999, 0.500001),
        (ONE_MINUS, 0.51),
        (ONE_MINUS, 0.5 + f64::EPSILON / 2.0),
        (0.2500001, 0.50000009),
        (0.26, 0.505),
        (0.99, 0.994),
    ] {
        for b in [knapsack(&mut rng, 10, 50), random_basis(&mut rng, 6, 7, 40)] {
            let r = lll_fp(&b, &LllOptions::new().with_delta(delta).with_eta(eta));
            check_reduction(&b, &r);
            assert!(
                is_lll_reduced(r.basis(), delta, eta),
                "δ = {delta}, η = {eta}"
            );
            assert!(reduced_by_definition(r.basis(), delta, eta, |_, _| false));
        }
    }
}

#[test]
fn empty_and_single_row_shapes() {
    let o = LllOptions::new();
    for (n, m) in [(0, 0), (0, 4), (4, 0), (1, 0), (3, 3)] {
        let b = Matrix::<Number>::zeros(n, m);
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(r.basis().shape(), (n, m));
        assert_eq!(r.rank(), 0);
    }
    for b in [
        basis(&[[0, 3, -4]]),
        basis(&[[7]]),
        basis(&[[0, 0], [0, 5]]),
    ] {
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(*r.basis(), b);
    }
    // Zero rows among others go first, in the order of the rest kept by the reduction.
    let b = basis(&[[1, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 0]]);
    let r = lll_fp(&b, &o);
    check_reduction(&b, &r);
    assert_eq!(
        *r.basis(),
        basis(&[[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 2, 0]])
    );
}

#[test]
fn dependent_rows_come_out_as_zero_rows_first() {
    let o = LllOptions::new();
    for (b, rank) in [
        (basis(&[[1, 2], [2, 4], [3, 6]]), 1),
        (basis(&[[3, 0], [5, 0]]), 1),
        (basis(&[[4, 2], [6, 3], [2, 0]]), 2),
        (
            basis(&[[6, 10, 15], [10, 15, 6], [15, 6, 10], [1, 1, 1]]),
            3,
        ),
    ] {
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(r.rank(), rank);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    }
    let mut rng = Rng(6);
    for (n, rk, m, bits) in [
        (5, 2, 5, 40),
        (8, 4, 6, 100),
        (12, 11, 12, 20),
        (6, 1, 2, 300),
    ] {
        let b = dependent_basis(&mut rng, n, rk, m, bits, 8);
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    }
}

/// The run in machine words up to the room it has. Its first input entries are the largest it
/// takes (columns of 30-bit numbers, knapsacks of 29-bit ones: Gram entries just below `2^60`),
/// and dependent rows of 8 and 29 bits, whose transformations pass `2^63`. Measured on a copy
/// that counts them, these inputs make row operations whose largest `|y| + |x|·|z|` reaches
/// 0.997 of `2^63` and that the test of the room admits, and 14 that it refuses, after which the
/// run goes on in integers of any size; with that test removed, the first input comes out with
/// `U·B ≠ B'`. The new diagonal entry of the Gram matrix is computed after the test, where it
/// cannot overflow (a debug build checks every operation).
#[test]
fn words_up_to_the_room_they_have() {
    let mut rng = Rng(30);
    let mut inputs = Vec::new();
    for n in [6, 10, 20] {
        inputs.push(random_basis(&mut rng, n, 1, 30));
    }
    for n in [10, 20] {
        inputs.push(knapsack(&mut rng, n, 29));
    }
    for (n, m) in [(12, 7), (16, 6), (20, 10)] {
        inputs.push(random_basis(&mut rng, n, m, 8));
    }
    for n in [5, 8] {
        inputs.push(random_basis(&mut rng, n, 2, 29));
    }
    for b in &inputs {
        let r = lll_fp(b, &LllOptions::new());
        check_reduction(b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
        let e = lll(b, &LllOptions::new());
        check_reduction(b, &e);
        assert!(is_lll_reduced(e.basis(), 0.99, 0.5));
    }
}

#[test]
fn huge_entries_beyond_the_range_of_f64() {
    // Gram matrices of entries of 2400 to 10000 bits: from 3000 bits the 53-bit floats need
    // their own exponent; below, the doubles scaled by a power of two run first.
    let mut rng = Rng(43);
    let o = LllOptions::new();
    for b in [
        knapsack(&mut rng, 16, 1500),
        knapsack(&mut rng, 6, 5000),
        random_basis(&mut rng, 4, 4, 3000),
        dependent_basis(&mut rng, 6, 4, 5, 1200, 10),
    ] {
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    }
}

#[test]
fn the_unique_reduced_basis_is_found() {
    let mut rng = Rng(10);
    for (n, zeros) in [(2, 0), (3, 1), (5, 0), (8, 2), (12, 0), (14, 3)] {
        let (b, t) = unique_basis(&mut rng, n, zeros);
        let r = lll_fp(&b, &LllOptions::new());
        check_reduction(&b, &r);
        assert!(is_t_up_to_signs(r.basis(), &t, zeros), "{:?}", r.basis());
        assert!(is_t_up_to_signs(
            lll(&b, &LllOptions::new()).basis(),
            &t,
            zeros
        ));
    }
}

#[test]
fn lll_fp_and_lll_agree() {
    // On the same bases both results are reduced, of the same rank, with first vectors equal or
    // within the LLL bound of each other.
    let (mut same, mut count) = (0, 0);
    for c in reference_cases().iter().filter(|c| c.n <= 16) {
        let b = c.basis();
        let o = LllOptions::new().with_delta(c.delta).with_eta(c.eta);
        let (f, e) = (lll_fp(&b, &o), lll(&b, &o));
        check_reduction(&b, &f);
        assert!(is_lll_reduced(f.basis(), c.delta, c.eta));
        assert!(is_lll_reduced(e.basis(), c.delta, c.eta));
        assert_eq!(f.rank(), e.rank());
        same += usize::from(first_norm2(&f) == first_norm2(&e));
        count += 1;
    }
    assert!(same * 10 >= count * 7, "{same} of {count}");
}

#[test]
fn first_vectors_match_fplll_and_flint() {
    let (mut same, mut count, mut sum, mut worst) = (0, 0, 0.0, 0.0f64);
    for c in reference_cases() {
        let b = c.basis();
        let r = lll_fp(&b, &LllOptions::new().with_delta(c.delta).with_eta(c.eta));
        assert_eq!(r.rank(), c.rank, "{} {} {}", c.kind, c.n, c.bits);
        assert!(is_lll_reduced(r.basis(), c.delta, c.eta));
        let ours = first_norm2(&r);
        let x = log2_ratio(&ours, &c.fplll);
        let bound = -((c.delta - c.eta * c.eta).log2()) * (c.rank as f64 - 1.0);
        assert!(x.abs() <= bound + 1e-9, "{} {}: ratio 2^{x}", c.kind, c.n);
        same += usize::from(ours == c.fplll || ours == c.flint);
        count += 1;
        sum += x;
        worst = worst.max(x);
    }
    assert_eq!(count, 66);
    assert!(same * 10 >= count * 7, "{same} of {count}");
    assert!(
        sum / count as f64 <= 0.1,
        "mean log₂ ratio {}",
        sum / count as f64
    );
    assert!(worst <= 2.0, "worst log₂ ratio {worst}");
}

/// `b` with a marker: a column of zeros, and a last row `(⌊7A/10⌋, 0, …, 0, 2A)` for `A = b₀₀`,
/// whose coefficient on the row `(A, 0, …)` is 0.7 and on the others about 0. With `η = 0.9` a
/// floating-point run leaves it as it is (`|μ| ≤ η̄`); the exact reduction, which reduces to 1/2,
/// subtracts that row from it. So the result tells which of them produced it.
pub fn with_marker(b: &Matrix<Number>) -> (Matrix<Number>, Vec<Number>) {
    let (n, m) = b.shape();
    let a = b[(0, 0)].clone();
    let mut marker = vec![Number::from(0u8); m + 1];
    marker[0] = &(&a * &Number::from(7u8)) / &Number::from(10u8);
    marker[m] = &a * &Number::from(2u8);
    let rows = Matrix::from_fn(n + 1, m + 1, |i, j| {
        if i == n {
            marker[j].clone()
        } else if j == m {
            Number::from(0u8)
        } else {
            b[(i, j)].clone()
        }
    });
    (rows, marker)
}

/// Whether the last row of `r` is `marker`.
pub fn keeps(r: &Matrix<Number>, marker: &[Number]) -> bool {
    (0..r.cols()).all(|j| r[(r.rows() - 1, j)] == marker[j])
}

#[test]
fn bases_that_fail_lovasz_by_a_hair_take_every_path() {
    // The 53-bit run has no margin at δ = 1 − 2⁻⁵³ and does not see the failure. For LOVASZ_35
    // the intervals prove it; for LOVASZ_200 and LOVASZ_500 they cannot decide, and the exact test
    // finds it. A run in BigFloat, whose δ̄ = δ + 2⁻⁵⁵ holds the margin, sees it and repairs it.
    // Each time the two rows must come out exchanged.
    let o = LllOptions::new().with_delta(ONE_MINUS);
    for rows in [LOVASZ_35, LOVASZ_200, LOVASZ_500] {
        let b = matrix_of(rows);
        assert!(!is_lll_reduced(&b, ONE_MINUS, 0.51));
        assert!(is_lll_reduced(&b, 0.99, 0.51));
        assert!(!reduced_by_definition(&b, ONE_MINUS, 0.51, |_, _| false));
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), ONE_MINUS, 0.51));
        assert_eq!(norm2(r.basis(), 0), norm2(&b, 1));
        assert_eq!(r.basis(), lll(&b, &o).basis());
        // With the marker the repair must be a floating-point run's, not the exact reduction's: a
        // BigFloat stage that failed, or a proof that refused every result, would leave the work
        // to the exact reduction, and it would take the marker's 0.7 down to −0.3.
        let (bm, marker) = with_marker(&b);
        let r = lll_fp(&bm, &o.with_eta(0.9));
        check_reduction(&bm, &r);
        assert!(is_lll_reduced(r.basis(), ONE_MINUS, 0.9));
        assert_eq!(norm2(r.basis(), 0), norm2(&b, 1));
        assert!(keeps(r.basis(), &marker), "{:?}", r.basis());
        // The exact reduction does take it down.
        assert!(!keeps(lll(&bm, &o).basis(), &marker));
    }
    // The deep bases are LLL-reduced already, by a hair too.
    for rows in [DEEP_35, DEEP_200, DEEP_500] {
        let b = matrix_of(rows);
        assert!(is_lll_reduced(&b, ONE_MINUS, 0.51));
        let r = lll_fp(&b, &o);
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), ONE_MINUS, 0.51));
    }
}

#[test]
fn delta_next_to_one_is_reduced_in_floating_point() {
    // At δ = 1 − 2⁻⁵³ the 53-bit floats have no room between δ and 1. The iteration bound must
    // still allow the runs the iterations they need: a knapsack of 24 numbers of 240 bits needs
    // far more than 4n + 1000 of them, and if the runs stopped there the exact reduction would do
    // the work and take the marker down. Below it, in its own two columns, a pair of rows whose
    // coefficient is 0.7.
    let k = knapsack(&mut Rng(83), 24, 240);
    let (n, m) = k.shape();
    let a = Number::from(1u8) << 300usize;
    let marker: Vec<Number> = (0..m + 2)
        .map(|j| {
            if j == m {
                &(&a * &Number::from(7u8)) / &Number::from(10u8)
            } else if j == m + 1 {
                &a * &Number::from(2u8)
            } else {
                Number::from(0u8)
            }
        })
        .collect();
    let b = Matrix::from_fn(n + 2, m + 2, |i, j| {
        if i < n {
            if j < m {
                k[(i, j)].clone()
            } else {
                Number::from(0u8)
            }
        } else if i == n {
            if j == m { a.clone() } else { Number::from(0u8) }
        } else {
            marker[j].clone()
        }
    });
    let r = lll_fp(&b, &LllOptions::new().with_delta(ONE_MINUS).with_eta(0.9));
    check_reduction(&b, &r);
    assert!(is_lll_reduced(r.basis(), ONE_MINUS, 0.9));
    assert!(keeps(r.basis(), &marker));
    let o = DeepLllOptions::new()
        .with_delta(ONE_MINUS)
        .with_eta(0.9)
        .with_depth(Some(5));
    let d = deep_lll(&b, &o);
    check_reduction(&b, &d);
    assert!(is_lll_reduced(d.basis(), ONE_MINUS, 0.9));
    assert!(keeps(d.basis(), &marker));
}

#[test]
fn the_proof_of_a_reduced_basis_costs_less_than_the_exact_test() {
    // A reduced basis of 16 rows of 2000 bits: `lll_fp` changes nothing and proves it in interval
    // arithmetic of 96 bits, while the exact integral Gram–Schmidt orthogonalisation (which `lll`
    // computes, and finds nothing to change) works on integers of up to 64000 bits, about a
    // hundred times the work here. A proof that always left the question open would make `lll_fp`
    // cost the exact work and more; so would `is_lll_reduced`, which takes the same intervals
    // first. They are timed in the same test, so the speed of the machine drops out; the proofs, a
    // few milliseconds, are timed five times and the fastest taken, so that a thread preempted
    // once on a loaded machine (CI runs the tests in parallel) does not fail it.
    let mut rng = Rng(89);
    let (n, bits) = (16, 2000);
    let base = Number::from(1u8) << bits;
    let b = Matrix::from_fn(n, n, |i, j| {
        if j > i {
            Number::from(0u8)
        } else if j == i {
            &base + &Number::from(i as u64)
        } else {
            rng.number(bits as u64 - 4)
        }
    });
    let o = LllOptions::new();
    let mut proved = std::time::Duration::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        let r = lll_fp(&b, &o);
        proved = proved.min(start.elapsed());
        assert_eq!(*r.basis(), b);
        assert_eq!(*r.transform(), Matrix::<Number>::identity(n));
    }
    let mut tested = std::time::Duration::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        assert!(is_lll_reduced(&b, 0.99, 0.51));
        tested = tested.min(start.elapsed());
    }
    let start = Instant::now();
    let r = lll(&b, &o);
    let exact = start.elapsed();
    assert_eq!(*r.basis(), b);
    assert!(
        proved * 8 < exact,
        "proof {proved:?}, exact reduction {exact:?}"
    );
    assert!(
        tested * 8 < exact,
        "test {tested:?}, exact reduction {exact:?}"
    );
}

#[test]
fn coefficients_of_exactly_one_half() {
    // μ₂₁ = 1/2 exactly; in 53-bit floats the first basis computes it exactly, the second as
    // 1/2 + 2⁻⁵³, which a size reduction to η̄ = 1/2 would turn back and forth.
    let tie = [
        basis(&[[3, 0, 0], [1, 6, 0], [1, 3, 6]]),
        basis(&[
            [3324, -4192, -3024],
            [3856, 4428, 2434],
            [800326, -4937672, 7720461],
        ]),
    ];
    for b in &tie {
        assert!(is_lll_reduced(b, 0.99, 0.5));
        for eta in [0.51, 0.5 + f64::EPSILON / 2.0] {
            let r = lll_fp(b, &LllOptions::new().with_eta(eta));
            check_reduction(b, &r);
            assert!(is_lll_reduced(r.basis(), 0.99, eta));
        }
    }
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1)")]
fn delta_one_panics() {
    lll_fp(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(1.0),
    );
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1)")]
fn delta_of_a_quarter_panics() {
    lll_fp(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(0.25),
    );
}

#[test]
#[should_panic(expected = "η must be in (1/2, √δ)")]
fn eta_of_one_half_panics() {
    lll_fp(&basis(&[[1, 0], [0, 1]]), &LllOptions::new().with_eta(0.5));
}

#[test]
#[should_panic(expected = "η must be in (1/2, √δ)")]
fn eta_beyond_the_root_of_delta_panics() {
    lll_fp(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(0.75).with_eta(0.87),
    );
}

#[test]
#[should_panic(expected = "η must be in (1/2, √δ)")]
fn eta_nan_panics() {
    lll_fp(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_eta(f64::NAN),
    );
}

/// The sizes the L² algorithm is for, too slow for a debug build: run with
/// `cargo test --release --test math -- --ignored lattice::l2`.
#[test]
#[ignore]
fn the_sizes_it_is_for() {
    let mut rng = Rng(1);
    let o = LllOptions::new();
    for b in [
        random_basis(&mut rng, 100, 100, 100),
        knapsack(&mut rng, 50, 1000),
        knapsack(&mut rng, 100, 100),
        knapsack(&mut rng, 40, 400),
        dependent_basis(&mut rng, 60, 50, 60, 100, 8),
        random_basis(&mut rng, 70, 60, 80),
    ] {
        let r = lll_fp(&b, &o);
        let (bb, u) = (r.basis(), r.transform());
        assert_eq!(&u.clone() * &b, *bb, "U·B = B'");
        assert!(is_lll_reduced(bb, 0.99, 0.51));
        assert_eq!(r.rank(), b.rank());
    }
}
