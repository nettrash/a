//! BKZ: `a::math::lattice::bkz`. The block conditions are checked exactly, by
//! `shortest_vector` on each projected block (made integral); the quality against fplll 5.5's
//! BKZ (`data/bkz_fplll.txt`, made by `data/gen_bkz.py`); and the invariants of every reduction.

use super::common::{Rng, basis, check_reduction, knapsack, random_basis};
use super::enumerate::make;
use a::math::lattice::{BkzOptions, Reduced, bkz, is_lll_reduced, shortest_vector};
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::math::rational::Rational;

const FPLLL: &str = include_str!("data/bkz_fplll.txt");

/// The nonzero rows of a reduced basis.
fn nonzero_rows(r: &Reduced) -> Vec<Vec<Number>> {
    let b = r.basis();
    (b.rows() - r.rank()..b.rows())
        .map(|i| (0..b.cols()).map(|j| b[(i, j)].clone()).collect())
        .collect()
}

/// The Gram–Schmidt vectors of independent rows, exactly.
fn gram_schmidt(rows: &[Vec<Number>]) -> Vec<Vec<Rational>> {
    let mut out: Vec<Vec<Rational>> = Vec::new();
    for b in rows {
        let mut v: Vec<Rational> = b.iter().map(Rational::from).collect();
        for s in &out {
            let ss: Rational = s.iter().map(|x| x * x).sum();
            let bs: Rational = b.iter().zip(s).map(|(x, y)| &Rational::from(x) * y).sum();
            let mu = &bs / &ss;
            for (x, y) in v.iter_mut().zip(s) {
                *x = &*x - &(&mu * y);
            }
        }
        out.push(v);
    }
    out
}

/// The rows `k..end` projected orthogonally to the rows before `k`, times `d_k = |b*₀|²···|b*ₖ₋₁|²`,
/// which makes them integral: the lattice of the block, scaled.
fn projected_block(
    rows: &[Vec<Number>],
    gs: &[Vec<Rational>],
    k: usize,
    end: usize,
) -> Matrix<Number> {
    let norms: Vec<Rational> = gs.iter().map(|s| s.iter().map(|x| x * x).sum()).collect();
    let dk: Rational = norms[..k]
        .iter()
        .fold(Rational::from(Number::from(1u8)), |a, x| &a * x);
    let m = rows[0].len();
    let mut out = Vec::new();
    for b in &rows[k..end] {
        let mut v: Vec<Rational> = b.iter().map(Rational::from).collect();
        for (s, ss) in gs[..k].iter().zip(&norms) {
            let bs: Rational = b.iter().zip(s).map(|(x, y)| &Rational::from(x) * y).sum();
            let mu = &bs / ss;
            for (x, y) in v.iter_mut().zip(s) {
                *x = &*x - &(&mu * y);
            }
        }
        let row: Vec<Number> = v
            .iter()
            .map(|x| {
                let y = x * &dk;
                assert!(y.is_integer(), "d_k·π_k(b) is integral");
                y.numer().clone()
            })
            .collect();
        out.push(row);
    }
    Matrix::from_fn(end - k, m, |i, j| out[i][j].clone())
}

/// The Gram–Schmidt coefficients `μᵢⱼ` and the squared norms `|b*ⱼ|²`, exactly.
fn coefficients(rows: &[Vec<Number>], gs: &[Vec<Rational>]) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let r: Vec<Rational> = gs.iter().map(|s| s.iter().map(|x| x * x).sum()).collect();
    let mu = rows
        .iter()
        .map(|b| {
            gs.iter()
                .zip(&r)
                .map(|(s, ss)| {
                    let bs: Rational = b.iter().zip(s).map(|(x, y)| &Rational::from(x) * y).sum();
                    &bs / ss
                })
                .collect()
        })
        .collect();
    (mu, r)
}

/// Whether the projected block `k..end` holds a nonzero vector shorter than `b*ₖ`: an exact
/// rational Fincke–Pohst search written for this test alone, a plain recursion over every
/// integer coordinate within the radius (`|xₗ − cₗ|² < (|b*ₖ|² − part)/|b*ₗ|²`, the centre
/// `cₗ = −Σⱼ xⱼ·μⱼₗ`), sharing nothing with the crate's enumeration. Of `±x` only the one whose
/// last nonzero coordinate is positive is tried.
fn shorter_in_block(mu: &[Vec<Rational>], r: &[Rational], k: usize, end: usize) -> bool {
    fn search(
        (mu, r): (&[Vec<Rational>], &[Rational]),
        (k, end, l): (usize, usize, usize),
        x: &mut [i64],
        part: Rational,
        zero_above: bool,
    ) -> bool {
        let bound = &r[k];
        let c: Rational = (l + 1..end)
            .map(|j| -(&Rational::from(Number::from(x[j - k])) * &mu[j][l]))
            .sum();
        let w = &(bound - &part) / &r[l];
        if !w.is_positive() {
            return false;
        }
        let s = i64::try_from(&w.ceil().isqrt()).unwrap() + 1;
        let lo = i64::try_from(&c.floor()).unwrap() - s;
        let hi = i64::try_from(&c.ceil()).unwrap() + s;
        let lo = if zero_above { lo.max(0) } else { lo };
        for v in lo..=hi {
            let dev = &Rational::from(Number::from(v)) - &c;
            let next = &part + &(&(&dev * &dev) * &r[l]);
            if &next >= bound {
                continue;
            }
            x[l - k] = v;
            let found = if l == k {
                !(zero_above && v == 0)
            } else {
                search((mu, r), (k, end, l - 1), x, next, zero_above && v == 0)
            };
            if found {
                x[l - k] = 0;
                return true;
            }
        }
        x[l - k] = 0;
        false
    }
    let mut x = vec![0i64; end - k];
    search(
        (mu, r),
        (k, end, end - 1),
        &mut x,
        Rational::from(Number::default()),
        true,
    )
}

/// Whether every block's first Gram–Schmidt vector is a shortest vector of its projected block,
/// by two oracles that must agree: `shortest_vector` on the block made integral, and the
/// independent search of [`shorter_in_block`].
fn blocks_are_reduced(rows: &[Vec<Number>], beta: usize) -> bool {
    let n = rows.len();
    let gs = gram_schmidt(rows);
    let (mu, r) = coefficients(rows, &gs);
    (0..n.saturating_sub(1)).all(|k| {
        let end = (k + beta).min(n);
        let block = projected_block(rows, &gs, k, end);
        let first: Number = (0..block.cols())
            .map(|j| &block[(0, j)] * &block[(0, j)])
            .sum();
        let by_the_crate = shortest_vector(&block).unwrap().norm2 == first;
        let by_the_test = !shorter_in_block(&mu, &r, k, end);
        assert_eq!(
            by_the_crate, by_the_test,
            "the two oracles disagree on block {k}"
        );
        by_the_test
    })
}

fn norm2(v: &[Number]) -> Number {
    v.iter().map(|x| x * x).sum()
}

/// The checks of every BKZ reduction of `b` with `opts`, run to the end.
fn check_bkz(b: &Matrix<Number>, opts: &BkzOptions) -> Reduced {
    let r = bkz(b, opts);
    check_reduction(b, &r);
    assert!(
        is_lll_reduced(r.basis(), opts.delta, opts.eta),
        "LLL-reduced"
    );
    // Run to the end: size-reduced to 1/2 whatever η, and Lovász's condition with δ = 1.
    assert!(
        is_lll_reduced(r.basis(), 1.0, 0.5),
        "size-reduced to 1/2, Lovász with δ = 1"
    );
    let rows = nonzero_rows(&r);
    assert!(
        blocks_are_reduced(&rows, opts.block_size),
        "every block reduced, β = {}",
        opts.block_size
    );
    r
}

#[test]
fn bkz_reduces_random_knapsack_and_q_ary_bases() {
    let mut rng = Rng(23);
    let sizes: &[(usize, u64)] = if cfg!(miri) {
        &[(3, 8)]
    } else {
        &[(4, 20), (6, 30), (8, 40), (10, 50), (12, 60)]
    };
    for &(n, bits) in sizes {
        for beta in [2, 3, 5, 8] {
            let k = knapsack(&mut rng, n, bits);
            check_bkz(&k, &BkzOptions::new(beta));
            let b = random_basis(&mut rng, n, n, bits / 4);
            if b.rank() == n {
                check_bkz(&b, &BkzOptions::new(beta));
            }
        }
    }
    if !cfg!(miri) {
        for (n, seed) in [(8, 1), (12, 2)] {
            check_bkz(&make("qary", n, n, 10, seed), &BkzOptions::new(6));
        }
    }
}

#[test]
fn the_block_oracles_find_blocks_that_are_not_reduced() {
    // (3, 1) − (1, 3) = (2, −2) is shorter than b₀ = (3, 1): the one block is not reduced.
    let rows = vec![
        vec![Number::from(3u8), Number::from(1u8)],
        vec![Number::from(1u8), Number::from(3u8)],
    ];
    assert!(!blocks_are_reduced(&rows, 2));
    // LLL-reduced knapsack bases whose first vector is not a shortest one: the full block is
    // not reduced, and both oracles say so (they are compared inside). LLL with δ = 3/4 leaves
    // such a first vector often (3 of the first 20 bases at δ = 0.99); bases are drawn until
    // three are found, so that a better LLL reduction cannot make the test fail.
    let mut rng = Rng(13);
    let mut found = 0;
    let tries = if cfg!(miri) { 2 } else { 300 };
    for _ in 0..tries {
        let b = knapsack(&mut rng, 14, 100);
        let r = bkz(
            &b,
            &BkzOptions::new(2).with_delta(0.75).with_max_tours(Some(0)),
        );
        let rows = nonzero_rows(&r);
        if norm2(&rows[0]) > shortest_vector(&b).unwrap().norm2 {
            assert!(!blocks_are_reduced(&rows, 14));
            found += 1;
            if found == 3 {
                break;
            }
        }
    }
    assert!(cfg!(miri) || found == 3, "{found} in {tries} bases");
}

#[test]
fn the_full_block_size_size_reduces_to_one_half() {
    // The audit's case: with η = 0.51 an earlier version left μ₂₁ ≈ −0.50285, so not HKZ.
    let b = basis(&[
        [-9, -27, -19, -5],
        [6, -31, -1, -18],
        [-14, 8, 13, 19],
        [5, -15, 18, 24],
    ]);
    let r = check_bkz(&b, &BkzOptions::new(4));
    assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
}

#[test]
fn block_size_two_satisfies_lovasz_with_delta_one() {
    // Each b*ₖ is shortest in its projected pair: |b*ₖ|² ≤ |b*ₖ₊₁|² + μ²·|b*ₖ|², Lovász's
    // condition with δ = 1, which LLL with δ = 0.99 does not promise.
    let mut rng = Rng(7);
    for n in [3, 6, 10, 14] {
        let b = knapsack(&mut rng, n, 8 * n as u64);
        let r = check_bkz(&b, &BkzOptions::new(2));
        assert!(is_lll_reduced(r.basis(), 1.0, 0.51));
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    }
}

#[test]
fn the_full_block_size_gives_a_hkz_reduced_basis() {
    let mut rng = Rng(11);
    for n in if cfg!(miri) {
        vec![3]
    } else {
        vec![4, 8, 12, 16]
    } {
        let b = knapsack(&mut rng, n, 10 * n as u64);
        for beta in [n, n + 5] {
            let r = check_bkz(&b, &BkzOptions::new(beta));
            let rows = nonzero_rows(&r);
            // b₀ is a shortest vector of the lattice, and so on down the projections: HKZ.
            assert_eq!(norm2(&rows[0]), shortest_vector(&b).unwrap().norm2);
        }
    }
}

#[test]
fn dependent_and_degenerate_bases_are_reduced() {
    // Dependent rows: zero rows first, the lattice BKZ-reduced after them.
    let b = make("combo", 6, 6, 12, 6);
    let r = check_bkz(&b, &BkzOptions::new(4));
    assert_eq!(r.rank(), 5);
    let b = random_basis(&mut Rng(3), 9, 5, 10);
    let r = check_bkz(&b, &BkzOptions::new(3));
    assert_eq!(r.rank(), 5);
    // 0 × 0, rows of zeros, no columns, one row.
    for b in [
        Matrix::<Number>::zeros(0, 0),
        Matrix::<Number>::zeros(3, 4),
        Matrix::<Number>::zeros(2, 0),
        basis(&[[0, -3, 4]]),
    ] {
        let r = bkz(&b, &BkzOptions::new(2));
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    }
    // Rows of thousands of bits.
    let b = knapsack(&mut Rng(5), 6, 1000);
    check_bkz(&b, &BkzOptions::new(4));
}

/// Bases with every kind of dependency: zero rows among the others, repeated and negated rows,
/// more rows than columns, combinations with large coefficients, rows of 300 bits.
fn dependent_bases() -> Vec<Matrix<Number>> {
    let mut rng = Rng(31);
    let mut out = Vec::new();
    for (n, m, bits) in [(4, 3, 8), (7, 4, 12), (10, 6, 20), (5, 2, 300), (9, 5, 300)] {
        out.push(random_basis(&mut rng, n, m, bits));
    }
    for (n, m, bits) in [(5, 5, 10), (8, 7, 40)] {
        let base = random_basis(&mut rng, n, m, bits);
        let c: Vec<Number> = (0..n).map(|_| rng.number(30)).collect();
        // The rows, a zero row, the first negated, the second again, and a combination.
        out.push(Matrix::from_fn(n + 4, m, |i, j| match i {
            _ if i < n => base[(i, j)].clone(),
            x if x == n => Number::default(),
            x if x == n + 1 => -&base[(0, j)],
            x if x == n + 2 => base[(1, j)].clone(),
            _ => (0..n).map(|l| &c[l] * &base[(l, j)]).sum(),
        }));
    }
    out
}

#[test]
fn dependent_rows_of_every_kind_are_reduced_by_both_reductions() {
    for b in dependent_bases()
        .into_iter()
        .take(if cfg!(miri) { 1 } else { 7 })
    {
        let rank = b.rank();
        assert!(rank < b.rows());
        // η > 1/2 takes the L² algorithm, η = 1/2 the exact LLL.
        for eta in [0.51, 0.5] {
            let r = check_bkz(&b, &BkzOptions::new(3).with_eta(eta));
            assert_eq!(r.rank(), rank);
            // The same lattice: the same shortest vector, from the rows given and from B'.
            let rows = nonzero_rows(&r);
            let lattice = Matrix::from_fn(rows.len(), b.cols(), |i, j| rows[i][j].clone());
            let s = shortest_vector(&b).unwrap();
            assert_eq!(s.norm2, shortest_vector(&lattice).unwrap().norm2);
            assert!(norm2(&rows[0]) >= s.norm2);
        }
    }
}

#[test]
fn other_parameters_are_honoured() {
    let mut rng = Rng(19);
    let b = knapsack(&mut rng, 10, 60);
    for (delta, eta) in [(0.75, 0.5), (0.9, 0.55), (0.999, 0.51), (0.3, 0.5)] {
        check_bkz(&b, &BkzOptions::new(4).with_delta(delta).with_eta(eta));
    }
    // No tour at all is the LLL reduction; one tour or more, LLL-reduced still.
    for tours in [0, 1, 2] {
        let r = bkz(&b, &BkzOptions::new(5).with_max_tours(Some(tours)));
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
    }
}

/// `log₂ x` for `x > 0`, of any size.
fn log2(x: &Number) -> f64 {
    let b = x.bits();
    let shift = b.saturating_sub(60);
    (x >> shift as usize).to_f64().log2() + shift as f64
}

/// How the first vectors compare with fplll's: the geometric mean of the ratios of the squared
/// norms (ours over fplll's), the mean root Hermite factors `(|b₀| / vol^{1/n})^{1/n}` of ours and
/// of fplll's, and the number of bases.
fn against_fplll(heavy: bool) -> (f64, f64, f64, usize) {
    let mut on = !heavy;
    let (mut log_ratio, mut ours_rhf, mut theirs_rhf, mut worst) = (0.0, 0.0, 0.0, 0.0f64);
    let mut count = 0;
    for line in FPLLL.lines() {
        if line.starts_with("# heavy") {
            on = heavy;
            continue;
        }
        if !on || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let (n, m, bits, seed, beta): (usize, usize, u64, u64, usize) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
            f[5].parse().unwrap(),
        );
        let b = make(f[0], n, m, bits, seed);
        let r = bkz(&b, &BkzOptions::new(beta));
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
        assert_eq!(r.rank(), f[7].parse::<usize>().unwrap());
        let rows = nonzero_rows(&r);
        let ours = norm2(&rows[0]);
        let theirs: Number = f[6].parse().unwrap();
        // Neither is far from the other on any one basis.
        let ratio = log2(&ours) - log2(&theirs);
        assert!(ratio.abs() <= 0.6, "{line}: ours {ours}");
        // vol² = det(B·Bᵀ), the same lattice for both.
        let rk = rows.len();
        let g: Matrix<Number> = Matrix::from_fn(rk, rk, |i, j| {
            rows[i].iter().zip(&rows[j]).map(|(x, y)| x * y).sum()
        });
        let log_vol = log2(&g.det()) / 2.0;
        let rhf = |log_b0_sq: f64| ((log_b0_sq / 2.0 - log_vol / rk as f64) / rk as f64).exp2();
        worst = worst.max(ratio.abs());
        ours_rhf += rhf(log2(&ours));
        theirs_rhf += rhf(log2(&theirs));
        log_ratio += ratio;
        count += 1;
    }
    let c = count as f64;
    println!("the largest ratio either way {:.4}", worst.exp2());
    ((log_ratio / c).exp2(), ours_rhf / c, theirs_rhf / c, count)
}

#[test]
fn the_first_vectors_are_as_short_as_fplll_finds_them() {
    if cfg!(miri) {
        return;
    }
    let (mean, ours, theirs, count) = against_fplll(false);
    println!(
        "{count} bases: |b₀|² over fplll's {mean:.4} (geometric mean); root Hermite factors {ours:.5} and fplll's {theirs:.5}"
    );
    assert_eq!(count, 36);
    // Measured: 0.9996, and 1.00366 against 1.00367. The bounds hold the quality, not the bits:
    // our blocks end exactly reduced, which implies fplll's condition (δ·|b*ₖ|² at most the
    // block's minimum), so on average the first vectors are no longer, and a change elsewhere
    // (the LLL reduction inside, say) moves single bases but not the average. 2% of |b₀|² and
    // 0.0005 of the root Hermite factor cover that spread; a BKZ short of its blocks is several
    // percent longer, and its blocks fail the exact checks of the other tests besides.
    assert!(
        mean <= 1.02 && ours <= theirs + 5e-4,
        "{mean} {ours} {theirs}"
    );
}

#[test]
#[ignore = "26 s in the debug profile (3 s in release): dimensions 30 to 60, blocks of 10 and 20"]
fn the_first_vectors_are_as_short_as_fplll_finds_them_in_larger_dimensions() {
    let (mean, ours, theirs, count) = against_fplll(true);
    println!(
        "{count} bases: |b₀|² over fplll's {mean:.4} (geometric mean); root Hermite factors {ours:.5} and fplll's {theirs:.5}"
    );
    assert_eq!(count, 16);
    // Measured: 0.9950, and 1.01255 against 1.01264; the bounds as above.
    assert!(
        mean <= 1.02 && ours <= theirs + 5e-4,
        "{mean} {ours} {theirs}"
    );
}

#[test]
#[should_panic(expected = "the block size must be at least 2, not 1")]
fn a_block_size_below_two_panics() {
    let _ = bkz(&basis(&[[1, 0], [0, 1]]), &BkzOptions::new(1));
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1), not 1")]
fn a_delta_of_one_panics() {
    let _ = bkz(
        &basis(&[[1, 0], [0, 1]]),
        &BkzOptions::new(2).with_delta(1.0),
    );
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1), not 0.25")]
fn a_delta_of_a_quarter_panics() {
    let _ = bkz(
        &basis(&[[1, 0], [0, 1]]),
        &BkzOptions::new(2).with_delta(0.25),
    );
}

#[test]
#[should_panic(expected = "η must be in [1/2, √δ), not 0.4 for δ = 0.99")]
fn an_eta_below_a_half_panics() {
    let _ = bkz(&basis(&[[1, 0], [0, 1]]), &BkzOptions::new(2).with_eta(0.4));
}

#[test]
#[should_panic(expected = "η must be in [1/2, √δ), not 0.9 for δ = 0.8")]
fn an_eta_beyond_the_root_of_delta_panics() {
    let _ = bkz(
        &basis(&[[1, 0], [0, 1]]),
        &BkzOptions::new(2).with_delta(0.8).with_eta(0.9),
    );
}
