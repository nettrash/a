//! LLL exactly: `a::math::lattice::lll` and `is_lll_reduced`; and the helpers that the tests of
//! `lll_fp` (l2.rs) and `deep_lll` (deep.rs) share with them: a check of reducedness from the
//! definitions in rationals, bases whose reduced basis is unique, rank-deficient bases, and the
//! reference data of fplll and FLINT (data/lll_reference.txt, from data/gen_lll_reference.py).

use super::common::{Rng, basis, check_reduction, knapsack, random_basis};
use a::math::lattice::{LllOptions, Reduced, is_lll_reduced, lll};
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::math::rational::Rational;

/// `n` rows of `m` columns and rank at most `r`: combinations, with coefficients of `cbits` bits,
/// of `r` random rows of `bits` bits (the rows first, then the coefficients, from `rng`).
pub fn dependent_basis(
    rng: &mut Rng,
    n: usize,
    r: usize,
    m: usize,
    bits: u64,
    cbits: u64,
) -> Matrix<Number> {
    let v = random_basis(rng, r, m, bits);
    let c = random_basis(rng, n, r, cbits);
    &c * &v
}

/// `|row i|²`.
pub fn norm2(b: &Matrix<Number>, i: usize) -> Number {
    (0..b.cols()).map(|j| &b[(i, j)] * &b[(i, j)]).sum()
}

/// `|b|²` of the first nonzero row of a reduction.
pub fn first_norm2(r: &Reduced) -> Number {
    norm2(r.basis(), r.basis().rows() - r.rank())
}

/// The Gram–Schmidt orthogonalisation of the rows of `b` in rationals, from the definition:
/// `b*ᵢ = bᵢ − Σ μᵢⱼ·b*ⱼ`, `μᵢⱼ = ⟨bᵢ, b*ⱼ⟩/|b*ⱼ|²` (zero where `b*ⱼ = 0`); the squared norms
/// `|b*ᵢ|²` and the coefficients.
pub fn rational_gso(b: &Matrix<Number>) -> (Vec<Rational>, Vec<Vec<Rational>>) {
    let (n, m) = b.shape();
    let zero = Rational::from(0);
    let mut star: Vec<Vec<Rational>> = Vec::with_capacity(n);
    let mut norms: Vec<Rational> = Vec::with_capacity(n);
    let mut mu = vec![vec![zero.clone(); n]; n];
    for i in 0..n {
        let mut v: Vec<Rational> = (0..m).map(|j| Rational::from(&b[(i, j)])).collect();
        for j in 0..i {
            if norms[j].is_zero() {
                continue;
            }
            let d: Rational = (0..m).map(|k| &star[j][k] * &b[(i, k)]).sum();
            let c = &d / &norms[j];
            for k in 0..m {
                v[k] -= &c * &star[j][k];
            }
            mu[i][j] = c;
        }
        norms.push(v.iter().map(|x| x * x).sum());
        star.push(v);
    }
    (norms, mu)
}

/// Whether the rows of `b` are reduced by the definitions, in rationals: zero rows first, the
/// others independent, `|μᵢⱼ| ≤ η`, and `|πᵢ(bₖ)|² ≥ δ·|b*ᵢ|²` at every `i < k` with
/// `checked(i, k)`, where `|πᵢ(bₖ)|² = Σ_{i ≤ j ≤ k} μ²ₖⱼ·|b*ⱼ|²` (`μₖₖ = 1`) counts the rows
/// of the lattice from its first; `i = k − 1` is Lovász's condition.
pub fn reduced_by_definition(
    b: &Matrix<Number>,
    delta: f64,
    eta: f64,
    checked: impl Fn(usize, usize) -> bool,
) -> bool {
    let zeros = (0..b.rows())
        .take_while(|&i| (0..b.cols()).all(|j| b[(i, j)].is_zero()))
        .count();
    let rest = Matrix::from_fn(b.rows() - zeros, b.cols(), |i, j| b[(i + zeros, j)].clone());
    let (norms, mu) = rational_gso(&rest);
    let (d, e) = (
        Rational::try_from(delta).unwrap(),
        Rational::try_from(eta).unwrap(),
    );
    let n = rest.rows();
    if norms.iter().any(Rational::is_zero) {
        return false;
    }
    for k in 0..n {
        if mu[k][..k].iter().any(|m| m.abs() > e) {
            return false;
        }
        let mut s = norms[k].clone();
        for i in (0..k).rev() {
            s += &(&mu[k][i] * &mu[k][i]) * &norms[i];
            if (i + 1 == k || checked(i, k)) && s < &d * &norms[i] {
                return false;
            }
        }
    }
    true
}

/// A basis of a lattice whose `(δ, η)`-LLL-reduced basis, for any `δ ≤ 1` and `η < 0.6`, is
/// unique up to the signs of its rows, and that basis `T`: `T` is lower triangular, its diagonal
/// `10·100ⁱ`, the entries below it multiples of their column's `100ʲ` by at most 4 (so
/// `|μᵢⱼ| ≤ 0.4`). The basis is `U·[T; 0]` with `zeros` zero rows and `U` unimodular, random:
/// row additions and exchanges. A reduced basis must hold `±t₀` first (every other vector is 100
/// times longer, which no LLL bound of fewer than 15 dimensions allows), then, by induction on
/// the projections, `±tᵢ` plus a combination of the rows before, which `|μ| ≤ η` makes zero.
pub fn unique_basis(rng: &mut Rng, n: usize, zeros: usize) -> (Matrix<Number>, Matrix<Number>) {
    let hundred = Number::from(100u8);
    let t = Matrix::from_fn(n, n, |i, j| {
        if j > i {
            Number::from(0u8)
        } else if j == i {
            Number::from(10u8) * hundred.pow(i as u32)
        } else {
            Number::from((rng.next() % 9) as i64 - 4) * hundred.pow(j as u32)
        }
    });
    let size = n + zeros;
    let mut rows: Vec<Vec<Number>> = (0..size)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i < n {
                        t[(i, j)].clone()
                    } else {
                        Number::from(0u8)
                    }
                })
                .collect()
        })
        .collect();
    if size > 1 {
        for _ in 0..4 * size {
            let i = (rng.next() % size as u64) as usize;
            let j = (rng.next() % size as u64) as usize;
            if i == j {
                continue;
            }
            if rng.next().is_multiple_of(4) {
                rows.swap(i, j);
            } else {
                let k = Number::from((rng.next() % 5) as i64 - 2);
                let add: Vec<Number> = rows[j].iter().map(|x| &k * x).collect();
                for (x, y) in rows[i].iter_mut().zip(add) {
                    *x += y;
                }
            }
        }
    }
    (Matrix::from_fn(size, n, |i, j| rows[i][j].clone()), t)
}

/// Whether the rows of `got` after `zeros` zero rows are those of `t`, each up to its sign.
pub fn is_t_up_to_signs(got: &Matrix<Number>, t: &Matrix<Number>, zeros: usize) -> bool {
    let m = t.cols();
    got.rows() == t.rows() + zeros
        && (0..zeros).all(|i| (0..m).all(|j| got[(i, j)].is_zero()))
        && (0..t.rows()).all(|i| {
            let same = (0..m).all(|j| got[(i + zeros, j)] == t[(i, j)]);
            let neg = (0..m).all(|j| got[(i + zeros, j)] == -t[(i, j)].clone());
            same || neg
        })
}

/// A case of data/lll_reference.txt: the basis, its parameters, the rank and the squared norm of
/// the first nonzero row that fplll and FLINT reduce it to.
pub struct Case {
    pub kind: String,
    pub n: usize,
    pub bits: u64,
    pub seed: u64,
    pub delta: f64,
    pub eta: f64,
    pub rank: usize,
    pub fplll: Number,
    pub flint: Number,
}

impl Case {
    /// The basis, generated as the generator script generates it.
    pub fn basis(&self) -> Matrix<Number> {
        let mut rng = Rng(self.seed);
        match self.kind.as_str() {
            "random" => random_basis(&mut rng, self.n, self.n, self.bits),
            "knapsack" => knapsack(&mut rng, self.n, self.bits),
            "dependent" => dependent_basis(&mut rng, self.n, self.n - 2, self.n, self.bits, 4),
            k => panic!("unknown kind {k}"),
        }
    }
}

/// The cases of data/lll_reference.txt.
pub fn reference_cases() -> Vec<Case> {
    include_str!("data/lll_reference.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            Case {
                kind: f[0].to_string(),
                n: f[1].parse().unwrap(),
                bits: f[2].parse().unwrap(),
                seed: f[3].parse().unwrap(),
                delta: f[4].parse().unwrap(),
                eta: f[5].parse().unwrap(),
                rank: f[6].parse().unwrap(),
                fplll: f[7].parse().unwrap(),
                flint: f[8].parse().unwrap(),
            }
        })
        .collect()
}

/// `log₂(a/b)` for positive `a`, `b`.
pub fn log2_ratio(a: &Number, b: &Number) -> f64 {
    let shift = a.bits().max(b.bits()).saturating_sub(60) as usize;
    ((a >> shift).to_f64() / (b >> shift).to_f64()).log2()
}

#[test]
fn a_small_basis_is_reduced_and_transformed() {
    let b = basis(&[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]);
    let r = lll(&b, &LllOptions::new());
    check_reduction(&b, &r);
    assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
    // The lattice has determinant ±3, which the reduction keeps.
    assert_eq!(r.basis().det().abs(), Number::from(3u8));
    // fplll and FLINT give this basis (gen_lll_flint.c's example).
    assert_eq!(*r.basis(), basis(&[[0, 1, 0], [1, 0, 1], [-1, 0, 2]]));
}

#[test]
fn random_and_knapsack_bases_are_reduced() {
    let mut rng = Rng(23);
    for (n, bits) in [(2, 10), (3, 30), (5, 20), (8, 40), (10, 16)] {
        let b = random_basis(&mut rng, n, n, bits);
        for delta in [0.3, 0.75, 0.99, 1.0] {
            let r = lll(&b, &LllOptions::new().with_delta(delta));
            check_reduction(&b, &r);
            assert!(
                is_lll_reduced(r.basis(), delta, 0.5),
                "n = {n}, δ = {delta}"
            );
            if n <= 5 {
                assert!(reduced_by_definition(r.basis(), delta, 0.5, |_, _| false));
            }
        }
        let k = knapsack(&mut rng, n, bits * 2);
        let r = lll(&k, &LllOptions::new());
        check_reduction(&k, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
    }
}

#[test]
fn the_check_refuses_what_is_not_reduced() {
    assert!(!is_lll_reduced(&basis(&[[1, 0], [5, 1]]), 0.99, 0.51));
    // Lovász's condition fails: |b*₁|² = 1/4·|b*₀|² at μ = 0.
    assert!(!is_lll_reduced(&basis(&[[2, 0], [0, 1]]), 0.99, 0.51));
    assert!(is_lll_reduced(&basis(&[[0, 1], [2, 0]]), 0.99, 0.51));
    // Zero rows only first.
    assert!(is_lll_reduced(&basis(&[[0, 0], [1, 0]]), 0.99, 0.51));
    assert!(!is_lll_reduced(&basis(&[[1, 0], [0, 0]]), 0.99, 0.51));
    assert!(!is_lll_reduced(
        &basis(&[[0, 0], [1, 0], [0, 0]]),
        0.99,
        0.51
    ));
    // Dependent rows that are not zero.
    assert!(!is_lll_reduced(&basis(&[[1, 0], [1, 0]]), 0.99, 0.51));
    // At the boundary: |b*₁|² = 1 = δ·|b*₀|² for δ = 1/4 exactly, and below it beyond.
    assert!(is_lll_reduced(&basis(&[[2, 0], [0, 1]]), 0.25, 0.51));
    assert!(!is_lll_reduced(
        &basis(&[[2, 0], [0, 1]]),
        0.25 + 1e-9,
        0.51
    ));
    // μ = 1/2 exactly: reduced for η = 1/2, not for any η below.
    assert!(is_lll_reduced(&basis(&[[2, 0], [1, 2]]), 0.99, 0.5));
    assert!(!is_lll_reduced(&basis(&[[2, 0], [1, 2]]), 0.99, 0.4999));
    // No rows, no columns, all zero.
    assert!(is_lll_reduced(&Matrix::<Number>::zeros(0, 0), 0.99, 0.51));
    assert!(is_lll_reduced(&Matrix::<Number>::zeros(0, 3), 0.99, 0.51));
    assert!(is_lll_reduced(&Matrix::<Number>::zeros(3, 0), 0.99, 0.51));
    assert!(is_lll_reduced(&Matrix::<Number>::zeros(2, 2), 0.99, 0.51));
}

#[test]
fn the_check_agrees_with_the_definition_in_rationals() {
    let mut rng = Rng(77);
    let mut seen = [0usize; 2];
    for round in 0..40 {
        let n = 2 + round % 5;
        let b = random_basis(&mut rng, n, n, 6 + (round % 3) as u64 * 10);
        let (delta, eta) = [(0.99, 0.51), (0.75, 0.5), (0.5, 0.7), (0.3, 0.52)][round % 4];
        // A reduced basis, and the same with a row or a coefficient spoiled.
        let r = lll(&b, &LllOptions::new().with_delta(delta));
        let mut spoiled = r.basis().clone();
        let (i, j) = (round % n, (round / n) % n);
        if i != j {
            spoiled.add_scaled_row(i, j, &Number::from(1 + (round % 2) as i64));
        } else {
            spoiled.swap_rows(0, n - 1);
        }
        for m in [&b, r.basis(), &spoiled] {
            let exact = is_lll_reduced(m, delta, eta);
            assert_eq!(
                exact,
                reduced_by_definition(m, delta, eta, |_, _| false),
                "{m:?} δ = {delta} η = {eta}"
            );
            seen[usize::from(exact)] += 1;
        }
    }
    // Both answers were tested, many times.
    assert!(seen[0] > 20 && seen[1] > 20, "{seen:?}");
}

#[test]
fn dependent_rows_come_out_as_zero_rows_first() {
    let o = LllOptions::new();
    // Repeated rows, multiples, and Euclid's algorithm on 3 and 5.
    for (b, rank) in [
        (basis(&[[1, 2], [1, 2]]), 1),
        (basis(&[[1, 2], [2, 4], [3, 6]]), 1),
        (basis(&[[3, 0], [5, 0]]), 1),
        (basis(&[[4, 2], [6, 3], [2, 0]]), 2),
        (basis(&[[0, 0, 0], [1, 2, 3], [0, 0, 0], [2, 4, 6]]), 1),
        (basis(&[[0, 0], [0, 0]]), 0),
        (
            basis(&[[6, 10, 15], [10, 15, 6], [15, 6, 10], [1, 1, 1]]),
            3,
        ),
    ] {
        let r = lll(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(r.rank(), rank, "{b:?}");
        assert!(is_lll_reduced(r.basis(), 0.99, 0.5), "{:?}", r.basis());
    }
    // 3 and 5 make the lattice ℤ: the reduced row is ±1.
    let r = lll(&basis(&[[3, 0], [5, 0]]), &o);
    assert_eq!(r.basis()[(1, 0)].abs(), Number::from(1u8));
    // Rank-deficient bases of many shapes, and bases with more rows than columns.
    let mut rng = Rng(5);
    for (n, rk, m, bits) in [
        (4, 2, 4, 20),
        (6, 3, 5, 30),
        (7, 6, 8, 12),
        (5, 1, 3, 80),
        (9, 4, 4, 10),
    ] {
        let b = dependent_basis(&mut rng, n, rk, m, bits, 4);
        let r = lll(&b, &o);
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
        assert!(reduced_by_definition(r.basis(), 0.99, 0.5, |_, _| false));
    }
    for n in [3, 6] {
        let b = random_basis(&mut rng, n + 3, n, 30);
        let r = lll(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(r.rank(), n);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
    }
}

#[test]
fn empty_and_single_row_shapes() {
    let o = LllOptions::new();
    for (n, m) in [(0, 0), (0, 3), (3, 0), (1, 0)] {
        let b = Matrix::<Number>::zeros(n, m);
        let r = lll(&b, &o);
        assert_eq!(r.basis().shape(), (n, m));
        assert_eq!(*r.transform(), Matrix::<Number>::identity(n));
        assert_eq!(r.rank(), 0);
    }
    for row in [[0, 0, 0, 0], [0, -7, 0, 3], [5, 0, 0, 0]] {
        let b = basis(&[row]);
        let r = lll(&b, &o);
        check_reduction(&b, &r);
        assert_eq!(*r.basis(), b, "one row is reduced as it is");
    }
    let r = lll(&basis(&[[-9]]), &o);
    assert_eq!((r.basis()[(0, 0)].clone(), r.rank()), (Number::from(-9), 1));
}

#[test]
fn delta_one_ends_with_a_reduced_basis() {
    let mut rng = Rng(31);
    let o = LllOptions::new().with_delta(1.0).with_eta(0.5);
    for (n, bits) in [(2, 40), (3, 20), (4, 12), (6, 10)] {
        for b in [
            random_basis(&mut rng, n, n, bits),
            knapsack(&mut rng, n, bits),
        ] {
            let r = lll(&b, &o);
            check_reduction(&b, &r);
            assert!(is_lll_reduced(r.basis(), 1.0, 0.5));
        }
    }
    // In two dimensions δ = 1 is Lagrange's reduction: the first row is a shortest vector.
    let b = basis(&[[1, 0], [1000, 1]]);
    let r = lll(&b, &o);
    assert_eq!(norm2(r.basis(), 0), Number::from(1u8));
    let b = basis(&[[31, 59], [26, 53]]);
    let r = lll(&b, &o);
    let shortest = (-60i64..=60)
        .flat_map(|x| (-60i64..=60).map(move |y| (x, y)))
        .filter(|&p| p != (0, 0))
        .map(|(x, y)| (31 * x + 26 * y).pow(2) + (59 * x + 53 * y).pow(2))
        .min()
        .unwrap();
    assert_eq!(norm2(r.basis(), 0), Number::from(shortest));
}

#[test]
fn huge_entries_are_reduced_exactly() {
    let mut rng = Rng(41);
    let o = LllOptions::new();
    for (b, bits) in [
        (knapsack(&mut rng, 6, 2000), 2000),
        (random_basis(&mut rng, 3, 3, 3000), 3000),
        (dependent_basis(&mut rng, 5, 3, 5, 1000, 8), 1000),
        (knapsack(&mut rng, 12, 600), 600),
    ] {
        let r = lll(&b, &o);
        check_reduction(&b, &r);
        assert!(is_lll_reduced(r.basis(), 0.99, 0.5), "{bits} bits");
    }
}

#[test]
fn the_reduced_basis_is_the_unique_one_where_there_is_one() {
    let mut rng = Rng(9);
    for (n, zeros) in [(1, 0), (2, 0), (2, 1), (3, 0), (4, 2), (6, 0), (8, 1)] {
        let (b, t) = unique_basis(&mut rng, n, zeros);
        for delta in [0.75, 0.99, 1.0] {
            let r = lll(&b, &LllOptions::new().with_delta(delta));
            check_reduction(&b, &r);
            assert!(is_t_up_to_signs(r.basis(), &t, zeros), "{:?}", r.basis());
        }
    }
}

#[test]
fn first_vectors_match_fplll_and_flint() {
    // fplll and FLINT reduce in floating point with η = 0.51; `lll` reduces exactly to 1/2. The
    // first vectors of LLL-reduced bases agree within the LLL bound, and in practice nearly.
    let (mut same, mut count, mut sum) = (0, 0, 0.0);
    for c in reference_cases().iter().filter(|c| c.n <= 12) {
        let b = c.basis();
        let r = lll(&b, &LllOptions::new().with_delta(c.delta).with_eta(c.eta));
        assert_eq!(r.rank(), c.rank, "{} {} {}", c.kind, c.n, c.bits);
        assert!(is_lll_reduced(r.basis(), c.delta, 0.5));
        let ours = first_norm2(&r);
        let x = log2_ratio(&ours, &c.fplll);
        // |b₀|² ≤ (δ − 1/4)^-(n−1)·λ₁² bounds both: their ratio is within that factor.
        let bound = -((c.delta - 0.2601f64).log2()) * (c.rank as f64 - 1.0);
        assert!(x.abs() <= bound + 1e-9, "{} {}: ratio 2^{x}", c.kind, c.n);
        same += usize::from(ours == c.fplll || ours == c.flint);
        count += 1;
        sum += x;
    }
    assert!(count >= 40);
    // The same first vector in most cases, and no worse on average.
    assert!(same * 10 >= count * 7, "{same} of {count}");
    assert!(
        sum / count as f64 <= 0.1,
        "mean log₂ ratio {}",
        sum / count as f64
    );
}

#[test]
fn the_reduction_is_to_one_half_whatever_eta() {
    let mut rng = Rng(3);
    for eta in [0.5, 0.51, 0.9] {
        let b = knapsack(&mut rng, 8, 40);
        let r = lll(&b, &LllOptions::new().with_eta(eta));
        assert!(is_lll_reduced(r.basis(), 0.99, 0.5));
    }
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1]")]
fn delta_of_a_quarter_panics() {
    lll(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(0.25),
    );
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1]")]
fn delta_above_one_panics() {
    lll(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(1.01),
    );
}

#[test]
#[should_panic(expected = "δ must be in (1/4, 1]")]
fn delta_nan_panics() {
    lll(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(f64::NAN),
    );
}

#[test]
#[should_panic(expected = "η must be in [1/2, √δ)")]
fn eta_below_one_half_panics() {
    lll(&basis(&[[1, 0], [0, 1]]), &LllOptions::new().with_eta(0.49));
}

#[test]
#[should_panic(expected = "η must be in [1/2, √δ)")]
fn eta_at_the_root_of_delta_panics() {
    // η² = δ exactly.
    lll(
        &basis(&[[1, 0], [0, 1]]),
        &LllOptions::new().with_delta(0.5625).with_eta(0.75),
    );
}

#[test]
#[should_panic(expected = "δ is finite")]
fn the_check_panics_on_a_delta_that_is_not_finite() {
    is_lll_reduced(&basis(&[[1, 0], [0, 1]]), f64::INFINITY, 0.51);
}
