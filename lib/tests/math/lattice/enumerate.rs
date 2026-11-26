//! Enumeration: `a::math::lattice::shortest_vector`, `closest_vector` and `short_vectors`,
//! against the references of `data/enum_*.txt`, made by `data/gen_enum.py`: for shortest
//! vectors fplll 5.5 and PARI/GP 2.19, which agree, checked up to dimension 16 by an exact
//! rational enumeration of the script's own; for closest vectors that exact enumeration, fplll's
//! CVP (not proved) only a second check; for short vectors PARI's `qfminim`. Also an exact brute
//! force on small lattices, and lattices whose short vectors are known.

use super::common::{Rng, basis, knapsack, random_basis};
use a::math::lattice::{LatticeVector, closest_vector, short_vectors, shortest_vector};
use a::math::matrix::Matrix;
use a::math::numbers::Number;

const SVP: &str = include_str!("data/enum_svp.txt");
const CVP: &str = include_str!("data/enum_cvp.txt");
const SHORT: &str = include_str!("data/enum_short.txt");

/// A basis of the reference files from its kind, shape, size and seed; `data/gen_enum.py`
/// mirrors it draw for draw.
pub fn make(kind: &str, n: usize, m: usize, bits: u64, seed: u64) -> Matrix<Number> {
    let mut rng = Rng(seed);
    match kind {
        "random" => random_basis(&mut rng, n, m, bits),
        "knapsack" => knapsack(&mut rng, n, bits),
        // floor(2^bits / (i + j + 1)): the Hilbert matrix, scaled and rounded.
        "hilbert" => Matrix::from_fn(n, n, |i, j| {
            &(Number::from(1u8) << bits as usize) / &Number::from(i + j + 1)
        }),
        // A·Aᵀ: a basis made of the rows of a Gram matrix.
        "gram" => {
            let a = random_basis(&mut rng, n, n, bits);
            &a * &a.transpose()
        }
        // [I | A; 0 | q·I], the q-ary lattice of a random A modulo q.
        "qary" => {
            let k = n / 2;
            let q = &rng.number(bits).abs() + 2u8;
            let mut b = Matrix::from_fn(n, n, |i, j| {
                if i != j {
                    Number::default()
                } else if i < k {
                    Number::from(1u8)
                } else {
                    q.clone()
                }
            });
            for i in 0..k {
                for j in k..n {
                    b[(i, j)] = rng.number(bits).abs().rem_euclid(&q);
                }
            }
            b
        }
        // n − 1 random rows and their combination with the coefficients 1, 2, …
        "combo" => {
            let base = random_basis(&mut rng, n - 1, m, bits);
            Matrix::from_fn(n, m, |i, j| {
                if i + 1 < n {
                    base[(i, j)].clone()
                } else {
                    (0..n - 1).map(|l| &base[(l, j)] * (l + 1)).sum()
                }
            })
        }
        // b₀ = (2, 0, …), bᵢ = (cᵢ, …, 2^(bits·i) at i, …) with cᵢ in 1..=4: the Gram–Schmidt
        // norms 4, 2^(2·bits), 2^(4·bits), …, beyond what an f64 can hold apart.
        "ill" => {
            let mut b = Matrix::from_fn(n, n, |i, j| {
                if i == 0 && j == 0 {
                    Number::from(2u8)
                } else if i == j {
                    Number::from(1u8) << (bits as usize * i)
                } else {
                    Number::default()
                }
            });
            for i in 1..n {
                b[(i, 0)] = &rng.number(2).abs() + 1u8;
            }
            b
        }
        _ => panic!("unknown kind {kind}"),
    }
}

/// A target of the reference files: random (`far`), a lattice vector of random coordinates of
/// `tbits` bits moved by up to 7 in each element (`near`), or a random first element and
/// `2^(tbits·j − 1)` after it (`half`).
pub fn target(kind: &str, b: &Matrix<Number>, tbits: u64, seed: u64) -> Vec<Number> {
    let mut rng = Rng(seed + 1_000_003);
    let (n, m) = b.shape();
    match kind {
        "far" => (0..m).map(|_| rng.number(tbits)).collect(),
        "near" => {
            let c: Vec<Number> = (0..n).map(|_| rng.number(tbits)).collect();
            let e: Vec<Number> = (0..m).map(|_| rng.number(3)).collect();
            (0..m)
                .map(|j| &(0..n).map(|i| &c[i] * &b[(i, j)]).sum::<Number>() + &e[j])
                .collect()
        }
        // Half-way between two multiples of each 2^(bits·i): the target of the "ill" bases.
        "half" => (0..m)
            .map(|j| {
                if j == 0 {
                    rng.number(3)
                } else {
                    Number::from(1u8) << (tbits as usize * j - 1)
                }
            })
            .collect(),
        _ => panic!("unknown target {kind}"),
    }
}

fn norm2(v: &[Number]) -> Number {
    v.iter().map(|x| x * x).sum()
}

fn diff(a: &[Number], b: &[Number]) -> Vec<Number> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The coordinates give the vector, and the norm is the vector's (or its distance to `t`).
fn check(b: &Matrix<Number>, v: &LatticeVector, t: Option<&[Number]>) {
    assert_eq!(v.coordinates.len(), b.rows());
    assert_eq!(v.vector.len(), b.cols());
    for j in 0..b.cols() {
        let e: Number = (0..b.rows()).map(|i| &v.coordinates[i] * &b[(i, j)]).sum();
        assert_eq!(e, v.vector[j], "the coordinates give the vector");
    }
    let n = match t {
        Some(t) => norm2(&diff(&v.vector, t)),
        None => norm2(&v.vector),
    };
    assert_eq!(n, v.norm2);
}

/// The first nonzero element is positive.
fn normalised(v: &[Number]) -> bool {
    v.iter()
        .find(|x| !x.is_zero())
        .is_some_and(Number::is_positive)
}

/// The lines of a reference file, without comments, as fields; stops at `# heavy` unless
/// `heavy`, where it starts.
fn lines(text: &str, heavy: bool) -> Vec<Vec<&str>> {
    let mut on = !heavy;
    let mut out = Vec::new();
    for l in text.lines() {
        if l.starts_with("# heavy") {
            on = heavy;
            continue;
        }
        if on && !l.starts_with('#') && !l.trim().is_empty() {
            out.push(l.split_whitespace().collect());
        }
    }
    out
}

fn shortest_against(heavy: bool) -> usize {
    let mut count = 0;
    for f in lines(SVP, heavy) {
        let (n, m, bits, seed) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
        );
        if cfg!(miri) && (n > 4 || count >= 4) {
            continue;
        }
        let b = make(f[0], n, m, bits, seed);
        let s = shortest_vector(&b).expect("a nonzero lattice");
        check(&b, &s, None);
        assert!(normalised(&s.vector));
        assert_eq!(s.norm2, f[5].parse::<Number>().unwrap(), "{f:?}");
        count += 1;
    }
    count
}

#[test]
fn shortest_vectors_agree_with_fplll_and_pari() {
    // Random, knapsack, Hilbert, Gram, q-ary, wide, tall and dependent bases of dimensions 1 to
    // 30 and of up to 3000 bits.
    let n = shortest_against(false);
    assert!(cfg!(miri) || n == 112, "{n} cases");
}

#[test]
#[ignore = "45 s in the debug profile: knapsack and q-ary lattices of dimension 25 to 30"]
fn shortest_vectors_of_the_heavier_sweep_agree_with_fplll_and_pari() {
    assert!(shortest_against(true) >= 26);
}

#[test]
fn closest_vectors_agree_with_the_exact_rational_enumeration() {
    let mut count = 0;
    for f in lines(CVP, false) {
        let (n, m, bits, seed): (usize, usize, u64, u64) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
        );
        if cfg!(miri) && (n > 4 || count >= 4) {
            continue;
        }
        let b = make(f[0], n, m, bits, seed);
        let t = target(f[5], &b, f[6].parse().unwrap(), seed);
        let c = closest_vector(&b, &t).unwrap();
        check(&b, &c, Some(&t));
        assert_eq!(c.norm2, f[7].parse::<Number>().unwrap(), "{f:?}");
        count += 1;
    }
    assert!(cfg!(miri) || count == 91, "{count} cases");
}

/// The sign-free checksum of the reference file: the sum over the vectors of
/// `(Σⱼ (j + 1)·vⱼ)²`.
fn checksum(vs: &[LatticeVector]) -> Number {
    vs.iter()
        .map(|v| {
            let s: Number = v.vector.iter().enumerate().map(|(j, x)| x * (j + 1)).sum();
            &s * &s
        })
        .sum()
}

#[test]
fn short_vectors_agree_with_qfminim() {
    let mut count = 0;
    for f in lines(SHORT, false) {
        let (n, m, bits, seed): (usize, usize, u64, u64) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
        );
        if cfg!(miri) && (n > 3 || count >= 4) {
            continue;
        }
        let b = make(f[0], n, m, bits, seed);
        let bound: Number = f[5].parse().unwrap();
        let s = short_vectors(&b, &bound);
        assert_eq!(s.len(), f[6].parse::<usize>().unwrap(), "{f:?}");
        let mut norms: Vec<(Number, usize)> = Vec::new();
        for v in &s {
            check(&b, v, None);
            assert!(normalised(&v.vector));
            assert!(v.norm2 <= bound && !v.norm2.is_zero());
            match norms.last_mut() {
                Some((x, c)) if *x == v.norm2 => *c += 1,
                _ => norms.push((v.norm2.clone(), 1)),
            }
        }
        let histogram: Vec<String> = norms.iter().map(|(x, c)| format!("{x}:{c}")).collect();
        let expected = if f.len() == 9 { f[7] } else { "" };
        assert_eq!(histogram.join(","), expected, "{f:?}");
        assert_eq!(
            checksum(&s),
            f[f.len() - 1].parse::<Number>().unwrap(),
            "{f:?}"
        );
        // In order of the norms, and of the elements among equal norms; no two alike.
        for w in s.windows(2) {
            assert!((&w[0].norm2, &w[0].vector) < (&w[1].norm2, &w[1].vector));
        }
        count += 1;
    }
    assert!(cfg!(miri) || count == 75, "{count} cases");
}

/// Every vector `Σ xᵢ·bᵢ` of a square basis with each `|xᵢ|` below what a vector of norm up to
/// `r2` can need (`|xᵢ| ≤ |v|·|column i of B⁻¹|`); empty if that is more than 20,000.
fn brute(b: &Matrix<Number>, r2: f64) -> Vec<Vec<Number>> {
    let n = b.rows();
    let m = b.cols();
    let (adj, det) = b.inverse().unwrap();
    let k: Vec<i64> = (0..n)
        .map(|i| {
            let c: f64 = (0..n).map(|j| adj[(j, i)].to_f64().powi(2)).sum();
            ((r2 * c).sqrt() / det.to_f64().abs()).floor() as i64 + 1
        })
        .collect();
    let mut out = Vec::new();
    if k.iter().map(|&k| 2 * k + 1).product::<i64>() > 20_000 {
        return out;
    }
    let mut x: Vec<i64> = k.iter().map(|k| -k).collect();
    loop {
        out.push(
            (0..m)
                .map(|j| (0..n).map(|i| &b[(i, j)] * x[i]).sum())
                .collect(),
        );
        let mut i = 0;
        loop {
            if i == n {
                return out;
            }
            x[i] += 1;
            if x[i] <= k[i] {
                break;
            }
            x[i] = -k[i];
            i += 1;
        }
    }
}

#[test]
fn small_lattices_agree_with_a_brute_force() {
    let mut rng = Rng(5);
    let mut checked = 0;
    for case in 0..if cfg!(miri) { 4 } else { 300 } {
        let n = 1 + case % 4;
        let b = random_basis(&mut rng, n, n, 3);
        let t: Vec<Number> = (0..n).map(|_| rng.number(3)).collect();
        if b.rank() < n {
            continue;
        }
        let shortest_row = (0..n)
            .map(|i| norm2(&(0..n).map(|j| b[(i, j)].clone()).collect::<Vec<_>>()))
            .min()
            .unwrap();
        // Room for 3·λ₁² and for every vector within the distance of the farthest target.
        let r2 = 3.0 * shortest_row.to_f64() + 400.0;
        let all = brute(&b, r2);
        if all.is_empty() {
            continue;
        }
        let best = all
            .iter()
            .map(|v| norm2(v))
            .filter(|x| !x.is_zero())
            .min()
            .unwrap();
        assert_eq!(shortest_vector(&b).unwrap().norm2, best, "case {case}");
        let bound = &best * 3u8;
        let within = all
            .iter()
            .filter(|v| {
                let x = norm2(v);
                !x.is_zero() && x <= bound
            })
            .count();
        assert_eq!(short_vectors(&b, &bound).len() * 2, within, "case {case}");
        let closest = all.iter().map(|v| norm2(&diff(v, &t))).min().unwrap();
        let c = closest_vector(&b, &t).unwrap();
        check(&b, &c, Some(&t));
        assert_eq!(c.norm2, closest, "case {case}");
        checked += 1;
    }
    println!("{checked} of 300 lattices checked by brute force");
    assert!(cfg!(miri) || checked == 204, "{checked} cases");
}

/// A basis of the root lattice E₈, doubled so that it is integral: its 240 shortest vectors have
/// the norm 8, the next 2160 the norm 16 (the theta series 1 + 240q + 2160q² + 6720q³ + …).
fn e8() -> Matrix<Number> {
    basis(&[
        [1, -1, -1, -1, -1, -1, -1, 1],
        [2, 2, 0, 0, 0, 0, 0, 0],
        [-2, 2, 0, 0, 0, 0, 0, 0],
        [0, -2, 2, 0, 0, 0, 0, 0],
        [0, 0, -2, 2, 0, 0, 0, 0],
        [0, 0, 0, -2, 2, 0, 0, 0],
        [0, 0, 0, 0, -2, 2, 0, 0],
        [0, 0, 0, 0, 0, -2, 2, 0],
    ])
}

#[test]
fn the_short_vectors_of_known_lattices_are_counted_right() {
    // ℤⁿ: the unit vectors, then the 2·n·(n − 1) of the norm 2.
    for n in 1..=6 {
        let b = Matrix::from_fn(n, n, |i, j| Number::from(u8::from(i == j)));
        assert_eq!(shortest_vector(&b).unwrap().norm2, Number::from(1u8));
        let s = short_vectors(&b, &Number::from(2u8));
        assert_eq!(s.len(), n + n * (n - 1));
        assert_eq!(s[0].vector[n - 1], Number::from(1u8), "(0, …, 0, 1) first");
    }
    // A₂ and D₄: 6 and 24 roots of the norm 2; D₄ has 24 vectors of the norm 4 too.
    let a2 = basis(&[[1, -1, 0], [0, 1, -1]]);
    assert_eq!(short_vectors(&a2, &Number::from(2u8)).len(), 3);
    let d4 = basis(&[[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 1, 1]]);
    assert_eq!(short_vectors(&d4, &Number::from(2u8)).len(), 12);
    assert_eq!(short_vectors(&d4, &Number::from(4u8)).len(), 24);
    // E₈ (doubled): 240 and 2160 vectors.
    let e = e8();
    assert_eq!(shortest_vector(&e).unwrap().norm2, Number::from(8u8));
    if !cfg!(miri) {
        let s = short_vectors(&e, &Number::from(16u8));
        assert_eq!(s.iter().filter(|v| v.norm2 == 8u8).count(), 120);
        assert_eq!(s.iter().filter(|v| v.norm2 == 16u8).count(), 1080);
        assert_eq!(s.len(), 1200);
    }
    // The same lattices given by other bases have the same vectors, in the same order.
    let u = basis(&[[1, 1], [1, 2]]);
    let a2u = &u * &a2;
    assert_eq!(
        short_vectors(&a2u, &Number::from(6u8))
            .iter()
            .map(|v| v.vector.clone())
            .collect::<Vec<_>>(),
        short_vectors(&a2, &Number::from(6u8))
            .iter()
            .map(|v| v.vector.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn degenerate_bases_are_handled() {
    let zero = Number::default;
    // 0 × 0, 0 × 3 and 3 × 0.
    for b in [
        Matrix::<Number>::zeros(0, 0),
        Matrix::<Number>::zeros(0, 3),
        Matrix::<Number>::zeros(3, 0),
    ] {
        assert!(shortest_vector(&b).is_none());
        assert!(short_vectors(&b, &Number::from(100u8)).is_empty());
        let t = vec![zero(); b.cols()];
        match closest_vector(&b, &t) {
            None => assert_eq!(b.rows(), 0),
            Some(c) => {
                assert_eq!(c.coordinates, vec![zero(); 3]);
                assert!(c.vector.is_empty() && c.norm2.is_zero());
            }
        }
    }
    // The zero lattice of rows of zeros: the closest vector is 0.
    let z = Matrix::<Number>::zeros(2, 3);
    assert!(shortest_vector(&z).is_none());
    assert!(short_vectors(&z, &Number::from(9u8)).is_empty());
    let t = [3, -4, 12].map(Number::from);
    let c = closest_vector(&z, &t).unwrap();
    check(&z, &c, Some(&t));
    assert_eq!(c.norm2, Number::from(169u8));
    // 1 × 1.
    let one = basis(&[[5]]);
    let s = shortest_vector(&one).unwrap();
    assert_eq!(
        (s.vector[0].clone(), s.norm2.clone()),
        (Number::from(5u8), Number::from(25u8))
    );
    assert_eq!(
        short_vectors(&one, &Number::from(100u8))
            .iter()
            .map(|v| v.vector[0].clone())
            .collect::<Vec<_>>(),
        [Number::from(5u8), Number::from(10u8)]
    );
    let c = closest_vector(&one, &[Number::from(-13)]).unwrap();
    assert_eq!(
        (c.vector[0].clone(), c.norm2.clone()),
        (Number::from(-15), Number::from(4u8))
    );
    // Bounds below 1.
    assert!(short_vectors(&one, &Number::default()).is_empty());
    assert!(short_vectors(&one, &Number::from(-5)).is_empty());
}

#[test]
fn dependent_and_non_square_bases_are_handled() {
    // Three rows on one line: the lattice ℤ·(1, 2).
    let b = basis(&[[2, 4], [3, 6], [-7, -14]]);
    let s = shortest_vector(&b).unwrap();
    check(&b, &s, None);
    assert_eq!(s.vector, [1, 2].map(Number::from));
    let v = short_vectors(&b, &Number::from(20u8));
    assert_eq!(v.len(), 2);
    assert_eq!(v[1].vector, [2, 4].map(Number::from));
    // A target off the line: the component across it counts.
    let t = [Number::from(5u8), Number::default()];
    let c = closest_vector(&b, &t).unwrap();
    check(&b, &c, Some(&t));
    assert_eq!(
        (c.vector.clone(), c.norm2.clone()),
        (
            vec![Number::from(1u8), Number::from(2u8)],
            Number::from(20u8)
        )
    );
    // Two rows in three dimensions, the target far outside their plane.
    let b = basis(&[[1, 0, 0], [0, 1, 0]]);
    let t = [7, -3, 1000].map(Number::from);
    let c = closest_vector(&b, &t).unwrap();
    assert_eq!(c.vector, [7, -3, 0].map(Number::from));
    assert_eq!(c.norm2, Number::from(1_000_000u32));
    // Five generators of a lattice of rank 2 in ℤ³.
    let b = basis(&[[6, 0, 3], [4, 2, 2], [0, 6, 0], [2, 2, 1], [10, 4, 5]]);
    let s = shortest_vector(&b).unwrap();
    check(&b, &s, None);
    // They all lie in the plane z = x/2, where they span (2a, 2b, a): the shortest is (0, 2, 0).
    assert_eq!(s.vector, [0, 2, 0].map(Number::from));
    assert_eq!(s.norm2, Number::from(4u8));
    assert_eq!(short_vectors(&b, &Number::from(5u8)).len(), 2);
    // Six rows of 200 bits in four dimensions: relations among them of some 800 bits, which
    // the removal of the dependencies must find. The lattice has the index 2 in ℤ⁴ (PARI's
    // `mathnf`), its minimum 1, and it holds the target (gen_enum.py's exact enumeration).
    if !cfg!(miri) {
        let b = random_basis(&mut Rng(41), 6, 4, 200);
        let s = shortest_vector(&b).unwrap();
        check(&b, &s, None);
        assert_eq!(s.norm2, Number::from(1u8));
        let t = [5, -7, 11, 13].map(Number::from);
        let c = closest_vector(&b, &t).unwrap();
        check(&b, &c, Some(&t));
        assert!(c.norm2.is_zero());
    }
}

#[test]
fn huge_and_negative_entries_are_exact() {
    let big = Number::from(1u8) << 1000;
    // Rows 2¹⁰⁰⁰ apart in one element: their difference (1, 0, −1) is the shortest vector.
    let b = Matrix::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => &big + 1u8,
        (0, 1) => -&big,
        (1, 0) => big.clone(),
        (1, 1) => -&big,
        (1, 2) => Number::from(1u8),
        (2, 2) => &big * &big,
        _ => Number::default(),
    });
    let s = shortest_vector(&b).unwrap();
    check(&b, &s, None);
    assert_eq!(s.norm2, Number::from(2u8));
    // A target of 3000 bits: the closest vector of ℤ² × 3ℤ is the target rounded.
    let b = basis(&[[1, 0, 0], [0, 1, 0], [0, 0, 3]]);
    let huge = Number::from(1u8) << 3000;
    let t = vec![-&huge, &huge + 1u8, &huge + 1u8];
    let c = closest_vector(&b, &t).unwrap();
    check(&b, &c, Some(&t));
    // 2³⁰⁰⁰ = 1 modulo 3, so 2³⁰⁰⁰ + 1 is 1 from a multiple of 3.
    assert_eq!(c.norm2, Number::from(1u8));
    // Targets on lattice vectors are at distance 0.
    let b = make("random", 6, 6, 40, 9);
    let x: Vec<Number> = (0..6).map(|i| Number::from(i as i64 * 7 - 20)).collect();
    let t: Vec<Number> = (0..6)
        .map(|j| (0..6).map(|i| &x[i] * &b[(i, j)]).sum())
        .collect();
    let c = closest_vector(&b, &t).unwrap();
    assert!(c.norm2.is_zero());
    assert_eq!(c.vector, t);
}

#[test]
fn a_target_that_floats_cannot_follow_is_found_exactly() {
    // Gram–Schmidt norms 4, 2⁵⁰⁰ and 2¹⁰⁰⁰, and a target half way at every level: the radius of
    // 2⁹⁹⁸ hides the level of 4 from any float, and the search runs in integers. The closest
    // vectors take one of the two upper rows, and are 2⁴⁹⁸ + 2⁹⁹⁸ away; Babai's point is 1
    // further.
    let p = |e: usize| Number::from(1u8) << e;
    let b = Matrix::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => Number::from(2u8),
        (1, 0) | (2, 0) => Number::from(1u8),
        (1, 1) => p(250),
        (2, 2) => p(500),
        _ => Number::default(),
    });
    let t = [Number::from(1u8), p(249), p(499)];
    let c = closest_vector(&b, &t).unwrap();
    check(&b, &c, Some(&t));
    assert_eq!(c.norm2, &p(498) + &p(998));
    assert_eq!(&c.coordinates[1] + &c.coordinates[2], Number::from(1u8));
}

#[test]
#[should_panic(expected = "the target has 2 elements, the basis 3 columns")]
fn a_target_of_the_wrong_length_panics() {
    let _ = closest_vector(
        &basis(&[[1, 0, 0]]),
        &[Number::default(), Number::default()],
    );
}
