//! Dixon's p-adic lifting, which `solve` takes for systems of at least 24 rows (16 when an
//! integer passes 63 bits) with at most an eighth as many right-hand sides: against FLINT's
//! `fmpq_mat_solve_fmpz_mat_dixon` and `fmpq_mat_solve_dixon` on the seeded systems of
//! `data/dixon_flint.txt` (written by `data/gen_dixon.c`), up to 100 × 100 and to elements of
//! 1000 bits; every solution also satisfies its system exactly. Singular and rank-deficient
//! systems, proved singular by the search for a prime with few and with many right-hand sides;
//! systems singular modulo the first primes the lifting tries; the branches of Dixon's
//! determinant for a prime that divides the divisor and for one that divides the determinant
//! alone; and systems without a right-hand side.

use super::common::{Rng, size};
use a::math::matrix::Matrix;
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;

/// An element of `bits` bits as `data/gen_dixon.c` draws it.
fn entry(r: &mut Rng, bits: usize) -> Number {
    let bytes = bits.div_ceil(8);
    let mut v: Vec<u8> = (0..bytes).map(|_| r.next() as u8).collect();
    if !bits.is_multiple_of(8) {
        v[bytes - 1] &= (1u8 << (bits % 8)) - 1;
    }
    let x = Number::new(v, ByteOrder::LittleEndian);
    if r.next() & 1 == 1 { -x } else { x }
}

/// A rational as `data/gen_dixon.c` draws it.
fn rational(r: &mut Rng, bits: usize) -> Rational {
    let n = entry(r, bits);
    Rational::new(n, Number::from(1 + r.next() % 1000))
}

const Q: u64 = (1 << 61) - 1;

/// `x mod Q` in `[0, Q)`.
fn residue(x: &Number) -> u64 {
    let q = Number::from(Q);
    u64::try_from(&((x % &q + &q) % &q)).unwrap()
}

/// The least common multiple of the denominators modulo `Q`, its bits, and the digest of
/// `data/gen_dixon.c`.
fn digest(x: &Matrix<Rational>) -> (u64, u64, u64) {
    let mut l = Number::from(1);
    let mut h: u128 = 0;
    for e in x.as_slice() {
        l = l.lcm(e.denom());
        h = (h * 1_000_003 + residue(e.numer()) as u128 + 7 * residue(e.denom()) as u128)
            % Q as u128;
    }
    (residue(&l), l.bits(), h as u64)
}

#[test]
fn solutions_agree_with_flint() {
    let data = include_str!("data/dixon_flint.txt");
    for line in data.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let num = |i: usize| f[i].parse::<u64>().unwrap();
        let (seed, n, k, bits) = (num(1), num(2) as usize, num(3) as usize, num(4) as usize);
        // The debug profile's budget; everything in the release profile.
        if cfg!(miri) && n > 16 || cfg!(debug_assertions) && n * n * bits > 1_200_000 {
            continue;
        }
        let mut r = Rng(seed);
        let x = if f[0] == "q" {
            let a = Matrix::from_rows(
                &(0..n)
                    .map(|_| (0..n).map(|_| rational(&mut r, bits)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            let b = Matrix::from_rows(
                &(0..n)
                    .map(|_| (0..k).map(|_| rational(&mut r, bits)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            let x = a.solve(&b);
            if let Some(x) = &x {
                assert_eq!(&a * x, b, "{line}");
            }
            x
        } else {
            let mut rows: Vec<Vec<Number>> = (0..n)
                .map(|_| (0..n).map(|_| entry(&mut r, bits)).collect())
                .collect();
            let b = Matrix::from_rows(
                &(0..n)
                    .map(|_| (0..k).map(|_| entry(&mut r, bits)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
            if f[0] == "s" {
                rows[n - 1] = (0..n).map(|j| &rows[0][j] + &rows[1][j]).collect();
            }
            let a = Matrix::from_rows(&rows);
            let x = a.solve(&b);
            if let Some(x) = &x {
                let aq = a.map(|v| Rational::from(v.clone()));
                assert_eq!(&aq * x, b.map(|v| Rational::from(v.clone())), "{line}");
            }
            x
        };
        match x {
            None => assert_eq!(f[5], "none", "{line}"),
            Some(x) => assert_eq!(digest(&x), (num(5), num(6), num(7)), "{line}"),
        }
    }
}

/// A random `m × n` matrix of numbers of `bits` bits.
fn random(r: &mut Rng, m: usize, n: usize, bits: usize) -> Matrix<Number> {
    Matrix::from_fn(m, n, |_, _| entry(r, bits))
}

#[test]
fn rank_deficient_systems_have_no_solution() {
    let mut r = Rng(0xD1_5DEF);
    let n = size(30, 6);
    for (rank, bits) in [(n - 1, 8), (n - 3, 64), (1, 200)] {
        let a = &random(&mut r, n, rank, bits) * &random(&mut r, rank, n, bits);
        let b = random(&mut r, n, 1, bits);
        assert_eq!(a.solve(&b), None, "rank {rank}");
        let aq = a.map(|x| Rational::from(x.clone()) / 7u32);
        assert_eq!(
            aq.solve(&b.map(|x| Rational::from(x.clone()))),
            None,
            "rank {rank}"
        );
        // A consistent right-hand side does not make a singular system solvable.
        let c = &a * &random(&mut r, n, 1, 8);
        assert_eq!(a.solve(&c), None, "rank {rank}");
    }
}

/// The four primes the lifting tries first, the largest below `2^62`, and the two after them.
const PRIMES: [u64; 6] = [
    0x3fff_ffff_ffff_ffc7,
    0x3fff_ffff_ffff_ffa9,
    0x3fff_ffff_ffff_ff8b,
    0x3fff_ffff_ffff_ff71,
    0x3fff_ffff_ffff_ff67,
    0x3fff_ffff_ffff_ff59,
];

#[test]
fn a_system_singular_modulo_the_first_primes_is_lifted_modulo_the_next() {
    // det(A) = p₁·p₂·p₃·p₄ for the four primes the lifting tries first, so that A is singular
    // modulo each of them and nonsingular: the search goes on to the fifth prime and lifts
    // modulo it. A = L·D·U with L and U unit triangular and D = diag(p₁·p₂·p₃·p₄, 1, …, 1).
    let mut r = Rng(0xD1_9417);
    let n = size(24, 5);
    let mut l = Matrix::<Number>::identity(n);
    let mut u = Matrix::<Number>::identity(n);
    for i in 0..n {
        for j in 0..i {
            l[(i, j)] = Number::from(r.next() % 7) - Number::from(3);
            u[(j, i)] = Number::from(r.next() % 7) - Number::from(3);
        }
    }
    let mut d = Matrix::<Number>::identity(n);
    d[(0, 0)] = PRIMES[..4]
        .iter()
        .fold(Number::from(1), |x, &p| x * Number::from(p));
    let a = &(&l * &d) * &u;
    assert_eq!(a.det(), d[(0, 0)]);
    let b = random(&mut r, n, 2, 64);
    let x = a.solve(&b).expect("a nonsingular system");
    assert_eq!(
        &a.map(|v| Rational::from(v.clone())) * &x,
        b.map(|v| Rational::from(v.clone()))
    );
}

/// `A` with row `n - 1` the sum of rows 0 and 1, of `bits` bits.
fn singular(r: &mut Rng, n: usize, bits: usize) -> Matrix<Number> {
    let mut a = random(r, n, n, bits);
    for j in 0..n {
        a[(n - 1, j)] = &a[(0, j)] + &a[(1, j)];
    }
    a
}

#[test]
fn singular_systems_are_proved_singular() {
    // Singular modulo every prime: the search proves it by a vector of the null space, with
    // few right-hand sides (the lifting) and with many (which the elimination would take), for
    // integers and for rationals, and the inverses.
    let mut r = Rng(0xD1_0515);
    let n = size(40, 6);
    for bits in [8, 200] {
        let a = singular(&mut r, n, bits);
        for k in [1, n / 8 + 1] {
            let b = random(&mut r, n, k, bits);
            assert_eq!(a.solve(&b), None, "{bits} bits, {k} columns");
            let aq = a.map(|x| Rational::from(x.clone()) / 3u32);
            let bq = b.map(|x| Rational::from(x.clone()));
            assert_eq!(aq.solve(&bq), None, "{bits} bits, {k} columns");
        }
        assert_eq!(a.inverse(), None);
        assert_eq!(a.inverse_rational(), None);
        assert_eq!(a.map(|x| Rational::from(x.clone())).inverse(), None);
        assert!(a.det().is_zero());
    }
    // A zero row: Hadamard's bound is 0 once the first primes have failed.
    let mut z = random(&mut r, n, n, 8);
    for j in 0..n {
        z[(7, j)] = Number::default();
    }
    assert_eq!(z.solve(&random(&mut r, n, 1, 8)), None);
    assert!(z.det().is_zero());
}

#[test]
fn a_singular_system_whose_rank_drops_modulo_a_prime_is_proved_singular() {
    // A = B·C of rank `rank` over the rationals, but of rank `rank - 1` modulo the fourth prime,
    // which divides the last column of B: the null vector from that prime's rank profile does
    // not satisfy A·w = 0, and the search proves A singular with the fifth prime, of full rank
    // `rank` again.
    let mut r = Rng(0xD1_0D20);
    let n = size(30, 6);
    let rank = n - 5;
    let mut b = random(&mut r, n, rank, 8);
    for i in 0..n {
        b[(i, rank - 1)] = &b[(i, rank - 1)] * &Number::from(PRIMES[3]);
    }
    let c = random(&mut r, rank, n, 8);
    let a = &b * &c;
    let x = random(&mut r, n, 1, 8);
    assert_eq!(a.solve(&x), None);
    assert_eq!(a.solve(&random(&mut r, n, n / 2, 8)), None);
    assert!(a.det().is_zero());
    assert_eq!(a.rank(), rank);
}

/// The fixed right-hand side of the lifting of Dixon's determinant: xorshift64 from
/// `0x9E3779B97F4A7C15`, each element the next word modulo 19, less 9.
fn divisor_rhs(n: usize) -> Vec<Number> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            Number::from((state % 19) as i64 - 9)
        })
        .collect()
}

/// `a` with its element `(i, j)` changed so that the second prime of the lifting (the first
/// whose residue of `det(A) / l` is taken) divides the determinant: `det` is linear in the
/// element, `x·c + d₀` with the cofactor `c`. The changed matrix and its determinant.
fn divisible(mut a: Matrix<Number>, i: usize, j: usize) -> (Matrix<Number>, Number) {
    let q = Number::from(PRIMES[1]);
    a[(i, j)] = Number::default();
    let d0 = a.det();
    a[(i, j)] = Number::from(1);
    let c = &a.det() - &d0;
    let x = (&(-&d0).rem_euclid(&q) * &c.rem_euclid(&q).mod_inverse(&q).unwrap()).rem_euclid(&q);
    a[(i, j)] = x.clone();
    let det = &(&x * &c) + &d0;
    assert!(det.rem_euclid(&q).is_zero() && !det.is_zero());
    (a, det)
}

#[test]
fn determinants_whose_primes_divide_the_divisor_or_only_the_determinant() {
    // Dixon's divisor `l` of the determinant, from the solution of `A·x = b` for its fixed `b`,
    // then the residues of `det / l` modulo the primes after the one of the lifting while their
    // product is within twice the bound over `l`. With the first column `p₂·b`, `x = e₁ / p₂`,
    // `l = p₂`, and the second prime, the first of the residues, divides `l`: it is passed
    // over (32 rows of integers of up to 66 bits, the divisor's band). With the first column `b`,
    // `x = e₁`, `l = 1`, and an element changed so that the second prime divides the determinant
    // alone: `A` is singular modulo it, and the residue of `det / l` there is 0 (28 rows of up to
    // 62 bits).
    let mut r = Rng(0xD1_DE7D);
    let small = |r: &mut Rng| Number::from((r.next() % 19) as i64 - 9);
    let n = 32;
    let rhs = divisor_rhs(n);
    let p2 = Number::from(PRIMES[1]);
    let c = Matrix::from_fn(n, n, |i, j| {
        if j == 0 {
            rhs[i].clone()
        } else {
            small(&mut r)
        }
    });
    let a = Matrix::from_fn(n, n, |i, j| {
        if j == 0 {
            &c[(i, j)] * &p2
        } else {
            c[(i, j)].clone()
        }
    });
    let det = &c.inverse().unwrap().1 * &p2;
    assert_eq!(a.det(), det);
    assert_eq!(a.inverse().unwrap().1, det);
    let n = 28;
    let rhs = divisor_rhs(n);
    let b = Matrix::from_fn(n, n, |i, j| {
        if j == 0 {
            rhs[i].clone()
        } else {
            small(&mut r)
        }
    });
    let (b, det) = divisible(b, 2, 4);
    assert_eq!(b.det(), det);
    assert_eq!(b.inverse().unwrap().1, det);
}

#[test]
fn systems_without_a_right_hand_side() {
    let mut r = Rng(0xD1_0000);
    let n = size(40, 5);
    let a = random(&mut r, n, n, 64);
    let empty = Matrix::<Number>::zeros(n, 0);
    assert_eq!(a.solve(&empty), Some(Matrix::zeros(n, 0)));
    let mut s = a.clone();
    for j in 0..n {
        s[(1, j)] = s[(0, j)].clone();
    }
    assert_eq!(s.solve(&empty), None);
}

#[test]
fn huge_elements() {
    // Elements of 4000 bits at 24 rows, past the registers from the start.
    let mut r = Rng(0xD1_4000);
    let n = size(24, 4);
    let a = random(
        &mut r,
        n,
        n,
        if cfg!(debug_assertions) { 1500 } else { 4000 },
    );
    let b = random(&mut r, n, 1, 3000);
    let x = a.solve(&b).expect("a random system is nonsingular");
    assert_eq!(
        &a.map(|v| Rational::from(v.clone())) * &x,
        b.map(|v| Rational::from(v.clone()))
    );
    // Cramer: the common denominator divides the determinant.
    let l = x
        .as_slice()
        .iter()
        .fold(Number::from(1), |l, e| l.lcm(e.denom()));
    assert!((a.det() % &l).is_zero());
}
