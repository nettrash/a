//! Determinants of integer matrices from their residues modulo word primes: against FLINT's
//! `fmpz_mat_det` on the seeded matrices of `data/multimod_flint.txt` (written by
//! `data/gen_multimod.c`), against Bareiss's elimination (the determinant that `inverse` returns
//! beside the adjugate) and Berkowitz's characteristic polynomial, and against the residues of
//! `ModMatrix`, on the orders and lengths where `det` takes the residues and on those around them.

use super::common::{Rng, size};
use a::math::matrix::{Matrix, ModMatrix};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;

/// An element of `bits` bits as `data/gen_multimod.c` draws it.
fn entry(r: &mut Rng, bits: usize) -> Number {
    let bytes = bits.div_ceil(8);
    let mut v: Vec<u8> = (0..bytes).map(|_| r.next() as u8).collect();
    if !bits.is_multiple_of(8) {
        v[bytes - 1] &= (1u8 << (bits % 8)) - 1;
    }
    let x = Number::new(v, ByteOrder::LittleEndian);
    if r.next() & 1 == 1 { -x } else { x }
}

/// The matrix of the kind `kind` (`r` random, `s` singular, `u` of determinant 1, `l` a long
/// first row, `m` a long first row and singular) as `data/gen_multimod.c` draws it.
fn drawn(kind: &str, seed: u64, n: usize, bits: usize) -> Matrix<Number> {
    let mut r = Rng(seed);
    if kind == "u" {
        let mut l = Matrix::<Number>::identity(n);
        let mut u = Matrix::<Number>::identity(n);
        for i in 0..n {
            for j in 0..i {
                l[(i, j)] = entry(&mut r, bits);
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                u[(i, j)] = entry(&mut r, bits);
            }
        }
        return &l * &u;
    }
    let long = kind == "l" || kind == "m";
    let mut rows: Vec<Vec<Number>> = (0..n)
        .map(|i| {
            let bits = if long && i > 0 { 8 } else { bits };
            (0..n).map(|_| entry(&mut r, bits)).collect()
        })
        .collect();
    if kind == "s" || kind == "m" {
        rows[n - 1] = (0..n).map(|j| &rows[0][j] + &rows[1][j]).collect();
    }
    Matrix::from_rows(&rows)
}

/// `x mod q` in `[0, q)`.
fn residue(x: &Number, q: u64) -> u64 {
    let q = Number::from(q);
    u64::try_from(&((x % &q + &q) % &q)).unwrap()
}

#[test]
fn determinants_agree_with_flint() {
    let data = include_str!("data/multimod_flint.txt");
    for line in data.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let num = |i: usize| f[i].parse::<u64>().unwrap();
        let (n, bits) = (num(2) as usize, num(3) as usize);
        // The debug profile's budget, by the work of the residues: about `n·bits / 61` primes,
        // each `n²` residues of `bits / 64` limbs and an elimination of `n³/3`; for a long row,
        // `(bits + 8·n) / 61` primes and `n` long integers. All of them in the release profile.
        let row = f[0] == "l" || f[0] == "m";
        let (primes, long) = if row {
            ((bits + 8 * n) / 61, n)
        } else {
            (n * bits / 61, n * n)
        };
        let work = primes * (long * bits.div_ceil(64) + n * n * n / 3);
        if cfg!(miri) && n > 9 || cfg!(debug_assertions) && work > 60_000_000 {
            continue;
        }
        let d = drawn(f[0], num(1), n, bits);
        let det = d.det();
        assert_eq!(residue(&det, (1 << 61) - 1), num(4), "{line}");
        assert_eq!(residue(&det, 18_446_744_073_709_551_557), num(5), "{line}");
        let sign: i64 = f[6].parse().unwrap();
        assert_eq!(
            if det.is_zero() {
                0
            } else if det.is_negative() {
                -1
            } else {
                1
            },
            sign,
            "{line}"
        );
        assert_eq!(det.bits(), num(7), "{line}");
    }
}

/// A random matrix of numbers of `bits` bits.
fn random(r: &mut Rng, n: usize, bits: usize) -> Matrix<Number> {
    Matrix::from_fn(n, n, |_, _| entry(r, bits))
}

#[test]
fn determinants_agree_with_bareiss_and_berkowitz() {
    // Inside the band of orders where `det` takes the residues for each length, at its edges,
    // and outside it.
    let mut r = Rng(0x3D37);
    let cases: &[(usize, usize)] = if cfg!(miri) {
        &[(9, 64)]
    } else {
        &[
            (19, 64),
            (20, 64),
            (31, 100),
            (16, 300),
            (20, 512),
            (12, 1600),
            (11, 1600),
            (15, 1000),
        ]
    };
    for &(n, bits) in cases {
        let a = random(&mut r, n, bits);
        let det = a.det();
        let (_, bareiss) = a.inverse().expect("a random matrix is nonsingular");
        assert_eq!(det, bareiss, "{n} × {n} of {bits} bits");
        if n <= 16 {
            let c = a.characteristic_polynomial();
            let berkowitz = if n % 2 == 0 { c[n].clone() } else { -&c[n] };
            assert_eq!(det, berkowitz, "{n} × {n} of {bits} bits");
        }
        // The residues modulo primes, by `ModMatrix`.
        for p in [2, 3, 1_000_003, 18_446_744_073_709_551_557] {
            assert_eq!(
                ModMatrix::from_matrix(&a, p).det(),
                residue(&det, p),
                "{n} × {n} of {bits} bits modulo {p}"
            );
        }
        // det(-A) = (-1)^n det(A), and a swap of two rows changes the sign.
        let neg = a.map(|x| -x);
        assert_eq!(neg.det(), if n % 2 == 0 { det.clone() } else { -&det });
        let swapped = Matrix::from_fn(n, n, |i, j| {
            let i = match i {
                0 => 1,
                1 => 0,
                _ => i,
            };
            a[(i, j)].clone()
        });
        assert_eq!(swapped.det(), -&det);
    }
}

#[test]
fn singular_matrices_and_matrices_with_a_zero_row_or_column() {
    let mut r = Rng(0x5137);
    let n = size(24, 9);
    for bits in [64, 300] {
        // A row that is a combination of two others.
        let mut a = random(&mut r, n, bits);
        for j in 0..n {
            a[(n - 1, j)] = &(&a[(0, j)] * &Number::from(3)) - &a[(5, j)];
        }
        assert!(a.det().is_zero(), "{bits}");
        assert_eq!(a.inverse(), None);
        // A zero row: Hadamard's bound is 0.
        let mut z = random(&mut r, n, bits);
        for j in 0..n {
            z[(3, j)] = Number::default();
        }
        assert!(z.det().is_zero(), "{bits}");
        // A zero column.
        let mut z = random(&mut r, n, bits);
        for i in 0..n {
            z[(i, 7)] = Number::default();
        }
        assert!(z.det().is_zero(), "{bits}");
        // Rank n - 2: the product of n × (n - 2) and (n - 2) × n matrices.
        let b = Matrix::from_fn(n, n - 2, |_, _| entry(&mut r, bits / 2));
        let c = Matrix::from_fn(n - 2, n, |_, _| entry(&mut r, bits / 2));
        assert!((&b * &c).det().is_zero(), "{bits}");
    }
}

#[test]
fn determinants_of_integers_at_the_edges_of_the_registers() {
    // Elements of magnitude 2^63 - 1, 2^63 and 2^64 - 1, which are in a register, one past it,
    // and the largest word, of both signs, in the band of the residues.
    let mut r = Rng(0xED6E);
    let n = size(20, 9);
    let edges = [
        Number::from(i64::MAX),
        Number::from(1u64 << 63),
        Number::from(u64::MAX),
        Number::from(i64::MIN),
        -Number::from(u64::MAX),
        Number::from(1),
        Number::default(),
    ];
    let a = Matrix::from_fn(n, n, |_, _| {
        let k = (r.next() % edges.len() as u64) as usize;
        &edges[k] - &Number::from(r.next() % 3)
    });
    let (_, bareiss) = a
        .inverse()
        .unwrap_or((Matrix::zeros(0, 0), Number::default()));
    assert_eq!(a.det(), bareiss);
}

#[test]
fn determinants_of_rationals_whose_integers_take_the_residues() {
    // `Matrix<Rational>::det` clears each row by its denominators: det(A / d) = det(A) / d^n for
    // the integers A and a common denominator d.
    let mut r = Rng(0xFAC7);
    let n = size(20, 9);
    let a = random(&mut r, n, 120);
    let d = Number::from(1_000_003u64 * 999_983);
    let q = a.map(|x| Rational::new(x.clone(), d.clone()));
    let want = Rational::new(a.det(), d.pow(n as u32));
    assert_eq!(q.det(), want);
}
