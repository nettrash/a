//! Tests of `a::math::matrix::BitMatrix`, the matrices over GF(2): against the outputs of M4RI
//! (`data/gf2_reference.txt`, from `data/gen_gf2_m4ri.c`, and recomputed by NTL's `mat_GF2` with
//! `data/gen_gf2_ntl.cpp`) on seeded matrices of 0 to 2048 rows and columns, and against naive
//! products and eliminations written here, bit by bit.

use super::common::size;
use a::math::matrix::{BitMatrix, Matrix};
use a::math::numbers::{ByteOrder, Number};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// The reference outputs of M4RI; see `data/gen_gf2_m4ri.c` for the format.
const REFERENCE: &str = include_str!("data/gf2_reference.txt");

/// SplitMix64's finaliser.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64's next output.
fn next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mix(*s)
}

/// The bits of a row's last word that hold columns.
fn last_mask(cols: usize) -> u64 {
    if cols.is_multiple_of(64) {
        u64::MAX
    } else {
        (1 << (cols % 64)) - 1
    }
}

/// The hash of `data/gen_gf2.h`: the shape, then every word, each through the finaliser.
fn hash(a: &BitMatrix) -> u64 {
    let mut h = mix(mix(a.rows() as u64) ^ a.cols() as u64);
    for &w in a.as_words() {
        h = mix(h ^ w);
    }
    h
}

/// The seeded matrices of `data/gen_gf2.h`, built word by word.
fn generate(kind: &str, m: usize, n: usize, seed: u64) -> BitMatrix {
    if kind == "low" {
        let r = m.min(n) / 3;
        let x = generate("dense", m, r, seed);
        let y = generate("dense", r, n, seed ^ 0xA5A5_A5A5_A5A5_A5A5);
        return naive_product(&x, &y);
    }
    let stride = n.div_ceil(64);
    let mut s = seed;
    let sparse = kind == "sparse" || kind == "permuted";
    let mut w = Vec::with_capacity(m * stride);
    for _ in 0..m {
        for k in 0..stride {
            let mut v = next(&mut s);
            if sparse {
                v &= next(&mut s);
                v &= next(&mut s);
            }
            if k == stride - 1 {
                v &= last_mask(n);
            }
            w.push(v);
        }
    }
    let flip = |w: &mut Vec<u64>, i: usize, j: usize| w[i * stride + j / 64] ^= 1 << (j % 64);
    let get = |w: &Vec<u64>, i: usize, j: usize| w[i * stride + j / 64] >> (j % 64) & 1 == 1;
    if kind == "permuted" {
        for i in 0..m {
            for j in 0..n.min(i + 1) {
                if get(&w, i, j) != (j == i) {
                    flip(&mut w, i, j);
                }
            }
        }
        for i in (1..m).rev() {
            let j = (next(&mut s) % (i as u64 + 1)) as usize;
            for k in 0..stride {
                w.swap(i * stride + k, j * stride + k);
            }
        }
    }
    if kind == "deficient" {
        for i in (3..m).filter(|i| i % 4 == 3) {
            for k in 0..stride {
                w[i * stride + k] = w[(i - 1) * stride + k] ^ w[(i - 3) * stride + k];
            }
        }
        for j in (2..n).filter(|j| j % 3 == 2) {
            for i in 0..m {
                if get(&w, i, j) != (get(&w, i, j - 1) ^ get(&w, i, j - 2)) {
                    flip(&mut w, i, j);
                }
            }
        }
    }
    BitMatrix::from_words(m, n, w)
}

/// `a · b`, a row of `b` for each 1 of `a`, read element by element.
fn naive_product(a: &BitMatrix, b: &BitMatrix) -> BitMatrix {
    assert_eq!(a.cols(), b.rows());
    let stride = b.cols().div_ceil(64);
    let mut w = vec![0u64; a.rows() * stride];
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            if a.get(i, j) {
                for (x, y) in w[i * stride..(i + 1) * stride]
                    .iter_mut()
                    .zip(b.row_words(j))
                {
                    *x ^= y;
                }
            }
        }
    }
    BitMatrix::from_words(a.rows(), b.cols(), w)
}

/// The rows as vectors of `bool`s.
fn to_bools(a: &BitMatrix) -> Vec<Vec<bool>> {
    (0..a.rows())
        .map(|i| (0..a.cols()).map(|j| a.get(i, j)).collect())
        .collect()
}

/// The reduced row echelon form and its pivots by Gauss and Jordan's elimination on `bool`s.
fn naive_rref(a: &BitMatrix) -> (BitMatrix, Vec<usize>) {
    let (m, n) = a.shape();
    let mut x = to_bools(a);
    let mut pivots = Vec::new();
    for c in 0..n {
        let r = pivots.len();
        let Some(p) = (r..m).find(|&i| x[i][c]) else {
            continue;
        };
        x.swap(p, r);
        let pivot = x[r].clone();
        for (i, row) in x.iter_mut().enumerate() {
            if i != r && row[c] {
                for (y, &v) in row.iter_mut().zip(&pivot) {
                    *y ^= v;
                }
            }
        }
        pivots.push(c);
    }
    (BitMatrix::from_fn(m, n, |i, j| x[i][j]), pivots)
}

/// Whether every bit beyond the last column is zero.
fn padding_is_zero(a: &BitMatrix) -> bool {
    let s = a.words_per_row();
    assert_eq!(a.as_words().len(), a.rows() * s);
    s == 0 || (0..a.rows()).all(|i| a.row_words(i)[s - 1] & !last_mask(a.cols()) == 0)
}

/// The hash of `std`, to see that equal matrices hash alike.
fn std_hash(a: &BitMatrix) -> u64 {
    let mut h = DefaultHasher::new();
    a.hash(&mut h);
    h.finish()
}

/// The lines of the reference with operation `op`, as fields, whose dimensions are at most
/// `limit` (scaled down under Miri).
fn cases(op: &str, limit: usize) -> Vec<Vec<&'static str>> {
    REFERENCE
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .filter(|f| f[0] == op)
        .filter(|f| {
            let dims: &[usize] = match op {
                "mul" => &[2, 3, 6],
                "ech" => &[2, 3],
                "inv" => &[2],
                _ => &[2, 3, 6],
            };
            dims.iter()
                .all(|&k| f[k].parse::<usize>().unwrap() <= limit)
        })
        .collect()
}

fn num(s: &str) -> usize {
    s.parse().unwrap()
}

fn seed(s: &str) -> u64 {
    s.parse().unwrap()
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap()
}

/// Checks a reduced row echelon form and its pivots against the definition.
fn assert_rref(r: &BitMatrix, pivots: &[usize]) {
    for (i, &p) in pivots.iter().enumerate() {
        assert!(r.get(i, p), "the pivot of row {i}");
        assert!(
            (0..p).all(|j| !r.get(i, j)),
            "row {i} is not zero before its pivot"
        );
        assert!(
            (0..r.rows()).all(|k| k == i || !r.get(k, p)),
            "column {p} has another 1"
        );
        if i > 0 {
            assert!(pivots[i - 1] < p, "the pivots are not ascending");
        }
    }
    assert!(
        (pivots.len()..r.rows()).all(|i| r.row_words(i).iter().all(|&w| w == 0)),
        "a row past the rank is not zero"
    );
}

#[test]
fn building_reading_and_writing_agree_element_by_element() {
    for &(m, n) in &[
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (3, 63),
        (2, 64),
        (4, 65),
        (65, 129),
    ] {
        let f = |i: usize, j: usize| (i * 7 + j * 3) % 5 < 2;
        let a = BitMatrix::from_fn(m, n, f);
        assert_eq!(a.shape(), (m, n));
        assert_eq!(a.words_per_row(), n.div_ceil(64));
        let rows: Vec<Vec<bool>> = (0..m).map(|i| (0..n).map(|j| f(i, j)).collect()).collect();
        if m > 0 {
            assert_eq!(BitMatrix::from_rows(&rows), a);
        }
        let mut b = BitMatrix::zeros(m, n);
        let mut c = BitMatrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(a.get(i, j), f(i, j));
                b.set(i, j, f(i, j));
                if f(i, j) {
                    c.flip(i, j);
                }
            }
        }
        assert_eq!(b, a);
        assert_eq!(c, a);
        let words: Vec<u64> = (0..m).flat_map(|i| a.row_words(i).to_vec()).collect();
        assert_eq!(words, a.as_words());
        assert_eq!(BitMatrix::from_words(m, n, words), a);
        let mut d = BitMatrix::zeros(m, n);
        for i in 0..m {
            d.set_row_words(i, a.row_words(i));
        }
        assert_eq!(d, a);
        // Setting to false and flipping back.
        if m > 0 && n > 0 {
            let mut e = a.clone();
            e.set(m - 1, n - 1, !f(m - 1, n - 1));
            assert_ne!(e, a);
            e.flip(m - 1, n - 1);
            assert_eq!(e, a);
        }
    }
    let i = BitMatrix::identity(130);
    for r in 0..130 {
        for c in 0..130 {
            assert_eq!(i.get(r, c), r == c);
        }
    }
    assert!(BitMatrix::identity(0).as_words().is_empty());
    assert_eq!(BitMatrix::default(), BitMatrix::zeros(0, 0));
    assert!(BitMatrix::zeros(3, 3).is_square() && !BitMatrix::zeros(3, 4).is_square());
}

#[test]
fn row_operations_swap_and_add_whole_rows() {
    let a = generate("dense", 5, 130, 1);
    let mut b = a.clone();
    b.swap_rows(1, 4);
    b.swap_rows(2, 2);
    for j in 0..130 {
        assert_eq!(b.get(1, j), a.get(4, j));
        assert_eq!(b.get(4, j), a.get(1, j));
        assert_eq!(b.get(2, j), a.get(2, j));
    }
    let mut c = a.clone();
    c.add_row(3, 0);
    c.add_row(0, 3);
    for j in 0..130 {
        assert_eq!(c.get(3, j), a.get(3, j) ^ a.get(0, j));
        assert_eq!(c.get(0, j), a.get(3, j));
    }
    c.add_row(2, 2);
    assert!(c.row_words(2).iter().all(|&w| w == 0));
}

#[test]
fn equal_matrices_hash_alike_and_shapes_tell_apart() {
    let a = generate("dense", 70, 130, 2);
    let mut b = BitMatrix::zeros(70, 130);
    for i in 0..70 {
        for j in (0..130).rev() {
            if a.get(i, j) {
                b.flip(i, j);
            }
        }
    }
    assert_eq!(a, b);
    assert_eq!(std_hash(&a), std_hash(&b));
    // Ones set and then cleared beyond a word's columns leave no trace.
    let mut c = BitMatrix::zeros(70, 130);
    for i in 0..70 {
        c.set(i, 129, true);
        c.set(i, 129, false);
    }
    assert_eq!(c, BitMatrix::zeros(70, 130));
    assert_eq!(std_hash(&c), std_hash(&BitMatrix::zeros(70, 130)));
    let shapes = [(0, 0), (0, 5), (5, 0), (0, 6), (6, 0)];
    for (x, &p) in shapes.iter().enumerate() {
        for (y, &q) in shapes.iter().enumerate() {
            assert_eq!(
                BitMatrix::zeros(p.0, p.1) == BitMatrix::zeros(q.0, q.1),
                x == y
            );
        }
    }
}

#[test]
fn the_padding_bits_stay_zero_after_every_operation() {
    for &(m, n) in &[(1, 1), (5, 63), (40, 65), (65, 127), (129, 130), (33, 200)] {
        if m * n > size(1 << 20, 3000) {
            continue;
        }
        let a = BitMatrix::from_fn(m, n, |_, _| true);
        let b = generate("dense", m, n, 3);
        let sq = generate("dense", n, n, 4);
        let r = generate("dense", m, 7, 5);
        let mut results = vec![
            a.clone(),
            &a + &b,
            a.transpose(),
            b.transpose(),
            &b * &sq,
            &a * &a.transpose(),
            a.reduced_row_echelon_form().0,
            b.reduced_row_echelon_form().0,
            b.null_space(),
            a.null_space(),
            (&b.transpose() * &a).null_space(),
        ];
        if let Some(x) = sq.inverse() {
            results.push(x);
        }
        if let Some(x) = b.solve(&r) {
            results.push(x);
        }
        if let Some(x) = b.solve(&(&b * &generate("dense", n, 3, 6))) {
            results.push(x);
        }
        let mut c = b.clone();
        c.echelonize();
        results.push(c);
        let mut d = b.clone();
        d.add_row(0, m - 1);
        d.swap_rows(0, m - 1);
        d += &a;
        d *= &sq;
        results.push(d);
        for (k, x) in results.iter().enumerate() {
            assert!(padding_is_zero(x), "result {k} for {m} × {n}");
        }
    }
}

#[test]
fn conversion_from_numbers_takes_parities_of_negative_and_huge_entries() {
    let big = Number::new(
        (0..1000).map(|k| (k * 37 + 11) as u8).collect(),
        ByteOrder::LittleEndian,
    );
    let odd = &big + &Number::from(if big.bit(0) { 0 } else { 1 });
    let even = &odd + &Number::from(1);
    let m = Matrix::from_fn(3, 70, |i, j| match (i + j) % 6 {
        0 => odd.clone(),
        1 => -&odd,
        2 => even.clone(),
        3 => -&even,
        4 => Number::from(-7),
        _ => Number::from(0),
    });
    let b = BitMatrix::from(&m);
    for i in 0..3 {
        for j in 0..70 {
            assert_eq!(b.get(i, j), matches!((i + j) % 6, 0 | 1 | 4), "({i}, {j})");
        }
    }
    let back = b.to_matrix();
    assert_eq!(back.shape(), (3, 70));
    assert_eq!(BitMatrix::from(&back), b);
    assert!(back.as_slice().iter().all(|x| *x == 0 || *x == 1));
    assert_eq!(
        BitMatrix::from(&Matrix::<Number>::zeros(0, 4)),
        BitMatrix::zeros(0, 4)
    );
}

#[test]
fn sums_and_products_take_owned_and_borrowed_operands() {
    let a = generate("dense", 70, 90, 7);
    let b = generate("sparse", 70, 90, 8);
    let c = generate("dense", 90, 65, 9);
    let s = BitMatrix::from_fn(70, 90, |i, j| a.get(i, j) ^ b.get(i, j));
    assert_eq!(&a + &b, s);
    assert_eq!(a.clone() + &b, s);
    assert_eq!(&a + b.clone(), s);
    assert_eq!(a.clone() + b.clone(), s);
    let mut t = a.clone();
    t += &b;
    assert_eq!(t, s);
    t += b.clone();
    assert_eq!(t, a);
    assert_eq!(&a + &a, BitMatrix::zeros(70, 90));
    let p = naive_product(&a, &c);
    assert_eq!(&a * &c, p);
    assert_eq!(a.clone() * &c, p);
    assert_eq!(&a * c.clone(), p);
    assert_eq!(a.clone() * c.clone(), p);
    let mut q = a.clone();
    q *= &c;
    assert_eq!(q, p);
    let mut q = a.clone();
    q *= c.clone();
    assert_eq!(q, p);
}

#[test]
fn products_match_the_schoolbook_product_bit_by_bit() {
    let mut s = 11u64;
    let dims = [0, 1, 2, 3, 7, 8, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129];
    for _ in 0..size(60, 6) {
        let pick = |s: &mut u64| dims[(next(s) % dims.len() as u64) as usize];
        let (m, l, n) = (pick(&mut s), pick(&mut s), pick(&mut s));
        let kind = ["dense", "sparse", "deficient"][(next(&mut s) % 3) as usize];
        let a = generate(kind, m, l, next(&mut s));
        let b = generate("dense", l, n, next(&mut s));
        let c = &a * &b;
        assert_eq!(c.shape(), (m, n));
        for i in 0..m {
            for j in 0..n {
                let x = (0..l).fold(false, |x, k| x ^ (a.get(i, k) & b.get(k, j)));
                assert_eq!(c.get(i, j), x, "{m} × {l} × {n} at ({i}, {j})");
            }
        }
    }
}

#[test]
fn products_match_m4ri_on_the_reference_cases() {
    let cases = cases("mul", size(2200, 65));
    assert_eq!(cases.len(), if cfg!(miri) { 14 } else { 34 });
    for f in cases {
        let (m, l, n) = (num(f[2]), num(f[3]), num(f[6]));
        let a = generate(f[1], m, l, seed(f[4]));
        let b = generate(f[5], l, n, seed(f[7]));
        let c = &a * &b;
        assert_eq!(hash(&c), hex(f[8]), "{f:?}");
        assert_eq!(c, naive_product(&a, &b), "{f:?}");
    }
}

/// The product of two 4096 × 4096 matrices, one level of Strassen–Winograd: about two seconds in
/// the debug profile here, more on CI, so it runs on request.
#[test]
#[ignore = "a 4096 × 4096 product; run with --ignored"]
fn the_product_of_4096_square_matrices_matches_m4ri() {
    let f = cases("mul", 4096)
        .into_iter()
        .find(|f| f[2] == "4096")
        .unwrap();
    let a = generate(f[1], 4096, 4096, seed(f[4]));
    let b = generate(f[5], 4096, 4096, seed(f[7]));
    let c = &a * &b;
    assert_eq!(hash(&c), hex(f[8]));
    // A product of the transposes, which takes the same path, the other way round.
    assert_eq!((&b.transpose() * &a.transpose()).transpose(), c);
}

#[test]
fn strassen_winograd_products_match_m4ri_down_to_halves_of_64() {
    // The reference products whose three dimensions are all at least 128, the others having one
    // too small to be halved: 128³, 129³ and 1000³ twice each, 2048³ and 2113 × 2177 × 2100, by
    // the recursion down to halves of 64 (from 128) and by one that stops at halves of 128 or
    // more (from 300), peeled or not.
    let mut recursed = 0;
    for f in cases("mul", size(2200, 129)) {
        let (m, l, n) = (num(f[2]), num(f[3]), num(f[6]));
        if m.min(l).min(n) < 128 {
            continue;
        }
        recursed += 1;
        let a = generate(f[1], m, l, seed(f[4]));
        let b = generate(f[5], l, n, seed(f[7]));
        for cutoff in [0, 300] {
            assert_eq!(
                hash(&a.mul_strassen(&b, cutoff)),
                hex(f[8]),
                "{f:?} from {cutoff}"
            );
        }
    }
    assert_eq!(recursed, if cfg!(miri) { 4 } else { 8 });
}

#[test]
fn strassen_winograd_products_of_peeled_shapes_match_the_schoolbook_product() {
    // Random shapes of 128 to 600 rows and columns, so that the recursion runs down to halves of
    // 64 and peels rows, inner columns and columns, each with and without a remainder, at its
    // levels.
    let dims = [
        128, 129, 191, 192, 193, 255, 256, 257, 320, 383, 384, 385, 511, 512, 513, 600,
    ];
    let mut s = 23u64;
    for _ in 0..size(40, 2) {
        let pick = |s: &mut u64| dims[(next(s) % size(dims.len(), 5) as u64) as usize];
        let (m, l, n) = (pick(&mut s), pick(&mut s), pick(&mut s));
        let kind = ["dense", "sparse", "deficient"][(next(&mut s) % 3) as usize];
        let a = generate(kind, m, l, next(&mut s));
        let b = generate("dense", l, n, next(&mut s));
        let p = naive_product(&a, &b);
        for cutoff in [128, 129, 200] {
            assert_eq!(
                a.mul_strassen(&b, cutoff),
                p,
                "{m} × {l} × {n} from {cutoff}"
            );
        }
    }
}

#[test]
fn strassen_winograd_products_are_associative_with_thin_products() {
    let n = size(2112, 200);
    let a = generate("dense", n, n, 12);
    let b = generate("deficient", n, n + 65, 13);
    let x = generate("dense", n + 65, 3, 14);
    // (A·B)·X, A·B by two levels of halves with every dimension peeled, against A·(B·X), with
    // thin products only.
    assert_eq!(&a.mul_strassen(&b, n / 4) * &x, &a * &(&b * &x));
}

#[test]
fn transposes_match_the_definition() {
    for &(m, n) in &[
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (63, 65),
        (64, 64),
        (65, 63),
        (129, 200),
        (1000, 7),
    ] {
        if m * n > size(1 << 20, 5000) {
            continue;
        }
        let a = generate("dense", m, n, (m * 1000 + n) as u64);
        let t = a.transpose();
        assert_eq!(t.shape(), (n, m));
        for i in 0..m {
            for j in 0..n {
                assert_eq!(t.get(j, i), a.get(i, j));
            }
        }
        assert_eq!(t.transpose(), a);
    }
    let (m, l, n) = (size(300, 70), size(200, 65), size(130, 66));
    let a = generate("dense", m, l, 15);
    let b = generate("dense", l, n, 16);
    assert_eq!((&a * &b).transpose(), &b.transpose() * &a.transpose());
}

#[test]
fn echelon_forms_ranks_and_null_spaces_match_m4ri() {
    let cases = cases("ech", size(2200, 65));
    assert_eq!(cases.len(), if cfg!(miri) { 30 } else { 100 });
    for f in cases {
        let (m, n) = (num(f[2]), num(f[3]));
        let a = generate(f[1], m, n, seed(f[4]));
        let rank = num(f[5]);
        assert_eq!(a.rank(), rank, "{f:?}");
        let (r, pivots) = a.reduced_row_echelon_form();
        assert_eq!(pivots.len(), rank, "{f:?}");
        assert_eq!(hash(&r), hex(f[6]), "{f:?}");
        assert_rref(&r, &pivots);
        let mut e = a.clone();
        assert_eq!(e.echelonize(), pivots);
        assert_eq!(e, r);
        let x = a.null_space();
        assert_eq!(x.shape(), (n, n - rank), "{f:?}");
        assert_eq!(hash(&x), hex(f[7]), "{f:?}");
        assert_eq!(&a * &x, BitMatrix::zeros(m, n - rank), "{f:?}");
        assert_eq!(x.rank(), n - rank, "{f:?}");
        if m == n {
            assert_eq!(a.det(), rank == n, "{f:?}");
        }
        if m <= 200 && n <= 200 {
            assert_eq!((r, pivots), naive_rref(&a), "{f:?}");
        }
        assert_eq!(a.transpose().rank(), rank, "{f:?}");
    }
}

#[test]
fn inverses_match_m4ri_and_multiply_to_the_identity() {
    let cases = cases("inv", size(2200, 65));
    assert_eq!(cases.len(), if cfg!(miri) { 20 } else { 40 });
    let mut inverted = 0;
    for f in cases {
        let n = num(f[2]);
        let a = generate(f[1], n, n, seed(f[3]));
        let x = a.inverse();
        assert_eq!(x.is_some(), f[4] != "singular", "{f:?}");
        assert_eq!(a.det(), x.is_some(), "{f:?}");
        if let Some(x) = x {
            assert_eq!(hash(&x), hex(f[4]), "{f:?}");
            assert_eq!(&a * &x, BitMatrix::identity(n), "{f:?}");
            assert_eq!(&x * &a, BitMatrix::identity(n), "{f:?}");
            inverted += 1;
        }
    }
    assert_eq!(inverted, if cfg!(miri) { 12 } else { 25 });
}

#[test]
fn solutions_match_m4ri_and_satisfy_the_system() {
    let cases = cases("solve", size(2200, 65));
    assert_eq!(cases.len(), if cfg!(miri) { 18 } else { 60 });
    for f in cases {
        let (m, n, k) = (num(f[2]), num(f[3]), num(f[6]));
        let a = generate(f[1], m, n, seed(f[4]));
        let b = if f[5] == "image" {
            naive_product(&a, &generate("dense", n, k, seed(f[7])))
        } else {
            generate(f[5], m, k, seed(f[7]))
        };
        let (ra, rab) = (num(f[8]), num(f[9]));
        assert_eq!(a.rank(), ra, "{f:?}");
        let x = a.solve(&b);
        assert_eq!(x.is_some(), rab == ra, "{f:?}");
        assert_eq!(x.is_none(), f[10] == "none", "{f:?}");
        if let Some(x) = x {
            assert_eq!(x.shape(), (n, k));
            assert_eq!(hash(&x), hex(f[10]), "{f:?}");
            assert_eq!(&a * &x, b, "{f:?}");
        }
        if f[5] == "image" {
            assert!(f[10] != "none", "{f:?}");
        }
    }
}

/// The fastest of three runs of `f`, in seconds, and its result.
fn fastest<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let x = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(x);
    }
    (best, out.unwrap())
}

#[test]
fn rows_whose_first_ones_are_far_right_cost_no_more_than_rows_that_start_dense() {
    // 32 rows whose bits are random in their last 64 columns and zero before, against the same
    // rows with the random block first: both have rank 32, and each needs one pass over the
    // words. An elimination that searched each strip's zero rows for their first 1 again would
    // be quadratic in the columns: measured in the debug profile, 0.39 s against 0.0019 s (a
    // ratio of 206) before the fix, and 0.00016 s against 0.0019 s (0.08) after it. The bound
    // of 20 and the 0.05 s are margins for a loaded runner.
    let (m, n) = (32, size(1 << 16, 256));
    let mut s = 29u64;
    let block = generate("dense", m, 64, next(&mut s));
    let late = BitMatrix::from_fn(m, n, |i, j| j >= n - 64 && block.get(i, j + 64 - n));
    let early = BitMatrix::from_fn(m, n, |i, j| j < 64 && block.get(i, j));
    let (t_late, r_late) = fastest(|| late.rank());
    let (t_early, r_early) = fastest(|| early.rank());
    assert_eq!((r_late, r_early), (block.rank(), block.rank()));
    let (r, pivots) = late.reduced_row_echelon_form();
    let (q, p) = block.reduced_row_echelon_form();
    assert_eq!(pivots, p.iter().map(|&j| j + n - 64).collect::<Vec<_>>());
    assert!((0..m).all(|i| (0..64).all(|j| r.get(i, n - 64 + j) == q.get(i, j))));
    if !cfg!(miri) {
        assert!(
            t_late < 20.0 * t_early + 0.05,
            "rank of the late block {t_late:.4} s, of the early one {t_early:.4} s"
        );
    }
}

#[test]
fn rows_zero_in_the_matrix_but_not_in_the_right_hand_side_end_the_search_at_once() {
    // A wide system whose even rows are zero, the odd ones a 1 in their own column: with a
    // right-hand side of ones it has no solution, with one from the image it has. The rows of
    // [A | b] zero in A can give no pivot; a search that kept reading them would cost every
    // strip a pass over A's words: measured in the debug profile, 0.91 s against 0.0057 s for
    // the solvable system (a ratio of 161) before the fix, and 0.0057 s for both (1.0) after it.
    let (m, w) = (256, size(1 << 16, 256));
    let a = BitMatrix::from_fn(m, w, |i, j| i % 2 == 1 && j == i);
    let ones = BitMatrix::from_fn(m, 1, |_, _| true);
    let image = BitMatrix::from_fn(m, 1, |i, _| i % 2 == 1);
    let (t_none, x) = fastest(|| a.solve(&ones));
    let (t_some, y) = fastest(|| a.solve(&image));
    assert_eq!(x, None);
    assert_eq!(
        y,
        Some(BitMatrix::from_fn(w, 1, |i, _| i % 2 == 1 && i < m))
    );
    assert_eq!(a.rank(), m / 2);
    if !cfg!(miri) {
        assert!(
            t_none < 20.0 * t_some + 0.05,
            "the unsolvable system {t_none:.4} s, the solvable one {t_some:.4} s"
        );
    }
}

#[test]
fn rows_without_a_one_in_the_matrix_are_dropped_while_pivots_keep_coming() {
    // A wide system whose odd rows have one 1 each, spread evenly over all the columns, and
    // whose even rows are zero, with a right-hand side of ones: no solution, and every strip
    // with a pivot comes after the even rows are known to have no 1 in A. Kept in play, they
    // would be searched again in each of those strips; dropped, the system costs about as much
    // as the solvable one of its odd rows alone, the reference. Measured in the debug profile:
    // 0.024 s against 0.014 s (a ratio of 1.8), and with those rows kept in play 1.35 s (89).
    let (m, w) = (size(1024, 64), size(1 << 17, 256));
    let gap = w / (m / 2);
    let mut a = BitMatrix::zeros(m, w);
    let mut odd = BitMatrix::zeros(m / 2, w);
    for i in 0..m / 2 {
        a.set(2 * i + 1, i * gap, true);
        odd.set(i, i * gap, true);
    }
    let (t_none, x) = fastest(|| a.solve(&BitMatrix::from_fn(m, 1, |_, _| true)));
    let (t_odd, y) = fastest(|| odd.solve(&BitMatrix::from_fn(m / 2, 1, |_, _| true)));
    assert_eq!(x, None);
    assert_eq!(
        y,
        Some(BitMatrix::from_fn(w, 1, |i, _| i % gap == 0 && i / gap < m / 2))
    );
    assert_eq!(a.rank(), m / 2);
    if !cfg!(miri) {
        assert!(
            t_none < 20.0 * t_odd + 0.05,
            "the unsolvable system {t_none:.4} s, its odd rows alone {t_odd:.4} s"
        );
    }
}

#[test]
fn small_random_matrices_agree_with_a_naive_elimination() {
    // Shapes on both sides of the switch from classical elimination to M4RI at 32 rows, and of
    // the words' boundaries, so that strips straddle two words.
    let mut s = 17u64;
    for t in 0..size(400, 20) {
        let m = (next(&mut s) % 70) as usize;
        let n = (next(&mut s) % 140) as usize;
        let kind = ["dense", "sparse", "deficient", "low", "permuted"][t % 5];
        let a = generate(kind, m, n, next(&mut s));
        let (r, pivots) = naive_rref(&a);
        assert_eq!(
            a.reduced_row_echelon_form(),
            (r.clone(), pivots.clone()),
            "{kind} {m} × {n}"
        );
        assert_eq!(a.rank(), pivots.len());
        let x = a.null_space();
        assert_eq!(x.shape(), (n, n - pivots.len()));
        assert_eq!(&a * &x, BitMatrix::zeros(m, n - pivots.len()));
        assert_eq!(x.rank(), n - pivots.len());
        // The canonical basis: a 1 in each free row of its own column, zeros in the others.
        let free: Vec<usize> = (0..n).filter(|j| !pivots.contains(j)).collect();
        for (c, &f) in free.iter().enumerate() {
            for (d, &g) in free.iter().enumerate() {
                assert_eq!(x.get(g, c), c == d);
            }
            for (i, &p) in pivots.iter().enumerate() {
                assert_eq!(x.get(p, c), r.get(i, f));
            }
        }
        // A right-hand side from the image is solved, by the solution with zeros in the free rows.
        let b = naive_product(&a, &generate("dense", n, 2, next(&mut s)));
        let y = a.solve(&b).unwrap();
        assert_eq!(&a * &y, b);
        assert!(free.iter().all(|&f| !y.get(f, 0) && !y.get(f, 1)));
        // A random one is solved exactly when its rank does not raise that of `[A | B]`.
        let c = generate("dense", m, 1, next(&mut s));
        let ac = BitMatrix::from_fn(
            m,
            n + 1,
            |i, j| if j < n { a.get(i, j) } else { c.get(i, 0) },
        );
        assert_eq!(a.solve(&c).is_some(), ac.rank() == pivots.len());
        if m == n {
            let ai = BitMatrix::from_fn(
                n,
                2 * n,
                |i, j| if j < n { a.get(i, j) } else { j - n == i },
            );
            let (q, p) = naive_rref(&ai);
            let inverse = (p.len() >= n && (n == 0 || p[n - 1] == n - 1))
                .then(|| BitMatrix::from_fn(n, n, |i, j| q.get(i, n + j)));
            assert_eq!(a.inverse(), inverse, "{kind} {n} × {n}");
            assert_eq!(a.det(), inverse.is_some());
        }
    }
}

#[test]
fn empty_zero_and_identity_matrices_have_their_known_forms() {
    let e = BitMatrix::zeros(0, 0);
    assert_eq!(e.rank(), 0);
    assert!(e.det());
    assert_eq!(e.inverse(), Some(e.clone()));
    assert_eq!(e.null_space(), BitMatrix::zeros(0, 0));
    assert_eq!(e.reduced_row_echelon_form(), (e.clone(), vec![]));
    assert_eq!(
        e.solve(&BitMatrix::zeros(0, 3)),
        Some(BitMatrix::zeros(0, 3))
    );
    assert_eq!(&e * &e, e);
    // No equations: everything solves them.
    let w = BitMatrix::zeros(0, 70);
    assert_eq!(w.null_space(), BitMatrix::identity(70));
    assert_eq!(
        w.solve(&BitMatrix::zeros(0, 2)),
        Some(BitMatrix::zeros(70, 2))
    );
    assert_eq!(&BitMatrix::zeros(70, 0) * &w, BitMatrix::zeros(70, 70));
    // No unknowns: only a zero right-hand side is solved.
    let t = BitMatrix::zeros(70, 0);
    assert_eq!(t.null_space(), BitMatrix::zeros(0, 0));
    assert_eq!(
        t.solve(&BitMatrix::zeros(70, 1)),
        Some(BitMatrix::zeros(0, 1))
    );
    let mut one = BitMatrix::zeros(70, 1);
    one.set(69, 0, true);
    assert_eq!(t.solve(&one), None);
    for n in [1, 31, 32, 33, 64, 65, 100] {
        let z = BitMatrix::zeros(n, n);
        assert_eq!(z.rank(), 0);
        assert!(!z.det());
        assert_eq!(z.inverse(), None);
        assert_eq!(z.null_space(), BitMatrix::identity(n));
        let i = BitMatrix::identity(n);
        assert_eq!(i.rank(), n);
        assert_eq!(i.inverse(), Some(i.clone()));
        assert_eq!(i.null_space(), BitMatrix::zeros(n, 0));
        assert_eq!(i.reduced_row_echelon_form(), (i.clone(), (0..n).collect()));
        let ones = BitMatrix::from_fn(n, n, |_, _| true);
        assert_eq!(ones.rank(), 1);
        assert_eq!(ones.det(), n == 1);
        assert_eq!(
            &ones * &ones,
            if n % 2 == 0 { z.clone() } else { ones.clone() }
        );
    }
}

#[test]
fn display_and_debug_write_the_rows_as_bits() {
    let a = BitMatrix::from_rows(&[[true, false, true], [false, false, true]]);
    assert_eq!(a.to_string(), "101\n001");
    assert_eq!(
        format!("{a:?}"),
        "BitMatrix { rows: 2, cols: 3, data: [\"101\", \"001\"] }"
    );
    assert_eq!(BitMatrix::zeros(0, 4).to_string(), "");
    assert_eq!(BitMatrix::zeros(2, 0).to_string(), "\n");
}

#[test]
#[should_panic(expected = "(2, 0) is out of range")]
fn reading_outside_the_matrix_panics() {
    let _ = BitMatrix::zeros(2, 2).get(2, 0);
}

#[test]
#[should_panic(expected = "(0, 64) is out of range")]
fn writing_outside_the_matrix_panics() {
    BitMatrix::zeros(2, 64).set(0, 64, true);
}

#[test]
#[should_panic(expected = "(1, 3) is out of range")]
fn flipping_outside_the_matrix_panics() {
    BitMatrix::zeros(2, 3).flip(1, 3);
}

#[test]
#[should_panic(expected = "the rows are not all of the same length")]
fn ragged_rows_panic() {
    let _ = BitMatrix::from_rows(&[vec![true], vec![true, false]]);
}

#[test]
#[should_panic(expected = "a bit beyond the last column is set")]
fn words_with_padding_bits_panic() {
    let _ = BitMatrix::from_words(1, 3, vec![0b1000]);
}

#[test]
#[should_panic(expected = "the words are not rows · ⌈cols / 64⌉")]
fn too_few_words_panic() {
    let _ = BitMatrix::from_words(2, 65, vec![0; 3]);
}

#[test]
#[should_panic(expected = "a bit beyond the last column is set")]
fn a_row_with_padding_bits_panics() {
    BitMatrix::zeros(2, 65).set_row_words(0, &[0, 2]);
}

#[test]
#[should_panic(expected = "the row 2 is out of range")]
fn a_row_outside_the_matrix_panics() {
    let _ = BitMatrix::zeros(2, 65).row_words(2);
}

#[test]
#[should_panic(expected = "the words are not ⌈cols / 64⌉")]
fn a_row_of_the_wrong_number_of_words_panics() {
    BitMatrix::zeros(2, 65).set_row_words(0, &[0]);
}

#[test]
#[should_panic(expected = "the row 2 is out of range")]
fn setting_a_row_outside_the_matrix_panics() {
    BitMatrix::zeros(2, 3).set_row_words(2, &[0]);
}

#[test]
#[should_panic(expected = "the matrix is too large")]
fn a_zero_matrix_of_too_many_words_panics() {
    let _ = BitMatrix::zeros(usize::MAX, usize::MAX);
}

#[test]
#[should_panic(expected = "the matrix is too large")]
fn an_identity_of_too_many_words_panics() {
    let _ = BitMatrix::identity(usize::MAX);
}

#[test]
#[should_panic(expected = "the matrix is too large")]
fn a_matrix_from_a_function_of_too_many_words_panics() {
    let _ = BitMatrix::from_fn(usize::MAX, usize::MAX, |_, _| true);
}

#[test]
#[should_panic(expected = "the row is out of range")]
fn swapping_a_row_outside_the_matrix_panics() {
    BitMatrix::zeros(2, 2).swap_rows(0, 2);
}

#[test]
#[should_panic(expected = "the row is out of range")]
fn adding_a_row_outside_the_matrix_panics() {
    BitMatrix::zeros(2, 2).add_row(2, 0);
}

#[test]
#[should_panic(expected = "the matrices are not of the same shape")]
fn sums_of_different_shapes_panic() {
    let _ = &BitMatrix::zeros(2, 3) + &BitMatrix::zeros(3, 2);
}

#[test]
#[should_panic(
    expected = "the columns of the left matrix are not as many as the rows of the right one"
)]
fn products_of_mismatched_shapes_panic() {
    let _ = &BitMatrix::zeros(2, 3) * &BitMatrix::zeros(2, 3);
}

#[test]
#[should_panic(
    expected = "the columns of the left matrix are not as many as the rows of the right one"
)]
fn strassen_winograd_products_of_mismatched_shapes_panic() {
    let _ = BitMatrix::zeros(200, 300).mul_strassen(&BitMatrix::zeros(200, 300), 128);
}

#[test]
#[should_panic(expected = "the matrix is not square")]
fn the_determinant_of_a_matrix_that_is_not_square_panics() {
    let _ = BitMatrix::zeros(2, 3).det();
}

#[test]
#[should_panic(expected = "the matrix is not square")]
fn the_inverse_of_a_matrix_that_is_not_square_panics() {
    let _ = BitMatrix::zeros(3, 2).inverse();
}

#[test]
#[should_panic(expected = "the right-hand side doesn't have as many rows as the matrix")]
fn a_right_hand_side_of_other_rows_panics() {
    let _ = BitMatrix::zeros(3, 2).solve(&BitMatrix::zeros(2, 1));
}
