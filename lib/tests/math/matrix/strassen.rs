//! Strassen–Winograd's products of matrices of numbers against the classical product written
//! here: the same numbers, little-endian, at and around the crossovers for each length of the
//! elements (96 up to 16 bytes, 72 up to 48, 64 up to 64, 48 up to 128, 32 up to 256, 16 up to
//! 1024, 12 beyond), below the orders from which the residues take the product, with odd and even
//! dimensions, rectangles, zeros, both signs and both byte orders.

use super::common::{Rng, size};
use a::math::matrix::Matrix;
use a::math::numbers::{ByteOrder, Number};

/// A random matrix of numbers of `bytes` bytes, of either sign and byte order, with about one
/// element in `zeros` zero.
fn random(r: &mut Rng, m: usize, n: usize, bytes: usize, zeros: u64) -> Matrix<Number> {
    Matrix::from_fn(m, n, |_, _| {
        if r.next().is_multiple_of(zeros) {
            return Number::default();
        }
        let order = if r.next() & 1 == 1 {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        };
        let x = Number::new((0..bytes).map(|_| r.next() as u8).collect(), order);
        if r.next() & 1 == 1 { -x } else { x }
    })
}

/// The classical product, element by element.
fn classical(a: &Matrix<Number>, b: &Matrix<Number>) -> Matrix<Number> {
    Matrix::from_fn(a.rows(), b.cols(), |i, j| {
        let mut s = Number::default();
        for k in 0..a.cols() {
            s += &a[(i, k)] * &b[(k, j)];
        }
        s
    })
}

/// Checks `a·b` against the classical product, and that its elements are little-endian.
fn check(a: &Matrix<Number>, b: &Matrix<Number>, what: &str) {
    let c = a * b;
    assert_eq!(c, classical(a, b), "{what}");
    assert!(
        c.as_slice()
            .iter()
            .all(|x| x.byte_order == ByteOrder::LittleEndian),
        "{what}"
    );
}

#[test]
fn long_elements_from_sixteen_rows() {
    // Elements of 300 bytes: the crossover is 16; from 32 the residues take the product.
    let mut r = Rng(0x57_0300);
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(2, 2, 2)]
    } else {
        &[
            (15, 16, 17),
            (16, 16, 16),
            (17, 17, 17),
            (16, 17, 16),
            (33, 32, 35),
            (32, 32, 32),
        ]
    };
    for &(m, k, n) in shapes {
        let (a, b) = (random(&mut r, m, k, 300, 7), random(&mut r, k, n, 300, 7));
        check(&a, &b, &format!("{m} × {k} × {n}"));
    }
}

#[test]
fn elements_of_a_few_limbs_from_forty_eight_and_thirty_two_rows() {
    // Elements of 72 bytes: the crossover is 48, the residues from 57 rows; of 140 bytes: 32.
    let mut r = Rng(0x57_0100);
    let shapes: &[(usize, usize, usize, usize)] = if cfg!(miri) {
        &[]
    } else {
        &[
            (72, 47, 48, 49),
            (72, 48, 48, 48),
            (72, 49, 49, 49),
            (72, 50, 52, 49),
            (72, 56, 56, 56),
            (140, 31, 32, 33),
            (140, 32, 32, 32),
            (140, 33, 33, 33),
            (100, 64, 64, 65),
        ]
    };
    for &(bytes, m, k, n) in shapes {
        let (a, b) = (
            random(&mut r, m, k, bytes, 5),
            random(&mut r, k, n, bytes, 5),
        );
        check(&a, &b, &format!("{bytes} bytes, {m} × {k} × {n}"));
    }
}

#[test]
fn elements_of_two_to_eight_limbs_from_ninety_six_seventy_two_and_sixty_four_rows() {
    // Elements of 16 bytes: the crossover is 96; of 24 bytes: 72; of 56 bytes: 64, the residues
    // from 69 rows. One product below and one at each, for the time of the debug profile.
    let mut r = Rng(0x57_0016);
    let shapes: &[(usize, usize, usize, usize)] = if cfg!(miri) {
        &[]
    } else {
        &[
            (16, 95, 96, 96),
            (16, 96, 97, 96),
            (24, 71, 72, 72),
            (24, 73, 72, 72),
            (56, 63, 64, 64),
            (56, 64, 65, 64),
        ]
    };
    for &(bytes, m, k, n) in shapes {
        let (a, b) = (
            random(&mut r, m, k, bytes, 4),
            random(&mut r, k, n, bytes, 4),
        );
        check(&a, &b, &format!("{bytes} bytes, {m} × {k} × {n}"));
    }
}

#[test]
fn very_long_elements_from_twelve_rows() {
    // Elements of 1100 bytes: the crossover is 12.
    let mut r = Rng(0x57_1100);
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[]
    } else {
        &[(11, 12, 13), (12, 12, 12), (13, 13, 12)]
    };
    for &(m, k, n) in shapes {
        let (a, b) = (random(&mut r, m, k, 1100, 6), random(&mut r, k, n, 1100, 6));
        check(&a, &b, &format!("{m} × {k} × {n}"));
    }
}

#[test]
fn short_elements_from_one_hundred_and_twenty_eight_rows() {
    // Elements of 1 and 8 bytes, whose sums can pass a word: the sums of products in machine
    // arithmetic. Only one product, for the time of the classical product in the debug profile.
    let mut r = Rng(0x57_0008);
    let n = size(129, 3);
    for bytes in [1, 8] {
        let (a, b) = (
            random(&mut r, n, n - 1, bytes, 4),
            random(&mut r, n - 1, n + 1, bytes, 4),
        );
        check(&a, &b, &format!("{bytes} bytes"));
    }
}

#[test]
fn products_with_a_zero_dimension_or_a_single_row() {
    let mut r = Rng(0x57_0000);
    for (m, k, n) in [
        (0, 20, 20),
        (20, 0, 20),
        (20, 20, 0),
        (1, 40, 40),
        (40, 1, 40),
        (40, 40, 1),
    ] {
        let (a, b) = (random(&mut r, m, k, 300, 3), random(&mut r, k, n, 300, 3));
        check(&a, &b, &format!("{m} × {k} × {n}"));
    }
}

#[test]
fn matrix_products_agree_with_determinants() {
    // det(A·B) = det(A)·det(B), with the product split by the recursion.
    let mut r = Rng(0x57_0DE7);
    let n = size(20, 3);
    let (a, b) = (random(&mut r, n, n, 300, 9), random(&mut r, n, n, 300, 9));
    assert_eq!((&a * &b).det(), &a.det() * &b.det());
}
