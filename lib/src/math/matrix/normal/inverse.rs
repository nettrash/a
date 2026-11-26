//! Products by the inverse of a nonsingular matrix of integers that are themselves of integers,
//! the transformations of the Hermite forms, from their residues modulo word primes.
//!
//! `Y = s·X·M⁻¹` is `s·X·adj(M) / δ` for `δ = |det M|` (up to sign), so
//! `|y_ij| ≤ |s|·‖x_i‖₁·max |adj(M)| / δ`, and each element of the adjugate, a minor of order
//! `r − 1`, is at most the product of the lengths of the rows of `M` but the shortest (and of its
//! columns but the shortest), by Hadamard's bound. Modulo each prime `p` that does not divide `δ`,
//! `Y ≡ s·X·(M mod p)⁻¹` by [`ModMatrix`]'s inverse and product; the primes go on until their
//! product passes twice the bound, and the elements are rebuilt by the Chinese remainder theorem
//! ([`Crt`]). Nothing is guessed: when `Y` is of integers, as the callers know it is, the result
//! is `Y`. For `r × r` matrices whose elements have `b` bits and `Y` of `L` limbs, the work is
//! about `64·L/61` primes of `O(r^ω)` operations on words, against `O(r³)` operations on numbers
//! as long as the minors for a fraction-free elimination.

use super::super::modp::{Crt, Field, TopPrimes};
use super::super::{Matrix, ModMatrix};
use crate::math::numbers::Number;

/// The squared lengths of the rows of `m`.
fn row_squares(m: &Matrix<Number>) -> Vec<Number> {
    (0..m.rows())
        .map(|i| {
            m.row(i).fold(
                Number::default(),
                |s, x| {
                    if x.is_zero() { s } else { s + x * x }
                },
            )
        })
        .collect()
}

/// The product of all of `sq` but its least element, positive.
fn all_but_least(sq: &[Number]) -> Number {
    let least = (0..sq.len()).min_by(|&i, &j| sq[i].cmp(&sq[j]));
    sq.iter()
        .enumerate()
        .filter(|&(i, _)| Some(i) != least)
        .fold(Number::from(1u8), |p, (_, x)| p * x)
}

/// `s·x·m⁻¹` for the `k × r` matrix `x` and the nonsingular `r × r` matrix `m` with
/// `|det m| = det`, which must be a matrix of integers: from its residues modulo the primes below
/// `2^62` that do not divide `det`, from the top down.
pub(super) fn times_inverse(
    x: &Matrix<Number>,
    m: &Matrix<Number>,
    det: &Number,
    s: &Number,
) -> Matrix<Number> {
    let (k, r) = x.shape();
    debug_assert_eq!(m.shape(), (r, r));
    if k == 0 || r == 0 {
        return Matrix::zeros(k, r);
    }
    // The bound on the adjugate: the smaller of Hadamard's bounds by rows and by columns.
    let rows = all_but_least(&row_squares(m));
    let cols = all_but_least(&row_squares(&m.transpose()));
    let adj = rows.min(cols).isqrt() + 1u8;
    let longest = (0..k)
        .map(|i| {
            x.row(i).fold(Number::default(), |acc, v| {
                if v.is_zero() { acc } else { acc + v.abs() }
            })
        })
        .max()
        .unwrap_or_default();
    let bound = &(&(&s.abs() * &longest) * &adj) / det + 1u8;
    // `P > 2^(61·count) ≥ 2^(bits(bound) + 1) > 2·bound`.
    let count = (bound.bits() + 1).div_ceil(61) as usize;
    let mut crt = Crt::new(k * r, count);
    // The elements of `x` that are not zero, row after row: a sparse `x` (the Hermite forms of
    // random matrices are the identity but for a column) is multiplied by its rows' elements.
    let entries: Vec<(usize, usize)> = (0..k)
        .flat_map(|i| (0..r).map(move |j| (i, j)))
        .filter(|&(i, j)| !x[(i, j)].is_zero())
        .collect();
    // Below a density of 1/5 the product goes by rows of `m⁻¹` rather than by the product of the
    // matrices. Measured on an Apple M4 Max, densities 1/64 to 1 at r = 20, 50 and 100 with
    // elements of 10 and 200 bits: by rows took 0.95 to 0.99 of the time at 1/6, 1.00 to 1.03 at
    // 1/4, 1.10 to 1.77 from 1/2.
    let sparse = entries.len() * 5 < k * r;
    let mut y = vec![0u64; k * r];
    let mut taken = 0;
    let mut primes = TopPrimes::default();
    while taken < count {
        let prime = primes.next().expect("primes below 2^62");
        let Some(inv) = ModMatrix::from_matrix_prime(m, prime).inverse() else {
            // `p` divides the determinant.
            continue;
        };
        let f = Field::new(prime);
        let sp = if *s == 1u8 {
            None
        } else {
            Some(u64::try_from(&s.rem_euclid(&Number::from(prime))).expect("a residue"))
        };
        if sparse {
            // `y_i = Σ_l x_il·(m⁻¹)_l`, the scale folded into each `x_il`.
            y.iter_mut().for_each(|v| *v = 0);
            let by = Number::from(prime);
            let inv = inv.as_slice();
            for &(i, l) in &entries {
                let mut c = u64::try_from(&x[(i, l)].rem_euclid(&by)).expect("a residue");
                if let Some(sp) = sp {
                    c = f.mul(c, sp);
                }
                if c == 0 {
                    continue;
                }
                for (v, &w) in y[i * r..(i + 1) * r]
                    .iter_mut()
                    .zip(&inv[l * r..(l + 1) * r])
                {
                    *v = f.add(*v, f.mul(c, w));
                }
            }
        } else {
            let mut p = &ModMatrix::from_matrix_prime(x, prime) * &inv;
            if let Some(sp) = sp {
                p = p.scale(sp);
            }
            y.copy_from_slice(p.as_slice());
        }
        let powers = f.powers(crt.width());
        crt.add(f, prime, &y, &powers);
        taken += 1;
    }
    let mut out = vec![Number::default(); k * r];
    crt.finish_into(&mut out);
    // Row-major to column-major.
    let mut data = vec![Number::default(); k * r];
    for (e, v) in out.into_iter().enumerate() {
        data[(e % r) * k + e / r] = v;
    }
    Matrix::from_column_major(k, r, data)
}
