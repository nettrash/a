//! The Hermite normal form of a square nonsingular matrix by Pernet and Stein's method (*Fast
//! computation of Hermite normal forms of random integer matrices*, J. Number Theory 130, 2010),
//! after Micciancio and Warinschi (*A linear space algorithm for computing the Hermite normal
//! form*, ISSAC 2001): two determinants of order `n − 1`, one solution of a system, a form modulo
//! their greatest common divisor, which is small as a rule, and two rows added to it.
//!
//! For `A` of order `n`, `C` its first `n − 2` rows and `c`, `d` its last two, all on the first
//! `n − 1` columns, `W₁ = [C; c]` and `W₂ = [C; d]`: with `d₁ = det W₁`, `d₂ = det W₂` and
//! `g = gcd(d₁, d₂) = s·d₁ + t·d₂`, the matrix `W = [C; s·c + t·d]` has the determinant `g`, as the
//! determinant is linear in a row, and its rows are in the lattice of the rows of `A` (on the
//! first `n − 1` columns, with `w`, the last column of `A` on `C` and `s·c_n + t·d_n` below, beside
//! them). The Hermite form `H_W` of `W` is computed modulo `g`; that of `[W | w]`, of `n − 1` rows
//! of full rank whose pivots are those of `H_W`, is `[H_W | H_W·W⁻¹·w]`, as `U = H_W·W⁻¹` is its
//! transformation; and the lattice of `A` is that of `[W | w]` and the last two rows of `A`, which
//! are added to the form one after the other ([`add_row`]), then the elements above each pivot
//! reduced ([`reduce_above`]). The form is unique, so it is the one any method gives.
//!
//! The two determinants and the solution come from one solution of `W₁·[z₁ z₂] = [w₁ e]` (`w₁` the
//! last column of `A` on `W₁`'s rows, `e` the last unit vector) by Dixon's lifting: the least
//! common multiple `l` of its denominators divides `d₁` and is most of it as a rule, and
//! `d₁ / l` comes from its residues modulo primes, until their product passes twice Hadamard's
//! bound over `l` (one prime, as a rule); `d₂ = d₁·(1 + (d − c)·z₂)` (the determinant of a matrix
//! with one row changed), and `W⁻¹·w` by Sherman and Morrison's formula from `z₁` and `z₂`, `W`
//! being `W₁` with one row changed. As a rule `g` is 1 or small, `H_W` the identity or close to
//! it, and adding a row costs `O(n)` operations on numbers; the work is then that of the solution,
//! `O(n³)` operations on words and `O(n²·L)` for the lifting of `L` steps, against `O(n³)` on
//! numbers as long as the determinant for the form modulo it.

use super::hermite::{Rows, hermite_mod};
use super::word;
use crate::math::matrix::Matrix;
use crate::math::matrix::ModMatrix;
use crate::math::matrix::modp::{Field, TopPrimes};
use crate::math::numbers::{Divisor, Number};

/// The order from which a square nonsingular matrix takes this method rather than the form
/// modulo its determinant. Measured on an Apple M4 Max, orders 3 to 40 for elements of 1 to 100
/// bits, square, tall and wide, `hermite_form`, `hermite_normal_form` and `elementary_divisors`:
/// this method takes 0.59 to 0.83 of the time at order 6 or 7 for elements of 10 and 100 bits,
/// whose determinant passes a word (0.06 to 0.15 at 40), but 1.3 to 3.6 times as long for
/// elements of up to 7 bits while the determinant fits one and the form modulo it runs in words
/// (at every order to 16 for 1 to 3 bits); order 7 would take that loss to the matrices of 7
/// rows, so 8 is kept.
pub(super) const FROM: usize = 8;

/// The Hermite form modulo the positive multiple `det` of the determinant: in words when it is
/// below [`word::BELOW`], otherwise in numbers.
pub(super) fn hermite_mod_any(rows: Rows, n: usize, det: &Number) -> Rows {
    if det.bits() < 63 {
        word::hermite_mod(&rows, n, det)
    } else {
        hermite_mod(rows, n, det)
    }
}

/// The Hermite normal form of the rows of the square nonsingular `a`, of order at least 3, by
/// Pernet and Stein's method; `None` when the leading block of order `n − 1` is singular, which
/// the method needs to be nonsingular (or, about as rarely as a random residue is zero, when its
/// determinant is a multiple of the first prime below `2^62`).
pub(super) fn square(a: &Matrix<Number>) -> Option<Rows> {
    let n = a.rows();
    debug_assert!(n >= 3 && a.is_square());
    let k = n - 1;
    let w1 = Matrix::from_fn(k, k, |i, j| a[(i, j)].clone());
    // det W₁ modulo the first prime, the first residue of its determinant below: zero, and W₁ is
    // left to the caller's other method (as singular), not proved singular by the solution, which
    // costs about as much as one.
    let p0 = TopPrimes::default().next().expect("primes below 2^62");
    let d0 = ModMatrix::from_matrix_prime(&w1, p0).det();
    if d0 == 0 {
        return None;
    }
    let rhs = Matrix::from_fn(k, 2, |i, c| {
        if c == 0 {
            a[(i, k)].clone()
        } else {
            Number::from(u8::from(i == k - 1))
        }
    });
    let z = w1.solve(&rhs)?;
    // `l`, the least common multiple of the denominators, divides `d₁`.
    let mut l = Number::from(1u8);
    for q in z.as_slice() {
        let den = q.denom();
        if !l.is_multiple_of(den) {
            l = l.lcm(den);
        }
    }
    let d1 = det_by_divisor(&w1, &l, (p0, d0));
    // `Z₁ = d₁·z₁` and `Z₂ = d₁·z₂`, of integers (`d₁·W₁⁻¹` is the adjugate up to sign).
    let scaled = |c: usize| -> Vec<Number> {
        (0..k)
            .map(|i| {
                let q = &z[(i, c)];
                if q.is_zero() {
                    Number::default()
                } else {
                    q.numer() * &d1.div_exact(q.denom())
                }
            })
            .collect()
    };
    let (z1, z2) = (scaled(0), scaled(1));
    // `d₂ = d₁ + (d − c)·Z₂`.
    let mut d2 = d1.clone();
    for (j, x) in z2.iter().enumerate() {
        if !x.is_zero() {
            d2 += &(&a[(k, j)] - &a[(k - 1, j)]) * x;
        }
    }
    let (g, s, t) = d1.gcd_ext(&d2);
    // The row `s·c + t·d` of `W` with its last element, and `v = s·c + t·d − c`.
    let new_row: Vec<Number> = (0..n)
        .map(|j| &(&s * &a[(k - 1, j)]) + &(&t * &a[(k, j)]))
        .collect();
    let delta = &new_row[k] - &a[(k - 1, k)];
    // `W⁻¹·w = (g·Y₀ − (v·Y₀)·Z₂) / (g·d₁)` for `Y₀ = Z₁ + δ·Z₂`, by Sherman and Morrison: `W` is
    // `W₁` with its last row plus `v`, and `1 + v·z₂ = g / d₁`.
    let y0: Vec<Number> = z1
        .iter()
        .zip(&z2)
        .map(|(x, y)| {
            if y.is_zero() {
                x.clone()
            } else {
                x + &(&delta * y)
            }
        })
        .collect();
    let mut vy = Number::default();
    for (j, y) in y0.iter().enumerate() {
        let v = &new_row[j] - &a[(k - 1, j)];
        if !v.is_zero() && !y.is_zero() {
            vy += &v * y;
        }
    }
    // `g·W⁻¹·w`, of integers.
    let xg: Vec<Number> = y0
        .iter()
        .zip(&z2)
        .map(|(y, z)| {
            let mut num = &g * y;
            if !z.is_zero() {
                num -= &vy * z;
            }
            num.div_exact(&d1)
        })
        .collect();
    let hw: Rows = if g == 1u8 {
        (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if i == j {
                            Number::from(1u8)
                        } else {
                            Number::default()
                        }
                    })
                    .collect()
            })
            .collect()
    } else {
        let mut w: Rows = (0..k - 1)
            .map(|i| (0..k).map(|j| a[(i, j)].clone()).collect())
            .collect();
        w.push(new_row[..k].to_vec());
        hermite_mod_any(w, k, &g)
    };
    // `[H_W | H_W·W⁻¹·w]`.
    let mut h: Rows = hw
        .into_iter()
        .map(|mut row| {
            let mut acc = Number::default();
            for (x, y) in row.iter().zip(&xg) {
                if !x.is_zero() && !y.is_zero() {
                    acc += x * y;
                }
            }
            row.push(if g == 1u8 { acc } else { acc.div_exact(&g) });
            row
        })
        .collect();
    for i in [k - 1, k] {
        add_row(&mut h, a.row(i).cloned().collect());
    }
    if h.len() < n {
        return None;
    }
    reduce_above(&mut h);
    Some(h)
}

/// The column of the first element of `row` that is not zero.
fn lead(row: &[Number]) -> usize {
    row.iter().position(|x| !x.is_zero()).unwrap_or(row.len())
}

/// `x` modulo `d`, in `[0, |d|)`, when it is longer than `d` or negative.
#[inline]
fn shorten(x: &mut Number, d: &Divisor, bits: u64) {
    if !x.is_zero() && (x.is_negative() || x.bits() > bits) {
        *x = d.rem_euclid(x);
    }
}

/// Adds the row `v` to the rows `h` in echelon form (each row's first element positive, in a
/// column after the row before's), which stay in echelon form and span the lattice of both: `v` is
/// reduced by each row whose first element is in a column where `v` is not zero (a subtraction
/// when that element divides `v`'s, the extended Euclidean step otherwise), and what is left, if
/// anything, joins the rows. When the rows are of full rank, square and upper triangular, the
/// elements are kept below the product of their pivots, a multiple of the determinant.
pub(super) fn add_row(h: &mut Rows, mut v: Vec<Number>) {
    let n = v.len();
    let full = h.len() == n;
    let mut det = if full {
        (0..n).map(|i| h[i][i].clone()).product::<Number>()
    } else {
        Number::default()
    };
    let mut by = full.then(|| Divisor::new(&det));
    let mut i = 0;
    let mut from = 0;
    loop {
        let Some(c) = (from..n).find(|&j| !v[j].is_zero()) else {
            return;
        };
        while i < h.len() && lead(&h[i]) < c {
            i += 1;
        }
        if i == h.len() || lead(&h[i]) > c {
            if v[c].is_negative() {
                v.iter_mut().for_each(|x| *x = -&*x);
            }
            h.insert(i, v);
            return;
        }
        let row = &mut h[i];
        let (q, r) = v[c].div_rem(&row[c]);
        if r.is_zero() {
            for (x, y) in v[c..].iter_mut().zip(&row[c..]) {
                if !y.is_zero() {
                    *x -= &q * y;
                }
            }
        } else {
            let (e, x, y) = row[c].gcd_ext(&v[c]);
            let a = row[c].div_exact(&e);
            let b = v[c].div_exact(&e);
            for (rj, vj) in row[c..].iter_mut().zip(v[c..].iter_mut()) {
                if rj.is_zero() && vj.is_zero() {
                    continue;
                }
                let new_r = &(&x * &*rj) + &(&y * &*vj);
                let new_v = &(&a * &*vj) - &(&b * &*rj);
                *rj = new_r;
                *vj = new_v;
            }
            if full {
                // The pivot went from `a·e` to `e`.
                det = (&det / &a).abs();
                by = Some(Divisor::new(&det));
            }
        }
        if let Some(by) = &by {
            let bits = det.bits();
            for x in &mut v[c + 1..] {
                shorten(x, by, bits);
            }
            for x in &mut h[i][c + 1..] {
                shorten(x, by, bits);
            }
        }
        from = c + 1;
        i += 1;
    }
}

/// The square upper triangular `h` of full rank, its pivots positive, with each element above a
/// pivot brought into `[0, pivot)` by the pivot's row, column after column from the left, the
/// elements after it kept below the product of the pivots after it: its Hermite normal form.
pub(super) fn reduce_above(h: &mut Rows) {
    let n = h.len();
    // `after[l]`, the product of the pivots from `l` on.
    let mut after = vec![Number::from(1u8); n + 1];
    for l in (0..n).rev() {
        after[l] = &after[l + 1] * &h[l][l];
    }
    for l in 1..n {
        let (above, below) = h.split_at_mut(l);
        let pivot_row = &below[0];
        let pivot = &pivot_row[l];
        let by = Divisor::new(&after[l + 1]);
        let bits = after[l + 1].bits();
        for row in above {
            if row[l].is_zero() {
                continue;
            }
            let q = row[l].div_floor(pivot);
            if q.is_zero() {
                continue;
            }
            for (x, y) in row[l..].iter_mut().zip(&pivot_row[l..]) {
                if !y.is_zero() {
                    *x -= &q * y;
                }
            }
            for x in &mut row[l + 1..] {
                shorten(x, &by, bits);
            }
        }
    }
}

/// The determinant of the square `a` given a divisor `l` of it: `det / l` from its residues
/// modulo primes below `2^62`, from the top down, by the Chinese remainder theorem, until their
/// product passes twice Hadamard's bound over `l`, as the residue of least magnitude, times `l`;
/// `first` is the prime and residue of `det` already known.
fn det_by_divisor(a: &Matrix<Number>, l: &Number, first: (u64, u64)) -> Number {
    let n = a.rows();
    let mut sq = Number::from(1u8);
    for i in 0..n {
        let mut s = Number::default();
        for x in a.row(i) {
            if !x.is_zero() {
                s += x * x;
            }
        }
        sq *= &s;
    }
    let bound = &(&(sq.isqrt() + 1u8) / l) + 1u8;
    let twice = &bound * 2u8;
    let mut c = Number::default();
    let mut q = Number::from(1u8);
    let mut primes = TopPrimes::default();
    while q <= twice {
        let prime = primes.next().expect("primes below 2^62");
        let f = Field::new(prime);
        let lp = u64::try_from(&l.rem_euclid(&Number::from(prime))).expect("a residue");
        if lp == 0 {
            continue;
        }
        let dp = if prime == first.0 {
            first.1
        } else {
            ModMatrix::from_matrix_prime(a, prime).det()
        };
        let r = f.mul(dp, f.inv(lp));
        // Garner: `c + q·((r − c)·q⁻¹ mod p)`.
        let cp = u64::try_from(&c.rem_euclid(&Number::from(prime))).expect("a residue");
        let qp = u64::try_from(&q.rem_euclid(&Number::from(prime))).expect("a residue");
        let t = f.mul(f.sub(r, cp), f.inv(qp));
        if t != 0 {
            c += &(&q * t);
        }
        q *= prime;
    }
    let half = &q >> 1usize;
    let c = if c > half { c - &q } else { c };
    &c * l
}
