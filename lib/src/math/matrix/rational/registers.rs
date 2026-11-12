//! The exact algorithms of small matrices in registers: for matrices of up to [`ORDER`] rows whose
//! elements are rationals in words or integers of one limb, and whose minors Hadamard's bound
//! keeps below `2^61`; and the product of matrices of rationals in words.
//!
//! The rows are cleared of their denominators into an array of `i64`s on the stack, the words of
//! each rational read in place; one product of the rows' norms in floating point then bounds every
//! minor of the matrix, so that Bareiss's elimination and the fraction-free back substitution can
//! run with no check at all: a step `(a·p - b·c) / d` is two products in 128 bits, a subtraction
//! and an exact division by the previous pivot, the processor's division when the pivot divides
//! one or two elements and otherwise a product by the inverse of its odd part modulo `2^64`; the
//! sums of the back substitution fit in 128 bits, and every quotient in a register. Each element
//! of a result is then one or two greatest common divisors of words and a rational written in
//! words. Order 2 takes Cramer's rule instead, in 128 bits with no bound to check. Nothing is
//! allocated but the result's vector, against the vectors of the elimination of the parent module
//! (an [`Int`] for each element, one for the multiples, one for the divisors), which cost more than
//! the arithmetic at these orders.
//!
//! When a matrix is in words but its bound passes `2^61`, its cleared integers go to the parent
//! module's elimination, which takes them over without reading the matrix again.
//!
//! [`Int`]: super::Int

use super::super::Matrix;
use super::Int;
use crate::math::numbers::{ByteOrder, Magnitude, Number, fraction};
use crate::math::rational::Rational;

/// The largest order of a matrix in registers.
pub(super) const ORDER: usize = 8;

/// The most columns: a matrix and as many right-hand sides, or the identity beside it.
const COLS: usize = 2 * ORDER;

/// The bound on the product of the squares of the rows' norms below which every minor is below
/// `2^61`, by Hadamard's inequality: `2^122`. The floating-point product's rounding, below one part
/// in `2^45`, is far inside the margin that the arithmetic leaves (the steps need minors below
/// `2^63`, the sums of the back substitution products of them below `2^123.8`).
const BOUND: f64 = 5.316_911_983_139_664e36;

/// The greatest common divisor of two words, either of which may be zero.
#[inline(always)]
fn gcd(x: u64, y: u64) -> u64 {
    fraction::gcd_words(x, y)
}

/// `x / d` for a multiple `x` of `d` whose quotient is below `2^63` in magnitude, by the
/// inverse `inv` of `d`'s odd part modulo `2^64` and its zero bits `shift`: `q·(d >> shift) =
/// x >> shift` holds modulo `2^64` too.
#[inline(always)]
fn by_inverse(x: i128, shift: u32, inv: u64) -> i64 {
    ((x >> shift) as u64).wrapping_mul(inv) as i64
}

/// `x / d` for a multiple `x` of `d` whose quotient is below `2^63` in magnitude, by the
/// processor's division, of 64 bits when `x` fits: cheaper than an inverse for a divisor that
/// divides one or two elements.
#[inline(always)]
fn by_division(x: i128, d: i64) -> i64 {
    match i64::try_from(x) {
        Ok(x) => x / d,
        Err(_) => (x / i128::from(d)) as i64,
    }
}

/// The zero bits at the low end of the non-zero `d` and the inverse of its odd part modulo
/// `2^64`, sign included.
#[inline(always)]
fn inverse_of(d: i64) -> (u32, u64) {
    debug_assert!(d != 0);
    let shift = d.trailing_zeros();
    (shift, fraction::inverse_word((d >> shift) as u64))
}

/// From this many elements divided by one pivot, the division is a product by its inverse.
const INVERSE_FROM: usize = 3;

/// What the registers are asked for.
#[derive(Clone, Copy)]
pub(super) enum Op {
    /// The determinant of a matrix of rationals.
    Det,
    /// The determinant of a matrix of integers.
    DetInteger,
    /// The solution of `A·X = B`.
    Solve,
    /// The inverse, as rationals.
    Inverse,
    /// The adjugate and the determinant of a matrix of integers.
    Adjugate,
}

/// What the registers give.
pub(super) enum Out {
    /// A determinant of rationals.
    Rational(Rational),
    /// A determinant of integers.
    Number(Number),
    /// The elements of a solution or of an inverse, column after column, or `None` for a singular
    /// matrix.
    Rationals(Option<Vec<Rational>>),
    /// The adjugate, column after column, and the determinant, or `None` for a singular matrix.
    Adjugate(Option<(Vec<Number>, Number)>),
}

/// What became of a matrix.
pub(super) enum Outcome {
    /// The result, in registers.
    Done(Out),
    /// The matrix in words, whose minors may pass the registers: its integers read, row after
    /// row, without the identity, and its rows' multiples, for the elimination on limbs.
    Ints(Vec<Int>, Vec<Int>),
    /// A matrix with an element beyond the words, or too large.
    No,
}

/// The matrices whose elements, side by side, make the rows.
#[derive(Clone, Copy)]
pub(super) enum Source<'a> {
    /// Of rationals.
    Rationals(&'a [&'a Matrix<Rational>]),
    /// Of integers.
    Numbers(&'a [&'a Matrix<Number>]),
}

/// `op` on the matrices `src`, in registers when they fit, the results in the byte order `order`.
pub(super) fn run(src: Source<'_>, op: Op, order: ByteOrder) -> Outcome {
    let n = match src {
        Source::Rationals(p) => p[0].rows(),
        Source::Numbers(p) => p[0].rows(),
    };
    // The order 2 by Cramer's rule, and an array of four rows for the next, cheaper to fill.
    // The top left element first, which settles most matrices that cannot be in registers before
    // an array is filled: one in numbers, or an integer from `2^61` (`2^63` for Cramer's rule),
    // whose square alone passes Hadamard's bound.
    let first = match src {
        Source::Rationals(p) => p[0].as_slice().first().is_some_and(|x| x.word().is_some()),
        Source::Numbers(p) => p[0].as_slice().first().is_some_and(|x| {
            let m = Magnitude::of(x);
            m.bytes.len() <= 8 && (m.small() as u64) >> if n == 2 { 63 } else { 61 } == 0
        }),
    };
    if !first {
        return Outcome::No;
    }
    let width: usize = match src {
        Source::Rationals(p) => p.iter().map(|p| p.cols()).sum(),
        Source::Numbers(p) => p.iter().map(|p| p.cols()).sum(),
    };
    if n == 2 && width <= 4 {
        // `None` only when an element is beyond the words, which the array would find too.
        two(src, op, order).map_or(Outcome::No, Outcome::Done)
    } else if n <= 4 {
        run_in::<4, 32>(src, op, order)
    } else if n <= ORDER {
        run_in::<ORDER, { ORDER * COLS }>(src, op, order)
    } else {
        Outcome::No
    }
}

/// [`run`] in an array of `N` rows and `C = 2·N²` integers.
#[inline(always)]
fn run_in<const N: usize, const C: usize>(src: Source<'_>, op: Op, order: ByteOrder) -> Outcome {
    let identity = matches!(op, Op::Inverse | Op::Adjugate);
    let (n, width) = match src {
        Source::Rationals(p) => (p[0].rows(), p.iter().map(|p| p.cols()).sum()),
        Source::Numbers(p) => (p[0].rows(), p.iter().map(|p| p.cols()).sum()),
    };
    let cols = width + if identity { n } else { 0 };
    if n == 0 || n > N || n * cols > C {
        return Outcome::No;
    }
    let mut r = Regs::<N, C> {
        m: [0; C],
        l: [0; N],
        n,
        cols,
        width,
    };
    if identity {
        for i in 0..n {
            r.m[i * cols + width + i] = 1;
        }
    }
    let read = match src {
        Source::Rationals(p) => r.read_rationals(p),
        Source::Numbers(p) => r.read_numbers(p, true),
    };
    if read.is_none() {
        return Outcome::No;
    }
    if !r.bounded() {
        let (m, l) = r.ints();
        return Outcome::Ints(m, l);
    }
    Outcome::Done(match op {
        Op::Det => Out::Rational(r.det(order)),
        Op::DetInteger => Out::Number(r.det_integer(order)),
        Op::Solve => Out::Rationals(r.solve(order)),
        Op::Inverse => Out::Rationals(r.inverse(order)),
        Op::Adjugate => Out::Adjugate(r.adjugate(order)),
    })
}

/// A matrix of integers in registers, row after row: `[A | B]` with `A` square of order at most
/// `N`, the rows of a matrix of rationals multiplied by the least common multiples of their
/// denominators, in an array of `C` integers.
struct Regs<const N: usize, const C: usize> {
    /// The integers, `cols` to a row.
    m: [i64; C],
    /// The least common multiple of each row's denominators, 1 for integers.
    l: [u64; N],
    /// The number of rows, the order of `A`.
    n: usize,
    /// The columns of `[A | B]`.
    cols: usize,
    /// The columns read from the matrices, before an identity beside them.
    width: usize,
}

/// The integer of the sign `negative` and the magnitude `v`, when it is below `2^63`.
#[inline(always)]
fn signed(negative: bool, v: u64) -> Option<i64> {
    if v >> 63 != 0 {
        return None;
    }
    Some(if negative { -(v as i64) } else { v as i64 })
}

impl<const N: usize, const C: usize> Regs<N, C> {
    /// Reads the matrices of rationals `parts` side by side, each row multiplied by the least
    /// common multiple of its denominators; `None` unless every element is in words and every
    /// integer and multiple is below `2^63`.
    #[inline(always)]
    fn read_rationals(&mut self, parts: &[&Matrix<Rational>]) -> Option<()> {
        let (cols, width) = (self.cols, self.width);
        for i in 0..self.n {
            // The least common multiple of the row's denominators.
            let mut l = 1u64;
            for p in parts {
                for x in p.row(i) {
                    let w = x.word()?;
                    if l == 1 {
                        l = w.den;
                    } else if w.den != 1 && w.den != l {
                        let g = gcd(l, w.den);
                        l = (l / g).checked_mul(w.den)?;
                    }
                }
            }
            if l >> 63 != 0 {
                return None;
            }
            let row = &mut self.m[i * cols..i * cols + width];
            let mut at = 0;
            for p in parts {
                for x in p.row(i) {
                    let w = x.word()?;
                    let v = if w.den == l {
                        w.num
                    } else {
                        w.num.checked_mul(l / w.den)?
                    };
                    row[at] = signed(w.negative, v)?;
                    at += 1;
                }
            }
            self.l[i] = l;
        }
        Some(())
    }

    /// Reads the matrices of integers `parts` side by side; `None` unless every integer is below
    /// `2^63` in magnitude and, when `bound` is set, the rows read so far keep Hadamard's bound
    /// below [`BOUND`]: the elimination on limbs reads integers as fast as this, so a matrix that
    /// cannot stay in registers stops here.
    #[inline(always)]
    fn read_numbers(&mut self, parts: &[&Matrix<Number>], bound: bool) -> Option<()> {
        let (cols, width) = (self.cols, self.width);
        let mut product = 1.0f64;
        for i in 0..self.n {
            self.l[i] = 1;
            let row = &mut self.m[i * cols..i * cols + width];
            let mut at = 0;
            let mut norm = 0.0f64;
            for p in parts {
                for x in p.row(i) {
                    let m = Magnitude::of(x);
                    if m.bytes.len() > 8 {
                        return None;
                    }
                    let v = signed(m.negative, m.small() as u64)?;
                    norm += (v as f64) * (v as f64);
                    row[at] = v;
                    at += 1;
                }
            }
            product *= norm.max(1.0);
            if bound && product >= BOUND {
                return None;
            }
        }
        Some(())
    }

    /// Whether every minor of the matrix is below `2^61`: whether the product of the squares of
    /// the rows' norms, each at least 1, is below [`BOUND`].
    #[inline(always)]
    fn bounded(&self) -> bool {
        let mut p = 1.0f64;
        for row in self.m.chunks_exact(self.cols).take(self.n) {
            let s: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
            p *= s.max(1.0);
        }
        p < BOUND
    }

    /// The integers read from the matrices, row after row, and the rows' multiples, for the
    /// elimination beyond the registers.
    #[cold]
    #[inline(never)]
    fn ints(&self) -> (Vec<Int>, Vec<Int>) {
        let mut m = Vec::with_capacity(self.n * self.width);
        for row in self.m.chunks_exact(self.cols).take(self.n) {
            m.extend(row[..self.width].iter().map(|&x| Int::S(x)));
        }
        let l = self.l[..self.n].iter().map(|&x| Int::S(x as i64)).collect();
        (m, l)
    }

    /// Bareiss's elimination of `[A | B]` with the pivots on the diagonal, each the first element
    /// that is not zero at or below its row: the number of row swaps, or `None` when `A` is
    /// singular. The inverses of the pivots it made are left in `inv`, 0 for those it did not.
    #[inline(always)]
    fn eliminate(&mut self, inv: &mut [(u32, u64); N]) -> Option<usize> {
        let (n, cols) = (self.n, self.cols);
        let m = &mut self.m[..n * cols];
        let mut swaps = 0;
        for c in 0..n {
            let p = (c..n).find(|&i| m[i * cols + c] != 0)?;
            if p != c {
                let (a, b) = m.split_at_mut(p * cols);
                a[c * cols..(c + 1) * cols].swap_with_slice(&mut b[..cols]);
                swaps += 1;
            }
            if c + 1 == n {
                break;
            }
            // Minors below `2^61`: each product below `2^122`, and every quotient a minor.
            if c == 0 {
                step(m, c, cols, |x| x as i64);
            } else {
                let d = m[(c - 1) * cols + c - 1];
                if (n - c - 1) * (cols - c - 1) < INVERSE_FROM {
                    step(m, c, cols, |x| by_division(x, d));
                } else {
                    let (shift, v) = inverse_of(d);
                    inv[c - 1] = (shift, v);
                    step(m, c, cols, |x| by_inverse(x, shift, v));
                }
            }
        }
        Some(swaps)
    }

    /// The determinant of `A`, the rows' multiples divided out, in the byte order `order`.
    #[inline(always)]
    fn det(&mut self, order: ByteOrder) -> Rational {
        let n = self.n;
        let Some(swaps) = self.eliminate(&mut [(0, 0); N]) else {
            return Rational::from_word(false, 0, 1, order);
        };
        let d = self.m[(n - 1) * self.cols + n - 1];
        let negative = (d < 0) != (swaps % 2 == 1);
        over(
            negative,
            u128::from(d.unsigned_abs()),
            &mut self.l[..n],
            order,
        )
    }

    /// The elimination and the back substitution for the right-hand sides `B`: `None` when `A` is
    /// singular, and otherwise `f(i, k, y, d)` for each element of the solution, column after
    /// column, where `y / d` is the element in row `i` of the right-hand side `k` and `d` the last
    /// pivot.
    #[inline(always)]
    fn back(&mut self, mut f: impl FnMut(usize, usize, i64, i64)) -> Option<usize> {
        let mut inv = [(0, 0); N];
        let swaps = self.eliminate(&mut inv)?;
        let (n, cols) = (self.n, self.cols);
        let m = &self.m;
        let d = m[(n - 1) * cols + n - 1];
        let by_inverses = cols - n >= INVERSE_FROM;
        if by_inverses {
            for (i, v) in inv[..n - 1].iter_mut().enumerate() {
                if v.1 == 0 {
                    *v = inverse_of(m[i * cols + i]);
                }
            }
        }
        let mut y = [0i64; N];
        for k in n..cols {
            y[n - 1] = m[(n - 1) * cols + k];
            for i in (0..n - 1).rev() {
                let row = &m[i * cols..(i + 1) * cols];
                // `D·b_i - Σ u_ij·y_j`: Cramer's numerators and minors below `2^61`, at most 9
                // products below `2^122`.
                let mut acc = d as i128 * row[k] as i128;
                for (&u, &yj) in row[i + 1..n].iter().zip(&y[i + 1..n]) {
                    acc -= u as i128 * yj as i128;
                }
                y[i] = if by_inverses {
                    by_inverse(acc, inv[i].0, inv[i].1)
                } else {
                    by_division(acc, row[i])
                };
            }
            for (i, &yi) in y[..n].iter().enumerate() {
                f(i, k - n, yi, d);
            }
        }
        Some(swaps)
    }

    /// The solution `X` of `A·X = B`, column after column, in the byte order `order`; `None` when
    /// `A` is singular.
    #[inline(always)]
    fn solve(&mut self, order: ByteOrder) -> Option<Vec<Rational>> {
        let mut x = Vec::with_capacity(self.n * (self.cols - self.n));
        self.back(|_, _, y, d| x.push(fraction_of(y, d, 1, order)))?;
        Some(x)
    }

    /// The inverse of the matrix of rationals `A`, of which this is `[L·A | I]`: `(L·A)⁻¹·L`,
    /// column `j` of the inverse of the integers times the multiple `l_j`; in the byte order
    /// `order`, or `None` when `A` is singular.
    #[inline(always)]
    fn inverse(&mut self, order: ByteOrder) -> Option<Vec<Rational>> {
        let mut x = Vec::with_capacity(self.n * self.n);
        let l = self.l;
        self.back(|_, j, y, d| x.push(fraction_of(y, d, l[j], order)))?;
        Some(x)
    }

    /// The adjugate of the matrix of integers `A`, of which this is `[A | I]`, column after column,
    /// and its determinant, in the byte order `order`; `None` when `A` is singular.
    fn adjugate(&mut self, order: ByteOrder) -> Option<(Vec<Number>, Number)> {
        let n = self.n;
        let mut adj = Vec::with_capacity(n * n);
        let mut det = 0;
        let swaps = self.back(|_, _, y, d| {
            adj.push(y);
            det = d;
        })?;
        let odd = swaps % 2 == 1;
        let sign = |x: i64| if odd { -x } else { x };
        let adj = adj
            .into_iter()
            .map(|y| Number::new_from_i64(sign(y), order))
            .collect();
        Some((adj, Number::new_from_i64(sign(det), order)))
    }

    /// The determinant of the matrix of integers `A`, in the byte order `order`.
    #[inline(always)]
    fn det_integer(&mut self, order: ByteOrder) -> Number {
        let n = self.n;
        let Some(swaps) = self.eliminate(&mut [(0, 0); N]) else {
            return Number::new_from_u64(0, order);
        };
        let d = self.m[(n - 1) * self.cols + n - 1];
        Number::new_from_i64(if swaps % 2 == 1 { -d } else { d }, order)
    }
}

/// The step `c` of Bareiss's elimination of the rows `m` of `cols` integers below the pivot's:
/// `x ← (x·p - b·y) / d` for each element `x` right of the pivot's column, with `y` the element of
/// the pivot's row in its column, `p` the pivot, `b` the row's element in the pivot's column, and
/// the exact division by the previous pivot `d` given as `div`.
#[inline(always)]
fn step(m: &mut [i64], c: usize, cols: usize, div: impl Fn(i128) -> i64) {
    let (top, rest) = m.split_at_mut((c + 1) * cols);
    let pivot_row = &top[c * cols..];
    let pivot = pivot_row[c] as i128;
    for row in rest.chunks_exact_mut(cols) {
        let b = row[c] as i128;
        for (x, &y) in row[c + 1..].iter_mut().zip(&pivot_row[c + 1..]) {
            *x = div(*x as i128 * pivot - b * y as i128);
        }
    }
}

/// The rational `±num / ∏ l` in lowest terms, in the byte order `order`: `num` reduced by each
/// multiple in turn, so that what is left of it is prime to what is left of each.
#[inline(always)]
fn over(negative: bool, mut num: u128, l: &mut [u64], order: ByteOrder) -> Rational {
    let mut den = 1u128;
    let mut wide = true;
    for l in l.iter_mut() {
        if *l != 1 {
            let g = gcd(rem(num, *l), *l);
            if g != 1 {
                num = quo(num, g);
                *l /= g;
            }
            match den.checked_mul(u128::from(*l)) {
                Some(x) => den = x,
                None => wide = false,
            }
        }
    }
    if wide {
        return Rational::from_u128(negative && num != 0, num, den, order);
    }
    long_over(negative, num, l, order)
}

/// The rational `±num / ∏ l`, already in lowest terms, whose denominator passes 128 bits.
#[cold]
#[inline(never)]
fn long_over(negative: bool, num: u128, l: &[u64], order: ByteOrder) -> Rational {
    let den = l.iter().fold(Number::new_from_u64(1, order), |p, &l| p * l);
    let mut num = Number::new_from_u128(num, order);
    num.negative = negative && !num.is_zero();
    Rational::from_reduced(num, den)
}

/// The rational `y·s / d` in lowest terms, in the byte order `order`, for `d` not zero and `s`
/// positive, of 128 bits: in words through [`fraction_of`] when `y` and `d` fit in them.
#[inline]
fn fraction_wide(y: i128, d: i128, s: u64, order: ByteOrder) -> Rational {
    if let (Ok(y), Ok(d)) = (i64::try_from(y), i64::try_from(d)) {
        return fraction_of(y, d, s, order);
    }
    long_fraction(y, d, s, order)
}

/// [`fraction_wide`] beyond the words.
#[cold]
#[inline(never)]
fn long_fraction(y: i128, d: i128, s: u64, order: ByteOrder) -> Rational {
    if y == 0 {
        return Rational::from_word(false, 0, 1, order);
    }
    let negative = (y < 0) != (d < 0);
    let (mut a, mut b) = (y.unsigned_abs(), d.unsigned_abs());
    let g = fraction::gcd_wide(a, b);
    a /= g;
    b /= g;
    let g = gcd(rem(b, s), s);
    let (s, b) = (s / g, b / u128::from(g));
    match a.checked_mul(u128::from(s)) {
        Some(a) => Rational::from_u128(negative, a, b, order),
        None => {
            let mut num = Number::new_from_u128(a, order) * s;
            num.negative = negative;
            Rational::from_reduced(num, Number::new_from_u128(b, order))
        }
    }
}

/// `op` on matrices of order 2 with at most 2 right-hand sides, by Cramer's rule on the rows
/// cleared of their denominators, in 128 bits: `None` when an element is not in words or an integer
/// or a multiple reaches `2^63`.
#[inline(always)]
fn two(src: Source<'_>, op: Op, order: ByteOrder) -> Option<Out> {
    let width = match src {
        Source::Rationals(p) => p.iter().map(|p| p.cols()).sum(),
        Source::Numbers(p) => p.iter().map(|p| p.cols()).sum(),
    };
    debug_assert!(width <= 4);
    let mut r = Regs::<2, 8> {
        m: [0; 8],
        l: [0; 2],
        n: 2,
        cols: width,
        width,
    };
    match src {
        Source::Rationals(p) => r.read_rationals(p)?,
        Source::Numbers(p) => r.read_numbers(p, false)?,
    }
    let w = width;
    let m = |i: usize, j: usize| i128::from(r.m[i * w + j]);
    let (a, b, c, e) = (m(0, 0), m(0, 1), m(1, 0), m(1, 1));
    // Products of magnitudes below `2^63`, below `2^126`, and their difference.
    let d = a * e - b * c;
    let integer = |x: i128| Int::wide(x).number(order);
    Some(match op {
        Op::Det => Out::Rational(over(d < 0, d.unsigned_abs(), &mut r.l, order)),
        Op::DetInteger => Out::Number(integer(d)),
        Op::Adjugate => Out::Adjugate((d != 0).then(|| {
            let adj = [e, -c, -b, a].map(integer).into();
            (adj, integer(d))
        })),
        Op::Inverse => Out::Rationals((d != 0).then(|| {
            let (l0, l1) = (r.l[0], r.l[1]);
            vec![
                fraction_wide(e, d, l0, order),
                fraction_wide(-c, d, l0, order),
                fraction_wide(-b, d, l1, order),
                fraction_wide(a, d, l1, order),
            ]
        })),
        Op::Solve => {
            if d == 0 {
                Out::Rationals(None)
            } else {
                let mut x = Vec::with_capacity(2 * (w - 2));
                for k in 2..w {
                    let (y0, y1) = (m(0, k), m(1, k));
                    // Below `2^63` each, so the products and differences fit.
                    x.push(fraction_wide(e * y0 - b * y1, d, 1, order));
                    x.push(fraction_wide(a * y1 - c * y0, d, 1, order));
                }
                Out::Rationals(Some(x))
            }
        }
    })
}

/// The rational `y·s / d` in lowest terms, in the byte order `order`, for `d` not zero and `s`
/// positive: by the greatest common divisors of `y` and `d` and of `s` and what is left of `d`.
#[inline]
fn fraction_of(y: i64, d: i64, s: u64, order: ByteOrder) -> Rational {
    if y == 0 {
        return Rational::from_word(false, 0, 1, order);
    }
    let negative = (y < 0) != (d < 0);
    let (mut a, mut b) = (y.unsigned_abs(), d.unsigned_abs());
    if b != 1 {
        let g = gcd(a, b);
        a /= g;
        b /= g;
    }
    if s == 1 {
        return Rational::from_word(negative, a, b, order);
    }
    let g = gcd(s, b);
    Rational::from_u128(
        negative,
        u128::from(a) * u128::from(s / g),
        u128::from(b / g),
        order,
    )
}

/// The elements of a row or a column `xs` of rationals in words times the least common multiple of
/// their denominators, into `out`, their magnitudes or-ed into `bits`: the multiple, or `None` when
/// an element is not in words or an integer or the multiple reaches `2^63`.
#[inline]
fn clear_line<'a>(
    xs: impl Iterator<Item = &'a Rational> + Clone,
    out: &mut [i64],
    bits: &mut u64,
) -> Option<u64> {
    let mut l = 1u64;
    for x in xs.clone() {
        let w = x.word()?;
        if l == 1 {
            l = w.den;
        } else if w.den != 1 && w.den != l {
            let g = gcd(l, w.den);
            l = (l / g).checked_mul(w.den)?;
        }
    }
    if l >> 63 != 0 {
        return None;
    }
    for (o, x) in out.iter_mut().zip(xs) {
        let w = x.word()?;
        let v = if w.den == l {
            w.num
        } else {
            w.num.checked_mul(l / w.den)?
        };
        *bits |= v;
        *o = signed(w.negative, v)?;
    }
    Some(l)
}

/// The elements of a small buffer: on the stack when they fit, and otherwise in a vector.
const STACK: usize = 64;

/// `x mod d` for a non-zero `d`, in 64 bits when `x` fits.
#[inline(always)]
fn rem(x: u128, d: u64) -> u64 {
    if x >> 64 == 0 {
        x as u64 % d
    } else {
        (x % u128::from(d)) as u64
    }
}

/// `x / d` for a non-zero `d`, in 64 bits when `x` fits.
#[inline(always)]
fn quo(x: u128, d: u64) -> u128 {
    if x >> 64 == 0 {
        u128::from(x as u64 / d)
    } else {
        x / u128::from(d)
    }
}

/// The rational `s / (l·r)` in lowest terms, little-endian, for positive `l` and `r`: `s` reduced
/// by its greatest common divisor with `l`, then what is left of it by that with `r`.
#[inline]
fn quotient(s: i128, l: u64, r: u64) -> Rational {
    let order = ByteOrder::LittleEndian;
    if s == 0 {
        return Rational::from_word(false, 0, 1, order);
    }
    let negative = s < 0;
    let mut a = s.unsigned_abs();
    let (mut l, mut r) = (l, r);
    if l != 1 {
        let g = gcd(rem(a, l), l);
        if g != 1 {
            a = quo(a, g);
            l /= g;
        }
    }
    if r != 1 {
        let g = gcd(rem(a, r), r);
        if g != 1 {
            a = quo(a, g);
            r /= g;
        }
    }
    Rational::from_u128(negative, a, u128::from(l) * u128::from(r), order)
}

/// `c = a·b` for the column-major `m × k` matrix of rationals `a` and `k × n` one `b`, `k` not
/// zero, little-endian, in registers: the rows of `a` and the columns of `b` cleared of their
/// denominators into `i64`s, the sums of their products in 64 bits when the integers' lengths and
/// `k` allow it and in 128 bits when they allow that, and each element of `c` reduced once by its
/// multiples. False, with `c` as it was, when an element is not in words, an integer or a multiple
/// reaches `2^63`, or the sums could pass 128 bits.
pub(super) fn product(
    c: &mut [Rational],
    a: &[Rational],
    b: &[Rational],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let (mut sa, mut sb) = ([0i64; STACK], [0i64; STACK]);
    let (mut sl, mut sr) = ([1u64; STACK], [1u64; STACK]);
    let (mut va, mut vb, mut vl, mut vr);
    let ai: &mut [i64] = if m * k <= STACK {
        &mut sa[..m * k]
    } else {
        va = vec![0; m * k];
        &mut va
    };
    let bi: &mut [i64] = if k * n <= STACK {
        &mut sb[..k * n]
    } else {
        vb = vec![0; k * n];
        &mut vb
    };
    let l: &mut [u64] = if m <= STACK {
        &mut sl[..m]
    } else {
        vl = vec![1; m];
        &mut vl
    };
    let r: &mut [u64] = if n <= STACK {
        &mut sr[..n]
    } else {
        vr = vec![1; n];
        &mut vr
    };
    let (mut xa, mut xb) = (0u64, 0u64);
    for (i, (row, li)) in ai.chunks_exact_mut(k).zip(l.iter_mut()).enumerate() {
        match clear_line(a[i..].iter().step_by(m), row, &mut xa) {
            Some(x) => *li = x,
            None => return false,
        }
    }
    for (j, (col, rj)) in bi.chunks_exact_mut(k).zip(r.iter_mut()).enumerate() {
        match clear_line(b[j * k..(j + 1) * k].iter(), col, &mut xb) {
            Some(x) => *rj = x,
            None => return false,
        }
    }
    let bits = |x: u64| 64 - x.leading_zeros();
    let sum = usize::BITS - (k - 1).leading_zeros();
    let total = bits(xa) + bits(xb) + sum;
    if total > 126 {
        return false;
    }
    for (j, (col, &rj)) in bi.chunks_exact(k).zip(r.iter()).enumerate() {
        let out = &mut c[j * m..(j + 1) * m];
        for ((o, row), &li) in out.iter_mut().zip(ai.chunks_exact(k)).zip(l.iter()) {
            let s = if total <= 62 {
                i128::from(row.iter().zip(col).map(|(x, y)| x * y).sum::<i64>())
            } else {
                row.iter()
                    .zip(col)
                    .map(|(&x, &y)| x as i128 * y as i128)
                    .sum()
            };
            *o = if (li | rj) == 1 {
                Rational::from_u128(s < 0, s.unsigned_abs(), 1, ByteOrder::LittleEndian)
            } else {
                quotient(s, li, rj)
            };
        }
    }
    true
}
