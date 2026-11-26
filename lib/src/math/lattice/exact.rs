//! LLL reduction exactly, by the integral LLL of de Weger and Cohen (Cohen, *A Course in
//! Computational Algebraic Number Theory*, Algorithm 2.6.7): with `dᵢ` the Gram determinant of
//! the first `i` vectors, the Gram–Schmidt coefficients are held as the integers
//! `λᵢⱼ = dⱼ₊₁·μᵢⱼ`, and every division of the algorithm is exact.
//!
//! Linearly dependent rows are handled as Pohst's modified LLL (MLLL) handles them, in the
//! integral form of Cohen's Algorithm 2.7.2: a flag tells for each row whether its Gram–Schmidt
//! vector `b*ᵢ` is nonzero, `dᵢ` is the Gram determinant of the independent rows among the first
//! `i`, and a row whose `b*ₖ` is zero is exchanged downwards (a Euclidean algorithm on its
//! coefficient `μₖ,ₖ₋₁`) until it is the zero vector at the front. The potential, the product of
//! the `dᵢ₊₁` of the independent rows `i`, is a positive integer; it falls below `δ` times itself
//! at an exchange of two independent rows and to a quarter of itself or less at an exchange that
//! shortens a Gram–Schmidt vector by the coefficient `μ` of a dependent row, and an exchange that
//! only moves a zero Gram–Schmidt vector down a place, which leaves it as it is, happens at most
//! `n²` times in all. So the reduction ends, also for `δ = 1`.
//!
//! The same engine inserts deeply (Schnorr and Euchner, 1994) when it is asked to: it is the exact
//! fallback of [`deep_lll`](super::deep_lll). With `Tᵢ = dᵢ·|πᵢ(bₖ)|²`, the Gram determinant of
//! `b₀, …, bᵢ₋₁, bₖ`, an integer, the deep exchange condition `|πᵢ(bₖ)|² < δ·|b*ᵢ|²` is
//! `den·Tᵢ < a·dᵢ₊₁` for `δ = a/den`, and `Tᵢ = (dᵢ·Tᵢ₊₁ + λₖᵢ²)/dᵢ₊₁` exactly, downwards from
//! `Tₖ = dₖ₊₁`.

use super::l2_certify::{Verdict, certify, certify_dd, certify_f64};
use super::zint::{Work, Z, dot as zdot, sub_mul_row as zsub_mul_row};
use super::{LllOptions, Reduced};
use crate::math::float::BigFloat;
use crate::math::matrix::Matrix;
use crate::math::numbers::Number;
use crate::math::rational::Rational;

/// Where a reduction may insert a vector that breaks a condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Insertion {
    /// LLL: one place down at a time, where Lovász's condition fails.
    Lll,
    /// Deep insertions, at the first place `i < k` where `|πᵢ(bₖ)|² < δ·|b*ᵢ|²`: anywhere for
    /// `None`; for `Some(t)` only at the first `t` places of the lattice's rows or the last `t`
    /// before `k`, and always at `k − 1` (LLL's exchange).
    Deep(Option<usize>),
}

impl Insertion {
    /// Whether the vector at `k` may be inserted at `i < k`, the rows of the lattice beginning at
    /// `first` (the zero rows are before it).
    pub(super) fn allowed(self, i: usize, k: usize, first: usize) -> bool {
        match self {
            Insertion::Lll => i + 1 == k,
            Insertion::Deep(None) => true,
            Insertion::Deep(Some(t)) => i + 1 == k || i - first < t || k - i <= t,
        }
    }

    /// The place to go on from after a vector at `k` was inserted at `i`: after it, for LLL and
    /// for insertions anywhere, whose tests at `k` covered every place before `i`; at it again
    /// for a limited depth, where the places allowed from `i` are not all those allowed from `k`.
    pub(super) fn next_after(self, i: usize) -> usize {
        match self {
            Insertion::Lll | Insertion::Deep(None) => i + 1,
            Insertion::Deep(Some(_)) => i.max(1),
        }
    }
}

/// Checks `δ` for a reduction: in `(1/4, 1]`, or in `(1/4, 1)` when `one` is false.
///
/// # Panics
///
/// Panics if it is not.
#[track_caller]
pub(super) fn check_delta(delta: f64, one: bool) {
    if one {
        assert!(
            delta > 0.25 && delta <= 1.0,
            "δ must be in (1/4, 1], not {delta}"
        );
    } else {
        assert!(
            delta > 0.25 && delta < 1.0,
            "δ must be in (1/4, 1), not {delta}"
        );
    }
}

/// Checks `η` for a reduction with `δ`: in `[1/2, √δ)`, or in `(1/2, √δ)` when `half` is false.
///
/// # Panics
///
/// Panics if it is not.
#[track_caller]
pub(super) fn check_eta(eta: f64, delta: f64, half: bool) {
    // `η² < δ` exactly: `η²` in an `f64` is rounded, so the test is on the rationals.
    let below = eta.is_finite() && eta >= 0.0 && {
        let e = Rational::try_from(eta).expect("η is finite");
        let d = Rational::try_from(delta).expect("δ is finite");
        &e * &e < d
    };
    if half {
        assert!(
            eta >= 0.5 && below,
            "η must be in [1/2, √δ), not {eta} for δ = {delta}"
        );
    } else {
        assert!(
            eta > 0.5 && below,
            "η must be in (1/2, √δ), not {eta} for δ = {delta}"
        );
    }
}

/// A finite float as the fraction `a / b` it is exactly, `b > 0`.
pub(super) fn fraction(x: f64) -> (Number, Number) {
    let r = Rational::try_from(x).expect("a finite parameter");
    (r.numer().clone(), r.denom().clone())
}

/// The rows of `m`.
pub(super) fn rows_of(m: &Matrix<Number>) -> Vec<Vec<Number>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)].clone()).collect())
        .collect()
}

/// The `n × n` identity, as rows.
pub(super) fn identity_rows(n: usize) -> Vec<Vec<Number>> {
    (0..n)
        .map(|i| (0..n).map(|j| Number::from(u8::from(i == j))).collect())
        .collect()
}

/// The matrix of the rows `r`, each of `cols` elements, without copying them.
pub(super) fn matrix_of(r: Vec<Vec<Number>>, cols: usize) -> Matrix<Number> {
    let n = r.len();
    Matrix::from_row_major(n, cols, r.into_iter().flatten().collect())
}

/// Whether every element of `v` is zero.
pub(super) fn is_zero_row(v: &[Number]) -> bool {
    v.iter().all(Number::is_zero)
}

/// `⟨a, b⟩`.
pub(super) fn dot(a: &[Number], b: &[Number]) -> Number {
    let mut s = Number::default();
    for (x, y) in a.iter().zip(b) {
        if !x.is_zero() && !y.is_zero() {
            s += x * y;
        }
    }
    s
}

/// `x ← x − q·y`, elementwise.
pub(super) fn sub_mul_row(x: &mut [Number], q: &Number, y: &[Number]) {
    for (x, y) in x.iter_mut().zip(y) {
        if !y.is_zero() {
            *x -= q * y;
        }
    }
}

/// `|2x| > y` for `y > 0`, with a test of the lengths first, which settles most cases without
/// an allocation.
fn twice_exceeds(x: &Number, y: &Number) -> bool {
    let (bx, by) = (x.bits() + 1, y.bits());
    if bx < by {
        return false;
    }
    if bx > by {
        return true;
    }
    (x.abs() << 1usize) > *y
}

/// The integral Gram–Schmidt data of the rows `b[0..lam.len()]` and of the next row: appends
/// `λₖⱼ` (zero where row `j` is dependent), `dₖ₊₁` and the flag of row `k = lam.len()`.
fn gso_row(
    b: &[Vec<Number>],
    d: &mut Vec<Number>,
    lam: &mut Vec<Vec<Number>>,
    free: &mut Vec<bool>,
) {
    gso_row_by(|k, j| dot(&b[k], &b[j]), d, lam, free);
}

/// [`gso_row`] with the inner products `⟨b_k, b_j⟩` from `inner(k, j)`.
fn gso_row_by(
    mut inner: impl FnMut(usize, usize) -> Number,
    d: &mut Vec<Number>,
    lam: &mut Vec<Vec<Number>>,
    free: &mut Vec<bool>,
) {
    let k = lam.len();
    let mut row: Vec<Number> = Vec::with_capacity(k);
    for j in 0..k {
        if !free[j] {
            row.push(Number::default());
            continue;
        }
        let mut u = inner(k, j);
        // A dependent row `i` has `λ = 0` and `dᵢ₊₁ = dᵢ`: its step would leave `u` as it is.
        for i in (0..j).filter(|&i| free[i]) {
            u = (&(&d[i + 1] * &u) - &(&row[i] * &lam[j][i])).div_exact(&d[i]);
        }
        row.push(u);
    }
    let mut u = inner(k, k);
    for i in (0..k).filter(|&i| free[i]) {
        u = (&(&d[i + 1] * &u) - &(&row[i] * &row[i])).div_exact(&d[i]);
    }
    if u.is_zero() {
        free.push(false);
        d.push(d[k].clone());
    } else {
        free.push(true);
        d.push(u);
    }
    lam.push(row);
}

/// The integral reduction: the rows, the transformation applied to them, and the integral
/// Gram–Schmidt data of the first `lam.len()` rows.
pub(super) struct Integral {
    /// The rows, in [`Z`]s updated in place.
    b: Vec<Vec<Z>>,
    /// The transformation: the rows of `U` with `U·B₀ = B` for the rows `B₀` given.
    u: Vec<Vec<Z>>,
    /// The working memory of the updates.
    w: Work,
    /// `d[i]`: the Gram determinant of the independent rows among the first `i`; `d[0] = 1`.
    d: Vec<Number>,
    /// `lam[i][j] = d[j + 1]·μᵢⱼ` for `j < i`, zero where row `j` is dependent.
    lam: Vec<Vec<Number>>,
    /// Whether `b*ᵢ ≠ 0`.
    free: Vec<bool>,
}

impl Integral {
    /// The reduction of the rows `b`, with the transformation so far `u`.
    pub(super) fn new(b: Vec<Vec<Number>>, u: Vec<Vec<Number>>) -> Self {
        let ints = |v: Vec<Vec<Number>>| -> Vec<Vec<Z>> {
            v.iter().map(|r| r.iter().map(Z::of).collect()).collect()
        };
        Integral {
            b: ints(b),
            u: ints(u),
            w: Work::default(),
            d: vec![Number::from(1u8)],
            lam: Vec::new(),
            free: Vec::new(),
        }
    }

    /// The rows and the transformation, as numbers.
    pub(super) fn into_rows(self) -> (Vec<Vec<Number>>, Vec<Vec<Number>>) {
        let numbers = |v: Vec<Vec<Z>>| -> Vec<Vec<Number>> {
            v.iter()
                .map(|r| r.iter().map(Z::number).collect())
                .collect()
        };
        (numbers(self.b), numbers(self.u))
    }

    /// The next row's integral Gram–Schmidt data, its inner products from the rows in place.
    fn next_gso_row(&mut self) {
        let (b, w) = (&self.b, &mut self.w);
        gso_row_by(
            |k, j| zdot(&b[k], &b[j], w).number(),
            &mut self.d,
            &mut self.lam,
            &mut self.free,
        );
    }

    /// The rank: the number of independent rows. Meaningful after [`run`](Self::run).
    pub(super) fn rank(&self) -> usize {
        self.free.iter().filter(|&&f| f).count()
    }

    /// RED(k, l): `bₖ ← bₖ − q·bₗ` with `q` the integer nearest to `μₖₗ`, unless `|μₖₗ| ≤ 1/2`.
    fn reduce(&mut self, k: usize, l: usize) {
        if !self.free[l] {
            return;
        }
        let dl = &self.d[l + 1];
        if !twice_exceeds(&self.lam[k][l], dl) {
            return;
        }
        // q = ⌊(2λ + d) / 2d⌋, the nearest integer to λ / d (no tie: |2λ| > d).
        let q = (&(&self.lam[k][l] << 1usize) + dl).div_floor(&(dl << 1usize));
        let qz = Z::of(&q);
        let (lo, hi) = self.b.split_at_mut(k);
        zsub_mul_row(&mut hi[0], &qz, &lo[l], &mut self.w);
        let (lo, hi) = self.u.split_at_mut(k);
        zsub_mul_row(&mut hi[0], &qz, &lo[l], &mut self.w);
        let (lo, hi) = self.lam.split_at_mut(k);
        hi[0][l] -= &q * dl;
        sub_mul_row(&mut hi[0][..l], &q, &lo[l][..l]);
    }

    /// Whether Lovász's condition fails at `k`, both rows independent:
    /// `den·(dₖ₊₁·dₖ₋₁ + λ²) < a·dₖ²`.
    fn lovasz_fails(&self, k: usize, a: &Number, den: &Number) -> bool {
        let d = &self.d;
        let l = &self.lam[k][k - 1];
        let left = den * &(&(&d[k + 1] * &d[k - 1]) + &(l * l));
        let right = a * &(&d[k] * &d[k]);
        left < right
    }

    /// SWAP(k): exchanges the rows `k − 1` and `k`, row `k − 1` independent, and brings the
    /// integral Gram–Schmidt data of the rows known up to date.
    fn swap(&mut self, k: usize) {
        let known = self.lam.len();
        self.b.swap(k, k - 1);
        self.u.swap(k, k - 1);
        {
            let (head, tail) = self.lam.split_at_mut(k);
            head[k - 1][..k - 1].swap_with_slice(&mut tail[0][..k - 1]);
        }
        let l = self.lam[k][k - 1].clone();
        let d = &mut self.d;
        let lam = &mut self.lam;
        if self.free[k] {
            // Both independent: Cohen's SWAPI. `λₖ,ₖ₋₁` stays as it is.
            let bb = (&(&d[k - 1] * &d[k + 1]) + &(&l * &l)).div_exact(&d[k]);
            for row in lam.iter_mut().take(known).skip(k + 1) {
                let t = core::mem::take(&mut row[k]);
                row[k] = (&(&d[k + 1] * &row[k - 1]) - &(&l * &t)).div_exact(&d[k]);
                row[k - 1] = (&(&bb * &t) + &(&l * &row[k])).div_exact(&d[k + 1]);
            }
            d[k] = bb;
        } else if l.is_zero() {
            // `bₖ` was in the span of the rows before `k − 1`: the zero Gram–Schmidt vector moves
            // down a place, and `b*ₖ₋₁` up, unchanged.
            self.free[k - 1] = false;
            self.free[k] = true;
            d[k] = d[k - 1].clone();
            for row in lam.iter_mut().take(known).skip(k + 1) {
                row.swap(k - 1, k);
            }
        } else {
            // `b*ₖ = 0` and `μ = λ/dₖ ≠ 0`: the new `b*ₖ₋₁` is `μ·b*ₖ₋₁`, the row now at `k` is
            // dependent, `μ` becomes `1/μ` (λ stays), and the determinants from `k` on are
            // multiplied by `μ²`, with the coefficients of the rows above.
            let dk = core::mem::take(&mut d[k]);
            let dk_new = (&l * &l).div_exact(&dk);
            for row in lam.iter_mut().take(known).skip(k + 1) {
                row[k - 1] = (&l * &row[k - 1]).div_exact(&dk);
            }
            for j in k + 1..known {
                if self.free[j] {
                    for row in lam.iter_mut().take(known).skip(j + 1) {
                        row[j] = (&dk_new * &row[j]).div_exact(&dk);
                    }
                }
            }
            for x in d.iter_mut().take(known + 1).skip(k + 1) {
                *x = (&dk_new * &*x).div_exact(&dk);
            }
            d[k] = dk_new;
        }
    }

    /// The first place `i` in `first..k`, allowed by `ins`, where the deep exchange condition
    /// fails for the row `k`: `den·Tᵢ < a·dᵢ₊₁`, all rows from `first` to `k` independent.
    fn insertion_place(
        &self,
        k: usize,
        first: usize,
        a: &Number,
        den: &Number,
        ins: Insertion,
    ) -> Option<usize> {
        let d = &self.d;
        let lk = &self.lam[k];
        let mut t = vec![Number::default(); k + 1 - first];
        t[k - first] = d[k + 1].clone();
        for i in (first..k).rev() {
            t[i - first] = (&(&d[i] * &t[i + 1 - first]) + &(&lk[i] * &lk[i])).div_exact(&d[i + 1]);
        }
        (first..k).find(|&i| ins.allowed(i, k, first) && den * &t[i - first] < a * &d[i + 1])
    }

    /// Reduces the rows: zero rows first, then a basis of the lattice that is size-reduced to
    /// `1/2` and meets Lovász's condition with `δ = a/den`, and with deep insertions the deep
    /// exchange condition at every place `ins` allows.
    pub(super) fn run(&mut self, a: &Number, den: &Number, ins: Insertion) {
        let n = self.b.len();
        if n == 0 {
            return;
        }
        if self.lam.is_empty() {
            self.next_gso_row();
        }
        let mut k = 1;
        while k < n {
            if k == self.lam.len() {
                self.next_gso_row();
            }
            if ins != Insertion::Lll && self.free[k] && self.free[k - 1] {
                for l in (0..k).rev() {
                    self.reduce(k, l);
                }
                // The rows before `k`: the zero rows, then independent rows.
                let first = (0..k).find(|&i| self.free[i]).unwrap_or(k);
                match self.insertion_place(k, first, a, den, ins) {
                    Some(i) => {
                        for j in (i + 1..=k).rev() {
                            self.swap(j);
                        }
                        k = ins.next_after(i);
                    }
                    None => k += 1,
                }
                continue;
            }
            if self.free[k - 1] {
                self.reduce(k, k - 1);
            }
            let swap = self.free[k - 1] && (!self.free[k] || self.lovasz_fails(k, a, den));
            if swap {
                self.swap(k);
                k = (k - 1).max(1);
            } else {
                for l in (0..k - 1).rev() {
                    self.reduce(k, l);
                }
                k += 1;
            }
        }
    }
}

/// Whether the rows `b` are reduced, exactly: zero rows first, the others independent,
/// `|μᵢⱼ| ≤ ea/eb`, and the exchange condition with `δ = a/den` at every place `ins` allows
/// (Lovász's condition alone for [`Insertion::Lll`]).
pub(super) fn reduced_exactly(
    rows: &[Vec<Number>],
    (a, den): (&Number, &Number),
    (ea, eb): (&Number, &Number),
    ins: Insertion,
) -> bool {
    let zeros = rows.iter().take_while(|r| is_zero_row(r)).count();
    let b = &rows[zeros..];
    let (mut d, mut lam, mut free) = (vec![Number::from(1u8)], Vec::new(), Vec::new());
    for _ in 0..b.len() {
        gso_row(b, &mut d, &mut lam, &mut free);
        if !free[free.len() - 1] {
            return false;
        }
    }
    for k in 0..b.len() {
        for j in 0..k {
            if eb * &lam[k][j].abs() > ea * &d[j + 1] {
                return false;
            }
        }
        if k == 0 {
            continue;
        }
        // Tᵢ = dᵢ·|πᵢ(bₖ)|², downwards from Tₖ = dₖ₊₁; the condition at `i` is den·Tᵢ ≥ a·dᵢ₊₁.
        let mut t = d[k + 1].clone();
        for i in (0..k).rev() {
            t = (&(&d[i] * &t) + &(&lam[k][i] * &lam[k][i])).div_exact(&d[i + 1]);
            if ins.allowed(i, k, 0) && den * &t < a * &d[i + 1] {
                return false;
            }
            if ins == Insertion::Lll {
                break;
            }
        }
    }
    true
}

/// LLL-reduces the rows of `basis` exactly, to `|μᵢⱼ| ≤ 1/2` and Lovász's condition with
/// [`delta`](LllOptions::delta) (the dyadic rational the float is); `eta` is checked but the
/// reduction is to `1/2` whatever it is.
///
/// The rows may be linearly dependent: the result has a zero row for each dependency, first, and
/// then a reduced basis of the lattice, with [`rank`](Reduced::rank) its dimension. The
/// reduction is Cohen's integral LLL, with Pohst's handling of dependent rows, on integers alone,
/// so the result is reduced exactly, for any size of the entries; from 4 rows (and for `δ` above
/// `0.2601 = 0.51²` by a margin of `1/64`) it starts from the result of the L² algorithm
/// ([`lll_fp`](super::lll_fp)) for `min(δ, 0.99)` and `η = 0.51`, proved reduced to those, which
/// leaves the integral reduction little more than its size reduction to `1/2` and the few
/// exchanges that reduction asks for. The result is `(δ, 1/2)`-reduced, with `U` unimodular and
/// `U·B = B'`; which of the lattice's reduced bases it is, is not promised beyond that (see [`Reduced`](super::Reduced)): since it starts from L², it is often another
/// reduced basis than 0.2.45's integral reduction from the rows gave.
///
/// The work is polynomial for a fixed `δ < 1`: `O(n²·log B)` exchanges for `n` rows of norm at
/// most `B`, each followed by `O(n·(n + m))` operations on integers of `O(n·log B)` bits for `m`
/// columns, for the integral reduction from the rows given; after L² the integral data of the
/// reduced rows, `O(n²·(n + m))` such operations, and as a rule few exchanges. It also ends for `δ = 1`: the potential is an integer that falls at every exchange
/// but those that only move a zero Gram–Schmidt vector down a place, which leave it as it is and
/// happen at most `n²` times in all; but no polynomial bound is known then.
///
/// # Panics
///
/// Panics if `δ` is not in `(1/4, 1]` or `η` not in `[1/2, √δ)`.
///
/// # Examples
///
/// ```
/// use a::math::lattice::{LllOptions, lll};
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// // A knapsack-like basis: its reduced first vector is far shorter than any row given.
/// let b = Matrix::from_rows(&[[1, 0, 0, 1345], [0, 1, 0, 35], [0, 0, 1, 154]])
///     .map(|&x| Number::from(x));
/// let r = lll(&b, &LllOptions::new());
/// let norm2 = |i: usize| (0..4).map(|j| r.basis()[(i, j)].to_f64().powi(2)).sum::<f64>();
/// assert!(norm2(0) < 35.0 * 35.0);
/// assert_eq!(&r.transform().clone() * &b, *r.basis());
///
/// // Dependent rows: a zero row first for the one dependency, then a basis of the lattice 2ℤ × ℤ.
/// let d = Matrix::from_rows(&[[4, 2], [6, 3], [2, 0]]).map(|&x| Number::from(x));
/// let r = lll(&d, &LllOptions::new());
/// assert_eq!(r.rank(), 2);
/// assert!(r.basis()[(0, 0)].is_zero() && r.basis()[(0, 1)].is_zero());
/// let lattice = Matrix::from_fn(2, 2, |i, j| r.basis()[(i + 1, j)].clone());
/// assert_eq!(lattice.det().abs(), Number::from(2));
/// ```
#[track_caller]
pub fn lll(basis: &Matrix<Number>, opts: &LllOptions) -> Reduced {
    check_delta(opts.delta, true);
    check_eta(opts.eta, opts.delta, true);
    let n = basis.rows();
    if n >= EXACT_AFTER_L2 && opts.delta > 0.2601 + 1.0 / 64.0 {
        // L² first, to (min(δ, 0.99), 0.51), proved; then the integral reduction from its result.
        let first = super::l2::reduce_fp(basis, opts.delta.min(0.99), 0.51, Insertion::Lll);
        let (a, den) = fraction(opts.delta);
        let mut e = Integral::new(rows_of(first.basis()), rows_of(first.transform()));
        e.run(&a, &den, Insertion::Lll);
        let rank = e.rank();
        let (b, u) = e.into_rows();
        return Reduced {
            basis: matrix_of(b, basis.cols()),
            transform: matrix_of(u, n),
            rank,
        };
    }
    reduce_exactly(basis, opts.delta, Insertion::Lll)
}

/// The rows from which [`lll`] reduces by L² before the integral reduction. Measured on an Apple
/// M4 Max, 2 to 16 rows of knapsack, uniform, q-ary, NTRU-like and small bases, L² first against
/// the integral reduction alone: L² first takes 0.54 to 0.78 of the time of knapsacks from 3 rows
/// (0.66 at 2 for 200 bits), but 1.17 to 1.51 of that of uniform bases of 10 and 100 bits and of
/// small ones at 3 rows and 1.22 to 1.30 at 4 and 5 for 100 bits and small ones; neither 3 nor 5
/// is faster everywhere, and from 6 rows L² first wins for all (0.33 to 0.95).
pub(super) const EXACT_AFTER_L2: usize = 4;

/// Reduces the rows of `basis` by the integral engine alone, with `δ` and insertions as `ins`
/// says, the size reduction to `1/2`.
pub(super) fn reduce_exactly(basis: &Matrix<Number>, delta: f64, ins: Insertion) -> Reduced {
    let (a, den) = fraction(delta);
    let n = basis.rows();
    let mut e = Integral::new(rows_of(basis), identity_rows(n));
    e.run(&a, &den, ins);
    let rank = e.rank();
    let (b, u) = e.into_rows();
    Reduced {
        basis: matrix_of(b, basis.cols()),
        transform: matrix_of(u, n),
        rank,
    }
}

/// Whether the rows of `basis` are `(δ, η)`-LLL-reduced, exactly: zero rows, if any, first; the
/// others linearly independent, with `|μᵢⱼ| ≤ η` for `j < i` and
/// `|b*ₖ|² ≥ (δ − μ²ₖ,ₖ₋₁)·|b*ₖ₋₁|²`, in the exact rationals the floats are.
///
/// The test is exact. The Cholesky factorisation of the exact Gram matrix in interval arithmetic
/// decides almost every basis, a condition whose interval lies on one side of its bound holding
/// or failing for certain: first in doubles stepped outwards after each operation (while the Gram
/// matrix's entries are below `2^500`), then in balls of double-doubles (entries below `2^400`),
/// then, for entries of more than 5 bits, in `BigFloat` bounds rounded outwards at `64 + n` bits
/// and twice that.
/// A condition the intervals leave open goes to the test on integers alone, by the integral
/// Gram–Schmidt orthogonalisation: `O(n²·(n + m))` operations on integers of `O(n·log B)` bits,
/// for `n` rows of `m` columns and norm at most `B`.
///
/// # Panics
///
/// Panics if `δ` or `η` is not finite.
///
/// # Examples
///
/// ```
/// use a::math::lattice::is_lll_reduced;
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// let good = Matrix::from_rows(&[[1, 0], [0, 1]]).map(|&x| Number::from(x));
/// let bad = Matrix::from_rows(&[[1, 0], [5, 1]]).map(|&x| Number::from(x));
/// assert!(is_lll_reduced(&good, 0.99, 0.51));
/// assert!(!is_lll_reduced(&bad, 0.99, 0.51));
/// ```
#[track_caller]
pub fn is_lll_reduced(basis: &Matrix<Number>, delta: f64, eta: f64) -> bool {
    assert!(delta.is_finite(), "δ is finite");
    assert!(eta.is_finite(), "η is finite");
    let rows = rows_of(basis);
    let zeros = rows.iter().take_while(|r| is_zero_row(r)).count();
    let b = &rows[zeros..];
    // Intervals first, of the exact Gram matrix's Cholesky factorisation: they decide almost
    // every basis at a few bits a row; a condition they leave open goes to the exact test.
    let zb: Vec<Vec<Z>> = b.iter().map(|r| r.iter().map(Z::of).collect()).collect();
    let mut w = Work::default();
    let g: Vec<Vec<Number>> = (0..zb.len())
        .map(|i| {
            (0..=i)
                .map(|j| zdot(&zb[i], &zb[j], &mut w).number())
                .collect()
        })
        .collect();
    match certify_f64(&g, delta, eta, Insertion::Lll) {
        Verdict::Reduced => return true,
        Verdict::NotReduced => return false,
        Verdict::Unsure => {}
    }
    // Then balls of double-doubles, about twice the bits at a few times the cost of the doubles.
    match certify_dd(&g, delta, eta, Insertion::Lll) {
        Verdict::Reduced => return true,
        Verdict::NotReduced => return false,
        Verdict::Unsure => {}
    }
    // A basis they leave open (the intervals widen with the depth of the factorisation,
    // about a bit a row in doubles): of entries of a few bits, the exact test's integers stay
    // short and it is the cheaper (a reduced knapsack basis of 100 rows of 3-bit entries: 21 ms
    // against 32 ms for the intervals); otherwise the intervals of BigFloats first, at `64 + n`
    // bits, then twice that. Measured on an Apple M4 Max on bases reduced by `lll_fp` of 120 to
    // 300 rows (knapsacks, q-ary, NTRU-like and Goldstein–Mayer) that the doubles and double-doubles
    // leave open: for entries of 6 to 8 bits the intervals first took 0.51 to 0.74 of the time of
    // the exact test; against `64 + 2n` bits, `64 + n` took 0.75 to 0.87 of the time, `32 + n`
    // 0.68 to 0.80, `128 + n` 0.79 to 0.96, `64 + n/2` 0.38 to 1.58 (too short for a knapsack of
    // 200 rows, which then took a doubling) and 64 bits alone up to 3.0 times as long; one
    // doubling more or less changed nothing, as the first precision decided every basis.
    let short = b.iter().flatten().all(|x| x.bits() <= 5);
    let mut p = 64 + b.len() as u64;
    for _ in 0..if short { 0 } else { 2 } {
        if p > BigFloat::MAX_PREC {
            break;
        }
        match certify(&g, delta, eta, Insertion::Lll, p) {
            Verdict::Reduced => return true,
            Verdict::NotReduced => return false,
            Verdict::Unsure => p *= 2,
        }
    }
    let (a, den) = fraction(delta);
    let (ea, eb) = fraction(eta);
    reduced_exactly(&rows, (&a, &den), (&ea, &eb), Insertion::Lll)
}
