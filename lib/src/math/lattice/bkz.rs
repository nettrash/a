//! Block Korkine–Zolotarev reduction (Schnorr and Euchner, *Lattice basis reduction: improved
//! practical algorithms and solving subset sum problems*, Math. Programming 66, 1994), by
//! enumeration in blocks, as fplll's BKZ 1.0 without pruning.
//!
//! A tour visits the blocks `[k, min(k + β, n))` for `k = 0, …, n − 2`: the enumeration finds a
//! shortest vector of the block's lattice projected orthogonally to `b₀, …, b_{k−1}`, and if it
//! is short enough it becomes the block's first vector, the block completed to a basis by
//! unimodular steps on the vector's coordinates, and the block is LLL-reduced again. Short
//! enough is first, as Schnorr and Euchner and fplll have it, a squared norm below
//! `δ·|b*ₖ|²`; once a tour inserts nothing, any vector shorter than `b*ₖ`, until a tour inserts
//! nothing again. The first kind saves tours that the second alone would take from the start
//! (28 and 36 tours in all against 42 and 38 on two knapsack lattices of rank 60 with
//! `β = 20`); the second makes the end exact.
//!
//! The tours run in two stages. The first, in [`bkz_fast`](super::bkz_fast), is fplll's design:
//! the Gram–Schmidt orthogonalisation kept in floating point across insertions and tours, a
//! float enumeration per block, and after an insertion an uncertified local L² on the block's
//! rows alone; it does nearly all the work. Then, once, the certified LLL reduction of the whole
//! basis, and tours on the exact Gram–Schmidt data with the exact enumeration, inserting any
//! vector strictly shorter than `b*ₖ`, until a tour inserts nothing; usually the first does.
//! Last, a size reduction to `|μᵢⱼ| ≤ 1/2`, exact, which changes no Gram–Schmidt vector.
//!
//! This module also holds the area's one call of an LLL reduction, [`reduce`].

use super::bkz_fast::{self, Outcome};
use super::enum_gso::Gso;
use super::enum_search::{Levels, least};
use super::{LllOptions, Reduced};
use crate::math::matrix::Matrix;
use crate::math::numbers::Number;
use crate::math::rational::Rational;

/// The parameters of [`bkz`].
///
/// # Examples
///
/// ```
/// use a::math::lattice::BkzOptions;
///
/// let o = BkzOptions::new(20).with_max_tours(Some(8));
/// assert_eq!((o.block_size, o.delta, o.eta, o.max_tours), (20, 0.99, 0.51, Some(8)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BkzOptions {
    /// The block size `β ≥ 2`; `β = 2` is LLL, `β = n` HKZ reduction.
    pub block_size: usize,
    /// `δ` of the LLL reductions inside; `0.99` by default.
    pub delta: f64,
    /// `η` of the LLL reductions inside; `0.51` by default.
    pub eta: f64,
    /// The most tours through the blocks, `None` for as many as it takes until a tour changes
    /// nothing; `None` by default.
    pub max_tours: Option<usize>,
}

impl BkzOptions {
    /// Blocks of `block_size`, `δ = 0.99`, `η = 0.51`, tours until nothing changes.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        BkzOptions {
            block_size,
            delta: 0.99,
            eta: 0.51,
            max_tours: None,
        }
    }

    /// Sets [`delta`](Self::delta).
    #[must_use]
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Sets [`eta`](Self::eta).
    #[must_use]
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Sets [`max_tours`](Self::max_tours).
    #[must_use]
    pub fn with_max_tours(mut self, max_tours: Option<usize>) -> Self {
        self.max_tours = max_tours;
        self
    }
}

/// LLL-reduces the rows of `b`, which may be dependent (the zero rows first, one for each
/// dependency, then the reduced basis): the one place the enumeration and BKZ call an LLL.
///
/// It is the L² algorithm, [`lll_fp`](super::lll_fp), whose result is proved `(δ, η)`-reduced,
/// whenever its parameters allow: `η > 1/2` and `δ < 1`. For `η = 1/2` exactly, which L² does
/// not take (its lazy size reduction needs room above 1/2), and for `δ = 1`, it is the exact
/// [`lll`](super::lll), which reduces to `|μᵢⱼ| ≤ 1/2` and allows `δ = 1`.
pub(super) fn reduce(b: &Matrix<Number>, delta: f64, eta: f64) -> Reduced {
    let opts = LllOptions::new().with_delta(delta).with_eta(eta);
    if eta > 0.5 && delta < 1.0 {
        super::lll_fp(b, &opts)
    } else {
        super::lll(b, &opts)
    }
}

/// Whether the rows of `b` are independent modulo the prime `2⁶¹ − 1`, which proves them
/// independent over the integers (a rank modulo a prime is at most the rank); `false` says
/// nothing. A word-size elimination, far cheaper than the exact rank of large numbers.
fn full_rank_modulo(b: &Matrix<Number>) -> bool {
    const P: u64 = (1 << 61) - 1;
    let (n, m) = b.shape();
    if n > m {
        return false;
    }
    let mulmod = |x: u64, y: u64| (u128::from(x) * u128::from(y) % u128::from(P)) as u64;
    let inverse = |x: u64| {
        // x^(P−2) by squaring.
        let (mut r, mut base, mut e) = (1u64, x, P - 2);
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, base);
            }
            base = mulmod(base, base);
            e >>= 1;
        }
        r
    };
    let mut a: Vec<Vec<u64>> = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| u64::try_from(&b[(i, j)].rem_euclid(&Number::from(P))).expect("below P"))
                .collect()
        })
        .collect();
    let mut row = 0;
    for col in 0..m {
        if row == n {
            break;
        }
        let Some(p) = (row..n).find(|&i| a[i][col] != 0) else {
            continue;
        };
        a.swap(row, p);
        let inv = inverse(a[row][col]);
        let (top, below) = a.split_at_mut(row + 1);
        let pivot = &top[row];
        for r in below.iter_mut() {
            let f = mulmod(r[col], inv);
            if f == 0 {
                continue;
            }
            for (x, &y) in r[col..].iter_mut().zip(&pivot[col..]) {
                *x = (*x + P - mulmod(f, y)) % P;
            }
        }
        row += 1;
    }
    row == n
}

/// The rows of `m`.
fn rows_of(m: &Matrix<Number>, from: usize) -> Vec<Vec<Number>> {
    (from..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)].clone()).collect())
        .collect()
}

/// `(rᵢ, rⱼ) ← (p·rᵢ + q·rⱼ, s·rᵢ + t·rⱼ)`.
fn combine2(v: &mut [Vec<Number>], i: usize, j: usize, c: [&Number; 4]) {
    let [p, q, s, t] = c;
    let a = core::mem::take(&mut v[i]);
    let b = core::mem::take(&mut v[j]);
    v[i] = a.iter().zip(&b).map(|(x, y)| &(p * x) + &(q * y)).collect();
    v[j] = a.iter().zip(&b).map(|(x, y)| &(s * x) + &(t * y)).collect();
}

/// Makes `Σ xᵢ·b_{k+i}` (divided by the gcd of `x`, which is 1 for a shortest vector) the row
/// `k`, by unimodular steps on the rows `k, k + 1, …`: from the last coefficient down, each pair
/// of nonzero coefficients `(y, z)` with `g = a·y + b·z` their gcd becomes `(g, 0)` by
/// `(r, s) ← ((y/g)·r + (z/g)·s, −b·r + a·s)`, of determinant 1. The same steps apply to the
/// rows of the transformation.
fn insert(rows: &mut [Vec<Number>], trans: &mut [Vec<Number>], k: usize, x: &[i64]) {
    let mut y: Vec<Number> = x.iter().map(|&v| Number::from(v)).collect();
    for j in (1..y.len()).rev() {
        if y[j].is_zero() {
            continue;
        }
        if y[j - 1].is_zero() {
            y.swap(j - 1, j);
            rows.swap(k + j - 1, k + j);
            trans.swap(k + j - 1, k + j);
            continue;
        }
        let (g, a, b) = y[j - 1].gcd_ext(&y[j]);
        let p = &y[j - 1] / &g;
        let q = &y[j] / &g;
        let nb = -&b;
        combine2(rows, k + j - 1, k + j, [&p, &q, &nb, &a]);
        combine2(trans, k + j - 1, k + j, [&p, &q, &nb, &a]);
        y[j - 1] = g;
        y[j] = Number::default();
    }
}

/// The state of a BKZ reduction: the independent rows, the rows of the transformation that give
/// them, and their Gram–Schmidt data as far as they are known.
struct State {
    rows: Vec<Vec<Number>>,
    trans: Vec<Vec<Number>>,
    gso: Gso,
    delta: f64,
    eta: f64,
}

impl State {
    /// LLL-reduces the rows `0..end`, the transformation with them; the Gram–Schmidt data are
    /// kept up to the first row that changed.
    fn reduce_prefix(&mut self, end: usize) {
        let cols = self.rows[0].len();
        let sub = Matrix::from_fn(end, cols, |i, j| self.rows[i][j].clone());
        let r = reduce(&sub, self.delta, self.eta);
        let (b, u) = (r.basis(), r.transform());
        // A row of an independent basis is unchanged exactly when its row of U is a unit row.
        let first = (0..end)
            .find(|&i| (0..cols).any(|j| b[(i, j)] != self.rows[i][j]))
            .unwrap_or(end);
        let width = self.trans[0].len();
        let mut fresh: Vec<Vec<Number>> = Vec::with_capacity(end - first);
        for i in first..end {
            let mut t = vec![Number::default(); width];
            for l in 0..end {
                let c = &u[(i, l)];
                if c.is_zero() {
                    continue;
                }
                for (s, e) in t.iter_mut().zip(&self.trans[l]) {
                    if !e.is_zero() {
                        *s += &(c * e);
                    }
                }
            }
            fresh.push(t);
        }
        for (i, t) in (first..end).zip(fresh) {
            self.trans[i] = t;
            self.rows[i] = (0..cols).map(|j| b[(i, j)].clone()).collect();
        }
        self.gso.truncate(first);
    }

    /// Orthogonalises the rows up to `end`.
    fn orthogonalise(&mut self, end: usize) {
        let ok = self.gso.extend(&self.rows, end);
        debug_assert!(ok, "the rows of a BKZ basis are independent");
    }

    /// Size-reduces every row to `|μᵢⱼ| ≤ 1/2`, exactly, on the integral Gram–Schmidt data
    /// (Cohen's RED): `bₖ ← bₖ − q·bₗ` with `q` the integer nearest to `λₖₗ/d_{l+1}`, from
    /// `l = k − 1` down. No Gram–Schmidt vector changes, nor any projected block.
    fn size_reduce_exactly(&mut self) {
        let n = self.rows.len();
        self.orthogonalise(n);
        for k in 1..n {
            for l in (0..k).rev() {
                let d = &self.gso.d[l + 1];
                let q = (&(&self.gso.lam[k][l] * 2u8) + d).div_floor(&(d * 2u8));
                if q.is_zero() {
                    continue;
                }
                for v in [&mut self.rows, &mut self.trans] {
                    let (lo, hi) = v.split_at_mut(k);
                    for (x, y) in hi[0].iter_mut().zip(&lo[l]) {
                        if !y.is_zero() {
                            *x -= &(&q * y);
                        }
                    }
                }
                let (lo, hi) = self.gso.lam.split_at_mut(k);
                let row = &mut hi[0];
                row[l] -= &(&q * d);
                for (x, y) in row[..l].iter_mut().zip(&lo[l][..l]) {
                    *x -= &(&q * y);
                }
            }
        }
    }
}

/// BKZ-reduces the rows of `basis` with blocks of [`block_size`](BkzOptions::block_size),
/// exactly: Schnorr and Euchner's algorithm, the shortest vector of each block found by
/// enumeration.
///
/// The result is `(δ, η)`-LLL-reduced (as [`is_lll_reduced`](super::is_lll_reduced) checks
/// it), with `U·B = B'` for the unimodular [`transform`](Reduced::transform), and for dependent
/// rows the zero rows first. Tours insert first only vectors with `|πₖ(v)|² < δ·|b*ₖ|²`, as
/// Schnorr and Euchner and fplll do, and then any vector shorter than `b*ₖ`; the last tours
/// compare in exact arithmetic, on the exact Gram–Schmidt data, and run until one inserts
/// nothing. So when the reduction ends that way — always, with `max_tours` `None` — every
/// block's first Gram–Schmidt vector `b*ₖ` is a shortest nonzero vector of the lattice of
/// `b_k, …, b_{min(k+β,n)−1}` projected orthogonally to `b₀, …, b_{k−1}`, exactly, which is
/// more than fplll's BKZ promises (`δ·|b*ₖ|²` at most the block's minimum); the basis is then
/// also size-reduced to `|μᵢⱼ| ≤ 1/2`, whatever `η`, and satisfies Lovász's condition with
/// `δ = 1` (each `b*ₖ₋₁` is shortest in its block, which holds `πₖ₋₁(bₖ)`). With `β ≥ n` that is
/// HKZ reduction and `b₀` is a shortest vector of the lattice. When it stops at
/// [`max_tours`](BkzOptions::max_tours) (counting every tour) the basis is still
/// `(δ, η)`-LLL-reduced, but the blocks are promised nothing.
///
/// Most tours run in floating point, in fplll's way: the Gram–Schmidt orthogonalisation kept
/// from one insertion to the next, and after an insertion a local L² on the block's rows only.
/// Then the basis is LLL-reduced once by the certified [`lll_fp`](super::lll_fp) (the exact
/// [`lll`](super::lll) for `η = 1/2`) and the exact tours finish the work. The reduction
/// always ends: the floating-point tours are at most `4n + 64` for each of at most three starts
/// (a start that fails is followed by the certified LLL and a fresh start), `3·(4n + 64)` in
/// all, and in the exact ones each
/// insertion and each exchange of LLL makes the Gram determinants `(d₁, d₂, …)`, positive
/// integers, smaller in lexicographic order. The work of a block is that of an enumeration in
/// dimension `β`, `2^O(β²)`: block sizes up to about 30 are practical.
///
/// # Panics
///
/// Panics if the block size is below 2, `δ` not in `(1/4, 1)` or `η` not in `[1/2, √δ)`.
///
/// # Examples
///
/// ```
/// use a::math::lattice::{BkzOptions, bkz, is_lll_reduced, shortest_vector};
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// let b = Matrix::from_rows(&[
///     [1, 0, 0, 0, 0, 31_415],
///     [0, 1, 0, 0, 0, 27_182],
///     [0, 0, 1, 0, 0, 14_142],
///     [0, 0, 0, 1, 0, 17_320],
///     [0, 0, 0, 0, 1, 22_360],
/// ])
/// .map(|&x| Number::from(x));
/// let r = bkz(&b, &BkzOptions::new(5));
/// assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
/// assert_eq!(&r.transform().clone() * &b, *r.basis());
/// // With the block as large as the lattice: HKZ, size-reduced to 1/2, the first vector a
/// // shortest one.
/// assert!(is_lll_reduced(r.basis(), 1.0, 0.5));
/// let first: Number = (0..6).map(|j| &r.basis()[(0, j)] * &r.basis()[(0, j)]).sum();
/// assert_eq!(first, shortest_vector(&b).unwrap().norm2);
/// ```
#[track_caller]
pub fn bkz(basis: &Matrix<Number>, opts: &BkzOptions) -> Reduced {
    let BkzOptions {
        block_size,
        delta,
        eta,
        max_tours,
    } = *opts;
    assert!(
        block_size >= 2,
        "the block size must be at least 2, not {block_size}"
    );
    assert!(
        delta > 0.25 && delta < 1.0,
        "δ must be in (1/4, 1), not {delta}"
    );
    assert!(
        eta >= 0.5 && eta * eta < delta,
        "η must be in [1/2, √δ), not {eta} for δ = {delta}"
    );
    // Rows proved independent go to the floating-point tours as they are, which reduce them
    // first; others are LLL-reduced by the certified reduction, their zero rows set apart.
    let independent = basis.rows() >= 2 && full_rank_modulo(basis);
    let (head_rows, head_trans, rows, trans) = if independent {
        let n = basis.rows();
        let identity = (0..n)
            .map(|i| (0..n).map(|j| Number::from(u8::from(i == j))).collect())
            .collect();
        (Vec::new(), Vec::new(), rows_of(basis, 0), identity)
    } else {
        let r0 = reduce(basis, delta, eta);
        if r0.rank() < 2 {
            return r0;
        }
        let zeros = basis.rows() - r0.rank();
        let (mut b, mut u) = (rows_of(r0.basis(), 0), rows_of(r0.transform(), 0));
        let (rows, trans) = (b.split_off(zeros), u.split_off(zeros));
        (b, u, rows, trans)
    };
    let n = rows.len();
    let beta = block_size.min(n);
    // The tours in floating point, then the certified reduction of the whole basis, once.
    let mut tours = 0;
    let mut st = State {
        rows,
        trans,
        gso: Gso::new(),
        delta,
        eta,
    };
    let mut outcome = Outcome::Unsettled;
    for _ in 0..3 {
        let rows = core::mem::take(&mut st.rows);
        let trans = core::mem::take(&mut st.trans);
        let (rows, trans, out) =
            bkz_fast::run(rows, trans, (delta, eta), beta, max_tours, &mut tours);
        (st.rows, st.trans, outcome) = (rows, trans, out);
        // The certified reduction: of the result, or of a basis the floats could not carry,
        // which the precision of L² can, before the tours in floats go on from it.
        st.gso.truncate(0);
        st.reduce_prefix(n);
        if outcome != Outcome::Unsettled {
            break;
        }
    }
    // The exact tours. δ as the fraction a/c it is exactly: a tour of the first kind inserts a
    // vector of the block only if d_k·|π_k(v)|² < δ·d_{k+1}, one of the second kind if it is
    // below d_{k+1} at all. They begin with the second kind when the float tours settled.
    let dl = Rational::try_from(delta).expect("δ is finite");
    let (da, dc) = (dl.numer().clone(), dl.denom().clone());
    let mut strict = outcome == Outcome::Converged;
    let mut clean = false;
    while outcome != Outcome::Stopped && max_tours.is_none_or(|t| tours < t) {
        tours += 1;
        let mut inserted = false;
        // The rows 0..reduced are LLL-reduced; after an insertion, those up to its block's end.
        let mut reduced = n;
        for k in 0..n - 1 {
            let end = (k + beta).min(n);
            if reduced < end {
                st.reduce_prefix(end);
                reduced = end;
            }
            st.orthogonalise(end);
            // d_k·|π_k(v)|² < d_k·|b*ₖ|² = d_{k+1}, or below δ times that: the largest integer
            // below the bound.
            let bound = if strict {
                &st.gso.d[k + 1] - 1u8
            } else {
                (&(&da * &st.gso.d[k + 1]) - 1u8).div_floor(&dc)
            };
            let levels = Levels {
                gso: &st.gso,
                lo: k,
                hi: end,
                target: None,
            };
            if let Some((x, _)) = least(&levels, &bound) {
                insert(&mut st.rows, &mut st.trans, k, &x);
                st.gso.truncate(k);
                st.reduce_prefix(end);
                reduced = end;
                inserted = true;
            }
        }
        if !inserted {
            if strict {
                clean = true;
                break;
            }
            strict = true;
        }
    }
    // Every block reduced: the size reduction to 1/2 keeps the Gram–Schmidt vectors, so the
    // blocks, and Lovász's condition with δ = 1.
    if clean {
        st.size_reduce_exactly();
    }
    let (mut rows, mut trans) = (head_rows, head_trans);
    rows.extend(st.rows);
    trans.extend(st.trans);
    let cols = basis.cols();
    let m = basis.rows();
    Reduced {
        basis: Matrix::from_fn(m, cols, |i, j| rows[i][j].clone()),
        transform: Matrix::from_fn(m, m, |i, j| trans[i][j].clone()),
        rank: n,
    }
}
