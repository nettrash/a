//! The L² algorithm of Nguyen and Stehlé (*An LLL algorithm with quadratic complexity*, SIAM J.
//! Comput. 39, 2009, Figures 4 and 5) over a floating-point type [`Fp`]: the basis and its Gram
//! matrix are kept exactly, in integers, and the Gram–Schmidt coefficients are computed in
//! floating point from the exact Gram matrix, a row at a time, by the Cholesky factorisation
//! algorithm (CFA); a row is size-reduced lazily, by as many passes as it takes, each pass
//! recomputing its coefficients from the updated exact Gram matrix; and a vector is inserted at
//! the place where Lovász's condition stops failing (or, with deep insertions, at the first place
//! where the deep exchange condition fails).
//!
//! A row that becomes zero (its exact squared norm is zero) is moved to the end and left out:
//! that is how the rows of a dependent basis come apart, as in Pohst's MLLL.
//!
//! A run fails, rather than loops, when the precision is too low for the basis: when a
//! coefficient is not finite, a squared norm is not positive, the size reduction of a row stops
//! shortening it, an exchange undoes the one before it, or the number of iterations passes the
//! bound that the potential gives.

use super::exact::{Insertion, dot, sub_mul_row};
use super::l2_float::Fp;
use crate::math::float::Round;
use crate::math::numbers::Number;
use crate::math::rational::Rational;

/// A run of L² that the precision did not carry through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Failure;

/// The exact part of a reduction: the rows, the transformation applied to them, the Gram matrix
/// of the rows, and how many rows are the lattice's: those after `active` are zero.
pub(super) struct Basis {
    /// The rows.
    pub(super) b: Vec<Vec<Number>>,
    /// The rows of `U`, `U·B₀ = B` for the rows `B₀` given.
    pub(super) u: Vec<Vec<Number>>,
    /// The lower triangle of the Gram matrix: `g[i][j] = ⟨bᵢ, bⱼ⟩` for `j ≤ i`.
    g: Vec<Vec<Number>>,
    /// The rows before it are the lattice's; those from it on are zero rows found.
    pub(super) active: usize,
}

impl Basis {
    /// The reduction of the rows `b`, with the transformation so far `u`.
    pub(super) fn new(b: Vec<Vec<Number>>, u: Vec<Vec<Number>>) -> Basis {
        let g = (0..b.len())
            .map(|i| (0..=i).map(|j| dot(&b[i], &b[j])).collect())
            .collect();
        let active = b.len();
        Basis { b, u, g, active }
    }

    /// Whether the Gram matrix kept is that of the rows: the invariant of every step.
    fn gram_is_exact(&self) -> bool {
        (0..self.b.len()).all(|i| (0..=i).all(|j| self.g[i][j] == dot(&self.b[i], &self.b[j])))
    }

    /// `⟨bᵢ, bⱼ⟩`.
    fn gram(&self, i: usize, j: usize) -> &Number {
        if j <= i { &self.g[i][j] } else { &self.g[j][i] }
    }

    /// `bₖ ← bₖ − Σ xᵢ·bᵢ` for the pairs `(i, xᵢ)`, `i < k`, and the Gram matrix with it.
    fn apply(&mut self, k: usize, xs: &[(usize, Number)]) {
        for (i, x) in xs {
            let i = *i;
            let (lo, hi) = self.b.split_at_mut(k);
            sub_mul_row(&mut hi[0], x, &lo[i]);
            let (lo, hi) = self.u.split_at_mut(k);
            sub_mul_row(&mut hi[0], x, &lo[i]);
            // ⟨bₖ − x·bᵢ, bₖ − x·bᵢ⟩ = Gₖₖ − x·(2·Gₖᵢ − x·Gᵢᵢ), with the old Gₖᵢ.
            let t = &(&self.g[k][i] << 1usize) - &(x * &self.g[i][i]);
            self.g[k][k] -= x * &t;
            for j in 0..self.active {
                if j == k {
                    continue;
                }
                let t = x * self.gram(i, j);
                if t.is_zero() {
                    continue;
                }
                if j < k {
                    self.g[k][j] -= t;
                } else {
                    self.g[j][k] -= t;
                }
            }
        }
    }

    /// Moves the row `from` to the place `to`, the rows between shifting by one, with the Gram
    /// matrix: each entry is moved, none copied.
    fn move_row(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        let r = self.b.remove(from);
        self.b.insert(to, r);
        let r = self.u.remove(from);
        self.u.insert(to, r);
        let (lo, hi) = (from.min(to), from.max(to));
        // The old index of the row now at `p`.
        let old = |p: usize| -> usize {
            if p < lo || p > hi {
                p
            } else if p == to {
                from
            } else if from > to {
                p - 1
            } else {
                p + 1
            }
        };
        // The pairs with the larger index in `lo..=hi`, old and new, are the same: the entries
        // of those rows are taken out once each and put where they now belong.
        let mut window: Vec<Vec<Number>> =
            (lo..=hi).map(|i| core::mem::take(&mut self.g[i])).collect();
        #[allow(
            clippy::needless_range_loop,
            reason = "`p` also picks the old row and the length of the new one"
        )]
        for p in lo..=hi {
            let a = old(p);
            let row = (0..=p)
                .map(|q| {
                    let c = old(q);
                    let (x, y) = if c <= a { (a, c) } else { (c, a) };
                    core::mem::take(&mut window[x - lo][y])
                })
                .collect();
            self.g[p] = row;
        }
        for row in self.g.iter_mut().skip(hi + 1) {
            if from > to {
                row[lo..=hi].rotate_right(1);
            } else {
                row[lo..=hi].rotate_left(1);
            }
        }
    }

    /// Moves the zero row `k` after the rows of the lattice.
    fn drop_zero(&mut self, k: usize) {
        self.move_row(k, self.active - 1);
        self.active -= 1;
    }
}

/// The floating-point Gram–Schmidt data of L²: `r[i][j] ≈ ⟨bᵢ, b*ⱼ⟩` for `j ≤ i`,
/// `mu[i][j] ≈ μᵢⱼ` for `j < i`, valid for the rows before the one being reduced, and
/// `s[j] ≈ |πⱼ(bₖ)|²` for that row `k`.
struct Gso<F> {
    r: Vec<Vec<F>>,
    mu: Vec<Vec<F>>,
    s: Vec<F>,
    prec: u64,
}

impl<F: Fp> Gso<F> {
    fn new(n: usize, prec: u64) -> Self {
        let zero = F::of_f64(0.0, prec);
        Gso {
            r: vec![vec![zero.clone(); n]; n],
            mu: vec![vec![zero.clone(); n]; n],
            s: vec![zero; n + 1],
            prec,
        }
    }

    /// Step 2 of the CFA for the row `k`: its coefficients from the exact Gram matrix and those
    /// of the rows before it, and the squared norms `s[j]` of its projections.
    fn row(&mut self, st: &Basis, k: usize) -> Result<(), Failure> {
        let p = self.prec;
        for j in 0..k {
            let mut t = F::of_number(st.gram(k, j), p);
            for i in 0..j {
                t = t.sub_mul(&self.mu[j][i], &self.r[k][i]);
            }
            let m = t.div(&self.r[j][j]);
            if !m.is_finite() {
                return Err(Failure);
            }
            self.r[k][j] = t;
            self.mu[k][j] = m;
        }
        self.s[0] = F::of_number(st.gram(k, k), p);
        for j in 0..k {
            self.s[j + 1] = self.s[j].sub_mul(&self.mu[k][j], &self.r[k][j]);
        }
        Ok(())
    }

    /// The size reduction of Figure 5: until every `|μ̄ₖⱼ| ≤ η̄`, rounds the coefficients from the
    /// top down, each correcting those below it, and subtracts the rows exactly. Tells whether it
    /// changed the row.
    fn size_reduce(&mut self, st: &mut Basis, k: usize, eta: &F) -> Result<bool, Failure> {
        let p = self.prec;
        // Each pass must shorten the row (an integer); a pass that does not is allowed twice in a
        // row, where the rounding of a coefficient near 1/2 may go either way.
        let cap = 64 + 2 * st.g[k][k].bits();
        let mut stalls = 0;
        for pass in 0..cap {
            self.row(st, k)?;
            if (0..k).all(|j| self.mu[k][j].abs_le(eta)) {
                return Ok(pass > 0);
            }
            let mut xs: Vec<(usize, Number)> = Vec::new();
            for i in (0..k).rev() {
                let x = self.mu[k][i].rint().ok_or(Failure)?;
                if x.is_zero() {
                    continue;
                }
                let xf = F::of_number(&x, p);
                let (lo, hi) = self.mu.split_at_mut(k);
                for j in 0..i {
                    hi[0][j] = hi[0][j].sub_mul(&xf, &lo[i][j]);
                }
                xs.push((i, x));
            }
            if xs.is_empty() {
                return Err(Failure);
            }
            let before = st.g[k][k].clone();
            st.apply(k, &xs);
            if st.g[k][k] >= before {
                stalls += 1;
                if stalls > 2 {
                    return Err(Failure);
                }
            } else {
                stalls = 0;
            }
        }
        Err(Failure)
    }
}

/// A bound on the iterations of a run: an exchange of LLL with the factor `δ̄` lowers the
/// potential `Π dᵢ`, at most `Πᵢ |bᵢ|^(2(n − i))` by Hadamard's inequality, below `δ̄` times
/// itself; the bound counts exchanges that lower it by `(1 + δ̄)/2` each, to leave room for the
/// rounding of the tests, adds the steps up, keeps below about `2·10¹⁵` (no run is that long), and
/// for deep insertions, which the potential does not bound, multiplies by `n + 1`. `gap` is
/// `1 − δ̄`, from the exact `δ̄`: positive, and at least the `1 − δ̄` of the float type, so that
/// the bound is right also where `δ̄` and `(1 + δ̄)/2` would round to 1 in an `f64`.
fn iteration_cap(st: &Basis, gap: f64, ins: Insertion) -> u64 {
    let n = st.active as u64;
    let bits: u64 = (0..st.active)
        .map(|i| (n - i as u64) * (st.g[i][i].bits() + 1))
        .sum();
    // −log₂(1 − gap/2), without the rounding of 1 − gap/2.
    let per = -(-gap / 2.0).ln_1p() / core::f64::consts::LN_2;
    let swaps = if per > 0.0 {
        (bits as f64 / per).min(1e15) as u64
    } else {
        1e15 as u64
    };
    let cap = 4 * n + 2 * swaps + 1000;
    match ins {
        Insertion::Lll => cap,
        Insertion::Deep(_) => cap.saturating_mul(n + 1),
    }
}

/// Runs L² with the floating-point type `F` of precision `prec` on `st`, with the tests at the
/// exact `delta` and `eta` (which leave a margin to the `δ` and `η` the result must meet), rounded
/// towards `δ` and `η`: `δ̄` down, never below `δ` (a float), and `η̄` up, never above `η`, so
/// that `δ ≤ δ̄ < 1` and `1/2 < η̄ ≤ η` in every float type; inserting as `ins` says. On success
/// the rows before `st.active` are reduced as far as the floating-point tests tell, and those
/// after it are zero; on failure `st` is a basis of the same lattice, with its transformation,
/// from which to go on.
pub(super) fn run<F: Fp>(
    st: &mut Basis,
    prec: u64,
    delta: &Rational,
    eta: &Rational,
    ins: Insertion,
) -> Result<(), Failure> {
    let n = st.b.len();
    let mut gso = Gso::<F>::new(n, prec);
    let delta_f = F::of_rational(delta, prec, Round::Down);
    let eta_f = F::of_rational(eta, prec, Round::Up);
    while st.active > 0 && st.g[0][0].is_zero() {
        st.drop_zero(0);
    }
    if st.active == 0 {
        return Ok(());
    }
    gso.r[0][0] = F::of_number(&st.g[0][0], prec);
    let gap = (&Rational::from(1) - delta).to_f64();
    let cap = iteration_cap(st, gap, ins);
    let mut iterations = 0u64;
    // The place of the last insertion, while the row it displaced is the one being reduced.
    let mut displaced: Option<usize> = None;
    let mut k = 1;
    while k < st.active {
        iterations += 1;
        if iterations > cap {
            return Err(Failure);
        }
        let last = displaced.take().filter(|&p| p + 1 == k);
        let changed = gso.size_reduce(st, k, &eta_f)?;
        if st.g[k][k].is_zero() {
            st.drop_zero(k);
            continue;
        }
        // Lovász's test at `i`: δ̄·r̄ᵢᵢ > s̄ᵢ fails the condition.
        let fails = |gso: &Gso<F>, i: usize| gso.s[i].lt(&delta_f.mul(&gso.r[i][i]));
        let lll_place = || {
            let mut i = k;
            while i > 0 && fails(&gso, i - 1) {
                i -= 1;
            }
            i
        };
        let place = match ins {
            Insertion::Lll => lll_place(),
            Insertion::Deep(_) => {
                match (0..k).find(|&i| ins.allowed(i, k, 0) && fails(&gso, i)) {
                    // A vector whose projection there looks empty is in the span of the rows
                    // before: it goes down as LLL takes it, as a dependent row does in MLLL.
                    Some(i) if gso.s[i].is_positive() => i,
                    Some(_) => lll_place(),
                    None => k,
                }
            }
        };
        if !gso.s[place].is_positive() {
            return Err(Failure);
        }
        // The row displaced by the last insertion, unchanged, going back in front of the row
        // inserted there: in exact arithmetic its test at that place holds (its projection there
        // is the old |b*ₚ|², above δ̄ times the new one), so only rounding can ask for it, and
        // the two rows would change places for ever.
        if let Some(p) = last {
            let undo = match ins {
                Insertion::Deep(Some(_)) => place == p,
                Insertion::Lll | Insertion::Deep(None) => place <= p,
            };
            if !changed && undo {
                return Err(Failure);
            }
        }
        if place != k {
            st.move_row(k, place);
            // The rows from `place + 1` on are computed again when they are reached; the row
            // moved keeps its coefficients on the rows before `place`.
            gso.r.swap(k, place);
            gso.mu.swap(k, place);
        }
        gso.r[place][place] = gso.s[place].clone();
        k = if place == k {
            k + 1
        } else {
            let next = ins.next_after(place);
            if next == place + 1 {
                displaced = Some(place);
            }
            next
        };
    }
    debug_assert!(st.gram_is_exact(), "the Gram matrix follows the rows");
    Ok(())
}
