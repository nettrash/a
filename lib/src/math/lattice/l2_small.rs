//! L² ([`super::l2_core`]) for the bases whose rows, transformation and Gram matrix fit in
//! machine words, as fplll keeps them in `long`s when they fit: the rows, `U` and the whole Gram
//! matrix (both triangles) as flat arrays of `i64`, each row with the largest magnitude in it,
//! so that a row operation `y ← y − x·z` is checked once, `max|y| + |x|·max|z| < 2⁶³`, and then
//! runs on the words without a test; and the Gram–Schmidt data in plain `f64`. The algorithm is
//! that of [`super::l2_core::run`] step for step, with every float operation rounded once in
//! `f64`, while the words have room. A step whose words would overflow, or a float that is not
//! finite, ends the run with the basis it reached, from which [`super::l2_core`] goes on with
//! integers of any size: that run starts again from the basis (its Gram–Schmidt data computed
//! afresh), so its steps, and the reduced basis it ends with, can differ from those of a run in
//! integers of any size from the start (as on some bases of dependent rows, whose
//! transformations pass `2^63`: another reduced basis of the same lattice, or the same one but
//! for the sign of a row). Nothing here is proved; the result is certified afterwards like every
//! other.

use super::exact::Insertion;
use super::l2_core::{Basis, Failure};
use super::zint::Z;
use crate::math::float::{BigFloat, Round};
use crate::math::rational::Rational;

/// The largest magnitude a word may reach: `i64::MAX`.
const TOP: u128 = i64::MAX as u128;

/// The bound below which every entry of the Gram matrix must be for the words to be taken: room
/// for the first steps.
const GRAM_BELOW: u64 = 1 << 60;

/// The exact part of a reduction in words: rows, transformation and full Gram matrix, row-major,
/// with the largest magnitude of each row.
pub(super) struct Small {
    n: usize,
    m: usize,
    w: usize,
    b: Vec<i64>,
    u: Vec<i64>,
    g: Vec<i64>,
    bmax: Vec<u64>,
    umax: Vec<u64>,
    gmax: Vec<u64>,
    active: usize,
}

/// The largest magnitude in `v`.
fn max_abs(v: &[i64]) -> u64 {
    v.iter().fold(0, |m, x| m.max(x.unsigned_abs()))
}

/// `dst ← dst − x·src` for magnitudes that keep every result in a word, the largest magnitude of
/// the result returned.
#[inline]
fn axpy(dst: &mut [i64], src: &[i64], x: i64) -> u64 {
    let mut top = 0u64;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = d.wrapping_sub(x.wrapping_mul(s));
        top = top.max(d.unsigned_abs());
    }
    top
}

/// Rows `k` (to change) and `i ≠ k` of the row-major `v` of rows `width` long.
#[inline]
fn pair(v: &mut [i64], width: usize, k: usize, i: usize) -> (&mut [i64], &[i64]) {
    if k > i {
        let (lo, hi) = v.split_at_mut(k * width);
        (&mut hi[..width], &lo[i * width..(i + 1) * width])
    } else {
        let (lo, hi) = v.split_at_mut(i * width);
        (&mut lo[k * width..(k + 1) * width], &hi[..width])
    }
}

/// Whether `a + |x|·b` stays a word.
#[inline]
fn fits(a: u64, x: i64, b: u64) -> bool {
    u128::from(a) + u128::from(x.unsigned_abs()) * u128::from(b) <= TOP
}

/// The bits of a nonnegative word.
fn bits(x: i64) -> u64 {
    64 - u64::from(x.unsigned_abs().leading_zeros())
}

impl Small {
    /// The basis `st` in words, when its entries fit: rows and `U` in words, and every entry of
    /// the Gram matrix below [`GRAM_BELOW`].
    pub(super) fn of(st: &Basis) -> Option<Small> {
        let n = st.b.len();
        let m = st.b.first().map_or(0, Vec::len);
        let w = st.u.first().map_or(0, Vec::len);
        let word = |z: &Z| match z {
            Z::S(v) => Some(*v),
            Z::B(..) => None,
        };
        let mut b = Vec::with_capacity(n * m);
        let mut u = Vec::with_capacity(n * w);
        for i in 0..n {
            for z in &st.b[i] {
                b.push(word(z)?);
            }
            for z in &st.u[i] {
                u.push(word(z)?);
            }
        }
        let mut g = vec![0i64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = word(st.gram_entry(i, j))?;
                if v.unsigned_abs() >= GRAM_BELOW {
                    return None;
                }
                g[i * n + j] = v;
                g[j * n + i] = v;
            }
        }
        let bmax = (0..n).map(|i| max_abs(&b[i * m..(i + 1) * m])).collect();
        let umax = (0..n).map(|i| max_abs(&u[i * w..(i + 1) * w])).collect();
        let gmax = (0..n).map(|i| max_abs(&g[i * n..(i + 1) * n])).collect();
        Some(Small {
            n,
            m,
            w,
            b,
            u,
            g,
            bmax,
            umax,
            gmax,
            active: st.active,
        })
    }

    /// Back to integers of any size.
    pub(super) fn into_basis(self) -> Basis {
        let (n, m, w) = (self.n, self.m, self.w);
        let rows = |v: &[i64], width: usize| -> Vec<Vec<Z>> {
            (0..n)
                .map(|i| {
                    v[i * width..(i + 1) * width]
                        .iter()
                        .map(|&x| Z::S(x))
                        .collect()
                })
                .collect()
        };
        let g = (0..n)
            .map(|i| (0..=i).map(|j| Z::S(self.g[i * n + j])).collect())
            .collect();
        Basis::from_parts(rows(&self.b, m), rows(&self.u, w), g, self.active)
    }

    #[inline]
    fn gram(&self, i: usize, j: usize) -> i64 {
        self.g[i * self.n + j]
    }

    /// `bₖ ← bₖ − x·bᵢ`, with `U` and the Gram matrix; `Failure` where a word would overflow.
    fn sub_row(&mut self, k: usize, i: usize, x: i64) -> Result<(), Failure> {
        let (n, m, w) = (self.n, self.m, self.w);
        let gki = self.gram(k, i);
        let gii = self.gram(i, i);
        let gkk = self.gram(k, k);
        let room = |s: &Small| {
            fits(s.bmax[k], x, s.bmax[i])
                && fits(s.umax[k], x, s.umax[i])
                && fits(s.gmax[k], x, s.gmax[i])
        };
        if !room(self) {
            // The maxima of the Gram rows only grow between their recomputations: the true ones.
            for r in [k, i] {
                self.gmax[r] = max_abs(&self.g[r * n..(r + 1) * n]);
            }
            if !room(self) {
                return Err(Failure);
            }
        }
        // ⟨bₖ − x·bᵢ, bₖ − x·bᵢ⟩ = Gₖₖ − 2x·Gₖᵢ + x²·Gᵢᵢ, in 128 bits, once the room is known:
        // the maximum of row i of the Gram matrix bounds |Gᵢᵢ| and |Gₖᵢ|, so |x|·|Gᵢᵢ| and
        // |x|·|Gₖᵢ| are words, x²·|Gᵢᵢ| < 2^126 and 2|x·Gₖᵢ| < 2^64, and the sum cannot overflow.
        let xx = i128::from(x);
        let kk = i128::from(gkk) - 2 * xx * i128::from(gki) + xx * xx * i128::from(gii);
        if !(0..=i128::from(i64::MAX)).contains(&kk) {
            return Err(Failure);
        }
        let (dst, src) = pair(&mut self.b, m, k, i);
        self.bmax[k] = axpy(dst, src, x);
        let (dst, src) = pair(&mut self.u, w, k, i);
        self.umax[k] = axpy(dst, src, x);
        // Row k of the Gram matrix: ⟨bₖ, bⱼ⟩ − x·⟨bᵢ, bⱼ⟩ for every j, then the diagonal, then
        // column k from row k.
        let (dst, src) = pair(&mut self.g, n, k, i);
        axpy(dst, src, x);
        self.g[k * n + k] = kk as i64;
        let mut top = 0u64;
        for j in 0..n {
            let v = self.g[k * n + j];
            top = top.max(v.unsigned_abs());
            if j != k {
                self.g[j * n + k] = v;
                self.gmax[j] = self.gmax[j].max(v.unsigned_abs());
            }
        }
        self.gmax[k] = top;
        Ok(())
    }

    /// Moves the row `from` to the place `to`, the rows between shifting by one.
    fn move_row(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        let (n, m, w) = (self.n, self.m, self.w);
        let (lo, hi) = (from.min(to), from.max(to));
        let rotate = |v: &mut [i64], width: usize| {
            let block = &mut v[lo * width..(hi + 1) * width];
            if from > to {
                block.rotate_right(width);
            } else {
                block.rotate_left(width);
            }
        };
        rotate(&mut self.b, m);
        rotate(&mut self.u, w);
        rotate(&mut self.g, n);
        for row in self.g.chunks_exact_mut(n) {
            if from > to {
                row[lo..=hi].rotate_right(1);
            } else {
                row[lo..=hi].rotate_left(1);
            }
        }
        for v in [&mut self.bmax, &mut self.umax, &mut self.gmax] {
            if from > to {
                v[lo..=hi].rotate_right(1);
            } else {
                v[lo..=hi].rotate_left(1);
            }
        }
    }

    fn drop_zero(&mut self, k: usize) {
        self.move_row(k, self.active - 1);
        self.active -= 1;
    }
}

/// The Gram–Schmidt data in `f64`, row-major `n × n`, as [`super::l2_core`]'s.
struct Gso {
    n: usize,
    r: Vec<f64>,
    mu: Vec<f64>,
    s: Vec<f64>,
}

impl Gso {
    /// The coefficients of row `k` from the Gram matrix and the rows before it, and the squared
    /// norms of its projections.
    fn row(&mut self, st: &Small, k: usize) -> Result<(), Failure> {
        let n = self.n;
        for j in 0..k {
            let mut t = st.gram(k, j) as f64;
            let (rk, mj) = (&self.r[k * n..k * n + j], &self.mu[j * n..j * n + j]);
            for (a, b) in mj.iter().zip(rk) {
                t -= a * b;
            }
            let q = t / self.r[j * n + j];
            if !q.is_finite() {
                return Err(Failure);
            }
            self.r[k * n + j] = t;
            self.mu[k * n + j] = q;
        }
        self.s[0] = st.gram(k, k) as f64;
        for j in 0..k {
            self.s[j + 1] = self.s[j] - self.mu[k * n + j] * self.r[k * n + j];
        }
        Ok(())
    }

    /// The lazy size reduction of row `k`, as [`super::l2_core`]'s: until every `|μₖⱼ| ≤ η̄`.
    fn size_reduce(&mut self, st: &mut Small, k: usize, eta: f64) -> Result<bool, Failure> {
        let n = self.n;
        let cap = 64 + 2 * bits(st.gram(k, k));
        let mut stalls = 0;
        let mut xs: Vec<(usize, i64)> = Vec::new();
        for pass in 0..cap {
            self.row(st, k)?;
            if self.mu[k * n..k * n + k].iter().all(|m| m.abs() <= eta) {
                return Ok(pass > 0);
            }
            xs.clear();
            for i in (0..k).rev() {
                let x = self.mu[k * n + i].round();
                if x == 0.0 {
                    continue;
                }
                if x.is_nan() || x.abs() >= 4.0e18 {
                    return Err(Failure);
                }
                let (lo, hi) = self.mu.split_at_mut(k * n);
                for (a, b) in hi[..i].iter_mut().zip(&lo[i * n..i * n + i]) {
                    *a -= x * b;
                }
                xs.push((i, x as i64));
            }
            if xs.is_empty() {
                return Err(Failure);
            }
            let before = st.gram(k, k);
            for &(i, x) in &xs {
                st.sub_row(k, i, x)?;
            }
            if st.gram(k, k) >= before {
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

/// `r` rounded in the direction `round` to an `f64`, as [`super::l2_float::F64`] takes it.
fn of_rational(r: &Rational, round: Round) -> f64 {
    BigFloat::from_rational(r, 53, round)
        .0
        .to_f64_round(round)
        .0
}

/// [`super::l2_core::run`] in words and `f64`: the same steps, the same tests, the same failures,
/// and a failure, besides, where a word would overflow (after which the run in integers of any
/// size starts again from the basis reached, and need not end with the basis of a run that had
/// been in them from the start).
pub(super) fn run(
    st: &mut Small,
    delta: &Rational,
    eta: &Rational,
    ins: Insertion,
) -> Result<(), Failure> {
    let n = st.n;
    let mut gso = Gso {
        n,
        r: vec![0.0; n * n],
        mu: vec![0.0; n * n],
        s: vec![0.0; n + 1],
    };
    let delta_f = of_rational(delta, Round::Down);
    let eta_f = of_rational(eta, Round::Up);
    if !(delta_f.is_finite() && eta_f.is_finite()) {
        return Err(Failure);
    }
    while st.active > 0 && st.gram(0, 0) == 0 {
        st.drop_zero(0);
    }
    if st.active == 0 {
        return Ok(());
    }
    gso.r[0] = st.gram(0, 0) as f64;
    let gap = (&Rational::from(1) - delta).to_f64();
    let cap = iteration_cap(st, gap, ins);
    let mut iterations = 0u64;
    let mut displaced: Option<usize> = None;
    let mut k = 1;
    while k < st.active {
        iterations += 1;
        if iterations > cap {
            return Err(Failure);
        }
        let last = displaced.take().filter(|&p| p + 1 == k);
        let changed = gso.size_reduce(st, k, eta_f)?;
        if st.gram(k, k) == 0 {
            st.drop_zero(k);
            continue;
        }
        let fails = |gso: &Gso, i: usize| gso.s[i] < delta_f * gso.r[i * n + i];
        let lll_place = || {
            let mut i = k;
            while i > 0 && fails(&gso, i - 1) {
                i -= 1;
            }
            i
        };
        let place = match ins {
            Insertion::Lll => lll_place(),
            Insertion::Deep(_) => match (0..k).find(|&i| ins.allowed(i, k, 0) && fails(&gso, i)) {
                Some(i) if gso.s[i] > 0.0 => i,
                Some(_) => lll_place(),
                None => k,
            },
        };
        if !(gso.s[place] > 0.0 && gso.s[place].is_finite()) {
            return Err(Failure);
        }
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
            // The row moved keeps its coefficients on the rows before `place`.
            for v in [&mut gso.r, &mut gso.mu] {
                let (lo, hi) = v.split_at_mut(k * n);
                lo[place * n..place * n + n].swap_with_slice(&mut hi[..n]);
            }
        }
        gso.r[place * n + place] = gso.s[place];
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
    Ok(())
}

/// [`super::l2_core`]'s bound on the iterations, from the same potential.
fn iteration_cap(st: &Small, gap: f64, ins: Insertion) -> u64 {
    let n = st.active as u64;
    let bits: u64 = (0..st.active)
        .map(|i| (n - i as u64) * (bits(st.gram(i, i)) + 1))
        .sum();
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
