//! PSLQ's iterations in doubles between exact refreshes: the two levels of the multi-level PSLQ
//! of Bailey and Broadhurst ("Parallel integer relation detection: techniques and
//! applications", Math. Comp. 70, 2001).
//!
//! A block takes `H / 2^frac` and `y` (scaled to a largest entry below 1) to doubles from the
//! exact state, and `A' = B' = I` in words, and runs the iterations of the exact search on them:
//! the same exchange, rotation and reductions, the multipliers rounded from the doubles. Before
//! each iteration after the first it ends where the exact search could stop: an entry of `y`
//! (a changed one, the dot product of the first `y` and a column of `B'`) that the doubles, with
//! a bound on their error, cannot tell apart from the tolerance of a relation; the floating
//! bound `1/max |Hⱼⱼ|` within half a bit of a proof; the test of the precision within two bits,
//! `A` and `B` taken at their largest. It ends there too once an entry of `A'` or `B'` has
//! passed `2^26` (the doubles of `H` lose about as many bits), and inside an iteration when one
//! passes `2^31` or where the doubles cannot go on (a zero diagonal entry, a multiplier beyond
//! `2^31`): the exact search then makes the iteration's other reductions (and an iteration
//! whose rotation the doubles cannot make is left to it with its exchange).
//!
//! The exact state then takes the block: `y ← y·B'`, `B ← B·B'`, `A ← A'·A` and `H ← A'·H`,
//! all exactly (the rotations of the block are not carried over: `H` is `A·H₀` times an
//! orthogonal matrix, which leaves its rows' Gram matrix, all that PSLQ reads of it, unchanged).
//! The next block starts from the lower factor of that `H` in doubles, by Householder
//! reflections; the exact one, by reflections rounded in the fixed point of `H`, is made only
//! for exact iterations and where a test of the stops is within the margins (the factor is the
//! same as the rotations' up to the signs of its columns, which are set to the block's). The
//! doubles only steer: every multiplier is an integer, so `A` stays exact and unimodular,
//! `B = A⁻¹` and `y = X·B` exact, and relations and bounds are tested and proved on them as
//! before.

use super::{Search, TRUSTED_BITS, log2_abs, round_div};
use crate::math::lattice::zint::Z;
use crate::math::numbers::Number;

/// The size of `n` from which the search runs in blocks. Measured on an Apple M4 Max on the
/// bench's polynomials, logarithms and bounds and on random relations of 3 to 24 numbers: from
/// 5, 6 or 8 the exact iterations took 4 to 11 times as long at 4 to 7 numbers; from 3, one input
/// of 3 numbers took 0.53 of the time and another 1.07.
pub(super) const FROM: usize = 4;

/// The iterations of a block at most. Measured from 1024 to 16384 on the same inputs: no
/// difference beyond 0.3 %, the blocks ending on the other conditions first.
const ITERATIONS: usize = 1 << 12;

/// The largest entry of `A'` and `B'`, and multiplier, in a block: the largest for which a
/// reduction's products and sums stay in an `i64` (`2^31 + 2^31·2^31 < 2^63`). Measured on an
/// Apple M4 Max on the bench's polynomials, logarithms and bounds and on 49 random relations of 3
/// to 24 numbers: against `2^30`, 0.63 to 0.71 of the time for 5 inputs whose multipliers pass
/// `2^30`, 0.998 to 1.005 for the others, every answer the same; `2^26` to `2^29` took up to 4.9
/// times as long.
const ENTRY: i64 = 1 << 31;

/// The entry of `A'` or `B'` past which a block ends between two iterations, so that few end
/// inside one (whose other reductions need the exact lower form). Measured from `2^20` to
/// `2^29` on the same inputs: the best bound differs from input to input (`2^24` took 0.73 of
/// the time for 16 numbers but 1.02 to 1.03 for 6, 13 and 24; `2^28` 0.94 for the polynomial of
/// degree 6 but 1.2 to 2.5 for some of 8 to 24 numbers), and none is faster for all.
const SOFT: i64 = 1 << 26;

/// A block of fewer iterations than this is followed by [`PAUSE`] exact ones. Measured from 4
/// to 64 and from 16 to 256: no difference beyond 0.5 % but for a `SHORT` of 64, which took up
/// to 5.8 times as long.
pub(super) const SHORT: usize = 16;
pub(super) const PAUSE: usize = 64;

/// The unit roundoff of the doubles.
const U: f64 = f64::EPSILON / 2.0;

/// `f·2^k`, in two steps so that neither factor leaves the range of the doubles (zero below it).
fn ldexp(f: f64, k: i64) -> f64 {
    let k = k.clamp(-2400, 2400) as i32;
    let a = k / 2;
    f * f64::powi(2.0, a) * f64::powi(2.0, k - a)
}

/// `z·2^s` as a double.
fn to_f64(z: &Z, s: i64) -> f64 {
    if z.is_zero() {
        return 0.0;
    }
    let (f, e) = z.top();
    let v = ldexp(f, e + s);
    if z.is_negative() { -v } else { v }
}

/// The bits of `|v|`.
fn bits(v: i64) -> u64 {
    u64::from(64 - v.unsigned_abs().leading_zeros())
}

fn identity(n: usize) -> Vec<i64> {
    (0..n * n).map(|k| i64::from(k / n == k % n)).collect()
}

impl Search {
    /// Up to `cap` iterations in doubles, then carried over to the exact state, an iteration
    /// cut short in its reductions finished there by the exact ones. Returns the iterations run,
    /// 0 when none could start, and `None` where an exact reduction finds the precision run out.
    pub(super) fn block(&mut self, cap: usize) -> Option<usize> {
        let n = self.n;
        let w = n - 1;
        let cap = cap.min(ITERATIONS);
        let frac = self.frac as i64;
        let mut hd: Vec<f64> = self.h.iter().map(|z| to_f64(z, -frac)).collect();
        if !self.lower {
            // `A·H₀` since the last lower form: its lower factor, in doubles.
            lq_f64(&mut hd, n);
        }
        // y / 2^ytop at the start; an entry of y·B' is its dot product with a column of B'.
        let ytop = self.y.iter().map(Z::bits).max().unwrap_or(0) as i64;
        let y0: Vec<f64> = self.y.iter().map(|z| to_f64(z, -ytop)).collect();
        // The columns of B' changed since the last test of y.
        let mut changed = vec![false; n];
        let (mut ap, mut bp) = (identity(n), identity(n));
        let (mut amax, mut bmax) = (1i64, 1i64);
        // ⌈log₂ n⌉ at least.
        let lg = u64::from(usize::BITS - n.leading_zeros());
        let a_bits = self.a.iter().map(Z::bits).max().unwrap_or(0);
        let b_bits = self.b.iter().map(Z::bits).max().unwrap_or(0);
        let err_top = self.input.err.iter().copied().max().unwrap_or(0);
        let max_norm_log2 = self.max_norm.as_ref().map(log2_abs);
        let mut done = 0;
        // Where an iteration was cut short: its exchange r and the reduction (i, j) it stopped
        // at, made or not.
        let mut cut: Option<(usize, usize, usize, bool)> = None;
        'run: while done < cap {
            if done > 0 {
                if amax > SOFT || bmax > SOFT {
                    break;
                }
                // The tolerance of a relation, Σ |Bᵢⱼ|·δᵢ ≤ n·max |B|·max δ, over 2^ytop.
                let b_now = b_bits + bits(bmax) + lg + 1;
                let tol = ldexp(1.0, (b_now + err_top + lg) as i64 - ytop);
                // The entries of y that changed (an exchange only moves them with their
                // columns), each a dot product of doubles and integers below 2^53: its error is
                // below (n + 4)·u·Σ |y0ᵢ|·|B'ᵢⱼ|, the conversions of y0 counted.
                let mut near = false;
                for j in 0..n {
                    if !changed[j] {
                        continue;
                    }
                    let (mut v, mut a) = (0.0f64, 0.0f64);
                    for i in 0..n {
                        let b = bp[i * n + j] as f64;
                        v += y0[i] * b;
                        a += y0[i].abs() * b.abs();
                    }
                    if v.abs() <= 2.0 * (n + 4) as f64 * U * a + tol {
                        near = true;
                        break;
                    }
                    changed[j] = false;
                }
                if near {
                    break;
                }
                let largest = (0..w).map(|j| hd[j * w + j].abs()).fold(0.0, f64::max);
                if largest.is_nan() || largest <= 0.0 {
                    break;
                }
                let lt = largest.log2();
                if let Some(m) = max_norm_log2 {
                    let floating = -lt;
                    if floating + 0.5 >= m && floating + 0.5 > self.proved_at + 1.0 {
                        break;
                    }
                }
                let size = (a_bits + bits(amax)).max(b_bits + bits(bmax)) + lg + 1;
                if (size + TRUSTED_BITS + 2) as f64 >= lt + frac as f64 + 1.0 {
                    break;
                }
            }
            done += 1;
            // The exchange of the rows r and r + 1 that maximise γ^r·|Hᵣᵣ|.
            let mut r = 0;
            let mut best = f64::NEG_INFINITY;
            for i in 0..w {
                let v = hd[i * w + i].abs().log2() + (i + 1) as f64 * self.gamma_log2;
                if v > best {
                    best = v;
                    r = i;
                }
            }
            changed.swap(r, r + 1);
            for k in 0..w {
                hd.swap(r * w + k, (r + 1) * w + k);
            }
            for k in 0..n {
                ap.swap(r * n + k, (r + 1) * n + k);
                bp.swap(k * n + r, k * n + r + 1);
            }
            // The rotation of the columns r and r + 1 that makes Hᵣ,ᵣ₊₁ zero.
            if r + 2 < n {
                let (p, q) = (hd[r * w + r], hd[r * w + r + 1]);
                let t0 = p.hypot(q);
                if t0.is_nan() || t0 <= 0.0 || t0.is_infinite() {
                    // The exchange undone: the exact iteration makes it again.
                    changed.swap(r, r + 1);
                    for k in 0..w {
                        hd.swap(r * w + k, (r + 1) * w + k);
                    }
                    for k in 0..n {
                        ap.swap(r * n + k, (r + 1) * n + k);
                        bp.swap(k * n + r, k * n + r + 1);
                    }
                    done -= 1;
                    break;
                }
                let (c, s) = (p / t0, q / t0);
                for i in r..n {
                    let (x, y) = (hd[i * w + r], hd[i * w + r + 1]);
                    hd[i * w + r] = c * x + s * y;
                    hd[i * w + r + 1] = c * y - s * x;
                }
                hd[r * w + r + 1] = 0.0;
            }
            // The reductions of the rows below, as the exact search makes them.
            for i in r + 1..n {
                for j in (0..=(i - 1).min(r + 1)).rev() {
                    let hjj = hd[j * w + j];
                    if hjj == 0.0 || hjj.is_nan() {
                        cut = Some((r, i, j, false));
                        break 'run;
                    }
                    // |Hᵢⱼ/Hⱼⱼ| < 1/2 exactly when |Hᵢⱼ| < |Hⱼⱼ|/2: no division for a zero t.
                    if hd[i * w + j].abs() < 0.5 * hjj.abs() {
                        continue;
                    }
                    let t = (hd[i * w + j] / hjj).round();
                    if t == 0.0 {
                        continue;
                    }
                    if t.is_nan() || t.abs() >= ENTRY as f64 {
                        cut = Some((r, i, j, false));
                        break 'run;
                    }
                    // |t|, |A'|, |B'| ≤ 2^31: the products and sums fit.
                    let ti = t as i64;
                    changed[j] = true;
                    for k in 0..=j {
                        hd[i * w + k] -= t * hd[j * w + k];
                    }
                    for k in 0..n {
                        let v = ap[i * n + k] - ti * ap[j * n + k];
                        ap[i * n + k] = v;
                        amax = amax.max(v.abs());
                        let u = bp[k * n + j] + ti * bp[k * n + i];
                        bp[k * n + j] = u;
                        bmax = bmax.max(u.abs());
                    }
                    if amax > ENTRY || bmax > ENTRY {
                        cut = Some((r, i, j, true));
                        break 'run;
                    }
                }
            }
        }
        if done > 0 {
            self.carry(&ap, &bp, &hd);
        }
        if let Some((r, i0, j0, made)) = cut {
            // The rest of the iteration's reductions, in their order (rows down, columns from
            // the right), exactly, on the lower form.
            self.ensure_lower();
            for i in i0..n {
                for j in (0..=(i - 1).min(r + 1)).rev() {
                    if i == i0 && (j > j0 || (j == j0 && made)) {
                        continue;
                    }
                    self.reduce(i, j)?;
                }
            }
        }
        Some(done)
    }

    /// The exact state after a block of transforms `A'` and `B'`, `hd` the block's `H`.
    fn carry(&mut self, ap: &[i64], bp: &[i64], hd: &[f64]) {
        let n = self.n;
        let w = n - 1;
        let wk = &mut self.w;
        let mut y = vec![Z::S(0); n];
        for (j, yj) in y.iter_mut().enumerate() {
            for i in 0..n {
                let t = bp[i * n + j];
                if t != 0 {
                    yj.add_mul(&Z::S(t), &self.y[i], wk);
                }
            }
        }
        self.y = y;
        let mut a = vec![Z::S(0); n * n];
        for i in 0..n {
            for l in 0..n {
                let t = ap[i * n + l];
                if t != 0 {
                    for k in 0..n {
                        a[i * n + k].add_mul(&Z::S(t), &self.a[l * n + k], wk);
                    }
                }
            }
        }
        self.a = a;
        let mut b = vec![Z::S(0); n * n];
        for k in 0..n {
            for l in 0..n {
                let s = &self.b[k * n + l];
                if s.is_zero() {
                    continue;
                }
                for j in 0..n {
                    let t = bp[l * n + j];
                    if t != 0 {
                        b[k * n + j].add_mul(&Z::S(t), s, wk);
                    }
                }
            }
        }
        self.b = b;
        let mut g = vec![Z::S(0); n * w];
        for i in 0..n {
            for l in 0..n {
                let t = ap[i * n + l];
                if t != 0 {
                    for k in 0..w {
                        g[i * w + k].add_mul(&Z::S(t), &self.h[l * w + k], wk);
                    }
                }
            }
        }
        // H ← A'·H, its lower form left until an exact step needs it; the block's diagonal for
        // the tests of the stops until then.
        self.h = g;
        self.lower = false;
        self.diag = (0..w).map(|j| hd[j * w + j]).collect();
    }

    /// Whether a test of [`Search::check_stop`] on `H` is within the block's margins, from the
    /// doubles of the last block: then it is made on the lower form.
    pub(super) fn near_stop(&self) -> bool {
        let largest = self.diag.iter().map(|d| d.abs()).fold(0.0, f64::max);
        if largest.is_nan() || largest <= 0.0 {
            return true;
        }
        let lt = largest.log2();
        if let Some(m) = &self.max_norm {
            let floating = -lt;
            if floating + 0.5 >= log2_abs(m) && floating + 0.5 > self.proved_at + 1.0 {
                return true;
            }
        }
        let size = self.a.iter().chain(&self.b).map(Z::bits).max().unwrap_or(0);
        (size + TRUSTED_BITS + 2) as f64 >= lt + self.frac as f64 + 1.0
    }

    /// `H` made lower trapezoidal again after blocks, by [`lower`].
    pub(super) fn ensure_lower(&mut self) {
        if !self.lower {
            self.h = lower(&self.h, self.n, self.frac, &self.diag);
            self.lower = true;
        }
    }
}

/// `g` (`n × (n − 1)` by rows) times an orthogonal matrix, lower trapezoidal, in doubles: by
/// Householder reflections.
fn lq_f64(g: &mut [f64], n: usize) {
    let w = n - 1;
    for i in 0..w {
        let m = w - i;
        if m < 2 {
            break;
        }
        let s2: f64 = (i..w).map(|j| g[i * w + j] * g[i * w + j]).sum();
        if s2 == 0.0 || !s2.is_finite() {
            continue;
        }
        let sigma = s2.sqrt();
        let u0 = g[i * w + i];
        let v0 = if u0 < 0.0 { u0 - sigma } else { u0 + sigma };
        let beta = sigma * (sigma + u0.abs());
        for k in i + 1..n {
            let mut d = g[k * w + i] * v0;
            for j in 1..m {
                d += g[k * w + i + j] * g[i * w + i + j];
            }
            let c = d / beta;
            g[k * w + i] -= c * v0;
            for j in 1..m {
                g[k * w + i + j] -= c * g[i * w + i + j];
            }
        }
        g[i * w + i] = if u0 < 0.0 { sigma } else { -sigma };
        for j in i + 1..w {
            g[i * w + j] = 0.0;
        }
    }
}

/// `G` (`n × (n − 1)` by rows, in fixed point at `frac` bits) times an orthogonal matrix, lower
/// trapezoidal: by Householder reflections, rounded at `frac` bits, the signs of the diagonal
/// those of `diag`.
fn lower(g: &[Z], n: usize, frac: usize, diag: &[f64]) -> Vec<Z> {
    let w = n - 1;
    let mut g: Vec<Number> = g.iter().map(Z::number).collect();
    let half = Number::from(1) << (frac - 1);
    for i in 0..w {
        let m = w - i;
        if m > 1 {
            let s2: Number = (i..w).map(|j| &g[i * w + j] * &g[i * w + j]).sum();
            if !s2.is_zero() {
                let sigma = s2.isqrt();
                let u0 = g[i * w + i].clone();
                // v: the row with v₀ = u₀ + sign(u₀)·σ, no cancellation; |v|²/2 = σ(σ + |u₀|).
                let v0 = if u0.is_negative() {
                    &u0 - &sigma
                } else {
                    &u0 + &sigma
                };
                let beta = &sigma * &(&sigma + &u0.abs());
                let v: Vec<Number> = core::iter::once(v0)
                    .chain((i + 1..w).map(|j| g[i * w + j].clone()))
                    .collect();
                if !beta.is_zero() {
                    for k in i + 1..n {
                        let mut s = Number::from(0);
                        for (j, vj) in v.iter().enumerate() {
                            s += &g[k * w + i + j] * vj;
                        }
                        if s.is_zero() {
                            continue;
                        }
                        // c·2^frac for the row's coefficient c = (g_k·v)/β.
                        s <<= frac;
                        let c = round_div(&s, &beta);
                        for (j, vj) in v.iter().enumerate() {
                            // c·v_j / 2^frac, rounded half up, as round_shr.
                            let mut p = &c * vj;
                            p += &half;
                            p >>= frac;
                            g[k * w + i + j] -= &p;
                        }
                    }
                    // The row itself: (−sign(u₀)·σ, 0, …, 0).
                    g[i * w + i] = if u0.is_negative() { sigma } else { -sigma };
                    for j in i + 1..w {
                        g[i * w + j] = Number::from(0);
                    }
                }
            }
        }
        // The sign of the block's diagonal entry.
        let d = &g[i * w + i];
        if !d.is_zero() && d.is_negative() != (diag[i] < 0.0) {
            for k in i..n {
                g[k * w + i] = -core::mem::take(&mut g[k * w + i]);
            }
        }
    }
    g.iter().map(Z::of).collect()
}
