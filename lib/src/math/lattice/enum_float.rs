//! The enumeration of Schnorr and Euchner in `f64`, made safe for an exact answer.
//!
//! The tree is Fincke and Pohst's: at level `l` the coefficient `x_l` ranges over the integers
//! with `ℓ_l = ℓ_{l+1} + (x_l − c_l)²·r_l ≤ R²`, where `c_l = t_l − Σ_{j>l} x_j·μ_{j,l}` is the
//! centre and `r_l = |b*_l|²`; Schnorr and Euchner visit them in the zig-zag order of
//! `|x_l − c_l|`, depth first, so that the radius can shrink as soon as a better leaf is found.
//! The centres are kept as partial sums `σ_{l,j} = Σ_{i≥j} x_i·μ_{i,l}`, each brought up to date
//! from the highest coefficient that changed since it was last computed, so that a node costs
//! `O(1)` on average (the bookkeeping of Gama, Nguyen and Regev, *Lattice enumeration using
//! extreme pruning*, Eurocrypt 2010, appendix).
//!
//! **Exactness.** Instead of `ℓ_l` the search compares a lower bound of it with the radius
//! enlarged by the rounding the bound can carry, so that it visits every node whose true
//! `ℓ_l ≤ R²`, and some more; the caller checks every leaf exactly. The bound:
//!
//! * the data `μ̃`, `t̃` and `r̃` are the exact rationals rounded once (relative error below
//!   `1.01·u`, `u = 2⁻⁵³`); values below `2⁻⁹⁰⁰` are taken as zero and the error that makes is
//!   covered by an absolute term; `r̃` is lowered by `4u` (and to at most `2¹⁰⁰⁰`), so that it is
//!   at most `r_l`;
//! * the centre is a recursive sum of at most `n` products of an integer and a float, whose
//!   error is at most `γ_n·Σ_{j>l} |x_j·μ̃_{j,l}|` (Higham, *Accuracy and Stability of Numerical
//!   Algorithms*, §3.1), with the errors of `μ̃` and `t̃` and of the last subtraction on top:
//!   `E_l = (2n + 16)·u·(Σ_{j>l}|x_j|·max_{j>l}|μ̃_{j,l}| + |t̃_l| + |c̃_l| + 2⁻⁸⁰⁰)` bounds all of
//!   them, with room for the rounding of `E_l` itself;
//! * the distance to the centre is then at least `y = |x_l − c̃_l|·(1 − 2u) − E_l` (or zero), as
//!   computed, and `ℓ̃_l = ℓ̃_{l+1} + y²·r̃_l` adds terms that are all non-negative, so that each
//!   level's rounding makes it at most `(1 + u)⁵` larger than the exact lower bound: over `n`
//!   levels `ℓ̃_l ≤ (1 + u)^{5n}·ℓ_l`. The caller enlarges the radius by more than that.
//!
//! Every comparison is false on a NaN, which would prune; none arises: the data are finite, a
//! node is entered only below a finite radius, and a centre beyond `2⁵⁰` stops the search (the
//! caller then enumerates exactly).

/// `u`, the unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;

/// `1 − 2u`, exactly.
const ONE_MINUS_2U: f64 = 1.0 - f64::EPSILON;

/// A centre or a sum of coefficients beyond this stops the float search: coefficients must stay
/// exact integers in an `f64`, with room for the zig-zag.
const LIMIT: f64 = (1u64 << 50) as f64;

/// The data of a float enumeration of `n` levels, numbered from 0 at the bottom.
pub(super) struct FloatLevels {
    pub(super) n: usize,
    /// `μ̃_{j,l}` at `j·n + l`, for `l < j`.
    pub(super) mu: Vec<f64>,
    /// Lower bounds of `r_l`, scaled by the same power of two as the radius.
    pub(super) r: Vec<f64>,
    /// `t̃_l`, the target's part of the centres; zeros without a target.
    pub(super) t: Vec<f64>,
    /// `max_{j>l} |μ̃_{j,l}| + 2⁻⁸⁰⁰`.
    pub(super) mmax: Vec<f64>,
    /// Without a target: the zero vector is left out, and of `±x` only the one whose highest
    /// nonzero coefficient is positive is visited.
    pub(super) symmetric: bool,
}

/// The search stopped because a coefficient grew too large for an `f64`.
#[derive(Debug)]
pub(super) struct TooLarge;

/// What a value below this is taken as: zero.
pub(super) const TINY: f64 = 1.0e-271; // below 2⁻⁹⁰⁰ ≈ 1.2·10⁻²⁷¹

/// `2⁻⁸⁰⁰`, the absolute part of the error bounds.
const ABS: f64 = 1.5e-241; // above 2⁻⁸⁰⁰ ≈ 1.5·10⁻²⁴¹

impl FloatLevels {
    /// The largest `|μ̃_{j,l}|` of each level, with the absolute term.
    pub(super) fn column_maxima(n: usize, mu: &[f64]) -> Vec<f64> {
        (0..n)
            .map(|l| {
                let m = (l + 1..n).map(|j| mu[j * n + l].abs()).fold(0.0, f64::max);
                m + ABS
            })
            .collect()
    }

    /// Visits, depth first in Schnorr and Euchner's order, every coefficient vector whose
    /// partial norms are within `radius` (scaled as `r`), and some more, and calls `leaf` with
    /// each one that reaches the bottom; `leaf` returns a smaller radius, or `None` to keep it.
    pub(super) fn run(
        &self,
        mut radius: f64,
        leaf: &mut dyn FnMut(&[f64]) -> Option<f64>,
    ) -> Result<(), TooLarge> {
        let n = self.n;
        if n == 0 {
            return Ok(());
        }
        let stride = n + 1;
        let kappa = (2 * n + 16) as f64 * U;
        let mut x = vec![0.0f64; n];
        let mut dx = vec![0.0f64; n];
        let mut centre = vec![0.0f64; n];
        let mut err = vec![0.0f64; n];
        // part[l]: the lower bound of ℓ_l of the coefficients above and at l; part[n] = 0.
        let mut part = vec![0.0f64; n + 1];
        // sigma[l·stride + j] = Σ_{i≥j} x_i·μ̃_{i,l} for l < j ≤ n; sigma[l·stride + n] = 0.
        let mut sigma = vec![0.0f64; n * stride];
        // stale[l]: the highest coefficient that changed since row l of sigma was computed (l
        // itself when none did).
        let mut stale = vec![n - 1; n];
        // above[l] = Σ_{j>l} |x_j|.
        let mut above = vec![0.0f64; n];
        // zero[l]: no target, and every x_j = 0 for j > l.
        let mut zero = vec![false; n];

        // Enters level l with the coefficients above it fixed: its centre, its error bound and
        // its first coefficient.
        macro_rules! enter {
            ($l:expr) => {{
                let l = $l;
                let top = stale[l];
                let row = l * stride;
                for j in (l + 1..=top).rev() {
                    sigma[row + j] = sigma[row + j + 1] + x[j] * self.mu[j * n + l];
                }
                if l > 0 && stale[l - 1] < top {
                    stale[l - 1] = top;
                }
                stale[l] = l;
                let c = self.t[l] - sigma[row + l + 1];
                above[l] = if l + 1 < n {
                    above[l + 1] + x[l + 1].abs()
                } else {
                    0.0
                };
                if !(c.abs() <= LIMIT && above[l] <= LIMIT) {
                    return Err(TooLarge);
                }
                centre[l] = c;
                zero[l] = self.symmetric && (l + 1 == n || (zero[l + 1] && x[l + 1] == 0.0));
                err[l] = kappa * (above[l] * self.mmax[l] + self.t[l].abs() + c.abs() + ABS);
                if zero[l] {
                    // The centre is 0: x_l = 0, 1, 2, … (and not 0 at the bottom).
                    x[l] = if l == 0 { 1.0 } else { 0.0 };
                    dx[l] = 1.0;
                } else {
                    let r = c.round();
                    x[l] = r;
                    dx[l] = if c >= r { 1.0 } else { -1.0 };
                }
                if l > 0 && stale[l - 1] < l {
                    stale[l - 1] = l;
                }
            }};
        }

        // The next coefficient of level l in the zig-zag: round(c), then alternately on the
        // side of c and the other, each further from c than the last.
        macro_rules! advance {
            ($l:expr) => {{
                let l = $l;
                if zero[l] {
                    x[l] += 1.0;
                } else {
                    x[l] += dx[l];
                    dx[l] = -dx[l] - dx[l].signum();
                }
                if x[l].abs() > LIMIT {
                    return Err(TooLarge);
                }
                if l > 0 && stale[l - 1] < l {
                    stale[l - 1] = l;
                }
            }};
        }

        let mut l = n - 1;
        enter!(l);
        loop {
            let d = (x[l] - centre[l]).abs();
            let mut y = d * ONE_MINUS_2U - err[l];
            if y < 0.0 {
                y = 0.0;
            }
            let p = part[l + 1] + y * y * self.r[l];
            if p <= radius {
                if l == 0 {
                    if let Some(r) = leaf(&x) {
                        radius = r;
                    }
                    advance!(0);
                    continue;
                }
                part[l] = p;
                l -= 1;
                enter!(l);
                continue;
            }
            // Every later coefficient of this level is further from the centre: up a level.
            l += 1;
            if l == n {
                return Ok(());
            }
            advance!(l);
        }
    }
}
