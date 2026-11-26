//! Clenshaw–Curtis quadrature (Clenshaw and Curtis 1960), adaptive in the degree of each panel
//! and in the panels.
//!
//! A panel holds the values of the integrand at the `N + 1` Chebyshev–Lobatto points
//! `mid + half·cos(kπ/N)` of its interval, `N = 2^m`, from `m = 5` (33 points) to `m = 8` (257).
//! The points of `N` are every other point of `2N`, so a panel doubles its degree by evaluating
//! only the `N` new ones, and the estimates of the coarser rules `N/2`, `N/4`, … are sums over
//! the values it already has. The weights are those of the interpolant integrated exactly, from
//! the cosine series `w_k = (c_k/N)(1 − Σ_{j≤N/2} b_j cos(2jkπ/N)/(4j² − 1))` (Trefethen 2008).
//!
//! The error estimate of a panel at `N` asks for two successive agreements: it is the larger of
//! `|I_N − I_{N/2}|` and `|I_{N/2} − I_{N/4}|`, enlarged by the sum `r/(1 − r)` of the geometric
//! tail where the ratio `r` of the two is a half or more, and never less than `50 ε ∫|f|`. Nested
//! points cannot see an integrand that aliases on all of them — `T₆₄` is 1 at the 33 points and
//! at the 17 and 9 among them — so each panel is also evaluated at two points off its grid, and
//! where the interpolant through the points of the level misses the integrand there by more than
//! `√ε` of its size, the miss times the width of the panel is the least its error can be
//! (Chebfun's sample test, Trefethen et al.).
//!
//! The panel whose error is largest is refined: doubled, or, at 257 points, bisected into two
//! panels of 33 points that reuse the values at its ends and its middle. A value of the panel
//! that the points of the half it lies in do not account for (neither point beside it a tenth as
//! large) is kept with the half as a [`Witness`], its share of the panel's `∫|f|` a lower bound
//! on the half's error, until a point beside it sees a tenth of it; a half keeps up to four, the
//! heaviest, as an interval of QAGS does. What is guaranteed is that much: a seen value is not
//! dropped while the points about it see less than a tenth of it. A point that sees a tenth
//! accounts for it, and if that panel is bisected in turn the point's own value and share, of
//! the smaller panel, are what stands for the feature: the evidence of a narrow peak can so
//! shrink by a tenth and more a step and fall below a loose tolerance before a point finds the
//! peak, and a value seen far down a peak's tail is a share far below the peak's. A bisected panel
//! keeps its own estimate, and the estimate of the whole is the best cover of the interval: for
//! each panel, the sum of its halves where their errors add up to less than its own, else its
//! own. The errors of the panels in the cover are summed, each holding what the rounding of its
//! points could move its sum by, and, as in QAGS, a cover whose error is a tenth of its estimate
//! or more is not accepted unless the error is at the round-off of the sums.

use super::kronrod::{WITNESSES, resolved};
use super::{Integrand, Outcome, Settings};
use crate::math::numerics::Real;
use core::f64::consts::PI;
use std::collections::BinaryHeap;

/// The first and the last level, `N = 2^m`.
const FIRST: u32 = 5;
const LAST: u32 = 8;
/// The coarsest level whose estimate is read off a panel, for the error estimate of `FIRST`.
const COARSEST: u32 = FIRST - 2;
/// Where a panel is checked off its grid, on `[-1, 1]`: rational points, which no Chebyshev
/// point `cos(kπ/N)` but 0 and ±1 can be (Niven's theorem).
const SAMPLES: [f64; 2] = [0.37, -0.61];

/// The weights of the rule of `N = 2^m` points on `[-1, 1]`, for every level up to [`LAST`].
struct Weights {
    by_level: Vec<Vec<f64>>,
}

impl Weights {
    /// The table, computed once: the weights depend on nothing but the level, and computing all
    /// four levels by the cosine series costs about 44,000 cosines, far more than the
    /// integration of a smooth integrand.
    fn get() -> &'static Self {
        static TABLE: std::sync::OnceLock<Weights> = std::sync::OnceLock::new();
        TABLE.get_or_init(Self::new)
    }

    fn new() -> Self {
        let by_level = (0..=LAST)
            .map(|m| {
                if m < COARSEST {
                    Vec::new()
                } else {
                    weights(1 << m)
                }
            })
            .collect();
        Self { by_level }
    }
}

/// The Clenshaw–Curtis weights of the `N + 1` points `cos(kπ/N)`, `N` even, which sum to 2. The
/// cosine series cancels where the weights are smallest, at the ends — `w₀ = (1 − S)/N` with
/// `S` near 1 — and costs them relative accuracy (they stay within 4 units in the last place of
/// the largest weight, which is what a sum of the rule's terms feels); the two end weights have
/// the closed form `1/(N² − 1)` and take it.
fn weights(n: usize) -> Vec<f64> {
    let half = n / 2;
    let ends = 1.0 / ((n * n - 1) as f64);
    (0..=n)
        .map(|k| {
            if k == 0 || k == n {
                return ends;
            }
            let theta = PI * k as f64 / n as f64;
            let mut s = 0.0;
            for j in 1..=half {
                let b = if j == half { 1.0 } else { 2.0 };
                let jf = j as f64;
                s += b * (2.0 * jf * theta).cos() / (4.0 * jf * jf - 1.0);
            }
            let c = if k == 0 || k == n { 1.0 } else { 2.0 };
            c * (1.0 - s) / n as f64
        })
        .collect()
}

/// `cos(kπ/N)` as `sin((N/2 − k)π/N)`: exactly 0 in the middle and symmetric.
fn cosine(k: usize, n: usize) -> f64 {
    (PI * (n as f64 / 2.0 - k as f64) / n as f64).sin()
}

/// The point `k` of the grid of `N + 1` points on `[a, b]`, `k = 0` at `b`, as [`evaluate`]
/// evaluates the integrand there.
fn grid_point(a: f64, b: f64, k: usize, n: usize) -> f64 {
    let (mid, half) = (a + 0.5 * (b - a), 0.5 * (b - a));
    if k == 0 {
        b
    } else if k == n {
        a
    } else if 2 * k == n {
        mid
    } else {
        mid + half * cosine(k, n)
    }
}

/// A panel, or a node of the tree of bisections.
struct Panel {
    a: f64,
    b: f64,
    /// The level, `N = 2^level`.
    level: u32,
    /// The values at the `N + 1` points, `k = 0` at `b`; emptied once the panel is bisected.
    values: Vec<f64>,
    /// The values at the two points off the grid, `SAMPLES` mapped onto the panel.
    samples: [f64; 2],
    value: f64,
    error: f64,
    /// `∫|f|` over the panel by its rule.
    resabs: f64,
    /// Its halves, once bisected, and the panel it is a half of.
    halves: Option<(usize, usize)>,
    parent: Option<usize>,
    /// The best cover of its interval, kept up to date from the leaves up: the estimate, its
    /// error and `∫|f|`.
    cover: (f64, f64, f64),
    /// Samples of the panels this one was bisected from, inside this one, that the points of
    /// this one do not account for: the evidence of features between them (see [`Witness`]), at
    /// most [`WITNESSES`], the heaviest first.
    witnesses: Vec<Witness>,
}

/// A value the integrand was seen to take, at `x`, that the points of a panel do not account
/// for, and its share `mass` of the `∫|f|` of the panel that saw it. A panel at 257 points that
/// sees a peak at one point is bisected into halves of 33 points, which can miss it entirely, see
/// nothing, and agree on nothing to the last digit: their cover replaced the panel, and
/// `∫₀¹ e^(−((x−0.286)/0.001)²) dx` came out `9.9·10⁻⁶⁰` within its estimate after a point had
/// seen 0.95 (review round 2). The witness stays with the half, and with its halves, as a lower
/// bound on its error, until the points about it see a tenth of what it saw.
///
/// A half keeps up to [`WITNESSES`] of them, as an interval of QAGS does, and the least its
/// error can be is the sum of their masses. One was too few: of two peaks of width `3·10⁻⁴`
/// about 0.160 and 0.303 on `[0, 1]`, which the panel of 257 points saw at 0.9997 and 0.98 of
/// their heights, the half `[0, 0.5]` kept the evidence of the heavier, its points then found
/// that one, and the other was lost: `5.3·10⁻⁴ ± 2.8·10⁻¹⁰`, converged, at `10⁻⁸` for `8.5·10⁻⁴`
/// (the confirmation of review round 2).
#[derive(Clone, Copy, Debug)]
struct Witness {
    x: f64,
    value: f64,
    mass: f64,
}

/// How far below a witness the points of a panel next to it may see the integrand before the
/// panel is taken not to account for it: a factor of ten, as in QAGS (`kronrod::EVIDENCE`).
const EVIDENCE: f64 = 10.0;

/// The points of `panel` and the sizes `|f|` there, ascending: its grid at its level and its two
/// points off the grid.
fn points(panel: &Panel) -> Vec<(f64, f64)> {
    let n = 1usize << panel.level;
    let (a, b) = (panel.a, panel.b);
    let (mid, half) = (a + 0.5 * (b - a), 0.5 * (b - a));
    let mut out: Vec<(f64, f64)> = panel
        .values
        .iter()
        .enumerate()
        .map(|(k, v)| (grid_point(a, b, k, n), v.abs()))
        .chain(
            SAMPLES
                .iter()
                .zip(&panel.samples)
                .map(|(&u, v)| (mid + half * u, v.abs())),
        )
        .collect();
    out.sort_by(|p, q| p.0.total_cmp(&q.0));
    out
}

/// Whether the points next to the witness — the last at or before it and the first after it —
/// see the integrand at least a tenth ([`EVIDENCE`]) as large as the witness saw it.
fn explained(points: &[(f64, f64)], w: &Witness) -> bool {
    let i = points.partition_point(|p| p.0 <= w.x);
    [i.checked_sub(1), (i < points.len()).then_some(i)]
        .into_iter()
        .flatten()
        .any(|k| EVIDENCE * points[k].1 >= w.value)
}

/// The samples of a panel about to be bisected, each with its share of the panel's `∫|f|`: the
/// grid of its level by its weights, a point off the grid by the spacing of the grid there.
fn witnesses(panel: &Panel, w: &Weights) -> Vec<Witness> {
    let n = 1usize << panel.level;
    let (a, b) = (panel.a, panel.b);
    let (mid, half) = (a + 0.5 * (b - a), 0.5 * (b - a));
    let weights = &w.by_level[panel.level as usize];
    let grid = panel.values.iter().enumerate().map(|(k, v)| Witness {
        x: grid_point(a, b, k, n),
        value: v.abs(),
        mass: weights[k] * v.abs() * half.abs(),
    });
    let off = SAMPLES.iter().zip(&panel.samples).map(|(&u, v)| Witness {
        x: mid + half * u,
        value: v.abs(),
        mass: PI / n as f64 * (1.0 - u * u).sqrt() * v.abs() * half.abs(),
    });
    grid.chain(off)
        .chain(panel.witnesses.iter().copied())
        .collect()
}

/// The evidence a bisected panel leaves to its half on `[a, b]`: of its samples there (`seen`,
/// its own witnesses among them), the heaviest [`WITNESSES`] the points of the half do not
/// account for.
fn evidence(seen: &[Witness], half: &Panel) -> Vec<Witness> {
    let points = points(half);
    let mut found: Vec<Witness> = seen
        .iter()
        .filter(|w| half.a <= w.x && w.x <= half.b && !explained(&points, w))
        .copied()
        .collect();
    found.sort_by(|p, q| q.mass.total_cmp(&p.mass));
    found.truncate(WITNESSES);
    found
}

/// The largest error first, in the queue of the leaves; an entry whose error is no longer the
/// leaf's (it was refined since) is passed over.
struct ByError(f64, usize);

impl PartialEq for ByError {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}
impl Eq for ByError {}
impl PartialOrd for ByError {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByError {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The estimates of `panel` at its level and the two below, and its error estimate.
fn assess<T: Real>(panel: &mut Panel, w: &Weights, s: &Settings) {
    let half = 0.5 * (panel.b - panel.a);
    let m = panel.level;
    let estimate = |j: u32| -> (f64, f64) {
        let stride = 1usize << (m - j);
        let wj = &w.by_level[j as usize];
        let (mut sum, mut abs) = (0.0, 0.0);
        for (k, &wk) in wj.iter().enumerate() {
            let v = panel.values[k * stride];
            sum += wk * v;
            abs += wk * v.abs();
        }
        (half * sum, half.abs() * abs)
    };
    let (fine, resabs) = estimate(m);
    let (mid, _) = estimate(m - 1);
    let (coarse, _) = estimate(m - 2);
    let d1 = (fine - mid).abs();
    let d0 = (mid - coarse).abs();
    // Two successive agreements: the larger of the last two differences, so that one level
    // agreeing with the one below by the accident of aliasing (T₁₀₀ is T₂₈ on both the 33 and
    // the 65 points) is not taken for convergence; and where they fall slowly, by a ratio of a
    // half or more, the geometric tail r/(1 − r) they predict.
    let mut error = d1.max(d0);
    if d0 > 0.0 {
        let r = d1 / d0;
        if (0.5..1.0).contains(&r) {
            error = error.max(d1 * r / (1.0 - r));
        }
    }
    error = error.max(50.0 * s.eps * resabs);
    // The sample test: the interpolant of the level against the integrand off the grid.
    let n = 1usize << m;
    let mut size = panel.values.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut miss = 0.0f64;
    for (&u, &v) in SAMPLES.iter().zip(&panel.samples) {
        size = size.max(v.abs());
        miss = miss.max((interpolant(&panel.values, n, u) - v).abs());
    }
    if miss > s.eps.sqrt() * size {
        error = error.max(miss * (panel.b - panel.a).abs());
    }
    // The rounding of the points, which every level of the panel shares and no difference of its
    // levels sees: a point off where the rule has it by `δx` moves its term by `wₖ f′(x) δx`, and
    // `wₖ` is the spacing of the points, so the sum moves by about `Σ δx |fₖ − fₖ₋₁|` over
    // neighbouring points (as in `tanh_sinh`, which says why the two parts are counted as they
    // are): a point is formed in `f64` as `mid + half·cos(kπ/N)`, up to about `ε max(|a|, |b|)`
    // off, in full; and an `f32` integrand is called at it rounded to `f32`, by an amount known
    // exactly, as four standard deviations of a sum of independent errors. A peak of width 0.06
    // about 101 came out 7.5·10⁻¹⁵ off with the estimate 1.3·10⁻¹⁵ before this term (review
    // round 2).
    let formed = f64::EPSILON * panel.a.abs().max(panel.b.abs());
    let rounded = |k: usize| {
        let x = grid_point(panel.a, panel.b, k, n);
        (T::from_f64(x).to_f64() - x).abs()
    };
    let (mut moved, mut scatter) = (0.0f64, 0.0f64);
    for (k, v) in panel.values.windows(2).enumerate() {
        let change = (v[1] - v[0]).abs();
        let at = if v[1].abs() >= v[0].abs() { k + 1 } else { k };
        moved += formed * change;
        scatter += (rounded(at) * change).powi(2);
    }
    error += moved + 4.0 * scatter.sqrt();
    if !fine.is_finite() {
        error = f64::INFINITY;
    }
    // The evidence of the features the points of the panel have not found yet.
    if !panel.witnesses.is_empty() {
        let points = points(panel);
        panel.witnesses.retain(|w| !explained(&points, w));
        error = error.max(panel.witnesses.iter().map(|w| w.mass).sum());
    }
    panel.value = fine;
    panel.error = error;
    panel.resabs = resabs;
}

/// The interpolant through the values at the `N + 1` points `cos(kπ/N)` (`k = 0` at 1), at `u`
/// in `[-1, 1]`: the barycentric formula of the second kind for these points, whose weights are
/// `(−1)^k`, halved at the ends (Salzer 1972), which is stable on the whole interval (Higham
/// 2004).
fn interpolant(values: &[f64], n: usize, u: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (k, &v) in values.iter().enumerate() {
        let d = u - cosine(k, n);
        if d == 0.0 {
            return v;
        }
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let c = if k == 0 || k == n { 0.5 * sign } else { sign } / d;
        num += c * v;
        den += c;
    }
    num / den
}

/// The values of `f` at the two points off the grid of the panel on `[a, b]`.
fn off_grid<T: Real, F: FnMut(T) -> T + ?Sized>(
    f: &mut Integrand<'_, T, F>,
    a: f64,
    b: f64,
) -> Result<[f64; 2], f64> {
    let (mid, half) = (a + 0.5 * (b - a), 0.5 * (b - a));
    let mut out = [0.0; 2];
    for (slot, &u) in out.iter_mut().zip(&SAMPLES) {
        let v = f.at(mid + half * u);
        if !v.is_finite() {
            return Err(v);
        }
        *slot = v;
    }
    Ok(out)
}

/// The values of `f` at the points of level `m` on `[a, b]` that `known` (the values at the
/// points of the ends and the middle, or of the level below) does not hold.
fn evaluate<T: Real, F: FnMut(T) -> T + ?Sized>(
    f: &mut Integrand<'_, T, F>,
    a: f64,
    b: f64,
    m: u32,
    known: Known,
) -> Result<Vec<f64>, f64> {
    let n = 1usize << m;
    let mut values = vec![0.0; n + 1];
    let at = |k: usize, f: &mut Integrand<'_, T, F>| -> Result<f64, f64> {
        let v = f.at(grid_point(a, b, k, n));
        if v.is_finite() { Ok(v) } else { Err(v) }
    };
    match known {
        Known::Ends {
            right,
            middle,
            left,
        } => {
            for (k, slot) in values.iter_mut().enumerate() {
                let known = if k == 0 {
                    right
                } else if k == n {
                    left
                } else if 2 * k == n {
                    middle
                } else {
                    None
                };
                *slot = match known {
                    Some(v) => v,
                    None => at(k, f)?,
                };
            }
        }
        Known::Coarser(old) => {
            for (k, slot) in values.iter_mut().enumerate() {
                *slot = if k % 2 == 0 { old[k / 2] } else { at(k, f)? };
            }
        }
    }
    Ok(values)
}

/// What a new set of values can take from values already had.
enum Known<'a> {
    /// The values at the right end, the middle and the left end, where known.
    Ends {
        right: Option<f64>,
        middle: Option<f64>,
        left: Option<f64>,
    },
    /// All the values of the level below.
    Coarser(&'a [f64]),
}

/// Brings the best covers up to date from panel `i` to the root: a panel's is its halves' sum where
/// their errors add up to less than its own, else its own estimate.
fn update(panels: &mut [Panel], mut i: usize) {
    loop {
        let p = &panels[i];
        let cover = match p.halves {
            None => (p.value, p.error, p.resabs),
            Some((l, r)) => {
                let ((lv, le, la), (rv, re, ra)) = (panels[l].cover, panels[r].cover);
                if le + re <= p.error {
                    (lv + rv, le + re, la + ra)
                } else {
                    (p.value, p.error, p.resabs)
                }
            }
        };
        panels[i].cover = cover;
        match panels[i].parent {
            Some(up) => i = up,
            None => return,
        }
    }
}

/// Whether a panel may be bisected: its halves must be wide enough for the 257 points of a half to
/// be distinct in the integrand's precision (the closest two are `(1 − cos(π/256))` of the
/// half-width apart, near `7.5·10⁻⁵`).
fn divisible<T: Real>(a: f64, b: f64) -> bool {
    let big = a.abs().max(b.abs());
    let ulp = (big * T::EPSILON.to_f64()).max(T::MIN_POSITIVE.to_f64());
    0.25 * (b - a) > 65_536.0 * ulp
}

/// Clenshaw–Curtis on the finite `[a, b]`; `None` if the budget does not allow the first panel.
pub(super) fn integrate<T: Real, F: FnMut(T) -> T + ?Sized>(
    f: &mut Integrand<'_, T, F>,
    a: f64,
    b: f64,
    s: &Settings,
) -> Option<Outcome> {
    // The first panel's points and its two off the grid.
    let first = (1usize << FIRST) + 1 + SAMPLES.len();
    if s.max_evaluations < first {
        return None;
    }
    let w = Weights::get();
    let unknown = Known::Ends {
        right: None,
        middle: None,
        left: None,
    };
    let values = match evaluate(f, a, b, FIRST, unknown) {
        Ok(v) => v,
        Err(v) => return Some(Outcome::failed(v)),
    };
    let samples = match off_grid(f, a, b) {
        Ok(v) => v,
        Err(v) => return Some(Outcome::failed(v)),
    };
    let mut root = Panel {
        a,
        b,
        level: FIRST,
        values,
        samples,
        value: 0.0,
        error: 0.0,
        resabs: 0.0,
        halves: None,
        parent: None,
        cover: (0.0, 0.0, 0.0),
        witnesses: Vec::new(),
    };
    assess::<T>(&mut root, w, s);
    // An estimate that overflows from finite values: no refinement brings it back.
    if !root.value.is_finite() {
        return Some(Outcome::failed(root.value));
    }
    let mut panels = vec![root];
    update(&mut panels, 0);
    let mut queue = BinaryHeap::new();
    queue.push(ByError(panels[0].error, 0));
    let mut leaves = 1usize;
    loop {
        let (value, error, absolute) = panels[0].cover;
        // As in QAGS, an error estimate a tenth of the estimate or more is not accepted unless
        // it is at the round-off of the sums: the points have grazed a feature, not resolved it.
        if s.met(error, value) && resolved(error, value, absolute, s.eps) {
            return Some(Outcome {
                value,
                error,
                converged: true,
            });
        }
        // The leaf of the largest error.
        let worst = loop {
            let ByError(e, i) = queue.pop().expect("the tree has a leaf");
            if panels[i].halves.is_none() && e.total_cmp(&panels[i].error).is_eq() {
                break i;
            }
        };
        let p = &panels[worst];
        let (pa, pb, level) = (p.a, p.b, p.level);
        let stop = Outcome {
            value,
            error,
            converged: false,
        };
        if level < LAST {
            let n = 1usize << level;
            if f.evaluations + n > s.max_evaluations {
                return Some(stop);
            }
            let old = std::mem::take(&mut panels[worst].values);
            let values = match evaluate(f, pa, pb, level + 1, Known::Coarser(&old)) {
                Ok(v) => v,
                Err(v) => return Some(Outcome::failed(v)),
            };
            let p = &mut panels[worst];
            p.values = values;
            p.level = level + 1;
            assess::<T>(p, w, s);
            if !p.value.is_finite() {
                return Some(Outcome::failed(p.value));
            }
            queue.push(ByError(p.error, worst));
            update(&mut panels, worst);
        } else {
            let cost = 2 * (1usize << FIRST) - 1 + 2 * SAMPLES.len();
            if leaves + 1 > s.max_subintervals
                || f.evaluations + cost > s.max_evaluations
                || !divisible::<T>(pa, pb)
            {
                return Some(stop);
            }
            let li = panels.len();
            let n = 1usize << level;
            let seen = witnesses(&panels[worst], w);
            let old = std::mem::take(&mut panels[worst].values);
            let (right, middle, left) = (old[0], old[n / 2], old[n]);
            let mid = pa + 0.5 * (pb - pa);
            let left_values = Known::Ends {
                right: Some(middle),
                middle: None,
                left: Some(left),
            };
            let right_values = Known::Ends {
                right: Some(right),
                middle: None,
                left: Some(middle),
            };
            let lv = match evaluate(f, pa, mid, FIRST, left_values) {
                Ok(v) => v,
                Err(v) => return Some(Outcome::failed(v)),
            };
            let rv = match evaluate(f, mid, pb, FIRST, right_values) {
                Ok(v) => v,
                Err(v) => return Some(Outcome::failed(v)),
            };
            let (ls, rs) = match (off_grid(f, pa, mid), off_grid(f, mid, pb)) {
                (Ok(l), Ok(r)) => (l, r),
                (Err(v), _) | (_, Err(v)) => return Some(Outcome::failed(v)),
            };
            let mut halves =
                [(pa, mid, lv, ls), (mid, pb, rv, rs)].map(|(a, b, values, samples)| Panel {
                    a,
                    b,
                    level: FIRST,
                    values,
                    samples,
                    value: 0.0,
                    error: 0.0,
                    resabs: 0.0,
                    halves: None,
                    parent: Some(worst),
                    cover: (0.0, 0.0, 0.0),
                    witnesses: Vec::new(),
                });
            for h in &mut halves {
                h.witnesses = evidence(&seen, h);
                assess::<T>(h, w, s);
                if !h.value.is_finite() {
                    return Some(Outcome::failed(h.value));
                }
                h.cover = (h.value, h.error, h.resabs);
            }
            let [l, r] = halves;
            queue.push(ByError(l.error, li));
            queue.push(ByError(r.error, li + 1));
            panels.push(l);
            panels.push(r);
            panels[worst].halves = Some((li, li + 1));
            update(&mut panels, worst);
            leaves += 1;
        }
    }
}
