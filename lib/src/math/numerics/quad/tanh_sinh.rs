//! The double exponential rules of Takahasi and Mori (1974): the trapezoidal rule in `t` after
//! the substitution `x = c + h·tanh((π/2) sinh t)` (tanh-sinh, a finite interval),
//! `x = a ± e^((π/2) sinh t)` (exp-sinh, a half-line) or `x = sinh((π/2) sinh t)` (sinh-sinh,
//! the line).
//!
//! The nodes `t = kh` lie on two sides of the centre `t = 0`. A side towards a finite end is
//! summed out to the last node that is distinct from the end in the precision of the integrand:
//! each node is formed from its distance to that end, `h·2q/(1 + q)` with `q = e^(−2u)`,
//! `u = (π/2) sinh |t|`, which is accurate however small, so that an end at 0 is approached to
//! the smallest normal number and a nonzero end to within an ulp; the sum never stops on the size
//! of the integrand. A side towards an infinite end is summed until its terms have died away. The
//! step halves from 1, and each level evaluates only the nodes between those of the level before,
//! the budget for them set aside before the level begins. The error estimate holds, besides the
//! differences of the levels, the mass beyond the outermost nodes, the round-off of the sums and
//! what the rounding of the nodes could move them by.

use super::kronrod::resolved;
use super::{Integrand, Outcome, Range, Settings};
use crate::math::numerics::Real;
use core::f64::consts::FRAC_PI_2;

/// The most levels: a step of `2^−20`, far below what any integrand needs before the budget of
/// evaluations runs out.
const MAX_LEVEL: u32 = 20;

/// The substitution.
#[derive(Clone, Copy)]
enum Map {
    /// tanh-sinh on `[a, b]`: centre `c`, half-width `h`.
    Finite { a: f64, b: f64, h: f64 },
    /// exp-sinh on `[a, ∞)` (`sign = 1`) or `(−∞, a]` (`sign = −1`): `x = a + sign·e^u`.
    Half { a: f64, sign: f64 },
    /// sinh-sinh on the line: `x = sinh u`.
    Whole,
}

/// A node of the rule: the point where the integrand is evaluated and the factor `x′(t)` its
/// value is multiplied by; `None` past the end of the side (the point would round onto a finite
/// end, or leave the range of the type).
fn node<T: Real>(map: Map, t: f64) -> Option<(f64, f64)> {
    let u = FRAC_PI_2 * t.sinh();
    let max = T::MAX.to_f64();
    let tiny = T::MIN_POSITIVE.to_f64();
    let round = |x: f64| T::from_f64(x).to_f64();
    match map {
        Map::Finite { a, b, h } => {
            if t == 0.0 {
                return Some((a + h, h * FRAC_PI_2));
            }
            let q = (-2.0 * u.abs()).exp();
            let d = h * 2.0 * q / (1.0 + q);
            let w = h * FRAC_PI_2 * t.cosh() * 4.0 * q / ((1.0 + q) * (1.0 + q));
            let (x, end) = if t > 0.0 { (b - d, b) } else { (a + d, a) };
            (d >= tiny && round(x) != end && w > 0.0).then_some((x, w))
        }
        Map::Half { a, sign } => {
            let e = u.exp();
            let w = e * FRAC_PI_2 * t.cosh();
            let x = a + sign * e;
            let inside = if t < 0.0 {
                e >= tiny && round(x) != a
            } else {
                x.abs() <= max
            };
            (inside && w.is_finite()).then_some((x, w))
        }
        Map::Whole => {
            let x = u.sinh();
            let w = u.cosh() * FRAC_PI_2 * t.cosh();
            (x.abs() <= max && w.is_finite()).then_some((x, w))
        }
    }
}

/// Whether the side of `sign` runs to an infinite end, where it stops on the size of its terms.
fn infinite(map: Map, sign: f64) -> bool {
    match map {
        Map::Finite { .. } => false,
        Map::Half { .. } => sign > 0.0,
        Map::Whole => true,
    }
}

/// One side of the centre.
struct Side {
    sign: f64,
    infinite: bool,
    /// The largest `|t|` evaluated, and whether the side has reached its end.
    reach: f64,
    ended: bool,
    /// The two outermost nodes with distinct points, `(|t|, s, f)`: `s` the distance of the point
    /// the integrand was evaluated at to the finite end (as the type rounds the point), or its
    /// distance to the origin of the map on an infinite side, and `f` the value there.
    outer: [(f64, f64, f64); 2],
    count: usize,
}

impl Side {
    fn new(map: Map, sign: f64) -> Self {
        Self {
            sign,
            infinite: infinite(map, sign),
            reach: 0.0,
            ended: false,
            outer: [(0.0, 0.0, 0.0); 2],
            count: 0,
        }
    }

    fn record(&mut self, t: f64, s: f64, f: f64) {
        if t > self.outer[0].0 {
            if s != self.outer[0].1 {
                self.outer[1] = self.outer[0];
                self.count += 1;
            }
            self.outer[0] = (t, s, f);
        } else if t > self.outer[1].0 && s != self.outer[0].1 {
            self.outer[1] = (t, s, f);
        }
        self.count = self.count.max(1);
    }

    /// An estimate of the integral beyond the outermost node, from a power law through the two
    /// outermost values: towards a finite end `f ≈ C sᵖ` in the distance `s`, whose integral
    /// over `(0, s₁)` is `f₁ s₁/(p + 1)`; towards infinity `f ≈ C s^(−q)`, whose integral over
    /// `(s₁, ∞)` is `f₁ s₁/(q − 1)`. Twice that, and infinite where the power law is not
    /// integrable. The double exponential decay of a regular integrand makes this negligible; an
    /// integrand singular at a nonzero end, which the nodes can approach only to an ulp, gets the
    /// mass it hides there, and an integrand decaying too slowly an infinite estimate.
    fn beyond(&self) -> f64 {
        let ((_, s1, f1), (_, s2, f2)) = (self.outer[0], self.outer[1]);
        if f1 == 0.0 {
            return 0.0;
        }
        if self.count < 2 {
            return f64::INFINITY;
        }
        // The exponent of the power law, 0 where the two values do not share a sign.
        let slope = if f2 != 0.0 && f1.signum() == f2.signum() {
            (f1 / f2).abs().ln() / (s1 / s2).ln()
        } else {
            0.0
        };
        let mass = if self.infinite {
            let q = -slope;
            if q <= 1.0 {
                return f64::INFINITY;
            }
            f1.abs() * s1 / (q - 1.0)
        } else {
            let p = slope;
            if p <= -1.0 {
                return f64::INFINITY;
            }
            f1.abs() * s1 / (p + 1.0)
        };
        2.0 * mass
    }
}

/// The distance-like variable of a point on a side, for [`Side::beyond`]: on a finite side the
/// distance of the point as the integrand's type rounds it to the end, on an infinite side its
/// distance to the origin of the map.
fn spread<T: Real>(map: Map, t: f64, x: f64) -> f64 {
    let x = T::from_f64(x).to_f64();
    match map {
        Map::Finite { a, b, .. } => {
            if t > 0.0 {
                b - x
            } else {
                x - a
            }
        }
        Map::Half { a, .. } => (x - a).abs(),
        Map::Whole => x.abs(),
    }
}

/// The error of the newest level, from the differences between the levels: `d` between it and
/// the level before, `d1` the difference before that, `m1` the larger of the two differences
/// before this one (`d1` and the one before it), all infinite where the levels are too few to
/// have them, and `floor` the round-off of the sums.
///
/// While the rule converges double exponentially (`fast`) — the correct digits about doubling a
/// level, as they do for an integrand analytic inside the interval, whatever it does at the ends
/// — the difference to the level before is the error of that level, and far more than the error
/// of this one: the estimate is `d`. Where the convergence is slower (a kink, a jump or a
/// singularity inside the interval, where the trapezoidal rule converges only algebraically, and
/// erratically, as the bad point moves against the nodes), a single difference can be small by
/// accident — on `√|x − 0.3|` one level in four agrees with the level before ten times better
/// than its error — so the estimate is the larger of the last two differences, times the sum
/// `r/(1 − r)` of the geometric tail where their ratio `r` to the two before is a half or more,
/// and infinite where the differences are not falling.
fn discretisation(d: f64, d1: f64, m1: f64, floor: f64, fast: bool) -> f64 {
    if d <= floor || fast {
        return d;
    }
    let m = d.max(d1);
    if !m1.is_finite() {
        return m;
    }
    let r = m / m1;
    if r >= 1.0 {
        return f64::INFINITY;
    }
    m * (r / (1.0 - r)).max(1.0)
}

/// Whether the difference `d` fell from the difference `d1` before it as the double exponential
/// convergence makes it fall, to at most the 1.5th power of `d1` relative to `scale`, the
/// integral of `|f|`: the regime in which a single difference is a safe error estimate, once it
/// has held for two levels in a row.
///
/// The regime is one of correct digits doubling, so it is not entered from a level that had none:
/// `d1` must be below a tenth of `scale` (the threshold of [`resolved`]). Without that, any fall
/// from a difference as large as the integral passed, and a narrow Lorentzian on `[1, ∞)` was
/// accepted 29% off at the error 9·10⁻⁷ (review round 2): one node of level 9 fell near its
/// centre, the levels after it halved that node's share, and the new nodes of level 13 happened
/// to make up what level 12 lost — falls of `3.9·10⁻⁴`, `1.7·10⁻⁴` and `9·10⁻⁷` on an integral
/// of `2.6·10⁻⁴`, the first two no fall of a converging rule at all.
fn falls_fast(d: f64, d1: f64, scale: f64) -> bool {
    if !d1.is_finite() {
        return false;
    }
    let before = d1 / scale;
    before < 1.0 && d / scale <= before * before.sqrt()
}

/// The rule on `range`.
pub(super) fn integrate<T: Real>(
    f: &mut Integrand<'_, T>,
    range: Range,
    s: &Settings,
) -> Option<Outcome> {
    let map = match range {
        Range::Finite(a, b) => Map::Finite {
            a,
            b,
            h: 0.5 * (b - a),
        },
        Range::Upper(a) => Map::Half { a, sign: 1.0 },
        Range::Lower(b) => Map::Half { a: b, sign: -1.0 },
        Range::Whole => Map::Whole,
        Range::Empty => unreachable!("the empty range is answered before"),
    };
    let mut sides = [Side::new(map, 1.0), Side::new(map, -1.0)];
    let (x0, w0) = node::<T>(map, 0.0).expect("the centre is a node");
    // An interval with no number of the type at its centre, only its ends (`[1, 1 + ε]`), or a
    // half-line whose end is so large that `a ± 1` rounds onto it: there is no point to
    // evaluate the integrand at that is not an end, and nothing is evaluated.
    let centre = T::from_f64(x0).to_f64();
    let on_an_end = match map {
        Map::Finite { a, b, .. } => centre == a || centre == b,
        Map::Half { a, .. } => centre == a,
        Map::Whole => false,
    };
    if on_an_end {
        return Some(Outcome::failed(0.0));
    }
    let f0 = f.at(x0);
    let g0 = w0 * f0;
    if !g0.is_finite() {
        return Some(Outcome::failed(g0));
    }
    // A half-width below the smallest normal number of the type: every node but the centre is
    // nearer an end than that (`h·2q/(1 + q) < h`), and none can be formed. One point is no
    // estimate of an error.
    if matches!(map, Map::Finite { h, .. } if h < T::MIN_POSITIVE.to_f64()) {
        return Some(Outcome::failed(g0));
    }
    // The sum of the level before, and the sum of the absolute values of all the terms, which
    // times the step is the integral of |f|.
    let mut previous = 0.0;
    let mut last_difference = f64::INFINITY;
    let mut last_envelope = f64::INFINITY;
    let mut fell_fast = false;
    let mut absolute = g0.abs();
    let mut best = Outcome {
        value: g0,
        error: f64::INFINITY,
        converged: false,
    };
    for level in 0..=MAX_LEVEL {
        let h = 0.5f64.powi(level as i32);
        // The nodes between those of the level before, `t = h, 3h, …` below the reach of each
        // side, which the budget must allow before the level is begun; and while one side is
        // extended past its reach, the other side's are held in reserve.
        let between = |side: &Side| {
            let (mut count, mut t) = (0usize, h);
            while level > 0 && t < side.reach {
                count += 1;
                t += 2.0 * h;
            }
            count
        };
        let inner = [between(&sides[0]), between(&sides[1])];
        if f.evaluations + inner[0] + inner[1] > s.max_evaluations {
            break;
        }
        let mut level_sum = if level == 0 { g0 } else { 0.0 };
        // How far the rounding of the points could move the sum. A node at `xᵢ` off where the rule
        // has it by `δxᵢ` shifts its term `h wᵢ f(xᵢ)` by `h wᵢ f′(xᵢ) δxᵢ`, and `h wᵢ` is the
        // spacing of the nodes, so the sum moves by about `Σ δxᵢ |fᵢ − fᵢ₋₁|` over neighbouring
        // nodes — here the nodes of this level on each side, in order from the centre, `2h`
        // apart. Every level shares the rounding of the points it shares, and no difference of
        // levels sees it: a Gaussian of width 0.061 about 21.4 came out 1.5·10⁻¹⁵ off with the
        // estimate 4.3·10⁻¹⁶ before this term, one of width 0.001 about −0.0008 on `[−1, 1]`
        // 1.8·10⁻¹⁷ off with 5.5·10⁻¹⁸ (review round 2). The node is formed in `f64` from the end
        // it approaches, `x = b − d`, up to about `ε(|x| + d)` off, an error that the cases above
        // show to be systematic: it counts in full (`moved`). An `f32` integrand is then called
        // at the node rounded to `f32`, by `|δxᵢ|` known exactly and as likely up as down from one
        // node to the next: that counts as a sum of independent errors, four standard deviations
        // of it (`scatter`), where in full it would take the whole `f32` tolerance from
        // `cos 200x`, whose true error from it is a hundredth of that.
        let mut moved = 0.0f64;
        let mut scatter = 0.0f64;
        let scale = |t: f64, x: f64| {
            let point = T::from_f64(x).to_f64();
            (
                f64::EPSILON * (point.abs() + spread::<T>(map, t, x)),
                (point - x).abs(),
            )
        };
        for (i, side) in sides.iter_mut().enumerate() {
            let reserve = if i == 0 { inner[1] } else { 0 };
            let mut last = (scale(0.0, x0), f0);
            let mut shift = |t: f64, x: f64, v: f64| {
                let x = scale(t, x);
                // The change between two nodes is the slope about the one where the integrand is
                // larger: on the stretched tail of an infinite side neighbours can be `10¹⁰` and
                // `10⁴⁵`, and the far node's rounding has nothing to do with the near one's value.
                let (formed, rounded) = if v.abs() >= last.1.abs() { x } else { last.0 };
                let change = (v - last.1).abs();
                moved += formed * change;
                scatter += (rounded * change).powi(2);
                last = (x, v);
            };
            if level > 0 {
                let mut t = h;
                while t < side.reach {
                    let (x, w) = node::<T>(map, side.sign * t)
                        .expect("a node inside the reach of its side is a node");
                    let v = f.at(x);
                    let g = w * v;
                    if !g.is_finite() {
                        return Some(Outcome::failed(g));
                    }
                    shift(side.sign * t, x, v);
                    level_sum += g;
                    absolute += g.abs();
                    side.record(t, spread::<T>(map, side.sign * t, x), g / w);
                    t += 2.0 * h;
                }
            }
            // Past the reach, at this level's step: a finite side as far as its nodes stay
            // distinct from the end, which a finer step can take further; an infinite side until
            // its terms are negligible beside the integral of |f|, two in a row on the first
            // level and one after it.
            if side.infinite && side.ended {
                continue;
            }
            let mut negligible = 0;
            let mut t = if level == 0 { 1.0 } else { side.reach + h };
            while f.evaluations + reserve < s.max_evaluations {
                let Some((x, w)) = node::<T>(map, side.sign * t) else {
                    side.ended = true;
                    break;
                };
                let v = f.at(x);
                let g = w * v;
                if !g.is_finite() {
                    return Some(Outcome::failed(g));
                }
                shift(side.sign * t, x, v);
                level_sum += g;
                absolute += g.abs();
                side.record(t, spread::<T>(map, side.sign * t, x), g / w);
                side.reach = t;
                if side.infinite {
                    if g.abs() <= 0.25 * s.eps * absolute {
                        negligible += 1;
                        if negligible >= if level == 0 { 2 } else { 1 } {
                            side.ended = level > 0;
                            break;
                        }
                    } else {
                        negligible = 0;
                    }
                }
                t += h;
            }
        }
        let sum = if level == 0 {
            level_sum
        } else {
            0.5 * previous + h * level_sum
        };
        // A sum that overflows from finite terms: no finer step brings it back.
        if !sum.is_finite() {
            return Some(Outcome::failed(sum));
        }
        if level > 0 {
            let difference = (sum - previous).abs();
            let tails = sides[0].beyond() + sides[1].beyond();
            let roundoff = 10.0 * s.eps * h * absolute;
            // The rounding of the points the integrand is evaluated at, which every level shares
            // and no difference of levels sees (see `moved`): in the error, but not a floor below
            // which a difference of levels is taken for the error, which it may not be.
            let points = moved + 4.0 * scatter.sqrt();
            let envelope = difference.max(last_difference);
            // A single fast fall can be an accident of the trapezoidal rule at an inner kink: the
            // regime is trusted after two in a row.
            let falls = falls_fast(difference, last_difference, h * absolute);
            let error = discretisation(
                difference,
                last_difference,
                last_envelope,
                roundoff,
                falls && fell_fast,
            ) + tails
                + roundoff
                + points;
            fell_fast = falls;
            // No result is accepted before the third level, nor while the levels are still
            // moving apart: a level that finds mass the coarser ones missed (a peak between
            // their nodes) is not the end of the refinement, however small both are. Nor, as in
            // QAGS, with an error estimate a tenth of the estimate or more, unless that error is
            // at the round-off of the sums: the nodes have grazed a feature, not resolved it.
            let settling = difference <= last_difference || difference <= roundoff;
            last_envelope = envelope;
            best = Outcome {
                value: sum,
                error,
                converged: level >= 2
                    && settling
                    && s.met(error, sum)
                    && resolved(error, sum, h * absolute, s.eps),
            };
            if best.converged {
                return Some(best);
            }
            // Once the error is the mass beyond the nodes, which no finer step reaches, the
            // refinement can stop.
            if level >= 3 && tails > s.bound(sum) && difference <= 0.1 * tails {
                return Some(best);
            }
            last_difference = difference;
        } else {
            best.value = sum;
        }
        previous = sum;
        if f.evaluations >= s.max_evaluations {
            break;
        }
    }
    Some(best)
}
