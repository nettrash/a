//! Adaptive cubature over a box: Genz and Malik's embedded rules (A. C. Genz and A. A. Malik, "An
//! adaptive algorithm for numerical integration over an n-dimensional rectangular region",
//! J. Comput. Appl. Math. 6, 1980), with the error estimate from null rules of Berntsen, Espelid
//! and Genz's DCUHRE (ACM TOMS 17, 1991).
//!
//! On the cube `[-1, 1]^d` the rule of degree 7 takes the centre, the `2d` points `±λ₂eᵢ` and
//! the `2d` points `±λ₃eᵢ` on the axes, the `2d(d − 1)` points `±λ₄eᵢ ± λ₄eⱼ` in the planes of
//! two axes and the `2^d` points `(±λ₅, …, ±λ₅)`, with `λ₂² = 9/70`, `λ₃² = λ₄² = 9/10` and
//! `λ₅² = 9/19`; the rule of degree 5 takes the same points but the last `2^d`. The weights below
//! are Genz and Malik's; the test `the_genz_malik_rules_have_the_degrees_they_claim` checks that
//! they integrate the monomials up to their degrees exactly, and no further.
//!
//! The error estimate of a box is built from three null rules on the same points, of degrees 5
//! (`I₇ − I₅`, Genz and Malik's estimate), 3 and 1, as DCUHRE builds its own: where they fall
//! off with the degree the box resolves the integrand, and the estimate is ten times
//! `|I₇ − I₅|`; where they do not, ten times the largest of them (see `null_rule_error`). The
//! fourth divided difference of the integrand along each axis, which the points on the axes give
//! for nothing, says along which axis the box is to be halved. Each box is evaluated once, when it
//! is made; the boxes wait in a heap by their errors, and the totals are running sums, summed
//! afresh by compensated summation before the result is accepted — which it is only with an
//! error below a tenth of it, or at the round-off of the sums, as in the one-dimensional rules.
//!
//! The estimate assumes that the integrand is smooth on the scale of each box — a limit this
//! rule shares with every rule of its kind, for no finite set of points certifies a feature
//! narrower than their spacing. A kink or a jump that runs through the band between a box's
//! outermost points (`λ₃ = 0.949` of the half-width) and its faces is not seen by any of its
//! points, and its error then goes unseen too; such an integrand is integrated reliably by
//! splitting the box at the discontinuity. A narrow peak can be missed by the points of the
//! first boxes altogether; its tail beyond a face between two boxes is lost where the box across
//! the face has no point near the peak; and a box many widths of the peak long, whose points see
//! it along one line only, overestimates it, for its rule spreads the values seen there over the
//! spacing of its points, and its null rules, combinations of the same values, can agree with
//! it. The rates are in the documentation of [`cubature`](super::cubature); a box about the peak
//! integrated apart from the rest is the remedy.

use super::kronrod::{neumaier, resolved};
use super::{QuadOptions, QuadResult, Settings, in_type};
use crate::math::numerics::Real;
use std::collections::BinaryHeap;

/// One box: its centre and half-widths, the estimate and the error of the rule on it, and the
/// axis to halve it along.
struct Cell {
    centre: Vec<f64>,
    half: Vec<f64>,
    value: f64,
    error: f64,
    /// `∫|f|` over the box by the rule.
    resabs: f64,
    axis: usize,
}

/// The largest error first, in the queue of the boxes.
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

/// The weights of the two rules in `d` dimensions, each scaled so that it sums to 1 (the rule
/// times the volume of the box is the integral): `(w₁, …, w₅)` of degree 7 and `(w₁′, …, w₄′)`
/// of degree 5.
fn weights(d: usize) -> ([f64; 5], [f64; 4]) {
    let n = d as f64;
    let seven = [
        (12824.0 - 9120.0 * n + 400.0 * n * n) / 19683.0,
        980.0 / 6561.0,
        (1820.0 - 400.0 * n) / 19683.0,
        200.0 / 19683.0,
        6859.0 / 19683.0 / 2f64.powi(d as i32),
    ];
    let five = [
        (729.0 - 950.0 * n + 50.0 * n * n) / 729.0,
        245.0 / 486.0,
        (265.0 - 100.0 * n) / 1458.0,
        25.0 / 729.0,
    ];
    (seven, five)
}

/// The number of points of the rule of degree 7 in `d` dimensions, `2^d + 2d² + 2d + 1`, or
/// `None` where it does not fit a `usize`.
pub(super) fn points(d: usize) -> Option<usize> {
    let corners = 1usize.checked_shl(u32::try_from(d).ok()?)?;
    let rest = d.checked_mul(d)?.checked_mul(2)?.checked_add(2 * d + 1)?;
    if d >= usize::BITS as usize {
        return None;
    }
    corners.checked_add(rest)
}

/// The rule on the box of `centre` and `half`; `Err` with the value if the integrand gives one
/// that is not a number.
fn rule<T: Real>(
    f: &mut impl FnMut(&[T]) -> T,
    centre: &[f64],
    half: &[f64],
    point: &mut Vec<T>,
    eps: f64,
) -> Result<Cell, f64> {
    let d = centre.len();
    let (l2, l3, l4, l5) = (
        (9.0f64 / 70.0).sqrt(),
        (9.0f64 / 10.0).sqrt(),
        (9.0f64 / 10.0).sqrt(),
        (9.0f64 / 19.0).sqrt(),
    );
    let (w7, w5) = weights(d);
    let mut eval = |offsets: &[(usize, f64)]| -> Result<f64, f64> {
        point.clear();
        point.extend(centre.iter().map(|&c| T::from_f64(c)));
        for &(i, o) in offsets {
            point[i] = T::from_f64(centre[i] + o * half[i]);
        }
        let v = f(point).to_f64();
        if v.is_finite() { Ok(v) } else { Err(v) }
    };
    let f0 = eval(&[])?;
    let (mut s2, mut s3, mut s4, mut s5) = (0.0, 0.0, 0.0, 0.0);
    let (mut a2, mut a3, mut a4, mut a5) = (0.0, 0.0, 0.0, 0.0);
    let mut axis = 0;
    let mut largest = -1.0;
    for i in 0..d {
        let (p2, m2) = (eval(&[(i, l2)])?, eval(&[(i, -l2)])?);
        let (p3, m3) = (eval(&[(i, l3)])?, eval(&[(i, -l3)])?);
        s2 += p2 + m2;
        s3 += p3 + m3;
        a2 += p2.abs() + m2.abs();
        a3 += p3.abs() + m3.abs();
        // The fourth difference along the axis, free of the second (λ₂²/λ₃² = 1/7); the widest
        // axis among equal ones.
        let fourth = ((p2 + m2 - 2.0 * f0) - (p3 + m3 - 2.0 * f0) / 7.0).abs();
        if fourth > largest || (fourth == largest && half[i] > half[axis]) {
            largest = fourth;
            axis = i;
        }
    }
    for i in 0..d {
        for j in i + 1..d {
            for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                let v = eval(&[(i, si * l4), (j, sj * l4)])?;
                s4 += v;
                a4 += v.abs();
            }
        }
    }
    let mut offsets: Vec<(usize, f64)> = (0..d).map(|i| (i, l5)).collect();
    for mask in 0u64..(1u64 << d) {
        for (i, o) in offsets.iter_mut().enumerate() {
            o.1 = if mask >> i & 1 == 1 { -l5 } else { l5 };
        }
        let v = eval(&offsets)?;
        s5 += v;
        a5 += v.abs();
    }
    let volume: f64 = half.iter().map(|h| 2.0 * h).product();
    let seven = w7[0] * f0 + w7[1] * s2 + w7[2] * s3 + w7[3] * s4 + w7[4] * s5;
    let absolute = w7[0].abs() * f0.abs() + w7[1] * a2 + w7[2].abs() * a3 + w7[3] * a4 + w7[4] * a5;
    let value = volume * seven;
    let error = null_rule_error(d, &w7, &w5, [f0, s2, s3, s4, s5], volume)
        .max(50.0 * eps * volume * absolute);
    Ok(Cell {
        centre: centre.to_vec(),
        half: half.to_vec(),
        value,
        error,
        resabs: volume * absolute,
        axis,
    })
}

/// The error estimate of a box, from three null rules on the points of the rule — rules whose
/// weights integrate every polynomial up to a degree to zero (Berntsen, Espelid and Genz 1991):
/// `N₅ = I₇ − I₅` of degree 5, `N₃` the difference of the rules of degree 3 on the centre and
/// the points `±λ₂eᵢ` and on the centre and `±λ₃eᵢ`, and `N₁` the difference of the means over
/// those two sets of points; each applied to the integrand and scaled to the norm of the weights
/// of the rule of degree 7. `sums` are the values at the centre and the sums over the four kinds
/// of points; `w7` and `w5` the weights of the two rules.
///
/// Where the null rules fall off with their degree — by the ratio `r < ½` — the box resolves the
/// integrand, and the estimate is ten times Genz and Malik's `|I₇ − I₅|`; where they do not,
/// ten times the largest of the three. The factor ten is a margin for what a single null rule of
/// degree 5 cannot see: on the Genz families it was needed in 2 to 5 dimensions, where
/// `|I₇ − I₅|` alone was 18 to 56 times below the true error of a corner peak on a box near the
/// peak's pole, and with it no converged result of the four smooth families was beyond its
/// estimate (`tests/math/numerics/quad/cubature.rs`).
fn null_rule_error(d: usize, w7: &[f64; 5], w5: &[f64; 4], sums: [f64; 5], volume: f64) -> f64 {
    const MARGIN: f64 = 10.0;
    let nd = d as f64;
    let sizes = [
        1.0,
        2.0 * nd,
        2.0 * nd,
        2.0 * nd * (nd - 1.0),
        2f64.powi(d as i32),
    ];
    let norm = |w: &[f64; 5]| {
        w.iter()
            .zip(&sizes)
            .map(|(w, n)| n * w * w)
            .sum::<f64>()
            .sqrt()
    };
    let apply = |w: &[f64; 5]| -> f64 { w.iter().zip(&sums).map(|(w, s)| w * s).sum() };
    let n5 = [
        w7[0] - w5[0],
        w7[1] - w5[1],
        w7[2] - w5[2],
        w7[3] - w5[3],
        w7[4],
    ];
    // The two rules of degree 3, a₀f₀ + a₂S₂ and b₀f₀ + b₃S₃, have a₂ = 1/(6λ₂²) = 35/27 and
    // b₃ = 1/(6λ₃²) = 5/27, and their weights sum to 1.
    let n3 = [-60.0 * nd / 27.0, 35.0 / 27.0, -5.0 / 27.0, 0.0, 0.0];
    let n1 = [0.0, 0.5 / nd, -0.5 / nd, 0.0, 0.0];
    let basic = norm(w7);
    let scaled = |w: &[f64; 5]| volume * apply(w).abs() * basic / norm(w);
    let (e5, e3, e1) = (scaled(&n5), scaled(&n3), scaled(&n1));
    let ratio = |a: f64, b: f64| {
        if b > 0.0 {
            a / b
        } else if a > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    };
    let r = ratio(e5, e3).max(ratio(e3, e1));
    if r < 0.5 {
        MARGIN * (volume * apply(&n5)).abs()
    } else {
        MARGIN * e5.max(e3).max(e1)
    }
}

/// The cubature: see [`cubature`](super::cubature).
#[track_caller]
pub(super) fn integrate<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    assert_eq!(
        a.len(),
        b.len(),
        "the two corners of the box have different dimensions"
    );
    assert!(!a.is_empty(), "the box has no dimensions");
    let d = a.len();
    if a.iter().chain(b).any(|x| !x.to_f64().is_finite()) {
        return None;
    }
    let s = Settings::of(opts)?;
    let mut sign = 1.0;
    let mut centre = Vec::with_capacity(d);
    let mut half = Vec::with_capacity(d);
    for (&lo, &hi) in a.iter().zip(b) {
        let (lo, hi) = (lo.to_f64(), hi.to_f64());
        if lo > hi {
            sign = -sign;
        }
        let (l, h) = (lo.min(hi), lo.max(hi));
        centre.push(l + 0.5 * (h - l));
        half.push(0.5 * (h - l));
    }
    let per_box = points(d).filter(|&p| p <= s.max_evaluations)?;
    if half.contains(&0.0) {
        return Some(QuadResult {
            value: T::ZERO,
            error: T::ZERO,
            evaluations: 0,
            converged: true,
        });
    }
    if d == 1 {
        let (lo, hi) = if sign > 0.0 {
            (a[0], b[0])
        } else {
            (b[0], a[0])
        };
        let mut x = [T::ZERO];
        let r = super::quad_g7k15(
            |t: T| {
                x[0] = t;
                f(&x)
            },
            lo,
            hi,
            opts,
        )?;
        return Some(QuadResult {
            value: T::from_f64(sign) * r.value,
            ..r
        });
    }
    let mut point = Vec::with_capacity(d);
    let mut evaluations = 0usize;
    let finish = |value: f64, error: f64, evaluations: usize, converged: bool| {
        in_type(sign * value, error, evaluations, converged)
    };
    let first = match rule(&mut f, &centre, &half, &mut point, s.eps) {
        Ok(cell) if cell.value.is_finite() => cell,
        // A value of the integrand that is not a number, or an estimate that overflows.
        Ok(cell) => return Some(finish(cell.value, f64::INFINITY, per_box, false)),
        Err(v) => return Some(finish(v, f64::INFINITY, per_box, false)),
    };
    evaluations += per_box;
    // The totals are kept as running sums, and summed afresh by compensated summation whenever
    // they say the tolerance is met, and at the end.
    let (mut value, mut error) = (first.value, first.error);
    let mut cells = vec![first];
    let mut queue = BinaryHeap::new();
    queue.push(ByError(cells[0].error, 0));
    let exact = |cells: &[Cell]| {
        (
            neumaier(cells.iter().map(|c| c.value)),
            neumaier(cells.iter().map(|c| c.error)),
        )
    };
    // As in the one-dimensional rules, an estimate is accepted only once its error is below a
    // tenth of it, or at the round-off of the sums: an error estimate comparable to the estimate
    // says that the points have seen the tail of a feature, not resolved it. A Gaussian of width
    // 0.03 about (0.7, 0.3) in the unit square was accepted on the first box as `10⁻²⁰` with the
    // error `9·10⁻²⁰` at the default tolerances, for `2.8·10⁻³` (review round 2).
    let accept = |cells: &[Cell], value: f64, error: f64| {
        s.met(error, value)
            && resolved(
                error,
                value,
                neumaier(cells.iter().map(|c| c.resabs)),
                s.eps,
            )
    };
    loop {
        if s.met(error, value) {
            (value, error) = exact(&cells);
            if accept(&cells, value, error) {
                return Some(finish(value, error, evaluations, true));
            }
        }
        let room =
            cells.len() < s.max_subintervals && evaluations + 2 * per_box <= s.max_evaluations;
        let ByError(_, worst) = queue.pop().expect("there is a box");
        let axis = cells[worst].axis;
        let (c, h) = (cells[worst].centre[axis], cells[worst].half[axis]);
        // A box too narrow along its axis to halve without its points running together.
        let narrow = 0.5 * h <= 64.0 * s.eps * c.abs().max(s.tiny / s.eps);
        if !room || narrow {
            let (value, error) = exact(&cells);
            return Some(finish(value, error, evaluations, false));
        }
        let mut half_child = cells[worst].half.clone();
        half_child[axis] = 0.5 * h;
        let mut children = Vec::with_capacity(2);
        for side in [-0.5, 0.5] {
            let mut centre_child = cells[worst].centre.clone();
            centre_child[axis] = c + side * h;
            match rule(&mut f, &centre_child, &half_child, &mut point, s.eps) {
                Ok(cell) if cell.value.is_finite() => children.push(cell),
                Ok(cell) => {
                    return Some(finish(
                        cell.value,
                        f64::INFINITY,
                        evaluations + per_box,
                        false,
                    ));
                }
                Err(v) => return Some(finish(v, f64::INFINITY, evaluations + per_box, false)),
            }
            evaluations += per_box;
        }
        let right = children.pop().expect("two halves");
        let left = children.pop().expect("two halves");
        value += left.value + right.value - cells[worst].value;
        error += left.error + right.error - cells[worst].error;
        queue.push(ByError(left.error, worst));
        cells[worst] = left;
        queue.push(ByError(right.error, cells.len()));
        cells.push(right);
    }
}
