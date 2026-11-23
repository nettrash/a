//! Adaptive Gauss–Kronrod quadrature as QUADPACK's QAGS does it (Piessens, de Doncker-Kapenga,
//! Überhuber and Kahaner, *QUADPACK*, 1983; de Doncker 1978), with QAGI's map of an infinite
//! interval onto `(0, 1]`.
//!
//! * The interval with the largest error estimate is bisected, the intervals held in a list
//!   sorted by their error estimates.
//! * The error estimate of an interval is QUADPACK's: the difference of the Kronrod and Gauss
//!   estimates scaled by `resasc = ∫|f − mean|` as `resasc · min(1, (200 |K − G| / resasc)^1.5)`,
//!   and never below `50 ε resabs`, `resabs = ∫|f|`, so that it does not claim more than the
//!   rounding of the values allows.
//! * Round-off is detected as QUADPACK detects it: a bisection that leaves the estimate where it
//!   was while the error does not fall, ten times, or an error that grows after bisection twenty
//!   times, ends the integration as not converged; so does an interval so narrow that its
//!   Kronrod nodes could round onto its ends.
//! * The sequence of the estimates as the bisection homes in on a difficult point is
//!   extrapolated by Wynn's epsilon algorithm, which turns the slow convergence towards an
//!   integrable end singularity into a fast one, and QUADPACK's tests decide between the
//!   extrapolated and the plain estimate and flag a divergent integral.
//!
//! Departures from QUADPACK, all towards honesty: no interval is bisected so far that a node
//! could round onto an end; the totals are summed afresh after every bisection instead of kept
//! as running sums that cancel; a value of a bisected interval that the nodes of the half it
//! lies in do not account for stays in that half's error as the evidence of a feature between
//! them, until a node beside it sees a tenth of it (up to four such values an interval, the
//! heaviest, and the centre of a bisected interval for each of its halves); the rounding of the
//! extrapolation is measured by copies of it fed perturbed
//! estimates; the floor `50 ε ∫|f|` on an interval's error is kept below `tiny/(50ε)`, as
//! `50 ε tiny`; and a result whose error estimate is a tenth of it or more is not accepted unless
//! the error is at the round-off of the sums.
//!
//! The result is converged only when its error estimate is within the tolerances.

use super::epsilon::Epsilon;
use super::{Integrand, Outcome, Settings};
use crate::math::numerics::Real;

/// A Gauss–Kronrod pair on `[-1, 1]` held as the nodes of non-negative abscissa: `(u, wk, wg)` is
/// the node pair `±u` with the Kronrod weight of one node and its Gauss weight where the Gauss
/// rule has it. The last entry is the centre.
pub(super) type Table = [(f64, f64, Option<f64>)];

// The published nodes and weights of the pairs (Piessens et al. 1983, the routines `qk15` and
// `qk21`), here as the correctly rounded `f64`s of the true values, which the test
// `the_kronrod_tables_are_the_rules_to_the_last_bit` recomputes with mpmath.
#[allow(clippy::excessive_precision)]
pub(super) const G7K15: [(f64, f64, Option<f64>); 8] = [
    (9.91455371120812612e-01, 2.29353220105292244e-02, None),
    (
        9.49107912342758486e-01,
        6.30920926299785578e-02,
        Some(1.29484966168869703e-01),
    ),
    (8.64864423359769097e-01, 1.04790010322250188e-01, None),
    (
        7.41531185599394460e-01,
        1.40653259715525919e-01,
        Some(2.79705391489276645e-01),
    ),
    (5.86087235467691148e-01, 1.69004726639267910e-01, None),
    (
        4.05845151377397184e-01,
        1.90350578064785420e-01,
        Some(3.81830050505118923e-01),
    ),
    (2.07784955007898481e-01, 2.04432940075298886e-01, None),
    (0.0, 2.09482141084727819e-01, Some(4.17959183673469403e-01)),
];

#[allow(clippy::excessive_precision)]
pub(super) const G10K21: [(f64, f64, Option<f64>); 11] = [
    (9.95657163025808090e-01, 1.16946388673718742e-02, None),
    (
        9.73906528517171743e-01,
        3.25581623079647248e-02,
        Some(6.66713443086881380e-02),
    ),
    (9.30157491355708244e-01, 5.47558965743519949e-02, None),
    (
        8.65063366688984536e-01,
        7.50396748109199568e-02,
        Some(1.49451349150580587e-01),
    ),
    (7.80817726586416905e-01, 9.31254545836976005e-02, None),
    (
        6.79409568299024436e-01,
        1.09387158802297643e-01,
        Some(2.19086362515982042e-01),
    ),
    (5.62757134668604664e-01, 1.23491976262065845e-01, None),
    (
        4.33395394129247213e-01,
        1.34709217311473339e-01,
        Some(2.69266719309996350e-01),
    ),
    (2.94392862701460201e-01, 1.42775938577060085e-01, None),
    (
        1.48874338981631216e-01,
        1.47739104901338486e-01,
        Some(2.95524224714752870e-01),
    ),
    (0.0, 1.49445554002916897e-01, None),
];

/// How the variable of the rule maps onto the variable of the integrand.
#[derive(Clone, Copy)]
pub(super) enum Map {
    /// `x = t` on a finite interval.
    Identity,
    /// `x = a + (1 − t)/t`, `t ∈ (0, 1]`, for `[a, ∞)`: QAGI's map, which puts the infinite end
    /// at `t = 0`, where the floating point numbers are dense.
    Upper(f64),
    /// `x = b − (1 − t)/t` for `(−∞, b]`.
    Lower(f64),
    /// Both halves of the line folded onto `(0, 1]`: `x = ±(1 − t)/t`.
    Whole,
}

/// The integrand in the variable of the rule: `f(x(t)) · x′(t)`. A point beyond the largest
/// finite value of the integrand's type is not evaluated, and counts as 0 there: its share of an
/// integral that exists is below anything the type can hold.
fn mapped<T: Real>(f: &mut Integrand<'_, T>, map: Map, t: f64) -> f64 {
    let far = |x: f64| x.abs() > T::MAX.to_f64();
    match map {
        Map::Identity => f.at(t),
        Map::Upper(a) | Map::Lower(a) => {
            let d = (1.0 - t) / t;
            let x = if let Map::Upper(_) = map {
                a + d
            } else {
                a - d
            };
            // Divided by t twice, as QUADPACK's `qk15i` does.
            if far(x) { 0.0 } else { f.at(x) / t / t }
        }
        Map::Whole => {
            let d = (1.0 - t) / t;
            if far(d) {
                0.0
            } else {
                (f.at(d) + f.at(-d)) / t / t
            }
        }
    }
}

/// The most nodes of a rule of the tables, G10–K21's.
const MOST: usize = 21;

/// A sample of the rule on an interval: the point `t` in the rule's variable, the size
/// `value = |g(t)|` of the integrand there, and `mass`, its share `w·|g(t)|·h` of the rule's
/// estimate of `∫|g|` (half of it for the centre, which belongs to both halves of the interval).
#[derive(Clone, Copy, Debug)]
struct Peak {
    t: f64,
    value: f64,
    mass: f64,
}

/// What the rule said about one interval.
#[derive(Clone, Copy)]
struct Interval {
    result: f64,
    error: f64,
    /// `∫|f|` on the interval, QUADPACK's `resabs`.
    resabs: f64,
    /// `∫|f − mean|`, QUADPACK's `resasc`.
    resasc: f64,
    /// The samples of the rule, ascending in `t`; the first `count` are the rule's.
    samples: [Peak; MOST],
    count: usize,
}

/// The pair `table` on `[a, b]`, with QUADPACK's error estimate.
fn rule<T: Real>(
    f: &mut Integrand<'_, T>,
    map: Map,
    a: f64,
    b: f64,
    table: &Table,
    s: &Settings,
) -> Interval {
    let (centre, half) = centre_and_half(a, b);
    let m = table.len() - 1;
    let count = 2 * m + 1;
    let (_, wkc, wgc) = table[m];
    let fc = mapped(f, map, centre);
    let mut resk = wkc * fc;
    let mut resg = wgc.map_or(0.0, |w| w * fc);
    let mut resabs = resk.abs();
    let h = half.abs();
    let none = Peak {
        t: 0.0,
        value: 0.0,
        mass: 0.0,
    };
    let mut samples = [none; MOST];
    samples[m] = Peak {
        t: centre,
        value: fc.abs(),
        mass: 0.5 * wkc * fc.abs() * h,
    };
    // The values at the nodes, to measure the spread about the mean afterwards. The sums run in
    // the order of the table, which is the order in which they reproduce scipy.integrate.quad's
    // decisions (its evaluation counts) on the battery of the tests.
    let mut values = [(0.0f64, 0.0f64); 10];
    for (j, &(u, wk, wg)) in table[..m].iter().enumerate() {
        let d = half * u;
        let (t1, t2) = (centre - d, centre + d);
        let (f1, f2) = (mapped(f, map, t1), mapped(f, map, t2));
        values[j] = (f1, f2);
        let sample = |t: f64, v: f64| Peak {
            t,
            value: v.abs(),
            mass: wk * v.abs() * h,
        };
        samples[j] = sample(t1, f1);
        samples[count - 1 - j] = sample(t2, f2);
        let sum = f1 + f2;
        resk += wk * sum;
        if let Some(wg) = wg {
            resg += wg * sum;
        }
        resabs += wk * (f1.abs() + f2.abs());
    }
    let mean = 0.5 * resk;
    let mut resasc = wkc * (fc - mean).abs();
    for (j, &(_, wk, _)) in table[..m].iter().enumerate() {
        let (f1, f2) = values[j];
        resasc += wk * ((f1 - mean).abs() + (f2 - mean).abs());
    }
    let result = resk * half;
    let resabs = resabs * h;
    let resasc = resasc * h;
    let mut error = ((resk - resg) * half).abs();
    if resasc != 0.0 && error != 0.0 {
        error = resasc * (200.0 * error / resasc).powf(1.5).min(1.0);
    }
    // QUADPACK's floor, which it drops below `tiny/(50ε)`; there the sums carry the rounding of
    // subnormal numbers, a fixed quantum, and the floor is 50 of it: `∫₀^(10⁻³¹⁰) dx` came out an
    // ulp off with the error 0 (review round 2).
    if resabs > 0.0 {
        error = error.max(50.0 * s.eps * resabs.max(s.tiny));
    }
    if !result.is_finite() {
        error = f64::INFINITY;
    }
    Interval {
        result,
        error,
        resabs,
        resasc,
        samples,
        count,
    }
}

/// The centre and the half-width of `[a, b]`, as QUADPACK forms them, `(a + b)/2` and `(b − a)/2`,
/// unless the sum or the difference overflows (an interval wider than the largest `f64`), where
/// they are formed from the halves of the ends.
pub(super) fn centre_and_half(a: f64, b: f64) -> (f64, f64) {
    let (sum, width) = (a + b, b - a);
    let centre = if sum.is_finite() {
        0.5 * sum
    } else {
        0.5 * a + 0.5 * b
    };
    let half = if width.is_finite() {
        0.5 * width
    } else {
        0.5 * b - 0.5 * a
    };
    (centre, half)
}

/// The point of the integrand's type at which the rule's variable `t` evaluates it (the
/// positive one of the pair on the whole line).
fn point<T: Real>(map: Map, t: f64) -> f64 {
    let x = match map {
        Map::Identity => t,
        Map::Upper(a) => a + (1.0 - t) / t,
        Map::Lower(b) => b - (1.0 - t) / t,
        Map::Whole => (1.0 - t) / t,
    };
    T::from_f64(x).to_f64()
}

/// Whether the interval `[a, b]` of the rule's variable may be bisected: each half must be at
/// least 2048 units in the last place wide, both in the rule's variable (an `f64`) and, mapped,
/// as the points of the integrand's type it is evaluated at. Narrower, the nodes of a half
/// would be rounded by more than a fraction of their spacing and the rule would no longer be
/// itself — its error estimate least of all — and its outermost nodes could round onto the
/// ends, where the integrand may be infinite; so the bisection that homes in on a singular end
/// stops there, and the extrapolation and the error estimate account for what lies beyond.
fn divisible<T: Real>(map: Map, a: f64, b: f64) -> bool {
    const ROOM: f64 = 2048.0;
    let ulp = |x: f64, eps: f64, tiny: f64| (x.abs() * eps).max(tiny);
    let big = a.abs().max(b.abs());
    let (mid, half) = centre_and_half(a, b);
    if half <= ROOM * ulp(big, f64::EPSILON, f64::MIN_POSITIVE) {
        return false;
    }
    let (eps, tiny) = (T::EPSILON.to_f64(), T::MIN_POSITIVE.to_f64());
    [(a, mid), (mid, b)].into_iter().all(|(lo, hi)| {
        let (x0, x1) = (point::<T>(map, lo), point::<T>(map, hi));
        // An infinite end (t = 0 of a map) leaves the half infinitely wide; two make no room.
        let width = (x1 - x0).abs();
        width > ROOM * ulp(x0.abs().max(x1.abs()), eps, tiny) || width.is_infinite()
    })
}

/// The largest error estimate, as a fraction of the estimate, that a result may be accepted with
/// above the round-off floor (see [`resolved`]).
pub(super) const RESOLVED: f64 = 0.1;

/// Whether an error estimate is one a result may be accepted with: below a tenth
/// ([`RESOLVED`]) of the result, or at the round-off level `100 ε ∫|f|` of the sums. QUADPACK
/// accepts any estimate within the tolerances, and so passes off as converged a feature the rule
/// has only grazed: on `∫₀^∞ e^(−(x−30)²) dx = √π` at `epsabs = 10⁻⁶`, QAGI (and
/// `scipy.integrate.quad`) return `5.5·10⁻⁷` with the error estimate `9.9·10⁻⁷`, the peak between
/// the nodes of every panel but one.
///
/// Every error estimate of these rules is a difference of a finer and a coarser estimate (`K − G`,
/// two levels or two degrees), trusted because the finer one is far more accurate than the
/// coarser. Where the nodes have seen only the tail of a feature, the coarser estimate holds
/// less of it than the finer, anything from nothing to most of it, and the difference is then a
/// large part of the finer estimate itself: with a positive integrand and `0 < coarse < 2·fine`
/// it is always below the estimate, which is why a test of the error against the whole estimate
/// let through results `10⁶⁸` times their estimates off (review round 2). An estimate the
/// coarser rule does not reproduce to its first digit (an error of a tenth of it or more) has
/// not been resolved, whatever the absolute tolerance allows; measured on 1200 far or narrow
/// features (review round 2), converged results within their estimates have so large an error
/// in 7 of 1087 runs (tanh-sinh), 2 of 585 (Clenshaw–Curtis) and 5 of 1728 (QAGS), the results
/// beyond them in 421 of 566, 211 of 255 and 7 of 47. Where the integral is 0 by cancellation,
/// the error is at the round-off floor of the rule and the result is accepted.
pub(super) fn resolved(error: f64, value: f64, absolute: f64, eps: f64) -> bool {
    error < RESOLVED * value.abs() || error <= 100.0 * eps * absolute
}

/// Why an integration stopped short of its tolerances.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Stop {
    /// It did not.
    None,
    /// The limit on the subintervals or the evaluations.
    Budget,
    /// Round-off, which the error estimate cannot get below.
    RoundOff,
    /// An interval too narrow to bisect further: a point where the integrand misbehaves.
    Narrow,
    /// The extrapolation did not converge.
    Extrapolation,
}

/// A sum by Neumaier's compensated summation.
pub(super) fn neumaier(xs: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    s + c
}

/// QUADPACK's `qpsrt`: after the interval `maxerr` was bisected into itself and the interval
/// `last`, the halves are put in their places in `order` (`iord`), the interval numbers by
/// decreasing error estimate, which is kept in order for its first `jupbn` places only — all of
/// them while fewer than half the subintervals allowed are used, fewer after, as no interval
/// beyond them will be bisected before the limit — and the next interval to bisect is the one at
/// place `nrmax`. Everything is numbered from 1, as in QUADPACK.
fn qpsrt(
    limit: usize,
    last: usize,
    maxerr: &mut usize,
    nrmax: &mut usize,
    segs: &[Segment],
    order: &mut [usize],
) {
    if last <= 2 {
        order[1] = 1;
        order[2] = 2;
    } else {
        let errmax = segs[*maxerr].error;
        // Should the bisection have raised the error, the larger half moves up past the places
        // before `nrmax`.
        if *nrmax != 1 {
            for _ in 1..*nrmax {
                let isucc = order[*nrmax - 1];
                if errmax <= segs[isucc].error {
                    break;
                }
                order[*nrmax] = isucc;
                *nrmax -= 1;
            }
        }
        let jupbn = if last > limit / 2 + 2 {
            limit + 3 - last
        } else {
            last
        };
        let errmin = segs[last].error;
        let jbnd = jupbn - 1;
        let ibeg = *nrmax + 1;
        // The larger half, from the top down.
        let mut place = None;
        if ibeg <= jbnd {
            for i in ibeg..=jbnd {
                let isucc = order[i];
                if errmax >= segs[isucc].error {
                    place = Some(i);
                    break;
                }
                order[i - 1] = isucc;
            }
        }
        match place {
            None => {
                order[jbnd] = *maxerr;
                order[jupbn] = last;
            }
            Some(i) => {
                order[i - 1] = *maxerr;
                // The smaller half, from the bottom up.
                let mut k = jbnd;
                let mut placed = false;
                for _ in i..=jbnd {
                    let isucc = order[k];
                    if errmin < segs[isucc].error {
                        order[k + 1] = last;
                        placed = true;
                        break;
                    }
                    order[k + 1] = isucc;
                    k -= 1;
                }
                if !placed {
                    order[i] = last;
                }
            }
        }
    }
    *maxerr = order[*nrmax];
}

/// How far below a sample of a bisected interval the nodes of its half next to the sample may
/// see the integrand before the half is taken not to account for it: a factor of ten. A half
/// whose nodes resolve the integrand near the sample see about what it saw (a peak resolved by
/// the nodes cannot fall far between two of them); a half that misses a feature the sample saw
/// sees orders of magnitude less.
const EVIDENCE: f64 = 10.0;

/// How many witnesses an interval keeps, the heaviest (a panel of `clenshaw_curtis` as many).
/// One was too few: the half of `[0, 1]` in QAGI's variable that held two bumps had the evidence
/// of both, kept the heavier, and lost the other bump, seen at 0.09 of its height, half of
/// `∫₀^∞` of the two (review round 2); a few distinct features in one interval are what an
/// integrand may have, more than four are for a rule that is told where they are.
pub(super) const WITNESSES: usize = 4;

/// The witnesses of an interval, heaviest first.
type Witnesses = [Option<Peak>; WITNESSES];

/// The total mass of the witnesses.
fn mass(witnesses: &Witnesses) -> f64 {
    witnesses.iter().flatten().map(|w| w.mass).sum()
}

/// One interval of the bisection, as QAGS keeps it in its lists.
#[derive(Clone, Copy)]
struct Segment {
    a: f64,
    b: f64,
    result: f64,
    /// The error estimate: the rule's, or the mass of the witnesses where that is larger.
    error: f64,
    /// The rule's own error estimate, which QUADPACK's tests for round-off compare.
    own: f64,
    /// `∫|g|` by the rule, QUADPACK's `resabs`.
    resabs: f64,
    /// The rule's samples, ascending; the first `count`.
    samples: [Peak; MOST],
    count: usize,
    /// Samples of the intervals this one was bisected from, inside this one, that the nodes of
    /// this one do not account for: the evidence of features between them.
    witnesses: Witnesses,
    /// Whether the interval counts in QUADPACK's `erlarg`, the error over the large intervals.
    large: bool,
}

impl Segment {
    fn new(rule: &Interval, a: f64, b: f64, witnesses: Witnesses, large: bool) -> Self {
        Self {
            a,
            b,
            result: rule.result,
            error: rule.error.max(mass(&witnesses)),
            own: rule.error,
            resabs: rule.resabs,
            samples: rule.samples,
            count: rule.count,
            witnesses,
            large,
        }
    }
}

/// Whether the nodes of `half` next to the sample `peak` — the last at or before it and the
/// first after it — see the integrand at least a tenth ([`EVIDENCE`]) as large as `peak` saw it.
fn explained(half: &Interval, peak: &Peak) -> bool {
    let samples = &half.samples[..half.count];
    let i = samples.partition_point(|s| s.t <= peak.t);
    [i.checked_sub(1), (i < samples.len()).then_some(i)]
        .into_iter()
        .flatten()
        .any(|k| EVIDENCE * samples[k].value >= peak.value)
}

/// The evidence the bisected interval `parent` leaves to its half `half` on `[lo, hi]` (the
/// left one or the right one, the halves meeting at `mid`): of the parent's samples on that side
/// (its centre on both) and of the parent's own witnesses that lie there, those the half does
/// not account for, the heaviest [`WITNESSES`] of them.
///
/// Every sample of the side, not only its largest: on `[0, 1]`, with peaks of width `3·10⁻⁴`
/// about 0.0643 and 0.2499, the second 0.895 high, the interval `[0, 0.5]` saw the first at 0.30
/// of its height and the second at 0.84 at its centre, which was the largest sample of both its
/// sides; offered only the largest, the left half had no evidence of the first peak, its nodes
/// missed it, and `quad_g7k15` returned `4.76·10⁻⁴ ± 7.8·10⁻¹⁴` at `10⁻¹²` for `1.008·10⁻³`
/// (round 3).
///
/// A sample at `mid` itself, the parent's centre, lies at an end of both halves, and each must
/// account for it: on `[−1, 1]` a peak of width 0.001 about −0.0021 is seen at the centre as
/// 0.012, the left half's nodes find the peak, and the right half's nearest node, 0.0022 inside
/// it, sees `10⁻⁸` — what lies between, `2.6·10⁻⁶` of the peak's tail, was in no error estimate
/// while either half could account for the centre, and `quad_g7k15` came out `1.5·10⁻³` off
/// relatively at the error `2.8·10⁻¹⁴` (review round 2). What such a sample can hide is in the
/// gap between the end and the half's outermost node, so its mass there is its value times the
/// gap, which halves with each bisection that homes in on the end: a jump exactly at the centre,
/// which no node of one half can ever account for, costs the bisections that bring that mass
/// below the tolerance, where it was taken on trust before.
fn witness(
    parent: &Segment,
    left: bool,
    mid: f64,
    half: &Interval,
    (lo, hi): (f64, f64),
) -> Witnesses {
    let on_side = |p: &Peak| p.t == mid || (p.t < mid) == left;
    let own = parent.samples[..parent.count].iter().copied();
    let inherited = parent.witnesses.into_iter().flatten();
    let samples = &half.samples[..half.count];
    let mut found: Vec<Peak> = own
        .chain(inherited)
        .filter(|p| on_side(p) && !explained(half, p))
        .map(|p| {
            // A sample at an end of the half: the share of the gap next to it.
            let gap = if p.t == lo {
                Some(samples[0].t - lo)
            } else if p.t == hi {
                Some(hi - samples[half.count - 1].t)
            } else {
                None
            };
            match gap {
                Some(g) => Peak {
                    mass: p.value * g.abs(),
                    ..p
                },
                None => p,
            }
        })
        .collect();
    found.sort_by(|x, y| y.mass.total_cmp(&x.mass));
    let mut out = [None; WITNESSES];
    for (slot, w) in out.iter_mut().zip(found) {
        *slot = Some(w);
    }
    out
}

/// How many copies of the extrapolation measure its rounding.
const SHADOWS: usize = 4;

/// Copies of the extrapolation of QAGS fed the same estimates, each moved by its rounding — `ε`
/// times `∫|f|` over the intervals — with a sign drawn at random (a fixed sequence, so that a
/// result is the same from one run to the next): how far their results scatter from the
/// extrapolation's own is the effect of the rounding of the estimates on it. The epsilon
/// algorithm magnifies that rounding by `1/(1 − ρ)` and its powers, `ρ` the ratio by which the
/// estimates converge — a thousandfold and more as `ρ` nears 1, towards a singularity like
/// `t^(−0.97)` — and its own error estimate, the spread of its last three results, does not see
/// it: on `∫ x^(−1.032) dx` over `[26.49, ∞)` the three agreed within `8·10⁻¹²` and were all
/// `2·10⁻¹¹` off, where the same algorithm on the exact estimates comes within `2·10⁻¹⁵`.
struct Shadows {
    tables: [Epsilon; SHADOWS],
    /// The state of the SplitMix64 sequence the signs are drawn from.
    state: u64,
    eps: f64,
}

impl Shadows {
    fn new(eps: f64) -> Self {
        Self {
            tables: core::array::from_fn(|_| Epsilon::new(eps)),
            state: 0x005e_ed0f_5ad0_0715,
            eps,
        }
    }

    /// `v` moved by `±ε·absolute`, the sign the next bit of the sequence.
    fn moved(&mut self, v: f64, absolute: f64) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let sign = if (z ^ (z >> 31)) >> 63 == 1 {
            1.0
        } else {
            -1.0
        };
        v + sign * self.eps * absolute
    }

    fn seed(&mut self, v: f64, absolute: f64) {
        for k in 0..SHADOWS {
            let moved = self.moved(v, absolute);
            self.tables[k].seed(moved);
        }
    }

    /// Extrapolates the moved copies of `v`; the largest distance of their results from
    /// `result`, the extrapolation of `v` itself.
    fn push(&mut self, v: f64, absolute: f64, result: f64) -> f64 {
        let mut scatter = 0.0f64;
        for k in 0..SHADOWS {
            let moved = self.moved(v, absolute);
            let (r, _) = self.tables[k].push(moved);
            scatter = scatter.max((r - result).abs());
        }
        scatter
    }
}

/// The integral of `f` over `[a, b]` in the variable of the rule (`[0, 1]` for the infinite
/// maps), by QAGS with the pair `table`: QUADPACK's `qagse` step for step (with its lists, the
/// order `qpsrt` keeps them in, and its counters), the subinterval limit the smaller of
/// `max_subintervals` and what `max_evaluations` allows, and three tests of its own.
///
/// * An interval is not bisected where a node of a half could round onto an end of it.
/// * The totals of the estimates, of their errors and of `∫|f|` are summed afresh over the
///   intervals after every bisection, by compensated summation. QUADPACK keeps running sums,
///   `errsum + erro12 − errmax`, and they cancel: a first panel of QAGI that saw a peak at one
///   node, with an error estimate of 86, bisected into halves that saw nothing, left
///   `86 + 2·10⁻⁵⁶ − 86 = 0` as the error of the whole, and the result `8·10⁻⁵⁶` was accepted
///   for `√π`.
/// * A sample of a bisected interval that the nodes of the half it lies in do not account for
///   (neither node about it a tenth as large) is evidence of a feature between those nodes. It
///   stays with the half, and with the halves of the half, as a lower bound on its error — the
///   sample's share of the parent's `∫|f|`, or, for the parent's centre, which lies at an end of
///   both halves and is evidence for each, its value times the gap between that end and the
///   half's outermost node — until a node of an interval about it sees a tenth of what it saw;
///   and it is added to the error of an extrapolated result, which cannot see it. An interval
///   keeps up to [`WITNESSES`] of them, the heaviest. So a value that an interval saw does not
///   vanish between the nodes of its halves, even where their values are exactly zero, while a
///   node beside it sees less than a tenth of it. Once one sees that much, the node's own value
///   and share, of the smaller interval, are what stands for the feature at the next bisection:
///   the evidence of a narrow feature can so shrink by a tenth and more a step, and fall below a
///   loose tolerance before a node finds the feature.
pub(super) fn qags<T: Real>(
    f: &mut Integrand<'_, T>,
    map: Map,
    a: f64,
    b: f64,
    table: &Table,
    s: &Settings,
) -> Outcome {
    let per_rule = (2 * table.len() - 1) * if let Map::Whole = map { 2 } else { 1 };
    // The k subintervals of QAGS cost (2k − 1) applications of the rule.
    let limit = s
        .max_subintervals
        .min((s.max_evaluations / per_rule).div_ceil(2))
        .max(1);
    let (epmach, uflow) = (s.eps, s.tiny);
    let bound = |v: f64| s.epsabs.max(s.epsrel * v.abs());
    let first = rule(f, map, a, b, table, s);
    if !first.result.is_finite() {
        return Outcome::failed(first.result);
    }
    let (result0, defabs) = (first.result, first.resabs);
    let mut errbnd = bound(result0);
    let mut stop = Stop::None;
    if first.error <= 100.0 * epmach * defabs && first.error > errbnd {
        stop = Stop::RoundOff;
    }
    if limit == 1 {
        stop = Stop::Budget;
    }
    // QUADPACK's acceptance of the first estimate: within the tolerances and not at its cap
    // `resasc`, which would say only that the rule did not resolve the integrand.
    let accept = first.error <= errbnd
        && first.error != first.resasc
        && resolved(first.error, result0, defabs, epmach);
    if stop != Stop::None || accept || first.error == 0.0 {
        return Outcome {
            value: result0,
            error: first.error,
            converged: stop == Stop::None
                && first.error <= errbnd
                && resolved(first.error, result0, defabs, epmach),
        };
    }
    // The intervals, numbered from 1 as in QUADPACK (the entry 0 unused), grown as the bisection
    // goes rather than reserved for the whole of a budget that may be vast.
    let mut segs = Vec::with_capacity(limit.min(64) + 1);
    segs.push(Segment::new(&first, a, a, [None; WITNESSES], false));
    segs.push(Segment::new(&first, a, b, [None; WITNESSES], true));
    let mut order = vec![0usize; 3];
    order[1] = 1;
    let mut table_eps = Epsilon::new(epmach);
    table_eps.seed(result0);
    // The rounding of the estimates, as the extrapolation feels it: the same extrapolation, in
    // `SHADOWS` copies, of the estimates moved by their rounding `ε ∫|f|` with random signs.
    let mut shadows = Shadows::new(epmach);
    shadows.seed(result0, defabs);
    // The rounding error of the best extrapolation: how far the copies landed from it.
    let mut noise = 0.0f64;
    let mut maxerr = 1usize;
    let (mut area, mut errsum, mut absolute) = (result0, first.error, defabs);
    // The evidence of the samples no node has accounted for yet, summed.
    let mut lost = 0.0f64;
    // The extrapolated result and its error, QUADPACK's `result` and `abserr` with `oflow`.
    let mut result = f64::NAN;
    let mut abserr = f64::INFINITY;
    let mut nrmax = 1usize;
    let mut ktmin = 0usize;
    let (mut extrap, mut noext, mut ierro) = (false, false, false);
    let (mut iroff1, mut iroff2, mut iroff3) = (0usize, 0usize, 0usize);
    let one_sign = result0.abs() >= (1.0 - 50.0 * epmach) * defabs;
    let (mut small, mut ertest, mut correc) = (0.0f64, 0.0f64, 0.0f64);
    for last in 2..=limit {
        let parent = segs[maxerr];
        if !divisible::<T>(map, parent.a, parent.b) {
            stop = Stop::Narrow;
            break;
        }
        // Bisect the interval with the nrmax-th largest error estimate.
        let (a1, b2) = (parent.a, parent.b);
        let b1 = centre_and_half(a1, b2).0;
        let a2 = b1;
        let left = rule(f, map, a1, b1, table, s);
        let right = rule(f, map, a2, b2, table, s);
        let area12 = left.result + right.result;
        if !area12.is_finite() {
            return Outcome::failed(area12);
        }
        // QUADPACK's tests for round-off, on the rule's own estimates.
        let own12 = left.error + right.error;
        if left.resasc != left.error && right.resasc != right.error {
            if (parent.result - area12).abs() <= 1e-5 * area12.abs() && own12 >= 0.99 * parent.own {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            if last > 10 && own12 > parent.own {
                iroff3 += 1;
            }
        }
        let large = (b1 - a1).abs() > small;
        let wl = witness(&parent, true, b1, &left, (a1, b1));
        let wr = witness(&parent, false, b1, &right, (a2, b2));
        let l = Segment::new(&left, a1, b1, wl, large);
        let r = Segment::new(&right, a2, b2, wr, large);
        // The half with the larger error keeps the number of the interval bisected.
        let (keep, new) = if r.error > l.error { (r, l) } else { (l, r) };
        segs[maxerr] = keep;
        segs.push(new);
        if order.len() < last + 2 {
            order.resize(last + 2, 0);
        }
        // The totals afresh, where QUADPACK updates running sums that cancel.
        let current = &segs[1..];
        errsum = neumaier(current.iter().map(|s| s.error));
        area = neumaier(current.iter().map(|s| s.result));
        absolute = neumaier(current.iter().map(|s| s.resabs));
        lost = neumaier(
            current
                .iter()
                .flat_map(|s| s.witnesses.into_iter().flatten())
                .map(|w| w.mass),
        );
        errbnd = bound(area);
        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            stop = Stop::RoundOff;
        }
        if iroff2 >= 5 {
            ierro = true;
        }
        if last == limit {
            stop = Stop::Budget;
        }
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * epmach) * (a2.abs() + 1000.0 * uflow) {
            stop = Stop::Narrow;
        }
        qpsrt(limit, last, &mut maxerr, &mut nrmax, &segs, &mut order);

        if errsum <= errbnd && resolved(errsum, area, absolute, epmach) {
            return Outcome {
                value: area,
                error: errsum,
                converged: true,
            };
        }
        if stop != Stop::None {
            break;
        }
        if last == 2 {
            small = (b - a).abs() * 0.375;
            segs.iter_mut().for_each(|s| s.large = true);
            ertest = errbnd;
            table_eps.seed(area);
            shadows.seed(area, absolute);
            continue;
        }
        if noext {
            continue;
        }
        if !extrap {
            // While the interval to bisect next is a large one, bisect as QAG does.
            if (segs[maxerr].b - segs[maxerr].a).abs() > small {
                continue;
            }
            extrap = true;
            nrmax = 2;
        }
        // QUADPACK's `erlarg`, the error over the intervals wider than `small` (and those there
        // were at its last reset), summed afresh as the other totals are.
        let erlarg = neumaier(segs[1..].iter().filter(|s| s.large).map(|s| s.error));
        // The smallest interval has the largest error: before extrapolating, bisect the large
        // intervals whose errors add up to more than the extrapolation could hope to beat.
        if !ierro && erlarg > ertest {
            let jupbnd = if last > 2 + limit / 2 {
                limit + 3 - last
            } else {
                last
            };
            // QUADPACK's do-loop, as many passes as places from `nrmax` on entry to `jupbnd`,
            // moving `nrmax` down the list as it goes.
            let mut found = false;
            let passes = (jupbnd + 1).saturating_sub(nrmax);
            for _ in 0..passes {
                maxerr = order[nrmax];
                if (segs[maxerr].b - segs[maxerr].a).abs() > small {
                    found = true;
                    break;
                }
                nrmax += 1;
            }
            if found {
                continue;
            }
        }
        // Extrapolate.
        let (reseps, abseps) = table_eps.push(area);
        let scatter = shadows.push(area, absolute, reseps);
        // A result on which three elements of a column agreed to the precision of the values
        // has shown its rounding already; the copies, moved off that agreement, fall back on
        // coarser elements and measure nothing about it (as on a jump, whose estimates the
        // algorithm extrapolates exactly).
        let scatter = if table_eps.settled() { 0.0 } else { scatter };

        ktmin += 1;
        // QUADPACK's test that the extrapolation has stopped improving, against the rules' own
        // errors: the evidence of a feature not found yet is no reason to stop looking for it.
        let owned = neumaier(segs[1..].iter().map(|s| s.own));
        if ktmin > 5 && abserr < 1e-3 * owned {
            stop = Stop::Extrapolation;
        }
        if abseps < abserr {
            ktmin = 0;
            abserr = abseps;
            result = reseps;
            noise = scatter;
            correc = erlarg;
            ertest = bound(reseps);
            // Accepted with what its error estimate does not see: the rounding, and the
            // evidence of the features no node has found yet.
            if abserr + noise + lost <= ertest {
                break;
            }
        }
        if table_eps.len() == 1 {
            noext = true;
        }
        if stop == Stop::Extrapolation {
            break;
        }
        // Back to the largest error, among intervals half as wide.
        maxerr = order[1];
        nrmax = 1;
        extrap = false;
        small *= 0.5;
        segs.iter_mut().for_each(|s| s.large = true);
    }
    finish(
        (result, abserr),
        (area, errsum),
        correc,
        stop,
        ierro,
        one_sign,
        (defabs, absolute),
        noise + lost,
        s,
    )
}

/// QUADPACK's choice between the extrapolated and the plain estimate at the end of QAGS, and
/// its test for a divergent integral. `sums` is `(area, errsum)`, the plain estimate and its
/// error; `(defabs, absolute)` are `∫|f|` by the first application of the rule and summed over
/// the intervals; `unseen` is what the extrapolation's error estimate cannot see, the rounding
/// error of the extrapolated result (the scatter of the shadow tables) and the evidence of the
/// features no node has found yet. The error of an extrapolated result is the epsilon
/// algorithm's, with its floor of `5 ε |result|` as in QUADPACK, plus `unseen`: it is not held
/// to the floor `50 ε ∫|f|` of an interval's error, which in `f32` would take most of the
/// default tolerance from results measured a hundred times more accurate than that.
#[allow(clippy::too_many_arguments)]
fn finish(
    extrapolated: (f64, f64),
    sums: (f64, f64),
    correction: f64,
    mut stop: Stop,
    roff_table: bool,
    one_sign: bool,
    (defabs, absolute): (f64, f64),
    unseen: f64,
    s: &Settings,
) -> Outcome {
    let (area, errsum) = sums;
    let bound = |v: f64| s.epsabs.max(s.epsrel * v.abs());
    let good =
        |error: f64, value: f64| error <= bound(value) && resolved(error, value, absolute, s.eps);
    let plain = |stop: Stop| Outcome {
        value: area,
        error: errsum,
        converged: stop == Stop::None && good(errsum, area),
    };
    let (result, abserr) = extrapolated;
    if abserr == f64::INFINITY {
        return plain(stop);
    }
    let mut abserr = abserr + unseen;
    if stop != Stop::None || roff_table {
        if roff_table {
            abserr += correction;
        }
        if stop == Stop::None {
            stop = Stop::RoundOff;
        }
        if result == 0.0 || area == 0.0 {
            if abserr > errsum {
                return plain(stop);
            }
            if area == 0.0 {
                return Outcome {
                    value: result,
                    error: abserr,
                    converged: false,
                };
            }
        } else if abserr / result.abs() > errsum / area.abs() {
            return plain(stop);
        }
    }
    // The test for divergence: an integral of one sign far below the integral of |f| is fine;
    // otherwise a plain and an extrapolated estimate two orders of magnitude apart, or an error
    // larger than the estimate, say that the integral does not exist.
    if one_sign || result.abs().max(area.abs()) > defabs * 0.01 {
        let ratio = result / area;
        if !(0.01..=100.0).contains(&ratio) || errsum > area.abs() {
            return Outcome {
                value: result,
                error: abserr.max(errsum),
                converged: false,
            };
        }
    }
    Outcome {
        value: result,
        error: abserr,
        converged: stop == Stop::None && good(abserr, result),
    }
}
