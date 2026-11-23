//! The cubature: the degrees of Genz and Malik's rules, the six test families of Genz (1984)
//! against their closed forms in 2 to 6 dimensions, the cost of a box and of the adaptive loop,
//! and the dimensions the rule refuses (audit n_quad-14).

use a::math::numerics::{QuadOptions, QuadResult, cubature, quad_g7k15};
use a::math::special::erf;
use core::f64::consts::PI;

/// The points of the rule of degree 7 in `d` dimensions.
fn per_box(d: usize) -> usize {
    (1 << d) + 2 * d * d + 2 * d + 1
}

/// The mean of `x^e` over `[-1, 1]^d`.
fn mean_monomial(e: &[u32]) -> f64 {
    e.iter()
        .map(|&k| {
            if k % 2 == 1 {
                0.0
            } else {
                1.0 / (f64::from(k) + 1.0)
            }
        })
        .product()
}

/// Every exponent vector of `d` entries with sum `degree`.
fn exponents(d: usize, degree: u32) -> Vec<Vec<u32>> {
    if d == 1 {
        return vec![vec![degree]];
    }
    (0..=degree)
        .flat_map(|k| {
            exponents(d - 1, degree - k).into_iter().map(move |mut e| {
                e.push(k);
                e
            })
        })
        .collect()
}

#[test]
fn the_genz_malik_rules_have_the_degrees_they_claim() {
    // One box on [−1, 1]^d: the rule of degree 7 integrates every monomial of degree 7 or less
    // exactly and x₁⁸ not (by 5e-3 of the volume); the error of a polynomial of degree 5 or less
    // is the round-off floor, as the rule of degree 5 is exact on it too, and such a polynomial
    // converges on the first box.
    let one_box = QuadOptions::new().with_max_subintervals(1);
    for d in 2..=6usize {
        let volume = 2f64.powi(d as i32);
        for degree in 0..=7u32 {
            for e in exponents(d, degree) {
                let f = |x: &[f64]| {
                    x.iter()
                        .zip(&e)
                        .map(|(t, &k)| t.powi(k as i32))
                        .product::<f64>()
                };
                let r = cubature(f, &vec![-1.0; d], &vec![1.0; d], &one_box).unwrap();
                let want = volume * mean_monomial(&e);
                assert!(
                    (r.value - want).abs() <= 1e-14 * volume,
                    "d = {d}, x^{e:?}: {} against {want}",
                    r.value
                );
                assert_eq!(r.evaluations, per_box(d));
            }
        }
        let r = cubature(
            |x: &[f64]| x[0].powi(8),
            &vec![-1.0; d],
            &vec![1.0; d],
            &one_box,
        )
        .unwrap();
        assert!(
            (r.value - volume / 9.0).abs() > 1e-3 * volume,
            "d = {d}: x⁸ is exact"
        );
        // x₁²x₂² + x₁⁴ − 3x₂⁴ x₁ + 1: degree 5, the first box converges at the defaults.
        let p =
            |x: &[f64]| x[0] * x[0] * x[1] * x[1] + x[0].powi(4) - 3.0 * x[1].powi(4) * x[0] + 1.0;
        let r = cubature(p, &vec![-1.0; d], &vec![1.0; d], &QuadOptions::new()).unwrap();
        let want = volume * (1.0 / 9.0 + 1.0 / 5.0 + 1.0);
        assert!(r.converged && r.evaluations == per_box(d), "d = {d}: {r:?}");
        assert!((r.value - want).abs() <= r.error, "d = {d}: {r:?}");
    }
}

/// A deterministic generator for the parameters of the families.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The families of Genz (1984) on the unit cube, with the difficulty `Σ aᵢ` he gives them.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    Oscillatory,
    ProductPeak,
    CornerPeak,
    Gaussian,
    Continuous,
    Discontinuous,
}

impl Family {
    fn difficulty(self) -> f64 {
        match self {
            Family::Oscillatory => 9.0,
            Family::ProductPeak => 7.25,
            Family::CornerPeak => 1.85,
            Family::Gaussian => 7.03,
            Family::Continuous => 20.4,
            Family::Discontinuous => 4.3,
        }
    }

    /// The integrand with the parameters `a` and `u`, and its integral over the box
    /// `[lo, hi]` from the closed form.
    fn integral(self, a: &[f64], u: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
        let d = a.len();
        match self {
            Family::Oscillatory => {
                // Re e^(2πiu₁) Πⱼ (e^(iaⱼhⱼ) − e^(iaⱼlⱼ))/(iaⱼ).
                let (mut re, mut im) = ((2.0 * PI * u[0]).cos(), (2.0 * PI * u[0]).sin());
                for j in 0..d {
                    let (s1, c1) = (a[j] * hi[j]).sin_cos();
                    let (s0, c0) = (a[j] * lo[j]).sin_cos();
                    // (cos + i sin)(h) − (cos + i sin)(l), divided by i a.
                    let (dr, di) = (c1 - c0, s1 - s0);
                    let (fr, fi) = (di / a[j], -dr / a[j]);
                    (re, im) = (re * fr - im * fi, re * fi + im * fr);
                }
                re
            }
            Family::ProductPeak => (0..d)
                .map(|j| a[j] * ((a[j] * (hi[j] - u[j])).atan() - (a[j] * (lo[j] - u[j])).atan()))
                .product(),
            Family::CornerPeak => {
                let mut s = 0.0;
                for mask in 0u32..(1 << d) {
                    let mut t = 1.0;
                    let mut upper = 0;
                    for j in 0..d {
                        if mask >> j & 1 == 1 {
                            t += a[j] * hi[j];
                            upper += 1;
                        } else {
                            t += a[j] * lo[j];
                        }
                    }
                    s += if upper % 2 == 0 { 1.0 } else { -1.0 } / t;
                }
                let factorial: f64 = (1..=d).map(|k| k as f64).product();
                s / (factorial * a.iter().product::<f64>())
            }
            Family::Gaussian => (0..d)
                .map(|j| {
                    PI.sqrt() / (2.0 * a[j])
                        * (erf(a[j] * (hi[j] - u[j])) - erf(a[j] * (lo[j] - u[j])))
                })
                .product(),
            Family::Continuous => (0..d)
                .map(|j| {
                    // ∫ e^(−a|x − u|) over [l, h], for a box on one side of u or the other.
                    let side = |x: f64| {
                        if x >= u[j] {
                            1.0 - (-a[j] * (x - u[j])).exp()
                        } else {
                            -(1.0 - (-a[j] * (u[j] - x)).exp())
                        }
                    };
                    (side(hi[j]) - side(lo[j])) / a[j]
                })
                .product(),
            Family::Discontinuous => (0..d)
                .map(|j| {
                    let h = if j < 2 { hi[j].min(u[j]) } else { hi[j] };
                    if h <= lo[j] {
                        0.0
                    } else {
                        ((a[j] * h).exp() - (a[j] * lo[j]).exp()) / a[j]
                    }
                })
                .product(),
        }
    }

    fn at(self, a: &[f64], u: &[f64], x: &[f64]) -> f64 {
        let dot = |w: &[f64]| w.iter().zip(x).map(|(p, q)| p * q).sum::<f64>();
        match self {
            Family::Oscillatory => (2.0 * PI * u[0] + dot(a)).cos(),
            Family::ProductPeak => (0..a.len())
                .map(|j| 1.0 / (a[j].powi(-2) + (x[j] - u[j]).powi(2)))
                .product(),
            Family::CornerPeak => (1.0 + dot(a)).powi(-(a.len() as i32 + 1)),
            Family::Gaussian => (-(0..a.len())
                .map(|j| (a[j] * (x[j] - u[j])).powi(2))
                .sum::<f64>())
            .exp(),
            Family::Continuous => (-(0..a.len())
                .map(|j| a[j] * (x[j] - u[j]).abs())
                .sum::<f64>())
            .exp(),
            Family::Discontinuous => {
                if x[0] > u[0] || x[1] > u[1] {
                    0.0
                } else {
                    dot(a).exp()
                }
            }
        }
    }

    /// Random parameters: `a` scaled to the difficulty, `u` in the cube.
    fn parameters(self, d: usize, rng: &mut Lcg) -> (Vec<f64>, Vec<f64>) {
        let mut a: Vec<f64> = (0..d).map(|_| 0.05 + rng.next()).collect();
        let s: f64 = a.iter().sum();
        for x in &mut a {
            *x *= self.difficulty() / s;
        }
        let u = (0..d).map(|_| rng.next()).collect();
        (a, u)
    }
}

/// Whether a result keeps its promise.
fn honest(r: &QuadResult<f64>, exact: f64) -> bool {
    !r.converged || (r.value - exact).abs() <= r.error + 1e-14 * exact.abs()
}

#[test]
fn the_closed_forms_of_the_families_are_right() {
    // The closed forms the tests below rest on, against the adaptive rule itself on a box away
    // from every kink, to 1e-9 (the inclusion and exclusion of the corner peak's form loses some
    // (aᵢhᵢ)^(−d) ulps on a box of side h), and against their own pieces: a box split in two.
    let mut rng = Lcg(7);
    for family in [
        Family::Oscillatory,
        Family::ProductPeak,
        Family::CornerPeak,
        Family::Gaussian,
        Family::Continuous,
        Family::Discontinuous,
    ] {
        for d in [2usize, 3] {
            let (a, u) = family.parameters(d, &mut rng);
            let (lo, hi) = (vec![0.0; d], vec![1.0; d]);
            let whole = family.integral(&a, &u, &lo, &hi);
            let mut mid = hi.clone();
            mid[0] = 0.5;
            let mut lo2 = lo.clone();
            lo2[0] = 0.5;
            let halves = family.integral(&a, &u, &lo, &mid) + family.integral(&a, &u, &lo2, &hi);
            assert!((whole - halves).abs() <= 1e-14 * whole.abs(), "{family:?}");
            // A box of side 0.1 away from every kink and jump (inside the support of the
            // discontinuous family), where the adaptive rule is reliable; small enough a box
            // would make the inclusion and exclusion of the corner peak's closed form cancel.
            let (l, h): (Vec<f64>, Vec<f64>) = u
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    if family == Family::Discontinuous && j < 2 {
                        (0.25 * x, 0.5 * x)
                    } else if x > 0.5 {
                        (0.1, 0.2)
                    } else {
                        (0.8, 0.9)
                    }
                })
                .unzip();
            let want = family.integral(&a, &u, &l, &h);
            // A relative tolerance only, as far from its centre the Gaussian is small; the
            // closed form within the rule's own estimate, converged or not.
            let relative = QuadOptions::new().with_epsabs(0.0).with_epsrel(1e-10);
            let got = cubature(|x: &[f64]| family.at(&a, &u, x), &l, &h, &relative).unwrap();
            assert!(
                (got.value - want).abs() <= got.error + 1e-9 * want.abs(),
                "{family:?} d = {d}: {got:?} against {want}"
            );
        }
    }
}

#[test]
fn the_smooth_families_are_integrated_honestly_in_two_to_six_dimensions() {
    // Genz's oscillatory, product peak, corner peak and Gaussian families, 6 random members
    // each in d = 2 … 6 at the tolerance 1e-6: no converged result is beyond its estimate (the
    // plain |I₇ − I₅| was 18 to 56 times below the error of corner peaks in 3 to 5 dimensions),
    // and in two dimensions every one converges at the defaults.
    let cases = if cfg!(miri) { 1 } else { 6 };
    let opts = QuadOptions::new().with_epsabs(1e-6).with_epsrel(1e-6);
    let mut converged = 0;
    for family in [
        Family::Oscillatory,
        Family::ProductPeak,
        Family::CornerPeak,
        Family::Gaussian,
    ] {
        for d in 2..=6usize {
            let mut rng = Lcg(1000 + d as u64);
            for _ in 0..cases {
                let (a, u) = family.parameters(d, &mut rng);
                let exact = family.integral(&a, &u, &vec![0.0; d], &vec![1.0; d]);
                let f = |x: &[f64]| family.at(&a, &u, x);
                let r = cubature(f, &vec![0.0; d], &vec![1.0; d], &opts).unwrap();
                assert!(
                    honest(&r, exact),
                    "{family:?} d = {d} a = {a:?} u = {u:?}: {r:?}, exact {exact}"
                );
                if d == 2 {
                    assert!(r.converged, "{family:?} d = 2: {r:?}");
                }
                converged += usize::from(r.converged);
            }
        }
    }
    // All of two dimensions, and some beyond (in 3 to 6 dimensions the default 200 boxes are
    // few for 1e-6).
    assert!(converged >= 4 * cases + cases, "only {converged} converged");
}

#[test]
fn a_kink_or_a_jump_is_integrated_once_the_box_is_split_at_it() {
    // Genz's continuous family has kinks on the planes xᵢ = uᵢ, his discontinuous one a jump on
    // x₁ = u₁ and x₂ = u₂: split there, the pieces are smooth and integrated to their tolerance.
    let mut rng = Lcg(99);
    let tolerance = |d: usize| if d == 2 { 1e-8 } else { 1e-6 };
    for d in [2usize, 3] {
        for family in [Family::Continuous, Family::Discontinuous] {
            let (a, u) = family.parameters(d, &mut rng);
            let f = |x: &[f64]| family.at(&a, &u, x);
            // The boxes of the split: each axis cut at uᵢ (the continuous family), or the one
            // box where the discontinuous one is not 0.
            let boxes: Vec<(Vec<f64>, Vec<f64>)> = if family == Family::Continuous {
                (0u32..(1 << d))
                    .map(|mask| {
                        (0..d)
                            .map(|j| {
                                if mask >> j & 1 == 1 {
                                    (u[j], 1.0)
                                } else {
                                    (0.0, u[j])
                                }
                            })
                            .unzip()
                    })
                    .collect()
            } else {
                let hi: Vec<f64> = (0..d).map(|j| if j < 2 { u[j] } else { 1.0 }).collect();
                vec![(vec![0.0; d], hi)]
            };
            let (mut value, mut error) = (0.0, 0.0);
            let opts = QuadOptions::new()
                .with_epsabs(tolerance(d))
                .with_epsrel(tolerance(d));
            for (lo, hi) in &boxes {
                let r = cubature(f, lo, hi, &opts).unwrap();
                assert!(r.converged, "{family:?} d = {d} on {lo:?}–{hi:?}: {r:?}");
                value += r.value;
                error += r.error;
            }
            let exact = family.integral(&a, &u, &vec![0.0; d], &vec![1.0; d]);
            assert!(
                (value - exact).abs() <= error,
                "{family:?} d = {d}: {value} against {exact}"
            );
        }
    }
}

#[test]
fn each_box_is_evaluated_once() {
    // Audit n_quad-14: the old rule re-evaluated every box on every pass, 225 × 20100 points in
    // two dimensions for 200 boxes. Now k boxes cost (2k − 1) applications of the rule.
    let hard = |x: &[f64]| (x[0] - 0.3).abs().sqrt() * (40.0 * x[1]).sin();
    for (d, m) in [(2usize, 200usize), (2, 7), (3, 50), (4, 10)] {
        let opts = QuadOptions::new().with_max_subintervals(m);
        let r = cubature(hard, &vec![0.0; d], &vec![1.0; d], &opts).unwrap();
        assert!(!r.converged);
        assert_eq!(
            r.evaluations,
            per_box(d) * (2 * m - 1),
            "d = {d}, {m} boxes"
        );
    }
    // The budget of evaluations stops it as well, before it is exceeded.
    let opts = QuadOptions::new().with_max_evaluations(1000);
    let r = cubature(hard, &[0.0, 0.0], &[1.0, 1.0], &opts).unwrap();
    assert!(!r.converged && r.evaluations <= 1000 && r.evaluations > 1000 - 2 * per_box(2));
}

#[test]
fn a_box_of_many_dimensions_is_refused_or_integrated_never_aborted() {
    // Audit n_quad-14: 12 dimensions asked for 3e15 bytes and aborted, 15 overflowed. A box
    // whose rule does not fit the budget of evaluations is refused; one that fits is integrated.
    let one = |_: &[f64]| 1.0;
    for d in [12usize, 15, 17, 30, 64, 100, 1000] {
        let r = cubature(one, &vec![0.0; d], &vec![1.0; d], &QuadOptions::new());
        if d <= 15 {
            let r = r.unwrap();
            assert!(
                r.converged && (r.value - 1.0).abs() < 1e-12,
                "d = {d}: {r:?}"
            );
            assert_eq!(r.evaluations, per_box(d));
        } else {
            assert!(r.is_none(), "d = {d}");
        }
    }
    // The old rule could not meet the defaults on the constant 1 in six dimensions.
    let r = cubature(one, &[0.0; 6], &[1.0; 6], &QuadOptions::new()).unwrap();
    assert!(r.converged && r.evaluations == per_box(6) && (r.value - 1.0).abs() < 1e-14);
}

#[test]
fn cubature_in_f32_and_in_one_dimension() {
    // f32 at its own default tolerance, and one dimension, where the cubature is quad_g7k15.
    let r = cubature(
        |v: &[f32]| (v[0] + v[1]).exp(),
        &[0.0, 0.0],
        &[1.0, 1.0],
        &QuadOptions::new(),
    )
    .unwrap();
    let want = (core::f64::consts::E - 1.0).powi(2);
    assert!(
        r.converged && (f64::from(r.value) - want).abs() <= f64::from(r.error) + 1e-6,
        "{r:?}"
    );
    let f = |x: f64| x.sin() * (-x).exp();
    let opts = QuadOptions::new();
    let one = cubature(|v: &[f64]| f(v[0]), &[0.0], &[3.0], &opts).unwrap();
    assert_eq!(Some(one), quad_g7k15(f, 0.0, 3.0, &opts));
    let reversed = cubature(|v: &[f64]| f(v[0]), &[3.0], &[0.0], &opts).unwrap();
    assert_eq!(reversed.value, -one.value);
    // A value that is not a number ends the integration.
    let r = cubature(
        |v: &[f64]| if v[0] > 0.9 { f64::NAN } else { 1.0 },
        &[0.0, 0.0],
        &[1.0, 1.0],
        &opts,
    )
    .unwrap();
    assert!(!r.converged);
}

#[test]
fn a_box_that_saw_only_the_tail_of_a_peak_is_not_accepted() {
    // Review round 2, R2-B3. The cubature accepted any box whose error met the tolerances: a
    // Gaussian of width 0.03 about (0.7, 0.3) in the unit square came back from the first box as
    // 1.0·10⁻²⁰ ± 9.0·10⁻²⁰ at the default tolerances, 17 evaluations, for 2.8·10⁻³; in three
    // dimensions, one about (0.238, 0.572, 0.788) as 1.9·10⁻⁴² ± 9.9·10⁻⁴². As in the
    // one-dimensional rules, an estimate is accepted only with an error below a tenth of it, or
    // at the round-off of the sums.
    let gauss = |c: Vec<f64>, w: f64| {
        move |x: &[f64]| {
            (-x.iter()
                .zip(&c)
                .map(|(x, c)| ((x - c) / w).powi(2))
                .sum::<f64>())
            .exp()
        }
    };
    let wide = |tol: f64| {
        QuadOptions::new()
            .with_epsabs(tol)
            .with_epsrel(tol)
            .with_max_subintervals(2000)
            .with_max_evaluations(400_000)
    };
    let w = 0.03;
    let cases: [(Vec<f64>, QuadOptions<f64>); 4] = [
        (vec![0.7, 0.3], QuadOptions::new()),
        (
            vec![0.7017147883437291, 0.3004278629006387],
            QuadOptions::new(),
        ),
        (vec![0.7017147883437291, 0.3004278629006387], wide(1e-8)),
        (
            vec![0.2378203978683119, 0.5715589539804056, 0.787582151731798],
            wide(1e-4),
        ),
    ];
    for (c, opts) in cases {
        let d = c.len();
        let exact = (w * PI.sqrt()).powi(d as i32);
        let r = cubature(gauss(c.clone(), w), &vec![0.0; d], &vec![1.0; d], &opts).unwrap();
        assert!(
            r.evaluations > per_box(d),
            "{c:?}: accepted on the first box, {r:?}"
        );
        assert!(honest(&r, exact), "{c:?}: {r:?}, exact {exact:e}");
        assert!(
            !r.converged || r.error < 0.1 * r.value.abs(),
            "{c:?}: {r:?}"
        );
    }
}

#[test]
fn narrow_peaks_are_beyond_the_estimate_no_more_often_than_documented_and_never_in_pieces() {
    // The confirmation of review round 2, R2-B3-residual. No rule that samples can certify a
    // feature narrower than the spacing of its points, and on Gaussian peaks of widths 0.01 and
    // 0.003 well inside the unit square and cube the cubature returns results beyond their
    // estimates, converged: peaks between all the points of the first boxes (every value 0, or a
    // far tail of 10⁻²⁵ and less that boxes seeing exactly 0 replace), the tail of a peak beyond
    // a face between two boxes, and boxes many widths long whose points see the peak along one
    // line only, which overestimate it (twice, at worst). The documentation states the rates of
    // the review's sweep — 60 random centres per dimension, width and tolerance, 2000 boxes and
    // 400,000 evaluations — and this pins them as upper bounds. Its advice, a box about the peak
    // integrated apart from the rest (the box cut at c ± 5w along every axis), is honest on every
    // one of the same peaks.
    let documented: [(usize, f64, [usize; 3]); 4] = [
        // dimension, width, runs beyond their estimates at 1e-4, 1e-6 and 1e-8
        (2, 0.01, [6, 8, 10]),
        (2, 0.003, [46, 44, 45]),
        (3, 0.01, [6, 6, 9]),
        (3, 0.003, [59, 59, 59]),
    ];
    let count = if cfg!(miri) { 2 } else { 60 };
    for (d, w, bounds) in documented {
        let exact = (w * PI.sqrt()).powi(d as i32);
        for (tol, bound) in [1e-4, 1e-6, 1e-8].into_iter().zip(bounds) {
            let opts = QuadOptions::new()
                .with_epsabs(tol)
                .with_epsrel(tol)
                .with_max_subintervals(2000)
                .with_max_evaluations(400_000);
            // The centres of the review's probe (cub_peaks.rs), at least six widths inside.
            let mut s = 12345u64;
            let (mut beyond, mut whole) = (0, 0);
            for _ in 0..count {
                let c: Vec<f64> = (0..d)
                    .map(|_| {
                        s = s
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                        6.0 * w + u * (1.0 - 12.0 * w)
                    })
                    .collect();
                let f = |x: &[f64]| {
                    (-x.iter()
                        .zip(&c)
                        .map(|(x, c)| ((x - c) / w).powi(2))
                        .sum::<f64>())
                    .exp()
                };
                let r = cubature(f, &vec![0.0; d], &vec![1.0; d], &opts).unwrap();
                if !honest(&r, exact) {
                    beyond += 1;
                }
                // In pieces: three along each axis, the middle one c ± 5w.
                let (mut value, mut error, mut converged) = (0.0, 0.0, true);
                for k in 0..3usize.pow(d as u32) {
                    let (mut lo, mut hi) = (vec![0.0; d], vec![0.0; d]);
                    for (i, &ci) in c.iter().enumerate() {
                        let cuts = [0.0, ci - 5.0 * w, ci + 5.0 * w, 1.0];
                        let j = k / 3usize.pow(i as u32) % 3;
                        (lo[i], hi[i]) = (cuts[j], cuts[j + 1]);
                    }
                    let p = cubature(f, &lo, &hi, &opts).unwrap();
                    value += p.value;
                    error += p.error;
                    converged &= p.converged;
                }
                let pieces = QuadResult {
                    value,
                    error,
                    evaluations: 0,
                    converged,
                };
                assert!(
                    converged && honest(&pieces, exact),
                    "d {d} w {w} tol {tol:e} c {c:?} in pieces: {pieces:?}, exact {exact:e}"
                );
                whole += usize::from(r.converged);
            }
            assert!(
                beyond <= bound,
                "d {d} w {w} tol {tol:e}: {beyond} of {count} beyond their estimates, documented {bound}"
            );
            assert_eq!(whole, count, "d {d} w {w} tol {tol:e}: not all converged");
        }
    }
}
