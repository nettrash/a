//! Monte Carlo and quasi-Monte Carlo: the generator against the published algorithm
//! (`data/gen_xoshiro.py`), the Sobol' points against SciPy's (`data/gen_sobol.py`), the
//! integrals the old sequence got wrong (audit n_quad-1), the coverage of the reported standard
//! errors, `f32` (n_quad-12) and the contract (n_quad-13).

use a::math::numerics::{
    MonteCarloOptions, SOBOL_MAX_DIMENSION, Sobol, monte_carlo, monte_carlo_with,
    quasi_monte_carlo, quasi_monte_carlo_with,
};
use std::cell::RefCell;

/// An integrand of the tables of cases.
type G = fn(&[f64]) -> f64;

const SOBOL_REF: &str = include_str!("data/sobol_ref.txt");

thread_local! {
    static SEEN: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

#[test]
fn the_generator_is_xoshiro256_starstar_seeded_by_splitmix64() {
    // The points of plain Monte Carlo on [0, 1] are (k + ½)/2⁵², k the 52 leading bits of each
    // output: the first eight for the seeds 1 and 42, from gen_xoshiro.py.
    let want: [(u64, [u64; 8]); 2] = [
        (
            1,
            [
                3_165_678_505_884_785,
                2_343_838_167_626_596,
                2_585_542_216_680_100,
                1_762_387_346_335_338,
                3_139_812_457_030_195,
                646_590_971_183_066,
                319_959_208_615_761,
                1_716_702_132_075_294,
            ],
        ),
        (
            42,
            [
                377_685_245_215_468,
                1_706_775_315_665_171,
                3_062_643_252_502_089,
                4_164_446_803_999_542,
                4_466_687_738_785_477,
                3_466_598_347_184_021,
                3_239_252_663_314_013,
                3_828_097_711_259_232,
            ],
        ),
    ];
    for (seed, outputs) in want {
        SEEN.with(|s| s.borrow_mut().clear());
        let opts = MonteCarloOptions::new().with_samples(8).with_seed(seed);
        let r = monte_carlo_with(
            |x: &[f64]| {
                SEEN.with(|s| s.borrow_mut().push(x[0]));
                x[0]
            },
            &[0.0],
            &[1.0],
            &opts,
        )
        .unwrap();
        assert_eq!(r.evaluations, 8);
        let seen = SEEN.with(|s| s.borrow().clone());
        let got: Vec<u64> = seen
            .iter()
            .map(|&x| (x * 2f64.powi(52) - 0.5) as u64)
            .collect();
        assert_eq!(got, outputs, "seed {seed}");
    }
}

/// The reference lines `tag d …` of the Sobol' data.
fn sobol_ref(tag: &str) -> Vec<Vec<u64>> {
    SOBOL_REF
        .lines()
        .filter(|l| l.split_whitespace().next() == Some(tag))
        .map(|l| {
            l.split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn the_sobol_points_are_scipys_point_for_point() {
    // scipy.stats.qmc.Sobol(d, scramble=False, bits=32): the first 64 points in full in 1 to 8
    // dimensions, and in all 1024 dimensions the point of index 1000 and a checksum of the first
    // 4096 points, Σ (i + 1) xᵢ over the 32-bit integers of each coordinate.
    for row in sobol_ref("full") {
        let (d, i) = (row[0] as usize, row[1] as usize);
        let mut s = Sobol::new(d).unwrap();
        let mut p = vec![0.0; d];
        for _ in 0..=i {
            assert!(s.next_point(&mut p));
        }
        let got: Vec<u64> = p.iter().map(|&x| (x * 2f64.powi(32)) as u64).collect();
        assert_eq!(got, row[2..], "d = {d}, point {i}");
    }
    let count = if cfg!(miri) { 16 } else { 4096 };
    let mut s = Sobol::new(SOBOL_MAX_DIMENSION).unwrap();
    let mut p = vec![0.0; SOBOL_MAX_DIMENSION];
    let mut sums = vec![0u64; SOBOL_MAX_DIMENSION];
    let mut at_1000 = Vec::new();
    for i in 0..count {
        assert!(s.next_point(&mut p));
        for (sum, &x) in sums.iter_mut().zip(&p) {
            let v = (x * 2f64.powi(32)) as u64;
            *sum = sum.wrapping_add((i as u64 + 1).wrapping_mul(v));
        }
        if i == 1000 {
            at_1000 = p.iter().map(|&x| (x * 2f64.powi(32)) as u64).collect();
        }
    }
    if count == 4096 {
        for row in sobol_ref("point1000") {
            assert_eq!(
                at_1000[row[0] as usize - 1],
                row[1],
                "point 1000, dimension {}",
                row[0]
            );
        }
        for row in sobol_ref("checksum") {
            assert_eq!(
                sums[row[0] as usize - 1],
                row[1],
                "checksum of dimension {}",
                row[0]
            );
        }
    }
}

#[test]
fn every_coordinate_of_the_sobol_sequence_is_stratified() {
    // Audit n_quad-1: the old coordinates 2, 4, 17 and 32 took 8, 16, 64 and 128 values over
    // 65536 points. In the Sobol' sequence the first 2^m values of every coordinate are the
    // 2^m multiples of 2^−m, each once.
    let m = if cfg!(miri) { 4 } else { 10 };
    let mut s = Sobol::new(SOBOL_MAX_DIMENSION).unwrap();
    let mut p = vec![0.0; SOBOL_MAX_DIMENSION];
    let mut seen = vec![vec![false; 1 << m]; SOBOL_MAX_DIMENSION];
    for _ in 0..1 << m {
        assert!(s.next_point(&mut p));
        for (cells, &x) in seen.iter_mut().zip(&p) {
            cells[(x * f64::from(1u32 << m)) as usize] = true;
        }
    }
    for (j, cells) in seen.iter().enumerate() {
        assert!(cells.iter().all(|&c| c), "coordinate {j} is not stratified");
    }
}

#[test]
fn the_quasi_random_rule_gets_right_what_the_old_sequence_got_wrong() {
    // Audit n_quad-1: ∫ xyz over the unit cube came out 0.10937 for 1/8 at every N, the simplex
    // x + y + z ≤ 1 0.1992 for 1/6, ∏ xᵢ in six dimensions 0.02427 for 1/64, and ∏ (xᵢ + ½) in
    // forty 91.8 for 1. Each is now within its error estimate, and the estimate is far below
    // the old errors (measured: 1.2e-5, 3.1e-5, 2.4e-4, 3.5e-5, 9.4e-4 and 1.7e-2, the true
    // errors at most 1.5 times them).
    let n = if cfg!(miri) { 1 << 6 } else { 1 << 16 };
    let cases: [(&str, G, usize, f64, f64); 6] = [
        ("z", |x| x[2], 3, 0.5, 1e-4),
        ("xyz", |x| x[0] * x[1] * x[2], 3, 0.125, 1e-4),
        (
            "the simplex",
            |x| if x[0] + x[1] + x[2] <= 1.0 { 1.0 } else { 0.0 },
            3,
            1.0 / 6.0,
            1e-3,
        ),
        ("∏ xᵢ in 6", |x| x.iter().product(), 6, 1.0 / 64.0, 2e-4),
        (
            "∏ (xᵢ + ½) in 10",
            |x| x.iter().map(|t| t + 0.5).product(),
            10,
            1.0,
            5e-3,
        ),
        (
            "∏ (xᵢ + ½) in 40",
            |x| x.iter().map(|t| t + 0.5).product(),
            40,
            1.0,
            5e-2,
        ),
    ];
    for (what, f, d, exact, size) in cases {
        let r = quasi_monte_carlo(f, &vec![0.0; d], &vec![1.0; d], n).unwrap();
        assert!(r.converged && r.evaluations == n, "{what}: {r:?}");
        let err = (r.value - exact).abs();
        assert!(err <= 4.0 * r.error, "{what}: {r:?}, exact {exact}");
        if !cfg!(miri) {
            assert!(
                r.error < size,
                "{what}: the error estimate {:e} is large",
                r.error
            );
        }
    }
    // The quasi-random points beat pseudo-random ones of the same number by far on a smooth
    // integrand: e^(x + y + z) over the cube.
    let f = |x: &[f64]| (x[0] + x[1] + x[2]).exp();
    let q = quasi_monte_carlo(f, &[0.0; 3], &[1.0; 3], n).unwrap();
    let p = monte_carlo(f, &[0.0; 3], &[1.0; 3], n).unwrap();
    assert!(q.error * 20.0 < p.error, "{q:?} {p:?}");
}

#[test]
fn the_reported_errors_cover_the_true_ones_as_often_as_they_should() {
    // Over 200 seeds: plain Monte Carlo's standard error is that of a normal mean (95.4% of the
    // errors within two of it, 99.7% within three), the quasi-random rule's that of Student's t
    // with 7 degrees of freedom for its 8 replications (91.4% and 98.0%). Measured with 400
    // seeds of 4096 samples: 95.5–96.3% and 99.5–99.8%; 91.2–92.2% and 97.8–98.8%.
    let seeds = if cfg!(miri) { 2 } else { 200 };
    let cases: [(G, usize, f64); 2] = [
        (|x| x.iter().map(|t| t + 0.5).product(), 5, 1.0),
        (
            |x| {
                let r2: f64 = x.iter().map(|t| (t - 0.5) * (t - 0.5)).sum();
                if r2 <= 0.25 { 1.0 } else { 0.0 }
            },
            4,
            core::f64::consts::PI * core::f64::consts::PI / 32.0,
        ),
    ];
    for (f, d, exact) in cases {
        for quasi in [false, true] {
            let (mut two, mut three) = (0, 0);
            for seed in 1..=seeds {
                let opts = MonteCarloOptions::new()
                    .with_samples(1 << 10)
                    .with_seed(seed);
                let (a, b) = (vec![0.0; d], vec![1.0; d]);
                let r = if quasi {
                    quasi_monte_carlo_with(f, &a, &b, &opts)
                } else {
                    monte_carlo_with(f, &a, &b, &opts)
                }
                .unwrap();
                let z = (r.value - exact).abs() / r.error;
                two += usize::from(z <= 2.0);
                three += usize::from(z <= 3.0);
            }
            if seeds == 200 {
                let (two, three) = (two as f64 / 200.0, three as f64 / 200.0);
                let (want_two, want_three) = if quasi { (0.87, 0.95) } else { (0.92, 0.98) };
                assert!(
                    two >= want_two && three >= want_three,
                    "d = {d}, quasi {quasi}: {two} within two, {three} within three"
                );
            }
        }
    }
}

#[test]
fn f32_integrands_are_summed_in_f64() {
    // Audit n_quad-12: the old sums were kept in f32, and ∫₀¹ 1 came out 0.25 at 2^26 samples.
    // Now the f32 and the f64 rules see the same points (rounded) and agree to the rounding of
    // f32, and the constant 1 is exactly 1.
    let n = if cfg!(miri) { 1 << 6 } else { 1 << 18 };
    let opts = MonteCarloOptions::new().with_samples(n).with_seed(3);
    let single = monte_carlo_with(|x: &[f32]| x[0] * x[1], &[0.0; 2], &[1.0; 2], &opts).unwrap();
    let double = monte_carlo_with(|x: &[f64]| x[0] * x[1], &[0.0; 2], &[1.0; 2], &opts).unwrap();
    assert!(
        (f64::from(single.value) - double.value).abs() <= 1e-6 * double.value,
        "{single:?} {double:?}"
    );
    let one = monte_carlo_with(|_: &[f32]| 1.0, &[0.0], &[1.0], &opts).unwrap();
    assert_eq!((one.value, one.error), (1.0, 0.0));
    let q = quasi_monte_carlo_with(|x: &[f32]| x[0] * x[1], &[0.0; 2], &[1.0; 2], &opts).unwrap();
    assert!(
        (f64::from(q.value) - 0.25).abs() <= 4.0 * f64::from(q.error) + 1e-7,
        "{q:?}"
    );
}

#[test]
fn the_monte_carlo_contract() {
    // Audit n_quad-13: a NaN integrand gave a NaN with the error 0, converged; no samples gave 0,
    // converged; 1/√x was ∞ from the corner x = 0.
    let opts = MonteCarloOptions::new().with_samples(64);
    for quasi in [false, true] {
        let run = |f: fn(&[f64]) -> f64, a: &[f64], b: &[f64], o: &MonteCarloOptions| {
            if quasi {
                quasi_monte_carlo_with(f, a, b, o)
            } else {
                monte_carlo_with(f, a, b, o)
            }
        };
        let r = run(|_| f64::NAN, &[0.0], &[1.0], &opts).unwrap();
        assert!(
            !r.converged && r.error.is_infinite(),
            "quasi {quasi}: {r:?}"
        );
        for samples in [0usize, 1] {
            assert!(run(|x| x[0], &[0.0], &[1.0], &opts.with_samples(samples)).is_none());
        }
        assert!(run(|x| x[0], &[f64::NAN], &[1.0], &opts).is_none());
        assert!(run(|x| x[0], &[0.0], &[f64::INFINITY], &opts).is_none());
        let r = run(|x| x[0], &[0.0, 2.0], &[1.0, 2.0], &opts).unwrap();
        assert!(r.converged && r.value == 0.0 && r.evaluations == 0);
        let forward = run(|x| x[0] * x[1], &[0.0, 0.0], &[1.0, 2.0], &opts).unwrap();
        let reversed = run(|x| x[0] * x[1], &[1.0, 0.0], &[0.0, 2.0], &opts).unwrap();
        assert_eq!(reversed.value, -forward.value);
        // The same seed gives the same result, another seed another.
        let again = run(|x| x[0] * x[1], &[0.0, 0.0], &[1.0, 2.0], &opts).unwrap();
        assert_eq!(again, forward);
        let other = run(
            |x| x[0] * x[1],
            &[0.0, 0.0],
            &[1.0, 2.0],
            &opts.with_seed(2),
        )
        .unwrap();
        assert_ne!(other.value, forward.value);
    }
    // Plain Monte Carlo never samples a face: 1/√x on [0, 1], and 1/√(1 − x) in f32 on [1, 2]
    // shifted, where the old rule's first point was the corner.
    let r = monte_carlo(|x: &[f64]| 1.0 / x[0].sqrt(), &[0.0], &[1.0], 1 << 12).unwrap();
    assert!(r.converged && r.value.is_finite());
    let r = monte_carlo(
        |x: &[f32]| 1.0 / (2.0 - x[0]).sqrt(),
        &[1.0],
        &[2.0],
        1 << 16,
    )
    .unwrap();
    assert!(r.converged && r.value.is_finite(), "{r:?}");
    // The quasi-random rule's own limits.
    assert!(
        quasi_monte_carlo_with(|x: &[f64]| x[0], &[0.0], &[1.0], &opts.with_replications(1))
            .is_none()
    );
    assert!(
        quasi_monte_carlo_with(|x: &[f64]| x[0], &[0.0], &[1.0], &opts.with_samples(7)).is_none()
    );
    let d = SOBOL_MAX_DIMENSION + 1;
    assert!(quasi_monte_carlo(|x: &[f64]| x[0], &vec![0.0; d], &vec![1.0; d], 64).is_none());
    assert!(Sobol::new(0).is_none() && Sobol::new(d).is_none());
}

#[test]
#[should_panic(expected = "different dimensions")]
fn a_monte_carlo_box_with_mismatched_corners_panics() {
    // Audit n_quad-13: the old rule indexed out of bounds, or truncated silently.
    let _ = monte_carlo(|x: &[f64]| x[0], &[0.0, 0.0], &[1.0], 64);
}

#[test]
#[should_panic(expected = "the dimension of the sequence")]
fn a_sobol_point_of_the_wrong_length_panics() {
    let mut s = Sobol::new(3).unwrap();
    let mut p = [0.0; 2];
    s.next_point(&mut p);
}
