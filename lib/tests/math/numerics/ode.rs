//! The initial value problems of [`a::math::numerics::ode`]: each method against a solution the
//! test knows in closed form, the orders of convergence it claims, and the stiff problems where an
//! explicit method cannot go.

use a::math::matrix::Matrix;
use a::math::numerics::{
    DormandPrince5, OdeMethod, RadauIia, Rk4, RkOptions, integrate, integrate_with, solve,
};

/// The options of a study of the methods at a step of their own choosing rather than the
/// controller's: no step is refused, and none is larger than the one asked for.
fn fixed(h: f64) -> RkOptions<f64> {
    RkOptions::new(1e30, 1e30)
        .with_h0(h)
        .with_hmax(h)
        .with_max_steps(1_000_000)
}

/// The steps a study refines a fixed step by, coarsest first.
fn steps() -> Vec<f64> {
    (0..5).map(|k| 0.4f64 / f64::from(1u32 << k)).collect()
}

/// y′ = −y², y(0) = 1, whose solution is 1/(1 + t). Nonlinear, so that no method does better than
/// its own order the way it does on a linear one.
fn square_decay(_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>) {
    out[(0, 0)] = -y[(0, 0)] * y[(0, 0)];
}

/// λ(y − sin kt) + k cos kt, whose solution is exactly sin kt: stiff at λ = −50, and richly
/// wiggling at k = 20, which is what keeps its error above the round-off of a f64 at the coarse
/// steps an implicit method takes.
fn manufactured(t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>) {
    let (lambda, k) = (-50.0, 20.0);
    out[(0, 0)] = lambda * (y[(0, 0)] - (k * t).sin()) + k * (k * t).cos();
}

/// The harmonic oscillator y₀′ = y₁, y₁′ = −y₀, at rest at one.
fn oscillator(_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>) {
    out[(0, 0)] = y[(1, 0)];
    out[(1, 0)] = -y[(0, 0)];
}

/// The successive quotients of a sequence of errors, which is the order of the method that made
/// them, as powers of two.
fn ratios(errors: &[f64]) -> Vec<f64> {
    errors
        .windows(2)
        .map(|pair| pair[0] / pair[1].max(f64::MIN_POSITIVE))
        .collect()
}

#[test]
fn the_explicit_methods_converge_at_their_order() {
    for (name, order, mut run) in [
        (
            "rk4",
            4,
            Box::new(|h: f64| {
                solve(
                    square_decay,
                    &Matrix::from_rows(&[[1.0]]),
                    0.0,
                    1.0,
                    Rk4::new(),
                    fixed(h),
                )
            }) as Box<dyn FnMut(f64) -> Option<Matrix<f64>>>,
        ),
        (
            "dp5",
            5,
            Box::new(|h: f64| {
                solve(
                    square_decay,
                    &Matrix::from_rows(&[[1.0]]),
                    0.0,
                    1.0,
                    DormandPrince5::new(),
                    fixed(h),
                )
            }) as Box<dyn FnMut(f64) -> Option<Matrix<f64>>>,
        ),
    ] {
        let errors: Vec<f64> = steps()
            .into_iter()
            .map(|h| {
                let y = run(h).unwrap_or_else(|| panic!("{name} gave up at h = {h}"));
                (y[(0, 0)] - 0.5).abs()
            })
            .collect();
        let ratios: Vec<f64> = ratios(&errors);
        // Rk4 returns the Richardson extrapolation of two half-steps, so it is measured here a
        // clear order above its own; the bound is the order it advertises, halved.
        let floor = (1u64 << (order - 1)) as f64;
        for (ratio, pair) in ratios.iter().zip(errors.windows(2)) {
            assert!(
                *ratio >= floor,
                "{name}: errors {pair:?} refine by only {ratio:.1}, short of 2^{}",
                order - 1
            );
        }
    }
}

#[test]
fn the_stiff_method_converges_at_its_order() {
    // The coarsest step, hλ = −20, is past where the corrector is worth starting; the four below
    // it are the measure.
    let errors: Vec<f64> = steps()
        .into_iter()
        .skip(1)
        .map(|h| {
            let y = solve(
                manufactured,
                &Matrix::from_rows(&[[0.0]]),
                0.0,
                1.0,
                RadauIia::new(),
                fixed(h),
            )
            .unwrap_or_else(|| panic!("radau gave up at h = {h}"));
            (y[(0, 0)] - (20.0f64).sin()).abs()
        })
        .collect();
    let ratios: Vec<f64> = ratios(&errors);
    for (ratio, pair) in ratios.iter().zip(errors.windows(2)) {
        assert!(
            *ratio >= 16.0,
            "radau: errors {pair:?} refine by only {ratio:.1}, short of 2⁴ — the collocation \
             system is not being solved"
        );
    }
}

#[test]
fn every_method_follows_the_harmonic_oscillator() {
    let want = [
        core::f64::consts::FRAC_PI_2.cos(),
        -core::f64::consts::FRAC_PI_2.sin(),
    ];
    let opts = RkOptions::new(1e-11, 1e-13);
    let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
    let quarter = core::f64::consts::FRAC_PI_2;
    let runs = [
        (
            "rk4",
            solve(oscillator, &y0, 0.0, quarter, Rk4::new(), opts),
        ),
        (
            "dp5",
            solve(oscillator, &y0, 0.0, quarter, DormandPrince5::new(), opts),
        ),
        (
            "radau",
            solve(oscillator, &y0, 0.0, quarter, RadauIia::new(), opts),
        ),
    ];
    for (name, got) in runs {
        let got = got.unwrap_or_else(|| panic!("{name} gave up on the oscillator"));
        for (i, want) in want.iter().enumerate() {
            assert!(
                (got[(i, 0)] - want).abs() < 1e-10,
                "{name} at π/2: {got:?} against {want:?}"
            );
        }
    }
}

#[test]
fn the_stiff_method_outlasts_the_explicit_one() {
    // y′ = λ(sin t − y) with λ = 1000: an explicit method is confined to a step of a few
    // thousandths for stability alone, however coarse the accuracy it was asked for.
    let lambda = 1000.0;
    let f = move |t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
        out[(0, 0)] = lambda * (t.sin() - y[(0, 0)]);
    };
    // The forced response, the transient e⁻¹⁰⁰⁰ᵗ having long since vanished.
    let want = (1e6 * 5.0f64.sin() - lambda * 5.0f64.cos()) / (1e6 + 1.0);
    let opts = RkOptions::new(1e-8, 1e-10);
    let y0 = Matrix::from_rows(&[[0.0]]);

    let stiff = integrate(f, &y0, 0.0, 5.0, RadauIia::new(), opts).unwrap();
    let explicit = integrate(f, &y0, 0.0, 5.0, DormandPrince5::new(), opts).unwrap();
    assert!(
        (stiff.y[(0, 0)] - want).abs() < 1e-6,
        "radau: {:.3e} against {want:.12}",
        stiff.y[(0, 0)]
    );
    assert!(
        (explicit.y[(0, 0)] - want).abs() < 1e-6,
        "dp5: {:.3e} against {want:.12}",
        explicit.y[(0, 0)]
    );
    assert!(
        stiff.steps * 20 < explicit.steps,
        "radau took {} steps against dp5's {}, which is not stiff enough to show anything",
        stiff.steps,
        explicit.steps
    );
    assert!(stiff.nfev >= RadauIia::<f64>::new().stages() * stiff.steps);
}

#[test]
fn the_stiff_method_solves_robertsons_problem() {
    // Robertson's chemical kinetics, the stiffness benchmark: 3e7 · 10⁷ a second, and the three
    // concentrations sum to one for all time.
    let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
        let (y1, y2, y3) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
        out[(0, 0)] = -0.04 * y1 + 1e4 * y2 * y3;
        out[(1, 0)] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2;
        out[(2, 0)] = 3e7 * y2 * y2;
    };
    let y0 = Matrix::from_rows(&[[1.0], [0.0], [0.0]]);
    let r = integrate(
        f,
        &y0,
        0.0,
        40.0,
        RadauIia::new(),
        RkOptions::new(1e-8, 1e-13),
    )
    .unwrap();
    let sum: f64 = r.y.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "the invariant drifted to {sum}");
    assert!(
        r.y.as_slice().iter().all(|v| *v >= 0.0),
        "negative: {:?}",
        r.y
    );
    for (got, want) in
        r.y.as_slice()
            .iter()
            .zip([0.715_827_07, 9.185_535_1e-6, 0.284_163_73])
    {
        assert!(
            (got - want).abs() <= 1e-7 * want.abs().max(1.0),
            "at t = 40: {got:.8e} against {want:.8e}"
        );
    }

    // The same problem defeats the explicit method however coarse the tolerance: it is confined to
    // a step of 1e−7 by the 3e7 rate, and gives up rather than report a truncated answer.
    let hopeless = integrate(
        f,
        &y0,
        0.0,
        40.0,
        DormandPrince5::new(),
        RkOptions::new(1e-4, 1e-8).with_max_steps(30_000),
    );
    assert!(hopeless.is_none(), "dp5 should not reach t = 40");
}

#[test]
fn an_analytic_jacobian_agrees_with_the_differenced_one() {
    let lambda = 1000.0;
    let f = move |t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
        out[(0, 0)] = lambda * (t.sin() - y[(0, 0)]);
    };
    let jac = |_t: f64, _y: &Matrix<f64>, _fy: &Matrix<f64>, out: &mut Matrix<f64>| {
        out[(0, 0)] = -lambda;
    };
    let opts = RkOptions::new(1e-8, 1e-10);
    let y0 = Matrix::from_rows(&[[0.0]]);
    let differenced = integrate(f, &y0, 0.0, 5.0, RadauIia::new(), opts).unwrap();
    let analytic = integrate_with(f, &y0, 0.0, 5.0, RadauIia::new(), opts, Some(jac)).unwrap();
    assert!(
        (differenced.y[(0, 0)] - analytic.y[(0, 0)]).abs() < 1e-8,
        "{:?} against {:?}",
        differenced.y[(0, 0)],
        analytic.y[(0, 0)]
    );
    assert_eq!(differenced.steps, analytic.steps);
}

#[test]
fn it_integrates_backwards_in_time() {
    let decay = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -y[(0, 0)];
    let r = integrate(
        decay,
        &Matrix::from_rows(&[[1.0f64.exp()]]),
        -1.0,
        0.0,
        DormandPrince5::new(),
        RkOptions::new(1e-12, 1e-14),
    )
    .unwrap();
    assert_eq!(r.t, 0.0);
    // y = e·e⁻ᵗ, which is one where it crosses the axis.
    assert!((r.y[(0, 0)] - 1.0).abs() < 1e-11, "{:?}", r.y);

    // Backwards over a stiff decay is a growing exponential, which is where it has to be accurate
    // rather than stable.
    let grow = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = 100.0 * y[(0, 0)];
    let back = integrate(
        grow,
        &Matrix::from_rows(&[[1.0]]),
        0.0,
        -0.05,
        RadauIia::new(),
        RkOptions::new(1e-10, 1e-12),
    )
    .unwrap();
    assert!(
        (back.y[(0, 0)] - (-5.0f64).exp()).abs() < 1e-9,
        "{:?}",
        back.y
    );
}

#[test]
fn it_lands_exactly_on_the_requested_time() {
    let decay = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -y[(0, 0)];
    for (t0, t1) in [(0.0, 1.0), (0.1, 3.3), (-2.0, 0.7), (0.0, -1.0)] {
        for method in ["rk4", "dp5", "radau"] {
            let opts = RkOptions::new(1e-9, 1e-11);
            let r = match method {
                "rk4" => integrate(
                    decay,
                    &Matrix::from_rows(&[[1.0]]),
                    t0,
                    t1,
                    Rk4::new(),
                    opts,
                ),
                "dp5" => integrate(
                    decay,
                    &Matrix::from_rows(&[[1.0]]),
                    t0,
                    t1,
                    DormandPrince5::new(),
                    opts,
                ),
                _ => integrate(
                    decay,
                    &Matrix::from_rows(&[[1.0]]),
                    t0,
                    t1,
                    RadauIia::new(),
                    opts,
                ),
            }
            .unwrap_or_else(|| panic!("{method} from {t0} to {t1}"));
            assert_eq!(r.t, t1, "{method} from {t0} to {t1}");
            assert!(r.steps >= 1);
        }
    }
}

#[test]
fn it_gives_up_rather_than_report_a_truncated_answer() {
    // Too few steps to get there: no answer, rather than the one it reached.
    let r = integrate(
        oscillator,
        &Matrix::from_rows(&[[1.0], [0.0]]),
        0.0,
        core::f64::consts::FRAC_PI_2,
        DormandPrince5::new(),
        RkOptions::new(1e-11, 1e-13).with_max_steps(1),
    );
    assert!(r.is_none(), "one step does not reach π/2");

    // A right-hand side that is not finite anywhere.
    let nan = |_t: f64, _y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = f64::NAN;
    let r = integrate(
        nan,
        &Matrix::from_rows(&[[1.0]]),
        0.0,
        1.0,
        DormandPrince5::new(),
        RkOptions::new(1e-8, 1e-10),
    );
    assert!(r.is_none(), "a NaN right-hand side is not an answer");
    let r = integrate(
        nan,
        &Matrix::from_rows(&[[1.0]]),
        0.0,
        1.0,
        RadauIia::new(),
        RkOptions::new(1e-8, 1e-10),
    );
    assert!(
        r.is_none(),
        "a NaN right-hand side is not an answer to the stiff method either"
    );

    // y′ = y², y(0) = 1 escapes to infinity at t = 1, well before the requested t = 2.
    let blow =
        |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = y[(0, 0)] * y[(0, 0)];
    for method in ["dp5", "radau"] {
        let r = match method {
            "dp5" => integrate(
                blow,
                &Matrix::from_rows(&[[1.0]]),
                0.0,
                2.0,
                DormandPrince5::new(),
                RkOptions::new(1e-8, 1e-10).with_max_steps(100_000),
            ),
            _ => integrate(
                blow,
                &Matrix::from_rows(&[[1.0]]),
                0.0,
                2.0,
                RadauIia::new(),
                RkOptions::new(1e-8, 1e-10).with_max_steps(100_000),
            ),
        };
        assert!(r.is_none(), "{method} integration through a blow-up");
    }
}

#[test]
fn it_integrates_in_f32() {
    let decay = |_t: f32, y: &Matrix<f32>, out: &mut Matrix<f32>| out[(0, 0)] = -y[(0, 0)];
    for method in ["dp5", "radau"] {
        let y = match method {
            "dp5" => solve(
                decay,
                &Matrix::from_rows(&[[1.0f32]]),
                0.0,
                1.0,
                DormandPrince5::new(),
                RkOptions::new(1e-6, 1e-8),
            ),
            _ => solve(
                decay,
                &Matrix::from_rows(&[[1.0f32]]),
                0.0,
                1.0,
                RadauIia::new(),
                RkOptions::new(1e-6, 1e-8),
            ),
        }
        .unwrap_or_else(|| panic!("{method} gave up in f32"));
        assert!(
            (y[(0, 0)] - (-1.0f32).exp()).abs() < 1e-5,
            "{method}: {} against {}",
            y[(0, 0)],
            (-1.0f32).exp()
        );
    }
}

#[test]
fn the_methods_report_their_order_and_their_stages() {
    assert_eq!(Rk4::<f64>::new().order(), 4);
    assert_eq!(DormandPrince5::<f64>::new().order(), 5);
    assert_eq!(RadauIia::<f64>::new().order(), 5);
    assert_eq!(Rk4::<f64>::new().stages(), 12);
    assert_eq!(DormandPrince5::<f64>::new().stages(), 7);
    assert_eq!(RadauIia::<f64>::new().stages(), 9);
    // A default method is the same method as a fresh one, which is all `Default` promises here.
    assert_eq!(
        RadauIia::<f64>::default().order(),
        RadauIia::<f64>::new().order()
    );
    assert_eq!(Rk4::<f64>::default().stages(), Rk4::<f64>::new().stages());
}

#[test]
fn the_evaluation_count_covers_the_stages() {
    let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
        out[(0, 0)] = y[(1, 0)];
        out[(1, 0)] = -y[(0, 0)];
    };
    let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
    let quarter = core::f64::consts::FRAC_PI_2;
    let opts = RkOptions::new(1e-9, 1e-11);
    for (name, r, stages) in [
        (
            "rk4",
            integrate(f, &y0, 0.0, quarter, Rk4::new(), opts).unwrap(),
            12,
        ),
        (
            "dp5",
            integrate(f, &y0, 0.0, quarter, DormandPrince5::new(), opts).unwrap(),
            7,
        ),
        (
            "radau",
            integrate(f, &y0, 0.0, quarter, RadauIia::new(), opts).unwrap(),
            9,
        ),
    ] {
        assert!(r.steps >= 1, "{name} took no steps");
        assert!(
            r.nfev >= stages * r.steps,
            "{name}: {} evaluations over {} steps, short of {} a step",
            r.nfev,
            r.steps,
            stages
        );
    }
}
