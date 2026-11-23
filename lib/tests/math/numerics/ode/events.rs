//! Events: their times against closed forms, their directions, terminal events, several in one
//! step, the zero at the start, a function that touches zero, and one that is not a number.

use super::problems::{col, miri_or, tol};
use a::math::matrix::Matrix;
use a::math::numerics::{
    DormandPrince5, DormandPrince853, Event, EventDirection, OdeFailure, OdeOptions, OdeStatus,
    RadauIia, Rk4, solve_ivp, solve_ivp_with,
};
use core::f64::consts::PI;

const G: f64 = 9.81;

/// A body in free fall: height and velocity.
fn falling(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = y[(1, 0)];
    dy[(1, 0)] = -G;
}

/// `(sin t, cos t)` from `(0, 1)`.
fn oscillator(_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = y[(1, 0)];
    dy[(1, 0)] = -y[(0, 0)];
}

/// Nothing moves: the events of these tests are functions of the time alone.
fn still(_t: f64, _y: &Matrix<f64>, dy: &mut Matrix<f64>) {
    dy[(0, 0)] = 0.0;
}

fn ground() -> Event<'static, f64> {
    Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])
        .with_direction(EventDirection::Falling)
        .with_terminal(true)
}

#[test]
fn a_falling_body_stops_on_the_ground() {
    let y0 = col(&[10.0, 0.0]);
    for solution in [
        solve_ivp(
            falling,
            &y0,
            0.0,
            100.0,
            DormandPrince5::new(),
            &mut [ground()],
        ),
        solve_ivp(
            falling,
            &y0,
            0.0,
            100.0,
            DormandPrince853::new(),
            &mut [ground()],
        ),
        solve_ivp(falling, &y0, 0.0, 100.0, Rk4::new(), &mut [ground()]),
        solve_ivp(falling, &y0, 0.0, 100.0, RadauIia::new(), &mut [ground()]),
    ] {
        let impact = (2.0 * 10.0 / G).sqrt();
        assert_eq!(solution.status, OdeStatus::Terminated);
        assert_eq!(solution.events.len(), 1);
        let event = &solution.events[0];
        assert_eq!((event.index, event.rising), (0, false));
        // The solution is a parabola, which every method follows to rounding.
        assert!(
            (event.t - impact).abs() < 1e-11,
            "{} against {impact}",
            event.t
        );
        assert_eq!(solution.t, event.t);
        assert_eq!(solution.y, event.y);
        // The reported point is where the height is zero or already below it.
        assert!(event.y[(0, 0)] <= 0.0 && event.y[(0, 0)] > -1e-12);
        assert!((event.y[(1, 0)] + G * impact).abs() < 1e-10);
        // The solution ends there.
        let dense = solution.dense.as_ref().unwrap();
        assert_eq!(dense.t1(), event.t);
        assert!(solution.eval(event.t + 1e-6).is_none());
    }
}

#[test]
fn a_bouncing_ball_bounces_on_time() {
    // Restarted from each impact with a tenth of the speed lost: the impacts come at
    // t₁ = √(2h/g) and then 2·eᵏ·v₁/g apart.
    let e = 0.9;
    let t_first = (2.0 * 10.0 / G).sqrt();
    let v_first = G * t_first;
    let (mut t, mut y) = (0.0, col(&[10.0, 0.0]));
    let mut want = t_first;
    let opts = tol::<f64>(1e-10);
    for k in 0..10 {
        let solution = solve_ivp_with(
            falling,
            &y,
            t,
            100.0,
            DormandPrince5::new(),
            &mut [ground()],
            &opts,
        );
        assert_eq!(solution.status, OdeStatus::Terminated, "bounce {k}");
        assert_eq!(
            solution.events.len(),
            1,
            "bounce {k}: {:?}",
            solution.events
        );
        t = solution.t;
        assert!(
            (t - want).abs() < 1e-10 * (k + 1) as f64,
            "bounce {k}: {t} against {want}"
        );
        y = solution.y.clone();
        y[(1, 0)] *= -e;
        want += 2.0 * e.powi(k + 1) * v_first / G;
    }
}

#[test]
fn the_zero_crossings_of_an_oscillator_are_counted_over_many_periods() {
    // sin t through a hundred periods and a quarter (two under Miri): zeros at kπ, k = 1 … 200;
    // the one at the start is not an event.
    let periods = miri_or(2, 100);
    let y0 = col(&[0.0, 1.0]);
    let end = 2.0 * PI * f64::from(periods) + 0.5 * PI;
    let opts = tol::<f64>(1e-12).with_dense_output(false);
    let either = Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]);
    let rising =
        Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]).with_direction(EventDirection::Rising);
    let falling =
        Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]).with_direction(EventDirection::Falling);
    let mut events = [either, rising, falling];
    let solution = solve_ivp_with(
        oscillator,
        &y0,
        0.0,
        end,
        DormandPrince853::new(),
        &mut events,
        &opts,
    );
    assert_eq!(solution.status, OdeStatus::Finished);
    let of = |i: usize| {
        solution
            .events
            .iter()
            .filter(|p| p.index == i)
            .collect::<Vec<_>>()
    };
    let (all, up, down) = (of(0), of(1), of(2));
    let n = periods as usize;
    assert_eq!((all.len(), up.len(), down.len()), (2 * n, n, n));
    for (k, p) in all.iter().enumerate() {
        let exact = PI * (k + 1) as f64;
        assert!((p.t - exact).abs() < 1e-8, "{k}: {} against {exact}", p.t);
        // Falling through π, 3π, …, rising through 2π, 4π, ….
        assert_eq!(p.rising, k % 2 == 1);
    }
    for (k, p) in up.iter().enumerate() {
        assert!((p.t - 2.0 * PI * (k + 1) as f64).abs() < 1e-8 && p.rising);
    }
    for (k, p) in down.iter().enumerate() {
        assert!((p.t - PI * (2 * k + 1) as f64).abs() < 1e-8 && !p.rising);
    }
    // In the order of their times, the earlier index first at a tie.
    for pair in solution.events.windows(2) {
        assert!(pair[0].t < pair[1].t || (pair[0].t == pair[1].t && pair[0].index < pair[1].index));
    }
}

#[test]
fn backwards_in_time_the_directions_are_those_of_the_integration() {
    // sin t from 0 down to −20.5π (−4.5π under Miri): going backwards it is negative at first,
    // so it rises through −π, falls through −2π, and so on.
    let zeros = miri_or(4, 20);
    let y0 = col(&[0.0, 1.0]);
    let end = -(f64::from(zeros) * PI + 0.5 * PI);
    let tolerance = miri_or(1e-7, 1e-10);
    let mut events = [Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])];
    let solution = solve_ivp_with(
        oscillator,
        &y0,
        0.0,
        end,
        DormandPrince5::new(),
        &mut events,
        &tol(tolerance),
    );
    assert_eq!(solution.events.len(), zeros as usize);
    for (k, p) in solution.events.iter().enumerate() {
        assert!(
            (p.t + PI * (k + 1) as f64).abs() < 1e3 * tolerance,
            "{k}: {}",
            p.t
        );
        assert_eq!(p.rising, k % 2 == 0);
    }
    // And a terminal one stops it.
    let mut events = [Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])
        .with_direction(EventDirection::Falling)
        .with_terminal(true)];
    let solution = solve_ivp_with(
        oscillator,
        &y0,
        0.0,
        end,
        DormandPrince5::new(),
        &mut events,
        &tol(tolerance),
    );
    assert_eq!(solution.status, OdeStatus::Terminated);
    assert!(
        (solution.t + 2.0 * PI).abs() < 1e2 * tolerance,
        "{}",
        solution.t
    );
}

#[test]
fn an_event_that_never_happens_is_never_reported() {
    let y0 = col(&[0.0, 1.0]);
    let mut events = [Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)] + 2.0).with_terminal(true)];
    let solution = solve_ivp(
        oscillator,
        &y0,
        0.0,
        miri_or(10.0, 50.0),
        DormandPrince5::new(),
        &mut events,
    );
    assert_eq!(solution.status, OdeStatus::Finished);
    assert!(solution.events.is_empty());
    assert_eq!(solution.t, miri_or(10.0, 50.0));
}

#[test]
fn a_function_that_touches_zero_without_crossing_is_not_an_event() {
    // (y − ½)² for y = e⁻ᵗ touches zero at ln 2 and never changes sign.
    let decay = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    let mut events = [Event::new(|_t: f64, y: &Matrix<f64>| {
        (y[(0, 0)] - 0.5).powi(2)
    })];
    let solution = solve_ivp(
        decay,
        &col(&[1.0]),
        0.0,
        3.0,
        DormandPrince5::new(),
        &mut events,
    );
    assert!(solution.events.is_empty());
    // Unless a step ends on the zero exactly: then it is reported once, as an arrival from the
    // side it came from. Four equal steps end on t = 1.
    let mut events = [Event::new(|t: f64, _y: &Matrix<f64>| (t - 1.0) * (t - 1.0))];
    let solution = solve_ivp(still, &col(&[0.0]), 0.0, 2.0, Rk4::fixed(4), &mut events);
    assert_eq!(solution.events.len(), 1);
    assert_eq!(
        (solution.events[0].t, solution.events[0].rising),
        (1.0, false)
    );
}

#[test]
fn a_zero_at_the_end_of_a_step_is_one_event() {
    // t − 1 is zero at the end of the second of four equal steps, and positive after.
    let mut events = [Event::new(|t: f64, _y: &Matrix<f64>| t - 1.0)];
    let solution = solve_ivp(still, &col(&[0.0]), 0.0, 2.0, Rk4::fixed(4), &mut events);
    assert_eq!(solution.events.len(), 1);
    assert_eq!(
        (solution.events[0].t, solution.events[0].rising),
        (1.0, true)
    );
}

#[test]
fn an_event_at_the_start_is_not_reported_but_one_in_the_first_step_is() {
    // t·(½ − t) is zero at the start and again at ½, inside the one step of the integration.
    // Leaving zero at the start is no event; crossing it at ½ is.
    let mut events = [Event::new(|t: f64, _y: &Matrix<f64>| t * (0.5 - t))];
    let solution = solve_ivp(still, &col(&[0.0]), 0.0, 1.0, Rk4::fixed(1), &mut events);
    assert_eq!(solution.events.len(), 1, "{:?}", solution.events);
    let event = &solution.events[0];
    assert!((event.t - 0.5).abs() < 1e-15 && !event.rising, "{event:?}");
    // The same function backwards from 0 is negative at once, and does not cross again.
    let solution = solve_ivp(still, &col(&[0.0]), 0.0, -1.0, Rk4::fixed(1), &mut events);
    assert!(solution.events.is_empty());
    // And a function that stays at zero for a while is no event at all.
    let mut events = [Event::new(
        |t: f64, _y: &Matrix<f64>| if t < 0.3 { 0.0 } else { t - 0.3 },
    )];
    let solution = solve_ivp(still, &col(&[0.0]), 0.0, 1.0, Rk4::fixed(1), &mut events);
    assert!(solution.events.is_empty());
}

#[test]
fn of_several_events_in_one_step_the_first_terminal_one_wins() {
    // One step from 0 to 1 holds all four zeros; the terminal ones are at 0.7 and 0.5.
    let at = |c: f64| move |t: f64, _y: &Matrix<f64>| t - c;
    let mut events = [
        Event::new(at(0.7)).with_terminal(true),
        Event::new(at(0.3)),
        Event::new(at(0.5)).with_terminal(true),
        Event::new(at(0.9)),
        Event::new(at(0.5)),
    ];
    let solution = solve_ivp(still, &col(&[0.0]), 0.0, 1.0, Rk4::fixed(1), &mut events);
    assert_eq!(solution.status, OdeStatus::Terminated);
    let found: Vec<(usize, f64)> = solution.events.iter().map(|p| (p.index, p.t)).collect();
    assert_eq!(found.len(), 3, "{found:?}");
    assert_eq!((found[0].0, found[1].0, found[2].0), (1, 2, 4));
    for ((_, t), want) in found.iter().zip([0.3, 0.5, 0.5]) {
        assert!((t - want).abs() < 1e-15, "{t} against {want}");
    }
    assert_eq!(solution.t, found[1].1);
    assert_eq!(solution.dense.as_ref().unwrap().t1(), solution.t);
}

#[test]
fn the_time_of_an_event_is_as_accurate_as_the_solution() {
    // The zeros of sin t at π, 2π, 3π, located on the dense output: their error follows the
    // tolerance of the integration.
    let y0 = col(&[0.0, 1.0]);
    for &t in miri_or(&[1e-4, 1e-6][..], &[1e-4, 1e-6, 1e-8, 1e-10][..]) {
        for method in 0..2 {
            let mut events = [Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])];
            let opts = tol::<f64>(t);
            let solution = if method == 0 {
                solve_ivp_with(
                    oscillator,
                    &y0,
                    0.0,
                    10.0,
                    DormandPrince5::new(),
                    &mut events,
                    &opts,
                )
            } else {
                solve_ivp_with(
                    oscillator,
                    &y0,
                    0.0,
                    10.0,
                    DormandPrince853::new(),
                    &mut events,
                    &opts,
                )
            };
            assert_eq!(solution.events.len(), 3);
            for (k, p) in solution.events.iter().enumerate() {
                let exact = PI * (k + 1) as f64;
                assert!(
                    (p.t - exact).abs() < 30.0 * t,
                    "{method} at {t:e}: {} against {exact}",
                    p.t
                );
                // And the state there is that of the solution, which is zero in its first
                // component to the same accuracy.
                assert!(p.y[(0, 0)].abs() < 30.0 * t);
            }
        }
    }
}

#[test]
fn a_terminal_event_is_not_found_again_from_its_own_point() {
    // Restarted from the impact, still falling: the height is at or below zero already.
    let y0 = col(&[10.0, 0.0]);
    let mut either = [Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]).with_terminal(true)];
    let first = solve_ivp(falling, &y0, 0.0, 100.0, DormandPrince5::new(), &mut either);
    assert_eq!(first.status, OdeStatus::Terminated);
    let again = solve_ivp(
        falling,
        &first.y,
        first.t,
        first.t + 1.0,
        DormandPrince5::new(),
        &mut either,
    );
    assert_eq!(again.status, OdeStatus::Finished);
    assert!(again.events.is_empty());
}

#[test]
fn an_event_function_that_is_not_a_number_ends_the_integration() {
    let y0 = col(&[0.0, 1.0]);
    let mut events = [Event::new(
        |t: f64, y: &Matrix<f64>| if t > 2.0 { f64::NAN } else { y[(0, 0)] },
    )];
    let solution = solve_ivp(
        oscillator,
        &y0,
        0.0,
        10.0,
        DormandPrince5::new(),
        &mut events,
    );
    assert_eq!(
        solution.status,
        OdeStatus::Failed(OdeFailure::EventNotFinite)
    );
    assert!(!solution.status.is_success());
    // It ended where it had reached, before 2 plus a step, with the event at π not yet passed.
    assert!(solution.t > 2.0 && solution.t < 3.0, "{}", solution.t);
    // NaN at the very start.
    let mut events = [Event::new(|_t: f64, _y: &Matrix<f64>| f64::NAN)];
    let solution = solve_ivp(
        oscillator,
        &y0,
        0.0,
        10.0,
        DormandPrince5::new(),
        &mut events,
    );
    assert_eq!(
        solution.status,
        OdeStatus::Failed(OdeFailure::EventNotFinite)
    );
    assert_eq!(solution.t, 0.0);
}

#[test]
fn events_work_without_the_dense_output_and_in_f32() {
    let y0 = col::<f32>(&[10.0, 0.0]);
    let f = |_t: f32, y: &Matrix<f32>, dy: &mut Matrix<f32>| {
        dy[(0, 0)] = y[(1, 0)];
        dy[(1, 0)] = -9.81;
    };
    let ground = Event::new(|_t: f32, y: &Matrix<f32>| y[(0, 0)])
        .with_direction(EventDirection::Falling)
        .with_terminal(true);
    let opts = OdeOptions::<f32>::new()
        .with_tolerances(1e-5, 1e-6)
        .with_dense_output(false);
    let solution = solve_ivp_with(
        f,
        &y0,
        0.0,
        100.0,
        DormandPrince853::new(),
        &mut [ground],
        &opts,
    );
    assert_eq!(solution.status, OdeStatus::Terminated);
    assert!(solution.dense.is_none());
    // As accurate as the f32 solution: its height near the ground carries the rounding of the
    // steps before, some 1e-6, at a speed of 14.
    let impact = (2.0f64 * 10.0 / 9.81).sqrt();
    assert!(
        (f64::from(solution.t) - impact).abs() < 1e-5,
        "{}",
        solution.t
    );
}
