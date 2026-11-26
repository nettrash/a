//! Event detection: the times at which a function of the solution passes through zero.
//!
//! An [`Event`] is a function `g(t, y)` of the time and the state. After every accepted step the
//! integration compares the sign of `g` at the two ends of the step; where it has changed, the
//! zero is located on the method's continuous extension of that step — no further evaluation of
//! the right-hand side is spent on it — by the crate's [`brent`](crate::math::numerics::brent),
//! with a bisection behind it that holds whatever Brent's method does.
//!
//! What is found is the zero of `g` along the *continuous extension*, to within four units in
//! the last place of the time. The solution it approximates is only as good as the tolerances of
//! the integration, so the time of an event is accurate to about the error of the dense output
//! divided by the slope of `g`.

use super::dense::DenseStep;
use crate::math::matrix::Matrix;
use crate::math::numerics::{Real, RootOptions, brent_with};
use core::fmt;

/// Which way an event function has to pass through zero for the event to count, along the
/// direction of the integration.
///
/// # Examples
///
/// ```
/// use a::math::numerics::EventDirection;
///
/// assert_ne!(EventDirection::Rising, EventDirection::Falling);
/// assert_eq!(EventDirection::default(), EventDirection::Either);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventDirection {
    /// From negative to positive.
    Rising,
    /// From positive to negative.
    Falling,
    /// Either way.
    #[default]
    Either,
}

/// An event to watch for during an integration: a function `g(t, y)` whose zeros along the
/// solution are wanted, the direction it must cross zero in, and whether the integration stops
/// there.
///
/// The rules, which [`solve_ivp`](super::solve_ivp) applies after every accepted step:
///
/// * An event is a change of the sign of `g` between the two ends of a step. Two crossings
///   inside one step cancel and go unnoticed; where that could happen, limit the step with
///   [`OdeOptions::with_hmax`](super::OdeOptions::with_hmax).
/// * The direction is that of the integration: [`EventDirection::Rising`] is from negative to
///   positive as the integration proceeds, which for an integration backwards in time is a
///   function that decreases with `t`.
/// * The time reported is the first located point at which `g` is zero or has already taken its
///   new sign, so an integration restarted from a terminal event does not find it again.
/// * Leaving zero is not an event, and so a function that is exactly zero at the very start of
///   an integration is not reported there. The sign it leaves zero with is read on the
///   continuous extension a hundred units in the last place after the start of the step, and a
///   crossing later in the same step is found from there.
/// * A function that touches zero without crossing it has not changed sign, and is not an event
///   — unless a step happens to end where it is zero to the last bit, which is reported once, as
///   an arrival from the side it came from, since that is all that can be seen there.
/// * When several events fall in one step they are reported in the order of their times; a
///   terminal one ends the integration at its time, and those after it are dropped.
///
/// A function that returns NaN at the end of a step, or while its zero is being located, ends the
/// integration with [`OdeFailure::EventNotFinite`](super::OdeFailure::EventNotFinite).
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, Event, EventDirection, OdeStatus, solve_ivp};
///
/// // A body dropped from a height of 10 under g = 9.81: where is it when it hits the ground?
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -9.81;
/// };
/// let ground = Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])
///     .with_direction(EventDirection::Falling)
///     .with_terminal(true);
/// let y0 = Matrix::from_rows(&[[10.0], [0.0]]);
/// let solution = solve_ivp(f, &y0, 0.0, 100.0, DormandPrince5::new(), &mut [ground]);
/// assert_eq!(solution.status, OdeStatus::Terminated);
/// let impact = (2.0 * 10.0 / 9.81f64).sqrt();
/// assert!((solution.t - impact).abs() < 1e-12);
/// assert_eq!(solution.events.len(), 1);
/// assert!(!solution.events[0].rising && solution.events[0].y[(0, 0)].abs() < 1e-12);
/// ```
pub struct Event<'a, T> {
    /// The function whose zeros are the event.
    function: EventFunction<'a, T>,
    /// The direction the function must cross zero in.
    pub direction: EventDirection,
    /// Whether the integration ends at the first occurrence.
    pub terminal: bool,
}

impl<'a, T> Event<'a, T> {
    /// The event of `g(t, y)` passing through zero in either direction, which is recorded and
    /// does not stop the integration.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{Event, EventDirection};
    ///
    /// let event = Event::new(|t: f64, _y: &Matrix<f64>| t - 1.0);
    /// assert_eq!(event.direction, EventDirection::Either);
    /// assert!(!event.terminal);
    /// ```
    pub fn new(g: impl FnMut(T, &Matrix<T>) -> T + 'a) -> Self {
        Event {
            function: Box::new(g),
            direction: EventDirection::Either,
            terminal: false,
        }
    }

    /// `self`, counting only the crossings in `direction`.
    #[must_use]
    pub fn with_direction(mut self, direction: EventDirection) -> Self {
        self.direction = direction;
        self
    }

    /// `self`, ending the integration at its first occurrence if `terminal`.
    #[must_use]
    pub fn with_terminal(mut self, terminal: bool) -> Self {
        self.terminal = terminal;
        self
    }

    /// The value of the event function at `(t, y)`.
    pub(super) fn value(&mut self, t: T, y: &Matrix<T>) -> T {
        (self.function)(t, y)
    }

    /// Whether a crossing that rises, or falls, is one this event counts.
    pub(super) fn admits(&self, rising: bool) -> bool {
        match self.direction {
            EventDirection::Rising => rising,
            EventDirection::Falling => !rising,
            EventDirection::Either => true,
        }
    }
}

impl<T> fmt::Debug for Event<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event")
            .field("direction", &self.direction)
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

/// An occurrence of an event: which of the events it was, when, and the state there.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, Event, solve_ivp};
///
/// // cos t, as the first component of the harmonic oscillator, changes sign at π/2 and 3π/2.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)];
/// };
/// let zero = Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]);
/// let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
/// let solution = solve_ivp(f, &y0, 0.0, 6.0, DormandPrince853::new(), &mut [zero]);
/// let points = &solution.events;
/// assert_eq!(points.len(), 2);
/// assert!(points.iter().all(|p| p.index == 0));
/// assert!((points[0].t - core::f64::consts::FRAC_PI_2).abs() < 1e-5 && !points[0].rising);
/// assert!((points[1].t - 3.0 * core::f64::consts::FRAC_PI_2).abs() < 1e-5 && points[1].rising);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EventPoint<T> {
    /// The position of the event in the list the integration was given.
    pub index: usize,
    /// The time of the event.
    pub t: T,
    /// The state at that time, from the continuous extension of the step it fell in.
    pub y: Matrix<T>,
    /// Whether the function went from negative to positive there, in the direction of the
    /// integration.
    pub rising: bool,
}

/// `−1`, `0` or `1` as `v` is negative, zero or positive.
pub(super) fn sign<T: Real>(v: T) -> i8 {
    if v > T::ZERO {
        1
    } else if v < T::ZERO {
        -1
    } else {
        0
    }
}

/// The value of `event` at the time `t` of the step `step`, on its continuous extension; `None`
/// if it is NaN. `scratch` is a state to evaluate the continuous extension into.
pub(super) fn value_on<T: Real>(
    event: &mut Event<'_, T>,
    step: &DenseStep<T>,
    t: T,
    scratch: &mut Matrix<T>,
) -> Option<T> {
    step.eval_into(t, scratch.as_mut_slice());
    let v = event.value(t, scratch);
    (!v.is_nan()).then_some(v)
}

/// The time between `from` and the end of the step `step` at which `event` leaves the sign
/// `before` it has at `from`, the function having another sign, not zero, at the end: the first
/// located point where the function is zero or has changed sign, with the last point where it had
/// not within four units in the last place of the time. `from` is the start of the step, or a
/// point just after it where the function is not zero. `scratch` is a state to evaluate the
/// continuous extension into.
///
/// `None` if the function returned NaN on the way.
pub(super) fn locate<T: Real>(
    event: &mut Event<'_, T>,
    step: &DenseStep<T>,
    from: T,
    before: i8,
    scratch: &mut Matrix<T>,
) -> Option<T> {
    let (a, b) = (from, step.t1());
    // The bracket: the function has the old sign at `near` and is zero or of the new sign at
    // `far`. Every evaluation Brent's method makes tightens it, so that what is known at the end
    // does not depend on which of its points that method chose to return.
    let mut bracket = Bracket {
        near: a,
        far: b,
        undefined: false,
    };
    let mut probe = |t: T, bracket: &mut Bracket<T>| -> T {
        step.eval_into(t, scratch.as_mut_slice());
        let v = event.value(t, scratch);
        if v.is_nan() {
            bracket.undefined = true;
        } else if (bracket.near < t && t < bracket.far) || (bracket.far < t && t < bracket.near) {
            if sign(v) == before {
                bracket.near = t;
            } else {
                bracket.far = t;
            }
        }
        v
    };
    let tolerance = T::from_f64(4.0) * T::EPSILON * a.abs().max(b.abs());
    let options = RootOptions::new()
        .with_xtol(tolerance)
        .with_rtol(T::ZERO)
        .with_max_iter(200);
    // The answer is read from the bracket, not from the root returned.
    let _ = brent_with(a, b, |t| probe(t, &mut bracket), &options);
    // Brent's method stops when its own test is met, which need not be this one; bisection
    // finishes the bracket, and is all there is if that method gave up.
    loop {
        if bracket.undefined {
            return None;
        }
        if (bracket.far - bracket.near).abs() <= tolerance {
            return Some(bracket.far);
        }
        let middle = bracket.near + (bracket.far - bracket.near) * T::HALF;
        if middle == bracket.near || middle == bracket.far {
            return Some(bracket.far);
        }
        probe(middle, &mut bracket);
    }
}

/// The function of an [`Event`], boxed so that events of different closures share a list.
type EventFunction<'a, T> = Box<dyn FnMut(T, &Matrix<T>) -> T + 'a>;

/// What is known of a zero while it is being located.
struct Bracket<T> {
    /// The latest time at which the function still had its old sign.
    near: T,
    /// The earliest time at which it was zero or had its new one.
    far: T,
    /// Whether the function returned NaN.
    undefined: bool,
}
