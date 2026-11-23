//! The Krylov solvers `cg`, `minres`, `gmres` and `bicgstab`: against scipy on the problems of
//! `data/iter_scipy.txt` (made by `data/gen_iter.py`, which builds the same systems), against the
//! defects the audit found (`p_iter-1` to `p_iter-13`), and on the edge cases of the contract.

use super::common::{Num, Rng, each_type, magnitude, size};
use super::precond::hermitian;
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;
use a::math::sparse::{
    FnOperator, Ic0, Ilu0, Jacobi, LinearOperator, Preconditioner, SolveOptions, SolveResult,
    SparseMatrix, StopReason, bicgstab, bicgstab_with, cg, cg_with, gmres, gmres_with, minres,
    minres_with,
};
use std::sync::mpsc;
use std::time::Duration;

// ---------------------------------------------------------------------------------------------
// The problems, as data/gen_iter.py builds them
// ---------------------------------------------------------------------------------------------

/// The element `re + im·i`, the imaginary part dropped for a real type.
pub(super) fn c<T: Num>(re: f64, im: f64) -> T {
    T::from_parts(T::Real::from_f64(re), T::Real::from_f64(im))
}

/// The element of a grid operator at the node `(r, c)`, as a pair of parts.
type Coefficient<'a> = &'a dyn Fn(usize, usize) -> (f64, f64);

/// The 5-point operator on a `k × k` grid, node `(r, c)` numbered `r·k + c`: the diagonal and
/// the elements towards the east, west, north and south neighbours, as functions of the node.
fn grid<T: Num>(
    k: usize,
    [diagonal, east, west, north, south]: [Coefficient; 5],
) -> SparseMatrix<T> {
    let at = |f: Coefficient, r: usize, col: usize| {
        let (re, im) = f(r, col);
        c::<T>(re, im)
    };
    let mut e = Vec::new();
    for r in 0..k {
        for col in 0..k {
            let i = r * k + col;
            e.push((i, i, at(diagonal, r, col)));
            if col + 1 < k {
                e.push((i, i + 1, at(east, r, col)));
            }
            if col > 0 {
                e.push((i, i - 1, at(west, r, col)));
            }
            if r + 1 < k {
                e.push((i, i + k, at(north, r, col)));
            }
            if r > 0 {
                e.push((i, i - k, at(south, r, col)));
            }
        }
    }
    SparseMatrix::from_coo(k * k, k * k, &e)
}

/// The tridiagonal matrix of the three constants.
fn tridiagonal<T: Num>(n: usize, lower: f64, diagonal: f64, upper: f64) -> SparseMatrix<T> {
    let m = n.saturating_sub(1);
    SparseMatrix::tridiagonal(
        &vec![T::real(lower); m],
        &vec![T::real(diagonal); n],
        &vec![T::real(upper); m],
    )
}

/// The problem `name` of `gen_iter.py` (a family and its size, `lap2d12`), with its right-hand
/// side and its starting guess.
pub(super) fn problem<T: Num>(name: &str) -> (SparseMatrix<T>, Vec<T>, Vec<T>) {
    let split = name.trim_end_matches(|ch: char| ch.is_ascii_digit()).len();
    let (family, k) = (&name[..split], name[split..].parse::<usize>().unwrap_or(0));
    let constant = |x: f64| move |_: usize, _: usize| (x, 0.0);
    let a: SparseMatrix<T> = match family {
        "lap1d" => tridiagonal(k, -1.0, 2.0, -1.0),
        "lap2d" => grid(
            k,
            [
                &constant(4.0),
                &constant(-1.0),
                &constant(-1.0),
                &constant(-1.0),
                &constant(-1.0),
            ],
        ),
        "scaled2d" => {
            let d = |i: usize| 2f64.powi((i % 4) as i32 - 1);
            let lap: SparseMatrix<T> = problem::<T>(&format!("lap2d{k}")).0;
            lap.map(|i, j, x| x.scale(T::Real::from_f64(d(i) * d(j))))
        }
        "shifted2d" => grid(
            k,
            [
                &constant(1.5),
                &constant(-1.0),
                &constant(-1.0),
                &constant(-1.0),
                &constant(-1.0),
            ],
        ),
        "saddle" => {
            let h = k / 2;
            let mut e: Vec<(usize, usize, T)> = Vec::new();
            for i in 0..k {
                e.push((i, i, T::real(2.0)));
                if i + 1 < k {
                    e.push((i, i + 1, T::real(-1.0)));
                    e.push((i + 1, i, T::real(-1.0)));
                }
            }
            for i in 0..h {
                for j in [2 * i, 2 * i + 1] {
                    e.push((k + i, j, T::real(1.0)));
                    e.push((j, k + i, T::real(1.0)));
                }
            }
            SparseMatrix::from_coo(k + h, k + h, &e)
        }
        "neumann2d" => {
            let degree = move |r: usize, col: usize| {
                let d = (col + 1 < k) as u8 + (col > 0) as u8 + (r + 1 < k) as u8 + (r > 0) as u8;
                (f64::from(d), 0.0)
            };
            grid(
                k,
                [
                    &degree,
                    &constant(-1.0),
                    &constant(-1.0),
                    &constant(-1.0),
                    &constant(-1.0),
                ],
            )
        }
        "convdiff1d" => tridiagonal(k, -1.5, 2.0, -0.5),
        "convdiff2d" => grid(
            k,
            [
                &constant(4.0),
                &constant(-0.5),
                &constant(-1.5),
                &constant(-0.75),
                &constant(-1.25),
            ],
        ),
        "one" | "cone" => SparseMatrix::from_coo(1, 1, &[(0, 0, T::real(3.0))]),
        "magnetic2d" | "magshift2d" => {
            let d = if family == "magnetic2d" { 3.0 } else { 1.0 };
            let phase = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
            let east = move |r: usize, _: usize| (-phase[r % 4].0, -phase[r % 4].1);
            let west = move |r: usize, _: usize| (-phase[r % 4].0, phase[r % 4].1);
            grid(
                k,
                [&constant(d), &east, &west, &constant(-1.0), &constant(-1.0)],
            )
        }
        "helmholtz2d" => {
            let diagonal = |_: usize, _: usize| (2.5, -0.25);
            grid(
                k,
                [
                    &diagonal,
                    &constant(-1.0),
                    &constant(-1.0),
                    &constant(-1.0),
                    &constant(-1.0),
                ],
            )
        }
        "cconvdiff2d" => {
            let diagonal = |_: usize, _: usize| (4.0, 1.0);
            let east = |_: usize, _: usize| (-0.5, 0.25);
            let west = |_: usize, _: usize| (-1.5, -0.25);
            let north = |_: usize, _: usize| (-0.75, 0.5);
            grid(k, [&diagonal, &east, &west, &north, &constant(-1.25)])
        }
        "graded" | "cgraded" => graded(k, family == "cgraded"),
        _ => panic!("no problem {name}"),
    };
    let n = a.rows();
    let b = (0..n)
        .map(|i| {
            if family == "graded" || family == "cgraded" {
                c::<T>(
                    ((7 * i + 3) % 11) as f64 - 5.0,
                    ((5 * i + 1) % 7) as f64 - 3.0,
                )
            } else if family == "neumann2d" {
                // In the range of the matrix, whose null space is the constants: both parts
                // sum to 0.
                let (u, v) = (
                    |i: usize| ((7 * i + 3) % 11) as f64,
                    |i: usize| ((5 * i + 1) % 7) as f64,
                );
                c::<T>(u(i) - u(n - 1 - i), v(i) - v(n - 1 - i))
            } else {
                c::<T>(
                    ((13 * i + 5) % 17) as f64 - 8.0,
                    ((7 * i + 3) % 11) as f64 - 5.0,
                )
            }
        })
        .collect();
    let x0 = (0..n)
        .map(|i| {
            c::<T>(
                (((5 * i + 1) % 7) as f64 - 3.0) / 4.0,
                (((3 * i + 2) % 5) as f64 - 2.0) / 4.0,
            )
        })
        .collect();
    (a, b, x0)
}

/// `−(c·u')' + 2⁻¹⁰·u` on `k` points with `c_{i+1/2} = 2^e(i)`, `e(i) = 8 − ⌊16·|(i mod 24) − 12| / 24⌋`,
/// as `gen_iter.py` builds it: symmetric positive definite, a condition number of about 10⁶, a
/// diagonal far from constant. With `phases` the element `(i, i + 1)` is multiplied by `iⁱ` and
/// `(i + 1, i)` by its conjugate, a Hermitian matrix unitarily similar to the real one.
fn graded<T: Num>(k: usize, phases: bool) -> SparseMatrix<T> {
    let coefficient = |i: usize| 2f64.powi(8 - ((i % 24).abs_diff(12) * 16 / 24) as i32);
    let phase = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    let mut e = Vec::new();
    for i in 0..k {
        let diagonal = coefficient(i) + coefficient(i + 1) + 2f64.powi(-10);
        e.push((i, i, c::<T>(diagonal, 0.0)));
        if i + 1 < k {
            let (p, q) = if phases { phase[i % 4] } else { (1.0, 0.0) };
            let v = -coefficient(i + 1);
            e.push((i, i + 1, c::<T>(v * p, v * q)));
            e.push((i + 1, i, c::<T>(v * p, -v * q)));
        }
    }
    SparseMatrix::from_coo(k, k, &e)
}

/// The weights of `gen_iter.py`, `wᵢ = ((3i + 1) mod 5) − 2`.
fn weight(i: usize) -> f64 {
    ((3 * i + 1) % 5) as f64 - 2.0
}

// ---------------------------------------------------------------------------------------------
// Checks made apart from the library
// ---------------------------------------------------------------------------------------------

/// A sum carried as a float and the rounding errors it has dropped (Knuth's two-sum, and the
/// exact error of a product from a fused multiply-add).
struct Compensated {
    sum: f64,
    error: f64,
}

impl Compensated {
    fn new(x: f64) -> Self {
        Self { sum: x, error: 0.0 }
    }

    /// Subtracts `p·q`.
    fn subtract(&mut self, p: f64, q: f64) {
        let product = p * q;
        let dropped = p.mul_add(q, -product);
        let sum = self.sum - product;
        let part = sum - self.sum;
        self.error += (self.sum - (sum - part)) + (-product - part) - dropped;
        self.sum = sum;
    }

    fn value(&self) -> f64 {
        self.sum + self.error
    }
}

/// `‖b − A·x‖₂` formed here rather than by the library, each element with compensated sums in
/// `f64`: the exact residual of `f32` data, and within a few units of `ε²` of it for `f64`.
pub(super) fn true_residual<T: Num>(a: &SparseMatrix<T>, x: &[T], b: &[T]) -> f64 {
    let mut total = 0.0f64;
    for (i, &bi) in b.iter().enumerate() {
        let mut re = Compensated::new(bi.re().to_f64());
        let mut im = Compensated::new(bi.im().to_f64());
        for (j, aij) in a.row(i) {
            let (p, q) = (aij.re().to_f64(), aij.im().to_f64());
            let (u, v) = (x[j].re().to_f64(), x[j].im().to_f64());
            re.subtract(p, u);
            re.subtract(-q, v);
            im.subtract(p, v);
            im.subtract(q, u);
        }
        let (re, im) = (re.value(), im.value());
        total += re * re + im * im;
    }
    total.sqrt()
}

/// `‖v‖₂` in `f64`.
fn norm<T: Num>(v: &[T]) -> f64 {
    v.iter().map(|&x| magnitude(x).powi(2)).sum::<f64>().sqrt()
}

/// What every result must satisfy: `converged` exactly when its reason says so, the residual it
/// reports that of the `x` it returns, and, when it says converged, that residual within
/// `max(rtol·‖b‖₂, atol)` as computed here.
fn assert_honest<T: Num>(r: &SolveResult<T>, a: &SparseMatrix<T>, b: &[T], rtol: f64, atol: f64) {
    assert_eq!(
        r.converged,
        r.reason == StopReason::Converged,
        "{:?}",
        r.reason
    );
    assert_eq!(r.x.len(), b.len());
    let truth = true_residual(a, &r.x, b);
    let reported = r.residual.to_f64();
    if truth.is_finite() {
        assert!(
            (reported - truth).abs() <= 1e-5 * truth + 1e-300,
            "reported residual {reported:e}, formed afresh {truth:e}"
        );
    }
    if r.converged {
        let threshold = (rtol * norm(b)).max(atol);
        assert!(
            truth <= threshold * (1.0 + 8.0 * T::EPS) + f64::MIN_POSITIVE,
            "converged with a residual of {truth:e} against a tolerance of {threshold:e}"
        );
    }
}

/// Runs `f` on a thread of its own and fails if it has not returned within `seconds`: the
/// regressions of a method that looped for ever must fail rather than hang.
pub(super) fn within<R: Send + 'static>(seconds: u64, f: impl FnOnce() -> R + Send + 'static) -> R {
    if cfg!(miri) {
        return f();
    }
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = send.send(f());
    });
    match receive.recv_timeout(Duration::from_secs(seconds)) {
        Ok(r) => r,
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("no return within {seconds} s"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the method panicked"),
    }
}

/// The method `name` with the options and the preconditioner.
fn solve<T: Num, A: LinearOperator<T> + ?Sized>(
    name: &str,
    a: &A,
    b: &[T],
    opts: &SolveOptions<T>,
    pre: Option<&dyn Preconditioner<T>>,
) -> Option<SolveResult<T>> {
    match name {
        "cg" => cg_with(a, b, opts, pre),
        "minres" => minres_with(a, b, opts, pre),
        "gmres" => gmres_with(a, b, opts, pre),
        "bicgstab" => bicgstab_with(a, b, opts, pre),
        _ => panic!("no method {name}"),
    }
}

const METHODS: [&str; 4] = ["cg", "minres", "gmres", "bicgstab"];

/// The element type's tolerance of the reference runs, 1e-10 or 1e-4.
fn reference_rtol<T: Num>() -> f64 {
    if T::EPS < 1e-10 { 1e-10 } else { 1e-4 }
}

// ---------------------------------------------------------------------------------------------
// Against scipy
// ---------------------------------------------------------------------------------------------

/// The runs of scipy, from `data/gen_iter.py`.
const SCIPY: &str = include_str!("data/iter_scipy.txt");

/// The name of the element type in `iter_scipy.txt`.
fn type_name<T: Num>() -> &'static str {
    match (T::IS_COMPLEX, T::EPS < 1e-10) {
        (false, true) => "f64",
        (false, false) => "f32",
        (true, true) => "c64",
        (true, false) => "c32",
    }
}

/// How far beyond the range scipy's own counts span under rounding-level changes of `b`
/// (`low` to `high`, see `gen_iter.py`) a count may lie: seven samples of that spread
/// understate it, BiCGSTAB more than the methods that minimise something (a fused
/// multiply-add alone moves its count on the 20 x 20 Laplacian from 51 to 55), and `f32` more
/// than `f64`.
fn margin(method: &str, single: bool, high: usize) -> usize {
    let (fixed, part) = match (method, single) {
        ("bicgstab", false) => (3, 0.10),
        ("bicgstab", true) => (4, 0.15),
        (_, false) => (2, 0.05),
        (_, true) => (3, 0.10),
    };
    fixed.max((part * high as f64).ceil() as usize)
}

/// Every run of `iter_scipy.txt` for the type: the iteration count in scipy's class, the same
/// verdict, an honest result, and a solution within what the two residuals allow of scipy's.
fn agrees_with_scipy<T: Num>() {
    let name = type_name::<T>();
    let single = T::EPS > 1e-10;
    let mut failures = Vec::new();
    // The runs that converged where scipy, in single precision, did not.
    let mut beyond = Vec::new();
    let mut runs = 0;
    let mut worst = std::collections::BTreeMap::<&str, (usize, usize, f64)>::new();
    for line in SCIPY.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f[1] != name {
            continue;
        }
        runs += 1;
        let (problem_name, method, pre, restart, start) = (f[0], f[2], f[3], f[4], f[5]);
        let rtol: f64 = f[6].parse().unwrap();
        let (iterations, converged): (usize, bool) = (f[7].parse().unwrap(), f[8] == "1");
        let (low, high): (usize, usize) = (f[9].parse().unwrap(), f[10].parse().unwrap());
        let wx = Complex::new(f[11].parse::<f64>().unwrap(), f[12].parse::<f64>().unwrap());
        let (residual, inverse): (f64, f64) = (f[13].parse().unwrap(), f[14].parse().unwrap());
        let (a, b, x0) = problem::<T>(problem_name);
        let n = b.len();
        if cfg!(miri) && (n > 64 || high > 64) {
            continue;
        }
        let m: Option<Box<dyn Preconditioner<T>>> = match pre {
            "none" => None,
            "jacobi" => Some(Box::new(Jacobi::new(&a).unwrap())),
            "ic0" => Some(Box::new(Ic0::new(&a).unwrap())),
            "ilu0" => Some(Box::new(Ilu0::new(&a).unwrap())),
            _ => panic!("{pre}"),
        };
        let budget = 20 * n;
        let mut opts = SolveOptions::new()
            .with_rtol(T::Real::from_f64(rtol))
            .with_max_iter(budget);
        let restart: usize = restart.parse().unwrap();
        if restart > 0 {
            opts = opts.with_restart(restart);
        }
        if start == "guess" {
            opts = opts.with_initial_guess(&x0);
        }
        let r = solve(method, &a, &b, &opts, m.as_deref()).unwrap();
        assert_honest(&r, &a, &b, rtol, 0.0);
        let label = format!("{problem_name} {name} {method} {pre} {restart} {start}");
        let allowed = margin(method, single, high);
        let (from, to) = (low.saturating_sub(allowed), high + allowed);
        if converged && !r.converged {
            failures.push(format!(
                "{label}: scipy converged in {iterations}, this did not ({:?} after {})",
                r.reason, r.iterations
            ));
        } else if !converged && r.converged && r.iterations + allowed < budget {
            // In single precision scipy sums its inner products in the element type, and these
            // methods in f64, so they may get further: the residual was checked above, by a
            // computation of its own. In double precision the two should agree.
            if single {
                beyond.push(format!("{label}: {} iterations", r.iterations));
            } else {
                failures.push(format!(
                    "{label}: scipy did not converge in {budget}, this did in {}",
                    r.iterations
                ));
            }
        } else if converged && !(from..=to).contains(&r.iterations) {
            failures.push(format!(
                "{label}: {} iterations, scipy {iterations} ({low} to {high})",
                r.iterations
            ));
        }
        if converged {
            // How far outside scipy's own range, and how far from scipy's count on the data.
            let outside = low
                .saturating_sub(r.iterations)
                .max(r.iterations.saturating_sub(high));
            let entry = worst.entry(method).or_insert((0, 0, 0.0));
            let d = r.iterations.abs_diff(iterations);
            entry.0 = entry.0.max(outside);
            entry.1 = entry.1.max(d);
            entry.2 = f64::max(entry.2, d as f64 / iterations as f64);
        }
        // |w·(x − x_scipy)| ≤ ‖w‖·‖A⁻¹‖·(‖r‖ + ‖r_scipy‖), the error of a weighted sum.
        let mut wx_here = Complex::new(0.0, 0.0);
        let mut size = 0.0;
        for (i, &x) in r.x.iter().enumerate() {
            wx_here += Complex::new(x.re().to_f64(), x.im().to_f64()) * weight(i);
            size += weight(i).abs() * magnitude(x);
        }
        let w_norm = (0..n).map(|i| weight(i).powi(2)).sum::<f64>().sqrt();
        let bound = w_norm * inverse * (true_residual(&a, &r.x, &b) + residual) * 1.001
            + 64.0 * f64::EPSILON * size;
        let gap = (wx_here - wx).abs();
        if gap > bound {
            failures.push(format!(
                "{label}: w·x differs from scipy's by {gap:e}, more than {bound:e}"
            ));
        }
    }
    eprintln!(
        "{name}: {runs} runs against scipy; worst (beyond scipy's range, |Δ| from its count, relative) by method: {worst:?}"
    );
    eprintln!("{name}: converged where scipy did not: {beyond:?}");
    assert!(cfg!(miri) || runs >= 25, "{runs} runs");
    assert!(
        failures.is_empty(),
        "{} of {runs}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_solvers_agree_with_scipy_in_f64() {
    agrees_with_scipy::<f64>();
}

#[test]
fn the_solvers_agree_with_scipy_in_f32() {
    agrees_with_scipy::<f32>();
}

#[test]
fn the_solvers_agree_with_scipy_in_complex_f64() {
    agrees_with_scipy::<Complex<f64>>();
}

#[test]
fn the_solvers_agree_with_scipy_in_complex_f32() {
    agrees_with_scipy::<Complex<f32>>();
}

// ---------------------------------------------------------------------------------------------
// The contract at its edges
// ---------------------------------------------------------------------------------------------

/// Every method on `A·x = b` with the options and no preconditioner.
fn all<T: Num>(
    a: &SparseMatrix<T>,
    b: &[T],
    opts: &SolveOptions<T>,
) -> Vec<(&'static str, SolveResult<T>)> {
    METHODS
        .iter()
        .map(|&m| (m, solve(m, a, b, opts, None).unwrap()))
        .collect()
}

#[test]
fn a_zero_right_hand_side_gives_zero_at_once() {
    fn check<T: Num>() {
        let (a, _, x0) = problem::<T>("lap2d6");
        let b = vec![T::zero(); 36];
        for opts in [
            SolveOptions::new(),
            SolveOptions::new().with_initial_guess(&x0),
        ] {
            for (m, r) in all(&a, &b, &opts) {
                assert!(r.converged && r.reason == StopReason::Converged, "{m}");
                assert_eq!((r.iterations, r.matvecs), (0, 0), "{m}");
                assert!(r.x.iter().all(|x| x.is_zero()), "{m}");
                assert_eq!(r.residual, T::Real::ZERO);
            }
        }
    }
    each_type!(check);
}

#[test]
fn a_system_of_size_zero_is_solved_at_once() {
    fn check<T: Num>() {
        let a = SparseMatrix::<T>::zeros(0, 0);
        for (m, r) in all(&a, &[], &SolveOptions::new()) {
            assert!(r.converged && r.x.is_empty() && r.iterations == 0, "{m}");
        }
        let f = FnOperator::new(0, 0, |_: &[T], _: &mut [T]| {});
        assert!(cg(&f, &[]).unwrap().converged);
        assert!(gmres(&f, &[]).unwrap().converged);
    }
    each_type!(check);
}

#[test]
fn a_one_by_one_system_takes_one_iteration() {
    fn check<T: Num>() {
        let a = SparseMatrix::from_coo(1, 1, &[(0, 0, T::real(3.0))]);
        let b = [c::<T>(-3.0, 2.0)];
        for (m, r) in all(&a, &b, &SolveOptions::new()) {
            assert!(r.converged && r.iterations == 1, "{m}: {r:?}");
            assert!(
                magnitude(r.x[0] - c::<T>(-1.0, 2.0 / 3.0)) <= 4.0 * T::EPS,
                "{m}"
            );
        }
        // A complex one for the methods that need no symmetry.
        let a = SparseMatrix::from_coo(1, 1, &[(0, 0, c::<T>(2.0, -1.0))]);
        for m in ["gmres", "bicgstab"] {
            let r = solve(m, &a, &b, &SolveOptions::new(), None).unwrap();
            assert!(r.converged && r.iterations == 1, "{m}");
            assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
        }
    }
    each_type!(check);
}

#[test]
fn a_starting_guess_that_passes_the_test_is_returned_at_once() {
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>("lap2d8");
        let rtol = T::Real::from_f64(64.0 * T::EPS);
        let tight = gmres_with(
            &a,
            &b,
            &SolveOptions::new().with_rtol(rtol).with_restart(64),
            None,
        )
        .unwrap();
        assert!(tight.converged);
        let opts = SolveOptions::new().with_initial_guess(&tight.x);
        for (m, r) in all(&a, &b, &opts) {
            assert!(
                r.converged && r.iterations == 0 && r.matvecs == 1,
                "{m}: {r:?}"
            );
            assert_eq!(r.x, tight.x, "{m}");
        }
    }
    each_type!(check);
}

#[test]
fn the_iteration_limit_ends_a_run_at_the_point_it_reached() {
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>("lap2d12");
        let opts = SolveOptions::new()
            .with_rtol(T::Real::from_f64(1e-12))
            .with_max_iter(7);
        for (m, r) in all(&a, &b, &opts) {
            assert!(
                !r.converged && r.reason == StopReason::IterationLimit,
                "{m}"
            );
            assert_eq!(r.iterations, 7, "{m}");
            assert_honest(&r, &a, &b, 1e-12, 0.0);
            assert!(true_residual(&a, &r.x, &b) < norm(&b), "{m} got nowhere");
        }
    }
    each_type!(check);
}

#[test]
fn every_method_counts_its_iterations_against_max_iter() {
    // The audit's p_iter-10: gmres ran a whole cycle of 50 or 200 before it looked at max_iter.
    let n = 200;
    let a = tridiagonal::<f64>(n, -1.0, 2.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
    for (max_iter, restart) in [(0, 50), (5, 50), (5, 200), (37, 10)] {
        let opts = SolveOptions::new()
            .with_max_iter(max_iter)
            .with_restart(restart);
        for (m, r) in all(&a, &b, &opts) {
            assert_eq!(
                r.iterations, max_iter,
                "{m} with max_iter {max_iter}, restart {restart}"
            );
            assert_eq!(r.reason, StopReason::IterationLimit, "{m}");
            assert_honest(&r, &a, &b, 1.5e-8, 0.0);
        }
    }
    let r = gmres_with(&a, &b, &SolveOptions::new().with_max_iter(0), None).unwrap();
    assert!(r.x.iter().all(|&x| x == 0.0) && r.matvecs == 0);
}

#[test]
fn data_that_are_not_finite_end_in_a_report() {
    // The audit's p_iter-9: an infinite element of b was a converged solve with x = 0.
    fn check<T: Num>() {
        let (a, b, x0) = problem::<T>("lap2d4");
        for v in [T::Real::INFINITY, -T::Real::INFINITY, T::Real::NAN] {
            let bad = T::from_parts(v, T::Real::ZERO);
            let mut spoilt_b = b.clone();
            spoilt_b[3] = bad;
            for (m, r) in all(&a, &spoilt_b, &SolveOptions::new()) {
                assert!(
                    !r.converged && r.reason == StopReason::NonFinite,
                    "{m}, b with {v:?}"
                );
                assert_eq!(r.iterations, 0, "{m}");
            }
            let mut guess = x0.clone();
            guess[5] = bad;
            for (m, r) in all(&a, &b, &SolveOptions::new().with_initial_guess(&guess)) {
                assert!(
                    !r.converged && r.reason == StopReason::NonFinite,
                    "{m}, x0 with {v:?}"
                );
            }
            let spoilt_a = a.map(|i, j, x| if (i, j) == (6, 7) { bad } else { x });
            for (m, r) in all(&spoilt_a, &b, &SolveOptions::new()) {
                assert!(
                    !r.converged && r.reason == StopReason::NonFinite,
                    "{m}, A with {v:?}"
                );
            }
        }
        // A preconditioner that answers NaN.
        struct Broken;
        impl<T: Num> Preconditioner<T> for Broken {
            fn apply(&self, _: &[T], z: &mut [T]) {
                z.fill(T::from_parts(T::Real::NAN, T::Real::ZERO));
            }
        }
        for m in METHODS {
            let r = solve(m, &a, &b, &SolveOptions::new(), Some(&Broken)).unwrap();
            assert!(!r.converged && r.reason == StopReason::NonFinite, "{m}");
            assert!(r.x.iter().all(|x| x.is_finite()), "{m} returned NaN");
        }
    }
    each_type!(check);
}

#[test]
fn options_no_method_can_serve_give_none() {
    let (a, b, _) = problem::<f64>("lap2d4");
    for opts in [
        SolveOptions::new().with_rtol(-1.0),
        SolveOptions::new().with_rtol(f64::NAN),
        SolveOptions::new().with_rtol(f64::INFINITY),
        SolveOptions::new().with_atol(-1e-3),
        SolveOptions::new().with_atol(f64::INFINITY),
    ] {
        for m in METHODS {
            assert!(solve(m, &a, &b, &opts, None).is_none(), "{m}");
        }
    }
    let opts = SolveOptions::new().with_restart(0);
    assert!(gmres_with(&a, &b, &opts, None).is_none());
    // The other methods take no notice of the restart length.
    assert!(cg_with(&a, &b, &opts, None).unwrap().converged);
    assert!(bicgstab_with(&a, &b, &opts, None).unwrap().converged);
}

#[test]
#[should_panic(expected = "square")]
fn a_rectangular_operator_panics() {
    let f = FnOperator::new(2, 3, |_: &[f64], _: &mut [f64]| {});
    let _ = cg(&f, &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "right-hand side has the wrong length")]
fn a_right_hand_side_of_the_wrong_length_panics() {
    let _ = gmres(&tridiagonal::<f64>(4, -1.0, 2.0, -1.0), &[1.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "starting guess has the wrong length")]
fn a_starting_guess_of_the_wrong_length_panics() {
    let a = tridiagonal::<f64>(4, -1.0, 2.0, -1.0);
    let _ = minres_with(
        &a,
        &[1.0; 4],
        &SolveOptions::new().with_initial_guess(&[0.0; 3]),
        None,
    );
}

#[test]
#[should_panic(expected = "wrong length for the preconditioner")]
fn a_preconditioner_of_another_size_panics() {
    // The audit's p_iter-14: cg returned None with no reason.
    let a = tridiagonal::<f64>(4, -1.0, 2.0, -1.0);
    let m = Jacobi::new(&tridiagonal::<f64>(3, -1.0, 2.0, -1.0)).unwrap();
    let _ = cg_with(&a, &[1.0; 4], &SolveOptions::new(), Some(&m));
}

#[test]
fn the_methods_run_on_any_linear_operator() {
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>(if T::IS_COMPLEX {
            "magnetic2d6"
        } else {
            "lap2d6"
        });
        let dense = a.to_matrix();
        let sparse = a.clone();
        let f = FnOperator::new(36, 36, move |x: &[T], y: &mut [T]| {
            sparse.mul_vec_into(x, y)
        });
        let rtol = reference_rtol::<T>();
        let opts = SolveOptions::new().with_rtol(T::Real::from_f64(rtol));
        let reference: Vec<usize> = METHODS
            .iter()
            .map(|&m| solve(m, &a, &b, &opts, None).unwrap().iterations)
            .collect();
        let operators: [&dyn LinearOperator<T>; 3] = [&a, &dense, &f];
        for (k, op) in operators.into_iter().enumerate() {
            for (m, &count) in METHODS.iter().zip(&reference) {
                let r = solve(m, op, &b, &opts, None).unwrap();
                assert!(r.converged, "{m} on operator {k}");
                assert!(r.iterations.abs_diff(count) <= 2, "{m} on operator {k}");
                if k < 2 {
                    assert_honest(&r, &a, &b, rtol, 0.0);
                } else {
                    // A function forms its residual in the working precision, with the rounding
                    // error of its product, about ε·‖|A|·|x| + |b|‖₂ (see `LinearOperator`).
                    let mut size = 0.0;
                    for (i, &bi) in b.iter().enumerate() {
                        let row: f64 = a
                            .row(i)
                            .map(|(j, x)| magnitude(x) * magnitude(r.x[j]))
                            .sum();
                        size += (row + magnitude(bi)).powi(2);
                    }
                    let slack = 8.0 * T::EPS * size.sqrt();
                    let truth = true_residual(&a, &r.x, &b);
                    assert!((r.residual.to_f64() - truth).abs() <= slack, "{m}");
                    assert!(truth <= rtol * norm(&b) + slack, "{m}");
                }
            }
        }
    }
    each_type!(check);
}

#[test]
fn the_residual_of_a_matrix_is_formed_to_its_last_rounding() {
    // 1 − 3·fl(1/3) is 2⁻⁵⁴ in f64 and −2⁻²⁵ in f32; formed in the working precision it is 0,
    // which is what an operator known only by its products gives.
    let mut r = [0.0f64];
    let a = SparseMatrix::from_diagonal(&[3.0f64]);
    a.residual(&[1.0 / 3.0], &[1.0], &mut r);
    assert_eq!(r, [2f64.powi(-54)]);
    a.to_matrix().residual(&[1.0 / 3.0], &[1.0], &mut r);
    assert_eq!(r, [2f64.powi(-54)]);
    FnOperator::new(1, 1, |x: &[f64], y: &mut [f64]| y[0] = 3.0 * x[0]).residual(
        &[1.0 / 3.0],
        &[1.0],
        &mut r,
    );
    assert_eq!(r, [0.0]);
    let mut r = [0.0f32];
    SparseMatrix::from_diagonal(&[3.0f32]).residual(&[1.0 / 3.0], &[1.0], &mut r);
    assert_eq!(r, [-(2f32.powi(-25))]);
    // Both parts of a complex one: (3 + 3i)·fl(1/3) = (1 + i)(1 − 2⁻⁵⁴).
    let mut r = [Complex::new(0.0f64, 0.0)];
    let a = SparseMatrix::from_diagonal(&[Complex::new(3.0f64, 3.0)]);
    a.residual(
        &[Complex::new(1.0 / 3.0, 0.0)],
        &[Complex::new(1.0, 1.0)],
        &mut r,
    );
    assert_eq!(r, [Complex::new(2f64.powi(-54), 2f64.powi(-54))]);
}

// ---------------------------------------------------------------------------------------------
// The defects the audit found, one by one
// ---------------------------------------------------------------------------------------------

/// The cyclic shift `A·eⱼ = eⱼ₊₁` of `n` elements.
fn shift<T: Num>(n: usize) -> SparseMatrix<T> {
    let e: Vec<(usize, usize, T)> = (0..n).map(|j| ((j + 1) % n, j, T::one())).collect();
    SparseMatrix::from_coo(n, n, &e)
}

#[test]
fn gmres_returns_on_every_exact_breakdown() {
    // The audit's p_iter-1: when the first Arnoldi vector is an eigenvector the draft repeated
    // the same empty cycle for ever. Every case runs under a watchdog.
    fn check<T: Num>() {
        within(120, || {
            let one = |a: SparseMatrix<T>, b: Vec<T>, x: Vec<T>, opts: SolveOptions<T>| {
                let r = gmres_with(&a, &b, &opts, None).unwrap();
                assert!(r.converged && r.iterations == 1, "{a:?}: {r:?}");
                for (x, y) in r.x.iter().zip(&x) {
                    assert!(magnitude(*x - *y) <= 4.0 * T::EPS * magnitude(*y).max(1.0));
                }
            };
            let real = |v: &[f64]| v.iter().map(|&x| T::real(x)).collect::<Vec<T>>();
            // 2·x = 1, and a complex 1 x 1.
            one(
                SparseMatrix::from_diagonal(&real(&[2.0])),
                real(&[1.0]),
                real(&[0.5]),
                SolveOptions::new(),
            );
            let z = c::<T>(2.0, -1.0);
            let w = c::<T>(1.0, 3.0);
            one(
                SparseMatrix::from_diagonal(&[z]),
                vec![w],
                vec![w / z],
                SolveOptions::new(),
            );
            // The identity with a unit vector, a diagonal matrix with one nonzero on the right
            // and a budget of 5 iterations, a block whose right-hand side lives in a 1 x 1 block,
            // and the 2 x 2 Laplacian with its eigenvector.
            one(
                SparseMatrix::identity(3),
                real(&[1.0, 0.0, 0.0]),
                real(&[1.0, 0.0, 0.0]),
                SolveOptions::new(),
            );
            let restarted = SolveOptions::new().with_max_iter(5).with_restart(2);
            one(
                SparseMatrix::from_diagonal(&real(&[2.0, 3.0, 4.0])),
                real(&[0.0, 6.0, 0.0]),
                real(&[0.0, 2.0, 0.0]),
                restarted,
            );
            let block = SparseMatrix::from_coo(
                4,
                4,
                &[
                    (0, 0, T::real(2.0)),
                    (0, 1, T::real(-1.0)),
                    (1, 0, T::real(-1.0)),
                    (1, 1, T::real(2.0)),
                    (1, 2, T::real(-1.0)),
                    (2, 1, T::real(-1.0)),
                    (2, 2, T::real(2.0)),
                    (3, 3, T::real(5.0)),
                ],
            );
            one(
                block,
                real(&[0.0, 0.0, 0.0, 1.0]),
                real(&[0.0, 0.0, 0.0, 0.2]),
                SolveOptions::new(),
            );
            one(
                tridiagonal(2, -1.0, 2.0, -1.0),
                real(&[1.0, 1.0]),
                real(&[1.0, 1.0]),
                SolveOptions::new(),
            );
            // An exact preconditioner, with any right-hand side: A·M⁻¹ = I.
            let (a, b, _) = problem::<T>("convdiff1d30");
            let exact = Ilu0::new(&a).unwrap();
            let r = gmres_with(&a, &b, &SolveOptions::new(), Some(&exact)).unwrap();
            assert!(r.converged && r.iterations == 1);
            let d = SparseMatrix::from_diagonal(&real(&[1.0, 3.0, 9.0]));
            let r = gmres_with(
                &d,
                &real(&[1.0, 1.0, 1.0]),
                &SolveOptions::new(),
                Some(&Jacobi::new(&d).unwrap()),
            )
            .unwrap();
            assert!(r.converged && r.iterations == 1);
            // The zero matrix, and a singular one with b in its null space or with b not in its
            // range: no solution, an honest report, and no hang.
            let zero = SparseMatrix::<T>::zeros(3, 3);
            let r = gmres(&zero, &real(&[1.0, 2.0, 3.0])).unwrap();
            assert!(!r.converged && r.reason == StopReason::Breakdown);
            let s = SparseMatrix::from_coo(
                2,
                2,
                &[
                    (0, 0, T::real(1.0)),
                    (0, 1, T::real(-1.0)),
                    (1, 0, T::real(-1.0)),
                    (1, 1, T::real(1.0)),
                ],
            );
            let r = gmres(&s, &real(&[1.0, 1.0])).unwrap();
            assert!(
                !r.converged
                    && r.reason == StopReason::Breakdown
                    && r.x.iter().all(|x| x.is_zero())
            );
            let r = gmres(&s, &real(&[1.0, 0.0])).unwrap();
            assert!(!r.converged && r.reason == StopReason::Breakdown);
            // The least squares solution of that system: its residual is the part of b in the
            // null space, (1/2, 1/2).
            assert!((r.residual.to_f64() - 0.5f64.sqrt()).abs() <= 8.0 * T::EPS);
            assert_honest(&r, &s, &real(&[1.0, 0.0]), 1.5e-8, 0.0);
        });
    }
    each_type!(check);
}

#[test]
fn gmres_keeps_the_last_column_when_the_space_is_exhausted() {
    // The audit's p_iter-2: the draft dropped the column of the step at which the Krylov space
    // became invariant, and needed a second cycle or never converged.
    fn check<T: Num>() {
        within(120, || {
            for n in [3, 8] {
                // A·x = e₁ for the cyclic shift has the solution eₙ, reached at step n exactly.
                let mut b = vec![T::zero(); n];
                b[0] = T::one();
                let r = gmres_with(
                    &shift::<T>(n),
                    &b,
                    &SolveOptions::new().with_restart(n),
                    None,
                )
                .unwrap();
                assert!(r.converged && r.iterations == n, "shift {n}: {r:?}");
                for (i, &x) in r.x.iter().enumerate() {
                    let want = if i + 1 == n { 1.0 } else { 0.0 };
                    assert!(magnitude(x - T::real(want)) <= 8.0 * T::EPS);
                }
            }
            // The Arnoldi vectors of a tridiagonal matrix started at e₁ are ±eₖ exactly, so the
            // space is exhausted at step n: scipy needs 3, 10 and 100 iterations.
            for n in [3, 10, size(100, 20)] {
                let mut b = vec![T::zero(); n];
                b[0] = T::one();
                let opts = SolveOptions::new()
                    .with_restart(n)
                    .with_rtol(T::Real::from_f64(reference_rtol::<T>()));
                let r = gmres_with(&tridiagonal::<T>(n, -1.0, 2.0, -1.0), &b, &opts, None).unwrap();
                assert!(
                    r.converged && r.iterations <= n,
                    "lap1d {n}: {} iterations",
                    r.iterations
                );
            }
        });
    }
    each_type!(check);
}

#[test]
fn gmres_stops_a_restart_that_makes_no_progress() {
    // Restarted every two steps, the cyclic shift of three is where it started after each
    // cycle: the method says so instead of cycling to its budget.
    fn check<T: Num>() {
        let mut b = vec![T::zero(); 3];
        b[0] = T::one();
        let r = gmres_with(
            &shift::<T>(3),
            &b,
            &SolveOptions::new().with_restart(2),
            None,
        )
        .unwrap();
        assert!(!r.converged && r.reason == StopReason::Stagnated && r.iterations == 2);
        assert!(r.x.iter().all(|x| x.is_zero()));
    }
    each_type!(check);
}

#[test]
fn gmres_measures_the_residual_of_the_system_whatever_the_preconditioner() {
    // The audit's p_iter-7: the preconditioned residual was compared with the tolerance of the
    // unpreconditioned one, so that Jacobi, which only rescales the 2-D Laplacian, cut every
    // cycle short: 59 iterations without it, 834 with it at a scale of 10⁶. On the right the
    // residual the method sees is the true one at every scale.
    let opts = SolveOptions::new().with_restart(60).with_max_iter(3000);
    let (lap, b, _) = problem::<f64>("lap2d20");
    let base = gmres_with(&lap, &b, &opts, None).unwrap().iterations;
    assert_eq!(base, 59);
    let scales: &[f64] = if cfg!(miri) {
        &[1.0, 1e6]
    } else {
        &[1e-9, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12]
    };
    for &scale in scales {
        let a = lap.map(|_, _, x| x * scale);
        let plain = gmres_with(&a, &b, &opts, None).unwrap();
        let jacobi = gmres_with(&a, &b, &opts, Some(&Jacobi::new(&a).unwrap())).unwrap();
        assert!(plain.converged && jacobi.converged);
        assert!(
            plain.iterations.abs_diff(base) <= 1 && jacobi.iterations.abs_diff(base) <= 1,
            "scale {scale}: {} and {}",
            plain.iterations,
            jacobi.iterations
        );
    }
    // In f32, GMRES(5) on the 1-D Laplacian of 100 from e₁: scipy 1575 iterations, the draft
    // with Jacobi (the division by 2) did not converge in 2000.
    if cfg!(miri) {
        return;
    }
    let n = 100;
    let a = tridiagonal::<f32>(n, -1.0, 2.0, -1.0);
    let mut b = vec![0.0f32; n];
    b[0] = 1.0;
    let opts = SolveOptions::new()
        .with_restart(5)
        .with_rtol(1e-4)
        .with_max_iter(3000);
    let plain = gmres_with(&a, &b, &opts, None).unwrap();
    let jacobi = gmres_with(&a, &b, &opts, Some(&Jacobi::new(&a).unwrap())).unwrap();
    assert!(plain.converged && jacobi.converged);
    assert!(
        plain.iterations.abs_diff(1575) <= 80 && jacobi.iterations.abs_diff(plain.iterations) <= 5,
        "{} and {}",
        plain.iterations,
        jacobi.iterations
    );
}

#[test]
fn gmres_keeps_its_basis_orthogonal_on_an_ill_conditioned_matrix() {
    // With the basis orthogonal to working precision, a cycle over the whole space solves a
    // system as far as its conditioning allows, and a short second one reaches a tolerance
    // near the rounding level. Here κ = 10⁶ and rtol = 10⁻¹²: 43 and 44 iterations of 40-step
    // cycles; with one pass of Gram-Schmidt only (the review of round 1 deleted the second and
    // no test noticed), 69 and 62.
    fn check<T: Num>() {
        let n = 40;
        let d: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(6.0 * i as f64 / (n - 1) as f64))
            .collect();
        let b = vec![T::one(); n];
        let opts = SolveOptions::new()
            .with_restart(n)
            .with_rtol(T::Real::from_f64(1e-12));
        let diagonal =
            SparseMatrix::from_diagonal(&d.iter().map(|&x| T::real(x)).collect::<Vec<T>>());
        let mut e: Vec<(usize, usize, T)> = (0..n).map(|i| (i, i, T::real(d[i]))).collect();
        e.extend((0..n - 1).map(|i| (i, i + 1, T::one())));
        let bidiagonal = SparseMatrix::from_coo(n, n, &e);
        for (name, a) in [("diagonal", diagonal), ("bidiagonal", bidiagonal)] {
            let r = gmres_with(&a, &b, &opts, None).unwrap();
            assert_honest(&r, &a, &b, 1e-12, 0.0);
            assert!(
                r.converged && r.iterations <= n + n / 4,
                "{name}: {}",
                r.iterations
            );
        }
    }
    check::<f64>();
    check::<Complex<f64>>();
}

#[test]
fn gmres_takes_memory_for_the_iterations_it_runs() {
    // The audit's p_iter-11: the draft allocated the (m + 1)² Hessenberg matrix at every cycle,
    // so "no restart" cost O(n²) memory: 1.2 GB at n = 12 000 for a system solved in ten steps.
    // Here, with the restart length the whole space, 10⁵ unknowns.
    let n = size(100_000, 50);
    let a = tridiagonal::<f64>(n, -1.0, 10.0, -2.0);
    let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
    let opts = SolveOptions::new()
        .with_restart(usize::MAX)
        .with_rtol(1e-10);
    let r = gmres_with(&a, &b, &opts, None).unwrap();
    assert!(r.converged && r.iterations < 30, "{r:?}");
}

#[test]
fn bicgstab_is_van_der_vorsts_method() {
    // The audit's p_iter-3: alpha stood where omega belongs in the direction update, and the
    // draft diverged where the method converges in a few steps. The counts of a textbook
    // BiCGSTAB (the audit's) and of scipy, rtol 1e-10.
    let opts = SolveOptions::new().with_rtol(1e-10).with_max_iter(2000);
    let a = tridiagonal::<f64>(3, -1.0, 2.0, -1.0);
    let r = bicgstab_with(&a, &[1.0, 1.0, 1.0], &SolveOptions::new(), None).unwrap();
    assert!(r.converged && r.iterations <= 3);
    assert!(
        r.x.iter()
            .zip([1.5, 2.0, 1.5])
            .all(|(x, y)| (x - y).abs() < 1e-7)
    );
    let a = SparseMatrix::from_coo(
        4,
        4,
        &[
            (0, 0, 4.0),
            (0, 1, 1.0),
            (0, 3, -1.0),
            (1, 0, -1.0),
            (1, 1, 5.0),
            (1, 2, 2.0),
            (2, 1, 1.0),
            (2, 2, 6.0),
            (2, 3, 1.5),
            (3, 0, 2.0),
            (3, 2, -1.0),
            (3, 3, 7.0),
        ],
    );
    let r = bicgstab_with(&a, &[1.0, 2.0, 3.0, 4.0], &opts, None).unwrap();
    assert!(r.converged && r.iterations <= 5, "4 x 4: {}", r.iterations);
    let n = 50;
    let a = tridiagonal::<f64>(n, -1.5, 2.0, -0.5);
    let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
    let r = bicgstab_with(&a, &b, &opts, None).unwrap();
    assert!(
        r.converged && r.iterations.abs_diff(50) <= 8,
        "convection-diffusion: {}",
        r.iterations
    );
    let (a, b, _) = problem::<f64>("lap2d20");
    let r = bicgstab_with(&a, &b, &opts, None).unwrap();
    // 51 in the textbook form, 52 in scipy's, 55 with the fused multiply-adds of this one (a
    // Python model of its arithmetic gives 55 too).
    assert!(
        r.converged && (48..=60).contains(&r.iterations),
        "2-D Laplacian: {}",
        r.iterations
    );
    assert_honest(&r, &a, &b, 1e-10, 0.0);
}

#[test]
fn bicgstab_never_passes_a_wrong_answer_off_as_converged() {
    // The audit's p_iter-3, its consequence: converged = true with a 47 % wrong x in f32 (the
    // 2-D Laplacian of 6 x 6, Jacobi, default options), and x[0] = −2.2e8 for 17 x 17.
    let cases: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(6, 13, 7)]
    } else {
        &[(6, 13, 7), (17, 3, 17), (20, 13, 17)]
    };
    for &(k, multiplier, modulus) in cases {
        let n = k * k;
        let (a64, _, _) = problem::<f64>(&format!("lap2d{k}"));
        let a = a64.map(|_, _, x| x);
        let a32: SparseMatrix<f32> = SparseMatrix::from_coo(
            n,
            n,
            &a.row_pointers()
                .windows(2)
                .enumerate()
                .flat_map(|(i, w)| (w[0]..w[1]).map(move |p| (i, p)))
                .map(|(i, p)| (i, a.col_indices()[p], a.values()[p] as f32))
                .collect::<Vec<_>>(),
        );
        let b: Vec<f32> = (0..n)
            .map(|i| ((i * multiplier + 1) % modulus) as f32 - (modulus / 2) as f32)
            .collect();
        let r = bicgstab_with(
            &a32,
            &b,
            &SolveOptions::new(),
            Some(&Jacobi::new(&a32).unwrap()),
        )
        .unwrap();
        assert_honest(&r, &a32, &b, f32::EPSILON.sqrt() as f64, 0.0);
        assert!(r.converged, "{k} x {k}: {:?}", r.reason);
        // The solution in f64, to compare the f32 one with: well within the tolerance times
        // the condition number of the Laplacian.
        let b64: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let exact = cg_with(&a64, &b64, &SolveOptions::new().with_rtol(1e-13), None).unwrap();
        let error =
            r.x.iter()
                .zip(&exact.x)
                .map(|(&x, &y)| (f64::from(x) - y).powi(2))
                .sum::<f64>()
                .sqrt();
        assert!(
            error <= 0.05 * norm(&exact.x),
            "{k} x {k}: relative error {}",
            error / norm(&exact.x)
        );
    }
}

#[test]
fn bicgstab_stops_at_the_half_step_that_solves_the_system() {
    // The audit's p_iter-4: when the first half of a step landed on the solution, s = 0, the
    // draft returned None.
    fn check<T: Num>() {
        let real = |v: &[f64]| v.iter().map(|&x| T::real(x)).collect::<Vec<T>>();
        let cases: Vec<(SparseMatrix<T>, Vec<T>, Vec<T>)> = vec![
            (
                SparseMatrix::identity(3),
                real(&[1.0, 2.0, 3.0]),
                real(&[1.0, 2.0, 3.0]),
            ),
            (
                SparseMatrix::from_diagonal(&real(&[2.0])),
                real(&[1.0]),
                real(&[0.5]),
            ),
            (
                SparseMatrix::from_diagonal(&real(&[2.0, 4.0, 8.0])),
                real(&[0.0, 1.0, 0.0]),
                real(&[0.0, 0.25, 0.0]),
            ),
            // b an eigenvector of a nonsymmetric matrix: A·(1, 1) = 3·(1, 1).
            (
                SparseMatrix::from_coo(
                    2,
                    2,
                    &[
                        (0, 0, T::real(1.0)),
                        (0, 1, T::real(2.0)),
                        (1, 0, T::real(4.0)),
                        (1, 1, T::real(-1.0)),
                    ],
                ),
                real(&[3.0, 3.0]),
                real(&[1.0, 1.0]),
            ),
        ];
        for (a, b, x) in cases {
            let r = bicgstab(&a, &b).unwrap();
            assert!(r.converged && r.iterations == 1 && r.matvecs == 2, "{r:?}");
            for (u, v) in r.x.iter().zip(&x) {
                assert!(magnitude(*u - *v) <= 4.0 * T::EPS);
            }
        }
        // And the matrix of the module's examples, whose solution is (1, 1, 1, 1).
        let a = tridiagonal::<T>(4, -1.0, 2.0, -1.0);
        let r = bicgstab(&a, &real(&[1.0, 0.0, 0.0, 1.0])).unwrap();
        assert!(r.converged && r.x.iter().all(|&x| magnitude(x - T::one()) < 1e-3));
    }
    each_type!(check);
}

#[test]
fn bicgstab_reports_a_breakdown() {
    fn check<T: Num>() {
        // A quarter turn is skew-symmetric, so r̂ᴴ·A·r̂ vanishes at the first step.
        let turn = SparseMatrix::from_coo(2, 2, &[(0, 1, T::real(-1.0)), (1, 0, T::real(1.0))]);
        let r = bicgstab(&turn, &[T::real(1.0), T::real(2.0)]).unwrap();
        assert!(!r.converged && r.reason == StopReason::Breakdown);
        assert!(r.x.iter().all(|x| x.is_zero()));
        // A singular matrix and a right-hand side outside its range end in a report.
        let s = SparseMatrix::from_coo(
            2,
            2,
            &[
                (0, 0, T::real(1.0)),
                (0, 1, T::real(-1.0)),
                (1, 0, T::real(-1.0)),
                (1, 1, T::real(1.0)),
            ],
        );
        let r = within(60, move || {
            bicgstab(&s, &[T::real(1.0), T::real(0.0)]).unwrap()
        });
        assert!(!r.converged);
    }
    each_type!(check);
}

#[test]
fn bicgstab_breaks_down_where_rho_or_omega_vanishes() {
    // The review of round 1 found that no test reached these two: deleting the test of ρ let
    // every test pass. With b = e₀ (so r̂ = r₀ = e₀): A = [[2, 1, 1], [1, 3, 0], [−1, 0, 4]] makes
    // the second ρ = r̂ᴴ·r₁ exactly zero, and A = [[1, 0, 0], [1, 0, 1], [1, −1, 0]] makes
    // tᴴ·s and so ω zero at the first step. scipy's bicgstab reports both, info −10.
    fn check<T: Num>() {
        let matrix = |rows: [[f64; 3]; 3]| {
            let e: Vec<(usize, usize, T)> = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .filter(|&(i, j)| rows[i][j] != 0.0)
                .map(|(i, j)| (i, j, T::real(rows[i][j])))
                .collect();
            SparseMatrix::from_coo(3, 3, &e)
        };
        let b = [T::one(), T::zero(), T::zero()];
        for rows in [
            [[2.0, 1.0, 1.0], [1.0, 3.0, 0.0], [-1.0, 0.0, 4.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        ] {
            let a = matrix(rows);
            let r = bicgstab(&a, &b).unwrap();
            assert_eq!(r.reason, StopReason::Breakdown, "{rows:?}: {r:?}");
            assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
            assert!(r.residual.to_f64() <= 1.0, "worse than the start");
            // GMRES, which cannot break down this way, solves both.
            assert!(gmres(&a, &b).unwrap().converged);
        }
    }
    each_type!(check);
}

#[test]
fn no_method_returns_a_point_worse_than_its_start() {
    // The review of round 1: on a system none of them can solve, the path's Laplacian with a
    // right-hand side outside its range, cg returned a residual of 192.8 and bicgstab one of
    // 94.2 where x = 0 has 6.29, and minres 7.1·10¹⁶. The starting guess is a point reached.
    fn check<T: Num>() {
        let n = 30;
        let a = free_laplacian::<T>(n, false);
        let b: Vec<T> = (0..n).map(|i| T::real(1.0 + i as f64 / 100.0)).collect();
        // A least squares solution to start from, which none can improve on by much.
        let good = minres(&a, &b).unwrap().x;
        for x0 in [None, Some(good)] {
            let mut opts = SolveOptions::new();
            if let Some(x0) = &x0 {
                opts = opts.with_initial_guess(x0);
            }
            let start = match &x0 {
                Some(x0) => true_residual(&a, x0, &b),
                None => norm(&b),
            };
            for m in METHODS {
                let r = within(120, {
                    let (a, b, opts) = (a.clone(), b.clone(), opts.clone());
                    move || solve(m, &a, &b, &opts, None).unwrap()
                });
                assert!(!r.converged, "{m}");
                assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
                assert!(
                    true_residual(&a, &r.x, &b) <= start,
                    "{m} from {}: {:e} against {start:e}",
                    if x0.is_some() { "a guess" } else { "zero" },
                    r.residual.to_f64()
                );
            }
        }
    }
    each_type!(check);
}

#[test]
fn a_starting_guess_far_from_the_solution_does_not_spoil_the_inner_products() {
    // The review of round 1: from x₀ = 10³⁰·(1, …, 1) on the 10 × 10 Laplacian in f32, the
    // inner products scaled for the first residual underflowed once the residual had come down
    // by 10¹⁹, and cg reported NotPositiveDefinite and bicgstab Breakdown on a positive definite
    // matrix. The scalars of both are carried in f64 now.
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>("lap2d10");
        let far = vec![T::real(1e30); 100];
        let opts = SolveOptions::new().with_initial_guess(&far);
        for m in METHODS {
            let r = solve(m, &a, &b, &opts, None).unwrap();
            assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
            assert!(r.converged, "{m}: {:?} after {}", r.reason, r.iterations);
        }
    }
    check::<f32>();
    check::<Complex<f32>>();
}

#[test]
fn converged_means_the_true_residual_passes_the_test() {
    // The audit's p_iter-8: cg and bicgstab tested the residual of their recurrences, and cg
    // said converged on the 1-D Laplacian of 1000 in f32 with a true residual 21 times the
    // tolerance. Here every method, on that system and on larger ones.
    for n in [size(500, 40), size(1000, 60), size(2000, 80)] {
        let a = tridiagonal::<f32>(n, -1.0, 2.0, -1.0);
        let b: Vec<f32> = (0..n)
            .map(|i| ((i as f64 + 1.0) * core::f64::consts::PI / (n as f64 + 1.0)).sin() as f32)
            .collect();
        for (m, r) in all(&a, &b, &SolveOptions::new().with_max_iter(20 * n)) {
            assert_honest(&r, &a, &b, f32::EPSILON.sqrt() as f64, 0.0);
            assert!(
                r.converged || r.reason == StopReason::Stagnated,
                "{m} on {n}: {:?}",
                r.reason
            );
        }
    }
    // In f64 the short recurrences drift too once the tolerance nears the rounding level: the
    // audit found the true residual up to 279 times the tolerance for rtol ≤ 1e-14. (GMRES forms
    // the residual afresh at every restart in any case.)
    let n = size(2000, 60);
    let a = tridiagonal::<f64>(n, -1.0, 2.0, -1.0);
    let b: Vec<f64> = (0..n).map(|i| ((i * 13 + 5) % 17) as f64 - 8.0).collect();
    for rtol in [1e-12, 1e-14] {
        let opts = SolveOptions::new().with_rtol(rtol).with_max_iter(20 * n);
        for m in ["cg", "minres", "bicgstab"] {
            let r = solve(m, &a, &b, &opts, None).unwrap();
            assert_honest(&r, &a, &b, rtol, 0.0);
            assert!(
                r.converged || r.reason == StopReason::Stagnated,
                "{m} at {rtol}: {:?}",
                r.reason
            );
        }
    }
}

#[test]
fn right_hand_sides_of_any_scale_are_solved() {
    // The audit's p_iter-12: r·z left the range of the type and cg and bicgstab returned None:
    // f32 with b of 10¹⁹ or 10⁻²³ (atol 0), f64 with b of 10¹⁶⁰ or 10⁻¹⁷⁰.
    fn check<T: Num>(scales: &[f64]) {
        let a = tridiagonal::<T>(4, -1.0, 2.0, -1.0);
        let x = [4.0, 7.0, 8.0, 6.0];
        for &scale in scales {
            let b: Vec<T> = (1..=4)
                .map(|k| c::<T>(k as f64 * scale, -(k as f64) * scale))
                .collect();
            for (m, r) in all(&a, &b, &SolveOptions::new()) {
                assert!(r.converged, "{m} at {scale:e}: {:?}", r.reason);
                for (k, &y) in r.x.iter().enumerate() {
                    let want = c::<T>(x[k] * scale, -x[k] * scale);
                    assert!(
                        magnitude(y - want) <= 1e-3 * x[k] * scale,
                        "{m} at {scale:e}"
                    );
                }
            }
        }
    }
    // Down to the subnormal numbers too, 10⁻⁴⁰ in f32 and 10⁻³¹⁰ in f64, which the review of
    // round 1 found made MINRES's α/β overflow and end in NonFinite.
    check::<f32>(&[1e19, 1e20, 1e-18, 1e-23, 1e-30, 1e-40]);
    check::<Complex<f32>>(&[1e19, 1e-23, 1e-40]);
    check::<f64>(&[1e160, 1e300, 1e-170, 1e-300, 1e-310]);
    check::<Complex<f64>>(&[1e160, 1e-170, 1e-310]);
}

#[test]
fn every_method_stops_at_the_first_iterate_that_passes_the_test() {
    // Where its recurrence follows the residual, a method stops at the first iterate whose
    // residual passes the test: one iteration fewer leaves it short. In double precision on a
    // well-conditioned system it also looks only once, at the end. The review of round 1 turned
    // the sign of a term of MINRES's recurrence for the residual with a preconditioner, which
    // puts the estimate 10 % off, and no test noticed.
    // In double precision the estimates follow the residual to about 10⁻¹⁴ of it, so the test is
    // made at thirty tolerances from 10⁻¹ to 10⁻¹⁰: an estimate a few per cent off makes one of
    // them see the method stop a step late, or look and go on. In single precision the residual
    // a recurrence carries drifts by more than that, and one tolerance is checked.
    fn check<T: Num>(name: &str) {
        let (a, b, _) = problem::<T>(name);
        let double = T::EPS < 1e-10;
        let tolerances: Vec<f64> = if double {
            (0..30)
                .map(|j| 10f64.powf(-1.0 - 9.0 * j as f64 / 29.0))
                .collect()
        } else {
            vec![1e-3]
        };
        let jacobi = Jacobi::new(&a).unwrap();
        let ic = Ic0::new(&a).unwrap();
        let preconditioners: [Option<&dyn Preconditioner<T>>; 3] = [None, Some(&jacobi), Some(&ic)];
        for rtol in tolerances {
            let opts = SolveOptions::new().with_rtol(T::Real::from_f64(rtol));
            for m in METHODS {
                for (k, &pre) in preconditioners.iter().enumerate() {
                    let r = solve(m, &a, &b, &opts, pre).unwrap();
                    assert!(r.converged, "{m} {k} at {rtol:e}: {r:?}");
                    let short = opts.clone().with_max_iter(r.iterations - 1);
                    let s = solve(m, &a, &b, &short, pre).unwrap();
                    assert!(
                        !s.converged,
                        "{m} {k} at {rtol:e}: converged after {} iterations already, stopped after {}",
                        r.iterations - 1,
                        r.iterations
                    );
                    if double && (m == "cg" || m == "minres") {
                        assert_eq!(
                            r.matvecs,
                            r.iterations + 1,
                            "{m} {k} at {rtol:e}: looked more than once"
                        );
                    }
                }
            }
        }
    }
    // With IC(0) on the 12 x 12 Laplacian, the turned sign puts MINRES's estimate below the
    // tolerance 10⁻¹ after the first step, where the residual is not.
    for name in ["scaled2d12", "lap2d12"] {
        check::<f32>(name);
        check::<f64>(name);
        check::<Complex<f32>>(name);
        check::<Complex<f64>>(name);
    }
}

#[test]
fn the_default_absolute_tolerance_is_zero() {
    // The audit's p_iter-13: with atol = ε a small right-hand side was solved by x = 0.
    assert_eq!(SolveOptions::<f32>::new().atol, 0.0);
    let a = tridiagonal::<f32>(4, -1.0, 2.0, -1.0);
    for (m, r) in all(&a, &[1e-8, 2e-8, 3e-8, 4e-8], &SolveOptions::new()) {
        assert!(r.converged && r.iterations > 0, "{m}");
        assert!((r.x[1] - 7e-8).abs() <= 1e-11, "{m}: {:?}", r.x);
    }
    let a = tridiagonal::<f64>(4, -1.0, 2.0, -1.0);
    for (m, r) in all(&a, &[1e-17, 2e-17, 3e-17, 4e-17], &SolveOptions::new()) {
        assert!(r.converged && (r.x[2] - 8e-17).abs() <= 1e-24, "{m}");
    }
}

#[test]
fn zero_tolerances_go_as_far_as_the_arithmetic_and_keep_the_point() {
    // The audit's p_iter-12, its last case: rtol = atol = 0, "as far as the precision goes",
    // ended in None after the recurrence had reached 10⁻¹⁷ and lost the iterate.
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>("lap2d8");
        let opts = SolveOptions::new()
            .with_rtol(T::Real::ZERO)
            .with_atol(T::Real::ZERO)
            .with_max_iter(500);
        let exact = a
            .to_matrix()
            .lu()
            .solve(&Matrix::from_column_major(64, 1, b.clone()))
            .unwrap();
        for (m, r) in all(&a, &b, &opts) {
            assert_honest(&r, &a, &b, 0.0, 0.0);
            assert!(
                matches!(r.reason, StopReason::Converged | StopReason::Stagnated),
                "{m}: {:?}",
                r.reason
            );
            assert!(r.iterations < 500, "{m} ran to its budget");
            let error: f64 = (0..64)
                .map(|i| magnitude(r.x[i] - exact[(i, 0)]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(error <= 1e3 * T::EPS * norm(&r.x), "{m}: {error:e}");
        }
    }
    each_type!(check);
}

// ---------------------------------------------------------------------------------------------
// What each method promises of the matrices it is for
// ---------------------------------------------------------------------------------------------

#[test]
fn cg_finds_out_a_matrix_that_is_not_positive_definite() {
    fn check<T: Num>() {
        // Indefinite, negative definite, and indefinite with a positive first curvature.
        let indefinite = SparseMatrix::from_diagonal(&[T::real(1.0), T::real(-1.0)]);
        let r = cg(&indefinite, &[T::real(1.0), T::real(1.0)]).unwrap();
        assert!(!r.converged && r.reason == StopReason::NotPositiveDefinite);
        let negative = tridiagonal::<T>(10, 1.0, -2.0, 1.0);
        let r = cg(&negative, &[T::real(1.0); 10]).unwrap();
        assert!(r.reason == StopReason::NotPositiveDefinite && r.iterations == 1);
        let (shifted, b, _) = problem::<T>("shifted2d10");
        let r = cg(&shifted, &b).unwrap();
        assert!(!r.converged, "{r:?}");
        assert_honest(&r, &shifted, &b, T::EPS.sqrt(), 0.0);
    }
    each_type!(check);
}

/// The preconditioner −I, Hermitian and negative definite.
struct Negative;

impl<T: Num> Preconditioner<T> for Negative {
    fn apply(&self, r: &[T], z: &mut [T]) {
        for (z, &r) in z.iter_mut().zip(r) {
            *z = -r;
        }
    }
}

#[test]
fn cg_and_minres_find_out_a_preconditioner_that_is_not_positive_definite() {
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>("lap2d6");
        for m in ["cg", "minres"] {
            let r = solve(m, &a, &b, &SolveOptions::new(), Some(&Negative)).unwrap();
            assert!(
                !r.converged && r.reason == StopReason::NotPositiveDefinite,
                "{m}: {r:?}"
            );
            assert_eq!(r.iterations, 0, "{m}");
        }
        // GMRES and BiCGSTAB take any nonsingular preconditioner, this one too.
        for m in ["gmres", "bicgstab"] {
            let r = solve(m, &a, &b, &SolveOptions::new(), Some(&Negative)).unwrap();
            assert!(r.converged, "{m}");
        }
    }
    each_type!(check);
}

#[test]
fn minres_solves_a_singular_consistent_system_with_the_solution_of_least_norm() {
    // The 2-D graph Laplacian is singular, its null space the constants; a right-hand side that
    // sums to zero is in its range. From x = 0 MINRES stays in the range, so its solution is the
    // one of least norm: it sums to zero as well.
    fn check<T: Num>() {
        let (a, b, _) = problem::<T>("neumann2d10");
        let rtol = reference_rtol::<T>();
        let r = minres_with(
            &a,
            &b,
            &SolveOptions::new().with_rtol(T::Real::from_f64(rtol)),
            None,
        )
        .unwrap();
        assert!(r.converged);
        assert_honest(&r, &a, &b, rtol, 0.0);
        let total = r.x.iter().fold(T::zero(), |s, &x| s + x);
        assert!(
            magnitude(total) <= 1e3 * T::EPS * norm(&r.x) * 10.0,
            "the sum is {total:?}"
        );
        // The same from a starting guess: the null-space part of the guess stays.
        let (_, _, x0) = problem::<T>("neumann2d10");
        let shift = x0.iter().fold(T::zero(), |s, &x| s + x);
        let r = minres_with(
            &a,
            &b,
            &SolveOptions::new()
                .with_rtol(T::Real::from_f64(rtol))
                .with_initial_guess(&x0),
            None,
        )
        .unwrap();
        assert!(r.converged);
        let total = r.x.iter().fold(T::zero(), |s, &x| s + x);
        assert!(magnitude(total - shift) <= 1e3 * T::EPS * norm(&r.x) * 10.0);
    }
    each_type!(check);
}

#[test]
fn minres_reports_a_singular_system_that_is_not_consistent() {
    fn check<T: Num>() {
        let s = SparseMatrix::from_coo(
            2,
            2,
            &[
                (0, 0, T::real(1.0)),
                (0, 1, T::real(-1.0)),
                (1, 0, T::real(-1.0)),
                (1, 1, T::real(1.0)),
            ],
        );
        let b = [T::real(1.0), T::real(0.0)];
        let r = within(60, move || minres(&s, &b).unwrap());
        assert!(!r.converged && r.reason == StopReason::Breakdown, "{r:?}");
        // The residual is that of a least squares solution: the part of b in the null space.
        assert!((r.residual.to_f64() - 0.5f64.sqrt()).abs() <= 8.0 * T::EPS);
        // The Neumann Laplacian with a right-hand side that does not sum to zero: its part along
        // the constants, Σ bᵢ / √n = 1 / 10, is the least squares residual.
        let (a, mut b, _) = problem::<T>("neumann2d10");
        b[0] += T::one();
        let r = minres(&a, &b).unwrap();
        assert!(!r.converged && r.reason == StopReason::Breakdown, "{r:?}");
        assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
        assert!(
            r.residual.to_f64() <= 0.1 * 1.01,
            "{:e}",
            r.residual.to_f64()
        );
    }
    each_type!(check);
}

/// The Laplacian of a path (`two_d` false) or of a `k × k` grid with free ends: singular, its
/// null space the constants.
fn free_laplacian<T: Num>(k: usize, two_d: bool) -> SparseMatrix<T> {
    if two_d {
        return problem::<T>(&format!("neumann2d{k}")).0;
    }
    let mut e = Vec::new();
    for i in 0..k {
        e.push((i, i, T::real(f64::from((i > 0) as u8 + (i + 1 < k) as u8))));
        if i + 1 < k {
            e.push((i, i + 1, T::real(-1.0)));
            e.push((i + 1, i, T::real(-1.0)));
        }
    }
    SparseMatrix::from_coo(k, k, &e)
}

/// The three right-hand sides of the review of round 1 that are not in the range of a free
/// Laplacian of `n` nodes: a ramp on a constant, one that sums to 1, and `e₀`.
fn outside_the_range(n: usize) -> [Vec<f64>; 3] {
    let u = |i: usize| ((7 * i + 3) % 11) as f64;
    [
        (0..n).map(|i| 1.0 + i as f64 / 100.0).collect(),
        (0..n)
            .map(|i| u(i) - u(n - 1 - i) + f64::from(u8::from(i == 0)))
            .collect(),
        (0..n).map(|i| f64::from(u8::from(i == 0))).collect(),
    ]
}

/// `|Σ bᵢ| / √n`: the norm of the part of `b` along the constants, which is the residual of a
/// least squares solution of a system with a free Laplacian.
fn least_squares_residual<T: Num>(b: &[T]) -> f64 {
    let (re, im) = b.iter().fold((0.0, 0.0), |(re, im), &x| {
        (re + x.re().to_f64(), im + x.im().to_f64())
    });
    re.hypot(im) / (b.len() as f64).sqrt()
}

/// `b` with each element moved by up to two units in the last place: element `i` times
/// `1 + k·ε` for `k = ((7i² + 3i + s) mod 5) − 2`. A right-hand side as exact as `b`, whose
/// rounding takes the method down another path.
fn nudged<T: Num>(b: &[T], s: usize) -> Vec<T> {
    b.iter()
        .enumerate()
        .map(|(i, &x)| {
            let k = ((7 * i * i + 3 * i + s) % 5) as f64 - 2.0;
            x * T::real(1.0 + k * T::EPS)
        })
        .collect()
}

/// The Laplacian of a path of `n` nodes with free ends taken to `U·L·Uᴴ` by a diagonal unitary
/// `U`: a Hermitian matrix with complex elements that is exactly singular in every type. The
/// element `(j, j + 1)` is `−gⱼ` and `(j + 1, j)` its conjugate, for Gaussian integers `gⱼ` that
/// run through `3 + 4i`, `5 + 12i`, `8 + 15i` and `7 + 24i` from the `shift`-th on, and the
/// diagonal holds the sums of the integer moduli `|gⱼ|` of the row. The null space is spanned by
/// `u` with `u₀ = 1` and `uⱼ₊₁ = uⱼ·ḡⱼ / |gⱼ|`, returned in `f64`.
fn gaussian_path<T: Num>(n: usize, shift: usize) -> (SparseMatrix<T>, Vec<(f64, f64)>) {
    const G: [(f64, f64, f64); 4] = [
        (3.0, 4.0, 5.0),
        (5.0, 12.0, 13.0),
        (8.0, 15.0, 17.0),
        (7.0, 24.0, 25.0),
    ];
    let edge = |j: usize| G[(j + shift) % 4];
    let mut e = Vec::new();
    for i in 0..n {
        let left = if i > 0 { edge(i - 1).2 } else { 0.0 };
        let right = if i + 1 < n { edge(i).2 } else { 0.0 };
        e.push((i, i, T::real(left + right)));
        if i + 1 < n {
            let (p, q, _) = edge(i);
            e.push((i, i + 1, c::<T>(-p, -q)));
            e.push((i + 1, i, c::<T>(-p, q)));
        }
    }
    let mut u = vec![(1.0, 0.0); n];
    for j in 0..n - 1 {
        let ((re, im), (p, q, m)) = (u[j], edge(j));
        u[j + 1] = ((re * p + im * q) / m, (im * p - re * q) / m);
    }
    (SparseMatrix::from_coo(n, n, &e), u)
}

/// `|uᴴ·b| / ‖u‖`, in `f64`: the norm of the part of `b` along `u`, which for a null vector `u`
/// spanning the null space is the residual of a least squares solution.
fn along<T: Num>(u: &[(f64, f64)], b: &[T]) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (&(p, q), &x) in u.iter().zip(b) {
        let (s, t) = (x.re().to_f64(), x.im().to_f64());
        re += p * s + q * t;
        im += p * t - q * s;
    }
    let length = u.iter().map(|&(p, q)| p * p + q * q).sum::<f64>().sqrt();
    re.hypot(im) / length
}

#[test]
fn minres_finds_a_least_squares_solution_of_a_system_that_is_not_consistent() {
    // The review of round 1, its B2: past the least squares residual the iterates drifted along
    // the null space while the estimate went on falling, and MINRES returned points with
    // residuals of 10⁴ to 10¹⁹ (the path of 30 nodes: 7.1·10¹⁶ against a least squares residual
    // of 6.27 and ‖b‖ = 6.29). Now it stops there, says why, and returns a least squares
    // solution: within 1 % of the least squares residual, and never below it.
    //
    // Then x86_64: the complex path of the end of this test, whose phases come from the
    // platform's sine and cosine, ran to the iteration limit there. In its second Lanczos process
    // the step past a near null vector brought the residual down by 2 % where the estimate
    // claimed 18 %; the claim counted as borne out, nothing watched the rest of the drift, and
    // the iterates went on to a residual of 10¹⁶. Claims are now measured against the smallest
    // residual formed afresh, on the evidence of a null vector anywhere in the run, and a claim
    // borne out by less than half stays watched. The same weakness made the old method run to the
    // limit on exactly singular systems: the right-hand sides nudged by a unit or two in the last
    // place (the 1-D Laplacian of 10 nodes and the 16 × 16 grid in f64 and Complex<f64>) and the
    // exactly singular complex paths (four of the twelve in Complex<f32>, one in Complex<f64>).
    fn check<T: Num>() {
        let (mut worst, mut most) = (1.0f64, 0);
        let mut verify = |a: &SparseMatrix<T>, b: Vec<T>, least: f64, label: &str| {
            let (a2, b2) = (a.clone(), b.clone());
            let r = within(120, move || minres(&a2, &b2).unwrap());
            assert_honest(&r, a, &b, T::EPS.sqrt(), 0.0);
            assert_eq!(r.reason, StopReason::Breakdown, "{label}: {r:?}");
            let truth = true_residual(a, &r.x, &b);
            assert!(
                truth <= 1.01 * least && truth >= least * (1.0 - 1e-9),
                "{label}: residual {truth:e}, least squares {least:e}"
            );
            worst = worst.max(truth / least);
            most = most.max(r.iterations);
        };
        for (k, two_d) in [
            (10, false),
            (30, false),
            (100, false),
            (6, true),
            (10, true),
            (16, true),
        ] {
            let a = free_laplacian::<T>(k, two_d);
            let n = a.rows();
            for (which, b) in outside_the_range(n).into_iter().enumerate() {
                let b: Vec<T> = b.into_iter().map(T::real).collect();
                let label = format!("{k}{} b{which}", if two_d { "²" } else { "" });
                verify(&a, b.clone(), least_squares_residual(&b), &label);
                for s in 1..=4 {
                    let b = nudged(&b, s);
                    let least = least_squares_residual(&b);
                    verify(&a, b, least, &format!("{label} nudged {s}"));
                }
            }
        }
        if T::IS_COMPLEX {
            // A complex right-hand side, and exactly singular Hermitian matrices with complex
            // elements.
            let rhs = |n: usize| -> Vec<T> {
                (0..n)
                    .map(|i| c::<T>(1.0 + i as f64 / 100.0, ((5 * i + 1) % 7) as f64 / 7.0))
                    .collect()
            };
            for n in [10, 30, 100] {
                for shift in 0..4 {
                    let (a, u) = gaussian_path::<T>(n, shift);
                    let null: Vec<T> = u.iter().map(|&(re, im)| c::<T>(re, im)).collect();
                    assert!(
                        norm(&a.mul_vec(&null)) <= 1e3 * T::EPS * (n as f64).sqrt(),
                        "u is not a null vector"
                    );
                    let b = rhs(n);
                    let least = along(&u, &b);
                    verify(&a, b, least, &format!("complex path {n} from g{shift}"));
                }
            }
        }
        eprintln!(
            "{}: residual at most {worst:.9} times the least squares one, at most {most} iterations",
            core::any::type_name::<T>()
        );
        // The path's Laplacian with the element (j, j + 1) times e^(iθj) and (j + 1, j) times its
        // conjugate, the phases computed by the platform's sine and cosine: singular only up to
        // the rounding of its elements, so a matrix of its own on every platform, singular to
        // working precision but not exactly. The null space of the matrix it rounds is spanned by
        // u with uⱼ = e^(−iθj(j−1)/2), and |uᴴ·b| / √n is the least squares residual there. The
        // method ends with Breakdown as on a singular matrix, or at the iteration limit (aarch64
        // and x86_64 both say Breakdown, x86_64 said IterationLimit before the fix); either way
        // with a point no worse than a least squares solution of that singular matrix. Below it
        // the residual may go, the matrix not being singular.
        if T::IS_COMPLEX {
            let n = 30;
            let theta = 0.375;
            let path = free_laplacian::<T>(n, false);
            let phase = |j: usize| c::<T>((theta * j as f64).cos(), (theta * j as f64).sin());
            let a = path.map(|i, j, x| match j.cmp(&i) {
                core::cmp::Ordering::Greater => x * phase(i),
                core::cmp::Ordering::Less => x * phase(j).conj(),
                core::cmp::Ordering::Equal => x,
            });
            let u: Vec<T> = (0..n)
                .map(|j| {
                    let angle = -theta * (j * j.saturating_sub(1)) as f64 / 2.0;
                    c::<T>(angle.cos(), angle.sin())
                })
                .collect();
            let null = a.mul_vec(&u);
            assert!(norm(&null) <= 1e3 * T::EPS, "u is not a null vector");
            let b: Vec<T> = (0..n)
                .map(|i| c::<T>(1.0 + i as f64 / 100.0, ((5 * i + 1) % 7) as f64 / 7.0))
                .collect();
            let projection = u
                .iter()
                .zip(&b)
                .fold(T::zero(), |s, (&u, &b)| s + u.conj() * b);
            let least = magnitude(projection) / (n as f64).sqrt();
            let r = minres(&a, &b).unwrap();
            assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
            assert!(
                matches!(r.reason, StopReason::Breakdown | StopReason::IterationLimit),
                "{r:?}"
            );
            let truth = true_residual(&a, &r.x, &b);
            assert!(
                truth <= 1.01 * least,
                "rounded phases: {truth:e} against {least:e}"
            );
            eprintln!(
                "{}: rounded phases: {:?} after {} iterations, residual {:.9} times the least squares one",
                core::any::type_name::<T>(),
                r.reason,
                r.iterations,
                truth / least
            );
        }
    }
    each_type!(check);
}

#[test]
fn minres_with_a_preconditioner_keeps_the_best_point_of_a_system_that_is_not_consistent() {
    // With Jacobi the least squares solution MINRES finds is one in the norm of M⁻¹, so its
    // 2-norm residual is above the least squares one, but it is still never worse than b's.
    fn check<T: Num>() {
        for (k, two_d) in [(10, false), (30, false), (6, true), (10, true)] {
            let a = free_laplacian::<T>(k, two_d);
            let m = Jacobi::new(&a).unwrap();
            for b in outside_the_range(a.rows()) {
                let b: Vec<T> = b.into_iter().map(T::real).collect();
                let r = minres_with(&a, &b, &SolveOptions::new(), Some(&m)).unwrap();
                assert_honest(&r, &a, &b, T::EPS.sqrt(), 0.0);
                assert_eq!(r.reason, StopReason::Breakdown, "{k} {two_d}: {r:?}");
                let truth = true_residual(&a, &r.x, &b);
                assert!(
                    truth <= 1.05 * least_squares_residual(&b) && truth <= norm(&b),
                    "{k} {two_d}: {truth:e}"
                );
            }
        }
    }
    each_type!(check);
}

/// The review of round 1's example of B1: `−(c·u')' + 10⁻³·u` on `n` points with
/// `c_{i+1/2} = e^(a·sin(0.3·i))`, and `bᵢ = ((7i + 3) mod 11) − 5`; for a complex type the
/// matrix and the right-hand side are taken to `U·A·Uᴴ` and `U·b` for a diagonal unitary `U`,
/// a system with complex elements and the same solution up to `U`.
fn exponential_diffusion<T: Num>(n: usize, amplitude: f64) -> (SparseMatrix<T>, Vec<T>) {
    let coefficient = |i: usize| (amplitude * (0.3 * i as f64).sin()).exp();
    let unitary = |i: usize| {
        let angle = if T::IS_COMPLEX { 0.75 * i as f64 } else { 0.0 };
        (angle.cos(), angle.sin())
    };
    let mut e = Vec::new();
    for i in 0..n {
        e.push((
            i,
            i,
            c::<T>(coefficient(i) + coefficient(i + 1) + 1e-3, 0.0),
        ));
        if i + 1 < n {
            // (U·A·Uᴴ)ᵢ,ᵢ₊₁ = uᵢ·aᵢ,ᵢ₊₁·ūᵢ₊₁.
            let (p, q) = unitary(i);
            let (s, t) = unitary(i + 1);
            let (re, im) = (p * s + q * t, q * s - p * t);
            let v = -coefficient(i + 1);
            e.push((i, i + 1, c::<T>(v * re, v * im)));
            e.push((i + 1, i, c::<T>(v * re, -v * im)));
        }
    }
    let b = (0..n)
        .map(|i| {
            let (p, q) = unitary(i);
            let v = ((7 * i + 3) % 11) as f64 - 5.0;
            c::<T>(v * p, v * q)
        })
        .collect();
    (SparseMatrix::from_coo(n, n, &e), b)
}

#[test]
fn minres_with_a_preconditioner_goes_on_while_the_residual_it_minimises_falls() {
    // The review of round 1, its B1: with Jacobi, preconditioned MINRES minimises the residual
    // in the norm of M⁻¹, whose 2-norm may stand still or rise for many steps, and the stall
    // rule that watched the 2-norm ended valid runs with Stagnated: after 17 iterations at
    // 8.6 % of ‖b‖ (f64, n = 100, a = 7, rtol 10⁻⁶) and after 56 at 3.9 times the tolerance
    // (f32, n = 80, a = 2.5, rtol 10⁻³). scipy's MINRES with the same preconditioner reaches
    // the tolerance at 59 and 65 (review1/py/bug_minres_scipy64.py and bug_minres_scipy.py;
    // elements from exp and sin are not exact, hence the margin).
    fn check<T: Num>() {
        let (n, amplitude, rtol, scipy) = if T::EPS < 1e-10 {
            (100, 7.0, 1e-6, 59)
        } else {
            (80, 2.5, 1e-3, 65)
        };
        let (a, b) = exponential_diffusion::<T>(n, amplitude);
        let m = Jacobi::new(&a).unwrap();
        let opts = SolveOptions::new().with_rtol(T::Real::from_f64(rtol));
        let r = minres_with(&a, &b, &opts, Some(&m)).unwrap();
        assert_honest(&r, &a, &b, rtol, 0.0);
        assert!(r.converged, "{:?} after {}", r.reason, r.iterations);
        eprintln!(
            "{}: minres with Jacobi {} iterations (scipy {scipy})",
            core::any::type_name::<T>(),
            r.iterations
        );
        assert!(
            r.iterations.abs_diff(scipy) <= 4,
            "{} against scipy's {scipy}",
            r.iterations
        );
        // The conjugate gradient method agrees on this positive definite system.
        let g = cg_with(&a, &b, &opts, Some(&m)).unwrap();
        assert!(
            g.converged && g.iterations.abs_diff(r.iterations) <= 20,
            "{}",
            g.iterations
        );
    }
    each_type!(check);
}

/// A random unsymmetric matrix whose real diagonal exceeds the sum of its row and its column
/// off the diagonal, halved, by a quarter at least: its Hermitian part is positive definite.
fn positive_real<T: Num>(r: &mut Rng, n: usize, density: usize) -> SparseMatrix<T> {
    let mut entries = Vec::new();
    let mut sums = vec![0.0f64; n];
    for _ in 0..density {
        let (i, j) = (r.below(n), r.below(n));
        if i != j {
            let x = T::random(r);
            sums[i] += magnitude(x) / 2.0;
            sums[j] += magnitude(x) / 2.0;
            entries.push((i, j, x));
        }
    }
    for (i, &sum) in sums.iter().enumerate() {
        entries.push((i, i, c::<T>(sum + 0.25 + r.unit().abs(), r.unit())));
    }
    SparseMatrix::from_coo(n, n, &entries)
}

#[test]
fn the_methods_solve_random_systems_of_every_element_type() {
    // Hermitian positive definite systems with complex elements for cg and minres, and
    // unsymmetric ones for gmres and bicgstab, against the dense LU solution. The Hermitian
    // part of each is positive definite with its eigenvalues above a quarter (a diagonal that
    // dominates its row, and its row and column together for the unsymmetric ones), so
    // ‖A⁻¹‖₂ ≤ 4 and ‖x − x*‖ ≤ 4·‖b − A·x‖. Restarted GMRES converges on such a matrix
    // whatever the restart length (Elman, 1982).
    fn check<T: Num>() {
        let mut rng = Rng(0x5eed_0101);
        let rtol = reference_rtol::<T>();
        let opts = SolveOptions::new().with_rtol(T::Real::from_f64(rtol));
        for (n, density) in [
            (1, 0),
            (3, 4),
            (17, 40),
            (60, 300),
            (size(200, 40), size(1200, 120)),
        ] {
            let h = hermitian::<T>(&mut rng, n, density);
            let u = positive_real::<T>(&mut rng, n, density);
            let b: Vec<T> = (0..n).map(|_| T::random(&mut rng)).collect();
            for (methods, a) in [
                (&["cg", "minres", "gmres", "bicgstab"][..], &h),
                (&["gmres", "bicgstab"][..], &u),
            ] {
                let exact = a
                    .to_matrix()
                    .lu()
                    .solve(&Matrix::from_column_major(n, 1, b.clone()))
                    .unwrap();
                for &m in methods {
                    let r = solve(m, a, &b, &opts, None).unwrap();
                    assert!(r.converged, "{m} on {n}: {:?}", r.reason);
                    assert_honest(&r, a, &b, rtol, 0.0);
                    let error: f64 = (0..n)
                        .map(|i| magnitude(r.x[i] - exact[(i, 0)]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let bound = 4.0 * r.residual.to_f64() + 64.0 * n as f64 * T::EPS * norm(&r.x);
                    assert!(
                        error <= bound,
                        "{m} on {n}: error {error:e} against {bound:e}"
                    );
                }
            }
        }
    }
    each_type!(check);
}

#[test]
fn the_result_counts_its_products() {
    // cg: one product an iteration and one for the residual formed afresh at the end; with a
    // starting guess one more for its residual. bicgstab: two an iteration.
    let (a, b, x0) = problem::<f64>("lap2d8");
    let r = cg(&a, &b).unwrap();
    assert_eq!(r.matvecs, r.iterations + 1);
    let r = cg_with(&a, &b, &SolveOptions::new().with_initial_guess(&x0), None).unwrap();
    assert_eq!(r.matvecs, r.iterations + 2);
    let r = minres(&a, &b).unwrap();
    assert_eq!(r.matvecs, r.iterations + 1);
    let r = bicgstab(&a, &b).unwrap();
    assert!(
        r.matvecs == 2 * r.iterations + 1 || r.matvecs == 2 * r.iterations,
        "{r:?}"
    );
    // gmres forms the residual afresh at the end of every cycle.
    let r = gmres_with(&a, &b, &SolveOptions::new().with_restart(5), None).unwrap();
    assert_eq!(r.matvecs, r.iterations + r.iterations.div_ceil(5));
}
