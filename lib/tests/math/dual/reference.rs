//! Every function and its derivatives against SymPy's derivatives evaluated by mpmath at 50 digits
//! (`data/gen_reference.py`), in `f64` and `f32`, through every way the crate forms them: `Dual`
//! for the first derivative, `HyperDual` and `Dual<Dual<_>>` for the second, `Dual<HyperDual<_>>`
//! and `HyperDual<Dual<_>>` for the third, and the hyper-dual cross term `f′·d + f″·b·c`.
//!
//! Errors are in units in the last place of the reference rounded to the type (an `f64` reference
//! is itself within half an ulp of the exact value); a component whose reference is outside the
//! normal range of the type, or is not finite, is not compared. `largest_errors_seen` prints the
//! table of the largest error of each function and component (run with `--nocapture`), and
//! asserts the bounds as the other test does.

use a::math::dual::{Dual, DualNum, HyperDual};
use std::collections::BTreeMap;

const DATA: &str = include_str!("data/reference.txt");

/// A function of the data by its name and parameter, written once for every type.
pub fn apply<D: DualNum>(name: &str, p: f64, x: D) -> D {
    let c = |v: f64| D::from_f64(v);
    match name {
        "recip" => x.recip(),
        "sqrt" => x.sqrt(),
        "cbrt" => x.cbrt(),
        "exp" => x.exp(),
        "exp2" => x.exp2(),
        "exp_m1" => x.exp_m1(),
        "ln" => x.ln(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "ln_1p" => x.ln_1p(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "asinh" => x.asinh(),
        "acosh" => x.acosh(),
        "atanh" => x.atanh(),
        "powi" => x.powi(p as i32),
        "powf" => x.powf(D::from_f64(p).re()),
        "log" => x.log(D::from_f64(p).re()),
        "g1" => x.exp().sin() / (x * x + c(1.0)),
        "g2" => (x * x).ln_1p() * x.atan().powi(3),
        "g3" => x.cosh().sqrt() * (x * c(0.5)).tanh() - x.powf(D::from_f64(2.5).re()),
        "g4" => (-(x * x)).exp() * (x * c(3.0)).cos(),
        "g5" => (x.powi(3) - x).asinh() / (x.sin() + c(2.0)),
        "g6" => x.sin().atan2((x * c(2.0)).cos() + c(2.0)) * (x * x + c(1.0)).sqrt(),
        _ => panic!("no function {name}"),
    }
}

/// A function of two variables of the data by its name.
pub fn apply2<D: DualNum>(name: &str, x: D, y: D) -> D {
    match name {
        "atan2" => x.atan2(y),
        "hypot" => x.hypot(y),
        "pow" => x.pow(y),
        _ => panic!("no function {name}"),
    }
}

/// One point of a function of one variable: the reference value and three derivatives.
pub struct Unary {
    pub name: String,
    pub param: f64,
    pub both: bool,
    pub x: f64,
    pub f: [f64; 4],
}

/// One point of a function of two variables: the value and the first and second partials.
pub struct Binary {
    pub name: String,
    pub both: bool,
    pub x: f64,
    pub y: f64,
    /// `f, fx, fy, fxx, fxy, fyy`.
    pub f: [f64; 6],
}

fn number(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| panic!("{s} is not a number"))
}

pub fn unary_points() -> Vec<Unary> {
    DATA.lines()
        .filter(|l| l.starts_with("u "))
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(w.len(), 9, "{l}");
            Unary {
                name: w[1].to_string(),
                param: if w[2] == "-" { 0.0 } else { number(w[2]) },
                both: w[3] == "b",
                x: number(w[4]),
                f: [number(w[5]), number(w[6]), number(w[7]), number(w[8])],
            }
        })
        .collect()
}

pub fn binary_points() -> Vec<Binary> {
    DATA.lines()
        .filter(|l| l.starts_with("b "))
        .map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(w.len(), 11, "{l}");
            let mut f = [0.0; 6];
            for (k, v) in f.iter_mut().enumerate() {
                *v = number(w[5 + k]);
            }
            Binary {
                name: w[1].to_string(),
                both: w[2] == "b",
                x: number(w[3]),
                y: number(w[4]),
                f,
            }
        })
        .collect()
}

/// The float types under test, as `f64` values and their precision.
trait Precision: DualNum<Float = Self> + Into<f64> {
    const BITS: i32;
    const MIN: f64;
    const MAX: f64;
    const NAME: &'static str;
}

impl Precision for f64 {
    const BITS: i32 = 53;
    const MIN: f64 = f64::MIN_POSITIVE;
    const MAX: f64 = f64::MAX;
    const NAME: &'static str = "f64";
}

impl Precision for f32 {
    const BITS: i32 = 24;
    const MIN: f64 = f32::MIN_POSITIVE as f64;
    const MAX: f64 = f32::MAX as f64;
    const NAME: &'static str = "f32";
}

/// The error of `got` against the reference `want` in ulps of `want` in the type `P`, or `None`
/// where `want` is not a normal number of the type. A zero reference takes the ulp of 1.
fn ulps<P: Precision>(got: P, want: f64) -> Option<f64> {
    let got: f64 = got.into();
    if !want.is_finite() || want.abs() > P::MAX || (want != 0.0 && want.abs() < P::MIN) {
        return None;
    }
    let scale = if want == 0.0 { 1.0 } else { want.abs() };
    // 2^e in two halves, as 2^e alone underflows to 0 through `powi`'s reciprocal for e < −1023.
    let e = scale.log2().floor() as i32 - (P::BITS - 1);
    let ulp = 2f64.powi(e / 2) * 2f64.powi(e - e / 2);
    if got == want {
        return Some(0.0);
    }
    Some(if got.is_nan() {
        f64::INFINITY
    } else {
        (got - want).abs() / ulp
    })
}

/// The largest error of each function, component and type, with the point where it was seen.
#[derive(Default)]
struct Table {
    worst: BTreeMap<(String, &'static str, &'static str), (f64, f64)>,
    compared: usize,
}

impl Table {
    fn record(&mut self, func: &str, part: &'static str, ty: &'static str, x: f64, e: Option<f64>) {
        let Some(e) = e else { return };
        self.compared += 1;
        let slot = self
            .worst
            .entry((func.to_string(), part, ty))
            .or_insert((0.0, x));
        if e > slot.0 || e.is_nan() {
            *slot = (e, x);
        }
    }
}

fn label(p: &Unary) -> String {
    match p.name.as_str() {
        "powi" | "powf" | "log" => format!("{}({})", p.name, p.param),
        _ => p.name.clone(),
    }
}

/// Every form of every function at every point, in the type `P`.
fn unary_errors<P: Precision>(table: &mut Table, cross: &mut f64) {
    let one = P::ONE;
    let zero = P::ZERO;
    for p in unary_points() {
        if P::BITS == 24 && !p.both {
            continue;
        }
        let x = P::from_f64(p.x);
        let name = label(&p);
        let (n, q) = (p.name.as_str(), p.param);
        let f = |part, ty, e| (part, ty, e);
        let mut rows = Vec::new();
        rows.push(f("f", P::NAME, ulps(apply(n, q, x), p.f[0])));
        let d = apply(n, q, Dual::new(x, one));
        rows.push(f("f (Dual)", P::NAME, ulps(d.re, p.f[0])));
        rows.push(f("f′ (Dual)", P::NAME, ulps(d.eps, p.f[1])));
        let h = apply(n, q, HyperDual::new(x, one, one, zero));
        rows.push(f("f (HyperDual)", P::NAME, ulps(h.re, p.f[0])));
        rows.push(f("f′ (HyperDual ε₁)", P::NAME, ulps(h.eps1, p.f[1])));
        rows.push(f("f′ (HyperDual ε₂)", P::NAME, ulps(h.eps2, p.f[1])));
        rows.push(f("f″ (HyperDual)", P::NAME, ulps(h.eps1eps2, p.f[2])));
        // The nested types carry the lower derivatives as values of their own, and where one of
        // them overflows, its product with a zero tangent part is NaN (see `edges.rs`): those
        // points are not compared for them.
        let overflows = p.f[..3].iter().any(|v| v.abs() > P::MAX);
        let nested = |e: Option<f64>| if overflows { None } else { e };
        let dd = apply(n, q, Dual::new(Dual::new(x, one), Dual::new(one, zero)));
        rows.push(f(
            "f″ (Dual<Dual>)",
            P::NAME,
            nested(ulps(dd.eps.eps, p.f[2])),
        ));
        let dh = apply(
            n,
            q,
            Dual::new(
                HyperDual::new(x, one, one, zero),
                HyperDual::new(one, zero, zero, zero),
            ),
        );
        rows.push(f(
            "f‴ (Dual<HyperDual>)",
            P::NAME,
            nested(ulps(dh.eps.eps1eps2, p.f[3])),
        ));
        let hd = apply(
            n,
            q,
            HyperDual::new(
                Dual::new(x, one),
                Dual::new(one, zero),
                Dual::new(one, zero),
                Dual::new(zero, zero),
            ),
        );
        rows.push(f(
            "f‴ (HyperDual<Dual>)",
            P::NAME,
            nested(ulps(hd.eps1eps2.eps, p.f[3])),
        ));
        for (part, ty, e) in rows {
            table.record(&name, part, ty, p.x, e);
        }
        // The cross term of a general hyper-dual number: f′·d + f″·b·c, relative to the larger of
        // its two terms (which may cancel).
        let (b, c, dd) = (0.75, -1.25, 0.5);
        let g = apply(
            n,
            q,
            HyperDual::new(x, P::from_f64(b), P::from_f64(c), P::from_f64(dd)),
        );
        let want = p.f[1] * dd + p.f[2] * b * c;
        let size = (p.f[1] * dd).abs().max((p.f[2] * b * c).abs());
        if size.is_finite() && size > P::MIN && size < P::MAX {
            let got: f64 = g.eps1eps2.into();
            let rel = (got - want).abs() / size / 2f64.powi(1 - P::BITS);
            if rel > *cross {
                *cross = rel;
            }
        }
    }
}

/// Every partial derivative of the functions of two variables, in the type `P`.
fn binary_errors<P: Precision>(table: &mut Table) {
    let (one, zero) = (P::ONE, P::ZERO);
    for p in binary_points() {
        if P::BITS == 24 && !p.both {
            continue;
        }
        let (x, y) = (P::from_f64(p.x), P::from_f64(p.y));
        let n = p.name.as_str();
        let mut rows = Vec::new();
        rows.push(("f", ulps(apply2(n, x, y), p.f[0])));
        let dx = apply2(n, Dual::new(x, one), Dual::new(y, zero));
        let dy = apply2(n, Dual::new(x, zero), Dual::new(y, one));
        rows.push(("fx (Dual)", ulps(dx.eps, p.f[1])));
        rows.push(("fy (Dual)", ulps(dy.eps, p.f[2])));
        let h = |a: [P; 3], b: [P; 3]| {
            apply2(
                n,
                HyperDual::new(x, a[0], a[1], a[2]),
                HyperDual::new(y, b[0], b[1], b[2]),
            )
        };
        let xy = h([one, zero, zero], [zero, one, zero]);
        let xx = h([one, one, zero], [zero, zero, zero]);
        let yy = h([zero, zero, zero], [one, one, zero]);
        rows.push(("fx (HyperDual)", ulps(xy.eps1, p.f[1])));
        rows.push(("fy (HyperDual)", ulps(xy.eps2, p.f[2])));
        rows.push(("fxx (HyperDual)", ulps(xx.eps1eps2, p.f[3])));
        rows.push(("fxy (HyperDual)", ulps(xy.eps1eps2, p.f[4])));
        rows.push(("fyy (HyperDual)", ulps(yy.eps1eps2, p.f[5])));
        for (part, e) in rows {
            table.record(n, part, P::NAME, p.x * 1e3 + p.y, e);
        }
    }
}

fn all_errors() -> (Table, f64, f64) {
    let mut table = Table::default();
    let (mut cross64, mut cross32) = (0.0, 0.0);
    unary_errors::<f64>(&mut table, &mut cross64);
    unary_errors::<f32>(&mut table, &mut cross32);
    binary_errors::<f64>(&mut table);
    binary_errors::<f32>(&mut table);
    (table, cross64, cross32)
}

/// The bound each component is held to, in ulps: the value is the standard library's own; each
/// derivative takes a few more roundings, more for the deeper nestings; `powi` of a large power
/// multiplies them up as the power itself does; and a composition passes on the rounding of its
/// inner values amplified by its own condition (at 4, `sin(e^x)/(1 + x²)` has `e^x ≈ 55` inside
/// a sine).
fn bound(func: &str, part: &str) -> f64 {
    let base = if part.starts_with("f ") || part == "f" {
        4.0
    } else if part.starts_with("f′") || part.starts_with("fx") || part.starts_with("fy (") {
        8.0
    } else if part.starts_with("f″")
        || part.starts_with("fxx")
        || part.starts_with("fxy")
        || part.starts_with("fyy")
    {
        16.0
    } else {
        32.0
    };
    let large_power = func
        .strip_prefix("powi(")
        .and_then(|n| n.trim_end_matches(')').parse::<i32>().ok())
        .is_some_and(|n| n.unsigned_abs() > 5);
    let factor = if large_power || func.starts_with('g') {
        8.0
    } else {
        1.0
    };
    base * factor
}

#[test]
fn every_function_and_derivative_is_within_its_bound_of_mpmath() {
    let (table, cross64, cross32) = all_errors();
    check(&table, cross64, cross32);
}

/// Asserts every component within its bound, the number of comparisons, and the cross term.
fn check(table: &Table, cross64: f64, cross32: f64) {
    let mut failures = Vec::new();
    for ((func, part, ty), (e, x)) in &table.worst {
        if e.is_nan() || *e > bound(func, part) {
            failures.push(format!("{func} {part} {ty}: {e:.2} ulps at {x:e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        table.compared >= 8_700,
        "only {} comparisons",
        table.compared
    );
    // The cross term against the reference's own combination, in ulps of the larger term.
    assert!(cross64 <= 16.0, "cross term f64: {cross64}");
    assert!(cross32 <= 16.0, "cross term f32: {cross32}");
}

#[test]
fn largest_errors_seen() {
    let (table, cross64, cross32) = all_errors();
    let mut by_func: BTreeMap<(&str, &str), Vec<(f64, &str)>> = BTreeMap::new();
    for ((func, part, ty), (e, _)) in &table.worst {
        by_func
            .entry((func.as_str(), ty))
            .or_default()
            .push((*e, part));
    }
    println!("largest error seen, in ulps, by function, type and component");
    for ((func, ty), parts) in by_func {
        let line: Vec<String> = parts.iter().map(|(e, p)| format!("{p} {e:.2}")).collect();
        println!("{func:12} {ty}  {}", line.join(" | "));
    }
    println!(
        "cross term f′·d + f″·b·c: f64 {cross64:.2}, f32 {cross32:.2} ulps of the larger term"
    );
    println!("{} comparisons", table.compared);
    // The table is the evidence for the stated bounds, so it holds to them too.
    check(&table, cross64, cross32);
}
