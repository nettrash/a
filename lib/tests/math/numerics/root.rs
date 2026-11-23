//! The zeros of a function of one real variable: the five methods of `a::math::numerics::root`.
//!
//! The bracketing methods are held, step for step, to a model of them in Python and compared with
//! SciPy's `brentq` (`data/gen_roots.py`), on the model's own values where a function calls the
//! platform's libm (`data/roots_libm.txt`); every answer of the batteries is judged by the
//! definition of a root, a sign change of the same float function within the tolerance, found by
//! a scan of every float in it; and each defect of the audit has a test here that fails on the
//! code before it was fixed.

use a::math::numerics::{
    RootOptions, bisect, bisect_with, brent, brent_with, halley, halley_with, newton, newton_with,
    secant, secant_with,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

/// The Dottie number, the fixed point of the cosine and the root of cos x − x.
const DOTTIE: f64 = 0.739_085_133_215_160_7;

/// √2, the root of x² − 2 nearest 1.
const SQRT_2: f64 = core::f64::consts::SQRT_2;

/// The golden ratio, the root of x² − x − 1 above 1.
const PHI: f64 = 1.618_033_988_749_895;

/// The relative tolerance of the defaults, 4ε.
const RTOL: f64 = 4.0 * f64::EPSILON;

/// `x` to the power `n` by repeated products, as `data/gen_roots.py` forms it, so that the two
/// evaluate the battery's functions to the same bits.
fn pw(x: f64, n: u32) -> f64 {
    let mut r = 1.0;
    for _ in 0..n {
        r *= x;
    }
    r
}

/// The same for `f32`.
fn pw32(x: f32, n: u32) -> f32 {
    let mut r = 1.0;
    for _ in 0..n {
        r *= x;
    }
    r
}

/// `f` with a count of its evaluations.
fn counted<'a, T>(f: &'a dyn Fn(T) -> T, n: &'a Cell<usize>) -> impl Fn(T) -> T + 'a {
    move |x| {
        n.set(n.get() + 1);
        f(x)
    }
}

/// Whether the float function `f` changes sign, or is zero, within `tol` of `x`: by a scan of
/// every float in `[x − tol, x + tol]` (clamped to `[lo, hi]`) when there are at most 2¹⁶ of them,
/// and by its two ends and `x` otherwise. A scan, not two probes, because rounding can make a
/// function change sign more than once in so short an interval.
fn sign_change_near(f: &dyn Fn(f64) -> f64, x: f64, tol: f64, lo: f64, hi: f64) -> bool {
    let fx = f(x);
    if fx == 0.0 {
        return true;
    }
    let (from, to) = ((x - tol).max(lo), (x + tol).min(hi));
    let (mut t, mut n) = (from, 0);
    if (to - from)
        <= 65_536.0
            * (from.abs().max(to.abs()) * f64::EPSILON).max(f64::MIN_POSITIVE * f64::EPSILON)
    {
        while t <= to && n < 1 << 17 {
            let v = f(t);
            if v == 0.0 || (v < 0.0) != (fx < 0.0) {
                return true;
            }
            t = t.next_up();
            n += 1;
        }
        return false;
    }
    [from, to].iter().any(|&t| {
        let v = f(t);
        v == 0.0 || (v < 0.0) != (fx < 0.0)
    })
}

/// The same for `f32`.
fn sign_change_near32(f: &dyn Fn(f32) -> f32, x: f32, tol: f32, lo: f32, hi: f32) -> bool {
    let fx = f(x);
    if fx == 0.0 {
        return true;
    }
    let (from, to) = ((x - tol).max(lo), (x + tol).min(hi));
    let mut t = from;
    let mut n = 0;
    while t <= to && n < 1 << 17 {
        let v = f(t);
        if v == 0.0 || (v < 0.0) != (fx < 0.0) {
            return true;
        }
        t = t.next_up();
        n += 1;
    }
    false
}

/// The distance within which an open method's answer must have its sign change: the tolerance,
/// or one float where that is finer.
fn reach(opts: &RootOptions<f64>, x: f64) -> f64 {
    opts.tol_at(x)
        .max((x.abs() * f64::EPSILON).max(f64::MIN_POSITIVE * f64::EPSILON))
}

/// A small deterministic generator (xorshift64*), for the random batteries.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[a, b)`.
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.unit()
    }
}

// ---------------------------------------------------------------------------------------------
// The battery of brackets, against SciPy and against the model of the algorithm
// ---------------------------------------------------------------------------------------------

type Case = (&'static str, f64, f64, Box<dyn Fn(f64) -> f64>);

/// The brackets of `data/gen_roots.py`, in its order and written in its operations: the classic
/// problems of Alefeld, Potra and Shi, and hard cases — multiple roots, roots at zero, flat
/// functions, a huge dynamic range, the largest and the smallest floats, poles, jumps and kinks.
fn battery() -> Vec<Case> {
    let pi = core::f64::consts::PI;
    let mut v: Vec<Case> = Vec::new();
    let mut c =
        |name: &'static str, a: f64, b: f64, f: Box<dyn Fn(f64) -> f64>| v.push((name, a, b, f));
    c("sqrt2", 1.0, 2.0, Box::new(|x| x * x - 2.0));
    c("dottie", 0.0, 1.0, Box::new(|x: f64| x.cos() - x));
    c("cubic", 1.0, 2.0, Box::new(|x| x * x * x - x - 2.0));
    c("wallis", 2.0, 3.0, Box::new(|x| x * x * x - 2.0 * x - 5.0));
    c("exp2", 0.0, 1.0, Box::new(|x: f64| x.exp() - 2.0));
    c("aps01", pi / 2.0, pi, Box::new(|x: f64| x.sin() - x / 2.0));
    for (name, n) in [("aps02_1", 1.0), ("aps02_2", 2.0), ("aps02_5", 5.0)] {
        let (a, b) = (n * n + 1e-9, (n + 1.0) * (n + 1.0) - 1e-9);
        c(
            name,
            a,
            b,
            Box::new(|x| {
                let mut s = 0.0;
                for i in 1..=20 {
                    let i = f64::from(i);
                    let d = x - i * i;
                    s += (2.0 * i - 5.0) * (2.0 * i - 5.0) / (d * d * d);
                }
                -2.0 * s
            }),
        );
    }
    c(
        "aps03_a",
        -9.0,
        31.0,
        Box::new(|x: f64| -40.0 * x * (-x).exp()),
    );
    c(
        "aps03_c",
        -9.0,
        31.0,
        Box::new(|x: f64| -200.0 * x * (-3.0 * x).exp()),
    );
    c("aps04_4_0.2", 0.0, 5.0, Box::new(|x| pw(x, 4) - 0.2));
    c("aps04_12_1.0", 0.0, 5.0, Box::new(|x| pw(x, 12) - 1.0));
    c("aps05", 0.0, 1.5, Box::new(|x: f64| x.sin() - 0.5));
    for (name, n) in [
        ("aps06_1", 1.0f64),
        ("aps06_20", 20.0),
        ("aps06_100", 100.0),
    ] {
        c(
            name,
            0.0,
            1.0,
            Box::new(move |x: f64| 2.0 * x * (-n).exp() - 2.0 * (-n * x).exp() + 1.0),
        );
    }
    for (name, n) in [("aps07_5", 5.0), ("aps07_20", 20.0)] {
        c(
            name,
            0.0,
            1.0,
            Box::new(move |x: f64| {
                (1.0 + (1.0 - n) * (1.0 - n)) * x - (1.0 - n * x) * (1.0 - n * x)
            }),
        );
    }
    for (name, n) in [("aps08_2", 2), ("aps08_10", 10), ("aps08_20", 20)] {
        c(name, 0.0, 1.0, Box::new(move |x| x * x - pw(1.0 - x, n)));
    }
    for (name, n) in [("aps09_1", 1.0), ("aps09_5", 5.0), ("aps09_20", 20.0)] {
        c(
            name,
            0.0,
            1.0,
            Box::new(move |x| (1.0 + pw(1.0 - n, 4)) * x - pw(1.0 - n * x, 4)),
        );
    }
    for (name, n) in [("aps10_1", 1), ("aps10_15", 15)] {
        c(
            name,
            0.0,
            1.0,
            Box::new(move |x: f64| (-f64::from(n) * x).exp() * (x - 1.0) + pw(x, n)),
        );
    }
    for (name, n) in [("aps11_2", 2.0), ("aps11_20", 20.0)] {
        c(
            name,
            0.01,
            1.0,
            Box::new(move |x: f64| (n * x - 1.0) / ((n - 1.0) * x)),
        );
    }
    for (name, n) in [("aps12_2", 2.0f64), ("aps12_9", 9.0), ("aps12_33", 33.0)] {
        c(
            name,
            1.0,
            100.0,
            Box::new(move |x: f64| x.powf(1.0 / n) - n.powf(1.0 / n)),
        );
    }
    c(
        "aps13",
        -1.0,
        4.0,
        Box::new(|x: f64| {
            let t = x * x;
            if x == 0.0 { 0.0 } else { x * (-1.0 / t).exp() }
        }),
    );
    for (name, n) in [("aps14_1", 1.0), ("aps14_40", 40.0)] {
        c(
            name,
            -1e4,
            pi / 2.0,
            Box::new(move |x: f64| {
                if x >= 0.0 {
                    (n / 20.0) * (x / 1.5 + x.sin() - 1.0)
                } else {
                    -n / 20.0
                }
            }),
        );
    }
    for (name, n) in [("aps15_20", 20.0), ("aps15_1000", 1000.0)] {
        c(
            name,
            -1e4,
            1e-4,
            Box::new(move |x: f64| {
                if x > 2e-3 / (1.0 + n) {
                    core::f64::consts::E - 1.859
                } else if x >= 0.0 {
                    ((n + 1.0) * x / 2e-3).exp() - 1.859
                } else {
                    -0.859
                }
            }),
        );
    }
    c("mult3", 0.5, 3.0, Box::new(|x| pw(x - 1.0, 3)));
    c("mult5", 0.0, 3.0, Box::new(|x| pw(x - 1.0, 5)));
    c("mult9", 0.0, 3.5, Box::new(|x| pw(x - 1.0, 9)));
    c("zero_x", -1.0, 2.0, Box::new(|x| x));
    c("zero_x3", -1.0, 2.0, Box::new(|x| x * x * x));
    c("zero_x9", -1.0, 2.0, Box::new(|x| pw(x, 9)));
    c("zero_sin", -1.0, 2.0, Box::new(|x: f64| x.sin()));
    c("zero_cbrt", -1.0, 2.0, Box::new(|x: f64| x.cbrt()));
    c(
        "zero_xex",
        -9.0,
        31.0,
        Box::new(|x: f64| -40.0 * x * (-x).exp()),
    );
    c("flat_1e-6", 0.0, 1.0, Box::new(|x| (x - 0.3) * 1e-6));
    c("flat_1e-300", 0.0, 1.0, Box::new(|x| (x - 0.3) * 1e-300));
    c(
        "dyn_exp",
        -700.0,
        700.0,
        Box::new(|x: f64| x.exp() - (-600.0f64).exp()),
    );
    c(
        "dyn_exp50",
        -10.0,
        14.0,
        Box::new(|x: f64| (50.0 * x).exp() - 5.0),
    );
    c("big_lin", 0.0, f64::MAX, Box::new(|x| x - 1e308));
    c(
        "big_sym",
        -f64::MAX,
        f64::MAX,
        Box::new(|x| x / 1e308 - 1.0),
    );
    c("small_sq", 0.0, 1.0, Box::new(|x| x * x - 1e-300));
    c(
        "small_sub",
        0.0,
        1e-300,
        Box::new(|x: f64| x.sqrt() - 1e-155),
    );
    c("wide_sq", 0.0, 1e20, Box::new(|x| x * x - 2.0));
    c("wide_lin", 1e-6, 1e9, Box::new(|x| x - 1234.5));
    c("pole_tan", 1.0, 2.0, Box::new(|x: f64| x.tan()));
    c("pole_inv", 0.0, 1.0, Box::new(|x| 1.0 / (x - 0.3)));
    c("end_a", 1.0, 2.0, Box::new(|x| x - 1.0));
    c("near_end", 1.0, 2.0, Box::new(|x| x - (1.0 + f64::EPSILON)));
    c(
        "step",
        0.0,
        1.0,
        Box::new(|x| if x < 0.3 { -1.0 } else { 1.0 }),
    );
    c(
        "kink",
        0.0,
        1.0,
        Box::new(|x: f64| (x - 0.3).abs() + x - 0.5),
    );
    c("rev_sqrt2", 2.0, 1.0, Box::new(|x| x * x - 2.0));
    v
}

/// A row of the battery's reference table.
type Row = (
    &'static str,
    usize,
    f64,
    usize,
    f64,
    usize,
    f64,
    usize,
    f64,
    usize,
    f64,
);

/// Printed by `data/gen_roots.py`: for each bracket, SciPy 1.18's `brentq` at its own tolerances
/// (`xtol` 2·10⁻¹², `rtol` 4ε) and at the least `xtol` it accepts (5·10⁻³²⁴), as evaluations and
/// root; then the script's model of this crate's `brent` and `bisect` at their defaults, and of
/// `brent` at SciPy's tolerances. Zero evaluations and a NaN root mean that the method did not
/// converge in 10,000 iterations (or, for `brentq` on `big_sym`, refused the bracket).
const TABLE: [Row; 63] = [
    (
        "sqrt2",
        8,
        f64::from_bits(0x3ff6a09e667f3c87),
        9,
        f64::from_bits(0x3ff6a09e667f3bcc),
        9,
        f64::from_bits(0x3ff6a09e667f3bcc),
        52,
        f64::from_bits(0x3ff6a09e667f3bcc),
        8,
        f64::from_bits(0x3ff6a09e667f3c87),
    ),
    (
        "dottie",
        8,
        f64::from_bits(0x3fe7a695dd83ce03),
        9,
        f64::from_bits(0x3fe7a695dd83ce2d),
        10,
        f64::from_bits(0x3fe7a695dd83ce2d),
        53,
        f64::from_bits(0x3fe7a695dd83ce2c),
        8,
        f64::from_bits(0x3fe7a695dd83ce03),
    ),
    (
        "cubic",
        9,
        f64::from_bits(0x3ff857923f586105),
        9,
        f64::from_bits(0x3ff857923f586105),
        9,
        f64::from_bits(0x3ff857923f586105),
        52,
        f64::from_bits(0x3ff857923f586104),
        9,
        f64::from_bits(0x3ff857923f586105),
    ),
    (
        "wallis",
        8,
        f64::from_bits(0x4000c1a4350819e4),
        8,
        f64::from_bits(0x4000c1a4350819e4),
        8,
        f64::from_bits(0x4000c1a4350819e4),
        51,
        f64::from_bits(0x4000c1a4350819e4),
        8,
        f64::from_bits(0x4000c1a4350819e4),
    ),
    (
        "exp2",
        8,
        f64::from_bits(0x3fe62e42fefa39ef),
        8,
        f64::from_bits(0x3fe62e42fefa39ef),
        8,
        f64::from_bits(0x3fe62e42fefa39ef),
        51,
        f64::from_bits(0x3fe62e42fefa39f0),
        8,
        f64::from_bits(0x3fe62e42fefa39ef),
    ),
    (
        "aps01",
        9,
        f64::from_bits(0x3ffe53f1cbeaa94a),
        9,
        f64::from_bits(0x3ffe53f1cbeaa9bc),
        9,
        f64::from_bits(0x3ffe53f1cbeaa9bc),
        52,
        f64::from_bits(0x3ffe53f1cbeaa9be),
        9,
        f64::from_bits(0x3ffe53f1cbeaa94a),
    ),
    (
        "aps02_1",
        15,
        f64::from_bits(0x40082eee3dd8e9fe),
        16,
        f64::from_bits(0x40082eee3dd8ea02),
        17,
        f64::from_bits(0x40082eee3dd8ea02),
        52,
        f64::from_bits(0x40082eee3dd8ea06),
        15,
        f64::from_bits(0x40082eee3dd8e9fe),
    ),
    (
        "aps02_2",
        9,
        f64::from_bits(0x401abc29e4b8c6e8),
        10,
        f64::from_bits(0x401abc29e4b8c6e8),
        11,
        f64::from_bits(0x401abc29e4b8c6e8),
        52,
        f64::from_bits(0x401abc29e4b8c6eb),
        9,
        f64::from_bits(0x401abc29e4b8c6e7),
    ),
    (
        "aps02_5",
        13,
        f64::from_bits(0x403dd406b4c0fdcf),
        13,
        f64::from_bits(0x403dd406b4c0fdcf),
        15,
        f64::from_bits(0x403dd406b4c0fdcf),
        51,
        f64::from_bits(0x403dd406b4c0fdca),
        13,
        f64::from_bits(0x403dd406b4c0fdcf),
    ),
    (
        "aps03_a",
        17,
        f64::from_bits(0xbc4a65a8e0000000),
        20,
        f64::from_bits(0x0000000000000000),
        7,
        f64::from_bits(0x0000000000000000),
        71,
        f64::from_bits(0x8000000000000000),
        17,
        f64::from_bits(0xbc4a65a8a0000000),
    ),
    (
        "aps03_c",
        20,
        f64::from_bits(0xbc3aba03c0000000),
        21,
        f64::from_bits(0x0000000000000000),
        7,
        f64::from_bits(0x0000000000000000),
        71,
        f64::from_bits(0x8000000000000000),
        20,
        f64::from_bits(0xbc3aba0380000000),
    ),
    (
        "aps04_4_0.2",
        15,
        f64::from_bits(0x3fe56652116c816c),
        15,
        f64::from_bits(0x3fe56652116c816c),
        17,
        f64::from_bits(0x3fe56652116c816b),
        57,
        f64::from_bits(0x3fe56652116c816f),
        15,
        f64::from_bits(0x3fe56652116c816c),
    ),
    (
        "aps04_12_1.0",
        17,
        f64::from_bits(0x3fefffffffffffc2),
        17,
        f64::from_bits(0x3ff0000000000000),
        18,
        f64::from_bits(0x3ff0000000000000),
        57,
        f64::from_bits(0x3ff0000000000002),
        17,
        f64::from_bits(0x3fefffffffffffc2),
    ),
    (
        "aps05",
        9,
        f64::from_bits(0x3fe0c152382d7365),
        10,
        f64::from_bits(0x3fe0c152382d7365),
        10,
        f64::from_bits(0x3fe0c152382d7365),
        54,
        f64::from_bits(0x3fe0c152382d7366),
        9,
        f64::from_bits(0x3fe0c152382d7365),
    ),
    (
        "aps06_1",
        8,
        f64::from_bits(0x3fdb09dff28cad95),
        8,
        f64::from_bits(0x3fdb09dff28cad99),
        8,
        f64::from_bits(0x3fdb09dff28cad99),
        54,
        f64::from_bits(0x3fdb09dff28cad98),
        8,
        f64::from_bits(0x3fdb09dff28cad95),
    ),
    (
        "aps06_20",
        13,
        f64::from_bits(0x3fa1be9bff1edf5e),
        13,
        f64::from_bits(0x3fa1be9bff1edf5c),
        14,
        f64::from_bits(0x3fa1be9bff1edf5d),
        61,
        f64::from_bits(0x3fa1be9bff1edf5c),
        13,
        f64::from_bits(0x3fa1be9bff1edf5e),
    ),
    (
        "aps06_100",
        14,
        f64::from_bits(0x3f7c642ccb776bf1),
        15,
        f64::from_bits(0x3f7c642ccb7dbacb),
        18,
        f64::from_bits(0x3f7c642ccb7dbacc),
        67,
        f64::from_bits(0x3f7c642ccb7dbacc),
        14,
        f64::from_bits(0x3f7c642ccb776bf1),
    ),
    (
        "aps07_5",
        9,
        f64::from_bits(0x3fa3a97fd07a164d),
        10,
        f64::from_bits(0x3fa3a97fd07a119e),
        11,
        f64::from_bits(0x3fa3a97fd07a119d),
        61,
        f64::from_bits(0x3fa3a97fd07a119c),
        9,
        f64::from_bits(0x3fa3a97fd07a164d),
    ),
    (
        "aps07_20",
        8,
        f64::from_bits(0x3f646dc5db96e9d0),
        9,
        f64::from_bits(0x3f646dc5db96e5ac),
        10,
        f64::from_bits(0x3f646dc5db96e5ac),
        69,
        f64::from_bits(0x3f646dc5db96e5ae),
        8,
        f64::from_bits(0x3f646dc5db96e9d0),
    ),
    (
        "aps08_2",
        3,
        f64::from_bits(0x3fe0000000000000),
        3,
        f64::from_bits(0x3fe0000000000000),
        3,
        f64::from_bits(0x3fe0000000000000),
        3,
        f64::from_bits(0x3fe0000000000000),
        3,
        f64::from_bits(0x3fe0000000000000),
    ),
    (
        "aps08_10",
        10,
        f64::from_bits(0x3fcf602b2b7eab97),
        11,
        f64::from_bits(0x3fcf602b2b7eace6),
        13,
        f64::from_bits(0x3fcf602b2b7eace7),
        57,
        f64::from_bits(0x3fcf602b2b7eace4),
        10,
        f64::from_bits(0x3fcf602b2b7eab97),
    ),
    (
        "aps08_20",
        13,
        f64::from_bits(0x3fc51c2142f6b2a8),
        13,
        f64::from_bits(0x3fc51c2142f6b2a8),
        16,
        f64::from_bits(0x3fc51c2142f6b2a6),
        57,
        f64::from_bits(0x3fc51c2142f6b2a4),
        13,
        f64::from_bits(0x3fc51c2142f6b2a8),
    ),
    (
        "aps09_1",
        10,
        f64::from_bits(0x3fd1a1ec7a782e15),
        10,
        f64::from_bits(0x3fd1a1ec7a782e15),
        12,
        f64::from_bits(0x3fd1a1ec7a782e15),
        55,
        f64::from_bits(0x3fd1a1ec7a782e14),
        10,
        f64::from_bits(0x3fd1a1ec7a782e15),
    ),
    (
        "aps09_5",
        7,
        f64::from_bits(0x3f6da1a02ab41e3a),
        8,
        f64::from_bits(0x3f6da1a02ab41e3a),
        11,
        f64::from_bits(0x3f6da1a02ab41e3c),
        68,
        f64::from_bits(0x3f6da1a02ab41e3b),
        7,
        f64::from_bits(0x3f6da1a02ab41e3a),
    ),
    (
        "aps09_20",
        6,
        f64::from_bits(0x3ee0150bb4772e7e),
        7,
        f64::from_bits(0x3ee0150bb02d754c),
        9,
        f64::from_bits(0x3ee0150bb02d754b),
        70,
        f64::from_bits(0x3ee0150bb02d754c),
        6,
        f64::from_bits(0x3ee0150bb4772e7e),
    ),
    (
        "aps10_1",
        8,
        f64::from_bits(0x3fd9aaefc0224672),
        9,
        f64::from_bits(0x3fd9aaefc0224781),
        9,
        f64::from_bits(0x3fd9aaefc0224781),
        55,
        f64::from_bits(0x3fd9aaefc0224784),
        8,
        f64::from_bits(0x3fd9aaefc0224672),
    ),
    (
        "aps10_15",
        12,
        f64::from_bits(0x3fe18ab5984e0f3d),
        12,
        f64::from_bits(0x3fe18ab5984e0f3d),
        13,
        f64::from_bits(0x3fe18ab5984e0f3d),
        53,
        f64::from_bits(0x3fe18ab5984e0f3c),
        12,
        f64::from_bits(0x3fe18ab5984e0f3d),
    ),
    (
        "aps11_2",
        4,
        f64::from_bits(0x3fe0000000000000),
        4,
        f64::from_bits(0x3fe0000000000000),
        4,
        f64::from_bits(0x3fe0000000000000),
        54,
        f64::from_bits(0x3fe0000000000000),
        4,
        f64::from_bits(0x3fe0000000000000),
    ),
    (
        "aps11_20",
        14,
        f64::from_bits(0x3fa9999999999968),
        14,
        f64::from_bits(0x3fa999999999999a),
        13,
        f64::from_bits(0x3fa9999999999999),
        55,
        f64::from_bits(0x3fa9999999999998),
        14,
        f64::from_bits(0x3fa9999999999968),
    ),
    (
        "aps12_2",
        6,
        f64::from_bits(0x4000000000000000),
        6,
        f64::from_bits(0x4000000000000000),
        6,
        f64::from_bits(0x4000000000000000),
        56,
        f64::from_bits(0x4000000000000001),
        6,
        f64::from_bits(0x4000000000000000),
    ),
    (
        "aps12_9",
        11,
        f64::from_bits(0x4022000000000004),
        11,
        f64::from_bits(0x4022000000000004),
        12,
        f64::from_bits(0x4022000000000005),
        52,
        f64::from_bits(0x4021fffffffffffe),
        12,
        f64::from_bits(0x4021fffffffffffd),
    ),
    (
        "aps12_33",
        10,
        f64::from_bits(0x40407ffffffffff8),
        10,
        f64::from_bits(0x404080000000000d),
        10,
        f64::from_bits(0x404080000000000c),
        51,
        f64::from_bits(0x40407ffffffffffb),
        10,
        f64::from_bits(0x40407ffffffffff7),
    ),
    (
        "aps13",
        18,
        f64::from_bits(0x3f8f9534c4d725f0),
        18,
        f64::from_bits(0x3f8f1025741b02b8),
        9,
        f64::from_bits(0x0010a74498a55c53),
        4,
        f64::from_bits(0x0004000000000000),
        18,
        f64::from_bits(0x3f8f953553cdcd78),
    ),
    (
        "aps14_1",
        21,
        f64::from_bits(0x3fe3f63916bf1867),
        21,
        f64::from_bits(0x3fe3f63916bf1867),
        11,
        f64::from_bits(0x3fe3f63916bf1867),
        57,
        f64::from_bits(0x3fe3f63916bf1868),
        21,
        f64::from_bits(0x3fe3f63916bf1867),
    ),
    (
        "aps14_40",
        21,
        f64::from_bits(0x3fe3f63916bf1867),
        21,
        f64::from_bits(0x3fe3f63916bf1867),
        11,
        f64::from_bits(0x3fe3f63916bf1867),
        57,
        f64::from_bits(0x3fe3f63916bf1868),
        21,
        f64::from_bits(0x3fe3f63916bf1867),
    ),
    (
        "aps15_20",
        34,
        f64::from_bits(0x3f0ef5bb6916bf00),
        36,
        f64::from_bits(0x3f0ef5bb6916ef40),
        12,
        f64::from_bits(0x3f0ef5bb6916ef40),
        54,
        f64::from_bits(0x3f0ef5bb6916ef41),
        34,
        f64::from_bits(0x3f0ef5bb6916bf00),
    ),
    (
        "aps15_1000",
        40,
        f64::from_bits(0x3eb4c8c56c4c2ade),
        42,
        f64::from_bits(0x3eb4c8c56c24e112),
        19,
        f64::from_bits(0x3eb4c8c56c24e111),
        65,
        f64::from_bits(0x3eb4c8c56c24e111),
        40,
        f64::from_bits(0x3eb4c8c56c4c2ade),
    ),
    (
        "mult3",
        121,
        f64::from_bits(0x3feffffffffffb0e),
        154,
        f64::from_bits(0x3fefffffffffffff),
        113,
        f64::from_bits(0x3ff0000000000001),
        54,
        f64::from_bits(0x3ff0000000000002),
        121,
        f64::from_bits(0x3feffffffffffb0e),
    ),
    (
        "mult5",
        108,
        f64::from_bits(0x3feffffffffffa0b),
        134,
        f64::from_bits(0x3ff0000000000000),
        128,
        f64::from_bits(0x3ff0000000000000),
        55,
        f64::from_bits(0x3feffffffffffffe),
        108,
        f64::from_bits(0x3feffffffffffa0b),
    ),
    (
        "mult9",
        116,
        f64::from_bits(0x3feffffffffffdf9),
        140,
        f64::from_bits(0x3ff0000000000001),
        139,
        f64::from_bits(0x3ff0000000000000),
        55,
        f64::from_bits(0x3ff0000000000002),
        116,
        f64::from_bits(0x3ff0000000000531),
    ),
    (
        "zero_x",
        3,
        f64::from_bits(0x0000000000000000),
        3,
        f64::from_bits(0x0000000000000000),
        3,
        f64::from_bits(0x0000000000000000),
        71,
        f64::from_bits(0x8000000000000000),
        3,
        f64::from_bits(0x0000000000000000),
    ),
    (
        "zero_x3",
        123,
        f64::from_bits(0xbd41928887234de8),
        830,
        f64::from_bits(0xa9576b7d0ae66400),
        9,
        f64::from_bits(0x000f415eaea18a3a),
        4,
        f64::from_bits(0x8008000000000000),
        123,
        f64::from_bits(0xbd41928887234df8),
    ),
    (
        "zero_x9",
        114,
        f64::from_bits(0xbd33fd685cae9fa0),
        256,
        f64::from_bits(0xb852124c53756b40),
        8,
        f64::from_bits(0x8006a83f4fa324a0),
        4,
        f64::from_bits(0x8008000000000000),
        114,
        f64::from_bits(0xbd492a232324c848),
    ),
    (
        "zero_sin",
        8,
        f64::from_bits(0x3d40a9de71600000),
        9,
        f64::from_bits(0x0000000000000000),
        10,
        f64::from_bits(0x0000000000000000),
        71,
        f64::from_bits(0x8000000000000000),
        8,
        f64::from_bits(0x3d40a9de71600000),
    ),
    (
        "zero_cbrt",
        33,
        f64::from_bits(0xbd53e0d7da269688),
        1329,
        f64::from_bits(0x0000000000000000),
        62,
        f64::from_bits(0x0000000000000000),
        71,
        f64::from_bits(0x8000000000000000),
        33,
        f64::from_bits(0xbd53e0d7dd701cfc),
    ),
    (
        "zero_xex",
        17,
        f64::from_bits(0xbc4a65a8e0000000),
        20,
        f64::from_bits(0x0000000000000000),
        7,
        f64::from_bits(0x0000000000000000),
        71,
        f64::from_bits(0x8000000000000000),
        17,
        f64::from_bits(0xbc4a65a8a0000000),
    ),
    (
        "flat_1e-6",
        3,
        f64::from_bits(0x3fd3333333333333),
        3,
        f64::from_bits(0x3fd3333333333333),
        3,
        f64::from_bits(0x3fd3333333333333),
        55,
        f64::from_bits(0x3fd3333333333334),
        3,
        f64::from_bits(0x3fd3333333333333),
    ),
    (
        "flat_1e-300",
        3,
        f64::from_bits(0x3fd3333333333333),
        3,
        f64::from_bits(0x3fd3333333333333),
        3,
        f64::from_bits(0x3fd3333333333333),
        55,
        f64::from_bits(0x3fd3333333333334),
        3,
        f64::from_bits(0x3fd3333333333333),
    ),
    (
        "dyn_exp",
        34,
        f64::from_bits(0xc082c00000000000),
        34,
        f64::from_bits(0xc082c00000000000),
        28,
        f64::from_bits(0xc082c00000000000),
        55,
        f64::from_bits(0xc082c00000000001),
        26,
        f64::from_bits(0xc082c00000000000),
    ),
    (
        "dyn_exp50",
        18,
        f64::from_bits(0x3fa07b0b7ff4fe08),
        19,
        f64::from_bits(0x3fa07b0b7ff43169),
        22,
        f64::from_bits(0x3fa07b0b7ff43169),
        65,
        f64::from_bits(0x3fa07b0b7ff4316a),
        18,
        f64::from_bits(0x3fa07b0b7ff4fe08),
    ),
    (
        "big_lin",
        53,
        f64::from_bits(0x7fe1ccf385ebc89f),
        53,
        f64::from_bits(0x7fe1ccf385ebc89f),
        4,
        f64::from_bits(0x7fe1ccf385ebc8a0),
        53,
        f64::from_bits(0x7fe1ccf385ebc8a3),
        4,
        f64::from_bits(0x7fe1ccf385ebc8a0),
    ),
    (
        "big_sym",
        0,
        f64::NAN,
        0,
        f64::NAN,
        6,
        f64::from_bits(0x7fe1ccf385ebc8a0),
        55,
        f64::from_bits(0x7fe1ccf385ebc8a3),
        6,
        f64::from_bits(0x7fe1ccf385ebc8a0),
    ),
    (
        "small_sq",
        3,
        f64::from_bits(0x0000000000000000),
        1098,
        f64::from_bits(0x20ca2fe76a3f9476),
        12,
        f64::from_bits(0x20ca2fe76a3f9475),
        67,
        f64::from_bits(0x20ca2fe76a3f9472),
        3,
        f64::from_bits(0x0000000000000000),
    ),
    (
        "small_sub",
        2,
        f64::from_bits(0x0000000000000000),
        0,
        f64::NAN,
        10,
        f64::from_bits(0x000012688b70e62b),
        62,
        f64::from_bits(0x000012688b70e62b),
        2,
        f64::from_bits(0x0000000000000000),
    ),
    (
        "wide_sq",
        126,
        f64::from_bits(0x3ff6a09e667f43b7),
        127,
        f64::from_bits(0x3ff6a09e667f3bcd),
        13,
        f64::from_bits(0x3ff6a09e667f3bcd),
        70,
        f64::from_bits(0x3ff6a09e667f3bca),
        33,
        f64::from_bits(0x3ff6a09e667f39c3),
    ),
    (
        "wide_lin",
        3,
        f64::from_bits(0x40934a0000000000),
        3,
        f64::from_bits(0x40934a0000000000),
        3,
        f64::from_bits(0x40934a0000000000),
        62,
        f64::from_bits(0x40934a0000000002),
        3,
        f64::from_bits(0x40934a0000000000),
    ),
    (
        "pole_tan",
        53,
        f64::from_bits(0x3ff921fb5444160a),
        67,
        f64::from_bits(0x3ff921fb54442d1c),
        67,
        f64::from_bits(0x3ff921fb54442d1c),
        52,
        f64::from_bits(0x3ff921fb54442d1c),
        53,
        f64::from_bits(0x3ff921fb5444160a),
    ),
    (
        "pole_inv",
        44,
        f64::from_bits(0x3fd3333333339998),
        57,
        f64::from_bits(0x3fd3333333333335),
        58,
        f64::from_bits(0x3fd3333333333335),
        55,
        f64::from_bits(0x3fd3333333333334),
        44,
        f64::from_bits(0x3fd3333333339998),
    ),
    (
        "end_a",
        2,
        f64::from_bits(0x3ff0000000000000),
        2,
        f64::from_bits(0x3ff0000000000000),
        1,
        f64::from_bits(0x3ff0000000000000),
        1,
        f64::from_bits(0x3ff0000000000000),
        1,
        f64::from_bits(0x3ff0000000000000),
    ),
    (
        "near_end",
        3,
        f64::from_bits(0x3ff0000000000000),
        3,
        f64::from_bits(0x3ff0000000000002),
        3,
        f64::from_bits(0x3ff0000000000002),
        52,
        f64::from_bits(0x3ff0000000000004),
        3,
        f64::from_bits(0x3ff0000000000000),
    ),
    (
        "step",
        41,
        f64::from_bits(0x3fd3333333338000),
        54,
        f64::from_bits(0x3fd3333333333334),
        55,
        f64::from_bits(0x3fd3333333333334),
        55,
        f64::from_bits(0x3fd3333333333334),
        41,
        f64::from_bits(0x3fd3333333338000),
    ),
    (
        "kink",
        7,
        f64::from_bits(0x3fd999999999999a),
        7,
        f64::from_bits(0x3fd999999999999a),
        7,
        f64::from_bits(0x3fd999999999999a),
        55,
        f64::from_bits(0x3fd999999999999c),
        7,
        f64::from_bits(0x3fd999999999999a),
    ),
    (
        "rev_sqrt2",
        8,
        f64::from_bits(0x3ff6a09e667f3c87),
        9,
        f64::from_bits(0x3ff6a09e667f3bcc),
        9,
        f64::from_bits(0x3ff6a09e667f3bcc),
        52,
        f64::from_bits(0x3ff6a09e667f3bcc),
        8,
        f64::from_bits(0x3ff6a09e667f3c87),
    ),
];

/// The brackets of the battery whose functions call the platform's libm (cos, sin, tan, exp, pow,
/// cbrt), as `LIBM` of `data/gen_roots.py` lists them. The libms of different platforms differ in
/// the last bit (Apple's and glibc's do), and a last bit can decide the last steps of a method. The
/// other functions are IEEE arithmetic and square roots, rounded alike on every platform.
const LIBM: [&str; 25] = [
    "dottie",
    "exp2",
    "aps01",
    "aps03_a",
    "aps03_c",
    "aps05",
    "aps06_1",
    "aps06_20",
    "aps06_100",
    "aps10_1",
    "aps10_15",
    "aps12_2",
    "aps12_9",
    "aps12_33",
    "aps13",
    "aps14_1",
    "aps14_40",
    "aps15_20",
    "aps15_1000",
    "zero_sin",
    "zero_cbrt",
    "zero_xex",
    "dyn_exp",
    "dyn_exp50",
    "pole_tan",
];

/// A run of a method on a bracket: its answer and the points it evaluated `f` at, in order.
type Trace = (Option<f64>, Vec<u64>);

/// The model's runs on the brackets of [`LIBM`], from `data/gen_roots.py` on aarch64 macOS
/// (`data/roots_libm.txt`): for each bracket and each run — `brent` at the defaults, `xtol` for
/// `brent` at SciPy's `xtol` of 2·10⁻¹², `bisect` at the defaults — every point evaluated, in
/// order, with the value the model got there from Apple's libm. Replayed, they let the steps be
/// compared on the values the model saw, whatever libm the platform has.
struct Replay {
    /// The points of each run, by bracket and run.
    points: HashMap<(&'static str, &'static str), Vec<u64>>,
    /// The value at each point of any run, by bracket.
    values: HashMap<&'static str, HashMap<u64, u64>>,
}

impl Replay {
    /// Reads `data/roots_libm.txt`.
    fn load() -> Self {
        let mut points: HashMap<_, Vec<u64>> = HashMap::new();
        let mut values: HashMap<_, HashMap<u64, u64>> = HashMap::new();
        for line in include_str!("data/roots_libm.txt").lines() {
            if line.starts_with('#') {
                continue;
            }
            let w: Vec<&'static str> = line.split_whitespace().collect();
            let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
            let (x, v) = (hex(w[2]), hex(w[3]));
            points.entry((w[0], w[1])).or_default().push(x);
            values.entry(w[0]).or_default().insert(x, v);
        }
        Replay { points, values }
    }

    /// Runs `method` on the bracket `name` of the function `f`, with the model's value at each
    /// point the model evaluated and `f`'s own elsewhere (where the run has already left the
    /// model's steps). On a bracket outside [`LIBM`] that is `f` itself.
    fn run(
        &self,
        name: &str,
        f: &dyn Fn(f64) -> f64,
        method: impl FnOnce(&dyn Fn(f64) -> f64) -> Option<f64>,
    ) -> Trace {
        let seen = RefCell::new(Vec::new());
        let values = self.values.get(name);
        let g = |x: f64| {
            seen.borrow_mut().push(x.to_bits());
            values
                .and_then(|v| v.get(&x.to_bits()))
                .map_or_else(|| f(x), |&v| f64::from_bits(v))
        };
        let r = method(&g);
        (r, seen.into_inner())
    }

    /// Asserts that a replayed run took the model's steps: as many evaluations, the same answer to
    /// the bit, and on a bracket of [`LIBM`] the same points in the same order.
    fn check(&self, name: &str, run: &str, trace: &Trace, evals: usize, root: f64) {
        let (r, points) = trace;
        assert_eq!(points.len(), evals, "{name}: {r:?}");
        assert_eq!(r.map(f64::to_bits), Some(root.to_bits()), "{name}");
        if let Some(model) = self.points.get(&(name, run)) {
            assert!(points == model, "{name}: not the points of the model");
        }
    }
}

#[test]
fn the_replayed_values_are_the_libms_own_to_its_rounding() {
    // The data cover every run of the model on the brackets that call the libm, and no other
    // bracket. Its values are the platform's own up to the rounding of the libm: a relative 10⁻¹²
    // of the larger of the value and 1 is far above that rounding (each libm value is within an ulp
    // or two, and where a function cancels, near its root, the terms that cancel are of size 5 or
    // less) and far below a slip in writing a function in Rust. On aarch64 macOS, where the data
    // were made with the same libm, they are the same bits: that holds the Rust functions to the
    // script's operation for operation, which the replay would otherwise hide.
    let replay = Replay::load();
    let mut names: Vec<_> = replay.values.keys().copied().collect();
    names.sort_unstable();
    let mut libm = LIBM.to_vec();
    libm.sort_unstable();
    assert_eq!(names, libm);
    for &name in &LIBM {
        for run in ["brent", "xtol", "bisect"] {
            assert!(replay.points.contains_key(&(name, run)), "{name} {run}");
        }
    }
    let (mut points, mut differ) = (0, 0);
    for (name, _, _, f) in battery() {
        let Some(values) = replay.values.get(name) else {
            continue;
        };
        for (&x, &v) in values {
            let (x, want, got) = (f64::from_bits(x), f64::from_bits(v), f(f64::from_bits(x)));
            assert!(
                got == want || (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{name}: f({x:e}) = {got:e}, the model's {want:e}"
            );
            points += 1;
            differ += usize::from(got.to_bits() != want.to_bits());
        }
    }
    // The 1,995 distinct points of the model's 2,265 evaluations.
    assert_eq!(points, 1995);
    // (Not under Miri, whose libm functions are its own.)
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) && !cfg!(miri) {
        assert_eq!(differ, 0);
    }
}

#[test]
fn brent_takes_the_steps_of_its_model_at_the_default_tolerances() {
    // The model in Python is Brent's ALGOL procedure written out, with the two safeguards of the
    // bisection for a tolerance Brent's specification leaves open; the Rust code must take the same
    // steps, which pins every decision of the algorithm, and not only the roots it reaches. Where
    // the function calls the libm the steps are compared on the model's values (`Replay`): with
    // glibc's, cos x − x takes 8 evaluations rather than 10, to an answer an ulp away, since the
    // last bits of the cosine decide the last steps.
    let replay = Replay::load();
    for ((name, a, b, f), row) in battery().into_iter().zip(TABLE) {
        assert_eq!(name, row.0);
        let trace = replay.run(name, &*f, |g| brent(a, b, g));
        replay.check(name, "brent", &trace, row.5, row.6);
    }
}

#[test]
fn brent_at_scipys_tolerances_takes_the_steps_of_brentq() {
    // At xtol = 2e-12 the method is Brent's unchanged, the algorithm of brentq, and takes its
    // steps: the same number of evaluations on 57 of the 63 brackets. The others are told apart
    // and stated: on five of them this code needs fewer, since its interpolation (Brent's own
    // formulas, products of ratios) does not overflow where brentq's divided differences do
    // (dyn_exp, big_lin; brentq refuses big_sym outright), since a bracket as wide as [0, 1e20]
    // gets the safeguarded bisection of a tolerance that halving cannot reach (wide_sq: 33 against
    // 126), or since a root at the first end is returned before the second is evaluated (end_a);
    // and on aps12_9 rounding in the two forms of the interpolation decides one step differently
    // (12 against 11). brentq ran on aarch64 macOS, with Apple's libm, and the model's values are
    // replayed where the function calls it (`Replay`), so that both are compared on the same values
    // on every platform: with glibc's own, the answer for the cube root, −2.8·10⁻¹³ (its root is
    // zero, and the xtol 2·10⁻¹²), moves by 396 ulps in the same 33 evaluations.
    let differ = [
        "aps12_9", "dyn_exp", "big_lin", "big_sym", "wide_sq", "end_a",
    ];
    let opts = RootOptions::new().with_xtol(2e-12);
    let replay = Replay::load();
    let mut same = 0;
    for ((name, a, b, f), row) in battery().into_iter().zip(TABLE) {
        let trace = replay.run(name, &*f, |g| brent_with(a, b, g, &opts));
        // The model at the same tolerances, exactly.
        replay.check(name, "xtol", &trace, row.9, row.10);
        let (r, n) = (trace.0.unwrap(), trace.1.len());
        let (scipy_evals, scipy_root) = (row.1, row.2);
        if differ.contains(&name) {
            assert_ne!(n, scipy_evals, "{name} was listed as different");
            if scipy_evals > 0 {
                assert!(n < scipy_evals || name == "aps12_9", "{name}");
            }
        } else {
            assert_eq!(n, scipy_evals, "{name}");
            same += 1;
        }
        // Both answers are within the tolerance of a sign change, so within twice it of each other,
        // unless the float function is zero on a whole interval (aps13, x·e^(−1/x²), underflows to
        // zero for |x| < 0.037) and both are exact zeros of it.
        if scipy_evals > 0 && !(f(r) == 0.0 && f(scipy_root) == 0.0) {
            let tol = 2e-12 + RTOL * r.abs();
            assert!(
                (r - scipy_root).abs() <= 2.0 * tol,
                "{name}: {r} against {scipy_root}"
            );
        }
    }
    assert_eq!(same, 57);
}

#[test]
fn brent_converges_where_brentq_cannot_reach_a_relative_tolerance() {
    // At the least absolute tolerance SciPy accepts, brentq needs 830 evaluations for x³ at zero,
    // 1,329 for its cube root and 1,098 for the root 1e-150 of x² − 1e-300, and does not converge
    // in 10,000 on √x − 1e-155; the defaults here take 9, 62, 12 and 10, by the safeguards of the
    // bisection. On brackets where neither safeguard acts they take the steps brentq takes. The
    // counts are those of the model's values (`Replay`; of the cases here only the cube root calls
    // the libm); the answers are checked on the platform's own function too.
    let cases: HashMap<_, _> = battery().into_iter().map(|c| (c.0, c)).collect();
    let table: HashMap<_, _> = TABLE.iter().map(|r| (r.0, *r)).collect();
    let replay = Replay::load();
    for (name, ours, theirs) in [
        ("zero_x3", 9, 830),
        ("zero_cbrt", 62, 1329),
        ("small_sq", 12, 1098),
        ("small_sub", 10, 0),
        ("wide_sq", 13, 127),
        ("mult3", 113, 154),
    ] {
        let (_, a, b, f) = &cases[name];
        let (lo, hi) = (a.min(*b), a.max(*b));
        let (r, points) = replay.run(name, &**f, |g| brent(*a, *b, g));
        assert_eq!((points.len(), table[name].3), (ours, theirs), "{name}");
        for r in [r.unwrap(), brent(*a, *b, &**f).unwrap()] {
            let tol = (RTOL * r.abs()).max(f64::MIN_POSITIVE * f64::EPSILON);
            assert!(sign_change_near(&**f, r, tol, lo, hi), "{name}: {r}");
        }
    }
    // Over the brackets both finish: 1,176 evaluations here against 4,899 for brentq (with glibc's
    // libm, 1,174: the two of cos x − x).
    let mut total = (0, 0);
    for ((name, a, b, f), row) in battery().into_iter().zip(TABLE) {
        if row.3 > 0 {
            let (r, points) = replay.run(name, &*f, |g| brent(a, b, g));
            r.unwrap_or_else(|| panic!("{name}"));
            total.0 += points.len();
            total.1 += row.3;
        }
    }
    assert_eq!(total, (1176, 4899));
}

#[test]
fn bisect_takes_the_steps_of_its_model() {
    let replay = Replay::load();
    for ((name, a, b, f), row) in battery().into_iter().zip(TABLE) {
        let trace = replay.run(name, &*f, |g| bisect(a, b, g));
        replay.check(name, "bisect", &trace, row.7, row.8);
    }
}

#[test]
fn every_answer_of_the_battery_is_within_its_tolerance_of_a_sign_change() {
    // With the platform's own libm: every answer within its tolerance of a sign change of the
    // function the method saw, in the evaluations of the model's steps where the function is
    // IEEE arithmetic, the same everywhere, and within four of them where it calls the libm, whose
    // last bits can change the last decisions (glibc's against Apple's: cos x − x, 8 for 10 by
    // `brent`; none by `bisect`). Within the bounds of the methods: 387 and 222 evaluations.
    let opts = RootOptions::new();
    for ((name, a, b, f), row) in battery().into_iter().zip(TABLE) {
        let (lo, hi) = (a.min(b), a.max(b));
        for (method, model, most) in [("brent", row.5, 387), ("bisect", row.7, 222)] {
            let n = Cell::new(0);
            let g = counted(&*f, &n);
            let r = if method == "brent" {
                brent(a, b, g)
            } else {
                bisect(a, b, g)
            };
            let r = r.unwrap_or_else(|| panic!("{method} {name}: no answer"));
            assert!(
                (lo..=hi).contains(&r),
                "{method} {name}: {r} outside the bracket"
            );
            let tol = opts
                .tol_at(r)
                .max((r.abs() * f64::EPSILON).max(f64::MIN_POSITIVE * f64::EPSILON));
            assert!(
                sign_change_near(&*f, r, tol, lo, hi),
                "{method} {name}: {r} is not a root"
            );
            let n = n.get();
            if LIBM.contains(&name) {
                assert!(
                    n.abs_diff(model) <= 4,
                    "{method} {name}: {n} against {model}"
                );
            } else {
                assert_eq!(n, model, "{method} {name}");
            }
            assert!(n <= most, "{method} {name}: {n}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The defects of the audit, each with the inputs that showed it
// ---------------------------------------------------------------------------------------------

#[test]
fn a_bracket_given_high_end_first_is_the_same_bracket() {
    // n_roots-1: with the ends reversed, the width came out negative and the first midpoint
    // passed for converged (bisect(2, 1, x² − 2) was 1.5).
    let f = |x: f64| x * x - 2.0;
    for (a, b) in [(1.0, 2.0), (2.0, 1.0), (100.0, 0.0)] {
        let x = bisect(a, b, f).unwrap();
        assert!(
            (x - SQRT_2).abs() <= RTOL * SQRT_2,
            "bisect({a}, {b}) = {x}"
        );
        assert_eq!(
            brent(a, b, f).map(|x| (x - SQRT_2).abs() <= RTOL * SQRT_2),
            Some(true)
        );
    }
    let x = bisect(1.0, 0.0, |x: f64| x.cos() - x).unwrap();
    assert!((x - DOTTIE).abs() <= RTOL);
    let x = bisect(2.0f32, 1.0, |x: f32| x * x - 2.0).unwrap();
    assert!((f64::from(x) - SQRT_2).abs() <= 4.0 * f64::from(f32::EPSILON) * SQRT_2);
}

#[test]
fn halley_takes_no_stationary_point_for_a_root() {
    // n_roots-2: where f′ = 0 the old step was zero and the point was returned as a root.
    let (f, df, ddf) = (|x: f64| x * x - 1.0, |x: f64| 2.0 * x, |_x: f64| 2.0);
    assert_eq!(halley(0.0, f, df, ddf), None);
    assert_eq!(
        halley(0.0, |x: f64| x.cos(), |x: f64| -x.sin(), |x: f64| -x.cos()),
        None
    );
    // (x² − 1)² + 1/2 has no real root; x = 1 is a minimum with f = 1/2.
    let g = |x: f64| (x * x - 1.0) * (x * x - 1.0) + 0.5;
    let dg = |x: f64| 4.0 * x * (x * x - 1.0);
    let ddg = |x: f64| 12.0 * x * x - 4.0;
    assert_eq!(halley(1.0, g, dg, ddg), None);
    // With an absolute tolerance a start merely near the minimum of x² − 2 was enough (it gave
    // 3e-10); now the iteration leaves it and finds the root, or gives nothing.
    let opts = RootOptions::new().with_xtol(1.48e-8);
    let r = halley_with(
        1e-10,
        |x: f64| x * x - 2.0,
        |x: f64| 2.0 * x,
        |_x: f64| 2.0,
        &opts,
    );
    assert!((r.unwrap() - SQRT_2).abs() <= 1.48e-8 + RTOL * SQRT_2);
}

#[test]
fn halleys_step_falls_back_to_newtons_where_the_correction_would_reverse_it() {
    // x³ − 1 from −0.5: Newton's step is n = f/f′ = −1.5 and n·f″/(2f′) = 3, so Halley's
    // correction n/(1 − 3) would turn the step round, to −1.25, away from the root. The step
    // taken is Newton's, which lands on the root 1 exactly, where f is zero: one iteration.
    let (f, df, ddf) = (
        |x: f64| x * x * x - 1.0,
        |x: f64| 3.0 * x * x,
        |x: f64| 6.0 * x,
    );
    let opts = RootOptions::new().with_max_iter(1);
    assert_eq!(halley_with(-0.5, f, df, ddf, &opts), Some(1.0));
    // In f32 the same numbers are exact.
    let (g, dg, ddg) = (
        |x: f32| x * x * x - 1.0,
        |x: f32| 3.0 * x * x,
        |x: f32| 6.0 * x,
    );
    assert_eq!(
        halley_with(-0.5f32, g, dg, ddg, &RootOptions::new().with_max_iter(1)),
        Some(1.0)
    );
}

#[test]
fn secant_returns_no_point_that_is_not_a_root() {
    // n_roots-3: after a wild first step the next two landed back on the start and the second,
    // "not having moved", was accepted: secant(0.1, 0.2, x¹⁰ − 1) was 0.2, where f = −0.9999999.
    let c7 = [
        -0.652_675_293_549_949_6,
        0.176_177_358_101_240_47,
        -0.495_761_168_943_097_63,
        -0.827_931_391_140_367_8,
        0.117_495_703_762_003_47,
        0.553_300_245_482_729_6,
        0.508_117_349_729_318_3,
        -0.124_684_686_415_344_36,
    ];
    type Start = (f64, f64, Box<dyn Fn(f64) -> f64>);
    let cases: [Start; 4] = [
        (0.1, 0.2, Box::new(|x: f64| x.powi(10) - 1.0)),
        (0.5, 0.6, Box::new(|x: f64| x.powi(20) - 1.0)),
        (2.0, 1.0, Box::new(|x: f64| (50.0 * x).exp() - 5.0)),
        (
            0.217_462_525_132_348_98,
            0.046_328_747_099_934_62,
            Box::new(move |x| c7.iter().rev().fold(0.0, |s, &k| s * x + k)),
        ),
    ];
    let opts = RootOptions::new();
    for (x0, x1, f) in &cases {
        if let Some(r) = secant(*x0, *x1, &**f) {
            assert!(
                sign_change_near(&**f, r, reach(&opts, r), f64::MIN, f64::MAX),
                "{r} is not a root"
            );
        }
    }
    assert_eq!(secant(0.1, 0.2, |x: f64| x.powi(10) - 1.0), None);
    assert_eq!(secant(0.5, 0.6, |x: f64| x.powi(20) - 1.0), None);
    // The f32 cubics of the audit.
    let cubic = |c: [f32; 4]| move |x: f32| ((c[3] * x + c[2]) * x + c[1]) * x + c[0];
    for (c, x0, x1) in [
        (
            [0.326_342_82, -0.946_084_14, -0.920_769_1, 0.487_391_6],
            1.628_638_5f32,
            1.591_445_2,
        ),
        (
            [0.817_448_4, -0.113_659_98, -0.140_332_22, 0.185_594_68],
            0.600_542_07,
            0.150_053_98,
        ),
    ] {
        let f = cubic(c);
        if let Some(r) = secant(x0, x1, f) {
            let tol = (4.0 * f32::EPSILON * r.abs()).max(f32::MIN_POSITIVE * f32::EPSILON);
            assert!(
                sign_change_near32(&f, r, tol, f32::MIN, f32::MAX),
                "{r} is not a root"
            );
        }
    }
}

#[test]
fn a_root_at_zero_is_found_with_the_default_options() {
    // n_roots-8: a relative tolerance is zero at zero, so the old bracketing methods could stop
    // only by landing on zero and gave None. They now stop at two neighbouring floats around the
    // sign change, or on zero itself, within their default limit.
    type F = fn(f64) -> f64;
    let cases: [(f64, f64, F); 7] = [
        (-1.0, 2.0, |x| x),
        (-1.0, 2.0, |x: f64| x.sin()),
        (-9.0, 31.0, |x: f64| -40.0 * x * (-x).exp()),
        (-1.0, 2.0, |x| x * x * x),
        (-1.0, 2.0, |x| pw(x, 9)),
        (-1.0, 2.0, |x: f64| x.cbrt()),
        (-3.0, 1.0, |x| 2.0 * x + x * x * x),
    ];
    for (a, b, f) in cases {
        for (method, r) in [("bisect", bisect(a, b, f)), ("brent", brent(a, b, f))] {
            let r = r.unwrap_or_else(|| panic!("{method}({a}, {b}) found nothing"));
            assert!(r.abs() < 1e-300, "{method}: {r}");
            assert!(sign_change_near(
                &f,
                r,
                f64::MIN_POSITIVE * f64::EPSILON,
                a,
                b
            ));
        }
    }
    // And the root 1e-150 of x² − 1e-300 inside [0, 1], to the default 4ε relative.
    let r: f64 = brent(0.0, 1.0, |x| x * x - 1e-300).unwrap();
    assert!((r - 1e-150).abs() <= RTOL * 1e-150);
    let r = bisect(0.0, 1e20, |x| x * x - 2.0).unwrap();
    assert!((r - SQRT_2).abs() <= RTOL * SQRT_2);
}

#[test]
fn a_multiple_root_at_zero_needs_a_tolerance_of_an_open_method() {
    // The rest of n_roots-8, for the open methods. Newton on x² halves its estimate a step and
    // never meets a relative tolerance at zero; x² does not change sign there, so no tolerance on
    // the root can verify it either, and the honest answer is None. A tolerance on the value finds
    // it; at the triple root of x³, which does change sign, an absolute tolerance does too.
    assert_eq!(newton(1.0, |x: f64| x * x, |x: f64| 2.0 * x), None);
    assert_eq!(secant(1.0, 2.0, |x: f64| x * x), None);
    let by_value = RootOptions::new().with_ftol(1e-20);
    let r = newton_with(1.0, |x: f64| x * x, |x: f64| 2.0 * x, &by_value).unwrap();
    assert!(r * r <= 1e-20 && r > 0.0);
    let r = secant_with(1.0, 2.0, |x: f64| x * x, &by_value).unwrap();
    assert!(r * r <= 1e-20);
    let by_root = RootOptions::new().with_xtol(1e-12);
    let r = newton_with(1.0, |x: f64| x * x * x, |x: f64| 3.0 * x * x, &by_root).unwrap();
    assert!(r.abs() <= 1e-12 + RTOL * r.abs());
}

#[test]
fn brent_stops_on_two_neighbouring_floats() {
    // n_roots-9: between two neighbours further apart than the tolerance — always so among the
    // subnormals, and for an rtol below about ε/2 — no step changed anything, and the old brent
    // spent max_iter on it and gave None.
    let n = Cell::new(0);
    let f = |x: f64| x.sqrt() - 1e-155;
    let r = brent(0.0, 1e-300, counted(&f, &n)).unwrap();
    assert!(
        (r - 1e-310).abs() <= 2.0 * f64::MIN_POSITIVE * f64::EPSILON,
        "{r:e}"
    );
    assert!(n.get() <= 12, "{} evaluations", n.get());
    for rtol in [0.0, f64::EPSILON / 2.0] {
        let opts = RootOptions::new().with_rtol(rtol);
        for x in [
            brent_with(1.0, 2.0, |x| x * x - 2.0, &opts).unwrap(),
            bisect_with(1.0, 2.0, |x| x * x - 2.0, &opts).unwrap(),
        ] {
            assert!(
                (x - SQRT_2).abs() <= f64::EPSILON * SQRT_2,
                "rtol {rtol}: {x}"
            );
        }
    }
}

#[test]
fn brent_among_the_subnormals_ends_within_one_float_of_the_sign_change() {
    // Review, round 1: brent's half-width (c − b)/2 was formed as c/2 − b/2, whose halves of the
    // smallest subnormals round to zero, so it stopped with the estimate and the contrapoint two
    // floats apart around zero: a step at zero, −1 below it and 1 from −0 on, came back as
    // +5·10⁻³²⁴, two floats from the change between −5·10⁻³²⁴ and −0. The brackets are those of
    // the review's stress run.
    let step = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
    for (a, b) in [
        (-1.534_106_548_298_048_5, 1.846_219_702_819_462),
        (2.439_893_696_382_024_6, -1.756_579_208_310_351),
        (-6.313_194_446_214_198e-237, 5.099_893_906_999_709e-237),
        (7.103_930_976_767_058e-182, -2.107_142_115_177_497_7e-181),
    ] {
        let x = brent(a, b, step).unwrap();
        let tol = f64::MIN_POSITIVE * f64::EPSILON;
        assert!(
            sign_change_near(&step, x, tol, a.min(b), a.max(b)),
            "[{a:e}, {b:e}]: {x:e}"
        );
    }
    // And in f32: a step at −6.4·10⁻³⁹, where (x + 2.27547·10⁻⁴⁰)·1.1339331·10⁻⁷ stops
    // underflowing to zero, came back three floats from its change against a reach of 2.2.
    let r: f32 = -2.275_47e-40;
    let f = move |x: f32| {
        if (x - r) * 1.133_933_1e-7 < 0.0 {
            -1.0
        } else {
            1.0
        }
    };
    let (a, b) = (2.307_947e7_f32, -2.377_031_8e7_f32);
    let x = brent(a, b, f).unwrap();
    let reach = (4.0 * f32::EPSILON * x.abs()).max(f32::MIN_POSITIVE * f32::EPSILON);
    assert!(sign_change_near32(&f, x, reach, b, a), "{x:e}");
}

#[test]
fn an_end_that_is_not_finite_is_no_bracket() {
    // n_roots-10: the midpoint of 0 and ∞ is ∞, which the old code took for converged.
    let f = |x: f64| x.atan() - 1.0;
    let g = |x: f64| x.atan() + 1.0;
    for (a, b) in [
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NAN, 2.0),
    ] {
        assert_eq!(bisect(a, b, f), None);
        assert_eq!(brent(a, b, f), None);
    }
    assert_eq!(brent(f64::NEG_INFINITY, 0.0, g), None);
    // A start that is not finite is no start.
    assert_eq!(newton(f64::INFINITY, f, |x: f64| 1.0 / (1.0 + x * x)), None);
    assert_eq!(secant(f64::NAN, 1.0, f), None);
    // An infinite value at a finite end is a sign, which is all bisection asks of it: ln at 0.
    let r = bisect(0.0, 2.0, |x: f64| x.ln()).unwrap();
    assert!((r - 1.0).abs() <= RTOL);
    let r = brent(0.0, 2.0, |x: f64| x.ln()).unwrap();
    assert!((r - 1.0).abs() <= RTOL);
}

#[test]
fn the_steps_neither_underflow_nor_overflow() {
    // n_roots-11: x·f(y) and 2f·f′ were formed before dividing, and underflowed or overflowed.
    let f = |x: f64| x * x - 1e-300;
    let r = secant(1e-140, 2e-140, f).unwrap();
    assert!((r - 1e-150).abs() <= 2.0 * RTOL * 1e-150, "{r:e}");
    let r = halley(1e-140, f, |x: f64| 2.0 * x, |_x: f64| 2.0).unwrap();
    assert!((r - 1e-150).abs() <= 2.0 * RTOL * 1e-150, "{r:e}");
    assert_eq!(secant(1e-150, 2e-150, |x| x - 1e-160), Some(1e-160));
    assert_eq!(secant(2e200, 3e200, |x| x - 1e200), Some(1e200));
    // From 1e110 each Halley step divides x by three: 234 steps, within the default limit.
    let r = halley(1e110, |x: f64| x * x - 2.0, |x: f64| 2.0 * x, |_x: f64| 2.0).unwrap();
    assert!((r - SQRT_2).abs() <= RTOL * SQRT_2);
}

#[test]
fn a_pole_is_returned_only_without_a_tolerance_on_the_value() {
    // n_roots-12: a sign change through a pole is all a bracketing method can see. Without an
    // ftol it is returned, as documented; with one it is refused, since |f| grows there without
    // bound. (The old brent returned it in both cases, contrary to its documentation.)
    let by_value = RootOptions::new().with_ftol(1e-6);
    type F = fn(f64) -> f64;
    let poles: [(f64, f64, F, f64); 3] = [
        (1.0, 2.0, |x: f64| x.tan(), core::f64::consts::FRAC_PI_2),
        (0.0, 1.0, |x| 1.0 / (x - 0.3), 0.3),
        (-1.0, 1.0, |x| 1.0 / x, 0.0),
    ];
    for (a, b, f, pole) in poles {
        for r in [bisect(a, b, f), brent(a, b, f)] {
            let r = r.unwrap();
            assert!(
                (r - pole).abs() <= 2.0 * RTOL * pole.abs().max(f64::MIN_POSITIVE),
                "{r}"
            );
        }
        assert_eq!(bisect_with(a, b, f, &by_value), None);
        assert_eq!(brent_with(a, b, f, &by_value), None);
    }
    // A root is still found with the same ftol, and its value is that small.
    let r = brent_with(1.0, 2.0, |x: f64| x * x - 2.0, &by_value).unwrap();
    assert!((r * r - 2.0).abs() <= 1e-6);
}

#[test]
fn an_exact_zero_is_an_answer_at_the_start_and_on_the_last_step() {
    // n_roots-13: with the default ftol of zero an exact zero was not recognised, at the start or
    // on the last permitted step.
    let sq = |x: f64| (x - 1.0) * (x - 1.0);
    assert_eq!(newton(1.0, sq, |x: f64| 2.0 * (x - 1.0)), Some(1.0));
    assert_eq!(
        halley(1.0, sq, |x: f64| 2.0 * (x - 1.0), |_x: f64| 2.0),
        Some(1.0)
    );
    assert_eq!(newton(0.0, |x: f64| x * x, |x: f64| 2.0 * x), Some(0.0));
    let one = RootOptions::new().with_max_iter(1);
    assert_eq!(
        newton_with(0.0, |x| 3.0 * x - 6.0, |_x| 3.0, &one),
        Some(2.0)
    );
    assert_eq!(secant_with(0.0, 1.0, |x| 3.0 * x - 6.0, &one), Some(2.0));
    let none = RootOptions::new().with_max_iter(0);
    assert_eq!(
        newton_with(2.0, |x| x * x - 4.0, |x| 2.0 * x, &none),
        Some(2.0)
    );
    assert_eq!(
        halley_with(2.0, |x| x * x - 4.0, |x| 2.0 * x, |_x| 2.0, &none),
        Some(2.0)
    );
    assert_eq!(bisect_with(2.0, 3.0, |x| x * x - 4.0, &none), Some(2.0));
}

// ---------------------------------------------------------------------------------------------
// Bisection
// ---------------------------------------------------------------------------------------------

#[test]
fn bisect_finishes_within_the_steps_the_format_allows() {
    // Halving the length takes a binade a step on the way to a root near zero (about 1,075 steps
    // from [-1, 2] to x = 0); halving the count of floats in turn with it takes at most 80
    // iterations of any f64 bracket, as here over the widest brackets and roots of every size.
    let roots = [
        0.0,
        5e-324,
        -5e-324,
        1e-310,
        -1e-300,
        1e-200,
        2.2e-308,
        1e-20,
        -0.5,
        1.0,
        3.7,
        1e20,
        -1e200,
        1e300,
        1.7e308,
        -f64::MAX / 2.0,
    ];
    let brackets = [
        (-f64::MAX, f64::MAX),
        (0.0, f64::MAX),
        (-1.0, 2.0),
        (-f64::MAX, 1.0),
        (-1e-300, 1e300),
    ];
    let mut worst = 0;
    for &(a, b) in &brackets {
        for &r in &roots {
            if !(a < r && r < b) {
                continue;
            }
            for opts in [RootOptions::new(), RootOptions::new().with_rtol(0.0)] {
                let n = Cell::new(0);
                let f = move |x: f64| {
                    if x < r {
                        -1.0
                    } else if x > r {
                        1.0
                    } else {
                        0.0
                    }
                };
                let x = bisect_with(a, b, counted(&f, &n), &opts).unwrap();
                assert!(
                    sign_change_near(&f, x, opts.tol_at(x).max(5e-324), a, b),
                    "{r} in [{a}, {b}]: {x}"
                );
                worst = worst.max(n.get() - 2);
            }
        }
    }
    assert!(worst <= 80, "{worst} iterations");
    // f32: at most 40.
    let mut worst = 0;
    for (a, b) in [(-f32::MAX, f32::MAX), (0.0, f32::MAX), (-1.0, 2.0)] {
        for r in [0.0f32, 1e-45, -1e-40, 1e-30, 0.5, 3.0, 1e30, -1e37] {
            if !(a < r && r < b) {
                continue;
            }
            let n = Cell::new(0);
            let f = move |x: f32| {
                if x < r {
                    -1.0
                } else if x > r {
                    1.0
                } else {
                    0.0
                }
            };
            let x = bisect_with(a, b, counted(&f, &n), &RootOptions::new().with_rtol(0.0)).unwrap();
            assert!(sign_change_near32(&f, x, 1e-45, a, b), "{r}: {x}");
            worst = worst.max(n.get() - 2);
        }
    }
    assert!(worst <= 40, "{worst} iterations of f32");
}

#[test]
fn bisect_with_an_absolute_tolerance_is_the_textbook_method() {
    // An xtol that halving the length reaches in time makes it the method of the textbooks: the
    // bracket [0, 1] halved ten times is within 10⁻³, in 10 + 2 evaluations.
    let opts = RootOptions::new().with_xtol(1e-3);
    let n = Cell::new(0);
    let f = |x: f64| x.cos() - x;
    let x = bisect_with(0.0, 1.0, counted(&f, &n), &opts).unwrap();
    assert_eq!(n.get(), 12);
    assert!((x - DOTTIE).abs() <= 1e-3);
    // And 52 iterations take [1, 2] to the default 4ε.
    let n = Cell::new(0);
    let g = |x: f64| x * x - 2.0;
    let x = bisect(1.0, 2.0, counted(&g, &n)).unwrap();
    assert_eq!(n.get(), 52);
    assert!((x - SQRT_2).abs() <= RTOL * SQRT_2);
}

#[test]
fn a_bracketed_root_is_never_left() {
    // sin has roots at π and 2π; the bracket [3, 4] can only be asking for the first, and every
    // answer of a bracketing method lies inside its bracket.
    for x in [
        brent(3.0, 4.0, |x: f64| x.sin()),
        bisect(3.0, 4.0, |x: f64| x.sin()),
    ] {
        let x = x.unwrap();
        assert!((3.0..=4.0).contains(&x));
        assert!((x - core::f64::consts::PI).abs() <= RTOL * core::f64::consts::PI);
    }
}

#[test]
fn a_root_at_an_end_of_the_bracket_is_returned_exactly() {
    assert_eq!(brent(1.0, 2.0, |x: f64| x - 1.0), Some(1.0));
    assert_eq!(brent(1.0, 2.0, |x: f64| 2.0 - x), Some(2.0));
    assert_eq!(bisect(1.0, 2.0, |x: f64| x - 2.0), Some(2.0));
    assert_eq!(bisect(2.0, 1.0, |x: f64| x - 2.0), Some(2.0));
    // A degenerate bracket on a root is that root; off one it is no bracket.
    assert_eq!(bisect(2.0, 2.0, |x: f64| x - 2.0), Some(2.0));
    assert_eq!(brent(1.0, 1.0, |x: f64| x - 2.0), None);
}

#[test]
fn without_a_sign_change_a_bracketing_method_refuses() {
    assert_eq!(bisect(0.0, 1.0, |x: f64| x * x + 1.0), None);
    assert_eq!(brent(0.0, 1.0, |x: f64| x * x + 1.0), None);
    // Two roots inside, no sign change at the ends: nothing to bracket.
    assert_eq!(brent(-2.0, 2.0, |x: f64| x * x - 1.0), None);
}

// ---------------------------------------------------------------------------------------------
// Brent
// ---------------------------------------------------------------------------------------------

#[test]
fn brent_needs_few_evaluations_on_smooth_problems() {
    // x² − 2 on [1, 2]: the old variant took 37 evaluations; Brent's method takes 9, as brentq.
    for (a, b, f, want, most) in [
        (1.0, 2.0, (|x| x * x - 2.0) as fn(f64) -> f64, SQRT_2, 9),
        (0.0, 1.0, |x: f64| x.cos() - x, DOTTIE, 10),
        (1.0, 2.0, |x| x * x - x - 1.0, PHI, 9),
        (
            2.0,
            3.0,
            |x| x * x * x - 2.0 * x - 5.0,
            2.094_551_481_542_326_5,
            8,
        ),
    ] {
        let n = Cell::new(0);
        let x = brent(a, b, counted(&f, &n)).unwrap();
        assert!(n.get() <= most, "{} evaluations", n.get());
        assert!((x - want).abs() <= RTOL * want, "{x} against {want}");
    }
}

#[test]
fn a_root_of_higher_multiplicity_is_still_found() {
    // At a multiple root interpolation converges only linearly, and Brent's method needs about
    // three steps for each halving of the bracket: 113, 128 and 139 evaluations for the third,
    // fifth and ninth powers at the default tolerances, within the default limit (brentq needs
    // 154, 134 and 140 at the least tolerance it accepts).
    for (a, b, k, most) in [(0.5, 3.0, 3, 113), (0.0, 3.0, 5, 128), (0.0, 3.5, 9, 139)] {
        let f = move |x: f64| pw(x - 1.0, k);
        let n = Cell::new(0);
        let x = brent(a, b, counted(&f, &n)).unwrap();
        assert_eq!(n.get(), most);
        assert!(sign_change_near(&f, x, RTOL * 2.0, a, b));
        assert!((x - 1.0).abs() <= 4.0 * RTOL, "{x}");
        let y = bisect(a, b, f).unwrap();
        assert!((y - 1.0).abs() <= 2.0 * RTOL, "{y}");
    }
}

#[test]
fn the_bracketing_methods_converge_on_every_random_bracket() {
    // Ten families of functions with a root at a random place, scaled by 10^±20, on random
    // brackets in either order, some with the root at zero: every call must answer, inside its
    // bracket, within the tolerance of a sign change, and within the bounds of the methods.
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let n = if cfg!(miri) { 40 } else { 4000 };
    // Per family: evaluations by bisect and by brent, and the largest number of iterations.
    let mut sums = [[0usize; 2]; 10];
    let mut worst = [0usize; 2];
    for k in 0..n {
        let family = k % 10;
        let scale = 10f64.powf(rng.range(-20.0, 20.0));
        let r = if k % 7 == 0 {
            0.0
        } else {
            rng.range(-1.0, 1.0) * scale
        };
        let (left, right) = (rng.range(0.01, 3.0) * scale, rng.range(0.01, 3.0) * scale);
        let (a, b) = if k % 2 == 0 {
            (r - left, r + right)
        } else {
            (r + right, r - left)
        };
        let s = rng.range(0.1, 10.0) / scale;
        let f: Box<dyn Fn(f64) -> f64> = match family {
            0 => Box::new(move |x| (x - r) * s),
            1 => Box::new(move |x| {
                let t = (x - r) * s;
                t * (t * t + 0.5)
            }),
            2 => Box::new(move |x: f64| ((x - r) * s).tanh()),
            3 => Box::new(move |x: f64| ((x - r) * s).exp() - 1.0),
            4 => Box::new(move |x| pw((x - r) * s, 3)),
            5 => Box::new(move |x: f64| ((x - r) * s).abs().sqrt().copysign(x - r)),
            6 => Box::new(move |x| {
                let t = (x - r) * s;
                t / (1.0 + t * t)
            }),
            7 => Box::new(move |x| {
                let t = (x - r) * s;
                if t < 0.0 { 3.0 * t } else { 0.01 * t }
            }),
            8 => Box::new(move |x: f64| ((x - r) * s / 10.5).sin()),
            _ => Box::new(move |x| pw((x - r) * s, 5) + (x - r) * s * 1e-3),
        };
        let (lo, hi) = (a.min(b), a.max(b));
        for method in [0, 1] {
            let count = Cell::new(0);
            let g = counted(&*f, &count);
            let x = if method == 0 {
                bisect(a, b, g)
            } else {
                brent(a, b, g)
            };
            let x = x
                .unwrap_or_else(|| panic!("case {k}: method {method} found nothing on [{a}, {b}]"));
            assert!((lo..=hi).contains(&x), "case {k}");
            let tol = (RTOL * x.abs()).max(f64::MIN_POSITIVE * f64::EPSILON);
            assert!(
                sign_change_near(&*f, x, tol, lo, hi),
                "case {k}, method {method}: {x} for {r}"
            );
            sums[family][method] += count.get();
            worst[method] = worst[method].max(count.get() - 2);
        }
    }
    assert!(worst[0] <= 80, "bisect: {} iterations", worst[0]);
    assert!(worst[1] <= 256, "brent: {} iterations", worst[1]);
    // Where the root is simple and the function smooth (linear, tanh, exponential, rational,
    // sine), Brent averages at most 12 evaluations to bisection's 55 or so; at the triple root of
    // family 4 it converges only linearly, about 100. Over all of them it needs less than half.
    let per = (n / 10) as f64;
    for family in [0, 2, 3, 6, 8] {
        let mean = sums[family][1] as f64 / per;
        assert!(
            mean <= 12.0,
            "family {family}: brent averages {mean} evaluations"
        );
    }
    let total: [usize; 2] = [
        sums.iter().map(|s| s[0]).sum(),
        sums.iter().map(|s| s[1]).sum(),
    ];
    assert!(2 * total[1] < total[0], "{total:?}");
}

#[test]
fn f32_brackets_converge_to_f32_tolerance() {
    let mut rng = Rng(0xfeed_beef_dead_cafe);
    let n = if cfg!(miri) { 20 } else { 2000 };
    let tol32 = |x: f32| (4.0 * f32::EPSILON * x.abs()).max(f32::MIN_POSITIVE * f32::EPSILON);
    for k in 0..n {
        let scale = 10f64.powf(rng.range(-10.0, 10.0));
        let r = if k % 5 == 0 {
            0.0
        } else {
            (rng.range(-1.0, 1.0) * scale) as f32
        };
        let a = r - (rng.range(0.01, 3.0) * scale) as f32;
        let b = r + (rng.range(0.01, 3.0) * scale) as f32;
        let s = (1.0 / scale) as f32;
        let f = move |x: f32| {
            let t = (x - r) * s;
            t * (t * t + 0.5)
        };
        for x in [bisect(a, b, f), brent(b, a, f)] {
            let x = x.unwrap();
            assert!((a..=b).contains(&x));
            assert!(
                sign_change_near32(&f, x, tol32(x), a, b),
                "case {k}: {x} for {r}"
            );
        }
    }
    // The default rtol of f32 is 4·f32::EPSILON, and √2 comes out to it.
    let x = brent(1.0f32, 2.0, |x: f32| x * x - 2.0).unwrap();
    assert!((f64::from(x) - SQRT_2).abs() <= 4.0 * f64::from(f32::EPSILON) * SQRT_2);
    assert_eq!(RootOptions::<f32>::new().rtol, 4.0 * f32::EPSILON);
}

// ---------------------------------------------------------------------------------------------
// The open methods
// ---------------------------------------------------------------------------------------------

/// A random polynomial of degree in `degrees`, its coefficients uniform in [-1, 1].
fn polynomial(rng: &mut Rng, degrees: (u64, u64)) -> Vec<f64> {
    let d = degrees.0 + rng.next() % (degrees.1 - degrees.0 + 1);
    (0..=d).map(|_| rng.range(-1.0, 1.0)).collect()
}

/// Horner's rule for `c` (constant first) and its first two derivatives.
fn horner(c: &[f64], x: f64) -> (f64, f64, f64) {
    let (mut p, mut d, mut dd) = (0.0, 0.0, 0.0);
    for &k in c.iter().rev() {
        dd = dd * x + 2.0 * d;
        d = d * x + p;
        p = p * x + k;
    }
    (p, d, dd)
}

#[test]
fn the_open_methods_return_only_verified_roots_on_random_polynomials() {
    // The audit's statistics: on such polynomials the old secant returned non-roots for 0.2% of
    // its answers at degrees 3 to 7 and 13% at 10 to 20. Every answer must now be within its
    // tolerance of a sign change of the same float polynomial, or a zero of it.
    let n = if cfg!(miri) { 30 } else { 6000 };
    let opts = RootOptions::new();
    for degrees in [(3, 7), (10, 20)] {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ degrees.0);
        let mut answers = [0usize; 3];
        for _ in 0..n {
            let c = polynomial(&mut rng, degrees);
            let x0 = rng.range(-2.0, 2.0);
            let x1 = x0 + rng.range(-0.1, 0.1);
            let f = |x: f64| horner(&c, x).0;
            let rs = [
                secant(x0, x1, f),
                newton(x0, f, |x| horner(&c, x).1),
                halley(x0, f, |x| horner(&c, x).1, |x| horner(&c, x).2),
            ];
            for (i, r) in rs.into_iter().enumerate() {
                if let Some(r) = r {
                    answers[i] += 1;
                    assert!(
                        sign_change_near(&f, r, reach(&opts, r), f64::MIN, f64::MAX),
                        "method {i} on {c:?} from {x0}, {x1}: {r} is not a root"
                    );
                }
            }
        }
        // Most problems are solved by each method; the rest have no real root nearby, or lead
        // the iteration astray, and are refused.
        for (i, a) in answers.iter().enumerate() {
            assert!(
                *a > n / 3,
                "method {i} answered only {a} of {n} at degrees {degrees:?}"
            );
        }
    }
}

#[test]
fn the_open_methods_return_only_verified_roots_in_f32() {
    let n = if cfg!(miri) { 30 } else { 6000 };
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut answers = 0;
    for _ in 0..n {
        let c: Vec<f32> = polynomial(&mut rng, (2, 6))
            .iter()
            .map(|&v| v as f32)
            .collect();
        let x0 = rng.range(-2.0, 2.0) as f32;
        let x1 = x0 + rng.range(-0.1, 0.1) as f32;
        let f = |x: f32| c.iter().rev().fold(0.0f32, |s, &k| s * x + k);
        let df = |x: f32| {
            let (mut p, mut d) = (0.0f32, 0.0f32);
            for &k in c.iter().rev() {
                d = d * x + p;
                p = p * x + k;
            }
            d
        };
        let ddf = |x: f32| {
            let (mut p, mut d, mut dd) = (0.0f32, 0.0f32, 0.0f32);
            for &k in c.iter().rev() {
                dd = dd * x + 2.0 * d;
                d = d * x + p;
                p = p * x + k;
            }
            dd
        };
        for r in [secant(x0, x1, f), newton(x0, f, df), halley(x0, f, df, ddf)]
            .into_iter()
            .flatten()
        {
            answers += 1;
            let tol = (4.0 * f32::EPSILON * r.abs()).max(f32::MIN_POSITIVE * f32::EPSILON);
            assert!(
                sign_change_near32(&f, r, tol, f32::MIN, f32::MAX),
                "{c:?} from {x0}, {x1}: {r}"
            );
        }
    }
    assert!(answers > n, "{answers} answers");
}

/// The orders of convergence an error sequence shows: log e(k+1) / log e(k), from the errors
/// above 10⁻¹² (below, rounding takes over).
fn orders(errors: &[f64]) -> Vec<f64> {
    errors
        .windows(2)
        .filter(|w| w[1] > 1e-12 && w[0] < 0.1)
        .map(|w| w[1].ln() / w[0].ln())
        .collect()
}

#[test]
fn the_open_methods_converge_at_their_orders() {
    // On e^x − 2, from 2 (and 2.1 for the secant): the errors are recorded at every evaluation.
    let root = core::f64::consts::LN_2;
    let trace = std::cell::RefCell::new(Vec::new());
    let f = |x: f64| {
        trace.borrow_mut().push((x - root).abs());
        x.exp() - 2.0
    };
    newton(2.0, f, |x: f64| x.exp()).unwrap();
    let newton_orders = orders(&trace.take());
    halley(2.0, f, |x: f64| x.exp(), |x: f64| x.exp()).unwrap();
    let halley_orders = orders(&trace.take());
    secant(2.0, 2.1, f).unwrap();
    let secant_orders = orders(&trace.take()[2..]);
    // The orders are log ratios of errors, which approach the order from below.
    assert!(
        newton_orders.iter().any(|&p| (1.9..2.2).contains(&p)),
        "{newton_orders:?}"
    );
    assert!(
        halley_orders.iter().any(|&p| (2.8..3.5).contains(&p)),
        "{halley_orders:?}"
    );
    assert!(
        secant_orders.iter().any(|&p| (1.5..1.8).contains(&p)),
        "{secant_orders:?}"
    );
}

#[test]
fn every_open_method_finds_the_same_root() {
    let (f, df, ddf) = (
        |x: f64| x * x - x - 1.0,
        |x: f64| 2.0 * x - 1.0,
        |_x: f64| 2.0,
    );
    for x in [
        brent(1.0, 2.0, f),
        secant(1.0, 2.0, f),
        newton(1.0, f, df),
        halley(1.0, f, df, ddf),
    ] {
        assert!((x.unwrap() - PHI).abs() <= RTOL * PHI);
    }
}

#[test]
fn an_open_method_that_wanders_reports_none() {
    // arctan with its true derivative: from 2 Newton's steps overshoot further each time, to
    // infinity. (The old test passed the constant 1 for the derivative and could not fail.)
    assert_eq!(
        newton(2.0, |x: f64| x.atan(), |x: f64| 1.0 / (1.0 + x * x)),
        None
    );
    // x³ − 2x + 2 from 0 cycles between 0 and 1 for ever, never near its root at −1.77.
    let (f, df) = (
        |x: f64| x * x * x - 2.0 * x + 2.0,
        |x: f64| 3.0 * x * x - 2.0,
    );
    assert_eq!(newton(0.0, f, df), None);
    // A zero derivative on the way, and one that is not finite.
    assert_eq!(newton(1.0, |x: f64| x * x - 2.0, |_x: f64| 0.0), None);
    assert_eq!(newton(1.0, |x: f64| x * x - 2.0, |_x: f64| f64::NAN), None);
    // A NaN value at the start or on the way.
    assert_eq!(newton(1.0, |_x: f64| f64::NAN, |_x: f64| 1.0), None);
    assert_eq!(
        secant(1.0, 2.0, |x: f64| if x > 1.5 { f64::NAN } else { x - 3.0 }),
        None
    );
    assert_eq!(
        brent(-1.0, 1.0, |x: f64| if x > 0.0 { x - 0.5 } else { f64::NAN }),
        None
    );
    // The iteration cap: one Newton step from 1 is 1.5, still far from √2.
    let one = RootOptions::new().with_max_iter(1);
    assert_eq!(newton_with(1.0, |x| x * x - 2.0, |x| 2.0 * x, &one), None);
    // Starting points that coincide give the secant nothing to draw.
    assert_eq!(secant(1.0, 1.0, |x: f64| x * x - 2.0), None);
    // And a flat chord: equal values at two different points.
    assert_eq!(secant(-1.0, 1.0, |x: f64| x * x - 2.0), None);
}

#[test]
fn a_step_that_no_longer_moves_is_no_answer() {
    // With a derivative of 10³⁰ the step from 1 is far below the spacing of the floats, so the
    // estimate stays at 1, where x² + 1 = 2: no root, and no sign change near it.
    assert_eq!(newton(1.0, |x: f64| x * x + 1.0, |_x: f64| 1e30), None);
    assert_eq!(
        halley(1.0, |x: f64| x * x + 1.0, |_x: f64| 1e30, |_x: f64| 0.0),
        None
    );
    // The secant's chord through a jump of 10³⁰ stops on the far side of it, where f is 1.
    let jump = |x: f64| if x < 1.5 { -1e30 } else { 1.0 };
    assert_eq!(secant(1.0, 2.0, jump), None);
}

#[test]
fn a_zero_derivative_at_a_root_is_no_obstacle() {
    // cos′ vanishes at 0, but the root of cos x − x is elsewhere and Newton's first step from 0
    // takes the derivative there, −1.
    let x = newton(0.0, |x: f64| x.cos() - x, |x: f64| -x.sin() - 1.0).unwrap();
    assert!((x - DOTTIE).abs() <= RTOL);
}

// ---------------------------------------------------------------------------------------------
// The options
// ---------------------------------------------------------------------------------------------

#[test]
fn ftol_accepts_on_the_value_and_only_there() {
    // With an ftol each method stops as soon as |f| is that small, sooner than the root's own
    // tolerance would, and never returns a value above it.
    let f = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;
    let opts = RootOptions::new().with_ftol(1e-6);
    let n = Cell::new(0);
    let x = bisect_with(1.0, 2.0, counted(&f, &n), &opts).unwrap();
    assert!(
        f(x).abs() <= 1e-6 && n.get() < 30,
        "{} evaluations",
        n.get()
    );
    for x in [
        brent_with(1.0, 2.0, f, &opts),
        secant_with(1.0, 2.0, f, &opts),
        newton_with(1.0, f, df, &opts),
        halley_with(1.0, f, df, |_x| 2.0, &opts),
    ] {
        assert!(f(x.unwrap()).abs() <= 1e-6);
    }
    // An ftol below what the floats can give refuses even the best estimate.
    let tiny = RootOptions::new().with_ftol(1e-30);
    assert_eq!(brent_with(1.0, 2.0, f, &tiny), None);
    assert_eq!(bisect_with(1.0, 2.0, f, &tiny), None);
}

#[test]
fn xtol_bounds_the_error_of_a_flat_function() {
    // A slope of 10⁻⁶ makes the tolerance on x the only one that matters: (x − 0.3)·10⁻⁶ on
    // [0, 1] is also solved by bisection to within xtol, the cubic one by Brent.
    let opts = RootOptions::new().with_xtol(1e-12);
    let x = bisect_with(0.0, 1.0, |x: f64| (x - 0.3) * 1e-6, &opts).unwrap();
    assert!((x - 0.3).abs() <= 1e-12 + RTOL * 0.3);
    let x = brent_with(0.0, 1.0, |x: f64| pw(x - 0.3, 3) * 1e-6, &opts).unwrap();
    assert!((x - 0.3).abs() <= 1e-12 + RTOL * 0.3, "{x}");
}

#[test]
fn invalid_tolerances_answer_nothing() {
    let f = |x: f64| x * x - 2.0;
    for opts in [
        RootOptions::new().with_xtol(-1.0),
        RootOptions::new().with_rtol(f64::NAN),
        RootOptions::new().with_ftol(f64::INFINITY),
        RootOptions::new().with_xtol(f64::NAN),
    ] {
        assert_eq!(bisect_with(1.0, 2.0, f, &opts), None);
        assert_eq!(brent_with(1.0, 2.0, f, &opts), None);
        assert_eq!(secant_with(1.0, 2.0, f, &opts), None);
        assert_eq!(newton_with(1.0, f, |x| 2.0 * x, &opts), None);
        assert_eq!(halley_with(1.0, f, |x| 2.0 * x, |_x| 2.0, &opts), None);
    }
}

#[test]
fn the_defaults_are_stated() {
    let o = RootOptions::<f64>::new();
    assert_eq!(
        (o.xtol, o.rtol, o.ftol, o.max_iter),
        (0.0, 4.0 * f64::EPSILON, 0.0, 512)
    );
    assert_eq!(RootOptions::<f64>::default(), o);
    assert_eq!(o.tol_at(-2.0), 8.0 * f64::EPSILON);
}

#[test]
fn f32_problems_are_solved_to_f32_tolerance() {
    let tol = 4.0 * f64::from(f32::EPSILON);
    let f = |x: f32| x * x - 2.0;
    let df = |x: f32| 2.0 * x;
    for x in [
        bisect(1.0f32, 2.0, f),
        brent(1.0f32, 2.0, f),
        secant(1.0f32, 2.0, f),
        newton(1.0f32, f, df),
        halley(1.0f32, f, df, |_x| 2.0),
    ] {
        assert!((f64::from(x.unwrap()) - SQRT_2).abs() <= tol * SQRT_2);
    }
    let x = bisect(0.0f32, 1.0, |x: f32| x.cos() - x).unwrap();
    assert!((f64::from(x) - DOTTIE).abs() <= tol);
    // Roots at zero in f32 as in f64.
    assert!(brent(-1.0f32, 2.0, |x: f32| pw32(x, 3)).unwrap().abs() < 1e-37);
    assert!(bisect(-1.0f32, 2.0, |x: f32| x).unwrap().abs() < 1e-44);
}

#[test]
fn a_root_far_from_the_origin_keeps_its_relative_accuracy() {
    // The root is 10⁸ and the slope there 2·10⁸, so only a relative test can pass.
    let x = brent(1e8, 2e8, |x: f64| x * x - 1e16).unwrap();
    assert!((x - 1e8).abs() <= RTOL * 1e8);
    let x = bisect(1e8, 2e8, |x: f64| x * x - 1e16).unwrap();
    assert!((x - 1e8).abs() <= RTOL * 1e8);
}

#[test]
fn a_root_near_the_origin_is_found_to_its_relative_accuracy() {
    // 3·10⁻⁹ and 10⁻²⁰, by both bracketing methods and the defaults alone.
    for (f, want) in [
        ((|x: f64| x - 3e-9) as fn(f64) -> f64, 3e-9),
        (|x: f64| (x - 1e-20) * (x * x + 1.0), 1e-20),
        (|x: f64| x * x - 1e-40, 1e-20),
    ] {
        for x in [brent(0.0, 1.0, f), bisect(0.0, 1.0, f)] {
            let x = x.unwrap();
            assert!(
                (x - want).abs() <= 2.0 * RTOL * want,
                "{x:e} against {want:e}"
            );
        }
    }
}
