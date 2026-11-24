//! A reference for the quaternion functions in the crate's correctly rounded `BigFloat`, at 160
//! bits: the same polar formulas as mpmath in `data/gen_mpmath.py`, evaluated far beyond the
//! precision of an `f64`, so that random sweeps need no data file. It shares no floating-point
//! code with `a::math::quaternion`: only the formulas, which the mpmath data check.

use a::math::float::BigFloat;
use a::math::quaternion::Quaternion;

const P: u64 = 160;

pub type Big = [BigFloat; 4];

pub fn big(q: Quaternion<f64>) -> Big {
    [q.w, q.x, q.y, q.z].map(|t| BigFloat::from(t).with_prec(P))
}

pub fn small(b: &Big) -> Quaternion<f64> {
    Quaternion::new(b[0].to_f64(), b[1].to_f64(), b[2].to_f64(), b[3].to_f64())
}

fn zero() -> BigFloat {
    BigFloat::from(0.0).with_prec(P)
}

pub fn mul(a: &Big, b: &Big) -> Big {
    [
        &(&(&a[0] * &b[0]) - &(&a[1] * &b[1])) - &(&(&a[2] * &b[2]) + &(&a[3] * &b[3])),
        &(&(&a[0] * &b[1]) + &(&a[1] * &b[0])) + &(&(&a[2] * &b[3]) - &(&a[3] * &b[2])),
        &(&(&a[0] * &b[2]) - &(&a[1] * &b[3])) + &(&(&a[2] * &b[0]) + &(&a[3] * &b[1])),
        &(&(&a[0] * &b[3]) + &(&a[1] * &b[2])) - &(&(&a[2] * &b[1]) - &(&a[3] * &b[0])),
    ]
}

fn scale(a: &Big, t: &BigFloat) -> Big {
    [&a[0] * t, &a[1] * t, &a[2] * t, &a[3] * t]
}

fn vlen2(a: &Big) -> BigFloat {
    &(&(&a[1] * &a[1]) + &(&a[2] * &a[2])) + &(&a[3] * &a[3])
}

pub fn norm2(a: &Big) -> BigFloat {
    &(&a[0] * &a[0]) + &vlen2(a)
}

pub fn norm(q: Quaternion<f64>) -> f64 {
    norm2(&big(q)).sqrt().to_f64()
}

pub fn recip(q: Quaternion<f64>) -> Quaternion<f64> {
    let a = big(q);
    let n = norm2(&a);
    small(&[&a[0] / &n, &(-&a[1]) / &n, &(-&a[2]) / &n, &(-&a[3]) / &n])
}

pub fn product(p: Quaternion<f64>, q: Quaternion<f64>) -> Quaternion<f64> {
    small(&mul(&big(p), &big(q)))
}

/// `p·q⁻¹`.
pub fn quotient(p: Quaternion<f64>, q: Quaternion<f64>) -> Quaternion<f64> {
    let b = big(q);
    let n = norm2(&b);
    let c = [b[0].clone(), -&b[1], -&b[2], -&b[3]];
    let m = mul(&big(p), &c);
    small(&[&m[0] / &n, &m[1] / &n, &m[2] / &n, &m[3] / &n])
}

pub fn exp_big(a: &Big) -> Big {
    let r = vlen2(a).sqrt();
    let e = a[0].exp();
    if r.is_zero() {
        return [e, a[1].clone(), a[2].clone(), a[3].clone()];
    }
    let (s, c) = r.sin_cos();
    let f = &(&e * &s) / &r;
    [&e * &c, &a[1] * &f, &a[2] * &f, &a[3] * &f]
}

pub fn ln_big(a: &Big) -> Big {
    let r = vlen2(a).sqrt();
    let re = &norm2(a).ln() * 0.5;
    if r.is_zero() {
        let x = if a[0] < 0.0 {
            a[1].atan2(&a[0])
        } else {
            zero()
        };
        return [re, x, a[2].clone(), a[3].clone()];
    }
    let f = &r.atan2(&a[0]) / &r;
    [re, &a[1] * &f, &a[2] * &f, &a[3] * &f]
}

pub fn exp(q: Quaternion<f64>) -> Quaternion<f64> {
    small(&exp_big(&big(q)))
}

pub fn ln(q: Quaternion<f64>) -> Quaternion<f64> {
    small(&ln_big(&big(q)))
}

/// `exp(ln(q)·p)`.
pub fn powq(q: Quaternion<f64>, p: Quaternion<f64>) -> Quaternion<f64> {
    small(&exp_big(&mul(&ln_big(&big(q)), &big(p))))
}

pub fn powf(q: Quaternion<f64>, t: f64) -> Quaternion<f64> {
    small(&exp_big(&scale(&ln_big(&big(q)), &BigFloat::from(t))))
}

/// The principal square root, `exp(ln(q)/2)`.
pub fn sqrt(q: Quaternion<f64>) -> Quaternion<f64> {
    powf(q, 0.5)
}

/// `q·(0 + v)·q⁻¹`, the vector part.
pub fn rotate(q: Quaternion<f64>, v: [f64; 3]) -> [f64; 3] {
    let a = big(q);
    let n = norm2(&a);
    let c = [&a[0] / &n, &(-&a[1]) / &n, &(-&a[2]) / &n, &(-&a[3]) / &n];
    let r = mul(&mul(&a, &big(Quaternion::from_parts(0.0, v))), &c);
    [r[1].to_f64(), r[2].to_f64(), r[3].to_f64()]
}

/// The four products of each part of `a·b`, with their signs.
fn terms(a: &Big, b: &Big) -> [[BigFloat; 4]; 4] {
    let m = |x: &BigFloat, y: &BigFloat| x * y;
    [
        [
            m(&a[0], &b[0]),
            -m(&a[1], &b[1]),
            -m(&a[2], &b[2]),
            -m(&a[3], &b[3]),
        ],
        [
            m(&a[0], &b[1]),
            m(&a[1], &b[0]),
            m(&a[2], &b[3]),
            -m(&a[3], &b[2]),
        ],
        [
            m(&a[0], &b[2]),
            -m(&a[1], &b[3]),
            m(&a[2], &b[0]),
            m(&a[3], &b[1]),
        ],
        [
            m(&a[0], &b[3]),
            m(&a[1], &b[2]),
            -m(&a[2], &b[1]),
            m(&a[3], &b[0]),
        ],
    ]
}

/// What a part of a product checks: `got` within `err` of the exact part `t`, an infinity
/// counting as beyond the largest float, so that `+∞` passes where `t + err` reaches past it and
/// `−∞` where `t − err` does (both only for a part whose error band straddles zero). A NaN fails.
fn part_ok(t: &BigFloat, err: &BigFloat, got: f64) -> bool {
    if got.is_nan() {
        return false;
    }
    if got.is_infinite() {
        return if got > 0.0 {
            &(t + err) - f64::MAX >= 0.0
        } else {
            &(t - err) + f64::MAX <= 0.0
        };
    }
    let d = &(&BigFloat::from(got).with_prec(P) - t).abs() - err;
    d <= 0.0
}

/// The per-part check of a product: each part of `got` within `c·ε` times the sum of the
/// absolute values of that part's four products of the exact part, or the infinity of its sign
/// where that reaches past the largest float. Returns whether a part of the exact product
/// overflows, how many parts of an overflowing product are finite there, and whether all are
/// right.
pub fn check_product_parts(
    p: Quaternion<f64>,
    q: Quaternion<f64>,
    got: Quaternion<f64>,
    c: f64,
) -> (bool, usize, bool, f64) {
    let t = terms(&big(p), &big(q));
    let g = [got.w, got.x, got.y, got.z];
    let eps = f64::EPSILON * c;
    let mut overflow = false;
    let mut finite = 0;
    let mut ok = true;
    let mut worst = 0.0f64;
    for (part, &x) in t.iter().zip(g.iter()) {
        let exact = &(&(&part[0] + &part[1]) + &part[2]) + &part[3];
        let size = &(&(&part[0].abs() + &part[1].abs()) + &part[2].abs()) + &part[3].abs();
        if exact.to_f64().is_infinite() {
            overflow = true;
        } else {
            finite += 1;
            if x.is_finite() && !size.is_zero() {
                let e = &(&BigFloat::from(x).with_prec(P) - &exact).abs() / &size;
                worst = worst.max(e.to_f64() / f64::EPSILON);
            }
        }
        ok &= part_ok(&exact, &(&size * eps), x);
    }
    (overflow, if overflow { finite } else { 0 }, ok, worst)
}

/// The normwise check of `q^n`, the exact power formed by repeated squaring in 160 bits (whose
/// exponent range has no limit that matters here): each part of `got` within `tol·|qⁿ|` of the
/// exact part, or the infinity of its sign where that reaches past the largest float; and
/// whether a part of the exact power overflows.
pub fn check_power(q: Quaternion<f64>, n: u32, got: Quaternion<f64>, tol: f64) -> (bool, bool) {
    let mut base = big(q);
    let mut r: Option<Big> = None;
    let mut n = n;
    while n > 0 {
        if n & 1 == 1 {
            r = Some(match r {
                None => base.clone(),
                Some(r) => mul(&r, &base),
            });
        }
        n >>= 1;
        if n > 0 {
            base = mul(&base, &base);
        }
    }
    let exact = r.unwrap();
    let err = &norm2(&exact).sqrt() * tol;
    let g = [got.w, got.x, got.y, got.z];
    let overflow = exact.iter().any(|t| t.to_f64().is_infinite());
    let ok = exact
        .iter()
        .zip(g.iter())
        .all(|(t, &x)| part_ok(t, &err, x));
    (overflow, ok)
}
