//! The inverse circular and hyperbolic functions, each part correctly rounded, with MPC's branch
//! cuts and special values (`mpc_asin`, `mpc_acos`, `mpc_atan`, `mpc_asinh`, `mpc_acosh`,
//! `mpc_atanh`).
//!
//! For `z = x + yi` with `x`, `y` finite and not zero:
//!
//! - `asin` and `acos` by the formulas of Hull, Fairgrieve and Tang (1997): with `r = |z + 1|`,
//!   `s = |z - 1|` and `A = (r + s)/2 >= 1`, `Re asin z = atan2(x, √(A² - x²))`, `Re acos z =
//!   atan2(√(A² - x²), x)` and `Im asin z = -Im acos z = sign(y)·ln(A + √(A² - 1))`, where `A - 1`
//!   and `A - |x|` are sums of terms that are not below zero (`r - (|x| + 1) = y²/(r + |x| + 1)`,
//!   `s - |1 - |x|| = y²/(s + |1 - |x||)`), so that nothing cancels; the logarithm is `ln(1 + t)`
//!   of `t = (A - 1) + √((A - 1)(A + 1))`.
//! - `atan z = atan2(2x, 1 - x² - y²)/2 + i·sign(y)·ln(1 + 4|y|/(x² + (|y| - 1)²))/4`, `1 - x² -
//!   y²` exact (or cut to the working precision when the parts lie far apart, see
//!   [`norm_m1`](super::real::norm_m1)).
//! - `asinh z = -i asin(iz)`, `atanh z = -i atan(iz)` and `acosh z = ±i acos z` (the sign of `y`'s),
//!   as MPC, the precisions and the modes swapped with the parts.
//! - For a tiny `z` the parts lie within less than their last place of `x` and `y`: they are
//!   rounded beside them by the first term of the series ([`odd_tiny`](super::real::odd_tiny)).
//!
//! No part is exact for `x` and `y` not zero (by the Lindemann–Weierstrass theorem, as `sin`,
//! `cos` and `tan` of an algebraic number not zero are transcendental).

use super::BigComplex;
use super::real::{
    self, Part, Step, abs, add, atan2, check, div, exactly, inf, ln1p, mul, nan, nans, negated,
    norm_m1, odd_tiny, of, one, pair, pi_part, set, shl, signed, sqr, sqrt, step, xadd, zero_of,
};
use crate::math::float::arith::round_for;
use crate::math::float::sincos::Fx;
use crate::math::float::{BigFloat, Round};
use core::cmp::Ordering;

type Parts = (Part, Part);

/// A function of the parts of `x + yi`.
type Fun = fn(&BigFloat, &BigFloat, (u64, u64), (Round, Round)) -> Parts;

/// The parts of `asin` and `acos` of `|x| + |y|i` at the working precision `w`: `(√(A² - x²), ln(A
/// + √(A² - 1)))`, `None` when a step's bound is too wide.
fn hull(x: &Fx, y: &Fx, ge1: bool, w: u64) -> Option<(Fx, Fx)> {
    let w2 = w + 16;
    let xp1 = xadd(x, &one(), w2);
    let xm1 = xadd(x, &real::neg(one()), w2);
    let y2 = sqr(y, w2);
    let r = sqrt(&add(&sqr(&xp1, w2), &y2, w2), w2)?;
    let s = sqrt(&add(&sqr(&xm1, w2), &y2, w2), w2)?;
    let t1 = div(&y2, &add(&r, &xp1, w2), w2)?;
    // `s ± (|x| - 1)` for `|x| >= 1`, `s ∓ ...` below.
    let s_plus = add(&s, &xm1, w2);
    let s_minus = add(&s, &real::neg(xm1.clone()), w2);
    let (am1, amx) = if ge1 {
        (add(&t1, &s_plus, w2), add(&t1, &div(&y2, &s_plus, w2)?, w2))
    } else {
        (
            add(&t1, &div(&y2, &s_minus, w2)?, w2),
            add(&t1, &s_minus, w2),
        )
    };
    let (am1, amx) = (shl(am1, -1), shl(amx, -1));
    let apx = add(&shl(add(&r, &s, w2), -1), x, w2);
    let d = sqrt(&mul(&amx, &apx, w2), w2)?;
    let t = add(
        &am1,
        &sqrt(&mul(&am1, &add(&am1, &real::int(2), w2), w2), w2)?,
        w2,
    );
    let l = ln1p(&t, w2)?;
    Some((d, l))
}

/// `asin(x + yi)`, or `acos(x + yi)` when `cos`.
fn asin_acos(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round), cos: bool) -> Parts {
    let ny = y.is_sign_negative();
    let fx = of(x);
    let fy = of(y);
    let (ax, ay) = (abs(fx.clone()), abs(fy.clone()));
    let ge1 = real::cmp_mag(&ax, &one()) != Ordering::Less;
    // Tiny: `asin z = z + z³/6 + ...`; `acos z = π/2 - asin z`, its imaginary part `-Im asin z`.
    let mut early: [Option<Part>; 2] = [None, None];
    if !cos {
        early[0] = odd_tiny(&fx, &fy, 1, false, p.0, r.0);
    }
    early[1] = if cos {
        odd_tiny(&fx, &fy, 1, true, p.1, round_for(r.1, true)).map(negated)
    } else {
        odd_tiny(&fx, &fy, 1, true, p.1, r.1)
    };
    real::ziv2(p, r, |w, need| {
        let mut out: [Option<Step>; 2] = [None, None];
        let mut n = need;
        for i in 0..2 {
            if need[i]
                && let Some(v) = early[i].take()
            {
                out[i] = Some(Err(v));
                n[i] = false;
            }
        }
        if !n[0] && !n[1] {
            return out;
        }
        let Some((d, l)) = hull(&ax, &ay, ge1, w) else {
            for (i, o) in out.iter_mut().enumerate() {
                if n[i] {
                    *o = Some(Ok(real::undecided()));
                }
            }
            return out;
        };
        if n[0] {
            let v = if cos {
                atan2(&d, &fx, w)
            } else {
                atan2(&fx, &d, w)
            };
            out[0] = Some(v.map_or(Ok(real::undecided()), |v| step(&v)));
        }
        if n[1] {
            let mut l = l;
            l.negative = ny != cos;
            out[1] = Some(step(&l));
        }
        out
    })
}

/// `asin(x + yi)`.
fn asin(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round)) -> Parts {
    let (nx, ny) = (x.is_sign_negative(), y.is_sign_negative());
    if x.is_nan() || y.is_nan() {
        if x.is_infinite() || y.is_infinite() {
            return (exactly(nan(p.0)), exactly(inf(p.1, ny)));
        }
        if x.is_zero() {
            return (set(x, p.0, r.0), exactly(nan(p.1)));
        }
        return nans(p);
    }
    if x.is_infinite() || y.is_infinite() {
        let re = if x.is_infinite() {
            pi_part(if y.is_infinite() { 1 } else { 2 }, nx, p.0, r.0)
        } else {
            exactly(zero_of(p.0, nx))
        };
        return (re, exactly(inf(p.1, ny)));
    }
    if y.is_zero() {
        let ax = x.abs();
        if ax > 1 {
            let im = signed(ny, r.1, |m| ax.acosh_round(p.1, m));
            return (pi_part(2, nx, p.0, r.0), im);
        }
        return (x.asin_round(p.0, r.0), exactly(zero_of(p.1, ny)));
    }
    if x.is_zero() {
        return (exactly(zero_of(p.0, nx)), y.asinh_round(p.1, r.1));
    }
    asin_acos(x, y, p, r, false)
}

/// `acos(x + yi)`.
fn acos(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round)) -> Parts {
    let (nx, ny) = (x.is_sign_negative(), y.is_sign_negative());
    if x.is_nan() || y.is_nan() {
        if x.is_infinite() || y.is_infinite() {
            return (exactly(nan(p.0)), exactly(inf(p.1, !ny)));
        }
        if x.is_zero() {
            return (pi_part(2, false, p.0, r.0), exactly(nan(p.1)));
        }
        return nans(p);
    }
    if x.is_infinite() || y.is_infinite() {
        let re = if x.is_infinite() {
            match (y.is_infinite(), nx) {
                (true, false) => pi_part(1, false, p.0, r.0),
                (true, true) => pi_part(3, false, p.0, r.0),
                (false, false) => exactly(zero_of(p.0, false)),
                (false, true) => pi_part(4, false, p.0, r.0),
            }
        } else {
            pi_part(2, false, p.0, r.0)
        };
        return (re, exactly(inf(p.1, !ny)));
    }
    if y.is_zero() {
        let ax = x.abs();
        if ax > 1 {
            let im = signed(!ny, r.1, |m| ax.acosh_round(p.1, m));
            let re = if nx {
                pi_part(4, false, p.0, r.0)
            } else {
                exactly(zero_of(p.0, false))
            };
            return (re, im);
        }
        return (x.acos_round(p.0, r.0), exactly(zero_of(p.1, !ny)));
    }
    if x.is_zero() {
        let im = signed(!ny, r.1, |m| y.abs().asinh_round(p.1, m));
        return (pi_part(2, false, p.0, r.0), im);
    }
    asin_acos(x, y, p, r, true)
}

/// `atan(x + yi)`.
fn atan(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round)) -> Parts {
    let (nx, ny) = (x.is_sign_negative(), y.is_sign_negative());
    if x.is_nan() || y.is_nan() {
        if x.is_nan() {
            let im = if y.is_zero() || y.is_infinite() {
                exactly(zero_of(p.1, ny))
            } else {
                exactly(nan(p.1))
            };
            return (exactly(nan(p.0)), im);
        }
        if x.is_infinite() {
            return (pi_part(2, nx, p.0, r.0), exactly(zero_of(p.1, false)));
        }
        return nans(p);
    }
    if x.is_infinite() || y.is_infinite() {
        return (pi_part(2, nx, p.0, r.0), exactly(zero_of(p.1, ny)));
    }
    if y.is_zero() {
        return (x.atan_round(p.0, r.0), exactly(zero_of(p.1, ny)));
    }
    let ay = y.abs();
    let one = BigFloat::from(1);
    if x.is_zero() {
        match ay.partial_cmp(&one).expect("finite") {
            Ordering::Less => return (exactly(zero_of(p.0, nx)), y.atanh_round(p.1, r.1)),
            Ordering::Equal => return (exactly(zero_of(p.0, nx)), exactly(inf(p.1, ny))),
            Ordering::Greater => {}
        }
    }
    let fx = if x.is_zero() { real::zero() } else { of(x) };
    let fy = of(y);
    let (ax, ayf) = (abs(fx.clone()), abs(fy.clone()));
    let (mut re, mut im): (Option<Part>, Option<Part>) = (
        if x.is_zero() {
            Some(pi_part(2, nx, p.0, r.0))
        } else {
            odd_tiny(&fx, &fy, -2, false, p.0, r.0)
        },
        if x.is_zero() {
            None
        } else {
            odd_tiny(&fx, &fy, -2, true, p.1, r.1)
        },
    );
    real::ziv2(p, r, |w, need| {
        let mut out: [Option<Step>; 2] = [None, None];
        if need[0] {
            out[0] = Some(match re.take() {
                Some(v) => Err(v),
                None => {
                    // `atan2(2x, 1 - x² - y²)/2`.
                    let den = real::neg(norm_m1(&ax, &ayf, w + 16));
                    match atan2(&shl(fx.clone(), 1), &den, w + 8) {
                        Some(a) => step(&shl(a, -1)),
                        None => Ok(real::undecided()),
                    }
                }
            });
        }
        if need[1] {
            out[1] = Some(match im.take() {
                Some(v) => Err(v),
                None => {
                    let w2 = w + 16;
                    let ym1 = xadd(&ayf, &real::neg(real::one()), w2);
                    let den = add(&sqr(&ax, w2), &sqr(&ym1, w2), w2);
                    let u = div(&shl(ayf.clone(), 2), &den, w2);
                    match u.and_then(|u| ln1p(&u, w2)) {
                        Some(l) => {
                            let mut l = shl(l, -2);
                            l.negative = ny;
                            step(&l)
                        }
                        None => Ok(real::undecided()),
                    }
                }
            });
        }
        out
    })
}

/// `-i·f(i·z)` for `f` of `asin`, `atan`: `f(-y + xi)` with the precisions and the modes
/// swapped (the real part's mode for the result's imaginary part turned over), then `(b, -a)`.
fn minus_i(f: Fun, z: &BigComplex, p: (u64, u64), r: (Round, Round)) -> Parts {
    let (a, b) = f(&-&z.im, &z.re, (p.1, p.0), (round_for(r.1, true), r.0));
    (b, negated(a))
}

/// `acosh z = ±i acos z`, the sign of `y`'s, as MPC's `mpc_acosh`.
fn acosh(z: &BigComplex, p: (u64, u64), r: (Round, Round)) -> Parts {
    if z.re.is_zero() && z.im.is_nan() {
        return nans(p);
    }
    if z.im.is_sign_negative() {
        // `-i(u + vi) = v - ui`.
        let (u, v) = acos(&z.re, &z.im, (p.1, p.0), (round_for(r.1, true), r.0));
        (v, negated(u))
    } else {
        // `i(u + vi) = -v + ui`.
        let (u, v) = acos(&z.re, &z.im, (p.1, p.0), (r.1, round_for(r.0, true)));
        (negated(v), u)
    }
}

macro_rules! function {
    (
        $name:ident, $round:ident, $parts:ident, $imp:expr,
        $doc:literal, $example:literal, $parts_example:literal, $plain_example:literal
    ) => {
        impl BigComplex {
            #[doc = $doc]
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::float::{BigComplex, BigFloat, Round};
            /// use core::cmp::Ordering;
            ///
            #[doc = $example]
            /// ```
            ///
            /// # Panics
            ///
            /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
            #[track_caller]
            pub fn $round(&self, prec: u64, round: Round) -> (BigComplex, (Ordering, Ordering)) {
                self.$parts((prec, prec), (round, round))
            }

            #[doc = concat!("[`", stringify!($round), "`](BigComplex::", stringify!($round), ") with a precision and a mode for each part.")]
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::float::{BigComplex, BigFloat, Round};
            /// use core::cmp::Ordering;
            ///
            #[doc = $parts_example]
            /// ```
            #[track_caller]
            pub fn $parts(
                &self,
                prec: (u64, u64),
                round: (Round, Round),
            ) -> (BigComplex, (Ordering, Ordering)) {
                let f: fn(&BigComplex, (u64, u64), (Round, Round)) -> Parts = $imp;
                pair(f(self, check(prec), round))
            }

            #[doc = concat!("[`", stringify!($round), "`](BigComplex::", stringify!($round), ") to nearest at the precisions of the parts.")]
            ///
            /// # Examples
            ///
            /// ```
            /// use a::math::float::{BigComplex, BigFloat};
            ///
            #[doc = $plain_example]
            /// ```
            pub fn $name(&self) -> BigComplex {
                self.$parts(self.prec(), (Round::Nearest, Round::Nearest)).0
            }
        }
    };
}

function!(
    asin,
    asin_round,
    asin_round_parts,
    |z, p, r| asin(&z.re, &z.im, p, r),
    "The arcsine, principal branch, each part rounded to `prec` bits in the mode `round`, and the \
     directions of the roundings, as MPC's `mpc_asin`: the real part in `[-π/2, π/2]`, the \
     imaginary part of the sign of `self`'s, with the branch cuts along the real axis beyond `±1`, \
     where the sign of a zero imaginary part picks the side: `asin(2 ± 0i) = π/2 ± i acosh 2`. The \
     special values: `asin(±∞ + yi) = ±π/2 ± ∞i` for a finite `y` and `±π/4 ± ∞i` for an \
     infinite one, `asin(x ± ∞i) = ±0 ± ∞i` for a finite `x`; `NaN ± ∞i` for a NaN with an \
     infinity; `±0 + NaN·i` for `±0 + NaN·i`; NaN in both parts for the other numbers with a NaN.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (a, o) = z.asin_round(53, Round::Nearest);\n\
     assert_eq!(*a.re(), BigFloat::from(0.6662394324925153));\n\
     assert_eq!(*a.im(), BigFloat::from(1.0612750619050357));",
    "let z = BigComplex::new(BigFloat::from(2.0), BigFloat::neg_zero(53));\n\
     let (a, o) = z.asin_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*a.im(), -BigFloat::from(2.0).acosh());",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(1));\n\
     assert_eq!(*z.asin().im(), BigFloat::from(1).asinh());"
);

function!(
    acos,
    acos_round,
    acos_round_parts,
    |z, p, r| acos(&z.re, &z.im, p, r),
    "The arccosine, principal branch, each part rounded to `prec` bits in the mode `round`, and \
     the directions of the roundings, as MPC's `mpc_acos`: `π/2 - asin z`, the real part in `[0, \
     π]`, the imaginary part of the sign opposite to `self`'s, with the branch cuts of `asin`. The \
     special values: `acos(+∞ + yi) = +0 ∓ ∞i`, `acos(-∞ + yi) = π ∓ ∞i` for a finite `y`, `π/4 \
     ∓ ∞i` and `3π/4 ∓ ∞i` for an infinite one; `acos(x ± ∞i) = π/2 ∓ ∞i`; `NaN ∓ ∞i` for a NaN \
     with an infinity; `π/2 + NaN·i` for `±0 + NaN·i`; NaN in both parts for the other numbers \
     with a NaN.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (a, o) = z.acos_round(53, Round::Nearest);\n\
     assert_eq!(*a.re(), BigFloat::from(0.9045568943023814));\n\
     assert_eq!(*a.im(), BigFloat::from(-1.0612750619050357));",
    "let z = BigComplex::new(BigFloat::from(0.5), BigFloat::zero(53));\n\
     let (a, o) = z.acos_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*a.re(), BigFloat::from(0.5).acos());\n\
     assert_eq!(*a.im(), BigFloat::neg_zero(53));",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::zero(53));\n\
     assert!(z.acos().re().is_zero());"
);

function!(
    atan,
    atan_round,
    atan_round_parts,
    |z, p, r| atan(&z.re, &z.im, p, r),
    "The arctangent, principal branch, each part rounded to `prec` bits in the mode `round`, and \
     the directions of the roundings, as MPC's `mpc_atan`: the real part in `[-π/2, π/2]`, with \
     the branch cuts along the imaginary axis beyond `±i`, where the sign of a zero real part \
     picks the side: `atan(±0 + 2i) = ±π/2 + i atanh(1/2)`; `atan(±0 ± i) = ±0 ± ∞i`. The special \
     values: a number with an infinite part goes to `±π/2 ± 0i` (the signs of the parts'); `NaN \
     ± 0i` for `NaN + yi` with `y` zero or infinite, `±π/2 + 0i` for `±∞ + NaN·i`; NaN in both \
     parts for the other numbers with a NaN.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (a, o) = z.atan_round(53, Round::Nearest);\n\
     assert_eq!(*a.re(), BigFloat::from(1.0172219678978514));\n\
     assert_eq!(*a.im(), BigFloat::from(0.40235947810852507));",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(0.5));\n\
     let (a, o) = z.atan_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*a.im(), BigFloat::from(0.5).atanh());\n\
     assert_eq!(o.0, Ordering::Equal);",
    "let z = BigComplex::new(BigFloat::from(1.0), BigFloat::zero(53));\n\
     assert_eq!(*z.atan().re(), BigFloat::from(core::f64::consts::FRAC_PI_4));"
);

function!(
    asinh,
    asinh_round,
    asinh_round_parts,
    |z, p, r| minus_i(asin, z, p, r),
    "The inverse hyperbolic sine, principal branch, each part rounded to `prec` bits in the mode \
     `round`, and the directions of the roundings, as MPC's `mpc_asinh`: `-i asin(iz)`, and so its \
     branch cuts (along the imaginary axis beyond `±i`) and its special values.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (a, o) = z.asinh_round(53, Round::Nearest);\n\
     assert_eq!(*a.re(), BigFloat::from(1.0612750619050357));\n\
     assert_eq!(*a.im(), BigFloat::from(0.6662394324925153));",
    "let z = BigComplex::new(BigFloat::from(2.0), BigFloat::zero(53));\n\
     let (a, o) = z.asinh_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*a.re(), BigFloat::from(2.0).asinh());\n\
     assert_eq!(o.1, Ordering::Equal);",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(0.5));\n\
     assert_eq!(*z.asinh().im(), BigFloat::from(0.5).asin());"
);

function!(
    acosh,
    acosh_round,
    acosh_round_parts,
    acosh,
    "The inverse hyperbolic cosine, principal branch, each part rounded to `prec` bits in the mode \
     `round`, and the directions of the roundings, as MPC's `mpc_acosh`: `i acos z` for `self`'s \
     imaginary part not below zero (its sign bit clear) and `-i acos z` otherwise, the real part \
     not below zero; the special values follow, but for `acosh(±0 + NaN·i)`, NaN in both parts.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (a, o) = z.acosh_round(53, Round::Nearest);\n\
     assert_eq!(*a.re(), BigFloat::from(1.0612750619050357));\n\
     assert_eq!(*a.im(), BigFloat::from(0.9045568943023814));",
    "let z = BigComplex::new(BigFloat::from(2.0), BigFloat::zero(53));\n\
     let (a, o) = z.acosh_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*a.re(), BigFloat::from(2.0).acosh());\n\
     assert_eq!(o.1, Ordering::Equal);",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::zero(53));\n\
     assert!(z.acosh().re().is_zero());"
);

function!(
    atanh,
    atanh_round,
    atanh_round_parts,
    |z, p, r| minus_i(atan, z, p, r),
    "The inverse hyperbolic tangent, principal branch, each part rounded to `prec` bits in the \
     mode `round`, and the directions of the roundings, as MPC's `mpc_atanh`: `-i atan(iz)`, and \
     so its branch cuts (along the real axis beyond `±1`) and its special values, `atanh(±1 ± 0i) \
     = ±∞ ± 0i` among them.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (a, o) = z.atanh_round(53, Round::Nearest);\n\
     assert_eq!(*a.re(), BigFloat::from(0.40235947810852507));\n\
     assert_eq!(*a.im(), BigFloat::from(1.0172219678978514));",
    "let z = BigComplex::new(BigFloat::from(0.5), BigFloat::zero(53));\n\
     let (a, o) = z.atanh_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*a.re(), BigFloat::from(0.5).atanh());\n\
     assert_eq!(o.1, Ordering::Equal);",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::zero(53));\n\
     assert!(z.atanh().re().is_infinite());"
);
