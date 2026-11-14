//! The sine, the cosine and the tangent, and the hyperbolic functions through them, each part
//! correctly rounded, with MPC's special values (`mpc_sin`, `mpc_cos`, `mpc_sin_cos`, `mpc_tan`,
//! `mpc_sinh`, `mpc_cosh`, `mpc_tanh`).
//!
//! For `z = x + yi`:
//!
//! - `sin z = sin x cosh y + i cos x sinh y` and `cos z = cos x cosh y - i sin x sinh y`, products
//!   of the real functions of [`real`](super::real), each within a few units of `2^-w`
//!   relatively, so that nothing cancels; a part beyond the exponent range overflows by its mode.
//! - `tan z = (sin x cos x + i sinh y cosh y)/(cos² x + sinh² y)`, the denominator `(cos 2x + cosh
//!   2y)/2` written as a sum of squares, which cancels nothing. For a large `|y|` the imaginary part
//!   lies near `±1`: `|Im| - 1 = -(cos 2x + e^(-2|y|))/(2(cos² x + sinh² y))`, computed on its own
//!   and rounded beside `±1`; beyond the range of the exponential (`|y| >= 2^62`) that difference
//!   is below any precision, of the sign of `-cos 2x`, and the real part underflows.
//! - For `|x|, |y| < 2^-(p/2 + 4)`, `p` a part's precision, the parts lie within less than their
//!   last place of their first terms (`x`, `y`, `1`, `-xy`), and each difference from it is
//!   computed on its own from `sin x - x`, `sinh y - y`, `cos x - 1 = -2 sin²(x/2)`, `cosh y - 1 = 2
//!   sinh²(y/2)`, `sin x - x cos x` and `y cosh y - sinh y` (their series), and rounded beside it.
//! - `sinh z = swap(sin(swap z))`, `tanh z = swap(tan(swap z))` (`swap(a + bi) = b + ai`, the
//!   precisions and the modes swapped with the parts) and `cosh z = cos(-y + xi)`, as MPC.
//!
//! No part is exact for `x` and `y` not zero (the Lindemann–Weierstrass theorem).

use super::BigComplex;
use super::real::{
    self, Part, Step, add, check, div, exactly, inf, lead, mul, nan, nans, neg, odd_series, of,
    one, pair, range_step, set, shl, sign_known, sin_cos, sinh_cosh, sqr, step, sub, top, zero_of,
};
use crate::math::float::exp::Range;
use crate::math::float::sincos::Fx;
use crate::math::float::trig::near;
use crate::math::float::{BigFloat, Round};
use core::cell::RefCell;
use core::cmp::Ordering;

/// The parts of a function of `x + yi` for precisions and modes.
type Parts = (Part, Part);

/// A function of the parts of `x + yi`.
type Fun = fn(&BigFloat, &BigFloat, (u64, u64), (Round, Round)) -> Parts;

/// A finite float that is not zero.
#[inline]
fn finite(x: &BigFloat) -> bool {
    x.is_finite() && !x.is_zero()
}

/// The sign of the sine and of the cosine of a finite `x` (neither is zero unless `x` is).
fn signs(x: &BigFloat) -> (bool, bool) {
    let ((s, _), (c, _)) = x.sin_cos_round(2, Round::Nearest);
    (s.is_sign_negative(), c.is_sign_negative())
}

/// Whether both parts are below `2^-(p/2 + 4)`: then the parts lie near their first terms.
fn tiny(x: &Fx, y: &Fx, p: u64) -> bool {
    let t = -i128::from(p) - 8;
    2 * top(x) < t && 2 * top(y) < t
}

/// `sin(x + yi)`.
pub(super) fn sin(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round)) -> Parts {
    sin_with(x, y, p, r, &Kernels::default())
}

/// The real functions of `x` and `y` at the working precisions computed so far, which
/// [`BigComplex::sin_cos_round_parts`] shares between the sine and the cosine.
#[derive(Default)]
struct Kernels(RefCell<Vec<Kernel>>);

/// `w`, `sin x`, `cos x` and `sinh y`, `cosh y` (or where they lie beyond the range).
type Kernel = (u64, (Fx, Fx), Result<(Fx, Fx), Range>);

impl Kernels {
    /// The functions at `w`, computed unless they were.
    fn at(&self, fx: &Fx, fy: &Fx, w: u64) -> ((Fx, Fx), Result<(Fx, Fx), Range>) {
        let mut v = self.0.borrow_mut();
        if let Some(k) = v.iter().find(|k| k.0 == w) {
            return (k.1.clone(), k.2.clone());
        }
        let k = (w, sin_cos(fx, w), sinh_cosh(fy, w));
        let out = (k.1.clone(), k.2.clone());
        v.push(k);
        out
    }
}

/// `sin(x + yi)`, sharing the real functions in `kernels`.
fn sin_with(
    x: &BigFloat,
    y: &BigFloat,
    p: (u64, u64),
    r: (Round, Round),
    kernels: &Kernels,
) -> Parts {
    if !x.is_finite() || !y.is_finite() {
        if x.is_nan() || y.is_nan() {
            if y.is_nan() {
                return if x.is_zero() {
                    (set(x, p.0, r.0), exactly(nan(p.1)))
                } else {
                    nans(p)
                };
            }
            return if y.is_infinite() || y.is_zero() {
                (exactly(nan(p.0)), set(y, p.1, r.1))
            } else {
                nans(p)
            };
        }
        if x.is_infinite() {
            return if finite(y) {
                nans(p)
            } else {
                (exactly(nan(p.0)), set(y, p.1, r.1))
            };
        }
        // `x` finite, `y` infinite.
        if x.is_zero() {
            return (set(x, p.0, r.0), set(y, p.1, r.1));
        }
        let (sn, cn) = signs(x);
        return (
            exactly(inf(p.0, sn)),
            exactly(inf(p.1, cn != y.is_sign_negative())),
        );
    }
    if y.is_zero() {
        let (_, cn) = signs(x);
        let cn = cn && !x.is_zero();
        return (
            x.sin_round(p.0, r.0),
            exactly(zero_of(p.1, y.is_sign_negative() != cn)),
        );
    }
    if x.is_zero() {
        return (set(x, p.0, r.0), y.sinh_round(p.1, r.1));
    }
    let (fx, fy) = (of(x), of(y));
    real::ziv2(p, r, |w, need| {
        let mut out: [Option<Step>; 2] = [None, None];
        let (tr, ti) = (tiny(&fx, &fy, p.0), tiny(&fx, &fy, p.1));
        let ((s, c), hyp) = kernels.at(&fx, &fy, w);
        let Ok((sh, ch)) = hyp else {
            // `|y| >= 2^62`: `cosh y`, `|sinh y|` beyond the range, the signs of `sin x`, `cos x`
            // and `y`'s.
            for (i, (f, n)) in [(&s, false), (&c, fy.negative)].into_iter().enumerate() {
                if need[i] && sign_known(f) {
                    out[i] = Some(range_step(
                        Range::Over,
                        f.negative != n,
                        [p.0, p.1][i],
                        [r.0, r.1][i],
                    ));
                }
            }
            return out;
        };
        if need[0] {
            out[0] = Some(if tr {
                // `x + (sin x - x) cosh y + x (cosh y - 1)`.
                let chm = coshm1(&fy, w);
                let d = add(
                    &mul(&odd_series(&fx, w, true, false), &ch, w),
                    &mul(&fx, &chm, w),
                    w,
                );
                lead(&fx, &d, w, p.0, r.0)
            } else {
                step(&mul(&s, &ch, w))
            });
        }
        if need[1] {
            out[1] = Some(if ti {
                // `y + (cos x - 1) sinh y + (sinh y - y)`.
                let cm = cosm1(&fx, w);
                let d = add(&mul(&cm, &sh, w), &odd_series(&fy, w, false, false), w);
                lead(&fy, &d, w, p.1, r.1)
            } else {
                step(&mul(&c, &sh, w))
            });
        }
        out
    })
}

/// `cos x - 1 = -2 sin²(x/2)`.
fn cosm1(x: &Fx, w: u64) -> Fx {
    let (h, _) = sin_cos(&shl(x.clone(), -1), w);
    neg(shl(sqr(&h, w), 1))
}

/// `cosh y - 1 = 2 sinh²(y/2)`.
fn coshm1(y: &Fx, w: u64) -> Fx {
    let (h, _) = sinh_cosh(&shl(y.clone(), -1), w).expect("a tiny argument");
    shl(sqr(&h, w), 1)
}

/// `cos(x + yi)`.
pub(super) fn cos(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round)) -> Parts {
    cos_with(x, y, p, r, &Kernels::default())
}

/// `cos(x + yi)`, sharing the real functions in `kernels`.
fn cos_with(
    x: &BigFloat,
    y: &BigFloat,
    p: (u64, u64),
    r: (Round, Round),
    kernels: &Kernels,
) -> Parts {
    let same = x.is_sign_negative() == y.is_sign_negative();
    if !x.is_finite() || !y.is_finite() {
        if x.is_nan() {
            let re = if y.is_infinite() {
                inf(p.0, false)
            } else {
                nan(p.0)
            };
            let im = if y.is_zero() {
                set(y, p.1, r.1)
            } else {
                exactly(nan(p.1))
            };
            return (exactly(re), im);
        }
        if y.is_nan() {
            let im = if x.is_zero() {
                set(x, p.1, r.1)
            } else {
                exactly(nan(p.1))
            };
            return (exactly(nan(p.0)), im);
        }
        if x.is_infinite() {
            let re = if y.is_infinite() {
                inf(p.0, same)
            } else {
                nan(p.0)
            };
            let im = if y.is_zero() {
                zero_of(p.1, same)
            } else {
                nan(p.1)
            };
            return (exactly(re), exactly(im));
        }
        // `x` finite, `y` infinite.
        if x.is_zero() {
            return (exactly(inf(p.0, false)), exactly(zero_of(p.1, same)));
        }
        let (sn, cn) = signs(x);
        return (
            exactly(inf(p.0, cn)),
            exactly(inf(p.1, y.is_sign_negative() == sn)),
        );
    }
    if y.is_zero() {
        let sn = x.is_sign_negative();
        let (sn, _) = if x.is_zero() { (sn, false) } else { signs(x) };
        return (
            x.cos_round(p.0, r.0),
            exactly(zero_of(p.1, y.is_sign_negative() == sn)),
        );
    }
    if x.is_zero() {
        return (y.cosh_round(p.0, r.0), exactly(zero_of(p.1, same)));
    }
    let (fx, fy) = (of(x), of(y));
    real::ziv2(p, r, |w, need| {
        let mut out: [Option<Step>; 2] = [None, None];
        let (tr, ti) = (tiny(&fx, &fy, p.0), tiny(&fx, &fy, p.1));
        let ((s, c), hyp) = kernels.at(&fx, &fy, w);
        let Ok((sh, ch)) = hyp else {
            for (i, (f, n)) in [(&c, false), (&s, !fy.negative)].into_iter().enumerate() {
                if need[i] && sign_known(f) {
                    out[i] = Some(range_step(
                        Range::Over,
                        f.negative != n,
                        [p.0, p.1][i],
                        [r.0, r.1][i],
                    ));
                }
            }
            return out;
        };
        if need[0] {
            out[0] = Some(if tr {
                // `1 + (cos x - 1) cosh y + (cosh y - 1)`.
                let d = add(&mul(&cosm1(&fx, w), &ch, w), &coshm1(&fy, w), w);
                lead(&one(), &d, w, p.0, r.0)
            } else {
                step(&mul(&c, &ch, w))
            });
        }
        if need[1] {
            out[1] = Some(if ti {
                // `-xy - (sin x - x) sinh y - x (sinh y - y)`.
                let l = neg(real::mul_exact(&fx, &fy));
                let d = neg(add(
                    &mul(&odd_series(&fx, w, true, false), &sh, w),
                    &mul(&fx, &odd_series(&fy, w, false, false), w),
                    w,
                ));
                lead(&l, &d, w, p.1, r.1)
            } else {
                step(&neg(mul(&s, &sh, w)))
            });
        }
        out
    })
}

/// `tan(x + yi)`.
pub(super) fn tan(x: &BigFloat, y: &BigFloat, p: (u64, u64), r: (Round, Round)) -> Parts {
    let unit = |negative: bool| set(&BigFloat::from(if negative { -1 } else { 1 }), p.1, r.1);
    if !x.is_finite() || !y.is_finite() {
        if x.is_nan() {
            return if y.is_infinite() {
                (exactly(zero_of(p.0, false)), unit(y.is_sign_negative()))
            } else {
                nans(p)
            };
        }
        if y.is_nan() {
            return if x.is_zero() {
                (set(x, p.0, r.0), exactly(nan(p.1)))
            } else {
                nans(p)
            };
        }
        if x.is_infinite() {
            return if y.is_infinite() {
                (
                    exactly(zero_of(p.0, x.is_sign_negative())),
                    unit(y.is_sign_negative()),
                )
            } else {
                nans(p)
            };
        }
        // `x` finite, `y` infinite: `+0·sin x cos x ± i`.
        let n = if x.is_zero() {
            x.is_sign_negative()
        } else {
            let (sn, cn) = signs(x);
            sn != cn
        };
        return (exactly(zero_of(p.0, n)), unit(y.is_sign_negative()));
    }
    if x.is_zero() {
        return (set(x, p.0, r.0), y.tanh_round(p.1, r.1));
    }
    if y.is_zero() {
        return (x.tan_round(p.0, r.0), set(y, p.1, r.1));
    }
    let (fx, fy) = (of(x), of(y));
    let ny = y.is_sign_negative();
    real::ziv2(p, r, |w, need| {
        let mut out: [Option<Step>; 2] = [None, None];
        let (s, c) = sin_cos(&fx, w);
        // `cos 2x`, when `Im` lies near `±1`.
        let cos2 = || sin_cos(&shl(fx.clone(), 1), w).1;
        let Ok((sh, ch)) = sinh_cosh(&fy, w) else {
            // `|y| >= 2^62`: the real part `2 sin 2x·e^(-2|y|)` about, below the range, of the
            // sign of `sin 2x`; `|Im| - 1` of the sign of `-cos 2x` and below `2^(2^62)`.
            let sc = mul(&s, &c, w);
            if need[0] && sign_known(&sc) {
                out[0] = Some(range_step(Range::Under, sc.negative, p.0, r.0));
            }
            let c2 = cos2();
            if need[1] && sign_known(&c2) {
                let away = c2.negative;
                out[1] = near(ny, &[1], 0, away, -(1 << 62), p.1, r.1).map(Err);
            }
            return out;
        };
        let d = add(&sqr(&c, w), &sqr(&sh, w), w);
        let tr = tiny(&fx, &fy, p.0);
        let ti = tiny(&fx, &fy, p.1);
        if need[0] {
            out[0] = Some(if tr {
                // `x + [cos x (sin x - x cos x) - x sinh² y]/D`.
                let t = neg(odd_series(&fx, w, true, true));
                let num = sub(&mul(&c, &t, w), &mul(&fx, &sqr(&sh, w), w), w);
                match div(&num, &d, w) {
                    Some(q) => lead(&fx, &q, w, p.0, r.0),
                    None => Ok(real::undecided()),
                }
            } else {
                match div(&mul(&s, &c, w), &d, w) {
                    Some(q) => step(&q),
                    None => Ok(real::undecided()),
                }
            });
        }
        if need[1] {
            let big = d.exponent() > i128::from(p.1) + 12;
            out[1] = Some(if ti {
                // `y + [cosh y (sinh y - y cosh y) + y sin² x]/D`.
                let t = neg(odd_series(&fy, w, false, true));
                let num = add(&mul(&ch, &t, w), &mul(&fy, &sqr(&s, w), w), w);
                match div(&num, &d, w) {
                    Some(q) => lead(&fy, &q, w, p.1, r.1),
                    None => Ok(real::undecided()),
                }
            } else if big {
                // `±1 ∓ (cos 2x + e^(-2|y|))/(2D)`.
                let e2 = match div(&one(), &sqr(&add(&ch, &real::abs(sh.clone()), w), w), w) {
                    Some(e) => e,
                    None => return [out[0].take(), Some(Ok(real::undecided()))],
                };
                let num = add(&cos2(), &e2, w);
                match div(&num, &shl(d.clone(), 1), w) {
                    Some(q) => {
                        let l = real::exact(ny, vec![1], 0);
                        let q = if ny { q } else { neg(q) };
                        lead(&l, &q, w, p.1, r.1)
                    }
                    None => Ok(real::undecided()),
                }
            } else {
                match div(&mul(&sh, &ch, w), &d, w) {
                    Some(q) => step(&q),
                    None => Ok(real::undecided()),
                }
            });
        }
        out
    })
}

/// The parts swapped, with their precisions and modes: `f(swap z)` swapped back.
fn swapped(f: Fun, z: &BigComplex, p: (u64, u64), r: (Round, Round)) -> Parts {
    let (a, b) = f(&z.im, &z.re, (p.1, p.0), (r.1, r.0));
    (b, a)
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
    sin,
    sin_round,
    sin_round_parts,
    |z, p, r| sin(&z.re, &z.im, p, r),
    "The sine, each part rounded to `prec` bits in the mode `round`, and the directions of the \
     roundings, as MPC's `mpc_sin`: `sin x cosh y + i cos x sinh y`. The special values: `sin(x ± \
     0i) = sin x ± 0i` (the zero's sign turned over where `cos x < 0`); `sin(±0 + yi) = ±0 + i \
     sinh y`; `sin(x ± ∞i) = ±∞ ± ∞i` with the signs of `sin x` and of `cos x` times `±` for a \
     finite `x` not zero; `sin(±∞ + yi)` is NaN with an imaginary part `y` for `y` zero or \
     infinite and NaN otherwise; a NaN part gives NaN parts, but for `sin(±0 + NaN·i) = ±0 + \
     NaN·i` and `sin(NaN + yi) = NaN + yi` for `y` zero or infinite.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (s, o) = z.sin_round(53, Round::Nearest);\n\
     assert_eq!(*s.re(), BigFloat::from(1.2984575814159773)); // sin 1 cosh 1\n\
     assert_eq!(*s.im(), BigFloat::from(0.6349639147847361)); // cos 1 sinh 1\n\
     assert_eq!(o, (Ordering::Greater, Ordering::Greater));",
    "let z = BigComplex::new(BigFloat::from(2), BigFloat::zero(53));\n\
     let (s, o) = z.sin_round_parts((10, 10), (Round::Down, Round::Up));\n\
     assert_eq!(*s.im(), BigFloat::neg_zero(10)); // cos 2 < 0\n\
     assert_eq!(o.1, Ordering::Equal);",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(1));\n\
     assert_eq!(*z.sin().im(), BigFloat::from(1).sinh());"
);

function!(
    cos,
    cos_round,
    cos_round_parts,
    |z, p, r| cos(&z.re, &z.im, p, r),
    "The cosine, each part rounded to `prec` bits in the mode `round`, and the directions of the \
     roundings, as MPC's `mpc_cos`: `cos x cosh y - i sin x sinh y`. The special values: `cos(x ± \
     0i) = cos x ∓ 0i` (the zero's sign turned over where `sin x < 0`); `cos(±0 + yi) = cosh y ∓ \
     0i` (`-0` for parts of one sign); `cos(x ± ∞i) = +∞·cos x ∓ ∞·sin x` for a finite `x` not \
     zero and `+∞ ∓ 0i` for `±0`; `cos(±∞ + yi)` is `∓∞ + NaN·i` for an infinite `y` (`-∞` for \
     infinities of one sign), `NaN ∓ 0i` for a zero `y` and NaN otherwise; a NaN part gives NaN \
     parts, but `cos(NaN ± ∞i) = +∞ + NaN·i`, `cos(NaN ± 0i) = NaN ± 0i` and `cos(±0 + NaN·i) = \
     NaN ± 0i`.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (c, o) = z.cos_round(53, Round::Nearest);\n\
     assert_eq!(*c.re(), BigFloat::from(0.833730025131149)); // cos 1 cosh 1\n\
     assert_eq!(*c.im(), BigFloat::from(-0.9888977057628651)); // -sin 1 sinh 1\n\
     assert_eq!(o.0, Ordering::Less);",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(2.0));\n\
     let (c, o) = z.cos_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*c.re(), BigFloat::from(2.0).cosh());\n\
     assert_eq!(*c.im(), BigFloat::neg_zero(53));\n\
     assert_eq!(o.1, Ordering::Equal);",
    "let z = BigComplex::from(0);\n\
     assert_eq!(z.cos(), BigComplex::new(BigFloat::from(1), BigFloat::neg_zero(32)));"
);

function!(
    tan,
    tan_round,
    tan_round_parts,
    |z, p, r| tan(&z.re, &z.im, p, r),
    "The tangent, each part rounded to `prec` bits in the mode `round`, and the directions of the \
     roundings, as MPC's `mpc_tan`: `(sin 2x + i sinh 2y)/(cos 2x + cosh 2y)`. The special values: \
     `tan(x ± 0i) = tan x ± 0i`, `tan(±0 + yi) = ±0 + i tanh y`; `tan(x ± ∞i) = ±0 ± 1i` for a \
     finite `x`, the zero of the sign of `sin x cos x`; `tan(±∞ ± ∞i) = ±0 ± 1i` and `tan(NaN ± \
     ∞i) = +0 ± 1i`; `tan(±0 + NaN·i) = ±0 + NaN·i`; NaN in both parts for the other numbers \
     with an infinite real part or a NaN.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (t, o) = z.tan_round(53, Round::Nearest);\n\
     assert_eq!(*t.re(), BigFloat::from(0.27175258531951174));\n\
     assert_eq!(*t.im(), BigFloat::from(1.0839233273386946));\n\
     assert_eq!(o, (Ordering::Greater, Ordering::Greater));",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(100));\n\
     let (t, o) = z.tan_round_parts((53, 53), (Round::Nearest, Round::TowardZero));\n\
     assert_eq!(*t.im(), 1); // just above 1, as cos 2 < 0\n\
     assert_eq!(o.1, Ordering::Less);",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(1));\n\
     assert_eq!(*z.tan().im(), BigFloat::from(1).tanh());"
);

function!(
    sinh,
    sinh_round,
    sinh_round_parts,
    |z, p, r| swapped(sin, z, p, r),
    "The hyperbolic sine, each part rounded to `prec` bits in the mode `round`, and the \
     directions of the roundings, as MPC's `mpc_sinh`: `sinh x cos y + i cosh x sin y`, the sine \
     of `y + xi` with its parts swapped, and so its special values.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (s, o) = z.sinh_round(53, Round::Nearest);\n\
     assert_eq!(*s.re(), BigFloat::from(0.6349639147847361));\n\
     assert_eq!(*s.im(), BigFloat::from(1.2984575814159773));\n\
     assert_eq!(o, (Ordering::Greater, Ordering::Greater));",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::zero(53));\n\
     let (s, o) = z.sinh_round_parts((24, 53), (Round::Up, Round::Nearest));\n\
     assert_eq!(*s.re(), BigFloat::from(1).sinh_round(24, Round::Up).0);\n\
     assert_eq!(o.0, Ordering::Greater);",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(1));\n\
     assert_eq!(*z.sinh().im(), BigFloat::from(1).sin());"
);

function!(
    cosh,
    cosh_round,
    cosh_round_parts,
    |z, p, r| cos(&-&z.im, &z.re, p, r),
    "The hyperbolic cosine, each part rounded to `prec` bits in the mode `round`, and the \
     directions of the roundings, as MPC's `mpc_cosh`: `cosh x cos y + i sinh x sin y`, the cosine \
     of `-y + xi`, and so its special values.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (c, o) = z.cosh_round(53, Round::Nearest);\n\
     assert_eq!(*c.re(), BigFloat::from(0.833730025131149));\n\
     assert_eq!(*c.im(), BigFloat::from(0.9888977057628651));\n\
     assert_eq!(o.0, Ordering::Less);",
    "let z = BigComplex::new(BigFloat::from(2.0), BigFloat::zero(53));\n\
     let (c, o) = z.cosh_round_parts((53, 53), (Round::Nearest, Round::Nearest));\n\
     assert_eq!(*c.re(), BigFloat::from(2.0).cosh());\n\
     assert_eq!(o.1, Ordering::Equal);",
    "let z = BigComplex::new(BigFloat::zero(53), BigFloat::from(1.0));\n\
     assert_eq!(*z.cosh().re(), BigFloat::from(1.0).cos());"
);

function!(
    tanh,
    tanh_round,
    tanh_round_parts,
    |z, p, r| swapped(tan, z, p, r),
    "The hyperbolic tangent, each part rounded to `prec` bits in the mode `round`, and the \
     directions of the roundings, as MPC's `mpc_tanh`: the tangent of `y + xi` with its parts \
     swapped, and so its special values.",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));\n\
     let (t, o) = z.tanh_round(53, Round::Nearest);\n\
     assert_eq!(*t.re(), BigFloat::from(1.0839233273386946));\n\
     assert_eq!(*t.im(), BigFloat::from(0.27175258531951174));\n\
     assert_eq!(o, (Ordering::Greater, Ordering::Greater));",
    "let z = BigComplex::new(BigFloat::from(100), BigFloat::from(1));\n\
     let (t, o) = z.tanh_round_parts((53, 53), (Round::TowardZero, Round::Nearest));\n\
     assert_eq!(o.0, Ordering::Less);",
    "let z = BigComplex::new(BigFloat::from(1), BigFloat::zero(53));\n\
     assert_eq!(*z.tanh().re(), BigFloat::from(1).tanh());"
);

impl BigComplex {
    /// The sine and the cosine, each part of each rounded to `prec` bits in the mode `round`,
    /// and the directions of the roundings, as MPC's `mpc_sin_cos`: [`sin_round`] and
    /// [`cos_round`] at once.
    ///
    /// [`sin_round`]: BigComplex::sin_round
    /// [`cos_round`]: BigComplex::cos_round
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(-2));
    /// let ((s, _), (c, _)) = z.sin_cos_round(64, Round::Nearest);
    /// assert_eq!(s, z.sin_round(64, Round::Nearest).0);
    /// assert_eq!(c, z.cos_round(64, Round::Nearest).0);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn sin_cos_round(
        &self,
        prec: u64,
        round: Round,
    ) -> (
        (BigComplex, (Ordering, Ordering)),
        (BigComplex, (Ordering, Ordering)),
    ) {
        self.sin_cos_round_parts((prec, prec), (round, round), (prec, prec), (round, round))
    }

    /// [`sin_cos_round`](BigComplex::sin_cos_round) with a precision and a mode for each part of
    /// the sine and of the cosine, MPC's `rop_sin`, `rnd_sin`, `rop_cos` and `rnd_cos`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(0.5), BigFloat::from(0.25));
    /// let r = (Round::Down, Round::Up);
    /// let ((s, os), (c, oc)) = z.sin_cos_round_parts((24, 30), r, (40, 10), r);
    /// assert_eq!((s, os), z.sin_round_parts((24, 30), r));
    /// assert_eq!((c, oc), z.cos_round_parts((40, 10), r));
    /// ```
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn sin_cos_round_parts(
        &self,
        sin_prec: (u64, u64),
        sin_round: (Round, Round),
        cos_prec: (u64, u64),
        cos_round: (Round, Round),
    ) -> (
        (BigComplex, (Ordering, Ordering)),
        (BigComplex, (Ordering, Ordering)),
    ) {
        let (sp, cp) = (check(sin_prec), check(cos_prec));
        let kernels = Kernels::default();
        (
            pair(sin_with(&self.re, &self.im, sp, sin_round, &kernels)),
            pair(cos_with(&self.re, &self.im, cp, cos_round, &kernels)),
        )
    }

    /// The sine and the cosine, to nearest at the precisions of the parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat};
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let (s, c) = z.sin_cos();
    /// assert_eq!((s, c), (z.sin(), z.cos()));
    /// ```
    pub fn sin_cos(&self) -> (BigComplex, BigComplex) {
        let p = self.prec();
        let n = (Round::Nearest, Round::Nearest);
        let ((s, _), (c, _)) = self.sin_cos_round_parts(p, n, p, n);
        (s, c)
    }
}
