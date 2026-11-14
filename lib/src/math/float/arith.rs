//! `+`, `-`, `*`, `/`, the fused multiply-add and the square root, each correctly rounded, with
//! the special values of IEEE 754: NaN for `∞ - ∞`, `0·∞`, `0/0`, `∞/∞` and the square root of a
//! number below zero; `x/0` an infinity of the sign of the quotient; and an exact zero sum `+0`,
//! or `-0` when rounding down, unless both terms are zeros of the same sign.
//!
//! Each operation first sorts out the special values, then takes one of three paths by the
//! lengths of its operands and of its result: one limb in registers, two limbs in registers (see
//! [`small`](super::small)), or any length on limbs (see [`general`](super::general)). The
//! results of the first two are rounded in registers and written into the result's vector; the
//! assigning operators and the `set_` methods write into the vector of the number they change.

use super::general::{self, Op};
use super::limbs;
use super::round::{Rounded, round_into, seal};
use super::small::{self, wide};
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The zero of an exact sum of zero, of terms of the signs `a` and `b`: their sign when they
/// agree, otherwise `-0` when rounding down and `+0` in the other modes.
#[inline]
fn zero_sum(a: bool, b: bool, round: Round) -> bool {
    if a == b { a } else { round == Round::Down }
}

/// A signed magnitude `±M·2^lsb`, `M` not zero and without zero limbs at the top.
#[derive(Clone, Copy)]
#[allow(dead_code, reason = "the interface of the other modules of floats")]
pub(crate) struct Mag<'a> {
    pub(crate) negative: bool,
    pub(crate) m: &'a [u64],
    pub(crate) lsb: i128,
}

impl BigFloat {
    /// The signed magnitude of a finite number that is not zero.
    #[inline]
    #[allow(dead_code, reason = "the interface of the other modules of floats")]
    pub(crate) fn mag(&self) -> Option<Mag<'_>> {
        self.parts()
            .map(|(negative, m, lsb)| Mag { negative, m, lsb })
    }

    /// The operand of a finite number that is not zero.
    #[inline]
    pub(crate) fn op(&self) -> Option<Op<'_>> {
        match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => Some(Op {
                negative: *negative,
                m: mant,
                exp: i128::from(*exp),
            }),
            _ => None,
        }
    }

    /// The vector of the mantissa, taken out of a finite number (which is left as NaN), or a new
    /// one: for a result written in place.
    #[inline]
    fn take_mant(&mut self) -> Vec<u64> {
        match core::mem::replace(&mut self.kind, Kind::Nan) {
            Kind::Finite { mant, .. } => mant,
            _ => Vec::new(),
        }
    }
}

/// What an operation does after the special values are sorted out.
enum Plan<'a> {
    /// The result, which needs no rounding.
    Done(Kind),
    /// A result rounded in registers; `None` for an exact zero sum.
    Small(Option<Rounded>),
    /// A result to compute on limbs.
    Big(Big<'a>),
}

/// The operations on limbs.
enum Big<'a> {
    /// The operand rounded to the precision, of its own sign.
    Round(Op<'a>),
    Add(Op<'a>, Op<'a>),
    Mul(bool, Op<'a>, Op<'a>),
    Div(bool, Op<'a>, Op<'a>),
    Sqrt(Op<'a>),
    Fma(bool, Op<'a>, Op<'a>, Op<'a>),
}

/// The result of a computation on limbs, into `out`.
fn run(big: Big<'_>, prec: u64, round: Round, out: Vec<u64>) -> (Kind, Ordering) {
    match big {
        Big::Round(x) => {
            let lsb = x.exp - 64 * x.m.len() as i128;
            round_into(x.negative, x.m, lsb, false, prec, round, out)
        }
        Big::Add(x, y) => general::add(x, y, prec, round, out),
        Big::Mul(negative, x, y) => {
            limbs::with_work(|w| general::mul(negative, x, y, prec, round, w, out))
        }
        Big::Div(negative, x, y) => {
            limbs::with_work(|w| general::div(negative, x, y, prec, round, w, out))
        }
        Big::Sqrt(x) => limbs::with_work(|w| general::sqrt(x, prec, round, w, out)),
        Big::Fma(negative, x, y, z) => {
            limbs::with_work(|w| general::fma(negative, x, y, z, prec, round, w, out))
        }
    }
}

/// The result of a plan, into `out`.
#[inline]
fn finish(plan: Plan<'_>, prec: u64, round: Round, out: Vec<u64>) -> (Kind, Ordering) {
    match plan {
        Plan::Done(kind) => (kind, Ordering::Equal),
        Plan::Small(Some(r)) => seal(r, prec, round, out),
        Plan::Small(None) => (
            Kind::Zero {
                negative: round == Round::Down,
            },
            Ordering::Equal,
        ),
        Plan::Big(big) => run(big, prec, round, out),
    }
}

/// `x + y` for finite operands that are not zero.
#[inline(always)]
fn add_plan<'a>(x: Op<'a>, y: Op<'a>, prec: u64, round: Round) -> Plan<'a> {
    match (x.m, y.m) {
        ([a], [b]) if prec <= 64 => Plan::Small(small::add1(
            x.negative, *a, x.exp, y.negative, *b, y.exp, prec, round,
        )),
        _ if x.m.len() <= 2 && y.m.len() <= 2 && prec <= 128 => Plan::Small(small::add2(
            x.negative,
            wide(x.m),
            x.exp,
            y.negative,
            wide(y.m),
            y.exp,
            prec,
            round,
        )),
        _ => Plan::Big(Big::Add(x, y)),
    }
}

/// `a ± b`, `b`'s sign turned over when `negate`.
#[inline]
fn add_kind<'a>(
    a: &'a BigFloat,
    b: &'a BigFloat,
    negate: bool,
    prec: u64,
    round: Round,
) -> Plan<'a> {
    match (&a.kind, &b.kind) {
        (Kind::Finite { .. }, Kind::Finite { .. }) => {
            let (x, mut y) = (a.op().expect("finite"), b.op().expect("finite"));
            y.negative ^= negate;
            add_plan(x, y, prec, round)
        }
        (Kind::Nan, _) | (_, Kind::Nan) => Plan::Done(Kind::Nan),
        (Kind::Inf { negative: x }, Kind::Inf { negative: y }) => {
            if *x == (*y != negate) {
                Plan::Done(Kind::Inf { negative: *x })
            } else {
                Plan::Done(Kind::Nan)
            }
        }
        (Kind::Inf { negative }, _) => Plan::Done(Kind::Inf {
            negative: *negative,
        }),
        (_, Kind::Inf { negative }) => Plan::Done(Kind::Inf {
            negative: *negative != negate,
        }),
        (Kind::Zero { negative: x }, Kind::Zero { negative: y }) => Plan::Done(Kind::Zero {
            negative: zero_sum(*x, *y != negate, round),
        }),
        (Kind::Zero { .. }, _) => {
            let mut y = b.op().expect("finite");
            y.negative ^= negate;
            Plan::Big(Big::Round(y))
        }
        (_, Kind::Zero { .. }) => Plan::Big(Big::Round(a.op().expect("finite"))),
    }
}

/// `x·y`, of the sign `negative`, for finite operands that are not zero.
#[inline(always)]
fn mul_plan<'a>(negative: bool, x: Op<'a>, y: Op<'a>, prec: u64, round: Round) -> Plan<'a> {
    match (x.m, y.m) {
        ([a], [b]) if prec <= 64 => Plan::Small(Some(small::mul1(
            negative, *a, x.exp, *b, y.exp, prec, round,
        ))),
        _ if x.m.len() <= 2 && y.m.len() <= 2 && prec <= 128 => Plan::Small(Some(small::mul2(
            negative,
            wide(x.m),
            x.exp,
            wide(y.m),
            y.exp,
            prec,
            round,
        ))),
        _ => Plan::Big(Big::Mul(negative, x, y)),
    }
}

/// `x·y` of two numbers that are not NaN, as a special value or a plan for finite operands.
enum Product<'a> {
    Nan,
    Inf(bool),
    Zero(bool),
    Finite(bool, Op<'a>, Op<'a>),
}

fn product<'a>(x: &'a BigFloat, y: &'a BigFloat) -> Product<'a> {
    let negative = x.is_sign_negative() != y.is_sign_negative();
    match (&x.kind, &y.kind) {
        (Kind::Nan, _) | (_, Kind::Nan) => Product::Nan,
        (Kind::Inf { .. }, Kind::Zero { .. }) | (Kind::Zero { .. }, Kind::Inf { .. }) => {
            Product::Nan
        }
        (Kind::Inf { .. }, _) | (_, Kind::Inf { .. }) => Product::Inf(negative),
        (Kind::Zero { .. }, _) | (_, Kind::Zero { .. }) => Product::Zero(negative),
        _ => Product::Finite(negative, x.op().expect("finite"), y.op().expect("finite")),
    }
}

#[inline]
fn mul_kind<'a>(a: &'a BigFloat, b: &'a BigFloat, prec: u64, round: Round) -> Plan<'a> {
    match product(a, b) {
        Product::Nan => Plan::Done(Kind::Nan),
        Product::Inf(negative) => Plan::Done(Kind::Inf { negative }),
        Product::Zero(negative) => Plan::Done(Kind::Zero { negative }),
        Product::Finite(negative, x, y) => mul_plan(negative, x, y, prec, round),
    }
}

#[inline]
fn div_kind<'a>(a: &'a BigFloat, b: &'a BigFloat, prec: u64, round: Round) -> Plan<'a> {
    let negative = a.is_sign_negative() != b.is_sign_negative();
    match (&a.kind, &b.kind) {
        (Kind::Finite { .. }, Kind::Finite { .. }) => {
            let (x, y) = (a.op().expect("finite"), b.op().expect("finite"));
            match (x.m, y.m) {
                ([p], [q]) if prec <= 64 => Plan::Small(Some(small::div1(
                    negative, *p, x.exp, *q, y.exp, prec, round,
                ))),
                _ if x.m.len() <= 2 && y.m.len() <= 2 && prec <= 128 => Plan::Small(Some(
                    small::div2(negative, wide(x.m), x.exp, wide(y.m), y.exp, prec, round),
                )),
                _ => Plan::Big(Big::Div(negative, x, y)),
            }
        }
        (Kind::Nan, _) | (_, Kind::Nan) => Plan::Done(Kind::Nan),
        (Kind::Inf { .. }, Kind::Inf { .. }) | (Kind::Zero { .. }, Kind::Zero { .. }) => {
            Plan::Done(Kind::Nan)
        }
        (Kind::Inf { .. }, _) | (_, Kind::Zero { .. }) => Plan::Done(Kind::Inf { negative }),
        (_, Kind::Inf { .. }) | (Kind::Zero { .. }, _) => Plan::Done(Kind::Zero { negative }),
    }
}

#[inline]
fn sqrt_kind(a: &BigFloat, prec: u64, round: Round) -> Plan<'_> {
    match &a.kind {
        Kind::Nan | Kind::Inf { negative: true } | Kind::Finite { negative: true, .. } => {
            Plan::Done(Kind::Nan)
        }
        Kind::Inf { negative: false } => Plan::Done(Kind::Inf { negative: false }),
        Kind::Zero { negative } => Plan::Done(Kind::Zero {
            negative: *negative,
        }),
        Kind::Finite { .. } => {
            let x = a.op().expect("finite");
            match x.m {
                [p] if prec <= 64 => Plan::Small(Some(small::sqrt1(*p, x.exp, prec, round))),
                _ if x.m.len() <= 2 && prec <= 127 => {
                    Plan::Small(Some(small::sqrt2(wide(x.m), x.exp, prec, round)))
                }
                [rest @ .., next, top] if prec <= 63 => Plan::Small(Some(small::sqrt_long(
                    *top,
                    *next,
                    rest.iter().any(|&v| v != 0),
                    x.exp,
                    prec,
                    round,
                ))),
                _ => Plan::Big(Big::Sqrt(x)),
            }
        }
    }
}

#[inline]
fn fma_kind<'a>(
    s: &'a BigFloat,
    a: &'a BigFloat,
    b: &'a BigFloat,
    prec: u64,
    round: Round,
) -> Plan<'a> {
    match (product(s, a), &b.kind) {
        (Product::Nan, _) | (_, Kind::Nan) => Plan::Done(Kind::Nan),
        (Product::Inf(x), Kind::Inf { negative: y }) => {
            if x == *y {
                Plan::Done(Kind::Inf { negative: x })
            } else {
                Plan::Done(Kind::Nan)
            }
        }
        (Product::Inf(negative), _) => Plan::Done(Kind::Inf { negative }),
        (_, Kind::Inf { negative }) => Plan::Done(Kind::Inf {
            negative: *negative,
        }),
        (Product::Zero(x), Kind::Zero { negative: y }) => Plan::Done(Kind::Zero {
            negative: zero_sum(x, *y, round),
        }),
        (Product::Zero(_), _) => Plan::Big(Big::Round(b.op().expect("finite"))),
        (Product::Finite(negative, x, y), Kind::Zero { .. }) => {
            mul_plan(negative, x, y, prec, round)
        }
        (Product::Finite(negative, x, y), _) => {
            let z = b.op().expect("finite");
            match (x.m, y.m) {
                ([p], [q]) if z.m.len() <= 2 && prec <= 128 => {
                    // The exact product of 128 bits, aligned at its top bit, then the sum.
                    let mut m = u128::from(*p) * u128::from(*q);
                    let mut e = x.exp + y.exp;
                    if m >> 127 == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    Plan::Small(small::add2(
                        negative,
                        m,
                        e,
                        z.negative,
                        wide(z.m),
                        z.exp,
                        prec,
                        round,
                    ))
                }
                _ => Plan::Big(Big::Fma(negative, x, y, z)),
            }
        }
    }
}

/// The result of `plan` at the precision `prec`, in a new vector.
#[inline]
fn result(plan: Plan<'_>, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let (kind, o) = finish(plan, prec, round, Vec::new());
    (BigFloat { prec, kind }, o)
}

/// An operation that assigns its result to its first operand.
#[derive(Clone, Copy)]
enum Assign {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
}

/// The plan of `x op y` (the square root of `x`).
#[inline(always)]
fn plan_of<'a>(op: Assign, x: &'a BigFloat, y: &'a BigFloat, prec: u64, round: Round) -> Plan<'a> {
    match op {
        Assign::Add => add_kind(x, y, false, prec, round),
        Assign::Sub => add_kind(x, y, true, prec, round),
        Assign::Mul => mul_kind(x, y, prec, round),
        Assign::Div => div_kind(x, y, prec, round),
        Assign::Sqrt => sqrt_kind(x, prec, round),
    }
}

impl BigFloat {
    /// Sets `self` to `self op other` at the precision `prec`: a result rounded in registers into
    /// the vector of `self`, one on limbs into the thread's spare vector, which then takes the old
    /// one.
    #[inline]
    fn assign_op(&mut self, op: Assign, other: &BigFloat, prec: u64, round: Round) -> Ordering {
        let small = match plan_of(op, self, other, prec, round) {
            Plan::Big(big) => {
                let (kind, o) = run(big, prec, round, limbs::take_spare());
                let old = core::mem::replace(&mut self.kind, kind);
                self.prec = prec;
                if let Kind::Finite { mant, .. } = old {
                    limbs::give_spare(mant);
                }
                return o;
            }
            Plan::Done(kind) => Err(kind),
            Plan::Small(r) => Ok(r),
        };
        self.prec = prec;
        match small {
            Err(kind) => {
                self.kind = kind;
                Ordering::Equal
            }
            Ok(Some(r)) => self.put(r, round),
            Ok(None) => {
                self.kind = Kind::Zero {
                    negative: round == Round::Down,
                };
                Ordering::Equal
            }
        }
    }

    /// Sets `self`, of its precision, to the result `r` rounded in registers: into its mantissa's
    /// limbs when it has as many, else as [`seal`] makes it.
    #[inline(always)]
    fn put(&mut self, r: Rounded, round: Round) -> Ordering {
        let prec = self.prec;
        let n = if prec <= 64 { 1 } else { 2 };
        let fits = (i128::from(BigFloat::EXP_MIN)..=i128::from(BigFloat::EXP_MAX)).contains(&r.exp);
        if fits
            && let Kind::Finite {
                negative,
                exp,
                mant,
            } = &mut self.kind
            && mant.len() == n
        {
            *negative = r.negative;
            *exp = r.exp as i64;
            if n == 1 {
                mant[0] = (r.m >> 64) as u64;
            } else {
                mant[0] = r.m as u64;
                mant[1] = (r.m >> 64) as u64;
            }
            return if r.negative {
                r.magnitude.reverse()
            } else {
                r.magnitude
            };
        }
        let out = self.take_mant();
        let (kind, o) = seal(r, prec, round, out);
        self.kind = kind;
        o
    }

    /// Sets `self` to the result of `plan` of other numbers, at the precision of `self`, written
    /// into the vector of `self`.
    #[inline]
    fn set_plan(&mut self, plan: Plan<'_>, round: Round) -> Ordering {
        let prec = self.prec;
        if let Plan::Small(Some(r)) = plan {
            return self.put(r, round);
        }
        let out = self.take_mant();
        let (kind, o) = finish(plan, prec, round, out);
        self.kind = kind;
        o
    }

    /// `self + other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(1).add_round(&BigFloat::from(0.25), 2, Round::Down);
    /// assert_eq!(x, BigFloat::from(1));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn add_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        result(add_kind(self, other, false, prec, round), prec, round)
    }

    /// `self - other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    #[track_caller]
    pub fn sub_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        result(add_kind(self, other, true, prec, round), prec, round)
    }

    /// `self · other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    #[track_caller]
    pub fn mul_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        result(mul_kind(self, other, prec, round), prec, round)
    }

    /// `self / other` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding; a number divided by zero is an infinity of the sign of the quotient, and `0/0`
    /// NaN.
    #[track_caller]
    pub fn div_round(&self, other: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        result(div_kind(self, other, prec, round), prec, round)
    }

    /// `self · a + b` rounded once to `prec` bits in the mode `round`, and the direction of the
    /// rounding, as IEEE 754's `fusedMultiplyAdd`.
    #[track_caller]
    pub fn mul_add_round(
        &self,
        a: &BigFloat,
        b: &BigFloat,
        prec: u64,
        round: Round,
    ) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        result(fma_kind(self, a, b, prec, round), prec, round)
    }

    /// `self · a + b` rounded once to nearest at the largest precision of the three, as
    /// `f64::mul_add`.
    pub fn mul_add(&self, a: &BigFloat, b: &BigFloat) -> BigFloat {
        let prec = self.prec.max(a.prec).max(b.prec);
        self.mul_add_round(a, b, prec, Round::Nearest).0
    }

    /// The square root rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding: NaN below zero, `-0` for `-0`, as IEEE 754 has it.
    #[track_caller]
    pub fn sqrt_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        result(sqrt_kind(self, prec, round), prec, round)
    }

    /// The square root, to nearest at the precision of `self`.
    pub fn sqrt(&self) -> BigFloat {
        self.sqrt_round(self.prec, Round::Nearest).0
    }

    /// Sets `self` to `a + b` rounded to the precision of `self` in the mode `round`, reusing its
    /// vector, and returns the direction of the rounding, as MPFR's `mpfr_add`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let mut r = BigFloat::zero(2);
    /// let o = r.set_add(&BigFloat::from(1), &BigFloat::from(0.25), Round::Up);
    /// assert_eq!(r, BigFloat::from(1.5));
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    pub fn set_add(&mut self, a: &BigFloat, b: &BigFloat, round: Round) -> Ordering {
        let plan = add_kind(a, b, false, self.prec, round);
        self.set_plan(plan, round)
    }

    /// Sets `self` to `a - b` rounded to the precision of `self` in the mode `round`, reusing its
    /// vector, and returns the direction of the rounding.
    pub fn set_sub(&mut self, a: &BigFloat, b: &BigFloat, round: Round) -> Ordering {
        let plan = add_kind(a, b, true, self.prec, round);
        self.set_plan(plan, round)
    }

    /// Sets `self` to `a · b` rounded to the precision of `self` in the mode `round`, reusing its
    /// vector, and returns the direction of the rounding.
    pub fn set_mul(&mut self, a: &BigFloat, b: &BigFloat, round: Round) -> Ordering {
        let plan = mul_kind(a, b, self.prec, round);
        self.set_plan(plan, round)
    }

    /// Sets `self` to `a / b` rounded to the precision of `self` in the mode `round`, reusing its
    /// vector, and returns the direction of the rounding.
    pub fn set_div(&mut self, a: &BigFloat, b: &BigFloat, round: Round) -> Ordering {
        let plan = div_kind(a, b, self.prec, round);
        self.set_plan(plan, round)
    }

    /// Sets `self` to `a · b + c` rounded once to the precision of `self` in the mode `round`,
    /// reusing its vector, and returns the direction of the rounding.
    pub fn set_mul_add(
        &mut self,
        a: &BigFloat,
        b: &BigFloat,
        c: &BigFloat,
        round: Round,
    ) -> Ordering {
        let plan = fma_kind(a, b, c, self.prec, round);
        self.set_plan(plan, round)
    }

    /// Sets `self` to the square root of `a` rounded to the precision of `self` in the mode
    /// `round`, reusing its vector, and returns the direction of the rounding.
    pub fn set_sqrt(&mut self, a: &BigFloat, round: Round) -> Ordering {
        let plan = sqrt_kind(a, self.prec, round);
        self.set_plan(plan, round)
    }

    /// Sets `self` to `self + other` rounded to the precision of `self` in the mode `round`, and
    /// returns the direction of the rounding, as MPFR's `mpfr_add (x, x, y, rnd)`.
    pub fn add_assign_round(&mut self, other: &BigFloat, round: Round) -> Ordering {
        let prec = self.prec;
        self.assign_op(Assign::Add, other, prec, round)
    }

    /// Sets `self` to `self - other` rounded to the precision of `self` in the mode `round`, and
    /// returns the direction of the rounding.
    pub fn sub_assign_round(&mut self, other: &BigFloat, round: Round) -> Ordering {
        let prec = self.prec;
        self.assign_op(Assign::Sub, other, prec, round)
    }

    /// Sets `self` to `self · other` rounded to the precision of `self` in the mode `round`, and
    /// returns the direction of the rounding.
    pub fn mul_assign_round(&mut self, other: &BigFloat, round: Round) -> Ordering {
        let prec = self.prec;
        self.assign_op(Assign::Mul, other, prec, round)
    }

    /// Sets `self` to `self / other` rounded to the precision of `self` in the mode `round`, and
    /// returns the direction of the rounding.
    pub fn div_assign_round(&mut self, other: &BigFloat, round: Round) -> Ordering {
        let prec = self.prec;
        self.assign_op(Assign::Div, other, prec, round)
    }

    /// Sets `self` to its square root rounded to the precision of `self` in the mode `round`, and
    /// returns the direction of the rounding.
    pub fn sqrt_assign_round(&mut self, round: Round) -> Ordering {
        let prec = self.prec;
        let x = BigFloat::nan(1);
        self.assign_op(Assign::Sqrt, &x, prec, round)
    }
}

/// `±(A·2^la) / (B·2^lb)` rounded, for magnitudes without zero limbs at the top.
pub(crate) fn div_mags(
    negative: bool,
    a: &[u64],
    la: i128,
    b: &[u64],
    lb: i128,
    prec: u64,
    round: Round,
) -> (BigFloat, Ordering) {
    let (x, y) = (Aligned::of(a, la), Aligned::of(b, lb));
    let (x, y) = (x.op(false), y.op(false));
    result(
        match (x.m, y.m) {
            ([p], [q]) if prec <= 64 => Plan::Small(Some(small::div1(
                negative, *p, x.exp, *q, y.exp, prec, round,
            ))),
            _ => Plan::Big(Big::Div(negative, x, y)),
        },
        prec,
        round,
    )
}

/// `sqrt(A·2^la)` rounded, for a magnitude without zero limbs at the top.
#[allow(dead_code, reason = "the interface of the other modules of floats")]
pub(crate) fn sqrt_mag(a: &[u64], la: i128, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let x = Aligned::of(a, la);
    result(Plan::Big(Big::Sqrt(x.op(false))), prec, round)
}

/// `a + b` for the signed magnitudes `a` and `b` rounded to `prec` in the mode `round`.
#[allow(dead_code, reason = "the interface of the other modules of floats")]
pub(crate) fn add_mags(a: Mag<'_>, b: Mag<'_>, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let (x, y) = (Aligned::of(a.m, a.lsb), Aligned::of(b.m, b.lsb));
    let (x, y) = (x.op(a.negative), y.op(b.negative));
    result(add_plan(x, y, prec, round), prec, round)
}

/// A magnitude aligned at its top bit, borrowed when it already is.
struct Aligned<'a> {
    m: std::borrow::Cow<'a, [u64]>,
    exp: i128,
}

impl<'a> Aligned<'a> {
    /// `m·2^lsb` for `m` not zero, without zero limbs at the top.
    fn of(m: &'a [u64], lsb: i128) -> Aligned<'a> {
        debug_assert!(m.last().is_some_and(|&t| t != 0));
        let lz = m[m.len() - 1].leading_zeros();
        let exp = lsb + 64 * m.len() as i128;
        if lz == 0 {
            Aligned {
                m: std::borrow::Cow::Borrowed(m),
                exp,
            }
        } else {
            let mut v = limbs::shl(m, u64::from(lz));
            v.truncate(m.len());
            Aligned {
                m: std::borrow::Cow::Owned(v),
                exp: exp - i128::from(lz),
            }
        }
    }

    fn op(&self, negative: bool) -> Op<'_> {
        Op {
            negative,
            m: &self.m,
            exp: self.exp,
        }
    }
}

/// The mode that rounds `-x` as `round` rounds `x` when `negate`: up and down swapped.
#[inline]
#[allow(dead_code, reason = "the interface of the other modules of floats")]
pub(crate) fn round_for(round: Round, negate: bool) -> Round {
    match (round, negate) {
        (Round::Up, true) => Round::Down,
        (Round::Down, true) => Round::Up,
        (r, _) => r,
    }
}

macro_rules! binary {
    ($Op:ident, $op:ident, $Assign:ident, $assign:ident, $kind:ident) => {
        impl $Op<&BigFloat> for &BigFloat {
            type Output = BigFloat;
            /// Rounded to nearest at the larger precision of the operands.
            #[inline]
            fn $op(self, rhs: &BigFloat) -> BigFloat {
                let prec = self.prec.max(rhs.prec);
                result(
                    plan_of(Assign::$kind, self, rhs, prec, Round::Nearest),
                    prec,
                    Round::Nearest,
                )
                .0
            }
        }
        impl $Op<BigFloat> for &BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(self, rhs: BigFloat) -> BigFloat {
                self.$op(&rhs)
            }
        }
        impl $Op<&BigFloat> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(mut self, rhs: &BigFloat) -> BigFloat {
                self.$assign(rhs);
                self
            }
        }
        impl $Op<BigFloat> for BigFloat {
            type Output = BigFloat;
            #[inline]
            fn $op(mut self, rhs: BigFloat) -> BigFloat {
                self.$assign(&rhs);
                self
            }
        }
        impl $Assign<&BigFloat> for BigFloat {
            /// Rounded to nearest at the larger precision of the operands, into the vector of
            /// `self`.
            #[inline]
            fn $assign(&mut self, rhs: &BigFloat) {
                let prec = self.prec.max(rhs.prec);
                self.assign_op(Assign::$kind, rhs, prec, Round::Nearest);
            }
        }
        impl $Assign<BigFloat> for BigFloat {
            #[inline]
            fn $assign(&mut self, rhs: BigFloat) {
                self.$assign(&rhs);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, Add);
binary!(Sub, sub, SubAssign, sub_assign, Sub);
binary!(Mul, mul, MulAssign, mul_assign, Mul);
binary!(Div, div, DivAssign, div_assign, Div);

impl Neg for BigFloat {
    type Output = BigFloat;
    /// The number with its sign turned over, exactly; NaN stays NaN.
    #[inline]
    fn neg(mut self) -> BigFloat {
        let negative = self.is_sign_negative();
        self.set_sign(!negative);
        self
    }
}

impl Neg for &BigFloat {
    type Output = BigFloat;
    #[inline]
    fn neg(self) -> BigFloat {
        -self.clone()
    }
}
