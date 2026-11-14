//! Rounding to integers, as MPFR's `mpfr_floor`, `mpfr_ceil`, `mpfr_trunc`, `mpfr_round`,
//! `mpfr_roundeven`, `mpfr_rint`, their `mpfr_rint_*` forms and `mpfr_frac`: NaN stays NaN, the
//! infinities and the zeros stay as they are, and a zero result has the sign of the number.

use super::limbs;
use super::round::{round_into, round_mag, signed};
use super::scale::{Dir, int_parts};
use super::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// The ways of [`BigFloat::floor`] and the others to choose an integer, for a number of the
/// sign `negative`: the direction of its magnitude.
#[derive(Clone, Copy)]
enum Way {
    Floor,
    Ceil,
    Trunc,
    Away,
    Even,
}

impl Way {
    #[inline]
    fn dir(self, negative: bool) -> Dir {
        match self {
            Way::Floor => Dir::of(negative, Round::Down),
            Way::Ceil => Dir::of(negative, Round::Up),
            Way::Trunc => Dir::Floor,
            Way::Away => Dir::Away,
            Way::Even => Dir::Nearest,
        }
    }
}

impl BigFloat {
    /// The integer that the finite number, not zero and with bits below its point, rounds to in
    /// the direction `dir` of its magnitude, at the number's precision, which holds it, and the
    /// direction of that rounding of the magnitude; a zero of the number's sign for zero.
    pub(crate) fn integer_same(&self, dir: Dir) -> (BigFloat, Ordering) {
        let (negative, exp, mant) = match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => (*negative, *exp, mant),
            _ => unreachable!("a finite number"),
        };
        let (exp, o, out) = integer_copy(mant, exp, dir);
        let kind = match exp {
            Some(exp) => Kind::Finite {
                negative,
                exp,
                mant: out,
            },
            None => Kind::Zero { negative },
        };
        (
            BigFloat {
                prec: self.prec,
                kind,
            },
            o,
        )
    }

    /// The integer that the number rounds to in the way `way`, rounded to `prec` bits in the mode
    /// `round`, and the direction of that rounding from the integer (as `mpfr_rint_floor` and the
    /// others).
    fn integer_round(&self, way: Way, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Finite { negative, .. } => {
                let (_, _, lsb) = self.parts().expect("a finite number");
                if lsb >= 0 {
                    // An integer already.
                    return self.prec_round(prec, round);
                }
                let exp = self.exponent().expect("a finite number");
                if prec == self.prec || exp <= 0 || exp as u64 <= prec {
                    // The integer has at most `exp` bits, or is `2^exp`: `prec` bits hold it.
                    let (int, _) = self.integer_at(way.dir(*negative), prec);
                    (int, Ordering::Equal)
                } else {
                    let (int, _) = self.integer_same(way.dir(*negative));
                    int.prec_round(prec, round)
                }
            }
            kind => (
                BigFloat {
                    prec,
                    kind: kind.clone(),
                },
                Ordering::Equal,
            ),
        }
    }

    /// The greatest integer not above the number, at its precision, which holds it, as
    /// `mpfr_floor` and `f64::floor`: `floor(-0.5) = -1`, `floor(0.5) = +0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(-2.5).floor(), BigFloat::from(-3));
    /// assert!(BigFloat::from(0.5).floor().is_sign_positive());
    /// ```
    pub fn floor(&self) -> BigFloat {
        self.integer_round(Way::Floor, self.prec, Round::Nearest).0
    }

    /// The least integer not below the number, at its precision, as `mpfr_ceil` and `f64::ceil`:
    /// `ceil(-0.5) = -0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(2.25).ceil(), BigFloat::from(3));
    /// assert!(BigFloat::from(-0.5).ceil().is_sign_negative());
    /// ```
    pub fn ceil(&self) -> BigFloat {
        self.integer_round(Way::Ceil, self.prec, Round::Nearest).0
    }

    /// The integer part, toward zero, at the number's precision, as `mpfr_trunc` and
    /// `f64::trunc`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(-2.75).trunc(), BigFloat::from(-2));
    /// ```
    pub fn trunc(&self) -> BigFloat {
        self.integer_round(Way::Trunc, self.prec, Round::Nearest).0
    }

    /// The nearest integer, halfway cases away from zero, at the number's precision, as
    /// `mpfr_round` and `f64::round`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(2.5).round(), BigFloat::from(3));
    /// assert_eq!(BigFloat::from(-0.5).round(), BigFloat::from(-1));
    /// ```
    pub fn round(&self) -> BigFloat {
        self.integer_round(Way::Away, self.prec, Round::Nearest).0
    }

    /// The nearest integer, halfway cases to the even one, at the number's precision, as
    /// `mpfr_roundeven` and `f64::round_ties_even`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(2.5).round_ties_even(), BigFloat::from(2));
    /// assert_eq!(BigFloat::from(3.5).round_ties_even(), BigFloat::from(4));
    /// ```
    pub fn round_ties_even(&self) -> BigFloat {
        self.integer_round(Way::Even, self.prec, Round::Nearest).0
    }

    /// [`floor`](Self::floor) rounded to `prec` bits in the mode `round`, and the direction of
    /// that rounding from the integer, as `mpfr_rint_floor`: two roundings, the second only when
    /// the integer has more than `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// // floor(1023.5) = 1023, which is 1024 to 2 bits, rounded up.
    /// let (x, o) = BigFloat::from(1023.5).floor_round(2, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(1024), Ordering::Greater));
    /// ```
    #[track_caller]
    pub fn floor_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        self.integer_round(Way::Floor, prec, round)
    }

    /// [`ceil`](Self::ceil) rounded to `prec` bits in the mode `round`, and the direction of that
    /// rounding from the integer, as `mpfr_rint_ceil`.
    #[track_caller]
    pub fn ceil_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        self.integer_round(Way::Ceil, prec, round)
    }

    /// [`trunc`](Self::trunc) rounded to `prec` bits in the mode `round`, and the direction of
    /// that rounding from the integer, as `mpfr_rint_trunc`.
    #[track_caller]
    pub fn trunc_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        self.integer_round(Way::Trunc, prec, round)
    }

    /// [`round`](Self::round) rounded to `prec` bits in the mode `round`, and the direction of
    /// that rounding from the integer, as `mpfr_rint_round`: 6.5 to 2 bits is 7 then 8.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(6.5).round_round(2, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(8), Ordering::Greater));
    /// ```
    #[track_caller]
    pub fn round_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        self.integer_round(Way::Away, prec, round)
    }

    /// [`round_ties_even`](Self::round_ties_even) rounded to `prec` bits in the mode `round`,
    /// and the direction of that rounding from the integer, as `mpfr_rint_roundeven`.
    #[track_caller]
    pub fn round_ties_even_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        self.integer_round(Way::Even, prec, round)
    }

    /// The integer of `prec` bits nearest to the number in the direction of the mode `round`,
    /// rounded once, and the direction from the number, as `mpfr_rint`: to nearest, 10.5 to 2
    /// bits is 12, the nearer of the two integers of 2 bits around it, 8 and 12. A number of more
    /// bits before its point than `prec` rounds as by [`prec_round`](Self::prec_round), the
    /// numbers of `prec` bits there being integers; one of fewer to an integer in the mode
    /// (ties to the even integer), which `prec` bits hold. `rint_round(p, Round::Down)` is
    /// `mpfr_floor` to `p` bits, and so on for `Up` and `TowardZero`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(10.5).rint_round(2, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(12), Ordering::Greater));
    /// let (x, o) = BigFloat::from(-2.5).rint_round(10, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(-2), Ordering::Greater));
    /// ```
    #[track_caller]
    pub fn rint_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Finite { negative, exp, .. } => {
                let (_, _, lsb) = self.parts().expect("a finite number");
                if lsb >= 0 || i128::from(*exp) >= i128::from(prec) {
                    return self.prec_round(prec, round);
                }
                // At most `exp < prec` bits, or `2^exp`: exact at `prec`.
                let (int, o) = self.integer_at(Dir::of(*negative, round), prec);
                (int, signed(*negative, o))
            }
            kind => (
                BigFloat {
                    prec,
                    kind: kind.clone(),
                },
                Ordering::Equal,
            ),
        }
    }

    /// [`integer_same`](Self::integer_same) at the precision `prec`, which must hold the integer
    /// (at least the number's exponent), in one pass into the result's limbs.
    fn integer_at(&self, dir: Dir, prec: u64) -> (BigFloat, Ordering) {
        let Kind::Finite {
            negative,
            exp,
            mant,
        } = &self.kind
        else {
            unreachable!("a finite number")
        };
        let n = prec.div_ceil(64) as usize;
        let (exp, o, out) = if n == mant.len() {
            integer_copy(mant, *exp, dir)
        } else {
            let mut out = vec![0; n];
            let (e, o) = integer_fit(mant, *exp, dir, &mut out);
            (e, o, out)
        };
        let kind = match exp {
            Some(exp) => Kind::Finite {
                negative: *negative,
                exp,
                mant: out,
            },
            None => Kind::Zero {
                negative: *negative,
            },
        };
        (BigFloat { prec, kind }, o)
    }

    /// Sets `self` to [`rint_round`](Self::rint_round) of `a` at the precision of `self`,
    /// reusing its vector, and returns the direction of the rounding from `a`, as MPFR's
    /// `mpfr_rint`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let mut r = BigFloat::zero(2);
    /// assert_eq!(r.set_rint(&BigFloat::from(10.5), Round::Nearest), Ordering::Greater);
    /// assert_eq!(r, BigFloat::from(12));
    /// ```
    pub fn set_rint(&mut self, a: &BigFloat, round: Round) -> Ordering {
        let prec = self.prec;
        let Kind::Finite {
            negative: na,
            exp: ea,
            mant: ma,
        } = &a.kind
        else {
            self.kind = a.kind.clone();
            return Ordering::Equal;
        };
        let (na, ea) = (*na, *ea);
        let lsb = i128::from(ea) - 64 * ma.len() as i128;
        if lsb >= 0 || i128::from(ea) >= i128::from(prec) {
            // The numbers of `prec` bits there are integers: a rounding to the precision.
            let out = self.take_mant();
            let (kind, o) = round_into(na, ma, lsb, false, prec, round, out);
            self.kind = kind;
            return o;
        }
        let dir = Dir::of(na, round);
        if let Kind::Finite {
            negative,
            exp,
            mant,
        } = &mut self.kind
        {
            let (e, o) = integer_fit(ma, ea, dir, mant);
            if let Some(e) = e {
                *negative = na;
                *exp = e;
            } else {
                self.kind = Kind::Zero { negative: na };
            }
            return signed(na, o);
        }
        let (x, o) = a.integer_at(dir, prec);
        self.kind = x.kind;
        signed(na, o)
    }

    /// Sets `self` to the greatest integer not above `a`, rounded to the precision of `self`
    /// toward negative infinity, reusing its vector, and returns the direction from `a`, as
    /// MPFR's `mpfr_floor`: [`set_rint`](Self::set_rint) with [`Round::Down`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    /// use core::cmp::Ordering;
    ///
    /// let mut r = BigFloat::zero(53);
    /// assert_eq!(r.set_floor(&BigFloat::from(-2.5)), Ordering::Less);
    /// assert_eq!(r, BigFloat::from(-3));
    /// ```
    pub fn set_floor(&mut self, a: &BigFloat) -> Ordering {
        self.set_rint(a, Round::Down)
    }

    /// Sets `self` to the least integer not below `a`, rounded to the precision of `self` toward
    /// positive infinity, as MPFR's `mpfr_ceil`: [`set_rint`](Self::set_rint) with
    /// [`Round::Up`].
    pub fn set_ceil(&mut self, a: &BigFloat) -> Ordering {
        self.set_rint(a, Round::Up)
    }

    /// Sets `self` to the integer part of `a`, rounded to the precision of `self` toward zero, as
    /// MPFR's `mpfr_trunc`: [`set_rint`](Self::set_rint) with [`Round::TowardZero`].
    pub fn set_trunc(&mut self, a: &BigFloat) -> Ordering {
        self.set_rint(a, Round::TowardZero)
    }

    /// The fractional part, `x - trunc(x)`, of the sign of the number, rounded to `prec` bits in
    /// the mode `round`, and the direction of the rounding, as `mpfr_frac`: a zero of the sign of
    /// the number for an integer and, as MPFR has it, for an infinity (where `f64::fract` gives
    /// NaN).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (x, o) = BigFloat::from(-3.375).fract_round(2, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(-0.375), Ordering::Equal));
    /// let (x, o) = BigFloat::from(2.625).fract_round(2, Round::Nearest);
    /// assert_eq!((x, o), (BigFloat::from(0.5), Ordering::Less)); // 0.101 in binary, a tie
    /// ```
    #[track_caller]
    pub fn fract_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let zero = |negative| {
            (
                BigFloat {
                    prec,
                    kind: Kind::Zero { negative },
                },
                Ordering::Equal,
            )
        };
        match &self.kind {
            Kind::Nan => (BigFloat::nan(prec), Ordering::Equal),
            Kind::Inf { negative } | Kind::Zero { negative } => zero(*negative),
            Kind::Finite { negative, exp, .. } => {
                let (_, m, lsb) = self.parts().expect("a finite number");
                if lsb >= 0 {
                    return zero(*negative);
                }
                if *exp <= 0 {
                    return self.prec_round(prec, round);
                }
                // The bits below the point.
                let (t, _, _) = int_parts(m, lsb);
                let int = limbs::shl(&t, (-lsb) as u64);
                let mut f = limbs::sub(m, &int);
                limbs::trim(&mut f);
                if f.is_empty() {
                    return zero(*negative);
                }
                round_mag(*negative, &f, lsb, false, prec, round)
            }
        }
    }

    /// The fractional part, `x - trunc(x)`, at the number's precision, which holds it, as
    /// [`fract_round`](Self::fract_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::from(-1.25).fract(), BigFloat::from(-0.25));
    /// ```
    pub fn fract(&self) -> BigFloat {
        self.fract_round(self.prec, Round::Nearest).0
    }

    /// Whether the number is an integer, as `mpfr_integer_p`: a zero or a finite number without
    /// bits below its point.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert!(BigFloat::from(-4.0).is_integer() && !BigFloat::from(4.5).is_integer());
    /// assert!(!BigFloat::infinity(5).is_integer());
    /// ```
    pub fn is_integer(&self) -> bool {
        match self.parts() {
            Some((_, m, lsb)) => lsb >= 0 || limbs::zero_below(m, (-lsb) as u64),
            None => self.is_zero(),
        }
    }
}

/// The integer that the finite number `0.mant·2^exp`, not zero, rounds to in the direction `dir`
/// of its magnitude, into the limbs `out` (`64·out.len()` bits, at least `exp`): the exponent of
/// the integer, `None` for zero, and the direction of the rounding of the magnitude. The limbs
/// from the unit's up are the mantissa's top ones, those below zero.
fn integer_fit(mant: &[u64], exp: i64, dir: Dir, out: &mut [u64]) -> (Option<i64>, Ordering) {
    let (len, n) = (mant.len(), out.len());
    debug_assert!(exp <= 64 * n as i64);
    let (up, o) = integer_dir(mant, exp, dir);
    if exp > 0 {
        // The unit's limb lies at or above the mantissa's lowest one there, `exp < 64·len`.
        let k = unit_limb(n, exp);
        debug_assert!(k + len >= n);
        zero(&mut out[..k]);
        out[k..].copy_from_slice(&mant[k + len - n..]);
    }
    (integer_apply(out, exp, up), o)
}

/// [`integer_fit`] into a copy of the mantissa, decided from the mantissa before the copy is made,
/// and none made for a zero.
fn integer_copy(mant: &[u64], exp: i64, dir: Dir) -> (Option<i64>, Ordering, Vec<u64>) {
    let (up, o) = integer_dir(mant, exp, dir);
    if exp <= 0 && !up {
        return (None, o, Vec::new());
    }
    let mut v = mant.to_vec();
    if exp > 0 {
        let k = unit_limb(v.len(), exp);
        zero(&mut v[..k]);
    }
    (integer_apply(&mut v, exp, up), o, v)
}

/// The limb of `n` that holds the unit's bit of a number of the exponent `exp`, `0 < exp ≤ 64n`.
#[inline]
fn unit_limb(n: usize, exp: i64) -> usize {
    ((64 * n as u64 - exp as u64) / 64) as usize
}

/// Whether the magnitude `0.mant·2^exp` rounds up to an integer in the direction `dir`, and the
/// direction of that rounding: the bits of the fraction, below the unit's, decide it, all of the
/// mantissa's below one, its top bit the half for a number from one half to one.
#[inline]
fn integer_dir(mant: &[u64], exp: i64, dir: Dir) -> (bool, Ordering) {
    let all = 64 * mant.len() as u64;
    let u = if exp < 0 {
        all + 1
    } else {
        all.saturating_sub(exp as u64)
    };
    let (half, rest, odd) = if u == 0 {
        (false, false, false)
    } else {
        (
            limbs::bit(mant, u - 1),
            !limbs::zero_below(mant, u - 1),
            limbs::bit(mant, u),
        )
    };
    let up = match dir {
        Dir::Floor => false,
        Dir::Ceil => half || rest,
        Dir::Nearest => half && (rest || odd),
        Dir::Away => half,
    };
    let o = if !(half || rest) {
        Ordering::Equal
    } else if up {
        Ordering::Greater
    } else {
        Ordering::Less
    };
    (up, o)
}

/// The integer of the exponent `exp` in the limbs `out`, whose limbs below the unit's are already
/// zero: the bits of the unit's limb below it cleared, and one unit added when `up`, a carry out of
/// the top making it the next power of two; a number below one is zero or one. Its exponent, or
/// `None` for zero.
#[inline]
fn integer_apply(out: &mut [u64], exp: i64, up: bool) -> Option<i64> {
    let n = out.len();
    if exp <= 0 {
        if !up {
            return None;
        }
        zero(&mut out[..n - 1]);
        out[n - 1] = 1 << 63;
        return Some(1);
    }
    let uo = 64 * n as u64 - exp as u64;
    let (k, b) = ((uo / 64) as usize, uo % 64);
    out[k] &= !((1u64 << b) - 1);
    let mut exp = exp;
    if up {
        let (v, mut carry) = out[k].overflowing_add(1u64 << b);
        out[k] = v;
        for x in &mut out[k + 1..] {
            if !carry {
                break;
            }
            (*x, carry) = x.overflowing_add(1);
        }
        if carry {
            out[n - 1] = 1 << 63;
            exp += 1;
        }
    }
    Some(exp)
}

/// Sets the limbs `v` to zero: a loop for the few limbs of the short numbers, which a call of the
/// library's would cost more than.
#[inline]
fn zero(v: &mut [u64]) {
    if v.len() > 16 {
        v.fill(0);
    } else {
        for x in v.iter_mut() {
            *x = 0;
        }
    }
}
