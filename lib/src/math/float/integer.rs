//! Rounding to integers, as MPFR's `mpfr_floor`, `mpfr_ceil`, `mpfr_trunc`, `mpfr_round`,
//! `mpfr_roundeven`, `mpfr_rint`, their `mpfr_rint_*` forms and `mpfr_frac`: NaN stays NaN, the
//! infinities and the zeros stay as they are, and a zero result has the sign of the number.

use super::limbs;
use super::round::{round_mag, signed};
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
    /// The integer that the finite number, not zero and not an integer, rounds to in the
    /// direction `dir` of its magnitude, at the number's precision, which holds it, and the
    /// direction of that rounding of the magnitude; a zero of the number's sign for zero.
    ///
    /// With bits before the point, the mantissa is copied with its fraction cleared, and one unit
    /// added when the direction rounds up, a carry out of the top making it the next power of two;
    /// a number below one is zero or one.
    pub(crate) fn integer_same(&self, dir: Dir) -> (BigFloat, Ordering) {
        let (negative, exp, mant) = match &self.kind {
            Kind::Finite {
                negative,
                exp,
                mant,
            } => (*negative, *exp, mant),
            _ => unreachable!("a finite number"),
        };
        let len = mant.len();
        let all = 64 * len as u64;
        // The bits of the fraction, below the unit's: all of them below one, the top one the half
        // for a number from one half to one.
        let u = if exp < 0 { all + 1 } else { all - exp as u64 };
        debug_assert!(u > 0 && (exp <= 0 || u < all));
        let half = limbs::bit(mant, u - 1);
        let rest = !limbs::zero_below(mant, u - 1);
        let unit_odd = limbs::bit(mant, u);
        let up = match dir {
            Dir::Floor => false,
            Dir::Ceil => half || rest,
            Dir::Nearest => half && (rest || unit_odd),
            Dir::Away => half,
        };
        let o = if !(half || rest) {
            Ordering::Equal
        } else if up {
            Ordering::Greater
        } else {
            Ordering::Less
        };
        if exp <= 0 {
            let kind = if up {
                let mut mant = vec![0; len];
                mant[len - 1] = 1 << 63;
                Kind::Finite {
                    negative,
                    exp: 1,
                    mant,
                }
            } else {
                Kind::Zero { negative }
            };
            return (
                BigFloat {
                    prec: self.prec,
                    kind,
                },
                o,
            );
        }
        let (k, b) = ((u / 64) as usize, u % 64);
        let mut v = mant.clone();
        v[..k].fill(0);
        let mut exp = exp;
        if b > 0 {
            v[k] &= !((1u64 << b) - 1);
        }
        if up {
            let (s, mut carry) = v[k].overflowing_add(1u64 << b);
            v[k] = s;
            for x in &mut v[k + 1..] {
                if !carry {
                    break;
                }
                (*x, carry) = x.overflowing_add(1);
            }
            if carry {
                v[len - 1] = 1 << 63;
                exp += 1;
            }
        }
        (
            BigFloat {
                prec: self.prec,
                kind: Kind::Finite {
                    negative,
                    exp,
                    mant: v,
                },
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
                let (int, _) = self.integer_same(way.dir(*negative));
                if prec == self.prec {
                    (int, Ordering::Equal)
                } else {
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
                let (int, o) = self.integer_same(Dir::of(*negative, round));
                let o = signed(*negative, o);
                // At most `exp < prec` bits, or `2^exp`: exact.
                let int = if prec == self.prec {
                    int
                } else {
                    int.prec_round(prec, round).0
                };
                (int, o)
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
