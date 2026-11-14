//! The forms into an existing number, as MPC's functions write into their first argument: each
//! part of the result rounded at the precision of that part of `self`, into its vector, and the
//! directions of the roundings. Their results are those of the `_round_parts` forms at those
//! precisions, special values and signs of zeros included.

use super::fast::{self, Acc};
use super::{BigComplex, Ternary};
use crate::math::float::{BigFloat, Kind, Round};
use core::cmp::Ordering;

/// Whether the window products take the parts: all finite and not zero, of fewer bits than those
/// from which a product takes three multiplications.
#[inline]
fn window(parts: &[&BigFloat]) -> bool {
    parts
        .iter()
        .all(|x| x.is_normal() && x.prec() < super::arith::MUL3)
}

/// A zero of the precision, without a vector, to write into.
#[inline]
fn blank(prec: u64) -> BigFloat {
    BigFloat {
        prec,
        kind: Kind::Zero { negative: false },
    }
}

/// Moves `x` into `part`, the vector `part` held given to the thread's pool.
#[inline]
fn swap_in(part: &mut BigFloat, mut x: BigFloat) {
    core::mem::swap(part, &mut x);
    fast::put_kind(&mut x, Kind::Nan);
}

impl BigComplex {
    /// Sets `self` to `a·b`, finite or not, at the precisions of its parts.
    pub(super) fn put_mul(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: (Round, Round),
    ) -> Ternary {
        let prec = self.prec();
        if a.is_finite() && b.is_finite() {
            let parts = [&a.re, &a.im, &b.re, &b.im];
            if let Some([p, q]) = fast::short_mul(parts, prec, round) {
                return (
                    fast::put_short(&mut self.re, p, round.0),
                    fast::put_short(&mut self.im, q, round.1),
                );
            }
            if window(&parts) {
                let [a, b, c, d] = parts;
                return (
                    fast::sum_products(&mut self.re, (a, c, false), (b, d, true), round.0),
                    fast::sum_products(&mut self.im, (a, d, false), (b, c, false), round.1),
                );
            }
            let mut acc = [Acc::new(), Acc::new()];
            if fast::exact_mul(parts, &mut acc) {
                return (
                    fast::put_acc(&mut self.re, &acc[0], 0, round.0),
                    fast::put_acc(&mut self.im, &acc[1], 0, round.1),
                );
            }
        }
        let (z, o) = a.mul_general(b, prec, round);
        *self = z;
        o
    }

    /// Sets `self` to `a²`, finite or not, at the precisions of its parts.
    pub(super) fn put_sqr(&mut self, a: &BigComplex, round: (Round, Round)) -> Ternary {
        let prec = self.prec();
        if a.is_finite() {
            let parts = [&a.re, &a.im];
            if let Some([p, q]) = fast::short_sqr(parts, prec, round) {
                return (
                    fast::put_short(&mut self.re, p, round.0),
                    fast::put_short(&mut self.im, q, round.1),
                );
            }
            if window(&parts) {
                let [a, b] = parts;
                return (
                    fast::sum_products(&mut self.re, (a, a, false), (b, b, true), round.0),
                    fast::scaled_product(&mut self.im, a, b, 1, round.1),
                );
            }
            let mut acc = [Acc::new(), Acc::new()];
            if fast::exact_sqr(parts, &mut acc) {
                return (
                    fast::put_acc(&mut self.re, &acc[0], 0, round.0),
                    fast::put_acc(&mut self.im, &acc[1], 0, round.1),
                );
            }
        }
        let (z, o) = a.sqr_general(prec, round);
        *self = z;
        o
    }

    /// Sets `self` to `a + b`, each part rounded to the precision of that part of `self` in the
    /// mode `round`, reusing their vectors, and returns the directions of the roundings, as
    /// MPC's `mpc_add`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let mut s = BigComplex::zero(2);
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// let w = BigComplex::new(BigFloat::from(0.25), BigFloat::from(-0.25));
    /// let o = s.set_add(&z, &w, Round::Down);
    /// assert_eq!(s, BigComplex::new(BigFloat::from(1), BigFloat::from(0.75)));
    /// assert_eq!(o, (Ordering::Less, Ordering::Equal));
    /// ```
    #[inline]
    pub fn set_add(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_add_parts(a, b, (round, round))
    }

    /// [`set_add`](BigComplex::set_add) with a mode for each part, the real part's first, as
    /// MPC's `mpc_rnd_t`.
    #[inline]
    pub fn set_add_parts(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            self.re.set_add(&a.re, &b.re, round.0),
            self.im.set_add(&a.im, &b.im, round.1),
        )
    }

    /// Sets `self` to `a - b`, each part rounded at the precision of that part of `self`, as
    /// MPC's `mpc_sub`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut d = BigComplex::zero(53);
    /// d.set_sub(&BigComplex::from(1), &BigComplex::i(53), Round::Nearest);
    /// assert_eq!(d, BigComplex::new(BigFloat::from(1), BigFloat::from(-1)));
    /// ```
    #[inline]
    pub fn set_sub(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_sub_parts(a, b, (round, round))
    }

    /// [`set_sub`](BigComplex::set_sub) with a mode for each part.
    #[inline]
    pub fn set_sub_parts(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            self.re.set_sub(&a.re, &b.re, round.0),
            self.im.set_sub(&a.im, &b.im, round.1),
        )
    }

    /// Sets `self` to `z + x` for a real `x`, as MPC's `mpc_add_fr`: the imaginary part `z`'s,
    /// rounded, of its sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut s = BigComplex::zero(53);
    /// s.set_add_real(&BigComplex::i(53), &BigFloat::from(2), Round::Nearest);
    /// assert_eq!(s, BigComplex::new(BigFloat::from(2), BigFloat::from(1)));
    /// ```
    #[inline]
    pub fn set_add_real(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_add_real_parts(z, x, (round, round))
    }

    /// [`set_add_real`](BigComplex::set_add_real) with a mode for each part.
    #[inline]
    pub fn set_add_real_parts(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            self.re.set_add(&z.re, x, round.0),
            fast::set_part(&mut self.im, &z.im, false, round.1),
        )
    }

    /// Sets `self` to `z - x` for a real `x`, as MPC's `mpc_sub_fr`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut d = BigComplex::zero(53);
    /// d.set_sub_real(&BigComplex::from(3), &BigFloat::from(1), Round::Nearest);
    /// assert_eq!(d, BigComplex::from(2));
    /// ```
    #[inline]
    pub fn set_sub_real(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_sub_real_parts(z, x, (round, round))
    }

    /// [`set_sub_real`](BigComplex::set_sub_real) with a mode for each part.
    #[inline]
    pub fn set_sub_real_parts(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            self.re.set_sub(&z.re, x, round.0),
            fast::set_part(&mut self.im, &z.im, false, round.1),
        )
    }

    /// Sets `self` to `z·x` for a real `x`, as MPC's `mpc_mul_fr`: `re·x + im·x·i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut p = BigComplex::zero(53);
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(-3));
    /// p.set_mul_real(&z, &BigFloat::from(0.5), Round::Nearest);
    /// assert_eq!(p, BigComplex::new(BigFloat::from(0.5), BigFloat::from(-1.5)));
    /// ```
    #[inline]
    pub fn set_mul_real(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_mul_real_parts(z, x, (round, round))
    }

    /// [`set_mul_real`](BigComplex::set_mul_real) with a mode for each part.
    #[inline]
    pub fn set_mul_real_parts(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            self.re.set_mul(&z.re, x, round.0),
            self.im.set_mul(&z.im, x, round.1),
        )
    }

    /// Sets `self` to `z / x` for a real `x`, as MPC's `mpc_div_fr`: `re/x + im/x·i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut q = BigComplex::zero(53);
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(-3));
    /// q.set_div_real(&z, &BigFloat::from(2), Round::Nearest);
    /// assert_eq!(q, BigComplex::new(BigFloat::from(0.5), BigFloat::from(-1.5)));
    /// ```
    #[inline]
    pub fn set_div_real(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_div_real_parts(z, x, (round, round))
    }

    /// [`set_div_real`](BigComplex::set_div_real) with a mode for each part.
    #[inline]
    pub fn set_div_real_parts(
        &mut self,
        z: &BigComplex,
        x: &BigFloat,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            self.re.set_div(&z.re, x, round.0),
            self.im.set_div(&z.im, x, round.1),
        )
    }

    /// Sets `self` to `-a`, each part rounded at the precision of that part of `self`, as MPC's
    /// `mpc_neg`: exact when the precisions are at least `a`'s.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut n = BigComplex::zero(53);
    /// n.set_neg(&BigComplex::new(BigFloat::from(1), BigFloat::from(-2)), Round::Nearest);
    /// assert_eq!(n, BigComplex::new(BigFloat::from(-1), BigFloat::from(2)));
    /// ```
    #[inline]
    pub fn set_neg(&mut self, a: &BigComplex, round: Round) -> (Ordering, Ordering) {
        self.set_neg_parts(a, (round, round))
    }

    /// [`set_neg`](BigComplex::set_neg) with a mode for each part.
    #[inline]
    pub fn set_neg_parts(&mut self, a: &BigComplex, round: (Round, Round)) -> (Ordering, Ordering) {
        (
            fast::set_part(&mut self.re, &a.re, true, round.0),
            fast::set_part(&mut self.im, &a.im, true, round.1),
        )
    }

    /// Sets `self` to the conjugate of `a`, `re - im·i`, each part rounded at the precision of
    /// that part of `self`, as MPC's `mpc_conj`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut c = BigComplex::zero(53);
    /// c.set_conj(&BigComplex::new(BigFloat::from(1), BigFloat::from(2)), Round::Nearest);
    /// assert_eq!(c, BigComplex::new(BigFloat::from(1), BigFloat::from(-2)));
    /// ```
    #[inline]
    pub fn set_conj(&mut self, a: &BigComplex, round: Round) -> (Ordering, Ordering) {
        self.set_conj_parts(a, (round, round))
    }

    /// [`set_conj`](BigComplex::set_conj) with a mode for each part.
    #[inline]
    pub fn set_conj_parts(
        &mut self,
        a: &BigComplex,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        (
            fast::set_part(&mut self.re, &a.re, false, round.0),
            fast::set_part(&mut self.im, &a.im, true, round.1),
        )
    }

    /// Sets `self` to `a·b`, each part rounded at the precision of that part of `self`, reusing
    /// their vectors, as MPC's `mpc_mul`: the parts `ac - bd` and `ad + bc` each the exact sum of
    /// two exact products rounded once, as [`mul_round`](BigComplex::mul_round) has them.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// let w = BigComplex::new(BigFloat::from(3), BigFloat::from(-1));
    /// let mut p = BigComplex::zero(53);
    /// let o = p.set_mul(&z, &w, Round::Nearest);
    /// assert_eq!(p, BigComplex::new(BigFloat::from(5), BigFloat::from(5)));
    /// assert_eq!(o, (Ordering::Equal, Ordering::Equal));
    /// ```
    #[inline]
    pub fn set_mul(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.put_mul(a, b, (round, round))
    }

    /// [`set_mul`](BigComplex::set_mul) with a mode for each part.
    #[inline]
    pub fn set_mul_parts(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        self.put_mul(a, b, round)
    }

    /// Sets `self` to `a²`, each part rounded at the precision of that part of `self`, as MPC's
    /// `mpc_sqr`: `a² - b²` and `2ab` each rounded once.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut s = BigComplex::zero(53);
    /// s.set_sqr(&BigComplex::new(BigFloat::from(2), BigFloat::from(1)), Round::Nearest);
    /// assert_eq!(s, BigComplex::new(BigFloat::from(3), BigFloat::from(4)));
    /// ```
    #[inline]
    pub fn set_sqr(&mut self, a: &BigComplex, round: Round) -> (Ordering, Ordering) {
        self.put_sqr(a, (round, round))
    }

    /// [`set_sqr`](BigComplex::set_sqr) with a mode for each part.
    #[inline]
    pub fn set_sqr_parts(&mut self, a: &BigComplex, round: (Round, Round)) -> (Ordering, Ordering) {
        self.put_sqr(a, round)
    }

    /// Sets `self` to `a / b`, each part rounded at the precision of that part of `self`, as
    /// MPC's `mpc_div` and [`div_round`](BigComplex::div_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut q = BigComplex::zero(53);
    /// let z = BigComplex::new(BigFloat::from(4), BigFloat::from(2));
    /// q.set_div(&z, &BigComplex::new(BigFloat::from(1), BigFloat::from(1)), Round::Nearest);
    /// assert_eq!(q, BigComplex::new(BigFloat::from(3), BigFloat::from(-1)));
    /// ```
    #[inline]
    pub fn set_div(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: Round,
    ) -> (Ordering, Ordering) {
        self.set_div_parts(a, b, (round, round))
    }

    /// [`set_div`](BigComplex::set_div) with a mode for each part.
    pub fn set_div_parts(
        &mut self,
        a: &BigComplex,
        b: &BigComplex,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        let (z, o) = a.div_round_parts(b, self.prec(), round);
        *self = z;
        o
    }

    /// Sets `self` to the principal square root of `a`, each part rounded at the precision of
    /// that part of `self`, as MPC's `mpc_sqrt` and [`sqrt_round`](BigComplex::sqrt_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut r = BigComplex::zero(53);
    /// r.set_sqrt(&BigComplex::new(BigFloat::from(3), BigFloat::from(4)), Round::Nearest);
    /// assert_eq!(r, BigComplex::new(BigFloat::from(2), BigFloat::from(1)));
    /// ```
    #[inline]
    pub fn set_sqrt(&mut self, a: &BigComplex, round: Round) -> (Ordering, Ordering) {
        super::sqrt::root_into(self, a, (round, round))
    }

    /// [`set_sqrt`](BigComplex::set_sqrt) with a mode for each part.
    #[inline]
    pub fn set_sqrt_parts(
        &mut self,
        a: &BigComplex,
        round: (Round, Round),
    ) -> (Ordering, Ordering) {
        super::sqrt::root_into(self, a, round)
    }

    /// Sets `self` to `self + other`, each part rounded at the precision of that part of `self`
    /// in the mode `round`, as MPC's `mpc_add (z, z, w, rnd)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut z = BigComplex::from(1);
    /// z.add_assign_round(&BigComplex::i(53), Round::Nearest);
    /// assert_eq!(z, BigComplex::new(BigFloat::from(1), BigFloat::from(1)));
    /// ```
    #[inline]
    pub fn add_assign_round(&mut self, other: &BigComplex, round: Round) -> (Ordering, Ordering) {
        (
            self.re.add_assign_round(&other.re, round),
            self.im.add_assign_round(&other.im, round),
        )
    }

    /// Sets `self` to `self - other`, each part rounded at the precision of that part of `self`.
    #[inline]
    pub fn sub_assign_round(&mut self, other: &BigComplex, round: Round) -> (Ordering, Ordering) {
        (
            self.re.sub_assign_round(&other.re, round),
            self.im.sub_assign_round(&other.im, round),
        )
    }

    /// Sets `self` to `self · other`, each part rounded at the precision of that part of `self`,
    /// as MPC's `mpc_mul (z, z, w, rnd)`: the parts computed before either is written.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut z = BigComplex::new(BigFloat::from(1), BigFloat::from(2));
    /// z.mul_assign_round(&BigComplex::new(BigFloat::from(3), BigFloat::from(-1)), Round::Nearest);
    /// assert_eq!(z, BigComplex::new(BigFloat::from(5), BigFloat::from(5)));
    /// ```
    pub fn mul_assign_round(&mut self, other: &BigComplex, round: Round) -> (Ordering, Ordering) {
        let prec = self.prec();
        let round = (round, round);
        if self.is_finite() && other.is_finite() {
            let parts = [&self.re, &self.im, &other.re, &other.im];
            if let Some([p, q]) = fast::short_mul(parts, prec, round) {
                return (
                    fast::put_short(&mut self.re, p, round.0),
                    fast::put_short(&mut self.im, q, round.1),
                );
            }
            if window(&parts) {
                // The real part into a spare number first: the imaginary part reads it.
                let [a, b, c, d] = parts;
                let mut re = blank(prec.0);
                let x = fast::sum_products(&mut re, (a, c, false), (b, d, true), round.0);
                let mut im = blank(prec.1);
                let y = fast::sum_products(&mut im, (a, d, false), (b, c, false), round.1);
                swap_in(&mut self.re, re);
                swap_in(&mut self.im, im);
                return (x, y);
            }
            let mut acc = [Acc::new(), Acc::new()];
            if fast::exact_mul(parts, &mut acc) {
                return (
                    fast::put_acc(&mut self.re, &acc[0], 0, round.0),
                    fast::put_acc(&mut self.im, &acc[1], 0, round.1),
                );
            }
        }
        let (z, o) = self.mul_general(other, prec, round);
        *self = z;
        o
    }

    /// Sets `self` to `self / other`, each part rounded at the precision of that part of `self`.
    pub fn div_assign_round(&mut self, other: &BigComplex, round: Round) -> (Ordering, Ordering) {
        let (z, o) = self.div_round_parts(other, self.prec(), (round, round));
        *self = z;
        o
    }

    /// Sets `self` to its square, each part rounded at the precision of that part of `self`, as
    /// MPC's `mpc_sqr (z, z, rnd)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    ///
    /// let mut z = BigComplex::new(BigFloat::from(1), BigFloat::from(1));
    /// z.sqr_assign_round(Round::Nearest);
    /// assert_eq!(z, BigComplex::new(BigFloat::from(0), BigFloat::from(2)));
    /// ```
    pub fn sqr_assign_round(&mut self, round: Round) -> (Ordering, Ordering) {
        let prec = self.prec();
        let round = (round, round);
        if self.is_finite() {
            let parts = [&self.re, &self.im];
            if let Some([p, q]) = fast::short_sqr(parts, prec, round) {
                return (
                    fast::put_short(&mut self.re, p, round.0),
                    fast::put_short(&mut self.im, q, round.1),
                );
            }
            if window(&parts) {
                // The real part into a spare number first: the imaginary part reads it.
                let [a, b] = parts;
                let mut re = blank(prec.0);
                let x = fast::sum_products(&mut re, (a, a, false), (b, b, true), round.0);
                let mut im = blank(prec.1);
                let y = fast::scaled_product(&mut im, a, b, 1, round.1);
                swap_in(&mut self.re, re);
                swap_in(&mut self.im, im);
                return (x, y);
            }
            let mut acc = [Acc::new(), Acc::new()];
            if fast::exact_sqr(parts, &mut acc) {
                return (
                    fast::put_acc(&mut self.re, &acc[0], 0, round.0),
                    fast::put_acc(&mut self.im, &acc[1], 0, round.1),
                );
            }
        }
        let (z, o) = self.sqr_general(prec, round);
        *self = z;
        o
    }
}

impl BigFloat {
    /// Sets `self` to the norm `|z|² = re² + im²` of the complex number `z`, rounded once to the
    /// precision of `self` in the mode `round`, reusing its vector, and returns the direction of
    /// the rounding, as MPC's `mpc_norm` and [`BigComplex::norm_round`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let mut n = BigFloat::zero(4);
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(4));
    /// assert_eq!(n.set_complex_norm(&z, Round::Down), Ordering::Less);
    /// assert_eq!(n, BigFloat::from(24));
    /// ```
    pub fn set_complex_norm(&mut self, z: &BigComplex, round: Round) -> Ordering {
        if !z.is_finite() {
            let kind = if z.is_infinite() {
                Kind::Inf { negative: false }
            } else {
                Kind::Nan
            };
            fast::put_kind(self, kind);
            return Ordering::Equal;
        }
        if self.prec <= 128
            && let (Some(aa), Some(bb)) = (
                fast::short_product(&z.re, &z.re, false),
                fast::short_product(&z.im, &z.im, false),
            )
        {
            let r = fast::short_sum(aa, bb, self.prec, round);
            return fast::put_short(self, r, round);
        }
        if self.prec <= 128
            && let (Some(aa), Some(bb)) = (
                fast::long_product(&z.re, &z.re, false),
                fast::long_product(&z.im, &z.im, false),
            )
        {
            let r = fast::long_sum(aa, bb, self.prec, round);
            return fast::put_short(self, r, round);
        }
        if window(&[&z.re, &z.im]) {
            return fast::sum_products(self, (&z.re, &z.re, false), (&z.im, &z.im, false), round);
        }
        let mut acc = Acc::new();
        let terms = [
            fast::Term::product(&z.re, &z.re, false),
            fast::Term::product(&z.im, &z.im, false),
        ];
        if fast::exact_sum(&terms, &mut acc) {
            return fast::put_acc(self, &acc, 0, round);
        }
        let (x, o) = z.norm_general(self.prec, round);
        *self = x;
        o
    }

    /// Sets `self` to the absolute value `|z| = √(re² + im²)` of the complex number `z`, rounded
    /// once to the precision of `self` in the mode `round`, reusing its vector, and returns the
    /// direction of the rounding, as MPC's `mpc_abs` and [`BigComplex::abs_round`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigComplex, BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let mut r = BigFloat::zero(53);
    /// let z = BigComplex::new(BigFloat::from(3), BigFloat::from(-4));
    /// assert_eq!(r.set_complex_abs(&z, Round::Nearest), Ordering::Equal);
    /// assert_eq!(r, BigFloat::from(5));
    /// ```
    pub fn set_complex_abs(&mut self, z: &BigComplex, round: Round) -> Ordering {
        if !z.is_finite() {
            let kind = if z.is_infinite() {
                Kind::Inf { negative: false }
            } else {
                Kind::Nan
            };
            fast::put_kind(self, kind);
            return Ordering::Equal;
        }
        if self.prec <= 127 {
            let mut acc = Acc::new();
            let terms = [
                fast::Term::product(&z.re, &z.re, false),
                fast::Term::product(&z.im, &z.im, false),
            ];
            if fast::exact_sum(&terms, &mut acc) {
                if let Some(zeros) = acc.zero {
                    return fast::put_zero(self, zeros.negative(round));
                }
                let r = fast::sqrt_acc(&acc, self.prec, round);
                return fast::put(self, r, round);
            }
        }
        if z.re.is_normal() && z.im.is_normal() {
            return fast::hypot_into(self, &z.re, &z.im, round);
        }
        let (x, o) = z.abs_general(self.prec, round);
        *self = x;
        o
    }
}
