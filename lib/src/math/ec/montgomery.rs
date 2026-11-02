//! Curves in Montgomery form, `B y^2 = x^3 + A x^2 + x`.

use super::curve::{Arithmetic, Curve, CurveError, choose_root, sec1_decode, sec1_encode};
use super::edwards::Edwards;
use super::field::{Base, Field, sqrt_ratio, with_prime};
use super::group::{Group, bit_length};
use super::point::Point;
use super::weierstrass::{
    Affine, Jacobian, Weierstrass, jacobian_equal, jacobian_group, normalize,
};
use crate::math::numbers::Number;

/// A curve `B y^2 = x^3 + A x^2 + x` over the field of a prime `p > 3`.
///
/// Products take Montgomery's ladder, which needs `x` alone, and [`x_mul`](Montgomery::x_mul)
/// gives that directly, as X25519 does; a product of a whole point recovers its `y` at the end.
/// Additions take Jacobian coordinates with formulas of their own for this form.
///
/// # Examples
///
/// ```
/// use a::math::ec::{Curve, Montgomery};
/// use a::math::numbers::{ByteOrder, Number};
///
/// let n = |x| Number::new_from_u32(x, ByteOrder::BigEndian);
/// // y^2 = x^3 + 6 x^2 + x over F_101, with the point (3, 36).
/// let curve = Montgomery::new(&n(6), &n(1), &n(101)).unwrap();
/// assert!(curve.point(&n(3), &n(36)).is_ok());
/// ```
pub struct Montgomery {
    /// The field.
    base: Base,
    /// `A` below `p`.
    a: Number,
    /// `B` below `p`.
    b: Number,
    /// `A` in the field's form.
    a_limbs: Vec<u64>,
    /// `B` in the field's form.
    b_limbs: Vec<u64>,
    /// `(A + 2) / 4` in the field's form, for the ladder.
    a24_limbs: Vec<u64>,
    /// `(A + 2) / 4` as a number, when it is short.
    a24_word: Option<u32>,
}

impl Montgomery {
    /// The curve `B y^2 = x^3 + A x^2 + x` modulo `p`, in `p`'s byte order. `a` is `A` and `b` is
    /// `B`, both reduced modulo `p`.
    ///
    /// # Errors
    ///
    /// [`CurveError::NotPrime`] when `p` is not a prime above 3, by
    /// [`is_probable_prime`](Number::is_probable_prime); [`CurveError::Singular`] when `B` is zero
    /// or `A^2 = 4` modulo `p`.
    pub fn new(a: &Number, b: &Number, p: &Number) -> Result<Montgomery, CurveError> {
        let three = Number::new_from_u8(3, p.byte_order);
        if p.is_negative() || *p <= three || !p.is_probable_prime() {
            return Err(CurveError::NotPrime);
        }
        let a = a.rem_euclid(p);
        let b = b.rem_euclid(p);
        if b.is_zero() || ((&a * &a) - 4u8).rem_euclid(p).is_zero() {
            return Err(CurveError::Singular);
        }
        let base = Base::new(p);
        let n = with_prime!(&base.prime, f => f.limbs());
        let (mut a_limbs, mut b_limbs, mut a24_limbs) = (vec![0; n], vec![0; n], vec![0; n]);
        let four = Number::new_from_u8(4, p.byte_order);
        let a24 = ((&a + 2u8) * four.mod_inverse(p).unwrap_or_default()).rem_euclid(p);
        let a24_word = u32::try_from(&a24).ok();
        with_prime!(&base.prime, f => {
            f.store(&f.enter(&a), &mut a_limbs);
            f.store(&f.enter(&b), &mut b_limbs);
            f.store(&f.enter(&a24), &mut a24_limbs);
        });
        Ok(Montgomery {
            base,
            a,
            b,
            a_limbs,
            b_limbs,
            a24_limbs,
            a24_word,
        })
    }

    /// The coefficient `A`, below `p`.
    pub fn a(&self) -> &Number {
        &self.a
    }

    /// The coefficient `B`, below `p`.
    pub fn b(&self) -> &Number {
        &self.b
    }

    /// The `x` of `k P`, for a point `P` with the coordinate `u`, or zero when `k P` is the point
    /// at infinity, by Montgomery's ladder, without `y`, as RFC 7748 defines X25519 (without its
    /// clamping of the scalar). `u` need not be on this curve: it may be on the curve's quadratic
    /// twist, and `k` counts by magnitude, as `x` is the same for `k P` and `-k P`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::ec::named;
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let curve = named::curve25519();
    /// let k = Number::new_from_u8(5, ByteOrder::LittleEndian);
    /// let g = curve.generator();
    /// let (x, _) = (&g * &k).coordinates().unwrap();
    /// assert_eq!(curve.x_mul(&g.coordinates().unwrap().0, &k), x);
    /// ```
    pub fn x_mul(&self, u: &Number, k: &Number) -> Number {
        let scalar = super::field::limbs_of(k);
        with_group!(self, g => {
            let f = g.f;
            let (x2, z2, _, _) = g.ladder(&f.enter(u), &scalar);
            match f.invert(&z2) {
                Some(i) => f.leave(&f.mul(&x2, &i)),
                None => f.leave(&f.zero()),
            }
        })
    }

    /// The short Weierstrass curve this one is isomorphic to, by
    /// `(x, y) -> (x / B + A / (3 B), y / B)`.
    pub fn to_weierstrass(&self) -> Weierstrass {
        let (a, b, p) = (&self.a, &self.b, &self.base.p);
        let i = |x: Number| x.rem_euclid(p).mod_inverse(p).unwrap_or_default();
        let inv_b = i(b.clone());
        let ib2 = (&inv_b * &inv_b).rem_euclid(p);
        let ib3 = (&ib2 * &inv_b).rem_euclid(p);
        let a2 = (a * a).rem_euclid(p);
        let a3 = (&a2 * a).rem_euclid(p);
        let wa = ((3u8 - &a2) * i(Number::new_from_u8(3, p.byte_order)) * ib2).rem_euclid(p);
        let wb =
            ((a3 * 2u8 - a * 9u8) * i(Number::new_from_u8(27, p.byte_order)) * ib3).rem_euclid(p);
        // A curve isomorphic to one that is not singular isn't, and `p` is a prime above 3.
        Weierstrass::new(&wa, &wb, p).unwrap_or_else(|_| unreachable!())
    }

    /// The point of `target`, from [`to_weierstrass`](Montgomery::to_weierstrass), that `point` of
    /// this curve corresponds to.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not the curve from [`to_weierstrass`](Montgomery::to_weierstrass).
    pub fn map_to_weierstrass<'t>(
        &self,
        target: &'t Weierstrass,
        point: &Point<'_, Montgomery>,
    ) -> Point<'t, Weierstrass> {
        let Some((x, y)) = point.coordinates() else {
            return target.identity();
        };
        let (a, b, p) = (&self.a, &self.b, &self.base.p);
        let inv_b = b.mod_inverse(p).unwrap_or_default();
        let three = Number::new_from_u8(3, p.byte_order);
        let shift = a * three.mod_inverse(p).unwrap_or_default();
        let wx = ((x + shift) * &inv_b).rem_euclid(p);
        let wy = (y * &inv_b).rem_euclid(p);
        target
            .point(&wx, &wy)
            .unwrap_or_else(|_| panic!("the target is not this curve's Weierstrass form"))
    }

    /// The twisted Edwards curve this one is birationally equivalent to, with
    /// `a = (A + 2) / B` and `d = (A - 2) / B`.
    ///
    /// # Errors
    ///
    /// [`CurveError::NotComplete`] when that curve's addition law is not complete.
    pub fn to_edwards(&self) -> Result<Edwards, CurveError> {
        let (a, b, p) = (&self.a, &self.b, &self.base.p);
        let inv_b = b.mod_inverse(p).unwrap_or_default();
        let ea = ((a + 2u8) * &inv_b).rem_euclid(p);
        let ed = ((a - 2u8) * &inv_b).rem_euclid(p);
        Edwards::new(&ea, &ed, p)
    }

    /// The point of `target`, from [`to_edwards`](Montgomery::to_edwards), that `point` of this
    /// curve corresponds to: `(x / y, (x - 1) / (x + 1))`, with the point at infinity going to the
    /// neutral element and `(0, 0)` to `(0, -1)`.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not the curve from [`to_edwards`](Montgomery::to_edwards).
    pub fn map_to_edwards<'t>(
        &self,
        target: &'t Edwards,
        point: &Point<'_, Montgomery>,
    ) -> Point<'t, Edwards> {
        let p = &self.base.p;
        let Some((x, y)) = point.coordinates() else {
            return target.identity();
        };
        let (ex, ey) = if x.is_zero() {
            (x, (-Number::new_from_u8(1, p.byte_order)).rem_euclid(p))
        } else {
            let inv_y = y.mod_inverse(p).unwrap_or_default();
            let inv = (&x + 1u8).mod_inverse(p).unwrap_or_default();
            let ex = (&x * inv_y).rem_euclid(p);
            let ey = ((x - 1u8) * inv).rem_euclid(p);
            (ex, ey)
        };
        target
            .point(&ex, &ey)
            .unwrap_or_else(|_| panic!("the target is not this curve's Edwards form"))
    }
}

/// The arithmetic of a curve over the field `F`.
pub(super) struct Mon<'a, F: Field> {
    f: &'a F,
    a: F::E,
    b: F::E,
    a24: F::E,
    a24_word: Option<u32>,
}

/// Runs `$body` with `$g` bound to the group of the curve `$c`, compiled for each field.
macro_rules! with_group {
    ($c:expr, $g:ident => $body:expr) => {
        with_prime!(&$c.base.prime, f => {
            let $g = Mon {
                f,
                a: f.load(&$c.a_limbs),
                b: f.load(&$c.b_limbs),
                a24: f.load(&$c.a24_limbs),
                a24_word: $c.a24_word,
            };
            $body
        })
    };
}
use with_group;

impl<F: Field> Mon<'_, F> {
    /// `x^3 + A x^2 + x`, which is `B y^2`.
    fn cubic(&self, x: &F::E) -> F::E {
        let f = self.f;
        f.mul(x, &f.add(&f.mul(x, &f.add(x, &self.a)), &f.one()))
    }

    /// The ladder from the `x` coordinate `x1`: the projective `x` of `k P` and of `(k + 1) P`.
    ///
    /// The two points are `R0` and `R1`, whose difference stays `P`; each step, for a bit from the
    /// top, doubles the one the bit picks and adds them into the other: one differential addition
    /// and doubling, 5M + 4S and a product by `(A + 2) / 4`. The bit picks by index, as a branch
    /// on it would be mispredicted every other step and moving the points costs more.
    fn ladder(&self, x1: &F::E, k: &[u64]) -> (F::E, F::E, F::E, F::E) {
        let f = self.f;
        let mut x = [f.one(), x1.clone()];
        let mut z = [f.zero(), f.one()];
        for t in (0..bit_length(k)).rev() {
            let p = ((k[t / 64] >> (t % 64)) & 1) as usize;
            let q = 1 - p;
            let a = f.add(&x[p], &z[p]);
            let aa = f.sqr(&a);
            let b = f.sub(&x[p], &z[p]);
            let bb = f.sqr(&b);
            let e = f.sub(&aa, &bb);
            let c = f.add(&x[q], &z[q]);
            let d = f.sub(&x[q], &z[q]);
            let da = f.mul(&d, &a);
            let cb = f.mul(&c, &b);
            x[q] = f.sqr(&f.add(&da, &cb));
            z[q] = f.mul(x1, &f.sqr(&f.sub(&da, &cb)));
            x[p] = f.mul(&aa, &bb);
            let scaled = match self.a24_word {
                Some(w) => f.mul_word(&e, u64::from(w), &self.a24),
                None => f.mul(&self.a24, &e),
            };
            z[p] = f.mul(&e, &f.add(&bb, &scaled));
        }
        let [x2, x3] = x;
        let [z2, z3] = z;
        (x2, z2, x3, z3)
    }
}

impl<F: Field> Group for Mon<'_, F> {
    jacobian_group!();

    // With `Z3 = 2 B Y Z`: `X3 = B L^2 - A Z3^2 - 8 B^2 X Y^2`, and so on, for the slope `L / Z3`,
    // `L = 3 X^2 + 2 A X Z^2 + Z^4`.
    fn double(&self, p: &Jacobian<F::E>) -> Jacobian<F::E> {
        let f = self.f;
        let xx = f.sqr(&p.x);
        let zz = f.sqr(&p.z);
        let yy = f.sqr(&p.y);
        let l = f.add(
            &f.add(&f.twice(&xx), &xx),
            &f.add(&f.mul(&f.twice(&self.a), &f.mul(&p.x, &zz)), &f.sqr(&zz)),
        );
        let z3 = f.mul(&f.twice(&f.mul(&p.y, &p.z)), &self.b);
        let byy = f.mul(&self.b, &yy);
        let t = f.twice(&f.twice(&f.mul(&self.b, &f.mul(&p.x, &byy))));
        let x3 = f.sub(
            &f.sub(&f.mul(&self.b, &f.sqr(&l)), &f.mul(&self.a, &f.sqr(&z3))),
            &f.twice(&t),
        );
        let b8 = f.twice(&f.twice(&f.twice(&f.mul(&self.b, &f.sqr(&byy)))));
        let y3 = f.sub(&f.mul(&l, &f.sub(&t, &x3)), &b8);
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    // With `Z3 = H Z1 Z2`: `X3 = B R^2 - A Z3^2 - H^2 (U1 + U2)`, for the slope `R / Z3`.
    fn add(&self, p: &Jacobian<F::E>, q: &Jacobian<F::E>) -> Jacobian<F::E> {
        let f = self.f;
        if f.is_zero(&p.z) {
            return q.clone();
        }
        if f.is_zero(&q.z) {
            return p.clone();
        }
        let z1z1 = f.sqr(&p.z);
        let z2z2 = f.sqr(&q.z);
        let u1 = f.mul(&p.x, &z2z2);
        let u2 = f.mul(&q.x, &z1z1);
        let s1 = f.mul(&f.mul(&p.y, &q.z), &z2z2);
        let s2 = f.mul(&f.mul(&q.y, &p.z), &z1z1);
        let z12 = f.mul(&p.z, &q.z);
        self.chord(&u1, &u2, &s1, &s2, &z12, p)
    }

    fn add_cached(&self, p: &Jacobian<F::E>, c: &Affine<F::E>) -> Jacobian<F::E> {
        let f = self.f;
        if c.infinity {
            return p.clone();
        }
        if f.is_zero(&p.z) {
            return self.of_affine(&c.x, &c.y);
        }
        let z1z1 = f.sqr(&p.z);
        let u2 = f.mul(&c.x, &z1z1);
        let s2 = f.mul(&f.mul(&c.y, &p.z), &z1z1);
        self.chord(&p.x, &u2, &p.y, &s2, &p.z, p)
    }

    /// `k p` by the ladder, then `y` by Okeya and Sakurai's recovery from `x` of `k p` and of
    /// `(k + 1) p`.
    fn mul(&self, p: &Jacobian<F::E>, k: &[u64], negative: bool) -> Jacobian<F::E> {
        let f = self.f;
        let bits = bit_length(k);
        if bits == 0 || f.is_zero(&p.z) {
            return self.identity();
        }
        let affine = normalize(f, core::slice::from_ref(p));
        let (x, y) = (&affine[0].x, &affine[0].y);
        if f.is_zero(y) {
            // A point of order two.
            return if k[0] & 1 == 0 {
                self.identity()
            } else {
                p.clone()
            };
        }
        let (x2, z2, x3, z3) = self.ladder(x, k);
        let r = if f.is_zero(&z2) {
            self.identity()
        } else if f.is_zero(&z3) {
            // `(k + 1) p` at infinity.
            self.neg(p)
        } else {
            let v1 = f.mul(x, &z2);
            let v2 = f.add(&x2, &v1);
            let v3 = f.mul(&f.sqr(&f.sub(&x2, &v1)), &x3);
            let v1 = f.mul(&f.twice(&self.a), &z2);
            let v2 = f.add(&v2, &v1);
            let v4 = f.add(&f.mul(x, &x2), &z2);
            let v2 = f.mul(&v2, &v4);
            let v1 = f.mul(&v1, &z2);
            let v2 = f.sub(&v2, &v1);
            let v2 = f.mul(&v2, &z3);
            let yy = f.sub(&v2, &v3);
            let v1 = f.mul(&f.mul(&f.mul(&f.twice(&self.b), y), &z2), &z3);
            // `(X : Y : Z)` for `x = X / Z`, `y = Y / Z`, in Jacobian form.
            let (xr, zr) = (f.mul(&v1, &x2), f.mul(&v1, &z2));
            Jacobian {
                x: f.mul(&xr, &zr),
                y: f.mul(&yy, &f.sqr(&zr)),
                z: zr,
            }
        };
        if negative { self.neg(&r) } else { r }
    }
}

impl<F: Field> Mon<'_, F> {
    /// The sum of two points with `x = U / Z^2`, `y = S / Z^3`, for `Z = z12`, or their double or
    /// infinity when the `x`s are equal.
    fn chord(
        &self,
        u1: &F::E,
        u2: &F::E,
        s1: &F::E,
        s2: &F::E,
        z12: &F::E,
        p: &Jacobian<F::E>,
    ) -> Jacobian<F::E> {
        let f = self.f;
        let h = f.sub(u2, u1);
        let r = f.sub(s2, s1);
        if f.is_zero(&h) {
            return if f.is_zero(&r) {
                self.double(p)
            } else {
                self.identity()
            };
        }
        let hh = f.sqr(&h);
        let z3 = f.mul(z12, &h);
        let x3 = f.sub(
            &f.sub(&f.mul(&self.b, &f.sqr(&r)), &f.mul(&self.a, &f.sqr(&z3))),
            &f.mul(&hh, &f.add(u1, u2)),
        );
        let y3 = f.sub(
            &f.mul(&r, &f.sub(&f.mul(u1, &hh), &x3)),
            &f.mul(s1, &f.mul(&h, &hh)),
        );
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }
}

impl Arithmetic for Montgomery {
    const COORDINATES: usize = 3;

    fn base(&self) -> &Base {
        &self.base
    }

    fn same_curve(&self, other: &Self) -> bool {
        self.base.p == other.base.p && self.a == other.a && self.b == other.b
    }

    fn satisfied(&self, x: &Number, y: &Number) -> bool {
        with_group!(self, g => {
            let f = g.f;
            f.equal(&f.mul(&g.b, &f.sqr(&f.enter(y))), &g.cubic(&f.enter(x)))
        })
    }

    fn lift_limbs(&self, x: &Number, odd: bool) -> Option<Vec<u64>> {
        with_group!(self, g => {
            let f = g.f;
            let ex = f.enter(x);
            let y = sqrt_ratio(f, self.base.sqrt_plan(), &g.cubic(&ex), &g.b)?;
            let y = choose_root(f, y, odd)?;
            let mut r = vec![0; 3 * f.limbs()];
            g.store(&g.of_affine(&ex, &y), &mut r);
            Some(r)
        })
    }

    fn encode(&self, xy: Option<(&Number, &Number)>, compressed: bool) -> Vec<u8> {
        sec1_encode(self.base.bytes(), xy, compressed)
    }

    fn decode(&self, bytes: &[u8]) -> Option<Vec<u64>> {
        sec1_decode(self, bytes)
    }

    super::curve::arithmetic!(with_group);
}

impl Curve for Montgomery {
    /// `16 (A^2 - 4) / B^6` modulo `p`, that of the curve's general Weierstrass equation.
    fn discriminant(&self) -> Number {
        let (a, b, p) = (&self.a, &self.b, &self.base.p);
        let b2 = (b * b).rem_euclid(p);
        let b6 = (&b2 * &b2 * &b2).rem_euclid(p);
        (((a * a) - 4u8) * 16u8 * b6.mod_inverse(p).unwrap_or_default()).rem_euclid(p)
    }

    /// `256 (A^2 - 3)^3 / (A^2 - 4)` modulo `p`.
    fn j_invariant(&self) -> Number {
        let (a, p) = (&self.a, &self.base.p);
        let a2 = (a * a).rem_euclid(p);
        let t = (&a2 + (p - 3u8)).rem_euclid(p);
        let d = ((&a2 + p) - 4u8).rem_euclid(p);
        (&t * &t * &t * 256u16 * d.mod_inverse(p).unwrap_or_default()).rem_euclid(p)
    }
}
