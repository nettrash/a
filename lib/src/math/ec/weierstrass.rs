//! Curves in short Weierstrass form, `y^2 = x^3 + a x + b`.

use super::curve::{Arithmetic, Curve, CurveError, choose_root, sec1_decode, sec1_encode};
use super::field::{Base, Field, invert_all, limbs_of, number_of, sqrt, with_prime};
use super::group::{self, Group, bit_length, odd_multiples, wnaf};
use crate::math::numbers::Number;

/// A curve `y^2 = x^3 + a x + b` over the field of a prime `p > 3`.
///
/// Its points are held in Jacobian coordinates, `x = X / Z^2`, `y = Y / Z^3`, so that adding them
/// takes no inversion, with doublings of their own for `a = 0`, as on secp256k1, and `a = -3`, as
/// on the NIST curves.
///
/// # Examples
///
/// ```
/// use a::math::ec::{Curve, Weierstrass};
/// use a::math::numbers::{ByteOrder, Number};
///
/// let n = |x| Number::new_from_u32(x, ByteOrder::BigEndian);
/// // y^2 = x^3 + 2 x + 3 over F_97, which has the point (3, 6).
/// let curve = Weierstrass::new(&n(2), &n(3), &n(97)).unwrap();
/// let p = curve.point(&n(3), &n(6)).unwrap();
/// let (x, y) = (&p + &p).coordinates().unwrap();
/// assert_eq!((x, y), (n(80), n(10)));
/// ```
pub struct Weierstrass {
    /// The field.
    base: Base,
    /// `a` below `p`.
    a: Number,
    /// `b` below `p`.
    b: Number,
    /// `a` in the field's form.
    a_limbs: Vec<u64>,
    /// `b` in the field's form.
    b_limbs: Vec<u64>,
    /// Which doubling `a` allows.
    shape: Shape,
    /// The curve's endomorphism, when one is known.
    endomorphism: Option<Endomorphism>,
}

/// An efficient endomorphism of a curve with `a = 0` of prime order `n`, `(x, y) -> (beta x, y)`,
/// which is the product by `lambda` on every point: a scalar `k` splits as `k1 + k2 lambda` modulo
/// `n` into two of half its length, after Gallant, Lambert and Vanstone, and `k P` takes half the
/// doublings, as `k1 P + k2 (beta x, y)`.
struct Endomorphism {
    /// `beta` in the field's form.
    beta: Vec<u64>,
    /// The order of the group.
    n: Number,
    /// Half of `n`, for rounding.
    half: Number,
    /// A short basis of the lattice of the pairs `(a, b)` with `a + b lambda = 0` modulo `n`:
    /// `(a1, b1)` and `(a2, b2)`.
    basis: [Number; 4],
}

impl Endomorphism {
    /// `k` as `k1 + k2 lambda` modulo `n`, for the two signed numbers of about half its length: the
    /// vector `(k, 0)` less the point of the lattice nearest to it, found by rounding its
    /// coordinates in the basis, which are `b2 k / n` and `-b1 k / n`.
    fn split(&self, k: &Number) -> (Number, Number) {
        let [a1, b1, a2, b2] = &self.basis;
        let round = |x: Number| (x + &self.half).div_euclid(&self.n);
        let k = k.rem_euclid(&self.n);
        let c1 = round(b2 * &k);
        let c2 = round(-(b1 * &k));
        let k1 = k - &c1 * a1 - &c2 * a2;
        let k2 = -(&c1 * b1) - &c2 * b2;
        (k1, k2)
    }
}

/// Which doubling a curve's `a` allows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Shape {
    /// `a = 0`.
    Zero,
    /// `a = -3`.
    MinusThree,
    /// Any other.
    General,
}

impl Weierstrass {
    /// The curve `y^2 = x^3 + a x + b` modulo `p`, in `p`'s byte order. `a` and `b` are reduced
    /// modulo `p`.
    ///
    /// # Errors
    ///
    /// [`CurveError::NotPrime`] when `p` is not a prime above 3, by
    /// [`is_probable_prime`](Number::is_probable_prime); [`CurveError::Singular`] when
    /// `4 a^3 + 27 b^2` is zero modulo `p`.
    pub fn new(a: &Number, b: &Number, p: &Number) -> Result<Weierstrass, CurveError> {
        let three = Number::new_from_u8(3, p.byte_order);
        if p.is_negative() || *p <= three || !p.is_probable_prime() {
            return Err(CurveError::NotPrime);
        }
        let a = a.rem_euclid(p);
        let b = b.rem_euclid(p);
        let d = (&a * &a * &a * 4u8 + &b * &b * 27u8).rem_euclid(p);
        if d.is_zero() {
            return Err(CurveError::Singular);
        }
        let base = Base::new(p);
        let n = with_prime!(&base.prime, f => f.limbs());
        let (mut a_limbs, mut b_limbs) = (vec![0; n], vec![0; n]);
        with_prime!(&base.prime, f => {
            f.store(&f.enter(&a), &mut a_limbs);
            f.store(&f.enter(&b), &mut b_limbs);
        });
        let shape = if a.is_zero() {
            Shape::Zero
        } else if &a + 3u8 == *p {
            Shape::MinusThree
        } else {
            Shape::General
        };
        Ok(Weierstrass {
            base,
            a,
            b,
            a_limbs,
            b_limbs,
            shape,
            endomorphism: None,
        })
    }

    /// This curve, of prime order `n`, with the endomorphism `(x, y) -> (beta x, y)` and the basis
    /// `(a1, b1, a2, b2)` of [`Endomorphism`]. The products of the points by a number then take it.
    pub(super) fn with_endomorphism(
        mut self,
        beta: &Number,
        n: &Number,
        basis: [Number; 4],
    ) -> Self {
        let mut limbs = vec![0; self.a_limbs.len()];
        with_prime!(&self.base.prime, f => f.store(&f.enter(beta), &mut limbs));
        self.endomorphism = Some(Endomorphism {
            beta: limbs,
            n: n.clone(),
            half: n >> 1usize,
            basis,
        });
        self
    }

    /// The coefficient `a`, below `p`.
    pub fn a(&self) -> &Number {
        &self.a
    }

    /// The coefficient `b`, below `p`.
    pub fn b(&self) -> &Number {
        &self.b
    }
}

/// A point in Jacobian coordinates: `Z = 0` at infinity.
#[derive(Clone)]
pub(super) struct Jacobian<E> {
    pub(super) x: E,
    pub(super) y: E,
    pub(super) z: E,
}

/// A point in affine coordinates, for additions with `Z = 1`.
#[derive(Clone)]
pub(super) struct Affine<E> {
    pub(super) x: E,
    pub(super) y: E,
    pub(super) infinity: bool,
}

/// The arithmetic of a curve over the field `F`.
pub(super) struct Wei<'a, F: Field> {
    f: &'a F,
    a: F::E,
    b: F::E,
    shape: Shape,
    endomorphism: Option<&'a Endomorphism>,
}

/// Runs `$body` with `$g` bound to the group of the curve `$c`, compiled for each field.
macro_rules! with_group {
    ($c:expr, $g:ident => $body:expr) => {
        with_prime!(&$c.base.prime, f => {
            let $g = Wei {
                f,
                a: f.load(&$c.a_limbs),
                b: f.load(&$c.b_limbs),
                shape: $c.shape,
                endomorphism: $c.endomorphism.as_ref(),
            };
            $body
        })
    };
}

impl<F: Field> Wei<'_, F> {
    /// `x^3 + a x + b`.
    fn right_side(&self, x: &F::E) -> F::E {
        let f = self.f;
        f.add(&f.mul(x, &f.add(&f.sqr(x), &self.a)), &self.b)
    }

    /// `k p` by the endomorphism: the odd multiples of `p` once, the same multiples of its image
    /// from them by a product each, and one run of doublings for the two half-length scalars.
    fn split_mul(&self, e: &Endomorphism, p: &Jacobian<F::E>, k: &Number) -> Jacobian<F::E> {
        let f = self.f;
        let (k1, k2) = e.split(k);
        let (l1, l2) = (limbs_of(&k1), limbs_of(&k2));
        let w = group::window(bit_length(&l1).max(bit_length(&l2)));
        let (d1, d2) = (wnaf(&l1, w), wnaf(&l2, w));
        let mut first = normalize(f, &odd_multiples(self, p, 1 << (w - 2)));
        let beta = f.load(&e.beta);
        let mut second: Vec<Affine<F::E>> = first
            .iter()
            .map(|a| Affine {
                x: f.mul(&a.x, &beta),
                y: a.y.clone(),
                infinity: a.infinity,
            })
            .collect();
        if k1.is_negative() {
            first = first.iter().map(|a| self.neg_cached(a)).collect();
        }
        if k2.is_negative() {
            second = second.iter().map(|a| self.neg_cached(a)).collect();
        }
        let mut r = self.identity();
        for i in (0..d1.len().max(d2.len())).rev() {
            r = self.double(&r);
            for (digits, table) in [(&d1, &first), (&d2, &second)] {
                let d = digits.get(i).copied().unwrap_or(0);
                if d > 0 {
                    r = self.add_cached(&r, &table[(d as usize - 1) / 2]);
                } else if d < 0 {
                    r = self.sub_cached(&r, &table[((-d) as usize - 1) / 2]);
                }
            }
        }
        r
    }
}

/// Puts each point of `p` in affine coordinates, with one inversion for all.
pub(super) fn normalize<F: Field>(f: &F, p: &[Jacobian<F::E>]) -> Vec<Affine<F::E>> {
    let mut inverses: Vec<F::E> = p.iter().map(|q| q.z.clone()).collect();
    invert_all(f, &mut inverses);
    p.iter()
        .zip(inverses.iter())
        .map(|(q, i)| {
            if f.is_zero(&q.z) {
                return Affine {
                    x: f.zero(),
                    y: f.zero(),
                    infinity: true,
                };
            }
            let ii = f.sqr(i);
            Affine {
                x: f.mul(&q.x, &ii),
                y: f.mul(&q.y, &f.mul(&ii, i)),
                infinity: false,
            }
        })
        .collect()
}

/// Whether two points in Jacobian coordinates are the same: `X1 Z2^2 = X2 Z1^2` and
/// `Y1 Z2^3 = Y2 Z1^3`.
pub(super) fn jacobian_equal<F: Field>(f: &F, p: &Jacobian<F::E>, q: &Jacobian<F::E>) -> bool {
    let (inf_p, inf_q) = (f.is_zero(&p.z), f.is_zero(&q.z));
    if inf_p || inf_q {
        return inf_p == inf_q;
    }
    let (zz1, zz2) = (f.sqr(&p.z), f.sqr(&q.z));
    f.equal(&f.mul(&p.x, &zz2), &f.mul(&q.x, &zz1))
        && f.equal(
            &f.mul(&p.y, &f.mul(&zz2, &q.z)),
            &f.mul(&q.y, &f.mul(&zz1, &p.z)),
        )
}

/// Implements the storing and the affine forms that the curves with Jacobian coordinates share.
macro_rules! jacobian_group {
    () => {
        type F = F;
        type Proj = Jacobian<F::E>;
        type Cached = Affine<F::E>;
        const CACHED: usize = 2;

        #[inline(always)]
        fn field(&self) -> &F {
            self.f
        }

        fn identity(&self) -> Jacobian<F::E> {
            let f = self.f;
            Jacobian {
                x: f.one(),
                y: f.one(),
                z: f.zero(),
            }
        }

        #[inline(always)]
        fn is_identity(&self, p: &Jacobian<F::E>) -> bool {
            self.f.is_zero(&p.z)
        }

        fn load(&self, x: &[u64]) -> Jacobian<F::E> {
            let (f, n) = (self.f, self.f.limbs());
            Jacobian {
                x: f.load(&x[..n]),
                y: f.load(&x[n..2 * n]),
                z: f.load(&x[2 * n..3 * n]),
            }
        }

        fn store(&self, p: &Jacobian<F::E>, r: &mut [u64]) {
            let (f, n) = (self.f, self.f.limbs());
            f.store(&p.x, &mut r[..n]);
            f.store(&p.y, &mut r[n..2 * n]);
            f.store(&p.z, &mut r[2 * n..3 * n]);
        }

        fn neg(&self, p: &Jacobian<F::E>) -> Jacobian<F::E> {
            Jacobian {
                x: p.x.clone(),
                y: self.f.neg(&p.y),
                z: p.z.clone(),
            }
        }

        fn equal(&self, p: &Jacobian<F::E>, q: &Jacobian<F::E>) -> bool {
            jacobian_equal(self.f, p, q)
        }

        fn cache(&self, p: &[Jacobian<F::E>]) -> Vec<Affine<F::E>> {
            normalize(self.f, p)
        }

        fn neg_cached(&self, c: &Affine<F::E>) -> Affine<F::E> {
            Affine {
                x: c.x.clone(),
                y: self.f.neg(&c.y),
                infinity: c.infinity,
            }
        }

        fn store_cached(&self, c: &Affine<F::E>, r: &mut [u64]) {
            let (f, n) = (self.f, self.f.limbs());
            f.store(&c.x, &mut r[..n]);
            f.store(&c.y, &mut r[n..2 * n]);
            r[2 * n] = u64::from(c.infinity);
        }

        fn load_cached(&self, x: &[u64]) -> Affine<F::E> {
            let (f, n) = (self.f, self.f.limbs());
            Affine {
                x: f.load(&x[..n]),
                y: f.load(&x[n..2 * n]),
                infinity: x[2 * n] != 0,
            }
        }

        fn of_affine(&self, x: &F::E, y: &F::E) -> Jacobian<F::E> {
            Jacobian {
                x: x.clone(),
                y: y.clone(),
                z: self.f.one(),
            }
        }

        fn affine_all(
            &self,
            p: &[Jacobian<F::E>],
        ) -> Vec<Option<$crate::math::ec::group::Coordinates<F>>> {
            normalize(self.f, p)
                .into_iter()
                .map(|a| (!a.infinity).then_some((a.x, a.y)))
                .collect()
        }
    };
}
pub(super) use jacobian_group;

impl<F: Field> Group for Wei<'_, F> {
    jacobian_group!();

    fn double(&self, p: &Jacobian<F::E>) -> Jacobian<F::E> {
        let f = self.f;
        let z3 = f.twice(&f.mul(&p.y, &p.z));
        let (x3, y3) = match self.shape {
            // dbl-2009-l: 2M + 5S.
            Shape::Zero => {
                let a = f.sqr(&p.x);
                let b = f.sqr(&p.y);
                let c = f.sqr(&b);
                let d = f.twice(&f.sub(&f.sub(&f.sqr(&f.add(&p.x, &b)), &a), &c));
                let e = f.add(&f.twice(&a), &a);
                let x3 = f.sub(&f.sqr(&e), &f.twice(&d));
                let c8 = f.twice(&f.twice(&f.twice(&c)));
                (x3.clone(), f.sub(&f.mul(&e, &f.sub(&d, &x3)), &c8))
            }
            // dbl-2001-b: 3M + 5S.
            Shape::MinusThree => {
                let delta = f.sqr(&p.z);
                let gamma = f.sqr(&p.y);
                let beta = f.mul(&p.x, &gamma);
                let t = f.mul(&f.sub(&p.x, &delta), &f.add(&p.x, &delta));
                let alpha = f.add(&f.twice(&t), &t);
                let b4 = f.twice(&f.twice(&beta));
                let x3 = f.sub(&f.sqr(&alpha), &f.twice(&b4));
                let g8 = f.twice(&f.twice(&f.twice(&f.sqr(&gamma))));
                (x3.clone(), f.sub(&f.mul(&alpha, &f.sub(&b4, &x3)), &g8))
            }
            // dbl-2007-bl.
            Shape::General => {
                let xx = f.sqr(&p.x);
                let yy = f.sqr(&p.y);
                let yyyy = f.sqr(&yy);
                let zz = f.sqr(&p.z);
                let s = f.twice(&f.sub(&f.sub(&f.sqr(&f.add(&p.x, &yy)), &xx), &yyyy));
                let m = f.add(&f.add(&f.twice(&xx), &xx), &f.mul(&self.a, &f.sqr(&zz)));
                let x3 = f.sub(&f.sqr(&m), &f.twice(&s));
                let y8 = f.twice(&f.twice(&f.twice(&yyyy)));
                (x3.clone(), f.sub(&f.mul(&m, &f.sub(&s, &x3)), &y8))
            }
        };
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    fn mul(&self, p: &Jacobian<F::E>, k: &[u64], negative: bool) -> Jacobian<F::E> {
        match self.endomorphism {
            Some(e) if !self.is_identity(p) => {
                let scalar = number_of(k, e.n.byte_order);
                let r = self.split_mul(e, p, &scalar);
                if negative { self.neg(&r) } else { r }
            }
            _ => group::mul(self, p, k, negative),
        }
    }

    // add-2007-bl: 11M + 5S.
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
        let h = f.sub(&u2, &u1);
        let r = f.twice(&f.sub(&s2, &s1));
        if f.is_zero(&h) {
            return if f.is_zero(&r) {
                self.double(p)
            } else {
                self.identity()
            };
        }
        let i = f.sqr(&f.twice(&h));
        let j = f.mul(&h, &i);
        let v = f.mul(&u1, &i);
        let x3 = f.sub(&f.sub(&f.sqr(&r), &j), &f.twice(&v));
        let y3 = f.sub(&f.mul(&r, &f.sub(&v, &x3)), &f.twice(&f.mul(&s1, &j)));
        let z3 = f.mul(&f.sub(&f.sub(&f.sqr(&f.add(&p.z, &q.z)), &z1z1), &z2z2), &h);
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    // madd-2007-bl: 7M + 4S.
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
        let h = f.sub(&u2, &p.x);
        let r = f.twice(&f.sub(&s2, &p.y));
        if f.is_zero(&h) {
            return if f.is_zero(&r) {
                self.double(p)
            } else {
                self.identity()
            };
        }
        let hh = f.sqr(&h);
        let i = f.twice(&f.twice(&hh));
        let j = f.mul(&h, &i);
        let v = f.mul(&p.x, &i);
        let x3 = f.sub(&f.sub(&f.sqr(&r), &j), &f.twice(&v));
        let y3 = f.sub(&f.mul(&r, &f.sub(&v, &x3)), &f.twice(&f.mul(&p.y, &j)));
        let z3 = f.sub(&f.sub(&f.sqr(&f.add(&p.z, &h)), &z1z1), &hh);
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }
}

impl Arithmetic for Weierstrass {
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
            f.equal(&f.sqr(&f.enter(y)), &g.right_side(&f.enter(x)))
        })
    }

    fn lift_limbs(&self, x: &Number, odd: bool) -> Option<Vec<u64>> {
        with_group!(self, g => {
            let f = g.f;
            let ex = f.enter(x);
            let plan = self.base.sqrt_plan();
            let y = sqrt(f, plan, &g.right_side(&ex))?;
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

impl Curve for Weierstrass {
    /// `-16 (4 a^3 + 27 b^2)` modulo `p`.
    fn discriminant(&self) -> Number {
        let (a, b, p) = (&self.a, &self.b, &self.base.p);
        (-((a * a * a * 4u8 + b * b * 27u8) * 16u8)).rem_euclid(p)
    }

    /// `1728 (4 a^3) / (4 a^3 + 27 b^2)` modulo `p`.
    fn j_invariant(&self) -> Number {
        let (a, b, p) = (&self.a, &self.b, &self.base.p);
        let four = (a * a * a * 4u8).rem_euclid(p);
        let d = (&four + b * b * 27u8).rem_euclid(p);
        // The curve is not singular, so `d` has an inverse.
        let i = d.mod_inverse(p).unwrap_or_default();
        (four * i * 1728u16).rem_euclid(p)
    }
}
