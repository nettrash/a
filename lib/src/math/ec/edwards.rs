//! Twisted Edwards curves, `a x^2 + y^2 = 1 + d x^2 y^2`.

use super::curve::{Arithmetic, Curve, CurveError, choose_root};
use super::field::{Base, Field, invert_all, sqrt_ratio, with_prime};
use super::group::{Coordinates, Group};
use super::montgomery::Montgomery;
use super::point::Point;
use crate::math::numbers::{ByteOrder, Number};

/// A twisted Edwards curve `a x^2 + y^2 = 1 + d x^2 y^2` over the field of a prime `p > 3`, with
/// `a` a square and `d` not, which makes its addition law complete: one formula for every pair of
/// points, doublings and the neutral element `(0, 1)` included.
///
/// Its points are held in extended coordinates, `x = X / Z`, `y = Y / Z`, `x y = T / Z`, the
/// fastest known form of the group, with a formula of its own for `a = -1`, as on Ed25519.
///
/// # Examples
///
/// ```
/// use a::math::ec::{Curve, Edwards};
/// use a::math::numbers::{ByteOrder, Number};
///
/// let n = |x| Number::new_from_u32(x, ByteOrder::BigEndian);
/// // x^2 + y^2 = 1 + 2 x^2 y^2 over F_13, where 1 is a square and 2 is not.
/// let curve = Edwards::new(&n(1), &n(2), &n(13)).unwrap();
/// let o = curve.identity();
/// assert_eq!(o.coordinates(), Some((n(0), n(1))));
/// ```
pub struct Edwards {
    /// The field.
    base: Base,
    /// `a` below `p`.
    a: Number,
    /// `d` below `p`.
    d: Number,
    /// `a` in the field's form.
    a_limbs: Vec<u64>,
    /// `d` in the field's form.
    d_limbs: Vec<u64>,
    /// Whether `a = -1`.
    minus_one: bool,
}

impl Edwards {
    /// The curve `a x^2 + y^2 = 1 + d x^2 y^2` modulo `p`, in `p`'s byte order. `a` and `d` are
    /// reduced modulo `p`.
    ///
    /// # Errors
    ///
    /// [`CurveError::NotPrime`] when `p` is not a prime above 3, by
    /// [`is_probable_prime`](Number::is_probable_prime); [`CurveError::Singular`] when `a`, `d` or
    /// `a - d` is zero modulo `p`; [`CurveError::NotComplete`] when `a` is not a square or `d` is.
    pub fn new(a: &Number, d: &Number, p: &Number) -> Result<Edwards, CurveError> {
        let three = Number::new_from_u8(3, p.byte_order);
        if p.is_negative() || *p <= three || !p.is_probable_prime() {
            return Err(CurveError::NotPrime);
        }
        let a = a.rem_euclid(p);
        let d = d.rem_euclid(p);
        if a.is_zero() || d.is_zero() || a == d {
            return Err(CurveError::Singular);
        }
        if a.jacobi(p) != 1 || d.jacobi(p) != -1 {
            return Err(CurveError::NotComplete);
        }
        let base = Base::new(p);
        let n = with_prime!(&base.prime, f => f.limbs());
        let (mut a_limbs, mut d_limbs) = (vec![0; n], vec![0; n]);
        with_prime!(&base.prime, f => {
            f.store(&f.enter(&a), &mut a_limbs);
            f.store(&f.enter(&d), &mut d_limbs);
        });
        let minus_one = &a + 1u8 == *p;
        Ok(Edwards {
            base,
            a,
            d,
            a_limbs,
            d_limbs,
            minus_one,
        })
    }

    /// The coefficient `a`, below `p`.
    pub fn a(&self) -> &Number {
        &self.a
    }

    /// The coefficient `d`, below `p`.
    pub fn d(&self) -> &Number {
        &self.d
    }

    /// The Montgomery curve this one is birationally equivalent to, with
    /// `A = 2 (a + d) / (a - d)` and `B = 4 / (a - d)`.
    pub fn to_montgomery(&self) -> Montgomery {
        let (a, d, p) = (&self.a, &self.d, &self.base.p);
        let i = (a - d).rem_euclid(p).mod_inverse(p).unwrap_or_default();
        let ma = ((a + d) * 2u8 * &i).rem_euclid(p);
        let mb = (&i * 4u8).rem_euclid(p);
        // The curve is not singular, and so neither is this one; `p` was checked.
        Montgomery::new(&ma, &mb, p).unwrap_or_else(|_| unreachable!())
    }

    /// The point of `target`, from [`to_montgomery`](Edwards::to_montgomery), that `point` of this
    /// curve corresponds to: `((1 + y) / (1 - y), (1 + y) / ((1 - y) x))`, with the neutral element
    /// going to the point at infinity and `(0, -1)` to `(0, 0)`.
    ///
    /// # Panics
    ///
    /// Panics if `target` is not the curve from [`to_montgomery`](Edwards::to_montgomery).
    pub fn map_to_montgomery<'t>(
        &self,
        target: &'t Montgomery,
        point: &Point<'_, Edwards>,
    ) -> Point<'t, Montgomery> {
        let p = &self.base.p;
        let Some((x, y)) = point.coordinates() else {
            return target.identity();
        };
        if x.is_zero() {
            if y == Number::new_from_u8(1, p.byte_order) {
                return target.identity();
            }
            let zero = Number::new_from_u8(0, p.byte_order);
            return target
                .point(&zero, &zero)
                .unwrap_or_else(|_| panic!("the target is not this curve's Montgomery form"));
        }
        let inv = (1u8 - &y).rem_euclid(p).mod_inverse(p).unwrap_or_default();
        let u = ((&y + 1u8) * inv).rem_euclid(p);
        let v = (&u * x.mod_inverse(p).unwrap_or_default()).rem_euclid(p);
        target
            .point(&u, &v)
            .unwrap_or_else(|_| panic!("the target is not this curve's Montgomery form"))
    }

    /// The bytes of RFC 8032's encoding of an element: enough for the prime and one bit more.
    fn encoded_len(&self) -> usize {
        self.base.p.bits() as usize / 8 + 1
    }
}

/// A point in extended coordinates.
#[derive(Clone)]
pub(super) struct Extended<E> {
    x: E,
    y: E,
    z: E,
    t: E,
}

/// A point prepared to be added: for `a = -1`, `(Y + X, Y - X, 2 Z, 2 d T)`; otherwise
/// `(X, Y, Z, d T)`.
#[derive(Clone)]
pub(super) struct Cached<E> {
    u: E,
    v: E,
    z: E,
    t: E,
}

/// A doubling without its `T`, and the two factors of `T`.
struct Doubled<E> {
    x: E,
    y: E,
    z: E,
    e: E,
    h: E,
}

/// The arithmetic of a curve over the field `F`.
pub(super) struct Edw<'a, F: Field> {
    f: &'a F,
    a: F::E,
    d: F::E,
    minus_one: bool,
}

/// Runs `$body` with `$g` bound to the group of the curve `$c`, compiled for each field.
macro_rules! with_group {
    ($c:expr, $g:ident => $body:expr) => {
        with_prime!(&$c.base.prime, f => {
            let $g = Edw {
                f,
                a: f.load(&$c.a_limbs),
                d: f.load(&$c.d_limbs),
                minus_one: $c.minus_one,
            };
            $body
        })
    };
}

impl<F: Field> Edw<'_, F> {
    /// The `X`, `Y`, `Z` of `2 p`, and the two factors of `T`: dbl-2008-hwcd without its `T`, 3M +
    /// 4S, which is all a run of doublings needs but for the last.
    #[inline]
    fn double_parts(&self, x: &F::E, y: &F::E, z: &F::E) -> Doubled<F::E> {
        let f = self.f;
        let a = f.sqr(x);
        let b = f.sqr(y);
        let c = f.twice(&f.sqr(z));
        let d = if self.minus_one {
            f.neg(&a)
        } else {
            f.mul(&self.a, &a)
        };
        let e = f.sub(&f.sub(&f.sqr(&f.add(x, y)), &a), &b);
        let g = f.add(&d, &b);
        let ff = f.sub(&g, &c);
        let h = f.sub(&d, &b);
        Doubled {
            x: f.mul(&e, &ff),
            y: f.mul(&g, &h),
            z: f.mul(&ff, &g),
            e,
            h,
        }
    }

    /// The point from its cached form's four products: `E`, `F`, `G`, `H`.
    #[inline(always)]
    fn combine(&self, e: &F::E, ff: &F::E, g: &F::E, h: &F::E) -> Extended<F::E> {
        let f = self.f;
        Extended {
            x: f.mul(e, ff),
            y: f.mul(g, h),
            z: f.mul(ff, g),
            t: f.mul(e, h),
        }
    }
}

impl<F: Field> Group for Edw<'_, F> {
    type F = F;
    type Proj = Extended<F::E>;
    type Cached = Cached<F::E>;
    const CACHED: usize = 4;

    #[inline(always)]
    fn field(&self) -> &F {
        self.f
    }

    fn identity(&self) -> Extended<F::E> {
        let f = self.f;
        Extended {
            x: f.zero(),
            y: f.one(),
            z: f.one(),
            t: f.zero(),
        }
    }

    fn is_identity(&self, p: &Extended<F::E>) -> bool {
        let f = self.f;
        f.is_zero(&p.x) && f.equal(&p.y, &p.z)
    }

    fn load(&self, x: &[u64]) -> Extended<F::E> {
        let (f, n) = (self.f, self.f.limbs());
        Extended {
            x: f.load(&x[..n]),
            y: f.load(&x[n..2 * n]),
            z: f.load(&x[2 * n..3 * n]),
            t: f.load(&x[3 * n..4 * n]),
        }
    }

    fn store(&self, p: &Extended<F::E>, r: &mut [u64]) {
        let (f, n) = (self.f, self.f.limbs());
        f.store(&p.x, &mut r[..n]);
        f.store(&p.y, &mut r[n..2 * n]);
        f.store(&p.z, &mut r[2 * n..3 * n]);
        f.store(&p.t, &mut r[3 * n..4 * n]);
    }

    fn neg(&self, p: &Extended<F::E>) -> Extended<F::E> {
        let f = self.f;
        Extended {
            x: f.neg(&p.x),
            y: p.y.clone(),
            z: p.z.clone(),
            t: f.neg(&p.t),
        }
    }

    fn double(&self, p: &Extended<F::E>) -> Extended<F::E> {
        let d = self.double_parts(&p.x, &p.y, &p.z);
        Extended {
            t: self.f.mul(&d.e, &d.h),
            x: d.x,
            y: d.y,
            z: d.z,
        }
    }

    fn double_n(&self, p: &mut Extended<F::E>, n: usize) {
        if n == 0 {
            return;
        }
        let (mut x, mut y, mut z) = (p.x.clone(), p.y.clone(), p.z.clone());
        for _ in 1..n {
            let d = self.double_parts(&x, &y, &z);
            (x, y, z) = (d.x, d.y, d.z);
        }
        let d = self.double_parts(&x, &y, &z);
        *p = Extended {
            t: self.f.mul(&d.e, &d.h),
            x: d.x,
            y: d.y,
            z: d.z,
        };
    }

    fn add(&self, p: &Extended<F::E>, q: &Extended<F::E>) -> Extended<F::E> {
        self.add_cached(p, &self.cache_one(q))
    }

    fn equal(&self, p: &Extended<F::E>, q: &Extended<F::E>) -> bool {
        let f = self.f;
        f.equal(&f.mul(&p.x, &q.z), &f.mul(&q.x, &p.z))
            && f.equal(&f.mul(&p.y, &q.z), &f.mul(&q.y, &p.z))
    }

    fn cache(&self, p: &[Extended<F::E>]) -> Vec<Cached<F::E>> {
        p.iter().map(|q| self.cache_one(q)).collect()
    }

    // add-2008-hwcd-3 for `a = -1`, 8M; add-2008-hwcd otherwise, 9M and a product by `a`.
    fn add_cached(&self, p: &Extended<F::E>, c: &Cached<F::E>) -> Extended<F::E> {
        let f = self.f;
        if self.minus_one {
            let a = f.mul(&f.sub(&p.y, &p.x), &c.v);
            let b = f.mul(&f.add(&p.y, &p.x), &c.u);
            let cc = f.mul(&p.t, &c.t);
            let d = f.mul(&p.z, &c.z);
            self.combine(
                &f.sub(&b, &a),
                &f.sub(&d, &cc),
                &f.add(&d, &cc),
                &f.add(&b, &a),
            )
        } else {
            let a = f.mul(&p.x, &c.u);
            let b = f.mul(&p.y, &c.v);
            let cc = f.mul(&p.t, &c.t);
            let d = f.mul(&p.z, &c.z);
            let e = f.sub(
                &f.sub(&f.mul(&f.add(&p.x, &p.y), &f.add(&c.u, &c.v)), &a),
                &b,
            );
            self.combine(
                &e,
                &f.sub(&d, &cc),
                &f.add(&d, &cc),
                &f.sub(&b, &f.mul(&self.a, &a)),
            )
        }
    }

    fn neg_cached(&self, c: &Cached<F::E>) -> Cached<F::E> {
        let f = self.f;
        if self.minus_one {
            Cached {
                u: c.v.clone(),
                v: c.u.clone(),
                z: c.z.clone(),
                t: f.neg(&c.t),
            }
        } else {
            Cached {
                u: f.neg(&c.u),
                v: c.v.clone(),
                z: c.z.clone(),
                t: f.neg(&c.t),
            }
        }
    }

    fn store_cached(&self, c: &Cached<F::E>, r: &mut [u64]) {
        let (f, n) = (self.f, self.f.limbs());
        f.store(&c.u, &mut r[..n]);
        f.store(&c.v, &mut r[n..2 * n]);
        f.store(&c.z, &mut r[2 * n..3 * n]);
        f.store(&c.t, &mut r[3 * n..4 * n]);
    }

    fn load_cached(&self, x: &[u64]) -> Cached<F::E> {
        let (f, n) = (self.f, self.f.limbs());
        Cached {
            u: f.load(&x[..n]),
            v: f.load(&x[n..2 * n]),
            z: f.load(&x[2 * n..3 * n]),
            t: f.load(&x[3 * n..4 * n]),
        }
    }

    fn of_affine(&self, x: &F::E, y: &F::E) -> Extended<F::E> {
        let f = self.f;
        Extended {
            x: x.clone(),
            y: y.clone(),
            z: f.one(),
            t: f.mul(x, y),
        }
    }

    fn affine_all(&self, p: &[Extended<F::E>]) -> Vec<Option<Coordinates<F>>> {
        let f = self.f;
        let mut z: Vec<F::E> = p.iter().map(|q| q.z.clone()).collect();
        invert_all(f, &mut z);
        p.iter()
            .zip(z.iter())
            .map(|(q, i)| Some((f.mul(&q.x, i), f.mul(&q.y, i))))
            .collect()
    }
}

impl<F: Field> Edw<'_, F> {
    /// `q` prepared to be added.
    fn cache_one(&self, q: &Extended<F::E>) -> Cached<F::E> {
        let f = self.f;
        if self.minus_one {
            Cached {
                u: f.add(&q.y, &q.x),
                v: f.sub(&q.y, &q.x),
                z: f.twice(&q.z),
                t: f.twice(&f.mul(&q.t, &self.d)),
            }
        } else {
            Cached {
                u: q.x.clone(),
                v: q.y.clone(),
                z: q.z.clone(),
                t: f.mul(&q.t, &self.d),
            }
        }
    }

    /// Whether `(x, y)` satisfies the equation.
    fn on_curve(&self, x: &F::E, y: &F::E) -> bool {
        let f = self.f;
        let (xx, yy) = (f.sqr(x), f.sqr(y));
        f.equal(
            &f.add(&f.mul(&self.a, &xx), &yy),
            &f.add(&f.one(), &f.mul(&self.d, &f.mul(&xx, &yy))),
        )
    }

    /// An `x` for the given `y`: its square is `(y^2 - 1) / (d y^2 - a)`, whose denominator isn't
    /// zero, as `a` is a square and `d` is not.
    fn x_of(&self, y: &F::E, base: &Base) -> Option<F::E> {
        let f = self.f;
        let yy = f.sqr(y);
        let u = f.sub(&yy, &f.one());
        let v = f.sub(&f.mul(&self.d, &yy), &self.a);
        sqrt_ratio(f, base.sqrt_plan(), &u, &v)
    }
}

impl Arithmetic for Edwards {
    const COORDINATES: usize = 4;

    fn base(&self) -> &Base {
        &self.base
    }

    fn same_curve(&self, other: &Self) -> bool {
        self.base.p == other.base.p && self.a == other.a && self.d == other.d
    }

    fn satisfied(&self, x: &Number, y: &Number) -> bool {
        with_group!(self, g => g.on_curve(&g.f.enter(x), &g.f.enter(y)))
    }

    fn lift_limbs(&self, y: &Number, odd: bool) -> Option<Vec<u64>> {
        with_group!(self, g => {
            let f = g.f;
            let ey = f.enter(y);
            let x = choose_root(f, g.x_of(&ey, &self.base)?, odd)?;
            let mut r = vec![0; 4 * f.limbs()];
            g.store(&g.of_affine(&x, &ey), &mut r);
            Some(r)
        })
    }

    fn encode(&self, xy: Option<(&Number, &Number)>, compressed: bool) -> Vec<u8> {
        let len = self.encoded_len();
        let le = |n: &Number| {
            let mut b = super::curve::fixed_be(n, len);
            b.reverse();
            b
        };
        let Some((x, y)) = xy else {
            return Vec::new();
        };
        if compressed {
            let mut out = le(y);
            out[len - 1] |= u8::from(x.bit(0)) << 7;
            out
        } else {
            let mut out = le(x);
            out.extend_from_slice(&le(y));
            out
        }
    }

    fn decode(&self, bytes: &[u8]) -> Option<Vec<u64>> {
        let len = self.encoded_len();
        let order = self.base.p.byte_order;
        let number = |b: &[u8]| {
            let mut n = Number::new(b.to_vec(), ByteOrder::LittleEndian);
            if order == ByteOrder::BigEndian {
                n.value.reverse();
                n.byte_order = order;
            }
            (n < self.base.p).then_some(n)
        };
        if bytes.len() == len {
            let odd = bytes[len - 1] >> 7 != 0;
            let mut b = bytes.to_vec();
            b[len - 1] &= 0x7f;
            self.lift_limbs(&number(&b)?, odd)
        } else if bytes.len() == 2 * len {
            let (x, y) = (number(&bytes[..len])?, number(&bytes[len..])?);
            self.satisfied(&x, &y).then(|| self.affine_limbs(&x, &y))
        } else {
            None
        }
    }

    super::curve::arithmetic!(with_group);
}

impl Curve for Edwards {
    /// `a d (a - d)^4 / 16` modulo `p`, the discriminant of the curve's Montgomery form.
    fn discriminant(&self) -> Number {
        let (a, d, p) = (&self.a, &self.d, &self.base.p);
        let m = (a - d).rem_euclid(p);
        let m2 = (&m * &m).rem_euclid(p);
        let sixteen = Number::new_from_u8(16, p.byte_order);
        (a * d * &m2 * &m2 * sixteen.mod_inverse(p).unwrap_or_default()).rem_euclid(p)
    }

    /// `16 (a^2 + 14 a d + d^2)^3 / (a d (a - d)^4)` modulo `p`.
    fn j_invariant(&self) -> Number {
        let (a, d, p) = (&self.a, &self.d, &self.base.p);
        let t = (a * a + a * d * 14u8 + d * d).rem_euclid(p);
        let m = (a - d).rem_euclid(p);
        let m2 = (&m * &m).rem_euclid(p);
        let den = (a * d * &m2 * &m2).rem_euclid(p);
        (&t * &t * &t * 16u8 * den.mod_inverse(p).unwrap_or_default()).rem_euclid(p)
    }
}
