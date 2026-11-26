//! Hashing to the pairing's groups on the engine's arithmetic: RFC 9380's maps, the same points
//! as [`hash`](super::super::hash) defines, computed with fewer inversions.
//!
//! * `hash_to_field` reduces each piece of `L` bytes, read in big-endian order, by one Montgomery
//!   reduction of the double-width number and one product by `R³`: `REDC(t)·R³/R = tR`, the
//!   element `t mod p` in Montgomery's form.
//! * The simplified SWU map over `F_p` is the straight-line form of appendix F.2 with the
//!   `sqrt_ratio` of F.2.1.2 for `p ≡ 3 (mod 4)`: one exponentiation and no inversion; its point
//!   stays a fraction `(xₙ/x_d, y)`, which the isogeny takes homogeneously, so that a map to the
//!   curve gives Jacobian coordinates without inverting anything.
//! * The simplified SWU map over `F_p²` and the Shallue–van de Woestijne map keep the plain forms
//!   of section 6, on the engine's square roots.
//! * `hash_to_curve` applies the isogeny once, to the sum of the two points on `E′` (section
//!   6.6.3 allows it, the isogeny being a homomorphism).

use super::fp::Wide;
use super::group::{Coord, Jac};
use super::{Engine, Fp, Fp2};

/// The element `t mod p` of a big-endian `okm` of at most `16 N` bytes.
pub(crate) fn from_okm<E: Engine<N>, const N: usize>(okm: &[u8]) -> Fp<E, N> {
    let mut t: Wide = [0u64; 12];
    for (i, chunk) in okm.rchunks(8).enumerate() {
        let mut w = [0u8; 8];
        w[8 - chunk.len()..].copy_from_slice(chunk);
        t[i] = u64::from_be_bytes(w);
    }
    // t < 2^(8L) ≤ pR, as L ≤ ⌈(log p + 128)/8⌉ for both curves.
    let r3 = &E::tables().r3;
    Fp::redc(&t).mul(r3)
}

/// The simplified SWU map to `E′: y² = x³ + A·x + B` over `F_p`, `p ≡ 3 (mod 4)`.
pub(crate) struct SswuFp<E, const N: usize> {
    pub(crate) a: Fp<E, N>,
    pub(crate) b: Fp<E, N>,
    pub(crate) z: Fp<E, N>,
    /// `√(−Z)`.
    c2: Fp<E, N>,
    /// `(p − 3)/4`.
    c1: [u64; N],
}

impl<E: Engine<N>, const N: usize> SswuFp<E, N> {
    pub(crate) fn new(a: Fp<E, N>, b: Fp<E, N>, z: Fp<E, N>) -> Self {
        let c2 = z
            .neg()
            .sqrt()
            .expect("−Z is a square: Z is not, and p ≡ 3 (mod 4)");
        // (p − 3)/4 = (p + 1)/4 − 1.
        let mut c1 = E::tables().sqrt_exp;
        let mut i = 0;
        loop {
            let (v, borrow) = c1[i].overflowing_sub(1);
            c1[i] = v;
            if !borrow {
                break;
            }
            i += 1;
        }
        SswuFp { a, b, z, c2, c1 }
    }

    /// `(isQR, √(u/v) or √(Z u/v))`, appendix F.2.1.2.
    fn sqrt_ratio(&self, u: &Fp<E, N>, v: &Fp<E, N>) -> (bool, Fp<E, N>) {
        let tv1 = v.square();
        let tv2 = u.mul(v);
        let tv1 = tv1.mul(&tv2);
        let y1 = tv1.pow(&self.c1).mul(&tv2);
        let y2 = y1.mul(&self.c2);
        let is_qr = y1.square().mul(v) == *u;
        (is_qr, if is_qr { y1 } else { y2 })
    }

    /// The point `(xₙ / x_d, y)` of `E′` that `u` maps to.
    pub(crate) fn map(&self, u: &Fp<E, N>) -> (Fp<E, N>, Fp<E, N>, Fp<E, N>) {
        let one = Fp::one();
        let tv1 = self.z.mul(&u.square());
        let mut tv2 = tv1.square();
        tv2 = tv2.add(&tv1);
        let tv3 = self.b.mul(&tv2.add(&one));
        let tv4 = if tv2.is_zero() { self.z } else { tv2.neg() };
        let tv4 = self.a.mul(&tv4);
        let mut t2 = tv3.square();
        let mut tv6 = tv4.square();
        let tv5 = self.a.mul(&tv6);
        t2 = t2.add(&tv5).mul(&tv3);
        tv6 = tv6.mul(&tv4);
        t2 = t2.add(&self.b.mul(&tv6));
        let x = tv1.mul(&tv3);
        let (is_qr, y1) = self.sqrt_ratio(&t2, &tv6);
        let mut y = tv1.mul(u).mul(&y1);
        let xn = if is_qr { tv3 } else { x };
        if is_qr {
            y = y1;
        }
        if u.is_odd() != y.is_odd() {
            y = y.neg();
        }
        (xn, tv4, y)
    }
}

/// Horner's rule for `Σ cᵢ xⁱ` homogenised: `Σ cᵢ xₙⁱ x_d^(d − i)` for the degree `d` given, with
/// the powers of `x_d` passed in.
fn horner<C: Coord>(c: &[C], xn: &C, xd_pows: &[C], d: usize) -> C {
    // Σ cᵢ xₙⁱ x_d^(d−i), by Horner on xₙ with the powers of x_d.
    let k = c.len() - 1;
    let mut acc = c[k];
    for i in (0..k).rev() {
        acc = acc.mul(xn).add(&c[i].mul(&xd_pows[k - i]));
    }
    // Now acc = Σ cᵢ xₙⁱ x_d^(k−i); scale to degree d.
    if d > k { acc.mul(&xd_pows[d - k]) } else { acc }
}

/// An isogeny `(x, y) ↦ (x_num(x)/x_den(x), y·y_num(x)/y_den(x))`.
pub(crate) struct Iso<C> {
    pub(crate) x_num: Vec<C>,
    pub(crate) x_den: Vec<C>,
    pub(crate) y_num: Vec<C>,
    pub(crate) y_den: Vec<C>,
}

impl<C: Coord> Iso<C> {
    /// The image of `(xₙ/x_d, y)` in Jacobian coordinates, the point at infinity where a
    /// denominator vanishes: with `X/D` and `y·Y/W` the two coordinates, `Z = D·W`,
    /// `X' = X·D·W²` and `Y' = y·Y·D³·W²`.
    pub(crate) fn map(&self, xn: &C, xd: &C, y: &C) -> Jac<C> {
        let d = self
            .y_den
            .len()
            .max(self.y_num.len())
            .max(self.x_num.len())
            .max(self.x_den.len())
            - 1;
        let mut pows = Vec::with_capacity(d + 1);
        pows.push(C::one());
        for i in 1..=d {
            let next = pows[i - 1].mul(xd);
            pows.push(next);
        }
        // Each of the four as a polynomial of degree d in (xₙ, x_d): the ratios are unchanged.
        let xnum = horner(&self.x_num, xn, &pows, d);
        let xden = horner(&self.x_den, xn, &pows, d);
        let ynum = horner(&self.y_num, xn, &pows, d);
        let yden = horner(&self.y_den, xn, &pows, d);
        let z = xden.mul(&yden);
        if z.is_zero() {
            return Jac::identity();
        }
        let w2 = yden.square();
        let x = xnum.mul(&xden).mul(&w2);
        let d3 = xden.square().mul(&xden);
        let yy = y.mul(&ynum).mul(&d3).mul(&w2);
        Jac { x, y: yy, z }
    }

    /// The image of a Jacobian point `(X : Y : Z)` of the source curve: `x = X/Z²`, `y = Y/Z³`.
    pub(crate) fn map_jac(&self, p: &Jac<C>) -> Jac<C> {
        if p.is_identity() {
            return Jac::identity();
        }
        // x = X / Z², and y = Y/Z³: the ratio y_num/y_den is in x alone, so take y = Y·(1/Z³)
        // as Y with an extra Z³ in the denominator of the y-coordinate.
        let z2 = p.z.square();
        let z3 = z2.mul(&p.z);
        let d = self
            .y_den
            .len()
            .max(self.y_num.len())
            .max(self.x_num.len())
            .max(self.x_den.len())
            - 1;
        let mut pows = Vec::with_capacity(d + 1);
        pows.push(C::one());
        for i in 1..=d {
            let next = pows[i - 1].mul(&z2);
            pows.push(next);
        }
        let xnum = horner(&self.x_num, &p.x, &pows, d);
        let xden = horner(&self.x_den, &p.x, &pows, d);
        let ynum = horner(&self.y_num, &p.x, &pows, d);
        let yden = horner(&self.y_den, &p.x, &pows, d).mul(&z3);
        let z = xden.mul(&yden);
        if z.is_zero() {
            return Jac::identity();
        }
        let w2 = yden.square();
        let x = xnum.mul(&xden).mul(&w2);
        let d3 = xden.square().mul(&xden);
        let yy = p.y.mul(&ynum).mul(&d3).mul(&w2);
        Jac { x, y: yy, z }
    }
}

/// `P + Q` on `y² = x³ + A·x + B` with `A ≠ 0`: "add-2007-bl", which does not use `A`, and its
/// doubling "dbl-2007-bl" with `A` for equal points.
pub(crate) fn add_a<C: Coord>(p: &Jac<C>, q: &Jac<C>, a: &C) -> Jac<C> {
    if p.is_identity() {
        return *q;
    }
    if q.is_identity() {
        return *p;
    }
    let z1z1 = p.z.square();
    let z2z2 = q.z.square();
    let u1 = p.x.mul(&z2z2);
    let u2 = q.x.mul(&z1z1);
    let s1 = p.y.mul(&q.z).mul(&z2z2);
    let s2 = q.y.mul(&p.z).mul(&z1z1);
    let h = u2.sub(&u1);
    let r = s2.sub(&s1).double();
    if h.is_zero() {
        if !r.is_zero() {
            return Jac::identity();
        }
        // dbl-2007-bl.
        let xx = p.x.square();
        let yy = p.y.square();
        let yyyy = yy.square();
        let zz = z1z1;
        let s = p.x.add(&yy).square().sub(&xx).sub(&yyyy).double();
        let m = xx.double().add(&xx).add(&a.mul(&zz.square()));
        let t = m.square().sub(&s.double());
        let y3 = m.mul(&s.sub(&t)).sub(&yyyy.double().double().double());
        let z3 = p.y.add(&p.z).square().sub(&yy).sub(&zz);
        return Jac { x: t, y: y3, z: z3 };
    }
    let i = h.double().square();
    let j = h.mul(&i);
    let v = u1.mul(&i);
    let x3 = r.square().sub(&j).sub(&v.double());
    let y3 = r.mul(&v.sub(&x3)).sub(&s1.mul(&j).double());
    let z3 = p.z.add(&q.z).square().sub(&z1z1).sub(&z2z2).mul(&h);
    Jac {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// `inv0`: the inverse, or 0 for 0.
fn inv0<C: Coord>(x: &C) -> C {
    x.inverse().unwrap_or_else(C::zero)
}

/// The simplified SWU map to `E′` over `F_p²`, the plain form of section 6.6.2.
pub(crate) struct SswuFp2<E, const N: usize> {
    pub(crate) a: Fp2<E, N>,
    pub(crate) b: Fp2<E, N>,
    pub(crate) z: Fp2<E, N>,
    minus_b_over_a: Fp2<E, N>,
    exceptional: Fp2<E, N>,
}

impl<E: Engine<N>, const N: usize> SswuFp2<E, N> {
    pub(crate) fn new(a: Fp2<E, N>, b: Fp2<E, N>, z: Fp2<E, N>) -> Self {
        let inv_a = a.inverse().expect("A ≠ 0");
        let ba = b.mul(&inv_a);
        SswuFp2 {
            a,
            b,
            z,
            minus_b_over_a: ba.neg(),
            exceptional: ba.mul(&z.inverse().expect("Z ≠ 0")),
        }
    }

    fn g(&self, x: &Fp2<E, N>) -> Fp2<E, N> {
        x.square().add(&self.a).mul(x).add(&self.b)
    }

    /// The point of `E′` that `u` maps to, affine.
    pub(crate) fn map(&self, u: &Fp2<E, N>) -> (Fp2<E, N>, Fp2<E, N>) {
        let zu2 = self.z.mul(&u.square());
        let tv1 = inv0(&zu2.square().add(&zu2));
        let x1 = if tv1.is_zero() {
            self.exceptional
        } else {
            self.minus_b_over_a.mul(&tv1.add(&Fp2::one()))
        };
        let (x, y) = match self.g(&x1).sqrt() {
            Some(y) => (x1, y),
            None => {
                let x2 = zu2.mul(&x1);
                let y = self
                    .g(&x2)
                    .sqrt()
                    .expect("g(x2) is a square when g(x1) is not");
                (x2, y)
            }
        };
        let y = if y.sgn0() == u.sgn0() { y } else { y.neg() };
        (x, y)
    }
}

/// The Shallue–van de Woestijne map to `y² = x³ + A·x + B` over `F_p`, section 6.6.1.
pub(crate) struct Svdw<E, const N: usize> {
    a: Fp<E, N>,
    b: Fp<E, N>,
    z: Fp<E, N>,
    gz: Fp<E, N>,
    c2: Fp<E, N>,
    c3: Fp<E, N>,
    c4: Fp<E, N>,
}

impl<E: Engine<N>, const N: usize> Svdw<E, N> {
    pub(crate) fn new(a: Fp<E, N>, b: Fp<E, N>, z: Fp<E, N>) -> Self {
        let small = |v: u64| Fp::<E, N>::from_u64(v);
        let gz = z.square().add(&a).mul(&z).add(&b);
        let k = small(3).mul(&z.square()).add(&small(4).mul(&a));
        let mut c3 = gz
            .mul(&k)
            .neg()
            .sqrt()
            .expect("−g(Z)(3Z² + 4A) is a square");
        if c3.is_odd() {
            c3 = c3.neg();
        }
        let c4 = small(4)
            .neg()
            .mul(&gz)
            .mul(&k.inverse().expect("3Z² + 4A ≠ 0"));
        let c2 = z.mul(&small(2).inverse().expect("p is odd")).neg();
        Svdw {
            a,
            b,
            z,
            gz,
            c2,
            c3,
            c4,
        }
    }

    fn g(&self, x: &Fp<E, N>) -> Fp<E, N> {
        x.square().add(&self.a).mul(x).add(&self.b)
    }

    /// The point that `u` maps to, affine.
    pub(crate) fn map(&self, u: &Fp<E, N>) -> (Fp<E, N>, Fp<E, N>) {
        let one = Fp::one();
        let tv1 = u.square().mul(&self.gz);
        let tv2 = one.add(&tv1);
        let tv1 = one.sub(&tv1);
        let tv3 = inv0(&tv1.mul(&tv2));
        let tv5 = u.mul(&tv1).mul(&tv3).mul(&self.c3);
        let x1 = self.c2.sub(&tv5);
        let x2 = self.c2.add(&tv5);
        let x3 = self.z.add(&self.c4.mul(&tv2.square().mul(&tv3).square()));
        let (x, y) = [x1, x2]
            .into_iter()
            .find_map(|x| self.g(&x).sqrt().map(|y| (x, y)))
            .unwrap_or_else(|| {
                let y = self
                    .g(&x3)
                    .sqrt()
                    .expect("g(x3) is a square when g(x1), g(x2) are not");
                (x3, y)
            });
        let y = if y.is_odd() == u.is_odd() { y } else { y.neg() };
        (x, y)
    }
}
