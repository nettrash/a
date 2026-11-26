//! The optimal ate pairing on the engine's arithmetic: the same Miller loop and final
//! exponentiation as [`miller`](super::super::miller) describes, formula for formula, so that the
//! values are the public ones.

use super::super::curve::{Family, Twist};
use super::{Engine, Fp, Fp2, Fp6, Fp12};

/// An affine point of the curve and one of the twist.
pub(crate) type Pair<E, const N: usize> = ((Fp<E, N>, Fp<E, N>), (Fp2<E, N>, Fp2<E, N>));

/// A point `(X : Y : Z)` of the twist, `x = X/Z`, `y = Y/Z`.
struct Proj<E, const N: usize> {
    x: Fp2<E, N>,
    y: Fp2<E, N>,
    z: Fp2<E, N>,
}

/// A line `a + b x_P + c y_P`, scaled by a factor of `F_p²`.
struct Line<E, const N: usize> {
    a: Fp2<E, N>,
    b: Fp2<E, N>,
    c: Fp2<E, N>,
}

impl<E: Engine<N>, const N: usize> Proj<E, N> {
    /// `2T` and the tangent at `T` (Costello, Lange and Naehrig in the form of Aranha et al.).
    #[inline]
    fn double(&mut self, twelve_xi: bool, b3: &Fp2<E, N>) -> Option<Line<E, N>> {
        if self.z.is_zero() {
            return None;
        }
        if self.y.is_zero() {
            self.set_infinity();
            return None;
        }
        let (x, y, z) = (&self.x, &self.y, &self.z);
        let a = x.mul(y).half();
        let b = y.square();
        let c = z.square();
        // BLS12's twists have b' = 4ξ, 3b' = 12ξ: the product by additions.
        let e = if twelve_xi {
            let t = c.mul_by_xi();
            let t = t.double().add(&t);
            t.double().double()
        } else {
            c.mul(b3)
        };
        let f = e.double().add(&e);
        let g = b.add(&f).half();
        let h = y.add(z).square().sub(&b).sub(&c);
        let i = e.sub(&b);
        let j = x.square();
        let e2 = e.square();
        let x3 = a.mul(&b.sub(&f));
        let y3 = g.square().sub(&e2.double().add(&e2));
        let z3 = b.mul(&h);
        self.x = x3;
        self.y = y3;
        self.z = z3;
        Some(Line {
            a: i,
            b: j.double().add(&j),
            c: h.neg(),
        })
    }

    /// `T + Q` for an affine `Q` and the line through them ("madd-1998-cmo").
    #[inline]
    fn add(
        &mut self,
        q: &(Fp2<E, N>, Fp2<E, N>),
        twelve_xi: bool,
        b3: &Fp2<E, N>,
    ) -> Option<Line<E, N>> {
        let (xq, yq) = q;
        if self.z.is_zero() {
            self.x = *xq;
            self.y = *yq;
            self.z = Fp2::one();
            return None;
        }
        let u = yq.mul(&self.z).sub(&self.y);
        let v = xq.mul(&self.z).sub(&self.x);
        if v.is_zero() {
            if u.is_zero() {
                return self.double(twelve_xi, b3);
            }
            self.set_infinity();
            return None;
        }
        let line = Line {
            a: v.mul(yq).sub(&u.mul(xq)),
            b: u,
            c: v.neg(),
        };
        let uu = u.square();
        let vv = v.square();
        let vvv = v.mul(&vv);
        let r = vv.mul(&self.x);
        let a = uu.mul(&self.z).sub(&vvv).sub(&r.double());
        let x3 = v.mul(&a);
        let y3 = u.mul(&r.sub(&a)).sub(&vvv.mul(&self.y));
        let z3 = vvv.mul(&self.z);
        self.x = x3;
        self.y = y3;
        self.z = z3;
        Some(line)
    }

    fn set_infinity(&mut self) {
        self.x = Fp2::zero();
        self.y = Fp2::one();
        self.z = Fp2::zero();
    }
}

/// `f` times the line at `p`, at the positions of the twist.
#[inline]
fn evaluate<E: Engine<N>, const N: usize>(
    f: &Fp12<E, N>,
    l: &Line<E, N>,
    p: &(Fp<E, N>, Fp<E, N>),
    twist: Twist,
) -> Fp12<E, N> {
    let bx = l.b.mul_by_fp(&p.0);
    let cy = l.c.mul_by_fp(&p.1);
    match twist {
        Twist::D => f.mul_by_034(&cy, &bx, &l.a),
        Twist::M => f.mul_by_014(&l.a, &bx, &cy),
    }
}

/// The line at `P` as an element of `F_p¹²`: the product of 1 by it.
fn as_fp12<E: Engine<N>, const N: usize>(
    l: &Line<E, N>,
    p: &(Fp<E, N>, Fp<E, N>),
    twist: Twist,
) -> Fp12<E, N> {
    let bx = l.b.mul_by_fp(&p.0);
    let cy = l.c.mul_by_fp(&p.1);
    let z = Fp2::zero();
    match twist {
        // 1·(c₀ + c₃w + c₄vw), as `mul_by_034` gives it.
        Twist::D => Fp12::new(Fp6::new(cy, z, z), Fp6::new(bx, l.a, z)),
        // 1·(c₀ + c₁v + c₄vw), as `mul_by_014` gives it.
        Twist::M => Fp12::new(Fp6::new(l.a, bx, z), Fp6::new(z, cy, z)),
    }
}

/// The product of the Miller functions of the pairs of affine points (`P` of the curve, `Q` of the
/// twist), points at infinity left out by the caller.
pub(crate) fn miller_loop<E: Engine<N>, const N: usize>(pairs: &[Pair<E, N>]) -> Fp12<E, N> {
    let mut f = Fp12::one();
    if pairs.is_empty() {
        return f;
    }
    let curve = E::curve();
    let t = E::tables();
    let (b3, twist) = (&t.b3_twist, curve.twist);
    let xi = Fp2::new(Fp::from_u64(E::XI0), Fp::one());
    let twelve_xi = *b3 == xi.double().add(&xi).double().double();
    let mut ts: Vec<Proj<E, N>> = pairs
        .iter()
        .map(|(_, q)| Proj {
            x: q.0,
            y: q.1,
            z: Fp2::one(),
        })
        .collect();
    let minus: Vec<(Fp2<E, N>, Fp2<E, N>)> = pairs.iter().map(|(_, q)| (q.0, q.1.neg())).collect();
    let digits = &t.loop_digits;
    // While f is still 1, its square is 1 and its product by a line the line itself.
    let mut one = true;
    for i in (0..digits.len() - 1).rev() {
        if !one {
            f = f.square();
        }
        for (tp, (p, _)) in ts.iter_mut().zip(pairs) {
            if let Some(l) = tp.double(twelve_xi, b3) {
                f = if one {
                    one = false;
                    as_fp12(&l, p, twist)
                } else {
                    evaluate(&f, &l, p, twist)
                };
            }
        }
        if digits[i] != 0 {
            for (k, (tp, (p, q))) in ts.iter_mut().zip(pairs).enumerate() {
                let add = if digits[i] > 0 { q } else { &minus[k] };
                if let Some(l) = tp.add(add, twelve_xi, b3) {
                    f = evaluate(&f, &l, p, twist);
                }
            }
        }
    }
    if t.loop_negative {
        f = f.conjugate();
        for tp in &mut ts {
            tp.y = tp.y.neg();
        }
    }
    if curve.family == Family::Bn {
        let [gx, gy] = &t.psi;
        let [nx, ny] = &t.psi2;
        for (tp, (p, (x, y))) in ts.iter_mut().zip(pairs) {
            let q1 = (x.conjugate().mul(gx), y.conjugate().mul(gy));
            let q2 = (x.mul_by_fp(nx), y.mul_by_fp(ny).neg());
            if let Some(l) = tp.add(&q1, twelve_xi, b3) {
                f = evaluate(&f, &l, p, twist);
            }
            if let Some(l) = tp.add(&q2, twelve_xi, b3) {
                f = evaluate(&f, &l, p, twist);
            }
        }
    }
    f
}

/// `g^x` in the cyclotomic subgroup.
#[inline]
pub(crate) fn pow_x<E: Engine<N>, const N: usize>(g: &Fp12<E, N>) -> Fp12<E, N> {
    let t = E::tables();
    let r = g.cyclotomic_pow_digits(&t.x_digits);
    if t.x_negative { r.conjugate() } else { r }
}

/// The final exponentiation of a nonzero `f`: the easy part, then the hard part's chain of the
/// curve's family.
///
/// # Panics
///
/// Panics if `f` is 0.
#[track_caller]
pub(crate) fn final_exponentiation<E: Engine<N>, const N: usize>(f: &Fp12<E, N>) -> Fp12<E, N> {
    let inv = f.inverse().expect("the value of a Miller loop is not 0");
    let f1 = f.conjugate().mul(&inv);
    let f = f1.frobenius(2).mul(&f1);
    let ex = |g: &Fp12<E, N>| pow_x::<E, N>(g);
    match E::curve().family {
        Family::Bls12 => {
            let a = ex(&f).mul(&f.conjugate());
            let b = ex(&a).mul(&a.conjugate());
            let c = ex(&b).mul(&b.frobenius(1));
            let d = ex(&ex(&c)).mul(&c.frobenius(2)).mul(&c.conjugate());
            d.mul(&f.cyclotomic_square()).mul(&f)
        }
        Family::Bn => {
            let fx = ex(&f);
            let f2x = fx.cyclotomic_square();
            let f4x = f2x.cyclotomic_square();
            let f6x = f4x.mul(&f2x);
            let f6x2 = ex(&f6x);
            let f12x2 = f6x2.cyclotomic_square();
            let f12x3 = ex(&f12x2);
            let a = f12x3.mul(&f6x2).mul(&f6x);
            let b = a.mul(&f2x.conjugate());
            let l0 = a.mul(&f6x2).mul(&f);
            let l3 = b.mul(&f.conjugate()).frobenius(3);
            l0.mul(&b.frobenius(1)).mul(&a.frobenius(2)).mul(&l3)
        }
    }
}
