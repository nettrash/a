//! The optimal ate pairing: Miller's loop on the twist in homogeneous projective coordinates, and
//! the final exponentiation.

use super::curve::{Family, PairingCurve, Twist};
use super::g1::G1;
use super::g2::G2;
use super::gt::Gt;
use crate::math::field::{Fp, Fp2, Fp12};

/// A point `(X : Y : Z)` of the twist, `x = X/Z`, `y = Y/Z`, the point at infinity for `Z = 0`.
struct Proj<'c> {
    x: Fp2<'c>,
    y: Fp2<'c>,
    z: Fp2<'c>,
}

/// A line of the Miller loop, scaled by a factor in `F_p²` (which the final exponentiation
/// removes): its value at `P = (x_P, y_P)` is `a + b x_P + c y_P` at the positions of the twist.
struct Line<'c> {
    a: Fp2<'c>,
    b: Fp2<'c>,
    c: Fp2<'c>,
}

impl<'c> Proj<'c> {
    /// `2T` and the tangent at `T`, by the formulas of Costello, Lange and Naehrig (*Faster pairing
    /// computations on curves with high-degree twists*, PKC 2010, §5) for `y² = x³ + b'`, in the
    /// form of Aranha, Barreto, Longa and Ricardini (*The realm of the pairings*, SAC 2013, §4):
    /// `A = XY/2, B = Y², C = Z², E = 3b'C, F = 3E, G = (B + F)/2, H = (Y + Z)² − B − C,
    /// I = E − B, J = X²`, then `2T = (A(B − F) : G² − 3E² : BH)` and the line
    /// `I + 3J x_P − H y_P`, which is `−2YZ` times the tangent. `None` when the tangent is vertical
    /// or `T` is at infinity: its value is then in `F_p⁶` and the final exponentiation kills it.
    fn double(&mut self, curve: &PairingCurve) -> Option<Line<'c>> {
        if self.z.is_zero() {
            return None;
        }
        if self.y.is_zero() {
            self.set_infinity();
            return None;
        }
        let (x, y, z) = (&self.x, &self.y, &self.z);
        let a = (x * y).mul_by_base(&curve.half);
        let b = y.square();
        let c = z.square();
        let e = &c * &curve.b3_twist;
        let f = &e.double() + &e;
        let g = (&b + &f).mul_by_base(&curve.half);
        let h = (y + z).square() - &b - &c;
        let i = &e - &b;
        let j = x.square();
        let e2 = e.square();
        let x3 = &a * &(&b - &f);
        let y3 = g.square() - (&e2.double() + &e2);
        let z3 = &b * &h;
        self.x = x3;
        self.y = y3;
        self.z = z3;
        Some(Line {
            a: i,
            b: &j.double() + &j,
            c: -h,
        })
    }

    /// `T + Q` for an affine `Q` and the line through them, by the mixed addition "madd-1998-cmo"
    /// of the Explicit-Formulas Database: `u = y_Q Z − Y`, `v = x_Q Z − X`, the line is
    /// `(v y_Q − u x_Q) + u x_P − v y_P`, which is `−v` times the line through `T` and `Q`.
    /// `None` when it is vertical (`T = −Q`) or when `T` was at infinity.
    fn add(&mut self, q: &(Fp2<'c>, Fp2<'c>), curve: &PairingCurve) -> Option<Line<'c>> {
        let (xq, yq) = q;
        if self.z.is_zero() {
            self.x = xq.clone();
            self.y = yq.clone();
            self.z = Fp2::one(xq.tower());
            return None;
        }
        let u = &(yq * &self.z) - &self.y;
        let v = &(xq * &self.z) - &self.x;
        if v.is_zero() {
            if u.is_zero() {
                return self.double(curve);
            }
            self.set_infinity();
            return None;
        }
        let line = Line {
            a: &(&v * yq) - &(&u * xq),
            b: u.clone(),
            c: -&v,
        };
        let uu = u.square();
        let vv = v.square();
        let vvv = &v * &vv;
        let r = &vv * &self.x;
        let a = &(&(&uu * &self.z) - &vvv) - &r.double();
        let x3 = &v * &a;
        let y3 = &(&u * &(&r - &a)) - &(&vvv * &self.y);
        let z3 = &vvv * &self.z;
        self.x = x3;
        self.y = y3;
        self.z = z3;
        Some(line)
    }

    /// Sets the point at infinity, `(0 : 1 : 0)`.
    fn set_infinity(&mut self) {
        let t = self.x.tower();
        self.x = Fp2::zero(t);
        self.y = Fp2::one(t);
        self.z = Fp2::zero(t);
    }
}

/// `f` times the value of `line` at `p`: a sparse product, at the positions `0, 3, 4` of a D-type
/// twist (`(x w², y w³)` makes the line `c y_P + b x_P w + a w³`) or `0, 1, 4` of an M-type twist
/// (times `w³`, the line through `(x/w², y/w³)` is `a + b x_P v + c y_P v w`).
fn evaluate<'c>(f: &Fp12<'c>, line: &Line<'c>, p: &(Fp<'c>, Fp<'c>), twist: Twist) -> Fp12<'c> {
    let bx = line.b.mul_by_base(&p.0);
    let cy = line.c.mul_by_base(&p.1);
    match twist {
        Twist::D => f.mul_by_034(&cy, &bx, &line.a),
        Twist::M => f.mul_by_014(&line.a, &bx, &cy),
    }
}

impl PairingCurve {
    /// The product of the Miller functions of the optimal ate pairing for the pairs, before the
    /// final exponentiation: one loop for all of them, with one square of `F_p¹²` per bit.
    ///
    /// The loop runs over the non-adjacent form of `6x + 2` for BN (65 doublings and 21
    /// additions on BN254), with the two lines through `ψ(Q)` and `−ψ²(Q)` after it, and over the
    /// binary form of `|x|` for BLS12 (63 doublings and 5 additions on BLS12-381), conjugated when
    /// `x < 0`
    /// (Vercauteren, *Optimal pairings*, IEEE Trans. Inf. Theory 56, 2010). `T` is held in
    /// homogeneous projective coordinates on the twist, and each line is evaluated at `P` as a
    /// sparse element of `F_p¹²` (see [`Fp12::mul_by_034`] and [`Fp12::mul_by_014`]). Pairs with a
    /// point at infinity contribute 1.
    ///
    /// The value is a representative: lines are scaled by factors of `F_p²`, which the final
    /// exponentiation removes, so it equals other implementations' only after that.
    ///
    /// # Panics
    ///
    /// Panics if a point is of another curve.
    #[track_caller]
    #[must_use]
    pub fn miller_loop<'c>(&'c self, pairs: &[(&G1<'c>, &G2<'c>)]) -> Fp12<'c> {
        let mut ps = Vec::with_capacity(pairs.len());
        let mut qs = Vec::with_capacity(pairs.len());
        for (p, q) in pairs {
            self.check(p.curve);
            self.check(q.curve);
            if let (Some(pa), Some(qa)) = (p.coordinates(), q.coordinates()) {
                ps.push(pa);
                qs.push(qa);
            }
        }
        let mut f = Fp12::one(self.tower);
        if ps.is_empty() {
            return f;
        }
        let mut ts: Vec<Proj<'c>> = qs
            .iter()
            .map(|(x, y)| Proj {
                x: x.clone(),
                y: y.clone(),
                z: Fp2::one(self.tower),
            })
            .collect();
        let minus: Vec<(Fp2<'c>, Fp2<'c>)> = qs.iter().map(|(x, y)| (x.clone(), -y)).collect();
        let digits = &self.loop_digits;
        for i in (0..digits.len() - 1).rev() {
            f = f.square();
            for (t, p) in ts.iter_mut().zip(&ps) {
                if let Some(l) = t.double(self) {
                    f = evaluate(&f, &l, p, self.twist);
                }
            }
            if digits[i] != 0 {
                let add = if digits[i] > 0 { &qs } else { &minus };
                for ((t, p), q) in ts.iter_mut().zip(&ps).zip(add) {
                    if let Some(l) = t.add(q, self) {
                        f = evaluate(&f, &l, p, self.twist);
                    }
                }
            }
        }
        if self.loop_scalar.is_negative() {
            f = f.conjugate();
            for t in &mut ts {
                t.y = -&t.y;
            }
        }
        if self.family == Family::Bn {
            let [gx, gy] = &self.psi;
            let [nx, ny] = &self.psi2;
            for ((t, p), (x, y)) in ts.iter_mut().zip(&ps).zip(&qs) {
                let q1 = (&x.conjugate() * gx, &y.conjugate() * gy);
                let q2 = (x.mul_by_base(nx), -y.mul_by_base(ny));
                if let Some(l) = t.add(&q1, self) {
                    f = evaluate(&f, &l, p, self.twist);
                }
                if let Some(l) = t.add(&q2, self) {
                    f = evaluate(&f, &l, p, self.twist);
                }
            }
        }
        f
    }

    /// `f^(c (p¹² − 1)/r)`, the final exponentiation, as an element of `GT`.
    ///
    /// The easy part `f^((p⁶ − 1)(p² + 1))` takes a conjugation, an inversion and a Frobenius
    /// map, and leaves an element of the cyclotomic subgroup, where squares are Granger and Scott's
    /// and inverses are conjugates. The hard part raises to `c (p⁴ − p² + 1)/r` by products by
    /// `x` and Frobenius maps:
    ///
    /// * BLS12: `c = 3`, as `3 (p⁴ − p² + 1)/r = (x − 1)² (x + p) (x² + p² − 1) + 3` (Hayashida,
    ///   Hayasaka and Teruya, *Efficient final exponentiation via cyclotomic structure for pairings
    ///   over families of elliptic curves*, ePrint 2020/875): five powers by `x`.
    /// * BN: `c = 2x(6x² + 3x + 1)`, by the chain of Fuentes-Castañeda, Knapp and
    ///   Rodríguez-Henríquez (*Faster hashing to G2*, SAC 2011, §5) for
    ///   `λ₀ + λ₁p + λ₂p² + λ₃p³` with `λ₀ = 12x³ + 12x² + 6x + 1`, `λ₁ = 12x³ + 6x² + 4x`,
    ///   `λ₂ = 12x³ + 6x² + 6x`, `λ₃ = 12x³ + 6x² + 4x − 1`: three powers by `x`.
    ///
    /// The factor `c`, prime to `r`, is the one of arkworks, of the `bls12_381` crate and of blst,
    /// whose pairings this one equals. Others differ by a fixed power. The IETF draft on
    /// pairing-friendly curves raises `f_{x,Q}(P)` to `(p¹² − 1)/r` alone: on BLS12-381 its value
    /// is `e^(1/3)`. py_ecc raises to `(p¹² − 1)/r` alone too, and on BLS12-381 loops over `|x|`
    /// without conjugating: its values are `e^(1/c)` on BN254 and `e^(−1/3)` on BLS12-381. To
    /// compare a value of `GT` with theirs, raise it to `c⁻¹ mod r` (`e.pow(&c.mod_inverse(r).unwrap())`),
    /// and for py_ecc on BLS12-381 take the inverse of that. A check that a product of pairings
    /// is 1, as signatures and proofs are verified, gives the same answer as all of them, since
    /// `c` is prime to `r`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is 0 (no Miller loop gives 0) or not of the curve's tower.
    #[track_caller]
    #[must_use]
    pub fn final_exponentiation<'c>(&'c self, f: &Fp12<'c>) -> Gt<'c> {
        assert!(
            f.tower().xi() == self.tower.xi(),
            "elements of different towers"
        );
        let inv = f.inverse().expect("the value of a Miller loop is not 0");
        let f1 = &f.conjugate() * &inv;
        let f = &f1.frobenius(2) * &f1;
        let x = &self.x;
        let ex = |g: &Fp12<'c>| g.cyclotomic_pow(x);
        let r = match self.family {
            Family::Bls12 => {
                let a = &ex(&f) * &f.conjugate();
                let b = &ex(&a) * &a.conjugate();
                let c = &ex(&b) * &b.frobenius(1);
                let d = &(&ex(&ex(&c)) * &c.frobenius(2)) * &c.conjugate();
                &(&d * &f.cyclotomic_square()) * &f
            }
            Family::Bn => {
                let fx = ex(&f);
                let f2x = fx.cyclotomic_square();
                let f4x = f2x.cyclotomic_square();
                let f6x = &f4x * &f2x;
                let f6x2 = ex(&f6x);
                let f12x2 = f6x2.cyclotomic_square();
                let f12x3 = ex(&f12x2);
                let a = &(&f12x3 * &f6x2) * &f6x;
                let b = &a * &f2x.conjugate();
                let l0 = &(&a * &f6x2) * &f;
                let l3 = (&b * &f.conjugate()).frobenius(3);
                &(&(&l0 * &b.frobenius(1)) * &a.frobenius(2)) * &l3
            }
        };
        Gt::from_fp12_unchecked(self, r)
    }

    /// The optimal ate pairing `e(P, Q)`: [`miller_loop`](PairingCurve::miller_loop) then
    /// [`final_exponentiation`](PairingCurve::final_exponentiation). Bilinear and non-degenerate on
    /// `G1 × G2`; 1 when either point is at infinity.
    ///
    /// For points of the curves outside `G1` and `G2` (from `new_on_curve`) it returns an element
    /// of `GT` without panicking, but not a pairing.
    ///
    /// # Panics
    ///
    /// Panics if a point is of another curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::PairingCurve;
    /// use a::math::numbers::Number;
    ///
    /// let c = PairingCurve::bn254();
    /// let (p, q) = (c.g1_generator(), c.g2_generator());
    /// let (a, b) = (Number::from(6u8), Number::from(7u8));
    /// let e = c.pairing(&(&p * &a), &(&q * &b));
    /// assert_eq!(e, c.pairing(&p, &q).pow(&Number::from(42u8)));
    /// assert_eq!(e, c.pairing(&(&p * &b), &(&q * &a)));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn pairing<'c>(&'c self, p: &G1<'c>, q: &G2<'c>) -> Gt<'c> {
        self.final_exponentiation(&self.miller_loop(&[(p, q)]))
    }

    /// The product `∏ e(Pᵢ, Qᵢ)` with one Miller loop and one final exponentiation, as the checks of
    /// signatures and proofs use it; 1 for no pairs.
    ///
    /// # Panics
    ///
    /// Panics if a point is of another curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::PairingCurve;
    /// use a::math::numbers::Number;
    ///
    /// let c = PairingCurve::bls12_381();
    /// let (p, q) = (c.g1_generator(), c.g2_generator());
    /// let five = Number::from(5u8);
    /// // e(5P, Q) · e(−P, 5Q) = 1.
    /// let product = c.multi_pairing(&[(&(&p * &five), &q), (&-&p, &(&q * &five))]);
    /// assert!(product.is_identity());
    /// assert!(c.multi_pairing(&[]).is_identity());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn multi_pairing<'c>(&'c self, pairs: &[(&G1<'c>, &G2<'c>)]) -> Gt<'c> {
        self.final_exponentiation(&self.miller_loop(pairs))
    }
}
