//! `F_p⁶ = F_p²[v] / (v³ − ξ)` and `F_p¹² = F_p⁶[w] / (w² − v)`, the floors of the public
//! [`Tower`](crate::math::field::Tower) with the same bases and formulas, so that an element here
//! and its image there are the same element.

use super::Engine;
use super::fp::Fp;
use super::fp2::Fp2;

/// `c0 + c1 v + c2 v²`.
pub(crate) struct Fp6<E, const N: usize> {
    pub(crate) c0: Fp2<E, N>,
    pub(crate) c1: Fp2<E, N>,
    pub(crate) c2: Fp2<E, N>,
}

impl<E, const N: usize> Clone for Fp6<E, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, const N: usize> Copy for Fp6<E, N> {}

impl<E, const N: usize> PartialEq for Fp6<E, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1 && self.c2 == other.c2
    }
}

impl<E, const N: usize> Eq for Fp6<E, N> {}

impl<E: Engine<N>, const N: usize> Fp6<E, N> {
    #[inline(always)]
    pub(crate) const fn new(c0: Fp2<E, N>, c1: Fp2<E, N>, c2: Fp2<E, N>) -> Self {
        Fp6 { c0, c1, c2 }
    }

    #[inline(always)]
    pub(crate) const fn zero() -> Self {
        Fp6::new(Fp2::zero(), Fp2::zero(), Fp2::zero())
    }

    #[inline(always)]
    pub(crate) const fn one() -> Self {
        Fp6::new(Fp2::one(), Fp2::zero(), Fp2::zero())
    }

    #[inline(always)]
    pub(crate) fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }

    #[inline(always)]
    pub(crate) fn add(&self, o: &Self) -> Self {
        Fp6::new(self.c0.add(&o.c0), self.c1.add(&o.c1), self.c2.add(&o.c2))
    }

    #[inline(always)]
    pub(crate) fn sub(&self, o: &Self) -> Self {
        Fp6::new(self.c0.sub(&o.c0), self.c1.sub(&o.c1), self.c2.sub(&o.c2))
    }

    #[inline(always)]
    pub(crate) fn double(&self) -> Self {
        Fp6::new(self.c0.double(), self.c1.double(), self.c2.double())
    }

    #[inline(always)]
    pub(crate) fn neg(&self) -> Self {
        Fp6::new(self.c0.neg(), self.c1.neg(), self.c2.neg())
    }

    /// The product by `v`: `ξ c₂ + c₀ v + c₁ v²`.
    #[inline(always)]
    pub(crate) fn mul_by_v(&self) -> Self {
        Fp6::new(self.c2.mul_by_xi(), self.c0, self.c1)
    }

    /// Karatsuba's six products of `F_p²`.
    #[inline]
    pub(crate) fn mul(&self, b: &Self) -> Self {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        let (b0, b1, b2) = (&b.c0, &b.c1, &b.c2);
        let v0 = a0.mul(b0);
        let v1 = a1.mul(b1);
        let v2 = a2.mul(b2);
        let c0 = a1
            .add(a2)
            .mul(&b1.add(b2))
            .sub(&v1)
            .sub(&v2)
            .mul_by_xi()
            .add(&v0);
        let c1 = a0
            .add(a1)
            .mul(&b0.add(b1))
            .sub(&v0)
            .sub(&v1)
            .add(&v2.mul_by_xi());
        let c2 = a0.add(a2).mul(&b0.add(b2)).sub(&v0).sub(&v2).add(&v1);
        Fp6::new(c0, c1, c2)
    }

    /// Chung and Hasan's squaring (SQR2).
    #[inline]
    pub(crate) fn square(&self) -> Self {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        let s0 = a0.square();
        let s1 = a0.mul(a1).double();
        let s2 = a0.sub(a1).add(a2).square();
        let s3 = a1.mul(a2).double();
        let s4 = a2.square();
        Fp6::new(
            s0.add(&s3.mul_by_xi()),
            s1.add(&s4.mul_by_xi()),
            s1.add(&s2).add(&s3).sub(&s0).sub(&s4),
        )
    }

    /// The product by an element of `F_p²`.
    #[inline(always)]
    pub(crate) fn mul_by_fp2(&self, k: &Fp2<E, N>) -> Self {
        Fp6::new(self.c0.mul(k), self.c1.mul(k), self.c2.mul(k))
    }

    /// The product by `b₀ + b₁ v`.
    #[inline]
    pub(crate) fn mul_by_01(&self, b0: &Fp2<E, N>, b1: &Fp2<E, N>) -> Self {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        let v0 = a0.mul(b0);
        let v1 = a1.mul(b1);
        Fp6::new(
            a1.add(a2).mul(b1).sub(&v1).mul_by_xi().add(&v0),
            a0.add(a1).mul(&b0.add(b1)).sub(&v0).sub(&v1),
            a0.add(a2).mul(b0).sub(&v0).add(&v1),
        )
    }

    /// The product by `b₁ v`.
    #[inline]
    pub(crate) fn mul_by_1(&self, b1: &Fp2<E, N>) -> Self {
        Fp6::new(
            self.c2.mul(b1).mul_by_xi(),
            self.c0.mul(b1),
            self.c1.mul(b1),
        )
    }

    /// `1 / self`, `None` for 0.
    pub(crate) fn inverse(&self) -> Option<Self> {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        let t0 = a0.square().sub(&a1.mul(a2).mul_by_xi());
        let t1 = a2.square().mul_by_xi().sub(&a0.mul(a1));
        let t2 = a1.square().sub(&a0.mul(a2));
        let n = a0
            .mul(&t0)
            .add(&a2.mul(&t1).add(&a1.mul(&t2)).mul_by_xi())
            .inverse()?;
        Some(Fp6::new(t0.mul(&n), t1.mul(&n), t2.mul(&n)))
    }
}

/// `c0 + c1 w`.
pub(crate) struct Fp12<E, const N: usize> {
    pub(crate) c0: Fp6<E, N>,
    pub(crate) c1: Fp6<E, N>,
}

impl<E, const N: usize> Clone for Fp12<E, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, const N: usize> Copy for Fp12<E, N> {}

impl<E, const N: usize> PartialEq for Fp12<E, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl<E, const N: usize> Eq for Fp12<E, N> {}

impl<E: Engine<N>, const N: usize> Fp12<E, N> {
    #[inline(always)]
    pub(crate) const fn new(c0: Fp6<E, N>, c1: Fp6<E, N>) -> Self {
        Fp12 { c0, c1 }
    }

    #[inline(always)]
    pub(crate) const fn one() -> Self {
        Fp12::new(Fp6::one(), Fp6::zero())
    }

    #[inline(always)]
    pub(crate) fn is_one(&self) -> bool {
        *self == Self::one()
    }

    #[inline(always)]
    pub(crate) fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    #[inline(always)]
    pub(crate) fn conjugate(&self) -> Self {
        Fp12::new(self.c0, self.c1.neg())
    }

    /// Karatsuba over `F_p⁶`: 18 products of `F_p²`.
    #[inline]
    pub(crate) fn mul(&self, b: &Self) -> Self {
        let v0 = self.c0.mul(&b.c0);
        let v1 = self.c1.mul(&b.c1);
        let c1 = self
            .c0
            .add(&self.c1)
            .mul(&b.c0.add(&b.c1))
            .sub(&v0)
            .sub(&v1);
        Fp12::new(v0.add(&v1.mul_by_v()), c1)
    }

    /// The complex squaring over `F_p⁶`: two products.
    #[inline]
    pub(crate) fn square(&self) -> Self {
        let (a0, a1) = (&self.c0, &self.c1);
        let ab = a0.mul(a1);
        let c0 = a0
            .add(a1)
            .mul(&a0.add(&a1.mul_by_v()))
            .sub(&ab)
            .sub(&ab.mul_by_v());
        Fp12::new(c0, ab.double())
    }

    /// `1 / self`, `None` for 0.
    pub(crate) fn inverse(&self) -> Option<Self> {
        let n = self
            .c0
            .square()
            .sub(&self.c1.square().mul_by_v())
            .inverse()?;
        Some(Fp12::new(self.c0.mul(&n), self.c1.mul(&n).neg()))
    }

    /// The `p^k`-th power, by the tower's Frobenius coefficients.
    pub(crate) fn frobenius(&self, k: usize) -> Self {
        let k = k % 12;
        if k == 0 {
            return *self;
        }
        let t = E::tables();
        let f = |x: &Fp2<E, N>, i: usize| -> Fp2<E, N> {
            let c = &t.frob[k][i];
            let x = x.frobenius(k);
            if c.c1.is_zero() {
                x.mul_by_fp(&c.c0)
            } else {
                x.mul(c)
            }
        };
        Fp12::new(
            Fp6::new(
                self.c0.c0.frobenius(k),
                f(&self.c0.c1, 2),
                f(&self.c0.c2, 4),
            ),
            Fp6::new(f(&self.c1.c0, 1), f(&self.c1.c1, 3), f(&self.c1.c2, 5)),
        )
    }

    /// `f · (c₀ + c₃ w + c₄ v w)`, the line of a D-type twist.
    #[inline]
    pub(crate) fn mul_by_034(&self, c0: &Fp2<E, N>, c3: &Fp2<E, N>, c4: &Fp2<E, N>) -> Self {
        let (a, b) = (&self.c0, &self.c1);
        let t0 = a.mul_by_fp2(c0);
        let t1 = b.mul_by_01(c3, c4);
        let c1 = a.add(b).mul_by_01(&c0.add(c3), c4).sub(&t0).sub(&t1);
        Fp12::new(t0.add(&t1.mul_by_v()), c1)
    }

    /// `f · (c₀ + c₁ v + c₄ v w)`, the line of an M-type twist.
    #[inline]
    pub(crate) fn mul_by_014(&self, c0: &Fp2<E, N>, c1: &Fp2<E, N>, c4: &Fp2<E, N>) -> Self {
        let (a, b) = (&self.c0, &self.c1);
        let t0 = a.mul_by_01(c0, c1);
        let t1 = b.mul_by_1(c4);
        let s = a.add(b).mul_by_01(c0, &c1.add(c4)).sub(&t0).sub(&t1);
        Fp12::new(t0.add(&t1.mul_by_v()), s)
    }

    /// Whether this is in the cyclotomic subgroup: `f^(p⁴) · f = f^(p²)`, and not 0.
    pub(crate) fn is_cyclotomic(&self) -> bool {
        !self.is_zero() && self.frobenius(4).mul(self) == self.frobenius(2)
    }

    /// Granger and Scott's square in the cyclotomic subgroup, the formulas of the public tower.
    #[inline]
    pub(crate) fn cyclotomic_square(&self) -> Self {
        let (g0, g1, g2) = (&self.c0.c0, &self.c0.c1, &self.c0.c2);
        let (h0, h1, h2) = (&self.c1.c0, &self.c1.c1, &self.c1.c2);
        let sq4 = |x0: &Fp2<E, N>, x1: &Fp2<E, N>| {
            let a = x0.square();
            let b = x1.square();
            let c = x0.add(x1).square().sub(&a).sub(&b);
            (a.add(&b.mul_by_xi()), c)
        };
        // 3x ± 2y as 2(x ± y) + x: three sums instead of four.
        let plus = |x: &Fp2<E, N>, y: &Fp2<E, N>| x.add(y).double().add(x);
        let minus = |x: &Fp2<E, N>, y: &Fp2<E, N>| x.sub(y).double().add(x);
        let (a0, a1) = sq4(g0, h1);
        let (b0, b1) = sq4(h0, g2);
        let (c0, c1) = sq4(g1, h2);
        Fp12::new(
            Fp6::new(minus(&a0, g0), minus(&b0, g1), minus(&c0, g2)),
            Fp6::new(plus(&c1.mul_by_xi(), h0), plus(&a1, h1), plus(&b1, h2)),
        )
    }

    /// Karabina's compressed form of an element of the cyclotomic subgroup ("Squaring in
    /// cyclotomic subgroups", Math. Comp. 82, 2013): with `f = Σ fᵢzⁱ` over `F_p²` and `z = w`,
    /// `z² = v`, the coefficients `(f₁, f₄, f₂, f₅)` of `z`, `z⁴`, `z²` and `z⁵`, which are
    /// Karabina's `(g₂, g₃, g₄, g₅)`; the other two follow from them on the subgroup.
    #[inline]
    fn compress(&self) -> [Fp2<E, N>; 4] {
        [self.c1.c0, self.c0.c2, self.c0.c1, self.c1.c2]
    }

    /// The square in the compressed form, six squares in `F_p²`:
    /// `h₂ = 2g₂ + 6ξg₄g₅`, `h₃ = 3(g₄² + ξg₅²) − 2g₃`, `h₄ = 3(g₂² + ξg₃²) − 2g₄` and
    /// `h₅ = 2g₅ + 6g₂g₃`, the products `2gᵢgⱼ` as `(gᵢ + gⱼ)² − gᵢ² − gⱼ²`.
    #[inline]
    fn compressed_square(c: &[Fp2<E, N>; 4]) -> [Fp2<E, N>; 4] {
        let [g2, g3, g4, g5] = c;
        let s4 = g4.square();
        let s5 = g5.square();
        let s2 = g2.square();
        let s3 = g3.square();
        let t45 = g4.add(g5).square().sub(&s4).sub(&s5);
        let t23 = g2.add(g3).square().sub(&s2).sub(&s3);
        // 3x ± 2y as 2(x ± y) + x: three sums instead of four.
        let plus = |x: &Fp2<E, N>, y: &Fp2<E, N>| x.add(y).double().add(x);
        let minus = |x: &Fp2<E, N>, y: &Fp2<E, N>| x.sub(y).double().add(x);
        let h2 = plus(&t45.mul_by_xi(), g2);
        let h3 = minus(&s4.add(&s5.mul_by_xi()), g3);
        let h4 = minus(&s2.add(&s3.mul_by_xi()), g4);
        let h5 = plus(&t23, g5);
        [h2, h3, h4, h5]
    }

    /// The element of a compressed form, or `None` when `g₂ = g₃ = 0`, where it is not
    /// determined: `g₁ = (ξg₅² + 3g₄² − 2g₃)/(4g₂)`, or `2g₄g₅/g₃` when `g₂ = 0`, and
    /// `g₀ = ξ(2g₁² + g₂g₅ − 3g₃g₄) + 1`.
    fn decompress(c: &[Fp2<E, N>; 4]) -> Option<Self> {
        let [g2, g3, g4, g5] = c;
        let g1 = if g2.is_zero() {
            g4.mul(g5).double().mul(&g3.inverse()?)
        } else {
            let s4 = g4.square();
            let n = g5
                .square()
                .mul_by_xi()
                .add(&s4.double().add(&s4))
                .sub(&g3.double());
            n.mul(&g2.double().double().inverse()?)
        };
        let t = g3.mul(g4);
        let g0 = g1
            .square()
            .double()
            .add(&g2.mul(g5))
            .sub(&t.double().add(&t))
            .mul_by_xi()
            .add(&Fp2::one());
        Some(Fp12::new(Fp6::new(g0, *g4, *g3), Fp6::new(*g2, g1, *g5)))
    }

    /// `self^(2ⁿ)` on the cyclotomic subgroup: Granger and Scott's squares for short runs,
    /// Karabina's compressed ones for long runs, whose decompression costs an inversion.
    fn cyclotomic_square_n(&self, n: usize) -> Self {
        // The shortest run worth compressing (an inversion against n·6 products in F_p). Measured
        // on an Apple M4 Max, runs of 1 to 32 squares each way, alternated: the compressed run
        // takes 1.04 times the plain one's time at 8 squares on BN254 and BLS12-381, 0.99 at 9,
        // 0.85–0.88 at 14–16 and 0.75 at 32.
        if n >= 9 {
            let mut c = self.compress();
            for _ in 0..n {
                c = Self::compressed_square(&c);
            }
            if let Some(r) = Self::decompress(&c) {
                return r;
            }
        }
        let mut r = *self;
        for _ in 0..n {
            r = r.cyclotomic_square();
        }
        r
    }

    /// `self^d` on the cyclotomic subgroup, for the signed binary digits `d`, least significant
    /// first, the top one non-zero: `f⁻¹` is the conjugate there.
    pub(crate) fn cyclotomic_pow_digits(&self, digits: &[i8]) -> Self {
        let Some((&top, rest)) = digits.split_last() else {
            return Self::one();
        };
        let inv = self.conjugate();
        let mut r = if top > 0 { *self } else { inv };
        let mut pending = 0;
        for &d in rest.iter().rev() {
            pending += 1;
            if d != 0 {
                r = r.cyclotomic_square_n(pending);
                pending = 0;
                r = r.mul(if d > 0 { self } else { &inv });
            }
        }
        r.cyclotomic_square_n(pending)
    }

    /// The twelve coefficients of `F_p`, in the public tower's order.
    pub(crate) fn coefficients(&self) -> [Fp<E, N>; 12] {
        let f2 = [
            &self.c0.c0,
            &self.c0.c1,
            &self.c0.c2,
            &self.c1.c0,
            &self.c1.c1,
            &self.c1.c2,
        ];
        core::array::from_fn(|i| {
            if i % 2 == 0 {
                f2[i / 2].c0
            } else {
                f2[i / 2].c1
            }
        })
    }

    /// The element of twelve coefficients in the public tower's order.
    pub(crate) fn from_coefficients(c: &[Fp<E, N>; 12]) -> Self {
        let e = |i: usize| Fp2::new(c[i], c[i + 1]);
        Fp12::new(Fp6::new(e(0), e(2), e(4)), Fp6::new(e(6), e(8), e(10)))
    }
}
