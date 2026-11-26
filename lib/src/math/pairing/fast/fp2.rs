//! `F_p² = F_p[u] / (u² + 1)`, as both curves' towers have it (`β = −1`).

use super::Engine;
use super::fp::{Fp, mul_wide, wide_add, wide_sub_plain};

/// `c0 + c1 u`.
pub(crate) struct Fp2<E, const N: usize> {
    pub(crate) c0: Fp<E, N>,
    pub(crate) c1: Fp<E, N>,
}

impl<E, const N: usize> Clone for Fp2<E, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, const N: usize> Copy for Fp2<E, N> {}

impl<E, const N: usize> PartialEq for Fp2<E, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl<E, const N: usize> Eq for Fp2<E, N> {}

impl<E: Engine<N>, const N: usize> Fp2<E, N> {
    #[inline(always)]
    pub(crate) const fn new(c0: Fp<E, N>, c1: Fp<E, N>) -> Self {
        Fp2 { c0, c1 }
    }

    #[inline(always)]
    pub(crate) const fn zero() -> Self {
        Fp2::new(Fp::zero(), Fp::zero())
    }

    #[inline(always)]
    pub(crate) const fn one() -> Self {
        Fp2::new(Fp::one(), Fp::zero())
    }

    #[inline(always)]
    pub(crate) fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    #[inline(always)]
    pub(crate) fn add(&self, o: &Self) -> Self {
        Fp2::new(self.c0.add(&o.c0), self.c1.add(&o.c1))
    }

    #[inline(always)]
    pub(crate) fn sub(&self, o: &Self) -> Self {
        Fp2::new(self.c0.sub(&o.c0), self.c1.sub(&o.c1))
    }

    #[inline(always)]
    pub(crate) fn double(&self) -> Self {
        Fp2::new(self.c0.double(), self.c1.double())
    }

    #[inline(always)]
    pub(crate) fn neg(&self) -> Self {
        Fp2::new(self.c0.neg(), self.c1.neg())
    }

    #[inline(always)]
    pub(crate) fn conjugate(&self) -> Self {
        Fp2::new(self.c0, self.c1.neg())
    }

    /// The `p^k`-th power: the conjugate for odd `k`.
    #[inline(always)]
    pub(crate) fn frobenius(&self, k: usize) -> Self {
        if k % 2 == 1 { self.conjugate() } else { *self }
    }

    /// Karatsuba's three products, reduced twice instead of three times:
    /// `(a₀b₀ − a₁b₁) + ((a₀ + a₁)(b₀ + b₁) − a₀b₀ − a₁b₁) u`, the products of double width
    /// (the sums `a₀ + a₁ < 2p` unreduced), each coefficient one Montgomery reduction.
    #[inline(always)]
    pub(crate) fn mul(&self, o: &Self) -> Self {
        let v0 = mul_wide::<N>(&self.c0.0, &o.c0.0);
        let v1 = mul_wide::<N>(&self.c1.0, &o.c1.0);
        let s = mul_wide::<N>(&self.c0.add_lazy(&self.c1), &o.c0.add_lazy(&o.c1));
        let c0 = Fp::redc(&Fp::<E, N>::wide_sub(&v0, &v1));
        let c1 = Fp::redc(&wide_sub_plain::<N>(&s, &wide_add::<N>(&v0, &v1)));
        Fp2::new(c0, c1)
    }

    /// `(a₀ + a₁)(a₀ − a₁) + 2a₀a₁ u`: two Montgomery products (on aarch64 the kernels' products
    /// of reduced factors are faster than the double-width products of unreduced ones, by 7 % in
    /// a chain of dependent squares).
    #[inline(always)]
    pub(crate) fn square(&self) -> Self {
        let s = self.c0.add(&self.c1).mul(&self.c0.sub(&self.c1));
        let t = self.c0.mul(&self.c1);
        Fp2::new(s, t.double())
    }

    /// `self / 2`.
    #[inline(always)]
    pub(crate) fn half(&self) -> Self {
        Fp2::new(self.c0.half(), self.c1.half())
    }

    /// The product by an element of `F_p`.
    #[inline(always)]
    pub(crate) fn mul_by_fp(&self, k: &Fp<E, N>) -> Self {
        Fp2::new(self.c0.mul(k), self.c1.mul(k))
    }

    /// The product by `ξ = XI0 + u`: `(XI0 a₀ − a₁) + (a₀ + XI0 a₁) u`.
    #[inline(always)]
    pub(crate) fn mul_by_xi(&self) -> Self {
        let (a0, a1) = (self.c0, self.c1);
        let (k0, k1) = match E::XI0 {
            1 => (a0, a1),
            9 => {
                let e0 = a0.double().double().double();
                let e1 = a1.double().double().double();
                (e0.add(&a0), e1.add(&a1))
            }
            _ => unreachable!("ξ is 1 + u or 9 + u"),
        };
        Fp2::new(k0.sub(&a1), a0.add(&k1))
    }

    /// The norm `a₀² + a₁²`.
    #[inline(always)]
    pub(crate) fn norm(&self) -> Fp<E, N> {
        self.c0.square().add(&self.c1.square())
    }

    /// `1 / self`, `None` for 0.
    pub(crate) fn inverse(&self) -> Option<Self> {
        let n = self.norm().inverse()?;
        Some(Fp2::new(self.c0.mul(&n), self.c1.mul(&n).neg()))
    }

    /// RFC 9380's `sgn0` for `m = 2`: the parity of `c₀`, or of `c₁` when `c₀ = 0`.
    pub(crate) fn sgn0(&self) -> bool {
        let s0 = self.c0.is_odd();
        let z0 = self.c0.is_zero();
        s0 || (z0 && self.c1.is_odd())
    }

    /// A square root, if there is one, by the complex method: with `n = √(a₀² + a₁²)`,
    /// `x₀ = √((a₀ + n)/2)` (or with `−n`) and `x₁ = a₁ / (2x₀)`.
    pub(crate) fn sqrt(&self) -> Option<Self> {
        if self.c1.is_zero() {
            // An element of F_p: √a₀ if a₀ is a square, else √(−a₀) u.
            return match self.c0.sqrt() {
                Some(r) => Some(Fp2::new(r, Fp::zero())),
                None => self.c0.neg().sqrt().map(|r| Fp2::new(Fp::zero(), r)),
            };
        }
        let n = self.norm().sqrt()?;
        let half = E::tables().half;
        let mut d = self.c0.add(&n).mul(&half);
        let x0 = match d.sqrt() {
            Some(x) => x,
            None => {
                d = self.c0.sub(&n).mul(&half);
                d.sqrt()?
            }
        };
        let x1 = self.c1.mul(&x0.double().inverse()?);
        let r = Fp2::new(x0, x1);
        (r.square() == *self).then_some(r)
    }
}
