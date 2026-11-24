//! `F_p⁶ = F_p²[v] / (v³ − ξ)`, the second floor of a [`Tower`].

use super::tower::{Tower, tower_ops};
use super::tower_fp2::Fp2;
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};

/// An element `c₀ + c₁ v + c₂ v²` of `F_p⁶`, with coefficients in [`Fp2`], which borrows its
/// [`Tower`].
///
/// Products are Karatsuba's (Devegili, Ó hÉigeartaigh, Scott and Dahab, ePrint 2006/471, §4), six
/// products of `F_p²`; squares are Chung and Hasan's SQR2 (*Asymmetric squaring formulae*, ARITH
/// 2007), two products and three squares; the inverse is the adjugate over the norm to `F_p²`, one
/// inversion of `F_p²`.
///
/// # Panics
///
/// The operators panic for elements of different towers, and `/` for a zero divisor.
///
/// # Examples
///
/// ```
/// use a::math::field::{Fp2, Fp6, Tower};
/// use a::math::numbers::Number;
///
/// let t = Tower::bn254();
/// let v = Fp6::new(Fp2::zero(t), Fp2::one(t), Fp2::zero(t));
/// // v³ = ξ = 9 + u.
/// assert_eq!(&(&v * &v) * &v, Fp6::from_fp2(t.xi()));
/// let x = Fp6::new(t.xi(), Fp2::one(t), t.xi().square());
/// assert_eq!(&x / &x, Fp6::one(t));
/// assert_eq!(Fp6::from_fp2(x.norm()), &(&x * &x.frobenius(2)) * &x.frobenius(4));
/// ```
#[derive(Clone)]
pub struct Fp6<'t> {
    /// The coefficient of 1.
    pub(super) c0: Fp2<'t>,
    /// The coefficient of `v`.
    pub(super) c1: Fp2<'t>,
    /// The coefficient of `v²`.
    pub(super) c2: Fp2<'t>,
}

impl<'t> Fp6<'t> {
    /// `c₀ + c₁ v + c₂ v²`.
    ///
    /// # Panics
    ///
    /// Panics if the coefficients are of different towers.
    #[track_caller]
    #[must_use]
    pub fn new(c0: Fp2<'t>, c1: Fp2<'t>, c2: Fp2<'t>) -> Fp6<'t> {
        c0.t.check(c1.t);
        c0.t.check(c2.t);
        Fp6 { c0, c1, c2 }
    }

    /// The element `c₀` of `F_p²` in `F_p⁶`.
    #[must_use]
    pub fn from_fp2(c0: Fp2<'t>) -> Fp6<'t> {
        let t = c0.t;
        Fp6 {
            c0,
            c1: Fp2::zero(t),
            c2: Fp2::zero(t),
        }
    }

    /// 0.
    #[must_use]
    pub fn zero(t: &'t Tower<'t>) -> Fp6<'t> {
        Fp6::from_fp2(Fp2::zero(t))
    }

    /// 1.
    #[must_use]
    pub fn one(t: &'t Tower<'t>) -> Fp6<'t> {
        Fp6::from_fp2(Fp2::one(t))
    }

    /// The tower.
    #[must_use]
    pub fn tower(&self) -> &'t Tower<'t> {
        self.c0.t
    }

    /// The coefficient of 1.
    #[must_use]
    pub fn c0(&self) -> &Fp2<'t> {
        &self.c0
    }

    /// The coefficient of `v`.
    #[must_use]
    pub fn c1(&self) -> &Fp2<'t> {
        &self.c1
    }

    /// The coefficient of `v²`.
    #[must_use]
    pub fn c2(&self) -> &Fp2<'t> {
        &self.c2
    }

    /// Whether it is 0.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }

    /// Whether it is 1.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.c0.is_one() && self.c1.is_zero() && self.c2.is_zero()
    }

    /// The square, by Chung and Hasan's SQR2: `s₀ = c₀²`, `s₁ = 2c₀c₁`, `s₂ = (c₀ − c₁ + c₂)²`,
    /// `s₃ = 2c₁c₂`, `s₄ = c₂²`, and then `s₀ + ξs₃`, `s₁ + ξs₄`, `s₁ + s₂ + s₃ − s₀ − s₄`.
    #[must_use]
    pub fn square(&self) -> Fp6<'t> {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        let s0 = a0.square();
        let s1 = (a0 * a1).double();
        let s2 = (a0 - a1 + a2).square();
        let s3 = (a1 * a2).double();
        let s4 = a2.square();
        Fp6 {
            c0: &s0 + &s3.mul_by_xi(),
            c1: &s1 + &s4.mul_by_xi(),
            c2: s1 + s2 + s3 - s0 - s4,
        }
    }

    /// Twice the element.
    #[must_use]
    pub fn double(&self) -> Fp6<'t> {
        Fp6 {
            c0: self.c0.double(),
            c1: self.c1.double(),
            c2: self.c2.double(),
        }
    }

    /// The product by `v`, the non-residue of `F_p¹² = F_p⁶[w] / (w² − v)`:
    /// `ξ c₂ + c₀ v + c₁ v²`.
    #[must_use]
    pub fn mul_by_v(&self) -> Fp6<'t> {
        Fp6 {
            c0: self.c2.mul_by_xi(),
            c1: self.c0.clone(),
            c2: self.c1.clone(),
        }
    }

    /// The product by an element of `F_p²`: three products of `F_p²`.
    ///
    /// # Panics
    ///
    /// Panics if `k` is of a different tower.
    #[track_caller]
    #[must_use]
    pub fn mul_by_fp2(&self, k: &Fp2<'t>) -> Fp6<'t> {
        Fp6 {
            c0: &self.c0 * k,
            c1: &self.c1 * k,
            c2: &self.c2 * k,
        }
    }

    /// The product by `b₀ + b₁ v`: five products of `F_p²` instead of six.
    ///
    /// # Panics
    ///
    /// Panics if `b₀` or `b₁` is of a different tower.
    #[track_caller]
    #[must_use]
    pub fn mul_by_01(&self, b0: &Fp2<'t>, b1: &Fp2<'t>) -> Fp6<'t> {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        Fp6 {
            c0: ((a1 + a2) * b1 - &v1).mul_by_xi() + &v0,
            c1: (a0 + a1) * (b0 + b1) - &v0 - &v1,
            c2: (a0 + a2) * b0 - v0 + v1,
        }
    }

    /// The product by `b₁ v`: three products of `F_p²`.
    ///
    /// # Panics
    ///
    /// Panics if `b₁` is of a different tower.
    #[track_caller]
    #[must_use]
    pub fn mul_by_1(&self, b1: &Fp2<'t>) -> Fp6<'t> {
        Fp6 {
            c0: (&self.c2 * b1).mul_by_xi(),
            c1: &self.c0 * b1,
            c2: &self.c1 * b1,
        }
    }

    /// `x^(pᵏ)`, by the coefficients of the tower: the Frobenius map of `F_p²` on each coefficient,
    /// and `v^(pᵏ) = ξ^((pᵏ − 1)/3) v`.
    #[must_use]
    pub fn frobenius(&self, k: usize) -> Fp6<'t> {
        let t = self.c0.t;
        let k = k % 6;
        Fp6 {
            c0: self.c0.frobenius(k),
            c1: &self.c1.frobenius(k) * &t.frobenius_coefficient(k, 2),
            c2: &self.c2.frobenius(k) * &t.frobenius_coefficient(k, 4),
        }
    }

    /// The norm to `F_p²`, `x · x^(p²) · x^(p⁴) = c₀³ + ξ c₁³ + ξ² c₂³ − 3ξ c₀c₁c₂`.
    #[must_use]
    pub fn norm(&self) -> Fp2<'t> {
        let (a, b, c) = self.adjugate();
        self.norm_from(&a, &b, &c)
    }

    /// The trace to `F_p²`, `x + x^(p²) + x^(p⁴) = 3 c₀`.
    #[must_use]
    pub fn trace(&self) -> Fp2<'t> {
        &self.c0.double() + &self.c0
    }

    /// The inverse, or `None` for 0: the adjugate `(c₀² − ξc₁c₂, ξc₂² − c₀c₁, c₁² − c₀c₂)` over
    /// the norm, which is `c₀ A + ξ (c₂ B + c₁ C)` for that adjugate `(A, B, C)`.
    #[must_use]
    pub fn inverse(&self) -> Option<Fp6<'t>> {
        let (a, b, c) = self.adjugate();
        let n = self.norm_from(&a, &b, &c).inverse()?;
        Some(Fp6 {
            c0: &a * &n,
            c1: &b * &n,
            c2: &c * &n,
        })
    }

    /// Whether it is a square of `F_p⁶` (0 is): when its norm to `F_p²` is a square there, since
    /// `N(x)^((p² − 1)/2) = x^((p⁶ − 1)/2)`.
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.norm().is_square()
    }

    /// The `e`-th power, by squaring and multiplying; a negative `e` is a power of the inverse.
    ///
    /// # Panics
    ///
    /// Panics if `e` is negative and the element is 0.
    #[track_caller]
    #[must_use]
    pub fn pow(&self, e: &Number) -> Fp6<'t> {
        if e.is_negative() {
            let inv = self.inverse().expect("0 has no inverse");
            return inv.pow(&-e);
        }
        let mut r = Fp6::one(self.c0.t);
        for i in (0..e.bits()).rev() {
            r = r.square();
            if e.bit(i) {
                r = &r * self;
            }
        }
        r
    }

    /// The adjugate `(A, B, C)`: `x · (A + B v + C v²)` is the norm.
    fn adjugate(&self) -> (Fp2<'t>, Fp2<'t>, Fp2<'t>) {
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        (
            a0.square() - (a1 * a2).mul_by_xi(),
            a2.square().mul_by_xi() - a0 * a1,
            a1.square() - a0 * a2,
        )
    }

    /// The norm from the adjugate.
    fn norm_from(&self, a: &Fp2<'t>, b: &Fp2<'t>, c: &Fp2<'t>) -> Fp2<'t> {
        &self.c0 * a + (&self.c2 * b + &self.c1 * c).mul_by_xi()
    }

    /// The sum.
    #[track_caller]
    pub(super) fn add_ref(a: &Fp6<'t>, b: &Fp6<'t>) -> Fp6<'t> {
        Fp6 {
            c0: &a.c0 + &b.c0,
            c1: &a.c1 + &b.c1,
            c2: &a.c2 + &b.c2,
        }
    }

    /// The difference.
    #[track_caller]
    pub(super) fn sub_ref(a: &Fp6<'t>, b: &Fp6<'t>) -> Fp6<'t> {
        Fp6 {
            c0: &a.c0 - &b.c0,
            c1: &a.c1 - &b.c1,
            c2: &a.c2 - &b.c2,
        }
    }

    /// The product, by Karatsuba: `vᵢ = aᵢbᵢ`, and then
    /// `c₀ = v₀ + ξ((a₁ + a₂)(b₁ + b₂) − v₁ − v₂)`, `c₁ = (a₀ + a₁)(b₀ + b₁) − v₀ − v₁ + ξv₂`,
    /// `c₂ = (a₀ + a₂)(b₀ + b₂) − v₀ − v₂ + v₁`.
    #[track_caller]
    pub(super) fn mul_ref(a: &Fp6<'t>, b: &Fp6<'t>) -> Fp6<'t> {
        let (a0, a1, a2) = (&a.c0, &a.c1, &a.c2);
        let (b0, b1, b2) = (&b.c0, &b.c1, &b.c2);
        let v0 = a0 * b0;
        let v1 = a1 * b1;
        let v2 = a2 * b2;
        Fp6 {
            c0: ((a1 + a2) * (b1 + b2) - &v1 - &v2).mul_by_xi() + &v0,
            c1: (a0 + a1) * (b0 + b1) - &v0 - &v1 + v2.mul_by_xi(),
            c2: (a0 + a2) * (b0 + b2) - v0 - &v2 + v1,
        }
    }

    /// The quotient.
    #[track_caller]
    pub(super) fn div_ref(a: &Fp6<'t>, b: &Fp6<'t>) -> Fp6<'t> {
        let inv = b.inverse().expect("division by zero");
        Fp6::mul_ref(a, &inv)
    }

    /// The negation.
    pub(super) fn neg_ref(&self) -> Fp6<'t> {
        Fp6 {
            c0: -&self.c0,
            c1: -&self.c1,
            c2: -&self.c2,
        }
    }
}

tower_ops!(Fp6);

impl PartialEq for Fp6<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1 && self.c2 == other.c2
    }
}

impl Eq for Fp6<'_> {}

impl Hash for Fp6<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.c0.hash(h);
        self.c1.hash(h);
        self.c2.hash(h);
    }
}

impl fmt::Debug for Fp6<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `(c₀) + (c₁)·v + (c₂)·v²` in decimal.
impl fmt::Display for Fp6<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) + ({})·v + ({})·v²", self.c0, self.c1, self.c2)
    }
}
