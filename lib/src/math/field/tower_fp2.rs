//! `F_p² = F_p[u] / (u² − β)`, the first floor of a [`Tower`].

use super::prime::Fp;
use super::tower::{Tower, tower_ops};
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};

/// An element `c₀ + c₁ u` of `F_p²`, which borrows its [`Tower`].
///
/// Products are Karatsuba's, three products of `F_p`; squares are the complex squaring,
/// `(c₀ + c₁)(c₀ + β c₁) − (1 + β) c₀ c₁`, two products; the inverse is the conjugate over the norm
/// `c₀² − β c₁²`, one inversion of `F_p`.
///
/// # Panics
///
/// The operators panic for elements of different towers, and `/` for a zero divisor.
///
/// # Examples
///
/// ```
/// use a::math::field::{Fp2, Tower};
/// use a::math::numbers::Number;
///
/// let t = Tower::bls12_381();
/// let n = |x: u32| Number::from(x);
/// let a = Fp2::from_numbers(t, &n(1), &n(2)); // 1 + 2u
/// let b = Fp2::from_numbers(t, &n(3), &n(4)); // 3 + 4u
/// // (1 + 2u)(3 + 4u) = 3 − 8 + (4 + 6) u, as u² = −1.
/// assert_eq!(&a * &b, Fp2::new(t, -t.base().from_u64(5), t.base().from_u64(10)));
/// assert_eq!(&(&a * &b) / &b, a);
/// let r = b.sqrt().unwrap();
/// assert_eq!(r.square(), b);
/// ```
#[derive(Clone)]
pub struct Fp2<'t> {
    /// The tower.
    pub(super) t: &'t Tower<'t>,
    /// The coefficient of 1.
    pub(super) c0: Fp<'t>,
    /// The coefficient of `u`.
    pub(super) c1: Fp<'t>,
}

impl<'t> Fp2<'t> {
    /// `c₀ + c₁ u`.
    ///
    /// # Panics
    ///
    /// Panics if `c₀` or `c₁` is not an element of the tower's prime field.
    #[track_caller]
    #[must_use]
    pub fn new(t: &'t Tower<'t>, c0: Fp<'t>, c1: Fp<'t>) -> Fp2<'t> {
        let p = t.base.modulus();
        assert!(
            c0.field().modulus() == p && c1.field().modulus() == p,
            "elements of different fields"
        );
        Fp2 { t, c0, c1 }
    }

    /// `c₀ + c₁ u`, each reduced modulo `p`.
    #[must_use]
    pub fn from_numbers(t: &'t Tower<'t>, c0: &Number, c1: &Number) -> Fp2<'t> {
        Fp2 {
            t,
            c0: t.base.element(c0),
            c1: t.base.element(c1),
        }
    }

    /// The element `c₀` of `F_p` in `F_p²`.
    ///
    /// # Panics
    ///
    /// Panics if `c₀` is not an element of the tower's prime field.
    #[track_caller]
    #[must_use]
    pub fn from_base(t: &'t Tower<'t>, c0: Fp<'t>) -> Fp2<'t> {
        let zero = t.base.zero();
        Fp2::new(t, c0, zero)
    }

    /// 0.
    #[must_use]
    pub fn zero(t: &'t Tower<'t>) -> Fp2<'t> {
        Fp2 {
            t,
            c0: t.base.zero(),
            c1: t.base.zero(),
        }
    }

    /// 1.
    #[must_use]
    pub fn one(t: &'t Tower<'t>) -> Fp2<'t> {
        Fp2 {
            t,
            c0: t.base.one(),
            c1: t.base.zero(),
        }
    }

    /// The tower.
    #[must_use]
    pub fn tower(&self) -> &'t Tower<'t> {
        self.t
    }

    /// The coefficient of 1.
    #[must_use]
    pub fn c0(&self) -> &Fp<'t> {
        &self.c0
    }

    /// The coefficient of `u`.
    #[must_use]
    pub fn c1(&self) -> &Fp<'t> {
        &self.c1
    }

    /// Whether it is 0.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    /// Whether it is 1.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.c0.is_one() && self.c1.is_zero()
    }

    /// Whether it lies in `F_p`: `c₁ = 0`.
    #[must_use]
    pub fn is_in_base(&self) -> bool {
        self.c1.is_zero()
    }

    /// The square, by the complex method: `(c₀ + c₁)(c₀ + βc₁) − (1 + β)c₀c₁` and `2c₀c₁`, two
    /// products of `F_p`, and two products by `β` unless `β = −1`.
    #[must_use]
    pub fn square(&self) -> Fp2<'t> {
        let t = self.t;
        let (a0, a1) = (&self.c0, &self.c1);
        let ab = a0 * a1;
        let c0 = if t.beta_is_minus_one {
            (a0 + a1) * (a0 - a1)
        } else {
            // (a₀ + a₁)(a₀ + β a₁) − a₀a₁ − β a₀a₁ = a₀² + β a₁².
            (a0 + a1) * (a0 + &t.mul_by_beta(a1)) - &ab - t.mul_by_beta(&ab)
        };
        Fp2 {
            t,
            c0,
            c1: ab.double(),
        }
    }

    /// Twice the element.
    #[must_use]
    pub fn double(&self) -> Fp2<'t> {
        Fp2 {
            t: self.t,
            c0: self.c0.double(),
            c1: self.c1.double(),
        }
    }

    /// The conjugate `c₀ − c₁ u`, the image under the Frobenius map `x ↦ xᵖ`.
    #[must_use]
    pub fn conjugate(&self) -> Fp2<'t> {
        Fp2 {
            t: self.t,
            c0: self.c0.clone(),
            c1: -&self.c1,
        }
    }

    /// `x^(pᵏ)`: the element for even `k`, its conjugate for odd `k`.
    #[must_use]
    pub fn frobenius(&self, k: usize) -> Fp2<'t> {
        if k % 2 == 1 {
            self.conjugate()
        } else {
            self.clone()
        }
    }

    /// The norm to `F_p`, `x · x̄ = c₀² − β c₁²`.
    #[must_use]
    pub fn norm(&self) -> Fp<'t> {
        self.c0.square() - self.t.mul_by_beta(&self.c1.square())
    }

    /// The trace to `F_p`, `x + x̄ = 2 c₀`.
    #[must_use]
    pub fn trace(&self) -> Fp<'t> {
        self.c0.double()
    }

    /// The inverse, the conjugate over the norm, or `None` for 0.
    #[must_use]
    pub fn inverse(&self) -> Option<Fp2<'t>> {
        let n = self.norm().inverse()?;
        Some(Fp2 {
            t: self.t,
            c0: &self.c0 * &n,
            c1: -(&self.c1 * &n),
        })
    }

    /// The product by an element of `F_p`.
    ///
    /// # Panics
    ///
    /// Panics if `k` is not an element of the tower's prime field.
    #[track_caller]
    #[must_use]
    pub fn mul_by_base(&self, k: &Fp<'t>) -> Fp2<'t> {
        Fp2 {
            t: self.t,
            c0: &self.c0 * k,
            c1: &self.c1 * k,
        }
    }

    /// The product by `ξ`, the non-residue of `F_p⁶ = F_p²[v] / (v³ − ξ)`: two products of `F_p`
    /// when `ξ₁ = 1`, four otherwise.
    #[must_use]
    pub fn mul_by_xi(&self) -> Fp2<'t> {
        let t = self.t;
        let [x0, x1] = &t.xi;
        let (a0, a1) = (&self.c0, &self.c1);
        if t.xi1_is_one {
            // (a₀ + a₁u)(ξ₀ + u) = a₀ξ₀ + β a₁ + (a₀ + a₁ξ₀) u.
            Fp2 {
                t,
                c0: a0 * x0 + t.mul_by_beta(a1),
                c1: a0 + &(a1 * x0),
            }
        } else {
            Fp2 {
                t,
                c0: a0 * x0 + t.mul_by_beta(&(a1 * x1)),
                c1: a0 * x1 + a1 * x0,
            }
        }
    }

    /// The `e`-th power, by squaring and multiplying; a negative `e` is a power of the inverse.
    ///
    /// # Panics
    ///
    /// Panics if `e` is negative and the element is 0.
    #[track_caller]
    #[must_use]
    pub fn pow(&self, e: &Number) -> Fp2<'t> {
        if e.is_negative() {
            let inv = self.inverse().expect("0 has no inverse");
            return inv.pow(&-e);
        }
        let mut r = Fp2::one(self.t);
        for i in (0..e.bits()).rev() {
            r = r.square();
            if e.bit(i) {
                r = &r * self;
            }
        }
        r
    }

    /// The `e`-th power.
    #[must_use]
    pub fn pow_u64(&self, e: u64) -> Fp2<'t> {
        self.pow(&Number::from(e))
    }

    /// Whether it is a square of `F_p²` (0 is): when its norm is a square of `F_p`, as the norm
    /// maps the squares of `F_p²*` onto those of `F_p*`.
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.norm().legendre() >= 0
    }

    /// The smaller square root in the order of ZCash and arkworks on `F_p²` (0 for 0), or `None` if
    /// the element is not a square: of `±r`, the one whose `c₁` is below `p/2`, or, when `c₁ = 0`,
    /// whose `c₀` is, so that on `F_p` it is the root [`Fp::sqrt`] gives.
    ///
    /// By the norm, the complex method (Scott, *Implementing cryptographic pairings*, Pairing 2007;
    /// Adj and Rodríguez-Henríquez, *Square root computation over even extension fields*, IEEE
    /// Trans. Computers 63, 2014), for any odd `p` and non-square `β`: with `γ = √(c₀² − β c₁²)` in
    /// `F_p`, one of `δ = (c₀ ± γ)/2` is a square, as their product `β c₁²/4` is not, and
    /// `√δ + c₁/(2√δ) · u` is a root. It takes a Legendre symbol, two square roots of `F_p` and an
    /// inversion (one square root when `c₁ = 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Tower};
    /// use a::math::numbers::Number;
    ///
    /// let t = Tower::bn254();
    /// let x = Fp2::from_numbers(t, &Number::from(5u8), &Number::from(11u8));
    /// let r = x.square().sqrt().unwrap();
    /// assert!(r == x || r == -&x);
    /// assert!(r.c1().value() < t.base().modulus() - r.c1().value());
    /// assert_eq!(x.square().mul_by_xi().sqrt(), None); // ξ is not a square
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Option<Fp2<'t>> {
        let t = self.t;
        let r = if self.c1.is_zero() {
            match self.c0.sqrt() {
                Some(r) => Fp2 {
                    t,
                    c0: r,
                    c1: t.base.zero(),
                },
                // c₀ is not a square and neither is β, so c₀/β is: (s u)² = s² β = c₀.
                None => {
                    let s = (&self.c0 / &t.beta).sqrt()?;
                    Fp2 {
                        t,
                        c0: t.base.zero(),
                        c1: s,
                    }
                }
            }
        } else {
            let gamma = self.norm().sqrt()?;
            let mut delta = (&self.c0 + &gamma) * &t.half;
            if delta.legendre() != 1 {
                delta = (&self.c0 - &gamma) * &t.half;
            }
            let x0 = delta.sqrt()?;
            let x1 = &self.c1 * &(x0.double().inverse()?);
            Fp2 { t, c0: x0, c1: x1 }
        };
        Some(if r.is_larger() { -r } else { r })
    }

    /// Whether it is the larger of `±x` in the order of ZCash and arkworks: `c₁ > (p − 1)/2`, or
    /// `c₁ = 0` and `c₀ > (p − 1)/2`.
    fn is_larger(&self) -> bool {
        let p = self.t.base.modulus();
        let large = |x: &Fp<'t>| x.value() * 2u8 > p - 1u8;
        large(&self.c1) || (self.c1.is_zero() && large(&self.c0))
    }

    /// The sign of RFC 9380 (§4.1): the parity of `c₀`, or of `c₁` when `c₀ = 0`.
    #[must_use]
    pub fn sgn0(&self) -> bool {
        let s0 = self.c0.value().bit(0);
        let z0 = self.c0.is_zero();
        let s1 = self.c1.value().bit(0);
        s0 || (z0 && s1)
    }

    /// The sum.
    #[track_caller]
    pub(super) fn add_ref(a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
        a.t.check(b.t);
        Fp2 {
            t: a.t,
            c0: &a.c0 + &b.c0,
            c1: &a.c1 + &b.c1,
        }
    }

    /// The difference.
    #[track_caller]
    pub(super) fn sub_ref(a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
        a.t.check(b.t);
        Fp2 {
            t: a.t,
            c0: &a.c0 - &b.c0,
            c1: &a.c1 - &b.c1,
        }
    }

    /// The product, by Karatsuba: three products of `F_p`.
    #[track_caller]
    pub(super) fn mul_ref(a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
        a.t.check(b.t);
        let t = a.t;
        let v0 = &a.c0 * &b.c0;
        let v1 = &a.c1 * &b.c1;
        let c1 = (&a.c0 + &a.c1) * (&b.c0 + &b.c1) - &v0 - &v1;
        Fp2 {
            t,
            c0: v0 + t.mul_by_beta(&v1),
            c1,
        }
    }

    /// The quotient.
    #[track_caller]
    pub(super) fn div_ref(a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
        let inv = b.inverse().expect("division by zero");
        Fp2::mul_ref(a, &inv)
    }

    /// The negation.
    pub(super) fn neg_ref(&self) -> Fp2<'t> {
        Fp2 {
            t: self.t,
            c0: -&self.c0,
            c1: -&self.c1,
        }
    }
}

tower_ops!(Fp2);

impl PartialEq for Fp2<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1 && self.t.same(other.t)
    }
}

impl Eq for Fp2<'_> {}

impl Hash for Fp2<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.c0.hash(h);
        self.c1.hash(h);
    }
}

impl fmt::Debug for Fp2<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `c₀ + c₁·u` in decimal.
impl fmt::Display for Fp2<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}·u", self.c0, self.c1)
    }
}
