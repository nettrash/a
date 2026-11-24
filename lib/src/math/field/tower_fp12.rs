//! `F_p¹² = F_p⁶[w] / (w² − v)`, the top of a [`Tower`], with the products by line values and the
//! arithmetic of the cyclotomic subgroup that pairings need.

use super::prime::Fp;
use super::tower::{Tower, tower_ops};
use super::tower_fp2::Fp2;
use super::tower_fp6::Fp6;
use crate::math::numbers::Number;
use core::fmt;
use core::hash::{Hash, Hasher};

/// An element `c₀ + c₁ w` of `F_p¹²`, with coefficients in [`Fp6`], which borrows its [`Tower`].
///
/// Its twelve coefficients of `F_p`, in the order of [`coefficients`](Fp12::coefficients), are
/// those of `c₀.c₀`, `c₀.c₁`, `c₀.c₂`, `c₁.c₀`, `c₁.c₁`, `c₁.c₂`, each `c₀` then `c₁` of `F_p²`:
/// the order of arkworks and of the `bls12_381` crate. As a sum of powers of `w` over `F_p²`, the
/// six coefficients `c₀.c₀, c₁.c₀, c₀.c₁, c₁.c₁, c₀.c₂, c₁.c₂` are those of `1, w, w², …, w⁵`.
///
/// Products are Karatsuba's, three products of `F_p⁶`; squares the complex method, two; the
/// inverse is the conjugate over the norm `c₀² − v c₁²` to `F_p⁶`. For the elements of the
/// cyclotomic subgroup, those of order dividing `Φ₁₂(p) = p⁴ − p² + 1` where pairings take their
/// values, [`cyclotomic_square`](Fp12::cyclotomic_square) takes nine squares of `F_p²` instead of
/// twelve products, and the inverse is the conjugate.
///
/// # Panics
///
/// The operators panic for elements of different towers, and `/` for a zero divisor.
///
/// # Examples
///
/// ```
/// use a::math::field::{Fp2, Fp6, Fp12, Tower};
/// use a::math::numbers::Number;
///
/// let t = Tower::bls12_381();
/// let w = Fp12::new(Fp6::zero(t), Fp6::one(t));
/// // w⁶ = ξ = 1 + u.
/// assert_eq!(w.pow(&Number::from(6u8)), Fp12::from_fp6(Fp6::from_fp2(t.xi())));
/// // w^p = ξ^((p − 1)/6) w.
/// let g = t.frobenius_coefficient(1, 1);
/// assert_eq!(w.frobenius(1), Fp12::new(Fp6::zero(t), Fp6::from_fp2(g)));
/// assert_eq!(w.conjugate(), -&w);
/// ```
#[derive(Clone)]
pub struct Fp12<'t> {
    /// The coefficient of 1.
    pub(super) c0: Fp6<'t>,
    /// The coefficient of `w`.
    pub(super) c1: Fp6<'t>,
}

impl<'t> Fp12<'t> {
    /// `c₀ + c₁ w`.
    ///
    /// # Panics
    ///
    /// Panics if the coefficients are of different towers.
    #[track_caller]
    #[must_use]
    pub fn new(c0: Fp6<'t>, c1: Fp6<'t>) -> Fp12<'t> {
        c0.c0.t.check(c1.c0.t);
        Fp12 { c0, c1 }
    }

    /// The element `c₀` of `F_p⁶` in `F_p¹²`.
    #[must_use]
    pub fn from_fp6(c0: Fp6<'t>) -> Fp12<'t> {
        let t = c0.c0.t;
        Fp12 {
            c0,
            c1: Fp6::zero(t),
        }
    }

    /// The element of the twelve coefficients of `F_p` in the order of
    /// [`coefficients`](Fp12::coefficients), each reduced modulo `p`.
    ///
    /// # Panics
    ///
    /// Panics unless there are twelve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp12, Tower};
    /// use a::math::numbers::Number;
    ///
    /// let t = Tower::bn254();
    /// let c: Vec<Number> = (1..=12u32).map(Number::from).collect();
    /// let x = Fp12::from_coefficients(t, &c);
    /// assert_eq!(x.coefficients().map(|c| c.value()).to_vec(), c);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn from_coefficients(t: &'t Tower<'t>, c: &[Number]) -> Fp12<'t> {
        assert!(c.len() == 12, "an element of F_p¹² has twelve coefficients");
        let e = |i: usize| Fp2::from_numbers(t, &c[i], &c[i + 1]);
        Fp12 {
            c0: Fp6 {
                c0: e(0),
                c1: e(2),
                c2: e(4),
            },
            c1: Fp6 {
                c0: e(6),
                c1: e(8),
                c2: e(10),
            },
        }
    }

    /// The twelve coefficients of `F_p`: `c₀.c₀.c₀, c₀.c₀.c₁, c₀.c₁.c₀, …, c₁.c₂.c₁`.
    #[must_use]
    pub fn coefficients(&self) -> [Fp<'t>; 12] {
        let f2 = [
            &self.c0.c0,
            &self.c0.c1,
            &self.c0.c2,
            &self.c1.c0,
            &self.c1.c1,
            &self.c1.c2,
        ];
        core::array::from_fn(|i| {
            let e = f2[i / 2];
            if i % 2 == 0 {
                e.c0.clone()
            } else {
                e.c1.clone()
            }
        })
    }

    /// 0.
    #[must_use]
    pub fn zero(t: &'t Tower<'t>) -> Fp12<'t> {
        Fp12::from_fp6(Fp6::zero(t))
    }

    /// 1.
    #[must_use]
    pub fn one(t: &'t Tower<'t>) -> Fp12<'t> {
        Fp12::from_fp6(Fp6::one(t))
    }

    /// The tower.
    #[must_use]
    pub fn tower(&self) -> &'t Tower<'t> {
        self.c0.c0.t
    }

    /// The coefficient of 1.
    #[must_use]
    pub fn c0(&self) -> &Fp6<'t> {
        &self.c0
    }

    /// The coefficient of `w`.
    #[must_use]
    pub fn c1(&self) -> &Fp6<'t> {
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

    /// The square, by the complex method: `(c₀ + c₁)(c₀ + v c₁) − c₀c₁ − v c₀c₁` and `2c₀c₁`, two
    /// products of `F_p⁶`.
    #[must_use]
    pub fn square(&self) -> Fp12<'t> {
        let (a0, a1) = (&self.c0, &self.c1);
        let ab = a0 * a1;
        let c0 = (a0 + a1) * (a0 + &a1.mul_by_v()) - &ab - ab.mul_by_v();
        Fp12 {
            c0,
            c1: ab.double(),
        }
    }

    /// The conjugate `c₀ − c₁ w`, which is `x^(p⁶)`; in the cyclotomic subgroup, the inverse.
    #[must_use]
    pub fn conjugate(&self) -> Fp12<'t> {
        Fp12 {
            c0: self.c0.clone(),
            c1: -&self.c1,
        }
    }

    /// `x^(pᵏ)`, by the coefficients of the tower: each coefficient of `wⁱ` over `F_p²` mapped by
    /// the Frobenius map of `F_p²` (a conjugation for odd `k`) and multiplied by
    /// `ξ^(i(pᵏ − 1)/6)`: five products of `F_p²` by constants, which are products by elements of
    /// `F_p` where the constants lie there, as they do for even `k`; no power.
    #[must_use]
    pub fn frobenius(&self, k: usize) -> Fp12<'t> {
        let t = self.tower();
        let k = k % 12;
        if k == 0 {
            return self.clone();
        }
        let f = |x: &Fp2<'t>, i: usize| -> Fp2<'t> {
            let c = &t.frob[k][i];
            let x = x.frobenius(k);
            if c[1].is_zero() {
                x.mul_by_base(&c[0])
            } else {
                &x * &Fp2::new(t, c[0].clone(), c[1].clone())
            }
        };
        Fp12 {
            c0: Fp6 {
                c0: self.c0.c0.frobenius(k),
                c1: f(&self.c0.c1, 2),
                c2: f(&self.c0.c2, 4),
            },
            c1: Fp6 {
                c0: f(&self.c1.c0, 1),
                c1: f(&self.c1.c1, 3),
                c2: f(&self.c1.c2, 5),
            },
        }
    }

    /// The norm to `F_p⁶`, `x · x̄ = c₀² − v c₁²`.
    #[must_use]
    pub fn norm(&self) -> Fp6<'t> {
        self.c0.square() - self.c1.square().mul_by_v()
    }

    /// The trace to `F_p⁶`, `x + x̄ = 2 c₀`.
    #[must_use]
    pub fn trace(&self) -> Fp6<'t> {
        self.c0.double()
    }

    /// The inverse, the conjugate over the norm, or `None` for 0.
    #[must_use]
    pub fn inverse(&self) -> Option<Fp12<'t>> {
        let n = self.norm().inverse()?;
        Some(Fp12 {
            c0: &self.c0 * &n,
            c1: -(&self.c1 * &n),
        })
    }

    /// Whether it is a square of `F_p¹²` (0 is): when its norm to `F_p⁶` is a square there, since
    /// `N(x)^((p⁶ − 1)/2) = x^((p¹² − 1)/2)`.
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
    pub fn pow(&self, e: &Number) -> Fp12<'t> {
        if e.is_negative() {
            let inv = self.inverse().expect("0 has no inverse");
            return inv.pow(&-e);
        }
        let mut r = Fp12::one(self.tower());
        for i in (0..e.bits()).rev() {
            r = r.square();
            if e.bit(i) {
                r = &r * self;
            }
        }
        r
    }

    /// The product by `c₀ + c₃ w + c₄ v w`, the value of a line of the pairing on a D-type twist:
    /// thirteen products of `F_p²` instead of eighteen. The indices are those of the six
    /// coefficients over `F_p²` in the order `c₀.c₀, c₀.c₁, c₀.c₂, c₁.c₀, c₁.c₁, c₁.c₂`.
    ///
    /// # Panics
    ///
    /// Panics if the coefficients are of a different tower.
    #[track_caller]
    #[must_use]
    pub fn mul_by_034(&self, c0: &Fp2<'t>, c3: &Fp2<'t>, c4: &Fp2<'t>) -> Fp12<'t> {
        let (a, b) = (&self.c0, &self.c1);
        let t0 = a.mul_by_fp2(c0);
        let t1 = b.mul_by_01(c3, c4);
        let c1 = (a + b).mul_by_01(&(c0 + c3), c4) - &t0 - &t1;
        Fp12 {
            c0: t0 + t1.mul_by_v(),
            c1,
        }
    }

    /// The product by `c₀ + c₁ v + c₄ v w`, the value of a line of the pairing on an M-type twist:
    /// thirteen products of `F_p²` instead of eighteen, with the indices of
    /// [`mul_by_034`](Fp12::mul_by_034).
    ///
    /// # Panics
    ///
    /// Panics if the coefficients are of a different tower.
    #[track_caller]
    #[must_use]
    pub fn mul_by_014(&self, c0: &Fp2<'t>, c1: &Fp2<'t>, c4: &Fp2<'t>) -> Fp12<'t> {
        let (a, b) = (&self.c0, &self.c1);
        let t0 = a.mul_by_01(c0, c1);
        let t1 = b.mul_by_1(c4);
        let s = (a + b).mul_by_01(c0, &(c1 + c4)) - &t0 - &t1;
        Fp12 {
            c0: t0 + t1.mul_by_v(),
            c1: s,
        }
    }

    /// Whether it lies in the cyclotomic subgroup, of the elements of order dividing
    /// `Φ₁₂(p) = p⁴ − p² + 1`: `x^(p⁴) · x = x^(p²)`. Two Frobenius maps and a product.
    #[must_use]
    pub fn is_cyclotomic(&self) -> bool {
        !self.is_zero() && &self.frobenius(4) * self == self.frobenius(2)
    }

    /// The square of an element of the cyclotomic subgroup, by Granger and Scott (*Faster squaring
    /// in the cyclotomic subgroup of sixth degree extensions*, PKC 2010, §3.1): over
    /// `F_p⁴ = F_p²[s] / (s² − ξ)` with `s = w³`, the element is `A + B w + C w²` and its square
    /// `(3A² − 2Ā) + (3sC² + 2B̄) w + (3B² − 2C̄) w²`, nine squares of `F_p²`.
    ///
    /// For an element outside the subgroup the result is not its square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Fp6, Fp12, Tower};
    /// use a::math::numbers::Number;
    ///
    /// let t = Tower::bn254();
    /// let x = Fp12::new(Fp6::new(t.xi(), Fp2::one(t), Fp2::zero(t)), Fp6::one(t));
    /// // x^((p⁶ − 1)(p² + 1)) lies in the cyclotomic subgroup.
    /// let y = &x.conjugate() / &x;
    /// let y = &y.frobenius(2) * &y;
    /// assert!(y.is_cyclotomic());
    /// assert_eq!(y.cyclotomic_square(), y.square());
    /// assert_eq!(y.cyclotomic_pow(&Number::from(-5)), y.pow(&Number::from(-5)));
    /// ```
    #[must_use]
    pub fn cyclotomic_square(&self) -> Fp12<'t> {
        // A = g₀ + h₁ s, B = h₀ + g₂ s, C = g₁ + h₂ s, for c₀ = g₀ + g₁ v + g₂ v² and
        // c₁ = h₀ + h₁ v + h₂ v², since v = w², s = w³, w⁴ = s w and w⁵ = s w².
        let (g0, g1, g2) = (&self.c0.c0, &self.c0.c1, &self.c0.c2);
        let (h0, h1, h2) = (&self.c1.c0, &self.c1.c1, &self.c1.c2);
        // (x₀ + x₁ s)² = (x₀² + ξ x₁²) + 2x₀x₁ s, with 2x₀x₁ = (x₀ + x₁)² − x₀² − x₁².
        let sq4 = |x0: &Fp2<'t>, x1: &Fp2<'t>| {
            let a = x0.square();
            let b = x1.square();
            let c = (x0 + x1).square() - &a - &b;
            (a + b.mul_by_xi(), c)
        };
        let three = |x: &Fp2<'t>| &x.double() + x;
        let (a0, a1) = sq4(g0, h1);
        let (b0, b1) = sq4(h0, g2);
        let (c0, c1) = sq4(g1, h2);
        // A' = 3A² − 2Ā: g₀' = 3a₀ − 2g₀, h₁' = 3a₁ + 2h₁.
        // B' = 3sC² + 2B̄, with s(c₀ + c₁ s) = ξc₁ + c₀ s: h₀' = 3ξc₁ + 2h₀, g₂' = 3c₀ − 2g₂.
        // C' = 3B² − 2C̄: g₁' = 3b₀ − 2g₁, h₂' = 3b₁ + 2h₂.
        Fp12 {
            c0: Fp6 {
                c0: three(&a0) - g0.double(),
                c1: three(&b0) - g1.double(),
                c2: three(&c0) - g2.double(),
            },
            c1: Fp6 {
                c0: three(&c1.mul_by_xi()) + h0.double(),
                c1: three(&a1) + h1.double(),
                c2: three(&b1) + h2.double(),
            },
        }
    }

    /// The `e`-th power of an element of the cyclotomic subgroup, by squaring and multiplying with
    /// [`cyclotomic_square`](Fp12::cyclotomic_square), over the non-adjacent form of `e` (an
    /// inverse there is a conjugate) or its binary form when that has no more nonzero digits, as
    /// for BLS12-381's `|x|`; a negative `e` is a power of the conjugate.
    ///
    /// For an element outside the subgroup the result is not its power.
    #[must_use]
    pub fn cyclotomic_pow(&self, e: &Number) -> Fp12<'t> {
        let digits = signed_digits(&e.abs());
        let Some((_, rest)) = digits.split_last() else {
            return Fp12::one(self.tower());
        };
        // The top digit is 1: begin with the element, one square fewer.
        let inv = self.conjugate();
        let mut r = self.clone();
        for &d in rest.iter().rev() {
            r = r.cyclotomic_square();
            match d {
                1 => r = &r * self,
                -1 => r = &r * &inv,
                _ => {}
            }
        }
        if e.is_negative() { r.conjugate() } else { r }
    }

    /// The sum.
    #[track_caller]
    pub(super) fn add_ref(a: &Fp12<'t>, b: &Fp12<'t>) -> Fp12<'t> {
        Fp12 {
            c0: &a.c0 + &b.c0,
            c1: &a.c1 + &b.c1,
        }
    }

    /// The difference.
    #[track_caller]
    pub(super) fn sub_ref(a: &Fp12<'t>, b: &Fp12<'t>) -> Fp12<'t> {
        Fp12 {
            c0: &a.c0 - &b.c0,
            c1: &a.c1 - &b.c1,
        }
    }

    /// The product, by Karatsuba: `a₀b₀ + v a₁b₁` and `(a₀ + a₁)(b₀ + b₁) − a₀b₀ − a₁b₁`.
    #[track_caller]
    pub(super) fn mul_ref(a: &Fp12<'t>, b: &Fp12<'t>) -> Fp12<'t> {
        let v0 = &a.c0 * &b.c0;
        let v1 = &a.c1 * &b.c1;
        let c1 = (&a.c0 + &a.c1) * (&b.c0 + &b.c1) - &v0 - &v1;
        Fp12 {
            c0: v0 + v1.mul_by_v(),
            c1,
        }
    }

    /// The quotient.
    #[track_caller]
    pub(super) fn div_ref(a: &Fp12<'t>, b: &Fp12<'t>) -> Fp12<'t> {
        let inv = b.inverse().expect("division by zero");
        Fp12::mul_ref(a, &inv)
    }

    /// The negation.
    pub(super) fn neg_ref(&self) -> Fp12<'t> {
        Fp12 {
            c0: -&self.c0,
            c1: -&self.c1,
        }
    }
}

/// The digits in `{−1, 0, 1}` of `e ≥ 0`, the least significant first, for a power: its
/// non-adjacent form, or its binary form when the non-adjacent one has as many nonzero digits (it
/// is then a digit longer, a square more).
fn signed_digits(e: &Number) -> Vec<i8> {
    let n = naf(e);
    let weight = n.iter().filter(|&&d| d != 0).count() as u64;
    if weight < e.count_ones() {
        n
    } else {
        (0..e.bits()).map(|i| i8::from(e.bit(i))).collect()
    }
}

/// The non-adjacent form of `e ≥ 0`, its least significant digit first: digits in `{−1, 0, 1}`, no
/// two adjacent nonzero.
fn naf(e: &Number) -> Vec<i8> {
    let mut digits = Vec::with_capacity(e.bits() as usize + 1);
    let mut k = e.clone();
    let four = Number::from(4u8);
    while !k.is_zero() {
        if k.bit(0) {
            // 1 when k ≡ 1 (mod 4), −1 when k ≡ 3.
            let d: i8 = if k.rem_euclid(&four) == 1u8 { 1 } else { -1 };
            k = if d == 1 { k - 1u8 } else { k + 1u8 };
            digits.push(d);
        } else {
            digits.push(0);
        }
        k >>= 1usize;
    }
    digits
}

tower_ops!(Fp12);

impl PartialEq for Fp12<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl Eq for Fp12<'_> {}

impl Hash for Fp12<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.c0.hash(h);
        self.c1.hash(h);
    }
}

impl fmt::Debug for Fp12<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// `(c₀) + (c₁)·w` in decimal.
impl fmt::Display for Fp12<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) + ({})·w", self.c0, self.c1)
    }
}
