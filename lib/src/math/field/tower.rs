//! The towers of extensions `F_p² → F_p⁶ → F_p¹²` of pairings:
//!
//! * `F_p² = F_p[u] / (u² − β)` for a non-square `β` of `F_p`,
//! * `F_p⁶ = F_p²[v] / (v³ − ξ)` for a `ξ` of `F_p²` that is neither a square nor a cube,
//! * `F_p¹² = F_p⁶[w] / (w² − v)`, so that `w⁶ = ξ`.
//!
//! This is the representation of Devegili, Ó hÉigeartaigh, Scott and Dahab (*Multiplication and
//! squaring on pairing-friendly fields*, ePrint 2006/471) and of the IETF draft on pairing-friendly
//! curves, the one arkworks, the `bls12_381` crate and blst use: BN254 has `β = −1` and
//! `ξ = 9 + u`, BLS12-381 `β = −1` and `ξ = 1 + u`. A [`Tower`] holds the constants of its
//! arithmetic, computed once: `β`, `ξ` and the coefficients `ξ^(i(pᵏ − 1)/6)` of the Frobenius
//! maps; its elements [`Fp2`], [`Fp6`] and [`Fp12`] borrow it, as an [`Fp`] borrows its
//! [`PrimeField`].
//!
//! The tower borrows its prime field rather than owning a copy of it, so that its coefficients are
//! elements of that very field ([`Fp2::c0`] is an [`Fp`] that adds to the caller's elements without
//! a conversion) and its constants are stored in the field's own form once.
//!
//! Like the rest of the library this is not constant-time: it is for mathematics and for verifying
//! with public data, not for secret keys.

use super::prime::{Fp, PrimeField};
use super::prime_form::{INLINE, Value};
use super::tower_fp2::Fp2;
use crate::math::numbers::Number;
use core::fmt;
use std::sync::OnceLock;

/// The tower `F_p² → F_p⁶ → F_p¹²` over a prime field: `u² = β`, `v³ = ξ`, `w² = v`.
///
/// It is built once, by [`Tower::new`] or as [`Tower::bn254`] and [`Tower::bls12_381`], with the
/// constants of the Frobenius maps computed then; its elements borrow it.
///
/// # Examples
///
/// ```
/// use a::math::field::{Fp2, Fp6, Fp12, Tower};
/// use a::math::numbers::Number;
///
/// let t = Tower::bn254();
/// let a = Fp2::from_numbers(t, &Number::from(3u8), &Number::from(4u8));
/// assert_eq!(a.norm(), t.base().from_u64(25)); // 3² + 4², as u² = −1
/// let x = Fp12::new(Fp6::new(a.clone(), a.square(), Fp2::one(t)), Fp6::one(t));
/// assert_eq!(&x * &x.inverse().unwrap(), Fp12::one(t));
/// assert_eq!(x.frobenius(12), x);
/// ```
pub struct Tower<'f> {
    /// The prime field.
    pub(super) base: &'f PrimeField,
    /// `β`, with `u² = β`.
    pub(super) beta: Fp<'f>,
    /// Whether `β = −1`, so that a product by it is a negation.
    pub(super) beta_is_minus_one: bool,
    /// `ξ = ξ₀ + ξ₁ u`, with `v³ = ξ`.
    pub(super) xi: [Fp<'f>; 2],
    /// Whether `ξ₁ = 1`, which saves two products in each product by `ξ`.
    pub(super) xi1_is_one: bool,
    /// `ξ₀` when it is a number from 1 to 16, whose products are sums; else 0.
    pub(super) xi0_small: u64,
    /// The limbs of `β`, `ξ₀` and `ξ₁` as an inline form holds them (zeros for a longer field),
    /// for the arithmetic on limbs, which so reads them with no match on the representation.
    pub(super) consts_raw: [[u64; INLINE]; 3],
    /// `frob[k][i] = ξ^(i (pᵏ − 1) / 6)` for `k` in `0..12` and `i` in `0..6`, the factor of `wⁱ`
    /// in its `pᵏ`-th power: `(wⁱ)^(pᵏ) = frob[k][i] · wⁱ`.
    pub(super) frob: Vec<[[Fp<'f>; 2]; 6]>,
    /// `1/2`.
    pub(super) half: Fp<'f>,
}

impl<'f> Tower<'f> {
    /// The tower over `base` with `u² = β` and `v³ = ξ = ξ₀ + ξ₁ u` (each reduced modulo `p`).
    ///
    /// The checks of `ξ` and the Frobenius coefficients take two powers of `ξ`, by `(p² − 1)/3` and
    /// `(p − 1)/6`, and some seventy products of `F_p²`.
    ///
    /// # Panics
    ///
    /// Panics unless `p ≡ 1 (mod 6)` (which every BN and BLS12 prime is: the coefficients
    /// `ξ^(i(p − 1)/6)` need it), if `β` is a square modulo `p` (then `u² − β` is reducible), or if
    /// `ξ` is a square or a cube in `F_p²` (then `w⁶ − ξ` is reducible).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, PrimeField, Tower};
    /// use a::math::numbers::Number;
    ///
    /// // p = 103 ≡ 1 (mod 6) and ≡ 3 (mod 4), so −1 is not a square.
    /// let f = PrimeField::new(&Number::from(103u8));
    /// let t = Tower::new(&f, &Number::from(-1), (&Number::from(2u8), &Number::from(1u8)));
    /// let x = Fp2::from_numbers(&t, &Number::from(5u8), &Number::from(7u8));
    /// assert_eq!(x.frobenius(1), x.conjugate());
    /// assert_eq!(x.pow(&Number::from(103u32 * 103)), x);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(base: &'f PrimeField, beta: &Number, xi: (&Number, &Number)) -> Tower<'f> {
        let p = base.modulus();
        assert!(
            p.rem_euclid(&Number::from(6u8)) == 1u8,
            "the tower needs p ≡ 1 (mod 6)"
        );
        let beta = base.element(beta);
        assert!(beta.legendre() == -1, "β must not be a square modulo p");
        let xi = [base.element(xi.0), base.element(xi.1)];
        let minus_one = -base.one();
        let consts_raw = [&beta, &xi[0], &xi[1]].map(|c| match c.raw() {
            Value::Limbs(l) => *l,
            _ => [0; INLINE],
        });
        let mut t = Tower {
            base,
            beta_is_minus_one: beta == minus_one,
            beta,
            xi1_is_one: xi[1].is_one(),
            xi0_small: {
                let v = xi[0].value();
                let small = if v.bits() <= 5 { v.to_f64() as u64 } else { 0 };
                if (1..=16).contains(&small) { small } else { 0 }
            },
            consts_raw,
            xi,
            frob: Vec::new(),
            half: base.from_u64(2).inverse().expect("p is odd"),
        };
        t.frob = frobenius_table(&t);
        t
    }

    /// The tower of BN254 (alt_bn128): `p` of [`PrimeField::bn254_base`], `u² = −1`, `ξ = 9 + u`.
    #[must_use]
    pub fn bn254() -> &'static Tower<'static> {
        static T: OnceLock<Tower<'static>> = OnceLock::new();
        T.get_or_init(|| {
            Tower::new(
                PrimeField::bn254_base(),
                &Number::from(-1),
                (&Number::from(9u8), &Number::from(1u8)),
            )
        })
    }

    /// The tower of BLS12-381: `p` of [`PrimeField::bls12_381_base`], `u² = −1`, `ξ = 1 + u`.
    #[must_use]
    pub fn bls12_381() -> &'static Tower<'static> {
        static T: OnceLock<Tower<'static>> = OnceLock::new();
        T.get_or_init(|| {
            Tower::new(
                PrimeField::bls12_381_base(),
                &Number::from(-1),
                (&Number::from(1u8), &Number::from(1u8)),
            )
        })
    }

    /// The prime field `F_p`.
    #[must_use]
    pub fn base(&self) -> &'f PrimeField {
        self.base
    }

    /// `β`, the square of `u`.
    #[must_use]
    pub fn beta(&self) -> Fp<'f> {
        self.beta.clone()
    }

    /// `ξ`, the cube of `v` and the sixth power of `w`.
    #[must_use]
    pub fn xi(&self) -> Fp2<'_> {
        Fp2::new(self, self.xi[0].clone(), self.xi[1].clone())
    }

    /// The coefficient `ξ^(i(pᵏ − 1)/6)` of `wⁱ` in its `pᵏ`-th power, for `k` modulo 12 and `i`
    /// below 6: the constant that [`Fp12::frobenius`](super::Fp12::frobenius) multiplies by.
    ///
    /// # Panics
    ///
    /// Panics if `i ≥ 6`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Tower};
    ///
    /// let t = Tower::bls12_381();
    /// // w^(p⁶) = −w: the sixth power of the Frobenius map is the conjugation of F_p¹² / F_p⁶.
    /// assert_eq!(t.frobenius_coefficient(6, 1), -Fp2::one(t));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn frobenius_coefficient(&self, k: usize, i: usize) -> Fp2<'_> {
        assert!(i < 6, "i must be below 6");
        let [c0, c1] = &self.frob[k % 12][i];
        Fp2::new(self, c0.clone(), c1.clone())
    }

    /// Whether two towers are the same: the same prime, `β` and `ξ`.
    pub(super) fn same(&self, other: &Tower<'_>) -> bool {
        core::ptr::eq(
            self as *const Tower<'_> as *const u8,
            other as *const Tower<'_> as *const u8,
        ) || (self.base.modulus() == other.base.modulus()
            && self.beta == other.beta
            && self.xi == other.xi)
    }

    /// Panics unless `other` is this tower.
    #[track_caller]
    pub(super) fn check(&self, other: &Tower<'_>) {
        if !self.same(other) {
            different_towers();
        }
    }

    /// `β x`.
    pub(super) fn mul_by_beta(&self, x: &Fp<'f>) -> Fp<'f> {
        if self.beta_is_minus_one {
            -x
        } else {
            x * &self.beta
        }
    }
}

impl fmt::Debug for Tower<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tower {{ p: {}, β: {}, ξ: {} + {}·u }}",
            self.base.modulus(),
            self.beta,
            self.xi[0],
            self.xi[1]
        )
    }
}

/// Panics for elements of different towers.
#[cold]
#[inline(never)]
#[track_caller]
pub(super) fn different_towers() -> ! {
    panic!("elements of different towers")
}

/// Checks that `ξ` is neither a square nor a cube in `F_p²` and computes `ξ^(i(pᵏ − 1)/6)`, with
/// `(wⁱ)^(pᵏ) = ((wⁱ)^(pᵏ⁻¹))^p = (γₖ₋₁ wⁱ)^p = conj(γₖ₋₁) γ₁ wⁱ`: one power and products after.
#[track_caller]
fn frobenius_table<'f>(t: &Tower<'f>) -> Vec<[[Fp<'f>; 2]; 6]> {
    let p = t.base.modulus();
    let one = Number::from(1u8);
    let xi = t.xi();
    // ξ is a square of F_p² exactly when its norm is a square of F_p.
    assert!(xi.norm().legendre() == -1, "ξ must not be a square in F_p²");
    let q1 = &(p * p) - &one;
    assert!(
        !xi.pow(&(&q1 / &Number::from(3u8))).is_one(),
        "ξ must not be a cube in F_p²"
    );
    let g = xi.pow(&((p - &one) / &Number::from(6u8)));
    let mut first = vec![Fp2::one(t)];
    for i in 1..6 {
        let next = &first[i - 1] * &g;
        first.push(next);
    }
    let mut rows: Vec<Vec<Fp2<'_>>> = vec![vec![Fp2::one(t); 6], first.clone()];
    for k in 2..12 {
        let row = (0..6)
            .map(|i| &rows[k - 1][i].conjugate() * &first[i])
            .collect();
        rows.push(row);
    }
    // Copied out of the borrow of `t` into elements of the field itself.
    rows.iter()
        .map(|row| {
            core::array::from_fn(|i| {
                [
                    t.base.element(&row[i].c0().value()),
                    t.base.element(&row[i].c1().value()),
                ]
            })
        })
        .collect()
}

/// The binary operators of a tower element for the four combinations of owned and borrowed
/// operands, and their assignments, from a function on two borrows.
macro_rules! tower_binary {
    ($ty:ident, $tr:ident, $m:ident, $atr:ident, $am:ident, $f:path) => {
        $crate::math::field::tower::tower_binary!(@ops $ty, $tr, $m, $f);
        impl<'t> core::ops::$atr<&$ty<'t>> for $ty<'t> {
            #[track_caller]
            fn $am(&mut self, rhs: &$ty<'t>) {
                *self = $f(self, rhs);
            }
        }
        impl<'t> core::ops::$atr<$ty<'t>> for $ty<'t> {
            #[track_caller]
            fn $am(&mut self, rhs: $ty<'t>) {
                *self = $f(self, &rhs);
            }
        }
    };
    // With `$fa`, the assignments in place: only the coefficients' limbs are rewritten.
    ($ty:ident, $tr:ident, $m:ident, $atr:ident, $am:ident, $f:path, $fa:path) => {
        $crate::math::field::tower::tower_binary!(@ops $ty, $tr, $m, $f);
        impl<'t> core::ops::$atr<&$ty<'t>> for $ty<'t> {
            #[track_caller]
            fn $am(&mut self, rhs: &$ty<'t>) {
                $fa(self, rhs);
            }
        }
        impl<'t> core::ops::$atr<$ty<'t>> for $ty<'t> {
            #[track_caller]
            fn $am(&mut self, rhs: $ty<'t>) {
                $fa(self, &rhs);
            }
        }
    };
    (@ops $ty:ident, $tr:ident, $m:ident, $f:path) => {
        impl<'t> core::ops::$tr<&$ty<'t>> for &$ty<'t> {
            type Output = $ty<'t>;
            #[track_caller]
            fn $m(self, rhs: &$ty<'t>) -> $ty<'t> {
                $f(self, rhs)
            }
        }
        impl<'t> core::ops::$tr<$ty<'t>> for &$ty<'t> {
            type Output = $ty<'t>;
            #[track_caller]
            fn $m(self, rhs: $ty<'t>) -> $ty<'t> {
                $f(self, &rhs)
            }
        }
        impl<'t> core::ops::$tr<&$ty<'t>> for $ty<'t> {
            type Output = $ty<'t>;
            #[track_caller]
            fn $m(self, rhs: &$ty<'t>) -> $ty<'t> {
                $f(&self, rhs)
            }
        }
        impl<'t> core::ops::$tr<$ty<'t>> for $ty<'t> {
            type Output = $ty<'t>;
            #[track_caller]
            fn $m(self, rhs: $ty<'t>) -> $ty<'t> {
                $f(&self, &rhs)
            }
        }
    };
}

/// `+ - * /` and `-x` for a tower element, from its functions `add`, `sub`, `mul`, `div` and `neg`
/// on borrows.
macro_rules! tower_ops {
    ($ty:ident) => {
        $crate::math::field::tower::tower_binary!(
            $ty,
            Add,
            add,
            AddAssign,
            add_assign,
            $ty::add_ref,
            $ty::add_assign_ref
        );
        $crate::math::field::tower::tower_binary!(
            $ty,
            Sub,
            sub,
            SubAssign,
            sub_assign,
            $ty::sub_ref,
            $ty::sub_assign_ref
        );
        $crate::math::field::tower::tower_binary!(
            $ty,
            Mul,
            mul,
            MulAssign,
            mul_assign,
            $ty::mul_ref,
            $ty::mul_assign_ref
        );
        $crate::math::field::tower::tower_binary!(
            $ty,
            Div,
            div,
            DivAssign,
            div_assign,
            $ty::div_ref
        );
        impl<'t> core::ops::Neg for &$ty<'t> {
            type Output = $ty<'t>;
            fn neg(self) -> $ty<'t> {
                self.neg_ref()
            }
        }
        impl<'t> core::ops::Neg for $ty<'t> {
            type Output = $ty<'t>;
            fn neg(self) -> $ty<'t> {
                self.neg_ref()
            }
        }
    };
}

pub(super) use tower_binary;
pub(super) use tower_ops;
