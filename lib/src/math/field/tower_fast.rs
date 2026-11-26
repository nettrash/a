//! The arithmetic of the towers in the base field's own form, for the Montgomery forms of four and
//! six limbs, which BN254's and BLS12-381's base fields take: an operation of the tower matches the
//! form once, reads its coefficients' limbs, computes on arrays of limbs with the field's kernels
//! (in registers and on the stack, no element of `F_p` built and no form matched between them),
//! and writes the coefficients of the result once. Other towers take the formulas on elements
//! (`tower_fp2`, `tower_fp6`, `tower_fp12`), which are the same.
//!
//! The constants come from the [`Tower`]: `β = −1` makes the products by `β` negations, and a `ξ`
//! of the form `ξ₀ + u` with a small `ξ₀` (BLS12-381's `1 + u`, BN254's `9 + u`) makes the products
//! by `ξ` sums: `(a₀ + a₁u)(ξ₀ + u) = (ξ₀a₀ + βa₁) + (a₀ + ξ₀a₁)u`, `ξ₀a` by doublings and sums.

use super::prime::Fp;
use super::prime_form::Arith;
use super::tower::Tower;
use super::tower_fp2::Fp2;
use super::tower_fp6::Fp6;
use super::tower_fp12::Fp12;
use crate::math::ec::field::{Field, Mont};

/// An element of `F_p²`: `c₀ + c₁u`.
pub(super) type E2<const N: usize> = [[u64; N]; 2];
/// An element of `F_p⁶`: `c₀ + c₁v + c₂v²`.
pub(super) type E6<const N: usize> = [E2<N>; 3];
/// An element of `F_p¹²`: `c₀ + c₁w`.
pub(super) type E12<const N: usize> = [E6<N>; 2];

/// The arithmetic of a tower over the Montgomery form of `N` limbs, with its constants as limbs.
pub(super) struct K<'t, const N: usize> {
    f: &'t Mont<N>,
    t: &'t Tower<'t>,
    /// `β = −1`.
    beta_m1: bool,
    beta: [u64; N],
    xi: E2<N>,
    /// `ξ₁ = 1`, and `ξ₀` when it is a small number (at most 16), else 0.
    xi1_one: bool,
    xi0_small: u64,
}

/// Runs `$body` with `$k` bound to the fast arithmetic of the tower `$t` when its base field has
/// one of the forms it is written for, and does nothing otherwise.
macro_rules! fast {
    ($t:expr, $k:ident => $body:expr) => {{
        let t = $t;
        match t.base.form_ref() {
            $crate::math::field::prime_form::Form::Odd($crate::math::ec::field::Prime::Mont4(
                f,
            )) => {
                let $k = $crate::math::field::tower_fast::K::<4>::new(f, t);
                #[allow(unreachable_code)]
                return $body;
            }
            $crate::math::field::prime_form::Form::Odd($crate::math::ec::field::Prime::Mont6(
                f,
            )) => {
                let $k = $crate::math::field::tower_fast::K::<6>::new(f, t);
                #[allow(unreachable_code)]
                return $body;
            }
            _ => {}
        }
    }};
}
pub(super) use fast;

impl<'t, const N: usize> K<'t, N>
where
    Mont<N>: Arith<E = [u64; N]>,
{
    #[inline(always)]
    pub(super) fn new(f: &'t Mont<N>, t: &'t Tower<'t>) -> Self {
        K {
            f,
            t,
            beta_m1: t.beta_is_minus_one,
            beta: core::array::from_fn(|i| t.consts_raw[0][i]),
            xi: [
                core::array::from_fn(|i| t.consts_raw[1][i]),
                core::array::from_fn(|i| t.consts_raw[2][i]),
            ],
            xi1_one: t.xi1_is_one,
            xi0_small: t.xi0_small,
        }
    }

    // ------------------------------------------------------------ loads and stores

    #[inline(always)]
    pub(super) fn ld(&self, x: &Fp<'_>) -> [u64; N] {
        *self.f.get(x.raw())
    }

    #[inline(always)]
    pub(super) fn ld2(&self, x: &Fp2<'_>) -> E2<N> {
        [self.ld(&x.c0), self.ld(&x.c1)]
    }

    #[inline(always)]
    pub(super) fn ld6(&self, x: &Fp6<'_>) -> E6<N> {
        [self.ld2(&x.c0), self.ld2(&x.c1), self.ld2(&x.c2)]
    }

    #[inline(always)]
    pub(super) fn ld12(&self, x: &Fp12<'_>) -> E12<N> {
        [self.ld6(&x.c0), self.ld6(&x.c1)]
    }

    #[inline(always)]
    pub(super) fn st(&self, x: [u64; N]) -> Fp<'t> {
        Fp::from_raw(self.t.base, self.f.put(x))
    }

    #[inline(always)]
    pub(super) fn st2(&self, x: E2<N>) -> Fp2<'t> {
        Fp2 {
            t: self.t,
            c0: self.st(x[0]),
            c1: self.st(x[1]),
        }
    }

    #[inline(always)]
    pub(super) fn st6(&self, x: E6<N>) -> Fp6<'t> {
        Fp6 {
            c0: self.st2(x[0]),
            c1: self.st2(x[1]),
            c2: self.st2(x[2]),
        }
    }

    #[inline(always)]
    pub(super) fn st12(&self, x: E12<N>) -> Fp12<'t> {
        Fp12 {
            c0: self.st6(x[0]),
            c1: self.st6(x[1]),
        }
    }

    /// Writes `x` into the element `dst` of the base field, in place.
    #[inline(always)]
    pub(super) fn put_into(&self, x: [u64; N], dst: &mut Fp<'_>) {
        self.f.update(dst.raw_mut(), |_| x);
    }

    #[inline(always)]
    pub(super) fn put2(&self, x: E2<N>, dst: &mut Fp2<'_>) {
        self.put_into(x[0], &mut dst.c0);
        self.put_into(x[1], &mut dst.c1);
    }

    #[inline(always)]
    pub(super) fn put6(&self, x: E6<N>, dst: &mut Fp6<'_>) {
        self.put2(x[0], &mut dst.c0);
        self.put2(x[1], &mut dst.c1);
        self.put2(x[2], &mut dst.c2);
    }

    #[inline(always)]
    pub(super) fn put12(&self, x: E12<N>, dst: &mut Fp12<'_>) {
        self.put6(x[0], &mut dst.c0);
        self.put6(x[1], &mut dst.c1);
    }

    // ------------------------------------------------------------ F_p

    #[inline(always)]
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.f.add(a, b)
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.f.sub(a, b)
    }

    #[inline(always)]
    fn neg(&self, a: &[u64; N]) -> [u64; N] {
        self.f.neg(a)
    }

    #[inline(always)]
    fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.f.mul(a, b)
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; N]) -> [u64; N] {
        self.f.sqr(a)
    }

    #[inline(always)]
    fn dbl(&self, a: &[u64; N]) -> [u64; N] {
        self.f.add(a, a)
    }

    /// `k a` for a small `k`, by doublings and sums.
    #[inline(always)]
    fn small(&self, a: &[u64; N], k: u64) -> [u64; N] {
        let mut r = *a;
        for i in (0..63 - k.leading_zeros()).rev() {
            r = self.dbl(&r);
            if (k >> i) & 1 == 1 {
                r = self.add(&r, a);
            }
        }
        r
    }

    #[inline(always)]
    fn mul_beta(&self, a: &[u64; N]) -> [u64; N] {
        if self.beta_m1 {
            self.neg(a)
        } else {
            self.mul(a, &self.beta)
        }
    }

    /// `a − β b`.
    #[inline(always)]
    fn sub_beta(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        if self.beta_m1 {
            self.add(a, b)
        } else {
            self.sub(a, &self.mul(b, &self.beta))
        }
    }

    /// `a + β b`.
    #[inline(always)]
    fn add_beta(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        if self.beta_m1 {
            self.sub(a, b)
        } else {
            self.add(a, &self.mul(b, &self.beta))
        }
    }

    /// `ξ₀ a`.
    #[inline(always)]
    fn mul_xi0(&self, a: &[u64; N]) -> [u64; N] {
        if self.xi0_small != 0 {
            self.small(a, self.xi0_small)
        } else {
            self.mul(a, &self.xi[0])
        }
    }

    pub(super) fn inv(&self, a: &[u64; N]) -> Option<[u64; N]> {
        self.f.invert_value(a)
    }

    // ------------------------------------------------------------ F_p²

    #[inline(always)]
    pub(super) fn add2(&self, a: &E2<N>, b: &E2<N>) -> E2<N> {
        [self.add(&a[0], &b[0]), self.add(&a[1], &b[1])]
    }

    #[inline(always)]
    pub(super) fn sub2(&self, a: &E2<N>, b: &E2<N>) -> E2<N> {
        [self.sub(&a[0], &b[0]), self.sub(&a[1], &b[1])]
    }

    #[inline(always)]
    pub(super) fn neg2(&self, a: &E2<N>) -> E2<N> {
        [self.neg(&a[0]), self.neg(&a[1])]
    }

    #[inline(always)]
    pub(super) fn dbl2(&self, a: &E2<N>) -> E2<N> {
        [self.dbl(&a[0]), self.dbl(&a[1])]
    }

    /// Karatsuba: three products of `F_p`, by the lazy reduction of [`Mont::mul_complex`] when
    /// `β = −1`.
    #[inline(always)]
    pub(super) fn mul2(&self, a: &E2<N>, b: &E2<N>) -> E2<N> {
        if self.beta_m1 {
            return self.f.mul_complex(a, b);
        }
        let v0 = self.mul(&a[0], &b[0]);
        let v1 = self.mul(&a[1], &b[1]);
        let s = self.mul(&self.add(&a[0], &a[1]), &self.add(&b[0], &b[1]));
        [self.add_beta(&v0, &v1), self.sub(&self.sub(&s, &v0), &v1)]
    }

    /// The complex squaring: two products of `F_p` when `β = −1`, by [`Mont::sqr_complex`].
    #[inline(always)]
    pub(super) fn sqr2(&self, a: &E2<N>) -> E2<N> {
        if self.beta_m1 {
            return self.f.sqr_complex(a);
        }
        let ab = self.mul(&a[0], &a[1]);
        let c0 = if self.beta_m1 {
            self.mul(&self.add(&a[0], &a[1]), &self.sub(&a[0], &a[1]))
        } else {
            let t = self.mul(&self.add(&a[0], &a[1]), &self.add_beta(&a[0], &a[1]));
            self.sub(&self.sub(&t, &ab), &self.mul_beta(&ab))
        };
        [c0, self.dbl(&ab)]
    }

    #[inline(always)]
    pub(super) fn conj2(&self, a: &E2<N>) -> E2<N> {
        [a[0], self.neg(&a[1])]
    }

    /// `a ξ`.
    #[inline(always)]
    pub(super) fn mul_xi(&self, a: &E2<N>) -> E2<N> {
        if self.xi1_one {
            [
                self.add_beta(&self.mul_xi0(&a[0]), &a[1]),
                self.add(&a[0], &self.mul_xi0(&a[1])),
            ]
        } else {
            let x1 = &self.xi[1];
            [
                self.add_beta(&self.mul_xi0(&a[0]), &self.mul(&a[1], x1)),
                self.add(&self.mul(&a[0], x1), &self.mul_xi0(&a[1])),
            ]
        }
    }

    /// `c₀² − βc₁²`.
    #[inline(always)]
    pub(super) fn norm2(&self, a: &E2<N>) -> [u64; N] {
        self.sub_beta(&self.sqr(&a[0]), &self.sqr(&a[1]))
    }

    pub(super) fn inv2(&self, a: &E2<N>) -> Option<E2<N>> {
        let n = self.inv(&self.norm2(a))?;
        Some([self.mul(&a[0], &n), self.neg(&self.mul(&a[1], &n))])
    }

    #[inline(always)]
    pub(super) fn mul2_base(&self, a: &E2<N>, k: &[u64; N]) -> E2<N> {
        [self.mul(&a[0], k), self.mul(&a[1], k)]
    }

    /// `a^(pᵏ)` on `F_p²`: the conjugate for odd `k`.
    #[inline(always)]
    fn frob2(&self, a: &E2<N>, k: usize) -> E2<N> {
        if k % 2 == 1 { self.conj2(a) } else { *a }
    }

    /// The product by the constant `ξ^(i(pᵏ − 1)/6)` of the Frobenius maps, by `F_p` alone when it
    /// lies there.
    #[inline(always)]
    fn mul_frob(&self, a: &E2<N>, k: usize, i: usize) -> E2<N> {
        let c = &self.t.frob[k][i];
        let c0 = self.ld(&c[0]);
        if c[1].is_zero() {
            self.mul2_base(a, &c0)
        } else {
            self.mul2(a, &[c0, self.ld(&c[1])])
        }
    }

    // ------------------------------------------------------------ F_p⁶

    #[inline(always)]
    pub(super) fn add6(&self, a: &E6<N>, b: &E6<N>) -> E6<N> {
        [
            self.add2(&a[0], &b[0]),
            self.add2(&a[1], &b[1]),
            self.add2(&a[2], &b[2]),
        ]
    }

    #[inline(always)]
    pub(super) fn sub6(&self, a: &E6<N>, b: &E6<N>) -> E6<N> {
        [
            self.sub2(&a[0], &b[0]),
            self.sub2(&a[1], &b[1]),
            self.sub2(&a[2], &b[2]),
        ]
    }

    #[inline(always)]
    pub(super) fn neg6(&self, a: &E6<N>) -> E6<N> {
        [self.neg2(&a[0]), self.neg2(&a[1]), self.neg2(&a[2])]
    }

    #[inline(always)]
    pub(super) fn dbl6(&self, a: &E6<N>) -> E6<N> {
        [self.dbl2(&a[0]), self.dbl2(&a[1]), self.dbl2(&a[2])]
    }

    /// Karatsuba: six products of `F_p²`.
    pub(super) fn mul6(&self, a: &E6<N>, b: &E6<N>) -> E6<N> {
        let v0 = self.mul2(&a[0], &b[0]);
        let v1 = self.mul2(&a[1], &b[1]);
        let v2 = self.mul2(&a[2], &b[2]);
        let t0 = self.mul2(&self.add2(&a[1], &a[2]), &self.add2(&b[1], &b[2]));
        let c0 = self.add2(&self.mul_xi(&self.sub2(&self.sub2(&t0, &v1), &v2)), &v0);
        let t1 = self.mul2(&self.add2(&a[0], &a[1]), &self.add2(&b[0], &b[1]));
        let c1 = self.add2(&self.sub2(&self.sub2(&t1, &v0), &v1), &self.mul_xi(&v2));
        let t2 = self.mul2(&self.add2(&a[0], &a[2]), &self.add2(&b[0], &b[2]));
        let c2 = self.add2(&self.sub2(&self.sub2(&t2, &v0), &v2), &v1);
        [c0, c1, c2]
    }

    /// Chung and Hasan's SQR2: two products and three squares of `F_p²`.
    pub(super) fn sqr6(&self, a: &E6<N>) -> E6<N> {
        let s0 = self.sqr2(&a[0]);
        let s1 = self.dbl2(&self.mul2(&a[0], &a[1]));
        let s2 = self.sqr2(&self.add2(&self.sub2(&a[0], &a[1]), &a[2]));
        let s3 = self.dbl2(&self.mul2(&a[1], &a[2]));
        let s4 = self.sqr2(&a[2]);
        [
            self.add2(&s0, &self.mul_xi(&s3)),
            self.add2(&s1, &self.mul_xi(&s4)),
            self.sub2(&self.sub2(&self.add2(&self.add2(&s1, &s2), &s3), &s0), &s4),
        ]
    }

    /// `a v`: `(ξc₂, c₀, c₁)`.
    #[inline(always)]
    pub(super) fn mul_v(&self, a: &E6<N>) -> E6<N> {
        [self.mul_xi(&a[2]), a[0], a[1]]
    }

    #[inline(always)]
    pub(super) fn mul6_fp2(&self, a: &E6<N>, k: &E2<N>) -> E6<N> {
        [
            self.mul2(&a[0], k),
            self.mul2(&a[1], k),
            self.mul2(&a[2], k),
        ]
    }

    /// The product by `b₀ + b₁v`: five products of `F_p²`.
    pub(super) fn mul6_01(&self, a: &E6<N>, b0: &E2<N>, b1: &E2<N>) -> E6<N> {
        let v0 = self.mul2(&a[0], b0);
        let v1 = self.mul2(&a[1], b1);
        let c0 = self.add2(
            &self.mul_xi(&self.sub2(&self.mul2(&self.add2(&a[1], &a[2]), b1), &v1)),
            &v0,
        );
        let c1 = self.sub2(
            &self.sub2(
                &self.mul2(&self.add2(&a[0], &a[1]), &self.add2(b0, b1)),
                &v0,
            ),
            &v1,
        );
        let c2 = self.add2(
            &self.sub2(&self.mul2(&self.add2(&a[0], &a[2]), b0), &v0),
            &v1,
        );
        [c0, c1, c2]
    }

    /// The product by `b₁v`: three products of `F_p²`.
    pub(super) fn mul6_1(&self, a: &E6<N>, b1: &E2<N>) -> E6<N> {
        [
            self.mul_xi(&self.mul2(&a[2], b1)),
            self.mul2(&a[0], b1),
            self.mul2(&a[1], b1),
        ]
    }

    pub(super) fn frob6(&self, a: &E6<N>, k: usize) -> E6<N> {
        let k = k % 6;
        [
            self.frob2(&a[0], k),
            self.mul_frob(&self.frob2(&a[1], k), k, 2),
            self.mul_frob(&self.frob2(&a[2], k), k, 4),
        ]
    }

    /// The adjugate `(c₀² − ξc₁c₂, ξc₂² − c₀c₁, c₁² − c₀c₂)`.
    fn adjugate6(&self, a: &E6<N>) -> [E2<N>; 3] {
        [
            self.sub2(&self.sqr2(&a[0]), &self.mul_xi(&self.mul2(&a[1], &a[2]))),
            self.sub2(&self.mul_xi(&self.sqr2(&a[2])), &self.mul2(&a[0], &a[1])),
            self.sub2(&self.sqr2(&a[1]), &self.mul2(&a[0], &a[2])),
        ]
    }

    /// `c₀A + ξ(c₂B + c₁C)` for the adjugate `(A, B, C)`.
    fn norm6_from(&self, a: &E6<N>, adj: &[E2<N>; 3]) -> E2<N> {
        let s = self.add2(&self.mul2(&a[2], &adj[1]), &self.mul2(&a[1], &adj[2]));
        self.add2(&self.mul2(&a[0], &adj[0]), &self.mul_xi(&s))
    }

    pub(super) fn norm6(&self, a: &E6<N>) -> E2<N> {
        self.norm6_from(a, &self.adjugate6(a))
    }

    pub(super) fn inv6(&self, a: &E6<N>) -> Option<E6<N>> {
        let adj = self.adjugate6(a);
        let n = self.inv2(&self.norm6_from(a, &adj))?;
        Some([
            self.mul2(&adj[0], &n),
            self.mul2(&adj[1], &n),
            self.mul2(&adj[2], &n),
        ])
    }

    // ------------------------------------------------------------ F_p¹²

    #[inline(always)]
    pub(super) fn add12(&self, a: &E12<N>, b: &E12<N>) -> E12<N> {
        [self.add6(&a[0], &b[0]), self.add6(&a[1], &b[1])]
    }

    #[inline(always)]
    pub(super) fn sub12(&self, a: &E12<N>, b: &E12<N>) -> E12<N> {
        [self.sub6(&a[0], &b[0]), self.sub6(&a[1], &b[1])]
    }

    #[inline(always)]
    pub(super) fn neg12(&self, a: &E12<N>) -> E12<N> {
        [self.neg6(&a[0]), self.neg6(&a[1])]
    }

    /// Karatsuba: three products of `F_p⁶`.
    pub(super) fn mul12(&self, a: &E12<N>, b: &E12<N>) -> E12<N> {
        let v0 = self.mul6(&a[0], &b[0]);
        let v1 = self.mul6(&a[1], &b[1]);
        let s = self.mul6(&self.add6(&a[0], &a[1]), &self.add6(&b[0], &b[1]));
        [
            self.add6(&v0, &self.mul_v(&v1)),
            self.sub6(&self.sub6(&s, &v0), &v1),
        ]
    }

    /// The complex squaring: two products of `F_p⁶`.
    pub(super) fn sqr12(&self, a: &E12<N>) -> E12<N> {
        let ab = self.mul6(&a[0], &a[1]);
        let t = self.mul6(
            &self.add6(&a[0], &a[1]),
            &self.add6(&a[0], &self.mul_v(&a[1])),
        );
        [
            self.sub6(&self.sub6(&t, &ab), &self.mul_v(&ab)),
            self.dbl6(&ab),
        ]
    }

    #[inline(always)]
    pub(super) fn conj12(&self, a: &E12<N>) -> E12<N> {
        [a[0], self.neg6(&a[1])]
    }

    pub(super) fn frob12(&self, a: &E12<N>, k: usize) -> E12<N> {
        let k = k % 12;
        if k == 0 {
            return *a;
        }
        [
            [
                self.frob2(&a[0][0], k),
                self.mul_frob(&self.frob2(&a[0][1], k), k, 2),
                self.mul_frob(&self.frob2(&a[0][2], k), k, 4),
            ],
            [
                self.mul_frob(&self.frob2(&a[1][0], k), k, 1),
                self.mul_frob(&self.frob2(&a[1][1], k), k, 3),
                self.mul_frob(&self.frob2(&a[1][2], k), k, 5),
            ],
        ]
    }

    /// `c₀² − v c₁²`.
    pub(super) fn norm12(&self, a: &E12<N>) -> E6<N> {
        self.sub6(&self.sqr6(&a[0]), &self.mul_v(&self.sqr6(&a[1])))
    }

    pub(super) fn inv12(&self, a: &E12<N>) -> Option<E12<N>> {
        let n = self.inv6(&self.norm12(a))?;
        Some([self.mul6(&a[0], &n), self.neg6(&self.mul6(&a[1], &n))])
    }

    /// The product by a line value `c₀ + c₃w + c₄vw` (a D-type twist).
    pub(super) fn mul_034(&self, a: &E12<N>, c0: &E2<N>, c3: &E2<N>, c4: &E2<N>) -> E12<N> {
        let t0 = self.mul6_fp2(&a[0], c0);
        let t1 = self.mul6_01(&a[1], c3, c4);
        let s = self.mul6_01(&self.add6(&a[0], &a[1]), &self.add2(c0, c3), c4);
        [
            self.add6(&t0, &self.mul_v(&t1)),
            self.sub6(&self.sub6(&s, &t0), &t1),
        ]
    }

    /// The product by a line value `c₀ + c₁v + c₄vw` (an M-type twist).
    pub(super) fn mul_014(&self, a: &E12<N>, c0: &E2<N>, c1: &E2<N>, c4: &E2<N>) -> E12<N> {
        let t0 = self.mul6_01(&a[0], c0, c1);
        let t1 = self.mul6_1(&a[1], c4);
        let s = self.mul6_01(&self.add6(&a[0], &a[1]), c0, &self.add2(c1, c4));
        [
            self.add6(&t0, &self.mul_v(&t1)),
            self.sub6(&self.sub6(&s, &t0), &t1),
        ]
    }

    /// Granger and Scott's square in the cyclotomic subgroup, over `F_p⁴ = F_p²[s]/(s² − ξ)`.
    pub(super) fn cyc_sqr(&self, a: &E12<N>) -> E12<N> {
        let [[g0, g1, g2], [h0, h1, h2]] = a;
        // (x₀ + x₁s)² = (x₀² + ξx₁²) + 2x₀x₁s, with 2x₀x₁ = (x₀ + x₁)² − x₀² − x₁².
        let sq4 = |x0: &E2<N>, x1: &E2<N>| {
            let a = self.sqr2(x0);
            let b = self.sqr2(x1);
            let c = self.sub2(&self.sub2(&self.sqr2(&self.add2(x0, x1)), &a), &b);
            (self.add2(&a, &self.mul_xi(&b)), c)
        };
        let three = |x: &E2<N>| self.add2(&self.dbl2(x), x);
        let (a0, a1) = sq4(g0, h1);
        let (b0, b1) = sq4(h0, g2);
        let (c0, c1) = sq4(g1, h2);
        [
            [
                self.sub2(&three(&a0), &self.dbl2(g0)),
                self.sub2(&three(&b0), &self.dbl2(g1)),
                self.sub2(&three(&c0), &self.dbl2(g2)),
            ],
            [
                self.add2(&three(&self.mul_xi(&c1)), &self.dbl2(h0)),
                self.add2(&three(&a1), &self.dbl2(h1)),
                self.add2(&three(&b1), &self.dbl2(h2)),
            ],
        ]
    }

    /// `aᵉ` by squares and products from the top bit of the limbs `e`, the low limb first.
    pub(super) fn pow12(&self, a: &E12<N>, e: &[u64]) -> E12<N> {
        let bits = bit_length(e);
        if bits == 0 {
            return self.one12();
        }
        let mut r = *a;
        for i in (0..bits - 1).rev() {
            r = self.sqr12(&r);
            if (e[i / 64] >> (i % 64)) & 1 == 1 {
                r = self.mul12(&r, a);
            }
        }
        r
    }

    /// `aᵉ` in the cyclotomic subgroup over the signed digits `d` of `e`, the low one first and
    /// the top one 1: cyclotomic squares, and products by `a` or by its conjugate, its inverse there.
    pub(super) fn cyc_pow(&self, a: &E12<N>, d: &[i8]) -> E12<N> {
        let Some((_, rest)) = d.split_last() else {
            return self.one12();
        };
        let inv = self.conj12(a);
        let mut r = *a;
        for &x in rest.iter().rev() {
            r = self.cyc_sqr(&r);
            match x {
                1 => r = self.mul12(&r, a),
                -1 => r = self.mul12(&r, &inv),
                _ => {}
            }
        }
        r
    }

    #[inline(always)]
    pub(super) fn one2(&self) -> E2<N> {
        [self.f.one(), [0; N]]
    }

    #[inline(always)]
    pub(super) fn one6(&self) -> E6<N> {
        [self.one2(), [[0; N]; 2], [[0; N]; 2]]
    }

    #[inline(always)]
    pub(super) fn one12(&self) -> E12<N> {
        [self.one6(), [[[0; N]; 2]; 3]]
    }

    /// `aᵉ` on `F_p²`.
    pub(super) fn pow2(&self, a: &E2<N>, e: &[u64]) -> E2<N> {
        let bits = bit_length(e);
        if bits == 0 {
            return self.one2();
        }
        let mut r = *a;
        for i in (0..bits - 1).rev() {
            r = self.sqr2(&r);
            if (e[i / 64] >> (i % 64)) & 1 == 1 {
                r = self.mul2(&r, a);
            }
        }
        r
    }

    /// `aᵉ` on `F_p⁶`.
    pub(super) fn pow6(&self, a: &E6<N>, e: &[u64]) -> E6<N> {
        let bits = bit_length(e);
        if bits == 0 {
            return self.one6();
        }
        let mut r = *a;
        for i in (0..bits - 1).rev() {
            r = self.sqr6(&r);
            if (e[i / 64] >> (i % 64)) & 1 == 1 {
                r = self.mul6(&r, a);
            }
        }
        r
    }
}

/// The bits of the number with the limbs `e`, the low limb first.
fn bit_length(e: &[u64]) -> usize {
    match e.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * i + 64 - e[i].leading_zeros() as usize,
        None => 0,
    }
}
