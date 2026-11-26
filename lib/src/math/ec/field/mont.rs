//! Montgomery's arithmetic modulo an odd prime in arrays of limbs.

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use super::aarch64::{
    add_plain_4, add_plain_6, mont_add_4, mont_add_6, mont_mul_nc_4, mont_mul_nc_6, mont_sub_4,
    mont_sub_6, mul_wide_4, mul_wide_6, p256_mul, p256_sqr, p384_mul, p384_sqr, pseudo_add_4,
    pseudo_mul_4, pseudo_sqr_4, pseudo_sub_4, redc_wide_4, redc_wide_6, sub_wide_4, sub_wide_6,
    sub_wide_mod_4, sub_wide_mod_6,
};
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use super::same;
use super::{Field, array_of, halve, number_of};
use crate::math::field::TypedKernel;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use crate::math::field::TypedSpecial;
use crate::math::numbers::{Number, basecase, redc_inverse, redc_limbs};
use core::hint::select_unpredictable;
use core::mem::MaybeUninit;

/// From this many limbs the products and reductions are the kernels': measured on an Apple M4 Max,
/// a product on a curve of 6 limbs took 602,000 cycles with the rows here and 763,000 with the
/// kernels, whose calls cost more than their speed gives at this length.
const KERNEL: usize = 7;

/// The most limbs of a field in an array.
const MAX: usize = 9;

/// `1 / m mod 2^64` for an odd `m`, by Newton's iteration, which doubles the bits that are right.
const fn invert_limb(m: u64) -> u64 {
    let mut x = m;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(x)));
        i += 1;
    }
    x
}

/// The field modulo an odd prime `m` of `N` limbs, `R = 2^(64 N)`, whose elements are `x R mod m`,
/// fully reduced, below `m`.
#[derive(Clone)]
pub(crate) struct Mont<const N: usize> {
    /// The prime, the low limb first.
    m: [u64; N],
    /// `-1 / m mod 2^64`.
    minv: u64,
    /// `-1 / m mod 2^128`, for the kernel's reduction.
    minv2: [u64; 2],
    /// `R mod m`: one.
    one: [u64; N],
    /// `R^2 mod m`.
    r2: [u64; N],
    /// Whether the top bit of `m` is clear, `m < R / 2`: then a product's rows never carry out of
    /// `N + 1` limbs and keep below `2 m`, and the kernels of four and six limbs on aarch64 drop
    /// the carries above (decided here, once).
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_endian = "little", not(miri))),
        allow(dead_code)
    )]
    spare: bool,
    /// Whether the top two bits of `m` are clear, `4 m < R`: then a sum of two elements below `2 m`
    /// is one, and the products of two such sums below `m R`, which the lazy reduction of
    /// [`mul_complex`](Mont::mul_complex) takes (decided here, once).
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_endian = "little", not(miri))),
        allow(dead_code)
    )]
    spare2: bool,
    /// The prime as a number.
    p: Number,
}

impl<const N: usize> Mont<N> {
    /// The field modulo the prime `p`, with the limbs `m`.
    pub(in super::super) fn new(p: &Number, m: [u64; N]) -> Self {
        let r = Number::new_from_u8(1, p.byte_order) << (64 * N);
        let one = r.rem_euclid(p);
        let r2 = (&one * &one).rem_euclid(p);
        Mont {
            m,
            minv: invert_limb(m[0]).wrapping_neg(),
            minv2: redc_inverse(m[0], if N > 1 { m[1 % N] } else { 0 }),
            one: array_of(&one),
            r2: array_of(&r2),
            spare: m[N - 1] >> 63 == 0,
            spare2: m[N - 1] >> 62 == 0,
            p: p.clone(),
        }
    }

    /// `a b / R mod m` for a typed field ([`TypedKernel`]): the kernels of [`redc_product`]
    /// (`Mont::redc_product`), and P-256's and P-384's, chosen by the constants `k`, which a
    /// constant folds away.
    #[inline(always)]
    pub(crate) fn typed_mul(k: &TypedKernel<N>, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        {
            match (k.special, N) {
                (TypedSpecial::P256, 4) => return *same(&p256_mul(same(a), same(b))),
                (TypedSpecial::P384, 6) => return *same(&p384_mul(same(a), same(b))),
                _ => {}
            }
            if k.spare {
                match N {
                    4 => return *same(&mont_mul_nc_4(same(a), same(b), same(&k.m), k.minv)),
                    6 => return *same(&mont_mul_nc_6(same(a), same(b), same(&k.m), k.minv)),
                    _ => {}
                }
            }
        }
        if N >= KERNEL {
            let mut t = MaybeUninit::<[u64; 2 * MAX]>::uninit();
            let t = t.as_mut_ptr().cast::<u64>();
            // SAFETY: `t` holds `2 * MAX >= 2 N` limbs (a typed field has at most `MAX`), apart
            // from `a` and `b`, and the kernel writes all the `2 N` that `reduce` reads.
            return unsafe {
                basecase::mul(t, a.as_ptr(), b.as_ptr(), N);
                reduce(t, &k.m, k.minv2)
            };
        }
        rows(a, b, &k.m, k.minv)
    }

    /// `a^2 / R mod m` for a typed field.
    #[inline(always)]
    pub(crate) fn typed_sqr(k: &TypedKernel<N>, a: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        {
            match (k.special, N) {
                (TypedSpecial::P256, 4) => return *same(&p256_sqr(same(a))),
                (TypedSpecial::P384, 6) => return *same(&p384_sqr(same(a))),
                _ => {}
            }
            if k.spare {
                match N {
                    4 => return *same(&mont_mul_nc_4(same(a), same(a), same(&k.m), k.minv)),
                    6 => return *same(&mont_mul_nc_6(same(a), same(a), same(&k.m), k.minv)),
                    _ => {}
                }
            }
        }
        if N >= KERNEL {
            let mut t = MaybeUninit::<[u64; 2 * MAX]>::uninit();
            let t = t.as_mut_ptr().cast::<u64>();
            // SAFETY: as in `typed_mul`.
            return unsafe {
                basecase::sqr(t, a.as_ptr(), N);
                reduce(t, &k.m, k.minv2)
            };
        }
        rows(a, a, &k.m, k.minv)
    }

    /// `a + b mod m` for a typed field.
    #[inline(always)]
    pub(crate) fn typed_add(k: &TypedKernel<N>, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        match N {
            4 => return *same(&mont_add_4(same(a), same(b), same(&k.m))),
            6 => return *same(&mont_add_6(same(a), same(b), same(&k.m))),
            _ => {}
        }
        let (s, carry) = add_n(a, b);
        let (d, borrow) = sub_n(&s, &k.m);
        select(carry || !borrow, &d, &s)
    }

    /// `a - b mod m` for a typed field.
    #[inline(always)]
    pub(crate) fn typed_sub(k: &TypedKernel<N>, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        match N {
            4 => return *same(&mont_sub_4(same(a), same(b), same(&k.m))),
            6 => return *same(&mont_sub_6(same(a), same(b), same(&k.m))),
            _ => {}
        }
        let (d, borrow) = sub_n(a, b);
        let (s, _) = add_n(&d, &k.m);
        select(borrow, &s, &d)
    }

    /// `-a mod m` for a typed field.
    #[inline(always)]
    pub(crate) fn typed_neg(k: &TypedKernel<N>, a: &[u64; N]) -> [u64; N] {
        let (d, _) = sub_n(&k.m, a);
        select(a.iter().fold(0, |x, l| x | l) == 0, a, &d)
    }

    /// `a b` modulo a prime `2^k - c` of four limbs with `2^256 = f` (`f < 2^62`), not reduced
    /// below it, for a typed field: `Pseudo`'s kernel.
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    #[inline(always)]
    pub(crate) fn typed_pseudo_mul(a: &[u64; N], b: &[u64; N], f: u64) -> [u64; N] {
        *same(&pseudo_mul_4(same(a), same(b), f))
    }

    /// `a^2` modulo the prime with `2^256 = f`, not reduced below it.
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    #[inline(always)]
    pub(crate) fn typed_pseudo_sqr(a: &[u64; N], f: u64) -> [u64; N] {
        *same(&pseudo_sqr_4(same(a), f))
    }

    /// `a + b` modulo the prime with `2^256 = f`, not reduced below it.
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    #[inline(always)]
    pub(crate) fn typed_pseudo_add(a: &[u64; N], b: &[u64; N], f: u64) -> [u64; N] {
        *same(&pseudo_add_4(same(a), same(b), f))
    }

    /// `a - b` modulo the prime with `2^256 = f`, not reduced below it.
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    #[inline(always)]
    pub(crate) fn typed_pseudo_sub(a: &[u64; N], b: &[u64; N], f: u64) -> [u64; N] {
        *same(&pseudo_sub_4(same(a), same(b), f))
    }

    /// For a prime of one limb, the prime and `-1 / m mod 2^64`.
    #[inline(always)]
    pub(crate) fn word(&self) -> Option<(u64, u64)> {
        (N == 1).then(|| (self.m[0], self.minv))
    }

    /// REDC of the `2 N` limbs at `t`, below `R m`, which it clobbers, by the kernel.
    ///
    /// # Safety
    ///
    /// `t` must hold `2 N` limbs.
    #[inline(always)]
    unsafe fn reduce(&self, t: *mut u64) -> [u64; N] {
        // SAFETY: forwarded from the caller.
        unsafe { reduce(t, &self.m, self.minv2) }
    }

    /// `a b / R mod m`.
    #[inline(always)]
    fn redc_product(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if self.spare {
            match N {
                4 => return *same(&mont_mul_nc_4(same(a), same(b), same(&self.m), self.minv)),
                6 => return *same(&mont_mul_nc_6(same(a), same(b), same(&self.m), self.minv)),
                _ => {}
            }
        }
        if N >= KERNEL {
            let mut t = MaybeUninit::<[u64; 2 * MAX]>::uninit();
            let t = t.as_mut_ptr().cast::<u64>();
            // SAFETY: `t` holds `2 * MAX >= 2 N` limbs, apart from `a` and `b`, and the kernel
            // writes all the `2 N` that `reduce` reads.
            return unsafe {
                basecase::mul(t, a.as_ptr(), b.as_ptr(), N);
                self.reduce(t)
            };
        }
        self.rows(a, b)
    }

    /// `a^2 / R mod m`.
    #[inline(always)]
    fn redc_square(&self, a: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if self.spare {
            match N {
                // The product's rows overlap their reductions, which a square's separate
                // reduction cannot: the product by itself is the faster square.
                4 => return *same(&mont_mul_nc_4(same(a), same(a), same(&self.m), self.minv)),
                6 => return *same(&mont_mul_nc_6(same(a), same(a), same(&self.m), self.minv)),
                _ => {}
            }
        }
        if N >= KERNEL {
            let mut t = MaybeUninit::<[u64; 2 * MAX]>::uninit();
            let t = t.as_mut_ptr().cast::<u64>();
            // SAFETY: as in `redc_product`.
            return unsafe {
                basecase::sqr(t, a.as_ptr(), N);
                self.reduce(t)
            };
        }
        self.rows(a, a)
    }

    /// The product `(a₀ + a₁u)(b₀ + b₁u)` in `F_m[u] / (u² + 1)`, the arithmetic of `F_m²` when
    /// `−1` is not a square: `(a₀b₀ − a₁b₁) + ((a₀ + a₁)(b₀ + b₁) − a₀b₀ − a₁b₁)u`, Karatsuba's
    /// three products.
    ///
    /// For a prime with two spare bits, by four and six limbs on aarch64, by a lazy reduction: the
    /// three products as double-width values, of `2 N` limbs and not reduced, the sums `a₀ + a₁`
    /// and `b₀ + b₁` not reduced either (below `2 m`, so their product is below `4 m² < m R`), then
    /// `a₀b₀ − a₁b₁` modulo `m R` (plus `m R` on a borrow) and `(a₀ + a₁)(b₀ + b₁) − a₀b₀ − a₁b₁`,
    /// which is `a₀b₁ + a₁b₀ < 2 m²` exactly, each reduced once: two reductions in place of three,
    /// and no reduction of the sums. Otherwise three Montgomery products.
    #[inline(always)]
    pub(crate) fn mul_complex(&self, a: &[[u64; N]; 2], b: &[[u64; N]; 2]) -> [[u64; N]; 2] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if self.spare2 {
            macro_rules! lazy {
                ($n:literal, $mul:ident, $add:ident, $sub:ident, $sub_mod:ident, $redc:ident) => {{
                    let (a0, a1, b0, b1): (&[u64; $n], _, _, _) =
                        (same(&a[0]), same(&a[1]), same(&b[0]), same(&b[1]));
                    let m: &[u64; $n] = same(&self.m);
                    let t0 = $mul(a0, b0);
                    let t1 = $mul(a1, b1);
                    let t2 = $mul(&$add(a0, a1), &$add(b0, b1));
                    let c0 = $sub_mod(&t0, &t1, m);
                    let c1 = $sub(&$sub(&t2, &t0), &t1);
                    return [
                        *same(&$redc(&c0, m, self.minv)),
                        *same(&$redc(&c1, m, self.minv)),
                    ];
                }};
            }
            match N {
                4 => lazy!(
                    4,
                    mul_wide_4,
                    add_plain_4,
                    sub_wide_4,
                    sub_wide_mod_4,
                    redc_wide_4
                ),
                6 => lazy!(
                    6,
                    mul_wide_6,
                    add_plain_6,
                    sub_wide_6,
                    sub_wide_mod_6,
                    redc_wide_6
                ),
                _ => {}
            }
        }
        let v0 = self.redc_product(&a[0], &b[0]);
        let v1 = self.redc_product(&a[1], &b[1]);
        let s = self.redc_product(&self.add(&a[0], &a[1]), &self.add(&b[0], &b[1]));
        [self.sub(&v0, &v1), self.sub(&self.sub(&s, &v0), &v1)]
    }

    /// The square `(a₀ + a₁u)²` in `F_m[u] / (u² + 1)`: `(a₀ + a₁)(a₀ − a₁) + 2a₀a₁u`, two
    /// products.
    ///
    /// For a prime with two spare bits, by four and six limbs on aarch64, the operands of the
    /// products are not reduced: `a₀ + a₁`, `a₀ + (m − a₁)` and `2a₀`, each below `2m`. The
    /// kernel's rows then stay below `3m < R` and its result `(xy + Qm)/R < (4m² + Rm)/R < 2m`, so
    /// that its one subtraction still gives a residue below `m`.
    #[inline(always)]
    pub(crate) fn sqr_complex(&self, a: &[[u64; N]; 2]) -> [[u64; N]; 2] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if self.spare2 {
            macro_rules! lazy {
                ($n:literal, $mul:ident, $add:ident) => {{
                    let (a0, a1): (&[u64; $n], &[u64; $n]) = (same(&a[0]), same(&a[1]));
                    let m: &[u64; $n] = same(&self.m);
                    let (neg, _) = sub_n(m, a1);
                    let c0 = $mul(&$add(a0, a1), &$add(a0, &neg), m, self.minv);
                    let c1 = $mul(&$add(a0, a0), a1, m, self.minv);
                    return [*same(&c0), *same(&c1)];
                }};
            }
            match N {
                4 => lazy!(4, mont_mul_nc_4, add_plain_4),
                6 => lazy!(6, mont_mul_nc_6, add_plain_6),
                _ => {}
            }
        }
        let c0 = self.redc_product(&self.add(&a[0], &a[1]), &self.sub(&a[0], &a[1]));
        let ab = self.redc_product(&a[0], &a[1]);
        [c0, self.add(&ab, &ab)]
    }

    /// The limbs of the number `x` stands for, below `m`: `x / R mod m`, one reduction.
    #[inline(always)]
    pub(crate) fn canonical(&self, x: &[u64; N]) -> [u64; N] {
        let mut one = [0; N];
        one[0] = 1;
        self.redc_product(x, &one)
    }

    /// The element of the number with the limbs `x`, below `m`: `x R mod m`, one product by
    /// `R^2`.
    #[inline(always)]
    pub(crate) fn residue(&self, x: &[u64; N]) -> [u64; N] {
        self.redc_product(x, &self.r2)
    }

    /// `a b / R mod m`, a row of the product and a row of the reduction for each limb.
    #[inline(always)]
    fn rows(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        rows(a, b, &self.m, self.minv)
    }
}

/// `a b / R mod m`, a row of the product and a row of the reduction for each limb, for the prime
/// `m` with `minv = -1 / m mod 2^64`.
#[inline(always)]
fn rows<const N: usize>(a: &[u64; N], b: &[u64; N], m: &[u64; N], minv: u64) -> [u64; N] {
    let mut t = [0u64; N];
    let mut top = 0u64;
    for &bi in b {
        let mut c = 0u64;
        for j in 0..N {
            let p = u128::from(a[j]) * u128::from(bi) + u128::from(t[j]) + u128::from(c);
            t[j] = p as u64;
            c = (p >> 64) as u64;
        }
        let s = u128::from(top) + u128::from(c);
        let (tn, tn1) = (s as u64, (s >> 64) as u64);
        let q = t[0].wrapping_mul(minv);
        let p = u128::from(q) * u128::from(m[0]) + u128::from(t[0]);
        let mut c = (p >> 64) as u64;
        for j in 1..N {
            let p = u128::from(q) * u128::from(m[j]) + u128::from(t[j]) + u128::from(c);
            t[j - 1] = p as u64;
            c = (p >> 64) as u64;
        }
        let s = u128::from(tn) + u128::from(c);
        t[N - 1] = s as u64;
        top = tn1 + (s >> 64) as u64;
    }
    // Below `2 m`: one subtraction when it is at least `m`.
    let (d, borrow) = sub_n(&t, m);
    select(top != 0 || !borrow, &d, &t)
}

/// REDC of the `2 N` limbs at `t`, below `R m`, which it clobbers, by the kernel, for the prime
/// `m` with `minv2 = -1 / m mod 2^128`.
///
/// # Safety
///
/// `t` must hold `2 N` limbs.
#[inline(always)]
unsafe fn reduce<const N: usize>(t: *mut u64, m: &[u64; N], minv2: [u64; 2]) -> [u64; N] {
    let mut r = MaybeUninit::<[u64; N]>::uninit();
    // SAFETY: `r` has room for the `N` limbs the kernel writes, apart from `t`, and `m` holds `N`
    // limbs.
    let (r, over) = unsafe {
        let over = redc_limbs(r.as_mut_ptr().cast(), t, m.as_ptr(), N, minv2);
        (r.assume_init(), over)
    };
    // The kernel's result is below `R + m`, and a residue here below `m`.
    let (d, borrow) = sub_n(&r, m);
    select(over || !borrow, &d, &r)
}

/// `a + b` and the carry out.
#[inline(always)]
fn add_n<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut r = [0; N];
    let mut carry = false;
    for i in 0..N {
        let (x, c1) = a[i].overflowing_add(b[i]);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        r[i] = x;
        carry = c1 | c2;
    }
    (r, carry)
}

/// `a - b` and the borrow out.
#[inline(always)]
fn sub_n<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut r = [0; N];
    let mut borrow = false;
    for i in 0..N {
        let (x, b1) = a[i].overflowing_sub(b[i]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        r[i] = x;
        borrow = b1 | b2;
    }
    (r, borrow)
}

/// `a` when `c` holds, else `b`, without a branch: the reductions go either way at random.
#[inline(always)]
fn select<const N: usize>(c: bool, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut r = [0; N];
    for i in 0..N {
        r[i] = select_unpredictable(c, a[i], b[i]);
    }
    r
}

impl<const N: usize> Field for Mont<N> {
    type E = [u64; N];

    #[inline(always)]
    fn limbs(&self) -> usize {
        N
    }

    #[inline(always)]
    fn load(&self, x: &[u64]) -> [u64; N] {
        let mut r = [0; N];
        r.copy_from_slice(&x[..N]);
        r
    }

    #[inline(always)]
    fn store(&self, x: &[u64; N], r: &mut [u64]) {
        r[..N].copy_from_slice(x);
    }

    #[inline(always)]
    fn zero(&self) -> [u64; N] {
        [0; N]
    }

    #[inline(always)]
    fn one(&self) -> [u64; N] {
        self.one
    }

    #[inline(always)]
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        match N {
            4 => return *same(&mont_add_4(same(a), same(b), same(&self.m))),
            6 => return *same(&mont_add_6(same(a), same(b), same(&self.m))),
            _ => {}
        }
        let (s, carry) = add_n(a, b);
        let (d, borrow) = sub_n(&s, &self.m);
        select(carry || !borrow, &d, &s)
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        match N {
            4 => return *same(&mont_sub_4(same(a), same(b), same(&self.m))),
            6 => return *same(&mont_sub_6(same(a), same(b), same(&self.m))),
            _ => {}
        }
        let (d, borrow) = sub_n(a, b);
        let (s, _) = add_n(&d, &self.m);
        select(borrow, &s, &d)
    }

    #[inline(always)]
    fn neg(&self, a: &[u64; N]) -> [u64; N] {
        let (d, _) = sub_n(&self.m, a);
        select(self.is_zero(a), a, &d)
    }

    #[inline(always)]
    fn half(&self, a: &[u64; N]) -> [u64; N] {
        halve(a, &self.m)
    }

    #[inline(always)]
    fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.redc_product(a, b)
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; N]) -> [u64; N] {
        self.redc_square(a)
    }

    #[inline(always)]
    fn is_zero(&self, a: &[u64; N]) -> bool {
        a.iter().fold(0, |x, l| x | l) == 0
    }

    #[inline(always)]
    fn equal(&self, a: &[u64; N], b: &[u64; N]) -> bool {
        a == b
    }

    fn enter(&self, x: &Number) -> [u64; N] {
        let r = array_of(&x.rem_euclid(&self.p));
        self.redc_product(&r, &self.r2)
    }

    fn leave(&self, x: &[u64; N]) -> Number {
        let mut one = [0; N];
        one[0] = 1;
        number_of(&self.redc_product(x, &one), self.p.byte_order)
    }

    #[inline(always)]
    fn modulus(&self) -> &Number {
        &self.p
    }
}
