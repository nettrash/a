//! Montgomery's arithmetic modulo an odd prime in arrays of limbs.

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use super::aarch64::{mont_add_4, mont_add_6, mont_sub_4, mont_sub_6};
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use super::same;
use super::{Field, array_of, number_of};
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
pub(in super::super) struct Mont<const N: usize> {
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
            p: p.clone(),
        }
    }

    /// REDC of the `2 N` limbs at `t`, below `R m`, which it clobbers, by the kernel.
    ///
    /// # Safety
    ///
    /// `t` must hold `2 N` limbs.
    #[inline(always)]
    unsafe fn reduce(&self, t: *mut u64) -> [u64; N] {
        let mut r = MaybeUninit::<[u64; N]>::uninit();
        // SAFETY: `r` has room for the `N` limbs the kernel writes, apart from `t`, and `m` holds
        // `N` limbs.
        let (r, over) = unsafe {
            let over = redc_limbs(r.as_mut_ptr().cast(), t, self.m.as_ptr(), N, self.minv2);
            (r.assume_init(), over)
        };
        // The kernel's result is below `R + m`, and a residue here below `m`.
        let (d, borrow) = sub_n(&r, &self.m);
        select(over || !borrow, &d, &r)
    }

    /// `a b / R mod m`.
    #[inline(always)]
    fn redc_product(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
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

    /// `a b / R mod m`, a row of the product and a row of the reduction for each limb.
    #[inline(always)]
    fn rows(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let m = &self.m;
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
            let q = t[0].wrapping_mul(self.minv);
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
