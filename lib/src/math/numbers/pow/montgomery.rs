//! Montgomery's arithmetic modulo an odd number `m` of `n` limbs (`B = 2^64`, `R = B^n`): a residue
//! `x` is held as `x R mod m`, and the product of two such residues is reduced by REDC, which
//! divides by `R` modulo `m` without a division: it adds the multiple `q m` of `m` that clears the
//! low `n` limbs of the product and drops them.
//!
//! - One limb and two: the residues in registers, [`Mont1`] and [`Mont2`], each product reduced as
//!   it is formed; three to seven limbs too, [`MontN`], by the coarsely integrated operand scanning,
//!   a row of the product and a row of the reduction for each limb.
//! - Below [`REDC_N`] limbs: the product by the multiplication's kernels (its square by the
//!   square's), then REDC by the kernel, [`redc_1`]: two limbs of the quotient a pass on aarch64,
//!   `(u_j + u_{j+1} B) (-1/m) mod B^2` times `m` added in by the two-row kernel, a limb a pass
//!   elsewhere, `n^2` limb products either way.
//! - From [`REDC_N`] limbs: REDC by products, [`redc_n`]: the quotient of all `n` limbs at once,
//!   the low half of the product of the low limbs and `1/m mod R` (Hensel's quotient, see
//!   [`mullo`]), then its product by `m`, whose high half is subtracted from the product's.
//!
//! A residue is kept below `R`, not always below `m`, as GMP's `mpn_powm` keeps it: REDC of a
//! product of two such residues is below `R + m`, and one subtraction of `m` when it carries out of
//! the `n` limbs brings it below `R`. The last reduction, of the residue alone, gives at most `m`,
//! which one comparison makes the remainder.

use super::super::div::limbs::{
    bdiv_q, binvert_limb, div_qr, div_qr_1, div_qr_2, invert_3by2, invert_limb, qr_itch, rshift,
};
use super::super::mul::limbs::{add_n, cmp_n, mul, mul_itch, sqr, sqr_itch, sub_n};
use super::kernels::redc;
use super::low::{mullo, mullo_itch};
use super::window::{Registers, Ring};
use core::cmp::Ordering;
use core::ptr;

/// From this many limbs in the modulus, REDC is computed by products, [`redc_n`], instead of by the
/// kernel's passes: measured on an Apple M4 Max with exponents of 256 bits, a modulus of 64 limbs
/// took 2.09 million cycles by passes against 2.40 million by products, one of 128 limbs 7.8 to 8.1
/// million against 7.2 to 7.5, and one of 96 limbs about as long either way.
pub(super) const REDC_N: usize = 100;

/// `n` rounded up to an even number of limbs, so that a buffer after `n` limbs of an aligned one
/// is aligned too.
#[inline(always)]
pub(in super::super) const fn even(n: usize) -> usize {
    (n + 1) & !1
}

// Remainders by division

/// The limbs of scratch space that [`rem`] needs for a dividend of `nu` limbs by a divisor of `n`:
/// the division's, of the dividend shifted, with the limb above it or without.
pub(in super::super) fn rem_itch(nu: usize, n: usize) -> usize {
    even(n) + 2 * even(nu + 1) + qr_itch(nu + 1, n).max(qr_itch(nu, n))
}

/// Sets the `n` limbs at `r` to the remainder of the `nu` limbs at `u` times `B^k` by the `n`
/// limbs at `m`, whose top limb is not zero.
///
/// The divisor is shifted so that its top bit is set, and the dividend with it, as the division
/// does, and the remainder shifted back.
///
/// # Safety
///
/// The limbs must be valid, apart, `n >= 1`; the scratch at `s` holds [`rem_itch`]`(nu + k, n)`
/// limbs and starts on 16 bytes.
pub(in super::super) unsafe fn rem(
    r: *mut u64,
    u: *const u64,
    nu: usize,
    k: usize,
    m: *const u64,
    n: usize,
    s: *mut u64,
) {
    let total = nu + k;
    // SAFETY: as the caller guarantees: the divisor's `n` limbs at `s`, the dividend's `total + 1`
    // after them, the quotient's `total + 1` after those and the division's scratch last.
    unsafe {
        if total < n {
            ptr::write_bytes(r, 0, n);
            ptr::copy_nonoverlapping(u, r.add(k), nu);
            return;
        }
        let shift = (*m.add(n - 1)).leading_zeros();
        let d = s;
        let x = d.add(even(n));
        let q = x.add(even(total + 1));
        let t = q.add(even(total + 1));
        lshift(d, m, n, shift);
        ptr::write_bytes(x, 0, k);
        *x.add(total) = lshift(x.add(k), u, nu, shift);
        match n {
            1 => {
                let rest = div_qr_1::<false>(q, x, total, *x.add(total), *d, invert_limb(*d));
                *r = rest >> shift;
            }
            2 => {
                let v = invert_3by2(*d.add(1), *d);
                let (r1, r0) = div_qr_2(q, x, total + 1, *d.add(1), *d, v);
                (*r, *r.add(1)) = if shift == 0 {
                    (r0, r1)
                } else {
                    ((r0 >> shift) | (r1 << (64 - shift)), r1 >> shift)
                };
            }
            _ => {
                let v = invert_3by2(*d.add(n - 1), *d.add(n - 2));
                let nn = if *x.add(total) != 0 { total + 1 } else { total };
                div_qr(q, x, nn, d, n, v, t);
                if shift != 0 {
                    rshift(x, n, shift);
                }
                ptr::copy_nonoverlapping(x, r, n);
            }
        }
    }
}

/// Sets the `n` limbs at `r` to the `n` limbs at `x` shifted left by `s` bits, `s < 64`, and
/// returns the bits shifted out of the top.
///
/// # Safety
///
/// The limbs must be valid; `r` may be `x`.
#[inline]
pub(in super::super) unsafe fn lshift(r: *mut u64, x: *const u64, n: usize, s: u32) -> u64 {
    // SAFETY: as the caller guarantees; limb `i` of `x` is read before limb `i` of `r` is
    // written, from the top.
    unsafe {
        if s == 0 {
            ptr::copy(x, r, n);
            return 0;
        }
        let out = *x.add(n - 1) >> (64 - s);
        for i in (1..n).rev() {
            *r.add(i) = (*x.add(i) << s) | (*x.add(i - 1) >> (64 - s));
        }
        *r = *x << s;
        out
    }
}

// Montgomery's reduction on limbs

/// Montgomery's arithmetic modulo the odd `n` limbs at `m`, `n >= 3` (the shorter moduli are in
/// registers), as the module documentation describes.
pub(in super::super) struct Montgomery {
    /// The modulus, odd, its top limb not zero.
    pub(in super::super) m: *const u64,
    /// Its limbs.
    pub(in super::super) n: usize,
    /// `-1/m mod B^2`, the low limb first, for [`redc_1`].
    minv: [u64; 2],
    /// `1/m mod R`, `n` limbs, for [`redc_n`]; null below [`REDC_N`] limbs.
    inv: *const u64,
}

impl Montgomery {
    /// The limbs of working memory that [`Montgomery::new`] keeps for a modulus of `n` limbs.
    pub(in super::super) fn new_itch(n: usize) -> usize {
        if n >= REDC_N { 2 * even(n) } else { 0 }
    }

    /// The limbs of scratch space that the products and squares need for a modulus of `n` limbs.
    pub(in super::super) fn itch(n: usize) -> usize {
        let redc = if n >= REDC_N {
            n + 2 * n + mullo_itch(n).max(mul_itch(n, n))
        } else {
            0
        };
        2 * n + mul_itch(n, n).max(sqr_itch(n)).max(redc)
    }

    /// The arithmetic modulo the `n` limbs at `m`, with `1/m mod R` computed in the
    /// [`Montgomery::new_itch`] limbs at `s` when REDC takes products.
    ///
    /// # Safety
    ///
    /// The limbs of `m` must stay valid while the arithmetic is used, `m` odd and `n >= 3`, and
    /// the working memory at `s` too.
    pub(in super::super) unsafe fn new(m: *const u64, n: usize, s: *mut u64) -> Self {
        // SAFETY: as the caller guarantees.
        unsafe {
            let v = binvert_limb(*m);
            let inv = if n >= REDC_N {
                // `1/m mod B^n`, Hensel's quotient of 1 by `m`.
                let one = s.add(even(n));
                ptr::write_bytes(one, 0, n);
                *one = 1;
                bdiv_q(s, one, n, m, n, v);
                s.cast_const()
            } else {
                ptr::null()
            };
            // `1/m mod B^2`: the high limb of `v` corrected by Newton's step.
            let c = ((u128::from(*m) * u128::from(v)) >> 64) as u64;
            let v1 = c
                .wrapping_add((*m.add(1)).wrapping_mul(v))
                .wrapping_mul(v)
                .wrapping_neg();
            Montgomery {
                m,
                n,
                minv: [v.wrapping_neg(), !v1],
                inv,
            }
        }
    }

    /// Sets the `n` limbs at `r` to `u / R mod m`, below `R`, for the `2n` limbs at `u`, below
    /// `R m`, which it clobbers; `t` is the scratch.
    ///
    /// # Safety
    ///
    /// The limbs must be valid, `r` apart from `u`, and the scratch that of [`Montgomery::itch`]
    /// past `2n` limbs.
    #[inline(always)]
    pub(in super::super) unsafe fn redc(&self, r: *mut u64, u: *mut u64, t: *mut u64) {
        // SAFETY: forwarded.
        unsafe {
            if self.inv.is_null() {
                redc_1(r, u, self.m, self.n, self.minv);
            } else {
                redc_n(r, u, self.m, self.n, self.inv, t);
            }
        }
    }

    /// Sets the `n` limbs at `r` to the residue of `x`, below `m`, from the residue `x R` at `a`:
    /// the conversion out of Montgomery's form.
    ///
    /// # Safety
    ///
    /// The limbs must be valid, `r` may be `a`, and the scratch that of [`Montgomery::itch`].
    pub(in super::super) unsafe fn plain(&self, r: *mut u64, a: *const u64, t: *mut u64) {
        let n = self.n;
        // SAFETY: as the caller guarantees.
        unsafe {
            ptr::copy_nonoverlapping(a, t, n);
            ptr::write_bytes(t.add(n), 0, n);
            self.redc(r, t, t.add(2 * n));
            self.reduce(r);
        }
    }

    /// Subtracts `m` from the `n` limbs at `r` when they are at least `m`: a residue below `2m`
    /// becomes the remainder.
    ///
    /// # Safety
    ///
    /// The limbs must be valid.
    #[inline(always)]
    pub(in super::super) unsafe fn reduce(&self, r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            if cmp_n(r, self.m, self.n) != Ordering::Less {
                sub_n(r, r, self.m, self.n);
            }
        }
    }
}

impl Ring for Montgomery {
    #[inline(always)]
    fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    unsafe fn sqr(&self, r: *mut u64, a: *const u64, t: *mut u64) {
        let n = self.n;
        // SAFETY: the square goes to the scratch, then is reduced into `r`.
        unsafe {
            sqr(t, a, n, t.add(2 * n));
            self.redc(r, t, t.add(2 * n));
        }
    }

    #[inline(always)]
    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64, t: *mut u64) {
        let n = self.n;
        // SAFETY: as in `sqr`.
        unsafe {
            mul(t, a, n, b, n, t.add(2 * n));
            self.redc(r, t, t.add(2 * n));
        }
    }
}

/// REDC by the kernel: sets the `n` limbs at `r` to `u / R mod m`, below `R`, for the `2n` limbs at
/// `u`, below `R m`, which it clobbers; the kernel's result, below `R + m`, less `m` when it carries
/// out of the `n` limbs.
///
/// # Safety
///
/// `n >= 1`; the limbs must be valid, `r` apart from `u`, and `minv` must be `-1/m mod B^2`.
#[inline(always)]
unsafe fn redc_1(r: *mut u64, u: *mut u64, m: *const u64, n: usize, minv: [u64; 2]) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if redc(r, u, m, n, minv) {
            sub_n(r, r, m, n);
        }
    }
}

/// REDC by products: sets the `n` limbs at `r` to `u / R mod m`, below `R`, for the `2n` limbs at
/// `u`, below `R m`, with `inv = 1/m mod R`.
///
/// Hensel's quotient `q = u inv mod R` makes `q m` agree with `u` in its low `n` limbs, so
/// `(u - q m) / R` is the difference of the high halves exactly, at least `-m` as `q m < R m`: one
/// addition of `m` when it is negative makes it a residue below `R`.
///
/// # Safety
///
/// The limbs must be valid, `r` apart from `u`, and the scratch at `t` hold
/// `3n + max(mullo_itch(n), mul_itch(n, n))` limbs.
unsafe fn redc_n(
    r: *mut u64,
    u: *const u64,
    m: *const u64,
    n: usize,
    inv: *const u64,
    t: *mut u64,
) {
    // SAFETY: as the caller guarantees: the quotient's `n` limbs at `t`, the product's `2n` after
    // them, and the scratch of both after that.
    unsafe {
        let q = t;
        let p = t.add(n);
        let rest = p.add(2 * n);
        mullo(q, u, inv, n, rest);
        mul(p, m, n, q, n, rest);
        if sub_n(r, u.add(n), p.add(n), n) {
            add_n(r, r, m, n);
        }
    }
}

// One and two limbs in registers

/// Montgomery's arithmetic modulo an odd `m` of one limb, `R = B`, with the residues fully reduced,
/// below `m`.
#[derive(Clone, Copy)]
pub(in super::super) struct Mont1 {
    /// The modulus, odd.
    pub(in super::super) m: u64,
    /// `1/m mod B`.
    inv: u64,
}

impl Mont1 {
    /// The arithmetic modulo the odd `m`.
    #[inline(always)]
    pub(in super::super) fn new(m: u64) -> Self {
        Mont1 {
            m,
            inv: binvert_limb(m),
        }
    }

    /// `(hi B + lo) / B mod m` for `hi < m`: with `q = lo / m mod B`, `q m` agrees with `lo` in its
    /// low limb, so the result is `hi` less the high limb of `q m`, plus `m` when that is negative
    /// (GMP's `MPN_REDC_0`).
    #[inline(always)]
    pub(in super::super) fn redc(&self, hi: u64, lo: u64) -> u64 {
        let q = lo.wrapping_mul(self.inv);
        let p = ((u128::from(q) * u128::from(self.m)) >> 64) as u64;
        let (r, borrow) = hi.overflowing_sub(p);
        if borrow { r.wrapping_add(self.m) } else { r }
    }

    /// `x` from its residue `x B mod m`.
    #[inline(always)]
    pub(in super::super) fn plain(&self, a: u64) -> u64 {
        self.redc(0, a)
    }
}

impl Registers for Mont1 {
    type V = u64;

    #[inline(always)]
    fn sqr(&self, a: u64) -> u64 {
        let p = u128::from(a) * u128::from(a);
        self.redc((p >> 64) as u64, p as u64)
    }

    #[inline(always)]
    fn mul(&self, a: u64, b: u64) -> u64 {
        let p = u128::from(a) * u128::from(b);
        self.redc((p >> 64) as u64, p as u64)
    }
}

/// Montgomery's arithmetic modulo an odd `m` of two limbs, `R = B^2`, with the residues fully
/// reduced, below `m`.
#[derive(Clone, Copy)]
pub(in super::super) struct Mont2 {
    /// The modulus, odd, the low limb first.
    pub(in super::super) m: [u64; 2],
    /// `-1/m mod B^2`, the low limb first.
    minv: [u64; 2],
}

impl Mont2 {
    /// The arithmetic modulo the odd `m`.
    #[inline(always)]
    pub(in super::super) fn new(m: [u64; 2]) -> Self {
        // `1/m mod B^2`: the inverse of the low limb, corrected by Newton's step, negated.
        let v = binvert_limb(m[0]);
        let c = ((u128::from(m[0]) * u128::from(v)) >> 64) as u64;
        let v1 = c
            .wrapping_add(m[1].wrapping_mul(v))
            .wrapping_mul(v)
            .wrapping_neg();
        Mont2 {
            m,
            minv: [v.wrapping_neg(), !v1],
        }
    }

    /// REDC of the four limbs `u`, below `R m`: the quotient of both low limbs at once,
    /// `q = u (-1/m) mod B^2`, whose product by `m` makes the low half zero, then one subtraction
    /// of `m` when the high half is not below it. The two limbs of `q` take three independent
    /// products, so a REDC waits for one product less than two steps of a limb would.
    #[inline(always)]
    pub(in super::super) fn redc(&self, u: [u64; 4]) -> [u64; 2] {
        let [m0, m1] = self.m;
        let [n0, n1] = self.minv;
        let [u0, u1, u2, u3] = u;
        let p = u128::from(u0) * u128::from(n0);
        let q0 = p as u64;
        let q1 = ((p >> 64) as u64)
            .wrapping_add(u0.wrapping_mul(n1))
            .wrapping_add(u1.wrapping_mul(n0));
        // `u + q m`, whose low two limbs are zero: the carry out of them is 1 unless both of `u`'s
        // are zero.
        let qm = mul_2([q0, q1], [m0, m1]);
        let carry = u64::from(u0 | u1 != 0);
        let lo = u128::from(u2) | (u128::from(u3) << 64);
        let hi = u128::from(qm[2]) | (u128::from(qm[3]) << 64);
        let (s, c1) = lo.overflowing_add(hi);
        let (s, c2) = s.overflowing_add(u128::from(carry));
        // Below `2m < 2R`: at most `R` more than the two limbs hold.
        let m = u128::from(m0) | (u128::from(m1) << 64);
        let r = if c1 | c2 || s >= m {
            s.wrapping_sub(m)
        } else {
            s
        };
        [r as u64, (r >> 64) as u64]
    }

    /// `x` from its residue `x R mod m`.
    #[inline(always)]
    pub(in super::super) fn plain(&self, a: [u64; 2]) -> [u64; 2] {
        self.redc([a[0], a[1], 0, 0])
    }
}

/// The four limbs of the product of the two-limb `a` and `b`.
#[inline(always)]
fn mul_2(a: [u64; 2], b: [u64; 2]) -> [u64; 4] {
    let p00 = u128::from(a[0]) * u128::from(b[0]);
    let p01 = u128::from(a[0]) * u128::from(b[1]);
    let p10 = u128::from(a[1]) * u128::from(b[0]);
    let p11 = u128::from(a[1]) * u128::from(b[1]);
    let c1 = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
    let c2 = (c1 >> 64) + (p01 >> 64) + (p10 >> 64) + u128::from(p11 as u64);
    let c3 = (c2 >> 64) + (p11 >> 64);
    [p00 as u64, c1 as u64, c2 as u64, c3 as u64]
}

/// The four limbs of the square of the two-limb `a`: the cross product once, doubled.
#[inline(always)]
fn sqr_2(a: [u64; 2]) -> [u64; 4] {
    let p00 = u128::from(a[0]) * u128::from(a[0]);
    let p01 = u128::from(a[0]) * u128::from(a[1]);
    let p11 = u128::from(a[1]) * u128::from(a[1]);
    // `2 p01` has 129 bits: its low limb and the two above it.
    let (d0, d1, d2) = (
        (p01 as u64) << 1,
        ((p01 >> 63) as u64) | (((p01 >> 64) as u64) << 1),
        (p01 >> 127) as u64,
    );
    let c1 = (p00 >> 64) + u128::from(d0);
    let c2 = (c1 >> 64) + u128::from(d1) + u128::from(p11 as u64);
    let c3 = (c2 >> 64) + u128::from(d2) + (p11 >> 64);
    [p00 as u64, c1 as u64, c2 as u64, c3 as u64]
}

impl Registers for Mont2 {
    type V = [u64; 2];

    #[inline(always)]
    fn sqr(&self, a: [u64; 2]) -> [u64; 2] {
        self.redc(sqr_2(a))
    }

    #[inline(always)]
    fn mul(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        self.redc(mul_2(a, b))
    }
}

// Three to eight limbs in registers

/// Montgomery's arithmetic modulo an odd `m` of `N` limbs, `R = B^N`, with the residues fully
/// reduced, below `m`, in arrays that the compiler keeps in registers: each product is reduced as
/// it is formed, a row of the product then a row of the reduction for each limb (the coarsely
/// integrated operand scanning of Koç, Acar and Kaliski), unrolled.
#[derive(Clone, Copy)]
pub(in super::super) struct MontN<const N: usize> {
    /// The modulus, odd, the low limb first.
    pub(in super::super) m: [u64; N],
    /// `-1/m mod B`.
    minv: u64,
}

impl<const N: usize> MontN<N> {
    /// The arithmetic modulo the odd `m`.
    #[inline(always)]
    pub(in super::super) fn new(m: [u64; N]) -> Self {
        MontN {
            m,
            minv: binvert_limb(m[0]).wrapping_neg(),
        }
    }

    /// `x` from its residue `x R mod m`.
    #[inline(always)]
    pub(in super::super) fn plain(&self, a: [u64; N]) -> [u64; N] {
        let mut one = [0; N];
        one[0] = 1;
        self.mul(a, one)
    }
}

impl<const N: usize> Registers for MontN<N> {
    type V = [u64; N];

    #[inline(always)]
    fn sqr(&self, a: [u64; N]) -> [u64; N] {
        self.mul(a, a)
    }

    #[inline(always)]
    fn mul(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        let m = &self.m;
        let mut t = [0u64; N];
        let mut top = 0u64;
        for &bi in b.iter() {
            // `t += a b_i`, into `N + 2` limbs.
            let mut c = 0u64;
            for j in 0..N {
                let p = u128::from(a[j]) * u128::from(bi) + u128::from(t[j]) + u128::from(c);
                t[j] = p as u64;
                c = (p >> 64) as u64;
            }
            let s = u128::from(top) + u128::from(c);
            let (tn, tn1) = (s as u64, (s >> 64) as u64);
            // `t += q m`, zero at limb 0, shifted down a limb.
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
        // Below `2m`: one subtraction when it is at least `m`.
        let mut d = [0u64; N];
        let mut borrow = false;
        for j in 0..N {
            let (x, b1) = t[j].overflowing_sub(m[j]);
            let (x, b2) = x.overflowing_sub(u64::from(borrow));
            d[j] = x;
            borrow = b1 | b2;
        }
        if top != 0 || !borrow { d } else { t }
    }
}
