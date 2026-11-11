//! Low halves of products: the `n` least significant limbs of a product of two numbers of `n`
//! limbs, which is the product modulo `B^n` (`B = 2^64`), for the powers modulo a power of two and
//! for the quotients of Montgomery's reduction by products (Hensel's quotient, see
//! [`super::montgomery`]).

use super::super::mul::limbs::{add_n, mul, mul_itch, sqr, sqr_itch};
use super::super::scratch::{Scratch, need};
use super::kernels::row;
use super::montgomery::{lshift, registers};
use super::window::{Registers, Ring};

/// From this many limbs, a low half is computed by divide and conquer, [`mullo_dc`]: below, by
/// rows of decreasing length, half the products of the whole product, or by the kernels of the
/// low halves in registers, which take nine to sixteen limbs where the target has them, and then
/// divide and conquer from eighteen, the halves going to the kernels.
pub(super) const MULLO_DC: usize = if registers::has_low(16) { 18 } else { 40 };

/// The limbs of scratch space that [`mullo`] needs for operands of `n` limbs.
pub(in super::super) fn mullo_itch(n: usize) -> usize {
    if n < MULLO_DC || registers::has_low(n) {
        0
    } else {
        let low = split(n);
        2 * n + mul_itch(low, low).max(mullo_itch(n - low))
    }
}

/// The limbs of the low part of a divide-and-conquer low half of `n` limbs, whose whole product is
/// computed: what is left for the two low halves of the high parts is about the share of `n` that
/// makes their cost as small as the whole product's method allows (GMP's ratios: a half below
/// Toom-3, 0.69 under Toom-3, 0.78 from Toom-4).
#[inline]
fn split(n: usize) -> usize {
    // Up to 32 limbs with the kernels of the low halves, the high part of half the limbs, which they
    // take.
    if registers::has_low(16) && n <= 32 {
        return n - n / 2;
    }
    // The multiplication's thresholds of Toom-3 and Toom-4, 120 and 224 limbs, over the shares.
    let high = if n < 120 * 36 / 25 {
        n * 11 / 36
    } else if n < 224 * 40 / 31 {
        n * 9 / 40
    } else {
        n * 7 / 39
    };
    n - high
}

/// Sets the `n` limbs at `r` to the low `n` limbs of the product of the `n` limbs at `x` and the
/// `n` at `y`.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` and `y` must be valid for reads, those of `r` for writes, apart from
/// both, and the [`mullo_itch`]`(n)` limbs at `s` for reads and writes, apart from all.
pub(in super::super) unsafe fn mullo(
    r: *mut u64,
    x: *const u64,
    y: *const u64,
    n: usize,
    s: Scratch,
) {
    need!(s, mullo_itch(n));
    // SAFETY: forwarded; the kernel writes `r` after it has read `x` and `y`.
    unsafe {
        if registers::has_low(n) {
            registers::low(r, x, y, n);
        } else if n < MULLO_DC {
            mullo_basecase(r, x, y, n);
        } else {
            mullo_dc(r, x, y, n, s);
        }
    }
}

/// [`mullo`] by rows: row `j` adds `x` times `y_j` at position `j`, its `n - j` limbs that fall
/// below `n`.
///
/// # Safety
///
/// As for [`mullo`].
unsafe fn mullo_basecase(r: *mut u64, x: *const u64, y: *const u64, n: usize) {
    // SAFETY: as the caller guarantees; row `j` reads and writes limbs `j..n` of `r`.
    unsafe {
        core::ptr::write_bytes(r, 0, n);
        for j in 0..n {
            row(r.add(j), x, n - j, *y.add(j));
        }
    }
}

/// [`mullo`] by divide and conquer, after GMP's `mpn_dc_mullo_n`: with `x = x1 B^l + x0` and
/// `y = y1 B^l + y0` for `l = split(n)`, the low half is that of `x0 y0`, computed whole, plus the
/// low halves of `x1 y0` and `x0 y1` at `B^l`.
///
/// # Safety
///
/// As for [`mullo`], with `n >= 2`.
unsafe fn mullo_dc(r: *mut u64, x: *const u64, y: *const u64, n: usize, s: Scratch) {
    need!(s, mullo_itch(n));
    let low = split(n);
    let high = n - low;
    // SAFETY: as the caller guarantees: the whole product of `2 low >= n` limbs at `s`, the low
    // halves of `high` limbs at `s + n`, below `s + 2n`, and the scratch of both after `2n`.
    unsafe {
        let (s, rest) = s.split(2 * n);
        mul(s, x, low, y, low, rest);
        core::ptr::copy_nonoverlapping(s, r, n);
        mullo(s.add(n), x.add(low), y, high, rest);
        add_n(r.add(low), r.add(low), s.add(n), high);
        mullo(s.add(n), x, y.add(low), high, rest);
        add_n(r.add(low), r.add(low), s.add(n), high);
    }
}

/// From this many limbs, a low half of a square is computed by divide and conquer, [`sqrlo_dc`]:
/// below, by the rows of the products of different limbs, doubled, a quarter of the products of the
/// whole square, or by the kernels, as for [`MULLO_DC`].
pub(super) const SQRLO_DC: usize = MULLO_DC;

/// The limbs of scratch space that [`sqrlo`] needs for an operand of `n` limbs.
pub(in super::super) fn sqrlo_itch(n: usize) -> usize {
    if n < SQRLO_DC || registers::has_low(n) {
        0
    } else {
        let low = split(n);
        2 * n + sqr_itch(low).max(mullo_itch(n - low))
    }
}

/// Sets the `n` limbs at `r` to the low `n` limbs of the square of the `n` limbs at `x`, as GMP's
/// `mpn_sqrlo` does: about half the products of [`mullo`], as the products of two different limbs
/// come in pairs.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads, those of `r` for writes, apart from `x`,
/// and the [`sqrlo_itch`]`(n)` limbs at `s` for reads and writes, apart from both.
pub(in super::super) unsafe fn sqrlo(r: *mut u64, x: *const u64, n: usize, s: Scratch) {
    need!(s, sqrlo_itch(n));
    // SAFETY: forwarded; the kernel writes `r` after it has read `x`.
    unsafe {
        if registers::has_low(n) {
            registers::low(r, x, core::ptr::null(), n);
        } else if n < SQRLO_DC {
            sqrlo_basecase(r, x, n);
        } else {
            sqrlo_dc(r, x, n, s);
        }
    }
}

/// [`sqrlo`] by rows: row `i` adds `x_i` times the limbs `x_(i+1)..x_(n-1-i)` at position `2i + 1`,
/// the products of two different limbs that fall below `n`, once; then the sum is doubled, and the
/// squares `x_i^2` added at the even positions.
///
/// # Safety
///
/// As for [`sqrlo`].
unsafe fn sqrlo_basecase(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: as the caller guarantees; row `i` reads and writes limbs `2i + 1..n` of `r`, its carry
    // out, which belongs at `n`, dropped.
    unsafe {
        core::ptr::write_bytes(r, 0, n);
        let mut i = 0;
        while 2 * i + 1 < n {
            row(r.add(2 * i + 1), x.add(i + 1), n - 1 - 2 * i, *x.add(i));
            i += 1;
        }
        lshift(r, r, n, 1);
        // The squares, the low limb at `2i` and the high one above it, and the carry up to `n`.
        let mut carry = 0u64;
        for k in 0..n {
            let xi = *x.add(k / 2);
            let sq = u128::from(xi) * u128::from(xi);
            let d = if k % 2 == 0 {
                sq as u64
            } else {
                (sq >> 64) as u64
            };
            let t = u128::from(*r.add(k)) + u128::from(d) + u128::from(carry);
            *r.add(k) = t as u64;
            carry = (t >> 64) as u64;
        }
    }
}

/// [`sqrlo`] by divide and conquer: with `x = x1 B^l + x0` for `l = split(n)`, the low half is that
/// of `x0^2`, computed whole, plus twice the low half of `x0 x1` at `B^l`.
///
/// # Safety
///
/// As for [`sqrlo`], with `n >= 2`.
unsafe fn sqrlo_dc(r: *mut u64, x: *const u64, n: usize, s: Scratch) {
    need!(s, sqrlo_itch(n));
    let low = split(n);
    let high = n - low;
    // SAFETY: as the caller guarantees: the whole square of `2 low >= n` limbs at `s`, of which the
    // low `n` are copied out before the low half of `high` limbs goes to `s + n`, below `s + 2n`,
    // and the scratch of both after `2n`.
    unsafe {
        let (s, rest) = s.split(2 * n);
        sqr(s, x, low, rest);
        core::ptr::copy_nonoverlapping(s, r, n);
        mullo(s.add(n), x, x.add(low), high, rest);
        lshift(s.add(n), s.add(n), high, 1);
        add_n(r.add(low), r.add(low), s.add(n), high);
    }
}

/// The ring of residues modulo `B^n`, by the low halves of products.
pub(in crate::math::numbers) struct Low {
    /// The limbs of a residue.
    pub(in crate::math::numbers) n: usize,
}

impl Low {
    /// The limbs of scratch space that the ring's operations need.
    pub(in crate::math::numbers) fn itch(n: usize) -> usize {
        n + mullo_itch(n).max(sqrlo_itch(n))
    }
}

impl Ring for Low {
    fn n(&self) -> usize {
        self.n
    }

    unsafe fn sqr(&self, r: *mut u64, a: *const u64, t: Scratch) {
        need!(t, Low::itch(self.n));
        // SAFETY: the low half goes to the scratch, then over `r`; or the kernel, which reads `a`
        // before it writes `r`.
        unsafe {
            if registers::has_low(self.n) {
                registers::low(r, a, core::ptr::null(), self.n);
                return;
            }
            let (p, t) = t.split(self.n);
            sqrlo(p, a, self.n, t);
            core::ptr::copy_nonoverlapping(p, r, self.n);
        }
    }

    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64, t: Scratch) {
        need!(t, Low::itch(self.n));
        // SAFETY: as in `sqr`.
        unsafe {
            if registers::has_low(self.n) {
                registers::low(r, a, b, self.n);
                return;
            }
            let (p, t) = t.split(self.n);
            mullo(p, a, b, self.n, t);
            core::ptr::copy_nonoverlapping(p, r, self.n);
        }
    }
}

/// The ring of residues modulo `B^2` in registers, by wrapping products of `u128`s.
pub(in crate::math::numbers) struct Low2;

impl Registers for Low2 {
    type V = u128;

    #[inline(always)]
    fn sqr(&self, a: u128) -> u128 {
        a.wrapping_mul(a)
    }

    #[inline(always)]
    fn mul(&self, a: u128, b: u128) -> u128 {
        a.wrapping_mul(b)
    }
}

/// The ring of residues modulo `B^N` in registers, for `N` from 3 to 8: the low `N` limbs of each
/// product, a row of the product for each limb of one factor, cut at limb `N`, unrolled.
pub(in crate::math::numbers) struct LowN<const N: usize>;

impl<const N: usize> Registers for LowN<N> {
    type V = [u64; N];

    #[inline(always)]
    fn sqr(&self, a: [u64; N]) -> [u64; N] {
        self.mul(a, a)
    }

    #[inline(always)]
    fn mul(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        let mut r = [0u64; N];
        for i in 0..N {
            let mut carry = 0u64;
            for j in 0..N - i {
                let t =
                    u128::from(a[i]) * u128::from(b[j]) + u128::from(r[i + j]) + u128::from(carry);
                r[i + j] = t as u64;
                carry = (t >> 64) as u64;
            }
        }
        r
    }
}
