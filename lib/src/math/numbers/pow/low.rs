//! Low halves of products: the `n` least significant limbs of a product of two numbers of `n`
//! limbs, which is the product modulo `B^n` (`B = 2^64`), for the powers modulo a power of two and
//! for the quotients of Montgomery's reduction by products (Hensel's quotient, see
//! [`super::montgomery`]).

use super::super::mul::limbs::{add_n, mul, mul_itch};
use super::kernels::row;
use super::window::{Registers, Ring};

/// From this many limbs, a low half is computed by divide and conquer, [`mullo_dc`]: below, by
/// rows of decreasing length, half the products of the whole product.
pub(super) const MULLO_DC: usize = 40;

/// The limbs of scratch space that [`mullo`] needs for operands of `n` limbs.
pub(in super::super) fn mullo_itch(n: usize) -> usize {
    if n < MULLO_DC {
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
    s: *mut u64,
) {
    // SAFETY: forwarded.
    unsafe {
        if n < MULLO_DC {
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
unsafe fn mullo_dc(r: *mut u64, x: *const u64, y: *const u64, n: usize, s: *mut u64) {
    let low = split(n);
    let high = n - low;
    // SAFETY: as the caller guarantees: the whole product of `2 low >= n` limbs at `s`, the low
    // halves of `high` limbs at `s + n`, below `s + 2n`, and the scratch of both after `2n`.
    unsafe {
        let rest = s.add(2 * n);
        mul(s, x, low, y, low, rest);
        core::ptr::copy_nonoverlapping(s, r, n);
        mullo(s.add(n), x.add(low), y, high, rest);
        add_n(r.add(low), r.add(low), s.add(n), high);
        mullo(s.add(n), x, y.add(low), high, rest);
        add_n(r.add(low), r.add(low), s.add(n), high);
    }
}

/// The ring of residues modulo `B^n`, by the low halves of products.
pub(super) struct Low {
    /// The limbs of a residue.
    pub(super) n: usize,
}

impl Low {
    /// The limbs of scratch space that the ring's operations need.
    pub(super) fn itch(n: usize) -> usize {
        n + mullo_itch(n)
    }
}

impl Ring for Low {
    fn n(&self) -> usize {
        self.n
    }

    unsafe fn sqr(&self, r: *mut u64, a: *const u64, t: *mut u64) {
        // SAFETY: the low half goes to the scratch, then over `r`.
        unsafe {
            mullo(t, a, a, self.n, t.add(self.n));
            core::ptr::copy_nonoverlapping(t, r, self.n);
        }
    }

    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64, t: *mut u64) {
        // SAFETY: as in `sqr`.
        unsafe {
            mullo(t, a, b, self.n, t.add(self.n));
            core::ptr::copy_nonoverlapping(t, r, self.n);
        }
    }
}

/// The ring of residues modulo `B^2` in registers, by wrapping products of `u128`s.
pub(super) struct Low2;

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
pub(super) struct LowN<const N: usize>;

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
