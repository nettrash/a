//! Exponentiation by a sliding window, left to right, after GMP's `mpn_powm`: the exponent's bits
//! from the top, a square for each, and for each window of up to `k` bits that starts and ends with
//! a one, a product by the window's odd power of the base from a table of `2^(k - 1)` of them.
//!
//! The arithmetic is a [`Ring`] on limbs in memory, or a [`Registers`] on values in registers;
//! the exponent is held as limbs, the least significant first.

/// The arithmetic of a ring of residues of `n` limbs in memory: Montgomery's modulo an odd
/// number, or the low limbs of products modulo a power of two.
pub(in super::super) trait Ring {
    /// The limbs of a residue.
    fn n(&self) -> usize;

    /// Sets the `n` limbs at `r` to the square of the `n` limbs at `a`, with the scratch at `t`.
    ///
    /// # Safety
    ///
    /// The limbs must be valid; `r` may be `a`; the scratch is that of the ring.
    unsafe fn sqr(&self, r: *mut u64, a: *const u64, t: *mut u64);

    /// Sets the `n` limbs at `r` to the product of the `n` limbs at `a` and `b`, with the scratch
    /// at `t`.
    ///
    /// # Safety
    ///
    /// The limbs must be valid; `r` may be `a` or `b`; the scratch is that of the ring.
    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64, t: *mut u64);
}

/// The arithmetic of a ring of residues held in registers.
pub(in super::super) trait Registers {
    /// A residue.
    type V: Copy;

    /// The square of `a`.
    fn sqr(&self, a: Self::V) -> Self::V;

    /// The product of `a` and `b`.
    fn mul(&self, a: Self::V, b: Self::V) -> Self::V;
}

/// The window size for an exponent of `bits` bits: the `k` that minimizes the products, about
/// `2^(k - 1) + bits / (k + 1)`, GMP's table, never more than `bits`.
#[inline]
pub(super) fn window(bits: usize) -> u32 {
    const LIMITS: [usize; 9] = [7, 25, 81, 241, 673, 1793, 4609, 11521, 28161];
    let mut k = 1;
    for limit in LIMITS {
        if bits > limit {
            k += 1;
        }
    }
    k
}

/// The longest window of [`power_in_registers`], whose table it keeps on the stack.
const MAX_REGISTER_WINDOW: u32 = 6;

/// Bit `i` of the exponent `e`, counted from 1: `getbit(e, 1)` is the lowest.
///
/// # Safety
///
/// The limb holding the bit must be valid for reads.
#[inline(always)]
unsafe fn bit(e: *const u64, i: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { (*e.add((i - 1) / 64) >> ((i - 1) % 64)) & 1 != 0 }
}

/// The `k` bits of the exponent `e` below bit `i` (counted from 1), or the `i` lowest ones when
/// there are no more, as a number.
///
/// # Safety
///
/// `1 <= k <= 63`, and the limbs holding the bits valid for reads.
#[inline(always)]
pub(super) unsafe fn bits(e: *const u64, i: usize, k: u32) -> u64 {
    let k = k as usize;
    // SAFETY: as the caller guarantees: limbs `(i - k) / 64` and the one above it, which holds
    // bit `i - 1` when the bits straddle two limbs.
    unsafe {
        if i <= k {
            return *e & ((1 << i) - 1);
        }
        let low = i - k;
        let (l, s) = (low / 64, low % 64);
        let mut r = *e.add(l) >> s;
        if 64 - s < k {
            r |= *e.add(l + 1) << (64 - s);
        }
        r & ((1 << k) - 1)
    }
}

/// Sets the `n` limbs at `r` to `b^e` in `ring`, for the exponent of `ebits` bits at `e`, at least
/// 1, and the residue `b` of `n` limbs at `table`, which is overwritten by the table of the
/// window's odd powers, `2^(window(ebits) - 1)` residues of `stride` limbs each. `t` is the ring's
/// scratch.
///
/// # Safety
///
/// The limbs must be valid for reads and writes, apart, the exponent's top bit set; `stride >= n`.
pub(super) unsafe fn power<R: Ring>(
    ring: &R,
    r: *mut u64,
    e: *const u64,
    ebits: usize,
    table: *mut u64,
    stride: usize,
    t: *mut u64,
) {
    let n = ring.n();
    let k = window(ebits);
    // SAFETY: as the caller guarantees: the table holds `2^(k - 1)` residues, the exponent its
    // `ebits` bits, and every bit read is at most bit `ebits`.
    unsafe {
        if k > 1 {
            // `b^2` in `r`, then `b^3, b^5, ...` each from the one before.
            ring.sqr(r, table, t);
            for i in 1..1usize << (k - 1) {
                ring.mul(table.add(i * stride), table.add((i - 1) * stride), r, t);
            }
        }
        // The top window: `k` bits, its top one set, without its low zeros.
        let mut i = ebits - k as usize;
        let w = bits(e, ebits, k);
        let zeros = w.trailing_zeros();
        i += zeros as usize;
        core::ptr::copy_nonoverlapping(table.add((w >> zeros >> 1) as usize * stride), r, n);
        while i != 0 {
            if !bit(e, i) {
                ring.sqr(r, r, t);
                i -= 1;
                continue;
            }
            // The next window starts with this one: up to `k` bits, ending with a one.
            let w = bits(e, i, k);
            let zeros = w.trailing_zeros();
            let len = k.min(i as u32) - zeros;
            i -= len as usize;
            for _ in 0..len {
                ring.sqr(r, r, t);
            }
            ring.mul(r, r, table.add((w >> zeros >> 1) as usize * stride), t);
        }
    }
}

/// `b^e` in `ring`, for the exponent of `ebits` bits at `e`, at least 1: as [`power`] does, with
/// the residues in registers and a window of at most [`MAX_REGISTER_WINDOW`] bits.
///
/// # Safety
///
/// The exponent's limbs must be valid for reads, its top bit set.
#[inline(always)]
pub(super) unsafe fn power_in_registers<R: Registers>(
    ring: &R,
    b: R::V,
    e: *const u64,
    ebits: usize,
) -> R::V {
    let k = window(ebits).min(MAX_REGISTER_WINDOW);
    let mut table = [b; 1 << (MAX_REGISTER_WINDOW - 1)];
    if k > 1 {
        let b2 = ring.sqr(b);
        for i in 1..1usize << (k - 1) {
            table[i] = ring.mul(table[i - 1], b2);
        }
    }
    // SAFETY: as the caller guarantees: every bit read is at most bit `ebits`.
    unsafe {
        let mut i = ebits - k as usize;
        let w = bits(e, ebits, k);
        let zeros = w.trailing_zeros();
        i += zeros as usize;
        let mut r = table[(w >> zeros >> 1) as usize];
        while i != 0 {
            if !bit(e, i) {
                r = ring.sqr(r);
                i -= 1;
                continue;
            }
            let w = bits(e, i, k);
            let zeros = w.trailing_zeros();
            let len = k.min(i as u32) - zeros;
            i -= len as usize;
            for _ in 0..len {
                r = ring.sqr(r);
            }
            r = ring.mul(r, table[(w >> zeros >> 1) as usize]);
        }
        r
    }
}
