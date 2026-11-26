//! `GF(2¹²⁸)` and GHASH by `PMULL` on aarch64, where the target enables it (`aes`, as Apple's
//! aarch64 does): products of 64-bit halves by `vmull_p64` in vector registers, Karatsuba's three for
//! a product of 128 bits, and the reduction by `x¹²⁸ = x⁷ + x² + x + 1` as two more products by
//! `0x87`: the top word times `0x87` folds onto the two words below it, and the word above the low
//! half, so made, once more onto the low half.
//!
//! GHASH takes eight blocks at a time, then four (Gueron and Kounavis's aggregated reduction, as OpenSSL and
//! RustCrypto's polyval do): `Y ← ((Y + X₁)H⁸ + X₂H⁷ + … + X₈H)`, the eight products of 256 bits
//! summed unreduced and reduced once, with `H²…H⁸` made once per call. A block of GCM's
//! bit-reflected order becomes the polynomial in the natural order by reversing the bits of each of
//! its bytes (`rbit`), the byte order being the order of the words in a register already.
//!
//! Every function here is `unsafe` only for the intrinsics, which the target enables at compile
//! time (the module is compiled only then).

use core::arch::aarch64::{
    uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64, vld1q_u8, vmull_high_p64,
    vmull_p64, vrbitq_u8, vreinterpretq_p64_u64, vreinterpretq_u64_p128, vreinterpretq_u64_u8,
};

/// `x¹²⁸ mod (x¹²⁸ + x⁷ + x² + x + 1)`.
const R: u64 = 0x87;

/// The vector of the polynomial `a`, its low word in lane 0.
#[inline(always)]
fn vec(a: u128) -> uint64x2_t {
    // SAFETY: a plain reinterpretation of 16 bytes.
    unsafe { core::mem::transmute::<u128, uint64x2_t>(a) }
}

/// The polynomial of the vector `v`.
#[inline(always)]
fn poly(v: uint64x2_t) -> u128 {
    // SAFETY: a plain reinterpretation of 16 bytes.
    unsafe { core::mem::transmute::<uint64x2_t, u128>(v) }
}

/// The carry-less product of the low lanes.
#[inline(always)]
fn clmul_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // SAFETY: PMULL is enabled at compile time for this module.
    unsafe { vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b))) }
}

/// The carry-less product of the high lanes.
#[inline(always)]
fn clmul_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // SAFETY: as above.
    unsafe {
        vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }
}

/// The sum of the two lanes, in lane 0 (and lane 1).
#[inline(always)]
fn fold_lanes(a: uint64x2_t) -> uint64x2_t {
    // SAFETY: lane shuffles and an exclusive or.
    unsafe { veorq_u64(a, vextq_u64::<1>(a, a)) }
}

/// The 256-bit product, unreduced, as its three Karatsuba parts: low, middle, high.
#[derive(Clone, Copy)]
struct Wide {
    lo: uint64x2_t,
    mid: uint64x2_t,
    hi: uint64x2_t,
}

impl Wide {
    /// `a b`, with `bx` the sum of `b`'s lanes (in lane 0).
    #[inline(always)]
    fn of(a: uint64x2_t, b: uint64x2_t, bx: uint64x2_t) -> Wide {
        Wide {
            lo: clmul_lo(a, b),
            mid: clmul_lo(fold_lanes(a), bx),
            hi: clmul_hi(a, b),
        }
    }

    #[inline(always)]
    fn add(self, o: Wide) -> Wide {
        // SAFETY: exclusive ors.
        unsafe {
            Wide {
                lo: veorq_u64(self.lo, o.lo),
                mid: veorq_u64(self.mid, o.mid),
                hi: veorq_u64(self.hi, o.hi),
            }
        }
    }

    /// The product modulo `x¹²⁸ + x⁷ + x² + x + 1`.
    #[inline(always)]
    fn reduce(self) -> uint64x2_t {
        // SAFETY: lane shuffles, exclusive ors and PMULL, enabled for this module.
        unsafe {
            let zero = vdupq_n_u64(0);
            let r = vdupq_n_u64(R);
            // The middle part of Karatsuba's: lo + hi + mid, at the word above the low one.
            let mid = veorq_u64(self.mid, veorq_u64(self.lo, self.hi));
            // Words (p0, p1) and (p2, p3).
            let mut x = veorq_u64(self.lo, vextq_u64::<1>(zero, mid));
            let mut y = veorq_u64(self.hi, vextq_u64::<1>(mid, zero));
            // p3 x¹⁹² = p3 · 0x87 at word 1: its low word onto p1, its high one onto p2.
            let t = clmul_hi(y, r);
            x = veorq_u64(x, vextq_u64::<1>(zero, t));
            y = veorq_u64(y, vextq_u64::<1>(t, zero));
            // p2 x¹²⁸ = p2 · 0x87 onto (p0, p1).
            veorq_u64(x, clmul_lo(y, r))
        }
    }
}

/// `a b` in `GF(2¹²⁸)`, the polynomials in the natural order.
#[inline(always)]
pub(super) fn mul(a: u128, b: u128) -> u128 {
    let (a, b) = (vec(a), vec(b));
    poly(Wide::of(a, b, fold_lanes(b)).reduce())
}

/// `a²`: the two halves squared, and reduced. The cross terms of a square cancel, so its middle
/// part is zero: as `reduce` adds the low and high parts to the middle one, it is given their sum.
#[inline(always)]
pub(super) fn sqr(a: u128) -> u128 {
    let a = vec(a);
    let (lo, hi) = (clmul_lo(a, a), clmul_hi(a, a));
    // SAFETY: an exclusive or.
    let mid = unsafe { veorq_u64(lo, hi) };
    poly(Wide { lo, mid, hi }.reduce())
}

/// The block at `p` (16 bytes) as the polynomial in the natural order.
///
/// # Safety
///
/// `p` must point to 16 readable bytes.
#[inline(always)]
unsafe fn block(p: *const u8) -> uint64x2_t {
    // SAFETY: the caller's; `vld1q_u8` has no alignment requirement.
    unsafe { vreinterpretq_u64_u8(vrbitq_u8(vld1q_u8(p))) }
}

/// GHASH's steps `Y ← (Y + Xᵢ) H` over the whole blocks of `x`, from `y`, in the natural order.
pub(super) fn ghash(y: u128, h: u128, x: &[u8]) -> u128 {
    debug_assert!(x.len().is_multiple_of(16));
    let n = x.len() / 16;
    let p = x.as_ptr();
    let hv = vec(h);
    let hx = fold_lanes(hv);
    let mut y = vec(y);
    let mut i = 0;
    // From eight blocks on: eight blocks a reduction, then four for what is left of four or more.
    // Measured on an Apple M4 Max (1 to 16 blocks, the bound from 1 to 12 alternated): with the
    // key's powers in the thread's cache they pay from four blocks (0.56–0.65 times the time of
    // the blocks one by one at 4 to 7, 0.31–0.41 at 8 to 11); made anew, as when two keys
    // alternate, they cost 1.5–2.3 times that time at 4 to 7 blocks and 1.06–1.18 at 8 to 11. A
    // lower bound would slow the alternating keys, a higher one the single key: eight stays.
    if n >= 8 {
        // H, H², …, H⁸ and the sums of their lanes; the powers of the last key a thread used are
        // kept, so that the messages of one key make them once.
        let pw = powers(h);
        let pw: [uint64x2_t; 8] = core::array::from_fn(|k| vec(pw[k]));
        let px: [uint64x2_t; 8] = core::array::from_fn(|k| fold_lanes(pw[k]));
        while i + 8 <= n {
            // SAFETY: blocks i to i + 7 are within `x`.
            let xs: [uint64x2_t; 8] =
                core::array::from_fn(|k| unsafe { block(p.add(16 * (i + k))) });
            // SAFETY: an exclusive or.
            let first = unsafe { veorq_u64(y, xs[0]) };
            let mut acc = Wide::of(first, pw[7], px[7]);
            for k in 1..8 {
                acc = acc.add(Wide::of(xs[k], pw[7 - k], px[7 - k]));
            }
            y = acc.reduce();
            i += 8;
        }
        while i + 4 <= n {
            // SAFETY: blocks i to i + 3 are within `x`.
            let xs: [uint64x2_t; 4] =
                core::array::from_fn(|k| unsafe { block(p.add(16 * (i + k))) });
            // SAFETY: an exclusive or.
            let first = unsafe { veorq_u64(y, xs[0]) };
            let mut acc = Wide::of(first, pw[3], px[3]);
            for k in 1..4 {
                acc = acc.add(Wide::of(xs[k], pw[3 - k], px[3 - k]));
            }
            y = acc.reduce();
            i += 4;
        }
    }
    while i < n {
        // SAFETY: block i is within `x`; an exclusive or.
        let s = unsafe { veorq_u64(y, block(p.add(16 * i))) };
        y = Wide::of(s, hv, hx).reduce();
        i += 1;
    }
    poly(y)
}

std::thread_local! {
    /// The last key of this thread and its powers `H, H², …, H⁸`: a cache of one entry.
    static POWERS: core::cell::Cell<(u128, [u128; 8])> = const { core::cell::Cell::new((0, [0; 8])) };
}

/// `H, H², …, H⁸`, from the thread's cache when `h` is its last key.
fn powers(h: u128) -> [u128; 8] {
    let cached = POWERS.with(|c| c.get());
    if cached.0 == h && h != 0 {
        return cached.1;
    }
    let mut pw = [h; 8];
    for k in 1..8 {
        pw[k] = mul(pw[k - 1], h);
    }
    POWERS.with(|c| c.set((h, pw)));
    pw
}
