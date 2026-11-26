//! The NEON kernels of the sieve, through the intrinsics of `core::arch`, for the 64-bit
//! little-endian aarch64 targets of the crate's assembly with NEON enabled; the module `portable`
//! has the same functions in plain Rust for the other targets and for Miri, and gives the same
//! results.
//!
//! - [`root_hits`]: the primes whose roots a position of a block hits, eight sixteen-bit lanes at
//!   a time, their lanes found from a mask narrowed to a byte each rather than tested one by one.
//! - [`scan`]: the bytes of a block with their top bit set, sixty-four at a time, each chunk set
//!   back to the initial value in the same pass.
//! - [`run_few`]: the medium primes with one to three sure hits a block, four at a time: their
//!   positions moved in vector registers and the bytes added through their lanes, in fewer steps
//!   than the scalar loop, whose bookkeeping outweighs those few hits.
//! - [`fill_run`]: the bucket entries of the large primes of a run of the same number of sure
//!   hits, four primes at a time: the buckets and the entries of their roots in vector registers,
//!   the writes through their lanes.
//! - [`push`]: an entry written to a bucket of the large primes without the checks of its bounds,
//!   which had been a fifth of the instructions of the most frequent step of the bucket sieve (the
//!   portable one checks them; the sizes of number that reach the buckets are beyond Miri).

use core::arch::aarch64::{
    uint16x8_t, uint32x4_t, vaddq_u16, vaddq_u32, vaddvq_u16, vandq_u16, vandq_u32, vbslq_u32,
    vceqq_u16, vceqzq_u16, vcltq_u32, vcltzq_s8, vdupq_n_u8, vdupq_n_u16, vdupq_n_u32,
    vget_lane_u64, vget_low_u16, vgetq_lane_u32, vld1_u8, vld1_u16, vld1q_u8, vld1q_u16, vld1q_u32,
    vmaxvq_u8, vminq_u32, vmlsq_u16, vmovl_u8, vmovl_u16, vmovn_u32, vmull_high_u16, vmull_u16,
    vorrq_u8, vorrq_u16, vorrq_u32, vreinterpret_u64_u8, vreinterpretq_s8_u8, vreinterpretq_u16_u8,
    vreinterpretq_u16_u32, vshlq_n_u32, vshrn_n_u16, vshrq_n_u32, vst1_u16, vst1q_u8, vsubq_u16,
    vsubq_u32, vuzp2q_u16,
};

/// The lanes of the primes `p` (and `m = ⌊2^16/p⌋`) one of whose roots `r1`, `r2` the offset in
/// `ov` hits: all ones where it does.
#[inline(always)]
unsafe fn hit_lanes(
    ov: uint16x8_t,
    p: uint16x8_t,
    m: uint16x8_t,
    r1: uint16x8_t,
    r2: uint16x8_t,
) -> uint16x8_t {
    // SAFETY: NEON is enabled on the targets this module is compiled for.
    unsafe {
        let q = vuzp2q_u16(
            vreinterpretq_u16_u32(vmull_u16(vget_low_u16(ov), vget_low_u16(m))),
            vreinterpretq_u16_u32(vmull_high_u16(ov, m)),
        );
        let x = vmlsq_u16(ov, q, p);
        let d1 = vsubq_u16(x, r1);
        let d2 = vsubq_u16(x, r2);
        vorrq_u16(
            vorrq_u16(vceqzq_u16(d1), vceqq_u16(d1, p)),
            vorrq_u16(vceqzq_u16(d2), vceqq_u16(d2, p)),
        )
    }
}

/// The sum of the logarithms `lg[k]` of the primes `p[k]` whose roots the offset `o` hits, as
/// [`root_hits`] finds them, without a branch.
#[inline(always)]
pub(super) fn root_sum(o: u16, p: &[u16], m: &[u16], r1: &[u16], r2: &[u16], lg: &[u8]) -> u32 {
    let n = p
        .len()
        .min(m.len())
        .min(r1.len())
        .min(r2.len())
        .min(lg.len());
    let whole = n / 8 * 8;
    let mut sum = 0u32;
    // SAFETY: NEON is enabled on the targets this module is compiled for, and every load reads
    // eight lanes from `i < whole <= n`, within each slice.
    unsafe {
        let ov = vdupq_n_u16(o);
        let mut acc = vdupq_n_u16(0);
        let mut i = 0;
        while i < whole {
            let h = hit_lanes(
                ov,
                vld1q_u16(p.as_ptr().add(i)),
                vld1q_u16(m.as_ptr().add(i)),
                vld1q_u16(r1.as_ptr().add(i)),
                vld1q_u16(r2.as_ptr().add(i)),
            );
            acc = vaddq_u16(acc, vandq_u16(h, vmovl_u8(vld1_u8(lg.as_ptr().add(i)))));
            i += 8;
        }
        sum += u32::from(vaddvq_u16(acc));
    }
    for k in whole..n {
        let x =
            o.wrapping_sub((((u32::from(o) * u32::from(m[k])) >> 16) as u16).wrapping_mul(p[k]));
        let (d1, d2) = (x.wrapping_sub(r1[k]), x.wrapping_sub(r2[k]));
        if d1 == 0 || d1 == p[k] || d2 == 0 || d2 == p[k] {
            sum += u32::from(lg[k]);
        }
    }
    sum
}

/// Calls `f(k)` for each `k` whose prime `p[k]` has a root `r1[k]` or `r2[k]` (below it) that the
/// offset `o` hits modulo the prime, and rarely for others (each found prime is to be checked by a
/// division): `o mod p` is `o − ⌊o·m/2^16⌋·p` or that plus `p`, for `m = ⌊2^16/p⌋`.
#[inline(always)]
pub(super) fn root_hits(
    o: u16,
    p: &[u16],
    m: &[u16],
    r1: &[u16],
    r2: &[u16],
    mut f: impl FnMut(usize),
) {
    let n = p.len().min(m.len()).min(r1.len()).min(r2.len());
    let whole = n / 8 * 8;
    // SAFETY: NEON is enabled on the targets this module is compiled for, and every load reads
    // eight lanes from `i < whole <= n`, within each slice.
    unsafe {
        let ov = vdupq_n_u16(o);
        let mut i = 0;
        while i < whole {
            let h = hit_lanes(
                ov,
                vld1q_u16(p.as_ptr().add(i)),
                vld1q_u16(m.as_ptr().add(i)),
                vld1q_u16(r1.as_ptr().add(i)),
                vld1q_u16(r2.as_ptr().add(i)),
            );
            let mut bits = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(h)));
            while bits != 0 {
                let k = bits.trailing_zeros() / 8;
                f(i + k as usize);
                bits &= !(0xFF << (8 * k));
            }
            i += 8;
        }
    }
    for k in whole..n {
        let x =
            o.wrapping_sub((((u32::from(o) * u32::from(m[k])) >> 16) as u16).wrapping_mul(p[k]));
        let (d1, d2) = (x.wrapping_sub(r1[k]), x.wrapping_sub(r2[k]));
        if d1 == 0 || d1 == p[k] || d2 == 0 || d2 == p[k] {
            f(k);
        }
    }
}

/// Whether [`run_few`] sieves the runs of one to three sure hits.
pub(super) const RUN_FEW: bool = true;

/// Adds `lg` to byte `i` of the block, for `i < block.len()`.
#[inline(always)]
unsafe fn add_byte(block: &mut [u8], i: usize, lg: u8) {
    debug_assert!(i < block.len());
    // SAFETY: the caller makes sure that `i` is within the block.
    unsafe {
        let x = block.get_unchecked_mut(i);
        *x = x.wrapping_add(lg);
    }
}

/// Adds the logarithms `lgs` of the medium primes `ps`, of `SURE` (1 to 3) sure hits a root in a
/// block of `end` bytes, at their positions in the block (from `m1`, `m2`, each below its prime,
/// kept in `b1`, `b2` for the trial division), and moves the positions to the next block; the last
/// hit of each root, if any, is the only one tested, and adds to the guard of 64 bytes after the
/// block when it misses. Four primes at a time; those past the last multiple of four are left to
/// the caller: the number of primes done is returned.
///
/// # Safety
///
/// `block` is `end + 64` bytes long at least, and each of the first `n/4·4` primes (`n` the
/// shortest length of the slices) has its positions in `m1` and `m2` below it and
/// `SURE·p <= end`: the bytes added to are not checked.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_few<const SURE: u32>(
    block: &mut [u8],
    end: u32,
    ps: &[u16],
    lgs: &[u8],
    m1: &mut [u16],
    m2: &mut [u16],
    b1: &mut [u16],
    b2: &mut [u16],
) -> usize {
    let n = ps
        .len()
        .min(lgs.len())
        .min(m1.len())
        .min(m2.len())
        .min(b1.len())
        .min(b2.len());
    let whole = n / 4 * 4;
    debug_assert!(block.len() >= end as usize + 64);
    // SAFETY: NEON is enabled on the targets this module is compiled for; every load and store
    // reads or writes four lanes from `i < whole <= n`, within each slice; and every byte added
    // to is within the block: the positions are below their primes, as the caller makes sure, so
    // the `SURE` sure hits are below `SURE·p <= end`, and the last one is sent to the guard,
    // `end + (r & 63)`, when it is not below `end`.
    unsafe {
        let endv = vdupq_n_u32(end);
        let m63 = vdupq_n_u32(63);
        let mut i = 0;
        while i < whole {
            let p = vmovl_u16(vld1_u16(ps.as_ptr().add(i)));
            let s1 = vld1_u16(m1.as_ptr().add(i));
            let s2 = vld1_u16(m2.as_ptr().add(i));
            vst1_u16(b1.as_mut_ptr().add(i), s1);
            vst1_u16(b2.as_mut_ptr().add(i), s2);
            let mut r1 = vmovl_u16(s1);
            let mut r2 = vmovl_u16(s2);
            let lg = [lgs[i], lgs[i + 1], lgs[i + 2], lgs[i + 3]];
            macro_rules! lanes {
                ($v:expr) => {{
                    let v: uint32x4_t = $v;
                    add_byte(block, vgetq_lane_u32::<0>(v) as usize, lg[0]);
                    add_byte(block, vgetq_lane_u32::<1>(v) as usize, lg[1]);
                    add_byte(block, vgetq_lane_u32::<2>(v) as usize, lg[2]);
                    add_byte(block, vgetq_lane_u32::<3>(v) as usize, lg[3]);
                }};
            }
            for _ in 0..SURE {
                lanes!(r1);
                lanes!(r2);
                r1 = vaddq_u32(r1, p);
                r2 = vaddq_u32(r2, p);
            }
            // The last hits: in the block, or in the guard; the positions moved on past them.
            let in1 = vcltq_u32(r1, endv);
            let in2 = vcltq_u32(r2, endv);
            lanes!(vminq_u32(r1, vaddq_u32(endv, vandq_u32(r1, m63))));
            lanes!(vminq_u32(r2, vaddq_u32(endv, vandq_u32(r2, m63))));
            r1 = vbslq_u32(in1, vaddq_u32(r1, p), r1);
            r2 = vbslq_u32(in2, vaddq_u32(r2, p), r2);
            vst1_u16(m1.as_mut_ptr().add(i), vmovn_u32(vsubq_u32(r1, endv)));
            vst1_u16(m2.as_mut_ptr().add(i), vmovn_u32(vsubq_u32(r2, endv)));
            i += 4;
        }
    }
    whole
}

/// Whether [`fill_run`] writes the buckets of runs of large primes.
pub(super) const FILL_RUN: bool = true;

/// Writes the bucket entries of the large primes `ps` (at indices `first..` of their slice, all of
/// `sure` sure hits in an interval of `last` blocks of `2^16` bytes) from their roots `r1`, `r2`:
/// for each root and hit, the entry `index << 16 | position` in the bucket `block·4 + lane` of
/// `lens` and `buckets`, the lane the prime's index in the factor base modulo 4 (`lane` for the
/// first), as in the scalar loop, and the last step of each root in bucket `last·4 + lane` when it
/// passes the interval. Four primes at a time, in the order of the scalar loop for each bucket;
/// those past the last multiple of four are left to the caller: the number of primes done is
/// returned.
///
/// # Safety
///
/// The buckets have room, as the writes are not checked: `lens` has `(last + 1)·4` counts and
/// `buckets` at least as many buckets, and no bucket gets more than `CAP` entries (each takes at
/// most one entry for each root of each prime of its lane in the slice, at most `SLICE/4` primes
/// as the lanes follow the indices, when the primes are at least a block and the roots below
/// them).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fill_run<const CAP: usize>(
    ps: &[u32],
    r1s: &[u32],
    r2s: &[u32],
    first: u32,
    lane: u32,
    sure: u32,
    last: u32,
    lens: &mut [u32],
    buckets: &mut [[u32; CAP]],
) -> usize {
    let n = ps.len().min(r1s.len()).min(r2s.len());
    let whole = n / 4 * 4;
    debug_assert!(lens.len() == (last as usize + 1) * 4 && lens.len() <= buckets.len());
    // SAFETY: NEON is enabled on the targets this module is compiled for; every load reads four
    // lanes from `i < whole <= n`, within each slice; each bucket index is a block below `last`, or
    // `last` itself for the last steps, times 4 plus a lane below 4, so below `lens.len()`; and each
    // count stays below CAP (`push`), as the caller makes sure: the lane of each prime is its index
    // modulo 4, as in the scalar loop, so that a lane has at most `SLICE/4` primes of the slice,
    // each with at most one entry a root in a bucket.
    unsafe {
        let lanes = [lane & 3, (lane + 1) & 3, (lane + 2) & 3, (lane + 3) & 3];
        let lanev = vld1q_u32(lanes.as_ptr());
        let ramp = [
            first << 16,
            (first + 1) << 16,
            (first + 2) << 16,
            (first + 3) << 16,
        ];
        let mut e = vld1q_u32(ramp.as_ptr());
        let step = vdupq_n_u32(4 << 16);
        let low = vdupq_n_u32(0xFFFF);
        let lastv = vdupq_n_u32(last);
        let mut i = 0;
        while i < whole {
            let p = vld1q_u32(ps.as_ptr().add(i));
            let mut r1 = vld1q_u32(r1s.as_ptr().add(i));
            let mut r2 = vld1q_u32(r2s.as_ptr().add(i));
            macro_rules! lanes {
                ($b:expr, $r:expr) => {{
                    let b: uint32x4_t = vorrq_u32(vshlq_n_u32::<2>($b), lanev);
                    let ent: uint32x4_t = vorrq_u32(e, vandq_u32($r, low));
                    push(
                        lens,
                        buckets,
                        vgetq_lane_u32::<0>(b) as usize,
                        vgetq_lane_u32::<0>(ent),
                    );
                    push(
                        lens,
                        buckets,
                        vgetq_lane_u32::<1>(b) as usize,
                        vgetq_lane_u32::<1>(ent),
                    );
                    push(
                        lens,
                        buckets,
                        vgetq_lane_u32::<2>(b) as usize,
                        vgetq_lane_u32::<2>(ent),
                    );
                    push(
                        lens,
                        buckets,
                        vgetq_lane_u32::<3>(b) as usize,
                        vgetq_lane_u32::<3>(ent),
                    );
                }};
            }
            for _ in 0..sure {
                lanes!(vshrq_n_u32::<16>(r1), r1);
                lanes!(vshrq_n_u32::<16>(r2), r2);
                r1 = vaddq_u32(r1, p);
                r2 = vaddq_u32(r2, p);
            }
            lanes!(vminq_u32(vshrq_n_u32::<16>(r1), lastv), r1);
            lanes!(vminq_u32(vshrq_n_u32::<16>(r2), lastv), r2);
            e = vaddq_u32(e, step);
            i += 4;
        }
    }
    whole
}

/// Writes `entry` after the `lens[b]` entries of bucket `b` and adds one to `lens[b]`.
///
/// # Safety
///
/// `b < lens.len() <= buckets.len()` and `lens[b] < CAP`, which the bucket sieve makes sure of:
/// neither is checked.
#[inline(always)]
pub(super) unsafe fn push<const CAP: usize>(
    lens: &mut [u32],
    buckets: &mut [[u32; CAP]],
    b: usize,
    entry: u32,
) {
    debug_assert!(b < lens.len() && lens.len() <= buckets.len());
    debug_assert!((lens[b] as usize) < CAP);
    // SAFETY: as the caller makes sure, `b` is below the lengths of both slices and `lens[b]`
    // below `CAP`: every access is within its slice and array.
    unsafe {
        let count = lens.get_unchecked_mut(b);
        let k = *count as usize;
        *buckets.get_unchecked_mut(b).get_unchecked_mut(k) = entry;
        *count = k as u32 + 1;
    }
}

/// Pushes to `out` the position `i | byte << 16` of each byte of `block` (a multiple of 64 long)
/// with its top bit set, and sets every byte back to `init`.
#[inline(always)]
pub(super) fn scan(block: &mut [u8], init: u8, out: &mut Vec<u32>) {
    let (chunks, _) = block.as_chunks_mut::<64>();
    // SAFETY: NEON is enabled on the targets this module is compiled for, and each chunk is read
    // and written within its 64 bytes.
    unsafe {
        let iv = vdupq_n_u8(init);
        for (c, chunk) in chunks.iter_mut().enumerate() {
            let at = chunk.as_mut_ptr();
            let a = vorrq_u8(vld1q_u8(at), vld1q_u8(at.add(16)));
            let b = vorrq_u8(vld1q_u8(at.add(32)), vld1q_u8(at.add(48)));
            if vmaxvq_u8(vorrq_u8(a, b)) & 0x80 != 0 {
                found(chunk, c, out);
            }
            vst1q_u8(at, iv);
            vst1q_u8(at.add(16), iv);
            vst1q_u8(at.add(32), iv);
            vst1q_u8(at.add(48), iv);
        }
    }
}

/// Pushes the positions of the bytes of chunk `c` with their top bit set, out of the scan's loop
/// so that it keeps its registers.
#[inline(never)]
#[cold]
fn found(chunk: &[u8; 64], c: usize, out: &mut Vec<u32>) {
    // SAFETY: NEON is enabled on the targets this module is compiled for, and the loads read the
    // chunk's 64 bytes.
    unsafe {
        for q in 0..4 {
            let x = vld1q_u8(chunk.as_ptr().add(16 * q));
            // A nibble for each byte: 0xF where the top bit is set.
            let top = vcltzq_s8(vreinterpretq_s8_u8(x));
            let mut bits = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(
                vreinterpretq_u16_u8(top),
            )));
            while bits != 0 {
                let k = bits.trailing_zeros() / 4;
                let i = 16 * q + k as usize;
                out.push((64 * c + i) as u32 | u32::from(chunk[i]) << 16);
                bits &= !(0xF << (4 * k));
            }
        }
    }
}
