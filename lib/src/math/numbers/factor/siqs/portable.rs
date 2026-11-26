//! The portable kernels of the sieve, for the targets without the NEON ones of the module
//! `aarch64` and under Miri: the same functions in plain Rust, with the same results.

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
    let hit = |k: usize| {
        let x =
            o.wrapping_sub((((u32::from(o) * u32::from(m[k])) >> 16) as u16).wrapping_mul(p[k]));
        let (d1, d2) = (x.wrapping_sub(r1[k]), x.wrapping_sub(r2[k]));
        d1 == 0 || d1 == p[k] || d2 == 0 || d2 == p[k]
    };
    // Sixteen at a time, the lanes looked at again only when one hits.
    let whole = n / 16 * 16;
    for c in (0..whole).step_by(16) {
        let mut bits = 0u32;
        for k in 0..16 {
            bits |= u32::from(hit(c + k)) << k;
        }
        while bits != 0 {
            f(c + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
    for k in whole..n {
        if hit(k) {
            f(k);
        }
    }
}

/// The sum of the logarithms `lg[k]` of the primes `p[k]` whose roots the offset `o` hits, as
/// [`root_hits`] finds them.
#[inline(always)]
pub(super) fn root_sum(o: u16, p: &[u16], m: &[u16], r1: &[u16], r2: &[u16], lg: &[u8]) -> u32 {
    let n = p
        .len()
        .min(m.len())
        .min(r1.len())
        .min(r2.len())
        .min(lg.len());
    let mut sum = 0u32;
    for k in 0..n {
        let x =
            o.wrapping_sub((((u32::from(o) * u32::from(m[k])) >> 16) as u16).wrapping_mul(p[k]));
        let (d1, d2) = (x.wrapping_sub(r1[k]), x.wrapping_sub(r2[k]));
        if d1 == 0 || d1 == p[k] || d2 == 0 || d2 == p[k] {
            sum += u32::from(lg[k]);
        }
    }
    sum
}

/// Whether [`run_few`] sieves the runs of one to three sure hits: not here, where the medium
/// sieve's own loop does.
pub(super) const RUN_FEW: bool = false;

/// The NEON kernel's counterpart, never called (see [`RUN_FEW`]): no primes done.
///
/// # Safety
///
/// None here; `unsafe` as the NEON kernel is, whose bytes added to are not checked.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_few<const SURE: u32>(
    _block: &mut [u8],
    _end: u32,
    _ps: &[u16],
    _lgs: &[u8],
    _m1: &mut [u16],
    _m2: &mut [u16],
    _b1: &mut [u16],
    _b2: &mut [u16],
) -> usize {
    0
}

/// Whether [`fill_run`] writes the buckets of runs of large primes: not here, where the bucket
/// sieve's own loop does.
pub(super) const FILL_RUN: bool = false;

/// The NEON kernel's counterpart, never called (see [`FILL_RUN`]): no primes done.
///
/// # Safety
///
/// None here; `unsafe` as the NEON kernel is, whose writes are not checked.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fill_run<const CAP: usize>(
    _ps: &[u32],
    _r1s: &[u32],
    _r2s: &[u32],
    _first: u32,
    _lane: u32,
    _sure: u32,
    _last: u32,
    _lens: &mut [u32],
    _buckets: &mut [[u32; CAP]],
) -> usize {
    0
}

/// Writes `entry` after the `lens[b]` entries of bucket `b` and adds one to `lens[b]`, for
/// `b < lens.len() <= buckets.len()` and `lens[b] < CAP`, which the bucket sieve makes sure of.
///
/// # Safety
///
/// None here, where the indices are checked; `unsafe` as the NEON kernel's is, which does not
/// check them.
#[inline(always)]
pub(super) unsafe fn push<const CAP: usize>(
    lens: &mut [u32],
    buckets: &mut [[u32; CAP]],
    b: usize,
    entry: u32,
) {
    let k = lens[b] as usize;
    debug_assert!(k < CAP);
    // The index modulo CAP, which it is below, spares one of the checks of the bounds.
    buckets[b][k % CAP] = entry;
    lens[b] = k as u32 + 1;
}

/// Pushes to `out` the position `i | byte << 16` of each byte of `block` (a multiple of 64 long)
/// with its top bit set, and sets every byte back to `init`.
#[inline(always)]
pub(super) fn scan(block: &mut [u8], init: u8, out: &mut Vec<u32>) {
    let (chunks, _) = block.as_chunks_mut::<64>();
    for (c, chunk) in chunks.iter_mut().enumerate() {
        let mut acc = 0u64;
        for k in 0..8 {
            acc |= u64::from_le_bytes(chunk[8 * k..8 * k + 8].try_into().expect("8 bytes"));
        }
        if acc & 0x8080_8080_8080_8080 != 0 {
            for (k, &x) in chunk.iter().enumerate() {
                if x & 0x80 != 0 {
                    out.push((c * 64 + k) as u32 | u32::from(x) << 16);
                }
            }
        }
        chunk.fill(init);
    }
}
