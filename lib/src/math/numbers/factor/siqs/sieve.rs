//! The sieve: for each polynomial, the interval `[−M, M)` in blocks of [`BLOCK`] bytes (or one
//! shorter block for small numbers), each byte the sum of the logarithms of the primes that divide
//! the polynomial's value there, and the bytes that reach the threshold handed to the trial division
//! ([`super::tdiv`]).
//!
//! - The tiny primes, below a bound of the parameters (256 to 1024 by the size of the number), are
//!   not sieved, the small-prime variation (Pomerance 1985): they hit most bytes and add little.
//!   Their share is found exactly instead, from their roots, for each byte that reaches a first
//!   threshold set lower by a margin: a candidate is a byte whose sum with that share reaches the
//!   second, the threshold proper. At 65 digits the two took 17 % less time than one lowered by
//!   the tiny primes' mean, finding more relations with fewer candidates.
//! - The medium primes, below [`BLOCK`], are sieved block by block from their two roots, the
//!   positions carried from one block to the next, in runs of primes with the same number of hits
//!   a root that are certain: those by a loop of a constant count, the last one, if any, without a
//!   branch; the runs of one to three by a vector kernel where there is one, four primes at a time.
//! - The larger primes hit a block at most twice, most of them not at all: when a polynomial
//!   starts, their roots are moved to the new polynomial and their hits written to buckets, one for
//!   each block, slice of [`SLICE`] primes and lane of every [`LANES`]-th prime (Wambach and Wettig
//!   1995; yafu), a root's last step past the interval to a bucket of the lane that no block reads.
//!   Each block then adds the logarithm of each slice at the positions of its buckets, and the
//!   trial division finds the large primes of a candidate in the buckets of its block.
//! - Each byte starts at `128 − threshold`, so that a candidate is a byte with its top bit set,
//!   found sixty-four bytes at a time.

use super::arith::{inverse, mul_mod};
use super::fb::FactorBase;
use super::kernels::{FILL_RUN, RUN_FEW, fill_run, push, root_sum, run_few};
use super::linalg::GAUSS_MAX;
use super::params::Params;
use super::poly::{Chooser, NO_ROOT, b_values};
use super::relations::Store;
use super::wide::{MAX_LIMBS, Wide};

/// The bits of the length of a block.
pub(super) const BLOCK_BITS: u32 = 16;

/// The length of a block, in bytes: half of the 128 KiB data cache of the cores measured on.
pub(super) const BLOCK: usize = 1 << BLOCK_BITS;

/// The primes per slice of the bucket sieve: below `2^16`, as a bucket entry holds the index of its
/// prime in the slice in 16 bits, beside the position in the block.
pub(super) const SLICE: usize = 2048;

/// The bytes after a block that the steps past it add to.
const GUARD: usize = 64;

/// The smallest prime of `A`.
const A_MIN: u32 = 64;

/// The relations beyond the size of the factor base that the sieve collects when the matrix is
/// small enough for Gaussian elimination: each vector of the null space gives a factor with
/// probability about 1/2, and there are at least as many as the excess.
const SMALL_EXCESS: usize = 12;

/// The margin, in bits, of the test of the size of a candidate's cofactor before its buckets are
/// searched for its large primes.
const MARGIN: f64 = 2.0;

/// The relations beyond the size of the factor base when the matrix goes to block Lanczos, which
/// finds the vectors of the null space less surely than elimination: at 55 digits 32 took 6 % less
/// time than 96.
const LANCZOS_EXCESS: usize = 32;

/// The state of the sieve for one number.
pub(super) struct Siqs<'a> {
    /// The number, odd, limbs least significant first.
    pub(super) n: &'a [u64],
    /// `kN`.
    pub(super) kn: Vec<u64>,
    /// The factor base.
    pub(super) fb: FactorBase,
    /// The logarithm of each prime, scaled.
    pub(super) logp: Vec<u8>,
    /// The primes `[1, tiny_end)` are not sieved.
    pub(super) tiny_end: usize,
    /// The primes `[tiny_end, med_end)` are sieved block by block; those from `med_end` on through
    /// the buckets.
    pub(super) med_end: usize,
    /// The number of limbs of the wide numbers.
    pub(super) w: usize,
    /// Whether the polynomials are `(2Ax + B)² − kN = 4A·g(x)`, for `kN ≡ 1 (mod 4)`, rather than
    /// `(Ax + B)² − kN = A·g(x)`: their values are half as large.
    pub(super) two_a: bool,
    /// The number of blocks of the interval.
    pub(super) blocks: usize,
    /// The length of a block: [`BLOCK`], or less for an interval of one block.
    pub(super) blen: u32,
    /// Half the interval.
    pub(super) m: u32,
    /// The value each byte starts at.
    pub(super) init: u8,
    /// The least sum of the logarithms of the sieved primes and of the tiny primes that divide a
    /// candidate's value, scaled, for it to be tried.
    pub(super) stage2: u32,
    /// `M mod p` for each prime of the base.
    pub(super) mmod: Vec<u32>,
    /// The large primes are below this.
    pub(super) lp_bound: u64,
    /// The cofactors of two large primes are below this (0 without them).
    pub(super) dlp_bound: u64,
    /// The square of the largest prime of the factor base.
    pub(super) pmax2: u64,
    /// The relations needed.
    pub(super) target: usize,
    /// The choice of `A`.
    pub(super) chooser: Chooser,
    /// The current polynomial.
    pub(super) poly: Poly,
    /// The indices of the primes of each `A` taken so far.
    pub(super) a_list: Vec<Vec<u32>>,
    /// The block being sieved, and the guard after it.
    pub(super) block: Vec<u8>,
    /// The runs of medium primes with the same number of sure hits a block: start, end, hits.
    pub(super) runs: Vec<(u32, u32, u32)>,
    /// The logarithms of the medium primes, 0 for those of the current `A`, whose roots are
    /// [`NO_ROOT`] and whose steps fall anywhere.
    pub(super) lg_med: Vec<u8>,
    /// The positions of the next hits of the medium primes, from the start of the next block.
    pub(super) next1: Vec<u16>,
    /// The same for their second roots.
    pub(super) next2: Vec<u16>,
    /// The buckets: for each slice, [`CAP`] entries for each lane of each block, then for each
    /// lane of the misses, the steps past the interval, which no block reads.
    pub(super) buckets: Vec<u32>,
    /// The number of entries in each bucket.
    pub(super) bucket_len: Vec<u32>,
    /// The index of the first prime of each slice.
    pub(super) slice_start: Vec<u32>,
    /// The scaled logarithm of the primes of each slice.
    pub(super) slice_log: Vec<u8>,
    /// The candidates of a block: the position, and the byte above it.
    pub(super) candidates: Vec<u32>,
    /// The scale of the logarithms.
    pub(super) scale: f64,
    /// The bits of the largest cofactor kept, and a margin for the rounding of the logarithms.
    pub(super) keep_bits: f64,
    /// The bits of the cofactors that are primes above the large-prime bound, within the margin.
    pub(super) gap: (f64, f64),
    /// The odd primes below [`BLOCK`] as `u16`s (0 for 2), the trial division finding them by
    /// their roots.
    pub(super) p16: Vec<u16>,
    /// `⌊2^16/p⌋` for them.
    pub(super) m16: Vec<u16>,
    /// The positions of the first hits of their first roots in the block being sieved, kept for
    /// the trial division of its candidates (modulo `2^16`): by the sieve for the medium primes,
    /// from the roots for the tiny ones.
    pub(super) rb1: Vec<u16>,
    /// The same for their second roots.
    pub(super) rb2: Vec<u16>,
    /// The length of a block modulo each tiny prime.
    pub(super) tiny_step: Vec<u32>,
    /// The relations found.
    pub(super) store: Store,
    /// The factors of a candidate, as indices into the factor base.
    pub(super) factors: Vec<u32>,
}

/// The current polynomial.
pub(super) struct Poly {
    /// The indices of the primes of `A`.
    pub(super) a_idx: Vec<u32>,
    /// `A`.
    pub(super) a: Wide,
    /// `A^(−1) mod 2^(64·w)`.
    pub(super) a_inv: Wide,
    /// `B`.
    pub(super) b: Wide,
    /// The coefficient of `x` in `g(x)`: `B` for `(2Ax + B)²`, `2B` for `(Ax + B)²`.
    pub(super) b2: Wide,
    /// The coefficient of `x` in `Y`: `2A` or `A`.
    pub(super) a2: Wide,
    /// `2·B_l` for each prime of `A`.
    pub(super) bl2: Vec<Wide>,
    /// `C = (B² − kN)/A`.
    pub(super) c: Wide,
    /// The first root of each prime, as a position in the interval, modulo the prime.
    pub(super) root1: Vec<u32>,
    /// The second root.
    pub(super) root2: Vec<u32>,
    /// `2·B_l·A^(−1) mod p` for each prime of `A` (`l`) and of the factor base: `bainv[l·F + j]`.
    pub(super) bainv: Vec<u32>,
    /// The polynomial's number for this `A`.
    pub(super) v: u32,
    /// The number of this `A` in the list.
    pub(super) a_number: u32,
}

impl<'a> Siqs<'a> {
    /// The sieve for the odd `n` with no small factor, its multiplier's `kN`, the factor base and the
    /// parameters.
    pub(super) fn new(n: &'a [u64], kn: Vec<u64>, fb: FactorBase, params: &Params) -> Siqs<'a> {
        let size = fb.prime.len();
        let blocks = params.blocks.max(1);
        // An interval of one block may be shorter than a block.
        let blen = if blocks == 1 {
            params.interval.clamp(4096, BLOCK)
        } else {
            BLOCK
        };
        let len = blocks * blen;
        let m = (len / 2) as u32;
        let kn_bits = bits_of(&kn) as f64;
        let two_a = kn[0] & 3 == 1;
        // The largest value of |g| is about M·sqrt(kN/2), half as much for (2Ax + B)².
        let top = f64::from(m).log2() + 0.5 * kn_bits - 0.5 - if two_a { 1.0 } else { 0.0 };
        let w = ((top + 6.0) / 64.0).ceil() as usize;
        let w = w.clamp(1, MAX_LIMBS);
        let pmax = u64::from(fb.prime[size - 1]);
        let lp_bound = (pmax * u64::from(params.lp_mult))
            .min(pmax * pmax)
            .min(1 << 32);
        let dlp_bound = if params.dlp {
            ((lp_bound as f64).powf(1.8) as u64).min(lp_bound.saturating_mul(lp_bound))
        } else {
            0
        };
        // The tiny primes, which are not sieved, include those of the multiplier.
        let tiny_end = fb
            .prime
            .partition_point(|&p| p < params.tiny)
            .max(fb.k_end)
            .max(1);
        let med_end = fb
            .prime
            .partition_point(|&p| (p as usize) < BLOCK)
            .max(tiny_end);
        let p16: Vec<u16> = fb.prime[..med_end]
            .iter()
            .map(|&p| if p > 2 { p as u16 } else { 0 })
            .collect();
        let m16: Vec<u16> = fb.prime[..med_end]
            .iter()
            .map(|&p| if p > 2 { (65536 / p) as u16 } else { 0 })
            .collect();
        let mut runs = Vec::new();
        let mut j = tiny_end;
        while j < med_end {
            let sure = blen as u32 / fb.prime[j];
            let end = j + fb.prime[j..med_end].partition_point(|&p| blen as u32 / p == sure);
            runs.push((j as u32, end as u32, sure));
            j = end;
        }
        // What 2 contributes on average, in bits, by `kN mod 8`: the trial division finds it, and
        // the tiny primes' share is found exactly before.
        let two_bits = match (two_a, kn[0] & 7) {
            (_, 1) => 2.0,
            (false, 5) => 1.0,
            (true, _) => 0.0,
            _ => 0.5,
        };
        let target = top - f64::from(params.slack) / 10.0 * (lp_bound as f64).log2() - two_bits;
        let margin = f64::from(params.margin) / 10.0;
        let threshold = (target - margin).max(8.0);
        // Logarithms scaled so that the threshold is at most 100: a byte can then not wrap.
        let scale = (100.0 / threshold).min(1.0);
        let logp: Vec<u8> = fb
            .prime
            .iter()
            .map(|&p| (f64::from(p).log2() * scale).round().max(1.0) as u8)
            .collect();
        let init = 128 - (threshold * scale).round() as u8;
        let stage2 = (target * scale).round().max(0.0) as u32;
        // The slices of the large primes.
        let mut slice_start = Vec::new();
        let mut slice_log = Vec::new();
        let mut j = med_end;
        while j < size {
            slice_start.push(j as u32);
            let mid = (j + SLICE / 2).min(size - 1);
            slice_log.push(logp[mid]);
            j += SLICE;
        }
        let slices = slice_start.len();
        // `A` is about sqrt(2kN)/M, half as much for (2Ax + B)², of primes among the medium ones.
        let a_target = 0.5 * (kn_bits + 1.0) - f64::from(m).log2() - if two_a { 1.0 } else { 0.0 };
        let a_min = A_MIN;
        let a_first = fb
            .prime
            .partition_point(|&p| p < a_min)
            .max(fb.k_end)
            .max(1);
        let a_end = med_end.max(a_first + 1).min(size);
        let seed = n[0] ^ 0x5149_5351_5349_4553;
        let chooser = Chooser::new(&fb, a_target, params.a_bits, a_first, a_end, seed);
        // Gaussian elimination takes any excess of columns, Lanczos a few more.
        let excess = if size + SMALL_EXCESS <= GAUSS_MAX {
            SMALL_EXCESS
        } else {
            LANCZOS_EXCESS
        };
        let target = size + excess;
        Siqs {
            n,
            kn,
            lg_med: logp[..med_end].to_vec(),
            logp,
            tiny_end,
            med_end,
            w,
            two_a,
            blocks,
            blen: blen as u32,
            m,
            init,
            stage2,
            scale,
            // The bits of the largest cofactor kept, and those of the cofactors that would be primes
            // above the large-prime bound, with a margin of two bits for the rounding of the
            // logarithms of the large primes: with 4 the scans of the buckets were a quarter more
            // at 70 digits, for the same relations; with 1, 0.3 % of them were lost.
            keep_bits: ((if dlp_bound > 0 { dlp_bound } else { lp_bound }) as f64).log2() + MARGIN,
            gap: if dlp_bound > 0 {
                (
                    (lp_bound as f64).log2() + MARGIN,
                    2.0 * (pmax as f64).log2() - MARGIN,
                )
            } else {
                (0.0, 0.0)
            },
            mmod: fb.prime.iter().map(|&p| m % p).collect(),
            lp_bound,
            dlp_bound,
            pmax2: pmax * pmax,
            target,
            chooser,
            poly: Poly {
                a_idx: Vec::new(),
                a: Wide::default(),
                a_inv: Wide::default(),
                b: Wide::default(),
                b2: Wide::default(),
                a2: Wide::default(),
                bl2: Vec::new(),
                c: Wide::default(),
                root1: vec![NO_ROOT; size],
                root2: vec![NO_ROOT; size],
                bainv: Vec::new(),
                v: 0,
                a_number: 0,
            },
            a_list: Vec::new(),
            block: vec![init; BLOCK + GUARD],
            runs,
            next1: vec![0; med_end],
            next2: vec![0; med_end],
            buckets: vec![0u32; slices * region_of(blocks)],
            bucket_len: vec![0u32; slices * (blocks + 1) * LANES],
            slice_start,
            slice_log,
            candidates: Vec::with_capacity(256),
            p16,
            m16,
            rb1: vec![0; med_end],
            rb2: vec![0; med_end],
            tiny_step: fb.prime[..tiny_end]
                .iter()
                .map(|&p| blen as u32 % p)
                .collect(),
            store: Store::new(),
            factors: Vec::with_capacity(64),
            fb,
        }
    }

    /// Sieves until there are enough relations, or no new `A` is found (`false`).
    pub(super) fn collect(&mut self) -> bool {
        while self.store.count() < self.target {
            let Some(a_idx) = self.chooser.next() else {
                return false;
            };
            self.start_a(a_idx);
            let polys = 1u32 << (self.poly.a_idx.len() - 1);
            for v in 0..polys {
                if v > 0 {
                    self.next_b(v);
                } else {
                    self.fill_buckets();
                }
                self.sieve_poly();
            }
        }
        true
    }

    /// Sets up the first polynomial of the `A` of primes `a_idx`.
    #[inline(never)]
    fn start_a(&mut self, a_idx: Vec<u32>) {
        let w = self.w;
        let size = self.fb.prime.len();
        let s = a_idx.len();
        let (a, bl, _) = b_values(&self.fb, &a_idx);
        let two_a = self.two_a;
        let poly = &mut self.poly;
        poly.a = Wide::of(&a, w);
        poly.a_inv = poly.a.inverse(w);
        poly.a2 = if two_a {
            poly.a.add(&poly.a, w)
        } else {
            poly.a
        };
        poly.bl2 = bl
            .iter()
            .map(|b| Wide::of(b, w).add(&Wide::of(b, w), w))
            .collect();
        let mut b = Wide::default();
        for x in &bl {
            b = b.add(&Wide::of(x, w), w);
        }
        // For (2Ax + B)², B odd: A added when the sum is even, the same for every sign of the B_l.
        let odd_a = two_a && b.0[0] & 1 == 0;
        if odd_a {
            b = b.add(&poly.a, w);
        }
        poly.b = b;
        poly.b2 = if two_a { b } else { b.add(&b, w) };
        poly.c = c_of(&b, &self.kn, &poly.a_inv, w, two_a);
        poly.v = 0;
        poly.a_number = self.a_list.len() as u32;
        poly.bainv.clear();
        poly.bainv.resize(s * size, 0);
        // The remainders of A and the B_l by each prime of the base, from their limbs: independent
        // divisions rather than a chain of products.
        let mut bmod = vec![0u32; s];
        for j in 1..size {
            let p = self.fb.prime[j];
            let amod = rem_limbs(&a, p);
            if amod == 0 {
                // A prime of A.
                poly.root1[j] = NO_ROOT;
                poly.root2[j] = NO_ROOT;
                continue;
            }
            for (m, b) in bmod.iter_mut().zip(bl.iter()) {
                *m = rem_limbs(b, p);
            }
            // Y = 2Ax + B: the roots are (2A)^(-1)(±t − B).
            let ainv = inverse(if two_a { add_mod(amod, amod, p) } else { amod }, p);
            let mut bsum = if odd_a { amod } else { 0 };
            for (l, &bl) in bmod.iter().enumerate() {
                poly.bainv[l * size + j] = mul_mod(add_mod(bl, bl, p), ainv, p);
                bsum = add_mod(bsum, bl, p);
            }
            let t = self.fb.sqrt[j];
            let minus_t = if t == 0 { 0 } else { p - t };
            let mmod = self.mmod[j];
            poly.root1[j] = add_mod(mul_mod(ainv, sub_mod(t, bsum, p), p), mmod, p);
            poly.root2[j] = add_mod(mul_mod(ainv, sub_mod(minus_t, bsum, p), p), mmod, p);
        }
        for &j in &poly.a_idx {
            self.lg_med[j as usize] = self.logp[j as usize];
        }
        for &j in &a_idx {
            self.lg_med[j as usize] = 0;
        }
        poly.a_idx = a_idx.clone();
        self.a_list.push(a_idx);
    }

    /// Moves to polynomial `v > 0` of the current `A`: `B` and the roots of the medium primes, then
    /// those of the large primes with their buckets.
    #[inline(never)]
    fn next_b(&mut self, v: u32) {
        let w = self.w;
        let size = self.fb.prime.len();
        let nu = v.trailing_zeros() as usize;
        // Bit nu of the Gray code of v: set when B_nu turns negative.
        let minus = (v >> (nu + 1)) & 1 == 0;
        let poly = &mut self.poly;
        poly.v = v;
        if minus {
            poly.b = poly.b.sub(&poly.bl2[nu], w);
        } else {
            poly.b = poly.b.add(&poly.bl2[nu], w);
        }
        poly.b2 = if self.two_a {
            poly.b
        } else {
            poly.b.add(&poly.b, w)
        };
        poly.c = c_of(&poly.b, &self.kn, &poly.a_inv, w, self.two_a);
        let d = &poly.bainv[nu * size..(nu + 1) * size];
        let lo = 1;
        // B decreasing by 2B_nu moves the roots up by d, increasing moves them down.
        update_roots(
            &mut poly.root1[lo..size],
            &d[lo..size],
            &self.fb.prime[lo..size],
            minus,
        );
        update_roots(
            &mut poly.root2[lo..size],
            &d[lo..size],
            &self.fb.prime[lo..size],
            minus,
        );
        for &j in &poly.a_idx {
            poly.root1[j as usize] = NO_ROOT;
            poly.root2[j as usize] = NO_ROOT;
        }
        self.fill_buckets();
    }

    /// Writes the hits of the large primes to the buckets.
    #[inline(never)]
    fn fill_buckets(&mut self) {
        let size = self.fb.prime.len();
        let blocks = self.blocks;
        let len = blocks as u32 * self.blen;
        let per = (blocks + 1) * LANES;
        let region = region_of(blocks);
        let last = blocks as u32;
        for (sl, &start) in self.slice_start.iter().enumerate() {
            let start = start as usize;
            let end = (start + SLICE).min(size);
            let lens = &mut self.bucket_len[sl * per..(sl + 1) * per];
            lens.fill(0);
            let (buckets, _) = self.buckets[sl * region..(sl + 1) * region].as_chunks_mut::<CAP>();
            // One bucket of CAP entries for each count: a bucket's index checked once.
            let buckets = &mut buckets[..lens.len()];
            let poly = &self.poly;
            let primes = &self.fb.prime[..end];
            let mut j = start;
            while j < end {
                // A root of p has floor(len/p) hits for sure and at most one more: the primes of
                // the same number of sure hits together.
                let sure = len / primes[j];
                let run = if sure == 0 {
                    end
                } else {
                    j + primes[j..end].partition_point(|&p| len / p == sure)
                };
                // Four primes at a time by the vector kernel where there is one, the rest here.
                let mut from = j;
                if FILL_RUN {
                    // SAFETY: the primes of the slice are at least a block and their roots below
                    // them, so each root has at most one entry in each bucket of its lane; the lane
                    // of each prime is its index modulo 4, here and below, and a lane has at most
                    // `SLICE/4` primes of the slice: no bucket gets more than `CAP` entries. `lens`
                    // has `(last + 1)·LANES` counts, as many as `buckets` has buckets.
                    from += unsafe {
                        fill_run(
                            &primes[j..run],
                            &poly.root1[j..run],
                            &poly.root2[j..run],
                            (j - start) as u32,
                            (j & (LANES - 1)) as u32,
                            sure,
                            last,
                            lens,
                            buckets,
                        )
                    };
                }
                for (j, &p) in primes.iter().enumerate().take(run).skip(from) {
                    let (mut r1, mut r2) = (poly.root1[j], poly.root2[j]);
                    let e = ((j - start) as u32) << BLOCK_BITS;
                    let lane = j & (LANES - 1);
                    for _ in 0..sure {
                        for r in [&mut r1, &mut r2] {
                            let b = (*r >> BLOCK_BITS) as usize * LANES + lane;
                            // SAFETY: a sure hit is below the interval, so `b` is the bucket of a
                            // block below `last`, below `lens.len()`; the count stays below `CAP`,
                            // as for `fill_run` above.
                            unsafe { push(lens, buckets, b, e | (*r & (BLOCK as u32 - 1))) };
                            *r += p;
                        }
                    }
                    // The last hit, if any: a miss is written to the lane's bucket after the
                    // blocks', which no block reads, so that hits and misses take the same steps.
                    for r in [r1, r2] {
                        let b = (r >> BLOCK_BITS).min(last) as usize * LANES + lane;
                        // SAFETY: `b` is the bucket of a block at most `last`, below `lens.len()`;
                        // the count stays below `CAP`, as above.
                        unsafe { push(lens, buckets, b, e | (r & (BLOCK as u32 - 1))) };
                    }
                }
                j = run;
            }
        }
    }

    /// Sieves the current polynomial, block by block, and tries each candidate.
    fn sieve_poly(&mut self) {
        let (lo, hi) = (self.tiny_end, self.med_end);
        // The roots of the medium primes are below them, below 2^16 (NO_ROOT, of the primes of A,
        // set to 0 below).
        for (n, &r) in self.next1[lo..hi].iter_mut().zip(&self.poly.root1[lo..hi]) {
            *n = r as u16;
        }
        for (n, &r) in self.next2[lo..hi].iter_mut().zip(&self.poly.root2[lo..hi]) {
            *n = r as u16;
        }
        // The primes of A have no roots: they start at 0, as the medium sieve's positions must be
        // below their primes, and add nothing (their logarithms are 0).
        for &j in &self.poly.a_idx {
            let j = j as usize;
            if (lo..hi).contains(&j) {
                self.next1[j] = 0;
                self.next2[j] = 0;
            }
        }
        for b in 0..self.blocks {
            // The positions of the tiny primes' roots in the block, for their share of the
            // candidates and the trial division: the roots, then a block's length less each time.
            for j in 1..lo.min(hi) {
                let p = self.fb.prime[j];
                if b == 0 {
                    // The roots of the primes of A are NO_ROOT: any position does for them.
                    self.rb1[j] = self.poly.root1[j].min(p - 1) as u16;
                    self.rb2[j] = self.poly.root2[j].min(p - 1) as u16;
                } else {
                    let step = self.tiny_step[j];
                    self.rb1[j] = sub_mod(u32::from(self.rb1[j]), step, p) as u16;
                    self.rb2[j] = sub_mod(u32::from(self.rb2[j]), step, p) as u16;
                }
            }
            // The block was set to the initial value by the scan of the one before.
            self.sieve_medium();
            self.sieve_large(b);
            self.scan();
            let candidates = core::mem::take(&mut self.candidates);
            for &c in &candidates {
                self.candidate(b, c);
            }
            self.candidates = candidates;
        }
    }

    /// Adds the logarithms of the medium primes to the block, and moves their positions on to the
    /// next block. A root of `p` has `⌊BLOCK/p⌋` hits for sure and at most one more: the primes of the
    /// same number of sure hits are taken together, those hits by a loop of a constant count, and
    /// the last one without a branch, adding to a byte of the guard after the block when it misses.
    #[inline(never)]
    fn sieve_medium(&mut self) {
        let block: &mut [u8; BLOCK + GUARD] = (&mut self.block[..]).try_into().expect("a block");
        let (primes, lgs) = (&self.p16[..], &self.lg_med[..]);
        let (n1, n2) = (&mut self.next1[..], &mut self.next2[..]);
        let (rb1, rb2) = (&mut self.rb1[..], &mut self.rb2[..]);
        let end = self.blen;
        // The primes of a run with `sure` sure hits; written out in place, so that a run takes no
        // new reference to the block (which Miri would check whole every time).
        macro_rules! run {
            ($s:expr, $e:expr, $sure:expr) => {{
                let (ps, ls) = (&primes[$s..$e], &lgs[$s..$e]);
                let (m1, m2) = (&mut n1[$s..$e], &mut n2[$s..$e]);
                let (b1, b2) = (&mut rb1[$s..$e], &mut rb2[$s..$e]);
                for j in 0..ps.len().min(ls.len()).min(m1.len()).min(m2.len()).min(b1.len()).min(b2.len()) {
                    let (p, lg) = (u32::from(ps[j]), ls[j]);
                    let (mut r1, mut r2) = (u32::from(m1[j]), u32::from(m2[j]));
                    b1[j] = r1 as u16;
                    b2[j] = r2 as u16;
                    debug_assert!(r1 < p && r2 < p && $sure * p <= end);
                    // Two hits of each root a step: those `sure` hits are below `sure·p`, at
                    // most the block's length, as the positions are below `p`.
                    let (p2, pairs): (u32, u32) = (2 * p, $sure / 2);
                    for _ in 0..pairs {
                        // SAFETY: below `sure·p <= end`, as just said.
                        unsafe {
                            add_at(block, r1 as usize, lg);
                            add_at(block, r2 as usize, lg);
                            add_at(block, (r1 + p) as usize, lg);
                            add_at(block, (r2 + p) as usize, lg);
                        }
                        r1 += p2;
                        r2 += p2;
                    }
                    if $sure & 1 == 1 {
                        // SAFETY: as above.
                        unsafe {
                            add_at(block, r1 as usize, lg);
                            add_at(block, r2 as usize, lg);
                        }
                        r1 += p;
                        r2 += p;
                    }
                    // The last hits, in the block or in its guard of 64 bytes.
                    let i1 = if r1 < end { r1 } else { end + (r1 & 63) } as usize;
                    // SAFETY: `i1 < end + 64 <= BLOCK + GUARD`.
                    unsafe { add_at(block, i1, lg) };
                    r1 = if r1 < end { r1 + p } else { r1 };
                    let i2 = if r2 < end { r2 } else { end + (r2 & 63) } as usize;
                    // SAFETY: as for `i1`.
                    unsafe { add_at(block, i2, lg) };
                    r2 = if r2 < end { r2 + p } else { r2 };
                    m1[j] = (r1 - end) as u16;
                    m2[j] = (r2 - end) as u16;
                }
            }};
        }
        for &(s, e, sure) in &self.runs {
            let (s, e) = (s as usize, e as usize);
            // One to three sure hits by the vector kernel where there is one, four primes at a
            // time, and the primes it leaves by the loop.
            let mut s = s;
            if RUN_FEW && (1..=3).contains(&sure) {
                let (ps, ls) = (&primes[s..e], &lgs[s..e]);
                let (m1, m2) = (&mut n1[s..e], &mut n2[s..e]);
                let (b1, b2) = (&mut rb1[s..e], &mut rb2[s..e]);
                let blk = &mut block[..];
                // SAFETY: the positions are below their primes (see `sieve_poly` and the end of
                // `run!`), the primes of the run have `sure` sure hits in a block of `end` bytes,
                // `sure·p <= end`, and the block has its guard of 64 bytes after `end`.
                s += unsafe {
                    match sure {
                        1 => run_few::<1>(blk, end, ps, ls, m1, m2, b1, b2),
                        2 => run_few::<2>(blk, end, ps, ls, m1, m2, b1, b2),
                        _ => run_few::<3>(blk, end, ps, ls, m1, m2, b1, b2),
                    }
                };
            }
            // Few sure hits unrolled, so that the loop does not cost more than they do.
            match sure {
                0 => run!(s, e, 0u32),
                1 => run!(s, e, 1u32),
                2 => run!(s, e, 2u32),
                3 => run!(s, e, 3u32),
                4 => run!(s, e, 4u32),
                _ => run!(s, e, sure),
            }
        }
    }

    /// Adds the logarithms of the large primes of the bucket of block `b`.
    #[inline(never)]
    fn sieve_large(&mut self, b: usize) {
        let block: &mut [u8; BLOCK + GUARD] = (&mut self.block[..]).try_into().expect("a block");
        const MASK: usize = BLOCK - 1;
        for sl in 0..self.slice_start.len() {
            let lg = self.slice_log[sl];
            for lane in 0..LANES {
                let entries = bucket(&self.buckets, &self.bucket_len, self.blocks, sl, b, lane);
                // Four at a time, so that the loop costs less than the additions.
                let (quads, rest) = entries.as_chunks::<4>();
                for q in quads {
                    for &e in q {
                        let i = e as usize & MASK;
                        block[i] = block[i].wrapping_add(lg);
                    }
                }
                for &e in rest {
                    let i = e as usize & MASK;
                    block[i] = block[i].wrapping_add(lg);
                }
            }
        }
    }

    /// The positions of the block whose bytes have their top bit set.
    #[inline(never)]
    fn scan(&mut self) {
        self.candidates.clear();
        let len = self.blen as usize;
        super::kernels::scan(&mut self.block[..len], self.init, &mut self.candidates);
        // And the guard, which the steps past the block added to.
        self.block[BLOCK..].fill(self.init);
        // The tiny primes' exact share of each byte found, from their roots, and the candidates
        // whose sum reaches the second threshold.
        let te = self.tiny_end;
        let (p, m) = (&self.p16[1..te], &self.m16[1..te]);
        let (r1, r2, lg) = (&self.rb1[1..te], &self.rb2[1..te], &self.lg_med[1..te]);
        let (init, t2) = (u32::from(self.init), self.stage2);
        self.candidates
            .retain(|&c| (c >> 16) - init + root_sum(c as u16, p, m, r1, r2, lg) >= t2);
    }
}

/// Adds `lg` to byte `i` of the block, without a check of the bounds.
///
/// # Safety
///
/// `i < BLOCK + GUARD`, which the medium sieve makes sure of: each position it starts from is
/// below its prime, and of the hits from it those that are sure lie below the block's length and
/// the last is sent to the guard should it pass the block.
#[inline(always)]
unsafe fn add_at(block: &mut [u8; BLOCK + GUARD], i: usize, lg: u8) {
    debug_assert!(i < BLOCK + GUARD);
    // SAFETY: `i < BLOCK + GUARD`, the length of the array, as the caller makes sure. A check of
    // the bounds (or a mask in its place) was one instruction of the five of each hit, the most
    // frequent operation of the sieve.
    unsafe {
        let x = block.get_unchecked_mut(i);
        *x = x.wrapping_add(lg);
    }
}

/// `C = (B² − kN)/A`, or `(B² − kN)/(4A)` (`two_a`), modulo `2^(64·w)`, from the inverse of `A`:
/// the division is exact, and the quotient by 4 a shift of `4C`, which is below `2^(64·w − 1)`.
fn c_of(b: &Wide, kn: &[u64], a_inv: &Wide, w: usize, two_a: bool) -> Wide {
    let c = b.mul(b, w).sub(&Wide::of(kn, w), w).mul(a_inv, w);
    if two_a { c.shr2(w) } else { c }
}

/// The remainder of the limbs `x` (least significant first, at least one) by `p < 2^32`: the top
/// limb by one division, each other by two on its halves.
#[inline(always)]
fn rem_limbs(x: &[u64], p: u32) -> u32 {
    let p = u64::from(p);
    let (top, rest) = x.split_last().expect("a limb");
    let mut r = top % p;
    for &l in rest.iter().rev() {
        r = ((r << 32) | (l >> 32)) % p;
        r = ((r << 32) | (l & 0xFFFF_FFFF)) % p;
    }
    r as u32
}

/// `r + d mod p` for `r, d < p < 2^31`, without a branch.
#[inline(always)]
fn add_mod(r: u32, d: u32, p: u32) -> u32 {
    let x = (r + d).wrapping_sub(p);
    x.wrapping_add(p & ((x as i32 >> 31) as u32))
}

/// `r − d mod p` for `r, d < p < 2^31`, without a branch.
#[inline(always)]
fn sub_mod(r: u32, d: u32, p: u32) -> u32 {
    let x = r.wrapping_sub(d);
    x.wrapping_add(p & ((x as i32 >> 31) as u32))
}

/// Moves the roots `r` by `+d` (`up`) or `−d` modulo the primes `p`.
#[inline(never)]
fn update_roots(r: &mut [u32], d: &[u32], p: &[u32], up: bool) {
    let n = r.len().min(d.len()).min(p.len());
    let (r, d, p) = (&mut r[..n], &d[..n], &p[..n]);
    if up {
        for i in 0..n {
            r[i] = add_mod(r[i], d[i], p[i]);
        }
    } else {
        for i in 0..n {
            r[i] = sub_mod(r[i], d[i], p[i]);
        }
    }
}

/// The bits of the limbs `x`.
pub(super) fn bits_of(x: &[u64]) -> u32 {
    let mut n = x.len();
    while n > 0 && x[n - 1] == 0 {
        n -= 1;
    }
    if n == 0 {
        0
    } else {
        64 * (n as u32 - 1) + 64 - x[n - 1].leading_zeros()
    }
}

/// The entries of the bucket of slice `sl`, block `b` and lane `lane`, among `blocks` blocks.
#[inline(always)]
pub(super) fn bucket<'b>(
    buckets: &'b [u32],
    lens: &[u32],
    blocks: usize,
    sl: usize,
    b: usize,
    lane: usize,
) -> &'b [u32] {
    let i = b * LANES + lane;
    let k = lens[sl * (blocks + 1) * LANES + i] as usize;
    let at = sl * region_of(blocks) + i * CAP;
    &buckets[at..at + k]
}

/// The entries of the buckets of a slice: [`CAP`] for each lane of each block and of the misses
/// after them.
fn region_of(blocks: usize) -> usize {
    (blocks + 1) * LANES * CAP
}

/// The buckets of each block and slice, the primes taken in turn: their counts, which the
/// writes of consecutive primes increment, then do not wait on each other.
pub(super) const LANES: usize = 4;

/// The room of a bucket: two entries for each prime of its lane.
const CAP: usize = 2 * SLICE / LANES;
