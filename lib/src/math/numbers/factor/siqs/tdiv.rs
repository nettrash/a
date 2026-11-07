//! The trial division of the candidates: the value `g(x)` of the polynomial at a byte that reached
//! the threshold, divided by the primes of the factor base that divide it, found without dividing:
//! the tiny primes and those of `A` by an exact-division test, the medium primes by comparing
//! `x mod p` with their roots, and the large primes by looking up the candidate's position in its
//! block's buckets. What is left is 1 (a full relation), a prime below the large-prime bound (a
//! partial relation), or, with two large primes, a composite below their bound that splits into two
//! such primes.

use super::arith::{div_exact_1, divides, fastmod, rho_64, sprp2};
use super::relations::Relation;
use super::sieve::{BLOCK, LANES, Siqs, bucket};
use super::wide::MAX_LIMBS;

impl Siqs<'_> {
    /// Tries the candidate at position `off` of block `b`.
    #[inline(never)]
    pub(super) fn candidate(&mut self, b: usize, packed: u32) {
        let (off, byte) = (packed & 0xFFFF, (packed >> 16) as u8);
        let w = self.w;
        let i = b as u32 * self.blen + off;
        let x = i64::from(i) - i64::from(self.m);
        let poly = &self.poly;
        // g(x) = (A x + 2B) x + C, or (A x + B) x + C for (2Ax + B)².
        let g = poly
            .a
            .mul_i64(x, w)
            .add(&poly.b2, w)
            .mul_i64(x, w)
            .add(&poly.c, w);
        let negative = g.is_negative(w);
        let g = if negative { g.neg(w) } else { g };
        let mut v: [u64; MAX_LIMBS] = g.0;
        let mut len = w;
        while len > 0 && v[len - 1] == 0 {
            len -= 1;
        }
        if len == 0 {
            return;
        }
        let factors = &mut self.factors;
        factors.clear();
        factors.extend_from_slice(&poly.a_idx);
        // 2.
        let mut z = 0;
        while v[z / 64] >> (z % 64) & 1 == 0 {
            z += 1;
        }
        if z > 0 {
            shift_right(&mut v, &mut len, z);
            for _ in 0..z {
                factors.push(0);
            }
        }
        let fb = &self.fb;
        // What the sieved primes found add to the byte, once each.
        let (lg_med, tiny_end) = (&self.lg_med, self.tiny_end);
        let found = core::cell::Cell::new(0u32);
        let mut take = |j: usize, v: &mut [u64; MAX_LIMBS], len: &mut usize| {
            let p = u64::from(fb.prime[j]);
            let inv = fb.inv64[j];
            let mut first = true;
            while divides(&v[..*len], p, inv) {
                div_exact_1(&mut v[..*len], p, inv);
                while *len > 0 && v[*len - 1] == 0 {
                    *len -= 1;
                }
                factors.push(j as u32);
                if first && j >= tiny_end && j < lg_med.len() {
                    found.set(found.get() + u32::from(lg_med[j]));
                }
                first = false;
            }
        };
        // The primes up to 256 by the test of exact division.
        let split = self.root_test;
        for j in 1..split {
            take(j, &mut v, &mut len);
        }
        for &j in &poly.a_idx {
            take(j as usize, &mut v, &mut len);
        }
        // The medium primes, sixteen at a time: `off mod p` from `⌊off·⌊2^16/p⌋/2^16⌋`, which is the
        // quotient or one less, so that the remainder modulo 2^16 is that of a root or of a root
        // plus p when p divides (and rarely otherwise: the division tells).
        let (lo, hi) = (split, self.med_end);
        let o = off as u16;
        let n = hi - lo;
        let (ps, ms) = (&self.p16[lo..hi], &self.m16[lo..hi]);
        let (rb1, rb2) = (&self.rb1[lo..hi], &self.rb2[lo..hi]);
        let rem = |o: u16, p: u16, m: u16| {
            o.wrapping_sub((((u32::from(o) * u32::from(m)) >> 16) as u16).wrapping_mul(p))
        };
        for c in 0..n / 16 {
            let range = 16 * c..16 * c + 16;
            let p: &[u16; 16] = ps[range.clone()].try_into().expect("16");
            let m: &[u16; 16] = ms[range.clone()].try_into().expect("16");
            let r1: &[u16; 16] = rb1[range.clone()].try_into().expect("16");
            let r2: &[u16; 16] = rb2[range].try_into().expect("16");
            let mut hit = 0u16;
            for k in 0..16 {
                let x = rem(o, p[k], m[k]);
                let (d1, d2) = (x.wrapping_sub(r1[k]), x.wrapping_sub(r2[k]));
                hit |= u16::from(d1 == 0)
                    | u16::from(d1 == p[k])
                    | u16::from(d2 == 0)
                    | u16::from(d2 == p[k]);
            }
            if hit != 0 {
                for k in 0..16 {
                    let x = rem(o, p[k], m[k]);
                    let (d1, d2) = (x.wrapping_sub(r1[k]), x.wrapping_sub(r2[k]));
                    if d1 == 0 || d1 == p[k] || d2 == 0 || d2 == p[k] {
                        take(lo + 16 * c + k, &mut v, &mut len);
                    }
                }
            }
        }
        for j in lo + 16 * (n / 16)..hi {
            let r = fastmod(i, fb.fastmod[j], fb.prime[j]);
            if r == poly.root1[j] || r == poly.root2[j] {
                take(j, &mut v, &mut len);
            }
        }
        // What the large primes added to the byte tells the size of their product: a cofactor
        // that would stay too large spares the search of the buckets.
        let large = i64::from(byte.wrapping_sub(self.init)) - i64::from(found.get());
        let bits = 64 * len as i64 - i64::from(v[len.max(1) - 1].leading_zeros());
        let rest = bits as f64 - large.max(0) as f64 / self.scale;
        if rest > self.keep_bits || (rest > self.gap.0 && rest < self.gap.1) {
            return;
        }
        // The large primes: the entries of the block's buckets at the candidate's position,
        // sixteen entries compared at a time.
        for (sl, &start) in self.slice_start.iter().enumerate() {
            for lane in 0..LANES {
                let entries = bucket(&self.buckets, &self.bucket_len, self.blocks, sl, b, lane);
                let (chunks, rest) = entries.as_chunks::<16>();
                for chunk in chunks {
                    let mut hit = 0u32;
                    for &e in chunk {
                        hit |= u32::from(e & (BLOCK as u32 - 1) == off);
                    }
                    if hit != 0 {
                        for &e in chunk {
                            if e & (BLOCK as u32 - 1) == off {
                                take(
                                    start as usize + (e >> super::sieve::BLOCK_BITS) as usize,
                                    &mut v,
                                    &mut len,
                                );
                            }
                        }
                    }
                }
                for &e in rest {
                    if e & (BLOCK as u32 - 1) == off {
                        take(
                            start as usize + (e >> super::sieve::BLOCK_BITS) as usize,
                            &mut v,
                            &mut len,
                        );
                    }
                }
            }
        }
        if len > 1 {
            return;
        }
        let c = v[0];
        let lp = if c == 1 {
            [1, 1]
        } else if c < self.lp_bound {
            [c as u32, 1]
        } else if c < self.dlp_bound && c > self.pmax2 && !sprp2(c) {
            // Below the square of the largest prime of the base, the cofactor, which has no prime
            // factor in the base, is a prime: above, if it is not, it may split into two.
            let Some(f) = rho_64(c, 1 << 16) else {
                return;
            };
            let (p, q) = (f.min(c / f), f.max(c / f));
            if q >= self.lp_bound {
                return;
            }
            [p as u32, q as u32]
        } else {
            return;
        };
        let y = poly.a2.mul_i64(x, w).add(&poly.b, w);
        let y = if y.is_negative(w) { y.neg(w) } else { y };
        let key = y.0[0];
        let rel = Relation {
            a: poly.a_number,
            v: poly.v,
            x: x as i32,
            negative,
            lp,
            start: 0,
            len: 0,
        };
        self.store.add(rel, &self.factors, key);
    }
}

/// Shifts the limbs `v[..len]` right by `z` bits and trims them.
fn shift_right(v: &mut [u64; MAX_LIMBS], len: &mut usize, z: usize) {
    let (q, r) = (z / 64, z % 64);
    let n = *len;
    for k in 0..n {
        let lo = if k + q < n { v[k + q] } else { 0 };
        let hi = if k + q + 1 < n { v[k + q + 1] } else { 0 };
        v[k] = if r == 0 {
            lo
        } else {
            (lo >> r) | (hi << (64 - r))
        };
    }
    while *len > 0 && v[*len - 1] == 0 {
        *len -= 1;
    }
}
