//! Integers rebuilt from their residues modulo word primes by the Chinese remainder theorem, one
//! prime at a time (Garner's incremental form: `x ← x + P·((r − x)·P⁻¹ mod p)`, `P ← P·p`), each
//! held as limbs, and taken at the end as the residues of least magnitude.

use super::field::Field;
use crate::math::numbers::nat::addmul_1;
use crate::math::numbers::{ByteOrder, Number, from_limbs};

/// `len` integers being rebuilt from their residues modulo at most `count` primes.
pub(in crate::math::matrix) struct Crt {
    /// The limbs of each, `width` apiece.
    x: Vec<u64>,
    width: usize,
    /// The product `P` of the primes so far, `width` limbs, `mlen` of them in use.
    modulus: Vec<u64>,
    mlen: usize,
    /// The primes taken so far.
    taken: usize,
}

impl Crt {
    /// `len` integers, for at most `count` primes.
    pub(in crate::math::matrix) fn new(len: usize, count: usize) -> Crt {
        let width = count + 1;
        let mut modulus = vec![0u64; width];
        modulus[0] = 1;
        Crt {
            x: vec![0u64; len * width],
            width,
            modulus,
            mlen: 1,
            taken: 0,
        }
    }

    /// The limbs the residues of the integers need powers for: [`Field::powers`] of this length
    /// at least.
    pub(in crate::math::matrix) fn width(&self) -> usize {
        self.width
    }

    /// Takes the residues `r` of the integers modulo the prime `p` of `f`, with `powers` the
    /// residues of `2^(64·i)` for `i` below [`width`](Crt::width) at least.
    pub(in crate::math::matrix) fn add(&mut self, f: Field, p: u64, r: &[u64], powers: &[u64]) {
        let (width, mlen) = (self.width, self.mlen);
        debug_assert!(
            self.taken < width - 1,
            "more primes than the integers were made for"
        );
        if self.taken == 0 {
            for (e, &v) in r.iter().enumerate() {
                self.x[e * width] = v;
            }
        } else {
            // x ← x + P·((r − x mod p)·P⁻¹ mod p), P the product of the primes before.
            let pm = f.of_limbs_by(false, &self.modulus[..mlen], powers);
            let inv = f.inv(pm);
            for (e, &v) in r.iter().enumerate() {
                let xe = &mut self.x[e * width..(e + 1) * width];
                let xm = f.of_limbs_by(false, &xe[..mlen], powers);
                let t = f.mul(f.sub(v, xm), inv);
                if t != 0 {
                    let carry = addmul_1(&mut xe[..mlen], &self.modulus[..mlen], t);
                    xe[mlen] = xe[mlen].wrapping_add(carry);
                }
            }
        }
        // P ← P·p.
        let mut carry = 0u64;
        for w in &mut self.modulus[..mlen] {
            let t = *w as u128 * p as u128 + carry as u128;
            *w = t as u64;
            carry = (t >> 64) as u64;
        }
        self.modulus[mlen] = carry;
        self.mlen = self
            .modulus
            .iter()
            .rposition(|&w| w != 0)
            .map_or(1, |i| i + 1);
        self.taken += 1;
    }

    /// The integers, each the residue of least magnitude modulo the product of the primes:
    /// `x − P` when `x > P/2`. Each written into `out`, which holds zeros, where it is not zero.
    pub(in crate::math::matrix) fn finish_into(self, out: &mut [Number]) {
        let (width, mlen) = (self.width, self.mlen);
        let modulus = &self.modulus[..mlen];
        let half: Vec<u64> = {
            let mut h = modulus.to_vec();
            let mut carry = 0;
            for w in h.iter_mut().rev() {
                let next = *w & 1;
                *w = (*w >> 1) | (carry << 63);
                carry = next;
            }
            h
        };
        for (e, out) in out.iter_mut().enumerate() {
            let xe = &self.x[e * width..(e + 1) * width];
            let top = xe.iter().rposition(|&w| w != 0).map_or(0, |i| i + 1);
            if top == 0 {
                continue;
            }
            let above = top > mlen || (top == mlen && greater(&xe[..mlen], &half));
            if above {
                // P − x, negative.
                let mut d = modulus.to_vec();
                let mut borrow = false;
                for (w, &v) in d.iter_mut().zip(&xe[..mlen]) {
                    let (s, b1) = w.overflowing_sub(v);
                    let (s, b2) = s.overflowing_sub(borrow as u64);
                    *w = s;
                    borrow = b1 || b2;
                }
                let mut v = from_limbs(&d, ByteOrder::LittleEndian);
                v.negative = true;
                *out = v;
            } else {
                *out = from_limbs(&xe[..top], ByteOrder::LittleEndian);
            }
        }
    }
}

/// `x > y` for magnitudes of the same length.
fn greater(x: &[u64], y: &[u64]) -> bool {
    for (a, b) in x.iter().rev().zip(y.iter().rev()) {
        if a != b {
            return a > b;
        }
    }
    false
}
