//! Pollard's rho method for logarithms in a ring of residues: Teske's walk of 32 multipliers and
//! distinguished points.

use super::super::modular::int::Int;
use super::super::modular::ring::Ring;
use std::collections::HashMap;

/// The number of multipliers of the walk, a power of two: Teske (2001) found that 20 or more make
/// the walk behave as a random one.
const MULTIPLIERS: usize = 32;

/// The outcome of a walk.
pub(in super::super) enum Walk<I> {
    /// The logarithm.
    Found(I),
    /// Only collisions that tell nothing: `h` is not a power of `g`, or the walk was unlucky.
    Degenerate,
}

/// A small generator of the walk's exponents, the same for every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number below `q`.
    fn below<I: Int>(&mut self, q: &I) -> I {
        let l = q.limbs();
        let n = l.as_slice().len();
        let v: Vec<u64> = (0..n + 1).map(|_| self.next()).collect();
        I::of_limbs_reduced(&v, q)
    }
}

/// The key of an element times an odd constant: its top five bits choose the multiplier, and the
/// distinguished points have the `d` bits from bit 32 zero, which depend on the key's bits from 0
/// to `32 + d`. Not the low bits of the product: those depend on the key's low bits alone, and the
/// residues of a ring by divisions are not in Montgomery's form; those of a group of odd order
/// modulo an even number are all odd (1 modulo 8 modulo a multiple of 8, say), and a walk would
/// never meet a distinguished point.
#[inline(always)]
fn mix(key: u64) -> u64 {
    key.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// `a + b mod q` for `a, b < q`.
#[inline(always)]
fn add_mod<I: Int>(a: &I, b: &I, q: &I) -> I {
    let room = q.sub(b);
    if *a >= room { a.sub(&room) } else { a.add(b) }
}

/// The walks that run at once: each step waits for a product, and four independent ones keep the
/// multipliers busy. Measured on an Apple M4 Max modulo primes `2q + 1` of 41 bits, orders `q` of
/// 40 bits took 17.7 million cycles with one walk and 5.7 to 11.3 million with four (the medians
/// of runs whose luck differs); 30 bits 859,000 and 225,000 to 425,000.
const WALKS: usize = 4;

/// A walk: the element `g^a h^b`, its exponents and the steps since its last distinguished point.
struct Walker<V, I> {
    x: V,
    a: I,
    b: I,
    steps: u64,
}

/// A `k` with `g^k = h` in `r`, for `g` of the prime order `q` and `h` with `h^q = 1`, by Pollard's
/// rho method (1978): walks `x -> x M_i` on elements `g^a h^b`, the multiplier `M_i = g^(u_i)
/// h^(v_i)` chosen by the element's key, until one comes to an element met before; then
/// `g^a h^b = g^(a') h^(b')` gives `k = (a' - a) / (b - b') mod q`. The elements met are recognised
/// by keeping the distinguished ones, those with `d` zero bits in their key, about one in `2^d`,
/// in a table: a walk meets one soon after it joins a path met before, its own or another's
/// (van Oorschot and Wiener, 1999), so [`WALKS`] of them run side by side. `Degenerate` after
/// `tries` collisions with `b = b'`, which every collision is when `h` is not a power of `g`.
pub(in super::super) fn rho_in<R: Ring, I: Int>(
    r: &R,
    h: &R::V,
    g: &R::V,
    q: &I,
    tries: u32,
) -> Walk<I> {
    let bits = q.bits();
    // About `sqrt(q)` steps in all; a distinguished point every `2^d` of them keeps the table to
    // about `sqrt(q) / 2^d` entries and costs `2^d` steps more for each walk.
    let d = (bits / 4).clamp(0, 24);
    let mask = (1u64 << d) - 1;
    let mut rng = Rng(0x0123_4567_89AB_CDEF ^ bits);
    let mut mult: Vec<(R::V, I, I)> = Vec::with_capacity(MULTIPLIERS);
    for _ in 0..MULTIPLIERS {
        let u = rng.below(q);
        let v = rng.below(q);
        let m = r.mul(
            &r.pow(g, u.limbs().as_slice()),
            &r.pow(h, v.limbs().as_slice()),
        );
        mult.push((m, u, v));
    }
    let start = |rng: &mut Rng| {
        let a = rng.below(q);
        let b = rng.below(q);
        let x = r.mul(
            &r.pow(g, a.limbs().as_slice()),
            &r.pow(h, b.limbs().as_slice()),
        );
        Walker { x, a, b, steps: 0 }
    };
    let mut walkers: Vec<Walker<R::V, I>> = (0..WALKS).map(|_| start(&mut rng)).collect();
    let mut seen: HashMap<u64, Vec<(R::V, I, I)>> = HashMap::new();
    let mut degenerate = 0;
    // A walk that meets no distinguished point in this many steps is in a cycle without one.
    let longest = 64u64 << d;
    loop {
        for w in walkers.iter_mut() {
            let mixed = mix(r.key(&w.x));
            if (mixed >> 32) & mask == 0 || w.steps > longest {
                if w.steps <= longest {
                    let entry = seen.entry(mixed).or_default();
                    match entry.iter().find(|(y, _, _)| *y == w.x) {
                        Some((_, a2, b2)) => {
                            if *b2 != w.b {
                                // `k (b - b') = a' - a`.
                                let db = w.b.sub_mod(b2, q);
                                let da = a2.sub_mod(&w.a, q);
                                if let Some(inv) = db.inverse(q) {
                                    return Walk::Found(da.mul_mod(&inv, q));
                                }
                            }
                            degenerate += 1;
                            if degenerate >= tries {
                                return Walk::Degenerate;
                            }
                        }
                        None => {
                            entry.push((w.x.clone(), w.a.clone(), w.b.clone()));
                            w.steps = 0;
                            let i = (mixed >> 59) as usize;
                            let (m, u, v) = &mult[i];
                            w.x = r.mul(&w.x, m);
                            w.a = add_mod(&w.a, u, q);
                            w.b = add_mod(&w.b, v, q);
                            continue;
                        }
                    }
                }
                // A collision that tells nothing, or a cycle without a distinguished point: the
                // walk starts again elsewhere.
                *w = start(&mut rng);
                continue;
            }
            let i = (mixed >> 59) as usize;
            let (m, u, v) = &mult[i];
            w.x = r.mul(&w.x, m);
            w.a = add_mod(&w.a, u, q);
            w.b = add_mod(&w.b, v, q);
            w.steps += 1;
        }
    }
}
