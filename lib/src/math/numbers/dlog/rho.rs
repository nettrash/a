//! Pollard's rho method for logarithms in a ring of residues: Teske's walk of 32 multipliers and
//! distinguished points.

use super::super::modular::int::Int;
use super::super::modular::ring::Ring;

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

/// `a + b mod q` for `a, b < q`.
#[inline(always)]
fn add_mod<I: Int>(a: &I, b: &I, q: &I) -> I {
    let room = q.sub(b);
    if *a >= room { a.sub(&room) } else { a.add(b) }
}

/// The walks that run at once: each step waits for a product, a load and the choice of the next
/// multiplier, and eight independent ones keep the multipliers busy. Measured on an Apple M4 Max
/// modulo primes `2q + 1`, orders `q` of 40 bits took 17.7 million cycles with one walk, 5.7 to
/// 11.3 million with four in memory, and 3.9 million with eight in registers.
const WALKS: usize = 8;

/// The powers of a base by any exponent below `2^(4 w)`, from the table of its `x^(i 16^j)`,
/// `0 < i < 16`, `j < w`: one product for each non-zero digit in base 16, and the products of
/// different powers independent, where a power by squarings is a chain.
struct Powers<V> {
    /// `x^(i 16^j)` at `15 j + i - 1`.
    table: Vec<V>,
}

impl<V: Clone> Powers<V> {
    /// The table of `x` for exponents of `bits` bits.
    fn new<R: Ring<V = V>>(r: &R, x: &V, bits: u64) -> Powers<V> {
        let w = bits.div_ceil(4).max(1) as usize;
        let mut table = Vec::with_capacity(15 * w);
        let mut base = x.clone();
        for j in 0..w {
            let mut p = base.clone();
            table.push(p.clone());
            for _ in 1..15 {
                p = r.mul(&p, &base);
                table.push(p.clone());
            }
            if j + 1 < w {
                // `base^16 = base^15 base`.
                base = r.mul(&p, &base);
            }
        }
        Powers { table }
    }

    /// `x^e`, for `e` of at most the table's bits.
    fn pow<R: Ring<V = V>>(&self, r: &R, e: &[u64]) -> V {
        let mut acc: Option<V> = None;
        for (j, &l) in e.iter().enumerate() {
            for k in 0..16 {
                let digit = ((l >> (4 * k)) & 15) as usize;
                let at = 15 * (16 * j + k);
                if digit != 0 && at < self.table.len() {
                    let t = &self.table[at + digit - 1];
                    acc = Some(match acc {
                        None => t.clone(),
                        Some(a) => r.mul(&a, t),
                    });
                }
            }
        }
        acc.unwrap_or_else(|| r.one())
    }
}

/// The distinguished points met, by an open-addressing table of their keys: each slot holds 0
/// or the place plus one of a point in the list, and a probe goes on to the next slot.
struct Points<V, I> {
    slots: Vec<u32>,
    points: Vec<(u64, V, I, I)>,
}

impl<V: PartialEq, I> Points<V, I> {
    /// A table for about `n` points before it grows.
    fn new(n: usize) -> Points<V, I> {
        Points {
            slots: vec![0; (4 * n).next_power_of_two().max(64)],
            points: Vec::with_capacity(n),
        }
    }

    /// The point `x` of key `k`, when it was met; otherwise `x` is put in, with its exponents.
    fn meet(&mut self, k: u64, x: V, a: I, b: I) -> Option<&(u64, V, I, I)> {
        if 2 * (self.points.len() + 1) > self.slots.len() {
            self.grow();
        }
        let mask = self.slots.len() - 1;
        let mut i = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                self.slots[i] = self.points.len() as u32 + 1;
                self.points.push((k, x, a, b));
                return None;
            }
            let p = &self.points[s as usize - 1];
            if p.0 == k && p.1 == x {
                return Some(&self.points[s as usize - 1]);
            }
            i = (i + 1) & mask;
        }
    }

    /// Twice the slots, the points put back.
    fn grow(&mut self) {
        let n = 2 * self.slots.len();
        self.slots = vec![0; n];
        let mask = n - 1;
        for (j, p) in self.points.iter().enumerate() {
            let mut i = (p.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask;
            while self.slots[i] != 0 {
                i = (i + 1) & mask;
            }
            self.slots[i] = j as u32 + 1;
        }
    }
}

/// What a distinguished point told.
enum Met<V, I> {
    /// The logarithm.
    Found(I),
    /// A collision that tells nothing: the walk starts again there.
    Restart(V, I, I),
    /// A new distinguished point, kept: the walk goes on.
    Kept,
}

/// What the walks share besides their elements: the distinguished points met, the tables of the
/// powers of `g` and `h` and the generator of their starts, and the collisions that told nothing.
struct Shared<V, I> {
    seen: Points<V, I>,
    pg: Powers<V>,
    ph: Powers<V>,
    rng: Rng,
    degenerate: u32,
}

impl<V: Clone + PartialEq, I: Int> Shared<V, I> {
    /// A start `g^a h^b` for random `a` and `b`.
    fn start<R: Ring<V = V>>(&mut self, r: &R, q: &I) -> (V, I, I) {
        let a = self.rng.below(q);
        let b = self.rng.below(q);
        let x = r.mul(
            &self.pg.pow(r, a.limbs().as_slice()),
            &self.ph.pow(r, b.limbs().as_slice()),
        );
        (x, a, b)
    }
}

/// The distinguished point `x = g^a h^b` of key `k`: kept when new, and otherwise a collision,
/// which gives the logarithm when `b` differs from the other's, and a new start when not.
#[cold]
#[inline(never)]
fn meet<R: Ring, I: Int>(
    sh: &mut Shared<R::V, I>,
    r: &R,
    q: &I,
    x: R::V,
    a: I,
    b: I,
    k: u64,
) -> Met<R::V, I> {
    if let Some((_, _, a2, b2)) = sh.seen.meet(k, x, a.clone(), b.clone()) {
        if *b2 != b {
            // `k (b - b') = a' - a`.
            let db = b.sub_mod(b2, q);
            let da = a2.sub_mod(&a, q);
            if let Some(inv) = db.inverse(q) {
                return Met::Found(da.mul_mod(&inv, q));
            }
        }
        sh.degenerate += 1;
        let (x, a, b) = sh.start(r, q);
        return Met::Restart(x, a, b);
    }
    Met::Kept
}

/// A `k` with `g^k = h` in `r`, for `g` of the prime order `q` and `h` with `h^q = 1`, by Pollard's
/// rho method (1978): walks `x -> x M_i` on elements `g^a h^b`, the multiplier `M_i = g^(u_i)
/// h^(v_i)` chosen by the low bits of the element's [`Ring::walk_key`], until one comes to an
/// element met before; then `g^a h^b = g^(a') h^(b')` gives `k = (a' - a) / (b - b') mod q`. The
/// elements met are recognised by keeping the distinguished ones, those with `d` zero bits in their
/// key above the multiplier's, about one in `2^d`, in a table: a walk meets one soon after it joins
/// a path met before, its own or another's (van Oorschot and Wiener, 1999), so [`WALKS`] of them
/// run side by side, their elements in registers. `Degenerate` after `tries` collisions with
/// `b = b'`, which every collision is when `h` is not a power of `g`.
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
    let mask = ((1u64 << d) - 1) << 5;
    // About `1.25 sqrt(q) / 2^d` distinguished points: room for them up to `2^12`, and the table
    // grows beyond, so that an order out of reach takes no memory before its walks need it.
    let expected = 1usize << (bits / 2 - d).min(12);
    let mut sh = Shared {
        seen: Points::new(expected),
        pg: Powers::new(r, g, bits),
        ph: Powers::new(r, h, bits),
        rng: Rng(0x0123_4567_89AB_CDEF ^ bits),
        degenerate: 0,
    };
    let mut ms: Vec<R::V> = Vec::with_capacity(MULTIPLIERS);
    let mut us: Vec<I> = Vec::with_capacity(MULTIPLIERS);
    let mut vs: Vec<I> = Vec::with_capacity(MULTIPLIERS);
    for _ in 0..MULTIPLIERS {
        let (m, u, v) = sh.start(r, q);
        ms.push(m);
        us.push(u);
        vs.push(v);
    }
    let (Ok(mult), Ok(u), Ok(v)) = (
        <&[R::V; MULTIPLIERS]>::try_from(&ms[..]),
        <&[I; MULTIPLIERS]>::try_from(&us[..]),
        <&[I; MULTIPLIERS]>::try_from(&vs[..]),
    ) else {
        unreachable!()
    };
    let mut xs: [R::V; WALKS] = core::array::from_fn(|_| r.one());
    let mut a: [I; WALKS] = core::array::from_fn(|_| I::small(0));
    let mut b: [I; WALKS] = core::array::from_fn(|_| I::small(0));
    for w in 0..WALKS {
        (xs[w], a[w], b[w]) = sh.start(r, q);
    }
    // A walk that meets no distinguished point in this many rounds is in a cycle without one:
    // the walks that met none since the last look start again.
    let longest = 64u64 << d;
    let mut rounds = 0u64;
    let mut met = 0u32;
    // One step of walk `w`, written out for each so that the elements stay in registers.
    macro_rules! step {
        ($w:expr) => {{
            let k = r.walk_key(&xs[$w]);
            let mut moves = true;
            if k & mask == 0 {
                let (x, aa, bb) = (xs[$w].clone(), a[$w].clone(), b[$w].clone());
                match meet(&mut sh, r, q, x, aa, bb, k) {
                    Met::Found(k) => return Walk::Found(k),
                    Met::Restart(x, aa, bb) => {
                        if sh.degenerate >= tries {
                            return Walk::Degenerate;
                        }
                        (xs[$w], a[$w], b[$w]) = (x, aa, bb);
                        moves = false;
                    }
                    Met::Kept => met |= 1 << $w,
                }
            }
            if moves {
                let i = (k & (MULTIPLIERS as u64 - 1)) as usize;
                xs[$w] = r.mul(&xs[$w], &mult[i]);
                a[$w] = add_mod(&a[$w], &u[i], q);
                b[$w] = add_mod(&b[$w], &v[i], q);
            }
        }};
    }
    loop {
        step!(0);
        step!(1);
        step!(2);
        step!(3);
        step!(4);
        step!(5);
        step!(6);
        step!(7);
        rounds += 1;
        if rounds == longest {
            rounds = 0;
            for w in 0..WALKS {
                if met & (1 << w) == 0 {
                    (xs[w], a[w], b[w]) = sh.start(r, q);
                }
            }
            met = 0;
        }
    }
}
