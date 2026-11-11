//! Lenstra's elliptic curve method (1987) on Montgomery's curves `B y^2 = x^3 + A x^2 + x` (1987),
//! with the parametrisation of GMP-ECM's `-param 2` (Bouvier and Zimmermann): for `sigma >= 2`, the
//! point `sigma (-3, 3)` on the curve `y^2 = x^3 + 36` gives `x_3 = (3x + y + 6) / (2 (y - 3))`, the
//! curve of `(A + 2) / 4 = (1 - x_3)^3 (3 x_3 + 1) / (16 x_3^3)`, Suyama's of `u / v = x_3`, and the
//! starting point `x = 2`, which lies on it with a point of order 6, so that the group orders
//! modulo every prime are multiples of 12 and as likely to be smooth as those of Suyama's curves
//! (GMP-ECM's `-param 0`). The curves are those of `sigma = 2, 3, 4, ...` in turn, so that a
//! factorisation is reproducible; they are GMP-ECM's `-sigma 2:s`, whose stage 1 finds the same
//! factors. Only from a `sigma` with `sigma (-3, 3)` at infinity modulo every prime of `n` at once,
//! after which the curves would repeat modulo each, are the curves GMP-ECM's `-sigma 1:s` instead,
//! `(A + 2) / 4 = sigma^2 / 2^64` from `x = 2`.
//!
//! The arithmetic is on `x` and `z` alone: the doubling `2 (x : z)` and the addition of two points
//! whose difference is known take 5 and 6 products, and Montgomery's ladder computes `e P` with
//! one of each per bit of `e`, `(R0, R1) -> (2 R0, R0 + R1)` or `(R0 + R1, 2 R1)`, `R1 - R0 = P`
//! all along. With `P = (2 : 1)`, the product by `P`'s `x` in the addition is a doubling: a step
//! costs 4 squares and 5 products; the bit decides only which point is doubled, by a swap without
//! a branch.
//!
//! - Stage 1: `Q = E P` through the chunks of the exponent `E` (see [`stage1`](super::stage1)),
//!   from `x = 2` for the first, the point taken to affine after each by one inversion, which a
//!   factor of `n` common to `z` makes impossible: its greatest common divisor is the factor. When
//!   it is `n`, the chunk is gone over again a prime at a time. In registers, the curves go through
//!   stage 1 in blocks, step by step, so that the products of one curve fill the latency of
//!   another's: four for one and two limbs, two for three, one from four. A block's curves are
//!   made together, a point addition on `y^2 = x^3 + 36` each, and their parameters and points
//!   taken to affine by one inversion for the block.
//! - Stage 2: the continuation of [`stage2`](super::stage2) on `x`: the baby steps `j Q` by the
//!   chain `(j + 2) Q = j Q + 2 Q`, the giant steps `k D Q` by `(k + 1) D Q = k D Q + D Q`, both
//!   taken to affine together by Montgomery's trick, a block of giant steps at a time, so that each
//!   pair `(k, j)` costs one product, `x_(kD) - x_j`; or, for the longer ranges, the FFT
//!   continuation of [`ecmfft`](super::ecmfft) on the same steps, which hands the curve to the
//!   other when its product is 0 modulo `n` or its steps cannot be taken to affine modulo any
//!   prime of `n`.

use super::ecmfft::{Eval, FftPlan};
use super::field::{Arith, Job, invert, invert_all, mul_cold, mul_in_cold};
use super::pm1::{bit, bit_length, proper};
use super::primes::primes_in;
use super::rho::{Gcd, gcd};
use super::stage1::{Chunk, Stage1, prime_power};
use super::stage2::{BLOCK, Plan};

/// The first `sigma` of the curves.
pub(super) const FIRST_SIGMA: u64 = 2;

/// The curves of `sigma` from `first` for `count` of them, with the stages' exponent and plan.
pub(super) struct EcmJob<'a> {
    /// The first `sigma`.
    pub(super) first: u64,
    /// The curves.
    pub(super) count: u64,
    /// The first stage.
    pub(super) stage1: &'a Stage1,
    /// The second stage, if any.
    pub(super) plan: Option<&'a Plan>,
}

impl Job for EcmJob<'_> {
    type Out = Option<Vec<u64>>;

    fn run<A: Arith>(self, f: &A) -> Option<Vec<u64>> {
        curves(f, self.first, self.count, self.stage1, self.plan)
    }
}

/// The curves of `sigma` from `first`, `count` of them, in the arithmetic `f`: the first factor
/// that one of them finds, in the order of the curves, from either stage.
pub(super) fn curves<A: Arith>(
    f: &A,
    first: u64,
    count: u64,
    stage1: &Stage1,
    plan: Option<&Plan>,
) -> Option<Vec<u64>> {
    if count == 0 {
        return None;
    }
    let mut curves = Curves::new(f, first);
    curves_from(f, &mut curves, count, usize::MAX, stage1, plan)
}

/// [`curves`] for the next `count` curves of `curves`, which goes on from them, the first block of at
/// most `first` curves.
pub(super) fn curves_from<A: Arith>(
    f: &A,
    curves: &mut Curves<A::V>,
    count: u64,
    first: usize,
    stage1: &Stage1,
    plan: Option<&Plan>,
) -> Option<Vec<u64>> {
    let mut left = count;
    // The width of a block, decided by the type, so that the wider ladders are compiled for the
    // arithmetics that take them alone.
    let width = const { lanes::<A>() };
    let mut limit = first.max(1);
    while left > 0 {
        let take = left.min(width.min(limit) as u64) as usize;
        limit = width;
        // Three curves take a block of their own for one limb, where four would wait a third longer
        // for the products; for two limbs, a block of four with one curve unused.
        let one_limb = const { core::mem::size_of::<A::V>() == core::mem::size_of::<u64>() };
        let found = match (width, take) {
            (4, 4) => block::<A, 4>(f, curves, 4, stage1, plan),
            (4, 3) if one_limb => block::<A, 3>(f, curves, 3, stage1, plan),
            (4, 3) => block::<A, 4>(f, curves, 3, stage1, plan),
            (4, 2) | (2, 2) => block::<A, 2>(f, curves, 2, stage1, plan),
            _ => block::<A, 1>(f, curves, 1, stage1, plan),
        };
        if found.is_some() {
            return found;
        }
        left -= take as u64;
    }
    None
}

/// The curves a block takes in the arithmetic `A`, by the size of a residue: 4 for one limb and two
/// (in registers, where a product waits for the one before it and four curves' fill the wait), 2
/// for three limbs, 1 from four limbs and in memory (where the products of one curve already keep
/// the multipliers busy). Measured on an Apple M4 Max, stage 1 at `B1 = 3000`, per curve: one limb
/// 61, 38, 27 and 27 µs for 1, 2, 3 and 4 curves at once; two limbs 125, 106, 102 and 103 µs below
/// `2^126` (lazily reduced) and 177, 155, 147 and 142 µs above; three limbs 299 and 277 µs for one
/// and two; four limbs and more within 2 % for any number.
const fn lanes<A: Arith>() -> usize {
    let size = core::mem::size_of::<A::V>();
    if A::MEMORY || size > 3 * core::mem::size_of::<u64>() {
        1
    } else if size == 3 * core::mem::size_of::<u64>() {
        2
    } else {
        4
    }
}

/// A curve in stage 1.
enum State<V> {
    /// Going on, from the affine `x` (2 for the first chunk), with `(A + 2) / 4`.
    Alive(V, V),
    /// Done, with a factor.
    Found(Vec<u64>),
    /// Done, with nothing: every prime was caught at once, or the curve is not used.
    Dead,
}

/// The curves of GMP-ECM's `-param 2` in turn: the multiples `sigma P` of `P = (-3, 3)` on
/// `y^2 = x^3 + 36`, in Jacobian coordinates `(X : Y : Z)`, `x = X / Z^2`, `y = Y / Z^3`.
///
/// When `sigma P` is at infinity modulo every prime of `n` at once, `P` has the order `sigma`
/// modulo each of them: the multiples, and so the curves, would only repeat from there modulo each
/// prime (and the formulas cannot leave the point at infinity, which would leave every later curve
/// degenerate). The curves of that `sigma` and after are GMP-ECM's `-param 1` ones instead,
/// `(A + 2) / 4 = sigma^2 / 2^64`, also from `x = 2`, which do not repeat.
pub(super) struct Curves<V> {
    /// `sigma P`.
    x: V,
    y: V,
    z: V,
    /// The `sigma` of the next curve.
    sigma: u64,
    /// Whether the curves are GMP-ECM's `-param 1` ones from here on.
    param1: bool,
}

impl<V: Clone> Curves<V> {
    /// The curves from `sigma >= 2`: `sigma P` by doublings and additions of `P` from the top bit.
    pub(super) fn new<A: Arith<V = V>>(f: &A, sigma: u64) -> Self {
        let sigma = sigma.max(2);
        let mut c = Curves {
            x: f.zero(),
            y: f.small(3),
            z: f.one(),
            sigma,
            param1: false,
        };
        f.sub(&mut c.x, &f.zero(), &f.small(3));
        for i in (0..63 - sigma.leading_zeros()).rev() {
            c.double(f);
            if (sigma >> i) & 1 == 1 {
                c.add_p(f);
            }
        }
        c
    }

    /// `2 Q` for `y^2 = x^3 + 36` (`a = 0`): 2 products and 5 squares (dbl-2009-l).
    #[inline(never)]
    fn double<A: Arith<V = V>>(&mut self, f: &A) {
        let (mut a, mut b, mut c, mut d, mut e) =
            (f.zero(), f.zero(), f.zero(), f.zero(), f.zero());
        f.sqr(&mut a, &self.x);
        f.sqr(&mut b, &self.y);
        f.sqr(&mut c, &b);
        // `D = 2 ((X + B)^2 - A - C)`.
        f.add(&mut d, &self.x, &b);
        f.sqr_in(&mut d);
        f.sub_in(&mut d, &a);
        f.sub_in(&mut d, &c);
        let t = d.clone();
        f.add_in(&mut d, &t);
        // `E = 3 A`, `X3 = E^2 - 2 D`.
        f.add(&mut e, &a, &a);
        f.add_in(&mut e, &a);
        f.sqr(&mut self.x, &e);
        f.sub_in(&mut self.x, &d);
        f.sub_in(&mut self.x, &d);
        // `Z3 = 2 Y Z`, before `Y` changes.
        mul_in_cold(f, &mut self.z, &self.y);
        let t = self.z.clone();
        f.add_in(&mut self.z, &t);
        // `Y3 = E (D - X3) - 8 C`.
        f.sub_in(&mut d, &self.x);
        mul_cold(f, &mut self.y, &e, &d);
        for _ in 0..3 {
            let t = c.clone();
            f.add_in(&mut c, &t);
        }
        f.sub_in(&mut self.y, &c);
    }

    /// `Q + P` for `P = (-3, 3)`, affine (madd-2007-bl, with `P`'s coordinates small): 3 products
    /// and 4 squares.
    #[inline(never)]
    fn add_p<A: Arith<V = V>>(&mut self, f: &A) {
        let (mut zz, mut u, mut s, mut h, mut hh) =
            (f.zero(), f.zero(), f.zero(), f.zero(), f.zero());
        f.sqr(&mut zz, &self.z);
        // `U2 = -3 Z1^2`, `S2 = 3 Z1^3`.
        f.add(&mut u, &zz, &zz);
        f.add_in(&mut u, &zz);
        mul_cold(f, &mut s, &u, &self.z);
        let t = u.clone();
        f.sub(&mut u, &f.zero(), &t);
        // `H = U2 - X1`, `HH = H^2`, `I = 4 HH`, `J = H I`, `r = 2 (S2 - Y1)`, `V = X1 I`.
        f.sub(&mut h, &u, &self.x);
        f.sqr(&mut hh, &h);
        let mut i = f.zero();
        f.add(&mut i, &hh, &hh);
        let t = i.clone();
        f.add_in(&mut i, &t);
        let mut j = f.zero();
        mul_cold(f, &mut j, &h, &i);
        let mut r = f.zero();
        f.sub(&mut r, &s, &self.y);
        let t = r.clone();
        f.add_in(&mut r, &t);
        let mut v = f.zero();
        mul_cold(f, &mut v, &self.x, &i);
        // `Z3 = (Z1 + H)^2 - Z1^2 - HH`.
        f.add_in(&mut self.z, &h);
        f.sqr_in(&mut self.z);
        f.sub_in(&mut self.z, &zz);
        f.sub_in(&mut self.z, &hh);
        // `X3 = r^2 - J - 2 V`.
        f.sqr(&mut self.x, &r);
        f.sub_in(&mut self.x, &j);
        f.sub_in(&mut self.x, &v);
        f.sub_in(&mut self.x, &v);
        // `Y3 = r (V - X3) - 2 Y1 J`.
        mul_in_cold(f, &mut j, &self.y);
        let t = j.clone();
        f.add_in(&mut j, &t);
        f.sub_in(&mut v, &self.x);
        mul_cold(f, &mut self.y, &r, &v);
        f.sub_in(&mut self.y, &j);
    }

    /// The curve of the current `sigma`, as `(A + 2) / 4 = num / den`, and on to the next: with
    /// `N = 3 X Z + Y + 6 Z^3` and `M = 2 (Y - 3 Z^3)`, `x_3 = N / M` and
    /// `(A + 2) / 4 = (M - N)^3 (3N + M) / (16 N^3 M)`, both times `Z`, so that a point at infinity
    /// modulo a prime, where the curve would be degenerate, makes `den` share that prime with `n`.
    /// From a point at infinity modulo `n` itself, GMP-ECM's `-param 1` curves (see [`Curves`]).
    fn next<A: Arith<V = V>>(&mut self, f: &A) -> (V, V) {
        let sigma = self.sigma;
        self.sigma += 1;
        if !self.param1 && f.is_zero(&self.z) {
            self.param1 = true;
        }
        if self.param1 {
            // `(A + 2) / 4 = sigma^2 / 2^64`.
            let (s, w) = (f.small(sigma), f.small(1 << 32));
            let (mut num, mut den) = (f.zero(), f.zero());
            mul_cold(f, &mut num, &s, &s);
            mul_cold(f, &mut den, &w, &w);
            return (num, den);
        }
        let (mut z2, mut z3, mut n, mut m) = (f.zero(), f.zero(), f.zero(), f.zero());
        f.sqr(&mut z2, &self.z);
        mul_cold(f, &mut z3, &z2, &self.z);
        // `T = 3 Z^3`.
        let mut t3 = f.zero();
        f.add(&mut t3, &z3, &z3);
        f.add_in(&mut t3, &z3);
        // `N = 3 X Z + Y + 2 T`.
        mul_cold(f, &mut n, &self.x, &self.z);
        let xz = n.clone();
        f.add_in(&mut n, &xz);
        f.add_in(&mut n, &xz);
        f.add_in(&mut n, &self.y);
        f.add_in(&mut n, &t3);
        f.add_in(&mut n, &t3);
        // `M = 2 (Y - T)`.
        f.sub(&mut m, &self.y, &t3);
        let t = m.clone();
        f.add_in(&mut m, &t);
        // `num = (M - N)^3 (3N + M) Z`.
        let mut w = f.zero();
        f.sub(&mut w, &m, &n);
        let mut num = f.zero();
        f.sqr(&mut num, &w);
        mul_in_cold(f, &mut num, &w);
        let mut s = f.zero();
        f.add(&mut s, &n, &n);
        f.add_in(&mut s, &n);
        f.add_in(&mut s, &m);
        mul_in_cold(f, &mut num, &s);
        mul_in_cold(f, &mut num, &self.z);
        // `den = 16 N^3 M Z`.
        let mut den = f.zero();
        f.sqr(&mut den, &n);
        mul_in_cold(f, &mut den, &n);
        mul_in_cold(f, &mut den, &m);
        mul_in_cold(f, &mut den, &self.z);
        for _ in 0..4 {
            let t = den.clone();
            f.add_in(&mut den, &t);
        }
        self.add_p(f);
        (num, den)
    }

    /// The states of the next `count <= K` curves, `Dead` beyond, with one inversion for all:
    /// `Alive(2, (A + 2) / 4)`, or what the inversion gives for a curve when it is impossible.
    fn block<A: Arith<V = V>, const K: usize>(&mut self, f: &A, count: usize) -> [State<V>; K] {
        let mut nums: Vec<V> = Vec::with_capacity(count);
        let mut dens: Vec<V> = Vec::with_capacity(count);
        for _ in 0..count {
            let (num, den) = self.next(f);
            nums.push(num);
            dens.push(den);
        }
        let two = f.small(2);
        if invert_all(f, &mut dens).is_ok() {
            let mut it = nums.into_iter().zip(dens);
            return core::array::from_fn(|_| match it.next() {
                Some((mut a24, inv)) => {
                    mul_in_cold(f, &mut a24, &inv);
                    State::Alive(two.clone(), a24)
                }
                None => State::Dead,
            });
        }
        let mut it = nums.into_iter().zip(dens);
        core::array::from_fn(|_| match it.next() {
            Some((mut a24, den)) => match invert(f, &den) {
                Ok(inv) => {
                    mul_in_cold(f, &mut a24, &inv);
                    State::Alive(two.clone(), a24)
                }
                Err(g) => match proper(f, g) {
                    Some(g) => State::Found(g),
                    None => State::Dead,
                },
            },
            None => State::Dead,
        })
    }
}

/// The index of the first curve of the states with a factor, or their number.
fn first_found<V>(st: &[State<V>]) -> usize {
    st.iter()
        .position(|s| matches!(s, State::Found(_)))
        .unwrap_or(st.len())
}

/// The next `count <= K` curves of `curves` through stage 1 together, then stage 2 each: the factor
/// the curves would give one after the other, the first's, from either stage, before the second's.
fn block<A: Arith, const K: usize>(
    f: &A,
    curves: &mut Curves<A::V>,
    count: usize,
    stage1: &Stage1,
    plan: Option<&Plan>,
) -> Option<Vec<u64>> {
    let mut st: [State<A::V>; K] = curves.block(f, count);
    let b1 = stage1.b1;
    let two = f.small(2);
    let mut tmp: [Tmp<A::V>; K] = core::array::from_fn(|_| Tmp::new(f));
    let mut first = true;
    stage1.each(|chunk| {
        // The curves that still matter: those before the first with a factor.
        let limit = first_found(&st);
        if !st[..limit].iter().any(|s| matches!(s, State::Alive(..))) {
            return false;
        }
        let xs: [A::V; K] = core::array::from_fn(|i| match &st[i] {
            State::Alive(x, _) => x.clone(),
            _ => two.clone(),
        });
        let ks: [A::V; K] = core::array::from_fn(|i| match &st[i] {
            State::Alive(_, a24) => a24.clone(),
            _ => two.clone(),
        });
        let (qx, qz) = if first {
            ladder_lanes::<A, K, true>(f, &xs, &ks, &chunk.e, &mut tmp)
        } else {
            // The chunks after the first, from `B1` about 2.9 million, from the affine points: a
            // curve at a time, by the ladder that stage 2 takes too, so that only the ladders from
            // `x = 2` are compiled for the blocks.
            let mut qx: [A::V; K] = core::array::from_fn(|_| f.zero());
            let mut qz: [A::V; K] = core::array::from_fn(|_| f.zero());
            for i in 0..limit {
                if let State::Alive(x, a24) = &st[i] {
                    let (q, _) = ladder(f, x, a24, &chunk.e, &mut tmp[i]);
                    qx[i] = q.x;
                    qz[i] = q.z;
                }
            }
            (qx, qz)
        };
        first = false;
        let run: Vec<usize> = (0..limit)
            .filter(|&i| matches!(st[i], State::Alive(..)))
            .collect();
        let mut zs: Vec<A::V> = run.iter().map(|&i| qz[i].clone()).collect();
        if invert_all(f, &mut zs).is_ok() {
            for (&i, zi) in run.iter().zip(&zs) {
                let mut y = f.zero();
                f.mul(&mut y, &qx[i], zi);
                st[i] = State::Alive(y, ks[i].clone());
            }
        } else {
            for &i in &run {
                let q = Point {
                    x: qx[i].clone(),
                    z: qz[i].clone(),
                };
                st[i] = settle(f, &xs[i], ks[i].clone(), &q, chunk, b1, &mut tmp[i]);
            }
        }
        // The first curve's factor ends the block; otherwise on while one before the first factor
        // goes on.
        !matches!(st[0], State::Found(_))
    });
    for (s, t) in st.iter_mut().zip(tmp.iter_mut()) {
        match take(s) {
            State::Found(g) => return Some(g),
            State::Alive(x, a24) => {
                if let Some(plan) = plan
                    && let Some(g) = stage2(f, &x, &a24, plan, t)
                {
                    return Some(g);
                }
            }
            State::Dead => {}
        }
    }
    None
}

/// The state of a curve after a chunk, from the point `q` it reached and the affine `x` it
/// started from.
#[allow(clippy::too_many_arguments)]
fn settle<A: Arith>(
    f: &A,
    x: &A::V,
    a24: A::V,
    q: &Point<A::V>,
    chunk: &Chunk,
    b1: u64,
    t: &mut Tmp<A::V>,
) -> State<A::V> {
    match invert(f, &q.z) {
        Ok(zi) => {
            let mut y = f.zero();
            f.mul(&mut y, &q.x, &zi);
            State::Alive(y, a24)
        }
        Err(g) => {
            let found = if g == f.n() {
                backtrack(f, x.clone(), &a24, chunk, b1, t)
            } else {
                Some(g)
            };
            match found {
                Some(g) => State::Found(g),
                None => State::Dead,
            }
        }
    }
}

/// The state, taken, `Dead` left behind.
fn take<V>(s: &mut State<V>) -> State<V> {
    core::mem::replace(s, State::Dead)
}

/// `e P` for `K` curves at once, their steps interleaved, `P = (x1 : 1)` for each, with `SMALL`
/// when `x1` is 2 (then its products are doublings): the `x` and the `z` of each.
fn ladder_lanes<A: Arith, const K: usize, const SMALL: bool>(
    f: &A,
    x1: &[A::V; K],
    a24: &[A::V; K],
    e: &[u64],
    t: &mut [Tmp<A::V>; K],
) -> ([A::V; K], [A::V; K]) {
    let mut x2: [A::V; K] = core::array::from_fn(|_| f.one());
    let mut z2: [A::V; K] = core::array::from_fn(|_| f.zero());
    let mut x3: [A::V; K] = x1.clone();
    let mut z3: [A::V; K] = core::array::from_fn(|_| f.one());
    let mut swap = false;
    let bits = bit_length(e);
    for i in (0..bits).rev() {
        let b = bit(e, i);
        swap ^= b;
        for k in 0..K {
            f.cswap(swap, &mut x2[k], &mut x3[k]);
            f.cswap(swap, &mut z2[k], &mut z3[k]);
        }
        swap = b;
        step_lanes::<A, K, SMALL>(f, x1, a24, &mut x2, &mut z2, &mut x3, &mut z3, t);
    }
    for k in 0..K {
        f.cswap(swap, &mut x2[k], &mut x3[k]);
        f.cswap(swap, &mut z2[k], &mut z3[k]);
    }
    (x2, z2)
}

/// [`step`] for `K` curves, operation by operation, with the product by `x1` a doubling when
/// `SMALL` (`x1 = 2`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_lanes<A: Arith, const K: usize, const SMALL: bool>(
    f: &A,
    x1: &[A::V; K],
    a24: &[A::V; K],
    x2: &mut [A::V; K],
    z2: &mut [A::V; K],
    x3: &mut [A::V; K],
    z3: &mut [A::V; K],
    t: &mut [Tmp<A::V>; K],
) {
    // Each operation for every curve in turn, as `step` orders them.
    for k in 0..K {
        f.add(&mut t[k].a, &x2[k], &z2[k]);
    }
    for k in 0..K {
        f.sub(&mut t[k].b, &x2[k], &z2[k]);
    }
    for k in 0..K {
        f.add(&mut t[k].c, &x3[k], &z3[k]);
    }
    for k in 0..K {
        f.sub(&mut t[k].d, &x3[k], &z3[k]);
    }
    // `DA` and `CB`.
    for tk in t.iter_mut() {
        f.mul_in(&mut tk.d, &tk.a);
    }
    for tk in t.iter_mut() {
        f.mul_in(&mut tk.c, &tk.b);
    }
    // `AA` and `BB`.
    for tk in t.iter_mut() {
        f.sqr_in(&mut tk.a);
    }
    for tk in t.iter_mut() {
        f.sqr_in(&mut tk.b);
    }
    for tk in t.iter_mut() {
        f.add(&mut tk.t, &tk.d, &tk.c);
    }
    for k in 0..K {
        f.sqr(&mut x3[k], &t[k].t);
    }
    for tk in t.iter_mut() {
        f.sub(&mut tk.t, &tk.d, &tk.c);
    }
    for tk in t.iter_mut() {
        f.sqr_in(&mut tk.t);
    }
    for k in 0..K {
        if SMALL {
            f.add(&mut z3[k], &t[k].t, &t[k].t);
        } else {
            f.mul(&mut z3[k], &t[k].t, &x1[k]);
        }
    }
    for k in 0..K {
        f.mul(&mut x2[k], &t[k].a, &t[k].b);
    }
    // `E = AA - BB`, `z2 = E (BB + a24 E)`.
    for tk in t.iter_mut() {
        f.sub(&mut tk.u, &tk.a, &tk.b);
    }
    for k in 0..K {
        let tk = &mut t[k];
        f.mul(&mut tk.t, &tk.u, &a24[k]);
    }
    for tk in t.iter_mut() {
        f.add_in(&mut tk.t, &tk.b);
    }
    for k in 0..K {
        f.mul(&mut z2[k], &t[k].t, &t[k].u);
    }
}

/// A point `(x : z)`.
#[derive(Clone)]
struct Point<V> {
    x: V,
    z: V,
}

/// The working residues of the ladder's step.
struct Tmp<V> {
    a: V,
    b: V,
    c: V,
    d: V,
    t: V,
    u: V,
}

impl<V: Clone> Tmp<V> {
    fn new<A: Arith<V = V>>(f: &A) -> Self {
        Tmp {
            a: f.zero(),
            b: f.zero(),
            c: f.zero(),
            d: f.zero(),
            t: f.zero(),
            u: f.zero(),
        }
    }
}

/// `(e P, (e + 1) P)` for the affine `P = (x1 : 1)` and the exponent of the limbs `e`, by
/// Montgomery's ladder from `(O, P)`, `O = (1 : 0)`.
fn ladder<A: Arith>(
    f: &A,
    x1: &A::V,
    a24: &A::V,
    e: &[u64],
    t: &mut Tmp<A::V>,
) -> (Point<A::V>, Point<A::V>) {
    let (mut x2, mut z2) = (f.one(), f.zero());
    let (mut x3, mut z3) = (x1.clone(), f.one());
    let mut swap = false;
    let bits = bit_length(e);
    for i in (0..bits).rev() {
        let b = bit(e, i);
        swap ^= b;
        f.cswap(swap, &mut x2, &mut x3);
        f.cswap(swap, &mut z2, &mut z3);
        swap = b;
        step(f, x1, a24, &mut x2, &mut z2, &mut x3, &mut z3, t);
    }
    f.cswap(swap, &mut x2, &mut x3);
    f.cswap(swap, &mut z2, &mut z3);
    (Point { x: x2, z: z2 }, Point { x: x3, z: z3 })
}

/// The ladder's step: `(x2 : z2) <- 2 (x2 : z2)` and `(x3 : z3) <- (x2 : z2) + (x3 : z3)`, whose
/// difference is `(x1 : 1)`, in 4 squares and 6 products (the formulas of RFC 7748, with
/// `(A + 2) / 4`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step<A: Arith>(
    f: &A,
    x1: &A::V,
    a24: &A::V,
    x2: &mut A::V,
    z2: &mut A::V,
    x3: &mut A::V,
    z3: &mut A::V,
    t: &mut Tmp<A::V>,
) {
    f.add(&mut t.a, x2, z2);
    f.sub(&mut t.b, x2, z2);
    f.add(&mut t.c, x3, z3);
    f.sub(&mut t.d, x3, z3);
    // `DA` and `CB`.
    f.mul_in(&mut t.d, &t.a);
    f.mul_in(&mut t.c, &t.b);
    // `AA` and `BB`.
    f.sqr_in(&mut t.a);
    f.sqr_in(&mut t.b);
    f.add(&mut t.t, &t.d, &t.c);
    f.sqr(x3, &t.t);
    f.sub(&mut t.t, &t.d, &t.c);
    f.sqr_in(&mut t.t);
    f.mul(z3, &t.t, x1);
    f.mul(x2, &t.a, &t.b);
    // `E = AA - BB`, `z2 = E (BB + a24 E)`.
    f.sub(&mut t.u, &t.a, &t.b);
    f.mul(&mut t.t, &t.u, a24);
    f.add_in(&mut t.t, &t.b);
    f.mul(z2, &t.t, &t.u);
}

/// `2 P`.
fn double<A: Arith>(f: &A, p: &Point<A::V>, a24: &A::V, t: &mut Tmp<A::V>) -> Point<A::V> {
    f.add(&mut t.a, &p.x, &p.z);
    f.sqr_in(&mut t.a);
    f.sub(&mut t.b, &p.x, &p.z);
    f.sqr_in(&mut t.b);
    let mut x = f.zero();
    f.mul(&mut x, &t.a, &t.b);
    f.sub(&mut t.u, &t.a, &t.b);
    f.mul(&mut t.t, &t.u, a24);
    f.add_in(&mut t.t, &t.b);
    let mut z = f.zero();
    f.mul(&mut z, &t.t, &t.u);
    Point { x, z }
}

/// `P + Q` from `P - Q = D`.
fn add<A: Arith>(
    f: &A,
    p: &Point<A::V>,
    q: &Point<A::V>,
    d: &Point<A::V>,
    t: &mut Tmp<A::V>,
) -> Point<A::V> {
    f.add(&mut t.a, &p.x, &p.z);
    f.sub(&mut t.b, &p.x, &p.z);
    f.add(&mut t.c, &q.x, &q.z);
    f.sub(&mut t.d, &q.x, &q.z);
    f.mul_in(&mut t.d, &t.a);
    f.mul_in(&mut t.c, &t.b);
    f.add(&mut t.t, &t.d, &t.c);
    f.sqr_in(&mut t.t);
    let mut x = f.zero();
    f.mul(&mut x, &t.t, &d.z);
    f.sub(&mut t.t, &t.d, &t.c);
    f.sqr_in(&mut t.t);
    let mut z = f.zero();
    f.mul(&mut z, &t.t, &d.x);
    Point { x, z }
}

/// Stage 1 over the chunk again from the affine `x`, a prime at a time.
fn backtrack<A: Arith>(
    f: &A,
    mut x: A::V,
    a24: &A::V,
    chunk: &Chunk,
    b1: u64,
    t: &mut Tmp<A::V>,
) -> Option<Vec<u64>> {
    for p in primes_in(chunk.lo, chunk.hi) {
        let mut q = prime_power(p, b1);
        while q > 1 {
            let (r, _) = ladder(f, &x, a24, &[p], t);
            q /= p;
            match invert(f, &r.z) {
                Ok(zi) => f.mul(&mut x, &r.x, &zi),
                Err(g) => return proper(f, g),
            }
        }
    }
    None
}

/// The accumulators turned by one: the first, just multiplied, goes last.
#[inline(always)]
pub(super) fn rotate<V>(acc: &mut [V; 4]) {
    let [a0, a1, a2, a3] = acc;
    core::mem::swap(a0, a1);
    core::mem::swap(a1, a2);
    core::mem::swap(a2, a3);
}

/// The product of the four accumulators, in the first, the others set to 1.
pub(super) fn combine<'a, A: Arith>(f: &A, acc: &'a mut [A::V; 4]) -> &'a A::V {
    let [a0, a1, a2, a3] = acc;
    f.mul_in(a0, a1);
    f.mul_in(a2, a3);
    f.mul_in(a0, a2);
    let one = f.one();
    a1.clone_from(&one);
    a2.clone_from(&one);
    a3.clone_from(&one);
    &acc[0]
}

/// Stage 2 from the affine `x` of `Q`: the continuation by fast polynomial arithmetic when it costs
/// less (see [`ecmfft`](super::ecmfft)), otherwise, or when its product is 0 modulo `n`, the
/// baby-step giant-step one.
fn stage2<A: Arith>(
    f: &A,
    x: &A::V,
    a24: &A::V,
    plan: &Plan,
    t: &mut Tmp<A::V>,
) -> Option<Vec<u64>> {
    /// The continuation by polynomials: `None` for a product 0 modulo `n`, or steps that cannot be
    /// taken to affine modulo every prime of `n` at once, to be gone over by the other, else the
    /// result. Its blocks of giant steps are of `d` steps, up to 77,760, against the other's 128,
    /// and its greatest common divisor comes at the end: the other may still tell the primes apart.
    fn polynomials<A: Arith>(
        f: &A,
        x: &A::V,
        a24: &A::V,
        fp: &FftPlan,
        t: &mut Tmp<A::V>,
    ) -> Option<Option<Vec<u64>>> {
        let half = fp.d / 2;
        let q1 = Point {
            x: x.clone(),
            z: f.one(),
        };
        let q2 = double(f, &q1, a24, t);
        // The baby steps `j Q`, odd `j < D/2` prime to `D`, and `D Q`.
        let mut xs: Vec<A::V> = Vec::with_capacity(fp.js.len());
        let mut zs: Vec<A::V> = Vec::with_capacity(fp.js.len() + 1);
        let (mut prev, mut cur) = (q1.clone(), q1.clone());
        let (mut below, mut at) = (q1.clone(), q1.clone());
        let mut next_j = 0;
        let mut j = 1;
        loop {
            if next_j < fp.js.len() && fp.js[next_j] == j {
                xs.push(cur.x.clone());
                zs.push(cur.z.clone());
                next_j += 1;
            }
            if j + 1 == half {
                below = cur.clone();
            }
            if j == half {
                at = cur.clone();
            }
            if j > half {
                break;
            }
            let next = add(f, &cur, &q2, &prev, t);
            prev = core::mem::replace(&mut cur, next);
            j += 2;
        }
        let dq = if half % 2 == 1 {
            double(f, &at, a24, t)
        } else {
            add(f, &cur, &below, &q2, t)
        };
        zs.push(dq.z.clone());
        if let Err(g) = invert_all(f, &mut zs) {
            return handed(f, g);
        }
        let dzi = zs.pop().expect("the inverse of D Q's z");
        for (xi, zi) in xs.iter_mut().zip(&zs) {
            f.mul_in(xi, zi);
        }
        drop(zs);
        let mut xd = f.zero();
        f.mul(&mut xd, &dq.x, &dzi);
        let r = Point {
            x: xd.clone(),
            z: f.one(),
        };
        let mut eval = Eval::new(f, &xs);
        drop(xs);
        // The giant steps from `k_first D Q`, a block of `d` of them at a time, made affine.
        let d = eval.degree();
        let (mut g0, mut g1) = ladder(f, &xd, a24, &[fp.k_first], t);
        let mut gx: Vec<A::V> = Vec::with_capacity(d);
        let mut gz: Vec<A::V> = Vec::with_capacity(d);
        for _ in 0..fp.blocks {
            gx.clear();
            gz.clear();
            for _ in 0..d {
                gx.push(g0.x.clone());
                gz.push(g0.z.clone());
                let next = add(f, &g1, &r, &g0, t);
                g0 = core::mem::replace(&mut g1, next);
            }
            if let Err(g) = invert_all(f, &mut gz) {
                return handed(f, g);
            }
            for (xi, zi) in gx.iter_mut().zip(&gz) {
                f.mul_in(xi, zi);
            }
            eval.block(&gx);
        }
        match eval.finish() {
            Gcd::One => Some(None),
            Gcd::Factor(g) => Some(Some(g)),
            Gcd::Whole => None,
        }
    }

    /// The factor of a failed inversion of [`polynomials`], or `None` when it is `n`, for the
    /// baby-step giant-step continuation.
    fn handed<A: Arith>(f: &A, g: Vec<u64>) -> Option<Option<Vec<u64>>> {
        if g == f.n() { None } else { Some(Some(g)) }
    }

    if let Some(fp) = plan.ecm_fft(f.n().len())
        && let Some(found) = polynomials(f, x, a24, &fp, t)
    {
        return found;
    }
    let half = plan.d / 2;
    let q1 = Point {
        x: x.clone(),
        z: f.one(),
    };
    let q2 = double(f, &q1, a24, t);
    // The baby steps: `j Q` for the odd `j` up to `D/2 + 1`.
    let mut xs: Vec<A::V> = Vec::with_capacity(plan.js.len());
    let mut zs: Vec<A::V> = Vec::with_capacity(plan.js.len());
    let (mut prev, mut cur) = (q1.clone(), q1.clone());
    let (mut below, mut at) = (q1.clone(), q1.clone());
    let mut j = 1;
    loop {
        if j < half && plan.index_of(j) {
            xs.push(cur.x.clone());
            zs.push(cur.z.clone());
        }
        if j + 1 == half {
            below = cur.clone();
        }
        if j == half {
            at = cur.clone();
        }
        if j > half {
            break;
        }
        let next = add(f, &cur, &q2, &prev, t);
        prev = core::mem::replace(&mut cur, next);
        j += 2;
    }
    let dq = if half % 2 == 1 {
        double(f, &at, a24, t)
    } else {
        add(f, &cur, &below, &q2, t)
    };
    // The baby steps and `D Q` affine, together.
    zs.push(dq.z.clone());
    if let Err(g) = invert_all(f, &mut zs) {
        return proper(f, g);
    }
    let dzi = zs.pop().expect("the inverse of D Q's z");
    for (xi, zi) in xs.iter_mut().zip(&zs) {
        f.mul_in(xi, zi);
    }
    drop(zs);
    let mut xd = f.zero();
    f.mul(&mut xd, &dq.x, &dzi);
    let r = Point {
        x: xd.clone(),
        z: f.one(),
    };
    // The giant steps from `k_first D Q`.
    let (mut g0, mut g1) = ladder(f, &xd, a24, &[plan.k_first], t);
    // Four products accumulated in turn, so that their chains overlap.
    let mut acc = [f.one(), f.one(), f.one(), f.one()];
    let mut d = f.zero();
    let mut bitmap = Vec::new();
    let mut gz: Vec<A::V> = Vec::with_capacity(BLOCK as usize);
    let mut gx: Vec<A::V> = Vec::with_capacity(BLOCK as usize);
    let mut k = plan.k_first;
    while k <= plan.k_last {
        let count = BLOCK.min(plan.k_last - k + 1);
        plan.block(k, count, &mut bitmap);
        gx.clear();
        gz.clear();
        for _ in 0..count {
            gx.push(g0.x.clone());
            gz.push(g0.z.clone());
            let next = add(f, &g1, &r, &g0, t);
            g0 = core::mem::replace(&mut g1, next);
        }
        if let Err(g) = invert_all(f, &mut gz) {
            return proper(f, g);
        }
        for (xi, zi) in gx.iter_mut().zip(&gz) {
            f.mul_in(xi, zi);
        }
        for s in 0..count as usize {
            for w in 0..plan.words {
                let mut bits = bitmap[s * plan.words + w];
                while bits != 0 {
                    let i = 64 * w + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    f.sub(&mut d, &gx[s], &xs[i]);
                    f.mul_in(&mut acc[0], &d);
                    rotate(&mut acc);
                }
            }
        }
        let product = combine(f, &mut acc);
        match gcd(f, product) {
            Gcd::One => {}
            Gcd::Factor(g) => return Some(g),
            Gcd::Whole => {
                for s in 0..count as usize {
                    for w in 0..plan.words {
                        let mut bits = bitmap[s * plan.words + w];
                        while bits != 0 {
                            let i = 64 * w + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            f.sub(&mut d, &gx[s], &xs[i]);
                            match gcd(f, &d) {
                                Gcd::One => {}
                                Gcd::Factor(g) => return Some(g),
                                Gcd::Whole => return None,
                            }
                        }
                    }
                }
                return None;
            }
        }
        k += count;
    }
    None
}
