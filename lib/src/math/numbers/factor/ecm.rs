//! Lenstra's elliptic curve method (1987) on Montgomery's curves `B y^2 = x^3 + A x^2 + x` (1987),
//! with Suyama's parametrisation (1985), as GMP-ECM's `-param 0`: for `sigma >= 6`,
//! `u = sigma^2 - 5`, `v = 4 sigma`, the point `(u^3 : v^3)` and
//! `(A + 2) / 4 = (v - u)^3 (3u + v) / (16 u^3 v)`, whose group orders modulo every prime are
//! multiples of 12. The curves are those of `sigma = 6, 7, 8, ...` in turn, so that a
//! factorisation is reproducible; they are GMP-ECM's `-sigma 0:s`, whose stage 1 finds the same
//! factors, and a few more that its PRAC chains find by chance.
//!
//! The arithmetic is on `x` and `z` alone: the doubling `2 (x : z)` and the addition of two points
//! whose difference is known take 5 and 6 products, and Montgomery's ladder computes `e P` with
//! one of each per bit of `e`, `(R0, R1) -> (2 R0, R0 + R1)` or `(R0 + R1, 2 R1)`, `R1 - R0 = P`
//! all along. With `P` affine, `z = 1`, the pair costs 4 squares and 6 products; the bit decides
//! only which point is doubled, by a swap without a branch.
//!
//! - Stage 1: `Q = E P` through the chunks of the exponent `E` (see [`stage1`](super::stage1)),
//!   the point taken to affine after each by one inversion, which a factor of `n` common to `z`
//!   makes impossible: its greatest common divisor is the factor. When it is `n`, the chunk is gone
//!   over again a prime at a time.
//! - Stage 2: the continuation of [`stage2`](super::stage2) on `x`: the baby steps `j Q` by the
//!   chain `(j + 2) Q = j Q + 2 Q`, the giant steps `k D Q` by `(k + 1) D Q = k D Q + D Q`, both
//!   taken to affine together by Montgomery's trick, a block of giant steps at a time, so that each
//!   pair `(k, j)` costs one product, `x_(kD) - x_j`.

use super::field::{Arith, Job, invert, invert_all, mul_cold, mul_in_cold};
use super::pm1::{bit, bit_length, proper};
use super::primes::primes_in;
use super::rho::{Gcd, gcd};
use super::stage1::{Chunk, Stage1, prime_power};
use super::stage2::{BLOCK, Plan};

/// The first `sigma` of the curves.
pub(super) const FIRST_SIGMA: u64 = 6;

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
        // In registers, up to eight limbs, two curves go through stage 1 together, their steps
        // interleaved: a product waits for the ones before it, and the other curve's fill the
        // wait. Decided by the type, so that the pairs are compiled for those arithmetics alone.
        let pairs = !A::MEMORY;
        let mut i = 0;
        while i < self.count {
            let sigma = self.first + i;
            let found = if pairs && i + 1 < self.count {
                i += 2;
                curve_pair(f, sigma, self.stage1, self.plan)
            } else {
                i += 1;
                curve(f, sigma, self.stage1, self.plan)
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }
}

/// A curve in stage 1.
enum State<V> {
    /// Going on, from the affine `x`, with `(A + 2) / 4`.
    Alive(V, V),
    /// Done, with a factor.
    Found(Vec<u64>),
    /// Done, with nothing: every prime was caught at once.
    Dead,
}

/// The curve of `sigma` set up for stage 1.
fn start<A: Arith>(f: &A, sigma: u64) -> State<A::V> {
    match suyama(f, sigma) {
        Ok((x, a24)) => State::Alive(x, a24),
        Err(g) => match proper(f, g) {
            Some(g) => State::Found(g),
            None => State::Dead,
        },
    }
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

/// The curves of `sigma` and `sigma + 1`, stage 1 together and stage 2 each, with the result the
/// two would give one after the other: the first's factor, from either stage, before the
/// second's.
fn curve_pair<A: Arith>(
    f: &A,
    sigma: u64,
    stage1: &Stage1,
    plan: Option<&Plan>,
) -> Option<Vec<u64>> {
    let mut a = start(f, sigma);
    if let State::Found(g) = a {
        return Some(g);
    }
    let mut b = start(f, sigma + 1);
    let b1 = stage1.b1;
    let mut ta = Tmp::new(f);
    let mut tb = Tmp::new(f);
    stage1.each(|chunk| {
        let (na, nb) = match (&a, &b) {
            (State::Alive(xa, ka), State::Alive(xb, kb)) => {
                let (qa, qb) = ladder_pair(f, [xa, xb], [ka, kb], &chunk.e, &mut ta, &mut tb);
                let mut zs = [qa.z.clone(), qb.z.clone()];
                if invert_all(f, &mut zs).is_ok() {
                    let (mut ya, mut yb) = (f.zero(), f.zero());
                    f.mul(&mut ya, &qa.x, &zs[0]);
                    f.mul(&mut yb, &qb.x, &zs[1]);
                    (State::Alive(ya, ka.clone()), State::Alive(yb, kb.clone()))
                } else {
                    (
                        settle(f, xa, ka.clone(), &qa, chunk, b1, &mut ta),
                        settle(f, xb, kb.clone(), &qb, chunk, b1, &mut tb),
                    )
                }
            }
            (State::Alive(xa, ka), _) => {
                let (qa, _) = ladder(f, xa, ka, &chunk.e, &mut ta);
                (
                    settle(f, xa, ka.clone(), &qa, chunk, b1, &mut ta),
                    take(&mut b),
                )
            }
            (_, State::Alive(xb, kb)) => {
                let (qb, _) = ladder(f, xb, kb, &chunk.e, &mut tb);
                (
                    take(&mut a),
                    settle(f, xb, kb.clone(), &qb, chunk, b1, &mut tb),
                )
            }
            _ => return false,
        };
        a = na;
        b = nb;
        // The first curve's factor ends both; otherwise on while one goes on.
        !matches!(a, State::Found(_))
            && (matches!(a, State::Alive(..)) || matches!(b, State::Alive(..)))
    });
    match a {
        State::Found(g) => return Some(g),
        State::Alive(x, a24) => {
            if let Some(plan) = plan
                && let Some(g) = stage2(f, &x, &a24, plan, &mut ta)
            {
                return Some(g);
            }
        }
        State::Dead => {}
    }
    match b {
        State::Found(g) => Some(g),
        State::Alive(x, a24) => stage2(f, &x, &a24, plan?, &mut tb),
        State::Dead => None,
    }
}

/// The state, taken, `Dead` left behind.
fn take<V>(s: &mut State<V>) -> State<V> {
    core::mem::replace(s, State::Dead)
}

/// [`ladder`] for two curves at once, their steps interleaved.
fn ladder_pair<A: Arith>(
    f: &A,
    x1: [&A::V; 2],
    a24: [&A::V; 2],
    e: &[u64],
    ta: &mut Tmp<A::V>,
    tb: &mut Tmp<A::V>,
) -> (Point<A::V>, Point<A::V>) {
    let (mut x2a, mut z2a) = (f.one(), f.zero());
    let (mut x3a, mut z3a) = (x1[0].clone(), f.one());
    let (mut x2b, mut z2b) = (f.one(), f.zero());
    let (mut x3b, mut z3b) = (x1[1].clone(), f.one());
    let mut swap = false;
    let bits = bit_length(e);
    for i in (0..bits).rev() {
        let bt = bit(e, i);
        swap ^= bt;
        f.cswap(swap, &mut x2a, &mut x3a);
        f.cswap(swap, &mut z2a, &mut z3a);
        f.cswap(swap, &mut x2b, &mut x3b);
        f.cswap(swap, &mut z2b, &mut z3b);
        swap = bt;
        step_pair(
            f,
            x1,
            a24,
            [&mut x2a, &mut x2b],
            [&mut z2a, &mut z2b],
            [&mut x3a, &mut x3b],
            [&mut z3a, &mut z3b],
            ta,
            tb,
        );
    }
    f.cswap(swap, &mut x2a, &mut x3a);
    f.cswap(swap, &mut z2a, &mut z3a);
    f.cswap(swap, &mut x2b, &mut x3b);
    f.cswap(swap, &mut z2b, &mut z3b);
    (Point { x: x2a, z: z2a }, Point { x: x2b, z: z2b })
}

/// [`step`] for two curves, operation by operation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_pair<A: Arith>(
    f: &A,
    x1: [&A::V; 2],
    a24: [&A::V; 2],
    x2: [&mut A::V; 2],
    z2: [&mut A::V; 2],
    x3: [&mut A::V; 2],
    z3: [&mut A::V; 2],
    ta: &mut Tmp<A::V>,
    tb: &mut Tmp<A::V>,
) {
    let [x2a, x2b] = x2;
    let [z2a, z2b] = z2;
    let [x3a, x3b] = x3;
    let [z3a, z3b] = z3;
    f.add(&mut ta.a, x2a, z2a);
    f.add(&mut tb.a, x2b, z2b);
    f.sub(&mut ta.b, x2a, z2a);
    f.sub(&mut tb.b, x2b, z2b);
    f.add(&mut ta.c, x3a, z3a);
    f.add(&mut tb.c, x3b, z3b);
    f.sub(&mut ta.d, x3a, z3a);
    f.sub(&mut tb.d, x3b, z3b);
    f.mul_in(&mut ta.d, &ta.a);
    f.mul_in(&mut tb.d, &tb.a);
    f.mul_in(&mut ta.c, &ta.b);
    f.mul_in(&mut tb.c, &tb.b);
    f.sqr_in(&mut ta.a);
    f.sqr_in(&mut tb.a);
    f.sqr_in(&mut ta.b);
    f.sqr_in(&mut tb.b);
    f.add(&mut ta.t, &ta.d, &ta.c);
    f.add(&mut tb.t, &tb.d, &tb.c);
    f.sqr(x3a, &ta.t);
    f.sqr(x3b, &tb.t);
    f.sub(&mut ta.t, &ta.d, &ta.c);
    f.sub(&mut tb.t, &tb.d, &tb.c);
    f.sqr_in(&mut ta.t);
    f.sqr_in(&mut tb.t);
    f.mul(z3a, &ta.t, x1[0]);
    f.mul(z3b, &tb.t, x1[1]);
    f.mul(x2a, &ta.a, &ta.b);
    f.mul(x2b, &tb.a, &tb.b);
    f.sub(&mut ta.u, &ta.a, &ta.b);
    f.sub(&mut tb.u, &tb.a, &tb.b);
    f.mul(&mut ta.t, &ta.u, a24[0]);
    f.mul(&mut tb.t, &tb.u, a24[1]);
    f.add_in(&mut ta.t, &ta.b);
    f.add_in(&mut tb.t, &tb.b);
    f.mul(z2a, &ta.t, &ta.u);
    f.mul(z2b, &tb.t, &tb.u);
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

/// One curve, of `sigma`: a factor, or `None`.
fn curve<A: Arith>(f: &A, sigma: u64, stage1: &Stage1, plan: Option<&Plan>) -> Option<Vec<u64>> {
    let (mut x, a24) = match suyama(f, sigma) {
        Ok(c) => c,
        Err(g) => return proper(f, g),
    };
    let b1 = stage1.b1;
    let mut result: Option<Option<Vec<u64>>> = None;
    let mut tmp = Tmp::new(f);
    stage1.each(|chunk| {
        let (q, _) = ladder(f, &x, &a24, &chunk.e, &mut tmp);
        match invert(f, &q.z) {
            Ok(zi) => {
                f.mul(&mut x, &q.x, &zi);
                true
            }
            Err(g) => {
                result = Some(if g == f.n() {
                    backtrack(f, x.clone(), &a24, chunk, b1, &mut tmp)
                } else {
                    Some(g)
                });
                false
            }
        }
    });
    if let Some(r) = result {
        return r;
    }
    stage2(f, &x, &a24, plan?, &mut tmp)
}

/// The affine `x` of Suyama's point and `(A + 2) / 4` for `sigma`, from one inversion of
/// `16 u^3 v^4`; or the greatest common divisor that makes it impossible.
fn suyama<A: Arith>(f: &A, sigma: u64) -> Result<(A::V, A::V), Vec<u64>> {
    let s = f.small(sigma);
    let mut u = f.zero();
    mul_cold(f, &mut u, &s, &s);
    f.sub_in(&mut u, &f.small(5));
    let mut v = f.zero();
    f.add(&mut v, &s, &s);
    let w = v.clone();
    f.add_in(&mut v, &w);
    let mut u3 = f.zero();
    mul_cold(f, &mut u3, &u, &u);
    mul_in_cold(f, &mut u3, &u);
    let mut v3 = f.zero();
    mul_cold(f, &mut v3, &v, &v);
    mul_in_cold(f, &mut v3, &v);
    // `t = 16 u^3 v`, the denominator `t v^3`.
    let mut t = f.zero();
    mul_cold(f, &mut t, &u3, &v);
    for _ in 0..4 {
        let w = t.clone();
        f.add_in(&mut t, &w);
    }
    let mut den = f.zero();
    mul_cold(f, &mut den, &t, &v3);
    let inv = invert(f, &den)?;
    // `x = u^3 / v^3 = u^3 t / (t v^3)`.
    let mut x = f.zero();
    mul_cold(f, &mut x, &u3, &t);
    mul_in_cold(f, &mut x, &inv);
    // `(A + 2) / 4 = (v - u)^3 (3u + v) v^3 / (t v^3)`.
    let mut w = f.zero();
    f.sub(&mut w, &v, &u);
    let mut a24 = f.zero();
    mul_cold(f, &mut a24, &w, &w);
    mul_in_cold(f, &mut a24, &w);
    let mut z = f.zero();
    f.add(&mut z, &u, &u);
    f.add_in(&mut z, &u);
    f.add_in(&mut z, &v);
    mul_in_cold(f, &mut a24, &z);
    mul_in_cold(f, &mut a24, &v3);
    mul_in_cold(f, &mut a24, &inv);
    Ok((x, a24))
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

/// Stage 2 from the affine `x` of `Q`.
fn stage2<A: Arith>(
    f: &A,
    x: &A::V,
    a24: &A::V,
    plan: &Plan,
    t: &mut Tmp<A::V>,
) -> Option<Vec<u64>> {
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
