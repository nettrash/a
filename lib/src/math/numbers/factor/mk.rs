//! The second stage of `p - 1` by Montgomery and Kruppa's fast polynomial arithmetic ("An FFT
//! extension to the P - 1 factoring algorithm", 2010), as GMP-ECM's `pm1fs2.c` does it.
//!
//! For the residue `b` of the first stage, an odd `P` made of small primes and symmetric sets
//! `S_1` and `S_2` whose sums `k_1 + k_2` give each class prime to `P` once, the polynomial
//! `f(x) = prod_(k_1 in S_1) (x - b^(2 k_1))` is evaluated at `X_m = b^(2 k_2 + (2m + 1) P)` for
//! every `k_2` in `S_2` and `m` from `m_1` to `m_1 + nr - 1`: `f(X_m)` has the factor
//! `b^(2 k_1) (b^((2m + 1) P + 2 (k_2 - k_1)) - 1)` for each `k_1`, so the product of all the
//! values is 0 modulo a prime `p` of `n` when the order of `b` modulo `p` divides one of the
//! exponents `(2m + 1) P + 2k`, `k` in `S_1 + S_2`: every odd number prime to `P` from
//! `(2 m_1 + 1) P + 2 max(S)` to `(2 (m_1 + nr) - 1) P - 2 max(S)`, which holds `(B1, B2]`.
//!
//! The points `X_m` for one `k_2` are a geometric progression, `X_(m+1) = X_m r^2` with `r = b^P`,
//! and `f` is reciprocal, `f(x) = x^(s_1/2) sum_j f_j x^j` with `f_j = f_(-j)`: with
//! `2ij = (i + j)^2 - i^2 - j^2`, `F(x_0 r^(2i)) = x_0^(-i) r^(-i^2) sum_j h_j g_(i+j)` for
//! `h_j = f_j r^(-j^2)` and `g_t = x_0^t r^(t^2)`, so that the values at all `nr` points come out
//! of one product of polynomials (see [`poly`](super::poly)), the middle of the product of `h`,
//! of `s_1 + 1` coefficients, by `g`, of `l = s_1 + nr`; the factors `x_0^(-i) r^(-i^2)` are
//! units, left out.
//!
//! `S_1` and `S_2` are sums of arithmetic progressions of prime lengths ("atoms"): the classes
//! prime to a prime `q` of `P` are `(P/q) {-(q - 2), ..., -1, 1, ..., q - 2}`, odd numbers, whose
//! progression of length `q - 1` is the sum of progressions of the prime lengths of the factors of
//! `q - 1`. `f` is made from them, `f_(A+T)(x) = prod_(t in T) f_A(x b^(-2t))` up to units, the
//! longest atoms first ([`build_f`]). `P`, the split of the atoms and `nr` are those of the least
//! estimated time ([`MkPlan::new`]).
//!
//! For `p + 1` ([`pp1_stage2`]) `b` is the root `beta` of norm 1 of `X^2 - V X + 1` in
//! `Z_n[sqrt(Delta)]`, `Delta = V^2 - 4`: `f`'s coefficients are rational, `h` and `g` have two
//! coordinates, and of each value only the rational one is formed.

use super::field::{Arith, invert};
use super::pm1::{power, proper};
use super::poly::Kron;
use super::rho::{Gcd, gcd};
use std::sync::OnceLock;

/// The odd primes `P` is made of.
const PRIMES: [u64; 9] = [3, 5, 7, 11, 13, 17, 19, 23, 29];

/// An arithmetic progression of a prime length, symmetric about 0: `start + i step` for `i` below
/// `len`, `start = -(len - 1) step / 2`.
#[derive(Clone, Copy, Debug)]
struct Atom {
    /// The first element.
    start: i64,
    /// The step.
    step: i64,
    /// The length, a prime.
    len: u64,
}

/// The parameters of the stage for its bounds.
#[derive(Clone, Debug)]
pub(super) struct MkPlan {
    /// `P`, odd and square-free.
    p: u64,
    /// The atoms of `S_1` and of `S_2`.
    atoms1: Vec<Atom>,
    atoms2: Vec<Atom>,
    /// `|S_1|`, even, and `|S_2|`.
    s1: usize,
    s2: usize,
    /// The points of each evaluation, `nr`, and the first `m`.
    nr: usize,
    m1: i64,
    /// The estimated time, in the nanoseconds of [`product_ns`].
    pub(super) cost: f64,
}

/// The atoms of the classes prime to `P`: for each prime `q` of `P`, the progression `(P/q)
/// {-(q - 2), ..., q - 2}` of step `2 P/q`, as the sum of progressions of prime lengths.
fn atoms(p: u64) -> Vec<Atom> {
    let mut out = Vec::new();
    for &q in PRIMES.iter().filter(|&&q| p.is_multiple_of(q)) {
        // `R_(q - 1)` times `P/q`: the step-2 progression of length `q - 1`, split as
        // `R_(ab) = R_a + a R_b`.
        let mut mult = (2 * (p / q)) as i64;
        let mut rest = q - 1;
        let mut f = 2;
        while rest > 1 {
            while rest % f == 0 {
                let len = f;
                out.push(Atom {
                    start: -(mult * (len as i64 - 1)) / 2,
                    step: mult,
                    len,
                });
                mult *= len as i64;
                rest /= f;
            }
            f += 1;
        }
    }
    out
}

/// The largest element of the sum of the atoms.
fn atoms_max(a: &[Atom]) -> i64 {
    a.iter().map(|t| -t.start).sum()
}

/// Euler's totient of the odd square-free `p` over [`PRIMES`].
fn phi(p: u64) -> u64 {
    PRIMES
        .iter()
        .filter(|&&q| p.is_multiple_of(q))
        .map(|q| q - 1)
        .product()
}

/// The time of a product of residues of `k` limbs, in nanoseconds on an Apple M4 Max: in registers up
/// to 8 limbs, in memory beyond (measured as the squares of `p - 1`'s first stage). The costs below
/// are in these units; only their ratios decide.
pub(super) fn product_ns(k: usize) -> f64 {
    let k = k as f64;
    if k <= 8.0 {
        3.0 + 0.8 * k * k
    } else {
        40.0 + 0.45 * k * k
    }
}

/// The time of a product of polynomials of `la` and `lb` coefficients of `k` limbs by [`Kron`], and
/// the reduction of `outs` of its coefficients, in nanoseconds: the numbers' limbs at 15 to 25 ns
/// each, the reductions at about a product each.
fn kron_ns(k: usize, la: usize, lb: usize, outs: usize) -> f64 {
    let limbs = (la + lb) * (2 * k + 1);
    let per_limb = 15.0 + f64::from((limbs / 1000).max(1).ilog2());
    let limbs = limbs as f64;
    let redc = 10.0 + 0.4 * ((k + 1) * (k + 1)) as f64;
    limbs * per_limb + outs as f64 * redc
}

/// A candidate for `P`: the square-free odd products of [`PRIMES`], in increasing order, with the
/// numbers the choice needs.
struct Cand {
    /// `P`.
    p: u64,
    /// Its largest prime.
    top: u64,
    /// `phi(P)`.
    phi: u64,
    /// The largest element of the sum of its atoms.
    max: i64,
    /// The lengths of its atoms, increasing.
    lens: Vec<u8>,
}

/// The candidates for `P`, made once: 511 of them, about 20 KB.
fn candidates() -> &'static [Cand] {
    static CANDS: OnceLock<Vec<Cand>> = OnceLock::new();
    CANDS.get_or_init(|| {
        let mut v = Vec::new();
        for mask in 1u32..1 << PRIMES.len() {
            let (mut p, mut top) = (1u64, 0);
            for (i, &q) in PRIMES.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    p *= q;
                    top = q;
                }
            }
            let all = atoms(p);
            let mut lens: Vec<u8> = all.iter().map(|a| a.len as u8).collect();
            lens.sort_unstable();
            v.push(Cand {
                p,
                top,
                phi: phi(p),
                max: atoms_max(&all),
                lens,
            });
        }
        v.sort_by_key(|c| c.p);
        v
    })
}

impl MkPlan {
    /// The parameters that cost least for the bounds `b1 < b2` and a modulus of `k` limbs, or
    /// `None` when no `P` suits.
    pub(super) fn new(b1: u64, b2: u64, k: usize, pp1: bool) -> Option<MkPlan> {
        // The exponents are `i64`s: bounds past `2^62` take the other continuation.
        if b2 <= b1 || b1 < 3 || b2 > 1 << 62 {
            return None;
        }
        let mm = product_ns(k);
        let chained = if k <= 4 { 0.55 } else { 0.95 } * mm;
        // A lower bound of each candidate's cost, then the candidates in its order, each with the
        // divisors near its best `s_1`, until the bound passes the best cost.
        let mut cands: Vec<(f64, &Cand, usize, i64)> = Vec::with_capacity(512);
        // At least 16 points an evaluation: `2 P nr` covers the range.
        let largest = ((b2 - b1) / 32).max(15);
        let all = candidates();
        let end = all.partition_point(|c| c.p <= largest);
        for c in &all[..end] {
            // The primes of `P` must be in the first stage.
            if c.top > b1 {
                continue;
            }
            let p = c.p as i64;
            // `m_1`: the first exponent fully covered, `(2 m_1 + 1) P + 2 max`, at most `B1 + 1`;
            // the last, `(2 (m_1 + nr) - 1) P - 2 max`, at least `B2`.
            let m1 = (b1 as i128 + 1 - i128::from(p) - 2 * i128::from(c.max))
                .div_euclid(2 * i128::from(p));
            let top = (b2 as i128 + 2 * i128::from(c.max) + 3 * i128::from(p) - 1)
                .div_euclid(2 * i128::from(p));
            // In 64 bits first: on 32-bit targets `usize` holds only what passes.
            let nr = (top - m1).max(1) as u128;
            if (nr + 2) * (2 * k as u128 + 1) > MAX_LIMBS as u128 {
                continue;
            }
            let (nr, m1) = (nr as usize, m1 as i64);
            // At least `g` for all of `phi`, and for each point a product and its part of the
            // product of polynomials.
            let bound =
                (2.0 * c.phi as f64 * chained).max(nr as f64 * mm + kron_ns(k, 2, nr + 2, nr));
            cands.push((bound, c, nr, m1));
        }
        // The candidate of the least bound first, then those whose bounds beat its cost, in order.
        let first = cands
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.0.total_cmp(&b.1.0))?
            .0;
        cands.swap(0, first);
        let mut ds: Vec<u64> = Vec::with_capacity(1024);
        // (cost, P, s_1, nr, m_1)
        let mut best: Option<(f64, u64, usize, usize, i64)> = None;
        let (mut i, mut sorted) = (0, false);
        while i < cands.len() {
            let (bound, c, nr, m1) = cands[i];
            i += 1;
            if best.is_some_and(|b| bound >= b.0) {
                continue;
            }
            let phi = c.phi as usize;
            // `cost ~ phi a / s_1 + c s_1`: `a` an evaluation's part that does not grow with `s_1`,
            // `c` what `f` and `h` cost a coefficient.
            let a = 2.0 * nr as f64 * chained + kron_ns(k, 1, nr, nr) + nr as f64 * mm;
            let per = kron_ns(k, 1, 1, 2) + 7.0 * mm;
            let target = (phi as f64 * a / per).sqrt().max(2.0);
            divisors(&c.lens, &mut ds);
            for &s1 in &ds {
                let s1 = s1 as usize;
                let x = s1 as f64;
                if s1 % 2 == 1
                    || nr + s1 > MAX_LIMBS / (2 * k + 1)
                    || x < target / 4.0
                    || x > target * 4.0
                {
                    continue;
                }
                let cost = cost(k, s1, phi / s1, nr, pp1);
                if best.is_none_or(|b| cost < b.0) {
                    best = Some((cost, c.p, s1, nr, m1));
                }
            }
            // A candidate whose divisors all miss the window: its largest even one.
            if best.is_none() {
                let s1 = ds.iter().copied().filter(|d| d % 2 == 0).max().unwrap_or(2) as usize;
                if s1 >= 2 && nr + s1 <= MAX_LIMBS / (2 * k + 1) {
                    best = Some((cost(k, s1, phi / s1, nr, pp1), c.p, s1, nr, m1));
                }
            }
            // After the first, the others whose bounds beat the best, by increasing bounds.
            if !sorted && let Some(b) = best {
                sorted = true;
                cands.swap_remove(0);
                cands.retain(|c| c.0 < b.0);
                cands.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                i = 0;
            }
        }
        // No candidate has a divisor near its best `s_1` that fits `MAX_LIMBS`: past some `B2`
        // (about `10^12` at 16 limbs, `10^14` at 4) the best `s_1` of every `P` is too long. The
        // largest divisor of each that fits then, so that the stage stays a continuation by
        // polynomials rather than a product for each prime, with at most `2^20` evaluations.
        if best.is_none() {
            for &(_, c, nr, m1) in &cands {
                divisors(&c.lens, &mut ds);
                let room = (MAX_LIMBS / (2 * k + 1)).saturating_sub(nr) as u64;
                let phi = c.phi;
                let fits = ds
                    .iter()
                    .copied()
                    .filter(|&d| d % 2 == 0 && d <= room && phi / d <= 1 << 20)
                    .max();
                if let Some(s1) = fits {
                    let s1 = s1 as usize;
                    let cost = cost(k, s1, phi as usize / s1, nr, pp1);
                    if best.is_none_or(|b| cost < b.0) {
                        best = Some((cost, c.p, s1, nr, m1));
                    }
                }
            }
        }
        let (cost, p, s1, nr, m1) = best?;
        // The atoms of `S_1`: one of each length as `s_1` has its prime factors.
        let mut rest = s1 as u64;
        let (mut atoms1, mut atoms2) = (Vec::new(), Vec::new());
        for a in atoms(p) {
            if rest.is_multiple_of(a.len) {
                rest /= a.len;
                atoms1.push(a);
            } else {
                atoms2.push(a);
            }
        }
        debug_assert_eq!(rest, 1);
        Some(MkPlan {
            p,
            atoms1,
            atoms2,
            s1,
            s2: phi(p) as usize / s1,
            nr,
            m1,
            cost,
        })
    }
}

/// The most limbs of a factor of the products of polynomials, `l (2k + 1)` for `g` (`h` is
/// shorter): 2^21, 16 MiB; the product's number is at most twice that.
const MAX_LIMBS: usize = 1 << 21;

/// The time of the stage in nanoseconds (see [`product_ns`]): per evaluation, `g` (two products a
/// coefficient for `p - 1`, two products in the extension for `p + 1`, in two chains), the product
/// of polynomials (two for `p + 1`) and the outputs' product; once, `f` from the sets, about a
/// product of polynomials of `s_1` coefficients and four products of residues a coefficient
/// (twice that for `p + 1`), and `h`.
fn cost(k: usize, s1: usize, s2: usize, nr: usize, pp1: bool) -> f64 {
    let l = s1 + nr;
    let mm = product_ns(k);
    // Two independent chains: short products overlap.
    let chained = if k <= 4 { 0.55 } else { 0.95 } * mm;
    let (gc, conv) = if pp1 { (8.0, 2.0) } else { (2.0, 1.0) };
    let eval = gc * l as f64 * chained + conv * kron_ns(k, s1 + 1, l, nr) + nr as f64 * mm;
    let f = conv * kron_ns(k, s1, s1, 2 * s1) + conv * 4.0 * s1 as f64 * mm;
    // The powers, the buffers and the choice of the parameters: measured about 28 us and 350
    // products of residues; `p + 1`'s powers in the extension, a fifth more.
    let setup = f + 3.0 * s1 as f64 * mm + 350.0 * mm + 28_000.0;
    let total = s2 as f64 * (eval + 100.0 * mm) + setup;
    if pp1 { 1.2 * total } else { total }
}

/// The products of the sub-multisets of the sorted `lens`, each once, into `out`.
fn divisors(lens: &[u8], out: &mut Vec<u64>) {
    out.clear();
    out.push(1);
    let mut i = 0;
    while i < lens.len() {
        let q = u64::from(lens[i]);
        let mut e = 0;
        while i < lens.len() && u64::from(lens[i]) == q {
            e += 1;
            i += 1;
        }
        let base = out.len();
        let mut pw = 1;
        for _ in 0..e {
            pw *= q;
            for j in 0..base {
                out.push(out[j] * pw);
            }
        }
    }
}

/// `b^e` for a signed `e`, with `bi = 1/b`.
fn power_signed<A: Arith>(f: &A, b: &A::V, bi: &A::V, e: i64) -> A::V {
    match e {
        0 => f.one(),
        e if e > 0 => power(f, b, &[e as u64]),
        e => power(f, bi, &[e.unsigned_abs()]),
    }
}

/// The elements of the sum of the atoms, `b^(2k)` for each `k` of it, in the mixed radix of the
/// atoms, so that the element at `i` and the one at `len - 1 - i` are inverses.
fn sum_powers<A: Arith>(f: &A, b: &A::V, bi: &A::V, atoms: &[Atom]) -> Vec<A::V> {
    let mut list = vec![f.one()];
    for a in atoms {
        let first = power_signed(f, b, bi, 2 * a.start);
        let step = power_signed(f, b, bi, 2 * a.step);
        let mut elems = Vec::with_capacity(a.len as usize);
        let mut e = first;
        for _ in 0..a.len {
            elems.push(e.clone());
            f.mul_in(&mut e, &step);
        }
        let mut next = Vec::with_capacity(list.len() * elems.len());
        for x in &elems {
            for y in &list {
                let mut z = f.zero();
                f.mul(&mut z, y, x);
                next.push(z);
            }
        }
        list = next;
    }
    list
}

/// The sum of the atoms, the elements `k` in the order of [`sum_powers`].
fn sum_elements(atoms: &[Atom]) -> Vec<i64> {
    let mut list = vec![0i64];
    for a in atoms {
        let mut next = Vec::with_capacity(list.len() * a.len as usize);
        for i in 0..a.len as i64 {
            for &y in &list {
                next.push(y + a.start + i * a.step);
            }
        }
        list = next;
    }
    list
}

/// The product of the polynomials `polys`, by a tree of products.
fn product(kron: &mut Kron, mut polys: Vec<Vec<u64>>) -> Vec<u64> {
    let k = kron.limbs();
    while polys.len() > 1 {
        let mut next = Vec::with_capacity(polys.len().div_ceil(2));
        let mut it = polys.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => {
                    let len = a.len() / k + b.len() / k - 1;
                    let mut c = vec![0u64; len * k];
                    kron.mul(&a, &b, 0, &mut c);
                    next.push(c);
                }
                None => next.push(a),
            }
        }
        polys = next;
    }
    polys.pop().unwrap_or_default()
}

/// The atoms from the longest: each step multiplies `|T|` copies of the polynomial so far, so the
/// long ones go first, while it is short.
fn longest_first(atoms: &[Atom]) -> Vec<Atom> {
    let mut order = atoms.to_vec();
    order.sort_by_key(|a| core::cmp::Reverse(a.len));
    order
}

/// `x - 1`, the polynomial of the sum of no atoms, `{0}`.
fn start<A: Arith>(f: &A) -> Vec<u64> {
    let k = f.n().len();
    let one = f.one();
    let mut minus = f.zero();
    f.sub(&mut minus, &f.zero(), &one);
    let mut p = vec![0u64; 2 * k];
    f.put(&minus, &mut p[..k]);
    f.put(&one, &mut p[k..]);
    p
}

/// `F(x c)`: coefficient `j` times `c^j`.
fn scale<A: Arith>(f: &A, poly: &[u64], c: &A::V) -> Vec<u64> {
    let k = f.n().len();
    let mut out = vec![0u64; poly.len()];
    let mut w = f.one();
    for (src, dst) in poly.chunks_exact(k).zip(out.chunks_exact_mut(k)) {
        let mut v = f.of_raw(src);
        f.mul_in(&mut v, &w);
        f.put(&v, dst);
        f.mul_in(&mut w, c);
    }
    out
}

/// `prod_(k in S) (x - b^(2k))` up to a unit factor, for the sum `S` of the atoms, by Montgomery
/// and Kruppa's `f_(A+T)(x) = prod_(t in T) f_A(x b^(-2t))` up to units: an atom of length `q`
/// takes `q` scaled copies and `q - 1` products.
fn build_f<A: Arith>(f: &A, kron: &mut Kron, b: &A::V, bi: &A::V, atoms: &[Atom]) -> Vec<u64> {
    let mut poly = start(f);
    for a in longest_first(atoms) {
        // `c_t = b^(-2t)` for the elements `t = start + i step`.
        let mut c = power_signed(f, bi, b, 2 * a.start);
        let step = power_signed(f, bi, b, 2 * a.step);
        let mut copies = Vec::with_capacity(a.len as usize);
        for _ in 0..a.len {
            copies.push(scale(f, &poly, &c));
            f.mul_in(&mut c, &step);
        }
        poly = product(kron, copies);
    }
    poly
}

/// [`build_f`] for `b = beta` in `Z_n[sqrt(Delta)]` of norm 1: `f` has its coefficients in `Z_n`,
/// and the copies of `t` and `-t` together, `F(x c) F(x/c) = U^2 - Delta W^2` for `F(x c) = U +
/// sqrt(Delta) W`, take two real products.
fn build_f_ext<A: Arith>(
    ext: &Ext<'_, A>,
    kron: &mut Kron,
    beta: &E<A::V>,
    atoms: &[Atom],
) -> Vec<u64> {
    let f = ext.f;
    let k = f.n().len();
    let mut poly = start(f);
    for a in longest_first(atoms) {
        let mut copies = Vec::with_capacity(a.len as usize);
        if a.len % 2 == 1 {
            copies.push(poly.clone());
        }
        // The positive elements `t`, `c = beta^(-2t)`: `U_j = F_j x(c^j)`, `W_j = F_j y(c^j)`, the
        // powers by `c^(j + 1) = c^j V - c^(j - 1)`, `V = V_(-2t)`, two products each.
        for i in a.len.div_ceil(2)..a.len {
            let t = a.start + i as i64 * a.step;
            let vc = ext.v(beta, -2 * t);
            let mut u = vec![0u64; poly.len()];
            let mut w = vec![0u64; poly.len()];
            let mut wplain = vec![0u64; poly.len()];
            let mut prev = ext.pow_signed(beta, 2 * t);
            let mut cj = E {
                x: f.one(),
                y: f.zero(),
            };
            let mut next = cj.clone();
            let mut v = f.zero();
            for (((src, du), dw), dp) in poly
                .chunks_exact(k)
                .zip(u.chunks_exact_mut(k))
                .zip(w.chunks_exact_mut(k))
                .zip(wplain.chunks_exact_mut(k))
            {
                let fj = f.of_raw(src);
                f.mul(&mut v, &fj, &cj.x);
                f.put(&v, du);
                f.mul(&mut v, &fj, &cj.y);
                f.put(&v, dp);
                // `-Delta W_j`, so that `U U + (-Delta W) W` is the product.
                f.mul_in(&mut v, &ext.delta);
                let x = v.clone();
                f.sub(&mut v, &f.zero(), &x);
                f.put(&v, dw);
                ext.lin(&mut next, &cj, &vc, &prev);
                core::mem::swap(&mut prev, &mut cj);
                core::mem::swap(&mut cj, &mut next);
            }
            let len = 2 * (poly.len() / k) - 1;
            let mut prod = vec![0u64; len * k];
            kron.mul_add(&u, &u, &w, &wplain, 0, &mut prod);
            copies.push(prod);
        }
        poly = product(kron, copies);
    }
    poly
}

/// The second stage of `p - 1` from the residue `b` of the first: `Some` of a factor of `n` or of
/// `None`, or `None` when an evaluation's product was 0 modulo `n` and one factor of it alone is
/// (see [`backtrack`]), for the baby-step giant-step continuation to go over again.
pub(super) fn pm1_stage2<A: Arith>(f: &A, b: &A::V, plan: &MkPlan) -> Option<Option<Vec<u64>>> {
    let k = f.n().len();
    let bi = match invert(f, b) {
        Ok(v) => v,
        Err(g) => return Some(proper(f, g)),
    };
    let mut kron = Kron::new(f.n());
    let (s1, d, nr) = (plan.s1, plan.s1 / 2, plan.nr);
    let l = s1 + nr;
    // `f`, from the sets.
    let poly = build_f(f, &mut kron, b, &bi, &plan.atoms1);
    debug_assert_eq!(poly.len(), (s1 + 1) * k);
    // `h_j = f_j r^(-j^2)` for `r = b^P`, `f_j` the coefficient `d + j`, laid out symmetrically:
    // `h'_u = h_(|u - d|)` for `u` up to `2d`.
    let p = plan.p as i64;
    let r = power_signed(f, b, &bi, p);
    let ri = power_signed(f, &bi, b, p);
    let mut ri2 = f.zero();
    f.sqr(&mut ri2, &ri);
    let mut h = vec![0u64; (s1 + 1) * k];
    {
        // `w = r^(-j^2)`, `dw = r^(-(2j + 1))`.
        let mut w = f.one();
        let mut dw = ri.clone();
        for j in 0..=d {
            let mut c = f.of_raw(&poly[(d + j) * k..(d + j + 1) * k]);
            f.mul_in(&mut c, &w);
            f.put(&c, &mut h[(d + j) * k..(d + j + 1) * k]);
            if j > 0 {
                h.copy_within((d + j) * k..(d + j + 1) * k, (d - j) * k);
            }
            f.mul_in(&mut w, &dw);
            f.mul_in(&mut dw, &ri2);
        }
    }
    drop(poly);
    // The evaluations, one for each `k_2` of `S_2`.
    let ks2 = sum_elements(&plan.atoms2);
    debug_assert_eq!(ks2.len(), plan.s2);
    let m = (l - 1 - d) as i64;
    // `r^(M^2)` and `r^(-2M + 1)`, the same for every `k_2`.
    let rm2 = power(f, &r, &mul_wide(m, m));
    let rstep = power_signed(f, &ri, &r, 2 * m - 1);
    let mut r2 = f.zero();
    f.sqr(&mut r2, &r);
    let mut g = vec![0u64; l * k];
    let mut out = vec![0u64; nr * k];
    for &k2 in &ks2 {
        let e = 2 * k2 + (2 * plan.m1 + 1) * p;
        let x0 = power_signed(f, b, &bi, e);
        let x0i = power_signed(f, &bi, b, e);
        // `g_0 = x_0^M r^(M^2)`, `g_i = g_(i-1) x_0^(-1) r^(-2M + 2i - 1)`.
        let mut gi = power_signed(f, &x0, &x0i, m);
        f.mul_in(&mut gi, &rm2);
        let mut ratio = x0i.clone();
        f.mul_in(&mut ratio, &rstep);
        for i in 0..l {
            f.put(&gi, &mut g[i * k..(i + 1) * k]);
            f.mul_in(&mut gi, &ratio);
            f.mul_in(&mut ratio, &r2);
        }
        kron.mul(&h, &g, s1, &mut out);
        let mut acc = f.one();
        for c in out.chunks_exact(k) {
            f.mul_raw(&mut acc, c);
        }
        match gcd(f, &acc) {
            Gcd::One => {}
            Gcd::Factor(g) => return Some(Some(g)),
            Gcd::Whole => {
                let xs = sum_powers(f, b, &bi, &plan.atoms1);
                return backtrack(f, &out, &xs, &x0, &r2, nr);
            }
        }
    }
    Some(None)
}

/// `a b` for the signed `a` and `b`, as the limbs of its absolute value.
fn mul_wide(a: i64, b: i64) -> Vec<u64> {
    let x = u128::from(a.unsigned_abs()) * u128::from(b.unsigned_abs());
    let mut v = vec![x as u64, (x >> 64) as u64];
    while v.len() > 1 && v[v.len() - 1] == 0 {
        v.pop();
    }
    v
}

/// The evaluation whose product was 0 modulo `n`, again a value at a time, and a value that is
/// 0 again a factor of it at a time: `Some` of the factor, or `None` when one factor alone is 0.
///
/// The values go from the top of the range down, and a factor's exponent may be a multiple of
/// the orders of several primes (`q_1 q_2` up to `B2`, or the same `q` twice): when the first
/// factor that is not a unit is 0, the baby-step giant-step continuation, which goes up the primes
/// one product at a time, may still tell the primes apart, so it is left to it.
fn backtrack<A: Arith>(
    f: &A,
    out: &[u64],
    xs: &[A::V],
    x0: &A::V,
    r2: &A::V,
    nr: usize,
) -> Option<Option<Vec<u64>>> {
    let k = f.n().len();
    for (t, c) in out.chunks_exact(k).enumerate() {
        let v = f.of_raw(c);
        match gcd(f, &v) {
            Gcd::One => {}
            Gcd::Factor(g) => return Some(Some(g)),
            Gcd::Whole => {
                // The point `X = x_0 r^(2i)` of the output, `i = nr - 1 - t`, and its factors
                // `X - b^(2 k_1)`.
                let i = (nr - 1 - t) as u64;
                let mut x = power(f, r2, &[i.max(1)]);
                if i == 0 {
                    x = f.one();
                }
                f.mul_in(&mut x, x0);
                let mut dif = f.zero();
                for y in xs {
                    f.sub(&mut dif, &x, y);
                    match gcd(f, &dif) {
                        Gcd::One => {}
                        Gcd::Factor(g) => return Some(Some(g)),
                        Gcd::Whole => return None,
                    }
                }
                return None;
            }
        }
    }
    None
}

/// A residue of the extension `Z_n[sqrt(Delta)]`, `x + y sqrt(Delta)`.
#[derive(Clone)]
struct E<V> {
    x: V,
    y: V,
}

/// The arithmetic of `Z_n[sqrt(Delta)]`.
struct Ext<'a, A: Arith> {
    f: &'a A,
    /// `Delta`.
    delta: A::V,
}

impl<A: Arith> Ext<'_, A> {
    /// `r = a b`: `(ac + Delta bd) + ((a + b)(c + d) - ac - bd) sqrt(Delta)`, four products.
    fn mul(&self, r: &mut E<A::V>, a: &E<A::V>, b: &E<A::V>) {
        let f = self.f;
        let (mut t1, mut t2, mut t3, mut t4) = (f.zero(), f.zero(), f.zero(), f.zero());
        f.mul(&mut t1, &a.x, &b.x);
        f.mul(&mut t2, &a.y, &b.y);
        f.add(&mut t3, &a.x, &a.y);
        f.add(&mut t4, &b.x, &b.y);
        f.mul_in(&mut t3, &t4);
        f.sub_in(&mut t3, &t1);
        f.sub(&mut r.y, &t3, &t2);
        f.mul_in(&mut t2, &self.delta);
        f.add(&mut r.x, &t1, &t2);
    }

    /// `r = r b`.
    fn mul_in(&self, r: &mut E<A::V>, b: &E<A::V>) {
        let a = r.clone();
        self.mul(r, &a, b);
    }

    /// `a^e` for an element `a` of norm 1, `e >= 0`, from the top bit: the square of a norm-1
    /// element `(2 x^2 - 1) + 2 x y sqrt(Delta)` takes two products.
    fn pow(&self, a: &E<A::V>, e: u64) -> E<A::V> {
        let f = self.f;
        let mut r = E {
            x: f.one(),
            y: f.zero(),
        };
        if e == 0 {
            return r;
        }
        r = a.clone();
        let one = f.one();
        let (mut t, mut u) = (f.zero(), f.zero());
        for i in (0..63 - e.leading_zeros()).rev() {
            f.sqr(&mut u, &r.x);
            f.add(&mut t, &u, &u);
            f.mul(&mut u, &r.x, &r.y);
            f.sub(&mut r.x, &t, &one);
            f.add(&mut r.y, &u, &u);
            if e >> i & 1 == 1 {
                self.mul_in(&mut r, a);
            }
        }
        r
    }

    /// `a^e` for a signed `e` and `a` of norm 1, whose inverse is its conjugate.
    fn pow_signed(&self, a: &E<A::V>, e: i64) -> E<A::V> {
        let mut r = self.pow(a, e.unsigned_abs());
        if e < 0 {
            self.conj(&mut r);
        }
        r
    }

    /// The conjugate, in place: the inverse of an element of norm 1.
    fn conj(&self, a: &mut E<A::V>) {
        let y = a.y.clone();
        self.f.sub(&mut a.y, &self.f.zero(), &y);
    }

    /// `V_m = beta^m + beta^(-m)`, twice the rational coordinate of `beta^m`, for `beta` of norm 1.
    fn v(&self, beta: &E<A::V>, m: i64) -> A::V {
        let p = self.pow_signed(beta, m);
        let mut r = self.f.zero();
        self.f.add(&mut r, &p.x, &p.x);
        r
    }

    /// `r = a v - b` for a rational `v`: two products.
    fn lin(&self, r: &mut E<A::V>, a: &E<A::V>, v: &A::V, b: &E<A::V>) {
        let f = self.f;
        f.mul(&mut r.x, &a.x, v);
        f.sub_in(&mut r.x, &b.x);
        f.mul(&mut r.y, &a.y, v);
        f.sub_in(&mut r.y, &b.y);
    }

    /// Calls `visit(i, G)` for `G = beta^(a t^2 + b t)`, `t = t0 + i`, `i` below `count`, for `beta`
    /// of norm 1, by Montgomery's recurrence (GMP-ECM's `gfp_ext_rn2`): with `d(t) = 2at + a + b`
    /// the difference of the exponents and `G2 = G beta^(2a)`, `G(t) = G2(t - 1) V_(d(t - 2)) -
    /// G2(t - 2)` and `G2(t) = G2(t - 1) V_(d(t - 1)) - G(t - 2)`, five products a term against
    /// eight for two products in the extension.
    fn quadratic(
        &self,
        beta: &E<A::V>,
        a: i64,
        b: i64,
        t0: i64,
        count: usize,
        mut visit: impl FnMut(usize, &E<A::V>),
    ) {
        let f = self.f;
        let e = |t: i64| a * t * t + b * t;
        let d = |t: i64| 2 * a * t + a + b;
        let (mut g0, mut g1) = (
            self.pow_signed(beta, e(t0)),
            self.pow_signed(beta, e(t0 + 1)),
        );
        let (mut h0, mut h1) = (
            self.pow_signed(beta, e(t0) + 2 * a),
            self.pow_signed(beta, e(t0 + 1) + 2 * a),
        );
        let v2a = self.v(beta, 2 * a);
        // `V_(d(t - 2))` and `V_(d(t - 1))` for the next `t`.
        let (mut va, mut vb) = (self.v(beta, d(t0)), self.v(beta, d(t0 + 1)));
        let (mut g, mut h) = (g0.clone(), g0.clone());
        let mut vn = f.zero();
        for i in 0..count {
            match i {
                0 => visit(0, &g0),
                1 => visit(1, &g1),
                _ => {
                    self.lin(&mut g, &h1, &va, &h0);
                    self.lin(&mut h, &h1, &vb, &g0);
                    visit(i, &g);
                    // Shift: `(G, G2)(t - 2) <- (t - 1)`, `(t - 1) <- t`.
                    core::mem::swap(&mut g0, &mut g1);
                    core::mem::swap(&mut g1, &mut g);
                    core::mem::swap(&mut h0, &mut h1);
                    core::mem::swap(&mut h1, &mut h);
                    f.mul(&mut vn, &vb, &v2a);
                    f.sub_in(&mut vn, &va);
                    core::mem::swap(&mut va, &mut vb);
                    core::mem::swap(&mut vb, &mut vn);
                }
            }
        }
    }
}

/// The second stage of `p + 1` from `V = V_E(P_0)` of the first: `Some` of a factor of `n` or of
/// `None`, or `None` for the baby-step giant-step continuation, as [`pm1_stage2`].
///
/// The root `beta = (V + sqrt(Delta)) / 2` of `X^2 - V X + 1`, `Delta = V^2 - 4`, has norm 1, and
/// `V_m = beta^m + beta^(-m)`: the method of [`pm1_stage2`] with `b = beta` in `Z_n[sqrt(Delta)]`.
/// `f`, whose roots `beta^(2k)` come in inverse pairs, has its coefficients in `Z_n`, from
/// `V_(2k)`; `h` and `g` have two coordinates, and of the values only the rational coordinate is
/// kept, `h_x g_x + Delta h_y g_y`, which is 0 modulo `p` with the value.
pub(super) fn pp1_stage2<A: Arith>(f: &A, v: &A::V, plan: &MkPlan) -> Option<Option<Vec<u64>>> {
    let k = f.n().len();
    // `Delta = V^2 - 4`, `beta = V/2 + sqrt(Delta)/2`.
    let mut delta = f.zero();
    f.sqr(&mut delta, v);
    f.sub_in(&mut delta, &f.small(4));
    let half = match invert(f, &f.small(2)) {
        Ok(h) => h,
        Err(g) => return Some(proper(f, g)),
    };
    let mut bx = f.zero();
    f.mul(&mut bx, v, &half);
    let ext = Ext { f, delta };
    let beta = E { x: bx, y: half };
    let mut kron = Kron::new(f.n());
    let (s1, d, nr) = (plan.s1, plan.s1 / 2, plan.nr);
    let l = s1 + nr;
    let poly = build_f_ext(&ext, &mut kron, &beta, &plan.atoms1);
    debug_assert_eq!(poly.len(), (s1 + 1) * k);
    // `h_j = f_j r^(-j^2)` for `r = beta^P`: its coordinates, `x` and `Delta y`, symmetric.
    let p = plan.p as i64;
    let r = ext.pow_signed(&beta, p);
    let mut ri = r.clone();
    ext.conj(&mut ri);
    let mut ri2 = E {
        x: f.zero(),
        y: f.zero(),
    };
    ext.mul(&mut ri2, &ri, &ri);
    let mut hx = vec![0u64; (s1 + 1) * k];
    let mut hy = vec![0u64; (s1 + 1) * k];
    {
        let mut c = f.zero();
        ext.quadratic(&beta, -p, 0, 0, d + 1, |j, w| {
            let fj = f.of_raw(&poly[(d + j) * k..(d + j + 1) * k]);
            let (lo, hi) = ((d + j) * k, (d + j + 1) * k);
            f.mul(&mut c, &fj, &w.x);
            f.put(&c, &mut hx[lo..hi]);
            f.mul(&mut c, &fj, &w.y);
            f.mul_in(&mut c, &ext.delta);
            f.put(&c, &mut hy[lo..hi]);
            if j > 0 {
                hx.copy_within(lo..hi, (d - j) * k);
                hy.copy_within(lo..hi, (d - j) * k);
            }
        });
    }
    drop(ri2);
    drop(poly);
    let ks2 = sum_elements(&plan.atoms2);
    debug_assert_eq!(ks2.len(), plan.s2);
    let m = (l - 1 - d) as i64;
    let mut r2 = E {
        x: f.zero(),
        y: f.zero(),
    };
    ext.mul(&mut r2, &r, &r);
    let mut gx = vec![0u64; l * k];
    let mut gy = vec![0u64; l * k];
    let mut out = vec![0u64; nr * k];
    for &k2 in &ks2 {
        let e = 2 * k2 + (2 * plan.m1 + 1) * p;
        let x0 = ext.pow_signed(&beta, e);
        // `g_i = x_0^(M - i) r^((M - i)^2) = beta^(P t^2 + e t)` for `t = M - i`, from the
        // smallest `t`.
        let t0 = m - (l as i64 - 1);
        ext.quadratic(&beta, p, e, t0, l, |j, w| {
            let i = l - 1 - j;
            f.put(&w.x, &mut gx[i * k..(i + 1) * k]);
            f.put(&w.y, &mut gy[i * k..(i + 1) * k]);
        });
        kron.mul_add(&hx, &gx, &hy, &gy, s1, &mut out);
        let mut acc = f.one();
        for c in out.chunks_exact(k) {
            f.mul_raw(&mut acc, c);
        }
        match gcd(f, &acc) {
            Gcd::One => {}
            Gcd::Factor(g) => return Some(Some(g)),
            Gcd::Whole => {
                let xs = ext_sum_powers(&ext, &beta, &plan.atoms1);
                return pp1_backtrack(&ext, &out, &xs, &x0, &r2, nr);
            }
        }
    }
    Some(None)
}

/// [`backtrack`] in the extension: the factors `X - beta^(2 k_1)` of a value that is 0 modulo `n`
/// by their norms.
fn pp1_backtrack<A: Arith>(
    ext: &Ext<'_, A>,
    out: &[u64],
    xs: &[E<A::V>],
    x0: &E<A::V>,
    r2: &E<A::V>,
    nr: usize,
) -> Option<Option<Vec<u64>>> {
    let f = ext.f;
    let k = f.n().len();
    for (t, c) in out.chunks_exact(k).enumerate() {
        let v = f.of_raw(c);
        match gcd(f, &v) {
            Gcd::One => {}
            Gcd::Factor(g) => return Some(Some(g)),
            Gcd::Whole => {
                let i = (nr - 1 - t) as u64;
                let mut x = ext.pow(r2, i);
                ext.mul_in(&mut x, x0);
                let (mut zx, mut zy, mut nz, mut t2) = (f.zero(), f.zero(), f.zero(), f.zero());
                for y in xs {
                    f.sub(&mut zx, &x.x, &y.x);
                    f.sub(&mut zy, &x.y, &y.y);
                    f.sqr(&mut nz, &zx);
                    f.sqr(&mut t2, &zy);
                    f.mul_in(&mut t2, &ext.delta);
                    f.sub_in(&mut nz, &t2);
                    match gcd(f, &nz) {
                        Gcd::One => {}
                        Gcd::Factor(g) => return Some(Some(g)),
                        Gcd::Whole => return None,
                    }
                }
                return None;
            }
        }
    }
    None
}

/// The elements `beta^(2k)` of the sum of the atoms, in the order of [`sum_powers`].
fn ext_sum_powers<A: Arith>(ext: &Ext<'_, A>, beta: &E<A::V>, atoms: &[Atom]) -> Vec<E<A::V>> {
    let f = ext.f;
    let mut xs = vec![E {
        x: f.one(),
        y: f.zero(),
    }];
    for a in atoms {
        let mut e = ext.pow_signed(beta, 2 * a.start);
        let step = ext.pow_signed(beta, 2 * a.step);
        let mut elems = Vec::with_capacity(a.len as usize);
        for _ in 0..a.len {
            elems.push(e.clone());
            ext.mul_in(&mut e, &step);
        }
        let mut next = Vec::with_capacity(xs.len() * elems.len());
        for x in &elems {
            for y in &xs {
                let mut z = E {
                    x: f.zero(),
                    y: f.zero(),
                };
                ext.mul(&mut z, y, x);
                next.push(z);
            }
        }
        xs = next;
    }
    xs
}
