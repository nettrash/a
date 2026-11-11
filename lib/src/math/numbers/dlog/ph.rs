//! Pohlig and Hellman's reduction of a logarithm to logarithms in groups of prime order, and the
//! choice of method in those.

use super::super::modular::int::{Int, join, product};
use super::super::modular::ring::Ring;
use super::super::pow::montgomery::REGISTER_KERNELS;
use super::bsgs::Bsgs;
use super::index::{Index, Sieve};
use super::rho::{Walk, rho_in};

/// Below this prime order, a logarithm is found by stepping through the powers.
const STEPS: u64 = 16;

/// Up to this prime order the baby-step giant-step method is taken modulo numbers of more than a
/// word, beyond it Pollard's rho, whose walks need no table. Measured on an Apple M4 Max in
/// `pohlig_hellman`, orders of 19 to 27 bits modulo primes `c q + 1`: rho is the faster from orders
/// of about `2^21.5` on two limbs and on four, `2^23.3` on eight and `2^23.8` on nine when each
/// call's table has another size than the one before (a table of 16 to 128 KB then costs about
/// 3.5 µs more, fresh pages), and from about `2^25.5` on two limbs, `2^23.2` on four and `2^24`
/// to `2^25` on eight when the same call repeats (the allocator hands back the same memory).
pub(in super::super) const RHO_FROM: u64 = 1 << 23;

/// [`RHO_FROM`] for the residues of one word, whose products are cheap against the table's
/// look-ups and its allocation: rho is the faster there from orders of about `2^19.5` when the
/// tables' sizes change and `2^23.3` when the same call repeats; with `2^23` instead,
/// `discrete_log` modulo primes of 64 bits whose `p - 1` has primes of 22 and 23 bits took 1.2 to
/// 1.3 times as long (`pohlig_hellman` 1.4 to 1.5 times), while with `2^21` repeated calls took at
/// most 6 % longer.
const RHO_FROM_WORD: u64 = 1 << 21;

/// The order from which rho is taken in the ring `R`: [`RHO_FROM_WORD`] for residues of one word
/// (the odd moduli below `2^64` in Montgomery's form, and the powers of two up to `2^64`),
/// [`RHO_FROM`] for the others.
const fn rho_from<R: Ring>() -> u64 {
    if core::mem::size_of::<R::V>() <= core::mem::size_of::<u64>() {
        RHO_FROM_WORD
    } else {
        RHO_FROM
    }
}

/// The smallest prime order whose logarithms modulo a prime of `bits` bits are taken by the index
/// calculus rather than by Pollard's rho, where the index's sieve and linear algebra cost less than
/// the walks' `1.25 sqrt(q)` products: a power of two for each range of the index calculus's
/// parameters (`index.rs`), whose cost steps with them, one less beyond a word without the kernels
/// of the products in registers ([`REGISTER_KERNELS`]), where rho's steps on two limbs take about
/// 1.35 times as long.
///
/// Measured on an Apple M4 Max, `discrete_log` modulo primes `p = s t q + 1` with `q` prime and
/// `s t` smooth, by the index calculus against rho on the same targets: the index calculus took
/// 0.37 to 0.44 million cycles up to 40 bits, 0.66 to 0.72 million at 41 to 44, 1.0 to 1.1 million
/// at 45 to 48, 2.1 million at 53 to 56, 4.9 to 5.0 million at 61 to 64, 9.2 million at 66 and 68,
/// 34 million at 80, 87 million at 88, 0.21 billion at 96, 0.49 billion at 104 and 1.45 billion at
/// 112; rho about `4.3 sqrt(q)` on one limb and 12 to 14 `sqrt(q)` on two (16 to 19 without the
/// kernels), which they match at orders of `2^32.7` to `2^33.3` up to 40 bits, `2^34.5` to `2^34.8`
/// at 41 to 44, `2^36.7` to `2^36.8` at 49 to 52, `2^40.4` at 61 to 64, `2^38.7` to `2^39.0` at 65
/// to 68, `2^42.5` to `2^42.7` at 77 to 80, `2^47.5` to `2^47.9` at 93 to 96 and `2^53.3` to
/// `2^53.4` at 109 to 112 (without the kernels beyond a word 0.5 to 1.1 bits lower, 0.8 on
/// average).
pub(in super::super) fn index_from(bits: u32) -> u128 {
    let b = match bits {
        0..=40 => 33,
        41..=44 => 35,
        45..=48 => 36,
        49..=52 => 37,
        53..=56 => 38,
        57..=60 => 39,
        61..=64 => 40,
        65..=68 => 39,
        69..=72 => 40,
        73..=76 => 41,
        77..=80 => 43,
        81..=84 => 44,
        85..=88 => 45,
        89..=92 => 47,
        93..=96 => 48,
        97..=100 => 49,
        101..=104 => 50,
        105..=108 => 52,
        _ => 53,
    };
    let b = if bits > 64 && !REGISTER_KERNELS {
        b - 1
    } else {
        b
    };
    1 << b
}

/// The modulus `p^e`, `p` odd and `e >= 2`, of a ring, when it is known: in its group the
/// logarithms of order `p` are a division modulo `p`.
pub(in super::super) struct PrimePower<'a, I> {
    /// The prime.
    pub(in super::super) p: &'a I,
    /// `p^(e - 1)`.
    pub(in super::super) pe1: I,
}

/// How the logarithms to a base `γ` of prime order `q` are found.
enum Solver<'r, R: Ring, I: Int> {
    /// Through the powers `γ^j`, `j < q`.
    Steps(Vec<R::V>),
    /// For `q = p` modulo `p^e`: `γ = 1 + p^(e-1) c`, and `1 + p^(e-1) x` is `γ^(x / c)`.
    Linear { c_inv: I, q: I, pe1: I },
    /// By baby steps and giant steps, the table kept for every digit.
    Bsgs(Bsgs<'r, R>, u64),
    /// By Pollard's rho.
    Rho(R::V, I),
    /// By the index calculus modulo the prime of the ring: the logarithms modulo `q` of `z` and of
    /// the base, `γ`, whose quotient is the logarithm; the inverse of the base's is kept, with `γ`
    /// and `q`.
    Index(Box<Index>, I, R::V, I),
}

impl<'r, R: Ring, I: Int> Solver<'r, R, I> {
    /// The solver for the base `gamma` of prime order `q`, for `digits` logarithms; `prime` is the
    /// modulus of the ring when it is known to be a prime of at most 112 bits whose `p - 1` has `q`
    /// once, for the index calculus, with the base `g` and the cofactor that `γ = g^cof`.
    fn new(
        r: &'r R,
        gamma: &R::V,
        q: &I,
        digits: u32,
        pp: Option<&PrimePower<'_, I>>,
        prime: Option<(u128, &R::V, &I)>,
        kept: &mut Option<Sieve>,
    ) -> Solver<'r, R, I> {
        if let Some(prime) = prime
            && digits == 1
            && let Some(s) = Solver::index(r, gamma, q, prime, kept)
        {
            return s;
        }
        if let Some(pp) = pp.filter(|pp| *pp.p == *q) {
            // `γ = 1 + p^(e-1) c`, `c` prime to `p`.
            let plain = I::of_limbs(&r.plain(gamma));
            let c = plain.sub(&I::small(1)).div_exact(&pp.pe1).rem(q);
            if let Some(c_inv) = c.inverse(q) {
                return Solver::Linear {
                    c_inv,
                    q: q.clone(),
                    pe1: pp.pe1.clone(),
                };
            }
        }
        match q.word() {
            Some(w) if w < STEPS => {
                let mut powers = Vec::with_capacity(w as usize);
                let mut e = r.one();
                for _ in 0..w {
                    powers.push(e.clone());
                    e = r.mul(&e, gamma);
                }
                Solver::Steps(powers)
            }
            Some(w) if w < rho_from::<R>() => {
                // As many baby steps as giant steps for all the digits together.
                let s = isqrt_up(w.saturating_mul(u64::from(digits)));
                let inv = r.pow_u64(gamma, w - 1);
                Solver::Bsgs(Bsgs::new(r, gamma, &inv, s), w)
            }
            _ => Solver::Rho(gamma.clone(), q.clone()),
        }
    }

    /// The index calculus's solver for `γ = g^cof` of the prime order `q`, modulo the prime `p`,
    /// when `q` is large enough for it to pay and its logarithms came out. Out of line, so that
    /// the other solvers' setup stays small.
    #[inline(never)]
    fn index(
        r: &'r R,
        gamma: &R::V,
        q: &I,
        (p, g, cof): (u128, &R::V, &I),
        kept: &mut Option<Sieve>,
    ) -> Option<Solver<'r, R, I>> {
        let w = wide(q)?;
        if w < index_from(128 - p.leading_zeros()) {
            return None;
        }
        let ix = Index::new(p, w, kept)?;
        let lg = ix.log(wide_limbs(&r.plain(g)))?;
        // `γ = g^cof`: `L(γ) = cof L(g)`, where `g` is often small, and then at once smooth.
        let lg = I::of_limbs(&[lg as u64, (lg >> 64) as u64]).mul_mod(&cof.rem(q), q);
        let inv = lg.inverse(q)?;
        Some(Solver::Index(Box::new(ix), inv, gamma.clone(), q.clone()))
    }

    /// The logarithm of `z` to the base, below `q`, or `None` when `z` is not a power of it.
    fn log(&self, r: &R, z: &R::V) -> Option<I> {
        match self {
            Solver::Steps(powers) => powers
                .iter()
                .position(|p| p == z)
                .map(|j| I::small(j as u64)),
            Solver::Linear { c_inv, q, pe1 } => {
                // `z = 1 + p^(e-1) x`, `x < p`.
                let plain = I::of_limbs(&r.plain(z));
                if plain.is_zero() {
                    return None;
                }
                let x = plain.sub(&I::small(1));
                if !x.rem(pe1).is_zero() {
                    return None;
                }
                Some(x.div_exact(pe1).rem(q).mul_mod(c_inv, q))
            }
            Solver::Bsgs(t, q) => t.log(z, u128::from(*q)).map(|k| I::small(k as u64)),
            Solver::Index(ix, inv, gamma, q) => {
                // `z` is a power of `γ` exactly when its order divides `q`.
                if !r.is_one(&r.pow(z, q.limbs().as_slice())) {
                    return None;
                }
                if r.is_one(z) {
                    return Some(I::small(0));
                }
                debug_assert!(wide(q) == Some(ix.order()));
                if let Some(lz) = ix.log(wide_limbs(&r.plain(z))) {
                    let k = I::of_limbs(&[lz as u64, (lz >> 64) as u64]).mul_mod(inv, q);
                    if r.pow(gamma, k.limbs().as_slice()) == *z {
                        return Some(k);
                    }
                }
                // Never expected: the walks settle it.
                Solver::<R, I>::Rho(gamma.clone(), q.clone()).log(r, z)
            }
            Solver::Rho(gamma, q) => {
                if r.is_one(z) {
                    return Some(I::small(0));
                }
                // `z` is a power of `γ` exactly when its order divides `q`: otherwise the walks
                // would run in the larger group of `γ` and `z`, about the root of its order, for
                // an answer the check of the whole logarithm then turns down.
                if !r.is_one(&r.pow(z, q.limbs().as_slice())) {
                    return None;
                }
                // The exponents of the walk in words when they fit with room for a sum.
                let walk = match q.word() {
                    Some(w) if w < 1 << 63 => match rho_in::<R, u64>(r, z, gamma, &w, 8) {
                        Walk::Found(k) => Walk::Found(I::small(k)),
                        Walk::Degenerate => Walk::Degenerate,
                    },
                    _ => rho_in(r, z, gamma, q, 8),
                };
                match walk {
                    Walk::Found(k) => Some(k),
                    // In a group of prime order every element is a power of a generator, so only
                    // a run of bad luck ends here: the table settles it.
                    Walk::Degenerate => {
                        let w = q.word()?;
                        let inv = r.pow_u64(gamma, w - 1);
                        let t = Bsgs::new(r, gamma, &inv, isqrt_up(w));
                        t.log(z, u128::from(w)).map(|k| I::small(k as u64))
                    }
                }
            }
        }
    }
}

/// The integer when it fits in two limbs.
fn wide<I: Int>(x: &I) -> Option<u128> {
    match x.limbs().as_slice() {
        [] => Some(0),
        [a] => Some(u128::from(*a)),
        [a, b] => Some(u128::from(*a) | (u128::from(*b) << 64)),
        _ => None,
    }
}

/// The number of at most two limbs.
fn wide_limbs(x: &[u64]) -> u128 {
    x.iter()
        .take(2)
        .rev()
        .fold(0u128, |a, &l| (a << 64) | u128::from(l))
}

/// The square root rounded up.
pub(in super::super) fn isqrt_up(n: u64) -> u64 {
    let r = super::super::modular::word::isqrt(n);
    if r * r < n { r + 1 } else { r }
}

/// The logarithm of `h` to the base `g`, a unit of `r`, reduced modulo the order of `g`, with that
/// order as primes and exponents, for `n`, a multiple of the order given by its primes; `None` when
/// `h` is not a power of `g`, or when `g^n` is not 1.
///
/// Pohlig and Hellman (1978): for each prime power `q^f` of `n`, `g_q = g^(n / q^f)` and `h_q =
/// h^(n / q^f)`; the order of `g_q` is `q^f'`, `f' <= f`, and the logarithm of `h_q` to it is found
/// digit by digit in base `q`, each a logarithm to `γ = g_q^(q^(f'-1))`, of order `q`; the
/// logarithms modulo each `q^f'` are put together by the Chinese remainder theorem, and checked.
pub(in super::super) fn ph_in<R: Ring, I: Int>(
    r: &R,
    h: &R::V,
    g: &R::V,
    n: &[(I, u32)],
    pp: Option<&PrimePower<'_, I>>,
    prime: Option<u128>,
) -> Option<(I, Vec<(I, u32)>)> {
    let total = product(n);
    // The index calculus's sieve, for each large prime of `n`.
    let mut kept = None;
    let (mut x, mut modulus) = (I::small(0), I::small(1));
    let mut orders = Vec::with_capacity(n.len());
    for (q, f) in n {
        let ql = q.limbs();
        let qf = q.pow(*f);
        let cof = total.div_exact(&qf);
        let cl = cof.limbs();
        let gq = r.pow(g, cl.as_slice());
        let hq = r.pow(h, cl.as_slice());
        // The order of `g_q`, `q^f'`, and `γ`.
        let mut gamma = gq.clone();
        let mut fq = 0;
        if !r.is_one(&gamma) {
            loop {
                let next = r.pow(&gamma, ql.as_slice());
                fq += 1;
                if r.is_one(&next) {
                    break;
                }
                if fq == *f {
                    return None;
                }
                gamma = next;
            }
        }
        if fq == 0 {
            if !r.is_one(&hq) {
                return None;
            }
            continue;
        }
        // The index calculus only for a prime that divides `n` once: modulo `q` its logarithms do
        // not tell apart the powers of an element of order `q` when `q^2` divides `p - 1`.
        let solver = Solver::new(
            r,
            &gamma,
            q,
            fq,
            pp,
            prime.filter(|_| *f == 1).map(|p| (p, g, &cof)),
            &mut kept,
        );
        // `g_q^(-1) = g_q^(q^f' - 1)`, then its powers `g_q^(-q^i)`.
        let qfq = q.pow(fq);
        let mut inv = r.pow(&gq, qfq.sub(&I::small(1)).limbs().as_slice());
        let mut t = hq;
        let mut xq = I::small(0);
        let mut qi = I::small(1);
        for i in 0..fq {
            let z = r.pow(&t, q.pow(fq - 1 - i).limbs().as_slice());
            let d = solver.log(r, &z)?;
            if !d.is_zero() {
                xq = xq.add(&d.mul(&qi));
                t = r.mul(&t, &r.pow(&inv, d.limbs().as_slice()));
            }
            if i + 1 < fq {
                inv = r.pow(&inv, ql.as_slice());
                qi = qi.mul(q);
            }
        }
        (x, modulus) = join(&x, &modulus, &xq, &qfq)?;
        orders.push((q.clone(), fq));
    }
    let check = r.pow(g, x.limbs().as_slice());
    (check == *h).then_some((x, orders))
}
