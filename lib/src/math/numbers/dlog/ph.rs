//! Pohlig and Hellman's reduction of a logarithm to logarithms in groups of prime order, and the
//! choice of method in those.

use super::super::modular::int::{Int, join, product};
use super::super::modular::ring::Ring;
use super::bsgs::Bsgs;
use super::rho::{Walk, rho_in};

/// Below this prime order, a logarithm is found by stepping through the powers.
const STEPS: u64 = 16;

/// Up to this prime order the baby-step giant-step method is taken, beyond it Pollard's rho, whose
/// four walks need no table, whose look-ups miss the caches once it is large. Measured on an Apple
/// M4 Max modulo primes `2q + 1`: orders of 20 bits took 17,000 cycles by the table against 31,000
/// by rho, 22 bits 35,000 against 43,000, 24 bits 73,000 against 52,000, 28 bits 536,000 against
/// 287,000.
pub(in super::super) const RHO_FROM: u64 = 1 << 23;

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
}

impl<'r, R: Ring, I: Int> Solver<'r, R, I> {
    /// The solver for the base `gamma` of prime order `q`, for `digits` logarithms.
    fn new(
        r: &'r R,
        gamma: &R::V,
        q: &I,
        digits: u32,
        pp: Option<&PrimePower<'_, I>>,
    ) -> Solver<'r, R, I> {
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
            Some(w) if w < RHO_FROM => {
                // As many baby steps as giant steps for all the digits together.
                let s = isqrt_up(w.saturating_mul(u64::from(digits)));
                let inv = r.pow_u64(gamma, w - 1);
                Solver::Bsgs(Bsgs::new(r, gamma, &inv, s), w)
            }
            _ => Solver::Rho(gamma.clone(), q.clone()),
        }
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
            Solver::Rho(gamma, q) => {
                if r.is_one(z) {
                    return Some(I::small(0));
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
) -> Option<(I, Vec<(I, u32)>)> {
    let total = product(n);
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
        let solver = Solver::new(r, &gamma, q, fq, pp);
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
