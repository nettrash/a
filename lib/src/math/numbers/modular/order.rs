//! Multiplicative orders from a multiple of them given by its primes, modulo prime powers, and the
//! factorisations of the groups' orders they start from.

use super::int::{Int, lcm_factored, product};
use super::ring::{Low, Low2, Plain, Ring, limbs_of, with_odd};
use super::word;

/// The exponents, prime by prime, of the order of the unit `a` in `r`, for `n`, a multiple of the
/// order given by its primes with their exponents; `None` when `a^n` is not 1.
///
/// After Sutherland (2007): the primes are split in two halves, of products `n_1` and `n_2`, and
/// the orders of `a^(n_2)` and of `a^(n_1)` are found among the primes of `n_1` and of `n_2`, down
/// to a single prime `q`, whose exponent is the number of powers by `q` that bring the residue
/// to 1. That takes about `log2(n) log2(k)` products for `k` primes, against `k log2(n)` for one
/// power by `n / q^e` for each prime.
pub(in super::super) fn order_in<R: Ring, I: Int>(
    r: &R,
    a: &R::V,
    n: &[(I, u32)],
) -> Option<Vec<u32>> {
    let mut out = vec![0; n.len()];
    split(r, a.clone(), n, &mut out)?;
    Some(out)
}

/// [`order_in`] for the primes `n`, into `out`.
fn split<R: Ring, I: Int>(r: &R, a: R::V, n: &[(I, u32)], out: &mut [u32]) -> Option<()> {
    if r.is_one(&a) {
        out.fill(0);
        return Some(());
    }
    match n.len() {
        0 => None,
        1 => {
            let (q, f) = &n[0];
            let q = q.limbs();
            let mut b = a;
            let mut t = 0;
            while !r.is_one(&b) {
                if t == *f {
                    return None;
                }
                b = r.pow(&b, q.as_slice());
                t += 1;
            }
            out[0] = t;
            Some(())
        }
        k => {
            let (left, right) = n.split_at(k / 2);
            let (ol, or) = out.split_at_mut(k / 2);
            let al = r.pow(&a, product(right).limbs().as_slice());
            let ar = r.pow(&a, product(left).limbs().as_slice());
            split(r, al, left, ol)?;
            split(r, ar, right, or)
        }
    }
}

/// The primes of `n` with the exponents `e`, those of exponent 0 left out.
pub(in super::super) fn with_exponents<I: Int>(n: &[(I, u32)], e: &[u32]) -> Vec<(I, u32)> {
    n.iter()
        .zip(e)
        .filter(|&(_, &e)| e > 0)
        .map(|((q, _), &e)| (q.clone(), e))
        .collect()
}

/// The factorisation of `p - 1` for the prime `p`.
pub(in super::super) fn factor_below<I: Int>(p: &I) -> Vec<(I, u32)> {
    let p1 = p.sub(&I::small(1));
    factor_int(&p1)
}

/// The factorisation of `n >= 1`: in words below `2^64`, by [`Number::factor`] beyond.
pub(in super::super) fn factor_int<I: Int>(n: &I) -> Vec<(I, u32)> {
    if let Some(w) = n.word() {
        return word::factor(w)
            .iter()
            .map(|(p, e)| (I::small(p), e))
            .collect();
    }
    let n = n.number(super::super::ByteOrder::LittleEndian);
    n.factor()
        .into_factors()
        .into_iter()
        .map(|(p, e)| (I::of_limbs(&limbs_of(&p)), e))
        .collect()
}

/// The order of the plain `a`, prime to the odd prime `p`, modulo `p^e`, `e >= 1`, as its primes
/// with their exponents, for the primes of `p - 1`: the order modulo `p` first, in the smaller
/// ring, among the primes of `p - 1`, then the power of `p` that the residue to that power takes
/// modulo `p^e` to reach 1, at most `p^(e-1)`.
pub(in super::super) fn order_odd<I: Int>(
    a: &[u64],
    p: &I,
    e: u32,
    p1: &[(I, u32)],
) -> Vec<(I, u32)> {
    let pl = p.limbs();
    let exps = with_odd!(pl.as_slice(), |r| {
        let x = r.residue(a);
        order_in(&r, &x, p1)
    });
    // A unit's order divides `p - 1`.
    let exps = exps.unwrap_or_else(|| vec![0; p1.len()]);
    let mut out = with_exponents(p1, &exps);
    if e >= 2 {
        let pe = p.pow(e);
        let o = product(&out);
        let t = with_odd!(pe.limbs().as_slice(), |r| {
            let x = r.residue(a);
            let mut y = r.pow(&x, o.limbs().as_slice());
            let mut t = 0;
            while !r.is_one(&y) && t < e - 1 {
                y = r.pow(&y, pl.as_slice());
                t += 1;
            }
            t
        });
        if t > 0 {
            // `p` is above every prime of `p - 1`.
            out.push((p.clone(), t));
        }
    }
    out
}

/// The exponent of the order of the odd plain `a` modulo `2^e`: the number of squares that bring
/// it to 1, at most `e - 2` for `e >= 3`.
pub(in super::super) fn order_two(a: &[u64], e: u32) -> u32 {
    fn count<R: Ring>(r: &R, a: &[u64]) -> u32 {
        let mut x = r.residue(a);
        let mut t = 0;
        while !r.is_one(&x) {
            x = r.sqr(&x);
            t += 1;
        }
        t
    }
    match e {
        0 | 1 => 0,
        2..=64 => count(&Low::new(e), a),
        65..=128 => count(&Low2::new(e), a),
        _ => {
            let mut m = vec![0u64; (e / 64) as usize + 1];
            m[(e / 64) as usize] = 1 << (e % 64);
            count(&Plain::new(&m), a)
        }
    }
}

/// The order of the plain `a`, prime to the prime power `p^e`, modulo it, as primes with exponents.
pub(in super::super) fn order_prime_power<I: Int>(a: &[u64], p: &I, e: u32) -> Vec<(I, u32)> {
    if *p == I::small(2) {
        let t = order_two(a, e);
        return if t > 0 {
            vec![(p.clone(), t)]
        } else {
            Vec::new()
        };
    }
    order_odd(a, p, e, &factor_below(p))
}

/// The order of `a`, prime to `m`, modulo `m = prod p^e`, as primes with exponents: the least
/// common multiple of its orders modulo the prime powers.
pub(in super::super) fn order_mod<I: Int>(a: &I, m: &[(I, u32)]) -> Vec<(I, u32)> {
    let mut out: Vec<(I, u32)> = Vec::new();
    for (p, e) in m {
        let pe = p.pow(*e);
        let x = a.rem(&pe).limbs();
        let o = order_prime_power(x.as_slice(), p, *e);
        out = lcm_factored(&out, &o);
    }
    out
}
