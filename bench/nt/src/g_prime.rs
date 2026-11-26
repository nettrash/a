//! Primality: is_probable_prime, next_prime, prev_prime.
use crate::core::*;
#[allow(unused_imports)]
use crate::ffi::*;
use malachite_base::num::factorization::traits::IsPrime as _;

fn pool_size(bits: u32) -> usize {
    match bits {
        0..=128 => 4096,
        129..=256 => 1024,
        257..=512 => 256,
        513..=1024 => 64,
        1025..=2048 => 16,
        _ => 6,
    }
}

pub fn is_prime(ctx: &mut Ctx) {
    for &bits in &[32u32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let mut rng = Rng::new(1000 + bits as u64);
        let n = pool_size(bits);
        let primes: Vec<_> = (0..n.min(if bits >= 2048 { 8 } else { n })).map(|_| rng.prime(bits)).collect();
        let mut comps = vec![];
        let n = n.max(256);
        while comps.len() < n {
            let x = rng.bits(bits, true);
            if !x.is_probable_prime() {
                comps.push(x);
            }
        }
        let odds: Vec<_> = (0..n).map(|_| rng.bits(bits, true)).collect();
        for (kind, set) in [("primes", primes), ("odd composites", comps), ("random odd", odds)] {
            let pool = Pool::new(set);
            let params = format!("{bits} bits, {kind}");
            let words: Vec<u64> = if bits <= 64 { pool.nums.iter().map(lo).collect() } else { vec![] };
            let wides: Vec<u128> = if bits <= 128 { pool.nums.iter().map(to_u128).collect() } else { vec![] };
            let pr = &pool;
            let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| pr.get(i0 + j).is_probable_prime() as u64).sum())];
            #[cfg(has_gmp)]
            {
                let h = pool.gmp();
                cs.push(c("GMP", move |i0, k| unsafe { gmp_op(g::PRIME, h, null(), i0, k) }).note("mpz_probab_prime_p(n, 24) = BPSW"));
            }
            #[cfg(has_flint)]
            {
                let h = pool.flint();
                cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::BPSW, h, null(), i0, k, 0, 0, 0) }).note("fmpz_is_probabprime_BPSW"));
                if bits <= 64 {
                    let w = words.clone();
                    cs.push(c("FLINT n_is_prime", move |i0, k| unsafe { flint_nop(f::N_ISPRIME, w.as_ptr(), w.len(), i0, k) }));
                }
            }
            #[cfg(has_pari)]
            {
                let h = pool.pari();
                cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::ISPSEUDO, h, null(), i0, k, 0) }).note("ispseudoprime(n, 0) = BPSW"));
            }
            if bits <= 64 {
                let w = words.clone();
                cs.push(c("machine-prime", move |i0, k| (0..k).map(|j| machine_prime::is_prime(w[(i0 + j) % w.len()]) as u64).sum()).note("is_prime (ssmr): exact below 2^64"));
                let w = words.clone();
                cs.push(c("primal", move |i0, k| (0..k).map(|j| primal::is_prime(w[(i0 + j) % w.len()]) as u64).sum()).note("primal::is_prime: deterministic Miller-Rabin"));
                let w = words.clone();
                cs.push(c("malachite", move |i0, k| (0..k).map(|j| w[(i0 + j) % w.len()].is_prime() as u64).sum()).note("u64::is_prime (a port of FLINT's n_is_prime)"));
            } else if bits <= 128 {
                let w = wides.clone();
                cs.push(c("machine-prime", move |i0, k| (0..k).map(|j| machine_prime::is_prime_128(w[(i0 + j) % w.len()]) as u64).sum()).note("is_prime_128 (wide): base-2 strong test + Lucas-V"));
            }
            ctx.race(row("is_probable_prime", &params).check(pool.len().min(256)).max_k(4 * pool.len()).same_k(), cs);
        }
    }
}

pub fn next_prev(ctx: &mut Ctx) {
    for &bits in &[32u32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let mut rng = Rng::new(2000 + bits as u64);
        let n = match bits {
            0..=64 => 4096,
            65..=128 => 1024,
            129..=256 => 256,
            257..=512 => 64,
            513..=1024 => 16,
            1025..=2048 => 6,
            _ => 4,
        };
        let pool = Pool::new((0..n).map(|_| rng.bits(bits, false)).collect());
        let words: Vec<u64> = if bits <= 64 { pool.nums.iter().map(lo).collect() } else { vec![] };
        let params = format!("{bits} bits, random");
        let pr = &pool;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&pr.get(i0 + j).next_prime())).fold(0, u64::wrapping_add))];
        #[cfg(has_gmp)]
        {
            let h = pool.gmp();
            cs.push(c("GMP", move |i0, k| unsafe { gmp_op(g::NEXT, h, null(), i0, k) }).note("mpz_nextprime (also one Miller-Rabin test of the prime found)"));
        }
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::NEXT, h, null(), i0, k, 0, 0, 0) }).note("fmpz_nextprime(r, n, 0)"));
            if bits <= 64 {
                let w = words.clone();
                cs.push(c("FLINT n_nextprime", move |i0, k| unsafe { flint_nop(f::N_NEXT, w.as_ptr(), w.len(), i0, k) }));
            }
        }
        // PARI's nextprime(n) is the smallest prime >= n, precprime(n) the largest <= n: given n + 1
        // and n - 1, they find the same primes as ours.
        #[cfg(has_pari)]
        let up = Pool::new(pool.nums.iter().map(|x| x + &num(1)).collect());
        #[cfg(has_pari)]
        let down = Pool::new(pool.nums.iter().map(|x| x - &num(1)).collect());
        #[cfg(has_pari)]
        {
            let h = up.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::NEXT, h, null(), i0, k, 0) }).note("nextprime(n + 1)"));
        }
        ctx.race(row("next_prime", &params).check(if bits >= 2048 { 1 } else { pool.len().min(64) }).max_k(4 * pool.len()).same_k(), cs);

        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&pr.get(i0 + j).prev_prime().unwrap())).fold(0, u64::wrapping_add))];
        #[cfg(has_gmp)]
        {
            let h = pool.gmp();
            cs.push(c("GMP", move |i0, k| unsafe { gmp_op(g::PREV, h, null(), i0, k) }).note("mpz_prevprime"));
        }
        #[cfg(has_pari)]
        {
            let h = down.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::PREC, h, null(), i0, k, 0) }).note("precprime(n - 1)"));
        }
        ctx.race(row("prev_prime", &params).check(if bits >= 2048 { 1 } else { pool.len().min(64) }).max_k(4 * pool.len()).same_k(), cs);
    }
}
