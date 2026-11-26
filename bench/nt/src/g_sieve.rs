//! Sieves, prime counting and the n-th prime: primes(range), prime_pi, nth_prime.
use crate::core::*;
use a::math::numbers::{nth_prime, prime_pi, primes};
#[allow(unused_imports)]
use crate::ffi::*;
use malachite_base::num::factorization::traits::Primes as _;

const P64: u64 = u64::MAX; // 2^64 - 1, the end of [2^64 - w, 2^64)

pub fn sieve(ctx: &mut Ctx) {
    // Counting the primes of a range.
    let ranges: Vec<(u64, u64, &str, bool)> = vec![
        (0, 1_000_000, "[0, 10^6]", false),
        (0, 100_000_000, "[0, 10^8]", false),
        (0, 1_000_000_000, "[0, 10^9]", false),
        (0, 10_000_000_000, "[0, 10^10]", false),
        (1_000_000_000_000, 1_000_000_000_000 + 10_000_000_000, "[10^12, 10^12 + 10^10]", false),
        (1_000_000_000_000_000, 1_000_000_000_000_000 + 10_000_000_000, "[10^15, 10^15 + 10^10]", true),
        (1_000_000_000_000_000_000, 1_000_000_000_000_000_000 + 10_000_000_000, "[10^18, 10^18 + 10^10]", true),
        (10_000_000_000_000_000_000, 10_000_000_000_000_000_000 + 1_000_000_000, "[10^19, 10^19 + 10^9]", false),
        (1_000_000_000_000_000, 1_000_000_000_000_000 + 1_000_000, "[10^15, 10^15 + 10^6]", false),
        (1_000_000_000_000_000_000, 1_000_000_000_000_000_000 + 1_000_000, "[10^18, 10^18 + 10^6]", false),
        (1_000_000_000_000_000_000, 1_000_000_000_000_000_000 + 100_000_000, "[10^18, 10^18 + 10^8]", false),
        (P64 - 1_000_000 + 1, P64, "[2^64 - 10^6, 2^64)", false),
        (P64 - 10_000_000 + 1, P64, "[2^64 - 10^7, 2^64)", false),
        (P64 - 100_000_000 + 1, P64, "[2^64 - 10^8, 2^64)", false),
    ];
    for &(a, b, name, long) in &ranges {
        if !ctx.runs(long) {
            continue;
        }
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| primes(a..=b).count() as u64).sum())];
        #[cfg(has_primesieve)]
        cs.push(c("primesieve", move |_, k| (0..k).map(|_| unsafe { ps_count(a, b) }).sum()).note("primesieve_count_primes, 1 thread"));
        if a == 0 && b <= 10_000_000_000 {
            cs.push(c("primal", move |_, k| (0..k).map(|_| primal::StreamingSieve::prime_pi(b as usize) as u64).sum()).note("StreamingSieve::prime_pi"));
        }
        if a == 0 && b <= 1_000_000_000 {
            cs.push(c("malachite", move |_, k| (0..k).map(|_| u64::primes_less_than_or_equal_to(&b).count() as u64).sum()).note("primes_less_than_or_equal_to(b).count()"));
        }
        ctx.race(row("primes(range).count()", name).check(1), cs);
    }

    // Forming the primes one by one (`next`) and by `fold`, summed.
    let iter_ranges: Vec<(u64, u64, &str)> = vec![
        (0, 1_000_000_000, "[0, 10^9]"),
        (0, 10_000_000_000, "[0, 10^10]"),
        (1_000_000_000_000, 1_000_000_000_000 + 10_000_000_000, "[10^12, 10^12 + 10^10]"),
    ];
    for &(a, b, name) in &iter_ranges {
        if !ctx.runs(false) {
            continue;
        }
        let next = move || {
            let mut it = primes(a..=b);
            let mut s = 0u64;
            while let Some(p) = it.next() {
                s = s.wrapping_add(p);
            }
            s
        };
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| next()).fold(0, u64::wrapping_add))];
        #[cfg(has_primesieve)]
        cs.push(c("primesieve", move |_, k| (0..k).map(|_| unsafe { ps_iter_sum(a, b) }).fold(0, u64::wrapping_add)).note("primesieve_iterator, next_prime"));
        if a == 0 && b <= 1_000_000_000 {
            cs.push(
                c("primal", move |_, k| {
                    (0..k)
                        .map(|_| primal::Sieve::new(b as usize).primes_from(0).take_while(|&p| p as u64 <= b).fold(0u64, |s, p| s.wrapping_add(p as u64)))
                        .fold(0, u64::wrapping_add)
                })
                .note("Sieve::new(b).primes_from(0): the whole sieve kept (33 MB at 10^9)"),
            );
            cs.push(c("malachite", move |_, k| (0..k).map(|_| u64::primes_less_than_or_equal_to(&b).fold(0u64, u64::wrapping_add)).fold(0, u64::wrapping_add)));
        }
        ctx.race(row("primes(range) next, summed", name).check(1), cs);
        let fold = move || primes(a..=b).fold(0u64, u64::wrapping_add);
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| fold()).fold(0, u64::wrapping_add))];
        #[cfg(has_primesieve)]
        cs.push(c("primesieve", move |_, k| (0..k).map(|_| unsafe { ps_iter_sum(a, b) }).fold(0, u64::wrapping_add)).note("primesieve_iterator, next_prime"));
        ctx.race(row("primes(range).fold (sum)", name).check(1), cs);
    }

    // The first primes from a.
    let firsts: Vec<(u64, usize, &str)> = vec![
        (0, 10, "10 from 0"),
        (1_000_000, 10, "10 from 10^6"),
        (1_000_000_000, 10, "10 from 10^9"),
        (1_000_000_000_000, 100, "100 from 10^12"),
        (1_000_000_000_000_000_000, 100, "100 from 10^18"),
        (P64 - 1_000_000_000 + 1, 100, "100 from 2^64 - 10^9"),
    ];
    for &(a, k0, name) in &firsts {
        if !ctx.runs(false) {
            continue;
        }
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| primes(a..).take(k0).fold(0u64, u64::wrapping_add)).fold(0, u64::wrapping_add))];
        #[cfg(has_primesieve)]
        cs.push(c("primesieve", move |_, k| (0..k).map(|_| unsafe { ps_first(a, k0 as u64) }).fold(0, u64::wrapping_add)).note("a new primesieve_iterator from a (jump_to)"));
        if a <= 1_000_000_000 {
            cs.push(c("primal", move |_, k| (0..k).map(|_| primal::Primes::all().skip_while(|&p| (p as u64) < a).take(k0).fold(0u64, |s, p| s.wrapping_add(p as u64))).fold(0, u64::wrapping_add)).note("Primes::all() from 2, skipping to a"));
        }
        ctx.race(row("primes(a..).take(k)", name).check(1), cs);
    }
}

pub fn pi(ctx: &mut Ctx) {
    let xs: Vec<(u64, &str, bool)> = vec![
        (30_000, "3*10^4", false),
        (70_000, "7*10^4", false),
        (100_000, "10^5", false),
        (200_000, "2*10^5", false),
        (1_000_000, "10^6", false),
        (10_000_000, "10^7", false),
        (100_000_000, "10^8", false),
        (1_000_000_000, "10^9", false),
        (10_000_000_000, "10^10", false),
        (1_000_000_000_000, "10^12", false),
        (10_000_000_000_000, "10^13", false),
        (100_000_000_000_000, "10^14", false),
        (1_000_000_000_000_000, "10^15", false),
        (10_000_000_000_000_000, "10^16", false),
        (100_000_000_000_000_000, "10^17", true),
        (1_000_000_000_000_000_000, "10^18", true),
        (u64::MAX, "2^64 - 1", true),
    ];
    for &(x, name, long) in &xs {
        if !ctx.runs(long) {
            continue;
        }
        // The argument changes from call to call (64 values from x up, up to 10^10).
        // The argument changes by up to 6 % from x (within the same regime: 3*10^4 stays below 2^16).
        let step = if x <= 10_000_000_000 { (x / 1000).max(1) } else { 0 };
        let arg = move |i: usize| if x == u64::MAX { x } else { x + (i % 64) as u64 * step };
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| prime_pi(arg(i0 + j))).sum())];
        #[cfg(has_primesieve)]
        cs.push(c("primecount", move |i0, k| (0..k).map(|j| unsafe { pc_pi(arg(i0 + j)) }).sum()).note("primecount_pi, 1 thread"));
        if x <= 10_000_000_000 {
            cs.push(c("primal", move |i0, k| (0..k).map(|j| primal::StreamingSieve::prime_pi(arg(i0 + j) as usize) as u64).sum()).note("StreamingSieve::prime_pi (a sieve)"));
        }
        #[cfg(has_flint)]
        if x <= 100_000_000 {
            let v: Vec<u64> = (0..64).map(arg).collect();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_PRIMEPI, v.as_ptr(), v.len(), i0, k) }).note("n_prime_pi after flint_cleanup: its first call (it keeps what it sieved)"));
        }
        #[cfg(has_pari)]
        if x <= 10_000_000_000 {
            let pool = Pool::new((0..64).map(|i| num(arg(i))).collect());
            let h = pool.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::PRIMEPI, h, null(), i0, k, 0) }).note("primepi"));
        }
        ctx.race(row("prime_pi", name).check(if x >= 100_000_000_000_000 { 1 } else { 4 }), cs);
    }
}

pub fn nth(ctx: &mut Ctx) {
    let ns: Vec<(u64, &str)> = vec![
        (10_000, "10^4"),
        (30_000, "3*10^4"),
        (100_000, "10^5"),
        (1_000_000, "10^6"),
        (10_000_000, "10^7"),
        (1_000_000_000, "10^9"),
        (1_000_000_000_000, "10^12"),
        (100_000_000_000_000, "10^14"),
    ];
    for &(n, name) in &ns {
        if !ctx.runs(false) {
            continue;
        }
        // Spread by up to 6 % from n: PARI's prime(n) goes on from the last prime it found, so that
        // nearby arguments (n + 7i) took it 0.03 ms at 10^9 and arguments 10^6 apart 959 ms.
        let step = (n / 1000).max(7);
        let arg = move |i: usize| n + (i % 64) as u64 * step;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| nth_prime(arg(i0 + j))).sum())];
        #[cfg(has_primesieve)]
        cs.push(c("primecount", move |i0, k| (0..k).map(|j| unsafe { pc_nth(arg(i0 + j)) }).sum()).note("primecount_nth_prime, 1 thread"));
        if n <= 10_000_000 {
            cs.push(c("primal", move |i0, k| (0..k).map(|j| primal::StreamingSieve::nth_prime(arg(i0 + j) as usize) as u64).sum()).note("StreamingSieve::nth_prime (a sieve)"));
        }
        #[cfg(has_flint)]
        if n <= 10_000_000 {
            let v: Vec<u64> = (0..64).map(arg).collect();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_NTH, v.as_ptr(), v.len(), i0, k) }).note("n_nth_prime after flint_cleanup (it sieves from 2)"));
        }
        #[cfg(has_pari)]
        // At 10^9 a call of PARI's took minutes (its sieve from the end of its table, the arguments
        // not increasing): not timed there.
        if n <= 10_000_000 {
            let pool = Pool::new((0..64).map(|i| num(arg(i))).collect());
            let h = pool.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::PRIME, h, null(), i0, k, 0) }).note("prime(n): its table of primes below 2^20, else a sieve from the last prime it found (here up to 6 % away)"));
        }
        ctx.race(row("nth_prime", name).check(if n >= 1_000_000_000 { 1 } else { 4 }), cs);
    }
}
