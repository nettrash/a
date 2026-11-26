//! Arithmetic functions: divisors, divisor_count, divisor_sigma, totient, carmichael, mobius,
//! is_squarefree, radical, on Factorization and on Number.
use crate::core::*;
#[allow(unused_imports)]
use crate::ffi::*;
use a::math::numbers::{ByteOrder, Factorization, Number};

pub fn arith(ctx: &mut Ctx) {
    // Factorisations of 6 random primes of 64 to 200 bits (from 4096 such primes), the first and
    // the fourth squared (144 divisors).
    let mut rng = Rng::new(30_000);
    let primes: Vec<Number> = (0..4096).map(|_| { let b = 64 + rng.below(137) as u32; rng.prime(b) }).collect();
    let ng = 2048;
    let mut facs = vec![];
    let mut flat = vec![];
    let mut exps = vec![];
    let mut goff = vec![0u64];
    for _ in 0..ng {
        let mut ps: Vec<Number> = vec![];
        while ps.len() < 6 {
            let p = primes[rng.below(4096) as usize].clone();
            if !ps.contains(&p) {
                ps.push(p);
            }
        }
        let f: Vec<(Number, u32)> = ps.into_iter().enumerate().map(|(t, p)| (p, if t % 3 == 0 { 2 } else { 1 })).collect();
        for (p, e) in &f {
            flat.push(p.clone());
            exps.push(*e);
        }
        goff.push(flat.len() as u64);
        facs.push(Factorization::new(f, ByteOrder::LittleEndian));
    }
    let pflat = Pool::new(flat);
    #[cfg(has_flint)]
    let fq = unsafe { flint_facpool(pflat.flint(), exps.as_ptr(), goff.as_ptr(), ng) };
    #[cfg(has_pari)]
    let pq = unsafe { pari_facpool(pflat.pari(), exps.as_ptr(), goff.as_ptr(), ng) };
    let _ = (&exps, &goff);
    let fr = &facs;
    let params = "6 primes of 64 to 200 bits, 2 squared (144 divisors)";
    type F = fn(&Factorization) -> u64;
    let fns: Vec<(&str, F, i32, u64, i32, u64)> = vec![
        ("Factorization::totient", |f| lo(&f.totient()), 1, 0, 1, 0),
        ("Factorization::divisor_count", |f| lo(&f.divisor_count()), 3, 0, 2, 0),
        ("Factorization::divisor_sigma(1)", |f| lo(&f.divisor_sigma(1)), 3, 1, 3, 1),
        ("Factorization::divisor_sigma(2)", |f| lo(&f.divisor_sigma(2)), 3, 2, 3, 2),
        ("Factorization::carmichael", |f| lo(&f.carmichael()), 0, 0, 0, 0),
        ("Factorization::mobius", |f| (f.mobius() + 1) as u64, 2, 0, 4, 0),
        ("Factorization::divisors", |f| {
            let d = f.divisors();
            (d.len() as u64).wrapping_add(lo(d.last().unwrap()))
        }, 0, 0, 5, 0),
        ("Factorization::radical", |f| lo(&f.radical()), 0, 0, 6, 0),
        ("Factorization::value", |f| lo(&f.value()), 0, 0, 8, 0),
    ];
    for (name, fun, fop, farg, pop, parg) in fns {
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| fun(&fr[(i0 + j) % ng])).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        if fop != 0 {
            let note = match fop {
                1 => "fmpz_factor_euler_phi",
                2 => "fmpz_factor_moebius_mu",
                _ => "fmpz_factor_divisor_sigma",
            };
            cs.push(c("FLINT", move |i0, k| unsafe { flint_facop(fop, fq, i0, k, farg) }).note(note));
        }
        #[cfg(has_pari)]
        if pop != 0 {
            let note = match pop {
                1 => "eulerphi(fa)",
                2 => "numdiv(fa)",
                3 => "sumdivk(fa, k)",
                4 => "moebius(fa)",
                5 => "divisors(fa)",
                6 => "factorback of the primes",
                _ => "factorback(fa)",
            };
            cs.push(c("PARI", move |i0, k| unsafe { pari_facop(pop, pq, null(), i0, k, parg) }).note(note));
        }
        let _ = (fop, farg, pop, parg);
        ctx.race(row(name, params).check(64), cs);
    }

    // Divisors of highly composite numbers.
    let hc: Vec<(&str, Vec<(u64, u32)>)> = vec![
        ("720720 (240 divisors)", vec![(2, 4), (3, 2), (5, 1), (7, 1), (11, 1), (13, 1)]),
        ("2^6 3^4 5^2 7^2 11 13 17 19 23 (10,080 divisors)", vec![(2, 6), (3, 4), (5, 2), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1)]),
        ("897,612,484,786,617,600 (103,680 divisors)", vec![(2, 8), (3, 4), (5, 2), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (37, 1)]),
    ];
    for (name, f) in &hc {
        let fac = Factorization::new(f.iter().map(|&(p, e)| (num(p), e)).collect(), ByteOrder::LittleEndian);
        let n = fac.value();
        let (rf, rn) = (&fac, &n);
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| rn.divisors().len() as u64).sum())];
        #[cfg(has_flint)]
        {
            let pool = Pool::new(vec![n.clone()]);
            let h = pool.flint();
            cs.push(c("FLINT", move |_, k| unsafe { flint_op(f::DIVISORS, h, null(), 0, k, 0, 0, 0) }).note("arith_divisors(n): factors n"));
        }
        ctx.race(row("divisors (factors n)", name).check(1), cs);
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| {
            let d = rf.divisors();
            (d.len() as u64).wrapping_add(lo(d.last().unwrap()))
        }).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            let pp = Pool::new(f.iter().map(|&(p, _)| num(p)).collect());
            let ex: Vec<u32> = f.iter().map(|&(_, e)| e).collect();
            let go = [0u64, f.len() as u64];
            let q = unsafe { pari_facpool(pp.pari(), ex.as_ptr(), go.as_ptr(), 1) };
            cs.push(c("PARI", move |_, k| unsafe { pari_facop(p::F_DIVISORS, q, null(), 0, k, 0) }).note("divisors(fa)"));
        }
        ctx.race(row("Factorization::divisors", name).check(1), cs);
    }

    // The functions of random words, which factor them.
    let mut rng = Rng::new(31_000);
    let words: Vec<u64> = (0..4096).map(|_| rng.next()).collect();
    let pool = Pool::new(words.iter().map(|&w| num(w)).collect());
    let pr = &pool;
    type G = fn(&Number) -> u64;
    let wf: Vec<(&str, G, i32, i32)> = vec![
        ("totient", |x| lo(&x.totient()), 5, 17),
        ("mobius", |x| (x.mobius() + 1) as u64, 6, 18),
        ("is_squarefree", |x| x.is_squarefree() as u64, 7, 19),
        ("divisor_sigma(1)", |x| lo(&x.divisor_sigma(1)), 0, 24),
        ("divisor_count", |x| lo(&x.divisor_count()), 0, 25),
        ("carmichael", |x| lo(&x.carmichael()), 0, 0),
        ("radical", |x| lo(&x.radical()), 0, 0),
        ("divisors", |x| x.divisors().len() as u64, 0, 0),
    ];
    for (name, fun, fop, pop) in wf {
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| fun(pr.get(i0 + j))).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        if fop != 0 {
            let w = words.clone();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(fop, w.as_ptr(), w.len(), i0, k) }).note(match fop {
                5 => "n_euler_phi",
                6 => "n_moebius_mu",
                _ => "n_is_squarefree",
            }));
        }
        #[cfg(has_pari)]
        if pop != 0 {
            let h = pool.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(pop, h, null(), i0, k, 0) }).note(match pop {
                17 => "eulerphi(n)",
                18 => "moebius(n)",
                19 => "Z_issquarefree(n)",
                24 => "sumdiv(n)",
                _ => "numdiv(n)",
            }));
        }
        let _ = (fop, pop);
        ctx.race(row(name, "random words (u64)").check(256).same_k(), cs);
    }

    // Longer numbers: totient and mobius of random 128-bit numbers (factored on the way).
    let mut rng = Rng::new(32_000);
    let nums: Vec<Number> = (0..64).map(|_| rng.bits(128, false)).collect();
    let pool = Pool::new(nums);
    let pr = &pool;
    let n = pool.len();
    for (name, fun, fop, pop) in [("totient", (|x: &Number| lo(&x.totient())) as G, 17, 17), ("mobius", (|x: &Number| (x.mobius() + 1) as u64) as G, 18, 18)] {
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| pr.nums.iter().map(fun).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |_, k| (0..k).map(|_| unsafe { flint_op(fop, h, null(), 0, n, 0, 0, 0) }).fold(0, u64::wrapping_add)).note("fmpz_factor, then the function of the factorisation"));
        }
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |_, k| (0..k).map(|_| unsafe { pari_op(pop, h, null(), 0, n, 0) }).fold(0, u64::wrapping_add)).note(if pop == 17 { "eulerphi(n)" } else { "moebius(n): stops at a square factor" }));
        }
        let _ = (fop, pop);
        ctx.race(row(name, "random 128-bit numbers, mean of 64").per(n as f64).check(1).max_k(1), cs);
    }
}
