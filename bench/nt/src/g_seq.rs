//! Sequences: fibonacci, lucas, lucas_sequence, lucas_sequence_mod, primorial, partitions.
use crate::core::*;
#[allow(unused_imports)]
use crate::ffi::*;
use a::math::numbers::{ByteOrder, Number};
use malachite_base::num::number_theory::traits::{Fibonacci as _, LucasNumber as _, Primorial as _};
use malachite_nz::natural::Natural;

fn nat_lo(x: &Natural) -> u64 {
    x.limbs().next().unwrap_or(0)
}

const LE: ByteOrder = ByteOrder::LittleEndian;

/// Four consecutive indices from `n`, in turn, so that no branch is learnt.
fn idx(n: u64) -> Vec<u64> {
    (0..4).map(|t| n + t).collect()
}

fn ename(n: u64) -> String {
    let mut x = n;
    let mut e = 0;
    while x >= 10 && x % 10 == 0 {
        x /= 10;
        e += 1;
    }
    if e >= 4 { if x == 1 { format!("10^{e}") } else { format!("{x}*10^{e}") } } else { n.to_string() }
}

pub fn fib(ctx: &mut Ctx) {
    for &n in &[100u64, 187, 300, 374, 500, 748, 1000, 1500, 2000, 3000, 10_000, 100_000, 1_000_000, 10_000_000] {
        let ns = idx(n);
        let r = &ns;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::fibonacci(r[(i0 + j) % 4], LE))).fold(0, u64::wrapping_add))];
        #[cfg(has_gmp)]
        cs.push(c("GMP", move |i0, k| unsafe { gmp_seq(g::FIB, r.as_ptr(), 4, i0, k) }).note("mpz_fib_ui into a result allocated before"));
        #[cfg(has_flint)]
        cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_FIB, r.as_ptr(), 4, i0, k) }).note("fmpz_fib_ui into a result allocated before"));
        cs.push(c("malachite", move |i0, k| (0..k).map(|j| nat_lo(&Natural::fibonacci(r[(i0 + j) % 4]))).fold(0, u64::wrapping_add)).note("Natural::fibonacci (a new result, as ours)"));
        ctx.race(row("fibonacci", &format!("n = {}", ename(n))).check(4), cs);
    }
    for &n in &[100u64, 185, 300, 370, 500, 740, 1000, 3000, 10_000, 100_000, 1_000_000] {
        let ns = idx(n);
        let r = &ns;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::lucas(r[(i0 + j) % 4], LE))).fold(0, u64::wrapping_add))];
        #[cfg(has_gmp)]
        cs.push(c("GMP", move |i0, k| unsafe { gmp_seq(g::LUC, r.as_ptr(), 4, i0, k) }).note("mpz_lucnum_ui into a result allocated before"));
        cs.push(c("malachite", move |i0, k| (0..k).map(|j| nat_lo(&Natural::lucas_number(r[(i0 + j) % 4]))).fold(0, u64::wrapping_add)).note("Natural::lucas_number"));
        ctx.race(row("lucas", &format!("n = {}", ename(n))).check(4), cs);
    }
    // Lucas sequences, exactly.
    let params: Vec<(i64, i64)> = vec![(1, -1), (3, 1), (2, -1), (5, 3), (100_003, -999_331)];
    for &n in &[10u64, 100, 1000, 10_000, 100_000] {
        let ns = idx(n);
        for &(pp, qq) in &params {
            let (pn, qn) = (Number::new_from_i64(pp, LE), Number::new_from_i64(qq, LE));
            let (rp, rq, r) = (&pn, &qn, &ns);
            let mut cs = vec![c("a", move |i0, k| {
                (0..k)
                    .map(|j| {
                        let (u, v) = Number::lucas_sequence(rp, rq, r[(i0 + j) % 4]);
                        lo(&u).wrapping_add(lo(&v))
                    })
                    .fold(0, u64::wrapping_add)
            })];
            #[cfg(has_gmp)]
            if (pp, qq) == (1, -1) {
                cs.push(c("GMP", move |i0, k| unsafe { gmp_seq(g::FIB2L, r.as_ptr(), 4, i0, k) }).note("mpz_fib2_ui and L_n = F_n + 2 F_(n-1)"));
            } else if (pp, qq) == (3, 1) {
                let n2: Vec<u64> = r.iter().map(|x| 2 * x).collect();
                cs.push(c("GMP", move |i0, k| unsafe { gmp_seq(g::FIB2L, n2.as_ptr(), 4, i0, k) }).note("F_2n and L_2n by mpz_fib2_ui at 2n"));
            }
            ctx.race(row("lucas_sequence", &format!("({pp}, {qq}), n = {}", ename(n))).check(4), cs);
        }
    }
}

pub fn lucas_mod(ctx: &mut Ctx) {
    for &bits in &[64u32, 128, 256, 512, 1024, 2048, 4096] {
        let mut rng = Rng::new(40_000 + bits as u64);
        let nm = if bits >= 2048 { 4 } else { 16 };
        let mods: Vec<Number> = (0..nm).map(|_| rng.prime(bits)).collect();
        let kinds: [(&str, Option<(i64, i64)>); 4] = [("(1, -1)", Some((1, -1))), ("(3, 1)", Some((3, 1))), ("(1, -5)", Some((1, -5))), ("random P, Q", None)];
        for (kname, pq) in kinds {
            let ni = 64;
            let mut items = vec![]; // (P, Q, n, m), ours with Q as given
            let mut four = vec![]; // (P, Q mod m, n, m) for FLINT
            for i in 0..ni {
                let m = mods[i % nm].clone();
                let (p, q) = match pq {
                    Some((a, b)) => (Number::new_from_i64(a, LE), Number::new_from_i64(b, LE)),
                    None => (rng.below_num(&m), rng.below_num(&m)),
                };
                let n = rng.bits(bits, false);
                let qm = &(&(&q % &m) + &m) % &m;
                four.extend([p.clone(), qm, n.clone(), m.clone()]);
                items.push((p, q, n, m));
            }
            let it = &items;
            let mut cs = vec![c("a", move |i0, k| {
                (0..k)
                    .map(|j| {
                        let (p, q, n, m) = &it[(i0 + j) % ni];
                        lo(&Number::lucas_sequence_mod(p, q, n, m).1)
                    })
                    .fold(0, u64::wrapping_add)
            })];
            #[cfg(has_flint)]
            {
                let pool = Pool::new(four);
                let h = pool.flint();
                cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::LUCCHAIN, h, null(), i0, k, 0, 0, 0) }).note("fmpz_lucas_chain_full: V_n and V_(n+1) only (ours U_n and V_n)"));
                if pq == Some((3, 1)) {
                    cs.push(c("FLINT Q = 1", move |i0, k| unsafe { flint_op(f::LUCCHAIN1, h, null(), i0, k, 0, 0, 0) }).note("fmpz_lucas_chain (Q = 1): V_n and V_(n+1) only"));
                }
            }
            ctx.race(row("lucas_sequence_mod", &format!("{kname}, prime of {bits} bits, index of {bits} bits")).check(16), cs);
        }
    }
}

pub fn primorial(ctx: &mut Ctx) {
    for &n in &[100u64, 1000, 10_000, 100_000, 1_000_000, 10_000_000] {
        let ns = idx(n);
        let r = &ns;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::primorial(r[(i0 + j) % 4], LE))).fold(0, u64::wrapping_add))];
        #[cfg(has_gmp)]
        cs.push(c("GMP", move |i0, k| unsafe { gmp_seq(g::PRIMORIAL, r.as_ptr(), 4, i0, k) }).note("mpz_primorial_ui into a result allocated before"));
        #[cfg(has_flint)]
        cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_PRIMORIAL, r.as_ptr(), 4, i0, k) }).note("fmpz_primorial (keeps its table of primes)"));
        cs.push(c("malachite", move |i0, k| (0..k).map(|j| nat_lo(&Natural::primorial(r[(i0 + j) % 4]))).fold(0, u64::wrapping_add)).note("Natural::primorial"));
        ctx.race(row("primorial", &format!("n = {}", ename(n))).check(4), cs);
    }
}

pub fn partitions(ctx: &mut Ctx) {
    for &n in &[417u64, 500, 1000, 2000, 10_000, 100_000, 1_000_000, 10_000_000, 30_000_000, 100_000_000, 1_000_000_000] {
        let ns = idx(n);
        let r = &ns;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::partitions(r[(i0 + j) % 4], LE))).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_PART, r.as_ptr(), 4, i0, k) }).note("partitions_fmpz_ui, its caches (pi, logarithms) kept between calls"));
            if n >= 10_000_000 {
                cs.push(c("FLINT, caches cleared", move |i0, k| unsafe { flint_nop(f::N_PART_CLEAN, r.as_ptr(), 4, i0, k) }).note("flint_cleanup before each call"));
            }
        }
        #[cfg(has_pari)]
        if n <= 100_000_000 {
            let pool = Pool::new(r.iter().map(|&x| num(x)).collect());
            let h = pool.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::NUMBPART, h, null(), i0, k, 0) }).note("numbpart"));
        }
        ctx.race(row("partitions", &format!("n = {}", ename(n))).check(4), cs);
    }
}
