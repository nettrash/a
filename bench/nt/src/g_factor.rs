//! Factorisation: factor (words and longer), siqs, trial, rho, pm1, pp1, ecm, perfect powers.
use crate::core::*;
#[allow(unused_imports)]
use crate::ffi::*;
use a::math::numbers::{Number, factor};
use malachite_base::num::factorization::traits::{ExpressAsPower as _, Factor as _};

fn fac_sum(f: &a::math::numbers::Factorization) -> u64 {
    f.factors().iter().map(|(p, e)| lo(p).wrapping_mul(*e as u64)).fold(0, u64::wrapping_add)
}

fn pow10(d: u32) -> Number {
    num(10).pow(d)
}

/// A random number of exactly `d` decimal digits.
pub fn digits(rng: &mut Rng, d: u32) -> Number {
    let lo10 = pow10(d - 1);
    let span = &pow10(d) - &lo10;
    &rng.below_num(&span) + &lo10
}

/// A random prime of exactly `d` digits.
pub fn prime_digits(rng: &mut Rng, d: u32) -> Number {
    loop {
        let p = digits(rng, d).next_prime();
        if p < pow10(d) {
            return p;
        }
    }
}

/// A product of two random primes of `d/2` and `d - d/2` digits, with exactly `d` digits.
pub fn semiprime_digits(rng: &mut Rng, d: u32) -> Number {
    loop {
        let p = prime_digits(rng, d / 2);
        let q = prime_digits(rng, d - d / 2);
        let n = &p * &q;
        if n >= pow10(d - 1) && n < pow10(d) && p != q {
            return n;
        }
    }
}

/// A product of two random primes of `a` and `b` bits.
pub fn semiprime_bits(rng: &mut Rng, a: u32, b: u32) -> Number {
    loop {
        let p = rng.prime(a);
        let q = rng.prime(b);
        if p != q {
            return &p * &q;
        }
    }
}

pub fn factor_words(ctx: &mut Ctx) {
    let kinds: Vec<(&str, Box<dyn Fn(&mut Rng) -> u64>)> = vec![
        ("random of 64 bits", Box::new(|r: &mut Rng| r.next() | 1 << 63)),
        ("primes of 16 and 16 bits", Box::new(|r: &mut Rng| lo(&semiprime_bits(r, 16, 16)))),
        ("20 and 20", Box::new(|r: &mut Rng| lo(&semiprime_bits(r, 20, 20)))),
        ("24 and 24", Box::new(|r: &mut Rng| lo(&semiprime_bits(r, 24, 24)))),
        ("28 and 28", Box::new(|r: &mut Rng| lo(&semiprime_bits(r, 28, 28)))),
        ("32 and 32", Box::new(|r: &mut Rng| lo(&semiprime_bits(r, 32, 32)))),
        ("20 and 44", Box::new(|r: &mut Rng| lo(&semiprime_bits(r, 20, 44)))),
    ];
    for (name, mk) in &kinds {
        let mut rng = Rng::new(3000 + name.len() as u64);
        let words: Vec<u64> = (0..2048).map(|_| mk(&mut rng)).collect();
        let pool = Pool::new(words.iter().map(|&w| num(w)).collect());
        let pr = &pool;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| fac_sum(&pr.get(i0 + j).factor())).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            let w = words.clone();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_FACTOR, w.as_ptr(), w.len(), i0, k) }).note("n_factor"));
        }
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::FACTOR, h, null(), i0, k, 0) }).note("Z_factor, each number fresh"));
        }
        let w = words.clone();
        cs.push(c("malachite", move |i0, k| (0..k).map(|j| w[(i0 + j) % w.len()].factor().into_iter().map(|(p, e)| p.wrapping_mul(e as u64)).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add)).note("u64::factor (a port of FLINT's n_factor)"));
        let w = words.clone();
        cs.push(
            c("machine-factor", move |i0, k| {
                (0..k)
                    .map(|j| {
                        let f = machine_factor::factorize(w[(i0 + j) % w.len()]);
                        (0..f.len).map(|t| f.factors[t].wrapping_mul(f.powers[t] as u64)).fold(0, u64::wrapping_add)
                    })
                    .fold(0, u64::wrapping_add)
            })
            .note("factorize"),
        );
        ctx.race(row("factor (u64)", name).check(256).max_k(2048).same_k(), cs);
    }
}

/// Numbers with many medium prime factors: 2 times primes of `qb` bits up to `bits` bits (the
/// `p - 1` of the smooth primes of the discrete logarithms).
pub fn factor_smooth(ctx: &mut Ctx) {
    for &(bits, qb, n) in &[(64u32, 20u32, 64usize), (128, 20, 32), (128, 32, 16), (256, 24, 16), (256, 32, 8), (512, 28, 4), (512, 40, 2)] {
        let params = format!("2 times primes of {qb} bits, {bits} bits, mean of {n}");
        if !ctx.wants("factor", &params) {
            continue;
        }
        let mut rng = Rng::new(4500 + bits as u64 * 100 + qb as u64);
        let nums: Vec<Number> = (0..n)
            .map(|_| {
                let mut m = num(2);
                while (m.bits() as u32) + qb <= bits {
                    m = &m * &rng.prime(qb);
                }
                m
            })
            .collect();
        let pool = Pool::new(nums);
        let pr = &pool;
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| pr.nums.iter().map(|x| fac_sum(&x.factor())).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |_, k| (0..k).map(|_| unsafe { pari_op(p::FACTOR, h, null(), 0, n, 0) }).fold(0, u64::wrapping_add)).note("Z_factor"));
        }
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |_, k| (0..k).map(|_| unsafe { flint_op(f::FACTOR, h, null(), 0, n, 0, 0, 0) }).fold(0, u64::wrapping_add)).note("fmpz_factor"));
        }
        ctx.race(row("factor", &params).per(n as f64).check(1).max_k(1).notes("the same set every sample: the mean over the set"), cs);
    }
}

pub fn factor_long(ctx: &mut Ctx) {
    let sets: Vec<(&str, u32, usize)> = vec![
        ("semiprimes", 20, 48),
        ("semiprimes", 24, 48),
        ("semiprimes", 28, 48),
        ("semiprimes", 32, 32),
        ("semiprimes", 36, 24),
        ("semiprimes", 40, 16),
        ("semiprimes", 44, 12),
        ("semiprimes", 50, 6),
        ("semiprimes", 60, 3),
        ("random", 20, 256),
        ("random", 30, 256),
        ("random", 40, 96),
        ("random", 50, 32),
        ("random", 60, 16),
    ];
    for &(kind, d, n) in &sets {
        let params = format!("{kind} of {d} digits, mean of {n}");
        if !ctx.wants("factor", &params) {
            continue;
        }
        let mut rng = Rng::new(4000 + d as u64 + if kind == "random" { 100 } else { 0 });
        let nums: Vec<Number> = (0..n).map(|_| if kind == "random" { digits(&mut rng, d) } else { semiprime_digits(&mut rng, d) }).collect();
        let pool = Pool::new(nums);
        let pr = &pool;
        // Each call factors the whole set: the time per number is its mean.
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| pr.nums.iter().map(|x| fac_sum(&x.factor())).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |_, k| (0..k).map(|_| unsafe { pari_op(p::FACTOR, h, null(), 0, n, 0) }).fold(0, u64::wrapping_add)).note("Z_factor"));
        }
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |_, k| (0..k).map(|_| unsafe { flint_op(f::FACTOR, h, null(), 0, n, 0, 0, 0) }).fold(0, u64::wrapping_add)).note("fmpz_factor"));
        }
        ctx.race(row("factor", &params).per(n as f64).check(1).max_k(1).notes("the same set every sample: the mean over the set"), cs);
    }
}

/// msieve's factors from its output lines `pNN: <factor>` and `prpNN: <factor>`.
fn msieve_run(n: &Number, dir: &str) -> u64 {
    let out = std::process::Command::new("msieve")
        .args(["-q", "-t", "1", "-s", &format!("{dir}/msieve.dat"), "-l", &format!("{dir}/msieve.log")])
        .arg(n.to_string())
        .current_dir(dir)
        .output()
        .expect("msieve");
    let s = String::from_utf8_lossy(&out.stdout);
    let mut sum = 0u64;
    for line in s.lines() {
        if let Some((tag, v)) = line.split_once(": ") {
            // `p13`, `prp20` (probable primes), `c30` (composites).
            let digits = tag.trim_start_matches("prp").trim_start_matches(['p', 'c']);
            if tag.len() > digits.len() && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                sum = sum.wrapping_add(lo(&parse(v.trim())));
            }
        }
    }
    let _ = std::fs::remove_file(format!("{dir}/msieve.dat"));
    sum
}

fn have_msieve() -> bool {
    std::process::Command::new("msieve").arg("-h").output().is_ok()
}

pub fn siqs(ctx: &mut Ctx) {
    let sizes: Vec<(u32, usize, bool)> = vec![
        (26, 16, false),
        (28, 16, false),
        (30, 16, false),
        (32, 16, false),
        (36, 12, false),
        (40, 8, false),
        (44, 6, false),
        (45, 4, false),
        (50, 3, false),
        (55, 2, false),
        (60, 1, false),
        (65, 1, false),
        (70, 1, false),
        (75, 1, true),
        (80, 1, true),
    ];
    let dir = std::env::var("NT_SCRATCH").unwrap_or_else(|_| std::env::temp_dir().display().to_string());
    let dir = format!("{dir}/msieve");
    let _ = std::fs::create_dir_all(&dir);
    let msieve = have_msieve();
    // msieve's own start: a run on a small prime, the median of 9.
    let start_ns = if msieve {
        let mut v: Vec<f64> = (0..9)
            .map(|_| {
                let t = std::time::Instant::now();
                msieve_run(&num(1_000_003), &dir);
                t.elapsed().as_nanos() as f64
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[4]
    } else {
        0.0
    };
    for &(d, n, long) in &sizes {
        if !ctx.runs(long) {
            continue;
        }
        let params = format!("{d} digits, mean of {n} balanced semiprimes");
        if !ctx.wants("siqs", &params) {
            continue;
        }
        let mut rng = Rng::new(5000 + d as u64);
        let pool = Pool::new((0..n).map(|_| semiprime_digits(&mut rng, d)).collect());
        let pr = &pool;
        let sum2 = |f: &Number, x: &Number| lo(f).wrapping_add(lo(&(x / f)));
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| pr.nums.iter().map(|x| sum2(&factor::siqs(x).unwrap(), x)).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |_, k| (0..k).map(|_| unsafe { pari_op(p::MPQS, h, null(), 0, n, 0) }).fold(0, u64::wrapping_add)).note("factorint(n, 14): MPQS alone"));
        }
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |_, k| (0..k).map(|_| unsafe { flint_op(f::QSIEVE, h, null(), 0, n, 0, 0, 0) }).fold(0, u64::wrapping_add)).note("qsieve_factor, 1 thread"));
        }
        if msieve {
            let dir = dir.clone();
            let mut cm = c("msieve", move |_, k| (0..k).map(|_| pr.nums.iter().map(|x| msieve_run(x, &dir)).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add))
                .note(&format!("msieve 1.53 -q -t 1, a process a number, its start ({:.1} ms) not subtracted", start_ns / 1e6));
            if d < 40 {
                cm.note.push_str("; the start dominates below 40 digits");
            }
            cs.push(cm);
        }
        ctx.race(row("siqs", &params).per(n as f64).check(1).max_k(1), cs);
    }
}

pub fn trial(ctx: &mut Ctx) {
    let bounds: [(u64, u64, &str); 4] = [(1000, 168, "1000"), (8191, 1028, "8191"), (100_000, 9592, "10^5"), (1_000_000, 78498, "10^6")];
    for &bits in &[64u32, 128, 256, 1024, 4096] {
        let mut rng = Rng::new(6000 + bits as u64);
        // Numbers without a prime factor up to 10^6, so that every prime is tried.
        let mut nums = vec![];
        while nums.len() < 64 {
            let x = if bits == 64 { semiprime_bits(&mut rng, 32, 32) } else { rng.bits(bits, true) };
            if factor::trial(&x, 1_000_000).0.is_empty() {
                nums.push(x);
            }
        }
        let pool = Pool::new(nums);
        for &(b, np, bname) in &bounds {
            let pr = &pool;
            let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| factor::trial(pr.get(i0 + j), b).0.len() as u64).sum())];
            #[cfg(has_flint)]
            {
                let h = pool.flint();
                cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::TRIAL, h, null(), i0, k, np, 0, 0) }).note("fmpz_factor_trial_range(n, 0, pi(bound))"));
            }
            let _ = np;
            ctx.race(row("trial", &format!("{bits} bits, to {bname}")).check(8), cs);
        }
    }
}

pub fn rho(ctx: &mut Ctx) {
    for &fb in &[20u32, 24, 28, 32, 36, 40] {
        let n = if fb >= 36 { 8 } else { 16 };
        let mut rng = Rng::new(7000 + fb as u64);
        let pool = Pool::new((0..n).map(|_| semiprime_bits(&mut rng, fb, 128 - fb)).collect());
        let pr = &pool;
        let mut cs = vec![c("a", move |_, k| (0..k).map(|_| pr.nums.iter().map(|x| lo_minpair(&factor::rho(x, 1 << 40).unwrap(), x)).fold(0, u64::wrapping_add)).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |_, k| (0..k).map(|_| unsafe { flint_op(f::RHO, h, null(), 0, n, 1 << 26, 0, 0) }).fold(0, u64::wrapping_add)).note("fmpz_factor_pollard_brent, one try of 2^26 steps, again until it finds"));
        }
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |_, k| (0..k).map(|_| unsafe { pari_op(p::RHO, h, null(), 0, n, 1 << 26, ) }).fold(0, u64::wrapping_add)).note("Z_pollardbrent(n, 2^26, seed), seeds 1, 2, ... until it finds").nocheck());
        }
        ctx.race(row("rho", &format!("128-bit semiprimes, a prime of {fb} bits, mean of {n}")).per(n as f64).check(1).max_k(1), cs);
    }
}

pub fn pm1_pp1(ctx: &mut Ctx) {
    for &bits in &[128u32, 256, 512, 1024] {
        let mut rng = Rng::new(8000 + bits as u64);
        let pool = Pool::new((0..2).map(|_| semiprime_bits(&mut rng, bits / 2, bits / 2)).collect());
        for &b1 in &[100_000u64, 1_000_000] {
            for &both in &[false, true] {
                let b2 = if both { 100 * b1 } else { 0 };
                let params = format!("{bits} bits, B1 = {}, {}", if b1 == 100_000 { "10^5" } else { "10^6" }, if both { "B2 = 100 B1" } else { "stage 1" });
                let pr = &pool;
                let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| factor::pm1(pr.get(i0 + j), b1, b2).is_some() as u64).sum())];
                #[cfg(has_ecm)]
                {
                    let h = pool.gmp();
                    cs.push(c("GMP-ECM", move |i0, k| unsafe { ecm_op(e::PM1, h, i0, k, 0, b1 as f64, b2 as f64) }).note("-pm1, x0 = 3"));
                }
                ctx.race(row("pm1", &params).check(1).max_k(16), cs);

                let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| factor::pp1(pr.get(i0 + j), b1, b2).is_some() as u64).sum())];
                #[cfg(has_ecm)]
                {
                    let h = pool.gmp();
                    cs.push(c("GMP-ECM", move |i0, k| unsafe { ecm_op(e::PP1, h, i0, k, 0, b1 as f64, b2 as f64) }).note("-pp1 with the three starting values 2/7, 6/5, 6 of pp1"));
                }
                #[cfg(has_flint)]
                if both {
                    let h = pool.flint();
                    let b2s = ((100 * b1) as f64).sqrt() as u64;
                    cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::PP1, h, null(), i0, k, 0, b1, b2s) }).note("fmpz_factor_pp1, c = 3, 5, 7, B2_sqrt = sqrt(100 B1)"));
                }
                ctx.race(row("pp1 (3 starting values)", &params).check(1).max_k(16), cs);
            }
        }
    }
}

pub fn ecm(ctx: &mut Ctx) {
    // Microseconds per unit of B1 for one curve's stage 1, roughly, to choose the curves a call.
    let per_b1 = |bits: u32| match bits {
        0..=128 => 0.04,
        129..=256 => 0.16,
        257..=512 => 0.4,
        _ => 1.4,
    };
    let mut grid: Vec<(u32, u64, u64, &str)> = vec![];
    for &bits in &[128u32, 256, 512, 1024] {
        for &b1 in &[11_000u64, 50_000, 250_000] {
            grid.push((bits, b1, 0, "stage 1"));
        }
        for &b1 in &[11_000u64, 50_000, 250_000] {
            grid.push((bits, b1, 100 * b1, "B2 = 100 B1"));
        }
        if bits == 256 || bits == 512 {
            grid.push((bits, 1_000_000, 100_000_000, "B2 = 100 B1"));
        }
    }
    if ctx.only_long {
        grid.clear();
    }
    // GMP-ECM's levels: the B1 and B2 of its table for factors of 20 to 40 digits.
    for &bits in &[256u32, 512, 1024] {
        for &(b1, b2, lv) in &[(11_000u64, 1_900_000u64, "p20 level"), (50_000, 13_000_000, "p25 level"), (250_000, 130_000_000, "p30 level"), (1_000_000, 1_000_000_000, "p35 level"), (3_000_000, 5_700_000_000, "p40 level")] {
            if b1 == 3_000_000 && bits == 1024 {
                continue;
            }
            if !ctx.runs(bits == 1024 && b1 >= 1_000_000) {
                continue;
            }
            grid.push((bits, b1, b2, lv));
        }
    }
    for &(bits, b1, b2, what) in &grid {
        let params = format!("{bits} bits, B1 = {b1}, {what}{}", if b2 > 0 && !what.starts_with("B2") { format!(" (B2 = {b2})") } else { String::new() });
        if !ctx.wants("ecm", &params) {
            continue;
        }
        let t_curve = per_b1(bits) * b1 as f64 * 1e-6;
        let curves: u32 = if t_curve < 0.02 { 8 } else if t_curve < 0.2 { 4 } else { 2 };
        // Semiprimes that our curves do not factor (a curve that finds a factor stops the call
        // early): at 128 bits the primes of 64 bits are within the curves' reach from B1 = 5*10^4.
        let mut rng = Rng::new(9000 + bits as u64);
        let mut nums = vec![];
        while nums.len() < 2 {
            let n = semiprime_bits(&mut rng, bits / 2, bits / 2);
            if bits > 128 || factor::ecm(&n, b1, b2, curves).is_none() {
                nums.push(n);
            }
        }
        let pool = Pool::new(nums);
        let pr = &pool;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| factor::ecm(pr.get(i0 + j), b1, b2, curves).is_some() as u64).sum()).per(curves as f64)];
        #[cfg(has_ecm)]
        {
            let h = pool.gmp();
            cs.push(c("GMP-ECM", move |i0, k| unsafe { ecm_op(e::ECM, h, i0, k, (1u64 << 32) | 1000, b1 as f64, b2 as f64) }).note("-param 1 (its default here), one curve a call; parameters (and its batch product for B1) kept across calls"));
        }
        #[cfg(has_flint)]
        if what == "B2 = 100 B1" && b1 <= 50_000 {
            let h = pool.flint();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::ECM, h, null(), i0, k, curves as u64, b1, b2) }).per(curves as f64).note(&format!("fmpz_factor_ecm, its own curves, {curves} a call")));
        }
        let r = row("ecm (per curve)", &params).check(1).max_k(16).notes(&format!("ours {curves} curves a call, per curve; semiprimes the curves do not factor"));
        ctx.race(r, cs);
    }
}

pub fn perfect_power(ctx: &mut Ctx) {
    let mut shapes: Vec<(String, Box<dyn Fn(&mut Rng) -> Number>)> = vec![];
    for &b in &[64u32, 128, 256, 1024, 4096] {
        shapes.push((format!("random odd, {b} bits"), Box::new(move |r: &mut Rng| r.bits(b, true))));
    }
    for &b in &[64u32, 128, 256, 1024] {
        shapes.push((format!("prime, {b} bits"), Box::new(move |r: &mut Rng| r.prime(b))));
    }
    for &b in &[21u32, 85, 341, 1365] {
        shapes.push((format!("cube of a prime of {b} bits"), Box::new(move |r: &mut Rng| r.prime(b).pow(3))));
    }
    for &b in &[32u32, 64, 128, 512, 2048] {
        shapes.push((format!("square of a prime of {b} bits"), Box::new(move |r: &mut Rng| r.prime(b).pow(2))));
    }
    for &b in &[32u32, 512, 2048] {
        shapes.push((format!("square of {b} bits plus 2"), Box::new(move |r: &mut Rng| &r.prime(b).pow(2) + &num(2))));
    }
    for (name, mk) in &shapes {
        if !ctx.wants("perfect_power", name) && !ctx.wants("is_perfect_power", name) {
            continue;
        }
        let mut rng = Rng::new(9500 + name.len() as u64 * 31 + name.as_bytes()[name.len() - 6] as u64);
        let pool = Pool::new((0..64).map(|_| mk(&mut rng)).collect());
        let small = pool.nums.iter().all(|x| x.bits() <= 64);
        let pr = &pool;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| pr.get(i0 + j).perfect_power().map_or(0, |(_, e)| e as u64)).sum())];
        #[cfg(has_pari)]
        {
            let h = pool.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::ISPOWER, h, null(), i0, k, 0) }).note("ispower(n, , &r): the largest k and the root"));
        }
        #[cfg(has_flint)]
        {
            let h = pool.flint();
            cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::PPOW, h, null(), i0, k, 0, 0, 0) }).note("fmpz_is_perfect_power: one exponent and the root, not the largest"));
        }
        if small {
            let w: Vec<u64> = pool.nums.iter().map(lo).collect();
            cs.push(c("malachite", move |i0, k| (0..k).map(|j| w[(i0 + j) % w.len()].express_as_power().map_or(0, |(_, e)| e)).sum()).note("u64::express_as_power"));
        }
        ctx.race(row("perfect_power", name).check(64), cs);
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| pr.get(i0 + j).is_perfect_power() as u64).sum())];
        #[cfg(has_gmp)]
        {
            let h = pool.gmp();
            cs.push(c("GMP", move |i0, k| unsafe { gmp_op(g::PPOW, h, null(), i0, k) }).note("mpz_perfect_power_p"));
        }
        ctx.race(row("is_perfect_power", name).check(64), cs);
    }
}
