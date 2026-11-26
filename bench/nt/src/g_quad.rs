//! Squares modulo a number: jacobi, kronecker, sqrt_mod, sqrt_mod_all, cornacchia (and the
//! Factorization forms).
use crate::core::*;
#[allow(unused_imports)]
use crate::ffi::*;
use a::math::numbers::{ByteOrder, Factorization, Number};
use malachite_base::num::number_theory::traits::{JacobiSymbol as _, KroneckerSymbol as _};
use malachite_nz::natural::Natural;

fn natural(n: &Number) -> Natural {
    Natural::from_limbs_asc(&limbs(n))
}

pub fn kronecker(ctx: &mut Ctx) {
    for &bits in &[64u32, 128, 192, 256, 384, 512, 640, 768, 1024, 2048, 4096, 16_384] {
        let n = match bits {
            0..=1024 => 16_384,
            1025..=4096 => 2048,
            _ => 256,
        };
        let mut rng = Rng::new(20_000 + bits as u64);
        let a: Vec<Number> = (0..n).map(|_| rng.bits(bits, false)).collect();
        let b: Vec<Number> = (0..n).map(|_| rng.bits(bits, true)).collect();
        let ma: Vec<Natural> = a.iter().map(natural).collect();
        let mb: Vec<Natural> = b.iter().map(natural).collect();
        let (pa, pb) = (Pool::new(a), Pool::new(b));
        let (ra, rb) = (&pa, &pb);
        let params = format!("{bits} bits, fresh (a pool of {n})");
        for op in ["kronecker", "jacobi"] {
            if op == "jacobi" && ![64, 256, 1024, 4096].contains(&bits) {
                continue;
            }
            let jac = op == "jacobi";
            let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| (if jac { ra.get(i0 + j).jacobi(rb.get(i0 + j)) } else { ra.get(i0 + j).kronecker(rb.get(i0 + j)) } + 1) as u64).sum())];
            #[cfg(has_gmp)]
            {
                let (x, y) = (pa.gmp(), pb.gmp());
                cs.push(c("GMP", move |i0, k| unsafe { gmp_op(if jac { g::JACOBI } else { g::KRON }, x, y, i0, k) }).note(if jac { "mpz_jacobi" } else { "mpz_kronecker" }));
            }
            #[cfg(has_flint)]
            {
                let (x, y) = (pa.flint(), pb.flint());
                cs.push(c("FLINT", move |i0, k| unsafe { flint_op(if jac { f::JACOBI } else { f::KRON }, x, y, i0, k, 0, 0, 0) }).note(if jac { "fmpz_jacobi" } else { "fmpz_kronecker" }));
            }
            #[cfg(has_pari)]
            if !jac {
                let (x, y) = (pa.pari(), pb.pari());
                cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::KRON, x, y, i0, k, 0) }).note("kronecker"));
            }
            let (xa, xb) = (&ma, &mb);
            cs.push(c("malachite", move |i0, k| {
                (0..k)
                    .map(|j| {
                        let i = (i0 + j) % xa.len();
                        (if jac { (&xa[i]).jacobi_symbol(&xb[i]) } else { (&xa[i]).kronecker_symbol(&xb[i]) } as i64 + 1) as u64
                    })
                    .sum()
            }));
            ctx.race(row(op, &params).check(512), cs);
        }
    }
}

fn special(name: &str) -> Number {
    let two = num(2);
    match name {
        "2^255 - 19" => &two.pow(255) - &num(19),
        "2^521 - 1" => &two.pow(521) - &num(1),
        "P-224" => &(&two.pow(224) - &two.pow(96)) + &num(1),
        _ => unreachable!(),
    }
}

pub fn sqrt(ctx: &mut Ctx) {
    // Square roots modulo primes.
    let kinds = ["32 bits", "64 bits", "128 bits", "256 bits", "1024 bits", "2^255 - 19", "2^521 - 1", "P-224"];
    for name in kinds {
        let mut rng = Rng::new(21_000 + name.len() as u64 * 13 + name.as_bytes()[0] as u64);
        let primes: Vec<Number> = if name.ends_with("bits") {
            let bits: u32 = name.split(' ').next().unwrap().parse().unwrap();
            (0..if bits >= 1024 { 4 } else { 16 }).map(|_| rng.prime(bits)).collect()
        } else {
            vec![special(name)]
        };
        let n = 1024;
        let mut av = vec![];
        let mut pv = vec![];
        for i in 0..n {
            let p = &primes[i % primes.len()];
            let x = rng.below_num(p);
            av.push(&(&x * &x) % p);
            pv.push(p.clone());
        }
        let facs: Vec<Factorization> = pv.iter().map(|p| Factorization::new(vec![(p.clone(), 1)], ByteOrder::LittleEndian)).collect();
        let (pa, pp) = (Pool::new(av), Pool::new(pv));
        let (ra, rp, rf) = (&pa, &pp, &facs);
        let params = format!("prime {name}");
        let refs = |cs: &mut Vec<C<'_>>| {
            #[cfg(has_flint)]
            {
                let (x, y) = (pa.flint(), pp.flint());
                cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::SQRTMOD, x, y, i0, k, 0, 0, 0) }).note("fmpz_sqrtmod: takes p as prime"));
                if name == "32 bits" {
                    let mut w = vec![];
                    for i in 0..n {
                        w.push(lo(pa.get(i)));
                        w.push(lo(pp.get(i)));
                    }
                    cs.push(c("FLINT n_sqrtmod", move |i0, k| unsafe { flint_nop(f::N_SQRTMOD, w.as_ptr(), w.len(), i0, k) }).note("takes p as prime"));
                }
            }
            #[cfg(has_pari)]
            {
                let (x, y) = (pa.pari(), pp.pari());
                cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::FPSQRT, x, y, i0, k, 0) }).note("Fp_sqrt: takes p as prime"));
            }
        };
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&rf[(i0 + j) % n].sqrt_mod(ra.get(i0 + j)).unwrap())).fold(0, u64::wrapping_add))];
        refs(&mut cs);
        ctx.race(row("Factorization::sqrt_mod", &params).check(64), cs);
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&ra.get(i0 + j).sqrt_mod(rp.get(i0 + j)).unwrap())).fold(0, u64::wrapping_add))];
        refs(&mut cs);
        ctx.race(row("sqrt_mod", &params).check(64).notes("ours tests p (Baillie-PSW) and factors it; the references take p as prime"), cs);
    }

    // All the roots modulo words with a few prime factors.
    let kinds: Vec<(&str, Vec<u32>, u32)> = vec![("2 primes of 32 bits", vec![32, 32], 0), ("4 primes of 16 bits", vec![16, 16, 16, 16], 0), ("2^20 times a prime of 40 bits", vec![40], 20)];
    for (name, pb, two) in &kinds {
        let mut rng = Rng::new(22_000 + name.len() as u64);
        let nm = 64;
        let mut mods = vec![];
        let mut fms = vec![];
        while mods.len() < nm {
            let ps: Vec<Number> = pb.iter().map(|&b| rng.prime(b)).collect();
            let mut f: Vec<(Number, u32)> = ps.iter().map(|p| (p.clone(), 1)).collect();
            let mut m = ps.iter().fold(num(1), |a, p| &a * p);
            if *two > 0 {
                m = &m * &num(1 << two);
                f.push((num(2), *two));
            }
            if m.bits() <= 64 {
                mods.push(m);
                fms.push(Factorization::new(f, ByteOrder::LittleEndian));
            }
        }
        let n = 1024;
        let mut av = vec![];
        let mut mv = vec![];
        let mut fv = vec![];
        for i in 0..n {
            let m = &mods[i % nm];
            let x = loop {
                let x = rng.below_num(m);
                if x.gcd(m) == num(1) {
                    break x;
                }
            };
            av.push(&(&x * &x) % m);
            mv.push(m.clone());
            fv.push(fms[i % nm].clone());
        }
        let (pa, pm) = (Pool::new(av), Pool::new(mv));
        let (ra, rm, rf) = (&pa, &pm, &fv);
        let sum_all = |v: Vec<Number>| (v.len() as u64).wrapping_add(v.iter().map(lo).fold(0, u64::wrapping_add));
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| sum_all(ra.get(i0 + j).sqrt_mod_all(rm.get(i0 + j)))).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            let mut w = vec![];
            for i in 0..n {
                w.push(lo(pa.get(i)));
                w.push(lo(pm.get(i)));
            }
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_SQRTMODN, w.as_ptr(), w.len(), i0, k) }).note("n_factor then n_sqrtmodn: all the roots"));
        }
        ctx.race(row("sqrt_mod_all (factors m)", name).check(64), cs);
        let cs = vec![c("a", move |i0, k| (0..k).map(|j| sum_all(rf[(i0 + j) % n].sqrt_mod_all(ra.get(i0 + j)))).fold(0, u64::wrapping_add))];
        ctx.race(row("Factorization::sqrt_mod_all", name).check(0), cs);
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| ra.get(i0 + j).sqrt_mod(rm.get(i0 + j)).is_some() as u64).sum())];
        #[cfg(has_pari)]
        {
            let (x, y) = (pa.pari(), pm.pari());
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::ZNSQRT, x, y, i0, k, 0) }).note("Zn_sqrt(a, m): one root, not the smallest; factors m"));
        }
        ctx.race(row("sqrt_mod (factors m)", name).check(64), cs);
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| rf[(i0 + j) % n].sqrt_mod(ra.get(i0 + j)).is_some() as u64).sum())];
        #[cfg(has_pari)]
        {
            let mut exps = vec![];
            let mut goff = vec![0u64];
            let mut primes = vec![];
            for f in &fv {
                for (p, e) in f.factors() {
                    primes.push(p.clone());
                    exps.push(*e);
                }
                goff.push(primes.len() as u64);
            }
            let pp = Pool::new(primes);
            let q = unsafe { pari_facpool(pp.pari(), exps.as_ptr(), goff.as_ptr(), n) };
            let y = pa.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_facop(p::F_ZNSQRT, q, y, i0, k, 0) }).note("Zn_sqrt(a, factorisation of m): one root, not the smallest"));
        }
        ctx.race(row("Factorization::sqrt_mod", name).check(64), cs);
    }

    // Cornacchia's algorithm modulo primes the form represents.
    for &(d, bits) in &[(1u64, 64u32), (1, 128), (1, 192), (1, 256), (2, 256), (7, 1024)] {
        let mut rng = Rng::new(23_000 + d * 1000 + bits as u64);
        let dn = num(d);
        let md = Number::new_signed(dn.value.clone(), true, ByteOrder::LittleEndian);
        let np = if bits >= 1024 { 16 } else { 256 };
        let mut ps = vec![];
        while ps.len() < np {
            let p = rng.prime(bits);
            if md.kronecker(&p) == 1 {
                ps.push(p);
            }
        }
        let facs: Vec<Factorization> = ps.iter().map(|p| Factorization::new(vec![(p.clone(), 1)], ByteOrder::LittleEndian)).collect();
        let pp = Pool::new(ps);
        let pd = Pool::new(vec![dn.clone(); np]);
        let (rp, rf, rd) = (&pp, &facs, &dn);
        let first = |v: Vec<(Number, Number)>| v.first().map_or(0, |(x, y)| lo(x).wrapping_add(lo(y)));
        let params = format!("x^2 + {}y^2, prime of {bits} bits", if d == 1 { String::new() } else { d.to_string() });
        let refs = |cs: &mut Vec<C<'_>>| {
            #[cfg(has_pari)]
            {
                let (x, y) = (pd.pari(), pp.pari());
                cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::CORN, x, y, i0, k, 0) }).note("qfbcornacchia(d, p): takes p as prime, one solution"));
            }
        };
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| first(rf[(i0 + j) % np].cornacchia(rd))).fold(0, u64::wrapping_add))];
        refs(&mut cs);
        ctx.race(row("Factorization::cornacchia", &params).check(64).notes("ours all the solutions (two for d = 1)"), cs);
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| first(Number::cornacchia(rd, rp.get(i0 + j)))).fold(0, u64::wrapping_add))];
        refs(&mut cs);
        ctx.race(row("cornacchia", &params).check(64).notes("ours tests and factors m, all the solutions; PARI takes p as prime"), cs);
    }
}
