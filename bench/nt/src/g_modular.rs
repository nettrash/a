//! Modular functions: crt, multiplicative_order, primitive_root, discrete_log, dlog::{bsgs, rho,
//! pohlig_hellman}.
use crate::core::*;
#[allow(unused_imports)]
use crate::ffi::*;
use a::math::numbers::{ByteOrder, Factorization, Number, dlog};
use malachite_base::num::factorization::traits::PrimitiveRootPrime as _;

/// A prime `c·q + 1` of `bits` bits with `q` prime.
pub fn cq1_prime(rng: &mut Rng, bits: u32, c: u64) -> (Number, Number) {
    let cb = 64 - c.leading_zeros();
    loop {
        let q = rng.prime(bits - cb + 1);
        let p = &(&q * &num(c)) + &num(1);
        if p.bits() == bits as u64 && p.is_probable_prime() {
            return (p, q);
        }
    }
}

/// A prime `p` of about `bits` bits whose `p - 1` is 2 times primes of `qb` bits, and the
/// factorisation of `p - 1`.
fn smooth_prime(rng: &mut Rng, bits: u32, qb: u32) -> (Number, Factorization) {
    loop {
        let mut qs = vec![];
        let mut m = num(2);
        while (m.bits() as u32) + qb <= bits {
            let q = rng.prime(qb);
            m = &m * &q;
            qs.push(q);
        }
        let p = &m + &num(1);
        if p.is_probable_prime() {
            let mut f: Vec<(Number, u32)> = vec![(num(2), 1)];
            f.extend(qs.into_iter().map(|q| (q, 1)));
            return (p, Factorization::new(f, ByteOrder::LittleEndian));
        }
    }
}

pub fn crt(ctx: &mut Ctx) {
    // Two congruences modulo primes of `bits` bits.
    for &bits in &[64u32, 128, 256, 1024, 4096] {
        let mut rng = Rng::new(10_000 + bits as u64);
        let np = match bits { 4096 => 4, 1024 => 32, _ => 256 };
        let mut rs = vec![];
        let mut ms = vec![];
        for _ in 0..np {
            for _ in 0..2 {
                let m = rng.prime(bits);
                rs.push(rng.below_num(&m));
                ms.push(m);
            }
        }
        let sys: Vec<Vec<(Number, Number)>> = (0..np).map(|i| vec![(rs[2 * i].clone(), ms[2 * i].clone()), (rs[2 * i + 1].clone(), ms[2 * i + 1].clone())]).collect();
        let pr = Pool::new(rs);
        let pm = Pool::new(ms);
        let sy = &sys;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::crt(&sy[(i0 + j) % sy.len()]).unwrap().0)).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            let (a, b) = (pr.flint(), pm.flint());
            cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::CRT2, a, b, i0, k, 0, 0, 0) }).note("fmpz_CRT"));
        }
        #[cfg(has_pari)]
        {
            let (a, b) = (pr.pari(), pm.pari());
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::CHINESE, a, b, i0, k, 0) }).note("chinese of two intmods"));
        }
        ctx.race(row("crt", &format!("2 primes of {bits} bits")).check(32), cs);
    }
    // Many congruences.
    for &(cnt, bits) in &[(3usize, 64u32), (8, 64), (32, 64), (100, 64), (150, 64), (200, 64), (500, 64), (1000, 64), (10_000, 64), (8, 1024), (100, 1024)] {
        let groups = if bits >= 1024 { (4096 / (cnt * bits as usize / 64)).clamp(2, 16) } else { (65_536 / cnt).clamp(2, 256) };
        let mut rng = Rng::new(11_000 + cnt as u64 * 7 + bits as u64);
        let mut rs = vec![];
        let mut ms = vec![];
        for _ in 0..groups {
            for _ in 0..cnt {
                let m = rng.prime(bits);
                rs.push(rng.below_num(&m));
                ms.push(m);
            }
        }
        let sys: Vec<Vec<(Number, Number)>> = (0..groups).map(|g| (0..cnt).map(|t| (rs[g * cnt + t].clone(), ms[g * cnt + t].clone())).collect()).collect();
        let pr = Pool::new(rs);
        let pm = Pool::new(ms);
        let sy = &sys;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::crt(&sy[(i0 + j) % sy.len()]).unwrap().0)).fold(0, u64::wrapping_add))];
        #[cfg(has_flint)]
        {
            let (a, b) = (pr.flint(), pm.flint());
            cs.push(c("FLINT", move |i0, k| unsafe { flint_op(f::MCRT, a, b, i0, k, cnt as u64, 0, 0) }).note("fmpz_multi_CRT"));
        }
        #[cfg(has_pari)]
        {
            let (a, b) = (pr.pari(), pm.pari());
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::ZVCHINESE, a, b, i0, k, cnt as u64) }).note("ZV_chinese"));
        }
        ctx.race(row("crt", &format!("{cnt} primes of {bits} bits")).check(groups.min(8)), cs);
    }
}

pub fn order_root(ctx: &mut Ctx) {
    // multiplicative_order: (a, m) pairs.
    let kinds: Vec<(&str, u32, u64)> = vec![("primes of 32 bits", 32, 0), ("primes of 64 bits", 64, 0), ("random words", 64, 1), ("primes 2q + 1 of 128 bits", 128, 2), ("primes 30q + 1 of 256 bits", 256, 30), ("primes 2q + 1 of 1024 bits", 1024, 2)];
    for &(name, bits, kind) in &kinds {
        let mut rng = Rng::new(12_000 + bits as u64 + kind);
        let nm = match bits {
            0..=64 => 256,
            65..=256 => 16,
            _ => 4,
        };
        let mods: Vec<Number> = (0..nm)
            .map(|_| match kind {
                0 => rng.prime(bits),
                1 => rng.bits(64, true),
                c => cq1_prime(&mut rng, bits, c).0,
            })
            .collect();
        let mut av = vec![];
        let mut mv = vec![];
        for i in 0..1024.max(nm) {
            let m = &mods[i % nm];
            loop {
                let x = rng.below_num(m);
                if x.gcd(m) == num(1) && x > num(1) {
                    av.push(x);
                    mv.push(m.clone());
                    break;
                }
            }
        }
        let pa = Pool::new(av);
        let pm = Pool::new(mv);
        let (ra, rm) = (&pa, &pm);
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&ra.get(i0 + j).multiplicative_order(rm.get(i0 + j)).unwrap())).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            let (a, b) = (pa.pari(), pm.pari());
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::ZNORDER, a, b, i0, k, 0) }).note("znorder(Mod(a, m)), factoring the group order"));
        }
        ctx.race(row("multiplicative_order", name).check(32).same_k(), cs);
    }
    // primitive_root.
    let kinds: Vec<(&str, u32, u64)> = vec![("prime of 32 bits", 32, 0), ("prime of 48 bits", 48, 0), ("prime of 64 bits", 64, 0), ("2q + 1 of 128 bits", 128, 2), ("30q + 1 of 256 bits", 256, 30), ("2q + 1 of 1024 bits", 1024, 2)];
    for &(name, bits, kind) in &kinds {
        let mut rng = Rng::new(13_000 + bits as u64 + kind);
        let nm = match bits {
            0..=64 => 1024,
            65..=256 => 16,
            _ => 4,
        };
        let mods: Vec<Number> = (0..nm).map(|_| if kind == 0 { rng.prime(bits) } else { cq1_prime(&mut rng, bits, kind).0 }).collect();
        let pm = Pool::new(mods);
        let rm = &pm;
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&Number::primitive_root(rm.get(i0 + j)).unwrap())).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            let h = pm.pari();
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::PRIMROOT, h, null(), i0, k, 0) }).note("znprimroot (tests p too)"));
        }
        if bits <= 64 {
            let w: Vec<u64> = pm.nums.iter().map(lo).collect();
            #[cfg(has_flint)]
            {
                let w = w.clone();
                cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_PRIMROOT, w.as_ptr(), w.len(), i0, k) }).note("n_primitive_root_prime: takes p as prime (no test)"));
            }
            cs.push(c("malachite", move |i0, k| (0..k).map(|j| w[(i0 + j) % w.len()].primitive_root_prime()).fold(0, u64::wrapping_add)).note("u64::primitive_root_prime: takes p as prime (no test)"));
        }
        ctx.race(row("primitive_root", name).check(16).same_k(), cs);
    }
}

/// Targets of discrete logarithms: (h, g, p) with g the smallest primitive root of p.
struct Logs {
    h: Vec<Number>,
    g: Vec<Number>,
    m: Vec<Number>,
    fac: Vec<Factorization>,
}

fn logs_for(rng: &mut Rng, mods: &[(Number, Option<Factorization>)], targets: usize) -> Logs {
    let mut l = Logs { h: vec![], g: vec![], m: vec![], fac: vec![] };
    let roots: Vec<Number> = mods.iter().map(|(m, _)| Number::primitive_root(m).unwrap()).collect();
    for i in 0..targets {
        let t = i % mods.len();
        let m = &mods[t].0;
        let h = &rng.below_num(&(m - &num(2))) + &num(1);
        l.h.push(h);
        l.g.push(roots[t].clone());
        l.m.push(m.clone());
        if let Some(f) = &mods[t].1 {
            l.fac.push(f.clone());
        }
    }
    l
}

pub fn discrete_log(ctx: &mut Ctx) {
    let mut rows: Vec<(String, Vec<(Number, Option<Factorization>)>, usize)> = vec![];
    for &bits in &[32u32, 48, 56, 64] {
        if !ctx.runs(false) {
            break;
        }
        let mut rng = Rng::new(14_000 + bits as u64);
        let mods = (0..4).map(|_| (rng.prime(bits), None)).collect();
        rows.push((format!("random primes of {bits} bits"), mods, if bits <= 32 { 256 } else { 32 }));
    }
    for &(bits, qb) in &[(64u32, 20u32), (128, 20), (128, 32), (256, 24), (256, 32), (512, 28)] {
        if !ctx.runs(false) {
            break;
        }
        let mut rng = Rng::new(15_000 + bits as u64 * 100 + qb as u64);
        let mods = (0..4).map(|_| {
            let (p, f) = smooth_prime(&mut rng, bits, qb);
            (p, Some(f))
        });
        rows.push((format!("{bits} bits, p - 1 of {qb}-bit primes"), mods.collect(), 16));
    }
    for &bits in &[40u32, 44, 48, 56, 64, 68, 72, 80, 88, 96, 104, 112] {
        if !ctx.runs(bits >= 104) {
            continue;
        }
        let mut rng = Rng::new(16_000 + bits as u64);
        let mods = (0..2).map(|_| (cq1_prime(&mut rng, bits, 2).0, None)).collect();
        rows.push((format!("safe primes of {bits} bits"), mods, 8));
    }
    for (name, mods, nt) in rows {
        if !ctx.wants("discrete_log", &name) && !ctx.wants("pohlig_hellman", &name) {
            continue;
        }
        let mut rng = Rng::new(17_000 + name.len() as u64);
        let l = logs_for(&mut rng, &mods, nt);
        let (ph, pg, pm) = (Pool::new(l.h.clone()), Pool::new(l.g.clone()), Pool::new(l.m.clone()));
        let lr = &l;
        let n = l.h.len();
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&lr.h[(i0 + j) % n].discrete_log(&lr.g[(i0 + j) % n], &lr.m[(i0 + j) % n]).unwrap())).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        {
            // b = (g, p) pairs.
            let mut gp = vec![];
            for i in 0..n {
                gp.push(l.g[i].clone());
                gp.push(l.m[i].clone());
            }
            let pgp = Pool::new(gp);
            let (a, b) = (ph.pari(), pgp.pari());
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::ZNLOG, a, b, i0, k, 0) }).note("znlog(h, Mod(g, p)), factoring p - 1"));
            std::mem::forget(pgp);
        }
        #[cfg(has_flint)]
        if name == "random primes of 32 bits" {
            let mut w = vec![];
            for i in 0..n {
                w.push(lo(&l.h[i]));
                w.push(lo(&l.g[i]));
                w.push(lo(&l.m[i]));
            }
            cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_DLOG, w.as_ptr(), w.len(), i0, k) }).note("n_discrete_log_bsgs: the whole group by baby and giant steps"));
        }
        let _ = (&pg, &pm);
        ctx.race(row("discrete_log", &name).check(4).same_k(), cs);
        if !l.fac.is_empty() {
            let mut cs = vec![c("a", move |i0, k| {
                (0..k)
                    .map(|j| {
                        let i = (i0 + j) % n;
                        lo(&dlog::pohlig_hellman(&lr.h[i], &lr.g[i], &lr.m[i], &lr.fac[i]).unwrap())
                    })
                    .fold(0, u64::wrapping_add)
            })];
            #[cfg(has_pari)]
            {
                // Fp_log(h, g, ord, p) with ord = [p - 1, its factorisation]: (g, ord, p) threes.
                let mut v = vec![];
                for i in 0..n {
                    v.push(l.g[i].clone());
                    v.push(&l.m[i] - &num(1));
                    v.push(l.m[i].clone());
                }
                let pv = Pool::new(v);
                let (a, b) = (ph.pari(), pv.pari());
                cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::FPLOG, a, b, i0, k, 0) }).note("Fp_log(h, g, p - 1, p): PARI factors p - 1 (smooth, cheap)"));
                std::mem::forget(pv);
            }
            ctx.race(row("pohlig_hellman (p - 1 factored)", &name).check(4).same_k(), cs);
        }
    }
}

pub fn dlog_methods(ctx: &mut Ctx) {
    for &qb in &[16u32, 20, 22, 24, 26, 28, 30, 40] {
        let mut rng = Rng::new(18_000 + qb as u64);
        let groups: Vec<(Number, Number)> = (0..if qb == 40 { 4 } else { 8 }).map(|_| cq1_prime(&mut rng, qb + 1, 2)).collect();
        let g4 = num(4);
        let nt = 64;
        let mut hs = vec![];
        let mut ms = vec![];
        let mut qs = vec![];
        for i in 0..nt {
            let (p, q) = &groups[i % groups.len()];
            let e = rng.below_num(q);
            hs.push(g4.pow_mod(&e, p));
            ms.push(p.clone());
            qs.push(q.clone());
        }
        let facs: Vec<Factorization> = qs.iter().map(|q| Factorization::new(vec![(q.clone(), 1)], ByteOrder::LittleEndian)).collect();
        let (hr, mr, qr, fr, gr) = (&hs, &ms, &qs, &facs, &g4);
        let params = format!("order q of {qb} bits modulo 2q + 1, g = 4");
        type M = fn(&Number, &Number, &Number, &Number) -> Option<Number>;
        let methods: [(&str, M); 2] = [("dlog::bsgs", dlog::bsgs), ("dlog::rho", dlog::rho)];
        #[cfg(has_pari)]
        let mut ref_pools: Vec<(Pool, Pool)> = vec![];
        for (name, mf) in methods {
            let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&mf(&hr[(i0 + j) % nt], gr, &mr[(i0 + j) % nt], &qr[(i0 + j) % nt]).unwrap())).fold(0, u64::wrapping_add))];
            #[cfg(has_pari)]
            {
                let ph = Pool::new(hs.clone());
                let mut v = vec![];
                for i in 0..nt {
                    v.push(g4.clone());
                    v.push(qs[i].clone());
                    v.push(ms[i].clone());
                }
                let pv = Pool::new(v);
                let (a, b) = (ph.pari(), pv.pari());
                cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::FPLOG, a, b, i0, k, 0) }).note("Fp_log(h, g, q, p) with the order given"));
                ref_pools.push((ph, pv));
            }
            #[cfg(has_flint)]
            if qb <= 40 {
                let mut w = vec![];
                for i in 0..nt {
                    w.push(lo(&hs[i]));
                    w.push(4);
                    w.push(lo(&ms[i]));
                }
                cs.push(c("FLINT", move |i0, k| unsafe { flint_nop(f::N_DLOG, w.as_ptr(), w.len(), i0, k) }).note("n_discrete_log_bsgs: searches the order p - 1 = 2q (sqrt(2) more steps)"));
            }
            ctx.race(row(name, &params).check(8).same_k(), cs);
        }
        let mut cs = vec![c("a", move |i0, k| (0..k).map(|j| lo(&dlog::pohlig_hellman(&hr[(i0 + j) % nt], gr, &mr[(i0 + j) % nt], &fr[(i0 + j) % nt]).unwrap())).fold(0, u64::wrapping_add))];
        #[cfg(has_pari)]
        if let Some((ph, pv)) = ref_pools.first() {
            let (a, b) = (ph.pari(), pv.pari());
            cs.push(c("PARI", move |i0, k| unsafe { pari_op(p::FPLOG, a, b, i0, k, 0) }).note("Fp_log(h, g, q, p) with the order given"));
        }
        ctx.race(row("dlog::pohlig_hellman", &params).check(8).same_k(), cs);
    }
}
