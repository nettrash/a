// Bernoulli numbers (one and the vector) and harmonic numbers.
use a::math::rational::Rational;
use malachite_q::Rational as MQ;

use crate::data::*;
use crate::ffi::*;
use crate::harness::{Ent, agree, row};

fn ours_seq(kind: u8, n0: u64, step: u64) -> Ent<'static> {
    let mut j = 0u64;
    Ent::new("a", move |c| {
        for _ in 0..c {
            let n = n0 + step * (j % 4);
            let r = if kind == 0 { Rational::bernoulli(n, LE) } else { Rational::harmonic(n, LE) };
            std::hint::black_box(r);
            j += 1;
        }
    })
}

pub fn run(which: &str) {
    let long = which == "seq_long";
    let empty: Vec<Q> = vec![Q { neg: false, num: vec![0], den: vec![1] }];
    let bern: Vec<u64> = if long { vec![1_000_000] } else { vec![10, 40, 60, 100, 102, 300, 1000, 3000, 10_000, 100_000] };
    for n0 in bern {
        let want = Rational::bernoulli(n0, LE).to_string();
        #[cfg(has_flint)]
        let (f0, f1, f3) = (Flint::new(&empty, true), Flint::new(&empty, true), Flint::new(&empty, true));
        #[cfg(has_pari)]
        let pp = Pari::new(&empty);
        let mut ents = vec![ours_seq(0, n0, 2)];
        #[cfg(has_flint)]
        {
            f0.seq(0, n0, 2, 1);
            let nt = agree(std::slice::from_ref(&want), &[f0.str(0, 1, 10)], "bern flint");
            ents.push(Ent::new("FLINT _bernoulli_fmpq_ui", |c| f0.seq(0, n0, 2, c)).note(&nt));
            if !long {
                f1.seq(1, n0, 2, 1);
                let nt = agree(std::slice::from_ref(&want), &[f1.str(0, 1, 10)], "bern flint cached");
                ents.push(Ent::new("FLINT bernoulli_fmpq_ui (its cache)", |c| f1.seq(1, n0, 2, c)).note(&nt));
            }
            if n0 == 100_000 {
                f3.seq(3, n0, 2, 1);
                let nt = agree(std::slice::from_ref(&want), &[f3.str(0, 1, 10)], "bern flint zeta");
                ents.push(Ent::new("FLINT _bernoulli_fmpq_ui_zeta (the formula alone)", |c| f3.seq(3, n0, 2, c)).note(&nt));
            }
        }
        #[cfg(has_pari)]
        if !long {
            pp.seq(0, n0, 2, 1);
            let nt = agree(std::slice::from_ref(&want), &[pp.last(-1)], "bern pari");
            ents.push(Ent::new("PARI bernfrac", |c| pp.seq(0, n0, 2, c)).note("keeps a cache of the numbers it computed").note(&nt));
        }
        let note = if long { "PARI left out at 10^6 (81 s a call on the page)" } else { "" };
        row("seq", "Rational::bernoulli", &format!("n = {n0}, {}, {}, {} in turn", n0 + 2, n0 + 4, n0 + 6), ents, 1.0, note);
    }
    if long {
        return;
    }
    for n0 in [10u64, 60, 89, 100, 125, 300, 1000, 10_000, 100_000, 1_000_000] {
        let want = Rational::harmonic(n0, LE).to_string();
        #[cfg(has_flint)]
        let f = Flint::new(&empty, true);
        #[cfg(has_pari)]
        let pp = Pari::new(&empty);
        let mut ents = vec![ours_seq(1, n0, 1)];
        #[cfg(has_flint)]
        {
            f.seq(2, n0, 1, 1);
            let nt = agree(std::slice::from_ref(&want), &[f.str(0, 1, 10)], "harm flint");
            ents.push(Ent::new("FLINT fmpq_harmonic_ui", |c| f.seq(2, n0, 1, c)).note(&nt));
        }
        #[cfg(has_pari)]
        {
            pp.seq(1, n0, 1, 1);
            let nt = agree(std::slice::from_ref(&want), &[pp.last(-1)], "harm pari");
            ents.push(Ent::new("PARI harmonic", |c| pp.seq(1, n0, 1, c)).note(&nt));
        }
        let nt = agree(std::slice::from_ref(&want), &[MQ::harmonic_number(n0).to_string()], "harm mal");
        let mut j = 0u64;
        ents.push(Ent::new("malachite-q harmonic_number", move |c| {
            for _ in 0..c {
                std::hint::black_box(MQ::harmonic_number(n0 + (j % 4)));
                j += 1;
            }
        }).note(&nt));
        row("seq", "Rational::harmonic", &format!("n = {n0} to {} in turn", n0 + 3), ents, 1.0, "");
    }
    for n in [10u64, 30, 100, 150, 300, 1000, 10_000] {
        let v = Rational::bernoulli_numbers(n, LE);
        let idx: Vec<usize> = vec![0, 1, 2, n as usize];
        let want: Vec<String> = idx.iter().map(|&i| v[i].to_string()).collect();
        #[cfg(has_flint)]
        let slots: Vec<Q> = (0..=n).map(|_| empty[0].clone()).collect();
        #[cfg(has_flint)]
        let (f0, f1) = (Flint::new(&slots, true), Flint::new(&slots, true));
        #[cfg(has_pari)]
        let pp = Pari::new(&empty);
        let mut ents = vec![Ent::new("a", move |c| {
            for _ in 0..c {
                std::hint::black_box(Rational::bernoulli_numbers(n, LE));
            }
        })];
        #[cfg(has_flint)]
        {
            f0.bvec(0, n, 1);
            let theirs: Vec<String> = idx.iter().map(|&i| f0.str(i, 0, 10)).collect();
            let nt = agree(&want, &theirs, "bvec flint");
            ents.push(Ent::new("FLINT bernoulli_fmpq_vec_no_cache", |c| f0.bvec(0, n, c)).note(&nt));
            if n <= 1000 {
                f1.bvec(1, n, 1);
                let theirs: Vec<String> = idx.iter().map(|&i| f1.str(i, 0, 10)).collect();
                let nt = agree(&want, &theirs, "bvec flint arith");
                ents.push(Ent::new("FLINT arith_bernoulli_number_vec", |c| f1.bvec(1, n, c)).note(&nt));
            }
        }
        #[cfg(has_pari)]
        {
            let m = (n / 2) as i64;
            pp.bvec(m, 1);
            let nt = agree(&want[3..], &[pp.last(m + 1)], "bvec pari");
            ents.push(Ent::new("PARI bernvec(n/2)", move |c| pp.bvec(m, c)).note("the even-index numbers only, copied from its cache").note(&nt));
        }
        row("seq", "Rational::bernoulli_numbers", &format!("B_0 .. B_{n}"), ents, 1.0, "");
    }
}
