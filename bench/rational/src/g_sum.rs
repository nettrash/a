// Sums and products of many terms: term by term, and by Iterator::sum / product (a tree).
use a::math::rational::Rational;
use malachite_q::Rational as MQ;
use num_rational::BigRational;

use crate::data::*;
use crate::ffi::*;
use crate::harness::{Ent, agree, row};
use crate::libs::Show;

fn harmonic_terms(n: u64) -> Vec<Q> {
    (1..=n).map(|k| Q { neg: false, num: vec![1], den: trim(k.to_le_bytes().to_vec()) }).collect()
}

fn random_terms(rng: &mut Rng, n: usize, bits: u64) -> Vec<Q> {
    (0..n).map(|_| rand_q(rng, bits, bits, false, true)).collect()
}

/// One row each for term by term and for the tree; `mul` for products.
fn sums(label: &str, terms: &[Q], mul: bool, with_num: bool) {
    let n = terms.len() as f64;
    let o: Vec<Rational> = terms.iter().map(ours).collect();
    let m: Vec<MQ> = terms.iter().map(mal).collect();
    let nn: Vec<BigRational> = if with_num { terms.iter().map(num).collect() } else { vec![] };
    #[cfg(feature = "rug")]
    let r: Vec<rug::Rational> = terms.iter().map(rugq).collect();
    let g = Gmp::new(terms, true);
    #[cfg(has_flint)]
    let f = Flint::new(terms, true);

    // Term by term.
    let step_o = |o: &[Rational]| {
        let mut h = Rational::from(if mul { 1 } else { 0 });
        for t in o {
            if mul {
                h *= t;
            } else {
                h += t;
            }
        }
        h
    };
    let step_m = |m: &[MQ]| {
        let mut h = MQ::from(if mul { 1 } else { 0 });
        for t in m {
            if mul {
                h *= t;
            } else {
                h += t;
            }
        }
        h
    };
    let step_n = |v: &[BigRational]| {
        let mut h = BigRational::from_integer((if mul { 1 } else { 0 }).into());
        for t in v {
            if mul {
                h *= t;
            } else {
                h += t;
            }
        }
        h
    };
    #[cfg(feature = "rug")]
    let step_r = |v: &[rug::Rational]| {
        let mut h = rug::Rational::from(if mul { 1 } else { 0 });
        for t in v {
            if mul {
                h *= t;
            } else {
                h += t;
            }
        }
        h
    };
    let want = step_o(&o).show();
    let mut ents = vec![Ent::new("a", |c| {
        for _ in 0..c {
            std::hint::black_box(step_o(&o));
        }
    })];
    g.sum(if mul { 2 } else { 0 }, 1);
    let gn = agree(std::slice::from_ref(&want), &[g.str(0, 1, 16)], &format!("{label} gmp"));
    ents.push(Ent::new("GMP mpq", |c| g.sum(if mul { 2 } else { 0 }, c)).note(&gn));
    #[cfg(has_flint)]
    {
        f.sum(if mul { 2 } else { 0 }, 1);
        let fnote = agree(std::slice::from_ref(&want), &[f.str(0, 1, 16)], &format!("{label} flint"));
        ents.push(Ent::new("FLINT fmpq", |c| f.sum(if mul { 2 } else { 0 }, c)).note(&fnote));
    }
    let mn = agree(std::slice::from_ref(&want), &[step_m(&m).show()], &format!("{label} mal"));
    ents.push(Ent::new("malachite-q", |c| {
        for _ in 0..c {
            std::hint::black_box(step_m(&m));
        }
    }).note(&mn));
    if with_num {
        let nnote = agree(std::slice::from_ref(&want), &[step_n(&nn).show()], &format!("{label} num"));
        ents.push(Ent::new("num-rational BigRational", |c| {
            for _ in 0..c {
                std::hint::black_box(step_n(&nn));
            }
        }).note(&nnote));
    }
    #[cfg(feature = "rug")]
    {
        let rn = agree(std::slice::from_ref(&want), &[step_r(&r).show()], &format!("{label} rug"));
        ents.push(Ent::new("rug Rational", |c| {
            for _ in 0..c {
                std::hint::black_box(step_r(&r));
            }
        }).note(&rn));
    }
    let op = if mul { "a *= t (term by term)" } else { "a += t (term by term)" };
    row("sum", op, &format!("{label}; ns per term"), ents, n, "GMP/FLINT: mpq_add(h, h, t) into a variable that exists");

    // The tree.
    let mut ents = vec![Ent::new("a", |c| {
        for _ in 0..c {
            if mul {
                std::hint::black_box(o.iter().product::<Rational>());
            } else {
                std::hint::black_box(o.iter().sum::<Rational>());
            }
        }
    })];
    g.sum(if mul { 3 } else { 1 }, 1);
    let gn = agree(std::slice::from_ref(&want), &[g.str(0, 1, 16)], &format!("{label} gmp tree"));
    ents.push(Ent::new("GMP mpq (binary-counter tree)", |c| g.sum(if mul { 3 } else { 1 }, c)).note(&gn));
    #[cfg(has_flint)]
    {
        f.sum(if mul { 3 } else { 1 }, 1);
        let fnote = agree(std::slice::from_ref(&want), &[f.str(0, 1, 16)], &format!("{label} flint tree"));
        ents.push(Ent::new("FLINT fmpq (binary-counter tree)", |c| f.sum(if mul { 3 } else { 1 }, c)).note(&fnote));
    }
    let mtree = |m: &[MQ]| if mul { <MQ as std::iter::Product<&MQ>>::product(m.iter()) } else { <MQ as std::iter::Sum<&MQ>>::sum(m.iter()) };
    let mn = agree(std::slice::from_ref(&want), &[mtree(&m).show()], &format!("{label} mal tree"));
    ents.push(Ent::new("malachite-q (its Sum/Product)", |c| {
        for _ in 0..c {
            std::hint::black_box(mtree(&m));
        }
    }).note(&mn));
    #[cfg(feature = "rug")]
    {
        let rtree = |r: &[rug::Rational]| if mul { rug::Rational::from(rug::Rational::product(r.iter())) } else { rug::Rational::from(rug::Rational::sum(r.iter())) };
        let rn = agree(std::slice::from_ref(&want), &[rtree(&r).show()], &format!("{label} rug tree"));
        ents.push(Ent::new("rug Rational (its Sum/Product)", move |c| {
            for _ in 0..c {
                std::hint::black_box(rtree(&r));
            }
        }).note(&rn).note("rug's Sum/Product run term by term"));
    }
    let op = if mul { "Iterator::product" } else { "Iterator::sum" };
    row("sum", op, &format!("{label}; ns per term"), ents, n, "GMP/FLINT: the same binary-counter tree in C");
}

pub fn run() {
    let mut rng = Rng::new(0x5EED_0003);
    for n in [1000u64, 10000, 100000] {
        sums(&format!("H_{n}"), &harmonic_terms(n), false, n <= 1000);
    }
    sums("10000 random terms, parts < 2^16, random signs", &random_terms(&mut rng, 10000, 16), false, false);
    sums("10000 random terms, parts < 2^32, random signs", &random_terms(&mut rng, 10000, 32), false, false);
    sums("product of 10000 random terms, parts < 2^16", &random_terms(&mut rng, 10000, 16), true, false);
}
