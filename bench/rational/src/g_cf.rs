// Continued fractions, convergents, best approximations, the simplest rational between two,
// rational reconstruction.
use std::hint::black_box;

use a::math::numbers::Number;
use a::math::rational::Rational;
use malachite_base::num::conversion::traits::ToStringBase;
use malachite_nz::integer::Integer as MInt;
use malachite_nz::natural::Natural;
use malachite_q::Rational as MQ;
use malachite_q::rational::arithmetic::traits::{Approximate, SimplestRationalInInterval};
use malachite_q::rational::conversion::traits::{ContinuedFraction, Convergents};

use crate::cyc;
use crate::data::*;
use crate::ffi::*;
use crate::harness::{Ent, agree, row};

fn q_int(n: &Number) -> Q {
    Q { neg: n.is_negative(), num: le_bytes(n), den: vec![1] }
}

fn nat_of(n: &Number) -> Natural {
    nat(&le_bytes(n))
}

fn hexn(n: &Natural) -> String {
    n.to_string_base(16)
}

/// The sizes: (label, bits of a part, limbs).
fn sizes() -> Vec<(String, u64, u64)> {
    let mut v = vec![("16 bits".to_string(), 16u64, 1u64)];
    for n in [1u64, 2, 4, 16, 64, 256, 1024, 4096, 16384, 100000] {
        v.push((format!("{n} limbs"), 64 * n, n));
    }
    v
}

fn pool(limbs: u64) -> usize {
    if limbs >= 16384 { 1 } else if limbs >= 1024 { 2 } else { pool_for(limbs * 8, 4).min(2048) }
}

fn ent_ours<'a, A, R>(a: &'a [A], f: impl Fn(&'a A) -> R + 'a) -> Ent<'a> {
    let n = a.len();
    let mut j = 0usize;
    Ent::new("a", move |c| cyc!(j, n, c, black_box(f(black_box(&a[j])))))
}

fn quad_ok(limbs: u64) -> bool {
    limbs <= 4096
}

pub fn run(which: &str) {
    let mut rng = Rng::new(0x5EED_0006);
    let quad_note = "PARI and malachite (quadratic) left out past 4096 limbs";
    for (label, bits, limbs) in sizes() {
        let n = pool(limbs);
        let exact = bits > 64;
        let pos = |rng: &mut Rng, b: u64| rand_q(rng, b, b, exact, false);
        let xs: Vec<Q> = (0..n).map(|_| pos(&mut rng, bits)).collect();
        let o: Vec<Rational> = xs.iter().map(ours).collect();
        let k = n.min(16);
        let quad = quad_ok(limbs);
        let m: Vec<MQ> = if quad { xs.iter().map(mal).collect() } else { vec![] };
        #[cfg(has_flint)]
        let f = Flint::new(&xs, true);
        #[cfg(has_pari)]
        let p = if quad { Some(Pari::new(&xs)) } else { None };
        let note_q = if quad { "" } else { quad_note };

        if which == "cf" || which == "all" {
            // continued_fraction
            let cf_ours: Vec<Vec<Number>> = o.iter().take(k).map(|x| x.continued_fraction()).collect();
            let mut ents = vec![ent_ours(&o, |x| x.continued_fraction())];
            #[cfg(has_flint)]
            {
                let mine: Vec<String> = cf_ours.iter().map(|t| t.len().to_string()).collect();
                let theirs: Vec<String> = (0..k).map(|i| f.un_one(30, i, 0).to_string()).collect();
                let nt = agree(&mine, &theirs, "cf flint");
                let fr = &f;
                ents.push(Ent::new("FLINT fmpq_get_cfrac", move |c| {
                    fr.un(30, c, 0);
                }).note("with fmpq_cfrac_bound and the vector's allocation").note(&nt));
            }
            #[cfg(has_pari)]
            if let Some(p) = &p {
                let mine: Vec<String> = cf_ours.iter().map(|t| format!("[{}]", t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))).collect();
                let theirs: Vec<String> = (0..k).map(|i| { p.run_one(0, p, i); p.last(-1) }).collect();
                let nt = agree(&mine, &theirs, "cf pari");
                ents.push(Ent::new("PARI contfrac", move |c| {
                    p.run(0, p, c);
                }).note(&nt));
            }
            if quad {
                let mine: Vec<String> = cf_ours.iter().map(|t| t.iter().map(|x| x.to_str_radix(16)).collect::<Vec<_>>().join(",")).collect();
                let mcf = |x: &MQ| {
                    let (a0, it) = x.continued_fraction();
                    (a0, it.collect::<Vec<Natural>>())
                };
                let theirs: Vec<String> = m.iter().take(k).map(|x| {
                    let (a0, rest) = mcf(x);
                    let mut v = vec![a0.to_string_base(16)];
                    v.extend(rest.iter().map(hexn));
                    v.join(",")
                }).collect();
                let nt = agree(&mine, &theirs, "cf mal");
                let mr = &m;
                let nm = m.len();
                let mut j = 0usize;
                ents.push(Ent::new("malachite-q", move |c| cyc!(j, nm, c, black_box(mcf(black_box(&mr[j]))))).note("collected into a Vec").note(&nt));
            }
            row("cf", "continued_fraction", &format!("parts of {label}, positive; pool {n}"), ents, 1.0, note_q);

            // continued_fraction_into, into one vector again and again
            {
                let mut v: Vec<Number> = Vec::new();
                for x in o.iter().take(k) {
                    x.continued_fraction_into(&mut v);
                    assert_eq!(v, x.continued_fraction(), "continued_fraction_into");
                }
                let on = o.len();
                let or = &o;
                let mut j = 0usize;
                let mut ents = vec![Ent::new("a", move |c| {
                    for _ in 0..c {
                        black_box(&or[j]).continued_fraction_into(&mut v);
                        black_box(&v);
                        j += 1;
                        if j == on {
                            j = 0;
                        }
                    }
                })];
                #[cfg(has_flint)]
                {
                    let fr = &f;
                    ents.push(Ent::new("FLINT fmpq_get_cfrac", move |c| {
                        fr.un(32, c, 0);
                    }).note("into a vector of fmpz that exists"));
                }
                row("cf", "continued_fraction_into", &format!("parts of {label}, positive; pool {n}"), ents, 1.0, "");
            }

            // from_continued_fraction
            let terms: Vec<Vec<Number>> = o.iter().map(|x| x.continued_fraction()).collect();
            let want: Vec<String> = o.iter().take(k).map(|x| x.to_str_radix(16)).collect();
            let mine: Vec<String> = terms.iter().take(k).map(|t| Rational::from_continued_fraction(t).to_str_radix(16)).collect();
            let nt0 = agree(&want, &mine, "from_cf ours");
            let mut ents = vec![ent_ours(&terms, |t| Rational::from_continued_fraction(t)).note(&nt0)];
            #[cfg(has_flint)]
            {
                for i in 0..n {
                    f.prepare_cfrac(i);
                }
                let theirs: Vec<String> = (0..k).map(|i| { f.un_one(31, i, 0); f.str(0, 1, 16) }).collect();
                let nt = agree(&want, &theirs, "from_cf flint");
                let fr = &f;
                ents.push(Ent::new("FLINT fmpq_set_cfrac", move |c| {
                    fr.un(31, c, 0);
                }).note(&nt));
            }
            if quad {
                let mt: Vec<(MInt, Vec<Natural>)> = terms.iter().map(|t| (mint(&le_bytes(&t[0]), t[0].is_negative()), t[1..].iter().map(nat_of).collect())).collect();
                let theirs: Vec<String> = mt.iter().take(k).map(|(a0, r)| MQ::from_continued_fraction_ref(a0, r.iter()).to_string_base(16)).collect();
                let nt = agree(&want, &theirs, "from_cf mal");
                let nm = mt.len();
                let mut j = 0usize;
                ents.push(Ent::new("malachite-q", move |c| cyc!(j, nm, c, black_box(MQ::from_continued_fraction_ref(&mt[j].0, mt[j].1.iter())))).note(&nt));
            }
            row("cf", "from_continued_fraction", &format!("the terms of parts of {label}; pool {n}"), ents, 1.0, if quad { "" } else { "malachite left out past 4096 limbs" });

            // convergents
            if limbs <= 256 {
                let mine: Vec<String> = o.iter().take(k).map(|x| { let c = x.convergents(); format!("{}:{}", c.len(), c.last().unwrap().to_str_radix(16)) }).collect();
                let theirs: Vec<String> = m.iter().take(k).map(|x| { let c: Vec<MQ> = x.clone().convergents().collect(); format!("{}:{}", c.len(), c.last().unwrap().to_string_base(16)) }).collect();
                let nt = agree(&mine, &theirs, "convergents mal");
                let mut ents = vec![ent_ours(&o, |x| x.convergents())];
                let mr = &m;
                let nm = m.len();
                let mut j = 0usize;
                ents.push(Ent::new("malachite-q", move |c| cyc!(j, nm, c, black_box(mr[j].clone().convergents().collect::<Vec<MQ>>()))).note("a clone (it takes the rational by value) and collected").note(&nt));
                row("cf", "convergents", &format!("parts of {label}, positive; pool {n}"), ents, 1.0, "FLINT and PARI have no convergents of a rational");
            }
        }

        if which == "approx" || which == "all" {
            // limit_denominator, bounds of half the length and of 60 bits
            for (blabel, dbits) in [("bound of half the length", (bits / 2).max(8)), ("bound of 60 bits", 60u64)] {
                if dbits >= bits {
                    continue;
                }
                if !quad {
                    continue;
                }
                let ds: Vec<Number> = (0..n).map(|_| number(&rng.exact(dbits), false)).collect();
                let pairs: Vec<(&Rational, &Number)> = o.iter().zip(&ds).collect();
                let mine: Vec<String> = pairs.iter().take(k).map(|(x, d)| x.limit_denominator(d).to_string()).collect();
                let mut ents = vec![ent_ours(&pairs, |&(x, d)| x.limit_denominator(d))];
                #[cfg(has_pari)]
                let pd = Pari::new(&ds.iter().map(q_int).collect::<Vec<_>>());
                // Before 2.19.0 PARI's bestappr(x, D) can return a convergent where a closer
                // semiconvergent exists (2.19.0's CHANGES, Fixed 1): not the same work. And with
                // a bound of half the length its stack grows past 6 GB at 4096 limbs.
                #[cfg(has_pari)]
                let (pv, old_pari) = {
                    let v = pari_version();
                    (format!("{}.{}.{}", v.0, v.1, v.2), v < (2, 19, 0))
                };
                #[cfg(has_pari)]
                let mut row_note = "FLINT has none".to_string();
                #[cfg(not(has_pari))]
                let row_note = "FLINT has none".to_string();
                #[cfg(has_pari)]
                if let Some(p) = &p {
                    if old_pari && limbs > 1024 && dbits > 64 {
                        row_note = format!("{row_note}; no PARI row: PARI {pv}'s bestappr needs more than 6 GB of stack here");
                    } else {
                        let theirs: Vec<String> = (0..k).map(|i| { p.run_one(1, &pd, i); p.last(-1) }).collect();
                        let nt = if old_pari {
                            let bad = mine.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                            let differ = if bad > 0 { format!(" ({bad}/{} results differ)", mine.len()) } else { String::new() };
                            format!("not the same work: PARI {pv}'s bestappr can return a convergent where a closer semiconvergent exists (fixed in 2.19.0){differ}")
                        } else {
                            agree(&mine, &theirs, "limit pari")
                        };
                        ents.push(Ent::new("PARI bestappr(x, D)", move |c| {
                            p.run(1, &pd, c);
                        }).note(&nt));
                    }
                }
                let md: Vec<Natural> = ds.iter().map(nat_of).collect();
                let theirs: Vec<String> = m.iter().zip(&md).take(k).map(|(x, d)| x.approximate(d).to_string()).collect();
                let nt = agree(&mine, &theirs, "limit mal");
                let mr = &m;
                let nm = m.len();
                let mut j = 0usize;
                ents.push(Ent::new("malachite-q approximate", move |c| cyc!(j, nm, c, black_box((&mr[j]).approximate(&md[j])))).note(&nt));
                row("cf", "limit_denominator", &format!("parts of {label}, {blabel}; pool {n}"), ents, 1.0, &row_note);
            }
        }

        if which == "between" || which == "all" {
            // simplest_between: two random ends; x and x + 1/q (q of 1.5 times the length).
            for kind in 0..2 {
                let (lo, hi): (Vec<Q>, Vec<Q>) = if kind == 0 {
                    let ys: Vec<Q> = (0..n).map(|_| pos(&mut rng, bits)).collect();
                    xs.iter().zip(&ys).map(|(a, b)| { let (ra, rb) = (ours(a), ours(b)); if ra <= rb { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) } }).unzip()
                } else {
                    xs.iter().map(|a| {
                        let q = number(&if bits > 64 { rng.exact(bits * 3 / 2) } else { rng.upto(bits * 3 / 2) }, false);
                        let y = &ours(a) + &Rational::new(Number::from(1u32), q);
                        (a.clone(), of_ours(&y))
                    }).unzip()
                };
                let fl_ok = limbs <= 16384;
                if !fl_ok && !quad {
                    continue;
                }
                let ol: Vec<Rational> = lo.iter().map(ours).collect();
                let oh: Vec<Rational> = hi.iter().map(ours).collect();
                let pairs: Vec<(&Rational, &Rational)> = oh.iter().zip(&ol).collect();
                let mine: Vec<String> = pairs.iter().take(k).map(|(b, a)| Rational::simplest_between(a, b).to_str_radix(16)).collect();
                let mut ents = vec![ent_ours(&pairs, |&(b, a)| Rational::simplest_between(a, b)).note("ends given high first")];
                #[cfg(has_flint)]
                let (fl, fh) = (Flint::new(&lo, true), Flint::new(&hi, true));
                #[cfg(has_flint)]
                if fl_ok {
                    let theirs: Vec<String> = (0..k).map(|i| { fl.bin_one(12, &fh, i); fl.str(0, 1, 16) }).collect();
                    let nt = agree(&mine, &theirs, "between flint");
                    ents.push(Ent::new("FLINT fmpq_simplest_between", move |c| {
                        fl.bin(12, &fh, c);
                    }).note(&nt));
                }
                let (ml, mh): (Vec<MQ>, Vec<MQ>) = if quad { (lo.iter().map(mal).collect(), hi.iter().map(mal).collect()) } else { (vec![], vec![]) };
                if quad {
                    let theirs: Vec<String> = (0..k).map(|i| MQ::simplest_rational_in_closed_interval(&ml[i], &mh[i]).to_string_base(16)).collect();
                    let nt = agree(&mine, &theirs, "between mal");
                    let nm = ml.len();
                    let mut j = 0usize;
                    ents.push(Ent::new("malachite-q simplest_rational_in_closed_interval", move |c| cyc!(j, nm, c, black_box(MQ::simplest_rational_in_closed_interval(&ml[j], &mh[j])))).note(&nt));
                }
                let what = if kind == 0 { format!("two random ends of {label}, positive") } else { format!("x of {label} and x + 1/q, q of 1.5 times the length") };
                row("cf", "simplest_between", &format!("{what}; pool {n}"), ents, 1.0, if quad { "" } else { "malachite left out past 4096 limbs" });
            }
        }

        if which == "recon" || which == "all" {
            // reconstruct: a modulus of the size, p/q of up to half of it.
            let mk = |rng: &mut Rng, mbits: u64, pb: u64, qb: u64| -> (Number, Number, Rational) {
                loop {
                    let mm = number(&rng.exact(mbits), false);
                    let p = number(&rng.upto(pb), rng.bool());
                    let q = number(&rng.upto(qb), false);
                    if let Some(qi) = q.mod_inverse(&mm) {
                        let mut r = &(&p * &qi) % &mm;
                        if r.is_negative() {
                            r = &r + &mm;
                        }
                        return (r, mm, Rational::new(p, q));
                    }
                }
            };
            let half = (bits - 1) / 2 - 1;
            let trip: Vec<(Number, Number, Rational)> = (0..n).map(|_| mk(&mut rng, bits, half, half)).collect();
            let pr: Vec<(&Number, &Number)> = trip.iter().map(|t| (&t.0, &t.1)).collect();
            let mine: Vec<String> = pr.iter().take(k).map(|(r, mm)| Rational::reconstruct(r, mm).unwrap().to_string()).collect();
            let want: Vec<String> = trip.iter().take(k).map(|t| t.2.to_string()).collect();
            let nt0 = agree(&want, &mine, "recon ours");
            let mut ents = vec![ent_ours(&pr, |&(r, mm)| Rational::reconstruct(r, mm)).note(&nt0)];
            let rq: Vec<Q> = trip.iter().map(|t| q_int(&t.0)).collect();
            let mq: Vec<Q> = trip.iter().map(|t| q_int(&t.1)).collect();
            #[cfg(has_flint)]
            let (fr, fm) = (Flint::new(&rq, true), Flint::new(&mq, true));
            #[cfg(has_flint)]
            {
                let theirs: Vec<String> = (0..k).map(|i| { fr.bin_one(13, &fm, i); Rational::from_str_radix(&fr.str(0, 1, 16), 16).unwrap().to_string() }).collect();
                let nt = agree(&mine, &theirs, "recon flint");
                ents.push(Ent::new("FLINT fmpq_reconstruct_fmpz", move |c| {
                    fr.bin(13, &fm, c);
                }).note(&nt));
            }
            #[cfg(has_pari)]
            let (pr_, pm_) = if quad { (Some(Pari::new(&rq)), Some(Pari::new(&mq))) } else { (None, None) };
            #[cfg(has_pari)]
            if let (Some(a), Some(b)) = (pr_, pm_) {
                let theirs: Vec<String> = (0..k).map(|i| { a.run_one(2, &b, i); a.last(-1) }).collect();
                let nt = agree(&mine, &theirs, "recon pari");
                ents.push(Ent::new("PARI bestappr(Mod(r, m))", move |c| {
                    a.run(2, &b, c);
                }).note("includes building the Mod").note(&nt));
            }
            let mn: Vec<(Natural, Natural)> = if quad { trip.iter().map(|t| (nat_of(&t.0), nat_of(&t.1))).collect() } else { vec![] };
            if quad {
                let theirs: Vec<String> = mn.iter().take(k).map(|(r, mm)| MQ::reconstruct_ref(r, mm).unwrap().to_string()).collect();
                let nt = agree(&mine, &theirs, "recon mal");
                let nm = mn.len();
                let mut j = 0usize;
                ents.push(Ent::new("malachite-q reconstruct", move |c| cyc!(j, nm, c, black_box(MQ::reconstruct_ref(&mn[j].0, &mn[j].1)))).note(&nt));
            }
            row("cf", "reconstruct", &format!("modulus of {label}, p/q of up to half of it; pool {n}"), ents, 1.0, note_q);

            // reconstruct_with_bounds: N of 0.15 and D of 0.8 of the modulus's bits.
            let nb = ((bits as f64) * 0.15).floor().max(2.0) as u64;
            let db = ((bits as f64) * 0.8).floor() as u64;
            let trip: Vec<(Number, Number, Rational)> = (0..n).map(|_| mk(&mut rng, bits, nb - 1, db - 1)).collect();
            let nn = number(&rng.exact(nb), false);
            let dd = number(&rng.exact(db), false);
            let pr: Vec<(&Number, &Number)> = trip.iter().map(|t| (&t.0, &t.1)).collect();
            let mine: Vec<String> = pr.iter().take(k).map(|(r, mm)| Rational::reconstruct_with_bounds(r, mm, &nn, &dd).unwrap().to_string()).collect();
            let want: Vec<String> = trip.iter().take(k).map(|t| t.2.to_string()).collect();
            let nt0 = agree(&want, &mine, "recon2 ours");
            let (nr, dr) = (&nn, &dd);
            let mut ents = vec![ent_ours(&pr, move |&(r, mm)| Rational::reconstruct_with_bounds(r, mm, nr, dr)).note(&nt0)];
            let rq: Vec<Q> = trip.iter().map(|t| q_int(&t.0)).collect();
            let mq: Vec<Q> = trip.iter().map(|t| q_int(&t.1)).collect();
            let bq: Vec<Q> = trip.iter().map(|_| Q { neg: false, num: le_bytes(&nn), den: le_bytes(&dd) }).collect();
            #[cfg(has_flint)]
            let (fr, fm, fb) = (Flint::new(&rq, true), Flint::new(&mq, true), Flint::new(&bq, false));
            #[cfg(has_flint)]
            {
                let theirs: Vec<String> = (0..k).map(|i| { fr.recon2_one(&fm, &fb, i); Rational::from_str_radix(&fr.str(0, 1, 16), 16).unwrap().to_string() }).collect();
                let nt = agree(&mine, &theirs, "recon2 flint");
                ents.push(Ent::new("FLINT fmpq_reconstruct_fmpz_2", move |c| {
                    fr.recon2(&fm, &fb, c);
                }).note(&nt));
            }
            let mn: Vec<(Natural, Natural)> = if quad { trip.iter().map(|t| (nat_of(&t.0), nat_of(&t.1))).collect() } else { vec![] };
            let (mnn, mdd) = (nat_of(&nn), nat_of(&dd));
            if quad {
                let theirs: Vec<String> = mn.iter().take(k).map(|(r, mm)| MQ::reconstruct_with_bounds_ref(r, mm, &mnn, &mdd).unwrap().to_string()).collect();
                let nt = agree(&mine, &theirs, "recon2 mal");
                let nm = mn.len();
                let mut j = 0usize;
                ents.push(Ent::new("malachite-q reconstruct_with_bounds", move |c| cyc!(j, nm, c, black_box(MQ::reconstruct_with_bounds_ref(&mn[j].0, &mn[j].1, &mnn, &mdd)))).note(&nt));
            }
            row("cf", "reconstruct_with_bounds", &format!("modulus of {label}, N of 0.15 and D of 0.8 of its bits; pool {n}"), ents, 1.0, if quad { "" } else { "malachite left out past 4096 limbs" });
        }
    }
}
