// Text both ways: FromStr of fractions and decimals, Display, to_str_radix, from_str_radix.
use std::str::FromStr;

use a::math::rational::Rational;
use malachite_base::num::conversion::traits::{FromSciString, FromStringBase, ToStringBase};
use malachite_q::Rational as MQ;
use num_rational::BigRational;
use num_traits::Num;

use crate::data::*;
use crate::ffi::*;
use crate::g_misc::P;
use crate::harness::{Ent, agree, row};
use crate::libs::*;

const K: usize = 32;

fn digits(rng: &mut Rng, d: usize) -> String {
    let mut s = String::with_capacity(d);
    s.push((b'1' + rng.below(9) as u8) as char);
    for _ in 1..d {
        s.push((b'0' + rng.below(10) as u8) as char);
    }
    s
}

/// Fractions in lowest terms as text: parts of `bits` (words) or of `d` digits.
fn fractions(rng: &mut Rng, bits: Option<u64>, d: usize, n: usize) -> Vec<String> {
    (0..n)
        .map(|_| {
            let r = match bits {
                Some(b) => ours(&rand_q(rng, b, b, false, true)),
                None => {
                    let t = format!("{}{}/{}", if rng.bool() { "-" } else { "" }, digits(rng, d), digits(rng, d));
                    t.parse::<Rational>().unwrap()
                }
            };
            r.to_string()
        })
        .collect()
}

fn pools(strs: &[String], base: u32) -> (Gmp, Vec<Q>) {
    let qs: Vec<Q> = strs.iter().map(|s| of_ours(&Rational::from_str_radix(s, base).unwrap())).collect();
    let g = Gmp::new(&qs, true);
    for (i, s) in strs.iter().enumerate() {
        g.set_s(i, s);
    }
    (g, qs)
}

fn parse_row(op: &str, label: &str, strs: &[String], base: u32, note: &str) {
    let n = strs.len();
    let (g, _qs) = pools(strs, base);
    #[cfg(has_flint)]
    let f = Flint::new(&_qs, true);
    #[cfg(has_flint)]
    for (i, s) in strs.iter().enumerate() {
        f.set_s(i, s);
    }
    let with_num = !(base == 10 && strs[0].len() > 300_000);
    // num-rational reads no integer without "/" in radix 16 (an error): those get "/1".
    let nstrs: Vec<String> = strs.iter().map(|t| if t.contains('/') { t.clone() } else { format!("{t}/1") }).collect();
    let fixed = nstrs.iter().zip(strs).filter(|(a, b)| a != b).count();
    let mut s = if base == 10 {
        Set::new(op, mk1("a", strs, K, |t| t.parse::<Rational>().unwrap()))
    } else {
        Set::new(op, mk1("a", strs, K, move |t| Rational::from_str_radix(t, base).unwrap()))
    };
    s.add(gmp1("GMP mpq_set_str + mpq_canonicalize", &g, 16, base as i64, K, Res::Q));
    if base == 10 {
        s.add_unchecked(gmp1("GMP mpq_set_str (no reduction)", &g, 15, base as i64, K, Res::Q).0, "leaves the fraction as written: less work");
    }
    #[cfg(has_flint)]
    s.add(flint1("FLINT fmpq_set_str + fmpq_canonicalise", &f, 16, base as i64, K, Res::Q));
    if base == 10 {
        s.add(mk1("malachite-q", strs, K, |t| MQ::from_str(t).unwrap()));
    } else {
        s.add(mk1("malachite-q", strs, K, move |t| MQ::from_string_base(base as u8, t).unwrap()));
    }
    if with_num {
        s.add(mk1("num-rational BigRational", &nstrs, K, move |t| BigRational::from_str_radix(t, base).unwrap()));
    }
    let nt_fix = if with_num && fixed > 0 { format!("; num-rational: \"/1\" appended to the {fixed} integers of the pool, which it does not read in radix 16") } else { String::new() };
    #[cfg(feature = "rug")]
    s.add(mk1("rug Rational", strs, K, move |t| rug::Rational::from_str_radix(t, base as i32).unwrap()));
    let mut nt = note.to_string();
    nt.push_str(&nt_fix);
    if !with_num {
        nt.push_str("; num-rational left out (quadratic, seconds)");
    }
    row("text", op, &format!("{label}; pool {n}"), s.ents, 1.0, &nt);
}

fn write_row(op: &str, label: &str, strs: &[String], base: u32, display: bool) {
    let n = strs.len();
    let qs: Vec<Q> = strs.iter().map(|s| of_ours(&Rational::from_str_radix(s, base).unwrap())).collect();
    let p = P::new(&qs);
    let cap = strs.iter().map(|s| s.len()).max().unwrap() + 16;
    p.g.reserve(cap);
    #[cfg(has_flint)]
    p.f.reserve(cap);
    let mut s = if display {
        Set::new(op, mk1("a", &p.o, K, |x| x.to_string()))
    } else {
        Set::new(op, mk1("a", &p.o, K, move |x| x.to_str_radix(base)))
    };
    let k = K.min(n);
    let gch: Vec<String> = (0..k).map(|i| p.g.str(i, 0, base as i32)).collect();
    let gn = agree(s.ours_checks(), &gch, &format!("{op} gmp"));
    let g = &p.g;
    s.add_unchecked(Ent::new("GMP mpq_get_str (into a buffer)", move |c| {
        g.un(14, c, base as i64);
    }), &gn);
    #[cfg(has_flint)]
    {
        let fch: Vec<String> = (0..k).map(|i| p.f.str(i, 0, base as i32)).collect();
        let fnote = agree(s.ours_checks(), &fch, &format!("{op} flint"));
        let f = &p.f;
        s.add_unchecked(Ent::new("FLINT fmpq_get_str (into a buffer)", move |c| {
            f.un(14, c, base as i64);
        }), &fnote);
    }
    if base == 10 {
        s.add(mk1("malachite-q", &p.m, K, |x| x.to_string()));
    } else {
        s.add(mk1("malachite-q", &p.m, K, move |x| x.to_string_base(base as u8)));
    }
    let big = strs[0].len() > 300_000;
    match base {
        10 if !big => s.add(mk1("num-rational BigRational", &p.n, K, |x| x.to_string())),
        16 => s.add(mk1("num-rational BigRational", &p.n, K, |x| format!("{x:x}"))),
        2 => s.add(mk1("num-rational BigRational", &p.n, K, |x| format!("{x:b}"))),
        _ => {}
    }
    #[cfg(feature = "rug")]
    {
        if base == 10 {
            s.add(mk1("rug Rational", &p.r, K, |x| x.to_string()));
        } else {
            s.add(mk1("rug Rational", &p.r, K, move |x| x.to_string_radix(base as i32)));
        }
    }
    let note = if big && base == 10 { "num-rational left out (quadratic, seconds)" } else { "" };
    row("text", op, &format!("{label}; pool {n}"), s.ents, 1.0, note);
}

pub fn run() {
    let mut rng = Rng::new(0x5EED_0005);
    let fsizes: [(&str, Option<u64>, usize); 9] = [
        ("parts of 16 bits", Some(16), 0),
        ("parts of 32 bits", Some(32), 0),
        ("parts of 64 bits", Some(64), 0),
        ("10 digits a part", None, 10),
        ("100 digits a part", None, 100),
        ("1000 digits a part", None, 1000),
        ("10^4 digits a part", None, 10_000),
        ("10^5 digits a part", None, 100_000),
        ("10^6 digits a part", None, 1_000_000),
    ];
    for (label, bits, d) in fsizes {
        let n = if d >= 1_000_000 { 2 } else if d >= 100_000 { 4 } else if d >= 10_000 { 16 } else { 4096 };
        let strs = fractions(&mut rng, bits, d, n);
        parse_row("FromStr (fraction)", label, &strs, 10, "text in lowest terms; random signs");
        write_row("to_string (fraction)", label, &strs, 10, true);
        write_row("to_str_radix(10)", label, &strs, 10, false);
    }
    // Decimals I.F (and with an exponent).
    for (label, d, e) in [("4 digits", 4usize, false), ("10 digits", 10, false), ("19 digits", 19, false), ("10 digits, exponent -9..9", 10, true), ("100 digits", 100, false), ("1000 digits", 1000, false), ("10^4 digits", 10_000, false), ("10^5 digits", 100_000, false), ("10^6 digits", 1_000_000, false)] {
        let n = if d >= 100_000 { 4 } else if d >= 10_000 { 16 } else { 4096 };
        let strs: Vec<String> = (0..n)
            .map(|_| {
                let all = digits(&mut rng, d);
                let ip = (d / 2).max(1);
                let mut t = format!("{}{}.{}", if rng.bool() { "-" } else { "" }, &all[..ip], &all[ip..]);
                if e {
                    t.push_str(&format!("e{}", rng.below(19) as i64 - 9));
                }
                t
            })
            .collect();
        let mut s = Set::new("FromStr decimal", mk1("a", &strs, K, |t| t.parse::<Rational>().unwrap()));
        s.add(mk1("malachite-q from_sci_string", &strs, K, |t| MQ::from_sci_string(t).unwrap()));
        row("text", "FromStr (decimal)", &format!("I.F, {label} in all, random signs; pool {n}"), s.ents, 1.0, "GMP, FLINT, num-rational and rug read no decimals");
    }
    // Radices.
    for base in [2u32, 7, 16] {
        for (label, bits) in [("16b", 16u64), ("64b", 64), ("128b", 128), ("4096b", 4096)] {
            let n = if bits >= 4096 { 256 } else { 4096 };
            let strs: Vec<String> = (0..n).map(|_| ours(&rand_q(&mut rng, bits, bits, bits > 64, true)).to_str_radix(base)).collect();
            write_row(&format!("to_str_radix({base})"), &format!("parts of {label}"), &strs, base, false);
            parse_row(&format!("from_str_radix(_, {base})"), &format!("parts of {label}, in lowest terms"), &strs, base, "random signs");
        }
    }
}
