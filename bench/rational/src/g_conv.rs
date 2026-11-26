// Conversions: to and from floats, to integers, from primitive integers.
use a::math::numbers::Number;
use a::math::rational::Rational;
use malachite_base::num::conversion::traits::RoundingFrom;
use malachite_base::rounding_modes::RoundingMode;
use malachite_nz::integer::Integer as MInt;
use malachite_q::Rational as MQ;
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::data::*;
use crate::ffi::*;
use crate::g_misc::P;
use crate::harness::row;
use crate::libs::*;

const K: usize = 64;

/// Parts of the given bits, down to 3 fewer (as the page's sizes).
fn part(rng: &mut Rng, bits: u64) -> Vec<u8> {
    let b = if bits > 4 { bits - rng.below(4) } else { bits };
    rng.exact(b)
}

fn qpool(rng: &mut Rng, nb: u64, db: u64, n: usize) -> Vec<Q> {
    (0..n)
        .map(|_| {
            let num = part(rng, nb);
            let den = part(rng, db);
            reduce(rng.bool(), num, den)
        })
        .collect()
}

pub fn run() {
    let mut rng = Rng::new(0x5EED_0004);
    // to_f64
    let sizes: [(&str, u64, u64); 10] = [
        ("16b", 16, 16),
        ("32b", 32, 32),
        ("53b", 53, 53),
        ("64b", 64, 64),
        ("128b", 128, 128),
        ("512b", 512, 512),
        ("4096b", 4096, 4096),
        ("65536b", 65536, 65536),
        ("65536/64b", 65536, 64),
        ("64/65536b", 64, 65536),
    ];
    for (label, nb, db) in sizes {
        let n = pool_for(nb.max(db) / 64, 16).min(4096);
        let qs = qpool(&mut rng, nb, db, n);
        let p = P::new(&qs);
        let mut s = Set::new("to_f64", mk1("a", &p.o, K, |x| x.to_f64()));
        s.add_note(gmp1("GMP mpq_get_d", &p.g, 8, 0, K, Res::F64), "truncates toward zero");
        #[cfg(has_flint)]
        s.add_note(flint1("FLINT fmpq_get_d", &p.f, 8, 0, K, Res::F64), "truncates toward zero");
        s.add(mk1("malachite-q", &p.m, K, |x| f64::rounding_from(x, RoundingMode::Nearest).0));
        s.add(mk1("num-rational BigRational", &p.n, K, |x| x.to_f64().unwrap()));
        #[cfg(feature = "rug")]
        s.add_note(mk1("rug Rational", &p.r, K, |x| x.to_f64()), "truncates toward zero");
        row("conv", "to_f64", &format!("parts {label} (numerator/denominator), random signs; pool {n}"), s.ents, 1.0, "ours and malachite and num: nearest, ties to even");
    }
    // to_f32
    for (label, nb, db) in [("16b", 16u64, 16u64), ("64b", 64, 64), ("4096b", 4096, 4096)] {
        let n = pool_for(nb / 64, 16).min(4096);
        let qs = qpool(&mut rng, nb, db, n);
        let p = P::new(&qs);
        let mut s = Set::new("to_f32", mk1("a", &p.o, K, |x| x.to_f32()));
        s.add(mk1("malachite-q", &p.m, K, |x| f32::rounding_from(x, RoundingMode::Nearest).0));
        #[cfg(feature = "rug")]
        s.add_note(mk1("rug Rational", &p.r, K, |x| x.to_f32()), "rounds differently (toward zero)");
        row("conv", "to_f32", &format!("parts {label}, random signs; pool {n}"), s.ents, 1.0, "num-rational left out: its to_f32 rounds its to_f64 again");
    }
    // try_from(f64)
    let mut floats: Vec<(&str, Vec<f64>)> = Vec::new();
    let n = 4096;
    floats.push(("word: in [2^-11, 2^12), parts in words", (0..n).map(|_| {
        let e = rng.below(23) as i32 - 11;
        let m = 1.0 + (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let x = m * 2f64.powi(e);
        if rng.bool() { -x } else { x }
    }).collect()));
    floats.push(("mid: in [1e-10, 1e10]", (0..n).map(|_| {
        let l = -10.0 + 20.0 * (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let x = 10f64.powf(l);
        if rng.bool() { -x } else { x }
    }).collect()));
    floats.push(("any: random bits, finite", (0..n).map(|_| loop {
        let x = f64::from_bits(rng.next());
        if x.is_finite() {
            break x;
        }
    }).collect()));
    for (label, xs) in &floats {
        let qs: Vec<Q> = xs.iter().map(|_| Q { neg: false, num: vec![0], den: vec![1] }).collect();
        let g = Gmp::new(&qs, true);
        for (i, &x) in xs.iter().enumerate() {
            g.set_d(i, x);
        }
        let mut s = Set::new("try_from f64", mk1("a", xs, K, |&x| Rational::try_from(x).unwrap()));
        s.add(gmp1("GMP mpq_set_d (into an mpq_t that exists)", &g, 12, 0, K, Res::Q));
        s.add_unchecked(gmp1("GMP mpq_set_d (a new mpq_t each time)", &g, 13, 0, K, Res::Q).0, "");
        s.add(mk1("malachite-q", xs, K, |&x| MQ::try_from(x).unwrap()));
        s.add(mk1("num-rational BigRational", xs, K, |&x| BigRational::from_float(x).unwrap()));
        #[cfg(feature = "rug")]
        s.add(mk1("rug Rational", xs, K, |&x| rug::Rational::from_f64(x).unwrap()));
        row("conv", "Rational::try_from(f64)", &format!("{label}; pool {n}"), s.ents, 1.0, "exact on every side");
    }
    {
        let xs: Vec<f32> = (0..n).map(|_| loop {
            let x = f32::from_bits(rng.next() as u32);
            if x.is_finite() {
                break x;
            }
        }).collect();
        let mut s = Set::new("try_from f32", mk1("a", &xs, K, |&x| Rational::try_from(x).unwrap()));
        s.add(mk1("malachite-q", &xs, K, |&x| MQ::try_from(x).unwrap()));
        s.add(mk1("num-rational BigRational", &xs, K, |&x| BigRational::from_float(x).unwrap()));
        #[cfg(feature = "rug")]
        s.add(mk1("rug Rational", &xs, K, |&x| rug::Rational::from_f32(x).unwrap()));
        row("conv", "Rational::try_from(f32)", &format!("any finite f32; pool {n}"), s.ents, 1.0, "exact on every side");
    }
    // Number::try_from(&r) and i128::try_from(&r) of integers.
    for (label, bits) in [("16b", 16u64), ("64b", 64), ("127b", 127), ("4096b", 4096)] {
        let n = 4096;
        let qs: Vec<Q> = (0..n).map(|_| { let b = bits - rng.below(3); Q { neg: rng.bool(), num: rng.exact(b), den: vec![1] } }).collect();
        let p = P::new(&qs);
        let mut s = Set::new("Number::try_from", mk1("a", &p.o, K, |x| Number::try_from(x).unwrap()));
        s.add(gmp1("GMP mpz_set of the numerator", &p.g, 17, 0, K, Res::Z));
        #[cfg(has_flint)]
        s.add(flint1("FLINT fmpz_set of the numerator", &p.f, 40, 0, K, Res::Z));
        s.add(mk1("malachite-q", &p.m, K, |x| MInt::try_from(x).unwrap()));
        s.add(mk1("num-rational BigRational", &p.n, K, |x| if x.is_integer() { Some(x.to_integer()) } else { None }));
        #[cfg(feature = "rug")]
        s.add(mk1("rug Rational", &p.r, K, |x| if *x.denom() == 1 { Some(x.numer().clone()) } else { None }));
        row("conv", "Number::try_from(&r)", &format!("integers of {label}; pool {n}"), s.ents, 1.0, "GMP/FLINT: a copy into an integer that exists, no check");
        if bits <= 127 {
            let mut s = Set::new("i128::try_from", mk1("a", &p.o, K, |x| i128::try_from(x).unwrap()));
            s.add(mk1("malachite-q", &p.m, K, |x| i128::try_from(x).unwrap()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| if *x.denom() == 1 { x.numer().to_i128() } else { None }.unwrap()));
            row("conv", "i128::try_from(&r)", &format!("integers of {label} (up to 127 bits); pool {n}"), s.ents, 1.0, "");
        }
    }
}
