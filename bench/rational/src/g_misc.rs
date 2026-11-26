// Construction, rounding, the cheap unary operations, hashing, and arithmetic with
// primitive integers and integers.
use std::hash::{Hash, Hasher};

use a::math::numbers::Number;
use a::math::rational::Rational;
use malachite_base::num::arithmetic::traits::{Abs, Ceiling, Floor, Pow, Reciprocal, Sign};
use malachite_base::num::conversion::traits::RoundingFrom;
use malachite_base::rounding_modes::RoundingMode;
use malachite_nz::integer::Integer as MInt;
use malachite_q::Rational as MQ;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Signed;

use crate::data::*;
use crate::ffi::*;
use crate::g_arith::{Sz, limbs_sz, s16, w1};
use crate::harness::{Ent, agree, row};
use crate::libs::*;

const K: usize = 32;

pub struct P {
    pub o: Vec<Rational>,
    pub g: Gmp,
    #[cfg(has_flint)]
    pub f: Flint,
    pub m: Vec<MQ>,
    pub n: Vec<BigRational>,
    #[cfg(feature = "rug")]
    pub r: Vec<rug::Rational>,
}

impl P {
    pub fn new(qs: &[Q]) -> P {
        P {
            o: qs.iter().map(ours).collect(),
            g: Gmp::new(qs, true),
            #[cfg(has_flint)]
            f: Flint::new(qs, true),
            m: qs.iter().map(mal).collect(),
            n: qs.iter().map(num).collect(),
            #[cfg(feature = "rug")]
            r: qs.iter().map(rugq).collect(),
        }
    }
}

/// A word-at-a-time multiplicative hasher (as FxHash), the same for every library.
#[derive(Default)]
struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(w));
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64)
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64)
    }
}

fn fx<T: Hash>(x: &T) -> u64 {
    let mut h = Fx::default();
    x.hash(&mut h);
    h.finish()
}

fn parts(sz: &Sz) -> String {
    match sz.label.as_str() {
        "s" => "parts < 2^16".into(),
        "1" => "parts < 2^64 (words)".into(),
        l => format!("parts of {l} limbs"),
    }
}

fn sizes() -> Vec<Sz> {
    let mut v = vec![s16(), w1()];
    v.extend([2u64, 8, 64, 1024].iter().map(|&n| limbs_sz(n)));
    v
}

/// The numerator of the size, the denominator of half as many limbs (the same for words).
fn round_q(rng: &mut Rng, sz: &Sz) -> Q {
    let db = if sz.limbs() <= 1 { sz.bits } else { sz.bits / 2 };
    rand_q(rng, sz.bits, db, sz.exact, true)
}

pub fn run(which: &str) {
    let mut rng = Rng::new(0x5EED_0002);
    if which == "new" || which == "all" {
        for sz in sizes() {
            let n = pool_for(sz.limbs(), 4);
            // Raw fractions: random parts, not reduced.
            let raw: Vec<Q> = (0..n)
                .map(|_| {
                    let num = if sz.exact { rng.exact(sz.bits) } else { rng.upto(sz.bits) };
                    let den = if sz.exact { rng.exact(sz.bits) } else { rng.upto(sz.bits) };
                    Q { neg: rng.bool(), num, den }
                })
                .collect();
            let on: Vec<(Number, Number)> = raw.iter().map(|q| (number(&q.num, q.neg), number(&q.den, false))).collect();
            let mn: Vec<(MInt, MInt)> = raw.iter().map(|q| (mint(&q.num, q.neg), mint(&q.den, false))).collect();
            let nn: Vec<(BigInt, BigInt)> = raw.iter().map(|q| (bigint(&q.num, q.neg), bigint(&q.den, false))).collect();
            #[cfg(feature = "rug")]
            let rn: Vec<(rug::Integer, rug::Integer)> = raw.iter().map(|q| (rint(&q.num, q.neg), rint(&q.den, false))).collect();
            let g = Gmp::new(&raw, false);
            #[cfg(has_flint)]
            let f = Flint::new(&raw, false);
            let mut s = Set::new("new", mk1("a", &on, K, |p| Rational::new(p.0.clone(), p.1.clone())));
            s.add(gmp1("GMP mpq", &g, 7, 0, K, Res::Q));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq", &f, 7, 0, K, Res::Q));
            s.add(mk1("malachite-q", &mn, K, |p| MQ::from_integers(p.0.clone(), p.1.clone())));
            s.add(mk1("num-rational BigRational", &nn, K, |p| BigRational::new(p.0.clone(), p.1.clone())));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rn, K, |p| rug::Rational::from((p.0.clone(), p.1.clone()))));
            row("new", "Rational::new", &format!("{}; raw random fraction; pool {n}", parts(&sz)), s.ents, 1.0, "ours/malachite/num/rug take the parts by value: a clone of both is in their time; GMP/FLINT: mpq_set into a variable that exists, then canonicalise");
        }
        // From primitive integers.
        for (bits, label) in [(16u32, "i64 of 16 bits"), (63, "i64 of 63 bits")] {
            let n = 4096;
            let xs: Vec<i64> = (0..n).map(|_| { let v = (rng.next() >> (64 - bits)) as i64; if rng.bool() { -v } else { v } }).collect();
            let qs: Vec<Q> = xs.iter().map(|_| Q { neg: false, num: vec![0], den: vec![1] }).collect();
            let g = Gmp::new(&qs, true);
            #[cfg(has_flint)]
            let f = Flint::new(&qs, true);
            for (i, &x) in xs.iter().enumerate() {
                g.set_si(i, x);
                #[cfg(has_flint)]
                f.set_si(i, x);
            }
            let mut s = Set::new("from i64", mk1("a", &xs, K, |&x| Rational::from(x)));
            s.add(gmp1("GMP mpq", &g, 11, 0, K, Res::Q));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq", &f, 11, 0, K, Res::Q));
            s.add(mk1("malachite-q", &xs, K, |&x| MQ::from(x)));
            s.add(mk1("num-rational BigRational", &xs, K, |&x| BigRational::from_integer(BigInt::from(x))));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &xs, K, |&x| rug::Rational::from(x)));
            row("new", "Rational::from(i64)", &format!("{label}; pool {n}"), s.ents, 1.0, "GMP/FLINT: mpq_set_si / fmpq_set_si into a variable that exists");
        }
        {
            let n = 4096;
            let xs: Vec<i128> = (0..n).map(|_| { let v = ((rng.next() as i128) << 63) | (rng.next() >> 1) as i128; let v = v | (1i128 << 126); if rng.bool() { -v } else { v } }).collect();
            let mut s = Set::new("from i128", mk1("a", &xs, K, |&x| Rational::from(x)));
            s.add(mk1("malachite-q", &xs, K, |&x| MQ::from(x)));
            s.add(mk1("num-rational BigRational", &xs, K, |&x| BigRational::from_integer(BigInt::from(x))));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &xs, K, |&x| rug::Rational::from(x)));
            row("new", "Rational::from(i128)", &format!("i128 of 127 bits; pool {n}"), s.ents, 1.0, "");
        }
        // Clone.
        for sz in sizes().into_iter().take(5) {
            let n = pool_for(sz.limbs(), 4);
            let qs: Vec<Q> = (0..n).map(|_| sz.q(&mut rng, true)).collect();
            let p = P::new(&qs);
            let mut s = Set::new("clone", mk1("a", &p.o, K, |x| x.clone()));
            s.add(gmp1("GMP mpq", &p.g, 6, 0, K, Res::Q));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq", &p.f, 6, 0, K, Res::Q));
            s.add(mk1("malachite-q", &p.m, K, |x| x.clone()));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.clone()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| x.clone()));
            row("new", "clone", &format!("{}; pool {n}", parts(&sz)), s.ents, 1.0, "GMP/FLINT: mpq_set / fmpq_set into a variable that exists");
        }
    }
    if which == "round" || which == "all" {
        for sz in sizes() {
            let n = pool_for(sz.limbs(), 4);
            let qs: Vec<Q> = (0..n).map(|_| round_q(&mut rng, &sz)).collect();
            let p = P::new(&qs);
            let par = if sz.limbs() <= 1 { format!("{}; random signs; pool {n}", parts(&sz)) } else { format!("numerator {} limbs, denominator {}; random signs; pool {n}", sz.limbs(), sz.limbs() / 2) };
            let into = "GMP/FLINT: into an mpz/fmpz that exists";
            // floor
            let mut s = Set::new("floor", mk1("a", &p.o, K, |x| x.floor()));
            s.add(gmp1("GMP mpz_fdiv_q", &p.g, 0, 0, K, Res::Z));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpz_fdiv_q", &p.f, 0, 0, K, Res::Z));
            s.add(mk1("malachite-q", &p.m, K, |x| x.floor()));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.floor()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Integer::from(x.floor_ref())));
            row("round", "floor", &par, s.ents, 1.0, &format!("{into}; num-rational returns a Ratio"));
            // ceil
            let mut s = Set::new("ceil", mk1("a", &p.o, K, |x| x.ceil()));
            s.add(gmp1("GMP mpz_cdiv_q", &p.g, 1, 0, K, Res::Z));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpz_cdiv_q", &p.f, 1, 0, K, Res::Z));
            s.add(mk1("malachite-q", &p.m, K, |x| x.ceiling()));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.ceil()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Integer::from(x.ceil_ref())));
            row("round", "ceil", &par, s.ents, 1.0, &format!("{into}; num-rational returns a Ratio"));
            // trunc
            let mut s = Set::new("trunc", mk1("a", &p.o, K, |x| x.trunc()));
            s.add(gmp1("GMP mpz_tdiv_q", &p.g, 2, 0, K, Res::Z));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpz_tdiv_q", &p.f, 2, 0, K, Res::Z));
            s.add(mk1("malachite-q", &p.m, K, |x| MInt::rounding_from(x, RoundingMode::Down).0));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.trunc()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Integer::from(x.trunc_ref())));
            row("round", "trunc", &par, s.ents, 1.0, &format!("{into}; num-rational returns a Ratio"));
            // The `_into` forms, into a number that exists, as GMP and FLINT write.
            for (op, code) in [("floor_into", 0), ("ceil_into", 1), ("trunc_into", 2)] {
                let o = &p.o;
                let nn = o.len();
                let mut out = Number::from(0u8);
                let mut j = 0usize;
                let ent = Ent::new("a", move |c| {
                    crate::cyc!(j, nn, c, {
                        match code {
                            0 => o[j].floor_into(&mut out),
                            1 => o[j].ceil_into(&mut out),
                            _ => o[j].trunc_into(&mut out),
                        }
                        std::hint::black_box(&out);
                    })
                });
                let mut ents = vec![ent];
                let (gname, fname) = match code {
                    0 => ("GMP mpz_fdiv_q", "FLINT fmpz_fdiv_q"),
                    1 => ("GMP mpz_cdiv_q", "FLINT fmpz_cdiv_q"),
                    _ => ("GMP mpz_tdiv_q", "FLINT fmpz_tdiv_q"),
                };
                // The same work as the rows above, whose results were checked.
                ents.push(gmp1(gname, &p.g, code, 0, 0, Res::Z).0);
                #[cfg(has_flint)]
                ents.push(flint1(fname, &p.f, code, 0, 0, Res::Z).0);
                #[cfg(not(has_flint))]
                let _ = fname;
                row("round", op, &par, ents, 1.0, "into a number that exists on every side");
            }
            // round (halves away from zero)
            let mut s = Set::new("round", mk1("a", &p.o, K, |x| x.round()));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpz_ndiv_qr", &p.f, 12, 0, K, Res::Z));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.round()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Integer::from(x.round_ref())));
            row("round", "round", &par, s.ents, 1.0, "halves away from zero; FLINT: fmpz_ndiv_qr (quotient and remainder) into variables that exist; num-rational returns a Ratio");
            // round_ties_even
            let mut s = Set::new("round_ties_even", mk1("a", &p.o, K, |x| x.round_ties_even()));
            s.add(mk1("malachite-q", &p.m, K, |x| MInt::rounding_from(x, RoundingMode::Nearest).0));
            row("round", "round_ties_even", &par, s.ents, 1.0, "malachite: Integer::rounding_from(&q, Nearest)");
            // fract
            let mut s = Set::new("fract", mk1("a", &p.o, K, |x| x.fract()));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.fract()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Rational::from(x.rem_trunc_ref())));
            row("round", "fract", &par, s.ents, 1.0, "rug: rem_trunc_ref");
        }
    }
    if which == "unary" || which == "all" {
        for sz in sizes().into_iter().take(4) {
            let n = pool_for(sz.limbs(), 4);
            let mut qs: Vec<Q> = (0..n).map(|_| sz.q(&mut rng, true)).collect();
            for q in qs.iter_mut() {
                if q.num == [0] {
                    q.num = vec![1];
                }
            }
            let p = P::new(&qs);
            let par = format!("{}; random signs; pool {n}", parts(&sz));
            let into = "GMP/FLINT: into a variable that exists";
            let mut s = Set::new("abs", mk1("a", &p.o, K, |x| x.abs()));
            s.add(gmp1("GMP mpq_abs", &p.g, 3, 0, K, Res::Q));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_abs", &p.f, 3, 0, K, Res::Q));
            s.add(mk1("malachite-q", &p.m, K, |x| x.abs()));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| Signed::abs(x)));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Rational::from(x.abs_ref())));
            row("unary", "abs", &par, s.ents, 1.0, into);

            let mut s = Set::new("neg", mk1("a", &p.o, K, |x| -x));
            s.add(gmp1("GMP mpq_neg", &p.g, 4, 0, K, Res::Q));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_neg", &p.f, 4, 0, K, Res::Q));
            s.add(mk1("malachite-q", &p.m, K, |x| -x));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| -x));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Rational::from(-x)));
            row("unary", "-x", &par, s.ents, 1.0, into);

            let mut s = Set::new("recip", mk1("a", &p.o, K, |x| x.recip()));
            s.add(gmp1("GMP mpq_inv", &p.g, 5, 0, K, Res::Q));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_inv", &p.f, 5, 0, K, Res::Q));
            s.add(mk1("malachite-q", &p.m, K, |x| x.reciprocal()));
            s.add(mk1("num-rational BigRational", &p.n, K, |x| x.recip()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Rational::from(x.recip_ref())));
            row("unary", "recip", &par, s.ents, 1.0, into);

            let mut s = Set::new("signum", mk1("a", &p.o, K, |x| x.signum()));
            s.add(gmp1("GMP mpq_sgn", &p.g, 10, 0, K, Res::Int));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_sgn", &p.f, 10, 0, K, Res::Int));
            s.add_note(mk1("malachite-q", &p.m, K, |x| x.sign()), "gives an Ordering");
            s.add(mk1("num-rational BigRational", &p.n, K, |x| Signed::signum(x)));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &p.r, K, |x| rug::Rational::from(x.signum_ref())));
            row("unary", "signum", &par, s.ents, 1.0, "ours, num and rug give a rational; GMP/FLINT an int");

            for e in [3i32, -7] {
                let mut s = Set::new("pow", mk1("a", &p.o, K, move |x| x.pow(e)));
                s.add(gmp1("GMP mpz_pow_ui of both parts", &p.g, 9, e as i64, K, Res::Q));
                #[cfg(has_flint)]
                s.add(flint1("FLINT fmpq_pow_si", &p.f, 9, e as i64, K, Res::Q));
                s.add(mk1("malachite-q", &p.m, K, move |x| x.pow(e as i64)));
                s.add(mk1("num-rational BigRational", &p.n, K, move |x| num_traits::Pow::pow(x, e)));
                #[cfg(feature = "rug")]
                s.add(mk1("rug Rational", &p.r, K, move |x| rug::Rational::from(rug::ops::Pow::pow(x, e))));
                row("unary", &format!("pow({e})"), &par, s.ents, 1.0, into);
            }

            if sz.limbs() <= 1 {
                let mut s = Set::new("hash", mk1("a", &p.o, K, fx));
                s.add_unchecked(mk1("malachite-q", &p.m, K, fx).0, "hashes its own form");
                s.add_unchecked(mk1("num-rational BigRational", &p.n, K, fx).0, "hashes its own form");
                #[cfg(feature = "rug")]
                s.add_unchecked(mk1("rug Rational", &p.r, K, fx).0, "hashes its own form");
                row("unary", "hash", &par, s.ents, 1.0, "into a word-at-a-time multiplicative hasher (FxHash), the same for all");
            }
        }
    }
    if which == "prim" || which == "all" {
        for sz in [s16(), w1()] {
            let n = 16384;
            let qs: Vec<Q> = (0..n).map(|_| sz.q(&mut rng, true)).collect();
            let p = P::new(&qs);
            let bits = sz.bits;
            let us: Vec<u32> = (0..n).map(|_| (rng.next() >> (64 - bits.min(32))) as u32).collect();
            let is: Vec<i64> = (0..n).map(|_| { let v = (rng.next() >> (65 - bits)) as i64; if rng.bool() { -v } else { v } }).collect();
            let i32s: Vec<i32> = (0..n).map(|_| { let v = (rng.next() >> (64 - bits.min(31))) as i32; if rng.bool() { -v } else { v } }).collect();
            // The integer operands as rationals of denominator 1 (for FLINT's fmpz and the Numbers).
            let ints: Vec<Q> = (0..n).map(|_| { let v = if sz.exact { rng.exact(bits) } else { rng.upto(bits) }; Q { neg: rng.bool(), num: v, den: vec![1] } }).collect();
            let on: Vec<Number> = ints.iter().map(|q| number(&q.num, q.neg)).collect();
            let nn: Vec<BigInt> = ints.iter().map(|q| bigint(&q.num, q.neg)).collect();
            #[cfg(feature = "rug")]
            let rn: Vec<rug::Integer> = ints.iter().map(|q| rint(&q.num, q.neg)).collect();
            #[cfg(has_flint)]
            let fi = Flint::new(&ints, true);
            #[cfg(has_flint)]
            let set_si = |v: &dyn Fn(usize) -> i64| {
                for i in 0..n {
                    p.f.set_si(i, v(i));
                }
            };
            let par = format!("{}; pool {n}", parts(&sz));
            let pairs_u: Vec<(&Rational, u32)> = p.o.iter().zip(us.iter().copied()).collect();
            #[cfg(feature = "rug")]
            let rpairs_u: Vec<(&rug::Rational, u32)> = p.r.iter().zip(us.iter().copied()).collect();
            // &a + u32, &a - u32
            #[cfg(has_flint)]
            set_si(&|i| us[i] as i64);
            let mut s = Set::new("+u32", mk1("a", &pairs_u, K, |&(x, y)| x + y));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_add_si", &p.f, 20, 0, K, Res::Q));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rpairs_u, K, |&(x, y)| rug::Rational::from(x + y)));
            row("prim", "&a + u32", &format!("{par}; u32 of {} bits", bits.min(32)), s.ents, 1.0, "FLINT: into a variable that exists");
            let mut s = Set::new("-u32", mk1("a", &pairs_u, K, |&(x, y)| x - y));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_sub_si", &p.f, 21, 0, K, Res::Q));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rpairs_u, K, |&(x, y)| rug::Rational::from(x - y)));
            row("prim", "&a - u32", &format!("{par}; u32 of {} bits", bits.min(32)), s.ents, 1.0, "FLINT: into a variable that exists");
            // &a * i64, a += i64
            #[cfg(has_flint)]
            set_si(&|i| is[i]);
            let pairs_i: Vec<(&Rational, i64)> = p.o.iter().zip(is.iter().copied()).collect();
            #[cfg(feature = "rug")]
            let rpairs_i: Vec<(&rug::Rational, i64)> = p.r.iter().zip(is.iter().copied()).collect();
            let mut s = Set::new("*i64", mk1("a", &pairs_i, K, |&(x, y)| x * y));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_mul_si", &p.f, 22, 0, K, Res::Q));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rpairs_i, K, |&(x, y)| rug::Rational::from(x * y)));
            row("prim", "&a * i64", &format!("{par}; i64 of {} bits", bits - 1), s.ents, 1.0, "FLINT: into a variable that exists");
            let mut s = Set::new("+=i64", mk1("a", &pairs_i, K, |&(x, y)| { let mut t = x.clone(); t += y; t }));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_add_si", &p.f, 23, 0, K, Res::Q));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rpairs_i, K, |&(x, y)| { let mut t = x.clone(); t += y; t }));
            row("prim", "a += i64", &format!("{par}; i64 of {} bits; fresh copy of a", bits - 1), s.ents, 1.0, "FLINT: fmpq_set then fmpq_add_si(r, r, c)");
            // &a + &n, &a * &n
            #[cfg(has_flint)]
            {
                let _ = &fi;
            }
            let mut s = Set::new("+n", mk2("a", &p.o, &on, K, |x, y| x + y));
            #[cfg(has_flint)]
            s.add(flint2("FLINT fmpq_add_fmpz", &p.f, &fi, 10, K, Res::Q));
            s.add(mk2("num-rational BigRational", &p.n, &nn, K, |x, y| x + y));
            #[cfg(feature = "rug")]
            s.add(mk2("rug Rational", &p.r, &rn, K, |x, y| rug::Rational::from(x + y)));
            row("prim", "&a + &n", &format!("{par}; n: a Number of the same size"), s.ents, 1.0, "FLINT: into a variable that exists");
            let mut s = Set::new("*n", mk2("a", &p.o, &on, K, |x, y| x * y));
            #[cfg(has_flint)]
            s.add(flint2("FLINT fmpq_mul_fmpz", &p.f, &fi, 11, K, Res::Q));
            s.add(mk2("num-rational BigRational", &p.n, &nn, K, |x, y| x * y));
            #[cfg(feature = "rug")]
            s.add(mk2("rug Rational", &p.r, &rn, K, |x, y| rug::Rational::from(x * y)));
            row("prim", "&a * &n", &format!("{par}; n: a Number of the same size"), s.ents, 1.0, "FLINT: into a variable that exists");
            // a.cmp(i32), a == i64
            #[cfg(has_flint)]
            set_si(&|i| i32s[i] as i64);
            let pairs_c: Vec<(&Rational, i32)> = p.o.iter().zip(i32s.iter().copied()).collect();
            let mpairs_c: Vec<(&MQ, i32)> = p.m.iter().zip(i32s.iter().copied()).collect();
            #[cfg(feature = "rug")]
            let rpairs_c: Vec<(&rug::Rational, i32)> = p.r.iter().zip(i32s.iter().copied()).collect();
            let mut s = Set::new("cmp i32", mk1("a", &pairs_c, K, |&(x, y)| x.partial_cmp(&y).unwrap()));
            #[cfg(has_flint)]
            s.add(flint1("FLINT fmpq_cmp_si", &p.f, 24, 0, K, Res::Sign));
            s.add(mk1("malachite-q", &mpairs_c, K, |&(x, y)| x.partial_cmp(&y).unwrap()));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rpairs_c, K, |&(x, y)| x.partial_cmp(&y).unwrap()));
            row("prim", "a.cmp(i32)", &format!("{par}; i32 of {} bits", bits.min(31)), s.ents, 1.0, "");
            // Equal to an i64: one in four equal (integers among the rationals).
            let eqs: Vec<i64> = (0..n).map(|i| if i % 4 == 0 && p.o[i].is_integer() { i64::try_from(&p.o[i]).unwrap_or(0) } else { is[i] }).collect();
            let pairs_e: Vec<(&Rational, i64)> = p.o.iter().zip(eqs.iter().copied()).collect();
            let mpairs_e: Vec<(&MQ, i64)> = p.m.iter().zip(eqs.iter().copied()).collect();
            #[cfg(feature = "rug")]
            let rpairs_e: Vec<(&rug::Rational, i64)> = p.r.iter().zip(eqs.iter().copied()).collect();
            let mut s = Set::new("== i64", mk1("a", &pairs_e, K, |&(x, y)| *x == y));
            s.add(mk1("malachite-q", &mpairs_e, K, |&(x, y)| *x == y));
            #[cfg(feature = "rug")]
            s.add(mk1("rug Rational", &rpairs_e, K, |&(x, y)| *x == y));
            row("prim", "a == i64", &par, s.ents, 1.0, "");
        }
    }
}

/// The parts of a rational (into_parts of a fresh copy) and is_integer.
pub fn run_access() {
    use malachite_base::num::conversion::traits::IsInteger;
    let mut rng = Rng::new(0x5EED_0009);
    for sz in sizes().into_iter().take(4) {
        let n = pool_for(sz.limbs(), 4);
        let qs: Vec<Q> = (0..n).map(|i| if i % 4 == 0 { Q { den: vec![1], ..sz.q(&mut rng, true) } } else { sz.q(&mut rng, true) }).collect();
        let p = P::new(&qs);
        let par = format!("{}; random signs; pool {n}", parts(&sz));
        let mut s = Set::new("into_parts", mk1("a", &p.o, K, |x| { let (a, b) = x.clone().into_parts(); format!("{}/{}", a.to_str_radix(16), b.to_str_radix(16)) }));
        let mut ents = vec![run1("a", &p.o, |x| x.clone().into_parts())];
        std::mem::swap(&mut s.ents, &mut ents);
        let mut s2 = s;
        s2.add_unchecked(run1("malachite-q", &p.m, |x| x.clone().into_numerator_and_denominator()), "two Naturals, the sign dropped");
        let nch = agree(s2.ours_checks(), &p.n.iter().take(K).map(|x| { let (a, b) = x.clone().into_raw(); format!("{}/{}", a.to_str_radix(16), b.to_str_radix(16)) }).collect::<Vec<_>>(), "into_parts num");
        s2.add_unchecked(run1("num-rational BigRational", &p.n, |x| x.clone().into_raw()), &nch);
        #[cfg(feature = "rug")]
        {
            let rch = agree(s2.ours_checks(), &p.r.iter().take(K).map(|x| { let (a, b) = x.clone().into_numer_denom(); format!("{}/{}", a.to_string_radix(16), b.to_string_radix(16)) }).collect::<Vec<_>>(), "into_parts rug");
            s2.add_unchecked(run1("rug Rational", &p.r, |x| x.clone().into_numer_denom()), &rch);
        }
        let _ = std::mem::replace(&mut ents, Vec::new());
        row("new", "into_parts (of a fresh copy)", &par, s2.ents, 1.0, "the copy is in every time");
        let mut s = Set::new("is_integer", mk1("a", &p.o, K, |x| x.is_integer()));
        s.add(mk1("malachite-q", &p.m, K, |x| x.is_integer()));
        s.add(mk1("num-rational BigRational", &p.n, K, |x| x.is_integer()));
        #[cfg(feature = "rug")]
        s.add(mk1("rug Rational", &p.r, K, |x| *x.denom() == 1));
        row("new", "is_integer", &format!("{par}; one in four an integer"), s.ents, 1.0, "");
    }
}
