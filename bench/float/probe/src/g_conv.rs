//! Conversions (primitive floats and integers, `Number`, `Rational`, rounding to integers,
//! precisions, comparisons) and text (writing and reading digits).
use crate::ffi::*;
use crate::h::*;
use crate::libs::*;
use a::math::float::BigFloat;
use a::math::numbers::Number;
use a::math::rational::Rational;
use std::ffi::{CStr, CString};

use ast::AF;
use dsh::DF;
use mal::{MF2, RM};
use malachite_base::num::conversion::traits::RoundingFrom;
use rug::Assign;

pub const CP: [u64; 10] = [24, 53, 64, 113, 128, 256, 1000, 10_000, 100_000, 1_000_000];

fn rand_f64s(l: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..l)
        .map(|_| {
            let m = (rng.next() >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            let x = (0.25 + 3.75 * m) * if rng.next() & 1 == 1 { -1.0 } else { 1.0 };
            x
        })
        .collect()
}

fn cf(d: f64) -> Canon {
    Canon::Str(format!("{:016x}", d.to_bits()))
}

pub fn conv_rows() {
    for p in CP {
        let ps = format!("prec={p}");
        let l = pool(p, 2);
        let mut rng = Rng::new(p ^ 0xc0);
        let xs: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect();
        // ---- to_f64_round
        if wanted("conv", "to_f64_round", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_f64_round(N), |v| cf(v.0))));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf("MPFR mpfr_get_d", kern!(l, |k| unsafe { mpfr_get_d(xm[k].c(), RN) }, |v| cf(v))));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float f64::rounding_from", kern!(l, |k| f64::rounding_from(&xl[k], RM), |v| cf(v.0))));
            let xd: Vec<DF> = xs.iter().map(|r| dsh::f(r, p)).collect();
            conts.push(Cont::rf("dashu-float to_f64", kern!(l, |k| xd[k].to_f64().value(), |v| cf(v))));
            row("conv", "to_f64_round", &ps, l.min(8), conts);
        }
        // ---- set_f64 into an existing number, and from_f64_round with a new one
        let ds = rand_f64s(l, p ^ 0xd0);
        let draw: Vec<Raw> = ds.iter().map(|&d| Raw::from_f64(d)).collect();
        if wanted("conv", "set_f64", &ps) {
            let mut conts = Vec::new();
            let d1 = ds.clone();
            let mut r = BigFloat::zero(p);
            conts.push(Cont::ours(kern!(l, |k| r.set_f64(d1[k], N), |v| c_ours(&r))));
            let d2 = ds.clone();
            let mut rm = MF::new(p);
            conts.push(Cont::rf("MPFR mpfr_set_d", kern!(l, |k| unsafe { mpfr_set_d(rm.p(), d2[k], RN) }, |v| c_mpfr(rm.c()))));
            let d3 = ds.clone();
            let mut rr = rug::Float::new(p as u32);
            conts.push(Cont::rf("rug assign", kern!(l, |k| rr.assign(d3[k]), |v| rg::c(&rr))));
            row("conv", "set_f64", &ps, l.min(8), conts);
        }
        if wanted("conv", "from_f64_round", &ps) {
            let mut conts = Vec::new();
            let d1 = ds.clone();
            conts.push(Cont::ours(kern!(l, |k| BigFloat::from_f64_round(d1[k], p, N), |v| c_ours(&v.0))));
            let d2 = ds.clone();
            conts.push(Cont::rf(
                "MPFR init2+mpfr_set_d+clear",
                kern!(l, |k| unsafe { let mut r = MF::new(p); mpfr_set_d(r.p(), d2[k], RN); r }, |v| c_mpfr(v.c())),
            ));
            let d3 = ds.clone();
            conts.push(Cont::rf("rug with_val", kern!(l, |k| rug::Float::with_val(p as u32, d3[k]), |v| rg::c(&v))));
            let d4 = ds.clone();
            conts.push(Cont::rf(
                "malachite-float from_primitive_float_prec_round",
                kern!(l, |k| MF2::from_primitive_float_prec_round(d4[k], p, RM), |v| mal::c(&v.0)),
            ));
            let d5 = ds.clone();
            conts.push(Cont::rf(
                "dashu-float try_from+with_precision",
                kern!(l, |k| DF::try_from(d5[k]).unwrap().with_precision(p as usize).value(), |v| dsh::c(&v)),
            ));
            let wp_ = ast::wp(p);
            let tag = if wp_ != p { "word-precision" } else { "" };
            if !tag.is_empty() {
                let d6 = ds.clone();
                conts.push(Cont::ours_tag(tag, kern!(l, |k| BigFloat::from_f64_round(d6[k], wp_, N), |v| c_ours(&v.0))));
            }
            let d7 = ds.clone();
            let mut ct = Cont::rf("astro-float from_f64", kern!(l, |k| AF::from_f64(d7[k], wp_ as usize), |v| ast::c(&v))).vs(tag);
            if !tag.is_empty() {
                ct = ct.note(&format!("astro rounds to whole words: {wp_} bits"));
            }
            conts.push(ct);
            let _ = &draw;
            row("conv", "from_f64_round", &ps, l.min(8), conts);
        }
        // ---- floor / set_floor: a number with half its bits after the point
        let hs: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, (p / 2) as i64, (p / 2) as i64, true)).collect();
        if wanted("conv", "set_floor", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = hs.iter().map(|r| ours(r, p)).collect();
            let mut r = BigFloat::zero(p);
            conts.push(Cont::ours(kern!(l, |k| r.set_floor(&x[k]), |v| c_ours(&r))));
            let xm: Vec<MF> = hs.iter().map(|r| mpfr(r, p)).collect();
            let mut rm = MF::new(p);
            conts.push(Cont::rf("MPFR mpfr_floor", kern!(l, |k| unsafe { mpfr_floor(rm.p(), xm[k].c()) }, |v| c_mpfr(rm.c()))));
            let xr: Vec<rug::Float> = hs.iter().map(|r| rg::f(r, p)).collect();
            let mut rr = rug::Float::new(p as u32);
            conts.push(Cont::rf("rug assign(floor_ref)", kern!(l, |k| rr.assign(xr[k].floor_ref()), |v| rg::c(&rr))));
            row("conv", "set_floor", &ps, l.min(8), conts);
        }
        if wanted("conv", "floor", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = hs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].floor(), |v| c_ours(&v))));
            let xm: Vec<MF> = hs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf(
                "MPFR init2+mpfr_floor+clear",
                kern!(l, |k| unsafe { let mut r = MF::new(p); mpfr_floor(r.p(), xm[k].c()); r }, |v| c_mpfr(v.c())),
            ));
            let xr: Vec<rug::Float> = hs.iter().map(|r| rg::f(r, p)).collect();
            conts.push(Cont::rf("rug with_val(floor_ref)", kern!(l, |k| rug::Float::with_val(p as u32, xr[k].floor_ref()), |v| rg::c(&v))));
            let xd: Vec<DF> = hs.iter().map(|r| dsh::f(r, p)).collect();
            conts.push(Cont::rf("dashu-float floor", kern!(l, |k| xd[k].floor(), |v| dsh::c(&v))));
            let wp_ = ast::wp(p);
            let tag = if wp_ != p { "word-precision" } else { "" };
            if !tag.is_empty() {
                let xw: Vec<BigFloat> = hs.iter().map(|r| ours(r, wp_)).collect();
                conts.push(Cont::ours_tag(tag, kern!(l, |k| xw[k].floor(), |v| c_ours(&v))));
            }
            let xa: Vec<AF> = hs.iter().map(|r| ast::f(r, wp_)).collect();
            conts.push(Cont::rf("astro-float floor", kern!(l, |k| xa[k].floor(), |v| ast::c(&v))).vs(tag));
            row("conv", "floor", &ps, l.min(8), conts);
        }
        // ---- set_rint / rint_round to half the precision
        let h = (p / 2).max(1);
        if wanted("conv", "set_rint", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = hs.iter().map(|r| ours(r, p)).collect();
            let mut r = BigFloat::zero(h);
            conts.push(Cont::ours(kern!(l, |k| r.set_rint(&x[k], N), |v| c_ours(&r))));
            let xm: Vec<MF> = hs.iter().map(|r| mpfr(r, p)).collect();
            let mut rm = MF::new(h);
            conts.push(Cont::rf("MPFR mpfr_rint", kern!(l, |k| unsafe { mpfr_rint(rm.p(), xm[k].c(), RN) }, |v| c_mpfr(rm.c()))).note("result of p/2 bits"));
            row("conv", "set_rint", &ps, l.min(8), conts);
        }
        if wanted("conv", "rint_round(p/2)", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = hs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].rint_round(h, N), |v| c_ours(&v.0))));
            let xm: Vec<MF> = hs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf(
                "MPFR init2+mpfr_rint+clear",
                kern!(l, |k| unsafe { let mut r = MF::new(h); mpfr_rint(r.p(), xm[k].c(), RN); r }, |v| c_mpfr(v.c())),
            ));
            row("conv", "rint_round(p/2)", &ps, l.min(8), conts);
        }
        // ---- prec_round to half the precision (a new number)
        if wanted("conv", "prec_round(p/2)", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].prec_round(h, N), |v| c_ours(&v.0))));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf(
                "MPFR init2+mpfr_set+clear",
                kern!(l, |k| unsafe { let mut r = MF::new(h); mpfr_set(r.p(), xm[k].c(), RN); r }, |v| c_mpfr(v.c())),
            ));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf(
                "malachite-float from_float_prec_round_ref",
                kern!(l, |k| MF2::from_float_prec_round_ref(&xl[k], h, RM), |v| mal::c(&v.0)),
            ));
            row("conv", "prec_round(p/2)", &ps, l.min(8), conts);
        }
        // ---- comparisons
        if wanted("conv", "a < b", &ps) {
            let ys: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect();
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            let y: Vec<BigFloat> = ys.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k] < y[k], |v| Canon::Str(v.to_string()))));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            let ym: Vec<MF> = ys.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf("MPFR mpfr_cmp", kern!(l, |k| unsafe { mpfr_cmp(xm[k].c(), ym[k].c()) < 0 }, |v| Canon::Str(v.to_string()))));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            let yl: Vec<MF2> = ys.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float <", kern!(l, |k| xl[k] < yl[k], |v| Canon::Str(v.to_string()))));
            let xd: Vec<DF> = xs.iter().map(|r| dsh::f(r, p)).collect();
            let yd: Vec<DF> = ys.iter().map(|r| dsh::f(r, p)).collect();
            conts.push(Cont::rf("dashu-float <", kern!(l, |k| xd[k] < yd[k], |v| Canon::Str(v.to_string()))));
            row("conv", "a < b", &ps, l.min(8), conts);
        }
        // ---- integers: to_i64_round, to_number_round, from_number; rationals
        if p <= 1000 && wanted("conv", "to_i64_round", &ps) {
            let is: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, 30, 62, true)).collect();
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = is.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_i64_round(N), |v| Canon::Str(v.unwrap().0.to_string()))));
            let xm: Vec<MF> = is.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf("MPFR mpfr_get_si", kern!(l, |k| unsafe { mpfr_get_si(xm[k].c(), RN) }, |v| Canon::Str(v.to_string()))).note("MPFR returns no direction"));
            let xl: Vec<MF2> = is.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float i64::rounding_from", kern!(l, |k| i64::rounding_from(&xl[k], RM), |v| Canon::Str(v.0.to_string()))));
            row("conv", "to_i64_round", &ps, l.min(8), conts);
        }
        if p <= 100_000 && wanted("conv", "to_number_round", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = hs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_number_round(N), |v| {
                let (n, _) = v.unwrap();
                let (neg, l) = number_limbs(&n);
                Canon::fin(neg, l, 0)
            })));
            let xm: Vec<MF> = hs.iter().map(|r| mpfr(r, p)).collect();
            let mut z = MZ::new();
            conts.push(Cont::rf("MPFR mpfr_get_z", kern!(l, |k| unsafe { mpfr_get_z(&mut z.0, xm[k].c(), RN) }, |v| {
                let (neg, l) = z.limbs();
                Canon::fin(neg, l, 0)
            }))
            .note("MPFR into an existing mpz_t"));
            let xl: Vec<MF2> = hs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf(
                "malachite-float Integer::rounding_from",
                kern!(l, |k| malachite_nz::integer::Integer::rounding_from(&xl[k], RM), |v| {
                    let n = v.0.unsigned_abs_ref().to_limbs_asc();
                    Canon::fin(v.0 < 0, n, 0)
                }),
            ));
            row("conv", "to_number_round", &ps, l.min(8), conts);
        }
        if p <= 100_000 && wanted("conv", "from_number (2p bits)", &ps) {
            // integers of 2p bits rounded to p
            let ns: Vec<Raw> = (0..l).map(|_| { let neg = rng.next() & 1 == 1; Raw::random(&mut rng, 2 * p, 2 * p as i64, neg) }).collect();
            let mut conts = Vec::new();
            let nn: Vec<Number> = ns.iter().map(|r| number(r.neg, &r.limbs)).collect();
            conts.push(Cont::ours(kern!(l, |k| BigFloat::from_number(&nn[k], p, N), |v| c_ours(&v.0))));
            let zs: Vec<MZ> = ns.iter().map(|r| MZ::from_limbs(r.neg, &r.limbs)).collect();
            conts.push(Cont::rf(
                "MPFR init2+mpfr_set_z+clear",
                kern!(l, |k| unsafe { let mut r = MF::new(p); mpfr_set_z(r.p(), &zs[k].0, RN); r }, |v| c_mpfr(v.c())),
            ));
            let is: Vec<malachite_nz::integer::Integer> = ns
                .iter()
                .map(|r| {
                    let n = malachite_nz::integer::Integer::from(mal::Natural::from_limbs_asc(&r.limbs));
                    if r.neg { -n } else { n }
                })
                .collect();
            conts.push(Cont::rf(
                "malachite-float from_integer_prec_round_ref",
                kern!(l, |k| MF2::from_integer_prec_round_ref(&is[k], p, RM), |v| mal::c(&v.0)),
            ));
            row("conv", "from_number (2p bits)", &ps, l.min(8), conts);
        }
        if p <= 10_000 && wanted("conv", "from_rational (p-bit terms)", &ps) {
            let mut conts = Vec::new();
            let nr: Vec<(Raw, Raw)> = (0..l)
                .map(|_| { let neg = rng.next() & 1 == 1; (Raw::random(&mut rng, p, p as i64, neg), Raw::random(&mut rng, p, p as i64, false)) })
                .collect();
            let qs: Vec<Rational> = nr
                .iter()
                .map(|(a, b)| {
                    let mut b = b.clone();
                    b.limbs[0] |= 1;
                    Rational::from(number(a.neg, &a.limbs)) / Rational::from(number(false, &b.limbs))
                })
                .collect();
            conts.push(Cont::ours(kern!(l, |k| BigFloat::from_rational(&qs[k], p, N), |v| c_ours(&v.0))));
            let mq: Vec<MQ> = nr
                .iter()
                .map(|(a, b)| {
                    let mut b = b.clone();
                    b.limbs[0] |= 1;
                    let mut q = MQ::new();
                    let za = MZ::from_limbs(a.neg, &a.limbs);
                    let zb = MZ::from_limbs(false, &b.limbs);
                    unsafe {
                        mpz_set(&mut q.0.num, &za.0);
                        mpz_set(&mut q.0.den, &zb.0);
                        mpq_canonicalize(&mut q.0);
                    }
                    q
                })
                .collect();
            conts.push(Cont::rf(
                "MPFR init2+mpfr_set_q+clear",
                kern!(l, |k| unsafe { let mut r = MF::new(p); mpfr_set_q(r.p(), &mq[k].0, RN); r }, |v| c_mpfr(v.c())),
            ));
            let lq: Vec<malachite_q::Rational> = nr
                .iter()
                .map(|(a, b)| {
                    let mut b = b.clone();
                    b.limbs[0] |= 1;
                    let n = malachite_nz::integer::Integer::from(mal::Natural::from_limbs_asc(&a.limbs));
                    let n = if a.neg { -n } else { n };
                    malachite_q::Rational::from_integers(n, malachite_nz::integer::Integer::from(mal::Natural::from_limbs_asc(&b.limbs)))
                })
                .collect();
            conts.push(Cont::rf(
                "malachite-float from_rational_prec_round_ref",
                kern!(l, |k| MF2::from_rational_prec_round_ref(&lq[k], p, RM), |v| mal::c(&v.0)),
            ));
            row("conv", "from_rational (p-bit terms)", &ps, l.min(8), conts);
        }
        if p <= 100_000 && wanted("conv", "Rational::try_from", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| Rational::try_from(&x[k]).unwrap(), |v| Canon::Skip)));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            let mut q = MQ::new();
            conts.push(Cont::rf("MPFR mpfr_get_q", kern!(l, |k| unsafe { mpfr_get_q(&mut q.0, xm[k].c()) }, |v| Canon::Skip)).note("MPFR into an existing mpq_t"));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float Rational::try_from", kern!(l, |k| malachite_q::Rational::try_from(&xl[k]).unwrap(), |v| Canon::Skip)));
            row("conv", "Rational::try_from", &ps, 0, conts);
        }
    }
}

// ------------------------------------------------------------------------------------------
// Text.

fn mpfr_digits(x: &MF, n: usize) -> Canon {
    unsafe {
        let mut e: std::os::raw::c_long = 0;
        let s = mpfr_get_str(std::ptr::null_mut(), &mut e, 10, n, x.c(), RN);
        let t = CStr::from_ptr(s).to_string_lossy().trim_start_matches('-').to_string();
        mpfr_free_str(s);
        Canon::Str(format!("{t}e{e}"))
    }
}

/// Reads `s` back with MPFR at `p` bits: the canonical value a written text stands for.
fn mpfr_read(s: &str, p: u64) -> Canon {
    let mut r = MF::new(p);
    let c = CString::new(s.replace('@', "e")).unwrap();
    unsafe { mpfr_strtofr(r.p(), c.as_ptr(), std::ptr::null_mut(), 10, RN) };
    c_mpfr(r.c())
}

pub fn text_rows() {
    // writing
    for p in CP {
        let ps = format!("prec={p}");
        let l = pool(p, 2).min(16);
        let mut rng = Rng::new(p ^ 0x7e);
        let xs: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect();
        if wanted("text", "to_digits(10, None)", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_digits(10, None, N), |v| {
                let (d, e, _) = v.unwrap();
                Canon::Str(format!("{d}e{e}"))
            })));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf(
                "MPFR mpfr_get_str(n=0)+free",
                kern!(l, |k| unsafe {
                    let mut e: std::os::raw::c_long = 0;
                    let s = mpfr_get_str(std::ptr::null_mut(), &mut e, 10, 0, xm[k].c(), RN);
                    mpfr_free_str(s);
                    e
                }, |v| mpfr_digits(&xm[k], 0)),
            ));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf(
                "malachite-float get_str(n=0)",
                kern!(l, |k| malachite_float::float::conversion::string::get_str::get_str(&xl[k], 10, 0, RM), |v| {
                    let (d, e, _) = v.unwrap();
                    Canon::Str(format!("{}e{e}", String::from_utf8(d).unwrap().trim_start_matches('-')))
                }),
            ));
            row("text", "to_digits(10, None)", &ps, l.min(6), conts);
        }
        if p >= 1000 && wanted("text", "to_digits(10, Some(20))", &ps) {
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_digits(10, Some(20), N), |v| {
                let (d, e, _) = v.unwrap();
                Canon::Str(format!("{d}e{e}"))
            })));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf(
                "MPFR mpfr_get_str(n=20)+free",
                kern!(l, |k| unsafe {
                    let mut e: std::os::raw::c_long = 0;
                    let s = mpfr_get_str(std::ptr::null_mut(), &mut e, 10, 20, xm[k].c(), RN);
                    mpfr_free_str(s);
                    e
                }, |v| mpfr_digits(&xm[k], 20)),
            ));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf(
                "malachite-float get_str(n=20)",
                kern!(l, |k| malachite_float::float::conversion::string::get_str::get_str(&xl[k], 10, 20, RM), |v| {
                    let (d, e, _) = v.unwrap();
                    Canon::Str(format!("{}e{e}", String::from_utf8(d).unwrap().trim_start_matches('-')))
                }),
            ));
            row("text", "to_digits(10, Some(20))", &ps, l.min(6), conts);
        }
        if wanted("text", "to_string()", &ps) {
            // Display: ours the shortest digits that read back; the others 1 + ceil(p·log10 2).
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_string(), |v| mpfr_read(&v, p))));
            let xr: Vec<rug::Float> = xs.iter().map(|r| rg::f(r, p)).collect();
            conts.push(Cont::rf("rug to_string", kern!(l, |k| xr[k].to_string(), |v| mpfr_read(&v, p))).note("rug writes all the digits of mpfr_get_str, a writes the shortest that read back; compared by reading back"));
            let xl: Vec<MF2> = xs.iter().map(|r| mal::f(r, p)).collect();
            conts.push(Cont::rf("malachite-float to_string", kern!(l, |k| xl[k].to_string(), |v| mpfr_read(&v, p))).note("1+ceil(p log10 2) digits; compared by reading back"));
            let xd: Vec<DF> = xs.iter().map(|r| dsh::f(r, p)).collect();
            conts.push(Cont::rf("dashu-float to_decimal+to_string", kern!(l, |k| xd[k].to_decimal().value().to_string(), |v| mpfr_read(&v, p))).loose().note("dashu's decimal of the same digit count, not checked to read back"));
            row("text", "to_string()", &ps, l.min(6), conts);
        }
    }
    // writing with exponents up to ±2^60 at 53 bits
    {
        let p = 53;
        let ps = "prec=53 exponents to ±2^60".to_string();
        if wanted("text", "to_digits(10, None)", &ps) {
            let l = 16;
            let mut rng = Rng::new(0xe60);
            let xs: Vec<Raw> = (0..l).map(|_| Raw::rand_range(&mut rng, p, -(1 << 60), 1 << 60, true)).collect();
            let mut conts = Vec::new();
            let x: Vec<BigFloat> = xs.iter().map(|r| ours(r, p)).collect();
            conts.push(Cont::ours(kern!(l, |k| x[k].to_digits(10, None, N), |v| {
                let (d, e, _) = v.unwrap();
                Canon::Str(format!("{d}e{e}"))
            })));
            let xm: Vec<MF> = xs.iter().map(|r| mpfr(r, p)).collect();
            conts.push(Cont::rf(
                "MPFR mpfr_get_str(n=0)+free",
                kern!(l, |k| unsafe {
                    let mut e: std::os::raw::c_long = 0;
                    let s = mpfr_get_str(std::ptr::null_mut(), &mut e, 10, 0, xm[k].c(), RN);
                    mpfr_free_str(s);
                    e
                }, |v| mpfr_digits(&xm[k], 0)),
            ));
            row("text", "to_digits(10, None)", &ps, 6, conts);
        }
    }
    // reading
    for &n in &[20usize, 100, 1000, 10_000, 100_000, 1_000_000] {
        let worth = ((n as f64) * std::f64::consts::LOG2_10).ceil() as u64;
        for (label, p) in [("worth (ceil(n·log₂ ten) bits)", worth), ("double (fifty-three bits)", 53u64)] {
            let ps = format!("digits={n} {label}");
            if !wanted("text", "from_str_radix(10)", &ps) {
                continue;
            }
            let l = if n >= 100_000 { 4 } else if n >= 1000 { 8 } else { 32 };
            let mut rng = Rng::new(n as u64 ^ p);
            let texts: Vec<String> = (0..l)
                .map(|_| {
                    let mut s = String::with_capacity(n + 8);
                    if rng.next() & 1 == 1 {
                        s.push('-');
                    }
                    s.push((b'1' + (rng.below(9) as u8)) as char);
                    s.push('.');
                    for _ in 1..n {
                        s.push((b'0' + rng.below(10) as u8) as char);
                    }
                    s.push_str(&format!("e{}", rng.below(41) as i64 - 20));
                    s
                })
                .collect();
            let mut conts = Vec::new();
            let t1 = texts.clone();
            conts.push(Cont::ours(kern!(l, |k| BigFloat::from_str_radix(&t1[k], 10, p, N).unwrap(), |v| c_ours(&v.0))));
            let cs: Vec<CString> = texts.iter().map(|t| CString::new(t.as_str()).unwrap()).collect();
            let mut r = MF::new(p);
            conts.push(Cont::rf(
                "MPFR mpfr_strtofr",
                kern!(l, |k| unsafe { mpfr_strtofr(r.p(), cs[k].as_ptr(), std::ptr::null_mut(), 10, RN) }, |v| c_mpfr(r.c())),
            )
            .note("MPFR into an existing variable"));
            let t2 = texts.clone();
            conts.push(Cont::rf(
                "malachite-float strtofr",
                kern!(l, |k| malachite_float::float::conversion::string::strtofr::strtofr(&t2[k], 10, p, RM), |v| mal::c(&v.0)),
            ));
            let t3 = texts.clone();
            conts.push(Cont::rf(
                "rug parse+with_val",
                kern!(l, |k| rug::Float::with_val(p as u32, rug::Float::parse(&t3[k]).unwrap()), |v| rg::c(&v)),
            ));
            let t4 = texts.clone();
            conts.push(Cont::rf(
                "dashu-float DBig parse+with_base_and_precision",
                kern!(l, |k| {
                    let d: dashu_float::DBig = t4[k].parse().unwrap();
                    d.with_base_and_precision::<2>(p as usize).value()
                }, |v| {
                    let w: DF = v.with_rounding();
                    dsh::c(&w)
                }),
            )
            .loose()
            .note("dashu reads decimal exactly then converts with HalfAway rounding"));
            let wp_ = ast::wp(p);
            let tag = if wp_ != p { "word-precision" } else { "" };
            if !tag.is_empty() {
                let t5 = texts.clone();
                conts.push(Cont::ours_tag(tag, kern!(l, |k| BigFloat::from_str_radix(&t5[k], 10, wp_, N).unwrap(), |v| c_ours(&v.0))));
            }
            let t6 = texts.clone();
            let mut cc = ast::Consts::new().unwrap();
            let mut ct = Cont::rf(
                "astro-float parse",
                kern!(l, |k| AF::parse(&t6[k], astro_float::Radix::Dec, wp_ as usize, ast::RM, &mut cc), |v| ast::c(&v)),
            )
            .vs(tag);
            if !tag.is_empty() {
                ct = ct.note(&format!("astro rounds to whole words: {wp_} bits"));
            }
            conts.push(ct);
            row("text", "from_str_radix(10)", &ps, l.min(6), conts);
        }
    }
}

/// Short decimal texts (1 to 19 significant digits, as printed doubles) at 53 bits, and 100
/// hexadecimal digits at their 400 bits.
pub fn text_short_rows() {
    for (label, radix, p) in [("1 to 19 digits, 53 bits", 10u32, 53u64), ("100 hex digits, 400 bits", 16, 400)] {
        let ps = format!("radix={radix} {label}");
        if !wanted("text", "from_str_radix", &ps) {
            continue;
        }
        let l = 64usize;
        let mut rng = Rng::new(0x5407 ^ radix as u64);
        let texts: Vec<String> = (0..l)
            .map(|_| {
                let n = if radix == 10 { 1 + rng.below(19) as usize } else { 100 };
                let dig = |v: u64| std::char::from_digit(v as u32, radix).unwrap();
                let mut s = String::new();
                if rng.next() & 1 == 1 {
                    s.push('-');
                }
                s.push(dig(1 + rng.below(radix as u64 - 1)));
                if n > 1 {
                    s.push('.');
                    for _ in 1..n {
                        s.push(dig(rng.below(radix as u64)));
                    }
                }
                if radix == 10 {
                    s.push_str(&format!("e{}", rng.below(61) as i64 - 30));
                } else {
                    s.push_str(&format!("@{}", rng.below(21) as i64 - 10));
                }
                s
            })
            .collect();
        let mut conts = Vec::new();
        let t1 = texts.clone();
        conts.push(Cont::ours(kern!(l, |k| BigFloat::from_str_radix(&t1[k], radix, p, N).unwrap(), |v| c_ours(&v.0))));
        let cs: Vec<CString> = texts.iter().map(|t| CString::new(t.as_str()).unwrap()).collect();
        let mut r = MF::new(p);
        conts.push(
            Cont::rf("MPFR mpfr_strtofr", kern!(l, |k| unsafe { mpfr_strtofr(r.p(), cs[k].as_ptr(), std::ptr::null_mut(), radix as _, RN) }, |v| c_mpfr(r.c())))
                .note("MPFR into an existing variable"),
        );
        let t2 = texts.clone();
        conts.push(Cont::rf(
            "malachite-float strtofr",
            kern!(l, |k| malachite_float::float::conversion::string::strtofr::strtofr(&t2[k], radix as u8, p, RM), |v| mal::c(&v.0)),
        ));
        if radix == 10 {
            let t3 = texts.clone();
            conts.push(Cont::rf("rug parse+with_val", kern!(l, |k| rug::Float::with_val(p as u32, rug::Float::parse(&t3[k]).unwrap()), |v| rg::c(&v))));
        }
        row("text", "from_str_radix", &ps, 16, conts);
    }
}

pub fn run(g: &str) {
    match g {
        "conv" => conv_rows(),
        "text" => {
            text_rows();
            text_short_rows();
        }
        _ => panic!("conv|text"),
    }
}
