//! a::math::uint's Montgomery<N> and Uint::{pow_mod, mul_mod, add_mod, mod_inverse} against ruint,
//! crypto-bigint, GMP (mpz_powm, mpz_invert, mpn), malachite-nz and OpenSSL's BN, at 64 to 4096
//! bits (N = 1, 2, 4, 6, 8, 16, 32, 64). Moduli are random of full width (top bit set), odd, or
//! even with 1 to 32 trailing zeros; operands below the modulus; exponents of N random limbs.
//! Every side's results are compared with ours before timing.

#[path = "../common.rs"]
mod common;
#[path = "../gmp.rs"]
mod gmp;

use a::math::uint::{Montgomery, Uint};
use common::*;
use crypto_bigint::modular::{MontyForm, MontyParams};
use crypto_bigint::{NonZero, Odd};
use malachite_base::num::arithmetic::traits::{ModAdd, ModInverse, ModMul, ModPow};
use malachite_nz::natural::Natural;
use std::hint::black_box;

type Ru<const B: usize, const L: usize> = ruint::Uint<B, L>;
type Cb<const L: usize> = crypto_bigint::Uint<L>;

fn nat(l: &[u64]) -> Natural {
    Natural::from_limbs_asc(l)
}
fn nat_limbs<const N: usize>(n: &Natural) -> [u64; N] {
    let v = n.to_limbs_asc();
    let mut out = [0u64; N];
    out[..v.len()].copy_from_slice(&v);
    out
}
fn sig(l: &[u64]) -> usize {
    let mut n = l.len();
    while n > 0 && l[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// −m⁻¹ mod 2⁶⁴ for odd m₀, by Newton's iteration.
fn neg_inv(m0: u64) -> u64 {
    let mut y = m0;
    for _ in 0..6 {
        y = y.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(y)));
    }
    y.wrapping_neg()
}

/// A random x below m (rejection on the top limb).
fn below<const N: usize>(rng: &mut Rng, m: &[u64; N]) -> [u64; N] {
    loop {
        let x: [u64; N] = rng.limbs();
        if Uint::from_limbs(x) < Uint::from_limbs(*m) {
            return x;
        }
    }
}

fn modulus<const N: usize>(rng: &mut Rng, odd: bool) -> [u64; N] {
    let mut m: [u64; N] = rng.limbs();
    m[N - 1] |= 1 << 63;
    if odd {
        m[0] |= 1;
    } else {
        // 1 to 32 trailing zeros, the top bit kept
        let k = 1 + rng.below(32) as u32;
        let mut u = Uint::<N>::from_limbs(m);
        u = (u >> k) << k;
        m = *u.as_limbs();
        if N == 1 && m[0] == 1 << 63 {
            m[0] |= 1 << 40;
        }
    }
    m
}

macro_rules! mont_bench {
    ($fname:ident, $n:literal, $bits:literal) => {
        #[allow(clippy::too_many_lines, unused_variables, unused_mut)]
        fn $fname(out: &Out, seed: u64) {
            const N: usize = $n;
            const BITS: usize = $bits;
            let g = "mont";
            let p = format!("N={N} ({BITS} bit)");
            let mut rng = Rng::new(seed);
            let len = pool_len(3 * 8 * N).max(256);
            let m: [u64; N] = modulus(&mut rng, true);
            let mu = Uint::<N>::from_limbs(m);
            let f = Montgomery::new(&mu);
            let inv = neg_inv(m[0]);
            let rm = Ru::<BITS, N>::from_limbs(m);
            let cp = MontyParams::new_vartime(Odd::new(Cb::<N>::from_words(m)).unwrap());

            // ---- Montgomery products ----------------------------------------------------
            {
                // forms below m: ours, ruint and crypto-bigint use the same R = 2^(64N)
                let l: Vec<([u64; N], [u64; N])> = (0..len).map(|_| (below(&mut rng, &m), below(&mut rng, &m))).collect();
                let o: Vec<(Uint<N>, Uint<N>)> = l.iter().map(|(a, b)| (Uint::from_limbs(*a), Uint::from_limbs(*b))).collect();
                let r: Vec<(Ru<BITS, N>, Ru<BITS, N>)> = l.iter().map(|(a, b)| (Ru::from_limbs(*a), Ru::from_limbs(*b))).collect();
                // MontyForm from its Montgomery form directly
                let c: Vec<(MontyForm<N>, MontyForm<N>)> = l
                    .iter()
                    .map(|(a, b)| {
                        let fa = MontyForm::new(&Cb::from_words(*a), cp);
                        let fb = MontyForm::new(&Cb::from_words(*b), cp);
                        (fa, fb)
                    })
                    .collect();
                // the inputs of crypto-bigint as forms of the same numbers: ours on (to_mont(a), to_mont(b))
                let oc: Vec<(Uint<N>, Uint<N>)> = c
                    .iter()
                    .map(|(a, b)| (Uint::from_limbs(a.as_montgomery().to_words()), Uint::from_limbs(b.as_montgomery().to_words())))
                    .collect();
                let want: Vec<[u64; N]> = o.iter().map(|(a, b)| *f.mul(a, b).as_limbs()).collect();
                same("mmul", "ruint", &r, &want, |(a, b)| *a.mul_redc(*b, rm, inv).as_limbs());
                let wantc: Vec<[u64; N]> = oc.iter().map(|(a, b)| *f.mul(a, b).as_limbs()).collect();
                same("mmul", "cb", &c, &wantc, |(a, b)| a.mul(b).as_montgomery().to_words());
                #[cfg(has_openssl)]
                let ossl = {
                    let ctx = unsafe { gmp::BN_CTX_new() };
                    let mont = unsafe { gmp::BN_MONT_CTX_new() };
                    let bm = gmp::Bn::of(&m);
                    unsafe { gmp::BN_MONT_CTX_set(mont, bm.0, ctx) };
                    let bl: Vec<(gmp::Bn, gmp::Bn)> = l.iter().map(|(a, b)| (gmp::Bn::of(a), gmp::Bn::of(b))).collect();
                    (ctx, mont, bm, bl)
                };
                let fr = &f;
                let mut s = vec![
                    mk("ours", "", &o, move |(a, b)| fr.mul(a, b)),
                    mk("ruint", "mul_redc", &r, move |(a, b)| a.mul_redc(*b, rm, inv)),
                    mk("crypto-bigint", "MontyForm::mul (each value carries its params)", &c, |(a, b)| a.mul(b)),
                ];
                #[cfg(has_openssl)]
                {
                    let (ctx, mont, _, bl) = &ossl;
                    let (ctx, mont) = (*ctx, *mont);
                    let rr = gmp::Bn::new();
                    for ((a, b), w) in bl.iter().zip(&want) {
                        unsafe { gmp::BN_mod_mul_montgomery(rr.0, a.0, b.0, mont, ctx) };
                        assert_eq!(&rr.limbs::<N>(), w, "mmul: OpenSSL");
                    }
                    s.push(side_n("OpenSSL BN_mod_mul_montgomery", "BIGNUM operands, BN_MONT_CTX", move || {
                        for (a, b) in black_box(bl) {
                            unsafe { gmp::BN_mod_mul_montgomery(rr.0, a.0, b.0, mont, ctx) };
                        }
                        black_box(&rr);
                    }));
                }
                out.row(g, "Montgomery::mul", &format!("{p}, random odd modulus"), len, "independent products", s);

                let want: Vec<[u64; N]> = o.iter().map(|(a, _)| *f.square(a).as_limbs()).collect();
                same("msqr", "ruint", &r, &want, |(a, _)| *a.square_redc(rm, inv).as_limbs());
                let wantc: Vec<[u64; N]> = oc.iter().map(|(a, _)| *f.square(a).as_limbs()).collect();
                same("msqr", "cb", &c, &wantc, |(a, _)| a.square().as_montgomery().to_words());
                let mut s = vec![
                    mk("ours", "a product of a number by itself", &o, move |(a, _)| fr.square(a)),
                    mk("ruint", "square_redc", &r, move |(a, _)| a.square_redc(rm, inv)),
                    mk("crypto-bigint", "MontyForm::square", &c, |(a, _)| a.square()),
                ];
                #[cfg(has_openssl)]
                {
                    let (ctx, mont, _, bl) = &ossl;
                    let (ctx, mont) = (*ctx, *mont);
                    let rr = gmp::Bn::new();
                    for ((a, _), w) in bl.iter().zip(&want) {
                        unsafe { gmp::BN_mod_mul_montgomery(rr.0, a.0, a.0, mont, ctx) };
                        assert_eq!(&rr.limbs::<N>(), w, "msqr: OpenSSL");
                    }
                    s.push(side_n("OpenSSL BN_mod_mul_montgomery", "a·a (its squaring path)", move || {
                        for (a, _) in black_box(bl) {
                            unsafe { gmp::BN_mod_mul_montgomery(rr.0, a.0, a.0, mont, ctx) };
                        }
                        black_box(&rr);
                    }));
                }
                out.row(g, "Montgomery::square", &format!("{p}, random odd modulus"), len, "independent squarings", s);

                // to and from the form
                let want: Vec<[u64; N]> = o.iter().map(|(a, _)| *f.to_montgomery(a).as_limbs()).collect();
                let cx: Vec<(Cb<N>, ())> = l.iter().map(|(a, _)| (Cb::from_words(*a), ())).collect();
                same("tomont", "cb", &cx, &want, |(a, _)| MontyForm::new(a, cp).as_montgomery().to_words());
                let s2: Vec<(MontyForm<N>, ())> = c.iter().map(|(a, _)| (*a, ())).collect();
                let mut s = vec![
                    mk("ours", "", &o, move |(a, _)| fr.to_montgomery(a)),
                    mk("crypto-bigint", "MontyForm::new", &cx, move |(a, _)| MontyForm::new(a, cp)),
                ];
                out.row(g, "Montgomery::to_montgomery", &p, len, "", s);
                let want: Vec<[u64; N]> = oc.iter().map(|(a, _)| *f.from_montgomery(a).as_limbs()).collect();
                same("frommont", "cb", &s2, &want, |(a, _)| a.retrieve().to_words());
                let mut s = vec![
                    mk("ours", "", &oc, move |(a, _)| fr.from_montgomery(a)),
                    mk("crypto-bigint", "MontyForm::retrieve", &s2, |(a, _)| a.retrieve()),
                ];
                out.row(g, "Montgomery::from_montgomery", &p, len, "", s);
                #[cfg(has_openssl)]
                unsafe {
                    gmp::BN_CTX_free(ossl.0);
                    gmp::BN_MONT_CTX_free(ossl.1);
                }

                // the form's inverse
                let want: Vec<Option<[u64; N]>> = oc.iter().map(|(a, _)| f.inverse(a).map(|x| *x.as_limbs())).collect();
                same("minv", "cb", &s2, &want, |(a, _)| {
                    Option::<MontyForm<N>>::from(a.inv_vartime()).map(|x| x.as_montgomery().to_words())
                });
                let ilen0 = (2048 / N).clamp(16, 256);
                let (oci, s2i) = (&oc[..ilen0], &s2[..ilen0]);
                let mut s = vec![
                    mk("ours", "", oci, move |(a, _)| fr.inverse(a)),
                    mk("crypto-bigint", "MontyForm::inv_vartime", s2i, |(a, _)| a.inv_vartime()),
                    mk("crypto-bigint (ct)", "MontyForm::inv, constant-time", s2i, |(a, _)| a.inv()),
                ];
                out.row(g, "Montgomery::inverse", &format!("{p}, random odd modulus"), ilen0, "", s);
            }

            // ---- the constants of a modulus ---------------------------------------------
            {
                let ms: Vec<([u64; N], ())> = (0..len.min(512)).map(|_| (modulus(&mut rng, true), ())).collect();
                let om: Vec<(Uint<N>, ())> = ms.iter().map(|(a, _)| (Uint::from_limbs(*a), ())).collect();
                let cm: Vec<(Odd<Cb<N>>, ())> = ms.iter().map(|(a, _)| (Odd::new(Cb::from_words(*a)).unwrap(), ())).collect();
                let want: Vec<[u64; N]> = om.iter().map(|(a, _)| *Montgomery::new(a).one().as_limbs()).collect();
                same("mnew", "cb", &cm, &want, |(a, _)| MontyForm::one(MontyParams::new_vartime(*a)).as_montgomery().to_words());
                let mut s = vec![
                    mk("ours", "", &om, |(a, _)| Montgomery::new(a)),
                    mk("crypto-bigint", "MontyParams::new_vartime (also R³)", &cm, |(a, _)| MontyParams::new_vartime(*a)),
                ];
                #[cfg(has_openssl)]
                {
                    let ctx = unsafe { gmp::BN_CTX_new() };
                    let bl: Vec<gmp::Bn> = ms.iter().map(|(a, _)| gmp::Bn::of(a)).collect();
                    let mont = unsafe { gmp::BN_MONT_CTX_new() };
                    s.push(side_n("OpenSSL BN_MONT_CTX_set", "into a reused BN_MONT_CTX", move || {
                        for b in black_box(&bl) {
                            unsafe { gmp::BN_MONT_CTX_set(mont, b.0, ctx) };
                        }
                        black_box(mont);
                    }));
                }
                out.row(g, "Montgomery::new", &format!("{p}, random odd modulus"), ms.len(), "", s);
            }

            // ---- powers -----------------------------------------------------------------
            let plen = (256 / N).max(2);
            for odd in [true, false] {
                let ms: Vec<[u64; N]> = (0..plen).map(|_| modulus(&mut rng, odd)).collect();
                let l: Vec<([u64; N], [u64; N], [u64; N])> = ms
                    .iter()
                    .map(|mm| {
                        let mut e: [u64; N] = rng.limbs();
                        e[N - 1] |= 1 << 63;
                        (below(&mut rng, mm), e, *mm)
                    })
                    .collect();
                let o: Vec<(Uint<N>, Uint<N>, Uint<N>)> = l
                    .iter()
                    .map(|(x, e, mm)| (Uint::from_limbs(*x), Uint::from_limbs(*e), Uint::from_limbs(*mm)))
                    .collect();
                let r: Vec<(Ru<BITS, N>, Ru<BITS, N>, Ru<BITS, N>)> = l
                    .iter()
                    .map(|(x, e, mm)| (Ru::from_limbs(*x), Ru::from_limbs(*e), Ru::from_limbs(*mm)))
                    .collect();
                let nl: Vec<(Natural, Natural, Natural)> = l.iter().map(|(x, e, mm)| (nat(x), nat(e), nat(mm))).collect();
                let want: Vec<[u64; N]> = o.iter().map(|(x, e, mm)| *x.pow_mod(e, mm).as_limbs()).collect();
                same("powmod", "ruint", &r, &want, |(x, e, mm)| *x.pow_mod(*e, *mm).as_limbs());
                same("powmod", "malachite", &nl, &want, |(x, e, mm)| nat_limbs::<N>(&x.mod_pow(e, mm)));
                let cl: Vec<(Cb<N>, Cb<N>, Odd<Cb<N>>)> = if odd {
                    l.iter()
                        .map(|(x, e, mm)| (Cb::from_words(*x), Cb::from_words(*e), Odd::new(Cb::from_words(*mm)).unwrap()))
                        .collect()
                } else {
                    Vec::new()
                };
                #[cfg(has_gmp)]
                let gl: Vec<(gmp::Z, gmp::Z, gmp::Z)> = l.iter().map(|(x, e, mm)| (gmp::Z::of(x), gmp::Z::of(e), gmp::Z::of(mm))).collect();
                #[cfg(has_openssl)]
                let bl: Vec<(gmp::Bn, gmp::Bn, gmp::Bn)> = l.iter().map(|(x, e, mm)| (gmp::Bn::of(x), gmp::Bn::of(e), gmp::Bn::of(mm))).collect();
                let mut s = vec![
                    mk("ours", "", &o, |(x, e, mm)| x.pow_mod(e, mm)),
                    mk("a Number (diagnostic)", "this crate's Number::pow_mod, with the conversions", &o, |(x, e, mm)| {
                        use a::math::numbers::Number;
                        Uint::<N>::try_from(&Number::from(x).pow_mod(&Number::from(e), &Number::from(mm))).unwrap()
                    }),
                    mk("ruint", "square-and-multiply with mul_mod", &r, |(x, e, mm)| x.pow_mod(*e, *mm)),
                    mk("malachite-nz", "allocates", &nl, |(x, e, mm)| x.mod_pow(e, mm)),
                ];
                if odd {
                    same("powmod", "cb", &cl, &want, |(x, e, mm)| {
                        MontyForm::new(x, MontyParams::new_vartime(*mm)).pow(e).retrieve().to_words()
                    });
                    s.push(mk(
                        "crypto-bigint",
                        "MontyParams::new_vartime + MontyForm::pow (constant-time) + retrieve",
                        &cl,
                        |(x, e, mm)| MontyForm::new(x, MontyParams::new_vartime(*mm)).pow(e).retrieve(),
                    ));
                }
                #[cfg(has_gmp)]
                {
                    let gl = &gl;
                    let mut rr = gmp::Z::new();
                    for ((x, e, mm), w) in gl.iter().zip(&want) {
                        unsafe { gmp::__gmpz_powm(rr.m(), x.p(), e.p(), mm.p()) };
                        assert_eq!(&rr.limbs::<N>(), w, "powmod: GMP");
                    }
                    s.push(side_n("GMP mpz_powm", "into a preallocated mpz", move || {
                        for (x, e, mm) in black_box(gl) {
                            unsafe { gmp::__gmpz_powm(rr.m(), x.p(), e.p(), mm.p()) };
                        }
                        black_box(&mut rr);
                    }));
                }
                #[cfg(has_openssl)]
                {
                    let bl = &bl;
                    let ctx = unsafe { gmp::BN_CTX_new() };
                    let rr = gmp::Bn::new();
                    for ((x, e, mm), w) in bl.iter().zip(&want) {
                        unsafe { gmp::BN_mod_exp(rr.0, x.0, e.0, mm.0, ctx) };
                        assert_eq!(&rr.limbs::<N>(), w, "powmod: OpenSSL");
                    }
                    s.push(side_n(
                        "OpenSSL BN_mod_exp",
                        if odd { "Montgomery context set up per call" } else { "even modulus: reciprocal method" },
                        move || {
                            for (x, e, mm) in black_box(bl) {
                                unsafe { gmp::BN_mod_exp(rr.0, x.0, e.0, mm.0, ctx) };
                            }
                            black_box(&rr);
                        },
                    ));
                }
                let what = if odd { "odd" } else { "even" };
                out.row(
                    g,
                    "Uint::pow_mod",
                    &format!("{p}, random {what} modulus, exponent of {BITS} bits"),
                    plen,
                    "",
                    s,
                );

                if odd {
                    // with the constants of the modulus kept: one modulus, many powers
                    let xs: Vec<(Uint<N>, Uint<N>)> = o.iter().map(|(x, e, _)| (f.to_montgomery(x), *e)).collect();
                    let cx: Vec<(MontyForm<N>, Cb<N>)> = l
                        .iter()
                        .map(|(x, e, _)| (MontyForm::new(&Cb::from_words(*x), cp), Cb::from_words(*e)))
                        .collect();
                    let want: Vec<[u64; N]> = xs.iter().map(|(x, e)| *f.pow(x, e).as_limbs()).collect();
                    let wantc: Vec<[u64; N]> = cx
                        .iter()
                        .map(|(x, e)| *f.pow(&Uint::from_limbs(x.as_montgomery().to_words()), &Uint::<N>::from_limbs(e.to_words())).as_limbs())
                        .collect();
                    same("mpow", "cb", &cx, &wantc, |(x, e)| x.pow(e).as_montgomery().to_words());
                    let fr = &f;
                    let mut s = vec![
                        mk("ours", "", &xs, move |(x, e)| fr.pow(x, e)),
                        mk("crypto-bigint", "MontyForm::pow (constant-time window)", &cx, |(x, e)| x.pow(e)),
                    ];
                    #[cfg(has_openssl)]
                    {
                        let ctx = unsafe { gmp::BN_CTX_new() };
                        let mont = unsafe { gmp::BN_MONT_CTX_new() };
                        let bm = gmp::Bn::of(&m);
                        unsafe { gmp::BN_MONT_CTX_set(mont, bm.0, ctx) };
                        let bl: Vec<(gmp::Bn, gmp::Bn)> = l.iter().map(|(x, e, _)| (gmp::Bn::of(x), gmp::Bn::of(e))).collect();
                        let rr = gmp::Bn::new();
                        // results are numbers, not forms: check against ours by value
                        for ((x, e), (ox, oe)) in bl.iter().zip(&xs) {
                            unsafe { gmp::BN_mod_exp_mont(rr.0, x.0, e.0, bm.0, ctx, mont) };
                            assert_eq!(&rr.limbs::<N>(), f.from_montgomery(&f.pow(ox, oe)).as_limbs(), "mpow: OpenSSL");
                        }
                        s.push(side_n(
                            "OpenSSL BN_mod_exp_mont",
                            "BN_MONT_CTX kept; takes and gives numbers (two conversions)",
                            move || {
                                // the whole Bn moved in, not only its pointer (edition-2021 captures)
                                let bm = &bm;
                                for (x, e) in black_box(&bl) {
                                    unsafe { gmp::BN_mod_exp_mont(rr.0, x.0, e.0, bm.0, ctx, mont) };
                                }
                                black_box(&rr);
                            },
                        ));
                    }
                    out.row(
                        g,
                        "Montgomery::pow",
                        &format!("{p}, random odd modulus, exponent of {BITS} bits"),
                        plen,
                        "the modulus's constants computed once",
                        s,
                    );
                }
            }

            // ---- inverses ---------------------------------------------------------------
            let ilen = (2048 / N).clamp(16, 512);
            for odd in [true, false] {
                let l: Vec<([u64; N], [u64; N])> = (0..ilen)
                    .map(|_| {
                        let mm = modulus(&mut rng, odd);
                        let mut x = below(&mut rng, &mm);
                        if !odd {
                            x[0] |= 1;
                        }
                        (x, mm)
                    })
                    .collect();
                let o: Vec<(Uint<N>, Uint<N>)> = l.iter().map(|(x, mm)| (Uint::from_limbs(*x), Uint::from_limbs(*mm))).collect();
                let r: Vec<(Ru<BITS, N>, Ru<BITS, N>)> = l.iter().map(|(x, mm)| (Ru::from_limbs(*x), Ru::from_limbs(*mm))).collect();
                let nl: Vec<(Natural, Natural)> = l.iter().map(|(x, mm)| (nat(x), nat(mm))).collect();
                let want: Vec<Option<[u64; N]>> = o.iter().map(|(x, mm)| x.mod_inverse(mm).map(|v| *v.as_limbs())).collect();
                same("inv", "ruint", &r, &want, |(x, mm)| x.inv_mod(*mm).map(|v| *v.as_limbs()));
                same("inv", "malachite", &nl, &want, |(x, mm)| x.mod_inverse(mm).map(|v| nat_limbs::<N>(&v)));
                let cl: Vec<(Cb<N>, Cb<N>)> = l.iter().map(|(x, mm)| (Cb::from_words(*x), Cb::from_words(*mm))).collect();
                let co: Vec<(Cb<N>, Odd<Cb<N>>)> = cl.iter().filter(|_| odd).map(|(x, mm)| (*x, Odd::new(*mm).unwrap())).collect();
                #[cfg(has_gmp)]
                let gl: Vec<(gmp::Z, gmp::Z)> = l.iter().map(|(x, mm)| (gmp::Z::of(x), gmp::Z::of(mm))).collect();
                #[cfg(has_openssl)]
                let bl: Vec<(gmp::Bn, gmp::Bn)> = l.iter().map(|(x, mm)| (gmp::Bn::of(x), gmp::Bn::of(mm))).collect();
                let prepared: Vec<(MontyForm<N>, ())> = if odd {
                    l.iter()
                        .map(|(x, mm)| {
                            let params = MontyParams::new_vartime(Odd::new(Cb::from_words(*mm)).unwrap());
                            (MontyForm::new(&Cb::from_words(*x), params), ())
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let mut s = vec![
                    mk("ours", "", &o, |(x, mm)| x.mod_inverse(mm)),
                    mk("a Number (diagnostic)", "this crate's Number::mod_inverse, with the conversions", &o, |(x, mm)| {
                        a::math::numbers::Number::from(x).mod_inverse(&a::math::numbers::Number::from(mm)).map(|v| Uint::<N>::try_from(&v).unwrap())
                    }),
                    mk("ruint", "inv_mod", &r, |(x, mm)| x.inv_mod(*mm)),
                    mk("malachite-nz", "allocates", &nl, |(x, mm)| x.mod_inverse(mm)),
                ];
                if odd {
                    same("inv", "cb", &co, &want, |(x, mm)| Option::<Cb<N>>::from(x.inv_odd_mod(mm)).map(|v| v.to_words()));
                    s.push(mk("crypto-bigint", "inv_odd_mod (safegcd, constant-time)", &co, |(x, mm)| x.inv_odd_mod(mm)));
                    s.push(mk("crypto-bigint (prepared)", "MontyParams prepared, MontyForm::inv_vartime of the form", &prepared, |(x, _)| x.inv_vartime()));
                } else {
                    same("inv", "cb", &cl, &want, |(x, mm)| Option::<Cb<N>>::from(x.inv_mod(mm)).map(|v| v.to_words()));
                    s.push(mk("crypto-bigint", "inv_mod (constant-time)", &cl, |(x, mm)| x.inv_mod(mm)));
                }
                #[cfg(has_gmp)]
                {
                    let gl = &gl;
                    let mut rr = gmp::Z::new();
                    for ((x, mm), w) in gl.iter().zip(&want) {
                        let ok = unsafe { gmp::__gmpz_invert(rr.m(), x.p(), mm.p()) };
                        assert_eq!(if ok != 0 { Some(rr.limbs::<N>()) } else { None }, *w, "inv: GMP");
                    }
                    s.push(side_n("GMP mpz_invert", "into a preallocated mpz", move || {
                        for (x, mm) in black_box(gl) {
                            unsafe { gmp::__gmpz_invert(rr.m(), x.p(), mm.p()) };
                        }
                        black_box(&mut rr);
                    }));
                }
                #[cfg(has_openssl)]
                {
                    let bl = &bl;
                    let ctx = unsafe { gmp::BN_CTX_new() };
                    let rr = gmp::Bn::new();
                    let mut ok_all = true;
                    for ((x, mm), w) in bl.iter().zip(&want) {
                        let p = unsafe { gmp::BN_mod_inverse(rr.0, x.0, mm.0, ctx) };
                        let got = if p.is_null() { None } else { Some(rr.limbs::<N>()) };
                        ok_all &= got == *w;
                    }
                    assert!(ok_all, "inv: OpenSSL");
                    s.push(side_n("OpenSSL BN_mod_inverse", "", move || {
                        for (x, mm) in black_box(bl) {
                            unsafe { gmp::BN_mod_inverse(rr.0, x.0, mm.0, ctx) };
                        }
                        black_box(&rr);
                    }));
                }
                let what = if odd { "odd" } else { "even" };
                out.row(g, "Uint::mod_inverse", &format!("{p}, random {what} modulus"), ilen, "", s);
            }

            // ---- products and sums modulo any m -----------------------------------------
            for odd in [true, false] {
                let l: Vec<([u64; N], [u64; N], [u64; N])> = (0..len.min(2048))
                    .map(|_| {
                        let mm = modulus(&mut rng, odd);
                        (below(&mut rng, &mm), below(&mut rng, &mm), mm)
                    })
                    .collect();
                let o: Vec<(Uint<N>, Uint<N>, Uint<N>)> =
                    l.iter().map(|(a, b, mm)| (Uint::from_limbs(*a), Uint::from_limbs(*b), Uint::from_limbs(*mm))).collect();
                let r: Vec<(Ru<BITS, N>, Ru<BITS, N>, Ru<BITS, N>)> =
                    l.iter().map(|(a, b, mm)| (Ru::from_limbs(*a), Ru::from_limbs(*b), Ru::from_limbs(*mm))).collect();
                let c: Vec<(Cb<N>, Cb<N>, NonZero<Cb<N>>)> = l
                    .iter()
                    .map(|(a, b, mm)| (Cb::from_words(*a), Cb::from_words(*b), NonZero::new(Cb::from_words(*mm)).unwrap()))
                    .collect();
                let nl: Vec<(Natural, Natural, Natural)> = l.iter().map(|(a, b, mm)| (nat(a), nat(b), nat(mm))).collect();
                let want: Vec<[u64; N]> = o.iter().map(|(a, b, mm)| *a.mul_mod(b, mm).as_limbs()).collect();
                same("mulmod", "ruint", &r, &want, |(a, b, mm)| *a.mul_mod(*b, *mm).as_limbs());
                same("mulmod", "cb", &c, &want, |(a, b, mm)| a.mul_mod_vartime(b, mm).to_words());
                same("mulmod", "malachite", &nl, &want, |(a, b, mm)| nat_limbs::<N>(&a.mod_mul(b, mm)));
                #[cfg(has_openssl)]
                let bl: Vec<(gmp::Bn, gmp::Bn, gmp::Bn)> = l.iter().map(|(a, b, mm)| (gmp::Bn::of(a), gmp::Bn::of(b), gmp::Bn::of(mm))).collect();
                let mut s = vec![
                    mk("ours", "", &o, |(a, b, mm)| a.mul_mod(b, mm)),
                    mk("a Number (diagnostic)", "this crate's Number product and remainder, with the conversions", &o, |(a, b, mm)| {
                        let p = &a::math::numbers::Number::from(a) * &a::math::numbers::Number::from(b);
                        Uint::<N>::try_from(&p.rem_euclid(&a::math::numbers::Number::from(mm))).unwrap()
                    }),
                    mk("ruint", "", &r, |(a, b, mm)| a.mul_mod(*b, *mm)),
                    mk("crypto-bigint", "mul_mod_vartime", &c, |(a, b, mm)| a.mul_mod_vartime(b, mm)),
                    mk("malachite-nz", "allocates", &nl, |(a, b, mm)| a.mod_mul(b, mm)),
                ];
                #[cfg(has_gmp)]
                {
                    let gf = |(a, b, mm): &([u64; N], [u64; N], [u64; N])| {
                        let mut prod = [0u64; 2 * N];
                        unsafe { gmp::__gmpn_mul_n(prod.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N as _) };
                        let mut q = [0u64; 2 * N];
                        let mut rr = [0u64; N];
                        let nn = sig(&prod).max(1);
                        if nn >= N {
                            unsafe {
                                gmp::__gmpn_tdiv_qr(q.as_mut_ptr(), rr.as_mut_ptr(), 0, prod.as_ptr(), nn as _, mm.as_ptr(), N as _)
                            };
                        } else {
                            rr[..nn].copy_from_slice(&prod[..nn]);
                        }
                        rr
                    };
                    same("mulmod", "gmp", &l, &want, gf);
                    s.push(mk("GMP mpn_mul_n + mpn_tdiv_qr", "", &l, gf));
                }
                #[cfg(has_openssl)]
                {
                    let bl = &bl;
                    let ctx = unsafe { gmp::BN_CTX_new() };
                    let rr = gmp::Bn::new();
                    for ((a, b, mm), w) in bl.iter().zip(&want) {
                        unsafe { gmp::BN_mod_mul(rr.0, a.0, b.0, mm.0, ctx) };
                        assert_eq!(&rr.limbs::<N>(), w, "mulmod: OpenSSL");
                    }
                    s.push(side_n("OpenSSL BN_mod_mul", "", move || {
                        for (a, b, mm) in black_box(bl) {
                            unsafe { gmp::BN_mod_mul(rr.0, a.0, b.0, mm.0, ctx) };
                        }
                        black_box(&rr);
                    }));
                }
                let what = if odd { "odd" } else { "even" };
                out.row(g, "Uint::mul_mod", &format!("{p}, random {what} modulus, reduced operands"), l.len(), "", s);

                if odd {
                    let want: Vec<[u64; N]> = o.iter().map(|(a, b, mm)| *a.add_mod(b, mm).as_limbs()).collect();
                    same("addmod", "ruint", &r, &want, |(a, b, mm)| *a.add_mod(*b, *mm).as_limbs());
                    let c2: Vec<(Cb<N>, Cb<N>, Cb<N>)> = c.iter().map(|(a, b, mm)| (*a, *b, *mm.as_ref())).collect();
                    same("addmod", "cb", &c2, &want, |(a, b, mm)| a.add_mod(b, mm).to_words());
                    same("addmod", "malachite", &nl, &want, |(a, b, mm)| nat_limbs::<N>(&a.mod_add(b, mm)));
                    let mut s = vec![
                        mk("ours", "", &o, |(a, b, mm)| a.add_mod(b, mm)),
                        mk("ruint", "", &r, |(a, b, mm)| a.add_mod(*b, *mm)),
                        mk("crypto-bigint", "needs reduced operands", &c2, |(a, b, mm)| a.add_mod(b, mm)),
                        mk("malachite-nz", "allocates", &nl, |(a, b, mm)| a.mod_add(b, mm)),
                    ];
                    out.row(g, "Uint::add_mod", &format!("{p}, reduced operands"), l.len(), "", s);
                }
            }
        }
    };
}

mont_bench!(run1, 1, 64);
mont_bench!(run2, 2, 128);
mont_bench!(run4, 4, 256);
mont_bench!(run6, 6, 384);
mont_bench!(run8, 8, 512);
mont_bench!(run16, 16, 1024);
mont_bench!(run32, 32, 2048);
mont_bench!(run64, 64, 4096);

fn main() {
    let out = Out::new("numtypes");
    let only: Option<usize> = std::env::var("NTBENCH_N").ok().and_then(|s| s.parse().ok());
    let runs: [(usize, fn(&Out, u64)); 8] =
        [(1, run1), (2, run2), (4, run4), (6, run6), (8, run8), (16, run16), (32, run32), (64, run64)];
    for (n, f) in runs {
        if only.is_none_or(|k| k == n) {
            f(&out, 100 + n as u64);
        }
    }
}
