//! Lattice reductions and enumeration: lll, lll_fp, deep_lll, is_lll_reduced, bkz,
//! shortest_vector, closest_vector, short_vectors.
#![allow(unused_variables, unused_mut)]
use crate::inputs::*;
#[allow(unused_imports)]
use crate::refs;
use crate::timing::Side;
use crate::{Ctx, Ref};
use a::math::lattice::{
    BkzOptions, DeepLllOptions, LllOptions, bkz as a_bkz, closest_vector, deep_lll, is_lll_reduced,
    lll as a_lll, lll_fp, short_vectors, shortest_vector,
};
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use std::hint::black_box;

/// The length in bits of the longest entry.
fn max_bits(b: &Matrix<Number>) -> u64 {
    b.as_slice().iter().map(|x| x.bits()).max().unwrap_or(0)
}

fn quick() -> bool {
    std::env::var("LATBENCH_QUICK").is_ok()
}

/// What a reference's reduced basis is, against ours: reduced by the exact test (to the
/// parameters given), and the log₂ of the ratio of the first squared norms.
fn reduced_note(ours: &Matrix<Number>, out: Option<String>, r: usize, c: usize, delta: f64, eta: f64, check: bool) -> String {
    match out {
        None => "no output".into(),
        Some(s) if s == "FAIL" => "ref reported failure".into(),
        Some(s) if s.starts_with("ERROR") => s,
        Some(s) => {
            let m = parse_matrix(&s, r, c);
            let red = if !check {
                "unchecked"
            } else if is_lll_reduced(&m, delta, eta) {
                "ref output (δ,η)-reduced"
            } else if is_lll_reduced(&m, delta, 0.55) {
                "ref output reduced only with η=0.55"
            } else {
                "ref output NOT reduced at (δ,η)"
            };
            let lr = log2_ratio(&first_norm2(&m), &first_norm2(ours));
            format!("{red}; |b0|² ref/ours = 2^{lr:+.2}")
        }
    }
}

pub fn lll_cases() -> Vec<(String, Matrix<Number>)> {
    let mut rng = Rng(23);
    let mut v: Vec<(String, Matrix<Number>)> = Vec::new();
    let q = quick();
    for &(n, b) in if q { &[(20usize, 200u64)][..] } else { &[(20, 200), (40, 400), (60, 600), (80, 800), (100, 100), (50, 1000), (100, 1000)][..] } {
        v.push((format!("knapsack n={n} bits={b}"), knapsack(&mut rng, n, b)));
    }
    if q {
        return v;
    }
    for &(d, qb) in &[(20usize, 10u64), (40, 12), (40, 20), (80, 14)] {
        v.push((format!("NTRU-like n={} q={qb}b", 2 * d), ntru(&mut rng, d, qb)));
    }
    for &(n, qb) in &[(40usize, 20u64), (80, 20), (120, 30)] {
        v.push((format!("q-ary n={n} k={} q={qb}b", n / 2), qary(&mut rng, n, n / 2, qb)));
    }
    for &(n, b) in &[(20usize, 10u64), (50, 100), (100, 10), (100, 100), (200, 10)] {
        v.push((format!("uniform n={n} bits={b}"), uniform(&mut rng, n, n, b)));
    }
    for &(n, b) in &[(40usize, 400u64), (60, 600), (80, 800), (100, 1000)] {
        v.push((format!("Goldstein-Mayer n={n} bits={b}"), goldstein_mayer(&mut rng, n, b)));
    }
    v
}

/// `lll_fp` (L², always with U) against fplll, FLINT, NTL and PARI.
pub fn lll(ctx: &Ctx) {
    let o = LllOptions::new();
    for (name, b) in lll_cases() {
        if !ctx.wants("lll_fp", &name) {
            continue;
        }
        let (r, c) = b.shape();
        let t = text(&b);
        let ours = lll_fp(&b, &o);
        let check = r <= 120;
        #[cfg(has_fplll)]
        let fp = refs::fplll::m(&t, r, c);
        #[cfg(has_flint)]
        let fl = refs::flint::z(&t, r, c);
        #[cfg(has_ntl)]
        let nt = refs::ntl::m(&t, r, c);
        #[cfg(has_pari)]
        let pa = refs::pari::z(&t, r, c, true);
        #[cfg(has_ntl)]
        let xd = refs::ntl::needs_xd(max_bits(&b));
        #[cfg(has_ntl)]
        let (k1, k2, s1, s2) = if xd {
            (refs::ntl::LLL_XD, refs::ntl::G_LLL_XD, "LLL_XD", "G_LLL_XD")
        } else {
            (refs::ntl::LLL_FP, refs::ntl::G_LLL_FP, "LLL_FP", "G_LLL_FP")
        };
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_fplll)]
        {
            use refs::fplll as F;
            let n1 = reduced_note(ours.basis(), F::lll(&fp, 0.99, 0.51, F::WRAPPER, true, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new("fplll 5.5 lll wrapper, with U", n1, |_| {
                black_box(F::lll(&fp, 0.99, 0.51, F::WRAPPER, true, false));
            }));
            refs_v.push(Ref::new("fplll 5.5 lll wrapper, no U", "computes no U (ours always does)", |_| {
                black_box(F::lll(&fp, 0.99, 0.51, F::WRAPPER, false, false));
            }));
            let n3 = reduced_note(ours.basis(), F::lll(&fp, 0.99, 0.51, F::PROVED, true, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new("fplll 5.5 lll proved, with U", n3, |_| {
                black_box(F::lll(&fp, 0.99, 0.51, F::PROVED, true, false));
            }));
            let n4 = reduced_note(ours.basis(), F::lll(&fp, 0.99, 0.51, F::HEURISTIC, true, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new("fplll 5.5 lll heuristic, with U", format!("no proof of the result; {n4}"), |_| {
                black_box(F::lll(&fp, 0.99, 0.51, F::HEURISTIC, true, false));
            }));
        }
        #[cfg(has_flint)]
        {
            use refs::flint as L;
            let n1 = reduced_note(ours.basis(), L::lll(&fl, 0.99, 0.51, true, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new("FLINT 3.6 fmpz_lll, with U", n1, |_| {
                black_box(L::lll(&fl, 0.99, 0.51, true, false));
            }));
            refs_v.push(Ref::new("FLINT 3.6 fmpz_lll, no U", "computes no U", |_| {
                black_box(L::lll(&fl, 0.99, 0.51, false, false));
            }));
        }
        #[cfg(has_ntl)]
        {
            use refs::ntl as N;
            let why = if xd { "LLL_FP stops on entries this long; " } else { "" };
            let n1 = reduced_note(ours.basis(), N::lll(&nt, k1, 0.99, 0, true, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new(format!("NTL 11.6 {s1}, with U"), format!("{why}δ=0.99, NTL's own η; no proof; {n1}"), |_| {
                black_box(N::lll(&nt, k1, 0.99, 0, true, false));
            }));
            let n2 = reduced_note(ours.basis(), N::lll(&nt, k2, 0.99, 0, true, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new(format!("NTL 11.6 {s2}, with U"), format!("{why}Givens; no proof; {n2}"), |_| {
                black_box(N::lll(&nt, k2, 0.99, 0, true, false));
            }));
        }
        #[cfg(has_pari)]
        {
            use refs::pari as P;
            let n1 = reduced_note(ours.basis(), P::qflll(&pa, 0, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new("PARI 2.19 qflll (FLATTER), T and x·T", n1, |_| {
                black_box(P::qflll(&pa, 0, false));
            }));
            let n2 = reduced_note(ours.basis(), P::qflll(&pa, 1, true), r, c, 0.99, 0.51, check);
            refs_v.push(Ref::new("PARI 2.19 qflll(,1) (no FLATTER), T and x·T", n2, |_| {
                black_box(P::qflll(&pa, 1, false));
            }));
        }
        ctx.row(
            "lll",
            "lll_fp",
            &format!("{name}; δ=0.99 η=0.51; with U"),
            Side::new("ours", |_| {
                black_box(lll_fp(&b, &o));
            }),
            refs_v,
            "",
        );
    }
}

/// The exact integral `lll` against NTL's exact LLL, fplll's proved L² and FLINT's.
pub fn lll_exact(ctx: &Ctx) {
    let o = LllOptions::new();
    let mut rng = Rng(77);
    let mut cases: Vec<(String, Matrix<Number>)> = Vec::new();
    for &(n, b) in if quick() { &[(10usize, 100u64)][..] } else { &[(10, 100), (20, 200), (30, 300), (40, 400)][..] } {
        cases.push((format!("knapsack n={n} bits={b}"), knapsack(&mut rng, n, b)));
    }
    if !quick() {
        for &(n, b) in &[(10usize, 30u64), (20, 10), (30, 10)] {
            cases.push((format!("uniform n={n} bits={b}"), uniform(&mut rng, n, n, b)));
        }
        cases.push(("NTRU-like n=20 q=10b".into(), ntru(&mut rng, 10, 10)));
        cases.push(("NTRU-like n=40 q=12b".into(), ntru(&mut rng, 20, 12)));
        cases.push(("Goldstein-Mayer n=20 bits=200".into(), goldstein_mayer(&mut rng, 20, 200)));
    }
    for (name, b) in cases {
        if !ctx.wants("lll", &name) {
            continue;
        }
        let (r, c) = b.shape();
        let t = text(&b);
        let ours = a_lll(&b, &o);
        #[cfg(has_ntl)]
        let nt = refs::ntl::m(&t, r, c);
        #[cfg(has_fplll)]
        let fp = refs::fplll::m(&t, r, c);
        #[cfg(has_flint)]
        let fl = refs::flint::z(&t, r, c);
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_ntl)]
        {
            use refs::ntl as N;
            let n1 = reduced_note(ours.basis(), N::lll(&nt, N::LLL_EXACT, 0.99, 0, true, true), r, c, 0.99, 0.5, true);
            refs_v.push(Ref::new("NTL 11.6 LLL (exact, a/b=99/100), with U", format!("same algorithm (integral LLL); {n1}"), |_| {
                black_box(N::lll(&nt, N::LLL_EXACT, 0.99, 0, true, false));
            }));
        }
        #[cfg(has_fplll)]
        {
            use refs::fplll as F;
            let n1 = reduced_note(ours.basis(), F::lll(&fp, 0.99, 0.51, F::PROVED, true, true), r, c, 0.99, 0.51, true);
            refs_v.push(Ref::new("fplll 5.5 lll proved, with U", format!("η=0.51 (ours 1/2); {n1}"), |_| {
                black_box(F::lll(&fp, 0.99, 0.51, F::PROVED, true, false));
            }));
        }
        #[cfg(has_flint)]
        {
            use refs::flint as L;
            let n1 = reduced_note(ours.basis(), L::lll(&fl, 0.99, 0.51, true, true), r, c, 0.99, 0.51, true);
            refs_v.push(Ref::new("FLINT 3.6 fmpz_lll, with U", format!("η=0.51 (ours 1/2); {n1}"), |_| {
                black_box(L::lll(&fl, 0.99, 0.51, true, false));
            }));
        }
        ctx.row(
            "lll",
            "lll",
            &format!("{name}; δ=0.99, exact to η=1/2; with U"),
            Side::new("ours", |_| {
                black_box(a_lll(&b, &o));
            }),
            refs_v,
            "",
        );
    }
}

/// `deep_lll` against NTL's LLL_FP and G_LLL_FP with deep insertions.
pub fn deep(ctx: &Ctx) {
    let mut rng = Rng(91);
    let sizes: &[(usize, u64)] = if quick() { &[(16, 160)] } else { &[(16, 160), (24, 240), (32, 320), (40, 400)] };
    for &(n, bits) in sizes {
        let b = knapsack(&mut rng, n, bits);
        let (r, c) = b.shape();
        let t = text(&b);
        #[cfg(has_ntl)]
        let nt = refs::ntl::m(&t, r, c);
        for depth in [None, Some(5usize)] {
            let o = DeepLllOptions::new().with_depth(depth);
            let name = format!("knapsack n={n} bits={bits}; δ=0.99 η=0.51; depth {}", depth.map_or("none (anywhere)".to_string(), |d| d.to_string()));
            if !ctx.wants("deep_lll", &name) {
                continue;
            }
            let ours = deep_lll(&b, &o);
            let ntl_deep = depth.unwrap_or(n);
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_ntl)]
            {
                use refs::ntl as N;
                let n1 = reduced_note(ours.basis(), N::lll(&nt, N::LLL_FP, 0.99, ntl_deep, true, true), r, c, 0.99, 0.51, true);
                refs_v.push(Ref::new(format!("NTL 11.6 LLL_FP deep={ntl_deep}, with U"), format!("no proof; NTL's depth is the first places only; {n1}"), |_| {
                    black_box(N::lll(&nt, N::LLL_FP, 0.99, ntl_deep, true, false));
                }));
            }
            ctx.row(
                "lll",
                "deep_lll",
                &name,
                Side::new("ours", |_| {
                    black_box(deep_lll(&b, &o));
                }),
                refs_v,
                "",
            );
        }
    }
}

/// `is_lll_reduced` on reduced bases against FLINT's `fmpz_lll_is_reduced`.
pub fn is_reduced(ctx: &Ctx) {
    let o = LllOptions::new();
    let mut rng = Rng(5);
    let mut cases: Vec<(String, Matrix<Number>)> = vec![("knapsack n=40 bits=400".into(), knapsack(&mut rng, 40, 400))];
    if !quick() {
        cases.push(("knapsack n=100 bits=100".into(), knapsack(&mut rng, 100, 100)));
        cases.push(("uniform n=50 bits=100".into(), uniform(&mut rng, 50, 50, 100)));
        cases.push(("Goldstein-Mayer n=80 bits=800".into(), goldstein_mayer(&mut rng, 80, 800)));
        cases.push(("q-ary n=80 k=40 q=20b".into(), qary(&mut rng, 80, 40, 20)));
    }
    for (name, b0) in cases {
        let b = lll_fp(&b0, &o).basis().clone();
        let (r, c) = b.shape();
        let t = text(&b);
        #[cfg(has_flint)]
        let fl = refs::flint::z(&t, r, c);
        let mut refs_v: Vec<Ref> = Vec::new();
        let prec = 2 * r + 64;
        #[cfg(has_flint)]
        {
            let agree = refs::flint::lll_is_reduced(&fl, 0.99, 0.51, prec) == is_lll_reduced(&b, 0.99, 0.51);
            refs_v.push(Ref::new(
                format!("FLINT 3.6 fmpz_lll_is_reduced (prec {prec})"),
                format!("d, then mpfr, then exact; answers agree: {agree}"),
                |_| {
                    black_box(refs::flint::lll_is_reduced(&fl, 0.99, 0.51, prec));
                },
            ));
        }
        ctx.row(
            "lll",
            "is_lll_reduced",
            &format!("lll_fp output of {name}; δ=0.99 η=0.51"),
            Side::new("ours", |_| {
                black_box(is_lll_reduced(&b, 0.99, 0.51));
            }),
            refs_v,
            "",
        );
    }
}

/// `bkz` against fplll's BKZ 1.0 and 2.0 and NTL's BKZ_FP.
pub fn bkz(ctx: &Ctx) {
    let mut rng = Rng(31);
    let mut cases: Vec<(String, Matrix<Number>, Vec<usize>)> = Vec::new();
    if quick() {
        cases.push(("knapsack n=30 bits=300".into(), knapsack(&mut rng, 30, 300), vec![10]));
    } else {
        cases.push(("knapsack n=40 bits=400".into(), knapsack(&mut rng, 40, 400), vec![10, 20, 30]));
        cases.push(("knapsack n=60 bits=600".into(), knapsack(&mut rng, 60, 600), vec![10, 20]));
        cases.push(("knapsack n=80 bits=800".into(), knapsack(&mut rng, 80, 800), vec![10, 20]));
        cases.push(("q-ary n=60 k=30 q=20b".into(), qary(&mut rng, 60, 30, 20), vec![10, 20, 30]));
        cases.push(("q-ary n=80 k=40 q=20b".into(), qary(&mut rng, 80, 40, 20), vec![10, 20]));
        cases.push(("NTRU-like n=60 q=12b".into(), ntru(&mut rng, 30, 12), vec![10, 20]));
        cases.push(("Goldstein-Mayer n=60 bits=600".into(), goldstein_mayer(&mut rng, 60, 600), vec![10, 20, 30]));
        cases.push(("uniform n=60 bits=20".into(), uniform(&mut rng, 60, 60, 20), vec![10, 20]));
    }
    for (name, b, betas) in cases {
        let (r, c) = b.shape();
        let t = text(&b);
        #[cfg(has_fplll)]
        let fp = refs::fplll::m(&t, r, c);
        #[cfg(has_ntl)]
        let nt = refs::ntl::m(&t, r, c);
        for beta in betas {
            let params = format!("{name}; β={beta}; δ=0.99; with U");
            if !ctx.wants("bkz", &params) {
                continue;
            }
            let o = BkzOptions::new(beta);
            let ours = a_bkz(&b, &o);
            #[cfg(has_ntl)]
            let xd = refs::ntl::needs_xd(max_bits(&b));
            #[cfg(has_ntl)]
            let (k, s1, why) = if xd {
                (refs::ntl::BKZ_XD, "BKZ_XD", "BKZ_FP stops on entries this long; ")
            } else {
                (refs::ntl::BKZ_FP, "BKZ_FP", "")
            };
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_fplll)]
            {
                use refs::fplll as F;
                let n1 = reduced_note(ours.basis(), F::bkz(&fp, beta, 0.99, 1, true, true), r, c, 0.99, 0.51, true);
                refs_v.push(Ref::new("fplll 5.5 BKZ 1.0 (LLL first), with U", format!("blocks to δ·|b*k|² ≤ λ1(block), ours exact; {n1}"), |_| {
                    black_box(F::bkz(&fp, beta, 0.99, 1, true, false));
                }));
                let n2 = reduced_note(ours.basis(), F::bkz(&fp, beta, 0.99, 2, true, true), r, c, 0.99, 0.51, true);
                refs_v.push(Ref::new("fplll 5.5 BKZ 2.0 (default.json, auto-abort, GH bound), with U", format!("pruned and aborted early: a weaker reduction; {n2}"), |_| {
                    black_box(F::bkz(&fp, beta, 0.99, 2, true, false));
                }));
            }
            #[cfg(has_ntl)]
            {
                use refs::ntl as N;
                let n1 = reduced_note(ours.basis(), N::bkz(&nt, k, beta, 0.99, true, true), r, c, 0.99, 0.51, true);
                refs_v.push(Ref::new(format!("NTL 11.6 {s1}, with U"), format!("{why}no pruning, no proof; {n1}"), |_| {
                    black_box(N::bkz(&nt, k, beta, 0.99, true, false));
                }));
            }
            ctx.row(
                "bkz",
                "bkz",
                &params,
                Side::new("ours", |_| {
                    black_box(a_bkz(&b, &o));
                }),
                refs_v,
                "",
            );
        }
    }
}

/// shortest_vector, closest_vector and short_vectors against fplll and PARI.
pub fn enumeration(ctx: &Ctx) {
    let mut rng = Rng(41);
    let mut svp: Vec<(String, Matrix<Number>)> = Vec::new();
    let gm_dims: &[usize] = if quick() { &[20] } else { &[20, 30, 36, 40, 44] };
    for &n in gm_dims {
        svp.push((format!("Goldstein-Mayer n={n} bits={}", 10 * n), goldstein_mayer(&mut rng, n, 10 * n as u64)));
    }
    if !quick() {
        svp.push(("knapsack n=30 bits=300".into(), knapsack(&mut rng, 30, 300)));
        svp.push(("uniform n=30 bits=20".into(), uniform(&mut rng, 30, 30, 20)));
        svp.push(("q-ary n=40 k=20 q=12b".into(), qary(&mut rng, 40, 20, 12)));
    }
    for (name, b) in &svp {
        if !ctx.wants("shortest_vector", name) {
            continue;
        }
        let (r, c) = b.shape();
        let t = text(b);
        let ours = shortest_vector(b).unwrap().norm2;
        #[cfg(has_fplll)]
        let fp = refs::fplll::m(&t, r, c);
        #[cfg(has_pari)]
        let pa = refs::pari::z(&t, r, c, true);
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_fplll)]
        {
            use refs::fplll as F;
            let a1 = F::svp(&fp, 2, true).unwrap_or_default();
            refs_v.push(Ref::new("fplll 5.5 LLL + shortest_vector (SVPM_PROVED)", format!("norm² equal: {}", a1 == ours.to_string()), |_| {
                black_box(F::svp(&fp, 2, false));
            }));
            let a2 = F::svp(&fp, 0, true).unwrap_or_default();
            refs_v.push(Ref::new("fplll 5.5 LLL + shortest_vector (SVPM_FAST)", format!("not proved; norm² equal: {}", a2 == ours.to_string()), |_| {
                black_box(F::svp(&fp, 0, false));
            }));
        }
        #[cfg(has_pari)]
        {
            use refs::pari as P;
            let a1 = P::svp(&pa, 0, true).unwrap_or_default();
            refs_v.push(Ref::new("PARI 2.19 Gram + qfminim (flag 0)", format!("Fincke-Pohst on the Gram matrix; min equal: {}", a1 == ours.to_string()), |_| {
                black_box(P::svp(&pa, 0, false));
            }));
        }
        ctx.row(
            "enum",
            "shortest_vector",
            name,
            Side::new("ours", |_| {
                black_box(shortest_vector(b));
            }),
            refs_v,
            "includes the LLL reduction on both sides",
        );
    }

    // Closest vectors: square bases of full rank, targets of the size of the entries.
    let mut cvp: Vec<(String, Matrix<Number>, Vec<Number>)> = Vec::new();
    let dims: &[usize] = if quick() { &[20] } else { &[20, 30, 36, 40] };
    for &n in dims {
        let b = goldstein_mayer(&mut rng, n, 10 * n as u64);
        let tgt: Vec<Number> = (0..n).map(|j| rng.unsigned(if j == 0 { 10 * n as u64 } else { 8 })).collect();
        cvp.push((format!("Goldstein-Mayer n={n} bits={}; random target", 10 * n), b, tgt));
    }
    if !quick() {
        for &n in &[20usize, 30] {
            let b = uniform(&mut rng, n, n, 20);
            let tgt: Vec<Number> = (0..n).map(|_| rng.number(22)).collect();
            cvp.push((format!("uniform n={n} bits=20; random target of 22 bits"), b, tgt));
        }
    }
    for (name, b, tgt) in &cvp {
        if !ctx.wants("closest_vector", name) {
            continue;
        }
        let (r, c) = b.shape();
        let t = text(b);
        let tt: String = tgt.iter().map(|x| format!("{x} ")).collect();
        let ours = closest_vector(b, tgt).unwrap().norm2;
        #[cfg(has_fplll)]
        let (fp, ft) = (refs::fplll::m(&t, r, c), refs::fplll::m(&tt, 1, c));
        #[cfg(has_pari)]
        let (pa, pt) = (refs::pari::z(&t, r, c, true), refs::pari::z(&tt, 1, c, true));
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_fplll)]
        {
            use refs::fplll as F;
            let a1 = F::cvp(&fp, &ft, 0, true).unwrap_or_default();
            let rel = if a1 == ours.to_string() { "equal".to_string() } else { format!("ref {a1} vs ours {ours}") };
            refs_v.push(Ref::new("fplll 5.5 LLL + closest_vector (CVPM_FAST)", format!("not proved; distance² {rel}"), |_| {
                black_box(F::cvp(&fp, &ft, 0, false));
            }));
        }
        #[cfg(has_pari)]
        {
            use refs::pari as P;
            let a1 = P::cvp(&pa, &pt, true).unwrap_or_default();
            let o: f64 = ours.to_string().parse().unwrap_or(f64::NAN);
            let close = a1.parse::<f64>().is_ok_and(|x| (x - o).abs() <= 1e-9 * o.abs().max(1.0));
            let rel = if a1 == ours.to_string() {
                "equal".to_string()
            } else if close {
                "equal (PARI's in floating point)".to_string()
            } else {
                format!("ref {a1} vs ours {ours}: PARI's floating-point search is wrong here")
            };
            refs_v.push(Ref::new("PARI 2.19 solve + Gram + qfcvp", format!("on the coordinates (rational target); distance² {rel}"), |_| {
                black_box(P::cvp(&pa, &pt, false));
            }));
        }
        ctx.row(
            "enum",
            "closest_vector",
            name,
            Side::new("ours", |_| {
                black_box(closest_vector(b, tgt));
            }),
            refs_v,
            "",
        );
    }

    // Short vectors: every vector below a multiple of the minimum.
    let mut sv: Vec<(String, Matrix<Number>, u32)> = Vec::new();
    if quick() {
        sv.push(("uniform n=12 bits=6".into(), uniform(&mut rng, 12, 12, 6), 2));
    } else {
        sv.push(("uniform n=12 bits=6".into(), uniform(&mut rng, 12, 12, 6), 3));
        sv.push(("uniform n=16 bits=6".into(), uniform(&mut rng, 16, 16, 6), 2));
        sv.push(("uniform n=20 bits=8".into(), uniform(&mut rng, 20, 20, 8), 2));
        sv.push(("Goldstein-Mayer n=24 bits=240".into(), goldstein_mayer(&mut rng, 24, 240), 2));
        sv.push(("Goldstein-Mayer n=30 bits=300".into(), goldstein_mayer(&mut rng, 30, 300), 2));
    }
    for (name, b, mult) in &sv {
        let min = shortest_vector(b).unwrap().norm2;
        let bound = &min * &Number::from(*mult);
        let params = format!("{name}; bound {mult}·λ1²");
        if !ctx.wants("short_vectors", &params) {
            continue;
        }
        let (r, c) = b.shape();
        let t = text(b);
        let count = short_vectors(b, &bound).len();
        #[cfg(has_pari)]
        let pa = refs::pari::z(&t, r, c, true);
        let bs = bound.to_string();
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_pari)]
        {
            let a1 = refs::pari::short(&pa, &bs, true).unwrap_or_default();
            refs_v.push(Ref::new(
                "PARI 2.19 Gram + qfminim(G, B)",
                format!("unsorted, vectors in coordinates; count equal: {} ({count} pairs, PARI N = {a1})", a1 == (2 * count).to_string() || a1 == count.to_string()),
                |_| {
                    black_box(refs::pari::short(&pa, &bs, false));
                },
            ));
        }
        ctx.row(
            "enum",
            "short_vectors",
            &params,
            Side::new("ours", |_| {
                black_box(short_vectors(b, &bound));
            }),
            refs_v,
            "",
        );
    }
}

/// One run of ours (lll_fp) and of fplll's wrapper on each case, to plan the sizes.
pub fn probe(ctx: &Ctx) {
    let o = LllOptions::new();
    for (name, b) in lll_cases() {
        if !ctx.wants("probe", &name) {
            continue;
        }
        let (r, c) = b.shape();
        let t0 = std::time::Instant::now();
        black_box(lll_fp(&b, &o));
        let ours = t0.elapsed().as_secs_f64();
        #[cfg(has_fplll)]
        let fp = refs::fplll::m(&text(&b), r, c);
        let t1 = std::time::Instant::now();
        #[cfg(has_fplll)]
        black_box(refs::fplll::lll(&fp, 0.99, 0.51, refs::fplll::WRAPPER, true, false));
        let theirs = t1.elapsed().as_secs_f64();
        eprintln!("{name}: ours {ours:.3} s, fplll wrapper with U {theirs:.3} s");
    }
}
