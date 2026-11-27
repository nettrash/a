//! Normal forms (Hermite, Smith) and exact linear algebra on integers and rationals: the
//! determinant (Bareiss, multimodular, Dixon's divisor), solutions and inverses (Dixon's
//! lifting, proofs of singularity), products (Strassen–Winograd on numbers).
#![allow(unused_variables, unused_mut)]
use crate::inputs::*;
#[allow(unused_imports)]
use crate::refs;
use crate::timing::Side;
use crate::{Ctx, Ref};
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::math::rational::Rational;
use std::hint::black_box;

fn quick() -> bool {
    std::env::var("LATBENCH_QUICK").is_ok()
}

fn sorted_divisors(v: &[Number]) -> Vec<String> {
    let mut s: Vec<Number> = v.to_vec();
    s.sort();
    s.iter().map(|x| x.to_string()).collect()
}

pub fn normal(ctx: &Ctx) {
    let mut rng = Rng(101);
    let mut cases: Vec<(String, Matrix<Number>)> = Vec::new();
    let sq: &[usize] = if quick() { &[10] } else { &[10, 20, 50, 100, 200] };
    for &n in sq {
        cases.push((format!("square n={n} bits=10"), uniform(&mut rng, n, n, 10)));
    }
    if !quick() {
        for &n in &[10usize, 20, 50] {
            cases.push((format!("square n={n} bits=100"), uniform(&mut rng, n, n, 100)));
        }
        cases.push(("30×30 rank 20 (product of 10-bit factors)".into(), low_rank(&mut rng, 30, 30, 20, 10)));
        cases.push(("60×60 rank 40 (product of 8-bit factors)".into(), low_rank(&mut rng, 60, 60, 40, 8)));
        cases.push(("wide 20×40 bits=10".into(), uniform(&mut rng, 20, 40, 10)));
        cases.push(("tall 40×20 bits=10".into(), uniform(&mut rng, 40, 20, 10)));
        cases.push(("tall 500×5 bits=10".into(), uniform(&mut rng, 500, 5, 10)));
    }
    for (name, a) in &cases {
        let (r, c) = a.shape();
        let t = text(a);
        let square_full = r == c && !a.det().is_zero();
        let big = r.max(c);
        #[cfg(has_flint)]
        let fl = refs::flint::z(&t, r, c);
        #[cfg(has_pari)]
        let pa_t = refs::pari::z(&t, r, c, true);
        #[cfg(has_pari)]
        let pa = refs::pari::z(&t, r, c, false);
        #[cfg(has_ntl)]
        let nt = refs::ntl::m(&t, r, c);

        // hermite_form
        if ctx.wants("hermite_form", name) {
            let h = a.hermite_form();
            let ht = text(&h);
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_flint)]
            {
                let same = refs::flint::hnf(&fl, false, true).unwrap_or_default() == ht;
                refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_hnf", format!("H equal: {same}"), |_| {
                    black_box(refs::flint::hnf(&fl, false, false));
                }));
            }
            #[cfg(has_pari)]
            refs_v.push(Ref::new("PARI 2.19 mathnf (columns of Aᵀ)", "PARI's convention (columns, triangular from the other end)", |_| {
                black_box(refs::pari::mathnf(&pa_t, 0, false));
            }));
            #[cfg(has_ntl)]
            if square_full {
                let d = a.det().abs().to_string();
                let same = refs::ntl::hnf(&nt, true).unwrap_or_default() == d;
                refs_v.push(Ref::new("NTL 11.6 determinant + HNF(W, A, D)", format!("square nonsingular only; diagonal product = |det|: {same}"), |_| {
                    black_box(refs::ntl::hnf(&nt, false));
                }));
            }
            ctx.row("normal", "hermite_form", name, Side::new("ours", |_| {
                black_box(a.hermite_form());
            }), refs_v, "");
        }
        // hermite_normal_form (with U)
        if big <= 100 && ctx.wants("hermite_normal_form", name) {
            let (h, u) = a.hermite_normal_form();
            let (ht, ut) = (text(&h), text(&u));
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_flint)]
            {
                let theirs = refs::flint::hnf(&fl, true, true).unwrap_or_default();
                let (fh, fu) = theirs.split_once("| ").unwrap_or(("", ""));
                let what = if fh != ht {
                    "H equal: false".to_string()
                } else if fu == ut {
                    "H and U equal".to_string()
                } else {
                    "H equal, U another valid one (not unique for dependent rows)".to_string()
                };
                refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_hnf_transform", what, |_| {
                    black_box(refs::flint::hnf(&fl, true, false));
                }));
            }
            #[cfg(has_pari)]
            {
                refs_v.push(Ref::new("PARI 2.19 mathnf(, 1) (columns of Aᵀ)", "[H, U]; U not unique for dependent rows", |_| {
                    black_box(refs::pari::mathnf(&pa_t, 1, false));
                }));
                refs_v.push(Ref::new("PARI 2.19 mathnf(, 4) (columns of Aᵀ)", "[H, U] with LLL along the way (hnflll)", |_| {
                    black_box(refs::pari::mathnf(&pa_t, 4, false));
                }));
            }
            ctx.row("normal", "hermite_normal_form", name, Side::new("ours", |_| {
                black_box(a.hermite_normal_form());
            }), refs_v, "");
        }
        // elementary_divisors
        if ctx.wants("elementary_divisors", name) {
            let d = a.elementary_divisors();
            let dt: String = d.iter().map(|x| format!("{x} ")).collect();
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_flint)]
            {
                let same = refs::flint::snf(&fl, false, true).unwrap_or_default() == dt;
                refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_snf", format!("divisors equal: {same}"), |_| {
                    black_box(refs::flint::snf(&fl, false, false));
                }));
            }
            #[cfg(has_pari)]
            {
                let p = refs::pari::matsnf(&pa, 0, true).unwrap_or_default();
                let pv: Vec<Number> = p.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|s| s.trim().parse().ok()).collect();
                let k = r.min(c);
                let mut pv2 = pv.clone();
                pv2.retain(|x| !x.is_zero());
                let mut dd = d.clone();
                dd.retain(|x| !x.is_zero());
                let same = sorted_divisors(&pv2) == sorted_divisors(&dd);
                refs_v.push(Ref::new("PARI 2.19 matsnf", format!("nonzero divisors equal: {same}"), |_| {
                    black_box(refs::pari::matsnf(&pa, 0, false));
                }));
            }
            ctx.row("normal", "elementary_divisors", name, Side::new("ours", |_| {
                black_box(a.elementary_divisors());
            }), refs_v, "");
        }
        // smith_normal_form (with U, V)
        if big <= 50 && ctx.wants("smith_normal_form", name) {
            let (s, _, _) = a.smith_normal_form();
            let k = r.min(c);
            let dt: String = (0..k).map(|i| format!("{} ", s[(i, i)])).collect();
            let mut refs_v: Vec<Ref> = Vec::new();
            #[allow(unused_mut)]
            let mut note = String::new();
            #[cfg(has_flint)]
            if refs::flint::has_snf_transform() {
                let same = refs::flint::snf(&fl, true, true).unwrap_or_default() == dt;
                refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_snf_transform", format!("S equal: {same}; U, V not unique"), |_| {
                    black_box(refs::flint::snf(&fl, true, false));
                }));
            } else {
                note = format!("no FLINT row: FLINT {} has no fmpz_mat_snf_transform (3.5 and later)", refs::flint::version());
            }
            #[cfg(has_pari)]
            refs_v.push(Ref::new("PARI 2.19 matsnf(, 1)", "[U, V, D]", |_| {
                black_box(refs::pari::matsnf(&pa, 1, false));
            }));
            ctx.row("normal", "smith_normal_form", name, Side::new("ours", |_| {
                black_box(a.smith_normal_form());
            }), refs_v, &note);
        }
    }
}

struct Kind {
    name: &'static str,
    make: fn(&mut Rng, usize, usize) -> Matrix<Number>,
}

fn small_ints(rng: &mut Rng, r: usize, c: usize) -> Matrix<Number> {
    small(rng, r, c, -99, 99)
}
fn ints64(rng: &mut Rng, r: usize, c: usize) -> Matrix<Number> {
    uniform(rng, r, c, 64)
}
fn ints1000(rng: &mut Rng, r: usize, c: usize) -> Matrix<Number> {
    uniform(rng, r, c, 1000)
}

const SMALL: Kind = Kind { name: "integers in −99..=99", make: small_ints };
const W64: Kind = Kind { name: "64-bit integers", make: ints64 };
const B1000: Kind = Kind { name: "1000-bit integers", make: ints1000 };

pub fn exact(ctx: &Ctx) {
    let mut rng = Rng(7);
    let q = quick();
    // ---- det ----
    let det_plan: Vec<(&Kind, Vec<usize>)> = if q {
        vec![(&SMALL, vec![16])]
    } else {
        vec![(&SMALL, vec![16, 32, 64, 128, 256]), (&W64, vec![16, 32, 64, 128]), (&B1000, vec![10, 20, 40])]
    };
    for (k, sizes) in &det_plan {
        for &n in sizes {
            let params = format!("n={n}, {}", k.name);
            if !ctx.wants("det", &params) {
                continue;
            }
            let insts: Vec<Matrix<Number>> = (0..if n <= 32 { 3 } else { 1 }).map(|_| (k.make)(&mut rng, n, n)).collect();
            det_row(ctx, &params, &insts);
        }
    }
    if !q {
        // A singular matrix: the last row the sum of the first two.
        for (k, n) in [(&W64, 40usize), (&SMALL, 64)] {
            let mut a = (k.make)(&mut rng, n, n);
            for j in 0..n {
                let s = &a[(0, j)] + &a[(1, j)];
                a[(n - 1, j)] = s;
            }
            det_row(ctx, &format!("n={n}, {}, singular (row n = row 1 + row 2)", k.name), &[a]);
        }
    }
    // ---- solve, one column ----
    let solve_plan: Vec<(&Kind, Vec<usize>)> = if q {
        vec![(&SMALL, vec![16])]
    } else {
        vec![(&SMALL, vec![16, 32, 64, 128, 256]), (&W64, vec![16, 32, 64, 128]), (&B1000, vec![10, 20, 40])]
    };
    for (k, sizes) in &solve_plan {
        for &n in sizes {
            let params = format!("n={n}, one column, {}", k.name);
            if !ctx.wants("solve", &params) {
                continue;
            }
            let insts: Vec<(Matrix<Number>, Matrix<Number>)> = (0..if n <= 32 { 3 } else { 1 }).map(|_| ((k.make)(&mut rng, n, n), (k.make)(&mut rng, n, 1))).collect();
            solve_row(ctx, &params, &insts);
        }
    }
    if !q {
        // More right-hand sides than an eighth of the rows: the elimination path.
        let a = small_ints(&mut rng, 64, 64);
        let b = small_ints(&mut rng, 64, 16);
        solve_row(ctx, "n=64, 16 columns, integers in −99..=99", &[(a, b)]);
        // Singular systems: proved singular.
        for (k, n) in [(&SMALL, 40usize), (&W64, 40)] {
            let mut a = (k.make)(&mut rng, n, n);
            for j in 0..n {
                let s = &a[(0, j)] + &a[(1, j)];
                a[(n - 1, j)] = s;
            }
            let b = (k.make)(&mut rng, n, 1);
            solve_row(ctx, &format!("n={n}, one column, {}, singular", k.name), &[(a, b)]);
        }
    }
    // ---- inverse_rational and inverse ----
    let inv_plan: Vec<(&Kind, Vec<usize>)> = if q { vec![(&SMALL, vec![16])] } else { vec![(&SMALL, vec![16, 32, 64, 128]), (&W64, vec![16, 32])] };
    for (k, sizes) in &inv_plan {
        for &n in sizes {
            let insts: Vec<Matrix<Number>> = (0..if n <= 32 { 3 } else { 1 }).map(|_| (k.make)(&mut rng, n, n)).collect();
            let params = format!("n={n}, {}", k.name);
            if ctx.wants("inverse_rational", &params) {
                inv_rational_row(ctx, &params, &insts);
            }
            if ctx.wants("inverse", &params) && n <= 64 {
                inv_row(ctx, &params, &insts);
            }
        }
    }
    // ---- products ----
    let mul_plan: Vec<(&Kind, Vec<usize>)> = if q { vec![(&W64, vec![32])] } else { vec![(&SMALL, vec![64, 128, 256]), (&W64, vec![32, 64, 128, 256]), (&B1000, vec![16, 32, 64])] };
    for (k, sizes) in &mul_plan {
        for &n in sizes {
            let params = format!("n={n}, {}", k.name);
            if !ctx.wants("mul", &params) {
                continue;
            }
            let x = (k.make)(&mut rng, n, n);
            let y = (k.make)(&mut rng, n, n);
            mul_row(ctx, &params, &x, &y);
        }
    }
    // ---- matrices of rationals ----
    let qsizes: &[usize] = if q { &[16] } else { &[16, 32, 64] };
    for &n in qsizes {
        let a = fractions(&mut rng, n, n);
        let b = fractions(&mut rng, n, 1);
        let params = format!("n={n}, fractions p/q, |p| ≤ 99, 1 ≤ q ≤ 99");
        qrows(ctx, &params, &a, &b);
    }
}

fn det_row(ctx: &Ctx, params: &str, insts: &[Matrix<Number>]) {
    let n = insts[0].rows();
    let texts: Vec<String> = insts.iter().map(text).collect();
    let want: Vec<String> = insts.iter().map(|a| a.det().to_string()).collect();
    #[cfg(has_flint)]
    let fl: Vec<_> = texts.iter().map(|t| refs::flint::z(t, n, n)).collect();
    #[cfg(has_pari)]
    let pa: Vec<_> = texts.iter().map(|t| refs::pari::z(t, n, n, false)).collect();
    let mut refs_v: Vec<Ref> = Vec::new();
    #[cfg(has_flint)]
    {
        let same = fl.iter().zip(&want).all(|(h, w)| refs::flint::det(h, true).as_deref() == Some(w.as_str()));
        refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_det", format!("equal: {same}"), |k| {
            black_box(refs::flint::det(&fl[k % fl.len()], false));
        }));
        refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_det_bareiss", "", |k| {
            black_box(refs::flint::det_bareiss(&fl[k % fl.len()], false));
        }));
    }
    #[cfg(has_pari)]
    {
        let same = pa.iter().zip(&want).all(|(h, w)| refs::pari::det(h, true).as_deref() == Some(w.as_str()));
        refs_v.push(Ref::new("PARI 2.19 det", format!("equal: {same}"), |k| {
            black_box(refs::pari::det(&pa[k % pa.len()], false));
        }));
    }
    ctx.row("exact", "Matrix<Number>::det", params, Side::new("ours", |k| {
        black_box(insts[k % insts.len()].det());
    }), refs_v, &format!("{} instances alternated", insts.len()));
}

fn solve_row(ctx: &Ctx, params: &str, insts: &[(Matrix<Number>, Matrix<Number>)]) {
    let n = insts[0].0.rows();
    let m = insts[0].1.cols();
    let want: Vec<String> = insts.iter().map(|(a, b)| a.solve(b).map_or(String::new(), |x| text(&x))).collect();
    #[cfg(has_flint)]
    let fl: Vec<_> = insts.iter().map(|(a, b)| (refs::flint::z(&text(a), n, n), refs::flint::z(&text(b), n, m))).collect();
    #[cfg(has_pari)]
    let pa: Vec<_> = insts.iter().map(|(a, b)| (refs::pari::z(&text(a), n, n, false), refs::pari::z(&text(b), n, m, false))).collect();
    let mut refs_v: Vec<Ref> = Vec::new();
    #[cfg(has_flint)]
    {
        let same = fl.iter().zip(&want).all(|((a, b), w)| refs::flint::solve(a, b, true).as_deref() == Some(w.as_str()));
        refs_v.push(Ref::new("FLINT 3.6 fmpq_mat_solve_fmpz_mat", format!("X equal: {same}"), |k| {
            let (a, b) = &fl[k % fl.len()];
            black_box(refs::flint::solve(a, b, false));
        }));
        refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_solve_fflu + division", "Bareiss", |k| {
            let (a, b) = &fl[k % fl.len()];
            black_box(refs::flint::solve_fflu(a, b, false));
        }));
    }
    #[cfg(has_pari)]
    refs_v.push(Ref::new("PARI 2.19 matsolve (gauss)", "raises on a singular matrix: not timed there", |k| {
        let (a, b) = &pa[k % pa.len()];
        black_box(refs::pari::solve(a, b, false));
    }));
    #[cfg(has_pari)]
    if want[0].is_empty() {
        refs_v.pop();
    }
    ctx.row("exact", "Matrix<Number>::solve", params, Side::new("ours", |k| {
        let (a, b) = &insts[k % insts.len()];
        black_box(a.solve(b));
    }), refs_v, &format!("{} instances alternated", insts.len()));
}

fn inv_rational_row(ctx: &Ctx, params: &str, insts: &[Matrix<Number>]) {
    let n = insts[0].rows();
    let want: Vec<String> = insts.iter().map(|a| a.inverse_rational().map_or(String::new(), |x| text(&x))).collect();
    #[cfg(has_flint)]
    let fl: Vec<_> = insts.iter().map(|a| refs::flint::z(&text(a), n, n)).collect();
    #[cfg(has_pari)]
    let pa: Vec<_> = insts.iter().map(|a| refs::pari::z(&text(a), n, n, false)).collect();
    let mut refs_v: Vec<Ref> = Vec::new();
    #[cfg(has_flint)]
    {
        let same = fl.iter().zip(&want).all(|(a, w)| refs::flint::inv(a, true, true).as_deref() == Some(w.as_str()));
        refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_inv + fmpq_mat_set_fmpz_mat_div_fmpz", format!("equal: {same}"), |k| {
            black_box(refs::flint::inv(&fl[k % fl.len()], true, false));
        }));
    }
    #[cfg(has_pari)]
    refs_v.push(Ref::new("PARI 2.19 RgM_inv", "", |k| {
        black_box(refs::pari::inv(&pa[k % pa.len()], false));
    }));
    ctx.row("exact", "Matrix<Number>::inverse_rational", params, Side::new("ours", |k| {
        black_box(insts[k % insts.len()].inverse_rational());
    }), refs_v, &format!("{} instances alternated", insts.len()));
}

fn inv_row(ctx: &Ctx, params: &str, insts: &[Matrix<Number>]) {
    let n = insts[0].rows();
    #[cfg(has_flint)]
    let fl: Vec<_> = insts.iter().map(|a| refs::flint::z(&text(a), n, n)).collect();
    let mut refs_v: Vec<Ref> = Vec::new();
    #[cfg(has_flint)]
    {
        // FLINT's (B, den) with A·B = den·I; ours (adj A, det A).
        let k = fl.iter().zip(insts).filter(|(h, a)| refs::flint::inv(h, false, true).map(|d| d.trim_start_matches('-').to_string()) == Some(a.det().abs().to_string())).count();
        refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_inv (B, den)", format!("A·B = den·I, den |det| on {k} of {} (else a divisor of it)", fl.len()), |k| {
            black_box(refs::flint::inv(&fl[k % fl.len()], false, false));
        }));
    }
    ctx.row("exact", "Matrix<Number>::inverse", params, Side::new("ours", |k| {
        black_box(insts[k % insts.len()].inverse());
    }), refs_v, &format!("(adj, det); {} instances alternated", insts.len()));
}

fn mul_row(ctx: &Ctx, params: &str, x: &Matrix<Number>, y: &Matrix<Number>) {
    let (n, m) = x.shape();
    let p = y.cols();
    let want = text(&(x * y));
    #[cfg(has_flint)]
    let (fx, fy) = (refs::flint::z(&text(x), n, m), refs::flint::z(&text(y), m, p));
    #[cfg(has_pari)]
    let (px, py) = (refs::pari::z(&text(x), n, m, false), refs::pari::z(&text(y), m, p, false));
    let mut refs_v: Vec<Ref> = Vec::new();
    #[cfg(has_flint)]
    {
        let same = refs::flint::mul(&fx, &fy, true).as_deref() == Some(want.as_str());
        refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_mul", format!("equal: {same}"), |_| {
            black_box(refs::flint::mul(&fx, &fy, false));
        }));
    }
    #[cfg(has_pari)]
    refs_v.push(Ref::new("PARI 2.19 ZM_mul", "", |_| {
        black_box(refs::pari::mul(&px, &py, false));
    }));
    ctx.row("exact", "Matrix<Number> * Matrix<Number>", params, Side::new("ours", |_| {
        black_box(x * y);
    }), refs_v, "");
}

fn qrows(ctx: &Ctx, params: &str, a: &Matrix<Rational>, b: &Matrix<Rational>) {
    let n = a.rows();
    let ta = text(a);
    let tb = text(b);
    #[cfg(has_flint)]
    let (fa, fb) = (refs::flint::q(&ta, n, n), refs::flint::q(&tb, n, 1));
    #[cfg(has_pari)]
    let (pa, pb) = (refs::pari::q(&ta, n, n), refs::pari::q(&tb, n, 1));
    if ctx.wants("Matrix<Rational>::solve", params) {
        let want = a.solve(b).map_or(String::new(), |x| text(&x));
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_flint)]
        {
            let same = refs::flint::qsolve(&fa, &fb, true).as_deref() == Some(want.as_str());
            refs_v.push(Ref::new("FLINT 3.6 fmpq_mat_solve", format!("equal: {same}"), |_| {
                black_box(refs::flint::qsolve(&fa, &fb, false));
            }));
        }
        #[cfg(has_pari)]
        refs_v.push(Ref::new("PARI 2.19 matsolve (gauss)", "", |_| {
            black_box(refs::pari::solve(&pa, &pb, false));
        }));
        ctx.row("exact", "Matrix<Rational>::solve", &format!("{params}, one column"), Side::new("ours", |_| {
            black_box(a.solve(b));
        }), refs_v, "");
    }
    if ctx.wants("Matrix<Rational>::det", params) {
        let want = a.det().to_string();
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_flint)]
        {
            let same = refs::flint::qdet(&fa, true).as_deref() == Some(want.as_str());
            refs_v.push(Ref::new("FLINT 3.6 fmpq_mat_det", format!("equal: {same}"), |_| {
                black_box(refs::flint::qdet(&fa, false));
            }));
        }
        #[cfg(has_pari)]
        refs_v.push(Ref::new("PARI 2.19 det", "", |_| {
            black_box(refs::pari::det(&pa, false));
        }));
        ctx.row("exact", "Matrix<Rational>::det", params, Side::new("ours", |_| {
            black_box(a.det());
        }), refs_v, "");
    }
}
