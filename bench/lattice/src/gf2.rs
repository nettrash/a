//! BitMatrix (GF(2)) against M4RI and NTL's mat_GF2.
#![allow(unused_variables, unused_mut)]
use crate::inputs::*;
#[allow(unused_imports)]
use crate::refs;
use crate::timing::Side;
use crate::{Ctx, Ref};
use a::math::matrix::{BitMatrix, Matrix};
use a::math::numbers::Number;
use std::hint::black_box;

fn random(rng: &mut Rng, r: usize, c: usize) -> BitMatrix {
    let wpr = c.div_ceil(64);
    let mut w = Vec::with_capacity(r * wpr);
    let tail = c % 64;
    for _ in 0..r {
        for k in 0..wpr {
            let mut x = rng.next();
            if k == wpr - 1 && tail != 0 {
                x &= (1u64 << tail) - 1;
            }
            w.push(x);
        }
    }
    BitMatrix::from_words(r, c, w)
}

fn invertible(rng: &mut Rng, n: usize) -> BitMatrix {
    loop {
        let a = random(rng, n, n);
        if a.det() {
            return a;
        }
    }
}

/// Rank `k`: `X·Y` of `n × k` and `k × c` random factors.
fn low_rank(rng: &mut Rng, r: usize, c: usize, k: usize) -> BitMatrix {
    &random(rng, r, k) * &random(rng, k, c)
}

fn digest(a: &BitMatrix) -> u64 {
    fnv(FNV_INIT, a.as_words())
}

pub fn gf2(ctx: &Ctx) {
    let mut rng = Rng(2);
    let q = std::env::var("LATBENCH_QUICK").is_ok();
    // ---- products ----
    let sizes: &[usize] = if q { &[256] } else { &[256, 1024, 4096, 8192, 16384] };
    for &n in sizes {
        let params = format!("n={n}, uniform");
        if !ctx.wants("BitMatrix * BitMatrix", &params) {
            continue;
        }
        let k = if n <= 1024 { 3 } else { 1 };
        let insts: Vec<(BitMatrix, BitMatrix)> = (0..k).map(|_| (random(&mut rng, n, n), random(&mut rng, n, n))).collect();
        #[cfg(has_m4ri)]
        let m4: Vec<_> = insts.iter().map(|(a, b)| (refs::m4ri::m(a.as_words(), n, n), refs::m4ri::m(b.as_words(), n, n))).collect();
        #[cfg(has_ntl)]
        let nt: Vec<_> = if n <= 8192 { insts.iter().map(|(a, b)| (refs::ntl::gf2(a.as_words(), n, n), refs::ntl::gf2(b.as_words(), n, n))).collect() } else { Vec::new() };
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_m4ri)]
        {
            let same = insts.iter().zip(&m4).all(|((a, b), (x, y))| refs::m4ri::op(0, x, Some(y), 0, true) == digest(&(a * b)));
            refs_v.push(Ref::new("M4RI 20260122 mzd_mul (default cutoff 4096)", format!("equal: {same}"), |k| {
                let (x, y) = &m4[k % m4.len()];
                black_box(refs::m4ri::op(0, x, Some(y), 0, false));
            }));
            refs_v.push(Ref::new("M4RI 20260122 mzd_mul_m4rm", "M4RM alone", |k| {
                let (x, y) = &m4[k % m4.len()];
                black_box(refs::m4ri::op(1, x, Some(y), 0, false));
            }));
        }
        #[cfg(has_ntl)]
        if !nt.is_empty() {
            let same = insts.iter().zip(&nt).all(|((a, b), (x, y))| refs::ntl::gf2_op(0, x, Some(y), true) == digest(&(a * b)));
            refs_v.push(Ref::new("NTL 11.6 mat_GF2 mul", format!("equal: {same}"), |k| {
                let (x, y) = &nt[k % nt.len()];
                black_box(refs::ntl::gf2_op(0, x, Some(y), false));
            }));
        }
        ctx.row("gf2", "BitMatrix * BitMatrix", &params, Side::new("ours", |k| {
            let (a, b) = &insts[k % insts.len()];
            black_box(a * b);
        }), refs_v, &format!("{} instances alternated", insts.len()));
    }
    // ---- the Strassen–Winograd crossover ----
    if !q {
        for &n in &[4096usize, 8192] {
            let a = random(&mut rng, n, n);
            let b = random(&mut rng, n, n);
            #[cfg(has_m4ri)]
            let (x, y) = (refs::m4ri::m(a.as_words(), n, n), refs::m4ri::m(b.as_words(), n, n));
            for cutoff in [1024usize, 2048, 4096] {
                let params = format!("n={n}, cutoff {cutoff}");
                if !ctx.wants("BitMatrix::mul_strassen", &params) {
                    continue;
                }
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    refs_v.push(Ref::new(format!("M4RI 20260122 mzd_mul (cutoff {cutoff})"), "the same cutoff", |_| {
                        black_box(refs::m4ri::op(0, &x, Some(&y), cutoff as i32, false));
                    }));
                    refs_v.push(Ref::new("M4RI 20260122 mzd_mul (default cutoff 4096)", "", |_| {
                        black_box(refs::m4ri::op(0, &x, Some(&y), 0, false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::mul_strassen", &params, Side::new("ours", |_| {
                    black_box(a.mul_strassen(&b, cutoff));
                }), refs_v, "");
            }
        }
    }
    // ---- eliminations ----
    let el_sizes: &[usize] = if q { &[256] } else { &[256, 1024, 4096, 8192, 16384] };
    for &n in el_sizes {
        let k = if n <= 1024 { 3 } else { 1 };
        // rank and rref, of uniform (nearly full rank) and of rank n/2.
        for (kind, mk) in [("uniform", 0), ("rank n/2", 1)] {
            let insts: Vec<BitMatrix> = (0..k).map(|_| if mk == 0 { random(&mut rng, n, n) } else { low_rank(&mut rng, n, n, n / 2) }).collect();
            let params = format!("n={n}, {kind}");
            #[cfg(has_m4ri)]
            let m4: Vec<_> = insts.iter().map(|a| refs::m4ri::m(a.as_words(), n, n)).collect();
            #[cfg(has_ntl)]
            let nt: Vec<_> = if n <= 8192 { insts.iter().map(|a| refs::ntl::gf2(a.as_words(), n, n)).collect() } else { Vec::new() };
            if ctx.wants("BitMatrix::rank", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    let same = insts.iter().zip(&m4).all(|(a, x)| refs::m4ri::op(2, x, None, 0, true) == a.rank() as u64);
                    for (arg, what) in [(0, "mzd_echelonize (default: M4RI or PLUQ by density)"), (1, "mzd_echelonize_m4ri"), (2, "mzd_echelonize_pluq")] {
                        let m4 = &m4;
                        refs_v.push(Ref::new(format!("M4RI 20260122 {what}, not full"), format!("on a copy; ranks equal: {same}"), move |k| {
                            black_box(refs::m4ri::op(2, &m4[k % m4.len()], None, arg, false));
                        }));
                    }
                }
                #[cfg(has_ntl)]
                if !nt.is_empty() {
                    let same = insts.iter().zip(&nt).all(|(a, x)| refs::ntl::gf2_op(2, x, None, true) == a.rank() as u64);
                    let nt = &nt;
                    refs_v.push(Ref::new("NTL 11.6 mat_GF2 gauss", format!("on a copy; ranks equal: {same}"), move |k| {
                        black_box(refs::ntl::gf2_op(2, &nt[k % nt.len()], None, false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::rank", &params, Side::new("ours", |k| {
                    black_box(insts[k % insts.len()].rank());
                }), refs_v, &format!("{} instances alternated", insts.len()));
            }
            if ctx.wants("BitMatrix::reduced_row_echelon_form", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    let same = insts.iter().zip(&m4).all(|(a, x)| {
                        let (r, piv) = a.reduced_row_echelon_form();
                        refs::m4ri::op(3, x, None, 0, true) == (digest(&r) ^ piv.len() as u64)
                    });
                    for (arg, what) in [(0, "mzd_echelonize (default)"), (1, "mzd_echelonize_m4ri"), (2, "mzd_echelonize_pluq")] {
                        let m4 = &m4;
                        refs_v.push(Ref::new(format!("M4RI 20260122 {what}, full"), format!("on a copy; forms equal: {same}"), move |k| {
                            black_box(refs::m4ri::op(3, &m4[k % m4.len()], None, arg, false));
                        }));
                    }
                }
                ctx.row("gf2", "BitMatrix::reduced_row_echelon_form", &params, Side::new("ours", |k| {
                    black_box(insts[k % insts.len()].reduced_row_echelon_form());
                }), refs_v, &format!("{} instances alternated", insts.len()));
            }
            if mk == 0 && n <= 4096 && ctx.wants("BitMatrix::echelonize", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    let m4 = &m4;
                    refs_v.push(Ref::new("M4RI 20260122 mzd_echelonize (default), full", "copy included on both sides", move |k| {
                        black_box(refs::m4ri::op(3, &m4[k % m4.len()], None, 0, false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::echelonize", &params, Side::new("ours", |k| {
                    let mut a = insts[k % insts.len()].clone();
                    black_box(a.echelonize());
                }), refs_v, "in place on a clone");
            }
            if mk == 1 && n <= 8192 && ctx.wants("BitMatrix::null_space", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    let same = insts.iter().zip(&m4).all(|(a, x)| refs::m4ri::op(6, x, None, 0, true) == a.null_space().cols() as u64);
                    let m4 = &m4;
                    refs_v.push(Ref::new("M4RI 20260122 mzd_kernel_left_pluq", format!("a basis, not the canonical one; dimensions equal: {same}"), move |k| {
                        black_box(refs::m4ri::op(6, &m4[k % m4.len()], None, 0, false));
                    }));
                }
                #[cfg(has_ntl)]
                if !nt.is_empty() {
                    let ntt: Vec<_> = nt.iter().map(refs::ntl::gf2_transpose).collect();
                    let same = insts.iter().zip(&ntt).all(|(a, x)| refs::ntl::gf2_op(7, x, None, true) == a.null_space().cols() as u64);
                    refs_v.push(Ref::new("NTL 11.6 mat_GF2 kernel of Aᵀ", format!("a basis; dimensions equal: {same}"), move |k| {
                        black_box(refs::ntl::gf2_op(7, &ntt[k % ntt.len()], None, false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::null_space", &params, Side::new("ours", |k| {
                    black_box(insts[k % insts.len()].null_space());
                }), refs_v, &format!("canonical basis; {} instances alternated", insts.len()));
            }
        }
        // inverse, solve, det on invertible matrices.
        if n <= 8192 {
            let insts: Vec<(BitMatrix, BitMatrix)> = (0..k).map(|_| (invertible(&mut rng, n), random(&mut rng, n, 1))).collect();
            let params = format!("n={n}, invertible");
            #[cfg(has_m4ri)]
            let m4: Vec<_> = insts.iter().map(|(a, b)| (refs::m4ri::m(a.as_words(), n, n), refs::m4ri::m(b.as_words(), n, 1))).collect();
            #[cfg(has_ntl)]
            let nt: Vec<_> = insts.iter().map(|(a, b)| (refs::ntl::gf2(a.as_words(), n, n), refs::ntl::gf2(b.as_words(), n, 1))).collect();
            if ctx.wants("BitMatrix::inverse", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    let same = insts.iter().zip(&m4).all(|((a, _), (x, _))| refs::m4ri::op(4, x, None, 0, true) == digest(&a.inverse().unwrap()));
                    let m4 = &m4;
                    refs_v.push(Ref::new("M4RI 20260122 mzd_inv_m4ri", format!("equal: {same}"), move |k| {
                        black_box(refs::m4ri::op(4, &m4[k % m4.len()].0, None, 0, false));
                    }));
                }
                #[cfg(has_ntl)]
                {
                    let same = insts.iter().zip(&nt).all(|((a, _), (x, _))| refs::ntl::gf2_op(5, x, None, true) == digest(&a.inverse().unwrap()));
                    let nt = &nt;
                    refs_v.push(Ref::new("NTL 11.6 mat_GF2 inv", format!("equal: {same}"), move |k| {
                        black_box(refs::ntl::gf2_op(5, &nt[k % nt.len()].0, None, false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::inverse", &params, Side::new("ours", |k| {
                    black_box(insts[k % insts.len()].0.inverse());
                }), refs_v, &format!("{} instances alternated", insts.len()));
            }
            if ctx.wants("BitMatrix::solve", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_m4ri)]
                {
                    let same = insts.iter().zip(&m4).all(|((a, b), (x, y))| refs::m4ri::op(5, x, Some(y), 0, true) == digest(&a.solve(b).unwrap()));
                    let m4 = &m4;
                    refs_v.push(Ref::new("M4RI 20260122 mzd_solve_left (consistency checked)", format!("copies included; equal: {same}"), move |k| {
                        let (x, y) = &m4[k % m4.len()];
                        black_box(refs::m4ri::op(5, x, Some(y), 0, false));
                    }));
                }
                #[cfg(has_ntl)]
                {
                    let nt = &nt;
                    refs_v.push(Ref::new("NTL 11.6 mat_GF2 solve(d, A, x, b)", "square A only", move |k| {
                        let (x, y) = &nt[k % nt.len()];
                        black_box(refs::ntl::gf2_op(4, x, Some(y), false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::solve", &format!("{params}, one column"), Side::new("ours", |k| {
                    let (a, b) = &insts[k % insts.len()];
                    black_box(a.solve(b));
                }), refs_v, &format!("{} instances alternated", insts.len()));
            }
            if ctx.wants("BitMatrix::det", &params) {
                let mut refs_v: Vec<Ref> = Vec::new();
                #[cfg(has_ntl)]
                {
                    let same = insts.iter().zip(&nt).all(|((a, _), (x, _))| refs::ntl::gf2_op(3, x, None, true) == a.det() as u64);
                    let nt = &nt;
                    refs_v.push(Ref::new("NTL 11.6 mat_GF2 determinant", format!("equal: {same}"), move |k| {
                        black_box(refs::ntl::gf2_op(3, &nt[k % nt.len()].0, None, false));
                    }));
                }
                #[cfg(has_m4ri)]
                {
                    let m4 = &m4;
                    refs_v.push(Ref::new("M4RI 20260122 mzd_echelonize (default) rank = n", "M4RI has no det: the rank test", move |k| {
                        black_box(refs::m4ri::op(2, &m4[k % m4.len()].0, None, 0, false));
                    }));
                }
                ctx.row("gf2", "BitMatrix::det", &params, Side::new("ours", |k| {
                    black_box(insts[k % insts.len()].0.det());
                }), refs_v, &format!("{} instances alternated", insts.len()));
            }
        }
    }
    // ---- O(n²) operations ----
    for &n in if q { &[256usize][..] } else { &[1024, 4096, 16384][..] } {
        let a = random(&mut rng, n, n);
        let b = random(&mut rng, n, n);
        #[cfg(has_m4ri)]
        let (x, y) = (refs::m4ri::m(a.as_words(), n, n), refs::m4ri::m(b.as_words(), n, n));
        #[cfg(has_ntl)]
        let nx = refs::ntl::gf2(a.as_words(), n, n);
        let params = format!("n={n}");
        if ctx.wants("BitMatrix::transpose", &params) {
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_m4ri)]
            {
                let same = refs::m4ri::op(7, &x, None, 0, true) == digest(&a.transpose());
                refs_v.push(Ref::new("M4RI 20260122 mzd_transpose", format!("equal: {same}"), |_| {
                    black_box(refs::m4ri::op(7, &x, None, 0, false));
                }));
            }
            #[cfg(has_ntl)]
            {
                let same = refs::ntl::gf2_op(10, &nx, None, true) == digest(&a.transpose());
                refs_v.push(Ref::new("NTL 11.6 mat_GF2 transpose", format!("equal: {same}"), |_| {
                    black_box(refs::ntl::gf2_op(10, &nx, None, false));
                }));
            }
            ctx.row("gf2", "BitMatrix::transpose", &params, Side::new("ours", |_| {
                black_box(a.transpose());
            }), refs_v, "");
        }
        if ctx.wants("BitMatrix + BitMatrix", &params) {
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_m4ri)]
            {
                let same = refs::m4ri::op(8, &x, Some(&y), 0, true) == digest(&(&a + &b));
                refs_v.push(Ref::new("M4RI 20260122 mzd_add", format!("equal: {same}"), |_| {
                    black_box(refs::m4ri::op(8, &x, Some(&y), 0, false));
                }));
            }
            ctx.row("gf2", "BitMatrix + BitMatrix", &params, Side::new("ours", |_| {
                black_box(&a + &b);
            }), refs_v, "");
        }
    }
    // ---- row operations in place ----
    for &n in if q { &[1024usize][..] } else { &[1024, 16384][..] } {
        let mut a = random(&mut rng, 64.max(n / 64), n);
        let (i, j) = (0, a.rows() / 2);
        let params = format!("{} × {n}, rows {i} and {j}", a.rows());
        #[cfg(has_m4ri)]
        let mut x = refs::m4ri::m(a.as_words(), a.rows(), n);
        #[cfg(has_m4ri)]
        let mut x2 = refs::m4ri::m(a.as_words(), a.rows(), n);
        let mut a2 = a.clone();
        if ctx.wants("BitMatrix::add_row", &params) {
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_m4ri)]
            refs_v.push(Ref::new("M4RI 20260122 mzd_row_add", "", |_| {
                refs::m4ri::rowop(0, &mut x, j, i);
            }));
            ctx.row("gf2", "BitMatrix::add_row", &params, Side::new("ours", |_| {
                a.add_row(i, j);
            }), refs_v, "in place");
        }
        if ctx.wants("BitMatrix::swap_rows", &params) {
            let mut refs_v: Vec<Ref> = Vec::new();
            #[cfg(has_m4ri)]
            refs_v.push(Ref::new("M4RI 20260122 mzd_row_swap", "", |_| {
                refs::m4ri::rowop(1, &mut x2, i, j);
            }));
            ctx.row("gf2", "BitMatrix::swap_rows", &params, Side::new("ours", |_| {
                a2.swap_rows(i, j);
            }), refs_v, "in place");
        }
    }
    // ---- From<&Matrix<Number>>: no reference converts parities of big integers ----
    for &(n, bits) in if q { &[(64usize, 64u64)][..] } else { &[(256, 64), (256, 1000)][..] } {
        let params = format!("n={n}, entries of {bits} bits");
        if !ctx.wants("BitMatrix::from(&Matrix<Number>)", &params) {
            continue;
        }
        let m: Matrix<Number> = uniform(&mut rng, n, n, bits);
        ctx.row("gf2", "BitMatrix::from(&Matrix<Number>)", &params, Side::new("ours", |_| {
            black_box(BitMatrix::from(&m));
        }), Vec::new(), "no reference has this conversion");
    }
}
