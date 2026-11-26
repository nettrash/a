//! ModMatrix (GF(p), p < 2^64) against FLINT's nmod_mat and NTL's mat_zz_p (p < 2^60).
#![allow(unused_variables, unused_mut)]
use crate::inputs::*;
#[allow(unused_imports)]
use crate::refs;
use crate::timing::Side;
use crate::{Ctx, Ref};
use a::math::matrix::ModMatrix;
use std::hint::black_box;

const PRIMES: &[(u64, &str)] = &[
    (65_521, "p=65521 (16 bits)"),
    (2_147_483_647, "p=2^31−1"),
    (1_125_899_906_842_597, "p=2^50−27"),
    (4_611_686_018_427_387_847, "p=2^62−57"),
    (18_446_744_073_709_551_557, "p=2^64−59"),
];

fn random(rng: &mut Rng, r: usize, c: usize, p: u64) -> ModMatrix {
    ModMatrix::from_fn(r, c, p, |_, _| rng.next() % p)
}

/// A matrix of rank `k`: the product of random `r × k` and `k × c` factors.
fn deficient(rng: &mut Rng, r: usize, c: usize, k: usize, p: u64) -> ModMatrix {
    let x = random(rng, r, k, p);
    let y = random(rng, k, c, p);
    &x * &y
}

fn digest(a: &ModMatrix) -> u64 {
    fnv(FNV_INIT, a.as_slice())
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Mul,
    Ple,
    Rank,
    Det,
    Solve,
    Inverse,
    Rref,
    Null,
    Add,
    Sub,
    Scale,
    Transpose,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Mul => "ModMatrix * ModMatrix",
            Op::Ple => "ModMatrix::ple",
            Op::Rank => "ModMatrix::rank",
            Op::Det => "ModMatrix::det",
            Op::Solve => "ModMatrix::solve",
            Op::Inverse => "ModMatrix::inverse",
            Op::Rref => "ModMatrix::rref",
            Op::Null => "ModMatrix::null_space",
            Op::Add => "ModMatrix + ModMatrix",
            Op::Sub => "ModMatrix - ModMatrix",
            Op::Scale => "ModMatrix::scale",
            Op::Transpose => "ModMatrix::transpose",
        }
    }
    /// The FLINT op code of the shim.
    fn flint(self) -> i32 {
        match self {
            Op::Mul => 0,
            Op::Ple => 1,
            Op::Rank => 2,
            Op::Det => 3,
            Op::Solve => 4,
            Op::Inverse => 5,
            Op::Rref => 6,
            Op::Null => 7,
            Op::Add => 8,
            Op::Sub => 9,
            Op::Transpose => 10,
            Op::Scale => 11,
        }
    }
    fn ntl(self) -> Option<i32> {
        match self {
            Op::Mul => Some(0),
            Op::Rank => Some(2),
            Op::Det => Some(3),
            Op::Solve => Some(4),
            Op::Inverse => Some(5),
            Op::Null => Some(7),
            _ => None,
        }
    }
}

/// The operation; its result's digest (or the scalar it is) for the check, nothing but the
/// operation itself when timed (`want` false: the result is passed to `black_box` and dropped).
fn run_ours(op: Op, a: &ModMatrix, b: &ModMatrix, want: bool) -> u64 {
    fn d(x: ModMatrix, want: bool) -> u64 {
        if want {
            digest(&x)
        } else {
            black_box(x);
            0
        }
    }
    match op {
        Op::Mul => d(a * b, want),
        Op::Ple => a.ple().rank() as u64,
        Op::Rank => a.rank() as u64,
        Op::Det => a.det(),
        Op::Solve => a.solve(b).map_or(0, |x| d(x, want)),
        Op::Inverse => a.inverse().map_or(0, |x| d(x, want)),
        Op::Rref => {
            let (r, piv) = a.rref();
            d(r, want) ^ if want { piv.len() as u64 } else { black_box(piv).len() as u64 }
        }
        Op::Null => d(a.null_space(), want),
        Op::Add => d(a + b, want),
        Op::Sub => d(a - b, want),
        Op::Scale => d(a.scale(12345 % a.modulus()), want),
        Op::Transpose => d(a.transpose(), want),
    }
}

fn row(ctx: &Ctx, rng: &mut Rng, op: Op, n: usize, p: u64, pname: &str) {
    let params = format!("n={n}, {pname}");
    if !ctx.wants(op.name(), &params) {
        return;
    }
    let k = if n <= 256 { 3 } else { 1 };
    let insts: Vec<(ModMatrix, ModMatrix)> = (0..k)
        .map(|_| {
            let a = if op == Op::Null { deficient(rng, n, n, n - n / 8, p) } else { random(rng, n, n, p) };
            let bc = if op == Op::Solve { 1 } else { n };
            (a, random(rng, n, bc, p))
        })
        .collect();
    #[cfg(has_flint)]
    let fl: Vec<_> = insts.iter().map(|(a, b)| (refs::flint::n(a.as_slice(), n, n, p), refs::flint::n(b.as_slice(), b.rows(), b.cols(), p))).collect();
    #[cfg(has_ntl)]
    let nt: Vec<_> = if op.ntl().is_some() && 64 - p.leading_zeros() <= refs::ntl::sp_nbits() {
        insts.iter().map(|(a, b)| (refs::ntl::zp(a.as_slice(), n, n, p), refs::ntl::zp(b.as_slice(), b.rows(), b.cols(), p))).collect()
    } else {
        Vec::new()
    };
    let fo = op.flint();
    let no = op.ntl().unwrap_or(0);
    let mut refs_v: Vec<Ref> = Vec::new();
    #[cfg(has_flint)]
    {
        let checkable = !matches!(op, Op::Ple);
        let same = insts.iter().zip(&fl).all(|((a, b), (fa, fb))| {
            let ours = run_ours(op, a, b, true);
            let theirs = refs::flint::nop(fo, fa, Some(fb), true);
            match op {
                // ple: FLINT's LU returns the rank, ours too.
                Op::Ple => ours == theirs,
                // rref: FLINT's digest of the whole form xor the rank; ours xor the pivot count.
                _ => ours == theirs,
            }
        });
        let what = match op {
            Op::Ple => "nmod_mat_lu (in place on a copy)",
            Op::Mul => "nmod_mat_mul",
            Op::Rank => "nmod_mat_rank",
            Op::Det => "nmod_mat_det",
            Op::Solve => "nmod_mat_solve",
            Op::Inverse => "nmod_mat_inv",
            Op::Rref => "nmod_mat_rref (on a copy)",
            Op::Null => "nmod_mat_nullspace",
            Op::Add => "nmod_mat_add",
            Op::Sub => "nmod_mat_sub",
            Op::Scale => "nmod_mat_scalar_mul",
            Op::Transpose => "nmod_mat_transpose",
        };
        refs_v.push(Ref::new(format!("FLINT 3.6 {what}"), format!("results equal: {same}"), |k| {
            let (fa, fb) = &fl[k % fl.len()];
            black_box(refs::flint::nop(fo, fa, Some(fb), false));
        }));
    }
    #[cfg(has_ntl)]
    if !nt.is_empty() {
        let what = match op {
            Op::Mul => "mul",
            Op::Rank => "gauss (rank, on a copy)",
            Op::Det => "determinant",
            Op::Solve => "solve(d, A, x, b)",
            Op::Inverse => "inv(d, X, A)",
            Op::Null => "kernel of Aᵀ (its dimension)",
            _ => "",
        };
        let same = insts.iter().zip(&nt).all(|((a, b), (na, nb))| {
            let theirs = refs::ntl::zp_op(no, na, Some(nb), true);
            match op {
                Op::Null => theirs == a.null_space().cols() as u64,
                Op::Solve => theirs == a.solve(b).map_or(0, |x| fnv(FNV_INIT, x.as_slice())),
                _ => theirs == run_ours(op, a, b, true),
            }
        });
        refs_v.push(Ref::new(format!("NTL 11.6 mat_zz_p {what}"), format!("results equal: {same}"), |k| {
            let (na, nb) = &nt[k % nt.len()];
            black_box(refs::ntl::zp_op(no, na, Some(nb), false));
        }));
    }
    let deficient_note = if op == Op::Null { "rank n − n/8" } else { "uniform random" };
    ctx.row(
        "modp",
        op.name(),
        &params,
        Side::new("ours", |k| {
            let (a, b) = &insts[k % insts.len()];
            black_box(run_ours(op, a, b, false));
        }),
        refs_v,
        &format!("{deficient_note}; {} instances alternated", insts.len()),
    );
}

pub fn modp(ctx: &Ctx) {
    let mut rng = Rng(13);
    let q = std::env::var("LATBENCH_QUICK").is_ok();
    let (p31, p62) = (PRIMES[1], PRIMES[3]);
    // Products, across the Strassen–Winograd crossover (128) and up to 4096.
    let mul_sizes: &[usize] = if q { &[64, 256] } else { &[64, 96, 128, 160, 192, 256, 512, 1024, 2048, 4096] };
    for &(p, pn) in &[p31, p62] {
        for &n in mul_sizes {
            row(ctx, &mut rng, Op::Mul, n, p, pn);
        }
    }
    if !q {
        for &(p, pn) in &[PRIMES[0], PRIMES[2], PRIMES[4]] {
            for n in [256, 1024] {
                row(ctx, &mut rng, Op::Mul, n, p, pn);
            }
        }
    }
    // Eliminations.
    let el_sizes: &[usize] = if q { &[64] } else { &[64, 256, 1024, 2048] };
    for op in [Op::Ple, Op::Rank, Op::Det, Op::Solve, Op::Inverse, Op::Rref] {
        for &(p, pn) in &[p31, p62] {
            for &n in el_sizes {
                row(ctx, &mut rng, op, n, p, pn);
            }
        }
    }
    if !q {
        for &(p, pn) in &[p31, p62] {
            row(ctx, &mut rng, Op::Ple, 4096, p, pn);
        }
        for &(p, pn) in &[PRIMES[0], PRIMES[2], PRIMES[4]] {
            for n in [256, 1024] {
                row(ctx, &mut rng, Op::Ple, n, p, pn);
            }
        }
        for &(p, pn) in &[p31, p62] {
            for n in [256, 1024] {
                row(ctx, &mut rng, Op::Null, n, p, pn);
            }
        }
        for op in [Op::Add, Op::Sub, Op::Scale, Op::Transpose] {
            row(ctx, &mut rng, op, 1024, p62.0, p62.1);
        }
    }
    // from_matrix: residues of a matrix of numbers.
    for &(n, bits) in if q { &[(64usize, 64u64)][..] } else { &[(256, 64), (256, 1000), (1024, 64)][..] } {
        let params = format!("n={n}, entries of {bits} bits, p=2^62−57");
        if !ctx.wants("ModMatrix::from_matrix", &params) {
            continue;
        }
        let a = uniform(&mut rng, n, n, bits);
        let p = p62.0;
        #[cfg(has_flint)]
        let fz = refs::flint::z(&text(&a), n, n);
        let mut refs_v: Vec<Ref> = Vec::new();
        #[cfg(has_flint)]
        {
            let same = refs::flint::from_z(&fz, p, true) == digest(&ModMatrix::from_matrix(&a, p));
            refs_v.push(Ref::new("FLINT 3.6 fmpz_mat_get_nmod_mat", format!("equal: {same}"), |_| {
                black_box(refs::flint::from_z(&fz, p, false));
            }));
        }
        ctx.row("modp", "ModMatrix::from_matrix", &params, Side::new("ours", |_| {
            black_box(ModMatrix::from_matrix(&a, p));
        }), refs_v, "");
    }
}
