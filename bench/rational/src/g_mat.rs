// Matrices of rationals (det, solve, inverse, product) and Matrix<Number>'s solutions and
// inverses over the rationals, below the thresholds of Dixon's lifting (area lattice).
use std::hint::black_box;

use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::math::rational::Rational;

use crate::data::*;
#[allow(unused_imports)]
use crate::ffi::*;
use crate::harness::{Ent, agree, row};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ints,
    Fracs,
    W64,
    W256,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Ints => "integers in -99..=99",
            Kind::Fracs => "fractions p/q, |p| <= 99, 1 <= q <= 99",
            Kind::W64 => "64-bit parts",
            Kind::W256 => "256-bit parts",
        }
    }
    fn q(self, rng: &mut Rng, int: bool) -> Q {
        match self {
            Kind::Ints => {
                let v = rng.below(199) as i64 - 99;
                Q { neg: v < 0, num: trim(v.unsigned_abs().to_le_bytes().to_vec()), den: vec![1] }
            }
            Kind::Fracs => {
                let p = rng.below(199) as i64 - 99;
                let q = 1 + rng.below(99);
                reduce(p < 0, trim(p.unsigned_abs().to_le_bytes().to_vec()), trim(q.to_le_bytes().to_vec()))
            }
            Kind::W64 | Kind::W256 => {
                let b = if self == Kind::W64 { 64 } else { 256 };
                if int {
                    Q { neg: rng.bool(), num: rng.exact(b), den: vec![1] }
                } else {
                    rand_q(rng, b, b, true, true)
                }
            }
        }
    }
}

struct Pool {
    a: Vec<Vec<Vec<Q>>>,
    b: Vec<Vec<Vec<Q>>>,
}

fn make_pool(rng: &mut Rng, kind: Kind, n: usize, m: usize, k: usize, int: bool) -> Pool {
    let mut a = Vec::new();
    let mut b = Vec::new();
    while a.len() < k {
        let ma: Vec<Vec<Q>> = (0..n).map(|_| (0..n).map(|_| kind.q(rng, int)).collect()).collect();
        let mr = Matrix::from_fn(n, n, |i, j| ours(&ma[i][j]));
        if mr.det().is_zero() {
            continue;
        }
        a.push(ma);
        b.push((0..n).map(|_| (0..m).map(|_| kind.q(rng, int)).collect()).collect());
    }
    Pool { a, b }
}

fn pool_k(n: usize) -> usize {
    (256 / (n * n)).clamp(2, 64)
}

fn show_entries(m: &Matrix<Rational>, op: i32, n: usize, hex: bool) -> Vec<String> {
    let f = |r: &Rational| if hex { r.to_str_radix(16) } else { r.to_string() };
    match op {
        1 => vec![f(&m[(0, 0)]), f(&m[(n - 1, 0)])],
        _ => vec![f(&m[(0, 0)]), f(&m[(n - 1, n - 1)])],
    }
}

/// op: 0 det, 1 solve, 2 inverse, 3 product (A * A); 4 Matrix<Number>::solve, 5 inverse_rational.
fn mat_row(rng: &mut Rng, kind: Kind, n: usize, op: i32) {
    let k = pool_k(n);
    let int = op >= 4;
    let pool = make_pool(rng, kind, n, 1, k, int);
    let oa: Vec<Matrix<Rational>> = pool.a.iter().map(|m| Matrix::from_fn(n, n, |i, j| ours(&m[i][j]))).collect();
    let ob: Vec<Matrix<Rational>> = pool.b.iter().map(|m| Matrix::from_fn(n, 1, |i, j| ours(&m[i][j]))).collect();
    let za: Vec<Matrix<Number>> = pool.a.iter().map(|m| Matrix::from_fn(n, n, |i, j| number(&m[i][j].num, m[i][j].neg))).collect();
    let zb: Vec<Matrix<Number>> = pool.b.iter().map(|m| Matrix::from_fn(n, 1, |i, j| number(&m[i][j].num, m[i][j].neg))).collect();
    // Our results for the check, as hex and as decimal.
    let res = |t: usize, hex: bool| -> Vec<String> {
        match op {
            0 => vec![if hex { oa[t].det().to_str_radix(16) } else { oa[t].det().to_string() }],
            1 => show_entries(&oa[t].solve(&ob[t]).unwrap(), 1, n, hex),
            2 => show_entries(&oa[t].inverse().unwrap(), 2, n, hex),
            3 => show_entries(&(&oa[t] * &oa[t]), 3, n, hex),
            4 => show_entries(&za[t].solve(&zb[t]).unwrap(), 1, n, hex),
            _ => show_entries(&za[t].inverse_rational().unwrap(), 2, n, hex),
        }
    };
    let kk = k.min(4);
    let mine_hex: Vec<String> = (0..kk).flat_map(|t| res(t, true)).collect();
    #[cfg(has_pari)]
    let mine_dec: Vec<String> = (0..kk).flat_map(|t| res(t, false)).collect();
    let mut j = 0usize;
    let (oar, obr, zar, zbr) = (&oa, &ob, &za, &zb);
    let ours_ent = Ent::new("a", move |c| {
        for _ in 0..c {
            match op {
                0 => {
                    black_box(oar[j].det());
                }
                1 => {
                    black_box(oar[j].solve(&obr[j]));
                }
                2 => {
                    black_box(oar[j].inverse());
                }
                3 => {
                    black_box(&oar[j] * &oar[j]);
                }
                4 => {
                    black_box(zar[j].solve(&zbr[j]));
                }
                _ => {
                    black_box(zar[j].inverse_rational());
                }
            }
            j += 1;
            if j == k {
                j = 0;
            }
        }
    });
    let mut ents = vec![ours_ent];
    #[cfg(has_flint)]
    let fm = FlintMat::new(k, n, 1);
    #[cfg(has_flint)]
    {
        for t in 0..k {
            for i in 0..n {
                for jj in 0..n {
                    fm.set(0, t, i, jj, &pool.a[t][i][jj]);
                    fm.set(2, t, i, jj, &pool.a[t][i][jj]);
                }
                fm.set(1, t, i, 0, &pool.b[t][i][0]);
                fm.set(3, t, i, 0, &pool.b[t][i][0]);
            }
        }
        let theirs: Vec<String> = (0..kk)
            .flat_map(|t| {
                fm.run_one(op, t);
                match op {
                    0 => vec![fm.str(0, 0, 0)],
                    1 | 4 => vec![fm.str(1, 0, 0), fm.str(1, n - 1, 0)],
                    _ => vec![fm.str(2, 0, 0), fm.str(2, n - 1, n - 1)],
                }
            })
            .collect();
        let nt = agree(&mine_hex, &theirs, "mat flint");
        let name = match op {
            0 => "FLINT fmpq_mat_det",
            1 => "FLINT fmpq_mat_solve",
            2 => "FLINT fmpq_mat_inv",
            3 => "FLINT fmpq_mat_mul",
            4 => "FLINT fmpq_mat_solve_fmpz_mat",
            _ => "FLINT fmpz_mat_inv + fmpq_mat_set_fmpz_mat_div_fmpz",
        };
        ents.push(Ent::new(name, move |c| {
            fm.run(op, c);
        }).note(&nt));
    }
    #[cfg(has_pari)]
    let pm = PariMat::new(k, n, 1);
    #[cfg(has_pari)]
    {
        for t in 0..k {
            let (pa, pb) = (&pool.a[t], &pool.b[t]);
            if int {
                pm.set(t, n, 1, |i, jj| Q { den: vec![1], ..pa[i][jj].clone() }, |i, jj| Q { den: vec![1], ..pb[i][jj].clone() });
            } else {
                pm.set(t, n, 1, |i, jj| pa[i][jj].clone(), |i, jj| pb[i][jj].clone());
            }
        }
        let pop = match op {
            4 => 1,
            5 => 2,
            x => x,
        };
        let theirs: Vec<String> = (0..kk)
            .flat_map(|t| {
                pm.run_one(pop, t);
                match pop {
                    0 => vec![pm.str(0, 0, 0)],
                    1 => vec![pm.str(1, 0, 0), pm.str(1, n - 1, 0)],
                    _ => vec![pm.str(2, 0, 0), pm.str(2, n - 1, n - 1)],
                }
            })
            .collect();
        let nt = agree(&mine_dec, &theirs, "mat pari");
        let name = match pop {
            0 => "PARI det",
            1 => "PARI RgM_solve",
            2 => "PARI RgM_inv",
            _ => "PARI gmul",
        };
        ents.push(Ent::new(name, move |c| {
            pm.run(pop, c);
        }).note(&nt));
    }
    let (opname, what) = match op {
        0 => ("Matrix<Rational>::det", ""),
        1 => ("Matrix<Rational>::solve", ", one column"),
        2 => ("Matrix<Rational>::inverse", ""),
        3 => ("&a * &b of Matrix<Rational>", " (A * A)"),
        4 => ("Matrix<Number>::solve", ", one column, over the rationals"),
        _ => ("Matrix<Number>::inverse_rational", ""),
    };
    let lab = if int && kind == Kind::W64 { "64-bit integers" } else { kind.label() };
    row("matrix", opname, &format!("n = {n}{what}; {lab}; pool {k}"), ents, 1.0, "");
}

pub fn run() {
    let mut rng = Rng::new(0x5EED_0007);
    for kind in [Kind::Ints, Kind::Fracs, Kind::W64, Kind::W256] {
        let ns: &[usize] = if kind == Kind::W256 { &[2, 4, 8] } else { &[2, 3, 4, 6, 8, 16] };
        for &n in ns {
            for op in 0..3 {
                mat_row(&mut rng, kind, n, op);
            }
        }
        let ms: &[usize] = match kind {
            Kind::Ints | Kind::Fracs => &[2, 4, 8, 16, 32, 64],
            Kind::W64 => &[2, 4, 8, 16],
            Kind::W256 => &[2, 4],
        };
        for &n in ms {
            mat_row(&mut rng, kind, n, 3);
        }
    }
    for kind in [Kind::Ints, Kind::W64] {
        for n in [2usize, 4, 8, 16] {
            mat_row(&mut rng, kind, n, 4);
            mat_row(&mut rng, kind, n, 5);
        }
    }
}

/// Rank and null space of singular matrices (rank n - 2: the last two rows are sums of others).
pub fn run_rank() {
    let mut rng = Rng::new(0x5EED_0008);
    for kind in [Kind::Ints, Kind::Fracs] {
        for n in [4usize, 8, 16] {
            let k = pool_k(n);
            let mats: Vec<Vec<Vec<Q>>> = (0..k)
                .map(|_| {
                    let mut m: Vec<Vec<Q>> = (0..n).map(|_| (0..n).map(|_| kind.q(&mut rng, false)).collect()).collect();
                    for j in 0..n {
                        let s = &ours(&m[0][j]) + &ours(&m[1][j]);
                        let d = &ours(&m[2][j]) - &ours(&m[3][j]);
                        m[n - 2][j] = of_ours(&s);
                        m[n - 1][j] = of_ours(&d);
                    }
                    m
                })
                .collect();
            let oa: Vec<Matrix<Rational>> = mats.iter().map(|m| Matrix::from_fn(n, n, |i, j| ours(&m[i][j]))).collect();
            for op in [0, 1] {
                let mine: Vec<String> = oa.iter().take(4).map(|a| if op == 0 { a.rank().to_string() } else { a.null_space().cols().to_string() }).collect();
                let mut j = 0usize;
                let oar = &oa;
                let mut ents = vec![Ent::new("a", move |c| {
                    for _ in 0..c {
                        if op == 0 {
                            black_box(oar[j].rank());
                        } else {
                            black_box(oar[j].null_space());
                        }
                        j += 1;
                        if j == k {
                            j = 0;
                        }
                    }
                })];
                #[cfg(has_flint)]
                {
                    let fm = FlintMat::new(k, n, 1);
                    for (t, m) in mats.iter().enumerate() {
                        for i in 0..n {
                            for jj in 0..n {
                                fm.set(0, t, i, jj, &m[i][jj]);
                            }
                        }
                    }
                    let theirs: Vec<String> = (0..mine.len()).map(|t| { let r = fm.run_one(6, t) as usize; if op == 0 { r.to_string() } else { (n - r).to_string() } }).collect();
                    let nt = agree(&mine, &theirs, "rank flint");
                    ents.push(Ent::new("FLINT fmpq_mat_rref", move |c| {
                        fm.run(6, c);
                    }).note(if op == 0 { "the reduced form, whose rank it returns" } else { "the reduced form, from which the basis is read off (not timed)" }).note(&nt));
                }
                #[cfg(has_pari)]
                {
                    let pm = PariMat::new(k, n, 1);
                    for (t, m) in mats.iter().enumerate() {
                        pm.set(t, n, 1, |i, jj| m[i][jj].clone(), |_, _| Q { neg: false, num: vec![0], den: vec![1] });
                    }
                    let pop = if op == 0 { 4 } else { 5 };
                    let theirs: Vec<String> = (0..mine.len()).map(|t| { let r = pm.run_one(pop, t); if op == 0 { pm.str(0, 0, 0) } else { (r - 1).to_string() } }).collect();
                    let nt = agree(&mine, &theirs, "rank pari");
                    ents.push(Ent::new(if op == 0 { "PARI rank" } else { "PARI ker" }, move |c| {
                        pm.run(pop, c);
                    }).note(if op == 0 { "" } else { "its own basis of the kernel" }).note(&nt));
                }
                let opname = if op == 0 { "Matrix<Rational>::rank" } else { "Matrix<Rational>::null_space" };
                row("matrix", opname, &format!("n = {n}, rank n - 2; {}; pool {k}", kind.label()), ents, 1.0, "");
            }
        }
    }
}

/// Matrix<Number>::det of integers (the page of matrices; known from #27: 1.02-1.03 at n = 2-4).
pub fn run_zdet() {
    let mut rng = Rng::new(0x5EED_000A);
    for kind in [Kind::Ints, Kind::W64] {
        for n in [2usize, 3, 4, 6, 8, 16] {
            let k = pool_k(n);
            let pool = make_pool(&mut rng, kind, n, 1, k, true);
            let za: Vec<Matrix<Number>> = pool.a.iter().map(|m| Matrix::from_fn(n, n, |i, j| number(&m[i][j].num, m[i][j].neg))).collect();
            let kk = k.min(4);
            let mine_hex: Vec<String> = za.iter().take(kk).map(|a| a.det().to_str_radix(16)).collect();
            #[cfg(has_pari)]
            let mine_dec: Vec<String> = za.iter().take(kk).map(|a| a.det().to_string()).collect();
            let mut j = 0usize;
            let zar = &za;
            let mut ents = vec![Ent::new("a", move |c| {
                for _ in 0..c {
                    black_box(zar[j].det());
                    j += 1;
                    if j == k {
                        j = 0;
                    }
                }
            })];
            #[cfg(has_flint)]
            {
                let fm = FlintMat::new(k, n, 1);
                for t in 0..k {
                    for i in 0..n {
                        for jj in 0..n {
                            fm.set(2, t, i, jj, &pool.a[t][i][jj]);
                        }
                    }
                }
                let theirs: Vec<String> = (0..kk).map(|t| { fm.run_one(7, t); fm.str(7, 0, 0) }).collect();
                let nt = agree(&mine_hex, &theirs, "zdet flint");
                ents.push(Ent::new("FLINT fmpz_mat_det", move |c| {
                    fm.run(7, c);
                }).note(&nt));
            }
            #[cfg(has_pari)]
            {
                let pm = PariMat::new(k, n, 1);
                for t in 0..k {
                    let pa = &pool.a[t];
                    pm.set(t, n, 1, |i, jj| Q { den: vec![1], ..pa[i][jj].clone() }, |_, _| Q { neg: false, num: vec![0], den: vec![1] });
                }
                let theirs: Vec<String> = (0..kk).map(|t| { pm.run_one(0, t); pm.str(0, 0, 0) }).collect();
                let nt = agree(&mine_dec, &theirs, "zdet pari");
                ents.push(Ent::new("PARI det", move |c| {
                    pm.run(0, c);
                }).note(&nt));
            }
            let lab = if kind == Kind::W64 { "64-bit integers" } else { kind.label() };
            row("matrix", "Matrix<Number>::det", &format!("n = {n}; {lab}; pool {k}"), ents, 1.0, "");
        }
    }
}
