//! Matrices of `BigFloat` and `BigComplex`: products (every element correctly rounded) against
//! an `mpfr_fma` loop (rounded at each step), `mpfr_dot` (correctly rounded) and Arb's
//! `arb_mat_mul`; solves against Gaussian elimination with partial pivoting by `mpfr_fma`.
use crate::ffi::*;
use crate::h::*;
use crate::libs::*;
use a::math::float::{BigComplex, BigFloat};
use a::math::matrix::Matrix;

const MP: [u64; 4] = [53, 113, 256, 1000];
const MN: [usize; 3] = [4, 8, 16];

fn entries(n: usize, p: u64, seed: u64) -> Vec<Raw> {
    let mut rng = Rng::new(seed ^ p ^ (n as u64) << 32);
    (0..n * n).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect()
}

/// Row by row (the matrix's own storage is by columns).
fn c_mat(m: &Matrix<BigFloat>) -> Canon {
    Canon::Many((0..m.rows()).flat_map(|i| (0..m.cols()).map(move |j| c_ours(&m[(i, j)]))).collect())
}

fn c_matc(m: &Matrix<BigComplex>) -> Canon {
    Canon::Many((0..m.rows()).flat_map(|i| (0..m.cols()).map(move |j| c_ours2(&m[(i, j)]))).collect())
}

pub fn product_rows() {
    for n in MN {
        for p in MP {
            let op = format!("&a * &b, {n}x{n}");
            let ps = format!("prec={p}");
            if !wanted("matrix", &op, &ps) {
                continue;
            }
            let l = 2usize;
            let ar: Vec<Vec<Raw>> = (0..l).map(|i| entries(n, p, 11 + i as u64)).collect();
            let br: Vec<Vec<Raw>> = (0..l).map(|i| entries(n, p, 21 + i as u64)).collect();
            let mut conts = Vec::new();
            {
                let a: Vec<Matrix<BigFloat>> = ar.iter().map(|v| Matrix::from_fn(n, n, |i, j| ours(&v[i * n + j], p))).collect();
                let b: Vec<Matrix<BigFloat>> = br.iter().map(|v| Matrix::from_fn(n, n, |i, j| ours(&v[i * n + j], p))).collect();
                conts.push(Cont::ours(kern!(l, |k| &a[k] * &b[k], |v| c_mat(&v))));
            }
            {
                // mpfr_dot: row and column pointer arrays prepared outside the timing
                let mut a: Vec<Vec<MF>> = ar.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                let mut b: Vec<Vec<MF>> = br.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                let rows: Vec<Vec<Vec<P>>> = a.iter_mut().map(|m| (0..n).map(|i| (0..n).map(|j| &mut m[i * n + j].0 as P).collect()).collect()).collect();
                let cols: Vec<Vec<Vec<P>>> = b.iter_mut().map(|m| (0..n).map(|j| (0..n).map(|i| &mut m[i * n + j].0 as P).collect()).collect()).collect();
                let mut c: Vec<MF> = (0..n * n).map(|_| MF::new(p)).collect();
                let k = kern!(
                    l,
                    |k| unsafe {
                        let _ = (&a, &b);
                        for i in 0..n {
                            for j in 0..n {
                                mpfr_dot(c[i * n + j].p(), rows[k][i].as_ptr(), cols[k][j].as_ptr(), n as _, RN);
                            }
                        }
                    },
                    |v| Canon::Many(c.iter().map(|x| c_mpfr(x.c())).collect())
                );
                conts.push(Cont::rf("MPFR mpfr_dot per element", k).note("same work: each element correctly rounded; into existing variables"));
            }
            {
                let a: Vec<Vec<MF>> = ar.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                let b: Vec<Vec<MF>> = br.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                let mut c: Vec<MF> = (0..n * n).map(|_| MF::new(p)).collect();
                let k = kern!(
                    l,
                    |k| unsafe {
                        for i in 0..n {
                            for j in 0..n {
                                let r = c[i * n + j].p();
                                mpfr_mul(r, a[k][i * n].c(), b[k][j].c(), RN);
                                for t in 1..n {
                                    mpfr_fma(r, a[k][i * n + t].c(), b[k][t * n + j].c(), r, RN);
                                }
                            }
                        }
                    },
                    |v| Canon::Skip
                );
                conts.push(Cont::rf("MPFR mpfr_fma loop", k).loose().note("does less: rounded at each step; into existing variables"));
            }
            #[cfg(has_arb)]
            {
                let ap = crate::g_func::arb_prec(p);
                let mk = |v: &Vec<Raw>| unsafe {
                    let mut m = std::mem::MaybeUninit::<arb::ArbMat>::uninit();
                    arb::arb_mat_init(m.as_mut_ptr(), n as _, n as _);
                    let mut m = m.assume_init();
                    for i in 0..n {
                        for j in 0..n {
                            let x = mpfr(&v[i * n + j], p);
                            arb::arf_set_mpfr(m.entries.add(i * m.stride as usize + j), x.c());
                        }
                    }
                    m
                };
                let a: Vec<arb::ArbMat> = ar.iter().map(mk).collect();
                let b: Vec<arb::ArbMat> = br.iter().map(mk).collect();
                let mut c = unsafe {
                    let mut m = std::mem::MaybeUninit::<arb::ArbMat>::uninit();
                    arb::arb_mat_init(m.as_mut_ptr(), n as _, n as _);
                    m.assume_init()
                };
                let k = kern!(l, |k| unsafe { arb::arb_mat_mul(&mut c, &a[k], &b[k], ap) }, |v| Canon::Skip);
                conts.push(Cont::rf("Arb arb_mat_mul (balls, does less)", k).loose().note(&format!("does less: balls at {ap} bits; matrices leaked at exit")));
            }
            row("matrix", &op, &ps, l, conts);
        }
    }
}

pub fn solve_rows() {
    for n in MN {
        for p in MP {
            let op = format!("a.solve(&b), {n}x{n}");
            let ps = format!("prec={p}");
            if !wanted("matrix", &op, &ps) {
                continue;
            }
            let l = 2usize;
            let ar: Vec<Vec<Raw>> = (0..l).map(|i| entries(n, p, 31 + i as u64)).collect();
            let br: Vec<Vec<Raw>> = (0..l).map(|i| {
                let mut rng = Rng::new(41 + i as u64 ^ p);
                (0..n).map(|_| Raw::rand_range(&mut rng, p, -1, 2, true)).collect()
            }).collect();
            let mut conts = Vec::new();
            {
                let a: Vec<Matrix<BigFloat>> = ar.iter().map(|v| Matrix::from_fn(n, n, |i, j| ours(&v[i * n + j], p))).collect();
                let b: Vec<Matrix<BigFloat>> = br.iter().map(|v| Matrix::from_fn(n, 1, |i, _| ours(&v[i], p))).collect();
                conts.push(Cont::ours(kern!(l, |k| a[k].solve(&b[k]).unwrap(), |v| Canon::Skip)));
            }
            {
                let a0: Vec<Vec<MF>> = ar.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                let b0: Vec<Vec<MF>> = br.iter().map(|v| v.iter().map(|r| mpfr(r, p)).collect()).collect();
                let mut w: Vec<MF> = (0..n * n).map(|_| MF::new(p)).collect();
                let mut y: Vec<MF> = (0..n).map(|_| MF::new(p)).collect();
                let mut x: Vec<MF> = (0..n).map(|_| MF::new(p)).collect();
                let mut f = MF::new(p);
                let k = kern!(
                    l,
                    |k| unsafe {
                        for i in 0..n * n {
                            mpfr_set(w[i].p(), a0[k][i].c(), RN);
                        }
                        for i in 0..n {
                            mpfr_set(y[i].p(), b0[k][i].c(), RN);
                        }
                        let mut perm: Vec<usize> = (0..n).collect();
                        for c in 0..n {
                            // partial pivoting
                            let mut best = c;
                            for i in c + 1..n {
                                if crate::g_mat::cmpabs(&w[perm[i] * n + c], &w[perm[best] * n + c]) > 0 {
                                    best = i;
                                }
                            }
                            perm.swap(c, best);
                            let pc = perm[c];
                            for i in c + 1..n {
                                let pi = perm[i];
                                mpfr_div(f.p(), w[pi * n + c].c(), w[pc * n + c].c(), RN);
                                let fp = f.p();
                                *(&mut (*fp).sign) = -(*fp).sign;
                                for j in c + 1..n {
                                    let t = w[pi * n + j].p();
                                    mpfr_fma(t, f.c(), w[pc * n + j].c(), t, RN);
                                }
                                let t = y[pi].p();
                                mpfr_fma(t, f.c(), y[pc].c(), t, RN);
                            }
                        }
                        for i in (0..n).rev() {
                            let pi = perm[i];
                            mpfr_set(f.p(), y[pi].c(), RN);
                            for j in i + 1..n {
                                mpfr_fms(f.p(), w[pi * n + j].c(), x[j].c(), f.c(), RN);
                                let fp = f.p();
                                *(&mut (*fp).sign) = -(*fp).sign;
                            }
                            mpfr_div(x[i].p(), f.c(), w[pi * n + i].c(), RN);
                        }
                    },
                    |v| Canon::Skip
                );
                conts.push(Cont::rf("MPFR elimination by mpfr_fma", k).loose().note("Gaussian elimination with partial pivoting rounded at each step (ours: each element one dot product rounded once); copies of A and b included"));
            }
            row("matrix", &op, &ps, 0, conts);
        }
    }
}

pub fn cmpabs(a: &MF, b: &MF) -> i32 {
    unsafe extern "C" {
        fn mpfr_cmpabs(a: C, b: C) -> std::os::raw::c_int;
    }
    unsafe { mpfr_cmpabs(a.c(), b.c()) }
}

pub fn complex_rows() {
    for n in [4usize, 8] {
        for p in [53u64, 256] {
            let op = format!("&a * &b complex, {n}x{n}");
            let ps = format!("prec={p}");
            if !wanted("matrix", &op, &ps) {
                continue;
            }
            let l = 2usize;
            let ar: Vec<Vec<(Raw, Raw)>> = (0..l).map(|i| {
                let re = entries(n, p, 51 + i as u64);
                let im = entries(n, p, 61 + i as u64);
                re.into_iter().zip(im).collect()
            }).collect();
            let br: Vec<Vec<(Raw, Raw)>> = (0..l).map(|i| {
                let re = entries(n, p, 71 + i as u64);
                let im = entries(n, p, 81 + i as u64);
                re.into_iter().zip(im).collect()
            }).collect();
            let mut conts = Vec::new();
            {
                let a: Vec<Matrix<BigComplex>> = ar.iter().map(|v| Matrix::from_fn(n, n, |i, j| ours_c(&v[i * n + j].0, &v[i * n + j].1, p))).collect();
                let b: Vec<Matrix<BigComplex>> = br.iter().map(|v| Matrix::from_fn(n, n, |i, j| ours_c(&v[i * n + j].0, &v[i * n + j].1, p))).collect();
                conts.push(Cont::ours(kern!(l, |k| &a[k] * &b[k], |v| c_matc(&v))));
            }
            {
                let mut a: Vec<Vec<MC>> = ar.iter().map(|v| v.iter().map(|(x, y)| mpc(x, y, p)).collect()).collect();
                let mut b: Vec<Vec<MC>> = br.iter().map(|v| v.iter().map(|(x, y)| mpc(x, y, p)).collect()).collect();
                let rows: Vec<Vec<Vec<*mut Mpc>>> = a.iter_mut().map(|m| (0..n).map(|i| (0..n).map(|j| &mut m[i * n + j].0 as *mut Mpc).collect()).collect()).collect();
                let cols: Vec<Vec<Vec<*mut Mpc>>> = b.iter_mut().map(|m| (0..n).map(|j| (0..n).map(|i| &mut m[i * n + j].0 as *mut Mpc).collect()).collect()).collect();
                let mut c: Vec<MC> = (0..n * n).map(|_| MC::new(p)).collect();
                let k = kern!(
                    l,
                    |k| unsafe {
                        let _ = (&a, &b);
                        for i in 0..n {
                            for j in 0..n {
                                mpc_dot(c[i * n + j].p(), rows[k][i].as_ptr(), cols[k][j].as_ptr(), n as _, 0);
                            }
                        }
                    },
                    |v| Canon::Many(c.iter().map(|x| c_mpc(x.c())).collect())
                );
                conts.push(Cont::rf("MPC mpc_dot per element", k).note("each part of the sum rounded once, as ours"));
            }
            row("matrix", &op, &ps, l, conts);
        }
    }
}

pub fn run(g: &str) {
    match g {
        "product" => product_rows(),
        "solve" => solve_rows(),
        "complex" => complex_rows(),
        "debug" => debug_product(),
        _ => panic!("matrix: product|solve|complex"),
    }
}

/// Correctness probe (no timing): which elements of the product differ from mpfr_dot, and
/// whether `BigFloat::dot_round` of the row and column agrees with either.
pub fn debug_product() {
    let (n, p) = (8usize, 113u64);
    let ar = entries(n, p, 11);
    let br = entries(n, p, 21);
    let a = Matrix::from_fn(n, n, |i, j| ours(&ar[i * n + j], p));
    let b = Matrix::from_fn(n, n, |i, j| ours(&br[i * n + j], p));
    let c = &a * &b;
    let mut am: Vec<MF> = ar.iter().map(|r| mpfr(r, p)).collect();
    let mut bm: Vec<MF> = br.iter().map(|r| mpfr(r, p)).collect();
    let mut bad = 0;
    for i in 0..n {
        for j in 0..n {
            let row: Vec<P> = (0..n).map(|t| &mut am[i * n + t].0 as P).collect();
            let col: Vec<P> = (0..n).map(|t| &mut bm[t * n + j].0 as P).collect();
            let mut r = MF::new(p);
            unsafe { mpfr_dot(r.p(), row.as_ptr(), col.as_ptr(), n as _, RN) };
            let xs: Vec<BigFloat> = (0..n).map(|t| a[(i, t)].clone()).collect();
            let ys: Vec<BigFloat> = (0..n).map(|t| b[(t, j)].clone()).collect();
            let d = BigFloat::dot_round(&xs, &ys, p, N).0;
            let (cm, co, cd) = (c_mpfr(r.c()), c_ours(&c[(i, j)]), c_ours(&d));
            if cm != co || cd != co {
                bad += 1;
                if bad <= 4 {
                    println!("({i},{j}) prod {} | dot_round {} | mpfr_dot {} | prod prec {}", co.short(), cd.short(), cm.short(), c[(i, j)].prec());
                }
            }
        }
    }
    println!("{bad} of {} elements differ", n * n);
    // the harness's construction
    let mut a2: Vec<Vec<MF>> = vec![ar.iter().map(|r| mpfr(r, p)).collect()];
    let mut b2: Vec<Vec<MF>> = vec![br.iter().map(|r| mpfr(r, p)).collect()];
    let rows: Vec<Vec<Vec<P>>> = a2.iter_mut().map(|m| (0..n).map(|i| (0..n).map(|j| &mut m[i * n + j].0 as P).collect()).collect()).collect();
    let cols: Vec<Vec<Vec<P>>> = b2.iter_mut().map(|m| (0..n).map(|j| (0..n).map(|i| &mut m[i * n + j].0 as P).collect()).collect()).collect();
    for j in 0..3 {
        let mut r = MF::new(p);
        unsafe { mpfr_dot(r.p(), rows[0][0].as_ptr(), cols[0][j].as_ptr(), n as _, RN) };
        println!("(0,{j}) harness-style {} vs ours {}", c_mpfr(r.c()).short(), c_ours(&c[(0, j)]).short());
        for t in 0..n {
            println!("   col ptr {:?} expected {:?}", cols[0][j][t], &mut b2[0][t * n + j].0 as P);
        }
    }
}
