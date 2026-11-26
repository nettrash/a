//! Kernel experiments for phase 2 (not part of the scoreboard): variants of the CSR products on
//! raw arrays, raced against each other.

use crate::mats::{self, Csr};
use crate::timing::{Contender, keep, note, race};

fn v0(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64]) {
    for (i, y) in y.iter_mut().enumerate() {
        let mut sum = 0.0;
        for k in p[i]..p[i + 1] {
            sum += v[k] * x[c[k]];
        }
        *y = sum;
    }
}

fn v1(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64]) {
    for (i, y) in y.iter_mut().enumerate() {
        let (a, b) = (p[i], p[i + 1]);
        let mut sum = 0.0;
        for (&j, &a) in c[a..b].iter().zip(&v[a..b]) {
            sum += a * unsafe { *x.get_unchecked(j) };
        }
        *y = sum;
    }
}

fn v1w(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64]) {
    // The row bounds from windows, the row slices without checks.
    for (w, y) in p.windows(2).zip(y.iter_mut()) {
        let (a, b) = (w[0], w[1]);
        let (cs, vs) = unsafe { (c.get_unchecked(a..b), v.get_unchecked(a..b)) };
        let mut sum = 0.0;
        for (&j, &a) in cs.iter().zip(vs) {
            sum += a * unsafe { *x.get_unchecked(j) };
        }
        *y = sum;
    }
}

fn v4(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64]) {
    for (w, y) in p.windows(2).zip(y.iter_mut()) {
        let (a, b) = (w[0], w[1]);
        let (cs, vs) = unsafe { (c.get_unchecked(a..b), v.get_unchecked(a..b)) };
        let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
        let mut cc = cs.chunks_exact(4);
        let mut vc = vs.chunks_exact(4);
        for (ci, vi) in (&mut cc).zip(&mut vc) {
            unsafe {
                s0 += vi[0] * *x.get_unchecked(ci[0]);
                s1 += vi[1] * *x.get_unchecked(ci[1]);
                s2 += vi[2] * *x.get_unchecked(ci[2]);
                s3 += vi[3] * *x.get_unchecked(ci[3]);
            }
        }
        for (&j, &a) in cc.remainder().iter().zip(vc.remainder()) {
            s0 += a * unsafe { *x.get_unchecked(j) };
        }
        *y = (s0 + s1) + (s2 + s3);
    }
}

fn s0(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for (i, &xi) in x.iter().enumerate() {
        for k in p[i]..p[i + 1] {
            y[c[k]] += v[k] * xi;
        }
    }
}

fn s1(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for (w, &xi) in p.windows(2).zip(x) {
        let (a, b) = (w[0], w[1]);
        let (cs, vs) = unsafe { (c.get_unchecked(a..b), v.get_unchecked(a..b)) };
        for (&j, &a) in cs.iter().zip(vs) {
            unsafe { *y.get_unchecked_mut(j) += a * xi };
        }
    }
}

/// SpMM, column-major X (n × k) and Y (m × k): column by column (the current way).
fn mm0(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64], n: usize, m: usize, k: usize) {
    for col in 0..k {
        v1w(p, c, v, &x[col * n..(col + 1) * n], &mut y[col * m..(col + 1) * m]);
    }
}

/// SpMM row-blocked: a row's elements read once for B columns at a time.
fn mmb<const B: usize>(p: &[usize], c: &[usize], v: &[f64], x: &[f64], y: &mut [f64], n: usize, m: usize, k: usize) {
    let mut col = 0;
    while col + B <= k {
        for i in 0..m {
            let (a, b) = (p[i], p[i + 1]);
            let (cs, vs) = unsafe { (c.get_unchecked(a..b), v.get_unchecked(a..b)) };
            let mut acc = [0.0f64; B];
            for (&j, &a) in cs.iter().zip(vs) {
                for t in 0..B {
                    acc[t] += a * unsafe { *x.get_unchecked((col + t) * n + j) };
                }
            }
            for t in 0..B {
                y[(col + t) * m + i] = acc[t];
            }
        }
        col += B;
    }
    for col in col..k {
        v1w(p, c, v, &x[col * n..(col + 1) * n], &mut y[col * m..(col + 1) * m]);
    }
}

pub fn run() {
    for name in ["lap2d-100", "lap2d-1000", "elast3d-30", "randunsym-1000000"] {
        let a = mats::by_name(name);
        spmv(name, &a);
    }
    for name in ["lap2d-316", "randunsym-100000"] {
        let a = mats::by_name(name);
        for k in [4usize, 16, 64] {
            spmm(name, &a, k);
        }
    }
}

fn spmv(name: &str, a: &Csr) {
    note(&format!("kern spmv {name}"));
    let n = a.nrows;
    let x = mats::rand_vec(n, 1);
    let (p, c, v) = (&a.ptr[..], &a.idx[..], &a.val[..]);
    let mut want = vec![0.0; n];
    v0(p, c, v, &x, &mut want);
    let mut y = vec![0.0; n];
    for (f, exact) in [(v1 as fn(&[usize], &[usize], &[f64], &[f64], &mut [f64]), true), (v1w, true), (v4, false)] {
        f(p, c, v, &x, &mut y);
        let d = want.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(if exact { d == 0.0 } else { d < 1e-12 }, "variant differs {d}");
    }
    let mut cs: Vec<Contender> = Vec::new();
    for (label, f) in [("v0 current", v0 as fn(&[usize], &[usize], &[f64], &[f64], &mut [f64])), ("v1 zip+unchecked x", v1), ("v1w windows", v1w), ("v4 4 accumulators", v4), ("s0 scatter current", s0), ("s1 scatter unchecked", s1)] {
        let mut y = vec![0.0; n];
        let x = &x;
        cs.push(Contender::new(label, move || {
            f(p, c, v, x, &mut y);
            keep(&y);
        }));
    }
    let st = race(&mut cs);
    for s in &st {
        println!("kern\tspmv\t{name}\t{}\t{:.0}\t{:.3}", s.name, s.median, s.median / st[0].median);
    }
}

fn spmm(name: &str, a: &Csr, k: usize) {
    note(&format!("kern spmm {name} {k}"));
    let (m, n) = (a.nrows, a.ncols);
    let x = mats::rand_vec(n * k, 2);
    let (p, c, v) = (&a.ptr[..], &a.idx[..], &a.val[..]);
    let mut want = vec![0.0; m * k];
    mm0(p, c, v, &x, &mut want, n, m, k);
    let mut y = vec![0.0; m * k];
    mmb::<4>(p, c, v, &x, &mut y, n, m, k);
    assert!(want == y);
    mmb::<8>(p, c, v, &x, &mut y, n, m, k);
    assert!(want == y);
    let mut cs: Vec<Contender> = Vec::new();
    type F = fn(&[usize], &[usize], &[f64], &[f64], &mut [f64], usize, usize, usize);
    for (label, f) in [("mm0 by columns", mm0 as F), ("mmb4", mmb::<4> as F), ("mmb8", mmb::<8> as F)] {
        let mut y = vec![0.0; m * k];
        let x = &x;
        cs.push(Contender::new(label, move || {
            f(p, c, v, x, &mut y, n, m, k);
            keep(&y);
        }));
    }
    let st = race(&mut cs);
    for s in &st {
        println!("kern\tspmm k={k}\t{name}\t{}\t{:.0}\t{:.3}", s.name, s.median, s.median / st[0].median);
    }
}
