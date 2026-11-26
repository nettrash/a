//! Products modulo primes below `2^32` by the floating-point product of
//! [`gemm`](super::super::blas::gemm), exactly: residues small enough that every sum of products
//! stays below `2^53` are multiplied as doubles, each sum then an integer the double holds, at
//! the speed of the `f64` kernels; reduced modulo `p` once. Residues the bound does not allow
//! whole are split into halves of 16 bits, `x = x₁·2^16 + x₀`, and the product is Karatsuba's
//! three, `A₁B₁`, `A₀B₀` and `(A₁ + A₀)(B₁ + B₀)`, whose sums are below `2^34·k`. The inner
//! dimension goes in runs short enough for the bound.

use super::super::blas::{Op, gemm};
use super::field::{Field, Width};
use super::product::Mode;

/// The largest sum the doubles hold exactly.
const EXACT: u64 = 1 << 53;

/// The least `min(m, k, n)` from which the doubles are used: for whole residues; for halves
/// (three products and the copies) where the integer kernels hold a sum of `k` products in one
/// word; and for halves where they need two, which makes them slower. Measured on an Apple M4
/// Max in the release profile, the doubles against the integer kernels and their
/// Strassen–Winograd recursion for primes of 16 to 32 bits, `min(m, k, n)` from 8 to 128 on
/// square and thin shapes and up to 4096 rows: whole residues take 0.95 of the time at 13 rows
/// and 0.76 at 16 (1.01 to 1.04 at 12), thin shapes less; halves 0.96 to 0.99 at 64 (1.03 at 60)
/// for primes of 24 to 29 bits, but 0.94 to 0.97 at 24 already (1.02 to 1.03 at 22) for primes of
/// 30 to 32 bits; at 4096 rows whole residues 0.30 and halves (31 bits) 0.73 of the recursion's
/// time, so there is no upper edge. Against the earlier 32 and 64 for both kinds of halves, PLE,
/// rank, `solve` and `inverse` took 0.87 to 0.99 of the time from 32 rows, products 0.59 to 0.98,
/// and nothing more than 1.005.
const FROM_WHOLE: usize = 13;
const FROM_HALVES: usize = 64;
const FROM_HALVES_WIDE: usize = 24;

/// `C (=, +=, -=) A·B` for row-major views modulo a prime below `2^32`, by doubles; `false`,
/// with `c` untouched, when the prime or the shape is not for them.
#[allow(clippy::too_many_arguments)]
pub(super) fn product(
    f: Field,
    c: &mut [u64],
    ldc: usize,
    a: &[u64],
    lda: usize,
    b: &[u64],
    ldb: usize,
    (m, k, n): (usize, usize, usize),
    mode: Mode,
) -> bool {
    let p = f.p;
    if p >= 1 << 32 {
        return false;
    }
    let q = (p - 1) * (p - 1);
    // Whole residues: (p − 1)²·run ≤ 2^53.
    let whole = EXACT / q.max(1);
    // Halves: (2^17 − 2)²·run ≤ 2^53 for the sums of halves.
    let halves = EXACT / ((1u64 << 17) - 2).pow(2);
    let split = (whole as usize) < k.min(256);
    let from = if !split {
        FROM_WHOLE
    } else if f.width(k) == Width::One {
        FROM_HALVES
    } else {
        FROM_HALVES_WIDE
    };
    if m.min(k).min(n) < from {
        return false;
    }
    let run = if split {
        halves as usize
    } else {
        whole as usize
    }
    .clamp(1, k);
    // Column-major doubles: Aᵀ (k × m) is row-major A, Bᵀ (n × k) row-major B; Cᵀ = Bᵀ·Aᵀ.
    let mut t = vec![0u64; m * n];
    let mut first = true;
    let mut p0 = 0;
    while p0 < k {
        let r = run.min(k - p0);
        let part = if split {
            halves_product(a, lda, b, ldb, (m, r, n), p0, f)
        } else {
            whole_product(a, lda, b, ldb, (m, r, n), p0, f)
        };
        for (x, y) in t.iter_mut().zip(part) {
            *x = if first { y } else { f.add(*x, y) };
        }
        first = false;
        p0 += r;
    }
    for i in 0..m {
        let cr = &mut c[i * ldc..i * ldc + n];
        let tr = &t[i * n..(i + 1) * n];
        for (x, &y) in cr.iter_mut().zip(tr) {
            *x = match mode {
                Mode::Set => y,
                Mode::Add => f.add(*x, y),
                Mode::Sub => f.sub(*x, y),
            };
        }
    }
    true
}

/// The doubles of `x(v)` for the `rows × cols` block, column-major.
fn doubles(rows: usize, cols: usize, x: impl Fn(usize, usize) -> u64) -> Vec<f64> {
    let mut d = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            d.push(x(i, j) as f64);
        }
    }
    d
}

/// `Cᵀ = Bᵀ·Aᵀ` in doubles for column-major blocks: `bt` of `n × r`, `at` of `r × m`.
fn gemm_t(bt: &[f64], at: &[f64], (m, r, n): (usize, usize, usize)) -> Vec<f64> {
    let mut ct = vec![0.0f64; n * m];
    gemm(n, m, r, 1.0, bt, n, Op::N, at, r, Op::N, 0.0, &mut ct, n);
    ct
}

/// The residues of the row-major `m × n` product, `w(e)` of the element `e` of the
/// column-major `n × m` blocks of exact sums: the same index, `Cᵀ` column-major being `C`
/// row-major.
fn residues((m, n): (usize, usize), w: impl Fn(usize) -> u64) -> Vec<u64> {
    (0..m * n).map(w).collect()
}

/// The product of the residues `a[.., p0..p0 + r]` and `b[p0..p0 + r, ..]`, whole.
fn whole_product(
    a: &[u64],
    lda: usize,
    b: &[u64],
    ldb: usize,
    (m, r, n): (usize, usize, usize),
    p0: usize,
    f: Field,
) -> Vec<u64> {
    // Aᵀ (r × m) column-major: element (p, i) = a[i·lda + p0 + p].
    let at = doubles(r, m, |p, i| a[i * lda + p0 + p]);
    // Bᵀ (n × r) column-major: element (j, p) = b[(p0 + p)·ldb + j].
    let bt = doubles(n, r, |j, p| b[(p0 + p) * ldb + j]);
    let ct = gemm_t(&bt, &at, (m, r, n));
    residues((m, n), |e| f.reduce(ct[e] as u64))
}

/// The same product by halves of 16 bits, Karatsuba's three products.
fn halves_product(
    a: &[u64],
    lda: usize,
    b: &[u64],
    ldb: usize,
    (m, r, n): (usize, usize, usize),
    p0: usize,
    f: Field,
) -> Vec<u64> {
    let lo = |x: u64| x & 0xffff;
    let hi = |x: u64| x >> 16;
    let a_at = |p: usize, i: usize| a[i * lda + p0 + p];
    let b_at = |j: usize, p: usize| b[(p0 + p) * ldb + j];
    let a1 = doubles(r, m, |p, i| hi(a_at(p, i)));
    let a0 = doubles(r, m, |p, i| lo(a_at(p, i)));
    let b1 = doubles(n, r, |j, p| hi(b_at(j, p)));
    let b0 = doubles(n, r, |j, p| lo(b_at(j, p)));
    let hh = gemm_t(&b1, &a1, (m, r, n));
    let ll = gemm_t(&b0, &a0, (m, r, n));
    let asum: Vec<f64> = a1.iter().zip(&a0).map(|(x, y)| x + y).collect();
    let bsum: Vec<f64> = b1.iter().zip(&b0).map(|(x, y)| x + y).collect();
    let mid = gemm_t(&bsum, &asum, (m, r, n));
    residues((m, n), |e| {
        let (h, l) = (hh[e] as u64 as u128, ll[e] as u64 as u128);
        // (A₁ + A₀)(B₁ + B₀) − A₁B₁ − A₀B₀ = A₁B₀ + A₀B₁, below 2^34·r.
        let cross = mid[e] as u64 as u128 - h - l;
        f.reduce_wide((h << 32) + (cross << 16) + l)
    })
}
