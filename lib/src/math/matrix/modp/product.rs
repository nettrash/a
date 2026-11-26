//! Products of matrices modulo `p`, on row-major views (a slice and the distance between its
//! rows): the classical product as dot products of the rows of `A` with the rows of `Bᵀ`, packed
//! once, each sum of products held unreduced in one, two or three words and reduced once; above
//! [`CROSSOVER`], Strassen–Winograd's recursion ([`strassen`](super::super::strassen)) on the
//! classical product.

use super::super::strassen::{self, Ring};
use super::field::{Field, Width};

/// The order from which a product is split into Strassen–Winograd's seven: `min(m, k, n)` of at
/// least this. Measured on an Apple M4 Max in the release profile, `n × n` by `n × n` products
/// for `n` of 512 and 1024 modulo primes of 20, 31, 62 and 64 bits, with crossovers of 64, 96,
/// 128, 160, 192 and 256: 96 and 128 were the fastest or within noise of it, and the recursion
/// took 0.73 to 0.78 of the classical product's time at 1024; one level alone at 128 was 0.85 to
/// 1.01 of it. Both give the same residues.
pub(in crate::math::matrix) const CROSSOVER: usize = 128;

/// What a product does with the matrix it is written to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::math::matrix) enum Mode {
    /// `C = A·B`.
    Set,
    /// `C = C + A·B`.
    Add,
    /// `C = C - A·B`.
    Sub,
}

/// `C (=, +=, -=) A·B` for the `m × k` view `a`, the `k × n` view `b` and the `m × n` view `c`,
/// classically: `Bᵀ` packed, then each element one dot product of words reduced once.
#[allow(clippy::too_many_arguments)]
pub(in crate::math::matrix) fn classical(
    f: Field,
    c: &mut [u64],
    ldc: usize,
    a: &[u64],
    lda: usize,
    b: &[u64],
    ldb: usize,
    (m, k, n): (usize, usize, usize),
    mode: Mode,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if mode == Mode::Set {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0);
            }
        }
        return;
    }
    // The columns of `B` as rows, so that each dot product reads two runs of memory.
    let mut bt = vec![0u64; n * k];
    for p in 0..k {
        let row = &b[p * ldb..p * ldb + n];
        for (j, &x) in row.iter().enumerate() {
            bt[j * k + p] = x;
        }
    }
    match f.width(k) {
        Width::One => rows(f, c, ldc, a, lda, &bt, (m, k, n), mode, Field::dot1),
        Width::Two => rows(f, c, ldc, a, lda, &bt, (m, k, n), mode, Field::dot2),
        Width::Three => rows(f, c, ldc, a, lda, &bt, (m, k, n), mode, Field::dot3),
    }
}

/// The loops of [`classical`] for one width of the sums, its dot product `dot`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn rows(
    f: Field,
    c: &mut [u64],
    ldc: usize,
    a: &[u64],
    lda: usize,
    bt: &[u64],
    (m, k, n): (usize, usize, usize),
    mode: Mode,
    dot: impl Fn(Field, &[u64], &[u64]) -> u64,
) {
    for i in 0..m {
        let ar = &a[i * lda..i * lda + k];
        let cr = &mut c[i * ldc..i * ldc + n];
        for (x, bj) in cr.iter_mut().zip(bt.chunks_exact(k)) {
            let d = dot(f, ar, bj);
            *x = match mode {
                Mode::Set => d,
                Mode::Add => f.add(*x, d),
                Mode::Sub => f.sub(*x, d),
            };
        }
    }
}

/// The ring of residues modulo `p` for Strassen–Winograd's recursion.
struct Residues(Field);

impl Ring for Residues {
    type E = u64;

    fn zero(&self) -> u64 {
        0
    }

    fn add(&self, a: &u64, b: &u64) -> u64 {
        self.0.add(*a, *b)
    }

    fn sub(&self, a: &u64, b: &u64) -> u64 {
        self.0.sub(*a, *b)
    }

    fn classical(
        &self,
        c: &mut [u64],
        ldc: usize,
        a: &[u64],
        lda: usize,
        b: &[u64],
        ldb: usize,
        shape: (usize, usize, usize),
        accumulate: bool,
    ) {
        let mode = if accumulate { Mode::Add } else { Mode::Set };
        classical(self.0, c, ldc, a, lda, b, ldb, shape, mode);
    }

    fn split(&self, m: usize, k: usize, n: usize) -> bool {
        m.min(k).min(n) >= CROSSOVER
    }
}

/// `C (=, +=, -=) A·B` for views, as [`classical`], by Strassen–Winograd's recursion when the
/// three dimensions reach [`CROSSOVER`]; the residues are the classical product's.
#[allow(clippy::too_many_arguments)]
pub(in crate::math::matrix) fn product(
    f: Field,
    c: &mut [u64],
    ldc: usize,
    a: &[u64],
    lda: usize,
    b: &[u64],
    ldb: usize,
    (m, k, n): (usize, usize, usize),
    mode: Mode,
) {
    // Primes below 2^32: exactly in doubles.
    if super::doubles::product(f, c, ldc, a, lda, b, ldb, (m, k, n), mode) {
        return;
    }
    if m.min(k).min(n) < CROSSOVER {
        classical(f, c, ldc, a, lda, b, ldb, (m, k, n), mode);
        return;
    }
    let r = Residues(f);
    if mode == Mode::Set {
        strassen::product(&r, c, ldc, a, lda, b, ldb, (m, k, n));
        return;
    }
    let mut t = vec![0u64; m * n];
    strassen::product(&r, &mut t, n, a, lda, b, ldb, (m, k, n));
    for (i, t) in t.chunks_exact(n).enumerate() {
        let cr = &mut c[i * ldc..i * ldc + n];
        for (x, &y) in cr.iter_mut().zip(t) {
            *x = if mode == Mode::Add {
                f.add(*x, y)
            } else {
                f.sub(*x, y)
            };
        }
    }
}
