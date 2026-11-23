//! The PLE decomposition modulo `p`, `A = P·L·E` with `E` in row echelon form (Jeannerod, Pernet
//! and Storjohann, "Rank-profile revealing Gaussian elimination and the CUP matrix decomposition",
//! JSC 2013; Ibarra, Moran and Hui's LSP, 1982), and the triangular solves the other operations
//! build on it.
//!
//! The decomposition halves the columns: the left half is decomposed, its row swaps applied to
//! the whole rows; the top right block becomes `L₁₁⁻¹·A₁₂` by a triangular solve, the bottom right
//! one `A₂₂ - L₂₁·A₁₂` by a product; then the bottom right block is decomposed. A block of at most
//! [`BASE`] columns is eliminated classically, a row at a time with Shoup's products. Most of the
//! work is products, which [`product`](super::product) computes by Strassen–Winograd's recursion
//! when they are large, so the decomposition takes `O(n^ω)` operations, `ω = log₂ 7`; below the
//! crossover, about `n³/3` products of words (and as many sums) for an `n × n` matrix.
//!
//! The result is stored over `A`, row-major, as LAPACK stores its LU factors: row `i < r` holds
//! the row of `E` from its pivot column `q_i` on, and the multiplier `L[i][k]` (`i > k`) is in
//! row `i` of column `q_k`, where `E` has a zero. Everything else is zero.

use super::field::Field;
use super::product::{Mode, product};

/// The number of columns up to which a block is eliminated classically. Measured on an Apple M4
/// Max in the release profile, determinants of random `n × n` matrices for `n` from 64 to 512
/// modulo primes of 20, 62 and 64 bits, with blocks of 16, 32, 48, 64, 96 and 128 columns and
/// with none: 16 was the fastest or within 3% of it, and the classical elimination of the whole
/// matrix took 1.5 to 2.2 times as long at 512.
const BASE: usize = 16;

/// The rows up to which a triangular solve is a substitution, a row at a time: measured as
/// [`BASE`], determinants and inverses, with 8, 16, 32, 64 and 128 rows, 8 and 16 the fastest.
const TRSM_BASE: usize = 16;

/// The decomposition of a matrix stored over it.
#[derive(Clone, Debug)]
pub(in crate::math::matrix) struct Echelon {
    /// The rank `r`.
    pub(in crate::math::matrix) rank: usize,
    /// The pivot columns `q_0 < q_1 < … < q_(r-1)`.
    pub(in crate::math::matrix) pivots: Vec<usize>,
    /// Row `i` of `P⁻¹·A` is row `perm[i]` of `A`.
    pub(in crate::math::matrix) perm: Vec<usize>,
    /// Whether `P` is an odd permutation.
    pub(in crate::math::matrix) odd: bool,
}

/// The state of a decomposition: the field, the distance between rows, and the permutation and
/// the pivots found so far.
struct Ple {
    f: Field,
    ld: usize,
    perm: Vec<usize>,
    pivots: Vec<usize>,
    odd: bool,
}

/// Swaps the whole rows `i` and `j` of the row-major `a` of rows `ld` long.
fn swap_rows(a: &mut [u64], ld: usize, i: usize, j: usize) {
    let (i, j) = (i.min(j), i.max(j));
    let (x, y) = a.split_at_mut(j * ld);
    x[i * ld..(i + 1) * ld].swap_with_slice(&mut y[..ld]);
}

impl Ple {
    /// Decomposes the block of the rows `r0..` and the columns `c0..c1`, its row swaps applied to
    /// whole rows: its rank.
    fn run(&mut self, a: &mut [u64], r0: usize, c0: usize, c1: usize) -> usize {
        let m = a.len() / self.ld;
        if r0 == m || c0 == c1 {
            return 0;
        }
        if c1 - c0 <= BASE {
            return self.classical(a, r0, c0, c1);
        }
        let mid = c0 + (c1 - c0) / 2;
        let start = self.pivots.len();
        let r1 = self.run(a, r0, c0, mid);
        if r1 > 0 {
            let (f, ld) = (self.f, self.ld);
            let piv = &self.pivots[start..start + r1];
            let l11 = gather(a, ld, r0, r0 + r1, piv);
            let (top, bottom) = a.split_at_mut((r0 + r1) * ld);
            let x = &mut top[r0 * ld + mid..];
            lower_unit(f, &l11, r1, x, ld, r1, c1 - mid);
            let below = m - r0 - r1;
            if below > 0 {
                let l21 = gather(bottom, ld, 0, below, piv);
                let x = &top[r0 * ld + mid..];
                product(
                    f,
                    &mut bottom[mid..],
                    ld,
                    &l21,
                    r1,
                    x,
                    ld,
                    (below, r1, c1 - mid),
                    Mode::Sub,
                );
            }
        }
        let r2 = self.run(a, r0 + r1, mid, c1);
        r1 + r2
    }

    /// Gaussian elimination of the block of the rows `r0..` and the columns `c0..c1`, a column
    /// at a time: the first row with an element in the column is the pivot row, the rows below
    /// lose their multiples of it, and the multipliers take the zeros' places. Its rank.
    fn classical(&mut self, a: &mut [u64], r0: usize, c0: usize, c1: usize) -> usize {
        let (f, ld) = (self.f, self.ld);
        let m = a.len() / ld;
        let mut r = r0;
        for j in c0..c1 {
            if r == m {
                break;
            }
            let Some(i) = (r..m).find(|&i| a[i * ld + j] != 0) else {
                continue;
            };
            if i != r {
                swap_rows(a, ld, i, r);
                self.perm.swap(i, r);
                self.odd = !self.odd;
            }
            let inv = f.inv(a[r * ld + j]);
            let invs = f.shoup(inv);
            let (top, rest) = a.split_at_mut((r + 1) * ld);
            let pivot = &top[r * ld + j + 1..r * ld + c1];
            for row in rest.chunks_exact_mut(ld) {
                let x = row[j];
                if x == 0 {
                    continue;
                }
                let l = f.mul_shoup(x, inv, invs);
                row[j] = l;
                f.sub_mul(&mut row[j + 1..c1], pivot, l);
            }
            self.pivots.push(j);
            r += 1;
        }
        r - r0
    }
}

/// The decomposition of the row-major `rows × cols` matrix `a`, stored over it.
pub(in crate::math::matrix) fn decompose(
    f: Field,
    a: &mut [u64],
    rows: usize,
    cols: usize,
) -> Echelon {
    debug_assert_eq!(a.len(), rows * cols);
    let mut s = Ple {
        f,
        ld: cols,
        perm: (0..rows).collect(),
        pivots: Vec::new(),
        odd: false,
    };
    let rank = if rows == 0 || cols == 0 {
        0
    } else {
        s.run(a, 0, 0, cols)
    };
    Echelon {
        rank,
        pivots: s.pivots,
        perm: s.perm,
        odd: s.odd,
    }
}

/// The elements of the rows `r0..r1` of `a` (rows `ld` long) in the columns `cols`, as a
/// contiguous row-major matrix.
pub(in crate::math::matrix) fn gather(
    a: &[u64],
    ld: usize,
    r0: usize,
    r1: usize,
    cols: &[usize],
) -> Vec<u64> {
    let mut out = Vec::with_capacity((r1 - r0) * cols.len());
    for i in r0..r1 {
        let row = &a[i * ld..];
        out.extend(cols.iter().map(|&c| row[c]));
    }
    out
}

/// `X ← L⁻¹·X` for the `r × r` lower triangular `l` with a unit diagonal (row-major, rows `ldl`
/// long; only the part below the diagonal is read) and the `r × w` view `x`: halved down to
/// [`TRSM_BASE`] rows, the work off the diagonal a product.
pub(in crate::math::matrix) fn lower_unit(
    f: Field,
    l: &[u64],
    ldl: usize,
    x: &mut [u64],
    ldx: usize,
    r: usize,
    w: usize,
) {
    if r == 0 || w == 0 {
        return;
    }
    if r <= TRSM_BASE {
        for i in 1..r {
            let (above, below) = x.split_at_mut(i * ldx);
            let xi = &mut below[..w];
            for k in 0..i {
                f.sub_mul(xi, &above[k * ldx..k * ldx + w], l[i * ldl + k]);
            }
        }
        return;
    }
    let h = r / 2;
    lower_unit(f, l, ldl, x, ldx, h, w);
    let (x1, x2) = x.split_at_mut(h * ldx);
    product(
        f,
        x2,
        ldx,
        &l[h * ldl..],
        ldl,
        x1,
        ldx,
        (r - h, h, w),
        Mode::Sub,
    );
    lower_unit(f, &l[h * ldl + h..], ldl, x2, ldx, r - h, w);
}

/// `X ← U⁻¹·X` for the `r × r` upper triangular `u` with a unit diagonal (row-major, rows `ldu`
/// long; only the part above the diagonal is read) and the `r × w` view `x`, as [`lower_unit`].
pub(in crate::math::matrix) fn upper_unit(
    f: Field,
    u: &[u64],
    ldu: usize,
    x: &mut [u64],
    ldx: usize,
    r: usize,
    w: usize,
) {
    if r == 0 || w == 0 {
        return;
    }
    if r <= TRSM_BASE {
        for i in (0..r - 1).rev() {
            let (above, below) = x.split_at_mut((i + 1) * ldx);
            let xi = &mut above[i * ldx..i * ldx + w];
            for k in i + 1..r {
                let off = (k - i - 1) * ldx;
                f.sub_mul(xi, &below[off..off + w], u[i * ldu + k]);
            }
        }
        return;
    }
    let h = r / 2;
    let (x1, x2) = x.split_at_mut(h * ldx);
    upper_unit(f, &u[h * ldu + h..], ldu, x2, ldx, r - h, w);
    product(f, x1, ldx, &u[h..], ldu, x2, ldx, (h, r - h, w), Mode::Sub);
    upper_unit(f, u, ldu, x1, ldx, h, w);
}
