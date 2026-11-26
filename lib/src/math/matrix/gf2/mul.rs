//! Products over `GF(2)`: M4RM, the method of the Four Russians (Arlazarov, Dinic, Kronrod and
//! Faradzev, 1970), as Albrecht, Bard and Hart arrange it ("Algorithm 898: Efficient
//! multiplication of dense matrices over GF(2)", ACM TOMS 37, 2010), and Strassen's method in
//! Winograd's form above it for large matrices.
//!
//! For `C = A·B`, M4RM takes the rows of `B` in groups of `k`. The `2^k` sums of a group are
//! built into a table in Gray-code order, each entry the one before it plus one row; row `i` of
//! `C` then gains the entry that the `k` bits of row `i` of `A` in the group's columns select.
//! A group costs `2^k + m` row additions where the schoolbook product takes `k·m/2`, so with
//! `k = ⌊log₂ m⌋ − 1`, at most 8, a product of `n × n` matrices is about `n³ / (64·k)` word
//! operations.
//! As in the paper, several tables (here `TABLES`) are built at once, so that each row of `C`
//! is visited once for all of them, and the columns of `B` and `C` are taken in blocks narrow
//! enough that the tables stay in the caches (here the M4's, 4 MiB of tables).

use super::BitMatrix;

/// Strassen–Winograd from this many rows, inner columns and columns, all three, so that no half
/// is below 4096; below it M4RM alone. Measured on an Apple M4 Max with the M4RM of this file:
/// one level took 1.36 of M4RM's time at 2048, 1.13 at 3072 and 4096, 0.99 at 6144 and 0.91 at
/// 8192; whole products from 4096 to 16384 rows, square and not, with the recursion from 3072,
/// 6144, 7168, 8192 and 10240, twice: 6144 to 8192 alike (0.86 to 0.91 of the time from 3072,
/// 1.00 for 8192 × 16384 · 16384 × 8192), 8192 the best by a little. The halves are taken at
/// multiples of 64 columns so that no word is shifted.
const STRASSEN_FROM: usize = 8192;

/// The tables built at once. Measured on an Apple M4 Max with the group size below, against 4 and
/// 16 tables (each with a loop over them in place of `combine8`): 4 took 0.86 to 0.89 of the time
/// of 8 at 64 and 128 rows but 1.07 to 1.12 from 512 to 2048, 16 took 1.09 to 1.36.
const TABLES: usize = 8;

/// The words of all the tables together, 4 MiB, which sets the width of the blocks of columns.
/// Measured on an Apple M4 Max, `2^13` to `2^21` words: a block fewer costs more than the tables'
/// leaving the caches, as the bits of `A` are read and the rows of `C` visited once a block;
/// against `2^16`, `2^19` took 0.88 of the time at 4096 rows and columns, 0.78 at 8192, 0.44 at
/// 16384 and 0.74 to 0.75 for 32768 and 65536 columns, where `2^20` and `2^21` took 0.80 to 0.82;
/// `2^15` and less took up to 2.05 times as long; it changes nothing below 2048 columns.
const TABLE_WORDS: usize = 1 << 19;

/// `a · b`, by Strassen–Winograd from `STRASSEN_FROM`.
pub(super) fn product(a: &BitMatrix, b: &BitMatrix) -> BitMatrix {
    product_from(a, b, STRASSEN_FROM)
}

/// `a · b`, by Strassen–Winograd while all three dimensions are at least `cutoff` (at least 128,
/// so that every half has a word), by M4RM below.
pub(super) fn product_from(a: &BitMatrix, b: &BitMatrix, cutoff: usize) -> BitMatrix {
    debug_assert_eq!(a.cols, b.rows);
    debug_assert!(cutoff >= 128);
    if a.rows >= cutoff && a.cols >= cutoff && b.cols >= cutoff {
        return strassen_winograd(a, b, cutoff);
    }
    let mut c = BitMatrix::zeros(a.rows, b.cols);
    add_product(&mut c, a, b);
    c
}

/// The group size for `m` rows of `A`: `⌊log₂ m⌋ − 1`, from 1 to 8, so that building a table
/// costs about half the additions that use it. Measured on an Apple M4 Max with the offsets 0
/// to 3 and the caps 6 to 10: smaller groups take 0.63 to 0.80 of the time from 128 to 1024 rows,
/// but every other offset or cap is slower somewhere by more than 2 % (1.18 at 32 rows for an
/// offset of 2, 1.08 at 64 for 3, 1.06 to 1.18 for a cap of 6 or 7 on 4096 rows by 256 columns).
fn group_size(m: usize) -> usize {
    (m.max(1).ilog2() as usize).saturating_sub(1).clamp(1, 8)
}

/// `c += a · b` by M4RM, or by the schoolbook method (a row of `b` added for each 1 of a row of
/// `a`) when the groups would be of one row.
fn add_product(c: &mut BitMatrix, a: &BitMatrix, b: &BitMatrix) {
    let (m, l) = (a.rows, a.cols);
    if m == 0 || l == 0 || b.cols == 0 {
        return;
    }
    let k = group_size(m);
    if k == 1 {
        schoolbook(c, a, b);
    } else {
        m4rm(c, a, b, k);
    }
}

/// `c += a · b`, a row of `b` for each 1 of a row of `a`.
fn schoolbook(c: &mut BitMatrix, a: &BitMatrix, b: &BitMatrix) {
    let sb = b.stride;
    for i in 0..a.rows {
        let crow = &mut c.data[i * sb..(i + 1) * sb];
        for (w, &word) in a.data[i * a.stride..(i + 1) * a.stride].iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let j = 64 * w + bits.trailing_zeros() as usize;
                for (x, y) in crow.iter_mut().zip(&b.data[j * sb..(j + 1) * sb]) {
                    *x ^= y;
                }
                bits &= bits - 1;
            }
        }
    }
}

/// The `k` bits of a row of `A` from column `j`, the row's words starting at `data[at]`.
#[inline]
fn bits(data: &[u64], at: usize, j: usize, k: usize) -> usize {
    let (w, s) = (at + j / 64, j % 64);
    let mut v = data[w] >> s;
    if s + k > 64 {
        v |= data[w + 1] << (64 - s);
    }
    (v & ((1u64 << k) - 1)) as usize
}

/// `c += a · b` by M4RM with groups of `k` rows of `b`.
fn m4rm(c: &mut BitMatrix, a: &BitMatrix, b: &BitMatrix, k: usize) {
    let (m, l, sa, sb) = (a.rows, a.cols, a.stride, b.stride);
    // The blocks of columns: as wide as the tables' room allows, at least a word.
    let block = (TABLE_WORDS / (TABLES << k)).clamp(1, sb);
    let mut tables = vec![0u64; TABLES * (block << k)];
    for w0 in (0..sb).step_by(block) {
        let width = block.min(sb - w0);
        let size = width << k;
        for j0 in (0..l).step_by(TABLES * k) {
            // The groups `j0 + t·k ..` of this pass, the last perhaps shorter.
            let mut groups = [(0usize, 0usize); TABLES];
            let mut count = 0;
            for (t, group) in groups.iter_mut().enumerate() {
                let start = j0 + t * k;
                if start >= l {
                    break;
                }
                let kk = k.min(l - start);
                *group = (start, kk);
                count += 1;
                // The table: entry `v` the sum of the rows `start + i` of `b` for the bits `i`
                // of `v`, built in Gray-code order, `g ^ (g >> 1)` after `g − 1`'s, which
                // differs from it in bit `trailing_zeros(g)` alone.
                let table = &mut tables[t * size..(t + 1) * size];
                table[..width].fill(0);
                let mut index = 0;
                for g in 1..1usize << kk {
                    let i = g.trailing_zeros() as usize;
                    let previous = index;
                    index ^= 1 << i;
                    let row = (start + i) * sb + w0;
                    for w in 0..width {
                        table[index * width + w] = table[previous * width + w] ^ b.data[row + w];
                    }
                }
            }
            for i in 0..m {
                let crow = &mut c.data[i * sb + w0..i * sb + w0 + width];
                if count == TABLES {
                    // The eight entries the row selects (entry 0, of zeros, where its bits are
                    // zero), added to the row in one pass: one load and store of the row a word
                    // for the eight, the sums of words in vector registers.
                    let e: [&[u64]; TABLES] = core::array::from_fn(|t| {
                        let (start, kk) = groups[t];
                        let v = bits(&a.data, i * sa, start, kk);
                        &tables[t * size + v * width..t * size + (v + 1) * width]
                    });
                    combine8(crow, &e);
                    continue;
                }
                for (t, &(start, kk)) in groups[..count].iter().enumerate() {
                    let v = bits(&a.data, i * sa, start, kk);
                    if v != 0 {
                        let entry = &tables[t * size + v * width..t * size + (v + 1) * width];
                        for (x, y) in crow.iter_mut().zip(entry) {
                            *x ^= y;
                        }
                    }
                }
            }
        }
    }
}

/// `row ^= e₀ ^ e₁ ^ … ^ e₇` for eight rows of the row's length.
#[inline(always)]
fn combine8(row: &mut [u64], e: &[&[u64]; TABLES]) {
    let w = row.len();
    let (e0, e1, e2, e3) = (&e[0][..w], &e[1][..w], &e[2][..w], &e[3][..w]);
    let (e4, e5, e6, e7) = (&e[4][..w], &e[5][..w], &e[6][..w], &e[7][..w]);
    for x in 0..w {
        row[x] ^= (e0[x] ^ e1[x]) ^ (e2[x] ^ e3[x]) ^ ((e4[x] ^ e5[x]) ^ (e6[x] ^ e7[x]));
    }
}

/// `a · b` by Strassen's method in Winograd's form, seven products of halves and fifteen sums,
/// on the leading part whose dimensions are multiples of 128; the rows and columns beyond it
/// (fewer than 128 of each) are peeled off and done by M4RM.
fn strassen_winograd(a: &BitMatrix, b: &BitMatrix, cutoff: usize) -> BitMatrix {
    let (m, l, n) = (a.rows, a.cols, b.cols);
    // The halves: rows, and words of the inner dimension and of the columns.
    let (m2, l2, n2) = (m / 128 * 64, l / 128, n / 128);
    let (a11, a12) = (
        a.block(0, m2, 0, l2, 64 * l2),
        a.block(0, m2, l2, 2 * l2, 64 * l2),
    );
    let (a21, a22) = (
        a.block(m2, 2 * m2, 0, l2, 64 * l2),
        a.block(m2, 2 * m2, l2, 2 * l2, 64 * l2),
    );
    let (b11, b12) = (
        b.block(0, 64 * l2, 0, n2, 64 * n2),
        b.block(0, 64 * l2, n2, 2 * n2, 64 * n2),
    );
    let (b21, b22) = (
        b.block(64 * l2, 128 * l2, 0, n2, 64 * n2),
        b.block(64 * l2, 128 * l2, n2, 2 * n2, 64 * n2),
    );
    // Over GF(2) a difference is a sum.
    let s1 = &a21 + &a22;
    let s2 = &s1 + &a11;
    let s3 = &a11 + &a21;
    let s4 = &a12 + &s2;
    let t1 = &b12 + &b11;
    let t2 = &b22 + &t1;
    let t3 = &b22 + &b12;
    let t4 = &t2 + &b21;
    let p1 = product_from(&a11, &b11, cutoff);
    let p2 = product_from(&a12, &b21, cutoff);
    let p3 = product_from(&s4, &b22, cutoff);
    let p4 = product_from(&a22, &t4, cutoff);
    let p5 = product_from(&s1, &t1, cutoff);
    let p6 = product_from(&s2, &t2, cutoff);
    let p7 = product_from(&s3, &t3, cutoff);
    let u2 = &p1 + &p6;
    let u3 = &u2 + &p7;
    let u4 = &u2 + &p5;
    let mut c = BitMatrix::zeros(m, n);
    c.add_block(0, 0, &(p1 + &p2));
    c.add_block(0, n2, &(u4 + &p3));
    c.add_block(m2, 0, &(&u3 + &p4));
    c.add_block(m2, n2, &(u3 + &p5));
    let (me, le, ne) = (2 * m2, 128 * l2, 128 * n2);
    if le < l {
        // The inner columns beyond the even part, for the even rows and columns.
        let a3 = a.block(0, me, 2 * l2, a.stride, l - le);
        let b3 = b.block(le, l, 0, 2 * n2, ne);
        c.add_block(0, 0, &product_from(&a3, &b3, cutoff));
    }
    if ne < n {
        // The columns beyond, for the even rows.
        let a3 = a.block(0, me, 0, a.stride, l);
        let b3 = b.block(0, l, 2 * n2, b.stride, n - ne);
        c.add_block(0, 2 * n2, &product_from(&a3, &b3, cutoff));
    }
    if me < m {
        // The rows beyond.
        let a3 = a.block(me, m, 0, a.stride, l);
        c.add_block(me, 0, &product_from(&a3, b, cutoff));
    }
    c
}
