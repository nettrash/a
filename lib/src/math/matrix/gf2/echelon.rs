//! Elimination over `GF(2)`: M4RI, Bard's method of the Four Russians for echelon forms
//! ("Accelerating cryptanalysis with the Method of Four Russians", 2006; Albrecht and Bard's
//! M4RI library), with classical Gaussian elimination below 32 rows; and from it solutions,
//! inverses and null spaces.
//!
//! M4RI takes the columns in strips of `k`. In a strip it finds up to `k` pivots among the rows
//! not yet used, by classical elimination on the strip's bits alone: a row is a candidate for the
//! pivot of a column when its strip bits, reduced by the strip's pivots so far, have that
//! column's bit set, and only the row chosen is reduced in full. The pivot rows are reduced among
//! themselves so that they are the identity on their pivot columns; then a table of the `2^k`
//! sums of them is built in Gray-code order, one row addition each, and every other row is
//! cleared of the pivot columns by adding the one entry that its own bits there select. A strip
//! costs about `2^k + m` row additions where classical elimination takes `k·m/2`, so with
//! `k = ⌊log₂ m⌋ − 1` (at most 16, from `2^17` rows) the whole is `O(n³ / log n)` word
//! operations.

use super::{BitMatrix, xor_rows};

/// Classical elimination below this many rows, where a table would cost more than it saves.
const CLASSICAL_BELOW: usize = 32;

/// The widest strip, `2^MAX_K` table entries, reached from `2^17` rows: below that the width
/// keeps growing with `log₂ m`, so the elimination stays `O(n³ / log n)`.
const MAX_K: usize = 16;

/// The strip width for `rows` rows: `⌊log₂ rows⌋ − 1`, from 2 to `MAX_K` (from 4 in use, as below
/// `CLASSICAL_BELOW` rows the elimination is classical), so that the table of `2^k` entries
/// costs at most half the additions that clear the rows, and holds at most half as many words as
/// the matrix. (Measured on an Apple M4 up to 16384 rows, a cap of 8, 10 or 12 gave the same
/// times within the noise.)
fn strip_width(rows: usize) -> usize {
    (rows.max(1).ilog2() as usize)
        .saturating_sub(1)
        .clamp(2, MAX_K)
}

/// Brings `m` to row echelon form by row operations, its pivots chosen in the columns before
/// `limit` alone, and returns the pivot columns, ascending: row `i` is zero before column
/// `pivots[i]` and 1 there, the pivot rows are the identity on the pivot columns of their strip,
/// and the rows from `pivots.len()` are zero before `limit`. With `reduced`, the pivot columns
/// are cleared in the rows above the pivots too: the reduced row echelon form.
pub(super) fn echelonize(m: &mut BitMatrix, reduced: bool, limit: usize) -> Vec<usize> {
    debug_assert!(limit <= m.cols);
    if m.rows < CLASSICAL_BELOW {
        classical(m, reduced, limit)
    } else {
        m4ri(m, reduced, limit, strip_width(m.rows))
    }
}

/// Gaussian elimination a column at a time: the first row from the current one with a 1 in the
/// column is the pivot, and is added to every other row with a 1 there.
fn classical(m: &mut BitMatrix, reduced: bool, limit: usize) -> Vec<usize> {
    let s = m.stride;
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..limit {
        if r == m.rows {
            break;
        }
        let (w, bit) = (c / 64, 1u64 << (c % 64));
        let Some(p) = (r..m.rows).find(|&i| m.data[i * s + w] & bit != 0) else {
            continue;
        };
        m.swap_rows(p, r);
        let first = if reduced { 0 } else { r + 1 };
        for i in first..m.rows {
            if i != r && m.data[i * s + w] & bit != 0 {
                xor_rows(&mut m.data, s, i, r, w);
            }
        }
        pivots.push(c);
        r += 1;
    }
    pivots
}

/// The `width` bits of a row from column `c`, the row's words starting at `data[at]`.
#[inline]
fn window(data: &[u64], at: usize, c: usize, width: usize) -> usize {
    let (w, s) = (at + c / 64, c % 64);
    let mut v = data[w] >> s;
    if s + width > 64 {
        v |= data[w + 1] << (64 - s);
    }
    (v & ((1u64 << width) - 1)) as usize
}

/// A row that may hold a strip's next pivot: its index and its strip bits, reduced by the
/// strip's first `applied` pivots.
#[derive(Clone, Copy)]
struct Candidate {
    row: usize,
    bits: usize,
    applied: usize,
}

/// M4RI with strips of `k` columns.
///
/// The search for a strip's pivots reads a row's strip bits at most once a strip, the first time
/// a column needs a row past those read so far, and keeps the rows whose bits are not zero in a
/// short list, reduced lazily by the pivots found since; so a dense strip reads a few rows. A
/// row whose strip bits are zero is searched once for its first 1 before `limit`: the column is
/// kept in `first`, and the row costs one comparison a strip until the strips reach that column,
/// as it is not touched before then. A row with no 1 left before `limit` (zero, or in `solve`
/// zero in `A` but not in `B`) can never be a pivot: it is moved below the rows still in play and
/// never read again. When a strip finds no pivot, every row in play has been read, and the strips
/// go on from the first 1 of any of them. The pivot rows are moved to their places when the
/// strip's search is done.
fn m4ri(m: &mut BitMatrix, reduced: bool, limit: usize, k: usize) -> Vec<usize> {
    debug_assert!((1..=MAX_K).contains(&k));
    let s = m.stride;
    let mut pivots = Vec::new();
    let mut table: Vec<u64> = Vec::new();
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut dead: Vec<usize> = Vec::new();
    // Row `i` is zero before column `first[i]`, a bound that the search raises when it finds a
    // row's strip bits zero; the rows from `active` on have no 1 before `limit`.
    let mut first = vec![0usize; m.rows];
    let mut active = m.rows;
    let (mut r, mut c) = (0, 0);
    while c < limit && r < active {
        let width = k.min(limit - c);
        let end = c + width;
        // The pivot rows' words from the strip's first; the words before it are zero in all the
        // rows from `r`.
        let w0 = c / 64;
        let len = s - w0;
        // The strip's pivots: their rows, their columns' offsets in the strip, and their strip
        // bits, the identity on the pivot columns.
        let mut rows = [0usize; MAX_K];
        let mut offsets = [0usize; MAX_K];
        let mut bits = [0usize; MAX_K];
        let mut found = 0;
        candidates.clear();
        dead.clear();
        // The first row whose strip bits have not been read, and the first 1 of the rows read
        // whose strip bits are zero.
        let mut next = r;
        let mut lowest = usize::MAX;
        for t in 0..width {
            if r + found == active {
                break;
            }
            // The first candidate, in the order of the rows, whose reduced bits have bit `t`:
            // first among those read, dropping those reduced to zero, then among the rows after.
            let mut pick = None;
            let mut kept = 0;
            let mut seen = 0;
            while seen < candidates.len() {
                let mut e = candidates[seen];
                seen += 1;
                for q in e.applied..found {
                    if e.bits >> offsets[q] & 1 == 1 {
                        e.bits ^= bits[q];
                    }
                }
                e.applied = found;
                if e.bits == 0 {
                    continue;
                }
                candidates[kept] = e;
                kept += 1;
                if e.bits >> t & 1 == 1 {
                    pick = Some(kept - 1);
                    break;
                }
            }
            candidates.copy_within(seen.., kept);
            candidates.truncate(candidates.len() - (seen - kept));
            while pick.is_none() && next < active {
                let i = next;
                next += 1;
                if first[i] >= end {
                    lowest = lowest.min(first[i]);
                    continue;
                }
                let mut v = window(&m.data, i * s, c, width);
                if v == 0 {
                    match first_one(&m.data, i * s, end, limit) {
                        Some(f) => {
                            first[i] = f;
                            lowest = lowest.min(f);
                        }
                        None => dead.push(i),
                    }
                    continue;
                }
                for q in 0..found {
                    if v >> offsets[q] & 1 == 1 {
                        v ^= bits[q];
                    }
                }
                if v != 0 {
                    candidates.push(Candidate {
                        row: i,
                        bits: v,
                        applied: found,
                    });
                    if v >> t & 1 == 1 {
                        pick = Some(candidates.len() - 1);
                    }
                }
            }
            let Some(e) = pick.map(|j| candidates.remove(j)) else {
                continue;
            };
            let p = e.row;
            // The new pivot row, reduced in full by the strip's pivots so far.
            let mut v = window(&m.data, p * s, c, width);
            for q in 0..found {
                if v >> offsets[q] & 1 == 1 {
                    v ^= bits[q];
                    xor_rows(&mut m.data, s, p, rows[q], w0);
                }
            }
            debug_assert_eq!(v, e.bits);
            // The earlier pivot rows cleared of the new pivot's column.
            for q in 0..found {
                if bits[q] >> t & 1 == 1 {
                    bits[q] ^= v;
                    xor_rows(&mut m.data, s, rows[q], p, w0);
                }
            }
            rows[found] = p;
            offsets[found] = t;
            bits[found] = v;
            found += 1;
        }
        if found == 0 {
            // No row was moved. Every row in play was read for the strip's first column, and
            // none has a 1 before `lowest`, so the strips before it would find nothing either.
            debug_assert!(next == active && candidates.is_empty());
            active = drop_rows(m, &mut first, &dead, active);
            c = end.max(lowest);
            continue;
        }
        // The pivot rows to `r, r + 1, …`, in the order of their columns, and the rows without a
        // 1 followed: only those among the first `found` rows from `r` can be moved, perhaps more
        // than once.
        let moved = dead.partition_point(|&i| i < r + found);
        for q in 0..found {
            let (to, from) = (r + q, rows[q]);
            if from != to {
                m.swap_rows(from, to);
                first.swap(from, to);
                for later in &mut rows[q + 1..found] {
                    if *later == to {
                        *later = from;
                    }
                }
                for z in &mut dead[..moved] {
                    if *z == to {
                        *z = from;
                    }
                }
            }
        }
        if moved > 0 {
            dead.sort_unstable();
        }
        // The table, indexed by a row's strip bits at the pivot columns (the others masked off):
        // entry `v` is the sum of the pivot rows whose columns' bits are set in `v`, built in
        // Gray-code order so that each entry is the one before plus one row.
        table.resize(table.len().max(len << width), 0);
        table[..len].fill(0);
        let mut mask = 0;
        for &o in &offsets[..found] {
            mask |= 1 << o;
        }
        let mut index = 0;
        for g in 1..1usize << found {
            let q = g.trailing_zeros() as usize;
            let previous = index;
            index ^= 1 << offsets[q];
            let row = (r + q) * s + w0;
            for w in 0..len {
                table[index * len + w] = table[previous * len + w] ^ m.data[row + w];
            }
        }
        // Every other row cleared of the pivot columns by one entry. A row whose bound in
        // `first` is past the strip has zero strip bits and is not touched; a row that is has its
        // bound below the strip and is zero on the strip after it, so the bounds stay true.
        let above = if reduced { 0..r } else { 0..0 };
        let rows = above.chain(r + found..active);
        clear(&mut m.data, s, rows, c, width, mask, &table[..len << width]);
        active = drop_rows(m, &mut first, &dead, active);
        pivots.extend(offsets[..found].iter().map(|&o| c + o));
        r += found;
        c = end;
    }
    pivots
}

/// The first column from `from` and before `limit` where the row whose words start at `data[at]`
/// has a 1, or `None`.
fn first_one(data: &[u64], at: usize, from: usize, limit: usize) -> Option<usize> {
    if from >= limit {
        return None;
    }
    let last = (limit - 1) / 64;
    let mut w = from / 64;
    let mut word = data[at + w] & u64::MAX << (from % 64);
    loop {
        if w == last {
            word &= super::last_word_mask(limit);
            return (word != 0).then(|| 64 * w + word.trailing_zeros() as usize);
        }
        if word != 0 {
            return Some(64 * w + word.trailing_zeros() as usize);
        }
        w += 1;
        word = data[at + w];
    }
}

/// Clears `rows`, whose words are `s` apart in `data`, of the pivot columns of the strip of
/// `width` columns from `c` (whose bits are `mask`) by adding the entry of `table` that their
/// bits there select. A function of its own, as the loop compiles to slower code inside `m4ri`
/// (measured: 15% slower on 262144 × 300).
#[inline(never)]
fn clear(
    data: &mut [u64],
    s: usize,
    rows: impl Iterator<Item = usize>,
    c: usize,
    width: usize,
    mask: usize,
    table: &[u64],
) {
    let w0 = c / 64;
    let len = s - w0;
    for i in rows {
        let v = window(data, i * s, c, width) & mask;
        if v != 0 {
            let row = &mut data[i * s + w0..(i + 1) * s];
            for (x, y) in row.iter_mut().zip(&table[v * len..(v + 1) * len]) {
                *x ^= y;
            }
        }
    }
}

/// Moves the rows `dead`, ascending and below `active`, none with a 1 before the elimination's
/// limit, to the end of the rows in play, their bounds in `first` with them, and returns the new
/// end.
fn drop_rows(m: &mut BitMatrix, first: &mut [usize], dead: &[usize], mut active: usize) -> usize {
    for &i in dead.iter().rev() {
        active -= 1;
        m.swap_rows(i, active);
        first.swap(i, active);
    }
    active
}

/// `[A | B]` with the columns of `B` from the first word after those of `A`, so that no word is
/// shifted; the columns between are zero and never pivots.
fn augment(a: &BitMatrix, b: &BitMatrix) -> BitMatrix {
    debug_assert_eq!(a.rows, b.rows);
    let mut m = BitMatrix::zeros(a.rows, 64 * a.stride + b.cols);
    for i in 0..a.rows {
        let row = &mut m.data[i * m.stride..(i + 1) * m.stride];
        row[..a.stride].copy_from_slice(&a.data[i * a.stride..(i + 1) * a.stride]);
        row[a.stride..].copy_from_slice(&b.data[i * b.stride..(i + 1) * b.stride]);
    }
    m
}

/// A solution of `A·X = B`, the rows of the columns without a pivot zero, or `None`; with
/// `full_rank`, `None` also when `A` has a column without a pivot (for the inverse).
pub(super) fn solve(a: &BitMatrix, b: &BitMatrix, full_rank: bool) -> Option<BitMatrix> {
    let mut m = augment(a, b);
    let pivots = echelonize(&mut m, true, a.cols);
    if full_rank && pivots.len() < a.cols {
        return None;
    }
    // Consistent when the rows without a pivot, zero in `A`, are zero in `B` too.
    let s = m.stride;
    if (pivots.len()..m.rows).any(|i| {
        m.data[i * s + a.stride..(i + 1) * s]
            .iter()
            .any(|&w| w != 0)
    }) {
        return None;
    }
    let mut x = BitMatrix::zeros(a.cols, b.cols);
    for (i, &p) in pivots.iter().enumerate() {
        x.data[p * x.stride..(p + 1) * x.stride]
            .copy_from_slice(&m.data[i * s + a.stride..(i + 1) * s]);
    }
    Some(x)
}

/// The basis of the null space from the reduced row echelon form: one column for each column
/// without a pivot.
pub(super) fn null_space(a: &BitMatrix) -> BitMatrix {
    let mut m = a.clone();
    let pivots = echelonize(&mut m, true, a.cols);
    let n = a.cols;
    // The columns without a pivot, as a mask of words and by their index among themselves.
    let mut free = vec![u64::MAX; m.stride];
    if let Some(last) = free.last_mut() {
        *last = super::last_word_mask(n);
    }
    for &p in &pivots {
        free[p / 64] &= !(1 << (p % 64));
    }
    let mut index = vec![usize::MAX; n];
    let mut nullity = 0;
    for (j, slot) in index.iter_mut().enumerate() {
        if free[j / 64] >> (j % 64) & 1 == 1 {
            *slot = nullity;
            nullity += 1;
        }
    }
    let mut x = BitMatrix::zeros(n, nullity);
    for (j, &f) in index.iter().enumerate() {
        if f != usize::MAX {
            x.data[j * x.stride + f / 64] |= 1 << (f % 64);
        }
    }
    for (i, &p) in pivots.iter().enumerate() {
        let row = &m.data[i * m.stride..(i + 1) * m.stride];
        for (w, (&word, &mask)) in row.iter().zip(&free).enumerate() {
            let mut bits = word & mask;
            while bits != 0 {
                let f = index[64 * w + bits.trailing_zeros() as usize];
                x.data[p * x.stride + f / 64] |= 1 << (f % 64);
                bits &= bits - 1;
            }
        }
    }
    x
}
