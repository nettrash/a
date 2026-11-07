//! The end: the matrix of the relations' exponents modulo 2, filtered, its null space, and for
//! each vector of it the congruence of squares `X² ≡ Y² (mod N)`, with `X` the product of the
//! relations' `Ax + B` and `Y` the square root of the product of their `A·g(x)`, known from its
//! factors; `gcd(X − Y, N)` is a factor of `N` other than 1 and `N` for about half of the vectors.

use super::super::super::{ByteOrder, Number};
use super::linalg::{Matrix, null_space};
use super::poly::b_values;
use super::sieve::Siqs;
use super::wide::Wide;
use std::collections::HashMap;

/// The columns kept beyond the rows.
const EXCESS: usize = 64 + 32;

impl Siqs<'_> {
    /// A factor of `n` from the relations found, as limbs, or `None`.
    pub(super) fn finish(&self) -> Option<Vec<u64>> {
        let sets = self.store.columns();
        let size = self.fb.prime.len();
        // The odd exponents of each set: row 0 for −1, row j + 1 for prime j.
        let mut cols: Vec<Vec<u32>> = Vec::with_capacity(sets.len());
        let mut all = Vec::new();
        for set in &sets {
            all.clear();
            let mut negative = false;
            for &r in set {
                let rel = &self.store.rels[r as usize];
                negative ^= rel.negative;
                all.extend(self.store.factors_of(r as usize).iter().map(|&j| j + 1));
            }
            all.sort_unstable();
            let mut col = Vec::new();
            if negative {
                col.push(0);
            }
            let mut i = 0;
            while i < all.len() {
                let mut k = i;
                while k < all.len() && all[k] == all[i] {
                    k += 1;
                }
                if (k - i) & 1 == 1 {
                    col.push(all[i]);
                }
                i = k;
            }
            cols.push(col);
        }
        let (keep, rows) = filter(&cols, size + 1);
        // The matrix of the columns kept, the rows renumbered.
        let mut start = vec![0u32];
        let mut entries = Vec::new();
        for &c in &keep {
            for &r in &cols[c] {
                entries.push(rows[r as usize]);
            }
            start.push(entries.len() as u32);
        }
        let nrows = rows.iter().filter(|&&r| r != u32::MAX).count();
        let matrix = Matrix {
            rows: nrows,
            start,
            entries,
        };
        let deps = null_space(&matrix, self.n[0] ^ 0x4C41_4E43_5A4F_5321);
        let mut any = 0u64;
        for &d in &deps {
            any |= d;
        }
        let n = number_of(self.n);
        let mut cache: HashMap<u32, (Wide, Vec<Wide>, Wide)> = HashMap::new();
        for t in 0..64 {
            if any >> t & 1 == 0 {
                continue;
            }
            let mut rels = Vec::new();
            for (k, &d) in deps.iter().enumerate() {
                if d >> t & 1 == 1 {
                    rels.extend_from_slice(&sets[keep[k]]);
                }
            }
            if let Some(f) = self.congruence(&rels, &n, &mut cache) {
                return Some(f);
            }
        }
        None
    }

    /// The factor `gcd(X − Y, N)` from the relations `rels`, whose product is a square, as limbs, if
    /// it is not 1 or `N`.
    fn congruence(
        &self,
        rels: &[u32],
        n: &Number,
        cache: &mut HashMap<u32, (Wide, Vec<Wide>, Wide)>,
    ) -> Option<Vec<u64>> {
        let size = self.fb.prime.len();
        let order = ByteOrder::LittleEndian;
        let w = self.w;
        let mut exps = vec![0u32; size];
        let mut large: HashMap<u32, u32> = HashMap::new();
        let mut x = Number::new_from_u8(1, order);
        for &r in rels {
            let rel = &self.store.rels[r as usize];
            // The coefficient of x in Y, the B_l, and what B has besides them: A for (2Ax + B)²
            // when the sum of the B_l is even (whatever their signs).
            let (a2, bl, extra) = cache.entry(rel.a).or_insert_with(|| {
                let (a, bl, _) = b_values(&self.fb, &self.a_list[rel.a as usize]);
                let a = Wide::of(&a, w);
                let bl: Vec<Wide> = bl.iter().map(|b| Wide::of(b, w)).collect();
                let sum = bl.iter().fold(Wide::default(), |s, b| s.add(b, w));
                let extra = if self.two_a && sum.0[0] & 1 == 0 {
                    a
                } else {
                    Wide::default()
                };
                (if self.two_a { a.add(&a, w) } else { a }, bl, extra)
            });
            let gray = rel.v ^ (rel.v >> 1);
            let mut b = *extra;
            for (l, bv) in bl.iter().enumerate() {
                if l + 1 < bl.len() && gray >> l & 1 == 1 {
                    b = b.sub(bv, w);
                } else {
                    b = b.add(bv, w);
                }
            }
            // Y = A2·x + B, exact in w limbs; its sign does not matter, Y² being what counts.
            let y = a2.mul_i64(i64::from(rel.x), w).add(&b, w);
            let y = if y.is_negative(w) { y.neg(w) } else { y };
            x = (&x * &number_of(&y.0[..w])) % n;
            for &j in self.store.factors_of(r as usize) {
                exps[j as usize] += 1;
            }
            for &p in &rel.lp {
                if p != 1 {
                    *large.entry(p).or_insert(0) += 1;
                }
            }
        }
        if exps.iter().any(|&e| e & 1 == 1) || large.values().any(|&e| e & 1 == 1) {
            return None;
        }
        // Y: the primes to half their exponents, multiplied in words before each product modulo N.
        let mut y = Number::new_from_u8(1, order);
        let mut word = 1u64;
        let push = |p: u64, y: &mut Number, word: &mut u64| match word.checked_mul(p) {
            Some(w) => *word = w,
            None => {
                *y = (&*y * &Number::new_from_u64(*word, order)) % n;
                *word = p;
            }
        };
        for (j, &e) in exps.iter().enumerate() {
            for _ in 0..e / 2 {
                push(u64::from(self.fb.prime[j]), &mut y, &mut word);
            }
        }
        for (&p, &e) in &large {
            for _ in 0..e / 2 {
                push(u64::from(p), &mut y, &mut word);
            }
        }
        // Each Y² − kN of (2Ax + B)² is 4A·g(x): a 2 more in Y for each relation.
        if self.two_a {
            for _ in 0..rels.len() {
                push(2, &mut y, &mut word);
            }
        }
        y = (&y * &Number::new_from_u64(word, order)) % n;
        let one = Number::new_from_u8(1, order);
        for z in [&x - &y, &x + &y] {
            let g = z.gcd(n);
            if g > one && g < *n {
                return Some(limbs_of(&g));
            }
        }
        None
    }
}

/// The columns to keep and the new number of each row (`u32::MAX` for rows left out): the columns
/// with a row of weight 1 removed until there are none, then the heaviest beyond the rows plus
/// [`EXCESS`] (removing them can make singletons again).
fn filter(cols: &[Vec<u32>], nrows: usize) -> (Vec<usize>, Vec<u32>) {
    let mut alive = vec![true; cols.len()];
    let mut weight = vec![0u32; nrows];
    for c in cols {
        for &r in c {
            weight[r as usize] += 1;
        }
    }
    loop {
        let mut changed = true;
        while changed {
            changed = false;
            for (k, c) in cols.iter().enumerate() {
                if alive[k] && c.iter().any(|&r| weight[r as usize] == 1) {
                    alive[k] = false;
                    for &r in c {
                        weight[r as usize] -= 1;
                    }
                    changed = true;
                }
            }
        }
        let live_rows = weight.iter().filter(|&&w| w > 0).count();
        let live_cols = alive.iter().filter(|&&a| a).count();
        if live_cols <= live_rows + EXCESS {
            break;
        }
        // Drop the heaviest columns beyond the excess.
        let mut order: Vec<usize> = (0..cols.len()).filter(|&k| alive[k]).collect();
        order.sort_by_key(|&k| core::cmp::Reverse(cols[k].len()));
        for &k in order.iter().take(live_cols - live_rows - EXCESS) {
            alive[k] = false;
            for &r in &cols[k] {
                weight[r as usize] -= 1;
            }
        }
    }
    let keep: Vec<usize> = (0..cols.len()).filter(|&k| alive[k]).collect();
    let mut rows = vec![u32::MAX; nrows];
    let mut next = 0u32;
    for (r, &w) in weight.iter().enumerate() {
        if w > 0 {
            rows[r] = next;
            next += 1;
        }
    }
    (keep, rows)
}

/// The number of the limbs `x`, least significant first.
pub(super) fn number_of(x: &[u64]) -> Number {
    let mut bytes = Vec::with_capacity(8 * x.len());
    for &l in x {
        bytes.extend_from_slice(&l.to_le_bytes());
    }
    Number::new(bytes, ByteOrder::LittleEndian)
}

/// The limbs of the magnitude of `x`, least significant first, without zero limbs at the top.
pub(super) fn limbs_of(x: &Number) -> Vec<u64> {
    let len = x.value.len();
    let mut out = vec![0u64; len.div_ceil(8)];
    for (i, &b) in x.value.iter().enumerate() {
        let pos = match x.byte_order {
            ByteOrder::LittleEndian => i,
            ByteOrder::BigEndian => len - 1 - i,
        };
        out[pos / 8] |= u64::from(b) << (8 * (pos % 8));
    }
    while out.last() == Some(&0) {
        out.pop();
    }
    out
}
