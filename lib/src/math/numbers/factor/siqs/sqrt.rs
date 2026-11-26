//! The end: the matrix of the relations' exponents modulo 2, filtered, its null space, and for
//! each vector of it the congruence of squares `X² ≡ Y² (mod N)`, with `X` the product of the
//! relations' `Ax + B` and `Y` the square root of the product of their `A·g(x)`, known from its
//! factors, both in Montgomery's arithmetic modulo `N`; `gcd(X − Y, N)` is a factor of `N` other
//! than 1 and `N` for about half of the vectors.

use super::super::super::{ByteOrder, Number};
use super::super::field::{Arith, Job, gcd_n, with_field};
use super::linalg::{Matrix, null_space};
use super::poly::b_values;
use super::relations::Words;
use super::sieve::Siqs;
use super::wide::Wide;
use std::collections::HashMap;

/// The columns kept beyond the rows.
const EXCESS: usize = 64 + 32;

/// The values each `A` gives the relations' `Y`: the coefficient of `x`, the `B_l`, and what `B`
/// has besides them.
type Bs = (Wide, Vec<Wide>, Wide);

impl Siqs<'_> {
    /// A factor of `n` from the relations found, as limbs, or `None`.
    pub(super) fn finish(&self) -> Option<Vec<u64>> {
        let sets = self.store.columns();
        let size = self.fb.prime.len();
        // The odd exponents of each set, one column each: row 0 for −1, row j + 1 for prime j.
        let mut start = Vec::with_capacity(sets.len() + 1);
        start.push(0u32);
        let mut entries = Vec::with_capacity(16 * sets.len());
        let mut all = Vec::new();
        for set in &sets {
            all.clear();
            let mut negative = false;
            for &r in set {
                negative ^= self.store.rels[r as usize].negative;
                all.extend_from_slice(self.store.factors_of(r as usize));
            }
            all.sort_unstable();
            if negative {
                entries.push(0);
            }
            let mut i = 0;
            while i < all.len() {
                let mut k = i + 1;
                while k < all.len() && all[k] == all[i] {
                    k += 1;
                }
                if (k - i) & 1 == 1 {
                    entries.push(all[i] + 1);
                }
                i = k;
            }
            start.push(entries.len() as u32);
        }
        let (keep, rows) = filter(&start, &entries, size + 1);
        // The matrix of the columns kept, the rows renumbered.
        let mut mstart = Vec::with_capacity(keep.len() + 1);
        mstart.push(0u32);
        let mut mentries = Vec::with_capacity(entries.len());
        for &c in &keep {
            for &r in &entries[start[c] as usize..start[c + 1] as usize] {
                mentries.push(rows[r as usize]);
            }
            mstart.push(mentries.len() as u32);
        }
        let nrows = rows.iter().filter(|&&r| r != u32::MAX).count();
        let matrix = Matrix {
            rows: nrows,
            start: mstart,
            entries: mentries,
        };
        let deps = null_space(&matrix, self.n[0] ^ 0x4C41_4E43_5A4F_5321);
        with_field(
            self.n,
            Squares {
                siqs: self,
                deps: &deps,
                sets: &sets,
                keep: &keep,
            },
        )
    }

    /// The factor `gcd(X − Y, N)` from the relations `rels`, whose product is a square, as limbs, if
    /// it is not 1 or `N`: `X` and `Y` as residues of the arithmetic `f` modulo `N`.
    fn congruence<A: Arith>(
        &self,
        f: &A,
        rels: &[u32],
        cache: &mut HashMap<u32, Bs, Words>,
        exps: &mut [u32],
        large: &mut Vec<u32>,
    ) -> Option<Vec<u64>> {
        let w = self.w;
        exps.fill(0);
        large.clear();
        let mut x = f.one();
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
            f.mul_in(&mut x, &residue_of(f, &y.0[..w]));
            for &j in self.store.factors_of(r as usize) {
                exps[j as usize] += 1;
            }
            for &p in &rel.lp {
                if p != 1 {
                    large.push(p);
                }
            }
        }
        if exps.iter().any(|&e| e & 1 == 1) {
            return None;
        }
        large.sort_unstable();
        if large.len() & 1 == 1 || large.chunks(2).any(|c| c[0] != c[1]) {
            return None;
        }
        // Y: the primes to half their exponents, multiplied in words before each product modulo N.
        let mut y = f.one();
        let mut word = 1u64;
        let mut push = |p: u64, y: &mut A::V| match word.checked_mul(p) {
            Some(v) => word = v,
            None => {
                f.mul_in(y, &f.small(word));
                word = p;
            }
        };
        for (j, &e) in exps.iter().enumerate() {
            for _ in 0..e / 2 {
                push(u64::from(self.fb.prime[j]), &mut y);
            }
        }
        for pair in large.chunks(2) {
            push(u64::from(pair[0]), &mut y);
        }
        // Each Y² − kN of (2Ax + B)² is 4A·g(x): a 2 more in Y for each relation.
        if self.two_a {
            for _ in 0..rels.len() {
                push(2, &mut y);
            }
        }
        f.mul_in(&mut y, &f.small(word));
        let mut z = f.zero();
        for minus in [true, false] {
            if minus {
                f.sub(&mut z, &x, &y);
            } else {
                f.add(&mut z, &x, &y);
            }
            let g = gcd_n(f, &z);
            if !(g.len() == 1 && g[0] == 1) && g.as_slice() != f.n() {
                return Some(g);
            }
        }
        None
    }
}

/// The square roots of the products of the vectors of the null space, in the arithmetic modulo
/// `N`, until one gives a factor.
struct Squares<'s, 'a> {
    /// The sieve and its relations.
    siqs: &'s Siqs<'a>,
    /// The vectors, bit `t` of word `k` set when column `k` is in vector `t`.
    deps: &'s [u64],
    /// The relations of each set.
    sets: &'s [Vec<u32>],
    /// The set of each column.
    keep: &'s [usize],
}

impl Job for Squares<'_, '_> {
    type Out = Option<Vec<u64>>;

    fn run<A: Arith>(self, f: &A) -> Option<Vec<u64>> {
        let mut cache = HashMap::default();
        let mut exps = vec![0u32; self.siqs.fb.prime.len()];
        let mut large = Vec::new();
        let mut rels = Vec::new();
        let any = self.deps.iter().fold(0, |a, &d| a | d);
        for t in 0..64 {
            if any >> t & 1 == 0 {
                continue;
            }
            rels.clear();
            for (k, &d) in self.deps.iter().enumerate() {
                if d >> t & 1 == 1 {
                    rels.extend_from_slice(&self.sets[self.keep[k]]);
                }
            }
            if let Some(g) = self
                .siqs
                .congruence(f, &rels, &mut cache, &mut exps, &mut large)
            {
                return Some(g);
            }
        }
        None
    }
}

/// The residue of the plain number of the limbs `y`, reduced modulo `N` should it not be below.
fn residue_of<A: Arith>(f: &A, y: &[u64]) -> A::V {
    let n = f.n();
    let mut len = y.len();
    while len > 0 && y[len - 1] == 0 {
        len -= 1;
    }
    let y = &y[..len];
    if len < n.len() || (len == n.len() && below(y, n)) {
        return f.residue(y);
    }
    // Not met by the sieve's values, which are below N; kept for safety.
    let r = &number_of(y) % &number_of(n);
    f.residue(&limbs_of(&r))
}

/// Whether the limbs `a` are below the limbs `b`, of the same length.
fn below(a: &[u64], b: &[u64]) -> bool {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        if x != y {
            return x < y;
        }
    }
    false
}

/// The columns to keep and the new number of each row (`u32::MAX` for rows left out): the columns
/// (`start`, `entries`: the rows of column `c` are `entries[start[c]..start[c + 1]]`) with a row of
/// weight 1 removed until there are none, then the heaviest beyond the rows plus [`EXCESS`]
/// (removing them can make singletons again).
fn filter(start: &[u32], entries: &[u32], nrows: usize) -> (Vec<usize>, Vec<u32>) {
    let ncols = start.len() - 1;
    let col = |c: usize| &entries[start[c] as usize..start[c + 1] as usize];
    let mut alive = vec![true; ncols];
    let mut weight = vec![0u32; nrows];
    for &r in entries {
        weight[r as usize] += 1;
    }
    loop {
        let mut changed = true;
        while changed {
            changed = false;
            for (k, a) in alive.iter_mut().enumerate() {
                if *a && col(k).iter().any(|&r| weight[r as usize] == 1) {
                    *a = false;
                    for &r in col(k) {
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
        let mut order: Vec<usize> = (0..ncols).filter(|&k| alive[k]).collect();
        order.sort_by_key(|&k| core::cmp::Reverse(col(k).len()));
        for &k in order.iter().take(live_cols - live_rows - EXCESS) {
            alive[k] = false;
            for &r in col(k) {
                weight[r as usize] -= 1;
            }
        }
    }
    let keep: Vec<usize> = (0..ncols).filter(|&k| alive[k]).collect();
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
