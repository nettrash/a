//! The determinant of a matrix of integers by its residues: `det(A) mod p` for word primes `p` by
//! the elimination over `GF(p)` of [`ModMatrix`](super::super::ModMatrix) (its PLE decomposition),
//! as many primes as make their product pass twice Hadamard's bound `|det A| ≤ Π_i ‖a_i‖`, then
//! the Chinese remainder theorem ([`Number::crt`]) and the residue of least magnitude. Nothing
//! is guessed: the primes stop at the bound, never when the residues seem to settle, so the
//! result is the determinant for every matrix, singular ones included.
//!
//! The work for `n × n` integers of up to `L` limbs is about `c ≈ 64·n·L/61` primes (Hadamard's
//! bound has about `n·(64·L + log₂ n / 2)` bits), and for each prime `n³/3` products of words for
//! the elimination and the residues of the `n²` integers. Taken one prime at a time, the residues
//! cost `L` products of words each, `n²·L·c ≈ n³·L²` in all, which passes the elimination's
//! `n³·c/3 ≈ n⁴·L/3` once `L > n/3`. From [`TREE_FROM`] limbs they come from a remainder tree
//! instead (as FLINT's `fmpz_comb` does): the primes in runs of [`LEAF`], the products of the runs
//! paired up a tree, and each integer reduced down it, modulo each node the remainder modulo its
//! parent, so that an integer costs about `n·M(L)·log L` (`M` the cost of a product of `L` limbs)
//! for all the primes rather than `L·c`. The whole is then about `n³·M(L)·log L + n⁴·L/3`,
//! against Bareiss's `n³/3` products of numbers as long as the minors, up to `n·L` limbs.

use super::super::modp::{Field, det_in_place};
use super::{B, Int, S};
use crate::math::numbers::{ByteOrder, Number, from_limbs, is_prime_u64, limbs_of};

/// The primes are those below this, from the top: each is above `2^61`.
const TOP: u64 = 1 << 62;

/// The bits each prime contributes for certain to the product of the primes, as each is above
/// `2^61`.
const PRIME_BITS: u64 = 61;

/// The length in limbs of the longest integer from which the residues come from a remainder
/// tree rather than one prime at a time. Measured on an Apple M4 Max in the release profile,
/// determinants of random `12 × 12` matrices, one prime at a time against the tree: for elements
/// of 64 limbs 8.2 ms against 10.8, of 256 limbs 65.8 against 76.3, of 512 limbs 214 against 207,
/// of 1024 limbs 755 against 583, of 2048 limbs 2.84 s against 1.49; at 9375 limbs (600 000
/// bits) 62.9 s one prime at a time (an audit's measurement) against 9.87 s.
const TREE_FROM: usize = 512;

/// The primes of a leaf of the remainder tree: their product is below `2^(62·16)`, 16 limbs.
const LEAF: usize = 16;

/// The words of residues the tree keeps at once, `n²` for each prime of a run: the primes go
/// through it in runs of `2^22 / n²`.
const TREE_WORDS: usize = 1 << 22;

/// An integer of the matrix as its residues need it: a word with its sign, or limbs.
enum Entry<'a> {
    Word(i64),
    Limbs(bool, &'a [u64]),
}

/// A product tree over a run of primes, for the residues of long integers modulo all of them:
/// `levels[0]` the products of the primes [`LEAF`] at a time, each level above the products of
/// pairs of the one below, the last level the product of the run.
struct Tree {
    levels: Vec<Vec<Number>>,
    fields: Vec<Field>,
    /// The residues of `2^(64·i)`, `i < LEAF`, for each prime, [`LEAF`] a prime.
    powers: Vec<u64>,
}

impl Tree {
    /// The tree over the primes `ps`.
    fn new(ps: &[u64]) -> Tree {
        let leaves: Vec<Number> = ps
            .chunks(LEAF)
            .map(|c| {
                c.iter()
                    .fold(Number::from(1u8), |s, &p| s * Number::from(p))
            })
            .collect();
        let mut levels = vec![leaves];
        while levels.last().is_some_and(|l| l.len() > 1) {
            let up = levels.last().map_or(Vec::new(), |l| {
                l.chunks(2)
                    .map(|c| {
                        if c.len() == 2 {
                            &c[0] * &c[1]
                        } else {
                            c[0].clone()
                        }
                    })
                    .collect()
            });
            levels.push(up);
        }
        let fields: Vec<Field> = ps.iter().map(|&p| Field::new(p)).collect();
        let powers = fields.iter().flat_map(|f| f.powers(LEAF)).collect();
        Tree {
            levels,
            fields,
            powers,
        }
    }

    /// The residues of the integer of the sign `neg` and the magnitude `r` modulo the primes
    /// under the node `index` of `level`, `r` below the product of its parent (or any integer
    /// at the root): `out[k·stride + e]` for the `k`-th prime of the run.
    #[allow(clippy::too_many_arguments)]
    fn descend(
        &self,
        level: usize,
        index: usize,
        r: &Number,
        neg: bool,
        out: &mut [u64],
        stride: usize,
        e: usize,
    ) {
        let node = &self.levels[level][index];
        let reduced;
        let r = if r.bits() >= node.bits() {
            reduced = r % node;
            &reduced
        } else {
            r
        };
        if level == 0 {
            // At most `LEAF` limbs below the product of `LEAF` primes.
            let limbs = limbs_of(r);
            let start = index * LEAF;
            let end = (start + LEAF).min(self.fields.len());
            for k in start..end {
                let w = &self.powers[k * LEAF..(k + 1) * LEAF];
                out[k * stride + e] = self.fields[k].of_limbs_by(neg, &limbs, w);
            }
            return;
        }
        let below = self.levels[level - 1].len();
        for child in [2 * index, 2 * index + 1] {
            if child < below {
                self.descend(level - 1, child, r, neg, out, stride, e);
            }
        }
    }
}

/// The squared Euclidean length of a vector of integers.
fn length2<'a>(v: impl Iterator<Item = &'a Int>) -> Number {
    v.fold(Number::default(), |s, x| {
        let x = x.number(ByteOrder::LittleEndian);
        s + &x * &x
    })
}

/// The number of bits `h` with `|det A| < 2^h` by Hadamard's bound on the rows and on the
/// columns of the `n × n` matrix whose rows start every `stride` elements of `m`, the smaller of
/// the two; `None` when a row or a column is zero, and the determinant with it.
pub(super) fn hadamard_bits(m: &[Int], n: usize, stride: usize) -> Option<u64> {
    let mut rows = Number::from(1u8);
    let mut cols = Number::from(1u8);
    for i in 0..n {
        let r = length2(m[i * stride..i * stride + n].iter());
        let c = length2((0..n).map(|j| &m[j * stride + i]));
        if r.is_zero() || c.is_zero() {
            return None;
        }
        rows *= &r;
        cols *= &c;
    }
    // `|det A|² ≤ D² < 2^bits(D²)`, so `|det A| < 2^⌈bits(D²) / 2⌉`.
    let bits = rows.bits().min(cols.bits());
    Some(bits.div_ceil(2))
}

/// The primes below [`TOP`], from the top down: each above `2^61`.
pub(super) struct Primes(u64);

impl Primes {
    /// The primes from the largest below [`TOP`].
    pub(super) fn new() -> Primes {
        Primes(TOP - 1)
    }
}

impl Iterator for Primes {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        while !is_prime_u64(self.0) {
            self.0 -= 2;
        }
        let p = self.0;
        self.0 -= 2;
        Some(p)
    }
}

/// The determinant of the `n × n` matrix of integers held row after row in `m`, `n ≥ 1`.
pub(super) fn det(m: &[Int], n: usize) -> Int {
    let Some(h) = hadamard_bits(m, n, n) else {
        return S(0);
    };
    // `M > 2^(61·count) ≥ 2^(h + 1) > 2·|det A|`.
    let count = (h + 1).div_ceil(PRIME_BITS) as usize;
    let entries: Vec<Entry<'_>> = m
        .iter()
        .map(|x| match x {
            S(v) => Entry::Word(*v),
            B(b) => Entry::Limbs(b.neg, &b.d),
        })
        .collect();
    let limbs = m
        .iter()
        .map(|x| if let B(b) = x { b.d.len() } else { 1 })
        .max()
        .unwrap_or(1);
    let ps: Vec<u64> = Primes::new().take(count).collect();
    let mut congruences = Vec::with_capacity(count);
    let nn = n * n;
    if limbs < TREE_FROM {
        let mut residues = vec![0u64; nn];
        for &p in &ps {
            let f = Field::new(p);
            let powers = f.powers(limbs);
            for (r, x) in residues.iter_mut().zip(&entries) {
                *r = match x {
                    Entry::Word(v) => f.of_i64(*v),
                    Entry::Limbs(neg, d) => f.of_limbs_by(*neg, d, &powers),
                };
            }
            let d = det_in_place(f, &mut residues, n);
            congruences.push((Number::from(d), Number::from(p)));
        }
    } else {
        let run = (TREE_WORDS / nn).max(LEAF) / LEAF * LEAF;
        for run in ps.chunks(run) {
            let tree = Tree::new(run);
            let top = tree.levels.len() - 1;
            let mut out = vec![0u64; run.len() * nn];
            for (e, x) in entries.iter().enumerate() {
                match x {
                    Entry::Word(v) => {
                        for (k, f) in tree.fields.iter().enumerate() {
                            out[k * nn + e] = f.of_i64(*v);
                        }
                    }
                    Entry::Limbs(neg, d) => {
                        let r = from_limbs(d, ByteOrder::LittleEndian);
                        tree.descend(top, 0, &r, *neg, &mut out, nn, e);
                    }
                }
            }
            for (k, (f, &p)) in tree.fields.iter().zip(run).enumerate() {
                let d = det_in_place(*f, &mut out[k * nn..(k + 1) * nn], n);
                congruences.push((Number::from(d), Number::from(p)));
            }
        }
    }
    let (x, modulus) = Number::crt(&congruences).expect("coprime moduli");
    let half = &modulus >> 1usize;
    let x = if x > half { x - &modulus } else { x };
    Int::of(&x)
}
