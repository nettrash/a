//! The exact Gram–Schmidt data of [`Gso`] from their residues modulo word primes.
//!
//! The Gram determinants `d_i` and the coefficients `λ_ij = d_{j+1}·μ_ij` are integers (minors
//! of the Gram matrix `G`), and modulo a prime `p` that divides none of the `d_i` they come from
//! the factorisation `G = L·D·Lᵀ` over `GF(p)`: `L` holds the `μ_ij`, `D` the `r_jj = d_{j+1}/d_j`,
//! the PLE decomposition of [`ModMatrix`] finds both (with no row exchanged, as every leading
//! minor is a unit), and `d_{j+1} = d_j·r_jj`, `λ_ij = d_{j+1}·μ_ij`. Hadamard's inequality
//! bounds them: `d_j ≤ Π_{l<j} G_ll` and `|λ_ij| ≤ √(G_ii·G_jj)·Π_{l<j} G_ll`, so each is the
//! residue of least magnitude modulo the product of the primes once that product passes twice
//! its bound (each element taking only the primes its bound asks for), rebuilt one prime at a
//! time by the Chinese remainder theorem (Garner's incremental form). A prime that divides a
//! `d_i` shows as a row exchange or a rank below `n` and is skipped; the product of the primes
//! skipped divides `Π d_i`, so more of them than its length over 61 bits allows prove a `d_i`
//! zero, and the rows dependent, which the elimination of [`Gso::of`] then finds. Nothing is
//! guessed: the result is [`Gso::of`]'s.
//!
//! For `n` rows whose Gram–Schmidt vectors have `b` bits, about `n·b/30` primes, each `O(n^ω)`
//! operations on words for the decomposition and `n²` products for the residues, and the
//! rebuilding about `n²·c²/6` products of words for `c` primes; against `n³/6` steps of a
//! product and an exact division of numbers as long as the minors for the elimination.

use super::enum_gso::{Gso, dot};
use crate::math::matrix::ModMatrix;
use crate::math::numbers::nat::addmul_1;
use crate::math::numbers::{ByteOrder, Number, from_limbs, is_prime_u64, limbs_of};

/// The number of rows from which the data come from residues rather than the elimination.
/// Measured on an Apple M4 Max, `n` from 4 to 40 on bases reduced by `lll_fp` (uniform of 10 to
/// 400 bits, knapsacks, Goldstein–Mayer and q-ary), residues against [`Gso::of`]: for entries of
/// 10 bits the residues take 0.93 of the time at 16 rows (1.02 at 14, 1.18 at 12), 0.62 at 40;
/// for 50 to 400 bits they pay only from 20 to 24 rows (0.85 to 0.91 there, 1.06 to 1.30 at 16 to
/// 20), which a larger `FROM` would win at the cost of the short entries from 16 rows.
const FROM: usize = 16;

/// The bits each prime contributes for certain: each is above `2^61`.
const PRIME_BITS: u64 = 61;

/// The integers modulo a prime `p < 2^62` in Montgomery's form, for the residues of long
/// integers and the products of residues.
#[derive(Clone, Copy)]
struct Field {
    p: u64,
    /// `−p⁻¹ mod 2^64`.
    neg_inv: u64,
    /// `2^128 mod p`.
    r2: u64,
}

impl Field {
    fn new(p: u64) -> Field {
        // `p⁻¹ mod 2^64` by Newton's iteration, from `p·p ≡ 1 (mod 8)`.
        let mut inv = p;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        }
        let r = ((1u128 << 64) % u128::from(p)) as u64;
        Field {
            p,
            neg_inv: inv.wrapping_neg(),
            r2: (u128::from(r) * u128::from(r) % u128::from(p)) as u64,
        }
    }

    /// `t·2^−64 mod p` for `t < p·2^64`.
    #[inline(always)]
    fn redc(self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.neg_inv);
        let u = ((t + u128::from(m) * u128::from(self.p)) >> 64) as u64;
        if u >= self.p { u - self.p } else { u }
    }

    /// `a·b mod p` for `a, b < p`, both and the result in the ordinary form.
    #[inline(always)]
    fn mul(self, a: u64, b: u64) -> u64 {
        self.redc(u128::from(self.redc(u128::from(a) * u128::from(b))) * u128::from(self.r2))
    }

    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        if a >= b { a - b } else { a + (self.p - b) }
    }

    /// `a⁻¹ mod p` for `a` not zero modulo `p`.
    fn inv(self, a: u64) -> u64 {
        let (mut x, mut base, mut e) = (1u64, a, self.p - 2);
        while e > 0 {
            if e & 1 == 1 {
                x = self.mul(x, base);
            }
            base = self.mul(base, base);
            e >>= 1;
        }
        x
    }

    /// `2^(64·(i + 1)) mod p` for `i < len`: with them `redc(w·m_i)` is `w·2^(64·i) mod p`.
    fn powers(self, len: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(len);
        // `2^64 mod p` is `redc(2^128 mod p)`; each next one is `redc(x·2^128 mod p)`.
        let mut x = self.redc(u128::from(self.r2));
        for _ in 0..len {
            out.push(x);
            x = self.redc(u128::from(x) * u128::from(self.r2));
        }
        out
    }

    /// The residue of the magnitude `limbs` (little-endian) with `powers` from
    /// [`powers`](Field::powers).
    #[inline]
    fn of_limbs(self, limbs: &[u64], powers: &[u64]) -> u64 {
        let mut acc = 0u64;
        for (&w, &m) in limbs.iter().zip(powers) {
            if w != 0 {
                let t = self.redc(u128::from(w) * u128::from(m));
                acc += t;
                if acc >= self.p {
                    acc -= self.p;
                }
            }
        }
        acc
    }
}

/// The primes below `2^62`, from the top down.
struct Primes(u64);

impl Iterator for Primes {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        while self.0 > 3 {
            self.0 -= 2;
            if is_prime_u64(self.0) {
                return Some(self.0);
            }
        }
        None
    }
}

/// An integer being rebuilt from its residues: its limbs, and the number of primes it needs.
struct Entry {
    limbs: Vec<u64>,
    count: usize,
}

impl Gso {
    /// The data of the first `n` rows of `rows`, from their residues from [`FROM`] rows on (by
    /// [`Gso::of`] below); `None` if the rows are linearly dependent.
    pub(super) fn of_residues(rows: &[Vec<Number>], n: usize) -> Option<Gso> {
        if n < FROM {
            return Gso::of(rows, n);
        }
        let mut g: Vec<Vec<Number>> = vec![vec![Number::default(); n]; n];
        for i in 0..n {
            for j in 0..=i {
                let x = dot(&rows[i], &rows[j]);
                if j < i {
                    g[j][i] = x.clone();
                }
                g[i][j] = x;
            }
        }
        let bits: Vec<u64> = (0..n).map(|i| g[i][i].bits()).collect();
        if bits.contains(&0) {
            // A zero row.
            return None;
        }
        // `S_j = Σ_{l<j} bits(G_ll)`: `d_j < 2^S_j`.
        let mut prefix = vec![0u64; n + 1];
        for j in 0..n {
            prefix[j + 1] = prefix[j] + bits[j];
        }
        let primes_for = |b: u64| (b + 2).div_ceil(PRIME_BITS) as usize;
        // The entries: `d_1, …, d_n`, then `λ_ij` row after row, `j < i`.
        let mut entries: Vec<Entry> = Vec::with_capacity(n + n * (n - 1) / 2);
        for &bound in &prefix[1..=n] {
            entries.push(Entry {
                limbs: Vec::new(),
                count: primes_for(bound),
            });
        }
        for i in 1..n {
            for j in 0..i {
                entries.push(Entry {
                    limbs: Vec::new(),
                    count: primes_for((bits[i] + bits[j]).div_ceil(2) + prefix[j]),
                });
            }
        }
        let most = entries.iter().map(|e| e.count).max().unwrap_or(0);
        for e in &mut entries {
            e.limbs = vec![0u64; e.count + 1];
        }
        // The product of the primes skipped divides `Π d_i < 2^(Σ S_j)`.
        let skips_allowed = (prefix.iter().sum::<u64>() / PRIME_BITS) as usize + 1;
        let mut skipped = 0;
        // The products of the first `k` primes taken, for each `k ≤ most`.
        let mut moduli: Vec<Vec<u64>> = vec![vec![1]];
        let glimbs: Vec<Vec<(bool, Vec<u64>)>> = g
            .iter()
            .map(|row| row.iter().map(|x| (x.is_negative(), limbs_of(x))).collect())
            .collect();
        let longest = glimbs
            .iter()
            .flatten()
            .map(|(_, l)| l.len())
            .max()
            .unwrap_or(1);
        let mut residues = vec![0u64; n];
        let mut primes = Primes((1 << 62) + 1);
        let mut taken = 0;
        while taken < most {
            let p = primes.next().expect("primes below 2^62");
            let f = Field::new(p);
            let powers = f.powers(longest.max(most + 1));
            let rows_mod: Vec<Vec<u64>> = glimbs
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|(neg, l)| {
                            let r = f.of_limbs(l, &powers);
                            if *neg && r != 0 { p - r } else { r }
                        })
                        .collect()
                })
                .collect();
            let ple = ModMatrix::from_rows(&rows_mod, p).ple();
            if ple.rank() < n || ple.permutation().iter().enumerate().any(|(i, &q)| i != q) {
                // `p` divides a Gram determinant.
                skipped += 1;
                if skipped > skips_allowed {
                    return Gso::of(rows, n);
                }
                continue;
            }
            let (l, e) = (ple.l(), ple.e());
            // `d_{j+1} = d_j·r_jj`, `λ_ij = d_{j+1}·μ_ij`.
            let mut d = 1u64;
            for (j, r) in residues.iter_mut().enumerate() {
                d = f.mul(d, e.get(j, j));
                *r = d;
            }
            let modulus = &moduli[taken];
            let mlen = modulus.len();
            let inv = f.inv(f.of_limbs(modulus, &powers));
            let update = |entry: &mut Entry, r: u64| {
                if entry.count <= taken {
                    return;
                }
                let x = &mut entry.limbs;
                let xm = f.of_limbs(&x[..mlen], &powers);
                let t = f.mul(f.sub(r, xm), inv);
                if t != 0 {
                    let carry = addmul_1(&mut x[..mlen], modulus, t);
                    x[mlen] = x[mlen].wrapping_add(carry);
                }
            };
            let mut at = 0;
            for &r in &residues {
                update(&mut entries[at], r);
                at += 1;
            }
            for i in 1..n {
                for (j, &d) in residues[..i].iter().enumerate() {
                    update(&mut entries[at], f.mul(d, l.get(i, j)));
                    at += 1;
                }
            }
            // The next product, `modulus·p`.
            let mut next = Vec::with_capacity(mlen + 1);
            let mut carry = 0u128;
            for &w in modulus {
                let t = u128::from(w) * u128::from(p) + carry;
                next.push(t as u64);
                carry = t >> 64;
            }
            if carry != 0 {
                next.push(carry as u64);
            }
            moduli.push(next);
            taken += 1;
        }
        // The residues of least magnitude.
        let value = |entry: &Entry| -> Number {
            let q = &moduli[entry.count];
            let x = &entry.limbs;
            let top = x.iter().rposition(|&w| w != 0).map_or(0, |i| i + 1);
            if top == 0 {
                return Number::default();
            }
            let xv = from_limbs(&x[..top], ByteOrder::LittleEndian);
            let qv = from_limbs(q, ByteOrder::LittleEndian);
            if &xv + &xv > qv { xv - qv } else { xv }
        };
        let mut d = Vec::with_capacity(n + 1);
        d.push(Number::from(1u8));
        for e in &entries[..n] {
            d.push(value(e));
        }
        let mut lam: Vec<Vec<Number>> = Vec::with_capacity(n);
        lam.push(Vec::new());
        let mut at = n;
        for i in 1..n {
            lam.push(
                (0..i)
                    .map(|_| {
                        let v = value(&entries[at]);
                        at += 1;
                        v
                    })
                    .collect(),
            );
        }
        // `Gso::of` keeps row 0 with no coefficients; a zero determinant is a dependence the
        // skipped primes would have caught.
        debug_assert!(d.iter().all(|x| x.is_positive()));
        Some(Gso { d, lam })
    }
}
