//! What the three forms of curve have in common: a group whose points are held in a projective
//! form, and the scalar multiplications built on its additions and doublings.
//!
//! - One scalar: the width-`w` non-adjacent form, odd digits with at least `w - 1` zeros between
//!   them, over a table of the point's odd multiples.
//! - A few scalars: Straus's method, one run of doublings for all of them.
//! - Many scalars: Pippenger's method, whose buckets grow with the logarithm of their number.
//! - A point known beforehand: its multiples by each digit at each position of a signed fixed
//!   window, computed once, so that a product takes additions only.

use super::field::Field;

/// The two coordinates of a point in the field `F`.
pub(super) type Coordinates<F> = (<F as Field>::E, <F as Field>::E);

/// A group of points of a curve.
pub(super) trait Group {
    /// The field of the coordinates.
    type F: Field;
    /// A point in a projective form.
    type Proj: Clone;
    /// A point prepared to be added, as a table holds it.
    type Cached: Clone;

    /// The elements a cached point is stored with, the flag of [`store_cached`] aside.
    ///
    /// [`store_cached`]: Group::store_cached
    const CACHED: usize;

    /// The field.
    fn field(&self) -> &Self::F;
    /// The neutral element.
    fn identity(&self) -> Self::Proj;
    /// Whether `p` is the neutral element.
    fn is_identity(&self, p: &Self::Proj) -> bool;
    /// The point stored in `x`.
    fn load(&self, x: &[u64]) -> Self::Proj;
    /// Stores `p` in `r`.
    fn store(&self, p: &Self::Proj, r: &mut [u64]);
    /// `-p`.
    fn neg(&self, p: &Self::Proj) -> Self::Proj;
    /// `2 p`.
    fn double(&self, p: &Self::Proj) -> Self::Proj;
    /// `p + q`.
    fn add(&self, p: &Self::Proj, q: &Self::Proj) -> Self::Proj;
    /// Whether `p` and `q` are the same point.
    fn equal(&self, p: &Self::Proj, q: &Self::Proj) -> bool;
    /// The points of `p` prepared to be added, with one inversion for them all if any.
    fn cache(&self, p: &[Self::Proj]) -> Vec<Self::Cached>;
    /// `p + c`.
    fn add_cached(&self, p: &Self::Proj, c: &Self::Cached) -> Self::Proj;
    /// `-c`.
    fn neg_cached(&self, c: &Self::Cached) -> Self::Cached;
    /// Stores `c` in `r`.
    fn store_cached(&self, c: &Self::Cached, r: &mut [u64]);
    /// The cached point stored in `x`.
    fn load_cached(&self, x: &[u64]) -> Self::Cached;
    /// The point `(x, y)`.
    fn of_affine(&self, x: &<Self::F as Field>::E, y: &<Self::F as Field>::E) -> Self::Proj;
    /// The affine coordinates of each point, `None` for one at infinity, with one inversion.
    fn affine_all(&self, p: &[Self::Proj]) -> Vec<Option<Coordinates<Self::F>>>;

    /// Doubles `p` `n` times.
    #[inline]
    fn double_n(&self, p: &mut Self::Proj, n: usize) {
        for _ in 0..n {
            *p = self.double(p);
        }
    }

    /// `p - c`.
    #[inline]
    fn sub_cached(&self, p: &Self::Proj, c: &Self::Cached) -> Self::Proj {
        self.add_cached(p, &self.neg_cached(c))
    }

    /// `k p` for the magnitude `k` with the sign `negative`.
    fn mul(&self, p: &Self::Proj, k: &[u64], negative: bool) -> Self::Proj {
        mul(self, p, k, negative)
    }
}

/// The bits of the magnitude `k`.
pub(super) fn bit_length(k: &[u64]) -> usize {
    match k.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * i + 64 - k[i].leading_zeros() as usize,
        None => 0,
    }
}

/// The `n` bits of `k` from bit `at`, with zeros above it.
#[inline(always)]
fn bits_at(k: &[u64], at: usize, n: u32) -> u64 {
    let (l, s) = (at / 64, at % 64);
    let low = k.get(l).copied().unwrap_or(0) >> s;
    let high = if s == 0 {
        0
    } else {
        k.get(l + 1).copied().unwrap_or(0) << (64 - s)
    };
    (low | high) & ((1 << n) - 1)
}

/// The width of the non-adjacent form for a scalar of `bits` bits: the one that minimizes the
/// `2^(w - 2)` additions of the table and the `bits / (w + 1)` of the product. Measured on an Apple
/// M4 Max, a product on P-256 took 86,000, 85,000 and 89,000 cycles for widths 4, 5 and 6, on P-384
/// 327,000, 314,000 and 321,000, and on P-521 457,000, 452,000 and 452,000.
pub(super) fn window(bits: usize) -> u32 {
    match bits {
        0..=23 => 2,
        24..=75 => 3,
        76..=199 => 4,
        200..=639 => 5,
        640..=1279 => 6,
        1280..=3071 => 7,
        _ => 8,
    }
}

/// The width-`w` non-adjacent form of `k`, the lowest digit first: odd digits below `2^(w - 1)` in
/// absolute value, at least `w - 1` zeros after each.
pub(super) fn wnaf(k: &[u64], w: u32) -> Vec<i8> {
    let len = bit_length(k) + 1;
    let mut digits = vec![0i8; len];
    let width = 1u64 << w;
    let mut carry = 0;
    let mut at = 0;
    while at < len {
        let d = carry + bits_at(k, at, w);
        if d & 1 == 0 {
            at += 1;
            continue;
        }
        if d < width / 2 {
            carry = 0;
            digits[at] = d as i8;
        } else {
            carry = 1;
            digits[at] = (d as i64 - width as i64) as i8;
        }
        at += w as usize;
    }
    digits
}

/// `p, 3 p, 5 p`, ..., `n` of them.
pub(super) fn odd_multiples<G: Group + ?Sized>(g: &G, p: &G::Proj, n: usize) -> Vec<G::Proj> {
    let mut table = Vec::with_capacity(n);
    table.push(p.clone());
    if n > 1 {
        let two = g.double(p);
        for i in 1..n {
            let next = g.add(&table[i - 1], &two);
            table.push(next);
        }
    }
    table
}

/// `k p` by the non-adjacent form.
pub(super) fn mul<G: Group + ?Sized>(g: &G, p: &G::Proj, k: &[u64], negative: bool) -> G::Proj {
    let bits = bit_length(k);
    if bits == 0 || g.is_identity(p) {
        return g.identity();
    }
    let w = window(bits);
    let digits = wnaf(k, w);
    let multiples = odd_multiples(g, p, 1 << (w - 2));
    let table = g.cache(&multiples);
    let mut i = digits.len();
    while digits[i - 1] == 0 {
        i -= 1;
    }
    i -= 1;
    // The top digit of a positive number is positive.
    let mut r = multiples[(digits[i] as usize - 1) / 2].clone();
    let mut zeros = 0;
    while i > 0 {
        i -= 1;
        zeros += 1;
        let d = digits[i];
        if d != 0 {
            g.double_n(&mut r, zeros);
            zeros = 0;
            r = if d > 0 {
                g.add_cached(&r, &table[(d as usize - 1) / 2])
            } else {
                g.sub_cached(&r, &table[((-d) as usize - 1) / 2])
            };
        }
    }
    g.double_n(&mut r, zeros);
    if negative { g.neg(&r) } else { r }
}

/// From this many terms [`multi_mul`] takes Pippenger's method instead of Straus's: measured on an
/// Apple M4 Max on P-256, 64 terms took 1.4 million cycles either way, 96 took 2.10 million by
/// Straus's and 1.94 by Pippenger's, and 112 took 2.40 and 2.19.
const PIPPENGER: usize = 64;

/// The sum of `k p`, for the magnitudes `k` with their signs and the points `p`.
pub(super) fn multi_mul<G: Group + ?Sized>(g: &G, terms: &[(&[u64], bool, G::Proj)]) -> G::Proj {
    if terms.len() < PIPPENGER {
        straus(g, terms)
    } else {
        pippenger(g, terms)
    }
}

/// Straus's method: the non-adjacent forms of all the scalars read together, one run of doublings
/// for all of them.
fn straus<G: Group + ?Sized>(g: &G, terms: &[(&[u64], bool, G::Proj)]) -> G::Proj {
    let mut digits = Vec::with_capacity(terms.len());
    let mut all = Vec::new();
    let mut start = Vec::with_capacity(terms.len());
    let mut len = 0;
    for (k, negative, p) in terms {
        let bits = bit_length(k);
        let w = window(bits);
        start.push(all.len());
        let base = if *negative { g.neg(p) } else { p.clone() };
        all.extend(odd_multiples(g, &base, 1 << (w - 2)));
        let d = wnaf(k, w);
        len = len.max(d.len());
        digits.push(d);
    }
    let table = g.cache(&all);
    let mut r = g.identity();
    for i in (0..len).rev() {
        r = g.double(&r);
        for (t, d) in digits.iter().enumerate() {
            let d = d.get(i).copied().unwrap_or(0);
            if d > 0 {
                r = g.add_cached(&r, &table[start[t] + (d as usize - 1) / 2]);
            } else if d < 0 {
                r = g.sub_cached(&r, &table[start[t] + ((-d) as usize - 1) / 2]);
            }
        }
    }
    r
}

/// The width of Pippenger's windows for `n` terms, `log2 n - 1`, as each window costs an addition
/// per term and two per bucket: measured on an Apple M4 Max on P-256, a bit wider than `log2 n - 2`
/// took 8 % less time for 128 terms, 5 % for 256 and 512, and as much for 1,024 and 4,096.
fn bucket_bits(n: usize) -> u32 {
    (usize::BITS - n.leading_zeros()).saturating_sub(2).max(2)
}

/// The digit of the signed window at bit `at` of `k`, `c` bits wide, with the `carry` from the
/// window below, which it updates: digits from `-2^(c - 1)` to `2^(c - 1)`.
#[inline(always)]
fn signed_digit(k: &[u64], at: usize, c: u32, carry: &mut u64) -> i64 {
    let d = bits_at(k, at, c) + *carry;
    if d > 1 << (c - 1) {
        *carry = 1;
        d as i64 - (1i64 << c)
    } else {
        *carry = 0;
        d as i64
    }
}

/// Pippenger's method: the terms sorted by digit into buckets for each window, each window
/// computed from the top bucket down with a running sum, so that a point in bucket `d` counts `d`
/// times at a cost of two additions for each bucket.
fn pippenger<G: Group + ?Sized>(g: &G, terms: &[(&[u64], bool, G::Proj)]) -> G::Proj {
    let c = bucket_bits(terms.len());
    let bits = terms.iter().map(|t| bit_length(t.0)).max().unwrap_or(0);
    let windows = (bits + 1).div_ceil(c as usize);
    let based: Vec<G::Proj> = terms
        .iter()
        .map(|(_, negative, p)| if *negative { g.neg(p) } else { p.clone() })
        .collect();
    let table = g.cache(&based);
    // The digits of each term in each window, the lowest window first.
    let mut digits = vec![0i32; windows * terms.len()];
    for (t, (k, _, _)) in terms.iter().enumerate() {
        let mut carry = 0;
        for w in 0..windows {
            digits[w * terms.len() + t] = signed_digit(k, w * c as usize, c, &mut carry) as i32;
        }
    }
    let mut r = g.identity();
    let mut buckets = vec![g.identity(); 1 << (c - 1)];
    for w in (0..windows).rev() {
        g.double_n(&mut r, c as usize);
        for b in buckets.iter_mut() {
            *b = g.identity();
        }
        for (t, entry) in table.iter().enumerate() {
            let d = digits[w * terms.len() + t];
            if d > 0 {
                let b = (d - 1) as usize;
                buckets[b] = g.add_cached(&buckets[b], entry);
            } else if d < 0 {
                let b = (-d - 1) as usize;
                buckets[b] = g.sub_cached(&buckets[b], entry);
            }
        }
        let mut running = g.identity();
        let mut sum = g.identity();
        for b in buckets.iter().rev() {
            running = g.add(&running, b);
            sum = g.add(&sum, &running);
        }
        r = g.add(&r, &sum);
    }
    r
}

/// The multiples of a point fixed in advance, for scalars of up to `bits` bits: for each window of
/// `width` bits, the digits from 1 to `2^(width - 1)` times the window's power of two, stored.
pub struct Table {
    /// The limbs of all the entries.
    pub(super) limbs: Vec<u64>,
    /// The bits of a window.
    pub(super) width: u32,
    /// The bits of the longest scalar.
    pub(super) bits: usize,
}

/// The window of a table for scalars of `bits` bits: each window costs an addition in a product, and
/// each bit of width doubles the table. Measured on an Apple M4 Max, P-256's generator took 12,200
/// and 10,800 cycles for widths 7 and 8, and P-521's 69,700 and 59,500; at width 8 a table takes
/// 300 KiB on P-256 and 1.3 MiB on P-521.
fn table_width(bits: usize) -> u32 {
    match bits {
        0..=63 => 4,
        64..=127 => 5,
        128..=191 => 6,
        _ => 8,
    }
}

/// The limbs of a cached point: `CACHED` elements and a flag.
fn cached_limbs<G: Group + ?Sized>(g: &G) -> usize {
    G::CACHED * g.field().limbs() + 1
}

/// The table of `p` for scalars of up to `bits` bits.
pub(super) fn prepare<G: Group + ?Sized>(g: &G, p: &G::Proj, bits: usize) -> Table {
    let width = table_width(bits);
    let windows = (bits + 1).div_ceil(width as usize);
    let digits = 1usize << (width - 1);
    let mut all = Vec::with_capacity(windows * digits);
    let mut base = p.clone();
    for _ in 0..windows {
        let mut m = base.clone();
        all.push(m.clone());
        for _ in 1..digits {
            m = g.add(&m, &base);
            all.push(m.clone());
        }
        // `2^(width - 1) base`, doubled.
        base = g.double(&m);
    }
    let size = cached_limbs(g);
    let mut limbs = vec![0; all.len() * size];
    for (c, r) in g.cache(&all).iter().zip(limbs.chunks_mut(size)) {
        g.store_cached(c, r);
    }
    Table { limbs, width, bits }
}

/// `k p` from the table of `p`, for `k` of at most `table.bits` bits.
pub(super) fn prepared_mul<G: Group + ?Sized>(
    g: &G,
    table: &Table,
    k: &[u64],
    negative: bool,
) -> G::Proj {
    let width = table.width;
    let windows = (table.bits + 1).div_ceil(width as usize);
    let digits = 1usize << (width - 1);
    let size = cached_limbs(g);
    let mut r = g.identity();
    let mut carry = 0;
    for w in 0..windows {
        let d = signed_digit(k, w * width as usize, width, &mut carry);
        if d != 0 {
            let at = (w * digits + d.unsigned_abs() as usize - 1) * size;
            let c = g.load_cached(&table.limbs[at..at + size]);
            r = if d > 0 {
                g.add_cached(&r, &c)
            } else {
                g.sub_cached(&r, &c)
            };
        }
    }
    if negative { g.neg(&r) } else { r }
}
