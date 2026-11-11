//! The Chinese remainder theorem: a few congruences modulo words in registers, two of a few limbs
//! on the stack, more modulo words joined one at a time on limbs, as Garner's method does, and from
//! 150 of them by the product and remainder trees in limbs; longer moduli, or the results of groups
//! of words, joined in pairs up a balanced tree, or by the trees of numbers when there are many.

use super::super::div::limbs::{div_2by1, invert_limb};
use super::super::mul::limbs::{mul, mul_itch};
use super::super::pow::montgomery::{rem, rem_itch};
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
use super::int::join;
use super::ring::{from_limbs, magnitude_64, magnitude_128, read_limbs};
use super::word;

/// Up to this many congruences modulo words are joined one at a time; more are joined in groups of
/// this many, whose results are joined in pairs up a tree, which takes subquadratic time in all.
pub(in super::super) const CHAIN: usize = 64;

/// A word prepared for remainders by it: shifted so that its top bit is set, with the reciprocal.
#[derive(Clone, Copy)]
struct Word {
    /// Its leading zeros.
    s: u32,
    /// `n << s`.
    d: u64,
    /// The reciprocal of `d`.
    v: u64,
}

impl Word {
    fn new(n: u64) -> Word {
        let s = n.leading_zeros();
        let d = n << s;
        Word {
            s,
            d,
            v: invert_limb(d),
        }
    }

    /// The quotient `v / n` for `v < n B`, by the reciprocal.
    #[inline(always)]
    fn quotient(&self, v: u128) -> u64 {
        let v = v << self.s;
        // `v < n B`, so `v 2^s < d B`: the high limb is below `d`.
        div_2by1((v >> 64) as u64, v as u64, self.d, self.v).0
    }

    /// `a b mod n` for `a, b < n`, by the reciprocal: the product shifted by `s` is below `d B`.
    #[inline(always)]
    fn mul_mod(&self, a: u64, b: u64) -> u64 {
        let p = u128::from(a) * u128::from(b);
        let p = p << self.s;
        let (hi, lo) = ((p >> 64) as u64, p as u64);
        // The bits shifted out of the top: none, as `a b < n^2 <= n B`, so `(a b) 2^s < d B`.
        div_2by1(hi, lo, self.d, self.v).1 >> self.s
    }

    /// The remainders of the limbs `x` and `y`, of the same length, by the word, in one pass:
    /// both shifted left by `s` bits on the fly, divided from the top by the shifted word, and the
    /// remainders shifted back.
    #[inline]
    fn rem2(&self, x: &[u64], y: &[u64]) -> (u64, u64) {
        let (s, d, v) = (self.s, self.d, self.v);
        let n = x.len();
        let (mut rx, mut ry) = (0u64, 0u64);
        if s == 0 {
            for i in (0..n).rev() {
                rx = div_2by1(rx, x[i], d, v).1;
                ry = div_2by1(ry, y[i], d, v).1;
            }
            return (rx, ry);
        }
        // The limb above the top, the bits shifted out of it.
        rx = div_2by1(0, x[n - 1] >> (64 - s), d, v).1;
        ry = div_2by1(0, y[n - 1] >> (64 - s), d, v).1;
        for i in (0..n).rev() {
            let below_x = if i > 0 { x[i - 1] >> (64 - s) } else { 0 };
            let below_y = if i > 0 { y[i - 1] >> (64 - s) } else { 0 };
            rx = div_2by1(rx, (x[i] << s) | below_x, d, v).1;
            ry = div_2by1(ry, (y[i] << s) | below_y, d, v).1;
        }
        (rx >> s, ry >> s)
    }
}

/// `x += m t` on limbs, `x` with room for the carry.
fn add_times(x: &mut Vec<u64>, m: &[u64], t: u64) {
    if x.len() < m.len() + 1 {
        x.resize(m.len() + 1, 0);
    }
    let mut carry = 0u64;
    for i in 0..m.len() {
        let p = u128::from(m[i]) * u128::from(t) + u128::from(x[i]) + u128::from(carry);
        x[i] = p as u64;
        carry = (p >> 64) as u64;
    }
    let mut i = m.len();
    while carry != 0 {
        if i == x.len() {
            x.push(0);
        }
        let (s, c) = x[i].overflowing_add(carry);
        x[i] = s;
        carry = u64::from(c);
        i += 1;
    }
}

/// `m *= t` on limbs.
fn times(m: &mut Vec<u64>, t: u64) {
    let mut carry = 0u64;
    for l in m.iter_mut() {
        let p = u128::from(*l) * u128::from(t) + u128::from(carry);
        *l = p as u64;
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        m.push(carry);
    }
}

/// The congruences `x = r_i (mod m_i)` modulo words, `r_i < m_i`, joined one at a time: `(x, l)`
/// as limbs, `l` the least common multiple, `x < l`; or `None` when they contradict each other.
///
/// Each step brings in `x = s (mod n)`: with `g = gcd(l, n)`, which must divide `s - x`, and
/// `n' = n / g`, `x + l t` with `t = ((s - x) / g) ((l / g)^(-1)) mod n'` is the solution modulo
/// `l n'`. It needs `l` and `x` modulo `n`, by one pass over both.
pub(in super::super) fn chain(c: &[(u64, u64)]) -> Option<(Vec<u64>, Vec<u64>)> {
    let (x, l, i) = match registers(c)? {
        Registers::Done(x, l) => (x, l, c.len()),
        Registers::Full(x, l, i) => (x, l, i),
    };
    limbs_from(c, x, l, i)
}

/// [`chain`]'s steps on limbs, from the solution `x` modulo `l` of the congruences before the one
/// at `i`.
pub(in super::super) fn limbs_from(
    c: &[(u64, u64)],
    x: u128,
    l: u128,
    i: usize,
) -> Option<(Vec<u64>, Vec<u64>)> {
    let mut xl = vec![x as u64, (x >> 64) as u64];
    let mut ll = vec![l as u64, (l >> 64) as u64];
    for &(s, n) in &c[i..] {
        if n == 1 {
            continue;
        }
        let w = Word::new(n);
        xl.resize(ll.len(), 0);
        let (xr, lr) = w.rem2(&xl, &ll);
        let Some((t, n1)) = step(s, n, xr, lr)? else {
            continue;
        };
        add_times(&mut xl, &ll, t);
        times(&mut ll, n1);
        while xl.last() == Some(&0) && xl.len() > 1 {
            xl.pop();
        }
    }
    Some((xl, ll))
}

/// How far [`registers`] got: all the congruences, or those before the one at the index, whose
/// modulus would pass two words.
pub(in super::super) enum Registers {
    /// The solution `x` modulo `l` of all of them.
    Done(u128, u128),
    /// The solution modulo `l` of those before the index.
    Full(u128, u128, usize),
}

/// The steps of [`chain`] in registers, while the modulus fits in two words: no vector, and the
/// remainders and products modulo each word by its reciprocal. `None` when the congruences
/// contradict each other.
#[inline]
pub(in super::super) fn registers(c: &[(u64, u64)]) -> Option<Registers> {
    let (mut x, mut l) = (0u128, 1u128);
    for (i, &(s, n)) in c.iter().enumerate() {
        if n == 1 {
            continue;
        }
        let w = Word::new(n);
        let (xr, lr) = w.rem2(&[x as u64, (x >> 64) as u64], &[l as u64, (l >> 64) as u64]);
        let Some((t, n1)) = step_by(&w, s, n, xr, lr)? else {
            continue;
        };
        let Some(next) = l.checked_mul(u128::from(n1)) else {
            return Some(Registers::Full(x, l, i));
        };
        x += l * u128::from(t);
        l = next;
    }
    Some(Registers::Done(x, l))
}

/// [`step`] with the products modulo `n` by its reciprocal in `w`, for the coprime case.
#[inline]
fn step_by(w: &Word, s: u64, n: u64, xr: u64, lr: u64) -> Option<Option<(u64, u64)>> {
    if let Some(inv) = word::inverse(lr, n) {
        let d = if s >= xr { s - xr } else { n - (xr - s) };
        return Some(Some((w.mul_mod(d, inv), n)));
    }
    step(s, n, xr, lr)
}

/// One step of [`chain`]: for `x = xr` and `l = lr` modulo `n`, the `t` and `n' = n / gcd(l, n)`
/// that make `x + l t` the solution modulo `l n'`; `Some(None)` when `n'` is 1 (nothing to add),
/// and `None` when the congruences contradict each other. The inverse comes first: it exists
/// exactly when `l` and `n` are coprime, the common case, which needs no greatest common divisor.
#[inline]
fn step(s: u64, n: u64, xr: u64, lr: u64) -> Option<Option<(u64, u64)>> {
    let d = if s >= xr { s - xr } else { n - (xr - s) };
    if let Some(inv) = word::inverse(lr, n) {
        return Some(Some((word::mul_mod(d, inv, n), n)));
    }
    let g = word::gcd(lr, n);
    if d % g != 0 {
        return None;
    }
    let n1 = n / g;
    if n1 == 1 {
        return Some(None);
    }
    let inv = word::inverse((lr / g) % n1, n1)?;
    Some(Some((word::mul_mod((d / g) % n1, inv, n1), n1)))
}

/// The congruences `(r_i, m_i)`, `r_i < m_i`, joined in pairs up a balanced tree.
pub(in super::super) fn tree(mut c: Vec<(Number, Number)>) -> Option<(Number, Number)> {
    if c.is_empty() {
        return Some((
            Number::new_from_u8(0, ByteOrder::LittleEndian),
            Number::new_from_u8(1, ByteOrder::LittleEndian),
        ));
    }
    while c.len() > 1 {
        let mut next = Vec::with_capacity(c.len().div_ceil(2));
        let mut it = c.into_iter();
        while let Some((r, m)) = it.next() {
            match it.next() {
                Some((s, n)) => next.push(join(&r, &m, &s, &n)?),
                None => next.push((r, m)),
            }
        }
        c = next;
    }
    c.pop()
}

/// The congruences `(r_i, m_i)`, `r_i < m_i`, for pairwise coprime moduli, by the product and
/// remainder trees (as PARI's `ZV_chinese` and FLINT's `fmpz_multi_CRT` take them): with `P` the
/// product of the moduli, `c_i = (P / m_i) mod m_i = (P mod m_i^2) / m_i` comes down the tree of
/// the remainders of `P` by the squares of the products, `t_i = r_i c_i^(-1) mod m_i` at each leaf,
/// and `x = sum t_i P / m_i` goes up the product tree as `x_l M_r + x_r M_l`, reduced by `P` at the
/// top. `None` when a `c_i` has no inverse: the moduli are not pairwise coprime.
pub(in super::super) fn fast(c: &[(Number, Number)]) -> Option<(Number, Number)> {
    let k = c.len();
    // The product tree, the moduli at the bottom.
    let mut levels: Vec<Vec<Number>> = vec![c.iter().map(|(_, m)| m.clone()).collect()];
    while levels.last().is_some_and(|l| l.len() > 1) {
        let below = levels.last().map(|l| l.as_slice()).unwrap_or(&[]);
        let up: Vec<Number> = below
            .chunks(2)
            .map(|p| {
                if p.len() == 2 {
                    &p[0] * &p[1]
                } else {
                    p[0].clone()
                }
            })
            .collect();
        levels.push(up);
    }
    let p = levels.last()?[0].clone();
    // Down: the remainders of `P` by the squares.
    let mut rems = vec![p.clone()];
    for level in levels.iter().rev().skip(1) {
        let mut next = Vec::with_capacity(level.len());
        for (i, m) in level.iter().enumerate() {
            let parent = &rems[i / 2];
            next.push(parent.rem_euclid(&(m * m)));
        }
        rems = next;
    }
    // The leaves, in words for the moduli of a word.
    let mut xs: Vec<Number> = Vec::with_capacity(k);
    for (i, (r, m)) in c.iter().enumerate() {
        if let (Some(mw), Some(rem)) = (magnitude_64(m), magnitude_128(&rems[i])) {
            let rw = magnitude_64(r).unwrap_or(0);
            let ci = ((rem / u128::from(mw)) % u128::from(mw)) as u64;
            let inv = word::inverse(ci, mw)?;
            xs.push(Number::new_from_u64(
                word::mul_mod(rw, inv, mw),
                ByteOrder::LittleEndian,
            ));
            continue;
        }
        let ci = rems[i].div_exact(m).rem_euclid(m);
        let inv = if *m == 1u8 {
            Number::new_from_u8(0, ByteOrder::LittleEndian)
        } else {
            ci.mod_inverse(m)?
        };
        xs.push((r * &inv).rem_euclid(m));
    }
    // Up: `x_l M_r + x_r M_l`.
    for level in &levels[..levels.len() - 1] {
        let mut next = Vec::with_capacity(xs.len().div_ceil(2));
        for (i, pair) in xs.chunks(2).enumerate() {
            if pair.len() == 2 {
                next.push(&(&pair[0] * &level[2 * i + 1]) + &(&pair[1] * &level[2 * i]));
            } else {
                next.push(pair[0].clone());
            }
        }
        xs = next;
    }
    let x = xs.pop()?.rem_euclid(&p);
    Some((x, p))
}

/// From this many congruences on, they are first tried as pairwise coprime by [`fast`] when a
/// modulus is longer than a word: measured on an Apple M4 Max, 8 primes of 1024 bits took 226,000
/// cycles so, against 276,000 for PARI's `ZV_chinese`.
pub(in super::super) const FAST: usize = 8;

/// From this many congruences modulo words on, they are tried by [`fast_words`]: measured on an
/// Apple M4 Max, the chain of [`chain`] took 121,000 cycles for 100 primes of 64 bits, and the
/// trees 266,000 for 150 against about 272,000 for the chain, whose time grows with the square of
/// the count. 12 under Miri, so that its tests take the trees.
pub(in super::super) const FAST_WORDS: usize = if cfg!(miri) { 12 } else { 150 };

/// The congruences modulo words joined: one chain for few, groups joined up a tree for many.
pub(in super::super) fn words(c: &[(u64, u64)]) -> Option<(Number, Number)> {
    if c.len() <= CHAIN {
        let (x, l) = chain(c)?;
        return Some((
            from_limbs(&x, ByteOrder::LittleEndian),
            from_limbs(&l, ByteOrder::LittleEndian),
        ));
    }
    let mut parts = Vec::with_capacity(c.len().div_ceil(CHAIN));
    for group in c.chunks(CHAIN) {
        let (x, l) = chain(group)?;
        parts.push((
            from_limbs(&x, ByteOrder::LittleEndian),
            from_limbs(&l, ByteOrder::LittleEndian),
        ));
    }
    tree(parts)
}

// Two congruences on the stack

/// Up to this many limbs a modulus, two congruences are joined on the stack by [`pair`]: the
/// products are the schoolbook kernels' and the remainders the schoolbook division's, neither of
/// which needs working memory beyond the remainder's own.
pub(in super::super) const PAIR: usize = 16;

/// The working memory of the remainders of [`pair`]: the divisor and the dividend shifted, with the
/// limb above it, and the quotient.
const PAIR_SCRATCH: usize = PAIR + 4 * PAIR + 8;

/// Limbs on the stack, aligned to 16 bytes for the kernels.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Buf<const N: usize>([u64; N]);

/// Sets `r` to the `n` limbs of the `nu` limbs `u` modulo the `n` limbs `m`, whose top one is not
/// zero, and returns the number of limbs of the remainder.
fn rem_into(r: &mut [u64], u: &[u64], m: &[u64]) -> usize {
    let (nu, n) = (u.len(), m.len());
    debug_assert!(n <= PAIR && nu <= 2 * PAIR + 1 && r.len() >= n);
    let mut s = Buf([0u64; PAIR_SCRATCH]);
    let s = Scratch::new(s.0.as_mut_ptr(), PAIR_SCRATCH);
    // SAFETY: the scratch holds `rem_itch(nu, n)` limbs, which the schoolbook division takes for
    // divisors below `DC_DIV` (40) limbs, and starts on 16 bytes; `r` has room for `n` limbs.
    unsafe { rem(r.as_mut_ptr(), u.as_ptr(), nu, 0, m.as_ptr(), n, s) };
    trimmed_len(&r[..n])
}

/// The limbs without the zeros at the top.
fn trimmed_len(x: &[u64]) -> usize {
    x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)
}

/// Sets `r` to the product of the limbs `x` and `y`, not empty, and returns its number of limbs.
fn mul_into(r: &mut [u64], x: &[u64], y: &[u64]) -> usize {
    let ((p, np), (q, nq)) = if x.len() >= y.len() {
        ((x.as_ptr(), x.len()), (y.as_ptr(), y.len()))
    } else {
        ((y.as_ptr(), y.len()), (x.as_ptr(), x.len()))
    };
    debug_assert!(nq <= PAIR && np + nq <= r.len());
    // SAFETY: the shorter operand has at most 16 limbs, below Karatsuba's threshold, so the product
    // is the schoolbook kernels', with no working memory; `r` has room for it, apart from both.
    unsafe { mul(r.as_mut_ptr(), p, np, q, nq, Scratch::none()) };
    trimmed_len(&r[..np + nq])
}

/// The residue `r mod m` of any number `r` (its magnitude at most `2 PAIR` limbs) into `out`, for
/// the limbs `m`: its number of limbs, or `None` when `r` is longer.
fn residue_into(out: &mut [u64; PAIR], r: &Number, m: &[u64]) -> Option<usize> {
    let mut x = Buf([0u64; 2 * PAIR]);
    let nx = r.significant().div_ceil(8);
    if nx > 2 * PAIR {
        return None;
    }
    // SAFETY: the limbs are plain integers; `read_limbs` writes the first `nx`.
    let xs = unsafe {
        core::slice::from_raw_parts_mut(x.0.as_mut_ptr().cast::<core::mem::MaybeUninit<u64>>(), nx)
    };
    read_limbs(r, xs);
    let n = rem_into(out, &x.0[..nx], m);
    if r.is_negative() && n != 0 {
        // `m - (|r| mod m)`.
        let mut borrow = false;
        for (i, o) in out[..m.len()].iter_mut().enumerate() {
            let y = if i < n { *o } else { 0 };
            let (d, b1) = m[i].overflowing_sub(y);
            let (d, b2) = d.overflowing_sub(u64::from(borrow));
            *o = d;
            borrow = b1 | b2;
        }
        return Some(trimmed_len(&out[..m.len()]));
    }
    Some(n)
}

/// The congruences `x = r1 (mod m1)` and `x = r2 (mod m2)` joined on the stack, for moduli of at
/// most [`PAIR`] limbs and residues of any sign of at most twice that: `Some((x, m1 m2))` in the
/// byte order `order`, or `None` when that does not apply, the moduli having a common factor or
/// being too long, and the caller joins them as numbers. By `x = r1 + m1 t`, `t = (r2 - r1)
/// m1^(-1) mod m2`, with the inverse by [`Number::mod_inverse`] and the rest in limbs: no number
/// but the inverse and the two results.
pub(in super::super) fn pair(
    (r1, m1): (&Number, &Number),
    (r2, m2): (&Number, &Number),
    order: ByteOrder,
) -> Option<(Number, Number)> {
    let mut a = [0u64; PAIR];
    let mut b = [0u64; PAIR];
    let (n1, n2) = (m1.significant().div_ceil(8), m2.significant().div_ceil(8));
    if n1 > PAIR || n2 > PAIR || n1 == 0 || n2 == 0 {
        return None;
    }
    // SAFETY (both): plain integers; `read_limbs` writes the first `n` limbs.
    let n1 = read_limbs(m1, unsafe {
        core::slice::from_raw_parts_mut(a.as_mut_ptr().cast(), n1)
    });
    let n2 = read_limbs(m2, unsafe {
        core::slice::from_raw_parts_mut(b.as_mut_ptr().cast(), n2)
    });
    let (ma, mb) = (&a[..n1], &b[..n2]);
    if mb == [1] || ma == [1] {
        return None;
    }
    let inv = m1.mod_inverse(m2)?;
    let mut iv = [0u64; PAIR];
    let ni = inv.significant().div_ceil(8);
    // SAFETY: as above; the inverse is below `m2`.
    let ni = read_limbs(&inv, unsafe {
        core::slice::from_raw_parts_mut(iv.as_mut_ptr().cast(), ni)
    });
    let mut x1 = [0u64; PAIR];
    let nx1 = residue_into(&mut x1, r1, ma)?;
    let mut x2 = [0u64; PAIR];
    residue_into(&mut x2, r2, mb)?;
    // `r1 mod m2`, then `d = r2 - r1 mod m2`.
    let mut y = [0u64; PAIR];
    let ny = if nx1 == 0 {
        0
    } else {
        rem_into(&mut y, &x1[..nx1], mb)
    };
    let mut d = [0u64; PAIR];
    {
        let mut borrow = false;
        for i in 0..n2 {
            let (s, b1) = x2[i].overflowing_sub(if i < ny { y[i] } else { 0 });
            let (s, b2) = s.overflowing_sub(u64::from(borrow));
            d[i] = s;
            borrow = b1 | b2;
        }
        if borrow {
            // Plus `m2`.
            let mut carry = false;
            for i in 0..n2 {
                let (s, c1) = d[i].overflowing_add(mb[i]);
                let (s, c2) = s.overflowing_add(u64::from(carry));
                d[i] = s;
                carry = c1 | c2;
            }
        }
    }
    let nd = trimmed_len(&d[..n2]);
    // `t = d inv mod m2`.
    let mut t = [0u64; PAIR];
    let nt = if nd == 0 || ni == 0 {
        0
    } else {
        let mut p = [0u64; 2 * PAIR];
        let np = mul_into(&mut p, &d[..nd], &iv[..ni]);
        rem_into(&mut t, &p[..np], mb)
    };
    // `x = r1 + m1 t` and `l = m1 m2`.
    let mut x = [0u64; 2 * PAIR + 1];
    let mut nx = 0;
    if nt > 0 {
        nx = mul_into(&mut x, ma, &t[..nt]);
    }
    let len = nx.max(nx1);
    let mut carry = false;
    for i in 0..len {
        let (s, c1) = x[i].overflowing_add(if i < nx1 { x1[i] } else { 0 });
        let (s, c2) = s.overflowing_add(u64::from(carry));
        x[i] = s;
        carry = c1 | c2;
    }
    x[len] = u64::from(carry);
    let nx = trimmed_len(&x[..=len]);
    let mut l = [0u64; 2 * PAIR];
    let nl = mul_into(&mut l, ma, mb);
    Some((from_limbs(&x[..nx], order), from_limbs(&l[..nl], order)))
}

// Many words: the trees in limbs

/// A level of a tree in limbs: the limbs of its nodes one after another in one buffer, each node
/// starting on an even limb of a buffer aligned to 16 bytes (the kernels load limbs in pairs), with
/// where each starts and how many limbs it has, none at the top zero.
struct Level {
    /// The limbs, in pairs.
    buf: Vec<u128>,
    /// The nodes' starts and lengths, in limbs.
    nodes: Vec<(usize, usize)>,
}

impl Level {
    /// A level of `count` nodes with room for `limbs` limbs and their padding.
    fn with_capacity(count: usize, limbs: usize) -> Level {
        Level {
            buf: Vec::with_capacity(limbs.div_ceil(2) + count),
            nodes: Vec::with_capacity(count),
        }
    }

    /// The limbs of node `i`.
    fn node(&self, i: usize) -> &[u64] {
        let (o, n) = self.nodes[i];
        // SAFETY: the buffer's pairs are plain limbs, and the node lies in it.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr().cast::<u64>().add(o), n) }
    }

    /// Appends a node of `n` limbs, zeroed, and returns them to be written; its length is then set
    /// by [`Level::trim_last`].
    fn push(&mut self, n: usize) -> &mut [u64] {
        let o = 2 * self.buf.len();
        self.buf.resize(self.buf.len() + n.div_ceil(2), 0);
        self.nodes.push((o, n));
        // SAFETY: as in `node`, the limbs just added.
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr().cast::<u64>().add(o), n) }
    }

    /// Drops the zero limbs at the top of the last node.
    fn trim_last(&mut self) {
        let i = self.nodes.len() - 1;
        let n = trimmed_len(self.node(i));
        self.nodes[i].1 = n;
    }
}

/// Working memory for the trees' products and remainders, grown as they need, in pairs of limbs so
/// that it starts on 16 bytes.
struct Work(Vec<u128>);

impl Work {
    /// At least `n` limbs.
    fn get(&mut self, n: usize) -> Scratch {
        if 2 * self.0.len() < n {
            self.0.resize(n.div_ceil(2), 0);
        }
        Scratch::new(self.0.as_mut_ptr().cast(), 2 * self.0.len())
    }
}

/// Sets `r` to the product of the limbs `x` and `y`, not empty, with the working memory `w`.
fn mul_work(r: &mut [u64], x: &[u64], y: &[u64], w: &mut Work) {
    let ((p, np), (q, nq)) = if x.len() >= y.len() {
        ((x.as_ptr(), x.len()), (y.as_ptr(), y.len()))
    } else {
        ((y.as_ptr(), y.len()), (x.as_ptr(), x.len()))
    };
    debug_assert!(r.len() >= np + nq);
    let s = w.get(mul_itch(np, nq));
    // SAFETY: the scratch holds the product's plan and starts on 16 bytes; `r` has room for the
    // product, apart from both operands.
    unsafe { mul(r.as_mut_ptr(), p, np, q, nq, s) };
}

/// Sets `r`, of `m`'s length, to the limbs `u` modulo the limbs `m`, the top one not zero, with the
/// working memory `w`.
fn rem_work(r: &mut [u64], u: &[u64], m: &[u64], w: &mut Work) {
    let s = w.get(rem_itch(u.len(), m.len()));
    // SAFETY: the scratch holds the remainder's plan and starts on 16 bytes; `r` has `m`'s length.
    unsafe {
        rem(
            r.as_mut_ptr(),
            u.as_ptr(),
            u.len(),
            0,
            m.as_ptr(),
            m.len(),
            s,
        )
    };
}

/// [`fast`] for congruences modulo words, `(r_i, m_i)` with `r_i < m_i`, every tree in limbs, with
/// no number but the two results: the product tree a buffer a level, the remainders of `P` by the
/// squares of the nodes a level at a time, the leaves in words, and the sums up the tree in limbs.
/// `None` when the moduli are not pairwise coprime.
pub(in super::super) fn fast_words(c: &[(u64, u64)]) -> Option<(Vec<u64>, Vec<u64>)> {
    let k = c.len();
    if k == 0 {
        return Some((Vec::new(), vec![1]));
    }
    let mut w = Work(Vec::new());
    // The product tree, the moduli at the bottom.
    let mut levels: Vec<Level> = Vec::new();
    let mut bottom = Level::with_capacity(k, k);
    for &(_, m) in c {
        bottom.push(1)[0] = m;
    }
    levels.push(bottom);
    while levels.last().is_some_and(|l| l.nodes.len() > 1) {
        let below = levels.last().unwrap_or_else(|| unreachable!());
        let n = below.nodes.len();
        let total: usize = below.nodes.iter().map(|&(_, len)| len).sum();
        let mut up = Level::with_capacity(n.div_ceil(2), total + n);
        for i in (0..n).step_by(2) {
            if i + 1 < n {
                let (a, b) = (below.node(i), below.node(i + 1));
                let (la, lb) = (a.len(), b.len());
                let r = up.push(la + lb);
                // The nodes are copied out of `below` first: `up` is another buffer.
                mul_work(r, a, b, &mut w);
                up.trim_last();
            } else {
                let a = below.node(i);
                up.push(a.len()).copy_from_slice(a);
            }
        }
        levels.push(up);
    }
    let top = levels.len() - 1;
    let p = levels[top].node(0).to_vec();
    // Down: the remainders of `P` by the squares of the nodes, a level at a time; at the top `P`.
    let mut rems = Level::with_capacity(1, p.len());
    rems.push(p.len()).copy_from_slice(&p);
    let mut square: Vec<u64> = Vec::new();
    for j in (0..top).rev() {
        let level = &levels[j];
        let n = level.nodes.len();
        let total: usize = level.nodes.iter().map(|&(_, len)| 2 * len).sum();
        let mut next = Level::with_capacity(n, total + n);
        for i in 0..n {
            let parent = rems.node(i / 2);
            let m = level.node(i);
            square.clear();
            square.resize(2 * m.len(), 0);
            mul_work(&mut square, m, m, &mut w);
            let ns = trimmed_len(&square);
            let r = next.push(ns);
            if below(parent, &square[..ns]) {
                // Below the square already (a node carried up alone has its parent's remainder).
                r[..parent.len()].copy_from_slice(parent);
            } else {
                rem_work(r, parent, &square[..ns], &mut w);
            }
            next.trim_last();
        }
        rems = next;
    }
    // The leaves: `c_i = (P mod m_i^2) / m_i`, and `t_i = r_i c_i^(-1) mod m_i`.
    let mut xs = Level::with_capacity(k, k);
    for (i, &(r, m)) in c.iter().enumerate() {
        let v = rems.node(i);
        let v = u128::from(v.first().copied().unwrap_or(0))
            | (u128::from(v.get(1).copied().unwrap_or(0)) << 64);
        if m == 1 {
            xs.push(0);
            continue;
        }
        let wm = Word::new(m);
        let ci = wm.quotient(v);
        let inv = word::inverse(ci, m)?;
        let t = wm.mul_mod(r, inv);
        xs.push(1)[0] = t;
        xs.trim_last();
    }
    // Up: `x = x_l M_r + x_r M_l`.
    let mut t1: Vec<u64> = Vec::new();
    let mut t2: Vec<u64> = Vec::new();
    for level in &levels[..top] {
        let n = level.nodes.len();
        let total: usize = level.nodes.iter().map(|&(_, len)| len).sum();
        let mut up = Level::with_capacity(n.div_ceil(2), total + n);
        for i in (0..n).step_by(2) {
            if i + 1 >= n {
                let x = xs.node(i);
                if x.is_empty() {
                    up.push(0);
                } else {
                    up.push(x.len()).copy_from_slice(x);
                }
                continue;
            }
            let (xl, xr) = (xs.node(i), xs.node(i + 1));
            let (ml, mr) = (level.node(i), level.node(i + 1));
            let len = (xl.len() + mr.len()).max(xr.len() + ml.len()) + 1;
            t1.clear();
            t1.resize(len, 0);
            t2.clear();
            t2.resize(len, 0);
            if !xl.is_empty() {
                mul_work(&mut t1, xl, mr, &mut w);
            }
            if !xr.is_empty() {
                mul_work(&mut t2, xr, ml, &mut w);
            }
            let r = up.push(len);
            let mut carry = false;
            for j in 0..len {
                let (s, c1) = t1[j].overflowing_add(t2[j]);
                let (s, c2) = s.overflowing_add(u64::from(carry));
                r[j] = s;
                carry = c1 | c2;
            }
            up.trim_last();
        }
        xs = up;
    }
    // `x mod P`.
    let x = xs.node(0);
    let mut out = vec![0u64; p.len()];
    if below(x, &p) {
        out[..x.len()].copy_from_slice(x);
    } else {
        rem_work(&mut out, x, &p, &mut w);
    }
    Some((out, p))
}

/// Whether the limbs `a` are below the limbs `b`, both without zeros at the top.
fn below(a: &[u64], b: &[u64]) -> bool {
    a.len() < b.len() || a.len() == b.len() && a.iter().rev().cmp(b.iter().rev()).is_lt()
}
