//! Products of polynomials modulo a word `n`: the schoolbook method with one reduction a
//! coefficient, Kronecker substitution into the crate's multiplication of integers, and the
//! number-theoretic transforms of the multiplication applied to the coefficients directly.
//!
//! The coefficients are residues below `n`, the constant first; a product of `la` and `lb`
//! coefficients has `la + lb - 1`, the top one possibly 0 for a composite `n`.

use super::nmod::WordMod;
use crate::math::numbers::nat;

/// The width of the sums of a product's coefficients before their reduction: a sum of `m`
/// products of two residues is below `m (n - 1)^2`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Acc {
    /// Residues below `2^32` whose sums stay below `2^64`, or are taken in pieces of `t` terms
    /// that do, by the vector kernels of 32-bit products.
    Half(usize),
    /// Below `2^128`.
    Two,
    /// Any, in three words.
    Three,
}

/// The width of the sums of `m` products of residues modulo `n`.
#[inline]
pub(crate) fn acc(m: usize, n: u64) -> Acc {
    let sq = u128::from(n - 1) * u128::from(n - 1);
    if n <= 1 << 32 {
        // The terms whose sum fits a word.
        let t = (u128::from(u64::MAX) / sq.max(1)).min(usize::MAX as u128) as usize;
        if t >= m || t >= 16 {
            return Acc::Half(t);
        }
    }
    match sq.checked_mul(m as u128) {
        Some(_) => Acc::Two,
        None => Acc::Three,
    }
}

/// `Σ a_i b_i` of 32-bit values in a word, for sums that fit it.
#[inline(always)]
fn dot_half(a: &[u32], b: &[u32]) -> u64 {
    let n = a.len().min(b.len());
    a[..n].iter().zip(&b[..n]).fold(0u64, |s, (&x, &y)| {
        s.wrapping_add(u64::from(x) * u64::from(y))
    })
}

/// `Σ a_i b_i` of 32-bit values, `t` terms at a time in a word.
#[inline(always)]
fn dot_halves(a: &[u32], b: &[u32], t: usize) -> u128 {
    let n = a.len().min(b.len());
    if n <= t {
        return u128::from(dot_half(a, b));
    }
    a[..n]
        .chunks(t)
        .zip(b[..n].chunks(t))
        .map(|(x, y)| u128::from(dot_half(x, y)))
        .sum()
}

/// `Σ a_i b_i` modulo `2^128`, two sums of the products unrolled by four.
#[inline(always)]
fn dot_two(a: &[u64], b: &[u64]) -> u128 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (mut s0, mut s1) = (0u128, 0u128);
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (x, y) in ac.iter().zip(bc) {
        s0 = s0.wrapping_add(u128::from(x[0]) * u128::from(y[0]));
        s1 = s1.wrapping_add(u128::from(x[1]) * u128::from(y[1]));
        s0 = s0.wrapping_add(u128::from(x[2]) * u128::from(y[2]));
        s1 = s1.wrapping_add(u128::from(x[3]) * u128::from(y[3]));
    }
    for (&x, &y) in ar.iter().zip(br) {
        s0 = s0.wrapping_add(u128::from(x) * u128::from(y));
    }
    s0.wrapping_add(s1)
}

/// `s += p` in three words `(c, s)`.
#[inline(always)]
pub(crate) fn add3(s: &mut u128, c: &mut u64, p: u128) {
    let (t, o) = s.overflowing_add(p);
    *s = t;
    *c += u64::from(o);
}

/// `Σ a_i b_i` in three words, two sums unrolled by four.
#[inline(always)]
fn dot_three(a: &[u64], b: &[u64]) -> (u64, u128) {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (mut s0, mut c0, mut s1, mut c1) = (0u128, 0u64, 0u128, 0u64);
    let (ac, ar) = a.as_chunks::<4>();
    let (bc, br) = b.as_chunks::<4>();
    for (x, y) in ac.iter().zip(bc) {
        add3(&mut s0, &mut c0, u128::from(x[0]) * u128::from(y[0]));
        add3(&mut s1, &mut c1, u128::from(x[1]) * u128::from(y[1]));
        add3(&mut s0, &mut c0, u128::from(x[2]) * u128::from(y[2]));
        add3(&mut s1, &mut c1, u128::from(x[3]) * u128::from(y[3]));
    }
    for (&x, &y) in ar.iter().zip(br) {
        add3(&mut s0, &mut c0, u128::from(x) * u128::from(y));
    }
    add3(&mut s0, &mut c0, s1);
    (c0 + c1, s0)
}

/// Up to this many coefficients, the reversed operand of the schoolbook method lies on the
/// stack.
const STACK: usize = 64;

/// Calls `f` with the coefficients `b` reversed, in a buffer on the stack when they are few.
#[inline(always)]
fn reversed<T: Copy + Default, R>(
    b: &[u64],
    conv: impl Fn(u64) -> T,
    f: impl FnOnce(&[T]) -> R,
) -> R {
    if b.len() <= STACK {
        let mut buf = [T::default(); STACK];
        for (o, &x) in buf.iter_mut().zip(b.iter().rev()) {
            *o = conv(x);
        }
        f(&buf[..b.len()])
    } else {
        let v: Vec<T> = b.iter().rev().map(|&x| conv(x)).collect();
        f(&v)
    }
}

/// Calls `f` with the coefficients `a` as 32-bit values, on the stack when they are few.
#[inline(always)]
fn halves<R>(a: &[u64], f: impl FnOnce(&[u32]) -> R) -> R {
    if a.len() <= STACK {
        let mut buf = [0u32; STACK];
        for (o, &x) in buf.iter_mut().zip(a) {
            *o = x as u32;
        }
        f(&buf[..a.len()])
    } else {
        let v: Vec<u32> = a.iter().map(|&x| x as u32).collect();
        f(&v)
    }
}

/// The schoolbook sums of products of 32-bit residues by NEON's widening multiply-adds, 16
/// coefficients of the product at a time: for each coefficient `a_i` the 16 coefficients of `b`
/// that land on them, from `b` padded with zeros, `umlal` by `a_i` into eight vectors of two
/// 64-bit sums. On an Apple M4 it is 2 to 5 times as fast as the sums one coefficient at a time
/// up to 64 coefficients, and loses past about 200, where the dot products are used.
#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon {
    use core::arch::aarch64::{
        uint64x2_t, vdupq_n_u64, vget_low_u32, vld1q_u32, vmlal_high_n_u32, vmlal_n_u32, vst1q_u64,
    };

    /// The zeros on each side of `b`.
    pub(super) const PAD: usize = 16;

    /// The longest operands taken.
    pub(super) const MAX: usize = 128;

    /// Sets `r` (`la + lb - 1` words) to the sums `Σ a_i b_(k-i)` of the 32-bit values, for `bp`
    /// holding `b` at `PAD` with `PAD` zeros on each side, `lb <= MAX`, and sums that fit a word.
    pub(super) fn slide(r: &mut [u64], a: &[u32], bp: &[u32], lb: usize) {
        let la = a.len();
        let len = la + lb - 1;
        assert!(r.len() == len && bp.len() >= lb + 2 * PAD && la >= 1 && lb >= 1);
        let mut k0 = 0;
        while k0 < len {
            // SAFETY: NEON is part of every aarch64 target; the loads read bp[k0 + PAD - i ..
            // k0 + PAD - i + 16] for i in ilo..ihi, where k0 + PAD - i >= k0 + PAD - (k0 + 15) =
            // PAD - 15 >= 1 and k0 + PAD - i + 16 <= k0 + PAD - (k0 + 1 - lb) + 16 = lb + PAD + 15
            // < lb + 2 PAD, inside bp; the stores write the 16 words of `out`.
            unsafe {
                let mut acc: [uint64x2_t; 8] = [vdupq_n_u64(0); 8];
                let ilo = (k0 + 1).saturating_sub(lb);
                let ihi = (k0 + 16).min(la);
                for i in ilo..ihi {
                    let x = *a.get_unchecked(i);
                    let p = bp.as_ptr().add(k0 + PAD - i);
                    let w0 = vld1q_u32(p);
                    let w1 = vld1q_u32(p.add(4));
                    let w2 = vld1q_u32(p.add(8));
                    let w3 = vld1q_u32(p.add(12));
                    acc[0] = vmlal_n_u32(acc[0], vget_low_u32(w0), x);
                    acc[1] = vmlal_high_n_u32(acc[1], w0, x);
                    acc[2] = vmlal_n_u32(acc[2], vget_low_u32(w1), x);
                    acc[3] = vmlal_high_n_u32(acc[3], w1, x);
                    acc[4] = vmlal_n_u32(acc[4], vget_low_u32(w2), x);
                    acc[5] = vmlal_high_n_u32(acc[5], w2, x);
                    acc[6] = vmlal_n_u32(acc[6], vget_low_u32(w3), x);
                    acc[7] = vmlal_high_n_u32(acc[7], w3, x);
                }
                let mut out = [0u64; 16];
                for (t, v) in acc.iter().enumerate() {
                    vst1q_u64(out.as_mut_ptr().add(2 * t), *v);
                }
                let m = 16.min(len - k0);
                r[k0..k0 + m].copy_from_slice(&out[..m]);
            }
            k0 += 16;
        }
    }
}

/// [`school`] by the NEON kernel when it applies: sums of 32-bit residues that fit a word, and
/// operands of at most 128 coefficients. Returns whether it did.
#[allow(unused_variables)]
fn school_neon(r: &mut [u64], a: &[u64], b: &[u64], m: &WordMod) -> bool {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        let (la, lb) = (a.len(), b.len());
        if la > neon::MAX || lb > neon::MAX {
            return false;
        }
        let Acc::Half(t) = acc(la.min(lb), m.n) else {
            return false;
        };
        if t < la.min(lb) {
            return false;
        }
        let mut av = [0u32; neon::MAX];
        for (o, &x) in av.iter_mut().zip(a) {
            *o = x as u32;
        }
        let mut bp = [0u32; neon::MAX + 2 * neon::PAD];
        for (o, &x) in bp[neon::PAD..].iter_mut().zip(b) {
            *o = x as u32;
        }
        neon::slide(r, &av[..la], &bp[..lb + 2 * neon::PAD], lb);
        for x in r.iter_mut() {
            *x = m.reduce1(*x);
        }
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// The schoolbook product of `a` and `b` into `r`, of `la + lb - 1` coefficients, each a sum of
/// products reduced once: coefficient `k` the dot product of `a[lo..hi]` and `b` reversed.
pub(crate) fn school(r: &mut [u64], a: &[u64], b: &[u64], m: &WordMod) {
    let (la, lb) = (a.len(), b.len());
    debug_assert!(la >= 1 && lb >= 1 && r.len() == la + lb - 1);
    if la.min(lb) <= TINY && la.max(lb) <= 2 * TINY {
        tiny(r, a, b, m);
        return;
    }
    if school_neon(r, a, b, m) {
        return;
    }
    // b[k - i] = rb[lb - 1 - k + i].
    match acc(la.min(lb), m.n) {
        Acc::Half(t) => halves(a, |a| {
            reversed(
                b,
                |x| x as u32,
                |rb| {
                    for (k, out) in r.iter_mut().enumerate() {
                        let (lo, hi) = terms(k, la, lb);
                        let o = lb - 1 + lo - k;
                        let s = dot_halves(&a[lo..hi], &rb[o..o + hi - lo], t);
                        *out = m.reduce_u128(s);
                    }
                },
            )
        }),
        Acc::Two => reversed(
            b,
            |x| x,
            |rb| {
                for (k, out) in r.iter_mut().enumerate() {
                    let (lo, hi) = terms(k, la, lb);
                    let o = lb - 1 + lo - k;
                    *out = m.reduce_u128(dot_two(&a[lo..hi], &rb[o..o + hi - lo]));
                }
            },
        ),
        Acc::Three => reversed(
            b,
            |x| x,
            |rb| {
                for (k, out) in r.iter_mut().enumerate() {
                    let (lo, hi) = terms(k, la, lb);
                    let o = lb - 1 + lo - k;
                    let (c, s) = dot_three(&a[lo..hi], &rb[o..o + hi - lo]);
                    *out = m.reduce3(c, (s >> 64) as u64, s as u64);
                }
            },
        ),
    }
}

/// Up to this many coefficients in the shorter operand (and twice as many in the longer), a
/// product by [`tiny`].
const TINY: usize = 4;

/// The schoolbook product of short operands into `r`, the sums of products on the stack in the
/// width they need (one word, two, or three), one reduction a coefficient: none of the set-up of
/// the vector kernels, which costs more than the products here.
fn tiny(r: &mut [u64], a: &[u64], b: &[u64], m: &WordMod) {
    let short = a.len().min(b.len());
    match acc(short, m.n) {
        Acc::Half(t) if t >= short => {
            let mut s = [0u64; 3 * TINY];
            for (i, &x) in a.iter().enumerate() {
                for (o, &y) in s[i..].iter_mut().zip(b) {
                    *o = o.wrapping_add(x * y);
                }
            }
            for (o, &v) in r.iter_mut().zip(&s) {
                *o = m.reduce1(v);
            }
        }
        Acc::Half(_) | Acc::Two => {
            let mut s = [0u128; 3 * TINY];
            for (i, &x) in a.iter().enumerate() {
                for (o, &y) in s[i..].iter_mut().zip(b) {
                    *o = o.wrapping_add(u128::from(x) * u128::from(y));
                }
            }
            for (o, &v) in r.iter_mut().zip(&s) {
                *o = m.reduce_u128(v);
            }
        }
        Acc::Three => {
            let mut s = [0u128; 3 * TINY];
            let mut c = [0u64; 3 * TINY];
            for (i, &x) in a.iter().enumerate() {
                for (j, &y) in b.iter().enumerate() {
                    add3(&mut s[i + j], &mut c[i + j], u128::from(x) * u128::from(y));
                }
            }
            for (j, o) in r.iter_mut().enumerate() {
                *o = m.reduce3(c[j], (s[j] >> 64) as u64, s[j] as u64);
            }
        }
    }
}

/// The ranges of `a` and `b`, the latter reversed, whose products land on coefficient `k`.
#[inline(always)]
fn terms(k: usize, la: usize, lb: usize) -> (usize, usize) {
    let lo = k.saturating_sub(lb - 1);
    let hi = k.min(la - 1);
    (lo, hi + 1)
}

/// The schoolbook square of `a` into `r`, of `2 la - 1` coefficients: the products `a_i a_j`
/// for `i < j` summed and doubled, the squares added, one reduction a coefficient.
pub(crate) fn school_sqr(r: &mut [u64], a: &[u64], m: &WordMod) {
    let la = a.len();
    debug_assert!(la >= 1 && r.len() == 2 * la - 1);
    // The pairs i < k - i: i from lo to top = min(hi, ceil(k / 2)), a[k - i] = ra[la - 1 - k + i].
    let pairs = |k: usize| {
        let (lo, hi) = terms(k, la, la);
        (lo, hi.min(k.div_ceil(2)), la - 1 + lo - k)
    };
    if la <= TINY {
        tiny(r, a, a, m);
        return;
    }
    if school_neon(r, a, a, m) {
        return;
    }
    let finish = |k: usize, mut c: u64, mut s: u128| {
        // Doubled, then the square of the middle coefficient added.
        c = (c << 1) | (s >> 127) as u64;
        s <<= 1;
        if k.is_multiple_of(2) {
            let x = a[k / 2];
            add3(&mut s, &mut c, u128::from(x) * u128::from(x));
        }
        m.reduce3(c, (s >> 64) as u64, s as u64)
    };
    match acc(la, m.n) {
        Acc::Half(t) => halves(a, |h| {
            reversed(
                a,
                |x| x as u32,
                |ra| {
                    for (k, out) in r.iter_mut().enumerate() {
                        let (lo, top, o) = pairs(k);
                        let s = if lo < top {
                            dot_halves(&h[lo..top], &ra[o..o + top - lo], t)
                        } else {
                            0
                        };
                        *out = finish(k, 0, s);
                    }
                },
            )
        }),
        Acc::Two => reversed(
            a,
            |x| x,
            |ra| {
                for (k, out) in r.iter_mut().enumerate() {
                    let (lo, top, o) = pairs(k);
                    let s = if lo < top {
                        dot_two(&a[lo..top], &ra[o..o + top - lo])
                    } else {
                        0
                    };
                    *out = finish(k, 0, s);
                }
            },
        ),
        Acc::Three => reversed(
            a,
            |x| x,
            |ra| {
                for (k, out) in r.iter_mut().enumerate() {
                    let (lo, top, o) = pairs(k);
                    let (c, s) = if lo < top {
                        dot_three(&a[lo..top], &ra[o..o + top - lo])
                    } else {
                        (0, 0)
                    };
                    *out = finish(k, c, s);
                }
            },
        ),
    }
}

// Kronecker substitution

/// The bits of `x`, 0 for 0.
#[inline(always)]
fn bits(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// The bits of a field of Kronecker substitution for products of `m` terms modulo `n`: every sum
/// of `m` products of residues is below `2^f`.
#[inline]
pub(crate) fn ks_field(m: usize, n: u64) -> usize {
    (2 * bits(n - 1) + bits(m as u64)) as usize
}

/// Lays the coefficients `a`, each below `2^f`, `f` bits apart into `out`, zeroed and long
/// enough.
fn pack(out: &mut [u64], a: &[u64], f: usize) {
    let mut p = 0usize;
    for &c in a {
        let (w, sh) = (p / 64, p % 64);
        out[w] |= c << sh;
        if sh != 0 && w + 1 < out.len() {
            out[w + 1] |= c >> (64 - sh);
        }
        p += f;
    }
}

/// The `f <= 192` bits of `x` from bit `p`, as three words, the least significant first.
#[inline(always)]
fn field(x: &[u64], p: usize, f: usize) -> (u64, u64, u64) {
    let (w, sh) = (p / 64, (p % 64) as u32);
    let get = |i: usize| x.get(w + i).copied().unwrap_or(0);
    let (l0, l1, l2, l3) = (get(0), get(1), get(2), get(3));
    let join = |lo: u64, hi: u64| {
        if sh == 0 {
            lo
        } else {
            (lo >> sh) | (hi << (64 - sh))
        }
    };
    let (mut w0, mut w1, mut w2) = (join(l0, l1), join(l1, l2), join(l2, l3));
    match f {
        0..64 => {
            w0 &= (1u64 << f) - 1;
            w1 = 0;
            w2 = 0;
        }
        64..128 => {
            if f > 64 {
                w1 &= (1u64 << (f - 64)) - 1;
            } else {
                w1 = 0;
            }
            w2 = 0;
        }
        _ => {
            if f < 192 {
                if f > 128 {
                    w2 &= (1u64 << (f - 128)) - 1;
                } else {
                    w2 = 0;
                }
            }
        }
    }
    (w0, w1, w2)
}

/// The product of `a` and `b` (the square of `a` when `b` is empty) into `r` by Kronecker
/// substitution: each polynomial evaluated at `2^f` for the field `f` of [`ks_field`], one
/// product of integers, and the coefficients read back from their fields and reduced.
pub(crate) fn ks(r: &mut [u64], a: &[u64], b: &[u64], m: &WordMod, scratch: &mut Vec<u64>) {
    let (la, lb, square) = (a.len(), b.len(), b.is_empty());
    let terms = if square { la } else { la.min(lb) };
    let f = ks_field(terms, m.n);
    let xa_n = (la * f).div_ceil(64);
    let mut xa = vec![0u64; xa_n];
    pack(&mut xa, a, f);
    let mut p = Vec::new();
    if square {
        nat::sqr_into(&mut p, &xa, scratch);
    } else {
        let mut xb = vec![0u64; (lb * f).div_ceil(64)];
        pack(&mut xb, b, f);
        nat::mul_into(&mut p, &xa, &xb, scratch);
    }
    let mut pos = 0usize;
    if f <= 64 {
        for out in r.iter_mut() {
            let (w0, _, _) = field(&p, pos, f);
            *out = m.reduce1(w0);
            pos += f;
        }
    } else if f <= 128 {
        for out in r.iter_mut() {
            let (w0, w1, _) = field(&p, pos, f);
            *out = m.reduce3(0, w1, w0);
            pos += f;
        }
    } else {
        for out in r.iter_mut() {
            let (w0, w1, w2) = field(&p, pos, f);
            *out = m.reduce3(w2, w1, w0);
            pos += f;
        }
    }
}

// The transforms

/// The product of `a` and `b` (the square of `a` when `b` is empty) into `r` by the
/// number-theoretic transforms, cutting the longer operand when the product is longer than they
/// take.
pub(crate) fn ntt(r: &mut [u64], a: &[u64], b: &[u64], m: &WordMod) {
    let (la, lb, square) = (a.len(), b.len(), b.is_empty());
    let bits = m.bits();
    if let Some(p) = nat::word_poly_plan(la, lb, bits) {
        nat::word_poly_mul(r, a, b, m.n, p);
        return;
    }
    // Too long: pieces of the longer operand times the shorter, added at their places.
    let (x, y) = if square || la >= lb {
        (a, if square { a } else { b })
    } else {
        (b, a)
    };
    let ly = y.len();
    if ly > nat::WORD_POLY_MAX / 8 {
        // Both long: pieces would take a number of transforms growing as the product of the
        // lengths; Kronecker substitution into the integers' multiplication (Toom's methods on
        // top of its transforms) grows as their sum.
        ks(r, a, b, m, &mut Vec::new());
        return;
    }
    let piece = nat::WORD_POLY_MAX + 1 - ly;
    r.fill(0);
    let mut t = vec![0u64; piece + ly - 1];
    for (i, chunk) in x.chunks(piece).enumerate() {
        let t = &mut t[..chunk.len() + ly - 1];
        let p = nat::word_poly_plan(chunk.len(), ly, bits).expect("a piece the transforms take");
        nat::word_poly_mul(t, chunk, y, m.n, p);
        for (o, &v) in r[i * piece..].iter_mut().zip(t.iter()) {
            *o = m.add(*o, v);
        }
    }
}

// The choice

/// The method of a product, by the operands' lengths and the modulus.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Method {
    School,
    Kronecker,
    Transforms,
}

/// Whether the transforms run on NEON lanes here; elsewhere, four times slower, they start
/// later and Kronecker substitution takes the lengths between.
pub(crate) const NEON: bool = cfg!(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    not(miri)
));

/// Up to this many coefficients in the longer operand, products of residues below `2^32` whose
/// sums fit a word take the vector kernel of the schoolbook method (NEON), which beats the
/// transforms there on an Apple M4.
const NEON_SCHOOL: usize = 128;

/// The method of a product of `la` by `lb` coefficients (a square when `lb` is 0) modulo `m`.
pub(crate) fn method(la: usize, lb: usize, m: &WordMod) -> Method {
    let short = if lb == 0 { la } else { la.min(lb) };
    let long = la.max(lb);
    if short <= school_max(m.bits(), lb == 0) {
        return Method::School;
    }
    if NEON && long <= NEON_SCHOOL && matches!(acc(short, m.n), Acc::Half(t) if t >= short) {
        // The vector kernel's sums of 32-bit products in a word.
        return Method::School;
    }
    if NEON || long + short >= if cfg!(miri) { 40 } else { 8192 } {
        return Method::Transforms;
    }
    Method::Kronecker
}

/// Up to this many coefficients in the shorter operand, the schoolbook method: its sums of
/// products cost about 0.06 ns a product below `2^32` and 0.25 ns above on an Apple M4, which
/// the transforms beat from about these lengths.
fn school_max(bits: u32, square: bool) -> usize {
    if cfg!(miri) {
        return 6;
    }
    match (bits, square) {
        (0..=32, false) => 64,
        (0..=32, true) => 96,
        (33..=56, false) => 56,
        (33..=56, true) => 96,
        (_, false) => 64,
        (_, true) => 96,
    }
}

/// The product of `a` and `b`, both not empty: `la + lb - 1` coefficients.
pub(crate) fn mul(a: &[u64], b: &[u64], m: &WordMod) -> Vec<u64> {
    let mut r = vec![0u64; a.len() + b.len() - 1];
    mul_into(&mut r, a, b, m);
    r
}

/// The square of `a`, not empty: `2 la - 1` coefficients.
pub(crate) fn sqr(a: &[u64], m: &WordMod) -> Vec<u64> {
    let mut r = vec![0u64; 2 * a.len() - 1];
    sqr_into(&mut r, a, m);
    r
}

/// [`mul`] into `r`, of `la + lb - 1` coefficients.
pub(crate) fn mul_into(r: &mut [u64], a: &[u64], b: &[u64], m: &WordMod) {
    match method(a.len(), b.len(), m) {
        Method::School => school(r, a, b, m),
        Method::Kronecker => ks(r, a, b, m, &mut Vec::new()),
        Method::Transforms => ntt(r, a, b, m),
    }
}

/// [`sqr`] into `r`, of `2 la - 1` coefficients.
pub(crate) fn sqr_into(r: &mut [u64], a: &[u64], m: &WordMod) {
    match method(a.len(), 0, m) {
        Method::School => school_sqr(r, a, m),
        Method::Kronecker => ks(r, a, &[], m, &mut Vec::new()),
        Method::Transforms => ntt(r, a, &[], m),
    }
}

/// The product of `a` and `b` modulo `x^l - 1` for `l = 2^k` at least both lengths, by the
/// transforms when they are the method for operands this long; `None` otherwise.
pub(crate) fn mul_cyclic(a: &[u64], b: &[u64], k: u32, m: &WordMod) -> Option<Vec<u64>> {
    let (la, lb) = (a.len(), b.len());
    if method(la, lb, m) != Method::Transforms || !NEON && !cfg!(miri) {
        return None;
    }
    let p = nat::word_poly_plan_cyclic(la.min(lb), k, m.bits())?;
    let mut r = vec![0u64; 1 << k];
    nat::word_poly_mul_cyclic(&mut r, a, b, m.n, p);
    Some(r)
}

// Division and evaluation

/// `Σ a_i b_i mod n` for residues, the sum taken in the width its length needs and reduced once.
pub(crate) fn dot_mod(a: &[u64], b: &[u64], m: &WordMod) -> u64 {
    let len = a.len().min(b.len());
    match acc(len, m.n) {
        Acc::Half(t) if t >= len => {
            let s = a[..len]
                .iter()
                .zip(&b[..len])
                .fold(0u64, |s, (&x, &y)| s.wrapping_add(x * y));
            m.reduce1(s)
        }
        Acc::Half(_) | Acc::Two => m.reduce_u128(dot_two(a, b)),
        Acc::Three => {
            let (c, s) = dot_three(a, b);
            m.reduce3(c, (s >> 64) as u64, s as u64)
        }
    }
}

/// The quotient and the remainder of `a` by `d` (residues, `d` not empty) with `inv` the inverse
/// of the leading coefficient of `d`, by the schoolbook method with delayed reduction: the
/// remainder's coefficients are kept as sums, `a_j + Σ (n - t_i) d_(j-i)` for the quotient's
/// coefficients `t_i` (in a word for small moduli and few steps, in two or three otherwise),
/// and only the coefficient that gives the next quotient coefficient is reduced at each step,
/// the others once at the end. The remainder is `ld - 1` coefficients, not trimmed; the quotient
/// `la - ld + 1`, or none when `want_q` is false.
pub(crate) fn div_rem_school(a: &[u64], d: &[u64], inv: u64, m: &WordMod) -> (Vec<u64>, Vec<u64>) {
    let mut q = vec![0u64; (a.len() + 1).saturating_sub(d.len())];
    let r = rem_school(a, d, inv, m, Some(&mut q));
    (q, r)
}

/// The remainder of [`div_rem_school`], the quotient's coefficients into `q` when given.
pub(crate) fn rem_school(
    a: &[u64],
    d: &[u64],
    inv: u64,
    m: &WordMod,
    q: Option<&mut [u64]>,
) -> Vec<u64> {
    let (la, ld) = (a.len(), d.len());
    if la < ld {
        let mut r = a.to_vec();
        r.resize(ld - 1, 0);
        return r;
    }
    let lq = la - ld + 1;
    // Each sum takes at most min(lq, ld - 1) products and its first coefficient.
    let steps = lq.min(ld - 1) + 1;
    match acc(steps, m.n) {
        Acc::Half(t) if t >= steps => rem_half(a, d, inv, m, q),
        Acc::Half(_) | Acc::Two => rem_two(a, d, inv, m, q),
        Acc::Three => rem_three(a, d, inv, m, q),
    }
}

/// [`rem_school`] with the sums in a word, the products of 32-bit residues by the vector
/// kernels.
fn rem_half(a: &[u64], d: &[u64], inv: u64, m: &WordMod, mut q: Option<&mut [u64]>) -> Vec<u64> {
    let (la, ld) = (a.len(), d.len());
    let mut w: Vec<u64> = a.to_vec();
    let dl: Vec<u32> = d[..ld - 1].iter().map(|&x| x as u32).collect();
    for i in (0..=la - ld).rev() {
        let t = m.mul(m.reduce1(w[i + ld - 1]), inv);
        if let Some(q) = q.as_deref_mut() {
            q[i] = t;
        }
        if t == 0 {
            continue;
        }
        let nt = u64::from((m.n - t) as u32);
        for (x, &y) in w[i..i + ld - 1].iter_mut().zip(&dl) {
            *x = x.wrapping_add(nt * u64::from(y));
        }
    }
    w.truncate(ld - 1);
    for x in w.iter_mut() {
        *x = m.reduce1(*x);
    }
    w
}

/// [`rem_school`] with the sums in two words.
fn rem_two(a: &[u64], d: &[u64], inv: u64, m: &WordMod, mut q: Option<&mut [u64]>) -> Vec<u64> {
    let (la, ld) = (a.len(), d.len());
    let mut w: Vec<u128> = a.iter().map(|&x| u128::from(x)).collect();
    let dl = &d[..ld - 1];
    for i in (0..=la - ld).rev() {
        let t = m.mul(m.reduce_u128(w[i + ld - 1]), inv);
        if let Some(q) = q.as_deref_mut() {
            q[i] = t;
        }
        if t == 0 {
            continue;
        }
        let nt = m.n - t;
        for (x, &y) in w[i..i + ld - 1].iter_mut().zip(dl) {
            *x = x.wrapping_add(u128::from(nt) * u128::from(y));
        }
    }
    w.truncate(ld - 1);
    w.iter().map(|&x| m.reduce_u128(x)).collect()
}

/// [`rem_school`] with the sums in three words.
fn rem_three(a: &[u64], d: &[u64], inv: u64, m: &WordMod, mut q: Option<&mut [u64]>) -> Vec<u64> {
    let (la, ld) = (a.len(), d.len());
    let mut w: Vec<u128> = a.iter().map(|&x| u128::from(x)).collect();
    let mut c: Vec<u64> = vec![0; la];
    let dl = &d[..ld - 1];
    for i in (0..=la - ld).rev() {
        let top = i + ld - 1;
        let v = m.reduce3(c[top], (w[top] >> 64) as u64, w[top] as u64);
        let t = m.mul(v, inv);
        if let Some(q) = q.as_deref_mut() {
            q[i] = t;
        }
        if t == 0 {
            continue;
        }
        let nt = m.n - t;
        for ((x, k), &y) in w[i..top].iter_mut().zip(&mut c[i..top]).zip(dl) {
            add3(x, k, u128::from(nt) * u128::from(y));
        }
    }
    (0..ld - 1)
        .map(|j| m.reduce3(c[j], (w[j] >> 64) as u64, w[j] as u64))
        .collect()
}

/// [`rem_school`] in place, without allocating: `a` left with its `ld - 1` low coefficients,
/// `d32` the low `ld - 1` coefficients of `d` as 32-bit values (or empty when `n > 2^32`), and
/// the sums of two and three words in `w` and `c`.
pub(crate) fn rem_school_in(
    a: &mut Vec<u64>,
    d: &[u64],
    d32: &[u32],
    inv: u64,
    m: &WordMod,
    w: &mut Vec<u128>,
    c: &mut Vec<u64>,
) {
    let (la, ld) = (a.len(), d.len());
    if la < ld {
        a.resize(ld - 1, 0);
        return;
    }
    let lq = la - ld + 1;
    let steps = lq.min(ld - 1) + 1;
    match acc(steps, m.n) {
        Acc::Half(t) if t >= steps && d32.len() == ld - 1 => {
            for i in (0..lq).rev() {
                let t = m.mul(m.reduce1(a[i + ld - 1]), inv);
                if t == 0 {
                    continue;
                }
                let nt = u64::from((m.n - t) as u32);
                for (x, &y) in a[i..i + ld - 1].iter_mut().zip(d32) {
                    *x = x.wrapping_add(nt * u64::from(y));
                }
            }
            a.truncate(ld - 1);
            for x in a.iter_mut() {
                *x = m.reduce1(*x);
            }
        }
        Acc::Half(_) | Acc::Two => {
            w.clear();
            w.extend(a.iter().map(|&x| u128::from(x)));
            let dl = &d[..ld - 1];
            for i in (0..lq).rev() {
                let t = m.mul(m.reduce_u128(w[i + ld - 1]), inv);
                if t == 0 {
                    continue;
                }
                let nt = m.n - t;
                for (x, &y) in w[i..i + ld - 1].iter_mut().zip(dl) {
                    *x = x.wrapping_add(u128::from(nt) * u128::from(y));
                }
            }
            a.truncate(ld - 1);
            for (o, &x) in a.iter_mut().zip(w.iter()) {
                *o = m.reduce_u128(x);
            }
        }
        Acc::Three => {
            w.clear();
            w.extend(a.iter().map(|&x| u128::from(x)));
            c.clear();
            c.resize(la, 0);
            let dl = &d[..ld - 1];
            for i in (0..lq).rev() {
                let top = i + ld - 1;
                let v = m.reduce3(c[top], (w[top] >> 64) as u64, w[top] as u64);
                let t = m.mul(v, inv);
                if t == 0 {
                    continue;
                }
                let nt = m.n - t;
                for ((x, k), &y) in w[i..top].iter_mut().zip(&mut c[i..top]).zip(dl) {
                    add3(x, k, u128::from(nt) * u128::from(y));
                }
            }
            a.truncate(ld - 1);
            for (j, o) in a.iter_mut().enumerate() {
                *o = m.reduce3(c[j], (w[j] >> 64) as u64, w[j] as u64);
            }
        }
    }
}

/// The value of the polynomial `a` at the residue `x`, by Horner's rule with Shoup's constant of
/// `x`, in two chains (the even and the odd coefficients, in `x^2`) that the core overlaps.
pub(crate) fn eval(a: &[u64], x: u64, m: &WordMod) -> u64 {
    if a.len() < 8 || m.n >> 63 != 0 {
        return a.iter().rev().fold(0, |v, &c| m.add(m.mul(v, x), c));
    }
    let xp = m.shoup(x);
    let x2 = m.mul(x, x);
    let x2p = m.shoup(x2);
    // a(x) = e(x^2) + x o(x^2).
    let (mut e, mut o) = (0u64, 0u64);
    let mut i = a.len();
    if i % 2 == 1 {
        i -= 1;
        e = a[i];
    }
    while i >= 2 {
        i -= 2;
        e = m.add(m.mul_shoup(e, x2, x2p), a[i]);
        o = m.add(m.mul_shoup(o, x2, x2p), a[i + 1]);
    }
    m.add(e, m.mul_shoup(o, x, xp))
}

/// The composition `p(g)` modulo a word, on vectors of residues (`p`, `g` without zeros at the
/// top, `g` of at least two coefficients): Horner's rule for up to 8 coefficients of `p`, and
/// divide and conquer beyond, `p = lo + x^h hi` giving `lo(g) + g^h hi(g)` with the powers
/// `g^(2^i)` squared once. Not trimmed.
pub(crate) fn compose(p: &[u64], g: &[u64], m: &WordMod) -> Vec<u64> {
    let mut pows = vec![g.to_vec()];
    while (1usize << pows.len()) < p.len() {
        let next = sqr(pows.last().expect("not empty"), m);
        pows.push(next);
    }
    compose_part(p, &pows, m)
}

/// [`compose`] of the coefficients `p` with `pows[i] = g^(2^i)`.
fn compose_part(p: &[u64], pows: &[Vec<u64>], m: &WordMod) -> Vec<u64> {
    let len = p.len();
    if len <= 8 {
        let g = &pows[0];
        let mut v = vec![p[len - 1]];
        for &c in p[..len - 1].iter().rev() {
            v = mul(&v, g, m);
            v[0] = m.add(v[0], c);
        }
        return v;
    }
    let e = (usize::BITS - 1 - (len - 1).leading_zeros()) as usize;
    let h = 1usize << e;
    let mut a = compose_part(&p[..h], pows, m);
    let b = compose_part(&p[h..], pows, m);
    let t = mul(&b, &pows[e], m);
    if a.len() < t.len() {
        a.resize(t.len(), 0);
    }
    for (x, &y) in a.iter_mut().zip(&t) {
        *x = m.add(*x, y);
    }
    a
}
