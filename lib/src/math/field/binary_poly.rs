//! Polynomials over `GF(2)` held as words, the coefficient of `xⁱ` bit `i mod 64` of word
//! `i / 64`, for the binary fields: degrees, shifted sums, division, the greatest common divisor,
//! and the inverse modulo a polynomial by the extended Euclidean algorithm.
//!
//! Each step of Euclid's algorithm over `GF(2)[x]` cancels the leading term of one polynomial by
//! the other shifted under it, `u ← u + xʲ·v` with `j = deg u − deg v`, a pass over the words: so a
//! division, a gcd or an inverse of polynomials of degree `m` takes `O(m)` such passes, `O(m²/64)`
//! word operations (Hankerson, Menezes and Vanstone, "Guide to Elliptic Curve Cryptography", 2004,
//! Algorithm 2.48).

/// The degree of `a`, or `None` for 0.
#[must_use]
pub(super) fn degree(a: &[u64]) -> Option<usize> {
    a.iter()
        .rposition(|&w| w != 0)
        .map(|i| 64 * i + 63 - a[i].leading_zeros() as usize)
}

/// The degree of `a`, which is nonzero, looking down from bit `from`.
fn degree_below(a: &[u64], from: usize) -> Option<usize> {
    let mut i = from / 64;
    let w = a[i] & (u64::MAX >> (63 - from % 64));
    if w != 0 {
        return Some(64 * i + 63 - w.leading_zeros() as usize);
    }
    while i > 0 {
        i -= 1;
        if a[i] != 0 {
            return Some(64 * i + 63 - a[i].leading_zeros() as usize);
        }
    }
    None
}

/// `dst += src·x^shift`; the bits of the sum beyond `dst` must be zero (only checked in debug).
pub(super) fn add_shifted(dst: &mut [u64], src: &[u64], shift: usize) {
    let (q, r) = (shift / 64, shift % 64);
    if r == 0 {
        for (i, &w) in src.iter().enumerate() {
            if q + i < dst.len() {
                dst[q + i] ^= w;
            } else {
                debug_assert!(w == 0, "a shifted sum overflows");
            }
        }
    } else {
        for (i, &w) in src.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let (lo, hi) = (w << r, w >> (64 - r));
            if q + i < dst.len() {
                dst[q + i] ^= lo;
            } else {
                debug_assert!(lo == 0, "a shifted sum overflows");
            }
            if q + i + 1 < dst.len() {
                dst[q + i + 1] ^= hi;
            } else {
                debug_assert!(hi == 0, "a shifted sum overflows");
            }
        }
    }
}

/// `dst += t·x^pos` for a single word `t` whose shifted bits all fall within `dst`.
#[inline]
pub(super) fn add_word_at(dst: &mut [u64], t: u64, pos: usize) {
    let (q, r) = (pos / 64, pos % 64);
    dst[q] ^= t << r;
    if r != 0 {
        let hi = t >> (64 - r);
        if hi != 0 {
            dst[q + 1] ^= hi;
        }
    }
}

/// The quotient and the remainder of `a` by `b`, which is nonzero: the remainder in `a`'s words,
/// the quotient in `a.len()` words.
///
/// # Panics
///
/// Panics if `b` is 0.
#[must_use]
pub(super) fn div_rem(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let db = degree(b).expect("division by the zero polynomial");
    let mut r = a.to_vec();
    let mut q = vec![0u64; a.len()];
    let mut dr = degree(&r);
    while let Some(d) = dr {
        if d < db {
            break;
        }
        let j = d - db;
        q[j / 64] ^= 1 << (j % 64);
        add_shifted(&mut r, b, j);
        dr = degree_below(&r, d);
    }
    (q, r)
}

/// The monic greatest common divisor of `a` and `b`, by Euclid's algorithm.
#[must_use]
pub(super) fn gcd(a: &[u64], b: &[u64]) -> Vec<u64> {
    let len = a.len().max(b.len());
    let mut u = a.to_vec();
    let mut v = b.to_vec();
    u.resize(len, 0);
    v.resize(len, 0);
    let mut du = degree(&u);
    let mut dv = degree(&v);
    loop {
        let Some(dvv) = dv else { return u };
        // u ← u mod v.
        while let Some(d) = du {
            if d < dvv {
                break;
            }
            add_shifted(&mut u, &v, d - dvv);
            du = degree_below(&u, d);
        }
        core::mem::swap(&mut u, &mut v);
        core::mem::swap(&mut du, &mut dv);
    }
}

/// The inverse of `a` modulo `f`, of degree `m` (`f`'s words hold bit `m`), for `a` nonzero of
/// degree below `m` and prime to `f`, in `n` words: Hankerson, Menezes and Vanstone, Algorithm
/// 2.48. The invariants `a·g₁ ≡ u` and `a·g₂ ≡ v (mod f)` hold throughout, and each step lowers
/// the degree of `u` or of `v`; at `u = 1`, `g₁` is the inverse. Returns `None` if `a` is 0 or
/// shares a factor with `f`.
#[must_use]
pub(super) fn inverse(a: &[u64], f: &[u64], n: usize) -> Option<Vec<u64>> {
    let len = f.len();
    let mut u = a.to_vec();
    u.resize(len, 0);
    let mut v = f.to_vec();
    let mut g1 = vec![0u64; len];
    g1[0] = 1;
    let mut g2 = vec![0u64; len];
    let mut du = degree(&u)?;
    let mut dv = degree(&v)?;
    while du != 0 {
        if du < dv {
            core::mem::swap(&mut u, &mut v);
            core::mem::swap(&mut g1, &mut g2);
            core::mem::swap(&mut du, &mut dv);
        }
        let j = du - dv;
        add_shifted(&mut u, &v, j);
        add_shifted(&mut g1, &g2, j);
        du = degree_below(&u, du)?;
    }
    g1.truncate(n);
    Some(g1)
}

/// The `width ≤ 64` bits of `a` from bit `pos` on, as a word.
#[must_use]
pub(super) fn bits(a: &[u64], pos: usize, width: usize) -> u64 {
    let (q, r) = (pos / 64, pos % 64);
    let mut v = a.get(q).map_or(0, |&w| w >> r);
    if r != 0 {
        v |= a.get(q + 1).map_or(0, |&w| w << (64 - r));
    }
    if width < 64 {
        v & ((1 << width) - 1)
    } else {
        v
    }
}

/// The bits of `src` from bit `shift` on, into `dst` (all of whose words are written).
pub(super) fn shift_right_into(dst: &mut [u64], src: &[u64], shift: usize) {
    let (q, r) = (shift / 64, shift % 64);
    let word = |i: usize| src.get(i).copied().unwrap_or(0);
    for (i, d) in dst.iter_mut().enumerate() {
        *d = if r == 0 {
            word(q + i)
        } else {
            (word(q + i) >> r) | (word(q + i + 1) << (64 - r))
        };
    }
}
