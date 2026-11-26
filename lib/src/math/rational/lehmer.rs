//! Rational reconstruction for a modulus of up to 256 bits, in registers: Wang's method, Euclid's
//! algorithm on the modulus and the residue to the first remainder at most the bound `n`, by
//! Lehmer's algorithm (Knuth's algorithm L, TAOCP 4.5.2) on four limbs.
//!
//! Each round takes the top 63 bits of both remainders at the shift of the larger, and follows
//! Euclid's steps on them with the cofactors of the matrix of the steps while two estimates of the
//! quotient agree, which makes it the true quotient; a step is taken only while the remainder it
//! divides by is certainly above `n`, its estimate less the cofactors' error, so that no round
//! passes the first remainder at most `n`. The matrix is then applied to the four-limb remainders
//! and to the cofactors of the residue. When no step is certain, one division of the full
//! remainders; once the larger fits in 128 bits, Euclid's steps in `u128`s. The cofactors of the
//! residue only grow, so they are kept in 128 bits for a bound `d` below `2^128`: one that would
//! pass them is past `d` too, and the reconstruction fails there.

use crate::math::numbers::{ByteOrder, Number};

/// A magnitude of four limbs, the least significant first.
pub(super) type L4 = [u64; 4];

/// The bits of `a`.
#[inline(always)]
fn bits(a: &L4) -> u32 {
    for i in (0..4).rev() {
        if a[i] != 0 {
            return 64 * i as u32 + 64 - a[i].leading_zeros();
        }
    }
    0
}

/// The 64 bits of `a` from the bit `s` up, for an `a` below `2^(s + 64)`.
#[inline(always)]
fn top(a: &L4, s: u32) -> u64 {
    let (i, b) = ((s / 64) as usize, s % 64);
    let lo = a[i] >> b;
    let hi = if b != 0 && i + 1 < 4 {
        a[i + 1] << (64 - b)
    } else {
        0
    };
    lo | hi
}

/// `a` as a `u128` when it fits.
#[inline(always)]
fn wide(a: &L4) -> Option<u128> {
    (a[2] | a[3] == 0).then(|| u128::from(a[0]) | u128::from(a[1]) << 64)
}

/// Whether `a > n`.
#[inline(always)]
fn above(a: &L4, n: u128) -> bool {
    wide(a).is_none_or(|a| a > n)
}

/// `x·f` in five limbs.
#[inline(always)]
fn mul_1(x: &L4, f: u64) -> [u64; 5] {
    let mut r = [0u64; 5];
    let mut c = 0u64;
    for i in 0..4 {
        let t = u128::from(x[i]) * u128::from(f) + u128::from(c);
        r[i] = t as u64;
        c = (t >> 64) as u64;
    }
    r[4] = c;
    r
}

/// `p - q` for five-limb `p >= q`, of which the result fits in four limbs.
#[inline(always)]
fn sub(p: [u64; 5], q: [u64; 5]) -> L4 {
    let mut r = [0u64; 5];
    let mut borrow = false;
    for i in 0..5 {
        let (s, b1) = p[i].overflowing_sub(q[i]);
        let (s, b2) = s.overflowing_sub(u64::from(borrow));
        r[i] = s;
        borrow = b1 | b2;
    }
    debug_assert!(!borrow && r[4] == 0);
    [r[0], r[1], r[2], r[3]]
}

/// `x·a + y·b` for cofactors of opposite signs (or a zero one) whose combination is not negative:
/// the remainder that the matrix of Lehmer's round gives.
#[inline(always)]
fn combine(a: &L4, x: i128, b: &L4, y: i128) -> L4 {
    let (pa, pb) = (
        mul_1(a, x.unsigned_abs() as u64),
        mul_1(b, y.unsigned_abs() as u64),
    );
    if y > 0 { sub(pb, pa) } else { sub(pa, pb) }
}

/// `(q, r)` with `a = q·b + r`, `0 <= r < b`, for `a >= b > 0`, by the numbers' division.
#[cold]
fn div_rem(a: &L4, b: &L4) -> (Option<u128>, L4) {
    let num = |x: &L4| {
        let v: Vec<u8> = x.iter().flat_map(|l| l.to_le_bytes()).collect();
        Number::new(v, ByteOrder::LittleEndian)
    };
    let (q, r) = num(a).div_rem(&num(b));
    let limbs = |n: &Number| {
        let mut l = [0u64; 4];
        for (i, byte) in n.value.iter().enumerate() {
            l[i / 8] |= u64::from(*byte) << (8 * (i % 8));
        }
        l
    };
    let q = (q.value.len() <= 16).then(|| {
        let l = limbs(&q);
        u128::from(l[0]) | u128::from(l[1]) << 64
    });
    (q, limbs(&r))
}

/// `x / y` for `x >= y > 0`: by the hardware's division when both fit in 64 bits, by
/// subtractions for the small quotients that most steps have.
#[inline(always)]
fn quotient(x: u128, y: u128) -> u128 {
    if (x | y) >> 64 == 0 {
        return u128::from(x as u64 / y as u64);
    }
    let r = x - y;
    if r < y {
        return 1;
    }
    if r - y < y {
        return 2;
    }
    x / y
}

/// `t0 + q·t1` when it fits in 128 bits.
#[inline(always)]
fn next(t0: u128, q: u128, t1: u128) -> Option<u128> {
    q.checked_mul(t1)?.checked_add(t0)
}

/// Wang's reconstruction from the residue `x`, `n < x < m`, modulo `m` of up to four limbs, for the
/// bounds `n` on the numerator and `d < 2^128` on the denominator: the first remainder `p <= n`
/// of Euclid's algorithm on `m` and `x` and its cofactor `t` (`p ≡ t·x`), as the sign of `t`, `p`
/// and `|t|`, when `|t| <= d`; `None` when `|t| > d`. The caller checks that `p` and `t` are
/// prime to each other.
pub(super) fn reconstruct(x: L4, m: L4, n: u128, d: u128) -> Option<(bool, u128, u128)> {
    let (mut a, mut b) = (m, x);
    let (mut t0, mut t1) = (0u128, 1u128);
    let mut positive = true;
    while above(&b, n) {
        if let Some(r0) = wide(&a) {
            // Euclid's steps in `u128`s to the bound.
            let (mut r0, mut r1) = (r0, wide(&b).unwrap_or(0));
            while r1 > n {
                let q = quotient(r0, r1);
                (r0, r1) = (r1, r0 - q * r1);
                (t0, t1) = (t1, next(t0, q, t1)?);
                positive = !positive;
                if t1 > d {
                    return None;
                }
            }
            b = [r1 as u64, (r1 >> 64) as u64, 0, 0];
            break;
        }
        // The top 63 bits of both at the shift of the larger, which has more than 128 bits, and of
        // the bound at that shift (below `2^63`, as `n < 2^128`): with the cofactors, below `2^63`
        // too, the sums of the quotients' estimates fit in 64 bits, for the hardware's division.
        let s = bits(&a) - 63;
        let (mut u, mut v) = (i128::from(top(&a, s)), i128::from(top(&b, s)));
        let nh = if s >= 128 { 0 } else { (n >> s) as i128 };
        let (mut ca, mut cb, mut cc, mut cd) = (1i128, 0i128, 0i128, 1i128);
        let mut steps = 0u32;
        loop {
            // The divisor `v` stands for `cc·a + cd·b` within `|cc| + |cd|` units of `2^s`: the
            // step only when that remainder is certainly above `n`.
            if v - cc.abs() - cd.abs() <= nh
                || v + cc <= 0
                || v + cd <= 0
                || u + ca < 0
                || u + cb < 0
            {
                break;
            }
            // `q = ⌊(u + ca) / (v + cc)⌋`, by subtraction for the quotient of 1 most steps have,
            // and the same of `(u + cb) / (v + cd)`, checked by products.
            let (n1, d1, n2, d2) = (
                (u + ca) as u64,
                (v + cc) as u64,
                (u + cb) as u64,
                (v + cd) as u64,
            );
            let q = if n1 < d1 {
                break;
            } else if n1 - d1 < d1 {
                1
            } else {
                n1 / d1
            };
            let p = u128::from(q) * u128::from(d2);
            if p > u128::from(n2) || u128::from(n2) - p >= u128::from(d2) {
                break;
            }
            let q = i128::from(q);
            (ca, cc) = (cc, ca - q * cc);
            (cb, cd) = (cd, cb - q * cd);
            (u, v) = (v, u - q * v);
            steps += 1;
        }
        if steps == 0 {
            // No certain step: one division of the full remainders. A quotient past 128 bits
            // makes a cofactor past `d`.
            let (q, r) = div_rem(&a, &b);
            (t0, t1) = (t1, next(t0, q?, t1)?);
            positive = !positive;
            (a, b) = (b, r);
        } else {
            (a, b) = (combine(&a, ca, &b, cb), combine(&a, cc, &b, cd));
            let f = |c: i128| c.unsigned_abs();
            let n0 = f(ca).checked_mul(t0)?.checked_add(f(cb).checked_mul(t1)?)?;
            let n1 = f(cc).checked_mul(t0)?.checked_add(f(cd).checked_mul(t1)?)?;
            (t0, t1) = (n0, n1);
            positive ^= steps % 2 == 1;
        }
        if t1 > d {
            return None;
        }
    }
    let p = wide(&b).unwrap_or_else(|| unreachable!());
    (t1 <= d).then_some((!positive && p != 0, p, t1))
}
