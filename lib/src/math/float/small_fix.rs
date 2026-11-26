//! The kernels of the exponential and the logarithm for at most [`SMALL`] limbs of fraction, in
//! arrays on the stack: the algorithm and the error bound of [`exp_fix`](super::exp::exp_fix) step
//! for step, and the logarithm by tables computed when the crate is compiled ([`ln_small`]),
//! without the memory allocations that cost more than the arithmetic at these sizes.

use super::ball::Rad;
use super::consts::ln2_approx;
use super::exp::{INV_LN2, isqrt, s_target, terms};
use super::fix;

/// The most limbs of fraction the array kernels take: a working precision of 256 bits.
pub(crate) const SMALL: usize = 4;

/// A magnitude of up to six limbs, the least significant first.
pub(crate) type A = [u64; 6];

/// `floor(a b 2^-W)` for `W = 64 n`, the product's limbs beyond the array's dropped (they must be
/// zero), `L <= 12`.
#[inline]
fn mulhi<const L: usize>(a: &[u64], b: &[u64], n: usize) -> [u64; L] {
    let mut p = [0u64; 24];
    let (la, lb) = (len(a), len(b));
    for i in 0..la {
        let ai = a[i];
        if ai == 0 {
            continue;
        }
        let mut c = 0u64;
        for j in 0..lb {
            let t = u128::from(ai) * u128::from(b[j]) + u128::from(p[i + j]) + u128::from(c);
            p[i + j] = t as u64;
            c = (t >> 64) as u64;
        }
        p[i + lb] = c;
    }
    let mut r = [0u64; L];
    r.copy_from_slice(&p[n..n + L]);
    debug_assert!(p[n + L..].iter().all(|&x| x == 0));
    r
}

/// The length without the zero limbs at the top.
#[inline]
fn len(a: &[u64]) -> usize {
    let mut k = a.len();
    while k > 0 && a[k - 1] == 0 {
        k -= 1;
    }
    k
}

/// The number of bits.
#[inline]
fn bits<const L: usize>(a: &[u64; L]) -> u64 {
    let k = len(a);
    if k == 0 {
        0
    } else {
        64 * (k as u64 - 1) + u64::from(64 - a[k - 1].leading_zeros())
    }
}

/// `a + b`, which fits.
#[inline]
fn add<const L: usize>(a: &[u64; L], b: &[u64; L]) -> [u64; L] {
    let mut r = [0u64; L];
    let mut c = false;
    for i in 0..L {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(u64::from(c));
        r[i] = s;
        c = c1 || c2;
    }
    debug_assert!(!c);
    r
}

/// `a - b` for `a >= b`.
#[inline]
fn sub<const L: usize>(a: &[u64; L], b: &[u64; L]) -> [u64; L] {
    let mut r = [0u64; L];
    let mut c = false;
    for i in 0..L {
        let (s, c1) = a[i].overflowing_sub(b[i]);
        let (s, c2) = s.overflowing_sub(u64::from(c));
        r[i] = s;
        c = c1 || c2;
    }
    debug_assert!(!c);
    r
}

/// The order of the magnitudes.
#[inline]
fn cmp<const L: usize>(a: &[u64; L], b: &[u64; L]) -> core::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// `±a + ±b`.
#[inline]
fn sadd<const L: usize>(na: bool, a: &[u64; L], nb: bool, b: &[u64; L]) -> (bool, [u64; L]) {
    if na == nb {
        return (na, add(a, b));
    }
    match cmp(a, b) {
        core::cmp::Ordering::Less => (nb, sub(b, a)),
        _ => {
            let r = sub(a, b);
            (na && len(&r) > 0, r)
        }
    }
}

/// `floor(a / 2^s)`.
#[inline]
fn shr<const L: usize>(a: &[u64; L], s: u64) -> [u64; L] {
    let (k, b) = ((s / 64) as usize, (s % 64) as u32);
    let mut r = [0u64; L];
    for i in 0..L {
        if i + k < L {
            let lo = a[i + k] >> b;
            let hi = if b != 0 && i + k + 1 < L {
                a[i + k + 1] << (64 - b)
            } else {
                0
            };
            r[i] = lo | hi;
        }
    }
    r
}

/// `a 2^s`, which fits.
#[inline]
fn shl<const L: usize>(a: &[u64; L], s: u64) -> [u64; L] {
    let (k, b) = ((s / 64) as usize, (s % 64) as u32);
    let mut r = [0u64; L];
    for i in (k..L).rev() {
        let hi = a[i - k] << b;
        let lo = if b != 0 && i > k {
            a[i - k - 1] >> (64 - b)
        } else {
            0
        };
        r[i] = hi | lo;
    }
    r
}

/// The limbs `a` in an array.
#[inline]
fn of<const L: usize>(a: &[u64]) -> [u64; L] {
    let mut r = [0u64; L];
    let k = len(a);
    debug_assert!(k <= L);
    r[..k].copy_from_slice(&a[..k]);
    r
}

/// [`exp_fix`](super::exp::exp_fix) for `n <= SMALL`: `(y, k, ε)` with `e^x = y 2^(k - W) (1 +
/// θ)`, `|θ| <= ε`, by the same steps and the same bound.
pub(crate) fn exp_small(neg: bool, x: &[u64], n: usize) -> (A, i64, Rad) {
    debug_assert!(n <= SMALL);
    let w = 64 * n as u64;
    let u = Rad::pow2(-i128::from(w));
    let mut one = [0u64; 6];
    one[n] = 1;
    let x: A = of(x);
    let bx = bits(&x);
    if bx == 0 {
        return (one, 0, Rad::ZERO);
    }
    let ex = bx as i64 - w as i64;
    debug_assert!(ex <= 62 && (n < 4 || ex <= 56));
    let mut k: i64 = if ex <= 0 {
        if neg { -1 } else { 0 }
    } else {
        let top = if bx >= 64 {
            shr(&x, bx - 64)[0]
        } else {
            x[0] << (64 - bx)
        };
        let q = ((u128::from(top) * u128::from(INV_LN2)) >> (127 - ex)) as i64;
        if neg { -q } else { q }
    };
    let (r, delta_r) = if k == 0 && !neg && ex <= 0 {
        (x, Rad::ZERO)
    } else {
        let g = 64 - (k.unsigned_abs() + 8).leading_zeros() as u64 + 4;
        let a = ln2_approx(w + g + 8);
        let (lv, dropped) = fix::of_mag(&a.m, a.lsb + i128::from(g), n);
        let lf: A = of(&lv);
        let mut delta_l = Rad::pow2(a.err);
        if dropped {
            delta_l = delta_l.add(Rad::pow2(-i128::from(w + g)));
        }
        loop {
            // `|k| L` in six limbs: `L < 2^(W + g)`, `|k| < 2^(g - 4)`.
            let c = k.unsigned_abs();
            let mut kl = [0u64; 6];
            let mut carry = 0u64;
            for i in 0..6 {
                let t = u128::from(lf[i]) * u128::from(c) + u128::from(carry);
                kl[i] = t as u64;
                carry = (t >> 64) as u64;
            }
            debug_assert!(carry == 0);
            let kl = shr(&kl, g);
            let (rn, r) = sadd(neg, &x, k >= 0, &kl);
            if rn && len(&r) > 0 {
                k -= 1;
                continue;
            }
            if bits(&r) > w {
                k += 1;
                continue;
            }
            break (r, delta_l.mul(Rad::int(c)).add(u));
        }
    };
    let br = bits(&r);
    if br == 0 {
        return (one, k, Rad::int(2).mul(delta_r).add(u));
    }
    let er = br as i64 - w as i64;
    let s = (s_target(w) + er).max(0) as u64;
    let t = shr(&r, s);
    let tau = (s as i64 - er) as u64;
    // The series, as in `taylor`.
    let big_n = terms(tau, w);
    let lq = (64 * 16 / fix::log2_16(big_n + 2).max(16)).max(1);
    let m = (isqrt(big_n) + 1).min(lq).clamp(1, 15) as usize;
    let mut pw = [[0u64; 6]; 16];
    pw[1] = t;
    for j in 2..=m {
        pw[j] = mulhi(&pw[j - 1][..n], &t[..n], n);
    }
    let mut sum = [0u64; 6];
    let mut first = true;
    let mut err = 0u64;
    let mut b = big_n + 1;
    while b > 0 {
        let mut a = b;
        let mut q: u64 = 1;
        while a > 0 && ((b - a) as usize) < m {
            match q.checked_mul(a) {
                Some(v) => {
                    q = v;
                    a -= 1;
                }
                None => break,
            }
        }
        let l = (b - a) as usize;
        let mut acc = if first {
            first = false;
            [0u64; 6]
        } else {
            mulhi(&pw[l][..n], &sum[..n + 1], n)
        };
        let mut c: u64 = 1;
        let mut i = b;
        while i > a {
            i -= 1;
            c *= i + 1;
            if i == a {
                fix::add_at(&mut acc[..n + 2], n, c);
            } else {
                fix::addmul1(&mut acc[..n + 2], &pw[(i - a) as usize][..n], c);
            }
        }
        fix::div1(&mut acc[..n + 2], q);
        err = err.div_ceil(q) + 8;
        sum = acc;
        b = a;
    }
    let e = err + 1;
    let mut y = sum;
    for _ in 0..s {
        y = mulhi(&y[..n + 1], &y[..n + 1], n);
    }
    let slack = Rad::up(513, -9);
    let eps_s = Rad::int(e + 1).mul(u).shl(i128::from(s)).mul(slack);
    if !eps_s.le(Rad::pow2(-20)) {
        return (y, k, Rad::int(1));
    }
    let eps_in = delta_r.add(u.shl(i128::from(s))).shl(1);
    (y, k, eps_s.add(eps_in).mul(slack))
}

/// The limbs of the constants' arithmetic when the crate is compiled: ten after the point (640
/// bits) and two before it.
type C = [u64; 12];

/// The limbs after the point of [`C`].
const CF: usize = 10;

/// `floor(a / d)` for `d` not zero.
const fn c_div(a: C, d: u64) -> C {
    let mut q = [0u64; 12];
    let mut r: u128 = 0;
    let mut i = 12;
    while i > 0 {
        i -= 1;
        let cur = (r << 64) | a[i] as u128;
        q[i] = (cur / d as u128) as u64;
        r = cur % d as u128;
    }
    q
}

/// `a·m`, which fits.
const fn c_mul(a: C, m: u64) -> C {
    let mut r = [0u64; 12];
    let mut carry: u128 = 0;
    let mut i = 0;
    while i < 12 {
        let t = a[i] as u128 * m as u128 + carry;
        r[i] = t as u64;
        carry = t >> 64;
        i += 1;
    }
    r
}

/// `a + b`, which fits.
const fn c_add(a: C, b: C) -> C {
    let mut r = [0u64; 12];
    let mut carry = 0u64;
    let mut i = 0;
    while i < 12 {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(carry);
        r[i] = s;
        carry = (c1 || c2) as u64;
        i += 1;
    }
    r
}

/// `a - b` for `a >= b`.
const fn c_sub(a: C, b: C) -> C {
    let mut r = [0u64; 12];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < 12 {
        let (s, c1) = a[i].overflowing_sub(b[i]);
        let (s, c2) = s.overflowing_sub(borrow);
        r[i] = s;
        borrow = (c1 || c2) as u64;
        i += 1;
    }
    r
}

/// Whether `a >= b`.
const fn c_ge(a: &C, b: &C) -> bool {
    let mut i = 12;
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

/// `2 atanh(p/q)` for `0 <= p < q < 2^32` at 640 bits after the point, from below within `2^-629`:
/// `Σ_j 2 z^(2j+1)/(2j+1)` for `z = p/q`, each `τ_j = floor(floor(τ_(j-1) p²) / q²)` within `j` units
/// of `z^(2j+1)` (a floor each, the error before times `z² < 1`), each term `floor(τ_j/(2j+1))`
/// within `1.5`, until `τ_j` is zero, which leaves a tail below one unit: at most 1.5 units for
/// each of at most 300 terms (`z <= 1/3` needs 202), doubled, below `2^10`, all from below.
const fn atanh2(p: u64, q: u64) -> C {
    let mut num = [0u64; 12];
    num[CF] = p;
    let mut tau = c_div(num, q);
    let (p2, q2) = (p * p, q * q);
    let mut sum = [0u64; 12];
    let mut j = 0u64;
    loop {
        let mut zero = true;
        let mut i = 0;
        while i < 12 {
            zero &= tau[i] == 0;
            i += 1;
        }
        if zero {
            break;
        }
        sum = c_add(sum, c_div(tau, 2 * j + 1));
        tau = c_div(c_mul(tau, p2), q2);
        j += 1;
    }
    c_add(sum, sum)
}

/// `floor(2^1280 / d)` for `d` of at most 642 bits, by the long division of the bits.
const fn c_inv(d: C) -> C {
    let mut q = [0u64; 12];
    let mut r = [0u64; 12];
    let mut i: i64 = 1280;
    while i >= 0 {
        // `r = 2r + bit i of 2^1280`.
        let mut k = 11;
        while k > 0 {
            r[k] = r[k] << 1 | r[k - 1] >> 63;
            k -= 1;
        }
        r[0] <<= 1;
        if i == 1280 {
            r[0] |= 1;
        }
        if c_ge(&r, &d) {
            r = c_sub(r, d);
            q[(i / 64) as usize] |= 1 << (i % 64);
        }
        i -= 1;
    }
    q
}

/// `ln 2` at 640 bits, from below within `2^-629`.
const LN2_C: C = atanh2(1, 3);

/// `1/ln 2` at 640 bits, within `2^-627`: the quotient's floor, and `ln 2`'s error `δ` moves it
/// by `δ/ln² 2 < 2.1 δ`.
const INV_LN2_C: C = c_inv(LN2_C);

/// `1/ln 10` at 640 bits, within `2^-627`, `ln 10 = 3 ln 2 + 2 atanh(1/9)` within `2^-627`, and
/// `δ/ln² 10 < δ/5`.
const INV_LN10_C: C = c_inv(c_add(c_mul(LN2_C, 3), atanh2(1, 9)));

/// The most limbs after the point of the logarithm by tables.
pub(crate) const LN_LIMBS: usize = 8;

/// The limbs after the point of the tables of logarithms: 576 bits, cut from the 640 of the sums
/// (within `2^-575` from below).
const TAB: usize = 9;

/// The least `d` of the first table, `ln(d/128)` for `d` in `FIRST..FIRST + 96`: `u` in
/// `[0.625, 1.375)`, beyond the `[2/3, 4/3)` of the kernel.
const FIRST: u64 = 80;

/// `|ln(d/128)|` for `d = FIRST + i`, at [`TAB`] limbs after the point, computed when the crate is
/// compiled: `2 atanh(|d - 128| / (d + 128))`.
static LN_T1: [[u64; TAB]; 96] = table1();

/// `ln(1 + j/2^14)` for `j < 208`, at [`TAB`] limbs after the point: `2 atanh(j / (2^15 + j))`.
static LN_T2: [[u64; TAB]; 208] = table2();

/// The most terms of the series: `ceil((64 LN_LIMBS + 1)/14) - 1`.
const TERMS: usize = 37;

/// `floor(2^576/k)` for `2 <= k <= TERMS` (and nothing for 0 and 1), the coefficients of the
/// series.
static INV_K: [[u64; TAB]; TERMS + 1] = inverses();

const fn top(c: C) -> [u64; TAB] {
    let mut t = [0u64; TAB];
    let mut i = 0;
    while i < TAB {
        t[i] = c[CF - TAB + i];
        i += 1;
    }
    t
}

const fn table1() -> [[u64; TAB]; 96] {
    let mut t = [[0u64; TAB]; 96];
    let mut i = 0;
    while i < 96 {
        let d = FIRST + i as u64;
        let p = d.abs_diff(128);
        t[i] = top(atanh2(p, d + 128));
        i += 1;
    }
    t
}

const fn table2() -> [[u64; TAB]; 208] {
    let mut t = [[0u64; TAB]; 208];
    let mut j = 0;
    while j < 208 {
        t[j] = top(atanh2(j as u64, (1 << 15) + j as u64));
        j += 1;
    }
    t
}

const fn inverses() -> [[u64; TAB]; TERMS + 1] {
    let mut t = [[0u64; TAB]; TERMS + 1];
    let mut one = [0u64; 12];
    one[CF] = 1;
    let mut k = 2;
    while k <= TERMS {
        t[k] = top(c_div(one, k as u64));
        k += 1;
    }
    t
}

/// A table entry cut to `n` limbs after the point, from below.
#[inline]
fn cut<const L: usize>(t: &[u64; TAB], n: usize) -> [u64; L] {
    let mut r = [0u64; L];
    r[..n].copy_from_slice(&t[TAB - n..]);
    r
}

/// `1/k` at `n` limbs after the point, from below (exact for `k = 1`).
#[inline]
fn inv_k<const L: usize>(k: usize, n: usize) -> [u64; L] {
    if k == 1 {
        let mut r = [0u64; L];
        r[n] = 1;
        r
    } else {
        cut(&INV_K[k], n)
    }
}

/// The logarithm for `n <= LN_LIMBS` limbs after the point, in arrays of `L >= n + 3` limbs: the
/// sign and the magnitude of `y` and the bound of `|ln u - y|` for the fixed-point `u` in about
/// `[2/3, 4/3)`, by tables, as Arb's logarithm at short precisions.
///
/// With `d_1 = floor(128 u)`, `y_1 = floor(128 u / d_1)` in `[1, 1 + 1/d_1)`; with `d_2 =
/// floor(2^14 y_1)`, `y_2 = floor(2^14 y_1 / d_2) = 1 + t` and `t < 2^-14`; `ln u = ln(d_1/128) +
/// ln(d_2/2^14) + ln(1 + t)` but for the two floors (one unit each, as `y_i >= 1`). `ln(1 + t) = t
/// q_1`, `q_k = 1/k - t q_(k+1)`, `q_N = 1/N`, by Horner's rule: each `q_k` within `2 + t e_(k+1)`
/// units (the coefficient and the product cut), at most `2/(1 - t)`; `t q_1` within `1 + 2t/(1 - t)
/// < 1.01` units, and the tail of the series, below `t^(N+1)/(N+1) <= 2^-(W+1)`, half a unit. The
/// two table entries cut to `W` bits are within a unit and `2^-574` each: at most `6u + 2^-573` in
/// all. `None` when `u` lies beyond the first table.
pub(crate) fn ln_small<const L: usize>(u: &[u64], n: usize) -> Option<(bool, [u64; L], Rad)> {
    debug_assert!((1..=LN_LIMBS).contains(&n) && n + 3 <= L);
    let w = 64 * n as u64;
    let unit = Rad::pow2(-i128::from(w));
    let u: [u64; L] = of(fix::trimmed(u));
    let d1 = u[n] << 7 | u[n - 1] >> 57;
    if !(FIRST..FIRST + 96).contains(&d1) {
        return None;
    }
    let mut y1 = shl(&u, 7);
    fix::div1(&mut y1[..n + 1], d1);
    debug_assert!(y1[n] == 1);
    let j2 = (y1[n - 1] >> 50) as usize;
    debug_assert!(j2 < 208);
    let mut t = shl(&y1, 14);
    fix::div1(&mut t[..n + 1], (1 << 14) + j2 as u64);
    debug_assert!(t[n] == 1);
    t[n] = 0;
    let mut s = [0u64; L];
    let tb = bits(&t);
    if tb > 0 {
        // `t < 2^-(W - tb)`: the least `N` with `(N + 1)(W - tb) >= W + 1`.
        let nt = ((w + 1).div_ceil(w - tb) - 1).max(1) as usize;
        let mut q: [u64; L] = inv_k(nt, n);
        for k in (1..nt).rev() {
            let tq: [u64; L] = mulhi(&t[..n], &q[..n + 1], n);
            q = sub(&inv_k(k, n), &tq);
        }
        s = mulhi(&t[..n], &q[..n + 1], n);
    }
    let t1: [u64; L] = cut(&LN_T1[(d1 - FIRST) as usize], n);
    let pos = add(&cut(&LN_T2[j2], n), &s);
    let (neg, y) = sadd(false, &pos, d1 < 128, &t1);
    Some((neg, y, unit.mul(Rad::int(6)).add(Rad::pow2(-573))))
}

/// The base of a logarithm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Base {
    E,
    Two,
    Ten,
}

/// `floor(M 2^s)` in `len <= L` limbs (the value must fit) and whether bits were dropped.
#[inline]
fn window<const L: usize>(m: &[u64], s: i128, len: usize) -> ([u64; L], bool) {
    let mut r = [0u64; L];
    if s >= 0 {
        let (k, b) = ((s / 64) as usize, (s % 64) as u32);
        for (i, &l) in m.iter().enumerate() {
            if i + k < len {
                r[i + k] |= l << b;
            }
            if b != 0 && i + k + 1 < len {
                r[i + k + 1] |= l >> (64 - b);
            }
        }
        return (r, false);
    }
    let d = s.unsigned_abs();
    if d >= 64 * m.len() as u128 {
        return (r, m.iter().any(|&l| l != 0));
    }
    let (k, b) = ((d / 64) as usize, (d % 64) as u32);
    let mut dropped = m[..k].iter().any(|&l| l != 0);
    if b != 0 {
        dropped |= m[k] << (64 - b) != 0;
    }
    for (i, ri) in r.iter_mut().enumerate().take(len) {
        let j = i + k;
        if j >= m.len() {
            break;
        }
        let hi = if b != 0 && j + 1 < m.len() {
            m[j + 1] << (64 - b)
        } else {
            0
        };
        *ri = m[j] >> b | hi;
    }
    (r, dropped)
}

/// `floor(a c 2^-640)` for the constant `c` of [`C`] (ten limbs after the point, one before),
/// `a` of at most `n + 1` limbs, the result in `n + 1` limbs (it must fit).
#[inline]
fn mul_const<const L: usize>(a: &[u64; L], n: usize, c: &C) -> [u64; L] {
    let mut p = [0u64; 24];
    for i in 0..=n {
        let ai = a[i];
        if ai == 0 {
            continue;
        }
        let mut carry = 0u64;
        for j in 0..CF + 1 {
            let t = u128::from(ai) * u128::from(c[j]) + u128::from(p[i + j]) + u128::from(carry);
            p[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        p[i + CF + 1] = carry;
    }
    let mut r = [0u64; L];
    r[..n + 1].copy_from_slice(&p[CF..CF + n + 1]);
    debug_assert!(p[CF + n + 1..].iter().all(|&x| x == 0));
    r
}

/// The logarithm in the base `base` of `x = M 2^lsb` (above zero) for `u = x 2^-e` in about
/// `[2/3, 4/3)`, at `W = 64 n` bits after the point (`n <= LN_LIMBS`), in arrays of `L >= n + 3`
/// limbs: the sign, the magnitude and the bound of its error, `None` when `u` lies beyond the
/// kernel's tables.
///
/// `u` cut to `W` bits is within a unit below, which moves `ln u` by at most `1.5u`; `e ln 2` with
/// 640 bits of `ln 2` and cut to `W` bits, within `u + |e| 2^-629 < u + 2^-566`. For base 2,
/// `log2 x = e + ln(u)/ln 2`, the error of `ln u` times `1/ln 2 < 1.45`, the product cut (a unit)
/// and the constant's error times `|ln u| < 1/2`; for base 10, `(e ln 2 + ln u)/ln 10`, the error
/// times `1/ln 10 < 1/2`, a unit and `2^63 2^-627`.
pub(crate) fn log_small<const L: usize>(
    m: &[u64],
    lsb: i128,
    e: i64,
    n: usize,
    base: Base,
) -> Option<(bool, [u64; L], Rad)> {
    let w = 64 * n as u64;
    let unit = Rad::pow2(-i128::from(w));
    let (u, dropped): ([u64; L], bool) = window(m, lsb - i128::from(e) + i128::from(w), n + 1);
    let (mut neg, mut y, mut err) = ln_small::<L>(&u[..n + 1], n)?;
    if dropped {
        err = err.add(unit.shl(1));
    }
    if base == Base::Two {
        y = mul_const(&y, n, &INV_LN2_C);
        err = err.mul(Rad::up(3, -1)).add(unit).add(Rad::pow2(-620));
        if e != 0 {
            let mut ei = [0u64; L];
            ei[n] = e.unsigned_abs();
            (neg, y) = sadd(neg, &y, e < 0, &ei);
        }
        return Some((neg, y, err));
    }
    if e != 0 {
        let el = c_mul(LN2_C, e.unsigned_abs());
        let mut l = [0u64; L];
        l[..n + 1].copy_from_slice(&el[CF - n..CF + 1]);
        debug_assert!(el[CF + 1] == 0);
        (neg, y) = sadd(neg, &y, e < 0, &l);
        err = err.add(unit).add(Rad::pow2(-566));
    }
    if base == Base::Ten {
        y = mul_const(&y, n, &INV_LN10_C);
        err = err.shl(-1).add(unit).add(Rad::pow2(-563));
    }
    Some((neg, y, err))
}
