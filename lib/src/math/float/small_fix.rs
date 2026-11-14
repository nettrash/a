//! The kernels of the exponential and the logarithm for at most [`SMALL`] limbs of fraction, in
//! arrays on the stack: the algorithms and the error bounds of [`exp_fix`](super::exp::exp_fix)
//! and [`ln_fix`](super::log::ln_fix), step for step, without the memory allocations that cost
//! more than the arithmetic at these sizes.

use super::ball::Rad;
use super::consts::ln2_approx;
use super::exp::{INV_LN2, isqrt, s_target, terms};
use super::fix;

/// The most limbs of fraction the array kernels take: a working precision of 192 bits.
pub(crate) const SMALL: usize = 3;

/// A magnitude of up to six limbs, the least significant first.
pub(crate) type A = [u64; 6];

/// `floor(a b 2^-W)` for `W = 64 n`, the operands' limbs beyond the product's six dropped.
#[inline]
fn mulhi(a: &[u64], b: &[u64], n: usize) -> A {
    let mut p = [0u64; 12];
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
    let mut r = [0u64; 6];
    r.copy_from_slice(&p[n..n + 6]);
    debug_assert!(p[n + 6..].iter().all(|&x| x == 0));
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
fn bits(a: &A) -> u64 {
    let k = len(a);
    if k == 0 {
        0
    } else {
        64 * (k as u64 - 1) + u64::from(64 - a[k - 1].leading_zeros())
    }
}

/// `a + b`, which fits.
#[inline]
fn add(a: &A, b: &A) -> A {
    let mut r = [0u64; 6];
    let mut c = false;
    for i in 0..6 {
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
fn sub(a: &A, b: &A) -> A {
    let mut r = [0u64; 6];
    let mut c = false;
    for i in 0..6 {
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
fn cmp(a: &A, b: &A) -> core::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// `±a + ±b`.
#[inline]
fn sadd(na: bool, a: &A, nb: bool, b: &A) -> (bool, A) {
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
fn shr(a: &A, s: u64) -> A {
    let (k, b) = ((s / 64) as usize, (s % 64) as u32);
    let mut r = [0u64; 6];
    for i in 0..6 {
        if i + k < 6 {
            let lo = a[i + k] >> b;
            let hi = if b != 0 && i + k + 1 < 6 {
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
fn shl(a: &A, s: u64) -> A {
    let (k, b) = ((s / 64) as usize, (s % 64) as u32);
    let mut r = [0u64; 6];
    for i in (k..6).rev() {
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
fn of(a: &[u64]) -> A {
    let mut r = [0u64; 6];
    let k = len(a);
    debug_assert!(k <= 6);
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
    let x = of(x);
    let bx = bits(&x);
    if bx == 0 {
        return (one, 0, Rad::ZERO);
    }
    let ex = bx as i64 - w as i64;
    debug_assert!(ex <= 62);
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
        let lf = of(&lv);
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

/// [`ln_fix`](super::log::ln_fix) for `n <= SMALL`, one step of Newton's iteration from the
/// `f64` logarithm (which `ln_fix` takes too at these sizes): the sign and the magnitude of `y`
/// and the bound of `|ln u - y|`, the same as `ln_fix`'s.
pub(crate) fn ln_small(u: &[u64], n: usize) -> (bool, A, Rad) {
    debug_assert!(n <= SMALL);
    let w = 64 * n as u64;
    let ul = Rad::pow2(-i128::from(w));
    let mut one = [0u64; 6];
    one[n] = 1;
    let u = of(u);
    let (dn, d) = sadd(false, &u, true, &one);
    if len(&d) == 0 {
        return (false, [0u64; 6], Rad::ZERO);
    }
    // The seed.
    let bd = bits(&d);
    let df = {
        let top = if bd > 64 { shr(&d, bd - 64)[0] } else { d[0] };
        let sh = if bd > 64 { bd as i64 - 64 } else { 0 };
        let v = top as f64 * fix::pow2f(sh - w as i64);
        if dn { -v } else { v }
    };
    let (mut yn, mut y) = if df.abs() > 1e-300 {
        let y0 = df.ln_1p();
        let b = y0.abs().to_bits();
        let (mant, e) = if b >> 52 == 0 {
            (b << 1, -1075i64)
        } else {
            ((b & ((1 << 52) - 1)) | 1 << 52, (b >> 52) as i64 - 1075)
        };
        let s = e + w as i64;
        let m = [mant, 0, 0, 0, 0, 0];
        let y = if s >= 0 {
            shl(&m, s as u64)
        } else {
            shr(&m, (-s) as u64)
        };
        (y0 < 0.0 && len(&y) > 0, y)
    } else {
        (dn, d)
    };
    let (e, k, eps) = exp_small(!yn && len(&y) > 0, &y[..n + 1], n);
    let mut p = mulhi(&u[..n + 2], &e[..n + 2], n);
    if k < 0 {
        p = shr(&p, k.unsigned_abs());
    } else if k > 0 {
        p = shl(&p, k as u64);
    }
    let (vn, v) = sadd(false, &p, true, &one);
    let v2 = mulhi(&v, &v, n);
    let mut v3 = mulhi(&v2, &v, n);
    fix::div1(&mut v3, 3);
    let half = shr(&v2, 1);
    let (pn, poly) = sadd(vn, &v, !false, &half);
    let (pn, poly) = sadd(pn, &poly, vn, &v3);
    (yn, y) = sadd(yn, &y, pn, &poly);
    let vb = bits(&v);
    let vmag = if vb == 0 {
        Rad::ZERO
    } else {
        Rad::pow2(vb as i128 - i128::from(w))
    };
    let dv = eps.mul(Rad::int(1).add(vmag)).add(ul.shl(1));
    if !vmag.le(Rad::pow2(-2)) || !dv.le(Rad::pow2(-4)) {
        return (yn, y, Rad::int(1));
    }
    let rem = vmag.mul(vmag).mul(vmag).mul(vmag).shl(-1);
    (yn, y, dv.shl(1).add(ul.mul(Rad::int(5))).add(rem))
}
