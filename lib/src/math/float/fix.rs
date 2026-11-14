//! Fixed-point numbers for the kernels of the exponential and the logarithm: a magnitude `X` of
//! limbs, the least significant first, that stands for `X·2^-W`, `W = 64 n` for the `n` limbs of
//! the fraction, so that a product is the integer product with its `n` low limbs dropped, a
//! truncation below one unit of `2^-W`.

use super::limbs;
use crate::math::numbers::nat;

/// `floor(a·b·2^-W)` for `W = 64 n`, at most `len` limbs kept (the value must fit).
pub(crate) fn mul(a: &[u64], b: &[u64], n: usize, len: usize, scratch: &mut Scratch) -> Vec<u64> {
    let mut out = Vec::new();
    mul_to(&mut out, a, b, n, len, scratch);
    out
}

/// [`mul`] into the vector `out`, `len` limbs, its memory reused.
pub(crate) fn mul_to(
    out: &mut Vec<u64>,
    a: &[u64],
    b: &[u64],
    n: usize,
    len: usize,
    scratch: &mut Scratch,
) {
    let same = core::ptr::eq(a, b);
    let (a, b) = (trimmed(a), trimmed(b));
    out.clear();
    out.resize(len, 0);
    if a.is_empty() || b.is_empty() {
        return;
    }
    let r = &mut scratch.prod;
    if same {
        nat::sqr_into(r, a, &mut scratch.work);
    } else {
        nat::mul_into(r, a, b, &mut scratch.work);
    }
    if r.len() > n {
        let hi = &r[n..];
        let k = hi.len().min(len);
        out[..k].copy_from_slice(&hi[..k]);
        debug_assert!(
            hi[k..].iter().all(|&x| x == 0),
            "a fixed-point product too long"
        );
    }
}

/// `floor(a²·2^-W)`.
#[inline]
pub(crate) fn sqr(a: &[u64], n: usize, len: usize, scratch: &mut Scratch) -> Vec<u64> {
    mul(a, a, n, len, scratch)
}

/// The slice without its zero limbs at the top.
#[inline]
pub(crate) fn trimmed(a: &[u64]) -> &[u64] {
    let mut k = a.len();
    while k > 0 && a[k - 1] == 0 {
        k -= 1;
    }
    &a[..k]
}

/// The working memory of the products.
#[derive(Default)]
pub(crate) struct Scratch {
    prod: Vec<u64>,
    work: Vec<u64>,
}

/// `a = floor(a / d)` in place, for `d` not zero; returns the remainder.
pub(crate) fn div1(a: &mut [u64], d: u64) -> u64 {
    debug_assert!(d != 0);
    let s = d.leading_zeros();
    let dn = d << s;
    let v = nat::invert_limb(dn);
    let len = a.len();
    if len == 0 {
        return 0;
    }
    if s == 0 {
        let mut r = 0u64;
        for i in (0..len).rev() {
            let (q, rr) = nat::div_2by1(r, a[i], dn, v);
            a[i] = q;
            r = rr;
        }
        return r;
    }
    // The dividend shifted left by `s` on the fly; its top `s` bits start the remainder.
    let mut r = a[len - 1] >> (64 - s);
    for i in (0..len).rev() {
        let lo = if i > 0 { a[i - 1] >> (64 - s) } else { 0 };
        let u0 = a[i] << s | lo;
        let (q, rr) = nat::div_2by1(r, u0, dn, v);
        a[i] = q;
        r = rr;
    }
    r >> s
}

/// `a += b·c` for `a` long enough to hold the result.
#[inline]
pub(crate) fn addmul1(a: &mut [u64], b: &[u64], c: u64) {
    let b = trimmed(b);
    if b.is_empty() || c == 0 {
        return;
    }
    let k = b.len();
    let carry = nat::addmul_1(&mut a[..k], b, c);
    add_at(a, k, carry);
}

/// `a += c·2^(64 i)`.
#[inline]
pub(crate) fn add_at(a: &mut [u64], i: usize, c: u64) {
    let mut c = c;
    let mut j = i;
    while c != 0 {
        debug_assert!(j < a.len(), "a fixed-point sum too long");
        let (s, o) = a[j].overflowing_add(c);
        a[j] = s;
        c = u64::from(o);
        j += 1;
    }
}

/// `a·c` for a limb `c`.
#[inline]
pub(crate) fn mul1(a: &[u64], c: u64) -> Vec<u64> {
    let mut r = vec![0u64; a.len() + 1];
    if !a.is_empty() && c != 0 {
        let carry = nat::addmul_1(&mut r[..a.len()], a, c);
        r[a.len()] = carry;
    }
    r
}

/// A signed fixed-point number.
#[derive(Clone, Debug)]
pub(crate) struct SFix {
    pub(crate) neg: bool,
    pub(crate) m: Vec<u64>,
}

impl SFix {
    /// Zero.
    #[inline]
    pub(crate) fn zero() -> SFix {
        SFix {
            neg: false,
            m: Vec::new(),
        }
    }

    /// Whether it is zero.
    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        self.m.iter().all(|&x| x == 0)
    }

    /// `self + o`.
    pub(crate) fn add(&self, o: &SFix) -> SFix {
        let (a, b) = (trimmed(&self.m), trimmed(&o.m));
        if self.neg == o.neg {
            return SFix {
                neg: self.neg,
                m: limbs::add(a, b),
            };
        }
        match limbs::cmp(a, b) {
            core::cmp::Ordering::Less => SFix {
                neg: o.neg,
                m: limbs::sub(b, a),
            },
            _ => {
                let m = limbs::sub(a, b);
                SFix {
                    neg: self.neg && !m.is_empty(),
                    m,
                }
            }
        }
    }

    /// `self - o`.
    #[inline]
    pub(crate) fn sub(&self, o: &SFix) -> SFix {
        let mut o = o.clone();
        o.neg = !o.neg;
        self.add(&o)
    }

    /// The value as an `f64` (for estimates only), at `W = 64 n`.
    pub(crate) fn to_f64(&self, n: usize) -> f64 {
        let m = trimmed(&self.m);
        if m.is_empty() {
            return 0.0;
        }
        let bits = limbs::bit_len(m);
        let (top, sh) = if bits > 64 {
            (limbs::shr(m, bits - 64)[0], bits as i64 - 64)
        } else {
            (m[0], 0)
        };
        let v = top as f64 * pow2f(sh - 64 * n as i64);
        if self.neg { -v } else { v }
    }
}

/// `2^e` as an `f64`, zero or infinity beyond its range.
#[inline]
pub(crate) fn pow2f(e: i64) -> f64 {
    if e < -1074 {
        0.0
    } else if e > 1023 {
        f64::INFINITY
    } else if e < -1022 {
        f64::from_bits(1u64 << (e + 1074))
    } else {
        f64::from_bits(((e + 1023) as u64) << 52)
    }
}

/// `floor(M·2^(lsb + W))` for `W = 64 n`: the fixed-point form of `M·2^lsb`, and whether bits
/// were dropped.
pub(crate) fn of_mag(m: &[u64], lsb: i128, n: usize) -> (Vec<u64>, bool) {
    let s = lsb + 64 * n as i128;
    if s >= 0 {
        (limbs::shl(m, s as u64), false)
    } else {
        let d = (-s) as u64;
        let dropped = !limbs::zero_below(m, d);
        (limbs::shr(m, d), dropped)
    }
}

/// `floor(m / 2^s)` keeping the fixed-point scale.
#[inline]
pub(crate) fn shr(m: &[u64], s: u64) -> Vec<u64> {
    limbs::shr(m, s)
}

/// The number of bits of `a`, zero limbs at the top allowed.
#[inline]
pub(crate) fn bits(a: &[u64]) -> u64 {
    limbs::bit_len(trimmed(a))
}

/// A lower bound of `16 log2(i)` for `i >= 1`: the integer part and a table of the next four bits.
#[inline]
pub(crate) fn log2_16(i: u64) -> u64 {
    // floor(16 log2(1 + j/16)).
    const T: [u64; 16] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15];
    let fl = 63 - u64::from(i.leading_zeros());
    let j = if fl >= 4 {
        (i >> (fl - 4)) & 15
    } else {
        (i << (4 - fl)) & 15
    };
    16 * fl + T[j as usize]
}
