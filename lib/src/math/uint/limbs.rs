//! The limb kernels of the fixed-width integers, `const` so that the methods built on them can be:
//! comparison, products truncated to the width, shifts, and division by Knuth's algorithm D.
//!
//! The limbs are 64-bit words, the low one first. A product of two limbs and the carries fit in a
//! `u128`, which is what the kernels compute in.

use core::cmp::Ordering;

/// The number of limbs of `x` up to its highest nonzero one: 0 for zero.
#[inline]
pub(super) const fn significant(x: &[u64]) -> usize {
    let mut n = x.len();
    while n > 0 && x[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// Compares `a` and `b`, of the same length, from the high limb down.
#[inline]
pub(super) const fn compare(a: &[u64], b: &[u64]) -> Ordering {
    let mut i = a.len();
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return if a[i] < b[i] {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
    }
    Ordering::Equal
}

/// `a + b·c + carry` as its low and high limbs; it never overflows two limbs.
#[inline(always)]
pub(super) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 * c as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// The product `a·b` modulo `2^(64·N)` and whether the true product is `2^(64·N)` or more.
///
/// Only the products of limbs that land below the width are computed, at most `N(N + 1)/2`, over
/// the significant limbs: the true product overflows exactly when a pair of nonzero limbs lands at
/// or above the width, or a row's carry does.
pub(super) const fn mul_low<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut r = [0u64; N];
    let (la, lb) = (significant(a), significant(b));
    let mut overflow = false;
    let mut i = 0;
    while i < la {
        let ai = a[i];
        if ai != 0 {
            // Of b's limbs, those below N − i land inside the width.
            let jmax = if lb < N - i { lb } else { N - i };
            let mut carry = 0;
            let mut j = 0;
            while j < jmax {
                (r[i + j], carry) = mac(r[i + j], ai, b[j], carry);
                j += 1;
            }
            if i + jmax < N {
                // jmax = lb: the row's carry is the next limb, not written by a row before.
                r[i + jmax] = carry;
            } else if carry != 0 {
                overflow = true;
            }
            // A row cut short drops a product of ai with a nonzero limb: b's highest, at least.
            if jmax < lb {
                overflow = true;
            }
        }
        i += 1;
    }
    (r, overflow)
}

/// `x·m + a` modulo `2^(64·N)` and the limb above, the carry out.
#[inline]
pub(super) const fn mul_add_1<const N: usize>(x: &[u64; N], m: u64, a: u64) -> ([u64; N], u64) {
    let mut r = [0u64; N];
    let mut carry = a;
    let mut i = 0;
    while i < N {
        (r[i], carry) = mac(0, x[i], m, carry);
        i += 1;
    }
    (r, carry)
}

/// `x << k` for `k < 64·N`, the bits shifted out at the top dropped.
pub(super) const fn shl<const N: usize>(x: &[u64; N], k: u32) -> [u64; N] {
    let (whole, s) = ((k / 64) as usize, k % 64);
    let mut r = [0u64; N];
    let mut i = N;
    while i > whole {
        i -= 1;
        let src = i - whole;
        r[i] = if s == 0 {
            x[src]
        } else if src > 0 {
            (x[src] << s) | (x[src - 1] >> (64 - s))
        } else {
            x[src] << s
        };
    }
    r
}

/// `x >> k` for `k < 64·N`, the limbs above `x` taken as `fill`: 0 for the logical shift,
/// `u64::MAX` for the arithmetic shift of a negative number.
pub(super) const fn shr<const N: usize>(x: &[u64; N], k: u32, fill: u64) -> [u64; N] {
    let (whole, s) = ((k / 64) as usize, k % 64);
    let mut r = [fill; N];
    let mut i = 0;
    while i + whole < N {
        let low = x[i + whole];
        let high = if i + whole + 1 < N {
            x[i + whole + 1]
        } else {
            fill
        };
        r[i] = if s == 0 {
            low
        } else {
            (low >> s) | (high << (64 - s))
        };
        i += 1;
    }
    r
}

/// The quotient and the remainder of `u` by the one-limb divisor `d > 0`, by the plain loop from
/// the high limb down, each step a two-limb by one-limb division.
pub(super) const fn div_rem_1<const N: usize>(u: &[u64; N], d: u64) -> ([u64; N], u64) {
    let mut q = [0u64; N];
    let d = d as u128;
    let mut rem: u128 = 0;
    let mut i = significant(u);
    while i > 0 {
        i -= 1;
        let cur = (rem << 64) | u[i] as u128;
        q[i] = (cur / d) as u64;
        rem = cur % d;
    }
    (q, rem as u64)
}

/// The quotient and the remainder of `u` by `v`, which must not be zero, by Knuth's algorithm D
/// (*The Art of Computer Programming*, vol. 2, 4.3.1).
///
/// The divisor's `n` significant limbs are shifted up until its top bit is set (D1), and the
/// dividend as far, into `n + m + 1` limbs. Each quotient limb, from the top (D2), is estimated
/// from the top two limbs of the remainder over the top limb of the divisor, then lowered while the
/// divisor's second limb shows it too large (D3): at most twice, and afterwards it is the true limb
/// or one more. The divisor times the estimate is subtracted (D4), and added back once if that
/// went below zero (D6), which happens with a probability of about `2/2⁶⁴` on random operands. The
/// remainder is the low `n` limbs, shifted back down (D8). A divisor of one limb takes the plain
/// loop of [`div_rem_1`] instead. `O(n·m)` limb products.
///
/// # Panics
///
/// If `v` is zero; the callers check first, to panic with the message of their operation.
pub(super) const fn div_rem<const N: usize>(u: &[u64; N], v: &[u64; N]) -> ([u64; N], [u64; N]) {
    let n = significant(v);
    assert!(n > 0, "attempt to divide by zero");
    let len = significant(u);
    if len < n || (len == n && matches!(compare(u, v), Ordering::Less)) {
        return ([0; N], *u);
    }
    if n == 1 {
        let (q, r) = div_rem_1(u, v[0]);
        let mut rem = [0u64; N];
        rem[0] = r;
        return (q, rem);
    }
    // D1: normalise. With n ≥ 2 limbs, N ≥ 2 and the 2N limbs of `w` hold the len + 1 of the
    // shifted dividend.
    let s = v[n - 1].leading_zeros();
    let mut vn = [0u64; N];
    let mut i = 0;
    while i < n {
        vn[i] = if s == 0 {
            v[i]
        } else if i > 0 {
            (v[i] << s) | (v[i - 1] >> (64 - s))
        } else {
            v[i] << s
        };
        i += 1;
    }
    let mut w = [[0u64; N]; 2];
    let un = w.as_flattened_mut();
    un[len] = if s == 0 { 0 } else { u[len - 1] >> (64 - s) };
    i = 0;
    while i < len {
        un[i] = if s == 0 {
            u[i]
        } else if i > 0 {
            (u[i] << s) | (u[i - 1] >> (64 - s))
        } else {
            u[i] << s
        };
        i += 1;
    }
    let (v1, v0) = (vn[n - 1], vn[n - 2]);
    let b: u128 = 1 << 64;
    let mut q = [0u64; N];
    // D2: the quotient limbs from j = len − n down to 0.
    let mut j = len - n + 1;
    while j > 0 {
        j -= 1;
        // D3: the estimate from the top two limbs, at most 2⁶⁴ + 1 as the remainder so far is below
        // the divisor; then the test with the third limb, which leaves the estimate at most one
        // too large.
        let num = ((un[j + n] as u128) << 64) | un[j + n - 1] as u128;
        let mut qhat = num / v1 as u128;
        let mut rhat = num - qhat * v1 as u128;
        while qhat >= b || qhat * v0 as u128 > ((rhat << 64) | un[j + n - 2] as u128) {
            qhat -= 1;
            rhat += v1 as u128;
            if rhat >= b {
                break;
            }
        }
        let qhat = qhat as u64;
        // D4: subtract qhat times the divisor from the n + 1 limbs at j.
        let mut carry = 0u64;
        let mut borrow = false;
        i = 0;
        while i < n {
            let (lo, hi) = mac(0, qhat, vn[i], carry);
            carry = hi;
            let (d, b1) = un[i + j].overflowing_sub(lo);
            let (d, b2) = d.overflowing_sub(borrow as u64);
            un[i + j] = d;
            borrow = b1 | b2;
            i += 1;
        }
        let (d, b1) = un[j + n].overflowing_sub(carry);
        let (d, b2) = d.overflowing_sub(borrow as u64);
        un[j + n] = d;
        if b1 | b2 {
            // D6: the estimate was one too large; add the divisor back, the carry out of the top
            // limb cancelling the borrow.
            q[j] = qhat.wrapping_sub(1);
            let mut c = false;
            i = 0;
            while i < n {
                let (t, c1) = un[i + j].overflowing_add(vn[i]);
                let (t, c2) = t.overflowing_add(c as u64);
                un[i + j] = t;
                c = c1 | c2;
                i += 1;
            }
            un[j + n] = un[j + n].wrapping_add(c as u64);
        } else {
            q[j] = qhat;
        }
    }
    // D8: the remainder, the low n limbs shifted back down.
    let mut r = [0u64; N];
    i = 0;
    while i < n {
        r[i] = if s == 0 {
            un[i]
        } else {
            (un[i] >> s) | (un[i + 1] << (64 - s))
        };
        i += 1;
    }
    (q, r)
}
