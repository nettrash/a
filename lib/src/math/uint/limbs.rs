//! The limb kernels of the fixed-width integers, `const` so that the methods built on them can be:
//! comparison, products truncated to the width, shifts, and division by Knuth's algorithm D.
//!
//! The limbs are 64-bit words, the low one first. A product of two limbs and the carries fit in a
//! `u128`, which is what the kernels compute in. No kernel divides a `u128`: on targets without a
//! 128-by-64-bit division instruction (aarch64 among them) that is a call into a software routine,
//! tens of cycles a limb. The quotient limbs come instead from products by a precomputed
//! reciprocal, the 2-by-1 and 3-by-2 divisions of Möller and Granlund ("Improved division by
//! invariant integers", IEEE Transactions on Computers 60(2), 2011).

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

/// Up to this many limbs the truncated products and squares run loops of fixed length, which
/// unroll, whatever the operands; above, the rows stop at their significant limbs.
// Measured on an Apple M4 Max (sweep 2 to 16 limbs; operands of the full width, of about half the
// width, of mixed widths and of products that just fit, by `wrapping_mul`, `checked_mul`, `pow`,
// `checked_pow` and `wrapping_pow`): up to 10 limbs the loops of fixed length take 0.36 to 1.00 of
// the time of the rows over the significant limbs; from 11 limbs on, where they no longer unroll,
// they take 1.2 to 1.8 times as long for products of operands short of the full width.
const FIXED: usize = 10;

/// The product `a·b` modulo `2^(64·N)`: the products of limbs that land below the width, the top
/// limb of each row by a low product alone. Where the significant limbs fit, the full product of
/// them; for operands of the full width (or up to [`FIXED`] limbs), all `N(N + 1)/2` in loops of
/// fixed length, which unroll; otherwise rows over the significant limbs.
#[inline]
pub(super) const fn mul_low<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let (la, lb) = (significant(a), significant(b));
    if la + lb <= N {
        mul_short(a, b, la, lb)
    } else if N <= FIXED || (la == N && lb == N) {
        mul_low_fixed(a, b)
    } else {
        mul_low_rows(a, b, la, lb)
    }
}

/// [`mul_low`] in rows over the `la` significant limbs of `a`, each over those of `b` that land
/// below the width.
const fn mul_low_rows<const N: usize>(
    a: &[u64; N],
    b: &[u64; N],
    la: usize,
    lb: usize,
) -> [u64; N] {
    let mut r = [0u64; N];
    let mut i = 0;
    while i < la {
        let ai = a[i];
        let end = if lb < N - i { lb } else { N - i };
        let mut carry = 0;
        let mut j = 0;
        while j + 1 < end {
            (r[i + j], carry) = mac(r[i + j], ai, b[j], carry);
            j += 1;
        }
        if end == N - i {
            r[N - 1] = r[N - 1]
                .wrapping_add(ai.wrapping_mul(b[N - 1 - i]))
                .wrapping_add(carry);
        } else {
            // A row inside the width: its last product and the carry, into a limb no row before
            // has reached.
            let (lo, hi) = mac(r[i + j], ai, b[j], carry);
            r[i + j] = lo;
            r[i + j + 1] = hi;
        }
        i += 1;
    }
    r
}

/// [`mul_low`] in rows of fixed length.
#[inline]
const fn mul_low_fixed<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut r = [0u64; N];
    let mut i = 0;
    while i < N {
        let ai = a[i];
        let mut carry = 0;
        let mut j = 0;
        while j + 1 < N - i {
            (r[i + j], carry) = mac(r[i + j], ai, b[j], carry);
            j += 1;
        }
        r[N - 1] = r[N - 1]
            .wrapping_add(ai.wrapping_mul(b[N - 1 - i]))
            .wrapping_add(carry);
        i += 1;
    }
    r
}

/// The product of the `la` and `lb` low limbs of `a` and `b`, `la + lb ≤ N`, which fits.
const fn mul_short<const N: usize>(a: &[u64; N], b: &[u64; N], la: usize, lb: usize) -> [u64; N] {
    let mut r = [0u64; N];
    let mut i = 0;
    while i < la {
        let ai = a[i];
        let mut carry = 0;
        let mut j = 0;
        while j < lb {
            (r[i + j], carry) = mac(r[i + j], ai, b[j], carry);
            j += 1;
        }
        // i + lb < N; no row before has reached it.
        r[i + lb] = carry;
        i += 1;
    }
    r
}

/// The square `a²` modulo `2^(64·N)`: each product `aᵢ·aⱼ` with `i < j` and `i + j < N` once,
/// the sum doubled by a shift, then the squares `aᵢ²` added, about half the products of
/// [`mul_low`]; over the significant limbs above [`FIXED`] limbs.
#[inline]
pub(super) const fn sqr_low<const N: usize>(a: &[u64; N]) -> [u64; N] {
    let mut r = [0u64; N];
    if N == 0 {
        return r;
    }
    let la = if N <= FIXED { N } else { significant(a) };
    // The cross products, rows i with j from i + 1 to the end of the row.
    let mut i = 0;
    while i + 1 < la && 2 * i + 1 < N {
        let ai = a[i];
        let end = if la < N - i { la } else { N - i };
        let mut carry = 0;
        let mut j = i + 1;
        while j + 1 < end {
            (r[i + j], carry) = mac(r[i + j], ai, a[j], carry);
            j += 1;
        }
        if end == N - i {
            r[N - 1] = r[N - 1]
                .wrapping_add(ai.wrapping_mul(a[N - 1 - i]))
                .wrapping_add(carry);
        } else {
            let (lo, hi) = mac(r[i + j], ai, a[j], carry);
            r[i + j] = lo;
            r[i + j + 1] = hi;
        }
        i += 1;
    }
    // Doubled.
    let mut k = N - 1;
    while k > 0 {
        r[k] = (r[k] << 1) | (r[k - 1] >> 63);
        k -= 1;
    }
    r[0] <<= 1;
    // The squares on the diagonal, aᵢ² at 2i.
    let mut carry = 0u64;
    let mut i = 0;
    while i < la && 2 * i < N {
        let p = a[i] as u128 * a[i] as u128;
        let t = r[2 * i] as u128 + (p as u64) as u128 + carry as u128;
        r[2 * i] = t as u64;
        carry = 0;
        if 2 * i + 1 < N {
            let t2 = r[2 * i + 1] as u128 + (p >> 64) + (t >> 64);
            r[2 * i + 1] = t2 as u64;
            carry = (t2 >> 64) as u64;
        }
        i += 1;
    }
    // A carry past the last square runs up through the limbs above it.
    let mut k = 2 * i;
    while carry != 0 && k < N {
        let (v, c) = r[k].overflowing_add(carry);
        r[k] = v;
        carry = c as u64;
        k += 1;
    }
    r
}

/// Whether the true product `a·b` is `2^(64·N)` or more: no, when the lengths in bits sum to at
/// most `64·N`; yes, above `64·N + 1`; and at `64·N + 1` exactly when the full product's high half
/// is not zero.
#[inline]
pub(super) const fn mul_overflows<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
    if N == 1 {
        return a[0].overflowing_mul(b[0]).1;
    }
    if N == 2 {
        let x = a[0] as u128 | (a[1 % N] as u128) << 64;
        let y = b[0] as u128 | (b[1 % N] as u128) << 64;
        return x.overflowing_mul(y).1;
    }
    let (la, lb) = (bit_length(a), bit_length(b));
    if la == 0 || lb == 0 || la + lb <= 64 * N as u32 {
        false
    } else if la + lb > 64 * N as u32 + 1 {
        true
    } else {
        mul_high_nonzero(a, b)
    }
}

/// The number of bits up to the highest set one.
#[inline]
pub(super) const fn bit_length(x: &[u64]) -> u32 {
    let n = significant(x);
    if n == 0 {
        0
    } else {
        64 * n as u32 - x[n - 1].leading_zeros()
    }
}

/// Whether the high half of the full product `a·b` is nonzero.
#[cold]
const fn mul_high_nonzero<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
    let mut low = [0u64; N];
    let mut high = [0u64; N];
    let mut i = 0;
    while i < N {
        let mut c = 0;
        let mut j = 0;
        while j < N {
            let k = i + j;
            if k < N {
                (low[k], c) = mac(low[k], a[i], b[j], c);
            } else {
                (high[k - N], c) = mac(high[k - N], a[i], b[j], c);
            }
            j += 1;
        }
        high[i] = c;
        i += 1;
    }
    significant(&high) != 0
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

/// `⌊(2^19 − 3·2^8)/d₉⌋` for the nine-bit `d₉` from 256 to 511: the start of [`reciprocal_word`].
const RECIPROCAL_START: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = ((((1u32 << 19) - 3 * (1 << 8)) / (256 + i as u32)) & 0xffff) as u16;
        i += 1;
    }
    t
};

/// The reciprocal `⌊(2¹²⁸ − 1)/d⌋ − 2⁶⁴` of a normalised limb `d ≥ 2⁶³`, by Möller and Granlund's
/// Algorithm 3: an 11-bit start from a table, two Newton steps in 64-bit products, a third that
/// doubles the precision again, and a last correction that makes it exact. About a dozen products,
/// no division.
pub(super) const fn reciprocal_word(d: u64) -> u64 {
    debug_assert!(d >> 63 == 1, "the divisor must be normalised");
    let d0 = d & 1;
    let d9 = d >> 55;
    let d40 = (d >> 24) + 1;
    let d63 = (d >> 1) + d0;
    let v0 = RECIPROCAL_START[(d9 - 256) as usize] as u64;
    let v1 = (v0 << 11) - ((v0 * v0 * d40) >> 40) - 1;
    let v2 = (v1 << 13) + ((v1 * ((1u64 << 60) - v1 * d40)) >> 47);
    // 2⁹⁶ − v₂·d₆₃ + ⌊v₂/2⌋·d₀, modulo 2⁶⁴, where 2⁹⁶ is 0.
    let e = ((v2 >> 1) & d0.wrapping_neg()).wrapping_sub(v2.wrapping_mul(d63));
    let v3 = (v2 << 31).wrapping_add(((v2 as u128 * e as u128) >> 65) as u64);
    // v₃ − ⌊(v₃ + 2⁶⁴ + 1)·d / 2⁶⁴⌋, modulo 2⁶⁴.
    let high = (((v3 as u128 + 1) * d as u128) >> 64) as u64;
    v3.wrapping_sub(high.wrapping_add(d))
}

/// The reciprocal `⌊(2¹⁹² − 1)/(d₁·2⁶⁴ + d₀)⌋ − 2⁶⁴` of a normalised two-limb divisor (`d₁ ≥ 2⁶³`),
/// from that of `d₁` adjusted by `d₀` (Möller and Granlund, Algorithm 6).
pub(super) const fn reciprocal_3by2(d1: u64, d0: u64) -> u64 {
    let mut v = reciprocal_word(d1);
    let mut p = d1.wrapping_mul(v).wrapping_add(d0);
    if p < d0 {
        v = v.wrapping_sub(1);
        if p >= d1 {
            v = v.wrapping_sub(1);
            p = p.wrapping_sub(d1);
        }
        p = p.wrapping_sub(d1);
    }
    let t = v as u128 * d0 as u128;
    let (t1, t0) = ((t >> 64) as u64, t as u64);
    p = p.wrapping_add(t1);
    if p < t1 {
        v = v.wrapping_sub(1);
        if p > d1 || (p == d1 && t0 >= d0) {
            v = v.wrapping_sub(1);
        }
    }
    v
}

/// The quotient and the remainder of `u₁·2⁶⁴ + u₀` by the normalised `d`, for `u₁ < d`, with `v`
/// the reciprocal of `d` (Möller and Granlund, Algorithm 4): one product by `v`, one by `d`, and
/// at most two corrections, the second rare.
#[inline(always)]
pub(super) const fn div_2by1(u1: u64, u0: u64, d: u64, v: u64) -> (u64, u64) {
    // v·u₁ + u₁·2⁶⁴ + u₀ < 2¹²⁸ for u₁ < d.
    let q = (v as u128 * u1 as u128).wrapping_add(((u1 as u128) << 64) | u0 as u128);
    let mut q1 = ((q >> 64) as u64).wrapping_add(1);
    let q0 = q as u64;
    let mut r = u0.wrapping_sub(q1.wrapping_mul(d));
    if r > q0 {
        q1 = q1.wrapping_sub(1);
        r = r.wrapping_add(d);
    }
    if r >= d {
        q1 += 1;
        r -= d;
    }
    (q1, r)
}

/// The quotient of `u₂·2¹²⁸ + u₁·2⁶⁴ + u₀` by the normalised two-limb `d₁·2⁶⁴ + d₀` and the two
/// limbs of the remainder, for `(u₂, u₁) < (d₁, d₀)`, with `v` from [`reciprocal_3by2`] (Möller and
/// Granlund, Algorithm 5). The quotient is below 2⁶⁴.
#[inline(always)]
pub(super) const fn div_3by2(
    u2: u64,
    u1: u64,
    u0: u64,
    d1: u64,
    d0: u64,
    v: u64,
) -> (u64, u64, u64) {
    let q = (v as u128 * u2 as u128).wrapping_add(((u2 as u128) << 64) | u1 as u128);
    let mut q1 = (q >> 64) as u64;
    let q0 = q as u64;
    let r1 = u1.wrapping_sub(q1.wrapping_mul(d1));
    let d = ((d1 as u128) << 64) | d0 as u128;
    let mut r = (((r1 as u128) << 64) | u0 as u128)
        .wrapping_sub(d0 as u128 * q1 as u128)
        .wrapping_sub(d);
    q1 = q1.wrapping_add(1);
    if (r >> 64) as u64 >= q0 {
        q1 = q1.wrapping_sub(1);
        r = r.wrapping_add(d);
    }
    if r >= d {
        q1 += 1;
        r -= d;
    }
    (q1, (r >> 64) as u64, r as u64)
}

/// `⌊√t⌋` of a `u128`: from 64 bits by the standard library's `u64::isqrt`; above, `t` shifted
/// up by an even amount until one of its top two bits is set, its root by [`sqrt_rem_2`], shifted
/// back down.
pub(super) const fn isqrt_u128(t: u128) -> u64 {
    if t >> 64 == 0 {
        return (t as u64).isqrt();
    }
    let sh = t.leading_zeros() & !1;
    let tn = t << sh;
    sqrt_rem_2((tn >> 64) as u64, tn as u64).0 >> (sh / 2)
}

/// The quotient and the remainder of `u` by the one-limb divisor `d > 0`: one limb or less by the
/// hardware's 64-bit division, longer ones from the high limb down by [`div_2by1`], the divisor and
/// the dividend shifted until the divisor's top bit is set.
pub(super) const fn div_rem_1<const N: usize>(u: &[u64; N], d: u64) -> ([u64; N], u64) {
    let mut q = [0u64; N];
    let len = significant(u);
    if len <= 1 {
        if len == 1 {
            q[0] = u[0] / d;
            return (q, u[0] % d);
        }
        return (q, 0);
    }
    let s = d.leading_zeros();
    let dn = d << s;
    let v = reciprocal_word(dn);
    // The top limb of the shifted dividend, below 2^s ≤ dn, is the first remainder.
    let mut r = if s == 0 { 0 } else { u[len - 1] >> (64 - s) };
    let mut i = len;
    while i > 0 {
        i -= 1;
        let limb = if s == 0 {
            u[i]
        } else if i > 0 {
            (u[i] << s) | (u[i - 1] >> (64 - s))
        } else {
            u[i] << s
        };
        let (qi, ri) = div_2by1(r, limb, dn, v);
        q[i] = qi;
        r = ri;
    }
    (q, r >> s)
}

/// The quotient and the remainder of `u` by `v`, which must not be zero, by Knuth's algorithm D
/// (*The Art of Computer Programming*, vol. 2, 4.3.1).
///
/// The divisor's `n` significant limbs are shifted up until its top bit is set (D1), and the
/// dividend as far, into `len + 1` limbs; [`div_normalized`] takes the quotient limbs from the top
/// (D2 to D7), and the remainder is the low `n` limbs, shifted back down (D8). A divisor of one
/// limb takes the plain loop of [`div_rem_1`] instead. `O(n·m)` limb products.
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
        vn[i] = shift_in(v, i, s);
        i += 1;
    }
    let mut w = [[0u64; N]; 2];
    let un = w.as_flattened_mut();
    i = 0;
    while i <= len {
        un[i] = shift_in(u, i, s);
        i += 1;
    }
    let mut q = [0u64; N];
    let d = vn.split_at(n).0;
    let inv = reciprocal_3by2(d[n - 1], d[n - 2]);
    if n == N {
        div_normalized::<N>(un, len, d, inv, &mut q);
    } else {
        div_normalized::<0>(un, len, d, inv, &mut q);
    }
    // D8: the remainder, the low n limbs shifted back down; the limb above them is zero.
    un[n] = 0;
    let mut r = [0u64; N];
    i = 0;
    while i < n {
        r[i] = shift_out(un, i, s);
        i += 1;
    }
    (q, r)
}

/// Limb `i` of `x·2ˢ` for `s < 64`, `i` up to the length of `x`, without a branch on `s`: the
/// bits that come up from the limb below are shifted by `64 − s` in two steps, so that `s = 0`
/// brings none.
#[inline(always)]
pub(super) const fn shift_in(x: &[u64], i: usize, s: u32) -> u64 {
    let high = if i < x.len() { x[i] } else { 0 };
    let low = if i > 0 { x[i - 1] } else { 0 };
    high << s | (low >> 1) >> (63 - s)
}

/// Limb `i` of `x/2ˢ` for `s < 64` and `i + 1` below the length of `x`, without a branch on `s`.
#[inline(always)]
pub(super) const fn shift_out(x: &[u64], i: usize, s: u32) -> u64 {
    x[i] >> s | (x[i + 1] << 1) << (63 - s)
}

/// Steps D2 to D7 of Knuth's algorithm D: the `len + 1` limbs at `u` divided by the `n ≥ 2` limbs
/// of `d`, both normalised — the top bit of `d` set, the top `n` limbs of `u` below `d`, as they
/// are when both were shifted by the same amount — with `v` from [`reciprocal_3by2`] of the top two
/// limbs of `d`. The remainder is left in the low `n` limbs of `u`, the limbs above them unspecified,
/// and the `len − n + 1` quotient limbs go to `q` where it has room for them (an empty `q` takes
/// none). With `M > 0`, `n` must be `M`, a constant, so that the loops unroll.
///
/// Each quotient limb, from the top, is the quotient of the top three limbs of the partial
/// remainder by the top two of `d` (D3), by [`div_3by2`] with the reciprocal: the true limb or one
/// more. The top two limbs of the partial remainder are kept in registers, as the 3-by-2 division
/// gives them, so only the low `n − 2` limbs of `d` times the estimate are subtracted (D4), and the
/// borrow off the two on top; `d` is added back once if that went below zero (D6), with a
/// probability of about `2/2⁶⁴` on random operands.
#[inline(always)]
pub(super) const fn div_normalized<const M: usize>(
    u: &mut [u64],
    len: usize,
    d: &[u64],
    v: u64,
    q: &mut [u64],
) {
    let n = if M > 0 { M } else { d.len() };
    debug_assert!(n >= 2 && n == d.len() && len >= n && d[n - 1] >> 63 == 1);
    let (v1, v0) = (d[n - 1], d[n - 2]);
    let (mut r1, mut r0) = (u[len], u[len - 1]);
    let mut j = len - n + 1;
    while j > 0 {
        j -= 1;
        let mut i;
        if r1 == v1 && r0 == v0 {
            // The remainder so far is below the divisor, so its top two limbs are at most the
            // divisor's. Equal, the quotient limb is 2⁶⁴ − 1 exactly: with V = (v1, v0)·b^(n−2) + Y
            // and the partial remainder P = b·(v1, v0)·b^(n−2) + X, P − (b − 1)·V =
            // (v1, v0)·b^(n−2) + X − (b − 1)·Y, and (v1, v0)·b^(n−2) ≥ 2⁶³·b^(n−1) > (b − 1)·Y.
            // So the whole divisor times it is subtracted from the n + 1 limbs at j (D4), with
            // nothing to add back.
            u[j + n] = r1;
            u[j + n - 1] = r0;
            let mut carry = 0u64;
            let mut borrow = false;
            i = 0;
            while i < n {
                let (lo, hi) = mac(0, u64::MAX, d[i], carry);
                carry = hi;
                let (x, b1) = u[i + j].overflowing_sub(lo);
                let (x, b2) = x.overflowing_sub(borrow as u64);
                u[i + j] = x;
                borrow = b1 | b2;
                i += 1;
            }
            debug_assert!(u[j + n] == carry + borrow as u64, "the limb is exact");
            if j < q.len() {
                q[j] = u64::MAX;
            }
            (r1, r0) = (u[j + n - 1], u[j + n - 2]);
            continue;
        }
        // D3: the quotient of the top three limbs by the divisor's top two, the true limb or one
        // more (as Knuth's test with the third limb finds), and the two limbs of that remainder.
        let (mut qhat, t1, t0) = div_3by2(r1, r0, u[j + n - 2], v1, v0, v);
        // D4: the low n − 2 limbs of the divisor times qhat off the n − 2 limbs below, the
        // borrow off the two limbs on top.
        let mut carry = 0u64;
        let mut borrow = false;
        i = 0;
        while i + 2 < n {
            let (lo, hi) = mac(0, qhat, d[i], carry);
            carry = hi;
            let (x, b1) = u[i + j].overflowing_sub(lo);
            let (x, b2) = x.overflowing_sub(borrow as u64);
            u[i + j] = x;
            borrow = b1 | b2;
            i += 1;
        }
        let low = carry as u128 + borrow as u128;
        let top = ((t1 as u128) << 64) | t0 as u128;
        let (mut top, negative) = top.overflowing_sub(low);
        if negative {
            // D6: one too large; the divisor added back, the carry out of the low limbs into the
            // top two, whose carry cancels the borrow.
            qhat -= 1;
            let mut c = false;
            i = 0;
            while i + 2 < n {
                let (x, c1) = u[i + j].overflowing_add(d[i]);
                let (x, c2) = x.overflowing_add(c as u64);
                u[i + j] = x;
                c = c1 | c2;
                i += 1;
            }
            top = top
                .wrapping_add(((v1 as u128) << 64) | v0 as u128)
                .wrapping_add(c as u128);
        }
        if j < q.len() {
            q[j] = qhat;
        }
        (r1, r0) = ((top >> 64) as u64, top as u64);
    }
    u[n - 1] = r1;
    u[n - 2] = r0;
}

/// `(s, r)` with `s = ⌊√t⌋` and `r = t − s² ≤ 2s` for the normalised `t = hi·2⁶⁴ + lo`,
/// `hi ≥ 2⁶²`: one step of Zimmermann's Karatsuba square root (*Karatsuba Square Root*, INRIA
/// RR-3805, 1999) on digits of 32 bits, `t = a₃·b³ + a₂·b² + a₁·b + a₀` with `b = 2³²`: the root
/// `s'` and the remainder `r'` of the top two digits (`hi`) by the standard library's `u64::isqrt`,
/// then `q = ⌊(r'·b + a₁)/(2s')⌋` by one 64-bit division, `s = s'·b + q` and
/// `r = (r'·b + a₁ − 2s'·q)·b + a₀ − q²`; as `a₃ ≥ b/4`, `s` is the root or one more, and a
/// negative `r` brings it down by one.
pub(super) const fn sqrt_rem_2(hi: u64, lo: u64) -> (u64, u128) {
    debug_assert!(hi >> 62 != 0, "the number must be normalised");
    let s1 = hi.isqrt();
    // 2³¹ ≤ s1 < 2³², and r1 ≤ 2·s1 < 2³³.
    let r1 = hi - s1 * s1;
    // x = r1·b + a₁ < 2⁶⁵, so its half y fits: ⌊y/s1⌋ = ⌊x/(2·s1)⌋, at most b.
    let y = (r1 << 31) | (lo >> 33);
    let (q, v) = (y / s1, y % s1);
    // x − 2·s1·q = 2·(y − s1·q) + the low bit of x, below 2·s1.
    let u = (v << 1) | ((lo >> 32) & 1);
    let s = ((s1 as u128) << 32) + q as u128;
    let r = (((u as u128) << 32) | (lo & 0xffff_ffff) as u128) as i128 - (q as i128 * q as i128);
    if r < 0 {
        // s is one above the root, which is below 2⁶⁴; (s − 1)² = s² − 2s + 1.
        ((s - 1) as u64, (r + 2 * s as i128 - 1) as u128)
    } else {
        (s as u64, r as u128)
    }
}

/// The square root with its remainder by Zimmermann's Karatsuba square root (*Karatsuba Square
/// Root*, INRIA RR-3805, 1999): writes `⌊√a⌋` of the `2L` limbs at `a`, `L = s.len() ≥ 1`, whose
/// top limb is at least 2⁶², to `s`, and leaves the remainder `a − s²`, at most `2s`, in the low
/// `L` limbs of `a` and its top bit in the result; `w` is scratch of at least `L` limbs.
///
/// With `l = ⌊L/2⌋`, `h = L − l` and `b = 2^(64·l)`, `a = H·b² + a₁·b + a₀`: the root `s'` and the
/// remainder `r'` of the top `2h` limbs `H`, recursively (normalised as `a` is, so
/// `s' ≥ 2^(64·h)/2 ≥ b/2`); then `q = ⌊(r'·b + a₁)/(2s')⌋`, as `⌊y/s'⌋` for the half `y` of the
/// dividend, by algorithm D, `s = s'·b + q` and `r = (r'·b + a₁ − 2s'·q)·b + a₀ − q²`. As
/// `s' ≥ b/2`, `s` is the root or one more: a negative `r` takes `2s − 1`, and `s` one. Two square
/// roots of half the length, a division of `L` limbs by `h` and a square of `l` limbs: about the
/// cost of the division.
pub(super) const fn sqrt_rem(s: &mut [u64], a: &mut [u64], w: &mut [u64]) -> bool {
    let len = s.len();
    debug_assert!(a.len() >= 2 * len && w.len() >= len && a[2 * len - 1] >> 62 != 0);
    if len == 1 {
        let (root, r) = sqrt_rem_2(a[1], a[0]);
        s[0] = root;
        a[0] = r as u64;
        return r >> 64 != 0;
    }
    let l = len / 2;
    let h = len - l;
    // s' to s[l..len], and r' to a[2l..2l + h] with its top bit c1.
    let c1 = {
        let (_, high) = s.split_at_mut(l);
        let (_, top) = a.split_at_mut(2 * l);
        sqrt_rem(high, top, w)
    };
    // The half y of x = c1·2^(64·len) + a[l..l + len] = r'·b + a₁ to w[0..len], and the low bit
    // of x.
    let low_bit = a[l] & 1;
    let mut i = 0;
    while i + 1 < len {
        w[i] = (a[l + i] >> 1) | (a[l + i + 1] << 63);
        i += 1;
    }
    w[len - 1] = (a[l + len - 1] >> 1) | ((c1 as u64) << 63);
    let (q, root) = s.split_at_mut(l);
    // The top h limbs of y are ⌊r'/2⌋ ≤ s': equal only where r' = 2s', and then the quotient is
    // b and s'·b + b one above the root, so q = b − 1 is taken at once, with the remainder
    // y − s'·(b − 1) = (y mod b) + s', of h limbs and a carry.
    let mut carry = 0u64;
    if !matches!(
        compare(w.split_at(len).0.split_at(l).1, root),
        Ordering::Less
    ) {
        i = 0;
        while i < l {
            q[i] = u64::MAX;
            i += 1;
        }
        while i < len {
            w[i] = 0;
            i += 1;
        }
        let mut c = false;
        i = 0;
        while i < h {
            let (x, c1) = w[i].overflowing_add(root[i]);
            let (x, c2) = x.overflowing_add(c as u64);
            w[i] = x;
            c = c1 | c2;
            i += 1;
        }
        carry = c as u64;
    } else if h == 1 {
        // A divisor of one limb, normalised, and l = 1: the 2-by-1 division.
        let (qq, r) = div_2by1(w[1], w[0], root[0], reciprocal_word(root[0]));
        q[0] = qq;
        w[0] = r;
    } else {
        let v = reciprocal_3by2(root[h - 1], root[h - 2]);
        div_normalized::<0>(w.split_at_mut(len).0, len - 1, root, v, q);
    }
    // u = 2·(y − s'·q) + the low bit, from w[0..h] and the carry, to a[l..len] above a₀, with its
    // top two bits in `top`.
    let mut top = (carry << 1) | (w[h - 1] >> 63);
    i = h;
    while i > 1 {
        i -= 1;
        a[l + i] = (w[i] << 1) | (w[i - 1] >> 63);
    }
    a[l] = (w[0] << 1) | low_bit;
    // r = u·b + a₀ − q²: the square of q's l limbs to w[0..2l], off a[0..len] and `top`.
    i = 0;
    while i < 2 * l {
        w[i] = 0;
        i += 1;
    }
    i = 0;
    while i < l {
        let mut c = 0u64;
        let mut j = 0;
        while j < l {
            (w[i + j], c) = mac(w[i + j], q[i], q[j], c);
            j += 1;
        }
        w[i + l] = c;
        i += 1;
    }
    let mut borrow = false;
    i = 0;
    while i < len {
        let x = if i < 2 * l { w[i] } else { 0 };
        let (x, b1) = a[i].overflowing_sub(x);
        let (x, b2) = x.overflowing_sub(borrow as u64);
        a[i] = x;
        borrow = b1 | b2;
        i += 1;
    }
    let negative;
    (top, negative) = top.overflowing_sub(borrow as u64);
    if negative {
        // r < 0, so s is one above the root: r + 2s − 1 as r + s, then s − 1, then + s again.
        let mut pass = 0;
        while pass < 2 {
            if pass == 1 {
                // s ≥ 1 loses one.
                i = 0;
                while i < len {
                    let x = if i < l { &mut q[i] } else { &mut root[i - l] };
                    let (y, b) = x.overflowing_sub(1);
                    *x = y;
                    if !b {
                        break;
                    }
                    i += 1;
                }
            }
            let mut c = false;
            i = 0;
            while i < len {
                let x = if i < l { q[i] } else { root[i - l] };
                let (y, c1) = a[i].overflowing_add(x);
                let (y, c2) = y.overflowing_add(c as u64);
                a[i] = y;
                c = c1 | c2;
                i += 1;
            }
            top = top.wrapping_add(c as u64);
            pass += 1;
        }
    }
    debug_assert!(top <= 1, "the remainder is at most 2s");
    top != 0
}
