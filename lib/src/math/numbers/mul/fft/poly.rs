//! Products of polynomials modulo a word `n` by the transforms of the multiplication: the
//! convolution of the coefficients, each below `n`, is computed modulo enough of the primes that
//! it is exact, and each coefficient is brought back modulo `n` from its residues.
//!
//! A coefficient below `n` is cut into `dn` digits of `b <= 28` bits, the least `dn` with
//! `dn b >= bits(n)`, from which [`split`] forms its residue modulo each prime as it forms those
//! of the digits of an integer. A coefficient of the product is a sum of at most `m` products,
//! `m` the length of the shorter operand, so below `m (n - 1)^2`, and `np` primes are used, the
//! fewest whose product `P` is more than twice that, so that the Chinese remainder theorem's `k`
//! is found as in the multiplication.
//!
//! The coefficient is `c = sum y_i (P / q_i) - k P` for the normalized residues `y_i` that
//! [`finish`] leaves, and modulo `n` it is `sum y_i ((P / q_i) mod n) + k (-P mod n)`: `np + 1`
//! products of a word by a residue, summed in 128 bits (below `2^97`), and one division by the
//! normalized `n 2^s` with its reciprocal, the constants taken times `2^s` so that the sum needs
//! no shift.
//!
//! The transforms are cyclic: [`mul_cyclic`] gives the product modulo `x^(2^k) - 1` for operands
//! of up to `2^k` coefficients, for the divisions that know the coefficients that wrap around.
//!
//! Polynomials over the integers of coefficients of at most 112 bits ([`mul_int`]) take the same
//! transforms with the digits of each coefficient's magnitude negated for a negative one, and
//! primes whose product `P` exceeds four times the largest sum: the residues are shifted by those
//! of `H = floor(P / 2)`, so that the theorem gives `c + H`, between `P / 4` and `3P / 4`, where
//! the estimate of its `k` holds, and `c` is that less `H`, computed in four limbs.

use super::super::super::div::limbs::{div_2by1, invert_limb};
use super::{
    Lanes, MAX_K, MIN_K, PRIME_DATA, PRIMES, V, barrett, centred, finish, forward, inverse,
    pointwise, pow_mod, split, twiddles,
};

/// The most coefficients of a product the transforms take: `2^21`.
pub(crate) const MAX_LEN: usize = 1 << MAX_K;

/// How a product is computed: the primes, the transform's `k` and the digits of a coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    /// The number of primes, 2 to 8.
    pub(crate) np: usize,
    /// The transforms' `k`, `2^k` points.
    pub(crate) k: u32,
    /// The digits of a coefficient.
    dn: usize,
    /// The bits of a digit.
    b: usize,
}

/// The bits of the product of the first `np` primes, rounded down.
fn primes_bits(np: usize) -> f64 {
    PRIMES[..np].iter().map(|&(q, _)| f64::from(q).log2()).sum()
}

/// The plan of a product of `la` and `lb` coefficients modulo a word of `bits` bits, if the
/// transforms take it: `la + lb - 1 <= 2^21`. `lb` is 0 for a square.
pub(crate) fn plan(la: usize, lb: usize, bits: u32) -> Option<Plan> {
    let (len, m) = if lb == 0 {
        (2 * la - 1, la)
    } else {
        (la + lb - 1, la.min(lb))
    };
    if len > MAX_LEN || la == 0 {
        return None;
    }
    // 2 m (n - 1)^2 < 2^(1 + bits(m) + 2 bits(n)), with a margin for the rounding of the logs.
    let need = 1.0 + f64::from(usize::BITS - m.leading_zeros()) + 2.0 * f64::from(bits) + 0.01;
    let np = (2..=8).find(|&np| primes_bits(np) > need)?;
    let bits = bits as usize;
    let dn = bits.div_ceil(28);
    let b = bits.div_ceil(dn);
    let k = (usize::BITS - (len - 1).leading_zeros()).max(MIN_K);
    Some(Plan { np, k, dn, b })
}

/// The `u32` digits of the product of the primes `0..np` but `skip` (none when `skip >= np`).
fn product_digits(np: usize, skip: usize) -> [u64; 8] {
    let mut p = [0u64; 8];
    p[0] = 1;
    for (i, &(q, _)) in PRIMES.iter().enumerate().take(np) {
        if i == skip {
            continue;
        }
        let mut carry = 0u64;
        for d in p.iter_mut() {
            let t = *d * u64::from(q) + carry;
            *d = t & 0xffff_ffff;
            carry = t >> 32;
        }
    }
    p
}

/// The `u32` digits `p` (least significant first) modulo `n`.
fn digits_mod(p: &[u64; 8], n: u64) -> u64 {
    let mut r = 0u128;
    for &d in p.iter().rev() {
        r = ((r << 32) | u128::from(d)) % u128::from(n);
    }
    r as u64
}

/// Sets digit `t` of coefficient `j` of `x` at `d[t stride + j]`, zeros up to `stride`.
///
/// # Safety
///
/// The `dn stride` digits at `d` must be valid for writes, `x.len() <= stride`.
unsafe fn digits(d: *mut i32, stride: usize, x: &[u64], dn: usize, b: usize) {
    let mask = (1u64 << b) - 1;
    // SAFETY: every digit written lies below `dn stride`.
    unsafe {
        for t in 0..dn {
            let row = d.add(t * stride);
            for (j, &c) in x.iter().enumerate() {
                *row.add(j) = ((c >> (t * b)) & mask) as i32;
            }
            for j in x.len()..stride {
                *row.add(j) = 0;
            }
        }
    }
}

/// Sets `r` to the `la + lb - 1` coefficients of the product of `a` and `b` (of `a` and `a` when
/// `b` is empty, a square) modulo `n`, all coefficients below `n`, by the plan `p` of
/// [`plan`]`(a.len(), b.len(), bits(n))`.
///
/// # Panics
///
/// Panics if `r` is not as long as the product, or `a` is empty.
pub(crate) fn mul(r: &mut [u64], a: &[u64], b: &[u64], n: u64, p: Plan) {
    let (la, lb, square) = (a.len(), b.len(), b.is_empty());
    let len = if square { 2 * la - 1 } else { la + lb - 1 };
    assert!(la >= 1 && r.len() == len && len <= 1 << p.k);
    convolve(r, a, b, n, p, 0);
}

/// The plan of a product modulo `x^(2^k) - 1` of operands of at most `2^k` coefficients, the
/// shorter of `m`, modulo a word of `bits` bits: as [`plan`], the sums of the wrapped
/// coefficients being sums of at most `m` products too.
pub(crate) fn plan_cyclic(m: usize, k: u32, bits: u32) -> Option<Plan> {
    if !(MIN_K..=MAX_K).contains(&k) {
        return None;
    }
    let need = 1.0 + f64::from(usize::BITS - m.leading_zeros()) + 2.0 * f64::from(bits) + 0.01;
    let np = (2..=8).find(|&np| primes_bits(np) > need)?;
    let bits = bits as usize;
    let dn = bits.div_ceil(28);
    Some(Plan {
        np,
        k,
        dn,
        b: bits.div_ceil(dn),
    })
}

/// Sets the `2^k` coefficients `r` to the product of `a` and `b` (of `a` and `a` when `b` is
/// empty) modulo `x^(2^k) - 1` and `n`, for operands of at most `2^k` coefficients and the plan
/// `p` of [`plan_cyclic`]: the transforms' cyclic convolution as it is, every coefficient past
/// the `2^k`-th wrapped around to the bottom.
///
/// # Panics
///
/// Panics if `r` does not have `2^k` coefficients, `a` is empty, or an operand is longer.
pub(crate) fn mul_cyclic(r: &mut [u64], a: &[u64], b: &[u64], n: u64, p: Plan) {
    let l = 1usize << p.k;
    assert!(!a.is_empty() && a.len() <= l && b.len() <= l && r.len() == l);
    convolve(r, a, b, n, p, 0);
}

/// Sets `r` to the coefficients `lo..lo + r.len()` of the product modulo `x^(2^k) - 1` of
/// [`mul_cyclic`], the Chinese remainder theorem taken for those alone (a middle product).
///
/// # Panics
///
/// Panics if `lo + r.len()` is more than `2^k`, `a` is empty, or an operand is longer.
pub(crate) fn mul_cyclic_range(r: &mut [u64], a: &[u64], b: &[u64], n: u64, p: Plan, lo: usize) {
    let l = 1usize << p.k;
    assert!(!a.is_empty() && a.len() <= l && b.len() <= l && lo + r.len() <= l);
    convolve(r, a, b, n, p, lo);
}

/// The convolution of [`mul`] and [`mul_cyclic`], its coefficients `lo..lo + r.len() <= 2^k`.
fn convolve(r: &mut [u64], a: &[u64], b: &[u64], n: u64, p: Plan, lo: usize) {
    let (la, lb) = (a.len(), b.len());
    let len = lo + r.len();
    let np = p.np;
    // The constants of the theorem modulo n, times 2^s, and of k.
    let s = n.leading_zeros();
    let dnorm = n << s;
    let v = invert_limb(dnorm);
    let pn = digits_mod(&product_digits(np, usize::MAX), n);
    let cp = ((n - pn) % n) << s;
    let cm: [u64; 8] = core::array::from_fn(|i| {
        if i < np {
            digits_mod(&product_digits(np, i), n) << s
        } else {
            0
        }
    });
    let ci: [u64; 8] = core::array::from_fn(|i| (1u64 << 60) / u64::from(PRIMES[i].0));
    let (dn, bb) = (p.dn, p.b);
    let mut buffer = Vec::new();
    let res = residues(
        &mut buffer,
        la,
        lb,
        len,
        p,
        // SAFETY: `digits` writes the `dn stride` digits the callee hands it room for.
        |d, stride| unsafe { digits(d, stride, a, dn, bb) },
        |d, stride| unsafe { digits(d, stride, b, dn, bb) },
    );
    let l = 1usize << p.k;
    if np == 2 {
        // P < 2^58: c is exact in a word, and reduced by Barrett's constant of n.
        let (q0, q1) = (u64::from(PRIMES[0].0), u64::from(PRIMES[1].0));
        let pp = q0 * q1;
        let bn = u64::MAX / n;
        let (r0, r1) = (&res[lo..len], &res[l + lo..l + len]);
        for ((out, &y0), &y1) in r.iter_mut().zip(r0).zip(r1) {
            let (y0, y1) = (y0 as u32 as u64, y1 as u32 as u64);
            let kk = (1u64 << 58) + y0 * ci[0] + y1 * ci[1];
            let c = (y0 * q1 + y1 * q0).wrapping_sub((kk >> 60) * pp);
            // floor(c bn / 2^64) is at most 2 below the quotient (c < 2^64, n >= 2).
            let qt = ((u128::from(c) * u128::from(bn)) >> 64) as u64;
            let mut x = c - qt * n;
            if x >= n {
                x -= n;
            }
            if x >= n {
                x -= n;
            }
            *out = x;
        }
        return;
    }
    let rows: [&[i32]; 8] = core::array::from_fn(|i| &res[i.min(np - 1) * l + lo..][..len - lo]);
    for (j, out) in r.iter_mut().enumerate() {
        let mut sum = 0u128;
        let mut kk = 1u64 << 58;
        for i in 0..np {
            let y = rows[i][j] as u32 as u64;
            sum += u128::from(y) * u128::from(cm[i]);
            kk += y * ci[i];
        }
        sum += u128::from(kk >> 60) * u128::from(cp);
        *out = div_2by1((sum >> 64) as u64, sum as u64, dnorm, v).1 >> s;
    }
}

/// The residues modulo each prime of the plan of the first `len` coefficients of the cyclic
/// convolution of two operands of `la` and `lb` coefficients (`lb` 0 for a square), normalized
/// to `[0, q)` and divided by the other primes' product (`y_i` of the Chinese remainder
/// theorem): `np` rows of `2^k` values in `buffer`, which is returned as a slice. The digits of
/// the operands are written by `da(d, stride)` and `db(d, stride)` (digit `t` of coefficient `j`
/// at `d[t stride + j]`, zeros up to `stride`), each below `2^b` in magnitude.
#[allow(clippy::too_many_arguments)]
fn residues(
    buffer: &mut Vec<i32>,
    la: usize,
    lb: usize,
    len: usize,
    p: Plan,
    da: impl FnOnce(*mut i32, usize),
    db: impl FnOnce(*mut i32, usize),
) -> &[i32] {
    let square = lb == 0;
    let (np, k, dn, b_bits) = (p.np, p.k, p.dn, p.b);
    let l = 1usize << k;
    debug_assert!(len <= l && la <= l && lb <= l && (MIN_K..=MAX_K).contains(&k));
    let (sa, sb) = (la.next_multiple_of(4), lb.next_multiple_of(4));
    let others = if square { 0 } else { l };
    buffer.clear();
    buffer.reserve((np + 2) * l + others + dn * (sa + sb));
    // SAFETY: the buffer holds the residues of the `np` primes, the twiddles, the transform of
    // `b` and the digits of both operands, each written before it is read; `MIN_K <= k <= MAX_K`
    // and the operands' coefficients fit their strides, multiples of 4. The rows of residues are
    // all written (by `finish`, the first `len` of each) before the slice is made of them.
    unsafe {
        let res = buffer.as_mut_ptr();
        let t = res.add(np * l);
        let bt = t.add(2 * l);
        let dpa = bt.add(others);
        let dpb = dpa.add(dn * sa);
        da(dpa, sa);
        if !square {
            db(dpb, sb);
        }
        for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
            let c = V::consts(&data.prime);
            twiddles::<V>(t, k, data);
            let r_t: [(i32, i32); 4] = core::array::from_fn(|t| data.pow2[t * b_bits]);
            let x = res.add(i * l);
            split::<V>(x, k, dpa, sa, la, dn, &r_t, t, c);
            forward::<V>(x, k, t, c);
            if square {
                pointwise::<V>(x, x, l, c);
            } else {
                split::<V>(bt, k, dpb, sb, lb, dn, &r_t, t, c);
                forward::<V>(bt, k, t, c);
                pointwise::<V>(x, bt, l, c);
            }
            inverse::<V>(x, k, t.add(l), c);
            // 2^32 L^-1 (P / q_i)^-1 mod q_i.
            let q = data.prime.q as u64;
            let rem = PRIMES[..np]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .fold(1u64, |acc, (_, &(qj, _))| acc * (u64::from(qj) % q) % q);
            let mu = pow_mod(rem, q - 2, q) * ((1u64 << 32) % q) % q;
            let mu = mu * (data.inv2k[k as usize] as i64).rem_euclid(q as i64) as u64 % q;
            let mu = centred(mu, q);
            finish::<V>(x, k, len, (mu, barrett(mu, q as u32)), t.add(l), c);
            // The values past `len` of the row were not written: zero them for the slice.
            for j in len..l {
                *x.add(j) = 0;
            }
        }
        core::slice::from_raw_parts(res, np * l)
    }
}

// Integer coefficients

/// The plan of a product of `la` and `lb` integer coefficients (`lb` 0 for a square) of at most
/// `coeff_bits <= 112` bits in magnitude, the product's coefficients below `2^bound_bits` in
/// magnitude: the fewest primes whose product exceeds four times that, so that the residues of
/// `c + floor(P/2)` give `c` exactly; `None` past the transforms or 8 primes.
pub(crate) fn plan_int(la: usize, lb: usize, coeff_bits: u32, bound_bits: u32) -> Option<Plan> {
    let len = if lb == 0 { 2 * la - 1 } else { la + lb - 1 };
    if len > MAX_LEN || la == 0 || coeff_bits == 0 || coeff_bits > 112 {
        return None;
    }
    let need = f64::from(bound_bits) + 2.01;
    let np = (2..=8).find(|&np| primes_bits(np) > need)?;
    let bits = coeff_bits as usize;
    let dn = bits.div_ceil(28);
    let k = (usize::BITS - (len - 1).leading_zeros()).max(MIN_K);
    Some(Plan {
        np,
        k,
        dn,
        b: bits.div_ceil(dn),
    })
}

/// Sets digit `t` of coefficient `j` of the signed `x` at `d[t stride + j]` (the digits of the
/// magnitude, negated for a negative coefficient), zeros up to `stride`.
///
/// # Safety
///
/// The `dn stride` digits at `d` must be valid for writes, `x.len() <= stride`.
unsafe fn signed_digits(d: *mut i32, stride: usize, x: &[i128], dn: usize, b: usize) {
    let mask = (1u128 << b) - 1;
    // SAFETY: every digit written lies below `dn stride`.
    unsafe {
        for t in 0..dn {
            let row = d.add(t * stride);
            for (j, &c) in x.iter().enumerate() {
                let m = ((c.unsigned_abs() >> (t * b)) & mask) as i32;
                *row.add(j) = if c < 0 { -m } else { m };
            }
            for j in x.len()..stride {
                *row.add(j) = 0;
            }
        }
    }
}

/// The product of the polynomials of the integer coefficients `a` and `b` (a square when `b` is
/// empty), each of at most 112 bits, by the plan of [`plan_int`]: the coefficients' magnitudes as
/// four limbs and their signs into `out`, of `la + lb - 1` (or `2 la - 1`) entries.
///
/// The residues `y_i` are shifted by those of `H = floor(P / 2)`, so that the theorem gives
/// `c + H` in `[0, P)`, below `3P / 4` as `|c| < P / 4`, where the estimate of `k` holds; `c` is
/// then that less `H`, in limbs.
pub(crate) fn mul_int(out: &mut [([u64; 4], bool)], a: &[i128], b: &[i128], p: Plan) {
    let (la, lb) = (a.len(), b.len());
    let len = out.len();
    let np = p.np;
    let (dn, bb) = (p.dn, p.b);
    // P, P / q_i and H = floor(P / 2) as four limbs.
    let limbs =
        |d: [u64; 8]| -> [u64; 4] { core::array::from_fn(|i| d[2 * i] | (d[2 * i + 1] << 32)) };
    let pp = limbs(product_digits(np, usize::MAX));
    let mi: [[u64; 4]; 8] = core::array::from_fn(|i| {
        if i < np {
            limbs(product_digits(np, i))
        } else {
            [0; 4]
        }
    });
    let h: [u64; 4] =
        core::array::from_fn(|i| (pp[i] >> 1) | (pp.get(i + 1).map_or(0, |x| x << 63)));
    // The residues of H divided by P / q_i: h_i = H (P / q_i)^-1 mod q_i.
    let hy: [u64; 8] = core::array::from_fn(|i| {
        if i >= np {
            return 0;
        }
        let q = u64::from(PRIMES[i].0);
        let hm = h
            .iter()
            .rev()
            .fold(0u128, |r, &l| ((r << 64) | u128::from(l)) % u128::from(q))
            as u64;
        let rem = PRIMES[..np]
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .fold(1u64, |acc, (_, &(qj, _))| acc * (u64::from(qj) % q) % q);
        hm * pow_mod(rem, q - 2, q) % q
    });
    let ci: [u64; 8] = core::array::from_fn(|i| (1u64 << 60) / u64::from(PRIMES[i].0));
    let mut buffer = Vec::new();
    let res = residues(
        &mut buffer,
        la,
        lb,
        len,
        p,
        // SAFETY: `signed_digits` writes the `dn stride` digits it is handed room for.
        |d, stride| unsafe { signed_digits(d, stride, a, dn, bb) },
        |d, stride| unsafe { signed_digits(d, stride, b, dn, bb) },
    );
    let l = 1usize << p.k;
    if np <= 4 {
        // P < 2^116: c + H, below P, is exact modulo 2^128.
        let w = |x: &[u64; 4]| u128::from(x[0]) | (u128::from(x[1]) << 64);
        let (p128, h128) = (w(&pp), w(&h));
        let m128: [u128; 4] = core::array::from_fn(|i| w(&mi[i]));
        let q: [u64; 4] = core::array::from_fn(|i| u64::from(PRIMES[i].0));
        for (j, o) in out.iter_mut().enumerate() {
            let mut acc = 0u128;
            let mut kk = 1u64 << 58;
            for i in 0..np {
                let mut y = res[i * l + j] as u32 as u64 + hy[i];
                if y >= q[i] {
                    y -= q[i];
                }
                kk += y * ci[i];
                acc = acc.wrapping_add(u128::from(y).wrapping_mul(m128[i]));
            }
            let v = acc
                .wrapping_sub(u128::from(kk >> 60).wrapping_mul(p128))
                .wrapping_sub(h128) as i128;
            let m = v.unsigned_abs();
            *o = ([m as u64, (m >> 64) as u64, 0, 0], v < 0);
        }
        return;
    }
    let crt = CrtInt {
        np,
        l,
        pp,
        h,
        mi,
        hy,
        ci,
    };
    if np <= 6 {
        crt.run::<3>(out, res);
    } else {
        crt.run::<4>(out, res);
    }
}

/// The constants of the theorem of [`mul_int`] for more than four primes.
struct CrtInt {
    np: usize,
    l: usize,
    pp: [u64; 4],
    h: [u64; 4],
    mi: [[u64; 4]; 8],
    hy: [u64; 8],
    ci: [u64; 8],
}

impl CrtInt {
    /// The coefficients from the residues `res` in `NL` limbs, enough for `P`: three for up to
    /// six primes (`P < 2^173`), four beyond.
    #[inline(always)]
    fn run<const NL: usize>(&self, out: &mut [([u64; 4], bool)], res: &[i32]) {
        let (np, l) = (self.np, self.l);
        for (j, o) in out.iter_mut().enumerate() {
            // c + H = Σ y'_i (P / q_i) - k P, modulo 2^(64 NL), exact as it is below P.
            let mut acc = [0u64; NL];
            let mut kk = 1u64 << 58;
            for i in 0..np {
                let q = u64::from(PRIMES[i].0);
                let mut y = res[i * l + j] as u32 as u64 + self.hy[i];
                if y >= q {
                    y -= q;
                }
                kk += y * self.ci[i];
                let mut carry = 0u64;
                for (a, &m) in acc.iter_mut().zip(&self.mi[i]) {
                    let v = u128::from(*a) + u128::from(y) * u128::from(m) + u128::from(carry);
                    *a = v as u64;
                    carry = (v >> 64) as u64;
                }
            }
            let kq = kk >> 60;
            // acc -= kq P + H.
            let mut borrow = 0u64;
            let mut c = [0u64; 4];
            for t in 0..NL {
                let sub = u128::from(kq) * u128::from(self.pp[t])
                    + u128::from(self.h[t])
                    + u128::from(borrow);
                let (v, under) = acc[t].overflowing_sub(sub as u64);
                c[t] = v;
                borrow = ((sub >> 64) as u64) + u64::from(under);
            }
            // The sign is the top bit of the NL limbs: c is negative, its magnitude 2^(64 NL) - c.
            if c[NL - 1] >> 63 == 1 {
                let mut carry = true;
                for v in c[..NL].iter_mut() {
                    let (s, o) = (!*v).overflowing_add(u64::from(carry));
                    *v = s;
                    carry = o;
                }
                *o = (c, true);
            } else {
                *o = (c, false);
            }
        }
    }
}

// Coefficients of several limbs, cut into pieces

/// How a product of polynomials of coefficients of several limbs is computed by
/// [`mul_split`]: each coefficient cut into `pieces` pieces of `wp <= 112` bits, each piece a
/// polynomial of its own, whose `2 pieces - 1` products are taken from one transform of each
/// piece modulo `np` primes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SplitPlan {
    /// The number of primes.
    pub(crate) np: usize,
    /// The transforms' `k`.
    pub(crate) k: u32,
    /// The pieces of a coefficient, and their bits.
    pub(crate) pieces: usize,
    wp: usize,
    /// The digits of a piece, and their bits.
    dn: usize,
    b: usize,
}

impl SplitPlan {
    /// The limbs of a coefficient of the product before its reduction: `Σ c_t 2^(t wp)`.
    pub(crate) fn out_limbs(&self) -> usize {
        // Each c_t has at most 2 wp + bits(m pieces) bits, below the primes' product.
        (self.wp * (2 * self.pieces - 2) + primes_bits(self.np) as usize + 2).div_ceil(64)
    }
}

/// The plan of least estimated cost of a product of `la` by `lb` coefficients (`lb` 0 for a
/// square) of at most `bits` bits each, if the transforms take it: the pieces from 1 to 16 and the
/// fewest primes for each, the cost the transforms (`pieces` of each operand forward, `2 pieces -
/// 1` back, each `k` levels) and the theorem's (about 6 a value and a prime).
pub(crate) fn plan_split(la: usize, lb: usize, bits: u32) -> Option<SplitPlan> {
    let (len, m) = if lb == 0 {
        (2 * la - 1, la)
    } else {
        (la + lb - 1, la.min(lb))
    };
    if len > MAX_LEN || la == 0 {
        return None;
    }
    let k = (usize::BITS - (len - 1).leading_zeros()).max(MIN_K);
    let bits = bits as usize;
    let mut best: Option<(SplitPlan, usize)> = None;
    for pieces in 1..=16usize {
        let wp = bits.div_ceil(pieces);
        if wp > 112 || (pieces > 1 && wp * (pieces - 1) >= bits) {
            continue;
        }
        let need = 1.0 + f64::from(usize::BITS - (m * pieces).leading_zeros()) + 2.0 * wp as f64;
        let Some(np) = (2..=8).find(|&np| primes_bits(np) > need + 0.01) else {
            continue;
        };
        let forward = if lb == 0 { pieces } else { 2 * pieces };
        let cost = np * ((forward + 2 * pieces - 1) * k as usize + 6 * (2 * pieces - 1));
        if best.is_none_or(|(_, c)| cost < c) {
            let dn = wp.div_ceil(28);
            let p = SplitPlan {
                np,
                k,
                pieces,
                wp,
                dn,
                b: wp.div_ceil(dn),
            };
            best = Some((p, cost));
        }
    }
    best.map(|(p, _)| p)
}

/// The `f <= 64` bits of the limbs `x` from bit `pos`, zeros past the end.
#[inline(always)]
fn bits_at(x: &[u64], pos: usize, f: usize) -> u64 {
    let (w, sh) = (pos / 64, pos % 64);
    let lo = x.get(w).copied().unwrap_or(0) >> sh;
    let hi = if sh != 0 && sh + f > 64 {
        x.get(w + 1).copied().unwrap_or(0) << (64 - sh)
    } else {
        0
    };
    let v = lo | hi;
    if f >= 64 { v } else { v & ((1u64 << f) - 1) }
}

/// Sets digit `t` of piece `piece` of coefficient `j` of the packed `x` (`kl` limbs a
/// coefficient) at `d[t stride + j]`, zeros up to `stride`.
///
/// # Safety
///
/// The `dn stride` digits at `d` must be valid for writes, `x.len() / kl <= stride`.
unsafe fn piece_digits(
    d: *mut i32,
    stride: usize,
    x: &[u64],
    kl: usize,
    piece: usize,
    p: &SplitPlan,
) {
    let nc = x.len() / kl;
    // SAFETY: every digit written lies below `dn stride`.
    unsafe {
        for t in 0..p.dn {
            let row = d.add(t * stride);
            let pos = piece * p.wp + t * p.b;
            let f = p.b.min((piece * p.wp + p.wp).saturating_sub(pos));
            for j in 0..nc {
                let c = &x[j * kl..(j + 1) * kl];
                *row.add(j) = if f == 0 { 0 } else { bits_at(c, pos, f) as i32 };
            }
            for j in nc..stride {
                *row.add(j) = 0;
            }
        }
    }
}

/// The product of the polynomials `a` and `b` of packed coefficients of `kl` limbs (`b` empty
/// for a square), by the plan `p` of [`plan_split`]: each coefficient `j` of the product, the
/// exact sum of products of the coefficients, into `out[j ow..(j + 1) ow]` (`ow` at least
/// [`SplitPlan::out_limbs`]), not reduced, for the first `out.len() / ow` coefficients.
///
/// The pieces `A_s` of `a` and `B_u` of `b` are transformed once a prime; the transform of each
/// product piece `C_t = Σ_(s+u=t) A_s B_u` is the sum of the pointwise products, reduced every
/// four (each below `0.95 q`, the sum of four within the lanes), and goes back by one inverse
/// transform. Each `c_t`, below half the primes' product, comes from its residues by the theorem
/// as in [`mul`], and the coefficient is `Σ c_t 2^(t wp)`.
///
/// # Panics
///
/// Panics if `out` is longer than the product or not a multiple of `ow`, or an operand is empty.
pub(crate) fn mul_split(
    out: &mut [u64],
    ow: usize,
    a: &[u64],
    b: &[u64],
    kl: usize,
    p: &SplitPlan,
) {
    let src = if b.is_empty() {
        Other::Square
    } else {
        Other::Operand(b)
    };
    split_core(out, ow, a, kl, src, p);
}

/// The pieces of an operand transformed by a plan of [`plan_split`], for products by it that
/// skip its transforms ([`mul_split_ready`]).
#[derive(Clone, Debug)]
pub(crate) struct SplitTransform {
    p: SplitPlan,
    /// The operand's coefficients.
    nb: usize,
    /// The transforms, piece after piece for each prime: `np pieces 2^k` values.
    t: Vec<i32>,
}

impl SplitTransform {
    /// The plan of the transforms.
    pub(crate) fn plan(&self) -> &SplitPlan {
        &self.p
    }
}

/// The operand `b` (packed, `kl` limbs a coefficient, not empty) transformed by the plan `p`,
/// for products by operands that the plan takes with it.
pub(crate) fn split_transform(b: &[u64], kl: usize, p: &SplitPlan) -> SplitTransform {
    let nb = b.len() / kl;
    let (np, k, pc) = (p.np, p.k, p.pieces);
    let l = 1usize << k;
    assert!(nb >= 1 && nb <= l);
    let sb = nb.next_multiple_of(4);
    let mut t: Vec<i32> = Vec::with_capacity(np * pc * l);
    let mut work: Vec<i32> = Vec::with_capacity(2 * l + p.dn * pc * sb);
    // SAFETY: `t` holds the np pc rows of l values, written by `split` and `forward`; `work`
    // the twiddles and the digits of the pieces.
    unsafe {
        let tw = work.as_mut_ptr();
        let db = tw.add(2 * l);
        for s in 0..pc {
            piece_digits(db.add(s * p.dn * sb), sb, b, kl, s, p);
        }
        for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
            let c = V::consts(&data.prime);
            twiddles::<V>(tw, k, data);
            let r_t: [(i32, i32); 4] = core::array::from_fn(|t| data.pow2[t * p.b]);
            for s in 0..pc {
                let x = t.as_mut_ptr().add((i * pc + s) * l);
                split::<V>(x, k, db.add(s * p.dn * sb), sb, nb, p.dn, &r_t, tw, c);
                forward::<V>(x, k, tw, c);
            }
        }
        t.set_len(np * pc * l);
    }
    SplitTransform { p: *p, nb, t }
}

/// [`mul_split`] of `a` by the operand transformed as `y`: the first `out.len() / ow`
/// coefficients of the product, which must have at most `2^k` coefficients, `a` at most as many
/// as the plan's shorter operand if `y`'s is longer.
pub(crate) fn mul_split_ready(
    out: &mut [u64],
    ow: usize,
    a: &[u64],
    kl: usize,
    y: &SplitTransform,
) {
    split_core(out, ow, a, kl, Other::Ready(y), &y.p);
}

/// The other operand of [`split_core`].
#[derive(Clone, Copy)]
enum Other<'a> {
    Square,
    Operand(&'a [u64]),
    Ready(&'a SplitTransform),
}

/// The products of [`mul_split`] and [`mul_split_ready`].
fn split_core(out: &mut [u64], ow: usize, a: &[u64], kl: usize, other: Other, p: &SplitPlan) {
    let la = a.len() / kl;
    let lb = match other {
        Other::Square => la,
        Other::Operand(b) => b.len() / kl,
        Other::Ready(y) => y.nb,
    };
    let len = la + lb - 1;
    let n_out = out.len() / ow;
    assert!(la >= 1 && out.len() == n_out * ow && n_out <= len && ow >= p.out_limbs());
    assert!(len <= 1 << p.k);
    let (np, k, pc) = (p.np, p.k, p.pieces);
    let l = 1usize << k;
    let nt = 2 * pc - 1;
    let sa = la.next_multiple_of(4);
    let (fb, sb) = match other {
        Other::Operand(_) => (pc, lb.next_multiple_of(4)),
        _ => (0, 0),
    };
    let fa = pc;
    // The residues of the products' pieces, nt rows for each prime; the transforms of the pieces
    // of a and b, the twiddles, and the digits of every piece.
    let mut res: Vec<i32> = Vec::with_capacity(np * nt * l);
    let mut work: Vec<i32> = Vec::with_capacity((fa + fb + 2) * l + p.dn * (fa * sa + fb * sb));
    // The constants of the theorem: P / q_i and P as four limbs.
    let limbs =
        |d: [u64; 8]| -> [u64; 4] { core::array::from_fn(|i| d[2 * i] | (d[2 * i + 1] << 32)) };
    let pp = limbs(product_digits(np, usize::MAX));
    let mi: [[u64; 4]; 8] = core::array::from_fn(|i| {
        if i < np {
            limbs(product_digits(np, i))
        } else {
            [0; 4]
        }
    });
    let ci: [u64; 8] = core::array::from_fn(|i| (1u64 << 60) / u64::from(PRIMES[i].0));
    // SAFETY: `work` holds the transforms of the pieces (fa + fb rows of l), the twiddles (2 l)
    // and the digits of the pieces (dn rows of the strides for each), `res` the np nt rows of l
    // values, each written before it is read; MIN_K <= k <= MAX_K and the coefficients fit their
    // strides, multiples of 4. Every row of `res` is written in full (by `finish` for the first
    // `n_out`, zeros after) before the slice is made. A ready transform has the plan's rows.
    unsafe {
        let wa = work.as_mut_ptr();
        let wb = wa.add(fa * l);
        let t = wb.add(fb * l);
        let da = t.add(2 * l);
        let db = da.add(p.dn * fa * sa);
        for s in 0..fa {
            piece_digits(da.add(s * p.dn * sa), sa, a, kl, s, p);
        }
        if let Other::Operand(b) = other {
            for s in 0..fb {
                piece_digits(db.add(s * p.dn * sb), sb, b, kl, s, p);
            }
        }
        let rp = res.as_mut_ptr();
        for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
            let c = V::consts(&data.prime);
            twiddles::<V>(t, k, data);
            let r_t: [(i32, i32); 4] = core::array::from_fn(|t| data.pow2[t * p.b]);
            for s in 0..fa {
                let x = wa.add(s * l);
                split::<V>(x, k, da.add(s * p.dn * sa), sa, la, p.dn, &r_t, t, c);
                forward::<V>(x, k, t, c);
            }
            for s in 0..fb {
                let x = wb.add(s * l);
                split::<V>(x, k, db.add(s * p.dn * sb), sb, lb, p.dn, &r_t, t, c);
                forward::<V>(x, k, t, c);
            }
            let bt: *const i32 = match other {
                Other::Square => wa,
                Other::Operand(_) => wb,
                Other::Ready(y) => y.t.as_ptr().add(i * pc * l),
            };
            // 2^32 L^-1 (P / q_i)^-1 mod q_i.
            let q = data.prime.q as u64;
            let rem = PRIMES[..np]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .fold(1u64, |acc, (_, &(qj, _))| acc * (u64::from(qj) % q) % q);
            let mu = pow_mod(rem, q - 2, q) * ((1u64 << 32) % q) % q;
            let mu = mu * (data.inv2k[k as usize] as i64).rem_euclid(q as i64) as u64 % q;
            let mu = centred(mu, q);
            for tt in 0..nt {
                let x = rp.add((i * nt + tt) * l);
                let lo = tt.saturating_sub(pc - 1);
                let hi = tt.min(pc - 1);
                for j in (0..l).step_by(4) {
                    let mut acc = V::splat(0);
                    for (cnt, s) in (lo..=hi).enumerate() {
                        let u = tt - s;
                        let prod = V::load(wa.add(s * l + j)).mont(V::load(bt.add(u * l + j)), c);
                        acc = acc.add(prod);
                        if cnt % 4 == 3 {
                            acc = acc.reduce(c);
                        }
                    }
                    acc.reduce(c).store(x.add(j));
                }
                inverse::<V>(x, k, t.add(l), c);
                finish::<V>(x, k, n_out, (mu, barrett(mu, q as u32)), t.add(l), c);
                for j in n_out..l {
                    *x.add(j) = 0;
                }
            }
        }
        res.set_len(np * nt * l);
    }
    // The theorem for each piece of each coefficient, added at its place: in three limbs for up
    // to six primes (P < 2^173), four beyond.
    out.fill(0);
    let th = PieceCrt {
        np,
        nt,
        l,
        pp,
        mi,
        ci,
        wp: p.wp,
    };
    if np <= 6 {
        th.run::<3>(out, ow, n_out, &res);
    } else {
        th.run::<4>(out, ow, n_out, &res);
    }
}

/// The constants of the theorem of [`split_core`].
struct PieceCrt {
    np: usize,
    nt: usize,
    l: usize,
    pp: [u64; 4],
    mi: [[u64; 4]; 8],
    ci: [u64; 8],
    wp: usize,
}

impl PieceCrt {
    /// Each `c_t` of each of the first `n_out` coefficients from its residues in `res`, in `NL`
    /// limbs (modulo `2^(64 NL)`, exact as `c_t < P / 2`), added at bit `t wp` of the coefficient.
    #[inline(always)]
    fn run<const NL: usize>(&self, out: &mut [u64], ow: usize, n_out: usize, res: &[i32]) {
        let (np, nt, l) = (self.np, self.nt, self.l);
        for tt in 0..nt {
            let shift = tt * self.wp;
            for j in 0..n_out {
                let mut acc = [0u64; NL];
                let mut kk = 1u64 << 58;
                for i in 0..np {
                    let y = res[(i * nt + tt) * l + j] as u32 as u64;
                    kk += y * self.ci[i];
                    let mut carry = 0u64;
                    for (tl, a) in acc.iter_mut().enumerate() {
                        let v = u128::from(*a)
                            + u128::from(y) * u128::from(self.mi[i][tl])
                            + u128::from(carry);
                        *a = v as u64;
                        carry = (v >> 64) as u64;
                    }
                }
                let kq = kk >> 60;
                let mut borrow = 0u64;
                for (tl, a) in acc.iter_mut().enumerate() {
                    let sub = u128::from(kq) * u128::from(self.pp[tl]) + u128::from(borrow);
                    let (v, under) = a.overflowing_sub(sub as u64);
                    *a = v;
                    borrow = ((sub >> 64) as u64) + u64::from(under);
                }
                add_shifted(&mut out[j * ow..(j + 1) * ow], &acc, shift);
            }
        }
    }
}

/// `r += x 2^shift`, the bits past `r` dropped (there are none for a coefficient's bound).
#[inline(always)]
fn add_shifted(r: &mut [u64], x: &[u64], shift: usize) {
    let (w, sh) = (shift / 64, (shift % 64) as u32);
    let mut carry = 0u64;
    let mut prev = 0u64;
    let n = r.len();
    for (i, &v) in x
        .iter()
        .enumerate()
        .chain(core::iter::once((x.len(), &0u64)))
    {
        let part = if sh == 0 {
            v
        } else {
            (v << sh) | (prev >> (64 - sh))
        };
        prev = v;
        let idx = w + i;
        if idx >= n {
            return;
        }
        let (s1, c1) = r[idx].overflowing_add(part);
        let (s2, c2) = s1.overflowing_add(carry);
        r[idx] = s2;
        carry = u64::from(c1) + u64::from(c2);
    }
    let mut idx = w + x.len() + 1;
    while carry != 0 && idx < n {
        let (s, c) = r[idx].overflowing_add(carry);
        r[idx] = s;
        carry = u64::from(c);
        idx += 1;
    }
}
