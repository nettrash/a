//! Horner's rule at many points at once modulo a word `n`, Montgomery's trick of inverses and the
//! sums of Lagrange's form, on the arithmetic of [`WordMod`].

use crate::math::numbers::nat::binvert;
pub(crate) use crate::math::poly::arith::nmod::WordMod;

/// `a·w mod n` for `a, w < n`, `wp` the [`WordMod::shoup`] of `w`: Shoup's product for
/// `n < 2^63`, reduced; the reciprocal above.
#[inline(always)]
pub(crate) fn mul_shoup_any(m: &WordMod, a: u64, w: u64, wp: u64) -> u64 {
    if m.n >= 1 << 63 {
        return m.mul(a, w);
    }
    let t = mul_shoup(a, w, wp, m.n);
    if t >= m.n { t - m.n } else { t }
}

/// `a·w mod n` by Shoup's method with `wp` [`WordMod::shoup`] of `w`, in `0..2n`: for `n < 2^63` and
/// any word `a`.
#[inline(always)]
fn mul_shoup(a: u64, w: u64, wp: u64, n: u64) -> u64 {
    let q = ((u128::from(a) * u128::from(wp)) >> 64) as u64;
    a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(n))
}

/// Montgomery's reduction modulo an odd word `n`: `redc(t) = t / 2^64 mod n` for `t < n 2^64`,
/// so that the product of `v` by `x' = x 2^64 mod n` reduces to `v·x mod n` itself, with no
/// residues to convert: three products a step, below `n` without a correction past the one
/// subtraction, for every odd `n` (the reciprocal's division takes about twice that).
#[derive(Clone, Copy)]
pub(crate) struct Redc {
    /// `n`, odd.
    n: u64,
    /// `1/n mod 2^64`.
    inv: u64,
}

impl Redc {
    /// The reduction modulo the odd `n`.
    #[inline]
    pub(crate) fn new(n: u64) -> Redc {
        debug_assert!(n & 1 == 1);
        Redc { n, inv: binvert(n) }
    }

    /// `x 2^64 mod n` for `x < n`, the factor that [`mul`](Redc::mul) takes.
    #[inline]
    pub(crate) fn factor(&self, m: &WordMod, x: u64) -> u64 {
        m.reduce2(x, 0)
    }

    /// `v·x mod n` for `v < n` and the [`factor`](Redc::factor) `xf` of `x`.
    #[inline(always)]
    pub(crate) fn mul(&self, v: u64, xf: u64) -> u64 {
        let t = u128::from(v) * u128::from(xf);
        let (hi, lo) = ((t >> 64) as u64, t as u64);
        let q = lo.wrapping_mul(self.inv);
        let p = ((u128::from(q) * u128::from(self.n)) >> 64) as u64;
        let (r, b) = hi.overflowing_sub(p);
        if b { r.wrapping_add(self.n) } else { r }
    }
}

/// How many points Horner's rule carries at once: independent chains for the core's multipliers.
const BLOCK: usize = 8;

/// Below this many steps or points, Horner's rule without the constants of each point.
const SHORT: usize = 4;

/// The values of the polynomial of the residues `c`, the constant first, at the residues `v`, in
/// place, by Horner's rule at `BLOCK` points at once, independent chains of products: for
/// `n < 2^30` on aarch64 32 points at once on NEON lanes; below `2^63` Shoup's products by each
/// point, the values kept below `3n` between the steps when `n < 2^62` and reduced at the end;
/// above, Montgomery's products (the reciprocal of `n` for an even `n`).
pub(crate) fn horner_many(m: &WordMod, c: &[u64], v: &mut [u64]) {
    let Some((&top, rest)) = c.split_last() else {
        v.fill(0);
        return;
    };
    if rest.is_empty() {
        v.fill(top);
        return;
    }
    if rest.len() < SHORT || v.len() < SHORT {
        // A few steps or points: the products by the reciprocal of n, without the constants of
        // each point.
        for x in v.iter_mut() {
            *x = rest.iter().rev().fold(top, |s, &a| m.add(m.mul(s, *x), a));
        }
        return;
    }
    let n = m.n;
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    let v = if n < 1 << 30 {
        let done = neon::horner(n as u32, top, rest, v);
        &mut v[done..]
    } else {
        v
    };
    let lazy = n < 1 << 62;
    let (vb, vr) = v.as_chunks_mut::<BLOCK>();
    for o in vb.iter_mut() {
        let x = *o;
        block(m, top, rest, &x, o, lazy);
    }
    if !vr.is_empty() {
        let mut xb = [0u64; BLOCK];
        let mut ob = [0u64; BLOCK];
        xb[..vr.len()].copy_from_slice(vr);
        block(m, top, rest, &xb, &mut ob, lazy);
        let len = vr.len();
        vr.copy_from_slice(&ob[..len]);
    }
}

/// Horner's rule at the `BLOCK` points `x`: by Shoup's products, `lazy` when `n < 2^62`; above,
/// Montgomery's products for an odd `n`, and for an even one Shoup's below `2^63` and the
/// reciprocal of `n` beyond.
#[inline(always)]
fn block(m: &WordMod, top: u64, rest: &[u64], x: &[u64; BLOCK], o: &mut [u64; BLOCK], lazy: bool) {
    let n = m.n;
    let mut v = [top; BLOCK];
    if !lazy && n & 1 == 1 {
        // Montgomery's products by x 2^64 mod n, for the odd n ≥ 2^62.
        let r = Redc::new(n);
        let xf: [u64; BLOCK] = core::array::from_fn(|j| r.factor(m, x[j]));
        for &a in rest.iter().rev() {
            for j in 0..BLOCK {
                v[j] = m.add(r.mul(v[j], xf[j]), a);
            }
        }
        *o = v;
        return;
    }
    if n >= 1 << 63 {
        for &a in rest.iter().rev() {
            for j in 0..BLOCK {
                v[j] = m.add(m.mul(v[j], x[j]), a);
            }
        }
        *o = v;
        return;
    }
    let xp: [u64; BLOCK] = core::array::from_fn(|j| m.shoup(x[j]));
    if lazy {
        // Each value below 3n < 2^64: the product in 0..2n, the coefficient below n.
        for &a in rest.iter().rev() {
            for j in 0..BLOCK {
                v[j] = mul_shoup(v[j], x[j], xp[j], n) + a;
            }
        }
        for j in 0..BLOCK {
            let mut t = v[j];
            if t >= n {
                t -= n;
            }
            if t >= n {
                t -= n;
            }
            o[j] = t;
        }
    } else {
        for &a in rest.iter().rev() {
            for j in 0..BLOCK {
                let mut t = mul_shoup(v[j], x[j], xp[j], n);
                if t >= n {
                    t -= n;
                }
                v[j] = m.add(t, a);
            }
        }
        *o = v;
    }
}

/// The inverses of the residues `a` modulo `n`, in place, by Montgomery's trick: one inverse and
/// three products each. `Err(i)` with the first `i` whose residue is not a unit (the slice then
/// holds garbage).
pub(crate) fn batch_inv(m: &WordMod, a: &mut [u64]) -> Result<(), usize> {
    if a.is_empty() {
        return Ok(());
    }
    let mut pre = Vec::with_capacity(a.len());
    let mut acc = 1u64;
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            return Err(i);
        }
        pre.push(acc);
        acc = m.mul(acc, x);
    }
    let Some(mut inv) = m.inv(acc) else {
        // Some residue shares a factor with n: find the first.
        let i = a.iter().position(|&x| m.inv(x).is_none()).unwrap_or(0);
        return Err(i);
    };
    for i in (0..a.len()).rev() {
        let x = a[i];
        a[i] = m.mul(inv, pre[i]);
        inv = m.mul(inv, x);
    }
    Ok(())
}

/// `out[j] += Σᵢ cᵢ·qᵢⱼ` for the quotients `qᵢ = M / (x - aᵢ)` of the monic `big` (`m + 1`
/// coefficients) by synthetic division, `q_{i,m-1} = 1`, `q_{i,j-1} = M_j + aᵢ·q_{i,j}`: the
/// partial sums of Horner's rule on `M` at `aᵢ`, `BLOCK` points at once, Shoup's products by each
/// `aᵢ` and `cᵢ`; `out` of `m` residues.
pub(crate) fn lagrange_sum(m: &WordMod, big: &[u64], a: &[u64], c: &[u64], out: &mut [u64]) {
    let len = out.len();
    debug_assert!(big.len() == len + 1 && a.len() == c.len());
    let n = m.n;
    let reduce = |t: u64| if t >= n { t - n } else { t };
    let (ab, ar) = a.as_chunks::<BLOCK>();
    let (cb, cr) = c.as_chunks::<BLOCK>();
    let mut pad_a = [0u64; BLOCK];
    let mut pad_c = [0u64; BLOCK];
    pad_a[..ar.len()].copy_from_slice(ar);
    pad_c[..cr.len()].copy_from_slice(cr);
    let tail = (!ar.is_empty()).then_some((&pad_a, &pad_c));
    for (x, w) in ab.iter().zip(cb.iter()).chain(tail) {
        if n >= 1 << 63 && n & 1 == 1 {
            // Montgomery's products, as Shoup's need n < 2^63.
            let r = Redc::new(n);
            let xf: [u64; BLOCK] = core::array::from_fn(|i| r.factor(m, x[i]));
            let wf: [u64; BLOCK] = core::array::from_fn(|i| r.factor(m, w[i]));
            let mut q = [1u64; BLOCK];
            for j in (0..len).rev() {
                let mut s = out[j];
                for i in 0..BLOCK {
                    s = m.add(s, r.mul(q[i], wf[i]));
                }
                out[j] = s;
                if j > 0 {
                    for i in 0..BLOCK {
                        q[i] = m.add(r.mul(q[i], xf[i]), big[j]);
                    }
                }
            }
            continue;
        }
        if n >= 1 << 63 {
            // The reciprocal of n, as Shoup's products need n < 2^63.
            let mut q = [1u64; BLOCK];
            for j in (0..len).rev() {
                let mut s = out[j];
                for i in 0..BLOCK {
                    s = m.add(s, m.mul(w[i], q[i]));
                }
                out[j] = s;
                if j > 0 {
                    for i in 0..BLOCK {
                        q[i] = m.add(m.mul(q[i], x[i]), big[j]);
                    }
                }
            }
            continue;
        }
        let xp: [u64; BLOCK] = core::array::from_fn(|i| m.shoup(x[i]));
        let wp: [u64; BLOCK] = core::array::from_fn(|i| m.shoup(w[i]));
        let mut q = [1u64; BLOCK];
        for j in (0..len).rev() {
            let mut s = out[j];
            for i in 0..BLOCK {
                s = reduce(s + reduce(mul_shoup(q[i], w[i], wp[i], n)));
            }
            out[j] = s;
            if j > 0 {
                let b = big[j];
                for i in 0..BLOCK {
                    q[i] = reduce(reduce(mul_shoup(q[i], x[i], xp[i], n)) + b);
                }
            }
        }
    }
}

/// Horner's rule at many points modulo `n < 2^30` with NEON: Shoup's products on 32-bit lanes, the
/// high halves of `v·x′` by `umull`/`umull2` and `uzp2`, `v·x − q·n` by `mul` and `mls`, the
/// values below `3n < 2^32` between the steps; 32 points at once, eight vectors of four.
#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon {
    use core::arch::aarch64::*;

    /// The points a block.
    const LANES: usize = 32;

    /// The values at the leading whole blocks of the points `xs`, in place: the number of
    /// points done.
    pub(super) fn horner(n: u32, top: u64, rest: &[u64], xs: &mut [u64]) -> usize {
        let blocks = xs.len() / LANES;
        let n64 = u64::from(n);
        for b in 0..blocks {
            let x: [u32; LANES] = core::array::from_fn(|i| xs[b * LANES + i] as u32);
            let xp: [u32; LANES] = core::array::from_fn(|i| ((u64::from(x[i]) << 32) / n64) as u32);
            let mut v = [0u32; LANES];
            // SAFETY: NEON is part of the aarch64 baseline; every load and store is of four
            // `u32`s inside the arrays of `LANES`, at offsets `4j`, `j < LANES / 4`.
            unsafe {
                let nv = vdupq_n_u32(n);
                let ld = |a: &[u32; LANES], j: usize| vld1q_u32(a.as_ptr().add(4 * j));
                let xv: [uint32x4_t; 8] = core::array::from_fn(|j| ld(&x, j));
                let pv: [uint32x4_t; 8] = core::array::from_fn(|j| ld(&xp, j));
                let mut acc = [vdupq_n_u32(top as u32); 8];
                for &a in rest.iter().rev() {
                    let c = vdupq_n_u32(a as u32);
                    for j in 0..8 {
                        let w = acc[j];
                        let lo = vmull_u32(vget_low_u32(w), vget_low_u32(pv[j]));
                        let hi = vmull_high_u32(w, pv[j]);
                        let q = vuzp2q_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi));
                        let r = vmlsq_u32(vmulq_u32(w, xv[j]), q, nv);
                        acc[j] = vaddq_u32(r, c);
                    }
                }
                for (j, a) in acc.iter().enumerate() {
                    let mut t = *a;
                    t = vminq_u32(t, vsubq_u32(t, nv));
                    t = vminq_u32(t, vsubq_u32(t, nv));
                    vst1q_u32(v.as_mut_ptr().add(4 * j), t);
                }
            }
            for i in 0..LANES {
                xs[b * LANES + i] = u64::from(v[i]);
            }
        }
        blocks * LANES
    }
}
