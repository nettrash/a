//! The portable lanes of the transforms, for Miri and the targets without NEON: four `i32`s,
//! each operation computing what the NEON instruction of the aarch64 lanes computes, so that both
//! give the same values, and the Chinese remainder theorem of four coefficients in 64-bit
//! integers.
//!
//! Measured on an Apple M4 Max, where the compiler vectorizes some of it, a product by this code
//! took 1.5 to 1.8 times as long as GMP 6.3's from 1024 limbs, and about four times as long as by
//! the NEON lanes.

use super::{Crt, Lanes, Prime};

/// From this many limbs in the shorter operand, a product is computed by the transforms; under
/// Miri, from 250, past the products that test Toom's methods.
pub(in super::super) const FFT_MUL: usize = if cfg!(miri) { 250 } else { 16_384 };

/// From this many limbs in the shorter operand, a product whose longer operand has at least
/// twice as many is computed by the transforms.
pub(in super::super) const FFT_MUL_UNBALANCED: usize = FFT_MUL / 2;

/// From this many limbs, a square is computed by the transforms.
pub(in super::super) const FFT_SQR: usize = FFT_MUL;

/// The lanes of the transforms on this target.
pub(super) type V = Port;

/// Four residues.
#[derive(Clone, Copy)]
pub(super) struct Port([i32; 4]);

/// The constants of a prime.
#[derive(Clone, Copy)]
pub(super) struct Constants {
    q: i32,
    v: i32,
    qinv: i32,
}

/// `a w - round(a wp / 2^31) q`, modulo `2^32`: `sqrdmulh`, `mul` and `mls`.
#[inline(always)]
fn mul(a: i32, w: i32, wp: i32, q: i32) -> i32 {
    let t = ((i64::from(a) * i64::from(wp) + (1 << 30)) >> 31) as i32;
    a.wrapping_mul(w).wrapping_sub(t.wrapping_mul(q))
}

/// `a - round(floor(a v / 2^31) / 2^28) q`: `sqdmulh`, `srshr` and `mls`.
#[inline(always)]
fn reduce(a: i32, v: i32, q: i32) -> i32 {
    let t = (i64::from(a) * i64::from(v)) >> 31;
    let t = ((t + (1 << 27)) >> 28) as i32;
    a.wrapping_sub(t.wrapping_mul(q))
}

/// `(floor(2 a b / 2^32) - floor(2 m q / 2^32)) / 2` with `m = a b qinv mod 2^32`: the
/// Montgomery product, `mul`, `sqdmulh`, `mul`, `sqdmulh` and `shsub`.
#[inline(always)]
fn mont(a: i32, b: i32, q: i32, qinv: i32) -> i32 {
    let hi = (i64::from(a) * i64::from(b)) >> 31;
    let m = a.wrapping_mul(b).wrapping_mul(qinv);
    let t = (i64::from(m) * i64::from(q)) >> 31;
    ((hi - t) >> 1) as i32
}

impl Lanes for Port {
    type C = Constants;

    #[inline(always)]
    fn consts(p: &Prime) -> Constants {
        Constants {
            q: p.q,
            v: p.v,
            qinv: p.qinv,
        }
    }

    #[inline(always)]
    unsafe fn load(p: *const i32) -> Self {
        // SAFETY: as the caller guarantees.
        Port(core::array::from_fn(|i| unsafe { *p.add(i) }))
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut i32) {
        for (i, &v) in self.0.iter().enumerate() {
            // SAFETY: as the caller guarantees.
            unsafe { *p.add(i) = v };
        }
    }

    #[inline(always)]
    unsafe fn load2(p: *const i32) -> (Self, Self) {
        // SAFETY: as the caller guarantees.
        let at = |i: usize| unsafe { *p.add(i) };
        (
            Port(core::array::from_fn(|i| at(2 * i))),
            Port(core::array::from_fn(|i| at(2 * i + 1))),
        )
    }

    #[inline(always)]
    unsafe fn load4(p: *const i32) -> (Self, Self, Self, Self) {
        // SAFETY: as the caller guarantees.
        let at = |i: usize| unsafe { *p.add(i) };
        let lane = |o: usize| Port(core::array::from_fn(|i| at(4 * i + o)));
        (lane(0), lane(1), lane(2), lane(3))
    }

    #[inline(always)]
    unsafe fn store2(a: Self, b: Self, p: *mut i32) {
        for i in 0..4 {
            // SAFETY: as the caller guarantees.
            unsafe {
                *p.add(2 * i) = a.0[i];
                *p.add(2 * i + 1) = b.0[i];
            }
        }
    }

    #[inline(always)]
    unsafe fn pair(p: *const i32) -> Self {
        // SAFETY: as the caller guarantees.
        unsafe { Port([*p, *p.add(1), 0, 0]) }
    }

    #[inline(always)]
    fn splat(x: i32) -> Self {
        Port([x; 4])
    }

    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Port(core::array::from_fn(|i| self.0[i].wrapping_add(o.0[i])))
    }

    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Port(core::array::from_fn(|i| self.0[i].wrapping_sub(o.0[i])))
    }

    #[inline(always)]
    fn mul(self, w: Self, wp: Self, c: Constants) -> Self {
        Port(core::array::from_fn(|i| {
            mul(self.0[i], w.0[i], wp.0[i], c.q)
        }))
    }

    #[inline(always)]
    fn mul_lanes<const LW: i32, const LP: i32>(self, v: Self, c: Constants) -> Self {
        let (w, wp) = (v.0[LW as usize], v.0[LP as usize]);
        Port(self.0.map(|a| mul(a, w, wp, c.q)))
    }

    #[inline(always)]
    fn reduce(self, c: Constants) -> Self {
        Port(self.0.map(|a| reduce(a, c.v, c.q)))
    }

    #[inline(always)]
    fn mont(self, o: Self, c: Constants) -> Self {
        Port(core::array::from_fn(|i| {
            mont(self.0[i], o.0[i], c.q, c.qinv)
        }))
    }

    #[inline(always)]
    fn transpose(v: [Self; 4]) -> [Self; 4] {
        core::array::from_fn(|e| Port(core::array::from_fn(|c| v[c].0[e])))
    }

    #[inline(always)]
    fn normalize(self, c: Constants) -> Self {
        Port(self.0.map(|a| a + (c.q & (a >> 31))))
    }

    #[inline(always)]
    fn constants(self, f: f64) -> Self {
        Port(self.0.map(|w| (f64::from(w) * f).round_ties_even() as i32))
    }

    #[inline(always)]
    fn rev(self) -> Self {
        let [a, b, c, d] = self.0;
        Port([d, c, b, a])
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Port(self.0.map(i32::wrapping_neg))
    }
}

/// Sets `out[e][o..o + 4]` to limb `e` of the coefficients `j..j + 4` of the product, from their
/// residues in `[0, q_i)` at `res[i] + j`, as the aarch64 kernel does: `ND` digits of 32 bits,
/// each accumulated in a `u64`, the carries moved up, pairs of digits made limbs. The sums wrap
/// as the vector lanes do: the coefficients past the product's last, whose residues are not
/// scaled and may be any values, are computed and then dropped by the caller.
///
/// # Safety
///
/// The residues `j..j + 4` of each prime must be valid for reads.
#[inline(always)]
pub(super) unsafe fn four<const ND: usize>(
    res: &[*const i32; 8],
    j: usize,
    c: &Crt,
    out: &mut [[u64; 8]; 4],
    o: usize,
) {
    for lane in 0..4 {
        let mut acc = [0u64; 8];
        let mut f = 0u64;
        for (i, r) in res.iter().enumerate().take(c.np) {
            // SAFETY: as the caller guarantees.
            let y = u64::from(unsafe { *r.add(j + lane) } as u32);
            f = f.wrapping_add(y * u64::from(c.ci[i]));
            for (a, &m) in acc.iter_mut().zip(&c.m[i]).take(ND) {
                *a = a.wrapping_add(y * u64::from(m));
            }
        }
        let k = f.wrapping_add(1 << 58) >> 60;
        for (a, &p) in acc.iter_mut().zip(&c.pneg).take(ND) {
            *a = a.wrapping_add(k * u64::from(p));
        }
        for d in 0..ND {
            if d + 1 < ND {
                acc[d + 1] = acc[d + 1].wrapping_add(acc[d] >> 32);
            }
            acc[d] &= 0xFFFF_FFFF;
        }
        for e in 0..ND.div_ceil(2) {
            let high = if 2 * e + 1 < ND { acc[2 * e + 1] } else { 0 };
            out[e][o + lane] = acc[2 * e] | high << 32;
        }
    }
}
