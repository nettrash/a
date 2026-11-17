//! Residues modulo an odd `n` of 2 to [`MAX_LIMBS`] limbs by Montgomery's products in registers
//! ([`Mont`]), with no residue converted: the product of `v` by `x' = x R mod n` (`R =
//! B^k`) is `v x R / R = v x mod n` itself, so Horner's rule at a point takes one Montgomery
//! product a step once `x'` is made, and a product of two residues takes two (the second by
//! `R² mod n`). Four points go at once, independent chains for the multipliers.

use crate::math::numbers::nat::Mont;
use crate::math::poly::modular::{Modulus, Work};

/// The most limbs of a modulus taken here.
pub(crate) const MAX_LIMBS: usize = 16;

/// How many points Horner's rule carries at once.
const LANES: usize = 4;

/// Montgomery's products of each number of limbs, made once.
#[derive(Clone, Copy)]
enum Any {
    L2(Mont<2>),
    L3(Mont<3>),
    L4(Mont<4>),
    L5(Mont<5>),
    L6(Mont<6>),
    L7(Mont<7>),
    L8(Mont<8>),
    L9(Mont<9>),
    L10(Mont<10>),
    L11(Mont<11>),
    L12(Mont<12>),
    L13(Mont<13>),
    L14(Mont<14>),
    L15(Mont<15>),
    L16(Mont<16>),
}

/// Montgomery's arithmetic modulo an odd `n` of `k` limbs, `2 ≤ k ≤ MAX_LIMBS`.
pub(crate) struct MontCtx {
    /// The products.
    any: Any,
    /// `n`.
    d: Vec<u64>,
    /// `R² mod n`.
    r2: Vec<u64>,
}

/// Calls `$f::<N>(m, $args)` with the products `m` of `N` limbs that `$any` holds.
macro_rules! by_limbs {
    ($any:expr, $f:ident, $($args:expr),*) => {
        match $any {
            Any::L2(m) => $f::<2>(m, $($args),*),
            Any::L3(m) => $f::<3>(m, $($args),*),
            Any::L4(m) => $f::<4>(m, $($args),*),
            Any::L5(m) => $f::<5>(m, $($args),*),
            Any::L6(m) => $f::<6>(m, $($args),*),
            Any::L7(m) => $f::<7>(m, $($args),*),
            Any::L8(m) => $f::<8>(m, $($args),*),
            Any::L9(m) => $f::<9>(m, $($args),*),
            Any::L10(m) => $f::<10>(m, $($args),*),
            Any::L11(m) => $f::<11>(m, $($args),*),
            Any::L12(m) => $f::<12>(m, $($args),*),
            Any::L13(m) => $f::<13>(m, $($args),*),
            Any::L14(m) => $f::<14>(m, $($args),*),
            Any::L15(m) => $f::<15>(m, $($args),*),
            Any::L16(m) => $f::<16>(m, $($args),*),
        }
    };
}

impl MontCtx {
    /// The arithmetic modulo `m`, if `n` is odd and of 2 to [`MAX_LIMBS`] limbs.
    pub(crate) fn new(m: &Modulus) -> Option<MontCtx> {
        let k = m.k();
        if !(2..=MAX_LIMBS).contains(&k) || m.d[0] & 1 == 0 {
            return None;
        }
        // R² = B^(2k).
        let mut x = vec![0u64; 2 * k + 1];
        x[2 * k] = 1;
        let mut r2 = vec![0u64; k];
        m.reduce(&mut r2, &x, &mut Work::default());
        let any = match k {
            2 => Any::L2(Mont::new(arr(&m.d))),
            3 => Any::L3(Mont::new(arr(&m.d))),
            4 => Any::L4(Mont::new(arr(&m.d))),
            5 => Any::L5(Mont::new(arr(&m.d))),
            6 => Any::L6(Mont::new(arr(&m.d))),
            7 => Any::L7(Mont::new(arr(&m.d))),
            8 => Any::L8(Mont::new(arr(&m.d))),
            9 => Any::L9(Mont::new(arr(&m.d))),
            10 => Any::L10(Mont::new(arr(&m.d))),
            11 => Any::L11(Mont::new(arr(&m.d))),
            12 => Any::L12(Mont::new(arr(&m.d))),
            13 => Any::L13(Mont::new(arr(&m.d))),
            14 => Any::L14(Mont::new(arr(&m.d))),
            15 => Any::L15(Mont::new(arr(&m.d))),
            16 => Any::L16(Mont::new(arr(&m.d))),
            _ => unreachable!("2 to 16 limbs"),
        };
        Some(MontCtx {
            any,
            d: m.d.clone(),
            r2,
        })
    }

    /// `out = a·b mod n`, residues of `k` limbs.
    #[inline]
    pub(crate) fn mul(&self, out: &mut [u64], a: &[u64], b: &[u64]) {
        by_limbs!(&self.any, mul_n, &self.r2, out, a, b);
    }

    /// `x R mod n`, the factor that [`mul_factor`](MontCtx::mul_factor) takes for `x`.
    pub(crate) fn factor(&self, out: &mut [u64], x: &[u64]) {
        by_limbs!(&self.any, mul_one, out, x, &self.r2);
    }

    /// `out = a·x mod n` for the [`factor`](MontCtx::factor) `xf` of `x`: one product.
    #[inline]
    pub(crate) fn mul_factor(&self, out: &mut [u64], a: &[u64], xf: &[u64]) {
        by_limbs!(&self.any, mul_one, out, a, xf);
    }

    /// The values of the packed polynomial `c` (at least one coefficient) at the packed points
    /// `xs`, into `out`, by Horner's rule.
    pub(crate) fn horner_many(&self, c: &[u64], xs: &[u64], out: &mut [u64]) {
        by_limbs!(&self.any, horner_n, &self.d, &self.r2, c, xs, out);
    }
}

/// The `N` limbs `x` as an array.
#[inline(always)]
fn arr<const N: usize>(x: &[u64]) -> [u64; N] {
    let mut a = [0u64; N];
    a.copy_from_slice(&x[..N]);
    a
}

/// `a + b mod n` for `a, b < n`.
#[inline(always)]
fn add_mod<const N: usize>(a: &[u64; N], b: &[u64], n: &[u64; N]) -> [u64; N] {
    let mut s = [0u64; N];
    let mut c = false;
    for i in 0..N {
        let (t, c1) = a[i].overflowing_add(b[i]);
        let (t, c2) = t.overflowing_add(u64::from(c));
        s[i] = t;
        c = c1 | c2;
    }
    let mut d = [0u64; N];
    let mut br = false;
    for i in 0..N {
        let (t, b1) = s[i].overflowing_sub(n[i]);
        let (t, b2) = t.overflowing_sub(u64::from(br));
        d[i] = t;
        br = b1 | b2;
    }
    // s - n when the sum carried out or is at least n.
    if c || !br { d } else { s }
}

/// `a·b mod n`: `(a b / R) R² / R`.
#[inline(always)]
fn mul_n<const N: usize>(m: &Mont<N>, r2: &[u64], out: &mut [u64], a: &[u64], b: &[u64]) {
    let t = m.mul(arr(a), arr(b));
    out[..N].copy_from_slice(&m.mul(t, arr(r2)));
}

/// `a b / R mod n`.
#[inline(always)]
fn mul_one<const N: usize>(m: &Mont<N>, out: &mut [u64], a: &[u64], b: &[u64]) {
    out[..N].copy_from_slice(&m.mul(arr(a), arr(b)));
}

/// Horner's rule at the points `xs` of the coefficients `c`, [`LANES`] points at once.
#[inline(never)]
fn horner_n<const N: usize>(
    m: &Mont<N>,
    d: &[u64],
    r2: &[u64],
    c: &[u64],
    xs: &[u64],
    out: &mut [u64],
) {
    let n: [u64; N] = arr(d);
    let r2: [u64; N] = arr(r2);
    let len = c.len() / N;
    let top: [u64; N] = arr(&c[(len - 1) * N..]);
    let npts = xs.len() / N;
    let mut i = 0;
    while i < npts {
        let lanes = LANES.min(npts - i);
        // x R mod n of each point, the padding lanes 0.
        let xf: [[u64; N]; LANES] = core::array::from_fn(|j| {
            if j < lanes {
                m.mul(arr(&xs[(i + j) * N..]), r2)
            } else {
                [0u64; N]
            }
        });
        let mut v = [top; LANES];
        for t in (0..len - 1).rev() {
            let a = &c[t * N..(t + 1) * N];
            for j in 0..LANES {
                v[j] = add_mod(&m.mul(v[j], xf[j]), a, &n);
            }
        }
        for j in 0..lanes {
            out[(i + j) * N..(i + j + 1) * N].copy_from_slice(&v[j]);
        }
        i += lanes;
    }
}
