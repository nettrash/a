//! The sine and the cosine at short working precisions, up to [`MAX_LIMBS`] limbs after the point,
//! on fixed-point numbers held in arrays, without allocations: the argument split as `r = a + ρ`,
//! `a = j/64` from a table of `sin a` and `cos a`, `0 <= ρ < 1/64`, the Taylor series of `ρ` by
//! Horner's rule with the coefficients `1/k!` of a table, and the addition formulas.

use super::limbs;
use super::sincos::{Fx, Lg};
use std::sync::OnceLock;

/// The most limbs after the point of the short path: 512 bits.
pub(super) const MAX_LIMBS: usize = 8;

/// The limbs of the tables after the point: 576 bits, so that the values cut to 512 bits are
/// within one ulp.
const TABLE_LIMBS: usize = 9;

/// The limbs of an array: the table's after the point and one before it.
const N: usize = TABLE_LIMBS + 1;

/// A fixed-point number of the tables: `TABLE_LIMBS` limbs after the point and one before it, the
/// least significant first.
type W = [u64; N];

/// The coefficients `1/k!` of the series: `floor(2^576/k!)`, the terms of the tables' series.
const TERMS: usize = 160;

/// `floor(2^576/k!)` for `k < TERMS`, computed when the crate is compiled.
static INV_FACT: [W; TERMS] = inv_factorials();

const fn inv_factorials() -> [W; TERMS] {
    let mut t = [[0u64; N]; TERMS];
    let mut v = [0u64; N];
    v[TABLE_LIMBS] = 1;
    t[0] = v;
    let mut k = 1;
    while k < TERMS {
        // `v = floor(v / k)`, which is `floor(2^576/k!)` as floors compose.
        let mut r: u128 = 0;
        let mut i = N;
        while i > 0 {
            i -= 1;
            let cur = (r << 64) | v[i] as u128;
            v[i] = (cur / k as u128) as u64;
            r = cur % k as u128;
        }
        t[k] = v;
        k += 1;
    }
    t
}

/// `floor(a·b·2^(-64(L - 1)))` for numbers of `L - 1` limbs after the point and one before it,
/// whose product is below `2^64`: the schoolbook product, limbs `L - 1 .. 2L - 1` of it.
#[inline(always)]
fn mul<const L: usize>(a: &[u64; L], b: &[u64; L]) -> [u64; L] {
    let (mut lo, mut hi) = ([0u64; L], [0u64; L]);
    for i in 0..L {
        let mut carry = 0u64;
        for (j, &bj) in b.iter().enumerate() {
            let k = i + j;
            let slot = if k < L { &mut lo[k] } else { &mut hi[k - L] };
            let t = u128::from(a[i]) * u128::from(bj) + u128::from(*slot) + u128::from(carry);
            *slot = t as u64;
            carry = (t >> 64) as u64;
        }
        // Limb `i + L`, which the rows before did not reach.
        hi[i] = carry;
    }
    debug_assert!(hi[L - 1] == 0);
    let mut r = [0u64; L];
    r[0] = lo[L - 1];
    r[1..].copy_from_slice(&hi[..L - 1]);
    r
}

/// `a + b`.
#[inline(always)]
fn add<const L: usize>(a: &[u64; L], b: &[u64; L]) -> [u64; L] {
    let mut r = [0u64; L];
    let mut carry = false;
    for i in 0..L {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(u64::from(carry));
        r[i] = s;
        carry = c1 || c2;
    }
    r
}

/// `a - b` for `a >= b`.
#[inline(always)]
fn sub<const L: usize>(a: &[u64; L], b: &[u64; L]) -> [u64; L] {
    let mut r = [0u64; L];
    let mut borrow = false;
    for i in 0..L {
        let (s, b1) = a[i].overflowing_sub(b[i]);
        let (s, b2) = s.overflowing_sub(u64::from(borrow));
        r[i] = s;
        borrow = b1 || b2;
    }
    debug_assert!(!borrow);
    r
}

/// A value of the tables cut to `L - 1` limbs after the point, within one ulp below, as
/// `floor(floor(x)/2^d) = floor(x/2^d)`.
#[inline(always)]
fn cut<const L: usize>(v: &W) -> [u64; L] {
    let mut r = [0u64; L];
    r.copy_from_slice(&v[N - L..]);
    r
}

/// The number of bits.
#[inline]
fn bits<const L: usize>(a: &[u64; L]) -> u64 {
    for i in (0..L).rev() {
        if a[i] != 0 {
            return 64 * i as u64 + u64::from(64 - a[i].leading_zeros());
        }
    }
    0
}

/// `Σ_(n < T) (-1)^n y^n / (2n + δ)!` by Horner's rule, `h ← c_n - y·h`, for `y = Y·2^-F < 1`
/// (`F = 64(L - 1)`), `δ` 0 or 1, with `T` the terms that leave a tail below half an ulp.
///
/// Every `h` lies in `[0, c_n]` (`y·c_(n+1) <= c_n/2`, and the floors keep the computed values
/// below the exact ones). The error of a step is at most that of `c_n` (one ulp), the product's
/// floor (one) and `e_y·h <= e_y/2` for an error `e_y` of `y`, plus `y` times the error before:
/// below `(2 + e_y/2)/(1 - y)` ulps; with the tail, 3.1 ulps for `y < 2^-12` and `e_y <= 1.04`,
/// and at most `2T + 1` for `y <= 1` exact (the tables).
fn horner<const L: usize>(y: &[u64; L], delta: usize) -> [u64; L] {
    let f = 64 * (L as u64 - 1);
    // `y < 2^(yb - F)`: the term `T` is below `2^(-T·gap) / (2T + δ)!`; `lf` a lower bound of
    // `log2 (2T + δ)!`.
    let gap = f.saturating_sub(bits(y));
    let mut t = 1usize;
    let mut lf: u64 = (2..=(2 + delta) as u64).map(log2_floor).sum();
    while t as u64 * gap + lf <= f + 1 {
        t += 1;
        let k = (2 * t + delta) as u64;
        lf += log2_floor(k - 1) + log2_floor(k);
    }
    assert!(
        2 * t + delta < TERMS,
        "the table of coefficients is long enough"
    );
    let mut h = cut::<L>(&INV_FACT[2 * (t - 1) + delta]);
    for n in (0..t - 1).rev() {
        let p = mul(y, &h);
        h = sub(&cut::<L>(&INV_FACT[2 * n + delta]), &p);
    }
    h
}

/// `floor(log2 t)` for `t >= 1`.
#[inline]
fn log2_floor(t: u64) -> u64 {
    u64::from(63 - t.leading_zeros())
}

/// `sin(j/64)` and `cos(j/64)` for `j <= 64` at 576 bits after the point, within `2T + 2 < 160`
/// ulps there, so below `1 + 2^-57` ulps at 512 bits or fewer: the series of `a = j/64`, `y = a²
/// <= 1` exact, `sin a = a·h₁(y)`, `cos a = h₀(y)`.
fn table() -> &'static [(W, W)] {
    static TABLE: OnceLock<Vec<(W, W)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=64u64)
            .map(|j| {
                let mut a = [0u64; N];
                // `j/64 = j·2^570·2^-576`: limb 8 from bit 58, and limb 9 for `j = 64`.
                a[TABLE_LIMBS - 1] = j << 58;
                a[TABLE_LIMBS] = j >> 6;
                let y = mul(&a, &a);
                (mul(&a, &horner(&y, 1)), horner(&y, 0))
            })
            .collect()
    })
}

/// The limbs of an array, without zeros at the top.
fn to_vec<const L: usize>(a: &[u64; L]) -> Vec<u64> {
    let mut v = a.to_vec();
    limbs::trim(&mut v);
    v
}

/// The sine and the cosine of `r = R·2^-g`, `0 < |r| <= 1` (of which the sign plays no part), at
/// `F = 64·fl` bits after the point, `1 <= fl <= MAX_LIMBS`, each when asked for: as
/// [`sin_cos`](super::sincos::sin_cos) gives them.
///
/// `r` is cut to `F` bits after the point (an error of an ulp), `j = floor(64r)` and `ρ = r -
/// j/64`, `y = ρ²` within 1.04 ulps (`2ρ + 1`). For `j = 0`, `sin r = r·h₁(y)` with the uncut `r`
/// (to `F + 64` bits), relatively within `4` ulps of `h₁` and one of `y`, and `cos r = h₀(y)`
/// within 5. Otherwise `sin r = sin a·h₀ + cos a·(ρ h₁)` and `cos r = cos a·h₀ - sin a·(ρ h₁)`:
/// the table's two ulps, `h`'s four, `ρ h₁`'s two (its floor and `ρ`'s), the four floors and the
/// ulp of `r` through `y` give at most `2·4 + 8 = 16` ulps each.
pub(super) fn sin_cos(
    r: &[u64],
    g: i128,
    fl: usize,
    want_sin: bool,
    want_cos: bool,
) -> (Option<Fx>, Option<Fx>) {
    match fl {
        1 => run::<2>(r, g, want_sin, want_cos),
        2 => run::<3>(r, g, want_sin, want_cos),
        3 => run::<4>(r, g, want_sin, want_cos),
        4 => run::<5>(r, g, want_sin, want_cos),
        5 => run::<6>(r, g, want_sin, want_cos),
        6 => run::<7>(r, g, want_sin, want_cos),
        7 => run::<8>(r, g, want_sin, want_cos),
        8 => run::<9>(r, g, want_sin, want_cos),
        _ => unreachable!("a short precision"),
    }
}

/// [`sin_cos`] at `L - 1` limbs after the point.
fn run<const L: usize>(
    r: &[u64],
    g: i128,
    want_sin: bool,
    want_cos: bool,
) -> (Option<Fx>, Option<Fx>) {
    let fl = L - 1;
    let f = 64 * fl as u64;
    let fi = i128::from(f);
    // `r` at `F` bits after the point.
    let s = fi - g;
    let rb = limbs::bit_len(r);
    let mut rf = [0u64; L];
    let src = if s >= 0 {
        limbs::shl(r, s as u64)
    } else {
        limbs::shr(r, s.unsigned_abs().min(u128::from(rb)) as u64)
    };
    debug_assert!(src.len() <= L);
    rf[..src.len()].copy_from_slice(&src);
    let j = (rf[fl] << 6 | rf[fl - 1] >> 58) as usize;
    if j == 0 {
        let y = mul(&rf, &rf);
        let sin = want_sin.then(|| {
            let h = horner(&y, 1);
            // `r` itself, cut to `F + 64` bits: one unit of its last place more, `2^F` units of
            // the product's.
            let (rr, gg, cut) = if rb > f + 64 {
                let d = rb - (f + 64);
                (limbs::shr(r, d), g - i128::from(d), true)
            } else {
                (r.to_vec(), g, false)
            };
            let mut err = super::sincos::lg_up(&rr).mul(Lg::of(5.0));
            if cut {
                err = err.add(Lg::pow2(fi));
            }
            Fx {
                negative: false,
                m: limbs::mul(&rr, &to_vec(&h)),
                lsb: -(gg + fi),
                err,
            }
        });
        let cos = want_cos.then(|| Fx {
            negative: false,
            m: to_vec(&horner(&y, 0)),
            lsb: -fi,
            err: Lg::of(5.0),
        });
        return (sin, cos);
    }
    let mut a = [0u64; L];
    a[fl - 1] = (j as u64) << 58;
    a[fl] = (j as u64) >> 6;
    let rho = sub(&rf, &a);
    let y = mul(&rho, &rho);
    let h0 = horner(&y, 0);
    let srho = mul(&rho, &horner(&y, 1));
    let (sa, ca) = &table()[j];
    let (sa, ca) = (cut::<L>(sa), cut::<L>(ca));
    let sin = want_sin.then(|| Fx {
        negative: false,
        m: to_vec(&add(&mul(&sa, &h0), &mul(&ca, &srho))),
        lsb: -fi,
        err: Lg::of(16.0),
    });
    let cos = want_cos.then(|| Fx {
        negative: false,
        m: to_vec(&sub(&mul(&ca, &h0), &mul(&sa, &srho))),
        lsb: -fi,
        err: Lg::of(16.0),
    });
    (sin, cos)
}
