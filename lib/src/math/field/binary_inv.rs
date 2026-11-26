//! The inverse in a binary field by Bernstein and Yang's division steps ("Fast constant-time gcd
//! computation and modular inversion", 2019, §3 and §6, for polynomials), in batches of a word.
//!
//! A division step on `(δ, f, g)`, `f(0) = 1`: if `δ > 0` and `g(0) = 1`, `(1 − δ, g, (f + g)/x)`;
//! otherwise `(1 + δ, f, (g + g(0)·f)/x)`. Then `x·(f′, g′) = T·(f, g)` for the matrix
//! `T = [[0, x], [1, 1]]` or `[[x, 0], [g(0), 1]]`, of determinant `x`, so that the common factors of
//! `f` and `g` are kept, but for powers of `x`, which `f(0) = 1` excludes. From `f` the modulus and
//! `g = a`, `g` is 0 after at most `2m − 1` steps for a modulus of degree `m` (Theorem 6.2, which is
//! the case `R₀` the reversal of `f`, of degree `m` as `f(0) = 1`), and `f` is then the greatest
//! common divisor, 1. With `d`, `e` such that `f ≡ d·a` and `g ≡ e·a` modulo the modulus, from
//! `d = 0`, `e = 1`, the inverse is `d` at the end.
//!
//! The steps of a batch of [`STEPS`] take only the low words of `f` and `g`: a step's choice is by
//! `g(0)` and `δ`, and after `j` steps the low bit of `g` depends on its `j + 1` low bits and on
//! those of `f`. The batch makes the product `M` of its matrices, whose entries are of degree at
//! most `STEPS` and so a word each, a run of zeros at the bottom of `g` in one shift; then
//! `(f, g) ← M·(f, g)/x^STEPS` (exact) and `(d, e) ← M·(d, e)/x^STEPS` modulo the modulus, made exact
//! as in Montgomery's reduction: `t = (d mod x^STEPS)·f⁻¹ mod x^STEPS`, and `d + t·f` is divisible by
//! `x^STEPS`, which keeps `d` and `e` below the degree `m`. The rows are carry-less products of a
//! word by the polynomials' words.

use super::binary_clmul::Kernel;
use core::hint::select_unpredictable as select;

/// The steps of a batch: the matrix's entries, of degree at most this, fit a word, and the low
/// word of `g` holds the bits the steps look at.
const STEPS: u32 = 63;

/// The longest modulus, in words, that the division steps take under `PMULL` (8191 bits; longer
/// ones take the extended Euclid): its buffers are on the stack, of the size of the class of the
/// modulus' length (2, 4, 8, 16, 40, 64 or 128 words). Measured on an Apple M4 Max (64 random
/// elements a field, alternated with the extended Euclid of `binary_poly`): the steps take 0.23–0.47
/// times its time up to 1000 bits, 0.44–0.58 up to 4100 and 0.61 from 5000 to 8191 bits.
pub(super) const MAX_PMULL: usize = 128;

/// The longest modulus, in words, that the division steps take under `PCLMULQDQ` (2559 bits): a
/// first estimate, not yet measured on x86_64.
pub(super) const MAX_PCLMULQDQ: usize = 40;

/// `f⁻¹ mod x⁶⁴`, for `f(0) = 1`: Newton's iteration `y ← f·y²` (over `GF(2)`, `y(2 − f·y)` is
/// `f·y²`), which doubles the bits that are right, from `y = 1`.
pub(super) fn inverse_mod_x64(f0: u64, clmul: impl Fn(u64, u64) -> u128) -> u64 {
    let mut y = 1u64;
    for _ in 0..6 {
        y = clmul(f0, clmul(y, y) as u64) as u64;
    }
    y
}

/// `STEPS` division steps on the low words of `f` and `g`: `δ` after them and the matrix
/// `[[u, v], [q, r]]`.
#[inline(always)]
fn batch(mut delta: i64, mut fl: u64, mut gl: u64) -> (i64, [u64; 4]) {
    let (mut u, mut v, mut q, mut r) = (1u64, 0u64, 0u64, 1u64);
    let mut i = STEPS;
    loop {
        // The zeros at the bottom of g, at most the steps left: as many steps of the second kind
        // with g(0) = 0, f's row multiplied by x each.
        let z = (gl | (1u64 << i)).trailing_zeros();
        gl >>= z;
        u <<= z;
        v <<= z;
        delta += i64::from(z);
        i -= z;
        if i == 0 {
            break;
        }
        // g(0) = 1: for δ > 0 the step of the first kind, which is the second after f and g (and
        // their rows) change places and δ its sign.
        let swap = delta > 0;
        let (nf, ng) = (select(swap, gl, fl), select(swap, fl, gl));
        let (nu, nq) = (select(swap, q, u), select(swap, u, q));
        let (nv, nr) = (select(swap, r, v), select(swap, v, r));
        delta = select(swap, -delta, delta);
        // g + f, now even: its division by x, the row of f by x and δ + 1 are the next round's
        // shift.
        fl = nf;
        gl = ng ^ nf;
        u = nu;
        v = nv;
        q = nq ^ nu;
        r = nr ^ nv;
    }
    (delta, [u, v, q, r])
}

/// `(a·x + b·y) / x^STEPS` into `out`, for `x`, `y` of `out.len()` words and the division exact.
#[inline(always)]
unsafe fn row_shift<K: Kernel, const W: usize>(
    a: u64,
    x: &[u64],
    b: u64,
    y: &[u64],
    out: &mut [u64],
) {
    let mut acc = [[0u64; W], [0u64; W]];
    let acc = acc.as_flattened_mut();
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        // SAFETY: forwarded from the caller (the CPU has the kernel's instruction).
        let (l1, h1) = unsafe { K::clmul(a, xi) };
        // SAFETY: as above.
        let (l2, h2) = unsafe { K::clmul(b, yi) };
        acc[i] ^= l1 ^ l2;
        acc[i + 1] ^= h1 ^ h2;
    }
    debug_assert_eq!(acc[0] & ((1 << STEPS) - 1), 0, "an exact division");
    for (i, o) in out.iter_mut().enumerate() {
        *o = (acc[i] >> STEPS) | (acc[i + 1] << (64 - STEPS));
    }
}

/// `(a·x + b·y + t·f) / x^STEPS` into `out`, `t` making the division exact: modulo `f`, for `x`,
/// `y` below its degree.
#[inline(always)]
unsafe fn row_mod<K: Kernel, const W: usize>(
    a: u64,
    x: &[u64],
    b: u64,
    y: &[u64],
    f: &[u64],
    finv: u64,
    out: &mut [u64],
) {
    let (n, l) = (out.len(), f.len());
    debug_assert!(l <= n + 1 && l < 2 * W);
    let mut acc = [[0u64; W], [0u64; W]];
    let acc = acc.as_flattened_mut();
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        // SAFETY: forwarded from the caller.
        let (l1, h1) = unsafe { K::clmul(a, xi) };
        // SAFETY: as above.
        let (l2, h2) = unsafe { K::clmul(b, yi) };
        acc[i] ^= l1 ^ l2;
        acc[i + 1] ^= h1 ^ h2;
    }
    // SAFETY: as above.
    let t = unsafe { K::clmul(acc[0], finv) }.0 & ((1 << STEPS) - 1);
    for (i, &w) in f.iter().enumerate() {
        // SAFETY: as above.
        let (lo, hi) = unsafe { K::clmul(t, w) };
        acc[i] ^= lo;
        acc[i + 1] ^= hi;
    }
    debug_assert_eq!(acc[0] & ((1 << STEPS) - 1), 0, "an exact division");
    debug_assert!(acc[n + 1..=l].iter().all(|&w| w == 0), "below the degree");
    for (i, o) in out.iter_mut().enumerate() {
        *o = (acc[i] >> STEPS) | (acc[i + 1] << (64 - STEPS));
    }
}

/// The inverse of `a` (of `n` words, below the degree `m` of `f`) modulo `f` (`l` words, bit `m`
/// included, `f(0) = 1`), `finv = f⁻¹ mod x⁶⁴`: `None` if `a` shares a factor with `f` (0
/// included), or if the batches run past their bound, which they do not (the caller then takes
/// another way).
///
/// # Safety
///
/// The CPU must have `K`'s instruction.
#[inline(always)]
pub(super) unsafe fn inverse<K: Kernel>(
    a: &[u64],
    f: &[u64],
    m: u32,
    finv: u64,
) -> Option<Vec<u64>> {
    if a.len() > f.len() {
        return None;
    }
    match f.len() {
        // SAFETY: forwarded from the caller.
        0..=2 => unsafe { inverse_in::<K, 2>(a, f, m, finv) },
        // SAFETY: as above.
        3..=4 => unsafe { inverse_in::<K, 4>(a, f, m, finv) },
        // SAFETY: as above.
        5..=8 => unsafe { inverse_in::<K, 8>(a, f, m, finv) },
        // SAFETY: as above.
        9..=16 => unsafe { inverse_in::<K, 16>(a, f, m, finv) },
        // SAFETY: as above.
        17..=40 => unsafe { inverse_in::<K, 40>(a, f, m, finv) },
        // SAFETY: as above.
        41..=64 => unsafe { inverse_in::<K, 64>(a, f, m, finv) },
        // SAFETY: as above.
        65..=MAX_PMULL => unsafe { inverse_in::<K, MAX_PMULL>(a, f, m, finv) },
        _ => None,
    }
}

/// [`inverse`] with buffers of `W` words, at least the modulus' `l`.
///
/// # Safety
///
/// As [`inverse`].
#[inline(always)]
unsafe fn inverse_in<K: Kernel, const W: usize>(
    a: &[u64],
    f: &[u64],
    m: u32,
    finv: u64,
) -> Option<Vec<u64>> {
    let (n, l) = (a.len(), f.len());
    debug_assert!(l <= W && n <= l);
    let mut fv = [0u64; W];
    let mut gv = [0u64; W];
    fv[..l].copy_from_slice(f);
    gv[..n].copy_from_slice(a);
    let (mut d, mut e) = ([0u64; W], [0u64; W]);
    e[0] = 1;
    let mut delta = 1i64;
    // At most 2m − 1 steps; one batch more for safety.
    let bound = (2 * m as usize).div_ceil(STEPS as usize) + 1;
    let mut rounds = 0;
    while gv[..l].iter().any(|&w| w != 0) {
        if rounds == bound {
            return None;
        }
        rounds += 1;
        let (nd, [u, v, q, r]) = batch(delta, fv[0], gv[0]);
        delta = nd;
        let (mut f2, mut g2) = ([0u64; W], [0u64; W]);
        // SAFETY: forwarded from the caller.
        unsafe {
            row_shift::<K, W>(u, &fv[..l], v, &gv[..l], &mut f2[..l]);
            row_shift::<K, W>(q, &fv[..l], r, &gv[..l], &mut g2[..l]);
        }
        fv = f2;
        gv = g2;
        let (mut d2, mut e2) = ([0u64; W], [0u64; W]);
        // SAFETY: as above.
        unsafe {
            row_mod::<K, W>(u, &d[..n], v, &e[..n], f, finv, &mut d2[..n]);
            row_mod::<K, W>(q, &d[..n], r, &e[..n], f, finv, &mut e2[..n]);
        }
        d = d2;
        e = e2;
    }
    (fv[0] == 1 && fv[1..l].iter().all(|&w| w == 0)).then(|| d[..n].to_vec())
}
