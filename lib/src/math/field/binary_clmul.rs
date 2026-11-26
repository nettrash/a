//! Carry-less products of polynomials over `GF(2)` held as 64-bit words, the coefficient of `xⁱ`
//! bit `i mod 64` of word `i / 64`, for the binary fields.
//!
//! - Two words: by `PMULL` (`vmull_p64`, the aarch64 cryptographic extension) or `PCLMULQDQ`
//!   (`_mm_clmulepi64_si128`, x86_64) where the CPU has them; otherwise by a window of 4 bits, the
//!   16 multiples of one word by the polynomials of degree below 4 in a table and the other word
//!   taken a nibble at a time from the top (the window method of Hankerson, Menezes and Vanstone,
//!   "Guide to Elliptic Curve Cryptography", 2004, §2.3.3, on a single word).
//! - Two polynomials of `n` words: Karatsuba's method (Karatsuba and Ofman, 1962), each product
//!   split at `⌈n / 2⌉` words into three of the halves, down to single words: `O(n^1.58)` word
//!   products.
//! - Squares: by spreading the bits, a zero after each, since `(Σ aᵢxⁱ)² = Σ aᵢx²ⁱ` over `GF(2)`
//!   (Hankerson, Menezes and Vanstone, Algorithm 2.39): linear in `n`, no products.
//!
//! The hardware kernels are the place of the binary fields where `unsafe` is needed: an intrinsic
//! of an instruction the CPU does not have is undefined behaviour. Each kernel is compiled into a
//! function with its target feature enabled, so that the intrinsic is inlined into the loops, and
//! [`mul`] calls it only after [`Clmul::is_available`] has found the instruction. Under Miri, which
//! runs no such intrinsics, only the portable kernel exists.

use super::binary::Clmul;

/// The carry-less product of two words, low word first.
pub(super) trait Kernel {
    /// `a·b` as polynomials over `GF(2)`, `(low, high)`.
    ///
    /// # Safety
    ///
    /// The CPU must have the kernel's instructions.
    unsafe fn clmul(a: u64, b: u64) -> (u64, u64);
}

/// The portable kernel.
pub(super) struct Portable;

impl Kernel for Portable {
    #[inline(always)]
    unsafe fn clmul(a: u64, b: u64) -> (u64, u64) {
        let r = clmul_portable(a, b);
        (r as u64, (r >> 64) as u64)
    }
}

/// `a·b` by a window of 4 bits: the 16 products of `a` by the polynomials of degree below 4, then
/// the nibbles of `b` from the top, by Horner's rule in `x⁴`.
#[inline]
pub(super) fn clmul_portable(a: u64, b: u64) -> u128 {
    let a = u128::from(a);
    let mut t = [0u128; 16];
    for i in 1..16 {
        t[i] = if i.is_multiple_of(2) {
            t[i / 2] << 1
        } else {
            t[i - 1] ^ a
        };
    }
    let mut r = 0u128;
    for k in (0..16).rev() {
        r = (r << 4) ^ t[(b >> (4 * k)) as usize & 15];
    }
    r
}

/// `PMULL` of the aarch64 cryptographic extension.
#[cfg(all(target_arch = "aarch64", not(miri)))]
pub(super) struct Pmull;

#[cfg(all(target_arch = "aarch64", not(miri)))]
impl Kernel for Pmull {
    #[inline(always)]
    unsafe fn clmul(a: u64, b: u64) -> (u64, u64) {
        // SAFETY: the caller's CPU has the AES extension, of which `PMULL` is part.
        let r = unsafe { core::arch::aarch64::vmull_p64(a, b) };
        (r as u64, (r >> 64) as u64)
    }
}

/// `PCLMULQDQ` of x86_64.
#[cfg(all(target_arch = "x86_64", not(miri)))]
pub(super) struct Pclmulqdq;

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl Kernel for Pclmulqdq {
    #[inline(always)]
    unsafe fn clmul(a: u64, b: u64) -> (u64, u64) {
        use core::arch::x86_64::{
            _mm_clmulepi64_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
        };
        // SAFETY: the caller's CPU has `PCLMULQDQ`; the moves are SSE2, which every x86_64 has.
        unsafe {
            let r = _mm_clmulepi64_si128(
                _mm_cvtsi64_si128(a as i64),
                _mm_cvtsi64_si128(b as i64),
                0x00,
            );
            (
                _mm_cvtsi128_si64(r) as u64,
                _mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)) as u64,
            )
        }
    }
}

/// A kernel's product of two polynomials of `n` words into `2n`, with its scratch.
type MulFn = unsafe fn(&[u64], &[u64], &mut [u64], &mut [u64]);

/// From this many words a product is split by Karatsuba's method; below, the schoolbook product.
const KARATSUBA: usize = 2;

/// The words of scratch a product of two polynomials of `n` words needs: the two sums of halves
/// and their product at each level of Karatsuba's method.
#[must_use]
pub(super) fn scratch_len(n: usize) -> usize {
    if n < KARATSUBA {
        0
    } else {
        let lo = n.div_ceil(2);
        4 * lo + scratch_len(lo)
    }
}

/// Sets `r[..2n]` to `a·b` for `a` and `b` of `n` words: the schoolbook product below
/// [`KARATSUBA`] words, otherwise `a = a₀ + a₁X`, `b = b₀ + b₁X` with `X = x^(64·⌈n/2⌉)` and
/// `a·b = a₀b₀ + ((a₀ + a₁)(b₀ + b₁) + a₀b₀ + a₁b₁)·X + a₁b₁·X²`, the three products by `rec`.
///
/// # Safety
///
/// The CPU must have `K`'s instructions, and `rec` must be safe to call on it with the same
/// arguments' shapes.
#[inline(always)]
unsafe fn karatsuba<K: Kernel>(a: &[u64], b: &[u64], r: &mut [u64], s: &mut [u64], rec: MulFn) {
    let n = a.len();
    if n < KARATSUBA {
        r[..2 * n].fill(0);
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                // SAFETY: forwarded from the caller.
                let (lo, hi) = unsafe { K::clmul(x, y) };
                r[i + j] ^= lo;
                r[i + j + 1] ^= hi;
            }
        }
        return;
    }
    let lo = n.div_ceil(2);
    let hi = n - lo;
    let (a0, a1) = a.split_at(lo);
    let (b0, b1) = b.split_at(lo);
    {
        let (r0, r1) = r.split_at_mut(2 * lo);
        // SAFETY: forwarded from the caller.
        unsafe {
            rec(a0, b0, r0, s);
            rec(a1, b1, &mut r1[..2 * hi], s);
        }
    }
    let (sa, rest) = s.split_at_mut(lo);
    let (sb, rest) = rest.split_at_mut(lo);
    let (mid, rest) = rest.split_at_mut(2 * lo);
    sa.copy_from_slice(a0);
    sb.copy_from_slice(b0);
    for i in 0..hi {
        sa[i] ^= a1[i];
        sb[i] ^= b1[i];
    }
    // SAFETY: forwarded from the caller.
    unsafe { rec(sa, sb, mid, rest) };
    for i in 0..2 * lo {
        mid[i] ^= r[i];
    }
    for i in 0..2 * hi {
        mid[i] ^= r[2 * lo + i];
    }
    for i in 0..2 * lo {
        r[lo + i] ^= mid[i];
    }
}

/// The portable product.
///
/// # Safety
///
/// None beyond the shapes [`mul`] checks: the portable kernel uses no instruction a CPU may lack.
unsafe fn mul_portable(a: &[u64], b: &[u64], r: &mut [u64], s: &mut [u64]) {
    // SAFETY: the portable kernel needs nothing of the CPU, and recurses into itself.
    unsafe { karatsuba::<Portable>(a, b, r, s, mul_portable) }
}

/// The product by `PMULL`, compiled with the AES extension enabled.
///
/// # Safety
///
/// The CPU must have the AES extension.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "aes")]
unsafe fn mul_pmull(a: &[u64], b: &[u64], r: &mut [u64], s: &mut [u64]) {
    // Karatsuba's halves down to eight words, then the schoolbook of `fixed_pmull`. Measured on an
    // Apple M4 Max (field products of 5 to 72 words, bases 4 to 9 alternated): the schoolbook wins
    // up to eight words (Karatsuba's halves take 1.34–1.51 times its time), and the halves of nine
    // take 0.91–0.95 times the time of its 81 products, so that the products of 9, 17–19, 33–36
    // and 65–72 words are 4–12 % faster than with a base of nine; the base does not change the
    // other lengths.
    #[cfg(target_feature = "aes")]
    if small_pmull(a, b, r) {
        return;
    }
    // SAFETY: forwarded from the caller; the recursion is into this function again.
    unsafe { karatsuba::<Pmull>(a, b, r, s, mul_pmull) }
}

/// The product by `PCLMULQDQ`, compiled with it enabled.
///
/// # Safety
///
/// The CPU must have `PCLMULQDQ`.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "pclmulqdq")]
unsafe fn mul_pclmulqdq(a: &[u64], b: &[u64], r: &mut [u64], s: &mut [u64]) {
    // SAFETY: forwarded from the caller; the recursion is into this function again.
    unsafe { karatsuba::<Pclmulqdq>(a, b, r, s, mul_pclmulqdq) }
}

/// The product of `n ≤ 8` words by `PMULL` where the target enables it at compile time: the
/// schoolbook's `n²` products of words, each 128-bit product added to the sum of its position in a
/// vector register, and the sums of the `2n − 1` positions put together at the end; inlined, with
/// no scratch, and unrolled for each `n`.
#[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
#[inline(always)]
fn fixed_pmull<const N: usize>(a: &[u64], b: &[u64], r: &mut [u64]) {
    use core::arch::aarch64::{
        uint64x2_t, vdupq_n_u64, veorq_u64, vgetq_lane_u64, vmull_p64, vreinterpretq_u64_p128,
    };
    let a: &[u64; N] = a.first_chunk().unwrap_or_else(|| unreachable!());
    let b: &[u64; N] = b.first_chunk().unwrap_or_else(|| unreachable!());
    // SAFETY: PMULL is enabled at compile time here; the rest are vector moves and exclusive ors.
    unsafe {
        let mut acc: [uint64x2_t; 15] = [vdupq_n_u64(0); 15];
        for i in 0..N {
            for j in 0..N {
                acc[i + j] = veorq_u64(acc[i + j], vreinterpretq_u64_p128(vmull_p64(a[i], b[j])));
            }
        }
        r[0] = vgetq_lane_u64::<0>(acc[0]);
        for k in 1..2 * N - 1 {
            r[k] = vgetq_lane_u64::<0>(acc[k]) ^ vgetq_lane_u64::<1>(acc[k - 1]);
        }
        r[2 * N - 1] = vgetq_lane_u64::<1>(acc[2 * N - 2]);
    }
}

/// [`fixed_pmull`] for the length of `a`, if it is at most 8 words; whether it was.
#[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
#[inline(always)]
fn small_pmull(a: &[u64], b: &[u64], r: &mut [u64]) -> bool {
    match a.len() {
        1 => fixed_pmull::<1>(a, b, r),
        2 => fixed_pmull::<2>(a, b, r),
        3 => fixed_pmull::<3>(a, b, r),
        4 => fixed_pmull::<4>(a, b, r),
        5 => fixed_pmull::<5>(a, b, r),
        6 => fixed_pmull::<6>(a, b, r),
        7 => fixed_pmull::<7>(a, b, r),
        8 => fixed_pmull::<8>(a, b, r),
        _ => return false,
    }
    true
}

/// Sets `r[..2n]` to the carry-less product of `a` and `b`, both of `n` words, by the kernel
/// `kind`, with `s` as scratch ([`scratch_len`] words).
///
/// # Panics
///
/// Panics if the shapes are wrong or the CPU does not have `kind`'s instruction.
#[track_caller]
pub(super) fn mul(kind: Clmul, a: &[u64], b: &[u64], r: &mut [u64], s: &mut [u64]) {
    let n = a.len();
    assert!(
        b.len() == n && r.len() >= 2 * n && s.len() >= scratch_len(n),
        "wrong shapes for a carry-less product"
    );
    assert!(kind.is_available(), "{kind:?} is not available on this CPU");
    #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
    if kind == Clmul::Pmull && small_pmull(a, b, r) {
        return;
    }
    match kind {
        // SAFETY: the portable kernel needs nothing of the CPU.
        Clmul::Portable => unsafe { mul_portable(a, b, r, s) },
        Clmul::Pmull => {
            #[cfg(all(target_arch = "aarch64", not(miri)))]
            // SAFETY: `is_available` found the AES extension just above.
            unsafe {
                mul_pmull(a, b, r, s);
            }
            #[cfg(not(all(target_arch = "aarch64", not(miri))))]
            unreachable!("PMULL is available only on aarch64");
        }
        Clmul::Pclmulqdq => {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: `is_available` found `PCLMULQDQ` just above.
            unsafe {
                mul_pclmulqdq(a, b, r, s);
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            unreachable!("PCLMULQDQ is available only on x86_64");
        }
    }
}

/// The inverse of `a` modulo `f`, of degree `m`, `f(0) = 1` and `finv = f⁻¹ mod x⁶⁴`, by the
/// division steps of [`binary_inv`](super::binary_inv) with `kind`'s carry-less products; `None`
/// for an `a` that shares a factor with `f`, or one the steps leave to another way: a modulus
/// longer than the kernel's bound, and under the portable kernel one outside 24 to 31 bits.
///
/// # Panics
///
/// Panics if the CPU does not have `kind`'s instruction.
#[track_caller]
pub(super) fn inverse(kind: Clmul, a: &[u64], f: &[u64], m: u32, finv: u64) -> Option<Vec<u64>> {
    assert!(kind.is_available(), "{kind:?} is not available on this CPU");
    use super::binary_inv::{MAX_PCLMULQDQ, MAX_PMULL};
    match kind {
        // Measured on an Apple M4 Max (64 random elements a field, alternated): with the portable
        // kernel's products of words the steps take 1.0–2.3 times the time of the extended Euclid
        // (`binary_poly`, the caller's other way), but from 24 to 31 bits, where one batch does
        // them all and they take 0.91–0.99 times its time.
        Clmul::Portable if !(24..32).contains(&m) => None,
        // SAFETY: the portable kernel needs nothing of the CPU.
        Clmul::Portable => unsafe { super::binary_inv::inverse::<Portable>(a, f, m, finv) },
        Clmul::Pmull if f.len() > MAX_PMULL => None,
        Clmul::Pclmulqdq if f.len() > MAX_PCLMULQDQ => None,
        Clmul::Pmull => {
            #[cfg(all(target_arch = "aarch64", not(miri)))]
            // SAFETY: `is_available` found the AES extension just above.
            return unsafe { inverse_pmull(a, f, m, finv) };
            #[cfg(not(all(target_arch = "aarch64", not(miri))))]
            unreachable!("PMULL is available only on aarch64");
        }
        Clmul::Pclmulqdq => {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: `is_available` found `PCLMULQDQ` just above.
            return unsafe { inverse_pclmulqdq(a, f, m, finv) };
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            unreachable!("PCLMULQDQ is available only on x86_64");
        }
    }
}

/// The inverse by `PMULL`, compiled with the AES extension enabled.
///
/// # Safety
///
/// The CPU must have the AES extension.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "aes")]
unsafe fn inverse_pmull(a: &[u64], f: &[u64], m: u32, finv: u64) -> Option<Vec<u64>> {
    // SAFETY: forwarded from the caller.
    unsafe { super::binary_inv::inverse::<Pmull>(a, f, m, finv) }
}

/// The inverse by `PCLMULQDQ`, compiled with it enabled.
///
/// # Safety
///
/// The CPU must have `PCLMULQDQ`.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "pclmulqdq")]
unsafe fn inverse_pclmulqdq(a: &[u64], f: &[u64], m: u32, finv: u64) -> Option<Vec<u64>> {
    // SAFETY: forwarded from the caller.
    unsafe { super::binary_inv::inverse::<Pclmulqdq>(a, f, m, finv) }
}

/// The 64 bits of `x` with a zero inserted above each: bit `i` of `x` at bit `2i`.
#[inline]
#[must_use]
pub(super) fn spread(x: u32) -> u64 {
    let mut x = u64::from(x);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555
}

/// The even bits of `x` gathered: bit `2i` of `x` at bit `i`, the inverse of [`spread`].
#[inline]
#[must_use]
pub(super) fn gather_even(x: u64) -> u32 {
    let mut x = x & 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    (x | (x >> 16)) as u32
}

/// Sets `r[..2n]` to the square of `a`, of `n` words, by spreading its bits.
pub(super) fn square(a: &[u64], r: &mut [u64]) {
    // A word's carry-less square is its bits spread: one `PMULL` of the word by itself.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
    {
        for (i, &w) in a.iter().enumerate() {
            // SAFETY: PMULL is enabled at compile time here.
            let p = unsafe { core::arch::aarch64::vmull_p64(w, w) };
            r[2 * i] = p as u64;
            r[2 * i + 1] = (p >> 64) as u64;
        }
        return;
    }
    #[allow(unreachable_code)]
    for (i, &w) in a.iter().enumerate() {
        r[2 * i] = spread(w as u32);
        r[2 * i + 1] = spread((w >> 32) as u32);
    }
}
