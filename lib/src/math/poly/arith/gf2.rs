//! Polynomials over GF(2), packed 64 coefficients a word (coefficient `i` at bit `i % 64` of word
//! `i / 64`), behind the word kernels for the modulus 2: products by carry-less multiplication of
//! words (`PMULL` on aarch64, `PCLMULQDQ` on x86_64, found at run time, a window method
//! elsewhere and under Miri) with Karatsuba's method on words, and for long operands
//! ([`by_words`]) by the transforms on words of the other moduli; squares by spreading the bits
//! (a square over GF(2) is linear), remainders 64 quotient coefficients at a time by Barrett's
//! method on words, which over GF(2) needs no correction, and for long divisors and quotients
//! ([`NEWTON_DEG_REUSED`], [`NEWTON_DEG`], [`NEWTON_QUOT`]) by Barrett's method on polynomials
//! (the quotient of the top `2 deg f` coefficients two products with `floor(x^(2 deg f) / f)`,
//! made by Newton's iteration, `g ← f·g² mod x^(2k)` over GF(2)); greatest common divisors by
//! Euclid's algorithm on the packed words below the degree [`HGCD_GCD`] and by the half gcd on
//! the packed products from it ([`gcd`](crate::math::poly::gcd)'s engine `gf2`).
//!
//! A `ModPolynomial` modulo 2 keeps one word a coefficient; the kernels here pack their operands,
//! work on the bits and unpack the result.

use core::sync::atomic::{AtomicU8, Ordering};

/// The coefficients `c` (each 0 or 1), packed; not trimmed. Eight coefficients at a time: their
/// low bytes gathered into one word, whose bits a product brings together.
pub(crate) fn pack(c: &[u64]) -> Vec<u64> {
    let mut b = vec![0u64; c.len().div_ceil(64)];
    for (w, chunk) in b.iter_mut().zip(c.chunks(64)) {
        let mut x = 0u64;
        let (eights, rest) = chunk.as_chunks::<8>();
        for (k, e) in eights.iter().enumerate() {
            let g = (e[0] & 1)
                | ((e[1] & 1) << 8)
                | ((e[2] & 1) << 16)
                | ((e[3] & 1) << 24)
                | ((e[4] & 1) << 32)
                | ((e[5] & 1) << 40)
                | ((e[6] & 1) << 48)
                | ((e[7] & 1) << 56);
            // Byte j's bit 0 to bit 56 + j: the eight bits in the top byte.
            x |= (g.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
        }
        let base = 8 * eights.len();
        for (i, &v) in rest.iter().enumerate() {
            x |= (v & 1) << (base + i);
        }
        *w = x;
    }
    b
}

/// The packed `b` as `out.len()` coefficients of one word each (zeros past `b`).
pub(crate) fn unpack_into(out: &mut [u64], b: &[u64]) {
    for (chunk, &w) in out
        .chunks_mut(64)
        .zip(b.iter().chain(core::iter::repeat(&0)))
    {
        for (j, o) in chunk.iter_mut().enumerate() {
            *o = (w >> j) & 1;
        }
    }
}

/// The packed `b` as coefficients of one word each, trimmed.
pub(crate) fn unpack(b: &[u64]) -> Vec<u64> {
    let len = degree(b).map_or(0, |d| d + 1);
    let mut out = vec![0u64; len];
    unpack_into(&mut out, b);
    out
}

/// Drops the zero words at the top.
#[inline]
pub(crate) fn trim(v: &mut Vec<u64>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// The degree, `None` for zero.
#[inline]
pub(crate) fn degree(b: &[u64]) -> Option<usize> {
    let i = b.iter().rposition(|&w| w != 0)?;
    Some(64 * i + 63 - b[i].leading_zeros() as usize)
}

/// `b · x^s`, packed.
fn shl(b: &[u64], s: u32) -> Vec<u64> {
    debug_assert!(s < 64);
    let mut r = vec![0u64; b.len() + 1];
    if s == 0 {
        r[..b.len()].copy_from_slice(b);
        return r;
    }
    for (i, &w) in b.iter().enumerate() {
        r[i] |= w << s;
        r[i + 1] = w >> (64 - s);
    }
    r
}

/// `b / x^s` (the low `s` bits dropped), packed.
fn shr(b: &[u64], s: u32) -> Vec<u64> {
    debug_assert!(s < 64);
    if s == 0 {
        return b.to_vec();
    }
    let mut r = vec![0u64; b.len()];
    for i in 0..b.len() {
        let hi = b.get(i + 1).map_or(0, |&w| w << (64 - s));
        r[i] = (b[i] >> s) | hi;
    }
    r
}

/// The 32 bits of `x` with a zero inserted above each.
#[inline(always)]
fn spread(x: u32) -> u64 {
    let mut x = u64::from(x);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555
}

/// The square, packed and trimmed: the bits spread, one coefficient in two.
pub(crate) fn sqr(a: &[u64]) -> Vec<u64> {
    let mut r = vec![0u64; 2 * a.len()];
    for (i, &w) in a.iter().enumerate() {
        r[2 * i] = spread(w as u32);
        r[2 * i + 1] = spread((w >> 32) as u32);
    }
    trim(&mut r);
    r
}

/// A carry-less multiplier of two words: the low and the high word of the product.
trait Clm {
    /// # Safety
    ///
    /// The CPU must have the kernel's instruction.
    unsafe fn cl(a: u64, b: u64) -> (u64, u64);
}

/// The window method: the 16 multiples of `a` by the polynomials of degree below 4, the bits of
/// `b` four at a time.
struct Portable;

impl Clm for Portable {
    #[inline(always)]
    unsafe fn cl(a: u64, b: u64) -> (u64, u64) {
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
            r = (r << 4) ^ t[((b >> (4 * k)) & 15) as usize];
        }
        (r as u64, (r >> 64) as u64)
    }
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
struct Pmull;

#[cfg(all(target_arch = "aarch64", not(miri)))]
impl Clm for Pmull {
    #[inline(always)]
    unsafe fn cl(a: u64, b: u64) -> (u64, u64) {
        // SAFETY: the caller's CPU has the AES extension, of which PMULL is part.
        let r = unsafe { core::arch::aarch64::vmull_p64(a, b) };
        (r as u64, (r >> 64) as u64)
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
struct Pclmul;

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl Clm for Pclmul {
    #[inline(always)]
    unsafe fn cl(a: u64, b: u64) -> (u64, u64) {
        use core::arch::x86_64::{
            _mm_clmulepi64_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
        };
        // SAFETY: the caller's CPU has PCLMULQDQ; the moves are SSE2, which every x86_64 has.
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

/// The kernel of this CPU: 1 portable, 2 PMULL, 3 PCLMULQDQ; 0 before the first look.
static KIND: AtomicU8 = AtomicU8::new(0);

#[inline]
fn kind() -> u8 {
    let k = KIND.load(Ordering::Relaxed);
    if k != 0 {
        return k;
    }
    let k = detect();
    KIND.store(k, Ordering::Relaxed);
    k
}

fn detect() -> u8 {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if !cfg!(a_baseline) && std::arch::is_aarch64_feature_detected!("aes") {
            return 2;
        }
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        if !cfg!(a_baseline) && std::arch::is_x86_feature_detected!("pclmulqdq") {
            return 3;
        }
    }
    1
}

/// Below this many words in both operands, the schoolbook product: measured on an Apple M4 Max
/// (`PMULL`; 4 to 512 words, square and unbalanced, thresholds 6 to 128), the schoolbook method
/// faster up to 27 words, a Karatsuba step from 28 (products of 12 to 512 words 1.3 to 1.6
/// times faster than with the step from 12).
const KARATSUBA: usize = 28;

/// `r ^= a·b` by the schoolbook method on words, `r` of at least `la + lb` words.
#[inline(always)]
unsafe fn school<C: Clm>(r: &mut [u64], a: &[u64], b: &[u64]) {
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            // SAFETY: forwarded from the caller.
            let (lo, hi) = unsafe { C::cl(x, y) };
            r[i + j] ^= lo;
            r[i + j + 1] ^= hi;
        }
    }
}

/// The scratch words [`kara`] takes for operands of `n` words: at each level the three half
/// products (`2·lo` words each, `lo = ⌈n/2⌉`) and four operands of `lo` words, then the next
/// level's.
fn kara_scratch(mut n: usize) -> usize {
    let mut t = 0;
    while n >= KARATSUBA {
        let lo = n.div_ceil(2);
        t += 10 * lo;
        n = lo;
    }
    t
}

/// `r ^= a·b` for `a` and `b` of `n` words each, `r` of at least `2n` words: the schoolbook
/// product below [`KARATSUBA`] words, else Karatsuba's three half products, their buffers carved
/// from `t` (at least [`kara_scratch`]`(n)` words), so that the recursion allocates nothing.
#[inline(always)]
unsafe fn kara<C: Clm>(r: &mut [u64], a: &[u64], b: &[u64], t: &mut [u64]) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    if n < KARATSUBA {
        // SAFETY: forwarded from the caller.
        unsafe { school::<C>(r, a, b) };
        return;
    }
    let lo = n.div_ceil(2);
    let hi = n - lo;
    let (a0, a1) = a.split_at(lo);
    let (b0, b1) = b.split_at(lo);
    // p0 = a0 b0 (2 lo words), p2 = a1 b1 (2 hi words), p1 = (a0 + a1)(b0 + b1) (2 lo words);
    // a1 and b1 padded to lo words.
    let (p0, t) = t.split_at_mut(2 * lo);
    let (p2, t) = t.split_at_mut(2 * lo);
    let (p1, t) = t.split_at_mut(2 * lo);
    let (sa, t) = t.split_at_mut(lo);
    let (sb, t) = t.split_at_mut(lo);
    let (a1p, t) = t.split_at_mut(lo);
    let (b1p, t) = t.split_at_mut(lo);
    p0.fill(0);
    p2.fill(0);
    p1.fill(0);
    sa.copy_from_slice(a0);
    sb.copy_from_slice(b0);
    for i in 0..hi {
        sa[i] ^= a1[i];
        sb[i] ^= b1[i];
    }
    a1p[..hi].copy_from_slice(a1);
    a1p[hi..].fill(0);
    b1p[..hi].copy_from_slice(b1);
    b1p[hi..].fill(0);
    // SAFETY: forwarded from the caller; `t` keeps kara_scratch(lo) words.
    unsafe {
        kara::<C>(p0, a0, b0, t);
        kara::<C>(p2, a1p, b1p, t);
        kara::<C>(p1, sa, sb, t);
    }
    for i in 0..2 * lo {
        p1[i] ^= p0[i] ^ p2[i];
        r[i] ^= p0[i];
        r[lo + i] ^= p1[i];
    }
    for i in 0..2 * hi {
        r[2 * lo + i] ^= p2[i];
    }
}

/// `r ^= a·b` for `la ≤ lb`, `r` of at least `la + lb` words: the longer operand cut into
/// pieces of the shorter's length, each a Karatsuba product, then the last, shorter piece times
/// `a` the same way (pieces of the last piece's length, and so on, the lengths going down as in
/// Euclid's algorithm), so that no product of two long operands is left to the schoolbook
/// method; a loop, not a recursion, so that it stays inlined into the kernel's
/// `#[target_feature]` function.
#[inline(always)]
unsafe fn mul_acc<C: Clm>(r: &mut [u64], a: &[u64], b: &[u64]) {
    // r[off..] ^= x·y, len x ≤ len y, r[off..] of at least len x + len y words; one scratch for
    // every Karatsuba product (x only gets shorter).
    let (mut x, mut y, mut off) = (a, b, 0usize);
    let mut t = vec![0u64; kara_scratch(x.len())];
    loop {
        let lx = x.len();
        if lx < KARATSUBA {
            // SAFETY: forwarded from the caller.
            unsafe { school::<C>(&mut r[off..], x, y) };
            return;
        }
        let mut start = 0;
        while start + lx <= y.len() {
            // SAFETY: forwarded from the caller.
            unsafe { kara::<C>(&mut r[off + start..], x, &y[start..start + lx], &mut t) };
            start += lx;
        }
        if start == y.len() {
            return;
        }
        // The rest of y, shorter than x, times x: r[off + start..] holds its length plus lx.
        let rest = &y[start..];
        off += start;
        y = x;
        x = rest;
    }
}

/// The reduction of `a` (normalised, shifted) modulo the divisor of `d`, in place, the quotient
/// words into `q` when given.
#[inline(always)]
unsafe fn rem_words<C: Clm>(a: &mut [u64], d: &Div, mut q: Option<&mut [u64]>) {
    let kf = d.kf;
    let f = &d.f;
    for t in (kf..a.len()).rev() {
        let top = a[t];
        if top == 0 {
            continue;
        }
        // SAFETY: forwarded from the caller.
        let (_, h) = unsafe { C::cl(top, d.mu) };
        let qw = top ^ h;
        if let Some(q) = q.as_deref_mut() {
            q[t - kf] = qw;
        }
        let base = t - kf;
        for (j, &fj) in f[..kf].iter().enumerate() {
            // SAFETY: forwarded from the caller.
            let (lo, hi) = unsafe { C::cl(qw, fj) };
            a[base + j] ^= lo;
            a[base + j + 1] ^= hi;
        }
        // The top word of f is 1.
        a[t] ^= qw;
        debug_assert_eq!(a[t], 0);
    }
}

macro_rules! kernels {
    ($name:ident, $feat:literal, $k:ty, $cfg:meta) => {
        #[$cfg]
        #[target_feature(enable = $feat)]
        unsafe fn $name(op: Op<'_>) {
            // SAFETY: forwarded from the caller, who checked the instruction.
            unsafe { run::<$k>(op) }
        }
    };
}

/// One of the operations the kernels run.
enum Op<'a> {
    /// `r ^= a·b`, `la ≤ lb`.
    Mul(&'a mut [u64], &'a [u64], &'a [u64]),
    /// A reduction of `a` modulo the divisor, the quotient's words into the slice when given.
    Rem(&'a mut [u64], &'a Div, Option<&'a mut [u64]>),
}

#[inline(always)]
unsafe fn run<C: Clm>(op: Op<'_>) {
    // SAFETY: forwarded from the caller.
    unsafe {
        match op {
            Op::Mul(r, a, b) => mul_acc::<C>(r, a, b),
            Op::Rem(a, d, q) => rem_words::<C>(a, d, q),
        }
    }
}

kernels!(
    run_pmull,
    "aes",
    Pmull,
    cfg(all(target_arch = "aarch64", not(miri)))
);
kernels!(
    run_pclmul,
    "pclmulqdq",
    Pclmul,
    cfg(all(target_arch = "x86_64", not(miri)))
);

/// Runs `op` by the kernel of this CPU.
fn dispatch(op: Op<'_>) {
    match kind() {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        // SAFETY: `kind` found the AES extension.
        2 => unsafe { run_pmull(op) },
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        // SAFETY: `kind` found PCLMULQDQ.
        3 => unsafe { run_pclmul(op) },
        // SAFETY: the portable kernel needs nothing of the CPU.
        _ => unsafe { run::<Portable>(op) },
    }
}

/// Below this many words in the shorter operand a product is always the carry-less Karatsuba
/// product; from it, [`by_words`] decides.
pub(crate) const WORD_PRODUCT: usize = if cfg!(miri) { usize::MAX } else { 4096 };

/// The constant of [`by_words`]'s model, measured on an Apple M4 Max with Karatsuba's step from
/// [`KARATSUBA`] words (both methods forced on 50 shapes of 400,000 to 10⁷ coefficients in the
/// shorter operand and 1 to 16 times that in the longer, exact multiples and not, operands of
/// words and packed ones): Karatsuba's product faster on every shape whose model value is below
/// 12.3 (by 1.02 to 1.8 times), the transforms faster or slower by up to 1.5 times above it, by
/// the smoothness of the transform's length; from 7.0 to 12.25 the products of the shapes
/// between the two up to 1.7 times faster, none slower.
const WORD_PRODUCT_C: f64 = 12.25;

/// Whether the product of `la ≤ lb` packed words goes by the methods on words of every modulus
/// ([`wmul::mul_into_words`](super::wmul::mul_into_words): the transforms, one word a
/// coefficient, reduced modulo 2) rather than Karatsuba's on the packed words: from
/// [`WORD_PRODUCT`] words in the shorter operand, when
/// `la^0.585·lb ≥ 12.25·(la + lb)·log₂(64(la + lb))`, a model of the costs (Karatsuba's `lb/la`
/// products of `la^1.585`, the transforms' `n log n`): equal lengths from 48,814 words
/// (3,124,033 coefficients), `lb = 1.5la` from 35,356, `lb = 2la` from 29,537, `lb = 4la` from
/// 22,162, and never below 18,751 words (at `lb ≈ 17la`; the bound grows again, as `log lb`, for
/// longer operands).
pub(crate) fn by_words(la: usize, lb: usize) -> bool {
    if la < WORD_PRODUCT {
        return false;
    }
    let (x, y) = (la as f64, lb as f64);
    x.powf(0.585) * y >= WORD_PRODUCT_C * (x + y) * (64.0 * (x + y)).log2()
}

/// The product, packed and trimmed.
pub(crate) fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (a, b) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if by_words(a.len(), b.len()) {
        let (x, y) = (unpack(a), unpack(b));
        if x.is_empty() || y.is_empty() {
            return Vec::new();
        }
        let mut r = vec![0u64; x.len() + y.len() - 1];
        super::wmul::mul_into_words(&mut r, &x, &y, &super::nmod::WordMod::new(2));
        let mut p = pack(&r);
        trim(&mut p);
        return p;
    }
    let mut r = vec![0u64; a.len() + b.len() + 1];
    dispatch(Op::Mul(&mut r, a, b));
    trim(&mut r);
    r
}

/// From this degree of a divisor kept for many remainders ([`Div::reused`], and the squares and
/// products of [`Div::pow`]) the remainders with a quotient of [`NEWTON_QUOT`] coefficients or
/// more go by Barrett's method on polynomials, `μ = floor(x^(2d) / f)` made once; below, by words
/// (the words cost about `(d/64)²` carry-less products a remainder, the polynomials two packed
/// products of `d` coefficients). Measured on an Apple M4 Max, `pow_mod` with a 65-bit exponent
/// (by a dense base and by `x`) and products modulo `f`, dense random operands, degrees 2,048 to
/// 262,144: the words faster up to 13,312 (by 4 to 9%), the polynomials 1.02 times faster at
/// 14,336, 1.08 at 16,384, 1.3 to 1.4 at 32,768 and 2.0 to 2.4 at 131,072. At most
/// [`NEWTON_DEG`].
pub(crate) const NEWTON_DEG_REUSED: usize = if cfg!(miri) { 130 } else { 14_336 };

/// The same for a divisor of one remainder ([`Div::new`]), whose `μ` is made for it alone (to the
/// precision its quotient needs): measured, `div_rem` of `2d − 1` by `d` coefficients 1.16 times
/// faster by words at 49,152, by polynomials 1.05 times faster at 65,536, 1.4 at 131,072 and 1.9
/// at 262,144 (of `d + d/4` by `d`: 1.27 times faster at 65,536); short quotients by
/// [`NEWTON_QUOT`].
pub(crate) const NEWTON_DEG: usize = if cfg!(miri) { 200 } else { 65_536 };

/// The fewest quotient coefficients of a remainder by Barrett's method on polynomials: a short
/// quotient costs `lq·d/4096` carry-less products by words, linear in `d`, and by polynomials two
/// products of the unbalanced lengths `lq` and `d`, a few times that below some tens of words.
/// Measured (quotients of 512 to 65,536 coefficients, divisors of 16,384 to 524,288): the
/// crossover at 1,700 to 3,500 quotient coefficients for a reused division (lower for longer
/// divisors), 2,800 to 4,100 for one division from [`NEWTON_DEG`]; at 4,096 the polynomials 1.0
/// to 1.3 times faster, at 8,192 1.1 to 1.7 times, the words 1.1 to 1.2 times faster at 3,072
/// for one division.
pub(crate) const NEWTON_QUOT: usize = if cfg!(miri) { 70 } else { 4096 };

/// The division by a fixed non-zero `f`: `f·x^σ` of a degree `64 kf`, its top word 1, and
/// Barrett's constant of its top 65 bits.
#[derive(Clone, Debug)]
pub(crate) struct Div {
    /// `f · x^σ`, `kf + 1` words.
    f: Vec<u64>,
    /// The shift `σ` that makes the degree a multiple of 64.
    s: u32,
    /// The degree of `f·x^σ` over 64.
    kf: usize,
    /// The low word of `floor(x^128 / F)` for `F` the top 65 bits of `f·x^σ` (the quotient is
    /// `x^64` plus it).
    mu: u64,
    /// The degree of `f`.
    pub(crate) deg: usize,
    /// For a divisor of at least [`NEWTON_DEG_REUSED`] coefficients, what Barrett's method on
    /// polynomials needs.
    big: Option<Big>,
    /// Whether the division is kept for many remainders ([`Div::reused`]).
    reused: bool,
}

/// Barrett's method on polynomials for the divisor `f` of the degree `d`: `f` itself (trimmed)
/// and `μ = floor(x^(2d) / f)` (`d + 1` coefficients), made at the first remainder of a reused
/// division that takes it.
/// For `deg a ≤ 2d − 1`, `floor(a / f) = floor(floor(a / x^d)·μ / x^d)` exactly: no correction,
/// over any field.
#[derive(Clone, Debug)]
struct Big {
    f: Vec<u64>,
    mu: std::sync::OnceLock<Vec<u64>>,
}

impl Div {
    /// The division by `f`, packed, not zero, for one remainder or a few.
    pub(crate) fn new(f: &[u64]) -> Div {
        Div::with(f, false)
    }

    /// The division by `f`, packed, not zero, kept for many remainders: `μ` made once for all.
    pub(crate) fn reused(f: &[u64]) -> Div {
        Div::with(f, true)
    }

    fn with(f: &[u64], reused: bool) -> Div {
        let deg = degree(f).expect("division by the zero polynomial");
        let s = ((64 - deg % 64) % 64) as u32;
        let mut fs = shl(f, s);
        trim(&mut fs);
        let kf = (deg + s as usize) / 64;
        debug_assert_eq!(fs.len(), kf + 1);
        debug_assert_eq!(fs[kf], 1);
        let mu = if kf == 0 {
            0
        } else {
            // floor(x^128 / F), F = x^64 + flo: after x^64 F the rest is flo x^64; then the
            // quotient's bits 63 down to 0.
            let flo = fs[kf - 1];
            let big = (1u128 << 64) | u128::from(flo);
            let mut rem = u128::from(flo) << 64;
            let mut q = 0u64;
            for i in (0..64).rev() {
                if (rem >> (64 + i)) & 1 == 1 {
                    rem ^= big << i;
                    q |= 1 << i;
                }
            }
            q
        };
        let big = (deg >= NEWTON_DEG_REUSED).then(|| Big {
            f: f[..deg / 64 + 1].to_vec(),
            mu: std::sync::OnceLock::new(),
        });
        Div {
            f: fs,
            s,
            kf,
            mu,
            deg,
            big,
            reused,
        }
    }

    /// Barrett's method on polynomials, when the divisor and the quotient of `a` (of the degree
    /// `da ≥ deg f`) are long enough for it (`reuse`: for a division kept for many remainders):
    /// the quotient (when `want_q`) and the remainder, trimmed. From the top, each window of the
    /// `2 deg f` top coefficients of what is left, of `lq ≤ deg f` quotient coefficients:
    /// `q = floor(a₁·μ_hi / x^(lq−1))` for `a₁` the window's top `lq` coefficients and `μ_hi`
    /// the top `lq` of `μ` (its lower ones reach no coefficient of `a₁·μ` from `x^d` on), and
    /// the window less `q·f`. For one remainder only the top coefficients of `μ` its quotient
    /// needs are made.
    fn newton(
        &self,
        a: &[u64],
        da: usize,
        want_q: bool,
        reuse: bool,
    ) -> Option<(Vec<u64>, Vec<u64>)> {
        let big = self.big.as_ref()?;
        let d = self.deg;
        let lq_all = da + 1 - d;
        if lq_all < NEWTON_QUOT || !reuse && d < NEWTON_DEG {
            return None;
        }
        let local;
        let (mu, ml): (&[u64], usize) = if reuse || big.mu.get().is_some() {
            (big.mu.get_or_init(|| mu_top(&big.f, d, d + 1)), d + 1)
        } else {
            let l = lq_all.min(d + 1);
            local = mu_top(&big.f, d, l);
            (&local, l)
        };
        let mut r = a[..da / 64 + 1].to_vec();
        let mut q = if want_q {
            vec![0u64; (da - d) / 64 + 1]
        } else {
            Vec::new()
        };
        while let Some(top) = degree(&r)
            && top >= d
        {
            let s = (top + 1).saturating_sub(2 * d);
            let lq = top + 1 - s - d;
            let a1 = extract(&r, s + d, lq);
            let mh = extract(mu, ml - lq, lq);
            let qb = extract(&mul(&a1, &mh), lq - 1, lq);
            xor_shifted(&mut r, &mul(&qb, &big.f), s);
            if want_q {
                xor_shifted(&mut q, &qb, s);
            }
            trim(&mut r);
        }
        trim(&mut q);
        Some((q, r))
    }

    /// `a mod f`, packed and trimmed.
    pub(crate) fn rem(&self, a: &[u64]) -> Vec<u64> {
        self.rem_in(a, self.reused)
    }

    /// [`rem`](Self::rem), `reuse` for a division kept for many remainders.
    fn rem_in(&self, a: &[u64], reuse: bool) -> Vec<u64> {
        if degree(a).is_none_or(|d| d < self.deg) {
            let mut r = a.to_vec();
            trim(&mut r);
            return r;
        }
        if self.kf == 0 {
            return Vec::new();
        }
        if let Some((_, r)) = self.newton(a, degree(a).expect("not zero"), false, reuse) {
            return r;
        }
        let mut t = shl(a, self.s);
        dispatch(Op::Rem(&mut t, self, None));
        t.truncate(self.kf);
        let mut r = shr(&t, self.s);
        trim(&mut r);
        r
    }

    /// The quotient and the remainder by `f`, packed and trimmed.
    pub(crate) fn div_rem(&self, a: &[u64]) -> (Vec<u64>, Vec<u64>) {
        if degree(a).is_none_or(|d| d < self.deg) {
            let mut r = a.to_vec();
            trim(&mut r);
            return (Vec::new(), r);
        }
        if self.kf == 0 {
            let mut q = a.to_vec();
            trim(&mut q);
            return (q, Vec::new());
        }
        if let Some(qr) = self.newton(a, degree(a).expect("not zero"), true, self.reused) {
            return qr;
        }
        let mut t = shl(a, self.s);
        let mut q = vec![0u64; t.len() - self.kf];
        dispatch(Op::Rem(&mut t, self, Some(&mut q)));
        t.truncate(self.kf);
        let mut r = shr(&t, self.s);
        trim(&mut r);
        trim(&mut q);
        (q, r)
    }

    /// `a·b mod f` for packed `a`, `b`.
    pub(crate) fn mul_mod(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        self.rem(&mul(a, b))
    }

    /// `a² mod f`.
    pub(crate) fn sqr_mod(&self, a: &[u64]) -> Vec<u64> {
        self.rem(&sqr(a))
    }

    /// `b^e mod f` for the exponent's limbs `e` (least significant first), by squares and
    /// products from the top bit; `x^e` (`b = x`) by squares and shifts.
    pub(crate) fn pow(&self, b: &[u64], e: &[u64]) -> Vec<u64> {
        // Many remainders by the same f: as a reused division.
        let rem = |x: &[u64]| self.rem_in(x, true);
        let b = rem(b);
        let bits = e
            .iter()
            .rposition(|&w| w != 0)
            .map(|i| 64 * i + 64 - e[i].leading_zeros() as usize);
        let Some(bits) = bits else {
            return rem(&[1]);
        };
        let is_x = b.len() == 1 && b[0] == 2;
        let mut r = b.clone();
        for i in (0..bits - 1).rev() {
            r = rem(&sqr(&r));
            if (e[i / 64] >> (i % 64)) & 1 == 1 {
                r = if is_x {
                    rem(&shl(&r, 1))
                } else {
                    rem(&mul(&r, &b))
                };
            }
        }
        r
    }
}

/// `a ^= b · x^s`, `a` long enough.
pub(crate) fn xor_shifted(a: &mut [u64], b: &[u64], s: usize) {
    let (ws, bs) = (s / 64, (s % 64) as u32);
    if bs == 0 {
        for (i, &w) in b.iter().enumerate() {
            a[ws + i] ^= w;
        }
        return;
    }
    for (i, &w) in b.iter().enumerate() {
        a[ws + i] ^= w << bs;
        if ws + i + 1 < a.len() {
            a[ws + i + 1] ^= w >> (64 - bs);
        }
    }
}

/// The coefficients `lo..lo + n` of `b` (zeros past it), packed and trimmed: `(b div x^lo) mod
/// x^n`.
pub(crate) fn extract(b: &[u64], lo: usize, n: usize) -> Vec<u64> {
    let (w, sh) = (lo / 64, (lo % 64) as u32);
    let nw = n.div_ceil(64);
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    let mut r: Vec<u64> = (0..nw)
        .map(|i| {
            let hi = if sh == 0 {
                0
            } else {
                at(w + i + 1) << (64 - sh)
            };
            (at(w + i) >> sh) | hi
        })
        .collect();
    if !n.is_multiple_of(64)
        && let Some(t) = r.last_mut()
    {
        *t &= (1u64 << (n % 64)) - 1;
    }
    trim(&mut r);
    r
}

/// The `n` coefficients of `b` below `x^n` in the reverse order, `x^(n−1)·b(1/x) mod x^n`,
/// packed and trimmed.
fn reverse(b: &[u64], n: usize) -> Vec<u64> {
    let nw = n.div_ceil(64);
    let low = extract(b, 0, n);
    let r: Vec<u64> = (0..nw)
        .map(|i| low.get(nw - 1 - i).copied().unwrap_or(0).reverse_bits())
        .collect();
    // The 64·nw coefficients reversed: the n wanted at the top.
    extract(&r, 64 * nw - n, n)
}

/// The inverse of the power series `f` (`f(0) = 1`) modulo `x^n`, `n ≥ 1`, by Newton's
/// iteration, which over GF(2) is `g ← g·(2 − f·g) = f·g² mod x^(2k)`: one square (spread bits)
/// and one product a step.
fn inv_series(f: &[u64], n: usize) -> Vec<u64> {
    if n <= 1 {
        return vec![1];
    }
    let g = inv_series(f, n.div_ceil(2));
    extract(&mul(&extract(f, 0, n), &sqr(&g)), 0, n)
}

/// The top `l ≤ d + 1` coefficients of `μ = floor(x^(2d) / f)` for `f` of the degree `d`, as a
/// polynomial of `l` coefficients (`μ div x^(d+1−l)`): the reversal of the inverse of the
/// reversed `f` modulo `x^l` (`x^(2d) = μ f + ρ`, `deg ρ < d`, read at `1/x`, gives
/// `rev(μ) = rev(f)⁻¹ mod x^(d+1)`, whose low `l` coefficients are `μ`'s top `l`).
fn mu_top(f: &[u64], d: usize, l: usize) -> Vec<u64> {
    reverse(&inv_series(&reverse(f, d + 1), l), l)
}

/// From this degree of both operands the greatest common divisor modulo 2 is the half gcd on the
/// packed products ([`crate::math::poly::gcd`]'s `gf2` engine), below Euclid's algorithm on the
/// packed words (Euclid's `Θ(n²/64)` word operations against the half gcd's `O(M(n) log n)` on
/// Karatsuba's products). Measured on an Apple M4 Max (2,048 to 16,384 against degrees of 1,536
/// to 262,144, with the engine's `HGCD_BASE`): on dense random operands the half gcd is as fast
/// at 2,048 and up to 1.47 times faster from 3,072 to 12,288, but on structured sparse ones
/// Euclid's few long steps win (`x^a − 1` against `x^b − 1` 2.3 to 4.4 times faster, trinomials
/// 1.1 to 1.3 times, up to 6,144), so the crossover stays at 6,144.
pub(crate) const HGCD_GCD: usize = if cfg!(miri) { 300 } else { 6144 };

/// The greatest common divisor (monic, as every non-zero polynomial over GF(2) is), packed and
/// trimmed: [`euclid`] while the shorter operand's degree is below [`HGCD_GCD`], the half gcd
/// from it.
pub(crate) fn gcd(a: &[u64], b: &[u64]) -> Vec<u64> {
    if let (Some(da), Some(db)) = (degree(a), degree(b))
        && da.min(db) >= HGCD_GCD
    {
        return crate::math::poly::gcd::gf2_gcd(a, b);
    }
    euclid(a, b, None, 0)
}

/// Euclid's algorithm on the packed words by shifted subtractions of the shorter from the longer:
/// the last non-zero remainder, packed and trimmed; with `rec`, the degree (plus `shift`) of each
/// divisor (its leading coefficient 1), stopping at a constant one, as the half gcd's engines'
/// `euclid` records them.
pub(crate) fn euclid(
    a: &[u64],
    b: &[u64],
    mut rec: Option<&mut Vec<(usize, u64)>>,
    shift: usize,
) -> Vec<u64> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    trim(&mut a);
    trim(&mut b);
    loop {
        let Some(db) = degree(&b) else {
            return a;
        };
        if let Some(rec) = rec.as_deref_mut() {
            rec.push((db + shift, 1));
            if db == 0 {
                return b;
            }
        }
        reduce(&mut a, &b, db, None);
        core::mem::swap(&mut a, &mut b);
    }
}

/// `a ← a mod b` (`b` of the degree `db`, trimmed) by shifted subtractions, the quotient's
/// coefficients into `q` (long enough) when given; `a` trimmed.
pub(crate) fn reduce(a: &mut Vec<u64>, b: &[u64], db: usize, mut q: Option<&mut [u64]>) {
    let Some(mut da) = degree(a) else {
        a.clear();
        return;
    };
    let lb = b.len();
    while da >= db {
        xor_shifted(a, &b[..lb], da - db);
        if let Some(q) = q.as_deref_mut() {
            q[(da - db) / 64] ^= 1 << ((da - db) % 64);
        }
        match degree(&a[..(da / 64 + 1).min(a.len())]) {
            Some(d) => da = d,
            None => {
                a.clear();
                return;
            }
        }
    }
    trim(a);
}

// The kernels on one word a coefficient, as a `ModPolynomial` modulo 2 holds them.

/// `r = a·b` (one word a coefficient, `r` of `la + lb - 1`), `b` empty for the square of `a`.
pub(crate) fn mul_into_words(r: &mut [u64], a: &[u64], b: &[u64]) {
    let pa = pack(a);
    let p = if b.is_empty() {
        sqr(&pa)
    } else {
        mul(&pa, &pack(b))
    };
    unpack_into(r, &p);
}

/// The quotient and the remainder of `a` by `f` (one word a coefficient), trimmed.
pub(crate) fn div_rem_words(a: &[u64], f: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let (q, r) = Div::new(&pack(f)).div_rem(&pack(a));
    (unpack(&q), unpack(&r))
}

/// The greatest common divisor (one word a coefficient), trimmed.
pub(crate) fn gcd_words(a: &[u64], b: &[u64]) -> Vec<u64> {
    unpack(&gcd(&pack(a), &pack(b)))
}
