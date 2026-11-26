//! The hot loops of products by scalars on fields of at most 256 bits, on elements of four words
//! held inline.
//!
//! The field's elements ([`F2m`]) live on the heap, so that each product or square of the
//! group law allocates, which costs more than the arithmetic itself on small fields: a product
//! of 64 × 64 bits by the public [`Clmul::mul`] takes some 7 ns, a product of the public field at
//! `m = 233` some 110 ns. Here the elements are `[u64; 4]`, the products Karatsuba's on the
//! `n = ⌈m / 64⌉` words (3, 6 and 9 carry-less products for 2, 3 and 4 words), the squares `n` of
//! them (or the bits spread out, under the portable kernel), and the reduction one of the public
//! field's two on the same words, chosen for the field by a rough count of word operations (and
//! kept, with Barrett's constants, for the thread's next products on that field):
//! the sparse reduction, chunks of at most `min(64, m − e₁)` bits from the top (`e₁` the second
//! exponent of `f`) each folded down by one shifted sum per term of `f(x) = xᵐ + Σ xᵏ`, which the
//! trinomials and pentanomials of the standards take, or Barrett's two products by `⌊x²ᵐ / f⌋`,
//! which moduli with a term close to `xᵐ` (many small chunks) and those of many terms take.
//! The formulas are those of [`Ops`](super::arith::Ops), and every step gives the same point (the
//! same López–Dahab coordinates) as there: Horner's rule of [`mul::horner`](super::mul::horner) and
//! the Montgomery ladder of [`mul::x_ladder`](super::mul::x_ladder) run here when the field is
//! narrow enough, their inputs and outputs converted once.

use super::arith::{Ld, Ops};
use super::mul::{Expansion, Step, bit_length, window, wnaf};
use crate::math::field::{Clmul, F2m};
use core::cell::RefCell;

/// An element: the words of a polynomial of degree below `m ≤ 256`, the bits from `m` on zero.
type W = [u64; 4];

/// A double-width product, before its reduction.
type D = [u64; 8];

/// A table of affine points, each with its opposite: `(x, y, x + y)`, `None` at infinity.
type Table = [Option<(W, W, W)>];

/// A point in López–Dahab coordinates, at infinity when `Z = 0`.
#[derive(Clone, Copy)]
struct P {
    x: W,
    y: W,
    z: W,
}

/// What a product by a coefficient costs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Zero,
    One,
    Other,
}

/// A reduction modulo `f` of a product of two elements, of degree below `2m − 1`; the arithmetic
/// is compiled for each, so that its inner loops hold only their own.
pub(super) trait Reduce: Sized {
    fn reduce(arith: &Arith<Self>, t: D) -> W;
}

/// The public field's sparse reduction: chunks of at most `chunk ≤ min(64, m − e₁)` bits from the
/// top, each within a word, folded down by one shifted sum per term of `f` below `xᵐ`.
pub(super) struct Sparse {
    chunk: usize,
}

/// Barrett's reduction, as the public field's: `q = ⌊⌊t / xᵐ⌋·μ / xᵐ⌋` and `t mod f = t + q·f`,
/// exact for `t` of degree below `2m` over `GF(2)[x]`.
#[derive(Clone, Copy)]
pub(super) struct Barrett {
    /// `μ − xᵐ` for `μ = ⌊x²ᵐ / f⌋`.
    mu: W,
    /// `f − xᵐ`.
    g: W,
}

impl Reduce for Sparse {
    #[inline(always)]
    fn reduce(arith: &Arith<Sparse>, t: D) -> W {
        arith.sparse(t, arith.reduction.chunk)
    }
}

impl Reduce for Barrett {
    #[inline(always)]
    fn reduce(arith: &Arith<Barrett>, t: D) -> W {
        arith.barrett(t, &arith.reduction.mu, &arith.reduction.g)
    }
}

/// The arithmetic of a curve over a field of at most 256 bits, its reduction chosen for `f` by
/// [`choose_reduction`] (once for a thread's run of products on one field: see [`LAST`]).
pub(super) enum Narrow {
    Sparse(Arith<Sparse>),
    Barrett(Arith<Barrett>),
}

/// The arithmetic of a curve over a field of at most 256 bits under the reduction `R`.
pub(super) struct Arith<R> {
    clmul: Clmul,
    /// The words of an element, `⌈m / 64⌉`.
    n: usize,
    /// The degree.
    m: usize,
    /// The exponents of `f` below `m`, in decreasing order.
    low: Vec<usize>,
    reduction: R,
    a: W,
    b: W,
    a_kind: Kind,
    b_kind: Kind,
}

/// The words of `x`, `n ≤ 4` of them, the rest zero.
fn words(x: &F2m<'_>) -> W {
    let mut w = [0u64; 4];
    w[..x.words().len()].copy_from_slice(x.words());
    w
}

fn kind(x: &W) -> Kind {
    if *x == [0; 4] {
        Kind::Zero
    } else if *x == [1, 0, 0, 0] {
        Kind::One
    } else {
        Kind::Other
    }
}

#[inline(always)]
fn add(x: &W, y: &W) -> W {
    [x[0] ^ y[0], x[1] ^ y[1], x[2] ^ y[2], x[3] ^ y[3]]
}

/// `x²` over `GF(2)` for one word: its bits spread to the even places of 128 bits.
#[inline(always)]
fn spread_bits(x: u64) -> u128 {
    // The 32 bits of `v` to the even places of 64, by halves.
    let half = |mut v: u64| {
        v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
        v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        (v | (v << 1)) & 0x5555_5555_5555_5555
    };
    u128::from(half(x & 0xffff_ffff)) | (u128::from(half(x >> 32)) << 64)
}

/// The carry-less products of words by a kernel, `PMULL` inline where the target enables it at
/// compile time: the public [`Clmul::mul`] checks its shapes and dispatches out of line, which
/// took k·P on these fields 1.3 to 2.9 times as long under PMULL (measured on an Apple M4 Max),
/// and whether the compiler inlines it into these loops depends on the codegen units.
#[derive(Clone, Copy)]
struct Word(Clmul);

impl Word {
    /// `a·b` over `GF(2)`.
    #[inline(always)]
    fn mul(self, a: u64, b: u64) -> u128 {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes", not(miri)))]
        if self.0 == Clmul::Pmull {
            // SAFETY: the target enables the AES extension at compile time, of which PMULL is part.
            return unsafe { core::arch::aarch64::vmull_p64(a, b) };
        }
        self.0.mul(a, b)
    }
}

#[inline(always)]
fn is_zero(x: &W) -> bool {
    (x[0] | x[1] | x[2] | x[3]) == 0
}

/// `⌊t / xᵐ⌋` for `t` of degree below `2m`: four words, of degree below `m`.
#[inline(always)]
fn high(t: &D, m: usize) -> W {
    let (q, r) = (m / 64, m % 64);
    let mut h = [0u64; 4];
    for (i, h) in h.iter_mut().enumerate() {
        let lo = t[i + q] >> r;
        let hi = if r > 0 && i + q + 1 < 8 {
            t[i + q + 1] << (64 - r)
        } else {
            0
        };
        *h = lo | hi;
    }
    h
}

/// The number of chunks of the sparse reduction of a product, of degree below `2m − 1`, `chunk`
/// bits at most and each within a word, as the public field counts them.
fn sparse_chunks(m: usize, chunk: usize) -> usize {
    let mut top = 2 * m - 1;
    let mut count = 0;
    while top > m {
        top = m.max(top.saturating_sub(chunk)).max((top - 1) / 64 * 64);
        count += 1;
    }
    count
}

/// What [`choose_reduction`] decides for a field: the sparse reduction's chunk, or Barrett's
/// constants.
#[derive(Clone, Copy)]
enum Choice {
    Sparse(usize),
    Barrett(Barrett),
}

/// A field's [`Choice`], with what identifies the field: `m`, its exponents below `m` and its
/// kernel.
struct Kept {
    m: usize,
    low: Vec<usize>,
    clmul: Clmul,
    choice: Choice,
}

std::thread_local! {
    /// The field of this thread's last product by a scalar and its [`Choice`]: a cache of one
    /// entry, so that the products on one field count the chunks and make Barrett's `μ` (an
    /// `m`-step division, up to some 1.4 µs) once, not at each product, which for a short scalar
    /// costs as much as the product.
    static LAST: RefCell<Option<Kept>> = const { RefCell::new(None) };
}

/// [`choose_reduction`] for the field, from [`LAST`] when it is the thread's last one.
fn reduction_of(m: usize, low: &[usize], clmul: Clmul) -> Choice {
    LAST.with_borrow_mut(|last| {
        if let Some(k) = last
            && k.m == m
            && k.clmul == clmul
            && k.low == low
        {
            return k.choice;
        }
        let choice = choose_reduction(m, low, clmul);
        *last = Some(Kept {
            m,
            low: low.to_vec(),
            clmul,
            choice,
        });
        choice
    })
}

/// The reduction for `f = xᵐ + Σ x^low` (`low` decreasing, `m ≤ 256`) under the kernel `clmul`:
/// the sparse one while its chunks cost no more than Barrett's two products. The sparse reduction
/// is counted as the public field's `choose_reduction` (`field/binary.rs`) counts it, `3 + t` for
/// each chunk of a product, `t` the terms below `xᵐ`. Barrett's, on the inline words, costs as much
/// as 20, 21, 21 and 35 of those units for 1 to 4 words under `PMULL`, and 52, 124, 232 and 350
/// under the portable kernel, whose products of words dominate (it makes 1, 3, 6 and 9 of them for
/// a product): less than the field's own count says, which includes the field's slices.
fn choose_reduction(m: usize, low: &[usize], clmul: Clmul) -> Choice {
    let e1 = low.first().copied().unwrap_or(0);
    let chunk = (m - e1).min(64);
    let sparse = sparse_chunks(m, chunk) * (3 + low.len());
    let n = m.div_ceil(64);
    // Measured on an Apple M4 Max: k·P with an m-bit k under each reduction, alternated (21 and
    // 31 samples), on 220 moduli of 61 to 256 bits (trinomials, pentanomials with e₁ from m − 1
    // down to 12, dense moduli of 9 to 65 terms). Each bound lies where Barrett's becomes the
    // faster; a count at which the moduli tried were mixed goes to the sparse one, so that none is
    // slower than under the former estimates `16n + 4` and `36p + 16`: at 21 units for 2 and 3
    // words, where the two are within 2 % but for x¹¹³ + x¹² + x⁷ + x² + 1 (the sparse one 16 %
    // faster), and at 35 for 4 words, where the pentanomials of small terms are 6 % faster by the
    // sparse one. Past the bounds Barrett's takes 0.97 down to 0.49 times the sparse one's time
    // under PMULL. PCLMULQDQ keeps the estimate, not yet measured.
    let barrett = match clmul {
        Clmul::Portable => [52, 124, 232, 350][n - 1],
        Clmul::Pmull => [20, 21, 21, 35][n - 1],
        Clmul::Pclmulqdq => 16 * n + 4,
    };
    if sparse <= barrett {
        return Choice::Sparse(chunk);
    }
    // μ = ⌊x²ᵐ / f⌋ by long division, one bit of the quotient (from xᵐ down) at a time; f is
    // added shifted by whole words and a bit shift, so that a dense f costs no more than a sparse
    // one.
    let mut f = [0u64; 5];
    f[m / 64] |= 1 << (m % 64);
    for &k in low {
        f[k / 64] |= 1 << (k % 64);
    }
    let mut r = [0u64; 9];
    r[2 * m / 64] = 1 << (2 * m % 64);
    let mut mu = [0u64; 5];
    for s in (0..=m).rev() {
        let i = s + m;
        if (r[i / 64] >> (i % 64)) & 1 == 0 {
            continue;
        }
        mu[s / 64] |= 1 << (s % 64);
        let (sq, sr) = (s / 64, s % 64);
        for (j, &w) in f.iter().enumerate() {
            r[j + sq] ^= w << sr;
            if sr > 0 && j + sq + 1 < 9 {
                r[j + sq + 1] ^= w >> (64 - sr);
            }
        }
    }
    debug_assert!((m..=2 * m).all(|i| (r[i / 64] >> (i % 64)) & 1 == 0));
    mu[m / 64] ^= 1 << (m % 64);
    let mut g = [0u64; 4];
    for &k in low {
        g[k / 64] |= 1 << (k % 64);
    }
    Choice::Barrett(Barrett {
        mu: [mu[0], mu[1], mu[2], mu[3]],
        g,
    })
}

/// `$body` on the arithmetic of `$narrow`, whichever its reduction.
macro_rules! on_arith {
    ($narrow:expr, $arith:ident => $body:expr) => {
        match $narrow {
            Narrow::Sparse($arith) => $body,
            Narrow::Barrett($arith) => $body,
        }
    };
}

impl Narrow {
    /// The arithmetic of the curve of `ops`, if its field has at most 256 bits.
    pub(super) fn new(ops: &Ops<'_>) -> Option<Narrow> {
        let modulus = ops.f.modulus();
        let m = *modulus.first()? as usize;
        if m > 256 {
            return None;
        }
        let a = words(&ops.a);
        let b = words(&ops.b);
        let low: Vec<usize> = modulus[1..].iter().map(|&e| e as usize).collect();
        let clmul = ops.f.clmul();
        let choice = reduction_of(m, &low, clmul);
        macro_rules! arith {
            ($reduction:expr) => {
                Arith {
                    clmul,
                    n: m.div_ceil(64),
                    m,
                    low,
                    reduction: $reduction,
                    a_kind: kind(&a),
                    b_kind: kind(&b),
                    a,
                    b,
                }
            };
        }
        Some(match choice {
            Choice::Sparse(chunk) => Narrow::Sparse(arith!(Sparse { chunk })),
            Choice::Barrett(r) => Narrow::Barrett(arith!(r)),
        })
    }

    /// [`mul::mul`](super::mul::mul) on inline elements.
    pub(super) fn mul_naf<'f>(&self, ops: &Ops<'f>, p: &Ld<'f>, k: &[u64]) -> Ld<'f> {
        on_arith!(self, arith => arith.mul_naf(ops, p, k))
    }

    /// The τ-adic product of [`Frobenius::mul`](super::koblitz::Frobenius::mul) on inline
    /// elements.
    pub(super) fn mul_tnaf<'f>(
        &self,
        ops: &Ops<'f>,
        p: &Ld<'f>,
        digits: &[i8],
        alphas: &[(i64, i64)],
    ) -> Ld<'f> {
        on_arith!(self, arith => arith.mul_tnaf(ops, p, digits, alphas))
    }

    /// [`mul::horner`](super::mul::horner) on inline elements.
    pub(super) fn horner<'f>(&self, ops: &Ops<'f>, terms: &[Expansion<'f>], step: Step) -> Ld<'f> {
        on_arith!(self, arith => arith.horner(ops, terms, step))
    }

    /// [`mul::x_ladder`](super::mul::x_ladder) on inline elements.
    pub(super) fn x_ladder<'f>(
        &self,
        ops: &Ops<'f>,
        x: &F2m<'f>,
        k: &[u64],
        bits: u64,
    ) -> Option<F2m<'f>> {
        on_arith!(self, arith => arith.x_ladder(ops, x, k, bits))
    }
}

impl<R: Reduce> Arith<R> {
    /// `t mod f` for a product `t`, of degree below `2m − 1`, by the field's reduction.
    #[inline(always)]
    fn reduce(&self, t: D) -> W {
        R::reduce(self, t)
    }

    /// `t mod f` by Barrett's reduction: `a = ⌊t / xᵐ⌋`, `q = ⌊a·μ / xᵐ⌋ = a + ⌊a·(μ − xᵐ) / xᵐ⌋`,
    /// and the bits of `t + q·(f − xᵐ)` below `m`, as `q·xᵐ` has none there.
    #[inline]
    fn barrett(&self, t: D, mu: &W, g: &W) -> W {
        let m = self.m;
        let a = high(&t, m);
        let q = add(&high(&self.wide(&a, mu), m), &a);
        let p = self.wide(&q, g);
        let mut r = [0u64; 4];
        for i in 0..self.n {
            r[i] = t[i] ^ p[i];
        }
        if !m.is_multiple_of(64) {
            r[self.n - 1] &= (1u64 << (m % 64)) - 1;
        }
        r
    }

    /// `t mod f` for `t` of degree below `2m − 1`, by chunks of at most `chunk ≤ m − e₁` bits from
    /// the top, each within a word, its multiples by the terms below `xᵐ` landing below it.
    #[inline]
    fn sparse(&self, mut t: D, chunk: usize) -> W {
        let m = self.m;
        let mut top = 2 * m - 1;
        while top > m {
            let hi = top - 1;
            let lo = m.max(top.saturating_sub(chunk)).max(hi / 64 * 64);
            let width = top - lo;
            let (wi, sh) = (lo / 64, lo % 64);
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            let x = (t[wi] >> sh) & mask;
            if x != 0 {
                t[wi] &= !(mask << sh);
                for &e in &self.low {
                    // The chunk times x^(lo − m + e), within the words below `lo`.
                    let pos = lo - m + e;
                    let (q, r) = (pos / 64, pos % 64);
                    t[q] ^= x << r;
                    // Only where the chunk crosses into the next word, as the positions alone
                    // say: a sum of zeros there would make the next chunk, often of that word,
                    // wait for it.
                    if r + width > 64 {
                        t[q + 1] ^= x >> (64 - r);
                    }
                }
            }
            top = lo;
        }
        [t[0], t[1], t[2], t[3]]
    }

    /// `x·y mod f`.
    #[inline]
    fn mul(&self, x: &W, y: &W) -> W {
        self.reduce(self.wide(x, y))
    }

    /// `x·y` before its reduction: Karatsuba's three carry-less products for two words, its two
    /// levels (nine) for four, the six of its three-term form for three; the schoolbook otherwise.
    #[inline(always)]
    fn wide(&self, x: &W, y: &W) -> D {
        let c = Word(self.clmul);
        // The 256-bit product of two words each, by three products.
        let two = |a0: u64, a1: u64, b0: u64, b1: u64| -> [u64; 4] {
            let p0 = c.mul(a0, b0);
            let p2 = c.mul(a1, b1);
            let p1 = c.mul(a0 ^ a1, b0 ^ b1) ^ p0 ^ p2;
            [
                p0 as u64,
                (p0 >> 64) as u64 ^ p1 as u64,
                (p1 >> 64) as u64 ^ p2 as u64,
                (p2 >> 64) as u64,
            ]
        };
        // `v` at word `at` of `t`.
        let put = |t: &mut D, at: usize, v: u128| {
            t[at] ^= v as u64;
            t[at + 1] ^= (v >> 64) as u64;
        };
        let mut t = [0u64; 8];
        match self.n {
            2 => t[..4].copy_from_slice(&two(x[0], x[1], y[0], y[1])),
            3 => {
                let (p0, p1, p2) = (c.mul(x[0], y[0]), c.mul(x[1], y[1]), c.mul(x[2], y[2]));
                let q01 = c.mul(x[0] ^ x[1], y[0] ^ y[1]);
                let q02 = c.mul(x[0] ^ x[2], y[0] ^ y[2]);
                let q12 = c.mul(x[1] ^ x[2], y[1] ^ y[2]);
                put(&mut t, 0, p0);
                put(&mut t, 1, q01 ^ p0 ^ p1);
                put(&mut t, 2, q02 ^ p0 ^ p1 ^ p2);
                put(&mut t, 3, q12 ^ p1 ^ p2);
                put(&mut t, 4, p2);
            }
            4 => {
                let lo = two(x[0], x[1], y[0], y[1]);
                let hi = two(x[2], x[3], y[2], y[3]);
                let mid = two(x[0] ^ x[2], x[1] ^ x[3], y[0] ^ y[2], y[1] ^ y[3]);
                for i in 0..4 {
                    t[i] ^= lo[i];
                    t[i + 4] ^= hi[i];
                    t[i + 2] ^= mid[i] ^ lo[i] ^ hi[i];
                }
            }
            n => {
                for (i, &xi) in x.iter().enumerate().take(n) {
                    for (j, &yj) in y.iter().enumerate().take(n) {
                        put(&mut t, i + j, c.mul(xi, yj));
                    }
                }
            }
        }
        t
    }

    /// `x²`: the squares of the words, no cross terms; by the instruction, or by spreading the
    /// bits under the portable kernel, whose product costs many times more.
    #[inline]
    fn square(&self, x: &W) -> W {
        let mut t = [0u64; 8];
        let spread = self.clmul == Clmul::Portable;
        for i in 0..self.n {
            let p = if spread {
                spread_bits(x[i])
            } else {
                Word(self.clmul).mul(x[i], x[i])
            };
            t[2 * i] = p as u64;
            t[2 * i + 1] = (p >> 64) as u64;
        }
        self.reduce(t)
    }

    fn times_a(&self, t: &W) -> W {
        match self.a_kind {
            Kind::Zero => [0; 4],
            Kind::One => *t,
            Kind::Other => self.mul(&self.a, t),
        }
    }

    fn times_b(&self, t: &W) -> W {
        match self.b_kind {
            Kind::Zero => [0; 4],
            Kind::One => *t,
            Kind::Other => self.mul(&self.b, t),
        }
    }

    const IDENTITY: P = P {
        x: [1, 0, 0, 0],
        y: [0; 4],
        z: [0; 4],
    };

    /// `2p`, as [`Ops::double`].
    fn double(&self, p: &P) -> P {
        if is_zero(&p.z) {
            return *p;
        }
        let x2 = self.square(&p.x);
        let z2 = self.square(&p.z);
        let z3 = self.mul(&x2, &z2);
        let bz4 = self.times_b(&self.square(&z2));
        let x3 = add(&self.square(&x2), &bz4);
        let inner = add(&add(&self.times_a(&z3), &self.square(&p.y)), &bz4);
        let y3 = add(&self.mul(&bz4, &z3), &self.mul(&x3, &inner));
        P {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `τ(p) = (X² : Y² : Z²)`.
    fn frobenius(&self, p: &P) -> P {
        P {
            x: self.square(&p.x),
            y: self.square(&p.y),
            z: self.square(&p.z),
        }
    }

    /// `p + (x₂, y₂)` for an affine point, as [`Ops::add_affine`].
    fn add_affine(&self, p: &P, qx: &W, qy: &W) -> P {
        if is_zero(&p.z) {
            return P {
                x: *qx,
                y: *qy,
                z: [1, 0, 0, 0],
            };
        }
        let z1s = self.square(&p.z);
        let a = add(&p.y, &self.mul(qy, &z1s));
        let b = add(&p.x, &self.mul(qx, &p.z));
        if is_zero(&b) {
            return if is_zero(&a) {
                self.double(p)
            } else {
                Self::IDENTITY
            };
        }
        let c = self.mul(&p.z, &b);
        let d = self.mul(&self.square(&b), &add(&c, &self.times_a(&z1s)));
        let z3 = self.square(&c);
        let e = self.mul(&a, &c);
        let x3 = add(&add(&self.square(&a), &d), &e);
        let f = add(&x3, &self.mul(qx, &z3));
        let g = self.mul(&add(qx, qy), &self.square(&z3));
        let y3 = add(&self.mul(&add(&e, &z3), &f), &g);
        P {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `−p = (X : X·Z + Y : Z)`.
    fn neg(&self, p: &P) -> P {
        if is_zero(&p.z) {
            return *p;
        }
        P {
            x: p.x,
            y: add(&self.mul(&p.x, &p.z), &p.y),
            z: p.z,
        }
    }

    /// `p + q`, as [`Ops::add`].
    fn add(&self, p: &P, q: &P) -> P {
        if is_zero(&p.z) {
            return *q;
        }
        if is_zero(&q.z) {
            return *p;
        }
        let z1s = self.square(&p.z);
        let z2s = self.square(&q.z);
        let a1 = self.mul(&p.y, &z2s);
        let a = add(&a1, &self.mul(&q.y, &z1s));
        let b1 = self.mul(&p.x, &q.z);
        let b = add(&b1, &self.mul(&q.x, &p.z));
        if is_zero(&b) {
            return if is_zero(&a) {
                self.double(p)
            } else {
                Self::IDENTITY
            };
        }
        let f = self.mul(&p.z, &q.z);
        let c = self.mul(&f, &b);
        let d = self.mul(&c, &b);
        let ac = self.mul(&a, &c);
        let x3 = add(
            &add(&self.square(&a), &ac),
            &self.mul(&d, &add(&b, &self.times_a(&f))),
        );
        let c2 = self.square(&c);
        let y3 = add(
            &add(
                &self.mul(&ac, &add(&self.mul(&b1, &d), &x3)),
                &self.mul(&x3, &c2),
            ),
            &self.mul(&a1, &self.square(&d)),
        );
        P {
            x: x3,
            y: y3,
            z: c2,
        }
    }

    /// The affine points `(x, y, x + y)` of `pts`, `None` at infinity, with one inversion of the
    /// public field for them all, as [`Ops::affine_all`].
    fn affine_all(&self, ops: &Ops<'_>, pts: &[P]) -> Vec<Option<(W, W, W)>> {
        let mut running = Vec::with_capacity(pts.len());
        let mut product: W = [1, 0, 0, 0];
        for q in pts {
            running.push(product);
            if !is_zero(&q.z) {
                product = self.mul(&product, &q.z);
            }
        }
        let n = self.n;
        let mut inverse = words(
            &ops.f
                .element(&product[..n])
                .inverse()
                .unwrap_or_else(|| unreachable!("a product of elements that are not zero")),
        );
        let mut out = vec![None; pts.len()];
        for (i, q) in pts.iter().enumerate().rev() {
            if is_zero(&q.z) {
                continue;
            }
            let zi = self.mul(&inverse, &running[i]);
            inverse = self.mul(&inverse, &q.z);
            let x = self.mul(&q.x, &zi);
            let y = self.mul(&q.y, &self.square(&zi));
            out[i] = Some((x, y, add(&x, &y)));
        }
        out
    }

    /// The point of `ops` in inline coordinates.
    fn of_ld(p: &Ld<'_>) -> P {
        P {
            x: words(&p.x),
            y: words(&p.y),
            z: words(&p.z),
        }
    }

    /// Horner's rule on digits and tables of affine points with their opposites.
    fn run(&self, terms: &[(&[i8], &Table)], step: Step) -> P {
        let len = terms.iter().map(|t| t.0.len()).max().unwrap_or(0);
        let mut r = Self::IDENTITY;
        for i in (0..len).rev() {
            if !is_zero(&r.z) {
                r = match step {
                    Step::Double => self.double(&r),
                    Step::Frobenius => self.frobenius(&r),
                };
            }
            for (digits, table) in terms {
                let d = digits.get(i).copied().unwrap_or(0);
                if d != 0
                    && let Some((x, y, ny)) = &table[(d.unsigned_abs() as usize - 1) / 2]
                {
                    r = self.add_affine(&r, x, if d < 0 { ny } else { y });
                }
            }
        }
        r
    }

    /// [`mul::mul`](super::mul::mul): `k·p` for the magnitude `k` by its non-adjacent form, the
    /// odd multiples of `p` computed here.
    fn mul_naf<'f>(&self, ops: &Ops<'f>, p: &Ld<'f>, k: &[u64]) -> Ld<'f> {
        let w = window(bit_length(k));
        let digits = wnaf(k, w);
        let count = 1usize << (w - 2);
        let q = Self::of_ld(p);
        let mut all = Vec::with_capacity(count);
        all.push(q);
        if count > 1 {
            let two = self.double(&q);
            for i in 1..count {
                let next = self.add(&all[i - 1], &two);
                all.push(next);
            }
        }
        let table = self.affine_all(ops, &all);
        self.to_ld(ops, &self.run(&[(&digits, &table)], Step::Double))
    }

    /// The τ-adic product of [`Frobenius::mul`](super::koblitz::Frobenius::mul) for its digits
    /// and digit set: the table `α_u·P = β·P + γ·τ(P)` from the multiples of `P` and `τ(P)`.
    fn mul_tnaf<'f>(
        &self,
        ops: &Ops<'f>,
        p: &Ld<'f>,
        digits: &[i8],
        alphas: &[(i64, i64)],
    ) -> Ld<'f> {
        let largest = alphas
            .iter()
            .map(|&(b, g)| b.unsigned_abs().max(g.unsigned_abs()))
            .max()
            .unwrap_or(1) as usize;
        let multiples = |q: P| {
            let mut all = vec![Self::IDENTITY, q];
            for j in 2..=largest {
                let next = self.add(&all[j - 1], &q);
                all.push(next);
            }
            all
        };
        let q = Self::of_ld(p);
        let (of_p, of_tp) = (multiples(q), multiples(self.frobenius(&q)));
        let signed = |all: &[P], c: i64| {
            let q = all[c.unsigned_abs() as usize];
            if c < 0 { self.neg(&q) } else { q }
        };
        let points: Vec<P> = alphas
            .iter()
            .map(|&(b, g)| self.add(&signed(&of_p, b), &signed(&of_tp, g)))
            .collect();
        let table = self.affine_all(ops, &points);
        self.to_ld(ops, &self.run(&[(digits, &table)], Step::Frobenius))
    }

    /// The point of the inline coordinates as one of `ops`.
    fn to_ld<'f>(&self, ops: &Ops<'f>, p: &P) -> Ld<'f> {
        let n = self.n;
        Ld {
            x: ops.f.element(&p.x[..n]),
            y: ops.f.element(&p.y[..n]),
            z: ops.f.element(&p.z[..n]),
        }
    }

    /// [`mul::horner`](super::mul::horner) on inline elements.
    fn horner<'f>(&self, ops: &Ops<'f>, terms: &[Expansion<'f>], step: Step) -> Ld<'f> {
        // The tables' affine points, each with its opposite (x, x + y).
        let tables: Vec<Vec<Option<(W, W, W)>>> = terms
            .iter()
            .map(|t| {
                t.table
                    .iter()
                    .map(|e| {
                        e.as_ref().map(|a| {
                            let (x, y) = (words(&a.x), words(&a.y));
                            (x, y, add(&x, &y))
                        })
                    })
                    .collect()
            })
            .collect();
        let runs: Vec<(&[i8], &Table)> = terms
            .iter()
            .zip(&tables)
            .map(|(t, table)| (t.digits.as_slice(), table.as_slice()))
            .collect();
        self.to_ld(ops, &self.run(&runs, step))
    }

    /// [`mul::x_ladder`](super::mul::x_ladder) on inline elements: `(X : Z)` of `jP` and
    /// `(j + 1)P`, the bits of `k` from the second highest down.
    fn x_ladder<'f>(&self, ops: &Ops<'f>, x: &F2m<'f>, k: &[u64], bits: u64) -> Option<F2m<'f>> {
        let xw = words(x);
        let double = |p: &(W, W)| {
            let x2 = self.square(&p.0);
            let z2 = self.square(&p.1);
            (
                add(&self.square(&x2), &self.times_b(&self.square(&z2))),
                self.mul(&x2, &z2),
            )
        };
        let sum = |p: &(W, W), q: &(W, W)| {
            let t = self.mul(&p.0, &q.1);
            let u = self.mul(&q.0, &p.1);
            let z = self.square(&add(&t, &u));
            (add(&self.mul(&xw, &z), &self.mul(&t, &u)), z)
        };
        let mut p1 = (xw, [1, 0, 0, 0]);
        let mut p2 = double(&p1);
        for i in (0..bits - 1).rev() {
            if (k[(i / 64) as usize] >> (i % 64)) & 1 == 1 {
                p1 = sum(&p1, &p2);
                p2 = double(&p2);
            } else {
                p2 = sum(&p1, &p2);
                p1 = double(&p1);
            }
        }
        let n = self.n;
        let zi = ops.f.element(&p1.1[..n]).inverse()?;
        Some(&ops.f.element(&p1.0[..n]) * &zi)
    }
}
