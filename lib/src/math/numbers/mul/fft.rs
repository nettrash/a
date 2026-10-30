//! Products by number-theoretic transforms modulo primes below 2^29, for the longest operands.
//!
//! # The method
//!
//! An operand is cut into coefficients of `w` bits, a polynomial in `X = 2^w` whose value is the
//! number; the product is the value of the product polynomial, whose coefficients are the
//! convolution of the operands' ones. The convolution is computed modulo `np` primes, each by a
//! cyclic transform of length `L = 2^k`, at least the number of coefficients of the product: the
//! transforms of the operands are multiplied pointwise, and the inverse transform gives the
//! convolution modulo the prime. Each coefficient of the product, at most `m (2^w - 1)^2` for a
//! shorter operand of `m` coefficients, is then found from its residues by the Chinese remainder
//! theorem, exactly while it is below half the product `P` of the primes, and the coefficients
//! are added at their positions, `w` bits apart.
//!
//! The primes are the eight largest of the form `c 2^21 + 1` below `2^29`, so that a residue fits
//! a 32-bit lane with room for the sums of a few butterflies, and every transform of up to `2^21`
//! points has its roots of unity. With `np` of them, `P` has 115 bits for 4 primes and 229 for 8,
//! and the coefficients have 48 to 104 bits: [`CONFIGS`] pairs each `np` with a width it makes
//! exact, and [`plan`] picks the pair and the length `L` that cost least by measured constants,
//! as `w` decides how many coefficients there are, and so whether they fit a length or need the
//! next one, twice as long. The longest transform takes operands of up to 11 MiB each, or a
//! longer one of up to 26 MiB by a short one.
//!
//! # The arithmetic
//!
//! A residue is a signed 32-bit integer, any representative of its class that the bounds below
//! allow. A product by a constant `w` takes three operations, after Becker, Hwang, Kannwischer,
//! Yang and Yang ("Neon NTT", 2022): with `w' = round(w 2^31 / q)` stored beside `w`,
//! `t = round(a w' / 2^31)` (`sqrdmulh`) and `r = a w - t q` (`mul`, `mls`), modulo `2^32`, which
//! is exact as `|r| <= q / 2 + |a| q / 2^32 < q` for any `|a| < 2^31`. A reduction by `q`
//! (`sqdmulh`, a rounding shift and `mls`) brings any value to at most `q / 2 + 4`, and products of
//! two variables are taken by signed Montgomery multiplication (Seiler, 2018), five operations
//! for `a b 2^-32` with `|r| < |a b| / 2^32 + q / 2`. All of them work on four lanes at once on
//! aarch64 (NEON), and on four `i32`s in plain Rust elsewhere.
//!
//! # The transforms
//!
//! The forward transform splits `x mod (X^L - 1)` into its residues modulo `X - r` for the `L`-th
//! roots of unity `r`, by Cooley and Tukey's butterflies `(a, b) -> (a + z b, a - z b)` on blocks,
//! where every block of every level has one twiddle `z`: with the roots in bit-reversed order,
//! the twiddle of block `b` is the same `Z[b]` at every level, `Z[0] = 1` and
//! `Z[2^i + c] = Z[c] w_(2^(i+2))` for a root `w_(2^(i+2))` of order `2^(i+2)`, so one table of
//! `L / 2` twiddles, built by that doubling, serves all levels. The inverse runs Gentleman and
//! Sande's butterflies `(a, b) -> (a + b, (a - b) / z)` from the last level to the first, with the
//! table of inverses, `Z[b]^-1 = -Z[3 2^i - 1 - b]` for `b` in `[2^i, 2^(i+1))`: a reversed copy.
//!
//! Levels go two at a time, four vectors of four lanes loaded and stored once per two levels, and
//! the last four levels of each block of 16 run in registers, the vectors transposed for the two
//! levels whose butterflies lie within a vector; the transform is left in that transposed order,
//! which the pointwise product does not mind and the inverse starts from. The first level (two
//! for an even `k`, so that the rest pair up) runs while the residues of the coefficients are
//! formed, where the upper half of a short operand is zero, and the last inverse ones while the
//! residues are scaled for the Chinese remainder theorem. A square takes one forward transform.
//!
//! Values are not reduced after every butterfly. Measured in units of the largest prime and with
//! `beta = q / 2^32 <= 0.1094`: a residue formed from the coefficient's digits is at most
//! `1.75 q`, and the forward transform reduces the first of the four vectors of each pair of
//! levels, which keeps every value at most `1.5547 q / (1 - beta (2 + beta)) = 2.03 q`; the
//! pointwise product is then below `0.95 q`, and the inverse, which reduces the two sums of the
//! first of its two levels, stays at most `q / (1 - 4 beta) = 1.78 q`, its sums of two below
//! `3.56 q`. Every value stays below `4.57 q = 2^31`.
//!
//! # The Chinese remainder theorem
//!
//! The last inverse levels multiply each residue `r_i` by `mu_i = 2^32 L^-1 (P / q_i)^-1`, which
//! takes out the factors of the Montgomery products and of the transforms, and normalizes it to
//! `y_i` in `[0, q_i)`. Then `c = sum y_i (P / q_i) - k P` with `k = floor(sum y_i / q_i)`: the
//! sum of fractions is `k + c / P` with `c / P < 1 / 2`, so `k` is
//! `floor((sum y_i floor(2^60 / q_i) + 2^58) / 2^60)`, whose error, below `2^32 / 2^60`, the
//! quarter added absorbs. `c` is computed modulo `2^(32 nd)`, `nd` digits of 32 bits that hold
//! `P`, by widening multiply-adds of the digits of `P / q_i` and of `-P` (four coefficients at a
//! time on aarch64), and the limbs of eight coefficients, `8 w` bits apart, are added into the
//! product, each limb of the product taking the pieces that land on it in one sum.
//!
//! # Memory
//!
//! The transforms, the twiddles and the digits of the coefficients take
//! `(np + 3) L + dn (cx + cy)` 32-bit words for a product and `(np + 2) L + dn cx` for a square
//! (`dn` digits of a coefficient, `cx` and `cy` coefficients), about 4 to 10 times the limbs of
//! the product: more than the scratch the other methods plan for, so they are allocated here, in
//! one buffer a product.

// The lanes and the Chinese remainder theorem of four coefficients, one set per target under
// the name `arch`, chosen as the kernels of the multiplication are.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
))]
#[path = "fft/aarch64.rs"]
mod arch;
#[cfg(not(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
)))]
#[path = "fft/portable.rs"]
mod arch;

use arch::{V, four};

pub(super) use arch::{FFT_MUL, FFT_MUL_UNBALANCED, FFT_SQR};

// Primes and their constants

/// The primes, the eight largest of the form `c 2^21 + 1` below `2^29`, the largest first, and a
/// generator of the multiplicative group of each.
const PRIMES: [(u32, u32); 8] = [
    (469_762_049, 3),
    (463_470_593, 3),
    (459_276_289, 11),
    (415_236_097, 5),
    (387_973_121, 6),
    (383_778_817, 5),
    (377_487_361, 7),
    (274_726_913, 3),
];

/// The longest transform, `2^21` points: `2^21` divides `q - 1` for every prime.
const MAX_K: u32 = 21;

/// The shortest transform, `2^5` points: the last four levels run on blocks of 16, after the
/// first one.
const MIN_K: u32 = 5;

/// `b^e mod q`.
const fn pow_mod(b: u64, mut e: u64, q: u64) -> u64 {
    let (mut r, mut x) = (1, b % q);
    while e > 0 {
        if e & 1 == 1 {
            r = r * x % q;
        }
        x = x * x % q;
        e >>= 1;
    }
    r
}

/// `x mod q` as the representative in `(-q/2, q/2]`.
const fn centred(x: u64, q: u64) -> i32 {
    let x = x % q;
    if x > q / 2 {
        (x as i64 - q as i64) as i32
    } else {
        x as i32
    }
}

/// The constant of a product by `w` modulo `q`: `round(w 2^31 / q)`, for `|w| < q`.
const fn barrett(w: i32, q: u32) -> i32 {
    let (x, q) = ((w as i64).unsigned_abs() << 31, q as u64);
    let r = ((x + q / 2) / q) as i64;
    (if w < 0 { -r } else { r }) as i32
}

/// A prime of the transforms and the constants of its arithmetic.
#[derive(Clone, Copy)]
pub(super) struct Prime {
    /// The prime.
    pub(super) q: i32,
    /// `q^-1 mod 2^32`, for Montgomery products.
    pub(super) qinv: i32,
    /// `round(2^59 / q)`, for reductions: a product by it, halved 28 more times, is the quotient.
    pub(super) v: i32,
}

/// The constants that depend on one prime only.
struct PrimeData {
    /// The prime.
    prime: Prime,
    /// A root of unity of order `2^j` and its constant for products, for `j <= 21`.
    roots: [(i32, i32); MAX_K as usize + 1],
    /// `2^-k mod q`, for `k <= 21`.
    inv2k: [i32; MAX_K as usize + 1],
    /// `2^e mod q` and its constant for products, for the digits of a coefficient.
    pow2: [(i32, i32); 112],
}

/// The constants of the prime `q` with the generator `g`.
const fn prime_data(q: u32, g: u32) -> PrimeData {
    let qq = q as u64;
    let mut inv: u32 = 1;
    let mut i = 0;
    while i < 5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        i += 1;
    }
    let prime = Prime {
        q: q as i32,
        qinv: inv as i32,
        v: (((1u64 << 59) + qq / 2) / qq) as i32,
    };
    let root = pow_mod(g as u64, (qq - 1) >> MAX_K, qq);
    let mut roots = [(0, 0); MAX_K as usize + 1];
    let mut inv2k = [0; MAX_K as usize + 1];
    let mut j = 0;
    while j <= MAX_K as usize {
        let w = centred(pow_mod(root, 1 << (MAX_K as usize - j), qq), qq);
        roots[j] = (w, barrett(w, q));
        inv2k[j] = centred(pow_mod(qq.div_ceil(2), j as u64, qq), qq);
        j += 1;
    }
    let mut pow2 = [(0, 0); 112];
    let mut e = 0;
    while e < 112 {
        let w = centred(pow_mod(2, e as u64, qq), qq);
        pow2[e] = (w, barrett(w, q));
        e += 1;
    }
    PrimeData {
        prime,
        roots,
        inv2k,
        pow2,
    }
}

/// The constants of the primes, computed when the crate is compiled.
static PRIME_DATA: [PrimeData; 8] = {
    let mut i = 0;
    let mut data = [const { prime_data(3, 2) }; 8];
    while i < 8 {
        data[i] = prime_data(PRIMES[i].0, PRIMES[i].1);
        i += 1;
    }
    data
};

/// The constants of the Chinese remainder theorem for the first `np` primes.
pub(super) struct Crt {
    /// The number of primes.
    pub(super) np: usize,
    /// The 32-bit digits that hold `P`, the product of the primes.
    pub(super) nd: usize,
    /// The digits of `P / q_i mod 2^(32 nd)`, the least significant first.
    pub(super) m: [[u32; 8]; 8],
    /// The digits of `-P mod 2^(32 nd)`.
    pub(super) pneg: [u32; 8],
    /// `floor(2^60 / q_i)`, below `2^32`.
    pub(super) ci: [u32; 8],
    /// `2^32 (P / q_i)^-1 mod q_i`, without the factor `L^-1`.
    mu: [i32; 8],
    /// The bits of `P`.
    bits: u32,
}

/// `a * b` for the 8 digits of `a`, modulo `2^256`.
const fn mul_digits(a: [u32; 8], b: u32) -> [u32; 8] {
    let mut r = [0; 8];
    let mut carry = 0u64;
    let mut i = 0;
    while i < 8 {
        let t = a[i] as u64 * b as u64 + carry;
        r[i] = t as u32;
        carry = t >> 32;
        i += 1;
    }
    r
}

/// The constants of the Chinese remainder theorem for the first `np` primes, `2 <= np <= 8`.
const fn crt_data(np: usize) -> Crt {
    let mut p = [0u32; 8];
    p[0] = 1;
    let mut i = 0;
    while i < np {
        p = mul_digits(p, PRIMES[i].0);
        i += 1;
    }
    let mut top = 7;
    while p[top] == 0 {
        top -= 1;
    }
    let bits = 32 * top as u32 + 32 - p[top].leading_zeros();
    let nd = bits.div_ceil(32) as usize;
    // -P = !P + 1, on nd digits.
    let mut pneg = [0u32; 8];
    let mut carry = 1u64;
    let mut d = 0;
    while d < nd {
        let t = (!p[d]) as u64 + carry;
        pneg[d] = t as u32;
        carry = t >> 32;
        d += 1;
    }
    let mut m = [[0u32; 8]; 8];
    let mut ci = [0u32; 8];
    let mut mu = [0i32; 8];
    i = 0;
    while i < np {
        let q = PRIMES[i].0 as u64;
        let mut v = [0u32; 8];
        v[0] = 1;
        let mut rem = 1u64;
        let mut j = 0;
        while j < np {
            if j != i {
                v = mul_digits(v, PRIMES[j].0);
                rem = rem * (PRIMES[j].0 as u64 % q) % q;
            }
            j += 1;
        }
        d = nd;
        while d < 8 {
            v[d] = 0;
            d += 1;
        }
        m[i] = v;
        ci[i] = ((1u64 << 60) / q) as u32;
        mu[i] = centred(pow_mod(rem, q - 2, q) * ((1u64 << 32) % q), q);
        i += 1;
    }
    Crt {
        np,
        nd,
        m,
        pneg,
        ci,
        mu,
        bits,
    }
}

/// The constants of the Chinese remainder theorem, by the number of primes.
static CRT: [Crt; 9] = {
    let mut c = [const { crt_data(2) }; 9];
    let mut np = 4;
    while np <= 8 {
        c[np] = crt_data(np);
        np += 1;
    }
    c
};

// Lanes

/// Four residues, with the operations of the transforms on them: NEON vectors on aarch64, four
/// `i32`s elsewhere, which compute the same values.
pub(super) trait Lanes: Copy {
    /// The constants of a prime, as the lanes use them.
    type C: Copy;
    /// The constants of `p`.
    fn consts(p: &Prime) -> Self::C;
    /// Four values at `p`.
    ///
    /// # Safety
    ///
    /// The 4 `i32`s at `p` must be valid for reads.
    unsafe fn load(p: *const i32) -> Self;
    /// Stores the four values at `p`.
    ///
    /// # Safety
    ///
    /// The 4 `i32`s at `p` must be valid for writes.
    unsafe fn store(self, p: *mut i32);
    /// The four values at `p`, `p + 2`, `p + 4`, `p + 6`, and those after each: four pairs, the
    /// twiddles and their constants of a table.
    ///
    /// # Safety
    ///
    /// The 8 `i32`s at `p` must be valid for reads.
    unsafe fn load2(p: *const i32) -> (Self, Self);
    /// The 16 values at `p` by their index modulo 4: eight pairs, the twiddles and constants of
    /// the even ones, then of the odd ones.
    ///
    /// # Safety
    ///
    /// The 16 `i32`s at `p` must be valid for reads.
    unsafe fn load4(p: *const i32) -> (Self, Self, Self, Self);
    /// Stores `a` and `b` interleaved at `p`, the inverse of [`Lanes::load2`].
    ///
    /// # Safety
    ///
    /// The 8 `i32`s at `p` must be valid for writes.
    unsafe fn store2(a: Self, b: Self, p: *mut i32);
    /// The two values at `p` in lanes 0 and 1: a twiddle and its constant.
    ///
    /// # Safety
    ///
    /// The 2 `i32`s at `p` must be valid for reads.
    unsafe fn pair(p: *const i32) -> Self;
    /// `x` in every lane.
    fn splat(x: i32) -> Self;
    /// The sums, modulo `2^32`.
    fn add(self, o: Self) -> Self;
    /// The differences, modulo `2^32`.
    fn sub(self, o: Self) -> Self;
    /// The products by the twiddles `w`, with their constants `wp`, lane by lane.
    fn mul(self, w: Self, wp: Self, c: Self::C) -> Self;
    /// The products by the twiddle in lane `LW` of `v`, its constant in lane `LP`.
    fn mul_lanes<const LW: i32, const LP: i32>(self, v: Self, c: Self::C) -> Self;
    /// The values reduced to at most `q / 2 + 4`.
    fn reduce(self, c: Self::C) -> Self;
    /// The Montgomery products `a b 2^-32`.
    fn mont(self, o: Self, c: Self::C) -> Self;
    /// The 4 by 4 transpose: lane `c` of vector `e` becomes lane `e` of vector `c`.
    fn transpose(v: [Self; 4]) -> [Self; 4];
    /// The values of `(-q, q)` moved to `[0, q)`.
    fn normalize(self, c: Self::C) -> Self;
    /// `round(w f)` for `|w| < q` and `f = 2^31 / q`: the constants of twiddles.
    fn constants(self, f: f64) -> Self;
    /// The lanes in reverse order.
    fn rev(self) -> Self;
    /// The negations.
    fn neg(self) -> Self;
}

// Twiddles

/// Builds the twiddles of a transform of `2^k` points modulo the prime of `d` at `t`: the pairs
/// `Z[b]`, `Z'[b]` for `b < 2^(k-1)`, then the inverses.
///
/// # Safety
///
/// `k >= MIN_K`; the `2^(k+1)` `i32`s at `t` must be valid for writes.
unsafe fn twiddles<S: Lanes>(t: *mut i32, k: u32, d: &PrimeData) {
    let half = 1usize << (k - 1);
    let (q, c) = (d.prime.q as i64, S::consts(&d.prime));
    let f = (1u64 << 31) as f64 / q as f64;
    // SAFETY: every write lies in the `2 half` pairs at `t`.
    unsafe {
        let (fw, inv) = (t, t.add(2 * half));
        // Z[0..4] = 1, w4, w8, w4 w8.
        let (w4, w8) = (d.roots[2].0 as i64, d.roots[3].0 as i64);
        for (i, z) in [1, w4, w8, w4 * w8].into_iter().enumerate() {
            let z = centred(z.rem_euclid(q) as u64, q as u64);
            *fw.add(2 * i) = z;
            *fw.add(2 * i + 1) = barrett(z, q as u32);
        }
        let mut i = 2;
        while (1usize << i) < half {
            let (w, wp) = d.roots[i + 2];
            let (w, wp) = (S::splat(w), S::splat(wp));
            let n = 1usize << i;
            for j in (0..n).step_by(4) {
                let (z, _) = S::load2(fw.add(2 * j));
                let z = z.mul(w, wp, c);
                S::store2(z, z.constants(f), fw.add(2 * (n + j)));
            }
            i += 1;
        }
        // The inverses: Z[0]^-1 = 1, Z[1]^-1 = -Z[1], and Z[2^i + c]^-1 = -Z[2^(i+1) - 1 - c].
        *inv = *fw;
        *inv.add(1) = *fw.add(1);
        for (b, m) in [(1, 1), (2, 3), (3, 2)] {
            *inv.add(2 * b) = -*fw.add(2 * m);
            *inv.add(2 * b + 1) = -*fw.add(2 * m + 1);
        }
        let mut n = 4;
        while n < half {
            for j in (0..n).step_by(4) {
                let (z, zp) = S::load2(fw.add(2 * (2 * n - 4 - j)));
                S::store2(z.rev().neg(), zp.rev().neg(), inv.add(2 * (n + j)));
            }
            n *= 2;
        }
    }
}

// The transforms

/// The forward levels `s` and `s + 1` on the `l` values at `a`, with the twiddles at `w`: blocks
/// of `m = l >> s`, butterflies of `m / 2` and of `m / 4` apart, the first of each four vectors
/// reduced.
///
/// # Safety
///
/// `l >> s >= 16`; the `l` values at `a` must be valid for reads and writes, and the twiddles of
/// the blocks, `2 (l >> s)` pairs, at `w` for reads.
#[inline(always)]
unsafe fn fwd4<S: Lanes>(a: *mut i32, l: usize, s: u32, w: *const i32, c: S::C) {
    let m = l >> s;
    let qr = m >> 2;
    // SAFETY: block `b` is `a[b m..(b + 1) m]`, its twiddles `Z[b]`, `Z[2b]` and `Z[2b + 1]` lie
    // below `2^(s+1)`.
    unsafe {
        for b in 0..(1usize << s) {
            let (z1, z23) = (S::pair(w.add(2 * b)), S::load(w.add(4 * b)));
            let base = a.add(b * m);
            for j in (0..qr).step_by(4) {
                let p0 = base.add(j);
                let a0 = S::load(p0).reduce(c);
                let a1 = S::load(p0.add(qr));
                let t2 = S::load(p0.add(2 * qr)).mul_lanes::<0, 1>(z1, c);
                let t3 = S::load(p0.add(3 * qr)).mul_lanes::<0, 1>(z1, c);
                let (b0, b2) = (a0.add(t2), a0.sub(t2));
                let u1 = a1.add(t3).mul_lanes::<0, 1>(z23, c);
                let u3 = a1.sub(t3).mul_lanes::<2, 3>(z23, c);
                b0.add(u1).store(p0);
                b0.sub(u1).store(p0.add(qr));
                b2.add(u3).store(p0.add(2 * qr));
                b2.sub(u3).store(p0.add(3 * qr));
            }
        }
    }
}

/// The last four forward levels on each block of 16 of the `l` values at `a`: two levels as in
/// [`fwd4`], then the four vectors transposed, so that the butterflies of the last two levels,
/// 2 and 1 apart, pair lanes of different vectors, with a twiddle a lane. The block is left
/// transposed.
///
/// # Safety
///
/// `l >= 16`; the `l` values at `a` must be valid for reads and writes, and the `l / 2` pairs at
/// `w` for reads.
#[inline(always)]
unsafe fn fwd_last4<S: Lanes>(a: *mut i32, l: usize, w: *const i32, c: S::C) {
    // SAFETY: block `b` is `a[16b..16b + 16]`, its twiddles lie below `8b + 8 <= l / 2`.
    unsafe {
        for b in 0..(l >> 4) {
            let p0 = a.add(16 * b);
            let (z1, z23) = (S::pair(w.add(2 * b)), S::load(w.add(4 * b)));
            let a0 = S::load(p0).reduce(c);
            let a1 = S::load(p0.add(4));
            let t2 = S::load(p0.add(8)).mul_lanes::<0, 1>(z1, c);
            let t3 = S::load(p0.add(12)).mul_lanes::<0, 1>(z1, c);
            let (b0, b2) = (a0.add(t2), a0.sub(t2));
            let u1 = a1.add(t3).mul_lanes::<0, 1>(z23, c);
            let u3 = a1.sub(t3).mul_lanes::<2, 3>(z23, c);
            let [t0, t1, t2, t3] = S::transpose([b0.add(u1), b0.sub(u1), b2.add(u3), b2.sub(u3)]);
            // Lane `e` holds sub-block `4b + e`: twiddles `Z[4b + e]`, then `Z[8b + 2e]` and
            // `Z[8b + 2e + 1]`.
            let t0 = t0.reduce(c);
            let (w2, w2p) = S::load2(w.add(8 * b));
            let x2 = t2.mul(w2, w2p, c);
            let x3 = t3.mul(w2, w2p, c);
            let (y0, y2) = (t0.add(x2), t0.sub(x2));
            let (we, wpe, wo, wpo) = S::load4(w.add(16 * b));
            let z1 = t1.add(x3).mul(we, wpe, c);
            let z3 = t1.sub(x3).mul(wo, wpo, c);
            y0.add(z1).store(p0);
            y0.sub(z1).store(p0.add(4));
            y2.add(z3).store(p0.add(8));
            y2.sub(z3).store(p0.add(12));
        }
    }
}

/// The inverse levels `s + 1` and `s`, as [`fwd4`] runs them forward: the two sums of the first
/// level reduced.
///
/// # Safety
///
/// As for [`fwd4`], with the inverse twiddles at `w`.
#[inline(always)]
unsafe fn inv4<S: Lanes>(a: *mut i32, l: usize, s: u32, w: *const i32, c: S::C) {
    let m = l >> s;
    let qr = m >> 2;
    // SAFETY: as in `fwd4`.
    unsafe {
        for b in 0..(1usize << s) {
            let (z1, z23) = (S::pair(w.add(2 * b)), S::load(w.add(4 * b)));
            let base = a.add(b * m);
            for j in (0..qr).step_by(4) {
                let p0 = base.add(j);
                let a0 = S::load(p0);
                let a1 = S::load(p0.add(qr));
                let a2 = S::load(p0.add(2 * qr));
                let a3 = S::load(p0.add(3 * qr));
                let b0 = a0.add(a1).reduce(c);
                let b1 = a0.sub(a1).mul_lanes::<0, 1>(z23, c);
                let b2 = a2.add(a3).reduce(c);
                let b3 = a2.sub(a3).mul_lanes::<2, 3>(z23, c);
                b0.add(b2).store(p0);
                b1.add(b3).store(p0.add(qr));
                b0.sub(b2).mul_lanes::<0, 1>(z1, c).store(p0.add(2 * qr));
                b1.sub(b3).mul_lanes::<0, 1>(z1, c).store(p0.add(3 * qr));
            }
        }
    }
}

/// The first four inverse levels on each block of 16, from the transposed order that
/// [`fwd_last4`] leaves.
///
/// # Safety
///
/// As for [`fwd_last4`], with the inverse twiddles at `w`.
#[inline(always)]
unsafe fn inv_last4<S: Lanes>(a: *mut i32, l: usize, w: *const i32, c: S::C) {
    // SAFETY: as in `fwd_last4`.
    unsafe {
        for b in 0..(l >> 4) {
            let p0 = a.add(16 * b);
            let (y0, y1) = (S::load(p0), S::load(p0.add(4)));
            let (y2, y3) = (S::load(p0.add(8)), S::load(p0.add(12)));
            let (we, wpe, wo, wpo) = S::load4(w.add(16 * b));
            let x0 = y0.add(y1).reduce(c);
            let x1 = y0.sub(y1).mul(we, wpe, c);
            let x2 = y2.add(y3).reduce(c);
            let x3 = y2.sub(y3).mul(wo, wpo, c);
            let (w2, w2p) = S::load2(w.add(8 * b));
            let t2 = x0.sub(x2).mul(w2, w2p, c);
            let t3 = x1.sub(x3).mul(w2, w2p, c);
            let [v0, v1, v2, v3] = S::transpose([x0.add(x2), x1.add(x3), t2, t3]);
            let (z1, z23) = (S::pair(w.add(2 * b)), S::load(w.add(4 * b)));
            let b0 = v0.add(v1).reduce(c);
            let b1 = v0.sub(v1).mul_lanes::<0, 1>(z23, c);
            let b2 = v2.add(v3).reduce(c);
            let b3 = v2.sub(v3).mul_lanes::<2, 3>(z23, c);
            b0.add(b2).store(p0);
            b1.add(b3).store(p0.add(4));
            b0.sub(b2).mul_lanes::<0, 1>(z1, c).store(p0.add(8));
            b1.sub(b3).mul_lanes::<0, 1>(z1, c).store(p0.add(12));
        }
    }
}

/// The levels that [`split`] runs: 1, or 2 for an even `k`, so that the levels after them up to
/// the last four pair up.
fn split_levels(k: u32) -> u32 {
    if k.is_multiple_of(2) { 2 } else { 1 }
}

/// The forward transform of the `2^k` values at `a` after [`split`], with the twiddles at `t`.
///
/// # Safety
///
/// `MIN_K <= k`; the `2^k` values at `a` must be valid for reads and writes, and the twiddles
/// of [`twiddles`] at `t` for reads.
unsafe fn forward<S: Lanes>(a: *mut i32, k: u32, t: *const i32, c: S::C) {
    let l = 1usize << k;
    let mut s = split_levels(k);
    // SAFETY: as the caller guarantees.
    unsafe {
        while s + 4 < k {
            fwd4::<S>(a, l, s, t, c);
            s += 2;
        }
        fwd_last4::<S>(a, l, t, c);
    }
}

/// The inverse transform of the `2^k` values at `a` but the levels of [`finish`], with the
/// inverse twiddles at `w`.
///
/// # Safety
///
/// As for [`forward`], with the inverse twiddles of [`twiddles`] at `w`.
unsafe fn inverse<S: Lanes>(a: *mut i32, k: u32, w: *const i32, c: S::C) {
    let l = 1usize << k;
    // SAFETY: as the caller guarantees.
    unsafe {
        inv_last4::<S>(a, l, w, c);
        let mut s = k as i32 - 6;
        while s >= split_levels(k) as i32 {
            inv4::<S>(a, l, s as u32, w, c);
            s -= 2;
        }
    }
}

/// Multiplies the `l` values at `a` by those at `b`, Montgomery products, or squares them when
/// `b` is `a`.
///
/// # Safety
///
/// The `l` values at `a` must be valid for reads and writes and those at `b` for reads; `l` is a
/// multiple of 4.
unsafe fn pointwise<S: Lanes>(a: *mut i32, b: *const i32, l: usize, c: S::C) {
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in (0..l).step_by(4) {
            S::load(a.add(i)).mont(S::load(b.add(i)), c).store(a.add(i));
        }
    }
}

// Coefficients and residues

/// A coefficient's residue modulo a prime from its digits at `d + j`, `stride` apart, and the
/// constants `r[t] = 2^(t b) mod q`: at most `2^b + (dn - 1) (q / 2 + beta 2^b) < 1.75 q`.
///
/// # Safety
///
/// The digits must be valid for reads.
#[inline(always)]
unsafe fn residue<S: Lanes>(
    d: *const i32,
    j: usize,
    stride: usize,
    dn: usize,
    r: &[(i32, i32); 4],
    c: S::C,
) -> S {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut x = S::load(d.add(j));
        for (t, &(w, wp)) in r.iter().enumerate().take(dn).skip(1) {
            x = x.add(S::load(d.add(t * stride + j)).mul(S::splat(w), S::splat(wp), c));
        }
        x
    }
}

/// Sets the `2^k` values at `a` to the residues of the `nc` coefficients whose digits are at `d`
/// (digit `t` of coefficient `j` at `d[t stride + j]`, zeros up to `stride`), zeros after them,
/// with the first [`split_levels`] levels of the forward transform run on them.
///
/// One level: `a[j] = a[j + L/2] = x[j]` while `x[j + L/2]` is zero, or the sum and difference
/// of the two, reduced. Two levels, with `Z[1]` for the second half: from the four quarters, each
/// reduced.
///
/// # Safety
///
/// `MIN_K <= k`, `nc <= stride`, `stride` a multiple of 4; the values at `a` must be valid for
/// writes, the digits for reads, and the twiddles at `t` for reads.
#[allow(clippy::too_many_arguments)]
unsafe fn split<S: Lanes>(
    a: *mut i32,
    k: u32,
    d: *const i32,
    stride: usize,
    nc: usize,
    dn: usize,
    r: &[(i32, i32); 4],
    t: *const i32,
    c: S::C,
) {
    let top = nc.next_multiple_of(4);
    let zero = S::splat(0);
    // SAFETY: the residues read lie below `top <= stride`, and the values written in `a`.
    unsafe {
        let res = |j: usize| residue::<S>(d, j, stride, dn, r, c);
        if split_levels(k) == 1 {
            let h = 1usize << (k - 1);
            for j in (0..h).step_by(4) {
                let (u, v) = if j + h < top {
                    let (u, v) = (res(j).reduce(c), res(j + h).reduce(c));
                    (u.add(v), u.sub(v))
                } else if j < top {
                    let u = res(j);
                    (u, u)
                } else {
                    (zero, zero)
                };
                u.store(a.add(j));
                v.store(a.add(j + h));
            }
            return;
        }
        let qq = 1usize << (k - 2);
        let z1 = S::pair(t.add(2));
        for j in (0..qq).step_by(4) {
            let quarter = |i: usize| {
                if j + i * qq < top {
                    res(j + i * qq).reduce(c)
                } else {
                    zero
                }
            };
            let (r0, r1) = (quarter(0), quarter(1));
            let (u0, u1, u2, u3) = if j + 2 * qq < top {
                let (r2, r3) = (quarter(2), quarter(3));
                (r0.add(r2), r1.add(r3), r0.sub(r2), r1.sub(r3))
            } else {
                (r0, r1, r0, r1)
            };
            let v3 = u3.mul_lanes::<0, 1>(z1, c);
            u0.add(u1).store(a.add(j));
            u0.sub(u1).store(a.add(j + qq));
            u2.add(v3).store(a.add(j + 2 * qq));
            u2.sub(v3).store(a.add(j + 3 * qq));
        }
    }
}

/// The last [`split_levels`] levels of the inverse transform of the `2^k` values at `a`, each
/// result multiplied by `mu` and moved to `[0, q)`; only the first `n` are needed and kept. `w`
/// holds the inverse twiddles.
///
/// # Safety
///
/// As for [`inverse`].
unsafe fn finish<S: Lanes>(a: *mut i32, k: u32, n: usize, mu: (i32, i32), w: *const i32, c: S::C) {
    let (m, mp) = (S::splat(mu.0), S::splat(mu.1));
    let scale = |x: S| x.mul(m, mp, c).normalize(c);
    // SAFETY: every access lies in the `2^k` values at `a` and the inverse twiddle `Z[1]^-1`.
    unsafe {
        if split_levels(k) == 1 {
            let h = 1usize << (k - 1);
            for j in (0..h.min(n.next_multiple_of(4))).step_by(4) {
                let (u, v) = (S::load(a.add(j)), S::load(a.add(j + h)));
                scale(u.add(v)).store(a.add(j));
                if j + h < n {
                    scale(u.sub(v)).store(a.add(j + h));
                }
            }
            return;
        }
        let qq = 1usize << (k - 2);
        let iz1 = S::pair(w.add(2));
        for j in (0..qq.min(n.next_multiple_of(4))).step_by(4) {
            let a0 = S::load(a.add(j));
            let a1 = S::load(a.add(j + qq));
            let a2 = S::load(a.add(j + 2 * qq));
            let a3 = S::load(a.add(j + 3 * qq));
            let b0 = a0.add(a1).reduce(c);
            let b1 = a0.sub(a1).reduce(c);
            let b2 = a2.add(a3).reduce(c);
            let b3 = a2.sub(a3).mul_lanes::<0, 1>(iz1, c);
            scale(b0.add(b2)).store(a.add(j));
            if j + qq < n {
                scale(b1.add(b3)).store(a.add(j + qq));
            }
            if j + 2 * qq < n {
                scale(b0.sub(b2)).store(a.add(j + 2 * qq));
            }
            if j + 3 * qq < n {
                scale(b1.sub(b3)).store(a.add(j + 3 * qq));
            }
        }
    }
}

/// Sets digit `t` of coefficient `j < nc`, bits `[j w + t b, j w + t b + b)` of the `n` limbs at
/// `x` (fewer for the last digit), at `d[t stride + j]`, and zeros up to `stride`.
///
/// # Safety
///
/// The `n` limbs at `x` must be valid for reads and the `dn stride` digits at `d` for writes.
unsafe fn digits(d: *mut i32, stride: usize, x: *const u64, n: usize, cf: Config, nc: usize) {
    // SAFETY: every limb read lies below `n`, every digit written below `dn stride`.
    unsafe {
        let get = |pos: usize, len: usize| -> i32 {
            let (o, sh) = (pos / 64, pos % 64);
            if o >= n {
                return 0;
            }
            let mut v = *x.add(o) >> sh;
            if sh + len > 64 && o + 1 < n {
                v |= *x.add(o + 1) << (64 - sh);
            }
            (v & ((1 << len) - 1)) as i32
        };
        #[cfg(target_endian = "little")]
        let start = digits_le(d, stride, x, n, cf, nc);
        #[cfg(not(target_endian = "little"))]
        let start = 0;
        for j in start..stride {
            for t in 0..cf.dn {
                let lo = t * cf.b;
                let len = cf.b.min(cf.w - lo);
                *d.add(t * stride + j) = if j < nc { get(j * cf.w + lo, len) } else { 0 };
            }
        }
    }
}

/// [`digits`] for the coefficients whose 8-byte loads lie within the operand, by unaligned
/// little-endian loads at the digits' bytes; returns how many it set.
///
/// # Safety
///
/// As for [`digits`].
#[cfg(target_endian = "little")]
unsafe fn digits_le(
    d: *mut i32,
    stride: usize,
    x: *const u64,
    n: usize,
    cf: Config,
    nc: usize,
) -> usize {
    /// The digits of the coefficients `0..safe` for the widths given as constants.
    ///
    /// # Safety
    ///
    /// As for [`digits`], with `safe` coefficients whose loads lie within the operand.
    #[inline(always)]
    unsafe fn run<const W: usize, const B: usize, const DN: usize>(
        d: *mut i32,
        stride: usize,
        x: *const u8,
        safe: usize,
    ) {
        // SAFETY: as the caller guarantees.
        unsafe {
            for j in 0..safe {
                let base = x.add(j * (W / 8));
                for t in 0..DN {
                    let lo = t * B;
                    let len = if B < W - lo { B } else { W - lo };
                    let v = u64::from_le_bytes(core::ptr::read_unaligned(base.add(lo / 8).cast()));
                    *d.add(t * stride + j) = ((v >> (lo % 8)) & ((1 << len) - 1)) as i32;
                }
            }
        }
    }
    // Coefficient `j` reads 8 bytes from byte `j w / 8 + (dn - 1) b / 8`.
    let (bytes, last) = (8 * n, (cf.dn - 1) * cf.b / 8);
    let safe = if bytes >= 8 + last {
        ((bytes - 8 - last) * 8 / cf.w + 1).min(nc)
    } else {
        0
    };
    let x = x.cast::<u8>();
    // SAFETY: the loads of the first `safe` coefficients end within the `8 n` bytes at `x`.
    unsafe {
        match (cf.w, cf.b, cf.dn) {
            (48, 24, 2) => run::<48, 24, 2>(d, stride, x, safe),
            (56, 28, 2) => run::<56, 28, 2>(d, stride, x, safe),
            (64, 22, 3) => run::<64, 22, 3>(d, stride, x, safe),
            (72, 24, 3) => run::<72, 24, 3>(d, stride, x, safe),
            (88, 22, 4) => run::<88, 22, 4>(d, stride, x, safe),
            (104, 26, 4) => run::<104, 26, 4>(d, stride, x, safe),
            _ => return 0,
        }
    }
    safe
}

// The Chinese remainder theorem

/// Adds coefficient `t` of a group, its limbs at `c[..][t]`, at bit `t W` of the window `acc`.
#[inline(always)]
fn add_coefficient<const W: usize, const ND: usize, const NL: usize>(
    acc: &mut [u128; 24],
    c: &[[u64; 8]; 4],
    t: usize,
) {
    let pos = t * W;
    let (a, s) = (pos / 64, (pos % 64) as u32);
    for i in 0..=NL {
        // The top piece holds the bits shifted out of the top limb, none when the digits do not
        // reach that far.
        if i == NL && (s == 0 || 32 * ND as u32 + s <= 64 * NL as u32) {
            break;
        }
        let lo = if i < NL { c[i][t] << s } else { 0 };
        let hi = if s > 0 && i > 0 {
            c[i - 1][t] >> (64 - s)
        } else {
            0
        };
        acc[a + i] += u128::from(lo | hi);
    }
}

/// Adds the coefficients `j..j + 8` of the product, whose residues are at `res[i]` for each
/// prime, at bit `j W` of the `rn` limbs at `r`, where limb `o` is: the limbs below the window
/// are written, those of the window above the group's `W / 8` are kept in `pend` for the next
/// group, and `carry` goes into the next limb written.
///
/// # Safety
///
/// The residues `j..j + 8` must be valid for reads and the limbs at `r` for writes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn group<const W: usize, const ND: usize, const NL: usize>(
    r: *mut u64,
    rn: usize,
    o: usize,
    res: &[*const i32; 8],
    j: usize,
    n: usize,
    crt: &Crt,
    pend: &mut [u128; 8],
    carry: &mut u128,
) {
    let mut limbs = [[0u64; 8]; 4];
    // SAFETY: as the caller guarantees.
    unsafe {
        four::<ND>(res, j, crt, &mut limbs, 0);
        four::<ND>(res, j + 4, crt, &mut limbs, 4);
    }
    for t in n.saturating_sub(j).min(8)..8 {
        for l in limbs.iter_mut() {
            l[t] = 0;
        }
    }
    let mut acc = [0u128; 24];
    acc[..8].copy_from_slice(pend);
    for t in 0..8 {
        add_coefficient::<W, ND, NL>(&mut acc, &limbs, t);
    }
    let (gl, span) = (W / 8, (7 * W) / 64 + NL + 1);
    for (e, &v) in acc.iter().enumerate().take(gl) {
        let s = v + *carry;
        if o + e < rn {
            // SAFETY: limb `o + e` lies below `rn`.
            unsafe { *r.add(o + e) = s as u64 };
        }
        *carry = s >> 64;
    }
    for (e, p) in pend.iter_mut().enumerate() {
        *p = if gl + e < span { acc[gl + e] } else { 0 };
    }
}

/// Sets the `rn` limbs at `r` to the sum of the `n` coefficients of the product, from their
/// residues, `l` apart at `res`, coefficient `j` at bit `j W`.
///
/// Returns the two limbs of the sum above the `rn` limbs: zeros for a product, which fits them.
///
/// # Safety
///
/// `n <= l`, `l` a multiple of 8; the `np l` residues at `res` must be valid for reads and the
/// limbs at `r` for writes; the sum is below `2^(64 rn + 128)`.
#[inline(never)]
unsafe fn crt<const W: usize, const ND: usize, const NL: usize>(
    r: *mut u64,
    rn: usize,
    res: *const i32,
    l: usize,
    n: usize,
    c: &Crt,
) -> [u64; 2] {
    // SAFETY: the residues of prime `i` are `res[i l..(i + 1) l]`; the groups read residues below
    // `n.next_multiple_of(8) <= l`.
    unsafe {
        let ptrs: [*const i32; 8] = core::array::from_fn(|i| res.add(i.min(c.np - 1) * l));
        let (mut pend, mut carry, mut o) = ([0u128; 8], 0u128, 0);
        for j in (0..n).step_by(8) {
            group::<W, ND, NL>(r, rn, o, &ptrs, j, n, c, &mut pend, &mut carry);
            o += W / 8;
        }
        let mut e = 0;
        while o + e < rn {
            let s = if e < 8 { pend[e] } else { 0 } + carry;
            *r.add(o + e) = s as u64;
            carry = s >> 64;
            e += 1;
        }
        // What lies above the `rn` limbs: nothing for a product, which fits them, and for a
        // cyclic product (see [`mul_cyclic`]) the part of its sum above `2^(w L)`, below `2^126`.
        let mut above = [0; 2];
        for a in &mut above {
            let s = if e < 8 { pend[e] } else { 0 } + carry;
            *a = s as u64;
            carry = s >> 64;
            e += 1;
        }
        debug_assert!(carry == 0 && (e..8).all(|e| pend[e] == 0));
        above
    }
}

// Configurations and plans

/// A width of coefficients, the primes that make it exact, and the digits of a coefficient.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Config {
    /// The number of primes.
    np: usize,
    /// The bits of a coefficient, a multiple of 8, so that 8 of them end on a limb.
    w: usize,
    /// The digits of a coefficient.
    dn: usize,
    /// The bits of a digit, at most 28: below `2^28`, a digit is a residue as it is.
    b: usize,
    /// The cycles of the Chinese remainder theorem for a coefficient, measured on an Apple M4
    /// Max.
    crt_cycles: usize,
}

/// The widths of coefficients, each with the fewest primes that make it exact for long
/// operands, and, measured on an Apple M4 Max, the cycles of the Chinese remainder theorem of a
/// coefficient, which grow with the primes, from 7.0 for 4 to 22.8 for 8. For a given number of
/// coefficients a product costs about `np / w` cycles a bit in the transforms, and from 0.15 to
/// 0.23 a bit in the theorem; it is lowest for 5 primes and 64 bits, 0.078 and 0.15, and the
/// others let a product fit a shorter transform. Of the pairs exact for the longest products,
/// these are the ones [`plan`] chose for some product or square of 150 limbs to 12 MiB, balanced
/// or up to 256 times longer than the shorter operand: 3 primes and 32 bits, 4 and 40, and 7 and
/// 80 never were, and 8 and 96 only where 8 and 104 are no longer exact, products of more than
/// 16 MiB, which Toom's methods cut into parts that are not.
const CONFIGS: [Config; 6] = [
    Config {
        np: 4,
        w: 48,
        dn: 2,
        b: 24,
        crt_cycles: 70,
    },
    Config {
        np: 5,
        w: 56,
        dn: 2,
        b: 28,
        crt_cycles: 100,
    },
    Config {
        np: 5,
        w: 64,
        dn: 3,
        b: 22,
        crt_cycles: 98,
    },
    Config {
        np: 6,
        w: 72,
        dn: 3,
        b: 24,
        crt_cycles: 122,
    },
    Config {
        np: 7,
        w: 88,
        dn: 4,
        b: 22,
        crt_cycles: 203,
    },
    Config {
        np: 8,
        w: 104,
        dn: 4,
        b: 26,
        crt_cycles: 228,
    },
];

/// How a product is computed: the configuration, the transforms' `k`, and the coefficients of
/// the operands.
#[derive(Clone, Copy)]
struct Plan {
    cf: Config,
    k: u32,
    cx: usize,
    cy: usize,
}

/// Whether coefficients of `w` bits are exact with `np` primes for a shorter operand of `m`
/// coefficients: `2 m (2^w - 1)^2 < P` holds when `m <= 2^(bits(P) - 2w - 2)`, as then the left
/// side is below `2^(bits(P) - 1) <= P`.
fn exact(cf: Config, m: usize) -> bool {
    let e = CRT[cf.np].bits as usize - 2 * cf.w - 2;
    e >= usize::BITS as usize || m <= 1 << e
}

/// The plan of least estimated cost for operands of `nx` and `ny` limbs (`ny` 0 for a square),
/// if any fits the transforms of up to `2^21` points.
///
/// The cost of a transform's point for one prime was measured on an Apple M4 Max at about
/// `0.6 k + 3` cycles for a product, the two forward transforms and the inverse, the residues,
/// the twiddles and the pointwise products together, and `0.4 k + 2` for a square; the Chinese
/// remainder theorem adds [`Config::crt_cycles`] a coefficient.
fn plan(nx: usize, ny: usize) -> Option<Plan> {
    let mut best: Option<(Plan, u64)> = None;
    // An operand of more than 2^22 limbs has more coefficients, of at most 104 bits, than any
    // transform has points; below, the counts of bits fit a `usize` on every target.
    if nx > 1 << 22 {
        return None;
    }
    for cf in CONFIGS {
        let cx = (64 * nx).div_ceil(cf.w);
        let cy = (64 * ny).div_ceil(cf.w);
        let (n, shorter) = if ny == 0 {
            (2 * cx - 1, cx)
        } else {
            (cx + cy - 1, cy.min(cx))
        };
        if !exact(cf, shorter) {
            continue;
        }
        let k = (usize::BITS - (n - 1).leading_zeros()).max(MIN_K);
        if k > MAX_K {
            continue;
        }
        let per_point = if ny == 0 {
            4 * u64::from(k) + 20
        } else {
            6 * u64::from(k) + 30
        };
        let cost = cf.np as u64 * (per_point << k) + n as u64 * cf.crt_cycles as u64;
        if best.is_none_or(|(_, c)| cost < c) {
            best = Some((Plan { cf, k, cx, cy }, cost));
        }
    }
    best.map(|(p, _)| p)
}

/// Whether a product of operands of `nx >= ny` limbs fits the transforms.
pub(super) fn fits(nx: usize, ny: usize) -> bool {
    plan(nx, ny).is_some()
}

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` limbs at
/// `y`.
///
/// # Safety
///
/// [`fits`]`(nx, ny)`, `nx >= ny >= 1`; the limbs of `x` and `y` must be valid for reads and those
/// of `r` for writes, `r` apart from both.
pub(super) unsafe fn mul(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize) {
    let p = plan(nx, ny).expect("a product that fits the transforms");
    // SAFETY: forwarded.
    unsafe { product(r, x, nx, y, ny, p) }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x`.
///
/// # Safety
///
/// [`fits`]`(n, n)`, `n >= 1`; the limbs of `x` must be valid for reads and those of `r` for
/// writes, apart.
pub(super) unsafe fn sqr(r: *mut u64, x: *const u64, n: usize) {
    let p = plan(n, 0).expect("a square that fits the transforms");
    // SAFETY: forwarded, with no second operand.
    unsafe { product(r, x, n, core::ptr::null(), 0, p) }
}

/// [`mul`] by the plan `p`, or [`sqr`] when `ny` is 0.
///
/// # Safety
///
/// As for [`mul`] or [`sqr`].
unsafe fn product(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize, p: Plan) {
    let (cf, k) = (p.cf, p.k);
    let (l, np, square) = (1usize << k, cf.np, ny == 0);
    let (sx, sy) = (p.cx.next_multiple_of(4), p.cy.next_multiple_of(4));
    let n = if square {
        2 * p.cx - 1
    } else {
        p.cx + p.cy - 1
    };
    let others = if square { 0 } else { l };
    let mut buffer = Vec::<i32>::with_capacity((np + 2) * l + others + cf.dn * (sx + sy));
    // SAFETY: the buffer holds the residues of the `np` primes, the transform of `y`, the
    // twiddles and the digits of both operands, each written before it is read.
    unsafe {
        let res = buffer.as_mut_ptr();
        let t = res.add(np * l);
        let yt = t.add(2 * l);
        let dx = yt.add(others);
        let dy = dx.add(cf.dn * sx);
        digits(dx, sx, x, nx, cf, p.cx);
        if !square {
            digits(dy, sy, y, ny, cf, p.cy);
        }
        let crt_data = &CRT[np];
        for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
            let c = V::consts(&data.prime);
            twiddles::<V>(t, k, data);
            let r_t: [(i32, i32); 4] = core::array::from_fn(|t| data.pow2[t * cf.b]);
            let a = res.add(i * l);
            split::<V>(a, k, dx, sx, p.cx, cf.dn, &r_t, t, c);
            forward::<V>(a, k, t, c);
            if square {
                pointwise::<V>(a, a, l, c);
            } else {
                split::<V>(yt, k, dy, sy, p.cy, cf.dn, &r_t, t, c);
                forward::<V>(yt, k, t, c);
                pointwise::<V>(a, yt, l, c);
            }
            inverse::<V>(a, k, t.add(l), c);
            let q = data.prime.q as i64;
            let mu = centred(
                (crt_data.mu[i] as i64 * data.inv2k[k as usize] as i64).rem_euclid(q) as u64,
                q as u64,
            );
            finish::<V>(a, k, n, (mu, barrett(mu, q as u32)), t.add(l), c);
        }
        let rn = if square { 2 * nx } else { nx + ny };
        let above = theorem(r, rn, res, l, n, cf);
        debug_assert!(above == [0, 0], "a product fits its limbs");
    }
}

/// Sets the `rn` limbs at `r` to the sum of the `n` coefficients whose residues for the primes of
/// `cf` are `l` apart at `res`, by [`crt`], and returns the two limbs above them.
///
/// # Safety
///
/// As for [`crt`].
unsafe fn theorem(
    r: *mut u64,
    rn: usize,
    res: *const i32,
    l: usize,
    n: usize,
    cf: Config,
) -> [u64; 2] {
    let crt_data = &CRT[cf.np];
    macro_rules! go {
        ($w:literal, $nd:literal, $nl:literal) => {
            // SAFETY: as the caller guarantees.
            unsafe { crt::<$w, $nd, $nl>(r, rn, res, l, n, crt_data) }
        };
    }
    match (cf.w, crt_data.nd) {
        (48, 4) => go!(48, 4, 2),
        (56, 5) => go!(56, 5, 3),
        (64, 5) => go!(64, 5, 3),
        (72, 6) => go!(72, 6, 3),
        (88, 7) => go!(88, 7, 4),
        (104, 8) => go!(104, 8, 4),
        _ => unreachable!("a configuration without its theorem"),
    }
}

// Wrap-around products

/// The plan of a product modulo `B^n - 1` for `n = w 2^k / 64` limbs: a cyclic convolution of
/// length `L = 2^k` of coefficients of `w` bits, by the configuration `cf`.
///
/// `X^L - 1` for `X = 2^w` is `B^n - 1`, so the cyclic convolution of the operands' coefficients,
/// which the transforms of length `L` compute whatever the number of coefficients of the product,
/// gives the product modulo `B^n - 1`: a coefficient past the `L`-th wraps around to the bottom.
/// That is a product of up to `n` limbs at the cost of one whose result has `n` limbs, half that
/// of a product of two operands of `n` limbs, where the division knows the high or the low half
/// of the product it needs (see the division's `newton`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in super::super) struct Cyclic {
    cf: Config,
    k: u32,
}

impl Cyclic {
    /// The limbs `n` of the modulus `B^n - 1`.
    pub(in super::super) fn limbs(self) -> usize {
        (self.cf.w << self.k) / 64
    }

    /// The estimated cycles of a product, in the units of [`plan`]: three transforms, or two when
    /// an operand is `ready` (a [`Transform`]), and the Chinese remainder theorem.
    pub(in super::super) fn cost(self, ready: bool) -> u64 {
        let transforms = if ready { 2 } else { 3 };
        transforms * self.transform_cost() + (self.cf.crt_cycles << self.k) as u64
    }

    /// The estimated cycles of a transform of an operand, its residues and its share of the
    /// pointwise products, in the units of [`plan`].
    pub(in super::super) fn transform_cost(self) -> u64 {
        let k = u64::from(self.k);
        self.cf.np as u64 * ((2 * k + 10) << k)
    }
}

/// The plan of least estimated cost for products modulo `B^n - 1` with `n >= min` limbs, whose
/// shorter operand has at most `shorter` limbs, if any fits the transforms; for every product by
/// it, each operand has at most `n` limbs.
///
/// Each coefficient of the cyclic convolution is a sum of at most as many products of two
/// coefficients as the shorter operand has, at most `L` (for each coefficient of one operand,
/// one of the other lands on it), so [`exact`] tells for it what it tells for a product.
pub(in super::super) fn cyclic(min: usize, shorter: usize) -> Option<Cyclic> {
    let mut best: Option<(Cyclic, u64)> = None;
    if min > 1 << 22 {
        return None;
    }
    for cf in CONFIGS {
        // The least `k` whose `w 2^k / 64` limbs are at least `min`.
        let coefficients = (64 * min).div_ceil(cf.w);
        let k = (usize::BITS - (coefficients - 1).leading_zeros()).max(MIN_K);
        if k > MAX_K {
            continue;
        }
        let m = ((64 * shorter).div_ceil(cf.w)).min(1 << k);
        if !exact(cf, m) {
            continue;
        }
        let p = Cyclic { cf, k };
        if best.is_none_or(|(_, c)| p.cost(false) < c) {
            best = Some((p, p.cost(false)));
        }
    }
    best.map(|(p, _)| p)
}

/// The twiddles of the first `np` primes for transforms of up to `2^k` points, shared by the
/// products of a division instead of built for each: the table of [`twiddles`] for `2^j` points is
/// the first half of each of its two halves for `2^(j+1)`, as `Z[b]` and its inverse depend on `b`
/// alone, so the tables for `2^k` serve every shorter transform.
#[derive(Clone)]
pub(in super::super) struct Twiddles {
    k: u32,
    np: usize,
    /// For each prime, the `2^k` values of the forward twiddles, then those of the inverse ones.
    t: Vec<i32>,
}

impl Twiddles {
    /// The twiddles for the products by the plans `plans`: of their most primes and longest
    /// transform.
    pub(in super::super) fn of(plans: &[Cyclic]) -> Twiddles {
        let k = plans.iter().map(|p| p.k).max().unwrap_or(MIN_K);
        let np = plans.iter().map(|p| p.cf.np).max().unwrap_or(1);
        let l = 1usize << k;
        let mut t = Vec::<i32>::with_capacity(2 * l * np);
        // SAFETY: each prime's `2^(k+1)` values are written by `twiddles`.
        unsafe {
            for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
                twiddles::<V>(t.as_mut_ptr().add(2 * l * i), k, data);
            }
            t.set_len(2 * l * np);
        }
        Twiddles { k, np, t }
    }

    /// The forward and the inverse twiddles of prime `i` for the plan `p`, if they are here.
    fn get(&self, i: usize, p: Cyclic) -> Option<(*const i32, *const i32)> {
        if p.k > self.k || i >= self.np {
            return None;
        }
        let base = self.t.as_ptr().wrapping_add((2 * i) << self.k);
        Some((base, base.wrapping_add(1 << self.k)))
    }
}

/// The forward and the inverse twiddles of prime `i` (whose data is `data`) for the plan `p`: from
/// `tw` when it has them, otherwise built at `t`, which has room for `2^(k+1)` values.
///
/// # Safety
///
/// The `2^(k+1)` values at `t` must be valid for writes.
unsafe fn twiddles_of(
    tw: Option<&Twiddles>,
    i: usize,
    p: Cyclic,
    data: &PrimeData,
    t: *mut i32,
) -> (*const i32, *const i32) {
    if let Some(pair) = tw.and_then(|tw| tw.get(i, p)) {
        return pair;
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        twiddles::<V>(t, p.k, data);
        (t, t.add(1 << p.k))
    }
}

/// An operand transformed by a plan, for products by it that skip its transforms: the residues
/// of its coefficients modulo each prime, transformed, in the order the forward transform leaves.
#[derive(Clone)]
pub(in super::super) struct Transform {
    /// The plan.
    plan: Cyclic,
    /// The `np 2^k` values, prime after prime.
    res: Vec<i32>,
    /// The operand's coefficients.
    nc: usize,
}

/// The `nx` limbs at `x` transformed by the plan `p`.
///
/// # Safety
///
/// `1 <= nx <= p.limbs()`, at most the `shorter` of the plan if it is the shorter operand of a
/// product; the limbs at `x` must be valid for reads.
pub(in super::super) unsafe fn transform(
    x: *const u64,
    nx: usize,
    p: Cyclic,
    tw: Option<&Twiddles>,
) -> Transform {
    let (cf, k) = (p.cf, p.k);
    let (l, np) = (1usize << k, cf.np);
    let cx = (64 * nx).div_ceil(cf.w);
    let sx = cx.next_multiple_of(4);
    debug_assert!(nx <= p.limbs() && cx <= l);
    let mut res = Vec::<i32>::with_capacity(np * l);
    let mut work = Vec::<i32>::with_capacity(2 * l + cf.dn * sx);
    // SAFETY: the vectors hold the transforms of the `np` primes, and the twiddles and the
    // digits of the operand, each written before it is read.
    unsafe {
        let (t, dx) = (work.as_mut_ptr(), work.as_mut_ptr().add(2 * l));
        digits(dx, sx, x, nx, cf, cx);
        for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
            let c = V::consts(&data.prime);
            let (tf, _) = twiddles_of(tw, i, p, data, t);
            let r_t: [(i32, i32); 4] = core::array::from_fn(|t| data.pow2[t * cf.b]);
            let a = res.as_mut_ptr().add(i * l);
            split::<V>(a, k, dx, sx, cx, cf.dn, &r_t, tf, c);
            forward::<V>(a, k, tf, c);
        }
        res.set_len(np * l);
    }
    Transform {
        plan: p,
        res,
        nc: cx,
    }
}

/// Sets the `n = p.limbs()` limbs at `r` to the product of the `nx` limbs at `x` and the operand
/// transformed as `y` modulo `B^n - 1`: a value from 0 to `B^n - 1`, where `B^n - 1` stands for 0.
/// A product below `B^n - 1` is the product itself. `tw` may hold the twiddles.
///
/// The `L` coefficients of the cyclic convolution are added at their positions, `w` bits apart,
/// into a sum `S` of the limbs at `r` and the two above them that [`crt`] returns: as each
/// coefficient is below `L 2^(2w)`, `S < 2^(w L) L 2^w (1 + 2^-w)`, so `S / B^n < 2^126`. The
/// limbs above `B^n` are then added at the bottom, as `B^n = 1` modulo `B^n - 1`, and so is the
/// carry out of the top, which leaves the sum minus `B^n` below `2^126`, and cannot carry again.
///
/// # Safety
///
/// `1 <= nx <= n`, `y` made by `p`, one of the two operands of at most the plan's `shorter`
/// limbs. The limbs at `x` must be valid for reads and the `n` limbs at `r` for writes, apart.
pub(in super::super) unsafe fn mul_cyclic(
    r: *mut u64,
    p: Cyclic,
    x: *const u64,
    nx: usize,
    y: &Transform,
    tw: Option<&Twiddles>,
) {
    let (cf, k) = (p.cf, p.k);
    let (l, np, n) = (1usize << k, cf.np, p.limbs());
    let cx = (64 * nx).div_ceil(cf.w);
    let sx = cx.next_multiple_of(4);
    debug_assert!(nx <= n && cx <= l && y.plan == p && exact(cf, cx.min(y.nc)));
    let mut buffer = Vec::<i32>::with_capacity((np + 2) * l + cf.dn * sx);
    // SAFETY: the buffer holds the residues of the `np` primes, the twiddles and the digits of
    // `x`, each written before it is read.
    unsafe {
        let res = buffer.as_mut_ptr();
        let t = res.add(np * l);
        let dx = t.add(2 * l);
        digits(dx, sx, x, nx, cf, cx);
        let crt_data = &CRT[np];
        for (i, data) in PRIME_DATA.iter().enumerate().take(np) {
            let c = V::consts(&data.prime);
            let (tf, ti) = twiddles_of(tw, i, p, data, t);
            let r_t: [(i32, i32); 4] = core::array::from_fn(|t| data.pow2[t * cf.b]);
            let a = res.add(i * l);
            split::<V>(a, k, dx, sx, cx, cf.dn, &r_t, tf, c);
            forward::<V>(a, k, tf, c);
            pointwise::<V>(a, y.res.as_ptr().add(i * l), l, c);
            inverse::<V>(a, k, ti, c);
            let q = data.prime.q as i64;
            let mu = centred(
                (crt_data.mu[i] as i64 * data.inv2k[k as usize] as i64).rem_euclid(q) as u64,
                q as u64,
            );
            finish::<V>(a, k, l, (mu, barrett(mu, q as u32)), ti, c);
        }
        let [a0, a1] = theorem(r, n, res, l, l, cf);
        let (s0, c0) = (*r).overflowing_add(a0);
        *r = s0;
        let (s1, c1) = (*r.add(1)).overflowing_add(a1);
        let (s1, c2) = s1.overflowing_add(u64::from(c0));
        *r.add(1) = s1;
        let mut carry = u64::from(c1 | c2);
        let mut i = 2;
        while carry != 0 && i < n {
            let (v, o) = (*r.add(i)).overflowing_add(1);
            *r.add(i) = v;
            carry = u64::from(o);
            i += 1;
        }
        if carry != 0 {
            let mut i = 0;
            loop {
                let (v, o) = (*r.add(i)).overflowing_add(1);
                *r.add(i) = v;
                if !o {
                    break;
                }
                i += 1;
            }
        }
    }
}
