//! The self-initialising quadratic sieve: [`siqs`].
//!
//! The quadratic sieve (Pomerance 1982) looks for many `Y` whose `Y² − kN` factors over a set of
//! small primes, the factor base, and combines them by linear algebra over GF(2) into a congruence
//! of squares `X² ≡ Y² (mod N)`, which gives a factor `gcd(X − Y, N)` with probability about 1/2.
//! The self-initialising version (Alford and Pomerance 1993; Contini 1997) takes the values of many
//! polynomials `(Ax + B)² − kN = A·g(x)` over a short interval, `A` a product of `s` primes of the
//! factor base and `2^(s−1)` values of `B` for each `A`, so cheap to move from one to the next that
//! the interval can be short and the values small. The parts:
//!
//! - [`fb`]: Knuth and Schroeppel's multiplier `k` and the factor base, the primes modulo which `kN`
//!   is a square, with a square root of `kN` modulo each;
//! - [`poly`]: the choice of `A`, the values `B_l`, the Gray code that orders the values of `B`;
//! - [`sieve`]: the sieve in blocks the size of half the data cache, the small primes left out, the
//!   large primes through buckets;
//! - [`tdiv`]: the trial division of the candidates, and the large primes;
//! - [`relations`]: the relations, and the cycles of those with large primes;
//! - [`linalg`]: Montgomery's block Lanczos method, and Gaussian elimination for small matrices;
//! - [`sqrt`]: the filtering of the matrix and the square roots;
//! - [`arith`] and [`wide`]: arithmetic on words and on numbers of a few limbs;
//! - `kernels`: the steps of the sieve and of the trial division in NEON on aarch64 (the module
//!   `aarch64`), in plain Rust elsewhere and under Miri (`portable`).
//!
//! Numbers below `2^64` are factored by Pollard's rho method, with Lehman's method should it fail.

mod arith;
mod fb;
// The kernels of the sieve, under the name `kernels`: NEON on the 64-bit little-endian aarch64
// targets of the crate's assembly, where NEON is enabled, as it is by default; the portable ones
// elsewhere and under Miri, which runs no SIMD of this kind.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    target_feature = "neon",
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
#[path = "siqs/aarch64.rs"]
mod kernels;
#[cfg(not(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    target_feature = "neon",
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
#[path = "siqs/portable.rs"]
mod kernels;
mod linalg;
mod params;
mod poly;
mod relations;
mod sieve;
mod sqrt;
mod tdiv;
mod wide;

use super::super::{ByteOrder, Number};
use sqrt::{limbs_of, number_of};

/// The largest number of bits of a number the sieve takes: beyond, the values of its polynomials
/// would not fit the wide numbers (and it would take years).
const MAX_BITS: u32 = 900;

/// The odd primes trial division tries before the sieve.
const TRIAL_TO: u32 = 1000;

/// A factor of the absolute value of `n` other than 1 and itself, by the self-initialising quadratic
/// sieve, or `None` when it finds none: always for 0, 1, a prime and a power of a prime, and for
/// numbers of more than 900 bits, which it would take years to factor. The factor is positive, in
/// the byte order of `n`. Meant for numbers of 24 to about 100 digits without small factors.
///
/// A prime factor below 1000 is found first (the smallest), a perfect power of a composite number
/// gives its root, and numbers below `2^64` are factored by Pollard's rho method. Otherwise the
/// sieve (Alford and Pomerance 1993, Contini 1997) collects relations `Y² − kN` that factor over a
/// factor base, with up to two large primes, combines them by Gaussian elimination or Montgomery's
/// block Lanczos method into congruences of squares `X² ≡ Y² (mod N)`, and gives the first
/// `gcd(X − Y, N)` that is a proper factor: for a product of two primes, one of them. The choices
/// are fixed, so that a number gives the same factor, in the same time, from run to run. The
/// sieve's parameters and logarithms are rounded from the platform's floating-point `log2`, `ln`
/// and `powf`, whose last bit Rust leaves to the platform (Apple's and glibc's differ on some of
/// those it takes), so that on another target a number may, rarely, give the other factor.
/// Measured on an Apple M4 Max, single-threaded, a 30-digit product of two primes
/// takes about 0.5 ms, a 40-digit one about 3 ms, a 60-digit one about 0.3 s, an 80-digit one
/// about 30 s and a 100-digit one about 40 minutes.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, factor};
///
/// let p: Number = "28162596453413674991".parse().unwrap();
/// let q: Number = "71875687879868738317".parse().unwrap();
/// let f = factor::siqs(&(&p * &q)).unwrap();
/// assert!(f == p || f == q);
/// assert_eq!(factor::siqs(&p), None);
/// assert_eq!(factor::siqs(&Number::new_from_u8(2, ByteOrder::BigEndian).pow(100)), None);
/// ```
pub fn siqs(n: &Number) -> Option<Number> {
    let order = n.byte_order;
    let m = limbs_of(n);
    let f = factor_limbs(&m)?;
    let f = number_of(&f);
    Some(if order == ByteOrder::BigEndian {
        let mut value = f.value;
        value.reverse();
        Number {
            value,
            negative: false,
            byte_order: order,
        }
    } else {
        f
    })
}

/// A factor of the number of limbs `n` (least significant first, no zero limb at the top).
fn factor_limbs(n: &[u64]) -> Option<Vec<u64>> {
    if n.is_empty() || (n.len() == 1 && n[0] < 4) {
        return None;
    }
    // Small factors, and powers of them.
    if n[0] & 1 == 0 {
        return if is_power_of(n, 2) {
            None
        } else {
            Some(vec![2])
        };
    }
    let mut found = None;
    super::super::factorial::primes::Sieve::new(u64::from(TRIAL_TO)).each(
        3,
        u64::from(TRIAL_TO),
        |p| {
            if found.is_none() && arith::rem_1(n, p as u32) == 0 {
                found = Some(p as u32);
            }
        },
    );
    if let Some(p) = found {
        return if is_power_of(n, p) {
            None
        } else {
            Some(vec![u64::from(p)])
        };
    }
    let number = number_of(n);
    if number.is_probable_prime() {
        return None;
    }
    // Perfect powers: m = r^e with r not a perfect power, which is a prime or has a factor.
    let mut root = number.clone();
    let mut e = 2u32;
    while u64::from(e) * 9 <= root.bits() {
        let r = root.nth_root(e);
        if r.pow(e) == root {
            root = r;
            e = 2;
            continue;
        }
        e += 1;
    }
    if root != number {
        return if root.is_probable_prime() {
            None
        } else {
            Some(limbs_of(&root))
        };
    }
    if n.len() == 1 {
        return small(n[0]).map(|f| vec![f]);
    }
    if sieve::bits_of(n) > MAX_BITS {
        return None;
    }
    run(n)
}

/// Whether `n` is a power of `p`.
fn is_power_of(n: &[u64], p: u32) -> bool {
    if p == 2 {
        return n.iter().map(|l| l.count_ones()).sum::<u32>() == 1;
    }
    let mut x = n.to_vec();
    let p64 = u64::from(p);
    let inv = arith::inverse_64(p64);
    loop {
        while x.last() == Some(&0) {
            x.pop();
        }
        if x == [1] {
            return true;
        }
        if !arith::divides(&x, p64, inv) {
            return false;
        }
        arith::div_exact_1(&mut x, p64, inv);
    }
}

/// A factor of the odd composite `n < 2^64` with no factor below 1000 that is not a prime power.
fn small(n: u64) -> Option<u64> {
    if let Some(f) = arith::rho_64(n, 1 << 22) {
        return Some(f);
    }
    lehman(n)
}

/// A factor of `n` by Lehman's method (1974): trial division to `n^(1/3)`, then for each
/// `k <= n^(1/3)` the `a` from `sqrt(4kn)` to `sqrt(4kn) + n^(1/6)/(4 sqrt(k))` whose `a² − 4kn` is a
/// square `b²`, which gives `gcd(a + b, n)`.
fn lehman(n: u64) -> Option<u64> {
    let c = (n as f64).cbrt() as u64 + 1;
    let mut p = 3;
    while p <= c {
        if n.is_multiple_of(p) {
            return Some(p);
        }
        p += 2;
    }
    let sixth = (n as f64).powf(1.0 / 6.0);
    for k in 1..=c {
        let fk = 4 * u128::from(k) * u128::from(n);
        let lo = isqrt_128(fk);
        let lo = if lo * lo < fk { lo + 1 } else { lo };
        let hi = isqrt_128(fk) + (sixth / (4.0 * (k as f64).sqrt())) as u128 + 1;
        for a in lo..=hi {
            let b2 = a * a - fk;
            let b = isqrt_128(b2);
            if b * b == b2 {
                let g = arith::gcd_64(((a + b) % u128::from(n)) as u64, n);
                if g > 1 && g < n {
                    return Some(g);
                }
            }
        }
    }
    None
}

/// `⌊√x⌋` of a `u128`.
fn isqrt_128(x: u128) -> u128 {
    if x == 0 {
        return 0;
    }
    let mut r = (x as f64).sqrt() as u128;
    while r * r > x {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= x {
        r += 1;
    }
    r
}

/// The sieve proper, for an odd `n` of at least 65 bits with no factor below 1000, not a prime and
/// not a perfect power.
fn run(n: &[u64]) -> Option<Vec<u64>> {
    let k = fb::multiplier(n);
    let kn = poly::mul_1(n, u64::from(k));
    let bits = sieve::bits_of(n);
    let params = params::for_bits(bits);
    let fb = match fb::build(n, k, params.fb_size) {
        fb::Built::Factor(p) => return Some(vec![u64::from(p)]),
        fb::Built::Base(fb) => fb,
    };
    let mut s = sieve::Siqs::new(n, kn, fb, &params);
    for _ in 0..4 {
        let enough = s.collect();
        let f = s.finish();
        if let Some(f) = f {
            return Some(f);
        }
        if !enough {
            return None;
        }
        // More relations, which give more vectors of the null space.
        s.target += 16 + s.target / 16;
    }
    None
}
