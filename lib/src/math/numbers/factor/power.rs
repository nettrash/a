//! Perfect powers: whether a number is `b^k` for some `k >= 2`, with the largest `k`.
//!
//! `x = c^K` with `K` the largest exponent is a `p`-th power for a prime `p` exactly when `p`
//! divides `K`, with the root `c^(K/p)`. So the primes `p` are tried in increasing order, each as
//! long as it gives a root, which replaces `x`: their product is `K`.
//!
//! - The exponent of 2 in `x`, and those of the odd primes to 47, are multiples of `K`. One
//!   remainder by the product of the squares of the odd primes to 29 and one by the product of
//!   the primes from 31 to 47 tell which divide `x`, and of those to 29 which only once: then `K`
//!   is 1, which settles almost every number. When they have a common divisor `g`, only the
//!   primes dividing `g` are tried.
//! - Otherwise each prime `p` up to the bits of `x` over `log2 53` is tried, for a root of at least
//!   53: first by residues, a `p`-th power being one modulo every prime: modulo 64 and the primes
//!   to 47 for squares, the primes `q = 1 mod p` to 47 for `p = 3, 5, 7, 11`, from the remainders
//!   already known, and up to four primes `q = 1 mod p` below `2^16` for the larger `p` whose root
//!   has more than 40 bits; then, when the root has at most 40 bits, by the floating-point root of
//!   the top bits, which is exact to 1, and its `p`-th power modulo `2^64` against the low limb, so
//!   that a large `p` costs a few products of limbs; otherwise by [`Number::nth_root`] or
//!   [`Number::isqrt_rem`], whose power, or remainder, is compared with `x`.
//! - Below `2^128`, the same in registers, the roots by the root's own routine for `u128`.

use super::super::div::limbs::{div_2by1, invert_limb};
use super::super::root::small_root;
use super::super::{ByteOrder, Number};
use super::field::{div_exact_1, gcd_u64, inverse_limb, limbs_of, number_of, trim};
use super::trial::{rem_by, shift_right};

/// The odd primes to 47, whose exponents in `x` the remainders give.
const SMALL: [u64; 14] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

/// The product of the squares of the odd primes to 29, below `2^64`.
const SQUARES: u64 = 9 * 25 * 49 * 121 * 169 * 289 * 361 * 529 * 841;

/// The product of the primes from 31 to 47.
const HIGH: u64 = 31 * 37 * 41 * 43 * 47;

/// `x` modulo the small primes: the remainders by [`SQUARES`] and [`HIGH`].
#[derive(Clone, Copy)]
struct Residues {
    /// `x mod SQUARES`.
    squares: u64,
    /// `x mod HIGH`.
    high: u64,
}

impl Residues {
    /// `x mod q` for a prime `q` of [`SMALL`].
    #[inline]
    fn of(&self, q: u64) -> u64 {
        if q <= 29 {
            self.squares % q
        } else {
            self.high % q
        }
    }

    /// The remainders of the limbs `y`, both in one pass, two independent chains a limb.
    fn limbs(y: &[u64]) -> Residues {
        const D1: u64 = SQUARES << SQUARES.leading_zeros();
        const D2: u64 = HIGH << HIGH.leading_zeros();
        let (v1, v2) = (invert_limb(D1), invert_limb(D2));
        let (mut r1, mut r2) = (0, 0);
        for &l in y.iter().rev() {
            r1 = div_2by1(r1, l, D1, v1).1;
            r2 = div_2by1(r2, l, D2, v2).1;
        }
        Residues {
            squares: r1 % SQUARES,
            high: r2 % HIGH,
        }
    }

    /// The remainders of `y`.
    fn u128(y: u128) -> Residues {
        Residues {
            squares: (y % u128::from(SQUARES)) as u64,
            high: (y % u128::from(HIGH)) as u64,
        }
    }
}

/// The `p`-th powers modulo `q < 64`, bit `r` set for each residue `r`.
const fn power_residues(p: u64, q: u64) -> u64 {
    let mut m = 0u64;
    let mut x = 0;
    while x < q {
        let mut y = 1;
        let mut i = 0;
        while i < p {
            y = y * x % q;
            i += 1;
        }
        m |= 1 << y;
        x += 1;
    }
    m
}

/// The residue filters: `(p, q, the p-th powers modulo q)` for `p = 2` and every odd prime `q` to
/// 47, and for `p = 3, 5, 7, 11` and the primes `q = 1 mod p` to 47.
const FILTERS: [(u64, u64, u64); 26] = {
    let pq: [(u64, u64); 26] = [
        (2, 3),
        (2, 5),
        (2, 7),
        (2, 11),
        (2, 13),
        (2, 17),
        (2, 19),
        (2, 23),
        (2, 29),
        (2, 31),
        (2, 37),
        (2, 41),
        (2, 43),
        (2, 47),
        (3, 7),
        (3, 13),
        (3, 19),
        (3, 31),
        (3, 37),
        (3, 43),
        (5, 11),
        (5, 31),
        (5, 41),
        (7, 29),
        (7, 43),
        (11, 23),
    ];
    let mut t = [(0, 0, 0); 26];
    let mut i = 0;
    while i < 26 {
        t[i] = (pq[i].0, pq[i].1, power_residues(pq[i].0, pq[i].1));
        i += 1;
    }
    t
};

/// The squares modulo 64, bit `r` set for each.
const SQUARES_64: u64 = power_residues(2, 64);

/// Whether the residues allow a `p`-th power: those of [`FILTERS`] for `p`, and modulo 64 for
/// squares, from the low limb `low`.
#[inline]
fn passes(res: &Residues, low: u64, p: u64) -> bool {
    if p == 2 && (SQUARES_64 >> (low & 63)) & 1 == 0 {
        return false;
    }
    FILTERS
        .iter()
        .all(|&(pp, q, mask)| pp != p || (mask >> res.of(q)) & 1 == 1)
}

/// The exponents of the odd primes to 47 in the odd `y`, through `divide(q)`, which gives that
/// of `q`, or 0 when it gave up counting: their greatest common divisor with `g` (that of 2, or 0
/// for none), and the smallest such prime (53 for none); `None` when the divisor is 1. For the
/// primes to 29, the remainder by their squares tells first whether one divides only once.
fn small_exponents(
    res: &Residues,
    mut g: u64,
    mut divide: impl FnMut(u64) -> u64,
) -> Option<(u64, u64)> {
    let mut smallest = 53;
    for &q in &SMALL {
        if res.of(q) != 0 {
            continue;
        }
        smallest = smallest.min(q);
        // `q` once only: `K` is 1.
        if q <= 29 && !res.squares.is_multiple_of(q * q) {
            return None;
        }
        let e = divide(q);
        if e == 1 {
            return None;
        }
        if e > 1 {
            g = gcd_u64(g, e);
            if g == 1 {
                return None;
            }
        }
    }
    Some((g, smallest))
}

/// The largest `K` with `x = c^K`, `x >= 2` given by its limbs, and the root `c`, or `None` when
/// `K` is 1; with `odd`, the largest odd `K`. `K` is at most `u32::MAX`: for numbers of more than
/// `2^32` bits, the product of the primes found while it fits.
pub(super) fn perfect_power_limbs(x: Vec<u64>, odd: bool) -> Option<(Vec<u64>, u32)> {
    let mut y = x;
    trim(&mut y);
    debug_assert!(y.len() > 1 || y[0] >= 2);
    if y.len() <= 2 {
        let v = u128::from(y[0]) | (u128::from(y.get(1).copied().unwrap_or(0)) << 64);
        return perfect_power_u128(v, odd).map(|(b, k)| {
            let mut b = vec![b as u64, (b >> 64) as u64];
            trim(&mut b);
            (b, k)
        });
    }
    // `x = 2^z y`, `y` odd.
    let z = trailing_zeros(&y);
    if z > 0 {
        shift_right(&mut y, z);
    }
    if y == [1] {
        // `x = 2^z`: `K = z`, or its odd part.
        let k = if odd { z >> z.trailing_zeros() } else { z };
        if k < 2 {
            return None;
        }
        let k = u32::try_from(k).ok()?;
        let e = z / u64::from(k);
        let mut b = vec![0u64; (e / 64 + 1) as usize];
        b[(e / 64) as usize] = 1 << (e % 64);
        return Some((b, k));
    }
    if z == 1 {
        return None;
    }
    let mut res = Residues::limbs(&y);
    let (mut g, smallest) = small_exponents(&res, z, |q| {
        // A pass of the number a division: given up after 64, the exponent then left unknown.
        let inv = inverse_limb(q);
        let mut t = y.clone();
        let mut e = 0;
        while let Some(d) = div_exact_1(&t, q, inv) {
            t = d;
            e += 1;
            if e > 64 {
                return 0;
            }
        }
        e
    })?;
    let mut k: u64 = 1;
    let mut p = if odd { 3 } else { 2 };
    'primes: loop {
        if g != 0 && g < p {
            break;
        }
        if g == 0 || g % p == 0 {
            loop {
                let bits = bit_length(&y);
                // A `p`-th power of a root of at least `smallest` has at least `p log2(smallest)`
                // bits.
                if (p as f64) * (smallest as f64).log2() > bits as f64 + 0.5 {
                    break 'primes;
                }
                if k.saturating_mul(p) > u64::from(u32::MAX) {
                    break 'primes;
                }
                if !passes(&res, y[0], p) {
                    break;
                }
                let Some(root) = root_if_power(&y, p, bits) else {
                    break;
                };
                y = root;
                res = Residues::limbs(&y);
                k *= p;
                if g != 0 {
                    g /= p;
                    if g % p != 0 {
                        break;
                    }
                }
            }
        } else if (p as f64) * (smallest as f64).log2() > bit_length(&y) as f64 + 0.5 {
            break;
        }
        p = next_prime_small(p);
    }
    if k == 1 {
        return None;
    }
    // The root: `2^(z/k) y`.
    let e = z / k;
    let mut b = y;
    if e > 0 {
        b = shift_left(&b, e);
    }
    Some((b, k as u32))
}

/// [`perfect_power_limbs`] for `2 <= x < 2^128`, in registers.
pub(super) fn perfect_power_u128(x: u128, odd: bool) -> Option<(u128, u32)> {
    let z = u64::from(x.trailing_zeros());
    let mut y = x >> z;
    if y == 1 {
        let k = if odd { z >> z.trailing_zeros() } else { z };
        if k < 2 {
            return None;
        }
        return Some((1 << (z / k), k as u32));
    }
    if z == 1 {
        return None;
    }
    let mut res = Residues::u128(y);
    let (mut g, smallest) = small_exponents(&res, z, |q| {
        let (mut t, mut e) = (y, 0);
        while t % u128::from(q) == 0 {
            t /= u128::from(q);
            e += 1;
        }
        e
    })?;
    let mut k: u64 = 1;
    let mut p: u64 = if odd { 3 } else { 2 };
    'primes: loop {
        if g != 0 && g < p {
            break;
        }
        let bits = 128 - u64::from(y.leading_zeros());
        if (p as f64) * (smallest as f64).log2() > bits as f64 + 0.5 {
            break;
        }
        if g == 0 || g % p == 0 {
            loop {
                if !passes(&res, y as u64, p) {
                    break;
                }
                let r = small_root(y, p as u32);
                if r.checked_pow(p as u32) != Some(y) {
                    break;
                }
                y = r;
                res = Residues::u128(y);
                k *= p;
                if g != 0 {
                    g /= p;
                    if g % p != 0 {
                        break;
                    }
                }
                let bits = 128 - u64::from(y.leading_zeros());
                if (p as f64) * (smallest as f64).log2() > bits as f64 + 0.5 {
                    break 'primes;
                }
            }
        }
        p = next_prime_small(p);
    }
    if k == 1 {
        return None;
    }
    Some((y << (z / k), k as u32))
}

/// The `p`-th root of the odd `y > 1` of `bits` bits when it is a `p`-th power.
fn root_if_power(y: &[u64], p: u64, bits: u64) -> Option<Vec<u64>> {
    if p == 2 {
        let n = number_of(y, ByteOrder::LittleEndian);
        let (root, rest) = n.isqrt_rem();
        return if rest.is_zero() {
            Some(limbs_of(&root))
        } else {
            None
        };
    }
    let root_bits = bits.div_ceil(p);
    if root_bits <= 40 {
        // The root from the top bits: `log2 y` to 2^-50 or so, so the root to well below 1, and
        // its power checked modulo `2^64` first.
        let est = float_root(y, bits, p);
        let low = y[0];
        for c in [est.saturating_sub(1), est, est + 1] {
            if c >= 3 && pow_low(c, p) == low {
                let n = Number::new_from_u64(c, ByteOrder::LittleEndian);
                if n.pow(p as u32) == number_of(y, ByteOrder::LittleEndian) {
                    return Some(vec![c]);
                }
            }
        }
        return None;
    }
    // Residues modulo primes `q = 1 mod p` below 2^16: a `p`-th power is one modulo each.
    if p > 11 && !passes_residues(y, p) {
        return None;
    }
    let n = number_of(y, ByteOrder::LittleEndian);
    let root = n.nth_root(p as u32);
    if root.pow(p as u32) == n {
        Some(limbs_of(&root))
    } else {
        None
    }
}

/// Whether `y` is a `p`-th power residue modulo the first primes `q = 1 mod p` below `2^16`, up
/// to four of them: `y^((q-1)/p) = 1 mod q`, or `q` divides `y`.
fn passes_residues(y: &[u64], p: u64) -> bool {
    let mut found = 0;
    let mut q = 2 * p + 1;
    while q < 1 << 16 && found < 4 {
        if is_prime_small(q) {
            found += 1;
            let s = q.leading_zeros();
            let r = rem_by(y, q << s, invert_limb(q << s)) % q;
            if r != 0 && pow_mod(r, (q - 1) / p, q) != 1 {
                return false;
            }
        }
        q += 2 * p;
    }
    true
}

/// `b^e mod m` for `m < 2^32`.
fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1 % m;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

/// `c^p mod 2^64`.
fn pow_low(c: u64, mut p: u64) -> u64 {
    let (mut b, mut r) = (c, 1u64);
    while p > 0 {
        if p & 1 == 1 {
            r = r.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        p >>= 1;
    }
    r
}

/// The `p`-th root of `y`, of `bits` bits, rounded to the nearest integer, from its top 64 bits.
fn float_root(y: &[u64], bits: u64, p: u64) -> u64 {
    // The top 64 bits, as `t 2^(bits - 64)`, or `y` itself.
    let (t, shift) = if bits <= 64 {
        (y[0] as f64, 0u64)
    } else {
        let s = bits - 64;
        let (l, b) = ((s / 64) as usize, (s % 64) as u32);
        let mut top = y[l] >> b;
        if b != 0 && l + 1 < y.len() {
            top |= y[l + 1] << (64 - b);
        }
        (top as f64, s)
    };
    let log = (t.log2() + shift as f64) / p as f64;
    let r = log.exp2().round();
    if r >= 1.8e19 { u64::MAX } else { r as u64 }
}

/// Whether the odd `q` is prime, by trial division (`q` stays small: below `2^16`, or below the
/// bits of the number).
fn is_prime_small(q: u64) -> bool {
    let mut d = 3;
    while d * d <= q {
        if q.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    q > 1
}

/// The next prime after `p`.
fn next_prime_small(p: u64) -> u64 {
    if p == 2 {
        return 3;
    }
    let mut q = p + 2;
    while !is_prime_small(q) {
        q += 2;
    }
    q
}

/// The number of bits of the limbs `x`, the top one not zero.
fn bit_length(x: &[u64]) -> u64 {
    64 * x.len() as u64 - u64::from(x.last().map_or(64, |l| l.leading_zeros()))
}

/// The number of zero bits at the bottom of the limbs `x`, not all zero.
fn trailing_zeros(x: &[u64]) -> u64 {
    let mut z = 0u64;
    for &l in x {
        if l != 0 {
            return z + u64::from(l.trailing_zeros());
        }
        z += 64;
    }
    z
}

/// The limbs `x` shifted left by `k` bits.
fn shift_left(x: &[u64], k: u64) -> Vec<u64> {
    let (l, s) = ((k / 64) as usize, (k % 64) as u32);
    let mut r = vec![0u64; l];
    r.extend_from_slice(x);
    r.push(0);
    if s != 0 {
        for i in (l + 1..r.len()).rev() {
            r[i] = (r[i] << s) | (r[i - 1] >> (64 - s));
        }
        r[l] <<= s;
    }
    trim(&mut r);
    r
}
