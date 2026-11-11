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

/// `x` modulo the small primes: the remainders by [`SQUARES`] and [`HIGH`], and, once the filters
/// need them, the remainders by the primes of [`SMALL`].
#[derive(Clone, Copy)]
struct Residues {
    /// `x mod SQUARES`.
    squares: u64,
    /// `x mod HIGH`.
    high: u64,
    /// `x mod q` for the primes `q` of [`SMALL`], in their order, when computed.
    small: [u8; 14],
}

impl Residues {
    /// The residues from the remainders by [`SQUARES`] and [`HIGH`].
    #[inline]
    fn from(squares: u64, high: u64) -> Residues {
        Residues {
            squares,
            high,
            small: [0; 14],
        }
    }

    /// `x mod SMALL[i]`, by a constant divisor once inlined in a loop over [`SMALL`].
    #[inline(always)]
    fn at(&self, i: usize) -> u64 {
        if SMALL[i] <= 29 {
            self.squares % SMALL[i]
        } else {
            self.high % SMALL[i]
        }
    }

    /// The remainders by the primes of [`SMALL`], computed now, for [`passes`].
    #[inline]
    fn with_small(mut self) -> Residues {
        for i in 0..SMALL.len() {
            self.small[i] = self.at(i) as u8;
        }
        self
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
        Residues::from(r1 % SQUARES, r2 % HIGH)
    }

    /// The remainders of `y`.
    fn u128(y: u128) -> Residues {
        Residues::from(
            (y % u128::from(SQUARES)) as u64,
            (y % u128::from(HIGH)) as u64,
        )
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

/// The range of the indices in [`FILTERS`] of the filters for `p`, which are consecutive.
const fn filters_of(p: u64) -> (usize, usize) {
    let (mut lo, mut i) = (FILTERS.len(), 0);
    while i < FILTERS.len() {
        if FILTERS[i].0 == p && lo == FILTERS.len() {
            lo = i;
        }
        i += 1;
    }
    let mut hi = lo;
    while hi < FILTERS.len() && FILTERS[hi].0 == p {
        hi += 1;
    }
    (lo, hi)
}

/// The index of the prime `q` in [`SMALL`].
const fn small_index(q: u64) -> usize {
    let mut i = 0;
    while SMALL[i] != q {
        i += 1;
    }
    i
}

/// The filters of [`FILTERS`] as `(index of q in SMALL, mask)`, in the same order.
const FILTER_INDEX: [(usize, u64); 26] = {
    let mut t = [(0, 0); 26];
    let mut i = 0;
    while i < 26 {
        t[i] = (small_index(FILTERS[i].1), FILTERS[i].2);
        i += 1;
    }
    t
};

/// The ranges of [`FILTERS`] for `p = 2, 3, 5, 7, 11`.
const FILTER_RANGES: [(usize, usize); 5] = [
    filters_of(2),
    filters_of(3),
    filters_of(5),
    filters_of(7),
    filters_of(11),
];

/// Whether the residues allow a `p`-th power: those of [`FILTERS`] for `p`, and modulo 64 for
/// squares, from the low limb `low`.
#[inline]
fn passes(res: &Residues, low: u64, p: u64) -> bool {
    if p == 2 && (SQUARES_64 >> (low & 63)) & 1 == 0 {
        return false;
    }
    let (lo, hi) = match p {
        2 => FILTER_RANGES[0],
        3 => FILTER_RANGES[1],
        5 => FILTER_RANGES[2],
        7 => FILTER_RANGES[3],
        11 => FILTER_RANGES[4],
        _ => return true,
    };
    FILTER_INDEX[lo..hi]
        .iter()
        .all(|&(i, mask)| (mask >> res.small[i]) & 1 == 1)
}

/// The exponents of the odd primes to 47 in the odd `y`, through `divide(q)`, which gives that
/// of `q`, or 0 when it gave up counting: their greatest common divisor with `g` (that of 2, or 0
/// for none), and the smallest such prime (53 for none); `None` when the divisor is 1. For the
/// primes to 29, the remainder by their squares tells first whether one divides only once.
fn small_exponents(
    res: &mut Residues,
    mut g: u64,
    mut divide: impl FnMut(u64) -> u64,
) -> Option<(u64, u64)> {
    let mut smallest = 53;
    for (i, &q) in SMALL.iter().enumerate() {
        // Kept for the filters, which run only when the loop ends.
        let r = res.at(i);
        res.small[i] = r as u8;
        if r != 0 {
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
    let (mut g, smallest) = small_exponents(&mut res, z, |q| {
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
    let log_smallest = (smallest as f64).log2();
    let mut ylog = log2_limbs(&y);
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
                if (p as f64) * log_smallest > bits as f64 + 0.5 {
                    break 'primes;
                }
                if k.saturating_mul(p) > u64::from(u32::MAX) {
                    break 'primes;
                }
                if !passes(&res, y[0], p) {
                    break;
                }
                let Some(root) = root_if_power(&y, p, bits, ylog) else {
                    break;
                };
                y = root;
                ylog = log2_limbs(&y);
                res = Residues::limbs(&y).with_small();
                k *= p;
                if g != 0 {
                    g /= p;
                    if g % p != 0 {
                        break;
                    }
                }
            }
        } else if (p as f64) * log_smallest > bit_length(&y) as f64 + 0.5 {
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
    if x >> 64 == 0 {
        return perfect_power_u64(x as u64, odd).map(|(b, k)| (u128::from(b), k));
    }
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
    let (mut g, smallest) = small_exponents(&mut res, z, |q| {
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
                res = Residues::u128(y).with_small();
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

/// [`perfect_power_u128`] for `2 <= x < 2^64`: the remainders by constant divisors, the roots in
/// floating point, checked exactly.
fn perfect_power_u64(x: u64, odd: bool) -> Option<(u64, u32)> {
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
    let mut res = Residues::from(y % SQUARES, y % HIGH);
    let (mut g, smallest) = small_exponents(&mut res, z, |q| {
        let (mut t, mut e) = (y, 0);
        while t % q == 0 {
            t /= q;
            e += 1;
        }
        e
    })?;
    let log_smallest = (smallest as f64).log2();
    let mut k: u64 = 1;
    let mut p: u64 = if odd { 3 } else { 2 };
    'primes: loop {
        if g != 0 && g < p {
            break;
        }
        let bits = 64 - u64::from(y.leading_zeros());
        if p as f64 * log_smallest > bits as f64 + 0.5 {
            break;
        }
        if g == 0 || g % p == 0 {
            loop {
                if !passes(&res, y, p) {
                    break;
                }
                let Some(r) = root_u64(y, p) else {
                    break;
                };
                y = r;
                res = Residues::from(y % SQUARES, y % HIGH).with_small();
                k *= p;
                if g != 0 {
                    g /= p;
                    if g % p != 0 {
                        break;
                    }
                }
                let bits = 64 - u64::from(y.leading_zeros());
                if p as f64 * log_smallest > bits as f64 + 0.5 {
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

/// The `p`-th root of `y` when it is a `p`-th power: the square root exactly, other roots from the
/// floating-point root, exact to 1, and checked.
#[inline]
fn root_u64(y: u64, p: u64) -> Option<u64> {
    if p == 2 {
        let r = super::primes::isqrt(y);
        return if r * r == y { Some(r) } else { None };
    }
    let f = y as f64;
    let est = if p == 3 {
        f.cbrt()
    } else {
        f.powf(1.0 / p as f64)
    };
    let r = est.round() as u64;
    [r, r.wrapping_sub(1), r + 1]
        .into_iter()
        .find(|&c| c >= 2 && c.checked_pow(p as u32) == Some(y))
}

/// The `p`-th root of the odd `y > 1` of `bits` bits, whose base-2 logarithm is `ylog`, when it is
/// a `p`-th power.
fn root_if_power(y: &[u64], p: u64, bits: u64, ylog: f64) -> Option<Vec<u64>> {
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
        // The root from the top bits: `log2 y` is known to `2^-52 log2 y`, so the root of a power,
        // below `2^40`, to `2^-6`: a root further from an integer rules the power out, and an
        // integer root is checked modulo `2^64` first.
        let r = (ylog / p as f64).exp2();
        let c = r.round();
        if (r - c).abs() > 1.0 / 64.0 || c < 3.0 {
            return None;
        }
        let c = c as u64;
        if pow_low(c, p) == y[0] {
            let n = Number::new_from_u64(c, ByteOrder::LittleEndian);
            if n.pow(p as u32) == number_of(y, ByteOrder::LittleEndian) {
                return Some(vec![c]);
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

/// The base-2 logarithm of the limbs `y`, the top one not zero, from their top 64 bits: to about
/// `2^-50`.
fn log2_limbs(y: &[u64]) -> f64 {
    let bits = bit_length(y);
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
    t.log2() + shift as f64
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
    // The primes to 127 from a mask, bit `q` set for each.
    const MASK: u128 = {
        let mut m = 0u128;
        let mut q = 2;
        while q < 128 {
            let mut d = 2;
            let mut prime = true;
            while d * d <= q {
                if q % d == 0 {
                    prime = false;
                }
                d += 1;
            }
            if prime {
                m |= 1 << q;
            }
            q += 1;
        }
        m
    };
    if p < 113 {
        return u64::from((MASK >> (p + 1)).trailing_zeros()) + p + 1;
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
