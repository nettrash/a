//! The index calculus for logarithms modulo a prime `p` of at most [`MOST_BITS`] bits, in its
//! subgroup of a large prime order `q`: the linear sieve of Coppersmith, Odlyzko and Schroeppel
//! (1986), the unknowns of its relations eliminated along a spanning forest, the logarithms of a
//! base of small primes by Gaussian elimination modulo `q`, and each logarithm then by a rational
//! reconstruction whose two halves are smooth.
//!
//! With `H = ceil(sqrt(p))` and `J = H^2 - p`, `(H + c1)(H + c2) = J + (c1 + c2) H + c1 c2
//! (mod p)`, a number of about `2 C sqrt(p)` for `|c1|, |c2| <= C`; when it factors over the primes
//! up to `B`, its logarithm gives a relation `L(H + c1) + L(H + c2) = sum e_l L(l)` modulo `q`
//! (`L(-1)` is 0 modulo the odd `q`). For each `c1` the numbers are a progression in `c2`, sieved
//! by each prime from the root of `(H + c1)(H + c2) = p (mod l)`. The `L(H + c)` are eliminated
//! through the graph of the `c` with an edge for each relation: along a spanning forest each is
//! `±L(H + root) + F` for a combination `F` of the base, and each edge off the forest gives a
//! relation of the base alone. These determine the base's logarithms up to a common factor, the
//! one-dimensional kernel of the relations modulo `q`, which is all a quotient of two logarithms
//! needs; they are reduced as they come, each entry once, from a sum of products not reduced.
//! The logarithm of any `z` comes from a multiple `z w^e = u / v (mod p)` with `|u|, |v| <
//! sqrt(p)` (Euclid's algorithm stopped halfway) whose `u` and `v` both factor over the base but
//! for at most one prime each a little beyond it, for a fixed `w` of known logarithm; such a
//! prime's logarithm comes from a pair of the sieve's range whose number it divides, smooth after
//! it, the `L(H + c)` following from the base's down the forest.

use super::super::modular::word::{Odd, inverse, isqrt};
use super::super::pow::montgomery::{Mont1, Mont2};
use super::super::pow::window::Registers;
use super::super::prime::tables::PRIMES;

/// The largest primes taken, in bits: beyond, the base would need primes past the tables'.
pub(in super::super) const MOST_BITS: u32 = 112;

/// The sieve's parameters for a prime of `bits` bits: the bound of the factor base `B` and the half
/// width `C` of the range of `c1` and `c2`, the fastest of those measured on an Apple M4 Max modulo
/// primes `2q + 1` (when chosen, in milliseconds for a logarithm: 40 bits 0.2, 48 bits 0.4, 56 bits
/// 0.8, 64 bits 2.0, 72 bits 6.1, 80 bits 17, 88 bits 51, 96 bits 85, 104 bits 232, 112 bits 687;
/// with the second level of the descent and the scan by words since: 0.1 at 40 bits, 0.6 at 56, 1.7
/// at 64, 4.7 at 72, 13 at 80, 30 at 88, 72 at 96, 181 at 104 and 538 at 112). A larger base makes
/// relations likelier but the linear algebra longer, a wider range more relations but larger
/// numbers; too few relations to fix the logarithms cost a second sieve.
fn parameters(bits: u32) -> (u32, i64) {
    match bits {
        0..=40 => (192, 100),
        41..=44 => (256, 160),
        45..=48 => (320, 220),
        49..=52 => (384, 280),
        53..=56 => (512, 330),
        57..=60 => (640, 450),
        61..=64 => (768, 600),
        65..=68 => (1152, 640),
        69..=72 => (1152, 900),
        73..=76 => (1536, 1100),
        77..=80 => (2048, 1300),
        81..=84 => (2560, 1700),
        85..=88 => (2560, 2600),
        89..=92 => (3072, 3400),
        93..=96 => (3584, 4200),
        97..=100 => (4096, 5400),
        101..=104 => (5120, 6500),
        105..=108 => (6144, 9000),
        _ => (8191, 11000),
    }
}

/// The largest bound of the base: the tables' primes are below 8192.
const MOST_PRIME: u32 = 8191;

/// The primes beyond the base taken in the halves of a logarithm are up to this many times the
/// relations: about one pair of the range in `l` has its number divisible by `l`, and of those as
/// many are smooth after it as the relations were among all.
const SECOND: u64 = 4;

/// The primes below this are not sieved: their logarithms are small, their progressions long.
const UNSIEVED: u32 = 8;

/// The bits a sieved position may lack and still be factored: the unsieved primes, the powers and
/// the rounding of the logarithms.
const SLACK: u32 = 10;

/// The rows sieved together, prime by prime.
const ROWS: usize = 16;

/// The most multiples tried in the logarithm of one element.
const MOST_TRIES: u32 = 1 << 22;

/// Arithmetic modulo an odd prime of at most 128 bits in Montgomery's form, in one limb ([`One`])
/// or two ([`Two`]).
trait Field {
    /// A residue.
    type V: Copy + PartialEq;

    /// The residue of 0.
    fn zero(&self) -> Self::V;

    /// The residue of 1.
    fn one(&self) -> Self::V;

    /// Differences and products.
    fn sub(&self, a: Self::V, b: Self::V) -> Self::V;
    fn mul(&self, a: Self::V, b: Self::V) -> Self::V;

    /// The residue of `x`, any size.
    fn of(&self, x: u128) -> Self::V;

    /// The plain number of a residue.
    fn plain(&self, a: Self::V) -> u128;

    /// The inverse of a residue not 0.
    fn inv(&self, a: Self::V) -> Self::V;

    /// The plain residue of the small `x`, of either sign, `|x|` below the modulus.
    fn plain_small(&self, x: i64) -> Self::V;

    /// The plain inverse of the plain residue `a`, not 0.
    fn plain_inv(&self, a: Self::V) -> Self::V;

    /// The number of the plain residue `a`.
    fn number(&self, a: Self::V) -> u128;

    /// `a R^2 mod m` for the plain `a`: the plain `b` times it is the residue of `a b`.
    fn scale(&self, a: Self::V) -> Self::V;

    /// A sum of products of residues, not reduced.
    type Wide: Copy;

    /// The empty sum.
    fn wide_zero(&self) -> Self::Wide;

    /// `acc += a b`, not reduced, for at most `2^10` terms.
    fn wide_add(&self, acc: &mut Self::Wide, a: Self::V, b: Self::V);

    /// The sum divided by `R` modulo `m`, below `m`: with `a` plain and `b` in Montgomery's form,
    /// the plain residue of the sum of the `a b`.
    fn wide_reduce(&self, acc: Self::Wide) -> Self::V;
}

/// Montgomery's arithmetic modulo an odd number of one limb, with what brings plain numbers in.
#[derive(Clone, Copy)]
struct One {
    ring: Mont1,
    /// The modulus.
    m: u64,
    /// `R mod m` and `R^2 mod m`, `R = 2^64`.
    one: u64,
    r2: u64,
}

impl One {
    /// The arithmetic modulo the odd `m`.
    fn new(m: u64) -> One {
        let one = ((1u128 << 64) % u128::from(m)) as u64;
        let r2 = ((u128::from(one) * u128::from(one)) % u128::from(m)) as u64;
        One {
            ring: Mont1::new(m),
            m,
            one,
            r2,
        }
    }
}

impl Field for One {
    type V = u64;
    type Wide = (u128, u64);

    #[inline(always)]
    fn zero(&self) -> u64 {
        0
    }

    #[inline(always)]
    fn one(&self) -> u64 {
        self.one
    }

    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        let (d, borrow) = a.overflowing_sub(b);
        if borrow { d.wrapping_add(self.m) } else { d }
    }

    #[inline(always)]
    fn mul(&self, a: u64, b: u64) -> u64 {
        self.ring.mul(a, b)
    }

    #[inline]
    fn of(&self, x: u128) -> u64 {
        let m = u128::from(self.m);
        self.ring.mul(if x < m { x } else { x % m } as u64, self.r2)
    }

    #[inline]
    fn plain(&self, a: u64) -> u128 {
        u128::from(self.ring.plain(a))
    }

    fn inv(&self, a: u64) -> u64 {
        self.of(u128::from(self.plain_inv(self.ring.plain(a))))
    }

    #[inline]
    fn plain_small(&self, x: i64) -> u64 {
        let a = x.unsigned_abs();
        if x < 0 && a != 0 { self.m - a } else { a }
    }

    fn plain_inv(&self, a: u64) -> u64 {
        // A residue not 0 of the prime is a unit.
        inverse(a, self.m).unwrap_or(0)
    }

    #[inline]
    fn number(&self, a: u64) -> u128 {
        u128::from(a)
    }

    fn scale(&self, a: u64) -> u64 {
        self.of(self.of(u128::from(a)).into())
    }

    #[inline(always)]
    fn wide_zero(&self) -> (u128, u64) {
        (0, 0)
    }

    #[inline(always)]
    fn wide_add(&self, acc: &mut (u128, u64), a: u64, b: u64) {
        let (s, carry) = acc.0.overflowing_add(u128::from(a) * u128::from(b));
        acc.0 = s;
        acc.1 += u64::from(carry);
    }

    #[inline]
    fn wide_reduce(&self, acc: (u128, u64)) -> u64 {
        // `acc = t B + l` with `t` of three limbs: `t mod m`, then REDC.
        let t = (u128::from(acc.1) << 64) | (acc.0 >> 64);
        let t = if t < u128::from(self.m) {
            t as u64
        } else {
            (t % u128::from(self.m)) as u64
        };
        self.ring.redc(t, acc.0 as u64)
    }
}

/// Montgomery's arithmetic modulo an odd number of two limbs, with what brings plain numbers in.
#[derive(Clone, Copy)]
struct Two {
    ring: Mont2,
    /// The modulus.
    m: u128,
    /// `R mod m` and `R^2 mod m`, `R = 2^128`.
    one: [u64; 2],
    r2: [u64; 2],
}

/// The two limbs of `x`.
#[inline(always)]
fn limbs(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

/// The number of two limbs.
#[inline(always)]
fn join(x: [u64; 2]) -> u128 {
    u128::from(x[0]) | (u128::from(x[1]) << 64)
}

/// `2 x mod m` for `x < m`.
#[inline(always)]
fn double(x: u128, m: u128) -> u128 {
    let (d, carry) = x.overflowing_add(x);
    if carry || d >= m {
        d.wrapping_sub(m)
    } else {
        d
    }
}

impl Two {
    /// The arithmetic modulo the odd `m >= 2^64`.
    fn new(m: u128) -> Two {
        // `R mod m`, then `R^2 mod m` by 128 doublings.
        let one = (u128::MAX % m + 1) % m;
        let mut r2 = one;
        for _ in 0..128 {
            r2 = double(r2, m);
        }
        Two {
            ring: Mont2::new(limbs(m)),
            m,
            one: limbs(one),
            r2: limbs(r2),
        }
    }
}

impl Field for Two {
    type V = [u64; 2];
    type Wide = (u128, u128, u128);

    #[inline(always)]
    fn zero(&self) -> [u64; 2] {
        [0, 0]
    }

    #[inline(always)]
    fn one(&self) -> [u64; 2] {
        self.one
    }

    #[inline(always)]
    fn sub(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (d, borrow) = join(a).overflowing_sub(join(b));
        limbs(if borrow { d.wrapping_add(self.m) } else { d })
    }

    #[inline(always)]
    fn mul(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        self.ring.mul(a, b)
    }

    #[inline]
    fn of(&self, x: u128) -> [u64; 2] {
        self.ring
            .mul(limbs(if x < self.m { x } else { x % self.m }), self.r2)
    }

    #[inline]
    fn plain(&self, a: [u64; 2]) -> u128 {
        join(self.ring.plain(a))
    }

    fn inv(&self, a: [u64; 2]) -> [u64; 2] {
        // Fermat: `a^(m - 2)`.
        let e = self.m - 2;
        let mut r = self.one;
        for i in (0..128 - e.leading_zeros()).rev() {
            r = self.ring.sqr(r);
            if (e >> i) & 1 == 1 {
                r = self.ring.mul(r, a);
            }
        }
        r
    }

    #[inline]
    fn plain_small(&self, x: i64) -> [u64; 2] {
        let a = u128::from(x.unsigned_abs());
        limbs(if x < 0 && a != 0 { self.m - a } else { a })
    }

    fn plain_inv(&self, a: [u64; 2]) -> [u64; 2] {
        // The residue of `a` inverted by Fermat's theorem, out of Montgomery's form.
        limbs(self.plain(self.inv(self.of(join(a)))))
    }

    #[inline]
    fn number(&self, a: [u64; 2]) -> u128 {
        join(a)
    }

    fn scale(&self, a: [u64; 2]) -> [u64; 2] {
        self.of(join(self.of(join(a))))
    }

    #[inline(always)]
    fn wide_zero(&self) -> (u128, u128, u128) {
        (0, 0, 0)
    }

    #[inline(always)]
    fn wide_add(&self, acc: &mut (u128, u128, u128), a: [u64; 2], b: [u64; 2]) {
        // Below `m < 2^117`, the high limbs are below `2^53`: the middle and high sums of up to
        // `2^10` terms do not overflow; the low sum carries into the high.
        let (lo, carry) = acc.0.overflowing_add(u128::from(a[0]) * u128::from(b[0]));
        acc.0 = lo;
        acc.1 += u128::from(a[0]) * u128::from(b[1]) + u128::from(a[1]) * u128::from(b[0]);
        acc.2 += u128::from(a[1]) * u128::from(b[1]) + u128::from(carry);
    }

    #[inline]
    fn wide_reduce(&self, acc: (u128, u128, u128)) -> [u64; 2] {
        // `lo + mid B + hi B^2`, below `2^11 m^2 < R m`: REDC.
        let (lo, mid, hi) = acc;
        let v1 = u128::from((lo >> 64) as u64) + u128::from(mid as u64);
        let v2 = u128::from((v1 >> 64) as u64) + (mid >> 64) + u128::from(hi as u64);
        let v3 = ((v2 >> 64) as u64).wrapping_add((hi >> 64) as u64);
        self.ring.redc([lo as u64, v1 as u64, v2 as u64, v3])
    }
}

/// The base of small primes: 2 and the odd primes up to the bound, with what tells divisibility
/// by each with a multiplication, modulo `2^64` and modulo `2^128`.
#[derive(Clone)]
struct Base {
    /// The primes, 2 first.
    primes: Vec<u32>,
    /// The inverse of each odd prime modulo `2^64` (0 for 2) and `floor((2^64 - 1) / l)`.
    inv: Vec<u64>,
    max: Vec<u64>,
    /// The same modulo `2^128`.
    inv2: Vec<u128>,
    max2: Vec<u128>,
}

impl Base {
    /// The primes up to `bound`, at most the tables' limit.
    fn new(bound: u32) -> Base {
        let mut b = Base {
            primes: vec![2u32],
            inv: vec![0],
            max: vec![u64::MAX / 2],
            inv2: vec![0],
            max2: vec![u128::MAX / 2],
        };
        for p in PRIMES.iter() {
            if p.p > u64::from(bound) {
                break;
            }
            b.primes.push(p.p as u32);
            b.inv.push(p.inv);
            b.max.push(p.max);
            // Newton's step from the inverse modulo `2^64` to one modulo `2^128`.
            let (l, v) = (u128::from(p.p), u128::from(p.inv));
            b.inv2
                .push(v.wrapping_mul(2u128.wrapping_sub(l.wrapping_mul(v))));
            b.max2.push(u128::MAX / l);
        }
        b
    }

    /// The exponents of `x >= 1` over the base, pushed to `out` as (index, exponent), when it
    /// factors over it: `false` otherwise, `out` then in any state.
    #[inline]
    fn factor(&self, x: u64, out: &mut Vec<(u32, u32)>) -> bool {
        out.clear();
        let z = x.trailing_zeros();
        if z > 0 {
            out.push((0, z));
        }
        self.factor_odd(x >> z, 1, out)
    }

    /// [`Base::factor`] for the odd `x` with no factor below the `k`-th prime: eight primes at a
    /// time are tested without a branch, and only a group with one that divides is looked at more
    /// closely, as most of them do not.
    #[inline]
    fn factor_odd(&self, mut x: u64, from: usize, out: &mut Vec<(u32, u32)>) -> bool {
        let n = self.primes.len();
        let mut k = from;
        while k < n {
            if x == 1 {
                return true;
            }
            let l = u64::from(self.primes[k]);
            if l * l > x {
                // No factor below `l`: `x` is a prime, in the base or not.
                let top = u64::from(self.primes[n - 1]);
                return x <= top && self.push_prime(x, out);
            }
            let end = (k + 8).min(n);
            let any = (k..end).fold(false, |a, i| {
                a | (x.wrapping_mul(self.inv[i]) <= self.max[i])
            });
            if any {
                for i in k..end {
                    let (inv, max) = (self.inv[i], self.max[i]);
                    let mut y = x.wrapping_mul(inv);
                    if y <= max {
                        let mut e = 0;
                        while y <= max {
                            x = y;
                            e += 1;
                            y = x.wrapping_mul(inv);
                        }
                        out.push((i as u32, e));
                    }
                }
            }
            k = end;
        }
        x == 1
    }

    /// The exponents of `x >= 1` over the base, pushed to `out`, and what is left, which has no
    /// prime of the base: eight primes at a time tested without a branch, as in
    /// [`Base::factor_odd`].
    fn factor_rest(&self, x: u64, out: &mut Vec<(u32, u32)>) -> u64 {
        out.clear();
        let z = x.trailing_zeros();
        if z > 0 {
            out.push((0, z));
        }
        let mut x = x >> z;
        let n = self.primes.len();
        let mut k = 1;
        while k < n && x != 1 {
            let end = (k + 8).min(n);
            let any = (k..end).fold(false, |a, i| {
                a | (x.wrapping_mul(self.inv[i]) <= self.max[i])
            });
            if any {
                for i in k..end {
                    let (inv, max) = (self.inv[i], self.max[i]);
                    let mut y = x.wrapping_mul(inv);
                    if y <= max {
                        let mut e = 0;
                        while y <= max {
                            x = y;
                            e += 1;
                            y = x.wrapping_mul(inv);
                        }
                        out.push((i as u32, e));
                    }
                }
            }
            k = end;
        }
        x
    }

    /// [`Base::factor`] for `x` of up to 128 bits: on two limbs until what is left fits in one.
    fn factor_wide(&self, x: u128, out: &mut Vec<(u32, u32)>) -> bool {
        if x >> 64 == 0 {
            return self.factor(x as u64, out);
        }
        out.clear();
        let z = x.trailing_zeros();
        if z > 0 {
            out.push((0, z));
        }
        let mut x = x >> z;
        for k in 1..self.primes.len() {
            if x >> 64 == 0 {
                return self.factor_odd(x as u64, k, out);
            }
            let (inv, max) = (self.inv2[k], self.max2[k]);
            let mut y = x.wrapping_mul(inv);
            if y <= max {
                let mut e = 0;
                while y <= max {
                    x = y;
                    e += 1;
                    y = x.wrapping_mul(inv);
                }
                out.push((k as u32, e));
            }
        }
        false
    }

    /// Pushes the prime `x` of the base, found by its place.
    fn push_prime(&self, x: u64, out: &mut Vec<(u32, u32)>) -> bool {
        match self.primes.binary_search(&(x as u32)) {
            Ok(k) => {
                out.push((k as u32, 1));
                true
            }
            Err(_) => false,
        }
    }
}

/// The relations of one sieve modulo a prime, kept for the other large primes of its `p - 1`.
pub(in super::super) struct Sieve {
    /// The prime.
    p: u128,
    /// The bound of the base and the half width of the range.
    bound: u32,
    half: i64,
    /// The base, the relations and their exponents.
    base: Base,
    relations: Vec<Relation>,
    exponents: Vec<(u32, u32)>,
}

impl Sieve {
    /// The sieve modulo `p` with the base up to `bound` over `[-half, half]`.
    fn new(p: u128, bound: u32, half: i64) -> Sieve {
        let base = Base::new(bound);
        let (relations, exponents) = sieve(p, &base, half);
        Sieve {
            p,
            bound,
            half,
            base,
            relations,
            exponents,
        }
    }
}

/// A relation: `(H + c1)(H + c2)` factors over the base, its exponents at `start..start + len` of
/// the list of all.
#[derive(Clone, Copy)]
struct Relation {
    c1: i32,
    c2: i32,
    start: u32,
    len: u32,
}

/// A sieved prime: where its table of first positions starts, and `c1 mod l` at the next row.
#[derive(Clone, Copy)]
struct Sieved {
    l: u32,
    log: u8,
    y: u32,
    at: u32,
}

/// `ceil(sqrt(p))` for `p >= 2`: a double's root, a step of Newton's, and the last correction.
fn root_up(p: u128) -> u128 {
    if p >> 64 == 0 {
        let r = u128::from(isqrt(p as u64));
        return if r * r == p { r } else { r + 1 };
    }
    // Within `2^12` of the root, then within one.
    let mut r = (p as f64).sqrt() as u128;
    r = (r + p / r) / 2;
    while r.checked_mul(r).is_none_or(|s| s > p) {
        r -= 1;
    }
    while (r + 1).checked_mul(r + 1).is_some_and(|s| s <= p) {
        r += 1;
    }
    if r * r == p { r } else { r + 1 }
}

/// The relations of the linear sieve modulo `p` over `c1 <= c2` in `[-half, half]`, with their
/// exponents.
fn sieve(p: u128, base: &Base, half: i64) -> (Vec<Relation>, Vec<(u32, u32)>) {
    let h = root_up(p);
    // `h <= 2^64`: `h^2 - p < 2h + 1`.
    let j = h.wrapping_mul(h).wrapping_sub(p) as i128;
    let h = h as i128;
    // For each prime, the first position `c2 - c1` of each row by `c1 mod l`: `(h + c1)(h + c2) =
    // p (mod l)`, so `c2 = p / (h + c1) - h`, none when `l` divides `h + c1`.
    let mut primes: Vec<Sieved> = Vec::with_capacity(base.primes.len());
    let mut table: Vec<u16> = Vec::new();
    let mut inv: Vec<u32> = Vec::new();
    for &l in &base.primes {
        if l < UNSIEVED {
            continue;
        }
        let pm = (p % u128::from(l)) as u32;
        let hm = (h % i128::from(l)) as u32;
        // `x mod l` for `x < 2^32`, by `m = ceil(2^32 / l)`: the quotient is right or one too
        // large.
        let m = (1u64 << 32).div_ceil(u64::from(l));
        let red = |x: u32| -> u32 {
            let qh = ((u64::from(x) * m) >> 32) as u32;
            let r = x.wrapping_sub(qh.wrapping_mul(l));
            if (r as i32) < 0 { r.wrapping_add(l) } else { r }
        };
        // The inverses modulo `l`: `l = (l / x) x + l mod x` gives `1/x = -(l / x) / (l mod x)`.
        inv.clear();
        inv.resize(l as usize, 0);
        inv[1] = 1;
        for x in 2..l {
            let qx = l / x;
            let rx = l - qx * x;
            inv[x as usize] = l - red(qx * inv[rx as usize]);
        }
        primes.push(Sieved {
            l,
            // `round(log2 l) = floor(log2(2 l^2) / 2)`.
            log: ((63 - (2 * u64::from(l) * u64::from(l)).leading_zeros()) / 2) as u8,
            y: (-half).rem_euclid(i64::from(l)) as u32,
            at: table.len() as u32,
        });
        let mut s = hm;
        for y in 0..l {
            if s == 0 {
                table.push(u16::MAX);
            } else {
                let r = red(pm * inv[s as usize]);
                // `r - hm - y mod l`.
                let mut t = r + 2 * l - hm - y;
                if t >= l {
                    t -= l;
                }
                if t >= l {
                    t -= l;
                }
                table.push(t as u16);
            }
            s += 1;
            if s == l {
                s = 0;
            }
        }
    }
    let width = (2 * half + 1) as usize;
    let stride = width + 16;
    let mut s = vec![0u8; ROWS * stride];
    let mut relations = Vec::new();
    let mut exponents = Vec::new();
    let mut f = Vec::new();
    let mut first = -half;
    while first <= half {
        let rows = ((half - first + 1) as usize).min(ROWS);
        for r in 0..rows {
            // The row and sixteen bytes past it, which the scan reads by words.
            let len = width - (first + half) as usize - r;
            s[r * stride..r * stride + len + 16].fill(0);
        }
        for sp in primes.iter_mut() {
            let l = sp.l as usize;
            let tab = &table[sp.at as usize..sp.at as usize + l];
            let mut y = sp.y as usize;
            for r in 0..rows {
                let t = tab[y];
                y += 1;
                if y == l {
                    y = 0;
                }
                if t != u16::MAX {
                    let len = width - (first + half) as usize - r;
                    let row = &mut s[r * stride..r * stride + len];
                    let mut at = t as usize;
                    while at < len {
                        row[at] = row[at].wrapping_add(sp.log);
                        at += l;
                    }
                }
            }
            sp.y = y as u32;
        }
        for r in 0..rows {
            let c1 = first + r as i64;
            let len = (half - c1 + 1) as usize;
            let row = &s[r * stride..r * stride + len + 16];
            // `V(c2) = (j + c1 h) + c2 (h + c1)`, from `c2 = c1`.
            let step = h + i128::from(c1);
            let v0 = j + i128::from(c1) * h + i128::from(c1) * step;
            let mut at = 0usize;
            let mut va = v0;
            while at < len {
                // Sixteen positions at once, against the smallest size among them: a byte `x` is
                // at least the floor `t` when `x + 128 - t` has its top bit set, the sums of the
                // logarithms being below 128 (a carry would only make a false candidate).
                let end = (at + 16).min(len);
                let vb = va + (end - 1 - at) as i128 * step;
                let low = if (va < 0) != (vb < 0) {
                    0
                } else {
                    va.unsigned_abs().min(vb.unsigned_abs())
                };
                let floor = (128 - low.leading_zeros()).saturating_sub(SLACK).min(128);
                let k = u64::from(128 - floor) * 0x0101_0101_0101_0101;
                let w = |i: usize| {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&row[i..i + 8]);
                    u64::from_le_bytes(b).wrapping_add(k)
                };
                if (w(at) | w(at + 8)) & 0x8080_8080_8080_8080 != 0 {
                    for (i, &x) in row[at..end].iter().enumerate() {
                        if u32::from(x) < floor {
                            continue;
                        }
                        let v = (va + i as i128 * step).unsigned_abs();
                        if u32::from(x) + SLACK >= 128 - v.leading_zeros()
                            && base.factor_wide(v, &mut f)
                        {
                            relations.push(Relation {
                                c1: c1 as i32,
                                c2: (c1 + (at + i) as i64) as i32,
                                start: exponents.len() as u32,
                                len: f.len() as u32,
                            });
                            exponents.extend_from_slice(&f);
                        }
                    }
                }
                at = end;
                va += 16 * step;
            }
        }
        first += rows as i64;
    }
    (relations, exponents)
}

/// The logarithms modulo `q` of the base's primes, up to a common factor, as plain numbers, from
/// the relations: `None` for a prime the relations do not determine, and `None` in all when they
/// leave more than one free; with them the logarithms of the `H + c`, in the same scale, those of
/// the components of the graph whose root an odd cycle fixed.
#[allow(clippy::type_complexity)]
fn solve<F: Field>(
    fq: &F,
    n: usize,
    half: i64,
    relations: &[Relation],
    exponents: &[(u32, u32)],
) -> Option<(Vec<Option<u128>>, Vec<Option<u128>>)> {
    let nc = (2 * half + 1) as usize;
    let vertex = |c: i32| (i64::from(c) + half) as usize;
    let exps = |e: u32| {
        let r = &relations[e as usize];
        &exponents[r.start as usize..(r.start + r.len) as usize]
    };
    // The graph of the `c`: for each vertex its edges, the relations, with the other end.
    let mut start = vec![0u32; nc + 1];
    for r in relations {
        start[vertex(r.c1) + 1] += 1;
        if r.c1 != r.c2 {
            start[vertex(r.c2) + 1] += 1;
        }
    }
    for v in 0..nc {
        start[v + 1] += start[v];
    }
    let mut fill = start.clone();
    let mut adjacent = vec![(0u32, 0u32); start[nc] as usize];
    for (e, r) in relations.iter().enumerate() {
        let (a, b) = (vertex(r.c1), vertex(r.c2));
        adjacent[fill[a] as usize] = (e as u32, b as u32);
        fill[a] += 1;
        if a != b {
            adjacent[fill[b] as usize] = (e as u32, a as u32);
            fill[b] += 1;
        }
    }
    // A spanning forest by breadth: `x_v = s_v x_root + F_v`, with `F_v = E - F_parent` along the
    // edge `E` to the parent, kept as the edge and the parent.
    const NONE: u32 = u32::MAX;
    let mut sign = vec![0i8; nc];
    let mut component = vec![0u32; nc];
    let mut parent = vec![(NONE, NONE); nc];
    let mut on_forest = vec![false; relations.len()];
    // The vertices in the order met, each component's root first.
    let mut queue: Vec<usize> = Vec::with_capacity(nc);
    let mut roots = 0u32;
    for root in 0..nc {
        if sign[root] != 0 || start[root] == start[root + 1] {
            continue;
        }
        sign[root] = 1;
        component[root] = roots;
        let mut next = queue.len();
        queue.push(root);
        while next < queue.len() {
            let v = queue[next];
            next += 1;
            for k in start[v]..start[v + 1] {
                let (e, w) = adjacent[k as usize];
                let w = w as usize;
                if sign[w] != 0 {
                    continue;
                }
                // `x_w = E - x_v`.
                sign[w] = -sign[v];
                component[w] = roots;
                parent[w] = (e, v as u32);
                on_forest[e as usize] = true;
                queue.push(w);
            }
        }
        roots += 1;
    }
    // `row += s F_v`: up the path to the root, the edges alternately added and taken.
    let add_path = |row: &mut [i32], mut v: usize, mut s: i32| {
        while parent[v].0 != NONE {
            let (e, up) = parent[v];
            for &(k, x) in exps(e) {
                row[k as usize] += s * x as i32;
            }
            s = -s;
            v = up as usize;
        }
    };
    // The primes in some relation.
    let mut present = vec![false; n];
    for &(k, _) in exponents {
        present[k as usize] = true;
    }
    let wanted = present.iter().filter(|&&x| x).count();
    // The edges off the forest, each a relation of the base, reduced as they come against the
    // rows so far, each zero before its pivot and 1 at it, kept in Montgomery's form: the row's
    // multiplier of each pivot is its entry at the pivot's column less the sum, not reduced, of
    // the products by the multipliers before, so that each entry takes one reduction.
    let mut pivots: Vec<(usize, Vec<F::V>)> = Vec::new();
    let mut is_pivot = vec![false; n];
    let mut root_of: Vec<Option<Vec<i32>>> = vec![None; roots as usize];
    let mut row = vec![0i32; n];
    let mut d = vec![fq.zero(); n];
    let mut acc = vec![fq.wide_zero(); n];
    for (e, r) in relations.iter().enumerate() {
        if on_forest[e] {
            continue;
        }
        if pivots.len() + 1 >= wanted {
            break;
        }
        let (a, b) = (vertex(r.c1), vertex(r.c2));
        // `(s_a + s_b) x_root + F_a + F_b = E`: the row `F_a + F_b - E`.
        row.iter_mut().for_each(|x| *x = 0);
        add_path(&mut row, a, 1);
        add_path(&mut row, b, 1);
        for &(k, x) in exps(e as u32) {
            row[k as usize] -= x as i32;
        }
        if sign[a] == sign[b] {
            // `2 x_root = -s_a row`: the first such in a component fixes `x_root`, and the
            // others, less it, are relations of the base.
            let s = i32::from(sign[a]);
            match &root_of[component[a] as usize] {
                None => {
                    root_of[component[a] as usize] = Some(row.iter().map(|&x| s * x).collect());
                    continue;
                }
                Some(fixed) => {
                    for (x, &y) in row.iter_mut().zip(fixed) {
                        *x = s * *x - y;
                    }
                }
            }
        }
        // The columns from the largest prime down: the large primes are in few relations, so
        // the pivots on them seldom touch a row, and the small ones, in nearly all, come last.
        for (y, &x) in d.iter_mut().zip(row.iter().rev()) {
            *y = fq.plain_small(i64::from(x));
        }
        acc.iter_mut().for_each(|x| *x = fq.wide_zero());
        for (pc, pr) in &pivots {
            let t = fq.sub(d[*pc], fq.wide_reduce(acc[*pc]));
            if t != fq.zero() {
                for (y, &x) in acc[*pc + 1..].iter_mut().zip(&pr[*pc + 1..]) {
                    fq.wide_add(y, t, x);
                }
            }
        }
        for ((y, x), &p) in d.iter_mut().zip(&acc).zip(&is_pivot) {
            *y = if p {
                fq.zero()
            } else {
                fq.sub(*y, fq.wide_reduce(*x))
            };
        }
        let Some(pc) = d.iter().position(|&x| x != fq.zero()) else {
            continue;
        };
        let s = fq.scale(fq.plain_inv(d[pc]));
        let mut pr = vec![fq.zero(); n];
        for (y, &x) in pr[pc..].iter_mut().zip(&d[pc..]) {
            *y = fq.mul(x, s);
        }
        is_pivot[pc] = true;
        pivots.push((pc, pr));
    }
    // One free column, the scale of the kernel: the others from the last row up, `x_pc = -sum
    // pr_k x_k`, `x` plain. The columns are the primes in reverse.
    let mut free = (0..n).filter(|&k| present[n - 1 - k] && !is_pivot[k]);
    let f0 = free.next()?;
    if free.next().is_some() {
        return None;
    }
    let mut x: Vec<Option<F::V>> = vec![None; n];
    x[f0] = Some(fq.plain_small(1));
    for (pc, pr) in pivots.iter().rev() {
        let mut sum = fq.wide_zero();
        let mut known = true;
        for k in pc + 1..n {
            if pr[k] != fq.zero() {
                match x[k] {
                    Some(y) => fq.wide_add(&mut sum, y, pr[k]),
                    None => {
                        known = false;
                        break;
                    }
                }
            }
        }
        if known {
            x[*pc] = Some(fq.sub(fq.zero(), fq.wide_reduce(sum)));
        }
    }
    // The base's logarithms in Montgomery's form, then those of the relations' numbers `V_e`, of
    // the roots fixed by an odd cycle, `2 x_root = -fixed`, and down the forest `x_w = L(V_e) -
    // x_v`.
    let logs: Vec<Option<F::V>> = x.into_iter().rev().collect();
    let mont: Vec<Option<F::V>> = logs
        .iter()
        .map(|y| y.map(|y| fq.of(fq.number(y))))
        .collect();
    let dot = |terms: &mut dyn Iterator<Item = (u32, i64)>| -> Option<F::V> {
        let (mut plus, mut minus) = (fq.wide_zero(), fq.wide_zero());
        for (k, e) in terms {
            let l = mont[k as usize]?;
            let a = fq.plain_small(e.abs());
            if e < 0 {
                fq.wide_add(&mut minus, a, l);
            } else {
                fq.wide_add(&mut plus, a, l);
            }
        }
        Some(fq.sub(fq.wide_reduce(plus), fq.wide_reduce(minus)))
    };
    let half_of = fq.plain_inv(fq.plain_small(2));
    let mut root_log: Vec<Option<F::V>> = vec![None; roots as usize];
    for (c, fixed) in root_of.iter().enumerate() {
        if let Some(fixed) = fixed {
            let t = dot(&mut fixed
                .iter()
                .enumerate()
                .filter(|(_, x)| **x != 0)
                .map(|(k, &x)| (k as u32, i64::from(x))));
            // `x_root = -t / 2`, plain.
            root_log[c] = t.map(|t| fq.mul(fq.of(fq.number(fq.sub(fq.zero(), t))), half_of));
        }
    }
    let mut clogs: Vec<Option<F::V>> = vec![None; nc];
    for &v in &queue {
        clogs[v] = match parent[v] {
            (NONE, _) => root_log[component[v] as usize],
            (e, up) => match clogs[up as usize] {
                Some(xu) => dot(&mut exps(e).iter().map(|&(k, x)| (k, i64::from(x))))
                    .map(|le| fq.sub(le, xu)),
                None => None,
            },
        };
    }
    Some((
        logs.into_iter().map(|y| y.map(|y| fq.number(y))).collect(),
        clogs.into_iter().map(|y| y.map(|y| fq.number(y))).collect(),
    ))
}

/// The rational reconstruction of `z` modulo `p`: `(u, v)` with `u = v z (mod p)` up to the sign
/// of `v`, `u < s` and `v <= p / s` for `s = ceil(sqrt(p))`, by Euclid's algorithm stopped at the
/// first remainder below `s`.
#[inline]
fn halves(z: u64, p: u64, s: u64) -> (u64, u64) {
    let (mut r0, mut r1) = (p, z);
    // The cofactors of `z`, their magnitudes: they alternate in sign.
    let (mut t0, mut t1) = (0u64, 1u64);
    while r1 >= s {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 + q * t1);
    }
    (r1, t1)
}

/// [`halves`] for a prime of two limbs, whose halves fit in one: the quotients of 1, 2 and 3, most
/// of them, by subtractions.
#[inline]
fn halves_wide(z: u128, p: u128, s: u128) -> (u128, u128) {
    let (mut r0, mut r1) = (p, z);
    let (mut t0, mut t1) = (0u128, 1u128);
    while r1 >= s {
        let mut r = r0 - r1;
        let q = if r < r1 {
            1
        } else {
            r -= r1;
            if r < r1 {
                2
            } else {
                r -= r1;
                if r < r1 {
                    3
                } else {
                    let q = r0 / r1;
                    r = r0 - q * r1;
                    q
                }
            }
        };
        (r0, r1) = (r1, r);
        (t0, t1) = (t1, t0 + q * t1);
    }
    (r1, t1)
}

/// `a b mod m` for `a, b < m`, by doublings: for the rare sums of the logarithms.
fn mul_mod(a: u128, mut b: u128, m: u128) -> u128 {
    let mut r = 0u128;
    let mut x = a;
    while b != 0 {
        if b & 1 == 1 {
            let (s, c) = r.overflowing_add(x);
            r = if c || s >= m { s.wrapping_sub(m) } else { s };
        }
        x = double(x, m);
        b >>= 1;
    }
    r
}

/// `a + b mod m` for `a, b < m`.
#[inline]
fn add_mod(a: u128, b: u128, m: u128) -> u128 {
    let (s, c) = a.overflowing_add(b);
    if c || s >= m { s.wrapping_sub(m) } else { s }
}

/// The logarithms of a base of small primes modulo a prime `p` of at most [`MOST_BITS`] bits,
/// modulo a prime `q` that divides `p - 1` once, up to a common factor; and from them the
/// logarithm of any element.
pub(in super::super) struct Index {
    /// The prime.
    p: u128,
    /// The order.
    q: u128,
    /// The base.
    base: Base,
    /// The logarithms of its primes, modulo `q`, those known.
    logs: Vec<Option<u128>>,
    /// `ceil(sqrt(p))`.
    s: u128,
    /// The multiplier of the elements whose halves are not smooth, a power of a prime of the
    /// base, and its logarithm.
    w: u128,
    lw: u128,
    /// `H = ceil(sqrt(p))`, the half width of the sieve's range, and the logarithms of the
    /// `H + c` known, at `c + half`: for the primes beyond the base, up to `second`, in the halves
    /// of a logarithm.
    h: u128,
    half: i64,
    clogs: Vec<Option<u128>>,
    second: u64,
}

impl Index {
    /// The logarithms modulo the prime `p`, of at most [`MOST_BITS`] bits, for the prime `q` that
    /// divides `p - 1` once; `None` when the relations did not determine them (the sieve and the
    /// base are widened seven times before that). The sieve's relations are taken from `kept` when
    /// it holds those modulo `p`, and left there for the next prime.
    pub(in super::super) fn new(p: u128, q: u128, kept: &mut Option<Sieve>) -> Option<Index> {
        let bits = 128 - p.leading_zeros();
        if !(3..=MOST_BITS).contains(&bits) || q < 3 {
            return None;
        }
        let (bound, half) = parameters(bits);
        // The relations of a sieve modulo `p` for another prime of `p - 1`, when there was one:
        // they do not depend on `q`.
        let mut sv = match kept.take() {
            Some(sv) if sv.p == p => sv,
            _ => Sieve::new(p, bound, half),
        };
        for _ in 0..8 {
            let n = sv.base.primes.len();
            let logs = if q >> 64 == 0 {
                solve(
                    &One::new(q as u64),
                    n,
                    sv.half,
                    &sv.relations,
                    &sv.exponents,
                )
            } else {
                solve(&Two::new(q), n, sv.half, &sv.relations, &sv.exponents)
            };
            if let Some((logs, clogs)) = logs {
                let base = sv.base.clone();
                let (half, count) = (sv.half, sv.relations.len());
                *kept = Some(sv);
                return Index::from_logs(p, q, base, logs, half, clogs, count);
            }
            // Too few relations: a wider range, and a larger base, which raises their yield.
            sv = Sieve::new(
                p,
                (sv.bound + sv.bound / 8).min(MOST_PRIME),
                sv.half + sv.half / 4,
            );
        }
        None
    }

    /// The index from the logarithms of the base and of the `H + c`, from that many relations.
    fn from_logs(
        p: u128,
        q: u128,
        base: Base,
        logs: Vec<Option<u128>>,
        half: i64,
        clogs: Vec<Option<u128>>,
        relations: usize,
    ) -> Option<Index> {
        // The multiplier: the first prime with a logarithm not 0, to a fixed power, not a multiple
        // of `q` (the constant is 5 139 199 82431689521877, and `w = 1` would try one element over
        // and over: modulo `6 82431689521877 + 1` most logarithms took 4 million tries, then rho).
        let k = (0..base.primes.len()).find(|&k| logs[k].is_some_and(|x| x != 0))?;
        let r = match 0x9E37_79B9_7F4A_7C15u128 % q {
            0 => 1,
            r => r,
        };
        let l = u128::from(base.primes[k]);
        let w = if p >> 64 == 0 {
            let mp = Odd::new(p as u64);
            u128::from(mp.from(mp.power(mp.to(l as u64), r as u64)))
        } else {
            let mp = Two::new(p);
            let (lr, mut x) = (mp.of(l), mp.one());
            for i in (0..128 - r.leading_zeros()).rev() {
                x = mp.ring.sqr(x);
                if (r >> i) & 1 == 1 {
                    x = mp.ring.mul(x, lr);
                }
            }
            mp.plain(x)
        };
        let lw = mul_mod(logs[k]?, r, q);
        // The primes beyond the base whose logarithms the range can give: a pair of the range in
        // `1/l` has `l` dividing its number, about `relations / l` of them smooth after it.
        let top = u64::from(*base.primes.last().unwrap_or(&2));
        let second = ((relations as u64) * SECOND).clamp(top, top * top.min(1 << 12));
        Some(Index {
            p,
            q,
            base,
            logs,
            s: root_up(p),
            w,
            lw,
            h: root_up(p),
            half,
            clogs,
            second,
        })
    }

    /// The logarithm of the prime `l` beyond the base, from a pair `(c1, c2)` of the sieve's range
    /// whose `L(H + c)` are known and whose number `V = (H + c1)(H + c2) - p` is `l` times a smooth
    /// one: `L(l) = L(H + c1) + L(H + c2) - L(V / l)`. For each `c1`, `l` divides `V` for `c2 =
    /// p / (H + c1) - H (mod l)`.
    fn beyond(&self, l: u64, f: &mut Vec<(u32, u32)>) -> Option<u128> {
        let lw = u128::from(l);
        let (pm, hm) = ((self.p % lw) as u64, (self.h % lw) as u64);
        let half = self.half;
        let q = self.q;
        for c1 in -half..=half {
            let Some(l1) = self.clogs[(c1 + half) as usize] else {
                continue;
            };
            let s = (u128::from(hm) + (i128::from(c1).rem_euclid(lw as i128)) as u128) % lw;
            let Some(inv) = inverse(s as u64, l) else {
                continue;
            };
            // `c2 = r - hm (mod l)`, the first at least `-half`.
            let r = (u128::from(pm) * u128::from(inv) % lw) as i128;
            let t = (r - i128::from(hm) + i128::from(half)).rem_euclid(lw as i128) as i64;
            let mut c2 = t - half;
            while c2 <= half {
                if let Some(l2) = self.clogs[(c2 + half) as usize] {
                    let a = self.h as i128 + i128::from(c1);
                    let b = self.h as i128 + i128::from(c2);
                    let v = (a * b - self.p as i128).unsigned_abs();
                    if v.is_multiple_of(lw) && self.base.factor_wide(v / lw, f) {
                        let mut lv = 0u128;
                        let mut known = true;
                        for &(k, e) in f.iter() {
                            match self.logs[k as usize] {
                                Some(x) => lv = add_mod(lv, mul_mod(x, u128::from(e), q), q),
                                None => known = false,
                            }
                        }
                        if known {
                            return Some(add_mod(add_mod(l1, l2, q), q - lv, q) % q);
                        }
                    }
                }
                c2 += l as i64;
            }
        }
        None
    }

    /// The logarithm of the half `x`: of its primes in the base, and of one prime beyond, up to
    /// `second`, when `large`; `Err` with that prime when it is there but not `large`.
    fn half_log(&self, x: u64, large: bool, f: &mut Vec<(u32, u32)>) -> Result<u128, u64> {
        if x == 0 {
            return Err(0);
        }
        let rest = self.base.factor_rest(x, f);
        let mut sum = 0u128;
        for &(k, e) in f.iter() {
            let Some(l) = self.logs[k as usize] else {
                return Err(0);
            };
            sum = add_mod(sum, mul_mod(l, u128::from(e), self.q), self.q);
        }
        if rest == 1 {
            return Ok(sum);
        }
        // No factor in the base: below the square of its bound, a prime.
        if rest > self.second {
            return Err(0);
        }
        if !large {
            return Err(rest);
        }
        let lr = self.beyond(rest, f).ok_or(0u64)?;
        Ok(add_mod(sum, lr, self.q))
    }

    /// The logarithm modulo `q` of the halves `u` and `v`, `L(u) - L(v)`, when both factor over
    /// the base but for at most one prime each up to `second`: those primes' logarithms, which take
    /// a search of the sieve's range, only once both halves are known to pass.
    #[inline]
    fn of_halves(&self, u: u64, v: u64, f: &mut Vec<(u32, u32)>) -> Option<u128> {
        let (lu, lv) = match (self.half_log(u, false, f), self.half_log(v, false, f)) {
            (Ok(a), Ok(b)) => (a, b),
            (Ok(a), Err(r)) if r != 0 => (a, self.half_log(v, true, f).ok()?),
            (Err(r), Ok(b)) if r != 0 => (self.half_log(u, true, f).ok()?, b),
            (Err(r), Err(t)) if r != 0 && t != 0 => (
                self.half_log(u, true, f).ok()?,
                self.half_log(v, true, f).ok()?,
            ),
            _ => return None,
        };
        Some(add_mod(lu, self.q - lv, self.q))
    }

    /// The logarithm modulo `q` of the plain `z`, `0 < z < p`, in the scale of the base's; `None`
    /// when no multiple tried was smooth.
    pub(in super::super) fn log(&self, z: u128) -> Option<u128> {
        let mut f = Vec::new();
        let mut lx = 0u128;
        // `z w^e`, until its halves are smooth: `L(z) + e L(w) = L(u) - L(v)`, the signs being of
        // no account modulo the odd `q`.
        if self.p >> 64 == 0 {
            let mp = Odd::new(self.p as u64);
            let (p, s, w) = (self.p as u64, self.s as u64, mp.to(self.w as u64));
            let mut x = mp.to(z as u64);
            for _ in 0..MOST_TRIES {
                let (u, v) = halves(mp.from(x), p, s);
                if let Some(l) = self.of_halves(u, v, &mut f) {
                    return Some(add_mod(l, self.q - lx, self.q) % self.q);
                }
                x = mp.times(x, w);
                lx = add_mod(lx, self.lw, self.q);
            }
        } else {
            let mp = Two::new(self.p);
            let w = mp.of(self.w);
            let mut x = mp.of(z);
            for _ in 0..MOST_TRIES {
                let (u, v) = halves_wide(mp.plain(x), self.p, self.s);
                // The halves are below `2^64`.
                if let Some(l) = self.of_halves(u as u64, v as u64, &mut f) {
                    return Some(add_mod(l, self.q - lx, self.q) % self.q);
                }
                x = mp.mul(x, w);
                lx = add_mod(lx, self.lw, self.q);
            }
        }
        None
    }

    /// The order `q`.
    pub(in super::super) fn order(&self) -> u128 {
        self.q
    }
}
