//! Number theory in machine words, for the moduli and the numbers below `2^64`: residues in
//! Montgomery's form in a register ([`Odd`]), powers, greatest common divisors and inverses, and
//! the factorisation of a word ([`factor`]), all without allocating.

use super::super::pow::montgomery::Mont1;
use super::super::pow::window::Registers;
use super::super::prime::is_prime_u64;

/// Montgomery's arithmetic modulo an odd `m` in a register, `R = 2^64`, the residues below `m`.
#[derive(Clone, Copy)]
pub(in super::super) struct Odd {
    /// The reduction modulo `m`.
    ring: Mont1,
    /// `R mod m`, the residue of 1.
    one: u64,
    /// `R^2 mod m`, by which a plain number is brought in.
    r2: u64,
}

impl Odd {
    /// The arithmetic modulo the odd `m`.
    #[inline]
    pub(in super::super) fn new(m: u64) -> Odd {
        debug_assert!(m & 1 == 1);
        let one = ((1u128 << 64) % u128::from(m)) as u64;
        let r2 = ((u128::from(one) * u128::from(one)) % u128::from(m)) as u64;
        Odd {
            ring: Mont1::new(m),
            one,
            r2,
        }
    }

    /// The modulus.
    #[inline(always)]
    pub(in super::super) fn m(&self) -> u64 {
        self.ring.m
    }

    /// The residue of 1.
    #[inline(always)]
    pub(in super::super) fn unit(&self) -> u64 {
        self.one
    }

    /// The residue `x R mod m` of the plain `x`, any word.
    #[inline(always)]
    pub(in super::super) fn to(&self, x: u64) -> u64 {
        let x = if x >= self.ring.m { x % self.ring.m } else { x };
        self.ring.mul(x, self.r2)
    }

    /// The plain number of the residue `a`, below `m`.
    #[inline(always)]
    pub(in super::super) fn from(&self, a: u64) -> u64 {
        self.ring.plain(a)
    }

    /// The product of two residues.
    #[inline(always)]
    pub(in super::super) fn times(&self, a: u64, b: u64) -> u64 {
        self.ring.mul(a, b)
    }

    /// The square of a residue.
    #[inline(always)]
    pub(in super::super) fn square(&self, a: u64) -> u64 {
        self.ring.sqr(a)
    }

    /// `a + b` for residues.
    #[inline(always)]
    pub(in super::super) fn plus(&self, a: u64, b: u64) -> u64 {
        let m = self.ring.m;
        let (s, carry) = a.overflowing_add(b);
        if carry || s >= m {
            s.wrapping_sub(m)
        } else {
            s
        }
    }

    /// `a - b` for residues.
    #[inline(always)]
    pub(in super::super) fn minus(&self, a: u64, b: u64) -> u64 {
        let (d, borrow) = a.overflowing_sub(b);
        if borrow {
            d.wrapping_add(self.ring.m)
        } else {
            d
        }
    }

    /// The residue `a` to the power `e`, by a window of two bits from the top: 1 for `e = 0`.
    #[inline]
    pub(in super::super) fn power(&self, a: u64, e: u64) -> u64 {
        if e == 0 {
            return self.one;
        }
        let a2 = self.square(a);
        let a3 = self.times(a2, a);
        let table = [self.one, a, a2, a3];
        let bits = 64 - e.leading_zeros();
        // From the top pair of bits, which is not zero.
        let mut i = (bits + 1) & !1;
        i -= 2;
        let mut r = table[(e >> i) as usize & 3];
        while i != 0 {
            i -= 2;
            r = self.square(self.square(r));
            let w = (e >> i) as usize & 3;
            if w != 0 {
                r = self.times(r, table[w]);
            }
        }
        r
    }
}

/// `a b mod m`, by a division.
#[inline(always)]
pub(in super::super) fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(m)) as u64
}

/// The greatest common divisor, by the binary method; `gcd(0, b) = b`.
#[inline]
pub(in super::super) fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

/// The inverse of `a` modulo `m >= 1`, below `m`, or `None` when they have a common factor:
/// Euclid's algorithm with the cofactor of `a`, the quotients of 1, 2 and 3 (more than three
/// quarters of them) by subtractions and the others by a division. Modulo 1 it is 0.
pub(in super::super) fn inverse(a: u64, m: u64) -> Option<u64> {
    if m == 1 {
        return Some(0);
    }
    let (mut r0, mut r1) = (m, a % m);
    // The cofactors of `a`, with their signs apart: `t0` and `t1` alternate in sign.
    let (mut t0, mut t1) = (0u64, 1u64);
    let mut odd = false;
    while r1 != 0 {
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
        odd = !odd;
    }
    if r0 != 1 {
        return None;
    }
    // `t0` is the magnitude of the cofactor; it is negative after an even number of steps.
    Some(if odd { t0 } else { m - t0 })
}

/// The integer square root, rounded down.
#[inline]
pub(in super::super) fn isqrt(n: u64) -> u64 {
    let mut r = (n as f64).sqrt() as u64;
    while u128::from(r) * u128::from(r) > u128::from(n) {
        r -= 1;
    }
    while u128::from(r + 1) * u128::from(r + 1) <= u128::from(n) {
        r += 1;
    }
    r
}

// Factorisation of a word

/// The odd primes below this are tried by division first.
const TRIAL: usize = 1 << 9;

/// An odd prime with what tells divisibility by it: `x inv mod 2^64 <= max` exactly when it
/// divides `x`.
#[derive(Clone, Copy)]
struct Small {
    /// The prime.
    p: u64,
    /// Its inverse modulo `2^64`.
    inv: u64,
    /// `floor((2^64 - 1) / p)`.
    max: u64,
}

/// The number of odd primes below [`TRIAL`].
const SMALL_COUNT: usize = {
    let mut n = 0;
    let mut x = 3;
    while x < TRIAL {
        let mut d = 3;
        let mut prime = true;
        while d * d <= x {
            if x.is_multiple_of(d) {
                prime = false;
                break;
            }
            d += 2;
        }
        if prime {
            n += 1;
        }
        x += 2;
    }
    n
};

/// The odd primes below [`TRIAL`], computed at compile time.
static SMALL: [Small; SMALL_COUNT] = {
    let mut t = [Small {
        p: 0,
        inv: 0,
        max: 0,
    }; SMALL_COUNT];
    let mut n = 0;
    let mut x = 3u64;
    while x < TRIAL as u64 {
        let mut d = 3;
        let mut prime = true;
        while d * d <= x {
            if x.is_multiple_of(d) {
                prime = false;
                break;
            }
            d += 2;
        }
        if prime {
            // The inverse by Newton's steps, each doubling the bits that are right.
            let mut v = x.wrapping_mul(3) ^ 2;
            let mut k = 0;
            while k < 4 {
                v = v.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(v)));
                k += 1;
            }
            t[n] = Small {
                p: x,
                inv: v,
                max: u64::MAX / x,
            };
            n += 1;
        }
        x += 2;
    }
    t
};

/// The factorisation of a word: at most 15 distinct primes (the product of the first 16 passes
/// `2^64`), in increasing order, with their exponents.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct Factors {
    /// The primes, increasing; the first `len` are set.
    pub(in super::super) p: [u64; 15],
    /// Their exponents.
    pub(in super::super) e: [u32; 15],
    /// How many there are.
    pub(in super::super) len: usize,
}

impl Factors {
    /// No primes: the factorisation of 1.
    pub(in super::super) const fn none() -> Factors {
        Factors {
            p: [0; 15],
            e: [0; 15],
            len: 0,
        }
    }

    /// The primes with their exponents.
    #[inline]
    pub(in super::super) fn iter(&self) -> impl Iterator<Item = (u64, u32)> + '_ {
        self.p[..self.len]
            .iter()
            .copied()
            .zip(self.e[..self.len].iter().copied())
    }

    /// Multiplies in `p^e`, keeping the primes in order.
    fn push(&mut self, p: u64, e: u32) {
        let mut i = self.len;
        while i > 0 && self.p[i - 1] > p {
            i -= 1;
        }
        if i > 0 && self.p[i - 1] == p {
            self.e[i - 1] += e;
            return;
        }
        let mut j = self.len;
        while j > i {
            self.p[j] = self.p[j - 1];
            self.e[j] = self.e[j - 1];
            j -= 1;
        }
        self.p[i] = p;
        self.e[i] = e;
        self.len += 1;
    }
}

/// The factorisation of `n >= 1` into primes: trial division by the odd primes below 512, then the
/// Baillie–PSW test, exact below `2^64`, and on what is left the search for a factor of
/// [`Number::factor`](super::super::Number::factor) in one limb: Hart's one line method and the
/// elliptic curve method.
pub(in super::super) fn factor(mut n: u64) -> Factors {
    debug_assert!(n != 0);
    let mut f = Factors::none();
    let z = n.trailing_zeros();
    if z > 0 {
        f.push(2, z);
        n >>= z;
    }
    for s in SMALL.iter() {
        if s.p * s.p > n {
            break;
        }
        if n.wrapping_mul(s.inv) <= s.max {
            let mut e = 0;
            loop {
                n = n.wrapping_mul(s.inv);
                e += 1;
                if n.wrapping_mul(s.inv) > s.max {
                    break;
                }
            }
            f.push(s.p, e);
        }
    }
    if n == 1 {
        return f;
    }
    // No factor below `TRIAL`: a number below its square is prime.
    if n < (TRIAL * TRIAL) as u64 {
        f.push(n, 1);
        return f;
    }
    split(n, 1, &mut f);
    f
}

/// Whether `n`, odd, above the squares of the trial primes, is prime: below `2^32` by the strong
/// tests to the bases 2, 7 and 61, which no composite below 4,759,123,141 passes (Jaeschke 1993),
/// in a tenth of the time of the Baillie–PSW test beyond.
pub(in super::super) fn is_prime(n: u64) -> bool {
    if n >> 32 != 0 {
        return is_prime_u64(n);
    }
    let f = Odd::new(n);
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    let minus = f.minus(0, f.unit());
    [2u64, 7, 61].iter().all(|&b| {
        if b % n == 0 {
            return true;
        }
        let mut x = f.power(f.to(b), d);
        if x == f.unit() || x == minus {
            return true;
        }
        for _ in 1..s {
            x = f.square(x);
            if x == minus {
                return true;
            }
        }
        false
    })
}

/// Multiplies the factorisation of the odd `n`, without factors below [`TRIAL`], to the power `e`
/// into `f`.
fn split(n: u64, e: u32, f: &mut Factors) {
    if is_prime(n) {
        f.push(n, e);
        return;
    }
    // A square is split by its root, which the methods for a factor would find only slowly.
    let r = isqrt(n);
    if r * r == n {
        split(r, 2 * e, f);
        return;
    }
    // Hart's one line method and the elliptic curve method in registers, as `factor` does.
    let d = super::super::factor::driver::find_factor_u64(n);
    split(d, e, f);
    split(n / d, e, f);
}
