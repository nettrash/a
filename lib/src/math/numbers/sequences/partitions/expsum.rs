//! The exponential sums `A_k(n)` of Rademacher's series as products of cosines, after Whiteman
//! ("A sum connected with the series for the partition function", 1956), as Johansson's
//! implementation (2012) takes them.
//!
//! `A_k(n) = Σ_(h mod k, gcd(h, k) = 1) e^(πi (s(h, k) - 2nh/k))`, `s` the Dedekind sum, is
//! multiplicative in a twisted way: for `k = k1 k2` with `k1` and `k2` coprime,
//! `A_k(n) = A_(k1)(n1) A_(k2)(n2)` for residues `n1` and `n2` that congruences give, and the sum
//! for a prime power is one cosine times a square root and a sign, or zero:
//!
//! - `k = 2^λ`: `(-1)^λ (-1/m) √k cos(π (k - m) / (2k))`, `(3m)^2 = 1 - 24n mod 8k`,
//! - `k = 3^λ`: `2 (-1)^(λ+1) (m/3) √(k/3) cos(π (3k - 8m) / (6k))`, `(8m)^2 = 1 - 24n mod 3k`,
//! - `k = p^λ`, `p > 3`: `2 (3/k) √k cos(4πm / k)` with `(24m)^2 = 1 - 24n mod k`, zero when that has no
//!   root; and when `p` divides `1 - 24n`, `(3/k) √k` for `λ = 1` and zero for `λ > 1`,
//!
//! with Jacobi symbols in parentheses, and 1, `(-1)^n` and `2 cos(π {1, 7, 5} / 18)` (signed) for
//! `k = 1, 2, 3`. The values were checked against Selberg's formula
//! `A_k(n) = √(k/3) Σ_((3l^2 + l)/2 = -n mod k) (-1)^l cos(π (6l + 1) / (6k))` for every `k` below 600
//! and every `n` modulo `k`.
//!
//! The parts of `k` with the primes 2 and 3 are split off one after the other, each with its
//! residue `n1` from the congruences. The factors `p^λ` of the rest need no residue of their own:
//! for `c = k / p^λ`, the congruences give `1 - 24 n_p = (1 - 24n) c^-2 mod p^λ`, so the factor's
//! root is `r c^-1` for a root `r` of `1 - 24n` modulo `p^λ`, the same for every `k` that `p`
//! divides, and its angle `m = r (24c)^-1` (the sign of the root does not move the cosine). The
//! roots modulo the primes are found once each, by Tonelli and Shanks, the first time a term needs
//! them, and kept in the sieve of the smallest prime factors, in the slots of the primes
//! ([`Sieve`]); a prime power's is lifted from its prime's by Hensel's method. This direct form was
//! checked against Selberg's formula too, for every `k` below 600 and every `n` modulo `k`.
//!
//! Every modulus is below `2^35` (`k` is below `2^30` for any `n < 2^64`): products are taken in
//! `u64`s below `2^32` and in `u128`s above, and inverses by Euclid's algorithm in `i64`s.

/// The most prime factors of a `k` below `2^64`.
const MOST_FACTORS: usize = 16;

/// `A_k(n) = prefactor √(sp / sq) Π cos(π a_i / b_i)`.
#[derive(Clone, Copy)]
pub(super) struct Sum {
    /// The integer factor; 0 when the sum is zero.
    pub(super) prefactor: i64,
    /// The numerator under the square root: a divisor of `k`.
    pub(super) sp: u64,
    /// The denominator under the square root: 1 or 3.
    pub(super) sq: u64,
    /// The cosines' angles `π a / b`, `0 <= a < 2b`.
    pub(super) cos: [(u64, u64); MOST_FACTORS],
    /// Their number.
    pub(super) len: usize,
}

impl Sum {
    /// The empty product: 1.
    pub(super) fn one() -> Sum {
        Sum {
            prefactor: 1,
            sp: 1,
            sq: 1,
            cos: [(0, 1); MOST_FACTORS],
            len: 0,
        }
    }

    /// Multiplies in `cos(π a / b)`, for `0 <= a` of any size.
    fn push_cos(&mut self, a: u64, b: u64) {
        // `cos` is of period `2π`.
        self.cos[self.len] = (a % (2 * b), b);
        self.len += 1;
    }
}

/// Arithmetic modulo `m`, `1 <= m < 2^63`.
#[derive(Clone, Copy)]
struct Modulo {
    m: u64,
}

impl Modulo {
    /// `a b mod m` for `a` and `b` below `m`.
    #[inline]
    fn mul(self, a: u64, b: u64) -> u64 {
        if self.m <= 1 << 32 {
            // Both below `2^32`.
            (a * b) % self.m
        } else {
            ((u128::from(a) * u128::from(b)) % u128::from(self.m)) as u64
        }
    }

    /// `x mod m`.
    #[inline]
    fn of(self, x: u64) -> u64 {
        x % self.m
    }

    /// `a^e mod m` for `a` below `m`.
    fn pow(self, mut a: u64, mut e: u64) -> u64 {
        let mut r = 1 % self.m;
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(r, a);
            }
            a = self.mul(a, a);
            e >>= 1;
        }
        r
    }

    /// The inverse of `a`, prime to `m`, by Euclid's algorithm.
    fn inverse(self, a: u64) -> u64 {
        let (mut r0, mut r1) = (self.m as i64, (a % self.m) as i64);
        let (mut t0, mut t1) = (0i64, 1i64);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        debug_assert!(r0 == 1);
        if t0 < 0 {
            (t0 + self.m as i64) as u64
        } else {
            t0 as u64
        }
    }

    /// `1 - 24n mod m`.
    #[inline]
    fn one_minus_24n(self, n: u64) -> u64 {
        let t = self.mul(self.of(24), self.of(n));
        self.of(1 + self.m - t)
    }
}

/// The Jacobi symbol `(a/n)` for an odd `n` and `a` below `n`.
fn jacobi(mut a: u64, mut n: u64) -> i64 {
    debug_assert!(n & 1 == 1 && a < n);
    let mut t = 1;
    while a != 0 {
        let z = a.trailing_zeros();
        a >>= z;
        if z & 1 == 1 && matches!(n & 7, 3 | 5) {
            t = -t;
        }
        if a & n & 2 != 0 {
            t = -t;
        }
        (a, n) = (n % a, a);
    }
    if n == 1 { t } else { 0 }
}

/// `(3/p^e)` for a prime `p > 3`: `(3/p) = 1` for `p = ±1 mod 12` and -1 for `p = ±5`, by
/// reciprocity, to the power `e`.
fn jacobi_3(p: u64, e: u32) -> i64 {
    if matches!(p % 12, 1 | 11) || e & 1 == 0 {
        1
    } else {
        -1
    }
}

/// A square root of `a` modulo the odd prime `p`, by Tonelli and Shanks, `a` below `p` and a
/// square.
fn sqrt_mod_prime(a: u64, p: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    let md = Modulo { m: p };
    if p & 3 == 3 {
        return md.pow(a, (p + 1) / 4);
    }
    // `p - 1 = q 2^s`, and a non-square `z`.
    let s = (p - 1).trailing_zeros();
    let q = (p - 1) >> s;
    let mut z = 2;
    while jacobi(z, p) != -1 {
        z += 1;
    }
    let mut c = md.pow(z, q);
    let mut x = md.pow(a, q.div_ceil(2));
    let mut t = md.pow(a, q);
    let mut m = s;
    while t != 1 {
        let mut i = 0;
        let mut t2 = t;
        while t2 != 1 {
            t2 = md.mul(t2, t2);
            i += 1;
        }
        let b = md.pow(c, 1 << (m - i - 1));
        x = md.mul(x, b);
        c = md.mul(b, b);
        t = md.mul(t, c);
        m = i;
    }
    x
}

/// The root `r` modulo `p` of `a`, prime to `p`, lifted to one modulo `p^e` by Newton's steps
/// `r - (r^2 - a) / (2r)`, each doubling the power of `p`; `a` below `pe`.
fn lift_root(mut r: u64, a: u64, p: u64, pe: u64) -> u64 {
    let mut m = p;
    while m < pe {
        m = m.saturating_mul(m).min(pe);
        let md = Modulo { m };
        let (r2, am) = (md.mul(r, r), md.of(a));
        let d = md.of(r2 + m - am);
        let inv = md.inverse(md.mul(2, r));
        r = md.of(r + m - md.mul(d, inv));
    }
    r
}

/// A square root of `a`, prime to `p`, modulo `p^e` for an odd prime `p`, or `None`.
fn sqrt_mod_power(a: u64, p: u64, pe: u64) -> Option<u64> {
    let ap = a % p;
    if jacobi(ap, p) != 1 {
        return None;
    }
    Some(lift_root(sqrt_mod_prime(ap, p), a, p, pe))
}

/// A square root of `a`, 1 modulo 8, modulo `2^e`, `e >= 3`: the root modulo `2^i` made one modulo
/// `2^(i+1)` by adding `2^(i-1)` when it is not.
fn sqrt_mod_two_power(a: u64, e: u32) -> u64 {
    let mut x = 1u128;
    for i in 3..e {
        if (x * x).wrapping_sub(u128::from(a)) & ((1u128 << (i + 1)) - 1) != 0 {
            x += 1 << (i - 1);
        }
    }
    x as u64
}

/// The flag of a prime's slot in the sieve.
const PRIME: u32 = 1 << 31;

/// The value of a prime's slot whose root is not known yet.
const UNKNOWN: u32 = 0;

/// The value of a prime's slot when `1 - 24n` is not a square modulo it.
const NO_ROOT: u32 = 1;

/// The value of a prime's slot when it divides `1 - 24n`.
const DIVIDES: u32 = 2;

/// The smallest prime factor of every number up to the terms' bound, and in the slots of the
/// primes, flagged with [`PRIME`], what is known of the roots of `1 - 24n` modulo them: [`UNKNOWN`],
/// [`NO_ROOT`], [`DIVIDES`], or the root plus 3. The roots are below the primes, which are below
/// `2^30`: the series of `n = 2^64 - 1` takes about `7.8 10^8` terms.
pub(super) struct Sieve {
    /// The slots.
    slot: Vec<u32>,
    /// The partitions' `n`.
    n: u64,
}

/// What a prime's slot tells of `1 - 24n` modulo it.
enum Root {
    /// Not a square.
    None,
    /// Divisible.
    Divides,
    /// A root.
    Is(u64),
}

impl Sieve {
    /// The sieve up to `top`, for `n`.
    pub(super) fn new(top: u64, n: u64) -> Sieve {
        let len = usize::try_from(top + 1).expect("capacity overflow");
        let mut slot = vec![0u32; len];
        for i in 2..len {
            if slot[i] == 0 {
                slot[i] = PRIME | UNKNOWN;
                let p = i as u32;
                let mut j = 2 * i;
                while j < len {
                    if slot[j] == 0 {
                        slot[j] = p;
                    }
                    j += i;
                }
            }
        }
        Sieve { slot, n }
    }

    /// The prime factors of `k`, up to the sieve's top, with their exponents, increasing.
    pub(super) fn factors(&self, mut k: u64, out: &mut Vec<(u64, u32)>) {
        out.clear();
        while k > 1 {
            let v = self.slot[k as usize];
            let p = if v & PRIME != 0 { k } else { u64::from(v) };
            let mut e = 0;
            while k.is_multiple_of(p) {
                k /= p;
                e += 1;
            }
            out.push((p, e));
        }
    }

    /// The root of `1 - 24n` modulo the prime `p > 3` of the sieve, found the first time.
    fn root(&mut self, p: u64) -> Root {
        let slot = &mut self.slot[p as usize];
        debug_assert!(*slot & PRIME != 0);
        let v = *slot & !PRIME;
        let v = if v == UNKNOWN {
            let a = Modulo { m: p }.one_minus_24n(self.n);
            let v = if a == 0 {
                DIVIDES
            } else if jacobi(a, p) != 1 {
                NO_ROOT
            } else {
                sqrt_mod_prime(a, p) as u32 + 3
            };
            *slot = PRIME | v;
            v
        } else {
            v
        };
        match v {
            NO_ROOT => Root::None,
            DIVIDES => Root::Divides,
            r => Root::Is(u64::from(r - 3)),
        }
    }
}

/// Multiplies `A_k(n)` for the prime power `k = p^e`, `p` 2 or 3, into `sum`, `n` below `k`.
fn prime_power(sum: &mut Sum, k: u64, n: u64, p: u64, e: u32) {
    match k {
        1 => return,
        2 => {
            if n & 1 == 1 {
                sum.prefactor = -sum.prefactor;
            }
            return;
        }
        3 => {
            let (sign, a) = match n % 3 {
                0 => (2, 1),
                1 => (-2, 7),
                _ => (-2, 5),
            };
            sum.prefactor *= sign;
            sum.push_cos(a, 18);
            return;
        }
        _ => {}
    }
    if p == 2 {
        let md = Modulo { m: 8 * k };
        let m = sqrt_mod_two_power(md.one_minus_24n(n), e + 3);
        let m = md.mul(m, md.inverse(3));
        // `(-1/m)`: -1 for `m = 3 mod 4`.
        if m & 3 == 3 {
            sum.prefactor = -sum.prefactor;
        }
        if e & 1 == 1 {
            sum.prefactor = -sum.prefactor;
        }
        sum.sp *= k;
        // The angle `k - m` modulo `2 (2k)`: `m < 8k`, so `k - m + 8k > 0`.
        sum.push_cos(9 * k - m, 2 * k);
        return;
    }
    debug_assert!(p == 3);
    let md = Modulo { m: 3 * k };
    let m = sqrt_mod_power(md.one_minus_24n(n), 3, 3 * k).expect("1 - 24n is a square modulo 3^e");
    let m = md.mul(m, md.inverse(8));
    sum.prefactor *= 2 * jacobi(m % 3, 3);
    if e & 1 == 0 {
        sum.prefactor = -sum.prefactor;
    }
    sum.sp *= k;
    sum.sq *= 3;
    // `3k - 8m` modulo `2 (6k)`, `m < 3k`: `3k - 8m + 24k >= 0`.
    sum.push_cos(27 * k - 8 * m, 6 * k);
}

/// `n1` with `(k2^2 d2 e) n1 = d2 e n + (k2^2 - 1) / d1 mod k1`.
fn solve(n: u64, k1: u64, k2: u64, d1: u64, d2: u64, e: u64) -> u64 {
    let md = Modulo { m: k1 };
    // `k2 < 2^30`: its square in a `u64`.
    let k22 = k2 * k2;
    let t = md.of((k22 - 1) / d1);
    let t = md.of(t + md.mul(md.of(d2 * e), md.of(n)));
    let u = md.mul(md.of(k22), md.of(d2 * e));
    md.mul(t, md.inverse(u))
}

/// `gcd(24, x)`.
fn gcd24(x: u64) -> u64 {
    let twos = 1 << x.trailing_zeros().min(3);
    if x.is_multiple_of(3) { 3 * twos } else { twos }
}

/// `A_k(n)` for `k >= 1` of the prime factors `factors` (increasing, with their exponents), and the
/// sieve's `n`: the parts of 2 and 3 split off by the congruences, the others directly from the
/// roots of `1 - 24n` (see the module's documentation).
pub(super) fn expsum(k: u64, factors: &[(u64, u32)], sieve: &mut Sieve) -> Sum {
    let mut sum = Sum::one();
    if k == 1 {
        return sum;
    }
    let n = sieve.n;
    // The parts of 2 and 3, from the bottom, with the residue of what is left.
    let small = factors.iter().take_while(|&&(p, _)| p <= 3).count();
    let (mut kk, mut nn) = (k, n % k);
    for (i, &(p, e)) in factors[..small].iter().enumerate() {
        if sum.prefactor == 0 {
            return sum;
        }
        if i + 1 == factors.len() {
            prime_power(&mut sum, kk, nn, p, e);
        } else if p == 2 && e == 1 {
            // `k = 2 k2`: `32 n2 = 8n + 1 mod k2`, `n1 = n - (k2^2 - 1) / 8 mod 2`.
            let k2 = kk / 2;
            let md = Modulo { m: k2 };
            let n2 = md.mul(md.inverse(32 % k2), md.of(md.mul(8 % k2, md.of(nn)) + 1));
            let n1 = u64::from(matches!(k2 & 7, 3 | 5)) ^ (nn & 1);
            prime_power(&mut sum, 2, n1, 2, 1);
            (kk, nn) = (k2, n2);
        } else if p == 2 && e == 2 {
            // `k = 4 k2`: `128 n2 = 8n + 5 mod k2`, `k2^2 n1 = n - 2 - (k2^2 - 1) / 8 mod 4`, and a
            // sign.
            let k2 = kk / 4;
            let md = Modulo { m: k2 };
            let n2 = md.mul(md.inverse(128 % k2), md.of(md.mul(8 % k2, md.of(nn)) + 5));
            let j = (k2 / 2) % 8;
            let n1 = (nn + 8 - 2 - (j * (j + 1) / 2) % 4) % 4;
            prime_power(&mut sum, 4, n1, 2, 2);
            sum.prefactor = -sum.prefactor;
            (kk, nn) = (k2, n2);
        } else {
            let k1 = p.pow(e);
            let k2 = kk / k1;
            let (d1, d2) = (gcd24(k1), gcd24(k2));
            let ee = 24 / (d1 * d2);
            let n1 = solve(nn, k1, k2, d1, d2, ee);
            let n2 = solve(nn, k2, k1, d2, d1, ee);
            prime_power(&mut sum, k1, n1, p, e);
            (kk, nn) = (k2, n2);
        }
    }
    // The factors of primes above 3, directly.
    for &(p, e) in &factors[small..] {
        if sum.prefactor == 0 {
            return sum;
        }
        let k1 = p.pow(e);
        let root = match sieve.root(p) {
            Root::None => {
                sum.prefactor = 0;
                return sum;
            }
            Root::Divides => {
                if e == 1 {
                    sum.prefactor *= jacobi_3(p, 1);
                    sum.sp *= k1;
                } else {
                    sum.prefactor = 0;
                }
                continue;
            }
            Root::Is(r) if e == 1 => r,
            Root::Is(r) => {
                let md = Modulo { m: k1 };
                lift_root(r, md.one_minus_24n(n), p, k1)
            }
        };
        let md = Modulo { m: k1 };
        let c = md.of(k / k1);
        let m = md.mul(root, md.inverse(md.mul(md.of(24), c)));
        sum.prefactor *= 2 * jacobi_3(p, e);
        sum.sp *= k1;
        sum.push_cos(4 * m, k1);
    }
    sum
}
