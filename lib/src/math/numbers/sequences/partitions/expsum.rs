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
    fn one() -> Sum {
        Sum {
            prefactor: 1,
            sp: 1,
            sq: 1,
            cos: [(0, 1); MOST_FACTORS],
            len: 0,
        }
    }

    /// Multiplies in `cos(π a / b)`, for `a` of any sign.
    fn push_cos(&mut self, a: i128, b: u64) {
        // `cos` is even, of period `2π`.
        let a = (a.unsigned_abs() % (2 * u128::from(b))) as u64;
        self.cos[self.len] = (a, b);
        self.len += 1;
    }
}

/// Arithmetic modulo `m`, which fits in a `u64`.
#[derive(Clone, Copy)]
struct Modulo {
    m: u64,
}

impl Modulo {
    #[inline]
    fn mul(self, a: u64, b: u64) -> u64 {
        if self.m <= 1 << 32 {
            // Both below `2^32`.
            (a * b) % self.m
        } else {
            ((u128::from(a) * u128::from(b)) % u128::from(self.m)) as u64
        }
    }

    #[inline]
    fn reduce(self, a: i128) -> u64 {
        a.rem_euclid(i128::from(self.m)) as u64
    }

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
        let (mut r0, mut r1) = (i128::from(self.m), i128::from(a % self.m));
        let (mut t0, mut t1) = (0i128, 1i128);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        debug_assert!(r0 == 1);
        self.reduce(t0)
    }
}

/// The Jacobi symbol `(a/n)` for an odd `n`.
fn jacobi(a: i128, n: u64) -> i64 {
    debug_assert!(n & 1 == 1);
    let mut a = a.rem_euclid(i128::from(n)) as u64;
    let mut n = n;
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

/// A square root of `a` modulo the odd prime `p`, by Tonelli and Shanks, or `None` when `a` is not
/// a square; `a` below `p`.
fn sqrt_mod_prime(a: u64, p: u64) -> Option<u64> {
    if a == 0 {
        return Some(0);
    }
    let md = Modulo { m: p };
    if jacobi(i128::from(a), p) != 1 {
        return None;
    }
    if p & 3 == 3 {
        return Some(md.pow(a, (p + 1) / 4));
    }
    // `p - 1 = q 2^s`, and a non-square `z`.
    let s = (p - 1).trailing_zeros();
    let q = (p - 1) >> s;
    let mut z = 2;
    while jacobi(i128::from(z), p) != -1 {
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
    Some(x)
}

/// A square root of `a`, prime to `p`, modulo `p^e` for an odd prime `p`, or `None`: the root modulo
/// `p` lifted by Newton's steps, `r - (r^2 - a) / (2r)`, each doubling the power of `p`.
fn sqrt_mod_power(a: u64, p: u64, pe: u64) -> Option<u64> {
    let mut r = sqrt_mod_prime(a % p, p)?;
    let mut m = p;
    while m < pe {
        m = m.saturating_mul(m).min(pe);
        let md = Modulo { m };
        let d = md.reduce(i128::from(md.mul(r, r)) - i128::from(a % m));
        let inv = md.inverse(md.mul(2, r));
        r = md.reduce(i128::from(r) - i128::from(md.mul(d, inv)));
    }
    Some(r)
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

/// Multiplies `A_k(n)` for the prime power `k = p^e` into `sum`, `n` below `k`.
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
    let r = 1 - 24 * i128::from(n);
    if p == 2 {
        let md = Modulo { m: 8 * k };
        let m = sqrt_mod_two_power(md.reduce(r), e + 3);
        let m = md.mul(m, md.inverse(3));
        // `(-1/m)`: -1 for `m = 3 mod 4`.
        if m & 3 == 3 {
            sum.prefactor = -sum.prefactor;
        }
        if e & 1 == 1 {
            sum.prefactor = -sum.prefactor;
        }
        sum.sp *= k;
        sum.push_cos(i128::from(k) - i128::from(m), 2 * k);
        return;
    }
    if p == 3 {
        let md = Modulo { m: 3 * k };
        let m = sqrt_mod_power(md.reduce(r), 3, 3 * k).expect("1 - 24n is a square modulo 3^e");
        let m = md.mul(m, md.inverse(8));
        sum.prefactor *= 2 * jacobi(i128::from(m), 3);
        if e & 1 == 0 {
            sum.prefactor = -sum.prefactor;
        }
        sum.sp *= k;
        sum.sq *= 3;
        sum.push_cos(3 * i128::from(k) - 8 * i128::from(m), 6 * k);
        return;
    }
    let md = Modulo { m: k };
    let m = md.reduce(r);
    if m.is_multiple_of(p) {
        if e == 1 {
            sum.prefactor *= jacobi(3, k);
            sum.sp *= k;
        } else {
            sum.prefactor = 0;
        }
        return;
    }
    let Some(m) = sqrt_mod_power(m, p, k) else {
        sum.prefactor = 0;
        return;
    };
    sum.prefactor *= 2 * jacobi(3, k);
    sum.sp *= k;
    let a = md.mul(m, md.inverse(24 % k));
    sum.push_cos(4 * i128::from(a), k);
}

/// `n1` with `(k2^2 d2 e) n1 = d2 e n + (k2^2 - 1) / d1 mod k1`.
fn solve(n: u64, k1: u64, k2: u64, d1: u64, d2: u64, e: u64) -> u64 {
    let md = Modulo { m: k1 };
    let k22 = u128::from(k2) * u128::from(k2);
    let t = ((k22 - 1) / u128::from(d1) % u128::from(k1)) as u64;
    let t = (u128::from(t) + u128::from(d2 * e) * u128::from(n)) % u128::from(k1);
    let u = (k22 % u128::from(k1)) as u64;
    let u = md.mul(md.mul(u, d2 * e % k1), 1);
    md.mul(t as u64, md.inverse(u))
}

/// `gcd(24, x)`.
fn gcd24(x: u64) -> u64 {
    let twos = 1 << x.trailing_zeros().min(3);
    if x.is_multiple_of(3) { 3 * twos } else { twos }
}

/// `A_k(n)` for `k >= 1` of the prime factors `factors` (increasing, with their exponents), and `n`.
pub(super) fn expsum(k: u64, n: u64, factors: &[(u64, u32)]) -> Sum {
    let mut sum = Sum::one();
    if k == 1 {
        return sum;
    }
    let (mut k, mut n) = (k, n % k);
    let last = factors.len() - 1;
    for &(p, e) in &factors[..last] {
        if sum.prefactor == 0 {
            return sum;
        }
        if p == 2 && e == 1 {
            // `k = 2 k2`: `32 n2 = 8n + 1 mod k2`, `n1 = n - (k2^2 - 1) / 8 mod 2`.
            let k2 = k / 2;
            let md = Modulo { m: k2 };
            let n2 = md.mul(md.inverse(32 % k2), md.reduce(8 * i128::from(n) + 1));
            let n1 = u64::from(matches!(k2 & 7, 3 | 5)) ^ (n & 1);
            prime_power(&mut sum, 2, n1, 2, 1);
            (k, n) = (k2, n2);
        } else if p == 2 && e == 2 {
            // `k = 4 k2`: `128 n2 = 8n + 5 mod k2`, `k2^2 n1 = n - 2 - (k2^2 - 1) / 8 mod 4`, and a
            // sign.
            let k2 = k / 4;
            let md = Modulo { m: k2 };
            let n2 = md.mul(md.inverse(128 % k2), md.reduce(8 * i128::from(n) + 5));
            let j = (k2 / 2) % 8;
            let n1 = (n + 8 - 2 - (j * (j + 1) / 2) % 4) % 4;
            prime_power(&mut sum, 4, n1, 2, 2);
            sum.prefactor = -sum.prefactor;
            (k, n) = (k2, n2);
        } else {
            let k1 = p.pow(e);
            let k2 = k / k1;
            let (d1, d2) = (gcd24(k1), gcd24(k2));
            let ee = 24 / (d1 * d2);
            let n1 = solve(n, k1, k2, d1, d2, ee);
            let n2 = solve(n, k2, k1, d2, d1, ee);
            prime_power(&mut sum, k1, n1, p, e);
            (k, n) = (k2, n2);
        }
    }
    if sum.prefactor != 0 {
        let (p, e) = factors[last];
        prime_power(&mut sum, k, n, p, e);
    }
    sum
}
