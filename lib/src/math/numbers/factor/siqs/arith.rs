//! Arithmetic on words: residues modulo the primes of the factor base (below `2^32`), their
//! inverses and square roots, inverses modulo `2^64`, exact division of a number of limbs by a word,
//! and the splitting of a cofactor of two large primes (Montgomery's arithmetic on one limb, a strong
//! probable-prime test to base 2, and Pollard's rho with Brent's cycle finding).

/// `a · b mod p`.
#[inline(always)]
pub(super) fn mul_mod(a: u32, b: u32, p: u32) -> u32 {
    ((u64::from(a) * u64::from(b)) % u64::from(p)) as u32
}

/// `a^e mod p`, for `p > 1`.
pub(super) fn pow_mod(a: u32, mut e: u32, p: u32) -> u32 {
    let mut base = a % p;
    let mut r = 1 % p;
    while e != 0 {
        if e & 1 == 1 {
            r = mul_mod(r, base, p);
        }
        base = mul_mod(base, base, p);
        e >>= 1;
    }
    r
}

/// The inverse of `a` modulo `p`, for `a` prime to `p`, by Euclid's extended algorithm: in `[0, p)`.
pub(super) fn inverse(a: u32, p: u32) -> u32 {
    let (mut r0, mut r1) = (p, a % p);
    let (mut t0, mut t1) = (0i64, 1i64);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - i64::from(q) * t1);
    }
    debug_assert!(r0 == 1, "not invertible");
    if t0 < 0 {
        t0 += i64::from(p);
    }
    t0 as u32
}

/// A square root of the quadratic residue `a` modulo the odd prime `p`, by Tonelli and Shanks's
/// method (with the direct formulas for `p ≡ 3 (mod 4)` and Atkin's for `p ≡ 5 (mod 8)`): in
/// `[0, p)`.
pub(super) fn sqrt_mod(a: u32, p: u32) -> u32 {
    let a = a % p;
    if a == 0 {
        return 0;
    }
    if p & 3 == 3 {
        return pow_mod(a, (p + 1) / 4, p);
    }
    if p & 7 == 5 {
        let a2 = mul_mod(a, 2, p);
        let v = pow_mod(a2, (p - 5) / 8, p);
        let i = mul_mod(a2, mul_mod(v, v, p), p);
        return mul_mod(mul_mod(a, v, p), (i + p - 1) % p, p);
    }
    // p - 1 = q · 2^s with s >= 3.
    let s = (p - 1).trailing_zeros();
    let q = (p - 1) >> s;
    let mut z = 2;
    while pow_mod(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
    }
    let mut m = s;
    let mut c = pow_mod(z, q, p);
    let mut t = pow_mod(a, q, p);
    let mut r = pow_mod(a, q.div_ceil(2), p);
    while t != 1 {
        let mut i = 0;
        let mut t2 = t;
        while t2 != 1 {
            t2 = mul_mod(t2, t2, p);
            i += 1;
        }
        let mut b = c;
        for _ in 0..m - i - 1 {
            b = mul_mod(b, b, p);
        }
        m = i;
        c = mul_mod(b, b, p);
        t = mul_mod(t, c, p);
        r = mul_mod(r, b, p);
    }
    r
}

/// The inverse of the odd `p` modulo `2^64`, by Newton's steps: each doubles the bits that are
/// right, from the three of `p` itself.
#[inline]
pub(super) fn inverse_64(p: u64) -> u64 {
    debug_assert!(p & 1 == 1);
    let mut x = p;
    for _ in 0..5 {
        x = x.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(x)));
    }
    x
}

/// Divides the limbs of `x`, least significant first, by the odd word `p`, which divides them
/// exactly, by Hensel's (Jebelean's) method: each limb of the quotient is the limb left times the
/// inverse `pinv` of `p` modulo `2^64`.
#[inline]
pub(super) fn div_exact_1(x: &mut [u64], p: u64, pinv: u64) {
    let mut c = 0u64;
    for l in x.iter_mut() {
        let (s, b) = l.overflowing_sub(c);
        let q = s.wrapping_mul(pinv);
        *l = q;
        c = ((u128::from(q) * u128::from(p)) >> 64) as u64 + u64::from(b);
    }
}

/// Whether the odd word `p` divides the limbs of `x`, by the same steps without the quotient: the
/// last carry is zero exactly when it does (as `x = q·p − c·2^(64·len)` with `0 <= c <= p`, and `c = p`
/// would make `x` negative).
#[inline]
pub(super) fn divides(x: &[u64], p: u64, pinv: u64) -> bool {
    let mut c = 0u64;
    for &l in x {
        let (s, b) = l.overflowing_sub(c);
        let q = s.wrapping_mul(pinv);
        c = ((u128::from(q) * u128::from(p)) >> 64) as u64 + u64::from(b);
    }
    c == 0
}

/// The remainder of the limbs of `x`, least significant first, by `p < 2^32`: two halves of a limb
/// at a time, from the top.
pub(super) fn rem_1(x: &[u64], p: u32) -> u32 {
    let p = u64::from(p);
    let mut r = 0u64;
    for &l in x.iter().rev() {
        r = ((r << 32) | (l >> 32)) % p;
        r = ((r << 32) | (l & 0xFFFF_FFFF)) % p;
    }
    r as u32
}

/// Montgomery's arithmetic modulo an odd word `n`: residues `x·2^64 mod n`.
#[derive(Clone, Copy)]
pub(super) struct Mont {
    /// The modulus.
    n: u64,
    /// `n^(−1) mod 2^64`.
    inv: u64,
    /// `2^64 mod n`, the residue of 1.
    one: u64,
}

impl Mont {
    /// The arithmetic modulo the odd `n > 1`.
    pub(super) fn new(n: u64) -> Mont {
        let inv = inverse_64(n);
        let one = ((1u128 << 64) % u128::from(n)) as u64;
        Mont { n, inv, one }
    }

    /// `t · 2^(−64) mod n` for `t < n · 2^64`: the high word of `t` less that of `m·n`, where
    /// `m = t·n^(−1) mod 2^64` makes the low words equal.
    #[inline(always)]
    fn redc(&self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.inv);
        let mn = ((u128::from(m) * u128::from(self.n)) >> 64) as u64;
        let hi = (t >> 64) as u64;
        let (r, borrow) = hi.overflowing_sub(mn);
        if borrow { r.wrapping_add(self.n) } else { r }
    }

    /// The product of two residues.
    #[inline(always)]
    pub(super) fn mul(&self, a: u64, b: u64) -> u64 {
        self.redc(u128::from(a) * u128::from(b))
    }

    /// The sum of two residues.
    #[inline(always)]
    pub(super) fn add(&self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_add(b);
        if c || s >= self.n {
            s.wrapping_sub(self.n)
        } else {
            s
        }
    }

    /// The residue of 1.
    pub(super) fn one(&self) -> u64 {
        self.one
    }

    /// The residue of `x < n`.
    pub(super) fn residue(&self, x: u64) -> u64 {
        ((u128::from(x) << 64) % u128::from(self.n)) as u64
    }
}

/// Whether the odd `n > 2` passes the strong probable-prime test to base 2. Every prime does; the
/// composite numbers that pass are rare, and for the cofactors of the sieve, which only need to be
/// told composite to be split, one that passes is merely a relation lost.
pub(super) fn sprp2(n: u64) -> bool {
    let f = Mont::new(n);
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    let one = f.one();
    let minus_one = n - one;
    let mut x = one;
    let bits = 64 - d.leading_zeros();
    for i in (0..bits).rev() {
        x = f.mul(x, x);
        if d >> i & 1 == 1 {
            x = f.add(x, x);
        }
    }
    if x == one || x == minus_one {
        return true;
    }
    for _ in 1..s {
        x = f.mul(x, x);
        if x == minus_one {
            return true;
        }
        if x == one {
            return false;
        }
    }
    false
}

/// The greatest common divisor of two words, by Stein's binary method.
pub(super) fn gcd_64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let z = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << z;
        }
    }
}

/// A factor of the odd composite `n` other than 1 and `n`, by Pollard's rho method (1975) with
/// Brent's cycle finding (1980) on `x² + c` in Montgomery's form, the differences multiplied
/// together for a gcd every 64 steps; `None` when the first few constants `c` find none within
/// `limit` steps each.
pub(super) fn rho_64(n: u64, limit: u32) -> Option<u64> {
    let f = Mont::new(n);
    for c in 1..=8u64 {
        let c = f.residue(c % n);
        let mut y = f.residue(2 % n);
        let mut r = 1u32;
        let mut q = f.one();
        let mut x;
        let mut ys;
        let mut steps = 0u32;
        let mut g;
        loop {
            x = y;
            for _ in 0..r {
                y = f.add(f.mul(y, y), c);
            }
            let mut k = 0u32;
            loop {
                ys = y;
                let m = 64.min(r - k);
                for _ in 0..m {
                    y = f.add(f.mul(y, y), c);
                    q = f.mul(q, x.abs_diff(y));
                }
                g = gcd_64(q, n);
                k += m;
                if k >= r || g != 1 {
                    break;
                }
            }
            steps += r;
            r = r.saturating_mul(2);
            if g != 1 || steps > limit {
                break;
            }
        }
        if g == n {
            // The product passed a multiple of n: step one difference at a time from the last
            // saved point.
            loop {
                ys = f.add(f.mul(ys, ys), c);
                g = gcd_64(x.abs_diff(ys), n);
                if g != 1 {
                    break;
                }
            }
        }
        if g != 1 && g != n {
            return Some(g);
        }
    }
    None
}
