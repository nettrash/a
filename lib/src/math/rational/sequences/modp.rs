//! Bernoulli numbers modulo a prime, by Harvey's method ("A multimodular algorithm for computing
//! Bernoulli numbers", 2010), for the low bits of a long numerator.
//!
//! For a prime `p >= 5`, an even `k` and `2^k != 1` modulo `p` (which excludes `p - 1` dividing
//! `k`), Voronoi's congruence with `c = 2` is
//! `(2^k - 1) B_k = k 2^(k-1) Σ_(j=1)^(p-1) j^(k-1) floor(2j / p)` modulo `p`, and `floor(2j / p)`
//! is the first bit of the binary expansion of `j / p`. Taking `j` and `p - j` together, whose bits
//! are complements and whose powers are opposite, the sum is `Σ_(j ∈ H) j^(k-1) (2 bit(j) - 1)` over
//! a half `H` of the units. The orbit of `x` under doubling, `x 2^b`, has the bits of the expansion
//! of `x / p` from the `b`-th on, and the powers `x^(k-1) c^b` with `c = 2^(k-1)`: so `H` is taken as
//! cosets `x_t <2>` of the order `m` of 2 (or half of each when `-1` is a power of 2, `m` even), each
//! coset's expansion computed 64 bits a step by a division by `p`, and the bits weighted by tables of
//! the sums of `c^j` over the bits of a byte, eight tables for the eight bytes of a step and the
//! powers `c^(64β)` of the steps. About a fifth of a cycle for each unit of `H`.

/// `a b mod p`, for `a` and `b` below `p < 2^32`.
#[inline(always)]
fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    a * b % p
}

/// `a^e mod p`.
fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1 % p;
    a %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}

/// The distinct prime factors of `n > 0`, by trial division by the odd primes `small` (which reach
/// `√n`) and 2.
fn prime_factors(mut n: u64, small: &[u64]) -> Vec<u64> {
    let mut f = Vec::new();
    if n.is_multiple_of(2) {
        f.push(2);
        n >>= n.trailing_zeros();
    }
    for &q in small {
        if q * q > n {
            break;
        }
        if n.is_multiple_of(q) {
            f.push(q);
            while n.is_multiple_of(q) {
                n /= q;
            }
        }
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// The primes up to which the bits of a coset are taken one at a time, below which the tables
/// would cost more than they save.
const DIRECT: u64 = 1 << 7;

/// The primes up to which the tables are of chunks of 4 bits rather than 8.
const NIBBLES: u64 = 1 << 18;

/// `B_k mod p` for an even `k >= 2` and a prime `5 <= p < 2^32`, or `None` when `2^k = 1` modulo `p`
/// (the method does not apply; this includes the primes of the denominator). `small` holds the odd
/// primes up to `√p` at least.
pub(super) fn bernoulli_mod_p(k: u64, p: u64, small: &[u64]) -> Option<u64> {
    debug_assert!((5..1 << 32).contains(&p) && k >= 2 && k.is_multiple_of(2));
    let factors = prime_factors(p - 1, small);
    // The order of 2.
    let mut m = p - 1;
    for &q in &factors {
        while m.is_multiple_of(q) && powmod(2, m / q, p) == 1 {
            m /= q;
        }
    }
    if k.is_multiple_of(m) {
        return None;
    }
    // Exponents of 2 are taken modulo its order.
    let e = (k - 1) % m;
    let c = powmod(2, e, p);
    // Bits taken from each coset, and the cosets.
    let (bits, cosets) = if m.is_multiple_of(2) {
        (m / 2, (p - 1) / m)
    } else {
        (m, (p - 1) / (2 * m))
    };
    // A generator of the cosets, `g^t` for `t < cosets`: `g^((p-1)/q) != 1` for every prime `q`
    // dividing their number.
    let (g, gk) = if cosets > 1 {
        let qs: Vec<u64> = factors
            .iter()
            .copied()
            .filter(|&q| cosets.is_multiple_of(q))
            .collect();
        let g = (3..p)
            .find(|&g| qs.iter().all(|&q| powmod(g, (p - 1) / q, p) != 1))
            .expect("a prime has a primitive root");
        (g, powmod(g, (k - 1) % (p - 1), p))
    } else {
        (1, 1)
    };
    // S = Σ_t x_t^(k-1) s_t and X = Σ_t x_t^(k-1), for s_t = Σ_(b < bits) bit_b(x_t / p) c^b.
    let (mut sum, mut weights) = (0u64, 0u64);
    let (mut x, mut xk) = (1u64, 1u64);
    if p < DIRECT {
        for _ in 0..cosets {
            // Σ_b bit_b(x / p) c^b, one bit at a time.
            let (mut y, mut w, mut s) = (x, 1u64, 0u64);
            for _ in 0..bits {
                y *= 2;
                if y >= p {
                    y -= p;
                    s += w;
                }
                w = mulmod(w, c, p);
            }
            sum = (sum + mulmod(xk, s % p, p)) % p;
            weights = (weights + xk) % p;
            x = mulmod(x, g, p);
            xk = mulmod(xk, gk, p);
        }
    } else if p < NIBBLES {
        (sum, weights) = by_tables::<4>(TableWalk {
            p,
            c,
            e,
            m,
            g,
            gk,
            bits,
            cosets,
        });
    } else {
        (sum, weights) = by_tables::<8>(TableWalk {
            p,
            c,
            e,
            m,
            g,
            gk,
            bits,
            cosets,
        });
    }
    // A = 2S - G X with G = Σ_(b < bits) c^b = (c^bits - 1) / (c - 1), and
    // B_k = k c A / (2^k - 1): one inverse.
    let k1 = (powmod(2, k % m, p) + p - 1) % p;
    let ck = mulmod(k % p, c, p);
    let (num, den) = if c == 1 {
        ((2 * sum + p - mulmod(bits % p, weights, p)) % p, k1)
    } else {
        let g = (powmod(c, bits % m, p) + p - 1) % p;
        let s2 = mulmod(2 * sum % p, c - 1, p);
        ((s2 + p - mulmod(g, weights, p)) % p, mulmod(c - 1, k1, p))
    };
    Some(mulmod(mulmod(ck, num, p), inverse(den, p), p))
}

/// `a^-1 mod p` for `a` prime to `p`, by Euclid's algorithm.
fn inverse(a: u64, p: u64) -> u64 {
    let (mut r0, mut r1) = (p as i64, a as i64);
    let (mut t0, mut t1) = (0i64, 1i64);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    debug_assert_eq!(r0, 1);
    t0.rem_euclid(p as i64) as u64
}

/// What the walk over the cosets by tables needs.
struct TableWalk {
    p: u64,
    c: u64,
    e: u64,
    m: u64,
    g: u64,
    gk: u64,
    bits: u64,
    cosets: u64,
}

/// `(S, X)` of [`bernoulli_mod_p`] by tables of chunks of `W` bits.
fn by_tables<const W: usize>(walk: TableWalk) -> (u64, u64) {
    let TableWalk {
        p,
        c,
        e,
        m,
        g,
        gk,
        bits,
        cosets,
    } = walk;
    let (mut sum, mut weights) = (0u64, 0u64);
    let (mut x, mut xk) = (1u64, 1u64);
    let tables = Tables::<W>::new(c, p);
    let steps = bits.div_ceil(64) as usize;
    // The steps but the last in four lanes of `per`, each lane's chain of remainders apart, so
    // that the four run at once; the weights past the steps are zero.
    let per = (steps - 1).div_ceil(LANES);
    let mut powers = powers_of(powmod(2, 64 * e % m, p), steps, p);
    let last_power = powers[steps - 1];
    powers.truncate(steps - 1);
    powers.resize(LANES * per, 0);
    let jump = powmod(2, (64 * per as u64) % m, p);
    let to_last = powmod(2, (64 * (steps as u64 - 1)) % m, p);
    let div = Divisor::new(p);
    let last = bits % 64;
    let keep = if last == 0 {
        !0u64
    } else {
        !0u64 << (64 - last)
    };
    for _ in 0..cosets {
        let mut y = [0u64; LANES];
        let mut start = x;
        for lane in y.iter_mut() {
            *lane = start;
            start = mulmod(start, jump, p);
        }
        let mut s = [0u128; LANES];
        for i in 0..per {
            for lane in 0..LANES {
                let q = div.step(&mut y[lane]);
                s[lane] += u128::from(tables.sum(q)) * u128::from(powers[lane * per + i]);
            }
        }
        let mut z = mulmod(x, to_last, p);
        let q = div.step(&mut z) & keep;
        let mut t = u128::from(tables.sum(q)) * u128::from(last_power);
        for lane in s {
            t += lane % u128::from(p);
        }
        let s = (t % u128::from(p)) as u64;
        sum = (sum + mulmod(xk, s, p)) % p;
        weights = (weights + xk) % p;
        x = mulmod(x, g, p);
        xk = mulmod(xk, gk, p);
    }
    (sum, weights)
}

/// The chains of remainders run at once.
const LANES: usize = 4;

/// A prime below `2^32` prepared for the steps of the binary expansion of `y / p`, by Granlund and
/// Montgomery's multiplication by `m = ceil(2^128 / p)`: for `y < p`, `floor(y 2^64 / p)` is
/// `floor(y m / 2^64)` exactly, as `y 2^64 < 2^96` and `p <= 2^32`.
struct Divisor {
    p: u64,
    hi: u64,
    lo: u64,
}

impl Divisor {
    fn new(p: u64) -> Divisor {
        let m = u128::MAX / u128::from(p) + 1;
        Divisor {
            p,
            hi: (m >> 64) as u64,
            lo: m as u64,
        }
    }

    /// The next 64 bits of the expansion of `y / p`, `floor(y 2^64 / p)`, and `y <- y 2^64 mod p`.
    #[inline(always)]
    fn step(&self, y: &mut u64) -> u64 {
        let q = (u128::from(*y) * u128::from(self.hi)
            + ((u128::from(*y) * u128::from(self.lo)) >> 64)) as u64;
        // y 2^64 - q p, below p: the low 64 bits of -q p.
        *y = q.wrapping_mul(self.p).wrapping_neg();
        q
    }
}

/// The sums of `c^(Wi + j)` over the bits `j` of a chunk of `W` bits (the most significant bit
/// first), for the chunk `i` of a step of 64 bits, the most significant first: `W = 8`, eight
/// tables of 256, or `W = 4`, sixteen of 16, which take an eighth of the time to make.
struct Tables<const W: usize> {
    t: Vec<u32>,
}

impl<const W: usize> Tables<W> {
    /// The entries of a table.
    const SIZE: usize = 1 << W;
    /// The tables.
    const COUNT: usize = 64 / W;

    fn new(c: u64, p: u64) -> Tables<W> {
        let mut t = vec![0u32; Self::SIZE * Self::COUNT];
        let mut base = 1u64;
        for i in 0..Self::COUNT {
            let table = &mut t[i * Self::SIZE..(i + 1) * Self::SIZE];
            // Two tables of 4 bits, the high and the low, then their sums when the chunk is a byte.
            let mut nibbles = [[0u64; 16]; 2];
            for nib in nibbles.iter_mut().take(W / 4) {
                let mut pw = [0u64; 4];
                for slot in pw.iter_mut() {
                    *slot = base;
                    base = mulmod(base, c, p);
                }
                for chunk in 1..16usize {
                    let low = chunk & (chunk - 1);
                    let j = 3 - chunk.trailing_zeros() as usize;
                    let mut s = nib[low] + pw[j];
                    if s >= p {
                        s -= p;
                    }
                    nib[chunk] = s;
                }
            }
            if W == 4 {
                for (slot, &v) in table.iter_mut().zip(&nibbles[0]) {
                    *slot = v as u32;
                }
            } else {
                for (chunk, slot) in table.iter_mut().enumerate() {
                    let mut s = nibbles[0][chunk >> 4] + nibbles[1][chunk & 15];
                    if s >= p {
                        s -= p;
                    }
                    *slot = s as u32;
                }
            }
        }
        Tables { t }
    }

    /// The weighted sum of the 64 bits of `q`, below `16p`.
    #[inline(always)]
    fn sum(&self, q: u64) -> u64 {
        let mut s = 0u64;
        for i in 0..Self::COUNT {
            let chunk = (q >> (64 - W * (i + 1))) as usize & (Self::SIZE - 1);
            s += u64::from(self.t[i * Self::SIZE + chunk]);
        }
        s
    }
}

/// `1, x, x^2, ..., x^(n-1)` modulo `p`, in four interleaved chains.
fn powers_of(x: u64, n: usize, p: u64) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    let mut w = 1u64;
    for _ in 0..n.min(4) {
        v.push(w);
        w = mulmod(w, x, p);
    }
    let x4 = w;
    for i in 4..n {
        let next = mulmod(v[i - 4], x4, p);
        v.push(next);
    }
    v
}
