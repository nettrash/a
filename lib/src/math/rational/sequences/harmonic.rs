//! The harmonic numbers `H_n = 1 + 1/2 + ... + 1/n`.
//!
//! Up to `H_88`, the last whose parts are below `2^128`, from a table computed at compile time, and
//! below [`NEAR`] as `H_88` plus the rest of the terms ([`near`]).
//! Beyond, each `k = 2^j m`, `m` odd, is gathered with the others of its odd part: with
//! `T = floor(log2 n)` and `t_m = floor(log2(n / m))`,
//! `H_n = 2^-T Σ_(m odd) c_m / m`, `c_m = 2^(T+1) - 2^(T - t_m)`, a sum over half the terms whose
//! denominator `Q`, the product of the odd numbers up to `n`, has half the bits of `n!`. The sum
//! `P / Q` is taken by binary splitting, its leaves in registers, and its top levels modulo `2^b`
//! for the `b` bits of the result's numerator.
//!
//! The reduced fraction needs no greatest common divisor of `P` and `Q`. Its denominator divides
//! `2^T L` for `L` the least common multiple of the odd numbers up to `n`, and `A = H_n 2^T L` is an
//! integer of `b` bits at most, which is `P L Q^-1` modulo `2^b` (`Q` is odd): an inverse by
//! Newton's method and two products of `b` bits. For an odd prime `p` with `p^e <= n < p^(e+1)`, the
//! terms of `H_n` divisible by `p^-e` are `1/(j p^e)` for `j` up to `k = floor(n / p^e) < p`, so
//! `p^e` divides the denominator unless `p` divides the numerator of `H_k`; those primes, rare, are
//! found from the table or by `H_k` modulo `p`, and only they are taken out of `A` and `L`, by the
//! greatest common divisor of `A` with their product.

use super::super::Rational;
use super::limbs::{
    Limbs, add, bits, div_1, mul, mul_1, normalize, number_u128, of_number, product, rem_1, shl,
    to_number, truncate,
};
use crate::math::numbers::{ByteOrder, primes};

/// The harmonic numbers below `2^128` in both parts: `H_0` to `H_88`, as (numerator, denominator),
/// each from the last by Knuth's sum `a/b + 1/k = (a k' + b') / (b k')` with `g = gcd(b, k)`,
/// `k' = k / g` and `b' = b / g`, then reduced by the divisors of `g`, which keeps every
/// intermediate below `2^128`.
pub(super) const TABLE: [(u128, u128); 89] = {
    let mut t = [(0u128, 1u128); 89];
    let mut k = 1;
    while k < 89 {
        let (a, b) = t[k - 1];
        let g = gcd(b, k as u128);
        let kk = k as u128 / g;
        let num = a * kk + b / g;
        let den = b * kk;
        let g2 = gcd(num, g);
        t[k] = (num / g2, den / g2);
        k += 1;
    }
    t
};

/// The greatest common divisor of two `u128`s, by Euclid's algorithm, at compile time.
const fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// `H_n`, in the byte order `order`: those in words from the table inline, in the caller, with
/// nothing to save or allocate (`H_0` to `H_46`), and the rest by [`beyond_words`].
#[inline]
pub(super) fn harmonic(n: u64, order: ByteOrder) -> Rational {
    if let Some(&(a, b)) = usize::try_from(n).ok().and_then(|i| TABLE.get(i))
        && (a | b) >> 64 == 0
    {
        return Rational::from_word(false, a as u64, b as u64, order);
    }
    beyond_words(n, order)
}

/// `H_n` when it is not in words, out of line so that the lookups keep no registers for it.
#[inline(never)]
fn beyond_words(n: u64, order: ByteOrder) -> Rational {
    if n < TABLE.len() as u64 {
        let (a, b) = TABLE[n as usize];
        return Rational::from_reduced(number_u128(a, false, order), number_u128(b, false, order));
    }
    if n < NEAR {
        return near(n, order);
    }
    let (num, den) = long(n);
    Rational::from_reduced(to_number(&num, false, order), to_number(&den, false, order))
}

/// Below this index, a harmonic number past the table is `H_88` plus the terms from `1/89`, added
/// as rationals, rather than the binary splitting with its reduction, whose preparation (the
/// primes, `L`, the inverse) costs more for so few terms. Measured on an Apple M4 Max: `H_89` took
/// 805 cycles so against 2490 for the splitting, `H_100` 1470 against 2850, `H_120` 2770 against
/// 3150, and `H_130` 3820 against 2900.
const NEAR: u64 = 125;

/// `H_n` just past the table: `H_88` plus the sum of the terms from `1/89`, in words while it fits
/// and then by the sums of rationals.
fn near(n: u64, order: ByteOrder) -> Rational {
    let last = TABLE.len() as u64 - 1;
    let (a, b) = TABLE[last as usize];
    let mut h = Rational::from_reduced(number_u128(a, false, order), number_u128(b, false, order));
    let (mut p, mut q) = (0u128, 1u128);
    for k in last + 1..=n {
        // p/q + 1/k = (p k + q) / (q k), while it fits.
        let next = q
            .checked_mul(u128::from(k))
            .and_then(|qk| Some((p.checked_mul(u128::from(k))?.checked_add(q)?, qk)));
        match next {
            Some((np, nq)) => (p, q) = (np, nq),
            None => {
                h += Rational::new(number_u128(p, false, order), number_u128(q, false, order));
                (p, q) = (1, u128::from(k));
            }
        }
    }
    h + Rational::new(number_u128(p, false, order), number_u128(q, false, order))
}

/// The parts of `H_n` for `n` above the table, in lowest terms.
fn long(n: u64) -> (Limbs, Limbs) {
    // `H_n` has about `1.44 n` bits; from `2^62` its parts pass any memory.
    assert!(n < 1 << 62, "capacity overflow");
    let t = 63 - n.leading_zeros() as u64;
    let (l, exceptional) = odd_lcm(n);
    // `H_n < ln n + 1 < 2^6`.
    let b = bits(&l) + t + 7;
    let (p, q) = odd_sum(n, t, b);
    let y = inverse(&q, b);
    let mut a = mul(&p, &l);
    truncate(&mut a, b);
    let mut a = mul(&a, &y);
    truncate(&mut a, b);
    let mut den = l;
    if !exceptional.is_empty() {
        // g = gcd(A, E) for E the product of the powers of the primes that may divide A.
        let g = if let [e] = exceptional[..] {
            gcd_u64(rem_1(&a, e), e)
        } else {
            0
        };
        if g > 1 {
            div_1(&mut a, g);
            div_1(&mut den, g);
        } else if g == 0 {
            let o = ByteOrder::LittleEndian;
            let e = to_number(&product(&exceptional), false, o);
            let an = to_number(&a, false, o);
            let g = (&an % &e).gcd(&e);
            if g != 1u8 {
                a = of_number(&an.div_exact(&g));
                den = of_number(&to_number(&den, false, o).div_exact(&g));
            }
        }
    }
    (a, shl(&den, t))
}

/// `L`, the least common multiple of the odd numbers up to `n`, the product of the largest powers
/// `p^e <= n` of the odd primes, and the limbs whose product is that of the powers `p^e` of the
/// primes `p` that divide the numerator of `H_k`, `k = floor(n / p^e)` (none when empty).
fn odd_lcm(n: u64) -> (Limbs, Vec<u64>) {
    let mut powers: Vec<u64> = Vec::new();
    let mut exceptional: Vec<u64> = Vec::new();
    // Above `√n`, `p^2 > n`.
    let root = n.isqrt();
    let (mut packed, mut open) = (1u64, 1u64);
    for p in primes(3..=n) {
        let mut q = p;
        if p <= root {
            while q <= n / p {
                q *= p;
            }
        }
        match packed.checked_mul(q) {
            Some(x) => packed = x,
            None => {
                powers.push(packed);
                packed = q;
            }
        }
        if divides_numerator(p, n / q) {
            match open.checked_mul(q) {
                Some(x) => open = x,
                None => {
                    exceptional.push(open);
                    open = q;
                }
            }
        }
    }
    powers.push(packed);
    if open != 1 {
        exceptional.push(open);
    }
    (product(&powers), exceptional)
}

/// The greatest common divisor of two words, by Euclid's algorithm.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Whether the prime `p` divides the numerator of `H_k`, for `k < p`.
fn divides_numerator(p: u64, k: u64) -> bool {
    if k < 2 {
        return false;
    }
    if k < TABLE.len() as u64 {
        // The numerator modulo `p`, in words: `hi 2^64 + lo`.
        let a = TABLE[k as usize].0;
        let (hi, lo) = ((a >> 64) as u64, a as u64);
        let r = if p < 1 << 32 {
            let b = ((1u128 << 64) % u128::from(p)) as u64;
            ((hi % p) * b % p + lo % p) % p
        } else {
            (a % u128::from(p)) as u64
        };
        return r == 0;
    }
    // `H_k = a / b` modulo `p`, `b = k!` prime to `p`.
    let (mut a, mut b) = (1u64, 1u64);
    if p < 1 << 32 {
        for m in 2..=k {
            a = (a * m + b) % p;
            b = b * m % p;
        }
    } else {
        let pp = u128::from(p);
        for m in 2..=k {
            a = ((u128::from(a) * u128::from(m) + u128::from(b)) % pp) as u64;
            b = (u128::from(b) * u128::from(m) % pp) as u64;
        }
    }
    a == 0
}

/// The limbs up to which the terms are added one leaf at a time into the same pair, in place;
/// beyond, the pairs are merged by a balanced tree.
const SEQUENTIAL: usize = 16;

/// `(P mod 2^b, Q mod 2^b)` for `P / Q = Σ_(m odd, m <= n) c_m / m`, `Q` the product of the odd
/// numbers up to `n`: leaves of as many terms as fit in registers, added into a pair in place up to
/// [`SEQUENTIAL`] limbs, `P <- P q + p Q` and `Q <- Q q`; then the pairs merged as a binary counter
/// merges, two of the same rank at a time, `P = P_1 Q_2 + P_2 Q_1` and `Q = Q_1 Q_2`, cut to `b` bits
/// once they pass them.
fn odd_sum(n: u64, t: u64, b: u64) -> (Limbs, Limbs) {
    let top = 1u64 << (t + 1);
    let mut stack: Vec<(Limbs, Limbs, u32)> = Vec::new();
    let mut acc_p: Limbs = Vec::with_capacity(SEQUENTIAL + 4);
    let mut acc_q: Limbs = Vec::with_capacity(SEQUENTIAL + 4);
    acc_q.push(1);
    let (mut p, mut q) = (0u128, 1u64);
    let mut m = 1u64;
    // `t_m = floor(log2(n / m))` is `tm` for `m` up to `end = floor(n / 2^tm)`.
    let mut tm = t;
    let mut end = n >> t;
    loop {
        let more = m <= n;
        let c = if more {
            while m > end {
                tm -= 1;
                end = n >> tm;
            }
            top - (1u64 << (t - tm))
        } else {
            0
        };
        // `p/q + c/m = (p m + c q) / (q m)`, while it fits.
        let fits = if more {
            q.checked_mul(m).and_then(|qm| {
                let pm = p.checked_mul(u128::from(m))?;
                let cq = u128::from(c).checked_mul(u128::from(q))?;
                Some((pm.checked_add(cq)?, qm))
            })
        } else {
            None
        };
        if let Some((np, nq)) = fits {
            p = np;
            q = nq;
            m += 2;
            continue;
        }
        // The leaf into the pair: P <- P q + p Q, Q <- Q q.
        mul_1(&mut acc_p, q);
        addmul(&mut acc_p, &acc_q, p);
        mul_1(&mut acc_q, q);
        if acc_q.len() >= SEQUENTIAL || !more {
            let pair = (
                core::mem::replace(&mut acc_p, Vec::with_capacity(SEQUENTIAL + 4)),
                core::mem::replace(&mut acc_q, Vec::with_capacity(SEQUENTIAL + 4)),
            );
            acc_q.push(1);
            stack.push((pair.0, pair.1, 0));
            while stack.len() >= 2
                && (stack[stack.len() - 1].2 == stack[stack.len() - 2].2 || !more)
            {
                let (p2, q2, r) = stack.pop().expect("two pairs");
                let (p1, q1, _) = stack.pop().expect("two pairs");
                let mut pp = mul(&p1, &q2);
                add(&mut pp, &mul(&p2, &q1));
                let mut qq = mul(&q1, &q2);
                truncate(&mut pp, b);
                truncate(&mut qq, b);
                stack.push((pp, qq, r + 1));
            }
        }
        if !more {
            break;
        }
        p = u128::from(c);
        q = m;
        m += 2;
    }
    let (p, q, _) = stack.pop().expect("one pair");
    (p, q)
}

/// `x += y c` in place, for a `u128` multiplier.
fn addmul(x: &mut Limbs, y: &[u64], c: u128) {
    if c == 0 || y.is_empty() {
        return;
    }
    let need = y.len() + 2;
    if x.len() < need {
        x.resize(need, 0);
    }
    for (shift, part) in [(0usize, c as u64), (1, (c >> 64) as u64)] {
        if part == 0 {
            continue;
        }
        let mut carry = 0u64;
        for (i, &l) in y.iter().enumerate() {
            let t = u128::from(l) * u128::from(part) + u128::from(x[i + shift]) + u128::from(carry);
            x[i + shift] = t as u64;
            carry = (t >> 64) as u64;
        }
        let mut i = y.len() + shift;
        while carry != 0 {
            let (s, o) = x[i].overflowing_add(carry);
            x[i] = s;
            carry = u64::from(o);
            i += 1;
        }
    }
    normalize(x);
}

/// `q^-1 mod 2^b` for an odd `q`, by Newton's method: `y <- y (2 - q y)`, which doubles the
/// limbs that are right.
fn inverse(q: &[u64], b: u64) -> Limbs {
    let q0 = q[0];
    // Right modulo 2^3 (`q q = 1 mod 8`), then 6, 12, 24, 48, 96 bits.
    let mut y0 = q0;
    for _ in 0..5 {
        y0 = y0.wrapping_mul(2u64.wrapping_sub(q0.wrapping_mul(y0)));
    }
    let words = b.div_ceil(64) as usize;
    let mut steps = Vec::new();
    let mut k = words;
    while k > 1 {
        steps.push(k);
        k = k.div_ceil(2);
    }
    let mut y: Limbs = vec![y0];
    let mut h = 1;
    for &k in steps.iter().rev() {
        // `y < B^h` is right modulo `B^h`, `2h >= k`: `q y = 1 + B^h e` modulo `B^k`, and
        // `y (1 - B^h e) = y + B^h ((-y e) mod B^(k - h))` is right modulo `B^k`.
        let mut qk = q[..k.min(q.len())].to_vec();
        normalize(&mut qk);
        let mut t = mul(&qk, &y);
        truncate(&mut t, 64 * k as u64);
        let mut e = if t.len() > h {
            t[h..].to_vec()
        } else {
            Vec::new()
        };
        normalize(&mut e);
        let mut u = mul(&y, &e);
        truncate(&mut u, 64 * (k - h) as u64);
        y.resize(k, 0);
        if !u.is_empty() {
            // `-u mod B^(k - h)`: the complement of `u`, plus one.
            let mut carry = true;
            for i in 0..k - h {
                let v = !u.get(i).copied().unwrap_or(0);
                let (s, c) = v.overflowing_add(u64::from(carry));
                y[h + i] = s;
                carry = c;
            }
        }
        normalize(&mut y);
        h = k;
    }
    truncate(&mut y, b);
    y
}
