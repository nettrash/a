//! Inversion modulo the engine's prime on its own limbs: Lehmer's extended Euclidean algorithm in
//! Knuth's form (TAOCP vol. 2, §4.5.2, Algorithm L), with leading digits of 62 bits.
//!
//! Each round runs Euclid's steps on the leading 62 bits of the two remainders, as long as the
//! quotients that the two extreme values of the remainders allow agree (so that they are the
//! true quotients), and then applies the 2 × 2 matrix of those steps, whose entries stay below
//! 2³¹ or so, to the remainders and to their cofactors at once: about 31 bits a round, 13 rounds
//! for 381 bits. The cofactors of Euclid's remainders alternate in sign, so only their
//! magnitudes are kept, with the parity of the step count, and every update of a magnitude is a
//! sum. A round whose first quotient cannot be read off the leading digits takes one step of
//! Euclid on the whole values instead; the last 64 bits are done by Euclid's steps on words.
//!
//! The public field inverts through `Number`'s Lehmer, which is the same method; this one keeps
//! the values on the stack in fixed limbs and skips the conversions.

/// The number of significant bits of `x`.
#[inline(always)]
fn bits<const N: usize>(x: &[u64; N]) -> u32 {
    for i in (0..N).rev() {
        if x[i] != 0 {
            return 64 * i as u32 + 64 - x[i].leading_zeros();
        }
    }
    0
}

/// The 64 bits of `x` from bit `s` on.
#[inline(always)]
fn window<const N: usize>(x: &[u64; N], s: u32) -> u64 {
    let (w, b) = ((s / 64) as usize, s % 64);
    let lo = x[w] >> b;
    if b == 0 || w + 1 >= N {
        lo
    } else {
        lo | (x[w + 1] << (64 - b))
    }
}

/// `m·x − n·y`, for a difference known to lie in `[0, 2^(64N))`.
#[inline(always)]
fn lin_sub<const N: usize>(x: &[u64; N], m: u64, y: &[u64; N], n: u64) -> [u64; N] {
    let mut r = [0u64; N];
    let (mut cx, mut cy, mut borrow) = (0u64, 0u64, false);
    for i in 0..N {
        let px = u128::from(x[i]) * u128::from(m) + u128::from(cx);
        cx = (px >> 64) as u64;
        let py = u128::from(y[i]) * u128::from(n) + u128::from(cy);
        cy = (py >> 64) as u64;
        let (d, b1) = (px as u64).overflowing_sub(py as u64);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        r[i] = d;
        borrow = b1 | b2;
    }
    r
}

/// `m·x + n·y`, for a sum known to lie below `2^(64N)`.
#[inline(always)]
fn lin_add<const N: usize>(x: &[u64; N], m: u64, y: &[u64; N], n: u64) -> [u64; N] {
    let mut r = [0u64; N];
    let (mut cx, mut cy, mut carry) = (0u64, 0u64, false);
    for i in 0..N {
        let px = u128::from(x[i]) * u128::from(m) + u128::from(cx);
        cx = (px >> 64) as u64;
        let py = u128::from(y[i]) * u128::from(n) + u128::from(cy);
        cy = (py >> 64) as u64;
        let (s, c1) = (px as u64).overflowing_add(py as u64);
        let (s, c2) = s.overflowing_add(u64::from(carry));
        r[i] = s;
        carry = c1 | c2;
    }
    r
}

/// `a ≥ b`.
fn at_least<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
    for i in (0..N).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

/// `x·2ᵗ`, for a value known to fit.
fn shl<const N: usize>(x: &[u64; N], t: u32) -> [u64; N] {
    let (w, b) = ((t / 64) as usize, t % 64);
    let mut r = [0u64; N];
    for i in (w..N).rev() {
        let lo = x[i - w] << b;
        let hi = if b > 0 && i > w {
            x[i - w - 1] >> (64 - b)
        } else {
            0
        };
        r[i] = lo | hi;
    }
    r
}

/// `1/x mod p` for `0 < x < p` and a prime `p` below `2^(64N − 2)`.
pub(crate) fn invert<const N: usize>(x: &[u64; N], p: &[u64; N]) -> [u64; N] {
    // Euclid's remainders u > v from (p, x), with the magnitudes of their cofactors of x.
    let (mut u, mut v) = (*p, *x);
    let (mut su, mut sv) = ([0u64; N], [0u64; N]);
    sv[0] = 1;
    // Whether u's cofactor is negative; v's has the other sign.
    let mut negative = true;
    loop {
        let n = bits(&u);
        if n <= 64 {
            // Euclid's steps on words, the matrix's entries below u's first value.
            let (mut x, mut y) = (u[0], v[0]);
            let (mut a0, mut b0, mut a1, mut b1) = (1u64, 0u64, 0u64, 1u64);
            while y != 0 {
                let q = x / y;
                (x, y) = (y, x - q * y);
                let (na, nb) = (a0 + q * a1, b0 + q * b1);
                (a0, b0, a1, b1) = (a1, b1, na, nb);
                negative = !negative;
            }
            su = lin_add(&su, a0, &sv, b0);
            break;
        }
        let s = n - 62;
        let (mut uh, mut vh) = (window(&u, s) as i64, window(&v, s) as i64);
        // The steps' matrix (a b; c d): its rows alternate in sign pattern, (+ −) then (− +).
        let (mut a, mut b, mut c, mut d) = (1i64, 0i64, 0i64, 1i64);
        let mut steps = 0u32;
        loop {
            let (d1, d2) = (vh + c, vh + d);
            let (n1, n2) = (uh + a, uh + b);
            if d1 <= 0 || d2 <= 0 || n1 < 0 || n2 < 0 {
                break;
            }
            let q = (n1 as u64) / (d1 as u64);
            // The same quotient for the other extreme: q·d₂ ≤ n₂ < (q + 1)·d₂.
            let qd = u128::from(q) * (d2 as u128);
            if qd > n2 as u128 || (n2 as u128) - qd >= d2 as u128 {
                break;
            }
            let q = q as i64;
            (a, c) = (c, a - q * c);
            (b, d) = (d, b - q * d);
            (uh, vh) = (vh, uh - q * vh);
            steps += 1;
        }
        if steps == 0 {
            // One step of Euclid on the whole values: u mod v by shifted subtractions, its
            // quotient's multiples of sv added to su.
            let bv = bits(&v);
            let (mut r, mut s) = (u, su);
            while at_least(&r, &v) {
                let mut t = bits(&r) - bv;
                let mut vt = shl(&v, t);
                if !at_least(&r, &vt) {
                    t -= 1;
                    vt = shl(&v, t);
                }
                r = lin_sub(&r, 1, &vt, 1);
                s = lin_add(&s, 1, &shl(&sv, t), 1);
            }
            (u, v, su, sv) = (v, r, sv, s);
            negative = !negative;
        } else {
            let nu = if b <= 0 {
                lin_sub(&u, a as u64, &v, b.unsigned_abs())
            } else {
                lin_sub(&v, b as u64, &u, a.unsigned_abs())
            };
            let nv = if d <= 0 {
                lin_sub(&u, c as u64, &v, d.unsigned_abs())
            } else {
                lin_sub(&v, d as u64, &u, c.unsigned_abs())
            };
            let nsu = lin_add(&su, a.unsigned_abs(), &sv, b.unsigned_abs());
            let nsv = lin_add(&su, c.unsigned_abs(), &sv, d.unsigned_abs());
            (u, v, su, sv) = (nu, nv, nsu, nsv);
            if steps & 1 == 1 {
                negative = !negative;
            }
        }
    }
    // u = 1 = ±su·x.
    if negative { lin_sub(p, 1, &su, 1) } else { su }
}
