//! The Jacobi symbol of long numbers by Lehmer's method, the symbol carried through the quotients,
//! after Möller ("Efficient computation of the Jacobi symbol", 2008), whose algorithm GMP's
//! `mpn_jacobi_n` takes.
//!
//! Euclid's algorithm on the pair `(a, b)`, `b` odd, reduces one number by a multiple of the other
//! at each step. The symbol `(a / b)` is `(-1)^e (x / y)` for the current pair, `y` the one that
//! plays the denominator, which is always odd: reducing the numerator by multiples of the
//! denominator leaves the symbol as it is; reducing the denominator by an odd numerator first
//! swaps their parts by reciprocity, `(x / y) = (-1)^[x = y = 3 (mod 4)] (y / x)`; and reducing it
//! by an even numerator `n` takes, for each subtraction, `(n / d) = (-1)^((d-1)/2) s(d - n, d)
//! (n / d - n)`, `s(x, y)` the sign of reciprocity, with `d - n` the new denominator. All of that
//! depends only on the quotient and on both numbers modulo 4, which the state below keeps, so the
//! symbol follows the quotients that Lehmer's method finds from the top limbs alone, and the whole
//! numbers are needed only at the end.

/// The state: bit 0 the sign `e`, bits 1-2 `a mod 4`, bits 3-4 `b mod 4`, bit 5 set when `a` is
/// the denominator.
type State = u8;

/// The state for the numerator `a` and the odd denominator `b`.
#[inline(always)]
pub(in super::super) fn init(a: u64, b: u64) -> State {
    debug_assert!(b & 1 == 1);
    (((a & 3) << 1) | ((b & 3) << 3)) as u8
}

/// The state after the step that reduces `a` by `q` times `b` (`a_reduced`), or `b` by `q` times
/// `a`, `q` given modulo 4.
const fn step(s: State, a_reduced: bool, q: u8) -> State {
    let mut e = s & 1;
    let xa = (s >> 1) & 3;
    let xb = (s >> 3) & 3;
    let mut den_a = (s >> 5) & 1 == 1;
    let (mut r, t) = if a_reduced { (xa, xb) } else { (xb, xa) };
    let reduced_is_den = den_a == a_reduced;
    if !reduced_is_den {
        // The numerator less multiples of the denominator: the symbol does not change.
        r = (r + 4 * 4 - (q * t) % 4) % 4;
    } else if t & 1 == 1 {
        // The denominator less multiples of an odd numerator: reciprocity first.
        if r == 3 && t == 3 {
            e ^= 1;
        }
        den_a = !den_a;
        r = (r + 4 * 4 - (q * t) % 4) % 4;
    } else {
        // The denominator less an even numerator, `q` times, each with its sign.
        let mut k = 0;
        while k < q {
            let d = r;
            let next = (d + 4 - t) % 4;
            if d == 3 {
                e ^= 1;
            }
            if d == 3 && next == 3 {
                e ^= 1;
            }
            r = next;
            k += 1;
        }
    }
    let (xa, xb) = if a_reduced { (r, t) } else { (t, r) };
    e | (xa << 1) | (xb << 3) | ((den_a as u8) << 5)
}

/// [`step`] for every state, reduced number and quotient modulo 4.
const TABLE: [State; 512] = {
    let mut t = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        let s = (i >> 3) as u8;
        let a_reduced = (i >> 2) & 1 == 1;
        let q = (i & 3) as u8;
        t[i] = step(s, a_reduced, q);
        i += 1;
    }
    t
};

/// The state after reducing `a` (`a_reduced`) or `b` by `q` times the other.
#[inline(always)]
pub(in super::super) fn update(s: State, a_reduced: bool, q: u64) -> State {
    TABLE[(usize::from(s) << 3) | (usize::from(a_reduced) << 2) | (q & 3) as usize]
}

/// Whether `a` is the denominator.
#[inline(always)]
pub(in super::super) fn a_is_denominator(s: State) -> bool {
    s & 32 != 0
}

/// The sign `(-1)^e`.
#[inline(always)]
pub(in super::super) fn sign(s: State) -> i32 {
    1 - 2 * i32::from(s & 1)
}

// Lehmer's steps

/// A matrix of single limbs `(u00, u01; u10, u11)` of determinant 1: the pair before the steps is
/// the matrix times the pair after them.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct Matrix {
    pub(in super::super) u00: u64,
    pub(in super::super) u01: u64,
    pub(in super::super) u10: u64,
    pub(in super::super) u11: u64,
}

/// The bound below which [`hgcd2`] reduces no number on two limbs: `2^65`.
const LIMIT_2: u64 = 2;

/// The bound on the top 64 bits of numbers below `2^96`: `2^33`, which is `2^65` again.
const LIMIT_1: u64 = 1 << 33;

/// The quotient and remainder of `n` by `d`, `n >= d`, the quotient below `2^63`: three
/// conditional subtractions for a quotient below 8, one bit at a time otherwise.
#[inline(always)]
fn divmod2(n: u128, d: u128) -> (u64, u128) {
    if (n >> 3) < d && d >> 125 == 0 {
        let mut r = n;
        let mut q = 0;
        for k in [2, 1, 0] {
            let (t, borrow) = r.overflowing_sub(d << k);
            r = core::hint::select_unpredictable(borrow, r, t);
            q = (q << 1) | u64::from(!borrow);
        }
        return (q, r);
    }
    long_q2(n, d)
}

/// [`divmod2`] for large quotients.
#[inline(never)]
fn long_q2(mut n: u128, d: u128) -> (u64, u128) {
    let k = d.leading_zeros().saturating_sub(n.leading_zeros());
    let mut d = d << k;
    let mut q = 0u64;
    for _ in 0..=k {
        let take = n >= d;
        q = (q << 1) | u64::from(take);
        n -= if take { d } else { 0 };
        d >>= 1;
    }
    (q, n)
}

/// The quotient and remainder of `n` by `d`, `n >= d > 0`.
#[inline(always)]
fn divmod1(n: u64, d: u64) -> (u64, u64) {
    if (n >> 3) < d && d >> 61 == 0 {
        let mut r = n;
        let mut q = 0;
        for k in [2, 1, 0] {
            let (t, borrow) = r.overflowing_sub(d << k);
            r = core::hint::select_unpredictable(borrow, r, t);
            q = (q << 1) | u64::from(!borrow);
        }
        return (q, r);
    }
    let q = n / d;
    (q, n - q * d)
}

/// A column of the matrix.
type Column = (u64, u64);

#[inline(always)]
fn plus_times(c: Column, q: u64, d: Column) -> Column {
    (c.0 + q * d.0, c.1 + q * d.1)
}

/// Möller's hgcd2 on the top two limbs of `a` and `b` (the top bit of one of them set), as the
/// greatest common divisor takes it, with each step's quotient carried into the state `s`; `None`
/// when not even one step can be taken. See the greatest common divisor's `hgcd2` for why every
/// step is a step of the whole numbers.
pub(in super::super) fn hgcd2(ah: u64, al: u64, bh: u64, bl: u64, s: &mut State) -> Option<Matrix> {
    if ah < LIMIT_2 || bh < LIMIT_2 {
        return None;
    }
    let (a, b) = (
        (u128::from(ah) << 64) | u128::from(al),
        (u128::from(bh) << 64) | u128::from(bl),
    );
    let (mut r0, mut r1, mut r0_is_a) = if a > b { (a, b, true) } else { (b, a, false) };
    let (mut c0, mut c1): (Column, Column) = if r0_is_a {
        ((1, 0), (0, 1))
    } else {
        ((0, 1), (1, 0))
    };
    let mut st = *s;
    let first = r0 - r1;
    if ((first >> 64) as u64) < LIMIT_2 {
        return None;
    }
    let (q, r) = divmod2(first, r1);
    if ((r >> 64) as u64) < LIMIT_2 {
        // The step with the quotient one less, `q`, which leaves the larger number larger.
        st = update(st, r0_is_a, q);
        *s = st;
        return Some(matrix(c0, plus_times(c1, q, c0), r0_is_a));
    }
    st = update(st, r0_is_a, q + 1);
    (c0, c1) = (plus_times(c1, q + 1, c0), c0);
    (r0, r1) = (r1, r);
    r0_is_a = !r0_is_a;
    while (r0 >> 64) as u64 >= 1 << 32 {
        let (q, r) = divmod2(r0, r1);
        if ((r >> 64) as u64) < LIMIT_2 {
            st = update(st, r0_is_a, q - 1);
            *s = st;
            return Some(matrix(c0, plus_times(c1, q - 1, c0), r0_is_a));
        }
        st = update(st, r0_is_a, q);
        (c0, c1) = (plus_times(c1, q, c0), c0);
        (r0, r1) = (r1, r);
        r0_is_a = !r0_is_a;
    }
    let (mut s0, mut s1) = ((r0 >> 32) as u64, (r1 >> 32) as u64);
    loop {
        let (q, r) = divmod1(s0, s1);
        if r < LIMIT_1 {
            st = update(st, r0_is_a, q - 1);
            *s = st;
            return Some(matrix(c0, plus_times(c1, q - 1, c0), r0_is_a));
        }
        st = update(st, r0_is_a, q);
        (c0, c1) = (plus_times(c1, q, c0), c0);
        (s0, s1) = (s1, r);
        r0_is_a = !r0_is_a;
    }
}

/// The matrix of the columns `c0` of the larger number and `c1` of the smaller.
#[inline(always)]
fn matrix(c0: Column, c1: Column, r0_is_a: bool) -> Matrix {
    let (ca, cb) = if r0_is_a { (c0, c1) } else { (c1, c0) };
    Matrix {
        u00: ca.0,
        u01: cb.0,
        u10: ca.1,
        u11: cb.1,
    }
}

/// Sets `(a, b)` to `(u11 a - u01 b, u00 b - u10 a)`, the pair reduced by the inverse of `m`, in
/// place over their `n` limbs.
fn reduce(a: &mut [u64], b: &mut [u64], m: Matrix) {
    let (mut ca, mut cb) = (0i128, 0i128);
    for i in 0..a.len() {
        let (x, y) = (i128::from(a[i]), i128::from(b[i]));
        let pa = i128::from(m.u11) * x - i128::from(m.u01) * y + ca;
        let pb = i128::from(m.u00) * y - i128::from(m.u10) * x + cb;
        a[i] = pa as u64;
        b[i] = pb as u64;
        ca = pa >> 64;
        cb = pb >> 64;
    }
    debug_assert!(ca == 0 && cb == 0);
}

/// The limbs without zeros at the top.
fn len(x: &[u64]) -> usize {
    x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)
}

/// The order of the two numbers of the same length.
fn cmp(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// `x -= y`, `x >= y`.
fn sub_in(x: &mut [u64], y: &[u64]) {
    let mut borrow = false;
    for (i, l) in x.iter_mut().enumerate() {
        let (d, b1) = l.overflowing_sub(y.get(i).copied().unwrap_or(0));
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        *l = d;
        borrow = b1 | b2;
    }
}

/// The Jacobi symbol `(a / b)` of limbs, the least significant first, for `a < b` and `b` odd:
/// Lehmer's steps while a number has three limbs or more, then the binary method on two, which
/// measured faster there.
pub(in super::super) fn jacobi(mut x: Vec<u64>, mut y: Vec<u64>) -> i32 {
    let n = y.len();
    x.resize(n, 0);
    let mut s = init(x.first().copied().unwrap_or(0), y[0]);
    loop {
        let n = len(&y).max(len(&x));
        if n <= 2 {
            return finish(&x, &y, s);
        }
        let (xh, xl, yh, yl) = top(&x[..n], &y[..n]);
        if let Some(m) = hgcd2(xh, xl, yh, yl, &mut s) {
            reduce(&mut x[..n], &mut y[..n], m);
        } else if !subdiv(&mut x, &mut y, n, &mut s) {
            // One number is much smaller than the other, or the two are close: one step by a
            // subtraction and a division, which may bring a number to 0.
            return finish(&x, &y, s);
        }
    }
}

/// Euclid's algorithm on `x > y`, limbs of the same length, by Lehmer's steps while the smaller
/// number has more than `bound + 128` bits: every step that hgcd2 takes leaves both numbers at
/// least `2^(bits - 64)` for the larger's `bits`, so none of them passes below `2^bound`. The pair
/// left is a pair of consecutive remainders of Euclid's sequence, the smaller still above
/// `2^bound`; Cornacchia's algorithm takes its last steps one at a time.
pub(in super::super) fn euclid_above(x: &mut [u64], y: &mut [u64], bound: u64) {
    let mut s = init(0, 1);
    loop {
        let n = len(x).max(len(y));
        let small = len(y).min(len(x));
        let small_bits = if small == 0 {
            0
        } else {
            let top = if len(x) <= len(y) {
                x[small - 1]
            } else {
                y[small - 1]
            };
            64 * small as u64 - u64::from(top.leading_zeros())
        };
        if n < 3 || small_bits <= bound + 128 {
            return;
        }
        let (xh, xl, yh, yl) = top(&x[..n], &y[..n]);
        match hgcd2(xh, xl, yh, yl, &mut s) {
            Some(m) => reduce(&mut x[..n], &mut y[..n], m),
            None => return,
        }
    }
}

/// The top 128 bits of `x` and `y`, of `n >= 2` limbs, shifted together so that the top bit of
/// one of them is set.
fn top(x: &[u64], y: &[u64]) -> (u64, u64, u64, u64) {
    let n = x.len();
    let mask = x[n - 1] | y[n - 1];
    let shift = mask.leading_zeros();
    let take = |v: &[u64]| -> (u64, u64) {
        let third = if n >= 3 { v[n - 3] } else { 0 };
        if shift == 0 {
            (v[n - 1], v[n - 2])
        } else {
            (
                (v[n - 1] << shift) | (v[n - 2] >> (64 - shift)),
                (v[n - 2] << shift) | (third >> (64 - shift)),
            )
        }
    };
    let (xh, xl) = take(x);
    let (yh, yl) = take(y);
    (xh, xl, yh, yl)
}

/// The step that hgcd2 cannot take: the larger number less the smaller, then the larger by the
/// smaller, the state carried through both; `false` when a number reaches 0.
fn subdiv(x: &mut Vec<u64>, y: &mut Vec<u64>, n: usize, s: &mut State) -> bool {
    use super::super::{ByteOrder, Number};
    use super::ring::{from_limbs, limbs_of};
    let x_larger = cmp(&x[..n], &y[..n]) == core::cmp::Ordering::Greater;
    let (big, small) = if x_larger {
        (&mut *x, &*y)
    } else {
        (&mut *y, &*x)
    };
    if len(small) == 0 {
        return false;
    }
    sub_in(&mut big[..n], &small[..n]);
    *s = update(*s, x_larger, 1);
    if cmp(&big[..n], &small[..n]) == core::cmp::Ordering::Less {
        return len(big) != 0;
    }
    if n <= 2 {
        let (b, d) = (to_u128(&big[..n]), to_u128(&small[..n]));
        let (q, r) = (b / d, b % d);
        *s = update(*s, x_larger, q as u64);
        big[0] = r as u64;
        big[1] = (r >> 64) as u64;
        return r != 0;
    }
    let bn = from_limbs(&big[..n], ByteOrder::LittleEndian);
    let sn = from_limbs(&small[..n], ByteOrder::LittleEndian);
    let (q, r): (Number, Number) = bn.div_rem(&sn);
    let ql = limbs_of(&q);
    *s = update(*s, x_larger, ql.first().copied().unwrap_or(0));
    let rl = limbs_of(&r);
    big.iter_mut().for_each(|l| *l = 0);
    big[..rl.len()].copy_from_slice(&rl);
    !rl.is_empty()
}

/// The symbol from the state and the pair, at most two limbs long now, or with a number 0.
fn finish(x: &[u64], y: &[u64], s: State) -> i32 {
    let (num, den) = if a_is_denominator(s) { (y, x) } else { (x, y) };
    if len(num) == 0 {
        // `(0 / d)` is 1 for `d = 1` only.
        return if len(den) == 1 && den[0] == 1 {
            sign(s)
        } else {
            0
        };
    }
    debug_assert!(len(num) <= 2 && len(den) <= 2);
    sign(s) * jacobi_128(to_u128(num), to_u128(den))
}

/// The Jacobi symbol `(x / n)` for an odd `n`, by the binary method: the factors of two of the
/// numerator taken out (each turns the sign when `n` is 3 or 5 modulo 8: bit 1 of `n ^ n/2`),
/// then the larger of two odd numbers less the smaller, swapped by reciprocity (which turns the
/// sign when both are 3 modulo 4: bit 1 of their and); the swap a select, without a branch.
#[inline]
pub(in super::super) fn jacobi_word(mut x: u64, mut n: u64) -> i32 {
    debug_assert!(n & 1 == 1);
    if x >= n {
        x %= n;
    }
    if x == 0 {
        return i32::from(n == 1);
    }
    // The sign is bit 1 of `bits`. The shift is counted on the difference before its sign is
    // made positive, which has the same zero bits, so that the two are not in one chain.
    let mut bits = 0u64;
    let z = x.trailing_zeros();
    x >>= z;
    bits ^= u64::from(z & 1) * (n ^ (n >> 1));
    while x != n {
        // Both odd: the smaller becomes the denominator, the difference the numerator.
        let (d, below) = x.overflowing_sub(n);
        let z = d.trailing_zeros();
        bits ^= if below { x & n } else { 0 };
        n = core::hint::select_unpredictable(below, x, n);
        x = (if below { d.wrapping_neg() } else { d }) >> z;
        bits ^= u64::from(z & 1) * (n ^ (n >> 1));
    }
    if n != 1 {
        return 0;
    }
    1 - 2 * ((bits >> 1) & 1) as i32
}

/// The limbs as a `u128`, of at most two significant ones.
fn to_u128(x: &[u64]) -> u128 {
    u128::from(x.first().copied().unwrap_or(0)) | (u128::from(x.get(1).copied().unwrap_or(0)) << 64)
}

/// The Jacobi symbol `(x / n)` for an odd `n` of up to two limbs, by [`jacobi_word`]'s binary
/// steps on two limbs while either number has a second one, then on one.
pub(in super::super) fn jacobi_128(mut x: u128, n: u128) -> i32 {
    debug_assert!(n & 1 == 1);
    if x >= n {
        // Most often below `2 n` for numbers of one length: a subtraction, not a division.
        x -= n;
        if x >= n {
            x %= n;
        }
    }
    if x == 0 {
        return i32::from(n == 1);
    }
    let mut bits = 0u64;
    let z = x.trailing_zeros();
    x >>= z;
    bits ^= u64::from(z & 1) * ((n ^ (n >> 1)) as u64);
    // On the limbs apart: the shift is counted on the low limb of the difference, which is not
    // zero but once in 2^64 (that case shifts by the whole limb first).
    let (mut xh, mut xl) = ((x >> 64) as u64, x as u64);
    let (mut nh, mut nl) = ((n >> 64) as u64, n as u64);
    while xh != 0 && nh != 0 {
        let (tl, b1) = xl.overflowing_sub(nl);
        let (th, b2) = xh.overflowing_sub(nh);
        let (th, b3) = th.overflowing_sub(u64::from(b1));
        let below = b2 | b3;
        if tl == 0 {
            if th == 0 {
                return 0;
            }
            // A whole limb of zeros: rare.
            let t = (u128::from(th) << 64) | u128::from(tl);
            let a = if below { t.wrapping_neg() } else { t };
            bits ^= if below { xl & nl } else { 0 };
            if below {
                (nh, nl) = (xh, xl);
            }
            let z = a.trailing_zeros();
            let a = a >> z;
            (xh, xl) = ((a >> 64) as u64, a as u64);
            bits ^= u64::from(z & 1) * (nl ^ (nl >> 1));
            continue;
        }
        let z = tl.trailing_zeros();
        bits ^= if below { xl & nl } else { 0 };
        nl = core::hint::select_unpredictable(below, xl, nl);
        nh = core::hint::select_unpredictable(below, xh, nh);
        // `|t|` by the mask of `below`.
        let m = 0u64.wrapping_sub(u64::from(below));
        let (al, c) = (tl ^ m).overflowing_sub(m);
        let ah = (th ^ m).wrapping_sub(m).wrapping_sub(u64::from(c));
        xl = (al >> z) | (ah << (64 - z));
        xh = ah >> z;
        bits ^= u64::from(z & 1) * (nl ^ (nl >> 1));
    }
    let x = (u128::from(xh) << 64) | u128::from(xl);
    let n = (u128::from(nh) << 64) | u128::from(nl);
    if n >> 64 == 0 {
        let sign = 1 - 2 * ((bits >> 1) & 1) as i32;
        return sign * jacobi_word((x % n) as u64, n as u64);
    }
    // `x` odd and of one limb: reciprocity, then `n` modulo `x`.
    bits ^= (x & n) as u64;
    let sign = 1 - 2 * ((bits >> 1) & 1) as i32;
    sign * jacobi_word((n % x) as u64, x as u64)
}
