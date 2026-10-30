//! Binomial coefficients `C(n, k)`, for `k <= n - k`.
//!
//! Every method computes the odd part of `C(n, k)`, whose power of two is counted apart by Kummer's
//! theorem: it is the number of carries when `k` and `n - k` are added in base 2,
//! `popcount(k) + popcount(n - k) - popcount(n)`.
//!
//! - Below `2^128` ([`small`]): the odd part is the product of the odd parts of the factors
//!   `n - k + 1` to `n` divided by that of `k!`, which modulo `2^128` is a product by its inverse,
//!   tabulated: `k` products of `u128`s.
//! - A small `k` ([`by_division`]): the odd parts of the `k` factors are packed into limbs and
//!   multiplied by a product tree, and the product divided exactly by the odd part of `k!`: by one
//!   limb up to `k = 25`, by Hensel's method or by the division beyond.
//! - A large `k` ([`by_primes`]): Goetgheluck's method ("Computing Binomial Coefficients",
//!   American Mathematical Monthly, 1987), as GMP's `mpz_goetgheluck_bin_uiui`: the exponent of
//!   each odd prime `p <= n` in `C(n, k)` is the number of carries when `k` and `n - k` are added
//!   in base `p`, which is 1 for the primes above `n - k`, 0 from `n / 2` to `n - k`, at most 1
//!   above `sqrt(n)`; the prime powers are packed into limbs and multiplied by a product tree.

use super::super::div::limbs::{bdiv_q, binvert_limb};
use super::super::root::quotient;
use super::Limbs;
use super::odd::{ODD_FACTORIALS, SMALL_LIMBS, SWING, odd_factorial, small_odd_factorial};
use super::primes::{Remainders, Sieve};
use super::product::{Packer, product, product_into};
use core::mem::MaybeUninit;

/// For `k` from 2 to 65, the largest `n` for which `C(n, k) < 2^128` (capped at `u64::MAX`).
/// Computed exactly with Python's integers; the tests check each bound and the number above it.
pub(super) const SMALL_N: [u64; 64] = [
    18_446_744_073_709_551_615,
    12_686_161_381_664,
    9_506_325_305,
    132_496_420,
    7_910_345,
    1_080_110,
    246_703,
    79_266,
    32_301,
    15_633,
    8602,
    5222,
    3424,
    2386,
    1748,
    1333,
    1052,
    854,
    710,
    603,
    521,
    457,
    406,
    365,
    332,
    305,
    282,
    262,
    246,
    232,
    220,
    209,
    200,
    192,
    185,
    179,
    174,
    169,
    164,
    161,
    157,
    154,
    151,
    149,
    147,
    145,
    143,
    141,
    140,
    139,
    137,
    136,
    136,
    135,
    134,
    134,
    133,
    133,
    132,
    132,
    132,
    132,
    131,
    131,
];

/// For `k` up to 65, the inverse modulo `2^128` of the odd part of `k!`: Newton's iteration
/// `x (2 - a x)`, which doubles the bits that are right, from `x = a`, right modulo 8.
const INVERSES: [u128; 66] = {
    let mut t = [1u128; 66];
    let mut odd = 1u128;
    let mut i = 1;
    while i < 66 {
        let f = (i as u128) >> (i as u128).trailing_zeros();
        odd = odd.wrapping_mul(f);
        let mut x = odd;
        let mut step = 0;
        while step < 6 {
            x = x.wrapping_mul(2u128.wrapping_sub(odd.wrapping_mul(x)));
            step += 1;
        }
        t[i] = x;
        i += 1;
    }
    t
};

/// A coefficient with `k` above this and above `5 sqrt(n)` is computed from its prime factors
/// ([`by_primes`]), and otherwise by a division ([`by_division`]). The division takes about the
/// square of the numerator's limbs, `k log2(n) / 64`, the sieve of the primes the time of `n`: the
/// two met at `k = 5 sqrt(n)` within a few percent, measured on an Apple M4 Max, from 158 for
/// `n = 1000` (1062 cycles by the division, 1296 from the primes, for `k = 125`; 2529 against 1496
/// for `k = 250`) to about 5000 for `n = 10^6` (1.76 million cycles against 1.30 million for
/// `k = 7812`). Below about 500, the sieve and the products of the primes cost a few hundred cycles
/// whatever `k`, and the division was faster up to `k = 100`: for `n = 300`, 603 cycles against
/// 724 for `k = 87`, and 824 against 730 for `k = 120`. GMP's `mpz_bin_uiui` takes the primes
/// from `k > n / 16`, which here would leave the division four to eight times as slow from `n` of
/// about `10^5`.
const BY_PRIMES_K: u64 = 100;

/// Whether `C(n, k)`, for `k <= n / 2`, is computed from its prime factors: `k > 100` and
/// `k > 5 sqrt(n)` (see [`BY_PRIMES_K`]).
fn by_primes_for(n: u64, k: u64) -> bool {
    k > BY_PRIMES_K && u128::from(k) * u128::from(k) > 25 * u128::from(n)
}

/// The number of limb products of Hensel's method up to which an exact quotient is computed by it,
/// as `div_exact` does; beyond, by the division.
const HENSEL_LIMIT: usize = 1 << 18;

/// The power of two in `C(n, k)`, by Kummer's theorem.
fn twos(n: u64, k: u64) -> u64 {
    u64::from(k.count_ones() + (n - k).count_ones() - n.count_ones())
}

/// The odd part of `x`, not zero.
#[inline(always)]
fn odd(x: u64) -> u64 {
    x >> x.trailing_zeros()
}

/// `C(n, k)` for `k <= n - k` when it is below `2^128`, and `None` otherwise.
#[inline]
pub(super) fn small(n: u64, k: u64) -> Option<u128> {
    match k {
        0 => Some(1),
        1 => Some(u128::from(n)),
        2..=65 if n <= SMALL_N[k as usize - 2] => {
            let mut p = 1u128;
            for i in 0..k {
                p = p.wrapping_mul(u128::from(odd(n - i)));
            }
            Some(p.wrapping_mul(INVERSES[k as usize]) << twos(n, k))
        }
        _ => None,
    }
}

/// Computes the odd part of `C(n, k)` for `2 <= k <= n - k`, at or above `2^128`, and hands its
/// limbs, their number and the power of two it takes to `place`; the limbs have one more after
/// them, for a shift.
pub(super) fn long<R>(n: u64, k: u64, place: impl FnOnce(&mut [u64], usize, u64) -> R) -> R {
    let shift = twos(n, k);
    if by_primes_for(n, k) {
        let mut x = by_primes(n, k);
        let len = x.len();
        x.push(0);
        return place(&mut x, len, shift);
    }
    if k < SWING {
        return small_by_division(n, k, |x, len| place(x, len, shift));
    }
    let mut x = by_division(n, k);
    let len = x.len();
    x.push(0);
    place(&mut x, len, shift)
}

/// [`by_division`] for `k` below [`SWING`], on the stack: the numerator of at most `k` limbs, the
/// odd part of `k!` of at most 23, and the quotient, by Hensel's method; hands the quotient's limbs
/// and their number to `place`, with a limb to spare after them. The buffers are not initialised:
/// each limb is written before it is read, which spares clearing 10 KiB a call.
fn small_by_division<R>(n: u64, k: u64, place: impl FnOnce(&mut [u64], usize) -> R) -> R {
    const LIMBS: usize = SWING as usize;
    // At most one factor to a limb, `k <= 255` of them, packed as the `Packer` does.
    let mut f = [MaybeUninit::<u64>::uninit(); LIMBS];
    let mut nf = 0;
    let limit = u64::MAX / n;
    let mut prod = 1u64;
    for x in n - k + 1..=n {
        let o = odd(x);
        if prod > limit {
            f[nf] = MaybeUninit::new(prod);
            nf += 1;
            prod = o;
        } else {
            prod *= o;
        }
    }
    f[nf] = MaybeUninit::new(prod);
    nf += 1;
    let mut p = [MaybeUninit::<u64>::uninit(); LIMBS + 1];
    let mut halves = [MaybeUninit::<u64>::uninit(); 2 * LIMBS + 64];
    let mut scratch = Vec::new();
    let p = p.as_mut_ptr().cast::<u64>();
    // SAFETY: the first `nf` limbs of `f` are written; the arrays hold the product of the `nf`
    // limbs, at most `nf`, with the limb to spare, and the halves of `product_into`.
    let pn = unsafe {
        let f = core::slice::from_raw_parts(f.as_ptr().cast::<u64>(), nf);
        product_into(f, p, halves.as_mut_ptr().cast(), &mut scratch)
    };
    if k < ODD_FACTORIALS.len() as u64 {
        // SAFETY: the product's `pn` limbs are written, and the spare one is written here.
        let x = unsafe {
            *p.add(pn) = 0;
            core::slice::from_raw_parts_mut(p, pn + 1)
        };
        let len = divide_by_limb(&mut x[..pn], ODD_FACTORIALS[k as usize]);
        return place(x, len);
    }
    let mut d = [0u64; SMALL_LIMBS];
    let dn = small_odd_factorial(k, &mut d);
    let qn = pn - dn + 1;
    let mut q = [MaybeUninit::<u64>::uninit(); LIMBS + 1];
    let q = q.as_mut_ptr().cast::<u64>();
    // SAFETY: the arrays hold the `qn <= pn` limbs of the dividend, which are clobbered, the `dn`
    // of the divisor and the `qn` of the quotient, all apart; `bdiv_q` writes the quotient's
    // limbs, and the spare one is written here.
    unsafe {
        bdiv_q(q, p, qn, d.as_ptr(), dn, binvert_limb(d[0]));
        let mut len = qn;
        while *q.add(len - 1) == 0 {
            len -= 1;
        }
        *q.add(qn) = 0;
        place(core::slice::from_raw_parts_mut(q, qn + 1), len)
    }
}

/// The odd part of `C(n, k)` as the product of the odd parts of `n - k + 1` to `n`, divided
/// exactly by the odd part of `k!`, for a `k` of at least [`SWING`].
fn by_division(n: u64, k: u64) -> Limbs {
    let mut packer = Packer::new(n, k as usize / 2 + 2);
    for f in n - k + 1..=n {
        packer.push(odd(f));
    }
    let mut p = product(&packer.finish());
    exact_quotient(&mut p, &odd_factorial(k))
}

/// Divides the limbs `x` in place by the odd limb `d`, which divides them exactly, by Hensel's
/// method, and returns the quotient's number of limbs: each limb of the quotient is the one that
/// makes the low limb of what is left zero.
fn divide_by_limb(x: &mut [u64], d: u64) -> usize {
    let v = binvert_limb(d);
    let mut borrow = 0u64;
    for limb in x.iter_mut() {
        let (t, b) = limb.overflowing_sub(borrow);
        let q = t.wrapping_mul(v);
        *limb = q;
        borrow = ((u128::from(q) * u128::from(d)) >> 64) as u64 + u64::from(b);
    }
    debug_assert!(borrow == 0, "the division was not exact");
    let mut len = x.len();
    while len > 1 && x[len - 1] == 0 {
        len -= 1;
    }
    len
}

/// `x / d` for an odd `d` of at least two limbs that divides `x` exactly: by Hensel's method,
/// from the low end, up to [`HENSEL_LIMIT`] limb products, and by the division beyond.
fn exact_quotient(x: &mut Limbs, d: &[u64]) -> Limbs {
    let (xn, dn) = (x.len(), d.len());
    let qn = xn - dn + 1;
    let m = qn.min(dn);
    if qn * m - m * m / 2 >= HENSEL_LIMIT {
        return quotient(x, d);
    }
    // A limb to spare, for a shift.
    let mut q = Vec::with_capacity(qn + 1);
    q.resize(qn, 0);
    // SAFETY: the vectors hold the `qn <= xn` limbs of the dividend, which are clobbered, the
    // `dn` of the divisor and the `qn` of the quotient, all apart.
    unsafe {
        bdiv_q(
            q.as_mut_ptr(),
            x.as_mut_ptr(),
            qn,
            d.as_ptr(),
            dn,
            binvert_limb(d[0]),
        );
    }
    while q.last() == Some(&0) {
        q.pop();
    }
    q
}

/// The odd part of `C(n, k)` from its odd prime factors, for `k <= n / 2`.
fn by_primes(n: u64, k: u64) -> Limbs {
    let sieve = Sieve::new(n);
    let mut packer = Packer::new(n, (n / 64) as usize + 4);
    let root = super::super::root::sqrt_rem_u128(u128::from(n)).0;
    // Up to `sqrt(n)`: `p` to the number of carries when `k` and `n - k` are added in base `p`,
    // digit by digit, at most `n` in all.
    sieve.each(3, root, |p| {
        let (mut a, mut b) = (n, k);
        let mut carry = 0;
        let mut f = 1;
        while a > 0 {
            // The digits of `n` and `k`; that of `n - k` is the difference, with a borrow when the
            // sum carries.
            let (da, db) = (a % p, b % p);
            carry = u64::from(da < db + carry);
            if carry == 1 {
                f *= p;
            }
            a /= p;
            b /= p;
        }
        packer.push(f);
    });
    // Above `sqrt(n)`, `n` has two digits: one carry when the low digit of `n` is below that of
    // `k`; above `n / 2` that is all the primes above `n - k`. The remainders by increasing primes
    // take a division only when the quotient changes.
    let (mut rn, mut rk) = (Remainders::of(n), Remainders::of(k));
    sieve.each(root + 1, n / 2, |p| {
        if rn.of_divisor(p) < rk.of_divisor(p) {
            packer.push(p);
        }
    });
    sieve.each((n - k + 1).max(n / 2 + 1), n, |p| packer.push(p));
    product(&packer.finish())
}
