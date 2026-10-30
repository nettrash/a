//! The odd part of `n!`, by Luschny's method of the swinging factorial ("Divide, Swing and Conquer
//! the Factorial", 2008), as GMP's `mpz_oddfac_1` computes it.
//!
//! The swinging factorial `n≀ = n! / (floor(n / 2)!)^2` is a product of primes: those above `n / 2`
//! once, those from `n / 3` to `n / 2` not at all, those from `sqrt(n)` to `n / 3` once when
//! `floor(n / p)` is odd, and each `p` below `sqrt(n)` to the number of odd `floor(n / p^i)`. So
//! the odd part of `n!` is the square of that of `floor(n / 2)!` times the odd part of `n≀`, a few
//! products of halving sizes, each from the primes of a sieve. Below [`SWING`], the odd part of
//! `m!` is the product of the odd numbers up to `m`, `m / 2`, `m / 4` and so on, each odd `j`
//! counted once for each `j 2^i <= m`: from a table of the products of the odd numbers up to 33,
//! and the odd numbers above in pairs, packed into limbs, all on the stack.

use super::super::mul::limbs::mul;
use super::super::root::{mul_1, sqrt_rem_u128};
use super::Limbs;
use super::primes::{Remainders, Sieve};
use super::product::{Packer, product, squared, times};
use core::mem::MaybeUninit;

/// From this `n`, the odd part of `n!` is computed from that of `n / 2` and the swinging factorial
/// of `n`; below, from its odd factors directly.
pub(super) const SWING: u64 = 256;

/// The limbs of the odd part of `m!` below [`SWING`], with one to spare: at most 23 are used.
pub(super) const SMALL_LIMBS: usize = 32;

/// The odd parts of `n!` for `n` up to 25, the last below `2^64`.
pub(super) const ODD_FACTORIALS: [u64; 26] = {
    let mut t = [1u64; 26];
    let mut i = 1;
    while i < 26 {
        let odd = (i as u64) >> (i as u64).trailing_zeros();
        t[i] = t[i - 1] * odd;
        i += 1;
    }
    t
};

/// The products of the odd numbers up to `2i + 1`, for `i` up to 16: up to `33!!`, the last below
/// `2^64`.
const ODD_DOUBLE_FACTORIALS: [u64; 17] = {
    let mut t = [1u64; 17];
    let mut i = 1;
    while i < 17 {
        t[i] = t[i - 1] * (2 * i as u64 + 1);
        i += 1;
    }
    t
};

/// The odd part of `n!`.
pub(super) fn odd_factorial(n: u64) -> Limbs {
    if n < ODD_FACTORIALS.len() as u64 {
        return vec![ODD_FACTORIALS[n as usize]];
    }
    // The number of halvings down to the direct method.
    let mut levels = 0;
    while n >> levels >= SWING {
        levels += 1;
    }
    let mut small = [0u64; SMALL_LIMBS];
    let len = small_odd_factorial(n >> levels, &mut small);
    let mut x = Vec::with_capacity(len + 1);
    x.extend_from_slice(&small[..len]);
    if levels > 0 {
        let sieve = Sieve::new(n);
        for level in (0..levels).rev() {
            let m = n >> level;
            x = times(&squared(&x), &swing(m, &sieve));
        }
    }
    x
}

/// Sets the limbs at `out` to the odd part of `m!` for `m` below [`SWING`], and returns their
/// number: each odd `j` from 3 up once for each `j 2^i <= m`, that is the odd numbers up to `m`,
/// `m / 2`, `m / 4` and so on, from the table up to 33 and in pairs above it, packed into limbs on
/// the stack and multiplied one limb at a time.
pub(super) fn small_odd_factorial(m: u64, out: &mut [u64; SMALL_LIMBS]) -> usize {
    debug_assert!(m < SWING);
    if m < ODD_FACTORIALS.len() as u64 {
        out[0] = ODD_FACTORIALS[m as usize];
        return 1;
    }
    // Below 256, the pairs above 33 take about 25 limbs, and the table 7.
    let mut f = [MaybeUninit::<u64>::uninit(); 64];
    let mut nf = 0;
    // A pair is below `m^2`, at most `2^16`.
    let limit = u64::MAX / (m * m);
    let mut prod = 1u64;
    let mut top = m;
    while top >= 3 {
        if top <= 33 {
            f[nf] = MaybeUninit::new(ODD_DOUBLE_FACTORIALS[(top as usize - 1) / 2]);
            nf += 1;
        } else {
            f[nf] = MaybeUninit::new(ODD_DOUBLE_FACTORIALS[16]);
            nf += 1;
            let mut j = 35;
            while j <= top {
                let pair = if j + 2 <= top { j * (j + 2) } else { j };
                if prod > limit {
                    f[nf] = MaybeUninit::new(prod);
                    nf += 1;
                    prod = pair;
                } else {
                    prod *= pair;
                }
                j += 4;
            }
        }
        top /= 2;
    }
    f[nf] = MaybeUninit::new(prod);
    nf += 1;
    // SAFETY: the first `nf` limbs are written.
    small_product(
        unsafe { core::slice::from_raw_parts(f.as_ptr().cast(), nf) },
        out,
    )
}

/// Sets the limbs at `out` to the product of the limbs `f`, none zero, which fits them, and returns
/// its number of limbs: two products of every other limb, one limb at a time, which the core
/// computes side by side, then multiplied together.
pub(super) fn small_product(f: &[u64], out: &mut [u64; SMALL_LIMBS]) -> usize {
    let mut a = [MaybeUninit::<u64>::uninit(); SMALL_LIMBS];
    let mut b = [MaybeUninit::<u64>::uninit(); SMALL_LIMBS];
    let (a, b) = (a.as_mut_ptr().cast::<u64>(), b.as_mut_ptr().cast::<u64>());
    let (mut na, mut nb) = (1, 1);
    // SAFETY: each array holds its product's limbs, all written before they are read, and room for
    // one more, as each product divides the whole one, which fits `out`; the product of the two
    // takes `nx + ny` limbs, at most one more than the whole product's, and below Karatsuba's
    // method it needs no scratch.
    unsafe {
        *a = f[0];
        *b = 1;
        let mut i = 1;
        while i < f.len() {
            let carry = mul_1(b, b, nb, f[i]);
            if carry != 0 {
                *b.add(nb) = carry;
                nb += 1;
            }
            if i + 1 < f.len() {
                let carry = mul_1(a, a, na, f[i + 1]);
                if carry != 0 {
                    *a.add(na) = carry;
                    na += 1;
                }
            }
            i += 2;
        }
        let ((x, nx), (y, ny)) = if na >= nb {
            ((a, na), (b, nb))
        } else {
            ((b, nb), (a, na))
        };
        mul(out.as_mut_ptr(), x, nx, y, ny, core::ptr::null_mut());
        let mut n = nx + ny;
        while out[n - 1] == 0 {
            n -= 1;
        }
        n
    }
}

/// The odd part of the swinging factorial of `m >= 9`, from the primes of `sieve`, which reaches
/// `m`.
fn swing(m: u64, sieve: &Sieve) -> Limbs {
    // About `m / ln m` primes up to `m`, several to a limb.
    let mut packer = Packer::new(m, (m / 64) as usize + 4);
    let root = sqrt_rem_u128(u128::from(m)).0;
    // The primes up to `sqrt(m)`, each to its power.
    sieve.each(3, root, |p| packer.push(prime_power(p, m)));
    // The primes from `sqrt(m)` to `m / 3`: once when `floor(m / p)` is odd, `m - p floor(m / p)`
    // having the parity of `m - floor(m / p)`, as `p` is odd.
    let mut r = Remainders::of(m);
    sieve.each(root + 1, m / 3, |p| {
        if (m ^ r.of_divisor(p)) & 1 == 1 {
            packer.push(p);
        }
    });
    // The primes above `m / 2`: once each.
    sieve.each(m / 2 + 1, m, |p| packer.push(p));
    product(&packer.finish())
}

/// The power of the odd prime `p <= sqrt(m)` in the swinging factorial of `m`: `p` to the number
/// of odd `floor(m / p^i)` for `i >= 1`, which is at most the largest power of `p` not above `m`.
fn prime_power(p: u64, m: u64) -> u64 {
    let mut q = m / p;
    let mut f = 1;
    while q > 0 {
        if q & 1 == 1 {
            f *= p;
        }
        q /= p;
    }
    f
}
