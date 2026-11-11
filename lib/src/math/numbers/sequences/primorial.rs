//! Primorials: the product of the primes up to `n`.
//!
//! Up to 102, from a table of the products below `2^128`; below 8192, from the primality test's
//! table of the products of consecutive odd primes that fit in a limb. Beyond, the odd primes from
//! the crate's segmented sieve ([`primes`](crate::math::numbers::primes)), packed as many to a limb
//! as fit. Either
//! way the limbs are multiplied by the factorials' balanced product tree, as GMP's
//! `mpz_primorial_ui` does, a limb at a time up to 16 limbs, and the product shifted by a bit for
//! the prime 2.

use super::super::factorial::product::product;
use super::super::prime::tables::{GROUP, GROUPS, LIMIT, PRIMES};
use super::super::root::{limbs_number, lshift, mul_1, small_number};
use super::super::{ByteOrder, Number};
use core::mem::MaybeUninit;

/// `n#` for `n` up to 102: 101# is the last below `2^128`.
const PRIMORIALS: [u128; 103] = {
    let mut t = [1u128; 103];
    let mut i = 2;
    while i < 103 {
        // `i` is prime when no number from 2 to its square root divides it.
        let mut prime = true;
        let mut d = 2;
        while d * d <= i {
            if i % d == 0 {
                prime = false;
            }
            d += 1;
        }
        t[i] = if prime {
            t[i - 1] * i as u128
        } else {
            t[i - 1]
        };
        i += 1;
    }
    t
};

/// On 32-bit targets, `n#` has more than `isize::MAX` bytes from about `ln(2) 8 (2^31 - 1)`, `1.1908
/// 10^10`: above this bound, Dusart's `theta(n) > n (1 - 1 / ln^3 n)` (2010, for `n >= 89,967,803`)
/// puts `log2(n#) = theta(n) / ln 2` beyond `8 (2^31 - 1)` bits, which is refused at once, rather
/// than after the sieve. Between the two, a product too long for the target fails as any allocation
/// of that size does.
#[cfg(target_pointer_width = "32")]
const LONGEST: u64 = 11_909_131_521;

/// `n#`, in the byte order `order`.
pub(in super::super) fn primorial(n: u64, order: ByteOrder) -> Number {
    if n < PRIMORIALS.len() as u64 {
        return small_number(order, PRIMORIALS[n as usize], false);
    }
    long_primorial(n, order)
}

/// The limbs up to which a product is accumulated a limb at a time, in the frame: as the product
/// tree does up to 16.
const SEQUENTIAL: usize = 16;

/// `n#` for `n` above the table and below [`LIMIT`]: the odd primes' products of the groups of the
/// primality test, those of the groups whose primes are all at most `n` and the product of the
/// primes up to `n` of the next group.
#[inline(never)]
fn grouped_primorial(n: u64, order: ByteOrder) -> Number {
    let mut f = [MaybeUninit::<u64>::uninit(); GROUPS];
    let mut len = 0;
    for g in &GROUP {
        let primes = &PRIMES[g.first as usize..g.end as usize];
        if primes[primes.len() - 1].p <= n {
            // The product, odd, shifted left by its leading zeros.
            f[len].write(g.d >> g.d.trailing_zeros());
            len += 1;
        } else {
            let mut x = 1;
            for p in primes.iter().take_while(|p| p.p <= n) {
                x *= p.p;
            }
            if x != 1 {
                f[len].write(x);
                len += 1;
            }
            break;
        }
    }
    // SAFETY: the first `len` limbs are written.
    let f = unsafe { core::slice::from_raw_parts(f.as_ptr().cast::<u64>(), len) };
    if len <= SEQUENTIAL {
        let mut r = [0u64; SEQUENTIAL + 1];
        r[0] = f[0];
        let mut rn = 1;
        for &x in &f[1..] {
            let rp = r.as_mut_ptr();
            // SAFETY: the product of `rn` limbs has room for a limb more; in place, through one
            // pointer.
            let carry = unsafe { mul_1(rp, rp, rn, x) };
            if carry != 0 {
                r[rn] = carry;
                rn += 1;
            }
        }
        return doubled(&mut r, rn, order);
    }
    let mut r = product(f);
    let rn = r.len();
    r.push(0);
    doubled(&mut r, rn, order)
}

/// The `rn` limbs at `r`, odd, times 2, in the byte order `order`; `r` has a limb more.
fn doubled(r: &mut [u64], mut rn: usize, order: ByteOrder) -> Number {
    // SAFETY: the limbs are those of `r`, which has room for the one shifted out.
    unsafe {
        let rp = r.as_mut_ptr();
        let out = lshift(rp, rp, rn, 1);
        if out != 0 {
            r[rn] = out;
            rn += 1;
        }
        limbs_number(r.as_ptr(), rn, false, order)
    }
}

/// `n#` for `n` above the table.
#[inline(never)]
fn long_primorial(n: u64, order: ByteOrder) -> Number {
    if n < LIMIT as u64 {
        return grouped_primorial(n, order);
    }
    #[cfg(target_pointer_width = "32")]
    if n > LONGEST {
        panic!("capacity overflow");
    }
    // `log2(n#) < 1.4427 theta(n) < 1.443 n` (Schoenfeld, 1976: `theta(x) < 1.000028 x`), and each
    // limb but the last holds more than `64 - log2(n)` bits of the product.
    let per_limb = 64 - u64::from(64 - n.leading_zeros());
    let room = (n / per_limb).saturating_mul(3) / 2 + GROUPS as u64 + 2;
    let mut f = Vec::with_capacity(usize::try_from(room).unwrap_or(usize::MAX));
    // The odd primes below `LIMIT` from the groups' products, then the sieve's, as many to a limb as
    // fit.
    f.extend(GROUP.iter().map(|g| g.d >> g.d.trailing_zeros()));
    let mut x = 1u64;
    for p in crate::math::numbers::primes(LIMIT as u64..=n) {
        match x.checked_mul(p) {
            Some(y) => x = y,
            None => {
                f.push(x);
                x = p;
            }
        }
    }
    if x != 1 {
        f.push(x);
    }
    let mut r = product(&f);
    let rn = r.len();
    r.push(0);
    doubled(&mut r, rn, order)
}
