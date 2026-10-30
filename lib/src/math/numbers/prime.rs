//! Primality of [`Number`]s: [`Number::is_probable_prime`], the Baillie–PSW test, and
//! [`Number::next_prime`], the next probable prime.
//!
//! A number is first divided by the small primes, in groups whose products fit in a limb: one pass
//! of the long number for each group, four groups a pass, and a multiplication for each prime of a
//! group to tell whether it divides the remainder. What is left is tested by Baillie–PSW
//! ([`bpsw`]): a strong test to base 2 and a strong Lucas test with Selfridge's parameters, in
//! Montgomery's form, in registers up to seven limbs. Numbers below `2^64` take a path of their own,
//! all in registers.
//!
//! The next prime is searched for in intervals of odd numbers, first sieved by the small primes, as
//! GMP's `mpz_nextprime` does ([`next`]).

mod bpsw;
mod next;
mod tables;

use super::add::{SMALL, read_small};
use super::mul::{Operand, STACK_LIMBS, on_heap, on_stack, to_limbs};
use super::pow::montgomery::{Mont1, Mont2, MontN, even};
use super::{ByteOrder, Number};
use bpsw::{Bits, LimbField, bpsw, bpsw_limbs, bpsw_limbs_itch, reciprocal, rem_by};
use core::mem::MaybeUninit;
use tables::{GROUP, Group, PRIMES, groups_below};

impl Number {
    /// Returns whether the number is a probable prime, by the Baillie–PSW test: `false` for
    /// everything below 2, negative numbers included, and for every composite number it finds
    /// a factor of or that fails the test; `true` for every prime.
    ///
    /// Small factors are found by trial division; the rest is a strong (Miller–Rabin) test to base 2
    /// and a strong Lucas test with Selfridge's parameters. The answer is exact below `2^64`, where
    /// the test has been checked against every composite that passes Fermat's test to base 2, which
    /// Feitsma and Galway listed (2009), by Gilchrist: none passes both.
    /// Above, no composite is known that passes it, though such numbers are believed to exist; GMP's
    /// `mpz_probab_prime_p(n, 24)` is the same test.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert!(Number::new_from_u64(1_000_000_007, ByteOrder::BigEndian).is_probable_prime());
    /// // 2047 = 23 * 89 passes the strong test to base 2, but not the Lucas test.
    /// assert!(!Number::new_from_u32(2047, ByteOrder::LittleEndian).is_probable_prime());
    /// assert!(!Number::new_from_i32(-7, ByteOrder::BigEndian).is_probable_prime());
    /// ```
    pub fn is_probable_prime(&self) -> bool {
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let x = unsafe { read_small(&self.value, self.byte_order) };
            if x == 0 || self.negative {
                return false;
            }
            if x >> 64 == 0 {
                return is_prime_u64(x as u64);
            }
            return x & 1 == 1 && is_prime_limbs_small(&[x as u64, (x >> 64) as u64]);
        }
        let n = Operand::of(self);
        if n.significant == 0 || self.negative {
            return false;
        }
        let nn = n.significant.div_ceil(8);
        let compute = |s: *mut MaybeUninit<u64>| {
            let s = s.cast::<u64>();
            // SAFETY: the working memory holds the limbs of `n` and what the test needs.
            unsafe {
                to_limbs(n, s);
                if nn == 1 {
                    return is_prime_u64(*s);
                }
                *s & 1 == 1 && is_prime_odd(s, nn, s.add(even(nn)))
            }
        };
        let need = even(nn) + prime_itch(nn);
        if need <= STACK_LIMBS {
            on_stack(compute)
        } else {
            on_heap(need, compute)
        }
    }

    /// Returns the smallest probable prime above `self`, in the byte order of `self`, in one new
    /// vector of exactly its length: 2 for anything below 2, negative numbers included.
    ///
    /// "Probable prime" is that of [`is_probable_prime`](Self::is_probable_prime): exact below
    /// `2^64`, and with no known exception above. The odd numbers above `self` are sieved by the
    /// small primes in intervals, and those left are tested in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let p = Number::new_from_u32(1_000_000_000, ByteOrder::BigEndian).next_prime();
    /// assert_eq!(p.value, 1_000_000_007u32.to_be_bytes());
    /// assert_eq!(Number::new_from_i32(-5, ByteOrder::BigEndian).next_prime().value, [2]);
    /// ```
    pub fn next_prime(&self) -> Number {
        next::next_prime(self)
    }
}

// Below 2^64

/// The primes up to which a number below `2^64` is divided before the test: each costs a
/// multiplication, and saves the test of the composites it finds. Measured on an Apple M4 Max,
/// random odd numbers of 64 bits took 311 cycles with the primes to 64, 286 to 128, 271 to 256 and
/// 259 to 512, and primes 2480 to 2550, 2540, 2550 to 2580 and 2605 to 2670.
const TRIAL_64: u64 = 256;

/// The index in [`PRIMES`] of the first prime above [`TRIAL_64`].
const TRIAL_64_END: usize = {
    let mut i = 0;
    while PRIMES[i].p <= TRIAL_64 {
        i += 1;
    }
    i
};

/// Whether `n` is a probable prime, exactly as it is below `2^64`.
fn is_prime_u64(n: u64) -> bool {
    if n < 4 {
        return n >= 2;
    }
    if n & 1 == 0 {
        return false;
    }
    for p in &PRIMES[..TRIAL_64_END] {
        if p.divides(n) {
            return n == p.p;
        }
    }
    // A composite has a prime factor at most its square root, and none is below the next prime.
    let next = PRIMES[TRIAL_64_END].p;
    if n < next * next {
        return true;
    }
    let f = Mont1::new(n);
    let bits = Bits {
        n: &n,
        len: 64 - n.leading_zeros() as usize,
    };
    // SAFETY: the bits are those of `n`, odd and without factors up to `TRIAL_64`.
    unsafe { bpsw(&f, bits, || selfridge_u64(n)) }
}

/// Selfridge's `D` for the odd `n`, at least 5, or `None` when `n` is a square or shares a factor
/// with the `D`s before it.
fn selfridge_u64(n: u64) -> Option<i64> {
    if is_square_u64(n) {
        return None;
    }
    let mut a = 5u64;
    loop {
        if a >= n {
            // Every `D` below `n` has a symbol of 1 or a factor in common, which for a number
            // without a small factor means it is prime; not reached for numbers beyond the trial
            // primes' squares, kept for safety.
            return Some(0);
        }
        let d = if a & 2 == 0 { a as i64 } else { -(a as i64) };
        match symbol(d, n % a, n & 3) {
            -1 => return Some(d),
            0 => return None,
            _ => a += 2,
        }
    }
}

/// Whether `n` is a perfect square.
fn is_square_u64(n: u64) -> bool {
    // Squares are 0, 1, 4 or 9 modulo 16.
    if (0x0213 >> (n & 15)) & 1 == 0 {
        return false;
    }
    let mut r = (n as f64).sqrt() as u64;
    while u128::from(r) * u128::from(r) > u128::from(n) {
        r -= 1;
    }
    while u128::from(r + 1) * u128::from(r + 1) <= u128::from(n) {
        r += 1;
    }
    r * r == n
}

/// The Jacobi symbol `(D/n)` for `|D| = a`, from `n mod a` and `n mod 4`, `n` odd and positive: by
/// reciprocity, `(a/n) = (n/a)` unless both are 3 modulo 4, and `(-1/n) = -1` when `n` is.
fn symbol(d: i64, n_mod_a: u64, n_mod_4: u64) -> i32 {
    let a = d.unsigned_abs();
    let mut j = jacobi(n_mod_a, a);
    if a & 3 == 3 && n_mod_4 == 3 {
        j = -j;
    }
    if d < 0 && n_mod_4 == 3 {
        j = -j;
    }
    j
}

/// The Jacobi symbol `(x/a)` for an odd `a`, by the binary method: the factors of two of `x` taken
/// out (each turns the sign over when `a` is 3 or 5 modulo 8), then the larger of the two odd
/// numbers less the smaller, swapped by reciprocity (which turns it over when both are 3 modulo 4).
fn jacobi(mut x: u64, mut a: u64) -> i32 {
    debug_assert!(a & 1 == 1);
    let mut t = 1;
    if x >= a {
        x %= a;
    }
    while x != 0 {
        let z = x.trailing_zeros();
        x >>= z;
        if z & 1 == 1 && matches!(a & 7, 3 | 5) {
            t = -t;
        }
        if x < a {
            if x & a & 2 != 0 {
                t = -t;
            }
            core::mem::swap(&mut x, &mut a);
        }
        x -= a;
    }
    if a == 1 { t } else { 0 }
}

// Long numbers

/// The primes up to which a number of `bits` bits is divided before the test: four times its bits,
/// from 256 to the table's 8192. Measured on an Apple M4 Max, random odd numbers of 512 bits took
/// 20,500 cycles with the primes to its bits, 19,000 to twice and 16,000 to four times; of 1024
/// bits 131,000, 94,000 and 94,600; primes the same.
fn trial_bound(bits: usize) -> u64 {
    (bits as u64 * 4).clamp(256, tables::LIMIT as u64)
}

/// The limbs of working memory of [`is_prime_odd`] for `nn` limbs.
fn prime_itch(nn: usize) -> usize {
    if nn >= 8 { bpsw_limbs_itch(nn) } else { 0 }
}

/// Whether the odd number of two limbs `n`, at least `2^64`, is a probable prime.
fn is_prime_limbs_small(n: &[u64; 2]) -> bool {
    // SAFETY: two limbs need no working memory.
    unsafe { is_prime_odd(n.as_ptr(), 2, core::ptr::null_mut()) }
}

/// The remainders of the `nn` limbs at `n` by four groups' `d`s at once, four independent chains a
/// limb.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline]
unsafe fn rem_by_4(n: *const u64, nn: usize, g: &[Group; 4]) -> [u64; 4] {
    let mut r = [0u64; 4];
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in (0..nn).rev() {
            let x = *n.add(i);
            for k in 0..4 {
                r[k] = super::div::limbs::div_2by1(r[k], x, g[k].d, g[k].v).1;
            }
        }
    }
    r
}

/// Whether one of the primes of the groups `g` divides the `nn` limbs at `n`.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn has_factor_in(n: *const u64, nn: usize, groups: &[Group]) -> bool {
    let (chunks, rest) = groups.as_chunks::<4>();
    for g in chunks {
        // SAFETY: as the caller guarantees.
        let r = unsafe { rem_by_4(n, nn, g) };
        for k in 0..4 {
            let primes = &PRIMES[g[k].first as usize..g[k].end as usize];
            if primes.iter().any(|p| p.divides(r[k])) {
                return true;
            }
        }
    }
    for g in rest {
        // SAFETY: as the caller guarantees.
        let r = unsafe { rem_by(n, nn, g.d, g.v) };
        let primes = &PRIMES[g.first as usize..g.end as usize];
        if primes.iter().any(|p| p.divides(r)) {
            return true;
        }
    }
    false
}

/// Whether the odd number of `nn >= 2` limbs at `n`, at least `2^64`, is a probable prime: trial
/// division, then Baillie–PSW, in registers up to seven limbs and in the [`prime_itch`] limbs at
/// `s` beyond.
///
/// # Safety
///
/// The limbs must be valid for reads, the top one not zero, and `s` for the working memory,
/// starting on 16 bytes.
unsafe fn is_prime_odd(n: *const u64, nn: usize, s: *mut u64) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let bits = Bits {
            n,
            len: 64 * nn - (*n.add(nn - 1)).leading_zeros() as usize,
        };
        let g = groups_below(trial_bound(bits.len));
        if has_factor_in(n, nn, &GROUP[..g]) {
            return false;
        }
        probable_prime_odd(n, nn, bits, s)
    }
}

/// Baillie–PSW for the odd number of `nn >= 2` limbs at `n`, at least `2^64` and with no prime
/// factor up to 47, in registers up to seven limbs and in the [`prime_itch`] limbs at `s` beyond.
///
/// # Safety
///
/// As for [`is_prime_odd`].
unsafe fn probable_prime_odd(n: *const u64, nn: usize, bits: Bits, s: *mut u64) -> bool {
    let select = || {
        // SAFETY: as the caller guarantees.
        unsafe { selfridge(n, nn) }
    };
    // SAFETY: as the caller guarantees.
    unsafe {
        match nn {
            2 => bpsw(&Mont2::new([*n, *n.add(1)]), bits, select),
            3 => bpsw(&MontN::<3>::new(limbs(n)), bits, select),
            4 => bpsw(&MontN::<4>::new(limbs(n)), bits, select),
            5 => bpsw(&MontN::<5>::new(limbs(n)), bits, select),
            6 => bpsw(&MontN::<6>::new(limbs(n)), bits, select),
            7 => bpsw(&MontN::<7>::new(limbs(n)), bits, select),
            _ => {
                let f = LimbField::new(n, nn, s);
                bpsw_limbs(&f, bits, s.add(LimbField::itch(nn)), select)
            }
        }
    }
}

/// The `N` limbs at `n`.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline(always)]
unsafe fn limbs<const N: usize>(n: *const u64) -> [u64; N] {
    let mut a = [0; N];
    // SAFETY: as the caller guarantees.
    unsafe { core::ptr::copy_nonoverlapping(n, a.as_mut_ptr(), N) };
    a
}

/// `3^2 5 7 11 ... 47`, the product of the odd primes to 47 and of 3 once more, below `2^64`: one
/// remainder by it gives those by the first `|D|`s of Selfridge's search and by the primes that
/// tell most squares apart.
const SELFRIDGE: u64 = 9 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47;

/// [`SELFRIDGE`] shifted so that its top bit is set.
const SELFRIDGE_D: u64 = SELFRIDGE << SELFRIDGE.leading_zeros();

/// The reciprocal of [`SELFRIDGE_D`].
const SELFRIDGE_V: u64 = (u128::MAX / SELFRIDGE_D as u128) as u64;

/// The odd primes to 47, for the squares.
const SQUARE_PRIMES: [u64; 14] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

/// Selfridge's `D` for the odd `n` of `nn >= 2` limbs, or `None` when `n` is a square or shares a
/// factor with the `D`s before it: the first of `5, -7, 9, -11, ...` whose Jacobi symbol `(D/n)`
/// is -1. A square's symbols are never -1, so squares are told apart first: the odd squares are 1
/// modulo 8 and squares modulo every prime, which leaves the square root to check for few others.
///
/// # Safety
///
/// The limbs must be valid for reads, the top one not zero.
unsafe fn selfridge(n: *const u64, nn: usize) -> Option<i64> {
    // SAFETY: as the caller guarantees.
    unsafe {
        let r = rem_by(n, nn, SELFRIDGE_D, SELFRIDGE_V) % SELFRIDGE;
        let low = *n;
        if low & 7 == 1
            && SQUARE_PRIMES
                .iter()
                .all(|&p| is_quadratic_residue(r % p, p))
            && is_square(n, nn)
        {
            return None;
        }
        let mut a = 5u64;
        loop {
            let d = if a & 2 == 0 { a as i64 } else { -(a as i64) };
            let n_mod_a = if SELFRIDGE.is_multiple_of(a) {
                r % a
            } else {
                let s = a.leading_zeros();
                rem_by(n, nn, a << s, reciprocal(a << s)) % a
            };
            match symbol(d, n_mod_a, low & 3) {
                -1 => return Some(d),
                0 => return None,
                _ => a += 2,
            }
        }
    }
}

/// Whether `r` is a square modulo the odd prime `p`, or zero: Euler's criterion, `r^((p-1)/2)`.
fn is_quadratic_residue(r: u64, p: u64) -> bool {
    r == 0 || jacobi(r, p) == 1
}

/// Whether the `nn` limbs at `n` are a perfect square: Newton's steps for the square root on
/// numbers, from a power of two above it. Called for the few numbers that look like squares.
///
/// # Safety
///
/// The limbs must be valid for reads, the top one not zero.
#[cold]
#[inline(never)]
unsafe fn is_square(n: *const u64, nn: usize) -> bool {
    // SAFETY: as the caller guarantees.
    let bytes: Vec<u8> = (0..nn)
        .flat_map(|i| unsafe { *n.add(i) }.to_le_bytes())
        .collect();
    let x = Number::new(bytes, ByteOrder::LittleEndian);
    // SAFETY: as the caller guarantees.
    let bits = 64 * nn - unsafe { *n.add(nn - 1) }.leading_zeros() as usize;
    let mut r = &Number::new_from_u8(1, ByteOrder::LittleEndian) << bits.div_ceil(2);
    loop {
        let next = &(&r + &(&x / &r)) >> 1;
        if !(&next - &r).is_negative() {
            break;
        }
        r = next;
    }
    (&r * &r).value == x.value
}
