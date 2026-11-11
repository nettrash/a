//! Modular arithmetic: the Chinese remainder theorem, multiplicative orders, primitive roots and
//! discrete logarithms.
//!
//! The moduli below `2^64` take paths of their own, on words in registers: their factorisation
//! ([`word::factor`]), their residues in Montgomery's form, and the orders and logarithms counted
//! in words. Longer moduli are factored by [`Number::factor`] and their residues kept in the ring
//! that fits their length ([`ring`]), in registers up to seven limbs, and eight for the discrete
//! logarithms ([`dlog`](super::dlog)). Both run the same algorithms, written once over the integers
//! they count in ([`int`]).

pub(super) mod crt;
pub(super) mod int;
pub(super) mod jacobi;
pub(super) mod log;
pub(super) mod order;
pub(super) mod ring;
pub(super) mod root;
pub(super) mod sqrt;
pub(super) mod word;

use super::{ByteOrder, Number};
use int::{Int, in_order, product};
use ring::limbs_of;

impl Number {
    /// The solution of the congruences `x = r (mod m)` for each `(r, m)`, coprime moduli or not:
    /// `Some((x, l))` with `l` the least common multiple of the moduli and `0 <= x < l`, or `None`
    /// when the congruences contradict each other. No congruences give `Some((0, 1))`. In the byte
    /// order of the first residue, little-endian when there are none.
    ///
    /// The residues may have any sign and length. Two congruences `x = r (mod m)` and
    /// `x = s (mod n)` have a solution exactly when `g = gcd(m, n)` divides `s - r`, and it is
    /// `r + m t` with `t = (s - r)/g (m/g)^(-1) mod n/g`, modulo `m n / g`. A few moduli below
    /// `2^64` are joined in registers, and two moduli of up to 16 limbs on the stack; more moduli
    /// below `2^64` one at a time on limbs, as Garner's method does, in groups of 64, and from 150
    /// of them by the product and remainder trees in limbs; longer moduli, and the groups' results,
    /// are joined in pairs up a balanced tree, by the greatest common divisor and the inverse.
    ///
    /// # Panics
    ///
    /// Panics if a modulus is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // Sunzi: a number that is 2 modulo 3, 3 modulo 5 and 2 modulo 7.
    /// let (x, l) = Number::crt(&[(n(2), n(3)), (n(3), n(5)), (n(2), n(7))]).unwrap();
    /// assert_eq!((x, l), (n(23), n(105)));
    /// // Moduli with a common factor: 1 modulo 4 and 3 modulo 6 is 9 modulo 12.
    /// assert_eq!(Number::crt(&[(n(1), n(4)), (n(3), n(6))]), Some((n(9), n(12))));
    /// // 1 modulo 4 and 2 modulo 6 contradict each other: one is odd, the other even.
    /// assert_eq!(Number::crt(&[(n(1), n(4)), (n(2), n(6))]), None);
    /// ```
    #[track_caller]
    pub fn crt(congruences: &[(Number, Number)]) -> Option<(Number, Number)> {
        for (_, m) in congruences {
            positive(m);
        }
        let order = congruences
            .first()
            .map_or(ByteOrder::LittleEndian, |c| c.0.byte_order);
        // A few congruences modulo words, the common case: in registers while the solution fits in
        // two words, without a vector.
        if congruences.len() <= FEW {
            let mut w = [(0u64, 0u64); FEW];
            let mut words = true;
            for (o, (r, m)) in w.iter_mut().zip(congruences) {
                match ring::magnitude_64(m) {
                    Some(m) => *o = (word_residue(r, m), m),
                    None => {
                        words = false;
                        break;
                    }
                }
            }
            if words {
                let w = &w[..congruences.len()];
                let (x, l) = match crt::registers(w)? {
                    crt::Registers::Done(x, l) => (
                        ring::from_limbs(&[x as u64, (x >> 64) as u64], order),
                        ring::from_limbs(&[l as u64, (l >> 64) as u64], order),
                    ),
                    crt::Registers::Full(x, l, i) => {
                        let (x, l) = crt::limbs_from(w, x, l, i)?;
                        (ring::from_limbs(&x, order), ring::from_limbs(&l, order))
                    }
                };
                return Some((x, l));
            }
        }
        // Two congruences of moduli of a few limbs, on the stack.
        if let [(r1, m1), (r2, m2)] = congruences
            && let Some(x) = crt::pair((r1, m1), (r2, m2), order)
        {
            return Some(x);
        }
        let long_moduli = congruences.iter().any(|(_, m)| m.significant() > 8);
        if !long_moduli && congruences.len() >= crt::FAST_WORDS {
            let c: Vec<(u64, u64)> = congruences
                .iter()
                .map(|(r, m)| {
                    let m = ring::magnitude_64(m).unwrap_or(1);
                    (word_residue(r, m), m)
                })
                .collect();
            if let Some((x, l)) = crt::fast_words(&c) {
                return Some((ring::from_limbs(&x, order), ring::from_limbs(&l, order)));
            }
            // The moduli are not pairwise coprime, and the trees of numbers below would find so
            // again: the joins take them.
        } else if long_moduli && congruences.len() >= crt::FAST {
            let c: Vec<(Number, Number)> = congruences
                .iter()
                .map(|(r, m)| {
                    let m = in_order(m.abs(), ByteOrder::LittleEndian);
                    (in_order(r.rem_euclid(&m), ByteOrder::LittleEndian), m)
                })
                .collect();
            if let Some((x, l)) = crt::fast(&c) {
                return Some((in_order(x, order), in_order(l, order)));
            }
        }
        let mut words = Vec::new();
        let mut long = Vec::new();
        for (r, m) in congruences {
            match u64::try_from(m) {
                Ok(w) => words.push((word_residue(r, w), w)),
                Err(_) => long.push((r.rem_euclid(m), in_order(m.abs(), ByteOrder::LittleEndian))),
            }
        }
        let (x, l) = if long.is_empty() {
            crt::words(&words)?
        } else {
            if !words.is_empty() {
                long.push(crt::words(&words)?);
            }
            crt::tree(long)?
        };
        Some((in_order(x, order), in_order(l, order)))
    }

    /// The multiplicative order of `self` modulo `m`, the smallest `k >= 1` with `self^k = 1
    /// (mod m)`, or `None` when `self` and `m` have a common factor. Modulo 1 it is 1. In the byte
    /// order of `self`.
    ///
    /// `m` is factored, and the order is that modulo each prime power, put together by their least
    /// common multiple: modulo `2^e` the number of squares that bring `self` to 1; modulo `p^e`, `p`
    /// odd, the order modulo `p` among the primes of `p - 1` (which are factored too), by
    /// Sutherland's halving of the primes (2007), then the power of `p` that brings
    /// `self^(order mod p)` to 1 modulo `p^e`. Its cost is that of the factorisations and of
    /// about `log2(p) log2(k)` products modulo each `p^e`, for `k` primes of `p - 1`.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // 2 has order 3 modulo 7: 2, 4, 1.
    /// assert_eq!(n(2).multiplicative_order(&n(7)), Some(n(3)));
    /// // Modulo the prime 2^64 + 13, 3 has the order p - 1 = 2^2 7 658812288346769701, and 9
    /// // half that.
    /// let p = (n(1) << 64usize) + 13u8;
    /// assert_eq!(n(3).multiplicative_order(&p), Some(&p - 1u8));
    /// assert_eq!(n(9).multiplicative_order(&p), Some((&p - 1u8) >> 1usize));
    /// assert_eq!(n(6).multiplicative_order(&n(9)), None);
    /// ```
    #[track_caller]
    pub fn multiplicative_order(&self, m: &Number) -> Option<Number> {
        positive(m);
        let order = self.byte_order;
        if let Ok(w) = u64::try_from(m) {
            if w == 1 {
                return Some(Number::new_from_u8(1, order));
            }
            let a = word_residue(self, w);
            if word::gcd(a, w) != 1 {
                return None;
            }
            let f = word::factor(w);
            let o = order::order_mod::<u64>(&a, f.as_slice());
            return Some(Number::new_from_u64(product(&o), order));
        }
        let m = in_order(m.abs(), ByteOrder::LittleEndian);
        let a = self.rem_euclid(&m);
        if !a.gcd(&m).is_one() {
            return None;
        }
        let f = long_factors(&m);
        let o = order::order_mod::<Number>(&a, &f);
        Some(in_order(product(&o), order))
    }

    /// The smallest primitive root modulo `m`, a number whose powers give every residue prime to
    /// `m`, or `None` when there is none: there is one exactly for 1, 2, 4, `p^k` and `2 p^k`,
    /// `p` an odd prime. Modulo 1 it is 0. In the byte order of `m`.
    ///
    /// The candidates `g` are tried in increasing order (only the odd ones modulo `2 p^k`): `g` is
    /// a primitive root modulo `p` when its Jacobi symbol is −1 and `g^((p-1)/q)` is not 1 for any
    /// odd prime `q` of `p - 1`, which is factored, and one modulo `p^k`, `k >= 2`, when moreover
    /// `g^(p-1)` is not 1 modulo `p^2`. PARI's `znprimroot` gives the smallest root modulo a
    /// prime, but not always modulo `2 p^k` (11 modulo 18, where this gives 5).
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// assert_eq!(Number::primitive_root(&n(7)), Some(n(3)));
    /// // 5 is a primitive root modulo the prime 40487 but not modulo its square.
    /// assert_eq!(Number::primitive_root(&n(40487)), Some(n(5)));
    /// assert_eq!(Number::primitive_root(&n(40487 * 40487)), Some(n(10)));
    /// assert_eq!(Number::primitive_root(&n(15)), None);
    /// ```
    #[track_caller]
    pub fn primitive_root(m: &Number) -> Option<Number> {
        positive(m);
        let order = m.byte_order;
        if let Ok(w) = u64::try_from(m) {
            let g = match w {
                1 => 0,
                2 => 1,
                4 => 3,
                _ if w % 4 == 0 => return None,
                _ => {
                    let two = w % 2 == 0;
                    let odd = if two { w / 2 } else { w };
                    let f = word::factor(odd);
                    let [(p, k)] = *f.as_slice() else {
                        return None;
                    };
                    let p1 = word::factor(p - 1);
                    root::smallest::<u64>(&p, k, two, p1.as_slice())
                }
            };
            return Some(Number::new_from_u64(g, order));
        }
        let m = in_order(m.abs(), ByteOrder::LittleEndian);
        let two = !m.bit(0);
        if two && !m.bit(1) {
            return None;
        }
        let odd = if two { &m >> 1usize } else { m };
        let f = long_factors(&odd);
        if f.len() != 1 {
            return None;
        }
        let (p, k) = &f[0];
        let p1 = order::factor_below(p);
        let g = root::smallest::<Number>(p, *k, two, &p1);
        Some(Number::new_from_u64(g, order))
    }

    /// The discrete logarithm of `self` to `base` modulo `m`: the smallest `k >= 0` with
    /// `base^k = self (mod m)`, or `None` when there is none. By Pohlig and Hellman's reduction to
    /// the prime powers dividing the order of `base`, each by the baby-step giant-step method,
    /// Pollard's rho or the index calculus, as fits its size. In the byte order of `self`.
    ///
    /// Any modulus and any `base` are taken, prime to `m` or not: `m` is factored, and each prime
    /// power `p^e` says what it can of `k`. One that `base` is prime to gives `k` modulo the order of
    /// `base` there: modulo the part of the order that divides `p - 1`, computed modulo `p`, and
    /// modulo the part that is a power of `p`, computed modulo `p^e` on the powers to `p - 1`, by
    /// Pohlig and Hellman (1978) over the primes of `p - 1`, factored, and `p` itself: each prime `q`
    /// of the order a digit at a time, by stepping through the powers for `q < 16`, by Shanks's baby
    /// steps and giant steps up to `2^21` modulo a word and `2^23` modulo longer numbers, by
    /// Pollard's rho beyond, and for `q = p` by a division, as the elements of order `p` modulo
    /// `p^e` are `1 + p^(e-1) c`. For a prime `p` of at most 112 bits whose `p - 1` has a large
    /// prime `q` once (from `2^33` up to 40 bits, `2^40` at 64 and `2^53` at 112), that `q` is
    /// taken by the index calculus: the linear sieve of Coppersmith, Odlyzko and Schroeppel
    /// (1986) finds the logarithms of a base of small primes modulo `q`, and each logarithm needed
    /// comes from a multiple whose rational reconstruction modulo `p` has smooth halves. One that
    /// divides `base` by `p^v` gives `k >= ceil(e / v)` when it divides `self` `e` times, and
    /// `k = v_p(self) / v` otherwise. The classes are joined by the Chinese remainder theorem. The
    /// cost is that of the factorisations and about `sqrt(q)` products modulo `p^e` for the
    /// largest prime `q` of the order, or that of the index calculus modulo `p`, subexponential:
    /// about a millisecond at 56 bits, 6 at 72 and 80 at 96.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = |x| Number::new_from_i64(x, ByteOrder::BigEndian);
    /// // 3^4 = 81 = 13 modulo 17.
    /// assert_eq!(n(13).discrete_log(&n(3), &n(17)), Some(n(4)));
    /// // 2 has order 3 modulo 7, and 3 is not one of its powers.
    /// assert_eq!(n(3).discrete_log(&n(2), &n(7)), None);
    /// // A base that is not prime to the modulus: 2^3 = 8 = 0 modulo 8, and 2^2 = 4 modulo 12.
    /// assert_eq!(n(0).discrete_log(&n(2), &n(8)), Some(n(3)));
    /// assert_eq!(n(4).discrete_log(&n(2), &n(12)), Some(n(2)));
    /// ```
    #[track_caller]
    pub fn discrete_log(&self, base: &Number, m: &Number) -> Option<Number> {
        positive(m);
        let order = self.byte_order;
        if let Ok(w) = u64::try_from(m) {
            if w == 1 {
                return Some(Number::new_from_u8(0, order));
            }
            let h = word_residue(self, w);
            if h == 1 {
                return Some(Number::new_from_u8(0, order));
            }
            let g = word_residue(base, w);
            let f: Vec<(u64, u32)> = word::factor(w).iter().collect();
            return log::discrete_log::<u64>(&h, &g, &w, &f)
                .map(|k| Number::new_from_u64(k, order));
        }
        let m = in_order(m.abs(), ByteOrder::LittleEndian);
        let h = self.rem_euclid(&m);
        if h.is_one() {
            return Some(Number::new_from_u8(0, order));
        }
        let g = base.rem_euclid(&m);
        let f = long_factors(&m);
        log::discrete_log::<Number>(&h, &g, &m, &f).map(|k| in_order(k, order))
    }
}

/// Up to this many congruences, [`Number::crt`] first tries them in registers.
const FEW: usize = 8;

/// Panics unless `m` is positive.
#[inline(always)]
#[track_caller]
pub(super) fn positive(m: &Number) {
    if m.negative || m.is_zero() {
        not_positive();
    }
}

/// Panics for a modulus that is zero or negative.
#[cold]
#[inline(never)]
#[track_caller]
pub(super) fn not_positive() -> ! {
    panic!("the modulus must be positive")
}

/// `x mod m`, Euclid's remainder, for a word `m >= 1`.
pub(super) fn word_residue(x: &Number, m: u64) -> u64 {
    let r = if let Some(v) = ring::magnitude_128(x) {
        (v % u128::from(m)) as u64
    } else {
        limbs_of(x).iter().rev().fold(0u64, |r, &l| {
            (((u128::from(r) << 64) | u128::from(l)) % u128::from(m)) as u64
        })
    };
    if x.is_negative() && r != 0 { m - r } else { r }
}

/// The factorisation of the number `m >= 2^64`, by [`Number::factor`], the primes little-endian.
pub(super) fn long_factors(m: &Number) -> Vec<(Number, u32)> {
    m.factor()
        .into_factors()
        .into_iter()
        .map(|(p, e)| (in_order(p, ByteOrder::LittleEndian), e))
        .collect()
}
