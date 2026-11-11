//! Integer sequences: [`Number::fibonacci`] and [`Number::lucas`], the Fibonacci and Lucas
//! numbers; [`Number::lucas_sequence`] and [`Number::lucas_sequence_mod`], the Lucas sequences
//! exactly and modulo a number; [`Number::primorial`], the product of the primes up to a number;
//! and [`Number::partitions`], the number of partitions of a number.
//!
//! The Fibonacci and Lucas numbers are doubled from tables, two squares a bit, as GMP's
//! `mpz_fib_ui` does ([`fibonacci`]), and the exact Lucas sequences the same way with `Q^k` beside
//! them ([`lucas`]), the squares and products of up to eight limbs by straight-line kernels in
//! registers ([`small`]). Modulo a number, the sequences are ladders over the index's bits in
//! Montgomery's form, the Baillie–PSW test's arithmetic, and modulo the power of two of an even
//! modulus by the low halves of products ([`lucas_mod`]). Primorials are balanced product trees of
//! the primes from the primality test's tables and the segmented sieve ([`primorial`]), and
//! partition numbers Hardy, Ramanujan and Rademacher's series with a bound of the error of each of
//! its terms, after Johansson ([`partitions`]).

mod fibonacci;
mod lucas;
mod lucas_mod;
mod partitions;
mod primorial;
mod small;

pub(crate) use partitions::{PI_TABLE, pi};

use super::{ByteOrder, Number};

impl Number {
    /// Returns the Fibonacci number `F_n` (`F_0 = 0`, `F_1 = 1`, `F_(k+2) = F_(k+1) + F_k`), in the
    /// byte order `byte_order`, in the shortest form, in one new vector.
    ///
    /// Up to `F_186`, the last below `2^128`, from a table, and up to `F_373` from one product of
    /// two numbers of the tables, `F_(2k) = F_k L_k` or `F_(2k+1) = F_(k+2) L_(k-1) + 2 (-1)^k`;
    /// beyond, by fast doubling from the table, as GMP's `mpz_fib_ui` does:
    /// `F_(2k-1) = F_k^2 + F_(k-1)^2` and `F_(2k+1) = 4 F_k^2 - F_(k-1)^2 + 2 (-1)^k`, two squares
    /// for each bit of `n`, and one product for the last, `F_(2k) = F_k (F_k + 2 F_(k-1))` or
    /// `F_(2k+1) = (2 F_k + F_(k-1)) (2 F_k - F_(k-1)) + 2 (-1)^k`. The time is that of a few
    /// products of the result's length, `0.694 n` bits.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector: with
    /// `capacity overflow` beyond `isize::MAX` bytes, and otherwise as the allocator fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::fibonacci(10, ByteOrder::BigEndian).value, [55]);
    /// // F_100 = 354,224,848,179,261,915,075.
    /// let f = Number::fibonacci(100, ByteOrder::LittleEndian);
    /// assert_eq!(f.to_string(), "354224848179261915075");
    /// ```
    pub fn fibonacci(n: u64, byte_order: ByteOrder) -> Number {
        fibonacci::fibonacci(n, byte_order)
    }

    /// Returns the Lucas number `L_n` (`L_0 = 2`, `L_1 = 1`, `L_(k+2) = L_(k+1) + L_k`), in the
    /// byte order `byte_order`, in the shortest form, in one new vector.
    ///
    /// Up to `L_184`, the last below `2^128`, from a table; beyond, as GMP's `mpz_lucnum_ui` does:
    /// the zero bits at the bottom of `n` as squares, `L_(2k) = L_k^2 - 2 (-1)^k`, from the table or
    /// from the odd part `2k + 1`, `L_(2k+1) = L_k L_(k+1) - (-1)^k` from the table up to 367 and
    /// `5 F_(k-1) (2 F_k + F_(k-1)) - 4 (-1)^k` beyond, with `F_k` and `F_(k-1)` from the doubling
    /// of [`fibonacci`](Self::fibonacci).
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for
    /// [`fibonacci`](Self::fibonacci).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::lucas(10, ByteOrder::BigEndian).value, [123]);
    /// // L_n = F_(n-1) + F_(n+1).
    /// let o = ByteOrder::LittleEndian;
    /// assert_eq!(Number::lucas(1000, o), &Number::fibonacci(999, o) + &Number::fibonacci(1001, o));
    /// ```
    pub fn lucas(n: u64, byte_order: ByteOrder) -> Number {
        fibonacci::lucas(n, byte_order)
    }

    /// Returns the Lucas sequences `(U_n(P, Q), V_n(P, Q))` of the integers `p` and `q`, of any size
    /// and sign: `U_0 = 0`, `U_1 = 1`, `V_0 = 2`, `V_1 = P`, and `X_(k+2) = P X_(k+1) - Q X_k`. Each
    /// is in the byte order of `p`, in the shortest form, in one new vector.
    ///
    /// By doubling from the top bit of `n`, two squares a bit: with `Q^k` beside them,
    /// `U_(2k-1) = U_k^2 - Q U_(k-1)^2`, `U_(2k+1) = (P^2 - 3Q) U_k^2 - Q^2 U_(k-1)^2 + 2 Q^k` and
    /// `U_(2k) = (U_(2k+1) + Q U_(2k-1)) / P`, an exact division, then `V_n = P U_n - 2Q U_(n-1)`;
    /// on limbs for `|P|` and `|Q|` below `2^30`, its first steps in `i128`s and each value a step
    /// in one pass over the squares' limbs, the division by `P` in the same pass; `P = ±1, Q = -1`
    /// and `P = ±3, Q = 1` from the Fibonacci and Lucas numbers (`U_n(3, 1) = F_(2n)`), and
    /// `P = 0`, `Q = 0` and `P^2 = 4Q` by their closed forms.
    ///
    /// # Panics
    ///
    /// Only when the results or their working memory cannot be allocated, as for
    /// [`fibonacci`](Self::fibonacci).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let o = ByteOrder::BigEndian;
    /// // U_n(3, 2) = 2^n - 1 and V_n(3, 2) = 2^n + 1.
    /// let (u, v) = Number::lucas_sequence(&Number::new_from_i32(3, o), &Number::new_from_i32(2, o), 20);
    /// assert_eq!((u.to_string(), v.to_string()), ("1048575".into(), "1048577".into()));
    /// // U_n(1, -1) and V_n(1, -1) are the Fibonacci and Lucas numbers.
    /// let (u, v) = Number::lucas_sequence(&Number::new_from_i32(1, o), &Number::new_from_i32(-1, o), 90);
    /// assert_eq!((u, v), (Number::fibonacci(90, o), Number::lucas(90, o)));
    /// ```
    pub fn lucas_sequence(p: &Number, q: &Number, n: u64) -> (Number, Number) {
        lucas::lucas_sequence(p, q, n)
    }

    /// Returns the Lucas sequences `(U_n(P, Q), V_n(P, Q))` modulo `m`, each Euclid's remainder, in
    /// `0..m`, for integers `p` and `q` of any size and sign and an index `n` of any size, in the
    /// byte order of `m`, in the shortest form, in one new vector each.
    ///
    /// The modulus is split as `m = 2^z m'` with `m'` odd. Modulo `m'` the residues are in
    /// Montgomery's form, in registers up to seven limbs, and modulo `2^z` they are the low limbs
    /// of products; the two results are put together by the Chinese remainder theorem. The index's
    /// bits are taken from the top by a ladder: `U_k` and `U_(k+1)` for any `P` and `Q`, a product
    /// and two squares a bit; for `Q = ±1`, `V_k` and `V_(k+1)` (Montgomery's ladder for Williams's
    /// method `p + 1`), a product and a square a bit, then `U_n = (2 V_(n+1) - P V_n) / D` for
    /// `D = P^2 - 4Q` prime to `m'`, or `U_k` and `V_k` for a short index and a small `P`; and for a
    /// long `Q`, `V_k` and `V_(k+1)` with `Q^k` beside them.
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or negative, or `n` negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let o = ByteOrder::LittleEndian;
    /// let (one, minus_one) = (Number::new_from_i32(1, o), Number::new_from_i32(-1, o));
    /// // F_(10^30) modulo 10^9 + 7, from an index of 100 bits.
    /// let n = Number::new_from_u128(10u128.pow(30), o);
    /// let m = Number::new_from_u64(1_000_000_007, o);
    /// let (u, v) = Number::lucas_sequence_mod(&one, &minus_one, &n, &m);
    /// assert!(u < m && v < m);
    /// // A Lucas probable prime: U_(n+1)(1, -1) = 0 modulo the prime 2^127 - 1, as D = 5 is not a
    /// // square modulo it.
    /// let p = &(&one << 127usize) - 1u8;
    /// assert!(Number::lucas_sequence_mod(&one, &minus_one, &(&p + 1u8), &p).0.is_zero());
    /// ```
    pub fn lucas_sequence_mod(p: &Number, q: &Number, n: &Number, m: &Number) -> (Number, Number) {
        lucas_mod::lucas_sequence_mod(p, q, n, m)
    }

    /// Returns the primorial `n#`, the product of the primes up to `n` (1 for `n < 2`), in the byte
    /// order `byte_order`, in the shortest form, in one new vector.
    ///
    /// Up to 102 from a table; below 8192 from the primality test's table of products of
    /// consecutive odd primes that fit in a limb; beyond, those products and the odd primes from the
    /// segmented sieve of [`primes`](super::primes), as many to a limb as fit. The limbs are
    /// multiplied by balanced product trees, as GMP's `mpz_primorial_ui` does, so that the longest
    /// products take Karatsuba's and Toom-Cook's methods and the transforms.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector: `n#` takes
    /// about `0.18 n` bytes, and its sieve its segments and the primes up to `sqrt(n)`. On 32-bit
    /// targets, above `1.19 10^10`, where `n#` passes `isize::MAX` bytes, with `capacity overflow`
    /// at once.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::primorial(10, ByteOrder::BigEndian).value, [210]);
    /// assert_eq!(Number::primorial(1, ByteOrder::BigEndian).value, [1]);
    /// // 100# has 121 bits.
    /// assert_eq!(Number::primorial(100, ByteOrder::LittleEndian).bits(), 121);
    /// ```
    pub fn primorial(n: u64, byte_order: ByteOrder) -> Number {
        primorial::primorial(n, byte_order)
    }

    /// Returns `p(n)`, the number of partitions of `n` (the ways to write `n` as a sum of positive
    /// integers, their order aside; `p(0) = 1`), in the byte order `byte_order`, in the shortest
    /// form, in one new vector.
    ///
    /// Up to 416 from a table of Euler's pentagonal recurrence; beyond, by Hardy, Ramanujan and
    /// Rademacher's series, as Johansson implements it (2012): about `0.25 √n` terms
    /// `√(3/k) (4 / (24n - 1)) A_k(n) U(C/k)`, `C = (π/6) √(24n - 1)` and
    /// `U(x) = cosh(x) - sinh(x)/x`, until Rademacher's bound of the remainder is at most 1/4; each
    /// `A_k(n)` a product of cosines by Whiteman's factorisation, and each term at a precision that
    /// falls as `k` grows, in doubles, on limbs on the stack, or on numbers, with a bound of its
    /// error. The sum is rounded once the bounds add up to less than 1/2. The time is that of the
    /// first term's exponential, of the result's `3.7 √n` bits.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::partitions(5, ByteOrder::BigEndian).value, [7]);
    /// // p(1000) = 24,061,467,864,032,622,473,692,149,727,991.
    /// let p = Number::partitions(1000, ByteOrder::LittleEndian);
    /// assert_eq!(p.to_string(), "24061467864032622473692149727991");
    /// ```
    pub fn partitions(n: u64, byte_order: ByteOrder) -> Number {
        partitions::partitions(n, byte_order)
    }
}
