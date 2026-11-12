//! The Bernoulli numbers `B_n`, with `B_1 = -1/2`.
//!
//! Up to `B_100` from tables. Beyond, `|B_n| = 2 n! ζ(n) / (2π)^n` ([`by_zeta`]): the denominator
//! `d` by von Staudt and Clausen's theorem, `K = 2 d n!` exactly, and `K ζ(n) / (2π)^n` to the bits
//! of `|B_n| d` and 8 more, rounded; `π` from the table of the partition numbers (16,384 bits) or
//! their Chudnovsky series, `(2π)^n` by powers cut to the precision, `1/ζ(n)` by its Euler product
//! over the primes `p` with `p^n` below `2^w`, each factor `z <- z - z / p^n` taken to the bits it
//! moves the product by. From `n = 8000` the low bits of `|B_n| d` come from its residues modulo
//! primes ([`modp`](super::modp)), joined by the Chinese remainder theorem, and the formula needs
//! only the rest: the multiple of their product nearest to the formula's value is added.
//!
//! The products and the quotients of the formula reuse a few vectors ([`Work`]): the squares and
//! products by the numbers' kernels into them, the quotients by divisors of up to
//! [`SCHOOL_DIVISION`] limbs by Knuth's algorithm D in them, so that a short number allocates a
//! handful of times rather than at every step.
//!
//! The table of all the numbers up to an index ([`bernoulli_numbers`]) is computed from the tangent
//! numbers when short ([`tangent_numbers`]), and otherwise by the same formula from the top index
//! down ([`table_by_zeta`]).

use super::super::Rational;
use super::limbs::{
    Limbs, bits, mul, mul_1, normalize, number_u128, odd_primes, of_number, product, shl, shl_into,
    shr, shr_assign, shr_into, sub, to_number,
};
use crate::math::numbers::nat::{div_2by1, invert_limb, mul_into, sqr_into};
use crate::math::numbers::{ByteOrder, Number, PI_TABLE, is_prime_u64, pi, primes};

/// `B_0, B_2, ..., B_58` as (numerator, denominator): `B_60` is the first whose numerator passes
/// `2^127`.
pub(super) const TABLE: [(i128, u64); 30] = [
    (1, 1),
    (1, 6),
    (-1, 30),
    (1, 42),
    (-1, 30),
    (5, 66),
    (-691, 2730),
    (7, 6),
    (-3617, 510),
    (43867, 798),
    (-174611, 330),
    (854513, 138),
    (-236364091, 2730),
    (8553103, 6),
    (-23749461029, 870),
    (8615841276005, 14322),
    (-7709321041217, 510),
    (2577687858367, 6),
    (-26315271553053477373, 1919190),
    (2929993913841559, 6),
    (-261082718496449122051, 13530),
    (1520097643918070802691, 1806),
    (-27833269579301024235023, 690),
    (596451111593912163277961, 282),
    (-5609403368997817686249127547, 46410),
    (495057205241079648212477525, 66),
    (-801165718135489957347924991853, 1590),
    (29149963634884862421418123812691, 798),
    (-2479392929313226753685415739663229, 870),
    (84483613348880041862046775994036021, 354),
];

/// The magnitudes of the numerators of `B_60, B_62, ..., B_100`, their limbs one after another,
/// the least significant first; the number of `B_(60 + 2i)` ends at [`LONG_END`]`[i]`. The signs
/// are those of every Bernoulli number: negative for `n = 0 mod 4`.
const LONG: [u64; 75] = [
    0x5047DAA8A7E64393,
    0x3FCD77896A343F43,
    0x0000000000000DF3,
    0xD9547361F5E55927,
    0x094101AF5D50D145,
    0x7F11711CAE210B81,
    0xCF399B169AABFC7C,
    0x0000000000000139,
    0x5FEB781D2A70B25D,
    0x847FD77286DFF384,
    0x000000000042089E,
    0xB76765ED9A363C79,
    0x6D0164E132C16CA9,
    0x0000000000038845,
    0x340AD867515A816B,
    0x0F1D5E8B97A3A66F,
    0x0000000107AFA25E,
    0xD17BE20C9315664D,
    0x12515C307E590205,
    0x003CD8C09A3E273F,
    0x22D78D2246E465AF,
    0x0323A2D34FC8423E,
    0x000000175E375B58,
    0xBCFBFEFECA7FFB09,
    0xD6CE058EF0E2F29A,
    0x000041E5B24381A3,
    0x774CC3B79EB45C9F,
    0xF685050B1191EA30,
    0x10EB32B9DC714E8B,
    0xB6FE7C5D41F609A3,
    0x0E0046E172E59E07,
    0x6CD6D086747768A8,
    0x00000000000002DD,
    0xA5075F246809594D,
    0x82F2C644816BFBB7,
    0x29ECAC4470D12BC3,
    0x000000000000010B,
    0xFE6D75CA93E13F3D,
    0x9F3DAC68E63AA1CD,
    0x1F2002A1046FF4C6,
    0x00000000133978DC,
    0x962077BFD79084B7,
    0x8AAE8D75791166F6,
    0xE95D0D91386304FB,
    0x0000000000019B29,
    0x2F2C2AA82F64BADF,
    0xD554D3D23FC9E873,
    0x242495C81AFBF274,
    0x00000030A4BD337B,
    0xB89A85CA61976D67,
    0xD852B7E8124D88BC,
    0x95346A5418CCE492,
    0x0000AAD5AA44E228,
    0x6CD99B8D53E67587,
    0xBE2C3FA060C5595B,
    0xA120545F9D05C713,
    0x0000BBB7F42443C5,
    0xD6EA05C62EBAB1E7,
    0xE74ACAEF0D671282,
    0xE7EC42F88F6ADC1F,
    0x0000B0E2803490F4,
    0xF4FF3683FCBE99BB,
    0x5DD07A328BA43B65,
    0x20D18760F53B1C62,
    0x6ACA47D00BE6048D,
    0x0000000000000723,
    0xED452F06F898C63F,
    0xC25CF05D9FA06D10,
    0x21C31E201EDD6FA9,
    0x9622B6A5B44A3F7A,
    0x2F4D0E7B433468DF,
    0xB5944A5E7E797ACD,
    0xE99995EDA6EBF5A3,
    0x997E8AEAE3D14E99,
    0x00000000000C7744,
];

/// Where each numerator of [`LONG`] ends.
const LONG_END: [u8; 21] = [
    3, 5, 8, 11, 14, 17, 20, 23, 26, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 75,
];

/// The denominators of `B_60, B_62, ..., B_100`.
const LONG_DEN: [u64; 21] = [
    56786730, 6, 510, 64722, 30, 4686, 140100870, 6, 30, 3318, 230010, 498, 3404310, 6, 61410,
    272118, 1410, 6, 4501770, 6, 33330,
];

/// The last index of the tables.
const TABLED: u64 = 100;

/// `B_n` for an even `n` up to [`TABLED`], from the tables, as numbers (in words they are taken
/// inline by [`bernoulli`]).
fn tabled(n: u64, order: ByteOrder) -> Rational {
    if n / 2 < TABLE.len() as u64 {
        let (a, b) = TABLE[(n / 2) as usize];
        return Rational::from_reduced(
            number_u128(a.unsigned_abs(), a < 0, order),
            number_u128(u128::from(b), false, order),
        );
    }
    let i = ((n - 2 * TABLE.len() as u64) / 2) as usize;
    let start = if i == 0 {
        0
    } else {
        usize::from(LONG_END[i - 1])
    };
    Rational::from_reduced(
        to_number(
            &LONG[start..usize::from(LONG_END[i])],
            n.is_multiple_of(4),
            order,
        ),
        number_u128(u128::from(LONG_DEN[i]), false, order),
    )
}

/// `B_n`, in the byte order `order`: those in words from the tables inline, in the caller, with
/// nothing to save or allocate, and the rest by [`beyond_words`].
#[inline]
pub(super) fn bernoulli(n: u64, order: ByteOrder) -> Rational {
    if n == 1 {
        return Rational::from_word(true, 1, 2, order);
    }
    if n & 1 == 1 {
        return Rational::from_word(false, 0, 1, order);
    }
    match word_of(n) {
        Some((negative, m, d)) => Rational::from_word(negative, m, d, order),
        None => beyond_words(n, order),
    }
}

/// The sign and the parts of an even `B_n` when it is in [`TABLE`] and they fit in 64 bits (all to
/// `B_34`, and `B_38`).
#[inline(always)]
fn word_of(n: u64) -> Option<(bool, u64, u64)> {
    let (a, b) = *TABLE.get(usize::try_from(n / 2).ok()?)?;
    let m = a.unsigned_abs();
    (m >> 64 == 0).then_some((a < 0, m as u64, b))
}

/// An even `B_n` that is not in words: from the tables up to [`TABLED`], and beyond by the zeta
/// function, out of line so that the lookups keep no registers for it.
#[inline(never)]
fn beyond_words(n: u64, order: ByteOrder) -> Rational {
    if n <= TABLED {
        return tabled(n, order);
    }
    let (num, den) = by_zeta(n);
    Rational::from_reduced(
        to_number(&num, n.is_multiple_of(4), order),
        to_number(&den, false, order),
    )
}

/// Up to this index, the table of Bernoulli numbers is computed from the tangent numbers; beyond,
/// by the zeta function from the top down. Measured on an Apple M4 Max, `B_0, ..., B_150` took
/// 142,000 cycles from the tangent numbers against 163,000, and `B_0, ..., B_200` 279,000 against
/// 267,000.
const TANGENT_TABLE: u64 = 150;

/// The denominator of `B_n` for an even `n >= 2`, by von Staudt and Clausen's theorem: the product
/// of the primes `p` with `p - 1` dividing `n`.
pub(super) fn denominator(n: u64) -> Limbs {
    let mut f: Vec<u64> = Vec::new();
    let mut packed = 1u64;
    each_divisor(n, |d| {
        if let Some(p) = d.checked_add(1)
            && is_prime_u64(p)
        {
            match packed.checked_mul(p) {
                Some(x) => packed = x,
                None => {
                    f.push(packed);
                    packed = p;
                }
            }
        }
    });
    f.push(packed);
    product(&f)
}

/// Calls `f` with each divisor of `n > 0`, in no particular order.
fn each_divisor(n: u64, mut f: impl FnMut(u64)) {
    // The prime factors by trial division; `n` stays small for any Bernoulli number that fits in
    // memory.
    let mut factors: Vec<(u64, u32)> = Vec::new();
    let mut m = n;
    let mut d = 2u64;
    while d * d <= m {
        if m.is_multiple_of(d) {
            let mut e = 0;
            while m.is_multiple_of(d) {
                m /= d;
                e += 1;
            }
            factors.push((d, e));
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if m > 1 {
        factors.push((m, 1));
    }
    let mut divisors = vec![1u64];
    for &(p, e) in &factors {
        let len = divisors.len();
        let mut q = 1;
        for _ in 0..e {
            q *= p;
            for i in 0..len {
                divisors.push(divisors[i] * q);
            }
        }
    }
    for d in divisors {
        f(d);
    }
}

/// A positive floating number `m 2^e`.
#[derive(Clone)]
struct Float {
    m: Limbs,
    e: i64,
}

impl Float {
    /// The number cut to its `w` leading bits, rounded down: within `2^(1 - w)` of it relatively.
    fn cut(mut self, w: u64) -> Float {
        self.cut_in(w);
        self
    }

    /// [`cut`](Self::cut) in place.
    fn cut_in(&mut self, w: u64) {
        let b = bits(&self.m);
        if b > w {
            shr_assign(&mut self.m, b - w);
            self.e += (b - w) as i64;
        }
    }

    /// `x <- x y`, cut to `w` bits, the product in the working memory.
    fn mul_in(&mut self, y: &Float, w: u64, work: &mut Work) {
        mul_into(&mut work.t, &self.m, &y.m, &mut work.s);
        normalize(&mut work.t);
        core::mem::swap(&mut self.m, &mut work.t);
        self.e += y.e;
        self.cut_in(w);
    }

    /// `x <- x^2`, cut to `w` bits, the square in the working memory.
    fn sqr_in(&mut self, w: u64, work: &mut Work) {
        sqr_into(&mut work.t, &self.m, &mut work.s);
        normalize(&mut work.t);
        core::mem::swap(&mut self.m, &mut work.t);
        self.e *= 2;
        self.cut_in(w);
    }
}

/// The vectors that the products and quotients of one computation reuse, so that a short one
/// allocates a few times rather than at every step.
#[derive(Default)]
struct Work {
    /// A product, swapped with the operand it replaces.
    t: Limbs,
    /// The scratch of the longer products.
    s: Limbs,
    /// A shifted operand of a quotient.
    a: Limbs,
    /// The normalised dividend and divisor of a quotient.
    n: Limbs,
    d: Limbs,
}

/// `x^n` from the top bit of `n`, each square and product cut to `w` bits: within
/// `4 n 2^(1 - w)` relatively, to the first order, of `x'^n` for the `x'` given.
fn power(x: &Float, n: u64, w: u64, work: &mut Work) -> Float {
    let mut acc = Float {
        m: Vec::with_capacity(2 * w.div_ceil(64) as usize + 2),
        e: 0,
    };
    power_into(&mut acc, x, n, w, work);
    acc
}

/// [`power`] into `acc`, whose vector is reused.
fn power_into(acc: &mut Float, x: &Float, n: u64, w: u64, work: &mut Work) {
    let small = x.e == 0 && x.m.len() == 1 && bits(&x.m) <= w;
    acc.m.clear();
    acc.m.extend_from_slice(&x.m);
    acc.e = x.e;
    acc.cut_in(w);
    for i in (0..63 - n.leading_zeros()).rev() {
        acc.sqr_in(w, work);
        if n >> i & 1 == 1 {
            if small {
                mul_1(&mut acc.m, x.m[0]);
                acc.cut_in(w);
            } else {
                acc.mul_in(x, w, work);
            }
        }
    }
}

/// Up to this many limbs in the divisor, a quotient is taken by the schoolbook method in the
/// working memory; beyond, on numbers, whose division is faster from there. Measured on an Apple
/// M4 Max, `B_300` took 18,400 cycles with 8, 19,400 with 16 and 20,800 with 32 (4 as 8), `B_600`
/// 56,600, 62,700 and 65,800, and `B_1000` 157,000, 156,000 and 167,000.
const SCHOOL_DIVISION: usize = 8;

/// `q = floor(a 2^s / b)` for magnitudes, `b` not zero and `s` of either sign (`b 2^-s` when
/// negative).
fn quotient_into(q: &mut Limbs, a: &[u64], s: i64, b: &[u64], work: &mut Work) {
    let (num, den): (&[u64], &[u64]) = if s >= 0 {
        shl_into(&mut work.a, a, s as u64);
        (&work.a, b)
    } else {
        shl_into(&mut work.a, b, s.unsigned_abs());
        (a, &work.a)
    };
    if den.len() > SCHOOL_DIVISION {
        let o = ByteOrder::LittleEndian;
        *q = of_number(&(&to_number(num, false, o) / &to_number(den, false, o)));
        return;
    }
    divide_into(q, num, den, &mut work.n, &mut work.d);
}

/// `q = floor(a / b)` by Knuth's algorithm D (volume 2, section 4.3.1), `b` normalised in `nd` and
/// `a` shifted as much in `nn`.
fn divide_into(q: &mut Limbs, a: &[u64], b: &[u64], nn: &mut Limbs, nd: &mut Limbs) {
    q.clear();
    let m = b.len();
    if a.len() < m {
        return;
    }
    if m == 1 {
        q.extend_from_slice(a);
        super::limbs::div_1(q, b[0]);
        return;
    }
    let sh = b[m - 1].leading_zeros();
    shl_into(nd, b, u64::from(sh));
    nd.truncate(m);
    shl_into(nn, a, u64::from(sh));
    nn.resize(a.len() + 1, 0);
    let len = a.len() - m + 1;
    q.resize(len, 0);
    let (top, next) = (u128::from(nd[m - 1]), u128::from(nd[m - 2]));
    let v = invert_limb(nd[m - 1]);
    for j in (0..len).rev() {
        // The estimate from the top two limbs, by the reciprocal of the divisor's top limb.
        let (u1, u0) = (nn[j + m], nn[j + m - 1]);
        let (mut qhat, mut rhat) = if u1 >= nd[m - 1] {
            (u128::from(u64::MAX), u128::from(u0) + top)
        } else {
            let (q, r) = div_2by1(u1, u0, nd[m - 1], v);
            (u128::from(q), u128::from(r))
        };
        while rhat >> 64 == 0 && qhat * next > ((rhat << 64) | u128::from(nn[j + m - 2])) {
            qhat -= 1;
            rhat += top;
        }
        // nn[j..=j + m] -= qhat nd.
        let (mut borrow, mut carry) = (0u64, 0u64);
        for i in 0..m {
            let p = qhat * u128::from(nd[i]) + u128::from(carry);
            carry = (p >> 64) as u64;
            let (d1, b1) = nn[j + i].overflowing_sub(p as u64);
            let (d2, b2) = d1.overflowing_sub(borrow);
            nn[j + i] = d2;
            borrow = u64::from(b1 | b2);
        }
        let (d1, b1) = nn[j + m].overflowing_sub(carry);
        let (d2, b2) = d1.overflowing_sub(borrow);
        nn[j + m] = d2;
        if b1 | b2 {
            // One too many: add the divisor back.
            qhat -= 1;
            let mut c = false;
            for i in 0..m {
                let (s1, c1) = nn[j + i].overflowing_add(nd[i]);
                let (s2, c2) = s1.overflowing_add(u64::from(c));
                nn[j + i] = s2;
                c = c1 | c2;
            }
            nn[j + m] = nn[j + m].wrapping_add(u64::from(c));
        }
        q[j] = qhat as u64;
    }
    normalize(q);
}

/// `log2(2π)`, rounded up.
const LOG2_TWO_PI: f64 = 2.651_496_129_472_319;

/// The bits of `n`.
fn bit_length(n: u64) -> u64 {
    u64::from(64 - n.leading_zeros())
}

/// `floor(π 2^f)` less at most 4: up to 16,384 bits cut from the table of the partition numbers,
/// beyond from their Chudnovsky series.
fn pi_fixed(f: u64) -> Limbs {
    let table = 64 * PI_TABLE.len() as u64;
    if f <= table {
        let mut x = shr(PI_TABLE, table - f);
        super::limbs::add(&mut x, &shl(&[3], f));
        x
    } else {
        of_number(&pi(f))
    }
}

/// `(|B_n| d, d)` for an even `n` above the table, `d` the denominator, from
/// `|B_n| = 2 n! ζ(n) / (2π)^n`: `|B_n| d` computed within less than 1/2 and rounded, or, for long
/// numerators, its low bits modulo primes by Harvey's method ([`modp`](super::modp)) and the rest
/// from the formula, to as many bits fewer.
fn by_zeta(n: u64) -> (Limbs, Limbs) {
    let o = ByteOrder::LittleEndian;
    let f = of_number(&Number::factorial(n, o));
    let den = denominator(n);
    let k = shl(&mul(&f, &den), 1);
    drop(f);
    // log2(|B_n| d) < bits(K) + log2 ζ(2) - n log2(2π), log2 ζ(2) < 0.73.
    let u = (bits(&k) as f64 + 1.75 - n as f64 * LOG2_TWO_PI)
        .ceil()
        .max(1.0) as u64
        + 1;
    let share = modular_bits(n, u);
    if share == 0 {
        let mut q = approximation(n, k, u + 8);
        super::limbs::add(&mut q, &[8]);
        return (shr(&q, 4), den);
    }
    // |B_n| d modulo the primes, joined: `R = |B_n| d mod M`.
    let (r, m) = modular(n, share, &den);
    // |B_n| d within M / 2^11, and the multiple of M nearest to its difference from R.
    let w = u.saturating_sub(bits(&m)) + 12;
    let q = approximation(n, k, w);
    let a = to_number(&shr(&q, 4), false, o);
    let (rn, mn) = (to_number(&r, false, o), to_number(&m, false, o));
    let t = (&(&a - &rn) + &(&mn >> 1usize)).div_floor(&mn);
    let x = &rn + &(&t * &mn);
    (of_number(&x), den)
}

/// The bits of `|B_n| d`, of `u` at most, taken modulo primes: none below [`MODULAR`], then a share
/// `α = 0.1 + 0.05 log2(n / 10^4)` that grows with `n` up to 0.22. Measured on an Apple M4 Max:
/// `B_5000` took 0.98 ms with none and 1.03 with 5 %; `B_10^4` 3.86 ms with none, 3.55 with 10 % and
/// 3.59 with 20 %; `B_(2 10^4)` 16.0 ms, 11.3, 10.4 with 15 % and 10.4; `B_(4 10^4)` 56.6, 36.0, 32.0
/// and 31.0 with 20 %; `B_10^5` 362 ms with none and 172, 157, 160 and 172 with 15, 20, 25 and 30 %;
/// `B_10^6` 9.24 s, 8.02, 8.09 and 9.13 with the same.
fn modular_bits(n: u64, u: u64) -> u64 {
    if n < MODULAR {
        return 0;
    }
    let alpha = (0.1 + 0.05 * (n as f64 / 1e4).log2()).clamp(0.0, 0.22);
    (u as f64 * alpha) as u64
}

/// The index from which the low bits of a Bernoulli number's numerator are taken modulo primes.
const MODULAR: u64 = 8000;

/// `(|B_n| d mod M, M)` for `M` the product of primes from 5 up, of at least `want` bits, those that
/// divide `d` or for which Harvey's method does not apply left out.
fn modular(n: u64, want: u64, den: &[u64]) -> (Limbs, Limbs) {
    let o = ByteOrder::LittleEndian;
    // The primes up to a bound that their bits reach (the product of the primes up to x has about
    // x / ln 2 bits), and more if needed.
    let mut limit = ((want as f64) * core::f64::consts::LN_2 * 1.1) as u64 + 1000;
    loop {
        let primes = odd_primes(limit);
        let root = (limit as f64).sqrt() as u64 + 2;
        let small: Vec<u64> = primes.iter().copied().take_while(|&q| q <= root).collect();
        let mut congruences: Vec<(Number, Number)> = Vec::new();
        let mut got = 0.0f64;
        let negative = n.is_multiple_of(4);
        for &p in primes.iter().filter(|&&p| p >= 5) {
            if let Some(b) = super::modp::bernoulli_mod_p(n, p, &small) {
                // |B_n| d mod p.
                let b = if negative { (p - b) % p } else { b };
                let r = (u128::from(b) * u128::from(super::limbs::rem_1(den, p)) % u128::from(p))
                    as u64;
                congruences.push((Number::new_from_u64(r, o), Number::new_from_u64(p, o)));
                got += (p as f64).log2();
                if got >= want as f64 {
                    let (x, m) = Number::crt(&congruences).expect("the primes are coprime");
                    return (of_number(&x), of_number(&m));
                }
            }
        }
        limit *= 2;
    }
}

/// `|B_n| d 2^4`, for `K = 2 d n!`, within `2^(u - w + 5)` units for `u` a bound of the bits of
/// `|B_n| d`, from `|B_n| d = K ζ(n) / (2π)^n` computed to `w` bits relatively: `(2π)^n` by powers
/// cut to `w + log2 n + 6` bits from `π` to as many, and `1/ζ(n) = Π_p (1 - p^-n)` in fixed point
/// with `w + log2 n + 4` bits, each `p^-n` to the bits it moves the product by. The errors add up
/// to less than `2^-w` relatively; the quotient is taken with 4 bits below the point.
fn approximation(n: u64, k: Limbs, w: u64) -> Limbs {
    let nl = bit_length(n);
    let mut work = Work::default();
    // (2π)^n.
    let wr = w + nl + 6;
    let two_pi = Float {
        m: pi_fixed(wr),
        e: 1 - wr as i64,
    };
    let r = power(&two_pi, n, wr, &mut work);
    // 1/ζ(n) 2^wz.
    let wz = w + nl + 4;
    let z = inverse_zeta(n, wz, &mut work);
    // (2π)^n / ζ(n), as `dm 2^de`.
    let mut dm = Float {
        m: z,
        e: -(wz as i64),
    };
    dm.mul_in(&r, w + 8, &mut work);
    let km = Float { m: k, e: 0 }.cut(w + 8);
    // |B_n| d 2^4 = (km / dm) 2^(ke - de + 4).
    let mut q = Vec::new();
    quotient_into(&mut q, &km.m, km.e - dm.e + 4, &dm.m, &mut work);
    q
}

/// `floor(2^w / ζ(n))` within a few units for every prime it takes, from the Euler product
/// `1/ζ(n) = Π_p (1 - p^-n)` over the primes whose `2^w p^-n` is at least 1: `z <- z - z / p^n`,
/// `z` and `p^n` cut to the bits of `2^w p^-n` and `log2 n + 8` more, one division.
fn inverse_zeta(n: u64, w: u64, work: &mut Work) -> Limbs {
    let nl = bit_length(n);
    // 2^w (1 - 2^-n).
    let mut z = shl(&[1], w);
    if n <= w {
        sub(&mut z, &shl(&[1], w - n));
    }
    let mut pn = Float {
        m: Vec::with_capacity(z.len() + 2),
        e: 0,
    };
    let (mut zt, mut delta) = (Vec::with_capacity(z.len()), Vec::with_capacity(z.len()));
    // The primes with `p^n <= 2^(w + 1)`.
    let limit = ((w + 1) as f64 / n as f64).exp2() as u64 + 2;
    for p in primes(3..=limit) {
        let lg = n as f64 * (p as f64).log2();
        if lg > w as f64 + 1.0 {
            break;
        }
        let t = (w as f64 - lg).floor().max(0.0) as u64 + 2;
        let prec = t + nl + 8;
        // z / p^n = (zt 2^(w - prec)) / (m 2^e).
        let cut = w.saturating_sub(prec);
        shr_into(&mut zt, &z, cut);
        power_into(&mut pn, &Float { m: vec![p], e: 0 }, n, prec, work);
        quotient_into(&mut delta, &zt, cut as i64 - pn.e, &pn.m, work);
        if delta.is_empty() {
            break;
        }
        sub(&mut z, &delta);
    }
    z
}

/// `B_0, ..., B_n`, in the byte order `order`.
pub(super) fn bernoulli_numbers(n: u64, order: ByteOrder) -> Vec<Rational> {
    let len = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_add(1))
        .expect("capacity overflow");
    let mut v: Vec<Rational> = Vec::with_capacity(len);
    // The tables', words written in place, and zeros where the others go.
    for k in 0..=n.min(TABLED) {
        v.push(bernoulli(k, order));
    }
    for _ in TABLED.min(n)..n {
        v.push(Rational::from_word(false, 0, 1, order));
    }
    let top = n & !1;
    let low = TABLED + 2;
    if top >= low && top <= TANGENT_TABLE {
        let t = tangent_numbers((top / 2) as usize);
        for k in low / 2..=top / 2 {
            let (num, den) = from_tangent(k, &t[k as usize]);
            v[2 * k as usize] = Rational::from_reduced(
                to_number(&num, k.is_multiple_of(2), order),
                to_number(&den, false, order),
            );
        }
    } else if top >= low {
        table_by_zeta(top, low, |k, num, den| {
            v[k as usize] = Rational::from_reduced(
                to_number(&num, k.is_multiple_of(4), order),
                to_number(&den, false, order),
            );
        });
    }
    v
}

/// `log2 x` for a magnitude `x` of at least one limb, from its top 64 bits.
fn log2_of(x: &[u64]) -> f64 {
    let b = bits(x);
    let top = shr(x, b.saturating_sub(64));
    let t = top.first().copied().unwrap_or(0) as f64;
    t.log2() + b.saturating_sub(64) as f64
}

/// `(|B_k| d_k, d_k)` for every even `k` from `top` down to `low`, handed to `put`, by the formula of
/// [`by_zeta`] run from the top index down, as FLINT's `bernoulli_rev` does: `G_k = 2 k! / (2π)^k`
/// from the last by a product with `(2π)^2` and a division by `k (k - 1)`, and
/// `ζ(k) = (1 - 2^-k)^-1 Σ_(j odd) j^-k` from the powers `j^-k` in fixed point, each from the last by
/// a product with the limb `j^2`. Both are kept to the bits of `|B_k| d_k` and `log2 top + 8` more,
/// which fall as `k` falls, the powers to whole limbs and a limb more, and the bounds of their errors
/// in units of their last bits; a number whose bound would not leave less than 1/4 is computed
/// alone, which the bounds do not let happen.
fn table_by_zeta(top: u64, low: u64, mut put: impl FnMut(u64, Limbs, Limbs)) {
    let nl = bit_length(top);
    let g = nl + 8;
    let fact = of_number(&Number::factorial(top, ByteOrder::LittleEndian));
    let mut lf = log2_of(&fact);
    // The bits of `|B_k| d_k`, bounded from `log2(2 d_k k!) + log2 ζ(2) - k log2(2π)` with 3 bits
    // for the rounding of the doubles.
    let bound = |k: u64, lf: f64, ld: f64| -> u64 {
        (1.0 + ld + lf + 0.75 - k as f64 * LOG2_TWO_PI)
            .ceil()
            .max(1.0) as u64
            + 3
    };
    // The denominators, and the largest of their bits up to each index, which keeps the precisions
    // falling with `k`.
    let mut dens: Vec<Limbs> = (low / 2..=top / 2).map(|i| denominator(2 * i)).collect();
    let mut most: Vec<f64> = Vec::with_capacity(dens.len());
    let mut m = 0.0f64;
    for d in &dens {
        m = m.max(bits(d) as f64);
        most.push(m);
    }
    let at = |k: u64| ((k - low) / 2) as usize;
    let mut u = bound(top, lf, most[at(top)]);
    // G to `p` bits; the powers in fixed point with `64 l` bits, `l` limbs.
    let mut p = u + 8 + g;
    let mut l = (p + 64).div_ceil(64) as usize;
    let wr = p + nl + 6;
    let two_pi = Float {
        m: pi_fixed(wr),
        e: 1 - wr as i64,
    };
    let mut work = Work::default();
    let mut four_pi2 = two_pi.clone();
    four_pi2.sqr_in(wr, &mut work);
    let r = power(&two_pi, top, wr, &mut work);
    let km = Float {
        m: shl(&fact, 1),
        e: 0,
    }
    .cut(p + 8);
    drop(fact);
    let s = km.e - r.e + p as i64 + 8 - bits(&km.m) as i64 + bits(&r.m) as i64;
    let mut gm = Float {
        m: {
            let mut q = Vec::new();
            quotient_into(&mut q, &km.m, s, &r.m, &mut work);
            q
        },
        e: km.e - r.e - s,
    }
    .cut(p);
    // The bound of G's error, in units of `2^-p` relatively.
    let mut err_g = 8.0f64;
    // The odd powers: `ys[i] = floor(2^(64 l) (2i + 3)^-k)`, the bounds of their errors in those
    // units, and their sum with 1.
    let mut ys: Vec<Limbs> = Vec::new();
    let mut err_y: Vec<f64> = Vec::new();
    let mut sum: Limbs = vec![0; l + 2];
    sum[l] = 1;
    let mut k = top;
    loop {
        let units = 64 * l as u64;
        // More powers while the tail beyond them could reach half a unit.
        let tail_of = |j0: u64, k: u64| {
            units as f64 - k as f64 * (j0 as f64).log2()
                + (1.0 + j0 as f64 / (2.0 * (k - 1) as f64)).log2()
        };
        while tail_of(2 * ys.len() as u64 + 3, k) >= -1.0 {
            let y = power_reciprocal(2 * ys.len() as u64 + 3, k, units, nl, &mut work);
            add_into(&mut sum, &y);
            ys.push(y);
            err_y.push(2.0);
        }
        let tail = tail_of(2 * ys.len() as u64 + 3, k).exp2();
        // ζ(k) 2^units = sum (1 + 2^-k + 2^-2k + ...).
        let mut zeta = sum.clone();
        normalize(&mut zeta);
        let mut i = k;
        while i <= units {
            add_into(&mut zeta, &shr(&sum, i));
            i += k;
        }
        normalize(&mut zeta);
        let err_z = err_y.iter().sum::<f64>() + tail + (units / k + 1) as f64;
        // |B_k| d_k 2^4 = d_k G ζ 2^(e + 4 - units).
        let d = core::mem::take(&mut dens[at(k)]);
        let x = mul(&mul(&gm.m, &zeta), &d);
        let sh = gm.e - units as i64 + 4;
        let mut q = if sh >= 0 {
            shl(&x, sh as u64)
        } else {
            shr(&x, sh.unsigned_abs())
        };
        // The error bound, in units of the result: `2^u (err_g 2^-p + err_z 2^-units)`, and the
        // floors.
        let err = (err_g * (u as f64 - p as f64).exp2()) + err_z * (u as f64 - units as f64).exp2();
        if err + 1.0 / 8.0 < 0.25 {
            super::limbs::add(&mut q, &[8]);
            put(k, shr(&q, 4), d);
        } else {
            let (num, den) = by_zeta(k);
            put(k, num, den);
        }
        if k <= low {
            break;
        }
        // Down to k - 2.
        lf -= ((k * (k - 1)) as f64).log2();
        let k2 = k - 2;
        u = bound(k2, lf, most[at(k2)]);
        let p2 = u + 8 + g;
        let l2 = (p2 + 64).div_ceil(64) as usize;
        let dl = l - l2;
        // G_(k-2) = G_k (2π)^2 / (k (k - 1)), `(2π)^2` cut to the limbs that count.
        let fp = four_pi2
            .m
            .len()
            .saturating_sub(p2.div_ceil(64) as usize + 2);
        let mut m = mul(&gm.m, &four_pi2.m[fp..]);
        let e = gm.e + four_pi2.e + 64 * fp as i64;
        let b = bits(&m);
        let c = b.saturating_sub(p2 + 64);
        m = shr(&m, c);
        super::limbs::div_1(&mut m, k * (k - 1));
        gm = Float { m, e: e + c as i64 }.cut(p2);
        // Errors in units of 2^-p2: the old ones shrink by 2^-(p - p2), `(2π)^2` and its cut add
        // theirs (within `6 2^-wr` and `2^(1 - 64 (len - fp) + 64)`), the cut and the division a
        // unit each.
        let kept = 64.0 * (four_pi2.m.len() - fp) as f64 - 64.0;
        err_g = err_g * (p2 as f64 - p as f64).exp2()
            + 6.0 * (p2 as f64 - wr as f64).exp2()
            + 2.0 * (p2 as f64 - kept).exp2()
            + 3.0;
        // y_j (k - 2) = y_j(k) j^2 B^-dl, summed as they are made.
        sum.clear();
        sum.resize(l2 + 2, 0);
        sum[l2] = 1;
        for (i, y) in ys.iter_mut().enumerate() {
            let j = 2 * i as u64 + 3;
            scale_into(y, j * j, dl, &mut sum);
            err_y[i] = err_y[i] * (j * j) as f64 * (-64.0 * dl as f64).exp2() + 1.0;
        }
        while ys.last().is_some_and(|y| y.is_empty()) {
            ys.pop();
            err_y.pop();
        }
        p = p2;
        l = l2;
        k = k2;
    }
}

/// `s + y` in place, `s` long enough for the sum.
#[inline]
fn add_into(s: &mut Limbs, y: &[u64]) {
    if s.len() <= y.len() {
        s.resize(y.len() + 1, 0);
    }
    let mut carry = 0u64;
    let (low, high) = s.split_at_mut(y.len());
    for (a, &b) in low.iter_mut().zip(y) {
        let (t, c1) = a.overflowing_add(b);
        let (t, c2) = t.overflowing_add(carry);
        *a = t;
        carry = u64::from(c1 | c2);
    }
    for a in high.iter_mut() {
        if carry == 0 {
            break;
        }
        let (t, c) = a.overflowing_add(carry);
        *a = t;
        carry = u64::from(c);
    }
}

/// `y <- floor(y c / B^drop)` and `s += y`, in one pass over the limbs of `y`; `s` long enough.
#[inline]
fn scale_into(y: &mut Limbs, c: u64, drop: usize, s: &mut Limbs) {
    let len = y.len();
    if drop > len {
        y.clear();
        return;
    }
    let mut carry = 0u64;
    // The product's limbs below `drop` only carry.
    for &x in &y[..drop.min(len)] {
        let t = u128::from(x) * u128::from(c) + u128::from(carry);
        carry = (t >> 64) as u64;
    }
    let mut sc = false;
    let mut out = 0;
    for i in drop..len {
        let t = u128::from(y[i]) * u128::from(c) + u128::from(carry);
        let v = t as u64;
        carry = (t >> 64) as u64;
        y[out] = v;
        let (a, c1) = s[out].overflowing_add(v);
        let (a, c2) = a.overflowing_add(u64::from(sc));
        s[out] = a;
        sc = c1 | c2;
        out += 1;
    }
    if carry != 0 {
        if out < len {
            y[out] = carry;
        } else {
            y.push(carry);
        }
        let (a, c1) = s[out].overflowing_add(carry);
        let (a, c2) = a.overflowing_add(u64::from(sc));
        s[out] = a;
        sc = c1 | c2;
        out += 1;
    }
    y.truncate(out);
    let mut i = out;
    while sc {
        let (a, c) = s[i].overflowing_add(1);
        s[i] = a;
        sc = c;
        i += 1;
    }
    normalize(y);
}

/// `floor(2^p j^-k)` within 2, by `j^k` cut to the bits of the result and `log2 k + 8` more, and a
/// division.
fn power_reciprocal(j: u64, k: u64, p: u64, nl: u64, work: &mut Work) -> Limbs {
    let lg = k as f64 * (j as f64).log2();
    let t = (p as f64 - lg).floor().max(0.0) as u64 + 2;
    let prec = t + nl + 8;
    let jk = power(&Float { m: vec![j], e: 0 }, k, prec, work);
    let s = p as i64 - jk.e;
    if s < 0 {
        return Vec::new();
    }
    let mut q = Vec::new();
    quotient_into(&mut q, &[1], s, &jk.m, work);
    q
}

/// The tangent numbers `T_1, ..., T_m` (`tan x = Σ T_k x^(2k-1) / (2k-1)!`), by Brent and Harvey's
/// recurrence ("Fast computation of Bernoulli, Tangent and Secant numbers", 2011): `T_k = (k-1)!`,
/// then for each `k` from 2, `T_j <- (j - k) T_(j-1) + (j - k + 2) T_j` for `j` from `k` up, in
/// place; `m^2 / 2` linear steps on numbers of up to `2m log2 m` bits.
fn tangent_numbers(m: usize) -> Vec<Limbs> {
    let mut t: Vec<Limbs> = Vec::with_capacity(m + 1);
    t.push(Vec::new());
    t.push(vec![1]);
    for k in 2..=m {
        let mut x = t[k - 1].clone();
        mul_1(&mut x, (k - 1) as u64);
        t.push(x);
    }
    for k in 2..=m {
        for j in k..=m {
            let (low, high) = t.split_at_mut(j);
            combine(
                &low[j - 1],
                (j - k) as u64,
                &mut high[0],
                (j - k + 2) as u64,
            );
        }
    }
    t
}

/// `y <- a x + b y` for limbs `a` and `b` below `2^32`.
#[inline]
fn combine(x: &[u64], a: u64, y: &mut Limbs, b: u64) {
    if y.len() < x.len() {
        y.resize(x.len(), 0);
    }
    let mut carry = 0u128;
    for (i, yi) in y.iter_mut().enumerate() {
        let xi = x.get(i).copied().unwrap_or(0);
        let t = u128::from(xi) * u128::from(a) + u128::from(*yi) * u128::from(b) + carry;
        *yi = t as u64;
        carry = t >> 64;
    }
    if carry != 0 {
        y.push(carry as u64);
    }
}

/// `(|B_2k| d_2k, d_2k)` from the tangent number `T_k`: `|B_2k| = 2k T_k / (4^k (4^k - 1))`.
fn from_tangent(k: u64, t: &[u64]) -> (Limbs, Limbs) {
    let d = denominator(2 * k);
    let mut x = mul(t, &d);
    mul_1(&mut x, 2 * k);
    let x = shr(&x, 2 * k);
    let mut q = shl(&[1], 2 * k);
    sub(&mut q, &[1]);
    let o = ByteOrder::LittleEndian;
    let n = to_number(&x, false, o).div_exact(&to_number(&q, false, o));
    (of_number(&n), d)
}
