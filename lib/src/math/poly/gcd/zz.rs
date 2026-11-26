//! Greatest common divisors and resultants over the integers.
//!
//! The coefficients are taken apart into their signs and limbs once ([`Int`]). The greatest common
//! divisor of two primitive polynomials is found by one of two methods:
//!
//! - the heuristic gcd (Char, Geddes and Gonnet): both polynomials evaluated at `ξ = 2^s`, the
//!   greatest common divisor of the two integers, its digits in base `ξ` read back, balanced, as a
//!   polynomial, and its primitive part kept when it divides both; for `ξ` above twice the smaller
//!   of the largest coefficients that makes it the greatest common divisor;
//! - the multimodular algorithm: the monic greatest common divisors modulo primes below `2^62`,
//!   scaled to the greatest common divisor of the leading coefficients, joined by the Chinese
//!   remainder theorem in the symmetric residues until they stop changing, the primitive part
//!   then kept when it divides both. A prime whose gcd has a higher degree is passed over; one of
//!   a lower degree starts the images again; one of degree 0 proves the polynomials coprime.
//!
//! Divisibility is decided by Kronecker substitution: the integer quotient of the values at `2^s`
//! read back as a polynomial and its coefficients bounded, so that its product by the divisor,
//! equal at `2^s` and of coefficients below `2^(s-1)`, is the dividend.
//!
//! The resultant is the multimodular one: modulo enough primes for Hadamard's bound, joined by
//! the Chinese remainder theorem.

use super::hgcd;
use super::modpoly::resultant_in;
use super::nmod::Word;
use crate::math::numbers::{ByteOrder, Number, from_limbs, is_prime_u64, limbs_of, nat};
use core::cmp::Ordering;
use std::sync::Mutex;

/// An integer as its sign and the limbs of its magnitude, without zeros at the top.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Int {
    /// Whether it is negative (never for 0).
    pub(crate) neg: bool,
    /// The magnitude.
    pub(crate) m: Vec<u64>,
}

/// `a` compared with `b`, both without zeros at the top.
fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        if x != y {
            return x.cmp(y);
        }
    }
    Ordering::Equal
}

/// Drops the zero limbs at the top.
#[inline]
fn trim(a: &mut Vec<u64>) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

/// `r += a` on magnitudes.
fn add_mag(r: &mut Vec<u64>, a: &[u64]) {
    if r.len() < a.len() {
        r.resize(a.len(), 0);
    }
    if nat::add_assign(r, a) {
        r.push(1);
    }
}

/// `r -= a` on magnitudes, `r ≥ a`.
fn sub_mag(r: &mut Vec<u64>, a: &[u64]) {
    let borrow = nat::sub_assign(r, a);
    debug_assert!(!borrow);
    trim(r);
}

impl Int {
    /// The integer `x`.
    pub(crate) fn of(x: &Number) -> Int {
        let m = limbs_of(x);
        Int {
            neg: x.is_negative() && !m.is_empty(),
            m,
        }
    }

    /// The number.
    pub(crate) fn number(&self) -> Number {
        let n = from_limbs(&self.m, ByteOrder::LittleEndian);
        if self.neg { -n } else { n }
    }

    /// Whether it is 0.
    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        self.m.is_empty()
    }

    /// The bits of the magnitude.
    #[inline]
    pub(crate) fn bits(&self) -> u64 {
        match self.m.last() {
            None => 0,
            Some(&t) => 64 * (self.m.len() as u64 - 1) + u64::from(64 - t.leading_zeros()),
        }
    }

    /// `self += sign·x` for the magnitude `x`, `sign` negative when `xneg`.
    fn add_signed(&mut self, x: &[u64], xneg: bool) {
        if x.is_empty() {
            return;
        }
        if self.m.is_empty() || self.neg == xneg {
            if self.m.is_empty() {
                self.neg = xneg;
            }
            add_mag(&mut self.m, x);
            return;
        }
        match cmp_mag(&self.m, x) {
            Ordering::Greater => sub_mag(&mut self.m, x),
            Ordering::Equal => {
                self.m.clear();
                self.neg = false;
            }
            Ordering::Less => {
                let mut r = x.to_vec();
                sub_mag(&mut r, &self.m);
                self.m = r;
                self.neg = xneg;
            }
        }
    }
}

/// `x·t` on magnitudes.
fn mul_1(x: &[u64], t: u64) -> Vec<u64> {
    let mut r = Vec::with_capacity(x.len() + 1);
    let mut carry = 0u64;
    for &l in x {
        let p = u128::from(l) * u128::from(t) + u128::from(carry);
        r.push(p as u64);
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        r.push(carry);
    }
    trim(&mut r);
    r
}

/// The polynomial's coefficients as [`Int`]s.
pub(crate) fn ints(c: &[Number]) -> Vec<Int> {
    c.iter().map(Int::of).collect()
}

/// The numbers of the [`Int`]s.
pub(crate) fn numbers(c: &[Int]) -> Vec<Number> {
    c.iter().map(Int::number).collect()
}

/// The largest number of bits of a coefficient.
fn max_bits(c: &[Int]) -> u64 {
    c.iter().map(Int::bits).max().unwrap_or(0)
}

/// The bits of `n`, 0 for 0.
#[inline]
fn bitlen(n: usize) -> u64 {
    u64::from(usize::BITS - n.leading_zeros())
}

// ----------------------------------------------------------------- residues modulo word primes

/// The primes below `2^62` found so far, from the top down, shared by all the calls.
static PRIMES: Mutex<Cache> = Mutex::new(Cache {
    primes: Vec::new(),
    next: 0,
});

/// The primes found so far, and the odd number the search goes on from (down), 0 before the first
/// window.
struct Cache {
    /// The primes, from the top down.
    primes: Vec<u64>,
    /// The largest odd number not yet searched; 0 before the first window.
    next: u64,
}

/// The odd numbers a window of the sieve covers (fewer under Miri, whose tests need few primes and
/// to which each test of a survivor is slow).
const WINDOW: usize = if cfg!(miri) { 1 << 9 } else { 1 << 14 };

/// Searches the next window of `WINDOW` odd numbers down from `c.next` (from `2^62 - 1` at first)
/// and appends its primes: the window sieved by the odd primes below 2^10, the survivors tested
/// by [`is_prime_u64`]. The search always moves on, whether or not the window holds a prime.
fn extend(c: &mut Cache) {
    const SMALL: [u64; 172] = small_primes();
    let hi = if c.next == 0 {
        (1u64 << 62) - 1
    } else {
        c.next
    };
    debug_assert!(hi & 1 == 1);
    // The candidates hi - 2i, i < WINDOW.
    let mut comp = vec![false; WINDOW];
    for &q in &SMALL[1..] {
        // hi - 2i ≡ 0 (mod q) for i ≡ hi·2⁻¹ (mod q), 2⁻¹ = (q + 1)/2.
        let mut i = ((hi % q) * q.div_ceil(2) % q) as usize;
        while i < WINDOW {
            comp[i] = true;
            i += q as usize;
        }
    }
    for (i, &composite) in comp.iter().enumerate() {
        let x = hi - 2 * i as u64;
        if !composite && is_prime_u64(x) {
            c.primes.push(x);
        }
    }
    c.next = hi - 2 * WINDOW as u64;
}

/// The primes below 2^10.
const fn small_primes() -> [u64; 172] {
    let mut out = [0u64; 172];
    let mut k = 0;
    let mut x = 2u64;
    while x < 1024 {
        let mut d = 2u64;
        let mut prime = true;
        while d * d <= x {
            if x.is_multiple_of(d) {
                prime = false;
                break;
            }
            d += 1;
        }
        if prime {
            out[k] = x;
            k += 1;
        }
        x += 1;
    }
    out
}

/// The `i`-th prime below `2^62` from the top down.
fn prime(i: usize) -> u64 {
    let mut c = PRIMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while c.primes.len() <= i {
        extend(&mut c);
    }
    c.primes[i]
}

/// The primes below `2^62`, from the top down.
pub(crate) struct Primes(usize);

impl Primes {
    /// The primes from `2^62` down.
    pub(crate) fn new() -> Primes {
        Primes(0)
    }
}

impl Iterator for Primes {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        let p = prime(self.0);
        self.0 += 1;
        Some(p)
    }
}

/// Remainders by the prime `p`: the powers of `2^64` modulo `p`, for the sum of each limb times its
/// power, three words summed and reduced once.
pub(crate) struct Table {
    /// The modulus.
    pub(crate) m: Word,
    /// `2^(64 i) mod p`.
    pow: Vec<u64>,
}

impl Table {
    /// The table modulo `p` for integers of up to `limbs` limbs.
    pub(crate) fn new(p: u64, limbs: usize) -> Table {
        let m = Word::new(p);
        let b = m.m.red2(1, 0);
        let mut pow = Vec::with_capacity(limbs.max(1));
        let mut x = m.m.red1(1);
        for _ in 0..limbs.max(1) {
            pow.push(x);
            x = m.m.mul(x, b);
        }
        Table { m, pow }
    }

    /// Makes room for integers of `limbs` limbs.
    fn grow(&mut self, limbs: usize) {
        let b = self.m.m.red2(1, 0);
        while self.pow.len() < limbs {
            let x = self.m.m.mul(*self.pow.last().expect("not empty"), b);
            self.pow.push(x);
        }
    }

    /// The magnitude `x` modulo `p`.
    #[inline]
    fn mag(&self, x: &[u64]) -> u64 {
        match x.len() {
            0 => 0,
            1 => self.m.m.red1(x[0]),
            _ => {
                let mut lo: u128 = 0;
                let mut hi: u64 = 0;
                for (&l, &w) in x.iter().zip(&self.pow) {
                    let (s, c) = lo.overflowing_add(u128::from(l) * u128::from(w));
                    lo = s;
                    hi += u64::from(c);
                }
                self.m.m.red3(hi, (lo >> 64) as u64, lo as u64)
            }
        }
    }

    /// `x` modulo `p`, in `0..p`.
    #[inline]
    pub(crate) fn int(&self, x: &Int) -> u64 {
        let r = self.mag(&x.m);
        if x.neg { self.m.m.neg(r) } else { r }
    }

    /// The polynomial modulo `p`, trimmed.
    pub(crate) fn poly(&self, c: &[Int]) -> Vec<u64> {
        let mut r: Vec<u64> = c.iter().map(|x| self.int(x)).collect();
        super::nmod::trim(&mut r);
        r
    }
}

/// The residues `r` (modulo `p`) taken into the images `h` modulo `m`, in the symmetric residues:
/// `h + m·t` with `t = (r - h)/m mod p` in `(-p/2, p/2]`; whether any changed. `m` becomes `m·p`.
fn crt_step(h: &mut [Int], m: &mut Vec<u64>, r: &[u64], t: &mut Table) -> bool {
    t.grow(m.len() + 2);
    let w = &t.m.m;
    let p = w.n;
    let minv = w.inv_lead(t.mag(m));
    let mut changed = false;
    for (x, &ri) in h.iter_mut().zip(r) {
        let hx = t.int(x);
        let d = w.mul(w.sub(ri, hx), minv);
        if d == 0 {
            continue;
        }
        changed = true;
        let (mag, neg) = if d > p / 2 { (p - d, true) } else { (d, false) };
        x.add_signed(&mul_1(m, mag), neg);
    }
    *m = mul_1(m, p);
    changed
}

/// The symmetric residues of `r` modulo `p`.
fn symmetric(r: &[u64], p: u64) -> Vec<Int> {
    r.iter()
        .map(|&x| {
            if x == 0 {
                Int::default()
            } else if x > p / 2 {
                Int {
                    neg: true,
                    m: vec![p - x],
                }
            } else {
                Int {
                    neg: false,
                    m: vec![x],
                }
            }
        })
        .collect()
}

// ----------------------------------------------------------------- Kronecker substitution

/// ORs the magnitude `x` shifted by `shift` bits into `out`.
fn or_shifted(out: &mut [u64], x: &[u64], shift: usize) {
    let (l, s) = (shift / 64, (shift % 64) as u32);
    for (i, &v) in x.iter().enumerate() {
        out[l + i] |= v << s;
        if s != 0 && v >> (64 - s) != 0 {
            out[l + i + 1] |= v >> (64 - s);
        }
    }
}

/// The value at `2^s` of the polynomial `c`, of coefficients below `2^(s-1)` in magnitude.
pub(crate) fn pack(c: &[Int], s: usize) -> Int {
    let limbs = (c.len() * s).div_ceil(64) + 1;
    let mut pos = vec![0u64; limbs];
    let mut neg = vec![0u64; limbs];
    let mut any_neg = false;
    for (i, x) in c.iter().enumerate() {
        if x.neg {
            any_neg = true;
            or_shifted(&mut neg, &x.m, i * s);
        } else {
            or_shifted(&mut pos, &x.m, i * s);
        }
    }
    trim(&mut pos);
    trim(&mut neg);
    if !any_neg {
        return Int { neg: false, m: pos };
    }
    let mut r = Int { neg: false, m: pos };
    r.add_signed(&neg, true);
    r
}

/// The `s` bits of `x` from the bit `p`, without zeros at the top.
fn bits_at(x: &[u64], p: usize, s: usize) -> Vec<u64> {
    let n = s.div_ceil(64);
    let (l, sh) = (p / 64, (p % 64) as u32);
    let mut r = vec![0u64; n];
    for (i, o) in r.iter_mut().enumerate() {
        let lo = x.get(l + i).copied().unwrap_or(0);
        let hi = x.get(l + i + 1).copied().unwrap_or(0);
        *o = if sh == 0 {
            lo
        } else {
            (lo >> sh) | (hi << (64 - sh))
        };
    }
    if !s.is_multiple_of(64) {
        r[n - 1] &= (1u64 << (s % 64)) - 1;
    }
    trim(&mut r);
    r
}

/// The digits of `x` in base `2^s`, balanced in `(-2^(s-1), 2^(s-1)]`, the lowest first, without
/// zeros at the top: the polynomial of coefficients below `2^(s-1)` whose value at `2^s` is `x`.
pub(crate) fn unpack(x: &Int, s: usize) -> Vec<Int> {
    let mut out = Vec::new();
    if x.is_zero() {
        return out;
    }
    let total = x.bits() as usize;
    // 2^s and 2^(s-1) as magnitudes.
    let mut full = vec![0u64; s / 64 + 1];
    full[s / 64] = 1u64 << (s % 64);
    let mut half = vec![0u64; (s - 1) / 64 + 1];
    half[(s - 1) / 64] = 1u64 << ((s - 1) % 64);
    let mut carry = false;
    let mut p = 0;
    while p < total || carry {
        let mut v = bits_at(&x.m, p, s);
        if carry {
            add_mag(&mut v, &[1]);
        }
        let d = if cmp_mag(&v, &half) == Ordering::Greater {
            carry = true;
            let mut m = full.clone();
            sub_mag(&mut m, &v);
            Int {
                neg: !x.neg && !m.is_empty(),
                m,
            }
        } else {
            carry = false;
            Int {
                neg: x.neg && !v.is_empty(),
                m: v,
            }
        };
        out.push(d);
        p += s;
    }
    while out.last().is_some_and(Int::is_zero) {
        out.pop();
    }
    out
}

/// `(q, r)` with `a = q·b + r` truncated, as numbers.
fn div_rem(a: &Int, b: &Int) -> (Int, Int) {
    let (q, r) = a.number().div_rem(&b.number());
    (Int::of(&q), Int::of(&r))
}

/// The quotient `a / g` when `g` divides `a` exactly, `None` otherwise, for `g` not zero: by
/// Kronecker substitution at `2^s`, the quotient's coefficients bounded so that the identity of
/// the values is that of the polynomials; first at the size of `a`'s coefficients, then, if the
/// quotient does not fit, at Mignotte's bound on the coefficients of a factor of `a`.
pub(crate) fn divides(a: &[Int], g: &[Int]) -> Option<Vec<Int>> {
    if a.is_empty() {
        return Some(Vec::new());
    }
    if a.len() < g.len() {
        return None;
    }
    let (la, lg) = (a.len(), g.len());
    if lg == 1 {
        let mut q = Vec::with_capacity(la);
        for x in a {
            let (qq, r) = div_rem(x, &g[0]);
            if !r.is_zero() {
                return None;
            }
            q.push(qq);
        }
        return Some(q);
    }
    // Quick refutations: the leading and the constant coefficients.
    if !div_rem(&a[la - 1], &g[lg - 1]).1.is_zero() {
        return None;
    }
    if !g[0].is_zero() && !div_rem(&a[0], &g[0]).1.is_zero() {
        return None;
    }
    let (ba, bg) = (max_bits(a), max_bits(g));
    let log = bitlen(la);
    let first = (ba + log + 8) as usize;
    // Mignotte: a factor q of a has |q|∞ ≤ 2^deg(q) |a|₂.
    let mignotte = (ba + (la - lg + 1) as u64 + log + 1) as usize;
    let last = mignotte + (bg + log + 3) as usize;
    for s in [first, last] {
        let s = s.max((bg + 2) as usize).max((ba + 2) as usize);
        let x = pack(a, s);
        let y = pack(g, s);
        let (q, r) = div_rem(&x, &y);
        if !r.is_zero() {
            return None;
        }
        let q = unpack(&q, s);
        if q.len() != la - lg + 1 {
            if s == last {
                return None;
            }
            continue;
        }
        let bq = max_bits(&q);
        if bq + bg + bitlen(lg.min(q.len())) + 1 < s as u64 {
            return Some(q);
        }
        if s == last {
            return None;
        }
    }
    None
}

// ----------------------------------------------------------------- greatest common divisors

/// The binary gcd of two words, `gcd(0, b) = b`.
fn gcd_word(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

/// The content (the positive gcd of the coefficients) and the primitive part of a positive leading
/// coefficient; for coefficients of one word, by the binary gcd and divisions of words.
fn content_pp_ints(mut v: Vec<Int>) -> (Int, Vec<Int>) {
    let lneg = v.last().is_some_and(|x| x.neg);
    let one = Int {
        neg: false,
        m: vec![1],
    };
    let flip = |v: &mut Vec<Int>| {
        if lneg {
            for x in v.iter_mut() {
                x.neg = !x.neg && !x.is_zero();
            }
        }
    };
    if v.iter().all(|x| x.m.len() <= 1) {
        let mut g = 0u64;
        for x in &v {
            g = gcd_word(g, x.m.first().copied().unwrap_or(0));
            if g == 1 {
                break;
            }
        }
        if g > 1 {
            for x in &mut v {
                if let Some(t) = x.m.first_mut() {
                    *t /= g;
                }
            }
        }
        flip(&mut v);
        let g = if g == 0 {
            Int::default()
        } else {
            Int {
                neg: false,
                m: vec![g],
            }
        };
        return (g, v);
    }
    let mut g = Number::default();
    for x in &v {
        g = g.gcd(&x.number());
        if crate::math::rational::is_one(&g) {
            break;
        }
    }
    if crate::math::rational::is_one(&g) {
        flip(&mut v);
        return (one, v);
    }
    let gs = if lneg { -&g } else { g.clone() };
    let v = v.iter().map(|x| Int::of(&(&x.number() / &gs))).collect();
    (Int::of(&g), v)
}

/// The primitive part of `c` not empty, with a positive leading coefficient.
fn primitive(c: Vec<Int>) -> Vec<Int> {
    content_pp_ints(c).1
}

/// The heuristic gcd of the primitive `a` and `b`, `len a ≥ len b ≥ 2`: `None` when it gives up.
pub(crate) fn heuristic(a: &[Int], b: &[Int]) -> Option<Vec<Int>> {
    let (ba, bb) = (max_bits(a), max_bits(b));
    let log = bitlen(a.len());
    let mut s = (ba.max(bb) + log + 6) as usize;
    for _ in 0..3 {
        let x = pack(a, s).number();
        let y = pack(b, s).number();
        let gam = Int::of(&x.gcd(&y));
        let g = unpack(&gam, s);
        if g.is_empty() {
            return None;
        }
        let g = primitive(g);
        if g.len() == 1 {
            // A constant divides both; the gcd is 1 (the theorem's bound holds for this s).
            return Some(g);
        }
        if g.len() <= b.len() && check(a, &g, s) && check(b, &g, s) {
            return Some(g);
        }
        s = s * 3 / 2 + 8;
    }
    None
}

/// Whether `g` divides `a` by the integers at `2^s` and the bound on the quotient read back.
fn check(a: &[Int], g: &[Int], s: usize) -> bool {
    if s < (max_bits(a) + 2) as usize {
        return false;
    }
    let (q, r) = div_rem(&pack(a, s), &pack(g, s));
    if !r.is_zero() {
        return false;
    }
    let q = unpack(&q, s);
    if q.len() + g.len() != a.len() + 1 {
        return false;
    }
    max_bits(&q) + max_bits(g) + bitlen(g.len().min(q.len())) + 1 < s as u64
}

/// The multimodular gcd of the primitive `a` and `b`, `len a ≥ len b ≥ 2`.
///
/// `first`: the monic gcd modulo one prime already known (of polynomials whose primitive parts
/// are `a` and `b`, the prime dividing neither leading coefficient), taken for that prime.
fn modular_from(a: &[Int], b: &[Int], mut first: Option<(u64, Vec<u64>)>) -> Vec<Int> {
    let (la, lb) = (a.len(), b.len());
    let lead = a[la - 1].number().gcd(&b[lb - 1].number());
    let lead = Int::of(&lead);
    let limbs = a.iter().chain(b).map(|x| x.m.len()).max().unwrap_or(1);
    let mut h: Vec<Int> = Vec::new();
    let mut m: Vec<u64> = Vec::new();
    let mut deg = usize::MAX;
    for p in Primes::new() {
        let mut t = Table::new(p, limbs);
        let g = match first.take() {
            Some((q, g)) if q == p => g,
            _ => {
                let (ap, bp) = (t.poly(a), t.poly(b));
                if ap.len() != la || bp.len() != lb {
                    continue;
                }
                hgcd::gcd(&t.m, &ap, &bp)
            }
        };
        let d = g.len() - 1;
        if d == 0 {
            return vec![Int {
                neg: false,
                m: vec![1],
            }];
        }
        if d > deg {
            continue;
        }
        // Monic, times the gcd of the leading coefficients.
        let w = &t.m.m;
        let f = w.mul(w.inv_lead(g[d]), t.int(&lead));
        let g: Vec<u64> = g.iter().map(|&x| w.mul(x, f)).collect();
        let changed = if d < deg {
            deg = d;
            h = symmetric(&g, p);
            m = vec![p];
            true
        } else {
            crt_step(&mut h, &mut m, &g, &mut t)
        };
        if changed {
            continue;
        }
        let cand = primitive(h.clone());
        if divides(b, &cand).is_some() && divides(a, &cand).is_some() {
            return cand;
        }
    }
    unreachable!("the primes below 2^62 are enough")
}

/// The greatest common divisor of the integer polynomials `a` and `b`, of a positive leading
/// coefficient, both not zero: the gcd of the contents times that of the primitive parts.
pub(crate) fn gcd(a: &[Number], b: &[Number]) -> Vec<Number> {
    let (ia, ib) = (ints(a), ints(b));
    // One prime first where the contents cost (coefficients of several limbs) or the heuristic
    // gcd would come first (short polynomials); its gcd serves the modular algorithm too.
    let wide = ia.iter().chain(&ib).any(|x| x.m.len() > 1);
    let mut first = None;
    if wide || a.len().max(b.len()) <= HEURISTIC_MAX_SHORT {
        match coprime_content(a, b, &ia, &ib) {
            Ok(c) => return vec![c],
            Err(f) => first = f,
        }
    }
    let (ca, pa) = content_pp_ints(ia);
    let (cb, pb) = content_pp_ints(ib);
    let (ca, cb) = (ca.number(), cb.number());
    let c = Int::of(&ca.gcd(&cb));
    let (pa, pb) = if pa.len() >= pb.len() {
        (pa, pb)
    } else {
        (pb, pa)
    };
    let g = if pb.len() == 1 {
        vec![Int {
            neg: false,
            m: vec![1],
        }]
    } else if pa == pb {
        pa
    } else if choose_heuristic(&pa, &pb) {
        heuristic(&pa, &pb).unwrap_or_else(|| modular_from(&pa, &pb, first))
    } else {
        modular_from(&pa, &pb, first)
    };
    let c = c.number();
    if crate::math::rational::is_one(&c) {
        numbers(&g)
    } else {
        g.iter().map(|x| &x.number() * &c).collect()
    }
}

/// The gcd of `a` and `b` (not zero) when they are coprime over the rationals, which their images
/// modulo a prime that divides neither leading coefficient show when their gcd there is a
/// constant (the degree of the gcd modulo such a prime is at least that over the rationals): then
/// the gcd over the integers is the positive gcd of all their coefficients, found without the
/// contents and primitive parts. Otherwise `Err` with that prime and its monic gcd, for the
/// modular algorithm, or `None` when the first two primes each divide a leading coefficient.
fn coprime_content(
    a: &[Number],
    b: &[Number],
    ia: &[Int],
    ib: &[Int],
) -> Result<Number, Option<(u64, Vec<u64>)>> {
    if a.len() < 2 || b.len() < 2 {
        return Err(None);
    }
    let limbs = ia.iter().chain(ib).map(|x| x.m.len()).max().unwrap_or(1);
    for p in Primes::new().take(2) {
        let t = Table::new(p, limbs);
        let (ap, bp) = (t.poly(ia), t.poly(ib));
        if ap.len() != a.len() || bp.len() != b.len() {
            continue;
        }
        let g = hgcd::gcd(&t.m, &ap, &bp);
        if g.len() != 1 {
            // Monic, for the modular algorithm's first prime.
            let w = &t.m.m;
            let f = w.inv_lead(g[g.len() - 1]);
            return Err(Some((p, g.iter().map(|&x| w.mul(x, f)).collect())));
        }
        let mut c = Number::default();
        for x in a.iter().chain(b) {
            c = c.gcd(x);
            if crate::math::rational::is_one(&c) {
                break;
            }
        }
        return Ok(c);
    }
    Err(None)
}

/// Up to this length the heuristic gcd comes first, whatever the coefficients.
pub(crate) const HEURISTIC_MAX_SHORT: usize = 8;

/// Whether the heuristic gcd is tried first, by the measured crossovers (Apple M4, against the
/// multimodular algorithm on the fraction-free Euclid modulo primes below 2^62): only for the
/// shortest polynomials, and for long ones of short coefficients (at most 48 bits from 200
/// coefficients), where one gcd of integers beats a few hundred steps modulo each of two primes.
/// (At 64 bits and 1000 coefficients the heuristic gcd is 15% faster with a common factor but
/// 1.75 times slower on coprime polynomials, which one prime settles.)
fn choose_heuristic(a: &[Int], b: &[Int]) -> bool {
    let bits = max_bits(a).max(max_bits(b));
    let len = a.len();
    len <= HEURISTIC_MAX_SHORT || (bits <= 48 && len >= 200)
}

// ----------------------------------------------------------------- resultants

/// An upper bound on `log2 |c|₂`, from the top two limbs of each coefficient.
fn log2_norm(c: &[Int]) -> f64 {
    let logs: Vec<f64> = c
        .iter()
        .filter(|x| !x.is_zero())
        .map(|x| {
            let n = x.m.len();
            let top = x.m[n - 1] as f64;
            let next = if n >= 2 { x.m[n - 2] as f64 } else { 0.0 };
            // (top + next/2^64 + 1/2^64) 2^(64(n-1)), rounded up.
            ((top + (next + 1.0) / 18_446_744_073_709_551_616.0).log2() + 64.0 * (n - 1) as f64)
                * (1.0 + 1e-12)
                + 1e-9
        })
        .collect();
    let mx = logs.iter().copied().fold(f64::MIN, f64::max);
    if logs.is_empty() {
        return 0.0;
    }
    let sum: f64 = logs.iter().map(|&l| (2.0 * (l - mx)).exp2()).sum();
    mx + 0.5 * sum.log2() + 1e-9
}

/// The resultant of the integer polynomials `a` and `b`, both of degree at least 1: modulo the
/// primes below `2^62` that divide neither leading coefficient, until their product passes twice
/// Hadamard's bound `|a|₂^deg b · |b|₂^deg a`, joined by the Chinese remainder theorem.
pub(crate) fn resultant(a: &[Number], b: &[Number]) -> Number {
    let (a, b) = (ints(a), ints(b));
    let (la, lb) = (a.len(), b.len());
    let bound = (lb - 1) as f64 * log2_norm(&a) + (la - 1) as f64 * log2_norm(&b) + 2.0;
    let limbs = a.iter().chain(&b).map(|x| x.m.len()).max().unwrap_or(1);
    let mut cong: Vec<(Number, Number)> = Vec::new();
    let mut bits = 0.0;
    for p in Primes::new() {
        if bits > bound {
            break;
        }
        let t = Table::new(p, limbs);
        let (ap, bp) = (t.poly(&a), t.poly(&b));
        if ap.len() != la || bp.len() != lb {
            continue;
        }
        let r = resultant_in(&t.m, &ap, &bp);
        cong.push((Number::from(r), Number::from(p)));
        bits += (p as f64).log2();
    }
    let (x, m) = Number::crt(&cong).expect("coprime moduli");
    // The symmetric residue.
    let half = &m >> 1usize;
    if x > half { &x - &m } else { x }
}

/// `(s, t, r)` with `s·a + t·b = r`, `r = res(a, b)` not zero, `deg s < deg b`, `deg t < deg a`,
/// for the coprime integer polynomials `a` and `b` of degree at least 1: the cofactors of the
/// extended gcds modulo primes below `2^62` times the resultants there, joined by the Chinese
/// remainder theorem past Hadamard's bound, which bounds every minor of the Sylvester matrix and so
/// the coefficients of `s` and `t` (Cramer's rule). The primes that divide the resultant are passed
/// over.
pub(crate) fn xgcd_coprime(a: &[Number], b: &[Number]) -> (Vec<Number>, Vec<Number>, Number) {
    let (a, b) = (ints(a), ints(b));
    let (la, lb) = (a.len(), b.len());
    let bound = (lb - 1) as f64 * log2_norm(&a) + (la - 1) as f64 * log2_norm(&b) + 2.0;
    let limbs = a.iter().chain(&b).map(|x| x.m.len()).max().unwrap_or(1);
    let mut hs: Vec<Int> = vec![Int::default(); lb - 1];
    let mut ht: Vec<Int> = vec![Int::default(); la - 1];
    let mut hr: Vec<Int> = vec![Int::default()];
    let mut m: Vec<u64> = Vec::new();
    let mut bits = 0.0;
    for p in Primes::new() {
        if bits > bound {
            break;
        }
        let mut t = Table::new(p, limbs);
        let (ap, bp) = (t.poly(&a), t.poly(&b));
        if ap.len() != la || bp.len() != lb {
            continue;
        }
        let r = resultant_in(&t.m, &ap, &bp);
        if r == 0 {
            continue;
        }
        let (g, s, u) = hgcd::xgcd(&t.m, &ap, &bp);
        debug_assert!(g.len() == 1);
        let w = &t.m.m;
        let f = w.mul(w.inv_lead(g[0]), r);
        let scaled = |v: &[u64], len: usize| -> Vec<u64> {
            let mut out: Vec<u64> = v.iter().map(|&x| w.mul(x, f)).collect();
            out.resize(len, 0);
            out
        };
        let (s, u) = (scaled(&s, lb - 1), scaled(&u, la - 1));
        if m.is_empty() {
            hs = symmetric(&s, p);
            ht = symmetric(&u, p);
            hr = symmetric(&[r], p);
            m = vec![p];
        } else {
            let mut m2 = m.clone();
            crt_step(&mut hs, &mut m2, &s, &mut t);
            let mut m3 = m.clone();
            crt_step(&mut ht, &mut m3, &u, &mut t);
            crt_step(&mut hr, &mut m, &[r], &mut t);
        }
        bits += (p as f64).log2();
    }
    (numbers(&hs), numbers(&ht), hr[0].number())
}
