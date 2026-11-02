//! The prime fields curves are defined over.
//!
//! A curve spends nearly all its time in the products of its field, so the field is chosen by the
//! shape of the prime when the curve is built, and everything above it is compiled for that field:
//!
//! - [`P521`]: P-521's prime `2^521 - 1`, in limbs of 58 bits, whose products need no carry chains.
//! - [`Pseudo`]: a prime `2^k - c` with a small `c`, as secp256k1's and Curve25519's, whose
//!   products are reduced by folding the high half back onto the low one.
//! - [`Mont`]: any other odd prime of up to nine limbs, P-521's length, in Montgomery's form.
//! - [`Big`]: a prime of any length, on [`Number`] and its [`Divisor`](crate::math::numbers::Divisor).

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod aarch64;
mod big;
mod mont;
mod p521;
mod pseudo;
mod special;

pub(super) use big::Big;
pub(super) use mont::Mont;
pub(super) use p521::P521;
pub(super) use pseudo::Pseudo;
pub(super) use special::Nist;

use crate::math::numbers::{ByteOrder, Number};
use std::sync::OnceLock;

/// The arithmetic of a prime field `F_p`.
///
/// Elements are held in the field's own form, which [`enter`](Field::enter) and
/// [`leave`](Field::leave) convert from and to numbers, and which need not be unique: compare them
/// with [`equal`](Field::equal) and [`is_zero`](Field::is_zero).
pub(super) trait Field {
    /// An element.
    type E: Clone;

    /// Whether a square costs so much less than a product that a formula is faster with a square
    /// and two subtractions in place of a product and an addition.
    const CHEAP_SQUARES: bool = false;

    /// The limbs of an element.
    fn limbs(&self) -> usize;

    /// The element stored in the [`limbs`](Field::limbs) limbs of `x`.
    fn load(&self, x: &[u64]) -> Self::E;

    /// Stores `x` in the [`limbs`](Field::limbs) limbs of `r`.
    fn store(&self, x: &Self::E, r: &mut [u64]);

    /// Zero.
    fn zero(&self) -> Self::E;

    /// One.
    fn one(&self) -> Self::E;

    /// `a + b`.
    fn add(&self, a: &Self::E, b: &Self::E) -> Self::E;

    /// `a - b`.
    fn sub(&self, a: &Self::E, b: &Self::E) -> Self::E;

    /// `-a`.
    fn neg(&self, a: &Self::E) -> Self::E;

    /// `a b`.
    fn mul(&self, a: &Self::E, b: &Self::E) -> Self::E;

    /// `a^2`.
    fn sqr(&self, a: &Self::E) -> Self::E;

    /// Whether `a` is zero.
    fn is_zero(&self, a: &Self::E) -> bool;

    /// The element congruent to `x`, of any sign and length.
    fn enter(&self, x: &Number) -> Self::E;

    /// The number in `[0, p)` that `x` stands for.
    fn leave(&self, x: &Self::E) -> Number;

    /// `a` times the short number `w`, which is the element `e`: a field whose elements are the
    /// numbers themselves can take a row of limb products for the whole product.
    #[inline(always)]
    fn mul_word(&self, a: &Self::E, _w: u64, e: &Self::E) -> Self::E {
        self.mul(a, e)
    }

    /// `2 a`.
    #[inline(always)]
    fn twice(&self, a: &Self::E) -> Self::E {
        self.add(a, a)
    }

    /// `a / 2`.
    fn half(&self, a: &Self::E) -> Self::E {
        self.mul(a, &self.enter(&((self.modulus() + 1u8) >> 1usize)))
    }

    /// Whether `a` equals `b`.
    #[inline(always)]
    fn equal(&self, a: &Self::E, b: &Self::E) -> bool {
        self.is_zero(&self.sub(a, b))
    }

    /// `1 / a`, or `None` for zero.
    fn invert(&self, a: &Self::E) -> Option<Self::E> {
        let x = self.leave(a);
        let p = self.modulus();
        x.mod_inverse(p).map(|r| self.enter(&r))
    }

    /// The prime.
    fn modulus(&self) -> &Number;
}

/// `x` to the power of the bits of `e`, the least significant limb first.
///
/// The exponents of square roots modulo the primes of the named curves are mostly long runs of
/// ones, `2^252 - 3` for Curve25519's and `2^519 - 1` for P-521's, so a run is appended by
/// products by `x^(2^j - 1)` for powers of two `j`, which the run at the top builds by doubling,
/// `x^(2^(2 j) - 1) = (x^(2^j - 1))^(2^j) x^(2^j - 1)`: Curve25519's takes 13 products, against 66
/// by windows alone. The rest goes by a sliding window of up to five bits, a product for each
/// window of odd value, whenever that takes fewer products than such a run.
pub(super) fn pow<F: Field>(f: &F, x: &F::E, e: &[u64]) -> F::E {
    let bit = |i: usize| (e[i / 64] >> (i % 64)) & 1 != 0;
    let mut i = bit_length_of(e);
    if i == 0 {
        return f.one();
    }
    // The run of ones from the top bit, and `ones[k] = x^(2^(2^k) - 1)` for `2^k` up to it.
    let run_at = |i: usize| (0..i).take_while(|&j| bit(i - 1 - j)).count();
    let top = run_at(i);
    let mut ones = vec![x.clone()];
    while 2 << (ones.len() - 1) <= top {
        let half = ones[ones.len() - 1].clone();
        let mut t = half.clone();
        for _ in 0..1usize << (ones.len() - 1) {
            t = f.sqr(&t);
        }
        ones.push(f.mul(&t, &half));
    }
    // A run of `n` ones appended to `r`, the longest pieces first, and how many products that is.
    let pieces = |mut n: usize| {
        let mut count = 0;
        while n > 0 {
            n -= 1 << (usize::BITS - 1 - n.leading_zeros()).min(ones.len() as u32 - 1);
            count += 1;
        }
        count
    };
    let append = |mut r: F::E, mut n: usize| {
        while n > 0 {
            let k = (usize::BITS - 1 - n.leading_zeros()).min(ones.len() as u32 - 1) as usize;
            for _ in 0..1usize << k {
                r = f.sqr(&r);
            }
            r = f.mul(&r, &ones[k]);
            n -= 1 << k;
        }
        r
    };
    let first = 1 << (ones.len() - 1);
    let mut r = append(ones[ones.len() - 1].clone(), top - first);
    i -= top;
    // The odd powers `x, x^3, ..., x^31` of the windows, made when one is first needed.
    let mut table: Vec<F::E> = vec![x.clone()];
    while i > 0 {
        if !bit(i - 1) {
            r = f.sqr(&r);
            i -= 1;
            continue;
        }
        let run = run_at(i);
        // With pieces of 16 ones and more, a window's table costs more than the rest of such an
        // exponent: secp256k1's ends in `1011`.
        if pieces(run) < run.div_ceil(5) || (ones.len() > 4 && table.len() == 1) {
            r = append(r, run);
            i -= run;
            continue;
        }
        // The window from this bit down to the lowest set one of the next five.
        let mut low = i.saturating_sub(5);
        while !bit(low) {
            low += 1;
        }
        let mut d = 0;
        for j in (low..i).rev() {
            d = 2 * d + usize::from(bit(j));
        }
        if d > 1 && table.len() == 1 {
            let x2 = f.sqr(x);
            for k in 1..16 {
                let t = f.mul(&table[k - 1], &x2);
                table.push(t);
            }
        }
        for _ in low..i {
            r = f.sqr(&r);
        }
        r = f.mul(&r, &table[d / 2]);
        i = low;
    }
    r
}

/// The bits of the magnitude `e`.
fn bit_length_of(e: &[u64]) -> usize {
    match e.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * i + 64 - e[i].leading_zeros() as usize,
        None => 0,
    }
}

/// Sets each element of `x` that is not zero to its inverse, by Montgomery's trick: the running
/// products, the inverse of the last, and two products back for each.
pub(super) fn invert_all<F: Field>(f: &F, x: &mut [F::E]) {
    let mut running = Vec::with_capacity(x.len());
    let mut product = f.one();
    for e in x.iter() {
        running.push(product.clone());
        if !f.is_zero(e) {
            product = f.mul(&product, e);
        }
    }
    let Some(mut inverse) = f.invert(&product) else {
        return;
    };
    for (e, before) in x.iter_mut().zip(running.iter()).rev() {
        if !f.is_zero(e) {
            let i = f.mul(&inverse, before);
            inverse = f.mul(&inverse, e);
            *e = i;
        }
    }
}

/// What a square root modulo `p` needs: `p - 1 = q 2^s`, `(q - 1) / 2`, and a power of a non-square.
pub struct SqrtPlan {
    s: u32,
    half: Vec<u64>,
    /// `z^q` for a non-square `z`, a generator of the group of order `2^s`, stored as the field
    /// stores its elements.
    c: Vec<u64>,
}

/// A square root of `a`, or `None` when it has none, by Tonelli and Shanks (see
/// [`Number::sqrt_mod`]): a single power for primes 3 modulo 4.
pub(super) fn sqrt<F: Field>(f: &F, plan: &SqrtPlan, a: &F::E) -> Option<F::E> {
    if f.is_zero(a) {
        return Some(f.zero());
    }
    let x = pow(f, a, &plan.half);
    let mut r = f.mul(a, &x);
    let mut b = f.mul(&r, &x);
    let one = f.one();
    if plan.s == 1 {
        return f.equal(&b, &one).then_some(r);
    }
    let mut c = f.load(&plan.c);
    let mut m = plan.s;
    while !f.equal(&b, &one) {
        let mut t = 0;
        let mut b2 = b.clone();
        while !f.equal(&b2, &one) {
            b2 = f.sqr(&b2);
            t += 1;
            if t == m {
                return None;
            }
        }
        let mut g = c;
        for _ in 0..m - t - 1 {
            g = f.sqr(&g);
        }
        r = f.mul(&r, &g);
        c = f.sqr(&g);
        b = f.mul(&b, &c);
        m = t;
    }
    Some(r)
}

/// A square root of `u / v`, for `v` not zero, or `None` when it has none. For primes 3 modulo 4
/// and 5 modulo 8 it takes one power and no inversion: `u v (u v^3)^((p - 3) / 4)`, and
/// `u v^3 (u v^7)^((p - 5) / 8)`, whose square is `u / v` or `-u / v`, the second fixed by a square
/// root of -1; other primes take an inversion and [`sqrt`].
pub(super) fn sqrt_ratio<F: Field>(f: &F, plan: &SqrtPlan, u: &F::E, v: &F::E) -> Option<F::E> {
    if plan.s > 2 {
        return sqrt(f, plan, &f.mul(u, &f.invert(v)?));
    }
    let v2 = f.sqr(v);
    let v3 = f.mul(&v2, v);
    let (front, inner) = if plan.s == 2 {
        let uv3 = f.mul(u, &v3);
        let uv7 = f.mul(&uv3, &f.sqr(&v2));
        (uv3, uv7)
    } else {
        (f.mul(u, v), f.mul(u, &v3))
    };
    let x = f.mul(&front, &pow(f, &inner, &plan.half));
    let vxx = f.mul(v, &f.sqr(&x));
    if f.equal(&vxx, u) {
        return Some(x);
    }
    // `z^q` for a non-square `z` has the order `2^s = 4`: its square is -1.
    (plan.s == 2 && f.equal(&vxx, &f.neg(u))).then(|| f.mul(&x, &f.load(&plan.c)))
}

/// The field of a curve, by the shape of its prime. A curve holds one for as long as it lives, so
/// the sizes of the variants matter little.
#[allow(clippy::large_enum_variant)]
pub enum Prime {
    Mont1(Mont<1>),
    Mont2(Mont<2>),
    Mont3(Mont<3>),
    Mont4(Mont<4>),
    Mont5(Mont<5>),
    Mont6(Mont<6>),
    Mont7(Mont<7>),
    Mont8(Mont<8>),
    Mont9(Mont<9>),
    Pseudo4(Pseudo<4>),
    Pseudo6(Pseudo<6>),
    Pseudo9(Pseudo<9>),
    P256(Nist<4>),
    P384(Nist<6>),
    P521(P521),
    Big(Big),
}

/// Runs `$body` with `$f` bound to the field in `$prime`, compiled for each of them.
macro_rules! with_prime {
    ($prime:expr, $f:ident => $body:expr) => {
        match $prime {
            $crate::math::ec::field::Prime::Mont1($f) => $body,
            $crate::math::ec::field::Prime::Mont2($f) => $body,
            $crate::math::ec::field::Prime::Mont3($f) => $body,
            $crate::math::ec::field::Prime::Mont4($f) => $body,
            $crate::math::ec::field::Prime::Mont5($f) => $body,
            $crate::math::ec::field::Prime::Mont6($f) => $body,
            $crate::math::ec::field::Prime::Mont7($f) => $body,
            $crate::math::ec::field::Prime::Mont8($f) => $body,
            $crate::math::ec::field::Prime::Mont9($f) => $body,
            $crate::math::ec::field::Prime::Pseudo4($f) => $body,
            $crate::math::ec::field::Prime::Pseudo6($f) => $body,
            $crate::math::ec::field::Prime::Pseudo9($f) => $body,
            $crate::math::ec::field::Prime::P256($f) => $body,
            $crate::math::ec::field::Prime::P384($f) => $body,
            $crate::math::ec::field::Prime::P521($f) => $body,
            $crate::math::ec::field::Prime::Big($f) => $body,
        }
    };
}
pub(super) use with_prime;

/// P-256's prime, `2^256 - 2^224 + 2^192 + 2^96 - 1`.
const P256_LIMBS: [u64; 4] = [u64::MAX, 0xffff_ffff, 0, 0xffff_ffff_0000_0001];

/// P-384's prime, `2^384 - 2^128 - 2^96 + 2^32 - 1`.
const P384_LIMBS: [u64; 6] = [
    0xffff_ffff,
    0xffff_ffff_0000_0000,
    u64::MAX - 1,
    u64::MAX,
    u64::MAX,
    u64::MAX,
];

/// P-521's prime, `2^521 - 1`.
const P521_LIMBS: [u64; 9] = [
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    u64::MAX,
    0x1ff,
];

impl Prime {
    /// The field of the odd prime `p`, the fastest for its shape.
    fn new(p: &Number) -> Prime {
        let l = limbs_of(p);
        if l == P256_LIMBS {
            return Prime::P256(Nist::new(Mont::new(p, P256_LIMBS)));
        }
        if l == P384_LIMBS {
            return Prime::P384(Nist::new(Mont::new(p, P384_LIMBS)));
        }
        if l == P521_LIMBS {
            return Prime::P521(P521::new(p));
        }
        macro_rules! pseudo_or_mont {
            ($($n:literal, $mont:ident, $pseudo:ident);+ $(;)?) => {
                match l.len() {
                    $($n => {
                        let m: [u64; $n] = l[..].try_into().unwrap_or([0; $n]);
                        match Pseudo::<$n>::new(p) {
                            Some(f) => Prime::$pseudo(f),
                            None => Prime::$mont(Mont::new(p, m)),
                        }
                    })+
                    1 => Prime::Mont1(Mont::new(p, [l[0]])),
                    2 => Prime::Mont2(Mont::new(p, [l[0], l[1]])),
                    3 => Prime::Mont3(Mont::new(p, [l[0], l[1], l[2]])),
                    5 => Prime::Mont5(Mont::new(p, [l[0], l[1], l[2], l[3], l[4]])),
                    7 => Prime::Mont7(Mont::new(p, [l[0], l[1], l[2], l[3], l[4], l[5], l[6]])),
                    8 => Prime::Mont8(Mont::new(
                        p,
                        [l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]],
                    )),
                    _ => Prime::Big(Big::new(p)),
                }
            };
        }
        pseudo_or_mont!(4, Mont4, Pseudo4; 6, Mont6, Pseudo6; 9, Mont9, Pseudo9)
    }
}

/// The field of a curve, with the prime and the plan for its square roots.
pub struct Base {
    /// The arithmetic.
    pub(super) prime: Prime,
    /// The prime, in the byte order the curve was given it.
    pub(super) p: Number,
    /// What square roots need, once one is asked for.
    sqrt: OnceLock<SqrtPlan>,
}

impl Base {
    /// The field of the odd prime `p`.
    pub(super) fn new(p: &Number) -> Base {
        Base {
            prime: Prime::new(p),
            p: p.clone(),
            sqrt: OnceLock::new(),
        }
    }

    /// What square roots in this field need.
    pub(super) fn sqrt_plan(&self) -> &SqrtPlan {
        self.sqrt.get_or_init(|| {
            let p = &self.p;
            let one = Number::new_from_u8(1, p.byte_order);
            let p1 = p - &one;
            let s = p1.trailing_zeros().unwrap_or(0);
            let q = &p1 >> s as usize;
            let half = (&q - &one) >> 1usize;
            let mut z = Number::new_from_u8(2, p.byte_order);
            while s > 1 && z.jacobi(p) != -1 {
                z = &z + &one;
            }
            let c = z.pow_mod(&q, p);
            SqrtPlan {
                s: s as u32,
                half: limbs_of(&half),
                c: with_prime!(&self.prime, f => {
                    let mut r = vec![0; f.limbs()];
                    f.store(&f.enter(&c), &mut r);
                    r
                }),
            }
        })
    }

    /// The bytes an element takes.
    pub(super) fn bytes(&self) -> usize {
        self.p.bits().div_ceil(8) as usize
    }
}

/// The limbs of the magnitude of `x`, the least significant first, without zeros at the top.
pub(super) fn limbs_of(x: &Number) -> Vec<u64> {
    let mut l: Vec<u64> = if x.byte_order == ByteOrder::LittleEndian {
        x.value
            .chunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(b)
            })
            .collect()
    } else {
        x.value
            .rchunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[8 - c.len()..].copy_from_slice(c);
                u64::from_be_bytes(b)
            })
            .collect()
    };
    while l.last() == Some(&0) {
        l.pop();
    }
    l
}

/// The number with the limbs `x`, the least significant first, in the byte order `order`.
pub(super) fn number_of(x: &[u64], order: ByteOrder) -> Number {
    let mut bytes = Vec::with_capacity(8 * x.len());
    for l in x {
        bytes.extend_from_slice(&l.to_le_bytes());
    }
    let mut n = Number::new(bytes, ByteOrder::LittleEndian);
    if order == ByteOrder::BigEndian {
        n.value.reverse();
        n.byte_order = ByteOrder::BigEndian;
    }
    n
}

/// `a / 2` modulo the odd `m`, for `a` of `N` limbs: `a`, or `a + m` when `a` is odd, shifted down
/// a bit, with the carry of the sum coming in at the top.
#[inline(always)]
fn halve<const N: usize>(a: &[u64; N], m: &[u64; N]) -> [u64; N] {
    let mask = (a[0] & 1).wrapping_neg();
    let mut s = [0; N];
    let mut carry = false;
    for i in 0..N {
        let (x, c1) = a[i].overflowing_add(m[i] & mask);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        s[i] = x;
        carry = c1 | c2;
    }
    let mut r = [0; N];
    for i in 0..N {
        let above = if i + 1 < N {
            s[i + 1]
        } else {
            u64::from(carry)
        };
        r[i] = (s[i] >> 1) | (above << 63);
    }
    r
}

/// `x` as the array of `M` limbs it is, in the branches of a field generic in `N` for `N = M`.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
#[inline(always)]
fn same<const N: usize, const M: usize>(x: &[u64; N]) -> &[u64; M] {
    x.as_slice().try_into().unwrap_or_else(|_| unreachable!())
}

/// The limbs of the number in `[0, p)`, `N` of them.
pub(super) fn array_of<const N: usize>(x: &Number) -> [u64; N] {
    let l = limbs_of(x);
    let mut r = [0; N];
    r[..l.len()].copy_from_slice(&l);
    r
}
