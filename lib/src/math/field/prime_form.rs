//! The forms of a prime field's arithmetic, chosen once from the prime, and the code that runs on
//! each.
//!
//! The forms are those of the curves of [`a::math::ec`](crate::math::ec), whose fields are reused
//! as they are: Montgomery's form in one to nine limbs (`Mont<N>`), the primes `2^k - c` with a
//! small `c` (`Pseudo<N>` for four, six and nine limbs), the NIST primes of P-256 and P-384
//! (`Nist<N>`, Montgomery's form with their own products where a target has them), P-521's
//! `2^521 - 1` in limbs of 58 bits (`P521`); beyond nine limbs, Montgomery's form of any length
//! ([`Wide`], in `prime_wide`). The field `GF(2)`, which none of them can be, has its own form,
//! [`Two`].
//!
//! An element of a field of up to nine limbs holds its limbs inline ([`Value::Limbs`]), as its
//! form holds them; the elements of the longer fields hold a vector of limbs ([`Value::Wide`]).
//! The assignments go through [`Arith::mul_into`] and its like, which write only the form's limbs
//! (in the element's own vector for [`Wide`]).
//! [`with_form`] matches the form once and runs its body compiled for that form's arithmetic,
//! the [`Arith`] trait, which reads and writes the elements in place.

use super::prime_inv::inverse;
use super::prime_wide::Wide;
use crate::math::ec::field::{
    Big, Field, Mont, Nist, P521, Prime, Pseudo, limbs_of, number_of, pow,
};
use crate::math::numbers::{ByteOrder, Number};
use std::sync::OnceLock;

/// The most limbs an element holds inline: P-521's nine.
pub(super) const INLINE: usize = 9;

/// An element as its field's form holds it.
#[derive(Clone)]
pub(super) enum Value {
    /// The limbs of an element of a field of up to [`INLINE`] limbs, in its form's own
    /// representation, the low limb first; the limbs the form does not use are zero.
    Limbs([u64; INLINE]),
    /// An element of a longer field: the number below the prime, little-endian.
    Big(Number),
    /// An element of a field of more than nine limbs in Montgomery's form ([`Wide`]).
    Wide(Vec<u64>),
    /// An element of such a field as it was made, from bytes or a number: its value, and its
    /// residue once an operation has needed it ([`Wide`]).
    WideValue(WideValue),
}

/// An element of a field of more than nine limbs made from its value, which its encodings and
/// `value` take as it is: the residue `xR mod p` is made (once) by the first operation on it.
#[derive(Clone)]
pub(super) struct WideValue {
    /// The value's limbs, below the prime.
    pub(super) value: Vec<u64>,
    /// The residue, made on the first use.
    pub(super) residue: OnceLock<Vec<u64>>,
}

/// The limbs of a number below the prime, the low limb first, on the stack for up to [`INLINE`]
/// limbs.
pub(super) struct Canon {
    /// The limbs, for a field of up to [`INLINE`] limbs.
    inline: [u64; INLINE],
    /// The limbs, for a longer field.
    heap: Vec<u64>,
    /// The number of limbs, the prime's.
    n: usize,
}

impl Canon {
    /// `n` zero limbs.
    #[inline(always)]
    pub(super) fn zeros(n: usize) -> Canon {
        Canon {
            inline: [0; INLINE],
            heap: if n > INLINE { vec![0; n] } else { Vec::new() },
            n,
        }
    }

    /// The limbs.
    #[inline(always)]
    pub(super) fn limbs(&self) -> &[u64] {
        if self.n > INLINE {
            &self.heap
        } else {
            &self.inline[..self.n]
        }
    }

    /// The limbs, to write.
    #[inline(always)]
    pub(super) fn limbs_mut(&mut self) -> &mut [u64] {
        if self.n > INLINE {
            &mut self.heap
        } else {
            &mut self.inline[..self.n]
        }
    }
}

/// The field `GF(2)`, whose elements are the bits 0 and 1 in one limb: a sum is an exclusive or
/// and a product an and.
#[derive(Clone)]
pub(super) struct Two {
    /// The prime 2.
    p: Number,
}

impl Field for Two {
    type E = [u64; 1];

    fn limbs(&self) -> usize {
        1
    }

    fn load(&self, x: &[u64]) -> [u64; 1] {
        [x[0] & 1]
    }

    fn store(&self, x: &[u64; 1], r: &mut [u64]) {
        r[0] = x[0];
    }

    fn zero(&self) -> [u64; 1] {
        [0]
    }

    fn one(&self) -> [u64; 1] {
        [1]
    }

    fn add(&self, a: &[u64; 1], b: &[u64; 1]) -> [u64; 1] {
        [a[0] ^ b[0]]
    }

    fn sub(&self, a: &[u64; 1], b: &[u64; 1]) -> [u64; 1] {
        [a[0] ^ b[0]]
    }

    fn neg(&self, a: &[u64; 1]) -> [u64; 1] {
        *a
    }

    fn mul(&self, a: &[u64; 1], b: &[u64; 1]) -> [u64; 1] {
        [a[0] & b[0]]
    }

    fn sqr(&self, a: &[u64; 1]) -> [u64; 1] {
        *a
    }

    fn is_zero(&self, a: &[u64; 1]) -> bool {
        a[0] == 0
    }

    fn enter(&self, x: &Number) -> [u64; 1] {
        // The parity of `x` is that of its magnitude, whatever its sign.
        [u64::from(x.bit(0))]
    }

    fn leave(&self, x: &[u64; 1]) -> Number {
        Number::new_from_u8(x[0] as u8, self.p.byte_order)
    }

    // 2 is 0 here: nothing is halved.
    fn half(&self, a: &[u64; 1]) -> [u64; 1] {
        *a
    }

    fn invert(&self, a: &[u64; 1]) -> Option<[u64; 1]> {
        (a[0] == 1).then_some(*a)
    }

    fn modulus(&self) -> &Number {
        &self.p
    }
}

/// The form of a field's arithmetic. A field holds one for as long as it lives, as a curve does
/// ec's `Prime`, so the sizes of the variants matter little.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub(super) enum Form {
    /// `GF(2)`.
    Two(Two),
    /// An odd prime of more than nine limbs, in Montgomery's form of any length.
    Wide(Wide),
    /// An odd prime, in the form ec chooses for it.
    Odd(Prime),
}

impl Form {
    /// The form for the prime `p`: [`Two`] for 2, and for an odd prime the fastest of ec's.
    pub(super) fn new(p: &Number) -> Form {
        if *p == 2u8 {
            Form::Two(Two { p: p.clone() })
        } else if p.bits() > 64 * INLINE as u64 {
            // From ten limbs on, Montgomery's form of any length; the inline forms stop at nine.
            // Measured on an Apple M4 Max (generic primes of 5 to 9 limbs in each form,
            // alternated): the inline forms' products take 0.40–0.80 times the Wide form's time and
            // their sums 0.59–0.77, their inverses the same; only square roots run faster in the
            // Wide form (0.49–0.80 times the inline forms' time).
            Form::Wide(Wide::new(p))
        } else {
            Form::Odd(Prime::new(p))
        }
    }

    /// The name of the form, for `Debug`.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Form::Two(_) => "GF(2)",
            Form::Wide(_) => "Montgomery, more than 9 limbs",
            Form::Odd(Prime::Mont1(_)) => "Montgomery, 1 limb",
            Form::Odd(Prime::Mont2(_)) => "Montgomery, 2 limbs",
            Form::Odd(Prime::Mont3(_)) => "Montgomery, 3 limbs",
            Form::Odd(Prime::Mont4(_)) => "Montgomery, 4 limbs",
            Form::Odd(Prime::Mont5(_)) => "Montgomery, 5 limbs",
            Form::Odd(Prime::Mont6(_)) => "Montgomery, 6 limbs",
            Form::Odd(Prime::Mont7(_)) => "Montgomery, 7 limbs",
            Form::Odd(Prime::Mont8(_)) => "Montgomery, 8 limbs",
            Form::Odd(Prime::Mont9(_)) => "Montgomery, 9 limbs",
            Form::Odd(Prime::Pseudo4(_)) => "2^k - c, 4 limbs",
            Form::Odd(Prime::Pseudo6(_)) => "2^k - c, 6 limbs",
            Form::Odd(Prime::Pseudo9(_)) => "2^k - c, 9 limbs",
            Form::Odd(Prime::P256(_)) => "P-256",
            Form::Odd(Prime::P384(_)) => "P-384",
            Form::Odd(Prime::P521(_)) => "P-521, limbs of 58 bits",
            Form::Odd(Prime::Big(_)) => "Number",
        }
    }
}

/// Runs `$body` with `$f` bound to the arithmetic of the form `$form`, compiled for each form.
macro_rules! with_form {
    ($form:expr, $f:ident => $body:expr) => {{
        use $crate::math::ec::field::Prime;
        use $crate::math::field::prime_form::Form;
        match $form {
            Form::Two($f) => $body,
            Form::Wide($f) => $body,
            Form::Odd(Prime::Mont1($f)) => $body,
            Form::Odd(Prime::Mont2($f)) => $body,
            Form::Odd(Prime::Mont3($f)) => $body,
            Form::Odd(Prime::Mont4($f)) => $body,
            Form::Odd(Prime::Mont5($f)) => $body,
            Form::Odd(Prime::Mont6($f)) => $body,
            Form::Odd(Prime::Mont7($f)) => $body,
            Form::Odd(Prime::Mont8($f)) => $body,
            Form::Odd(Prime::Mont9($f)) => $body,
            Form::Odd(Prime::Pseudo4($f)) => $body,
            Form::Odd(Prime::Pseudo6($f)) => $body,
            Form::Odd(Prime::Pseudo9($f)) => $body,
            Form::Odd(Prime::P256($f)) => $body,
            Form::Odd(Prime::P384($f)) => $body,
            Form::Odd(Prime::P521($f)) => $body,
            Form::Odd(Prime::Big($f)) => $body,
        }
    }};
}
pub(super) use with_form;

/// A form's arithmetic on the elements as [`Value`] holds them, and the conversions between its
/// representation and the limbs of the numbers below the prime.
pub(super) trait Arith: Field + Sized {
    /// The element `v` holds, borrowed in place.
    fn get<'a>(&self, v: &'a Value) -> &'a Self::E;

    /// `e`, to hold.
    fn put(&self, e: Self::E) -> Value;

    /// Sets the element `v` holds to `g` of it, in place: an inline element has only the form's
    /// limbs rewritten, the others (zeros) and the tag left as they are.
    fn update(&self, v: &mut Value, g: impl FnOnce(&Self::E) -> Self::E);

    /// `a *= b`, in place.
    #[inline(always)]
    fn mul_into(&self, a: &mut Value, b: &Value) {
        self.update(a, |x| self.mul(x, self.get(b)));
    }

    /// `a += b`, in place.
    #[inline(always)]
    fn add_into(&self, a: &mut Value, b: &Value) {
        self.update(a, |x| self.add(x, self.get(b)));
    }

    /// `a -= b`, in place.
    #[inline(always)]
    fn sub_into(&self, a: &mut Value, b: &Value) {
        self.update(a, |x| self.sub(x, self.get(b)));
    }

    /// Writes the limbs of the number below the prime that `e` stands for to `out`, which has
    /// the prime's [`limbs`](Field::limbs).
    fn canon(&self, e: &Self::E, out: &mut [u64]);

    /// The element of the number below the prime with the limbs `x`, as many as the prime's.
    fn of_canon(&self, x: &[u64]) -> Self::E;

    /// [`of_canon`](Arith::of_canon), to hold.
    #[inline(always)]
    fn put_canon(&self, x: &[u64]) -> Value {
        self.put(self.of_canon(x))
    }

    /// The element of any number, its residue modulo the prime, to hold.
    #[inline(always)]
    fn put_number(&self, x: &Number) -> Value {
        self.put(self.enter(x))
    }

    /// [`canon`](Arith::canon) of the element `v` holds.
    #[inline(always)]
    fn canon_value(&self, v: &Value, out: &mut [u64]) {
        self.canon(self.get(v), out);
    }

    /// Whether `a` and `b` are the same element, which the form may represent in more than one
    /// way.
    fn same(&self, a: &Self::E, b: &Self::E) -> bool;

    /// `x^e`, for the exponent with the limbs `e`, the low one first: ec's sliding windows and
    /// runs of ones.
    #[inline(always)]
    fn power(&self, x: &Self::E, e: &[u64]) -> Self::E {
        pow(self, x, e)
    }

    /// For Montgomery's form of one limb: the prime and `-1/p mod 2^64`, for sums of products
    /// reduced once (the extension fields' schoolbook).
    #[inline(always)]
    fn word_prime(&self) -> Option<(u64, u64)> {
        None
    }

    /// `1 / x`, or `None` for 0.
    #[inline(always)]
    fn invert_value(&self, x: &Self::E) -> Option<Self::E> {
        inverse(self, x)
    }

    /// The Legendre symbol of `x`, for an odd prime.
    #[inline(always)]
    fn legendre_value(&self, x: &Self::E) -> i32 {
        legendre_of_value(self, x)
    }
}

/// The `N` limbs at the bottom of an inline value.
#[inline(always)]
pub(super) fn inline<const N: usize>(v: &Value) -> &[u64; N] {
    match v {
        Value::Limbs(l) => l.first_chunk().unwrap_or_else(|| unreachable!()),
        _ => unreachable!("an element of a long field in a short one"),
    }
}

/// Sets the `N` limbs at the bottom of an inline value to `g` of them.
#[inline(always)]
fn update_inline<const N: usize>(v: &mut Value, g: impl FnOnce(&[u64; N]) -> [u64; N]) {
    match v {
        Value::Limbs(l) => {
            let head: &mut [u64; N] = l.first_chunk_mut().unwrap_or_else(|| unreachable!());
            *head = g(head);
        }
        _ => unreachable!("an element of a long field in a short one"),
    }
}

/// The `N` limbs `e`, inline.
#[inline(always)]
pub(super) fn held<const N: usize>(e: &[u64; N]) -> Value {
    let mut l = [0; INLINE];
    l[..N].copy_from_slice(e);
    Value::Limbs(l)
}

/// The array of the `N` limbs `x`.
#[inline(always)]
fn array<const N: usize>(x: &[u64]) -> [u64; N] {
    *x.first_chunk().unwrap_or_else(|| unreachable!())
}

impl<const N: usize> Arith for Mont<N> {
    #[inline(always)]
    fn word_prime(&self) -> Option<(u64, u64)> {
        // A prime above 2⁶³ takes one product per reduction anyway (the sum of two may pass p·2⁶⁴).
        self.word().filter(|&(p, _)| p >> 63 == 0)
    }

    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a [u64; N] {
        inline(v)
    }

    #[inline(always)]
    fn put(&self, e: [u64; N]) -> Value {
        held(&e)
    }

    #[inline(always)]
    fn update(&self, v: &mut Value, g: impl FnOnce(&[u64; N]) -> [u64; N]) {
        update_inline(v, g)
    }

    #[inline(always)]
    fn canon(&self, e: &[u64; N], out: &mut [u64]) {
        out.copy_from_slice(&self.canonical(e));
    }

    #[inline(always)]
    fn of_canon(&self, x: &[u64]) -> [u64; N] {
        self.residue(&array(x))
    }

    #[inline(always)]
    fn same(&self, a: &[u64; N], b: &[u64; N]) -> bool {
        // Montgomery's residues are fully reduced.
        a == b
    }
}

impl<const N: usize> Arith for Nist<N>
where
    Nist<N>: Field<E = [u64; N]>,
{
    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a [u64; N] {
        inline(v)
    }

    #[inline(always)]
    fn put(&self, e: [u64; N]) -> Value {
        held(&e)
    }

    #[inline(always)]
    fn update(&self, v: &mut Value, g: impl FnOnce(&[u64; N]) -> [u64; N]) {
        update_inline(v, g)
    }

    #[inline(always)]
    fn canon(&self, e: &[u64; N], out: &mut [u64]) {
        out.copy_from_slice(&self.mont().canonical(e));
    }

    #[inline(always)]
    fn of_canon(&self, x: &[u64]) -> [u64; N] {
        self.mont().residue(&array(x))
    }

    #[inline(always)]
    fn same(&self, a: &[u64; N], b: &[u64; N]) -> bool {
        a == b
    }
}

impl<const N: usize> Arith for Pseudo<N>
where
    Pseudo<N>: Field<E = [u64; N]>,
{
    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a [u64; N] {
        inline(v)
    }

    #[inline(always)]
    fn put(&self, e: [u64; N]) -> Value {
        held(&e)
    }

    #[inline(always)]
    fn update(&self, v: &mut Value, g: impl FnOnce(&[u64; N]) -> [u64; N]) {
        update_inline(v, g)
    }

    #[inline(always)]
    fn canon(&self, e: &[u64; N], out: &mut [u64]) {
        out.copy_from_slice(&self.canonical(e));
    }

    #[inline(always)]
    fn of_canon(&self, x: &[u64]) -> [u64; N] {
        array(x)
    }

    #[inline(always)]
    fn same(&self, a: &[u64; N], b: &[u64; N]) -> bool {
        // Any `N` limbs are an element: only the canonical ones compare.
        self.canonical(a) == self.canonical(b)
    }
}

impl Arith for P521 {
    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a [u64; 9] {
        inline(v)
    }

    #[inline(always)]
    fn put(&self, e: [u64; 9]) -> Value {
        Value::Limbs(e)
    }

    #[inline(always)]
    fn update(&self, v: &mut Value, g: impl FnOnce(&[u64; 9]) -> [u64; 9]) {
        update_inline(v, g)
    }

    #[inline(always)]
    fn canon(&self, e: &[u64; 9], out: &mut [u64]) {
        out.copy_from_slice(&self.to_limbs(e));
    }

    #[inline(always)]
    fn of_canon(&self, x: &[u64]) -> [u64; 9] {
        self.of_limbs(&array(x))
    }

    #[inline(always)]
    fn same(&self, a: &[u64; 9], b: &[u64; 9]) -> bool {
        self.to_limbs(a) == self.to_limbs(b)
    }
}

impl Arith for Big {
    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a Number {
        match v {
            Value::Big(x) => x,
            _ => unreachable!("an element of another form"),
        }
    }

    #[inline(always)]
    fn put(&self, e: Number) -> Value {
        Value::Big(e)
    }

    fn update(&self, v: &mut Value, g: impl FnOnce(&Number) -> Number) {
        let r = g(self.get(v));
        *v = Value::Big(r);
    }

    fn canon(&self, e: &Number, out: &mut [u64]) {
        let l = limbs_of(e);
        out[..l.len()].copy_from_slice(&l);
        out[l.len()..].fill(0);
    }

    fn of_canon(&self, x: &[u64]) -> Number {
        number_of(x, ByteOrder::LittleEndian)
    }

    #[inline(always)]
    fn same(&self, a: &Number, b: &Number) -> bool {
        // The elements are the numbers below the prime.
        a == b
    }

    // The elements are numbers: `Number`'s own power, by Montgomery's products on its limbs, and
    // its inverse, by Lehmer's extended Euclid, are faster than products through the divisor.
    fn power(&self, x: &Number, e: &[u64]) -> Number {
        x.pow_mod(&number_of(e, ByteOrder::LittleEndian), self.modulus())
    }

    fn invert_value(&self, x: &Number) -> Option<Number> {
        x.mod_inverse(self.modulus())
    }
}

impl Arith for Two {
    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a [u64; 1] {
        inline(v)
    }

    #[inline(always)]
    fn put(&self, e: [u64; 1]) -> Value {
        held(&e)
    }

    #[inline(always)]
    fn update(&self, v: &mut Value, g: impl FnOnce(&[u64; 1]) -> [u64; 1]) {
        update_inline(v, g)
    }

    #[inline(always)]
    fn canon(&self, e: &[u64; 1], out: &mut [u64]) {
        out[0] = e[0];
    }

    #[inline(always)]
    fn of_canon(&self, x: &[u64]) -> [u64; 1] {
        [x[0]]
    }

    #[inline(always)]
    fn same(&self, a: &[u64; 1], b: &[u64; 1]) -> bool {
        a == b
    }

    fn invert_value(&self, x: &[u64; 1]) -> Option<[u64; 1]> {
        self.invert(x)
    }
}

/// Writes the limbs of the magnitude of `x` to `out`, which has room for them, the low limb
/// first, and zeros above.
pub(super) fn read_limbs(x: &Number, out: &mut [u64]) {
    out.fill(0);
    let n = x.value.len();
    for (i, &b) in x.value.iter().enumerate() {
        let k = match x.byte_order {
            ByteOrder::LittleEndian => i,
            ByteOrder::BigEndian => n - 1 - i,
        };
        if b != 0 {
            out[k / 8] |= u64::from(b) << (8 * (k % 8));
        }
    }
}

/// Runs `f` on the limbs of the magnitude of `x`, the low limb first: on the stack for up to 18
/// limbs, twice the longest inline element, and in a vector beyond.
pub(super) fn with_limbs<R>(x: &Number, f: impl FnOnce(&[u64]) -> R) -> R {
    let n = x.value.len().div_ceil(8);
    if n <= 2 * INLINE {
        let mut buf = [0u64; 2 * INLINE];
        read_limbs(x, &mut buf[..n]);
        f(&buf[..n])
    } else {
        f(&limbs_of(x))
    }
}

/// The limbs of the number below the prime that `e` stands for.
#[inline(always)]
pub(super) fn canonical<F: Arith>(f: &F, e: &F::E) -> Canon {
    let mut c = Canon::zeros(f.limbs());
    f.canon(e, c.limbs_mut());
    c
}

/// [`canonical`] of the element `v` holds.
#[inline(always)]
pub(super) fn canonical_value<F: Arith>(f: &F, v: &Value) -> Canon {
    let mut c = Canon::zeros(f.limbs());
    f.canon_value(v, c.limbs_mut());
    c
}

/// The Legendre symbol of `e`, for an odd prime: [`Number::jacobi`] on the value, the binary
/// method up to six limbs (in assembly on aarch64) and Lehmer's steps with the symbol carried
/// through the quotients beyond, on the stack. Leaving the field's form costs one short number; Euler's criterion, the power
/// `e^((p - 1) / 2)`, which would stay in it, took 5 to 20 times as long on the named primes.
pub(super) fn legendre<F: Arith>(f: &F, e: &F::E) -> i32 {
    f.legendre_value(e)
}

/// [`legendre`] on the value: [`Number::jacobi`] of the canonical limbs.
fn legendre_of_value<F: Arith>(f: &F, e: &F::E) -> i32 {
    let c = canonical(f, e);
    number_of(c.limbs(), ByteOrder::LittleEndian).jacobi(f.modulus())
}
