//! Zech's logarithms for the small extension fields: for a field of at most [`max_order`]
//! elements over a prime of one word, the powers `hⁱ` of a primitive element `h`, their
//! logarithms, and Zech's logarithms `log(1 + hⁱ)`, three tables made on the first use of an
//! element and shared by the field's copies. An element of such a field is held as its logarithm
//! ([`Repr::Log`], [`ZERO`] for 0): a product is a sum of two logarithms, a sum
//! `hˣ + hʸ = h^(x + log(1 + h^(y − x)))`, a negation the product by `−1 = h^((q − 1)/2)`, an
//! inverse a difference, a power a product of a logarithm by the exponent modulo `q − 1`, a square
//! root a halving, the norm a power and the Frobenius map a product by `pⁱ`, as FLINT's `fq_zech`
//! does. The products, inverses, powers, Frobenius maps, characters and orders of logarithms need
//! no table; making an element from its coefficients, a sum, the coefficients, the norm and the
//! square root do, and the first of them makes the tables.
//!
//! For the conversions, an element is encoded as the integer `Σ wᵢ·pⁱ` of the words `wᵢ` its
//! coefficients hold in the base field's form (Montgomery's form of one word, whose residues
//! `cR mod p` are a permutation of `0..p`, or GF(2)'s bits), so that no coefficient is converted in
//! or out. `h` is the first primitive element in the order of the integers `Σ cᵢ·pⁱ` (its test,
//! `h^((q − 1)/ℓ) ≠ 1` for the primes `ℓ` of `q − 1`, on the coefficients' values). The tables
//! come from a walk through the group on the words, the multiplication being linear: for `p = 2`
//! by products by `h`, a few shifts and exclusive ors of a word each; for odd `p` by products by
//! `x`, a shift of the digits packed in a word and one sum modulo `p` for each term of
//! `xᵏ mod g`, with the encoding carried along, through the cosets of the powers of `x`: with
//! `m` the order of `x`, `r = (q − 1)/m` and `h^r = x^a` (`h^r` has the order `m`, so it is a
//! power of `x`, read in the walk of the powers of `x`), `x = h^s` for `s = r·(a⁻¹ mod m)`, and
//! the element `hʲ·xᵗ` (`j < r`, `t < m`) has the logarithm `j + s·t mod (q − 1)`: products by
//! `h` start the cosets. Each walk goes in four runs at once (four pieces of the powers, four
//! groups of cosets), whose chains of dependent operations overlap; `m` comes first, by the tests
//! `x^(m/ℓ) = 1` for the primes `ℓ` of `q − 1`.

use super::ext::{ExtensionField, Fpk, Repr};
use super::prime::Fp;
use super::prime_form::{Form, held, inline};
use crate::math::ec::field::{Prime, limbs_of};
use crate::math::numbers::Number;
use std::sync::OnceLock;

/// The most elements of a field with tables: three tables of 16-bit entries, `6q` bytes, at most
/// 384 KiB, made by a walk of `q` steps of a few operations of words each on the first use. Below
/// it, [`max_order`] says which fields take them.
pub(super) const MAX_ORDER: u64 = 1 << 16;

/// The most elements of a field of degree `k` with tables: [`MAX_ORDER`], but `2¹⁴` for `k = 2`
/// and 3, so that the tables are made where they repay their making within about 2000 operations.
///
/// Measured on an Apple M4 Max (each way alternated, 21 samples): the first use makes them in
/// 1.5 ns an element for `p = 2` and 2.5–4 ns for odd `p` (0.1 ms for `GF(2¹⁶)`, 0.24 ms for
/// `GF(3¹⁰)`, 0.21 ms for `GF(251²)`), and a product with a sum then takes 7–11 ns against the
/// words' 37–141 ns (`k` = 2 to 10) and the general form's 78–347 ns for `p = 2`. The tables so
/// repay their making after 300 such operations in `GF(2¹⁶)` and 430–1950 in the fields of 3⁸ to
/// 3¹⁰, 5⁶, 7⁵, 13⁴, 23³ and 127² elements; the fields of degree 2 and 3 beyond `2¹⁴` elements,
/// whose products gain least on the words, need 1950–7000 (29³ to 37³, 131² to 251²) and keep the
/// words, which make their first use some 100 to 200 times cheaper and each later product and sum
/// 4 to 5 times dearer.
pub(super) fn max_order(k: usize) -> u64 {
    if k <= 3 { 1 << 14 } else { MAX_ORDER }
}

/// The logarithm that stands for 0.
pub(super) const ZERO: u32 = u32::MAX;

/// The most coefficients of a field with tables (`2¹⁶`, for `p = 2`).
const KMAX: usize = 16;

/// The coefficients of a polynomial of degree below `k`, the constant first.
type Digits = [u32; KMAX];

/// The constants of the tables of a field of `q = pᵏ` elements, and the tables, made on the first
/// use.
pub(super) struct Zech {
    /// `p`.
    p: u32,
    /// `k`.
    k: usize,
    /// `q − 1`, the order of the group of units.
    n: u32,
    /// `⌈2³² / p⌉`, for the quotients by `p` of the encodings (exact below `2¹⁶`).
    pm: u64,
    /// The terms of `xᵏ mod g`, `(j, −gⱼ mod p)`, as values.
    low: Vec<(usize, u32)>,
    /// The word of 1 in the base field's form.
    one: u32,
    /// The tables.
    tables: OnceLock<Tables>,
}

/// The tables of a field.
struct Tables {
    /// `log[e]`, the logarithm to the base `h` of the element encoded `e` (`log[0]` unused).
    log: Vec<u16>,
    /// `exp[i]`, the encoding of `hⁱ`, for `i` in `0..q − 1`.
    exp: Vec<u16>,
    /// Zech's logarithms `plus[i] = log(1 + hⁱ)`, `u16::MAX` where `1 + hⁱ = 0`: the sums of
    /// elements held as logarithms, `hˣ + hʸ = hˣ(1 + h^(y − x))`.
    plus: Vec<u16>,
}

/// The arithmetic of `GF(p)[x] / g` for an odd `p`, on the coefficients' values or, the product
/// being linear, on their words in Montgomery's form by values.
struct Small<'a> {
    p: u32,
    k: usize,
    /// `⌈2³² / p⌉`: [`Small::rem`] is exact below `2³²/p`, above `2²³` for `p < 2⁸`.
    m: u64,
    low: &'a [(usize, u32)],
}

impl Small<'_> {
    /// `x mod p`, for `x` below `2³²/p`.
    #[inline(always)]
    fn rem(&self, x: u32) -> u32 {
        let q = ((u64::from(x) * self.m) >> 32) as u32;
        x - q * self.p
    }

    /// `a·b mod g`. The sums stay below `2k²p² ≤ 2¹⁹` (`p^k ≤ 2¹⁶` and `k ≥ 2` give `kp ≤ 512`),
    /// where [`Small::rem`] is exact.
    fn mul(&self, a: &Digits, b: &Digits) -> Digits {
        let k = self.k;
        let mut t = [0u32; 2 * KMAX];
        for (i, &ai) in a[..k].iter().enumerate() {
            if ai == 0 {
                continue;
            }
            for (ti, &bj) in t[i..i + k].iter_mut().zip(&b[..k]) {
                *ti += ai * bj;
            }
        }
        // xⁱ = x^(i − k)·xᵏ from the top down.
        for i in (k..2 * k - 1).rev() {
            let c = self.rem(t[i]);
            if c == 0 {
                continue;
            }
            for &(j, l) in self.low {
                t[i - k + j] += c * l;
            }
        }
        let mut r = [0u32; KMAX];
        for (ri, &ti) in r[..k].iter_mut().zip(&t[..k]) {
            *ri = self.rem(ti);
        }
        r
    }

    /// `aᵉ mod g`, `e ≥ 1`.
    fn pow(&self, a: &Digits, e: u32) -> Digits {
        let mut r = *a;
        for i in (0..31 - e.leading_zeros()).rev() {
            r = self.mul(&r, &r);
            if (e >> i) & 1 == 1 {
                r = self.mul(&r, a);
            }
        }
        r
    }
}

/// `a·b mod g` in `GF(2)[x] / g`, the polynomials as the bits of words, `g = xᵏ + low`.
fn mul_two(a: u32, b: u32, k: usize, low: u32) -> u32 {
    let mut t = 0u64;
    let mut b = b;
    while b != 0 {
        t ^= u64::from(a) << b.trailing_zeros();
        b &= b - 1;
    }
    for i in (k..2 * k - 1).rev() {
        if (t >> i) & 1 == 1 {
            t ^= (1 << i) ^ (u64::from(low) << (i - k));
        }
    }
    t as u32
}

/// `aᵉ mod g` in `GF(2)[x] / g`, `e ≥ 1`.
fn pow_two(a: u32, e: u32, k: usize, low: u32) -> u32 {
    let mut r = a;
    for i in (0..31 - e.leading_zeros()).rev() {
        r = mul_two(r, r, k, low);
        if (e >> i) & 1 == 1 {
            r = mul_two(r, a, k, low);
        }
    }
    r
}

/// The primes of `n`, by trial division (`n < 2¹⁶`).
fn primes_of(mut n: u32) -> Vec<u32> {
    let mut r = vec![];
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            r.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        r.push(n);
    }
    r
}

/// `a⁻¹ mod m`, for `a` prime to `m`.
fn inverse_mod(a: u32, m: u32) -> u32 {
    let (mut r0, mut r1, mut s0, mut s1) = (i64::from(m), i64::from(a), 0i64, 1i64);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
    }
    debug_assert_eq!(r0, 1, "a is prime to m");
    s0.rem_euclid(i64::from(m)) as u32
}

/// The walks advance this many runs at once, so that their chains of dependent operations
/// overlap.
const RUNS: usize = 4;

/// Where a walk writes: the encoding and Zech's encoding of `1 + v` of each element at its
/// position, and the position at its encoding.
struct Out<'a> {
    exp: &'a mut [u16],
    plus: &'a mut [u16],
    log: &'a mut [u16],
}

/// The walk of a field of odd `p` on the words, the digits of an element packed in a word, `b`
/// bits each (`b·k ≤ 20` for `pᵏ ≤ 2¹⁶`), with its encoding `Σ wᵢ·pⁱ` carried along.
struct Walk<'a> {
    p: u32,
    k: usize,
    b: u32,
    /// The mask of a digit, of all the digits, and the shift of the top one.
    dm: u32,
    all: u32,
    top: u32,
    /// `pʲ`.
    pw: Digits,
    /// The word of 1.
    one: u32,
    /// For each term of `xᵏ mod g`: its shift, `pʲ`, and the products of its coefficient by the
    /// digits.
    lows: Vec<(u32, u32, Vec<u32>)>,
    /// For each nonzero coefficient of `h`: its degree and its products by the digits.
    hs: Vec<(usize, Vec<u32>)>,
    small: &'a Small<'a>,
}

impl<'a> Walk<'a> {
    fn new(z: &Zech, small: &'a Small<'a>, h: &Digits) -> Walk<'a> {
        let (p, k) = (z.p, z.k);
        let b = 32 - (p - 1).leading_zeros();
        let mut pw = [0u32; KMAX];
        pw[0] = 1;
        for j in 1..k {
            pw[j] = pw[j - 1] * p;
        }
        let products = |c: u32| (0..p).map(|t| c * t % p).collect::<Vec<u32>>();
        Walk {
            p,
            k,
            b,
            dm: (1 << b) - 1,
            all: (1 << (b * k as u32)) - 1,
            top: b * (k as u32 - 1),
            pw,
            one: z.one,
            lows: z
                .low
                .iter()
                .map(|&(j, c)| (b * j as u32, pw[j], products(c)))
                .collect(),
            hs: (0..k)
                .filter(|&j| h[j] != 0)
                .map(|j| (j, products(h[j])))
                .collect(),
            small,
        }
    }

    /// `x·v` of the packed digits `v` and its encoding: the digits move up, the top one `t` comes
    /// back as `t·(xᵏ mod g)`, the encoding following each digit that changes.
    #[inline(always)]
    fn step(&self, v: u32, e: u32) -> (u32, u32) {
        let p = self.p;
        let t = v >> self.top;
        let mut v = (v << self.b) & self.all;
        let mut e = e.wrapping_sub(t * self.pw[self.k - 1]).wrapping_mul(p);
        for (s, pj, ct) in &self.lows {
            let u = (v >> s) & self.dm;
            let w = u + ct[t as usize];
            let w = if w >= p { w - p } else { w };
            let d = w.wrapping_sub(u);
            v = v.wrapping_add(d << s);
            e = e.wrapping_add(d.wrapping_mul(*pj));
        }
        (v, e)
    }

    /// The encoding of `1 + v`.
    #[inline(always)]
    fn plus_one(&self, v: u32, e: u32) -> u16 {
        let w0 = v & self.dm;
        let s = w0 + self.one;
        let s = if s >= self.p { s - self.p } else { s };
        (e - w0 + s) as u16
    }

    /// The words of a polynomial of values: the values times `R mod p`.
    fn words(&self, c: &Digits) -> Digits {
        let mut w = [0u32; KMAX];
        for (w, &c) in w[..self.k].iter_mut().zip(&c[..self.k]) {
            *w = self.small.rem(c * self.one);
        }
        w
    }

    /// The packed digits of the words `w` and their encoding.
    fn pack(&self, w: &Digits) -> (u32, u32) {
        let (mut v, mut e) = (0, 0);
        for i in (0..self.k).rev() {
            v = (v << self.b) | w[i];
            e = e * self.p + w[i];
        }
        (v, e)
    }

    /// `v·h` of the packed digits `v`, and its encoding: `Σ hⱼ·xʲ·v`, digit by digit.
    fn times_h(&self, v: u32) -> (u32, u32) {
        let (mut acc, mut e) = (0u32, 0u32);
        let (mut xv, mut j) = (v, 0);
        for (d, ct) in &self.hs {
            while j < *d {
                xv = self.step(xv, 0).0;
                j += 1;
            }
            for i in 0..self.k {
                let s = self.b * i as u32;
                let u = (acc >> s) & self.dm;
                let w = u + ct[((xv >> s) & self.dm) as usize];
                let w = if w >= self.p { w - self.p } else { w };
                let d = w.wrapping_sub(u);
                acc = acc.wrapping_add(d << s);
                e = e.wrapping_add(d.wrapping_mul(self.pw[i]));
            }
        }
        (acc, e)
    }

    /// The starts `start·stepᶜ` of the pieces of a coset (`steps` as values), packed.
    fn pieces(&self, start: &Digits, steps: &[Digits; RUNS]) -> ([u32; RUNS], [u32; RUNS]) {
        let mut v = [0u32; RUNS];
        let mut e = [0u32; RUNS];
        for c in 0..RUNS {
            (v[c], e[c]) = self.pack(&self.small.mul(start, &steps[c]));
        }
        (v, e)
    }

    /// `C` runs of `len` products by `x` at once, run `c` from `(v[c], e[c])` at the position
    /// `l[c]`, which advances by `ds` modulo `n`; their ends.
    #[allow(clippy::too_many_arguments)]
    fn run<const C: usize>(
        &self,
        mut v: [u32; C],
        mut e: [u32; C],
        mut l: [u32; C],
        ds: u32,
        n: u32,
        len: usize,
        out: &mut Out<'_>,
    ) -> ([u32; C], [u32; C], [u32; C]) {
        for _ in 0..len {
            for c in 0..C {
                let at = l[c] as usize;
                out.exp[at] = e[c] as u16;
                out.plus[at] = self.plus_one(v[c], e[c]);
                out.log[e[c] as usize] = l[c] as u16;
                (v[c], e[c]) = self.step(v[c], e[c]);
                l[c] = if l[c] + ds >= n {
                    l[c] + ds - n
                } else {
                    l[c] + ds
                };
            }
        }
        (v, e, l)
    }
}

impl Zech {
    /// The constants of the tables of the field, for at most [`max_order`] elements over a prime
    /// held in one word (Montgomery's form of one limb, or GF(2)); `None` otherwise. The tables are
    /// made on their first use.
    pub(super) fn new(f: &ExtensionField) -> Option<Zech> {
        let q = &f.q;
        if f.k < 2 || q.bits() > 17 {
            return None;
        }
        if !matches!(f.base.form_ref(), Form::Two(_) | Form::Odd(Prime::Mont1(_))) {
            return None;
        }
        let q = limbs_of(q)[0];
        if q > max_order(f.k) {
            return None;
        }
        // p ≤ 2⁸ (k ≥ 2), q ≤ 2¹⁶.
        let p = limbs_of(f.base.modulus())[0] as u32;
        let low = f
            .low
            .iter()
            .map(|(j, h)| (*j, limbs_of(h).first().copied().unwrap_or(0) as u32))
            .collect();
        Some(Zech {
            p,
            k: f.k,
            n: (q - 1) as u32,
            pm: (1u64 << 32).div_ceil(u64::from(p)),
            low,
            one: inline::<1>(f.base.one().raw())[0] as u32,
            tables: OnceLock::new(),
        })
    }

    /// The tables, made on the first call.
    #[inline(always)]
    fn t(&self) -> &Tables {
        self.tables.get_or_init(|| self.make())
    }

    /// The tables: `h`, then the walk through the group.
    #[cold]
    fn make(&self) -> Tables {
        if self.p == 2 {
            self.make_two()
        } else {
            self.make_odd()
        }
    }

    /// The tables for `p = 2`: the powers of `h` by products by it, the polynomials as bits (the
    /// words of GF(2) are the bits, so a polynomial's bits are its encoding), in [`RUNS`] runs
    /// from `h^(c·len)` at once.
    fn make_two(&self) -> Tables {
        let (k, n) = (self.k, self.n);
        let low = self.low.iter().fold(0u32, |m, &(j, _)| m | 1 << j);
        let primes = primes_of(n);
        // The first primitive element past the constants 0 and 1.
        let h = (2..=n)
            .find(|&h| primes.iter().all(|&r| pow_two(h, n / r, k, low) != 1))
            .expect("a finite field has a primitive element");
        let d = (31 - h.leading_zeros()) as usize;
        // v·h: the shifted copies of v, then the terms xᵐ, m ≥ k, of degree below k + d.
        let times_h = |v: u32| -> u32 {
            let mut t = 0u32;
            let mut b = h;
            while b != 0 {
                t ^= v << b.trailing_zeros();
                b &= b - 1;
            }
            for m in (k..k + d).rev() {
                let bit = (t >> m) & 1;
                t ^= bit.wrapping_neg() & ((1 << m) | (low << (m - k)));
            }
            t
        };
        let n = n as usize;
        let mut log = vec![0u16; n + 1];
        let mut exp = vec![0u16; n];
        // The encodings of 1 + hⁱ, made logarithms by `finish`.
        let mut plus = vec![0u16; n];
        let len = n / RUNS;
        let mut v = [1u32; RUNS];
        if len > 0 {
            for (c, v) in v.iter_mut().enumerate().skip(1) {
                *v = pow_two(h, (c * len) as u32, k, low);
            }
        }
        for i in 0..len {
            for (c, v) in v.iter_mut().enumerate() {
                let l = c * len + i;
                exp[l] = *v as u16;
                log[*v as usize] = l as u16;
                plus[l] = (*v ^ 1) as u16;
                *v = times_h(*v);
            }
        }
        // The last run ends at h^(RUNS·len); the few powers left.
        let mut w = if len > 0 { v[RUNS - 1] } else { 1 };
        for l in RUNS * len..n {
            exp[l] = w as u16;
            log[w as usize] = l as u16;
            plus[l] = (w ^ 1) as u16;
            w = times_h(w);
        }
        Self::finish(log, exp, plus)
    }

    /// The tables for odd `p`, by the cosets of the powers of `x` (see the module's text), each
    /// walk in [`RUNS`] runs at once: the powers of `x` in four pieces, then the other cosets four
    /// at a time (and the few left each in four pieces).
    fn make_odd(&self) -> Tables {
        let (p, k, n) = (self.p, self.k, self.n);
        let small = Small {
            p,
            k,
            m: self.pm,
            low: &self.low,
        };
        let primes = primes_of(n);
        let mut unit = [0u32; KMAX];
        unit[0] = 1;
        // The order m of x, from the primes of q − 1.
        let mut x = [0u32; KMAX];
        x[1] = 1;
        let mut m = n;
        for &r in &primes {
            while m.is_multiple_of(r) && small.pow(&x, m / r) == unit {
                m /= r;
            }
        }
        let r = n / m;
        // The first primitive element in the order of Σ cᵢpⁱ: x when it is one, else the first
        // past x and the constants (whose orders divide p − 1).
        let h = if r == 1 {
            x
        } else {
            (p + 1..=n)
                .map(|c| {
                    let mut h = [0u32; KMAX];
                    let mut c = c;
                    for d in &mut h[..k] {
                        *d = c % p;
                        c /= p;
                    }
                    h
                })
                .find(|h| primes.iter().all(|&r| small.pow(h, n / r) != unit))
                .expect("a finite field has a primitive element")
        };
        let w = Walk::new(self, &small, &h);
        let nn = n as usize;
        let mut log = vec![0u16; nn + 1];
        // The powers xᵗ, t < m, at the positions t, their logarithms when x is primitive; their
        // positions in `log` for now.
        let mm = m as usize;
        let mut xe = vec![0u16; mm];
        let mut xp = vec![0u16; mm];
        let len = mm / RUNS;
        // x^(c·len), c < RUNS, as values.
        let steps: [Digits; RUNS] = core::array::from_fn(|c| {
            if c == 0 || len == 0 {
                unit
            } else {
                small.pow(&x, (c * len) as u32)
            }
        });
        {
            let mut out = Out {
                exp: &mut xe,
                plus: &mut xp,
                log: &mut log,
            };
            let (v, e) = w.pieces(&w.words(&unit), &steps);
            let l: [u32; RUNS] = core::array::from_fn(|c| (c * len) as u32);
            let (v, e, _) = w.run(v, e, l, 1, m, len, &mut out);
            // The last piece ends at x^(RUNS·len); the few powers left.
            let rest = mm - RUNS * len;
            let from = if len > 0 {
                ([v[RUNS - 1]], [e[RUNS - 1]])
            } else {
                ([self.one], [self.one])
            };
            w.run(from.0, from.1, [(RUNS * len) as u32], 1, m, rest, &mut out);
        }
        if r == 1 {
            return Self::finish(log, xe, xp);
        }
        // h^r = x^a; the words of h^r are its values times R mod p; x = h^s.
        let hr = w.words(&small.pow(&h, r));
        let a = u32::from(log[w.pack(&hr).1 as usize]);
        let s = ((u64::from(r) * u64::from(inverse_mod(a, m))) % u64::from(n)) as u32;
        let mut exp = vec![0u16; nn];
        let mut plus = vec![0u16; nn];
        let mut l = 0;
        for (&e, &z) in xe.iter().zip(&xp) {
            exp[l as usize] = e;
            plus[l as usize] = z;
            log[e as usize] = l as u16;
            l = if l + s >= n { l + s - n } else { l + s };
        }
        drop((xe, xp));
        let mut out = Out {
            exp: &mut exp,
            plus: &mut plus,
            log: &mut log,
        };
        // The cosets hʲ·⟨x⟩, j = 1, …, r − 1: RUNS groups of `per` consecutive ones, run c from
        // h^(1 + c·per), each coset's start the last one's times h; the element hʲ·xᵗ at j + s·t.
        let per = (r - 1) / RUNS as u32;
        if per > 0 {
            let first: [u32; RUNS] = core::array::from_fn(|c| 1 + c as u32 * per);
            let mut start: [(u32, u32); RUNS] =
                core::array::from_fn(|c| w.pack(&w.words(&small.pow(&h, first[c]))));
            for j in 0..per {
                let v: [u32; RUNS] = core::array::from_fn(|c| start[c].0);
                let e: [u32; RUNS] = core::array::from_fn(|c| start[c].1);
                let l: [u32; RUNS] = core::array::from_fn(|c| first[c] + j);
                w.run(v, e, l, s, n, mm, &mut out);
                for st in &mut start {
                    *st = w.times_h(st.0);
                }
            }
        }
        // The cosets left, each in RUNS pieces.
        for j in 1 + RUNS as u32 * per..r {
            let (v, e) = w.pieces(&w.words(&small.pow(&h, j)), &steps);
            let l: [u32; RUNS] = core::array::from_fn(|c| {
                ((u64::from(j) + u64::from(s) * (c * len) as u64) % u64::from(n)) as u32
            });
            let (v, e, l) = w.run(v, e, l, s, n, len, &mut out);
            let rest = mm - RUNS * len;
            let from = if len > 0 {
                ([v[RUNS - 1]], [e[RUNS - 1]], [l[RUNS - 1]])
            } else {
                let (v, e) = w.pack(&w.words(&small.pow(&h, j)));
                ([v], [e], [j])
            };
            w.run(from.0, from.1, from.2, s, n, rest, &mut out);
        }
        Self::finish(log, exp, plus)
    }

    /// The tables, Zech's from the encodings of `1 + hⁱ` in `plus`.
    fn finish(log: Vec<u16>, exp: Vec<u16>, mut plus: Vec<u16>) -> Tables {
        for z in &mut plus {
            *z = if *z == 0 { u16::MAX } else { log[*z as usize] };
        }
        Tables { log, exp, plus }
    }

    /// The encoding of the coefficients `c`.
    #[inline(always)]
    fn encode(&self, c: &[Fp<'_>]) -> u32 {
        c.iter()
            .rev()
            .fold(0u32, |e, x| e * self.p + inline::<1>(x.raw())[0] as u32)
    }

    /// The coefficients of the encoding `e`.
    #[inline(always)]
    fn decode<'f>(&self, f: &'f ExtensionField, e: u16) -> Vec<Fp<'f>> {
        let mut e = u64::from(e);
        let p = u64::from(self.p);
        let mut c = Vec::with_capacity(self.k);
        for _ in 0..self.k {
            // e < 2¹⁶ and p < 2¹⁶: ⌈2³²/p⌉ gives the exact quotient.
            let d = (e * self.pm) >> 32;
            c.push(Fp::from_raw(&f.base, held(&[e - d * p])));
            e = d;
        }
        c
    }

    /// The logarithm of the element, or `None` for 0.
    #[inline(always)]
    fn log_of(&self, a: &Fpk<'_>) -> Option<u32> {
        match a.r {
            Repr::Log(l, _) => (l != ZERO).then_some(l),
            _ => unreachable!("an element of a field of Zech's tables is a logarithm"),
        }
    }

    /// `hⁱ`, for `i` below `2(q − 1)`.
    #[inline(always)]
    fn power<'f>(&self, f: &'f ExtensionField, i: u32) -> Fpk<'f> {
        Fpk::of_log(f, if i >= self.n { i - self.n } else { i })
    }

    /// The logarithm of the coefficients `c`.
    pub(super) fn log_of_coeffs(&self, c: &[Fp<'_>]) -> u32 {
        let e = self.encode(c);
        if e == 0 {
            ZERO
        } else {
            u32::from(self.t().log[e as usize])
        }
    }

    /// The logarithm of the coefficients' words `w`.
    pub(super) fn log_of_words(&self, w: &[u64]) -> u32 {
        let e = w.iter().rev().fold(0u32, |e, &x| e * self.p + x as u32);
        if e == 0 {
            ZERO
        } else {
            u32::from(self.t().log[e as usize])
        }
    }

    /// The coefficients of the element of the logarithm `l`.
    pub(super) fn coeffs_of_log<'f>(&self, f: &'f ExtensionField, l: u32) -> Vec<Fp<'f>> {
        if l == ZERO {
            return vec![f.base.zero(); self.k];
        }
        self.decode(f, self.t().exp[l as usize])
    }

    /// The product of two logarithms.
    #[inline(always)]
    pub(super) fn mul_log(&self, x: u32, y: u32) -> u32 {
        if x == ZERO || y == ZERO {
            return ZERO;
        }
        let s = x + y;
        if s >= self.n { s - self.n } else { s }
    }

    /// The sum of two logarithms: `hˣ + hʸ = h^(x + log(1 + h^(y − x)))`.
    #[inline(always)]
    pub(super) fn add_log(&self, x: u32, y: u32) -> u32 {
        if x == ZERO {
            return y;
        }
        if y == ZERO {
            return x;
        }
        let d = if y >= x { y - x } else { y + self.n - x };
        let z = self.t().plus[d as usize];
        if z == u16::MAX {
            return ZERO;
        }
        let s = x + u32::from(z);
        if s >= self.n { s - self.n } else { s }
    }

    /// The negation of a logarithm: `−1 = h^((q − 1)/2)` for odd `p`, `−a = a` for `p = 2`.
    #[inline(always)]
    pub(super) fn neg_log(&self, x: u32) -> u32 {
        if x == ZERO || self.p == 2 {
            return x;
        }
        let s = x + self.n / 2;
        if s >= self.n { s - self.n } else { s }
    }

    /// `e mod (q − 1)`.
    fn reduce(&self, e: &Number) -> u32 {
        let n = u128::from(self.n);
        limbs_of(e)
            .iter()
            .rev()
            .fold(0u128, |r, &l| ((r << 64) | u128::from(l)) % n) as u32
    }

    pub(super) fn inverse<'f>(&self, a: &Fpk<'f>) -> Option<Fpk<'f>> {
        self.log_of(a).map(|x| self.power(a.field(), self.n - x))
    }

    /// `aᵉ` for `e ≥ 0`, `a⁰ = 1`.
    pub(super) fn pow<'f>(&self, a: &Fpk<'f>, e: &Number) -> Fpk<'f> {
        let f = a.field();
        if e.is_zero() {
            return f.one();
        }
        match self.log_of(a) {
            Some(x) => {
                let e = u64::from(self.reduce(e));
                self.power(f, ((u64::from(x) * e) % u64::from(self.n)) as u32)
            }
            None => f.zero(),
        }
    }

    /// `a^(pⁱ)`.
    pub(super) fn frobenius<'f>(&self, a: &Fpk<'f>, i: usize) -> Fpk<'f> {
        let f = a.field();
        let n = u64::from(self.n);
        match self.log_of(a) {
            Some(x) => {
                let mut e = u64::from(x);
                for _ in 0..i {
                    e = (e * u64::from(self.p)) % n;
                }
                self.power(f, e as u32)
            }
            None => f.zero(),
        }
    }

    /// `N(a) = a^((q − 1)/(p − 1))`, an element of `GF(p)`.
    pub(super) fn norm<'f>(&self, a: &Fpk<'f>) -> Fp<'f> {
        let f = a.field();
        match self.log_of(a) {
            Some(x) => {
                let n = u64::from(self.n);
                let e = (u64::from(x) * (n / u64::from(self.p - 1))) % n;
                let c = self.decode(f, self.t().exp[e as usize]);
                c.into_iter().next().expect("k ≥ 2")
            }
            None => f.base.zero(),
        }
    }

    /// The quadratic character for odd `p`: 0 for 0, 1 for an even logarithm, −1 for an odd one.
    pub(super) fn legendre(&self, a: &Fpk<'_>) -> i32 {
        match self.log_of(a) {
            None => 0,
            Some(x) if x % 2 == 0 => 1,
            Some(_) => -1,
        }
    }

    /// For odd `p`, the smaller square root (the rule of [`Fpk::sqrt`]: the root whose highest
    /// nonzero coefficient is below `p/2`), or `None` for a non-square: `h^(x/2)` or its negation
    /// `h^(x/2 + (q − 1)/2)`, chosen by the value of the top nonzero word of `h^(x/2)`.
    pub(super) fn sqrt_smaller<'f>(&self, a: &Fpk<'f>) -> Option<Fpk<'f>> {
        let x = self.log_of(a)?;
        if x % 2 != 0 {
            return None;
        }
        let f = a.field();
        let r = x / 2;
        let p = u64::from(self.p);
        // The words of h^r from the bottom; the last nonzero one is the top coefficient's.
        let mut e = u64::from(self.t().exp[r as usize]);
        let mut top = 0u64;
        for _ in 0..self.k {
            let d = (e * self.pm) >> 32;
            let w = e - d * p;
            if w != 0 {
                top = w;
            }
            e = d;
        }
        let (fp, minv) = f.word;
        let value = if fp == 0 {
            top
        } else {
            // REDC of the residue: its value.
            let m = top.wrapping_mul(minv);
            let t = ((u128::from(top) + u128::from(m) * u128::from(fp)) >> 64) as u64;
            if t >= fp { t - fp } else { t }
        };
        Some(if 2 * value > p {
            Fpk::of_log(f, self.neg_log(r))
        } else {
            Fpk::of_log(f, r)
        })
    }

    /// The multiplicative order, `(q − 1)/gcd(x, q − 1)`.
    pub(super) fn order(&self, a: &Fpk<'_>) -> Option<Number> {
        let x = self.log_of(a)?;
        let (mut u, mut v) = (self.n, x);
        while v != 0 {
            (u, v) = (v, u % v);
        }
        Some(Number::from(self.n / u))
    }
}
