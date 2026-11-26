//! The arithmetic the factorisation over a prime field takes, on the coefficients of a
//! [`ModPolynomial`] (packed, `k` limbs each, trimmed): the word kernels of [`nmod`](super::nmod)
//! for a modulus of one word; for a longer one those of [`zn`](super::zn) and the arith area's
//! remainders modulo a fixed `f` (`ModInverse`), with the Brent–Kung blocks summed in `2k + 1`
//! limbs and reduced once.

use super::super::ModPolynomial;
use super::super::arith::mdiv::ModInverse;
use super::super::arith::mmul::LimbMod;
use super::super::modular::{Modulus, Work};
use super::nmod::{self, Modf, Powers, Wm};
use super::zn;
use crate::math::numbers::{ByteOrder, Number, from_limbs, limbs_of, nat};
use std::sync::Arc;

/// Polynomials modulo a prime `p`: the coefficients as a [`ModPolynomial`] packs them.
pub(crate) type P = Vec<u64>;

/// The field `F_p` and its polynomials.
#[derive(Clone)]
pub(crate) struct Fp {
    /// The modulus.
    pub(crate) m: Arc<Modulus>,
    /// The modulus as a word, when it is one.
    pub(crate) w: Option<Wm>,
}

/// A monic modulus `f` of degree at least 1, with what its remainders need: the inverse of its
/// reversal, kept.
pub(crate) enum Mf {
    /// Modulo a word.
    W(Modf),
    /// Modulo a longer `p`.
    B(ModInverse),
}

/// The baby steps of a Brent–Kung composition with one `h` modulo one `f`.
pub(crate) enum Pw {
    /// Modulo a word.
    W(Powers),
    /// Modulo a longer `p`: `1, h, …, h^(m-1)`, `h^m` and `deg f`.
    B(Vec<P>, P, usize),
}

impl Fp {
    /// The field of the modulus `m`, a prime.
    pub(crate) fn new(m: Arc<Modulus>) -> Fp {
        let w = m.word_mod().copied();
        Fp { m, w }
    }

    /// The limbs of a coefficient.
    #[inline]
    pub(crate) fn k(&self) -> usize {
        self.m.k()
    }

    /// The number of coefficients of `a`.
    #[inline]
    pub(crate) fn len(&self, a: &[u64]) -> usize {
        a.len() / self.k()
    }

    /// The degree of `a`, 0 for the zero polynomial too.
    #[inline]
    pub(crate) fn deg(&self, a: &[u64]) -> usize {
        self.len(a).saturating_sub(1)
    }

    /// Drops the zero coefficients at the top of `a`.
    pub(crate) fn trim(&self, a: &mut P) {
        let k = self.k();
        while a.len() >= k && a[a.len() - k..].iter().all(|&l| l == 0) {
            a.truncate(a.len() - k);
        }
    }

    /// The polynomial of the coefficients `a`.
    #[inline]
    pub(crate) fn wrap(&self, a: P) -> ModPolynomial {
        ModPolynomial::of(self.m.clone(), a)
    }

    /// The constant 1.
    pub(crate) fn one(&self) -> P {
        let mut c = vec![0u64; self.k()];
        c[0] = 1;
        c
    }

    /// `x`.
    pub(crate) fn x(&self) -> P {
        let mut c = vec![0u64; 2 * self.k()];
        c[self.k()] = 1;
        c
    }

    /// Whether `a` is the constant 1.
    #[inline]
    pub(crate) fn is_one(&self, a: &[u64]) -> bool {
        a.len() == self.k() && a[0] == 1 && a[1..].iter().all(|&l| l == 0)
    }

    /// `a + b`.
    pub(crate) fn add(&self, a: &[u64], b: &[u64]) -> P {
        if let Some(w) = self.w {
            let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
            let mut r = a.to_vec();
            for (x, y) in r.iter_mut().zip(b) {
                *x = w.add(*x, *y);
            }
            nmod::trim(&mut r);
            return r;
        }
        let mut r = self.wrap(a.to_vec());
        r += &self.wrap(b.to_vec());
        r.c.into_vec()
    }

    /// `a - b`.
    pub(crate) fn sub(&self, a: &[u64], b: &[u64]) -> P {
        if let Some(w) = self.w {
            let mut r = a.to_vec();
            if r.len() < b.len() {
                r.resize(b.len(), 0);
            }
            for (x, y) in r.iter_mut().zip(b) {
                *x = w.sub(*x, *y);
            }
            nmod::trim(&mut r);
            return r;
        }
        let mut r = self.wrap(a.to_vec());
        r -= &self.wrap(b.to_vec());
        r.c.into_vec()
    }

    /// `a - x`.
    pub(crate) fn sub_x(&self, a: &[u64]) -> P {
        self.sub(a, &self.x())
    }

    /// `a - 1`.
    pub(crate) fn sub_one(&self, a: &[u64]) -> P {
        self.sub(a, &self.one())
    }

    /// `a·b`.
    pub(crate) fn mul(&self, a: &[u64], b: &[u64]) -> P {
        if let Some(w) = self.w {
            return nmod::mul(&w, a, b);
        }
        zn::mul(&self.m, a, b)
    }

    /// The quotient and the remainder of `a` by `b`, not zero.
    pub(crate) fn divrem(&self, a: &[u64], b: &[u64]) -> (P, P) {
        if let Some(w) = self.w {
            return nmod::divrem(&w, a, b);
        }
        zn::divrem(&self.m, a, b)
    }

    /// The quotient of `a` by `b`, a divisor of it.
    pub(crate) fn div(&self, a: &[u64], b: &[u64]) -> P {
        let (q, r) = self.divrem(a, b);
        debug_assert!(r.is_empty());
        q
    }

    /// The remainder of `a` by `b`, not zero.
    pub(crate) fn rem(&self, a: &[u64], b: &[u64]) -> P {
        if let Some(w) = self.w {
            return nmod::rem(&w, a, b);
        }
        zn::divrem(&self.m, a, b).1
    }

    /// The monic greatest common divisor (Euclid's algorithm: the gcd seam, see [`nmod::gcd`]
    /// and [`zn::gcd`]).
    pub(crate) fn gcd(&self, a: &[u64], b: &[u64]) -> P {
        if let Some(w) = self.w {
            return nmod::gcd(&w, a, b);
        }
        zn::gcd(&self.m, a, b)
    }

    /// The derivative.
    pub(crate) fn derivative(&self, a: &[u64]) -> P {
        if let Some(w) = self.w {
            let mut r: P = a
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &c)| w.mul(c, w.reduce1(i as u64)))
                .collect();
            nmod::trim(&mut r);
            return r;
        }
        self.wrap(a.to_vec()).derivative().c.into_vec()
    }

    /// The polynomial of every `p`-th coefficient of `a`, `a = b(x^p)` with `b` returned, when the
    /// derivative of `a` is 0: its `p`-th root, as `cᵖ = c` in `F_p`.
    pub(crate) fn pth_root(&self, a: &[u64]) -> P {
        let k = self.k();
        let p = self
            .m
            .word()
            .map_or(usize::MAX, |p| usize::try_from(p).unwrap_or(usize::MAX));
        let n = self.len(a);
        let mut r = Vec::new();
        let mut i = 0usize;
        while i < n {
            r.extend_from_slice(&a[i * k..(i + 1) * k]);
            i = match i.checked_add(p) {
                Some(j) => j,
                None => break,
            };
        }
        r
    }

    /// The modulus `f`, monic, of degree at least 1.
    pub(crate) fn mf(&self, f: &[u64]) -> Mf {
        match &self.w {
            Some(w) => Mf::W(Modf::new(w, f)),
            None => Mf::B(ModInverse::new(&self.wrap(f.to_vec()))),
        }
    }

    /// `a mod f`.
    pub(crate) fn rem_mf(&self, a: &[u64], mf: &Mf) -> P {
        match mf {
            Mf::W(m) => m.rem(a),
            Mf::B(d) => d.rem(&self.wrap(a.to_vec())).c.into_vec(),
        }
    }

    /// `a·b mod f` for `a` and `b` reduced.
    pub(crate) fn mulmod(&self, a: &[u64], b: &[u64], mf: &Mf) -> P {
        match mf {
            Mf::W(m) => m.mulmod(a, b),
            Mf::B(d) => d
                .rem(&self.wrap(zn::mul_raw(&self.m, a, Some(b))))
                .c
                .into_vec(),
        }
    }

    /// `aᵉ mod f` for `a` reduced and the exponent of the limbs `e`.
    pub(crate) fn powmod(&self, a: &[u64], e: &[u64], mf: &Mf) -> P {
        match mf {
            Mf::W(m) => m.powmod(a, e),
            Mf::B(d) => d
                .pow_mod(
                    &self.wrap(a.to_vec()),
                    &from_limbs(e, ByteOrder::LittleEndian),
                )
                .c
                .into_vec(),
        }
    }

    /// `xᵉ mod f` for the exponent of the limbs `e`.
    pub(crate) fn powmod_x(&self, e: &[u64], mf: &Mf) -> P {
        match mf {
            Mf::W(m) => m.powmod_x(e),
            Mf::B(_) => {
                let x = self.rem_mf(&self.x(), mf);
                self.powmod(&x, e, mf)
            }
        }
    }

    /// `x^p mod f`.
    pub(crate) fn frobenius(&self, mf: &Mf) -> P {
        let e = self.m.d.clone();
        self.powmod_x(&e, mf)
    }

    /// The baby steps of the compositions with `h` modulo `f` of polynomials of `len`
    /// coefficients: `m` of them (`⌈√len⌉` when `None`).
    pub(crate) fn powers(&self, h: &[u64], mf: &Mf, len: usize, m: Option<usize>) -> Pw {
        match mf {
            Mf::W(f) => Pw::W(Powers::new(h, f, len, m)),
            Mf::B(d) => {
                let m = m
                    .unwrap_or_else(|| (len.max(1) as f64).sqrt().ceil() as usize)
                    .max(1);
                let h = self.rem_mf(h, mf);
                let mut p = self.rem_mf(&self.one(), mf);
                let mut rows = Vec::with_capacity(m);
                for _ in 0..m {
                    let next = self.mulmod(&p, &h, mf);
                    rows.push(p);
                    p = next;
                }
                Pw::B(rows, p, self.deg(&d.f.c))
            }
        }
    }

    /// `g(h) mod f` with the baby steps of `h`.
    pub(crate) fn compose(&self, g: &[u64], pw: &Pw, mf: &Mf) -> P {
        match (pw, mf, &self.w) {
            (Pw::W(p), Mf::W(m), Some(w)) => p.compose(w, g, m),
            (Pw::B(rows, hm, cols), Mf::B(..), _) => {
                let k = self.k();
                let m = rows.len();
                let n = self.len(g);
                if n == 0 {
                    return Vec::new();
                }
                let blocks = n.div_ceil(m);
                let red = LimbMod::new(&self.m.d, self.m.l.as_ref().expect("several limbs"));
                // Each coefficient of a block a sum of up to m products of residues, in 2k
                // limbs and the bits of m, reduced once.
                let width = 2 * k + 1;
                let mut acc = vec![0u64; width];
                let mut t = vec![0u64; 2 * k];
                let mut scratch = Vec::new();
                let mut block = |j: usize| {
                    let mut c = vec![0u64; cols * k];
                    let lo = j * m;
                    let hi = ((j + 1) * m).min(n);
                    for (col, o) in c.chunks_exact_mut(k).enumerate() {
                        acc.fill(0);
                        let mut any = false;
                        for (idx, row) in (lo..hi).zip(rows) {
                            let a = &g[idx * k..(idx + 1) * k];
                            let Some(b) = row.get(col * k..(col + 1) * k) else {
                                continue;
                            };
                            if a.iter().all(|&l| l == 0) || b.iter().all(|&l| l == 0) {
                                continue;
                            }
                            nat::mul_to(&mut t, a, b, &mut scratch);
                            let carry = nat::add_assign(&mut acc, &t);
                            debug_assert!(!carry);
                            any = true;
                        }
                        if any {
                            red.reduce(&acc, o);
                        }
                    }
                    self.trim(&mut c);
                    c
                };
                let mut r = block(blocks - 1);
                for j in (0..blocks - 1).rev() {
                    r = self.mulmod(&r, hm, mf);
                    r = self.add(&r, &block(j));
                }
                r
            }
            _ => unreachable!(),
        }
    }

    /// A pseudo-random polynomial of degree below `n`, not constant when `n ≥ 2`, from the
    /// state of a xorshift generator.
    pub(crate) fn random(&self, state: &mut u64, n: usize) -> P {
        let k = self.k();
        loop {
            let mut c = Vec::with_capacity(n * k);
            let mut limbs = vec![0u64; k + 1];
            let mut work = Work::default();
            let mut r = vec![0u64; k];
            for _ in 0..n {
                for l in limbs.iter_mut() {
                    *l = xorshift(state);
                }
                self.m.reduce(&mut r, &limbs, &mut work);
                c.extend_from_slice(&r);
            }
            self.trim(&mut c);
            if n < 2 || self.len(&c) >= 2 {
                return c;
            }
        }
    }

    /// The limbs of `n`.
    pub(crate) fn limbs(n: &Number) -> Vec<u64> {
        limbs_of(n)
    }
}

/// The xorshift generator of the seeds of the probabilistic splittings.
#[inline]
pub(crate) fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
