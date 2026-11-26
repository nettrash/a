//! Division of polynomials modulo `n` by Newton's method: the inverse of a power series, the
//! quotient from the reversed divisor's inverse, and the remainder modulo a fixed `f` with that
//! inverse computed once ([`ModInverse`]), on which the powers modulo `f` and Brent and Kung's
//! composition modulo `f` run; composition by divide and conquer.

use super::super::modular::ModPolynomial;
use super::wmul;
use crate::math::numbers::Number;

/// Up to this many coefficients, a power series modulo several limbs is inverted by the
/// schoolbook recurrence (modulo a word, see [`super::wrem::inv_series`]).
fn inv_school_max() -> usize {
    if cfg!(miri) { 8 } else { 16 }
}

/// Up to this many coefficients in the quotient or the divisor, the schoolbook division of one
/// division (which Newton's method would make invert the divisor first): 256 modulo a word, 128
/// modulo up to 4 limbs, 32 beyond, as the schoolbook's products of residues grow as `k²`.
fn div_school_max(k: usize) -> usize {
    if cfg!(miri) {
        8
    } else {
        match k {
            1 => 256,
            2..=4 => 128,
            _ => 32,
        }
    }
}

/// Up to this many coefficients in `f`, remainders modulo a fixed `f` of several limbs by the
/// schoolbook division; beyond, by the inverse of its reversal, computed once.
fn rem_school_max() -> usize {
    if cfg!(miri) { 8 } else { 32 }
}

impl ModPolynomial {
    /// The remainder by `x^n`: the coefficients below `x^n`.
    pub(crate) fn low(&self, n: usize) -> ModPolynomial {
        let k = self.m.k();
        let end = (n * k).min(self.c.len());
        ModPolynomial::of(self.m.clone(), self.c[..end].to_vec())
    }

    /// `(self·b) mod x^n`: modulo several limbs by the transforms on pieces with the theorem
    /// applied to the low `n` coefficients only, when they are the method.
    pub(crate) fn mul_low(&self, b: &ModPolynomial, n: usize) -> ModPolynomial {
        let (la, lb) = (self.length(), b.length());
        if la == 0 || lb == 0 || n == 0 {
            return self.zero_like();
        }
        if let Some(red) = self.m.l.as_ref() {
            let n_out = n.min(la + lb - 1);
            if let Some(c) = super::mmul::mul_low(&self.c, &b.c, n_out, &self.m.d, red) {
                return ModPolynomial::of(self.m.clone(), c);
            }
        }
        (self * b).low(n)
    }

    /// The quotient by `x^n`.
    pub(crate) fn high(&self, n: usize) -> ModPolynomial {
        let k = self.m.k();
        let c = self.c.get(n * k..).map_or_else(Vec::new, <[u64]>::to_vec);
        ModPolynomial {
            m: self.m.clone(),
            c: c.into(),
        }
    }

    /// The reversal of the first `n` coefficients, `x^(n-1) p(1/x)` for `p` of degree below `n`.
    pub(crate) fn rev(&self, n: usize) -> ModPolynomial {
        let k = self.m.k();
        let mut c = vec![0u64; n * k];
        for i in 0..n.min(self.length()) {
            c[(n - 1 - i) * k..(n - i) * k].copy_from_slice(self.limbs(i));
        }
        ModPolynomial::of(self.m.clone(), c)
    }

    /// The product by `x^n`.
    pub(crate) fn shl(&self, n: usize) -> ModPolynomial {
        if self.is_zero() {
            return self.clone();
        }
        let k = self.m.k();
        let mut c = vec![0u64; n * k];
        c.extend_from_slice(&self.c);
        ModPolynomial {
            m: self.m.clone(),
            c: c.into(),
        }
    }

    /// The inverse of the unit `c` modulo `n`, or the panic of a leading coefficient that is not.
    #[track_caller]
    fn unit_inverse(&self, c: &[u64], what: &str) -> Vec<u64> {
        match self.m.inv(c) {
            Some(v) => v,
            None => panic!("the {what} coefficient is not a unit modulo n"),
        }
    }

    /// The inverse of the power series modulo `x^n`: `g` with `self·g ≡ 1 (mod x^n)`, of at
    /// most `n` coefficients; the zero polynomial for `n = 0`. By the schoolbook recurrence for
    /// short series and Newton's iteration `g ← g + g (1 - self·g)`, which doubles the correct
    /// coefficients, beyond: about three products of the length `n`.
    ///
    /// # Panics
    ///
    /// Panics if the constant coefficient is not a unit modulo `n` (for `n > 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // 1 / (1 - x) = 1 + x + x² + … modulo 7 and x⁴.
    /// let p = ModPolynomial::from_u64(&[1, 6], 7);
    /// assert_eq!(p.inv_series(4), ModPolynomial::from_u64(&[1, 1, 1, 1], 7));
    /// ```
    #[track_caller]
    pub fn inv_series(&self, n: usize) -> ModPolynomial {
        if n == 0 {
            return self.zero_like();
        }
        if let Some(w) = self.m.word_mod() {
            // The constant's inverse and the series without a vector of limbs between.
            let c0 = self.c.first().copied().unwrap_or(0);
            let Some(inv) = w.inv(c0) else {
                panic!("the constant coefficient is not a unit modulo n");
            };
            let f = if self.is_zero() {
                &[0u64][..]
            } else {
                &self.c[..]
            };
            let g = super::wrem::inv_series(f, n, inv, w);
            return ModPolynomial::of(self.m.clone(), g);
        }
        let k = self.m.k();
        let c0 = if self.is_zero() {
            vec![0u64; k]
        } else {
            self.limbs(0).to_vec()
        };
        let inv = self.unit_inverse(&c0, "constant");
        self.inv_newton(n, &inv)
    }

    /// [`inv_series`](ModPolynomial::inv_series) with the inverse `inv` of the constant.
    fn inv_newton(&self, n: usize, inv: &[u64]) -> ModPolynomial {
        if let Some(w) = self.m.word_mod() {
            let f = if self.is_zero() {
                &[0u64][..]
            } else {
                &self.c[..]
            };
            let g = super::wrem::inv_series(f, n, inv[0], w);
            return ModPolynomial::of(self.m.clone(), g);
        }
        if n <= inv_school_max() {
            return self.inv_school(n, inv);
        }
        let h = n.div_ceil(2);
        let g = self.inv_newton(h, inv);
        // e = self·g mod x^n = 1 + x^h·t; then g - x^h (t·g mod x^(n - h)).
        let e = self.low(n).mul_low(&g, n);
        let t = e.high(h);
        let u = t.mul_low(&g, n - h);
        &g - &u.shl(h)
    }

    /// The inverse modulo `x^n` by the recurrence `g_j = -c0^-1 Σ f_i g_(j-i)`, for a modulus of
    /// several limbs (a word's goes to [`super::wrem::inv_series`]).
    fn inv_school(&self, n: usize, inv: &[u64]) -> ModPolynomial {
        let k = self.m.k();
        let mut g = vec![0u64; n * k];
        g[..k].copy_from_slice(inv);
        let ninv = {
            let mut v = inv.to_vec();
            self.m.neg(&mut v);
            v
        };
        let mut w = super::super::modular::Work::default();
        let mut s = vec![0u64; k];
        let mut t = vec![0u64; k];
        for j in 1..n {
            s.fill(0);
            for i in 1..=j.min(self.length().saturating_sub(1)) {
                self.m.mul(
                    &mut t,
                    self.limbs(i),
                    &g[(j - i) * k..(j - i + 1) * k],
                    &mut w,
                );
                self.m.add(&mut s, &t);
            }
            self.m.mul(&mut t, &s, &ninv, &mut w);
            g[j * k..(j + 1) * k].copy_from_slice(&t);
        }
        ModPolynomial::of(self.m.clone(), g)
    }

    /// The quotient and the remainder of the division by `d`, by the schoolbook method for short
    /// quotients or divisors and by Newton's method beyond: the reversed quotient is the reversed
    /// dividend times the inverse of the reversed divisor modulo `x^(la - ld + 1)`.
    #[track_caller]
    pub(crate) fn div_rem_fast(&self, d: &ModPolynomial) -> (ModPolynomial, ModPolynomial) {
        self.same_modulus(d);
        let ld = d.length();
        assert!(ld > 0, "division by the zero polynomial");
        if self.m.is_two() {
            // Packed in and out (every leading coefficient modulo 2 is the unit 1).
            if self.length() < ld {
                return (self.zero_like(), self.clone());
            }
            let (q, r) = super::gf2::Div::new(&d.c.bits()).div_rem(&self.c.bits());
            return (
                ModPolynomial::of_bits(self.m.clone(), q),
                ModPolynomial::of_bits(self.m.clone(), r),
            );
        }
        if let Some(w) = self.m.word_mod() {
            // The word path without a vector for the inverse of the leading coefficient.
            let Some(inv) = w.inv(d.c[ld - 1]) else {
                panic!("the leading coefficient is not a unit modulo n");
            };
            let la = self.length();
            if la < ld {
                return (self.zero_like(), self.clone());
            }
            if la - ld < div_school_max(1) || ld <= div_school_max(1) {
                let (q, r) = wmul::div_rem_school(&self.c, &d.c, inv, w);
                return (
                    ModPolynomial::of(self.m.clone(), q),
                    ModPolynomial::of(self.m.clone(), r),
                );
            }
        }
        let inv = self.unit_inverse(d.limbs(ld - 1), "leading");
        let la = self.length();
        if la < ld {
            return (self.zero_like(), self.clone());
        }
        let lq = la - ld + 1;
        let k = self.m.k();
        if lq <= div_school_max(k) || ld <= div_school_max(k) {
            return self.div_rem_school(d, &inv);
        }
        let dinv = d.rev(ld).inv_newton(lq, &inv);
        let q = self.rev(la).low(lq).mul_low(&dinv, lq).rev(lq);
        if let Some(w) = self.m.word_mod() {
            // r = a - q d below x^(ld - 1) by q d modulo x^L - 1 for L at least both lengths: the
            // coefficients of q d from ld - 1 on are those of a, so the wrapped ones are known,
            // r_j = a_j - (q d mod x^L - 1)_j + a_(j + L).
            let k = usize::BITS - (lq.max(ld) - 1).leading_zeros();
            if let Some(c) = wmul::mul_cyclic_low(&q.c, &d.c, k, ld - 1, w) {
                let (a, l) = (&self.c, 1usize << k);
                let r: Vec<u64> = (0..ld - 1)
                    .map(|j| {
                        let hi = a.get(j + l).copied().unwrap_or(0);
                        w.add(w.sub(a[j], c[j]), hi)
                    })
                    .collect();
                return (q, ModPolynomial::of(self.m.clone(), r));
            }
        }
        let r = &self.low(ld - 1) - &q.mul_low(&d.low(ld - 1), ld - 1);
        (q, r)
    }

    /// The remainder of the division by `d`, without the quotient where the method allows.
    #[track_caller]
    pub(crate) fn rem_fast(&self, d: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(d);
        let ld = d.length();
        assert!(ld > 0, "division by the zero polynomial");
        if self.m.is_two() {
            if self.length() < ld {
                return self.clone();
            }
            let r = super::gf2::Div::new(&d.c.bits()).rem(&self.c.bits());
            return ModPolynomial::of_bits(self.m.clone(), r);
        }
        if let Some(w) = self.m.word_mod() {
            let la = self.length();
            let Some(inv) = w.inv(d.c[ld - 1]) else {
                // Still a unit check, as the division makes it.
                panic!("the leading coefficient is not a unit modulo n");
            };
            if la < ld {
                return self.clone();
            }
            if ld <= div_school_max(1) || la - ld < div_school_max(1) {
                let r = wmul::rem_school(&self.c, &d.c, inv, w, None);
                return ModPolynomial::of(self.m.clone(), r);
            }
        }
        self.div_rem_fast(d).1
    }

    /// The schoolbook division by `d` with `inv` the inverse of its leading coefficient.
    fn div_rem_school(&self, d: &ModPolynomial, inv: &[u64]) -> (ModPolynomial, ModPolynomial) {
        if let Some(w) = self.m.word_mod() {
            let (q, r) = wmul::div_rem_school(&self.c, &d.c, inv[0], w);
            return (
                ModPolynomial::of(self.m.clone(), q),
                ModPolynomial::of(self.m.clone(), r),
            );
        }
        let red = self.m.l.as_ref().expect("several limbs");
        let (q, r) = super::mmul::div_rem_school(&self.c, &d.c, inv, &self.m.d, red);
        (
            ModPolynomial::of(self.m.clone(), q),
            ModPolynomial::of(self.m.clone(), r),
        )
    }

    /// The composition `self(g) mod f`, by Brent and Kung's method: with `m ≈ √(deg self)`, the
    /// powers `g⁰…gᵐ` modulo `f`, each block of `m` coefficients evaluated at `g` as a combination
    /// of them (a product of matrices whose sums of products are reduced once), and the blocks
    /// joined by Horner's rule in `gᵐ`: about `2√(deg self)` products modulo `f`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is zero, its leading coefficient is not a unit modulo `n`, or the moduli
    /// differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // Modulo 7 and x² + 1: (x² + x)(x + 1) = x² + 3x + 2 ≡ 3x + 1.
    /// let p = ModPolynomial::from_u64(&[0, 1, 1], 7);
    /// let g = ModPolynomial::from_u64(&[1, 1], 7);
    /// let f = ModPolynomial::from_u64(&[1, 0, 1], 7);
    /// assert_eq!(p.compose_mod(&g, &f), ModPolynomial::from_u64(&[1, 3], 7));
    /// ```
    #[track_caller]
    pub fn compose_mod(&self, g: &ModPolynomial, f: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(g);
        self.same_modulus(f);
        ModInverse::new(f).compose_mod(self, g)
    }

    /// The composition `self(g)`: by Horner's rule for short polynomials, and by divide and
    /// conquer beyond, `self = lo + x^h hi` giving `lo(g) + g^h hi(g)` with the powers `g^(2^i)`
    /// computed once.
    pub(crate) fn compose_fast(&self, g: &ModPolynomial) -> ModPolynomial {
        self.same_modulus(g);
        let n = self.length();
        if n <= 1 || g.length() <= 1 {
            // A constant g: the value at it.
            if n == 0 {
                return self.zero_like();
            }
            let v = self.eval(&g.coeff(0));
            return self.with_coeffs(&[v]);
        }
        if let Some(w) = self.m.word_mod() {
            return ModPolynomial::of(self.m.clone(), wmul::compose(&self.c, &g.c, w));
        }
        // g^(2^i) for 2^i below n.
        let mut pows = vec![g.clone()];
        while (1usize << pows.len()) < n {
            let last = pows.last().expect("not empty");
            pows.push(last * last);
        }
        self.compose_dc(0, n, &pows)
    }

    /// The composition of the coefficients `lo..lo + len` of `self` with `g`, `pows[i] =
    /// g^(2^i)`.
    fn compose_dc(&self, lo: usize, len: usize, pows: &[ModPolynomial]) -> ModPolynomial {
        if len <= 8 {
            let mut v = self.zero_like();
            for i in (lo..lo + len).rev() {
                v = &v * &pows[0];
                v += &ModPolynomial::of(self.m.clone(), self.limbs(i).to_vec());
            }
            return v;
        }
        // The largest power of 2 below len.
        let e = (usize::BITS - 1 - (len - 1).leading_zeros()) as usize;
        let h = 1usize << e;
        let a = self.compose_dc(lo, h, pows);
        let b = self.compose_dc(lo + h, len - h, pows);
        &a + &(&b * &pows[e])
    }
}

/// The remainder modulo a fixed `f` with the inverse of its reversal computed once, for the
/// products and powers modulo `f`: a remainder of a polynomial of degree below `2 deg f - 1` is
/// two products of the length `deg f`.
#[derive(Clone, Debug)]
pub(crate) struct ModInverse {
    /// The modulus `f`.
    pub(crate) f: ModPolynomial,
    /// The inverse of the reversal of `f` modulo `x^(deg f)`.
    finv: ModPolynomial,
    /// The inverse of the leading coefficient of `f`.
    inv: Vec<u64>,
    /// The same remainders on vectors of words, for a word modulus and `deg f ≥ 1`.
    word: Option<Box<super::wrem::WordRem>>,
    /// The transforms of the inverse and of `f` for the products of the remainders modulo a
    /// modulus of several limbs, by the transforms on pieces of the coefficients.
    ready: Option<Ready>,
}

/// The inverse of the reversed `f` and the low coefficients of `f` transformed once, for the
/// two products of each remainder: the quotient's reversal is the top of the dividend reversed
/// times the inverse (its low `lq` coefficients, the only ones the theorem is applied to), and
/// the remainder the dividend's low `deg f` coefficients less those of the quotient times `f`.
#[derive(Clone, Debug)]
struct Ready {
    finv: crate::math::numbers::nat::LimbPolyTransform,
    flow: crate::math::numbers::nat::LimbPolyTransform,
}

/// From this many coefficients in `f`, the remainders modulo a fixed `f` of several limbs take
/// the products by the transforms of the inverse and of `f`, kept.
fn ready_min() -> usize {
    if cfg!(miri) { 4 } else { 32 }
}

impl ModInverse {
    /// The remainders modulo `f`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is zero or its leading coefficient is not a unit modulo `n`.
    #[track_caller]
    pub(crate) fn new(f: &ModPolynomial) -> ModInverse {
        let lf = f.length();
        assert!(lf > 0, "division by the zero polynomial");
        let inv = f.unit_inverse(f.limbs(lf - 1), "leading");
        if let (Some(w), true) = (f.m.word_mod(), lf >= 2) {
            return ModInverse {
                f: f.clone(),
                finv: f.zero_like(),
                word: Some(Box::new(super::wrem::WordRem::new(&f.c, inv[0], *w, None))),
                inv,
                ready: None,
            };
        }
        let finv = if lf >= 2 && lf - 1 > rem_school_max() {
            f.rev(lf).inv_newton(lf - 1, &inv)
        } else {
            f.zero_like()
        };
        let k = f.m.k();
        let ready =
            if lf > ready_min() && !finv.is_zero() && super::mmul::use_split(lf - 1, lf - 1, k) {
                let bits =
                    f.m.l
                        .as_ref()
                        .map_or(0, |_| 64 * k as u32 - f.m.d[k - 1].leading_zeros());
                crate::math::numbers::nat::limb_poly_plan(lf - 1, lf - 1, bits).map(|p| {
                    let mut fi = finv.c.to_vec();
                    fi.resize((lf - 1) * k, 0);
                    Ready {
                        finv: crate::math::numbers::nat::limb_poly_transform(&fi, k, &p),
                        flow: crate::math::numbers::nat::limb_poly_transform(
                            &f.c[..(lf - 1) * k],
                            k,
                            &p,
                        ),
                    }
                })
            } else {
                None
            };
        ModInverse {
            f: f.clone(),
            finv,
            inv,
            word: None,
            ready,
        }
    }

    /// The remainder of `a` (of `lf ≤ la ≤ 2 lf - 2` coefficients) by the ready transforms.
    fn rem_ready(&self, a: &ModPolynomial, ready: &Ready) -> ModPolynomial {
        use crate::math::numbers::nat;
        let m = &a.m;
        let red = m.l.as_ref().expect("several limbs");
        let k = m.k();
        let (la, lf) = (a.length(), self.f.length());
        let lq = la - lf + 1;
        let ow = ready.finv.plan().out_limbs();
        // The top lq coefficients of a, reversed.
        let mut top = Vec::with_capacity(lq * k);
        for i in (la - lq..la).rev() {
            top.extend_from_slice(a.limbs(i));
        }
        let mut v = vec![0u64; lq * ow];
        nat::limb_poly_mul_ready(&mut v, ow, &top, k, &ready.finv);
        // q = the reversal of the low lq coefficients.
        let mut q = vec![0u64; lq * k];
        for (j, x) in v.chunks_exact(ow).enumerate() {
            red.reduce(x, &mut q[(lq - 1 - j) * k..(lq - j) * k]);
        }
        let mut w = vec![0u64; (lf - 1) * ow];
        nat::limb_poly_mul_ready(&mut w, ow, &q, k, &ready.flow);
        let mut r = a.c[..(lf - 1) * k].to_vec();
        let mut t = vec![0u64; k];
        for (j, x) in w.chunks_exact(ow).enumerate() {
            red.reduce(x, &mut t);
            m.sub(&mut r[j * k..(j + 1) * k], &t);
        }
        ModPolynomial::of(a.m.clone(), r)
    }

    /// `a mod f`.
    #[track_caller]
    pub(crate) fn rem(&self, a: &ModPolynomial) -> ModPolynomial {
        self.f.same_modulus(a);
        let (la, lf) = (a.length(), self.f.length());
        if la < lf {
            return a.clone();
        }
        if lf == 1 {
            return a.zero_like();
        }
        if let Some(w) = &self.word {
            let mut c = a.c.to_vec();
            w.rem(&mut c, &mut super::wrem::Work::default());
            return ModPolynomial::of(a.m.clone(), c);
        }
        if self.finv.is_zero() {
            return a.div_rem_school(&self.f, &self.inv).1;
        }
        if la > 2 * lf - 2 {
            // Longer than the inverse serves: the top 2 deg f coefficients reduced at a time.
            let mut a = a.clone();
            while a.length() > 2 * lf - 2 {
                let c = a.length() - (2 * lf - 2);
                let r = self.rem(&a.high(c));
                a = &a.low(c) + &r.shl(c);
            }
            return self.rem(&a);
        }
        if let Some(ready) = &self.ready {
            return self.rem_ready(a, ready);
        }
        let lq = la - lf + 1;
        let q = (&a.rev(la).low(lq) * &self.finv.low(lq)).low(lq).rev(lq);
        &a.low(lf - 1) - &(&q * &self.f.low(lf - 1)).low(lf - 1)
    }

    /// `a·b mod f`.
    pub(crate) fn mul_mod(&self, a: &ModPolynomial, b: &ModPolynomial) -> ModPolynomial {
        self.rem(&(a * b))
    }

    /// `a·x mod f`, for `a` of degree below that of `f`: a shift and one step of the division.
    fn mul_x(&self, a: &ModPolynomial) -> ModPolynomial {
        let lf = self.f.length();
        let s = a.shl(1);
        if s.length() < lf {
            return s;
        }
        // s - t f with t = lead(s) / lead(f).
        let k = a.m.k();
        let mut t = vec![0u64; k];
        let mut w = super::super::modular::Work::default();
        a.m.mul(&mut t, s.limbs(lf - 1), &self.inv, &mut w);
        let mut tf = self.f.scale(&a.m.number(&t));
        tf.c.words_mut().truncate((lf - 1) * k);
        let mut r = s.low(lf - 1);
        r -= &tf;
        r
    }

    /// `a^e mod f`, by sliding windows of squares and products modulo `f` (`x^e` by squares and
    /// shifts).
    #[track_caller]
    pub(crate) fn pow_mod(&self, a: &ModPolynomial, e: &Number) -> ModPolynomial {
        assert!(!e.is_negative(), "a negative exponent");
        let one = self.rem(&a.one_like());
        let bits = e.bits();
        if bits == 0 {
            return one;
        }
        let b = self.rem(a);
        if self.f.length() == 1 {
            return b;
        }
        if let Some(w) = &self.word {
            if b.is_zero() {
                return b;
            }
            return ModPolynomial::of(a.m.clone(), w.pow(&b.c, e));
        }
        let is_x = b.length() == 2 && b.limbs(0).iter().all(|&l| l == 0) && {
            let l = b.limbs(1);
            l[0] == 1 && l[1..].iter().all(|&v| v == 0)
        };
        if is_x {
            let mut r = b.clone();
            for i in (0..bits - 1).rev() {
                r = self.mul_mod(&r, &r);
                if e.bit(i) {
                    r = self.mul_x(&r);
                }
            }
            return r;
        }
        // Windows of up to w bits: the odd powers b, b^3, ..., b^(2^w - 1).
        let w: u64 = match bits {
            0..=24 => 1,
            25..=80 => 3,
            81..=240 => 4,
            241..=700 => 5,
            _ => 6,
        };
        let b2 = self.mul_mod(&b, &b);
        let mut odd = vec![b];
        for _ in 1..(1usize << (w - 1)) {
            let next = self.mul_mod(odd.last().expect("not empty"), &b2);
            odd.push(next);
        }
        let mut r = one;
        let mut started = false;
        let mut i = bits as i64 - 1;
        while i >= 0 {
            if !e.bit(i as u64) {
                if started {
                    r = self.mul_mod(&r, &r);
                }
                i -= 1;
                continue;
            }
            // The window from bit i down to the lowest set bit within w bits.
            let mut j = (i - w as i64 + 1).max(0);
            while !e.bit(j as u64) {
                j += 1;
            }
            let mut val = 0usize;
            for t in (j..=i).rev() {
                val = (val << 1) | usize::from(e.bit(t as u64));
            }
            if started {
                for _ in j..=i {
                    r = self.mul_mod(&r, &r);
                }
                r = self.mul_mod(&r, &odd[val >> 1]);
            } else {
                r = odd[val >> 1].clone();
                started = true;
            }
            i = j - 1;
        }
        r
    }

    /// `a(g) mod f` by Brent and Kung's method: with `m ≈ √(deg a)`, the powers `g^0..g^m` modulo
    /// `f`, each block of `m` coefficients of `a` evaluated at `g` as a combination of them (the
    /// rows of a product of matrices, the sums of products reduced once), and the blocks joined by
    /// Horner's rule in `g^m`.
    pub(crate) fn compose_mod(&self, a: &ModPolynomial, g: &ModPolynomial) -> ModPolynomial {
        let lf = self.f.length();
        if lf == 1 {
            return a.zero_like();
        }
        let la = a.length();
        if la == 0 {
            return a.zero_like();
        }
        let g = self.rem(g);
        if let Some(w) = &self.word {
            if g.is_zero() {
                return a.with_coeffs(&[a.coeff(0)]);
            }
            return ModPolynomial::of(a.m.clone(), w.compose(&a.c, &g.c));
        }
        let m = (la as f64).sqrt().ceil() as usize;
        let m = m.max(1);
        // g^0 .. g^m modulo f.
        let mut pows = vec![self.rem(&a.one_like())];
        for i in 1..=m {
            let next = if i == 1 {
                g.clone()
            } else {
                self.mul_mod(&pows[i - 1], &g)
            };
            pows.push(next);
        }
        let gm = pows.pop().expect("m >= 1");
        let blocks = la.div_ceil(m);
        let width = lf - 1;
        let k = a.m.k();
        let combos: Vec<ModPolynomial> = if let Some(wm) = a.m.word_mod() {
            // The matrix of the powers, row i the coefficients of g^i, and each block a row of
            // coefficients: the products by columns, summed in the width they need.
            // The powers by columns: cols[t m + i] the coefficient of x^t in g^i.
            let mut cols = vec![0u64; width * m];
            for (i, p) in pows.iter().enumerate() {
                for (t, &c) in p.c.iter().enumerate() {
                    cols[t * m + i] = c;
                }
            }
            (0..blocks)
                .map(|j| {
                    let cs = &a.c[j * m..((j + 1) * m).min(la)];
                    let out: Vec<u64> = (0..width)
                        .map(|t| wmul::dot_mod(cs, &cols[t * m..t * m + cs.len()], wm))
                        .collect();
                    ModPolynomial::of(a.m.clone(), out)
                })
                .collect()
        } else {
            // The same matrix product on limbs: each coefficient of a block's combination a sum
            // of products of residues, kept unreduced in 2k + 1 limbs (m < 2^64 products of two
            // residues below n), and reduced once.
            let red = a.m.l.as_ref().expect("several limbs");
            let mut acc = vec![0u64; 2 * k + 1];
            let mut t = vec![0u64; 2 * k];
            let mut scratch = Vec::new();
            (0..blocks)
                .map(|j| {
                    let lo = j * m;
                    let hi = ((j + 1) * m).min(la);
                    let mut out = vec![0u64; width * k];
                    for (tt, o) in out.chunks_exact_mut(k).enumerate() {
                        acc.fill(0);
                        for i in lo..hi {
                            let p = &pows[i - lo];
                            if tt >= p.length() {
                                continue;
                            }
                            let x = &a.c[i * k..(i + 1) * k];
                            let y = p.limbs(tt);
                            crate::math::numbers::nat::mul_to(&mut t, x, y, &mut scratch);
                            crate::math::numbers::nat::add_assign(&mut acc, &t);
                        }
                        red.reduce(&acc, o);
                    }
                    ModPolynomial::of(a.m.clone(), out)
                })
                .collect()
        };
        // Horner in g^m.
        let mut r = a.zero_like();
        for c in combos.iter().rev() {
            r = self.mul_mod(&r, &gm);
            r += c;
        }
        r
    }
}
