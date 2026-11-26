//! Square-free decompositions and complete factorisations: of polynomials modulo a prime (any
//! size, 2 and 3 included) by the distinct-degree factorisation and Cantor and Zassenhaus's
//! splitting or by Berlekamp's method, with Ben-Or's test of irreducibility and the roots; over the
//! integers and the rationals by Zassenhaus's method, a factorisation modulo a good prime lifted
//! by Hensel's lemma and recombined.

pub(crate) mod ff;
pub(crate) mod fp;
pub(crate) mod nmod;
pub(crate) mod zn;
pub(crate) mod zx;

use super::{ModPolynomial, Polynomial};
use crate::math::numbers::Number;
use crate::math::rational::Rational;
use fp::Fp;

impl ModPolynomial {
    /// The field of the modulus, after the check that it is prime.
    #[track_caller]
    fn field(&self) -> Fp {
        zx::assert_prime(&self.m);
        Fp::new(self.m.clone())
    }

    /// The square-free decomposition modulo a prime `p`: the leading coefficient `l`, and monic
    /// square-free polynomials `gᵢ`, not constant and pairwise coprime, with distinct
    /// multiplicities `eᵢ`, such that `self = l·Π gᵢ^eᵢ`, by degree, then by the coefficients from
    /// the leading one down. Yun's method on what the derivative sees, and the `p`-th root of the
    /// part whose derivative is 0, `Σ aᵢ x^(ip) = (Σ aᵢ xⁱ)^p` in characteristic `p`. `(0, [])` for
    /// the zero polynomial.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime (by a probable-prime test above `2^64`).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // Modulo 3: 2(x + 1)²(x³ + 2) = 2(x + 1)²(x + 2)³, as x³ + 2 = (x + 2)³.
    /// let p = |c: &[u64]| ModPolynomial::from_u64(c, 3);
    /// let f = &(&p(&[2]) * &p(&[1, 1]).pow(2)) * &p(&[2, 0, 0, 1]);
    /// let (l, parts) = f.squarefree();
    /// assert_eq!(l, Number::from(2));
    /// assert_eq!(parts, vec![(p(&[1, 1]), 2), (p(&[2, 1]), 3)]);
    /// ```
    #[track_caller]
    pub fn squarefree(&self) -> (Number, Vec<(ModPolynomial, u32)>) {
        let fp = self.field();
        let Some(l) = self.leading() else {
            return (Number::default(), Vec::new());
        };
        if self.length() == 1 {
            return (l, Vec::new());
        }
        let f = self.to_monic();
        let mut v: Vec<(ModPolynomial, u32)> = ff::squarefree(&fp, &f.c)
            .into_iter()
            .map(|(g, e)| (fp.wrap(g), e))
            .collect();
        let k = fp.k();
        v.sort_by(|a, b| ff::cmp_packed(&a.0.c, &b.0.c, k).then(a.1.cmp(&b.1)));
        (l, v)
    }

    /// The distinct-degree factorisation of a square-free polynomial modulo a prime: for each
    /// degree `d` of its irreducible factors, the monic product of those of degree `d`, and `d`,
    /// by degree; none for a constant. A short polynomial takes `x^(p^i) mod f` one `p`-th power
    /// after another; a longer one the baby steps and giant steps of Kaltofen and Shoup, on Brent
    /// and Kung's modular composition.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime, or the polynomial is zero or not square-free.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // Modulo 2: x⁵ + x⁴ + 1 = (x² + x + 1)(x³ + x + 1).
    /// let f = ModPolynomial::from_u64(&[1, 0, 0, 0, 1, 1], 2);
    /// let d = f.distinct_degree_factor();
    /// assert_eq!(d.len(), 2);
    /// assert_eq!(d[0], (ModPolynomial::from_u64(&[1, 1, 1], 2), 2));
    /// assert_eq!(d[1], (ModPolynomial::from_u64(&[1, 1, 0, 1], 2), 3));
    /// ```
    #[track_caller]
    pub fn distinct_degree_factor(&self) -> Vec<(ModPolynomial, u32)> {
        let fp = self.field();
        assert!(!self.is_zero(), "the zero polynomial");
        if self.length() == 1 {
            return Vec::new();
        }
        let f = self.to_monic();
        assert!(
            fp.is_one(&fp.gcd(&f.c, &fp.derivative(&f.c))),
            "the polynomial is not square-free"
        );
        ff::ddf(&fp, &f.c)
            .0
            .into_iter()
            .map(|(g, d)| (fp.wrap(g), d))
            .collect()
    }

    /// The monic irreducible factors of a polynomial modulo a prime whose irreducible factors are
    /// distinct and all of the degree `d`, as [`distinct_degree_factor`] gives them, sorted: by
    /// Cantor and Zassenhaus's random splittings (from a fixed seed, so the same every time), the
    /// greatest common divisor with `a^((p^d - 1)/2) - 1`, or for `p = 2` with the trace `a + a² +
    /// … + a^(2^(d-1))`, for random `a`. A constant has none.
    ///
    /// [`distinct_degree_factor`]: ModPolynomial::distinct_degree_factor
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime, the polynomial is zero, `d` is 0, or the polynomial is
    /// not such a product: its degree not a multiple of `d`, or a part found that does not split.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// // Modulo 7: x² + 1 and x² + x + 3, both irreducible, multiplied.
    /// let p = |c: &[u64]| ModPolynomial::from_u64(c, 7);
    /// let f = &p(&[1, 0, 1]) * &p(&[3, 1, 1]);
    /// assert_eq!(f.equal_degree_factor(2), vec![p(&[1, 0, 1]), p(&[3, 1, 1])]);
    /// ```
    #[track_caller]
    pub fn equal_degree_factor(&self, d: u32) -> Vec<ModPolynomial> {
        let fp = self.field();
        assert!(!self.is_zero(), "the zero polynomial");
        assert!(d > 0, "the degree 0");
        if self.length() == 1 {
            return Vec::new();
        }
        let f = self.to_monic();
        let n = fp.deg(&f.c);
        assert!(n.is_multiple_of(d as usize), "{}", ff::NOT_EQUAL_DEGREE);
        let mut state = ff::SEED;
        let mut out = Vec::new();
        ff::edf(&fp, &f.c, d as usize, None, &mut state, &mut out);
        let k = fp.k();
        out.sort_by(|a, b| ff::cmp_packed(a, b, k));
        out.into_iter().map(|g| fp.wrap(g)).collect()
    }

    /// The complete factorisation modulo a prime: the leading coefficient, and the monic
    /// irreducible factors with their multiplicities, by degree, then by the coefficients from
    /// the leading one down. The square-free decomposition, then the distinct-degree
    /// factorisation and the equal-degree splitting of each part; for `p` up to 7 and a part of
    /// degree up to 175 (`p` = 2), 600 (`p` = 3) or 300 (`p` = 5, 7), Berlekamp's method, as
    /// [`factor_berlekamp`](ModPolynomial::factor_berlekamp), which is faster there. `(0, [])`
    /// for the zero polynomial, `(c, [])` for a constant `c`.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // Modulo 5: 3x⁴ - 3 = 3(x - 1)(x - 2)(x - 3)(x - 4).
    /// let f = ModPolynomial::from_u64(&[2, 0, 0, 0, 3], 5);
    /// let (l, fs) = f.factor();
    /// assert_eq!(l, Number::from(3));
    /// let lin: Vec<ModPolynomial> = (1..5).map(|r| ModPolynomial::from_u64(&[5 - r, 1], 5)).collect();
    /// assert_eq!(fs, vec![(lin[3].clone(), 1), (lin[2].clone(), 1), (lin[1].clone(), 1), (lin[0].clone(), 1)]);
    /// ```
    #[track_caller]
    pub fn factor(&self) -> (Number, Vec<(ModPolynomial, u32)>) {
        self.factor_by(false)
    }

    /// The complete factorisation modulo a prime as [`factor`](ModPolynomial::factor) gives it,
    /// by Berlekamp's method for each square-free part: the kernel of `Q - I`, `Q` the matrix of
    /// the Frobenius map `a ↦ aᵖ` on `F_p[x]/(f)`, has the dimension of the number of factors, and
    /// its elements `v` split `f` by greatest common divisors with `v - s` for each `s` in `F_p`,
    /// for `p` up to 11; above, a random `v` of it is a constant `cᵢ` modulo each factor, so its
    /// minimal polynomial (by Berlekamp and Massey's algorithm) is `Π (x - c)` over its distinct
    /// values, whose roots `c` split `f` by `gcd(f, v - c)`.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// let f = ModPolynomial::from_u64(&[1, 0, 0, 0, 1, 1], 2);
    /// assert_eq!(f.factor_berlekamp(), f.factor());
    /// ```
    #[track_caller]
    pub fn factor_berlekamp(&self) -> (Number, Vec<(ModPolynomial, u32)>) {
        self.factor_by(true)
    }

    /// The factorisation, by Berlekamp's method when `berlekamp`.
    #[track_caller]
    fn factor_by(&self, berlekamp: bool) -> (Number, Vec<(ModPolynomial, u32)>) {
        let fp = self.field();
        let Some(l) = self.leading() else {
            return (Number::default(), Vec::new());
        };
        if self.length() == 1 {
            return (l, Vec::new());
        }
        let f = self.to_monic();
        let v = ff::factor(&fp, &f.c, berlekamp)
            .into_iter()
            .map(|(g, e)| (fp.wrap(g), e))
            .collect();
        (l, v)
    }

    /// Whether the polynomial is irreducible modulo a prime: of degree at least 1 and not a
    /// product of two of lower degrees. Ben-Or's test: `x^(p^i) - x` prime to `f` for every `i` up
    /// to half the degree, the Frobenius powers as [`distinct_degree_factor`] takes them, by baby
    /// steps and giant steps with the products of the differences of each interval of degrees,
    /// stopping at the first common factor; the roots first.
    ///
    /// [`distinct_degree_factor`]: ModPolynomial::distinct_degree_factor
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    ///
    /// assert!(ModPolynomial::from_u64(&[1, 0, 1], 7).is_irreducible()); // x² + 1
    /// assert!(!ModPolynomial::from_u64(&[1, 0, 1], 5).is_irreducible()); // (x + 2)(x + 3)
    /// ```
    #[track_caller]
    pub fn is_irreducible(&self) -> bool {
        let fp = self.field();
        if self.length() < 2 {
            return false;
        }
        let f = self.to_monic();
        ff::is_irreducible(&fp, &f.c)
    }

    /// The roots modulo a prime with their multiplicities, in increasing order: for each
    /// square-free part, its greatest common divisor with `x^p - x`, the product of its factors
    /// of degree 1, split by Cantor and Zassenhaus's method. For `p` below 64, from the
    /// polynomial folded modulo `x^p - x` instead: the roots of the folded polynomial (of a degree
    /// below `p`) at every residue, each multiplicity from the first Hasse derivative that does
    /// not vanish there (folded too), a multiplicity of `p` or more by synthetic divisions; for a
    /// word `p` from 64 up to the degree, every residue tried by synthetic division, each root
    /// divided out as often as it divides. None for a constant other than 0.
    ///
    /// # Panics
    ///
    /// Panics if the modulus is not prime or the polynomial is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::ModPolynomial;
    /// use a::math::numbers::Number;
    ///
    /// // Modulo 101: (x - 3)²(x - 50)(x² + 1), x² + 1 = (x - 10)(x - 91).
    /// let p = |c: &[u64]| ModPolynomial::from_u64(c, 101);
    /// let f = &(&p(&[98, 1]).pow(2) * &p(&[51, 1])) * &p(&[1, 0, 1]);
    /// let r: Vec<(u64, u32)> = f.roots().iter().map(|(x, e)| (u64::try_from(x).unwrap(), *e)).collect();
    /// assert_eq!(r, [(3, 2), (10, 1), (50, 1), (91, 1)]);
    /// ```
    #[track_caller]
    pub fn roots(&self) -> Vec<(Number, u32)> {
        let fp = self.field();
        assert!(!self.is_zero(), "the roots of the zero polynomial");
        if self.length() == 1 {
            return Vec::new();
        }
        let f = self.to_monic();
        if let (Some(w), Some(p)) = (self.m.word_mod(), self.m.word())
            && p < SMALL_ROOTS_P
        {
            return small_p_roots(&f.c, p, w);
        }
        ff::roots(&fp, &f.c)
    }
}

impl Polynomial<Number> {
    /// The square-free decomposition over the integers: the content with the sign of the leading
    /// coefficient, `c`, and primitive square-free polynomials `gᵢ` of positive leading
    /// coefficients, not constant and pairwise coprime, with distinct multiplicities `eᵢ`, such
    /// that `self = c·Π gᵢ^eᵢ`, by degree, then by the coefficients from the leading one down.
    /// Yun's method, after a check modulo a few primes, which says most square-free polynomials
    /// are. `(0, [])` for the zero polynomial, `(c, [])` for a constant `c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // -6(x - 1)(x + 2)²
    /// let f = &(&p(&[-6]) * &p(&[-1, 1])) * &p(&[2, 1]).pow(2);
    /// assert_eq!(f.squarefree(), (Number::from(-6), vec![(p(&[-1, 1]), 1), (p(&[2, 1]), 2)]));
    /// ```
    pub fn squarefree(&self) -> (Number, Vec<(Polynomial<Number>, u32)>) {
        if self.is_zero() {
            return (Number::default(), Vec::new());
        }
        let (c, g) = zx::split_content(self);
        if g.is_constant() {
            return (c, Vec::new());
        }
        let mut v = if zx::is_squarefree(&g) {
            vec![(g, 1)]
        } else {
            zx::yun(&g)
        };
        zx::sort_factors(&mut v);
        (c, v)
    }

    /// The complete factorisation over the integers: the content with the sign of the leading
    /// coefficient, and the primitive irreducible factors of positive leading coefficients with
    /// their multiplicities, by degree, then by the coefficients from the leading one down. The
    /// square-free decomposition, then for each part Zassenhaus's method: the factorisation
    /// modulo the good prime with the fewest factors of several (the degrees all of them allow
    /// kept), lifted quadratically by Hensel's lemma past twice Mignotte's bound, and recombined,
    /// the subsets of the lifted factors tested by their degree, the trace test and the constant
    /// coefficient before a trial division. `(0, [])` for the zero polynomial, `(c, [])` for a
    /// constant `c`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// // 2x⁴ - 2 = 2(x - 1)(x + 1)(x² + 1)
    /// let (c, fs) = p(&[-2, 0, 0, 0, 2]).factor();
    /// assert_eq!(c, Number::from(2));
    /// assert_eq!(fs, vec![(p(&[-1, 1]), 1), (p(&[1, 1]), 1), (p(&[1, 0, 1]), 1)]);
    /// ```
    pub fn factor(&self) -> (Number, Vec<(Polynomial<Number>, u32)>) {
        zx::factor(self)
    }

    /// Whether the polynomial is irreducible over the rationals: of degree at least 1 and not a
    /// product of two of lower degrees. The content does not count, as over the rationals: `2x +
    /// 2` is irreducible. One good prime modulo which it is irreducible, or degrees of the
    /// factors modulo several primes that no factor can have, tell most irreducible polynomials
    /// quickly; the others take the factorisation.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    ///
    /// let p = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// assert!(p(&[1, 0, 0, 0, 1]).is_irreducible()); // x⁴ + 1, reducible modulo every prime
    /// assert!(!p(&[4, 0, 0, 0, 1]).is_irreducible()); // (x² + 2x + 2)(x² - 2x + 2)
    /// ```
    pub fn is_irreducible(&self) -> bool {
        zx::is_irreducible(self)
    }
}

impl Polynomial<Rational> {
    /// The square-free decomposition over the rationals: a rational `c` and primitive
    /// square-free polynomials over the integers `gᵢ` of positive leading coefficients, with
    /// `self = c·Π gᵢ^eᵢ`, as [`Polynomial::<Number>::squarefree`] gives them for the primitive
    /// part. `(0, [])` for the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// // (x + 1/2)² / 3 = (1/12)(2x + 1)²
    /// let h = Rational::new(1.into(), 2.into());
    /// let f = Polynomial::new(vec![h.clone(), Rational::from(1)]).pow(2).scale(&Rational::new(1.into(), 3.into()));
    /// let n = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// assert_eq!(f.squarefree(), (Rational::new(1.into(), 12.into()), vec![(n(&[1, 2]), 2)]));
    /// ```
    pub fn squarefree(&self) -> (Rational, Vec<(Polynomial<Number>, u32)>) {
        let (c, p) = self.to_primitive();
        let (_, v) = p.squarefree();
        (c, v)
    }

    /// The complete factorisation over the rationals: a rational `c` and the primitive
    /// irreducible polynomials over the integers of positive leading coefficients with their
    /// multiplicities, `self = c·Π gᵢ^eᵢ`, as [`Polynomial::<Number>::factor`] gives them for the
    /// primitive part. `(0, [])` for the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::poly::Polynomial;
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// // x²/2 - 1/8 = (1/8)(2x - 1)(2x + 1)
    /// let f = Polynomial::new(vec![Rational::new((-1).into(), 8.into()), Rational::from(0), Rational::new(1.into(), 2.into())]);
    /// let n = |c: &[i64]| Polynomial::new(c.iter().map(|&x| Number::from(x)).collect());
    /// assert_eq!(f.factor(), (Rational::new(1.into(), 8.into()), vec![(n(&[-1, 2]), 1), (n(&[1, 2]), 1)]));
    /// ```
    pub fn factor(&self) -> (Rational, Vec<(Polynomial<Number>, u32)>) {
        let (c, p) = self.to_primitive();
        let (_, v) = p.factor();
        (c, v)
    }

    /// Whether the polynomial is irreducible over the rationals, as
    /// [`Polynomial::<Number>::is_irreducible`] tells it of the primitive part.
    pub fn is_irreducible(&self) -> bool {
        self.to_primitive().1.is_irreducible()
    }
}

/// Below this prime the roots and their multiplicities come from `f` folded modulo `x^p - x`
/// ([`small_p_roots`]: `deg` additions and products by small binomials a pass, and evaluations of
/// a degree below `p`), against the gcd with `x^p - x` by squarings modulo `f` (or every residue
/// by synthetic division up to 64 or the degree). Measured on an Apple M4 Max (primes 5 to 251,
/// degrees 3 to 10,000, random polynomials and products of linear factors): the fold faster from
/// a degree of about `1.5p` (25 to 120 times at 10,000 below 64), the other methods below it (up
/// to 37 times at degree 3) and for multiplicities above `p` (1.2 to 1.7 times); every other
/// bound on `p` alone is slower somewhere by more than that bound gains, so 64 is kept.
const SMALL_ROOTS_P: u64 = 64;

/// The roots of the monic `f` (words, at least two coefficients) modulo the small prime `p`
/// with their multiplicities, increasing. Each `x^k`, `k ≥ 1`, is `x^(1 + (k - 1) mod (p - 1))`
/// at every point of `F_p`, so a polynomial folded by those classes, of a degree below `p`, has
/// its values. The roots are those of `f` folded; a root `r` has the multiplicity of the first
/// `k` with `H_k f(r) ≠ 0`, the Hasse derivatives `H_k f = Σ C(i, k) c_i x^(i-k)` (`f`'s Taylor
/// coefficients at `r`), each folded in one pass with `C(i, k) ≡ C(i mod p, k)` (Lucas, `k < p`);
/// 0 has the number of zero coefficients at the bottom; a multiplicity of `p` or more is left to
/// synthetic divisions by `x - r`.
fn small_p_roots(f: &[u64], p: u64, w: &super::arith::nmod::WordMod) -> Vec<(Number, u32)> {
    let p_us = p as usize;
    // H_k f folded: class 0 for x^0, then 1, 2, …, p - 1, 1, 2, … for x^1, x^2, …; the sums
    // below len·p² < 2^64.
    let fold = |k: usize, bin: &[u64]| -> Vec<u64> {
        let mut h = vec![0u64; p_us];
        let (mut cls, mut im) = (0usize, k % p_us);
        for &c in &f[k..] {
            h[cls] += bin[im] * c;
            cls = if cls + 1 == p_us { 1 } else { cls + 1 };
            im = if im + 1 == p_us { 0 } else { im + 1 };
        }
        for v in h.iter_mut() {
            *v = w.reduce1(*v);
        }
        h
    };
    let at = |h: &[u64], r: u64| h.iter().rev().fold(0u64, |v, &a| w.add(w.mul(v, r), a));
    // C(j, k) mod p for j < p, row k = 0 then 1.
    let mut bin = vec![1u64; p_us];
    let g = fold(0, &bin);
    let mut roots: Vec<(u64, u32)> = (0..p).filter(|&r| at(&g, r) == 0).map(|r| (r, 0)).collect();
    if let Some((0, m)) = roots.first_mut() {
        *m = f.iter().take_while(|&&c| c == 0).count() as u32;
    }
    let mut k = 1usize;
    while k < p_us && k < f.len() && roots.iter().any(|&(_, m)| m == 0) {
        // Row k of Pascal's triangle modulo p from row k - 1: C(j, k) = Σ_(t < j) C(t, k - 1).
        let mut acc = 0u64;
        for b in bin.iter_mut() {
            let old = *b;
            *b = acc;
            acc = (acc + old) % p;
        }
        let h = fold(k, &bin);
        for (r, m) in roots.iter_mut() {
            if *m == 0 && at(&h, *r) != 0 {
                *m = k as u32;
            }
        }
        k += 1;
    }
    for (r, m) in roots.iter_mut().filter(|(_, m)| *m == 0) {
        let r = *r;
        let mut c = f.to_vec();
        loop {
            // c = (x - r)·q + c(r): q_(i-1) = c_i + r q_i from the top.
            let n = c.len() - 1;
            let mut q = vec![0u64; n];
            q[n - 1] = c[n];
            for i in (1..n).rev() {
                q[i - 1] = w.add(c[i], w.mul(r, q[i]));
            }
            if w.add(c[0], w.mul(r, q[0])) != 0 {
                break;
            }
            *m += 1;
            c = q;
            if c.len() <= 1 {
                break;
            }
        }
    }
    roots
        .into_iter()
        .map(|(r, m)| (Number::from(r), m))
        .collect()
}
