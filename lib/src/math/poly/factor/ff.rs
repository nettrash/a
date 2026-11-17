//! Factorisation over a prime field `F_p`, on monic polynomials as [`Fp`] holds them: the
//! square-free decomposition, the distinct-degree factorisation (one Frobenius power after another
//! for a short polynomial, the baby-step giant-step method of Kaltofen and Shoup for a longer
//! one), Cantor and Zassenhaus's equal-degree splitting, Berlekamp's method, Ben-Or's test of
//! irreducibility and the roots.

use super::super::arith::wmul;
use super::fp::{Fp, Mf, P};
use super::nmod::{self, A3, Acc, Wm, by_class, class};
use crate::math::numbers::{ByteOrder, Number, from_limbs};
use core::cmp::Ordering;

/// Up to this product of the degree and the bits of `p` the distinct-degree factorisation takes
/// one Frobenius power after another; above, baby steps and giant steps.
pub(crate) const DDF_NAIVE: usize = 48;

/// Up to this `p` Berlekamp's method splits by the greatest common divisors with `v - s` for each
/// `s` of `F_p`; above, by the roots of the minimal polynomials of random elements of the kernel
/// (measured on an Apple M4: from 17 on that is faster, at 11 the same).
const BERLEKAMP_SMALL_P: u64 = 11;

/// The panic of an equal-degree splitting of a polynomial that is not a product of distinct
/// irreducible factors of the degree.
pub(crate) const NOT_EQUAL_DEGREE: &str =
    "the polynomial is not a product of distinct irreducible factors of the degree";

/// The seed of the probabilistic splittings, so that they are deterministic.
pub(crate) const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The rows of the baby steps of Brent and Kung's composition modulo `f` of degree `n` that
/// serve `uses` compositions with the same `h`: the powers cost one product modulo `f` each and a
/// composition `⌈n/m⌉ - 1` (the matrix product costs `n²` products of residues whatever `m`), so
/// `m = √(uses·n)`, at least `√n` and at most `n` or `2^22/n` (the table's size).
pub(crate) fn rows(n: usize, uses: usize) -> Option<usize> {
    let m = ((uses.max(1) * n.max(1)) as f64).sqrt().ceil() as usize;
    let lo = (n.max(1) as f64).sqrt().ceil() as usize;
    // At most 2^22 residues in the table (32 MiB), at least √n rows.
    let hi = ((1usize << 22) / n.max(1)).clamp(lo, n.max(1));
    Some(m.clamp(lo, hi))
}

/// The baby steps `hᵢ = x^(p^i) mod f` for `i` from 2 to `l`, after `x` and `x^p` in `hs`: each the
/// `p`-th power of the last when `p` is small (about `1.5 bits(p)` products modulo `f`), by
/// Brent and Kung's composition with `x^p` otherwise.
fn baby(fp: &Fp, hs: &mut Vec<P>, l: usize, mf: &Mf, n: usize) {
    if l < 2 {
        return;
    }
    let bits = fp.m.n.bits() as usize;
    let comp = 2.0 * ((n as f64) / ((l - 1) as f64)).sqrt();
    if (bits as f64) <= comp {
        let p = fp.m.d.clone();
        for i in 2..=l {
            let next = fp.powmod(&hs[i - 1], &p, mf);
            hs.push(next);
        }
        return;
    }
    let pw = fp.powers(&hs[1], mf, n, rows(n, l - 1));
    for i in 2..=l {
        let next = fp.compose(&hs[i - 1], &pw, mf);
        hs.push(next);
    }
}

/// The baby steps `l` of the distinct-degree factorisation of a polynomial of degree `n`.
pub(crate) fn baby_steps(n: usize) -> usize {
    (((n as f64) / 2.0).sqrt().ceil() as usize).max(1)
}

/// `p` as a number.
fn p_number(fp: &Fp) -> Number {
    fp.m.n.clone()
}

/// The packed polynomials `a` and `b` of `k` limbs a coefficient compared by degree, then by the
/// coefficients from the leading one down.
pub(crate) fn cmp_packed(a: &[u64], b: &[u64], k: usize) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| {
        for i in (0..a.len() / k).rev() {
            let (x, y) = (&a[i * k..(i + 1) * k], &b[i * k..(i + 1) * k]);
            for j in (0..k).rev() {
                match x[j].cmp(&y[j]) {
                    Ordering::Equal => {}
                    o => return o,
                }
            }
        }
        Ordering::Equal
    })
}

/// The square-free decomposition of the monic `f`: pairwise coprime monic square-free `gᵢ`, not
/// constant, with distinct multiplicities `eᵢ`, `f = Π gᵢ^eᵢ`, in characteristic `p` by Yun's
/// method on what the derivative sees and the `p`-th root of what it does not.
pub(crate) fn squarefree(fp: &Fp, f: &[u64]) -> Vec<(P, u32)> {
    let mut out = Vec::new();
    sqf(fp, f.to_vec(), 1, &mut out);
    out
}

fn sqf(fp: &Fp, f: P, mult: u32, out: &mut Vec<(P, u32)>) {
    if fp.deg(&f) == 0 {
        return;
    }
    let d = fp.derivative(&f);
    let (mut c, mut w) = if d.is_empty() {
        (f, fp.one())
    } else {
        let c = fp.gcd(&f, &d);
        let w = fp.div(&f, &c);
        (c, w)
    };
    let mut i = 1u32;
    while !fp.is_one(&w) {
        let y = fp.gcd(&w, &c);
        let fac = fp.div(&w, &y);
        if !fp.is_one(&fac) {
            out.push((fac, i * mult));
        }
        c = fp.div(&c, &y);
        w = y;
        i += 1;
    }
    if !fp.is_one(&c) {
        // `c` is a p-th power, so p is at most its degree and fits a word.
        let p = fp.m.word().expect("p is at most the degree") as u32;
        let r = fp.pth_root(&c);
        sqf(fp, r, mult * p, out);
    }
}

/// The distinct-degree factorisation of the monic square-free `f` of degree at least 1: for each
/// degree `d` of its irreducible factors, their product and `d`, by degree.
///
/// Also `x^p mod f`, which the equal-degree splitting takes.
pub(crate) fn ddf(fp: &Fp, f: &[u64]) -> (Vec<(P, u32)>, P) {
    // Frobenius powers one after another cost the bits of p in squarings each: for a small p
    // and a short f, fewer than the compositions of the baby steps and giant steps.
    if fp.deg(f) * fp.m.n.bits() as usize <= DDF_NAIVE {
        ddf_naive(fp, f)
    } else {
        ddf_bsgs(fp, f)
    }
}

/// The distinct-degree factorisation by `x^(p^i) mod f` for `i = 1, 2, …`, each the `p`-th power
/// of the last, and a greatest common divisor with `x^(p^i) - x` for each.
pub(crate) fn ddf_naive(fp: &Fp, f: &[u64]) -> (Vec<(P, u32)>, P) {
    let mut out = Vec::new();
    let mut frem = f.to_vec();
    let mut mf = fp.mf(&frem);
    let p = fp.m.d.clone();
    let mut h = fp.frobenius(&mf);
    let sigma = h.clone();
    let mut i = 1usize;
    while 2 * i <= fp.deg(&frem) {
        let g = fp.gcd(&frem, &fp.sub_x(&h));
        if !fp.is_one(&g) {
            frem = fp.div(&frem, &g);
            out.push((g, i as u32));
            if 2 * (i + 1) > fp.deg(&frem) {
                break;
            }
            mf = fp.mf(&frem);
            h = fp.rem_mf(&h, &mf);
        }
        i += 1;
        if 2 * i > fp.deg(&frem) {
            break;
        }
        h = fp.powmod(&h, &p, &mf);
    }
    if fp.deg(&frem) > 0 {
        let d = fp.deg(&frem) as u32;
        out.push((frem, d));
    }
    (out, sigma)
}

/// The distinct-degree factorisation by baby steps and giant steps (Kaltofen and Shoup; von zur
/// Gathen and Shoup): `hᵢ = x^(p^i)` for `i < l` and `Hⱼ = x^(p^(lj))` by Brent and Kung's modular
/// composition, the factors of the degrees in `(l(j - 1), lj]` from the greatest common divisor
/// with `Πᵢ (Hⱼ - hᵢ)`, and each degree from those with `Hⱼ - hᵢ`.
pub(crate) fn ddf_bsgs(fp: &Fp, f: &[u64]) -> (Vec<(P, u32)>, P) {
    let n = fp.deg(f);
    let l = baby_steps(n);
    let m = n.div_ceil(2 * l);
    let mf = fp.mf(f);
    let sigma = fp.frobenius(&mf);
    let mut hs: Vec<P> = Vec::with_capacity(l + 1);
    hs.push(fp.rem_mf(&fp.x(), &mf));
    hs.push(sigma);
    baby(fp, &mut hs, l, &mf, n);
    let sigma = hs[1].clone();
    let big = hs[l].clone();
    let mut pwg = None;
    let mut out = Vec::new();
    let mut frem = f.to_vec();
    let mut hj = big.clone();
    for j in 1..=m {
        if 2 * (l * (j - 1) + 1) > fp.deg(&frem) {
            break;
        }
        if j > 1 {
            let pwg = pwg.get_or_insert_with(|| fp.powers(&big, &mf, n, rows(n, m - j + 1)));
            hj = fp.compose(&hj, pwg, &mf);
        }
        let mut prod = fp.sub(&hj, &hs[0]);
        for h in &hs[1..l] {
            prod = fp.mulmod(&prod, &fp.sub(&hj, h), &mf);
        }
        let mut g = fp.gcd(&frem, &prod);
        if fp.is_one(&g) {
            continue;
        }
        frem = fp.div(&frem, &g);
        for i in (0..l).rev() {
            if fp.deg(&g) == 0 {
                break;
            }
            let d = l * j - i;
            if 2 * d > fp.deg(&g) {
                // What is left is irreducible: its degree is in the interval.
                let dg = fp.deg(&g) as u32;
                out.push((g, dg));
                g = fp.one();
                break;
            }
            let t = fp.rem(&fp.sub(&hj, &hs[i]), &g);
            let gg = fp.gcd(&g, &t);
            if !fp.is_one(&gg) {
                g = fp.div(&g, &gg);
                out.push((gg, d as u32));
            }
        }
        debug_assert!(fp.is_one(&g));
    }
    if fp.deg(&frem) > 0 {
        let d = fp.deg(&frem) as u32;
        out.push((frem, d));
    }
    out.sort_by_key(|a| a.1);
    (out, sigma)
}

/// `(p^d - 1)/2` as limbs, for an odd `p`.
fn half_order(fp: &Fp, d: usize) -> Vec<u64> {
    let e = &(&p_number(fp).pow(d as u32) - &Number::from(1u8)) >> 1usize;
    Fp::limbs(&e)
}

/// The irreducible factors of the monic square-free `f` whose irreducible factors all have the
/// degree `d`, by Cantor and Zassenhaus's random splittings: for an odd `p` by the greatest common
/// divisor with `a^((p^d - 1)/2) - 1`, for `p = 2` with the trace `a + a² + … + a^(2^(d-1))`.
///
/// `sigma`, when given, is `x^p` modulo a multiple of `f`.
pub(crate) fn edf(
    fp: &Fp,
    f: &[u64],
    d: usize,
    sigma: Option<&[u64]>,
    state: &mut u64,
    out: &mut Vec<P>,
) {
    let n = fp.deg(f);
    if n == 0 {
        return;
    }
    if n == d {
        out.push(f.to_vec());
        return;
    }
    let mf = fp.mf(f);
    let two = fp.m.word() == Some(2);
    let bits = fp.m.n.bits() as usize;
    // For a large p, a^((p^d - 1)/2) = N^((p - 1)/2) with the norm N = a·a^p·…·a^(p^(d-1)), the
    // powers a^(p^i) by compositions with x^p: d - 1 compositions and the bits of p in squarings,
    // instead of d times the bits of p.
    let norm = !two && d > 1 && bits * bits > 4 * n;
    let e = if two {
        Vec::new()
    } else if norm {
        half_order(fp, 1)
    } else {
        half_order(fp, d)
    };
    let sigma = match (norm, sigma) {
        (false, _) => Vec::new(),
        (true, Some(s)) => fp.rem_mf(s, &mf),
        (true, None) => fp.frobenius(&mf),
    };
    let mut todo = vec![f.to_vec()];
    // Split every part with the same exponent, the parts of degree d done.
    while let Some(g) = todo.pop() {
        let n = fp.deg(&g);
        if n == d {
            out.push(g);
            continue;
        }
        let mfg = if n == fp.deg(f) {
            None
        } else {
            Some(fp.mf(&g))
        };
        let mfg_ref = mfg.as_ref().unwrap_or(&mf);
        assert!(n.is_multiple_of(d) && n > d, "{NOT_EQUAL_DEGREE}");
        let pw = norm.then(|| {
            let sg = fp.rem_mf(&sigma, mfg_ref);
            fp.powers(&sg, mfg_ref, n, rows(n, 2 * (d - 1)))
        });
        let mut tries = 0;
        loop {
            // A part with two factors splits with a probability of about 1/2 each time.
            tries += 1;
            assert!(tries <= 256, "{NOT_EQUAL_DEGREE}");
            let a = fp.random(state, n);
            let s = if two {
                let mut t = a.clone();
                let mut s = a;
                for _ in 1..d {
                    s = fp.sqrmod(&s, mfg_ref);
                    t = fp.add(&t, &s);
                }
                fp.gcd(&g, &t)
            } else {
                let base = match &pw {
                    Some(pw) => {
                        let mut nm = a.clone();
                        let mut ap = a;
                        for _ in 1..d {
                            ap = fp.compose(&ap, pw, mfg_ref);
                            nm = fp.mulmod(&nm, &ap, mfg_ref);
                        }
                        nm
                    }
                    None => a,
                };
                let b = fp.powmod(&base, &e, mfg_ref);
                fp.gcd(&g, &fp.sub_one(&b))
            };
            let ds = fp.deg(&s);
            if ds > 0 && ds < n {
                let h = fp.div(&g, &s);
                todo.push(s);
                todo.push(h);
                break;
            }
        }
    }
}

/// The irreducible factors of the monic square-free `f` of degree at least 1, by the
/// distinct-degree factorisation and the equal-degree splitting.
pub(crate) fn factor_squarefree(fp: &Fp, f: &[u64], state: &mut u64) -> Vec<P> {
    let mut out = Vec::new();
    let (parts, sigma) = ddf(fp, f);
    for (g, d) in parts {
        edf(fp, &g, d as usize, Some(&sigma), state, &mut out);
    }
    out
}

/// The complete factorisation of the monic `f`: its monic irreducible factors with their
/// multiplicities, by degree, then by the coefficients from the leading one down. `berlekamp`
/// takes Berlekamp's method for each square-free part.
pub(crate) fn factor(fp: &Fp, f: &[u64], berlekamp: bool) -> Vec<(P, u32)> {
    let mut out = Vec::new();
    let mut state = SEED;
    let bits = fp.m.n.bits();
    for (g, e) in squarefree(fp, f) {
        let parts = if berlekamp || fp.deg(&g) <= berlekamp_max(bits) {
            berlekamp_squarefree(fp, &g, &mut state)
        } else {
            factor_squarefree(fp, &g, &mut state)
        };
        out.extend(parts.into_iter().map(|h| (h, e)));
    }
    let k = fp.k();
    out.sort_by(|a, b| cmp_packed(&a.0, &b.0, k).then(a.1.cmp(&b.1)));
    out
}

/// The kernel of the rows of the `n×n` matrix `q` of words modulo `w`: Gaussian elimination to
/// echelon form with the entries below the pivots kept as unreduced sums of products (each
/// receives at most one product a pivot, so `A` of the class of `n` products holds them), each
/// pivot's column reduced as it is reached and its row once, then back-substitution for each free
/// column.
fn kernel_word<A: Acc>(w: &Wm, q: &[Vec<u64>], n: usize) -> Vec<P> {
    let mut rows: Vec<Vec<A>> = q
        .iter()
        .map(|r| r.iter().map(|&x| A::of(x)).collect())
        .collect();
    // The pivot column of each row of the echelon form, and the row made monic.
    let mut piv: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut row = 0usize;
    for col in 0..n {
        if row == n {
            break;
        }
        let found = rows[row..]
            .iter()
            .enumerate()
            .find_map(|(i, r)| Some((row + i, r[col].red(w))).filter(|&(_, v)| v != 0));
        let Some((r, v)) = found else {
            continue;
        };
        rows.swap(row, r);
        let inv = nmod::inv(w, v).expect("p is prime");
        let mut prow = vec![0u64; n];
        prow[col] = 1;
        for (x, a) in prow[col + 1..].iter_mut().zip(&rows[row][col + 1..]) {
            *x = w.mul(a.red(w), inv);
        }
        for qr in rows[row + 1..].iter_mut() {
            let c = qr[col].red(w);
            if c == 0 {
                continue;
            }
            let nc = w.n - c;
            for (x, &y) in qr[col + 1..].iter_mut().zip(&prow[col + 1..]) {
                x.madd(nc, y);
            }
        }
        piv.push((col, prow));
        row += 1;
    }
    let mut is_pivot = vec![false; n];
    for (c, _) in &piv {
        is_pivot[*c] = true;
    }
    let mut basis = Vec::new();
    for fc in (0..n).filter(|&c| !is_pivot[c]) {
        let mut v = vec![0u64; n];
        v[fc] = 1;
        for (c, prow) in piv.iter().rev() {
            if *c > fc {
                continue;
            }
            // v_c = -Σ_{j > c} prow_j v_j.
            let s = wmul::dot_mod(&prow[c + 1..], &v[c + 1..], w);
            v[*c] = w.neg(s);
        }
        nmod::trim(&mut v);
        basis.push(v);
    }
    basis
}

/// Up to this degree the factorisation modulo a `p` of `bits` bits takes Berlekamp's method (its
/// elimination costs `n³/3` products of residues, which for a small `p` sum in a word, against
/// the distinct-degree factorisation's products modulo `f`): measured on an Apple M4.
fn berlekamp_max(bits: u64) -> usize {
    match bits {
        0..=2 => 600,
        3 => 300,
        _ => 0,
    }
}

/// The scalars of `F_p` Berlekamp's elimination takes: words, or `k` limbs.
fn kernel(fp: &Fp, q: &mut [Vec<u64>], n: usize) -> Vec<P> {
    if let Some(w) = &fp.w {
        return by_class!(w, n, kernel_word(w, q, n));
    }
    // `q` is the n×n matrix of Q - I transposed, the entries packed in each row; the kernel of
    // its rows is the set of the v with v(x)^p ≡ v(x) mod f.
    let k = fp.k();
    let m = &fp.m;
    let mut work = super::super::modular::Work::default();
    let mut t = vec![0u64; k];
    let mut pivots: Vec<usize> = Vec::new(); // the pivot column of each row
    let mut row = 0usize;
    for col in 0..n {
        // A row from `row` with a nonzero entry in `col`.
        let Some(r) = (row..n).find(|&r| q[r][col * k..(col + 1) * k].iter().any(|&l| l != 0))
        else {
            continue;
        };
        q.swap(row, r);
        // Make the pivot 1.
        let inv = m.inv(&q[row][col * k..(col + 1) * k]).expect("p is prime");
        if let Some(w) = fp.w {
            let inv = inv[0];
            for x in q[row].iter_mut() {
                *x = w.mul(*x, inv);
            }
        } else {
            for j in 0..n {
                let e = q[row][j * k..(j + 1) * k].to_vec();
                m.mul(&mut q[row][j * k..(j + 1) * k], &e, &inv, &mut work);
            }
        }
        // Clear the column in the other rows.
        let prow = q[row].clone();
        for (r, qr) in q.iter_mut().enumerate() {
            if r == row {
                continue;
            }
            let c = qr[col * k..(col + 1) * k].to_vec();
            if c.iter().all(|&l| l == 0) {
                continue;
            }
            if let Some(w) = fp.w {
                let c = w.neg(c[0]);
                if w.n >> 63 == 0 {
                    // Shoup's product by the fixed c: two products and one correction.
                    let cs = w.shoup(c);
                    for (x, y) in qr.iter_mut().zip(&prow).skip(col) {
                        *x = w.add(*x, w.mul_shoup(*y, c, cs));
                    }
                } else {
                    for (x, y) in qr.iter_mut().zip(&prow).skip(col) {
                        if *y != 0 {
                            *x = w.add(*x, w.mul(c, *y));
                        }
                    }
                }
            } else {
                for j in col..n {
                    m.mul(&mut t, &c, &prow[j * k..(j + 1) * k], &mut work);
                    m.sub(&mut qr[j * k..(j + 1) * k], &t);
                }
            }
        }
        pivots.push(col);
        row += 1;
        if row == n {
            break;
        }
    }
    // The free columns give the kernel.
    let mut basis = Vec::new();
    let mut is_pivot = vec![false; n];
    for &c in &pivots {
        is_pivot[c] = true;
    }
    for fc in (0..n).filter(|&c| !is_pivot[c]) {
        let mut v = vec![0u64; n * k];
        v[fc * k] = 1;
        for (r, &pc) in pivots.iter().enumerate() {
            let mut e = q[r][fc * k..(fc + 1) * k].to_vec();
            m.neg(&mut e);
            v[pc * k..(pc + 1) * k].copy_from_slice(&e);
        }
        fp.trim(&mut v);
        basis.push(v);
    }
    basis
}

/// The irreducible factors of the monic square-free `f` of degree at least 1 by Berlekamp's
/// method: the kernel of `Q - I`, `Q` the matrix of the Frobenius map on `F_p[x]/(f)`, has the
/// dimension `r` of the number of factors, and its elements split `f` by greatest common
/// divisors, with `v - s` for each `s` in `F_p` for a small `p`, with `v^((p-1)/2) - 1` for random
/// elements `v` of it otherwise.
pub(crate) fn berlekamp_squarefree(fp: &Fp, f: &[u64], state: &mut u64) -> Vec<P> {
    let n = fp.deg(f);
    if n <= 1 {
        return vec![f.to_vec()];
    }
    let k = fp.k();
    let mf = fp.mf(f);
    let sigma = fp.frobenius(&mf);
    // The rows of Q: x^(ip) mod f. The matrix of Q - I transposed: column i of Q is row i.
    let mut qt = vec![vec![0u64; n * k]; n];
    let mut r = fp.one();
    for i in 0..n {
        for (j, c) in r.chunks_exact(k).enumerate() {
            qt[j][i * k..(i + 1) * k].copy_from_slice(c);
        }
        if i + 1 < n {
            r = fp.mulmod(&r, &sigma, &mf);
        }
    }
    for (i, row) in qt.iter_mut().enumerate() {
        let one = fp.one();
        fp.m.sub(&mut row[i * k..(i + 1) * k], &one);
    }
    let basis = kernel(fp, &mut qt, n);
    let r = basis.len();
    if r == 1 {
        return vec![f.to_vec()];
    }
    let mut factors = vec![f.to_vec()];
    let small = fp.m.word().is_some_and(|p| p <= BERLEKAMP_SMALL_P);
    if small {
        let p = fp.m.word().expect("a word");
        'outer: for v in basis.iter().filter(|v| fp.len(v) > 1) {
            let mut i = 0;
            while i < factors.len() {
                let u = factors[i].clone();
                if fp.deg(&u) <= 1 {
                    i += 1;
                    continue;
                }
                let vu = fp.rem(v, &u);
                let mut u = u;
                let mut split = Vec::new();
                for s in 0..p {
                    if fp.deg(&u) <= 1 {
                        break;
                    }
                    let vs = fp.sub(&vu, &[s]);
                    let g = fp.gcd(&u, &vs);
                    let dg = fp.deg(&g);
                    if dg > 0 && dg < fp.deg(&u) {
                        u = fp.div(&u, &g);
                        split.push(g);
                    }
                }
                factors[i] = u;
                factors.extend(split);
                if factors.len() == r {
                    break 'outer;
                }
                i += 1;
            }
        }
    } else {
        // Rounds in a row that split nothing: each splits every part of several factors but with
        // a probability of at most 1/p (two of its factors take the same value), so 64 of them
        // mean the kernel is not what it should be.
        let mut idle = 0u32;
        while factors.len() < r {
            let before = factors.len();
            // A random element of the kernel: on each irreducible factor fᵢ it is a constant cᵢ
            // of F_p, so its minimal polynomial modulo a part u is Π (x - c) over the distinct
            // values on u's factors, and gcd(u, v - c) splits u by them.
            let mut a = Vec::new();
            for v in &basis {
                let c = fp.random(state, 1);
                a = fp.add(&a, &fp.mul(v, &c));
            }
            let mut next = Vec::new();
            for u in factors.drain(..) {
                if fp.deg(&u) <= 1 {
                    next.push(u);
                    continue;
                }
                let au = fp.rem(&a, &u);
                let mp = min_poly(fp, &au, &u, r.min(fp.deg(&u)), state);
                if fp.deg(&mp) < 2 {
                    next.push(u);
                    continue;
                }
                let mut rest = u;
                for (c, _) in roots(fp, &mp) {
                    if fp.deg(&rest) == 0 {
                        break;
                    }
                    let mut cl = fp.m.residue(&c);
                    fp.m.neg(&mut cl);
                    let g = fp.gcd(&rest, &fp.add(&au, &cl));
                    if fp.deg(&g) > 0 {
                        rest = fp.div(&rest, &g);
                        next.push(g);
                    }
                }
                if fp.deg(&rest) > 0 {
                    next.push(rest);
                }
            }
            factors = next;
            idle = if factors.len() > before { 0 } else { idle + 1 };
            assert!(idle < 64, "Berlekamp's splitting made no progress");
        }
    }
    debug_assert_eq!(factors.len(), r);
    factors
}

/// The minimal polynomial of the sequence `⟨w, aⁱ mod u⟩` for `i < 2·bound` and a random `w`, by
/// Berlekamp and Massey's algorithm: a divisor of the minimal polynomial of `a` modulo `u` (of
/// degree at most `bound`), equal to it but for an unlucky `w`; monic.
fn min_poly(fp: &Fp, a: &[u64], u: &[u64], bound: usize, state: &mut u64) -> P {
    let k = fp.k();
    let m = &fp.m;
    let n = fp.deg(u);
    let mu = fp.mf(u);
    let w = fp.random(state, n);
    let mut work = super::super::modular::Work::default();
    let mut t = vec![0u64; k];
    // The sequence.
    let len = 2 * bound;
    let mut seq: Vec<Vec<u64>> = Vec::with_capacity(len);
    let mut pw = fp.one();
    for i in 0..len {
        let mut s = vec![0u64; k];
        for (x, y) in pw.chunks_exact(k).zip(w.chunks_exact(k)) {
            m.mul(&mut t, x, y, &mut work);
            m.add(&mut s, &t);
        }
        seq.push(s);
        if i + 1 < len {
            pw = fp.mulmod(&pw, a, &mu);
        }
    }
    // Berlekamp–Massey: the connection polynomial c (c₀ = 1) of the least length l.
    let zero = vec![0u64; k];
    let mut c: Vec<Vec<u64>> = vec![fp.one()];
    let mut b: Vec<Vec<u64>> = vec![fp.one()];
    let mut l = 0usize;
    let mut shift = 1usize;
    let mut bd = fp.one();
    for i in 0..len {
        // The discrepancy d = s_i + Σ c_j s_(i-j).
        let mut d = seq[i].clone();
        for j in 1..=l.min(c.len() - 1) {
            m.mul(&mut t, &c[j], &seq[i - j], &mut work);
            m.add(&mut d, &t);
        }
        if d == zero {
            shift += 1;
            continue;
        }
        // c -= (d / bd) x^shift b.
        let q = {
            let inv = m.inv(&bd).expect("p is prime");
            let mut q = vec![0u64; k];
            m.mul(&mut q, &d, &inv, &mut work);
            q
        };
        let old = c.clone();
        if c.len() < b.len() + shift {
            c.resize(b.len() + shift, zero.clone());
        }
        for (j, bj) in b.iter().enumerate() {
            m.mul(&mut t, &q, bj, &mut work);
            m.sub(&mut c[j + shift], &t);
        }
        if 2 * l <= i {
            l = i + 1 - l;
            b = old;
            bd = d;
            shift = 1;
        } else {
            shift += 1;
        }
    }
    // The minimal polynomial: x^l c(1/x).
    c.resize(l + 1, zero);
    let mut mp: Vec<u64> = c.iter().rev().flatten().copied().collect();
    fp.trim(&mut mp);
    mp
}

/// Whether the monic `f` is irreducible: Ben-Or's test, `gcd(f, x^(p^i) - x) = 1` for every `i`
/// up to `n/2`, the Frobenius powers as the distinct-degree factorisation takes them (one after
/// another for a small `p` and a short `f`, by baby steps and giant steps otherwise, the products
/// of `l` differences for each interval of degrees), stopping at the first common factor; the
/// roots, the likeliest factors, first.
pub(crate) fn is_irreducible(fp: &Fp, f: &[u64]) -> bool {
    let n = fp.deg(f);
    if fp.len(f) == 0 || n == 0 {
        return false;
    }
    if n == 1 {
        return true;
    }
    let mf = fp.mf(f);
    let sigma = fp.frobenius(&mf);
    let x = fp.rem_mf(&fp.x(), &mf);
    let coprime = |h: &[u64]| fp.is_one(&fp.gcd(f, h));
    if !coprime(&fp.sub(&sigma, &x)) {
        return false;
    }
    if n * fp.m.n.bits() as usize <= DDF_NAIVE {
        let p = fp.m.d.clone();
        let mut h = sigma;
        for _ in 2..=n / 2 {
            h = fp.powmod(&h, &p, &mf);
            if !coprime(&fp.sub(&h, &x)) {
                return false;
            }
        }
        return true;
    }
    let l = baby_steps(n);
    let m = n.div_ceil(2 * l);
    let mut hs: Vec<P> = Vec::with_capacity(l + 1);
    hs.push(x);
    hs.push(sigma);
    baby(fp, &mut hs, l, &mf, n);
    let big = hs[l].clone();
    let pwg = if m > 1 {
        Some(fp.powers(&big, &mf, n, rows(n, m - 1)))
    } else {
        None
    };
    let mut hj = big;
    for j in 1..=m {
        if j > 1 {
            hj = fp.compose(&hj, pwg.as_ref().expect("m > 1"), &mf);
        }
        let mut prod = fp.sub(&hj, &hs[0]);
        for h in &hs[1..l] {
            prod = fp.mulmod(&prod, &fp.sub(&hj, h), &mf);
        }
        if !coprime(&prod) {
            return false;
        }
    }
    true
}

/// The roots of the monic `f` of degree at least 1 with their multiplicities, in increasing
/// order: for each square-free part, its greatest common divisor with `x^p - x` split into
/// factors of degree 1; for `p` up to 64 or the degree, every residue tried by synthetic division
/// instead, each root divided out as often as it divides.
pub(crate) fn roots(fp: &Fp, f: &[u64]) -> Vec<(Number, u32)> {
    let k = fp.k();
    let mut out: Vec<(Vec<u64>, u32)> = Vec::new();
    let mut state = SEED;
    if let Some(w) = fp.w
        && (w.n <= 64 || w.n <= fp.deg(f) as u64)
    {
        // Every residue by synthetic division, each root divided out as often as it divides: no
        // greatest common divisor at all.
        let mut g = f.to_vec();
        let mut q = Vec::with_capacity(g.len());
        for x in 0..w.n {
            let mut e = 0u32;
            while g.len() >= 2 {
                // g = (x - c)·q + r, by Horner's rule from the top.
                q.clear();
                let mut v = 0u64;
                for &c in g.iter().rev() {
                    v = w.add(w.mul(v, x), c);
                    q.push(v);
                }
                if v != 0 {
                    break;
                }
                q.pop();
                q.reverse();
                core::mem::swap(&mut g, &mut q);
                e += 1;
            }
            if e > 0 {
                out.push((vec![x], e));
            }
            if g.len() < 2 {
                break;
            }
        }
        return out
            .into_iter()
            .map(|(r, e)| (from_limbs(&r, ByteOrder::LittleEndian), e))
            .collect();
    }
    for (g, e) in squarefree(fp, f) {
        let n = fp.deg(&g);
        match fp.w {
            Some(w) if w.n <= 64 || w.n <= n as u64 => {
                for x in 0..w.n {
                    let mut v = 0u64;
                    for &c in g.iter().rev() {
                        v = w.add(w.mul(v, x), c);
                    }
                    if v == 0 {
                        out.push((vec![x], e));
                    }
                }
            }
            _ => {
                let mf = fp.mf(&g);
                let sigma = fp.frobenius(&mf);
                let h = fp.gcd(&g, &fp.sub_x(&sigma));
                let mut lin = Vec::new();
                edf(fp, &h, 1, Some(&sigma), &mut state, &mut lin);
                for l in lin {
                    // x + c: the root -c.
                    let mut c = l[..k].to_vec();
                    fp.m.neg(&mut c);
                    out.push((c, e));
                }
            }
        }
    }
    out.sort_by(|a, b| cmp_packed(&a.0, &b.0, k));
    out.into_iter()
        .map(|(r, e)| (from_limbs(&r, ByteOrder::LittleEndian), e))
        .collect()
}
