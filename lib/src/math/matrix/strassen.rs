//! Strassen–Winograd multiplication: Winograd's variant of Strassen's method (Winograd 1971; the
//! schedule as Boyer, Dumas, Pernet and Zhou, "Memory efficient scheduling of
//! Strassen–Winograd's matrix multiplication algorithm", ISSAC 2009, write it), seven products of
//! halves and fifteen additions in place of eight products, recursively, down to a classical
//! product that the ring supplies. `O(n^2.81)` products of elements.
//!
//! The operands are row-major views, a slice and the distance between its rows, so that the
//! quadrants are views too; only the sums and the seven products are written to working memory.
//! A dimension that is odd is peeled: the even part goes through the recursion, and the last row,
//! the last column and the rank-one term of the last inner index are classical products.

use crate::math::numbers::Number;

/// A ring for the recursion: its zero, sums and differences, and the classical product.
pub(super) trait Ring {
    /// The elements.
    type E: Clone;

    /// 0.
    fn zero(&self) -> Self::E;

    /// `a + b`.
    fn add(&self, a: &Self::E, b: &Self::E) -> Self::E;

    /// `a - b`.
    fn sub(&self, a: &Self::E, b: &Self::E) -> Self::E;

    /// `C = A·B`, or `C = C + A·B` when `accumulate` is set, classically, for the `m × k` view
    /// `a`, the `k × n` view `b` and the `m × n` view `c`, `shape = (m, k, n)`.
    #[allow(clippy::too_many_arguments)]
    fn classical(
        &self,
        c: &mut [Self::E],
        ldc: usize,
        a: &[Self::E],
        lda: usize,
        b: &[Self::E],
        ldb: usize,
        shape: (usize, usize, usize),
        accumulate: bool,
    );

    /// Whether a product of `m × k` by `k × n` is split once more.
    fn split(&self, m: usize, k: usize, n: usize) -> bool;
}

/// `out = x ± y` for the `m × n` views `x` and `y`, written to the view `out`.
#[allow(clippy::too_many_arguments)]
fn combine_into<R: Ring>(
    r: &R,
    out: &mut [R::E],
    ldo: usize,
    (x, ldx): (&[R::E], usize),
    (y, ldy): (&[R::E], usize),
    m: usize,
    n: usize,
    subtract: bool,
) {
    for i in 0..m {
        let o = &mut out[i * ldo..i * ldo + n];
        let xr = &x[i * ldx..i * ldx + n];
        let yr = &y[i * ldy..i * ldy + n];
        for ((o, a), b) in o.iter_mut().zip(xr).zip(yr) {
            *o = if subtract { r.sub(a, b) } else { r.add(a, b) };
        }
    }
}

/// `x ± y` for the `m × n` views `x` and `y`, as a contiguous `m × n` matrix.
fn combine<R: Ring>(
    r: &R,
    x: (&[R::E], usize),
    y: (&[R::E], usize),
    m: usize,
    n: usize,
    subtract: bool,
) -> Vec<R::E> {
    let mut out = vec![r.zero(); m * n];
    combine_into(r, &mut out, n, x, y, m, n, subtract);
    out
}

/// `C = A·B` for the `m × k` view `a`, the `k × n` view `b` and the `m × n` view `c`, `shape =
/// (m, k, n)`, by Strassen–Winograd's recursion while [`Ring::split`] says so.
#[allow(clippy::too_many_arguments)]
pub(super) fn product<R: Ring>(
    r: &R,
    c: &mut [R::E],
    ldc: usize,
    a: &[R::E],
    lda: usize,
    b: &[R::E],
    ldb: usize,
    (m, k, n): (usize, usize, usize),
) {
    if m == 0 || n == 0 {
        return;
    }
    if m < 2 || k < 2 || n < 2 || !r.split(m, k, n) {
        r.classical(c, ldc, a, lda, b, ldb, (m, k, n), false);
        return;
    }
    let (h, t, w) = (m / 2, k / 2, n / 2);
    // The quadrants of the even parts, as views.
    let a11 = (a, lda);
    let a12 = (&a[t..], lda);
    let a21 = (&a[h * lda..], lda);
    let a22 = (&a[h * lda + t..], lda);
    let b11 = (b, ldb);
    let b12 = (&b[w..], ldb);
    let b21 = (&b[t * ldb..], ldb);
    let b22 = (&b[t * ldb + w..], ldb);
    // The eight sums of the operands.
    let s1 = combine(r, a21, a22, h, t, false);
    let s2 = combine(r, (&s1, t), a11, h, t, true);
    let s3 = combine(r, a11, a21, h, t, true);
    let s4 = combine(r, a12, (&s2, t), h, t, true);
    let t1 = combine(r, b12, b11, t, w, true);
    let t2 = combine(r, b22, (&t1, w), t, w, true);
    let t3 = combine(r, b22, b12, t, w, true);
    let t4 = combine(r, (&t2, w), b21, t, w, true);
    // The seven products of halves.
    let mul = |x: (&[R::E], usize), y: (&[R::E], usize)| {
        let mut out = vec![r.zero(); h * w];
        product(r, &mut out, w, x.0, x.1, y.0, y.1, (h, t, w));
        out
    };
    let p1 = mul(a11, b11);
    let p2 = mul(a12, b21);
    let p3 = mul((&s4, t), b22);
    let p4 = mul(a22, (&t4, w));
    let p5 = mul((&s1, t), (&t1, w));
    let p6 = mul((&s2, t), (&t2, w));
    let p7 = mul((&s3, t), (&t3, w));
    drop((s1, s2, s3, s4, t1, t2, t3, t4));
    // The seven sums of the products: C11 = P1 + P2, C12 = U4 + P3, C21 = U3 - P4, C22 = U3 + P5
    // with U2 = P1 + P6, U3 = U2 + P7 and U4 = U2 + P5.
    combine_into(r, c, ldc, (&p1, w), (&p2, w), h, w, false);
    let u2 = combine(r, (&p1, w), (&p6, w), h, w, false);
    let u3 = combine(r, (&u2, w), (&p7, w), h, w, false);
    let u4 = combine(r, (&u2, w), (&p5, w), h, w, false);
    combine_into(r, &mut c[w..], ldc, (&u4, w), (&p3, w), h, w, false);
    combine_into(r, &mut c[h * ldc..], ldc, (&u3, w), (&p4, w), h, w, true);
    combine_into(
        r,
        &mut c[h * ldc + w..],
        ldc,
        (&u3, w),
        (&p5, w),
        h,
        w,
        false,
    );
    // The peeled parts of odd dimensions.
    if k % 2 == 1 {
        r.classical(
            c,
            ldc,
            &a[k - 1..],
            lda,
            &b[(k - 1) * ldb..],
            ldb,
            (2 * h, 1, 2 * w),
            true,
        );
    }
    if n % 2 == 1 {
        r.classical(
            &mut c[n - 1..],
            ldc,
            a,
            lda,
            &b[n - 1..],
            ldb,
            (m, k, 1),
            false,
        );
    }
    if m % 2 == 1 {
        r.classical(
            &mut c[(m - 1) * ldc..],
            ldc,
            &a[(m - 1) * lda..],
            lda,
            b,
            ldb,
            (1, k, 2 * w),
            false,
        );
    }
}

/// The ring of numbers, its operands of at most `bytes` bytes, for the recursion.
struct Numbers {
    /// The order from which a product is split, by the length of the operands.
    from: usize,
}

impl Ring for Numbers {
    type E = Number;

    fn zero(&self) -> Number {
        Number::default()
    }

    fn add(&self, a: &Number, b: &Number) -> Number {
        a + b
    }

    fn sub(&self, a: &Number, b: &Number) -> Number {
        a - b
    }

    fn classical(
        &self,
        c: &mut [Number],
        ldc: usize,
        a: &[Number],
        lda: usize,
        b: &[Number],
        ldb: usize,
        (m, k, n): (usize, usize, usize),
        accumulate: bool,
    ) {
        for i in 0..m {
            let cr = &mut c[i * ldc..i * ldc + n];
            if !accumulate {
                cr.fill(Number::default());
            }
            for p in 0..k {
                let x = &a[i * lda + p];
                if x.is_zero() {
                    continue;
                }
                for (c, y) in cr.iter_mut().zip(&b[p * ldb..p * ldb + n]) {
                    if !y.is_zero() {
                        *c += x * y;
                    }
                }
            }
        }
    }

    fn split(&self, m: usize, k: usize, n: usize) -> bool {
        m.min(k).min(n) >= self.from
    }
}

/// The order from which a product of matrices of numbers is split into Strassen–Winograd's
/// seven, by the length in bytes of the longest element of the two operands. A sum of numbers
/// costs about as much as a product while they are short, so the seven products save less than
/// the eighth of the work they save for long ones, and the sums lengthen the elements by a few
/// bits, past a limb for some lengths. Measured on an Apple M4 Max in the release profile,
/// `n × n` products, the recursion at `n` and `n / 2` against the classical product: for
/// elements of 8 to 62 bits, 0.97 to 1.03 of its time at 64 and 128 for one level, 0.82 to 0.87
/// at 256 for two; for 128 to 512 bits, 0.99 to 1.07 at 128, 0.94 to 0.96 at 256; for 1024 bits,
/// 0.85 at 16 and 0.77 from 32; for 2048 bits 0.90 to 0.92 from 16, and for 4096 bits 0.92 from 8.
fn crossover(bytes: usize) -> usize {
    match bytes {
        0..=64 => 128,
        65..=256 => 32,
        _ => 16,
    }
}

/// `c = a · b` for column-major matrices of numbers, `a` of `m × k`, `b` of `k × n` and `c` of
/// `m × n`, which holds zeros: classically, column after column, below the crossover of the
/// operands' length ([`crossover`]), and by Strassen–Winograd's recursion above it, which gives
/// the same numbers, little-endian as the classical product leaves them.
pub(super) fn numbers(c: &mut [Number], a: &[Number], b: &[Number], m: usize, k: usize, n: usize) {
    let bytes = a.iter().chain(b).map(|x| x.value.len()).max().unwrap_or(0);
    let r = Numbers {
        from: crossover(bytes),
    };
    // Column-major `C = A·B` is row-major `Cᵀ = Bᵀ·Aᵀ`.
    if m.min(k).min(n) >= r.from {
        product(&r, c, m, b, k, a, m, (n, k, m));
        return;
    }
    for j in 0..n {
        let c = &mut c[j * m..(j + 1) * m];
        for p in 0..k {
            let x = &b[p + j * k];
            if x.is_zero() {
                continue;
            }
            for (c, a) in c.iter_mut().zip(&a[p * m..(p + 1) * m]) {
                if !a.is_zero() {
                    *c += a * x;
                }
            }
        }
    }
}
