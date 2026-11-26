//! Products of matrices of numbers whose elements all fit in a word, the sums of products in
//! machine arithmetic: in `f64`s when every sum is below `2^53`, so that the floating-point
//! product of [`gemm`](super::super::blas::gemm), every intermediate an integer it represents
//! exactly, computes them exactly at the speed of its kernels; in `i128`s when the magnitudes are
//! below `2^63` and the sums below `2^127`; in three words otherwise. The results are the numbers
//! of the classical product, little-endian.

use super::super::blas::{Op, gemm};
use crate::math::numbers::{ByteOrder, Magnitude, Number, from_limbs};

/// The largest sum of products the `f64` product computes exactly: every partial sum is an
/// integer of at most this magnitude, which a double holds.
const F64_EXACT: u128 = 1 << 53;

/// The elements of a matrix as words: their magnitudes, and their signs as masks (all ones for a
/// negative element, zero otherwise and for zero), in the order given.
struct Words {
    mag: Vec<u64>,
    sign: Vec<u64>,
    max: u64,
}

/// The words of `x`, or `None` if an element passes 64 bits.
fn words(x: &[Number]) -> Option<Words> {
    // A first pass that allocates nothing decides.
    let mut max = 0u64;
    for v in x {
        let m = Magnitude::of(v);
        if m.bytes.len() > 8 {
            return None;
        }
        max = max.max(m.small() as u64);
    }
    let mut mag = Vec::with_capacity(x.len());
    let mut sign = Vec::with_capacity(x.len());
    for v in x {
        let m = Magnitude::of(v);
        mag.push(m.small() as u64);
        sign.push(if m.negative { u64::MAX } else { 0 });
    }
    Some(Words { mag, sign, max })
}

/// `c = a·b` for column-major `a` of `m × k`, `b` of `k × n` and `c` of `m × n` (which holds
/// zeros), when every element of `a` and `b` fits in a word; `false`, with `c` untouched,
/// otherwise.
pub(super) fn product(
    c: &mut [Number],
    a: &[Number],
    b: &[Number],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let Some(wa) = words(a) else { return false };
    let Some(wb) = words(b) else { return false };
    if wa.max == 0 || wb.max == 0 {
        return true;
    }
    // Every sum of products is at most k·max|a|·max|b| in magnitude.
    let bound = (wa.max as u128 * wb.max as u128).checked_mul(k as u128);
    match bound {
        Some(t) if t < F64_EXACT => by_f64(c, &wa, &wb, m, k, n),
        Some(t) if t < 1u128 << 127 && wa.max < 1 << 63 && wb.max < 1 << 63 => {
            by_i128(c, &wa, &wb, m, k, n)
        }
        _ => by_three_words(c, &wa, &wb, m, k, n),
    }
    true
}

/// The sums in doubles: exact, as every partial sum of the product is an integer below `2^53`.
fn by_f64(c: &mut [Number], wa: &Words, wb: &Words, m: usize, k: usize, n: usize) {
    let to = |w: &Words| -> Vec<f64> {
        w.mag
            .iter()
            .zip(&w.sign)
            .map(|(&x, &s)| if s != 0 { -(x as f64) } else { x as f64 })
            .collect()
    };
    let (fa, fb) = (to(wa), to(wb));
    let mut fc = vec![0.0f64; m * n];
    gemm(m, n, k, 1.0, &fa, m, Op::N, &fb, k, Op::N, 0.0, &mut fc, m);
    for (c, &x) in c.iter_mut().zip(&fc) {
        if x != 0.0 {
            *c = Number::from(x as i64);
        }
    }
}

/// The signed values of words below `2^63`, row-major for `a` (its rows contiguous) and
/// column-major for `b` (its columns contiguous), so that each sum reads two runs of memory.
fn signed_rows(w: &Words, rows: usize, cols: usize, transpose: bool) -> Vec<i64> {
    let v = |i: usize| {
        let x = w.mag[i] as i64;
        if w.sign[i] != 0 { -x } else { x }
    };
    if transpose {
        // Column-major `rows × cols` to row-major.
        let mut out = vec![0i64; rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                out[i * cols + j] = v(i + j * rows);
            }
        }
        out
    } else {
        (0..rows * cols).map(v).collect()
    }
}

/// The sums in `i128`s: magnitudes below `2^63`, sums below `2^127`. Four columns of `b` at a
/// time, so that each element of a row of `a` read serves four sums.
fn by_i128(c: &mut [Number], wa: &Words, wb: &Words, m: usize, k: usize, n: usize) {
    let ar = signed_rows(wa, m, k, true);
    let bc = signed_rows(wb, k, n, false);
    let put = |c: &mut [Number], at: usize, s: i128| {
        if s != 0 {
            c[at] = Number::from(s);
        }
    };
    let mut j = 0;
    while j + 4 <= n {
        let cols = [
            &bc[j * k..(j + 1) * k],
            &bc[(j + 1) * k..(j + 2) * k],
            &bc[(j + 2) * k..(j + 3) * k],
            &bc[(j + 3) * k..(j + 4) * k],
        ];
        for i in 0..m {
            let row = &ar[i * k..(i + 1) * k];
            let mut s = [0i128; 4];
            for p in 0..k {
                let x = row[p] as i128;
                s[0] += x * cols[0][p] as i128;
                s[1] += x * cols[1][p] as i128;
                s[2] += x * cols[2][p] as i128;
                s[3] += x * cols[3][p] as i128;
            }
            for (d, &v) in s.iter().enumerate() {
                put(c, i + (j + d) * m, v);
            }
        }
        j += 4;
    }
    for j in j..n {
        let col = &bc[j * k..(j + 1) * k];
        for i in 0..m {
            let row = &ar[i * k..(i + 1) * k];
            let s: i128 = row
                .iter()
                .zip(col)
                .map(|(&x, &y)| x as i128 * y as i128)
                .sum();
            put(c, i + j * m, s);
        }
    }
}

/// The sums in three words, two's complement: the magnitudes of the products in 128 bits, each
/// negated by its sign mask before it is added.
fn by_three_words(c: &mut [Number], wa: &Words, wb: &Words, m: usize, k: usize, n: usize) {
    // `a` row-major, `b` column-major, magnitudes and signs side by side.
    let mut am = vec![0u64; m * k];
    let mut asg = vec![0u64; m * k];
    for p in 0..k {
        for i in 0..m {
            am[i * k + p] = wa.mag[i + p * m];
            asg[i * k + p] = wa.sign[i + p * m];
        }
    }
    /// One term: `acc += ±x·y` in 192-bit two's complement, `(lo, hi)`.
    #[inline(always)]
    fn term(acc: &mut (u128, u64), x: u64, xs: u64, y: u64, ys: u64) {
        let prod = x as u128 * y as u128;
        // The sign of the product, zero for a zero product: a negative product's negation is
        // !prod + 1 over 192 bits, all ones on top, and as prod is not zero, !prod + 1 does not
        // carry past 128 bits.
        let s = (xs ^ ys) & ((prod != 0) as u64).wrapping_neg();
        let mask = ((s as u128) << 64) | s as u128;
        let t = (prod ^ mask).wrapping_add((s & 1) as u128);
        let (r, carry) = acc.0.overflowing_add(t);
        acc.0 = r;
        acc.1 = acc.1.wrapping_add(s).wrapping_add(carry as u64);
    }
    /// The number of a 192-bit two's complement integer.
    fn number((lo, hi): (u128, u64)) -> Option<Number> {
        if lo == 0 && hi == 0 {
            return None;
        }
        let negative = (hi as i64) < 0;
        let (lo, hi) = if negative {
            let (l, borrow) = (!lo).overflowing_add(1);
            (l, (!hi).wrapping_add(borrow as u64))
        } else {
            (lo, hi)
        };
        let mut x = from_limbs(&[lo as u64, (lo >> 64) as u64, hi], ByteOrder::LittleEndian);
        x.negative = negative;
        Some(x)
    }
    // Four columns of `b` at a time, so that each element of a row of `a` serves four sums.
    let mut j = 0;
    while j < n {
        let w = (n - j).min(4);
        for i in 0..m {
            let (xm, xs) = (&am[i * k..(i + 1) * k], &asg[i * k..(i + 1) * k]);
            let mut acc = [(0u128, 0u64); 4];
            if w == 4 {
                let col = |d: usize| (j + d) * k;
                let (c0, c1, c2, c3) = (col(0), col(1), col(2), col(3));
                for p in 0..k {
                    let (x, s) = (xm[p], xs[p]);
                    term(&mut acc[0], x, s, wb.mag[c0 + p], wb.sign[c0 + p]);
                    term(&mut acc[1], x, s, wb.mag[c1 + p], wb.sign[c1 + p]);
                    term(&mut acc[2], x, s, wb.mag[c2 + p], wb.sign[c2 + p]);
                    term(&mut acc[3], x, s, wb.mag[c3 + p], wb.sign[c3 + p]);
                }
            } else {
                for (d, a) in acc.iter_mut().enumerate().take(w) {
                    let cd = (j + d) * k;
                    for p in 0..k {
                        term(a, xm[p], xs[p], wb.mag[cd + p], wb.sign[cd + p]);
                    }
                }
            }
            for (d, &a) in acc.iter().enumerate().take(w) {
                if let Some(x) = number(a) {
                    c[i + (j + d) * m] = x;
                }
            }
        }
        j += w;
    }
}
