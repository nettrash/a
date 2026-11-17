//! The passes of the real transforms in portable Rust: the reference for the vector kernels,
//! which compute the same expressions lane by lane, and their remainders.

use crate::math::complex::{Complex, Float};

/// The pairs `(k, m − k)` for `k` from `from` up to `(m − 1)/2` of the pass between a complex
/// transform of `m` values and the spectrum of `2m` reals:
///
/// with `a = src[k]`, `b = conj src[m − k]`, `u = s·(a + b)` and `t = tw[k]·(a − b)`,
/// `dst[k] = u + t` and `dst[m − k] = conj(u − t)`.
///
/// The forward pass takes `s = 1/2` and `tw[k] = −i·e^(−2πi·k/2m)/2`, the inverse `s = 1` and
/// `tw[k] = i·e^(2πi·k/2m)`.
///
/// # Safety
///
/// `src` and `dst` are valid for `m` values and are either the same or disjoint; `tw` is valid
/// for `(m + 1)/2` values.
#[inline]
pub(in crate::math::fft) unsafe fn pairs<T: Float>(
    dst: *mut Complex<T>,
    src: *const Complex<T>,
    m: usize,
    tw: *const Complex<T>,
    s: T,
    from: usize,
) {
    let mut k = from;
    while 2 * k < m {
        // SAFETY: 1 ≤ k < m − k < m; both values are read before either is written.
        unsafe {
            let a = *src.add(k);
            let b = (*src.add(m - k)).conj();
            let w = *tw.add(k);
            let (ur, ui) = ((a.re + b.re) * s, (a.im + b.im) * s);
            let (dr, di) = (a.re - b.re, a.im - b.im);
            let tr = w.re * dr - w.im * di;
            let ti = w.re * di + w.im * dr;
            *dst.add(k) = Complex::new(ur + tr, ui + ti);
            *dst.add(m - k) = Complex::new(ur - tr, ti - ui);
        }
        k += 1;
    }
}
