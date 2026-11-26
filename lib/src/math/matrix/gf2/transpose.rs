//! Transposes over `GF(2)`, by blocks of 64 × 64 bits, each transposed in registers by six
//! rounds of swaps of halves (Warren, "Hacker's Delight", 2nd ed., 7–3): in round `s`
//! (32, 16, …, 1) row `r` and row `r + s`, for `r` without the bit `s`, swap the half of `r` in
//! the columns with the bit `s` and the half of `r + s` in the columns without it, which
//! exchanges bit `s` of the row index with bit `s` of the column index; after all six rounds
//! every bit of the two indices is exchanged.

use super::BitMatrix;

/// Transposes the 64 × 64 block whose row `i` is `x[i]`, column `j` bit `j`.
fn transpose64(x: &mut [u64; 64]) {
    let mut s = 32;
    let mut mask: u64 = 0x0000_0000_FFFF_FFFF;
    while s != 0 {
        for base in (0..64).step_by(2 * s) {
            for r in base..base + s {
                let t = ((x[r] >> s) ^ x[r + s]) & mask;
                x[r] ^= t << s;
                x[r + s] ^= t;
            }
        }
        s >>= 1;
        mask ^= mask << s;
    }
}

/// `aᵀ`.
pub(super) fn transpose(a: &BitMatrix) -> BitMatrix {
    let mut t = BitMatrix::zeros(a.cols, a.rows);
    let mut x = [0u64; 64];
    for rb in (0..a.rows).step_by(64) {
        let rows = 64.min(a.rows - rb);
        for w in 0..a.stride {
            for (i, xi) in x.iter_mut().enumerate() {
                // The rows past the last are zero, so the padding of `t` stays zero.
                *xi = if i < rows {
                    a.data[(rb + i) * a.stride + w]
                } else {
                    0
                };
            }
            transpose64(&mut x);
            let cols = 64.min(a.cols - 64 * w);
            for (j, &xj) in x[..cols].iter().enumerate() {
                t.data[(64 * w + j) * t.stride + rb / 64] = xj;
            }
        }
    }
    t
}
