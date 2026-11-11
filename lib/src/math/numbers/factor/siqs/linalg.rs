//! Linear algebra over GF(2): vectors of the null space of the sparse matrix whose columns are the
//! relations (sets of relations) and whose rows are −1 and the primes of the factor base, a bit set
//! where the prime divides the product to an odd power.
//!
//! - Gaussian elimination on the dense rows, by the method of the four Russians, up to
//!   [`GAUSS_MAX`] columns, and should Lanczos fail on matrices not much larger.
//! - Montgomery's block Lanczos method (1995), on 64 vectors at once in the bits of a word: the
//!   iteration on the symmetric `BᵀB`, with Montgomery's choice of the subspace of each step, then
//!   the vectors of the null space of `B` among the combinations of the solution and the last
//!   iterate, by Gaussian elimination on their images. The arrangement follows msieve's.

use super::poly::Random;

/// A sparse matrix over GF(2), by columns.
pub(super) struct Matrix {
    /// The number of rows.
    pub(super) rows: usize,
    /// Where each column's rows start in `entries`, and the end.
    pub(super) start: Vec<u32>,
    /// The rows of the columns, each column's increasing.
    pub(super) entries: Vec<u32>,
}

impl Matrix {
    /// The number of columns.
    pub(super) fn cols(&self) -> usize {
        self.start.len() - 1
    }

    /// The rows of column `c`.
    fn col(&self, c: usize) -> &[u32] {
        &self.entries[self.start[c] as usize..self.start[c + 1] as usize]
    }

    /// `out = B·x`.
    fn mul(&self, x: &[u64], out: &mut [u64]) {
        out.fill(0);
        for (c, &xc) in x.iter().enumerate().take(self.cols()) {
            if xc != 0 {
                for &r in self.col(c) {
                    out[r as usize] ^= xc;
                }
            }
        }
    }

    /// `out = Bᵀ·y`.
    fn mul_transpose(&self, y: &[u64], out: &mut [u64]) {
        for (c, o) in out.iter_mut().enumerate().take(self.cols()) {
            let mut acc = 0;
            for &r in self.col(c) {
                acc ^= y[r as usize];
            }
            *o = acc;
        }
    }
}

/// The most columns of the matrices solved by Gaussian elimination rather than block Lanczos:
/// measured on an Apple M4 Max, elimination took 8 ms for 3800 columns against 10 for Lanczos,
/// and 60 ms for 7200 against 36.
pub(super) const GAUSS_MAX: usize = 5000;

/// Vectors of the null space of `m`: bit `t` of word `c` is the coefficient of column `c` in vector
/// `t`; the words of zero vectors are cleared. Empty when none is found.
pub(super) fn null_space(m: &Matrix, seed: u64) -> Vec<u64> {
    let n = m.cols();
    if n == 0 {
        return Vec::new();
    }
    let gauss_max = GAUSS_MAX;
    if n <= gauss_max {
        return gauss(m);
    }
    let mut random = Random::new(seed);
    for _ in 0..4 {
        if let Some(v) = lanczos(m, &mut random) {
            return v;
        }
    }
    if n <= 6000 { gauss(m) } else { Vec::new() }
}

/// The 64×64 matrix `xᵀy` for `x`, `y` of `n` rows of 64 bits: row `i` is the sum of the rows of `y`
/// whose row of `x` has bit `i` set; by eight tables of the bytes of `x`.
fn inner(x: &[u64], y: &[u64]) -> [u64; 64] {
    if x.len() < 2048 {
        // Few rows: their bits one by one, cheaper than the tables.
        let mut r = [0u64; 64];
        for (&a, &b) in x.iter().zip(y.iter()) {
            let mut a = a;
            while a != 0 {
                r[a.trailing_zeros() as usize] ^= b;
                a &= a - 1;
            }
        }
        return r;
    }
    let mut t = vec![[0u64; 256]; 8];
    for (&a, &b) in x.iter().zip(y.iter()) {
        for (k, tk) in t.iter_mut().enumerate() {
            tk[(a >> (8 * k)) as usize & 255] ^= b;
        }
    }
    let mut r = [0u64; 64];
    for (k, tk) in t.iter().enumerate() {
        for bit in 0..8 {
            let mut acc = 0;
            for (byte, &v) in tk.iter().enumerate() {
                if byte >> bit & 1 == 1 {
                    acc ^= v;
                }
            }
            r[8 * k + bit] = acc;
        }
    }
    r
}

/// `out ^= v·m` (or `out = v·m`) for `v` of `n` rows of 64 bits and the 64×64 matrix `m`, by eight
/// tables of the sums of the rows of `m` for each byte.
fn mul_64(v: &[u64], m: &[u64; 64], out: &mut [u64], accumulate: bool) {
    let mut t = vec![[0u64; 256]; 8];
    for (k, tk) in t.iter_mut().enumerate() {
        for byte in 1..256 {
            let low = byte & (byte - 1);
            let bit = (byte as u32).trailing_zeros() as usize;
            tk[byte] = tk[low] ^ m[8 * k + bit];
        }
    }
    for (o, &a) in out.iter_mut().zip(v.iter()) {
        let mut acc = 0;
        for (k, tk) in t.iter().enumerate() {
            acc ^= tk[(a >> (8 * k)) as usize & 255];
        }
        *o = if accumulate { *o ^ acc } else { acc };
    }
}

/// The product of two 64×64 matrices.
fn mul_64x64(a: &[u64; 64], b: &[u64; 64]) -> [u64; 64] {
    let mut r = [0u64; 64];
    for (ri, &ai) in r.iter_mut().zip(a.iter()) {
        let mut x = ai;
        let mut acc = 0;
        while x != 0 {
            acc ^= b[x.trailing_zeros() as usize];
            x &= x - 1;
        }
        *ri = acc;
    }
    r
}

/// Montgomery's choice of the columns `s` and the inverse `w` of the submatrix of `t = VᵀAV` they
/// make, with every column not chosen last time chosen now; the number chosen, or `None`.
fn choose(
    t: &[u64; 64],
    s: &mut [usize; 64],
    last: &[usize; 64],
    last_dim: usize,
    w: &mut [u64; 64],
) -> Option<usize> {
    let mut mm = [[0u64; 2]; 64];
    for i in 0..64 {
        mm[i] = [t[i], 1u64 << i];
    }
    // The columns of the last choice at the back.
    let mut cols = 64;
    let mut mask = 0u64;
    for &c in &last[..last_dim] {
        cols -= 1;
        s[cols] = c;
        mask |= 1 << c;
    }
    let mut dim = 0;
    for i in 0..64 {
        if mask >> i & 1 == 0 {
            s[dim] = i;
            dim += 1;
        }
    }
    dim = 0;
    for i in 0..64 {
        let bit = 1u64 << s[i];
        // A pivot row for column s[i].
        let mut found = None;
        for j in i..64 {
            if mm[s[j]][0] & bit != 0 {
                found = Some(j);
                break;
            }
        }
        if let Some(j) = found {
            mm.swap(s[i], s[j]);
            let pivot = mm[s[i]];
            for j in 0..64 {
                if j != i && mm[s[j]][0] & bit != 0 {
                    mm[s[j]][0] ^= pivot[0];
                    mm[s[j]][1] ^= pivot[1];
                }
            }
            s[dim] = s[i];
            dim += 1;
            continue;
        }
        // No pivot: the right half makes up for it.
        let mut found = None;
        for j in i..64 {
            if mm[s[j]][1] & bit != 0 {
                found = Some(j);
                break;
            }
        }
        let j = found?;
        mm.swap(s[i], s[j]);
        let pivot = mm[s[i]];
        for j in 0..64 {
            if j != i && mm[s[j]][1] & bit != 0 {
                mm[s[j]][0] ^= pivot[0];
                mm[s[j]][1] ^= pivot[1];
            }
        }
        mm[s[i]] = [0, 0];
    }
    for i in 0..64 {
        w[i] = mm[i][1];
    }
    let mut mask = 0u64;
    for &c in &s[..dim] {
        mask |= 1 << c;
    }
    for &c in &last[..last_dim] {
        mask |= 1 << c;
    }
    if mask != u64::MAX {
        return None;
    }
    Some(dim)
}

/// Block Lanczos: up to 64 vectors of the null space, or `None` when the iteration breaks down.
fn lanczos(m: &Matrix, random: &mut Random) -> Option<Vec<u64>> {
    let n = m.cols();
    let mut tmp = vec![0u64; m.rows];
    let a_mul = |x: &[u64], out: &mut [u64], tmp: &mut [u64]| {
        m.mul(x, tmp);
        m.mul_transpose(tmp, out);
    };
    let y: Vec<u64> = (0..n).map(|_| random.next()).collect();
    let mut v0 = vec![0u64; n];
    a_mul(&y, &mut v0, &mut tmp);
    let mut v = [v0.clone(), vec![0u64; n], vec![0u64; n]];
    let mut vnext = vec![0u64; n];
    let mut x = vec![0u64; n];
    let mut winv = [[0u64; 64]; 3];
    let mut vtav = [[0u64; 64]; 2];
    let mut vta2v = [[0u64; 64]; 2];
    let mut s = [[0usize; 64]; 2];
    for (i, c) in s[1].iter_mut().enumerate() {
        *c = i;
    }
    let mut dim1 = 64;
    let mut mask1 = u64::MAX;
    let mut id = [0u64; 64];
    for (i, r) in id.iter_mut().enumerate() {
        *r = 1 << i;
    }
    let limit = n / 60 + 100;
    let mut iter = 0;
    loop {
        iter += 1;
        if iter > limit {
            return None;
        }
        a_mul(&v[0], &mut vnext, &mut tmp);
        vtav[0] = inner(&v[0], &vnext);
        vta2v[0] = inner(&vnext, &vnext);
        if vtav[0].iter().all(|&r| r == 0) {
            break;
        }
        let (s0, s1) = s.split_at_mut(1);
        let mut w0 = [0u64; 64];
        let dim0 = choose(&vtav[0], &mut s0[0], &s1[0], dim1, &mut w0)?;
        if dim0 == 0 {
            return None;
        }
        winv[0] = w0;
        let mut mask0 = 0u64;
        for &c in &s[0][..dim0] {
            mask0 |= 1 << c;
        }
        for e in vnext.iter_mut() {
            *e &= mask0;
        }
        let vt_v0 = inner(&v[0], &v0);
        // D = Winv0 (VtA2V S Sᵀ + VtAV) + I
        let mut d = [0u64; 64];
        for i in 0..64 {
            d[i] = (vta2v[0][i] & mask0) ^ vtav[0][i];
        }
        let mut d = mul_64x64(&winv[0], &d);
        for i in 0..64 {
            d[i] ^= id[i];
        }
        // E = Winv1 VtAV0 S Sᵀ
        let mut e = mul_64x64(&winv[1], &vtav[0]);
        for r in e.iter_mut() {
            *r &= mask0;
        }
        // F = Winv2 (I + VtAV1 Winv1) ((VtA2V1 S1 S1ᵀ + VtAV1) S Sᵀ)
        let mut f = mul_64x64(&vtav[1], &winv[1]);
        for i in 0..64 {
            f[i] ^= id[i];
        }
        let f = mul_64x64(&winv[2], &f);
        let mut f2 = [0u64; 64];
        for i in 0..64 {
            f2[i] = ((vta2v[1][i] & mask1) ^ vtav[1][i]) & mask0;
        }
        let f = mul_64x64(&f, &f2);
        mul_64(&v[0], &d, &mut vnext, true);
        mul_64(&v[1], &e, &mut vnext, true);
        mul_64(&v[2], &f, &mut vnext, true);
        // X += V0 Winv0 V0ᵀ v0
        let dx = mul_64x64(&winv[0], &vt_v0);
        mul_64(&v[0], &dx, &mut x, true);
        // Rotate.
        v.rotate_right(1);
        core::mem::swap(&mut v[0], &mut vnext);
        winv[2] = winv[1];
        winv[1] = winv[0];
        vtav[1] = vtav[0];
        vta2v[1] = vta2v[0];
        s[1] = s[0];
        mask1 = mask0;
        dim1 = dim0;
    }
    // A(X + Y) = 0, and the last V has VᵀAV = 0: the null vectors of B are combinations of them.
    for (xi, &yi) in x.iter_mut().zip(y.iter()) {
        *xi ^= yi;
    }
    let out = combine(m, &x, &v[0]);
    if out.iter().all(|&w| w == 0) {
        None
    } else {
        Some(out)
    }
}

/// The combinations of the 128 columns of `x` and `v` that `B` takes to zero, as vectors.
fn combine(m: &Matrix, x: &[u64], v: &[u64]) -> Vec<u64> {
    let n = m.cols();
    let rows = m.rows;
    let words = rows.div_ceil(64).max(1);
    let mut bx = vec![0u64; rows];
    let mut bv = vec![0u64; rows];
    m.mul(x, &mut bx);
    m.mul(v, &mut bv);
    // The images, one vector of `rows` bits for each of the 128 columns.
    let mut img = vec![0u64; 128 * words];
    for r in 0..rows {
        let (wd, bit) = (r / 64, 1u64 << (r % 64));
        let mut a = bx[r];
        while a != 0 {
            let t = a.trailing_zeros() as usize;
            img[t * words + wd] |= bit;
            a &= a - 1;
        }
        let mut b = bv[r];
        while b != 0 {
            let t = b.trailing_zeros() as usize;
            img[(64 + t) * words + wd] |= bit;
            b &= b - 1;
        }
    }
    let mut comb: Vec<u128> = (0..128).map(|i| 1u128 << i).collect();
    let mut rank = 0;
    for r in 0..rows {
        let (wd, bit) = (r / 64, 1u64 << (r % 64));
        let Some(p) = (rank..128).find(|&i| img[i * words + wd] & bit != 0) else {
            continue;
        };
        if p != rank {
            for k in 0..words {
                img.swap(p * words + k, rank * words + k);
            }
            comb.swap(p, rank);
        }
        for i in rank + 1..128 {
            if img[i * words + wd] & bit != 0 {
                for k in 0..words {
                    img[i * words + k] ^= img[rank * words + k];
                }
                comb[i] ^= comb[rank];
            }
        }
        rank += 1;
        if rank == 128 {
            break;
        }
    }
    let combos: Vec<u128> = comb[rank..].iter().copied().take(64).collect();
    let mut out = vec![0u64; n];
    for (c, o) in out.iter_mut().enumerate() {
        let mut word = 0u64;
        for (t, &k) in combos.iter().enumerate() {
            let lo = k as u64;
            let hi = (k >> 64) as u64;
            let bit = ((x[c] & lo).count_ones() + (v[c] & hi).count_ones()) & 1;
            word |= u64::from(bit) << t;
        }
        *o = word;
    }
    clean(m, out)
}

/// Clears the vectors (bits) of `out` that are zero or that `B` does not take to zero.
fn clean(m: &Matrix, out: Vec<u64>) -> Vec<u64> {
    let mut check = vec![0u64; m.rows];
    m.mul(&out, &mut check);
    let mut bad = 0u64;
    for &r in &check {
        bad |= r;
    }
    let mut any = 0u64;
    for &o in &out {
        any |= o;
    }
    let keep = any & !bad;
    out.into_iter().map(|o| o & keep).collect()
}

/// Up to 64 combinations of columns that sum to zero, by Gaussian elimination on the dense rows:
/// the columns taken eight at a time by the method of the four Russians (Arlazarov, Dinic,
/// Kronrod and Faradžev 1970; Albrecht and Bard's M4RI), up to eight pivot rows for them, made to
/// have one pivot bit each, then each other row reduced by one table look-up of the sums of the
/// pivots for its eight bits rather than by up to eight sums after tests the branch predictor
/// misses; then the combinations, 64 at once in the bits of words, by substitution from the last
/// pivot row up, each with one free column set.
fn gauss(m: &Matrix) -> Vec<u64> {
    let n = m.cols();
    let rows = m.rows;
    let width = n.div_ceil(64).max(1);
    let mut a = vec![0u64; rows * width];
    for c in 0..n {
        for &r in m.col(c) {
            a[r as usize * width + c / 64] ^= 1 << (c % 64);
        }
    }
    let mut table = vec![0u64; 256 * width];
    let mut pivot_col = Vec::with_capacity(rows);
    let mut rank = 0;
    let mut c0 = 0;
    while c0 < n && rank < rows {
        // The window of columns [c0, c0 + kw), within word `wd`: the rows from `rank` on are zero
        // in the columns before it. The tables have 2^kw sums, about half the rows left.
        let (wd, sh) = (c0 / 64, c0 % 64);
        let left = rows - rank;
        let kw = ((usize::BITS - left.leading_zeros()) as usize)
            .saturating_sub(1)
            .clamp(1, 8)
            .min(64 - sh);
        let mask = (1u32 << kw) - 1;
        let bits = |a: &[u64], i: usize| (a[i * width + wd] >> sh) as u32 & mask;
        let mut k = 0;
        let mut pivot_bit = [0u32; 8];
        let mut pivot_bits = [0u32; 8];
        let mut i = rank;
        while i < rows && k < kw {
            let mut w = bits(&a, i);
            if w != 0 {
                let mut used = 0u32;
                for j in 0..k {
                    if w >> pivot_bit[j] & 1 == 1 {
                        w ^= pivot_bits[j];
                        used |= 1 << j;
                    }
                }
                if w != 0 {
                    for j in 0..k {
                        if used >> j & 1 == 1 {
                            xor_rows(&mut a, i, rank + j, wd, width);
                        }
                    }
                    swap_rows(&mut a, i, rank + k, width);
                    pivot_bit[k] = w.trailing_zeros();
                    pivot_bits[k] = w;
                    k += 1;
                }
            }
            i += 1;
        }
        // Each pivot without the pivot bits of the others, the later ones cleared first.
        for j in (0..k).rev() {
            for l in 0..j {
                if pivot_bits[l] >> pivot_bit[j] & 1 == 1 {
                    pivot_bits[l] ^= pivot_bits[j];
                    xor_rows(&mut a, rank + l, rank + j, wd, width);
                }
            }
        }
        // The rows below the pivots, each reduced by the sum of the pivots its bits take.
        let span = width - wd;
        if rank + k < rows && k > 0 {
            for s in 1..1usize << k {
                let low = s & (s - 1);
                let j = s.trailing_zeros() as usize;
                let (lo, hi) = table.split_at_mut(s * width);
                let (dst, prev) = (&mut hi[..span], &lo[low * width..low * width + span]);
                let pivot = &a[(rank + j) * width + wd..(rank + j + 1) * width];
                for (x, (&y, &z)) in dst.iter_mut().zip(prev.iter().zip(pivot)) {
                    *x = y ^ z;
                }
            }
            // The sum each value of the window's bits takes: the pivots of its set pivot bits.
            let mut choice = [0u8; 256];
            let mut of_bit = [0u8; 8];
            for j in 0..k {
                of_bit[pivot_bit[j] as usize] = 1 << j;
            }
            for w in 1..1usize << kw {
                choice[w] = choice[w & (w - 1)] | of_bit[w.trailing_zeros() as usize];
            }
            for i in rank + k..rows {
                let s = choice[bits(&a, i) as usize] as usize;
                let (row, sum) = (
                    &mut a[i * width + wd..(i + 1) * width],
                    &table[s * width..s * width + span],
                );
                for (x, &y) in row.iter_mut().zip(sum) {
                    *x ^= y;
                }
            }
        }
        for &b in &pivot_bit[..k] {
            pivot_col.push(c0 + b as usize);
        }
        rank += k;
        c0 += kw;
    }
    // The free columns, the first 64 of them each set in one combination; then each pivot
    // column, from the last pivot row up, the sum of the other columns of its row.
    let mut is_pivot = vec![false; n];
    for &c in &pivot_col {
        is_pivot[c] = true;
    }
    let mut out = vec![0u64; n];
    let mut t = 0;
    for (c, o) in out.iter_mut().enumerate() {
        if !is_pivot[c] && t < 64 {
            *o = 1 << t;
            t += 1;
        }
    }
    for i in (0..rank).rev() {
        let pc = pivot_col[i];
        let mut acc = 0u64;
        for (wi, &word) in a[i * width..(i + 1) * width].iter().enumerate() {
            let mut x = word;
            if wi == pc / 64 {
                x &= !(1 << (pc % 64));
            }
            while x != 0 {
                acc ^= out[64 * wi + x.trailing_zeros() as usize];
                x &= x - 1;
            }
        }
        out[pc] = acc;
    }
    clean(m, out)
}

/// Adds the words `from..width` of row `src` of `a` to row `dst`.
fn xor_rows(a: &mut [u64], dst: usize, src: usize, from: usize, width: usize) {
    let (d, s) = if dst < src {
        let (lo, hi) = a.split_at_mut(src * width);
        (
            &mut lo[dst * width + from..(dst + 1) * width],
            &hi[from..width],
        )
    } else {
        let (lo, hi) = a.split_at_mut(dst * width);
        (
            &mut hi[from..width],
            &lo[src * width + from..(src + 1) * width],
        )
    };
    for (x, &y) in d.iter_mut().zip(s) {
        *x ^= y;
    }
}

/// Swaps rows `i` and `j` of `a`.
fn swap_rows(a: &mut [u64], i: usize, j: usize, width: usize) {
    if i != j {
        let (lo, hi) = a.split_at_mut(i.max(j) * width);
        lo[i.min(j) * width..(i.min(j) + 1) * width].swap_with_slice(&mut hi[..width]);
    }
}
