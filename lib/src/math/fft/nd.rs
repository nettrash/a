//! Transforms of arrays of two and three dimensions, one dimension after the other: the last
//! (contiguous) one as a batch of rows, each other one by tiles of columns gathered into a
//! buffer, transformed there as a batch of contiguous sequences and scattered back, so that each
//! pass over the array reads and writes whole cache lines.

use super::{Fft, FftFloat, check_batch};
use crate::math::complex::Complex;

/// The two-dimensional discrete Fourier transform of `rows × cols` complex arrays stored row
/// after row (the column index varying fastest), planned once:
/// `X[k, l] = Σ x[i, j]·e^(−2πi·(ik/rows + jl/cols))`, the inverse unscaled
/// (`inverse(forward(x)) = rows·cols·x`). Each method takes a batch of arrays one after the other.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::fft::Fft2;
///
/// let fft = Fft2::<f64>::new(2, 3);
/// let mut x = [Complex::ZERO; 6];
/// x[0] = Complex::ONE;
/// fft.forward(&mut x);
/// assert!(x.iter().all(|&v| v == Complex::ONE));
/// ```
pub struct Fft2<T> {
    rows: Fft<T>,
    cols: Fft<T>,
}

impl<T: FftFloat> Fft2<T> {
    /// The plan for arrays of `rows` rows of `cols` values.
    pub fn new(rows: usize, cols: usize) -> Fft2<T> {
        Fft2 {
            rows: Fft::new(rows),
            cols: Fft::new(cols),
        }
    }

    /// The planned shape, `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows.len(), self.cols.len())
    }

    /// The forward transform of each array of `data`, in place.
    #[track_caller]
    pub fn forward(&self, data: &mut [Complex<T>]) {
        self.apply(data, false);
    }

    /// The unscaled inverse transform of each array of `data`, in place.
    #[track_caller]
    pub fn inverse(&self, data: &mut [Complex<T>]) {
        self.apply(data, true);
    }

    #[track_caller]
    fn apply(&self, data: &mut [Complex<T>], inverse: bool) {
        let (r, c) = self.shape();
        check_batch(data.len(), r * c, "Fft2");
        if r * c == 0 {
            return;
        }
        let mut work = Work::new(&[&self.rows, &self.cols]);
        rows(&self.cols, data, inverse, &mut work);
        for array in data.chunks_exact_mut(r * c) {
            columns(&self.rows, array, c, inverse, &mut work);
        }
    }
}

/// The three-dimensional discrete Fourier transform of `n0 × n1 × n2` complex arrays stored with
/// the last index varying fastest, planned once; as [`Fft2`] with one more dimension.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::fft::Fft3;
///
/// let fft = Fft3::<f32>::new(2, 2, 2);
/// let mut x = [Complex::ONE; 8];
/// fft.forward(&mut x);
/// assert_eq!(x[0], Complex::new(8.0, 0.0));
/// assert!(x[1..].iter().all(|&v| v == Complex::ZERO));
/// ```
pub struct Fft3<T> {
    n: [Fft<T>; 3],
}

impl<T: FftFloat> Fft3<T> {
    /// The plan for arrays of the shape `n0 × n1 × n2`.
    pub fn new(n0: usize, n1: usize, n2: usize) -> Fft3<T> {
        Fft3 {
            n: [Fft::new(n0), Fft::new(n1), Fft::new(n2)],
        }
    }

    /// The planned shape, `(n0, n1, n2)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n[0].len(), self.n[1].len(), self.n[2].len())
    }

    /// The forward transform of each array of `data`, in place.
    #[track_caller]
    pub fn forward(&self, data: &mut [Complex<T>]) {
        self.apply(data, false);
    }

    /// The unscaled inverse transform of each array of `data`, in place.
    #[track_caller]
    pub fn inverse(&self, data: &mut [Complex<T>]) {
        self.apply(data, true);
    }

    #[track_caller]
    fn apply(&self, data: &mut [Complex<T>], inverse: bool) {
        let (n0, n1, n2) = self.shape();
        let size = n0 * n1 * n2;
        check_batch(data.len(), size, "Fft3");
        if size == 0 {
            return;
        }
        let mut work = Work::new(&[&self.n[0], &self.n[1], &self.n[2]]);
        rows(&self.n[2], data, inverse, &mut work);
        for array in data.chunks_exact_mut(size) {
            for plane in array.chunks_exact_mut(n1 * n2) {
                columns(&self.n[1], plane, n2, inverse, &mut work);
            }
            columns(&self.n[0], array, n1 * n2, inverse, &mut work);
        }
    }
}

/// The most bytes of a tile of columns (gathered into a buffer, transformed there and
/// scattered back): about the second-level cache a core can count on.
const TILE_BYTES: usize = 1 << 20;

/// The most bytes of a row of a tile: two cache lines of 128 bytes, so that the gather and the
/// scatter read and write whole lines even where the stride of the rows makes them conflict in
/// the cache.
const TILE_ROW_BYTES: usize = 256;

/// The working memory of the passes: a tile and the scratch of the transforms.
struct Work<T> {
    tile: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: FftFloat> Work<T> {
    fn new(ffts: &[&Fft<T>]) -> Work<T> {
        let (mut tile, mut scratch) = (0, 0);
        for f in ffts {
            tile = tile.max(tile_cols::<T>(f.len()) * f.len());
            scratch = scratch.max(f.scratch_len());
        }
        Work {
            tile: vec![Complex::ZERO; tile],
            scratch: vec![Complex::ZERO; scratch],
        }
    }
}

/// The columns of a tile for transforms of the length `n`.
fn tile_cols<T>(n: usize) -> usize {
    let size = core::mem::size_of::<Complex<T>>();
    (TILE_BYTES / (size * n.max(1))).clamp(1, TILE_ROW_BYTES / size)
}

/// The transform of each contiguous run of the plan's length in `data`, as one batch.
fn rows<T: FftFloat>(fft: &Fft<T>, data: &mut [Complex<T>], inverse: bool, work: &mut Work<T>) {
    if fft.len() <= 1 {
        return;
    }
    if inverse {
        fft.inverse_with_scratch(data, &mut work.scratch);
    } else {
        fft.forward_with_scratch(data, &mut work.scratch);
    }
}

/// The transforms along a dimension of stride `stride`: for each offset `o` in `0..stride`, of
/// the values `data[o], data[o + stride], …`, by tiles of adjacent columns gathered into
/// contiguous sequences.
fn columns<T: FftFloat>(
    fft: &Fft<T>,
    data: &mut [Complex<T>],
    stride: usize,
    inverse: bool,
    work: &mut Work<T>,
) {
    let n = fft.len();
    if n <= 1 {
        return;
    }
    let width = tile_cols::<T>(n);
    let mut o = 0;
    while o < stride {
        let b = width.min(stride - o);
        let tile = &mut work.tile[..b * n];
        for j in 0..n {
            let row = &data[j * stride + o..j * stride + o + b];
            for (c, &v) in row.iter().enumerate() {
                tile[c * n + j] = v;
            }
        }
        if inverse {
            fft.inverse_with_scratch(tile, &mut work.scratch);
        } else {
            fft.forward_with_scratch(tile, &mut work.scratch);
        }
        for j in 0..n {
            let row = &mut data[j * stride + o..j * stride + o + b];
            for (c, v) in row.iter_mut().enumerate() {
                *v = tile[c * n + j];
            }
        }
        o += b;
    }
}

impl<T: FftFloat> core::fmt::Debug for Fft2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fft2")
            .field("rows", &self.rows.len())
            .field("cols", &self.cols.len())
            .finish_non_exhaustive()
    }
}

impl<T: FftFloat> core::fmt::Debug for Fft3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fft3")
            .field("n0", &self.n[0].len())
            .field("n1", &self.n[1].len())
            .field("n2", &self.n[2].len())
            .finish_non_exhaustive()
    }
}
