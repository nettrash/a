//! Transforms of arrays of two and three dimensions, one dimension after the other: the last
//! (contiguous) one as a batch of rows; each other one, for an array that fits in the caches,
//! by the passes of its transform run along the rows (the columns as interleaved sequences, the
//! vectors across them), else by tiles of columns gathered into a buffer, transformed there as a
//! batch of contiguous sequences and scattered back, so that each pass over the array reads and
//! writes whole cache lines.

use super::complex::{Algo, Isa};
use super::{Fft, FftFloat, check_batch, with_raw};
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
    down: Axis<T>,
}

impl<T: FftFloat> Fft2<T> {
    /// The plan for arrays of `rows` rows of `cols` values.
    pub fn new(rows: usize, cols: usize) -> Fft2<T> {
        let rows_plan = Fft::new(rows);
        Fft2 {
            down: Axis::new(&rows_plan, cols, rows * cols),
            rows: rows_plan,
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
        let need = self.cols.batch_work(data.len()).max(self.down.need());
        with_raw(need, |scr| {
            // SAFETY: `scr` holds `need` values, every one written before it is read; the
            // arrays are whole.
            unsafe {
                self.cols
                    .run_raw(data.as_mut_ptr(), data.len(), scr, need, inverse);
                for array in data.chunks_exact_mut(r * c) {
                    self.down.run(array, scr, inverse);
                }
            }
        });
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
    /// The second axis within each plane of `n1 × n2`, and the first over the whole array.
    mid: Axis<T>,
    top: Axis<T>,
}

impl<T: FftFloat> Fft3<T> {
    /// The plan for arrays of the shape `n0 × n1 × n2`.
    pub fn new(n0: usize, n1: usize, n2: usize) -> Fft3<T> {
        let n = [Fft::new(n0), Fft::new(n1), Fft::new(n2)];
        Fft3 {
            mid: Axis::new(&n[1], n2, n1 * n2),
            top: Axis::new(&n[0], n1 * n2, n0 * n1 * n2),
            n,
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
        let need = self.n[2]
            .batch_work(data.len())
            .max(self.mid.need())
            .max(self.top.need());
        with_raw(need, |scr| {
            // SAFETY: as for `Fft2`.
            unsafe {
                self.n[2].run_raw(data.as_mut_ptr(), data.len(), scr, need, inverse);
                for array in data.chunks_exact_mut(size) {
                    for plane in array.chunks_exact_mut(n1 * n2) {
                        self.mid.run(plane, scr, inverse);
                    }
                    self.top.run(array, scr, inverse);
                }
            }
        });
    }
}

/// The most bytes of a tile of columns (gathered into a buffer, transformed there and
/// scattered back): about the second-level cache a core can count on.
const TILE_BYTES: usize = 1 << 20;

/// The most bytes of a row of a tile: two cache lines of 128 bytes, so that the gather and the
/// scatter read and write whole lines even where the stride of the rows makes them conflict in
/// the cache.
const TILE_ROW_BYTES: usize = 256;

/// The most bytes of an array whose columns are transformed by the passes along its rows
/// (each pass reads and writes the whole array); a larger one goes by tiles. Measured on an
/// Apple M4 Max (arrays of 1 to 256 MiB, `f64` and `f32`, 2 to 8192 rows, two and three
/// dimensions): above this bound the passes take 0.33 to 0.9 of the tiles' time, except where a
/// column is one composite butterfly of 12 to 25 values and the stride of the rows a power of
/// two (16 × 65,536, 24 × 512 × 128: the values of a column fall into the same sets of the
/// caches), 1.08 to 1.56 times the tiles' time, and already 1.15 to 1.58 times at 2 to 4 MiB; a
/// larger bound would make those slower.
const ALONG_BYTES: usize = 4 << 20;

/// The transforms along one axis of stride `stride` (the columns of an array of `stride`
/// columns): by the passes along the rows, or by tiles.
enum Axis<T> {
    /// A length of 0 or 1: nothing to do.
    None,
    /// The passes of the columns as interleaved sequences, over arrays of `len` values.
    Along { algo: Algo<T>, need: usize },
    /// Tiles of columns gathered into a buffer, transformed by `fft` and scattered back.
    Tiles {
        fft: Fft<T>,
        stride: usize,
        width: usize,
    },
}

impl<T: FftFloat> Axis<T> {
    /// The transforms by `fft` of the columns of arrays of `len` values in rows of `stride`.
    fn new(fft: &Fft<T>, stride: usize, len: usize) -> Axis<T> {
        let n = fft.len();
        if n <= 1 || len == 0 {
            return Axis::None;
        }
        if len * core::mem::size_of::<Complex<T>>() <= ALONG_BYTES
            && let Some(algo) = Algo::interleaved(n, stride, Isa::detect())
        {
            let need = algo.scratch();
            return Axis::Along { algo, need };
        }
        Axis::Tiles {
            fft: fft.clone(),
            stride,
            width: tile_cols::<T>(n),
        }
    }

    /// The working memory of [`run`](Self::run).
    fn need(&self) -> usize {
        match self {
            Axis::None => 0,
            Axis::Along { need, .. } => *need,
            Axis::Tiles { fft, width, .. } => {
                let tile = width * fft.len();
                tile + fft.batch_work(tile)
            }
        }
    }

    /// The transforms of the columns of `data` (one array), with the working memory at `scr`
    /// of [`need`](Self::need) values, uninitialised.
    ///
    /// # Safety
    /// `scr` must be valid for [`need`](Self::need) values and not overlap `data`.
    unsafe fn run(&self, data: &mut [Complex<T>], scr: *mut Complex<T>, inverse: bool) {
        match self {
            Axis::None => {}
            Axis::Along { algo, .. } => {
                let p = data.as_mut_ptr();
                // SAFETY: the array, in place; the plan's working memory at `scr`.
                unsafe { algo.run(p, p, scr, inverse) }
            }
            Axis::Tiles { fft, stride, width } => {
                let (n, stride) = (fft.len(), *stride);
                let tile_len = width * n;
                // SAFETY: the caller's contract: the tile and then the transforms' scratch.
                let (tile, rest) = unsafe { (scr, scr.add(tile_len)) };
                let rest_len = fft.batch_work(tile_len);
                let mut o = 0;
                while o < stride {
                    let b = (*width).min(stride - o);
                    // SAFETY: the gather writes the `b·n` values of the tile, which the
                    // transform then reads and the scatter reads back; within `data`.
                    unsafe {
                        for j in 0..n {
                            let row = data.as_ptr().add(j * stride + o);
                            for c in 0..b {
                                tile.add(c * n + j).write(*row.add(c));
                            }
                        }
                        fft.run_raw(tile, b * n, rest, rest_len, inverse);
                        for j in 0..n {
                            let row = data.as_mut_ptr().add(j * stride + o);
                            for c in 0..b {
                                *row.add(c) = tile.add(c * n + j).read();
                            }
                        }
                    }
                    o += b;
                }
            }
        }
    }
}

/// The columns of a tile for transforms of the length `n`.
fn tile_cols<T>(n: usize) -> usize {
    let size = core::mem::size_of::<Complex<T>>();
    (TILE_BYTES / (size * n.max(1))).clamp(1, TILE_ROW_BYTES / size)
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
