//! Helpers shared by the lattice tests: bases, and the checks every reduction must pass.

use a::math::lattice::Reduced;
use a::math::matrix::Matrix;
use a::math::numbers::Number;

/// A matrix of numbers from rows of `i64`.
pub fn basis<const C: usize>(rows: &[[i64; C]]) -> Matrix<Number> {
    Matrix::from_rows(rows).map(|&x| Number::from(x))
}

/// A deterministic generator of pseudo-random words (SplitMix64).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number of `bits` random bits, of a random sign.
    pub fn number(&mut self, bits: u64) -> Number {
        let mut x = Number::default();
        let mut left = bits;
        while left > 0 {
            let take = left.min(64);
            let w = if take == 64 {
                self.next()
            } else {
                self.next() >> (64 - take)
            };
            x = (x << take as usize) + Number::from(w);
            left -= take;
        }
        if self.next() & 1 == 1 { -x } else { x }
    }
}

/// A random `n × m` basis of `bits`-bit integers.
pub fn random_basis(rng: &mut Rng, n: usize, m: usize, bits: u64) -> Matrix<Number> {
    Matrix::from_fn(n, m, |_, _| rng.number(bits))
}

/// A knapsack basis of `n` rows: `[I | aᵢ]` with `aᵢ` of `bits` bits, as fplll's `latticegen r`.
pub fn knapsack(rng: &mut Rng, n: usize, bits: u64) -> Matrix<Number> {
    Matrix::from_fn(n, n + 1, |i, j| {
        if j == 0 {
            rng.number(bits).abs()
        } else {
            Number::from(u8::from(i + 1 == j))
        }
    })
}

/// The checks of every reduction of `b`: `U·B = B'`, `U` unimodular, the zero rows first and
/// as many as the rank says.
pub fn check_reduction(b: &Matrix<Number>, r: &Reduced) {
    let (bb, u) = (r.basis(), r.transform());
    assert_eq!(u.shape(), (b.rows(), b.rows()), "U is square");
    assert_eq!(&u.clone() * b, *bb, "U·B = B'");
    assert_eq!(u.det().abs(), Number::from(1u8), "U is unimodular");
    let zeros = (0..bb.rows())
        .take_while(|&i| (0..bb.cols()).all(|j| bb[(i, j)].is_zero()))
        .count();
    assert_eq!(zeros + r.rank(), bb.rows(), "the zero rows come first");
    assert_eq!(r.rank(), b.rank(), "the rank of the lattice");
}
