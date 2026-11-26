//! The L² algorithm of Nguyen and Stehlé (*Floating-point LLL revisited*, Eurocrypt 2005; *An
//! LLL algorithm with quadratic complexity*, SIAM J. Comput. 39, 2009): LLL with the
//! Gram–Schmidt orthogonalisation in floating point, computed from the exact Gram matrix, and a
//! lazy size reduction.
//!
//! The precision is chosen adaptively, as fplll's wrapper chooses it: a first run in words and
//! plain doubles where the rows, `U` and the Gram matrix fit in words ([`super::l2_small`]), or
//! in plain doubles below `2^1000` (and below `2^2000` scaled by one power of two) and in 53-bit
//! floats with an exponent of their own ([`Dpe`], so that entries of any size fit), then, if that run fails or its result is not proved
//! reduced, runs in [`BigFloat`](crate::math::float::BigFloat)
//! from the precision of Nguyen and Stehlé's theorem, `d·log₂ρ + O(log d)` bits for `d` vectors
//! and `ρ = (1 + η)²/(δ − η²)`, doubled twice (as far as [`BigFloat::MAX_PREC`]), each run
//! starting from the basis the last one left.
//! Each result is checked by [`certify`]: the conditions in rigorous interval arithmetic and, for a
//! condition the intervals leave open, exactly. Should every run fail, the exact integral
//! reduction finishes the work. So the result is `(δ, η)`-LLL-reduced whatever happens, and
//! proved so; the floating point decides only how fast.

use super::exact::{
    Insertion, Integral, check_delta, check_eta, fraction, identity_rows, matrix_of,
    reduced_exactly, rows_of,
};
use super::l2_certify::{Verdict, certify, certify_dd, certify_f64};
use super::l2_core::{Basis, numbers, run, run_scaled};
use super::l2_float::{Big, Dpe, F64};
use super::l2_small::{self, Small};
use super::zint::Z;
use super::{LllOptions, Reduced};
use crate::math::float::BigFloat;
use crate::math::matrix::Matrix;
use crate::math::numbers::Number;
use crate::math::rational::Rational;

/// The `δ̄` and `η̄` of the floating-point tests, exactly: `δ̄ = δ + min((1 − δ)/4, 1/64)` and
/// `η̄ = η − min((η − 1/2)/2, 1/64)`, stricter than `δ` and `η` by a quarter and a half of the room
/// there is (at most `1/64`), so that the rounding of the tests cannot leave the result short of
/// `(δ, η)`. Each run rounds them towards `δ` and `η` (see [`run`]): they keep that margin in every
/// `BigFloat` run, whose precision holds them exactly; a 53-bit float holds no float strictly
/// between `δ` and `δ̄` when `1 − δ < 2⁻⁵¹`, nor between `η̄` and `η` for `η = 1/2 + 2⁻⁵³`, and that
/// run then tests with `δ` or `η` itself, its result resting on the proof alone, as always.
fn margins(delta: f64, eta: f64) -> (Rational, Rational) {
    let room = Rational::new(Number::from(1u8), Number::from(64u8));
    let (d, e) = (
        Rational::try_from(delta).expect("δ is finite"),
        Rational::try_from(eta).expect("η is finite"),
    );
    let dm = (&Rational::from(1) - &d) / 4;
    let em = (&e - &Rational::new(Number::from(1u8), Number::from(2u8))) / 2;
    (
        &d + if dm < room { &dm } else { &room },
        &e - if em < room { &em } else { &room },
    )
}

/// The precision of Nguyen and Stehlé's theorem for `d` vectors: `d·log₂ρ` bits for
/// `ρ = (1 + η)²/(δ − η²)`, and `2·log₂ d + 32` more for the lower-order terms, at least 106;
/// `None` beyond [`BigFloat::MAX_PREC`]. `δ − η²` is taken exactly: positive for valid
/// parameters (`η² < δ` is checked in rationals), and at least `2⁻¹⁰⁸` as the parameters are
/// floats, where the `f64` difference may be 0.
fn proof_precision(d: usize, delta: f64, eta: f64) -> Option<u64> {
    let e = Rational::try_from(eta).expect("η is finite");
    let gap = (&Rational::try_from(delta).expect("δ is finite") - &(&e * &e)).to_f64();
    let log2_rho = 2.0 * (1.0 + eta).log2() - gap.log2();
    let d = d.max(1) as f64;
    let p = (d * log2_rho + 2.0 * (d + 1.0).log2() + 32.0)
        .ceil()
        .max(106.0);
    (p.is_finite() && p <= BigFloat::MAX_PREC as f64).then_some(p as u64)
}

/// Whether the rows of the lattice in `st` are `(δ, η)`-reduced, with the exchange condition at
/// the places `ins` allows, and the rows after them zero: proved by intervals of `64 + 2d` bits,
/// then twice and four times that, and by the exact test if they leave a condition open.
fn verified(st: &mut Basis, delta: f64, eta: f64, ins: Insertion) -> bool {
    if !st.b[st.active..].iter().all(|r| r.iter().all(Z::is_zero)) {
        return false;
    }
    let g = st.gram_numbers();
    match certify_f64(&g, delta, eta, ins) {
        Verdict::Reduced => return true,
        Verdict::NotReduced => return false,
        Verdict::Unsure => {}
    }
    match certify_dd(&g, delta, eta, ins) {
        Verdict::Reduced => return true,
        Verdict::NotReduced => return false,
        Verdict::Unsure => {}
    }
    let mut p = 64 + 2 * g.len() as u64;
    for _ in 0..3 {
        if p > BigFloat::MAX_PREC {
            break;
        }
        match certify(&g, delta, eta, ins, p) {
            Verdict::Reduced => return true,
            Verdict::NotReduced => return false,
            Verdict::Unsure => p *= 2,
        }
    }
    let (a, den) = fraction(delta);
    let (ea, eb) = fraction(eta);
    let rows = numbers(&st.b[..st.active]);
    reduced_exactly(&rows, (&a, &den), (&ea, &eb), ins)
}

/// Reduces the rows of `basis` by L², inserting as `ins` says, to a basis proved
/// `(δ, η)`-reduced (with the deep exchange condition where `ins` asks for it).
pub(super) fn reduce_fp(basis: &Matrix<Number>, delta: f64, eta: f64, ins: Insertion) -> Reduced {
    let (n, m) = basis.shape();
    let mut st = Basis::new(rows_of(basis), identity_rows(n));
    let (dbar, ebar) = margins(delta, eta);
    // Plain doubles first while the Gram matrix's entries fit them with room for the
    // orthogonalisation (its products are at most the largest entry, by Cauchy and Schwarz):
    // the same steps as with an exponent of their own, faster; a run that leaves the range of
    // the doubles gives way to the doubles with an exponent, from the basis it reached. Measured
    // on an Apple M4 Max with knapsacks and uniform bases whose Gram matrices have 960 to 1020
    // bits: a bound of 900 to 1020 bits changes nothing beyond 0.5 %, the scaled doubles below
    // costing as much.
    // Words first, where the rows, U and the Gram matrix fit them: the same steps on `i64`s and
    // plain `f64`s; a run that would overflow goes on in integers of any size from where it is
    // (starting its steps again there, so that the reduced basis may differ from that of a run
    // in integers from the start; either is proved).
    let mut done = false;
    if let Some(mut small) = Small::of(&st) {
        let ok = l2_small::run(&mut small, &dbar, &ebar, ins).is_ok();
        st = small.into_basis();
        done = ok && verified(&mut st, delta, eta, ins);
    }
    done = done
        || (st.gram_bits() < 1000
            && run::<F64>(&mut st, 53, &dbar, &ebar, ins).is_ok()
            && verified(&mut st, delta, eta, ins));
    // Longer Gram matrices in plain doubles too, their entries taken times `2⁻ˢ` so that the
    // largest is about `2^1000`: every value of the run is then that of the floats with an
    // exponent of their own times the same power of two, while it stays in the range of the
    // doubles, which the run leaves (a NaN) where it would not. Up to Gram matrices of 3000 bits:
    // measured on an Apple M4 Max, knapsacks of 30 and 50 numbers and uniform bases of 20 and 40
    // rows, of 1000 to 1500 bits, against 2000 bits: 0.85 to 0.98 of the time from 2000 to 2400
    // bits, no more than 1.007 elsewhere (a run whose small entries leave the range ends at
    // once); a top of 950 bits took 1.06 to 1.10 of the time just below 2000 bits, one of 1010 or
    // 1020 the same as 1000.
    let bits = st.gram_bits();
    done = done
        || ((1000..3000).contains(&bits)
            && run_scaled::<F64>(&mut st, 53, &dbar, &ebar, ins, bits as i64 - 1000).is_ok()
            && verified(&mut st, delta, eta, ins));
    if !done {
        done = run::<Dpe>(&mut st, 53, &dbar, &ebar, ins).is_ok()
            && verified(&mut st, delta, eta, ins);
    }
    // At p, 2p and 4p bits, while they are precisions a BigFloat has; beyond, the exact
    // reduction is left to finish.
    let mut p = proof_precision(st.active, delta, eta);
    for _ in 0..3 {
        let Some(prec) = p.filter(|_| !done) else {
            break;
        };
        done = run::<Big>(&mut st, prec, &dbar, &ebar, ins).is_ok()
            && verified(&mut st, delta, eta, ins);
        p = prec.checked_mul(2).filter(|&q| q <= BigFloat::MAX_PREC);
    }
    let (b, u, rank) = if done {
        // The zero rows, found at the end, go first.
        let zeros = n - st.active;
        let (mut b, mut u) = (numbers(&st.b), numbers(&st.u));
        b.rotate_right(zeros);
        u.rotate_right(zeros);
        (b, u, st.active)
    } else {
        let (a, den) = fraction(delta);
        let mut e = Integral::new(numbers(&st.b), numbers(&st.u));
        e.run(&a, &den, ins);
        let rank = e.rank();
        let (b, u) = e.into_rows();
        (b, u, rank)
    };
    Reduced {
        basis: matrix_of(b, m),
        transform: matrix_of(u, n),
        rank,
    }
}

/// LLL-reduces the rows of `basis` by the L² algorithm of Nguyen and Stehlé: the result is
/// `(δ, η)`-LLL-reduced (as [`is_lll_reduced`](super::is_lll_reduced) checks it), the
/// Gram–Schmidt coefficients computed in floating point, and the result proved reduced afterwards.
///
/// The basis and its Gram matrix are kept exactly (in machine words while they fit); the
/// Gram–Schmidt coefficients are computed in floating point from the exact Gram matrix, first in
/// plain doubles or 53-bit floats with an exponent of their own and, where that is not enough, in
/// [`BigFloat`](crate::math::float::BigFloat)s of
/// the precision of Nguyen and Stehlé's theorem (`d·log₂ρ + 2·log₂(d + 1) + 32` bits for `d`
/// vectors and `ρ = (1 + η)²/(δ − η²)`, about `1.6·d + O(log d)` for the default parameters),
/// then of twice and four times that (as far as `2^40`, the greatest precision of a `BigFloat`).
/// The floating-point tests use `δ̄ = δ + min((1 − δ)/4, 1/64)` and
/// `η̄ = η − min((η − 1/2)/2, 1/64)`, stricter than `δ` and `η`, rounded towards them at the
/// precision of each run. The result of each run is checked in rigorous interval
/// arithmetic (and exactly where the intervals do not decide), and the exact
/// [`lll`](super::lll) finishes the work if no run passes, so the guarantee does not rest on
/// the floating point: the result is always `(δ, η)`-LLL-reduced, `U` unimodular and
/// `U·B = B'`; which of the lattice's reduced bases it is, is not promised beyond that (see [`Reduced`](super::Reduced)).
///
/// The rows may be linearly dependent: the result has a zero row for each dependency, first.
///
/// Nguyen and Stehlé bound the work of L² at their precision by `O(d⁴·m·(d + log B)·log B)` bit
/// operations for `d` rows of `m` columns and norm at most `B`: quadratic in `log B`, where the
/// exact reduction is cubic. The check of the result adds the exact Gram matrix, `O(d²·m)`
/// products, and `O(d³)` operations on intervals of `O(d)` bits.
///
/// # Panics
///
/// Panics if `δ` is not in `(1/4, 1)` or `η` not in `(1/2, √δ)`.
///
/// # Examples
///
/// ```
/// use a::math::lattice::{LllOptions, is_lll_reduced, lll_fp};
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// // A knapsack basis with numbers of 200 bits: [aᵢ | eᵢ].
/// let a: Vec<Number> = (1..=8u32)
///     .map(|i| Number::from(3u8).pow(126) / Number::from(i) + Number::from(i * i))
///     .collect();
/// let b = Matrix::from_fn(8, 9, |i, j| {
///     if j == 0 { a[i].clone() } else { Number::from(u8::from(i + 1 == j)) }
/// });
/// let r = lll_fp(&b, &LllOptions::new());
/// assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
/// assert_eq!(&r.transform().clone() * &b, *r.basis());
/// assert_eq!(r.rank(), 8);
/// ```
#[track_caller]
pub fn lll_fp(basis: &Matrix<Number>, opts: &LllOptions) -> Reduced {
    check_delta(opts.delta, false);
    check_eta(opts.eta, opts.delta, false);
    reduce_fp(basis, opts.delta, opts.eta, Insertion::Lll)
}
