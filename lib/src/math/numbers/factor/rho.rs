//! Pollard's rho method (1975) with Brent's cycle finding (1980): the sequence `y -> y^2 + c`
//! modulo `n` is eventually periodic modulo each prime `p` of `n`, after about `sqrt(p)` steps,
//! and the greatest common divisor of `n` and the difference of two terms that agree modulo `p`
//! is a multiple of `p`. Brent compares `x = y_(2^i - 1)` with the next `2^i` terms, and the
//! differences are multiplied together, [`BATCH`] at a time, so that a greatest common divisor
//! is taken once a batch. When it is `n`, all the primes cycled in that batch: the batch is gone
//! over again from its start, a greatest common divisor a step, and when that still gives `n`,
//! the next `c` is taken.
//!
//! The terms are residues in Montgomery's form: `y R -> (y R)^2 / R + c` is the sequence of
//! `y -> y^2 + c / R`, as good a constant as `c`.

use super::field::{Arith, Job};

/// The differences multiplied together between two greatest common divisors.
const BATCH: u64 = 128;

/// What a run of the method gives.
pub(super) enum Rho {
    /// A factor other than 1 and `n`.
    Found(Vec<u64>),
    /// Nothing in the steps allowed.
    Nothing,
}

/// The method on `n` for at most `steps` steps of the sequence, with the constants `c = 1, 2, ...`
/// in turn when one cycles modulo every prime at once.
pub(super) struct RhoJob {
    /// The steps allowed.
    pub(super) steps: u64,
}

impl Job for RhoJob {
    type Out = Rho;

    fn run<A: Arith>(self, f: &A) -> Rho {
        let mut left = self.steps;
        let mut c = 1;
        while left > 0 {
            match brent(f, c, &mut left) {
                Some(g) => return Rho::Found(g),
                None => c += 1,
            }
        }
        Rho::Nothing
    }
}

/// One run of Brent's method with the constant `c`, starting from 2, taking steps from `left`:
/// a factor, or `None` when the steps run out or the sequence cycles modulo every prime at once.
fn brent<A: Arith>(f: &A, c: u64, left: &mut u64) -> Option<Vec<u64>> {
    let c = f.small(c);
    let mut y = f.small(2);
    let mut x = y.clone();
    let mut ys = y.clone();
    let mut q = f.one();
    let mut d = f.zero();
    let mut r: u64 = 1;
    loop {
        x.clone_from(&y);
        // `y` from `y_(r - 1)` to `y_(2r - 1)`... the first `r` steps without products.
        let skip = r.min(*left);
        for _ in 0..skip {
            f.sqr_in(&mut y);
            f.add_in(&mut y, &c);
        }
        *left -= skip;
        if *left == 0 {
            return None;
        }
        let mut k = 0;
        while k < r {
            ys.clone_from(&y);
            let m = BATCH.min(r - k).min(*left);
            for _ in 0..m {
                f.sqr_in(&mut y);
                f.add_in(&mut y, &c);
                f.sub(&mut d, &x, &y);
                f.mul_in(&mut q, &d);
            }
            *left -= m;
            k += m;
            match gcd(f, &q) {
                Gcd::One => {}
                Gcd::Factor(g) => return Some(g),
                Gcd::Whole => {
                    // Over again from the batch's start, a greatest common divisor a step.
                    for _ in 0..m {
                        f.sqr_in(&mut ys);
                        f.add_in(&mut ys, &c);
                        f.sub(&mut d, &x, &ys);
                        match gcd(f, &d) {
                            Gcd::One => {}
                            Gcd::Factor(g) => return Some(g),
                            Gcd::Whole => return None,
                        }
                    }
                    return None;
                }
            }
            if *left == 0 {
                return None;
            }
        }
        r = r.saturating_mul(2);
    }
}

/// The greatest common divisor of a residue with `n`.
pub(super) enum Gcd {
    /// 1.
    One,
    /// A factor other than 1 and `n`.
    Factor(Vec<u64>),
    /// `n`: the residue is 0.
    Whole,
}

/// The greatest common divisor of the residue `a` with `n`.
pub(super) fn gcd<A: Arith>(f: &A, a: &A::V) -> Gcd {
    if f.is_zero(a) {
        return Gcd::Whole;
    }
    let n = f.n();
    if n.len() == 1 {
        let g = super::field::gcd_u64(f.low(a), n[0]);
        return if g == 1 {
            Gcd::One
        } else if g == n[0] {
            Gcd::Whole
        } else {
            Gcd::Factor(vec![g])
        };
    }
    let g = super::field::gcd_n(f, a);
    if g == [1] {
        Gcd::One
    } else if g == n {
        Gcd::Whole
    } else {
        Gcd::Factor(g)
    }
}
