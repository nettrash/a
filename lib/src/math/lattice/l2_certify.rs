//! A proof that a basis is reduced, by interval arithmetic: the Cholesky factorisation of its
//! exact Gram matrix in [`BigFloat`] intervals, every lower end rounded down and every upper end
//! up, so that each interval holds the exact Gram–Schmidt quantity it stands for. A condition
//! whose interval lies on the right side of its bound holds; one whose interval lies on the wrong
//! side fails; one whose interval holds the bound is not decided at that precision.
//!
//! The widths grow with the conditioning of the basis, which for a reduced one is bounded
//! (Nguyen and Stehlé), so a precision of a few bits for each dimension decides almost every
//! condition; the ones left, such as a coefficient of exactly `η`, are for the exact test.

use super::exact::Insertion;
use crate::math::float::{BigFloat, Round};
use crate::math::numbers::Number;

/// What the intervals tell of a basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// Every condition holds.
    Reduced,
    /// A condition fails.
    NotReduced,
    /// Some condition is not decided at this precision.
    Unsure,
}

/// A closed interval `[lo, hi]` of floats.
#[derive(Clone, Debug)]
struct Iv {
    lo: BigFloat,
    hi: BigFloat,
}

fn down(x: (BigFloat, core::cmp::Ordering)) -> BigFloat {
    x.0
}

impl Iv {
    fn of_number(n: &Number, p: u64) -> Iv {
        Iv {
            lo: BigFloat::from_number(n, p, Round::Down).0,
            hi: BigFloat::from_number(n, p, Round::Up).0,
        }
    }

    fn of_f64(x: f64) -> Iv {
        // 53 bits hold an f64 exactly.
        let v = BigFloat::from_f64_round(x, 53, Round::Nearest).0;
        Iv {
            lo: v.clone(),
            hi: v,
        }
    }

    fn add(&self, o: &Iv, p: u64) -> Iv {
        Iv {
            lo: down(self.lo.add_round(&o.lo, p, Round::Down)),
            hi: down(self.hi.add_round(&o.hi, p, Round::Up)),
        }
    }

    fn sub(&self, o: &Iv, p: u64) -> Iv {
        Iv {
            lo: down(self.lo.sub_round(&o.hi, p, Round::Down)),
            hi: down(self.hi.sub_round(&o.lo, p, Round::Up)),
        }
    }

    fn mul(&self, o: &Iv, p: u64) -> Iv {
        let lo = |x: &BigFloat, y: &BigFloat| down(x.mul_round(y, p, Round::Down));
        let hi = |x: &BigFloat, y: &BigFloat| down(x.mul_round(y, p, Round::Up));
        let (a1, a2, b1, b2) = (&self.lo, &self.hi, &o.lo, &o.hi);
        let (ap, an) = (*a1 >= 0.0, *a2 <= 0.0);
        let (bp, bn) = (*b1 >= 0.0, *b2 <= 0.0);
        match (ap, an, bp, bn) {
            (true, _, true, _) => Iv {
                lo: lo(a1, b1),
                hi: hi(a2, b2),
            },
            (true, _, _, true) => Iv {
                lo: lo(a2, b1),
                hi: hi(a1, b2),
            },
            (_, true, true, _) => Iv {
                lo: lo(a1, b2),
                hi: hi(a2, b1),
            },
            (_, true, _, true) => Iv {
                lo: lo(a2, b2),
                hi: hi(a1, b1),
            },
            (true, _, _, _) => Iv {
                lo: lo(a2, b1),
                hi: hi(a2, b2),
            },
            (_, true, _, _) => Iv {
                lo: lo(a1, b2),
                hi: hi(a1, b1),
            },
            (_, _, true, _) => Iv {
                lo: lo(a1, b2),
                hi: hi(a2, b2),
            },
            (_, _, _, true) => Iv {
                lo: lo(a2, b1),
                hi: hi(a1, b1),
            },
            _ => {
                let (l1, l2) = (lo(a1, b2), lo(a2, b1));
                let (h1, h2) = (hi(a1, b1), hi(a2, b2));
                Iv {
                    lo: if l1 < l2 { l1 } else { l2 },
                    hi: if h1 > h2 { h1 } else { h2 },
                }
            }
        }
    }

    /// `self / o` for `o.lo > 0`.
    fn div_pos(&self, o: &Iv, p: u64) -> Iv {
        let lo = |x: &BigFloat, y: &BigFloat| down(x.div_round(y, p, Round::Down));
        let hi = |x: &BigFloat, y: &BigFloat| down(x.div_round(y, p, Round::Up));
        Iv {
            lo: if self.lo >= 0.0 {
                lo(&self.lo, &o.hi)
            } else {
                lo(&self.lo, &o.lo)
            },
            hi: if self.hi >= 0.0 {
                hi(&self.hi, &o.lo)
            } else {
                hi(&self.hi, &o.hi)
            },
        }
    }

    /// `self²`.
    fn sqr(&self, p: u64) -> Iv {
        let (a, b) = (self.lo.abs(), self.hi.abs());
        let (small, large) = if a < b { (a, b) } else { (b, a) };
        let straddles = self.lo < 0.0 && self.hi > 0.0;
        Iv {
            lo: if straddles {
                BigFloat::zero(p)
            } else {
                down(small.mul_round(&small, p, Round::Down))
            },
            hi: down(large.mul_round(&large, p, Round::Up)),
        }
    }

    /// Whether the interval is `≥ 0` throughout (`Some(true)`), `< 0` throughout
    /// (`Some(false)`), or neither.
    fn sign(&self) -> Option<bool> {
        if self.lo >= 0.0 {
            Some(true)
        } else if self.hi < 0.0 {
            Some(false)
        } else {
            None
        }
    }
}

/// Folds a condition into the verdict so far: `Some(false)` ends it.
fn note(unsure: &mut bool, holds: Option<bool>) -> Result<(), Verdict> {
    match holds {
        Some(true) => Ok(()),
        Some(false) => Err(Verdict::NotReduced),
        None => {
            *unsure = true;
            Ok(())
        }
    }
}

/// Whether the rows whose exact Gram matrix has the lower triangle `g` are linearly
/// independent, with `|μᵢⱼ| ≤ η` and the exchange condition with `δ` at every place `ins`
/// allows, as intervals of precision `p` show it.
pub(super) fn certify(g: &[Vec<Number>], delta: f64, eta: f64, ins: Insertion, p: u64) -> Verdict {
    match certify_inner(g, delta, eta, ins, p) {
        Ok(v) | Err(v) => v,
    }
}

fn certify_inner(
    g: &[Vec<Number>],
    delta: f64,
    eta: f64,
    ins: Insertion,
    p: u64,
) -> Result<Verdict, Verdict> {
    let n = g.len();
    let (delta, eta) = (Iv::of_f64(delta), Iv::of_f64(eta));
    let mut r: Vec<Vec<Iv>> = Vec::with_capacity(n);
    let mut mu: Vec<Vec<Iv>> = Vec::with_capacity(n);
    let mut unsure = false;
    for i in 0..n {
        let mut ri: Vec<Iv> = Vec::with_capacity(i + 1);
        let mut mi: Vec<Iv> = Vec::with_capacity(i);
        for j in 0..=i {
            let mut t = Iv::of_number(&g[i][j], p);
            for k in 0..j {
                let mjk = if j < i { &mu[j][k] } else { &mi[k] };
                t = t.sub(&mjk.mul(&ri[k], p), p);
            }
            if j < i {
                let m = t.div_pos(&r[j][j], p);
                // |μ| ≤ η: η − μ ≥ 0 and η + μ ≥ 0.
                note(&mut unsure, eta.sub(&m, p).sign())?;
                note(&mut unsure, eta.add(&m, p).sign())?;
                ri.push(t);
                mi.push(m);
            } else {
                // The exact r_ii is never negative: an upper end at or below zero is a zero
                // vector, a dependency; a lower end there leaves nothing to divide by.
                if t.hi <= 0.0 {
                    return Err(Verdict::NotReduced);
                }
                if t.lo <= 0.0 {
                    return Err(Verdict::Unsure);
                }
                ri.push(t);
            }
        }
        // |πⱼ(bᵢ)|² = r_ii + Σ_{j ≤ l < i} μ²_il·r_ll ≥ δ·r_jj at each place allowed.
        let mut s = ri[i].clone();
        for j in (0..i).rev() {
            s = s.add(&mi[j].sqr(p).mul(&r[j][j], p), p);
            if ins.allowed(j, i, 0) {
                note(&mut unsure, s.sub(&delta.mul(&r[j][j], p), p).sign())?;
            }
            if ins == Insertion::Lll {
                break;
            }
        }
        r.push(ri);
        mu.push(mi);
    }
    Ok(if unsure {
        Verdict::Unsure
    } else {
        Verdict::Reduced
    })
}
