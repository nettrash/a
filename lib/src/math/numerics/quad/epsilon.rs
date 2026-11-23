//! Wynn's epsilon algorithm, which extrapolates the sequence of the approximations of an
//! adaptive rule to its limit, as QUADPACK's QAGS uses it (de Doncker 1978; Piessens et al.,
//! *QUADPACK*, 1983, the routine `qelg`).
//!
//! The ε-table of the sequence `S₀, S₁, …` is `ε₋₁⁽ᵏ⁾ = 0`, `ε₀⁽ᵏ⁾ = Sₖ` and
//! `εⱼ₊₁⁽ᵏ⁾ = εⱼ₋₁⁽ᵏ⁺¹⁾ + 1/(εⱼ⁽ᵏ⁺¹⁾ − εⱼ⁽ᵏ⁾)`; its even columns converge to the limit far faster
//! than the sequence when the error of `Sₖ` is a sum of geometric terms, which is the error of the
//! adaptive rule bisecting towards an end singularity of the integrand. Only the last ascending
//! diagonal of the table is kept, and it is extended by each new element. Each new element of
//! that diagonal is judged by the three elements of the column before it that produced it; the
//! one judged best is the result, and its error is estimated from the distance to the last three
//! results — a heuristic, but the one QUADPACK has been trusted with.

/// The most elements of the sequence kept: past it the oldest are dropped.
const LIMEXP: usize = 50;

/// The state of the extrapolation.
pub(super) struct Epsilon {
    /// The last diagonal, QUADPACK's `epstab`, 1-based: `tab[1..=n]`, with room for two more.
    tab: [f64; LIMEXP + 3],
    /// The number of its elements.
    n: usize,
    /// The last three results, for the error estimate.
    last: [f64; 3],
    /// The number of results produced.
    count: usize,
    /// Whether the last result came from three elements of a column that agree to the precision
    /// of the values, rather than from the spread of the last results.
    settled: bool,
    /// The unit roundoff of the type of the integrand's values.
    eps: f64,
}

impl Epsilon {
    /// An empty table, for values accurate to `eps`.
    pub(super) fn new(eps: f64) -> Self {
        Self {
            tab: [0.0; LIMEXP + 3],
            n: 0,
            last: [0.0; 3],
            count: 0,
            settled: false,
            eps,
        }
    }

    /// The number of elements in the table, which the extrapolation can shorten.
    pub(super) fn len(&self) -> usize {
        self.n
    }

    /// Whether the last result is one on which three elements of a column of the table agreed to
    /// the precision of the values: the limit reached, its error the elements' own spread, which
    /// the rounding of the sequence would have widened had it been magnified.
    pub(super) fn settled(&self) -> bool {
        self.settled
    }

    /// Adds an element of the sequence without extrapolating, as QAGS does with its first two.
    pub(super) fn seed(&mut self, s: f64) {
        self.n += 1;
        self.tab[self.n] = s;
    }

    /// Adds the next element of the sequence; returns the extrapolated limit and the estimate of
    /// its error, which is infinite until four results have been produced.
    pub(super) fn push(&mut self, s: f64) -> (f64, f64) {
        self.n += 1;
        self.tab[self.n] = s;
        let (result, error) = self.extrapolate();
        (result, error.max(5.0 * self.eps * result.abs()))
    }

    fn extrapolate(&mut self) -> (f64, f64) {
        let eps = self.eps;
        let tab = &mut self.tab;
        let mut n = self.n;
        self.count += 1;
        let mut result = tab[n];
        let mut error = f64::INFINITY;
        self.settled = false;
        if n < 3 {
            return (result, error);
        }
        tab[n + 2] = tab[n];
        let new_elements = (n - 1) / 2;
        // QUADPACK's `oflow`, the largest finite number and not an infinity: the tests below
        // compare its distance to an element with its own size times epsilon, which an
        // infinity would pass (∞ ≤ ∞) and cut the table short at every call.
        tab[n] = f64::MAX;
        let num = n;
        let mut k1 = n;
        for i in 1..=new_elements {
            let (k2, k3) = (k1 - 1, k1 - 2);
            let res = tab[k1 + 2];
            let (e0, e1, e2) = (tab[k3], tab[k2], res);
            let delta2 = e2 - e1;
            let err2 = delta2.abs();
            let tol2 = e2.abs().max(e1.abs()) * eps;
            let delta3 = e1 - e0;
            let err3 = delta3.abs();
            let tol3 = e1.abs().max(e0.abs()) * eps;
            if err2 <= tol2 && err3 <= tol3 {
                // e0, e1 and e2 agree to the precision of the values: the limit is reached.
                self.settled = true;
                return (res, err2 + err3);
            }
            let e3 = tab[k1];
            tab[k1] = e1;
            let delta1 = e1 - e3;
            let err1 = delta1.abs();
            let tol1 = e1.abs().max(e3.abs()) * eps;
            // Two elements too close to divide by their difference, or an irregular table: the
            // rest of the diagonal is dropped.
            if err1 <= tol1 || err2 <= tol2 || err3 <= tol3 {
                n = i + i - 1;
                break;
            }
            let ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
            if (ss * e1).abs() <= 1e-4 {
                n = i + i - 1;
                break;
            }
            let res = e1 + 1.0 / ss;
            tab[k1] = res;
            k1 -= 2;
            let err = err2 + (res - e2).abs() + err3;
            if err <= error {
                error = err;
                result = res;
            }
        }
        // Shift the table so that the next element has room, and drop the oldest when full.
        if n == LIMEXP {
            n = 2 * (LIMEXP / 2) - 1;
        }
        let mut ib = if num % 2 == 1 { 1 } else { 2 };
        for _ in 0..=new_elements {
            tab[ib] = tab[ib + 2];
            ib += 2;
        }
        if num != n {
            // The last n elements to the front.
            tab.copy_within(num - n + 1..=num, 1);
        }
        self.n = n;
        if self.count < 4 {
            self.last[self.count - 1] = result;
            return (result, f64::INFINITY);
        }
        let error = (result - self.last[2]).abs()
            + (result - self.last[1]).abs()
            + (result - self.last[0]).abs();
        self.last = [self.last[1], self.last[2], result];
        (result, error)
    }
}
