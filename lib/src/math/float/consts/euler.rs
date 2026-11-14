//! Euler's constant by the Brent–McMillan algorithm B3 (1980) with binary splitting, in the dual
//! numbers of FLINT's `mp_real` (F. Johansson, 2026):
//!
//! `γ = S0/I0 - K0/I0² - log m + E`, `|E| < 24 e^(-8m)` for `m ≥ 2` (Brent and Johansson, "A
//! bound for the error term in the Brent–McMillan algorithm", Math. Comp. 84 (2015), Theorem 1 and
//! Remark 2; MPFR's bound), with
//!
//! - `I0 = Σ_(k ≥ 0) u_k`, `S0 = Σ_(k ≥ 0) u_k H_k`, `u_k = (m^k / k!)²`, `H_k = Σ_(j ≤ k) 1/j`;
//! - `K0 = (1/(4m)) Σ_(k < 2m) ((2k)!)³ / ((k!)⁴ (16m)^(2k))`.
//!
//! With `F(ε) = Σ_(k ≥ 0) m^(2k) / Π_(j ≤ k) (j + ε)²` over the dual numbers (`ε² = 0`),
//! `I0 = F(0)` and `S0 = -F'(0)/2`, so one splitting of `F` over `Z[ε]/(ε²)` gives both sums: a
//! range holds `P = m^(2(b-a))` (a scalar, kept without its power of two), `D = Π (k + 1 + ε) =
//! D0 + D1 ε` and `T = T0 + T1 ε`, with `Q = D²` formed where a merge needs it. All of `P`, `D0`,
//! `D1`, `T0`, `T1` are positive (the ε-part of `T` too: it gathers the derivatives of the
//! products `Q`), so the numbers may be cut ([`Cut`]). At the end, with `X = T0 + D0²`,
//! `I0 = X/D0²` and `S0/I0 = (2 T0 D1 - T1 D0)/(2 D0 X)`.
//!
//! `m` is the cheapest number at least `(w + log2 24)·ln 2/8` whose prime factors are among 2, 3,
//! 5, 7, so that `log m` is an integer combination of a few logarithms of ratios close to 1, each
//! by Zuniga's series ([`log_ratio`]), or of `ln 2` alone when `m` is a power of two.

use super::super::limbs;
use super::num::{Cut, Num, quotient};
use super::series::{Poly, Series};

/// The terms of a leaf.
const LEAF: u64 = 16;

/// A node of the splitting of `F`.
struct Node {
    /// `P` without its power of two.
    p: Num,
    d0: Num,
    d1: Num,
    t0: Num,
    t1: Num,
}

/// `x·v` for a magnitude `v` of up to two limbs.
fn mul_small(x: &[u64], v: u128) -> Vec<u64> {
    limbs::mul(
        x,
        &[v as u64, (v >> 64) as u64][..if v >> 64 == 0 { 1 } else { 2 }],
    )
}

/// The splitting of `F` for the parameter `m = 2^z o`, `o` odd.
struct Split {
    /// `o²`.
    o2: u128,
    /// `2z`, the bits of the power of two of `m²`.
    z2: u64,
}

impl Split {
    /// The node of the terms `a..b`, with `P` when `need_p`.
    fn split(&self, a: u64, b: u64, need_p: bool, cut: &mut Cut) -> Node {
        if b - a <= LEAF {
            return self.leaf(a, b, cut);
        }
        let leaves = (b - a).div_ceil(LEAF);
        let mid = a + leaves / 2 * LEAF;
        let l = self.split(a, mid, true, cut);
        let r = self.split(mid, b, need_p, cut);
        // `Q2 = D2² = q0 + q1 ε`.
        let q0 = cut.sqr(&r.d0);
        let mut q1 = cut.mul(&r.d0, &r.d1);
        Cut::shl(&mut q1, 1);
        // `P1` with its power of two.
        let sh = self.z2 * (mid - a);
        // `T1 = P1 T2_1 + Q2_0 T1_1 + Q2_1 T1_0`, `T0 = P1 T2_0 + Q2_0 T1_0`.
        let mut u = cut.mul(&l.p, &r.t1);
        Cut::shl(&mut u, sh);
        let v = cut.mul(&q0, &l.t1);
        let u = cut.add(u, v);
        let v = cut.mul(&q1, &l.t0);
        drop(q1);
        let t1 = cut.add(u, v);
        let mut u = cut.mul(&l.p, &r.t0);
        Cut::shl(&mut u, sh);
        let v = cut.mul(&q0, &l.t0);
        drop(q0);
        let t0 = cut.add(u, v);
        // `D = D1 D2`.
        let u = cut.mul(&l.d0, &r.d1);
        let v = cut.mul(&l.d1, &r.d0);
        let d1 = cut.add(u, v);
        let d0 = cut.mul(&l.d0, &r.d0);
        let p = if need_p {
            cut.mul(&l.p, &r.p)
        } else {
            Num::default()
        };
        Node { p, d0, d1, t0, t1 }
    }

    /// The node of the terms `a..b` exactly, from the right: the single term `k`, `T = m²`,
    /// `Q = (k + 1 + ε)²`, `P = m²`, merged in front of the rest gives `T ← m² (Q + T)`,
    /// `D ← (k + 1 + ε) D`, `Q ← (k + 1 + ε)² Q`, `P ← m² P`.
    fn leaf(&self, a: u64, b: u64, cut: &mut Cut) -> Node {
        let mut k1 = u128::from(b);
        let (mut d0, mut d1) = (vec![b], vec![1u64]);
        let mut q0 = mul_small(&[b], k1);
        let mut q1 = mul_small(&[b], 2);
        let mut t0 = limbs::shl(&mul_small(&[1], self.o2), self.z2);
        let mut t1 = Vec::new();
        let mut p = mul_small(&[1], self.o2);
        for k in (a..b - 1).rev() {
            t0 = limbs::shl(&mul_small(&limbs::add(&q0, &t0), self.o2), self.z2);
            t1 = limbs::shl(&mul_small(&limbs::add(&q1, &t1), self.o2), self.z2);
            k1 = u128::from(k) + 1;
            let kk = k1 * k1;
            // `Q1 ← (k + 1)² Q1 + 2 (k + 1) Q0`, `Q0 ← (k + 1)² Q0`.
            q1 = limbs::add(&mul_small(&q1, kk), &mul_small(&q0, 2 * k1));
            q0 = mul_small(&q0, kk);
            d1 = limbs::add(&mul_small(&d1, k1), &d0);
            d0 = mul_small(&d0, k1);
            p = mul_small(&p, self.o2);
        }
        let _ = k1;
        let mut node = Node {
            p: Num::exact(false, p),
            d0: Num::exact(false, d0),
            d1: Num::exact(false, d1),
            t0: Num::exact(false, t0),
            t1: Num::exact(false, t1),
        };
        for x in [
            &mut node.p,
            &mut node.d0,
            &mut node.d1,
            &mut node.t0,
            &mut node.t1,
        ] {
            cut.cut(x);
        }
        node
    }
}

/// The primes of the sets of logarithms.
const PRIMES: [u64; 4] = [2, 3, 5, 7];

/// A set of ratios `u_i/v_i` whose logarithms give those of its primes: `log p_j = (1/den) Σ_i
/// A_ji log(u_i/v_i)` (FLINT's sets).
struct LogSet {
    u: &'static [u64],
    v: &'static [u64],
    a: &'static [i64],
    den: i64,
}

/// The sets for the primes up to 2, 3, 5 and 7 (the first is `ln 2` itself).
const SETS: [LogSet; 4] = [
    LogSet {
        u: &[2],
        v: &[1],
        a: &[1],
        den: 1,
    },
    LogSet {
        u: &[9, 256],
        v: &[8, 243],
        a: &[5, 2, 8, 3],
        den: 1,
    },
    LogSet {
        u: &[81, 32805, 25],
        v: &[80, 32768, 24],
        a: &[17, -7, 12, 27, -11, 19, 39, -16, 28],
        den: 1,
    },
    LogSet {
        u: &[2401, 4375, 225, 64],
        v: &[2400, 4374, 224, 63],
        a: &[
            34, -10, 54, 72, 54, -16, 86, 114, 79, -23, 126, 167, 96, -28, 152, 202,
        ],
        den: 2,
    },
];

/// The modelled cost of the splittings for `m`: `m (61 + log2 o)` for the odd part `o`, `58 m` for
/// a power of two, whose `P` is free, times one and the cost of the logarithms (FLINT's model,
/// measured there).
fn cost(m: u64, set: usize) -> f64 {
    let o = m >> m.trailing_zeros();
    let base = if o == 1 {
        58.0
    } else {
        61.0 + (o as f64).log2()
    };
    const LOGS: [f64; 4] = [0.063, 0.066, 0.074, 0.072];
    m as f64 * base * (1.0 + LOGS[set])
}

/// The cheapest `m ≥ m0` up to `1.5 m0 + 2` whose primes are among [`PRIMES`], and its set.
fn choose_m(m0: u64) -> (u64, usize) {
    let mut best: Option<(f64, u64, usize)> = None;
    for m in m0..=m0 + m0 / 2 + 2 {
        let mut t = m;
        let mut set = 0;
        for (j, &p) in PRIMES.iter().enumerate() {
            while t.is_multiple_of(p) {
                t /= p;
                set = j;
            }
        }
        if t != 1 {
            continue;
        }
        let c = cost(m, set);
        if best.is_none_or(|(bc, _, _)| c < bc) {
            best = Some((c, m, set));
        }
    }
    let (_, m, set) = best.expect("a power of two lies in the range");
    (m, set)
}

/// `log(u/v)` for `u > v > 0` to `w` bits after the point, within less than 2 units of `2^-w`,
/// by Zuniga's series (2025, FLINT's `_mp_real_log_ratio_zuniga`):
///
/// `log(u/v) = c Σ_(k ≥ 1) P(k)/Q(k) Π_(j < k) R(j)/Q(j)`, `P = [P0, P1]`,
/// `Q = den [5, -36, 36]`, `R = num [0, -1, 2]`, `num/den = d^6 / (6 u² v² s²)` in lowest terms,
/// `d = u - v`, `s = u + v`, `P0 = -s² (u² - 8uv + v²)`, `P1 = 2 (u² - 14uv + v²)(u² + 4uv + v²)`,
/// `c = -s num / (2 d^5)`.
///
/// For the ratios used (`u/v ≤ 2`), `P0 > 0 > P1` and `P(1) < 0`, so every `P(k)` is negative: the
/// series is summed with `-P` and `-c`, positive. The ratio of terms is
/// `|P(k+1)/P(k)| R(k)/Q(k+1)`, the first at most `1 + |P1|/(|P1| - P0)` (its value at `k = 1`, as
/// `|P(k)| = |P1| k - P0` grows), the second below `num/(18 den)` as `18 (2k² - k) < 36k² + 36k + 5`.
pub(super) fn log_ratio(u: u64, v: u64, w: u64) -> Vec<u64> {
    let (u, v) = (i128::from(u), i128::from(v));
    let (d, s) = (u - v, u + v);
    let (uu, vv, uv) = (u * u, v * v, u * v);
    let mut num = d.pow(6);
    let mut den = 6 * (uv * s) * (uv * s);
    let g = gcd(num, den);
    num /= g;
    den /= g;
    let p0 = -(s * s) * (uu + vv - 8 * uv);
    let p1 = 2 * (uu + vv - 14 * uv) * (uu + vv + 4 * uv);
    assert!(p0 > 0 && p1 < 0 && p0 + p1 < 0);
    let ratio = (1.0 + (-p1) as f64 / ((-p1 - p0) as f64)) * (num as f64 / (18.0 * den as f64));
    let series = Series {
        p: Poly(vec![-p0, -p1]),
        q: Poly(vec![5 * den, -36 * den, 36 * den]),
        r: Poly(vec![0, -num, 2 * num]),
        ratio: ratio * (1.0 + 1e-9),
        positive: true,
    };
    // `c = s num / (2 d^5)` (after the change of sign), below `2^8` for these ratios: the tail
    // times `c` at most `2^-(w + 2)`.
    let (cn, cd) = (s * num, 2 * d.pow(5));
    let cbits = (cn as f64 / cd as f64).log2().max(0.0).ceil();
    let n = series.terms(w as f64 + 2.0 + cbits + 1.0);
    fixed_ratio(&series, n, w, cn as u128, cd as u128, 0, (cbits as u64) + 1)
}

/// The greatest common divisor.
fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// `(c_Q Q + c_P T)/(c_D Q)` of a positive series summed over `n` terms, to `w` bits after the
/// point, the value below `2^int_bits`: within `1.25 + 1/4` units of `2^-w` of the sum's value,
/// to which the caller adds its tail.
///
/// The tree's numbers are cut to `W = w + int_bits + 40` bits; the quotient's relative error is
/// then below `(e_N + e_D + 1) 2^-W` for the error counts of the numerator and the denominator,
/// at most `1/4` unit of a result below `2^(w + int_bits)` while the counts stay below `2^37`, and
/// otherwise the sum is made again with 64 bits more (it never was, in the tests).
pub(super) fn fixed_ratio(
    series: &Series,
    n: u64,
    w: u64,
    cp: u128,
    cd: u128,
    cq: u128,
    int_bits: u64,
) -> Vec<u64> {
    let mut extra = 40;
    loop {
        let cw = w + int_bits + extra;
        let (t, q) = series.sum(n, Some(cw));
        let mut cut = Cut::new(Some(cw));
        let c = |x: u128| Num::of_i128(x as i128);
        let num = {
            let a = cut.mul(&c(cp), &t);
            let b = cut.mul(&c(cq), &q);
            cut.add(a, b)
        };
        let den = cut.mul(&c(cd), &q);
        if let Some(x) = fixed_quotient(&num, &den, w, cw) {
            return x;
        }
        extra += 64;
    }
}

/// `num/den·2^w` within `1.25 + 1/4` units for positive numbers cut to `cw` bits, or `None` when
/// their errors would allow more than `1/4` unit (see [`fixed_ratio`]).
pub(super) fn fixed_quotient(num: &Num, den: &Num, w: u64, cw: u64) -> Option<Vec<u64>> {
    let s = i128::from(w) + i128::from(num.e) - i128::from(den.e);
    let x = quotient(&num.m, s, &den.m);
    let units = num.err + den.err + 1;
    // `X (e_N + e_D + 1) 2^-W ≤ 2^(bits(X) + bits(units) - W) ≤ 1/4`.
    let bound = limbs::bit_len(&x) + u64::from(64 - units.leading_zeros());
    (bound + 2 <= cw).then_some(x)
}

/// `γ` to `w` bits after the point, within less than 2 units of `2^-w`, with `ln2(w)` giving
/// `ln 2` likewise.
///
/// Everything is computed to `w' = w + 3` bits and the sum `Y = Y1 - Y2 - Y3` shifted down: with
/// `Y1` (`S0/I0`) and `Y2` (`K0/I0²`) within `1.5` units of `2^-w'` each, `Y3` (`log m`) within 2,
/// the error term `E` and the truncation of `S0` and `I0` within 1 each, `Y` is within 7 units of
/// `2^-w'`, `7/8` of `2^-w`, and the floor of `Y/8` within `1 + 7/8`.
pub(super) fn gamma(w: u64, ln2: impl Fn(u64) -> Vec<u64>) -> Vec<u64> {
    let w1 = w + 3;
    // `24 e^(-8m) ≤ 2^-w1` for `m ≥ (w1 + log2 24) ln 2 / 8`; `0.0866434 > ln 2 / 8` and
    // `log2 24 < 4.585`.
    let m0 = (((w1 as f64 + 4.6) * 0.086_643_4).ceil() as u64).max(2);
    let (m, set) = choose_m(m0);
    let n = sum_terms(m, w1);

    // `S0/I0` and the parts of `I0` for `K0/I0²`.
    let mut extra = 40u64;
    let (y1, q0, x) = loop {
        // `S0/I0 < H_N + 1 < 2^6` for `N < 2^40`.
        let cw = w1 + 6 + extra;
        let mut cut = Cut::new(Some(cw));
        let z = m.trailing_zeros();
        let o = u128::from(m >> z);
        let split = Split {
            o2: o * o,
            z2: 2 * u64::from(z),
        };
        let node = split.split(0, n, false, &mut cut);
        drop(node.p);
        // `A = 2 T0 D1`, `B = T1 D0`, `A - B = 2 D0 X S0/I0 > 0`.
        let mut a = cut.mul(&node.t0, &node.d1);
        Cut::shl(&mut a, 1);
        let b = cut.mul(&node.t1, &node.d0);
        let q0 = cut.sqr(&node.d0);
        let x = cut.add(node.t0, q0.clone());
        let mut den = cut.mul(&node.d0, &x);
        Cut::shl(&mut den, 1);
        let diff = sub_cancel(&a, &b);
        if let Some(diff) = diff
            && let Some(y1) = fixed_quotient(&diff, &den, w1, cw)
        {
            break (y1, q0, x);
        }
        extra += 64;
    };

    // `K0/I0² = (Q_K + T_K) Q0² / (4m Q_K X²)`, about `2^-(w/2)`: to `w/2` bits and a margin.
    let y2 = {
        let mm = i128::from(m) * i128::from(m);
        let series = Series {
            p: Poly(vec![-1, 6, -12, 8]),
            q: Poly(vec![0, 32 * mm]),
            r: Poly(vec![-1, 6, -12, 8]),
            ratio: 0.5,
            positive: true,
        };
        let mut extra = 40u64;
        loop {
            // `I0 ≥ e^(2m) / (2π m e^(1/(6m)))` and `K0 < 1`: `K0/I0²` is below
            // `2^(-5.7708 m + 2 log2(2π m) + 1)`, at most `2^mag` units of `2^-w1` (about
            // `2^(w1/2)`), so `cw` bits keep 40 more than the quotient has. It only sizes the cut:
            // `fixed_quotient` checks the bound that the result relies on.
            let mag = w1 as f64 - 5.77 * m as f64 + 2.0 * (7.0 * m as f64).log2() + 2.0;
            let cw = (mag.max(0.0) as u64) + extra;
            let (t, q) = series.sum(2 * m - 1, Some(cw));
            let mut cut = Cut::new(Some(cw));
            let mut q0c = q0.clone();
            cut.cut(&mut q0c);
            let mut xc = x.clone();
            cut.cut(&mut xc);
            let num = {
                let s = cut.add(t, q.clone());
                let q02 = cut.sqr(&q0c);
                cut.mul(&s, &q02)
            };
            let den = {
                let x2 = cut.sqr(&xc);
                let qx = cut.mul(&q, &x2);
                cut.mul(&Num::of_i128(4 * i128::from(m)), &qx)
            };
            if let Some(y2) = fixed_quotient(&num, &den, w1, cw) {
                break y2;
            }
            extra += 64;
        }
    };
    drop(q0);
    drop(x);

    // `log m`.
    let y3 = log_m(m, set, w1, ln2);

    // `Y1 - Y2 - Y3 > 0`, then `floor(Y/8)`.
    let y = limbs::sub(&limbs::sub(&y1, &y2), &y3);
    limbs::shr(&y, 3)
}

/// `a - b` for `a > b > 0` within `a`'s and `b`'s errors grown by the cancellation:
/// `|δ(a - b)| ≤ max(e_a, e_b)(a + b)` and `(a + b)/(a - b) < 2^(top(a) + 1 - (top(a - b) - 1))`;
/// `None` if `b ≥ a` (the errors would then decide nothing).
fn sub_cancel(a: &Num, b: &Num) -> Option<Num> {
    let e = a.e.min(b.e);
    let ma = limbs::shl(&a.m, a.e - e);
    let mb = limbs::shl(&b.m, b.e - e);
    if limbs::cmp(&ma, &mb) != core::cmp::Ordering::Greater {
        return None;
    }
    let mut d = limbs::sub(&ma, &mb);
    limbs::trim(&mut d);
    let grow = (limbs::bit_len(&ma) + 2).saturating_sub(limbs::bit_len(&d));
    let err = a.err.max(b.err).checked_shl(grow as u32)?;
    (err < 1 << 32).then_some(Num {
        neg: false,
        m: d,
        e,
        err,
    })
}

/// The number of terms `N` of `S0` and `I0`: at least `2m` and `α m + 1`, `α = 4.970626 >
/// 3/W(3/e)` (Brent–Johansson), and enough for the tails to move `S0/I0` by at most `2^-w`.
///
/// For `k ≥ N ≥ 2m`, `u_(k+1)/u_k = m²/(k+1)² ≤ 1/4` and `H_(k+1) ≤ H_k (1 + 1/(k+1))`, so the
/// tails are at most `2 u_N` and `2 u_N H_N`, and they move `S0/I0` by at most
/// `(2 u_N H_N + H_N 2 u_N)/I0 = 4 u_N H_N / I0`. With Stirling's bounds
/// `k ln k - k + ln(2πk)/2 ≤ ln k! ≤ k ln k - k + ln(2πk)/2 + 1/(12k)`, `ln u_N ≤
/// 2(N ln m - N ln N + N - ln(2πN)/2)` and `I0 ≥ u_m ≥ e^(2m) / (2πm e^(1/(6m)))`; `H_N ≤ ln N +
/// 1`. The bound is evaluated in floating point with a bit of margin.
fn sum_terms(m: u64, w: u64) -> u64 {
    let mf = m as f64;
    let mut n = (2 * m).max((4.970_626 * mf).ceil() as u64 + 1);
    let ln_i0 = 2.0 * mf - (2.0 * core::f64::consts::PI * mf).ln() - 1.0 / (6.0 * mf);
    loop {
        let nf = n as f64;
        let ln_un = 2.0
            * (nf * mf.ln() - nf * nf.ln() + nf - 0.5 * (2.0 * core::f64::consts::PI * nf).ln());
        let ln_h = (nf.ln() + 1.0).ln();
        let log2 = (4f64.ln() + ln_un + ln_h - ln_i0) / core::f64::consts::LN_2;
        if log2 + 1.0 <= -(w as f64) {
            return n;
        }
        n += 1 + n / 64;
    }
}

/// `log m` to `w` bits after the point within less than 2 units, for `m` of the primes of the set
/// `set`: `Σ_j e_j log p_j = (1/den) Σ_i c_i log(u_i/v_i)` with `c_i = Σ_j e_j A_ji`, each logarithm
/// to `w + g` bits within 2 units, `g` the bits of `Σ |c_i|` and one more, so that the sum is
/// within `2 Σ |c_i| / 2^g < 1` unit of `2^-w` before the floor.
fn log_m(m: u64, set: usize, w: u64, ln2: impl Fn(u64) -> Vec<u64>) -> Vec<u64> {
    let ls = &SETS[set];
    let np = ls.u.len();
    let mut e = [0i64; 4];
    let mut t = m;
    for (j, &p) in PRIMES.iter().enumerate().take(np) {
        while t.is_multiple_of(p) {
            t /= p;
            e[j] += 1;
        }
    }
    debug_assert_eq!(t, 1);
    let c: Vec<i64> = (0..np)
        .map(|i| (0..np).map(|j| e[j] * ls.a[j * np + i]).sum())
        .collect();
    let total: u64 = c.iter().map(|x| x.unsigned_abs()).sum();
    let g = u64::from(64 - total.leading_zeros()) + 1;
    let mut pos: Vec<u64> = Vec::new();
    let mut neg: Vec<u64> = Vec::new();
    for (i, &ci) in c.iter().enumerate() {
        if ci == 0 {
            continue;
        }
        let l = if set == 0 {
            ln2(w + g)
        } else {
            log_ratio(ls.u[i], ls.v[i], w + g)
        };
        let term = limbs::mul(&l, &[ci.unsigned_abs()]);
        if ci > 0 {
            pos = limbs::add(&pos, &term);
        } else {
            neg = limbs::add(&neg, &term);
        }
    }
    let s = limbs::sub(&pos, &neg);
    let den = ls.den as u64;
    let s = if den == 2 { limbs::shr(&s, 1) } else { s };
    let _ = den;
    limbs::shr(&s, g)
}
