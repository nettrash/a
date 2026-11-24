//! Knuth's algorithm D at its rare steps, against `Number`: the add-back of step D6, the quotient
//! estimate at the base `2⁶⁴` in step D3, and quotients of every shape built as `u = q·v + r`.

use super::common::{Rng, size};
use a::math::numbers::Number;
use a::math::uint::Uint;

/// Checks `u / v` and `u % v` against `Number`, and the identity `u = q·v + r` with `r < v`.
fn check<const N: usize>(u: Uint<N>, v: Uint<N>) {
    let (q, r) = u.div_rem(v);
    let (nq, nr) = Number::from(u).div_rem(&Number::from(v));
    assert_eq!(
        (Number::from(q), Number::from(r)),
        (nq, nr),
        "{u:?} / {v:?}"
    );
    assert!(r < v);
    assert_eq!(u / v, q);
    assert_eq!(u % v, r);
    if v.bits() <= 64 {
        let (q1, r1) = u.div_rem_u64(v.as_limbs()[0]);
        assert_eq!(
            (q1, Uint::from_u64(r1)),
            (q, r),
            "{u:?} / {v:?} by one limb"
        );
    }
}

/// The add-back case: with the divisor `v = V'·b^(n−2) + W` (its top two limbs `V'` normalised,
/// `W ≥ 1` below them) and the dividend `u = Q·V'·b^(n−2) + U₀` with `U₀ < Q·W`, the estimate
/// from the top three limbs over `V'` is exactly `Q`, but `Q·v > u`: the true quotient is `Q − 1`
/// and step D4 goes below zero. Built for each `n` from 3 to the width `N`, with lower limbs below
/// it (when there is room) so that it happens at a lower quotient limb too, and with the divisor
/// shifted down by `s` bits so that the normalisation shifts it back. A divisor of the full width,
/// `n = N`, leaves no room for the extra limb of the dividend: there `s ≥ 1` and `Q ≤ 2^s`, so
/// that `u = Q·V'·b^(N−2) >> s` fits and the normalisation makes the extra limb. Returns the
/// number of divisions checked for each `n`.
fn add_back<const N: usize>(rng: &mut Rng) -> Vec<(usize, usize)> {
    let b = |k: usize| Number::from(1u64) << (64 * k);
    let mut checked = Vec::new();
    // With two limbs there is no W, and after D3 the estimate is never too large: n ≥ 3.
    for n in 3..=N {
        let mut count = 0;
        for _ in 0..size(30, 3) {
            let v1 = (1u64 << 63) | rng.next();
            let v0 = rng.next();
            let vp = (Number::from(v1) << 64) + Number::from(v0);
            let full = n == N;
            let s = if full {
                1 + rng.below(63) as usize
            } else {
                rng.below(64) as usize
            };
            let q = if full {
                Number::from(1 + rng.below(1 << s))
            } else {
                Number::from(1 + rng.below(u64::MAX - 1))
            };
            // W ≥ 1 below 2^(64(n − 2)) with its low s bits clear, so that v >> s is exact.
            let mut w = Number::from(0u64);
            for k in 0..n - 2 {
                w += Number::from(rng.next()) << (64 * k);
            }
            let w = (((w >> s) << s) + (Number::from(1u64) << s)).rem_euclid(&b(n - 2));
            let w = if w.is_zero() {
                Number::from(1u64) << s
            } else {
                w
            };
            let v = &vp * &b(n - 2) + &w;
            // U₀ = 0, which is below Q·W and keeps the low s bits clear as well.
            let u = &q * &vp * &b(n - 2);
            // Lower limbs below: the same case at the quotient limb `low`.
            let low = if full {
                0
            } else {
                rng.below((N - n) as u64) as usize
            };
            let mut below = Number::from(0u64);
            for k in 0..low {
                below += Number::from(rng.limb()) << (64 * k);
            }
            let u = (u << (64 * low)) + below;
            let (u, v) = (&u >> s, &v >> s);
            let (u, v) = (
                Uint::<N>::try_from(&u).expect("the dividend fits"),
                Uint::<N>::try_from(&v).expect("the divisor fits"),
            );
            // The divisor has n limbs.
            assert!(v.bits() > 64 * (n as u32 - 1));
            check(u, v);
            // Without lower limbs, the shifts are exact and the quotient is Q − 1.
            if low == 0 {
                assert_eq!(Number::from(u / v), &q - Number::from(1u64));
            }
            count += 1;
        }
        checked.push((n, count));
    }
    checked
}

#[test]
fn the_add_back_step_gives_the_true_quotient() {
    fn all<const N: usize>(rng: &mut Rng) {
        let checked = add_back::<N>(rng);
        // Every divisor length from 3 limbs to the full width was tried.
        assert_eq!(checked.len(), N - 2);
        for (n, count) in checked {
            assert_eq!(count, size(30, 3), "N = {N}, n = {n}");
        }
    }
    let mut rng = Rng(0xadd_bac);
    all::<3>(&mut rng);
    all::<4>(&mut rng);
    all::<5>(&mut rng);
    all::<6>(&mut rng);
    all::<7>(&mut rng);
    all::<8>(&mut rng);
    all::<9>(&mut rng);
    all::<16>(&mut rng);
    // The smallest instance: 2^192 / (2^191 + 2^64 − 1), whose estimate is 2 and quotient 1.
    let u = Uint::<4>::from_limbs([0, 0, 0, 1]);
    let v = Uint::<4>::from_limbs([u64::MAX, 0, 1 << 63, 0]);
    assert_eq!(u / v, Uint::ONE);
    check(u, v);
    // And at the full width of three limbs: 2^191 / ((2^191 + 2^64 − 2) / 2), which the
    // normalisation shifts by one bit back to 2^192 / (2^191 + 2^64 − 2): the estimate is 2 and
    // the quotient 1.
    let v = Uint::<3>::from_limbs([u64::MAX - 1, 0, 1 << 63]) >> 1u32;
    let u = Uint::<3>::from_limbs([0, 0, 1 << 63]);
    assert_eq!(u / v, Uint::ONE);
    check(u, v);
}

/// The estimate at the base: when the remainder's top limb equals the divisor's, the two-limb
/// estimate is `2⁶⁴` or more and must come down. `u = [u₀, u₁, v₁]` with `u₁ < v₀` over `v = [v₀,
/// v₁]`: the first quotient limb is 0, after which the remainder's top limb is `v₁`.
fn estimate_at_the_base<const N: usize>(rng: &mut Rng) {
    for _ in 0..size(200, 10) {
        let v1 = (1u64 << 63) | rng.next();
        let v0 = 1 + rng.below(u64::MAX - 1);
        let u1 = rng.below(v0);
        let mut ul = [0u64; N];
        let mut vl = [0u64; N];
        // At a random position, with random lower limbs under the dividend.
        let at = rng.below((N - 2) as u64) as usize;
        for x in ul.iter_mut().take(at) {
            *x = rng.limb();
        }
        ul[at] = rng.next();
        ul[at + 1] = u1;
        ul[at + 2] = v1;
        vl[0] = v0;
        vl[1] = v1;
        check(Uint::from_limbs(ul), Uint::from_limbs(vl));
        // The same with longer divisors: the top two limbs of the divisor as above.
        if N > 3 {
            let n = 2 + rng.below((N - 2) as u64) as usize;
            let mut vl = [0u64; N];
            for x in vl.iter_mut().take(n - 2) {
                *x = rng.limb();
            }
            vl[n - 2] = v0;
            vl[n - 1] = v1;
            let mut ul = [0u64; N];
            if n < N {
                for x in ul.iter_mut().take(N - 2) {
                    *x = rng.limb();
                }
                ul[N - 2] = u1;
                ul[N - 1] = v1;
                check(Uint::from_limbs(ul), Uint::from_limbs(vl));
            }
        }
    }
}

/// The estimate at the base that only the test `q̂ ≥ 2⁶⁴` brings down: the remainder's top two
/// limbs equal the divisor's, `(v₁, v₀)`, with its lower limbs below the divisor's, so that the
/// estimate is `2⁶⁴` and the test with the third limb does not fire. `u = [x, u₀, v₀, v₁]` over
/// `v = [w, v₀, v₁]` with `u₀ < w` and `v₀ < v₁`: the first quotient limb takes the add-back
/// to 0, and the second is `2⁶⁴ − 1` or one less.
fn top_limbs_equal<const N: usize>(rng: &mut Rng) {
    for _ in 0..size(200, 10) {
        let v1 = (1u64 << 63) | rng.next();
        let v0 = rng.below(v1);
        let w = 1 + rng.below(u64::MAX - 1);
        let u0 = rng.below(w);
        let at = rng.below((N - 3) as u64) as usize;
        let mut ul = [0u64; N];
        for x in ul.iter_mut().take(at + 1) {
            *x = rng.limb();
        }
        ul[at + 1] = u0;
        ul[at + 2] = v0;
        ul[at + 3] = v1;
        let mut vl = [0u64; N];
        vl[0] = w;
        vl[1] = v0;
        vl[2] = v1;
        let (u, v) = (Uint::<N>::from_limbs(ul), Uint::<N>::from_limbs(vl));
        check(u, v);
        // Shifted down, the normalisation brings the case back.
        let s = rng.below(64) as u32;
        if (w | u0) & ((1 << s) - 1) == 0 && ul[..=at].iter().all(|&x| x == 0) {
            check(u >> s, v >> s);
        }
    }
}

#[test]
fn the_quotient_estimate_at_the_base_is_corrected() {
    let mut rng = Rng(0x70b5);
    top_limbs_equal::<4>(&mut rng);
    top_limbs_equal::<5>(&mut rng);
    top_limbs_equal::<9>(&mut rng);
    top_limbs_equal::<16>(&mut rng);
    let mut rng = Rng(0xba5e);
    estimate_at_the_base::<3>(&mut rng);
    estimate_at_the_base::<4>(&mut rng);
    estimate_at_the_base::<5>(&mut rng);
    estimate_at_the_base::<8>(&mut rng);
    estimate_at_the_base::<9>(&mut rng);
    estimate_at_the_base::<16>(&mut rng);
}

/// Quotients of every shape: `u = q·v + r` with `q`, `v` and `r < v` drawn with limbs at the
/// edges (all ones, zero, one bit), the divisor of every length up to the width, normalised or
/// not, and `q` up to the largest that keeps `u` in range; the division must give `q` and `r`
/// back.
fn built<const N: usize>(rng: &mut Rng, count: usize) {
    for _ in 0..count {
        let v = rng.uint::<N>();
        if v.is_zero() {
            continue;
        }
        let r = match rng.below(4) {
            0 => v - Uint::ONE,
            1 => Uint::ZERO,
            _ => rng.uint::<N>() % v,
        };
        let qmax = (Uint::MAX - r) / v;
        let q = match rng.below(4) {
            0 => qmax,
            1 => qmax >> rng.below(64) as u32,
            _ => match qmax.checked_add(Uint::ONE) {
                Some(m) => rng.uint::<N>() % m,
                None => rng.uint::<N>(),
            },
        };
        let u = q * v + r;
        assert_eq!(
            Number::from(u),
            Number::from(q) * Number::from(v) + Number::from(r)
        );
        assert_eq!(u.div_rem(v), (q, r), "{u:?} / {v:?}");
        check(u, v);
    }
}

#[test]
fn quotients_built_as_q_v_plus_r_come_back() {
    let mut rng = Rng(0xb0117);
    let n = size(2000, 20);
    built::<1>(&mut rng, n);
    built::<2>(&mut rng, n);
    built::<3>(&mut rng, n);
    built::<4>(&mut rng, n);
    built::<5>(&mut rng, n);
    built::<6>(&mut rng, n);
    built::<7>(&mut rng, n);
    built::<8>(&mut rng, n);
    built::<9>(&mut rng, n);
    built::<16>(&mut rng, n / 2);
    built::<32>(&mut rng, n / 4);
}

#[test]
fn division_by_zero_panics_in_every_form() {
    use super::common::panics;
    let x = Uint::<3>::from_u64(5);
    let z = Uint::<3>::ZERO;
    assert!(panics(|| x / z));
    assert!(panics(|| x % z));
    assert!(panics(|| x.div_rem(z)));
    assert!(panics(|| x.div_rem_u64(0)));
    assert!(panics(|| x.wrapping_div(z)));
    assert!(panics(|| x.overflowing_rem(z)));
    assert!(panics(|| x.div_euclid(z)));
    assert!(panics(|| x.rem_euclid(z)));
    assert!(panics(|| x.div_ceil(z)));
    assert!(panics(|| x.next_multiple_of(z)));
    assert_eq!(x.checked_div(z), None);
    assert_eq!(x.checked_rem(z), None);
    assert_eq!(x.checked_next_multiple_of(z), None);
    assert!(!x.is_multiple_of(z));
    assert!(z.is_multiple_of(z));
    // The messages are the primitives'.
    let m = super::common::outcome(|| x / z).unwrap_err();
    assert_eq!(m, "attempt to divide by zero");
    let m = super::common::outcome(|| x % z).unwrap_err();
    assert_eq!(
        m,
        "attempt to calculate the remainder with a divisor of zero"
    );
}
