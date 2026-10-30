//! Division by Newton's method, for the longest divisions: an approximate inverse of the
//! divisor's top limbs, by Newton's iteration, and the quotient in blocks, each a product of the
//! inverse by the top limbs of the partial remainder, after GMP's `mpn_mu_div_qr` and Brent and
//! Zimmermann, "Modern Computer Arithmetic" (2010), §3.4. Divide and conquer spends a product at
//! each level of its recursion, O(M(n) log n) for products of n limbs in M(n); this, about three
//! products of its length for a quotient as long as the divisor, O(M(n)).
//!
//! # Products modulo `B^m - 1`
//!
//! Both the iteration and the division multiply where the high part or the whole of the product
//! is known up to a small range: there, the product modulo `B^m - 1` for a modulus just above
//! that range tells it, and the transforms of the multiplication compute a product modulo
//! `B^m - 1` as a cyclic convolution, at the cost of a product whose result has `m` limbs (see
//! the multiplication's `fft`): about half of the full one. An operand that several products
//! share, the inverse in each block's estimate and the divisor in each block's remainder, is
//! transformed once, and all the products of a division take their twiddles from one table.
//! Below [`CYCLIC_MIN`] limbs ([`CYCLIC_MIN_SHARED`] for a shared operand), a product modulo
//! `B^m - 1` is the product, folded.
//!
//! # The inverse
//!
//! [`invert`] gives, for a normalized `A` of `n` limbs (`B^n / 2 <= A < B^n`), `X = B^n + I` with
//! `A X < B^(2n) <= A (X + 2)`, so that `X` is `floor((B^(2n) - 1) / A)` or one less, from the
//! inverse `X_h` of the top `h` limbs of `A` (proof at [`newton_step`]): the last step takes a
//! product modulo `B^m - 1` and a product of about half its length, both by `X_h`, transformed
//! once, and the steps halve, so the whole costs about two products of `n` limbs.
//!
//! # The division
//!
//! [`run`] divides by `D` of `dn` limbs with the inverse `X` of `D_t + 1`, `D_t` the top limbs
//! of `D`: each block of up to `in` quotient limbs is the top limbs of `W_t X_in / B^in`, `W_t` the
//! top `in` limbs of the partial remainder with the next limbs of the dividend, at most 9 too
//! small (proof at [`run`]), and the partial remainder left, below `10 D`, is computed modulo
//! `B^m - 1` for an `m` of at least `dn + 1` limbs, then corrected by subtracting `D`. The
//! number of blocks comes from an estimate of the cost of the products ([`blocks`]). The quotient
//! alone skips the remainder of the last block when the limbs of the estimate below it show that
//! the error cannot change it. A `Divisor` keeps the inverse of all its limbs and the transforms
//! ([`Prepared`]), so that its later divisions take two products a block and no inverse.

use super::super::mul::fft::{Cyclic, Transform, Twiddles, cyclic, mul_cyclic, transform};
use super::super::mul::limbs::{add_1, add_n, cmp_n, mul, mul_itch, sub_1, sub_n};
use super::limbs::{DC_DIV, dc_itch, div_qr_classic, invert_3by2};
use core::cmp::Ordering;
use core::ptr;

/// From this many limbs in the divisor, a division whose quotient has at least [`MU_QUOTIENT`]
/// limbs takes Newton's method. Measured on an Apple M4 Max against divide and conquer, with
/// quotients of 32 times the divisor: 0.88–0.91 of its time for 128 limbs, 0.62–0.64 for 256 and
/// 0.45–0.47 for 384, but 1.31–1.35 for 96 and 1.37–1.39 for 64 limbs.
pub(in super::super) const MU_DIV: usize = if cfg!(miri) { 24 } else { 128 };

/// From this many limbs in the quotient, a division by a divisor of at least [`MU_DIV`] limbs
/// takes Newton's method. Measured on an Apple M4 Max against divide and conquer, by divisors of
/// 128 to 4096 limbs: the remainder of a quotient of 768 limbs took from 0.75 of its time to 1.04,
/// of 1024 limbs from 0.71 to 1.00, of 512 from 0.92 to 1.14; the quotient alone of 768 limbs
/// from 0.84 to 1.14, of 512 from 1.03 to 1.33.
pub(in super::super) const MU_QUOTIENT: usize = if cfg!(miri) { 40 } else { 768 };

/// From this many limbs in the divisor and in the quotient, a division by a divisor prepared for
/// Newton's method (see [`Prepared`]) takes it: its inverse is already there, and the products skip
/// a transform each. Measured on an Apple M4 Max against divide and conquer, a prepared division of
/// 256 limbs by 128 took 0.91 of its time, of 512 by 256 0.62, and of 1024 by 512 0.44, but of 128
/// by 64 1.41; decimal output, whose divisions by each power are few, was 10–14% faster from 64 KiB
/// with Newton's method for prepared divisors than without, and at 8 KiB 9% faster from 256 limbs
/// than from 128, where the first division by a power, which prepares it, costs more.
const MU_PREPARED: usize = if cfg!(miri) { 40 } else { 256 };

/// Below this many limbs, an inverse is a division of `B^(2n) - 1` (see [`invert`]). Measured on an
/// Apple M4 Max, the divisions of 4 KiB by 2 KiB to 16 KiB by 8 KiB were as fast from 40 to 110
/// limbs, and 5–17% slower from 150, more from 300.
const INV_NEWTON: usize = if cfg!(miri) { 8 } else { 80 };

/// From this many limbs of modulus, a product modulo `B^m - 1` is computed by the transforms;
/// below, as the product folded. Measured on an Apple M4 Max, divisions of 4 KiB by 2 KiB to 16
/// KiB by 8 KiB were 2–14% faster from 300 than from 150, which was itself 2–16% faster than 600.
const CYCLIC_MIN: usize = if cfg!(miri) { 40 } else { 300 };

/// [`CYCLIC_MIN`] for products by an operand that several of them share, transformed once: the
/// inverse in the estimates of a division of several blocks, the divisor in their remainders.
/// Measured on an Apple M4 Max, divisions with quotients of 8 to 32 times divisors of 192 to 256
/// limbs were 24–31% faster from 100 than from 300; from 60 to 100 as fast, and at 140 a divisor of
/// 128 limbs 26% slower.
const CYCLIC_MIN_SHARED: usize = if cfg!(miri) { 30 } else { 80 };

/// Whether a division of `nn` limbs by a prepared divisor of `dn` takes Newton's method.
#[inline]
pub(in super::super) fn mu_prepared(nn: usize, dn: usize) -> bool {
    dn >= MU_PREPARED && nn >= dn + MU_PREPARED.max(3)
}

/// Whether a division of `nn` limbs by `dn` takes Newton's method: from [`MU_DIV`] limbs in the
/// divisor and [`MU_QUOTIENT`] in the quotient, and a quotient of at least an eighth of the
/// divisor. Below, the product of the quotient by the divisor, which both methods take, costs the
/// most, and divide and conquer's division of the top limbs costs less than the inverse: measured on
/// an Apple M4 Max, the remainder of a quotient of 1024 limbs by a divisor of 131,072 took 1.22
/// times as long by Newton's method, of 4096 limbs 1.11, and of 8192 0.97; by a divisor of 32,768
/// limbs, 1.16 for 1024, 1.06 for 2048 and 0.88 for 4096; by one of 8192, 1.04 for 1024 and 0.84
/// for 2048. A quotient alone that short comes from the top limbs of both operands (see
/// `div_q_top`), whose division is balanced.
#[inline]
pub(in super::super) fn mu(nn: usize, dn: usize) -> bool {
    dn >= MU_DIV && nn >= dn + MU_QUOTIENT.max(3) && 8 * (nn - dn) >= dn
}

// Products modulo `B^m - 1`

/// How products modulo `B^m - 1` are computed: by the transforms, as a cyclic convolution, or as
/// the product folded.
#[derive(Clone, Copy)]
enum Modulus {
    Cyclic(Cyclic),
    Fold(usize),
}

impl Modulus {
    /// The way of least cost for a modulus of at least `min` limbs and products whose shorter
    /// operand has at most `shorter` limbs; `shared` when an operand is shared by several.
    fn of(min: usize, shorter: usize, shared: bool) -> Modulus {
        let from = if shared {
            CYCLIC_MIN_SHARED
        } else {
            CYCLIC_MIN
        };
        if min >= from
            && let Some(p) = cyclic(min, shorter)
        {
            Modulus::Cyclic(p)
        } else {
            Modulus::Fold(min)
        }
    }

    /// The limbs `m` of the modulus.
    fn limbs(self) -> usize {
        match self {
            Modulus::Cyclic(p) => p.limbs(),
            Modulus::Fold(m) => m,
        }
    }
}

/// An operand ready for products modulo a [`Modulus`]: its limbs, and their transform for a
/// cyclic one, which it owns.
struct Ready {
    x: *const u64,
    n: usize,
    t: Option<Transform>,
}

impl Ready {
    /// The `n` limbs at `x`, transformed for `m` when it is cyclic.
    ///
    /// # Safety
    ///
    /// `1 <= n <= m.limbs()`; the limbs must stay valid for reads, unchanged, while it is used.
    unsafe fn of(x: *const u64, n: usize, m: Modulus, tw: Option<&Twiddles>) -> Ready {
        let t = match m {
            // SAFETY: as the caller guarantees.
            Modulus::Cyclic(p) => Some(unsafe { transform(x, n, p, tw) }),
            Modulus::Fold(_) => None,
        };
        Ready { x, n, t }
    }

    /// The operand as products take it.
    fn side(&self) -> Side<'_> {
        Side {
            x: self.x,
            n: self.n,
            t: self.t.as_ref(),
        }
    }
}

/// An operand of [`mul_mod`]: its `n` limbs at `x`, and their transform for a cyclic modulus.
#[derive(Clone, Copy)]
struct Side<'a> {
    x: *const u64,
    n: usize,
    t: Option<&'a Transform>,
}

/// Sets the `m` limbs at `r` to the product of the `nx` limbs at `x` and `y` modulo `B^m - 1`,
/// `m` the modulus's limbs: from 0 to `B^m - 1`, which stands for 0.
///
/// # Safety
///
/// `1 <= nx <= m`, and `y` made for `m`: its limbs valid for reads, and transformed by its plan
/// when it is cyclic; the limbs at `x` must be valid for reads and the `m` at `r` for writes,
/// apart.
unsafe fn mul_mod(
    r: *mut u64,
    m: Modulus,
    x: *const u64,
    nx: usize,
    y: Side<'_>,
    tw: Option<&Twiddles>,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        match m {
            Modulus::Cyclic(p) => {
                let t = y.t.expect("an operand transformed for its modulus");
                mul_cyclic(r, p, x, nx, t, tw);
            }
            Modulus::Fold(mm) => {
                let (a, na, b, nb) = if nx >= y.n {
                    (x, nx, y.x, y.n)
                } else {
                    (y.x, y.n, x, nx)
                };
                let pn = na + nb;
                let mut t = Vec::<u64>::with_capacity(pn + mul_itch(na, nb));
                let t = t.as_mut_ptr();
                mul(t, a, na, b, nb, t.add(pn));
                if pn <= mm {
                    ptr::copy_nonoverlapping(t, r, pn);
                    ptr::write_bytes(r.add(pn), 0, mm - pn);
                } else {
                    ptr::copy_nonoverlapping(t, r, mm);
                    add_mod(r, mm, t.add(mm), pn - mm, 0);
                }
            }
        }
    }
}

/// Adds `y B^at` modulo `B^m - 1` to the `m` limbs at `r`, for the `ny` limbs at `y`: the limbs of
/// `y` that land at or above `B^m` are added at the bottom, and so is every carry out of the top,
/// `B^m = 1` modulo `B^m - 1`. The result is from 0 to `B^m - 1`, as the value before.
///
/// # Safety
///
/// `at < m`, `ny <= m`; the limbs at `r` must be valid for reads and writes and those at `y` for
/// reads, apart.
unsafe fn add_mod(r: *mut u64, m: usize, y: *const u64, ny: usize, at: usize) {
    debug_assert!(at < m && ny <= m);
    // SAFETY: as the caller guarantees: the low part of `y` lands in limbs `at..m`, its high part
    // in limbs `0..ny - (m - at) <= at`.
    unsafe {
        let low = ny.min(m - at);
        let mut carry = u64::from(add_n(r.add(at), r.add(at), y, low));
        carry = add_1(r.add(at + low), m - at - low, carry);
        if ny > low {
            let c = u64::from(add_n(r, r, y.add(low), ny - low));
            carry += add_1(r.add(ny - low), m - (ny - low), c);
        }
        while carry != 0 {
            carry = add_1(r, m, carry);
        }
    }
}

/// Subtracts `B^e` modulo `B^m - 1` from the `m` limbs at `r`, for `e < m`: a borrow out of the
/// top, `B^m` added, is `1` too much, and 1 is taken from the bottom, which cannot borrow again,
/// as the value is then at least `B^m - B^e`.
///
/// # Safety
///
/// The limbs at `r` must be valid for reads and writes.
unsafe fn sub_power(r: *mut u64, m: usize, e: usize) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if sub_1(r.add(e), m - e, 1) != 0 {
            sub_1(r, m, 1);
        }
    }
}

/// Whether the `n` limbs at `x` are all ones.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn all_ones(x: *const u64, n: usize) -> bool {
    // SAFETY: as the caller guarantees.
    (0..n).all(|i| unsafe { *x.add(i) } == u64::MAX)
}

// The inverse

/// Sets the `n` limbs at `x` to `I` for which `X = B^n + I` satisfies `A X < B^(2n) <= A (X + 2)`,
/// for the normalized `n` limbs `A` at `a`: by Newton's iteration from the inverse of the top half
/// of `A`, and below [`INV_NEWTON`] limbs as `floor((B^(2n) - 1) / A)`, by the division.
///
/// # Safety
///
/// `n >= 3`, `A` normalized; the limbs at `a` must be valid for reads and those at `x` for writes,
/// apart.
pub(super) unsafe fn invert(x: *mut u64, a: *const u64, n: usize, tw: Option<&Twiddles>) {
    // SAFETY: the `n` limbs at `a` are valid for reads, as the caller guarantees.
    debug_assert!(n >= 3 && unsafe { *a.add(n - 1) } >> 63 == 1);
    // SAFETY: as the caller guarantees.
    unsafe {
        if n < INV_NEWTON.max(6) {
            invert_base(x, a, n);
            return;
        }
        let l = (n - 1) / 2;
        let h = n - l;
        invert(x.add(l), a.add(l), h, tw);
        newton_step(x, a, n, h, tw);
    }
}

/// The modulus of the Newton step of [`invert`] to `n` limbs.
fn step_modulus(n: usize) -> Modulus {
    let h = n - (n - 1) / 2;
    Modulus::of((n + 1).max(2 * h + 1), h, false)
}

/// [`invert`] by the division: `B^(2n) - 1`, all ones, divided by `A`, whose quotient is `B^n + I`
/// as `B^n < (B^(2n) - 1) / A < 2 B^n`, and `X = floor((B^(2n) - 1) / A)` satisfies
/// `A X <= B^(2n) - 1 < A (X + 1)`.
///
/// # Safety
///
/// As for [`invert`].
unsafe fn invert_base(x: *mut u64, a: *const u64, n: usize) {
    let mut u = vec![u64::MAX; 2 * n];
    let mut t = Vec::<u64>::with_capacity(if n >= DC_DIV { dc_itch(2 * n, n) } else { 0 });
    // SAFETY: the dividend's `2n` limbs, the quotient's `n` at `x` and the scratch of the
    // division, apart.
    unsafe {
        let v = invert_3by2(*a.add(n - 1), *a.add(n - 2));
        let qh = div_qr_classic(x, u.as_mut_ptr(), 2 * n, a, n, v, t.as_mut_ptr());
        debug_assert!(qh, "the inverse is above B^n");
    }
}

/// The Newton step of [`invert`], after "Modern Computer Arithmetic", Algorithm 3.5: from `I_h`
/// in the top `h` limbs at `x`, the inverse of `A_h`, the top `h` limbs of `A`, sets the `n`
/// limbs at `x` to the inverse of `A`, with `l = n - h = floor((n - 1) / 2)`, so `h >= l + 1`.
///
/// With `X_h = B^h + I_h`, `A_h X_h < B^(2h) <= A_h (X_h + 2)`, and `A = A_h B^l + A_l`, the
/// product `T = A X_h` lies within `2 B^n` of `B^(n+h)`: `T < (B^(2h) - 1) B^l + B^l 2 B^h`, as
/// `X_h < 2 B^h`, and `T >= (B^(2h) - 2 A_h) B^l > B^(n+h) - 2 B^n`. So it is known from `T`
/// modulo `B^m - 1` for `m >= n + 1`, as `B^m - 1 > 4 B^n`: the residue of `T - B^(n+h)` is below
/// `2 B^n` for a value from 0 up, and above `B^m - 1 - 2 B^n` for a value below 0 (`B^m - 1`, all
/// ones, stands for 0). `T` is computed as `A I_h + A B^h`, a product of `n` limbs by `h`.
///
/// While `T >= B^(n+h)`, `X_h` takes one off and `T` takes `A` off: at most 4 times, as
/// `A >= B^n / 2`; each time `X_h > B^(n+h) / A > B^h`, so `I_h` stays at least 0. Then
/// `E = B^(n+h) - T` satisfies `0 < E < 2 B^n`: below `2 B^n` without a decrement, and at most `A`
/// after one. The step then sets `X = X_h B^l + floor(T_m X_h / B^(2h - l))` with
/// `T_m = floor(E / B^l)`, `h + 1` limbs, as `E < 2 B^n`.
///
/// Newton's step without the truncations, `X* = X_h B^l + X_h E / B^(2h)`, satisfies
/// `A X* = B^(2n) - E B^l + (B^(n+h) - E) E / B^(2h) = B^(2n) - E^2 / B^(2h)`, so
/// `A X* < B^(2n)`, and as `E^2 / B^(2h) < 4 B^(2l) <= 4 B^(n-1) <= 8 A / B`,
/// `A X* > B^(2n) - 8 A / B`. The truncations take `X` below `X*` by less than
/// `1 + X_h / B^(2h - l) < 1 + 2 B^(l - h) <= 1 + 2 / B`, and never above it, as
/// `T_m <= E / B^l`. So `A X <= A X* < B^(2n)`, and
/// `A (X + 2) > A X* + A (1 - 2 / B) > B^(2n) + A (1 - 10 / B) > B^(2n)`.
///
/// `T_m X_h` is `T_m I_h + T_m B^h`, below `4 B^(2h)`, `2h + 1` limbs; `I_h` is transformed once for
/// both products, the modulus at least `2h + 1` limbs, so that the second product is exact. When
/// `X_h` took `c` off, the second product is that of `I_h` before, less `c T_m`.
///
/// # Safety
///
/// As for [`invert`], with `I_h` in the top `h` limbs at `x`.
unsafe fn newton_step(x: *mut u64, a: *const u64, n: usize, h: usize, tw: Option<&Twiddles>) {
    let l = n - h;
    debug_assert!(h > l);
    let m = step_modulus(n);
    let mm = m.limbs();
    let mut t = Vec::<u64>::with_capacity(mm);
    let mut e = vec![0u64; n + 1];
    // SAFETY: `t` holds `m` limbs, `e` the `n + 1` of `E`; `I_h` is the top `h` limbs at `x`.
    unsafe {
        let t = t.as_mut_ptr();
        let e = e.as_mut_ptr();
        let ih = x.add(l);
        let ready = Ready::of(ih, h, m, tw);
        // `T - B^(n+h)` modulo `B^m - 1`.
        mul_mod(t, m, a, n, ready.side(), tw);
        add_mod(t, mm, a, n, h);
        sub_power(t, mm, (n + h) % mm);
        let positive = ((n + 1..mm).all(|i| *t.add(i) == 0) && *t.add(n) <= 1) || all_ones(t, mm);
        let mut c = 0;
        if positive {
            if !all_ones(t, mm) {
                ptr::copy_nonoverlapping(t, e, n + 1);
            }
            // `X_h` and `T` decremented until `T < B^(n+h)`: `e` holds `T - B^(n+h)` modulo
            // `B^(n+1)`, from 0 up, until it borrows.
            loop {
                sub_1(ih, h, 1);
                c += 1;
                let borrow = sub_n(e, e, a, n);
                if sub_1(e.add(n), 1, u64::from(borrow)) != 0 {
                    break;
                }
            }
            debug_assert!(c <= 4);
            // `E = B^(n+1) - e`.
            for i in 0..=n {
                *e.add(i) = !*e.add(i);
            }
            add_1(e, n + 1, 1);
        } else {
            for i in 0..=n {
                *e.add(i) = !*t.add(i);
            }
            debug_assert!((n + 1..mm).all(|i| *t.add(i) == u64::MAX));
        }
        debug_assert!(*e.add(n) <= 1);
        // `U = T_m X_h`, `T_m` the top `h + 1` limbs of `E`. A transform is of `I_h` before the
        // decrements, which are then taken off the product; a folded product reads `I_h` as it is.
        let tm = e.add(l);
        mul_mod(t, m, tm, h + 1, ready.side(), tw);
        if matches!(m, Modulus::Fold(_)) {
            c = 0;
        }
        for _ in 0..c {
            let borrow = sub_n(t, t, tm, h + 1);
            sub_1(t.add(h + 1), h, u64::from(borrow));
        }
        let carry = add_n(t.add(h), t.add(h), tm, h + 1);
        debug_assert!(!carry && (2 * h + 1..mm).all(|i| *t.add(i) == 0));
        // `I = I_h B^l + floor(U / B^(2h - l))`.
        ptr::copy_nonoverlapping(t.add(2 * h - l), x, l);
        let out = add_1(ih, h, *t.add(2 * h));
        debug_assert!(out == 0, "the inverse is below 2 B^n");
    }
}

/// Sets the `k` limbs at `x` to the inverse of `D_t + 1` for `D_t` the top `k` limbs of the
/// `dn` limbs at `d`: `I` for which `X = B^k + I` satisfies `D' X <= B^(2k) <= D' (X + 2)`,
/// `D' = D_t + 1`. When `D_t` is all ones, `D' = B^k`, and `I = 0` satisfies it.
///
/// # Safety
///
/// `3 <= k <= dn`, the divisor normalized; the limbs at `d` must be valid for reads and those at
/// `x` for writes.
unsafe fn top_inverse(x: *mut u64, d: *const u64, dn: usize, k: usize, tw: Option<&Twiddles>) {
    let mut a = Vec::<u64>::with_capacity(k);
    // SAFETY: as the caller guarantees.
    unsafe {
        let a = a.as_mut_ptr();
        ptr::copy_nonoverlapping(d.add(dn - k), a, k);
        if add_1(a, k, 1) != 0 {
            ptr::write_bytes(x, 0, k);
        } else {
            invert(x, a, k, tw);
        }
    }
}

// The division

/// The limbs of the inverse and the number of blocks of a division of `qn` quotient limbs by
/// `dn` divisor limbs: the inverse, of `in <= dn` limbs, gives blocks of up to `in` quotient limbs,
/// for the quotient alone the last one shorter than `in`, for the guard limbs of [`run`].
///
/// The number of blocks `b` is the one of least estimated cost, from the least that the divisor
/// allows, `ceil(qn / dn)`, to three more: with fewer, longer blocks the inverse costs more and the
/// products of each block are longer; with more, there are more of them. The estimate adds up the
/// transforms and the theorems of the plans of the products (see [`Cyclic::cost`]): the inverse's
/// steps, the transform of the inverse and of the divisor, and for each block a product by each of
/// them. Measured on an Apple M4 Max for quotients from half the divisor to five times it, of 1 to
/// 16 thousand limbs, its choice was within 4% of the fastest of two, three and four blocks, and
/// up to 22% faster than two, which GMP's `mpn_mu_div_qr_choose_in` takes for a balanced division.
fn blocks(qn: usize, dn: usize, rem: bool) -> (usize, usize) {
    // The limbs of the inverse for `b` blocks, and the blocks they then take.
    let shape = |b: usize| {
        let inn = if rem {
            qn.div_ceil(b)
        } else {
            (qn + 1).div_ceil(b)
        };
        let inn = inn.max(3).min(dn);
        (inn, qn.div_ceil(inn))
    };
    let first = qn.div_ceil(dn).max(1);
    let mut best: Option<((usize, usize), u64)> = None;
    for b in first..first + 4 {
        let (inn, b) = shape(b);
        if !rem && b * inn == qn && best.is_some() {
            // No guard limb: the last block is a full one.
            continue;
        }
        let Some(cost) = model(dn, inn, b, rem) else {
            return shape(first.max(if 3 * qn > dn { 2 } else { 1 }));
        };
        if best.is_none_or(|(_, c)| cost < c) {
            best = Some(((inn, b), cost));
        }
    }
    best.map(|(s, _)| s).unwrap_or_else(|| shape(first))
}

/// The estimated cost of a division by `dn` divisor limbs in `b` blocks with an inverse of `inn`
/// limbs (see [`blocks`]), or `None` when a product does not fit the transforms. Products below
/// [`CYCLIC_MIN`] are estimated as the transforms would compute them.
fn model(dn: usize, inn: usize, b: usize, rem: bool) -> Option<u64> {
    let plan = |min: usize, shorter: usize| cyclic(min, shorter);
    let mut cost = 0;
    // The inverse: each step transforms `I_h` and multiplies it twice.
    let mut n = inn;
    while n >= INV_NEWTON.max(6) {
        let h = n - (n - 1) / 2;
        let p = plan((n + 1).max(2 * h + 1), h)?;
        cost += p.transform_cost() + 2 * p.cost(true);
        n = h;
    }
    let q = plan(2 * inn, inn)?;
    cost += q.transform_cost() + b as u64 * q.cost(true);
    let remainders = if rem { b } else { b - 1 };
    if remainders > 0 {
        let r = plan(dn + 1, inn)?;
        cost += r.transform_cost() + remainders as u64 * r.cost(true);
    }
    Some(cost)
}

/// [`div_qr`](super::limbs::div_qr) by Newton's method: sets the `nn - dn` limbs at `q` to the
/// quotient of the `nn` limbs at `n` by the `dn` limbs at `d`, leaves the remainder in the low `dn`
/// limbs of `n`, and returns the quotient's limb above them, 0 or 1. Allocates its working memory.
///
/// # Safety
///
/// As for [`div_qr`](super::limbs::div_qr): `nn >= dn + 3`, `dn >= 3`, the divisor normalized, the
/// limbs apart.
pub(super) unsafe fn mu_div_qr(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
) -> bool {
    // SAFETY: forwarded.
    unsafe { mu_div(q, n, nn, d, dn, true) }
}

/// The quotient alone of [`mu_div_qr`]; the limbs at `n` are lost.
///
/// # Safety
///
/// As for [`mu_div_qr`].
pub(super) unsafe fn mu_div_q(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
) -> bool {
    // SAFETY: forwarded.
    unsafe { mu_div(q, n, nn, d, dn, false) }
}

/// The twiddles for the products by `moduli` that are cyclic, if any is.
fn twiddles_for(moduli: &[Modulus]) -> Option<Twiddles> {
    let plans: Vec<Cyclic> = moduli
        .iter()
        .filter_map(|m| match m {
            Modulus::Cyclic(p) => Some(*p),
            Modulus::Fold(_) => None,
        })
        .collect();
    (!plans.is_empty()).then(|| Twiddles::of(&plans))
}

/// The division of [`mu_div_qr`], or with `rem` false of [`mu_div_q`]: the inverse of the
/// divisor's top `in` limbs (see [`blocks`]), the transforms, and [`run`].
///
/// # Safety
///
/// As for [`mu_div_qr`].
unsafe fn mu_div(q: *mut u64, n: *mut u64, nn: usize, d: *const u64, dn: usize, rem: bool) -> bool {
    let qn = nn - dn;
    let (inn, b) = blocks(qn, dn, rem);
    let mq = Modulus::of(2 * inn, inn, b > 1);
    let mr = Modulus::of(dn + 1, inn, b > 2 || (rem && b > 1));
    // One table of twiddles for every product of the division.
    let twiddles = twiddles_for(&[mq, mr, step_modulus(inn)]);
    let tw = twiddles.as_ref();
    let mut inverse = Vec::<u64>::with_capacity(inn);
    // SAFETY: as the caller guarantees; the inverse's `in` limbs are in the vector.
    unsafe {
        let inv = inverse.as_mut_ptr();
        top_inverse(inv, d, dn, inn, tw);
        let ready_inv = Ready::of(inv, inn, mq, tw);
        // The divisor is transformed for the remainders when there is more than one to compute.
        let ready_d = (rem || b > 1).then(|| Ready::of(d, dn, mr, tw));
        let blocks = Blocks {
            inn,
            b,
            mq,
            inv: ready_inv.side(),
            mr,
            d: ready_d.as_ref().map(Ready::side),
            tw,
        };
        run(q, n, nn, d, dn, &blocks, rem)
    }
}

/// What a divisor keeps for its divisions by Newton's method (see the `Divisor`): the inverse of
/// all its limbs, plus one, and the transforms of that inverse and of the divisor for the products
/// of the blocks.
#[derive(Clone)]
pub(in super::super) struct Prepared {
    /// `I`, `dn` limbs: `X = B^dn + I` is the inverse of `D + 1`.
    inv: Vec<u64>,
    /// The products by the inverse, of `2 dn` limbs, and its transform for them.
    mq: Modulus,
    t_inv: Option<Transform>,
    /// The products by the divisor modulo `B^m - 1`, `m > dn`, and its transform for them.
    mr: Modulus,
    t_d: Option<Transform>,
}

/// Prepares the normalized `dn` limbs at `d` for divisions by Newton's method.
///
/// # Safety
///
/// `dn >= 3`; the limbs at `d` must be valid for reads.
pub(in super::super) unsafe fn prepare(d: *const u64, dn: usize) -> Prepared {
    let mq = Modulus::of(2 * dn, dn, true);
    let mr = Modulus::of(dn + 1, dn, true);
    let twiddles = twiddles_for(&[mq, mr, step_modulus(dn)]);
    let tw = twiddles.as_ref();
    let mut inv = vec![0; dn];
    // SAFETY: as the caller guarantees; the inverse's `dn` limbs are in the vector.
    unsafe {
        top_inverse(inv.as_mut_ptr(), d, dn, dn, tw);
        let t_inv = Ready::of(inv.as_ptr(), dn, mq, tw).t;
        let t_d = Ready::of(d, dn, mr, tw).t;
        Prepared {
            inv,
            mq,
            t_inv,
            mr,
            t_d,
        }
    }
}

/// [`mu_div_qr`], or with `rem` false [`mu_div_q`], by the divisor `d` prepared as `pre`, `v` the
/// reciprocal of its top two limbs: with the inverse of all its limbs, blocks of `dn` limbs, whose
/// products by the inverse and by the divisor both skip a transform. Of a quotient that is not a
/// multiple of `dn` limbs, the `r` limbs left over make a last block of their own, which costs as
/// much as the others, but for `r` below a quarter of `dn`, the quotient of `dn + r` limbs by the
/// divisor, which the schoolbook method or divide and conquer computes for less. A quotient
/// shorter than two thirds of the divisor takes one block, and the top limbs of the inverse, which
/// the bounds of [`run`] allow, with a product of its own length instead.
///
/// # Safety
///
/// As for [`mu_div_qr`], with `pre` prepared from the divisor.
#[allow(clippy::too_many_arguments)]
pub(in super::super) unsafe fn mu_div_prepared(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    v: u64,
    pre: &Prepared,
    rem: bool,
) -> bool {
    let qn = nn - dn;
    debug_assert!(pre.inv.len() == dn);
    let d_side = Side {
        x: d,
        n: dn,
        t: pre.t_d.as_ref(),
    };
    // SAFETY: as the caller guarantees: the full blocks divide the dividend's limbs from `r`, and
    // leave their partial remainder in the `dn` limbs from `r`, the top of the last division's.
    unsafe {
        if 3 * qn >= 2 * dn {
            let twiddles = twiddles_for(&[pre.mq, pre.mr]);
            let r = qn % dn;
            let r = if 4 * r < dn { r } else { 0 };
            let blocks = Blocks {
                inn: dn,
                b: (qn - r).div_ceil(dn),
                mq: pre.mq,
                inv: Side {
                    x: pre.inv.as_ptr(),
                    n: dn,
                    t: pre.t_inv.as_ref(),
                },
                mr: pre.mr,
                d: Some(d_side),
                tw: twiddles.as_ref(),
            };
            let qh = run(q.add(r), n.add(r), nn - r, d, dn, &blocks, rem || r > 0);
            if r > 0 {
                let mut t = Vec::<u64>::with_capacity(dc_itch(dn + r, dn));
                let high = div_qr_classic(q, n, dn + r, d, dn, v, t.as_mut_ptr());
                debug_assert!(!high, "a partial remainder is below the divisor");
            }
            return qh;
        }
        // One block: the top `in` limbs of `I`, one more than the quotient's for its guard limb.
        let inn = if rem { qn } else { qn + 1 };
        let mq = Modulus::of(2 * inn, inn, false);
        let twiddles = twiddles_for(&[mq, pre.mr]);
        let tw = twiddles.as_ref();
        let ready_inv = Ready::of(pre.inv.as_ptr().add(dn - inn), inn, mq, tw);
        let blocks = Blocks {
            inn,
            b: 1,
            mq,
            inv: ready_inv.side(),
            mr: pre.mr,
            d: Some(d_side),
            tw,
        };
        run(q, n, nn, d, dn, &blocks, rem)
    }
}

/// The inverse and the products of the blocks of [`run`].
struct Blocks<'a> {
    /// The limbs `in` of the estimates.
    inn: usize,
    /// The number of blocks.
    b: usize,
    /// The products of the estimates, and `I_in`, the top `in` limbs of the inverse.
    mq: Modulus,
    inv: Side<'a>,
    /// The products of the remainders, and the divisor made for them when it is shared.
    mr: Modulus,
    d: Option<Side<'a>>,
    /// The twiddles of the products, if any are shared.
    tw: Option<&'a Twiddles>,
}

/// The blocks of a division by Newton's method, as [`mu_div_qr`], or with `rem` false
/// [`mu_div_q`], computes them, with the inverse and the products of `blocks`.
///
/// The top quotient limb comes from a comparison, which leaves the top `dn` limbs of the dividend
/// below `D`. Then each block of `k <= in` quotient limbs, from the top, divides the window `W` of
/// the partial remainder, below `D`, and the next `k` limbs of the dividend, so `W < D B^k`.
///
/// The inverse is `X = B^ni + I` for `D' = D_t + 1`, `D_t` the top `ni >= in` limbs of `D`, with
/// `D' X <= B^(2 ni) <= D' (X + 2)`, and the estimate takes its top limbs,
/// `X_in = floor(X / B^(ni - in)) = B^in + I_in`. With `W' = W B^(in - k)` (`dn + in` limbs,
/// `W' < D B^in`), `W_t` its top `in` limbs, which are those of `W`, and `s = dn - ni`, the estimate
/// `Q_e = floor(W_t X_in / B^in)` is at most `floor(W' / D)` and at most 9 below it:
///
/// - `X_in <= B^(ni + in) / D'`, so
///   `Q_e <= W_t B^ni / D' = W_t B^dn / (D' B^s) <= W' / D`, as `W_t B^dn <= W'` and `D < D' B^s`.
/// - `X_in > (B^(2 ni) / D' - 2) / B^(ni - in) - 1 >= B^(ni + in) / D' - 3`, so
///   `Q_e > W_t X_in / B^in - 1 > W_t B^ni / D' - 4`, as `W_t < B^in`; and
///   `W' / D < (W_t + 1) B^dn / (D_t B^s) = (W_t + 1) B^ni / D_t`, so
///   `W' / D - Q_e < W_t B^ni / (D_t D') + B^ni / D_t + 4`, below `4 + 2 + 4`, as `W_t < B^ni`
///   and `D' > D_t >= B^ni / 2`.
///
/// The block's quotient `Q = floor(W / D) = floor(floor(W' / D) / B^(in - k))` is then the top
/// `k` limbs of `Q_e`, at most 9 too small for `k = in`, and at most 1 for `k < in`. The partial
/// remainder `W - Q D` is then from 0 to below `10 D < B^(dn + 1) - 1`, and is the value from 0 to
/// `B^m - 2` congruent to it modulo `B^m - 1` for `m >= dn + 1` (all ones stands for 0): `W`
/// folded, less the product modulo `B^m - 1`. `D` is subtracted while it is not below `D`.
///
/// For the quotient alone, the last block is shorter than `in`, when it can be: the `in - k` low
/// limbs of `Q_e`, `g`, are a guard, and when `g <= B^(in - k) - 10`, adding up to 9 cannot carry
/// into the top limbs, which are then the quotient, with no remainder to compute. Otherwise, a
/// chance of about `10 / 2^64` for random operands but usual for exact multiples, the remainder is
/// computed after all.
///
/// # Safety
///
/// As for [`mu_div_qr`], with `blocks` made for the divisor: `in <= dn`, `b` blocks of `in` limbs
/// covering the quotient, the last of at least one.
unsafe fn run(
    q: *mut u64,
    n: *mut u64,
    nn: usize,
    d: *const u64,
    dn: usize,
    blocks: &Blocks<'_>,
    rem: bool,
) -> bool {
    let qn = nn - dn;
    let (inn, b, mq, mr, tw) = (blocks.inn, blocks.b, blocks.mq, blocks.mr, blocks.tw);
    debug_assert!(inn <= dn && (b - 1) * inn < qn && qn <= b * inn);
    // SAFETY: as the caller guarantees: every window lies in the `nn` limbs at `n`, every block
    // of the quotient in the `qn` at `q`, and the products in the vectors.
    unsafe {
        let top = n.add(qn);
        let qh = cmp_n(top, d, dn) != Ordering::Less;
        if qh {
            sub_n(top, top, d, dn);
        }
        let mut p = Vec::<u64>::with_capacity(mq.limbs().max(mr.limbs()));
        let p = p.as_mut_ptr();
        let mut w = Vec::<u64>::with_capacity(mr.limbs());
        let w = w.as_mut_ptr();
        let mut fresh = None;
        let mut pos = qn;
        for block in 0..b {
            let k = if block + 1 < b {
                inn
            } else {
                qn - (b - 1) * inn
            };
            pos -= k;
            let wt = n.add(pos + dn + k - inn);
            // `Q_e = W_t + floor(W_t I_in / B^in)` in the top `in` limbs of the product.
            mul_mod(p, mq, wt, inn, blocks.inv, tw);
            let qe = p.add(inn);
            let carry = add_n(qe, qe, wt, inn);
            debug_assert!(!carry, "the estimate is below B^in");
            ptr::copy_nonoverlapping(qe.add(inn - k), q.add(pos), k);
            if !rem && block + 1 == b && k < inn {
                let g = inn - k;
                if !(all_ones(qe.add(1), g - 1) && *qe > u64::MAX - 9) {
                    break;
                }
            }
            let side = match blocks.d {
                Some(side) => side,
                None => fresh.get_or_insert_with(|| Ready::of(d, dn, mr, tw)).side(),
            };
            remainder(n.add(pos), dn, k, q.add(pos), d, mr, side, w, p, tw);
        }
        debug_assert!(pos == 0 || !rem);
        qh
    }
}

/// The partial remainder of a block of [`run`]: `W - Q D` for the window `W` of `dn + k` limbs at
/// `wp` and the estimate `Q` of `k` limbs at `qk`, at most 9 too small, computed modulo `B^m - 1`
/// and corrected; sets the low `dn` limbs at `wp` to it, below `D`, and adds the corrections to `Q`.
///
/// # Safety
///
/// `m >= dn + 1`, `k <= dn`, `side` the divisor's limbs made for `m`, `W < D B^k`, `Q` at most
/// `floor(W / D)` and at most 9 below; the limbs at `wp` valid for reads and writes, those at `qk`
/// too, the `m` limbs at `w` and at `p` for writes, all apart.
#[allow(clippy::too_many_arguments)]
unsafe fn remainder(
    wp: *mut u64,
    dn: usize,
    k: usize,
    qk: *mut u64,
    d: *const u64,
    m: Modulus,
    side: Side<'_>,
    w: *mut u64,
    p: *mut u64,
    tw: Option<&Twiddles>,
) {
    let mm = m.limbs();
    debug_assert!(mm > dn);
    // SAFETY: as the caller guarantees.
    unsafe {
        mul_mod(p, m, qk, k, side, tw);
        let wn = dn + k;
        if wn <= mm {
            ptr::copy_nonoverlapping(wp, w, wn);
            ptr::write_bytes(w.add(wn), 0, mm - wn);
        } else {
            ptr::copy_nonoverlapping(wp, w, mm);
            add_mod(w, mm, wp.add(mm), wn - mm, 0);
        }
        if sub_n(w, w, p, mm) {
            sub_1(w, mm, 1);
        }
        if all_ones(w, mm) {
            ptr::write_bytes(w, 0, mm);
        }
        debug_assert!((dn + 1..mm).all(|i| *w.add(i) == 0) && *w.add(dn) < 10);
        let mut top = *w.add(dn);
        while top != 0 || cmp_n(w, d, dn) != Ordering::Less {
            top -= u64::from(sub_n(w, w, d, dn));
            let out = add_1(qk, k, 1);
            debug_assert!(out == 0, "the block's quotient is below B^k");
        }
        ptr::copy_nonoverlapping(w, wp, dn);
    }
}
