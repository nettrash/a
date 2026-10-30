//! The Baillie–PSW test on an odd number `n` with no small factor: a strong (Miller–Rabin) test to
//! base 2, then a strong Lucas test with Selfridge's parameters, both in Montgomery's form.
//!
//! - The strong test to base 2 writes `n - 1 = d 2^s`, `d` odd, and computes `2^d mod n` from the
//!   top bit of `d`, a square for each bit and a doubling (an addition) for each one, as GMP's
//!   `mpn_2powm` does; then up to `s - 1` squares. `n` passes when `2^d = 1` or one of
//!   `2^(d 2^r) = -1`, `r < s`.
//! - The strong Lucas test takes the first `D` of `5, -7, 9, -11, 13, ...` with the Jacobi symbol
//!   `(D/n) = -1` (Selfridge's method A), `P = 1` and `Q = (1 - D) / 4`, writes `n + 1 = d 2^s`, and
//!   computes `U_d` and `U_{d-1}` of the Lucas sequence from the top bit of `d` with three squares a
//!   bit, as GMP's `mpz_lucas_mod` does, after Hackman: from `U_k` and `U_{k+1}`,
//!   `U_{2k} = U_{k+1}^2 - (U_{k+1} - U_k)^2` and `U_{2k+1} = U_{k+1}^2 - Q U_k^2`, and for a one
//!   bit `U_{2k+2} = U_{2k+1} - Q U_{2k}`. `n` passes when `U_d = 0`, or `V_d = U_d - 2Q U_{d-1}` is
//!   0, or one of `V_{d 2^r} = V_{d 2^(r-1)}^2 - 2Q^(d 2^(r-1))`, `r < s`, with
//!   `Q^d = (V_d^2 - D U_d^2) / 4`. For `D = 5`, `Q = -1`, the ladder of `V_k` and `V_{k+1}` takes
//!   one square and one product a bit instead ([`strong_lucas`]): measured on an Apple M4 Max, primes
//!   of 256 bits took 11 % less time with it, of 512 bits 13 % and of 1024 bits 9 %.
//!
//! The residues are in registers up to seven limbs ([`Field`]) and in limbs in memory beyond
//! ([`LimbField`]), always fully reduced, below `n`, so that equal residues are equal limbs.

use super::super::div::limbs::{div_2by1, invert_limb};
use super::super::mul::limbs::{add_n, cmp_n, sub_n};
use super::super::pow::montgomery::{Mont1, Mont2, MontN, Montgomery, even, rem, rem_itch};
use super::super::pow::window::{Registers, Ring};
use core::cmp::Ordering;
use core::ptr;

/// The arithmetic of residues modulo an odd `n` held in registers, fully reduced.
pub(super) trait Field: Registers {
    /// The residue of zero.
    fn zero(&self) -> Self::V;

    /// The residue of 1, `R mod n`.
    fn one(&self) -> Self::V;

    /// `a + b mod n`.
    fn add(&self, a: Self::V, b: Self::V) -> Self::V;

    /// `a - b mod n`.
    fn sub(&self, a: Self::V, b: Self::V) -> Self::V;

    /// `a / 2 mod n`: `a` halved, or `a + n` when `a` is odd.
    fn half(&self, a: Self::V) -> Self::V;

    /// Whether `a` and `b` are the same residue.
    fn same(&self, a: Self::V, b: Self::V) -> bool;

    /// The residue of the plain number `c`, below `n`.
    fn residue(&self, c: u64) -> Self::V;
}

impl Field for Mont1 {
    #[inline(always)]
    fn zero(&self) -> u64 {
        0
    }

    #[inline(always)]
    fn one(&self) -> u64 {
        // `B mod n = (B - n) mod n`.
        self.m.wrapping_neg() % self.m
    }

    #[inline(always)]
    fn add(&self, a: u64, b: u64) -> u64 {
        let (s, carry) = a.overflowing_add(b);
        if carry || s >= self.m {
            s.wrapping_sub(self.m)
        } else {
            s
        }
    }

    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        let (d, borrow) = a.overflowing_sub(b);
        if borrow { d.wrapping_add(self.m) } else { d }
    }

    #[inline(always)]
    fn half(&self, a: u64) -> u64 {
        // `(a + n) / 2` for an odd `a`, without the carry out of `a + n`.
        if a & 1 == 1 {
            (a >> 1) + (self.m >> 1) + 1
        } else {
            a >> 1
        }
    }

    #[inline(always)]
    fn same(&self, a: u64, b: u64) -> bool {
        a == b
    }

    #[inline]
    fn residue(&self, c: u64) -> u64 {
        ((u128::from(c) << 64) % u128::from(self.m)) as u64
    }
}

/// `a + b - n` or `a + b` for the `N`-limb residues `a` and `b`, below `n`: the sum reduced.
#[inline(always)]
fn add_mod<const N: usize>(a: [u64; N], b: [u64; N], m: &[u64; N]) -> [u64; N] {
    let mut s = [0; N];
    let mut carry = false;
    for j in 0..N {
        let (x, c1) = a[j].overflowing_add(b[j]);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        s[j] = x;
        carry = c1 | c2;
    }
    let mut d = [0; N];
    let mut borrow = false;
    for j in 0..N {
        let (x, b1) = s[j].overflowing_sub(m[j]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        d[j] = x;
        borrow = b1 | b2;
    }
    if carry || !borrow { d } else { s }
}

/// `a - b mod n` for the `N`-limb residues `a` and `b`, below `n`.
#[inline(always)]
fn sub_mod<const N: usize>(a: [u64; N], b: [u64; N], m: &[u64; N]) -> [u64; N] {
    let mut d = [0; N];
    let mut borrow = false;
    for j in 0..N {
        let (x, b1) = a[j].overflowing_sub(b[j]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        d[j] = x;
        borrow = b1 | b2;
    }
    if borrow {
        let mut carry = false;
        for j in 0..N {
            let (x, c1) = d[j].overflowing_add(m[j]);
            let (x, c2) = x.overflowing_add(u64::from(carry));
            d[j] = x;
            carry = c1 | c2;
        }
    }
    d
}

/// `a / 2 mod n` for the `N`-limb residue `a`, below `n`.
#[inline(always)]
fn half_mod<const N: usize>(a: [u64; N], m: &[u64; N]) -> [u64; N] {
    let mut s = a;
    let mut carry = false;
    if a[0] & 1 == 1 {
        for j in 0..N {
            let (x, c1) = a[j].overflowing_add(m[j]);
            let (x, c2) = x.overflowing_add(u64::from(carry));
            s[j] = x;
            carry = c1 | c2;
        }
    }
    let mut h = [0; N];
    for j in 0..N {
        let above = if j + 1 < N {
            s[j + 1]
        } else {
            u64::from(carry)
        };
        h[j] = (s[j] >> 1) | (above << 63);
    }
    h
}

/// The residue `c R mod n` of the plain `c`, for the `N`-limb `n`, by a division.
#[inline]
fn residue_of<const N: usize>(c: u64, m: &[u64; N]) -> [u64; N] {
    let mut r = [0; N];
    let mut s = [0u64; 64];
    // SAFETY: the dividend `c B^N` of `N + 1 <= 8` limbs by a divisor of `N <= 7` takes
    // `rem_itch(N + 1, N) <= 8 + 2 * 10` limbs of scratch, which `s` holds.
    unsafe { rem(r.as_mut_ptr(), &c, 1, N, m.as_ptr(), N, s.as_mut_ptr()) };
    r
}

impl Field for Mont2 {
    #[inline(always)]
    fn zero(&self) -> [u64; 2] {
        [0; 2]
    }

    #[inline(always)]
    fn one(&self) -> [u64; 2] {
        residue_of(1, &self.m)
    }

    #[inline(always)]
    fn add(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        add_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        sub_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn half(&self, a: [u64; 2]) -> [u64; 2] {
        half_mod(a, &self.m)
    }

    #[inline(always)]
    fn same(&self, a: [u64; 2], b: [u64; 2]) -> bool {
        a == b
    }

    #[inline]
    fn residue(&self, c: u64) -> [u64; 2] {
        residue_of(c, &self.m)
    }
}

impl<const N: usize> Field for MontN<N> {
    #[inline(always)]
    fn zero(&self) -> [u64; N] {
        [0; N]
    }

    #[inline(always)]
    fn one(&self) -> [u64; N] {
        residue_of(1, &self.m)
    }

    #[inline(always)]
    fn add(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        add_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        sub_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn half(&self, a: [u64; N]) -> [u64; N] {
        half_mod(a, &self.m)
    }

    #[inline(always)]
    fn same(&self, a: [u64; N], b: [u64; N]) -> bool {
        a == b
    }

    #[inline]
    fn residue(&self, c: u64) -> [u64; N] {
        residue_of(c, &self.m)
    }
}

/// The bits of `n`, the least significant limb first, for the tests' loops.
#[derive(Clone, Copy)]
pub(super) struct Bits {
    /// The limbs.
    pub(super) n: *const u64,
    /// The number of bits, the top one set.
    pub(super) len: usize,
}

impl Bits {
    /// Bit `i`, counted from 0.
    ///
    /// # Safety
    ///
    /// `i < len`, and the limbs valid for reads.
    #[inline(always)]
    unsafe fn get(&self, i: usize) -> bool {
        // SAFETY: as the caller guarantees.
        unsafe { (*self.n.add(i / 64) >> (i % 64)) & 1 == 1 }
    }

    /// The number of zero bits of `n - 1` at its low end, `n` odd and above 1: the position of the
    /// lowest one of `n` above bit 0.
    ///
    /// # Safety
    ///
    /// The limbs must be valid for reads.
    #[inline]
    unsafe fn zeros_below_minus_one(&self) -> usize {
        let mut i = 0;
        // SAFETY: `n - 1` is not zero, so a limb of it that is not zero comes before the end.
        unsafe {
            loop {
                let l = *self.n.add(i) & if i == 0 { !1 } else { !0 };
                if l != 0 {
                    return 64 * i + l.trailing_zeros() as usize;
                }
                i += 1;
            }
        }
    }

    /// The number of one bits of `n` at its low end: the zero bits of `n + 1` at its low end.
    ///
    /// # Safety
    ///
    /// The limbs must be valid for reads.
    #[inline]
    pub(super) unsafe fn ones_at_bottom(&self) -> usize {
        let mut i = 0;
        // SAFETY: the bits run out at the top limb, which has a zero above its top bit, or past the
        // top bit, which `len` bounds.
        unsafe {
            loop {
                let l = !*self.n.add(i);
                if l != 0 || 64 * (i + 1) >= self.len {
                    return (64 * i + l.trailing_zeros() as usize).min(self.len);
                }
                i += 1;
            }
        }
    }
}

/// `x Q mod n` for the residue `x` and a small `Q`, by additions for `|Q| <= 4`, otherwise by a
/// product with the residue `q` of `|Q|`, negated for a negative `Q`.
#[inline(always)]
fn times_q<F: Field>(f: &F, x: F::V, big_q: i64, q: F::V) -> F::V {
    let y = match big_q.unsigned_abs() {
        1 => x,
        2 => f.add(x, x),
        3 => f.add(f.add(x, x), x),
        4 => {
            let y = f.add(x, x);
            f.add(y, y)
        }
        _ => f.mul(x, q),
    };
    if big_q < 0 { f.sub(f.zero(), y) } else { y }
}

/// The strong test to base 2 in the field `f` modulo the odd `n` of the bits `bits`, above 2.
///
/// # Safety
///
/// The bits' limbs must be valid for reads.
#[inline(always)]
pub(super) unsafe fn strong_base_2<F: Field>(f: &F, bits: Bits) -> bool {
    // SAFETY: every bit read is below `bits.len`.
    unsafe {
        let s = bits.zeros_below_minus_one();
        let one = f.one();
        let minus_one = f.sub(f.zero(), one);
        // `2^d` for `d = (n - 1) / 2^s`: the bits of `n` from `s` up, the top one first, which makes
        // the power 2.
        let mut x = f.add(one, one);
        let mut i = bits.len - 1;
        while i > s {
            i -= 1;
            x = f.sqr(x);
            if bits.get(i) {
                x = f.add(x, x);
            }
        }
        if f.same(x, one) || f.same(x, minus_one) {
            return true;
        }
        for _ in 1..s {
            x = f.sqr(x);
            if f.same(x, minus_one) {
                return true;
            }
            if f.same(x, one) {
                return false;
            }
        }
        false
    }
}

/// The strong Lucas test in the field `f` modulo the odd `n` of the bits `bits`, at least 5 and not
/// a square, with Selfridge's `D`, for which `(D/n) = -1`: `P = 1` and `Q = (1 - D) / 4`.
///
/// # Safety
///
/// The bits' limbs must be valid for reads.
#[inline(always)]
pub(super) unsafe fn strong_lucas<F: Field>(f: &F, bits: Bits, big_d: i64) -> bool {
    if big_d == 5 {
        // SAFETY: forwarded.
        return unsafe { strong_lucas_5(f, bits) };
    }
    let big_q = (1 - big_d) / 4;
    // SAFETY: every bit read is below `bits.len`.
    unsafe {
        let b0 = bits.ones_at_bottom();
        let q = f.residue(big_q.unsigned_abs());
        let one = f.one();
        let (mut v, mut qk);
        if bits.len - 2 < b0 {
            // `n = 2^b0 - 1`, `d = 1`: `U_1 = 1`, `V_1 = P = 1`, `Q^1 = Q`.
            v = one;
            qk = if big_q < 0 { f.sub(f.zero(), q) } else { q };
        } else {
            // `U_k` and `U_{k+1}` for `k` the bits of `n` from the top down to bit `b0`, then
            // `k = d - 1`: `n + 1 = d 2^b0` agrees with `n` above bit `b0`, where `n` has a zero.
            let (mut u0, mut u1) = (one, one);
            let mut i = bits.len - 2;
            loop {
                let t1 = f.sqr(u1);
                let t2 = f.sqr(f.sub(u1, u0));
                let t3 = f.sqr(u0);
                let even = f.sub(t1, t2);
                let odd = f.sub(t1, times_q(f, t3, big_q, q));
                if bits.get(i) {
                    (u0, u1) = (odd, f.sub(odd, times_q(f, even, big_q, q)));
                } else {
                    (u0, u1) = (even, odd);
                }
                if i == b0 {
                    break;
                }
                i -= 1;
            }
            // `U_d` and `U_{d-1}`.
            if f.same(u1, f.zero()) {
                return true;
            }
            let w = times_q(f, u0, big_q, q);
            v = f.sub(u1, f.add(w, w));
            if f.same(v, f.zero()) {
                return true;
            }
            if b0 == 1 {
                return false;
            }
            // `Q^d = (V_d^2 - D U_d^2) / 4`.
            let du2 = f.mul(f.sqr(u1), f.residue(big_d.unsigned_abs()));
            let du2 = if big_d < 0 { f.sub(f.zero(), du2) } else { du2 };
            qk = f.half(f.half(f.sub(f.sqr(v), du2)));
        }
        // `V_{d 2^r}` for `r` from 1 to `b0 - 1`.
        for r in 1..b0 {
            v = f.sub(f.sqr(v), f.add(qk, qk));
            if f.same(v, f.zero()) {
                return true;
            }
            if r + 1 < b0 {
                qk = f.sqr(qk);
            }
        }
        false
    }
}

/// The strong Lucas test for `D = 5`, `P = 1`, `Q = -1`, which half of all numbers take: as `Q^k`
/// is `1` or `-1` by the parity of `k`, the ladder of `V_k` and `V_{k+1}` needs one square and one
/// product a bit, `V_{2k} = V_k^2 - 2Q^k` and `V_{2k+1} = V_k V_{k+1} - Q^k` (and `V_{2k+2} =
/// V_{k+1}^2 - 2Q^{k+1}` for a one bit), against three squares for any other `Q`, as GMP's
/// `mpn_strongfibo` takes the Fibonacci numbers. `U_d` is then 0 exactly when `2 V_{d+1} = V_d`,
/// as `D U_d = 2 V_{d+1} - V_d` and 5 is prime to `n`.
///
/// # Safety
///
/// As for [`strong_lucas`], with `(5/n) = -1`.
#[inline(always)]
unsafe fn strong_lucas_5<F: Field>(f: &F, bits: Bits) -> bool {
    // SAFETY: every bit read is below `bits.len`.
    unsafe {
        let b0 = bits.ones_at_bottom();
        let one = f.one();
        let two = f.add(one, one);
        // `V_1 = 1` and `V_2 = 3`, then the bits of `d = (n + 1) / 2^b0` below its top one: those
        // of `n` down to bit `b0 + 1`, and a one; none when `n = 2^b0 - 1`, `d = 1`.
        let (mut v0, mut v1) = (one, f.add(two, one));
        // Whether `k` is odd, so that `Q^k = -1`.
        let mut odd = true;
        if bits.len > b0 {
            let mut i = bits.len - 1;
            loop {
                let bit = if i > b0 + 1 {
                    i -= 1;
                    bits.get(i)
                } else {
                    i = 0;
                    true
                };
                // `V_k V_{k+1} - Q^k`, and `V_k^2 - 2Q^k` or `V_{k+1}^2 - 2Q^{k+1}`.
                let m = f.mul(v0, v1);
                let m = if odd { f.add(m, one) } else { f.sub(m, one) };
                if bit {
                    let s = f.sqr(v1);
                    (v0, v1) = (m, if odd { f.sub(s, two) } else { f.add(s, two) });
                } else {
                    let s = f.sqr(v0);
                    (v0, v1) = (if odd { f.add(s, two) } else { f.sub(s, two) }, m);
                }
                odd = bit;
                if i == 0 {
                    break;
                }
            }
        }
        // `U_d = 0`, or `V_d = 0`.
        if f.same(f.add(v1, v1), v0) || f.same(v0, f.zero()) {
            return true;
        }
        // `V_{2d} = V_d^2 + 2`, as `Q^d = -1` for an odd `d`; then `V_{2k} = V_k^2 - 2`.
        for r in 1..b0 {
            let s = f.sqr(v0);
            v0 = if r == 1 { f.add(s, two) } else { f.sub(s, two) };
            if f.same(v0, f.zero()) {
                return true;
            }
        }
        false
    }
}

/// Both tests in the field `f`, the Lucas test with Selfridge's `D`, which `select` finds, or which
/// tells that `n` is composite.
///
/// # Safety
///
/// The bits' limbs must be valid for reads; `n` odd, at least 5 and without the factors `select`
/// relies on being absent.
#[inline(always)]
pub(super) unsafe fn bpsw<F: Field>(
    f: &F,
    bits: Bits,
    select: impl FnOnce() -> Option<i64>,
) -> bool {
    // SAFETY: forwarded.
    unsafe {
        if !strong_base_2(f, bits) {
            return false;
        }
        match select() {
            Some(d) => strong_lucas(f, bits, d),
            None => false,
        }
    }
}

// Limbs in memory

/// The arithmetic of residues modulo an odd `n` of `nn` limbs in memory, fully reduced, below `n`:
/// Montgomery's products and squares followed by one subtraction of `n` when they are not below
/// it, which suffices as REDC of a product of two residues below `n` is below `2n`.
pub(super) struct LimbField {
    /// The Montgomery arithmetic.
    mont: Montgomery,
    /// The scratch of its products.
    t: *mut u64,
}

impl LimbField {
    /// The limbs of working memory that a field of `nn` limbs needs, for `Montgomery::new` and the
    /// scratch.
    pub(super) fn itch(nn: usize) -> usize {
        Montgomery::new_itch(nn) + Montgomery::itch(nn).max(rem_itch(nn + 1, nn))
    }

    /// The field modulo the odd `nn` limbs at `n`, in the working memory at `s`.
    ///
    /// # Safety
    ///
    /// `nn >= 3`; the limbs of `n` and the [`LimbField::itch`] at `s` must stay valid while the
    /// field is used, `s` start on 16 bytes.
    pub(super) unsafe fn new(n: *const u64, nn: usize, s: *mut u64) -> Self {
        // SAFETY: as the caller guarantees.
        unsafe {
            let mont = Montgomery::new(n, nn, s);
            LimbField {
                mont,
                t: s.add(Montgomery::new_itch(nn)),
            }
        }
    }

    /// The limbs of a residue.
    #[inline(always)]
    fn nn(&self) -> usize {
        self.mont.n
    }

    /// `r = a^2`.
    ///
    /// # Safety
    ///
    /// The residues must be valid; `r` may be `a`.
    #[inline(always)]
    pub(super) unsafe fn sqr(&self, r: *mut u64, a: *const u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            self.mont.sqr(r, a, self.t);
            self.mont.reduce(r);
        }
    }

    /// `r = a b`.
    ///
    /// # Safety
    ///
    /// The residues must be valid; `r` may be `a` or `b`.
    #[inline(always)]
    pub(super) unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            self.mont.mul(r, a, b, self.t);
            self.mont.reduce(r);
        }
    }

    /// `r = a + b`.
    ///
    /// # Safety
    ///
    /// The residues must be valid; `r` may be `a` or `b`.
    #[inline(always)]
    pub(super) unsafe fn add(&self, r: *mut u64, a: *const u64, b: *const u64) {
        let nn = self.nn();
        // SAFETY: as the caller guarantees.
        unsafe {
            if add_n(r, a, b, nn) || cmp_n(r, self.mont.m, nn) != Ordering::Less {
                sub_n(r, r, self.mont.m, nn);
            }
        }
    }

    /// `r = a - b`.
    ///
    /// # Safety
    ///
    /// The residues must be valid; `r` may be `a` or `b`.
    #[inline(always)]
    pub(super) unsafe fn sub(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            if sub_n(r, a, b, self.nn()) {
                add_n(r, r, self.mont.m, self.nn());
            }
        }
    }

    /// `r = -a`.
    ///
    /// # Safety
    ///
    /// The residues must be valid; `r` may be `a`.
    #[inline(always)]
    unsafe fn neg(&self, r: *mut u64, a: *const u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            if is_zero(a, self.nn()) {
                ptr::copy(a, r, self.nn());
            } else {
                sub_n(r, self.mont.m, a, self.nn());
            }
        }
    }

    /// `r = a / 2`.
    ///
    /// # Safety
    ///
    /// The residues must be valid; `r` may be `a`.
    unsafe fn half(&self, r: *mut u64, a: *const u64) {
        let nn = self.nn();
        // SAFETY: as the caller guarantees.
        unsafe {
            let carry = if *a & 1 == 1 {
                add_n(r, a, self.mont.m, nn)
            } else {
                ptr::copy(a, r, nn);
                false
            };
            for j in 0..nn {
                let above = if j + 1 < nn {
                    *r.add(j + 1)
                } else {
                    u64::from(carry)
                };
                *r.add(j) = (*r.add(j) >> 1) | (above << 63);
            }
        }
    }

    /// The residue of the plain `c` at `r`.
    ///
    /// # Safety
    ///
    /// The residue must be valid.
    unsafe fn residue(&self, r: *mut u64, c: u64) {
        // SAFETY: the scratch holds `rem_itch(nn + 1, nn)` limbs.
        unsafe { rem(r, &c, 1, self.nn(), self.mont.m, self.nn(), self.t) }
    }

    /// `r = x Q`, for the residue `q` of `|Q|`, by additions for `|Q| <= 4`.
    ///
    /// # Safety
    ///
    /// The residues must be valid, `r` apart from `x`.
    unsafe fn times_q(&self, r: *mut u64, x: *const u64, big_q: i64, q: *const u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            match big_q.unsigned_abs() {
                1 => ptr::copy_nonoverlapping(x, r, self.nn()),
                2 => self.add(r, x, x),
                3 => {
                    self.add(r, x, x);
                    self.add(r, r, x);
                }
                4 => {
                    self.add(r, x, x);
                    self.add(r, r, r);
                }
                _ => self.mul(r, x, q),
            }
            if big_q < 0 {
                self.neg(r, r);
            }
        }
    }
}

/// Whether the `n` limbs at `a` are all zero.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline(always)]
unsafe fn is_zero(a: *const u64, n: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { (0..n).all(|i| *a.add(i) == 0) }
}

/// Whether the `n` limbs at `a` and `b` are equal.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline(always)]
unsafe fn equal(a: *const u64, b: *const u64, n: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { cmp_n(a, b, n) == Ordering::Equal }
}

/// The limbs of working memory of [`bpsw_limbs`] besides the field's: ten residues.
pub(super) fn bpsw_limbs_itch(nn: usize) -> usize {
    LimbField::itch(nn) + 10 * even(nn)
}

/// [`bpsw`] for the limbs in memory: the field `f` and ten residues at `s`.
///
/// # Safety
///
/// As for [`bpsw`], with `nn >= 3`, the field `f` modulo `n`, and `s` valid for ten residues.
pub(super) unsafe fn bpsw_limbs(
    f: &LimbField,
    bits: Bits,
    s: *mut u64,
    select: impl FnOnce() -> Option<i64>,
) -> bool {
    let nn = f.nn();
    let stride = even(nn);
    // SAFETY: every residue lies in the working memory at `s`, apart from the others.
    unsafe {
        let slot = |i: usize| s.add(i * stride);
        let (one, minus_one, x) = (slot(0), slot(1), slot(2));
        // The strong test to base 2.
        f.residue(one, 1);
        f.neg(minus_one, one);
        let r = bits.zeros_below_minus_one();
        f.add(x, one, one);
        let mut i = bits.len - 1;
        while i > r {
            i -= 1;
            f.sqr(x, x);
            if bits.get(i) {
                f.add(x, x, x);
            }
        }
        if !(equal(x, one, nn) || equal(x, minus_one, nn)) {
            let mut passed = false;
            for _ in 1..r {
                f.sqr(x, x);
                if equal(x, minus_one, nn) {
                    passed = true;
                    break;
                }
                if equal(x, one, nn) {
                    return false;
                }
            }
            if !passed {
                return false;
            }
        }
        let Some(big_d) = select() else {
            return false;
        };
        let b0 = bits.ones_at_bottom();
        if big_d == 5 {
            // As in `strong_lucas_5`.
            let (two, v0, v1, m, t) = (slot(1), slot(2), slot(3), slot(4), slot(5));
            f.add(two, one, one);
            ptr::copy_nonoverlapping(one, v0, nn);
            f.add(v1, two, one);
            let mut odd = true;
            if bits.len > b0 {
                let mut i = bits.len - 1;
                loop {
                    let bit = if i > b0 + 1 {
                        i -= 1;
                        bits.get(i)
                    } else {
                        i = 0;
                        true
                    };
                    f.mul(m, v0, v1);
                    if odd {
                        f.add(m, m, one);
                    } else {
                        f.sub(m, m, one);
                    }
                    if bit {
                        f.sqr(t, v1);
                        ptr::copy_nonoverlapping(m, v0, nn);
                        if odd {
                            f.sub(v1, t, two);
                        } else {
                            f.add(v1, t, two);
                        }
                    } else {
                        f.sqr(t, v0);
                        ptr::copy_nonoverlapping(m, v1, nn);
                        if odd {
                            f.add(v0, t, two);
                        } else {
                            f.sub(v0, t, two);
                        }
                    }
                    odd = bit;
                    if i == 0 {
                        break;
                    }
                }
            }
            f.add(t, v1, v1);
            if equal(t, v0, nn) || is_zero(v0, nn) {
                return true;
            }
            for r in 1..b0 {
                f.sqr(v0, v0);
                if r == 1 {
                    f.add(v0, v0, two);
                } else {
                    f.sub(v0, v0, two);
                }
                if is_zero(v0, nn) {
                    return true;
                }
            }
            return false;
        }
        // The strong Lucas test, as in `strong_lucas`.
        let big_q = (1 - big_d) / 4;
        let (q, u0, u1, t1, t2, t3, v, qk) = (
            slot(1),
            slot(2),
            slot(3),
            slot(4),
            slot(5),
            slot(6),
            slot(7),
            slot(8),
        );
        f.residue(q, big_q.unsigned_abs());
        if bits.len - 2 < b0 {
            ptr::copy_nonoverlapping(one, v, nn);
            ptr::copy_nonoverlapping(q, qk, nn);
            if big_q < 0 {
                f.neg(qk, qk);
            }
        } else {
            ptr::copy_nonoverlapping(one, u0, nn);
            ptr::copy_nonoverlapping(one, u1, nn);
            let mut i = bits.len - 2;
            loop {
                f.sqr(t1, u1);
                f.sub(t2, u1, u0);
                f.sqr(t2, t2);
                f.sqr(t3, u0);
                // `U_{2k}` in `t2`, `U_{2k+1}` in `u1`.
                f.sub(t2, t1, t2);
                f.times_q(u0, t3, big_q, q);
                f.sub(u1, t1, u0);
                if bits.get(i) {
                    ptr::copy_nonoverlapping(u1, u0, nn);
                    f.times_q(t3, t2, big_q, q);
                    f.sub(u1, u1, t3);
                } else {
                    ptr::copy_nonoverlapping(t2, u0, nn);
                }
                if i == b0 {
                    break;
                }
                i -= 1;
            }
            if is_zero(u1, nn) {
                return true;
            }
            f.times_q(t1, u0, big_q, q);
            f.add(t1, t1, t1);
            f.sub(v, u1, t1);
            if is_zero(v, nn) {
                return true;
            }
            if b0 == 1 {
                return false;
            }
            f.sqr(t1, u1);
            f.residue(t2, big_d.unsigned_abs());
            f.mul(t1, t1, t2);
            if big_d < 0 {
                f.neg(t1, t1);
            }
            f.sqr(t2, v);
            f.sub(qk, t2, t1);
            f.half(qk, qk);
            f.half(qk, qk);
        }
        for r in 1..b0 {
            f.sqr(v, v);
            f.add(t1, qk, qk);
            f.sub(v, v, t1);
            if is_zero(v, nn) {
                return true;
            }
            if r + 1 < b0 {
                f.sqr(qk, qk);
            }
        }
        false
    }
}

/// The remainder of the `nn` limbs at `n` by the normalized `d` with its reciprocal `v`, which is a
/// remainder by any divisor of `d` too: limb by limb from the top.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline]
pub(super) unsafe fn rem_by(n: *const u64, nn: usize, d: u64, v: u64) -> u64 {
    let mut r = 0;
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in (0..nn).rev() {
            r = div_2by1(r, *n.add(i), d, v).1;
        }
    }
    r
}

/// The reciprocal of the normalized `d`, for [`rem_by`].
#[inline]
pub(super) fn reciprocal(d: u64) -> u64 {
    invert_limb(d)
}
