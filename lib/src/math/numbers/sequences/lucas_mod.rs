//! Lucas sequences modulo a number: `U_n(P, Q)` and `V_n(P, Q)` modulo `m`, for an index of any
//! size.
//!
//! The modulus is split as `m = 2^z m'` with `m'` odd. Modulo `m'` the residues are in Montgomery's
//! form, the arithmetic of the Baillie–PSW test ([`Field`] in registers up to seven limbs,
//! [`LimbField`] in limbs beyond); modulo `2^z` they are the low limbs of products ([`Low2`],
//! [`LowN`], [`Low`]). Both results are put together by the Chinese remainder theorem, as the
//! powers modulo an even number are.
//!
//! The index's bits are taken from the top, by one of four ladders:
//!
//! - [`u_ladder`], for any `P` and `Q`: `U_k` and `U_(k+1)`, with `U_(2k) = U_k (2 U_(k+1) - P U_k)`,
//!   `U_(2k+1) = U_(k+1)^2 - Q U_k^2` and `U_(2k+2) = U_(k+1) (P U_(k+1) - 2 Q U_k)`: a product and
//!   two squares a bit, and the products by `P` and `Q`, by additions when they are small; then
//!   `V_n = 2 U_(n+1) - P U_n`. The only one modulo `2^z`, and the one modulo `m'` when the
//!   others do not apply.
//! - [`uv_ladder`], for `Q = ±1`, a small `P` and a short index modulo an odd number:
//!   `U_(2k) = U_k V_k` and `V_(2k) = V_k^2 - 2 Q^k`, a product and a square a bit, and for a one
//!   bit `U_(2k+1) = (P U_(2k) + V_(2k)) / 2` and `V_(2k+1) = (D U_(2k) + P V_(2k)) / 2`,
//!   `D = P^2 - 4Q`, by additions and halvings.
//! - [`v_ladder`], for `Q = ±1` and any `P` modulo an odd number: `V_k` and `V_(k+1)`, with
//!   `V_(2k) = V_k^2 - 2 Q^k`, `V_(2k+1) = V_k V_(k+1) - P Q^k` and
//!   `V_(2k+2) = V_(k+1)^2 - 2 Q^(k+1)`, a product and a square a bit as Montgomery's ladder for
//!   the method `p + 1` takes them; then `U_n = (2 V_(n+1) - P V_n) / D`, when `D` is prime to
//!   the modulus, and otherwise the first ladder.
//! - [`v_ladder_q`], for a long `Q` modulo three limbs and more: the same with `Q^k` beside it,
//!   `Q^(2k) = (Q^k)^2` and `Q^(2k+1) = Q^k Q^(k+1)`, 3.5 to 4.5 products a bit against the first
//!   ladder's 4.5 to 5.5. Measured on an Apple M4 Max with prime moduli and indices of their
//!   length, 20 % less time modulo 256 bits and 448; modulo 64 bits as long, and modulo 128
//!   longer, where the inverse of `D` costs a sixth of the ladder.
//!
//! Moduli of eight limbs and more take the first, third and fourth ladders in limbs
//! ([`memory_ladder`], [`memory_v_ladder`]).

use super::super::div::limbs::{bdiv_q, binvert_limb};
use super::super::mul::limbs::{add_1, add_n, mul, mul_itch, sub_1, sub_n};
use super::super::mul::{Operand, STACK_LIMBS, on_heap, on_stack, to_limbs};
use super::super::pow::low::{Low, Low2, LowN, mullo, mullo_itch};
use super::super::pow::montgomery::{Mont1, Mont2, MontN, even, rem, rem_itch};
use super::super::pow::window::{Registers, Ring};
use super::super::prime::bpsw::{Field, LimbField};
use super::super::root::{limbs_number, rshift, small_number};
use super::super::scratch::{Scratch, need, scratch};
use super::super::{ByteOrder, Number};
use core::mem::MaybeUninit;
use core::ptr;

/// Panics for a modulus that is zero or negative.
#[cold]
#[inline(never)]
#[track_caller]
fn bad_modulus() -> ! {
    panic!("the modulus of a Lucas sequence must be positive")
}

/// Panics for a negative index.
#[cold]
#[inline(never)]
#[track_caller]
fn negative_index() -> ! {
    panic!("the index of a Lucas sequence must not be negative")
}

/// An integer given as its limbs, the least significant first, none for zero, and its sign.
#[derive(Clone, Copy)]
struct Int {
    /// The limbs.
    limbs: *const u64,
    /// Their number, without zero limbs at the top.
    n: usize,
    /// Whether the integer is negative; never for zero.
    negative: bool,
}

impl Int {
    /// The integer when it is at most `SMALL` in absolute value, which the ladders multiply by
    /// with additions.
    ///
    /// # Safety
    ///
    /// The limbs must be valid for reads.
    #[inline]
    unsafe fn small(&self) -> Option<i64> {
        match self.n {
            0 => Some(0),
            // SAFETY: as the caller guarantees.
            1 if unsafe { *self.limbs } <= SMALL => {
                let x = unsafe { *self.limbs } as i64;
                Some(if self.negative { -x } else { x })
            }
            _ => None,
        }
    }
}

/// The largest multiplier taken by additions: up to two doublings and an addition, which take less
/// time than a product in Montgomery's form even in registers.
const SMALL: u64 = 4;

/// The bits of the index, the least significant limb first.
#[derive(Clone, Copy)]
struct Index {
    /// The limbs.
    limbs: *const u64,
    /// The number of bits, the top one set; at least 1.
    bits: u64,
}

impl Index {
    /// Bit `i`.
    ///
    /// # Safety
    ///
    /// `i < bits`, and the limbs valid for reads.
    #[inline(always)]
    unsafe fn bit(&self, i: u64) -> bool {
        // SAFETY: as the caller guarantees; the limb's index is below the limbs' number, a
        // `usize`.
        unsafe { (*self.limbs.add((i / 64) as usize) >> (i % 64)) & 1 == 1 }
    }
}

// The ladders in registers

/// The arithmetic of residues held in registers, modulo an odd number in Montgomery's form or
/// modulo `B^N`.
trait Arith: Registers {
    /// The limbs of a residue.
    const LIMBS: usize;

    /// 0.
    fn zero(&self) -> Self::V;

    /// 1.
    fn one(&self) -> Self::V;

    /// `a + b`.
    fn add(&self, a: Self::V, b: Self::V) -> Self::V;

    /// `a - b`.
    fn sub(&self, a: Self::V, b: Self::V) -> Self::V;

    /// The residue of the integer `x`, with the scratch `s` of [`rem_itch`]`(x.n + LIMBS, LIMBS)`
    /// limbs.
    ///
    /// # Safety
    ///
    /// The integer's limbs must be valid for reads.
    unsafe fn of(&self, x: Int, s: Scratch) -> Self::V;

    /// Writes the `LIMBS` limbs of the plain residue of `a` at `r`, out of Montgomery's form.
    ///
    /// # Safety
    ///
    /// `r` must be valid for writing them.
    unsafe fn store(&self, a: Self::V, r: *mut u64);

    /// Writes the `LIMBS` limbs of `a` at `r` as they are.
    ///
    /// # Safety
    ///
    /// `r` must be valid for writing them.
    unsafe fn store_raw(&self, a: Self::V, r: *mut u64);
}

/// The arithmetic of residues modulo an odd number in Montgomery's form, which also halves and
/// inverts.
trait OddArith: Arith {
    /// `a / 2`.
    fn half(&self, a: Self::V) -> Self::V;

    /// Whether `a` and `b` are the same residue.
    fn same(&self, a: Self::V, b: Self::V) -> bool;

    /// The plain inverse of the plain residue of `a`, which a product by a residue in Montgomery's
    /// form takes out of it: `None` when it shares a factor with the modulus.
    fn inverse(&self, a: Self::V) -> Option<Self::V>;
}

/// `x c` for a small `c` (`|c| <= SMALL`), by additions.
#[inline(always)]
fn small_times<R: Arith>(r: &R, x: R::V, c: i64) -> R::V {
    let y = match c.unsigned_abs() {
        0 => r.zero(),
        1 => x,
        2 => r.add(x, x),
        3 => r.add(r.add(x, x), x),
        _ => {
            let y = r.add(x, x);
            r.add(y, y)
        }
    };
    if c < 0 { r.sub(r.zero(), y) } else { y }
}

/// A multiplier: a small integer, by additions, or a residue, by a product.
#[derive(Clone, Copy)]
enum Times<V> {
    /// At most [`SMALL`] in absolute value.
    Small(i64),
    /// The residue.
    By(V),
}

/// `x c`.
#[inline(always)]
fn times<R: Arith>(r: &R, x: R::V, c: Times<R::V>) -> R::V {
    match c {
        Times::Small(c) => small_times(r, x, c),
        Times::By(y) => r.mul(x, y),
    }
}

/// `(U_n, U_(n+1))` for `n >= 1`, the residue `p` of `P` and the multipliers of `P` and `Q`.
///
/// The bits are taken by a branch: measured on an Apple M4 Max with random indices, the same steps
/// with both halves computed and the results selected without a branch took 20 to 35 % longer
/// modulo numbers of 64 bits.
///
/// # Safety
///
/// The index's limbs must be valid for reads.
#[inline(always)]
unsafe fn u_ladder<R: Arith>(
    r: &R,
    p: R::V,
    tp: Times<R::V>,
    tq: Times<R::V>,
    n: Index,
) -> (R::V, R::V) {
    // `k = 1`: `U_1 = 1`, `U_2 = P`.
    let (mut u0, mut u1) = (r.one(), p);
    let mut i = n.bits - 1;
    while i > 0 {
        i -= 1;
        let s0 = r.sqr(u0);
        let s1 = r.sqr(u1);
        // `U_(2k+1) = U_(k+1)^2 - Q U_k^2`.
        let odd = r.sub(s1, times(r, s0, tq));
        // SAFETY: `i < bits`.
        if unsafe { n.bit(i) } {
            // `U_(2k+2) = U_(k+1) (P U_(k+1) - 2 Q U_k)`.
            let qu = times(r, u0, tq);
            let t = r.sub(times(r, u1, tp), r.add(qu, qu));
            (u0, u1) = (odd, r.mul(u1, t));
        } else {
            // `U_(2k) = U_k (2 U_(k+1) - P U_k)`.
            let t = r.sub(r.add(u1, u1), times(r, u0, tp));
            (u0, u1) = (r.mul(u0, t), odd);
        }
    }
    (u0, u1)
}

/// `(U_n, V_n)` for `n >= 1`, `Q = 1` or, when `q_negative`, `Q = -1`, the multiplier of `P` and
/// that of `D = P^2 - 4Q`, modulo an odd number.
///
/// # Safety
///
/// The index's limbs must be valid for reads.
#[inline(always)]
unsafe fn uv_ladder<R: OddArith>(
    r: &R,
    p: R::V,
    tp: Times<R::V>,
    td: Times<R::V>,
    q_negative: bool,
    n: Index,
) -> (R::V, R::V) {
    let one = r.one();
    let two = r.add(one, one);
    // `k = 1`: `U_1 = 1`, `V_1 = P`.
    let (mut u, mut v) = (one, p);
    // Whether `Q^k = -1`.
    let mut negative = q_negative;
    let mut i = n.bits - 1;
    while i > 0 {
        i -= 1;
        // `U_(2k) = U_k V_k`, `V_(2k) = V_k^2 - 2 Q^k`, `Q^(2k) = 1`.
        let s = r.sqr(v);
        u = r.mul(u, v);
        v = if negative {
            r.add(s, two)
        } else {
            r.sub(s, two)
        };
        negative = false;
        // SAFETY: `i < bits`.
        if unsafe { n.bit(i) } {
            let pu = times(r, u, tp);
            let du = times(r, u, td);
            let pv = times(r, v, tp);
            u = r.half(r.add(pu, v));
            v = r.half(r.add(du, pv));
            negative = q_negative;
        }
    }
    (u, v)
}

/// `(V_n, V_(n+1))` for `n >= 1`, `Q = 1` or, when `q_negative`, `Q = -1`, and the residue `p` of
/// `P`.
///
/// # Safety
///
/// The index's limbs must be valid for reads.
#[inline(always)]
unsafe fn v_ladder<R: Arith>(r: &R, p: R::V, q_negative: bool, n: Index) -> (R::V, R::V) {
    let one = r.one();
    let two = r.add(one, one);
    // `k = 1`: `V_1 = P`, `V_2 = P^2 - 2Q`.
    let s = r.sqr(p);
    let (mut v0, mut v1) = (
        p,
        if q_negative {
            r.add(s, two)
        } else {
            r.sub(s, two)
        },
    );
    // Whether `Q^k = -1`, `k` odd.
    let mut negative = q_negative;
    let mut i = n.bits - 1;
    while i > 0 {
        i -= 1;
        // `V_(2k+1) = V_k V_(k+1) - P Q^k`.
        let m = r.mul(v0, v1);
        let m = if negative { r.add(m, p) } else { r.sub(m, p) };
        // SAFETY: `i < bits`.
        if unsafe { n.bit(i) } {
            // `V_(2k+2) = V_(k+1)^2 - 2 Q^(k+1)`, `Q^(k+1) = -1` for an even `k`.
            let s = r.sqr(v1);
            let up = q_negative && !negative;
            (v0, v1) = (m, if up { r.add(s, two) } else { r.sub(s, two) });
            negative = q_negative;
        } else {
            // `V_(2k) = V_k^2 - 2 Q^k`.
            let s = r.sqr(v0);
            (v0, v1) = (
                if negative {
                    r.add(s, two)
                } else {
                    r.sub(s, two)
                },
                m,
            );
            negative = false;
        }
    }
    (v0, v1)
}

/// `(V_n, V_(n+1))` for `n >= 1`, the residue `p` of `P`, `q` of `Q`, and their multipliers: the
/// ladder of [`v_ladder`] with `Q^k` kept beside it, `Q^(2k) = (Q^k)^2` and
/// `Q^(2k+1) = Q^k Q^(k+1)`.
///
/// # Safety
///
/// The index's limbs must be valid for reads.
#[inline(always)]
unsafe fn v_ladder_q<R: Arith>(
    r: &R,
    p: R::V,
    tp: Times<R::V>,
    q: R::V,
    tq: Times<R::V>,
    n: Index,
) -> (R::V, R::V) {
    // `k = 1`: `V_1 = P`, `V_2 = P^2 - 2Q`, `Q^1 = Q`.
    let (mut v0, mut v1, mut qk) = (p, r.sub(r.sqr(p), r.add(q, q)), q);
    let mut i = n.bits - 1;
    while i > 0 {
        i -= 1;
        // `V_(2k+1) = V_k V_(k+1) - P Q^k`.
        let m = r.sub(r.mul(v0, v1), times(r, qk, tp));
        // SAFETY: `i < bits`.
        if unsafe { n.bit(i) } {
            // `V_(2k+2) = V_(k+1)^2 - 2 Q^(k+1)`.
            let q1 = times(r, qk, tq);
            let s = r.sub(r.sqr(v1), r.add(q1, q1));
            (v0, v1, qk) = (m, s, r.mul(qk, q1));
        } else {
            // `V_(2k) = V_k^2 - 2 Q^k`.
            let s = r.sub(r.sqr(v0), r.add(qk, qk));
            (v0, v1, qk) = (s, m, r.sqr(qk));
        }
    }
    (v0, v1)
}

/// The index's bits from which [`v_ladder`] and the inverse of `D` take less time than [`uv_ladder`]
/// for a modulus of `limbs` limbs. Measured on an Apple M4 Max with `P = 3`, `Q = 1` and with
/// `P = 1`, `Q = -1`: modulo 64 bits, indices of 32 bits took 1350 and 1220 cycles against 1310 and
/// 1190, and of 64 bits 2150 and 1850 against 2180 and 1980; modulo 128 bits, indices of 128 bits
/// 5540 and 5830 against 7580 and 7420, and modulo 256 bits 28,600 and 28,900 against 37,800 and
/// 35,600, the inverse taking 1100 and 2400 cycles of those.
#[inline]
fn v_ladder_from(limbs: usize) -> u64 {
    match limbs {
        1 => 48,
        2 => 80,
        _ => 96,
    }
}

/// Writes `U_n` and `V_n` modulo the odd modulus of the ring `r` at `u` and `v`, `LIMBS` limbs
/// each, for `n >= 1`.
///
/// # Safety
///
/// The integers' and the index's limbs must be valid for reads, `u` and `v` for writing; the
/// scratch holds [`rem_itch`]`(max(p.n, q.n) + LIMBS, LIMBS)` limbs.
#[inline(always)]
unsafe fn odd_registers<R: OddArith>(
    r: &R,
    p: Int,
    q: Int,
    n: Index,
    u: *mut u64,
    v: *mut u64,
    s: Scratch,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let pr = r.of(p, s);
        let qr = r.of(q, s);
        let (sp, sq) = (p.small(), q.small());
        let tp = sp.map_or(Times::By(pr), Times::Small);
        let tq = sq.map_or(Times::By(qr), Times::Small);
        let one = r.one();
        let q_unit = if r.same(qr, one) {
            Some(false)
        } else if r.same(qr, r.sub(r.zero(), one)) {
            Some(true)
        } else {
            None
        };
        if let Some(q_negative) = q_unit {
            // `D = P^2 - 4Q`.
            let four = small_times(r, one, 4);
            let s = r.sqr(pr);
            let d = if q_negative {
                r.add(s, four)
            } else {
                r.sub(s, four)
            };
            if let Some(c) = sp {
                // `D = c^2 -+ 4` from -4 to 20: by additions up to `SMALL`.
                let dc = c * c + if q_negative { 4 } else { -4 };
                let td = if dc.unsigned_abs() <= SMALL {
                    Times::Small(dc)
                } else {
                    Times::By(d)
                };
                if n.bits < v_ladder_from(R::LIMBS) {
                    let (un, vn) = uv_ladder(r, pr, tp, td, q_negative, n);
                    r.store(un, u);
                    r.store(vn, v);
                    return;
                }
            }
            if let Some(dinv) = r.inverse(d) {
                let (v0, v1) = v_ladder(r, pr, q_negative, n);
                // `U_n = (2 V_(n+1) - P V_n) / D`.
                let w = r.sub(r.add(v1, v1), times(r, v0, tp));
                r.store_raw(r.mul(w, dinv), u);
                r.store(v0, v);
                return;
            }
        }
        if sq.is_none() && R::LIMBS >= 3 {
            // `D = P^2 - 4Q`.
            let d = r.sub(r.sqr(pr), small_times(r, qr, 4));
            if let Some(dinv) = r.inverse(d) {
                let (v0, v1) = v_ladder_q(r, pr, tp, qr, tq, n);
                let w = r.sub(r.add(v1, v1), times(r, v0, tp));
                r.store_raw(r.mul(w, dinv), u);
                r.store(v0, v);
                return;
            }
        }
        let (u0, u1) = u_ladder(r, pr, tp, tq, n);
        r.store(u0, u);
        r.store(r.sub(r.add(u1, u1), times(r, u0, tp)), v);
    }
}

/// Writes `U_n` and `V_n` modulo `B^LIMBS` at `u` and `v`, `LIMBS` limbs each, for `n >= 1`.
///
/// # Safety
///
/// As for [`odd_registers`].
#[inline(always)]
unsafe fn low_registers<R: Arith>(
    r: &R,
    p: Int,
    q: Int,
    n: Index,
    u: *mut u64,
    v: *mut u64,
    s: Scratch,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let pr = r.of(p, s);
        let qr = r.of(q, s);
        let tp = p.small().map_or(Times::By(pr), Times::Small);
        let tq = q.small().map_or(Times::By(qr), Times::Small);
        let (u0, u1) = u_ladder(r, pr, tp, tq, n);
        r.store_raw(u0, u);
        r.store_raw(r.sub(r.add(u1, u1), times(r, u0, tp)), v);
    }
}

impl Arith for Mont1 {
    const LIMBS: usize = 1;

    #[inline(always)]
    fn zero(&self) -> u64 {
        Field::zero(self)
    }

    #[inline(always)]
    fn one(&self) -> u64 {
        Field::one(self)
    }

    #[inline(always)]
    fn add(&self, a: u64, b: u64) -> u64 {
        Field::add(self, a, b)
    }

    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        Field::sub(self, a, b)
    }

    #[inline]
    unsafe fn of(&self, x: Int, s: Scratch) -> u64 {
        if x.n == 0 {
            return 0;
        }
        let mut r = 0;
        // SAFETY: as the caller guarantees: `x B mod m`.
        unsafe { rem(&mut r, x.limbs, x.n, 1, &self.m, 1, s) };
        if x.negative {
            Field::sub(self, 0, r)
        } else {
            r
        }
    }

    #[inline]
    unsafe fn store(&self, a: u64, r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { *r = self.plain(a) };
    }

    #[inline]
    unsafe fn store_raw(&self, a: u64, r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { *r = a };
    }
}

impl OddArith for Mont1 {
    #[inline(always)]
    fn half(&self, a: u64) -> u64 {
        Field::half(self, a)
    }

    #[inline(always)]
    fn same(&self, a: u64, b: u64) -> bool {
        Field::same(self, a, b)
    }

    #[inline]
    fn inverse(&self, a: u64) -> Option<u64> {
        inverse_u64(self.plain(a), self.m)
    }
}

/// The inverse of `a` modulo the odd `m > 1`, or `None` when they share a factor: by the binary
/// method, `x a = u` and `y a = v` modulo `m` kept as the odd `v` and the other are brought down to
/// their greatest common divisor.
fn inverse_u64(a: u64, m: u64) -> Option<u64> {
    let half = |x: u64| {
        if x & 1 == 0 {
            x >> 1
        } else {
            (x >> 1) + (m >> 1) + 1
        }
    };
    let (mut u, mut v) = (a % m, m);
    let (mut x, mut y) = (1u64, 0u64);
    while u != 0 {
        while u & 1 == 0 {
            u >>= 1;
            x = half(x);
        }
        if u < v {
            core::mem::swap(&mut u, &mut v);
            core::mem::swap(&mut x, &mut y);
        }
        u -= v;
        x = if x >= y { x - y } else { x + (m - y) };
    }
    (v == 1).then_some(y)
}

impl Arith for Mont2 {
    const LIMBS: usize = 2;

    #[inline(always)]
    fn zero(&self) -> [u64; 2] {
        Field::zero(self)
    }

    #[inline(always)]
    fn one(&self) -> [u64; 2] {
        Field::one(self)
    }

    #[inline(always)]
    fn add(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Field::add(self, a, b)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Field::sub(self, a, b)
    }

    #[inline]
    unsafe fn of(&self, x: Int, s: Scratch) -> [u64; 2] {
        let mut r = [0; 2];
        if x.n == 0 {
            return r;
        }
        // SAFETY: as the caller guarantees: `x B^2 mod m`.
        unsafe { rem(r.as_mut_ptr(), x.limbs, x.n, 2, self.m.as_ptr(), 2, s) };
        if x.negative {
            Field::sub(self, [0; 2], r)
        } else {
            r
        }
    }

    #[inline]
    unsafe fn store(&self, a: [u64; 2], r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { self.store_raw(self.plain(a), r) };
    }

    #[inline]
    unsafe fn store_raw(&self, a: [u64; 2], r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { ptr::copy_nonoverlapping(a.as_ptr(), r, 2) };
    }
}

impl OddArith for Mont2 {
    #[inline(always)]
    fn half(&self, a: [u64; 2]) -> [u64; 2] {
        Field::half(self, a)
    }

    #[inline(always)]
    fn same(&self, a: [u64; 2], b: [u64; 2]) -> bool {
        Field::same(self, a, b)
    }

    #[inline]
    fn inverse(&self, a: [u64; 2]) -> Option<[u64; 2]> {
        inverse_limbs(&self.plain(a), &self.m)
    }
}

/// The inverse of the `N` limbs `a` modulo the odd `N` limbs `m`, or `None`, by the numbers' gcd.
fn inverse_limbs<const N: usize>(a: &[u64; N], m: &[u64; N]) -> Option<[u64; N]> {
    let number = |x: &[u64; N]| {
        // SAFETY: the array holds the `N` limbs.
        unsafe { limbs_number(x.as_ptr(), N, false, ByteOrder::LittleEndian) }
    };
    let inv = number(a).mod_inverse(&number(m))?;
    let mut r = [0u64; N];
    for (i, b) in inv.value.iter().enumerate() {
        r[i / 8] |= u64::from(*b) << (8 * (i % 8));
    }
    Some(r)
}

impl<const N: usize> Arith for MontN<N> {
    const LIMBS: usize = N;

    #[inline(always)]
    fn zero(&self) -> [u64; N] {
        Field::zero(self)
    }

    #[inline(always)]
    fn one(&self) -> [u64; N] {
        Field::one(self)
    }

    #[inline(always)]
    fn add(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        Field::add(self, a, b)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        Field::sub(self, a, b)
    }

    #[inline]
    unsafe fn of(&self, x: Int, s: Scratch) -> [u64; N] {
        let mut r = [0; N];
        if x.n == 0 {
            return r;
        }
        // SAFETY: as the caller guarantees: `x B^N mod m`.
        unsafe { rem(r.as_mut_ptr(), x.limbs, x.n, N, self.m.as_ptr(), N, s) };
        if x.negative {
            Field::sub(self, [0; N], r)
        } else {
            r
        }
    }

    #[inline]
    unsafe fn store(&self, a: [u64; N], r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { self.store_raw(self.plain(a), r) };
    }

    #[inline]
    unsafe fn store_raw(&self, a: [u64; N], r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { ptr::copy_nonoverlapping(a.as_ptr(), r, N) };
    }
}

impl<const N: usize> OddArith for MontN<N> {
    #[inline(always)]
    fn half(&self, a: [u64; N]) -> [u64; N] {
        Field::half(self, a)
    }

    #[inline(always)]
    fn same(&self, a: [u64; N], b: [u64; N]) -> bool {
        Field::same(self, a, b)
    }

    #[inline]
    fn inverse(&self, a: [u64; N]) -> Option<[u64; N]> {
        inverse_limbs(&self.plain(a), &self.m)
    }
}

impl Arith for Low2 {
    const LIMBS: usize = 2;

    #[inline(always)]
    fn zero(&self) -> u128 {
        0
    }

    #[inline(always)]
    fn one(&self) -> u128 {
        1
    }

    #[inline(always)]
    fn add(&self, a: u128, b: u128) -> u128 {
        a.wrapping_add(b)
    }

    #[inline(always)]
    fn sub(&self, a: u128, b: u128) -> u128 {
        a.wrapping_sub(b)
    }

    #[inline]
    unsafe fn of(&self, x: Int, _: Scratch) -> u128 {
        // SAFETY: as the caller guarantees.
        let y = unsafe {
            match x.n {
                0 => 0,
                1 => u128::from(*x.limbs),
                _ => u128::from(*x.limbs) | u128::from(*x.limbs.add(1)) << 64,
            }
        };
        if x.negative { y.wrapping_neg() } else { y }
    }

    #[inline]
    unsafe fn store(&self, a: u128, r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { self.store_raw(a, r) };
    }

    #[inline]
    unsafe fn store_raw(&self, a: u128, r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            *r = a as u64;
            *r.add(1) = (a >> 64) as u64;
        }
    }
}

impl<const N: usize> Arith for LowN<N> {
    const LIMBS: usize = N;

    #[inline(always)]
    fn zero(&self) -> [u64; N] {
        [0; N]
    }

    #[inline(always)]
    fn one(&self) -> [u64; N] {
        let mut r = [0; N];
        r[0] = 1;
        r
    }

    #[inline(always)]
    fn add(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        let mut s = [0; N];
        let mut carry = false;
        for j in 0..N {
            let (x, c1) = a[j].overflowing_add(b[j]);
            let (x, c2) = x.overflowing_add(u64::from(carry));
            s[j] = x;
            carry = c1 | c2;
        }
        s
    }

    #[inline(always)]
    fn sub(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        let mut d = [0; N];
        let mut borrow = false;
        for j in 0..N {
            let (x, b1) = a[j].overflowing_sub(b[j]);
            let (x, b2) = x.overflowing_sub(u64::from(borrow));
            d[j] = x;
            borrow = b1 | b2;
        }
        d
    }

    #[inline]
    unsafe fn of(&self, x: Int, _: Scratch) -> [u64; N] {
        let mut y = [0; N];
        // SAFETY: as the caller guarantees.
        unsafe { ptr::copy_nonoverlapping(x.limbs, y.as_mut_ptr(), x.n.min(N)) };
        if x.negative { self.sub([0; N], y) } else { y }
    }

    #[inline]
    unsafe fn store(&self, a: [u64; N], r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { self.store_raw(a, r) };
    }

    #[inline]
    unsafe fn store_raw(&self, a: [u64; N], r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe { ptr::copy_nonoverlapping(a.as_ptr(), r, N) };
    }
}

// The ladder in limbs in memory

/// The arithmetic of residues of `n` limbs in memory, modulo an odd number in Montgomery's form or
/// modulo `B^n`; `r` may be any operand.
trait Memory {
    /// The limbs of a residue.
    fn n(&self) -> usize;

    /// `r = a b`.
    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64);

    /// `r = a^2`.
    unsafe fn sqr(&self, r: *mut u64, a: *const u64);

    /// `r = a + b`.
    unsafe fn add(&self, r: *mut u64, a: *const u64, b: *const u64);

    /// `r = a - b`.
    unsafe fn sub(&self, r: *mut u64, a: *const u64, b: *const u64);

    /// `r = 1`.
    unsafe fn one(&self, r: *mut u64);
}

impl Memory for LimbField {
    #[inline(always)]
    fn n(&self) -> usize {
        self.nn()
    }

    #[inline(always)]
    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: forwarded.
        unsafe { LimbField::mul(self, r, a, b) }
    }

    #[inline(always)]
    unsafe fn sqr(&self, r: *mut u64, a: *const u64) {
        // SAFETY: forwarded.
        unsafe { LimbField::sqr(self, r, a) }
    }

    #[inline(always)]
    unsafe fn add(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: forwarded.
        unsafe { LimbField::add(self, r, a, b) }
    }

    #[inline(always)]
    unsafe fn sub(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: forwarded.
        unsafe { LimbField::sub(self, r, a, b) }
    }

    #[inline(always)]
    unsafe fn one(&self, r: *mut u64) {
        // SAFETY: forwarded.
        unsafe { self.residue(r, 1) }
    }
}

/// The ring modulo `B^n` in limbs, with its scratch.
struct LowMemory {
    /// The ring.
    low: Low,
    /// The scratch of its products, [`Low::itch`] limbs.
    t: Scratch,
}

impl Memory for LowMemory {
    #[inline(always)]
    fn n(&self) -> usize {
        self.low.n
    }

    #[inline(always)]
    unsafe fn mul(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: forwarded.
        unsafe { Ring::mul(&self.low, r, a, b, self.t) }
    }

    #[inline(always)]
    unsafe fn sqr(&self, r: *mut u64, a: *const u64) {
        // SAFETY: forwarded.
        unsafe { Ring::sqr(&self.low, r, a, self.t) }
    }

    #[inline(always)]
    unsafe fn add(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: forwarded.
        unsafe { add_n(r, a, b, self.low.n) };
    }

    #[inline(always)]
    unsafe fn sub(&self, r: *mut u64, a: *const u64, b: *const u64) {
        // SAFETY: forwarded.
        unsafe { sub_n(r, a, b, self.low.n) };
    }

    #[inline(always)]
    unsafe fn one(&self, r: *mut u64) {
        // SAFETY: as the caller guarantees.
        unsafe {
            ptr::write_bytes(r, 0, self.low.n);
            *r = 1;
        }
    }
}

/// The residues of [`memory_ladder`] and [`memory_v_ladder`] besides those of `P` and `Q`: seven
/// for the ladders, and five for `U_n` from `V_n`.
const MEMORY_SLOTS: usize = 12;

/// `r = x c` in limbs, for the multiplier `c`: `small` when it is small, and otherwise the residue
/// at `c`; `zero` is the residue 0.
///
/// # Safety
///
/// The residues must be valid, `r` apart from `x` and `zero`.
#[inline(always)]
unsafe fn memory_times<R: Memory>(
    f: &R,
    r: *mut u64,
    x: *const u64,
    small: Option<i64>,
    c: *const u64,
    zero: *const u64,
) {
    let n = f.n();
    // SAFETY: as the caller guarantees.
    unsafe {
        match small {
            None => f.mul(r, x, c),
            Some(c) => {
                match c.unsigned_abs() {
                    0 => ptr::write_bytes(r, 0, n),
                    1 => ptr::copy_nonoverlapping(x, r, n),
                    2 => f.add(r, x, x),
                    3 => {
                        f.add(r, x, x);
                        f.add(r, r, x);
                    }
                    _ => {
                        f.add(r, x, x);
                        f.add(r, r, r);
                    }
                }
                if c < 0 {
                    f.sub(r, zero, r);
                }
            }
        }
    }
}

/// `U_n` and `V_n` by [`u_ladder`] in limbs, written at `u` and `v`, from the residues of `P` at
/// `p` and of `Q` at `q`, for `n >= 1`, with [`MEMORY_SLOTS`] residues at `s`.
///
/// # Safety
///
/// The residues and the index's limbs must be valid, apart; `u` and `v` hold a residue each.
#[allow(clippy::too_many_arguments)]
unsafe fn memory_ladder<R: Memory>(
    f: &R,
    p: *const u64,
    sp: Option<i64>,
    q: *const u64,
    sq: Option<i64>,
    n: Index,
    u: *mut u64,
    v: *mut u64,
    s: *mut u64,
) {
    let nn = f.n();
    let stride = even(nn);
    // SAFETY: as the caller guarantees: every residue in a slot of its own.
    unsafe {
        let (u0, u1, s0, s1, t, w, zero) = (
            s,
            s.add(stride),
            s.add(2 * stride),
            s.add(3 * stride),
            s.add(4 * stride),
            s.add(5 * stride),
            s.add(6 * stride),
        );
        ptr::write_bytes(zero, 0, nn);
        f.one(u0);
        ptr::copy_nonoverlapping(p, u1, nn);
        let mut i = n.bits - 1;
        while i > 0 {
            i -= 1;
            f.sqr(s0, u0);
            f.sqr(s1, u1);
            // `U_(2k+1) = U_(k+1)^2 - Q U_k^2`, in `s1`.
            memory_times(f, t, s0, sq, q, zero);
            f.sub(s1, s1, t);
            if n.bit(i) {
                // `U_(2k+2) = U_(k+1) (P U_(k+1) - 2 Q U_k)`.
                memory_times(f, t, u0, sq, q, zero);
                f.add(t, t, t);
                memory_times(f, w, u1, sp, p, zero);
                f.sub(w, w, t);
                f.mul(t, u1, w);
                ptr::copy_nonoverlapping(s1, u0, nn);
                ptr::copy_nonoverlapping(t, u1, nn);
            } else {
                // `U_(2k) = U_k (2 U_(k+1) - P U_k)`.
                memory_times(f, t, u0, sp, p, zero);
                f.add(w, u1, u1);
                f.sub(w, w, t);
                f.mul(t, u0, w);
                ptr::copy_nonoverlapping(t, u0, nn);
                ptr::copy_nonoverlapping(s1, u1, nn);
            }
        }
        ptr::copy_nonoverlapping(u0, u, nn);
        // `V_n = 2 U_(n+1) - P U_n`.
        memory_times(f, t, u0, sp, p, zero);
        f.add(w, u1, u1);
        f.sub(v, w, t);
    }
}

/// `(V_n, V_(n+1))` by [`v_ladder`] in limbs, or by [`v_ladder_q`] when `unit` is `None`, written
/// at `v0` and `v1`, from the residues of `P` at `p` and of `Q` at `q`, for `n >= 1`; `unit` is
/// whether `Q = -1` when `Q = ±1`. The slots at `s` hold [`MEMORY_SLOTS`] residues.
///
/// # Safety
///
/// As for [`memory_ladder`].
#[allow(clippy::too_many_arguments)]
unsafe fn memory_v_ladder<R: Memory>(
    f: &R,
    p: *const u64,
    sp: Option<i64>,
    q: *const u64,
    sq: Option<i64>,
    unit: Option<bool>,
    n: Index,
    v0: *mut u64,
    v1: *mut u64,
    s: *mut u64,
) {
    let nn = f.n();
    let stride = even(nn);
    // SAFETY: as the caller guarantees: every residue in a slot of its own.
    unsafe {
        let slot = |i: usize| s.add(i * stride);
        let (two, qk, m, t, q1, zero) = (slot(0), slot(1), slot(2), slot(3), slot(4), slot(5));
        ptr::write_bytes(zero, 0, nn);
        f.one(t);
        f.add(two, t, t);
        // `k = 1`: `V_1 = P`, `V_2 = P^2 - 2Q`, `Q^1 = Q`.
        ptr::copy_nonoverlapping(p, v0, nn);
        f.sqr(v1, p);
        f.add(t, q, q);
        f.sub(v1, v1, t);
        ptr::copy_nonoverlapping(q, qk, nn);
        let q_negative = unit == Some(true);
        // Whether `Q^k = -1`, for `Q = ±1`.
        let mut negative = q_negative;
        let mut i = n.bits - 1;
        while i > 0 {
            i -= 1;
            // `V_(2k+1) = V_k V_(k+1) - P Q^k`.
            f.mul(m, v0, v1);
            if unit.is_some() {
                if negative {
                    f.add(m, m, p);
                } else {
                    f.sub(m, m, p);
                }
            } else {
                memory_times(f, t, qk, sp, p, zero);
                f.sub(m, m, t);
            }
            if n.bit(i) {
                // `V_(2k+2) = V_(k+1)^2 - 2 Q^(k+1)`, `Q^(2k+1) = Q^k Q^(k+1)`.
                f.sqr(v1, v1);
                if unit.is_some() {
                    if q_negative && !negative {
                        f.add(v1, v1, two);
                    } else {
                        f.sub(v1, v1, two);
                    }
                    negative = q_negative;
                } else {
                    memory_times(f, q1, qk, sq, q, zero);
                    f.add(t, q1, q1);
                    f.sub(v1, v1, t);
                    f.mul(qk, qk, q1);
                }
                ptr::copy_nonoverlapping(m, v0, nn);
            } else {
                // `V_(2k) = V_k^2 - 2 Q^k`, `Q^(2k) = (Q^k)^2`.
                f.sqr(v0, v0);
                if unit.is_some() {
                    if negative {
                        f.add(v0, v0, two);
                    } else {
                        f.sub(v0, v0, two);
                    }
                    negative = false;
                } else {
                    f.add(t, qk, qk);
                    f.sub(v0, v0, t);
                    f.sqr(qk, qk);
                }
                ptr::copy_nonoverlapping(m, v1, nn);
            }
        }
    }
}

// The driver

/// The limbs of an operand, read in.
fn limbs_of(x: &Number) -> (Operand, usize) {
    let o = Operand::of(x);
    (o, o.significant.div_ceil(8))
}

/// `(U_n mod m, V_n mod m)`, each in `0..m`, in the byte order of `m`.
pub(in super::super) fn lucas_sequence_mod(
    p: &Number,
    q: &Number,
    n: &Number,
    m: &Number,
) -> (Number, Number) {
    let order = m.byte_order;
    let (mo, mn) = limbs_of(m);
    if mn == 0 || m.negative {
        bad_modulus();
    }
    let (no, nl) = limbs_of(n);
    if n.negative && nl != 0 {
        negative_index();
    }
    let (po, pl) = limbs_of(p);
    let (qo, ql) = limbs_of(q);
    // SAFETY: the modulus's vector is valid for reads, and its magnitude not zero.
    let shape = unsafe { shape_of(mo) };
    let need = plan(shape, nl, pl, ql);
    let compute = |s: *mut MaybeUninit<u64>| {
        let s = scratch!(s.cast(), need);
        // SAFETY: the working memory holds the plan's limbs.
        unsafe {
            let (mp, s) = s.split(even(mn));
            let (np, s) = s.split(even(nl));
            let (pp, s) = s.split(even(pl));
            let (qp, s) = s.split(even(ql));
            to_limbs(mo, mp);
            to_limbs(no, np);
            to_limbs(po, pp);
            to_limbs(qo, qp);
            let p = Int {
                limbs: pp,
                n: pl,
                negative: p.negative && pl != 0,
            };
            let q = Int {
                limbs: qp,
                n: ql,
                negative: q.negative && ql != 0,
            };
            in_memory(mp, shape, np, nl, p, q, s, order)
        }
    };
    if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// The shape of the modulus `m = 2^z m'`: its limbs, `z`, the limbs of `m'` and those of `2^z`.
#[derive(Clone, Copy)]
struct Shape {
    /// The limbs of `m`.
    mn: usize,
    /// The zero bits at the bottom of `m`, in 64 bits as on 32-bit targets they need not fit.
    z: u64,
    /// The limbs of `m'`, 0 when `m` is a power of two.
    nodd: usize,
    /// The limbs of `2^z`, `z.div_ceil(64)`.
    kn: usize,
}

/// The shape of the modulus of the operand `m`, not zero.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
unsafe fn shape_of(m: Operand) -> Shape {
    let l = m.significant;
    // Byte `i` from the least significant one.
    // SAFETY: `i` is below the significant length.
    let byte = |i: usize| unsafe {
        if m.big_endian {
            *m.ptr.add(m.len - 1 - i)
        } else {
            *m.ptr.add(i)
        }
    };
    let mut i = 0;
    while byte(i) == 0 {
        i += 1;
    }
    // In 64 bits: on 32-bit targets those of 512 MiB do not fit a `usize`.
    let z = 8 * i as u64 + u64::from(byte(i).trailing_zeros());
    let bits = 8 * l as u64 - u64::from(byte(l - 1).leading_zeros());
    Shape {
        mn: l.div_ceil(8),
        z,
        nodd: if bits == z + 1 {
            0
        } else {
            (bits - z).div_ceil(64) as usize
        },
        kn: z.div_ceil(64) as usize,
    }
}

/// The limbs of working memory of [`lucas_sequence_mod`] for a modulus of the shape `shape`, an
/// index of `nl` limbs and `P` and `Q` of `pl` and `ql`: the operands, the parts' results, then the
/// most that one of the parts or the Chinese remainder theorem needs.
fn plan(shape: Shape, nl: usize, pl: usize, ql: usize) -> usize {
    let Shape { mn, nodd, kn, .. } = shape;
    let inputs = even(mn) + even(nl) + even(pl) + even(ql);
    let parts = 6 * even(mn + 1);
    let pq = pl.max(ql);
    let odd = if nodd == 0 {
        0
    } else if nodd <= 7 {
        rem_itch(pq + nodd, nodd)
    } else {
        ((2 + MEMORY_SLOTS) * even(nodd))
            .saturating_add(LimbField::itch(nodd))
            .saturating_add(rem_itch(pq + nodd, nodd))
    };
    let low = if kn <= 8 {
        0
    } else {
        ((2 + MEMORY_SLOTS) * even(kn)).saturating_add(Low::itch(kn))
    };
    let crt = if nodd == 0 || kn == 0 {
        0
    } else {
        (4 * even(kn)).saturating_add(mullo_itch(kn).max(mul_itch(nodd.max(kn), nodd.min(kn))))
    };
    // Saturated: on 32-bit targets the sums pass `usize::MAX` for moduli of about 200 MB, plans
    // that cannot be met, which the allocation refuses, where wrapped ones would be met and
    // overrun.
    inputs
        .saturating_add(parts)
        .saturating_add(odd.max(low).max(crt))
}

/// [`lucas_sequence_mod`] on the limbs read in: the modulus's `mn` at `mp`, the index's `nl` at
/// `np`, `P` and `Q`, in the working memory at `s`.
///
/// # Safety
///
/// The limbs must be valid, the modulus's top limb not zero, and the working memory hold what
/// [`plan`] counts past the operands.
#[allow(clippy::too_many_arguments)]
unsafe fn in_memory(
    mp: *mut u64,
    shape: Shape,
    np: *const u64,
    nl: usize,
    p: Int,
    q: Int,
    s: Scratch,
    order: ByteOrder,
) -> (Number, Number) {
    need!(
        s,
        plan(shape, nl, p.n, q.n) - even(shape.mn) - even(nl) - even(p.n) - even(q.n)
    );
    // SAFETY: as the caller guarantees; every buffer lies in the working memory.
    unsafe {
        let mn = shape.mn;
        if mn == 1 && *mp == 1 {
            // Everything is 0 modulo 1.
            return (small_number(order, 0, false), small_number(order, 0, false));
        }
        if nl == 0 {
            // `U_0 = 0`, `V_0 = 2 mod m`.
            let two = if mn == 1 && *mp == 2 { 0 } else { 2 };
            return (
                small_number(order, 0, false),
                small_number(order, two, false),
            );
        }
        let index = Index {
            limbs: np,
            bits: 64 * nl as u64 - u64::from((*np.add(nl - 1)).leading_zeros()),
        };
        let Shape { z, nodd, kn, .. } = shape;
        let (odd, s) = s.split(even(mn + 1));
        let (u1, s) = s.split(even(mn + 1));
        let (v1, s) = s.split(even(mn + 1));
        let (u2, s) = s.split(even(mn + 1));
        let (v2, s) = s.split(even(mn + 1));
        let (y, s) = s.split(even(mn + 1));
        if nodd > 0 {
            let (zl, zb) = ((z / 64) as usize, (z % 64) as u32);
            ptr::copy_nonoverlapping(mp.add(zl), odd, mn - zl);
            if zb != 0 {
                rshift(odd, odd, mn - zl, zb);
            }
            odd_part(odd, nodd, p, q, index, u1, v1, s);
        }
        if kn > 0 {
            low_part(kn, z, p, q, index, u2, v2, s);
        }
        if kn == 0 {
            return (number_of(u1, nodd, order), number_of(v1, nodd, order));
        }
        if nodd == 0 {
            return (number_of(u2, kn, order), number_of(v2, kn, order));
        }
        crt(shape, odd, u1, u2, y, s);
        let u = number_of(y, mn, order);
        crt(shape, odd, v1, v2, y, s);
        let v = number_of(y, mn, order);
        (u, v)
    }
}

/// The number of the `n` limbs at `r`, in the byte order `order`.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn number_of(r: *const u64, n: usize, order: ByteOrder) -> Number {
    // SAFETY: as the caller guarantees.
    unsafe { limbs_number(r, n, false, order) }
}

/// Writes `U_n` and `V_n` modulo the odd `m'` of `nodd` limbs at `m`, above 1, at `u` and `v`.
///
/// # Safety
///
/// As for [`in_memory`].
#[allow(clippy::too_many_arguments)]
unsafe fn odd_part(
    m: *const u64,
    nodd: usize,
    p: Int,
    q: Int,
    n: Index,
    u: *mut u64,
    v: *mut u64,
    s: Scratch,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        match nodd {
            1 => odd_registers(&Mont1::new(*m), p, q, n, u, v, s),
            2 => odd_registers(&Mont2::new([*m, *m.add(1)]), p, q, n, u, v, s),
            3 => odd_registers(&MontN::<3>::new(array(m)), p, q, n, u, v, s),
            4 => odd_registers(&MontN::<4>::new(array(m)), p, q, n, u, v, s),
            5 => odd_registers(&MontN::<5>::new(array(m)), p, q, n, u, v, s),
            6 => odd_registers(&MontN::<6>::new(array(m)), p, q, n, u, v, s),
            7 => odd_registers(&MontN::<7>::new(array(m)), p, q, n, u, v, s),
            _ => {
                let stride = even(nodd);
                let (pr, s) = s.split(stride);
                let (qr, s) = s.split(stride);
                let (slots, s) = s.split(MEMORY_SLOTS * stride);
                let itch = LimbField::itch(nodd);
                let f = LimbField::new(m, nodd, s.first(itch));
                let t = s.add(itch);
                to_residue(&f, pr, p, m, nodd, t);
                to_residue(&f, qr, q, m, nodd, t);
                let slot = |i: usize| slots.add(i * stride);
                // A plain 1, by which a product takes a residue out of Montgomery's form.
                let (one, x, d, v0, v1) = (slot(7), slot(8), slot(9), slot(10), slot(11));
                ptr::write_bytes(one, 0, nodd);
                *one = 1;
                // `Q = ±1`?
                f.one(x);
                let unit = if equal(qr, x, nodd) {
                    Some(false)
                } else {
                    f.neg(x, x);
                    equal(qr, x, nodd).then_some(true)
                };
                let (sp, sq) = (p.small(), q.small());
                if unit.is_some() || sq.is_none() {
                    // `D = P^2 - 4Q`, its plain residue and the inverse of that.
                    f.sqr(d, pr);
                    f.add(x, qr, qr);
                    f.add(x, x, x);
                    f.sub(d, d, x);
                    f.mul(d, d, one);
                    if inverse_in(d, m, nodd) {
                        memory_v_ladder(&f, pr, sp, qr, sq, unit, n, v0, v1, slots);
                        // `U_n = (2 V_(n+1) - P V_n) / D`, with a zero in a slot the ladder does
                        // not take.
                        let zero = slot(6);
                        ptr::write_bytes(zero, 0, nodd);
                        memory_times(&f, x, v0, sp, pr, zero);
                        f.add(v1, v1, v1);
                        f.sub(v1, v1, x);
                        f.mul(u, v1, d);
                        f.mul(v, v0, one);
                        return;
                    }
                }
                memory_ladder(&f, pr, sp, qr, sq, n, u, v, slots);
                f.mul(u, u, one);
                f.mul(v, v, one);
            }
        }
    }
}

/// Sets the residue at `r` to that of the integer `x` modulo the `nodd` limbs at `m`, in
/// Montgomery's form: `x B^nodd mod m`, negated for a negative `x`.
///
/// # Safety
///
/// The limbs must be valid; the scratch holds [`rem_itch`]`(x.n + nodd, nodd)` limbs.
unsafe fn to_residue(f: &LimbField, r: *mut u64, x: Int, m: *const u64, nodd: usize, t: Scratch) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if x.n == 0 {
            ptr::write_bytes(r, 0, nodd);
            return;
        }
        rem(r, x.limbs, x.n, nodd, m, nodd, t);
        if x.negative {
            f.neg(r, r);
        }
    }
}

/// Whether the `nodd` limbs at `d` are prime to those of `m`, and then sets them to their inverse
/// modulo `m`, by the numbers' gcd.
///
/// # Safety
///
/// The limbs must be valid.
unsafe fn inverse_in(d: *mut u64, m: *const u64, nodd: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        let x = limbs_number(d, nodd, false, ByteOrder::LittleEndian);
        let y = limbs_number(m, nodd, false, ByteOrder::LittleEndian);
        let Some(inv) = x.mod_inverse(&y) else {
            return false;
        };
        ptr::write_bytes(d, 0, nodd);
        for (i, b) in inv.value.iter().enumerate() {
            *d.add(i / 8) |= u64::from(*b) << (8 * (i % 8));
        }
        true
    }
}

/// Whether the `n` limbs at `a` and `b` are equal.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline]
unsafe fn equal(a: *const u64, b: *const u64, n: usize) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe { (0..n).all(|i| *a.add(i) == *b.add(i)) }
}

/// The `N` limbs at `m`.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline(always)]
unsafe fn array<const N: usize>(m: *const u64) -> [u64; N] {
    let mut a = [0; N];
    // SAFETY: as the caller guarantees.
    unsafe { ptr::copy_nonoverlapping(m, a.as_mut_ptr(), N) };
    a
}

/// Writes `U_n` and `V_n` modulo `2^z` at `u` and `v`, `kn = z.div_ceil(64)` limbs each.
///
/// # Safety
///
/// As for [`in_memory`].
#[allow(clippy::too_many_arguments)]
unsafe fn low_part(
    kn: usize,
    z: u64,
    p: Int,
    q: Int,
    n: Index,
    u: *mut u64,
    v: *mut u64,
    s: Scratch,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        match kn {
            1 | 2 => {
                let mut w = [0u64; 4];
                // One pointer for both halves: a second `as_mut_ptr` would retag `w` and
                // invalidate the first.
                let w = w.as_mut_ptr();
                low_registers(&Low2, p, q, n, w, w.add(2), s);
                ptr::copy_nonoverlapping(w, u, kn);
                ptr::copy_nonoverlapping(w.add(2), v, kn);
            }
            3 => low_registers(&LowN::<3>, p, q, n, u, v, s),
            4 => low_registers(&LowN::<4>, p, q, n, u, v, s),
            5 => low_registers(&LowN::<5>, p, q, n, u, v, s),
            6 => low_registers(&LowN::<6>, p, q, n, u, v, s),
            7 => low_registers(&LowN::<7>, p, q, n, u, v, s),
            8 => low_registers(&LowN::<8>, p, q, n, u, v, s),
            _ => {
                let stride = even(kn);
                let (pr, s) = s.split(stride);
                let (qr, s) = s.split(stride);
                let (slots, s) = s.split(MEMORY_SLOTS * stride);
                let f = LowMemory {
                    low: Low { n: kn },
                    t: s.first(Low::itch(kn)),
                };
                for (r, x) in [(pr, p), (qr, q)] {
                    let have = x.n.min(kn);
                    ptr::copy_nonoverlapping(x.limbs, r, have);
                    ptr::write_bytes(r.add(have), 0, kn - have);
                    if x.negative {
                        // `0 - x`, the zero in the first slot.
                        ptr::write_bytes(slots, 0, kn);
                        f.sub(r, slots, r);
                    }
                }
                memory_ladder(&f, pr, p.small(), qr, q.small(), n, u, v, slots);
            }
        }
        if !z.is_multiple_of(64) {
            let mask = (1u64 << (z % 64)) - 1;
            *u.add(kn - 1) &= mask;
            *v.add(kn - 1) &= mask;
        }
    }
}

/// Sets the `mn` limbs at `y` to the residue modulo `m = 2^z m'` that is `r1` modulo the odd `m'`
/// at `odd` and `r2` modulo `2^z`: `r1 + m' ((r2 - r1) / m' mod 2^z)`, below `m`.
///
/// # Safety
///
/// The limbs must be valid: `r1` has `nodd` limbs, `r2` `kn`, `y` room for `mn + 1`, and the
/// scratch the Chinese remainder theorem's part of [`plan`].
unsafe fn crt(
    shape: Shape,
    odd: *const u64,
    r1: *const u64,
    r2: *const u64,
    y: *mut u64,
    s: Scratch,
) {
    let Shape { mn, z, nodd, kn } = shape;
    // SAFETY: as the caller guarantees.
    unsafe {
        let (inv, s) = s.split(even(kn));
        let (num, s) = s.split(even(kn));
        let (d, s) = s.split(even(kn));
        let (x, rest) = s.split(even(kn));
        ptr::write_bytes(num, 0, kn);
        *num = 1;
        // `1 / m' mod B^kn`.
        bdiv_q(inv, num, kn, odd, nodd, binvert_limb(*odd));
        // `r2 - r1` modulo `B^kn`, `r1` cut or extended to `kn` limbs.
        if nodd >= kn {
            sub_n(d, r2, r1, kn);
        } else {
            let borrow = sub_n(d, r2, r1, nodd);
            ptr::copy_nonoverlapping(r2.add(nodd), d.add(nodd), kn - nodd);
            sub_1(d.add(nodd), kn - nodd, u64::from(borrow));
        }
        mullo(x, d, inv, kn, rest);
        if !z.is_multiple_of(64) {
            *x.add(kn - 1) &= (1 << (z % 64)) - 1;
        }
        if nodd >= kn {
            mul(y, odd, nodd, x, kn, rest);
        } else {
            mul(y, x, kn, odd, nodd, rest);
        }
        // `r1` and the correction, below `m`: the limbs from `mn` up are zero.
        let carry = add_n(y, y, r1, nodd);
        add_1(y.add(nodd), kn, u64::from(carry));
        debug_assert!(*y.add(mn) == 0 || mn == nodd + kn);
    }
}
