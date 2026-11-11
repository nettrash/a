//! Euclid's algorithm on magnitudes for the approximations of rationals
//! ([`crate::math::rational`]): every quotient, which are the terms of a continued fraction
//! ([`quotients`]), and the algorithm stopped at a bound on the remainders or on the cofactors
//! ([`partial`]), which rational reconstruction and the best approximations take.
//!
//! Both are the greatest common divisor's own reductions. [`quotients`] reduces the numbers as
//! [`gcd_dc`](super::hgcd::gcd_dc) and [`gcd_lehmer`](super::limbs::gcd_lehmer) do, the half gcd
//! on the top two thirds of long numbers and then hgcd2's matrices on the top two limbs, and keeps
//! the steps that they tell of (see [`Steps`]): a step not taken in full is followed by the rest of
//! its quotient on the same number, so the quotients of consecutive steps on one number add up to
//! Euclid's. The cost is that of the greatest common divisor, `O(M(n) log n)` for numbers of `n`
//! limbs, and the terms written.
//!
//! [`partial`] keeps a row of the reduction matrix, the cofactors of the second number. Its bound
//! gives a length the numbers can be reduced to without passing it: a remainder above `N` when the
//! numbers keep more limbs than `N`, and cofactors at most `D` when the numbers keep at least `A /
//! D`, for `|t_i| r_(i-1) <= A`. The half gcd takes them there, on top parts sized so that it
//! cannot go below that length, the row multiplied by each matrix, and the last steps are hgcd2's
//! matrices tried against the bound, and single divisions where one would pass it.

use super::super::mul::limbs::cmp_n;
use super::super::mul::{Operand, to_limbs};
use super::super::scratch::{Scratch, scratch};
use super::super::{ByteOrder, Number};
use super::hgcd::{Matrix, adjust_itch, hgcd_itch, hgcd_with, matrix_itch, row_itch, row_times};
use super::kernels::{addmul_1, extend_row, reduce_pair};
use super::limbs::{
    Matrix1, Row, Steps, add_times_itch, hgcd2_hinted_with, normalized, tdiv_itch, tdiv_qr,
    top_limbs, with_scratch,
};
use super::{limbs_number, small_number};
use core::cmp::Ordering;
use core::ptr;

/// From this many limbs, [`quotients`] reduces the numbers by the half gcd on their top two thirds
/// before Lehmer's method, as the greatest common divisor does from
/// [`GCD_DC_THRESHOLD`](super::hgcd::GCD_DC_THRESHOLD).
const CF_DC_THRESHOLD: usize = if cfg!(miri) { 6 } else { 600 };

/// From this many limbs, [`partial`] reduces the numbers by the half gcd while they are longer
/// than its bound allows by [`PARTIAL_DC_GAP`] limbs.
const PARTIAL_DC_THRESHOLD: usize = if cfg!(miri) { 6 } else { 200 };

/// See [`PARTIAL_DC_THRESHOLD`].
const PARTIAL_DC_GAP: usize = if cfg!(miri) { 3 } else { 40 };

/// The magnitude of `n` as limbs, the least significant first, in a vector of at least `room`
/// limbs, zero above, and its number of limbs without the zero limbs at the top.
fn limbs(n: &Number, room: usize) -> (Vec<u64>, usize) {
    let x = Operand::of(n);
    let len = x.significant.div_ceil(8);
    let mut v = vec![0u64; len.max(room)];
    if len > 0 {
        // SAFETY: the vector has room for the operand's limbs.
        unsafe { to_limbs(x, v.as_mut_ptr()) };
    }
    (v, len)
}

/// Working memory of `n` limbs, at least one.
fn work(n: usize) -> Vec<u64> {
    vec![0u64; n.max(1)]
}

/// The limbs of `v`, all of them, as a scratch.
#[inline(always)]
fn whole(v: &mut [u64]) -> Scratch {
    scratch!(v.as_mut_ptr(), v.len())
}

/// The quotients that a reduction tells of, merged into Euclid's and written as numbers.
struct Terms<'a> {
    out: &'a mut Vec<Number>,
    order: ByteOrder,
    /// The column of the quotient being summed, 2 when there is none.
    col: usize,
    /// The quotient being summed, when `big` is empty.
    small: u128,
    /// The quotient being summed when it has more than two limbs.
    big: Vec<u64>,
}

impl<'a> Terms<'a> {
    fn new(out: &'a mut Vec<Number>, order: ByteOrder) -> Terms<'a> {
        Terms {
            out,
            order,
            col: 2,
            small: 0,
            big: Vec::new(),
        }
    }

    /// Writes the quotient being summed.
    fn flush(&mut self) {
        if self.col == 2 {
            return;
        }
        if self.big.is_empty() {
            self.out.push(small_number(self.order, self.small, false));
        } else {
            // SAFETY: the limbs are those of the vector.
            self.out.push(unsafe {
                limbs_number(self.big.as_ptr(), self.big.len(), false, self.order)
            });
            self.big.clear();
        }
        self.small = 0;
        self.col = 2;
    }

    /// Starts a new quotient when the column changes.
    #[inline(always)]
    fn at(&mut self, col: usize) {
        if col != self.col {
            self.flush();
            self.col = col;
        }
    }

    /// Adds `q` to the quotient being summed.
    #[inline(always)]
    fn add_wide(&mut self, col: usize, q: u128) {
        self.at(col);
        if self.big.is_empty() {
            if let Some(s) = self.small.checked_add(q) {
                self.small = s;
                return;
            }
            self.promote();
        }
        add_limbs(&mut self.big, &[q as u64, (q >> 64) as u64]);
    }

    /// Moves the quotient being summed into `big`.
    fn promote(&mut self) {
        self.big.clear();
        self.big
            .extend_from_slice(&[self.small as u64, (self.small >> 64) as u64]);
        self.small = 0;
    }
}

impl Steps for Terms<'_> {
    #[inline(always)]
    fn step(&mut self, col: usize, q: u64) {
        self.add_wide(col, u128::from(q));
    }

    unsafe fn step_limbs(&mut self, col: usize, q: *const u64, qn: usize) {
        // SAFETY: as the caller guarantees.
        let q = unsafe { core::slice::from_raw_parts(q, qn) };
        if qn <= 2 {
            let w = u128::from(q[0]) | (u128::from(*q.get(1).unwrap_or(&0)) << 64);
            self.add_wide(col, w);
            return;
        }
        self.at(col);
        if self.big.is_empty() {
            self.promote();
        }
        add_limbs(&mut self.big, q);
    }
}

/// `x += y` on limbs, `x` growing as the sum needs, without zero limbs at the top.
fn add_limbs(x: &mut Vec<u64>, y: &[u64]) {
    if x.len() < y.len() {
        x.resize(y.len(), 0);
    }
    let mut carry = false;
    for (i, xi) in x.iter_mut().enumerate() {
        let yi = y.get(i).copied().unwrap_or(0);
        if i >= y.len() && !carry {
            break;
        }
        let (s, c1) = xi.overflowing_add(yi);
        let (s, c2) = s.overflowing_add(u64::from(carry));
        *xi = s;
        carry = c1 | c2;
    }
    if carry {
        x.push(1);
    }
    while x.len() > 1 && *x.last().unwrap_or(&1) == 0 {
        x.pop();
    }
}

/// One step of Euclid's `a - q b` or `b - q a` of a quotient of up to two limbs, told to `record`.
#[inline(always)]
fn step_wide<S: Steps>(record: &mut S, col: usize, q: u128) {
    if q >> 64 == 0 {
        record.step(col, q as u64);
    } else {
        let l = [q as u64, (q >> 64) as u64];
        // SAFETY: the two limbs are valid for reads.
        unsafe { record.step_limbs(col, l.as_ptr(), 2) };
    }
}

/// The larger of the numbers of `n` limbs at `a` and `b` and its length, the smaller and its
/// length, and the column of the larger's step: 1 when it is `a`. `a` when they are equal.
///
/// # Safety
///
/// The limbs must be valid for reads.
#[inline(always)]
unsafe fn order_pair(
    a: *mut u64,
    b: *mut u64,
    n: usize,
) -> (*mut u64, usize, *mut u64, usize, usize) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let an = normalized(a, n);
        let bn = normalized(b, n);
        let a_larger = match an.cmp(&bn) {
            Ordering::Equal => cmp_n(a, b, an) != Ordering::Less,
            order => order == Ordering::Greater,
        };
        if a_larger {
            (a, an, b, bn, 1)
        } else {
            (b, bn, a, an, 0)
        }
    }
}

/// One step of Euclid's algorithm on the numbers of `n` limbs at `a` and `b`, the smaller not zero:
/// the larger divided by the smaller, the quotient told to `record` and the remainder in place.
/// Returns the new length, the smaller one's, or `None` when the remainder is zero.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes and apart, and the `avail` limbs
/// at `t` for reads and writes, apart from them; the division's scratch comes from
/// [`with_scratch`] when they are not enough.
unsafe fn divide<S: Steps>(
    a: *mut u64,
    b: *mut u64,
    n: usize,
    record: &mut S,
    t: Scratch,
    avail: usize,
) -> Option<usize> {
    // SAFETY: as the caller guarantees.
    unsafe {
        let (l, ln, s, sn, col) = order_pair(a, b, n);
        debug_assert!(sn > 0);
        let qn = ln - sn + 1;
        let need = qn + tdiv_itch(ln, sn);
        let qlen = with_scratch(t, avail, need, |t| {
            let (q, w) = t.split(qn);
            tdiv_qr(q, l, ln, s, sn, w);
            let qlen = normalized(q, qn);
            record.step_limbs(col, q, qlen);
            qlen
        });
        debug_assert!(qlen > 0);
        ptr::write_bytes(l.add(sn), 0, n - sn);
        if normalized(l, sn) == 0 {
            None
        } else {
            Some(sn)
        }
    }
}

/// The limbs of scratch space that [`divide`] plans for on numbers of `n` limbs; a short divisor
/// may need more, which it takes from [`with_scratch`].
fn divide_itch(n: usize) -> usize {
    (n + 1) + tdiv_itch(n, n).max(tdiv_itch(n, 2))
}

/// Appends to `out` the quotients of Euclid's algorithm on the magnitudes of `a` and `b`, `|a| >
/// |b| > 0`, in the byte order `order`: the terms of the continued fraction of `|a| / |b|`, the
/// last one at least 2 when `|b|` does not divide `|a|`.
pub(crate) fn quotients(a: &Number, b: &Number, order: ByteOrder, out: &mut Vec<Number>) {
    let (mut x, na) = limbs(a, 0);
    let (_, nb) = limbs(b, 0);
    debug_assert!(nb > 0 && na >= nb);
    x.resize(na + 2, 0);
    let (mut y, _) = limbs(b, na + 2);
    let mut terms = Terms::new(out, order);
    let mut t = work(divide_itch(na));
    let avail = t.len();
    // SAFETY: both numbers have `na` limbs and two to spare, and the scratch `avail`.
    unsafe {
        let n = match divide(
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            na,
            &mut terms,
            whole(&mut t),
            avail,
        ) {
            Some(n) => n,
            None => {
                terms.flush();
                return;
            }
        };
        euclid(
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            n,
            &mut terms,
            whole(&mut t),
            avail,
        );
    }
    terms.flush();
}

/// Euclid's algorithm on the numbers of `n` limbs at `a` and `b`, not both zero at the top and
/// neither zero, to the end, every step told to `record`: the half gcd on the top two thirds from
/// [`CF_DC_THRESHOLD`] limbs, hgcd2's matrices down to two limbs, then divisions in registers.
///
/// # Safety
///
/// The `n` limbs at `a` and `b` must be valid for reads and writes, with two to spare, and the
/// `avail` limbs at `t`, at least [`divide_itch`]`(n)`, apart from them.
unsafe fn euclid<S: Steps>(
    a: *mut u64,
    b: *mut u64,
    mut n: usize,
    record: &mut S,
    t: Scratch,
    avail: usize,
) {
    // SAFETY: as the caller guarantees; the numbers keep at least `n` limbs, of which the top ones
    // are not both zero.
    unsafe {
        while n >= CF_DC_THRESHOLD {
            let p = 2 * n / 3;
            let k = n - p;
            let mn = k.div_ceil(2) + 1;
            let need = matrix_itch(k) + hgcd_itch(k).max(adjust_itch(p, mn));
            let mut w = work(need);
            let w = whole(&mut w);
            let mut m = Matrix::new(k, w.take(matrix_itch(k)));
            let rest = w.add(matrix_itch(k));
            let nn = hgcd_with(a.add(p), b.add(p), k, &mut m, rest, record);
            if nn > 0 {
                n = m.adjust(p + nn, a, b, p, rest);
            } else {
                match divide(a, b, n, record, t, avail) {
                    Some(k) => n = k,
                    None => return,
                }
            }
        }
        let mut ones = false;
        while n >= 2 {
            let (ah, al, bh, bl) = top_limbs(a, b, n);
            if let Some(m) = hgcd2_hinted_with(ah, al, bh, bl, &mut ones, record) {
                reduce_pair(a, b, n, m);
                n -= usize::from(*a.add(n - 1) | *b.add(n - 1) == 0);
            } else if n > 2 {
                match divide(a, b, n, record, t, avail) {
                    Some(k) => n = k,
                    None => return,
                }
            } else {
                break;
            }
        }
        let read = |p: *mut u64| {
            if n == 2 {
                u128::from(*p) | (u128::from(*p.add(1)) << 64)
            } else {
                u128::from(*p)
            }
        };
        euclid_u128(read(a), read(b), record);
    }
}

/// Euclid's algorithm on `x` and `y`, neither zero, to the end, every step told to `record`: on two
/// limbs while the smaller has two, then on one.
#[inline]
pub(super) fn euclid_u128<S: Steps>(mut x: u128, mut y: u128, record: &mut S) {
    debug_assert!(x != 0 && y != 0);
    while (x | y) >> 64 != 0 {
        if x >= y {
            let q = x / y;
            x -= q * y;
            step_wide(record, 1, q);
            if x == 0 {
                return;
            }
        } else {
            let q = y / x;
            y -= q * x;
            step_wide(record, 0, q);
            if y == 0 {
                return;
            }
        }
    }
    let (mut x, mut y) = (x as u64, y as u64);
    loop {
        if x >= y {
            let q = x / y;
            x -= q * y;
            record.step(1, q);
            if x == 0 {
                return;
            }
        } else {
            let q = y / x;
            y -= q * x;
            record.step(0, q);
            if y == 0 {
                return;
            }
        }
    }
}

/// The bound of [`partial`].
#[derive(Clone, Copy)]
pub(crate) enum Bound<'a> {
    /// Stop at the first remainder at most this, the second number counting as the first
    /// remainder.
    Remainder(&'a Number),
    /// Stop at the last state whose cofactors are at most this, at least 1, in magnitude.
    Cofactor(&'a Number),
}

/// What [`partial`] leaves: the two numbers, `larger > smaller`, and their cofactors with signs,
/// `x ≡ t_x b (mod a)` for the original `a` and `b`, in fact `x = s_x a + t_x b`.
pub(crate) struct Partial {
    pub(crate) larger: Number,
    pub(crate) smaller: Number,
    pub(crate) t_larger: Number,
    pub(crate) t_smaller: Number,
}

/// Euclid's algorithm on the magnitudes of `a` and `b`, `|a| > |b| >= 0`, stopped at `bound`,
/// with the cofactors of `b` of the two numbers it leaves, all in the byte order `order`.
///
/// With [`Bound::Remainder`]`(N)`, the smaller number is the first of Euclid's remainders `r_i`,
/// `i >= 1`, `r_0 = |a|` and `r_1 = |b|`, that is at most `N`, and the larger `r_(i-1)`. With
/// [`Bound::Cofactor`]`(D)`, the smaller is the last remainder whose cofactor is at most `D`, or
/// zero, and the larger is what is left of the remainder before it, which the next step would
/// divide by the smaller: `r_(i-1)` less some multiple of `r_i`, with a cofactor at most that of
/// `r_(i+1)`.
pub(crate) fn partial(a: &Number, b: &Number, bound: Bound<'_>, order: ByteOrder) -> Partial {
    let (xa, ya) = (Operand::of(a), Operand::of(b));
    let na = xa.significant.div_ceil(8);
    let nb = ya.significant.div_ceil(8);
    debug_assert!(na >= nb && na > 0);
    let (lim, remainder) = match bound {
        Bound::Remainder(v) => (Operand::of(v), true),
        Bound::Cofactor(v) => (Operand::of(v), false),
    };
    let bn = lim.significant.div_ceil(8);
    // The numbers and their copies, the row and its copies, the bound and the divisions' scratch,
    // in one vector.
    let room = na + 3;
    let cap = na + na / 2 + 8;
    let w = divide_itch(na) + add_times_itch(cap - 1, na) + na + 1;
    let mut buf = vec![0u64; 4 * room + 5 * cap + bn + 1 + w];
    let (x, rest) = buf.split_at_mut(room);
    let (y, rest) = rest.split_at_mut(room);
    let (tx, rest) = rest.split_at_mut(room);
    let (ty, rest) = rest.split_at_mut(room);
    let (u0, rest) = rest.split_at_mut(cap);
    let (u1, rest) = rest.split_at_mut(cap);
    let (r0, rest) = rest.split_at_mut(cap);
    let (r1, rest) = rest.split_at_mut(cap);
    let (keep, rest) = rest.split_at_mut(cap);
    let (limit, t) = rest.split_at_mut(bn + 1);
    // SAFETY: each slice has room for its operand's limbs.
    unsafe {
        to_limbs(xa, x.as_mut_ptr());
        if nb > 0 {
            to_limbs(ya, y.as_mut_ptr());
        }
        if bn > 0 {
            to_limbs(lim, limit.as_mut_ptr());
        }
    }
    let limit = &limit[..bn];
    // The row `(u0, u1) = (m00, m01)`: the cofactor of `a`'s slot is `-u1`, of `b`'s `+u0`.
    u0[0] = 1;
    let mut st = State {
        x,
        y,
        tx,
        ty,
        u0,
        u1,
        r0,
        r1,
        keep,
        t,
        n: na,
        un: 1,
    };
    let done = if remainder {
        cmp_limbs(&st.y[..nb], limit) != Ordering::Greater
    } else {
        nb == 0
    };
    if !done {
        // SAFETY: the numbers have `na` limbs and three to spare, the row `cap`, at least
        // `na + mn + 1` for the matrices of the half gcd.
        unsafe { reduce_bounded(&mut st, na, limit, remainder) };
    }
    let (n, un) = (st.n, st.un);
    // SAFETY: the slices hold the limbs read.
    unsafe {
        let (xn, yn) = (normalized(st.x.as_ptr(), n), normalized(st.y.as_ptr(), n));
        let x_larger = cmp_limbs(&st.x[..xn], &st.y[..yn]) == Ordering::Greater;
        let number =
            |v: &[u64], len: usize, negative: bool| limbs_number(v.as_ptr(), len, negative, order);
        let tx = number(st.u1, un, true);
        let ty = number(st.u0, un, false);
        let (nx, ny) = (number(st.x, xn, false), number(st.y, yn, false));
        if x_larger {
            Partial {
                larger: nx,
                smaller: ny,
                t_larger: tx,
                t_smaller: ty,
            }
        } else {
            Partial {
                larger: ny,
                smaller: nx,
                t_larger: ty,
                t_smaller: tx,
            }
        }
    }
}

/// The working state of [`partial`]: the numbers `x` and `y` of `n` limbs, the row `(u0, u1)` of
/// `un`, and room for their copies, a row entry's and the divisions' scratch. Copies change
/// places with what they copy.
struct State<'a> {
    x: &'a mut [u64],
    y: &'a mut [u64],
    tx: &'a mut [u64],
    ty: &'a mut [u64],
    u0: &'a mut [u64],
    u1: &'a mut [u64],
    r0: &'a mut [u64],
    r1: &'a mut [u64],
    keep: &'a mut [u64],
    t: &'a mut [u64],
    n: usize,
    un: usize,
}

/// The order of the magnitudes of the limbs, without zero limbs at the top.
fn cmp_limbs(x: &[u64], y: &[u64]) -> Ordering {
    x.len()
        .cmp(&y.len())
        .then_with(|| x.iter().rev().cmp(y.iter().rev()))
}

/// The order of the `n` limbs at `x`, which may have zero limbs at the top, and the limbs of `y`,
/// which have none.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn cmp_with(x: *const u64, n: usize, y: &[u64]) -> Ordering {
    // SAFETY: as the caller guarantees.
    unsafe {
        let xn = normalized(x, n);
        cmp_limbs(core::slice::from_raw_parts(x, xn), y)
    }
}

/// The reduction of [`partial`] on the state's numbers of `na` limbs and its row, to the bound
/// `limit`, on the remainders when `remainder`.
///
/// # Safety
///
/// The numbers must have `na + 3` limbs, the row's entries `na + na / 2 + 8`, zero above `un`;
/// the smaller number is above `limit` when `remainder`, and not zero.
unsafe fn reduce_bounded(st: &mut State<'_>, na: usize, limit: &[u64], remainder: bool) {
    let mut n = st.n;
    // SAFETY: as the caller guarantees; every kernel stays in the limbs counted.
    unsafe {
        if n >= PARTIAL_DC_THRESHOLD {
            // The length the numbers may be reduced to by the half gcd: more limbs than `N`, or at
            // least those of `A / D`, each with a limb to spare.
            let k = if remainder {
                limit.len() + 1
            } else {
                (na + 2).saturating_sub(limit.len())
            };
            let mut first = true;
            while n >= PARTIAL_DC_THRESHOLD && n > k + PARTIAL_DC_GAP {
                // The half gcd on the limbs from `p` keeps the numbers at least `B^(p + s - 1)`,
                // `s = (n - p) / 2 + 1`, so `p >= 2 k - n` keeps them at least `B^k`.
                let p = (if first { n / 2 } else { n / 3 }).max((2 * k).saturating_sub(n) + 1);
                first = false;
                if n - p < 4 {
                    break;
                }
                let kk = n - p;
                let mn = kk.div_ceil(2) + 1;
                let need = matrix_itch(kk)
                    + hgcd_itch(kk)
                        .max(adjust_itch(p, mn))
                        .max(row_itch(st.un, mn));
                let mut w = work(need);
                let w = whole(&mut w);
                let mut m = Matrix::new(kk, w.take(matrix_itch(kk)));
                let rest = w.add(matrix_itch(kk));
                let (x, y) = (st.x.as_mut_ptr(), st.y.as_mut_ptr());
                let nn = hgcd_with(x.add(p), y.add(p), kk, &mut m, rest, &mut ());
                if nn == 0 {
                    break;
                }
                n = m.adjust(p + nn, x, y, p, rest);
                let mut row = Row {
                    u0: st.u0.as_mut_ptr(),
                    u1: st.u1.as_mut_ptr(),
                    un: st.un,
                };
                row_times(&mut row, &m, rest);
                st.un = row.un;
            }
        }
        st.n = n;
        exact(st, limit, remainder);
    }
}

/// The last steps of [`reduce_bounded`]: hgcd2's matrices while the numbers they leave stay above
/// the bound, or their row at most it; the steps of the one that passes it one at a time; single
/// divisions where hgcd2 takes no step.
///
/// # Safety
///
/// As for [`reduce_bounded`].
unsafe fn exact(st: &mut State<'_>, limit: &[u64], remainder: bool) {
    let avail = st.t.len();
    let mut ones = false;
    let mut record = Record {
        steps: Vec::with_capacity(96),
    };
    // SAFETY: as the caller guarantees.
    unsafe {
        loop {
            let (n, un) = (st.n, st.un);
            if n >= 2 {
                let (ah, al, bh, bl) = top_limbs(st.x.as_ptr(), st.y.as_ptr(), n);
                record.steps.clear();
                if let Some(m) = hgcd2_hinted_with(ah, al, bh, bl, &mut ones, &mut record) {
                    if remainder {
                        st.tx[..n].copy_from_slice(&st.x[..n]);
                        st.ty[..n].copy_from_slice(&st.y[..n]);
                        reduce_pair(st.tx.as_mut_ptr(), st.ty.as_mut_ptr(), n, m);
                        let (_, _, s, sn, _) =
                            order_pair(st.tx.as_mut_ptr(), st.ty.as_mut_ptr(), n);
                        if cmp_with(s, sn, limit) == Ordering::Greater {
                            core::mem::swap(&mut st.x, &mut st.tx);
                            core::mem::swap(&mut st.y, &mut st.ty);
                            let (c0, c1) =
                                extend_row(st.u0.as_mut_ptr(), st.u1.as_mut_ptr(), un, m);
                            st.u0[un] = c0;
                            st.u1[un] = c1;
                            st.un += usize::from(c0 | c1 != 0);
                            st.n -= usize::from(st.x[n - 1] | st.y[n - 1] == 0);
                            continue;
                        }
                    } else {
                        st.r0[..=un].copy_from_slice(&st.u0[..=un]);
                        st.r1[..=un].copy_from_slice(&st.u1[..=un]);
                        let (c0, c1) = extend_row(st.r0.as_mut_ptr(), st.r1.as_mut_ptr(), un, m);
                        st.r0[un] = c0;
                        st.r1[un] = c1;
                        let rn = un + usize::from(c0 | c1 != 0);
                        let big = |v: &[u64]| cmp_with(v.as_ptr(), rn, limit) == Ordering::Greater;
                        if !big(st.r0) && !big(st.r1) {
                            core::mem::swap(&mut st.u0, &mut st.r0);
                            core::mem::swap(&mut st.u1, &mut st.r1);
                            st.un = rn;
                            reduce_pair(st.x.as_mut_ptr(), st.y.as_mut_ptr(), n, m);
                            st.n -= usize::from(st.x[n - 1] | st.y[n - 1] == 0);
                            continue;
                        }
                    }
                    if replay(st, &record, limit, remainder) {
                        return;
                    }
                    continue;
                }
            }
            if divide_bounded(st, limit, remainder, avail) {
                return;
            }
        }
    }
}

/// The steps of a batch of hgcd2's that passes the bound, one at a time from where it started:
/// on copies of the numbers for the remainders, and for the cofactors with the matrix of those
/// taken applied at once. Whether the bound is met, which ends the reduction.
///
/// # Safety
///
/// As for [`reduce_bounded`].
unsafe fn replay(st: &mut State<'_>, record: &Record, limit: &[u64], remainder: bool) -> bool {
    let (n, un) = (st.n, st.un);
    st.r0[..=un].copy_from_slice(&st.u0[..=un]);
    st.r1[..=un].copy_from_slice(&st.u1[..=un]);
    let mut rn = un;
    if remainder {
        st.tx[..n].copy_from_slice(&st.x[..n]);
        st.ty[..n].copy_from_slice(&st.y[..n]);
    }
    let mut m1 = Matrix1 {
        u00: 1,
        u01: 0,
        u10: 0,
        u11: 1,
    };
    let mut stop = false;
    for &(col, q) in &record.steps {
        let (to, from) = if col == 1 {
            (&mut *st.r1, &*st.r0)
        } else {
            (&mut *st.r0, &*st.r1)
        };
        if remainder {
            let (red, other) = if col == 1 {
                (&mut *st.tx, &*st.ty)
            } else {
                (&mut *st.ty, &*st.tx)
            };
            sub_times(red, other, n, q);
            let c = add_times_row(to, from, rn, q);
            rn += usize::from(c != 0);
            if cmp_limbs(top(red, n), limit) != Ordering::Greater {
                stop = true;
                break;
            }
        } else {
            st.keep[..=rn].copy_from_slice(&to[..=rn]);
            let c = add_times_row(to, from, rn, q);
            let len = rn + usize::from(c != 0);
            if cmp_limbs(top(to, len), limit) == Ordering::Greater {
                to[..=rn].copy_from_slice(&st.keep[..=rn]);
                stop = true;
                break;
            }
            rn = len;
            if col == 1 {
                m1.u01 += q * m1.u00;
                m1.u11 += q * m1.u10;
            } else {
                m1.u00 += q * m1.u01;
                m1.u10 += q * m1.u11;
            }
        }
    }
    if remainder {
        core::mem::swap(&mut st.x, &mut st.tx);
        core::mem::swap(&mut st.y, &mut st.ty);
    } else {
        // SAFETY: the numbers have `n` limbs, and the steps of `m1` are steps of theirs.
        unsafe { reduce_pair(st.x.as_mut_ptr(), st.y.as_mut_ptr(), n, m1) };
    }
    core::mem::swap(&mut st.u0, &mut st.r0);
    core::mem::swap(&mut st.u1, &mut st.r1);
    st.un = rn;
    if !stop {
        st.n -= usize::from(st.x[n - 1] | st.y[n - 1] == 0);
    }
    stop
}

/// One division of [`exact`], the larger number by the smaller, the cofactor's entry of the row
/// grown by the quotient; for the cofactors not taken when it would pass the bound. Whether the
/// reduction ends: the bound met, or a remainder of zero.
///
/// # Safety
///
/// As for [`reduce_bounded`], with the `avail` limbs of the state's scratch.
unsafe fn divide_bounded(st: &mut State<'_>, limit: &[u64], remainder: bool, avail: usize) -> bool {
    let (n, un) = (st.n, st.un);
    // SAFETY: as the caller guarantees.
    unsafe {
        // Every pointer into the numbers from one borrow of each, as a second would invalidate
        // the first's.
        let (px, py) = (st.x.as_mut_ptr(), st.y.as_mut_ptr());
        let (l, ln, s, sn, col) = order_pair(px, py, n);
        if sn == 0 {
            return true;
        }
        let to_u0 = col == 0;
        // The larger's slot, to restore when the step is not taken.
        let (keep, kept) = if col == 1 {
            (st.tx.as_mut_ptr(), px)
        } else {
            (st.ty.as_mut_ptr(), py)
        };
        if !remainder {
            ptr::copy_nonoverlapping(kept, keep, n);
        }
        let cap = st.u0.len();
        st.r0.copy_from_slice(st.u0);
        st.r1.copy_from_slice(st.u1);
        let mut row = Row {
            u0: st.r0.as_mut_ptr(),
            u1: st.r1.as_mut_ptr(),
            un,
        };
        let qn = ln - sn + 1;
        let need = qn + tdiv_itch(ln, sn).max(add_times_itch(cap - 1, qn));
        let w = whole(st.t);
        with_scratch(w, avail, need, |w| {
            let (q, w) = w.split(qn);
            tdiv_qr(q, l, ln, s, sn, w);
            let qlen = normalized(q, qn);
            row.add_times(to_u0, q, qlen, w);
        });
        let rn = row.un;
        if !remainder {
            let grown = if to_u0 { &*st.r0 } else { &*st.r1 };
            if cmp_with(grown.as_ptr(), rn, limit) == Ordering::Greater {
                // The step passes the bound: the larger as it was.
                ptr::copy_nonoverlapping(keep, kept, n);
                return true;
            }
        }
        core::mem::swap(&mut st.u0, &mut st.r0);
        core::mem::swap(&mut st.u1, &mut st.r1);
        st.un = rn;
        ptr::write_bytes(l.add(sn), 0, n - sn);
        st.n = sn;
        let rem = normalized(l, sn);
        rem == 0 || remainder && cmp_with(l, rem, limit) != Ordering::Greater
    }
}

/// What [`common_prefix`] leaves: the matrix `M = (m00, m01; m10, m11)` of the steps the two pairs
/// share, `(a, b) = M (a', b')` for each, and the two pairs `(a', b')` it leaves; `exhausted` when
/// it stopped at its budget of [`CF_DC_THRESHOLD`] limbs rather than where the expansions part.
pub(crate) struct Common {
    pub(crate) m: [Number; 4],
    pub(crate) lo: (Number, Number),
    pub(crate) hi: (Number, Number),
    pub(crate) exhausted: bool,
}

/// The last step of a batch, its column and quotient; column 2 when there is none.
struct Last {
    col: usize,
    q: u64,
}

/// The steps of one call of hgcd2, which takes quotients of one limb only, at most about 92.
struct Record {
    steps: Vec<(usize, u64)>,
}

impl Steps for Record {
    #[inline(always)]
    fn step(&mut self, col: usize, q: u64) {
        self.steps.push((col, q));
    }
    unsafe fn step_limbs(&mut self, col: usize, _: *const u64, _: usize) {
        self.steps.push((col, 0));
    }
}

/// `r -= q x` over `n` limbs, into `out`: whether it took a complete step, the result neither
/// negative nor zero and below `x`.
fn complete_step(out: &mut [u64], r: &[u64], x: &[u64], n: usize, q: u64) -> bool {
    let mut borrow = 0u128;
    for i in 0..n {
        let p = u128::from(x[i]) * u128::from(q) + borrow;
        let (v, b) = r[i].overflowing_sub(p as u64);
        out[i] = v;
        borrow = (p >> 64) + u128::from(b);
    }
    borrow == 0 && {
        let o = top(out, n);
        !o.is_empty() && cmp_limbs(o, top(x, n)) == Ordering::Less
    }
}

/// `r -= q x` over `n` limbs, a step of Euclid's that leaves `r` non-negative.
fn sub_times(r: &mut [u64], x: &[u64], n: usize, q: u64) {
    let mut borrow = 0u128;
    for i in 0..n {
        let p = u128::from(x[i]) * u128::from(q) + borrow;
        let (v, b) = r[i].overflowing_sub(p as u64);
        r[i] = v;
        borrow = (p >> 64) + u128::from(b);
    }
    debug_assert!(borrow == 0);
}

/// `r += q x` over `n` limbs of rows, the carry written at `n`; whether there was one.
fn add_times_row(r: &mut [u64], x: &[u64], n: usize, q: u64) -> u64 {
    let mut carry = 0u128;
    for i in 0..n {
        let p = u128::from(x[i]) * u128::from(q) + u128::from(r[i]) + carry;
        r[i] = p as u64;
        carry = p >> 64;
    }
    r[n] = carry as u64;
    carry as u64
}

/// `(a', b') = (u11 a - u01 b, u00 b - u10 a)` over the `n` limbs of `a` and `b` into `x` and `y`,
/// or `false` when either is negative.
fn reduce_checked(
    x: &mut [u64],
    y: &mut [u64],
    a: &[u64],
    b: &[u64],
    n: usize,
    m: Matrix1,
) -> bool {
    let (mut ca, mut cb) = (0i128, 0i128);
    for i in 0..n {
        let (u, v) = (u128::from(a[i]), u128::from(b[i]));
        let p = (u128::from(m.u11) * u) as i128 - (u128::from(m.u01) * v) as i128 + ca;
        let q = (u128::from(m.u00) * v) as i128 - (u128::from(m.u10) * u) as i128 + cb;
        x[i] = p as u64;
        y[i] = q as u64;
        ca = p >> 64;
        cb = q >> 64;
    }
    ca == 0 && cb == 0
}

/// The limbs of `v` up to `n` without the zero limbs at the top.
fn top(v: &[u64], n: usize) -> &[u64] {
    let mut k = n;
    while k > 0 && v[k - 1] == 0 {
        k -= 1;
    }
    &v[..k]
}

/// The steps of Euclid's algorithm that the pairs `(a1, b1)` and `(a2, b2)`, each of a larger
/// first number and neither zero, share, up to a batch of hgcd2's before the one in which their
/// quotients part: the steps are hgcd2's matrices of the first pair, less the last step when it
/// is not taken in full, each kept when the second pair stays positive under it with its larger
/// number in the same place, which is when its quotients are the same (the pairs change places
/// when the second is longer). Lehmer's method, one pass over both pairs a matrix; it stops after
/// reducing the longer pair by [`CF_DC_THRESHOLD`] limbs.
pub(crate) fn common_prefix(
    a1: &Number,
    b1: &Number,
    a2: &Number,
    b2: &Number,
    order: ByteOrder,
) -> Common {
    let longest = Operand::of(a1).significant.max(Operand::of(a2).significant) / 8;
    if longest < COMMON_DC_THRESHOLD {
        return common_lehmer(a1, b1, a2, b2, order);
    }
    let mut p1 = (a1.clone(), b1.clone());
    let mut p2 = (a2.clone(), b2.clone());
    let m = common_dc(&mut p1, &mut p2, order);
    let mut c = common_lehmer(&p1.0, &p1.1, &p2.0, &p2.1, order);
    if let Some(m) = m {
        c.m = times(&m, &c.m);
    }
    c
}

/// From this many limbs of the longer pair, [`common_prefix`] takes the steps that the pairs share
/// by the half gcd of the longer one first, on its top half and then on shorter tops where the
/// pairs part. Measured on an Apple M4 Max on `x` and `x + 1/q`, the longer pair of 250 to 1000
/// limbs took 0.71 to 0.80 of the time with Lehmer's method alone, and of 1600 limbs as long.
const COMMON_DC_THRESHOLD: usize = if cfg!(miri) { 8 } else { 1400 };

/// The shortest top on which [`common_dc`] tries the half gcd.
const COMMON_DC_MIN: usize = if cfg!(miri) { 4 } else { 48 };

/// The product `x y` of matrices `(m00, m01, m10, m11)`.
fn times(x: &[Number; 4], y: &[Number; 4]) -> [Number; 4] {
    let [a, b, c, d] = x;
    let [e, f, g, h] = y;
    [a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h]
}

/// The last run of steps on one number that a reduction tells of, its quotients summed: Euclid's
/// last quotient, or the part of it taken.
struct LastRun {
    col: usize,
    q: Vec<u64>,
}

impl Steps for LastRun {
    fn step(&mut self, col: usize, q: u64) {
        if col != self.col {
            self.col = col;
            self.q.clear();
        }
        add_limbs(&mut self.q, &[q]);
    }
    unsafe fn step_limbs(&mut self, col: usize, q: *const u64, qn: usize) {
        if col != self.col {
            self.col = col;
            self.q.clear();
        }
        // SAFETY: as the caller guarantees.
        add_limbs(&mut self.q, unsafe { core::slice::from_raw_parts(q, qn) });
    }
}

/// The steps that the pairs share, from the half gcd of the longer one's top half, or of shorter
/// tops when the pairs part within it: each matrix, without its last step when that is not taken
/// in full, kept when the other pair stays positive under it with its larger number in the same
/// place. Leaves the pairs reduced and returns the product of the matrices kept, if any.
fn common_dc(
    p1: &mut (Number, Number),
    p2: &mut (Number, Number),
    order: ByteOrder,
) -> Option<[Number; 4]> {
    let mut total: Option<[Number; 4]> = None;
    let len = |p: &(Number, Number)| {
        Operand::of(&p.0)
            .significant
            .max(Operand::of(&p.1).significant)
            .div_ceil(8)
    };
    // The top tried first: half the numbers, or twice the last one taken, which follows a common
    // part shorter than half the numbers down without trying the longer tops again.
    let mut next = usize::MAX;
    loop {
        let (d, o) = if len(p2) > len(p1) {
            (&mut *p2, &mut *p1)
        } else {
            (&mut *p1, &mut *p2)
        };
        let n = len(d);
        if n < COMMON_DC_THRESHOLD {
            break;
        }
        let mut k = (n / 2).min(next);
        let mut taken = None;
        while k >= COMMON_DC_MIN {
            if let Some(t) = dc_try(d, o, n, k, order) {
                taken = Some(t);
                break;
            }
            k /= 2;
        }
        next = 2 * k;
        let Some((dp, op, m)) = taken else {
            break;
        };
        (*d, *o) = (dp, op);
        total = Some(match total {
            Some(t) => times(&t, &m),
            None => m,
        });
    }
    total
}

/// One try of [`common_dc`]: the half gcd on the top `k` of the `n` limbs of the pair `d`, the
/// pairs it leaves and its matrix when `o` shares its steps.
#[allow(clippy::type_complexity)]
fn dc_try(
    d: &(Number, Number),
    o: &(Number, Number),
    n: usize,
    k: usize,
    order: ByteOrder,
) -> Option<((Number, Number), (Number, Number), [Number; 4])> {
    let (mut x, _) = limbs(&d.0, n + 3);
    let (mut y, _) = limbs(&d.1, n + 3);
    let p = n - k;
    let mn = k.div_ceil(2) + 1;
    let need = matrix_itch(k) + hgcd_itch(k).max(adjust_itch(p, mn));
    let mut w = work(need);
    let w = whole(&mut w);
    let mut last = LastRun {
        col: 2,
        q: Vec::new(),
    };
    // SAFETY: the numbers have `n` limbs and three to spare, the scratch the half gcd's plan.
    let (m, nx) = unsafe {
        let mut mtx = Matrix::new(k, w.take(matrix_itch(k)));
        let rest = w.add(matrix_itch(k));
        let nn = hgcd_with(
            x.as_mut_ptr().add(p),
            y.as_mut_ptr().add(p),
            k,
            &mut mtx,
            rest,
            &mut last,
        );
        if nn == 0 {
            return None;
        }
        let nx = mtx.adjust(p + nn, x.as_mut_ptr(), y.as_mut_ptr(), p, rest);
        let e = |i: usize, j: usize| limbs_number(mtx.p[i][j], mtx.n, false, order);
        ([e(0, 0), e(0, 1), e(1, 0), e(1, 1)], nx)
    };
    let [mut m00, mut m01, mut m10, mut m11] = m;
    // SAFETY: the vectors hold the limbs read.
    let (mut dx, mut dy) = unsafe {
        (
            limbs_number(x.as_ptr(), nx, false, order),
            limbs_number(y.as_ptr(), nx, false, order),
        )
    };
    // The last quotient, when not taken in full, undone.
    if last.col != 2 && !last.q.is_empty() {
        // SAFETY: the limbs are those of the vector.
        let q = unsafe { limbs_number(last.q.as_ptr(), last.q.len(), false, order) };
        if last.col == 1 && dx >= dy {
            dx = &dx + &(&q * &dy);
            m01 = &m01 - &(&q * &m00);
            m11 = &m11 - &(&q * &m10);
        } else if last.col == 0 && dy >= dx {
            dy = &dy + &(&q * &dx);
            m00 = &m00 - &(&q * &m01);
            m10 = &m10 - &(&q * &m11);
        }
    }
    if m01.is_zero() && m10.is_zero() {
        return None;
    }
    let ox = &(&m11 * &o.0) - &(&m01 * &o.1);
    let oy = &(&m00 * &o.1) - &(&m10 * &o.0);
    if !ox.is_positive() || !oy.is_positive() || (dx > dy) != (ox > oy) || dx == dy || ox == oy {
        return None;
    }
    Some(((dx, dy), (ox, oy), [m00, m01, m10, m11]))
}

/// [`common_prefix`] by Lehmer's method alone.
fn common_lehmer(a1: &Number, b1: &Number, a2: &Number, b2: &Number, order: ByteOrder) -> Common {
    let (_, l1) = limbs(a1, 0);
    let (_, l2) = limbs(a2, 0);
    let (mut x1, _) = limbs(a1, l1 + 1);
    let (mut y1, _) = limbs(b1, l1 + 1);
    let (mut x2, _) = limbs(a2, l2 + 1);
    let (mut y2, _) = limbs(b2, l2 + 1);
    let (mut s1, mut t1) = (vec![0u64; l1 + 1], vec![0u64; l1 + 1]);
    let (mut s2, mut t2) = (vec![0u64; l2 + 1], vec![0u64; l2 + 1]);
    // The matrices come from the longer pair, which keeps more steps in its top limbs: the pairs
    // are swapped when the other is longer, and swapped back at the end.
    let mut swapped = false;
    let start = l1.max(l2);
    let cap = l1.max(l2) + 4;
    let mut rows = [
        vec![0u64; cap],
        vec![0u64; cap],
        vec![0u64; cap],
        vec![0u64; cap],
    ];
    rows[0][0] = 1;
    rows[3][0] = 1;
    let mut un = 1;
    let (mut n1, mut n2) = (l1, l2);
    let mut ones = false;
    let mut exhausted = false;
    let mut record = Record {
        steps: Vec::with_capacity(96),
    };
    loop {
        if n2 > n1 {
            core::mem::swap(&mut x1, &mut x2);
            core::mem::swap(&mut y1, &mut y2);
            core::mem::swap(&mut s1, &mut s2);
            core::mem::swap(&mut t1, &mut t2);
            core::mem::swap(&mut n1, &mut n2);
            swapped = !swapped;
        }
        if n1 < 2 {
            break;
        }
        if start - n1 > CF_DC_THRESHOLD {
            exhausted = true;
            break;
        }
        // SAFETY: both numbers have `n1` limbs, not both zero at the top.
        let (ah, al, bh, bl) = unsafe { top_limbs(x1.as_ptr(), y1.as_ptr(), n1) };
        record.steps.clear();
        let Some(mut m) = hgcd2_hinted_with(ah, al, bh, bl, &mut ones, &mut record) else {
            break;
        };
        let last = record
            .steps
            .last()
            .copied()
            .map_or(Last { col: 2, q: 0 }, |(col, q)| Last { col, q });
        s1[..n1].copy_from_slice(&x1[..n1]);
        t1[..n1].copy_from_slice(&y1[..n1]);
        // SAFETY: the copies have `n1` limbs.
        unsafe { reduce_pair(s1.as_mut_ptr(), t1.as_mut_ptr(), n1, m) };
        if last.col != 2 && last.q != 0 {
            let (red, other) = if last.col == 1 {
                (&mut s1, &t1)
            } else {
                (&mut t1, &s1)
            };
            if cmp_limbs(top(red, n1), top(other, n1)) != Ordering::Less {
                // Not taken in full: undone, in the numbers and in the matrix.
                // SAFETY: both have `n1` limbs; the sum is the number before the step.
                let carry = unsafe { addmul_1(red.as_mut_ptr(), other.as_ptr(), n1, last.q) };
                debug_assert!(carry == 0);
                if last.col == 1 {
                    m.u01 -= last.q * m.u00;
                    m.u11 -= last.q * m.u10;
                } else {
                    m.u00 -= last.q * m.u01;
                    m.u10 -= last.q * m.u11;
                }
            }
        }
        if m.u01 == 0 && m.u10 == 0 {
            break;
        }
        let accepted = reduce_checked(&mut s2, &mut t2, &x2, &y2, n2, m) && {
            let order1 = cmp_limbs(top(&s1, n1), top(&t1, n1));
            let order2 = cmp_limbs(top(&s2, n2), top(&t2, n2));
            order1 == order2 && !top(&s2, n2).is_empty() && !top(&t2, n2).is_empty()
        };
        if !accepted {
            // The quotients part in this batch: its steps one at a time, while both take them.
            for &(col, q) in &record.steps {
                let ok = if col == 1 {
                    complete_step(&mut s1, &x1, &y1, n1, q)
                        && complete_step(&mut s2, &x2, &y2, n2, q)
                } else {
                    complete_step(&mut t1, &y1, &x1, n1, q)
                        && complete_step(&mut t2, &y2, &x2, n2, q)
                };
                if !ok {
                    break;
                }
                let [r00, r01, r10, r11] = &mut rows;
                let c = if col == 1 {
                    x1[..n1].copy_from_slice(&s1[..n1]);
                    x2[..n2].copy_from_slice(&s2[..n2]);
                    add_times_row(r01, r00, un, q) | add_times_row(r11, r10, un, q)
                } else {
                    y1[..n1].copy_from_slice(&t1[..n1]);
                    y2[..n2].copy_from_slice(&t2[..n2]);
                    add_times_row(r00, r01, un, q) | add_times_row(r10, r11, un, q)
                };
                un += usize::from(c != 0);
            }
            break;
        }
        core::mem::swap(&mut x1, &mut s1);
        core::mem::swap(&mut y1, &mut t1);
        core::mem::swap(&mut x2, &mut s2);
        core::mem::swap(&mut y2, &mut t2);
        while n1 > 0 && x1[n1 - 1] | y1[n1 - 1] == 0 {
            n1 -= 1;
        }
        while n2 > 0 && x2[n2 - 1] | y2[n2 - 1] == 0 {
            n2 -= 1;
        }
        // M = M m: each row times m.
        let [r00, r01, r10, r11] = &mut rows;
        // SAFETY: the rows have `cap` limbs, and their entries are at most the numbers.
        let grow = unsafe {
            let (c0, c1) = extend_row(r00.as_mut_ptr(), r01.as_mut_ptr(), un, m);
            let (c2, c3) = extend_row(r10.as_mut_ptr(), r11.as_mut_ptr(), un, m);
            r00[un] = c0;
            r01[un] = c1;
            r10[un] = c2;
            r11[un] = c3;
            c0 | c1 | c2 | c3 != 0
        };
        un += usize::from(grow);
    }
    if swapped {
        core::mem::swap(&mut x1, &mut x2);
        core::mem::swap(&mut y1, &mut y2);
        core::mem::swap(&mut n1, &mut n2);
    }
    // SAFETY: the vectors hold the limbs read.
    let number = |v: &[u64], len: usize| unsafe { limbs_number(v.as_ptr(), len, false, order) };
    Common {
        m: [
            number(&rows[0], un),
            number(&rows[1], un),
            number(&rows[2], un),
            number(&rows[3], un),
        ],
        lo: (number(&x1, n1), number(&y1, n1)),
        hi: (number(&x2, n2), number(&y2, n2)),
        exhausted,
    }
}
