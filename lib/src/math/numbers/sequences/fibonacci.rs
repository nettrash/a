//! Fibonacci and Lucas numbers by fast doubling, after GMP's `mpz_fib_ui` and `mpz_lucnum_ui`.
//!
//! Below `2^128` they come from tables. Beyond, `F_k` and `F_(k-1)` are doubled from the top bits of
//! the index down, two squares a bit ([`fib2`]):
//!
//! - `F_(2k-1) = F_k^2 + F_(k-1)^2`,
//! - `F_(2k+1) = 4 F_k^2 - F_(k-1)^2 + 2 (-1)^k`,
//! - `F_(2k) = F_(2k+1) - F_(2k-1) = 3 F_k^2 - 2 F_(k-1)^2 + 2 (-1)^k`,
//!
//! the pair kept being `F_(2k)` and `F_(2k-1)` for a zero bit and `F_(2k+1)` and `F_(2k)` for a one,
//! both computed from the squares in one pass over their limbs ([`halves`]). The first doubling,
//! from the table's pair of at most two limbs, is taken in registers, and the squares of up to
//! [`BASECASE_SQR`] limbs are the schoolbook kernel's, called directly. The last step of `F_n` takes
//! one product instead of two squares: `F_(2k) = F_k (F_k + 2 F_(k-1))` and
//! `F_(2k+1) = (2 F_k + F_(k-1)) (2 F_k - F_(k-1)) + 2 (-1)^k`. `L_n` takes the zero bits at the
//! bottom of `n` as squares, `L_(2k) = L_k^2 - 2 (-1)^k`, from the odd part `m = 2k + 1`, which is
//! `L_(2k+1) = 5 F_(k-1) (2 F_k + F_(k-1)) - 4 (-1)^k`, or from the table.

// The kernel of the doublings' linear part: assembly on 64-bit little-endian aarch64 where the
// system allows unaligned accesses, as the multiplication's kernels, and portable Rust elsewhere
// and under Miri.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
))]
#[path = "fibonacci/aarch64.rs"]
mod kernels;
#[cfg(not(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
)))]
#[path = "fibonacci/portable.rs"]
mod kernels;

use super::super::mul::basecase;
use super::super::mul::limbs::{add_1, add_n, mul, mul_itch, sqr, sqr_itch, sub_1, sub_n};
use super::super::mul::{STACK_LIMBS, on_heap, on_stack};
use super::super::root::{limbs_number, lshift, mul_1, small_number};
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
use super::small;
use core::mem::MaybeUninit;

/// `F_n` for `n` up to 186, the last below `2^128`.
pub(in super::super) const FIB: [u128; 187] = {
    let mut t = [0u128; 187];
    t[1] = 1;
    let mut i = 2;
    while i < 187 {
        t[i] = t[i - 1] + t[i - 2];
        i += 1;
    }
    t
};

/// `L_n = F_(n-1) + F_(n+1)` for `n` up to 184, the last below `2^128`; `L_0 = 2`.
const LUC: [u128; 185] = {
    let mut t = [2u128; 185];
    let mut i = 1;
    while i < 185 {
        t[i] = FIB[i - 1] + FIB[i + 1];
        i += 1;
    }
    t
};

/// The largest index whose `F_k` the table holds, for the start of [`fib2`].
const FIB_LAST: u64 = FIB.len() as u64 - 1;

/// The longest operand whose square is the schoolbook kernel's, called directly: the
/// multiplication squares by the schoolbook method below 56 limbs, where it measured faster than
/// Karatsuba's method up to 48 (see `mul/limbs.rs`); longer ones go through its dispatch.
const BASECASE_SQR: usize = 48;

/// The longest operands whose product is the schoolbook kernel's of equal lengths, the shorter one
/// extended by a zero limb: the multiplication takes Karatsuba's method from 24 limbs.
const BASECASE_MUL: usize = 20;

/// An upper bound of the limbs of `F_k`: `F_k <= phi^(k-1)`, and `log2(phi) / 64 < 1/92`.
#[inline]
fn fib_limbs(k: u64) -> u64 {
    k / 92 + 2
}

/// `F_n`, in the byte order `order`.
pub(in super::super) fn fibonacci(n: u64, order: ByteOrder) -> Number {
    if n <= FIB_LAST {
        return small_number(order, FIB[n as usize], false);
    }
    long_fibonacci(n, order)
}

/// `L_n`, in the byte order `order`.
pub(in super::super) fn lucas(n: u64, order: ByteOrder) -> Number {
    if n < LUC.len() as u64 {
        return small_number(order, LUC[n as usize], false);
    }
    long_lucas(n, order)
}

/// Two limbs, aligned to 16 bytes, as the kernels load and store limbs in pairs.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub(super) struct Pair([u64; 2]);

/// `n` rounded up to an even number of limbs, so that a buffer after `n` limbs of one that starts
/// on 16 bytes starts on 16 bytes too.
#[inline(always)]
pub(super) const fn even(n: usize) -> usize {
    (n + 1) & !1
}

/// Working memory for the squares and the products: the limbs left over on the stack, or a vector
/// that grows to the largest plan.
pub(super) struct Spare {
    /// The limbs left over on the stack, or null.
    stack: *mut u64,
    /// Their number.
    stack_len: usize,
    /// The vector, for plans that do not fit.
    heap: Vec<Pair>,
}

impl Spare {
    /// The `len` limbs at `stack`, and an empty vector.
    pub(super) fn new(stack: *mut u64, stack_len: usize) -> Spare {
        Spare {
            stack,
            stack_len,
            heap: Vec::new(),
        }
    }

    /// `n` limbs of working memory, apart from every buffer.
    #[inline]
    pub(super) fn get(&mut self, n: usize) -> Scratch {
        if n <= self.stack_len {
            return Scratch::new(self.stack, n);
        }
        let pairs = n.div_ceil(2);
        if self.heap.capacity() < pairs {
            // The old limbs are not needed: a new vector, not a copy.
            self.heap = Vec::with_capacity(pairs);
        }
        Scratch::new(self.heap.as_mut_ptr().cast(), n)
    }
}

/// The limbs of working memory up to which they are a buffer in the caller's frame, not in that of
/// [`on_stack`].
const SMALL_LIMBS: usize = 192;

/// Runs `f` on `need` limbs of working memory, those of the stack when they fit, with what is left
/// of the stack's limbs as the start of the spare memory.
#[inline(always)]
pub(super) fn with_memory<R>(need: usize, f: impl FnOnce(*mut u64, Spare) -> R) -> R {
    if need <= SMALL_LIMBS {
        let mut s = [const { MaybeUninit::<Pair>::uninit() }; SMALL_LIMBS / 2];
        let s = s.as_mut_ptr().cast::<u64>();
        // SAFETY: as below, in the buffer of `SMALL_LIMBS`.
        let rest = unsafe { s.add(even(need)) };
        return f(s, Spare::new(rest, SMALL_LIMBS - even(need)));
    }
    if need <= STACK_LIMBS {
        on_stack(|s: *mut MaybeUninit<u64>| {
            let s = s.cast::<u64>();
            // SAFETY: the stack holds `STACK_LIMBS` limbs, of which the first `need` are the
            // buffers, and the rest, from an even limb, the spare memory.
            let rest = unsafe { s.add(even(need)) };
            f(s, Spare::new(rest, STACK_LIMBS - even(need)))
        })
    } else {
        on_heap(need, |s: *mut MaybeUninit<u64>| {
            f(s.cast(), Spare::new(core::ptr::null_mut(), 0))
        })
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `a`: the schoolbook kernel up to
/// [`BASECASE_SQR`] limbs, and the multiplication's dispatch beyond.
///
/// # Safety
///
/// The limbs at `a` must be valid for reads and those at `r` for writes, apart from each other and
/// from the spare memory.
#[inline(always)]
pub(super) unsafe fn square(r: *mut u64, a: *const u64, n: usize, spare: &mut Spare) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if n <= small::MAX {
            small::sqr(r, a, n);
        } else if n <= BASECASE_SQR {
            basecase::sqr(r, a, n);
        } else {
            let t = spare.get(sqr_itch(n));
            sqr(r, a, n, t);
        }
    }
}

/// Sets the `na + nb` limbs at `r` to the product of the `na` limbs at `a` and the `nb` at `b`, of
/// either length: the schoolbook kernel of equal lengths when both have at most [`BASECASE_MUL`]
/// limbs, the shorter one then read up to the longer one's length, and the multiplication's
/// dispatch beyond. Returns the number of limbs of the product written, `2 max(na, nb)` or
/// `na + nb`.
///
/// # Safety
///
/// The limbs at `a` and `b` must be valid for reads up to `max(na, nb)` limbs each, those above
/// `na` and `nb` zero when both are at most [`BASECASE_MUL`]; those at `r` must be valid for
/// writing `2 max(na, nb)` limbs, apart from both and from the spare memory.
#[inline(always)]
unsafe fn product(
    r: *mut u64,
    a: *const u64,
    na: usize,
    b: *const u64,
    nb: usize,
    spare: &mut Spare,
) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        let m = na.max(nb);
        if m <= small::MAX {
            small::mul(r, a, b, m);
            return 2 * m;
        }
        if m <= BASECASE_MUL {
            basecase::mul(r, a, b, m);
            return 2 * m;
        }
        let (x, nx, y, ny) = if na >= nb {
            (a, na, b, nb)
        } else {
            (b, nb, a, na)
        };
        let t = spare.get(mul_itch(nx, ny));
        mul(r, x, nx, y, ny, t);
        nx + ny
    }
}

/// The state of the three carry chains of a doubling's linear part ([`halves`]).
pub(super) struct Chains {
    /// The carry of `s = x + y`, 0 or 1.
    cs: u64,
    /// The borrow of `d = 4x - y`, 0 or 1.
    bd: u64,
    /// The borrow of `f = d - s`, 0 or 1.
    bf: u64,
    /// The limb of `x` below the next one, whose top two bits `4x` shifts in.
    below: u64,
}

impl Chains {
    /// The limbs of `d`, `f` and `s` at the next position, from the limbs of `x` and `y` there,
    /// with `x_or` or-ed into the limb of `4x` and `y_or` into that of `y` for `d`.
    #[inline(always)]
    fn limb(&mut self, xi: u64, yi: u64, x_or: u64, y_or: u64) -> (u64, u64, u64) {
        let x4 = (xi << 2) | (self.below >> 62) | x_or;
        self.below = xi;
        let s = u128::from(xi) + u128::from(yi) + u128::from(self.cs);
        self.cs = (s >> 64) as u64;
        let d = u128::from(x4)
            .wrapping_sub(u128::from(yi | y_or))
            .wrapping_sub(u128::from(self.bd));
        self.bd = (d >> 127) as u64;
        let f = u128::from(d as u64)
            .wrapping_sub(u128::from(s as u64))
            .wrapping_sub(u128::from(self.bf));
        self.bf = (f >> 127) as u64;
        (d as u64, f as u64, s as u64)
    }
}

/// The linear part of a doubling: from `x = F_j^2` and `y = F_(j-1)^2`, `n` limbs each, sets the
/// `n + 1` limbs at `a` and `b` to `(F_(2j+1), F_(2j))` when `ONE` and to `(F_(2j), F_(2j-1))`
/// otherwise, in one pass of three carry chains: `s = F_(2j-1) = x + y`,
/// `d = F_(2j+1) = 4x - y + 2σ` and `f = F_(2j) = d - s`, for `σ = (-1)^j`, `plus` when `σ = 1`.
/// The `2σ` is or-ed in as GMP's `mpn_fib2_ui` does: into the low limb of `4x`, 0 modulo 4, for
/// `+2`, and into that of `y`, a square, 0 or 1 modulo 4, for `-2`. The limbs from the first
/// up to a multiple of four from the top are taken here, the blocks of four above them by the
/// kernel ([`kernels::blocks`]).
///
/// # Safety
///
/// `n >= 1`; the limbs at `x` and `y` must be valid for reads, those at `a` and `b` for writes,
/// all apart.
#[inline(always)]
unsafe fn halves<const ONE: bool>(
    a: *mut u64,
    b: *mut u64,
    x: *const u64,
    y: *const u64,
    n: usize,
    plus: bool,
) {
    let mut c = Chains {
        cs: 0,
        bd: 0,
        bf: 0,
        below: 0,
    };
    let head = 1 + (n - 1) % 4;
    // SAFETY: as the caller guarantees; the kernel takes the `n - head` limbs above the head, a
    // multiple of four.
    unsafe {
        for i in 0..head {
            let (x_or, y_or) = match (i, plus) {
                (0, true) => (2, 0),
                (0, false) => (0, 2),
                _ => (0, 0),
            };
            let (d, f, s) = c.limb(*x.add(i), *y.add(i), x_or, y_or);
            if ONE {
                *a.add(i) = d;
                *b.add(i) = f;
            } else {
                *a.add(i) = f;
                *b.add(i) = s;
            }
        }
        if n > head {
            kernels::blocks::<ONE>(
                a.add(head),
                b.add(head),
                x.add(head),
                y.add(head),
                (n - head) / 4,
                &mut c,
            );
        }
        // The top limbs: `4x` has two bits more, and neither difference is negative.
        let d = (c.below >> 62).wrapping_sub(c.bd);
        let f = d.wrapping_sub(c.cs).wrapping_sub(c.bf);
        if ONE {
            *a.add(n) = d;
            *b.add(n) = f;
        } else {
            *a.add(n) = f;
            *b.add(n) = c.cs;
        }
    }
}

/// The four limbs of `a^2`.
#[inline(always)]
fn square_u128(a: u128) -> [u64; 4] {
    let (a0, a1) = (u128::from(a as u64), u128::from((a >> 64) as u64));
    let (p00, p01, p11) = (a0 * a0, a0 * a1, a1 * a1);
    // `a^2 = p00 + 2 p01 B + p11 B^2`, a column at a time.
    let c1 = (p00 >> 64) + (u128::from(p01 as u64) << 1);
    let c2 = u128::from(p11 as u64) + ((p01 >> 64) << 1) + (c1 >> 64);
    [
        p00 as u64,
        c1 as u64,
        c2 as u64,
        ((p11 >> 64) + (c2 >> 64)) as u64,
    ]
}

/// The four limbs of `a b`.
#[inline(always)]
fn mul_u128(a: u128, b: u128) -> [u64; 4] {
    let (a0, a1) = (u128::from(a as u64), u128::from((a >> 64) as u64));
    let (b0, b1) = (u128::from(b as u64), u128::from((b >> 64) as u64));
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    // A column at a time: the middle one's three terms below `3 2^64`, in a `u128`.
    let c1 = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
    let c2 = u128::from(p11 as u64) + (p01 >> 64) + (p10 >> 64) + (c1 >> 64);
    [
        p00 as u64,
        c1 as u64,
        c2 as u64,
        ((p11 >> 64) + (c2 >> 64)) as u64,
    ]
}

/// The `n` limbs at `a` without the zero limbs at the top, at least one.
///
/// # Safety
///
/// The `n` limbs at `a` must be valid for reads.
#[inline(always)]
unsafe fn trim(a: *const u64, mut n: usize) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        while n > 1 && *a.add(n - 1) == 0 {
            n -= 1;
        }
    }
    n
}

/// Writes the two limbs of `v` at `r`.
///
/// # Safety
///
/// `r` must be valid for writing two limbs.
#[inline(always)]
unsafe fn put_u128(r: *mut u64, v: u128) {
    // SAFETY: as the caller guarantees.
    unsafe {
        *r = v as u64;
        *r.add(1) = (v >> 64) as u64;
    }
}

/// The limbs up to which the linear steps around [`fib2`] are taken in one pass of a loop; beyond,
/// by the addition's kernels, which run at about a cycle a limb.
const LINEAR: usize = 32;

/// Sets the `s` limbs at `r` to `L = F + 2G` for the `s` limbs of `F` at `f` and of `G` at `g`, and
/// returns the limb above them; `t` is `s` limbs of scratch.
///
/// # Safety
///
/// The limbs must be valid, `r` and `t` apart from `f`, `g` and each other.
#[inline(always)]
unsafe fn lucas_of(r: *mut u64, f: *const u64, g: *const u64, s: usize, t: *mut u64) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe {
        if s <= LINEAR {
            let mut c = 0u128;
            for i in 0..s {
                let v = u128::from(*f.add(i)) + (u128::from(*g.add(i)) << 1) + c;
                *r.add(i) = v as u64;
                c = v >> 64;
            }
            c as u64
        } else {
            let c = lshift(t, g, s, 1);
            c + u64::from(add_n(r, t, f, s))
        }
    }
}

/// `F_n` for `n` above the table: the pair at `n / 2` by [`fib2`], then one product.
#[inline(never)]
fn long_fibonacci(n: u64, order: ByteOrder) -> Number {
    let k = n / 2;
    if k <= FIB_LAST {
        return pair_fibonacci(n, order);
    }

    // The pair's buffers hold `F_k` with a limb to spare, the squares' buffers as many; the
    // product twice as many. Counted in 64 bits: on 32-bit targets the limbs of large indices do not
    // fit a `usize`.
    let fl = even(usize::try_from(fib_limbs(k) + 2).expect("capacity overflow"));
    let need = fl.checked_mul(6).expect("capacity overflow");
    with_memory(need, |w, mut spare| {
        // SAFETY: the working memory holds the six buffers of `fl` limbs: the pair `f`, `g`, the
        // squares `x`, `y`, and the result `r` of `2 fl`.
        unsafe {
            let (f, g, x, y, r) = (w, w.add(fl), w.add(2 * fl), w.add(3 * fl), w.add(4 * fl));
            let s = fib2(f, g, x, y, k, &mut spare);
            // `F_k` has `s` limbs, the top one not zero; `F_(k-1)` at most as many, its limbs
            // above its own zero.
            let len = if n & 1 == 1 {
                // `a = 2 F_k + F_(k-1)` and `b = 2 F_k - F_(k-1)`, of `s + 1` limbs, in `x` and
                // `y`; `b`'s top limb is at most 1. In one pass up to `LINEAR` limbs, and by the
                // addition's kernels beyond.
                let (ca, cb) = if s <= LINEAR {
                    let (mut ca, mut cb) = (0i128, 0i128);
                    for i in 0..s {
                        let (fi, gi) = (i128::from(*f.add(i)), i128::from(*g.add(i)));
                        let ta = (fi << 1) + gi + ca;
                        let tb = (fi << 1) - gi + cb;
                        *x.add(i) = ta as u64;
                        *y.add(i) = tb as u64;
                        ca = ta >> 64;
                        cb = tb >> 64;
                    }
                    (ca as u64, cb as u64)
                } else {
                    let c2 = lshift(r, f, s, 1);
                    let c3 = u64::from(add_n(x, r, g, s));
                    let b = u64::from(sub_n(y, r, g, s));
                    (c2 + c3, c2 - b)
                };
                *x.add(s) = ca;
                *y.add(s) = cb;
                let sa = s + usize::from(ca != 0);
                let sb = trim(y, s + 1);
                // Zeros up to the longer one, for the schoolbook kernel of equal lengths.
                for i in sb..sa {
                    *y.add(i) = 0;
                }
                let len = product(r, x, sa, y, sb, &mut spare);
                // `+ 2 (-1)^k`: the product is above 2.
                if k & 1 == 0 {
                    add_1(r, len, 2);
                } else {
                    sub_1(r, len, 2);
                }
                len
            } else {
                // `F_k + 2 F_(k-1)`, of `s + 1` limbs, in `x`, times `F_k`.
                let c = lucas_of(x, f, g, s, r);
                *x.add(s) = c;
                let sx = s + usize::from(c != 0);
                *f.add(s) = 0;
                product(r, x, sx, f, s, &mut spare)
            };
            limbs_number(r, len, false, order)
        }
    })
}

/// `F_n` for `n / 2` in the table: one product of two numbers from the tables, in registers.
///
/// `F_(2k) = F_k L_k` and `F_(2k+1) = F_(k+2) L_(k-1) + 2 (-1)^k` (the last step of [`fib2`]'s,
/// `2 F_k + F_(k-1) = F_(k+2)` and `2 F_k - F_(k-1) = L_(k-1)`), both factors below `2^128` up to
/// `k = 184`; for `k` of 185 and 186 from `F_k` and `F_(k-1)` in three limbs.
fn pair_fibonacci(n: u64, order: ByteOrder) -> Number {
    let k = n / 2;
    if k < LUC.len() as u64 {
        let k = k as usize;
        let mut r = if n & 1 == 0 {
            mul_u128(FIB[k], LUC[k])
        } else {
            mul_u128(FIB[k + 2], LUC[k - 1])
        };
        // SAFETY: the array holds the four limbs, and the product is above 2.
        unsafe {
            if n & 1 == 1 {
                if k & 1 == 0 {
                    add_1(r.as_mut_ptr(), 4, 2);
                } else {
                    sub_1(r.as_mut_ptr(), 4, 2);
                }
            }
            return limbs_number(r.as_ptr(), 4, false, order);
        }
    }
    let (f, g) = (FIB[k as usize], FIB[k as usize - 1]);
    // `(a, b)` of three limbs each, and the product of six.
    let wide = |x: u128, c: u64| -> [u64; 3] { [x as u64, (x >> 64) as u64, c] };
    let (a, b, add) = if n & 1 == 1 {
        // `(2 F_k + F_(k-1)) (2 F_k - F_(k-1)) + 2 (-1)^k`.
        let (t, c1) = f.overflowing_add(f);
        let (a, c2) = t.overflowing_add(g);
        let (b, c3) = t.overflowing_sub(g);
        (
            wide(a, u64::from(c1) + u64::from(c2)),
            wide(b, u64::from(c1) - u64::from(c3)),
            if k & 1 == 0 { 2i64 } else { -2 },
        )
    } else {
        // `F_k (F_k + 2 F_(k-1))`.
        let (t, c1) = g.overflowing_add(g);
        let (a, c2) = t.overflowing_add(f);
        (wide(a, u64::from(c1) + u64::from(c2)), wide(f, 0), 0)
    };
    let mut r = [0u64; 6];
    for i in 0..3 {
        let mut carry = 0u64;
        for j in 0..3 {
            let p = u128::from(a[i]) * u128::from(b[j]) + u128::from(r[i + j]) + u128::from(carry);
            r[i + j] = p as u64;
            carry = (p >> 64) as u64;
        }
        r[i + 3] = carry;
    }
    // SAFETY: the array holds the six limbs.
    unsafe {
        if add > 0 {
            add_1(r.as_mut_ptr(), 6, 2);
        } else if add < 0 {
            sub_1(r.as_mut_ptr(), 6, 2);
        }
        limbs_number(r.as_ptr(), 6, false, order)
    }
}

/// Sets the limbs at `f` to `F_k` and those at `g` to `F_(k-1)`, for `k` at least 2, and returns
/// their number: the top one of `F_k` is not zero, that of `F_(k-1)` may be.
///
/// The pair starts at the top bits of `k` that the table holds, and is doubled for each bit below:
/// the first time in registers, from the table's pair of at most two limbs, then by the squares of
/// the pair in `x` and `y` and [`halves`].
///
/// # Safety
///
/// The buffers `f`, `g`, `x` and `y` must be valid for writing [`fib_limbs`]`(k) + 2` limbs each,
/// apart from each other and from the spare memory.
pub(super) unsafe fn fib2(
    f: *mut u64,
    g: *mut u64,
    x: *mut u64,
    y: *mut u64,
    k: u64,
    spare: &mut Spare,
) -> usize {
    debug_assert!(k >= 2);
    // The bits of `k` below the start: `k >> bits` is at most the table's last index.
    let bits = 64 - (k / (FIB_LAST + 1)).leading_zeros();
    let first = k >> bits;
    // SAFETY: as the caller guarantees: each pair has at most the limbs of `F_k`, plus one for a
    // carry out of the top, and each square at most twice those of its operand.
    unsafe {
        let (a, b) = (FIB[first as usize], FIB[first as usize - 1]);
        if bits == 0 {
            put_u128(f, a);
            put_u128(g, b);
            return trim(f, 2);
        }
        // The first doubling, from `j = first`, in registers: the squares have four limbs, the
        // pair five.
        let (xa, ya) = (square_u128(a), square_u128(b));
        let plus = first & 1 == 0;
        if (k >> (bits - 1)) & 1 == 1 {
            halves::<true>(f, g, xa.as_ptr(), ya.as_ptr(), 4, plus);
        } else {
            halves::<false>(f, g, xa.as_ptr(), ya.as_ptr(), 4, plus);
        }
        *f.add(5) = 0;
        *g.add(5) = 0;
        let mut size = trim(f, 5);
        for i in (0..bits - 1).rev() {
            // The pair is `F_j` and `F_(j-1)` for `j = k >> (i + 1)`.
            square(x, f, size, spare);
            square(y, g, size, spare);
            let n2 = 2 * size;
            let plus = (k >> (i + 1)) & 1 == 0;
            if (k >> i) & 1 == 1 {
                halves::<true>(f, g, x, y, n2, plus);
            } else {
                halves::<false>(f, g, x, y, n2, plus);
            }
            // `F_(2j+1)` or `F_(2j)` is the larger: below `4 F_j^2`, of at most one limb more.
            size = trim(f, n2 + 1);
        }
        size
    }
}

/// `L_n` for `n` above the table: the zero bits at the bottom of `n` as squares, from the odd part
/// or from the table.
///
/// The start is in registers when the tables give it: `L_(2j) = L_j^2 - 2 (-1)^j` for the index
/// `j` the table holds, and for an odd one `L_(2k+1) = L_k L_(k+1) - (-1)^k` while `L_(k+1)` is in
/// the table; beyond, `L_(2k+1) = 5 F_(k-1) (2 F_k + F_(k-1)) - 4 (-1)^k`.
#[inline(never)]
fn long_lucas(n: u64, order: ByteOrder) -> Number {
    // The zero bits taken as squares: until the index is odd or in the table.
    let mut m = n;
    let mut zeros = 0u32;
    while m & 1 == 0 && m >= LUC.len() as u64 {
        m >>= 1;
        zeros += 1;
    }
    // The start in registers, and whether its index is odd.
    let start = if m < LUC.len() as u64 {
        // `n` above the table: a zero bit at least. `L_(2m) = L_m^2 - 2 (-1)^m`; a square is 0 or 1
        // modulo 4, so `+ 2` carries nowhere.
        let mut r = square_u128(LUC[m as usize]);
        if m & 1 == 1 {
            r[0] += 2;
        } else {
            // SAFETY: the array holds the four limbs, and the square is above 2.
            unsafe { sub_1(r.as_mut_ptr(), 4, 2) };
        }
        zeros -= 1;
        Some((r, false))
    } else if m / 2 + 1 < LUC.len() as u64 {
        // `m = 2k + 1`.
        let k = (m / 2) as usize;
        let mut r = mul_u128(LUC[k], LUC[k + 1]);
        // SAFETY: as above.
        unsafe {
            if k & 1 == 1 {
                add_1(r.as_mut_ptr(), 4, 1);
            } else {
                sub_1(r.as_mut_ptr(), 4, 1);
            }
        }
        Some((r, true))
    } else {
        None
    };
    if let Some((r, _)) = start
        && zeros == 0
    {
        // SAFETY: the array holds the four limbs.
        return unsafe { limbs_number(r.as_ptr(), 4, false, order) };
    }
    // `L_n < F_(n+2)`: the result has at most the limbs of `F_(n+2)`, and each square's buffer as
    // many with one to spare.
    let ll = even(usize::try_from(fib_limbs(n) + 3).expect("capacity overflow"));
    let need = ll.checked_mul(4).expect("capacity overflow");
    with_memory(need, |w, mut spare| {
        // SAFETY: the working memory holds four buffers of `ll` limbs: `f`, `g`, `x` and `y` for
        // the pair of the odd part and its squares, of which `x` and `y` then take the squares of
        // `L`.
        unsafe {
            let (f, g) = (w, w.add(ll));
            let (mut l, mut y) = (w.add(2 * ll), w.add(3 * ll));
            let mut size;
            // Whether the index of `L` in `l` is odd, for the sign of the next square's 2.
            let mut odd;
            if let Some((r, o)) = start {
                core::ptr::copy_nonoverlapping(r.as_ptr(), l, 4);
                size = trim(l, 4);
                odd = o;
            } else {
                // `m = 2k + 1`: `L_m = 5 F_(k-1) (2 F_k + F_(k-1)) - 4 (-1)^k`, the pair at `k`
                // in `f`, `g`, with `l` and `y` for its squares.
                let k = m / 2;
                let s = if k <= FIB_LAST {
                    put_u128(f, FIB[k as usize]);
                    put_u128(g, FIB[k as usize - 1]);
                    trim(f, 2)
                } else {
                    fib2(f, g, l, y, k, &mut spare)
                };
                let sg = trim(g, s);
                // `2 F_k + F_(k-1)`, of `s + 1` limbs, in place.
                let mut c = 0u128;
                for i in 0..s {
                    let t = (u128::from(*f.add(i)) << 1) + u128::from(*g.add(i)) + c;
                    *f.add(i) = t as u64;
                    c = t >> 64;
                }
                *f.add(s) = c as u64;
                let sf = s + usize::from(c != 0);
                for i in sg..sf {
                    *g.add(i) = 0;
                }
                size = product(l, f, sf, g, sg, &mut spare);
                size = trim(l, size);
                // Times 5.
                let c = mul_1(l, l, size, 5);
                *l.add(size) = c;
                size += usize::from(c != 0);
                if k & 1 == 1 {
                    add_1(l, size, 4);
                } else {
                    sub_1(l, size, 4);
                }
                odd = true;
            }
            for _ in 0..zeros {
                // `L_(2j) = L_j^2 - 2 (-1)^j`.
                square(y, l, size, &mut spare);
                size *= 2;
                size -= usize::from(*y.add(size - 1) == 0);
                if odd {
                    // A square is 0 or 1 modulo 4: no carry.
                    *y += 2;
                    odd = false;
                } else {
                    sub_1(y, size, 2);
                }
                core::mem::swap(&mut l, &mut y);
            }
            limbs_number(l, size, false, order)
        }
    })
}

/// `(F_n, L_n)`, in the byte order `order`: the pair at `n` by [`fib2`], and
/// `L_n = F_n + 2 F_(n-1)`.
pub(in super::super) fn fibonacci_and_lucas(n: u64, order: ByteOrder) -> (Number, Number) {
    if n < LUC.len() as u64 {
        let i = n as usize;
        return (
            small_number(order, FIB[i], false),
            small_number(order, LUC[i], false),
        );
    }
    // `L_n < F_(n+2)`, which has at most a limb more than `F_n`.
    let fl = even(usize::try_from(fib_limbs(n) + 2).expect("capacity overflow"));
    let need = fl.checked_mul(4).expect("capacity overflow");
    with_memory(need, |w, mut spare| {
        // SAFETY: the working memory holds the four buffers of `fl` limbs of `fib2`.
        unsafe {
            let (f, g, x, y) = (w, w.add(fl), w.add(2 * fl), w.add(3 * fl));
            let s = fib2(f, g, x, y, n, &mut spare);
            *x.add(s) = lucas_of(x, f, g, s, y);
            (
                limbs_number(f, s, false, order),
                limbs_number(x, s + 1, false, order),
            )
        }
    })
}
