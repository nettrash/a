//! Fibonacci and Lucas numbers by fast doubling, after GMP's `mpz_fib_ui` and `mpz_lucnum_ui`.
//!
//! Below `2^128` they come from tables. Beyond, `F_k` and `F_(k-1)` are doubled from the top bits of
//! the index down, two squares a bit ([`fib2`]):
//!
//! - `F_(2k-1) = F_k^2 + F_(k-1)^2`,
//! - `F_(2k+1) = 4 F_k^2 - F_(k-1)^2 + 2 (-1)^k`,
//! - `F_(2k) = F_(2k+1) - F_(2k-1)`,
//!
//! the pair kept being `F_(2k)` and `F_(2k-1)` for a zero bit and `F_(2k+1)` and `F_(2k)` for a one.
//! The last step of `F_n` takes one product instead of two squares: `F_(2k) = F_k (F_k + 2 F_(k-1))`
//! and `F_(2k+1) = (2 F_k + F_(k-1)) (2 F_k - F_(k-1)) + 2 (-1)^k`. `L_n` takes the zero bits at the
//! bottom of `n` as squares, `L_(2k) = L_k^2 - 2 (-1)^k`, from the odd part `m = 2k + 1`, which is
//! `L_(2k+1) = 5 F_(k-1) (2 F_k + F_(k-1)) - 4 (-1)^k`, or from the table.

use super::super::mul::limbs::{add_1, add_n, mul, mul_itch, sqr, sqr_itch, sub_1, sub_n};
use super::super::mul::{STACK_LIMBS, on_heap, on_stack};
use super::super::root::{limbs_number, lshift, small_number};
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
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

/// `F_n` for `n` above the table: the pair at `n / 2` by [`fib2`], then one product.
#[inline(never)]
fn long_fibonacci(n: u64, order: ByteOrder) -> Number {
    let k = n / 2;
    if k <= FIB_LAST {
        return pair_fibonacci(n, order);
    }

    // The pair's buffers hold `F_k` with a limb to spare; the result twice as many. Counted in
    // 64 bits: on 32-bit targets the limbs of large indices do not fit a `usize`.
    let fl = even(usize::try_from(fib_limbs(k) + 2).expect("capacity overflow"));
    let need = fl.checked_mul(5).expect("capacity overflow");
    with_memory(need, |w, mut spare| {
        // SAFETY: the working memory holds the five buffers of `fl` limbs: the pair `f`, `g`, the
        // square `x`, and the result `r` of `2 fl`.
        unsafe {
            let (f, g, x, r) = (w, w.add(fl), w.add(2 * fl), w.add(3 * fl));
            let s = fib2(f, g, x, k, &mut spare);
            // `F_k` has `s` limbs, the top one not zero; `F_(k-1)` at most as many.
            let len = if n & 1 == 1 {
                // `a = 2 F_k + F_(k-1)` and `b = 2 F_k - F_(k-1)`, of `s + 1` limbs; `b`'s top
                // limb is at most 1.
                let c2 = lshift(x, f, s, 1);
                let c3 = u64::from(add_n(f, x, g, s));
                *f.add(s) = c2 + c3;
                let b = u64::from(sub_n(g, x, g, s));
                *g.add(s) = c2 - b;
                let sa = s + usize::from(*f.add(s) != 0);
                let sb = s + usize::from(*g.add(s) != 0);
                let sb = trim(g, sb);
                let t = spare.get(mul_itch(sa, sb));
                mul(r, f, sa, g, sb, t);
                let len = sa + sb;
                // `+ 2 (-1)^k`: the product is above 2.
                if k & 1 == 0 {
                    add_1(r, len, 2);
                } else {
                    sub_1(r, len, 2);
                }
                len
            } else {
                // `F_k + 2 F_(k-1)`, of `s + 1` limbs, times `F_k`.
                let c = lshift(g, g, s, 1);
                let c = c + u64::from(add_n(g, g, f, s));
                *g.add(s) = c;
                let sg = s + usize::from(c != 0);
                let t = spare.get(mul_itch(sg, s));
                mul(r, g, sg, f, s, t);
                sg + s
            };
            limbs_number(r, len, false, order)
        }
    })
}

/// `F_n` for `n / 2` in the table: its last step from the table's pair, on the stack.
fn pair_fibonacci(n: u64, order: ByteOrder) -> Number {
    let k = n / 2;
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

/// The `n` limbs at `a` without the zero limbs at the top, at least one.
///
/// # Safety
///
/// The limbs must be valid for reads.
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

/// Sets the limbs at `f` to `F_k` and those at `g` to `F_(k-1)`, for `k` at least 2, and returns
/// their number: the top one of `F_k` is not zero, that of `F_(k-1)` may be.
///
/// The pair starts at the top bits of `k` that the table holds, and is doubled for each bit below.
/// The square of `F_k` goes to `x`.
///
/// # Safety
///
/// The buffers `f`, `g` and `x` must be valid for writing [`fib_limbs`]`(k) + 2` limbs each, apart
/// from each other and from the spare memory.
unsafe fn fib2(f: *mut u64, g: *mut u64, x: *mut u64, k: u64, spare: &mut Spare) -> usize {
    debug_assert!(k >= 2);
    // The bits of `k` below the start: `k >> bits` is at most the table's last index.
    let bits = 64 - (k / (FIB_LAST + 1)).leading_zeros();
    let first = k >> bits;
    // SAFETY: as the caller guarantees: each pair has at most the limbs of `F_k`, plus one for a
    // carry out of the top, and each square at most twice those of its operand.
    unsafe {
        let (a, b) = (FIB[first as usize], FIB[first as usize - 1]);
        *f = a as u64;
        *f.add(1) = (a >> 64) as u64;
        *g = b as u64;
        *g.add(1) = (b >> 64) as u64;
        // `F_93` and the ones below take one limb.
        let mut size = trim(f, 2);
        for i in (0..bits).rev() {
            // The pair is `F_j` and `F_(j-1)` for `j = k >> (i + 1)`.
            let j_odd = (k >> (i + 1)) & 1 == 1;
            let t = spare.get(sqr_itch(size));
            sqr(x, f, size, t);
            let t = spare.get(sqr_itch(size));
            sqr(f, g, size, t);
            size *= 2;
            // `F_j` has no zero limb at the top: its square at most one.
            size -= usize::from(*x.add(size - 1) == 0);
            // `F_(2j-1) = F_j^2 + F_(j-1)^2`.
            *g.add(size) = u64::from(add_n(g, x, f, size));
            // `F_(2j+1) = 4 F_j^2 - F_(j-1)^2 + 2 (-1)^j`: the 2 is or-ed into the low limb of
            // `F_(j-1)^2`, 0 or 1 modulo 4, for an odd `j`, and of `4 F_j^2` for an even one.
            if j_odd {
                *f |= 2;
            }
            let c = lshift(x, x, size, 2);
            if !j_odd {
                *x |= 2;
            }
            let borrow = u64::from(sub_n(f, x, f, size));
            *f.add(size) = c - borrow;
            size += usize::from(*f.add(size) != 0);
            // `F_(2j) = F_(2j+1) - F_(2j-1)`, in place of the one not kept.
            if (k >> i) & 1 == 1 {
                sub_n(g, f, g, size);
            } else {
                sub_n(f, f, g, size);
                // `F_(2j)` can have a zero at the top, and then `F_(2j-1)` too.
                size -= usize::from(*f.add(size - 1) == 0);
            }
        }
        size
    }
}

/// `L_n` for `n` above the table: the zero bits at the bottom of `n` as squares, from the odd part
/// or from the table.
#[inline(never)]
fn long_lucas(n: u64, order: ByteOrder) -> Number {
    // The zero bits taken as squares: until the index is odd or in the table.
    let mut m = n;
    let mut zeros = 0u32;
    while m & 1 == 0 && m >= LUC.len() as u64 {
        m >>= 1;
        zeros += 1;
    }
    // `L_n < F_(n+2)`: the result has at most the limbs of `F_(n+2)`, and each square's buffer as
    // many with one to spare.
    let ll = even(usize::try_from(fib_limbs(n) + 3).expect("capacity overflow"));
    let need = ll.checked_mul(4).expect("capacity overflow");
    with_memory(need, |w, mut spare| {
        // SAFETY: the working memory holds four buffers of `ll` limbs: `l` and `y` for the
        // squares, and two for the pair of the odd part, `f` and `g`, which the first squares
        // reuse.
        unsafe {
            let (mut l, mut y) = (w, w.add(ll));
            let (f, g) = (w.add(2 * ll), w.add(3 * ll));
            let mut size;
            // Whether the index of `L` in `l` is odd, for the sign of the first square's 2.
            let mut odd;
            if m < LUC.len() as u64 {
                let v = LUC[m as usize];
                *l = v as u64;
                *l.add(1) = (v >> 64) as u64;
                size = trim(l, 2);
                odd = m & 1 == 1;
            } else {
                // `m = 2k + 1`: `L_m = 5 F_(k-1) (2 F_k + F_(k-1)) - 4 (-1)^k`, the pair at `k`
                // in `f`, `g`, with `y` for the square.
                let k = m / 2;
                let s = fib2(f, g, y, k, &mut spare);
                let sg = trim(g, s);
                // `2 F_k + F_(k-1)`, of `s + 1` limbs.
                let c = lshift(f, f, s, 1);
                let c = c + u64::from(add_n(f, f, g, s));
                *f.add(s) = c;
                let sf = s + usize::from(c != 0);
                let t = spare.get(mul_itch(sf, sg));
                mul(l, f, sf, g, sg, t);
                size = sf + sg;
                size -= usize::from(*l.add(size - 1) == 0);
                // Times 5: `4 L + L`.
                let c = lshift(y, l, size, 2);
                let c = c + u64::from(add_n(l, l, y, size));
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
                let t = spare.get(sqr_itch(size));
                sqr(y, l, size, t);
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
    let need = fl.checked_mul(3).expect("capacity overflow");
    with_memory(need, |w, mut spare| {
        // SAFETY: the working memory holds the three buffers of `fl` limbs of `fib2`.
        unsafe {
            let (f, g, x) = (w, w.add(fl), w.add(2 * fl));
            let s = fib2(f, g, x, n, &mut spare);
            let c = lshift(x, g, s, 1);
            let c = c + u64::from(add_n(x, x, f, s));
            *x.add(s) = c;
            (
                limbs_number(f, s, false, order),
                limbs_number(x, s + 1, false, order),
            )
        }
    })
}
