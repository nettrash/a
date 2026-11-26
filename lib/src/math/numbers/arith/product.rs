//! Products of a few numbers in limbs, for the arithmetic functions of a factorisation.
//!
//! The functions of a factorisation are products of a few factors each: `p - 1`, `p + 1`, `p` and
//! its powers. Taken as numbers, each factor is a new vector and each product reads both operands'
//! bytes into limbs and writes the result back, which costs more than the products of numbers of
//! a few limbs themselves. Here the product is kept in limbs, in a buffer on the stack, each factor
//! read straight from its bytes into limbs beside it, multiplied in by the schoolbook kernels, and
//! the number written once at the end.

use super::super::modular::ring::{from_limbs, read_limbs};
use super::super::mul::limbs::mul;
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
use core::mem::MaybeUninit;

/// The limbs a product may reach: below twice Karatsuba's threshold (24 limbs), so that the shorter
/// operand of every product is below that, and the schoolbook kernels take it without working
/// memory. Longer products are taken as numbers, up a balanced tree.
pub(super) const LIMBS: usize = 46;

/// Limbs on the stack, aligned to 16 bytes for the kernels, which load and store them in pairs.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Buf([MaybeUninit<u64>; LIMBS]);

/// A product in limbs, not zero: `len` limbs of `bufs[cur]`, the least significant first, the top
/// one not zero; the other buffer takes the next product.
pub(super) struct Product {
    bufs: [Buf; 2],
    cur: usize,
    len: usize,
}

impl Product {
    /// The product of no factors, 1.
    #[inline(always)]
    pub(super) fn one() -> Product {
        let mut p = Product {
            bufs: [Buf([MaybeUninit::uninit(); LIMBS]); 2],
            cur: 0,
            len: 1,
        };
        p.bufs[0].0[0] = MaybeUninit::new(1);
        p
    }

    /// Sets the product to the limbs `x`, the top one not zero, of at most [`LIMBS`].
    #[inline(always)]
    pub(super) fn set(&mut self, x: &[u64]) {
        // Checked: `limbs` reads the `len` limbs set here, which the buffer must hold, and the
        // kernel of `mul` needs at least one.
        assert!(!x.is_empty() && x.len() <= LIMBS);
        for (o, &l) in self.bufs[0].0.iter_mut().zip(x) {
            *o = MaybeUninit::new(l);
        }
        self.cur = 0;
        self.len = x.len();
    }

    /// Sets the product to 1.
    #[inline(always)]
    pub(super) fn set_one(&mut self) {
        self.bufs[0].0[0] = MaybeUninit::new(1);
        self.cur = 0;
        self.len = 1;
    }

    /// Whether a factor of `k` more limbs fits.
    #[inline(always)]
    pub(super) fn room(&self, k: usize) -> bool {
        self.len + k <= LIMBS
    }

    /// The limbs.
    #[inline(always)]
    fn limbs(&self) -> &[u64] {
        // SAFETY: the first `len` limbs of the current buffer are written.
        unsafe { core::slice::from_raw_parts(self.bufs[self.cur].0.as_ptr().cast(), self.len) }
    }

    /// Multiplies by the limbs `f`, the top one not zero, for which [`room`](Self::room) holds.
    #[inline]
    pub(super) fn mul(&mut self, f: &[u64]) {
        // Checked: the kernel below writes `len + f.len()` limbs into a buffer of `LIMBS`, and
        // needs operands of a limb at least.
        assert!(!f.is_empty() && self.room(f.len()));
        debug_assert!(f[f.len() - 1] != 0);
        if let [w] = f {
            self.mul_word(*w);
            return;
        }
        if self.len == 1 {
            // SAFETY: the first limb is written.
            let w = unsafe { self.bufs[self.cur].0[0].assume_init() };
            self.set(f);
            if w != 1 {
                self.mul_word(w);
            }
            return;
        }
        let (n, k) = (self.len, f.len());
        let [b0, b1] = &mut self.bufs;
        let (x, r) = if self.cur == 0 {
            (&*b0, b1)
        } else {
            (&*b1, b0)
        };
        let x: *const u64 = x.0.as_ptr().cast();
        let r: *mut u64 = r.0.as_mut_ptr().cast();
        let ((p, np), (q, nq)) = if n >= k {
            ((x, n), (f.as_ptr(), k))
        } else {
            ((f.as_ptr(), k), (x, n))
        };
        // SAFETY: the shorter operand has fewer than 24 limbs, as `n + k <= LIMBS`, so the product
        // is the schoolbook kernels', which need no working memory; the result's `n + k` limbs fit
        // in the other buffer, apart from both operands.
        unsafe { mul(r, p, np, q, nq, Scratch::none()) };
        self.cur ^= 1;
        // SAFETY: the kernel wrote the `n + k` limbs.
        let top = unsafe { *r.add(n + k - 1) };
        self.len = n + k - usize::from(top == 0);
    }

    /// Multiplies by the word `w`, not zero, for which [`room`](Self::room)`(1)` holds.
    #[inline]
    pub(super) fn mul_word(&mut self, w: u64) {
        debug_assert!(w != 0 && self.room(1));
        let buf = &mut self.bufs[self.cur].0;
        let mut carry = 0u64;
        for l in &mut buf[..self.len] {
            // SAFETY: the first `len` limbs are written.
            let x = unsafe { l.assume_init() };
            let p = u128::from(x) * u128::from(w) + u128::from(carry);
            *l = MaybeUninit::new(p as u64);
            carry = (p >> 64) as u64;
        }
        if carry != 0 {
            buf[self.len] = MaybeUninit::new(carry);
            self.len += 1;
        }
    }

    /// Adds the word `w`, for which [`room`](Self::room)`(1)` holds.
    #[inline]
    pub(super) fn add_word(&mut self, w: u64) {
        debug_assert!(self.room(1));
        let buf = &mut self.bufs[self.cur].0;
        let mut carry = w;
        for l in &mut buf[..self.len] {
            if carry == 0 {
                return;
            }
            // SAFETY: the first `len` limbs are written.
            let (s, c) = unsafe { l.assume_init() }.overflowing_add(carry);
            *l = MaybeUninit::new(s);
            carry = u64::from(c);
        }
        if carry != 0 {
            buf[self.len] = MaybeUninit::new(carry);
            self.len += 1;
        }
    }

    /// Multiplies by `p^e` for the limbs `p` of a number at least 2, the top one not zero: a word's
    /// powers in pieces of a word each, a longer number one product at a time. `false` when the
    /// product does not fit, and it is then left part done.
    #[inline]
    pub(super) fn mul_power(&mut self, p: &[u64], e: u32) -> bool {
        if let [w] = *p {
            // `w^j`, the largest power below `2^64`, and the products by it.
            let (mut piece, mut j) = (w, 1u32);
            while j < e {
                match piece.checked_mul(w) {
                    Some(x) => {
                        piece = x;
                        j += 1;
                    }
                    None => break,
                }
            }
            let mut left = e;
            while left >= j {
                if !self.room(1) {
                    return false;
                }
                self.mul_word(piece);
                left -= j;
            }
            if left > 0 {
                if !self.room(1) {
                    return false;
                }
                self.mul_word(w.pow(left));
            }
            return true;
        }
        for _ in 0..e {
            if !self.room(p.len()) {
                return false;
            }
            self.mul(p);
        }
        true
    }

    /// The limbs, the least significant first.
    #[inline(always)]
    pub(super) fn as_limbs(&self) -> &[u64] {
        self.limbs()
    }

    /// The product as a number in the byte order `order`, in the shortest form.
    #[inline]
    pub(super) fn number(&self, order: ByteOrder) -> Number {
        from_limbs(self.limbs(), order)
    }
}

/// The limbs of a number of at most [`LIMBS`] limbs, read from its bytes: `len` of them, the least
/// significant first, the top one not zero.
pub(super) struct Limbs {
    buf: Buf,
    len: usize,
}

impl Limbs {
    /// No limbs yet.
    #[inline(always)]
    pub(super) fn new() -> Limbs {
        Limbs {
            buf: Buf([MaybeUninit::uninit(); LIMBS]),
            len: 0,
        }
    }

    /// Reads the magnitude of `x`, not zero, in place of the limbs before; `false` when it passes
    /// [`LIMBS`] limbs. In place, so that the buffer is never moved.
    #[inline]
    pub(super) fn read(&mut self, x: &Number) -> bool {
        let n = x.significant();
        if n > 8 * LIMBS || n == 0 {
            return false;
        }
        self.len = n.div_ceil(8);
        // `read_limbs` writes the first `len` limbs, which is all that is read.
        read_limbs(x, &mut self.buf.0[..self.len]);
        true
    }

    /// The limbs.
    #[inline(always)]
    pub(super) fn as_slice(&self) -> &[u64] {
        // SAFETY: the first `len` limbs are written.
        unsafe { core::slice::from_raw_parts(self.buf.0.as_ptr().cast(), self.len) }
    }

    /// Subtracts 1, from a number above 1 (so that it stays not zero).
    #[inline]
    pub(super) fn decrement(&mut self) {
        for l in &mut self.buf.0[..self.len] {
            // SAFETY: the first `len` limbs are written.
            let (d, borrow) = unsafe { l.assume_init() }.overflowing_sub(1);
            *l = MaybeUninit::new(d);
            if !borrow {
                break;
            }
        }
        // SAFETY: as above.
        if self.len > 1 && unsafe { self.buf.0[self.len - 1].assume_init() } == 0 {
            self.len -= 1;
        }
    }
}
