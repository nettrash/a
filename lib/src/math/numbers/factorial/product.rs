//! Products of many factors: small factors packed into limbs, and limbs multiplied by balanced
//! product trees.
//!
//! A factorial or a binomial coefficient is the product of thousands or millions of small factors.
//! Multiplied one by one into a growing product, `m` limbs of factors would cost `m^2 / 2` limb
//! products; split into halves, each multiplied the same way and the two products then together,
//! the work is that of the multiplication of the halves at each level, `O(M(m) log m)`, where the
//! longest products take Karatsuba's and Toom-Cook's methods. The factors are first packed, as
//! many into each limb as fit, so that the tree starts with fewer, fuller limbs.

use super::super::mul::limbs::{mul, mul_itch, sqr, sqr_itch};
use super::super::root::mul_1;
use super::Limbs;

/// Up to this many limbs, a product is accumulated one limb at a time; above, it is split in two
/// halves. Measured on an Apple M4 Max, 8 to 40 limbs made little difference: `C(10^4, 5000)` took
/// 23,100 cycles with 16 against 23,300 with 24 and 23,800 with 40, and `C(10^6, 200)` 4510
/// against 4600 and 4850, and `1000!` 7450 against 7690 and 7540.
const SEQUENTIAL: usize = 16;

/// Packs factors into limbs: each limb is the product of as many consecutive factors as fit.
pub(super) struct Packer {
    /// The limbs packed so far.
    pub(super) limbs: Limbs,
    /// The product of the factors not yet in a limb.
    product: u64,
    /// The largest product that any factor can still multiply: `u64::MAX` divided by the largest
    /// factor, so that a product of at most it times a factor fits a limb.
    limit: u64,
}

impl Packer {
    /// A packer of factors from 1 to `max`, with room for `capacity` limbs.
    pub(super) fn new(max: u64, capacity: usize) -> Packer {
        Packer {
            limbs: Vec::with_capacity(capacity),
            product: 1,
            limit: u64::MAX / max.max(1),
        }
    }

    /// Multiplies the factor `f`, from 1 to the packer's largest, in: into the open limb when the
    /// product is at most the limit, and otherwise closing it and opening the next with `f`.
    #[inline(always)]
    pub(super) fn push(&mut self, f: u64) {
        if self.product > self.limit {
            self.limbs.push(self.product);
            self.product = f;
        } else {
            self.product *= f;
        }
    }

    /// The limbs, the open one last.
    pub(super) fn finish(mut self) -> Limbs {
        if self.product != 1 || self.limbs.is_empty() {
            self.limbs.push(self.product);
        }
        self.limbs
    }
}

/// The product of the limbs `f`, none of them zero, as a magnitude, in a vector with room for a
/// limb more.
pub(super) fn product(f: &[u64]) -> Limbs {
    let n = f.len();
    if n == 0 {
        return vec![1];
    }
    // The product of `n` limbs has at most `n`.
    let mut r: Limbs = Vec::with_capacity(n + 1);
    let mut scratch = Vec::new();
    // The halves of `product_into` take `m` limbs and those of the larger half after them, at most
    // `m + ceil(m / 2) + ceil(ceil(m / 2) / 2) + ...`, below `2m` and a limb for each level.
    let halves = 2 * n + 64;
    // SAFETY: the vector holds the product's `n` limbs, all written before the length is set to
    // those it has, and the halves' buffer the limbs of `product_into`.
    unsafe {
        let len = if halves <= LOCAL {
            let mut local = [0u64; LOCAL];
            product_into(f, r.as_mut_ptr(), local.as_mut_ptr(), &mut scratch)
        } else {
            let mut h = Vec::<u64>::with_capacity(halves);
            product_into(f, r.as_mut_ptr(), h.as_mut_ptr(), &mut scratch)
        };
        r.set_len(len);
    }
    r
}

/// The limbs of the products of halves up to which they are kept on the stack.
const LOCAL: usize = 128;

/// Sets the limbs at `out` to the product of the `n` limbs `f`, none zero, and returns its number
/// of limbs, at most `n`: one limb at a time up to [`SEQUENTIAL`] limbs, and otherwise the
/// products of the two halves, computed the same way into the first `n` limbs at `halves`, with
/// the rest of them for their own halves, multiplied together with the scratch of `scratch`.
///
/// # Safety
///
/// The limbs at `out` must be valid for writing `n`, and those at `halves` for writing
/// `2n + log2(n)`, apart.
pub(super) unsafe fn product_into(
    f: &[u64],
    out: *mut u64,
    halves: *mut u64,
    scratch: &mut Vec<u64>,
) -> usize {
    let n = f.len();
    // SAFETY: as the caller guarantees: a product of limbs, none zero, takes at most as many
    // limbs as its factors; the halves take `n` limbs at `halves`, and theirs those of the larger
    // half after them.
    unsafe {
        if n <= SEQUENTIAL {
            *out = f[0];
            let mut len = 1;
            for &x in &f[1..] {
                let carry = mul_1(out, out, len, x);
                if carry != 0 {
                    *out.add(len) = carry;
                    len += 1;
                }
            }
            return len;
        }
        let h = n / 2;
        let (a, b) = (halves, halves.add(h));
        let rest = halves.add(n);
        let an = product_into(&f[..h], a, rest, scratch);
        let bn = product_into(&f[h..], b, rest, scratch);
        let ((x, xn), (y, yn)) = if an >= bn {
            ((a, an), (b, bn))
        } else {
            ((b, bn), (a, an))
        };
        let itch = mul_itch(xn, yn);
        if scratch.len() < itch {
            scratch.resize(itch, 0);
        }
        mul(out, x, xn, y, yn, scratch.as_mut_ptr());
        xn + yn - usize::from(*out.add(xn + yn - 1) == 0)
    }
}

/// `x y`, for magnitudes that are not zero.
pub(super) fn times(x: &[u64], y: &[u64]) -> Limbs {
    let (a, b) = if x.len() >= y.len() { (x, y) } else { (y, x) };
    // A limb to spare, for a shift.
    let mut r = Vec::with_capacity(a.len() + b.len() + 1);
    let mut s = vec![0u64; mul_itch(a.len(), b.len())];
    // SAFETY: the vectors hold the operands, the product, all of whose limbs are written before
    // the length is set, and the scratch, all apart; `a` is the longer operand.
    unsafe {
        mul(
            r.as_mut_ptr(),
            a.as_ptr(),
            a.len(),
            b.as_ptr(),
            b.len(),
            s.as_mut_ptr(),
        );
        r.set_len(a.len() + b.len());
    }
    if r.last() == Some(&0) {
        r.pop();
    }
    r
}

/// `x^2`, for a magnitude that is not zero.
pub(super) fn squared(x: &[u64]) -> Limbs {
    let mut r = Vec::with_capacity(2 * x.len() + 1);
    let mut s = vec![0u64; sqr_itch(x.len())];
    // SAFETY: the vectors hold the operand, the square, all of whose limbs are written before the
    // length is set, and the scratch, all apart.
    unsafe {
        sqr(r.as_mut_ptr(), x.as_ptr(), x.len(), s.as_mut_ptr());
        r.set_len(2 * x.len());
    }
    if r.last() == Some(&0) {
        r.pop();
    }
    r
}
