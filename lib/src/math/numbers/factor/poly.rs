//! Polynomials modulo `n` for the second stages: products by Kronecker substitution on the crate's
//! multiplication.
//!
//! A polynomial is a vector of its coefficients, the constant first, each the `k` limbs of a
//! residue modulo `n` as the arithmetic holds it ([`Arith::put`](super::field::Arith::put)), `x R
//! mod n`, below `2n`. The product of two is that of two numbers: each polynomial is evaluated at
//! `X = B^s`, its coefficients laid out `s` limbs apart, `s = 2k + 1`, so that a coefficient of the
//! product, a sum of at most `2^62` products below `4 n^2`, fits its `s` limbs and the product of
//! the two numbers holds the coefficients of the product side by side; the multiplication takes
//! the longest of them by its number-theoretic transforms. Each coefficient `t` wanted is then
//! reduced by REDC over `k + 1` limbs, `t / B^(k + 1) mod n`, below `n`, as `t < n B^(k + 1)`: in
//! the held form, the residue of the product's coefficient times `B^-1`, a unit. The methods
//! only multiply the coefficients they get together and take greatest common divisors with `n`,
//! or build polynomials whose roots are what count, so the unit is left in, the same for every
//! coefficient of a product.
//!
//! Products of at most [`SCHOOL`] coefficients in the shorter polynomial are made coefficient by
//! coefficient instead, the products of limbs summed before one REDC each, which gives the same
//! numbers.

use super::super::mul::basecase;
use super::super::mul::limbs::{add_n, cmp_n, mul, mul_itch, sqr, sqr_itch, sub_n};
use super::super::pow::montgomery::{redc_inverse, redc_limbs};
use super::super::scratch::Scratch;
use core::cmp::Ordering;

/// Up to this many coefficients in the shorter polynomial, a product is made coefficient by
/// coefficient.
const SCHOOL: usize = 6;

/// Two limbs aligned to 16 bytes, as the kernels' working memory starts.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Pair([u64; 2]);

/// A buffer of at least `limbs` limbs, aligned to 16 bytes, zeroed when `zero`.
fn buffer(v: &mut Vec<Pair>, limbs: usize, zero: bool) -> *mut u64 {
    let pairs = limbs.div_ceil(2);
    if zero {
        v.clear();
        v.resize(pairs, Pair([0; 2]));
    } else if v.len() < pairs {
        v.resize(pairs, Pair([0; 2]));
    }
    v.as_mut_ptr().cast()
}

/// The products of polynomials modulo the odd `n` of `k` limbs.
pub(super) struct Kron {
    /// The limbs of `n`.
    k: usize,
    /// The limbs between two coefficients of a number, `2k + 1`.
    s: usize,
    /// `n` and a zero limb above it, the modulus of REDC over `k + 1` limbs.
    m: Vec<u64>,
    /// `-1/n mod B^2`.
    minv: [u64; 2],
    /// The two numbers, the product and the multiplication's working memory.
    x: Vec<Pair>,
    y: Vec<Pair>,
    p: Vec<Pair>,
    q: Vec<Pair>,
    work: Vec<Pair>,
    /// A coefficient to reduce, `2k + 2` limbs, and its reduction, `k + 1`.
    u: Vec<u64>,
    r: Vec<u64>,
}

impl Kron {
    /// The products modulo the odd `n > 1` of the limbs `n`, the top one not zero.
    pub(super) fn new(n: &[u64]) -> Kron {
        let k = n.len();
        let mut m = n.to_vec();
        m.push(0);
        Kron {
            k,
            s: 2 * k + 1,
            minv: redc_inverse(m[0], m[1]),
            m,
            x: Vec::new(),
            y: Vec::new(),
            p: Vec::new(),
            q: Vec::new(),
            work: Vec::new(),
            u: vec![0; 2 * k + 2],
            r: vec![0; k + 1],
        }
    }

    /// The limbs of a coefficient.
    #[inline]
    pub(super) fn limbs(&self) -> usize {
        self.k
    }

    /// Sets `out` to the coefficients from `lo` of the product of `a` and `b`, as many as it holds,
    /// each times `B^-1`; coefficients past the product's last are 0. `a` and `b` may be the same
    /// polynomial, which is then squared.
    pub(super) fn mul(&mut self, a: &[u64], b: &[u64], lo: usize, out: &mut [u64]) {
        let k = self.k;
        debug_assert!(
            a.len().is_multiple_of(k) && b.len().is_multiple_of(k) && out.len().is_multiple_of(k)
        );
        let (la, lb) = (a.len() / k, b.len() / k);
        if la == 0 || lb == 0 {
            out.fill(0);
            return;
        }
        if la.min(lb) <= SCHOOL {
            self.school(&[(a, b)], lo, out);
            return;
        }
        let total = product(
            self.k,
            self.s,
            a,
            b,
            &mut self.x,
            &mut self.y,
            &mut self.p,
            &mut self.work,
        );
        let p: *const u64 = self.p.as_ptr().cast();
        self.extract(p, total, la + lb - 1, lo, out);
    }

    /// Sets `out` to the coefficients from `lo` of `a b + c d`, as [`Kron::mul`] does: one REDC a
    /// coefficient for both products. The lengths of `c` and `d` are those of `a` and `b`.
    pub(super) fn mul_add(
        &mut self,
        a: &[u64],
        b: &[u64],
        c: &[u64],
        d: &[u64],
        lo: usize,
        out: &mut [u64],
    ) {
        let k = self.k;
        debug_assert!(a.len() == c.len() && b.len() == d.len());
        let (la, lb) = (a.len() / k, b.len() / k);
        if la == 0 || lb == 0 {
            out.fill(0);
            return;
        }
        if la.min(lb) <= SCHOOL {
            self.school(&[(a, b), (c, d)], lo, out);
            return;
        }
        let total = product(
            self.k,
            self.s,
            a,
            b,
            &mut self.x,
            &mut self.y,
            &mut self.p,
            &mut self.work,
        );
        let other = product(
            self.k,
            self.s,
            c,
            d,
            &mut self.x,
            &mut self.y,
            &mut self.q,
            &mut self.work,
        );
        debug_assert_eq!(total, other);
        let (p, q): (*mut u64, *const u64) = (self.p.as_mut_ptr().cast(), self.q.as_ptr().cast());
        // SAFETY: both hold `total` limbs; a coefficient of the sum, below `8 L n^2`, fits its slot.
        unsafe { add_n(p, p, q, total) };
        self.extract(p, total, la + lb - 1, lo, out);
    }

    /// Reduces the coefficients from `lo` of the product at `p`, `total` limbs of `len` slots, into
    /// `out`.
    fn extract(&mut self, p: *const u64, total: usize, len: usize, lo: usize, out: &mut [u64]) {
        let (k, s) = (self.k, self.s);
        debug_assert!(total >= len * s);
        for (j, c) in out.chunks_exact_mut(k).enumerate() {
            let t = lo + j;
            if t >= len {
                c.fill(0);
                continue;
            }
            // SAFETY: slot `t`, `s` limbs from `t s`, lies within the `total` limbs.
            let slot = unsafe { core::slice::from_raw_parts(p.add(t * s), s) };
            self.u[..s].copy_from_slice(slot);
            self.u[s..].fill(0);
            self.reduce(c);
        }
    }

    /// [`Kron::mul`] or [`Kron::mul_add`] coefficient by coefficient: the products of limbs of
    /// each, of the pairs of polynomials of the same lengths, summed in `2k + 2` limbs, then
    /// reduced.
    fn school(&mut self, pairs: &[(&[u64], &[u64])], lo: usize, out: &mut [u64]) {
        let k = self.k;
        let (la, lb) = (pairs[0].0.len() / k, pairs[0].1.len() / k);
        let mut prod = vec![0u64; 2 * k];
        for (j, c) in out.chunks_exact_mut(k).enumerate() {
            let t = lo + j;
            if t + 1 >= la + lb {
                c.fill(0);
                continue;
            }
            self.u.fill(0);
            let first = t.saturating_sub(lb - 1);
            let last = t.min(la - 1);
            for &(a, b) in pairs {
                for i in first..=last {
                    let (x, y) = (&a[i * k..(i + 1) * k], &b[(t - i) * k..(t - i + 1) * k]);
                    // SAFETY: `k` limbs each, the product's `2k` apart from both; the sum's
                    // `2k + 2` limbs take the carries, as it stays below `8 L n^2`.
                    unsafe {
                        basecase::mul(prod.as_mut_ptr(), x.as_ptr(), y.as_ptr(), k);
                        let u = self.u.as_mut_ptr();
                        if add_n(u, u, prod.as_ptr(), 2 * k) {
                            let (s, carry) = (*u.add(2 * k)).overflowing_add(1);
                            *u.add(2 * k) = s;
                            *u.add(2 * k + 1) += u64::from(carry);
                        }
                    }
                }
            }
            self.reduce(c);
        }
    }

    /// Sets `c` to `u / B^(k + 1) mod n`, below `n`, for the `2k + 2` limbs `u`, below
    /// `n B^(k + 1)`, which it clobbers.
    #[inline]
    fn reduce(&mut self, c: &mut [u64]) {
        let k = self.k;
        let (r, u, m) = (self.r.as_mut_ptr(), self.u.as_mut_ptr(), self.m.as_ptr());
        // SAFETY: `u` holds `2k + 2` limbs, `r` and `m` `k + 1`, all apart; `minv` is
        // `-1/n mod B^2`, as `m` agrees with `n` there.
        unsafe {
            if redc_limbs(r, u, m, k + 1, self.minv) {
                sub_n(r, r, m, k + 1);
            }
            if cmp_n(r, m, k + 1) != Ordering::Less {
                sub_n(r, r, m, k + 1);
            }
        }
        debug_assert!(self.r[k] == 0);
        c.copy_from_slice(&self.r[..k]);
    }
}

/// The product of the polynomials `a` and `b`, of `k` limbs a coefficient, as numbers of
/// coefficients `s` limbs apart, in `p`: the limbs of all its slots, `(la + lb) s`, which it
/// returns, zeros past the product. `x`, `y` and `work` are working memory; `a` and `b` may be
/// the same polynomial, which is then squared.
#[allow(clippy::too_many_arguments)]
fn product(
    k: usize,
    s: usize,
    a: &[u64],
    b: &[u64],
    x: &mut Vec<Pair>,
    y: &mut Vec<Pair>,
    p: &mut Vec<Pair>,
    work: &mut Vec<Pair>,
) -> usize {
    let (la, lb) = (a.len() / k, b.len() / k);
    let square = core::ptr::eq(a, b);
    // The numbers: coefficient `i` at limb `i s`, the last one's `k` limbs at the top.
    let (nx, ny) = ((la - 1) * s + k, (lb - 1) * s + k);
    let xp = buffer(x, nx, true);
    // SAFETY: the buffer holds `nx` zeroed limbs, and coefficient `i`'s `k` limbs end at
    // `i s + k <= nx`.
    unsafe { pack(a, k, s, xp) };
    let yp = if square {
        xp
    } else {
        let yp = buffer(y, ny, true);
        // SAFETY: as for `x`.
        unsafe { pack(b, k, s, yp) };
        yp
    };
    // The product, and zeros to the end of the slots of the top coefficients.
    let total = (la + lb) * s;
    let pp = buffer(p, total, false);
    // SAFETY: the product's `nx + ny <= total` limbs, the operands apart from it, the working
    // memory as their plan says.
    unsafe {
        if square {
            let itch = sqr_itch(nx);
            let w = buffer(work, itch, false);
            sqr(pp, xp, nx, Scratch::new(w, itch));
        } else {
            let (big, nb, small, ns) = if nx >= ny {
                (xp, nx, yp, ny)
            } else {
                (yp, ny, xp, nx)
            };
            let itch = mul_itch(nb, ns);
            let w = buffer(work, itch, false);
            mul(pp, big, nb, small, ns, Scratch::new(w, itch));
        }
        core::ptr::write_bytes(pp.add(nx + ny), 0, total - nx - ny);
    }
    total
}

/// Lays the coefficients of `a`, `k` limbs each, out at `x`, `s` limbs apart.
///
/// # Safety
///
/// `x` holds `(len - 1) s + k` zeroed limbs for the `len` coefficients.
unsafe fn pack(a: &[u64], k: usize, s: usize, x: *mut u64) {
    for (i, c) in a.chunks_exact(k).enumerate() {
        // SAFETY: as the caller guarantees.
        unsafe { core::ptr::copy_nonoverlapping(c.as_ptr(), x.add(i * s), k) };
    }
}
