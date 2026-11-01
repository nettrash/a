//! The arithmetic modulo a prime just below a power of two.

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use super::aarch64::{
    p521_mul, pseudo_add_4, pseudo_add_6, pseudo_mul_4, pseudo_sqr_4, pseudo_sub_4, pseudo_sub_6,
};
use super::{Field, array_of, number_of};
use crate::math::numbers::{Number, basecase};
use core::mem::MaybeUninit;

/// The field modulo a prime `p = 2^k - c` of `N` limbs, for `k = 64 N - s` and a small `c`, so that
/// `2^(64 N) = c 2^s = f` modulo `p`, one limb: the high half of a product is folded back onto the
/// low half by a product by `f`.
///
/// An element is any value of the `N` limbs, not always below `p`: a sum or a product comes back
/// into them with a fold or two and no comparison, and only [`canonical`](Pseudo::canonical)
/// makes it unique.
pub(in super::super) struct Pseudo<const N: usize> {
    /// `c`.
    c: u64,
    /// `f = c 2^s`, below `2^62`, so that the folds can't carry twice.
    f: u64,
    /// `s`, below 64.
    s: u32,
    /// The prime as a number.
    p: Number,
}

impl<const N: usize> Pseudo<N> {
    /// The field modulo the prime `p` with the limbs `m`, when `p` has this shape.
    pub(in super::super) fn new(p: &Number) -> Option<Self> {
        let k = p.bits() as usize;
        let s = (64 * N - k) as u32;
        let c = (Number::new_from_u8(1, p.byte_order) << k) - p;
        let c = limbs_low(&c)?;
        if s >= 62 || c >= 1 << (62 - s) {
            return None;
        }
        Some(Pseudo {
            c,
            f: c << s,
            s,
            p: p.clone(),
        })
    }

    /// The unique element congruent to `a`, below `p`.
    #[inline(always)]
    fn canonical(&self, a: &[u64; N]) -> [u64; N] {
        let mut r = *a;
        if self.s != 0 {
            // `a = t 2^k + r = r + t c`, which stays below `2^k + f`.
            let t = r[N - 1] >> (64 - self.s);
            r[N - 1] &= u64::MAX >> self.s;
            r = add_1(&r, t * self.c).0;
        }
        // At least `p` exactly when adding `c` reaches `2^k`, and then the sum less `2^k` is the
        // difference.
        let (mut b, carry) = add_1(&r, self.c);
        let reached = if self.s == 0 {
            carry
        } else {
            let top = b[N - 1] >> (64 - self.s) != 0;
            b[N - 1] &= u64::MAX >> self.s;
            top
        };
        let mask = u64::from(reached).wrapping_neg();
        for i in 0..N {
            r[i] = (b[i] & mask) | (r[i] & !mask);
        }
        r
    }

    /// `a + b`, on the targets without kernels.
    #[cfg_attr(
        all(target_arch = "aarch64", target_endian = "little", not(miri)),
        allow(dead_code)
    )]
    #[inline(always)]
    fn portable_add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        // `2^(64 N) = f`, and a second carry leaves less than `f` to add it to.
        let (s, carry) = add_n(a, b);
        let (s, carry) = add_1(&s, self.f & u64::from(carry).wrapping_neg());
        let mut r = s;
        r[0] = r[0].wrapping_add(self.f & u64::from(carry).wrapping_neg());
        r
    }

    /// `a - b`, on the targets without kernels.
    #[cfg_attr(
        all(target_arch = "aarch64", target_endian = "little", not(miri)),
        allow(dead_code)
    )]
    #[inline(always)]
    fn portable_sub(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        // A borrow took `2^(64 N) = f` too much; a second one leaves too much to borrow a third
        // time, but its `f` can still borrow across the low limb.
        let mut f = [0; N];
        let (d, borrow) = sub_n(a, b);
        f[0] = self.f & u64::from(borrow).wrapping_neg();
        let (d, borrow) = sub_n(&d, &f);
        f[0] = self.f & u64::from(borrow).wrapping_neg();
        sub_n(&d, &f).0
    }

    /// `t` modulo `p`, in `N` limbs: `lo + hi f`, then the limb above it times `f` once more, whose
    /// carry out, if any, leaves too little for one more `f` to carry past the second limb.
    #[inline(always)]
    fn fold<const W: usize>(&self, t: &[u64; W]) -> [u64; N] {
        if self.c == 1 {
            return self.fold_shifted(t);
        }
        let f = self.f;
        let mut r = [0u64; N];
        let mut c = 0u64;
        for i in 0..N {
            let x = u128::from(t[N + i]) * u128::from(f) + u128::from(t[i]) + u128::from(c);
            r[i] = x as u64;
            c = (x >> 64) as u64;
        }
        let x = u128::from(c) * u128::from(f);
        let (a, carry) = add_2(&r, x as u64, (x >> 64) as u64);
        let (a, _) = add_2(&a, f & u64::from(carry).wrapping_neg(), 0);
        a
    }
}

impl<const N: usize> Pseudo<N> {
    /// [`fold`](Pseudo::fold) for `c = 1`, where `f` is the power `2^s`: shifts instead of products,
    /// which measured on an Apple M4 Max took P-521's doubling from 1,447 to 1,326 cycles before
    /// that field had a kernel.
    #[inline(always)]
    fn fold_shifted<const W: usize>(&self, t: &[u64; W]) -> [u64; N] {
        let s = self.s;
        // `lo + hi 2^s`: each limb of `hi`, shifted, spills `64 - s` bits into the next.
        let mut r = [0u64; N];
        let mut carry = false;
        let mut spill = 0u64;
        for i in 0..N {
            let x = (t[N + i] << s) | spill;
            spill = t[N + i] >> (64 - s);
            let (y, c1) = t[i].overflowing_add(x);
            let (y, c2) = y.overflowing_add(u64::from(carry));
            r[i] = y;
            carry = c1 | c2;
        }
        // The top: `spill + carry`, at most `2^s`, again times `2^s`, in two limbs.
        let top = spill + u64::from(carry);
        let (a, carry) = add_2(&r, top << s, top >> (64 - s));
        add_2(&a, self.f & u64::from(carry).wrapping_neg(), 0).0
    }
}

/// The low limb of `x`, if it has no others.
fn limbs_low(x: &Number) -> Option<u64> {
    let l = super::limbs_of(x);
    match l.len() {
        0 => Some(0),
        1 => Some(l[0]),
        _ => None,
    }
}

/// `a + x` and the carry out.
#[inline(always)]
fn add_1<const N: usize>(a: &[u64; N], x: u64) -> ([u64; N], bool) {
    add_2(a, x, 0)
}

/// `a + x0 + x1 2^64` and the carry out.
#[inline(always)]
fn add_2<const N: usize>(a: &[u64; N], x0: u64, x1: u64) -> ([u64; N], bool) {
    let mut r = *a;
    let (x, c1) = r[0].overflowing_add(x0);
    r[0] = x;
    let (y, c2) = r[1].overflowing_add(x1);
    let (y, c3) = y.overflowing_add(u64::from(c1));
    r[1] = y;
    let mut carry = c2 | c3;
    for l in r.iter_mut().skip(2) {
        let (x, c) = l.overflowing_add(u64::from(carry));
        *l = x;
        carry = c;
    }
    (r, carry)
}

/// `a + b` and the carry out.
#[inline(always)]
fn add_n<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut r = [0; N];
    let mut carry = false;
    for i in 0..N {
        let (x, c1) = a[i].overflowing_add(b[i]);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        r[i] = x;
        carry = c1 | c2;
    }
    (r, carry)
}

/// `a - b` and the borrow out.
#[inline(always)]
fn sub_n<const N: usize>(a: &[u64; N], b: &[u64; N]) -> ([u64; N], bool) {
    let mut r = [0; N];
    let mut borrow = false;
    for i in 0..N {
        let (x, b1) = a[i].overflowing_sub(b[i]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        r[i] = x;
        borrow = b1 | b2;
    }
    (r, borrow)
}

/// The `2 N` limbs of `a b`.
#[inline(always)]
fn product<const N: usize, const W: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; W] {
    let mut r = [0u64; W];
    for i in 0..N {
        let mut c = 0u64;
        for j in 0..N {
            let x = u128::from(a[j]) * u128::from(b[i]) + u128::from(r[i + j]) + u128::from(c);
            r[i + j] = x as u64;
            c = (x >> 64) as u64;
        }
        r[i + N] = c;
    }
    r
}

/// Generates the products of a field of a given `N`, as arrays of `2 N` limbs can't be written
/// with a generic `N`.
macro_rules! products {
    ($($n:literal => $w:literal),+) => {
        /// The product of a field of `N` limbs.
        pub(in super::super) trait Product<const N: usize> {
            /// `a b`, reduced.
            fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N];

            /// `a^2`, reduced.
            fn sqr(&self, a: &[u64; N]) -> [u64; N];
        }
        $(impl Product<$n> for Pseudo<$n> {
            #[inline(always)]
            fn mul(&self, a: &[u64; $n], b: &[u64; $n]) -> [u64; $n] {
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                if $n == 4 {
                    let mut r = MaybeUninit::<[u64; $n]>::uninit();
                    // SAFETY: each array holds the 4 limbs the kernel takes, all of which it
                    // writes, and `f < 2^62`.
                    return unsafe {
                        pseudo_mul_4(r.as_mut_ptr().cast(), a.as_ptr(), b.as_ptr(), self.f);
                        r.assume_init()
                    };
                }
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                if $n == 9 && self.c == 1 {
                    let mut r = MaybeUninit::<[u64; $n]>::uninit();
                    // SAFETY: each array holds the 9 limbs the kernel takes, all of which it
                    // writes, and the only prime `2^k - 1` of nine limbs is `2^521 - 1`.
                    return unsafe {
                        p521_mul(r.as_mut_ptr().cast(), a.as_ptr(), b.as_ptr());
                        r.assume_init()
                    };
                }
                if $n > 4 {
                    let mut t = MaybeUninit::<[u64; $w]>::uninit();
                    // SAFETY: `t` holds the `2 N` limbs the kernel writes, apart from `a` and `b`.
                    let t = unsafe {
                        basecase::mul(t.as_mut_ptr().cast(), a.as_ptr(), b.as_ptr(), $n);
                        t.assume_init()
                    };
                    return self.fold::<$w>(&t);
                }
                self.fold::<$w>(&product::<$n, $w>(a, b))
            }

            #[inline(always)]
            fn sqr(&self, a: &[u64; $n]) -> [u64; $n] {
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                if $n == 4 {
                    let mut r = MaybeUninit::<[u64; $n]>::uninit();
                    // SAFETY: as in `mul`.
                    return unsafe {
                        pseudo_sqr_4(r.as_mut_ptr().cast(), a.as_ptr(), self.f);
                        r.assume_init()
                    };
                }
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                if $n == 9 && self.c == 1 {
                    let mut r = MaybeUninit::<[u64; $n]>::uninit();
                    // SAFETY: as in `mul`.
                    return unsafe {
                        p521_mul(r.as_mut_ptr().cast(), a.as_ptr(), a.as_ptr());
                        r.assume_init()
                    };
                }
                if $n > 4 {
                    let mut t = MaybeUninit::<[u64; $w]>::uninit();
                    // SAFETY: as in `mul`.
                    let t = unsafe {
                        basecase::sqr(t.as_mut_ptr().cast(), a.as_ptr(), $n);
                        t.assume_init()
                    };
                    return self.fold::<$w>(&t);
                }
                self.fold::<$w>(&product::<$n, $w>(a, a))
            }
        })+
    };
}
products!(4 => 8, 6 => 12, 9 => 18);

impl<const N: usize> Field for Pseudo<N>
where
    Pseudo<N>: Product<N>,
{
    type E = [u64; N];

    #[inline(always)]
    fn limbs(&self) -> usize {
        N
    }

    #[inline(always)]
    fn load(&self, x: &[u64]) -> [u64; N] {
        let mut r = [0; N];
        r.copy_from_slice(&x[..N]);
        r
    }

    #[inline(always)]
    fn store(&self, x: &[u64; N], r: &mut [u64]) {
        r[..N].copy_from_slice(x);
    }

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
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        {
            let mut r = MaybeUninit::<[u64; N]>::uninit();
            let (p, a, b) = (r.as_mut_ptr().cast(), a.as_ptr(), b.as_ptr());
            // SAFETY: the fields have 4, 6 or 9 limbs, so each array holds what the kernel takes,
            // all of which it writes, and `f < 2^62`.
            unsafe {
                match N {
                    4 => pseudo_add_4(p, a, b, self.f),
                    6 => pseudo_add_6(p, a, b, self.f),
                    _ => return self.portable_add(&*a.cast(), &*b.cast()),
                }
                r.assume_init()
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
        self.portable_add(a, b)
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        {
            let mut r = MaybeUninit::<[u64; N]>::uninit();
            let (p, a, b) = (r.as_mut_ptr().cast(), a.as_ptr(), b.as_ptr());
            // SAFETY: as in `add`.
            unsafe {
                match N {
                    4 => pseudo_sub_4(p, a, b, self.f),
                    6 => pseudo_sub_6(p, a, b, self.f),
                    _ => return self.portable_sub(&*a.cast(), &*b.cast()),
                }
                r.assume_init()
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
        self.portable_sub(a, b)
    }

    #[inline(always)]
    fn neg(&self, a: &[u64; N]) -> [u64; N] {
        self.sub(&[0; N], a)
    }
    #[inline(always)]
    fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        Product::mul(self, a, b)
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; N]) -> [u64; N] {
        Product::sqr(self, a)
    }

    #[inline(always)]
    fn mul_word(&self, a: &[u64; N], w: u64, _e: &[u64; N]) -> [u64; N] {
        // One row, and its top limb folded back: at most `f w`, in two limbs.
        let mut r = [0u64; N];
        let mut c = 0u64;
        for i in 0..N {
            let x = u128::from(a[i]) * u128::from(w) + u128::from(c);
            r[i] = x as u64;
            c = (x >> 64) as u64;
        }
        let x = u128::from(c) * u128::from(self.f);
        let (r, carry) = add_2(&r, x as u64, (x >> 64) as u64);
        add_2(&r, self.f & u64::from(carry).wrapping_neg(), 0).0
    }

    #[inline(always)]
    fn is_zero(&self, a: &[u64; N]) -> bool {
        self.canonical(a).iter().fold(0, |x, l| x | l) == 0
    }

    fn enter(&self, x: &Number) -> [u64; N] {
        array_of(&x.rem_euclid(&self.p))
    }

    fn leave(&self, x: &[u64; N]) -> Number {
        number_of(&self.canonical(x), self.p.byte_order)
    }

    #[inline(always)]
    fn modulus(&self) -> &Number {
        &self.p
    }
}
