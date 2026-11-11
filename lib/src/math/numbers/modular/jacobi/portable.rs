//! The portable kernels of the Jacobi symbol, for Miri and for the targets without an assembly
//! kernel.

/// Binary steps on the odd numbers `u = u1 B + u0` and `v = v1 B + v0` while either has a second
/// limb, the symbol `(u / v)` carried in bit 1 of `bits`: the larger less the smaller, which takes
/// the place of the larger, the smaller the denominator (reciprocity turns the sign when both are 3
/// modulo 4), and the difference's factors of two shifted out (each turns it when the denominator
/// is 3 or 5 modulo 8). The magnitude of a negative difference is the complement of its high limb
/// and the negation of its low one, whose zeros are the shift; a low limb of zero, once in `2^64`,
/// is taken apart. Returns the numbers, of one limb each, and the bits; `None` when they meet on
/// two limbs, a common factor.
#[inline]
pub(super) fn steps_22(
    mut u0: u64,
    mut u1: u64,
    mut v0: u64,
    mut v1: u64,
    mut bits: u64,
) -> Option<(u64, u64, u64)> {
    debug_assert!(u0 & v0 & 1 == 1);
    while u1 | v1 != 0 {
        let (t0, b0) = u0.overflowing_sub(v0);
        let (t1, b1) = u1.overflowing_sub(v1);
        let (t1, b2) = t1.overflowing_sub(u64::from(b0));
        let below = b1 | b2;
        if t0 == 0 {
            if t1 == 0 {
                return None;
            }
            bits ^= if below { u0 & v0 } else { 0 };
            if below {
                (v1, v0) = (u1, u0);
            }
            let a = if below { t1.wrapping_neg() } else { t1 };
            let c = a.trailing_zeros();
            (u1, u0) = (0, a >> c);
            bits ^= u64::from(c & 1) * (v0 ^ (v0 >> 1));
            continue;
        }
        bits ^= if below { u0 & v0 } else { 0 };
        v0 = core::hint::select_unpredictable(below, u0, v0);
        v1 = core::hint::select_unpredictable(below, u1, v1);
        let a0 = if below { t0.wrapping_neg() } else { t0 };
        let a1 = if below { !t1 } else { t1 };
        let c = t0.trailing_zeros();
        u0 = (a0 >> c) | (a1 << (64 - c));
        u1 = a1 >> c;
        bits ^= u64::from(c & 1) * (v0 ^ (v0 >> 1));
    }
    Some((u0, v0, bits))
}

/// [`steps_22`]'s binary steps on numbers of `N` limbs, `u` and `v` odd, while either has its top
/// limb: they then fit in `N - 1`. The magnitude of a negative difference is the complement of its
/// limbs above the lowest and the negation of the lowest, when that is not zero; a lowest limb of
/// zero, once in `2^64`, is shifted out apart. `false` when the numbers meet, a common factor.
#[inline]
pub(in super::super) fn steps_nn<const N: usize>(
    u: &mut [u64; N],
    v: &mut [u64; N],
    bits: &mut u64,
) -> bool {
    debug_assert!(u[0] & v[0] & 1 == 1);
    while u[N - 1] | v[N - 1] != 0 {
        let mut t = [0u64; N];
        let mut borrow = false;
        for i in 0..N {
            let (d, b1) = u[i].overflowing_sub(v[i]);
            let (d, b2) = d.overflowing_sub(u64::from(borrow));
            t[i] = d;
            borrow = b1 | b2;
        }
        let below = borrow;
        *bits ^= if below { u[0] & v[0] } else { 0 };
        if below {
            *v = *u;
        }
        if t[0] == 0 {
            // The rare case: the magnitude in full, then shifted a limb at a time.
            if t.iter().all(|&l| l == 0) {
                return false;
            }
            if below {
                let mut carry = true;
                for l in t.iter_mut() {
                    let (s, c) = (!*l).overflowing_add(u64::from(carry));
                    *l = s;
                    carry = c;
                }
            }
            let mut z = 0u32;
            while t[0] == 0 {
                t.copy_within(1.., 0);
                t[N - 1] = 0;
                z += 64;
            }
            let c = t[0].trailing_zeros();
            for i in 0..N {
                let hi = if i + 1 < N && c > 0 {
                    t[i + 1] << (64 - c)
                } else {
                    0
                };
                u[i] = (t[i] >> c) | hi;
            }
            *bits ^= u64::from((z + c) & 1) * (v[0] ^ (v[0] >> 1));
            continue;
        }
        if below {
            t[0] = t[0].wrapping_neg();
            for l in t.iter_mut().skip(1) {
                *l = !*l;
            }
        }
        let c = t[0].trailing_zeros();
        for i in 0..N - 1 {
            u[i] = (t[i] >> c) | (t[i + 1] << (64 - c));
        }
        u[N - 1] = t[N - 1] >> c;
        *bits ^= u64::from(c & 1) * (v[0] ^ (v[0] >> 1));
    }
    true
}

/// [`steps_nn`] on three limbs.
#[inline]
pub(super) fn steps_3(u: &mut [u64; 3], v: &mut [u64; 3], bits: &mut u64) -> bool {
    steps_nn(u, v, bits)
}

/// [`steps_nn`] on four limbs.
#[inline]
pub(super) fn steps_4(u: &mut [u64; 4], v: &mut [u64; 4], bits: &mut u64) -> bool {
    steps_nn(u, v, bits)
}

/// [`steps_nn`] on five limbs.
#[inline]
pub(super) fn steps_5(u: &mut [u64; 5], v: &mut [u64; 5], bits: &mut u64) -> bool {
    steps_nn(u, v, bits)
}

/// [`steps_nn`] on six limbs.
#[inline]
pub(super) fn steps_6(u: &mut [u64; 6], v: &mut [u64; 6], bits: &mut u64) -> bool {
    steps_nn(u, v, bits)
}
