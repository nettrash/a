//! The aarch64 kernels of the greatest common divisor, in inline assembly: the quotient steps of
//! [`hgcd2`](super::limbs::hgcd2).
//!
//! A quotient below 8 is three conditional subtractions, of `4 d`, `2 d` and `d`: `subs` and
//! `csel` a bit on one limb, two instructions deep, and `subs`, `sbcs` and two `csel`s on two,
//! three deep; the quotient's bits come from the carry flag, off the path from one remainder to
//! the next (`cset`, then `adc` doubling). The compiler turned the same selects in Rust into a
//! comparison, a select of the subtrahend and a subtraction, three deep on one limb: measured on
//! an Apple M4 Max, `hgcd2` took 630 cycles a call so, against 560.

use crate::math::numbers::gcd::limbs::gcd_11;
use core::arch::asm;

// The portable kernels, of which the quotient steps are replaced here.
#[allow(dead_code)]
#[path = "portable.rs"]
mod portable;

pub(super) use portable::{addmul_1, extend_row, reduce_pair};

/// The quotient and the remainder of `n` by `d`, for `n < 8 d` and `d < 2^61`.
#[inline(always)]
pub(super) fn small_q1(n: u64, d: u64) -> (u64, u64) {
    let (r, q): (u64, u64);
    // SAFETY: the assembly reads and writes registers only, and names every one it writes.
    unsafe {
        asm!(
            "subs {t}, {r}, {d}, lsl #2",
            "csel {r}, {t}, {r}, hs",
            "cset {q}, hs",
            "subs {t}, {r}, {d}, lsl #1",
            "csel {r}, {t}, {r}, hs",
            "adc {q}, {q}, {q}",
            "subs {t}, {r}, {d}",
            "csel {r}, {t}, {r}, hs",
            "adc {q}, {q}, {q}",
            r = inout(reg) n => r, d = in(reg) d, t = out(reg) _, q = out(reg) q,
            options(pure, nomem, nostack),
        );
    }
    (q, r)
}

/// [`small_q1`] on two limbs, for `n < 8 d` and `d < 2^125`.
#[inline(always)]
pub(super) fn small_q2(n: u128, d: u128) -> (u64, u128) {
    let (mut lo, mut hi) = (n as u64, (n >> 64) as u64);
    let (dl, dh) = (d as u64, (d >> 64) as u64);
    let q: u64;
    // SAFETY: as in `small_q1`.
    unsafe {
        asm!(
            "extr {h4}, {dh}, {dl}, #62",
            "lsl {l4}, {dl}, #2",
            "extr {h2}, {dh}, {dl}, #63",
            "lsl {l2}, {dl}, #1",
            "subs {t0}, {lo}, {l4}",
            "sbcs {t1}, {hi}, {h4}",
            "csel {lo}, {t0}, {lo}, hs",
            "csel {hi}, {t1}, {hi}, hs",
            "cset {q}, hs",
            "subs {t0}, {lo}, {l2}",
            "sbcs {t1}, {hi}, {h2}",
            "csel {lo}, {t0}, {lo}, hs",
            "csel {hi}, {t1}, {hi}, hs",
            "adc {q}, {q}, {q}",
            "subs {t0}, {lo}, {dl}",
            "sbcs {t1}, {hi}, {dh}",
            "csel {lo}, {t0}, {lo}, hs",
            "csel {hi}, {t1}, {hi}, hs",
            "adc {q}, {q}, {q}",
            lo = inout(reg) lo, hi = inout(reg) hi, dl = in(reg) dl, dh = in(reg) dh,
            h4 = out(reg) _, l4 = out(reg) _, h2 = out(reg) _, l2 = out(reg) _,
            t0 = out(reg) _, t1 = out(reg) _, q = out(reg) q,
            options(pure, nomem, nostack),
        );
    }
    (q, (u128::from(hi) << 64) | u128::from(lo))
}

/// The greatest common divisor of the odd `u` and `v` of up to two limbs, by the binary method: on
/// two limbs while either number has a second one, then by [`gcd_11`].
///
/// The difference of two odd numbers is even and not zero, and its low limb is zero only when the
/// low limbs agree, a rare case taken apart: so the shift is the low limb's trailing zeros, below
/// 64, and the magnitude of a negative difference is the complement of its high limb and the
/// negation of its low one, with no carry between them. A step is then six instructions deep, as
/// GMP's own: measured on an Apple M4 Max, 486 cycles a call on random numbers, against 485 for
/// GMP 6.3's `mpn_gcd_22` and 687 for the same steps on `u128`s in Rust, where the compiler shifts
/// and negates all 128 bits and saves the flags between them.
#[inline]
pub(super) fn gcd_22(u: u128, v: u128) -> u128 {
    debug_assert!(u & v & 1 == 1);
    let (mut u1, mut u0, mut v1, mut v0) = ((u >> 64) as u64, u as u64, (v >> 64) as u64, v as u64);
    if u1 | v1 != 0 {
        // SAFETY: the assembly reads and writes registers only, and names every one it writes.
        unsafe {
            asm!(
                "2:",
                "subs {t0}, {u0}, {v0}",
                "sbcs {t1}, {u1}, {v1}",
                "cbz {t0}, 4f",
                "rbit {c}, {t0}",
                "csel {v0}, {u0}, {v0}, lo",
                "csel {v1}, {u1}, {v1}, lo",
                "cneg {t0}, {t0}, lo",
                "cinv {t1}, {t1}, lo",
                "clz {c}, {c}",
                "neg {n}, {c}",
                "lsr {u0}, {t0}, {c}",
                "lsl {x}, {t1}, {n}",
                "lsr {u1}, {t1}, {c}",
                "orr {u0}, {u0}, {x}",
                "orr {x}, {u1}, {v1}",
                "cbnz {x}, 2b",
                "b 6f",
                // The low limbs agree: the difference is the high limbs' alone.
                "4:",
                "cbz {t1}, 5f",
                "csel {v1}, {u1}, {v1}, lo",
                "cneg {t1}, {t1}, lo",
                "rbit {c}, {t1}",
                "clz {c}, {c}",
                "lsr {u0}, {t1}, {c}",
                "mov {u1}, #0",
                "cbnz {v1}, 2b",
                "b 6f",
                // The numbers are equal: the greatest common divisor is either.
                "5:",
                "mov {v0}, {u0}",
                "mov {v1}, {u1}",
                "6:",
                u0 = inout(reg) u0, u1 = inout(reg) u1, v0 = inout(reg) v0, v1 = inout(reg) v1,
                t0 = out(reg) _, t1 = out(reg) _, c = out(reg) _, n = out(reg) _, x = out(reg) _,
                options(pure, nomem, nostack),
            );
        }
        if u1 | v1 != 0 {
            // Equal numbers of two limbs.
            return u128::from(u0) | (u128::from(u1) << 64);
        }
    }
    u128::from(gcd_11(u0, v0))
}
