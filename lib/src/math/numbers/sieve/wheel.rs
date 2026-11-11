//! The wheels of the sieve, computed at compile time.
//!
//! A byte of the sieve holds the 30 numbers from a multiple of 30, `low + 30 j`, in the eight that
//! are prime to 2, 3 and 5: bit `k` is `low + 30 j + OFFSETS[k]`, as in Walisch's primesieve, so
//! that the bits run from 7 to 31 and 1 is never one of them.
//!
//! A sieving prime `p` crosses out `p m` for the multipliers `m` prime to 30 (the others are not in
//! the sieve). Its multiples prime to 30 repeat with `m` modulo 30: from one to the next, `m` grows
//! by `D30[w]` at the position `w` of `m mod 30` in [`M30`], the byte by `D30[w] (p / 30)` and a
//! correction that depends on `p mod 30` (its class, `c`) and `w` alone, and the bit is again
//! fixed by `c` and `w`. So a prime is held as `p / 30`, its class and position (its wheel index,
//! `8 c + w`) and the byte of its next multiple, and its steps are those of [`STEP30`]. Large primes
//! take the wheel of 210 instead, whose 48 positions skip the multiples of 7 too.

/// The numbers of the bits of a byte, from the multiple of 30 before the byte.
pub(super) const OFFSETS: [u64; 8] = [7, 11, 13, 17, 19, 23, 29, 31];

/// The bit of each residue modulo 30 of a number, as [`OFFSETS`] places them (1 is bit 7, as 31
/// is), or 8 for the residues that share a factor with 30.
static BIT_OF: [u8; 30] = {
    let mut t = [8u8; 30];
    let mut k = 0;
    while k < 8 {
        t[(OFFSETS[k] % 30) as usize] = k as u8;
        k += 1;
    }
    t
};

/// The value of each bit of a word of the sieve, eight bytes, from the word's multiple of 30: bit
/// `b` is `30 (b / 8) + OFFSETS[b % 8]`; one more, 0, for an empty word's trailing zeros.
pub(super) static BIT_VALUES: [u64; 65] = {
    let mut t = [0u64; 65];
    let mut b = 0;
    while b < 64 {
        t[b] = 30 * (b as u64 / 8) + OFFSETS[b % 8];
        b += 1;
    }
    t
};

/// The multipliers prime to 30, in the order of the positions of the wheel.
static M30: [u64; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

/// The steps from each multiplier of [`M30`] to the next, the last to 31.
pub(super) static D30: [u64; 8] = [6, 4, 2, 4, 2, 4, 6, 2];

/// The residues modulo 30 of the classes of primes, in the order of the wheel indices.
pub(super) static R30: [u64; 8] = [7, 11, 13, 17, 19, 23, 29, 1];

/// The class of each residue modulo 30 of a prime above 5, an index into [`R30`].
pub(super) static CLASS: [u8; 30] = {
    let mut t = [0u8; 30];
    let mut c = 0;
    while c < 8 {
        t[R30[c] as usize] = c as u8;
        c += 1;
    }
    t
};

/// The bit of the sieve that `p m` falls on, for `p` of class `c` and `m` at position `w`.
const fn bit30(c: usize, w: usize) -> u64 {
    BIT_OF[((R30[c] * M30[w]) % 30) as usize] as u64
}

/// The offset of [`bit30`] from its byte's multiple of 30.
const fn offset30(c: usize, w: usize) -> u64 {
    OFFSETS[bit30(c, w) as usize]
}

/// The mask that crosses out `p m` in its byte, and the bytes from it to the next multiple beyond
/// `D30[w] (p / 30)`, for each class `c` and position `w`: `STEP30[c][w]`.
pub(super) static STEP30: [[(u8, u64); 8]; 8] = {
    let mut t = [[(0u8, 0u64); 8]; 8];
    let mut c = 0;
    while c < 8 {
        let mut w = 0;
        while w < 8 {
            let next = offset30(c, (w + 1) % 8);
            let here = offset30(c, w) + R30[c] * D30[w];
            assert!((here - next).is_multiple_of(30));
            t[c][w] = (!(1u8 << bit30(c, w)), (here - next) / 30);
            w += 1;
        }
        c += 1;
    }
    t
};

/// The bytes from the multiple at position 0 to the one at position `w`, beyond `CUM30[w] (p / 30)`:
/// the sums of the corrections of [`STEP30`] before `w`, for each class.
pub(super) static CORR_CUM30: [[u64; 8]; 8] = {
    let mut t = [[0u64; 8]; 8];
    let mut c = 0;
    while c < 8 {
        let mut w = 1;
        while w < 8 {
            t[c][w] = t[c][w - 1] + STEP30[c][w - 1].1;
            w += 1;
        }
        c += 1;
    }
    t
};

/// The sums of the steps of [`D30`] before each position.
pub(super) static CUM30: [u64; 8] = [0, 6, 10, 12, 16, 18, 22, 28];

/// The first multiplier prime to 30 from each residue modulo 30, as the step to it and its
/// position: `(f, w)` for `q mod 30`, with `q + f` prime to 30 at position `w` of [`M30`].
pub(super) static INIT30: [(u8, u8); 30] = {
    let mut t = [(0u8, 0u8); 30];
    let mut r = 0;
    while r < 30 {
        let mut f = 0;
        loop {
            let m = (r + f) % 30;
            let mut w = 0;
            while w < 8 && M30[w] != m as u64 {
                w += 1;
            }
            if w < 8 {
                t[r] = (f as u8, w as u8);
                break;
            }
            f += 1;
        }
        r += 1;
    }
    t
};

/// The number of positions of the wheel of 210.
const POSITIONS: usize = 48;

/// The multipliers prime to 210 below 210, in the order of the positions of the wheel of 210.
static M210: [u64; POSITIONS] = {
    let mut t = [0u64; POSITIONS];
    let (mut m, mut w) = (1u64, 0);
    while m < 210 {
        if !m.is_multiple_of(2)
            && !m.is_multiple_of(3)
            && !m.is_multiple_of(5)
            && !m.is_multiple_of(7)
        {
            t[w] = m;
            w += 1;
        }
        m += 1;
    }
    t
};

/// One step of a large prime on the wheel of 210: the mask that crosses out its multiple, the
/// multiplier's step to the next (so many times `p / 30` bytes) and the bytes beyond, and the
/// wheel index of the next multiple. A power of two in size, eight bytes.
#[derive(Clone, Copy)]
pub(super) struct Step210 {
    /// The mask that crosses out the multiple in its byte.
    pub(super) mask: u8,
    /// The step of the multiplier to the next multiple.
    pub(super) factor: u8,
    /// The bytes from the multiple to the next beyond `factor (p / 30)`.
    pub(super) corr: u8,
    /// The wheel index of the next multiple.
    pub(super) next: u32,
}

/// The steps of [`Step210`] for each wheel index `48 c + w`, `c` the class of the prime and `w`
/// the position of its multiplier in [`M210`].
pub(super) static STEP210: [Step210; 8 * POSITIONS] = {
    let mut t = [Step210 {
        mask: 0,
        factor: 0,
        corr: 0,
        next: 0,
    }; 8 * POSITIONS];
    let mut c = 0;
    while c < 8 {
        let mut w = 0;
        while w < POSITIONS {
            let m = M210[w];
            let step = if w + 1 < POSITIONS {
                M210[w + 1] - m
            } else {
                211 - m
            };
            let bit = BIT_OF[((R30[c] * m) % 30) as usize] as usize;
            let next_bit = BIT_OF[((R30[c] * (m + step)) % 30) as usize] as usize;
            let here = OFFSETS[bit] + R30[c] * step;
            assert!((here - OFFSETS[next_bit]).is_multiple_of(30));
            t[c * POSITIONS + w] = Step210 {
                mask: !(1u8 << bit),
                factor: step as u8,
                corr: ((here - OFFSETS[next_bit]) / 30) as u8,
                next: (c * POSITIONS + (w + 1) % POSITIONS) as u32,
            };
            w += 1;
        }
        c += 1;
    }
    t
};

/// The first multiplier prime to 210 from each residue modulo 210, as the step to it and its
/// position in [`M210`].
pub(super) static INIT210: [(u8, u8); 210] = {
    let mut t = [(0u8, 0u8); 210];
    let mut r = 0;
    while r < 210 {
        let mut f = 0;
        loop {
            let m = ((r + f) % 210) as u64;
            let mut w = 0;
            while w < POSITIONS && M210[w] != m {
                w += 1;
            }
            if w < POSITIONS {
                t[r] = (f as u8, w as u8);
                break;
            }
            f += 1;
        }
        r += 1;
    }
    t
};

/// The masks that keep, in the byte of a number `n` (by `(n - low - 7) mod 30 + 7`, from 7 to 36),
/// the bits of the numbers from `n` up.
pub(super) static KEEP_FROM: [u8; 37] = {
    let mut t = [0u8; 37];
    let mut r = 0;
    while r < 37 {
        let mut m = 0u8;
        let mut k = 0;
        while k < 8 {
            if OFFSETS[k] >= r as u64 {
                m |= 1 << k;
            }
            k += 1;
        }
        t[r] = m;
        r += 1;
    }
    t
};

/// The masks that keep, in the byte of a number `n`, the bits of the numbers up to `n`.
pub(super) static KEEP_TO: [u8; 37] = {
    let mut t = [0u8; 37];
    let mut r = 0;
    while r < 37 {
        let mut m = 0u8;
        let mut k = 0;
        while k < 8 {
            if OFFSETS[k] <= r as u64 {
                m |= 1 << k;
            }
            k += 1;
        }
        t[r] = m;
        r += 1;
    }
    t
};

/// `(n - 7) mod 30 + 7`: the distance, from 7 to 36, of `n` from the multiple of 30 that the byte
/// holding `n` starts from, for `n` at least 7.
#[inline(always)]
pub(super) const fn byte_remainder(n: u64) -> u64 {
    (n - 7) % 30 + 7
}
