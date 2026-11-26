//! The bodies of the direct kernels, written once and expanded in each target's module over that
//! module's vector operations, so that the portable code runs the same steps as the SIMD code.
//!
//! A module expanding them defines, for its vector type `V` of `W` lanes and its tap type `TV`:
//! `zero() -> V`, `ld(*const T) -> V` (unaligned), `st(*mut T, V)`, `splat(T) -> V`,
//! `add(V, V) -> V`, `fma(acc, x, y) = acc + x·y`,
//! `tload(*const T) -> TV` (the taps `W` at a time), `tfma::<S>(acc, x, tv)`, the same with
//! the value `S` of `tv` (a lane on NEON, a broadcast from memory elsewhere), and
//! `combine(a, b) = a + (−b₁, b₀, −b₃, b₂, …)` for the complex kernels.

/// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`, real.
///
/// The outputs are taken `NA` vectors at a time and the taps `G` vectors at a time: for each lane
/// `s` of the taps, the `NA + G − 1` vectors `x[k + t + s + W·q ..]` are loaded once each, and the
/// vector `q` meets the tap `t + g·W + s` in the sums of the outputs `q − g`, which are kept apart
/// for each `g` so that `G·NA` multiply-adds wait on no other. The taps left over, fewer than
/// `G·W`, are taken one at a time.
macro_rules! fir_real {
    ($(#[$m:meta])* $name:ident, $t:ty, $w:literal, $na:literal, $g:literal, [$($s:literal),*]) => {
        /// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`.
        ///
        /// # Safety
        ///
        /// `x` must be valid for reads of `ny + nt − 1 + PAD` values, `taps` of `nt` and `y` for
        /// writes of `ny`.
        $(#[$m])*
        pub(crate) unsafe fn $name(x: *const $t, taps: *const $t, nt: usize, y: *mut $t, ny: usize) {
            // The block divides INTERIOR (64) and is at most PAD (64): see `conv.rs`.
            const { assert!(64 % ($na * $w) == 0) };
            // SAFETY: a block of outputs from `k` reads `x` below `k + NA·W + nt − 1`, which is
            // `ny + nt − 1 + PAD` at most as `NA·W ≤ PAD`; it writes `y` below `ny` only.
            unsafe {
                let mut k = 0;
                while k < ny {
                    let mut acc = [[zero(); $na]; $g];
                    let xb = x.add(k);
                    let mut t = 0;
                    while t + $g * $w <= nt {
                        let mut tv = [tload(taps.add(t)); $g];
                        for (g, v) in tv.iter_mut().enumerate().skip(1) {
                            *v = tload(taps.add(t + g * $w));
                        }
                        $(
                            let p = xb.add(t + $s);
                            for q in 0..$na + $g - 1 {
                                let v = ld(p.add(q * $w));
                                for g in 0..$g {
                                    if q >= g && q - g < $na {
                                        acc[g][q - g] = tfma::<$s>(acc[g][q - g], v, tv[g]);
                                    }
                                }
                            }
                        )*
                        t += $g * $w;
                    }
                    while t < nt {
                        let tb = splat(*taps.add(t));
                        let p = xb.add(t);
                        for (i, a) in acc[0].iter_mut().enumerate() {
                            *a = fma(*a, ld(p.add(i * $w)), tb);
                        }
                        t += 1;
                    }
                    for g in 1..$g {
                        for i in 0..$na {
                            acc[0][i] = add(acc[0][i], acc[g][i]);
                        }
                    }
                    if k + $na * $w <= ny {
                        for (i, a) in acc[0].iter().enumerate() {
                            st(y.add(k + i * $w), *a);
                        }
                    } else {
                        let mut tmp = [0.0 as $t; $na * $w];
                        for (i, a) in acc[0].iter().enumerate() {
                            st(tmp.as_mut_ptr().add(i * $w), *a);
                        }
                        core::ptr::copy_nonoverlapping(tmp.as_ptr(), y.add(k), ny - k);
                    }
                    k += $na * $w;
                }
            }
        }
    };
}

/// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`, complex, on interleaved parts.
///
/// A vector holds `W/2` complex values. With the tap `t = tr + i·ti`, the sums `A = Σ x·tr` and
/// `B = Σ x·ti` are taken lane by lane on the interleaved parts, and the products' sum is
/// `A + i·B`: `(A.re − B.im) + i·(A.im + B.re)`, which `combine(A, B)` forms once at the end
/// (the lanes of `B` swapped in pairs), so the steps take no shuffles. `NA` vectors of outputs at
/// a time; the taps a vector at a time, the lanes `(re, im)` of the tap `s` in it for the loads
/// from `x[k + t + s ..]`, and those left over one at a time.
macro_rules! fir_complex {
    ($(#[$m:meta])* $name:ident, $t:ty, $w:literal, $na:literal, [$(($s:literal, $lr:literal, $li:literal)),*]) => {
        /// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`, complex values as pairs of parts.
        ///
        /// # Safety
        ///
        /// `x` must be valid for reads of `ny + nt − 1 + PAD` complex values, `taps` of `nt` and
        /// `y` for writes of `ny`.
        $(#[$m])*
        pub(crate) unsafe fn $name(x: *const $t, taps: *const $t, nt: usize, y: *mut $t, ny: usize) {
            const C: usize = $w / 2;
            // The block divides INTERIOR (64) and is at most PAD (64): see `conv.rs`.
            const { assert!(64 % ($na * C) == 0) };
            // SAFETY: as for the real kernels, in pairs of parts.
            unsafe {
                let mut k = 0;
                while k < ny {
                    let mut aa = [zero(); $na];
                    let mut ab = [zero(); $na];
                    let xb = x.add(2 * k);
                    let mut t = 0;
                    while t + C <= nt {
                        let tv = tload(taps.add(2 * t));
                        $(
                            let p = xb.add(2 * (t + $s));
                            for i in 0..$na {
                                let v = ld(p.add(i * $w));
                                aa[i] = tfma::<$lr>(aa[i], v, tv);
                                ab[i] = tfma::<$li>(ab[i], v, tv);
                            }
                        )*
                        t += C;
                    }
                    while t < nt {
                        let (br, bi) = (splat(*taps.add(2 * t)), splat(*taps.add(2 * t + 1)));
                        let p = xb.add(2 * t);
                        for i in 0..$na {
                            let v = ld(p.add(i * $w));
                            aa[i] = fma(aa[i], v, br);
                            ab[i] = fma(ab[i], v, bi);
                        }
                        t += 1;
                    }
                    if k + $na * C <= ny {
                        for i in 0..$na {
                            st(y.add(2 * k + i * $w), combine(aa[i], ab[i]));
                        }
                    } else {
                        let mut tmp = [0.0 as $t; $na * $w];
                        for i in 0..$na {
                            st(tmp.as_mut_ptr().add(i * $w), combine(aa[i], ab[i]));
                        }
                        core::ptr::copy_nonoverlapping(tmp.as_ptr(), y.add(2 * k), 2 * (ny - k));
                    }
                    k += $na * C;
                }
            }
        }
    };
}

/// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny` and `1 ≤ nt ≤ 4`, real, reading `x` below
/// `ny + nt − 1` only (in place on the caller's sequence): the taps in registers, blocks of four
/// vectors of outputs, then single vectors, then the last values one by one.
macro_rules! fir_real_few {
    ($(#[$m:meta])* $name:ident, $t:ty, $w:literal) => {
        /// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`, `1 ≤ nt ≤ 4`.
        ///
        /// # Safety
        ///
        /// `x` must be valid for reads of `ny + nt − 1` values, `taps` of `nt` and `y` for writes
        /// of `ny`.
        $(#[$m])*
        pub(crate) unsafe fn $name(x: *const $t, taps: *const $t, nt: usize, y: *mut $t, ny: usize) {
            $(#[$m])*
            #[inline]
            unsafe fn run<const NT: usize>(x: *const $t, taps: *const $t, y: *mut $t, ny: usize) {
                // SAFETY: an output `k < ny` reads `x` below `k + NT ≤ ny + NT − 1`.
                unsafe {
                    let mut tv = [zero(); NT];
                    for (t, v) in tv.iter_mut().enumerate() {
                        *v = splat(*taps.add(t));
                    }
                    let mut k = 0;
                    while k + 4 * $w <= ny {
                        let mut a = [zero(); 4];
                        for (t, &v) in tv.iter().enumerate() {
                            for (i, a) in a.iter_mut().enumerate() {
                                *a = fma(*a, ld(x.add(k + i * $w + t)), v);
                            }
                        }
                        for (i, &a) in a.iter().enumerate() {
                            st(y.add(k + i * $w), a);
                        }
                        k += 4 * $w;
                    }
                    while k + $w <= ny {
                        let mut a = zero();
                        for (t, &v) in tv.iter().enumerate() {
                            a = fma(a, ld(x.add(k + t)), v);
                        }
                        st(y.add(k), a);
                        k += $w;
                    }
                    while k < ny {
                        let mut s: $t = 0.0;
                        for t in 0..NT {
                            s += *taps.add(t) * *x.add(k + t);
                        }
                        *y.add(k) = s;
                        k += 1;
                    }
                }
            }
            debug_assert!((1..=4).contains(&nt));
            // SAFETY: the caller's.
            unsafe {
                match nt {
                    1 => run::<1>(x, taps, y, ny),
                    2 => run::<2>(x, taps, y, ny),
                    3 => run::<3>(x, taps, y, ny),
                    _ => run::<4>(x, taps, y, ny),
                }
            }
        }
    };
}

/// [`fir_real_few`] for complex values on interleaved parts, the sums `A = Σ x·tr` and
/// `B = Σ x·ti` combined once per vector as in [`fir_complex`].
macro_rules! fir_complex_few {
    ($(#[$m:meta])* $name:ident, $t:ty, $w:literal) => {
        /// `y[k] = Σ_{t < nt} taps[t]·x[k + t]` for `k < ny`, `1 ≤ nt ≤ 4`, complex values as
        /// pairs of parts.
        ///
        /// # Safety
        ///
        /// `x` must be valid for reads of `ny + nt − 1` complex values, `taps` of `nt` and `y`
        /// for writes of `ny`.
        $(#[$m])*
        pub(crate) unsafe fn $name(x: *const $t, taps: *const $t, nt: usize, y: *mut $t, ny: usize) {
            $(#[$m])*
            #[inline]
            unsafe fn run<const NT: usize>(x: *const $t, taps: *const $t, y: *mut $t, ny: usize) {
                const C: usize = $w / 2;
                // SAFETY: as for the real kernel, in pairs of parts.
                unsafe {
                    let (mut tr, mut ti) = ([zero(); NT], [zero(); NT]);
                    for t in 0..NT {
                        tr[t] = splat(*taps.add(2 * t));
                        ti[t] = splat(*taps.add(2 * t + 1));
                    }
                    let mut k = 0;
                    while k + 2 * C <= ny {
                        let (mut a0, mut b0, mut a1, mut b1) = (zero(), zero(), zero(), zero());
                        for t in 0..NT {
                            let (v0, v1) = (ld(x.add(2 * (k + t))), ld(x.add(2 * (k + C + t))));
                            a0 = fma(a0, v0, tr[t]);
                            b0 = fma(b0, v0, ti[t]);
                            a1 = fma(a1, v1, tr[t]);
                            b1 = fma(b1, v1, ti[t]);
                        }
                        st(y.add(2 * k), combine(a0, b0));
                        st(y.add(2 * (k + C)), combine(a1, b1));
                        k += 2 * C;
                    }
                    while k + C <= ny {
                        let (mut a, mut b) = (zero(), zero());
                        for t in 0..NT {
                            let v = ld(x.add(2 * (k + t)));
                            a = fma(a, v, tr[t]);
                            b = fma(b, v, ti[t]);
                        }
                        st(y.add(2 * k), combine(a, b));
                        k += C;
                    }
                    while k < ny {
                        let (mut re, mut im): ($t, $t) = (0.0, 0.0);
                        for t in 0..NT {
                            let (xr, xi) = (*x.add(2 * (k + t)), *x.add(2 * (k + t) + 1));
                            let (wr, wi) = (*taps.add(2 * t), *taps.add(2 * t + 1));
                            re += xr * wr - xi * wi;
                            im += xr * wi + xi * wr;
                        }
                        *y.add(2 * k) = re;
                        *y.add(2 * k + 1) = im;
                        k += 1;
                    }
                }
            }
            debug_assert!((1..=4).contains(&nt));
            // SAFETY: the caller's.
            unsafe {
                match nt {
                    1 => run::<1>(x, taps, y, ny),
                    2 => run::<2>(x, taps, y, ny),
                    3 => run::<3>(x, taps, y, ny),
                    _ => run::<4>(x, taps, y, ny),
                }
            }
        }
    };
}
