//! a::math::uint's Uint<N> and Int<N> against ruint, crypto-bigint, bnum, GMP (mpn/mpz),
//! malachite-nz and the primitive u64/u128 (i64/i128), at N = 1, 2, 4, 6, 8, 16 limbs.
//! Per item: one operation on an operand pair from a pool of random operands (fresh per item);
//! every side's results are compared with ours before timing.

#[path = "../common.rs"]
mod common;
#[path = "../gmp.rs"]
mod gmp;

use a::math::uint::{Int, Uint};
use common::*;
use crypto_bigint::{Limb, NonZero};
use malachite_base::num::arithmetic::traits::{DivRem as MDivRem, FloorSqrt, Pow as MPow};
use malachite_base::num::conversion::traits::FromStringBase;
use malachite_nz::natural::Natural;
use num_integer::{Integer, Roots};
use std::cmp::Ordering;
use std::str::FromStr;

type Ru<const B: usize, const L: usize> = ruint::Uint<B, L>;
type Cb<const L: usize> = crypto_bigint::Uint<L>;
type CbI<const L: usize> = crypto_bigint::Int<L>;
type Bn<const L: usize> = bnum::BUint<L>;
type Bi<const L: usize> = bnum::BInt<L>;

fn nat(l: &[u64]) -> Natural {
    Natural::from_limbs_asc(l)
}
fn nat_limbs<const N: usize>(n: &Natural) -> [u64; N] {
    let v = n.to_limbs_asc();
    let mut out = [0u64; N];
    out[..v.len()].copy_from_slice(&v);
    out
}
fn p128<const N: usize>(l: &[u64; N]) -> u128 {
    l[0] as u128 | (l.get(1).copied().unwrap_or(0) as u128) << 64
}
fn sig(l: &[u64]) -> usize {
    let mut n = l.len();
    while n > 0 && l[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// A random number of exactly `limbs` significant limbs (its top limb nonzero), in N limbs.
fn sized<const N: usize>(rng: &mut Rng, limbs: usize) -> [u64; N] {
    let mut l = [0u64; N];
    for x in l.iter_mut().take(limbs) {
        *x = rng.next();
    }
    if limbs > 0 && l[limbs - 1] == 0 {
        l[limbs - 1] = 1;
    }
    l
}

/// A random number below 2^bits, in N limbs.
fn bits<const N: usize>(rng: &mut Rng, bits: u32) -> [u64; N] {
    let mut l = [0u64; N];
    for (i, x) in l.iter_mut().enumerate() {
        let lo = 64 * i as u32;
        if lo >= bits {
            break;
        }
        let v = rng.next();
        *x = if bits - lo >= 64 { v } else { v & ((1u64 << (bits - lo)) - 1) };
    }
    l
}

struct Pools<const N: usize, const BITS: usize> {
    l: Vec<([u64; N], [u64; N])>,
    ours: Vec<(Uint<N>, Uint<N>)>,
    ru: Vec<(Ru<BITS, N>, Ru<BITS, N>)>,
    cb: Vec<(Cb<N>, Cb<N>)>,
    bn: Vec<(Bn<N>, Bn<N>)>,
    pr: Vec<(u128, u128)>,
}

impl<const N: usize, const BITS: usize> Pools<N, BITS> {
    fn new(l: Vec<([u64; N], [u64; N])>) -> Self {
        Pools {
            ours: l.iter().map(|(a, b)| (Uint::from_limbs(*a), Uint::from_limbs(*b))).collect(),
            ru: l.iter().map(|(a, b)| (Ru::from_limbs(*a), Ru::from_limbs(*b))).collect(),
            cb: l.iter().map(|(a, b)| (Cb::from_words(*a), Cb::from_words(*b))).collect(),
            bn: l.iter().map(|(a, b)| (Bn::from_digits(*a), Bn::from_digits(*b))).collect(),
            pr: l.iter().map(|(a, b)| (p128(a), p128(b))).collect(),
            l,
        }
    }
    fn nat(&self) -> Vec<(Natural, Natural)> {
        self.l.iter().map(|(a, b)| (nat(a), nat(b))).collect()
    }
}

fn prim_name(n: usize, signed: bool) -> &'static str {
    match (n, signed) {
        (1, false) => "u64",
        (2, false) => "u128",
        (1, true) => "i64",
        _ => "i128",
    }
}

#[allow(clippy::too_many_lines)]
fn bench<const N: usize, const BITS: usize, const W: usize, const WBITS: usize>(
    out: &Out,
    seed: u64,
) {
    let mut rng = Rng::new(seed);
    let len = pool_len(2 * 8 * N).max(1024);
    let p = format!("N={N} ({BITS} bit)");
    let g = "uint";
    let full = Pools::<N, BITS>::new((0..len).map(|_| (rng.limbs(), rng.limbs())).collect());
    let lo64 = |x: u128| x as u64;

    // ---- sums and differences -----------------------------------------------------------
    {
        let f = &full;
        let want: Vec<[u64; N]> = f.ours.iter().map(|(a, b)| *a.wrapping_add(*b).as_limbs()).collect();
        same("add", "ruint", &f.ru, &want, |(a, b)| *a.wrapping_add(*b).as_limbs());
        same("add", "cb", &f.cb, &want, |(a, b)| a.wrapping_add(b).to_words());
        same("add", "bnum", &f.bn, &want, |(a, b)| *a.wrapping_add(*b).digits());
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.wrapping_add(*b)),
            mk("ruint", "", &f.ru, |(a, b)| a.wrapping_add(*b)),
            mk("crypto-bigint", "", &f.cb, |(a, b)| a.wrapping_add(b)),
            mk("bnum", "", &f.bn, |(a, b)| a.wrapping_add(*b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], [u64; N])| {
                let mut r = [0u64; N];
                unsafe { gmp::__gmpn_add_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N as _) };
                r
            };
            same("add", "gmp", &f.l, &want, gf);
            s.push(mk("GMP mpn_add_n", "a library call per item", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, b)| lo64(*a).wrapping_add(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, b)| a.wrapping_add(*b)));
        }
        out.row(g, "wrapping_add", &p, len, "", s);

        let want: Vec<[u64; N]> = f.ours.iter().map(|(a, b)| *a.wrapping_sub(*b).as_limbs()).collect();
        same("sub", "ruint", &f.ru, &want, |(a, b)| *a.wrapping_sub(*b).as_limbs());
        same("sub", "cb", &f.cb, &want, |(a, b)| a.wrapping_sub(b).to_words());
        same("sub", "bnum", &f.bn, &want, |(a, b)| *a.wrapping_sub(*b).digits());
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.wrapping_sub(*b)),
            mk("ruint", "", &f.ru, |(a, b)| a.wrapping_sub(*b)),
            mk("crypto-bigint", "", &f.cb, |(a, b)| a.wrapping_sub(b)),
            mk("bnum", "", &f.bn, |(a, b)| a.wrapping_sub(*b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], [u64; N])| {
                let mut r = [0u64; N];
                unsafe { gmp::__gmpn_sub_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N as _) };
                r
            };
            same("sub", "gmp", &f.l, &want, gf);
            s.push(mk("GMP mpn_sub_n", "a library call per item", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, b)| lo64(*a).wrapping_sub(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, b)| a.wrapping_sub(*b)));
        }
        out.row(g, "wrapping_sub", &p, len, "", s);
    }

    // ---- products -----------------------------------------------------------------------
    {
        let f = &full;
        let want: Vec<[u64; N]> = f.ours.iter().map(|(a, b)| *a.wrapping_mul(*b).as_limbs()).collect();
        same("mul", "ruint", &f.ru, &want, |(a, b)| *a.wrapping_mul(*b).as_limbs());
        same("mul", "cb", &f.cb, &want, |(a, b)| a.wrapping_mul(b).to_words());
        same("mul", "bnum", &f.bn, &want, |(a, b)| *a.wrapping_mul(*b).digits());
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.wrapping_mul(*b)),
            mk("ruint", "", &f.ru, |(a, b)| a.wrapping_mul(*b)),
            mk("crypto-bigint", "its wrapping_mul forms the full product (split_mul)", &f.cb, |(a, b)| {
                a.wrapping_mul(b)
            }),
            mk("bnum", "", &f.bn, |(a, b)| a.wrapping_mul(*b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], [u64; N])| {
                let mut r = [0u64; N];
                unsafe { gmp::__gmpn_mullo_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N as _) };
                r
            };
            same("mul", "gmp", &f.l, &want, gf);
            s.push(mk("GMP mpn_mullo_n", "GMP's internal (exported, undocumented) low product", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, b)| lo64(*a).wrapping_mul(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, b)| a.wrapping_mul(*b)));
        }
        out.row(g, "wrapping_mul", &p, len, "random full-width operands", s);

        // widening
        let want: Vec<([u64; N], [u64; N])> = f
            .ours
            .iter()
            .map(|(a, b)| {
                let (lo, hi) = a.widening_mul(*b);
                (*lo.as_limbs(), *hi.as_limbs())
            })
            .collect();
        let split = |w: &[u64; W]| -> ([u64; N], [u64; N]) {
            (w[..N].try_into().unwrap(), w[N..].try_into().unwrap())
        };
        same("wmul", "ruint", &f.ru, &want, |(a, b)| {
            split(a.widening_mul::<BITS, N, WBITS, W>(*b).as_limbs())
        });
        same("wmul", "cb", &f.cb, &want, |(a, b)| {
            let (lo, hi) = a.split_mul(b);
            (lo.to_words(), hi.to_words())
        });
        same("wmul", "bnum", &f.bn, &want, |(a, b)| {
            let (lo, hi) = a.widening_mul(*b);
            (*lo.digits(), *hi.digits())
        });
        let fnat = f.nat();
        same("wmul", "malachite", &fnat, &want, |(a, b)| split(&nat_limbs::<W>(&(a * b))));
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.widening_mul(*b)),
            mk("ruint", "", &f.ru, |(a, b)| a.widening_mul::<BITS, N, WBITS, W>(*b)),
            mk("crypto-bigint", "split_mul", &f.cb, |(a, b)| a.split_mul(b)),
            mk("bnum", "", &f.bn, |(a, b)| a.widening_mul(*b)),
            mk("malachite-nz", "Natural, allocates the result", &fnat, |(a, b)| a * b),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], [u64; N])| {
                let mut r = [0u64; W];
                unsafe { gmp::__gmpn_mul_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N as _) };
                r
            };
            same("wmul", "gmp", &f.l, &want, |x| split(&gf(x)));
            s.push(mk("GMP mpn_mul_n", "a library call per item", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "u64 × u64 → u128", &f.pr, |(a, b)| (*a as u64 as u128) * (*b as u64 as u128)));
        }
        out.row(g, "widening_mul", &p, len, "", s);

        // overflowing product of half-width operands (never overflows)
        let half = Pools::<N, BITS>::new(
            (0..len)
                .map(|_| (bits(&mut rng, (BITS / 2) as u32), bits(&mut rng, (BITS / 2) as u32)))
                .collect(),
        );
        let h = &half;
        let want: Vec<([u64; N], bool)> = h
            .ours
            .iter()
            .map(|(a, b)| {
                let (r, o) = a.overflowing_mul(*b);
                (*r.as_limbs(), o)
            })
            .collect();
        same("omul", "ruint", &h.ru, &want, |(a, b)| {
            let (r, o) = a.overflowing_mul(*b);
            (*r.as_limbs(), o)
        });
        same("omul", "bnum", &h.bn, &want, |(a, b)| {
            let (r, o) = a.overflowing_mul(*b);
            (*r.digits(), o)
        });
        let mut s = vec![
            mk("ours", "", &h.ours, |(a, b)| a.overflowing_mul(*b)),
            mk("ruint", "", &h.ru, |(a, b)| a.overflowing_mul(*b)),
            mk("bnum", "", &h.bn, |(a, b)| a.overflowing_mul(*b)),
        ];
        if N == 1 {
            s.push(mk("u64", "", &h.pr, |(a, b)| lo64(*a).overflowing_mul(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &h.pr, |(a, b)| a.overflowing_mul(*b)));
        }
        out.row(g, "overflowing_mul", &format!("{p}, operands of {} bits", BITS / 2), len, "", s);

        // one-limb factor
        let d: Vec<([u64; N], u64)> = f.l.iter().map(|(a, b)| (*a, b[0])).collect();
        let od: Vec<(Uint<N>, u64)> = d.iter().map(|(a, b)| (Uint::from_limbs(*a), *b)).collect();
        let cd: Vec<(Cb<N>, Cb<1>)> = d.iter().map(|(a, b)| (Cb::from_words(*a), Cb::from_words([*b]))).collect();
        let want: Vec<([u64; N], u64)> = od
            .iter()
            .map(|(a, b)| {
                let (lo, hi) = a.widening_mul_u64(*b);
                (*lo.as_limbs(), hi)
            })
            .collect();
        same("mul1", "cb", &cd, &want, |(a, b)| {
            let (lo, hi) = a.split_mul(b);
            (lo.to_words(), hi.to_words()[0])
        });
        let mut s = vec![
            mk("ours", "", &od, |(a, b)| a.widening_mul_u64(*b)),
            mk("crypto-bigint", "split_mul by a Uint<1>", &cd, |(a, b)| a.split_mul(b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], u64)| {
                let mut r = [0u64; N];
                let c = unsafe { gmp::__gmpn_mul_1(r.as_mut_ptr(), a.as_ptr(), N as _, *b) };
                (r, c)
            };
            same("mul1", "gmp", &d, &want, gf);
            s.push(mk("GMP mpn_mul_1", "a library call per item", &d, gf));
        }
        if N == 1 {
            s.push(mk("u64", "u64 × u64 → u128", &d, |(a, b)| a[0] as u128 * *b as u128));
        }
        out.row(g, "widening_mul_u64", &p, len, "", s);
    }

    // ---- quotients ----------------------------------------------------------------------
    let mut vls = vec![N.div_ceil(2)];
    if N > 1 {
        vls.push(N);
    }
    for vl in vls {
        let pool = Pools::<N, BITS>::new(
            (0..len).map(|_| (sized(&mut rng, N), sized(&mut rng, vl))).collect(),
        );
        let f = &pool;
        let want: Vec<([u64; N], [u64; N])> = f
            .ours
            .iter()
            .map(|(a, b)| {
                let (q, r) = a.div_rem(*b);
                (*q.as_limbs(), *r.as_limbs())
            })
            .collect();
        same("div", "ruint", &f.ru, &want, |(a, b)| {
            let (q, r) = a.div_rem(*b);
            (*q.as_limbs(), *r.as_limbs())
        });
        let cbn: Vec<(Cb<N>, NonZero<Cb<N>>)> = f.cb.iter().map(|(a, b)| (*a, NonZero::new(*b).unwrap())).collect();
        same("div", "cb", &cbn, &want, |(a, b)| {
            let (q, r) = a.div_rem_vartime(b);
            (q.to_words(), r.to_words())
        });
        same("div", "bnum", &f.bn, &want, |(a, b)| {
            let (q, r) = Integer::div_rem(a, b);
            (*q.digits(), *r.digits())
        });
        let fnat = f.nat();
        same("div", "malachite", &fnat, &want, |(a, b)| {
            let (q, r) = a.div_rem(b);
            (nat_limbs::<N>(&q), nat_limbs::<N>(&r))
        });
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.div_rem(*b)),
            mk("ruint", "", &f.ru, |(a, b)| a.div_rem(*b)),
            mk("crypto-bigint", "div_rem_vartime", &cbn, |(a, b)| a.div_rem_vartime(b)),
            mk("bnum", "num_integer::Integer::div_rem", &f.bn, |(a, b)| Integer::div_rem(a, b)),
            mk("malachite-nz", "Natural, allocates", &fnat, |(a, b)| a.div_rem(b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], [u64; N])| {
                let (nn, dn) = (sig(a), sig(b));
                let mut q = [0u64; N];
                let mut r = [0u64; N];
                if nn >= dn {
                    unsafe {
                        gmp::__gmpn_tdiv_qr(
                            q.as_mut_ptr(),
                            r.as_mut_ptr(),
                            0,
                            a.as_ptr(),
                            nn as _,
                            b.as_ptr(),
                            dn as _,
                        )
                    };
                } else {
                    r = *a;
                }
                (q, r)
            };
            same("div", "gmp", &f.l, &want, gf);
            s.push(mk("GMP mpn_tdiv_qr", "a library call per item", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, b)| (lo64(*a) / lo64(*b), lo64(*a) % lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "compiler-rt's __udivti3/__umodti3", &f.pr, |(a, b)| (a / b, a % b)));
        }
        out.row(g, "div_rem", &format!("{p}, u of {N} limbs, v of {vl}"), len, "", s);
    }
    {
        // one-limb divisor
        let d: Vec<([u64; N], u64)> = (0..len).map(|_| (sized(&mut rng, N), rng.next() | 1)).collect();
        let od: Vec<(Uint<N>, u64)> = d.iter().map(|(a, b)| (Uint::from_limbs(*a), *b)).collect();
        let want: Vec<([u64; N], u64)> = od
            .iter()
            .map(|(a, b)| {
                let (q, r) = a.div_rem_u64(*b);
                (*q.as_limbs(), r)
            })
            .collect();
        let rd: Vec<(Ru<BITS, N>, Ru<BITS, N>)> = d.iter().map(|(a, b)| (Ru::from_limbs(*a), Ru::from(*b))).collect();
        same("div1", "ruint", &rd, &want, |(a, b)| {
            let (q, r) = a.div_rem(*b);
            (*q.as_limbs(), r.as_limbs()[0])
        });
        let cd: Vec<(Cb<N>, NonZero<Limb>)> =
            d.iter().map(|(a, b)| (Cb::from_words(*a), NonZero::new(Limb(*b)).unwrap())).collect();
        same("div1", "cb", &cd, &want, |(a, b)| {
            let (q, r) = a.div_rem_limb(*b);
            (q.to_words(), r.0)
        });
        let bd: Vec<(Bn<N>, Bn<N>)> = d.iter().map(|(a, b)| (Bn::from_digits(*a), Bn::from(*b))).collect();
        same("div1", "bnum", &bd, &want, |(a, b)| {
            let (q, r) = Integer::div_rem(a, b);
            (*q.digits(), r.digits()[0])
        });
        let mut s = vec![
            mk("ours", "", &od, |(a, b)| a.div_rem_u64(*b)),
            mk("ruint", "div_rem by a Uint", &rd, |(a, b)| a.div_rem(*b)),
            mk("crypto-bigint", "div_rem_limb (reciprocal per call)", &cd, |(a, b)| a.div_rem_limb(*b)),
            mk("bnum", "div_rem by a BUint", &bd, |(a, b)| Integer::div_rem(a, b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], u64)| {
                let mut q = [0u64; N];
                let r = unsafe { gmp::__gmpn_divrem_1(q.as_mut_ptr(), 0, a.as_ptr(), N as _, *b) };
                (q, r)
            };
            same("div1", "gmp", &d, &want, gf);
            s.push(mk("GMP mpn_divrem_1", "a library call per item", &d, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &d, |(a, b)| (a[0] / b, a[0] % b)));
        } else if N == 2 {
            s.push(mk("u128", "u128 by a u128 below 2^64", &d, |(a, b)| {
                let x = p128(a);
                (x / *b as u128, x % *b as u128)
            }));
        }
        out.row(g, "div_rem_u64", &format!("{p}, odd 64-bit divisor"), len, "", s);
    }

    // ---- shifts and comparisons ---------------------------------------------------------
    {
        let k: Vec<([u64; N], u32)> = full.l.iter().map(|(a, _)| (*a, rng.below(BITS as u64) as u32)).collect();
        let ok: Vec<(Uint<N>, u32)> = k.iter().map(|(a, s)| (Uint::from_limbs(*a), *s)).collect();
        let rk: Vec<(Ru<BITS, N>, usize)> = k.iter().map(|(a, s)| (Ru::from_limbs(*a), *s as usize)).collect();
        let ck: Vec<(Cb<N>, u32)> = k.iter().map(|(a, s)| (Cb::from_words(*a), *s)).collect();
        let bk: Vec<(Bn<N>, u32)> = k.iter().map(|(a, s)| (Bn::from_digits(*a), *s)).collect();
        let pk: Vec<(u128, u32)> = k.iter().map(|(a, s)| (p128(a), *s)).collect();
        for left in [true, false] {
            let name = if left { "wrapping_shl" } else { "wrapping_shr" };
            let want: Vec<[u64; N]> = ok
                .iter()
                .map(|(a, s)| *if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }.as_limbs())
                .collect();
            same(name, "ruint", &rk, &want, |(a, s)| {
                *if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }.as_limbs()
            });
            same(name, "cb", &ck, &want, |(a, s)| {
                if left { a.shl_vartime(*s) } else { a.shr_vartime(*s) }.to_words()
            });
            same(name, "bnum", &bk, &want, |(a, s)| {
                *if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }.digits()
            });
            let mut s = vec![
                mk("ours", "", &ok, |(a, s)| if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }),
                mk("ruint", "", &rk, |(a, s)| if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }),
                mk("crypto-bigint", "shl_vartime/shr_vartime", &ck, |(a, s)| {
                    if left { a.shl_vartime(*s) } else { a.shr_vartime(*s) }
                }),
                mk("bnum", "", &bk, |(a, s)| if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }),
            ];
            if N == 1 {
                s.push(mk("u64", "", &pk, |(a, s)| {
                    if left { lo64(*a).wrapping_shl(*s) } else { lo64(*a).wrapping_shr(*s) }
                }));
            } else if N == 2 {
                s.push(mk("u128", "", &pk, |(a, s)| if left { a.wrapping_shl(*s) } else { a.wrapping_shr(*s) }));
            }
            out.row(g, name, &format!("{p}, random amount below {BITS}"), len, "", s);
        }
    }
    for eq_top in [false, true] {
        let pool = if eq_top {
            Pools::<N, BITS>::new(
                full.l
                    .iter()
                    .map(|(a, _)| {
                        let mut b = *a;
                        b[0] = rng.next();
                        (*a, b)
                    })
                    .collect(),
            )
        } else {
            Pools::<N, BITS>::new(full.l.clone())
        };
        let f = &pool;
        let want: Vec<Ordering> = f.ours.iter().map(|(a, b)| a.cmp(b)).collect();
        same("cmp", "ruint", &f.ru, &want, |(a, b)| a.cmp(b));
        same("cmp", "cb", &f.cb, &want, |(a, b)| a.cmp_vartime(b));
        same("cmp", "bnum", &f.bn, &want, |(a, b)| a.cmp(b));
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.cmp(b)),
            mk("ruint", "", &f.ru, |(a, b)| a.cmp(b)),
            mk("crypto-bigint", "cmp_vartime", &f.cb, |(a, b)| a.cmp_vartime(b)),
            mk("bnum", "", &f.bn, |(a, b)| a.cmp(b)),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, b): &([u64; N], [u64; N])| unsafe { gmp::__gmpn_cmp(a.as_ptr(), b.as_ptr(), N as _) }.cmp(&0);
            same("cmp", "gmp", &f.l, &want, gf);
            s.push(mk("GMP mpn_cmp", "a library call per item", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, b)| lo64(*a).cmp(&lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, b)| a.cmp(b)));
        }
        let what = if eq_top { "equal but for the low limb" } else { "random operands" };
        out.row(g, "cmp", &format!("{p}, {what}"), len, "", s);
    }

    // ---- roots, powers, logarithms ------------------------------------------------------
    {
        let f = &full;
        let want: Vec<[u64; N]> = f.ours.iter().map(|(a, _)| *a.isqrt().as_limbs()).collect();
        same("isqrt", "ruint", &f.ru, &want, |(a, _)| *a.root(2).as_limbs());
        same("isqrt", "cb", &f.cb, &want, |(a, _)| a.sqrt_vartime().to_words());
        same("isqrt", "bnum", &f.bn, &want, |(a, _)| *Roots::sqrt(a).digits());
        let fnat: Vec<(Natural, ())> = f.l.iter().map(|(a, _)| (nat(a), ())).collect();
        same("isqrt", "malachite", &fnat, &want, |(a, _)| nat_limbs::<N>(&a.floor_sqrt()));
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, _)| a.isqrt()),
            mk("ruint", "root(2)", &f.ru, |(a, _)| a.root(2)),
            mk("crypto-bigint", "sqrt_vartime", &f.cb, |(a, _)| a.sqrt_vartime()),
            mk("bnum", "num_integer::Roots::sqrt", &f.bn, |(a, _)| Roots::sqrt(a)),
            mk("malachite-nz", "floor_sqrt, allocates", &fnat, |(a, _)| a.floor_sqrt()),
        ];
        #[cfg(has_gmp)]
        {
            let gf = |(a, _): &([u64; N], [u64; N])| {
                let n = sig(a);
                let mut r = [0u64; N];
                unsafe { gmp::__gmpn_sqrtrem(r.as_mut_ptr(), std::ptr::null_mut(), a.as_ptr(), n as _) };
                r
            };
            same("isqrt", "gmp", &f.l, &want, gf);
            s.push(mk("GMP mpn_sqrtrem", "no remainder asked", &f.l, gf));
        }
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, _)| lo64(*a).isqrt()));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, _)| a.isqrt()));
        }
        out.row(g, "isqrt", &p, len, "", s);

        // wrapping powers: random base, exponent below 256
        let e: Vec<([u64; N], u32)> = f.l.iter().map(|(a, _)| (*a, 2 + rng.below(254) as u32)).collect();
        let oe: Vec<(Uint<N>, u32)> = e.iter().map(|(a, k)| (Uint::from_limbs(*a), *k)).collect();
        let re: Vec<(Ru<BITS, N>, Ru<BITS, N>)> = e.iter().map(|(a, k)| (Ru::from_limbs(*a), Ru::from(*k))).collect();
        let be: Vec<(Bn<N>, u32)> = e.iter().map(|(a, k)| (Bn::from_digits(*a), *k)).collect();
        let pe: Vec<(u128, u32)> = e.iter().map(|(a, k)| (p128(a), *k)).collect();
        let want: Vec<[u64; N]> = oe.iter().map(|(a, k)| *a.wrapping_pow(*k).as_limbs()).collect();
        same("wpow", "ruint", &re, &want, |(a, k)| *a.wrapping_pow(*k).as_limbs());
        same("wpow", "bnum", &be, &want, |(a, k)| *a.wrapping_pow(*k).digits());
        let mut s = vec![
            mk("ours", "", &oe, |(a, k)| a.wrapping_pow(*k)),
            mk("ruint", "", &re, |(a, k)| a.wrapping_pow(*k)),
            mk("bnum", "", &be, |(a, k)| a.wrapping_pow(*k)),
        ];
        if N == 1 {
            s.push(mk("u64", "", &pe, |(a, k)| lo64(*a).wrapping_pow(*k)));
        } else if N == 2 {
            s.push(mk("u128", "", &pe, |(a, k)| a.wrapping_pow(*k)));
        }
        out.row(g, "wrapping_pow", &format!("{p}, random base, exponent 2..255"), len, "", s);

        // exact powers: a base below 2^16 to the power BITS/16 − 1, which fits
        let k = (BITS / 16 - 1) as u32;
        let e: Vec<([u64; N], u32)> = (0..len).map(|_| (bits(&mut rng, 16), k)).collect();
        let oe: Vec<(Uint<N>, u32)> = e.iter().map(|(a, k)| (Uint::from_limbs(*a), *k)).collect();
        let re: Vec<(Ru<BITS, N>, Ru<BITS, N>)> = e.iter().map(|(a, k)| (Ru::from_limbs(*a), Ru::from(*k))).collect();
        let be: Vec<(Bn<N>, u32)> = e.iter().map(|(a, k)| (Bn::from_digits(*a), *k)).collect();
        let pe: Vec<(u128, u32)> = e.iter().map(|(a, k)| (p128(a), *k)).collect();
        let ne: Vec<(Natural, u64)> = e.iter().map(|(a, k)| (nat(a), *k as u64)).collect();
        let want: Vec<[u64; N]> = oe.iter().map(|(a, k)| *a.pow(*k).as_limbs()).collect();
        same("pow", "ruint", &re, &want, |(a, k)| *a.pow(*k).as_limbs());
        same("pow", "bnum", &be, &want, |(a, k)| *a.pow(*k).digits());
        same("pow", "malachite", &ne, &want, |(a, k)| nat_limbs::<N>(&a.pow(*k)));
        #[cfg(has_gmp)]
        let gpool: Vec<gmp::Z> = e.iter().map(|(a, _)| gmp::Z::of(a)).collect();
        let mut s = vec![
            mk("ours", "", &oe, |(a, k)| a.pow(*k)),
            mk("ruint", "", &re, |(a, k)| a.pow(*k)),
            mk("bnum", "", &be, |(a, k)| a.pow(*k)),
            mk("malachite-nz", "allocates", &ne, |(a, k)| a.pow(*k)),
        ];
        #[cfg(has_gmp)]
        {
            let gpool = &gpool;
            let mut r = gmp::Z::new();
            let got: Vec<[u64; N]> = gpool
                .iter()
                .map(|z| {
                    unsafe { gmp::__gmpz_pow_ui(r.m(), z.p(), k as _) };
                    r.limbs::<N>()
                })
                .collect();
            assert_eq!(got, want, "pow: GMP");
            s.push(side_n("GMP mpz_pow_ui", "into a preallocated mpz", move || {
                for z in std::hint::black_box(gpool) {
                    unsafe { gmp::__gmpz_pow_ui(r.m(), z.p(), k as _) };
                }
                std::hint::black_box(&mut r);
            }));
        }
        if N == 1 {
            s.push(mk("u64", "", &pe, |(a, k)| lo64(*a).pow(*k)));
        } else if N == 2 {
            s.push(mk("u128", "", &pe, |(a, k)| a.pow(*k)));
        }
        out.row(g, "pow", &format!("{p}, base below 2^16, exponent {k} (fits)"), len, "", s);

        // decimal logarithm
        let want: Vec<u32> = f.ours.iter().map(|(a, _)| a.ilog10()).collect();
        same("ilog10", "ruint", &f.ru, &want, |(a, _)| a.log10() as u32);
        same("ilog10", "bnum", &f.bn, &want, |(a, _)| a.ilog10());
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, _)| a.ilog10()),
            mk("ruint", "log10", &f.ru, |(a, _)| a.log10()),
            mk("bnum", "", &f.bn, |(a, _)| a.ilog10()),
        ];
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, _)| lo64(*a).ilog10()));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, _)| a.ilog10()));
        }
        out.row(g, "ilog10", &p, len, "", s);

        // to the nearest f64
        let want: Vec<u64> = f.ours.iter().map(|(a, _)| a.to_f64().to_bits()).collect();
        let mut s = vec![mk("ours", "", &f.ours, |(a, _)| a.to_f64())];
        let ru_ok = f.ru.iter().zip(&want).all(|((a, _), w)| f64::from(*a).to_bits() == *w);
        s.push(mk(
            "ruint",
            if ru_ok { "f64::from" } else { "f64::from; NOT the same rounding as ours on this pool" },
            &f.ru,
            |(a, _)| f64::from(*a),
        ));
        use bnum::cast::As;
        let bn_ok = f.bn.iter().zip(&want).all(|((a, _), w)| (*a).as_::<f64>().to_bits() == *w);
        s.push(mk(
            "bnum",
            if bn_ok { "as_::<f64>" } else { "as_::<f64>; NOT the same rounding as ours on this pool" },
            &f.bn,
            |(a, _)| (*a).as_::<f64>(),
        ));
        if N == 1 {
            s.push(mk("u64", "as f64", &f.pr, |(a, _)| lo64(*a) as f64));
        } else if N == 2 {
            s.push(mk("u128", "as f64", &f.pr, |(a, _)| *a as f64));
        }
        out.row(g, "to_f64", &p, len, "", s);
    }

    // ---- text ---------------------------------------------------------------------------
    for radix in [10u32, 16] {
        let f = &full;
        let want: Vec<String> = f.ours.iter().map(|(a, _)| a.to_str_radix(radix)).collect();
        let name = if radix == 10 { "to_string" } else { "to_str_radix(16)" };
        let r = radix as u64;
        same(name, "ruint", &f.ru, &want, |(a, _)| if radix == 10 { a.to_string() } else { format!("{a:x}") });
        same(name, "cb", &f.cb, &want, |(a, _)| a.to_string_radix_vartime(radix).to_lowercase());
        same(name, "bnum", &f.bn, &want, |(a, _)| a.to_str_radix(radix));
        let fnat: Vec<(Natural, ())> = f.l.iter().map(|(a, _)| (nat(a), ())).collect();
        same(name, "malachite", &fnat, &want, |(a, _)| if radix == 10 { a.to_string() } else { format!("{a:x}") });
        let _ = r;
        #[cfg(has_gmp)]
        let gpool: Vec<gmp::Z> = f.l.iter().map(|(a, _)| gmp::Z::of(a)).collect();
        let mut s = vec![
            if radix == 10 {
                mk("ours", "Display", &f.ours, |(a, _)| a.to_string())
            } else {
                mk("ours", "{:x}", &f.ours, |(a, _)| format!("{a:x}"))
            },
            if radix == 10 {
                mk("ruint", "Display", &f.ru, |(a, _)| a.to_string())
            } else {
                mk("ruint", "{:x}", &f.ru, |(a, _)| format!("{a:x}"))
            },
            mk("crypto-bigint", "to_string_radix_vartime", &f.cb, move |(a, _)| a.to_string_radix_vartime(radix)),
            mk("bnum", "to_str_radix", &f.bn, move |(a, _)| a.to_str_radix(radix)),
            if radix == 10 {
                mk("malachite-nz", "Display", &fnat, |(a, _)| a.to_string())
            } else {
                mk("malachite-nz", "{:x}", &fnat, |(a, _)| format!("{a:x}"))
            },
        ];
        #[cfg(has_gmp)]
        {
            let gpool = &gpool;
            let mut buf = vec![0u8; 24 * N + 8];
            for (z, w) in gpool.iter().zip(&want) {
                unsafe { gmp::__gmpz_get_str(buf.as_mut_ptr().cast(), radix as _, z.p()) };
                let s = std::ffi::CStr::from_bytes_until_nul(&buf).unwrap().to_str().unwrap();
                assert_eq!(s, w, "{name}: GMP");
            }
            s.push(side_n("GMP mpz_get_str", "into a caller's buffer, no allocation", move || {
                for z in std::hint::black_box(gpool) {
                    unsafe { gmp::__gmpz_get_str(buf.as_mut_ptr().cast(), radix as _, z.p()) };
                }
                std::hint::black_box(&mut buf);
            }));
        }
        if N == 1 {
            if radix == 10 {
                s.push(mk("u64", "Display", &f.pr, |(a, _)| lo64(*a).to_string()));
            } else {
                s.push(mk("u64", "{:x}", &f.pr, |(a, _)| format!("{:x}", lo64(*a))));
            }
        } else if N == 2 {
            if radix == 10 {
                s.push(mk("u128", "Display", &f.pr, |(a, _)| a.to_string()));
            } else {
                s.push(mk("u128", "{:x}", &f.pr, |(a, _)| format!("{a:x}")));
            }
        }
        out.row(g, name, &format!("{p}, random full width"), len, "a String per item", s);

        // and back
        let name = if radix == 10 { "from_str_radix(10)" } else { "from_str_radix(16)" };
        let texts: Vec<String> = want.clone();
        let wantv: Vec<[u64; N]> = full.l.iter().map(|(a, _)| *a).collect();
        same(name, "ours", &texts, &wantv, |t| *Uint::<N>::from_str_radix(t, radix).unwrap().as_limbs());
        same(name, "ruint", &texts, &wantv, |t| *Ru::<BITS, N>::from_str_radix(t, radix as u64).unwrap().as_limbs());
        same(name, "cb", &texts, &wantv, |t| Cb::<N>::from_str_radix_vartime(t, radix).unwrap().to_words());
        same(name, "bnum", &texts, &wantv, |t| *Bn::<N>::from_str_radix(t, radix).unwrap().digits());
        same(name, "malachite", &texts, &wantv, |t| {
            nat_limbs::<N>(&Natural::from_string_base(radix as u8, t).unwrap())
        });
        let mut s = vec![
            mk("ours", "", &texts, move |t| Uint::<N>::from_str_radix(t, radix).unwrap()),
            mk("ruint", "", &texts, move |t| Ru::<BITS, N>::from_str_radix(t, radix as u64).unwrap()),
            mk("crypto-bigint", "from_str_radix_vartime", &texts, move |t| {
                Cb::<N>::from_str_radix_vartime(t, radix).unwrap()
            }),
            mk("bnum", "", &texts, move |t| Bn::<N>::from_str_radix(t, radix).unwrap()),
            if radix == 10 {
                mk("malachite-nz", "FromStr, allocates", &texts, |t| Natural::from_str(t).unwrap())
            } else {
                mk("malachite-nz", "from_string_base, allocates", &texts, |t| {
                    Natural::from_string_base(16, t).unwrap()
                })
            },
        ];
        #[cfg(has_gmp)]
        {
            let cs: Vec<std::ffi::CString> = texts.iter().map(|t| std::ffi::CString::new(t.as_str()).unwrap()).collect();
            let mut r = gmp::Z::new();
            for (c, w) in cs.iter().zip(&wantv) {
                unsafe { gmp::__gmpz_set_str(r.m(), c.as_ptr(), radix as _) };
                assert_eq!(&r.limbs::<N>(), w, "{name}: GMP");
            }
            s.push(side_n("GMP mpz_set_str", "into a preallocated mpz", move || {
                for c in std::hint::black_box(&cs) {
                    unsafe { gmp::__gmpz_set_str(r.m(), c.as_ptr(), radix as _) };
                }
                std::hint::black_box(&mut r);
            }));
        }
        if N == 1 {
            s.push(mk("u64", "", &texts, move |t| u64::from_str_radix(t, radix).unwrap()));
        } else if N == 2 {
            s.push(mk("u128", "", &texts, move |t| u128::from_str_radix(t, radix).unwrap()));
        }
        out.row(g, name, &format!("{p}, random full width"), len, "", s);
    }

    // ---- bits, checked forms, rotations ---------------------------------------------------
    {
        let f = &full;
        // operands shifted right by a random amount, so that the leading zeros vary
        let lz = Pools::<N, BITS>::new(
            f.l.iter()
                .map(|(a, b)| (*Uint::<N>::from_limbs(*a).wrapping_shr(rng.below(BITS as u64) as u32).as_limbs(), *b))
                .collect(),
        );
        let want: Vec<u32> = lz.ours.iter().map(|(a, _)| a.leading_zeros()).collect();
        same("lz", "ruint", &lz.ru, &want, |(a, _)| a.leading_zeros() as u32);
        same("lz", "cb", &lz.cb, &want, |(a, _)| a.leading_zeros_vartime());
        same("lz", "bnum", &lz.bn, &want, |(a, _)| a.leading_zeros());
        let mut s = vec![
            mk("ours", "", &lz.ours, |(a, _)| a.leading_zeros()),
            mk("ruint", "", &lz.ru, |(a, _)| a.leading_zeros()),
            mk("crypto-bigint", "leading_zeros_vartime", &lz.cb, |(a, _)| a.leading_zeros_vartime()),
            mk("bnum", "", &lz.bn, |(a, _)| a.leading_zeros()),
        ];
        if N == 1 {
            s.push(mk("u64", "", &lz.pr, |(a, _)| lo64(*a).leading_zeros()));
        } else if N == 2 {
            s.push(mk("u128", "", &lz.pr, |(a, _)| a.leading_zeros()));
        }
        out.row(g, "leading_zeros", &format!("{p}, random bit lengths"), len, "", s);

        let want: Vec<u32> = f.ours.iter().map(|(a, _)| a.count_ones()).collect();
        same("popcnt", "ruint", &f.ru, &want, |(a, _)| a.count_ones() as u32);
        same("popcnt", "bnum", &f.bn, &want, |(a, _)| a.count_ones());
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, _)| a.count_ones()),
            mk("ruint", "", &f.ru, |(a, _)| a.count_ones()),
            mk("bnum", "", &f.bn, |(a, _)| a.count_ones()),
        ];
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, _)| lo64(*a).count_ones()));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, _)| a.count_ones()));
        }
        out.row(g, "count_ones", &p, len, "", s);

        let want: Vec<Option<[u64; N]>> = f.ours.iter().map(|(a, b)| a.checked_add(*b).map(|x| *x.as_limbs())).collect();
        same("cadd", "ruint", &f.ru, &want, |(a, b)| a.checked_add(*b).map(|x| *x.as_limbs()));
        same("cadd", "bnum", &f.bn, &want, |(a, b)| a.checked_add(*b).map(|x| *x.digits()));
        let mut s = vec![
            mk("ours", "", &f.ours, |(a, b)| a.checked_add(*b)),
            mk("ruint", "", &f.ru, |(a, b)| a.checked_add(*b)),
            mk("bnum", "", &f.bn, |(a, b)| a.checked_add(*b)),
        ];
        if N == 1 {
            s.push(mk("u64", "", &f.pr, |(a, b)| lo64(*a).checked_add(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &f.pr, |(a, b)| a.checked_add(*b)));
        }
        out.row(g, "checked_add", &format!("{p}, random operands (half overflow)"), len, "", s);

        // products that overflow about half the time: bit lengths summing to BITS ± 32
        let mixed = Pools::<N, BITS>::new(
            (0..len)
                .map(|_| {
                    let k = 1 + rng.below(BITS as u64 - 1) as i64;
                    let j = (BITS as i64 - k + rng.below(65) as i64 - 32).clamp(1, BITS as i64);
                    (bits(&mut rng, k as u32), bits(&mut rng, j as u32))
                })
                .collect(),
        );
        let m = &mixed;
        let want: Vec<Option<[u64; N]>> = m.ours.iter().map(|(a, b)| a.checked_mul(*b).map(|x| *x.as_limbs())).collect();
        same("cmul", "ruint", &m.ru, &want, |(a, b)| a.checked_mul(*b).map(|x| *x.as_limbs()));
        same("cmul", "bnum", &m.bn, &want, |(a, b)| a.checked_mul(*b).map(|x| *x.digits()));
        let mut s = vec![
            mk("ours", "", &m.ours, |(a, b)| a.checked_mul(*b)),
            mk("ruint", "", &m.ru, |(a, b)| a.checked_mul(*b)),
            mk("bnum", "", &m.bn, |(a, b)| a.checked_mul(*b)),
        ];
        if N == 1 {
            s.push(mk("u64", "", &m.pr, |(a, b)| lo64(*a).checked_mul(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &m.pr, |(a, b)| a.checked_mul(*b)));
        }
        out.row(g, "checked_mul", &format!("{p}, bit lengths summing to {BITS} ± 32"), len, "", s);

        let want: Vec<[u64; N]> = m.ours.iter().map(|(a, b)| *a.saturating_mul(*b).as_limbs()).collect();
        same("smul", "ruint", &m.ru, &want, |(a, b)| *a.saturating_mul(*b).as_limbs());
        same("smul", "bnum", &m.bn, &want, |(a, b)| *a.saturating_mul(*b).digits());
        let mut s = vec![
            mk("ours", "", &m.ours, |(a, b)| a.saturating_mul(*b)),
            mk("ruint", "", &m.ru, |(a, b)| a.saturating_mul(*b)),
            mk("bnum", "", &m.bn, |(a, b)| a.saturating_mul(*b)),
        ];
        if N == 1 {
            s.push(mk("u64", "", &m.pr, |(a, b)| lo64(*a).saturating_mul(lo64(*b))));
        } else if N == 2 {
            s.push(mk("u128", "", &m.pr, |(a, b)| a.saturating_mul(*b)));
        }
        out.row(g, "saturating_mul", &format!("{p}, bit lengths summing to {BITS} ± 32"), len, "", s);

        let k: Vec<([u64; N], u32)> = f.l.iter().map(|(a, _)| (*a, rng.below(BITS as u64) as u32)).collect();
        let ok: Vec<(Uint<N>, u32)> = k.iter().map(|(a, s)| (Uint::from_limbs(*a), *s)).collect();
        let rk: Vec<(Ru<BITS, N>, usize)> = k.iter().map(|(a, s)| (Ru::from_limbs(*a), *s as usize)).collect();
        let bk: Vec<(Bn<N>, u32)> = k.iter().map(|(a, s)| (Bn::from_digits(*a), *s)).collect();
        let pk: Vec<(u128, u32)> = k.iter().map(|(a, s)| (p128(a), *s)).collect();
        let want: Vec<[u64; N]> = ok.iter().map(|(a, s)| *a.rotate_left(*s).as_limbs()).collect();
        same("rotl", "ruint", &rk, &want, |(a, s)| *a.rotate_left(*s).as_limbs());
        // bnum masks a rotation's amount with BITS − 1, which is wrong at a width that is not a
        // power of two (known from #25): its row is left out there.
        let bn_ok = bk.iter().zip(&want).all(|((a, s), w)| a.rotate_left(*s).digits() == w);
        let mut s = vec![
            mk("ours", "", &ok, |(a, s)| a.rotate_left(*s)),
            mk("ruint", "", &rk, |(a, s)| a.rotate_left(*s)),
        ];
        if bn_ok {
            s.push(mk("bnum", "", &bk, |(a, s)| a.rotate_left(*s)));
        }
        if N == 1 {
            s.push(mk("u64", "", &pk, |(a, s)| lo64(*a).rotate_left(*s)));
        } else if N == 2 {
            s.push(mk("u128", "", &pk, |(a, s)| a.rotate_left(*s)));
        }
        out.row(
            g,
            "rotate_left",
            &format!("{p}, random amount below {BITS}"),
            len,
            if bn_ok { "" } else { "bnum left out: its rotate_left masks the amount with BITS − 1, wrong at this width" },
            s,
        );
    }

    // ---- signed -------------------------------------------------------------------------
    {
        let gi = "int";
        let l: Vec<([u64; N], [u64; N])> = full.l.clone();
        let oi: Vec<(Int<N>, Int<N>)> = l.iter().map(|(a, b)| (Int::from_limbs(*a), Int::from_limbs(*b))).collect();
        let bi: Vec<(Bi<N>, Bi<N>)> =
            l.iter().map(|(a, b)| (Bi::from_bits(Bn::from_digits(*a)), Bi::from_bits(Bn::from_digits(*b)))).collect();
        let pi: Vec<(i128, i128)> = l.iter().map(|(a, b)| (p128(a) as i128, p128(b) as i128)).collect();
        let lo = |x: i128| x as i64;
        let want: Vec<[u64; N]> = oi.iter().map(|(a, b)| *a.wrapping_mul(*b).as_limbs()).collect();
        same("imul", "bnum", &bi, &want, |(a, b)| *a.wrapping_mul(*b).to_bits().digits());
        let mut s = vec![
            mk("ours", "", &oi, |(a, b)| a.wrapping_mul(*b)),
            mk("bnum", "BInt", &bi, |(a, b)| a.wrapping_mul(*b)),
        ];
        if N == 1 {
            s.push(mk("i64", "", &pi, |(a, b)| lo(*a).wrapping_mul(lo(*b))));
        } else if N == 2 {
            s.push(mk("i128", "", &pi, |(a, b)| a.wrapping_mul(*b)));
        }
        out.row(gi, "wrapping_mul", &format!("{p}, random signs"), len, "", s);

        let want: Vec<[u64; N]> = oi.iter().map(|(a, _)| *a.wrapping_abs().as_limbs()).collect();
        same("iabs", "bnum", &bi, &want, |(a, _)| *a.wrapping_abs().to_bits().digits());
        let mut s = vec![
            mk("ours", "wrapping_abs", &oi, |(a, _)| a.wrapping_abs()),
            mk("bnum", "BInt::wrapping_abs", &bi, |(a, _)| a.wrapping_abs()),
        ];
        if N == 1 {
            s.push(mk("i64", "", &pi, |(a, _)| lo(*a).wrapping_abs()));
        } else if N == 2 {
            s.push(mk("i128", "", &pi, |(a, _)| a.wrapping_abs()));
        }
        out.row(gi, "wrapping_abs", &format!("{p}, random signs"), len, "", s);

        let sh: Vec<u32> = (0..len).map(|_| rng.below(BITS as u64) as u32).collect();
        let ois: Vec<(Int<N>, u32)> = oi.iter().zip(&sh).map(|((a, _), k)| (*a, *k)).collect();
        let bis: Vec<(Bi<N>, u32)> = bi.iter().zip(&sh).map(|((a, _), k)| (*a, *k)).collect();
        let pis: Vec<(i128, u32)> = pi.iter().zip(&sh).map(|((a, _), k)| (*a, *k)).collect();
        let want: Vec<[u64; N]> = ois.iter().map(|(a, k)| *a.wrapping_shr(*k).as_limbs()).collect();
        same("isar", "bnum", &bis, &want, |(a, k)| *a.wrapping_shr(*k).to_bits().digits());
        let mut s = vec![
            mk("ours", "", &ois, |(a, k)| a.wrapping_shr(*k)),
            mk("bnum", "BInt", &bis, |(a, k)| a.wrapping_shr(*k)),
        ];
        if N == 1 {
            s.push(mk("i64", "", &pis, |(a, k)| lo(*a).wrapping_shr(*k)));
        } else if N == 2 {
            s.push(mk("i128", "", &pis, |(a, k)| a.wrapping_shr(*k)));
        }
        out.row(gi, "wrapping_shr (arithmetic)", &format!("{p}, random signs and amounts"), len, "", s);

        // truncated division by a divisor of ⌈N/2⌉ limbs of either sign
        let vl = N.div_ceil(2);
        let l: Vec<([u64; N], [u64; N])> = (0..len)
            .map(|_| {
                let a: [u64; N] = rng.limbs();
                let mut b: [u64; N] = sized(&mut rng, vl);
                if vl == N {
                    b[N - 1] >>= 1;
                    b[N - 1] |= 1 << 62;
                }
                let b = if rng.next() & 1 == 1 { *Int::from_limbs(b).wrapping_neg().as_limbs() } else { b };
                (a, b)
            })
            .collect();
        let oi: Vec<(Int<N>, Int<N>)> = l.iter().map(|(a, b)| (Int::from_limbs(*a), Int::from_limbs(*b))).collect();
        let bi: Vec<(Bi<N>, Bi<N>)> =
            l.iter().map(|(a, b)| (Bi::from_bits(Bn::from_digits(*a)), Bi::from_bits(Bn::from_digits(*b)))).collect();
        let ci: Vec<(CbI<N>, NonZero<CbI<N>>)> = l
            .iter()
            .map(|(a, b)| (CbI::from_words(*a), NonZero::new(CbI::from_words(*b)).unwrap()))
            .collect();
        let pi: Vec<(i128, i128)> = l.iter().map(|(a, b)| (p128(a) as i128, p128(b) as i128)).collect();
        let want: Vec<([u64; N], [u64; N])> = oi
            .iter()
            .map(|(a, b)| {
                let (q, r) = a.div_rem(*b);
                (*q.as_limbs(), *r.as_limbs())
            })
            .collect();
        same("idiv", "bnum", &bi, &want, |(a, b)| {
            let (q, r) = Integer::div_rem(a, b);
            (*q.to_bits().digits(), *r.to_bits().digits())
        });
        same("idiv", "cb", &ci, &want, |(a, b)| {
            let (q, r) = a.checked_div_rem_vartime(b);
            (q.expect("quotient").to_words(), r.to_words())
        });
        let mut s = vec![
            mk("ours", "", &oi, |(a, b)| a.div_rem(*b)),
            mk("bnum", "BInt, num_integer::Integer::div_rem", &bi, |(a, b)| Integer::div_rem(a, b)),
            mk("crypto-bigint", "Int::checked_div_rem_vartime", &ci, |(a, b)| a.checked_div_rem_vartime(b)),
        ];
        if N == 1 {
            s.push(mk("i64", "", &pi, |(a, b)| (lo(*a) / lo(*b), lo(*a) % lo(*b))));
        } else if N == 2 {
            s.push(mk("i128", "", &pi, |(a, b)| (a / b, a % b)));
        }
        out.row(gi, "div_rem", &format!("{p}, v of {vl} limbs, random signs"), len, "", s);

        // text
        let oi: Vec<(Int<N>, ())> = full.l.iter().map(|(a, _)| (Int::from_limbs(*a), ())).collect();
        let bi: Vec<(Bi<N>, ())> = full.l.iter().map(|(a, _)| (Bi::from_bits(Bn::from_digits(*a)), ())).collect();
        let pi: Vec<(i128, ())> = full.l.iter().map(|(a, _)| (p128(a) as i128, ())).collect();
        let want: Vec<String> = oi.iter().map(|(a, _)| a.to_string()).collect();
        same("ito_string", "bnum", &bi, &want, |(a, _)| a.to_string());
        let mut s = vec![
            mk("ours", "Display", &oi, |(a, _)| a.to_string()),
            mk("bnum", "BInt, Display", &bi, |(a, _)| a.to_string()),
        ];
        if N == 1 {
            s.push(mk("i64", "", &pi, |(a, _)| lo(*a).to_string()));
        } else if N == 2 {
            s.push(mk("i128", "", &pi, |(a, _)| a.to_string()));
        }
        out.row(gi, "to_string", &format!("{p}, random signs"), len, "a String per item", s);

        let wantv: Vec<[u64; N]> = full.l.iter().map(|(a, _)| *a).collect();
        same("iparse", "ours", &want, &wantv, |t| *t.parse::<Int<N>>().unwrap().as_limbs());
        same("iparse", "bnum", &want, &wantv, |t| *Bi::<N>::from_str_radix(t, 10).unwrap().to_bits().digits());
        let mut s = vec![
            mk("ours", "str::parse", &want, |t| t.parse::<Int<N>>().unwrap()),
            mk("bnum", "BInt::from_str_radix", &want, |t| Bi::<N>::from_str_radix(t, 10).unwrap()),
        ];
        if N == 1 {
            s.push(mk("i64", "", &want, |t| t.parse::<i64>().unwrap()));
        } else if N == 2 {
            s.push(mk("i128", "", &want, |t| t.parse::<i128>().unwrap()));
        }
        out.row(gi, "from_str_radix(10)", &format!("{p}, random signs"), len, "", s);
    }
    let _ = prim_name;
}

fn main() {
    let out = Out::new("numtypes");
    let only: Option<usize> = std::env::var("NTBENCH_N").ok().and_then(|s| s.parse().ok());
    let runs: [(usize, fn(&Out, u64)); 6] = [
        (1, bench::<1, 64, 2, 128>),
        (2, bench::<2, 128, 4, 256>),
        (4, bench::<4, 256, 8, 512>),
        (6, bench::<6, 384, 12, 768>),
        (8, bench::<8, 512, 16, 1024>),
        (16, bench::<16, 1024, 32, 2048>),
    ];
    for (n, f) in runs {
        if only.is_none_or(|k| k == n) {
            f(&out, n as u64);
        }
    }
}
