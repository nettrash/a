//! The kernels of the arithmetic against the exact results rounded by integers: precisions at the
//! boundaries of one and two limbs and of the windows, operands longer and shorter than their
//! results, exponents apart by the amounts where the kernels change their method, terms that
//! cancel, squares, and the assigning forms against the others.

use super::common::{MODES, below, mantissa, mode, rational, reference, same, size, xorshift};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

/// Precisions where a kernel changes: one and two limbs, their last bits, the window's.
const PRECS: [u64; 30] = [
    1, 2, 3, 23, 24, 25, 52, 53, 54, 62, 63, 64, 65, 66, 100, 126, 127, 128, 129, 130, 191, 192,
    193, 255, 256, 257, 320, 511, 640, 1000,
];

/// A random precision: one of [`PRECS`] or any up to 1100.
fn prec(s: &mut u64) -> u64 {
    if below(s, 3) == 0 {
        1 + below(s, 1100)
    } else {
        PRECS[below(s, PRECS.len() as u64) as usize]
    }
}

/// A random float of the precision `p` and the exponent `e` (`0.1…·2^e`), of a random sign, its
/// mantissa all of the precision with long runs of zeros and ones.
fn of(s: &mut u64, p: u64, e: i64) -> BigFloat {
    let mut m = if below(s, 2) == 0 {
        mantissa(s, p)
    } else {
        limbed(s, p)
    };
    if xorshift(s) & 1 == 1 {
        m = -m;
    }
    let (x, o) = BigFloat::from_parts(&m, e - p as i64, p, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

/// A random integer of exactly `bits` bits made of 64-bit pieces from its top: zeros, ones, a
/// lone bit at either end, all but one of them, or random, the patterns that put a limb of a
/// window at its boundaries.
fn limbed(s: &mut u64, bits: u64) -> Number {
    let k = bits.div_ceil(64);
    let mut m = Number::from(0u8);
    for _ in 0..k {
        let piece: u64 = match below(s, 9) {
            0 => 0,
            1 => u64::MAX,
            2 => 1 << 63,
            3 => (1 << 63) - 1,
            4 => 1,
            5 => u64::MAX - 1,
            6 => (1 << 63) + 1,
            7 => (1 << 62) | 1,
            _ => xorshift(s),
        };
        m = (m << 64usize) + Number::from(piece);
    }
    let mut m = m >> (64 * k - bits) as usize;
    m.set_bit(bits - 1, true);
    m
}

/// A shift of exponents where the kernels change: 0, 1, 2, near a limb, two limbs, the
/// precision, far.
fn shift(s: &mut u64, p: u64) -> i64 {
    let k = match below(s, 10) {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 62 + below(s, 5),
        4 => 126 + below(s, 5),
        5 => p.saturating_sub(2) + below(s, 6),
        6 => p + 60 + below(s, 10),
        7 => 1000 + below(s, 3000),
        8 => 64 * (1 + below(s, 4)) + below(s, 2),
        _ => below(s, 200),
    };
    k as i64
}

/// Checks `got` against the exact `want` rounded, or the zero of an exact zero sum.
#[track_caller]
fn check(
    got: &(BigFloat, Ordering),
    want: &Rational,
    down: bool,
    prec: u64,
    round: Round,
    what: &str,
) {
    if want.is_zero() {
        let z = if down {
            BigFloat::neg_zero(prec)
        } else {
            BigFloat::zero(prec)
        };
        same(got, &(z, Ordering::Equal), what);
    } else {
        same(got, &reference(want, prec, round), what);
    }
}

/// An operation with the precision and the mode of its result.
type Op = fn(&BigFloat, &BigFloat, u64, Round) -> (BigFloat, Ordering);

/// The assigning forms give what the others give.
#[track_caller]
fn assigning(a: &BigFloat, b: &BigFloat, prec: u64, round: Round) {
    let target = |v: &BigFloat| v.prec_round(prec, Round::Nearest).0;
    let ops: [(&str, Op); 4] = [
        ("+", |x, y, p, r| x.add_round(y, p, r)),
        ("-", |x, y, p, r| x.sub_round(y, p, r)),
        ("*", |x, y, p, r| x.mul_round(y, p, r)),
        ("/", |x, y, p, r| x.div_round(y, p, r)),
    ];
    for (i, (name, f)) in ops.iter().enumerate() {
        let want = f(a, b, prec, round);
        // `set_*` into a number of the precision, of any value before.
        let mut r = if i % 2 == 0 {
            target(a)
        } else {
            BigFloat::nan(prec)
        };
        let o = match i {
            0 => r.set_add(a, b, round),
            1 => r.set_sub(a, b, round),
            2 => r.set_mul(a, b, round),
            _ => r.set_div(a, b, round),
        };
        same(&(r, o), &want, &format!("set {a:?} {name} {b:?}"));
        // `*_assign_round` on a copy of `a` at the precision.
        let mut x = target(a);
        let xa = x.clone();
        let want = f(&xa, b, prec, round);
        let o = match i {
            0 => x.add_assign_round(b, round),
            1 => x.sub_assign_round(b, round),
            2 => x.mul_assign_round(b, round),
            _ => x.div_assign_round(b, round),
        };
        same(&(x, o), &want, &format!("assign {xa:?} {name} {b:?}"));
    }
    // The operators, to nearest at the larger precision.
    let p = a.prec().max(b.prec());
    let mut x = a.clone();
    x += b;
    same(
        &(x, Ordering::Equal),
        &(a.add_round(b, p, Round::Nearest).0, Ordering::Equal),
        "+=",
    );
    let mut x = a.clone();
    x *= b;
    same(
        &(x, Ordering::Equal),
        &(a.mul_round(b, p, Round::Nearest).0, Ordering::Equal),
        "*=",
    );
    same(
        &(a.clone() - b, Ordering::Equal),
        &(a.sub_round(b, p, Round::Nearest).0, Ordering::Equal),
        "-",
    );
    same(
        &(a.clone() / b.clone(), Ordering::Equal),
        &(a.div_round(b, p, Round::Nearest).0, Ordering::Equal),
        "/",
    );
    // The operators on borrowed operands, a new result.
    for (name, got, want) in [
        ("&+", a + b, a.add_round(b, p, Round::Nearest).0),
        ("&-", a - b, a.sub_round(b, p, Round::Nearest).0),
        ("&*", a * b, a.mul_round(b, p, Round::Nearest).0),
        ("&/", a / b, a.div_round(b, p, Round::Nearest).0),
    ] {
        same(&(got, Ordering::Equal), &(want, Ordering::Equal), name);
    }
    let mut x = target(a);
    let xa = x.clone();
    let o = x.sqrt_assign_round(round);
    same(&(x, o), &xa.sqrt_round(prec, round), "sqrt_assign_round");
}

#[test]
fn sums_round_correctly_at_every_kernel() {
    let mut s = 0x0A11_0F5E_ED00_0001u64;
    for i in 0..size(6000, 60) {
        let (pa, pb, p) = (prec(&mut s), prec(&mut s), prec(&mut s));
        let ea = below(&mut s, 200) as i64 - 100;
        let a = of(&mut s, pa, ea);
        // `b` below `a` by a shift, or `-a` nudged (a difference that cancels), or `a` itself.
        let b = match i % 5 {
            0 => {
                let k = shift(&mut s, pa.max(pb));
                (-&a)
                    .prec_round(pb, mode(&mut s))
                    .0
                    .mul_round(
                        &BigFloat::from_parts(&super::common::one(), -k, 1, Round::Nearest).0,
                        pb,
                        Round::Nearest,
                    )
                    .0
            }
            1 => {
                let k = 1 + below(&mut s, (pa + 2) * 2);
                let pt = 1 + below(&mut s, 70);
                let tiny = of(&mut s, pt, ea - k as i64);
                (-&a).sub_round(&tiny, pb.max(pa) + 64, Round::Nearest).0
            }
            _ => {
                let k = shift(&mut s, p);
                of(&mut s, pb, ea - k)
            }
        };
        let (a, b) = if xorshift(&mut s) & 1 == 1 {
            (b, a)
        } else {
            (a, b)
        };
        let (ra, rb) = (rational(&a), rational(&b));
        for round in MODES {
            let down = round == Round::Down;
            check(
                &a.add_round(&b, p, round),
                &(&ra + &rb),
                down,
                p,
                round,
                &format!("{a:?} + {b:?} at {p}"),
            );
            check(
                &a.sub_round(&b, p, round),
                &(&ra - &rb),
                down,
                p,
                round,
                &format!("{a:?} - {b:?} at {p}"),
            );
        }
        if i % 16 == 0 {
            assigning(&a, &b, p, mode(&mut s));
        }
    }
}

#[test]
fn products_quotients_and_roots_round_correctly_at_every_kernel() {
    let mut s = 0xB0_D1E5_0002u64;
    for i in 0..size(4000, 40) {
        let (pa, pb, p) = (prec(&mut s), prec(&mut s), prec(&mut s));
        let (ea, eb) = (
            below(&mut s, 300) as i64 - 150,
            below(&mut s, 300) as i64 - 150,
        );
        let a = of(&mut s, pa, ea);
        let b = of(&mut s, pb, eb);
        let (ra, rb) = (rational(&a), rational(&b));
        for round in MODES {
            check(
                &a.mul_round(&b, p, round),
                &(&ra * &rb),
                false,
                p,
                round,
                &format!("{a:?} * {b:?} at {p}"),
            );
            check(
                &a.div_round(&b, p, round),
                &(&ra / &rb),
                false,
                p,
                round,
                &format!("{a:?} / {b:?} at {p}"),
            );
            check(
                &a.mul_round(&a, p, round),
                &(&ra * &ra),
                false,
                p,
                round,
                &format!("{a:?}^2 at {p}"),
            );
            let x = a.abs();
            let got = x.sqrt_round(p, round);
            // The root against its square: `r^2` and the neighbour's square bracket `x`.
            let r = rational(&got.0);
            let rx = rational(&x);
            let sq = &r * &r;
            assert_eq!(sq.cmp(&rx), got.1, "sqrt {x:?} at {p} {round:?}: {got:?}");
            if got.1 != Ordering::Equal {
                let ulp = Rational::try_from(
                    &BigFloat::from_parts(
                        &super::common::one(),
                        got.0.exponent().unwrap() - p as i64,
                        1,
                        Round::Nearest,
                    )
                    .0,
                )
                .unwrap();
                let other = if got.1 == Ordering::Less {
                    &r + &ulp
                } else {
                    &r - &ulp
                };
                let so = &other * &other;
                assert_eq!(
                    so.cmp(&rx),
                    got.1.reverse(),
                    "sqrt {x:?} at {p} {round:?} not the neighbour"
                );
                if round == Round::Nearest {
                    // The midpoint's square lies on the side of the result's.
                    let mid = (&r + &other) / Rational::from(2u32);
                    let sm = &mid * &mid;
                    assert_eq!(
                        sm.cmp(&rx),
                        got.1.reverse(),
                        "sqrt {x:?} at {p} not nearest"
                    );
                }
            }
        }
        if i % 3 == 0 {
            let (pc, ec) = (prec(&mut s), below(&mut s, 400) as i64 - 200);
            let c = of(&mut s, pc, ec);
            let rc = rational(&c);
            let round = mode(&mut s);
            check(
                &a.mul_add_round(&b, &c, p, round),
                &(&(&ra * &rb) + &rc),
                round == Round::Down,
                p,
                round,
                &format!("{a:?} * {b:?} + {c:?} at {p}"),
            );
            let mut r = BigFloat::zero(p);
            let o = r.set_mul_add(&a, &b, &c, round);
            same(&(r, o), &a.mul_add_round(&b, &c, p, round), "set_mul_add");
            let mut r = BigFloat::zero(p);
            let o = r.set_sqrt(&a.abs(), round);
            same(&(r, o), &a.abs().sqrt_round(p, round), "set_sqrt");
        }
    }
}

#[test]
fn a_difference_of_neighbours_cancels_exactly() {
    let mut s = 0xCA_9CE1_0003u64;
    for _ in 0..size(2000, 30) {
        let pa = prec(&mut s);
        let a = of(&mut s, pa, 7);
        // The neighbour of `a` at a precision near it, and a precision for the difference.
        let (b, _) = a.prec_round(pa.saturating_sub(below(&mut s, 3)).max(1), mode(&mut s));
        let b = b
            .add_round(
                &BigFloat::from_parts(
                    &super::common::one(),
                    7 - pa as i64 - below(&mut s, 3) as i64,
                    1,
                    Round::Nearest,
                )
                .0,
                pa + 2,
                Round::Nearest,
            )
            .0;
        let p = prec(&mut s);
        let (ra, rb) = (rational(&a), rational(&b));
        for round in MODES {
            check(
                &a.sub_round(&b, p, round),
                &(&ra - &rb),
                round == Round::Down,
                p,
                round,
                &format!("{a:?} - {b:?} at {p}"),
            );
            check(
                &b.sub_round(&a, p, round),
                &(&rb - &ra),
                round == Round::Down,
                p,
                round,
                &format!("{b:?} - {a:?} at {p}"),
            );
        }
    }
}

#[test]
fn exact_quotients_by_long_divisors_and_roots_with_long_tails() {
    let mut s = 0xD1_7A11_0004u64;
    for _ in 0..size(1500, 15) {
        // `x = y·c` exactly for a long `y` and a short `c`: `x / y = c` exactly, and a hair off.
        let (py, pc) = (300 + below(&mut s, 900), 1 + below(&mut s, 130));
        let (ey, ec) = (below(&mut s, 40) as i64 - 20, below(&mut s, 40) as i64 - 20);
        let y = of(&mut s, py, ey);
        let c = of(&mut s, pc, ec);
        let x = y.mul_round(&c, py + pc, Round::Nearest);
        assert_eq!(x.1, Ordering::Equal);
        let x = x.0;
        let k = 1 + below(&mut s, 2 * (py + pc));
        let hair = BigFloat::from_parts(
            &super::common::one(),
            x.exponent().unwrap() - k as i64,
            1,
            Round::Nearest,
        )
        .0;
        let x2 = x.add_round(&hair, py + pc + k + 2, Round::Nearest).0;
        let p = (pc as i64 + below(&mut s, 5) as i64 - 2).max(1) as u64;
        let (ry, rc) = (rational(&y), rational(&c));
        let rx2 = rational(&x2);
        for round in MODES {
            same(
                &x.div_round(&y, p, round),
                &reference(&rc, p, round),
                &format!("{x:?} / {y:?} at {p}"),
            );
            check(
                &x2.div_round(&y, p, round),
                &(&rx2 / &ry),
                false,
                p,
                round,
                &format!("{x2:?} / {y:?} at {p}"),
            );
            // `c^2` and `c^2(1 + 2^-k)` at length: the root `|c|` exactly, and just above it.
            let c2 = c.mul_round(&c, 2 * pc, Round::Nearest).0;
            same(
                &c2.sqrt_round(p, round),
                &reference(&rc.abs(), p, round),
                "sqrt c^2",
            );
            let long = c2
                .add_round(
                    &BigFloat::from_parts(
                        &super::common::one(),
                        c2.exponent().unwrap() - k as i64,
                        1,
                        Round::Nearest,
                    )
                    .0,
                    2 * pc + k + 2,
                    Round::Nearest,
                )
                .0;
            let got = long.sqrt_round(p, round);
            let r = rational(&got.0);
            assert_eq!(
                (&r * &r).cmp(&rational(&long)),
                got.1,
                "sqrt {long:?} at {p} {round:?}"
            );
            assert_ne!(got.1, Ordering::Equal, "sqrt {long:?}");
        }
    }
}

#[test]
fn primitives_and_numbers_on_either_side_round_at_the_floats_precision() {
    let x = BigFloat::from(1)
        .div_round(&BigFloat::from(3), 10, Round::Nearest)
        .0;
    let at = |v: BigFloat, p: u64| v.prec_round(p, Round::Nearest).0;
    // `x + 1` rounds to 10 bits, as `x + BigFloat::from(1)` at 10 would.
    let one = BigFloat::from(1);
    same(
        &(&x + 1, Ordering::Equal),
        &(x.add_round(&one, 10, Round::Nearest).0, Ordering::Equal),
        "x + 1",
    );
    same(
        &(1u8 + &x, Ordering::Equal),
        &(one.add_round(&x, 10, Round::Nearest).0, Ordering::Equal),
        "1 + x",
    );
    let two = BigFloat::from(2.0);
    same(
        &(2.0 * &x, Ordering::Equal),
        &(two.mul_round(&x, 10, Round::Nearest).0, Ordering::Equal),
        "2.0 * x",
    );
    let n = Number::from(7);
    let seven = BigFloat::from(&n);
    same(
        &(&x / &n, Ordering::Equal),
        &(x.div_round(&seven, 10, Round::Nearest).0, Ordering::Equal),
        "x / n",
    );
    same(
        &(&n - &x, Ordering::Equal),
        &(seven.sub_round(&x, 10, Round::Nearest).0, Ordering::Equal),
        "n - x",
    );
    let mut y = x.clone();
    y -= 3u8;
    same(
        &(y, Ordering::Equal),
        &(
            x.sub_round(&BigFloat::from(3), 10, Round::Nearest).0,
            Ordering::Equal,
        ),
        "x -= 3",
    );
    let mut y = x.clone();
    y *= -2i64;
    same(
        &(y.clone(), Ordering::Equal),
        &(at(-(&x * &two), 10), Ordering::Equal),
        "x *= -2",
    );
    let mut y = x.clone();
    y /= f32::INFINITY;
    assert!(y.is_zero() && y.prec() == 10);
    assert!((&x + f64::NAN).is_nan());
    let big = u128::MAX;
    same(
        &(&x * big, Ordering::Equal),
        &(
            x.mul_round(&BigFloat::from(big), 10, Round::Nearest).0,
            Ordering::Equal,
        ),
        "x * u128",
    );
    let z = &BigFloat::zero(30) - 0i32;
    assert!(z.is_zero() && z.is_sign_positive() && z.prec() == 30);
    let w = BigFloat::from(0.1f32) + BigFloat::from(5u8) - i128::MIN;
    assert_eq!(w.prec(), 24);
    let mut s = 0x9817_0005u64;
    for _ in 0..size(300, 10) {
        let a = super::common::float(&mut s, 200, 20);
        let v = xorshift(&mut s) as i64;
        let fv = f64::from_bits(xorshift(&mut s) >> 2);
        for (got, want) in [
            (
                &a + v,
                a.add_round(&BigFloat::from(v), a.prec(), Round::Nearest).0,
            ),
            (
                v - &a,
                BigFloat::from(v).sub_round(&a, a.prec(), Round::Nearest).0,
            ),
            (
                &a * fv,
                a.mul_round(&BigFloat::from(fv), a.prec(), Round::Nearest).0,
            ),
            (
                fv / &a,
                BigFloat::from(fv).div_round(&a, a.prec(), Round::Nearest).0,
            ),
        ] {
            same(
                &(got, Ordering::Equal),
                &(want, Ordering::Equal),
                "primitive",
            );
        }
    }
}

#[test]
fn operands_of_the_results_length_round_correctly() {
    // Both operands and the result of `n` limbs, `n ≥ 3`: every precision of the limbs, the last
    // bit of the last limb included, exponents apart by every amount where the shift changes,
    // and halfway sums.
    let mut s = 0x5A3E_0007u64;
    for i in 0..size(6000, 40) {
        let n = 3 + below(&mut s, 4);
        let pick = |s: &mut u64| {
            if below(s, 3) == 0 {
                64 * n
            } else {
                64 * (n - 1) + 1 + below(s, 64)
            }
        };
        let (pa, pb, p) = (pick(&mut s), pick(&mut s), pick(&mut s));
        let a = of(&mut s, pa, 0);
        let k = match below(&mut s, 6) {
            0 => below(&mut s, 3),
            1 => 64 * below(&mut s, n + 1) + below(&mut s, 3),
            2 => 64 * n - 1 + below(&mut s, 4),
            3 => p - 1 + below(&mut s, 3),
            _ => below(&mut s, 64 * n + 8),
        } as i64;
        let b = of(&mut s, pb, -k);
        // A halfway sum: `b` the half of the last place of `a` at `p`, or a multiple of it.
        let b = if i % 7 == 0 {
            let half =
                BigFloat::from_parts(&super::common::one(), -(p as i64) - 1, 1, Round::Nearest).0;
            let h = if xorshift(&mut s) & 1 == 1 {
                -half
            } else {
                half
            };
            h.mul_round(
                &BigFloat::from(1 + 2 * below(&mut s, 3) as i64),
                pb,
                Round::Nearest,
            )
            .0
        } else {
            b
        };
        let (ra, rb) = (rational(&a), rational(&b));
        for round in MODES {
            let down = round == Round::Down;
            check(
                &a.add_round(&b, p, round),
                &(&ra + &rb),
                down,
                p,
                round,
                &format!("{a:?} + {b:?} at {p}"),
            );
            check(
                &a.sub_round(&b, p, round),
                &(&ra - &rb),
                down,
                p,
                round,
                &format!("{a:?} - {b:?} at {p}"),
            );
        }
    }
}

#[test]
fn square_roots_of_long_mantissas_at_short_precisions() {
    // `s^2·(1 + 2^-k)` of three limbs and more: the bits below the top two limbs of the radicand,
    // the one shifted out for an odd exponent among them, make the root inexact.
    let mut st = 0x5097_0008u64;
    for _ in 0..size(400, 10) {
        let bs = 1 + below(&mut st, 31);
        let m = super::common::mantissa(&mut st, bs);
        let e = below(&mut st, 20) as i64 - 10;
        let r = BigFloat::from_parts(&m, e, bs, Round::Nearest).0;
        let sq = r.mul_round(&r, 2 * bs, Round::Nearest).0;
        for k in 100..200i64 {
            let t = BigFloat::from_parts(
                &super::common::one(),
                sq.exponent().unwrap() - k,
                1,
                Round::Nearest,
            )
            .0;
            let x = sq.add_round(&t, 260, Round::Nearest).0;
            let p = 1 + below(&mut st, 63);
            let round = mode(&mut st);
            let got = x.sqrt_round(p, round);
            let g = rational(&got.0);
            assert_eq!(
                (&g * &g).cmp(&rational(&x)),
                got.1,
                "sqrt {x:?} at {p} {round:?}"
            );
            assert_ne!(got.1, Ordering::Equal);
        }
    }
}

/// A value of the vectors: `prec:` then `nan`, `inf`, `-inf`, `0`, `-0` or `<hex>p<exponent>`.
fn field(s: &str) -> BigFloat {
    let (p, v) = s.split_once(':').expect("prec:value");
    super::vectors::float(v, p.parse().expect("a precision"))
}

/// MPFR's results for the forms into an existing number (`set_add` and the others, and the
/// assigning forms), into numbers of the result's precision that hold a value of their own (a
/// finite one, so that the kernels write into its limbs, or a special value, which has none),
/// with operands of the result's precision, of at most it in as many limbs, and of other lengths,
/// near the ends of the exponent range too; the forms with a new result alike.
#[test]
fn forms_into_numbers_match_mpfr() {
    let mut count = 0;
    for (i, line) in include_str!("data/arith_into.txt")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .step_by(size(1, 13))
        .enumerate()
    {
        let f: Vec<&str> = line.split(' ').collect();
        let eq = f.iter().position(|&t| t == "=").expect("=");
        let (op, p) = (f[0], f[1].parse::<u64>().expect("a precision"));
        let round = super::vectors::mode(f[2]);
        let x: Vec<BigFloat> = f[3..eq].iter().map(|t| field(t)).collect();
        let want = (field(f[eq + 1]), super::vectors::ternary(f[eq + 2]));
        let new = match op {
            "add" => x[0].add_round(&x[1], p, round),
            "sub" => x[0].sub_round(&x[1], p, round),
            "mul" => x[0].mul_round(&x[1], p, round),
            "sqr" => x[0].mul_round(&x[0], p, round),
            "div" => x[0].div_round(&x[1], p, round),
            "fma" => x[0].mul_add_round(&x[1], &x[2], p, round),
            _ => panic!("op {op}"),
        };
        same(&new, &want, line);
        let mut r = match i % 4 {
            0 => BigFloat::zero(p),
            1 => BigFloat::nan(p),
            _ => {
                BigFloat::from(-1)
                    .div_round(&BigFloat::from(3), p, Round::Nearest)
                    .0
            }
        };
        let o = match op {
            "add" => r.set_add(&x[0], &x[1], round),
            "sub" => r.set_sub(&x[0], &x[1], round),
            "mul" => r.set_mul(&x[0], &x[1], round),
            "sqr" => r.set_mul(&x[0], &x[0], round),
            "div" => r.set_div(&x[0], &x[1], round),
            _ => r.set_mul_add(&x[0], &x[1], &x[2], round),
        };
        same(&(r, o), &want, &format!("into: {line}"));
        if x[0].prec() == p && op != "fma" && op != "sqr" {
            let mut r = x[0].clone();
            let o = match op {
                "add" => r.add_assign_round(&x[1], round),
                "sub" => r.sub_assign_round(&x[1], round),
                "mul" => r.mul_assign_round(&x[1], round),
                _ => r.div_assign_round(&x[1], round),
            };
            same(&(r, o), &want, &format!("assign: {line}"));
        }
        count += 1;
    }
    assert!(count > size(1000, 50));
}

/// The forms into an existing number against the exact results rounded by integers: results of
/// the precisions where the kernels change, operands of the result's precision, of at most it in
/// as many limbs and of other lengths, exponents apart by the amounts where the kernels change,
/// terms that cancel, and fused multiply-adds of operands of one, two and four limbs and addends
/// up to eight; each into a number that holds a finite value of its own.
#[test]
fn forms_into_numbers_round_correctly_at_every_kernel() {
    let mut s = 0x1A70_F0E5_0000_0003u64;
    for i in 0..size(5000, 30) {
        let p = prec(&mut s);
        let pick = |s: &mut u64| -> u64 {
            match below(s, 6) {
                0..=2 => p,
                3 => 1 + below(s, p.div_ceil(64) * 64),
                4 => 1 + below(s, 256),
                _ => prec(s),
            }
        };
        let (pa, pb, pc) = (pick(&mut s), pick(&mut s), pick(&mut s));
        let ea = below(&mut s, 200) as i64 - 100;
        let a = of(&mut s, pa, ea);
        let b = if i % 5 == 0 {
            // A difference that cancels: `-a` at its own precision, nudged.
            let (pt, k) = (1 + below(&mut s, 70), 1 + below(&mut s, 2 * pb) as i64);
            let tiny = of(&mut s, pt, ea - k);
            (-&a)
                .prec_round(pb, mode(&mut s))
                .0
                .add_round(&tiny, pb, Round::Nearest)
                .0
        } else {
            let k = shift(&mut s, p.max(pa));
            of(&mut s, pb, ea - k)
        };
        let ec = ea + b.exponent().unwrap_or(0) - shift(&mut s, p) + 1;
        let c = of(&mut s, pc, ec);
        let (ra, rb, rc) = (rational(&a), rational(&b), rational(&c));
        for round in MODES {
            let down = round == Round::Down;
            let start = || {
                BigFloat::from(1)
                    .div_round(&BigFloat::from(3), p, Round::Nearest)
                    .0
            };
            let mut r = start();
            let o = r.set_add(&a, &b, round);
            check(
                &(r, o),
                &(&ra + &rb),
                down,
                p,
                round,
                &format!("set {a:?} + {b:?} at {p}"),
            );
            let mut r = start();
            let o = r.set_sub(&a, &b, round);
            check(
                &(r, o),
                &(&ra - &rb),
                down,
                p,
                round,
                &format!("set {a:?} - {b:?} at {p}"),
            );
            let mut r = start();
            let o = r.set_mul(&a, &b, round);
            check(
                &(r, o),
                &(&ra * &rb),
                false,
                p,
                round,
                &format!("set {a:?} * {b:?} at {p}"),
            );
            let mut r = start();
            let o = r.set_div(&a, &b, round);
            check(
                &(r, o),
                &(&ra / &rb),
                false,
                p,
                round,
                &format!("set {a:?} / {b:?} at {p}"),
            );
            if round == Round::Nearest && i % 4 == 0 {
                assigning(&a, &b, p, mode(&mut s));
            }
            let exact = &(&ra * &rb) + &rc;
            let what = format!("set {a:?} * {b:?} + {c:?} at {p}");
            let mut r = start();
            let o = r.set_mul_add(&a, &b, &c, round);
            check(&(r, o), &exact, down, p, round, &what);
            check(
                &a.mul_add_round(&b, &c, p, round),
                &exact,
                down,
                p,
                round,
                &what,
            );
        }
    }
}

/// Results past the ends of the exponent range by the forms into an existing number, whose
/// kernels leave them to the general path: the same as the forms with a new result, which round
/// them to an infinity or the largest number, or to zero or the smallest, by the mode.
#[test]
fn forms_into_numbers_overflow_and_underflow_as_the_others() {
    let mut s = 0x0E0F_10E5_0000_0005u64;
    for _ in 0..size(1500, 20) {
        let p = prec(&mut s);
        let (pa, pb) = (
            if below(&mut s, 2) == 0 {
                p
            } else {
                prec(&mut s)
            },
            p,
        );
        let top = BigFloat::EXP_MAX - below(&mut s, 3) as i64;
        let bottom = BigFloat::EXP_MIN + below(&mut s, 3) as i64;
        let k = below(&mut s, 3) as i64;
        let (ea, eb) = match below(&mut s, 4) {
            0 => (top, top - k),
            1 => (top, 1 + k),
            2 => (bottom, bottom + k.min(1)),
            _ => (bottom, 1 + k),
        };
        let (a, b) = (of(&mut s, pa, ea), of(&mut s, pb, eb));
        let ec = if below(&mut s, 2) == 0 { top } else { bottom };
        let c = of(&mut s, pb, ec);
        let round = mode(&mut s);
        let start = || BigFloat::from(5).prec_round(p, Round::Nearest).0;
        let pairs: [(&str, Op); 4] = [
            ("+", |x, y, p, r| x.add_round(y, p, r)),
            ("-", |x, y, p, r| x.sub_round(y, p, r)),
            ("*", |x, y, p, r| x.mul_round(y, p, r)),
            ("/", |x, y, p, r| x.div_round(y, p, r)),
        ];
        for (k, (name, f)) in pairs.iter().enumerate() {
            let want = f(&a, &b, p, round);
            let mut r = start();
            let o = match k {
                0 => r.set_add(&a, &b, round),
                1 => r.set_sub(&a, &b, round),
                2 => r.set_mul(&a, &b, round),
                _ => r.set_div(&a, &b, round),
            };
            same(&(r, o), &want, &format!("set {a:?} {name} {b:?} at {p}"));
        }
        let want = a.mul_add_round(&b, &c, p, round);
        let mut r = start();
        let o = r.set_mul_add(&a, &b, &c, round);
        same(&(r, o), &want, &format!("set {a:?} * {b:?} + {c:?} at {p}"));
        if let Some(e) = want.0.exponent() {
            assert!((BigFloat::EXP_MIN..=BigFloat::EXP_MAX).contains(&e));
        }
    }
}

/// Differences whose smaller term is all ones and lies just below the word of the kernels (one
/// or two limbs, the four of a fused multiply-add's sum): its bits that reach the word below, all
/// ones but the lowest, leave that word zero after the shift that refills the top bit, and only
/// the fraction below it says that the difference is not exact.
#[test]
fn differences_with_all_ones_just_below_the_word() {
    let mut s = 0xA110_0E5D_0000_0009u64;
    for &w in &[64u64, 128, 256] {
        let ones = (super::common::one() << w as usize) - super::common::one();
        let y = BigFloat::from_parts(&ones, -(w as i64), w, Round::Nearest).0;
        for &p in &[w - 1, w, w / 2 + 1, 1] {
            for d in (w - 2..=w + 3).step_by(size(1, 3)) {
                for k in 0..size(4, 1) {
                    let x = if k == 0 {
                        BigFloat::from_parts(&super::common::one(), 0, p, Round::Nearest).0
                    } else {
                        of(&mut s, p, 1).abs()
                    };
                    let yd = y
                        .mul_round(
                            &BigFloat::from_parts(
                                &super::common::one(),
                                -(d as i64),
                                1,
                                Round::Nearest,
                            )
                            .0,
                            w,
                            Round::Nearest,
                        )
                        .0;
                    let (rx, ry) = (rational(&x), rational(&yd));
                    for round in MODES {
                        let down = round == Round::Down;
                        let what = format!("{x:?} - {yd:?} at {p}");
                        check(
                            &x.sub_round(&yd, p, round),
                            &(&rx - &ry),
                            down,
                            p,
                            round,
                            &what,
                        );
                        let mut r = BigFloat::from(3).prec_round(p, Round::Nearest).0;
                        let o = r.set_sub(&x, &yd, round);
                        check(&(r, o), &(&rx - &ry), down, p, round, &what);
                        let one = BigFloat::from(1).prec_round(p, Round::Nearest).0;
                        let what = format!("{x:?} * 1 - {yd:?} at {p}");
                        check(
                            &x.mul_add_round(&one, &(-&yd), p, round),
                            &(&rx - &ry),
                            down,
                            p,
                            round,
                            &what,
                        );
                    }
                }
            }
        }
    }
}

/// Sums of operands of the result's length that carry, the smaller shifted by about a limb so
/// that the lowest bit shifted out is alone below the halfway point: the shift by one of the
/// carry moves it out of the word below, and only the sticky bit keeps it.
#[test]
fn same_length_sums_whose_last_bit_out_decides() {
    let mut s = 0x5A3E_0B17_0000_000Bu64;
    for n in 3..=5u64 {
        for p in [64 * n, 64 * n - 1, 64 * n - 2, 64 * n - 63] {
            for &d in [63i64, 64, 65, 127, 128, 129].iter().step_by(size(1, 2)) {
                for &low in [1u64, (1 << 63) | 1, 3, 2].iter().step_by(size(1, 3)) {
                    // `x` all ones at the top, so that the sum carries; `y` with `low` as its
                    // lowest limb.
                    let ones = (super::common::one() << (64 * n) as usize) - super::common::one();
                    let x =
                        BigFloat::from_parts(&ones, -((64 * n) as i64), 64 * n, Round::Nearest).0;
                    let x = x.prec_round(p, Round::Down).0;
                    let mut y = super::common::one() << (64 * n - 1) as usize;
                    for k in 1..n - 1 {
                        y += Number::from(xorshift(&mut s)) << (64 * k) as usize;
                    }
                    y += Number::from(low);
                    let y =
                        BigFloat::from_parts(&y, -((64 * n) as i64) - d, 64 * n, Round::Nearest).0;
                    let y = y.prec_round(p, Round::Down).0;
                    let (rx, ry) = (rational(&x), rational(&y));
                    for round in MODES {
                        let what = format!("{x:?} + {y:?} at {p}");
                        check(
                            &x.add_round(&y, p, round),
                            &(&rx + &ry),
                            false,
                            p,
                            round,
                            &what,
                        );
                        let mut r = BigFloat::from(3).prec_round(p, Round::Nearest).0;
                        let o = r.set_add(&x, &y, round);
                        check(&(r, o), &(&rx + &ry), false, p, round, &what);
                        let what = format!("{x:?} - {:?} at {p}", -&y);
                        check(
                            &x.sub_round(&-&y, p, round),
                            &(&rx + &ry),
                            false,
                            p,
                            round,
                            &what,
                        );
                    }
                }
            }
        }
    }
}

/// Differences of terms one exponent apart that cancel all the limbs of the result, `2^(e-1)` less
/// all ones of the result's length one exponent down: what is left is the bit shifted out of the
/// smaller term, alone, the word below the result's limbs, which becomes the whole mantissa.
#[test]
fn differences_that_cancel_every_limb() {
    for n in 3..=6u64 {
        let w = 64 * n;
        let ones = (super::common::one() << w as usize) - super::common::one();
        for p in [w, w - 1, w - 63] {
            for e in [0i64, 7, -300] {
                let x = BigFloat::from_parts(&super::common::one(), e - 1, p, Round::Nearest).0;
                let y = BigFloat::from_parts(&ones, e - 1 - w as i64, w, Round::Nearest).0;
                let y = y.prec_round(p, Round::Down).0;
                let (rx, ry) = (rational(&x), rational(&y));
                for round in MODES {
                    let what = format!("{x:?} - {y:?} at {p}");
                    let down = round == Round::Down;
                    check(
                        &x.sub_round(&y, p, round),
                        &(&rx - &ry),
                        down,
                        p,
                        round,
                        &what,
                    );
                    let mut r = BigFloat::from(1)
                        .div_round(&BigFloat::from(3), p, Round::Nearest)
                        .0;
                    let o = r.set_sub(&x, &y, round);
                    check(&(r, o), &(&rx - &ry), down, p, round, &what);
                }
            }
        }
    }
}
