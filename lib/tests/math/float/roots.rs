//! The short paths of the square root, the reciprocal square root, the roots of small degree and
//! the integer powers, at the precisions and the operands where they change method: every
//! precision of one and two limbs for the square root and the roots, the boundaries of the
//! fixed-width powers, mantissas of long runs (whose powers and roots sit next to the boundaries
//! of a rounding), and the exact and halfway results; against the bracket of a correct rounding,
//! the exact references, and MPFR's results on a file of vectors.

use super::common::{MODES, below, mantissa, reference, same, size, xorshift};
use super::power::{bracket, pow};
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// A mantissa of `bits` bits of one of the shapes that stress the short paths: runs (as the
/// common generator makes them), all ones, a power of two with a few bits at the bottom, or all
/// ones but a few at the bottom.
fn shaped(s: &mut u64, bits: u64) -> Number {
    match below(s, 4) {
        0 => mantissa(s, bits),
        1 => (Number::new_from_u8(1, ByteOrder::LittleEndian) << bits as usize) - Number::from(1),
        2 => {
            let top = Number::new_from_u8(1, ByteOrder::LittleEndian) << (bits - 1) as usize;
            let low = below(s, 1 << (bits - 1).min(6));
            top + Number::from(low)
        }
        _ => {
            let all = (Number::new_from_u8(1, ByteOrder::LittleEndian) << bits as usize)
                - Number::from(1);
            let low = below(s, 1 << (bits - 1).min(6));
            all - Number::from(low)
        }
    }
}

/// A float of `bits` bits of a shaped mantissa and an exponent of either parity near zero (or far
/// from it, one time in eight), negative when `negative`.
fn operand(s: &mut u64, bits: u64, negative: bool) -> BigFloat {
    let m = shaped(s, bits);
    let m = if negative { -m } else { m };
    let e = if below(s, 8) == 0 {
        below(s, 2001) as i64 - 1000
    } else {
        below(s, 9) as i64 - 4
    };
    let (x, o) = BigFloat::from_parts(&m, e - bits as i64, bits, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

/// The precision of an operand: one limb, two, three to four, or longer.
fn operand_bits(s: &mut u64) -> u64 {
    match below(s, 4) {
        0 => 1 + below(s, 64),
        1 => 65 + below(s, 64),
        2 => 129 + below(s, 128),
        _ => 257 + below(s, 800),
    }
}

/// Every precision from 1 to 130, operands of one limb to many: against the bracket of `r^2` and
/// `x`, into an existing number too.
#[test]
fn square_roots_in_registers_round_correctly() {
    let mut s = 0x9051_0001u64;
    for i in 0..size(6000, 60) {
        let p = 1 + (i as u64 % 130);
        let bits = operand_bits(&mut s);
        let x = operand(&mut s, bits, false);
        let rx = Rational::try_from(&x).unwrap();
        let round = MODES[below(&mut s, 4) as usize];
        let got = x.sqrt_round(p, round);
        bracket(&got, p, round, &format!("sqrt {x:?} at {p}"), |r| {
            (r * r).cmp(&rx)
        });
        let mut into = BigFloat::from(3).prec_round(p, Round::Nearest).0;
        let o = into.set_sqrt(&x, round);
        same(&(into, o), &got, "set_sqrt");
    }
}

/// Squares and the squares of midpoints: `y^2` for `y` of `p` bits is exact at `p` bits, and for
/// `y` of `p + 1` bits ending in a one, halfway between two numbers of `p` bits (which a mantissa
/// of more than two limbs can hold).
#[test]
fn exact_and_halfway_square_roots_are_found() {
    let mut s = 0x9051_0002u64;
    for i in 0..size(4000, 40) {
        let p = 1 + (i as u64 % 128);
        let by = if i % 2 == 0 { p } else { p + 1 };
        let mut y = mantissa(&mut s, by);
        y.set_bit(0, true);
        let e = below(&mut s, 41) as i64 - 20;
        let y = BigFloat::from_parts(&y, e - by as i64, by, Round::Nearest).0;
        let x = y.mul_round(&y, 2 * by, Round::Nearest);
        assert_eq!(x.1, Ordering::Equal);
        let ry = Rational::try_from(&y).unwrap();
        for round in MODES {
            same(
                &x.0.sqrt_round(p, round),
                &reference(&ry, p, round),
                &format!("sqrt {:?} at {p}", x.0),
            );
        }
    }
}

/// The reciprocal square root at every precision to 200 and at those of the division and Newton's
/// paths, operands of one limb to many.
#[test]
fn reciprocal_square_roots_on_every_path_round_correctly() {
    let mut s = 0x9051_0003u64;
    for i in 0..size(3000, 30) {
        let p = match i % 5 {
            0 => 200 + below(&mut s, 1000),
            1 if i % 3 == 0 => 2000 + below(&mut s, 3000),
            _ => 1 + (i as u64 % 200),
        };
        let bits = operand_bits(&mut s);
        let x = operand(&mut s, bits, false);
        let rx = Rational::try_from(&x).unwrap();
        let round = MODES[below(&mut s, 4) as usize];
        let got = x.rec_sqrt_round(p, round);
        bracket(&got, p, round, &format!("1/sqrt {x:?} at {p}"), |r| {
            (&(r * r) * &rx).cmp(&Rational::from(1))
        });
    }
}

/// The roots of degree 3 to 16 at every precision to 130, of operands of one limb to many and of
/// both signs for an odd degree.
#[test]
fn roots_of_small_degree_round_correctly() {
    let mut s = 0x9051_0004u64;
    for i in 0..size(4000, 30) {
        let p = 1 + (i as u64 % 130);
        let n = 3 + below(&mut s, 14) as u32;
        let negative = n % 2 == 1 && xorshift(&mut s) & 1 == 1;
        let bits = operand_bits(&mut s);
        let x = operand(&mut s, bits, negative);
        let rx = Rational::try_from(&x).unwrap();
        let round = MODES[below(&mut s, 4) as usize];
        let got = x.nth_root_round(n, p, round);
        bracket(&got, p, round, &format!("{x:?}^(1/{n}) at {p}"), |r| {
            pow(r, u64::from(n)).cmp(&rx)
        });
    }
}

/// `y^n` for `y` of `p` bits (exact roots) and of `p + 1` bits ending in a one (halfway ones), for
/// degrees 3 to 16 and precisions to 128.
#[test]
fn exact_and_halfway_roots_of_small_degree_are_found() {
    let mut s = 0x9051_0005u64;
    for i in 0..size(2000, 20) {
        let n = 3 + below(&mut s, 14) as u32;
        let p = 1 + below(&mut s, 128);
        let by = if i % 2 == 0 { p } else { p + 1 };
        let mut y = mantissa(&mut s, by);
        y.set_bit(0, true);
        let e = below(&mut s, 41) as i64 - 20;
        let y = BigFloat::from_parts(&y, e - by as i64, by, Round::Nearest).0;
        let y = if n % 2 == 1 && xorshift(&mut s) & 1 == 1 {
            -y
        } else {
            y
        };
        let x = y.powi_round(i64::from(n), by * u64::from(n), Round::Nearest);
        assert_eq!(x.1, Ordering::Equal);
        let ry = Rational::try_from(&y).unwrap();
        for round in MODES {
            same(
                &x.0.nth_root_round(n, p, round),
                &reference(&ry, p, round),
                &format!("{:?}^(1/{n}) at {p}", x.0),
            );
        }
    }
}

/// Integer powers beyond 64 bits on the fixed-width limbs and the general ones: exponents of both
/// signs to 40, operands of one limb to many, near 1 (all ones) and near powers of two.
#[test]
fn integer_powers_on_fixed_limbs_round_correctly() {
    let mut s = 0x9051_0006u64;
    for i in 0..size(2500, 20) {
        let p = match i % 4 {
            0 => 65 + below(&mut s, 64),
            1 => 129 + below(&mut s, 128),
            2 => 257 + below(&mut s, 256),
            _ => 513 + below(&mut s, 600),
        };
        let n = 2 + below(&mut s, 39) as i64;
        let n = if xorshift(&mut s) & 1 == 1 { -n } else { n };
        let bits = if i % 3 == 0 {
            1 + below(&mut s, 64)
        } else {
            operand_bits(&mut s).min(700)
        };
        let negative = xorshift(&mut s) & 1 == 1;
        let x = operand(&mut s, bits, negative);
        let rx = Rational::try_from(&x).unwrap();
        let v = pow(&rx, n.unsigned_abs());
        let v = if n < 0 { Rational::from(1) / v } else { v };
        let round = MODES[below(&mut s, 4) as usize];
        let got = x.powi_round(n, p, round);
        if got.1 == Ordering::Equal {
            assert_eq!(Rational::try_from(&got.0).unwrap(), v, "{x:?}^{n} at {p}");
            continue;
        }
        same(&got, &reference(&v, p, round), &format!("{x:?}^{n} at {p}"));
    }
}

/// The powers of all-ones mantissas of whole limbs: their binary expansions are runs of zeros and
/// ones as long as the limbs, which leave a cut power next to a boundary at every width.
#[test]
fn powers_of_long_runs_end() {
    for limbs in 1..=4u64 {
        let m = (Number::new_from_u8(1, ByteOrder::LittleEndian) << (64 * limbs) as usize)
            - Number::from(1);
        let x = BigFloat::from_parts(&m, -(64 * limbs as i64), 64 * limbs, Round::Nearest).0;
        let rx = Rational::try_from(&x).unwrap();
        for n in [2i64, 3, 7, -3, 31] {
            let v = pow(&rx, n.unsigned_abs());
            let v = if n < 0 { Rational::from(1) / v } else { v };
            for p in [24, 54, 64, 100, 128, 200, 300] {
                for round in MODES {
                    same(
                        &x.powi_round(n, p, round),
                        &reference(&v, p, round),
                        &format!("{x:?}^{n} at {p}"),
                    );
                }
            }
        }
    }
}

/// A float of the vectors' text: `prec:value`, the value `nan`, `inf`, `-inf`, `0`, `-0` or
/// `<hex mantissa>p<exponent>`.
fn parse(field: &str) -> BigFloat {
    let (p, v) = field.split_once(':').expect("prec:value");
    let p: u64 = p.parse().expect("a precision");
    match v {
        "nan" => BigFloat::nan(p),
        "inf" => BigFloat::infinity(p),
        "-inf" => BigFloat::neg_infinity(p),
        "0" => BigFloat::zero(p),
        "-0" => BigFloat::neg_zero(p),
        _ => {
            let (m, e) = v.split_once('p').expect("mantissa p exponent");
            let neg = m.starts_with('-');
            let m = Number::from_str_radix(m.trim_start_matches('-'), 16, ByteOrder::LittleEndian)
                .expect("hexadecimal");
            let m = if neg { -m } else { m };
            let (x, o) = BigFloat::from_parts(&m, e.parse().expect("exponent"), p, Round::Nearest);
            assert_eq!(o, Ordering::Equal);
            x
        }
    }
}

/// MPFR 4.2's results at the precisions of the short paths: `op prec mode x [k] = result ternary`.
#[test]
fn mpfr_vectors_of_the_short_paths_agree() {
    let data = include_str!("data/roots_vectors.txt");
    let step = if cfg!(miri) { 61 } else { 1 };
    let mut count = 0;
    for line in data
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .step_by(step)
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let eq = f.iter().position(|&t| t == "=").expect("=");
        let p: u64 = f[1].parse().unwrap();
        let round = match f[2] {
            "N" => Round::Nearest,
            "Z" => Round::TowardZero,
            "U" => Round::Up,
            _ => Round::Down,
        };
        let x = parse(f[3]);
        let want = (parse(f[eq + 1]), f[eq + 2].parse::<i8>().unwrap().cmp(&0));
        let got = match f[0] {
            "sqrt" => x.sqrt_round(p, round),
            "rec_sqrt" => x.rec_sqrt_round(p, round),
            "cbrt" => x.cbrt_round(p, round),
            "root5" => x.nth_root_round(5, p, round),
            "pow7" => x.powi_round(7, p, round),
            "powm3" => x.powi_round(-3, p, round),
            "pow1e9" => x.powi_round(1_000_000_007, p, round),
            "powi" => x.powi_round(f[4].parse().unwrap(), p, round),
            "rootn" => x.nth_root_round(f[4].parse().unwrap(), p, round),
            op => panic!("op {op}"),
        };
        same(&got, &want, line);
        count += 1;
    }
    assert!(count > 0);
}

/// A precision of every path: one and two limbs, the division and the fixed-width paths, Newton's.
fn path_prec(s: &mut u64, i: usize) -> u64 {
    match i % 6 {
        0 | 1 => 1 + below(s, 64),
        2 => 65 + below(s, 64),
        3 => 129 + below(s, 400),
        4 => 529 + below(s, 600),
        _ => 2000 + below(s, 1200),
    }
}

/// `y0` of `p` bits, or of `p + 1` bits ending in a one (a midpoint at `p` bits), positive.
fn boundary(s: &mut u64, p: u64) -> BigFloat {
    let b = if xorshift(s) & 1 == 1 { p } else { p + 1 };
    let mut m = mantissa(s, b);
    m.set_bit(0, true);
    let e = below(s, 21) as i64 - 10;
    BigFloat::from_parts(&m, e - b as i64, b, Round::Nearest).0
}

/// `v` rounded to `w` bits and moved by one unit of its last place up or down, so that a function
/// whose value at `v` is a boundary has its value next to it, and not on it.
fn next_to(v: &BigFloat, w: u64, s: &mut u64) -> BigFloat {
    let v = v.prec_round(w, Round::Nearest).0;
    let e = v.exponent().unwrap();
    let u = BigFloat::from_parts(&Number::from(1), e - w as i64, 1, Round::Nearest).0;
    if xorshift(s) & 1 == 1 {
        v.add_round(&u, w, Round::Nearest).0
    } else {
        v.sub_round(&u, w, Round::Nearest).0
    }
}

/// The square root, the reciprocal square root, the roots of small degree and the integer powers
/// of operands whose results lie within a unit of `2^(-2p - 40)` or so of a number of the
/// precision or of a midpoint: the inverse function of such a number rounded up or down to many
/// bits, so that the windows of the short paths' bounds hold the boundary.
#[test]
fn results_next_to_a_boundary_round_correctly() {
    let mut s = 0x9051_0007u64;
    for i in 0..size(1200, 12) {
        let p = path_prec(&mut s, i);
        let y = boundary(&mut s, p);
        let w = 2 * p + 40 + below(&mut s, 64);
        let round = MODES[below(&mut s, 4) as usize];
        // The square root: `x ≈ y^2`.
        let x = next_to(&y.mul_round(&y, 2 * p + 4, Round::Nearest).0, w, &mut s);
        let rx = Rational::try_from(&x).unwrap();
        let got = x.sqrt_round(p, round);
        bracket(&got, p, round, &format!("sqrt {x:?} at {p}"), |r| {
            (r * r).cmp(&rx)
        });
        // The reciprocal square root: `x ≈ 1/y^2`.
        let inv = BigFloat::from(1)
            .div_round(
                &y.mul_round(&y, w + 8, Round::Nearest).0,
                w + 8,
                Round::Nearest,
            )
            .0;
        let x = next_to(&inv, w, &mut s);
        let rx = Rational::try_from(&x).unwrap();
        let got = x.rec_sqrt_round(p, round);
        bracket(&got, p, round, &format!("1/sqrt {x:?} at {p}"), |r| {
            (&(r * r) * &rx).cmp(&Rational::from(1))
        });
        if p <= 600 {
            // A root of small degree: `x ≈ y^n`.
            let n = 3 + below(&mut s, 14) as u32;
            let wn = u64::from(n) * p + 40;
            let x = next_to(
                &y.powi_round(i64::from(n), wn + 8, Round::Nearest).0,
                wn,
                &mut s,
            );
            let rx = Rational::try_from(&x).unwrap();
            let got = x.nth_root_round(n, p, round);
            bracket(&got, p, round, &format!("{x:?}^(1/{n}) at {p}"), |r| {
                pow(r, u64::from(n)).cmp(&rx)
            });
            // An integer power: `x ≈ y^(1/n)`, `x^n` next to `y`.
            let k = 2 + below(&mut s, 12) as i64;
            let k = if xorshift(&mut s) & 1 == 1 { -k } else { k };
            let root = y
                .nth_root_round(k.unsigned_abs() as u32, w + 8, Round::Nearest)
                .0;
            let base = if k < 0 {
                BigFloat::from(1).div_round(&root, w + 8, Round::Nearest).0
            } else {
                root
            };
            let x = next_to(&base, w, &mut s);
            let rx = Rational::try_from(&x).unwrap();
            let v = pow(&rx, k.unsigned_abs());
            let v = if k < 0 { Rational::from(1) / v } else { v };
            same(
                &x.powi_round(k, p, round),
                &reference(&v, p, round),
                &format!("{x:?}^{k} at {p}"),
            );
        }
    }
}

/// `1 - 2^-64k` (all ones over whole limbs) at whole limbs of precision: its square root's
/// remainder equals the root, `R = S(S + 1)`, just below the midpoint.
#[test]
fn square_roots_of_whole_limbs_of_ones() {
    for limbs in 1..=3u64 {
        let m = (Number::new_from_u8(1, ByteOrder::LittleEndian) << (64 * limbs) as usize)
            - Number::from(1);
        for e in [-2i64, -1, 0, 1, 2] {
            let x = BigFloat::from_parts(&m, e - 64 * limbs as i64, 64 * limbs, Round::Nearest).0;
            let rx = Rational::try_from(&x).unwrap();
            for p in [64 * limbs - 1, 64 * limbs, 64 * limbs + 1] {
                for round in MODES {
                    let got = x.sqrt_round(p, round);
                    bracket(&got, p, round, &format!("sqrt {x:?} at {p}"), |r| {
                        (r * r).cmp(&rx)
                    });
                }
            }
        }
    }
}

/// The general square root at precisions within 16 bits below a whole number of limbs, where it
/// takes the root of those limbs with its remainder: exact squares, squares of midpoints, and
/// operands with long tails, against the references and the bracket.
#[test]
fn square_roots_just_below_whole_limbs_round_correctly() {
    let mut s = 0x9051_0008u64;
    for i in 0..size(1500, 15) {
        let p = 64 * (2 + below(&mut s, 6)) - below(&mut s, 16);
        let round = MODES[below(&mut s, 4) as usize];
        if i % 2 == 0 {
            let by = if i % 4 == 0 { p } else { p + 1 };
            let mut y = mantissa(&mut s, by);
            y.set_bit(0, true);
            let y = BigFloat::from_parts(
                &y,
                below(&mut s, 9) as i64 - 4 - by as i64,
                by,
                Round::Nearest,
            )
            .0;
            let x = y.mul_round(&y, 2 * by, Round::Nearest).0;
            same(
                &x.sqrt_round(p, round),
                &reference(&Rational::try_from(&y).unwrap(), p, round),
                &format!("sqrt {x:?} at {p}"),
            );
        } else {
            let bits = 2 * p + below(&mut s, 300);
            let x = operand(&mut s, bits, false);
            let rx = Rational::try_from(&x).unwrap();
            let got = x.sqrt_round(p, round);
            bracket(&got, p, round, &format!("sqrt {x:?} at {p}"), |r| {
                (r * r).cmp(&rx)
            });
        }
    }
}
