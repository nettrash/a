//! `BigComplex` against GNU MPC 1.4.1's results (`data/complex_vectors.txt`, made by the
//! generator in the complex area's work directory, every mode pair, precisions from 1 to about
//! 1100 bits, exact and halfway cases built in) and its special values
//! (`data/complex_specials.txt`, every number whose parts are NaN, ±∞, ±0, 1 or -2), and the
//! parts of products, quotients, norms and absolute values against the exact rationals rounded
//! by integers, at random precisions up to 10^4 bits.

use super::common::{MODES, below, float, mode, one, pow2, rational, reference, same, size};
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// A value of the vectors at the precision `prec`.
fn value(s: &str, prec: u64) -> BigFloat {
    match s {
        "nan" => BigFloat::nan(prec),
        "inf" => BigFloat::infinity(prec),
        "-inf" => BigFloat::neg_infinity(prec),
        "0" => BigFloat::zero(prec),
        "-0" => BigFloat::neg_zero(prec),
        _ => {
            let (negative, s) = match s.strip_prefix('-') {
                Some(s) => (true, s),
                None => (false, s),
            };
            let s = s.strip_prefix("0x").expect("a hexadecimal mantissa");
            let (m, e) = s.split_once('p').expect("an exponent");
            let mut m = Number::from_str_radix(m, 16, ByteOrder::LittleEndian).expect("hex");
            if negative {
                m = -m;
            }
            let (x, o) =
                BigFloat::from_parts(&m, e.parse().expect("exponent"), prec, Round::Nearest);
            assert_eq!(o, Ordering::Equal, "{s} is not exact at {prec} bits");
            x
        }
    }
}

/// An input of the vectors, `prec:value`.
fn input(s: &str) -> BigFloat {
    let (p, v) = s.split_once(':').expect("prec:value");
    value(v, p.parse().expect("precision"))
}

fn complex_input(re: &str, im: &str) -> BigComplex {
    BigComplex::new(input(re), input(im))
}

fn mode_of(c: u8) -> Round {
    match c {
        b'N' => Round::Nearest,
        b'Z' => Round::TowardZero,
        b'U' => Round::Up,
        b'D' => Round::Down,
        _ => panic!("a mode"),
    }
}

fn ternary(s: &str) -> Ordering {
    match s {
        "-1" => Ordering::Less,
        "0" => Ordering::Equal,
        "1" => Ordering::Greater,
        _ => panic!("a ternary"),
    }
}

/// Whether `got` is `want` as [`same`] has it.
fn agrees(got: &(BigFloat, Ordering), want: &(BigFloat, Ordering)) -> bool {
    got.0.prec() == want.0.prec()
        && got.1 == want.1
        && (got.0.is_nan() && want.0.is_nan()
            || got.0 == want.0 && got.0.is_sign_negative() == want.0.is_sign_negative())
}

/// The cases where this crate is IEEE 754's and MPC 1.4.1 is not: an exact zero part, rounded
/// down, whose sign MPC takes as `-0` for zeros of one sign `+0` (its `mul_real` and `mul_imag`),
/// or as `+0` for a cancellation or zeros of both signs (its `mpc_div`, which forms the
/// numerators toward zero, and its Karatsuba product). Both are the exact zero; only the sign
/// differs, and the crate's is the sum of IEEE 754 in the mode.
fn zero_sign_only(got: &(BigFloat, Ordering), want: &(BigFloat, Ordering), round: Round) -> bool {
    round == Round::Down
        && got.0.is_zero()
        && want.0.is_zero()
        && got.1 == Ordering::Equal
        && want.1 == Ordering::Equal
        && got.0.prec() == want.0.prec()
}

#[test]
fn every_vector_of_mpc_is_matched() {
    let data = include_str!("data/complex_vectors.txt");
    let mut zero_signs = 0;
    let mut checked = 0;
    let step = if cfg!(miri) { 37 } else { 1 };
    for (n, line) in data.lines().enumerate() {
        if n % step != 0 {
            continue;
        }
        let mut sections = line.split(" | ");
        let head: Vec<&str> = sections.next().unwrap().split(' ').collect();
        let args: Vec<&str> = sections.next().unwrap().split(' ').collect();
        let res: Vec<&str> = sections.next().unwrap().split(' ').collect();
        let (op, modes) = (head[0], head[1].as_bytes());
        let (pr, pi): (u64, u64) = (head[2].parse().unwrap(), head[3].parse().unwrap());
        let what = format!("line {}: {line}", n + 1);
        if op == "norm" || op == "abs" {
            let x = complex_input(args[0], args[1]);
            let round = mode_of(modes[0]);
            let got = if op == "norm" {
                x.norm_round(pr, round)
            } else {
                x.abs_round(pr, round)
            };
            same(&got, &(value(res[0], pr), ternary(res[1])), &what);
            checked += 1;
            continue;
        }
        let x = complex_input(args[0], args[1]);
        let prec = (pr, pi);
        let round = (mode_of(modes[0]), mode_of(modes[1]));
        let (z, o) = match op {
            "add" => x.add_round_parts(&complex_input(args[2], args[3]), prec, round),
            "sub" => x.sub_round_parts(&complex_input(args[2], args[3]), prec, round),
            "mul" => x.mul_round_parts(&complex_input(args[2], args[3]), prec, round),
            "div" => x.div_round_parts(&complex_input(args[2], args[3]), prec, round),
            "sqr" => x.sqr_round_parts(prec, round),
            "sqrt" => x.sqrt_round_parts(prec, round),
            "mul_fr" => x.mul_real_round_parts(&input(args[2]), prec, round),
            "div_fr" => x.div_real_round_parts(&input(args[2]), prec, round),
            "add_fr" => x.add_real_round_parts(&input(args[2]), prec, round),
            "sub_fr" => x.sub_real_round_parts(&input(args[2]), prec, round),
            "mul_si" => x.mul_int_round_parts(args[2].parse().unwrap(), prec, round),
            "mul_i" => x.mul_i_round_parts(args[2] == "-1", prec, round),
            _ => panic!("{what}: an unknown operation"),
        };
        let (re, im) = z.into_parts();
        let parts = [
            ((re, o.0), (value(res[0], pr), ternary(res[2])), round.0),
            ((im, o.1), (value(res[1], pi), ternary(res[3])), round.1),
        ];
        for (got, want, r) in &parts {
            if !agrees(got, want) && zero_sign_only(got, want, *r) {
                zero_signs += 1;
                continue;
            }
            same(got, want, &what);
        }
        checked += 1;
    }
    assert!(checked > 0);
    // The zero signs of MPC's that the crate does not share: few, and all rounding down.
    assert!(zero_signs <= 4, "{zero_signs} zeros of other signs");
}

/// A number of the special table: `n` NaN, `I` +∞, `J` -∞, `0` +0, `z` -0, `1` 1, `m` -2.
fn special(c: u8) -> BigFloat {
    match c {
        b'n' => BigFloat::nan(53),
        b'I' => BigFloat::infinity(53),
        b'J' => BigFloat::neg_infinity(53),
        b'0' => BigFloat::zero(53),
        b'z' => BigFloat::neg_zero(53),
        b'1' => BigFloat::from(1.0),
        b'm' => BigFloat::from(-2.0),
        _ => panic!("a special value"),
    }
}

/// The class of a result as the table writes it: `p` and `q` for finite numbers above and below
/// zero.
fn class(x: &BigFloat) -> u8 {
    if x.is_nan() {
        b'n'
    } else if x.is_infinite() {
        if x.is_sign_negative() { b'J' } else { b'I' }
    } else if x.is_zero() {
        if x.is_sign_negative() { b'z' } else { b'0' }
    } else if x.is_sign_negative() {
        b'q'
    } else {
        b'p'
    }
}

#[test]
fn special_values_are_mpcs() {
    let data = include_str!("data/complex_specials.txt");
    let mut differ = Vec::new();
    let (mut zero_signs, mut typos, mut squares, mut roots) = (0, 0, 0, 0);
    let step = if cfg!(miri) { 13 } else { 1 };
    for line in data.lines().step_by(step) {
        let (args, res) = line.split_once(" | ").unwrap();
        let args: Vec<&str> = args.split(' ').collect();
        let op = args[0];
        let x = args[1].as_bytes();
        let x = BigComplex::new(special(x[0]), special(x[1]));
        for r in res.split(' ') {
            let (m, want) = r.split_once(':').unwrap();
            let round = mode_of(m.as_bytes()[0]);
            let got: Vec<u8> = match op {
                "norm" => vec![class(&x.norm_round(53, round).0)],
                "abs" => vec![class(&x.abs_round(53, round).0)],
                _ => {
                    let z = match op {
                        "sqr" => x.sqr_round(53, round).0,
                        "sqrt" => x.sqrt_round(53, round).0,
                        _ => {
                            let y = args[2].as_bytes();
                            let y = BigComplex::new(special(y[0]), special(y[1]));
                            if op == "mul" {
                                x.mul_round(&y, 53, round).0
                            } else {
                                x.div_round(&y, 53, round).0
                            }
                        }
                    };
                    vec![class(z.re()), class(z.im())]
                }
            };
            if got == want.as_bytes() {
                continue;
            }
            let want = want.as_bytes();
            let zero = |c: u8| c == b'0' || c == b'z';
            if m == "DD"
                && got
                    .iter()
                    .zip(want)
                    .all(|(&g, &w)| g == w || zero(g) && zero(w))
            {
                // An exact zero rounded down: IEEE 754's sign (see `zero_sign_only`).
                zero_signs += 1;
            } else if op == "mul" && naive(&x, args[2]) == got {
                // MPC's `mul_infinite` reads `Im(x)` for `Re(y)` in one test of the imaginary
                // part, and makes `(∞ + i)·(∞ - ∞i)` `∞ - ∞i`; the C standard's `_Cmultd` has
                // `∞·(-∞) + 1·∞`, NaN, as the crate.
                typos += 1;
            } else if op == "sqr" && x.is_infinite() && x.is_nan_part() {
                // MPC's `mpc_sqr` makes `NaN ± ∞i` and `±∞ + NaN·i` NaN in both parts, its
                // `mpc_mul` of the number by itself an infinity, as Annex G has every product
                // of infinities; the crate's square is that product.
                let y = x.mul_round(&x, 53, round).0;
                assert_eq!(got, [class(y.re()), class(y.im())], "{line}");
                squares += 1;
            } else if op == "sqrt" && args[1] == "Jn" && got == b"nI" {
                // `√(-∞ + NaN·i) = NaN ± ∞i`, the sign of the NaN's: MPFR's NaN here has its
                // sign bit set, the crate's NaN has none.
                roots += 1;
            } else {
                differ.push(format!(
                    "{line} [{m}: ours {}]",
                    String::from_utf8(got).unwrap()
                ));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
    if step == 1 {
        assert_eq!((typos, squares, roots), (32, 8, 2));
        assert_eq!(zero_signs, 164);
    }
}

/// The classes of the C standard's naive product `(ac - bd) + (ad + bc)i` of the specials
/// `x` and `y` by IEEE 754's operations, when not both NaN.
fn naive(x: &BigComplex, y: &str) -> Vec<u8> {
    let y = y.as_bytes();
    let (a, b, c, d) = (x.re(), x.im(), &special(y[0]), &special(y[1]));
    let re = &(a * c) - &(b * d);
    let im = &(a * d) + &(b * c);
    vec![class(&re), class(&im)]
}

trait NanPart {
    fn is_nan_part(&self) -> bool;
}

impl NanPart for BigComplex {
    fn is_nan_part(&self) -> bool {
        self.re().is_nan() || self.im().is_nan()
    }
}

/// A random complex number: parts of random precisions up to `max`, exponents within `exps`,
/// sometimes far apart.
fn random(s: &mut u64, max: u64, exps: i64) -> BigComplex {
    let re = float(s, max, exps);
    let far = below(s, 6) == 0;
    let im = float(s, max, if far { 40 * exps } else { exps });
    if below(s, 2) == 0 {
        BigComplex::new(re, im)
    } else {
        BigComplex::new(im, re)
    }
}

/// Checks a part against the exact rational `q` rounded, or, for `q = 0`, a zero from terms
/// that cancel: `+0`, or `-0` rounding down.
#[track_caller]
fn part(got: (BigFloat, Ordering), q: &Rational, prec: u64, round: Round, what: &str) {
    if q.is_zero() {
        let want = if round == Round::Down {
            BigFloat::neg_zero(prec)
        } else {
            BigFloat::zero(prec)
        };
        same(&got, &(want, Ordering::Equal), what);
    } else {
        same(&got, &reference(q, prec, round), what);
    }
}

/// The square root of the dyadic rational `q > 0` rounded to `prec` bits in the mode `round`,
/// by integers: `S = floor(√(n·4^t))` of at least `prec + 2` bits for `q = n/4^h`; the root is
/// `S/2^(h+t)` when the remainder is zero, and otherwise strictly between `S` and `S + 1` units,
/// where no number of the precision and no halfway point lies, so that `(2S + 1)/2^(h+t+1)`
/// rounds as it does.
pub(super) fn sqrt_reference(q: &Rational, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let mut n = q.numer().abs();
    let j = q.denom().trailing_zeros().unwrap_or(0);
    assert_eq!(q.denom(), &pow2(j), "a dyadic rational");
    let mut h = j / 2;
    if j % 2 == 1 {
        n = &n << 1usize;
        h += 1;
    }
    let t = (prec + 4).saturating_sub(n.bits() / 2) + 1;
    let (root, rem) = (&n << (2 * t) as usize).isqrt_rem();
    let (m, k) = if rem.is_zero() {
        (root, h + t)
    } else {
        (&(&root << 1usize) + &one(), h + t + 1)
    };
    reference(&Rational::new(m, pow2(k)), prec, round)
}

#[test]
fn products_quotients_squares_norms_and_absolute_values_round_part_by_part() {
    let mut s = 0x5EED_C0C0_0000_0001u64;
    for i in 0..size(1500, 12) {
        let max = match i % 100 {
            0 if !cfg!(miri) => 10_000,
            n if n % 20 == 1 => 2500,
            n if n % 5 == 2 => 400,
            _ => 120,
        };
        let x = random(&mut s, max, 60);
        let mut y = random(&mut s, max, 60);
        if i % 9 == 4 {
            // Near cancellation in a part of the product.
            y = BigComplex::new(x.im().clone(), x.re().clone())
                .prec_round(max, mode(&mut s))
                .0;
        }
        let pmax = if max > 1000 { max + 200 } else { 300 };
        let prec = (1 + below(&mut s, pmax), 1 + below(&mut s, pmax));
        let round = (mode(&mut s), mode(&mut s));
        let (a, b) = (rational(x.re()), rational(x.im()));
        let (c, d) = (rational(y.re()), rational(y.im()));
        let what = format!("{x:?} and {y:?} at {prec:?} {round:?}");

        let (p, o) = x.mul_round_parts(&y, prec, round);
        let (re, im) = p.into_parts();
        part(
            (re, o.0),
            &(&a * &c - &b * &d),
            prec.0,
            round.0,
            &format!("re mul {what}"),
        );
        part(
            (im, o.1),
            &(&a * &d + &b * &c),
            prec.1,
            round.1,
            &format!("im mul {what}"),
        );

        let (q, o) = x.div_round_parts(&y, prec, round);
        let den = &c * &c + &d * &d;
        let (re, im) = q.into_parts();
        let (qr, qi) = ((&a * &c + &b * &d) / &den, (&b * &c - &a * &d) / &den);
        part((re, o.0), &qr, prec.0, round.0, &format!("re div {what}"));
        part((im, o.1), &qi, prec.1, round.1, &format!("im div {what}"));

        let (q, o) = x.sqr_round_parts(prec, round);
        let (re, im) = q.into_parts();
        part(
            (re, o.0),
            &(&a * &a - &b * &b),
            prec.0,
            round.0,
            &format!("re sqr {what}"),
        );
        let two = Rational::from(2u8);
        part(
            (im, o.1),
            &(&two * &a * &b),
            prec.1,
            round.1,
            &format!("im sqr {what}"),
        );

        let n = &a * &a + &b * &b;
        same(
            &x.norm_round(prec.0, round.0),
            &reference(&n, prec.0, round.0),
            &format!("norm {what}"),
        );
        same(
            &x.abs_round(prec.1, round.1),
            &sqrt_reference(&n, prec.1, round.1),
            &format!("abs {what}"),
        );
    }
}

#[test]
fn exact_and_halfway_quotients_and_roots_round_part_by_part() {
    // `z = q·w` divided by `w`, and `√(q²)`, rounded to the length of `q`'s parts and one bit
    // shorter (halfway between two numbers to nearest when the last bit is set).
    let mut s = 0x5EED_C0C0_0000_0002u64;
    for i in 0..size(1200, 12) {
        let max = if i % 40 == 0 && !cfg!(miri) { 3000 } else { 90 };
        let q = random(&mut s, max, 20);
        let w = random(&mut s, max, 20);
        let z = q.mul_round(&w, 4 * max + 200_000, Round::Nearest);
        assert_eq!(z.1, (Ordering::Equal, Ordering::Equal));
        let z = z.0;
        let shorter = i % 2 == 1;
        let prec = |x: &BigFloat| {
            if shorter {
                x.prec().max(2) - 1
            } else {
                x.prec()
            }
        };
        let p = (prec(q.re()), prec(q.im()));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("({z:?}) / ({w:?}) at {p:?} {round:?}");
        let (r, o) = z.div_round_parts(&w, p, round);
        let (re, im) = r.into_parts();
        same(
            &(re, o.0),
            &reference(&rational(q.re()), p.0, round.0),
            &what,
        );
        same(
            &(im, o.1),
            &reference(&rational(q.im()), p.1, round.1),
            &what,
        );

        // The principal root of `q²` is `q` or `-q`.
        let sq = q.sqr_round(4 * max + 200_000, Round::Nearest);
        assert_eq!(sq.1, (Ordering::Equal, Ordering::Equal));
        let principal = if q.re().is_sign_positive() {
            q.clone()
        } else {
            -&q
        };
        let what = format!("sqrt({:?}) at {p:?} {round:?}", sq.0);
        let (r, o) = sq.0.sqrt_round_parts(p, round);
        let (re, im) = r.into_parts();
        same(
            &(re, o.0),
            &reference(&rational(principal.re()), p.0, round.0),
            &what,
        );
        same(
            &(im, o.1),
            &reference(&rational(principal.im()), p.1, round.1),
            &what,
        );
    }
}

#[test]
fn square_roots_agree_with_more_precise_ones_and_square_back() {
    let mut s = 0x5EED_C0C0_0000_0003u64;
    for i in 0..size(1500, 10) {
        let max = if i % 50 == 0 && !cfg!(miri) {
            2000
        } else {
            150
        };
        let z = random(&mut s, max, 100);
        let prec = (1 + below(&mut s, max + 50), 1 + below(&mut s, max + 50));
        let round = (mode(&mut s), mode(&mut s));
        let (r, o) = z.sqrt_round_parts(prec, round);
        // At 80 bits more to nearest, then rounded: the same unless within 2^-80 of a halfway
        // point or a number of the precision, or exact.
        let (hi, ho) =
            z.sqrt_round_parts((prec.0 + 80, prec.1 + 80), (Round::Nearest, Round::Nearest));
        let what = format!("sqrt({z:?}) at {prec:?} {round:?}");
        assert!(r.re().is_sign_positive(), "{what}");
        assert_eq!(
            r.im().is_sign_negative(),
            z.im().is_sign_negative(),
            "{what}"
        );
        for (got, h, ho, p, rd) in [
            ((r.re().clone(), o.0), hi.re(), ho.0, prec.0, round.0),
            ((r.im().clone(), o.1), hi.im(), ho.1, prec.1, round.1),
        ] {
            if ho == Ordering::Equal {
                same(&got, &reference(&rational(h), p, rd), &what);
                continue;
            }
            // Unless the more precise root is itself a number of the precision or a halfway
            // point, where the exact one may lie on either side.
            let (hp, hpo) = h.prec_round(p, rd);
            if hpo != Ordering::Equal && h.prec_round(p + 1, Round::Nearest).1 != Ordering::Equal {
                same(&got, &(hp, hpo), &what);
            }
        }
        // `r²` near `z`: within a few units of the parts' last places.
        let back = hi.sqr_round(max + 300, Round::Nearest).0;
        let scale = z.abs_round(64, Round::Up).0;
        for (u, v) in [(back.re(), z.re()), (back.im(), z.im())] {
            let diff = rational(&u.sub_round(v, 64, Round::Up).0.abs());
            let bound = rational(&scale) * Rational::new(one(), pow2(prec.0.min(prec.1) + 70));
            assert!(diff <= bound, "{what}: r² is {back:?}");
        }
    }
}

#[test]
fn mixed_operations_are_the_parts_operations() {
    let mut s = 0x5EED_C0C0_0000_0004u64;
    for _ in 0..size(400, 10) {
        let z = random(&mut s, 120, 30);
        let x = float(&mut s, 120, 30);
        let prec = 1 + below(&mut s, 150);
        let round = mode(&mut s);
        let both = |r: (BigComplex, (Ordering, Ordering)),
                    re: (BigFloat, Ordering),
                    im: (BigFloat, Ordering)| {
            let (v, o) = r;
            let (a, b) = v.into_parts();
            same(&(a, o.0), &re, "re");
            same(&(b, o.1), &im, "im");
        };
        both(
            z.add_real_round(&x, prec, round),
            z.re().add_round(&x, prec, round),
            z.im().prec_round(prec, round),
        );
        both(
            z.sub_real_round(&x, prec, round),
            z.re().sub_round(&x, prec, round),
            z.im().prec_round(prec, round),
        );
        both(
            z.mul_real_round(&x, prec, round),
            z.re().mul_round(&x, prec, round),
            z.im().mul_round(&x, prec, round),
        );
        both(
            z.div_real_round(&x, prec, round),
            z.re().div_round(&x, prec, round),
            z.im().div_round(&x, prec, round),
        );
        let w = BigComplex::from(x.clone());
        // A real number as a complex one with `+0` imaginary part: the same values.
        let (p, _) = z.mul_round(&w, prec, round);
        assert_eq!(p, z.mul_real_round(&x, prec, round).0);
        assert_eq!(z.mul_i(false).mul_i(false), -&z);
        assert_eq!(z.mul_i(true).mul_i(false), z);
        assert_eq!(z.conj().mul_i(true), z.mul_i(false).conj());
        // The operators: each part at the larger precision of that part.
        let y = random(&mut s, 120, 30);
        let sum = &z + &y;
        assert_eq!(
            sum.prec(),
            (
                z.re().prec().max(y.re().prec()),
                z.im().prec().max(y.im().prec())
            )
        );
        assert_eq!(
            &z * &y,
            z.mul_round_parts(&y, sum.prec(), (Round::Nearest, Round::Nearest))
                .0
        );
        for r in MODES {
            let (q, _) = z.sqr_round(prec, r);
            assert_eq!(q, z.mul_round(&z, prec, r).0);
        }
    }
}

#[test]
fn long_products_round_part_by_part() {
    // Parts of 1500 bits and more, of exponents near each other: the products of three
    // multiplications, against the exact rationals; cancellations included.
    let mut s = 0x5EED_C0C0_0000_0005u64;
    for i in 0..size(40, 2) {
        let part = |s: &mut u64| {
            let p = if cfg!(miri) {
                1500
            } else {
                1500 + below(s, 2500)
            };
            let x = float(s, 1, 0);
            let m = super::common::mantissa(s, p);
            let m = if x.is_sign_negative() { -m } else { m };
            BigFloat::from_parts(&m, below(s, 20) as i64 - 10 - p as i64, p, Round::Nearest).0
        };
        let x = BigComplex::new(part(&mut s), part(&mut s));
        let y = match i % 4 {
            // `(a + bi)(b + ai)` and `(a + bi)(a - bi)`: real parts that cancel exactly.
            0 => BigComplex::new(x.im().clone(), x.re().clone()),
            1 => x.conj(),
            _ => BigComplex::new(part(&mut s), part(&mut s)),
        };
        let prec = (1 + below(&mut s, 5000), 1 + below(&mut s, 5000));
        let round = (mode(&mut s), mode(&mut s));
        let (a, b) = (rational(x.re()), rational(x.im()));
        let (c, d) = (rational(y.re()), rational(y.im()));
        let what = format!("long product {i} at {prec:?} {round:?}");
        let (p, o) = x.mul_round_parts(&y, prec, round);
        let (re, im) = p.into_parts();
        part_check(re, o.0, &(&a * &c - &b * &d), prec.0, round.0, &what);
        part_check(im, o.1, &(&a * &d + &b * &c), prec.1, round.1, &what);
    }
}

#[track_caller]
fn part_check(x: BigFloat, o: Ordering, q: &Rational, prec: u64, round: Round, what: &str) {
    part((x, o), q, prec, round, what);
}

#[test]
fn quotients_by_parts_far_apart_round_by_ziv_and_exactly() {
    // `z = q·w` over `w` with parts thousands of bits apart, so that `c² + d²` is too long to
    // take exactly and Ziv's strategy goes first: exact quotients (at `q`'s precisions), halfway
    // ones (a bit shorter), and quotients just beside them (`z` moved by one unit of its last
    // place far below), each part in every mode.
    let mut s = 0x5EED_C0C0_0000_0006u64;
    for i in 0..size(300, 4) {
        // A real or imaginary `q`: then `q·w` has parts as short as `q`'s and `w`'s, however far
        // apart `w`'s parts are, and the quotient's sums are long beside them.
        let x = float(&mut s, if cfg!(miri) { 30 } else { 120 }, 10);
        let q = if i % 4 < 2 {
            BigComplex::new(x.clone(), BigFloat::zero(x.prec()))
        } else {
            BigComplex::new(BigFloat::zero(x.prec()), x.clone())
        };
        let gap = 5000 + below(&mut s, 3000) as i64;
        let c = float(&mut s, 60, 5);
        let d = float(&mut s, 60, 5);
        let d = BigFloat::from_parts(
            &d.mantissa_exponent().unwrap().0,
            d.mantissa_exponent().unwrap().1 - gap,
            d.prec(),
            Round::Nearest,
        )
        .0;
        let w = if i % 2 == 0 {
            BigComplex::new(c, d)
        } else {
            BigComplex::new(d, c)
        };
        let mut z = q.mul_round(&w, 20_000, Round::Nearest).0;
        let shorter = i % 3 == 1;
        if i % 3 == 2 {
            // Moved by one unit about 60 bits below the quotient's last place: no longer exact,
            // very near the exact quotient.
            let e = z.re().exponent().unwrap_or(0) - 260;
            let e = BigFloat::from_parts(&one(), e, 1, Round::Nearest).0;
            z = z.add_real_round(&e, 20_000, Round::Nearest).0;
        }
        // The parts at their exact lengths, so that the sums of the quotient are long beside
        // them and Ziv's strategy goes first.
        let short = |x: &BigFloat| match x.mantissa_exponent() {
            Some((m, _)) => {
                let bits = m.bits() - m.trailing_zeros().unwrap_or(0);
                let (y, o) = x.prec_round(bits.max(1), Round::Nearest);
                assert_eq!(o, Ordering::Equal);
                y
            }
            None => x.clone(),
        };
        let z = BigComplex::new(short(z.re()), short(z.im()));
        let p = |x: &BigFloat| {
            if shorter {
                x.prec().max(2) - 1
            } else {
                x.prec()
            }
        };
        let prec = (p(q.re()), p(q.im()));
        let round = (mode(&mut s), mode(&mut s));
        let (a, b) = (rational(z.re()), rational(z.im()));
        let (c, d) = (rational(w.re()), rational(w.im()));
        let den = &c * &c + &d * &d;
        let what = format!("far quotient {i} at {prec:?} {round:?}");
        let (r, o) = z.div_round_parts(&w, prec, round);
        let (re, im) = r.into_parts();
        part(
            (re, o.0),
            &((&a * &c + &b * &d) / &den),
            prec.0,
            round.0,
            &what,
        );
        part(
            (im, o.1),
            &((&b * &c - &a * &d) / &den),
            prec.1,
            round.1,
            &what,
        );
    }
}

#[test]
fn roots_of_squares_with_a_small_part_near_the_working_precision() {
    // `√((x + yi)²)` with `y/x` about `2^-(w/2)`, `w` the first working precision: `y²`
    // is then near the last bit that the cut sums keep, and the exact root must still round.
    let mut s = 0x5EED_C0C0_0000_0007u64;
    for i in 0..size(400, 6) {
        let p = 16 + below(&mut s, 110);
        let guard = 2 * (64 - p.leading_zeros() as u64) + 16;
        let w = (p + guard) as i64;
        let delta = below(&mut s, 20) as i64 - 6;
        let x = float(&mut s, 1, 0).abs();
        let mx = super::common::mantissa(&mut s, p);
        let x = BigFloat::from_parts(&mx, -(p as i64), p, Round::Nearest)
            .0
            .copysign(&x);
        let my = super::common::mantissa(&mut s, p);
        let my = if below(&mut s, 2) == 0 { -my } else { my };
        let y = BigFloat::from_parts(&my, -(p as i64) - (w - delta) / 2 - 1, p, Round::Nearest).0;
        let q = BigComplex::new(x, y);
        let sq = q.sqr_round(100_000, Round::Nearest);
        assert_eq!(sq.1, (Ordering::Equal, Ordering::Equal));
        let shorter = i % 2 == 1;
        let prec = if shorter { (p - 1, p - 1) } else { (p, p) };
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("sqrt({:?}) at {prec:?} {round:?}", sq.0);
        let (r, o) = sq.0.sqrt_round_parts(prec, round);
        let (re, im) = r.into_parts();
        same(
            &(re, o.0),
            &reference(&rational(q.re()), prec.0, round.0),
            &what,
        );
        same(
            &(im, o.1),
            &reference(&rational(q.im()), prec.1, round.1),
            &what,
        );
    }
}

/// A part of a random complex number for the forms into a number: a float, or a zero, or (when
/// `special`) NaN or an infinity, now and then.
fn any_part(s: &mut u64, max: u64, exps: i64, special: bool) -> BigFloat {
    let prec = 1 + below(s, max);
    match below(s, 24) {
        0 => BigFloat::zero(prec),
        1 => BigFloat::neg_zero(prec),
        2 if special => BigFloat::nan(prec),
        3 if special => BigFloat::infinity(prec),
        4 if special => BigFloat::neg_infinity(prec),
        _ => float(s, max, exps),
    }
}

/// A number of the precisions `prec` to write into, holding something already: a finite value
/// (so that its vectors are reused), NaN or a zero.
fn target(s: &mut u64, prec: (u64, u64)) -> BigComplex {
    let fill = |s: &mut u64, p: u64| match below(s, 4) {
        0 => BigFloat::nan(p),
        1 => BigFloat::neg_zero(p),
        _ => float(s, p, 50).prec_round(p, Round::Nearest).0,
    };
    BigComplex::new(fill(s, prec.0), fill(s, prec.1))
}

/// The parts of `z` and their directions.
fn parts_of(z: BigComplex, o: (Ordering, Ordering)) -> [(BigFloat, Ordering); 2] {
    let (re, im) = z.into_parts();
    [(re, o.0), (im, o.1)]
}

/// A form that assigns: its name, the `_round_parts` form's result, and the number and the
/// directions it assigned.
type Assigned = (
    &'static str,
    (BigComplex, (Ordering, Ordering)),
    BigComplex,
    (Ordering, Ordering),
);

#[test]
fn forms_into_a_number_are_the_round_parts_forms() {
    let mut s = 0x5EED_C0C0_0000_0010u64;
    for i in 0..size(3000, 24) {
        let max = match i % 50 {
            0 if !cfg!(miri) => 1200,
            n if n % 5 == 1 => 300,
            n if n % 5 == 2 => 64,
            _ => 130,
        };
        let special = i % 7 == 3;
        let part = |s: &mut u64| any_part(s, max, 40, special);
        let x = BigComplex::new(part(&mut s), part(&mut s));
        let y = BigComplex::new(part(&mut s), part(&mut s));
        let r = part(&mut s);
        let pmax = if max > 300 { max + 100 } else { 300 };
        let prec = (1 + below(&mut s, pmax), 1 + below(&mut s, pmax));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("{x:?}, {y:?}, {r:?} at {prec:?} {round:?}");
        type Op<'a> = (
            &'a str,
            Box<dyn Fn(&mut BigComplex) -> (Ordering, Ordering) + 'a>,
            (BigComplex, (Ordering, Ordering)),
        );
        let ops: Vec<Op<'_>> = vec![
            (
                "add",
                Box::new(|z| z.set_add_parts(&x, &y, round)),
                x.add_round_parts(&y, prec, round),
            ),
            (
                "sub",
                Box::new(|z| z.set_sub_parts(&x, &y, round)),
                x.sub_round_parts(&y, prec, round),
            ),
            (
                "mul",
                Box::new(|z| z.set_mul_parts(&x, &y, round)),
                x.mul_round_parts(&y, prec, round),
            ),
            (
                "sqr",
                Box::new(|z| z.set_sqr_parts(&x, round)),
                x.sqr_round_parts(prec, round),
            ),
            (
                "div",
                Box::new(|z| z.set_div_parts(&x, &y, round)),
                x.div_round_parts(&y, prec, round),
            ),
            (
                "sqrt",
                Box::new(|z| z.set_sqrt_parts(&x, round)),
                x.sqrt_round_parts(prec, round),
            ),
            (
                "add_real",
                Box::new(|z| z.set_add_real_parts(&x, &r, round)),
                x.add_real_round_parts(&r, prec, round),
            ),
            (
                "sub_real",
                Box::new(|z| z.set_sub_real_parts(&x, &r, round)),
                x.sub_real_round_parts(&r, prec, round),
            ),
            (
                "mul_real",
                Box::new(|z| z.set_mul_real_parts(&x, &r, round)),
                x.mul_real_round_parts(&r, prec, round),
            ),
            (
                "div_real",
                Box::new(|z| z.set_div_real_parts(&x, &r, round)),
                x.div_real_round_parts(&r, prec, round),
            ),
            (
                "neg",
                Box::new(|z| z.set_neg_parts(&x, round)),
                (-&x).prec_round_parts(prec, round),
            ),
            (
                "conj",
                Box::new(|z| z.set_conj_parts(&x, round)),
                x.conj().prec_round_parts(prec, round),
            ),
        ];
        for (name, into, (want, wo)) in ops {
            let mut z = target(&mut s, prec);
            let o = into(&mut z);
            assert_eq!(z.prec(), prec, "{name} {what}");
            for (got, want) in parts_of(z, o).iter().zip(&parts_of(want, wo)) {
                same(got, want, &format!("{name} {what}"));
            }
        }
        // The norm and the absolute value into a float.
        let mut n = target(&mut s, prec).into_parts().0;
        let o = n.set_complex_norm(&x, round.0);
        same(
            &(n, o),
            &x.norm_round(prec.0, round.0),
            &format!("norm {what}"),
        );
        let mut a = target(&mut s, prec).into_parts().0;
        let o = a.set_complex_abs(&x, round.0);
        same(
            &(a, o),
            &x.abs_round(prec.0, round.0),
            &format!("abs {what}"),
        );
        // The forms that assign, at `self`'s precisions, one mode.
        let rd = round.0;
        let both = (rd, rd);
        let at = |z: &BigComplex| z.prec_round_parts(prec, (Round::Nearest, Round::Nearest)).0;
        let ops: Vec<Assigned> = {
            let mut v = Vec::new();
            let w = at(&x);
            let mut z = w.clone();
            let o = z.add_assign_round(&y, rd);
            v.push(("add_assign", w.add_round_parts(&y, prec, both), z, o));
            let mut z = w.clone();
            let o = z.sub_assign_round(&y, rd);
            v.push(("sub_assign", w.sub_round_parts(&y, prec, both), z, o));
            let mut z = w.clone();
            let o = z.mul_assign_round(&y, rd);
            v.push(("mul_assign", w.mul_round_parts(&y, prec, both), z, o));
            let mut z = w.clone();
            let o = z.div_assign_round(&y, rd);
            v.push(("div_assign", w.div_round_parts(&y, prec, both), z, o));
            let mut z = w.clone();
            let o = z.sqr_assign_round(rd);
            v.push(("sqr_assign", w.sqr_round_parts(prec, both), z, o));
            v
        };
        for (name, (want, wo), z, o) in ops {
            for (got, want) in parts_of(z, o).iter().zip(&parts_of(want, wo)) {
                same(got, want, &format!("{name} {what}"));
            }
        }
    }
}

/// A float of `p` bits whose mantissa is `m` (top bit set, at most `p` bits) times `2^e`, the
/// exponent `e` of `0.m·2^e`.
fn of_mantissa(m: &Number, e: i64, p: u64, negative: bool) -> BigFloat {
    let m = if negative { -m.clone() } else { m.clone() };
    let (x, o) = BigFloat::from_parts(&m, e - m.bits() as i64, p, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

#[test]
fn short_products_near_cancellation_round_part_by_part() {
    // Parts of one and two limbs (the products of 128 and 256 bits in registers) whose products
    // `ac` and `bd` agree in their top bits, or have exponents up to 300 apart (the shifts of the
    // 384-bit window past one and two limbs), against the exact rationals.
    let mut s = 0x5EED_C0C0_0000_0011u64;
    for i in 0..size(6000, 40) {
        let pa = 1 + below(&mut s, if i % 2 == 0 { 64 } else { 128 });
        let pb = 1 + below(&mut s, 128);
        let a = super::common::mantissa(&mut s, pa);
        let b = super::common::mantissa(&mut s, pb);
        let ea = below(&mut s, 40) as i64 - 20;
        let gap = match below(&mut s, 4) {
            0 => 0,
            1 => below(&mut s, 3) as i64,
            2 => below(&mut s, 140) as i64,
            _ => below(&mut s, 300) as i64,
        };
        let (na, nb) = (below(&mut s, 2) == 0, below(&mut s, 2) == 0);
        // `(a + b·i)·(b' + a'·i)` with `b'` and `a'` `b` and `a` changed in their last bits, so
        // that `ab' - ba'` cancels to their last bits.
        let tweak = |s: &mut u64, m: &Number, p: u64| -> Number {
            let d = Number::from(below(s, 5));
            let t = if below(s, 2) == 0 { m + &d } else { m - &d };
            if t.bits() == p && !t.is_zero() {
                t
            } else {
                m.clone()
            }
        };
        let (a2, b2) = (tweak(&mut s, &a, pa), tweak(&mut s, &b, pb));
        let x = BigComplex::new(
            of_mantissa(&a, ea, pa, na),
            of_mantissa(&b, ea - gap, pb, nb),
        );
        let y = BigComplex::new(
            of_mantissa(&b2, ea - gap, pb, nb),
            of_mantissa(&a2, ea, pa, na),
        );
        let prec = (1 + below(&mut s, 128), 1 + below(&mut s, 128));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("{x:?} · {y:?} at {prec:?} {round:?}");
        let (ra, rb, rc, rd) = (
            rational(x.re()),
            rational(x.im()),
            rational(y.re()),
            rational(y.im()),
        );
        let mut z = target(&mut s, prec);
        let o = z.set_mul_parts(&x, &y, round);
        let [re, im] = parts_of(z, o);
        part(
            re,
            &(&ra * &rc - &rb * &rd),
            prec.0,
            round.0,
            &format!("re {what}"),
        );
        part(
            im,
            &(&ra * &rd + &rb * &rc),
            prec.1,
            round.1,
            &format!("im {what}"),
        );
        // The square and the norm of `x`.
        let mut z = target(&mut s, prec);
        let o = z.set_sqr_parts(&x, round);
        let [re, im] = parts_of(z, o);
        part(
            re,
            &(&ra * &ra - &rb * &rb),
            prec.0,
            round.0,
            &format!("re sqr {what}"),
        );
        let two = Rational::from(2u8);
        part(
            im,
            &(&two * &ra * &rb),
            prec.1,
            round.1,
            &format!("im sqr {what}"),
        );
        let mut n = BigFloat::zero(prec.0);
        let o = n.set_complex_norm(&x, round.0);
        part(
            (n, o),
            &(&ra * &ra + &rb * &rb),
            prec.0,
            round.0,
            &format!("norm {what}"),
        );
        let mut m = BigFloat::zero(prec.1);
        let o = m.set_complex_abs(&x, round.1);
        same(
            &(m, o),
            &sqrt_reference(&(&ra * &ra + &rb * &rb), prec.1, round.1),
            &format!("abs {what}"),
        );
    }
}

#[test]
fn short_roots_round_part_by_part_across_the_exponent_range() {
    // The roots in registers (results to 113 bits) and on the pool (beyond), of parts of every
    // precision to 128 bits and exponents at the bottom of the range, the top and between, against a
    // root 100 bits more precise rounded (unless that one is within reach of a halfway point).
    let mut s = 0x5EED_C0C0_0000_0012u64;
    for i in 0..size(4000, 30) {
        // Both exponents near the top of the range, near the bottom, or near zero, at most 200
        // apart (parts much further apart take a root far more precise than this one).
        let base = match below(&mut s, 5) {
            0 => BigFloat::EXP_MAX - 300,
            1 => BigFloat::EXP_MIN + 300,
            _ => 0,
        };
        let e = |s: &mut u64| -> i64 { base + below(s, 200) as i64 - 100 };
        let (pa, pb) = (1 + below(&mut s, 128), 1 + below(&mut s, 128));
        let (ea, eb) = (e(&mut s), e(&mut s));
        let a = super::common::mantissa(&mut s, pa);
        let b = super::common::mantissa(&mut s, pb);
        let z = BigComplex::new(
            of_mantissa(&a, ea, pa, below(&mut s, 2) == 0),
            of_mantissa(&b, eb, pb, below(&mut s, 2) == 0),
        );
        let top = if i % 3 == 0 { 300 } else { 113 };
        let prec = (1 + below(&mut s, top), 1 + below(&mut s, top));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("sqrt({z:?}) at {prec:?} {round:?}");
        let mut r = target(&mut s, prec);
        let o = r.set_sqrt_parts(&z, round);
        let (hi, ho) = z.sqrt_round_parts(
            (prec.0 + 100, prec.1 + 100),
            (Round::Nearest, Round::Nearest),
        );
        for (got, h, ho, p, rd) in [
            ((r.re().clone(), o.0), hi.re(), ho.0, prec.0, round.0),
            ((r.im().clone(), o.1), hi.im(), ho.1, prec.1, round.1),
        ] {
            let (hp, hpo) = h.prec_round(p, rd);
            if ho == Ordering::Equal
                || hpo != Ordering::Equal
                    && h.prec_round(p + 1, Round::Nearest).1 != Ordering::Equal
            {
                same(&got, &(hp, hpo), &what);
            }
        }
    }
}

#[test]
fn products_and_squares_beyond_the_exponent_range_overflow_and_underflow_by_mode() {
    // A real part `ac - bd` with `bd = 0` (`b` zero): the product of floats, which overflows or
    // underflows as `mul_round` does, in the registers, the window products and the
    // accumulator alike.
    let mut s = 0x5EED_C0C0_0000_0013u64;
    for _ in 0..size(2000, 20) {
        let p = match below(&mut s, 3) {
            0 => 1 + below(&mut s, 64),
            1 => 65 + below(&mut s, 64),
            _ => 129 + below(&mut s, 400),
        };
        let big = below(&mut s, 2) == 0;
        let e = |s: &mut u64| -> i64 {
            if big {
                BigFloat::EXP_MAX / 2 + below(s, 4) as i64
            } else {
                BigFloat::EXP_MIN / 2 - below(s, 4) as i64
            }
        };
        let a = of_mantissa(
            &super::common::mantissa(&mut s, p),
            e(&mut s),
            p,
            below(&mut s, 2) == 0,
        );
        let c = of_mantissa(
            &super::common::mantissa(&mut s, p),
            e(&mut s),
            p,
            below(&mut s, 2) == 0,
        );
        let z = BigComplex::new(a.clone(), BigFloat::zero(p));
        let w = BigComplex::new(c.clone(), BigFloat::zero(p));
        let round = (mode(&mut s), mode(&mut s));
        let mut out = target(&mut s, (p, p));
        let o = out.set_mul_parts(&z, &w, round);
        same(
            &(out.re().clone(), o.0),
            &a.mul_round(&c, p, round.0),
            &format!("{a:?} · {c:?} at {p} {round:?}"),
        );
        let mut out = target(&mut s, (p, p));
        let o = out.set_sqr_parts(&z, round);
        same(
            &(out.re().clone(), o.0),
            &a.mul_round(&a, p, round.0),
            &format!("{a:?}² at {p} {round:?}"),
        );
        let mut n = BigFloat::zero(p);
        let o = n.set_complex_norm(&z, round.0);
        same(
            &(n, o),
            &a.mul_round(&a, p, round.0),
            &format!("|{a:?}|² at {p} {round:?}"),
        );
    }
}

#[test]
fn operators_that_assign_keep_the_larger_precisions() {
    let mut s = 0x5EED_C0C0_0000_0014u64;
    for _ in 0..size(500, 10) {
        let x = random(&mut s, 200, 20);
        let y = random(&mut s, 200, 20);
        let mut z = x.clone();
        z += &y;
        assert_eq!(z, &x + &y);
        assert_eq!(z.prec(), (&x + &y).prec());
        let mut z = x.clone();
        z -= &y;
        assert_eq!(z, &x - &y);
        let mut z = x.clone();
        z *= &y;
        assert_eq!(z, &x * &y);
        assert_eq!(z.prec(), (&x * &y).prec());
        let mut z = x.clone();
        z /= &y;
        assert_eq!(z, &x / &y);
    }
}

/// A mantissa of `bits` bits that is sparse: its top bit and a few others set.
fn sparse(s: &mut u64, bits: u64) -> Number {
    let mut m = Number::new_from_u8(0, ByteOrder::LittleEndian);
    m.set_bit(bits - 1, true);
    for _ in 0..below(s, 4) {
        m.set_bit(below(s, bits), true);
    }
    m
}

#[test]
fn sparse_short_products_round_part_by_part() {
    // Parts of one and two limbs with few bits set, so that the exact products have long runs
    // of zero limbs, and of exponents that put the two products of a part from 0 to 400 bits
    // apart (every shift of the 384-bit window and past it), each sign: the parts of the product,
    // the square, the norm and the absolute value against the exact rationals.
    let mut s = 0x5EED_C0C0_0000_0015u64;
    for _ in 0..size(30_000, 60) {
        let mut p = || {
            let bits = match below(&mut s, 3) {
                0 => 1 + below(&mut s, 64),
                _ => 65 + below(&mut s, 64),
            };
            let m = if below(&mut s, 3) == 0 {
                super::common::mantissa(&mut s, bits)
            } else {
                sparse(&mut s, bits)
            };
            (m, bits, below(&mut s, 2) == 0)
        };
        let (a, b, c, d) = (p(), p(), p(), p());
        let g = match below(&mut s, 4) {
            0 => below(&mut s, 3) as i64,
            1 => 120 + below(&mut s, 20) as i64,
            2 => 250 + below(&mut s, 20) as i64,
            _ => below(&mut s, 400) as i64,
        };
        // `ac` at exponent `ea + ec`, `bd` at `ea + ec - g` (and `ad`, `bc` alike).
        let (ea, ec) = (below(&mut s, 20) as i64 - 10, below(&mut s, 20) as i64 - 10);
        let eb = below(&mut s, 20) as i64 - 10 - g;
        let ed = ea + ec - eb - g - g;
        let x = BigComplex::new(
            of_mantissa(&a.0, ea, a.1, a.2),
            of_mantissa(&b.0, eb, b.1, b.2),
        );
        let y = BigComplex::new(
            of_mantissa(&c.0, ec, c.1, c.2),
            of_mantissa(&d.0, ed + g, d.1, d.2),
        );
        let prec = (1 + below(&mut s, 128), 1 + below(&mut s, 128));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("{x:?} · {y:?} at {prec:?} {round:?}");
        let (ra, rb, rc, rd) = (
            rational(x.re()),
            rational(x.im()),
            rational(y.re()),
            rational(y.im()),
        );
        let mut z = target(&mut s, prec);
        let o = z.set_mul_parts(&x, &y, round);
        let [re, im] = parts_of(z, o);
        part(
            re,
            &(&ra * &rc - &rb * &rd),
            prec.0,
            round.0,
            &format!("re {what}"),
        );
        part(
            im,
            &(&ra * &rd + &rb * &rc),
            prec.1,
            round.1,
            &format!("im {what}"),
        );
        let mut z = target(&mut s, prec);
        let o = z.set_sqr_parts(&x, round);
        let [re, im] = parts_of(z, o);
        part(
            re,
            &(&ra * &ra - &rb * &rb),
            prec.0,
            round.0,
            &format!("re sqr {what}"),
        );
        let two = Rational::from(2u8);
        part(
            im,
            &(&two * &ra * &rb),
            prec.1,
            round.1,
            &format!("im sqr {what}"),
        );
        let mut n = BigFloat::zero(prec.0);
        let o = n.set_complex_norm(&x, round.0);
        part(
            (n, o),
            &(&ra * &ra + &rb * &rb),
            prec.0,
            round.0,
            &format!("norm {what}"),
        );
        let mut m = BigFloat::zero(prec.1);
        let o = m.set_complex_abs(&x, round.1);
        same(
            &(m, o),
            &sqrt_reference(&(&ra * &ra + &rb * &rb), prec.1, round.1),
            &format!("abs {what}"),
        );
    }
}

#[test]
fn roots_just_beside_a_rounding_boundary_round_part_by_part() {
    // `√(q²)` for `q` whose parts are a number of the precision `p`, or a halfway point, moved by
    // `δ` of about the bound of the working precision's error (`2^(e + 4 - w)`, `w = p + guard`):
    // the attempts' intervals then hold the boundary, and must not decide on the wrong side.
    let mut s = 0x5EED_C0C0_0000_0016u64;
    for _ in 0..size(3000, 20) {
        let p = if below(&mut s, 2) == 0 {
            20 + below(&mut s, 94)
        } else {
            114 + below(&mut s, 200)
        };
        let guard = 2 * (64 - p.leading_zeros() as u64) + 16;
        let w = if p <= 113 {
            (p + guard).min(127)
        } else {
            p + guard
        };
        let q_part = |s: &mut u64, e: i64| -> BigFloat {
            let t = super::common::mantissa(s, p);
            // `t·2^k + h + δ`, `k` bits below: `h` half a unit (or none), `δ` near `2^(p + 4 - w)`
            // units of `t`'s last place... in units of `2^-k`.
            let k = w - p + 8;
            let mut m = &t << k as usize;
            if below(s, 2) == 0 {
                m = &m + &pow2(k - 1);
            }
            let j = (k as i64 - (w - p) as i64 + 4 + below(s, 9) as i64 - 6).max(0) as u64;
            let d = &pow2(j) + &Number::from(below(s, 1 << 10));
            m = if below(s, 2) == 0 { &m + &d } else { &m - &d };
            let bits = m.bits();
            let neg = below(s, 2) == 0;
            of_mantissa(&m, e, bits, neg)
        };
        let e = below(&mut s, 20) as i64 - 10;
        let re = q_part(&mut s, e);
        let shift = below(&mut s, 4) as i64;
        let im = q_part(&mut s, e - shift);
        let q = BigComplex::new(re.abs(), im);
        let sq = q.sqr_round(100_000, Round::Nearest);
        assert_eq!(sq.1, (Ordering::Equal, Ordering::Equal));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("sqrt({:?}) at {p} {round:?}", sq.0);
        let (r, o) = sq.0.sqrt_round_parts((p, p), round);
        let (a, b) = r.into_parts();
        same(&(a, o.0), &reference(&rational(q.re()), p, round.0), &what);
        same(&(b, o.1), &reference(&rational(q.im()), p, round.1), &what);
    }
}

/// The mode that rounds `-x` as `round` rounds `x`: up and down swapped.
fn swapped(round: Round) -> Round {
    match round {
        Round::Up => Round::Down,
        Round::Down => Round::Up,
        r => r,
    }
}

#[test]
fn roots_of_parts_astronomically_apart_end_and_round() {
    // `√(a + b·i)` with `|b|` so far below `|a|` (exponents near the top and the bottom of the
    // range, or just past the threshold of the direct decision) that the root is `√|a|` just
    // above and `|b|/(2√|a|)` just below: for a square `a` both are exact, and only the side
    // decides; Ziv's intervals alone would need `2(e_a - e_b)` bits.
    let mut s = 0x5EED_C0C0_0000_0017u64;
    for i in 0..size(1500, 20) {
        let p = 1 + below(&mut s, 200);
        let pc = 1 + below(&mut s, 100);
        let square = i % 3 != 0;
        let (ec, eb) = if i % 2 == 0 {
            (
                BigFloat::EXP_MAX / 2 - below(&mut s, 100) as i64,
                BigFloat::EXP_MIN + below(&mut s, 100) as i64,
            )
        } else {
            (
                below(&mut s, 20) as i64 - 10,
                -(2000 + below(&mut s, 500) as i64),
            )
        };
        let c = of_mantissa(&super::common::mantissa(&mut s, pc), ec, pc, false);
        let a = if square {
            c.mul_round(&c, 2 * pc, Round::Nearest).0
        } else {
            let m = super::common::mantissa(&mut s, pc);
            of_mantissa(&m, 2 * ec - below(&mut s, 2) as i64, pc, false)
        };
        let pb = 1 + below(&mut s, 100);
        let b = of_mantissa(
            &super::common::mantissa(&mut s, pb),
            eb,
            pb,
            below(&mut s, 2) == 0,
        );
        let negative_a = below(&mut s, 2) == 0;
        let a = if negative_a { -a } else { a };
        let round = (mode(&mut s), mode(&mut s));
        let z = BigComplex::new(a.clone(), b.clone());
        let what = format!("sqrt({z:?}) at {p} {round:?}");
        let (r, o) = z.sqrt_round_parts((p, p), round);
        // Which part is `t` and which `u`, and their modes.
        let (t, ot, rt, tn, u, ou, ru, un) = if negative_a {
            (
                r.im().clone(),
                o.1,
                round.1,
                b.is_sign_negative(),
                r.re().clone(),
                o.0,
                round.0,
                false,
            )
        } else {
            (
                r.re().clone(),
                o.0,
                round.0,
                false,
                r.im().clone(),
                o.1,
                round.1,
                b.is_sign_negative(),
            )
        };
        let flip = |x: &(BigFloat, Ordering), n: bool| -> (BigFloat, Ordering) {
            if n { (-&x.0, x.1.reverse()) } else { x.clone() }
        };
        let tiny = |x: &BigFloat| {
            BigFloat::from_parts(
                &one(),
                x.exponent().unwrap() - 4 * p as i64 - 1000,
                1,
                Round::Nearest,
            )
            .0
        };
        // `t`: `√|a|` just above, in the mode for its sign.
        let rt_mag = if tn { swapped(rt) } else { rt };
        // `√|a| + δ` for a `δ > 0` far below: the exact root (to `4·pc` bits) with `δ` added, or
        // the root itself when it is not dyadic.
        let (sx, sxo) = a.abs().sqrt_round(4 * pc + 8, Round::Nearest);
        let want_t = if sxo == Ordering::Equal {
            sx.add_round(&tiny(&sx), p, rt_mag)
        } else {
            a.abs().sqrt_round(p, rt_mag)
        };
        same(&(t, ot), &flip(&want_t, tn), &format!("t of {what}"));
        if square {
            // `u`: `|b|/(2|c|)` just below.
            let ru_mag = if un { swapped(ru) } else { ru };
            let half = b
                .abs()
                .mul_round(&BigFloat::from(0.5), pb, Round::Nearest)
                .0;
            let (qx, qxo) = half.div_round(&c, 4 * (pb + pc) + p + 8, Round::Nearest);
            let want_u = if qxo == Ordering::Equal {
                qx.sub_round(&tiny(&qx), p, ru_mag)
            } else {
                half.div_round(&c, p, ru_mag)
            };
            same(&(u, ou), &flip(&want_u, un), &format!("u of {what}"));
        }
    }
}

#[test]
fn long_sums_at_the_edges_of_the_window_round_part_by_part() {
    // Products of parts of up to two limbs (256 bits exact), summed in the 384-bit window of
    // `add4`, built so that each bit the window keeps apart from the precision decides:
    // - `(2^127 + 1) - (2^128 - 1)·2^-129`: a difference whose only bit below the round bit
    //   (at 128 bits) is in the window's low limb, past a tie;
    // - `(2^128 - 1)² - 3·(2^129 + 1)/3·2^-129`: a shift of 255 that drops the low limb of the
    //   second product alone (the middle limb's sticky bit), the rest of the difference a number
    //   of 127 bits;
    // - `(2^127 + 1)(2^128 - 2) + 5·(2^130 + 1)/5·2^-129`: a sum that carries out of the window
    //   (the first product normalised to `2^256 - 4`) and drops its last bit, `2^255 + 2^-129`.
    // Each scaled by `2^k`, negated, conjugated and turned by `i`, at precisions to 128 bits,
    // every mode, against the exact rationals.
    let n = |m: Number, e: i64, p: u64, negative: bool| of_mantissa(&m, e, p, negative);
    let ones = &pow2(128) - &one();
    let third = Number::from(3u8);
    let fifth = Number::from(5u8);
    let by3 = (&pow2(129) + &one()).div_rem(&third).0;
    let by5 = (&pow2(130) + &one()).div_rem(&fifth).0;
    assert_eq!(by3.bits(), 128);
    assert_eq!(by5.bits(), 128);
    let cases = [
        (
            n(&pow2(127) + &one(), 128, 128, false),
            n(ones.clone(), -1, 128, false),
            n(one(), 1, 1, false),
            n(one(), 1, 1, false),
        ),
        (
            n(ones.clone(), 128, 128, false),
            n(third.clone(), 2, 2, false),
            n(ones.clone(), 128, 128, false),
            n(by3.clone(), -1, 128, false),
        ),
        (
            n(&pow2(127) + &one(), 128, 128, false),
            n(fifth.clone(), -126, 3, true),
            n(&pow2(127) - &one(), 128, 127, false),
            n(by5.clone(), 128, 128, false),
        ),
    ];
    let mut s = 0x5EED_C0C0_0000_0019u64;
    let precs: &[u64] = if cfg!(miri) {
        &[2, 64, 127, 128]
    } else {
        &[1, 2, 53, 64, 65, 100, 126, 127, 128]
    };
    for (c, (a, b, cc, d)) in cases.iter().enumerate() {
        for v in 0..8 {
            if cfg!(miri) && v % 3 != 0 {
                continue;
            }
            let k = below(&mut s, 200) as i64 - 100;
            let scale = |x: &BigFloat| {
                x.mul_round(
                    &BigFloat::from_parts(&one(), k, 1, Round::Nearest).0,
                    x.prec(),
                    Round::Nearest,
                )
                .0
            };
            let mut x = BigComplex::new(scale(a), scale(b));
            let mut y = BigComplex::new(cc.clone(), d.clone());
            if v & 1 == 1 {
                x = -&x;
            }
            if v & 2 == 2 {
                x = x.conj();
                y = y.conj();
            }
            if v & 4 == 4 {
                // `(ix)(y)`: the parts swap, one negated.
                x = BigComplex::new(-x.im(), x.re().clone());
            }
            let (ra, rb, rc, rd) = (
                rational(x.re()),
                rational(x.im()),
                rational(y.re()),
                rational(y.im()),
            );
            let (re, im) = (&ra * &rc - &rb * &rd, &ra * &rd + &rb * &rc);
            for &p in precs {
                for round in MODES {
                    let prec = (p, p);
                    let round = (round, mode(&mut s));
                    let what = format!("case {c} variant {v}: {x:?} · {y:?} at {prec:?} {round:?}");
                    let (r, o) = x.mul_round_parts(&y, prec, round);
                    let (got_re, got_im) = r.into_parts();
                    part((got_re, o.0), &re, prec.0, round.0, &format!("re {what}"));
                    part((got_im, o.1), &im, prec.1, round.1, &format!("im {what}"));
                    let mut z = target(&mut s, prec);
                    let o = z.set_mul_parts(&x, &y, round);
                    let [zr, zi] = parts_of(z, o);
                    part(zr, &re, prec.0, round.0, &format!("into re {what}"));
                    part(zi, &im, prec.1, round.1, &format!("into im {what}"));
                }
            }
        }
    }
}

#[test]
fn roots_with_an_exact_part_beside_a_long_one_round_part_by_part() {
    // `√(q²)` for `q = m + n·i` with `m > 0` far longer than the working precision `w` and
    // `|n| < m` a number of the precision `p` or a halfway point of it: `t = m` is cut at every
    // step, so `u = |b|/(2t)` comes out a little above `|n|`, within the bound of the working
    // precision, and the decision of the long parts must leave it to the exact root. Then the
    // same with the parts' roles swapped, `q = n + m·i` with `n > 0` (`a < 0`).
    let mut s = 0x5EED_C0C0_0000_001Au64;
    for i in 0..size(800, 8) {
        let p = 1 + below(&mut s, 300);
        let guard = 2 * (64 - p.leading_zeros() as u64) + 16;
        let lm = p + guard + 20 + below(&mut s, 200);
        let m = super::common::mantissa(&mut s, lm);
        let nm = if i % 3 == 0 {
            &(&super::common::mantissa(&mut s, p) << 1usize) + &one()
        } else {
            super::common::mantissa(&mut s, p)
        };
        let em = below(&mut s, 20) as i64 - 10;
        let en = em - 1 - below(&mut s, 6) as i64;
        let mf = of_mantissa(&m, em, lm, false);
        let negative = below(&mut s, 2) == 0;
        let nf = of_mantissa(&nm, en, nm.bits(), negative);
        let q = if i % 2 == 0 {
            BigComplex::new(mf, nf)
        } else {
            BigComplex::new(nf.abs(), if negative { -mf } else { mf })
        };
        let sq = q.sqr_round(100_000, Round::Nearest);
        assert_eq!(sq.1, (Ordering::Equal, Ordering::Equal));
        let round = (mode(&mut s), mode(&mut s));
        let what = format!("sqrt({:?}) at {p} {round:?}", sq.0);
        let (r, o) = sq.0.sqrt_round_parts((p, p), round);
        let (a, b) = r.into_parts();
        same(&(a, o.0), &reference(&rational(q.re()), p, round.0), &what);
        same(&(b, o.1), &reference(&rational(q.im()), p, round.1), &what);
        let mut z = target(&mut s, (p, p));
        let o = z.set_sqrt_parts(&sq.0, round);
        let [a, b] = parts_of(z, o);
        same(
            &a,
            &reference(&rational(q.re()), p, round.0),
            &format!("into {what}"),
        );
        same(
            &b,
            &reference(&rational(q.im()), p, round.1),
            &format!("into {what}"),
        );
    }
}

#[test]
fn roots_of_real_parts_astronomically_below_the_imaginary_end_and_round() {
    // `√(a + b·i)` with `|a|` so far below `|b|` (exponents near the bottom and the top of the
    // range, or a few thousand apart) that both parts are `s = √(|b|/2)` within `|a|/|b|`: `t`
    // just above, `u` just below. For `b = ±2c²` `s = |c|` is exact and only the side decides;
    // Ziv's intervals alone would need `e_b - e_a` bits.
    let mut s = 0x5EED_C0C0_0000_001Bu64;
    for i in 0..size(1500, 20) {
        let p = 1 + below(&mut s, 200);
        let pc = 1 + below(&mut s, 100);
        let square = i % 3 != 0;
        let (ec, ea) = if i % 2 == 0 {
            (
                BigFloat::EXP_MAX / 2 - below(&mut s, 100) as i64,
                BigFloat::EXP_MIN + below(&mut s, 100) as i64,
            )
        } else {
            (
                below(&mut s, 20) as i64 - 10,
                -(2000 + below(&mut s, 500) as i64),
            )
        };
        let c = of_mantissa(&super::common::mantissa(&mut s, pc), ec, pc, false);
        let b = if square {
            let cc = c.mul_round(&c, 2 * pc, Round::Nearest).0;
            cc.mul_round(&BigFloat::from(2), 2 * pc, Round::Nearest).0
        } else {
            let m = super::common::mantissa(&mut s, pc);
            of_mantissa(&m, 2 * ec - below(&mut s, 2) as i64, pc, false)
        };
        let negative_b = below(&mut s, 2) == 0;
        let b = if negative_b { -b } else { b };
        let pa = 1 + below(&mut s, 100);
        let a = of_mantissa(
            &super::common::mantissa(&mut s, pa),
            ea,
            pa,
            below(&mut s, 2) == 0,
        );
        let round = (mode(&mut s), mode(&mut s));
        let z = BigComplex::new(a.clone(), b.clone());
        let what = format!("sqrt({z:?}) at {p} {round:?}");
        let (r, o) = z.sqrt_round_parts((p, p), round);
        // `s` just above (`t`) or just below (`u`), rounded in the mode for the part's
        // magnitude, and given the part's sign.
        let near = |above: bool, negative: bool, round: Round| -> (BigFloat, Ordering) {
            let mode = if negative { swapped(round) } else { round };
            // `|b|/2`, exactly, and its root to more bits than it can have when it is exact.
            let half = b
                .abs()
                .mul_round(&BigFloat::from(0.5), 2 * pc, Round::Nearest)
                .0;
            let (sx, sxo) = half.sqrt_round(4 * pc + 8, Round::Nearest);
            let want = if sxo == Ordering::Equal {
                let tiny = BigFloat::from_parts(
                    &one(),
                    sx.exponent().unwrap() - 4 * (p + pc) as i64 - 100,
                    1,
                    Round::Nearest,
                )
                .0;
                if above {
                    sx.add_round(&tiny, p, mode)
                } else {
                    sx.sub_round(&tiny, p, mode)
                }
            } else {
                half.sqrt_round(p, mode)
            };
            if negative {
                (-&want.0, want.1.reverse())
            } else {
                want
            }
        };
        // `(re, im)` is `(t, ±u)` for `a > 0` and `(u, ±t)` for `a < 0`.
        let t_is_re = !a.is_sign_negative();
        same(
            &(r.re().clone(), o.0),
            &near(t_is_re, false, round.0),
            &format!("re of {what}"),
        );
        same(
            &(r.im().clone(), o.1),
            &near(!t_is_re, negative_b, round.1),
            &format!("im of {what}"),
        );
    }
}
