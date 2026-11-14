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
