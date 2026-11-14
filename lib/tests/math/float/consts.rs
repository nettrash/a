//! The constants against MPFR's vectors (every mode, with the ternary), against long floors of
//! their values from MPFR (the rounding of every precision they decide, in every mode, the series
//! beyond the tables included), against fingerprints of MPFR's values at large precisions, and
//! against themselves at different precisions and from several threads.

use super::common::{MODES, same, size};
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

/// MPFR's results: `const prec mode mantissa exponent ternary`.
const MPFR: &str = include_str!("data/consts_mpfr.txt");
/// `floor(C 2^36000)` in hexadecimal by MPFR.
const FLOORS: &str = include_str!("data/consts_hex.txt");
/// The bits after the point of [`FLOORS`].
const FLOOR_BITS: u64 = 36_000;

type Function = fn(u64, Round) -> (BigFloat, Ordering);

/// The function of a constant's name.
fn function(name: &str) -> Function {
    match name {
        "pi" => BigFloat::pi_round,
        "e" => BigFloat::e_round,
        "ln2" => BigFloat::ln2_round,
        "euler" => BigFloat::euler_gamma_round,
        "catalan" => BigFloat::catalan_round,
        "ln10" => BigFloat::ln10_round,
        _ => panic!("no constant {name}"),
    }
}

/// The names of the constants.
const NAMES: [&str; 6] = ["pi", "e", "ln2", "euler", "catalan", "ln10"];

fn mode(m: &str) -> Round {
    match m {
        "N" => Round::Nearest,
        "Z" => Round::TowardZero,
        "U" => Round::Up,
        "D" => Round::Down,
        _ => panic!("no mode {m}"),
    }
}

fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::LittleEndian).unwrap()
}

#[test]
fn constants_agree_with_mpfr() {
    let mut count = 0;
    let lines: Vec<&str> = MPFR.lines().filter(|l| !l.starts_with('#')).collect();
    // Under Miri a sample: every 97th line.
    let step = if cfg!(miri) { 97 } else { 1 };
    for line in lines.iter().step_by(step) {
        let v: Vec<&str> = line.split(' ').collect();
        let (name, prec, round) = (v[0], v[1].parse::<u64>().unwrap(), mode(v[2]));
        let (want, o) =
            BigFloat::from_parts(&hex(v[3]), v[4].parse().unwrap(), prec, Round::Nearest);
        assert_eq!(o, Ordering::Equal);
        let t = match v[5] {
            "-1" => Ordering::Less,
            "1" => Ordering::Greater,
            _ => Ordering::Equal,
        };
        let got = function(name)(prec, round);
        same(&got, &(want, t), &format!("{name} {prec} {round:?}"));
        count += 1;
    }
    assert!(count >= if cfg!(miri) { 50 } else { 4 * 6 * 207 });
}

/// The rounding of `C` to `prec` bits in the mode `round` from `F = floor(C 2^f)`, `C` irrational:
/// `C 2^f = F + θ`, `0 < θ < 1`. With `F = H 2^(d+1) + r`, `H` of `prec` bits, `C 2^f` lies
/// strictly between `H 2^(d+1)` and `(H + 1) 2^(d+1)`, above the middle exactly when bit `d` of
/// `F` is set (`r + θ > 2^d` then, `r + θ < 2^d` otherwise): every mode is decided.
fn from_floor(f: &Number, bits: u64, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    let len = f.bits();
    let d = len.checked_sub(prec + 1)?;
    let half = f.bit(d);
    let h = f >> (d + 1) as usize;
    let up = match round {
        Round::Nearest => half,
        Round::TowardZero | Round::Down => false,
        Round::Up => true,
    };
    // `|C| = (H + θ) 2^(d + 1 - f)`, `0 < θ < 1`: below `H + 1/2` exactly when `half` is clear.
    let m = if up { &h + &Number::from(1) } else { h };
    let (x, o) = BigFloat::from_parts(&m, d as i64 + 1 - bits as i64, prec, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    Some((
        x,
        if up {
            Ordering::Greater
        } else {
            Ordering::Less
        },
    ))
}

#[test]
fn constants_agree_with_long_floors() {
    // Every precision up to 400, then a sweep to the end of the floors past the tables (16,384
    // bits after the point), where the series take over.
    let mut precs: Vec<u64> = (1..=400).collect();
    precs.extend((401..=FLOOR_BITS - 64).step_by(size(997, 9_973)));
    precs.extend([
        16_300, 16_330, 16_340, 16_350, 16_380, 16_400, 16_500, 39_900,
    ]);
    // Under Miri the tables' range alone, from the top 300 digits of the floors:
    // `floor(F 16^-(L - 300)) = floor(C 2^(f - 4(L - 300)))` for the `L` digits of `F`.
    let digits_kept = if cfg!(miri) { Some(300) } else { None };
    if cfg!(miri) {
        precs = vec![1, 2, 24, 53, 64, 113, 200, 1000];
    }
    let mut decided = 0;
    for line in FLOORS.lines().filter(|l| !l.starts_with('#')) {
        let (name, digits) = line.split_once(' ').unwrap();
        let (digits, bits) = match digits_kept {
            Some(h) => (&digits[..h], FLOOR_BITS - 4 * (digits.len() - h) as u64),
            None => (digits, FLOOR_BITS),
        };
        let f = hex(digits);
        // And the hard cases of the floors: the precisions whose rounding bit, or the bit after
        // it, starts a run of 12 or more equal bits, where an approximation is least decisive.
        let mut all = precs.clone();
        if !cfg!(miri) {
            all.extend(hard_cases(&f, 12).into_iter().filter(|&p| p <= bits - 64));
        }
        for &prec in &all {
            for round in MODES {
                if let Some(want) = from_floor(&f, bits, prec, round) {
                    let got = function(name)(prec, round);
                    same(&got, &want, &format!("{name} {prec} {round:?}"));
                    decided += 1;
                }
            }
        }
    }
    assert!(decided >= 6 * 4 * precs.len());
}

/// The precisions `p` and `p + 1` of each run of at least `len` equal bits of `f` that begins `p`
/// bits below its top (so that the rounding bit is just above the run, or the run's first bit).
fn hard_cases(f: &Number, len: u64) -> Vec<u64> {
    let bits = f.bits();
    let mut r = Vec::new();
    let mut i = 0;
    while i < bits {
        let b = f.bit(bits - 1 - i);
        let mut j = i + 1;
        while j < bits && f.bit(bits - 1 - j) == b {
            j += 1;
        }
        if j - i >= len && i >= 2 {
            r.extend([i - 1, i]);
        }
        i = j;
    }
    r
}

/// The low 64 bits of `floor(C 2^f)` from MPFR at large `f`.
const FINGERPRINTS: [(&str, u64, u64); 12] = [
    ("pi", 100_000, 0xE40C6ACB3CC4804C),
    ("e", 100_000, 0xB46CDAC069437BAD),
    ("ln2", 100_000, 0x4BD7CC8FE7F77885),
    ("euler", 100_000, 0xE1C071E6659CE975),
    ("catalan", 100_000, 0xD1386C00129BE0E9),
    ("ln10", 100_000, 0x26FAF5F3B2686704),
    ("pi", 300_000, 0x9F66E01B4F748A88),
    ("e", 300_000, 0xE168F9CF98692A69),
    ("ln2", 300_000, 0x13FD024ED8BE34D8),
    ("euler", 300_000, 0x56D3FCC6D2C686CB),
    ("catalan", 300_000, 0xB11AF809E470B347),
    ("ln10", 300_000, 0xB26EB122FE071803),
];

/// `floor(C 2^f)` from `x = m 2^e`, `C` rounded down to `f + 66` bits of precision: `x ≤ C <
/// x + 2^e`, `e + f ≤ -64` for these constants (below 4), so `floor(m 2^(e+f))` is the floor
/// unless the bits of `m` below the point are all ones (`C 2^f` might pass the next integer).
fn floor_of(name: &str, f: u64) -> Number {
    let (x, _) = function(name)(f + 66, Round::Down);
    let (m, e) = x.mantissa_exponent().unwrap();
    let s = (-(e + f as i64)) as usize;
    assert!(s >= 64);
    let low = &m - &(&(&m >> s) << s);
    assert_ne!(&low + &Number::from(1), Number::from(1) << s);
    &m >> s
}

#[test]
fn constants_agree_with_mpfr_at_large_precisions() {
    if cfg!(miri) {
        return;
    }
    let count = if cfg!(debug_assertions) { 6 } else { 12 };
    for &(name, f, want) in &FINGERPRINTS[..count] {
        let x = floor_of(name, f);
        let low = &x - &(&(&x >> 64usize) << 64usize);
        assert_eq!(low, Number::from(want), "{name} {f}");
    }
}

/// A shorter value first, computed alone, then a longer one, computed again (the shorter is then
/// cut from the longer's cache): each rounded down agrees with the other; up and down a unit
/// apart, to nearest one of them. Under Miri one series just past the tables, the longer value
/// first, the shorter cut from it.
fn agrees_with_itself(name: &str) {
    let (short, long) = if cfg!(miri) {
        (16_350, 16_400)
    } else {
        (60_000, 123_457)
    };
    let f = function(name);
    let (a, b) = if cfg!(miri) {
        let b = f(long, Round::Down).0;
        (f(short, Round::Down).0, b)
    } else {
        let a = f(short, Round::Down).0;
        (a, f(long, Round::Down).0)
    };
    assert_eq!(b.prec_round(short, Round::Down).0, a, "{name}");
    let up = f(long, Round::Up).0;
    assert!(up > b);
    let n = f(long, Round::Nearest).0;
    assert!(n == b || n == up);
    assert_eq!(f(long, Round::TowardZero).0, b);
}

#[test]
fn pi_agrees_with_itself() {
    agrees_with_itself("pi");
}

#[test]
fn e_agrees_with_itself() {
    agrees_with_itself("e");
}

#[test]
fn ln2_agrees_with_itself() {
    agrees_with_itself("ln2");
}

#[test]
fn euler_agrees_with_itself() {
    // Brent–McMillan just past the table takes products that run Miri on a 32-bit target out of
    // address space; Miri on the 64-bit targets runs it.
    if cfg!(all(miri, target_pointer_width = "32")) {
        return;
    }
    agrees_with_itself("euler");
}

#[test]
fn catalan_agrees_with_itself() {
    agrees_with_itself("catalan");
}

#[test]
fn ln10_agrees_with_itself() {
    agrees_with_itself("ln10");
}

#[test]
fn constants_are_shared_between_threads() {
    // Under Miri within the tables: the threads share no computation then.
    let precs: Vec<u64> = if cfg!(miri) {
        vec![1_000, 2_000]
    } else {
        vec![20_000, 30_000, 25_000, 40_000]
    };
    let results: Vec<Vec<BigFloat>> = std::thread::scope(|s| {
        let handles: Vec<_> = precs
            .iter()
            .map(|&p| {
                s.spawn(move || {
                    NAMES
                        .iter()
                        .map(|n| {
                            function(n)(p, Round::Nearest)
                                .0
                                .with_prec(size(16_000, 900) as u64)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in &results[1..] {
        assert_eq!(r, &results[0]);
    }
}

#[test]
fn constants_at_small_precisions() {
    assert_eq!(BigFloat::pi(53), BigFloat::from(core::f64::consts::PI));
    assert_eq!(BigFloat::pi(24), BigFloat::from(core::f32::consts::PI));
    assert_eq!(BigFloat::e(53), BigFloat::from(core::f64::consts::E));
    assert_eq!(BigFloat::ln2(53), BigFloat::from(core::f64::consts::LN_2));
    assert_eq!(
        BigFloat::euler_gamma(53),
        BigFloat::from(0.577_215_664_901_532_9_f64)
    );
    assert_eq!(
        BigFloat::catalan(53),
        BigFloat::from(0.915_965_594_177_219_f64)
    );
    for name in NAMES {
        for round in MODES {
            let (x, o) = function(name)(1, round);
            assert_eq!(x.prec(), 1);
            assert_ne!(o, Ordering::Equal);
        }
    }
    // π to one bit: 4 to nearest and up, 2 down.
    assert_eq!(BigFloat::pi(1), BigFloat::from(4));
    assert_eq!(BigFloat::pi_round(1, Round::Down).0, BigFloat::from(2));
}
