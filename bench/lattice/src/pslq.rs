//! PSLQ: `pslq` on BigFloats and `pslq_f64`, against mpmath's `pslq` (a Python co-process, so
//! that the samples still alternate) and PARI's `lindep` (an LLL method, in this process).
#![allow(unused_variables, unused_mut)]
use crate::timing::Side;
use crate::{Ctx, Ref};
#[allow(unused_imports)]
use crate::refs;
use a::math::float::BigFloat;
use a::math::lattice::{Pslq, PslqOptions, pslq as a_pslq, pslq_f64};
use a::math::numbers::Number;
use std::cell::RefCell;
use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Mpmath {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    banner: String,
}

impl Mpmath {
    fn start() -> Option<Mpmath> {
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/pslq_mpmath.py");
        let py = std::env::var("LATBENCH_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(py)
            .args(["-I", script])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let mut stdout = BufReader::new(child.stdout.take()?);
        let mut banner = String::new();
        stdout.read_line(&mut banner).ok()?;
        if !banner.starts_with("mpmath") {
            return None;
        }
        Some(Mpmath {
            child,
            stdin,
            stdout,
            banner: banner.trim().to_string(),
        })
    }

    fn cmd(&mut self, s: &str) -> String {
        self.stdin.write_all(s.as_bytes()).unwrap();
        self.stdin.write_all(b"\n").unwrap();
        self.stdin.flush().unwrap();
        let mut l = String::new();
        self.stdout.read_line(&mut l).unwrap();
        l.trim().to_string()
    }

    fn case(&mut self, x: &[(Number, i64)], prec: u64, maxcoeff: &Number, maxsteps: u64) {
        let mut s = format!("case {prec} {maxcoeff} {maxsteps}\n");
        for (m, e) in x {
            s.push_str(&format!("{m} {e}\n"));
        }
        s.push_str("end");
        assert_eq!(self.cmd(&s), "ok");
    }
}

impl Drop for Mpmath {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.child.wait();
    }
}

fn rounded(p: u64, f: impl Fn(u64) -> BigFloat) -> BigFloat {
    f(p + 64).with_prec(p)
}

fn powers_of_radicals(r1: u32, k1: u32, sign: i32, r2: u32, k2: u32, deg: i64, p: u64) -> Vec<BigFloat> {
    let alpha = |w: u64| {
        let a = BigFloat::from(r1).with_prec(w).nth_root(k1);
        let b = BigFloat::from(r2).with_prec(w).nth_root(k2);
        if sign > 0 { a + b } else { a - b }
    };
    (0..=deg).map(|k| rounded(p, |w| alpha(w).powi(k))).collect()
}

fn zeta3(w: u64) -> BigFloat {
    use a::math::float::Round;
    let w = w + 16;
    let mut s = BigFloat::zero(w);
    let mut c = Number::from(1);
    for k in 1u64.. {
        c = &c * &Number::from((2 * k) * (2 * k - 1)) / &Number::from(k * k);
        let d = &c * &Number::from(k * k * k);
        if d.bits() > w + 8 {
            break;
        }
        let t = BigFloat::from(1).div_round(&BigFloat::from(&d), w, Round::Nearest).0;
        s = if k % 2 == 1 { s.add_round(&t, w, Round::Nearest).0 } else { s.sub_round(&t, w, Round::Nearest).0 };
    }
    s.mul_round(&BigFloat::from(2.5), w, Round::Nearest).0
}

fn cases() -> Vec<(String, Vec<BigFloat>, Option<Number>)> {
    let quick = std::env::var("LATBENCH_QUICK").is_ok();
    let one = |p: u64| BigFloat::from(1).with_prec(p);
    let pow10 = |k: u32| Some(Number::from(10).pow(k));
    let mut v: Vec<(String, Vec<BigFloat>, Option<Number>)> = vec![(
        "minimal polynomial of √2+√3 (n=5, 200 bits)".into(),
        powers_of_radicals(2, 2, 1, 3, 2, 4, 200),
        None,
    )];
    if quick {
        return v;
    }
    // The same degree-6 polynomial (√2 + ∛3) at rising precision.
    for p in [128u64, 256, 512, 1024, 2048] {
        v.push((format!("minimal polynomial of √2+∛3 (n=7, {p} bits)"), powers_of_radicals(2, 2, 1, 3, 3, 6, p), None));
    }
    v.push(("minimal polynomial of ∛3+⁴√2 (n=13, 600 bits)".into(), powers_of_radicals(3, 3, 1, 2, 4, 12, 600), None));
    v.push(("minimal polynomial of ⁴√3−⁴√2 (n=17, 400 bits)".into(), powers_of_radicals(3, 4, -1, 2, 4, 16, 400), None));
    v.push(("minimal polynomial of ⁵√2+⁷√3 (n=36, 1000 bits)".into(), powers_of_radicals(2, 5, 1, 3, 7, 35, 1000), None));
    {
        let p = 200;
        let at = |q: u32| rounded(p, |w| (BigFloat::from(1).with_prec(w) / q).atan());
        v.push(("Machin π/4, atan 1/5, atan 1/239 (n=3, 200 bits)".into(), vec![rounded(p, |w| BigFloat::pi(w) / 4u32), at(5), at(239)], None));
        let mut x = vec![rounded(p, |w| BigFloat::pi(w) / 4u32)];
        x.extend([49, 57, 239, 110443].map(at));
        v.push(("four-term Machin (n=5, 200 bits)".into(), x, None));
    }
    {
        let p = 300;
        let ln = |k: u32, w: u64| BigFloat::from(k).with_prec(w).ln();
        let mut x: Vec<BigFloat> = [2u32, 3, 5, 7].iter().map(|&q| rounded(p, |w| ln(q, w))).collect();
        x.push(rounded(p, |w| ln(2, w) * 31u32 - ln(3, w) * 17u32 + ln(5, w) * 11u32 + ln(7, w) * 4u32));
        v.push(("logarithms ln 2, ln 3, ln 5, ln 7 and a combination (n=5, 300 bits)".into(), x, None));
    }
    v.push((
        "no relation below 10^12 among 1, π, ζ(3) (n=3, 300 bits)".into(),
        vec![one(300), rounded(300, BigFloat::pi), rounded(300, zeta3)],
        pow10(12),
    ));
    v.push((
        "no relation below 10^8 among 1, π, e, ln 2, γ, G (n=6, 400 bits)".into(),
        vec![
            one(400),
            rounded(400, BigFloat::pi),
            rounded(400, BigFloat::e),
            rounded(400, BigFloat::ln2),
            rounded(400, BigFloat::euler_gamma),
            rounded(400, BigFloat::catalan),
        ],
        pow10(8),
    ));
    v.push((
        "no polynomial below 10^6 for γ, degree 6 (n=7, 700 bits)".into(),
        (0..7).map(|k| rounded(700, |w| BigFloat::euler_gamma(w).powi(k))).collect(),
        pow10(6),
    ));
    v
}

fn relation_text(r: &Pslq) -> String {
    match r {
        Pslq::Relation(m) => m.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
        _ => "None".into(),
    }
}

/// The same relation up to sign.
fn same(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let neg: Vec<String> = b
        .split_whitespace()
        .map(|t| if let Some(s) = t.strip_prefix('-') { s.to_string() } else if t == "0" { t.to_string() } else { format!("-{t}") })
        .collect();
    a == neg.join(" ")
}

pub fn pslq(ctx: &Ctx) {
    let mp = RefCell::new(Mpmath::start());
    if let Some(m) = mp.borrow().as_ref() {
        eprintln!("{}", m.banner);
    }
    for (name, x, max_norm) in cases() {
        if !ctx.wants("pslq", &name) {
            continue;
        }
        let opts = PslqOptions::new().with_max_norm(max_norm.clone());
        let ours = a_pslq(&x, &opts);
        let ours_t = relation_text(&ours);
        let parts: Vec<(Number, i64)> = x.iter().map(|v| v.mantissa_exponent().unwrap()).collect();
        let precs: Vec<u64> = x.iter().map(|v| v.prec()).collect();
        let prec = *precs.iter().max().unwrap();
        let mut note = format!(
            "ours: {}",
            match &ours {
                Pslq::Relation(_) => format!("relation [{ours_t}]"),
                other => format!("no relation of norm below {:.4e}", other.bound().map_or(f64::NAN, |b| b.to_f64())),
            }
        );
        #[cfg(has_pari)]
        let pa = refs::pari::reals(&parts.iter().map(|(m, _)| m.to_string()).collect::<Vec<_>>(), &parts.iter().map(|&(_, e)| e).collect::<Vec<_>>(), &precs);
        let mut refs_v: Vec<Ref> = Vec::new();
        // mpmath: maxcoeff is a bound on max |c|, ours on the Euclidean norm.
        let big = deg35_skip(&name);
        if mp.borrow().is_some() && !big {
            let maxcoeff = max_norm.clone().unwrap_or_else(|| Number::from(1) << prec as usize);
            mp.borrow_mut().as_mut().unwrap().case(&parts, prec, &maxcoeff, 1_000_000);
            let r = mp.borrow_mut().as_mut().unwrap().cmd("check");
            let banner = mp.borrow().as_ref().unwrap().banner.clone();
            let agree = same(&r, &ours_t);
            refs_v.push(Ref::new(
                format!("{banner} pslq (co-process)"),
                format!(
                    "maxcoeff {} (∞-norm), tol 2^(-3/4·prec), maxsteps 10^6; pure-Python big integers; a pipe round trip (16 µs median, measured) in its time; same answer: {agree}{}",
                    if max_norm.is_some() { "= max_norm" } else { "= 2^prec" },
                    if agree { String::new() } else { format!(" (mpmath [{r}])") }
                ),
                |_| {
                    black_box(mp.borrow_mut().as_mut().unwrap().cmd("run 1"));
                },
            ));
        }
        #[cfg(has_pari)]
        if max_norm.is_none() {
            let r = refs::pari::lindep(&pa, true).unwrap_or_default();
            let r2: String = r.trim_matches(|c| c == '[' || c == ']' || c == '~').split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>().join(" ");
            let agree = same(&r2, &ours_t);
            refs_v.push(Ref::new(
                "PARI 2.19 lindep (LLL)",
                format!("an LLL method, no bound proved, precision rounded up to words; same answer: {agree}{}", if agree { String::new() } else { format!(" (PARI [{r2}])") }),
                |_| {
                    black_box(refs::pari::lindep(&pa, false));
                },
            ));
        }
        ctx.row(
            "pslq",
            "pslq",
            &name,
            Side::new("ours", |_| {
                black_box(a_pslq(&x, &opts));
            }),
            refs_v,
            &note,
        );
        note.clear();
    }

    // pslq_f64.
    let f64_cases: Vec<(&str, Vec<f64>, Option<Number>)> = vec![
        ("Machin π/4, atan 1/5, atan 1/239 (f64)", vec![std::f64::consts::FRAC_PI_4, (0.2f64).atan(), (1.0f64 / 239.0).atan()], None),
        ("ln 2, ln 3, ln 6 (f64)", vec![2f64.ln(), 3f64.ln(), 6f64.ln()], None),
        ("1, φ, φ² (f64)", {
            let phi = (1.0 + 5f64.sqrt()) / 2.0;
            vec![1.0, phi, phi * phi]
        }, None),
        ("no relation below 100 between π, e, 1 (f64)", vec![std::f64::consts::PI, std::f64::consts::E, 1.0], Some(Number::from(100))),
    ];
    for (name, x, max_norm) in f64_cases {
        if !ctx.wants("pslq_f64", name) {
            continue;
        }
        let opts = PslqOptions::new().with_max_norm(max_norm.clone());
        let ours = pslq_f64(&x, &opts);
        let ours_t = relation_text(&ours);
        let bf: Vec<BigFloat> = x.iter().map(|&v| BigFloat::from(v)).collect();
        let parts: Vec<(Number, i64)> = bf.iter().map(|v| v.mantissa_exponent().unwrap()).collect();
        #[cfg(has_pari)]
        let pa = refs::pari::reals(&parts.iter().map(|(m, _)| m.to_string()).collect::<Vec<_>>(), &parts.iter().map(|&(_, e)| e).collect::<Vec<_>>(), &vec![53; parts.len()]);
        let mut refs_v: Vec<Ref> = Vec::new();
        if mp.borrow().is_some() {
            let maxcoeff = max_norm.clone().unwrap_or_else(|| Number::from(1) << 53usize);
            mp.borrow_mut().as_mut().unwrap().case(&parts, 53, &maxcoeff, 1_000_000);
            let r = mp.borrow_mut().as_mut().unwrap().cmd("check");
            let agree = same(&r, &ours_t);
            let banner = mp.borrow().as_ref().unwrap().banner.clone();
            refs_v.push(Ref::new(
                format!("{banner} pslq (co-process)"),
                format!("prec 53; a pipe round trip (16 µs median, measured) included; same answer: {agree}{}", if agree { String::new() } else { format!(" (mpmath [{r}])") }),
                |_| {
                    black_box(mp.borrow_mut().as_mut().unwrap().cmd("run 1"));
                },
            ));
        }
        #[cfg(has_pari)]
        if max_norm.is_none() {
            let r = refs::pari::lindep(&pa, true).unwrap_or_default();
            let r2: String = r.trim_matches(|c| c == '[' || c == ']' || c == '~').split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>().join(" ");
            let agree = same(&r2, &ours_t);
            refs_v.push(Ref::new("PARI 2.19 lindep (LLL)", format!("64-bit reals; same answer: {agree}{}", if agree { String::new() } else { format!(" (PARI [{r2}])") }), |_| {
                black_box(refs::pari::lindep(&pa, false));
            }));
        }
        ctx.row(
            "pslq",
            "pslq_f64",
            name,
            Side::new("ours", |_| {
                black_box(pslq_f64(&x, &opts));
            }),
            refs_v,
            &format!("ours: {}", if ours_t == "None" { "no relation (bound)".to_string() } else { format!("relation [{ours_t}]") }),
        );
    }
}

/// mpmath on the degree-35 case takes minutes a run: timed once by `run.sh` apart, not here.
fn deg35_skip(name: &str) -> bool {
    name.contains("n=36") && std::env::var("LATBENCH_MPMATH_DEG35").is_err()
}
