//! The constants π, e, ln 2, ln 10, γ and Catalan's G: the first call in a fresh process (each
//! sample a new process, the libraries alternating), and a later call with the value cached.
use crate::ffi::*;
use crate::h::*;
use crate::libs::*;
use a::math::float::BigFloat;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const KP: [u64; 8] = [53, 64, 113, 256, 1000, 10_000, 100_000, 1_000_000];
pub const NAMES: [&str; 6] = ["pi", "e", "ln2", "ln10", "euler_gamma", "catalan"];

pub fn digest(c: &Canon) -> String {
    match c {
        Canon::Fin(n, l, e) => {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for w in l {
                h ^= *w;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            format!("{}{}:{}:{:x}", if *n { "-" } else { "" }, l.len(), e, h)
        }
        o => format!("{o:?}"),
    }
}

fn mpfr_const(name: &str, r: &mut MF) {
    unsafe {
        match name {
            "pi" => mpfr_const_pi(r.p(), RN),
            "e" => {
                let one = {
                    let mut o = MF::new(2);
                    mpfr_set_ui(o.p(), 1, RN);
                    o
                };
                mpfr_exp(r.p(), one.c(), RN)
            }
            "ln2" => mpfr_const_log2(r.p(), RN),
            "ln10" => {
                let mut t = MF::new(8);
                mpfr_set_ui(t.p(), 10, RN);
                mpfr_log(r.p(), t.c(), RN)
            }
            "euler_gamma" => mpfr_const_euler(r.p(), RN),
            _ => mpfr_const_catalan(r.p(), RN),
        };
    }
}

/// The forms into an existing number (added in the speed pass; no row before it).
fn ours_set(name: &str, x: &mut BigFloat) {
    let _ = match name {
        "pi" => x.set_pi(N),
        "e" => x.set_e(N),
        "ln2" => x.set_ln2(N),
        "ln10" => x.set_ln10(N),
        "euler_gamma" => x.set_euler_gamma(N),
        _ => x.set_catalan(N),
    };
}

fn ours_const(name: &str, p: u64) -> BigFloat {
    match name {
        "pi" => BigFloat::pi_round(p, N).0,
        "e" => BigFloat::e_round(p, N).0,
        "ln2" => BigFloat::ln2_round(p, N).0,
        "ln10" => BigFloat::ln10_round(p, N).0,
        "euler_gamma" => BigFloat::euler_gamma_round(p, N).0,
        _ => BigFloat::catalan_round(p, N).0,
    }
}

fn mal_const(name: &str, p: u64) -> mal::MF2 {
    use mal::{MF2, RM};
    match name {
        "pi" => MF2::pi_prec_round(p, RM).0,
        "e" => MF2::e_prec_round(p, RM).0,
        "ln2" => MF2::ln_2_prec_round(p, RM).0,
        "ln10" => MF2::ln_10_prec_round(p, RM).0,
        "euler_gamma" => MF2::eulers_constant_prec_round(p, RM).0,
        _ => MF2::catalans_constant_prec_round(p, RM).0,
    }
}

fn astro_const(name: &str, cc: &mut ast::Consts, p: u64) -> Option<ast::AF> {
    let w = ast::wp(p) as usize;
    Some(match name {
        "pi" => cc.pi(w, ast::RM),
        "e" => cc.e(w, ast::RM),
        "ln2" => cc.ln_2(w, ast::RM),
        "ln10" => cc.ln_10(w, ast::RM),
        _ => return None,
    })
}

fn dashu_const(name: &str, cache: Option<&mut dashu_float::ConstCache>, p: u64) -> Option<dsh::DF> {
    let cx = dsh::ctx(p);
    Some(match name {
        "pi" => cx.pi::<2>(cache).value(),
        "e" => cx.e::<2>().value(),
        "ln2" => match cache {
            Some(c) => c.ln2::<2, dsh::HalfEven>(p as usize),
            None => dashu_float::ConstCache::new().ln2::<2, dsh::HalfEven>(p as usize),
        },
        "ln10" => match cache {
            Some(c) => c.ln10::<2, dsh::HalfEven>(p as usize),
            None => dashu_float::ConstCache::new().ln10::<2, dsh::HalfEven>(p as usize),
        },
        _ => return None,
    })
}

#[cfg(has_arb)]
fn arb_const(name: &str, r: &mut arb::AB, p: u64) {
    let ap = crate::g_func::arb_prec(p);
    unsafe {
        match name {
            "pi" => arb::arb_const_pi(r.p(), ap),
            "e" => arb::arb_const_e(r.p(), ap),
            "ln2" => arb::arb_const_log2(r.p(), ap),
            "ln10" => arb::arb_const_log10(r.p(), ap),
            "euler_gamma" => arb::arb_const_euler(r.p(), ap),
            _ => arb::arb_const_catalan(r.p(), ap),
        }
    }
}

const LIBS: [&str; 6] = ["a", "MPFR", "malachite-float", "astro-float", "dashu-float", "Arb"];

fn lib_has(lib: &str, name: &str) -> bool {
    match lib {
        "astro-float" | "dashu-float" => matches!(name, "pi" | "e" | "ln2" | "ln10"),
        "Arb" => cfg!(has_arb),
        _ => true,
    }
}

/// In the child: one call, timed, then the digest of the result.
pub fn once(lib: &str, name: &str, p: u64) {
    let (ns, d) = match lib {
        "a" => {
            let mut v = None;
            let ns = once_t(|| v = Some(ours_const(name, p)));
            (ns, digest(&c_ours(&v.unwrap())))
        }
        "MPFR" => {
            let mut r = MF::new(p);
            let ns = once_t(|| mpfr_const(name, &mut r));
            (ns, digest(&c_mpfr(r.c())))
        }
        "malachite-float" => {
            let mut v = None;
            let ns = once_t(|| v = Some(mal_const(name, p)));
            (ns, digest(&mal::c(&v.unwrap())))
        }
        "astro-float" => {
            let mut v = None;
            let ns = once_t(|| {
                let mut cc = ast::Consts::new().unwrap();
                v = astro_const(name, &mut cc, p);
            });
            // astro rounds to whole words: compare with ours at that precision
            (ns, digest(&ast::c(&v.unwrap())))
        }
        "dashu-float" => {
            let mut v = None;
            let ns = once_t(|| v = dashu_const(name, None, p));
            (ns, digest(&dsh::c(&v.unwrap())))
        }
        #[cfg(has_arb)]
        "Arb" => {
            let mut r = arb::AB::new();
            let ns = once_t(|| arb_const(name, &mut r, p));
            (ns, digest(&c_mpfr(r.mid(p).c())))
        }
        _ => panic!("lib"),
    };
    println!("{ns}\t{d}");
}

fn once_t<F: FnOnce()>(f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e9
}

fn spawn_once(lib: &str, name: &str, p: u64, limit: Duration) -> Option<(f64, String)> {
    let exe = std::env::current_exe().unwrap();
    let mut ch = Command::new(exe)
        .args(["once", lib, name, &p.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .ok()?;
    let t0 = Instant::now();
    loop {
        match ch.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if t0.elapsed() > limit {
                    let _ = ch.kill();
                    let _ = ch.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(if t0.elapsed() < Duration::from_millis(50) { 1 } else { 20 }));
            }
            Err(_) => return None,
        }
    }
    let out = ch.wait_with_output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.trim().split('\t');
    let ns: f64 = it.next()?.parse().ok()?;
    Some((ns, it.next().unwrap_or("").to_string()))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

pub fn first_rows() {
    let limit = Duration::from_secs(40);
    for name in NAMES {
        let mut dead: Vec<bool> = vec![false; LIBS.len()];
        let mut last: Vec<Option<(f64, f64)>> = vec![None; LIBS.len()];
        for p in KP {
            // a library whose call here would take more than the limit, from its last time
            for i in 1..LIBS.len() {
                if let Some((p0, t0)) = last[i] {
                    if t0 * 1e-9 * (p as f64 / p0).powf(1.7) > limit.as_secs_f64() {
                        dead[i] = true;
                    }
                }
            }
            let op = format!("{name}_round (first call)");
            let ps = format!("prec={p}");
            if !wanted("consts-first", &op, &ps) {
                continue;
            }
            // probe: one sample each, to choose the number of samples
            let mut t: Vec<Vec<f64>> = vec![Vec::new(); LIBS.len()];
            let mut dg: Vec<String> = vec![String::new(); LIBS.len()];
            for (i, lib) in LIBS.iter().enumerate() {
                if !lib_has(lib, name) || dead[i] {
                    continue;
                }
                match spawn_once(lib, name, p, limit) {
                    Some((ns, d)) => {
                        t[i].push(ns);
                        dg[i] = d;
                    }
                    None => dead[i] = true,
                }
            }
            let slow = t.iter().filter_map(|v| v.first()).cloned().fold(0.0f64, f64::max);
            let samples = if cfg().quick {
                2
            } else if slow > 5e9 {
                3
            } else if slow > 5e8 {
                5
            } else {
                cfg().samples
            };
            for s in 1..samples {
                for j in 0..LIBS.len() {
                    let i = (s + j) % LIBS.len();
                    if t[i].is_empty() {
                        continue;
                    }
                    if let Some((ns, _)) = spawn_once(LIBS[i], name, p, limit) {
                        t[i].push(ns);
                    }
                }
            }
            if t[0].is_empty() {
                continue;
            }
            let ours = median(&mut t[0].clone());
            for i in 0..LIBS.len() {
                if !t[i].is_empty() {
                    last[i] = Some((p as f64, median(&mut t[i].clone())));
                }
            }
            // astro compares with ours at its word precision
            let ours_w = if ast::wp(p) != p && !t[3].is_empty() {
                let w = ast::wp(p);
                spawn_once("a", name, w, limit).map(|(_, d)| d)
            } else {
                None
            };
            for i in 1..LIBS.len() {
                if !lib_has(LIBS[i], name) {
                    continue;
                }
                let refn = match LIBS[i] {
                    "MPFR" => match name {
                        "e" => "MPFR mpfr_exp(1)",
                        "ln10" => "MPFR mpfr_log(10)",
                        _ => "MPFR mpfr_const_*",
                    },
                    "Arb" => "Arb (ball, does less)",
                    o => o,
                };
                let mut notes = vec!["each sample a fresh process".to_string()];
                if samples != cfg().samples {
                    notes.push(format!("{samples} samples"));
                }
                if t[i].is_empty() {
                    notes.push(if dead[i] { "not timed: one call > 40 s here or predicted so from a smaller precision (t·(p/p0)^1.7)".into() } else { "not run".into() });
                    println!("float\tconsts-first\t{op}\t{ps}\t{}\t{refn}\t\t\t{}", fmt_ns(ours), notes.join("; "));
                    continue;
                }
                let r = median(&mut t[i].clone());
                let want = if LIBS[i] == "astro-float" { ours_w.clone().unwrap_or(dg[0].clone()) } else { dg[0].clone() };
                if LIBS[i] == "Arb" {
                    notes.push(format!("does less: a ball at {} bits", crate::g_func::arb_prec(p)));
                    if dg[i] != want {
                        notes.push("midpoint differs from the rounded value".into());
                    }
                } else if LIBS[i] == "astro-float" && ast::wp(p) != p {
                    notes.push(format!("astro at whole words, {} bits", ast::wp(p)));
                }
                if LIBS[i] != "Arb" {
                    if dg[i] == want {
                        notes.push("outputs equal".into());
                    } else {
                        notes.push(format!("MISMATCH ({} vs {})", want, dg[i]));
                        eprintln!("MISMATCH consts-first/{op}/{ps} {}: {} vs {}", LIBS[i], want, dg[i]);
                    }
                }
                println!(
                    "float\tconsts-first\t{op}\t{ps}\t{}\t{refn}\t{}\t{}\t{}",
                    fmt_ns(ours),
                    fmt_ns(r),
                    sig3(ours / r),
                    notes.join("; ")
                );
            }
        }
    }
}

pub fn cached_rows() {
    for name in NAMES {
        for p in KP {
            let op = format!("{name}_round (cached)");
            let ps = format!("prec={p}");
            if !wanted("consts-cached", &op, &ps) {
                continue;
            }
            let mut conts: Vec<Cont> = Vec::new();
            let _ = ours_const(name, p);
            conts.push(Cont::ours(kern!(1, |k| ours_const(name, p), |v| c_ours(&v))));
            {
                let mut r = MF::new(p);
                mpfr_const(name, &mut r);
                let nm = match name {
                    "e" => "MPFR mpfr_exp(1) (no cache)",
                    "ln10" => "MPFR mpfr_log(10) (no cache)",
                    _ => "MPFR mpfr_const_* (cached)",
                };
                conts.push(Cont::rf(nm, kern!(1, |k| mpfr_const(name, &mut r), |v| c_mpfr(r.c()))).note("MPFR into an existing variable"));
            }
            {
                let _ = mal_const(name, p);
                conts.push(Cont::rf("malachite-float", kern!(1, |k| mal_const(name, p), |v| mal::c(&v))));
            }
            if lib_has("astro-float", name) {
                let w = ast::wp(p);
                let tag = if w != p { "word-precision" } else { "" };
                if !tag.is_empty() {
                    let _ = ours_const(name, w);
                    conts.push(Cont::ours_tag(tag, kern!(1, |k| ours_const(name, w), |v| c_ours(&v))));
                }
                let mut cc = ast::Consts::new().unwrap();
                let _ = astro_const(name, &mut cc, p);
                conts.push(Cont::rf("astro-float Consts", kern!(1, |k| astro_const(name, &mut cc, p).unwrap(), |v| ast::c(&v))).vs(tag));
            }
            if lib_has("dashu-float", name) {
                let mut cache = dashu_float::ConstCache::new();
                let _ = dashu_const(name, Some(&mut cache), p);
                let mut ct = Cont::rf("dashu-float ConstCache", kern!(1, |k| dashu_const(name, Some(&mut cache), p).unwrap(), |v| dsh::c(&v)));
                if name == "e" {
                    ct = ct.note("dashu's e has no cache");
                }
                conts.push(ct);
            }
            #[cfg(has_arb)]
            {
                let mut r = arb::AB::new();
                arb_const(name, &mut r, p);
                conts.push(
                    Cont::rf("Arb (ball, does less)", kern!(1, |k| arb_const(name, &mut r, p), |v| c_mpfr(r.mid(p).c())))
                        .loose()
                        .note("does less: a ball at the guard bits; into an existing variable"),
                );
            }
            row("consts-cached", &op, &ps, 1, conts);
            if std::env::var("FLOATBENCH_SET_CONSTS").is_ok() {
                // `set_pi`-style forms into an existing number, against MPFR's into a variable.
                let sop = format!("set_{name} (cached)");
                let mut x = BigFloat::zero(p);
                ours_set(name, &mut x);
                let mut sc = vec![Cont::ours(kern!(1, |k| ours_set(name, &mut x), |v| c_ours(&x)))];
                let mut r = MF::new(p);
                mpfr_const(name, &mut r);
                let nm = match name {
                    "e" => "MPFR mpfr_exp(1) (no cache)",
                    "ln10" => "MPFR mpfr_log(10) (no cache)",
                    _ => "MPFR mpfr_const_* (cached)",
                };
                sc.push(Cont::rf(nm, kern!(1, |k| mpfr_const(name, &mut r), |v| c_mpfr(r.c()))).note("MPFR into an existing variable"));
                row("consts-cached", &sop, &ps, 1, sc);
            }
        }
    }
}

pub fn run(g: &str) {
    match g {
        "first" => first_rows(),
        "cached" => cached_rows(),
        _ => panic!("consts: first|cached"),
    }
}
