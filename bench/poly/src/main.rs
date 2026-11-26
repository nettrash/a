//! Polynomials (a::math::poly) against FLINT, NTL, PARI, LAPACK and Rust FFT crates: the v0.2
//! speed-pass scoreboard. Usage: polybench [--out FILE] [GROUP ...]; it appends TSV rows
//! (without the header). Groups: wmul wdiv mlimb zarith qarith farith gcd eval roots factor.
//! POLYBENCH_QUICK=<maxlen> makes a smoke run (3 samples, sizes above maxlen skipped).

mod core;
mod eval;
mod factor;
mod fuzz;
mod gcd;
mod m;
mod roots;
mod w;
mod z;

use crate::core::Out;
use std::fs::OpenOptions;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("fuzz") {
        let seed = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
        let count = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
        let filter = args.get(3).cloned().unwrap_or_default();
        fuzz::fuzz(seed, count, &filter);
        return;
    }
    let mut path = None;
    if let Some(i) = args.iter().position(|a| a == "--out") {
        path = Some(args[i + 1].clone());
        args.drain(i..i + 2);
    }
    let w: Box<dyn std::io::Write> = match path {
        Some(p) => Box::new(OpenOptions::new().create(true).append(true).open(p).unwrap()),
        None => Box::new(std::io::sink()),
    };
    let quick = std::env::var("POLYBENCH_QUICK").ok().and_then(|q| q.parse().ok());
    let mut out = Out { w, filter: args, quick };
    w::wmul(&mut out);
    w::wdiv(&mut out);
    m::mlimb(&mut out);
    z::zarith(&mut out);
    z::qarith(&mut out);
    z::farith(&mut out);
    gcd::gcd(&mut out);
    eval::eval(&mut out);
    roots::roots(&mut out);
    factor::factor(&mut out);
}
