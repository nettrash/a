//! The fields scoreboard of the v0.2 speed pass: a::math::field against its references.
//!
//! `fieldbench [--header] [--samples N] <group>[:case,case…] …` prints the rows of the groups
//! (prime, ext, tower, binary (with the normal bases), gf2 (gf2_8, gf2_128, ghash)), tab-separated, in the scoreboard's columns.
//! Run it through bench/run.sh, which takes a measurement slot.

mod arkf;
mod binary;
mod common;
mod consts;
mod ext;
mod ffi;
mod gf2;
mod harness;
#[cfg(has_mcl)]
mod mcl;
mod prime;
mod tower;

fn main() {
    let mut groups: Vec<(String, Vec<String>)> = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--header" => println!("area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes"),
            "--check" => {
                // SAFETY: before any measurement, on the one thread.
                unsafe {
                    harness::CONFIG.check_only = true;
                }
            }
            "--samples" => {
                let n: usize = args.next().unwrap().parse().unwrap();
                // SAFETY: before any measurement, on the one thread.
                unsafe {
                    harness::CONFIG.samples = n.max(15);
                }
            }
            g => {
                let (name, cases) = g.split_once(':').unwrap_or((g, ""));
                let cases = cases.split(',').filter(|s| !s.is_empty()).map(String::from).collect();
                groups.push((name.to_string(), cases));
            }
        }
    }
    eprintln!(
        "fieldbench: C references built: {}; OpenSSL {}",
        env!("FIELDBENCH_CREFS"),
        ffi::openssl_version().unwrap_or_else(|| "-".into())
    );
    for (g, cases) in groups {
        match g.as_str() {
            "prime" => prime::run(&cases),
            "ext" => ext::run(&cases),
            "tower" => tower::run(&cases),
            "binary" => binary::run(&cases),
            "gf2" => gf2::run(&cases),
            other => eprintln!("fieldbench: unknown group {other}"),
        }
    }
}
