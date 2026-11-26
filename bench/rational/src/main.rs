// The scoreboard harness of a::math::rational against GMP, FLINT, PARI, malachite-q,
// num-rational and rug. Usage: ratbench <group>; writes TSV rows (no header) to stdout.
mod data;
mod ffi;
mod g_arith;
mod g_cf;
mod g_conv;
mod g_mat;
mod g_misc;
mod g_seq;
mod g_sum;
mod g_text;
mod harness;
mod libs;

pub const GROUPS: &[&str] = &[
    "arith", "inplace", "pool512", "mixed", "cmp", "new", "access", "round", "unary", "prim", "sum", "conv", "text", "cf", "approx",
    "between", "recon", "seq", "seq_long", "matrix", "matrank",
];

fn main() {
    let g = std::env::args().nth(1).unwrap_or_else(|| "list".into());
    match g.as_str() {
        "arith" | "inplace" | "pool512" | "mixed" | "cmp" => g_arith::run(&g),
        "new" | "round" | "unary" | "prim" => g_misc::run(&g),
        "sum" => g_sum::run(),
        "conv" => g_conv::run(),
        "text" => g_text::run(),
        "cf" | "approx" | "between" | "recon" => g_cf::run(&g),
        "seq" | "seq_long" => g_seq::run(&g),
        "matrix" => g_mat::run(),
        "matrank" => {
            g_mat::run_rank();
            g_mat::run_zdet();
        }
        "access" => g_misc::run_access(),
        "header" => println!("area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes"),
        "list" => println!("{}", GROUPS.join(" ")),
        _ => {
            eprintln!("unknown group {g}; groups: {}", GROUPS.join(" "));
            std::process::exit(2);
        }
    }
}
