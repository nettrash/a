//! The nt scoreboard: issue #15's functions of `a::math::numbers` against primesieve, primecount,
//! GMP, FLINT, PARI, GMP-ECM, msieve and Rust crates (primal, machine-prime, machine-factor,
//! malachite). `ntbench <task>` prints the rows of one task, tab-separated, on stdout; progress
//! on stderr. `ntbench list` lists the tasks. Environment: NT_SAMPLES (15), NT_BUDGET (seconds a
//! row, 15), NT_MIN_BATCH_MS (1), NT_FILTER (a substring of "operation params"), NT_LONG=1 (the
//! rows of minutes), NT_SCRATCH (msieve's working directory).
mod core;
mod ffi;
mod g_arith;
mod g_factor;
mod g_modular;
mod g_quad;
mod g_seq;
mod g_sieve;
mod g_prime;

type Task = (&'static str, &'static str, fn(&mut core::Ctx));

const TASKS: &[Task] = &[
    ("sieve", "sieve", g_sieve::sieve),
    ("pi", "sieve", g_sieve::pi),
    ("nth", "sieve", g_sieve::nth),
    ("isprime", "prime", g_prime::is_prime),
    ("nextprev", "prime", g_prime::next_prev),
    ("factorw", "factor", g_factor::factor_words),
    ("factor", "factor", g_factor::factor_long),
    ("factorsm", "factor", g_factor::factor_smooth),
    ("trial", "factor", g_factor::trial),
    ("rho", "factor", g_factor::rho),
    ("pm1pp1", "factor", g_factor::pm1_pp1),
    ("ecm", "factor", g_factor::ecm),
    ("ppow", "factor", g_factor::perfect_power),
    ("siqs", "siqs", g_factor::siqs),
    ("crt", "modular", g_modular::crt),
    ("order", "modular", g_modular::order_root),
    ("dlog", "modular", g_modular::discrete_log),
    ("dlogm", "modular", g_modular::dlog_methods),
    ("kron", "quadratic", g_quad::kronecker),
    ("sqrt", "quadratic", g_quad::sqrt),
    ("arith", "arith", g_arith::arith),
    ("fib", "sequences", g_seq::fib),
    ("lucmod", "sequences", g_seq::lucas_mod),
    ("primorial", "sequences", g_seq::primorial),
    ("part", "sequences", g_seq::partitions),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let task = args.get(1).map(String::as_str).unwrap_or("list");
    if task == "list" {
        for t in TASKS {
            println!("{}\t{}", t.0, t.1);
        }
        return;
    }
    if task == "header" {
        println!("area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes");
        return;
    }
    if task == "meta" {
        ffi::setup();
        println!("references: {}", ffi::versions().join(", "));
        println!("found by build.rs: {}", env!("NT_REFS_FOUND"));
        println!("clock: {:.2} GHz (a chain of dependent additions)", core::ghz());
        return;
    }
    let Some(t) = TASKS.iter().find(|t| t.0 == task) else {
        eprintln!("unknown task {task}; `ntbench list`");
        std::process::exit(2);
    };
    ffi::setup();
    let mut ctx = core::Ctx::new(t.1);
    // With NT_LONG=only, the tasks without long rows print nothing.
    if ctx.only_long && !["sieve", "pi", "siqs", "ecm", "dlog"].contains(&t.0) {
        return;
    }
    eprintln!("== {} ({}), clock {:.2} GHz", t.0, t.1, core::ghz());
    let t0 = std::time::Instant::now();
    (t.2)(&mut ctx);
    eprintln!("== {}: {} rows, {} check mismatches, {:.0} s", t.0, ctx.rows, ctx.mismatches, t0.elapsed().as_secs_f64());
}
