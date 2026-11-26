//! special-bench: times a::math::special against its references; TSV rows on stdout.
//!
//! special-bench [--group g1,g2] [--filter text] [--samples n] [--ours-only] [--list]
//!   --ours-only   time ours alone, print "key\tns" lines (for the fat-LTO figure)
//!   --lto FILE    read such lines and add "ours with fat LTO" notes where they differ > 10 %
use sbench::*;
use std::collections::HashMap;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut groups: Option<Vec<String>> = None;
    let mut filter: Option<String> = None;
    let mut opts = Opts::default();
    let mut list = false;
    let mut lto: HashMap<String, f64> = HashMap::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--group" => {
                i += 1;
                groups = Some(args[i].split(',').map(String::from).collect());
            }
            "--filter" => {
                i += 1;
                filter = Some(args[i].clone());
            }
            "--samples" => {
                i += 1;
                opts.samples = args[i].parse().unwrap();
            }
            "--ours-only" => opts.ours_only = true,
            "--list" => list = true,
            "--lto" => {
                i += 1;
                for l in std::fs::read_to_string(&args[i]).unwrap_or_default().lines() {
                    if let Some((k, v)) = l.rsplit_once('\t') {
                        if let Ok(x) = v.parse::<f64>() {
                            lto.insert(k.to_string(), x);
                        }
                    }
                }
            }
            a => panic!("unknown argument {a}"),
        }
        i += 1;
    }
    std::panic::set_hook(Box::new(|_| {}));
    #[cfg(has_gsl)]
    {
        unsafe extern "C" {
            fn r_gsl_init();
        }
        unsafe { r_gsl_init() };
    }
    let sel = |g: &str, key: &str| {
        groups.as_ref().is_none_or(|gs| gs.iter().any(|x| x == g))
            && filter.as_ref().is_none_or(|f| key.contains(f.as_str()))
    };
    let out = std::io::stdout();
    let mut out = out.lock();
    let c64 = cases64();
    let c32 = cases32();
    if list {
        for c in &c64 {
            writeln!(out, "{}\t{} refs", c.key(), c.refs.len()).unwrap();
        }
        for c in &c32 {
            writeln!(out, "{}\t{} refs", c.key(), c.refs.len()).unwrap();
        }
        return;
    }
    fn go<T: Elem>(
        c: &Case<T>,
        opts: &Opts,
        lto: &HashMap<String, f64>,
        out: &mut impl Write,
    ) {
        let t0 = std::time::Instant::now();
        let r = run_case(c, opts);
        if opts.ours_only {
            writeln!(out, "{}\t{}", c.key(), r.ours.median).unwrap();
        } else {
            for row in rows(c, &r, lto.get(&c.key()).copied()) {
                writeln!(out, "{row}").unwrap();
            }
        }
        out.flush().unwrap();
        eprintln!("{:6.2}s {}", t0.elapsed().as_secs_f64(), c.key());
    }
    for c in &c64 {
        if sel(c.group, &c.key()) {
            go(c, &opts, &lto, &mut out);
        }
    }
    for c in &c32 {
        if sel(c.group, &c.key()) {
            go(c, &opts, &lto, &mut out);
        }
    }
}
