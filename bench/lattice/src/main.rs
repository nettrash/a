//! The scoreboard harness of the area `lattice` (issue #23): lattices, normal forms, exact
//! linear algebra (Dixon, multimodular, Strassen), matrices over GF(p) and GF(2), against
//! fplll, FLINT, NTL, M4RI, PARI/GP and mpmath.
//!
//! `latbench <group>...` prints TSV rows (no header) to stdout, progress to stderr. Groups:
//! lll, deep, bkz, enum, pslq, normal, exact, modp, gf2. `LATBENCH_QUICK=1` for a smoke run,
//! `LATBENCH_FILTER=<text>` to keep the rows whose operation or params contain it.
mod exact;
mod inputs;
mod gf2;
mod lattice;
mod modp;
mod pslq;
mod refs;
mod timing;

use std::io::Write;
use timing::{Plan, Side, measure};

pub struct Ref<'a> {
    pub name: String,
    pub side: Side<'a>,
    pub note: String,
}

impl<'a> Ref<'a> {
    pub fn new(name: impl Into<String>, note: impl Into<String>, f: impl FnMut(usize) + 'a) -> Self {
        let name = name.into();
        Ref {
            side: Side::new(name.clone(), f),
            name,
            note: note.into(),
        }
    }
}

/// The versions written into the reference names ("FLINT 3.6 …") as found at run time.
fn versions() -> &'static Vec<(String, String)> {
    static V: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        #[allow(unused_mut)]
        let mut v: Vec<(String, String)> = Vec::new();
        let short = |s: String| s.split('.').take(2).collect::<Vec<_>>().join(".");
        #[cfg(has_flint)]
        v.push(("FLINT 3.6".into(), format!("FLINT {}", short(refs::flint::version()))));
        #[cfg(has_fplll)]
        v.push(("fplll 5.5".into(), format!("fplll {}", short(refs::fplll::version()))));
        #[cfg(has_ntl)]
        v.push(("NTL 11.6".into(), format!("NTL {}", short(refs::ntl::version()))));
        #[cfg(has_pari)]
        {
            let pv = refs::pari::version();
            let num = pv.split_whitespace().find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit())).unwrap_or("?").to_string();
            v.push(("PARI 2.19".into(), format!("PARI {}", short(num))));
        }
        #[cfg(has_m4ri)]
        if !env!("LATBENCH_M4RI_VERSION").is_empty() {
            v.push(("M4RI 20260122".into(), format!("M4RI {}", env!("LATBENCH_M4RI_VERSION"))));
        }
        v
    })
}

pub fn ref_name(name: &str) -> String {
    let mut s = name.to_string();
    for (from, to) in versions() {
        s = s.replace(from.as_str(), to);
    }
    s
}

pub fn sig3(x: f64) -> String {
    if !x.is_finite() || x <= 0.0 {
        return "-".into();
    }
    let d = (2 - x.log10().floor() as i32).max(0) as usize;
    format!("{:.*}", d, x)
}

fn ns(x: f64) -> String {
    if x >= 100.0 { format!("{:.0}", x) } else { format!("{:.1}", x) }
}

pub struct Ctx {
    pub plan: Plan,
    pub filter: Option<String>,
}

impl Ctx {
    pub fn wants(&self, op: &str, params: &str) -> bool {
        match &self.filter {
            None => true,
            Some(f) => f.split(',').any(|f| op.contains(f) || params.contains(f)),
        }
    }

    /// Times ours against each reference, interleaved, and prints one row per reference.
    pub fn row(&self, group: &str, op: &str, params: &str, ours: Side<'_>, refs: Vec<Ref<'_>>, notes: &str) {
        if !self.wants(op, params) {
            return;
        }
        eprintln!("[{group}] {op} {params} ...");
        let mut sides = vec![ours];
        let mut names = Vec::new();
        let mut rnotes = Vec::new();
        for r in refs {
            names.push(r.name);
            rnotes.push(r.note);
            sides.push(r.side);
        }
        let t = measure(&self.plan, &mut sides);
        let ours_ns = t.median[0];
        let mut common = String::new();
        if t.samples < 15 {
            common.push_str(&format!("{} samples; ", t.samples));
        }
        if t.iqr[0] > 0.1 {
            common.push_str(&format!("ours IQR {:.0}%; ", 100.0 * t.iqr[0]));
        }
        let mut out = std::io::stdout().lock();
        if names.is_empty() {
            let n = format!("{common}{notes}");
            writeln!(out, "lattice\t{group}\t{op}\t{params}\t{}\t(none)\t-\t-\t{}", ns(ours_ns), n.trim_end_matches("; ")).unwrap();
        }
        for (i, name) in names.iter().enumerate() {
            let name = ref_name(name);
            if let Some(msg) = &t.failed[i + 1] {
                let msg: String = msg.chars().take(160).collect();
                writeln!(out, "lattice\t{group}\t{op}\t{params}\t{}\t{name}\t-\t-\tthe reference failed, not timed: {msg}", ns(ours_ns)).unwrap();
                continue;
            }
            let r = t.median[i + 1];
            let mut n = common.clone();
            if t.iqr[i + 1] > 0.1 {
                n.push_str(&format!("ref IQR {:.0}%; ", 100.0 * t.iqr[i + 1]));
            }
            if !rnotes[i].is_empty() {
                n.push_str(&rnotes[i]);
                n.push_str("; ");
            }
            n.push_str(notes);
            writeln!(
                out,
                "lattice\t{group}\t{op}\t{params}\t{}\t{name}\t{}\t{}\t{}",
                ns(ours_ns),
                ns(r),
                sig3(ours_ns / r),
                n.trim_end_matches("; ")
            )
            .unwrap();
        }
        out.flush().unwrap();
        eprintln!(
            "    ours {} ns; {}",
            ns(ours_ns),
            names.iter().enumerate().map(|(i, n)| format!("{n} {} ({})", ns(t.median[i + 1]), sig3(ours_ns / t.median[i + 1]))).collect::<Vec<_>>().join("; ")
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ctx = Ctx {
        plan: Plan::from_env(),
        filter: std::env::var("LATBENCH_FILTER").ok(),
    };
    if args.is_empty() || args[0] == "versions" {
        #[cfg(has_flint)]
        eprintln!("FLINT {}", refs::flint::version());
        #[cfg(has_fplll)]
        eprintln!("fplll {}", refs::fplll::version());
        #[cfg(has_ntl)]
        eprintln!("NTL {} (SP bits {})", refs::ntl::version(), refs::ntl::sp_nbits());
        #[cfg(has_m4ri)]
        eprintln!("M4RI {}", if env!("LATBENCH_M4RI_VERSION").is_empty() { "(version unknown)" } else { env!("LATBENCH_M4RI_VERSION") });
        #[cfg(has_pari)]
        eprintln!("PARI {}", refs::pari::version());
        return;
    }
    for g in &args {
        match g.as_str() {
            "lll" => lattice::lll(&ctx),
            "probe" => lattice::probe(&ctx),
            "lll-exact" => lattice::lll_exact(&ctx),
            "deep" => lattice::deep(&ctx),
            "isred" => lattice::is_reduced(&ctx),
            "bkz" => lattice::bkz(&ctx),
            "enum" => lattice::enumeration(&ctx),
            "pslq" => pslq::pslq(&ctx),
            "normal" => exact::normal(&ctx),
            "exact" => exact::exact(&ctx),
            "modp" => modp::modp(&ctx),
            "gf2" => gf2::gf2(&ctx),
            other => panic!("unknown group {other}"),
        }
    }
}
