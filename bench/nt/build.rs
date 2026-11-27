// Finds the C references (GMP, FLINT, PARI, GMP-ECM, primesieve, primecount) by pkg-config or in
// the standard prefixes, compiles a shim for each one found (cc -O3, -mcpu=native on aarch64,
// -march=native on x86_64) into one static library, and sets cfg flags `has_<name>` for the
// harness. A reference that is not found is left out, with a warning; the harness then times
// our side only for its rows.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Lib {
    cfg: &'static str,
    header: &'static str,
    pkg: &'static str,
    link: &'static [&'static str],
    shim: &'static str,
    needs: &'static [&'static str],
}

const LIBS: &[Lib] = &[
    Lib { cfg: "gmp", header: "gmp.h", pkg: "gmp", link: &["gmp"], shim: "refs_gmp.c", needs: &[] },
    Lib { cfg: "flint", header: "flint/flint.h", pkg: "flint", link: &["flint"], shim: "refs_flint.c", needs: &["gmp"] },
    Lib { cfg: "pari", header: "pari/pari.h", pkg: "", link: &["pari"], shim: "refs_pari.c", needs: &["gmp"] },
    Lib { cfg: "ecm", header: "ecm.h", pkg: "", link: &["ecm"], shim: "refs_ecm.c", needs: &["gmp"] },
    Lib { cfg: "primesieve", header: "primesieve.h", pkg: "primesieve", link: &["primesieve", "primecount"], shim: "refs_ps.c", needs: &[] },
];

/// pkg-config's compile flags and `-L` directories of PKG, when it has a .pc file.
fn pkg_config(pkg: &str) -> Option<(Vec<String>, Vec<String>)> {
    if pkg.is_empty() {
        return None;
    }
    let c = Command::new("pkg-config").args(["--cflags", pkg]).output().ok()?;
    let l = Command::new("pkg-config").args(["--libs-only-L", pkg]).output().ok()?;
    if !c.status.success() || !l.status.success() {
        return None;
    }
    let split = |v: Vec<u8>| String::from_utf8_lossy(&v).split_whitespace().map(str::to_string).collect::<Vec<_>>();
    Some((split(c.stdout), split(l.stdout)))
}

/// The include directories of a prefix: its include/ and, on Debian and Ubuntu, the multiarch
/// one (gmp.h and pari/ live in /usr/include/x86_64-linux-gnu there).
fn include_dirs(p: &Path, multiarch: &str) -> Vec<PathBuf> {
    let mut v = vec![p.join("include")];
    if !multiarch.is_empty() {
        v.push(p.join("include").join(multiarch));
    }
    v
}

/// The library directories of a prefix: lib/, the multiarch lib/<triple>/ and lib64/.
fn lib_dirs(p: &Path, multiarch: &str) -> Vec<PathBuf> {
    let mut v = vec![p.join("lib")];
    if !multiarch.is_empty() {
        v.push(p.join("lib").join(multiarch));
    }
    v.push(p.join("lib64"));
    v
}

fn has_lib(d: &Path, name: &str) -> bool {
    ["so", "dylib", "a"].iter().any(|e| d.join(format!("lib{name}.{e}")).exists())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=NT_PREFIXES");
    println!("cargo:rerun-if-env-changed=NT_SKIP");
    // NT_SKIP=gmp,pari,...: leaves those references out on purpose (as if not installed).
    let skip: Vec<String> = env::var("NT_SKIP").unwrap_or_default().split(',').map(str::to_string).collect();
    for l in LIBS {
        println!("cargo::rustc-check-cfg=cfg(has_{})", l.cfg);
    }
    println!("cargo::rustc-check-cfg=cfg(has_primecount)");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let mut prefixes: Vec<PathBuf> = env::var("NT_PREFIXES")
        .map(|s| s.split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default();
    for p in ["/opt/homebrew", "/usr/local", "/usr"] {
        prefixes.push(PathBuf::from(p));
    }
    let multiarch = match arch.as_str() {
        "x86_64" => "x86_64-linux-gnu",
        "aarch64" => "aarch64-linux-gnu",
        _ => "",
    };
    let cpu = if arch == "aarch64" { "-mcpu=native" } else { "-march=native" };
    let mut found: Vec<&str> = Vec::new();
    let mut objects = Vec::new();
    for l in LIBS {
        if skip.iter().any(|x| x == l.cfg) {
            println!("cargo:warning=nt bench: {} left out (NT_SKIP)", l.cfg);
            continue;
        }
        if !l.needs.iter().all(|n| found.contains(n)) {
            println!("cargo:warning=nt bench: {} needs {:?}, left out", l.cfg, l.needs);
            continue;
        }
        // pkg-config first (its directories may be the compiler's defaults, then it prints no
        // -I or -L), then the prefixes with their multiarch directories.
        let pc = pkg_config(l.pkg);
        let (mut cflags, mut ldirs) = pc.unwrap_or_default();
        let mut inc_dirs: Vec<PathBuf> = cflags.iter().filter_map(|f| f.strip_prefix("-I")).map(PathBuf::from).collect();
        let mut lib_dirs_all: Vec<PathBuf> = ldirs.iter().filter_map(|f| f.strip_prefix("-L")).map(PathBuf::from).collect();
        for p in &prefixes {
            inc_dirs.extend(include_dirs(p, multiarch));
            lib_dirs_all.extend(lib_dirs(p, multiarch));
        }
        let header_dir = inc_dirs.iter().find(|d| d.join(l.header).exists()).cloned();
        let lib_dir = lib_dirs_all.iter().find(|d| has_lib(d, l.link[0])).cloned();
        let (Some(hd), Some(ld)) = (header_dir, lib_dir) else {
            println!("cargo:warning=nt bench: {} not found, its rows time our side only", l.cfg);
            continue;
        };
        // /usr/include and its multiarch directory are the compiler's own: no -I for them (an
        // -I/usr/include would break the #include_next of C++ headers).
        let inc = format!("-I{}", hd.display());
        if !hd.starts_with("/usr/include") && !cflags.contains(&inc) {
            cflags.push(inc);
        }
        let lib = format!("-L{}", ld.display());
        if !ldirs.contains(&lib) {
            ldirs.push(lib);
        }
        if l.cfg == "primesieve" && !inc_dirs.iter().any(|d| d.join("primecount.h").exists()) {
            println!("cargo:warning=nt bench: primecount not found, primesieve left out too");
            continue;
        }
        let obj = out.join(l.shim.replace(".c", ".o"));
        let st = Command::new("cc")
            .args(["-O3", cpu, "-std=c11", "-fPIC", "-c"])
            .args(&cflags)
            .arg(Path::new("csrc").join(l.shim))
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("cc");
        if !st.success() {
            println!("cargo:warning=nt bench: the shim of {} did not compile, left out", l.cfg);
            continue;
        }
        objects.push(obj);
        for d in &ldirs {
            println!("cargo:rustc-link-search=native={}", &d[2..]);
        }
        for n in l.link {
            println!("cargo:rustc-link-lib=dylib={n}");
        }
        println!("cargo:rustc-cfg=has_{}", l.cfg);
        if l.cfg == "primesieve" {
            println!("cargo:rustc-cfg=has_primecount");
        }
        found.push(l.cfg);
    }
    if !objects.is_empty() {
        let lib = out.join("libntrefs.a");
        let _ = std::fs::remove_file(&lib);
        let st = Command::new("ar").arg("crs").arg(&lib).args(&objects).status().expect("ar");
        assert!(st.success(), "ar failed");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=ntrefs");
        // The static shims come first on the command line; the libraries after them.
        for l in LIBS.iter().filter(|l| found.contains(&l.cfg)) {
            for n in l.link {
                println!("cargo:rustc-link-lib=dylib={n}");
            }
        }
    }
    println!("cargo:rustc-env=NT_REFS_FOUND={}", found.join(","));
}
