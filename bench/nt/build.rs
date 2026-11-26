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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=NT_PREFIXES");
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
        if !l.needs.iter().all(|n| found.contains(n)) {
            println!("cargo:warning=nt bench: {} needs {:?}, left out", l.cfg, l.needs);
            continue;
        }
        let (mut cflags, mut ldirs) = pkg_config(l.pkg).unwrap_or_default();
        let mut ok = cflags.iter().any(|f| f.starts_with("-I") && Path::new(&f[2..]).join(l.header).exists());
        if !ok {
            for p in &prefixes {
                let inc = p.join("include");
                if inc.join(l.header).exists() {
                    cflags.push(format!("-I{}", inc.display()));
                    for d in [p.join("lib"), p.join("lib").join(multiarch), p.join("lib64")] {
                        if d.exists() {
                            ldirs.push(format!("-L{}", d.display()));
                        }
                    }
                    ok = true;
                    break;
                }
            }
        }
        if !ok && Path::new("/usr/include").join(l.header).exists() {
            ok = true;
        }
        if !ok {
            println!("cargo:warning=nt bench: {} not found, its rows time our side only", l.cfg);
            continue;
        }
        if l.cfg == "primesieve" && !cflags.iter().any(|f| f.starts_with("-I") && Path::new(&f[2..]).join("primecount.h").exists())
            && !prefixes.iter().any(|p| p.join("include/primecount.h").exists())
        {
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
