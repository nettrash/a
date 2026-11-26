// Finds GMP, FLINT and PARI (headers and libraries) and compiles the C reference drivers at
// -O3 with the native CPU flags. Search order: $REF_PREFIX, then /opt/homebrew, /usr/local,
// /usr (with the Debian/Ubuntu multiarch directories). FLINT and PARI are optional: without
// them the harness leaves their rows out.
use std::path::{Path, PathBuf};

fn prefixes() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("REF_PREFIX") {
        for s in p.split(':').filter(|s| !s.is_empty()) {
            v.push(PathBuf::from(s));
        }
    }
    for p in ["/opt/homebrew", "/usr/local", "/usr"] {
        v.push(PathBuf::from(p));
    }
    v
}

fn include_dirs(prefix: &Path) -> Vec<PathBuf> {
    vec![
        prefix.join("include"),
        prefix.join("include/x86_64-linux-gnu"),
        prefix.join("include/aarch64-linux-gnu"),
    ]
}

fn lib_dirs(prefix: &Path) -> Vec<PathBuf> {
    vec![
        prefix.join("lib"),
        prefix.join("lib/x86_64-linux-gnu"),
        prefix.join("lib/aarch64-linux-gnu"),
        prefix.join("lib64"),
    ]
}

fn find_header(rel: &str) -> Option<PathBuf> {
    for p in prefixes() {
        for d in include_dirs(&p) {
            if d.join(rel).exists() {
                return Some(d);
            }
        }
    }
    None
}

fn find_lib(name: &str) -> Option<PathBuf> {
    for p in prefixes() {
        for d in lib_dirs(&p) {
            for ext in ["dylib", "so", "a"] {
                if d.join(format!("lib{name}.{ext}")).exists() {
                    return Some(d);
                }
            }
        }
    }
    None
}

fn native(b: &mut cc::Build) {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "aarch64" {
        b.flag("-mcpu=native");
    } else {
        b.flag("-march=native");
    }
    b.opt_level(3).flag("-O3").warnings(false);
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_flint)");
    println!("cargo::rustc-check-cfg=cfg(has_pari)");
    println!("cargo:rerun-if-changed=c/gmpflint.c");
    println!("cargo:rerun-if-changed=c/pari.c");
    println!("cargo:rerun-if-env-changed=REF_PREFIX");
    println!("cargo:rerun-if-env-changed=REF_NO_FLINT");
    println!("cargo:rerun-if-env-changed=REF_NO_PARI");

    let gmp_inc = find_header("gmp.h").expect("gmp.h not found (set REF_PREFIX)");
    let gmp_lib = find_lib("gmp").expect("libgmp not found (set REF_PREFIX)");
    let flint_inc = find_header("flint/fmpq.h").filter(|_| std::env::var("REF_NO_FLINT").is_err());
    let flint_lib = find_lib("flint");
    let pari_inc = find_header("pari/pari.h").filter(|_| std::env::var("REF_NO_PARI").is_err());
    let pari_lib = find_lib("pari");

    let mut b = cc::Build::new();
    b.file("c/gmpflint.c").include(&gmp_inc);
    native(&mut b);
    let has_flint = flint_inc.is_some() && flint_lib.is_some();
    if has_flint {
        b.include(flint_inc.as_ref().unwrap()).define("HAS_FLINT", None);
        println!("cargo:rustc-cfg=has_flint");
    }
    b.compile("refs_gmpflint");

    let has_pari = pari_inc.is_some() && pari_lib.is_some();
    if has_pari {
        let mut p = cc::Build::new();
        p.file("c/pari.c").include(pari_inc.as_ref().unwrap()).include(&gmp_inc);
        native(&mut p);
        p.compile("refs_pari");
        println!("cargo:rustc-cfg=has_pari");
    }

    let mut dirs = vec![gmp_lib];
    if has_flint {
        dirs.push(flint_lib.unwrap());
    }
    if has_pari {
        dirs.push(pari_lib.unwrap());
    }
    dirs.dedup();
    for d in dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    if has_pari {
        println!("cargo:rustc-link-lib=dylib=pari");
    }
    if has_flint {
        println!("cargo:rustc-link-lib=dylib=flint");
    }
    println!("cargo:rustc-link-lib=dylib=gmp");
}
