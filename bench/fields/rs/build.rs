//! Builds the C shims of the reference libraries that are installed, and tells the harness which
//! with `cfg(has_flint)`, `cfg(has_ntl)`, `cfg(has_gmp)` and `cfg(has_ossl)`.
//!
//! Each library is looked for with `pkg-config` when it is there, then in the prefixes of
//! `FIELDBENCH_PREFIXES` (separated by `:`), then in /opt/homebrew, /usr/local and /usr. Setting
//! `FIELDBENCH_NO_FLINT=1` (or `_NTL`, `_GMP`, `_OSSL`) leaves a library out. The shims are
//! compiled at -O3 for the native CPU (`-mcpu=native` on aarch64, `-march=native` on x86_64).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Lib {
    include: Vec<PathBuf>,
    link_dirs: Vec<PathBuf>,
    libs: Vec<String>,
}

fn pkg_config(name: &str) -> Option<Lib> {
    let out = Command::new("pkg-config").args(["--cflags", "--libs", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let mut lib = Lib { include: vec![], link_dirs: vec![], libs: vec![] };
    for tok in text.split_whitespace() {
        if let Some(d) = tok.strip_prefix("-I") {
            lib.include.push(d.into());
        } else if let Some(d) = tok.strip_prefix("-L") {
            lib.link_dirs.push(d.into());
        } else if let Some(l) = tok.strip_prefix("-l") {
            lib.libs.push(l.into());
        }
    }
    Some(lib)
}

fn prefixes() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = env::var("FIELDBENCH_PREFIXES")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    v.extend(["/opt/homebrew", "/usr/local", "/usr"].iter().map(PathBuf::from));
    v
}

/// The library whose header `header` is in a prefix's include directory and whose `libname` is in
/// its lib directory (or a multiarch one).
fn find(pc: &str, header: &str, libname: &str, extra: &[&str]) -> Option<Lib> {
    if let Some(l) = pkg_config(pc) {
        let has = l.include.iter().any(|d| d.join(header).exists()) || l.include.is_empty();
        if has {
            let mut l = l;
            for e in extra {
                if !l.libs.iter().any(|x| x == e) {
                    l.libs.push((*e).into());
                }
            }
            return Some(l);
        }
    }
    for p in prefixes() {
        let inc = p.join("include");
        if !inc.join(header).exists() {
            continue;
        }
        let mut dirs = vec![p.join("lib")];
        if let Ok(rd) = std::fs::read_dir(p.join("lib")) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().ends_with("-linux-gnu") {
                    dirs.push(e.path());
                }
            }
        }
        let found = dirs.iter().find(|d| {
            ["so", "dylib", "a"].iter().any(|x| d.join(format!("lib{libname}.{x}")).exists())
        });
        if let Some(d) = found {
            let mut libs = vec![libname.to_string()];
            libs.extend(extra.iter().map(|s| s.to_string()));
            return Some(Lib { include: vec![inc], link_dirs: vec![d.clone()], libs });
        }
    }
    None
}

fn native(b: &mut cc::Build) {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    b.flag("-O3");
    if arch == "aarch64" {
        b.flag_if_supported("-mcpu=native");
    } else if arch == "x86_64" {
        b.flag_if_supported("-march=native");
    }
    b.warnings(false);
}

fn link(l: &Lib) {
    for d in &l.link_dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
        }
    }
    for x in &l.libs {
        println!("cargo:rustc-link-lib={x}");
    }
}

fn skip(name: &str) -> bool {
    env::var(format!("FIELDBENCH_NO_{name}")).map(|v| v == "1").unwrap_or(false)
}

fn main() {
    for c in ["has_flint", "has_ntl", "has_gmp", "has_ossl", "has_mcl"] {
        println!("cargo::rustc-check-cfg=cfg({c})");
    }
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=FIELDBENCH_PREFIXES");
    for n in ["FLINT", "NTL", "GMP", "OSSL", "MCL"] {
        println!("cargo:rerun-if-env-changed=FIELDBENCH_NO_{n}");
    }
    let csrc = Path::new("csrc");
    let mut summary = vec![];

    if !skip("GMP") {
        if let Some(l) = find("gmp", "gmp.h", "gmp", &[]) {
            let mut b = cc::Build::new();
            native(&mut b);
            b.file(csrc.join("gmp_shim.c")).includes(&l.include).include(csrc);
            b.compile("fb_gmp");
            link(&l);
            println!("cargo:rustc-cfg=has_gmp");
            summary.push("gmp");
        }
    }
    if !skip("FLINT") {
        if let Some(l) = find("flint", "flint/flint.h", "flint", &["gmp", "mpfr"]) {
            let mut b = cc::Build::new();
            native(&mut b);
            b.file(csrc.join("flint_shim.c")).includes(&l.include).include(csrc);
            b.compile("fb_flint");
            link(&l);
            println!("cargo:rustc-cfg=has_flint");
            summary.push("flint");
        }
    }
    if !skip("NTL") {
        if let Some(l) = find("ntl", "NTL/ZZ.h", "ntl", &["gmp"]) {
            let mut b = cc::Build::new();
            native(&mut b);
            b.cpp(true).flag("-std=c++17");
            b.file(csrc.join("ntl_shim.cpp")).includes(&l.include).include(csrc);
            b.compile("fb_ntl");
            link(&l);
            println!("cargo:rustc-cfg=has_ntl");
            summary.push("ntl");
        }
    }
    if !skip("OSSL") {
        if let Some(l) = find("libcrypto", "openssl/bn.h", "crypto", &[]) {
            let mut b = cc::Build::new();
            native(&mut b);
            b.file(csrc.join("ossl_shim.c")).includes(&l.include).include(csrc);
            b.compile("fb_ossl");
            link(&l);
            println!("cargo:rustc-cfg=has_ossl");
            summary.push("openssl");
        }
    }
    // mcl (herumi/mcl, BSD-3), a build of its default MCL_FP_BIT=384, MCL_FR_BIT=256 made apart:
    // MCL_DIR, or bench/ext/mcl (built by bench/mcl.sh).
    println!("cargo:rerun-if-env-changed=MCL_DIR");
    if !skip("MCL") {
        let dir = env::var("MCL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../ext/mcl")
        });
        let lib = dir.join("lib/libmcl.a");
        println!("cargo:rerun-if-changed={}", lib.display());
        if lib.exists() {
            println!("cargo:rustc-link-search=native={}", dir.join("lib").display());
            println!("cargo:rustc-link-lib=static=mcl");
            println!("cargo:rustc-link-lib=dylib={}", if cfg!(target_os = "macos") { "c++" } else { "stdc++" });
            println!("cargo:rustc-cfg=has_mcl");
            summary.push("mcl");
        }
    }
    println!("cargo:rustc-env=FIELDBENCH_CREFS={}", summary.join(","));
}
