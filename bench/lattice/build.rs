//! Builds the C/C++ shims of the references that are installed, and sets `has_<lib>` for each.
//!
//! The libraries are looked for by `pkg-config` when it exists, else under the prefixes in
//! `LATBENCH_PREFIXES` (colon-separated), `/opt/homebrew`, `/usr/local` and `/usr`. The shims
//! are compiled at `-O3` with `-mcpu=native` (aarch64) or `-march=native` (x86_64).
use std::path::{Path, PathBuf};
use std::process::Command;

struct Lib {
    name: &'static str,
    header: &'static str,
    source: &'static str,
    cxx: bool,
    pkg: &'static str,
    libs: &'static [&'static str],
}

const LIBS: &[Lib] = &[
    Lib {
        name: "flint",
        header: "flint/flint.h",
        source: "csrc/flint_shim.c",
        cxx: false,
        pkg: "flint",
        libs: &["flint", "mpfr", "gmp"],
    },
    Lib {
        name: "fplll",
        header: "fplll.h",
        source: "csrc/fplll_shim.cpp",
        cxx: true,
        pkg: "fplll",
        libs: &["fplll", "mpfr", "gmp"],
    },
    Lib {
        name: "ntl",
        header: "NTL/ZZ.h",
        source: "csrc/ntl_shim.cpp",
        cxx: true,
        pkg: "ntl",
        libs: &["ntl", "gmp"],
    },
    Lib {
        name: "m4ri",
        header: "m4ri/m4ri.h",
        source: "csrc/m4ri_shim.c",
        cxx: false,
        pkg: "m4ri",
        libs: &["m4ri"],
    },
    Lib {
        name: "pari",
        header: "pari/pari.h",
        source: "csrc/pari_shim.c",
        cxx: false,
        pkg: "",
        libs: &["pari", "gmp"],
    },
];

fn pkg_config(pkg: &str, flag: &str) -> Option<String> {
    if pkg.is_empty() {
        return None;
    }
    let out = Command::new("pkg-config").args([flag, pkg]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=LATBENCH_PREFIXES");
    println!("cargo:rerun-if-env-changed=LATBENCH_SKIP");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let native = if arch == "aarch64" { "-mcpu=native" } else { "-march=native" };
    let skip = std::env::var("LATBENCH_SKIP").unwrap_or_default();
    let mut prefixes: Vec<String> = std::env::var("LATBENCH_PREFIXES")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    prefixes.extend(["/opt/homebrew", "/usr/local", "/usr"].map(str::to_string));
    let mut any_cxx = false;
    let mut link_dirs: Vec<String> = Vec::new();
    for lib in LIBS {
        println!("cargo::rustc-check-cfg=cfg(has_{})", lib.name);
        if skip.split(',').any(|s| s == lib.name) {
            continue;
        }
        let (cflags, ldirs): (Vec<String>, Vec<String>) =
            if let Some(c) = pkg_config(lib.pkg, "--cflags") {
                let l = pkg_config(lib.pkg, "--libs-only-L").unwrap_or_default();
                (
                    c.split_whitespace().map(str::to_string).collect(),
                    l.split_whitespace()
                        .filter_map(|s| s.strip_prefix("-L"))
                        .map(str::to_string)
                        .collect(),
                )
            } else {
                match prefixes
                    .iter()
                    .find(|p| Path::new(&format!("{p}/include/{}", lib.header)).exists())
                {
                    Some(p) => {
                        let mut l = vec![format!("{p}/lib")];
                        if os == "linux" {
                            l.push(format!("{p}/lib/{arch}-linux-gnu"));
                        }
                        (vec![format!("-I{p}/include")], l)
                    }
                    None => {
                        println!("cargo:warning=latbench: {} not found, its rows are skipped", lib.name);
                        continue;
                    }
                }
            };
        let mut cflags = cflags;
        // gmp.h and friends live in the prefix's include directory too.
        for p in &prefixes {
            if Path::new(&format!("{p}/include/gmp.h")).exists() {
                cflags.push(format!("-I{p}/include"));
                break;
            }
        }
        let obj = out.join(format!("{}_shim.o", lib.name));
        let compiler = if lib.cxx { "c++" } else { "cc" };
        let mut cmd = Command::new(compiler);
        cmd.args(["-O3", native, "-fPIC", "-c", lib.source, "-Icsrc", "-o"]).arg(&obj);
        if lib.cxx {
            cmd.arg("-std=c++17");
        }
        cmd.args(&cflags);
        let st = cmd.status().expect("the C compiler");
        if !st.success() {
            println!("cargo:warning=latbench: the {} shim did not compile, its rows are skipped", lib.name);
            continue;
        }
        let a = out.join(format!("lib{}_shim.a", lib.name));
        let _ = std::fs::remove_file(&a);
        let st = Command::new("ar").arg("rcs").arg(&a).arg(&obj).status().unwrap();
        assert!(st.success());
        println!("cargo:rustc-link-lib=static={}_shim", lib.name);
        for d in ldirs {
            if !link_dirs.contains(&d) {
                link_dirs.push(d);
            }
        }
        for l in lib.libs {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        println!("cargo:rustc-cfg=has_{}", lib.name);
        any_cxx |= lib.cxx;
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    for d in link_dirs {
        println!("cargo:rustc-link-search=native={d}");
        if os == "macos" || os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
        }
    }
    if any_cxx {
        println!("cargo:rustc-link-lib=dylib={}", if os == "macos" { "c++" } else { "stdc++" });
    }
}
