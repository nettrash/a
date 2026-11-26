// Compiles the C/C++ reference drivers that are available on this machine into one static
// library and tells the crate which ones it has (cfg has_boost, has_gsl).
//
// Search paths: SB_INCLUDE and SB_LIBDIR (colon-separated, set by run.sh from pkg-config on
// Linux), then /opt/homebrew, /usr/local and /usr. Compilers: CC / CXX (default cc / c++), at
// -O3 and -mcpu=native (aarch64) or -march=native (x86_64).
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn split(var: &str) -> Vec<PathBuf> {
    env::var(var)
        .ok()
        .map(|s| s.split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default()
}

fn find(dirs: &[PathBuf], rel: &str) -> Option<PathBuf> {
    dirs.iter().find(|d| d.join(rel).exists()).cloned()
}

fn compile(compiler: &str, flags: &[String], src: &Path, obj: &Path) -> bool {
    let st = Command::new(compiler).args(flags).arg("-c").arg(src).arg("-o").arg(obj).status();
    matches!(st, Ok(s) if s.success())
}

fn main() {
    for f in ["csrc/refs_libm.c", "csrc/refs_boost.cpp", "csrc/refs_gsl.c", "build.rs"] {
        println!("cargo:rerun-if-changed={f}");
    }
    for v in ["SB_INCLUDE", "SB_LIBDIR", "CC", "CXX", "SB_NO_BOOST", "SB_NO_GSL"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo::rustc-check-cfg=cfg(has_boost)");
    println!("cargo::rustc-check-cfg=cfg(has_gsl)");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".into());
    let tune = if arch == "aarch64" { "-mcpu=native" } else { "-march=native" };

    let mut inc = split("SB_INCLUDE");
    let mut lib = split("SB_LIBDIR");
    for p in ["/opt/homebrew", "/usr/local", "/usr"] {
        inc.push(PathBuf::from(p).join("include"));
        lib.push(PathBuf::from(p).join("lib"));
    }
    lib.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    lib.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));

    let mut base: Vec<String> = vec!["-O3".into(), tune.into(), "-fPIC".into()];
    let mut objs = Vec::new();

    let o = out.join("refs_libm.o");
    assert!(compile(&cc, &base, Path::new("csrc/refs_libm.c"), &o), "refs_libm.c");
    objs.push(o);

    if env::var("SB_NO_BOOST").is_err() {
        if let Some(d) = find(&inc, "boost/math/special_functions/gamma.hpp") {
            let mut f = base.clone();
            f.push("-std=c++17".into());
            f.push(format!("-I{}", d.display()));
            let o = out.join("refs_boost.o");
            if compile(&cxx, &f, Path::new("csrc/refs_boost.cpp"), &o) {
                objs.push(o);
                println!("cargo:rustc-cfg=has_boost");
                let stdlib = if os == "macos" { "c++" } else { "stdc++" };
                println!("cargo:rustc-link-lib=dylib={stdlib}");
            } else {
                println!("cargo:warning=Boost.Math found but refs_boost.cpp did not compile");
            }
        } else {
            println!("cargo:warning=Boost.Math not found: no Boost references");
        }
    }

    if env::var("SB_NO_GSL").is_err() {
        let h = find(&inc, "gsl/gsl_sf_gamma.h");
        let l = lib.iter().find(|d| {
            ["libgsl.dylib", "libgsl.so", "libgsl.a"].iter().any(|n| d.join(n).exists())
        });
        if let (Some(h), Some(l)) = (h, l) {
            base.push(format!("-I{}", h.display()));
            let o = out.join("refs_gsl.o");
            if compile(&cc, &base, Path::new("csrc/refs_gsl.c"), &o) {
                objs.push(o);
                println!("cargo:rustc-cfg=has_gsl");
                println!("cargo:rustc-link-search=native={}", l.display());
                println!("cargo:rustc-link-lib=dylib=gsl");
                println!("cargo:rustc-link-lib=dylib=gslcblas");
            } else {
                println!("cargo:warning=GSL found but refs_gsl.c did not compile");
            }
        } else {
            println!("cargo:warning=GSL not found: no GSL references");
        }
    }

    let a = out.join("libsbrefs.a");
    let _ = std::fs::remove_file(&a);
    let st = Command::new("ar").arg("crs").arg(&a).args(&objs).status().unwrap();
    assert!(st.success(), "ar");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=sbrefs");
    if os != "macos" {
        println!("cargo:rustc-link-lib=dylib=m");
    }
}
