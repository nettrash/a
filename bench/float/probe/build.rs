// GMP, MPFR and GNU MPC: on macOS the system's static archives (Homebrew's, as the earlier
// campaigns linked them), searched by pkg-config when it exists, else in the standard places;
// `FLOATBENCH_LIBDIR` (colon-separated) is searched first. Elsewhere gmp-mpfr-sys builds them from
// its bundled sources (Cargo.toml) and links them itself: nothing to find here. FLINT (Arb; a
// shared library) is found the same way on every system; `FLOATBENCH_NO_ARB=1` leaves Arb out.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find(name: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    dirs.iter().map(|d| d.join(name)).find(|p| p.exists())
}

fn pkg_libdir(pkg: &str) -> Option<PathBuf> {
    let o = Command::new("pkg-config").args(["--variable=libdir", pkg]).output().ok()?;
    if !o.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (!s.is_empty()).then(|| PathBuf::from(s))
}

fn main() {
    println!("cargo:rerun-if-env-changed=FLOATBENCH_LIBDIR");
    println!("cargo:rerun-if-env-changed=FLOATBENCH_NO_ARB");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(has_arb)");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("FLOATBENCH_LIBDIR") {
        dirs.extend(p.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    for pkg in ["gmp", "mpfr", "mpc", "flint"] {
        if let Some(d) = pkg_libdir(pkg) {
            dirs.push(d);
        }
    }
    for d in [
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
    ] {
        dirs.push(PathBuf::from(d));
    }
    let mut found = Vec::new();
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // A directory holding only the static archives, searched first, so that `-lgmp` from
        // gmp-mpfr-sys (rug) resolves to the same archive as ours: one copy of GMP and MPFR.
        let st = out.join("static");
        fs::create_dir_all(&st).unwrap();
        for lib in ["mpc", "mpfr", "gmp"] {
            let name = format!("lib{lib}.a");
            let a = find(&name, &dirs).unwrap_or_else(|| panic!("{name} not found in {dirs:?}"));
            let dst = st.join(&name);
            let _ = fs::remove_file(&dst);
            std::os::unix::fs::symlink(&a, &dst).unwrap();
            found.push(a.display().to_string());
        }
        println!("cargo:rustc-link-search=native={}", st.display());
        for lib in ["mpc", "mpfr", "gmp"] {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    } else {
        // gmp-mpfr-sys's own build (links = "gmp" gives its directories): rug and the probe's
        // bindings both use it.
        let d = env::var("DEP_GMP_LIB_DIR").unwrap_or_else(|_| "?".into());
        for lib in ["mpc", "mpfr", "gmp"] {
            found.push(format!("{d}/lib{lib}.a (gmp-mpfr-sys, from source)"));
        }
    }
    if env::var("FLOATBENCH_NO_ARB").is_err() {
        let so = if cfg!(target_os = "macos") { "libflint.dylib" } else { "libflint.so" };
        if let Some(f) = find(so, &dirs) {
            let f = fs::canonicalize(&f).unwrap_or(f);
            println!("cargo:rustc-link-arg={}", f.display());
            if let Some(dir) = Path::new(&f).parent() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            }
            println!("cargo:rustc-cfg=has_arb");
            found.push(f.display().to_string());
        }
    }
    println!("cargo:rustc-env=FB_LIBS={}", found.join(" "));
}
