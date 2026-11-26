// Compiles the C and C++ reference drivers (GSL, Boost) at -O3 with the native CPU flags and
// links GSL. Paths: $NUMPROBE_PREFIX (default
// /opt/homebrew on macOS, /usr on Linux), or pkg-config when it is present.
use std::process::Command;

fn pkg(name: &str, what: &str) -> Option<Vec<String>> {
    let out = Command::new("pkg-config").args([what, name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .map(String::from)
            .collect(),
    )
}

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=NUMPROBE_PREFIX");
    let macos = std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos";
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let prefix = std::env::var("NUMPROBE_PREFIX")
        .unwrap_or_else(|_| if macos { "/opt/homebrew".into() } else { "/usr".into() });
    let native = if arch == "aarch64" { "-mcpu=native" } else { "-march=native" };

    let mut incs = vec![format!("{prefix}/include")];
    if let Some(f) = pkg("gsl", "--cflags") {
        incs.extend(f.into_iter().filter_map(|s| s.strip_prefix("-I").map(String::from)));
    }
    let mut libdirs = vec![format!("{prefix}/lib")];
    if !macos {
        libdirs.push(format!("{prefix}/lib/{}-linux-gnu", arch));
    }
    if let Some(f) = pkg("gsl", "--libs") {
        libdirs.extend(f.into_iter().filter_map(|s| s.strip_prefix("-L").map(String::from)));
    }

    let mut c = cc::Build::new();
    c.file("csrc/gsl_probe.c")
        .opt_level(3)
        .flag(native)
        .define("NDEBUG", None)
        .define("HAVE_INLINE", None)
        .define("GSL_RANGE_CHECK_OFF", None)
        .warnings(false);
    for i in &incs {
        c.include(i);
    }
    c.compile("gslprobe");

    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .file("csrc/boost_probe.cpp")
        .opt_level(3)
        .flag(native)
        .flag("-std=c++17")
        .define("NDEBUG", None)
        .define("BOOST_UBLAS_NDEBUG", None)
        .warnings(false);
    for i in &incs {
        cpp.include(i);
    }
    cpp.compile("boostprobe");

    for d in &libdirs {
        println!("cargo:rustc-link-search=native={d}");
    }
    println!("cargo:rustc-link-lib=gsl");
    println!("cargo:rustc-link-lib=gslcblas");
    if macos {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
