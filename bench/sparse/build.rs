// Compiles the C references (SuiteSparse; Accelerate on macOS) at -O3 with the native CPU.
use std::path::PathBuf;
use std::process::Command;

fn pkg_config_include(name: &str) -> Option<PathBuf> {
    let out = Command::new("pkg-config").args(["--cflags-only-I", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    s.split_whitespace()
        .filter_map(|t| t.strip_prefix("-I"))
        .map(PathBuf::from)
        .find(|p| p.join("cholmod.h").exists())
}

fn pkg_config_libdir(name: &str) -> Option<String> {
    let out = Command::new("pkg-config").args(["--variable=libdir", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn main() {
    println!("cargo:rerun-if-changed=csrc/ss.c");
    println!("cargo:rerun-if-changed=csrc/accel.c");
    let target = std::env::var("TARGET").unwrap();
    let apple = target.contains("apple");
    let native = if target.starts_with("aarch64") { "-mcpu=native" } else { "-march=native" };

    let inc = pkg_config_include("CHOLMOD")
        .or_else(|| {
            ["/opt/homebrew/include/suitesparse", "/usr/local/include/suitesparse", "/usr/include/suitesparse"]
                .iter()
                .map(PathBuf::from)
                .find(|p| p.join("cholmod.h").exists())
        })
        .expect("SuiteSparse headers (cholmod.h) not found: install SuiteSparse (libsuitesparse-dev)");
    let libdir = pkg_config_libdir("CHOLMOD").or_else(|| {
        ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"]
            .iter()
            .find(|d| std::path::Path::new(&format!("{d}/libcholmod.so")).exists()
                || std::path::Path::new(&format!("{d}/libcholmod.dylib")).exists())
            .map(|s| s.to_string())
    });

    cc::Build::new()
        .file("csrc/ss.c")
        .include(&inc)
        .opt_level(3)
        .flag(native)
        .warnings(false)
        .compile("ssref");
    if let Some(d) = &libdir {
        println!("cargo:rustc-link-search=native={d}");
        // The libraries keep their own install names on macOS; on Linux an rpath helps.
        if !apple {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
        }
    }
    for lib in ["spqr", "umfpack", "klu", "cholmod", "cxsparse", "ldl", "camd", "colamd", "amd", "suitesparseconfig"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    if apple {
        cc::Build::new()
            .file("csrc/accel.c")
            .opt_level(3)
            .flag(native)
            .flag("-fblocks")
            .warnings(false)
            .compile("accelref");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=have_accelerate");
    }
    println!("cargo:rustc-check-cfg=cfg(have_accelerate)");
}
