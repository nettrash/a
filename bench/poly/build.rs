//! Builds the reference shims (FLINT, NTL, PARI, LAPACK) into static libraries, each replaced
//! by a stub when its headers are not found. macOS: Homebrew's prefix and Accelerate; Linux:
//! pkg-config where the library ships a .pc file, the standard paths otherwise.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn pkg_config(lib: &str, what: &str) -> Vec<String> {
    Command::new("pkg-config")
        .args([what, lib])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).split_whitespace().map(String::from).collect())
        .unwrap_or_default()
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let mut incs: Vec<String> = Vec::new();
    let mut libdirs: Vec<String> = Vec::new();
    if target_os == "macos" {
        let prefix = Command::new("brew")
            .arg("--prefix")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "/opt/homebrew".into());
        incs.push(format!("{prefix}/include"));
        libdirs.push(format!("{prefix}/lib"));
    } else {
        for f in pkg_config("flint", "--cflags-only-I") {
            incs.push(f.trim_start_matches("-I").to_string());
        }
        for f in pkg_config("flint", "--libs-only-L") {
            libdirs.push(f.trim_start_matches("-L").to_string());
        }
        incs.push("/usr/include".into());
        incs.push("/usr/local/include".into());
        libdirs.push("/usr/local/lib".into());
    }
    let found = |h: &str| incs.iter().any(|d| Path::new(d).join(h).exists());
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".into());
    let native = if target_arch == "aarch64" { "-mcpu=native" } else { "-march=native" };
    let shims = Path::new("shims");
    let build = |name: &str, src: &str, cpp: bool, prefix: &str, real: bool| {
        let obj = out.join(format!("{name}.o"));
        let mut cmd = Command::new(if cpp && real { &cxx } else { &cc });
        cmd.arg("-O3").arg(native).arg("-c").arg("-fPIC");
        if cpp && real {
            cmd.arg("-std=c++17");
        }
        for i in &incs {
            cmd.arg(format!("-I{i}"));
        }
        if real {
            cmd.arg(shims.join(src));
        } else {
            cmd.arg(format!("-DP={prefix}")).arg(shims.join("stub.c"));
        }
        cmd.arg("-o").arg(&obj);
        let st = cmd.status().expect("the C compiler");
        assert!(st.success(), "compiling {src} failed");
        let lib = out.join(format!("lib{name}.a"));
        let _ = std::fs::remove_file(&lib);
        let st = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().unwrap();
        assert!(st.success());
        println!("cargo:rustc-link-lib=static={name}");
        println!("cargo:rerun-if-changed=shims/{src}");
    };
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rerun-if-changed=shims/shim.h");
    println!("cargo:rerun-if-changed=shims/stub.c");
    let have_flint = found("flint/flint.h") && env::var("POLYBENCH_NO_FLINT").is_err();
    let have_ntl = found("NTL/ZZ.h") && env::var("POLYBENCH_NO_NTL").is_err();
    let have_pari = found("pari/pari.h") && env::var("POLYBENCH_NO_PARI").is_err();
    let have_lapack = (target_os == "macos"
        || ["/usr/lib/x86_64-linux-gnu/liblapack.so", "/usr/lib/aarch64-linux-gnu/liblapack.so", "/usr/lib/liblapack.so"]
            .iter()
            .any(|p| Path::new(p).exists()))
        && env::var("POLYBENCH_NO_LAPACK").is_err();
    build("pbflint", "flint.c", false, "sf", have_flint);
    build("pbntl", "ntl.cpp", true, "sn", have_ntl);
    build("pbpari", "pari.c", false, "sp", have_pari);
    build("pblapack", "lapack.c", false, "sl", have_lapack);
    for d in &libdirs {
        println!("cargo:rustc-link-search=native={d}");
        if target_os == "macos" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
        }
    }
    if have_flint {
        println!("cargo:rustc-link-lib=flint");
        println!("cargo:rustc-link-lib=mpfr");
    }
    if have_ntl {
        println!("cargo:rustc-link-lib=ntl");
        println!("cargo:rustc-link-lib={}", if target_os == "macos" { "c++" } else { "stdc++" });
    }
    if have_pari {
        println!("cargo:rustc-link-lib=pari");
    }
    if have_flint || have_ntl || have_pari {
        println!("cargo:rustc-link-lib=gmp");
    }
    if have_lapack {
        if target_os == "macos" {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        } else {
            println!("cargo:rustc-link-lib=lapack");
        }
    }
    let mut refs = Vec::new();
    if have_flint { refs.push("FLINT"); }
    if have_ntl { refs.push("NTL"); }
    if have_pari { refs.push("PARI"); }
    if have_lapack { refs.push(if target_os == "macos" { "Accelerate" } else { "LAPACK" }); }
    println!("cargo:rustc-env=POLYBENCH_REFS={}", refs.join(","));
    println!("cargo:rerun-if-env-changed=POLYBENCH_NO_FLINT");
    println!("cargo:rerun-if-env-changed=POLYBENCH_NO_NTL");
    println!("cargo:rerun-if-env-changed=POLYBENCH_NO_PARI");
    println!("cargo:rerun-if-env-changed=POLYBENCH_NO_LAPACK");
}
