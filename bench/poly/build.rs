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
        for lib in ["flint", "ntl", "gmp"] {
            for f in pkg_config(lib, "--cflags-only-I") {
                incs.push(f.trim_start_matches("-I").to_string());
            }
            for f in pkg_config(lib, "--libs-only-L") {
                libdirs.push(f.trim_start_matches("-L").to_string());
            }
        }
        incs.push("/usr/local/include".into());
        libdirs.push("/usr/local/lib".into());
    }
    // Searched for the headers but not passed with -I: the compiler's own directories, among them
    // Debian's and Ubuntu's multiarch one (gmp.h and PARI's pari/ are in
    // /usr/include/x86_64-linux-gnu there).
    let mut sys_incs: Vec<String> = Vec::new();
    if target_os != "macos" {
        sys_incs.push("/usr/include".into());
        sys_incs.push(format!("/usr/include/{target_arch}-linux-gnu"));
    }
    let found = |h: &str| incs.iter().chain(sys_incs.iter()).any(|d| Path::new(d).join(h).exists());
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
    let mut lapack_dirs: Vec<String> = pkg_config("lapack", "--libs-only-L").iter().map(|f| f.trim_start_matches("-L").to_string()).collect();
    lapack_dirs.extend([format!("/usr/lib/{target_arch}-linux-gnu"), "/usr/lib64".into(), "/usr/lib".into(), "/usr/local/lib".into()]);
    let have_lapack = (target_os == "macos" || lapack_dirs.iter().any(|d| Path::new(d).join("liblapack.so").exists()))
        && env::var("POLYBENCH_NO_LAPACK").is_err();
    for (name, have, var) in [
        ("FLINT", have_flint, "POLYBENCH_NO_FLINT"),
        ("NTL", have_ntl, "POLYBENCH_NO_NTL"),
        ("PARI", have_pari, "POLYBENCH_NO_PARI"),
        ("LAPACK", have_lapack, "POLYBENCH_NO_LAPACK"),
    ] {
        if !have && env::var(var).is_err() {
            println!("cargo:warning=poly bench: {name} not found, its rows are left out");
        }
    }
    build("pbflint", "flint.c", false, "sf", have_flint);
    build("pbntl", "ntl.cpp", true, "sn", have_ntl);
    build("pbpari", "pari.c", false, "sp", have_pari);
    build("pblapack", "lapack.c", false, "sl", have_lapack);
    if have_lapack && target_os != "macos" {
        if let Some(d) = lapack_dirs.iter().find(|d| Path::new(d).join("liblapack.so").exists()) {
            libdirs.push(d.clone());
        }
    }
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
