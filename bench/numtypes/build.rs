// Finds GMP and OpenSSL's libcrypto: by pkg-config where it is installed, else in the standard
// places (Homebrew on macOS, the Debian/Ubuntu multiarch directories on Linux). A library that is
// not found is left out, and the benchmarks skip its rows (cfg has_gmp / has_openssl).
use std::path::Path;
use std::process::Command;

fn pkg_config(name: &str) -> Option<Vec<String>> {
    let out = Command::new("pkg-config").args(["--libs-only-L", name]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    Some(s.split_whitespace().filter_map(|t| t.strip_prefix("-L").map(str::to_string)).collect())
}

fn find(lib: &str, pc: &str, extra: &[&str]) -> Option<String> {
    let mut dirs: Vec<String> = Vec::new();
    if let Some(d) = pkg_config(pc) {
        dirs.extend(d);
        if dirs.is_empty() {
            // pkg-config found it in a default linker directory
            return Some(String::new());
        }
    }
    dirs.extend(extra.iter().map(|s| s.to_string()));
    for d in [
        "/opt/homebrew/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
    ] {
        dirs.push(d.to_string());
    }
    for d in dirs {
        for ext in ["dylib", "so", "a"] {
            if Path::new(&format!("{d}/lib{lib}.{ext}")).exists() {
                return Some(d);
            }
        }
    }
    None
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(has_gmp)");
    println!("cargo::rustc-check-cfg=cfg(has_openssl)");
    println!("cargo::rerun-if-env-changed=NTBENCH_NO_GMP");
    println!("cargo::rerun-if-env-changed=NTBENCH_NO_OPENSSL");
    if std::env::var_os("NTBENCH_NO_OPENSSL").is_none() {
        if let Some(d) = find("crypto", "libcrypto", &["/opt/homebrew/opt/openssl@3/lib", "/usr/local/opt/openssl@3/lib"]) {
            if !d.is_empty() {
                println!("cargo::rustc-link-search=native={d}");
            }
            println!("cargo::rustc-link-lib=crypto");
            println!("cargo::rustc-cfg=has_openssl");
        } else {
            println!("cargo::warning=OpenSSL libcrypto not found: its rows are skipped");
        }
    }
    if std::env::var_os("NTBENCH_NO_GMP").is_none() {
        if let Some(d) = find("gmp", "gmp", &[]) {
            if !d.is_empty() {
                println!("cargo::rustc-link-search=native={d}");
            }
            println!("cargo::rustc-link-lib=gmp");
            println!("cargo::rustc-cfg=has_gmp");
        } else {
            println!("cargo::warning=GMP not found: its rows are skipped");
        }
    }
}
