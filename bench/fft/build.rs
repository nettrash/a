// Finds FFTW 3 (double and float) by pkg-config, $FFTW_LIB_DIR, or the standard directories,
// and links Accelerate on macOS.
use std::path::Path;
use std::process::Command;

fn has_lib(dir: &str, name: &str) -> bool {
    ["dylib", "so", "a"]
        .iter()
        .any(|e| Path::new(&format!("{dir}/lib{name}.{e}")).exists())
}

fn main() {
    println!("cargo:rerun-if-env-changed=FFTW_LIB_DIR");
    let mut dirs: Vec<String> = Vec::new();
    if let Ok(d) = std::env::var("FFTW_LIB_DIR") {
        dirs.push(d);
    }
    if let Ok(o) = Command::new("pkg-config")
        .args(["--libs-only-L", "fftw3", "fftw3f"])
        .output()
    {
        if o.status.success() {
            for t in String::from_utf8_lossy(&o.stdout).split_whitespace() {
                if let Some(d) = t.strip_prefix("-L") {
                    dirs.push(d.to_string());
                }
            }
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
        dirs.push(d.to_string());
    }
    let dir = dirs
        .iter()
        .find(|d| has_lib(d, "fftw3") && has_lib(d, "fftw3f"))
        .unwrap_or_else(|| panic!("FFTW 3 (fftw3 and fftw3f) not found in {dirs:?}; set FFTW_LIB_DIR"));
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=fftw3");
    println!("cargo:rustc-link-lib=fftw3f");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
