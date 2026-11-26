// Links GMP for the differential checks and the reference timings: found by pkg-config or in the
// standard prefixes (Homebrew on macOS, /usr on Linux).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut dirs = Vec::new();
    if let Ok(o) = std::process::Command::new("pkg-config").args(["--libs-only-L", "gmp"]).output() {
        for f in String::from_utf8_lossy(&o.stdout).split_whitespace() {
            if let Some(d) = f.strip_prefix("-L") {
                dirs.push(d.to_string());
            }
        }
    }
    for d in ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if std::path::Path::new(d).exists() {
            dirs.push(d.to_string());
        }
    }
    for d in dirs {
        println!("cargo:rustc-link-search=native={d}");
    }
    println!("cargo:rustc-link-lib=dylib=gmp");
}
