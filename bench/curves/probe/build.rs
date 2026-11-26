// Links mcl when it has been built (bench/mcl.sh builds it in bench/ext/mcl, or MCL_DIR points
// at a build): sets the cfg `have_mcl`. OpenSSL is loaded at run time (dlopen), so nothing here.
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(have_mcl)");
    println!("cargo::rerun-if-env-changed=MCL_DIR");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = std::env::var("MCL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../ext/mcl"));
    let lib = dir.join("lib/libmcl.a");
    println!("cargo::rerun-if-changed={}", lib.display());
    if lib.exists() {
        println!("cargo::rustc-link-search=native={}", dir.join("lib").display());
        println!("cargo::rustc-link-lib=static=mcl");
        if cfg!(target_os = "macos") {
            println!("cargo::rustc-link-lib=dylib=c++");
        } else {
            println!("cargo::rustc-link-lib=dylib=stdc++");
        }
        println!("cargo::rustc-cfg=have_mcl");
    }
}
