// SHA-2 (a::hash) and RFC 9380's expand_message_xmd against RustCrypto's sha2 (0.11.1: ARMv8
// SHA-256/SHA-512 instructions detected at run time; 0.10.9 with default features: its portable
// backend on aarch64), ring 0.17.14 (its own assembly, SHA instructions where detected), OpenSSL 3
// and 4 (EVP with a fetched EVP_MD and a reused EVP_MD_CTX; and the one-shot SHA256() & co.),
// blst's expand_message_xmd and RustCrypto elliptic-curve 0.13.8's ExpandMsgXmd.
//
// Messages of 0, 64, 1 KiB and 1 MiB bytes (fresh content per input: a pool of 8 messages of each
// length, 2 for 1 MiB). One-shot digests; every digest compared with ours before timing.
#![allow(unused_must_use)]
#[path = "../common.rs"]
mod common;
#[path = "../ossl.rs"]
mod ossl;

use common::*;
use ossl::Ossl;

fn main() {
    let mut b = Bench::new("sha2");
    let osls: Vec<Ossl> = [3u32, 4].into_iter().filter_map(Ossl::load).collect();
    let mut rng = Rng(3);
    let sizes: [(usize, &str); 5] = [(0, "0 B"), (64, "64 B"), (1024, "1 KiB"), (16384, "16 KiB"), (1 << 20, "1 MiB")];
    for (len, lname) in sizes {
        let pool = if len >= 1 << 20 { 2 } else { 8 };
        let msgs: Vec<Vec<u8>> = (0..pool).map(|_| rng.bytes(len)).collect();
        hash_rows(&mut b, &osls, &msgs, len, lname);
    }
    xmd_rows(&mut b);
}

fn hash_rows(b: &mut Bench, osls: &[Ossl], msgs: &[Vec<u8>], _len: usize, lname: &str) {
    let np = msgs.len();
    macro_rules! one {
        ($op:literal, $ours:path, $s11:ty, $s10:ty, $ring:expr, $evp:literal, $low:ident, $out:expr) => {{
            // Checks.
            for m in msgs.iter().take(2) {
                let w = $ours(m).to_vec();
                use sha2_11::Digest as _;
                assert_eq!(<$s11>::digest(m).to_vec(), w, concat!($op, " sha2 0.11"));
                assert_eq!(<$s10 as sha2_10::Digest>::digest(m).to_vec(), w, concat!($op, " sha2 0.10"));
                if let Some(alg) = $ring {
                    assert_eq!(ring::digest::digest(alg, m).as_ref().to_vec(), w, concat!($op, " ring"));
                }
                for o in osls {
                    let mut out = [0u8; 64];
                    if let Some(f) = o.$low {
                        // SAFETY: OpenSSL writes the digest into out.
                        unsafe { f(m.as_ptr(), m.len(), out.as_mut_ptr()) };
                        assert_eq!(out[..$out].to_vec(), w, concat!($op, " openssl one-shot"));
                    }
                }
            }
            let mut refs = vec![
                side("sha2 0.11.1", "ARMv8 SHA instructions (run-time detection)", |i| {
                    use sha2_11::Digest as _;
                    black_box(<$s11>::digest(&msgs[i % np]));
                }),
                side("sha2 0.10.9", "default features: portable backend", |i| {
                    black_box(<$s10 as sha2_10::Digest>::digest(&msgs[i % np]));
                }),
            ];
            if let Some(alg) = $ring {
                refs.push(side("ring 0.17.14", "", move |i| {
                    black_box(ring::digest::digest(alg, &msgs[i % np]));
                }));
            }
            for o in osls {
                let name = std::ffi::CString::new($evp).unwrap();
                // SAFETY: fetches the digest once; a context reused for every call.
                let md = unsafe { (o.EVP_MD_fetch)(core::ptr::null_mut(), name.as_ptr(), core::ptr::null()) };
                assert!(!md.is_null());
                // SAFETY: as above.
                let ctx = unsafe { (o.EVP_MD_CTX_new)() };
                {
                    let mut out = [0u8; 64];
                    let mut n: core::ffi::c_uint = 0;
                    let m = &msgs[0];
                    // SAFETY: EVP on its own objects; out holds 64 bytes.
                    unsafe {
                        (o.EVP_DigestInit_ex)(ctx, md, core::ptr::null_mut());
                        (o.EVP_DigestUpdate)(ctx, m.as_ptr(), m.len());
                        (o.EVP_DigestFinal_ex)(ctx, out.as_mut_ptr(), &mut n);
                    }
                    assert_eq!(out[..n as usize].to_vec(), $ours(m).to_vec(), concat!($op, " openssl evp"));
                }
                refs.push(side(&format!("{} EVP", o.name), "EVP_MD fetched once, EVP_MD_CTX reused", move |i| {
                    let m = &msgs[i % np];
                    let mut out = [0u8; 64];
                    let mut n: core::ffi::c_uint = 0;
                    // SAFETY: as above.
                    unsafe {
                        (o.EVP_DigestInit_ex)(ctx, md, core::ptr::null_mut());
                        (o.EVP_DigestUpdate)(ctx, m.as_ptr(), m.len());
                        (o.EVP_DigestFinal_ex)(ctx, out.as_mut_ptr(), &mut n);
                    }
                    black_box(out);
                }));
                if let Some(f) = o.$low {
                    refs.push(side(&format!("{} {}()", o.name, stringify!($low)), "the deprecated one-shot", move |i| {
                        let m = &msgs[i % np];
                        let mut out = [0u8; 64];
                        // SAFETY: as above.
                        unsafe { f(m.as_ptr(), m.len(), out.as_mut_ptr()) };
                        black_box(out);
                    }));
                }
            }
            b.run($op, lname, "ours: the SHA instructions where the CPU has them, else portable Rust", |i| {
                black_box($ours(&msgs[i % np]));
            }, refs);
        }};
    }
    one!("sha256", a::hash::sha256, sha2_11::Sha256, sha2_10::Sha256, Some(&ring::digest::SHA256), "SHA256", SHA256, 32);
    one!("sha224", a::hash::sha224, sha2_11::Sha224, sha2_10::Sha224, None::<&'static ring::digest::Algorithm>, "SHA224", SHA224, 28);
    one!("sha512", a::hash::sha512, sha2_11::Sha512, sha2_10::Sha512, Some(&ring::digest::SHA512), "SHA512", SHA512, 64);
    one!("sha384", a::hash::sha384, sha2_11::Sha384, sha2_10::Sha384, Some(&ring::digest::SHA384), "SHA384", SHA384, 48);
}

fn xmd_rows(b: &mut Bench) {
    use a::hash::{Sha256, Sha384, Sha512, expand_message_xmd};
    use elliptic_curve::hash2curve::{ExpandMsg, ExpandMsgXmd, Expander};
    let mut rng = Rng(77);
    let msgs: Vec<Vec<u8>> = (0..16).map(|_| rng.bytes(32)).collect();
    macro_rules! xmd {
        ($H:ty, $R:ty, $hname:literal, $dst:literal, $len:expr, $blst:expr) => {{
            let dst: &[u8] = $dst;
            let len: usize = $len;
            let rc = |m: &[u8]| {
                let mut out = vec![0u8; len];
                <ExpandMsgXmd<$R> as ExpandMsg>::expand_message(&[m], &[dst], len).unwrap().fill_bytes(&mut out);
                out
            };
            for m in msgs.iter().take(3) {
                let w = expand_message_xmd::<$H>(m, dst, len);
                assert_eq!(rc(m), w, "xmd elliptic-curve");
                if $blst {
                    let mut out = vec![0u8; len];
                    // SAFETY: blst writes len bytes.
                    unsafe { blst::blst_expand_message_xmd(out.as_mut_ptr(), len, m.as_ptr(), m.len(), dst.as_ptr(), dst.len()) };
                    assert_eq!(out, w, "xmd blst");
                }
            }
            let mut refs = vec![side("elliptic-curve 0.13.8", "ExpandMsgXmd over sha2 0.10.9 (portable)", |i| {
                black_box(rc(&msgs[i % 16]));
            })];
            if $blst {
                refs.push(side("blst 0.3.17", "blst_expand_message_xmd (SHA-256 only)", |i| {
                    let m = &msgs[i % 16];
                    let mut out = vec![0u8; len];
                    // SAFETY: blst writes len bytes.
                    unsafe { blst::blst_expand_message_xmd(out.as_mut_ptr(), len, m.as_ptr(), m.len(), dst.as_ptr(), dst.len()) };
                    black_box(out);
                }));
            }
            b.run(
                concat!("expand_message_xmd ", $hname),
                &format!("32-byte message, {}-byte tag, {} bytes out", dst.len(), len),
                "",
                |i| {
                    black_box(expand_message_xmd::<$H>(&msgs[i % 16], dst, len));
                },
                refs,
            );
        }};
    }
    for len in [32usize, 96, 128, 256] {
        xmd!(Sha256, sha2_10::Sha256, "SHA-256", b"QUUX-V01-CS02-with-expander-SHA256-128", len, true);
    }
    xmd!(Sha384, sha2_10::Sha384, "SHA-384", b"QUUX-V01-CS02-with-expander-SHA384-192", 144, false);
    for len in [96usize, 196] {
        xmd!(Sha512, sha2_10::Sha512, "SHA-512", b"QUUX-V01-CS02-with-expander-SHA512-256", len, false);
    }
}
