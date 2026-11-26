//! `expand_message_xmd`: RFC 9380's appendix K, an independent implementation on Python's hashlib
//! for the other hashes, lengths and tags, and the limits of section 5.3.1.

use super::common::{hex, lines, size, unhex};
use a::hash::{Digest, Sha224, Sha256, Sha384, Sha512, expand_message_xmd, sha256};

/// `expand_message_xmd` with the hash named as RFC 9380 names it.
fn xmd(name: &str, msg: &[u8], dst: &[u8], len: usize) -> Vec<u8> {
    match name {
        "SHA224" => expand_message_xmd::<Sha224>(msg, dst, len),
        "SHA256" => expand_message_xmd::<Sha256>(msg, dst, len),
        "SHA384" => expand_message_xmd::<Sha384>(msg, dst, len),
        "SHA512" => expand_message_xmd::<Sha512>(msg, dst, len),
        _ => panic!("no hash {name}"),
    }
}

/// `expand_message_xmd` with one hash.
type Expand = fn(&[u8], &[u8], usize) -> Vec<u8>;

/// Appendix K.1 to K.3, from `data/expand_xmd.txt` (see `data/gen_expand_xmd.py`).
const APPENDIX_K: &str = include_str!("data/expand_xmd.txt");

#[test]
fn appendix_k_vectors_for_sha256_and_sha512_are_reproduced() {
    let (mut hash, mut dst, mut msg, mut len) = (String::new(), String::new(), String::new(), 0);
    let mut count = 0;
    for line in lines(APPENDIX_K) {
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        match k {
            "hash" => hash = v.to_string(),
            "dst" => dst = v.to_string(),
            "dst_prime" => {
                // DST_prime is the tag, or SHA-256("H2C-OVERSIZE-DST-" || tag) for K.2's long
                // one, and its length.
                let want = unhex(v);
                let short = if dst.len() > 255 {
                    sha256(&[b"H2C-OVERSIZE-DST-".as_slice(), dst.as_bytes()].concat()).to_vec()
                } else {
                    dst.as_bytes().to_vec()
                };
                assert_eq!(want, [short.as_slice(), &[short.len() as u8]].concat());
            }
            "msg" => msg = v.to_string(),
            "len" => len = v.parse().unwrap(),
            "uniform" => {
                if cfg!(miri) && count % 5 != 0 {
                    count += 1;
                    continue;
                }
                let got = xmd(&hash, msg.as_bytes(), dst.as_bytes(), len);
                assert_eq!(hex(&got), v, "{hash} {dst:.20} {msg:.10} {len}");
                count += 1;
            }
            _ => panic!("unknown line {line}"),
        }
    }
    assert_eq!(count, 30);
}

/// Python's results, from `data/expand_extra.txt` (see `data/gen_expand_extra.py`).
const EXTRA: &str = include_str!("data/expand_extra.txt");

#[test]
fn every_sha2_agrees_with_an_independent_implementation() {
    // Tags of 255 bytes are used as they are and longer ones hashed first; the outputs go up to
    // the 255 blocks the RFC allows.
    let mut count = 0;
    for line in lines(EXTRA) {
        let f: Vec<&str> = line.split(' ').collect();
        let (ldst, lmsg, len): (usize, usize, usize) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        );
        if cfg!(miri) && len > 100 {
            continue;
        }
        let dst: Vec<u8> = (0..ldst).map(|i| ((7 * i + 3) % 256) as u8).collect();
        let msg: Vec<u8> = (0..lmsg).map(|i| ((13 * i + 5) % 256) as u8).collect();
        let out = xmd(f[0], &msg, &dst, len);
        assert_eq!(out.len(), len);
        assert_eq!(hex(&out[..len.min(16)]), f[4], "{line}");
        assert_eq!(hex(&sha256(&out)), f[5], "{line}");
        count += 1;
    }
    assert!(count >= if cfg!(miri) { 10 } else { 20 });
}

#[test]
fn a_tag_longer_than_255_bytes_is_replaced_by_its_hash() {
    // Section 5.3.3: DST = H("H2C-OVERSIZE-DST-" || DST) for a longer tag.
    let long = vec![b'x'; 300];
    let short = sha256(&[b"H2C-OVERSIZE-DST-".as_slice(), &long].concat());
    assert_eq!(
        expand_message_xmd::<Sha256>(b"m", &long, 40),
        expand_message_xmd::<Sha256>(b"m", &short, 40)
    );
    let short = a::hash::sha512(&[b"H2C-OVERSIZE-DST-".as_slice(), &long[..256]].concat());
    assert_eq!(
        expand_message_xmd::<Sha512>(b"", &long[..256], 7),
        expand_message_xmd::<Sha512>(b"", &short, 7)
    );
    // At 255 bytes the tag is its own.
    let t = vec![b'y'; 255];
    let hashed = sha256(&[b"H2C-OVERSIZE-DST-".as_slice(), &t].concat());
    assert_ne!(
        expand_message_xmd::<Sha256>(b"m", &t, 32),
        expand_message_xmd::<Sha256>(b"m", &hashed, 32)
    );
}

#[test]
fn outputs_of_every_length_are_as_long_as_asked_and_differ() {
    // The length is hashed into b_0, so no output is a prefix of another.
    let n = size(300, 40);
    let mut prev: Vec<u8> = Vec::new();
    for len in 1..=n {
        let out = expand_message_xmd::<Sha256>(b"msg", b"DST", len);
        assert_eq!(out.len(), len);
        if len > 1 {
            assert_ne!(&out[..prev.len()], &prev[..], "{len}");
        }
        prev = out;
    }
}

#[test]
fn no_bytes_are_given_for_a_length_of_zero() {
    assert!(expand_message_xmd::<Sha256>(b"abc", b"DST", 0).is_empty());
    assert!(expand_message_xmd::<Sha512>(b"", &[1; 300], 0).is_empty());
}

#[test]
fn the_largest_outputs_are_given_and_one_more_byte_panics() {
    assert_eq!(
        expand_message_xmd::<Sha256>(b"", b"D", 255 * 32).len(),
        8160
    );
    assert_eq!(
        expand_message_xmd::<Sha384>(b"", b"D", 255 * 48).len(),
        12240
    );
    assert_eq!(
        expand_message_xmd::<Sha512>(b"", b"D", 255 * 64).len(),
        16320
    );
    assert_eq!(
        expand_message_xmd::<Sha224>(b"", b"D", 255 * 28).len(),
        7140
    );
    // One byte past 255 blocks, the largest length of two bytes, one past it, and the largest
    // usize, for each hash.
    let hashes: [(usize, Expand); 4] = [
        (32, expand_message_xmd::<Sha256>),
        (48, expand_message_xmd::<Sha384>),
        (64, expand_message_xmd::<Sha512>),
        (28, expand_message_xmd::<Sha224>),
    ];
    for (b, f) in hashes {
        for len in [255 * b + 1, 65535, 65536, usize::MAX] {
            let r = std::panic::catch_unwind(|| f(b"", b"D", len));
            let msg = r.unwrap_err();
            let msg = msg.downcast_ref::<String>().unwrap();
            assert!(
                msg.contains("at most 65535 bytes and 255 blocks"),
                "{b} {len}: {msg}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "the domain separation tag must not be empty")]
fn an_empty_tag_panics() {
    let _ = expand_message_xmd::<Sha256>(b"abc", b"", 32);
}

/// A hash whose output is longer than its block, which `expand_message_xmd` refuses.
#[derive(Clone, Default)]
struct Wide(Sha512);

impl Digest for Wide {
    const OUTPUT_BYTES: usize = 64;
    const BLOCK_BYTES: usize = 32;
    type Output = [u8; 64];

    fn new() -> Wide {
        Wide(Sha512::new())
    }

    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    fn finalize(self) -> [u8; 64] {
        self.0.finalize()
    }
}

#[test]
#[should_panic(expected = "not longer than its block")]
fn a_hash_with_an_output_longer_than_its_block_panics() {
    let _ = expand_message_xmd::<Wide>(b"abc", b"DST", 32);
}

/// SHA-256 behind a wrapper, as a user's own `Digest` would be.
#[derive(Clone, Default)]
struct Wrapped(Sha256);

impl Digest for Wrapped {
    const OUTPUT_BYTES: usize = 32;
    const BLOCK_BYTES: usize = 64;
    type Output = [u8; 32];

    fn new() -> Wrapped {
        Wrapped(Sha256::new())
    }

    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    fn finalize(self) -> [u8; 32] {
        self.0.finalize()
    }
}

#[test]
fn a_digest_of_the_users_own_works_as_the_crates() {
    assert_eq!(
        expand_message_xmd::<Wrapped>(b"abc", b"QUUX", 100),
        expand_message_xmd::<Sha256>(b"abc", b"QUUX", 100)
    );
}
