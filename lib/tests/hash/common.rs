//! What the tests of the hashes share: hexadecimal, the hashes by name, and the reading of the
//! data files.

use a::hash::{sha224, sha256, sha384, sha512};

/// Scales a test down when it runs under Miri.
pub(crate) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// The bytes in the hexadecimal `s`; `-` is no bytes, as CAVP's empty message is written.
pub(crate) fn unhex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    assert!(s.len().is_multiple_of(2), "odd hexadecimal {s}");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// The bytes in lowercase hexadecimal.
pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The digest of `data` by the hash named as CAVP and RFC 9380 name it (`SHA256`, …).
pub(crate) fn by_name(name: &str, data: &[u8]) -> Vec<u8> {
    match name {
        "SHA224" => sha224(data).to_vec(),
        "SHA256" => sha256(data).to_vec(),
        "SHA384" => sha384(data).to_vec(),
        "SHA512" => sha512(data).to_vec(),
        _ => panic!("no hash {name}"),
    }
}

/// The lines of a data file that are not comments or blank.
pub(crate) fn lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}
