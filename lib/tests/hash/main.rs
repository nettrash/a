//! Tests of `a::hash`: SHA-224, SHA-256, SHA-384 and SHA-512 against FIPS 180-4's examples,
//! NIST's CAVP vectors and OpenSSL; `expand_message_xmd` against RFC 9380's appendix K and an
//! independent implementation on Python's hashlib.

mod common;
mod sha2;
mod xmd;
