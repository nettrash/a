[A](../../../README.md) › [Hash](../../../README.md#hash)

# SHA-2

`a::hash` computes the hash functions SHA-224, SHA-256, SHA-384 and SHA-512 of FIPS 180-4, and RFC 9380's `expand_message_xmd`, which [hashing to curves](../math/ec.md#hashing-to-curves) builds on: streaming hashers that take a message in pieces of any length, one-shot functions, and a trait over them for code that is generic over the hash.

**Nothing here is constant-time.** The hashers branch on the lengths of what they are given, and no care is taken against timing or cache attacks. They are for public data: checksums, test vectors, the messages of hashing to curves and of verifying signatures, not keys or other secrets that whoever can time them could learn.

| Item | What it is |
| --- | --- |
| `Sha256`, `Sha224` | Streaming hashers: `new()` (a `const fn`), `update(&[u8])`, `finalize()`, which gives `[u8; 32]` or `[u8; 28]`; `Default`, `Clone` (to fork a hash midway), `Debug` |
| `Sha512`, `Sha384` | The same, `finalize()` giving `[u8; 64]` or `[u8; 48]` |
| `sha224(&data)`, `sha256(&data)`, `sha384(&data)`, `sha512(&data)` | The digest of `data` in one call |
| `Digest` | The trait the four hashers implement: `OUTPUT_BYTES` (28, 32, 48, 64), `BLOCK_BYTES` (64 or 128), `type Output` (the byte array), `new`, `update`, `finalize`, and `digest(&data)` in one call. A hash of the user's own can implement it |
| `expand_message_xmd::<H>(&msg, &dst, len)` | RFC 9380's `expand_message_xmd` (section 5.3.1) on any `H: Digest`: `len` uniform bytes from a message and a domain separation tag, a tag longer than 255 bytes first hashed as section 5.3.3 requires |

## Examples

```rust
use a::hash::{Digest, Sha256, Sha384, Sha512, expand_message_xmd, sha256, sha512};

// FIPS 180-4's example "abc", in one call, in pieces and through the trait.
let d = sha256(b"abc");
assert_eq!(d[..4], [0xba, 0x78, 0x16, 0xbf]);
let mut h = Sha256::new();
h.update(b"a");
h.update(b"bc");
assert_eq!(h.finalize(), d);
assert_eq!(<Sha256 as Digest>::digest(b"abc"), d);

// A clone continues on its own.
let mut a = Sha512::new();
a.update(b"common prefix, ");
let mut b = a.clone();
a.update(b"one");
b.update(b"two");
assert_eq!(a.finalize(), sha512(b"common prefix, one"));
assert_eq!(b.finalize(), sha512(b"common prefix, two"));

// Generic code over the hash.
fn hex<H: Digest>(data: &[u8]) -> String {
    H::digest(data).as_ref().iter().map(|x| format!("{x:02x}")).collect()
}
assert!(hex::<Sha384>(b"abc").starts_with("cb00753f45a35e8b"));

// RFC 9380, appendix K.1: 32 uniform bytes from "abc" with SHA-256.
let u = expand_message_xmd::<Sha256>(b"abc", b"QUUX-V01-CS02-with-expander-SHA256-128", 32);
assert_eq!(u[..4], [0xd8, 0xcc, 0xab, 0x23]);
assert_eq!(expand_message_xmd::<Sha512>(b"abc", b"DST", 100).len(), 100);
```

## Method

- **SHA-256** (FIPS 180-4, section 6.2) takes the message in blocks of 64 bytes, each expanded into a schedule of 64 words of 32 bits by `Wₜ = σ₁(Wₜ₋₂) + Wₜ₋₇ + σ₀(Wₜ₋₁₅) + Wₜ₋₁₆` and mixed into eight working words by 64 rounds of `Σ₀`, `Σ₁`, `Ch` and `Maj` with the round constants, the first 32 bits of the fractional parts of the cube roots of the first 64 primes. **SHA-224** is the same function from the initial value of section 5.3.2, its digest cut to the first seven words. **SHA-512** (section 6.4) is the same construction on 64-bit words, blocks of 128 bytes and 80 rounds; **SHA-384** starts from the initial value of section 5.3.4 and keeps six words. The constants in the source were checked against those computed exactly from integer square and cube roots of the primes.
- A hasher holds the eight words, a partial block and the length so far. `update` fills the partial block and gives every whole block of its input to the compression function directly, without copying. `finalize` pads as section 5.1 says: a one bit, zeros, and the length in bits as a big-endian number of 64 bits (SHA-224, SHA-256) or 128 bits (SHA-384, SHA-512), which takes one block or two. The cost is one compression per block and one or two for the padding; the memory is one block.
- **`expand_message_xmd`** follows RFC 9380, section 5.3.1: `b₀ = H(Z_pad ‖ msg ‖ I2OSP(len, 2) ‖ 0 ‖ DST′)`, with `Z_pad` a block of zeros and `DST′ = DST ‖ I2OSP(len(DST), 1)`; then `b₁ = H(b₀ ‖ 1 ‖ DST′)` and `bᵢ = H((b₀ ⊕ bᵢ₋₁) ‖ i ‖ DST′)`, and the output is the first `len` bytes of `b₁ ‖ … ‖ b_ell`, `ell = ⌈len / OUTPUT_BYTES⌉`. A tag longer than 255 bytes is replaced by `H("H2C-OVERSIZE-DST-" ‖ DST)` (section 5.3.3). The message is hashed once, after the block of zeros, and then `ell` strings of about a block each. A length of 0 gives no bytes.

## Verified

- **FIPS 180-4's examples** (NIST's "Examples with Intermediate Values"): "abc", the message of 448 bits and that of 896 bits, under each of the four hashes, 12 digests; the empty message under each; a million times "a" under each, the third example of FIPS 180-2. NIST's "extremely long message", 64 bytes repeated 2²⁴ times (1 GiB), whose SHA-256 NIST gives and whose other digests are OpenSSL 4.1's (`openssl dgst`), in an ignored test run in the release profile: all four equal.
- **NIST CAVP** (CAVS 11.1, `shabytetestvectors.zip`): 44 short messages, at the 11 lengths about the block boundaries of each hash (0 to 3, 31, 32, 55 to 57, 63 and 64 bytes for SHA-224 and SHA-256; 0 to 3, 63, 64, 111 to 113, 127 and 128 for SHA-384 and SHA-512); the first two long messages of each, 8 (163 to 326 bytes); and the first 10 checkpoints of the Monte Carlo test of each, 1,000 chained hashes a checkpoint, 40 checkpoints. All equal. The selection is in `lib/tests/hash/data/sha_cavp.txt`, written by `gen_sha_cavp.py` beside it from NIST's files.
- **OpenSSL 4.1**: the messages of every length from 0 to 300 bytes under each hash. The test compares the first 8 bytes of each of the 1,204 digests (`sha_lengths.txt`, written by `gen_sha_lengths.c`); a probe compared them whole, 1,204 of 1,204 equal.
- **Streaming**: for each hash, a message of 300 bytes fed in pieces of every size from 1 to 300, with empty pieces between, gives the one-shot digest; so do pieces of changing sizes over 3,000 bytes, and clones taken midway.
- **RustCrypto's `sha2` 0.10.9**, in a probe outside the crate: every length from 0 to 2,000 bytes under each hash, and SHA-512 in random pieces at each length, 10,005 digests, all equal.
- **`expand_message_xmd`**: the 30 vectors of RFC 9380's appendix K.1, K.2 (a tag of 256 bytes) and K.3, and each section's `DST_prime`, that of K.2 being `SHA-256("H2C-OVERSIZE-DST-" ‖ DST)` and its length; 20 cases from an implementation written from the RFC's pseudocode on Python's `hashlib` (`gen_expand_extra.py`, which reproduces the 30 vectors of appendix K too): SHA-224, SHA-256, SHA-384 and SHA-512, tags of 1 to 5,000 bytes with 255 and 256 on either side of the limit, messages of 0 to 1,000 bytes, and outputs up to the largest, 255 blocks (7,140, 8,160, 12,240 and 16,320 bytes); and in the probe, RustCrypto's `elliptic-curve` 0.13.8 (`ExpandMsgXmd`) on 3,000 random messages, tags of 1 to 755 bytes and lengths up to 600 bytes and the largest, with SHA-256, SHA-384 and SHA-512: all equal. A byte past 255 blocks, 65,535 and 65,536 bytes and `usize::MAX` panic for each of the four hashes, as do an empty tag and a `Digest` whose output is longer than its block; a tag of 300 bytes gives what its hash, as the tag, gives.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 20 tests pass, and the ignored one is skipped.

To run the tests: `cargo test --test hash`, and `cargo test --release --test hash -- --ignored` for the message of 1 GiB. The data files are written again by the scripts beside them: `python3 gen_sha_cavp.py <shabytetestvectors> .`, `cc gen_sha_lengths.c -lcrypto && ./a.out 8 > sha_lengths.txt`, `python3 gen_expand_xmd.py rfc9380.txt > expand_xmd.txt` and `python3 gen_expand_extra.py > expand_extra.txt`.

## Limits

- Not constant-time, as said above.
- Messages are whole bytes: the CAVP vectors of lengths in bits that are not multiples of 8 are not supported. A message may be up to `2⁶¹ − 1` bytes for SHA-224 and SHA-256, `2¹²⁵ − 1` for SHA-384 and SHA-512, as FIPS 180-4 counts its length in 64 and 128 bits; `update` panics beyond, a limit no test can reach.
- Only these four: not SHA-512/224 or SHA-512/256, SHA-1 or SHA-3, and so not RFC 9380's `expand_message_xof` (SHAKE). There is no HMAC; the trait gives what one needs, the block length included.
- The compression functions are portable Rust: no use is made of the SHA instructions of ARMv8 or of x86's SHA extensions, and the state after `Z_pad`, the same for every call of `expand_message_xmd` with one hash, is computed again each time. Nothing is timed yet.
