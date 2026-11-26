[A](../../../README.md) › [Hash](../../../README.md#hash)

# SHA-2

`a::hash` computes the hash functions SHA-224, SHA-256, SHA-384 and SHA-512 of FIPS 180-4, and RFC 9380's `expand_message_xmd`, which [hashing to curves](../math/ec.md#hashing-to-curves) builds on: streaming hashers that take a message in pieces of any length, one-shot functions, and a trait over them for code that is generic over the hash.

**Nothing here is constant-time.** The hashers branch on the lengths of what they are given, and no care is taken against timing or cache attacks. They are for public data: checksums, test vectors, the messages of hashing to curves and of verifying signatures, not keys or other secrets that whoever can time them could learn.

| Item | What it is |
| --- | --- |
| `Sha256`, `Sha224` | Streaming hashers: `new()` (a `const fn`), `update(&[u8])`, `finalize()`, which gives `[u8; 32]` or `[u8; 28]`; `Default`, `Clone` (to fork a hash midway), `Debug` |
| `Sha512`, `Sha384` | The same, `finalize()` giving `[u8; 64]` or `[u8; 48]` |
| `sha224(&data)`, `sha256(&data)`, `sha384(&data)`, `sha512(&data)` | The digest of `data` in one call |
| `Digest` | The trait the four hashers implement: `OUTPUT_BYTES` (28, 32, 48, 64), `BLOCK_BYTES` (64 or 128), `type Output` (the byte array), `new`, `update`, `finalize`, `digest(&data)` in one call, and `after_zero_block()`, a hasher that has taken one block of zeros (`expand_message_xmd`'s `Z_pad`; the default feeds the zeros, the four hashers start from the state computed when the crate is built). A hash of the user's own can implement it |
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
- **The instructions.** On ARMv8 with FEAT_SHA256 (`SHA256H`, `SHA256H2`, `SHA256SU0`, `SHA256SU1`) a block of SHA-256 takes sixteen pairs of `SHA256H`/`SHA256H2` of four rounds each, the next four words of the schedule from `SHA256SU0` and `SHA256SU1`, and the sums of words and constants one group ahead; with FEAT_SHA512 a block of SHA-512 takes forty pairs of `SHA512H`/`SHA512H2` of two rounds. On x86_64 with the SHA extensions a block of SHA-256 takes thirty-two `SHA256RNDS2` of two rounds with `SHA256MSG1` and `SHA256MSG2` for the schedule. The targets that have the features at compile time (Apple's aarch64) call them directly; the others test the CPU at run time, as [binary fields](../math/field.md#binary-fields) do for carry-less products; Miri and the other targets take the portable functions, which keep the schedule as its last sixteen words and unroll the rounds eight at a time.
- A hasher holds the eight words, a partial block and the length so far. `update` fills the partial block and gives every whole block of its input to the compression function directly, without copying. `finalize` pads as section 5.1 says: a one bit, zeros, and the length in bits as a big-endian number of 64 bits (SHA-224, SHA-256) or 128 bits (SHA-384, SHA-512), which takes one block or two. The cost is one compression per block and one or two for the padding; the memory is one block.
- **`expand_message_xmd`** follows RFC 9380, section 5.3.1: `b₀ = H(Z_pad ‖ msg ‖ I2OSP(len, 2) ‖ 0 ‖ DST′)`, with `Z_pad` a block of zeros (whose state the four hashers know from the crate's build, [`Digest::after_zero_block`](https://docs.rs/nettrash-a/latest/a/hash/trait.Digest.html#method.after_zero_block), the default of which feeds the zeros) and `DST′ = DST ‖ I2OSP(len(DST), 1)`; then `b₁ = H(b₀ ‖ 1 ‖ DST′)` and `bᵢ = H((b₀ ⊕ bᵢ₋₁) ‖ i ‖ DST′)`, and the output is the first `len` bytes of `b₁ ‖ … ‖ b_ell`, `ell = ⌈len / OUTPUT_BYTES⌉`. A tag longer than 255 bytes is replaced by `H("H2C-OVERSIZE-DST-" ‖ DST)` (section 5.3.3). The message is hashed once, after the block of zeros, and then `ell` strings of about a block each. A length of 0 gives no bytes.

## Verified

- **FIPS 180-4's examples** (NIST's "Examples with Intermediate Values"): "abc", the message of 448 bits and that of 896 bits, under each of the four hashes, 12 digests; the empty message under each; a million times "a" under each, the third example of FIPS 180-2. NIST's "extremely long message", 64 bytes repeated 2²⁴ times (1 GiB), whose SHA-256 NIST gives and whose other digests are OpenSSL 4.1's (`openssl dgst`), in an ignored test run in the release profile: all four equal.
- **NIST CAVP** (CAVS 11.1, `shabytetestvectors.zip`): 44 short messages, at the 11 lengths about the block boundaries of each hash (0 to 3, 31, 32, 55 to 57, 63 and 64 bytes for SHA-224 and SHA-256; 0 to 3, 63, 64, 111 to 113, 127 and 128 for SHA-384 and SHA-512); the first two long messages of each, 8 (163 to 326 bytes); and the first 10 checkpoints of the Monte Carlo test of each, 1,000 chained hashes a checkpoint, 40 checkpoints. All equal. The selection is in `lib/tests/hash/data/sha_cavp.txt`, written by `gen_sha_cavp.py` beside it from NIST's files.
- **OpenSSL 4.1**: the messages of every length from 0 to 300 bytes under each hash. The test compares the first 8 bytes of each of the 1,204 digests (`sha_lengths.txt`, written by `gen_sha_lengths.c`); a probe compared them whole, 1,204 of 1,204 equal.
- **Streaming**: for each hash, a message of 300 bytes fed in pieces of every size from 1 to 300, with empty pieces between, gives the one-shot digest; so do pieces of changing sizes over 3,000 bytes, and clones taken midway.
- **RustCrypto's `sha2` 0.10.9**, in a probe outside the crate: every length from 0 to 2,000 bytes under each hash, and SHA-512 in random pieces at each length, 10,005 digests, all equal.
- **`expand_message_xmd`**: the 30 vectors of RFC 9380's appendix K.1, K.2 (a tag of 256 bytes) and K.3, and each section's `DST_prime`, that of K.2 being `SHA-256("H2C-OVERSIZE-DST-" ‖ DST)` and its length; 20 cases from an implementation written from the RFC's pseudocode on Python's `hashlib` (`gen_expand_extra.py`, which reproduces the 30 vectors of appendix K too): SHA-224, SHA-256, SHA-384 and SHA-512, tags of 1 to 5,000 bytes with 255 and 256 on either side of the limit, messages of 0 to 1,000 bytes, and outputs up to the largest, 255 blocks (7,140, 8,160, 12,240 and 16,320 bytes); and in the probe, RustCrypto's `elliptic-curve` 0.13.8 (`ExpandMsgXmd`) on 3,000 random messages, tags of 1 to 755 bytes and lengths up to 600 bytes and the largest, with SHA-256, SHA-384 and SHA-512: all equal. A byte past 255 blocks, 65,535 and 65,536 bytes and `usize::MAX` panic for each of the four hashes, as do an empty tag and a `Digest` whose output is longer than its block; a tag of 300 bytes gives what its hash, as the tag, gives.
- **The instructions and the portable functions, on the same inputs.** A run takes the compression functions of its machine: the instructions where the CPU has them, the portable functions with `RUSTFLAGS="--cfg a_baseline"` (which makes every run-time CPU check report absent, and the compile-time features of Apple's aarch64 unused) and under Miri. The tests run both ways against the same answers, so that each compares the two: besides the vectors above, the whole digests of the messages of every length from 0 to 300 bytes, one shot and in pieces of 7 bytes, and of 64 random messages of up to 2,048 bytes, one shot and in random pieces (empty, within a block, across blocks, runs of whole blocks) under 8 splittings, against Python's hashlib (`sha_whole.txt`, written by `gen_sha_whole.py`: the SHA-256 of each hash's digests concatenated, 1,204 and 256 digests). On Apple's aarch64 (M4 Max) both runs pass, the instructions of SHA-256 and SHA-512 against the portable functions; on x86_64 in the CI container both pass too, but its CPU has no SHA extensions, so that both take the portable SHA-256 there and the SHA extensions' function is run only on CPUs that have them.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 22 tests pass, and the ignored one is skipped.

To run the tests: `cargo test --test hash`, the same with `RUSTFLAGS="--cfg a_baseline"` (in a separate target directory) for the portable functions, and `cargo test --release --test hash -- --ignored` for the message of 1 GiB. The data files are written again by the scripts beside them: `python3 gen_sha_cavp.py <shabytetestvectors> .`, `cc gen_sha_lengths.c -lcrypto && ./a.out 8 > sha_lengths.txt`, `python3 gen_expand_xmd.py rfc9380.txt > expand_xmd.txt`, `python3 gen_expand_extra.py > expand_extra.txt` and `python3 gen_sha_whole.py > sha_whole.txt`.

## Limits

- Not constant-time, as said above.
- Messages are whole bytes: the CAVP vectors of lengths in bits that are not multiples of 8 are not supported. A message may be up to `2⁶¹ − 1` bytes for SHA-224 and SHA-256, `2¹²⁵ − 1` for SHA-384 and SHA-512, as FIPS 180-4 counts its length in 64 and 128 bits; `update` panics beyond, a limit no test can reach.
- Only these four: not SHA-512/224 or SHA-512/256, SHA-1 or SHA-3, and so not RFC 9380's `expand_message_xof` (SHAKE). There is no HMAC; the trait gives what one needs, the block length included.
- The SHA instructions are used where the CPU has them: SHA-256 and SHA-512 on ARMv8 (FEAT_SHA256 and FEAT_SHA512), SHA-256 on x86_64 (the SHA extensions, detected at run time); SHA-384 and SHA-512 are portable Rust on x86_64. The state after `Z_pad` is computed when the crate is built.

## Measured

Measured on an Apple M4 Max on 2026-11-26, between 06:22 and 06:23, with nothing else running on the machine, by the harness in the repository's `bench/curves` (its method in `bench/README.md` there). This crate is built as a user gets it, in the default release profile (opt-level 3, no LTO, no `target-cpu`) of rustc 1.100.0, so that on this machine it takes the ARMv8 SHA-256 and SHA-512 instructions; the Rust references are built in the same profile: RustCrypto's `sha2` 0.11.1 (the ARMv8 instructions, found at run time) and 0.10.9 (its default features, the portable code), ring 0.17.14, blst 0.3.17 and RustCrypto's `elliptic-curve` 0.13.8 (`ExpandMsgXmd` over `sha2` 0.10.9); OpenSSL 3.5.9 and 4.1.0 are Homebrew's, loaded at run time, through `EVP_Digest*` with the `EVP_MD` fetched once and its context reused, and through the one-shot `SHA256()` and the like. One thread. Every digest is compared with this crate's before the clock starts; then the sides alternate sample by sample in one process, and a figure is the median of 21 samples of at least 1 ms each. A cell gives this crate's time for a message of the length given and, in parentheses, that time over the time of the fastest reference, with its name: below 1, this crate is faster.

| Hash | 0 B | 64 B | 1 KiB | 16 KiB | 1 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sha256` | 23.2 ns (1.00× sha2 0.11) | 32.1 ns (0.921× sha2 0.11) | 358 ns (1.16× OpenSSL 4.1) | 5.66 µs (1.20× ring) | 362 µs (1.20× ring) |
| `sha224` | 22.0 ns (0.751× OpenSSL 4.1) | 33.0 ns (0.910× sha2 0.11) | 355 ns (1.16× OpenSSL 4.1) | 5.65 µs (1.19× OpenSSL 4.1) | 362 µs (1.20× OpenSSL 3.5) |
| `sha512` | 74.0 ns (1.26× sha2 0.11) | 68.7 ns (0.943× OpenSSL 4.1) | 569 ns (0.993× sha2 0.11) | 8.27 µs (0.999× sha2 0.11) | 526 µs (0.996× sha2 0.11) |
| `sha384` | 75.2 ns (1.24× sha2 0.11) | 69.2 ns (0.954× OpenSSL 4.1) | 570 ns (0.997× sha2 0.11) | 8.26 µs (0.997× sha2 0.11) | 527 µs (1.00× sha2 0.11) |

`expand_message_xmd` of a message of 32 bytes with a tag of 38, against blst's `blst_expand_message_xmd` (SHA-256 only) and `elliptic-curve`'s:

| Hash, bytes out | `expand_message_xmd` |
| --- | ---: |
| SHA-256, 32 | 123 ns (1.20× blst) |
| SHA-256, 96 | 255 ns (1.32× blst) |
| SHA-256, 128 | 318 ns (1.32× blst) |
| SHA-256, 256 | 582 ns (1.35× blst) |
| SHA-384, 144 | 394 ns (0.523× elliptic-curve) |
| SHA-512, 96 | 277 ns (0.468× elliptic-curve) |
| SHA-512, 196 | 450 ns (0.478× elliptic-curve) |

### Where the references are faster

Every case above 1.05 times the fastest reference, by cause:

- **SHA-256 and SHA-224 from 1 KiB** (1.16 to 1.20 times OpenSSL's and ring's): OpenSSL's SHA-256 is hand-scheduled ARMv8 assembly on the same instructions, and ring's the same code by way of BoringSSL; `sha2` 0.11's intrinsics, on those instructions too, run at this crate's speed (it takes 1.01 to 1.02 of their time there).
- **SHA-512 and SHA-384 of the empty message** (1.26 and 1.24 times `sha2` 0.11's): no established cause; a message of 64 bytes, also one block, takes 0.943 and 0.954 of the fastest reference's time.
- **`expand_message_xmd` with SHA-256** (1.20 to 1.35 times blst's): blst's SHA-256 is its own hand-written ARMv8 assembly, as OpenSSL's above; with SHA-384 and SHA-512, this crate takes 0.468 to 0.523 of `elliptic-curve`'s time.
