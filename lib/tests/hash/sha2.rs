//! SHA-224, SHA-256, SHA-384 and SHA-512: FIPS 180-4's examples, NIST CAVP's short, long and
//! Monte Carlo vectors, every length to 300 bytes against OpenSSL, whole digests of every length
//! and of random messages in random pieces against Python's hashlib, and streaming in pieces.
//!
//! The compression functions a run takes are those of the machine: the SHA instructions where the
//! CPU has them (Apple's aarch64, x86_64 with the SHA extensions), the portable functions under
//! `RUSTFLAGS="--cfg a_baseline"` (where every run-time CPU check reports absent) and under Miri.
//! Run both ways, the same tests compare the instructions and the portable functions with the
//! same answers, and so with each other.

use super::common::{by_name, hex, lines, size, unhex};
use a::hash::{Digest, Sha224, Sha256, Sha384, Sha512, sha224, sha256, sha384, sha512};

/// The messages of FIPS 180-4's examples (NIST's "Cryptographic Standards and Guidelines:
/// Examples with Intermediate Values", SHA-2): one block, two blocks of SHA-256 and two blocks of
/// SHA-512.
const ABC: &[u8] = b"abc";
const TWO_256: &[u8] = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
const TWO_512: &[u8] = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

#[test]
fn fips_180_4_examples_give_their_digests() {
    let cases: [(&str, &[u8], &str); 12] = [
        (
            "SHA224",
            ABC,
            "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7",
        ),
        (
            "SHA224",
            TWO_256,
            "75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525",
        ),
        (
            "SHA224",
            TWO_512,
            "c97ca9a559850ce97a04a96def6d99a9e0e0e2ab14e6b8df265fc0b3",
        ),
        (
            "SHA256",
            ABC,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            "SHA256",
            TWO_256,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        (
            "SHA256",
            TWO_512,
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        ),
        (
            "SHA384",
            ABC,
            "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7",
        ),
        (
            "SHA384",
            TWO_256,
            "3391fdddfc8dc7393707a65b1b4709397cf8b1d162af05abfe8f450de5f36bc6b0455a8520bc4e6f5fe95b1fe3c8452b",
        ),
        (
            "SHA384",
            TWO_512,
            "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712fcc7c71a557e2db966c3e9fa91746039",
        ),
        (
            "SHA512",
            ABC,
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        ),
        (
            "SHA512",
            TWO_256,
            "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445",
        ),
        (
            "SHA512",
            TWO_512,
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
        ),
    ];
    for (name, msg, want) in cases {
        assert_eq!(hex(&by_name(name, msg)), want, "{name} of {msg:?}");
    }
}

#[test]
fn the_empty_message_has_the_known_digests() {
    assert_eq!(
        hex(&sha224(b"")),
        "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"
    );
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha384(b"")),
        "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b"
    );
    assert_eq!(
        hex(&sha512(b"")),
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    );
}

#[test]
fn a_million_a_gives_the_digests_of_fips_180_2() {
    // FIPS 180-2's third example of each hash, the message of a million "a".
    let n = size(1_000_000, 1000);
    let msg = vec![b'a'; n];
    if n < 1_000_000 {
        // Under Miri: only that pieces and one shot agree.
        let mut h = Sha512::new();
        for c in msg.chunks(333) {
            h.update(c);
        }
        assert_eq!(h.finalize(), sha512(&msg));
        return;
    }
    let want = [
        (
            "SHA224",
            "20794655980c91d8bbb4c1ea97618a4bf03f42581948b2ee4ee7ad67",
        ),
        (
            "SHA256",
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        ),
        (
            "SHA384",
            "9d0e1809716474cb086e834e310a4a1ced149e9c00f248527972cec5704c2a5b07b8b3dc38ecc4ebae97ddd87f3d8985",
        ),
        (
            "SHA512",
            "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b",
        ),
    ];
    for (name, d) in want {
        assert_eq!(hex(&by_name(name, &msg)), d, "{name}");
    }
}

#[test]
#[ignore = "hashes 1 GiB with each hash: minutes in the debug profile; run with --release --ignored"]
fn a_gibibyte_message_gives_the_known_digests() {
    // NIST's "extremely long message": 64 bytes repeated 2^24 times. Its SHA-256 is NIST's; the
    // others are OpenSSL 4.1's (`openssl dgst`).
    let block = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno";
    let chunk: Vec<u8> = block.iter().copied().cycle().take(64 << 14).collect();
    let mut h = (Sha224::new(), Sha256::new(), Sha384::new(), Sha512::new());
    for _ in 0..1 << 10 {
        h.0.update(&chunk);
        h.1.update(&chunk);
        h.2.update(&chunk);
        h.3.update(&chunk);
    }
    assert_eq!(
        hex(&h.0.finalize()),
        "b5989713ca4fe47a009f8621980b34e6d63ed3063b2a0a2c867d8a85"
    );
    assert_eq!(
        hex(&h.1.finalize()),
        "50e72a0e26442fe2552dc3938ac58658228c0cbfb1d2ca872ae435266fcd055e"
    );
    assert_eq!(
        hex(&h.2.finalize()),
        "5441235cc0235341ed806a64fb354742b5e5c02a3c5cb71b5f63fb793458d8fdae599c8cd8884943c04f11b31b89f023"
    );
    assert_eq!(
        hex(&h.3.finalize()),
        "b47c933421ea2db149ad6e10fce6c7f93d0752380180ffd7f4629a712134831d77be6091b819ed352c2967a2e2d4fa5050723c9630691f1a05a7281dbe6c1086"
    );
}

/// NIST CAVP's vectors in `data/sha_cavp.txt` (see `data/gen_sha_cavp.py`).
const CAVP: &str = include_str!("data/sha_cavp.txt");

#[test]
fn cavp_short_and_long_messages_give_their_digests() {
    let (mut short, mut long) = (0, 0);
    for line in lines(CAVP) {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "short" | "long" => {
                let msg = unhex(f[2]);
                assert_eq!(
                    hex(&by_name(f[1], &msg)),
                    f[3],
                    "{} {} bytes",
                    f[1],
                    msg.len()
                );
                if f[0] == "short" {
                    short += 1;
                } else {
                    long += 1;
                }
            }
            "monte" | "check" => {}
            _ => panic!("unknown line {line}"),
        }
    }
    assert_eq!((short, long), (44, 8));
}

#[test]
fn cavp_monte_carlo_checkpoints_are_reached() {
    // SHAVS, section 6.4: from MD0 = MD1 = MD2 = Seed, MD_i = SHA(MD_(i-3) || MD_(i-2) ||
    // MD_(i-1)) for i in 3..=1002; MD_1002 is the checkpoint and the next seed.
    let rounds = size(1000, 10);
    let mut checked = 0;
    let mut seed = Vec::new();
    for line in lines(CAVP) {
        let f: Vec<&str> = line.split(' ').collect();
        match f[0] {
            "monte" => seed = unhex(f[2]),
            "check" => {
                if cfg!(miri) && f[2] != "0" {
                    continue;
                }
                let mut md = [seed.clone(), seed.clone(), seed.clone()];
                for _ in 0..rounds {
                    let m = [md[0].as_slice(), &md[1], &md[2]].concat();
                    let next = by_name(f[1], &m);
                    md = [md[1].clone(), md[2].clone(), next];
                }
                if rounds == 1000 {
                    assert_eq!(hex(&md[2]), f[3], "{} checkpoint {}", f[1], f[2]);
                }
                seed = md[2].clone();
                checked += 1;
            }
            _ => {}
        }
    }
    assert_eq!(checked, if cfg!(miri) { 4 } else { 40 });
}

/// The message `m[i] = (i·167 + 13) mod 256` of `gen_sha_lengths.c`.
fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| ((i * 167 + 13) % 256) as u8).collect()
}

/// The first 8 bytes of each digest by OpenSSL, for every length (see
/// `data/gen_sha_lengths.c`).
const LENGTHS: &str = include_str!("data/sha_lengths.txt");

#[test]
fn every_length_up_to_300_bytes_matches_openssl() {
    let msg = pattern(300);
    let mut seen = 0;
    for line in lines(LENGTHS) {
        let f: Vec<&str> = line.split(' ').collect();
        let n: usize = f[0].parse().unwrap();
        if cfg!(miri) && !n.is_multiple_of(37) {
            continue;
        }
        let m = &msg[..n];
        let got = [
            hex(&sha224(m)[..8]),
            hex(&sha256(m)[..8]),
            hex(&sha384(m)[..8]),
            hex(&sha512(m)[..8]),
        ];
        assert_eq!(got, [f[1], f[2], f[3], f[4]], "length {n}");
        seen += 1;
    }
    assert_eq!(seen, if cfg!(miri) { 9 } else { 301 });
}

/// The SHA-256 of each set of messages' digests by each hash, concatenated, by Python's hashlib
/// (see `data/gen_sha_whole.py`).
const WHOLE: &str = include_str!("data/sha_whole.txt");

/// The answer of `WHOLE` for the set `set` of `count` messages under the hash `name`.
fn whole(set: &str, count: usize, name: &str) -> &'static str {
    lines(WHOLE)
        .map(|l| l.split(' ').collect::<Vec<_>>())
        .find(|f| f[0] == set && f[1] == count.to_string() && f[2] == name)
        .unwrap_or_else(|| panic!("no answer for {set} {count} {name}"))[3]
}

/// The SHA-256 of the digests of `msgs` by `H`, each computed by `digest`, concatenated.
fn digests_hashed<H: Digest>(
    msgs: &[Vec<u8>],
    mut digest: impl FnMut(&[u8]) -> H::Output,
) -> String {
    let mut all = Sha256::new();
    for m in msgs {
        all.update(digest(m).as_ref());
    }
    hex(&all.finalize())
}

#[test]
fn every_length_up_to_300_bytes_gives_the_whole_digests_of_hashlib() {
    // One shot (every whole block in one call of the compression function) and in pieces of 7
    // bytes (a block at a time from the partial block).
    let count = size(301, 71);
    let msg = pattern(count);
    let msgs: Vec<Vec<u8>> = (0..count).map(|n| msg[..n].to_vec()).collect();
    fn check<H: Digest>(msgs: &[Vec<u8>], name: &str) {
        let want = whole("lengths", msgs.len(), name);
        assert_eq!(
            digests_hashed::<H>(msgs, |m| H::digest(m)),
            want,
            "{name}, one shot"
        );
        assert_eq!(
            digests_hashed::<H>(msgs, |m| in_pieces::<H>(m, 7)),
            want,
            "{name}, pieces"
        );
    }
    check::<Sha224>(&msgs, "SHA224");
    check::<Sha256>(&msgs, "SHA256");
    check::<Sha384>(&msgs, "SHA384");
    check::<Sha512>(&msgs, "SHA512");
}

/// SplitMix64 (Steele, Lea and Flood, OOPSLA 2014), as `data/gen_sha_whole.py` draws its
/// messages.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// The `count` messages of `data/gen_sha_whole.py`, of 0 to 2,048 bytes.
fn random_messages(count: usize) -> Vec<Vec<u8>> {
    let mut rng = SplitMix(0x5348_4132);
    (0..count)
        .map(|_| {
            let len = (rng.next() % 2049) as usize;
            let mut m = Vec::with_capacity(len + 8);
            while m.len() < len {
                m.extend_from_slice(&rng.next().to_le_bytes());
            }
            m.truncate(len);
            m
        })
        .collect()
}

/// Feeds `msg` to `H` in random pieces of 0 to 699 bytes: within a block, across one, and runs
/// of whole blocks, from any offset.
fn in_random_pieces<H: Digest>(msg: &[u8], rng: &mut SplitMix) -> H::Output {
    let mut h = H::new();
    let mut at = 0;
    while at < msg.len() {
        let r = rng.next();
        let piece = if r.is_multiple_of(5) {
            0
        } else {
            (r >> 8) as usize % 700
        };
        let end = (at + piece).min(msg.len());
        h.update(&msg[at..end]);
        at = end;
    }
    h.finalize()
}

#[test]
fn random_messages_in_random_pieces_give_the_whole_digests_of_hashlib() {
    let msgs = random_messages(size(64, 4));
    fn check<H: Digest>(msgs: &[Vec<u8>], name: &str, splits: u64) {
        let want = whole("random", msgs.len(), name);
        assert_eq!(
            digests_hashed::<H>(msgs, |m| H::digest(m)),
            want,
            "{name}, one shot"
        );
        for seed in 0..splits {
            let mut rng = SplitMix(seed);
            let got = digests_hashed::<H>(msgs, |m| in_random_pieces::<H>(m, &mut rng));
            assert_eq!(got, want, "{name}, pieces of seed {seed}");
        }
    }
    let splits = size(8, 1) as u64;
    check::<Sha224>(&msgs, "SHA224", splits);
    check::<Sha256>(&msgs, "SHA256", splits);
    check::<Sha384>(&msgs, "SHA384", splits);
    check::<Sha512>(&msgs, "SHA512", splits);
}

/// Feeds `msg` to `H` in pieces of `piece` bytes, after an empty piece.
fn in_pieces<H: Digest>(msg: &[u8], piece: usize) -> H::Output {
    let mut h = H::new();
    h.update(&[]);
    for c in msg.chunks(piece) {
        h.update(c);
        h.update(&[]);
    }
    h.finalize()
}

#[test]
fn streaming_in_pieces_of_every_size_equals_one_shot() {
    let n = size(300, 70);
    let msg = pattern(n);
    let one = (sha224(&msg), sha256(&msg), sha384(&msg), sha512(&msg));
    for piece in 1..=n {
        assert_eq!(in_pieces::<Sha224>(&msg, piece), one.0, "{piece}");
        assert_eq!(in_pieces::<Sha256>(&msg, piece), one.1, "{piece}");
        assert_eq!(in_pieces::<Sha384>(&msg, piece), one.2, "{piece}");
        assert_eq!(in_pieces::<Sha512>(&msg, piece), one.3, "{piece}");
    }
}

#[test]
fn streaming_in_pieces_of_changing_sizes_equals_one_shot() {
    // Pieces of 0, 1, 2, … bytes, so that each lands at another offset in a block.
    let msg = pattern(size(3000, 400));
    let mut h = (Sha256::new(), Sha512::new());
    let (mut at, mut piece) = (0, 0);
    while at < msg.len() {
        let end = (at + piece).min(msg.len());
        h.0.update(&msg[at..end]);
        h.1.update(&msg[at..end]);
        at = end;
        piece = (piece + 1) % 131;
    }
    assert_eq!(h.0.finalize(), sha256(&msg));
    assert_eq!(h.1.finalize(), sha512(&msg));
}

#[test]
fn a_clone_continues_independently_of_its_original() {
    let msg = pattern(200);
    let mut a = Sha384::new();
    a.update(&msg[..77]);
    let mut b = a.clone();
    a.update(&msg[77..]);
    b.update(&msg[77..150]);
    assert_eq!(a.finalize(), sha384(&msg));
    assert_eq!(b.finalize(), sha384(&msg[..150]));
    let mut c = Sha224::default();
    c.update(&msg[..64]);
    let d = c.clone();
    assert_eq!(d.finalize(), sha224(&msg[..64]));
}

/// The digest by the trait alone.
fn generic<H: Digest>(data: &[u8]) -> Vec<u8> {
    let mut h = H::default();
    h.update(data);
    let d = h.finalize();
    assert_eq!(d.as_ref().len(), H::OUTPUT_BYTES);
    assert_eq!(d, H::digest(data));
    d.as_ref().to_vec()
}

#[test]
fn the_digest_trait_gives_what_the_hashers_give() {
    let msg = pattern(129);
    assert_eq!(generic::<Sha224>(&msg), sha224(&msg));
    assert_eq!(generic::<Sha256>(&msg), sha256(&msg));
    assert_eq!(generic::<Sha384>(&msg), sha384(&msg));
    assert_eq!(generic::<Sha512>(&msg), sha512(&msg));
    assert_eq!(
        [
            Sha224::OUTPUT_BYTES,
            Sha256::OUTPUT_BYTES,
            Sha384::OUTPUT_BYTES,
            Sha512::OUTPUT_BYTES
        ],
        [28, 32, 48, 64]
    );
    assert_eq!(
        [
            Sha224::BLOCK_BYTES,
            Sha256::BLOCK_BYTES,
            Sha384::BLOCK_BYTES,
            Sha512::BLOCK_BYTES
        ],
        [64, 64, 128, 128]
    );
}

#[test]
fn hashers_are_usable_in_constants() {
    // `new` is a `const fn`: a hasher can start in a constant.
    const H: Sha256 = Sha256::new();
    let mut h = H;
    h.update(b"abc");
    assert_eq!(h.finalize(), sha256(b"abc"));
}
