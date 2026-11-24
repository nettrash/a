//! `GF(2¹²⁸)` of GCM: `a::math::field::Gf2_128` and GHASH against the test cases of McGrew and
//! Viega's GCM specification (2005, appendix B), against OpenSSL's AES-GCM on random inputs
//! (data/gf2_128_openssl.txt, from data/gen_gf2_128.c), and against
//! `BinaryField::new(&[128, 7, 2, 1, 0])`.

use a::math::field::{BinaryField, Clmul, Gf2_128};

/// The bytes of a hexadecimal string.
fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// A block from 32 hexadecimal digits.
fn block(s: &str) -> Gf2_128 {
    Gf2_128::from_bytes(hex(s).try_into().unwrap())
}

/// splitmix64, as data/gen_gf2_128.c draws its inputs.
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

/// The additional data of test cases 4–6, 10–12 and 16–18.
const A: &str = "feedfacedeadbeeffeedfacedeadbeefabaddad2";
/// The long IV of test cases 6, 12 and 18.
const IV60: &str = "9313225df88406e555909c5aff5269aa6a7a9538534f7da1e4c303d2a318a728\
                    c3c0c95156809539fcf0e2429a6b525416aedbf5a0de6a57a637b39b";

/// McGrew and Viega, appendix B, test cases 1–18: H, A, C and GHASH(H, A, C).
const CASES: [(&str, &str, &str, &str); 18] = [
    (
        "66e94bd4ef8a2c3b884cfa59ca342b2e",
        "",
        "",
        "00000000000000000000000000000000",
    ),
    (
        "66e94bd4ef8a2c3b884cfa59ca342b2e",
        "",
        "0388dace60b6a392f328c2b971b2fe78",
        "f38cbb1ad69223dcc3457ae5b6b0f885",
    ),
    (
        "b83b533708bf535d0aa6e52980d53b78",
        "",
        "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e\
         21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985",
        "7f1b32b81b820d02614f8895ac1d4eac",
    ),
    (
        "b83b533708bf535d0aa6e52980d53b78",
        A,
        "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e\
         21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091",
        "698e57f70e6ecc7fd9463b7260a9ae5f",
    ),
    (
        "b83b533708bf535d0aa6e52980d53b78",
        A,
        "61353b4c2806934a777ff51fa22a4755699b2a714fcdc6f83766e5f97b6c7423\
         73806900e49f24b22b097544d4896b424989b5e1ebac0f07c23f4598",
        "df586bb4c249b92cb6922877e444d37b",
    ),
    (
        "b83b533708bf535d0aa6e52980d53b78",
        A,
        "8ce24998625615b603a033aca13fb894be9112a5c3a211a8ba262a3cca7e2ca7\
         01e4a9a4fba43c90ccdcb281d48c7c6fd62875d2aca417034c34aee5",
        "1c5afe9760d3932f3c9a878aac3dc3de",
    ),
    (
        "aae06992acbf52a3e8f4a96ec9300bd7",
        "",
        "",
        "00000000000000000000000000000000",
    ),
    (
        "aae06992acbf52a3e8f4a96ec9300bd7",
        "",
        "98e7247c07f0fe411c267e4384b0f600",
        "e2c63f0ac44ad0e02efa05ab6743d4ce",
    ),
    (
        "466923ec9ae682214f2c082badb39249",
        "",
        "3980ca0b3c00e841eb06fac4872a2757859e1ceaa6efd984628593b40ca1e19c\
         7d773d00c144c525ac619d18c84a3f4718e2448b2fe324d9ccda2710acade256",
        "51110d40f6c8fff0eb1ae33445a889f0",
    ),
    (
        "466923ec9ae682214f2c082badb39249",
        A,
        "3980ca0b3c00e841eb06fac4872a2757859e1ceaa6efd984628593b40ca1e19c\
         7d773d00c144c525ac619d18c84a3f4718e2448b2fe324d9ccda2710",
        "ed2ce3062e4a8ec06db8b4c490e8a268",
    ),
    (
        "466923ec9ae682214f2c082badb39249",
        A,
        "0f10f599ae14a154ed24b36e25324db8c566632ef2bbb34f8347280fc4507057\
         fddc29df9a471f75c66541d4d4dad1c9e93a19a58e8b473fa0f062f7",
        "1e6a133806607858ee80eaf237064089",
    ),
    (
        "466923ec9ae682214f2c082badb39249",
        A,
        "d27e88681ce3243c4830165a8fdcf9ff1de9a1d8e6b447ef6ef7b79828666e45\
         81e79012af34ddd9e2f037589b292db3e67c036745fa22e7e9b7373b",
        "82567fb0b4cc371801eadec005968e94",
    ),
    (
        "dc95c078a2408989ad48a21492842087",
        "",
        "",
        "00000000000000000000000000000000",
    ),
    (
        "dc95c078a2408989ad48a21492842087",
        "",
        "cea7403d4d606b6e074ec5d3baf39d18",
        "83de425c5edc5d498f382c441041ca92",
    ),
    (
        "acbef20579b4b8ebce889bac8732dad7",
        "",
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa\
         8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad",
        "4db870d37cb75fcb46097c36230d1612",
    ),
    (
        "acbef20579b4b8ebce889bac8732dad7",
        A,
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa\
         8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
        "8bd0c4d8aacd391e67cca447e8c38f65",
    ),
    (
        "acbef20579b4b8ebce889bac8732dad7",
        A,
        "c3762df1ca787d32ae47c13bf19844cbaf1ae14d0b976afac52ff7d79bba9de0\
         feb582d33934a4f0954cc2363bc73f7862ac430e64abe499f47c9b1f",
        "75a34288b8c68f811c52b2e9a2f97f63",
    ),
    (
        "acbef20579b4b8ebce889bac8732dad7",
        A,
        "5a8def2f0c9e53f1f75d7853659e2a20eeb2b22aafde6419a058ab4f6f746bf4\
         0fc0c3b780f244452da3ebf1c5d82cdea2418997200ef82e44ae7e3f",
        "d5ffcf6fc5ac4d69722187421a7f170b",
    ),
];

/// The cases whose IV is not of 96 bits, so that `Y₀ = GHASH(H, {}, IV)`: the case, its IV and
/// its `Y₀`.
const IV_CASES: [(usize, &str, &str); 6] = [
    (5, "cafebabefacedbad", "c43a83c4c4badec4354ca984db252f7d"),
    (6, IV60, "3bab75780a31c059f83d2a44752f9864"),
    (11, "cafebabefacedbad", "a14378078d27258a6292737e1802ada5"),
    (12, IV60, "4505cdc367a054c5002820e96aebef27"),
    (17, "cafebabefacedbad", "0095df49dd90abe3e4d252475748f5d4"),
    (18, IV60, "0cd953e2140a5976079f8e2406bc8eb4"),
];

#[test]
fn ghash_agrees_with_the_test_cases_of_the_gcm_specification() {
    for (i, (h, a, c, g)) in CASES.iter().enumerate() {
        let got = Gf2_128::gcm_ghash(block(h), &hex(a), &hex(c));
        assert_eq!(got, block(g), "test case {}", i + 1);
        // By either kernel.
        let h = block(h);
        let mut y = Gf2_128::ZERO;
        let mut data = hex(a);
        data.resize(hex(a).len().div_ceil(16) * 16, 0);
        let mut cc = hex(c);
        cc.resize(cc.len().div_ceil(16) * 16, 0);
        data.extend(cc);
        data.extend((hex(a).len() as u64 * 8).to_be_bytes());
        data.extend((hex(c).len() as u64 * 8).to_be_bytes());
        for b in data.chunks(16) {
            y = (y + Gf2_128::from_bytes(b.try_into().unwrap())).mul_with(h, Clmul::Portable);
        }
        assert_eq!(y, block(g), "test case {} by the portable kernel", i + 1);
        assert_eq!(Gf2_128::ghash(h, &data), block(g));
    }
    for (case, iv, y0) in IV_CASES {
        let h = block(CASES[case - 1].0);
        assert_eq!(
            Gf2_128::gcm_ghash(h, &[], &hex(iv)),
            block(y0),
            "Y₀ of test case {case}"
        );
    }
}

#[test]
fn each_step_of_ghash_agrees_with_test_case_4() {
    // X₁ … X₆ of test case 4: two blocks of A (the second padded), four of C (the last padded).
    let x = [
        "ed56aaf8a72d67049fdb9228edba1322",
        "cd47221ccef0554ee4bb044c88150352",
        "54f5e1b2b5a8f9525c23924751a3ca51",
        "324f585c6ffc1359ab371565d6c45f93",
        "ca7dd446af4aa70cc3c0cd5abba6aa1c",
        "1590df9b2eb6768289e57d56274c8570",
    ];
    let (h, a, c, _) = CASES[3];
    let h = block(h);
    let mut data = hex(a);
    data.resize(32, 0);
    let mut cc = hex(c);
    cc.resize(64, 0);
    data.extend(cc);
    let mut y = Gf2_128::ZERO;
    for (i, b) in data.chunks(16).enumerate() {
        y = (y + Gf2_128::from_bytes(b.try_into().unwrap())) * h;
        assert_eq!(y, block(x[i]), "X{}", i + 1);
    }
    // And the tag of test case 4: GHASH + E(K, Y₀).
    let tag = Gf2_128::gcm_ghash(h, &hex(a), &hex(c)) + block("3247184b3c4f69a44dbcd22887bbb418");
    assert_eq!(tag, block("5bc94fbc3221a5db94fae95ae7121a47"));
}

#[test]
fn ghash_agrees_with_openssl_aes_gcm_on_random_inputs() {
    let lines: Vec<&str> = include_str!("data/gf2_128_openssl.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect();
    assert_eq!(lines.len(), 64);
    for (i, line) in lines.iter().enumerate() {
        let (h, g) = line.split_once(' ').unwrap();
        let mut rng = SplitMix(0x6763_6d00_0000_0000 + i as u64);
        let mut byte = || rng.next() as u8;
        let _key: Vec<u8> = (0..16).map(|_| byte()).collect();
        let _iv: Vec<u8> = (0..12).map(|_| byte()).collect();
        let alen = (rng.next() % 70) as usize;
        let clen = (rng.next() % 90) as usize;
        let a: Vec<u8> = (0..alen).map(|_| rng.next() as u8).collect();
        let c: Vec<u8> = (0..clen).map(|_| rng.next() as u8).collect();
        assert_eq!(Gf2_128::gcm_ghash(block(h), &a, &c), block(g), "case {i}");
    }
}

#[test]
fn products_agree_with_the_binary_field_and_across_kernels() {
    let f = BinaryField::new(&[128, 7, 2, 1, 0]);
    let p = BinaryField::with_clmul(&[128, 7, 2, 1, 0], Clmul::Portable);
    let kernels: Vec<Clmul> = [Clmul::Portable, Clmul::Pmull, Clmul::Pclmulqdq]
        .into_iter()
        .filter(|k| k.is_available())
        .collect();
    let mut rng = SplitMix(9);
    let mut values: Vec<u128> = vec![0, 1, 2, 1 << 127, u128::MAX, 0x87];
    values.extend((0..200).map(|_| u128::from(rng.next()) | u128::from(rng.next()) << 64));
    let words = |v: u128| [v as u64, (v >> 64) as u64];
    for pair in values.windows(2) {
        let (a, b) = (
            Gf2_128::from_polynomial(pair[0]),
            Gf2_128::from_polynomial(pair[1]),
        );
        let want = &f.element(&words(pair[0])) * &f.element(&words(pair[1]));
        assert_eq!(words((a * b).polynomial()), want.words());
        assert_eq!(
            (&p.element(&words(pair[0])) * &p.element(&words(pair[1]))).words(),
            want.words()
        );
        for &k in &kernels {
            assert_eq!(a.mul_with(b, k), a * b, "{k:?}");
        }
        if !b.is_zero() {
            assert_eq!((a / b) * b, a);
            let inv = f.element(&words(pair[1])).inverse().unwrap();
            assert_eq!(words(b.inverse().unwrap().polynomial()), inv.words());
        }
        assert_eq!(
            words(a.square().polynomial()),
            f.element(&words(pair[0])).square().words()
        );
    }
}

#[test]
fn inverses_powers_and_the_bit_order_of_blocks() {
    let mut rng = SplitMix(10);
    for _ in 0..20 {
        let a = Gf2_128::from_polynomial(u128::from(rng.next()) << 64 | u128::from(rng.next()));
        assert_eq!(a * a.inverse().unwrap(), Gf2_128::ONE);
        assert_eq!(a.pow(u128::MAX), Gf2_128::ONE);
        assert_eq!(a.pow(u128::MAX - 1), a.inverse().unwrap());
        assert_eq!(a.pow(0), Gf2_128::ONE);
        assert_eq!(a.pow(5), a * a * a * a * a);
        assert_eq!(Gf2_128::from_bytes(a.to_bytes()), a);
    }
    assert_eq!(Gf2_128::ZERO.inverse(), None);
    assert_eq!(Gf2_128::ZERO.pow(0), Gf2_128::ONE);
    // Bit i of the polynomial is bit 7 − (i mod 8) of byte i / 8 of the block.
    for i in 0..128 {
        let b = Gf2_128::from_polynomial(1 << i).to_bytes();
        for (j, &byte) in b.iter().enumerate() {
            let want = if j == i / 8 { 0x80 >> (i % 8) } else { 0 };
            assert_eq!(byte, want);
        }
    }
    let mut x = block("feedfacedeadbeeffeedfacedeadbeef");
    let y = x;
    x += Gf2_128::ONE;
    x -= Gf2_128::ONE;
    x *= y;
    x /= y;
    assert_eq!(x, y);
    assert_eq!(-y, y);
    assert_eq!(
        format!("{y:?}"),
        "Gf2_128(feedfacedeadbeeffeedfacedeadbeef)"
    );
    assert_eq!(Gf2_128::default(), Gf2_128::ZERO);
}

#[test]
#[should_panic(expected = "whole blocks")]
fn ghash_of_a_partial_block_panics() {
    let _ = Gf2_128::ghash(Gf2_128::ONE, &[0; 17]);
}

#[test]
#[should_panic(expected = "division by zero")]
fn division_by_zero_panics() {
    let _ = Gf2_128::ONE / Gf2_128::ZERO;
}
