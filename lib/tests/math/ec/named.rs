//! The named curves against vectors from OpenSSL and curve25519-dalek, and from RFC 7748 and RFC 8032
//! as a separate implementation in Python reproduced them.

use super::common::{hex, hex_le, size};
use a::math::ec::{Curve, Domain, Weierstrass, named};
use a::math::numbers::{ByteOrder, Number};

/// Checks `k G` on a curve in the three ways it can be computed, and its compressed form.
fn check(curve: &Domain<Weierstrass>, k: &str, x: &str, y: &str, compressed: &str) {
    let k = hex(k);
    let want = curve.point(&hex(x), &hex(y)).unwrap();
    assert_eq!(curve.mul_generator(&k), want);
    assert_eq!(&curve.generator() * &k, want);
    assert_eq!(want.coordinates(), Some((hex(x), hex(y))));
    let bytes = want.compress();
    assert_eq!(hex(compressed).value, strip(&bytes));
    assert_eq!(curve.point_from_bytes(&bytes).unwrap(), want);
    assert_eq!(curve.point_from_bytes(&want.to_bytes()).unwrap(), want);
}

/// The bytes without the zeros they start with.
fn strip(b: &[u8]) -> Vec<u8> {
    let at = b.iter().position(|&x| x != 0).unwrap_or(b.len() - 1);
    b[at..].to_vec()
}

#[test]
fn secp256k1_matches_openssl() {
    let curve = named::secp256k1();
    check(
        curve,
        "02",
        "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",
        "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A",
        "02C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",
    );
    check(
        curve,
        "03",
        "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",
        "388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672",
        "02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",
    );
    check(
        curve,
        "0130557A9FC4E90E33587DA2C7EC11365B80A5CAEF14395E83A8CDF2173C6186",
        "8C9A2709DA5762EC97633321DB37D7AE7FC4FD7AC4AB70749EC17D7EAC211777",
        "705BE64FE16C66ABACDB1872A97F614B51291CA74A2E3A20476AD51AE9F55AFC",
        "028C9A2709DA5762EC97633321DB37D7AE7FC4FD7AC4AB70749EC17D7EAC211777",
    );
}

#[test]
fn p256_matches_openssl() {
    let curve = named::p256();
    check(
        curve,
        "02",
        "7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978",
        "07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1",
        "037CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978",
    );
    check(
        curve,
        "03",
        "5ECBE4D1A6330A44C8F7EF951D4BF165E6C6B721EFADA985FB41661BC6E7FD6C",
        "8734640C4998FF7E374B06CE1A64A2ECD82AB036384FB83D9A79B127A27D5032",
        "025ECBE4D1A6330A44C8F7EF951D4BF165E6C6B721EFADA985FB41661BC6E7FD6C",
    );
    check(
        curve,
        "0130557A9FC4E90E33587DA2C7EC11365B80A5CAEF14395E83A8CDF2173C6186",
        "72F141B602EC3E743D11C6E11073B0CA6C85EC092BFF862CE80A5A2E2EEE1DC4",
        "B76CCA3EC2F082BE81E220809E7E35A964F88C73C1B4EAF38912169F2DCCFFE0",
        "0272F141B602EC3E743D11C6E11073B0CA6C85EC092BFF862CE80A5A2E2EEE1DC4",
    );
}

#[test]
fn p384_matches_openssl() {
    let curve = named::p384();
    check(
        curve,
        "02",
        "08D999057BA3D2D969260045C55B97F089025959A6F434D651D207D19FB96E9E4FE0E86EBE0E64F85B96A9C75295DF61",
        "8E80F1FA5B1B3CEDB7BFE8DFFD6DBA74B275D875BC6CC43E904E505F256AB4255FFD43E94D39E22D61501E700A940E80",
        "0208D999057BA3D2D969260045C55B97F089025959A6F434D651D207D19FB96E9E4FE0E86EBE0E64F85B96A9C75295DF61",
    );
    check(
        curve,
        "03",
        "077A41D4606FFA1464793C7E5FDC7D98CB9D3910202DCD06BEA4F240D3566DA6B408BBAE5026580D02D7E5C70500C831",
        "C995F7CA0B0C42837D0BBE9602A9FC998520B41C85115AA5F7684C0EDC111EACC24ABD6BE4B5D298B65F28600A2F1DF1",
        "03077A41D4606FFA1464793C7E5FDC7D98CB9D3910202DCD06BEA4F240D3566DA6B408BBAE5026580D02D7E5C70500C831",
    );
    check(
        curve,
        "0130557A9FC4E90E33587DA2C7EC11365B80A5CAEF14395E83A8CDF2173C6186ABD0F51A3F6489AED3F81D42678CB1D6",
        "68592B3FD9C10D72D82C3C88E8D461622A40656216BFB06F5606D34486FD242595A221FC1DF678F4E622D4E57007026A",
        "9B0F3BA35EE9E185FDD6C8A68A376EB2D39A2087B23DD12F87F2389A5335F7BBBB143EA37B973F55EE3853F4157549B5",
        "0368592B3FD9C10D72D82C3C88E8D461622A40656216BFB06F5606D34486FD242595A221FC1DF678F4E622D4E57007026A",
    );
}

#[test]
fn p521_matches_openssl() {
    let curve = named::p521();
    check(
        curve,
        "02",
        "00433C219024277E7E682FCB288148C282747403279B1CCC06352C6E5505D769BE97B3B204DA6EF55507AA104A3A35C5AF41CF2FA364D60FD967F43E3933BA6D783D",
        "00F4BB8CC7F86DB26700A7F3ECEEEED3F0B5C6B5107C4DA97740AB21A29906C42DBBB3E377DE9F251F6B93937FA99A3248F4EAFCBE95EDC0F4F71BE356D661F41B02",
        "0200433C219024277E7E682FCB288148C282747403279B1CCC06352C6E5505D769BE97B3B204DA6EF55507AA104A3A35C5AF41CF2FA364D60FD967F43E3933BA6D783D",
    );
    check(
        curve,
        "03",
        "01A73D352443DE29195DD91D6A64B5959479B52A6E5B123D9AB9E5AD7A112D7A8DD1AD3F164A3A4832051DA6BD16B59FE21BAEB490862C32EA05A5919D2EDE37AD7D",
        "013E9B03B97DFA62DDD9979F86C6CAB814F2F1557FA82A9D0317D2F8AB1FA355CEEC2E2DD4CF8DC575B02D5ACED1DEC3C70CF105C9BC93A590425F588CA1EE86C0E5",
        "0301A73D352443DE29195DD91D6A64B5959479B52A6E5B123D9AB9E5AD7A112D7A8DD1AD3F164A3A4832051DA6BD16B59FE21BAEB490862C32EA05A5919D2EDE37AD7D",
    );
    check(
        curve,
        "0130557A9FC4E90E33587DA2C7EC11365B80A5CAEF14395E83A8CDF2173C6186ABD0F51A3F6489AED3F81D42678CB1D6FB20456A8FB4D9FE23486D92B7DC01264B70",
        "012315EAF1EA0437C1C2C3F94177A61FEFA91BBF3A7E33940A9510E0B224D2ADE5EB7440C963F52BDEA90BEF845CE4EA8E3B10139B025E43075036A039E63117C377",
        "01F3388D3BFDA09DF8823803EA64ABD6FAA8C11C71A36CF6EB1D65D9D82340592DE89728C5141BF9E852689E80CBB1ACE86B8E367C882445166AB9D53E573E7A673B",
        "03012315EAF1EA0437C1C2C3F94177A61FEFA91BBF3A7E33940A9510E0B224D2ADE5EB7440C963F52BDEA90BEF845CE4EA8E3B10139B025E43075036A039E63117C377",
    );
}

/// Whatever the curve, its generator has the order it is said to have, on all the paths.
fn group_order<C: Curve>(curve: &Domain<C>) {
    let g = curve.generator();
    let n = curve.order();
    assert!((&g * n).is_identity());
    assert!(curve.mul_generator(n).is_identity());
    let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
    assert_eq!(&g * &(n + &one), g);
    assert_eq!(&g * &(n - &one), -&g);
    assert_eq!(curve.mul_generator(&(n + &one)), g);
    assert_eq!(curve.mul_generator(&(n - &one)), -&g);
    assert!(curve.in_subgroup(&g));
}

// A test each, because together they run a 32-bit Miri out of addresses.
#[test]
fn the_generator_of_secp256k1_has_the_stated_order() {
    group_order(named::secp256k1());
}

#[test]
fn the_generator_of_p256_has_the_stated_order() {
    group_order(named::p256());
}

#[test]
fn the_generator_of_p384_has_the_stated_order() {
    group_order(named::p384());
}

#[test]
fn the_generator_of_p521_has_the_stated_order() {
    group_order(named::p521());
}

#[test]
fn the_generator_of_curve25519_has_the_stated_order() {
    group_order(named::curve25519());
}

#[test]
fn the_generator_of_ed25519_has_the_stated_order() {
    group_order(named::ed25519());
}

#[test]
fn the_same_curve_comes_back() {
    assert!(core::ptr::eq(named::p256(), named::p256()));
    assert!(core::ptr::eq(named::ed25519(), named::ed25519()));
}

#[test]
fn secp256k1_is_right_for_scalars_of_every_kind() {
    // The endomorphism's splitting, against the plain sum of doublings.
    let curve = named::secp256k1();
    let g = curve.generator();
    let (one, n) = (
        Number::new_from_u8(1, ByteOrder::LittleEndian),
        curve.order(),
    );
    let half = n >> 1usize;
    for k in [
        Number::new_from_u8(0, ByteOrder::LittleEndian),
        one.clone(),
        &half + &one,
        half.clone(),
        n - &one,
        n * 3u8 + 5u8,
        (Number::new_from_u8(1, ByteOrder::LittleEndian) << 128usize) - 1u8,
    ] {
        let mut sum = curve.identity();
        for i in (0..k.bits()).rev() {
            sum = sum.double();
            if k.bit(i) {
                sum += &g;
            }
        }
        assert_eq!(&g * &k, sum);
        assert_eq!(&g * &-&k, -sum);
    }
}

#[test]
fn j_invariants() {
    assert!(named::secp256k1().j_invariant().is_zero());
    assert_eq!(
        named::ed25519().j_invariant(),
        named::curve25519().j_invariant()
    );
}

#[test]
fn ed25519_matches_dalek() {
    let curve = named::ed25519();
    for (k, compressed) in [
        (
            2u64,
            "C9A3F86AAE465F0E56513864510F3997561FA2C9E85EA21DC2292309F3CD6022",
        ),
        (
            3,
            "D4B4F5784868C3020403246717EC169FF79E26608EA126A1AB69EE77D1B16712",
        ),
        (
            1234567890123456789,
            "47406A630C0BF6F0505A76FB3405573C3D740B4DD5758201737620A1EDCBAEAE",
        ),
    ] {
        let k = Number::new_from_u64(k, ByteOrder::LittleEndian);
        let bytes = curve.mul_generator(&k).compress();
        assert_eq!(hex(compressed).value, strip(&bytes));
        assert_eq!(
            curve.point_from_bytes(&bytes).unwrap(),
            &curve.generator() * &k
        );
    }
}

#[test]
fn ed25519_public_key_of_rfc_8032() {
    // The first test of RFC 8032's section 7.1: the secret's hash, clamped, gives this scalar, and
    // the public key is the compressed product.
    let curve = named::ed25519();
    let s = hex("4FE94D9006F020A5A3C080D96827FFFD3C010AC0F12E7A42CB33284F86837C30");
    let public = hex("D75A980182B10AB7D54BFED3C964073A0EE172F3DAA62325AF021A68F707511A");
    assert_eq!(curve.mul_generator(&s).compress(), public.value);
}

#[test]
fn edwards_and_montgomery_25519_are_the_same_group() {
    let edwards = named::ed25519();
    let montgomery = edwards.to_montgomery();
    let g = edwards.generator();
    let u = edwards.map_to_montgomery(&montgomery, &g);
    // RFC 7748's base point has `u = 9`; this form has another `B`, so `v` differs.
    assert_eq!(
        u.coordinates().unwrap().0,
        Number::new_from_u8(9, ByteOrder::BigEndian)
    );
    for (k, u_le) in [
        (
            2u64,
            "FB4E68DD9C46AE5C5C0B351EED5C3F8F1471157D680C75D9B7F17318D542D320",
        ),
        (
            3,
            "123C71FBAF030AC059081C62674E82F864BA1BC2914D5345E6AB576D1ABC121C",
        ),
        (
            1234567890123456789,
            "19D60AAEFEFB405A76038B6B6EB5C4A1D71788D9605D1D6A5D301628BBD94B7D",
        ),
    ] {
        let k = Number::new_from_u64(k, ByteOrder::LittleEndian);
        let image = edwards.map_to_montgomery(&montgomery, &edwards.mul_generator(&k));
        assert_eq!(image, &u * &k);
        assert_eq!(image.coordinates().unwrap().0, hex_le(u_le));
        let c = named::curve25519();
        let from_x = c.x_mul(&Number::new_from_u8(9, ByteOrder::LittleEndian), &k);
        assert_eq!(from_x, hex_le(u_le));
    }
}

/// X25519's clamping of a scalar in little-endian bytes.
fn clamped(k: &str) -> Number {
    let mut b = hex_le(k);
    b.byte_order = ByteOrder::LittleEndian;
    let mut bytes = b.value.clone();
    bytes.resize(32, 0);
    bytes[0] &= 248;
    bytes[31] &= 127;
    bytes[31] |= 64;
    Number::new(bytes, ByteOrder::LittleEndian)
}

#[test]
fn x25519_of_rfc_7748() {
    let curve = named::curve25519();
    for (k, u, out) in [
        (
            "A546E36BF0527C9D3B16154B82465EDD62144C0AC1FC5A18506A2244BA449AC4",
            "E6DB6867583030DB3594C1A424B15F7C726624EC26B3353B10A903A6D0AB1C4C",
            "C3DA55379DE9C6908E94EA4DF28D084F32ECCF03491C71F754B4075577A28552",
        ),
        (
            "4B66E9D4D1B4673C5AD22691957D6AF5C11B6421E0EA01D42CA4169E7918BA0D",
            "E5210F12786811D3F4B7959D0538AE2C31DBE7106FC03C3EFC4CD549C715A493",
            "95CBDE9476E8907D7AADE45CB4B873F88B595A68799FA152E6F8F7647AAC7957",
        ),
    ] {
        // Bit 255 of `u` is ignored, as the RFC says.
        let mut u = hex_le(u);
        if u.bit(255) {
            u.set_bit(255, false);
        }
        assert_eq!(curve.x_mul(&u, &clamped(k)), hex_le(out));
    }
}

#[test]
fn x25519_iterated() {
    // RFC 7748's section 5.2: k and u both 9, then the output as the next k and the k as the next
    // u.
    let curve = named::curve25519();
    let mut k = hex_le("0900000000000000000000000000000000000000000000000000000000000000");
    let mut u = k.clone();
    for _ in 0..size(1, 1) {
        let next = curve.x_mul(&u, &{
            let mut bytes = k.value.clone();
            bytes.resize(32, 0);
            bytes[0] &= 248;
            bytes[31] &= 127;
            bytes[31] |= 64;
            Number::new(bytes, ByteOrder::LittleEndian)
        });
        u = k;
        k = next;
    }
    assert_eq!(
        k,
        hex_le("422C8E7A6227D7BCA1350B3E2BB7279F7897B87BB6854B783C60E80311AE3079")
    );
}
