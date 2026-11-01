//! The curves in common use: each function builds its curve on the first call and gives the same
//! one afterwards. The parameters are those of SEC 2, FIPS 186 and RFC 7748.

use super::curve::Curve;
use super::domain::Domain;
use super::edwards::Edwards;
use super::montgomery::Montgomery;
use super::weierstrass::Weierstrass;
use crate::math::numbers::{ByteOrder, Number};
use std::sync::OnceLock;

/// The number in hexadecimal, little-endian, the order `Number`'s fastest paths take.
fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::LittleEndian)
        .unwrap_or_else(|_| unreachable!("the constants are valid hexadecimal"))
}

/// The curve built from constants checked by the tests.
fn built<C: Curve>(
    curve: Result<C, super::curve::CurveError>,
    g: (&str, &str),
    order: &str,
    cofactor: u8,
) -> Domain<C> {
    let curve = curve.unwrap_or_else(|_| unreachable!("the constants make a curve"));
    Domain::new(
        curve,
        &hex(g.0),
        &hex(g.1),
        &hex(order),
        &Number::new_from_u8(cofactor, ByteOrder::LittleEndian),
    )
    .unwrap_or_else(|_| unreachable!("the constants make a domain"))
}

/// secp256k1, of SEC 2: `y^2 = x^3 + 7` modulo `2^256 - 2^32 - 977`.
///
/// # Examples
///
/// ```
/// use a::math::ec::named;
/// use a::math::numbers::{ByteOrder, Number};
///
/// let curve = named::secp256k1();
/// let g = curve.generator();
/// assert_eq!(curve.mul_generator(&Number::new_from_u8(2, ByteOrder::LittleEndian)), g.double());
/// assert!((&g * curve.order()).is_identity());
/// ```
pub fn secp256k1() -> &'static Domain<Weierstrass> {
    static CURVE: OnceLock<Domain<Weierstrass>> = OnceLock::new();
    CURVE.get_or_init(|| {
        let p = hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        let n = hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
        // `beta` is a cube root of unity modulo `p`, and the matching `lambda`, a cube root of
        // unity modulo `n`, gives the short basis of the pairs `(a, b)`, `a + b lambda = 0`.
        let a1 = hex("3086D221A7D46BCDE86C90E49284EB15");
        let b1 = -hex("E4437ED6010E88286F547FA90ABFE4C3");
        let a2 = hex("114CA50F7A8E2F3F657C1108D9D44CFD8");
        let curve = Weierstrass::new(&hex("0"), &hex("7"), &p).map(|c| {
            c.with_endomorphism(
                &hex("7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE"),
                &n,
                [a1.clone(), b1, a2, a1.clone()],
            )
        });
        built(
            curve,
            (
                "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
                "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
            ),
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            1,
        )
    })
}

/// P-256 of FIPS 186, also secp256r1: `y^2 = x^3 - 3 x + b` modulo
/// `2^256 - 2^224 + 2^192 + 2^96 - 1`.
pub fn p256() -> &'static Domain<Weierstrass> {
    static CURVE: OnceLock<Domain<Weierstrass>> = OnceLock::new();
    CURVE.get_or_init(|| {
        let p = hex("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF");
        built(
            Weierstrass::new(
                &(&p - 3u8),
                &hex("5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B"),
                &p,
            ),
            (
                "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
                "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
            ),
            "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
            1,
        )
    })
}

/// P-384 of FIPS 186, also secp384r1: `y^2 = x^3 - 3 x + b` modulo
/// `2^384 - 2^128 - 2^96 + 2^32 - 1`.
pub fn p384() -> &'static Domain<Weierstrass> {
    static CURVE: OnceLock<Domain<Weierstrass>> = OnceLock::new();
    CURVE.get_or_init(|| {
        let p = hex(concat!(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF"
        ));
        built(
            Weierstrass::new(
                &(&p - 3u8),
                &hex(concat!(
                    "B3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE814112",
                    "0314088F5013875AC656398D8A2ED19D2A85C8EDD3EC2AEF"
                )),
                &p,
            ),
            (
                concat!(
                    "AA87CA22BE8B05378EB1C71EF320AD746E1D3B628BA79B98",
                    "59F741E082542A385502F25DBF55296C3A545E3872760AB7"
                ),
                concat!(
                    "3617DE4A96262C6F5D9E98BF9292DC29F8F41DBD289A147C",
                    "E9DA3113B5F0B8C00A60B1CE1D7E819D7A431D7C90EA0E5F"
                ),
            ),
            concat!(
                "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "C7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973"
            ),
            1,
        )
    })
}

/// P-521 of FIPS 186, also secp521r1: `y^2 = x^3 - 3 x + b` modulo `2^521 - 1`.
pub fn p521() -> &'static Domain<Weierstrass> {
    static CURVE: OnceLock<Domain<Weierstrass>> = OnceLock::new();
    CURVE.get_or_init(|| {
        let p = (Number::new_from_u8(1, ByteOrder::LittleEndian) << 521usize) - 1u8;
        built(
            Weierstrass::new(
                &(&p - 3u8),
                &hex(concat!(
                    "51953EB9618E1C9A1F929A21A0B68540EEA2DA725B99B315F3B8B48991",
                    "8EF109E156193951EC7E937B1652C0BD3BB1BF073573DF883D2C34F1EF",
                    "451FD46B503F00"
                )),
                &p,
            ),
            (
                concat!(
                    "C6858E06B70404E9CD9E3ECB662395B4429C648139053FB521F828AF60",
                    "6B4D3DBAA14B5E77EFE75928FE1DC127A2FFA8DE3348B3C1856A429BF9",
                    "7E7E31C2E5BD66"
                ),
                concat!(
                    "11839296A789A3BC0045C8A5FB42C7D1BD998F54449579B446817AFBD1",
                    "7273E662C97EE72995EF42640C550B9013FAD0761353C7086A272C2408",
                    "8BE94769FD16650"
                ),
            ),
            concat!(
                "1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "FFFFFFFA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6",
                "FB71E91386409"
            ),
            1,
        )
    })
}

/// The prime `2^255 - 19` of Curve25519 and Ed25519.
fn prime25519() -> Number {
    (Number::new_from_u8(1, ByteOrder::LittleEndian) << 255usize) - 19u8
}

/// The order of the prime subgroup of Curve25519 and Ed25519, `2^252 + ...`.
const ORDER25519: &str = "1000000000000000000000000000000014DEF9DEA2F79CD65812631A5CF5D3ED";

/// Curve25519, of RFC 7748: `y^2 = x^3 + 486662 x^2 + x` modulo `2^255 - 19`, with cofactor 8.
/// [`x_mul`](Montgomery::x_mul) is the X25519 function.
pub fn curve25519() -> &'static Domain<Montgomery> {
    static CURVE: OnceLock<Domain<Montgomery>> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(
            Montgomery::new(&hex("76D06"), &hex("1"), &prime25519()),
            (
                "9",
                "20AE19A1B8A086B4E01EDD2C7748D14C923D4D7E6D7C61B229E9C5A27ECED3D9",
            ),
            ORDER25519,
            8,
        )
    })
}

/// Ed25519's curve, of RFC 8032: `-x^2 + y^2 = 1 + d x^2 y^2` modulo `2^255 - 19`, with
/// `d = -121665 / 121666` and cofactor 8.
pub fn ed25519() -> &'static Domain<Edwards> {
    static CURVE: OnceLock<Domain<Edwards>> = OnceLock::new();
    CURVE.get_or_init(|| {
        let p = prime25519();
        built(
            Edwards::new(
                &(&p - 1u8),
                &hex("52036CEE2B6FFE738CC740797779E89800700A4D4141D8AB75EB4DCA135978A3"),
                &p,
            ),
            (
                "216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A",
                "6666666666666666666666666666666666666666666666666666666666666658",
            ),
            ORDER25519,
            8,
        )
    })
}
