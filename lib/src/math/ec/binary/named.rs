//! The binary curves of SEC 2 ("Recommended elliptic curve domain parameters", version 1.0,
//! 2000), all 18 that OpenSSL knows, with NIST's names for the ten that FIPS 186 adopted: each
//! function builds its curve on the first call, checking that the generator is on the curve and
//! that the order times it is the point at infinity, and gives the same one afterwards.
//!
//! | SEC 2 | NIST | `m` | Kind | Cofactor |
//! | --- | --- | ---: | --- | ---: |
//! | sect113r1, sect113r2 | | 113 | random | 2 |
//! | sect131r1, sect131r2 | | 131 | random | 2 |
//! | sect163k1 | K-163 | 163 | Koblitz | 2 |
//! | sect163r1 | | 163 | random | 2 |
//! | sect163r2 | B-163 | 163 | random | 2 |
//! | sect193r1, sect193r2 | | 193 | random | 2 |
//! | sect233k1 | K-233 | 233 | Koblitz | 4 |
//! | sect233r1 | B-233 | 233 | random | 2 |
//! | sect239k1 | | 239 | Koblitz | 4 |
//! | sect283k1 | K-283 | 283 | Koblitz | 4 |
//! | sect283r1 | B-283 | 283 | random | 2 |
//! | sect409k1 | K-409 | 409 | Koblitz | 4 |
//! | sect409r1 | B-409 | 409 | random | 2 |
//! | sect571k1 | K-571 | 571 | Koblitz | 4 |
//! | sect571r1 | B-571 | 571 | random | 2 |
//!
//! The parameters are SEC 2's, which the tests compare with OpenSSL's.
//!
//! # Examples
//!
//! ```
//! use a::math::ec::binary::named;
//! use a::math::numbers::Number;
//!
//! let curve = named::by_name("K-163").unwrap();
//! assert!(core::ptr::eq(curve, named::sect163k1()));
//! assert_eq!(curve.field().degree(), 163);
//! let g = curve.generator();
//! assert_eq!(curve.mul_generator(&Number::from(2u8)), g.double());
//! assert_eq!(named::NAMES.len(), 18);
//! ```

use super::{BinaryCurve, BinaryDomain};
use crate::math::field::BinaryField;
use crate::math::numbers::Number;
use std::sync::OnceLock;

/// The names of the curves, as SEC 2 gives them, in its order.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert_eq!(named::NAMES[4], "sect163k1");
/// let koblitz = named::NAMES.iter().filter(|n| named::by_name(n).unwrap().is_koblitz());
/// assert_eq!(koblitz.count(), 6);
/// ```
pub const NAMES: [&str; 18] = [
    "sect113r1",
    "sect113r2",
    "sect131r1",
    "sect131r2",
    "sect163k1",
    "sect163r1",
    "sect163r2",
    "sect193r1",
    "sect193r2",
    "sect233k1",
    "sect233r1",
    "sect239k1",
    "sect283k1",
    "sect283r1",
    "sect409k1",
    "sect409r1",
    "sect571k1",
    "sect571r1",
];

/// The curve of the name, SEC 2's (`"sect233k1"`) or NIST's (`"K-233"`), in any case, or `None`
/// for another name.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::by_name("b-233").unwrap(), named::sect233r1()));
/// assert!(named::by_name("sect233").is_none());
/// ```
pub fn by_name(name: &str) -> Option<&'static BinaryDomain> {
    let name = name.to_ascii_lowercase();
    Some(match name.as_str() {
        "sect113r1" => sect113r1(),
        "sect113r2" => sect113r2(),
        "sect131r1" => sect131r1(),
        "sect131r2" => sect131r2(),
        "sect163k1" | "k-163" => sect163k1(),
        "sect163r1" => sect163r1(),
        "sect163r2" | "b-163" => sect163r2(),
        "sect193r1" => sect193r1(),
        "sect193r2" => sect193r2(),
        "sect233k1" | "k-233" => sect233k1(),
        "sect233r1" | "b-233" => sect233r1(),
        "sect239k1" => sect239k1(),
        "sect283k1" | "k-283" => sect283k1(),
        "sect283r1" | "b-283" => sect283r1(),
        "sect409k1" | "k-409" => sect409k1(),
        "sect409r1" | "b-409" => sect409r1(),
        "sect571k1" | "k-571" => sect571k1(),
        "sect571r1" | "b-571" => sect571r1(),
        _ => return None,
    })
}

/// The parameters of a curve: the exponents of the field's polynomial, and the elements and numbers
/// in hexadecimal, most significant digit first, the elements' bits those of the polynomial basis.
struct Spec {
    /// The exponents of the polynomial of the field.
    modulus: &'static [u32],
    /// `a`.
    a: &'static str,
    /// `b`.
    b: &'static str,
    /// The generator's `x`.
    x: &'static str,
    /// The generator's `y`.
    y: &'static str,
    /// The order of the generator.
    order: &'static str,
    /// The cofactor.
    cofactor: u8,
}

/// The words of the hexadecimal `s`, the least significant first.
fn words(s: &str) -> Vec<u64> {
    let digits = s.as_bytes();
    digits
        .rchunks(16)
        .map(|c| {
            let chunk = core::str::from_utf8(c).unwrap_or_else(|_| unreachable!("ASCII"));
            u64::from_str_radix(chunk, 16).unwrap_or_else(|_| unreachable!("hexadecimal"))
        })
        .collect()
}

/// The domain of the parameters, which the tests check.
fn built(s: &Spec) -> BinaryDomain {
    let f = BinaryField::new(s.modulus);
    let curve = BinaryCurve::new(&f.element(&words(s.a)), &f.element(&words(s.b)))
        .unwrap_or_else(|_| unreachable!("the constants make a curve"));
    let order = Number::from_str_radix(s.order, 16, crate::math::numbers::ByteOrder::LittleEndian)
        .unwrap_or_else(|_| unreachable!("the constants are valid hexadecimal"));
    BinaryDomain::new(
        curve,
        &f.element(&words(s.x)),
        &f.element(&words(s.y)),
        &order,
        &Number::from(s.cofactor),
    )
    .unwrap_or_else(|_| unreachable!("the constants make a domain"))
}

/// sect113r1 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹¹³)`, the field of `x¹¹³ + x⁹ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect113r1();
/// assert_eq!(curve.field().modulus(), [113, 9, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect113r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[113, 9, 0],
            a: "3088250CA6E7C7FE649CE85820F7",
            b: "E8BEE4D3E2260744188BE0E9C723",
            x: "9D73616F35F4AB1407D73562C10F",
            y: "A52830277958EE84D1315ED31886",
            order: "100000000000000D9CCEC8A39E56F",
            cofactor: 2,
        })
    })
}

/// sect113r2 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹¹³)`, the field of `x¹¹³ + x⁹ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect113r2();
/// assert_eq!(curve.field().modulus(), [113, 9, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect113r2() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[113, 9, 0],
            a: "689918DBEC7E5A0DD6DFC0AA55C7",
            b: "95E9A9EC9B297BD4BF36E059184F",
            x: "1A57A6A7B26CA5EF52FCDB8164797",
            y: "B3ADC94ED1FE674C06E695BABA1D",
            order: "10000000000000108789B2496AF93",
            cofactor: 2,
        })
    })
}

/// sect131r1 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹³¹)`, the field of `x¹³¹ + x⁸ + x³ + x² + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect131r1();
/// assert_eq!(curve.field().modulus(), [131, 8, 3, 2, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect131r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[131, 8, 3, 2, 0],
            a: "7A11B09A76B562144418FF3FF8C2570B8",
            b: "217C05610884B63B9C6C7291678F9D341",
            x: "81BAF91FDF9833C40F9C181343638399",
            y: "78C6E7EA38C001F73C8134B1B4EF9E150",
            order: "400000000000000023123953A9464B54D",
            cofactor: 2,
        })
    })
}

/// sect131r2 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹³¹)`, the field of `x¹³¹ + x⁸ + x³ + x² + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect131r2();
/// assert_eq!(curve.field().modulus(), [131, 8, 3, 2, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect131r2() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[131, 8, 3, 2, 0],
            a: "3E5A88919D7CAFCBF415F07C2176573B2",
            b: "4B8266A46C55657AC734CE38F018F2192",
            x: "356DCD8F2F95031AD652D23951BB366A8",
            y: "648F06D867940A5366D9E265DE9EB240F",
            order: "400000000000000016954A233049BA98F",
            cofactor: 2,
        })
    })
}

/// sect163k1 of SEC 2, NIST's K-163 (FIPS 186): the Koblitz curve `y² + x·y = x³ + x² + 1` over
/// `GF(2¹⁶³)`, the field of `x¹⁶³ + x⁷ + x⁶ + x³ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect163k1();
/// assert_eq!(curve.field().modulus(), [163, 7, 6, 3, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// assert_eq!(curve.koblitz_order().unwrap(), curve.order() * curve.cofactor());
/// ```
pub fn sect163k1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[163, 7, 6, 3, 0],
            a: "1",
            b: "1",
            x: "2FE13C0537BBC11ACAA07D793DE4E6D5E5C94EEE8",
            y: "289070FB05D38FF58321F2E800536D538CCDAA3D9",
            order: "4000000000000000000020108A2E0CC0D99F8A5EF",
            cofactor: 2,
        })
    })
}

/// sect163r1 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹⁶³)`, the field of `x¹⁶³ + x⁷ + x⁶ + x³ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect163r1();
/// assert_eq!(curve.field().modulus(), [163, 7, 6, 3, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect163r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[163, 7, 6, 3, 0],
            a: "7B6882CAAEFA84F9554FF8428BD88E246D2782AE2",
            b: "713612DCDDCB40AAB946BDA29CA91F73AF958AFD9",
            x: "369979697AB43897789566789567F787A7876A654",
            y: "435EDB42EFAFB2989D51FEFCE3C80988F41FF883",
            order: "3FFFFFFFFFFFFFFFFFFFF48AAB689C29CA710279B",
            cofactor: 2,
        })
    })
}

/// sect163r2 of SEC 2, NIST's B-163 (FIPS 186): one of its verifiably random curves,
/// `y² + x·y = x³ + x² + b`, over `GF(2¹⁶³)`, the field of `x¹⁶³ + x⁷ + x⁶ + x³ + 1`, with cofactor
/// 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect163r2();
/// assert_eq!(curve.field().modulus(), [163, 7, 6, 3, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect163r2() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[163, 7, 6, 3, 0],
            a: "1",
            b: "20A601907B8C953CA1481EB10512F78744A3205FD",
            x: "3F0EBA16286A2D57EA0991168D4994637E8343E36",
            y: "D51FBC6C71A0094FA2CDD545B11C5C0C797324F1",
            order: "40000000000000000000292FE77E70C12A4234C33",
            cofactor: 2,
        })
    })
}

/// sect193r1 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹⁹³)`, the field of `x¹⁹³ + x¹⁵ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect193r1();
/// assert_eq!(curve.field().modulus(), [193, 15, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect193r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[193, 15, 0],
            a: "17858FEB7A98975169E171F77B4087DE098AC8A911DF7B01",
            b: "FDFB49BFE6C3A89FACADAA7A1E5BBC7CC1C2E5D831478814",
            x: "1F481BC5F0FF84A74AD6CDF6FDEF4BF6179625372D8C0C5E1",
            y: "25E399F2903712CCF3EA9E3A1AD17FB0B3201B6AF7CE1B05",
            order: "1000000000000000000000000C7F34A778F443ACC920EBA49",
            cofactor: 2,
        })
    })
}

/// sect193r2 of SEC 2: one of its verifiably random curves, `y² + x·y = x³ + a·x² + b`, over
/// `GF(2¹⁹³)`, the field of `x¹⁹³ + x¹⁵ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect193r2();
/// assert_eq!(curve.field().modulus(), [193, 15, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect193r2() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[193, 15, 0],
            a: "163F35A5137C2CE3EA6ED8667190B0BC43ECD69977702709B",
            b: "C9BB9E8927D4D64C377E2AB2856A5B16E3EFB7F61D4316AE",
            x: "D9B67D192E0367C803F39E1A7E82CA14A651350AAE617E8F",
            y: "1CE94335607C304AC29E7DEFBD9CA01F596F927224CDECF6C",
            order: "10000000000000000000000015AAB561B005413CCD4EE99D5",
            cofactor: 2,
        })
    })
}

/// sect233k1 of SEC 2, NIST's K-233 (FIPS 186): the Koblitz curve `y² + x·y = x³ + 1` over
/// `GF(2²³³)`, the field of `x²³³ + x⁷⁴ + 1`, with cofactor 4.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect233k1();
/// assert_eq!(curve.field().modulus(), [233, 74, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// assert_eq!(curve.koblitz_order().unwrap(), curve.order() * curve.cofactor());
/// ```
pub fn sect233k1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[233, 74, 0],
            a: "0",
            b: "1",
            x: "17232BA853A7E731AF129F22FF4149563A419C26BF50A4C9D6EEFAD6126",
            y: "1DB537DECE819B7F70F555A67C427A8CD9BF18AEB9B56E0C11056FAE6A3",
            order: "8000000000000000000000000000069D5BB915BCD46EFB1AD5F173ABDF",
            cofactor: 4,
        })
    })
}

/// sect233r1 of SEC 2, NIST's B-233 (FIPS 186): one of its verifiably random curves,
/// `y² + x·y = x³ + x² + b`, over `GF(2²³³)`, the field of `x²³³ + x⁷⁴ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect233r1();
/// assert_eq!(curve.field().modulus(), [233, 74, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect233r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[233, 74, 0],
            a: "1",
            b: "66647EDE6C332C7F8C0923BB58213B333B20E9CE4281FE115F7D8F90AD",
            x: "FAC9DFCBAC8313BB2139F1BB755FEF65BC391F8B36F8F8EB7371FD558B",
            y: "1006A08A41903350678E58528BEBF8A0BEFF867A7CA36716F7E01F81052",
            order: "1000000000000000000000000000013E974E72F8A6922031D2603CFE0D7",
            cofactor: 2,
        })
    })
}

/// sect239k1 of SEC 2: the Koblitz curve `y² + x·y = x³ + 1` over `GF(2²³⁹)`, the field of
/// `x²³⁹ + x¹⁵⁸ + 1`, with cofactor 4.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect239k1();
/// assert_eq!(curve.field().modulus(), [239, 158, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// assert_eq!(curve.koblitz_order().unwrap(), curve.order() * curve.cofactor());
/// ```
pub fn sect239k1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[239, 158, 0],
            a: "0",
            b: "1",
            x: "29A0B6A887A983E9730988A68727A8B2D126C44CC2CC7B2A6555193035DC",
            y: "76310804F12E549BDB011C103089E73510ACB275FC312A5DC6B76553F0CA",
            order: "2000000000000000000000000000005A79FEC67CB6E91F1C1DA800E478A5",
            cofactor: 4,
        })
    })
}

/// sect283k1 of SEC 2, NIST's K-283 (FIPS 186): the Koblitz curve `y² + x·y = x³ + 1` over
/// `GF(2²⁸³)`, the field of `x²⁸³ + x¹² + x⁷ + x⁵ + 1`, with cofactor 4.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect283k1();
/// assert_eq!(curve.field().modulus(), [283, 12, 7, 5, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// assert_eq!(curve.koblitz_order().unwrap(), curve.order() * curve.cofactor());
/// ```
pub fn sect283k1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[283, 12, 7, 5, 0],
            a: "0",
            b: "1",
            x: concat!(
                "503213F78CA44883F1A3B8162F188E553CD265F23C1567A16876913B0C2A",
                "C2458492836"
            ),
            y: concat!(
                "1CCDA380F1C9E318D90F95D07E5426FE87E45C0E8184698E45962364E341",
                "16177DD2259"
            ),
            order: concat!(
                "1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE9AE2ED07577265DFF7F94451",
                "E061E163C61"
            ),
            cofactor: 4,
        })
    })
}

/// sect283r1 of SEC 2, NIST's B-283 (FIPS 186): one of its verifiably random curves,
/// `y² + x·y = x³ + x² + b`, over `GF(2²⁸³)`, the field of `x²⁸³ + x¹² + x⁷ + x⁵ + 1`, with
/// cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect283r1();
/// assert_eq!(curve.field().modulus(), [283, 12, 7, 5, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect283r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[283, 12, 7, 5, 0],
            a: "1",
            b: concat!(
                "27B680AC8B8596DA5A4AF8A19A0303FCA97FD7645309FA2A581485AF6263",
                "E313B79A2F5"
            ),
            x: concat!(
                "5F939258DB7DD90E1934F8C70B0DFEC2EED25B8557EAC9C80E2E198F8CDB",
                "ECD86B12053"
            ),
            y: concat!(
                "3676854FE24141CB98FE6D4B20D02B4516FF702350EDDB0826779C813F0D",
                "F45BE8112F4"
            ),
            order: concat!(
                "3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEF90399660FC938A90165B042",
                "A7CEFADB307"
            ),
            cofactor: 2,
        })
    })
}

/// sect409k1 of SEC 2, NIST's K-409 (FIPS 186): the Koblitz curve `y² + x·y = x³ + 1` over
/// `GF(2⁴⁰⁹)`, the field of `x⁴⁰⁹ + x⁸⁷ + 1`, with cofactor 4.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect409k1();
/// assert_eq!(curve.field().modulus(), [409, 87, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// assert_eq!(curve.koblitz_order().unwrap(), curve.order() * curve.cofactor());
/// ```
pub fn sect409k1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[409, 87, 0],
            a: "0",
            b: "1",
            x: concat!(
                "60F05F658F49C1AD3AB1890F7184210EFD0987E307C84C27ACCFB8F9F67C",
                "C2C460189EB5AAAA62EE222EB1B35540CFE9023746"
            ),
            y: concat!(
                "1E369050B7C4E42ACBA1DACBF04299C3460782F918EA427E6325165E9EA1",
                "0E3DA5F6C42E9C55215AA9CA27A5863EC48D8E0286B"
            ),
            order: concat!(
                "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE5F83B2D4",
                "EA20400EC4557D5ED3E3E7CA5B4B5C83B8E01E5FCF"
            ),
            cofactor: 4,
        })
    })
}

/// sect409r1 of SEC 2, NIST's B-409 (FIPS 186): one of its verifiably random curves,
/// `y² + x·y = x³ + x² + b`, over `GF(2⁴⁰⁹)`, the field of `x⁴⁰⁹ + x⁸⁷ + 1`, with cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect409r1();
/// assert_eq!(curve.field().modulus(), [409, 87, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect409r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[409, 87, 0],
            a: "1",
            b: concat!(
                "21A5C2C8EE9FEB5C4B9A753B7B476B7FD6422EF1F3DD674761FA99D6AC27",
                "C8A9A197B272822F6CD57A55AA4F50AE317B13545F"
            ),
            x: concat!(
                "15D4860D088DDB3496B0C6064756260441CDE4AF1771D4DB01FFE5B34E59",
                "703DC255A868A1180515603AEAB60794E54BB7996A7"
            ),
            y: concat!(
                "61B1CFAB6BE5F32BBFA78324ED106A7636B9C5A7BD198D0158AA4F5488D0",
                "8F38514F1FDF4B4F40D2181B3681C364BA0273C706"
            ),
            order: concat!(
                "10000000000000000000000000000000000000000000000000001E2AAD6A",
                "612F33307BE5FA47C3C9E052F838164CD37D9A21173"
            ),
            cofactor: 2,
        })
    })
}

/// sect571k1 of SEC 2, NIST's K-571 (FIPS 186): the Koblitz curve `y² + x·y = x³ + 1` over
/// `GF(2⁵⁷¹)`, the field of `x⁵⁷¹ + x¹⁰ + x⁵ + x² + 1`, with cofactor 4.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect571k1();
/// assert_eq!(curve.field().modulus(), [571, 10, 5, 2, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// assert_eq!(curve.koblitz_order().unwrap(), curve.order() * curve.cofactor());
/// ```
pub fn sect571k1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[571, 10, 5, 2, 0],
            a: "0",
            b: "1",
            x: concat!(
                "26EB7A859923FBC82189631F8103FE4AC9CA2970012D5D46024804801841",
                "CA44370958493B205E647DA304DB4CEB08CBBD1BA39494776FB988B47174",
                "DCA88C7E2945283A01C8972"
            ),
            y: concat!(
                "349DC807F4FBF374F4AEADE3BCA95314DD58CEC9F307A54FFC61EFC006D8",
                "A2C9D4979C0AC44AEA74FBEBBB9F772AEDCB620B01A7BA7AF1B320430C85",
                "91984F601CD4C143EF1C7A3"
            ),
            order: concat!(
                "200000000000000000000000000000000000000000000000000000000000",
                "00000000000131850E1F19A63E4B391A8DB917F4138B630D84BE5D639381",
                "E91DEB45CFE778F637C1001"
            ),
            cofactor: 4,
        })
    })
}

/// sect571r1 of SEC 2, NIST's B-571 (FIPS 186): one of its verifiably random curves,
/// `y² + x·y = x³ + x² + b`, over `GF(2⁵⁷¹)`, the field of `x⁵⁷¹ + x¹⁰ + x⁵ + x² + 1`, with
/// cofactor 2.
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// let curve = named::sect571r1();
/// assert_eq!(curve.field().modulus(), [571, 10, 5, 2, 0]);
/// assert!((&curve.generator() * curve.order()).is_identity());
/// ```
pub fn sect571r1() -> &'static BinaryDomain {
    static CURVE: OnceLock<BinaryDomain> = OnceLock::new();
    CURVE.get_or_init(|| {
        built(&Spec {
            modulus: &[571, 10, 5, 2, 0],
            a: "1",
            b: concat!(
                "2F40E7E2221F295DE297117B7F3D62F5C6A97FFCB8CEFF1CD6BA8CE4A9A1",
                "8AD84FFABBD8EFA59332BE7AD6756A66E294AFD185A78FF12AA520E4DE73",
                "9BACA0C7FFEFF7F2955727A"
            ),
            x: concat!(
                "303001D34B856296C16C0D40D3CD7750A93D1D2955FA80AA5F40FC8DB7B2",
                "ABDBDE53950F4C0D293CDD711A35B67FB1499AE60038614F1394ABFA3B4C",
                "850D927E1E7769C8EEC2D19"
            ),
            y: concat!(
                "37BF27342DA639B6DCCFFFEB73D69D78C6C27A6009CBBCA1980F8533921E",
                "8A684423E43BAB08A576291AF8F461BB2A8B3531D2F0485C19B16E2F1516",
                "E23DD3C1A4827AF1B8AC15B"
            ),
            order: concat!(
                "3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "FFFFFFFFFFFE661CE18FF55987308059B186823851EC7DD9CA1161DE93D5",
                "174D66E8382E9BB2FE84E47"
            ),
            cofactor: 2,
        })
    })
}

/// NIST's K-163 of FIPS 186: [`sect163k1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::k163(), named::sect163k1()));
/// ```
pub fn k163() -> &'static BinaryDomain {
    sect163k1()
}

/// NIST's B-163 of FIPS 186: [`sect163r2`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::b163(), named::sect163r2()));
/// ```
pub fn b163() -> &'static BinaryDomain {
    sect163r2()
}

/// NIST's K-233 of FIPS 186: [`sect233k1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::k233(), named::sect233k1()));
/// ```
pub fn k233() -> &'static BinaryDomain {
    sect233k1()
}

/// NIST's B-233 of FIPS 186: [`sect233r1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::b233(), named::sect233r1()));
/// ```
pub fn b233() -> &'static BinaryDomain {
    sect233r1()
}

/// NIST's K-283 of FIPS 186: [`sect283k1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::k283(), named::sect283k1()));
/// ```
pub fn k283() -> &'static BinaryDomain {
    sect283k1()
}

/// NIST's B-283 of FIPS 186: [`sect283r1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::b283(), named::sect283r1()));
/// ```
pub fn b283() -> &'static BinaryDomain {
    sect283r1()
}

/// NIST's K-409 of FIPS 186: [`sect409k1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::k409(), named::sect409k1()));
/// ```
pub fn k409() -> &'static BinaryDomain {
    sect409k1()
}

/// NIST's B-409 of FIPS 186: [`sect409r1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::b409(), named::sect409r1()));
/// ```
pub fn b409() -> &'static BinaryDomain {
    sect409r1()
}

/// NIST's K-571 of FIPS 186: [`sect571k1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::k571(), named::sect571k1()));
/// ```
pub fn k571() -> &'static BinaryDomain {
    sect571k1()
}

/// NIST's B-571 of FIPS 186: [`sect571r1`].
///
/// # Examples
///
/// ```
/// use a::math::ec::binary::named;
///
/// assert!(core::ptr::eq(named::b571(), named::sect571r1()));
/// ```
pub fn b571() -> &'static BinaryDomain {
    sect571r1()
}
