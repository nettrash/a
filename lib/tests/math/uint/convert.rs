//! Conversions: `Number` in every byte order and stored form, the primitive integers at widths
//! that always hold them, the floats correctly rounded both ways at every width, between widths,
//! and between `Uint` and `Int`.

use super::common::{Rng, exact, exact_signed, outcome, size};
use a::math::numbers::{ByteOrder, Number};
use a::math::uint::{Int, TryFromIntError, Uint};
use core::num::IntErrorKind;

/// A number of the sign and the magnitude `le` (least significant byte first), stored in `order`
/// with `pad` zero bytes above it, built field by field so that it is kept as given.
fn stored(le: &[u8], negative: bool, order: ByteOrder, pad: usize) -> Number {
    let mut value = le.to_vec();
    value.extend(std::iter::repeat_n(0, pad));
    if order == ByteOrder::BigEndian {
        value.reverse();
    }
    Number {
        value,
        negative,
        byte_order: order,
    }
}

#[test]
fn numbers_convert_by_value_in_every_byte_order_and_stored_form() {
    let mut rng = Rng(0xc0_0001);
    for _ in 0..size(400, 10) {
        let x = rng.uint::<4>();
        let mut le = vec![0u8; 32];
        x.write_le_bytes(&mut le);
        let len = le.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
            for pad in [0, 1, 9] {
                for cut in [len, 32] {
                    let n = stored(&le[..cut], false, order, pad);
                    assert_eq!(Uint::<4>::try_from(&n), Ok(x));
                    assert_eq!(Uint::<4>::try_from(n.clone()), Ok(x));
                    // The magnitude with a minus sign: negative unless it is zero.
                    let m = stored(&le[..cut], true, order, pad);
                    assert_eq!(Uint::<4>::try_from(&m).is_ok(), x.is_zero());
                    let expect = if x.is_zero() {
                        Some(Int::<5>::ZERO)
                    } else {
                        Some(-x.resize::<5>().cast_signed())
                    };
                    assert_eq!(Int::<5>::try_from(&m).ok(), expect);
                    assert_eq!(
                        Int::<4>::try_from(&n).ok(),
                        exact_signed::<4>(&Number::from(x))
                    );
                    // Narrower widths take it exactly when it fits.
                    assert_eq!(Uint::<2>::try_from(&n).ok(), x.try_resize::<2>());
                    assert_eq!(Uint::<9>::try_from(&n), Ok(x.resize::<9>()));
                }
            }
        }
    }
    // An empty magnitude and a zero marked negative are zero.
    for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
        for negative in [false, true] {
            assert_eq!(
                Uint::<3>::try_from(&stored(&[], negative, order, 0)),
                Ok(Uint::ZERO)
            );
            assert_eq!(
                Uint::<3>::try_from(&stored(&[0, 0], negative, order, 4)),
                Ok(Uint::ZERO)
            );
            assert_eq!(
                Int::<3>::try_from(&stored(&[], negative, order, 0)),
                Ok(Int::ZERO)
            );
        }
    }
    // The ends of the signed range, in both orders, with padding.
    let min = Number::from(Int::<3>::MIN);
    let mut le = min.value.clone();
    if min.byte_order == ByteOrder::BigEndian {
        le.reverse();
    }
    for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
        assert_eq!(
            Int::<3>::try_from(&stored(&le, true, order, 3)),
            Ok(Int::MIN)
        );
        assert!(Int::<3>::try_from(&stored(&le, false, order, 3)).is_err());
    }
    // Into Number and back, and the byte order Number gets.
    let n = Number::from(Uint::<3>::MAX);
    assert_eq!(n.byte_order, ByteOrder::LittleEndian);
    assert_eq!(n, (Number::from(1u64) << 192usize) - Number::from(1u64));
    let n = Number::from(&Int::<3>::MIN);
    assert_eq!(n, -(Number::from(1u64) << 191usize));
    assert_eq!(Number::from(&Uint::<3>::ZERO).value, [0]);
}

#[test]
fn primitive_integers_convert_at_every_width_that_holds_them() {
    fn at<const N: usize>() {
        assert_eq!(
            Uint::<N>::try_from(u128::MAX).map(Number::from),
            Ok(Number::from(u128::MAX))
        );
        assert_eq!(
            Int::<N>::try_from(i128::MIN).map(Number::from),
            Ok(Number::from(i128::MIN))
        );
        assert_eq!(
            Int::<N>::try_from(u128::MAX).map(Number::from),
            Ok(Number::from(u128::MAX))
        );
        assert_eq!(Int::<N>::from(i64::MIN), Int::from_i128(i64::MIN as i128));
        assert_eq!(
            Int::<N>::try_from(u64::MAX),
            Ok(Int::from_i128(u64::MAX as i128))
        );
        assert_eq!(Uint::<N>::from(u32::MAX), Uint::from_u64(u32::MAX as u64));
        assert!(Uint::<N>::try_from(-1i8).is_err());
        assert!(Uint::<N>::try_from(i128::MIN).is_err());
        let big = Uint::<N>::ONE << 128u32;
        assert!(u128::try_from(big).is_err());
        assert_eq!(u128::try_from(big - Uint::ONE), Ok(u128::MAX));
        assert!(i128::try_from(big.cast_signed()).is_err());
        assert_eq!(i128::try_from(-(Int::<N>::ONE << 127u32)), Ok(i128::MIN));
        assert!(i128::try_from(-(Int::<N>::ONE << 127u32) - Int::ONE).is_err());
        assert!(u8::try_from(Int::<N>::MINUS_ONE).is_err());
        assert!(u128::try_from(Int::<N>::MINUS_ONE).is_err());
        assert_eq!(i8::try_from(Int::<N>::from_i64(-128)), Ok(-128));
        assert!(i8::try_from(Int::<N>::from_i64(128)).is_err());
        assert_eq!(usize::try_from(Uint::<N>::from_u64(7)), Ok(7));
        assert_eq!(isize::try_from(Int::<N>::from_i64(-7)), Ok(-7));
    }
    at::<3>();
    at::<4>();
    at::<9>();
    at::<16>();
    // N = 1: u128 values beyond 64 bits do not fit, and from_u128 panics on them.
    assert!(Uint::<1>::try_from(1u128 << 64).is_err());
    assert!(outcome(|| Uint::<1>::from_u128(1 << 64)).is_err());
    assert!(outcome(|| Int::<1>::from_i128(i64::MAX as i128 + 1)).is_err());
    assert!(Int::<1>::try_from(u64::MAX).is_err());
    assert!(Int::<2>::try_from(u128::MAX).is_err());
    assert_eq!(Int::<2>::try_from(i128::MAX as u128), Ok(Int::MAX));
    let e: TryFromIntError = u8::try_from(Uint::<2>::MAX).unwrap_err();
    assert_eq!(e.to_string(), "number too large to fit in target type");
}

/// Checks the error of a failed conversion: `PosOverflow` and "too large" for a value above the
/// target's range, `NegOverflow` and "too small" below it; and the same text as the standard
/// library's error, `theirs`, when the toolchain is recent enough to say which way (Rust 1.88
/// wrote "out of range integral type conversion attempted" for both).
fn direction(ours: TryFromIntError, negative: bool, theirs: Option<String>, what: &str) {
    let (kind, text) = if negative {
        (
            IntErrorKind::NegOverflow,
            "number too small to fit in target type",
        )
    } else {
        (
            IntErrorKind::PosOverflow,
            "number too large to fit in target type",
        )
    };
    assert_eq!(*ours.kind(), kind, "{what}");
    assert_eq!(ours.to_string(), text, "{what}");
    if let Some(theirs) = theirs.filter(|t| t != "out of range integral type conversion attempted")
    {
        assert_eq!(ours.to_string(), theirs, "{what}");
    }
}

/// The error of a failed conversion of ours, or `None` for a conversion that cannot fail.
trait Ours {
    fn ours(self) -> TryFromIntError;
}

impl Ours for TryFromIntError {
    fn ours(self) -> TryFromIntError {
        self
    }
}

impl Ours for core::convert::Infallible {
    fn ours(self) -> TryFromIntError {
        match self {}
    }
}

fn failed<T, E: Ours>(r: Result<T, E>) -> Option<TryFromIntError> {
    r.err().map(Ours::ours)
}

/// The text of the standard library's error, if the conversion failed.
fn text<T, E: std::fmt::Display>(r: Result<T, E>) -> Option<String> {
    r.err().map(|e| e.to_string())
}

/// Every failing conversion between the fixed widths and a primitive type `$t`, at the edges.
macro_rules! errors_into {
    ($n:literal, $p:ty, $s:ty; $($t:ty),*) => {$(
        for &a in &[0 as $s, 1, -1, <$s>::MIN, <$s>::MAX, <$s>::MIN / 3, <$s>::MAX / 3, 300, -300, 70_000, -70_000] {
            let x = Int::<$n>::from_le_bytes(a.to_le_bytes());
            if let Some(e) = failed(<$t>::try_from(x)) {
                direction(e, a < 0, text(<$t>::try_from(a)), stringify!($t));
            }
            let u = a as $p;
            let y = Uint::<$n>::from_le_bytes(u.to_le_bytes());
            if let Some(e) = failed(<$t>::try_from(y)) {
                direction(e, false, text(<$t>::try_from(u)), stringify!($t));
            }
        }
        for v in [<$t>::MIN, <$t>::MAX, 0 as $t] {
            if let Some(e) = failed(Uint::<$n>::try_from(v)) {
                direction(e, v.to_string().starts_with('-'), text(<$p>::try_from(v)), stringify!($t));
            }
            if let Some(e) = failed(Int::<$n>::try_from(v)) {
                direction(e, v.to_string().starts_with('-'), text(<$s>::try_from(v)), stringify!($t));
            }
        }
    )*};
}

#[test]
fn conversion_errors_say_which_way_the_value_is_out_of_range() {
    errors_into!(1, u64, i64; u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
    errors_into!(2, u128, i128; u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
    // Between Uint and Int, and at a width beyond the primitives.
    direction(
        Uint::<3>::try_from(Int::<3>::MINUS_ONE).unwrap_err(),
        true,
        None,
        "Int -> Uint",
    );
    direction(
        Int::<3>::try_from(Uint::<3>::MAX).unwrap_err(),
        false,
        None,
        "Uint -> Int",
    );
    direction(
        i128::try_from(Int::<3>::MIN).unwrap_err(),
        true,
        None,
        "Int<3> -> i128",
    );
    direction(
        u128::try_from(Int::<3>::MIN).unwrap_err(),
        true,
        None,
        "Int<3> -> u128",
    );
    direction(
        u128::try_from(Uint::<3>::MAX).unwrap_err(),
        false,
        None,
        "Uint<3> -> u128",
    );
    direction(
        Int::<1>::try_from(i128::MIN).unwrap_err(),
        true,
        None,
        "i128 -> Int<1>",
    );
    direction(
        Int::<1>::try_from(u128::MAX).unwrap_err(),
        false,
        None,
        "u128 -> Int<1>",
    );
    direction(
        Uint::<1>::try_from(i128::MIN).unwrap_err(),
        true,
        None,
        "i128 -> Uint<1>",
    );
    assert_eq!(
        format!("{:?}", Uint::<2>::try_from(-1i8).unwrap_err()),
        "TryFromIntError(NegOverflow)"
    );
}

#[test]
fn uint_and_int_convert_into_each_other_when_the_value_fits() {
    let mut rng = Rng(0xc0_0002);
    for _ in 0..size(500, 10) {
        let x = rng.uint::<5>();
        let y = rng.int::<5>();
        assert_eq!(
            Int::<5>::try_from(x).ok(),
            exact_signed::<5>(&Number::from(x))
        );
        assert_eq!(Uint::<5>::try_from(y).ok(), exact::<5>(&Number::from(y)));
        assert_eq!(x.cast_signed().cast_unsigned(), x);
        assert_eq!(y.cast_unsigned().cast_signed(), y);
        // Between widths: zero- and sign-extension and truncation, as `as` does.
        assert_eq!(x.resize::<8>().resize::<5>(), x);
        assert_eq!(y.resize::<8>().resize::<5>(), y);
        assert_eq!(Number::from(y.resize::<8>()), Number::from(y));
        assert_eq!(x.resize::<2>().as_limbs(), &x.as_limbs()[..2]);
        assert_eq!(y.resize::<2>().as_limbs(), &y.as_limbs()[..2]);
        assert_eq!(x.try_resize::<2>(), exact::<2>(&Number::from(x)));
        assert_eq!(y.try_resize::<2>(), exact_signed::<2>(&Number::from(y)));
        assert_eq!(x.resize::<0>(), Uint::<0>::ZERO);
        assert_eq!(x.try_resize::<0>().is_some(), x.is_zero());
        assert_eq!(y.try_resize::<0>().is_some(), y.is_zero());
    }
}

/// The value of a finite `f64` truncated toward zero, exactly, as a number.
fn truncated(f: f64) -> Number {
    let b = f.to_bits();
    let exp = ((b >> 52) & 0x7FF) as i64;
    let m = if exp == 0 {
        (b & ((1 << 52) - 1)) << 1
    } else {
        (b & ((1 << 52) - 1)) | (1 << 52)
    };
    let e = exp.max(1) - 1075;
    let n = if e >= 0 {
        Number::from(m) << e as usize
    } else if e > -64 {
        Number::from(m >> -e)
    } else {
        Number::from(0u64)
    };
    if f < 0.0 { -n } else { n }
}

#[test]
fn floats_convert_both_ways_correctly_rounded_at_every_width() {
    fn at<const N: usize>(rng: &mut Rng, count: usize) {
        let bits = Uint::<N>::BITS as u64;
        let mut floats = vec![
            0.0,
            -0.0,
            0.7,
            -0.7,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE,
            5e-324,
            2f64.powi(63),
            2f64.powi(1023),
        ];
        for _ in 0..count {
            let e = 1023 + rng.below(bits + 20);
            floats.push(f64::from_bits(
                (rng.next() & ((1 << 52) - 1)) | (e.min(2046) << 52) | (rng.below(2) << 63),
            ));
        }
        for &f in &floats {
            let x = Uint::<N>::from_f64(f);
            let y = Int::<N>::from_f64(f);
            if f.is_nan() {
                assert_eq!((x, y), (Uint::ZERO, Int::ZERO));
                continue;
            }
            let (ux, sx) = if f.is_infinite() {
                (
                    if f > 0.0 { Uint::MAX } else { Uint::ZERO },
                    if f > 0.0 { Int::MAX } else { Int::MIN },
                )
            } else {
                let t = truncated(f);
                let u = if t.is_negative() {
                    Uint::ZERO
                } else {
                    exact(&t).unwrap_or(Uint::MAX)
                };
                let s =
                    exact_signed(&t).unwrap_or(if t.is_negative() { Int::MIN } else { Int::MAX });
                (u, s)
            };
            assert_eq!(x, ux, "{f:e}");
            assert_eq!(y, sx, "{f:e}");
            let g = f as f32;
            if g.is_finite() {
                let t = truncated(g as f64);
                assert_eq!(
                    Uint::<N>::from_f32(g),
                    if t.is_negative() {
                        Uint::ZERO
                    } else {
                        exact(&t).unwrap_or(Uint::MAX)
                    }
                );
            }
        }
        // To the floats: against Number's correctly rounded conversion.
        for _ in 0..count {
            let x = rng.uint::<N>();
            let y = rng.int::<N>();
            assert_eq!(x.to_f64().to_bits(), Number::from(x).to_f64().to_bits());
            assert_eq!(x.to_f32().to_bits(), Number::from(x).to_f32().to_bits());
            assert_eq!(y.to_f64().to_bits(), Number::from(y).to_f64().to_bits());
            assert_eq!(y.to_f32().to_bits(), Number::from(y).to_f32().to_bits());
        }
        assert_eq!(Uint::<N>::from_f32(f32::INFINITY), Uint::MAX);
        assert_eq!(Int::<N>::from_f32(f32::NEG_INFINITY), Int::MIN);
    }
    let mut rng = Rng(0xc0_0003);
    let n = size(500, 10);
    at::<1>(&mut rng, n);
    at::<2>(&mut rng, n);
    at::<3>(&mut rng, n);
    at::<5>(&mut rng, n);
    at::<16>(&mut rng, n);
    at::<17>(&mut rng, n);
    at::<32>(&mut rng, n / 2);
    // Halfway cases beyond 64 bits: 2^200 + 2^147 is halfway between two f64s and goes to the
    // even one; one more, in a bit far below the top 64, goes up.
    let half = (Uint::<4>::ONE << 200u32) + (Uint::ONE << 147u32);
    assert_eq!(half.to_f64(), 2f64.powi(200));
    assert_eq!((half + Uint::ONE).to_f64(), 2f64.powi(200) + 2f64.powi(148));
    let half = (Uint::<4>::ONE << 100u32) + (Uint::ONE << 76u32);
    assert_eq!(half.to_f32(), 2f32.powi(100));
    assert_eq!((half + Uint::ONE).to_f32(), 2f32.powi(100) + 2f32.powi(77));
    // Rounding up past the largest f64 gives infinity.
    let near = (Uint::<17>::ONE << 1024u32) - (Uint::ONE << 970u32);
    assert_eq!(near.to_f64(), f64::INFINITY);
    assert_eq!((near - Uint::ONE).to_f64(), f64::MAX);
    assert_eq!(Int::<16>::MIN.to_f64(), -(2f64.powi(1023)));
    assert_eq!(Int::<17>::MIN.to_f64(), f64::NEG_INFINITY);
}
