//! `Uint<1>` and `Int<1>` against `u64` and `i64`, `Uint<2>` and `Int<2>` against `u128` and
//! `i128`, method by method, on every pair of an edge set and seeded random values: the same
//! values, and the same inputs panicking. As the primitives' operators panic on overflow with
//! debug assertions and wrap without, these tests check both behaviours, run in each profile
//! (`cargo test` and `cargo test --release`).

use super::common::{Rng, edges, outcome, signed_edges, size};
use a::math::uint::{Int, Uint};
use core::cmp::Ordering;
use core::fmt::Debug;

/// The primitive counterpart of a result, to compare with the primitive's own.
trait Mirror {
    type P: PartialEq + Debug;
    fn mirror(self) -> Self::P;
}

impl Mirror for Uint<1> {
    type P = u64;
    fn mirror(self) -> u64 {
        self.as_limbs()[0]
    }
}

impl Mirror for Uint<2> {
    type P = u128;
    fn mirror(self) -> u128 {
        u128::from_le_bytes(self.to_le_bytes())
    }
}

impl Mirror for Int<1> {
    type P = i64;
    fn mirror(self) -> i64 {
        self.as_limbs()[0] as i64
    }
}

impl Mirror for Int<2> {
    type P = i128;
    fn mirror(self) -> i128 {
        i128::from_le_bytes(self.to_le_bytes())
    }
}

macro_rules! plain_mirror {
    ($($t:ty),*) => {$(
        impl Mirror for $t {
            type P = $t;
            fn mirror(self) -> $t {
                self
            }
        }
    )*};
}

plain_mirror!(
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    bool,
    (),
    String,
    Ordering
);

impl<T: Mirror> Mirror for Option<T> {
    type P = Option<T::P>;
    fn mirror(self) -> Self::P {
        self.map(T::mirror)
    }
}

impl<A: Mirror, B: Mirror> Mirror for (A, B) {
    type P = (A::P, B::P);
    fn mirror(self) -> Self::P {
        (self.0.mirror(), self.1.mirror())
    }
}

impl<T: Mirror, E> Mirror for Result<T, E> {
    type P = Option<T::P>;
    fn mirror(self) -> Self::P {
        self.ok().map(T::mirror)
    }
}

/// Compares our result with the primitive's: both panic, or both give the same value.
macro_rules! same {
    ($ours:expr, $theirs:expr, $($ctx:expr),*) => {{
        let ours = outcome(|| Mirror::mirror($ours));
        let theirs = outcome(|| $theirs);
        match (&ours, &theirs) {
            (Ok(a), Ok(b)) => assert_eq!(
                a, b, "{} on {:?}", stringify!($ours), ($($ctx,)*)
            ),
            (Err(_), Err(_)) => {}
            _ => panic!(
                "{}: ours {:?}, the primitive's {:?}, on {:?}",
                stringify!($ours), ours, theirs, ($($ctx,)*)
            ),
        }
    }};
}

/// The shift amounts and exponents the tests try, around the widths of one and two limbs.
const AMOUNTS: [u32; 16] = [
    0,
    1,
    2,
    31,
    63,
    64,
    65,
    100,
    126,
    127,
    128,
    129,
    191,
    255,
    1000,
    u32::MAX,
];

macro_rules! unsigned_against_primitive {
    ($test:ident, $n:literal, $p:ty, $s:ty, $seed:literal) => {
        #[test]
        fn $test() {
            type U = Uint<$n>;
            type I = Int<$n>;
            let mut rng = Rng($seed);
            let mut values: Vec<$p> = edges::<$n>().into_iter().map(Mirror::mirror).collect();
            for _ in 0..size(48, 4) {
                values.push(rng.uint::<$n>().mirror());
            }
            let ours = |a: $p| U::from_le_bytes(a.to_le_bytes());
            let ours_signed = |a: $s| I::from_le_bytes(a.to_le_bytes());
            for &a in &values {
                let x = ours(a);
                same!(x.count_ones(), a.count_ones(), a);
                same!(x.count_zeros(), a.count_zeros(), a);
                same!(x.leading_zeros(), a.leading_zeros(), a);
                same!(x.trailing_zeros(), a.trailing_zeros(), a);
                same!(x.leading_ones(), a.leading_ones(), a);
                same!(x.trailing_ones(), a.trailing_ones(), a);
                same!(x.swap_bytes(), a.swap_bytes(), a);
                same!(x.reverse_bits(), a.reverse_bits(), a);
                same!(!x, !a, a);
                same!(x.is_power_of_two(), a.is_power_of_two(), a);
                same!(x.next_power_of_two(), a.next_power_of_two(), a);
                same!(
                    x.checked_next_power_of_two(),
                    a.checked_next_power_of_two(),
                    a
                );
                same!(x.isqrt(), a.isqrt(), a);
                same!(x.ilog2(), a.ilog2(), a);
                same!(x.ilog10(), a.ilog10(), a);
                same!(x.checked_ilog2(), a.checked_ilog2(), a);
                same!(x.checked_ilog10(), a.checked_ilog10(), a);
                same!(x.checked_neg(), a.checked_neg(), a);
                same!(x.wrapping_neg(), a.wrapping_neg(), a);
                same!(x.overflowing_neg(), a.overflowing_neg(), a);
                same!(x.cast_signed(), a.cast_signed(), a);
                same!(x.to_f64().to_bits(), (a as f64).to_bits(), a);
                same!(x.to_f32().to_bits(), (a as f32).to_bits(), a);
                same!(x.to_string(), a.to_string(), a);
                same!(format!("{x:x}"), format!("{a:x}"), a);
                same!(format!("{x:#X}"), format!("{a:#X}"), a);
                same!(format!("{x:o}"), format!("{a:o}"), a);
                same!(format!("{x:#b}"), format!("{a:#b}"), a);
                same!(x.to_be_bytes() == a.to_be_bytes(), true, a);
                same!(U::from_be_bytes(a.to_be_bytes()), a, a);
                same!(u8::try_from(x), u8::try_from(a).ok(), a);
                same!(u16::try_from(x), u16::try_from(a).ok(), a);
                same!(u32::try_from(x), u32::try_from(a).ok(), a);
                same!(u64::try_from(x), u64::try_from(a).ok(), a);
                same!(u128::try_from(x), u128::try_from(a).ok(), a);
                same!(usize::try_from(x), usize::try_from(a).ok(), a);
                same!(i8::try_from(x), i8::try_from(a).ok(), a);
                same!(i16::try_from(x), i16::try_from(a).ok(), a);
                same!(i32::try_from(x), i32::try_from(a).ok(), a);
                same!(i64::try_from(x), i64::try_from(a).ok(), a);
                same!(i128::try_from(x), i128::try_from(a).ok(), a);
                same!(isize::try_from(x), isize::try_from(a).ok(), a);
                same!(I::try_from(x), <$s>::try_from(a).ok(), a);
                for &k in &AMOUNTS {
                    same!(x << k, a << k, a, k);
                    same!(x >> k, a >> k, a, k);
                    same!(x << (k as u8), a << (k as u8), a, k);
                    same!(x >> (k as i8), a >> (k as i8), a, k);
                    same!(
                        x << (k as i32).wrapping_neg(),
                        a << (k as i32).wrapping_neg(),
                        a,
                        k
                    );
                    same!(x >> (k as u64), a >> (k as u64), a, k);
                    same!(x << (k as i128), a << (k as i128), a, k);
                    same!(x >> (k as usize), a >> (k as usize), a, k);
                    same!(x << &(k as i16), a << &(k as i16), a, k);
                    same!(&x >> &(k as u16), &a >> &(k as u16), a, k);
                    same!(x.overflowing_shl(k), a.overflowing_shl(k), a, k);
                    same!(x.overflowing_shr(k), a.overflowing_shr(k), a, k);
                    same!(x.checked_shl(k), a.checked_shl(k), a, k);
                    same!(x.checked_shr(k), a.checked_shr(k), a, k);
                    same!(x.wrapping_shl(k), a.wrapping_shl(k), a, k);
                    same!(x.wrapping_shr(k), a.wrapping_shr(k), a, k);
                    same!(x.unbounded_shl(k), a.unbounded_shl(k), a, k);
                    same!(x.unbounded_shr(k), a.unbounded_shr(k), a, k);
                    same!(x.rotate_left(k), a.rotate_left(k), a, k);
                    same!(x.rotate_right(k), a.rotate_right(k), a, k);
                    same!(x.pow(k), a.pow(k), a, k);
                    same!(x.checked_pow(k), a.checked_pow(k), a, k);
                    same!(x.wrapping_pow(k), a.wrapping_pow(k), a, k);
                    same!(x.overflowing_pow(k), a.overflowing_pow(k), a, k);
                    same!(x.saturating_pow(k), a.saturating_pow(k), a, k);
                }
                for &b in &values {
                    let y = ours(b);
                    same!(x + y, a + b, a, b);
                    same!(x - y, a - b, a, b);
                    same!(x * y, a * b, a, b);
                    same!(x / y, a / b, a, b);
                    same!(x % y, a % b, a, b);
                    same!(x & y, a & b, a, b);
                    same!(x | y, a | b, a, b);
                    same!(x ^ y, a ^ b, a, b);
                    same!(x.cmp(&y), a.cmp(&b), a, b);
                    same!(x == y, a == b, a, b);
                    same!(x.checked_add(y), a.checked_add(b), a, b);
                    same!(x.checked_sub(y), a.checked_sub(b), a, b);
                    same!(x.checked_mul(y), a.checked_mul(b), a, b);
                    same!(x.checked_div(y), a.checked_div(b), a, b);
                    same!(x.checked_rem(y), a.checked_rem(b), a, b);
                    same!(x.checked_div_euclid(y), a.checked_div_euclid(b), a, b);
                    same!(x.checked_rem_euclid(y), a.checked_rem_euclid(b), a, b);
                    same!(x.wrapping_add(y), a.wrapping_add(b), a, b);
                    same!(x.wrapping_sub(y), a.wrapping_sub(b), a, b);
                    same!(x.wrapping_mul(y), a.wrapping_mul(b), a, b);
                    same!(x.wrapping_div(y), a.wrapping_div(b), a, b);
                    same!(x.wrapping_rem(y), a.wrapping_rem(b), a, b);
                    same!(x.wrapping_div_euclid(y), a.wrapping_div_euclid(b), a, b);
                    same!(x.wrapping_rem_euclid(y), a.wrapping_rem_euclid(b), a, b);
                    same!(x.overflowing_add(y), a.overflowing_add(b), a, b);
                    same!(x.overflowing_sub(y), a.overflowing_sub(b), a, b);
                    same!(x.overflowing_mul(y), a.overflowing_mul(b), a, b);
                    same!(x.overflowing_div(y), a.overflowing_div(b), a, b);
                    same!(x.overflowing_rem(y), a.overflowing_rem(b), a, b);
                    same!(
                        x.overflowing_div_euclid(y),
                        a.overflowing_div_euclid(b),
                        a,
                        b
                    );
                    same!(
                        x.overflowing_rem_euclid(y),
                        a.overflowing_rem_euclid(b),
                        a,
                        b
                    );
                    same!(x.saturating_add(y), a.saturating_add(b), a, b);
                    same!(x.saturating_sub(y), a.saturating_sub(b), a, b);
                    same!(x.saturating_mul(y), a.saturating_mul(b), a, b);
                    same!(x.saturating_div(y), a.saturating_div(b), a, b);
                    same!(x.div_euclid(y), a.div_euclid(b), a, b);
                    same!(x.rem_euclid(y), a.rem_euclid(b), a, b);
                    same!(x.div_ceil(y), a.div_ceil(b), a, b);
                    same!(x.next_multiple_of(y), a.next_multiple_of(b), a, b);
                    same!(
                        x.checked_next_multiple_of(y),
                        a.checked_next_multiple_of(b),
                        a,
                        b
                    );
                    same!(x.is_multiple_of(y), a.is_multiple_of(b), a, b);
                    same!(x.abs_diff(y), a.abs_diff(b), a, b);
                    same!(x.midpoint(y), a.midpoint(b), a, b);
                    same!(x.ilog(y), a.ilog(b), a, b);
                    same!(x.checked_ilog(y), a.checked_ilog(b), a, b);
                    let (bs, ys) = (b as $s, ours_signed(b as $s));
                    same!(x.checked_add_signed(ys), a.checked_add_signed(bs), a, b);
                    same!(x.wrapping_add_signed(ys), a.wrapping_add_signed(bs), a, b);
                    same!(
                        x.overflowing_add_signed(ys),
                        a.overflowing_add_signed(bs),
                        a,
                        b
                    );
                    same!(
                        x.saturating_add_signed(ys),
                        a.saturating_add_signed(bs),
                        a,
                        b
                    );
                }
            }
        }
    };
}

unsigned_against_primitive!(
    uint_1_behaves_as_u64_method_by_method,
    1,
    u64,
    i64,
    0x5eed_0001
);
unsigned_against_primitive!(
    uint_2_behaves_as_u128_method_by_method,
    2,
    u128,
    i128,
    0x5eed_0002
);

macro_rules! signed_against_primitive {
    ($test:ident, $n:literal, $s:ty, $p:ty, $seed:literal) => {
        #[test]
        fn $test() {
            type U = Uint<$n>;
            type I = Int<$n>;
            let mut rng = Rng($seed);
            let mut values: Vec<$s> = signed_edges::<$n>()
                .into_iter()
                .map(Mirror::mirror)
                .collect();
            for _ in 0..size(48, 4) {
                values.push(rng.int::<$n>().mirror());
            }
            let ours = |a: $s| I::from_le_bytes(a.to_le_bytes());
            let ours_unsigned = |a: $p| U::from_le_bytes(a.to_le_bytes());
            for &a in &values {
                let x = ours(a);
                same!(x.abs(), a.abs(), a);
                same!(x.checked_abs(), a.checked_abs(), a);
                same!(x.wrapping_abs(), a.wrapping_abs(), a);
                same!(x.overflowing_abs(), a.overflowing_abs(), a);
                same!(x.saturating_abs(), a.saturating_abs(), a);
                same!(x.unsigned_abs(), a.unsigned_abs(), a);
                same!(x.signum(), a.signum(), a);
                same!(x.is_positive(), a.is_positive(), a);
                same!(x.is_negative(), a.is_negative(), a);
                same!(-x, -a, a);
                same!(-&x, -&a, a);
                same!(x.checked_neg(), a.checked_neg(), a);
                same!(x.wrapping_neg(), a.wrapping_neg(), a);
                same!(x.overflowing_neg(), a.overflowing_neg(), a);
                same!(x.saturating_neg(), a.saturating_neg(), a);
                same!(x.isqrt(), a.isqrt(), a);
                same!(x.checked_isqrt(), a.checked_isqrt(), a);
                same!(x.ilog2(), a.ilog2(), a);
                same!(x.ilog10(), a.ilog10(), a);
                same!(x.checked_ilog2(), a.checked_ilog2(), a);
                same!(x.checked_ilog10(), a.checked_ilog10(), a);
                same!(x.count_ones(), a.count_ones(), a);
                same!(x.count_zeros(), a.count_zeros(), a);
                same!(x.leading_zeros(), a.leading_zeros(), a);
                same!(x.trailing_zeros(), a.trailing_zeros(), a);
                same!(x.leading_ones(), a.leading_ones(), a);
                same!(x.trailing_ones(), a.trailing_ones(), a);
                same!(x.swap_bytes(), a.swap_bytes(), a);
                same!(x.reverse_bits(), a.reverse_bits(), a);
                same!(!x, !a, a);
                same!(x.cast_unsigned(), a.cast_unsigned(), a);
                same!(x.to_f64().to_bits(), (a as f64).to_bits(), a);
                same!(x.to_f32().to_bits(), (a as f32).to_bits(), a);
                same!(x.to_string(), a.to_string(), a);
                same!(format!("{x:x}"), format!("{a:x}"), a);
                same!(format!("{x:#X}"), format!("{a:#X}"), a);
                same!(format!("{x:#o}"), format!("{a:#o}"), a);
                same!(format!("{x:b}"), format!("{a:b}"), a);
                same!(x.to_le_bytes() == a.to_le_bytes(), true, a);
                same!(I::from_be_bytes(a.to_be_bytes()), a, a);
                same!(u8::try_from(x), u8::try_from(a).ok(), a);
                same!(u16::try_from(x), u16::try_from(a).ok(), a);
                same!(u32::try_from(x), u32::try_from(a).ok(), a);
                same!(u64::try_from(x), u64::try_from(a).ok(), a);
                same!(u128::try_from(x), u128::try_from(a).ok(), a);
                same!(usize::try_from(x), usize::try_from(a).ok(), a);
                same!(i8::try_from(x), i8::try_from(a).ok(), a);
                same!(i16::try_from(x), i16::try_from(a).ok(), a);
                same!(i32::try_from(x), i32::try_from(a).ok(), a);
                same!(i64::try_from(x), i64::try_from(a).ok(), a);
                same!(i128::try_from(x), i128::try_from(a).ok(), a);
                same!(isize::try_from(x), isize::try_from(a).ok(), a);
                same!(U::try_from(x), <$p>::try_from(a).ok(), a);
                for &k in &AMOUNTS {
                    same!(x << k, a << k, a, k);
                    same!(x >> k, a >> k, a, k);
                    same!(x << (k as i8), a << (k as i8), a, k);
                    same!(
                        x >> (k as i64).wrapping_neg(),
                        a >> (k as i64).wrapping_neg(),
                        a,
                        k
                    );
                    same!(x >> (k as u128), a >> (k as u128), a, k);
                    same!(&x << &(k as isize), &a << &(k as isize), a, k);
                    same!(x.overflowing_shl(k), a.overflowing_shl(k), a, k);
                    same!(x.overflowing_shr(k), a.overflowing_shr(k), a, k);
                    same!(x.checked_shl(k), a.checked_shl(k), a, k);
                    same!(x.checked_shr(k), a.checked_shr(k), a, k);
                    same!(x.wrapping_shl(k), a.wrapping_shl(k), a, k);
                    same!(x.wrapping_shr(k), a.wrapping_shr(k), a, k);
                    same!(x.unbounded_shl(k), a.unbounded_shl(k), a, k);
                    same!(x.unbounded_shr(k), a.unbounded_shr(k), a, k);
                    same!(x.rotate_left(k), a.rotate_left(k), a, k);
                    same!(x.rotate_right(k), a.rotate_right(k), a, k);
                    same!(x.pow(k), a.pow(k), a, k);
                    same!(x.checked_pow(k), a.checked_pow(k), a, k);
                    same!(x.wrapping_pow(k), a.wrapping_pow(k), a, k);
                    same!(x.overflowing_pow(k), a.overflowing_pow(k), a, k);
                    same!(x.saturating_pow(k), a.saturating_pow(k), a, k);
                }
                for &b in &values {
                    let y = ours(b);
                    same!(x + y, a + b, a, b);
                    same!(x - y, a - b, a, b);
                    same!(x * y, a * b, a, b);
                    same!(x / y, a / b, a, b);
                    same!(x % y, a % b, a, b);
                    same!(x & y, a & b, a, b);
                    same!(x | y, a | b, a, b);
                    same!(x ^ y, a ^ b, a, b);
                    same!(x.cmp(&y), a.cmp(&b), a, b);
                    same!(x.checked_add(y), a.checked_add(b), a, b);
                    same!(x.checked_sub(y), a.checked_sub(b), a, b);
                    same!(x.checked_mul(y), a.checked_mul(b), a, b);
                    same!(x.checked_div(y), a.checked_div(b), a, b);
                    same!(x.checked_rem(y), a.checked_rem(b), a, b);
                    same!(x.checked_div_euclid(y), a.checked_div_euclid(b), a, b);
                    same!(x.checked_rem_euclid(y), a.checked_rem_euclid(b), a, b);
                    same!(x.wrapping_add(y), a.wrapping_add(b), a, b);
                    same!(x.wrapping_sub(y), a.wrapping_sub(b), a, b);
                    same!(x.wrapping_mul(y), a.wrapping_mul(b), a, b);
                    same!(x.wrapping_div(y), a.wrapping_div(b), a, b);
                    same!(x.wrapping_rem(y), a.wrapping_rem(b), a, b);
                    same!(x.wrapping_div_euclid(y), a.wrapping_div_euclid(b), a, b);
                    same!(x.wrapping_rem_euclid(y), a.wrapping_rem_euclid(b), a, b);
                    same!(x.overflowing_add(y), a.overflowing_add(b), a, b);
                    same!(x.overflowing_sub(y), a.overflowing_sub(b), a, b);
                    same!(x.overflowing_mul(y), a.overflowing_mul(b), a, b);
                    same!(x.overflowing_div(y), a.overflowing_div(b), a, b);
                    same!(x.overflowing_rem(y), a.overflowing_rem(b), a, b);
                    same!(
                        x.overflowing_div_euclid(y),
                        a.overflowing_div_euclid(b),
                        a,
                        b
                    );
                    same!(
                        x.overflowing_rem_euclid(y),
                        a.overflowing_rem_euclid(b),
                        a,
                        b
                    );
                    same!(x.saturating_add(y), a.saturating_add(b), a, b);
                    same!(x.saturating_sub(y), a.saturating_sub(b), a, b);
                    same!(x.saturating_mul(y), a.saturating_mul(b), a, b);
                    same!(x.saturating_div(y), a.saturating_div(b), a, b);
                    same!(x.div_euclid(y), a.div_euclid(b), a, b);
                    same!(x.rem_euclid(y), a.rem_euclid(b), a, b);
                    same!(x.abs_diff(y), a.abs_diff(b), a, b);
                    same!(x.midpoint(y), a.midpoint(b), a, b);
                    same!(x.ilog(y), a.ilog(b), a, b);
                    same!(x.checked_ilog(y), a.checked_ilog(b), a, b);
                    let (bu, yu) = (b as $p, ours_unsigned(b as $p));
                    same!(x.checked_add_unsigned(yu), a.checked_add_unsigned(bu), a, b);
                    same!(
                        x.wrapping_add_unsigned(yu),
                        a.wrapping_add_unsigned(bu),
                        a,
                        b
                    );
                    same!(
                        x.overflowing_add_unsigned(yu),
                        a.overflowing_add_unsigned(bu),
                        a,
                        b
                    );
                    same!(
                        x.saturating_add_unsigned(yu),
                        a.saturating_add_unsigned(bu),
                        a,
                        b
                    );
                    same!(x.checked_sub_unsigned(yu), a.checked_sub_unsigned(bu), a, b);
                    same!(
                        x.wrapping_sub_unsigned(yu),
                        a.wrapping_sub_unsigned(bu),
                        a,
                        b
                    );
                    same!(
                        x.overflowing_sub_unsigned(yu),
                        a.overflowing_sub_unsigned(bu),
                        a,
                        b
                    );
                    same!(
                        x.saturating_sub_unsigned(yu),
                        a.saturating_sub_unsigned(bu),
                        a,
                        b
                    );
                }
            }
        }
    };
}

signed_against_primitive!(
    int_1_behaves_as_i64_method_by_method,
    1,
    i64,
    u64,
    0x5eed_0003
);
signed_against_primitive!(
    int_2_behaves_as_i128_method_by_method,
    2,
    i128,
    u128,
    0x5eed_0004
);

/// `TryFrom` each primitive type `$t` into `$u` and `$i`, against `$p` and `$s`.
macro_rules! try_from_each {
    ($u:ty, $i:ty, $p:ty, $s:ty; $($t:ty),*) => {$(
        for v in [<$t>::MIN, <$t>::MAX, 0 as $t, 1 as $t, <$t>::MAX / 3, (<$t>::MIN / 3)] {
            same!(<$u>::try_from(v), <$p>::try_from(v).ok(), v);
            same!(<$i>::try_from(v), <$s>::try_from(v).ok(), v);
        }
    )*};
}

/// The conversions from the primitive integers against the primitives' own between themselves:
/// `From` where every width holds the value, `TryFrom` elsewhere, failing exactly when the
/// primitive of the same width fails.
macro_rules! from_primitives {
    ($test:ident, $n:literal, $p:ty, $s:ty) => {
        #[test]
        fn $test() {
            type U = Uint<$n>;
            type I = Int<$n>;
            try_from_each!(U, I, $p, $s; u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
            for v in [0u64, 1, u64::MAX] {
                same!(U::from(v), v as $p, v);
                same!(U::from(v as u32), v as u32 as $p, v);
                same!(U::from(v as u16), v as u16 as $p, v);
                same!(U::from(v as u8), v as u8 as $p, v);
                same!(U::from(v as usize), v as usize as $p, v);
                same!(I::from(v as i64), v as i64 as $s, v);
                same!(I::from(v as i32), v as i32 as $s, v);
                same!(I::from(v as i16), v as i16 as $s, v);
                same!(I::from(v as i8), v as i8 as $s, v);
                same!(I::from(v as isize), v as isize as $s, v);
                same!(I::from(v as u32), v as u32 as $s, v);
                same!(I::from(v as u16), v as u16 as $s, v);
                same!(I::from(v as u8), v as u8 as $s, v);
            }
            same!(U::from(true), <$p>::from(true), true);
            same!(I::from(true), <$s>::from(true), true);
            same!(U::from(false), <$p>::from(false), false);
        }
    };
}

from_primitives!(
    conversions_into_uint_1_and_int_1_fail_as_u64_and_i64_do,
    1,
    u64,
    i64
);
from_primitives!(
    conversions_into_uint_2_and_int_2_fail_as_u128_and_i128_do,
    2,
    u128,
    i128
);

/// Floats into the integers against `as`, on the float's edges and random bit patterns.
macro_rules! from_floats {
    ($test:ident, $n:literal, $p:ty, $s:ty, $seed:literal) => {
        #[test]
        fn $test() {
            type U = Uint<$n>;
            type I = Int<$n>;
            let mut floats = vec![
                0.0,
                -0.0,
                0.5,
                -0.5,
                1.0,
                -1.0,
                1.5,
                -2.5,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::MAX,
                f64::MIN,
                f64::MIN_POSITIVE,
                f64::EPSILON,
            ];
            for k in [52, 53, 63, 64, 65, 126, 127, 128, 129] {
                let p = 2f64.powi(k);
                floats.extend([p, -p, p * (1.0 - f64::EPSILON), -p * (1.0 - f64::EPSILON)]);
            }
            let mut rng = Rng($seed);
            for _ in 0..size(4000, 100) {
                // Random patterns with the exponent biased to the integers' range.
                let e = 1023 + rng.below(140);
                let bits = (rng.next() & ((1 << 52) - 1)) | (e << 52) | (rng.below(2) << 63);
                floats.push(f64::from_bits(bits));
            }
            for &f in &floats {
                same!(U::from_f64(f), f as $p, f);
                same!(I::from_f64(f), f as $s, f);
                let g = f as f32;
                same!(U::from_f32(g), g as $p, g);
                same!(I::from_f32(g), g as $s, g);
            }
        }
    };
}

from_floats!(
    floats_into_uint_1_and_int_1_convert_as_as_does,
    1,
    u64,
    i64,
    0x5eed_0005
);
from_floats!(
    floats_into_uint_2_and_int_2_convert_as_as_does,
    2,
    u128,
    i128,
    0x5eed_0006
);
