//! What the tests of addition and subtraction share: building and reading numbers, the
//! reference they are checked against, the checks, random operands, and operands for the tests
//! of where a result is built and of the long paths of the kernels.

use a::math::numbers::{ByteOrder, Number};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const ORDERS: [ByteOrder; 2] = [ByteOrder::BigEndian, ByteOrder::LittleEndian];

/// The byte orders of the two operands, in every combination.
pub(super) const ORDER_PAIRS: [(ByteOrder, ByteOrder); 4] = [
    (ByteOrder::BigEndian, ByteOrder::BigEndian),
    (ByteOrder::BigEndian, ByteOrder::LittleEndian),
    (ByteOrder::LittleEndian, ByteOrder::BigEndian),
    (ByteOrder::LittleEndian, ByteOrder::LittleEndian),
];

/// The signs of the two operands, `true` for negative, in every combination.
pub(super) const SIGN_PAIRS: [(bool, bool); 4] =
    [(false, false), (false, true), (true, false), (true, true)];

/// Scales an exhaustive or randomised test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Small deterministic xorshift generator, so the tests need no extra dependencies.
pub(super) fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A random number below `n`.
pub(super) fn below(state: &mut u64, n: usize) -> usize {
    (xorshift(state) % n as u64) as usize
}

// Building and reading numbers

/// The bytes of a magnitude given least significant first, put in `order`.
pub(super) fn in_order(magnitude: &[u8], order: ByteOrder) -> Vec<u8> {
    match order {
        ByteOrder::BigEndian => magnitude.iter().rev().copied().collect(),
        ByteOrder::LittleEndian => magnitude.to_vec(),
    }
}

/// The bytes of the magnitude of `n` as it stores them, least significant first, zero bytes
/// above the most significant one included.
pub(super) fn le_bytes(n: &Number) -> Vec<u8> {
    match n.byte_order {
        ByteOrder::BigEndian => n.value.iter().rev().copied().collect(),
        ByteOrder::LittleEndian => n.value.clone(),
    }
}

/// A number built from its fields, so that it is kept exactly as given: the magnitude
/// `magnitude`, least significant byte first and zero bytes above the most significant one
/// included, the sign `negative` and the byte order `order`.
pub(super) fn number(magnitude: &[u8], negative: bool, order: ByteOrder) -> Number {
    Number {
        value: in_order(magnitude, order),
        negative,
        byte_order: order,
    }
}

/// A number built from its fields as it stores them: `value` in byte order `order`, in a
/// vector with room for exactly `capacity` bytes.
pub(super) fn stored(value: &[u8], negative: bool, order: ByteOrder, capacity: usize) -> Number {
    assert!(
        capacity >= value.len(),
        "a vector holds at most its capacity"
    );
    // `Vec::with_capacity` gives exactly the capacity asked for.
    let mut bytes = Vec::with_capacity(capacity);
    // Natively the spare capacity holds junk, so that a sum read from past the end of the
    // bytes shows; under Miri it stays uninitialised, so that Miri reports such a read.
    if !cfg!(miri) {
        bytes.resize(capacity, 0xA5);
        bytes.clear();
    }
    bytes.extend_from_slice(value);
    Number {
        value: bytes,
        negative,
        byte_order: order,
    }
}

/// A non-negative number built from its fields as it stores them, as [`stored`] builds it.
pub(super) fn plus(value: &[u8], order: ByteOrder, capacity: usize) -> Number {
    stored(value, false, order, capacity)
}

/// A negative number built from its fields as it stores them, as [`stored`] builds it; for
/// zero bytes, a negative zero.
pub(super) fn minus(value: &[u8], order: ByteOrder, capacity: usize) -> Number {
    stored(value, true, order, capacity)
}

/// A copy of `n` with room for `spare` bytes more than it holds.
pub(super) fn copy(n: &Number, spare: usize) -> Number {
    stored(&n.value, n.negative, n.byte_order, n.value.len() + spare)
}

/// A copy of `n` with the capacity of `n`.
pub(super) fn copy_with_capacity(n: &Number) -> Number {
    stored(&n.value, n.negative, n.byte_order, n.value.capacity())
}

/// A copy of `n` with the opposite sign.
pub(super) fn negated(n: &Number) -> Number {
    Number {
        negative: !n.negative,
        ..copy(n, 0)
    }
}

/// Whether `x` and `y` are stored alike: the same bytes, sign and byte order.
pub(super) fn same(x: &Number, y: &Number) -> bool {
    x.value == y.value && x.negative == y.negative && x.byte_order == y.byte_order
}

/// Every way a number built from its fields can hold zero in byte order `order`: an empty
/// vector, one zero byte or several, each with either sign.
pub(super) fn zeros(order: ByteOrder) -> Vec<Number> {
    let mut zeros = Vec::new();
    for len in [0, 1, 2, 9] {
        for negative in [false, true] {
            zeros.push(number(&vec![0; len], negative, order));
        }
    }
    zeros
}

// Expected results

/// A sum as a test expects it: the bytes of its magnitude in the shortest form, least
/// significant first, and its sign.
#[derive(Debug, PartialEq)]
pub(super) struct Expected {
    pub(super) magnitude: Vec<u8>,
    pub(super) negative: bool,
}

/// The expected number with the magnitude `magnitude`, least significant byte first and zero
/// bytes above the most significant one allowed, and the sign `negative`, in the shortest
/// form: without those zero bytes, and zero as `[0]` and not negative.
pub(super) fn expected(magnitude: &[u8], negative: bool) -> Expected {
    match magnitude.iter().rposition(|&b| b != 0) {
        Some(top) => Expected {
            magnitude: magnitude[..=top].to_vec(),
            negative,
        },
        None => Expected {
            magnitude: vec![0],
            negative: false,
        },
    }
}

/// Zero, the sum of every addition that cancels.
pub(super) fn zero() -> Expected {
    expected(&[], false)
}

/// The integer `x` as an expected sum.
pub(super) fn expected_i128(x: i128) -> Expected {
    expected(&x.unsigned_abs().to_le_bytes(), x < 0)
}

/// The integer `n` holds, as an expected sum: what `n + 0` must give.
pub(super) fn value_of(n: &Number) -> Expected {
    expected(&le_bytes(n), n.negative)
}

/// The sum of the integers `a` and `b` hold, worked out independently of the implementation:
/// in two's complement on 32-bit limbs, where one addition serves every combination of signs
/// and no magnitudes are compared. The tests on small values check it against `i128`
/// arithmetic.
pub(super) fn reference(a: &Number, b: &Number) -> Expected {
    // Two limbs more than the longer vector fills leave room for the carry and the sign bit.
    let limbs = a.value.len().max(b.value.len()) / 4 + 2;
    let x = twos_complement(a, limbs);
    let y = twos_complement(b, limbs);
    let mut carry = 0;
    let mut sum: Vec<u32> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| {
            let wide = u64::from(x) + u64::from(y) + carry;
            carry = wide >> 32;
            wide as u32
        })
        .collect();
    // The carry out of the top limb is dropped, as two's complement addition does.
    let negative = sum[limbs - 1] >> 31 == 1;
    if negative {
        negate(&mut sum);
    }
    let magnitude: Vec<u8> = sum.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    expected(&magnitude, negative)
}

/// The integer `n` holds, in two's complement as `limbs` 32-bit limbs, least significant
/// first.
pub(super) fn twos_complement(n: &Number, limbs: usize) -> Vec<u32> {
    let mut x = vec![0; limbs];
    for (i, byte) in le_bytes(n).into_iter().enumerate() {
        x[i / 4] |= u32::from(byte) << (8 * (i % 4));
    }
    // Negating zero gives zero again: a negative zero is zero, as it must be.
    if n.negative {
        negate(&mut x);
    }
    x
}

/// Negates a number in two's complement in place: inverts every bit and adds one.
pub(super) fn negate(limbs: &mut [u32]) {
    let mut carry = true;
    for limb in limbs {
        (*limb, carry) = (!*limb).overflowing_add(u32::from(carry));
    }
}

// Checking results

/// `bytes` in hexadecimal for assertion messages, with the middle of a long vector left out.
pub(super) fn hex(bytes: &[u8]) -> String {
    if bytes.len() <= 40 {
        format!("{bytes:02x?}")
    } else {
        let tail = &bytes[bytes.len() - 16..];
        let len = bytes.len();
        format!("{:02x?} … {tail:02x?} ({len} bytes)", &bytes[..16])
    }
}

/// `n` as it is stored, for assertion messages: its sign, its bytes and its byte order.
pub(super) fn describe(n: &Number) -> String {
    let sign = if n.negative { "-" } else { "+" };
    let order = match n.byte_order {
        ByteOrder::BigEndian => "big-endian",
        ByteOrder::LittleEndian => "little-endian",
    };
    format!("{sign}{} {order}", hex(&n.value))
}

/// The first index at which two byte vectors differ, for assertion messages.
pub(super) fn first_difference(x: &[u8], y: &[u8]) -> usize {
    x.iter()
        .zip(y)
        .position(|(a, b)| a != b)
        .unwrap_or(x.len().min(y.len()))
}

/// Checks that `n` is in the shortest form: no zero byte above its most significant non-zero
/// byte, zero as the one byte `[0]`, and zero not negative. `inputs` describes the addition or
/// subtraction that gave `n`, for the messages; every result in these tests is checked here.
pub(super) fn assert_shortest(n: &Number, inputs: &dyn Fn() -> String) {
    let most_significant = match n.byte_order {
        ByteOrder::BigEndian => n.value.first(),
        ByteOrder::LittleEndian => n.value.last(),
    };
    match most_significant {
        None => panic!("{}: the result is empty", inputs()),
        Some(0) => {
            assert!(
                n.value.len() == 1,
                "{}: the result {} has zero bytes above its most significant byte",
                inputs(),
                describe(n)
            );
            assert!(!n.negative, "{}: the result is a negative zero", inputs());
        }
        Some(_) => {}
    }
}

/// Checks a sum or a difference: in the shortest form, in byte order `order`, and with the
/// magnitude and the sign of `expected`. `inputs` describes the operation, for the messages.
pub(super) fn assert_result(
    result: &Number,
    expected: &Expected,
    order: ByteOrder,
    inputs: &dyn Fn() -> String,
) {
    assert_shortest(result, inputs);
    assert!(
        result.byte_order == order,
        "{}: the result {} is not in the byte order of a",
        inputs(),
        describe(result)
    );
    let bytes = in_order(&expected.magnitude, order);
    assert!(
        result.value == bytes,
        "{}: the result is {}, expected {} (they differ from byte {})",
        inputs(),
        describe(result),
        hex(&bytes),
        first_difference(&result.value, &bytes)
    );
    assert!(
        result.negative == expected.negative,
        "{}: the result {} has the wrong sign",
        inputs(),
        describe(result)
    );
}

/// Under Miri, which runs these tests thousands of times slower, each form runs with one of
/// its capacity setups, and the tests that check many sums run one form per sum, instead of all
/// of them. This generator picks them at random: taken in turn, they would fall into step with
/// the loops of a test and leave some combinations out.
static PICK: AtomicU64 = AtomicU64::new(0x2545_F491_4F6C_DD1D);

/// A pick below `n` from [`PICK`].
pub(super) fn pick(n: usize) -> usize {
    let mut state = PICK.load(Ordering::Relaxed);
    let k = below(&mut state, n);
    PICK.store(state, Ordering::Relaxed);
    k
}

// Random operands

/// `len` random bytes, in runs of random bytes, zero bytes and `0xFF` bytes, so that long
/// carries and borrows occur, across 8-byte words as well.
pub(super) fn random_bytes(state: &mut u64, len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len);
    while bytes.len() < len {
        let run = (1 + below(state, 24)).min(len - bytes.len());
        match below(state, 4) {
            0 => bytes.resize(bytes.len() + run, 0),
            1 => bytes.resize(bytes.len() + run, 0xFF),
            _ => bytes.extend((0..run).map(|_| xorshift(state) as u8)),
        }
    }
    bytes
}

/// A random magnitude of exactly `len` bytes up to its most significant non-zero byte, least
/// significant first: empty for 0.
pub(super) fn random_magnitude(state: &mut u64, len: usize) -> Vec<u8> {
    let mut bytes = random_bytes(state, len);
    if bytes.last() == Some(&0) {
        bytes[len - 1] = 1 + below(state, 255) as u8;
    }
    bytes
}

/// A random operand of up to `max_len` bytes, one time in 10 of at most 17 and one time in 40
/// of a few thousand, with a random sign and byte order, and one time in four not in the
/// shortest form: with up to 9 zero bytes above its most significant byte. Zero may be empty,
/// and negative.
pub(super) fn random_operand(state: &mut u64, max_len: usize) -> Number {
    let len = match below(state, 40) {
        0 => size(2000, 200) + below(state, size(3000, 100)),
        1..=4 => below(state, 18),
        _ => below(state, max_len + 1),
    };
    let mut magnitude = random_magnitude(state, len);
    if below(state, 4) == 0 {
        magnitude.resize(len + 1 + below(state, 9), 0);
    }
    number(&magnitude, below(state, 2) == 0, ORDERS[below(state, 2)])
}

/// A random `i128` whose magnitude is below 2^126, so that every sum of two fits: of a random
/// bit length, with every bit set, or a power of two, and with a random sign.
pub(super) fn random_i128(state: &mut u64) -> i128 {
    let bits = below(state, 127) as u32;
    let random = (u128::from(xorshift(state)) << 64) | u128::from(xorshift(state));
    let magnitude: u128 = match below(state, 4) {
        0 => (1 << bits) - 1,
        1 => 1 << bits.min(125),
        _ => random.checked_shr(128 - bits).unwrap_or(0),
    };
    if below(state, 2) == 0 {
        magnitude as i128
    } else {
        -(magnitude as i128)
    }
}

// Where a result is built

/// Where the allocation promises put a result.
pub(super) enum Placement {
    /// In the vector of the left operand, allocating nothing.
    Left,
    /// In the vector of the right operand, allocating nothing.
    Right,
    /// In one new vector.
    New,
}

/// Operand pairs `(roomy, other)` whose sum fits in the vector of `roomy` with room to spare:
/// carries that make the sum longer, borrows that make it shorter, `roomy` the shorter
/// operand, mixed byte orders, a zero sum, and `roomy` not in the shortest form. Values are
/// written as stored.
pub(super) fn sums_with_room() -> Vec<(Number, Number)> {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    let mut long = vec![0xFF; 100];
    long[0] = 0x12;
    vec![
        // 0xFF + 1 = 0x01_00, one byte longer, in either byte order.
        (plus(&[0xFF], be, 4), plus(&[1], be, 1)),
        (plus(&[0xFF], le, 4), plus(&[1], le, 1)),
        (plus(&[0xFF], be, 4), plus(&[1], le, 1)),
        // A carry through 20 bytes.
        (plus(&[0xFF; 20], be, 32), plus(&[1], le, 1)),
        (minus(&[0xFF; 20], le, 32), minus(&[1], be, 1)),
        // 0x01_00 - 1 = 0xFF, one byte shorter.
        (plus(&[1, 0], be, 2), minus(&[1], be, 1)),
        (plus(&[0, 1], le, 2), minus(&[1], be, 1)),
        // 0x09_08_07_06_05 - 0x09_08_07_06_00 = 5, four bytes shorter: in big-endian order the
        // byte moves to the front.
        (
            plus(&[9, 8, 7, 6, 5], be, 5),
            minus(&[0, 6, 7, 8, 9], le, 5),
        ),
        (
            minus(&[5, 6, 7, 8, 9], le, 5),
            plus(&[9, 8, 7, 6, 0], be, 5),
        ),
        // `roomy` is the shorter operand: 1 + 0xFF_FF_FF = 0x01_00_00_00, and
        // 1 - 0x01_00_00 = -0xFF_FF.
        (plus(&[1], be, 8), plus(&[0xFF; 3], be, 3)),
        (plus(&[1], le, 8), plus(&[0xFF; 3], be, 3)),
        (plus(&[1], be, 4), minus(&[0, 0, 1], le, 3)),
        (plus(&[1], le, 4), minus(&[1, 0, 0], be, 3)),
        // 0x07_07 - 0x07_07 = 0.
        (minus(&[7, 7], be, 2), plus(&[7, 7], le, 2)),
        // Not in the shortest form: leading zeros in big-endian order, trailing zeros in
        // little-endian order, an empty vector, and a negative zero.
        (plus(&[0, 0, 0xFF], be, 3), plus(&[1], be, 1)),
        (plus(&[0xFF, 0, 0], le, 3), plus(&[1], le, 1)),
        (plus(&[0, 0, 1, 0], be, 4), minus(&[1], le, 1)),
        (plus(&[0, 1, 0, 0], le, 4), minus(&[1], be, 1)),
        (plus(&[], be, 4), plus(&[6, 5], le, 2)),
        (minus(&[0, 0], le, 2), plus(&[3], be, 1)),
        // Long numbers in mixed byte orders.
        (plus(&long, be, 101), plus(&long, le, 100)),
        (minus(&long, le, 120), plus(&long[..60], be, 60)),
    ]
}

/// Operand pairs `(roomy, other)` whose sum just fits in the vector of `roomy`: its capacity is
/// the length of the sum, while the other operand may be longer. Values are written as
/// stored.
pub(super) fn sums_that_just_fit() -> Vec<(Number, Number)> {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    vec![
        // 0xFF + 1 = 0x01_00 in two bytes.
        (plus(&[0xFF], be, 2), plus(&[1], le, 1)),
        (plus(&[0xFF], le, 2), plus(&[1], be, 1)),
        // 0xFF_FF_FF_FF_FF_FF_FF_FF + 1 in nine bytes.
        (plus(&[0xFF; 8], le, 9), plus(&[1], be, 1)),
        (minus(&[0xFF; 8], be, 9), minus(&[1], le, 1)),
        // 1 + 1 = 2 in one byte: no room for a carry that does not come.
        (plus(&[1], be, 1), plus(&[1], be, 1)),
        (plus(&[1], le, 1), plus(&[1], be, 1)),
        // 5 - 0x01_00 = -0xFB in one byte: the other operand is longer than the vector, the sum
        // is not.
        (plus(&[5], be, 1), minus(&[1, 0], be, 2)),
        (plus(&[5], le, 1), minus(&[0, 1], le, 2)),
        (plus(&[5], be, 1), minus(&[0, 1], le, 2)),
        // 1 - 0x01_00_00_00_00_00_00_00_00 = -0xFF_FF_FF_FF_FF_FF_FF_FF in eight bytes.
        (
            plus(&[1], be, 8),
            minus(&[1, 0, 0, 0, 0, 0, 0, 0, 0], be, 9),
        ),
        (
            plus(&[1], le, 8),
            minus(&[1, 0, 0, 0, 0, 0, 0, 0, 0], be, 9),
        ),
        // 1 + 0x00_00_00_01 = 2 in one byte: the zero bytes of the other operand do not count.
        (plus(&[1], be, 1), plus(&[0, 0, 0, 1], be, 4)),
        (plus(&[1], le, 1), plus(&[1, 0, 0, 0], le, 4)),
        // 7 - 7 = 0 and 0 + 0 = 0 in one byte.
        (plus(&[7], le, 1), minus(&[7], be, 1)),
        (plus(&[], be, 1), minus(&[], le, 0)),
        (minus(&[], le, 1), minus(&[0, 0, 0], be, 3)),
    ]
}

/// Operand pairs whose sum fits in neither vector. Values are written as stored.
pub(super) fn sums_without_room() -> Vec<(Number, Number)> {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    vec![
        // 0xFF + 1 = 0x01_00.
        (plus(&[0xFF], be, 1), plus(&[1], le, 1)),
        // 0 + 0 = [0], from two vectors that hold nothing.
        (plus(&[], be, 0), minus(&[], le, 0)),
        // -0xFF_FF_FF_FF_FF_FF_FF_FF + -0xFF_FF_FF_FF_FF_FF_FF_FF = -0x01_FF_FF_FF_FF_FF_FF_FF_FE.
        (minus(&[0xFF; 8], le, 8), minus(&[0xFF; 8], be, 8)),
        // A carry out of 100 bytes.
        (plus(&[0xFF; 100], be, 100), plus(&[0xFF; 100], le, 100)),
    ]
}

// Long operands, through every path of the kernels

/// Numbers of 8-byte limbs around the sizes at which the kernels change shape: the steps of 1,
/// 2 and 4 limbs and the loop of 8 on one carry chain; two interleaved chains from 32 limbs
/// little-endian (64 big-endian) and four from 64, with their rounds of 16-limb and 8-limb
/// blocks and the limbs left over; two chains again from 7168 limbs little-endian, and the
/// sizes around that, where the parts of four chains would meet in the cache. Under Miri,
/// which runs the portable kernels instead, up to 33 limbs.
pub(super) fn limb_counts() -> Vec<usize> {
    [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 95, 96, 97, 127, 128,
        129, 1023, 1024, 1025, 7167, 7168, 7169, 8191, 8192, 8193, 16384,
    ]
    .into_iter()
    .filter(|&limbs| limbs <= size(usize::MAX, 33))
    .collect()
}

/// The bytes above the whole limbs of the operands of the tests on long operands: none, and
/// a few that the kernels leave to the bytewise steps.
pub(super) const EXTRA_BYTES: [usize; 3] = [0, 3, 7];

/// A random magnitude of `limbs` whole limbs and `extra` bytes above them, least significant
/// byte first, whose limbs come in runs of all zeros, all ones or random bytes. A run is 1 to
/// 300 limbs long, or a multiple of 8 limbs, as the parts that the kernels split long operands
/// into are, so that the carries and borrows through these runs end at arbitrary limbs, the
/// first and last limbs of the parts included. The top byte is not zero.
pub(super) fn limb_runs(state: &mut u64, limbs: usize, extra: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 * limbs + extra);
    while bytes.len() < 8 * limbs {
        let run = if below(state, 2) == 0 {
            8 * (1 + below(state, 37))
        } else {
            1 + below(state, 300)
        };
        let run = run.min(limbs - bytes.len() / 8);
        match below(state, 3) {
            0 => bytes.resize(bytes.len() + 8 * run, 0),
            1 => bytes.resize(bytes.len() + 8 * run, 0xFF),
            _ => (0..run).for_each(|_| bytes.extend(xorshift(state).to_le_bytes())),
        }
    }
    bytes.extend((0..extra).map(|_| xorshift(state) as u8));
    if let Some(top @ 0) = bytes.last_mut() {
        *top = 1 + below(state, 255) as u8;
    }
    bytes
}
