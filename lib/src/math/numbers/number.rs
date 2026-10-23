use super::ByteOrder;

/// An integer of any size, stored as its sign and the bytes of its absolute value.
///
/// `value` holds the absolute value (the magnitude) of the integer as bytes, in the order
/// `byte_order` names: the most significant byte first for [`ByteOrder::BigEndian`], the
/// least significant first for [`ByteOrder::LittleEndian`]. `negative` holds the sign. This
/// is the sign-and-magnitude form most big-integer libraries use: `-2` is `[2]` with
/// `negative` set, not the two's-complement byte `[0xFE]`.
///
/// The constructors keep the shortest form: they drop the zero bytes above the most
/// significant non-zero byte and keep one byte for zero, so `1` and `-1` are `[1]`, and `0` is
/// `[0]`, in either order. An empty `value` is zero as well; the constructors keep an empty
/// vector empty. Zero is never negative: there is one zero, not a positive and a negative one.
///
/// The fields are public, so a `Number` can also be built directly; it is then stored exactly
/// as given, zero bytes and the sign of zero included.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number};
///
/// let big = Number::new_from_i32(-0x0001_0203, ByteOrder::BigEndian);
/// assert_eq!(big.value, [0x01, 0x02, 0x03]);
/// assert!(big.is_negative());
///
/// let little = Number::new_from_u32(0x0001_0203, ByteOrder::LittleEndian);
/// assert_eq!(little.value, [0x03, 0x02, 0x01]);
/// assert!(!little.is_negative());
/// ```
pub struct Number {
    /// The bytes of the absolute value of the number, in the order of `byte_order`.
    pub value: Vec<u8>,
    /// Whether the number is below zero; `false` for zero.
    pub negative: bool,
    /// The order of the bytes in `value`.
    pub byte_order: ByteOrder,
}

impl Number {
    /// Creates a non-negative number from its bytes in `byte_order`, dropping the zero bytes
    /// above the most significant non-zero byte: the leading zeros in big-endian order, the
    /// trailing zeros in little-endian order.
    ///
    /// Zero bytes below the most significant non-zero byte are part of the number and stay.
    /// A number whose bytes are all zero keeps one of them, `[0]`; an empty vector stays
    /// empty.
    ///
    /// Takes `O(n)` time for `n` bytes and reuses the allocation of `value`. For a negative
    /// number, see [`new_signed`](Self::new_signed).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new(vec![0, 0, 1, 0, 2], ByteOrder::BigEndian);
    /// assert_eq!(n.value, [1, 0, 2]);
    ///
    /// let n = Number::new(vec![2, 0, 1, 0, 0], ByteOrder::LittleEndian);
    /// assert_eq!(n.value, [2, 0, 1]);
    ///
    /// let zero = Number::new(vec![0, 0, 0], ByteOrder::BigEndian);
    /// assert_eq!(zero.value, [0]);
    /// ```
    pub fn new(mut value: Vec<u8>, byte_order: ByteOrder) -> Self {
        match byte_order {
            // The most significant byte comes first: drop everything before the first
            // non-zero byte, or all but the last byte when every byte is zero.
            ByteOrder::BigEndian => {
                let start = value
                    .iter()
                    .position(|&b| b != 0)
                    .unwrap_or(value.len().saturating_sub(1));
                value.drain(..start);
            }
            // The most significant byte comes last: cut everything after the last non-zero
            // byte, or all but the first byte when every byte is zero.
            ByteOrder::LittleEndian => {
                let end = value.iter().rposition(|&b| b != 0).map_or(1, |i| i + 1);
                value.truncate(end);
            }
        }
        Self {
            value,
            negative: false,
            byte_order,
        }
    }

    /// Creates a number from the bytes of its absolute value in `byte_order` and its sign,
    /// dropping the high zero bytes as [`new`](Self::new) does.
    ///
    /// The number is negative when `negative` is set and the bytes are not all zero: zero is
    /// never negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_signed(vec![0, 1, 0], true, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [1, 0]);
    /// assert!(n.is_negative());
    ///
    /// let zero = Number::new_signed(vec![0, 0], true, ByteOrder::BigEndian);
    /// assert_eq!(zero.value, [0]);
    /// assert!(!zero.is_negative());
    /// ```
    pub fn new_signed(value: Vec<u8>, negative: bool, byte_order: ByteOrder) -> Self {
        let number = Self::new(value, byte_order);
        // Once trimmed, zero is `[0]` or `[]` in either order: it keeps the sign of `new`.
        let negative = negative && !matches!(number.value.as_slice(), [] | [0]);
        Self { negative, ..number }
    }

    /// Creates a number from a `u8`, its only byte. Zero is kept as `[0]`.
    pub fn new_from_u8(value: u8, byte_order: ByteOrder) -> Self {
        Self::new(vec![value], byte_order)
    }

    /// Creates a number from the bytes of a `u16` in `byte_order`, without the high zero
    /// bytes (see [`new`](Self::new)).
    pub fn new_from_u16(value: u16, byte_order: ByteOrder) -> Self {
        let bytes = match byte_order {
            ByteOrder::BigEndian => value.to_be_bytes().to_vec(),
            ByteOrder::LittleEndian => value.to_le_bytes().to_vec(),
        };
        Self::new(bytes, byte_order)
    }

    /// Creates a number from the bytes of a `u32` in `byte_order`, without the high zero
    /// bytes (see [`new`](Self::new)).
    pub fn new_from_u32(value: u32, byte_order: ByteOrder) -> Self {
        let bytes = match byte_order {
            ByteOrder::BigEndian => value.to_be_bytes().to_vec(),
            ByteOrder::LittleEndian => value.to_le_bytes().to_vec(),
        };
        Self::new(bytes, byte_order)
    }

    /// Creates a number from the bytes of a `u64` in `byte_order`, without the high zero
    /// bytes (see [`new`](Self::new)).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_u64(0x0A0B_0C0D, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x0A, 0x0B, 0x0C, 0x0D]);
    ///
    /// let n = Number::new_from_u64(0x0A0B_0C0D, ByteOrder::LittleEndian);
    /// assert_eq!(n.value, [0x0D, 0x0C, 0x0B, 0x0A]);
    ///
    /// let zero = Number::new_from_u64(0, ByteOrder::LittleEndian);
    /// assert_eq!(zero.value, [0]);
    /// ```
    pub fn new_from_u64(value: u64, byte_order: ByteOrder) -> Self {
        let bytes = match byte_order {
            ByteOrder::BigEndian => value.to_be_bytes().to_vec(),
            ByteOrder::LittleEndian => value.to_le_bytes().to_vec(),
        };
        Self::new(bytes, byte_order)
    }

    /// Creates a number from the bytes of a `u128` in `byte_order`, without the high zero
    /// bytes (see [`new`](Self::new)).
    pub fn new_from_u128(value: u128, byte_order: ByteOrder) -> Self {
        let bytes = match byte_order {
            ByteOrder::BigEndian => value.to_be_bytes().to_vec(),
            ByteOrder::LittleEndian => value.to_le_bytes().to_vec(),
        };
        Self::new(bytes, byte_order)
    }

    /// Creates a number from the bytes of a `usize` in `byte_order`, without the high zero
    /// bytes (see [`new`](Self::new)).
    ///
    /// `usize` is as wide as a pointer, but as the high zero bytes are dropped, a value has
    /// the same bytes on every target.
    pub fn new_from_usize(value: usize, byte_order: ByteOrder) -> Self {
        let bytes = match byte_order {
            ByteOrder::BigEndian => value.to_be_bytes().to_vec(),
            ByteOrder::LittleEndian => value.to_le_bytes().to_vec(),
        };
        Self::new(bytes, byte_order)
    }

    /// Creates a non-negative number from bytes given as `i8`s in `byte_order`, without the
    /// high zero bytes (see [`new`](Self::new)).
    ///
    /// The `i8`s are only how the bytes are given, not signed digits: each one is the byte
    /// with the same bits (`-1` is `0xFF`, `-128` is `0x80`), and the number is never
    /// negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_i8_vec(vec![0, -1, 1], ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0xFF, 0x01]);
    /// assert!(!n.is_negative());
    /// ```
    pub fn new_from_i8_vec(value: Vec<i8>, byte_order: ByteOrder) -> Self {
        Self::new(value.iter().map(|&x| x as u8).collect(), byte_order)
    }

    /// Creates a number from an `i8`: its sign, and its absolute value as the only byte.
    /// `i8::MIN` becomes `[0x80]` with `negative` set. Zero is kept as `[0]`.
    pub fn new_from_i8(value: i8, byte_order: ByteOrder) -> Self {
        // `unsigned_abs` is exact for every value, `MIN` included, whose absolute value does
        // not fit in the signed type; the same holds for the wider types below.
        Self {
            negative: value < 0,
            ..Self::new_from_u8(value.unsigned_abs(), byte_order)
        }
    }

    /// Creates a number from an `i16`: its sign, and the bytes of its absolute value in
    /// `byte_order` without the high zero bytes (see [`new`](Self::new)).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_i16(-2, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x02]);
    /// assert!(n.is_negative());
    ///
    /// let n = Number::new_from_i16(i16::MIN, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x80, 0x00]);
    /// assert!(n.is_negative());
    /// ```
    pub fn new_from_i16(value: i16, byte_order: ByteOrder) -> Self {
        Self {
            negative: value < 0,
            ..Self::new_from_u16(value.unsigned_abs(), byte_order)
        }
    }

    /// Creates a number from an `i32`: its sign, and the bytes of its absolute value in
    /// `byte_order` without the high zero bytes (see [`new`](Self::new)).
    pub fn new_from_i32(value: i32, byte_order: ByteOrder) -> Self {
        Self {
            negative: value < 0,
            ..Self::new_from_u32(value.unsigned_abs(), byte_order)
        }
    }

    /// Creates a number from an `i64`: its sign, and the bytes of its absolute value in
    /// `byte_order` without the high zero bytes (see [`new`](Self::new)).
    pub fn new_from_i64(value: i64, byte_order: ByteOrder) -> Self {
        Self {
            negative: value < 0,
            ..Self::new_from_u64(value.unsigned_abs(), byte_order)
        }
    }

    /// Creates a number from an `i128`: its sign, and the bytes of its absolute value in
    /// `byte_order` without the high zero bytes (see [`new`](Self::new)).
    pub fn new_from_i128(value: i128, byte_order: ByteOrder) -> Self {
        Self {
            negative: value < 0,
            ..Self::new_from_u128(value.unsigned_abs(), byte_order)
        }
    }

    /// Creates a number from an `isize`: its sign, and the bytes of its absolute value in
    /// `byte_order` without the high zero bytes (see [`new`](Self::new)).
    ///
    /// As for [`new_from_usize`](Self::new_from_usize), a value has the same bytes on every
    /// target.
    pub fn new_from_isize(value: isize, byte_order: ByteOrder) -> Self {
        Self {
            negative: value < 0,
            ..Self::new_from_usize(value.unsigned_abs(), byte_order)
        }
    }

    /// Creates a non-negative number from the UTF-8 bytes of a string, read in `byte_order`,
    /// without the high zero bytes (see [`new`](Self::new)).
    ///
    /// The string is not parsed as digits: its bytes are the number, the way textbook RSA
    /// turns a message into an integer, so a leading `'-'` is the byte `0x2D`, not a sign.
    /// In big-endian order the first byte of the string is the most significant one, and
    /// leading `'\0'` characters are dropped as zero bytes; in little-endian order the first
    /// byte is the least significant one, and trailing `'\0'` characters are dropped. An empty
    /// string gives an empty `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x31_32_33 = 3,224,115, not 123.
    /// let n = Number::new_from_str("123", ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x31, 0x32, 0x33]);
    /// ```
    pub fn new_from_str(value: &str, byte_order: ByteOrder) -> Self {
        Self::new(value.as_bytes().to_vec(), byte_order)
    }

    /// Returns whether the number is below zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert!(Number::new_from_i32(-7, ByteOrder::BigEndian).is_negative());
    /// assert!(!Number::new_from_i32(7, ByteOrder::BigEndian).is_negative());
    /// assert!(!Number::new_from_i32(0, ByteOrder::BigEndian).is_negative());
    /// ```
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Returns the order of the bytes in `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_u8(7, ByteOrder::LittleEndian);
    /// assert!(n.byte_order() == ByteOrder::LittleEndian);
    /// ```
    pub fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }
}
