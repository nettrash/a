/// The order of the bytes of the absolute value of a [`Number`](super::Number).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// The most significant byte first: `0x0102` is `[0x01, 0x02]`, as
    /// [`u16::to_be_bytes`] writes it.
    BigEndian,
    /// The least significant byte first: `0x0102` is `[0x02, 0x01]`, as
    /// [`u16::to_le_bytes`] writes it.
    LittleEndian,
}
