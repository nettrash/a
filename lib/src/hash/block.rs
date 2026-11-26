//! The buffering of a Merkle–Damgård hash: the message cut into whole blocks for the compression
//! function, the rest held until more comes, and the padding of FIPS 180-4, section 5.1.

/// A partial block of `N` bytes and the length of the message so far.
#[derive(Clone, Debug)]
pub(super) struct Blocks<const N: usize> {
    /// The bytes after the last whole block.
    buffer: [u8; N],
    /// How many bytes of `buffer` are taken.
    filled: usize,
    /// The length of the message so far, in bytes.
    length: u128,
    /// The longest message, in bytes, whose length in bits fits the padding.
    limit: u128,
}

impl<const N: usize> Blocks<N> {
    /// An empty message, which may grow to `limit` bytes.
    pub(super) const fn new(limit: u128) -> Self {
        Blocks {
            buffer: [0; N],
            filled: 0,
            length: 0,
            limit,
        }
    }

    /// The buffer after one whole block has gone to the compression function: empty, with a
    /// length of `N` bytes.
    pub(super) const fn after_one_block(limit: u128) -> Self {
        Blocks {
            buffer: [0; N],
            filled: 0,
            length: N as u128,
            limit,
        }
    }

    /// Takes `data`, giving `compress` every block that is whole, several at once when it can.
    #[track_caller]
    pub(super) fn update(&mut self, mut data: &[u8], mut compress: impl FnMut(&[u8])) {
        // A `usize` length always fits 128 bits.
        let added = data.len() as u128;
        match self.length.checked_add(added) {
            Some(l) if l <= self.limit => self.length = l,
            _ => too_long(),
        }
        if self.filled > 0 {
            let take = (N - self.filled).min(data.len());
            self.buffer[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < N {
                return;
            }
            compress(&self.buffer);
            self.filled = 0;
        }
        let whole = data.len() - data.len() % N;
        if whole > 0 {
            compress(&data[..whole]);
        }
        let rest = &data[whole..];
        self.buffer[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    /// Pads the message as FIPS 180-4 does, a one bit, zeros and the length in bits as a
    /// big-endian number of `LEN` bytes, and gives the last blocks to `compress`.
    pub(super) fn finish<const LEN: usize>(mut self, mut compress: impl FnMut(&[u8])) {
        let bits = self.length << 3;
        let bits = bits.to_be_bytes();
        // The one bit, then zeros up to `N - LEN` bytes into a block.
        self.buffer[self.filled] = 0x80;
        self.filled += 1;
        if self.filled > N - LEN {
            self.buffer[self.filled..].fill(0);
            compress(&self.buffer);
            self.filled = 0;
        }
        self.buffer[self.filled..N - LEN].fill(0);
        self.buffer[N - LEN..].copy_from_slice(&bits[bits.len() - LEN..]);
        compress(&self.buffer);
    }
}

/// Panics for a message longer than the hash can take.
#[cold]
#[inline(never)]
#[track_caller]
fn too_long() -> ! {
    panic!("the message is longer than the hash can take")
}
