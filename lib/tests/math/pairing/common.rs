//! What the tests of the pairings share: the seeded generator and the digests of the reference
//! generators (`data/gen_ark.rs`, `data/gen_py.py`), and points outside the subgroups.

use a::math::field::{Fp, Fp2, Fp12};
use a::math::numbers::{ByteOrder, Number};
use a::math::pairing::{G1, G2, PairingCurve};

const REF: &str = include_str!("data/pairing_ref.txt");

/// The reference digest of `key`.
pub fn reference(key: &str) -> u64 {
    let line = REF
        .lines()
        .find(|l| l.split(' ').next() == Some(key))
        .unwrap_or_else(|| panic!("no reference {key}"));
    u64::from_str_radix(line.split(' ').nth(1).unwrap(), 16).unwrap()
}

/// SplitMix64, as the generators have it.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `n` words, the first the least significant.
    pub fn number(&mut self, n: usize) -> Number {
        let bytes: Vec<u8> = (0..n).flat_map(|_| self.next().to_le_bytes()).collect();
        Number::new(bytes, ByteOrder::LittleEndian)
    }

    /// An element of the curve's `F_p`, from one word more than `p` has, as the generators draw it.
    pub fn fp<'c>(&mut self, c: &'c PairingCurve) -> Fp<'c> {
        let words = c.base_field().bits().div_ceil(64) as usize + 1;
        c.base_field().element(&self.number(words))
    }

    pub fn fp2<'c>(&mut self, c: &'c PairingCurve) -> Fp2<'c> {
        let c0 = self.fp(c);
        let c1 = self.fp(c);
        Fp2::new(c.tower(), c0, c1)
    }
}

/// FNV-1a, 64 bits.
pub struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    pub fn eat(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn fp12(&mut self, x: &Fp12<'_>) {
        for c in x.coefficients() {
            self.eat(&be(&c.value(), c.field().bits().div_ceil(8) as usize));
        }
    }
}

/// `x` in big-endian order in `len` bytes.
pub fn be(x: &Number, len: usize) -> Vec<u8> {
    let mut b = x.value.clone();
    if x.byte_order == ByteOrder::LittleEndian {
        b.reverse();
    }
    let start = b.iter().position(|&v| v != 0).unwrap_or(b.len());
    let mut out = vec![0u8; len - (b.len() - start)];
    out.extend_from_slice(&b[start..]);
    out
}

/// The bytes of a hexadecimal string.
pub fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// The hexadecimal string of bytes.
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A number from hexadecimal.
pub fn num(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::BigEndian).unwrap()
}

/// Whether `y` is the larger of `±y`: above `(p − 1)/2`.
pub fn larger(y: &Fp<'_>) -> bool {
    y.value() * 2u8 > y.field().modulus() - 1u8
}

/// Whether `y` is the larger of `±y` in ZCash's order: by `c₁`, then by `c₀`.
pub fn larger2(y: &Fp2<'_>) -> bool {
    larger(y.c1()) || (y.c1().is_zero() && larger(y.c0()))
}

/// The point of `E(F_p)` with the least `x ≥ x₀` and the smaller `y`, as arkworks'
/// `get_point_from_x_unchecked(x, false)` gives it.
pub fn g1_from_x<'c>(c: &'c PairingCurve, mut x: Fp<'c>) -> G1<'c> {
    loop {
        if let Some(y) = (&(&x.square() * &x) + &c.b()).sqrt() {
            let y = if larger(&y) { -y } else { y };
            return G1::new_on_curve(c, x, y).unwrap();
        }
        x = &x + &c.base_field().one();
    }
}

/// The point of the twist with the least `x₀ + x₁u`, `x₀` counted up from the given, and the
/// smaller `y` in ZCash's order.
pub fn g2_from_x<'c>(c: &'c PairingCurve, mut x: Fp2<'c>) -> G2<'c> {
    let one = Fp2::one(c.tower());
    loop {
        if let Some(y) = (&(&x.square() * &x) + &c.twist_b()).sqrt() {
            let y = if larger2(&y) { -y } else { y };
            return G2::new_on_curve(c, x, y).unwrap();
        }
        x = &x + &one;
    }
}

/// The curves and their names in the references.
pub fn curves() -> [(&'static str, &'static PairingCurve); 2] {
    [
        ("bls", PairingCurve::bls12_381()),
        ("bn", PairingCurve::bn254()),
    ]
}
