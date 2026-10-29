[A](../../../README.md) › [Math](../../../README.md#math)

# Addition

[Numbers](numbers.md) add with `+` and `+=`, each operand owned or borrowed: `a + b`, `a + &b`, `&a + b` and `&a + &b` give a new `Number`, and `a += b` and `a += &b` add to `a`. The six forms give the same result for operands of any lengths, signs and byte orders; they differ only in the vector the sum is built in.

- The sum is exact. Each magnitude is read in its own byte order, and zero bytes above its most significant non-zero byte are ignored; an empty vector is zero, and a magnitude of zero is zero whatever `negative` says. Numbers built from the public fields, which need not be in the shortest form, add as the values they hold.
- The sum is in the byte order of the left operand in every form, `&a + b` included, and `+=` keeps the order of the number it changes. The two operands' orders may differ.
- The sum is in the shortest form: no zero bytes above the most significant one, and zero is `[0]`, never empty (a constructor keeps an empty vector empty) and never negative.
- The sum is built in an operand's vector when its capacity is at least the length of the sum (one byte for zero), and then nothing at all is allocated: in `a`'s vector for `a += &b`, `a += b`, `a + &b` and `a + b`; in `b`'s for `&a + b`, its bytes reversed first if the orders differ, and for `a + b` and `a += b` when `a`'s is too small. When no vector the form may use is large enough, and always for `&a + &b`, exactly one new vector is allocated.

The method is the schoolbook one: when the signs agree, the magnitudes are added and the sum takes their sign; when they differ, the smaller magnitude is subtracted from the larger and the sum takes the larger one's sign. This implementation carries it out as follows:

- Eight bytes at a time: the magnitudes are added or subtracted in limbs of 8 bytes, read unaligned straight from the vectors in either byte order; the bytes of the shorter operand that do not fill a limb go as one more limb, read from below their start and shifted, when there are at least three of them and a whole limb under them, and one at a time otherwise. Above the shorter operand, a carry or borrow goes on through the longer one a limb at a time while it lasts, and the rest of the longer one is copied, unless the sum is written over it at the same place. A big-endian limb has its bytes reversed as it is loaded, and the result again as it is stored (`rev` on aarch64, `bswap` on x86_64).
- The length first: before any byte of the sum is computed, its length (and, when the signs differ, which magnitude is the larger) is worked out from the most significant bytes down, usually in one step of 8 bytes and at most once over each operand. Only the bytes the sum keeps are then computed, and the vector that will hold them is chosen before anything is written: that is how `a += &b` grows or shrinks `a` in its own vector without reallocating.
- No branch on a coin toss: for random operands of the same length, whether the sum carries out of the top is a coin toss, and so is the carry (borrow) out of a shorter operand into the longer one. A branch on either is mispredicted half the time, at about 15 cycles each on the Apple M4 Max, half of what a sum of 32 bytes costs. So the top byte of the sum is written without a branch, as its planned length says, and the carry goes into the longer operand's next limb, or into the fewer than eight bytes above the shorter one as one more limb, without a branch too; only a carry that gets through that limb, all ones (all zeros), goes further.
- Long differences start at once: which magnitude is the larger is a coin toss for random operands of the same length too, and so a subtraction's operands are picked without a branch. But then the kernel cannot load a limb before the most significant limbs of both operands have been read and compared, which costs 12–23 cycles once the operands are out of the first-level cache. From 256 bytes little-endian and 512 big-endian, where the kernels run several carry chains, a difference built in an operand's vector picks them by branches instead: a predicted branch lets the kernel start while the comparison is still under way, and a mispredicted one costs about what the wait did. A difference whose operand is reversed into the other byte order first keeps the branch-free choice, as the reversal takes longer than the comparison.
- The carry in the carry flag: the carry (the borrow, in a subtraction) goes from limb to limb in the processor's carry flag, through inline assembly on 64-bit little-endian aarch64 (`adcs`, `sbcs`) and on x86_64 (`adc`, `sbb`), in loops whose counters do not touch the flag. LLVM compiles the same loop in portable Rust with a counter that overwrites the flag, and so rebuilds the carry from a register in every iteration: about 4 cycles per limb on the Apple M4 Max, and 1.75 unrolled by four, against 1 for the chain through the flag.
- Several carry chains on aarch64: one chain is bound by the latency of the carry, one cycle per limb, while the core can issue three loads and four flag-writing instructions per cycle. From 32 limbs (256 bytes) little-endian and 64 limbs (512 bytes) big-endian, the limbs are split into 2 or 4 parts, each but the first added as if no carry came into it, and the parts' carry chains run interleaved, a block of limbs of each in turn; the core renames the flags, so the chains overlap. Each part's carry then goes into the next part: it is added to that part's first limb without a branch, and goes further only through limbs that came out all ones (all zeros, in a subtraction), which random limbs almost never do. Little-endian uses two chains from 32 limbs, four from 64 and two again from 7168 (56 KiB), where the twelve memory streams of four chains can meet in the same sets of the first-level cache (four chains took 1.3–1.9 cycles per limb at 8192 and 16,384 limbs, two 0.8–1.0). Big-endian uses two chains from 64 limbs, as fast as four: with three byte reversals per limb, its loop is bound by how many instructions the core issues. A big-endian sum that grows in its own vector is written a few bytes above the operand, where a part would overwrite limbs of the part below it before they are read, so it takes one chain.
- One chain on x86_64: one `adc`/`sbb` chain, four limbs per round, the loop shape GMP uses; the pointers move with `lea` and the count steps with `dec`, neither of which writes the carry flag. It has been tested for correctness under emulation only; its speed has not been measured.
- Portable kernels elsewhere: the other targets, the ILP32 and big-endian aarch64 ones among them, and Miri, which cannot run inline assembly, use kernels in plain Rust: limbs of 8 bytes, the carry from pairs of `overflowing_add` (`overflowing_sub` in a subtraction), unrolled by four.

The aarch64 kernels alone, not whole additions, were measured on an Apple M4 Max at 4.0–4.45 GHz (the median of 11 interleaved runs, random limbs, the result in a separate vector), against one chain and GMP 6.3's `mpn_add_n`, which runs one chain, both little-endian. The figures are cycles per limb for calls that do not depend on each other, which the core overlaps, and, in parentheses, for calls that each wait for the carry out of the one before:

| Limbs (bytes) | One chain | Little-endian | Big-endian | GMP |
| --- | --- | --- | --- | --- |
| 8 (64) | 0.66 (1.46) | 0.79 (1.47) | 1.05 (1.49) | 0.93 (1.71) |
| 32 (256) | 0.53 (1.18) | 0.65 (0.69) | 0.81 (1.17) | 0.59 (1.24) |
| 64 (512) | 0.63 (1.11) | 0.61 (0.65) | 0.86 (0.88) | 0.70 (1.14) |
| 256 (2 KiB) | 0.93 (1.11) | 0.49 (0.52) | 0.76 (0.78) | 0.94 (1.11) |
| 1024 (8 KiB) | 1.04 (1.09) | 0.45 (0.47) | 0.73 (0.76) | 1.04 (1.10) |
| 65,536 (512 KiB) | 1.08 (1.06) | 0.88 (0.86) | 1.10 (1.06) | 1.08 (1.06) |

- Little-endian, from 256 to 4096 limbs, the kernels take about half the cycles of one chain or of GMP, near the core's limit for the loads and stores alone (0.49–0.53 cycles per limb with no arithmetic). Big-endian takes 0.73–0.76 cycles per limb there, against 0.97–1.05 for one big-endian chain.
- With several chains, a call does not wait for the one before it to finish its carry chain: its carry out depends on its carry in only through the hand-offs between parts, whose branches the core predicts, so dependent calls cost about what independent ones do.
- With the result written over the left operand, little-endian takes 0.46 cycles per limb at 1024 limbs and 0.69 at 65,536, big-endian 0.76 and 0.91.
- From 16,384 limbs (128 KiB), past the first-level cache, out of place, little-endian takes 0.86–0.92 cycles per limb up to a million limbs, against 1.07–1.09 for one chain, and big-endian is no faster than one chain (1.09–1.20 against 1.04–1.09); runs of that size varied by up to 10%.
- Subtraction takes within about 5% of the time of addition.
- A carry that has to go through whole parts, as when all ones are added to a number of the same length whose middle limbs are zero, costs about 0.4 cycles per limb it passes: 0.75–0.78 cycles per limb in all from 256 to 4096 limbs, against 0.95–1.04 for one chain, but 0.97 and 1.13 at 32 and 64 limbs, against 0.52 and 0.60.

Whole additions were measured on the same machine as `a += &b` with room in `a`, over 4 MiB of operand pairs of the length given, random magnitudes in the shortest form, against GMP 6.3's `mpz_add(a, a, b)` on the same values (the median of 15 interleaved runs, in cycles per call):

| Bytes | Same signs, little-endian | Same signs, big-endian | Opposite signs, little-endian | Opposite signs, big-endian | GMP, same signs | GMP, opposite signs |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 24 | 30 | 28 | 32 | 14 | 14 |
| 32 | 25 | 30 | 25 | 32 | 17 | 14 |
| 64 | 26 | 30 | 30 | 36 | 20 | 17 |
| 128 | 31 | 37 | 34 | 44 | 34 | 34 |
| 256 | 48 | 53 | 66 | 68 | 64 | 62 |
| 512 | 82 | 98 | 89 | 116 | 101 | 100 |
| 1024 | 132 | 196 | 163 | 208 | 165 | 158 |
| 4096 | 482 | 672 | 496 | 651 | 617 | 612 |

- Up to 16 bytes, where the sum is computed in registers, a call takes 13–21 cycles, as GMP's take 12–22.
- From 17 to 64 bytes a call takes 6–19 cycles more than GMP's, for what GMP's limb counts, kept normalised, spare it: the significant lengths of the operands and the plan of the sum, worked out before it is computed.
- `&a + &b` costs about 50 cycles more than `a += &b`, for allocating the sum's vector and freeing it, with the system allocator of macOS.

## Examples

Rust usage: `1000 + -1500` is `-500`, the bytes `[0x01, 0xF4]` with `negative` set:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(1000, ByteOrder::BigEndian);
  let b = Number::new_from_i32(-1500, ByteOrder::BigEndian);
  let sum = a + b;
  assert_eq!(sum.value, vec![0x01, 0xF4]);
  assert!(sum.is_negative());
}
```

`+=` grows a number in its own vector when the vector has room: in big-endian order, `0xFF + 1` is `[1, 0]`:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut value = Vec::with_capacity(2);
  value.push(0xFF);
  let mut n = Number::new(value, ByteOrder::BigEndian);
  let vector = n.value.as_ptr();

  let one = Number::new_from_u8(1, ByteOrder::BigEndian);
  n += &one;
  assert_eq!(n.value, vec![1, 0]);
  assert_eq!(n.value.as_ptr(), vector);
}
```

Mixed byte orders: the sum is in the order of the left operand:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let big = Number::new_from_u16(0x0102, ByteOrder::BigEndian);
  let little = Number::new_from_u16(0x0304, ByteOrder::LittleEndian);

  let sum = &big + &little;
  assert!(sum.byte_order() == ByteOrder::BigEndian);
  assert_eq!(sum.value, vec![0x04, 0x06]);

  let sum = &little + &big;
  assert!(sum.byte_order() == ByteOrder::LittleEndian);
  assert_eq!(sum.value, vec![0x06, 0x04]);
}
```

Zero is `[0]` and never negative, whether the operands cancel out or are zeros built from the fields with a sign:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let a = Number::new_from_i32(-1000, ByteOrder::LittleEndian);
  let b = Number::new_from_i32(1000, ByteOrder::LittleEndian);
  let zero = a + b;
  assert_eq!(zero.value, vec![0]);
  assert!(!zero.is_negative());

  let zeros = Number { value: vec![0, 0], negative: true, byte_order: ByteOrder::BigEndian };
  let empty = Number { value: Vec::new(), negative: true, byte_order: ByteOrder::BigEndian };
  let zero = zeros + empty;
  assert_eq!(zero.value, vec![0]);
  assert!(!zero.is_negative());
}
```

Borrowed and owned operands: `&a + &b` builds the sum in a new vector and leaves `a` as it was, and `a + &b` builds the same sum in `a`'s vector, which has room for it:

```rust
use a::math::numbers::{ByteOrder, Number};

fn main() {
  let mut value = Vec::with_capacity(8);
  value.extend_from_slice(&[0xFF, 0xFF]);
  let a = Number::new(value, ByteOrder::LittleEndian);
  let b = Number::new_from_u16(0x0101, ByteOrder::LittleEndian);

  let sum = &a + &b;
  assert_eq!(sum.value, vec![0x00, 0x01, 0x01]);
  assert_eq!(a.value, vec![0xFF, 0xFF]);

  let vector = a.value.as_ptr();
  let sum = a + &b;
  assert_eq!(sum.value, vec![0x00, 0x01, 0x01]);
  assert_eq!(sum.value.as_ptr(), vector);
}
```

## Efficiency

- Time complexity: O(n) for n bytes of the longer operand, whatever the signs and byte orders
  - The length of the sum: worked out first, from the most significant bytes, usually in one step of 8
  - The sum: only the bytes it keeps, mostly 8 at a time. When the byte orders agree and the sum is built in the longer operand's vector, that operand's bytes above the shorter one are touched only as far as a carry or borrow runs into them; big-endian, a sum longer or shorter than the vector was moves them all
- Space complexity: nothing besides the vector of the result; no allocation at all when the sum is built in an operand's vector, otherwise exactly one, of a new vector (always for `&a + &b`, a sum of zero included)
- Soundness: the bytes beyond a vector's length, which may be uninitialised, are never read; a borrowed operand is left as it was, and where the sum is written over an operand, each byte is read before it is overwritten
