[A](../../../README.md) › [Math](../../../README.md#math)

# Pairings

`a::math::pairing` computes the optimal ate pairing `e: G1 × G2 → GT` on the curves BN254 and BLS12-381, with their groups: `G1` on the curve over `F_p`, `G2` on its sextic twist over `F_p²`, and `GT`, the `r`-th roots of unity of `F_p¹²`. Its values are exactly those of arkworks on both curves, and of the `bls12_381` crate, blst and RELIC on BLS12-381. The coordinates and values are elements of the [towers](field.md#towers) `Tower::bn254()` and `Tower::bls12_381()`.

**Nothing here is constant-time**: the time of a product of a point by a scalar, of a pairing, of a decoding and of a hashing to a group depends on the values. It is for mathematics and for verifying with public data, as in checking a signature or a proof, not for secret keys. There is no protocol layer: no BLS signatures, no key generation, no random numbers.

- `PairingCurve::bn254()` and `PairingCurve::bls12_381()` are the curves, built once, with their parameters (`x()`, `order()`, `base_field()`, `scalar_field()`, `tower()`, `b()`, `twist_b()`, the cofactors), their generators and identities, and the pairing: `pairing(&p, &q)`, `multi_pairing(&[(&p, &q), …])`, and its halves `miller_loop(&pairs)` and `final_exponentiation(&f)`. The free function `pairing(&p, &q)` takes the curve from the points.
- `G1` and `G2` are points of the curve and of the twist, which borrow their curve. Every way to get one checks that it is on the curve, and all but `new_on_curve` that it is in the subgroup of order `r`. They add with `+`, negate and subtract with `-`, double with `double()`, and multiply by a `Number` of any sign and length with `*`, owned or borrowed, with the assignments; `==` compares them in whatever projective form they are held. `coordinates()` gives the affine `(x, y)` or `None` at infinity.
- `is_in_subgroup()` tests membership by endomorphisms. `clear_cofactor()` maps any point of the curve into the subgroup: by RFC 9380's maps on BLS12-381, by Fuentes-Castañeda, Knapp and Rodríguez-Henríquez's map on BN254's `G2` (RFC 9380 has no BN254 suite), and as the identity on BN254's `G1`, whose cofactor is 1. `mul_by_cofactor()` multiplies by the whole cofactor, and `G2::psi()` is the endomorphism `ψ`.
- Bytes: ZCash's encodings of BLS12-381 (`to_zcash_compressed`, `to_zcash_uncompressed`, `G1::from_zcash`, `G2::from_zcash`) and Ethereum's of BN254 (`to_ethereum`, `from_ethereum`), each decoding checked, with a `PointError` that says what is wrong; and Ethereum's precompiles for BN254 on bytes, `ethereum::{ec_add, ec_mul, ec_pairing}`.
- `Gt` is written multiplicatively: `*`, `/`, `pow(&k)` (reduced modulo `r`), `square()`, `inverse()` (the conjugate), `is_identity()`, `==`; `Gt::from_fp12` accepts an element of `F_p¹²` of order dividing `r`, and `value()` gives it back. Powers go by the decomposition of the exponent along the Frobenius map (`f^p = f^x` on BLS12-381: four digits of 64 bits; `f^p = f^(6x²)` on BN254: four of about 64 bits, from an LLL-reduced basis of the lattice of the relations and Babai's rounding), raised together by cyclotomic squares; `from_fp12` tests the cyclotomic subgroup, then `f^p = f^x` (Scott, ePrint 2021/1130) or `f^p = f^(6x²)`, which decide membership because `gcd(Φ₁₂(p), p − x) = r` and `p − 6x² = r`.
- `a::math::pairing::hash` hashes byte strings to `G1` and `G2`: RFC 9380's four suites for BLS12-381, and gnark-crypto's two for BN254's `G1`, as RFC 9380 has none for BN254; see [Hashing to G1 and G2](#hashing-to-g1-and-g2), the section after the limits; the [measurements](#measured) close the page.

## Example

```rust
use a::math::numbers::Number;
use a::math::pairing::ethereum::ec_pairing;
use a::math::pairing::{G1, G2, PairingCurve, pairing};

let c = PairingCurve::bls12_381();
let (g1, g2) = (c.g1_generator(), c.g2_generator());

// Bilinearity: e(aP, bQ) = e(P, Q)^(ab).
let (a, b) = (Number::from(1234u32), Number::from(56789u32));
let e = pairing(&g1, &g2);
assert_eq!(pairing(&(&g1 * &a), &(&g2 * &b)), e.pow(&(&a * &b)));
assert!(e.pow(c.order()).is_identity() && !e.is_identity());

// The check of a BLS signature σ = [sk] H on a point H of G1 with the public key [sk] g₂:
// e(σ, g₂) · e(H, −pk) = 1, one Miller loop for both and one final exponentiation.
let sk = Number::from(0x5eed_u32);
let h = &g1 * &Number::from(777u32);
let (pk, sigma) = (&g2 * &sk, &h * &sk);
assert!(c.multi_pairing(&[(&sigma, &g2), (&h, &-&pk)]).is_identity());

// Points travel as ZCash's bytes, checked when they come back.
let bytes = sigma.to_zcash_compressed();
assert_eq!(bytes.len(), 48);
assert_eq!(G1::from_zcash(c, &bytes).unwrap(), sigma);
assert_eq!(G2::from_zcash(c, &pk.to_zcash_uncompressed()).unwrap(), pk);
assert!(G1::from_zcash(c, &[0xff; 48]).is_err());

// BN254 as Ethereum's ecPairing precompile computes it: e(P, Q) · e(−P, Q) = 1.
let bn = PairingCurve::bn254();
let mut input = bn.g1_generator().to_ethereum();
input.extend(bn.g2_generator().to_ethereum());
input.extend((-bn.g1_generator()).to_ethereum());
input.extend(bn.g2_generator().to_ethereum());
assert_eq!(ec_pairing(&input).unwrap()[31], 1);
```

## The curves

| | BN254 (alt_bn128) | BLS12-381 |
| --- | --- | --- |
| Family | Barreto–Naehrig: `p = 36x⁴ + 36x³ + 24x² + 6x + 1`, `r = 36x⁴ + 36x³ + 18x² + 6x + 1` | Barreto–Lynn–Scott of degree 12: `r = x⁴ − x² + 1`, `p = (x − 1)²r/3 + x` |
| `x` | `4965661367192848881` | `−0xd201000000010000` |
| `p` | 254 bits, `0x30644e72…3c208c16d87cfd47` | 381 bits, `0x1a0111ea…b9feffffffffaaab` |
| `r` | 254 bits, `0x30644e72…43e1f593f0000001` | 255 bits, `0x73eda753…ffffffff00000001` |
| `E` | `y² = x³ + 3` | `y² = x³ + 4` |
| Tower | `u² = −1`, `ξ = 9 + u` | `u² = −1`, `ξ = 1 + u` |
| Twist | D-type, `y² = x³ + 3/ξ`, `(x, y) ↦ (xw², yw³)` | M-type, `y² = x³ + 4ξ`, `(x, y) ↦ (x/w², y/w³)` |
| Generators | `G1`: `(1, 2)`; `G2`: Ethereum's (EIP-197) | ZCash's, those of the IETF draft |
| `G1` cofactor | 1 | `(x − 1)²/3 = 0x396c8c005555e1568c00aaab0000aaab` |
| `G2` cofactor | `2p − r`, 254 bits | `(x⁸ − 4x⁷ + 5x⁶ − 4x⁴ + 6x³ − 4x² − 4x + 13)/9`, 507 bits |
| Miller loop | over `6x + 2`, 66 signed digits | over `|x|`, 64 bits, 6 of them ones |
| Security | about 100 bits | about 120 bits |

The security levels are those of Barbulescu and Duquesne (*Updating key size estimations for pairings*, J. Cryptology 32, 2019), after the number field sieve variants of Kim and Barbulescu.

## The pairing

`e(P, Q) = f(P)^(c(p¹² − 1)/r)`, where `f` is the Miller function of the optimal ate pairing (Vercauteren, *Optimal pairings*, IEEE Trans. Inf. Theory 56, 2010): `f = f_{6x+2,Q} · ℓ_{T,ψ(Q)} · ℓ_{T+ψ(Q),−ψ²(Q)}` with `T = [6x + 2]Q` for BN, `f = f_{x,Q}` for BLS12.

- **Miller's loop** keeps `T` in homogeneous projective coordinates on the twist and evaluates each line at `P` without leaving `F_p²`. Doubling: the formulas of Costello, Lange and Naehrig (*Faster pairing computations on curves with high-degree twists*, PKC 2010) in the form of Aranha, Barreto, Longa and Ricardini (*The realm of the pairings*, SAC 2013): `A = XY/2, B = Y², C = Z², E = 3b'C, F = 3E, G = (B + F)/2, H = (Y + Z)² − B − C, I = E − B, J = X²`, then `2T = (A(B − F) : G² − 3E² : BH)` and the line `I + 3J·x_P − H·y_P`, four products and six squares of `F_p²`. Addition of the affine `Q`: the mixed addition "madd-1998-cmo" of the Explicit-Formulas Database and the line `(v·y_Q − u·x_Q) + u·x_P − v·y_P` with `u = y_Q Z − Y`, `v = x_Q Z − X`, eleven products and two squares. Untwisted, the line of a D-type twist sits at the coefficients `1, w, w³` of `F_p¹²` and is multiplied in by `Fp12::mul_by_034`; that of an M-type twist, multiplied by `w³`, at `1, w², w³` (`Fp12::mul_by_014`): 13 products of `F_p²` each. Every line is a multiple of the true one by a factor of `F_p²`, which the final exponentiation removes. Each step squares `f` (12 products of `F_p²`).
  - BN254: the non-adjacent form of `6x + 2`, 65 doublings and 21 additions (the binary form would need 36), then the lines through `ψ(Q)` and `−ψ²(Q)`.
  - BLS12-381: the binary form of `|x|`, 63 doublings and 5 additions (its non-adjacent form would add a doubling), then `f` conjugated, as `x < 0` (the inverse and the conjugate agree after the final exponentiation).
  - The points are made affine together, with one inversion of `F_p` for all of them (Montgomery's trick, the twist's `Z` through its norm). While `f` is 1, the first step neither squares it nor multiplies: `f` becomes the first line.
  - `multi_pairing` runs one loop for all pairs: one square of `f` per step, a line per pair. A pair with a point at infinity contributes 1. The loop handles the degenerate steps (a vertical line, `T` at infinity) that only points outside `G2` reach, so that it never divides by zero.
- **The final exponentiation**: the easy part `f^((p⁶ − 1)(p² + 1))` is a conjugation, one inversion of `F_p¹²` (one of `F_p`), a Frobenius map and two products; it leaves an element of the cyclotomic subgroup, where squares are Granger and Scott's, runs of nine squares or more Karabina's compressed ones (*Squaring in cyclotomic subgroups*, Math. Comp. 82, 2013: six squares of `F_p²` instead of nine, then one inversion to decompress), and inverses conjugates. The hard part raises it to `c·(p⁴ − p² + 1)/r` by powers by `x` and Frobenius maps:
  - BLS12-381: `c = 3`, as `3(p⁴ − p² + 1)/r = (x − 1)²(x + p)(x² + p² − 1) + 3` (Hayashida, Hayasaka and Teruya, *Efficient final exponentiation via cyclotomic structure for pairings over families of elliptic curves*, ePrint 2020/875): five powers by `|x|` (63 cyclotomic squares and 5 products each, over its binary form), two Frobenius maps, one cyclotomic square and seven products.
  - BN254: `c = 2x(6x² + 3x + 1)`, by the chain of Fuentes-Castañeda, Knapp and Rodríguez-Henríquez (*Faster hashing to G2*, SAC 2011): `λ₀ + λ₁p + λ₂p² + λ₃p³` with `λ₀ = 12x³ + 12x² + 6x + 1`, `λ₁ = 12x³ + 6x² + 4x`, `λ₂ = 12x³ + 6x² + 6x`, `λ₃ = 12x³ + 6x² + 4x − 1`: three powers by `x` (62 cyclotomic squares and 23 products each, over its non-adjacent form), three more squares, three Frobenius maps, ten products.

  Both factors `c` are prime to `r`, so `e` is a bilinear, non-degenerate pairing; they are the factors of arkworks, the `bls12_381` crate, blst and RELIC, so that `e` equals their values. Libraries that raise to `(p¹² − 1)/r` alone give other powers of `e`. The IETF draft's `f_{x,Q}(P)^((p¹² − 1)/r)` is `e^(1/3)` on BLS12-381. py_ecc also raises to `(p¹² − 1)/r` alone and, on BLS12-381, loops over `|x|` without conjugating: its values are `e^(1/c)` on BN254 and `e^(−1/3)` on BLS12-381.
  - To compare a value of `GT` with theirs, raise it to `c⁻¹ mod r`, `e.pow(&c.mod_inverse(r).unwrap())`, which gives the IETF draft's value and py_ecc's on BN254; for py_ecc on BLS12-381, take the inverse of that. The tests check both: on the IETF draft's `e(g₁, g₂)`, and on py_ecc's values of two seeded pairings of each curve.
  - A pairing check, that a product of pairings is 1 (as `ecPairing` and the verification of signatures and proofs use it), gives the same answer here as in every one of these libraries, since `c` is prime to `r`.
- `final_exponentiation` maps any nonzero element of `F_p¹²` into `GT`, and panics for 0. On points outside `G1` and `G2` (from `new_on_curve`) the pairing returns an element of `GT` that is not a pairing, without panicking.

## The groups

| `G1` and `G2` | |
| --- | --- |
| `G1::new(c, x, y)`, `G2::new(c, x, y)` | the point, `Err(NotOnCurve)` off the curve, `Err(NotInSubgroup)` outside the subgroup; coordinates of another field or tower panic |
| `G1::new_on_curve`, `G2::new_on_curve` | any point of the curve, as hashing to the curve needs before clearing the cofactor |
| `c.g1_generator()`, `c.g2_generator()`, `c.g1_identity()`, `c.g2_identity()` | |
| `+`, `-`, `-p`, `*` by a `Number` on either side, `+=`, `-=`, `*=`, `double()` | Jacobian coordinates, "dbl-2009-l" and "add-2007-bl" of the Explicit-Formulas Database, on the pairing's own arithmetic. Products by the endomorphisms on points known to be in the subgroup (from the generator, a checked constructor or decoding, a hash or a cleared cofactor, and their sums and multiples), the scalar reduced modulo `r`: on `G1` the GLV method with `φ(x, y) = (βx, y)` — on BLS12-381 `k = a + b·x²` with `a`, `b` below `2¹²⁸` and `[k]P = [a]P − [b]φ(P)`, on BN254 two halves from the reduced basis of the lattice of `λ`; on `G2` the GLS method with `ψ` — on BLS12-381 the four digits of `k` in base `|x|` on `Q`, `−ψ(Q)`, `ψ²(Q)`, `−ψ³(Q)`, on BN254 four signed digits of about 64 bits from an LLL-reduced basis of the lattice of the relations `Σ vᵢ(6x²)ⁱ ≡ 0 (mod r)` and Babai's rounding, on `Q`, `ψ(Q)`, `ψ²(Q)`, `ψ³(Q)` (Galbraith and Scott, Pairing 2008); the digits' products share their doublings (Straus's method, width-4 or width-5 non-adjacent forms). Other points of the curve or the twist take the whole scalar, not reduced, by the width-4 or width-5 non-adjacent form, so that it acts on points outside the subgroups too |
| `is_in_subgroup()` | `G1` of BN254: always (its `E(F_p)` has prime order). `G1` of BLS12-381: by the GLV endomorphism `φ(x, y) = (βx, y)`, `β` a cube root of unity, `φ(P) = [−x²]P` (Scott, *A note on group membership tests for G1, G2 and GT on BLS pairing-friendly curves*, ePrint 2021/1130; proved in general by El Housni, Guillevic and Piellard, ePrint 2022/352): two products by the 64-bit `x`. `G2`: `ψ(Q) = [6x²]Q` (BN254) and `ψ(Q) = [x]Q` (BLS12-381) of the same papers: a product by 128 or 64 bits, against 255 for the product by `r` |
| `clear_cofactor()` | `G1` of BLS12-381: `[1 − x]P`, RFC 9380's `h_eff` (§8.8.1), since `E(F_p) ≅ Z/(\|x − 1\|r) × Z/(\|x − 1\|/3)`; `G2` of BLS12-381: Budroni and Pintore's `[x² − x − 1]Q + [x − 1]ψ(Q) + ψ²(2Q)` (*Efficient hash maps to G2 on BLS curves*, ePrint 2017/419), the product by RFC 9380's `h_eff` (§8.8.2, appendix G.3) on every point of the twist; `G2` of BN254: Fuentes-Castañeda, Knapp and Rodríguez-Henríquez's `[x]Q + ψ([3x]Q) + ψ²([x]Q) + ψ³(Q)` (*Faster hashing to G2*, §6.1), which is the product by `x + 3xp + xp² + p³ mod r` on `G2`; `G1` of BN254: the point itself |
| `mul_by_cofactor()` | the product by the whole cofactor, which arkworks' default clearing computes (and its clearing of BN254's `G2`) |
| `G2::psi()` | `ψ`: untwist, Frobenius, twist back, `(x̄γₓ, ȳγᵧ)` with `γₓ = ξ^((p − 1)/3)`, `γᵧ = ξ^((p − 1)/2)` on a D-type twist and their inverses on an M-type one; the product by `p` on `G2` |
| `coordinates()`, `is_identity()`, `curve()`, `==`, `Hash`, `Debug`, `Display` | the hash is that of the affine coordinates, the same for equal points |

| Encodings | |
| --- | --- |
| `to_zcash_compressed()`, `to_zcash_uncompressed()` (BLS12-381) | 48 and 96 bytes for `G1`, 96 and 192 for `G2`: big-endian coordinates, `x₁ ‖ x₀` for `x = x₀ + x₁u`, with three flags in the top bits of the first byte: `0x80` compressed, `0x40` the point at infinity (all other bits zero), `0x20` `y` the larger of `±y` (for `F_p²`, by `y₁`, then by `y₀` when `y₁ = 0`) |
| `G1::from_zcash(c, &bytes)`, `G2::from_zcash(c, &bytes)` | either length, as the `bls12_381` crate reads them: the compression flag must agree with the length, the infinity flag allows no other bit, the sort flag only on a compressed finite point; coordinates below `p`; `y` from `√(x³ + b)` and the sort flag; on the curve and in the subgroup |
| `to_ethereum()`, `G1::from_ethereum`, `G2::from_ethereum` (BN254) | EIP-196 and EIP-197: 64 bytes `x ‖ y` for `G1`, 128 bytes `x₁ ‖ x₀ ‖ y₁ ‖ y₀` for `G2` (the coefficient of `u` first), zeros for the point at infinity; decoding checks the coordinates below `p`, the curve, and for `G2` the subgroup |
| `ethereum::ec_add`, `ec_mul`, `ec_pairing` | the precompiles at addresses 6, 7 and 8: inputs read as if padded with zeros to 128 and 96 bytes (and the rest ignored) for `ecAdd` and `ecMul`, a multiple of 192 bytes for `ecPairing`, whose output is the 32-byte word 1 or 0; an invalid input is an error |
| `PointError` | `Length`, `Flags`, `NotInField`, `NotOnCurve`, `NotInSubgroup`; the ZCash methods panic on BN254 (its prime leaves two spare bits, not three) and the Ethereum ones on BLS12-381 |

## Verified

- **arkworks 0.5, the `bls12_381` crate 0.8 and blst 0.3**, by `lib/tests/math/pairing/data/gen_ark.rs`: for 6 seeded pairs of 256-bit scalars `(a, b)` on each curve, `[a]g₁` and `[b]g₂` (by their encodings: ZCash's for BLS12-381, which the `bls12_381` crate decodes back to its own `[a mod r]g₁`, and Ethereum's for BN254), `e([a]g₁, [b]g₂)` and the product of the six pairings are arkworks' exactly (digests in `pairing_ref.txt`); the generator asserts that the `bls12_381` crate and blst give the same six pairings on BLS12-381 as arkworks.
- **RELIC**: `e(g₁, g₂)` on BLS12-381 is the value Diego Aranha computed with RELIC, in the `bls12_381` crate's test `test_pairing_result_against_relic`, coefficient for coefficient.
- **py_ecc 8**, by `gen_py.py`: for the first two seeded pairs of each curve, py_ecc's pairing raised to `−3` (BLS12-381: py_ecc raises to `(p¹² − 1)/r` alone and loops over `|x|` without conjugating) and to `2x(6x² + 3x + 1)` (BN254) is this pairing exactly; and the recipe of the section above, `e.pow(&c.mod_inverse(r).unwrap())`, inverted on BLS12-381, gives py_ecc's own values exactly (`the_documented_recipe_gives_py_eccs_values`).
- **The IETF draft** on pairing-friendly curves (draft-irtf-cfrg-pairing-friendly-curves-11, appendix B): its `e(BP, BP')` on BLS12-381, whose `e₀, …, e₁₁` are the tower's coefficients of `f^((p¹² − 1)/r)`, cubed, is `e(g₁, g₂)`, and `e(g₁, g₂)^(1/3 mod r)` is the draft's value. Its generators are the curve's.
- **The exponents**: the identities `3(p⁴ − p² + 1)/r = (x − 1)²(x + p)(x² + p² − 1) + 3` and `2x(6x² + 3x + 1)(p⁴ − p² + 1)/r = λ₀ + λ₁p + λ₂p² + λ₃p³` hold exactly (in `Number`s), and both factors are prime to `r`.
- **Ethereum**: go-ethereum's tests of the precompiles (`core/vm/testdata/precompiles`, the vectors of EIP-196 and EIP-197 by chfast, cdetrio, jeff and others, in `eth_vectors.txt`): 16 of `ecAdd`, 19 of `ecMul` and 14 of `ecPairing`, up to ten pairs, all give the expected output. The failures: a coordinate equal to `p`, a point off the curve, a point of the twist outside `G2`, lengths not a multiple of 192; padded and over-long inputs; scalars `r` and `r + 1`.
- **ZCash encodings**: the encodings of `[i]g` for `i = 0, …, 999` in the four forms are byte for byte the files of the `bls12_381` crate's tests (their digests and lengths: 48 000, 96 000, 96 000 and 192 000 bytes); every 50th is decoded back by default, all 4 000 in an ignored test (`zcash_encodings_of_the_first_thousand_multiples_all_decode`). 123 encodings of three seeded points of each group, corrupted (each flag flipped, each coordinate plus `p`, the last bit flipped) and of the point at infinity with every combination of flags and a stray bit: the 22 the `bls12_381` crate accepts decode to its points, and the 101 it refuses are refused (`zcash_cases.txt`). The errors are those named for the length, the flags, `x = p`, `x = 0` (the point `(0, 2)` of order 3), `x = 1` (no square root) and points outside the subgroups.
- **Subgroups**: on 8 seeded points of each curve and its twist (outside the subgroups), `is_in_subgroup` is arkworks' answer and that of the product by `r`; `clear_cofactor` is arkworks' (BLS12-381, both groups) and lands in the subgroup; `mul_by_cofactor` is arkworks' (BLS12-381, both groups; BN254, `G2`). For every prime `ℓ` of the cofactors (BN254 `G2`: 10069, 5864401, 1875725156269 and a 178-bit prime; BLS12-381 `G1`: 3, 11, 10177, 859267, 52437899; `G2`: 13, 23, 2713, 11953, 262069 and a 448-bit prime, as PARI factors them), a point of order `ℓ`, alone and added to the generator, fails the test, and its cofactor is cleared to the point at infinity. That no point outside the subgroups passes is also shown, not only sampled: `ψ² − [t]ψ + [p] = 0` on the twist (checked on its points), so `ψ − [k]` is invertible on the `ℓ`-torsion unless `k² − tk + p ≡ 0 (mod ℓ)`, and `φ² + φ + 1 = 0` on `E(F_p)`, so `φ + [x²]` is invertible unless `x⁴ − x² + 1 ≡ 0 (mod ℓ)`; none of the fifteen primes is such an exception, while `k ≡ p` and `x⁴ − x² + 1 ≡ 0` modulo `r`.
- **`ψ` and the clearings**: `ψ` is the product by `p` on `G2`, `ψ¹²` the identity and no smaller power; Budroni and Pintore's map is the product by RFC 9380's `h_eff` on points outside `G2` as on `G2`, and `[1 − x]` RFC 9380's `h_eff` of `G1`; BN254's map is the product by `x + 3xp + xp² + p³ mod r` on `G2`.
- **The pairing's properties**: bilinearity in each argument and `e(aP, bQ) = e(P, Q)^(ab)` for 256-bit `a`, `b`; `e(g₁, g₂)` of order exactly `r`; `e(−P, Q) = e(P, −Q) = e(P, Q)⁻¹`; 1 for a point at infinity and for `[r]P`; `multi_pairing` the product of the pairings, with points at infinity among the pairs; `miller_loop` then `final_exponentiation` the pairing; elements of `F_p⁶` mapped to 1; values for points outside the subgroups, of small order included, without a panic.
- **The groups**: the group law (identity, inverses, commutativity, associativity, distributivity), products by every scalar from −19 to 19, by `r − 1`, `r`, `r + 1` and a 2560-bit scalar; the parameters (`p`, `r`, `x`, `b`, `b'` as arkworks has it, the cofactors, `#E(F_p) = p + 1 − t` with `t = 6x² + 1` or `x + 1`, and random points of the curve and of the twist killed by the orders of their groups).

To run the tests: `cargo test --test math -- pairing:: field::tower`, and `cargo test --test math -- --ignored all_decode` for the 4 000 decodings. To regenerate the data: `gen_ark.rs` is the `main.rs` of a probe crate with the dependencies named at its top; `gen_py.py` needs py_ecc, `gp`, and the directory of the `bls12_381` crate's test vectors.

## Limits

- Not constant-time, and no protocol layer: BLS signatures, their aggregation and proof-of-possession, and key generation are not here.
- Hashing to `G1` and `G2` is in [its own section](#hashing-to-g1-and-g2), with its own limits; BN254's `G2` has no hashing.
- Only the two curves; there is no constructor for other BN or BLS12 curves.
- The pairing is arkworks' `f^(c(p¹² − 1)/r)`. The IETF draft's values are its powers by `1/3` on BLS12-381; py_ecc's are its powers by `1/c` on BN254 and by `−1/3` on BLS12-381. Pairing checks agree with all of them.
- The clearing of BN254's `G2` is Fuentes-Castañeda et al.'s map, not arkworks' product by the cofactor (`mul_by_cofactor` gives that); on `G2` the two differ by a factor.
- No compressed encoding of BN254 (Ethereum has none), no EIP-2537 encoding of BLS12-381, no encoding of `GT`.
- There are no precomputed lines for a fixed `G2` point.

## Hashing to G1 and G2

`a::math::pairing::hash` hashes byte strings to BLS12-381's groups as RFC 9380 specifies (section 8.8), and to BN254's `G1` in the same manner: a message and a domain separation tag (DST) to elements of `F_p` or `F_p²`, those to points of the curve or of its twist, and the cofactor cleared, so that the point is in `G1` or `G2`. BLS signatures hash their messages so.

| Function | Suites | Field, hash, `k`, `L` | Map | Cofactor |
| --- | --- | --- | --- | --- |
| `hash::bls12_381_g1()` | `BLS12381G1_XMD:SHA-256_SSWU_RO_`, `BLS12381G1_XMD:SHA-256_SSWU_NU_` (RFC 9380, section 8.8.1) | `F_p`, SHA-256, 128, 64 | simplified SWU to `E′: y² = x³ + A′·x + B′`, `Z = 11`, then the 11-isogeny of appendix E.2 | `h_eff = 0xd201000000010001 = 1 − x`, by `G1::clear_cofactor` |
| `hash::bls12_381_g2()` | `BLS12381G2_XMD:SHA-256_SSWU_RO_`, `BLS12381G2_XMD:SHA-256_SSWU_NU_` (section 8.8.2) | `F_p²`, SHA-256, 128, 64 | simplified SWU over `F_p²` to `E′: y² = x³ + 240I·x + 1012(1 + I)`, `Z = −(2 + I)`, then the 3-isogeny of appendix E.3 | the `h_eff` of 636 bits, by Budroni and Pintore's method of appendix G.3, `G2::clear_cofactor` |
| `hash::bn254_g1()` | `BN254G1_XMD:SHA-256_SVDW_RO_`, `BN254G1_XMD:SHA-256_SVDW_NU_` (gnark-crypto's; RFC 9380 has no BN254 suite) | `F_p`, SHA-256, 128, 48 | Shallue–van de Woestijne (section 6.6.1), `Z = 1`, to `y² = x³ + 3` itself | none: `E(F_p)` is `G1` |

The suites are `&'static G1Suite` and `&'static G2Suite`, built on the first call, whose points are those of `PairingCurve::bls12_381()` or `PairingCurve::bn254()`:

| Item | What it is |
| --- | --- |
| `suite.hash_to_curve(&msg, &dst)` | `hash_to_curve`, the `_RO_` suite: two elements, two maps, their sum, the cofactor cleared |
| `suite.encode_to_curve(&msg, &dst)` | `encode_to_curve`, the `_NU_` suite: one element, one map, the cofactor cleared; not uniform |
| `suite.hash_to_field(&msg, &dst, count)`, `suite.map_to_curve(&u)`, `suite.clear_cofactor(&p)` | The steps; `map_to_curve` gives a point of the curve (`G1::new_on_curve`, `G2::new_on_curve`) before the cofactor is cleared |
| `ro_id()`, `nu_id()`, `curve()`, `field()` or `tower()`, `k()`, `l()`, `h_eff()`, `z()`, `sswu()`, `isogeny()`, `svdw()` | The IDs and parameters; on `G1Suite` the map's parts are `Option`s, as BLS12-381 and BN254 take different maps |
| `hash_to_field_fp2::<H>(tower, &msg, &dst, count, k)` | RFC 9380's `hash_to_field` for `m = 2`, into any tower's `F_p²`: each element `e₀ + e₁·u` from two pieces of `L` bytes |
| `SimplifiedSwuFp2::new(&a, &b, &z)`, `.map(&u)`, `SimplifiedSwuFp2::find_z(&a, &b)` | The simplified SWU map of section 6.6.2 over a tower's `F_p²`, checked against the four criteria on `Z`, and appendix H.2's search, which counts from the generator `I` |
| `IsogenyMapFp2::new(&x_num, &x_den, &y_num, &y_den)`, `.map(&x, &y)`, `.degree()`, `.x_num()`, … | A rational map of appendix E's shape over `F_p²`, the point at infinity where a denominator vanishes |

The map over `F_p` and its isogeny are those of [elliptic curves](ec.md#hashing-to-curves) (`SimplifiedSwu`, `IsogenyMap`, `ShallueVanDeWoestijne`), on the curve's `F_p`. Elements of another field or tower, points of another curve and an empty tag panic.

### Example

```rust
use a::math::pairing::{hash, pairing};

// RFC 9380, appendix J.9.1: BLS12381G1_XMD:SHA-256_SSWU_RO_ of "abc".
let g1 = hash::bls12_381_g1();
let p = g1.hash_to_curve(b"abc", b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_");
let (x, _) = p.coordinates().unwrap();
assert_eq!(
    x.value().to_str_radix(16),
    "3567bc5ef9c690c2ab2ecdf6a96ef1c139cc0b2f284dca0a9a7943388a49a3aee664ba5379a7655d3c68900be2f6903"
);

// Hashing to G2, as BLS signatures with keys in G1 do, and the steps it is made of.
let g2 = hash::bls12_381_g2();
let dst = b"MY-APP-V01-CS01-with-BLS12381G2_XMD:SHA-256_SSWU_RO_";
let q = g2.hash_to_curve(b"a message", dst);
assert!(q.is_in_subgroup());
let u = g2.hash_to_field(b"a message", dst, 2);
assert_eq!(g2.clear_cofactor(&(g2.map_to_curve(&u[0]) + g2.map_to_curve(&u[1]))), q);
let c = g2.curve();
assert_eq!(pairing(&c.g1_generator().double(), &q), pairing(&c.g1_generator(), &q.double()));

// BN254's G1, in gnark-crypto's suite.
let b = hash::bn254_g1();
assert_eq!(b.ro_id(), "BN254G1_XMD:SHA-256_SVDW_RO_");
assert!(b.encode_to_curve(b"abc", b"MY-DST").is_in_subgroup());
```

### Method

- **`hash_to_field` with `m = 2`** (section 5.2) asks `expand_message_xmd` for `2 · count · L` bytes; the element `uᵢ = e₀ + e₁·u` takes the pieces at `L·2i` and `L·(2i + 1)`, each read in big-endian order and reduced modulo `p`. The tower's `u`, with `u² = −1` on BLS12-381 (and BN254), is the RFC's `I`. `G1`'s elements are `hash_to_field` into `F_p` of [elliptic curves](ec.md#hashing-to-curves).
- **The simplified SWU map over `F_p²`** is that of section 6.6.2, on the tower's elements: `x₁ = (−B/A)·(1 + 1/(Z²u⁴ + Z·u²))`, or `B/(Z·A)` at `u = 0` (the only exceptional input: `−1` is a square of `F_p²`, as every element of `F_p` is, so `−1/Z` is not), `x = x₁` if `g(x₁)` is a square of `F_p²` and `x₂ = Z·u²·x₁` otherwise, and `y = √g(x)` with `sgn0(y) = sgn0(u)`, the `sgn0` of section 4.1 for `m = 2` (the parity of `c₀`, or of `c₁` when `c₀ = 0`). Squares and their roots are those of `Fp2` (the norm's Legendre symbol; the complex method). `new` checks the four criteria on `Z`; that `g(x) − Z` is irreducible over `F_p²` is that it has no root there, `gcd(x^(p²) − x, g − Z) = 1`, with `x^(p²)` computed modulo the cubic by squarings and the greatest common divisor by Euclid's algorithm over `F_p²`. `find_z` tries `I, −I, 1 + I, −(1 + I), 2 + I, …`, as the RFC's script counts from `F.gen()`, and finds `−(2 + I)` for BLS12-381's `E′`.
- **The isogenies** are evaluated by Horner's rule with one inversion for both denominators, and give the point at infinity where one of them is 0, as section 6.6.3 requires. On `G1` that case is reached: `x_den` of the 11-isogeny is the square of a polynomial of degree 5 with five roots in `F_p`, `g′` is a square at each, so the kernel's points of order 11 are on `E′(F_p)`, and the simplified SWU map reaches those over two of the roots from 16 elements `u`, which go to the identity. On `G2` it is not: `x_den = (x − x₀)²` with `g′(x₀)` not a square of `F_p²`.
- **Clearing the cofactor.** `G1`: `[1 − x]·P`, RFC 9380's `h_eff` (`G1::clear_cofactor`). `G2`: the steps of appendix G.3, `t₁ = x·Q`, `t₂ = ψ(Q)`, `t₃ = ψ²(2Q) − t₂`, `t₂ = x·(t₁ + t₂)`, `t₃ + t₂ − t₁ − Q`, which is `[x² − x − 1]Q + [x − 1]ψ(Q) + ψ²(2Q)` (Budroni and Pintore, *Efficient hash maps to G2 on BLS curves*, ePrint 2017/419) and the product by `h_eff` on every point of the twist (`G2::clear_cofactor`, with the pairing's `ψ`). BN254's `G1` has cofactor 1.
- **BN254's `G1`** takes the Shallue–van de Woestijne map of section 6.6.1 (`ShallueVanDeWoestijne` of [elliptic curves](ec.md#hashing-to-curves)) to `y² = x³ + 3`, whose `A = 0` the simplified SWU map cannot take, with the `Z = 1` that appendix H.1's search gives, `L = 48` and SHA-256, as gnark-crypto defines `BN254G1_XMD:SHA-256_SVDW_RO_` and `_NU_`. RFC 9380 defines no suite for BN254; these follow its naming (section 8.10) and its construction.
- **Cost.** A map to `G2` takes an inversion and one or two square roots of `F_p²` (each one to three exponentiations of `F_p` and, for a square, an inversion; see below), the 3-isogeny an inversion and four polynomials of degree 2 and 3, and the clearing two products by the 64-bit `x`, three maps `ψ` and a few additions; a map to `G1` an inversion and one or two square roots of `F_p`, and the 11-isogeny four polynomials of degree 10 to 15, and the clearing a product by the 64-bit `1 − x`. The arithmetic is the pairing's own (fixed limbs, inversion by Lehmer's algorithm on them); a square root of `F_p²` by the complex method takes exponentiations of `F_p` of 381 bits: for `a₁ ≠ 0`, one for `n = √(a₀² + a₁²)`, which ends it for a non-square (whose norm is no square), and for a square one more for `√((a₀ + n)/2)`, a third for `√((a₀ − n)/2)` when `(a₀ + n)/2` is not a square (exactly one of the two is, their product `−a₁²/4` being no square; so for about half of the squares), and an inversion for `x₁ = a₁/(2x₀)`; for `a₁ = 0`, one for `√a₀` and a second for `√(−a₀)` when `a₀` is not a square. A map to `G2` so takes two or three exponentiations when `g(x₁)` is a square, and three or four when it is not (one for `g(x₁)`, two or three for `g(x₂)`). A straight-line `sqrt_ratio` of `F_p²` (appendix F.2.1.1 or the `q ≡ 9 (mod 16)` root of appendix I.3) would be one exponentiation of 758 bits in `F_p²`, as many products of `F_p` as about four of `F_p`.

### Verified

- **RFC 9380, appendix J.9 and J.10**: all 20 vectors of the four suites (5 messages each). For the `_RO_` suites `u₀`, `u₁` (in `F_p²` on `G2`), `Q₀`, `Q₁` and `P`; for the `_NU_` suites `u₀`, `Q` and `P`; `P` equal to the cofactor cleared from `Q₀ + Q₁` (or `Q`), in the subgroup by the endomorphism test and by the product with `r`. All equal. The vectors are in `lib/tests/math/pairing/data/h2c_bls12381_vectors.txt`, extracted from the RFC's text by `gen_h2c_bls12381.py` beside them.
- **The parameters** of section 8.8: the IDs, `k`, `L`, `Z`, `A′` and `B′`, the degrees 11 and 3, `h_eff` of `G1` equal to `1 − x` and to `0xd201000000010001`, and `G2`'s `h_eff` as the RFC prints it. `SimplifiedSwu::find_z` gives 11 on `G1`'s `E′`, `SimplifiedSwuFp2::find_z` gives `−(2 + I)` on `G2`'s, and `ShallueVanDeWoestijne::find_z` gives 1 on BN254.
- **arkworks 0.5 and the `bls12_381` crate 0.8** (`experimental`), in a probe outside the crate (crates from the cargo cache, offline; SHA-256 by RustCrypto's `sha2` 0.10, behind digest 0.9's traits for `bls12_381`): 5,000 random messages of up to 299 bytes with random tags of 1 to 80, 255, 256 and 300 bytes. `hash_to_field` gives arkworks' `DefaultFieldHasher`'s elements (20,000, in `F_p` and `F_p²`); `hash_to_curve` gives arkworks' points on `G1` and `G2` (10,000); `hash_to_curve` and `encode_to_curve` give the `bls12_381` crate's (20,000); `map_to_curve` gives arkworks' `WBMap` on 5,003 chosen elements of `F_p` and of `F_p²` each (0, ±1 and random), compared as ZCash encodings. All equal. On the 16 elements that `G1`'s map takes into the isogeny's kernel this map gives the identity, as section 6.6.3 requires; arkworks' `IsogenyMap::apply` inverts the zero denominators as zero and gives the pair `(0, 0)`, which is not a point of the curve: arkworks departs from the RFC there.
- **The isogenies are isogenies**: on 5 random points of each `E′` (over `F_p` as an `a::math::ec` curve; over `F_p²` by affine formulas written in the test), the image of every sum of two is the sum of the images, of a double the double, of a negative the negative, of a multiple by a random 128-bit scalar the multiple (`G1`), each image on the curve or the twist. `x_den` of the 11-isogeny is a square and `y_den` a cube (factorised by `ModPolynomial::factor`), and each of the five roots of `x_den` goes to the point at infinity; the 16 elements that reach the kernel are found in the test by solving the map's two quadratics for `u`, and give the identity through `map_to_curve`, which the cofactor's clearing of a sum then ignores.
- **The simplified SWU map over `F_p²`**: on 40 random elements of BLS12-381's `F_p²`, `u = 0` among them, a point of `E′` with `x` as the definition chooses (computed again in the test), `sgn0(y) = sgn0(u)` and `−u` mapped to `(x, −y)`, and `u = 0` to `B/(Z·A)` and on through the isogeny. On the tower `F_103² = F_103[u]/(u² + 1)` in full: for two random curves, each candidate `Z` of appendix H.2's order is accepted or refused by `new` as the four criteria decide by brute force (every square of the 10,609 elements listed, every root of `g − Z` sought), `find_z` gives the first accepted, and every one of the 10,609 elements maps to a point of the curve with the sign of `u`.
- **The cofactors**: on random points of `E(F_p)` and of the twist outside the subgroups, `G1`'s clearing is the product by `h_eff`; `G2`'s is the product by the RFC's `h_eff` and equals appendix G.3 computed step by step in the test with `ψ` from its constants `1/(1 + I)^((p − 1)/3)` and `1/(1 + I)^((p − 1)/2)` and `ψ²` from `1/2^((p − 1)/3)`, which equal the pairing's `ψ` and `ψ∘ψ`; each result is in the subgroup.
- **BN254's `G1`**: gnark-crypto's 10 published vectors (`ecc/bn254/hash_vectors_test.go`, 5 for `HashToG1` and 5 for `EncodeToG1`: `u`, `Q₀`, `Q₁` or `Q`, and `P`), all equal; the data are in `h2c_bn254_vectors.txt`, extracted by `gen_h2c_bn254.py`. gnark-crypto is the only implementation of this suite available here.
- **Random messages**: on each suite, points in the group, not the identity, the same for the same input, different for another tag, and `encode_to_curve` apart from `hash_to_curve`.
- **The panics**: a square `Z`, `Z = −1`, `A = 0`, `B = 0` and a singular curve for the map over `F_p²`; an isogeny map without coefficients or with a leading 0; an element of BN254's `F_p` or `F_p²` given to BLS12-381's maps; a point of BN254 given to BLS12-381's clearings; an empty tag; 64 elements of `F_p²` and 128 of `F_p` (beyond 255 blocks of SHA-256) and `usize::MAX`.
- On x86_64 (Ubuntu 26.04, Rust 1.100, in the CI container) the same 27 tests pass.

To run the tests: `cargo test --test math -- pairing::hash`. The data files are written again by `python3 gen_h2c_bls12381.py rfc9380.txt > h2c_bls12381_vectors.txt` and `python3 gen_h2c_bn254.py hash_vectors_test.go > h2c_bn254_vectors.txt`.

### Limits

- **Not constant-time**: the maps branch on whether an element is a square. For public messages, as in verifying signatures, not for hashing secrets.
- `G1`'s simplified SWU map on BLS12-381 is the straight-line form of appendix F.2 (with `sqrt_ratio` for `p ≡ 3 (mod 4)`), its point a fraction that the isogeny takes homogeneously, and `hash_to_curve` applies the isogeny once to the sum of the two points; the map over `F_p²` and BN254's Shallue–van de Woestijne map are the plain forms of section 6, which cost less here than the straight-line ones (see the cost above).
- BN254's `G1` follows gnark-crypto, the one reference: it is not an RFC suite, and other hashings to BN254 (the "try and increment" of some Ethereum contracts, or other choices of `Z`) give other points. BN254's `G2` has no hashing here.
- The suites are made for the two curves only; `hash_to_field_fp2`, `SimplifiedSwuFp2` and `IsogenyMapFp2` are public for other uses, but there is no constructor of a suite.

## Measured

Measured on an Apple M4 Max on 2026-11-26, between 06:22 and 06:23, with nothing else running on the machine, by the harness in the repository's `bench/curves` (its method in `bench/README.md` there). This crate is built as a user gets it, in the default release profile (opt-level 3, no LTO, no `target-cpu`) of rustc 1.100.0, and so are the Rust references in the same probe crate: arkworks 0.5 (`ark-bn254`, `ark-bls12-381`), blst 0.3.17 (its own assembly), the `bls12_381` crate 0.8 and substrate-bn 0.6; mcl 4.44.0 is herumi/mcl at commit `a1c4e29e` in its 384-bit build, for both curves. One thread. Every output is compared with this crate's before the clock starts (Miller loops through the pairing, as their outputs differ by factors the final exponentiation removes); then the sides alternate sample by sample in one process, on inputs cycled from pools of 16 to 64 seeded values, and a figure is the median of 21 samples of at least 1 ms each. A cell gives this crate's time and, in parentheses, that time over the time of the fastest reference that does the same work, with its name (`ark` arkworks, `blst`, `bls12_381`, `mcl`, `substrate-bn`): below 1, this crate is faster. Not counted, as they promise more: blst's `blst_p1_mult` and `blst_p2_mult` and the `bls12_381` crate's products and powers, which are constant-time (blst's variable-time `blst_p1_unchecked_mult` and `blst_p2_unchecked_mult` are counted).

### The pairing and the groups

Products by scalars below `r`, on points from affine inputs; sums and doublings of projective points; the subgroup tests of points of the groups, the clearings of points outside them; `Gt::pow` by an exponent below `r`:

| Operation | BLS12-381 | BN254 |
| --- | ---: | ---: |
| `pairing` | 306 µs (1.09× mcl) | 216 µs (1.14× mcl) |
| `miller_loop` | 146 µs (1.19× mcl) | 101 µs (1.19× mcl) |
| `final_exponentiation` | 161 µs (1.01× mcl) | 113 µs (1.09× mcl) |
| `multi_pairing`, 2 pairs | 403 µs (1.10× mcl) | 288 µs (1.14× mcl) |
| `multi_pairing`, 4 pairs | 586 µs (1.10× mcl) | 430 µs (1.15× mcl) |
| `multi_pairing`, 8 pairs | 955 µs (1.10× mcl) | 708 µs (1.15× mcl) |
| `multi_pairing`, 16 pairs | 1.69 ms (1.10× mcl) | 1.27 ms (1.16× mcl) |
| `G1 * k`, `k < r` | 35.3 µs (1.08× mcl) | 20.6 µs (1.06× mcl) |
| `G2 * k`, `k < r` | 71.7 µs (1.28× mcl) | 41.9 µs (1.34× mcl) |
| `G1 + G1` | 293 ns (1.02× mcl) | 175 ns (1.06× mcl) |
| `G1::double` | 139 ns (0.905× blst) | 78.5 ns (0.896× ark) |
| `G2 + G2` | 850 ns (1.10× mcl) | 475 ns (1.18× mcl) |
| `G2::double` | 352 ns (1.07× mcl) | 202 ns (1.09× mcl) |
| `G1::coordinates` | 964 ns (0.731× mcl) | 578 ns (0.710× mcl) |
| `G2::coordinates` | 1.17 µs (0.772× mcl) | 717 ns (0.757× mcl) |
| `G1::is_in_subgroup` | 20.6 µs (0.898× blst) | 2.89 ns (3.92× ark) |
| `G2::is_in_subgroup` | 26.0 µs (1.03× mcl) | 41.6 µs (0.665× ark) |
| `G1::clear_cofactor` | 10.5 µs (0.705× ark) | 9.93 ns |
| `G2::clear_cofactor` | 55.1 µs (0.678× ark) | 25.9 µs |
| `G1::mul_by_cofactor` | 25.9 µs (0.727× ark) |  |
| `G2::mul_by_cofactor` | 235 µs (0.488× ark) | 67.6 µs (0.511× ark) |
| `G2::psi` | 120 ns | 68.3 ns |
| `G1::new`, checked | 20.9 µs (0.822× ark) |  |
| `G2::new`, checked | 26.3 µs (0.786× ark) |  |
| `Gt::pow`, `e < r` | 110 µs (1.13× mcl) | 68.0 µs (1.13× mcl) |
| `Gt * Gt` | 1.24 µs (1.31× mcl) | 742 ns (1.30× mcl) |
| `Gt::square` | 464 ns (1.02× blst) | 308 ns (0.885× ark) |
| `Gt::inverse` | 15.6 ns (0.795× mcl) | 12.4 ns (0.701× mcl) |
| `Gt::from_fp12` | 31.7 µs (0.891× blst) | 64.3 µs (0.485× mcl) |

Against blst the pairing of BLS12-381 takes 0.972 of its time, the Miller loop 1.10 and the multi-pairings 1.04 to 1.09; against arkworks, the pairings and multi-pairings take 0.510 to 0.540 of its time on both curves, and against substrate-bn 0.231 to 0.263. Affine coordinates take 0.710 to 0.772 of mcl's time, the clearings and the products by the cofactors 0.488 to 0.727 of arkworks's.

With precomputed lines for a fixed `Q`, which this crate does not keep (the limits above), blst's Miller loop takes 96.1 µs on BLS12-381, where this crate's plain loop takes 147 µs; arkworks's and the `bls12_381` crate's loops on prepared points take 185 and 167 µs there, and arkworks's 137 µs on BN254, against 101 µs. This is other work, not counted.

### Encodings and the precompiles

ZCash's encodings from projective points and their checked decodings (BLS12-381), Ethereum's (BN254), and its precompiles on bytes, against the steps the references' users take (the decodings with their curve and subgroup checks):

| Operation | Time |
| --- | ---: |
| `to_zcash_compressed`, `G1` | 1.00 µs (0.722× mcl) |
| `to_zcash_uncompressed`, `G1` | 1.05 µs (0.626× blst) |
| `to_zcash_compressed`, `G2` | 1.25 µs (0.786× mcl) |
| `to_zcash_uncompressed`, `G2` | 1.36 µs (0.704× blst) |
| `G1::from_zcash`, compressed | 29.6 µs (0.989× blst) |
| `G1::from_zcash`, uncompressed | 20.3 µs (0.896× blst) |
| `G2::from_zcash`, compressed | 51.5 µs (1.26× blst) |
| `G2::from_zcash`, uncompressed | 26.5 µs (1.01× blst) |
| `to_ethereum`, `G1` | 650 ns (0.464× ark) |
| `to_ethereum`, `G2` | 844 ns (0.523× ark) |
| `G1::from_ethereum` | 60.0 ns (1.04× ark) |
| `G2::from_ethereum` | 41.7 µs (0.618× ark) |
| `ethereum::ec_add` | 984 ns (0.595× ark) |
| `ethereum::ec_mul` | 21.2 µs (0.449× ark) |
| `ethereum::ec_pairing`, 1 pair | 260 µs (0.510× ark) |
| `ethereum::ec_pairing`, 2 pairs | 367 µs (0.522× ark) |
| `ethereum::ec_pairing`, 4 pairs | 597 µs (0.535× ark) |

### Hashing to G1 and G2

On a message of 32 bytes and a tag of 50 bytes (47 for BN254), against blst (`blst_hash_to_g1`, `blst_encode_to_g1`, `blst_map_to_g1` and their `G2` forms), arkworks's `MapToCurveBasedHasher` with its `WBMap`, the `bls12_381` crate's `experimental` hashing and mcl's `hashAndMapToWithDst`, the fastest of them in each cell; arkworks's and the `bls12_381` crate's SHA-256 are portable code. BN254's suite has no reference here (gnark-crypto's is in Go):

| Suite | `hash_to_curve` | `encode_to_curve` | `hash_to_field`, 2 elements | `map_to_curve` | `map_to_curve` and `clear_cofactor` |
| --- | ---: | ---: | ---: | ---: | ---: |
| BLS12-381, `G1` | 32.9 µs (1.09× blst) | 22.2 µs (1.05× blst) | 414 ns (0.289× bls12_381) | 11.6 µs (0.677× bls12_381) | 22.1 µs (1.04× blst) |
| BLS12-381, `G2` | 119 µs (1.33× blst) | 90.1 µs (1.23× blst) | 784 ns (0.312× bls12_381) | 32.3 µs (0.487× bls12_381) | 92.0 µs (1.26× blst) |
| BN254, `G1` | 15.6 µs | 6.36 µs | 312 ns | 7.66 µs |  |

The clearing of `G2`'s cofactor, alone, takes 55.3 µs of the 119 µs of `hash_to_curve` on `G2`.

### Where the references are faster

Every case above 1.05 times the fastest reference that does the same work, by cause:

- **The pairing and its parts, against mcl** (`pairing` 1.09 and 1.14, `miller_loop` 1.19 on both curves, `final_exponentiation` 1.09 on BN254, `multi_pairing` 1.10 and 1.14 to 1.16; `Gt * Gt` 1.31 and 1.30, `Gt::pow` 1.13): the operation counts of every phase are those of the literature, as in mcl; what is left is moving the elements of `F_p¹²`, 576 bytes each, and the additions between the products, and the spills around the assembly kernels of `F_p` and `F_p²`, which take their operands through memory, where mcl's code is assembly generated from its LLVM IR.
- **Sums, doublings and products by scalars on the curves, against mcl** (`G2 + G2` 1.10 and 1.18, `G2::double` 1.07 and 1.09, `G1 + G1` 1.06 on BN254; `G1 * k` 1.08 and 1.06, `G2 * k` 1.28 and 1.34): the same data movement and spills in the formulas, with endomorphisms, as mcl's (GLV on `G1`, GLS on `G2`); on `G2` the rest of the gap is not established.
- **BN254's `G1::is_in_subgroup`** (3.92 times arkworks's: 2.89 ns against 0.74 ns): the test is always true (BN254's `E(F_p)` has prime order), but the point is first converted to the pairing's own limbs.
- **`G2::from_zcash` of the compressed form** (1.26 times blst's): the square root in `F_p²`, by the complex method (two or three exponentiations of `F_p`; see the cost under [Hashing to G1 and G2](#hashing-to-g1-and-g2)), where the uncompressed decoding takes 1.01 of blst's time.
- **Hashing to BLS12-381's `G2`, against blst** (`hash_to_curve` 1.33, `encode_to_curve` 1.23, `map_to_curve` with the clearing 1.26): the clearing of the cofactor, 55.3 µs of the 119 µs of `hash_to_curve`, is two products by the 64-bit `x` on the twist at the cost of its sums and doublings (above), and the map's square roots of `F_p²` take two to four exponentiations of `F_p`; blst's arithmetic is hand-written assembly.
- **`hash_to_curve` on BLS12-381's `G1`** (1.09 times blst's): its two maps, at 1.04 of blst's time for a map and the clearing, on blst's hand-written assembly; the hashing to the field takes 414 ns of the 32.9 µs.
