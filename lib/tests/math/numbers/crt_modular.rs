//! Tests of the Chinese remainder theorem's paths (in registers, on limbs, two congruences on the
//! stack, the trees of many words) and of the orders and primitive roots of words: against plain
//! joins of numbers, at the lengths where the paths change.

use a::math::numbers::{ByteOrder, Number};

fn n(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::BigEndian)
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A number of `limbs` limbs, its top bit set, in the byte order `order`.
    fn limbs(&mut self, limbs: usize, order: ByteOrder) -> Number {
        let mut v: Vec<u8> = (0..limbs).flat_map(|_| self.next().to_le_bytes()).collect();
        *v.last_mut().unwrap() |= 0x80;
        if order == ByteOrder::BigEndian {
            v.reverse();
        }
        Number::new(v, order)
    }
}

/// The same number in the other byte order, with zero bytes above its top.
fn other(x: &Number) -> Number {
    let mut v = x.value.clone();
    v.reverse();
    let order = match x.byte_order {
        ByteOrder::BigEndian => ByteOrder::LittleEndian,
        ByteOrder::LittleEndian => ByteOrder::BigEndian,
    };
    match order {
        ByteOrder::BigEndian => v.insert(0, 0),
        ByteOrder::LittleEndian => v.push(0),
    }
    Number {
        value: v,
        negative: x.negative,
        byte_order: order,
    }
}

/// The congruences joined one at a time as numbers, by the gcd and the inverse: the tests' own.
fn plain_crt(c: &[(Number, Number)]) -> Option<(Number, Number)> {
    let (mut x, mut l) = (n(0), n(1));
    for (r, m) in c {
        let m = m.clone();
        let g = l.gcd(&m);
        let d = (r - &x).rem_euclid(&m);
        if !d.rem_euclid(&g).is_zero() {
            return None;
        }
        let mg = &m / &g;
        let t = if mg == 1u8 {
            n(0)
        } else {
            let inv = (&l / &g).rem_euclid(&mg).mod_inverse(&mg).unwrap();
            ((&d / &g) * inv).rem_euclid(&mg)
        };
        x = &x + &(&l * &t);
        l = &l * &mg;
        x = x.rem_euclid(&l);
    }
    Some((x, l))
}

fn check(c: &[(Number, Number)], what: &str) {
    let got = Number::crt(c);
    let want = plain_crt(c);
    assert_eq!(got, want, "{what}");
    if let (Some((x, l)), Some(first)) = (&got, c.first()) {
        assert_eq!(x.byte_order, first.0.byte_order, "{what}");
        assert_eq!(l.byte_order, first.0.byte_order, "{what}");
    }
}

/// The primes from `from` on, by trial division, `count` of them.
fn primes_from(from: u64, count: usize) -> Vec<u64> {
    (from..)
        .filter(|&q| q > 1 && (2..q).take_while(|d| d * d <= q).all(|d| q % d != 0))
        .take(count)
        .collect()
}

#[test]
fn crt_of_words_in_registers_and_on_limbs() {
    // One to eight words, whose solution passes two words from the third on, residues of any sign
    // and length, moduli of 1 and sharing factors.
    let mut r = Rng(0xc47_0001);
    let rounds = if cfg!(miri) { 4 } else { 200 };
    for round in 0..rounds {
        let k = 1 + round % 8;
        let shared = 1 + r.next() % 6;
        let x = r.limbs(1 + round % 9, ByteOrder::LittleEndian);
        let c: Vec<(Number, Number)> = (0..k)
            .map(|i| {
                let m = match (round + i) % 7 {
                    0 => 1,
                    1 => shared * (1 + r.next() % 1000),
                    _ => (r.next() >> (r.next() % 40)) | 1,
                };
                let m = Number::new_from_u64(m.max(1), ByteOrder::BigEndian);
                let res = match i % 3 {
                    0 => x.rem_euclid(&m),
                    1 => -(&x + &m),
                    _ => &x * 3u8,
                };
                let res = if round % 4 == 0 { other(&res) } else { res };
                (res, if round % 5 == 0 { other(&m) } else { m })
            })
            .collect();
        check(&c, &format!("round {round}, {k} words"));
    }
}

#[test]
fn crt_of_two_long_moduli_on_the_stack() {
    // Two moduli of 1 to 17 limbs (the stack takes up to 16), coprime or not, residues of both
    // signs, below the moduli or up to twice their length or longer.
    let mut r = Rng(0xc47_0002);
    let top = if cfg!(miri) { 3 } else { 17 };
    for k1 in 1..=top {
        for k2 in [1, k1, top + 1 - k1] {
            let m1 = &r.limbs(k1, ByteOrder::BigEndian) | &n(1);
            let m2 = &r.limbs(k2, ByteOrder::LittleEndian) | &n(1);
            let g = Number::new_from_u64(r.next() | 1, ByteOrder::BigEndian);
            for (a, b) in [(m1.clone(), m2.clone()), (&m1 * &g, &m2 * &g)] {
                let x = r.limbs(k1 + k2 + 1, ByteOrder::BigEndian);
                let res = [
                    (x.rem_euclid(&a), x.rem_euclid(&b)),
                    (-&x, &x * 5u8),
                    (r.limbs(2 * k1 + 3, ByteOrder::LittleEndian), -&x),
                    (x.rem_euclid(&a), (&x + 1u8).rem_euclid(&b)),
                ];
                for (ra, rb) in res {
                    check(
                        &[(ra.clone(), a.clone()), (rb.clone(), b.clone())],
                        &format!("{k1} and {k2} limbs"),
                    );
                    check(&[(other(&ra), other(&a)), (rb, b.clone())], "other forms");
                }
            }
        }
    }
}

#[test]
fn crt_of_many_words_by_the_trees() {
    // From 150 congruences modulo words on (12 under Miri), the product and remainder trees in
    // limbs; moduli that are not pairwise coprime take the other way.
    let (count, sizes): (usize, &[usize]) = if cfg!(miri) {
        (13, &[11, 12, 13])
    } else {
        (400, &[149, 150, 151, 255, 256, 257, 400])
    };
    let primes = primes_from(if cfg!(miri) { 1_000 } else { 1_000_000_007 }, count);
    let mut r = Rng(0xc47_0003);
    for &k in sizes {
        let x = r.limbs(k, ByteOrder::LittleEndian);
        let c: Vec<(Number, Number)> = primes[..k]
            .iter()
            .map(|&p| {
                let m = Number::new_from_u64(p, ByteOrder::BigEndian);
                (x.rem_euclid(&m), m)
            })
            .collect();
        check(&c, &format!("{k} primes"));
        let (y, l) = Number::crt(&c).unwrap();
        assert_eq!(y, x.rem_euclid(&l));
        // A modulus twice, and a modulus of 1.
        let mut d = c.clone();
        d.push(c[7].clone());
        d.push((n(5), n(1)));
        check(&d, &format!("{k} primes, one twice"));
        // A contradiction: the same modulus, another residue.
        let mut e = c.clone();
        e.push((&c[3].0 + 1u8, c[3].1.clone()));
        assert!(Number::crt(&e).is_none());
    }
}

#[test]
fn orders_and_roots_of_word_primes() {
    // Primes and composites above the trial division's 512^2 and below 2^32, which the strong tests
    // to 2, 7 and 61 decide, against the definitions: the order divides `p - 1` and no prime's
    // quotient of it takes the element to 1; a primitive root has the order `p - 1`.
    let primes: &[u64] = if cfg!(miri) {
        &[262_147]
    } else {
        &[
            262_147,
            1_000_003,
            2_147_483_647,
            4_294_967_291,
            3_215_031_751,
        ]
    };
    for &p in primes {
        let pn = Number::new_from_u64(p, ByteOrder::BigEndian);
        let is_prime = p != 3_215_031_751;
        let g = Number::primitive_root(&pn);
        assert_eq!(g.is_some(), is_prime, "{p}");
        if !is_prime {
            continue;
        }
        let g = g.unwrap();
        let p1 = &pn - 1u8;
        assert_eq!(g.multiplicative_order(&pn), Some(p1.clone()), "{p}");
        for a in [2u64, 3, 10, p - 1] {
            let an = Number::new_from_u64(a, ByteOrder::LittleEndian);
            let o = an.multiplicative_order(&pn).unwrap();
            assert!(p1.rem_euclid(&o).is_zero());
            assert_eq!(an.pow_mod(&o, &pn), n(1));
            assert_eq!(o.byte_order, ByteOrder::LittleEndian);
        }
    }
    // Strong pseudoprimes to 2 and 3 with no factor below 512: composite.
    for m in [1_373_653u64, 25_326_001] {
        let mn = Number::new_from_u64(m, ByteOrder::BigEndian);
        assert!(Number::primitive_root(&mn).is_none(), "{m}");
        assert!(mn.mobius() != -1, "{m}");
    }
}
