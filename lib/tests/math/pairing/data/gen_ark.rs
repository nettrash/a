// The generator of pairing_ref.txt, zcash_cases.txt and ../../field/data/tower_ref.txt: the
// main.rs of a probe crate (edition 2021) with
//   [dependencies] ark-bls12-381 = "0.5", ark-bn254 = "0.5", ark-ec = "0.5", ark-ff = "0.5",
//   ark-serialize = "0.5", bls12_381 = { version = "0.8", features = ["experimental"] },
//   group = "0.13", hex = "0.4", blst = "0.3"
// run with `cargo run --release`. Not part of the crate.
// Reference outputs for the tests of the area pairing, from arkworks (ark-bls12-381, ark-bn254
// 0.5), the bls12_381 crate 0.8 (zkcrypto) and blst 0.3 (BLS12-381 pairings only), on inputs
// from SplitMix64 that the tests regenerate. Prints `key value` lines: the digests of
// pairing_ref.txt and ../../field/data/tower_ref.txt, and the cases of zcash_cases.txt.
use ark_ec::pairing::Pairing;
use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{BigInteger, CyclotomicMultSubgroup, Field, PrimeField};
use group::Curve as _;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn words(&mut self, n: usize) -> Vec<u64> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn le_bytes(w: &[u64]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn eat(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn be<F: PrimeField>(x: &F, len: usize) -> Vec<u8> {
    let b = x.into_bigint().to_bytes_be();
    let b = &b[b.len() - len..];
    b.to_vec()
}

macro_rules! curve {
    ($name:literal, $m:ident, $E:ty, $words:expr, $len:expr) => {{
        use $m::{Fq, Fq12, Fq2, Fq6, G1Affine, G1Projective, G2Affine, G2Projective};
        let words: usize = $words;
        let len: usize = $len;
        let fp = |r: &mut Rng| Fq::from_le_bytes_mod_order(&le_bytes(&r.words(words)));
        let fp2 = |r: &mut Rng| {
            let a = fp(r);
            let b = fp(r);
            Fq2::new(a, b)
        };
        let fp6 = |r: &mut Rng| {
            let a = fp2(r);
            let b = fp2(r);
            let c = fp2(r);
            Fq6::new(a, b, c)
        };
        let fp12 = |r: &mut Rng| {
            let a = fp6(r);
            let b = fp6(r);
            Fq12::new(a, b)
        };
        let e2 = |h: &mut Fnv, x: &Fq2| {
            h.eat(&be(&x.c0, len));
            h.eat(&be(&x.c1, len));
        };
        let e6 = |h: &mut Fnv, x: &Fq6| {
            e2(h, &x.c0);
            e2(h, &x.c1);
            e2(h, &x.c2);
        };
        let e12 = |h: &mut Fnv, x: &Fq12| {
            e6(h, &x.c0);
            e6(h, &x.c1);
        };
        let out = |k: &str, h: &Fnv| println!("{}_{} {:016x}", $name, k, h.0);

        // The tower.
        let mut r = Rng(1);
        let names = [
            "fp12_mul", "fp12_square", "fp12_inverse", "fp12_div", "fp12_frob1", "fp12_frob2",
            "fp12_frob3", "fp12_frob6", "fp12_frob7", "fp12_pow", "fp12_mul_by_014",
            "fp12_mul_by_034", "fp12_cyclotomic", "fp12_norm", "fp6_mul", "fp6_square",
            "fp6_inverse", "fp6_frob1", "fp6_norm", "fp6_mul_by_01", "fp6_mul_by_1", "fp6_mul_by_v",
            "fp2_mul", "fp2_square", "fp2_inverse", "fp2_norm", "fp2_mul_by_xi", "fp2_is_square",
        ];
        let mut hs: Vec<Fnv> = names.iter().map(|_| Fnv::new()).collect();
        for _ in 0..16 {
            let a = fp12(&mut r);
            let b = fp12(&mut r);
            let a6 = fp6(&mut r);
            let b6 = fp6(&mut r);
            let a2 = fp2(&mut r);
            let b2 = fp2(&mut r);
            let k = r.words(1);
            let mut i = 0;
            let mut put12 = |hs: &mut Vec<Fnv>, x: &Fq12| {
                e12(&mut hs[i], x);
                i += 1;
            };
            put12(&mut hs, &(a * b));
            put12(&mut hs, &a.square());
            put12(&mut hs, &a.inverse().unwrap());
            put12(&mut hs, &(a * b.inverse().unwrap()));
            put12(&mut hs, &a.frobenius_map(1));
            put12(&mut hs, &a.frobenius_map(2));
            put12(&mut hs, &a.frobenius_map(3));
            put12(&mut hs, &a.frobenius_map(6));
            put12(&mut hs, &a.frobenius_map(7));
            put12(&mut hs, &a.pow(&k));
            let mut m = a;
            m.mul_by_014(&b.c0.c0, &b.c0.c1, &b.c1.c1);
            put12(&mut hs, &m);
            let mut m = a;
            m.mul_by_034(&b.c0.c0, &b.c1.c0, &b.c1.c1);
            put12(&mut hs, &m);
            // The cyclotomic subgroup: c = a^((p⁶ − 1)(p² + 1)), its square and its k-th power.
            let c1 = a.frobenius_map(6) * a.inverse().unwrap();
            let c = c1.frobenius_map(2) * c1;
            {
                let h = &mut hs[i];
                e12(h, &c);
                e12(h, &c.cyclotomic_square());
                e12(h, &c.cyclotomic_exp(&k));
            }
            i += 1;
            e6(&mut hs[i], &a.norm());
            i += 1;
            let mut put6 = |hs: &mut Vec<Fnv>, i: &mut usize, x: &Fq6| {
                e6(&mut hs[*i], x);
                *i += 1;
            };
            put6(&mut hs, &mut i, &(a6 * b6));
            put6(&mut hs, &mut i, &a6.square());
            put6(&mut hs, &mut i, &a6.inverse().unwrap());
            put6(&mut hs, &mut i, &a6.frobenius_map(1));
            e2(&mut hs[i], &a6.norm());
            i += 1;
            let mut m = a6;
            m.mul_by_01(&b6.c0, &b6.c1);
            put6(&mut hs, &mut i, &m);
            let mut m = a6;
            m.mul_by_1(&b6.c1);
            put6(&mut hs, &mut i, &m);
            let v = Fq6::new(Fq2::ZERO, Fq2::ONE, Fq2::ZERO);
            put6(&mut hs, &mut i, &(a6 * v));
            let mut put2 = |hs: &mut Vec<Fnv>, i: &mut usize, x: &Fq2| {
                e2(&mut hs[*i], x);
                *i += 1;
            };
            put2(&mut hs, &mut i, &(a2 * b2));
            put2(&mut hs, &mut i, &a2.square());
            put2(&mut hs, &mut i, &a2.inverse().unwrap());
            hs[i].eat(&be(&a2.norm(), len));
            i += 1;
            let xi = v * v * v;
            put2(&mut hs, &mut i, &(a2 * xi.c0));
            hs[i].eat(&[a2.sqrt().is_some() as u8, b2.sqrt().is_some() as u8, (a2 * a2).sqrt().is_some() as u8]);
        }
        for (n, h) in names.iter().zip(&hs) {
            out(n, h);
        }
    }};
}

fn bls_points() {
    use ark_bls12_381::{Bls12_381, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
    let len = 48;
    let fp = |r: &mut Rng| Fq::from_le_bytes_mod_order(&le_bytes(&r.words(7)));
    let fp2 = |r: &mut Rng| {
        let a = fp(r);
        let b = fp(r);
        Fq2::new(a, b)
    };
    let zk1 = |p: &G1Affine| -> Vec<u8> {
        let mut b = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(p, &mut b).unwrap();
        b
    };
    let zk2 = |p: &G2Affine| -> Vec<u8> {
        let mut b = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(p, &mut b).unwrap();
        b
    };
    let e12 = |h: &mut Fnv, x: &ark_bls12_381::Fq12| {
        for c in [x.c0, x.c1] {
            for d in [c.c0, c.c1, c.c2] {
                h.eat(&be(&d.c0, len));
                h.eat(&be(&d.c1, len));
            }
        }
    };
    // Pairings of seeded multiples, by arkworks and by the bls12_381 crate.
    let mut r = Rng(2);
    let (mut hg1, mut hg2, mut he, mut hz, mut hm, mut hb) = (Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new());
    let mut ps = vec![];
    let mut qs = vec![];
    for _ in 0..6 {
        let a = r.words(4);
        let b = r.words(4);
        let p = G1Affine::generator().mul_bigint(&a).into_affine();
        let q = G2Affine::generator().mul_bigint(&b).into_affine();
        hg1.eat(&zk1(&p));
        hg2.eat(&zk2(&q));
        let e = Bls12_381::pairing(p, q).0;
        e12(&mut he, &e);
        // The same with zkcrypto: scalars reduced modulo r, points from the encodings.
        let zp = bls12_381::G1Affine::from_compressed(&zk1(&p).try_into().unwrap()).unwrap();
        let zq = bls12_381::G2Affine::from_compressed(&zk2(&q).try_into().unwrap()).unwrap();
        let sa = bls12_381::Scalar::from_raw([a[0], a[1], a[2], a[3]]);
        assert_eq!((bls12_381::G1Affine::generator() * sa).to_affine(), zp);
        let ze = bls12_381::pairing(&zp, &zq);
        // zkcrypto's Gt has no accessor: compare through its Debug, the same twelve hex numbers.
        let s = format!("{:?}", ze);
        let hexes: Vec<String> = s
            .split(|c: char| !c.is_ascii_hexdigit() && c != 'x')
            .filter(|t| t.starts_with("0x"))
            .map(|t| t.to_string())
            .collect();
        assert_eq!(hexes.len(), 12);
        for t in &hexes {
            let v = hex::decode(&t[2..]).unwrap();
            hz.eat(&v);
        }
        // And with blst, from the same encodings.
        // SAFETY: blst's C functions on values of its own types, initialised by it.
        unsafe {
            let mut bp = blst::blst_p1_affine::default();
            let mut bq = blst::blst_p2_affine::default();
            assert_eq!(blst::blst_p1_uncompress(&mut bp, zk1(&p).as_ptr()), blst::BLST_ERROR::BLST_SUCCESS);
            assert_eq!(blst::blst_p2_uncompress(&mut bq, zk2(&q).as_ptr()), blst::BLST_ERROR::BLST_SUCCESS);
            let mut f = blst::blst_fp12::default();
            blst::blst_miller_loop(&mut f, &bq, &bp);
            let mut e = blst::blst_fp12::default();
            blst::blst_final_exp(&mut e, &f);
            for i in 0..2 {
                for j in 0..3 {
                    for k in 0..2 {
                        let mut out = [0u8; 48];
                        blst::blst_bendian_from_fp(out.as_mut_ptr(), &e.fp6[i].fp2[j].fp[k]);
                        hb.eat(&out);
                    }
                }
            }
        }
        ps.push(p);
        qs.push(q);
    }
    assert_eq!(he.0, hb.0, "arkworks and blst disagree");
    let m = Bls12_381::multi_pairing(ps.clone(), qs.clone()).0;
    e12(&mut hm, &m);
    assert_eq!(he.0, hz.0, "arkworks and zkcrypto disagree");
    println!("bls_g1_mul {:016x}", hg1.0);
    println!("bls_g2_mul {:016x}", hg2.0);
    println!("bls_pairing {:016x}", he.0);
    println!("bls_multi_pairing {:016x}", hm.0);

    // Points outside the subgroups: subgroup checks and cofactor clearings.
    let mut r = Rng(3);
    let (mut hs1, mut hc1, mut hh1, mut hs2, mut hc2, mut hh2) =
        (Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new());
    for _ in 0..8 {
        let mut x = fp(&mut r);
        let p = loop {
            if let Some(p) = G1Affine::get_point_from_x_unchecked(x, false) {
                break p;
            }
            x += Fq::ONE;
        };
        hs1.eat(&[p.is_in_correct_subgroup_assuming_on_curve() as u8]);
        hc1.eat(&zk1(&p.clear_cofactor()));
        hh1.eat(&zk1(&p.mul_by_cofactor()));
        let mut x = fp2(&mut r);
        let q = loop {
            if let Some(q) = G2Affine::get_point_from_x_unchecked(x, false) {
                break q;
            }
            x.c0 += Fq::ONE;
        };
        hs2.eat(&[q.is_in_correct_subgroup_assuming_on_curve() as u8]);
        hc2.eat(&zk2(&q.clear_cofactor()));
        hh2.eat(&zk2(&q.mul_by_cofactor()));
    }
    println!("bls_g1_subgroup {:016x}", hs1.0);
    println!("bls_g1_clear_cofactor {:016x}", hc1.0);
    println!("bls_g1_mul_by_cofactor {:016x}", hh1.0);
    println!("bls_g2_subgroup {:016x}", hs2.0);
    println!("bls_g2_clear_cofactor {:016x}", hc2.0);
    println!("bls_g2_mul_by_cofactor {:016x}", hh2.0);
    let _ = (G1Projective::generator(), G2Projective::generator(), Fr::ONE);
}

fn bn_points() {
    use ark_bn254::{Bn254, Fq, Fq2, G1Affine, G2Affine};
    let len = 32;
    let fp = |r: &mut Rng| Fq::from_le_bytes_mod_order(&le_bytes(&r.words(5)));
    let fp2 = |r: &mut Rng| {
        let a = fp(r);
        let b = fp(r);
        Fq2::new(a, b)
    };
    let eth1 = |p: &G1Affine| -> Vec<u8> {
        if p.is_zero() {
            return vec![0; 64];
        }
        let mut b = be(&p.x, len);
        b.extend(be(&p.y, len));
        b
    };
    let eth2 = |q: &G2Affine| -> Vec<u8> {
        if q.is_zero() {
            return vec![0; 128];
        }
        let mut b = be(&q.x.c1, len);
        b.extend(be(&q.x.c0, len));
        b.extend(be(&q.y.c1, len));
        b.extend(be(&q.y.c0, len));
        b
    };
    let e12 = |h: &mut Fnv, x: &ark_bn254::Fq12| {
        for c in [x.c0, x.c1] {
            for d in [c.c0, c.c1, c.c2] {
                h.eat(&be(&d.c0, len));
                h.eat(&be(&d.c1, len));
            }
        }
    };
    let mut r = Rng(2);
    let (mut hg1, mut hg2, mut he, mut hm) = (Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new());
    let mut ps = vec![];
    let mut qs = vec![];
    for _ in 0..6 {
        let a = r.words(4);
        let b = r.words(4);
        let p = G1Affine::generator().mul_bigint(&a).into_affine();
        let q = G2Affine::generator().mul_bigint(&b).into_affine();
        hg1.eat(&eth1(&p));
        hg2.eat(&eth2(&q));
        e12(&mut he, &Bn254::pairing(p, q).0);
        ps.push(p);
        qs.push(q);
    }
    e12(&mut hm, &Bn254::multi_pairing(ps, qs).0);
    println!("bn_g1_mul {:016x}", hg1.0);
    println!("bn_g2_mul {:016x}", hg2.0);
    println!("bn_pairing {:016x}", he.0);
    println!("bn_multi_pairing {:016x}", hm.0);

    let mut r = Rng(3);
    let (mut hs1, mut hs2, mut hh2) = (Fnv::new(), Fnv::new(), Fnv::new());
    for _ in 0..8 {
        let mut x = fp(&mut r);
        let p = loop {
            if let Some(p) = G1Affine::get_point_from_x_unchecked(x, false) {
                break p;
            }
            x += Fq::ONE;
        };
        hs1.eat(&[p.is_in_correct_subgroup_assuming_on_curve() as u8]);
        let mut x = fp2(&mut r);
        let q = loop {
            if let Some(q) = G2Affine::get_point_from_x_unchecked(x, false) {
                break q;
            }
            x.c0 += Fq::ONE;
        };
        hs2.eat(&[q.is_in_correct_subgroup_assuming_on_curve() as u8]);
        // arkworks clears the cofactor of BN254's G2 by the product by it.
        assert_eq!(q.clear_cofactor(), q.mul_by_cofactor());
        hh2.eat(&eth2(&q.mul_by_cofactor()));
    }
    println!("bn_g1_subgroup {:016x}", hs1.0);
    println!("bn_g2_subgroup {:016x}", hs2.0);
    println!("bn_g2_mul_by_cofactor {:016x}", hh2.0);
}


/// ZCash encodings, valid and not, with the verdict of the bls12_381 crate: `zcash_case <group>
/// <hex> <canonical compressed hex of the decoded point | err>`.
fn zcash_cases() {
    use bls12_381::{G1Affine, G1Projective, G2Affine, G2Projective, Scalar};
    let p = hex::decode("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab").unwrap();
    // b + p for 48 big-endian bytes, if it stays clear of the three flag bits.
    let add_p = |b: &[u8]| -> Option<Vec<u8>> {
        let mut out = vec![0u8; 48];
        let mut carry = 0u16;
        for i in (0..48).rev() {
            let s = b[i] as u16 + p[i] as u16 + carry;
            out[i] = s as u8;
            carry = s >> 8;
        }
        (carry == 0 && out[0] & 0xe0 == 0).then_some(out)
    };
    let verdict = |g: u8, c: &[u8]| -> String {
        if g == 1 {
            let a: Option<G1Affine> = if c.len() == 48 {
                G1Affine::from_compressed(&c.try_into().unwrap()).into()
            } else {
                G1Affine::from_uncompressed(&c.try_into().unwrap()).into()
            };
            a.map_or("err".to_string(), |a| hex::encode(a.to_compressed()))
        } else {
            let a: Option<G2Affine> = if c.len() == 96 {
                G2Affine::from_compressed(&c.try_into().unwrap()).into()
            } else {
                G2Affine::from_uncompressed(&c.try_into().unwrap()).into()
            };
            a.map_or("err".to_string(), |a| hex::encode(a.to_compressed()))
        }
    };
    let mut r = Rng(4);
    let mut cases: Vec<(u8, Vec<u8>)> = vec![];
    for _ in 0..3 {
        // Small multiples too, whose coordinates are as likely to be small as large.
        let s = Scalar::from_raw([r.next(), r.next(), r.next(), r.next() >> 2]);
        let a = (G1Projective::generator() * s).to_affine();
        let b = (G2Projective::generator() * s).to_affine();
        for (g, enc) in [
            (1u8, a.to_compressed().to_vec()),
            (1, a.to_uncompressed().to_vec()),
            (2, b.to_compressed().to_vec()),
            (2, b.to_uncompressed().to_vec()),
        ] {
            let n = enc.len();
            cases.push((g, enc.clone()));
            for flip in [0x80u8, 0x40, 0x20] {
                let mut e = enc.clone();
                e[0] ^= flip;
                cases.push((g, e));
            }
            // Each 48-byte coefficient plus p, where that fits.
            for at in (0..n).step_by(48) {
                let mut e = enc.clone();
                let flags = if at == 0 { e[0] & 0xe0 } else { 0 };
                if at == 0 {
                    e[0] &= 0x1f;
                }
                if let Some(bumped) = add_p(&e[at..at + 48]) {
                    e[at..at + 48].copy_from_slice(&bumped);
                    e[0] |= flags;
                    cases.push((g, e));
                }
            }
            let mut e = enc.clone();
            e[n - 1] ^= 1;
            cases.push((g, e));
        }
    }
    // The points at infinity and their corruptions; x = 0 (on G1's curve, of order 3).
    for (g, len) in [(1u8, 48usize), (1, 96), (2, 96), (2, 192)] {
        for first in [0xc0u8, 0x40, 0xe0, 0x60, 0x80, 0xa0, 0x00] {
            let mut e = vec![0u8; len];
            e[0] = first;
            cases.push((g, e.clone()));
            e[len - 1] = 1;
            cases.push((g, e));
        }
    }
    for (g, c) in &cases {
        println!("zcash_case g{} {} {}", g, hex::encode(c), verdict(*g, c));
    }
}

fn main() {
    curve!("bls", ark_bls12_381, ark_bls12_381::Bls12_381, 7, 48);
    curve!("bn", ark_bn254, ark_bn254::Bn254, 5, 32);
    bls_points();
    bn_points();
    zcash_cases();
}
