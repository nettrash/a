// Hashing to curves (RFC 9380): a::math::pairing::hash (BLS12-381's G1 and G2, BN254's G1) and
// a::math::ec::hash (P-256, P-384, P-521, secp256k1, curve25519, edwards25519) against blst
// 0.3.17, arkworks 0.5, the bls12_381 crate 0.8, mcl, RustCrypto's p256/p384/p521/k256 0.13 and
// curve25519-dalek 5.0.
//
// Messages: 16 seeded random messages of 32 bytes; tags: the RFC's "QUUX-V01-CS02-with-<suite>".
// Every output is compared with ours before timing. SHA-2 below each reference: blst and mcl their
// own; arkworks, bls12_381 and RustCrypto take RustCrypto's sha2 (0.10.9 and 0.9.9 with default
// features: the portable backend on aarch64); dalek takes sha2 0.11.1 (ARMv8 SHA-512 instructions,
// detected at run time).
#![allow(unused_must_use)]
#[path = "../common.rs"]
mod common;
#[cfg(have_mcl)]
#[path = "../mcl.rs"]
mod mcl;

use common::*;

use a::hash::{Sha256, Sha512};
use a::math::field::Fp2;
use a::math::pairing::hash as ph;
use a::math::pairing::{G1, G2};

const NM: usize = 16;

fn strip(h: &str) -> String {
    let t = h.trim_start_matches('0');
    if t.is_empty() { "0".into() } else { t.to_lowercase() }
}

fn main() {
    let mut b = Bench::new("hash_to_curve");
    let mut rng = Rng(5);
    let msgs: Vec<Vec<u8>> = (0..NM).map(|_| rng.bytes(32)).collect();
    bls(&mut b, &msgs);
    bn(&mut b, &msgs);
    prime(&mut b, &msgs);
    let _ = (Sha256::new(), Sha512::new());
}

fn bls(b: &mut Bench, msgs: &[Vec<u8>]) {
    use ark_bls12_381 as ab;
    use ark_ec::hashing::curve_maps::wb::WBMap;
    use ark_ec::hashing::map_to_curve_hasher::{MapToCurve as _, MapToCurveBasedHasher};
    use ark_ec::hashing::HashToCurve as _;
    use ark_ff::field_hashers::DefaultFieldHasher;
    use ark_serialize::CanonicalSerialize;
    use bls12_381::hash_to_curve::{ExpandMsgXmd, HashToCurve, HashToField, MapToCurve};

    type ZXmd = ExpandMsgXmd<sha2_09::Sha256>;
    type AH = DefaultFieldHasher<sha2_10::Sha256, 128>;
    type AG1 = MapToCurveBasedHasher<ab::G1Projective, AH, WBMap<ab::g1::Config>>;
    type AG2 = MapToCurveBasedHasher<ab::G2Projective, AH, WBMap<ab::g2::Config>>;
    type ZFp = <bls12_381::G1Projective as MapToCurve>::Field;
    type ZFp2 = <bls12_381::G2Projective as MapToCurve>::Field;

    let ser_c = |p: &dyn Fn(&mut Vec<u8>)| {
        let mut v = Vec::new();
        p(&mut v);
        v
    };
    let ark_c1 = |p: &ab::G1Affine| ser_c(&|v| p.serialize_compressed(v).unwrap());
    let ark_c2 = |p: &ab::G2Affine| ser_c(&|v| p.serialize_compressed(v).unwrap());
    let blst_c1 = |p: &blst::blst_p1| {
        let mut o = [0u8; 48];
        // SAFETY: writes 48 bytes.
        unsafe { blst::blst_p1_compress(o.as_mut_ptr(), p) };
        o.to_vec()
    };
    let blst_c2 = |p: &blst::blst_p2| {
        let mut o = [0u8; 96];
        // SAFETY: writes 96 bytes.
        unsafe { blst::blst_p2_compress(o.as_mut_ptr(), p) };
        o.to_vec()
    };
    #[cfg(have_mcl)]
    let mcl_name = {
        mcl::init(mcl::BLS12_381);
        // SAFETY: mcl's settings after init.
        unsafe {
            mcl::mclBn_setETHserialization(1);
            assert_eq!(mcl::mclBn_setMapToMode(mcl::MAP_TO_MODE_HASH_TO_CURVE), 0);
        }
        format!("mcl {}", mcl::version())
    };

    // ---------------- G1 ----------------
    let s = ph::bls12_381_g1();
    let ro = b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_";
    let nu = b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_NU_";
    let ag1 = <AG1 as ark_ec::hashing::HashToCurve<ab::G1Projective>>::new(ro).unwrap();
    let ah = <AH as ark_ff::field_hashers::HashToField<ab::Fq>>::new(ro);
    for m in msgs.iter().take(4) {
        let w = s.hash_to_curve(m, ro).to_zcash_compressed();
        assert_eq!(ark_c1(&ag1.hash(m).unwrap()), w, "ark g1 h2c");
        let z = <bls12_381::G1Projective as HashToCurve<ZXmd>>::hash_to_curve(m, ro);
        assert_eq!(bls12_381::G1Affine::from(z).to_compressed().to_vec(), w, "zk g1 h2c");
        let mut o = blst::blst_p1::default();
        // SAFETY: blst reads the message and tag.
        unsafe { blst::blst_hash_to_g1(&mut o, m.as_ptr(), m.len(), ro.as_ptr(), ro.len(), core::ptr::null(), 0) };
        assert_eq!(blst_c1(&o), w, "blst g1 h2c");
        #[cfg(have_mcl)]
        {
            let mut p = mcl::G1::default();
            // SAFETY: mcl reads the message and tag.
            assert_eq!(unsafe { mcl::mclBnG1_hashAndMapToWithDst(&mut p, m.as_ptr(), m.len(), ro.as_ptr(), ro.len()) }, 0);
            let mut o = [0u8; 48];
            // SAFETY: writes at most 48 bytes.
            unsafe { mcl::mclBnG1_serialize(o.as_mut_ptr(), 48, &p) };
            assert_eq!(o.to_vec(), w, "mcl g1 h2c");
        }
        let w = s.encode_to_curve(m, nu).to_zcash_compressed();
        let z = <bls12_381::G1Projective as HashToCurve<ZXmd>>::encode_to_curve(m, nu);
        assert_eq!(bls12_381::G1Affine::from(z).to_compressed().to_vec(), w, "zk g1 e2c");
        let mut o = blst::blst_p1::default();
        // SAFETY: as above.
        unsafe { blst::blst_encode_to_g1(&mut o, m.as_ptr(), m.len(), nu.as_ptr(), nu.len(), core::ptr::null(), 0) };
        assert_eq!(blst_c1(&o), w, "blst g1 e2c");
    }
    let p1 = "BLS12-381 G1, 32-byte message, 50-byte tag";
    let mut refs = vec![
        side("blst 0.3.17", "blst_hash_to_g1", |i| {
            let m = &msgs[i % NM];
            let mut o = blst::blst_p1::default();
            // SAFETY: blst reads the message and tag.
            unsafe { blst::blst_hash_to_g1(&mut o, m.as_ptr(), m.len(), ro.as_ptr(), ro.len(), core::ptr::null(), 0) };
            black_box(o);
        }),
        side("arkworks 0.5", "MapToCurveBasedHasher with WBMap; SHA-256 by sha2 0.10.9 (portable)", |i| {
            black_box(ag1.hash(&msgs[i % NM]).unwrap());
        }),
        side("bls12_381 0.8", "HashToCurve<ExpandMsgXmd<sha2 0.9.9 Sha256>> (portable SHA-256)", |i| {
            black_box(<bls12_381::G1Projective as HashToCurve<ZXmd>>::hash_to_curve(&msgs[i % NM], ro));
        }),
    ];
    #[cfg(have_mcl)]
    refs.push(side(&mcl_name, "mclBnG1_hashAndMapToWithDst, map-to mode HASH_TO_CURVE", |i| {
        let m = &msgs[i % NM];
        let mut p = mcl::G1::default();
        // SAFETY: mcl reads the message and tag.
        unsafe { mcl::mclBnG1_hashAndMapToWithDst(&mut p, m.as_ptr(), m.len(), ro.as_ptr(), ro.len()) };
        black_box(p);
    }));
    b.run("bls12_381_g1.hash_to_curve", p1, "", |i| {
        black_box(s.hash_to_curve(&msgs[i % NM], ro));
    }, refs);
    b.run("bls12_381_g1.encode_to_curve", p1, "", |i| {
        black_box(s.encode_to_curve(&msgs[i % NM], nu));
    }, vec![
        side("blst 0.3.17", "blst_encode_to_g1", |i| {
            let m = &msgs[i % NM];
            let mut o = blst::blst_p1::default();
            // SAFETY: blst reads the message and tag.
            unsafe { blst::blst_encode_to_g1(&mut o, m.as_ptr(), m.len(), nu.as_ptr(), nu.len(), core::ptr::null(), 0) };
            black_box(o);
        }),
        side("bls12_381 0.8", "portable SHA-256", |i| {
            black_box(<bls12_381::G1Projective as HashToCurve<ZXmd>>::encode_to_curve(&msgs[i % NM], nu));
        }),
    ]);
    // hash_to_field, two elements.
    for m in msgs.iter().take(4) {
        let u = s.hash_to_field(m, ro, 2);
        let a: [ab::Fq; 2] = <AH as ark_ff::field_hashers::HashToField<ab::Fq>>::hash_to_field::<2>(&ah, m);
        let mut z = [ZFp::default(); 2];
        <ZFp as HashToField>::hash_to_field::<ZXmd>(m, ro, &mut z);
        for k in 0..2 {
            use ark_ff::{BigInteger, PrimeField};
            let want = strip(&u[k].value().to_str_radix(16));
            assert_eq!(strip(&hex(&a[k].into_bigint().to_bytes_be())), want);
            assert_eq!(strip(&hex(&z[k].to_bytes())), want);
        }
    }
    b.run("bls12_381_g1.hash_to_field", &format!("{p1}, 2 elements"), "", |i| {
        black_box(s.hash_to_field(&msgs[i % NM], ro, 2));
    }, vec![
        side("arkworks 0.5", "DefaultFieldHasher<sha2 0.10.9 Sha256> (portable)", |i| {
            black_box(<AH as ark_ff::field_hashers::HashToField<ab::Fq>>::hash_to_field::<2>(&ah, &msgs[i % NM]));
        }),
        side("bls12_381 0.8", "portable SHA-256", |i| {
            let mut z = [ZFp::default(); 2];
            <ZFp as HashToField>::hash_to_field::<ZXmd>(&msgs[i % NM], ro, &mut z);
            black_box(z);
        }),
    ]);
    // map_to_curve on the same field elements.
    let ou: Vec<_> = msgs.iter().map(|m| s.hash_to_field(m, ro, 1).remove(0)).collect();
    let au: Vec<ab::Fq> = msgs.iter().map(|m| <AH as ark_ff::field_hashers::HashToField<ab::Fq>>::hash_to_field::<1>(&ah, m)[0]).collect();
    let zu: Vec<ZFp> = msgs
        .iter()
        .map(|m| {
            let mut z = [ZFp::default(); 1];
            <ZFp as HashToField>::hash_to_field::<ZXmd>(m, ro, &mut z);
            z[0]
        })
        .collect();
    let bu: Vec<blst::blst_fp> = ou
        .iter()
        .map(|u| {
            let be = u.to_bytes_be();
            let mut x = [0u8; 48];
            x[48 - be.len()..].copy_from_slice(&be);
            let mut f = blst::blst_fp::default();
            // SAFETY: reads 48 bytes.
            unsafe { blst::blst_fp_from_bendian(&mut f, x.as_ptr()) };
            f
        })
        .collect();
    for i in 0..4 {
        let q = s.map_to_curve(&ou[i]);
        let w = q.to_zcash_compressed();
        assert_eq!(ark_c1(&WBMap::<ab::g1::Config>::map_to_curve(au[i]).unwrap()), w, "ark map g1");
        assert_eq!(bls12_381::G1Affine::from(<bls12_381::G1Projective as MapToCurve>::map_to_curve(&zu[i])).to_compressed().to_vec(), w);
        let mut o = blst::blst_p1::default();
        // SAFETY: blst on its own types, v absent.
        unsafe { blst::blst_map_to_g1(&mut o, &bu[i], core::ptr::null()) };
        assert_eq!(blst_c1(&o), s.clear_cofactor(&q).to_zcash_compressed(), "blst map+clear g1");
    }
    b.run("bls12_381_g1.map_to_curve", "BLS12-381 G1, one element", "SSWU + 11-isogeny, no clearing", |i| {
        black_box(s.map_to_curve(&ou[i % NM]));
    }, vec![
        side("arkworks 0.5", "WBMap::map_to_curve (affine out)", |i| {
            black_box(WBMap::<ab::g1::Config>::map_to_curve(au[i % NM]).unwrap());
        }),
        side("bls12_381 0.8", "MapToCurve::map_to_curve", |i| {
            black_box(<bls12_381::G1Projective as MapToCurve>::map_to_curve(&zu[i % NM]));
        }),
    ]);
    b.run("bls12_381_g1.map_to_curve+clear_cofactor", "BLS12-381 G1, one element", "", |i| {
        black_box(s.clear_cofactor(&s.map_to_curve(&ou[i % NM])));
    }, vec![
        side("blst 0.3.17", "blst_map_to_g1(u, NULL)", |i| {
            let mut o = blst::blst_p1::default();
            // SAFETY: blst on its own types.
            unsafe { blst::blst_map_to_g1(&mut o, &bu[i % NM], core::ptr::null()) };
            black_box(o);
        }),
        side("arkworks 0.5", "WBMap::map_to_curve + clear_cofactor", |i| {
            use ark_ec::AffineRepr;
            black_box(WBMap::<ab::g1::Config>::map_to_curve(au[i % NM]).unwrap().clear_cofactor());
        }),
        side("bls12_381 0.8", "map_to_curve + clear_h", |i| {
            black_box(<bls12_381::G1Projective as MapToCurve>::map_to_curve(&zu[i % NM]).clear_h());
        }),
    ]);

    // ---------------- G2 ----------------
    let s = ph::bls12_381_g2();
    let ro = b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_RO_";
    let nu = b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_NU_";
    let ag2 = <AG2 as ark_ec::hashing::HashToCurve<ab::G2Projective>>::new(ro).unwrap();
    let ah2 = <AH as ark_ff::field_hashers::HashToField<ab::Fq2>>::new(ro);
    for m in msgs.iter().take(4) {
        let w = s.hash_to_curve(m, ro).to_zcash_compressed();
        assert_eq!(ark_c2(&ag2.hash(m).unwrap()), w, "ark g2 h2c");
        let z = <bls12_381::G2Projective as HashToCurve<ZXmd>>::hash_to_curve(m, ro);
        assert_eq!(bls12_381::G2Affine::from(z).to_compressed().to_vec(), w, "zk g2 h2c");
        let mut o = blst::blst_p2::default();
        // SAFETY: blst reads the message and tag.
        unsafe { blst::blst_hash_to_g2(&mut o, m.as_ptr(), m.len(), ro.as_ptr(), ro.len(), core::ptr::null(), 0) };
        assert_eq!(blst_c2(&o), w, "blst g2 h2c");
        #[cfg(have_mcl)]
        {
            let mut p = mcl::G2::default();
            // SAFETY: mcl reads the message and tag.
            assert_eq!(unsafe { mcl::mclBnG2_hashAndMapToWithDst(&mut p, m.as_ptr(), m.len(), ro.as_ptr(), ro.len()) }, 0);
            let mut o = [0u8; 96];
            // SAFETY: writes at most 96 bytes.
            unsafe { mcl::mclBnG2_serialize(o.as_mut_ptr(), 96, &p) };
            assert_eq!(o.to_vec(), w, "mcl g2 h2c");
        }
        let w = s.encode_to_curve(m, nu).to_zcash_compressed();
        let z = <bls12_381::G2Projective as HashToCurve<ZXmd>>::encode_to_curve(m, nu);
        assert_eq!(bls12_381::G2Affine::from(z).to_compressed().to_vec(), w, "zk g2 e2c");
        let mut o = blst::blst_p2::default();
        // SAFETY: as above.
        unsafe { blst::blst_encode_to_g2(&mut o, m.as_ptr(), m.len(), nu.as_ptr(), nu.len(), core::ptr::null(), 0) };
        assert_eq!(blst_c2(&o), w, "blst g2 e2c");
    }
    let p2 = "BLS12-381 G2, 32-byte message, 50-byte tag";
    let mut refs = vec![
        side("blst 0.3.17", "blst_hash_to_g2", |i| {
            let m = &msgs[i % NM];
            let mut o = blst::blst_p2::default();
            // SAFETY: blst reads the message and tag.
            unsafe { blst::blst_hash_to_g2(&mut o, m.as_ptr(), m.len(), ro.as_ptr(), ro.len(), core::ptr::null(), 0) };
            black_box(o);
        }),
        side("arkworks 0.5", "MapToCurveBasedHasher with WBMap; portable SHA-256", |i| {
            black_box(ag2.hash(&msgs[i % NM]).unwrap());
        }),
        side("bls12_381 0.8", "portable SHA-256", |i| {
            black_box(<bls12_381::G2Projective as HashToCurve<ZXmd>>::hash_to_curve(&msgs[i % NM], ro));
        }),
    ];
    #[cfg(have_mcl)]
    refs.push(side(&mcl_name, "mclBnG2_hashAndMapToWithDst, map-to mode HASH_TO_CURVE", |i| {
        let m = &msgs[i % NM];
        let mut p = mcl::G2::default();
        // SAFETY: mcl reads the message and tag.
        unsafe { mcl::mclBnG2_hashAndMapToWithDst(&mut p, m.as_ptr(), m.len(), ro.as_ptr(), ro.len()) };
        black_box(p);
    }));
    b.run("bls12_381_g2.hash_to_curve", p2, "", |i| {
        black_box(s.hash_to_curve(&msgs[i % NM], ro));
    }, refs);
    b.run("bls12_381_g2.encode_to_curve", p2, "", |i| {
        black_box(s.encode_to_curve(&msgs[i % NM], nu));
    }, vec![
        side("blst 0.3.17", "blst_encode_to_g2", |i| {
            let m = &msgs[i % NM];
            let mut o = blst::blst_p2::default();
            // SAFETY: blst reads the message and tag.
            unsafe { blst::blst_encode_to_g2(&mut o, m.as_ptr(), m.len(), nu.as_ptr(), nu.len(), core::ptr::null(), 0) };
            black_box(o);
        }),
        side("bls12_381 0.8", "portable SHA-256", |i| {
            black_box(<bls12_381::G2Projective as HashToCurve<ZXmd>>::encode_to_curve(&msgs[i % NM], nu));
        }),
    ]);
    for m in msgs.iter().take(4) {
        let u = s.hash_to_field(m, ro, 2);
        let a: [ab::Fq2; 2] = <AH as ark_ff::field_hashers::HashToField<ab::Fq2>>::hash_to_field::<2>(&ah2, m);
        let mut z = [ZFp2::default(); 2];
        <ZFp2 as HashToField>::hash_to_field::<ZXmd>(m, ro, &mut z);
        for k in 0..2 {
            use ark_ff::{BigInteger, PrimeField};
            let w0 = strip(&u[k].c0().value().to_str_radix(16));
            let w1 = strip(&u[k].c1().value().to_str_radix(16));
            assert_eq!(strip(&hex(&a[k].c0.into_bigint().to_bytes_be())), w0);
            assert_eq!(strip(&hex(&a[k].c1.into_bigint().to_bytes_be())), w1);
            assert_eq!(strip(&hex(&z[k].c0.to_bytes())), w0);
            assert_eq!(strip(&hex(&z[k].c1.to_bytes())), w1);
        }
    }
    b.run("bls12_381_g2.hash_to_field", &format!("{p2}, 2 elements"), "", |i| {
        black_box(s.hash_to_field(&msgs[i % NM], ro, 2));
    }, vec![
        side("arkworks 0.5", "DefaultFieldHasher<sha2 0.10.9 Sha256> (portable)", |i| {
            black_box(<AH as ark_ff::field_hashers::HashToField<ab::Fq2>>::hash_to_field::<2>(&ah2, &msgs[i % NM]));
        }),
        side("bls12_381 0.8", "portable SHA-256", |i| {
            let mut z = [ZFp2::default(); 2];
            <ZFp2 as HashToField>::hash_to_field::<ZXmd>(&msgs[i % NM], ro, &mut z);
            black_box(z);
        }),
    ]);
    let ou: Vec<Fp2> = msgs.iter().map(|m| s.hash_to_field(m, ro, 1).remove(0)).collect();
    let au: Vec<ab::Fq2> = msgs.iter().map(|m| <AH as ark_ff::field_hashers::HashToField<ab::Fq2>>::hash_to_field::<1>(&ah2, m)[0]).collect();
    let zu: Vec<ZFp2> = msgs
        .iter()
        .map(|m| {
            let mut z = [ZFp2::default(); 1];
            <ZFp2 as HashToField>::hash_to_field::<ZXmd>(m, ro, &mut z);
            z[0]
        })
        .collect();
    let bu: Vec<blst::blst_fp2> = ou
        .iter()
        .map(|u| {
            let mut f = blst::blst_fp2::default();
            for (k, c) in [u.c0(), u.c1()].into_iter().enumerate() {
                let be = c.to_bytes_be();
                let mut x = [0u8; 48];
                x[48 - be.len()..].copy_from_slice(&be);
                // SAFETY: reads 48 bytes.
                unsafe { blst::blst_fp_from_bendian(&mut f.fp[k], x.as_ptr()) };
            }
            f
        })
        .collect();
    for i in 0..4 {
        let q = s.map_to_curve(&ou[i]);
        let w = q.to_zcash_compressed();
        assert_eq!(ark_c2(&WBMap::<ab::g2::Config>::map_to_curve(au[i]).unwrap()), w, "ark map g2");
        assert_eq!(bls12_381::G2Affine::from(<bls12_381::G2Projective as MapToCurve>::map_to_curve(&zu[i])).to_compressed().to_vec(), w);
        let mut o = blst::blst_p2::default();
        // SAFETY: blst on its own types, v absent.
        unsafe { blst::blst_map_to_g2(&mut o, &bu[i], core::ptr::null()) };
        assert_eq!(blst_c2(&o), s.clear_cofactor(&q).to_zcash_compressed(), "blst map+clear g2");
    }
    b.run("bls12_381_g2.map_to_curve", "BLS12-381 G2, one element", "SSWU over Fp2 + 3-isogeny, no clearing", |i| {
        black_box(s.map_to_curve(&ou[i % NM]));
    }, vec![
        side("arkworks 0.5", "WBMap::map_to_curve (affine out)", |i| {
            black_box(WBMap::<ab::g2::Config>::map_to_curve(au[i % NM]).unwrap());
        }),
        side("bls12_381 0.8", "MapToCurve::map_to_curve", |i| {
            black_box(<bls12_381::G2Projective as MapToCurve>::map_to_curve(&zu[i % NM]));
        }),
    ]);
    b.run("bls12_381_g2.map_to_curve+clear_cofactor", "BLS12-381 G2, one element", "", |i| {
        black_box(s.clear_cofactor(&s.map_to_curve(&ou[i % NM])));
    }, vec![
        side("blst 0.3.17", "blst_map_to_g2(u, NULL)", |i| {
            let mut o = blst::blst_p2::default();
            // SAFETY: blst on its own types.
            unsafe { blst::blst_map_to_g2(&mut o, &bu[i % NM], core::ptr::null()) };
            black_box(o);
        }),
        side("arkworks 0.5", "WBMap::map_to_curve + clear_cofactor", |i| {
            use ark_ec::AffineRepr;
            black_box(WBMap::<ab::g2::Config>::map_to_curve(au[i % NM]).unwrap().clear_cofactor());
        }),
        side("bls12_381 0.8", "map_to_curve + clear_h", |i| {
            black_box(<bls12_381::G2Projective as MapToCurve>::map_to_curve(&zu[i % NM]).clear_h());
        }),
    ]);
    let qs: Vec<G2> = ou.iter().map(|u| s.map_to_curve(u)).collect();
    b.run("bls12_381_g2.clear_cofactor", "BLS12-381 G2, map output", "same as G2::clear_cofactor", |i| {
        black_box(s.clear_cofactor(&qs[i % NM]));
    }, vec![]);
    let _: Option<G1> = None;
}

fn bn(b: &mut Bench, msgs: &[Vec<u8>]) {
    let s = ph::bn254_g1();
    let ro = b"QUUX-V01-CS02-with-BN254G1_XMD:SHA-256_SVDW_RO_";
    let nu = b"QUUX-V01-CS02-with-BN254G1_XMD:SHA-256_SVDW_NU_";
    let p = "BN254 G1, 32-byte message, 47-byte tag";
    let note = "gnark-crypto's suite; no reference available here (Go not installed)";
    b.run("bn254_g1.hash_to_curve", p, note, |i| {
        black_box(s.hash_to_curve(&msgs[i % NM], ro));
    }, vec![]);
    b.run("bn254_g1.encode_to_curve", p, note, |i| {
        black_box(s.encode_to_curve(&msgs[i % NM], nu));
    }, vec![]);
    b.run("bn254_g1.hash_to_field", &format!("{p}, 2 elements"), note, |i| {
        black_box(s.hash_to_field(&msgs[i % NM], ro, 2));
    }, vec![]);
    let ou: Vec<_> = msgs.iter().map(|m| s.hash_to_field(m, ro, 1).remove(0)).collect();
    b.run("bn254_g1.map_to_curve", "BN254 G1, one element", "Shallue-van de Woestijne", |i| {
        black_box(s.map_to_curve(&ou[i % NM]));
    }, vec![]);
}

macro_rules! rustcrypto {
    ($b:expr, $msgs:expr, $ours:expr, $name:literal, $C:ty, $H:ty, $Fe:ty, $len:expr, $refname:literal, $ro:literal, $nu:literal) => {{
        use elliptic_curve::hash2curve::{ExpandMsgXmd, GroupDigest, MapToCurve as _};
        use elliptic_curve::sec1::ToEncodedPoint;
        let s = $ours;
        let msgs: &[Vec<u8>] = $msgs;
        let ro: &[u8] = $ro;
        let nu: &[u8] = $nu;
        let enc = |p: &a::math::ec::Point<'_, a::math::ec::Weierstrass>| {
            let (x, y) = p.coordinates().unwrap();
            (hex(&number_be(&x, $len)), hex(&number_be(&y, $len)))
        };
        let renc = |p: &elliptic_curve::ProjectivePoint<$C>| {
            let e = p.to_affine().to_encoded_point(false);
            (hex(e.x().unwrap()), hex(e.y().unwrap()))
        };
        for m in msgs.iter().take(4) {
            let r = <$C as GroupDigest>::hash_from_bytes::<ExpandMsgXmd<$H>>(&[m], &[ro]).unwrap();
            assert_eq!(renc(&r), enc(&s.hash_to_curve(m, ro)), concat!($name, " h2c"));
            let r = <$C as GroupDigest>::encode_from_bytes::<ExpandMsgXmd<$H>>(&[m], &[nu]).unwrap();
            assert_eq!(renc(&r), enc(&s.encode_to_curve(m, nu)), concat!($name, " e2c"));
        }
        let par = format!("{}, 32-byte message, {}-byte tag", $name, ro.len());
        $b.run(concat!($name, ".hash_to_curve"), &par, "", |i| {
            black_box(s.hash_to_curve(&msgs[i % NM], ro));
        }, vec![side($refname, "GroupDigest::hash_from_bytes; SHA-2 by sha2 0.10.9 (portable)", |i| {
            black_box(<$C as GroupDigest>::hash_from_bytes::<ExpandMsgXmd<$H>>(&[&msgs[i % NM]], &[ro]).unwrap());
        })]);
        $b.run(concat!($name, ".encode_to_curve"), &par, "", |i| {
            black_box(s.encode_to_curve(&msgs[i % NM], nu));
        }, vec![side($refname, "GroupDigest::encode_from_bytes; portable SHA-2", |i| {
            black_box(<$C as GroupDigest>::encode_from_bytes::<ExpandMsgXmd<$H>>(&[&msgs[i % NM]], &[nu]).unwrap());
        })]);
        $b.run(concat!($name, ".hash_to_field"), &format!("{par}, 2 elements"), "", |i| {
            black_box(s.hash_to_field(&msgs[i % NM], ro, 2));
        }, vec![side($refname, "elliptic_curve::hash2curve::hash_to_field; portable SHA-2", |i| {
            let mut out = [<$Fe>::default(); 2];
            elliptic_curve::hash2curve::hash_to_field::<ExpandMsgXmd<$H>, $Fe>(&[&msgs[i % NM]], &[ro], &mut out).unwrap();
            black_box(out);
        })]);
        let ou: Vec<_> = msgs.iter().map(|m| s.hash_to_field(m, ro, 1).remove(0)).collect();
        let ru: Vec<$Fe> = msgs
            .iter()
            .map(|m| {
                let mut out = [<$Fe>::default(); 1];
                elliptic_curve::hash2curve::hash_to_field::<ExpandMsgXmd<$H>, $Fe>(&[m], &[ro], &mut out).unwrap();
                out[0]
            })
            .collect();
        for i in 0..4 {
            assert_eq!(renc(&ru[i].map_to_curve()), enc(&s.map_to_curve(&ou[i])), concat!($name, " map"));
        }
        $b.run(concat!($name, ".map_to_curve"), &format!("{}, one element", $name), "", |i| {
            black_box(s.map_to_curve(&ou[i % NM]));
        }, vec![side($refname, "MapToCurve::map_to_curve", |i| {
            black_box(ru[i % NM].map_to_curve());
        })]);
    }};
}

fn prime(b: &mut Bench, msgs: &[Vec<u8>]) {
    use a::math::ec::hash as eh;
    rustcrypto!(b, msgs, eh::p256(), "p256", p256::NistP256, sha2_10::Sha256, <p256::NistP256 as elliptic_curve::hash2curve::GroupDigest>::FieldElement, 32, "p256 0.13.2",
        b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_RO_", b"QUUX-V01-CS02-with-P256_XMD:SHA-256_SSWU_NU_");
    rustcrypto!(b, msgs, eh::p384(), "p384", p384::NistP384, sha2_10::Sha384, <p384::NistP384 as elliptic_curve::hash2curve::GroupDigest>::FieldElement, 48, "p384 0.13.1",
        b"QUUX-V01-CS02-with-P384_XMD:SHA-384_SSWU_RO_", b"QUUX-V01-CS02-with-P384_XMD:SHA-384_SSWU_NU_");
    rustcrypto!(b, msgs, eh::p521(), "p521", p521::NistP521, sha2_10::Sha512, <p521::NistP521 as elliptic_curve::hash2curve::GroupDigest>::FieldElement, 66, "p521 0.13.3",
        b"QUUX-V01-CS02-with-P521_XMD:SHA-512_SSWU_RO_", b"QUUX-V01-CS02-with-P521_XMD:SHA-512_SSWU_NU_");
    rustcrypto!(b, msgs, eh::secp256k1(), "secp256k1", k256::Secp256k1, sha2_10::Sha256, <k256::Secp256k1 as elliptic_curve::hash2curve::GroupDigest>::FieldElement, 32, "k256 0.13.4",
        b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_", b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_NU_");

    // curve25519 (Montgomery form): no reference exposes RFC 9380's suite for it.
    let s = eh::curve25519();
    let ro = b"QUUX-V01-CS02-with-curve25519_XMD:SHA-512_ELL2_RO_";
    let nu = b"QUUX-V01-CS02-with-curve25519_XMD:SHA-512_ELL2_NU_";
    let par = format!("curve25519, 32-byte message, {}-byte tag", ro.len());
    b.run("curve25519.hash_to_curve", &par, "", |i| {
        black_box(s.hash_to_curve(&msgs[i % NM], ro));
    }, vec![]);
    b.run("curve25519.encode_to_curve", &par, "", |i| {
        black_box(s.encode_to_curve(&msgs[i % NM], nu));
    }, vec![]);
    b.run("curve25519.hash_to_field", &format!("{par}, 2 elements"), "", |i| {
        black_box(s.hash_to_field(&msgs[i % NM], ro, 2));
    }, vec![]);
    let ou: Vec<_> = msgs.iter().map(|m| s.hash_to_field(m, ro, 1).remove(0)).collect();
    b.run("curve25519.map_to_curve", "curve25519, one element", "Elligator 2", |i| {
        black_box(s.map_to_curve(&ou[i % NM]));
    }, vec![]);

    // edwards25519 against curve25519-dalek 5.0 (hash and encode are public; its map is not).
    let s = eh::edwards25519();
    let ro = b"QUUX-V01-CS02-with-edwards25519_XMD:SHA-512_ELL2_RO_";
    let nu = b"QUUX-V01-CS02-with-edwards25519_XMD:SHA-512_ELL2_NU_";
    use curve25519_dalek::edwards::EdwardsPoint;
    for m in msgs.iter().take(4) {
        let d = EdwardsPoint::hash_to_curve::<sha2_11::Sha512>(&[m], &[ro]);
        assert_eq!(d.compress().to_bytes().to_vec(), s.hash_to_curve(m, ro).compress(), "dalek h2c");
        let d = EdwardsPoint::encode_to_curve::<sha2_11::Sha512>(&[m], &[nu]);
        assert_eq!(d.compress().to_bytes().to_vec(), s.encode_to_curve(m, nu).compress(), "dalek e2c");
    }
    let par = format!("edwards25519, 32-byte message, {}-byte tag", ro.len());
    let dn = "EdwardsPoint::hash_to_curve::<sha2 0.11.1 Sha512> (ARMv8 SHA-512 instructions)";
    b.run("edwards25519.hash_to_curve", &par, "", |i| {
        black_box(s.hash_to_curve(&msgs[i % NM], ro));
    }, vec![side("curve25519-dalek 5.0.0", dn, |i| {
        black_box(EdwardsPoint::hash_to_curve::<sha2_11::Sha512>(&[&msgs[i % NM]], &[ro]));
    })]);
    b.run("edwards25519.encode_to_curve", &par, "", |i| {
        black_box(s.encode_to_curve(&msgs[i % NM], nu));
    }, vec![side("curve25519-dalek 5.0.0", "EdwardsPoint::encode_to_curve::<sha2 0.11.1 Sha512>", |i| {
        black_box(EdwardsPoint::encode_to_curve::<sha2_11::Sha512>(&[&msgs[i % NM]], &[nu]));
    })]);
    b.run("edwards25519.hash_to_field", &format!("{par}, 2 elements"), "", |i| {
        black_box(s.hash_to_field(&msgs[i % NM], ro, 2));
    }, vec![]);
    let ou: Vec<_> = msgs.iter().map(|m| s.hash_to_field(m, ro, 1).remove(0)).collect();
    b.run("edwards25519.map_to_curve", "edwards25519, one element", "Elligator 2 + rational map", |i| {
        black_box(s.map_to_curve(&ou[i % NM]));
    }, vec![]);
}
