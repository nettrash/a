#![allow(unused_must_use)]
// Pairings (a::math::pairing) on BN254 or BLS12-381 against arkworks 0.5, blst 0.3.17, the
// bls12_381 crate 0.8 (zkcrypto), mcl and substrate-bn 0.6 (BN254). One curve per process (mcl
// keeps one curve per process): `pairing bls12_381` or `pairing bn254`.
//
// Inputs: seeded random multiples of the generators, passed to every library through the same
// bytes (ZCash's encodings on BLS12-381, Ethereum's on BN254, mcl's "1 x y" strings); scalars
// below r, the same in every library. Every output is compared with ours before timing.
#[path = "../common.rs"]
mod common;
#[cfg(have_mcl)]
#[path = "../mcl.rs"]
mod mcl;

use common::*;

use a::math::numbers::{ByteOrder, Number};
use a::math::pairing::{G1, G2, Gt, PairingCurve};

const NP: usize = 16; // points in a pool
const NK: usize = 64; // scalars in a pool

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "bls12_381".into());
    match which.as_str() {
        "bls12_381" => bls::run(),
        "bn254" => bn::run(),
        w => panic!("unknown curve {w}"),
    }
}

/// Lower-case hex without leading zeros ("0" for zero), as mcl prints numbers.
fn strip(h: &str) -> String {
    let t = h.trim_start_matches('0');
    if t.is_empty() { "0".into() } else { t.to_lowercase() }
}

fn fp_hex(x: &a::math::field::Fp<'_>) -> String {
    strip(&x.value().to_str_radix(16))
}

fn gt_hex(e: &Gt<'_>) -> Vec<String> {
    e.value().coefficients().iter().map(fp_hex).collect()
}

fn ours_g1_str(p: &G1<'_>) -> String {
    match p.coordinates() {
        None => "0".into(),
        Some((x, y)) => format!("1 {} {}", fp_hex(&x), fp_hex(&y)),
    }
}

fn ours_g2_str(p: &G2<'_>) -> String {
    match p.coordinates() {
        None => "0".into(),
        Some((x, y)) => format!(
            "1 {} {} {} {}",
            fp_hex(x.c0()),
            fp_hex(x.c1()),
            fp_hex(y.c0()),
            fp_hex(y.c1())
        ),
    }
}

#[cfg(have_mcl)]
fn mcl_tokens(s: &str) -> Vec<String> {
    s.split_whitespace().map(strip).collect()
}

/// Seeded scalars below r, as 32 little-endian bytes, uniform below r.
fn scalars(seed: u64, r: &Number) -> Vec<[u8; 32]> {
    let mut rng = Rng(seed);
    let rb: [u8; 32] = number_be(r, 32).try_into().unwrap();
    let mask = 0xffu8 >> rb[0].leading_zeros();
    (0..NK)
        .map(|_| loop {
            let mut be: [u8; 32] = rng.bytes(32).try_into().unwrap();
            be[0] &= mask;
            if be < rb {
                be.reverse();
                break be;
            }
        })
        .collect()
}

fn num_le(b: &[u8; 32]) -> Number {
    Number::new(b.to_vec(), ByteOrder::LittleEndian)
}

mod bls {
    use super::*;
    use ark_bls12_381 as ab;
    use ark_ec::pairing::{MillerLoopOutput, Pairing, PairingOutput};
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, PrimeGroup};
    use ark_ff::{BigInteger, PrimeField as _};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};


    type AE = ab::Bls12_381;

    fn ark_fq_hex(x: &ab::Fq) -> String {
        strip(&hex(&x.into_bigint().to_bytes_be()))
    }
    fn ark_gt_hex(e: &ab::Fq12) -> Vec<String> {
        let mut v = Vec::new();
        for c in [e.c0, e.c1] {
            for d in [c.c0, c.c1, c.c2] {
                v.push(ark_fq_hex(&d.c0));
                v.push(ark_fq_hex(&d.c1));
            }
        }
        v
    }
    fn blst_gt_hex(e: &blst::blst_fp12) -> Vec<String> {
        let mut v = Vec::new();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..2 {
                    let mut out = [0u8; 48];
                    // SAFETY: writes 48 bytes.
                    unsafe { blst::blst_bendian_from_fp(out.as_mut_ptr(), &e.fp6[i].fp2[j].fp[k]) };
                    v.push(strip(&hex(&out)));
                }
            }
        }
        v
    }
    fn zk_gt_hex(e: &bls12_381::Gt) -> Vec<String> {
        // zkcrypto's Gt has no accessor: its Debug prints the twelve coefficients in hex.
        let s = format!("{e:?}");
        s.split(|c: char| !c.is_ascii_hexdigit() && c != 'x')
            .filter(|t| t.starts_with("0x"))
            .map(|t| strip(&t[2..]))
            .collect()
    }

    fn ser_c<T: CanonicalSerialize>(p: &T) -> Vec<u8> {
        let mut b = Vec::new();
        p.serialize_compressed(&mut b).unwrap();
        b
    }
    fn ser_u<T: CanonicalSerialize>(p: &T) -> Vec<u8> {
        let mut b = Vec::new();
        p.serialize_uncompressed(&mut b).unwrap();
        b
    }

    pub fn run() {
        let c = PairingCurve::bls12_381();
        let mut b = Bench::new("pairing");
        let curve = "BLS12-381";
        let ks = scalars(11, c.order());
        let mut rng = Rng(7);

        // Points: [a]g1, [b]g2 by arkworks, then everybody decodes the same bytes.
        let ag1: Vec<ab::G1Affine> = (0..NP)
            .map(|_| (ab::G1Projective::generator() * ab::Fr::from_le_bytes_mod_order(&rng.bytes(32))).into_affine())
            .collect();
        let ag2: Vec<ab::G2Affine> = (0..NP)
            .map(|_| (ab::G2Projective::generator() * ab::Fr::from_le_bytes_mod_order(&rng.bytes(32))).into_affine())
            .collect();
        let g1u: Vec<Vec<u8>> = ag1.iter().map(ser_u).collect();
        let g2u: Vec<Vec<u8>> = ag2.iter().map(ser_u).collect();
        let g1c: Vec<Vec<u8>> = ag1.iter().map(ser_c).collect();
        let g2c: Vec<Vec<u8>> = ag2.iter().map(ser_c).collect();

        let ps: Vec<G1> = g1u.iter().map(|x| G1::from_zcash(c, x).unwrap()).collect();
        let qs: Vec<G2> = g2u.iter().map(|x| G2::from_zcash(c, x).unwrap()).collect();
        let ok: Vec<Number> = ks.iter().map(num_le).collect();
        let afr: Vec<ab::Fr> = ks.iter().map(|k| ab::Fr::from_le_bytes_mod_order(k)).collect();
        let zfr: Vec<bls12_381::Scalar> = ks.iter().map(|k| bls12_381::Scalar::from_bytes(k).unwrap()).collect();
        let ap1: Vec<ab::G1Projective> = ag1.iter().map(|p| p.into_group()).collect();
        let ap2: Vec<ab::G2Projective> = ag2.iter().map(|p| p.into_group()).collect();
        let zp: Vec<bls12_381::G1Affine> =
            g1u.iter().map(|x| bls12_381::G1Affine::from_uncompressed(x.as_slice().try_into().unwrap()).unwrap()).collect();
        let zq: Vec<bls12_381::G2Affine> =
            g2u.iter().map(|x| bls12_381::G2Affine::from_uncompressed(x.as_slice().try_into().unwrap()).unwrap()).collect();
        let zpp: Vec<bls12_381::G1Projective> = zp.iter().map(|p| p.into()).collect();
        let zqp: Vec<bls12_381::G2Projective> = zq.iter().map(|p| p.into()).collect();
        let zqprep: Vec<bls12_381::G2Prepared> = zq.iter().map(|q| bls12_381::G2Prepared::from(*q)).collect();
        // blst
        let bp: Vec<blst::blst_p1_affine> = g1u
            .iter()
            .map(|x| {
                let mut a = blst::blst_p1_affine::default();
                // SAFETY: reads 96 bytes.
                assert_eq!(unsafe { blst::blst_p1_deserialize(&mut a, x.as_ptr()) }, blst::BLST_ERROR::BLST_SUCCESS);
                a
            })
            .collect();
        let bq: Vec<blst::blst_p2_affine> = g2u
            .iter()
            .map(|x| {
                let mut a = blst::blst_p2_affine::default();
                // SAFETY: reads 192 bytes.
                assert_eq!(unsafe { blst::blst_p2_deserialize(&mut a, x.as_ptr()) }, blst::BLST_ERROR::BLST_SUCCESS);
                a
            })
            .collect();
        let bpj: Vec<blst::blst_p1> = bp
            .iter()
            .map(|a| {
                let mut p = blst::blst_p1::default();
                // SAFETY: blst on its own types.
                unsafe { blst::blst_p1_from_affine(&mut p, a) };
                p
            })
            .collect();
        let bqj: Vec<blst::blst_p2> = bq
            .iter()
            .map(|a| {
                let mut p = blst::blst_p2::default();
                // SAFETY: blst on its own types.
                unsafe { blst::blst_p2_from_affine(&mut p, a) };
                p
            })
            .collect();
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
        let (mp, mq, mfr) = {
            mcl::init(mcl::BLS12_381);
            // SAFETY: mcl's settings, after init.
            unsafe {
                mcl::mclBn_setETHserialization(1);
                mcl::mclBn_verifyOrderG1(0);
                mcl::mclBn_verifyOrderG2(0);
            }
            let mp: Vec<mcl::G1> = ps.iter().map(|p| mcl::g1_from_str(&ours_g1_str(p))).collect();
            let mq: Vec<mcl::G2> = qs.iter().map(|q| mcl::g2_from_str(&ours_g2_str(q))).collect();
            let mfr: Vec<mcl::Fr> = ks.iter().map(|k| mcl::fr_le(k)).collect();
            (mp, mq, mfr)
        };
        #[cfg(have_mcl)]
        let mcl_name = format!("mcl {}", mcl::version());

        // ---------------- the pairing ----------------
        let par = format!("{curve}, 1 pair");
        // Checks: all equal ours.
        for i in 0..4 {
            let e = c.pairing(&ps[i], &qs[i]);
            let h = gt_hex(&e);
            assert_eq!(ark_gt_hex(&AE::pairing(ag1[i], ag2[i]).0), h, "ark pairing");
            assert_eq!(zk_gt_hex(&bls12_381::pairing(&zp[i], &zq[i])), h, "zk pairing");
            let mut f = blst::blst_fp12::default();
            let mut g = blst::blst_fp12::default();
            // SAFETY: blst on its own types.
            unsafe {
                blst::blst_miller_loop(&mut f, &bq[i], &bp[i]);
                blst::blst_final_exp(&mut g, &f);
            }
            assert_eq!(blst_gt_hex(&g), h, "blst pairing");
            #[cfg(have_mcl)]
            {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_pairing(&mut g, &mp[i], &mq[i]) };
                assert_eq!(mcl_tokens(&mcl::gt_str(&g)), h, "mcl pairing");
            }
        }
        {
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(AE::pairing(ag1[i % NP], ag2[(i / NP) % NP]));
                }),
                side("blst 0.3.17", "", |i| {
                    let mut f = blst::blst_fp12::default();
                    let mut g = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe {
                        blst::blst_miller_loop(&mut f, &bq[(i / NP) % NP], &bp[i % NP]);
                        blst::blst_final_exp(&mut g, &f);
                    }
                    black_box(g);
                }),
                side("bls12_381 0.8", "", |i| {
                    black_box(bls12_381::pairing(&zp[i % NP], &zq[(i / NP) % NP]));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_pairing(&mut g, &mp[i % NP], &mq[(i / NP) % NP]) };
                black_box(g);
            }));
            b.run("pairing", &par, "", |i| {
                black_box(c.pairing(&ps[i % NP], &qs[(i / NP) % NP]));
            }, refs);
        }

        // ---------------- Miller loop ----------------
        {
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(AE::multi_miller_loop([ag1[i % NP]], [ag2[(i / NP) % NP]]));
                }),
                side("blst 0.3.17", "", |i| {
                    let mut f = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_miller_loop(&mut f, &bq[(i / NP) % NP], &bp[i % NP]) };
                    black_box(f);
                }),
                side("bls12_381 0.8", "G2Prepared::from(Q) included: its loop takes precomputed lines", |i| {
                    let q = bls12_381::G2Prepared::from(zq[(i / NP) % NP]);
                    black_box(bls12_381::multi_miller_loop(&[(&zp[i % NP], &q)]));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_millerLoop(&mut g, &mp[i % NP], &mq[(i / NP) % NP]) };
                black_box(g);
            }));
            b.run(
                "miller_loop",
                &par,
                "outputs differ between libraries by factors the final exponentiation removes; checked through the pairing",
                |i| {
                    black_box(c.miller_loop(&[(&ps[i % NP], &qs[(i / NP) % NP])]));
                },
                refs,
            );
        }
        // With Q's lines precomputed once (fixed Q, as in verifying many signatures under a key).
        {
            let alines: Vec<<AE as Pairing>::G2Prepared> = ag2.iter().map(|q| (*q).into()).collect();
            let blines: Vec<Vec<blst::blst_fp6>> = bq
                .iter()
                .map(|q| {
                    let mut l = vec![blst::blst_fp6::default(); 68];
                    // SAFETY: blst writes 68 lines.
                    unsafe { blst::blst_precompute_lines(l.as_mut_ptr(), q) };
                    l
                })
                .collect();
            let refs = vec![
                side("arkworks 0.5", "G2Prepared reused (cloned per call)", |i| {
                    black_box(AE::multi_miller_loop([ag1[i % NP]], [alines[(i / NP) % NP].clone()]));
                }),
                side("blst 0.3.17", "blst_precompute_lines once, blst_miller_loop_lines per call", |i| {
                    let mut f = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_miller_loop_lines(&mut f, blines[(i / NP) % NP].as_ptr(), &bp[i % NP]) };
                    black_box(f);
                }),
                side("bls12_381 0.8", "G2Prepared reused", |i| {
                    black_box(bls12_381::multi_miller_loop(&[(&zp[i % NP], &zqprep[(i / NP) % NP])]));
                }),
            ];
            b.run(
                "miller_loop_fixed_q",
                &par,
                "ours has no precomputed lines: our plain Miller loop",
                |i| {
                    black_box(c.miller_loop(&[(&ps[i % NP], &qs[(i / NP) % NP])]));
                },
                refs,
            );
        }

        // ---------------- final exponentiation ----------------
        {
            let of: Vec<_> = (0..NP).map(|i| c.miller_loop(&[(&ps[i], &qs[(i + 3) % NP])])).collect();
            let af: Vec<_> = (0..NP).map(|i| AE::multi_miller_loop([ag1[i]], [ag2[(i + 3) % NP]])).collect();
            let bf: Vec<blst::blst_fp12> = (0..NP)
                .map(|i| {
                    let mut f = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_miller_loop(&mut f, &bq[(i + 3) % NP], &bp[i]) };
                    f
                })
                .collect();
            let zf: Vec<bls12_381::MillerLoopResult> =
                (0..NP).map(|i| bls12_381::multi_miller_loop(&[(&zp[i], &zqprep[(i + 3) % NP])])).collect();
            #[cfg(have_mcl)]
            let mf: Vec<mcl::GT> = (0..NP)
                .map(|i| {
                    let mut g = mcl::GT::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBn_millerLoop(&mut g, &mp[i], &mq[(i + 3) % NP]) };
                    g
                })
                .collect();
            for i in 0..2 {
                let h = gt_hex(&c.final_exponentiation(&of[i]));
                assert_eq!(ark_gt_hex(&AE::final_exponentiation(af[i]).unwrap().0), h);
            }
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(AE::final_exponentiation(MillerLoopOutput(af[i % NP].0)));
                }),
                side("blst 0.3.17", "", |i| {
                    let mut g = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_final_exp(&mut g, &bf[i % NP]) };
                    black_box(g);
                }),
                side("bls12_381 0.8", "", |i| {
                    black_box(zf[i % NP].final_exponentiation());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_finalExp(&mut g, &mf[i % NP]) };
                black_box(g);
            }));
            b.run("final_exponentiation", curve, "", |i| {
                black_box(c.final_exponentiation(&of[i % NP]));
            }, refs);
        }

        // ---------------- multi-pairing ----------------
        for n in [2usize, 4, 8, 16] {
            let par = format!("{curve}, {n} pairs");
            let pick = |i: usize, j: usize| (i + j * 5) % NP;
            {
                let pairs: Vec<(&G1, &G2)> = (0..n).map(|j| (&ps[pick(0, j)], &qs[pick(1, j)])).collect();
                let h = gt_hex(&c.multi_pairing(&pairs));
                let a = AE::multi_pairing((0..n).map(|j| ag1[pick(0, j)]), (0..n).map(|j| ag2[pick(1, j)]));
                assert_eq!(ark_gt_hex(&a.0), h, "ark multi");
            }
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(AE::multi_pairing((0..n).map(|j| ag1[pick(i, j)]), (0..n).map(|j| ag2[pick(i + 1, j)])));
                }),
                side("blst 0.3.17", "blst_miller_loop_n + blst_final_exp", |i| {
                    let pp: Vec<*const blst::blst_p1_affine> = (0..n).map(|j| &bp[pick(i, j)] as *const _).collect();
                    let qq: Vec<*const blst::blst_p2_affine> = (0..n).map(|j| &bq[pick(i + 1, j)] as *const _).collect();
                    let mut f = blst::blst_fp12::default();
                    let mut g = blst::blst_fp12::default();
                    // SAFETY: blst on arrays of n valid pointers.
                    unsafe {
                        blst::blst_miller_loop_n(&mut f, qq.as_ptr(), pp.as_ptr(), n);
                        blst::blst_final_exp(&mut g, &f);
                    }
                    black_box(g);
                }),
                side("bls12_381 0.8", "G2Prepared::from per pair included", |i| {
                    let preps: Vec<bls12_381::G2Prepared> =
                        (0..n).map(|j| bls12_381::G2Prepared::from(zq[pick(i + 1, j)])).collect();
                    let terms: Vec<(&bls12_381::G1Affine, &bls12_381::G2Prepared)> =
                        (0..n).map(|j| (&zp[pick(i, j)], &preps[j])).collect();
                    black_box(bls12_381::multi_miller_loop(&terms).final_exponentiation());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBn_millerLoopVec + mclBn_finalExp", |i| {
                let pp: Vec<mcl::G1> = (0..n).map(|j| mp[pick(i, j)]).collect();
                let qq: Vec<mcl::G2> = (0..n).map(|j| mq[pick(i + 1, j)]).collect();
                let mut f = mcl::GT::default();
                let mut g = mcl::GT::default();
                // SAFETY: mcl on arrays of n elements.
                unsafe {
                    mcl::mclBn_millerLoopVec(&mut f, pp.as_ptr(), qq.as_ptr(), n);
                    mcl::mclBn_finalExp(&mut g, &f);
                }
                black_box(g);
            }));
            b.run("multi_pairing", &par, "", |i| {
                let pairs: Vec<(&G1, &G2)> = (0..n).map(|j| (&ps[pick(i, j)], &qs[pick(i + 1, j)])).collect();
                black_box(c.multi_pairing(&pairs));
            }, refs);
        }

        // ---------------- scalar multiplication ----------------
        {
            // Checks.
            for i in 0..4 {
                let r = &ps[i] * &ok[i];
                let want = r.to_zcash_compressed();
                assert_eq!(ser_c(&(ap1[i] * afr[i])), want, "ark g1 mul");
                assert_eq!(bls12_381::G1Affine::from(zpp[i] * zfr[i]).to_compressed().to_vec(), want, "zk g1 mul");
                let mut o = blst::blst_p1::default();
                // SAFETY: blst on its own types, 32-byte scalar.
                unsafe { blst::blst_p1_mult(&mut o, &bpj[i], ks[i].as_ptr(), 255) };
                assert_eq!(blst_c1(&o), want, "blst g1 mul");
                // SAFETY: as above.
                unsafe { blst::blst_p1_unchecked_mult(&mut o, &bpj[i], ks[i].as_ptr(), 255) };
                assert_eq!(blst_c1(&o), want, "blst g1 unchecked mul");
                #[cfg(have_mcl)]
                {
                    let mut o = mcl::G1::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnG1_mul(&mut o, &mp[i], &mfr[i]) };
                    assert_eq!(mcl_tokens(&mcl::g1_str(&o)), mcl_tokens(&ours_g1_str(&r)), "mcl g1 mul");
                }
                let r2 = &qs[i] * &ok[i];
                let want2 = r2.to_zcash_compressed();
                assert_eq!(ser_c(&(ap2[i] * afr[i])), want2, "ark g2 mul");
                assert_eq!(bls12_381::G2Affine::from(zqp[i] * zfr[i]).to_compressed().to_vec(), want2, "zk g2 mul");
                let mut o2 = blst::blst_p2::default();
                // SAFETY: as above.
                unsafe { blst::blst_p2_mult(&mut o2, &bqj[i], ks[i].as_ptr(), 255) };
                assert_eq!(blst_c2(&o2), want2, "blst g2 mul");
                #[cfg(have_mcl)]
                {
                    let mut o = mcl::G2::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnG2_mul(&mut o, &mq[i], &mfr[i]) };
                    assert_eq!(mcl_tokens(&mcl::g2_str(&o)), mcl_tokens(&ours_g2_str(&r2)), "mcl g2 mul");
                }
            }
            let par = format!("{curve}, 255-bit scalar < r, affine input");
            let mut refs = vec![
                side("arkworks 0.5", "GLV", |i| {
                    black_box(ap1[i % NP] * afr[i % NK]);
                }),
                side("blst 0.3.17", "blst_p1_mult (constant-time)", |i| {
                    let mut o = blst::blst_p1::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p1_mult(&mut o, &bpj[i % NP], ks[i % NK].as_ptr(), 255) };
                    black_box(o);
                }),
                side("blst 0.3.17 vartime", "blst_p1_unchecked_mult (variable-time, width-4 window)", |i| {
                    let mut o = blst::blst_p1::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p1_unchecked_mult(&mut o, &bpj[i % NP], ks[i % NK].as_ptr(), 255) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "constant-time double-and-add", |i| {
                    black_box(zpp[i % NP] * zfr[i % NK]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG1_mul (GLV, variable-time)", |i| {
                let mut o = mcl::G1::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG1_mul(&mut o, &mp[i % NP], &mfr[i % NK]) };
                black_box(o);
            }));
            b.run("g1_mul", &par, "", |i| {
                black_box(&ps[i % NP] * &ok[i % NK]);
            }, refs);

            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ap2[i % NP] * afr[i % NK]);
                }),
                side("blst 0.3.17", "blst_p2_mult (constant-time)", |i| {
                    let mut o = blst::blst_p2::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p2_mult(&mut o, &bqj[i % NP], ks[i % NK].as_ptr(), 255) };
                    black_box(o);
                }),
                side("blst 0.3.17 vartime", "blst_p2_unchecked_mult (variable-time, width-4 window)", |i| {
                    let mut o = blst::blst_p2::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p2_unchecked_mult(&mut o, &bqj[i % NP], ks[i % NK].as_ptr(), 255) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "constant-time double-and-add", |i| {
                    black_box(zqp[i % NP] * zfr[i % NK]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG2_mul (GLS, variable-time)", |i| {
                let mut o = mcl::G2::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG2_mul(&mut o, &mq[i % NP], &mfr[i % NK]) };
                black_box(o);
            }));
            b.run("g2_mul", &par, "", |i| {
                black_box(&qs[i % NP] * &ok[i % NK]);
            }, refs);
        }

        // Projective (non-normalised) points for the additions, coordinates and encodings: the
        // products of the pool by the scalars, each library's own.
        let rs1: Vec<G1> = (0..NP).map(|i| &ps[i] * &ok[i]).collect();
        let rs2: Vec<G2> = (0..NP).map(|i| &qs[i] * &ok[i]).collect();
        let ar1: Vec<ab::G1Projective> = (0..NP).map(|i| ap1[i] * afr[i]).collect();
        let ar2: Vec<ab::G2Projective> = (0..NP).map(|i| ap2[i] * afr[i]).collect();
        let zr1: Vec<bls12_381::G1Projective> = (0..NP).map(|i| zpp[i] * zfr[i]).collect();
        let zr2: Vec<bls12_381::G2Projective> = (0..NP).map(|i| zqp[i] * zfr[i]).collect();
        let br1: Vec<blst::blst_p1> = (0..NP)
            .map(|i| {
                let mut o = blst::blst_p1::default();
                // SAFETY: blst on its own types.
                unsafe { blst::blst_p1_mult(&mut o, &bpj[i], ks[i].as_ptr(), 255) };
                o
            })
            .collect();
        let br2: Vec<blst::blst_p2> = (0..NP)
            .map(|i| {
                let mut o = blst::blst_p2::default();
                // SAFETY: blst on its own types.
                unsafe { blst::blst_p2_mult(&mut o, &bqj[i], ks[i].as_ptr(), 255) };
                o
            })
            .collect();
        #[cfg(have_mcl)]
        let (mr1, mr2): (Vec<mcl::G1>, Vec<mcl::G2>) = (
            (0..NP)
                .map(|i| {
                    let mut o = mcl::G1::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnG1_mul(&mut o, &mp[i], &mfr[i]) };
                    o
                })
                .collect(),
            (0..NP)
                .map(|i| {
                    let mut o = mcl::G2::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnG2_mul(&mut o, &mq[i], &mfr[i]) };
                    o
                })
                .collect(),
        );

        // ---------------- additions and doublings ----------------
        {
            let par = format!("{curve}, projective inputs");
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ar1[i % NP] + ar1[(i / NP + i + 1) % NP]);
                }),
                side("blst 0.3.17", "blst_p1_add_or_double", |i| {
                    let mut o = blst::blst_p1::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p1_add_or_double(&mut o, &br1[i % NP], &br1[(i / NP + i + 1) % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "complete formulas", |i| {
                    black_box(zr1[i % NP] + zr1[(i / NP + i + 1) % NP]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut o = mcl::G1::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG1_add(&mut o, &mr1[i % NP], &mr1[(i / NP + i + 1) % NP]) };
                black_box(o);
            }));
            b.run("g1_add", &par, "", |i| {
                black_box(&rs1[i % NP] + &rs1[(i / NP + i + 1) % NP]);
            }, refs);

            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ar1[i % NP].double());
                }),
                side("blst 0.3.17", "", |i| {
                    let mut o = blst::blst_p1::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p1_double(&mut o, &br1[i % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "", |i| {
                    black_box(zr1[i % NP].double());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut o = mcl::G1::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG1_dbl(&mut o, &mr1[i % NP]) };
                black_box(o);
            }));
            b.run("g1_double", &par, "", |i| {
                black_box(rs1[i % NP].double());
            }, refs);

            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ar2[i % NP] + ar2[(i / NP + i + 1) % NP]);
                }),
                side("blst 0.3.17", "blst_p2_add_or_double", |i| {
                    let mut o = blst::blst_p2::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p2_add_or_double(&mut o, &br2[i % NP], &br2[(i / NP + i + 1) % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "complete formulas", |i| {
                    black_box(zr2[i % NP] + zr2[(i / NP + i + 1) % NP]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut o = mcl::G2::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG2_add(&mut o, &mr2[i % NP], &mr2[(i / NP + i + 1) % NP]) };
                black_box(o);
            }));
            b.run("g2_add", &par, "", |i| {
                black_box(&rs2[i % NP] + &rs2[(i / NP + i + 1) % NP]);
            }, refs);

            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ar2[i % NP].double());
                }),
                side("blst 0.3.17", "", |i| {
                    let mut o = blst::blst_p2::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p2_double(&mut o, &br2[i % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "", |i| {
                    black_box(zr2[i % NP].double());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut o = mcl::G2::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG2_dbl(&mut o, &mr2[i % NP]) };
                black_box(o);
            }));
            b.run("g2_double", &par, "", |i| {
                black_box(rs2[i % NP].double());
            }, refs);

            // Affine coordinates of a projective point (one inversion).
            let mut refs = vec![
                side("arkworks 0.5", "into_affine", |i| {
                    black_box(ar1[i % NP].into_affine());
                }),
                side("blst 0.3.17", "blst_p1_to_affine", |i| {
                    let mut o = blst::blst_p1_affine::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p1_to_affine(&mut o, &br1[i % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "G1Affine::from", |i| {
                    black_box(bls12_381::G1Affine::from(zr1[i % NP]));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG1_normalize", |i| {
                let mut o = mcl::G1::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG1_normalize(&mut o, &mr1[i % NP]) };
                black_box(o);
            }));
            b.run("g1_coordinates", &par, "", |i| {
                black_box(rs1[i % NP].coordinates());
            }, refs);
            let mut refs = vec![
                side("arkworks 0.5", "into_affine", |i| {
                    black_box(ar2[i % NP].into_affine());
                }),
                side("blst 0.3.17", "blst_p2_to_affine", |i| {
                    let mut o = blst::blst_p2_affine::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_p2_to_affine(&mut o, &br2[i % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "G2Affine::from", |i| {
                    black_box(bls12_381::G2Affine::from(zr2[i % NP]));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG2_normalize", |i| {
                let mut o = mcl::G2::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG2_normalize(&mut o, &mr2[i % NP]) };
                black_box(o);
            }));
            b.run("g2_coordinates", &par, "", |i| {
                black_box(rs2[i % NP].coordinates());
            }, refs);
        }

        // ---------------- subgroup checks ----------------
        {
            let par = format!("{curve}, points of G1 (affine)");
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ag1[i % NP].is_in_correct_subgroup_assuming_on_curve());
                }),
                side("blst 0.3.17", "blst_p1_affine_in_g1", |i| {
                    // SAFETY: blst on its own types.
                    black_box(unsafe { blst::blst_p1_affine_in_g1(&bp[i % NP]) });
                }),
                side("bls12_381 0.8", "is_torsion_free", |i| {
                    black_box(bool::from(zp[i % NP].is_torsion_free()));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG1_isValidOrder", |i| {
                // SAFETY: mcl on its own types.
                black_box(unsafe { mcl::mclBnG1_isValidOrder(&mp[i % NP]) });
            }));
            b.run("g1_is_in_subgroup", &par, "", |i| {
                black_box(ps[i % NP].is_in_subgroup());
            }, refs);
            let par = format!("{curve}, points of G2 (affine)");
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ag2[i % NP].is_in_correct_subgroup_assuming_on_curve());
                }),
                side("blst 0.3.17", "blst_p2_affine_in_g2", |i| {
                    // SAFETY: blst on its own types.
                    black_box(unsafe { blst::blst_p2_affine_in_g2(&bq[i % NP]) });
                }),
                side("bls12_381 0.8", "is_torsion_free", |i| {
                    black_box(bool::from(zq[i % NP].is_torsion_free()));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG2_isValidOrder", |i| {
                // SAFETY: mcl on its own types.
                black_box(unsafe { mcl::mclBnG2_isValidOrder(&mq[i % NP]) });
            }));
            b.run("g2_is_in_subgroup", &par, "", |i| {
                black_box(qs[i % NP].is_in_subgroup());
            }, refs);
        }

        // ---------------- cofactors (points of the curve outside the subgroup) ----------------
        {
            let s1 = a::math::pairing::hash::bls12_381_g1();
            let s2 = a::math::pairing::hash::bls12_381_g2();
            let u1 = s1.hash_to_field(b"cofactor points", b"BENCH-DST", NP);
            let u2 = s2.hash_to_field(b"cofactor points", b"BENCH-DST", NP);
            let os1: Vec<G1> = u1.iter().map(|u| s1.map_to_curve(u)).collect();
            let os2: Vec<G2> = u2.iter().map(|u| s2.map_to_curve(u)).collect();
            let oa1: Vec<ab::G1Affine> =
                os1.iter().map(|p| ab::G1Affine::deserialize_uncompressed_unchecked(&p.to_zcash_uncompressed()[..]).unwrap()).collect();
            let oa2: Vec<ab::G2Affine> =
                os2.iter().map(|p| ab::G2Affine::deserialize_uncompressed_unchecked(&p.to_zcash_uncompressed()[..]).unwrap()).collect();
            let oz1: Vec<bls12_381::G1Projective> = os1
                .iter()
                .map(|p| bls12_381::G1Affine::from_uncompressed_unchecked(p.to_zcash_uncompressed().as_slice().try_into().unwrap()).unwrap().into())
                .collect();
            let oz2: Vec<bls12_381::G2Projective> = os2
                .iter()
                .map(|p| bls12_381::G2Affine::from_uncompressed_unchecked(p.to_zcash_uncompressed().as_slice().try_into().unwrap()).unwrap().into())
                .collect();
            for i in 0..4 {
                assert!(!os1[i].is_in_subgroup() && !os2[i].is_in_subgroup());
                let w1 = os1[i].clear_cofactor().to_zcash_compressed();
                assert_eq!(ser_c(&oa1[i].clear_cofactor()), w1, "ark clear g1");
                assert_eq!(bls12_381::G1Affine::from(oz1[i].clear_cofactor()).to_compressed().to_vec(), w1, "zk clear g1");
                let w2 = os2[i].clear_cofactor().to_zcash_compressed();
                assert_eq!(ser_c(&oa2[i].clear_cofactor()), w2, "ark clear g2");
                assert_eq!(bls12_381::G2Affine::from(oz2[i].clear_cofactor()).to_compressed().to_vec(), w2, "zk clear g2");
                assert_eq!(ser_c(&oa1[i].mul_by_cofactor()), os1[i].mul_by_cofactor().to_zcash_compressed(), "ark mul_by_cofactor g1");
                assert_eq!(ser_c(&oa2[i].mul_by_cofactor()), os2[i].mul_by_cofactor().to_zcash_compressed(), "ark mul_by_cofactor g2");
            }
            let par = format!("{curve}, curve points outside G1");
            b.run("g1_clear_cofactor", &par, "[1 - x]P", |i| {
                black_box(os1[i % NP].clear_cofactor());
            }, vec![
                side("arkworks 0.5", "affine in and out", |i| {
                    black_box(oa1[i % NP].clear_cofactor());
                }),
                side("bls12_381 0.8", "", |i| {
                    black_box(oz1[i % NP].clear_cofactor());
                }),
            ]);
            b.run("g1_mul_by_cofactor", &par, "126-bit cofactor", |i| {
                black_box(os1[i % NP].mul_by_cofactor());
            }, vec![side("arkworks 0.5", "", |i| {
                black_box(oa1[i % NP].mul_by_cofactor());
            })]);
            let par = format!("{curve}, twist points outside G2");
            b.run("g2_clear_cofactor", &par, "Budroni-Pintore", |i| {
                black_box(os2[i % NP].clear_cofactor());
            }, vec![
                side("arkworks 0.5", "Budroni-Pintore, affine in and out", |i| {
                    black_box(oa2[i % NP].clear_cofactor());
                }),
                side("bls12_381 0.8", "Budroni-Pintore", |i| {
                    black_box(oz2[i % NP].clear_cofactor());
                }),
            ]);
            b.run("g2_mul_by_cofactor", &par, "507-bit cofactor", |i| {
                black_box(os2[i % NP].mul_by_cofactor());
            }, vec![side("arkworks 0.5", "", |i| {
                black_box(oa2[i % NP].mul_by_cofactor());
            })]);
            b.run("g2_psi", &format!("{curve}, projective input"), "", |i| {
                black_box(rs2[i % NP].psi());
            }, vec![]);
        }

        // ---------------- encodings ----------------
        {
            for i in 0..4 {
                let w = rs1[i].to_zcash_compressed();
                assert_eq!(ser_c(&ar1[i]), w);
                assert_eq!(blst_c1(&br1[i]), w);
                assert_eq!(bls12_381::G1Affine::from(zr1[i]).to_compressed().to_vec(), w);
                #[cfg(have_mcl)]
                {
                    let mut o = [0u8; 48];
                    // SAFETY: writes at most 48 bytes.
                    let n = unsafe { mcl::mclBnG1_serialize(o.as_mut_ptr(), 48, &mr1[i]) };
                    assert_eq!((n, o.to_vec()), (48, w.clone()), "mcl g1 serialize");
                }
                let w2 = rs2[i].to_zcash_compressed();
                assert_eq!(ser_c(&ar2[i]), w2);
                assert_eq!(blst_c2(&br2[i]), w2);
                #[cfg(have_mcl)]
                {
                    let mut o = [0u8; 96];
                    // SAFETY: writes at most 96 bytes.
                    let n = unsafe { mcl::mclBnG2_serialize(o.as_mut_ptr(), 96, &mr2[i]) };
                    assert_eq!((n, o.to_vec()), (96, w2.clone()), "mcl g2 serialize");
                }
                assert_eq!(ser_u(&ar1[i]), rs1[i].to_zcash_uncompressed());
                assert_eq!(ser_u(&ar2[i]), rs2[i].to_zcash_uncompressed());
            }
            let par = format!("{curve}, from a projective point");
            let mut refs = vec![
                side("arkworks 0.5", "into_affine + serialize_compressed", |i| {
                    black_box(ser_c(&ar1[i % NP]));
                }),
                side("blst 0.3.17", "blst_p1_compress", |i| {
                    black_box(blst_c1(&br1[i % NP]));
                }),
                side("bls12_381 0.8", "G1Affine::from + to_compressed", |i| {
                    black_box(bls12_381::G1Affine::from(zr1[i % NP]).to_compressed());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG1_serialize (ETH mode = ZCash's)", |i| {
                let mut o = [0u8; 48];
                // SAFETY: writes at most 48 bytes.
                black_box(unsafe { mcl::mclBnG1_serialize(o.as_mut_ptr(), 48, &mr1[i % NP]) });
                black_box(o);
            }));
            b.run("g1_to_zcash_compressed", &par, "", |i| {
                black_box(rs1[i % NP].to_zcash_compressed());
            }, refs);
            let refs = vec![
                side("arkworks 0.5", "into_affine + serialize_uncompressed", |i| {
                    black_box(ser_u(&ar1[i % NP]));
                }),
                side("blst 0.3.17", "blst_p1_serialize", |i| {
                    let mut o = [0u8; 96];
                    // SAFETY: writes 96 bytes.
                    unsafe { blst::blst_p1_serialize(o.as_mut_ptr(), &br1[i % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "G1Affine::from + to_uncompressed", |i| {
                    black_box(bls12_381::G1Affine::from(zr1[i % NP]).to_uncompressed());
                }),
            ];
            b.run("g1_to_zcash_uncompressed", &par, "", |i| {
                black_box(rs1[i % NP].to_zcash_uncompressed());
            }, refs);
            let mut refs = vec![
                side("arkworks 0.5", "into_affine + serialize_compressed", |i| {
                    black_box(ser_c(&ar2[i % NP]));
                }),
                side("blst 0.3.17", "blst_p2_compress", |i| {
                    black_box(blst_c2(&br2[i % NP]));
                }),
                side("bls12_381 0.8", "G2Affine::from + to_compressed", |i| {
                    black_box(bls12_381::G2Affine::from(zr2[i % NP]).to_compressed());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG2_serialize (ETH mode = ZCash's)", |i| {
                let mut o = [0u8; 96];
                // SAFETY: writes at most 96 bytes.
                black_box(unsafe { mcl::mclBnG2_serialize(o.as_mut_ptr(), 96, &mr2[i % NP]) });
                black_box(o);
            }));
            b.run("g2_to_zcash_compressed", &par, "", |i| {
                black_box(rs2[i % NP].to_zcash_compressed());
            }, refs);
            let refs = vec![
                side("arkworks 0.5", "into_affine + serialize_uncompressed", |i| {
                    black_box(ser_u(&ar2[i % NP]));
                }),
                side("blst 0.3.17", "blst_p2_serialize", |i| {
                    let mut o = [0u8; 192];
                    // SAFETY: writes 192 bytes.
                    unsafe { blst::blst_p2_serialize(o.as_mut_ptr(), &br2[i % NP]) };
                    black_box(o);
                }),
                side("bls12_381 0.8", "G2Affine::from + to_uncompressed", |i| {
                    black_box(bls12_381::G2Affine::from(zr2[i % NP]).to_uncompressed());
                }),
            ];
            b.run("g2_to_zcash_uncompressed", &par, "", |i| {
                black_box(rs2[i % NP].to_zcash_uncompressed());
            }, refs);

            // Decoding, checked: in the field, on the curve, in the subgroup.
            let par = format!("{curve}, with the curve and subgroup checks");
            let mut refs = vec![
                side("arkworks 0.5", "deserialize_compressed (validates)", |i| {
                    black_box(ab::G1Affine::deserialize_compressed(&g1c[i % NP][..]).unwrap());
                }),
                side("blst 0.3.17", "blst_p1_uncompress + blst_p1_affine_in_g1", |i| {
                    let mut a = blst::blst_p1_affine::default();
                    // SAFETY: reads 48 bytes.
                    let e = unsafe { blst::blst_p1_uncompress(&mut a, g1c[i % NP].as_ptr()) };
                    // SAFETY: blst on its own types.
                    assert!(e == blst::BLST_ERROR::BLST_SUCCESS && unsafe { blst::blst_p1_affine_in_g1(&a) });
                    black_box(a);
                }),
                side("bls12_381 0.8", "from_compressed", |i| {
                    black_box(bls12_381::G1Affine::from_compressed(g1c[i % NP].as_slice().try_into().unwrap()).unwrap());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG1_deserialize + mclBnG1_isValidOrder", |i| {
                let mut p = mcl::G1::default();
                // SAFETY: reads 48 bytes; mcl on its own types.
                let n = unsafe { mcl::mclBnG1_deserialize(&mut p, g1c[i % NP].as_ptr(), 48) };
                // SAFETY: as above.
                assert!(n == 48 && unsafe { mcl::mclBnG1_isValidOrder(&p) } == 1);
                black_box(p);
            }));
            b.run("g1_from_zcash_compressed", &par, "", |i| {
                black_box(G1::from_zcash(c, &g1c[i % NP]).unwrap());
            }, refs);
            let refs = vec![
                side("arkworks 0.5", "deserialize_uncompressed (validates)", |i| {
                    black_box(ab::G1Affine::deserialize_uncompressed(&g1u[i % NP][..]).unwrap());
                }),
                side("blst 0.3.17", "blst_p1_deserialize + blst_p1_affine_in_g1", |i| {
                    let mut a = blst::blst_p1_affine::default();
                    // SAFETY: reads 96 bytes.
                    let e = unsafe { blst::blst_p1_deserialize(&mut a, g1u[i % NP].as_ptr()) };
                    // SAFETY: blst on its own types.
                    assert!(e == blst::BLST_ERROR::BLST_SUCCESS && unsafe { blst::blst_p1_affine_in_g1(&a) });
                    black_box(a);
                }),
                side("bls12_381 0.8", "from_uncompressed", |i| {
                    black_box(bls12_381::G1Affine::from_uncompressed(g1u[i % NP].as_slice().try_into().unwrap()).unwrap());
                }),
            ];
            b.run("g1_from_zcash_uncompressed", &par, "", |i| {
                black_box(G1::from_zcash(c, &g1u[i % NP]).unwrap());
            }, refs);
            let mut refs = vec![
                side("arkworks 0.5", "deserialize_compressed (validates)", |i| {
                    black_box(ab::G2Affine::deserialize_compressed(&g2c[i % NP][..]).unwrap());
                }),
                side("blst 0.3.17", "blst_p2_uncompress + blst_p2_affine_in_g2", |i| {
                    let mut a = blst::blst_p2_affine::default();
                    // SAFETY: reads 96 bytes.
                    let e = unsafe { blst::blst_p2_uncompress(&mut a, g2c[i % NP].as_ptr()) };
                    // SAFETY: blst on its own types.
                    assert!(e == blst::BLST_ERROR::BLST_SUCCESS && unsafe { blst::blst_p2_affine_in_g2(&a) });
                    black_box(a);
                }),
                side("bls12_381 0.8", "from_compressed", |i| {
                    black_box(bls12_381::G2Affine::from_compressed(g2c[i % NP].as_slice().try_into().unwrap()).unwrap());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnG2_deserialize + mclBnG2_isValidOrder", |i| {
                let mut p = mcl::G2::default();
                // SAFETY: reads 96 bytes; mcl on its own types.
                let n = unsafe { mcl::mclBnG2_deserialize(&mut p, g2c[i % NP].as_ptr(), 96) };
                // SAFETY: as above.
                assert!(n == 96 && unsafe { mcl::mclBnG2_isValidOrder(&p) } == 1);
                black_box(p);
            }));
            b.run("g2_from_zcash_compressed", &par, "", |i| {
                black_box(G2::from_zcash(c, &g2c[i % NP]).unwrap());
            }, refs);
            let refs = vec![
                side("arkworks 0.5", "deserialize_uncompressed (validates)", |i| {
                    black_box(ab::G2Affine::deserialize_uncompressed(&g2u[i % NP][..]).unwrap());
                }),
                side("blst 0.3.17", "blst_p2_deserialize + blst_p2_affine_in_g2", |i| {
                    let mut a = blst::blst_p2_affine::default();
                    // SAFETY: reads 192 bytes.
                    let e = unsafe { blst::blst_p2_deserialize(&mut a, g2u[i % NP].as_ptr()) };
                    // SAFETY: blst on its own types.
                    assert!(e == blst::BLST_ERROR::BLST_SUCCESS && unsafe { blst::blst_p2_affine_in_g2(&a) });
                    black_box(a);
                }),
                side("bls12_381 0.8", "from_uncompressed", |i| {
                    black_box(bls12_381::G2Affine::from_uncompressed(g2u[i % NP].as_slice().try_into().unwrap()).unwrap());
                }),
            ];
            b.run("g2_from_zcash_uncompressed", &par, "", |i| {
                black_box(G2::from_zcash(c, &g2u[i % NP]).unwrap());
            }, refs);

            // G1::new / G2::new from affine coordinates (curve and subgroup checks).
            let cs1: Vec<_> = ps.iter().map(|p| p.coordinates().unwrap()).collect();
            let cs2: Vec<_> = qs.iter().map(|p| p.coordinates().unwrap()).collect();
            b.run("g1_new", &format!("{curve}, affine coordinates, checked"), "", |i| {
                let (x, y) = &cs1[i % NP];
                black_box(G1::new(c, x.clone(), y.clone()).unwrap());
            }, vec![side("arkworks 0.5", "G1Affine::new (asserts curve and subgroup)", |i| {
                let p = &ag1[i % NP];
                black_box(ab::G1Affine::new(p.x, p.y));
            })]);
            b.run("g2_new", &format!("{curve}, affine coordinates, checked"), "", |i| {
                let (x, y) = &cs2[i % NP];
                black_box(G2::new(c, x.clone(), y.clone()).unwrap());
            }, vec![side("arkworks 0.5", "G2Affine::new (asserts curve and subgroup)", |i| {
                let p = &ag2[i % NP];
                black_box(ab::G2Affine::new(p.x, p.y));
            })]);
        }

        // ---------------- GT ----------------
        {
            let oe: Vec<Gt> = (0..NP).map(|i| c.pairing(&ps[i], &qs[(i + 1) % NP])).collect();
            let ae: Vec<PairingOutput<AE>> = (0..NP).map(|i| AE::pairing(ag1[i], ag2[(i + 1) % NP])).collect();
            let ze: Vec<bls12_381::Gt> = (0..NP).map(|i| bls12_381::pairing(&zp[i], &zq[(i + 1) % NP])).collect();
            let be: Vec<blst::blst_fp12> = (0..NP)
                .map(|i| {
                    let mut f = blst::blst_fp12::default();
                    let mut g = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe {
                        blst::blst_miller_loop(&mut f, &bq[(i + 1) % NP], &bp[i]);
                        blst::blst_final_exp(&mut g, &f);
                    }
                    g
                })
                .collect();
            #[cfg(have_mcl)]
            let me: Vec<mcl::GT> = (0..NP)
                .map(|i| {
                    let mut g = mcl::GT::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBn_pairing(&mut g, &mp[i], &mq[(i + 1) % NP]) };
                    g
                })
                .collect();
            for i in 0..3 {
                let h = gt_hex(&oe[i].pow(&ok[i]));
                assert_eq!(ark_gt_hex(&(ae[i] * afr[i]).0), h, "ark gt pow");
                assert_eq!(zk_gt_hex(&(ze[i] * zfr[i])), h, "zk gt pow");
                #[cfg(have_mcl)]
                {
                    let mut g = mcl::GT::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnGT_pow(&mut g, &me[i], &mfr[i]) };
                    assert_eq!(mcl_tokens(&mcl::gt_str(&g)), h, "mcl gt pow");
                }
                let h = gt_hex(&oe[i].square());
                assert_eq!(ark_gt_hex(&ae[i].double().0), h);
                let h = gt_hex(&(&oe[i] * &oe[i + 1]));
                assert_eq!(ark_gt_hex(&(ae[i] + ae[i + 1]).0), h);
            }
            let par = format!("{curve}, 255-bit exponent < r");
            let mut refs = vec![
                side("arkworks 0.5", "cyclotomic_exp", |i| {
                    black_box(ae[i % NP] * afr[i % NK]);
                }),
                side("bls12_381 0.8", "constant-time square-and-multiply", |i| {
                    black_box(ze[i % NP] * zfr[i % NK]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_pow (GLV)", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_pow(&mut g, &me[i % NP], &mfr[i % NK]) };
                black_box(g);
            }));
            b.run("gt_pow", &par, "", |i| {
                black_box(oe[i % NP].pow(&ok[i % NK]));
            }, refs);
            let par = curve.to_string();
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ae[i % NP] + ae[(i / NP + i + 1) % NP]);
                }),
                side("blst 0.3.17", "blst_fp12_mul", |i| {
                    let mut g = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_fp12_mul(&mut g, &be[i % NP], &be[(i / NP + i + 1) % NP]) };
                    black_box(g);
                }),
                side("bls12_381 0.8", "", |i| {
                    black_box(ze[i % NP] + ze[(i / NP + i + 1) % NP]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_mul(&mut g, &me[i % NP], &me[(i / NP + i + 1) % NP]) };
                black_box(g);
            }));
            b.run("gt_mul", &par, "", |i| {
                black_box(&oe[i % NP] * &oe[(i / NP + i + 1) % NP]);
            }, refs);
            let mut refs = vec![
                side("arkworks 0.5", "cyclotomic_square", |i| {
                    black_box(ae[i % NP].double());
                }),
                side("blst 0.3.17", "blst_fp12_cyclotomic_sqr", |i| {
                    let mut g = blst::blst_fp12::default();
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_fp12_cyclotomic_sqr(&mut g, &be[i % NP]) };
                    black_box(g);
                }),
                side("bls12_381 0.8", "Gt::double", |i| {
                    black_box(ze[i % NP].double());
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_sqr", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_sqr(&mut g, &me[i % NP]) };
                black_box(g);
            }));
            b.run("gt_square", &par, "", |i| {
                black_box(oe[i % NP].square());
            }, refs);
            let mut refs = vec![
                side("arkworks 0.5", "neg = cyclotomic_inverse", |i| {
                    black_box(-ae[i % NP]);
                }),
                side("blst 0.3.17", "blst_fp12_conjugate on a copy", |i| {
                    let mut g = be[i % NP];
                    // SAFETY: blst on its own types.
                    unsafe { blst::blst_fp12_conjugate(&mut g) };
                    black_box(g);
                }),
                side("bls12_381 0.8", "neg", |i| {
                    black_box(-ze[i % NP]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_inv (conjugate)", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_inv(&mut g, &me[i % NP]) };
                black_box(g);
            }));
            b.run("gt_inverse", &par, "", |i| {
                black_box(oe[i % NP].inverse());
            }, refs);
            // From F_p^12, checked to be in GT.
            let f12: Vec<_> = oe.iter().map(|e| e.value().clone()).collect();
            let mut refs = vec![side("blst 0.3.17", "blst_fp12_in_group", |i| {
                // SAFETY: blst on its own types.
                black_box(unsafe { blst::blst_fp12_in_group(&be[i % NP]) });
            })];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_isValid", |i| {
                // SAFETY: mcl on its own types.
                black_box(unsafe { mcl::mclBnGT_isValid(&me[i % NP]) });
            }));
            b.run("gt_from_fp12", &format!("{curve}, element of GT"), "includes a clone of the Fp12 (it takes it by value)", |i| {
                black_box(Gt::from_fp12(c, f12[i % NP].clone()).unwrap());
            }, refs);
        }
    }
}

mod bn {
    use super::*;
    use ark_bn254 as ab;
    use ark_ec::pairing::{MillerLoopOutput, Pairing, PairingOutput};
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, PrimeGroup};
    use ark_ff::{BigInteger, One, PrimeField as _, Zero};
    use sb::Group as _;
    use substrate_bn as sb;

    type AE = ab::Bn254;

    fn ark_fq_hex(x: &ab::Fq) -> String {
        strip(&hex(&x.into_bigint().to_bytes_be()))
    }
    fn ark_gt_hex(e: &ab::Fq12) -> Vec<String> {
        let mut v = Vec::new();
        for c in [e.c0, e.c1] {
            for d in [c.c0, c.c1, c.c2] {
                v.push(ark_fq_hex(&d.c0));
                v.push(ark_fq_hex(&d.c1));
            }
        }
        v
    }
    fn be32(x: &ab::Fq) -> [u8; 32] {
        x.into_bigint().to_bytes_be().try_into().unwrap()
    }
    fn fq_from_be(b: &[u8]) -> Option<ab::Fq> {
        let mut w = [0u64; 4];
        for (k, ch) in b.chunks(8).enumerate() {
            w[3 - k] = u64::from_be_bytes(ch.try_into().unwrap());
        }
        ab::Fq::from_bigint(ark_ff::BigInt::new(w))
    }
    // EIP-196/197 encodings by arkworks.
    fn ark_g1_eth(p: &ab::G1Projective) -> Vec<u8> {
        let a = p.into_affine();
        if a.is_zero() {
            return vec![0; 64];
        }
        let mut v = be32(&a.x).to_vec();
        v.extend(be32(&a.y));
        v
    }
    fn ark_g2_eth(p: &ab::G2Projective) -> Vec<u8> {
        let a = p.into_affine();
        if a.is_zero() {
            return vec![0; 128];
        }
        let mut v = be32(&a.x.c1).to_vec();
        v.extend(be32(&a.x.c0));
        v.extend(be32(&a.y.c1));
        v.extend(be32(&a.y.c0));
        v
    }
    fn ark_g1_from_eth(b: &[u8]) -> Option<ab::G1Affine> {
        let (x, y) = (fq_from_be(&b[..32])?, fq_from_be(&b[32..64])?);
        if x.is_zero() && y.is_zero() {
            return Some(<ab::G1Affine as AffineRepr>::zero());
        }
        let p = ab::G1Affine::new_unchecked(x, y);
        p.is_on_curve().then_some(p)
    }
    fn ark_g2_from_eth(b: &[u8]) -> Option<ab::G2Affine> {
        let x = ab::Fq2::new(fq_from_be(&b[32..64])?, fq_from_be(&b[..32])?);
        let y = ab::Fq2::new(fq_from_be(&b[96..128])?, fq_from_be(&b[64..96])?);
        if x.is_zero() && y.is_zero() {
            return Some(<ab::G2Affine as AffineRepr>::zero());
        }
        let p = ab::G2Affine::new_unchecked(x, y);
        (p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()).then_some(p)
    }
    // The precompiles as revm's arkworks backend computes them.
    fn ark_ec_add(input: &[u8]) -> [u8; 64] {
        let mut buf = [0u8; 128];
        let n = input.len().min(128);
        buf[..n].copy_from_slice(&input[..n]);
        let p = ark_g1_from_eth(&buf[..64]).unwrap();
        let q = ark_g1_from_eth(&buf[64..]).unwrap();
        ark_g1_eth(&(p + q)).try_into().unwrap()
    }
    fn ark_ec_mul(input: &[u8]) -> [u8; 64] {
        let mut buf = [0u8; 96];
        let n = input.len().min(96);
        buf[..n].copy_from_slice(&input[..n]);
        let p = ark_g1_from_eth(&buf[..64]).unwrap();
        let k = ab::Fr::from_be_bytes_mod_order(&buf[64..96]);
        ark_g1_eth(&(p * k)).try_into().unwrap()
    }
    fn ark_ec_pairing(input: &[u8]) -> [u8; 32] {
        assert!(input.len() % 192 == 0);
        let (mut g1, mut g2) = (Vec::new(), Vec::new());
        for ch in input.chunks(192) {
            g1.push(ark_g1_from_eth(&ch[..64]).unwrap());
            g2.push(ark_g2_from_eth(&ch[64..]).unwrap());
        }
        let mut out = [0u8; 32];
        out[31] = AE::multi_pairing(g1, g2).0.is_one() as u8;
        out
    }
    // The precompiles as substrate-bn's users (parity, revm's default backend) compute them.
    fn sb_fq(b: &[u8]) -> sb::Fq {
        sb::Fq::from_slice(b).unwrap()
    }
    fn sb_g1(b: &[u8]) -> sb::G1 {
        let (x, y) = (sb_fq(&b[..32]), sb_fq(&b[32..64]));
        if x.is_zero() && y.is_zero() {
            sb::G1::zero()
        } else {
            sb::AffineG1::new(x, y).unwrap().into()
        }
    }
    fn sb_g2(b: &[u8]) -> sb::G2 {
        let x = sb::Fq2::new(sb_fq(&b[32..64]), sb_fq(&b[..32]));
        let y = sb::Fq2::new(sb_fq(&b[96..128]), sb_fq(&b[64..96]));
        if x.is_zero() && y.is_zero() {
            sb::G2::zero()
        } else {
            sb::AffineG2::new(x, y).unwrap().into()
        }
    }
    fn sb_g1_eth(p: sb::G1) -> [u8; 64] {
        let mut out = [0u8; 64];
        if let Some(a) = sb::AffineG1::from_jacobian(p) {
            a.x().to_big_endian(&mut out[..32]).unwrap();
            a.y().to_big_endian(&mut out[32..]).unwrap();
        }
        out
    }
    fn sb_ec_add(input: &[u8]) -> [u8; 64] {
        let mut buf = [0u8; 128];
        let n = input.len().min(128);
        buf[..n].copy_from_slice(&input[..n]);
        sb_g1_eth(sb_g1(&buf[..64]) + sb_g1(&buf[64..]))
    }
    fn sb_ec_mul(input: &[u8]) -> [u8; 64] {
        let mut buf = [0u8; 96];
        let n = input.len().min(96);
        buf[..n].copy_from_slice(&input[..n]);
        let k = sb::Fr::from_slice(&buf[64..96]).unwrap();
        sb_g1_eth(sb_g1(&buf[..64]) * k)
    }
    fn sb_ec_pairing(input: &[u8]) -> [u8; 32] {
        let pairs: Vec<(sb::G1, sb::G2)> = input.chunks(192).map(|ch| (sb_g1(&ch[..64]), sb_g2(&ch[64..]))).collect();
        let mut out = [0u8; 32];
        out[31] = (sb::pairing_batch(&pairs) == sb::Gt::one()) as u8;
        out
    }

    pub fn run() {
        use a::math::pairing::ethereum;
        let c = PairingCurve::bn254();
        let mut b = Bench::new("pairing");
        let curve = "BN254";
        let ks = scalars(13, c.order());
        let mut rng = Rng(9);
        let ag1p: Vec<ab::G1Projective> = (0..NP)
            .map(|_| ab::G1Projective::generator() * ab::Fr::from_le_bytes_mod_order(&rng.bytes(32)))
            .collect();
        let ag2p: Vec<ab::G2Projective> = (0..NP)
            .map(|_| ab::G2Projective::generator() * ab::Fr::from_le_bytes_mod_order(&rng.bytes(32)))
            .collect();
        let g1e: Vec<Vec<u8>> = ag1p.iter().map(ark_g1_eth).collect();
        let g2e: Vec<Vec<u8>> = ag2p.iter().map(ark_g2_eth).collect();
        let ag1: Vec<ab::G1Affine> = g1e.iter().map(|x| ark_g1_from_eth(x).unwrap()).collect();
        let ag2: Vec<ab::G2Affine> = g2e.iter().map(|x| ark_g2_from_eth(x).unwrap()).collect();
        let ap1: Vec<ab::G1Projective> = ag1.iter().map(|p| p.into_group()).collect();
        let ap2: Vec<ab::G2Projective> = ag2.iter().map(|p| p.into_group()).collect();
        let ps: Vec<G1> = g1e.iter().map(|x| G1::from_ethereum(c, x).unwrap()).collect();
        let qs: Vec<G2> = g2e.iter().map(|x| G2::from_ethereum(c, x).unwrap()).collect();
        let ok: Vec<Number> = ks.iter().map(num_le).collect();
        let afr: Vec<ab::Fr> = ks.iter().map(|k| ab::Fr::from_le_bytes_mod_order(k)).collect();
        let sp: Vec<sb::G1> = g1e.iter().map(|x| sb_g1(x)).collect();
        let sq: Vec<sb::G2> = g2e.iter().map(|x| sb_g2(x)).collect();
        let sfr: Vec<sb::Fr> = ks
            .iter()
            .map(|k| {
                let mut be = *k;
                be.reverse();
                sb::Fr::from_slice(&be).unwrap()
            })
            .collect();
        #[cfg(have_mcl)]
        let (mp, mq, mfr) = {
            mcl::init(mcl::BN_SNARK1);
            // SAFETY: mcl's settings, after init.
            unsafe {
                mcl::mclBn_verifyOrderG1(0);
                mcl::mclBn_verifyOrderG2(0);
            }
            let mp: Vec<mcl::G1> = ps.iter().map(|p| mcl::g1_from_str(&ours_g1_str(p))).collect();
            let mq: Vec<mcl::G2> = qs.iter().map(|q| mcl::g2_from_str(&ours_g2_str(q))).collect();
            let mfr: Vec<mcl::Fr> = ks.iter().map(|k| mcl::fr_le(k)).collect();
            (mp, mq, mfr)
        };
        #[cfg(have_mcl)]
        let mcl_name = format!("mcl {}", mcl::version());

        // substrate-bn and mcl may compute other powers of the pairing: find out once.
        let e0 = c.pairing(&ps[0], &qs[0]);
        assert_eq!(ark_gt_hex(&AE::pairing(ag1[0], ag2[0]).0), gt_hex(&e0), "ark pairing");
        #[cfg(have_mcl)]
        let mcl_same = {
            let mut g = mcl::GT::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBn_pairing(&mut g, &mp[0], &mq[0]) };
            mcl_tokens(&mcl::gt_str(&g)) == gt_hex(&e0)
        };
        #[cfg(have_mcl)]
        let mcl_note = if mcl_same { "" } else { "mcl's value is another power of the pairing (its final exponentiation); pairing checks agree" };
        // substrate-bn: check bilinearity and pairing checks (its Gt has no accessor).
        {
            let pp = sp[0] * sfr[1];
            assert!(sb::pairing_batch(&[(pp, sq[0]), (-sp[0], sq[0] * sfr[1])]) == sb::Gt::one(), "substrate-bn bilinearity");
        }
        let sb_note = "substrate-bn's Gt has no accessor: checked by bilinearity and the precompile outputs";

        let par = format!("{curve}, 1 pair");
        {
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(AE::pairing(ag1[i % NP], ag2[(i / NP) % NP]));
                }),
                side("substrate-bn 0.6", sb_note, |i| {
                    black_box(sb::pairing(sp[i % NP], sq[(i / NP) % NP]));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, mcl_note, |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_pairing(&mut g, &mp[i % NP], &mq[(i / NP) % NP]) };
                black_box(g);
            }));
            b.run("pairing", &par, "", |i| {
                black_box(c.pairing(&ps[i % NP], &qs[(i / NP) % NP]));
            }, refs);
        }
        {
            let mut refs = vec![side("arkworks 0.5", "", |i| {
                black_box(AE::multi_miller_loop([ag1[i % NP]], [ag2[(i / NP) % NP]]));
            })];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_millerLoop(&mut g, &mp[i % NP], &mq[(i / NP) % NP]) };
                black_box(g);
            }));
            b.run("miller_loop", &par, "outputs differ between libraries by factors the final exponentiation removes", |i| {
                black_box(c.miller_loop(&[(&ps[i % NP], &qs[(i / NP) % NP])]));
            }, refs);
        }
        {
            let alines: Vec<<AE as Pairing>::G2Prepared> = ag2.iter().map(|q| (*q).into()).collect();
            b.run("miller_loop_fixed_q", &par, "ours has no precomputed lines: our plain Miller loop", |i| {
                black_box(c.miller_loop(&[(&ps[i % NP], &qs[(i / NP) % NP])]));
            }, vec![side("arkworks 0.5", "G2Prepared reused (cloned per call)", |i| {
                black_box(AE::multi_miller_loop([ag1[i % NP]], [alines[(i / NP) % NP].clone()]));
            })]);
        }
        {
            let of: Vec<_> = (0..NP).map(|i| c.miller_loop(&[(&ps[i], &qs[(i + 3) % NP])])).collect();
            let af: Vec<_> = (0..NP).map(|i| AE::multi_miller_loop([ag1[i]], [ag2[(i + 3) % NP]])).collect();
            #[cfg(have_mcl)]
            let mf: Vec<mcl::GT> = (0..NP)
                .map(|i| {
                    let mut g = mcl::GT::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBn_millerLoop(&mut g, &mp[i], &mq[(i + 3) % NP]) };
                    g
                })
                .collect();
            for i in 0..2 {
                assert_eq!(ark_gt_hex(&AE::final_exponentiation(af[i]).unwrap().0), gt_hex(&c.final_exponentiation(&of[i])));
            }
            let mut refs = vec![side("arkworks 0.5", "", |i| {
                black_box(AE::final_exponentiation(MillerLoopOutput(af[i % NP].0)));
            })];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, mcl_note, |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBn_finalExp(&mut g, &mf[i % NP]) };
                black_box(g);
            }));
            b.run("final_exponentiation", curve, "", |i| {
                black_box(c.final_exponentiation(&of[i % NP]));
            }, refs);
        }
        for n in [2usize, 4, 8, 16] {
            let par = format!("{curve}, {n} pairs");
            let pick = |i: usize, j: usize| (i + j * 5) % NP;
            {
                let pairs: Vec<(&G1, &G2)> = (0..n).map(|j| (&ps[pick(0, j)], &qs[pick(1, j)])).collect();
                let a = AE::multi_pairing((0..n).map(|j| ag1[pick(0, j)]), (0..n).map(|j| ag2[pick(1, j)]));
                assert_eq!(ark_gt_hex(&a.0), gt_hex(&c.multi_pairing(&pairs)));
            }
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(AE::multi_pairing((0..n).map(|j| ag1[pick(i, j)]), (0..n).map(|j| ag2[pick(i + 1, j)])));
                }),
                side("substrate-bn 0.6", "pairing_batch", |i| {
                    let pairs: Vec<(sb::G1, sb::G2)> = (0..n).map(|j| (sp[pick(i, j)], sq[pick(i + 1, j)])).collect();
                    black_box(sb::pairing_batch(&pairs));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBn_millerLoopVec + mclBn_finalExp", |i| {
                let pp: Vec<mcl::G1> = (0..n).map(|j| mp[pick(i, j)]).collect();
                let qq: Vec<mcl::G2> = (0..n).map(|j| mq[pick(i + 1, j)]).collect();
                let mut f = mcl::GT::default();
                let mut g = mcl::GT::default();
                // SAFETY: mcl on arrays of n elements.
                unsafe {
                    mcl::mclBn_millerLoopVec(&mut f, pp.as_ptr(), qq.as_ptr(), n);
                    mcl::mclBn_finalExp(&mut g, &f);
                }
                black_box(g);
            }));
            b.run("multi_pairing", &par, "", |i| {
                let pairs: Vec<(&G1, &G2)> = (0..n).map(|j| (&ps[pick(i, j)], &qs[pick(i + 1, j)])).collect();
                black_box(c.multi_pairing(&pairs));
            }, refs);
        }

        // Scalar multiplication.
        for i in 0..4 {
            let r = &ps[i] * &ok[i];
            assert_eq!(ark_g1_eth(&(ap1[i] * afr[i])), r.to_ethereum(), "ark g1 mul");
            assert_eq!(sb_g1_eth(sp[i] * sfr[i]).to_vec(), r.to_ethereum(), "sb g1 mul");
            let r2 = &qs[i] * &ok[i];
            assert_eq!(ark_g2_eth(&(ap2[i] * afr[i])), r2.to_ethereum(), "ark g2 mul");
            #[cfg(have_mcl)]
            {
                let mut o = mcl::G1::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG1_mul(&mut o, &mp[i], &mfr[i]) };
                assert_eq!(mcl_tokens(&mcl::g1_str(&o)), mcl_tokens(&ours_g1_str(&r)));
                let mut o = mcl::G2::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnG2_mul(&mut o, &mq[i], &mfr[i]) };
                assert_eq!(mcl_tokens(&mcl::g2_str(&o)), mcl_tokens(&ours_g2_str(&r2)));
            }
        }
        let par = format!("{curve}, 254-bit scalar < r, affine input");
        let mut refs = vec![
            side("arkworks 0.5", "GLV", |i| {
                black_box(ap1[i % NP] * afr[i % NK]);
            }),
            side("substrate-bn 0.6", "", |i| {
                black_box(sp[i % NP] * sfr[i % NK]);
            }),
        ];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "mclBnG1_mul (GLV, variable-time)", |i| {
            let mut o = mcl::G1::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG1_mul(&mut o, &mp[i % NP], &mfr[i % NK]) };
            black_box(o);
        }));
        b.run("g1_mul", &par, "", |i| {
            black_box(&ps[i % NP] * &ok[i % NK]);
        }, refs);
        let mut refs = vec![
            side("arkworks 0.5", "", |i| {
                black_box(ap2[i % NP] * afr[i % NK]);
            }),
            side("substrate-bn 0.6", "", |i| {
                black_box(sq[i % NP] * sfr[i % NK]);
            }),
        ];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "mclBnG2_mul (GLS, variable-time)", |i| {
            let mut o = mcl::G2::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG2_mul(&mut o, &mq[i % NP], &mfr[i % NK]) };
            black_box(o);
        }));
        b.run("g2_mul", &par, "", |i| {
            black_box(&qs[i % NP] * &ok[i % NK]);
        }, refs);

        let rs1: Vec<G1> = (0..NP).map(|i| &ps[i] * &ok[i]).collect();
        let rs2: Vec<G2> = (0..NP).map(|i| &qs[i] * &ok[i]).collect();
        let ar1: Vec<ab::G1Projective> = (0..NP).map(|i| ap1[i] * afr[i]).collect();
        let ar2: Vec<ab::G2Projective> = (0..NP).map(|i| ap2[i] * afr[i]).collect();
        let sr1: Vec<sb::G1> = (0..NP).map(|i| sp[i] * sfr[i]).collect();
        let sr2: Vec<sb::G2> = (0..NP).map(|i| sq[i] * sfr[i]).collect();
        #[cfg(have_mcl)]
        let (mr1, mr2): (Vec<mcl::G1>, Vec<mcl::G2>) = (
            (0..NP)
                .map(|i| {
                    let mut o = mcl::G1::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnG1_mul(&mut o, &mp[i], &mfr[i]) };
                    o
                })
                .collect(),
            (0..NP)
                .map(|i| {
                    let mut o = mcl::G2::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBnG2_mul(&mut o, &mq[i], &mfr[i]) };
                    o
                })
                .collect(),
        );
        let par = format!("{curve}, projective inputs");
        let nx = |i: usize| (i / NP + i + 1) % NP;
        let mut refs = vec![
            side("arkworks 0.5", "", |i| {
                black_box(ar1[i % NP] + ar1[nx(i)]);
            }),
            side("substrate-bn 0.6", "", |i| {
                black_box(sr1[i % NP] + sr1[nx(i)]);
            }),
        ];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "", |i| {
            let mut o = mcl::G1::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG1_add(&mut o, &mr1[i % NP], &mr1[nx(i)]) };
            black_box(o);
        }));
        b.run("g1_add", &par, "", |i| {
            black_box(&rs1[i % NP] + &rs1[nx(i)]);
        }, refs);
        let mut refs = vec![side("arkworks 0.5", "", |i| {
            black_box(ar1[i % NP].double());
        })];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "", |i| {
            let mut o = mcl::G1::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG1_dbl(&mut o, &mr1[i % NP]) };
            black_box(o);
        }));
        b.run("g1_double", &par, "", |i| {
            black_box(rs1[i % NP].double());
        }, refs);
        let mut refs = vec![
            side("arkworks 0.5", "", |i| {
                black_box(ar2[i % NP] + ar2[nx(i)]);
            }),
            side("substrate-bn 0.6", "", |i| {
                black_box(sr2[i % NP] + sr2[nx(i)]);
            }),
        ];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "", |i| {
            let mut o = mcl::G2::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG2_add(&mut o, &mr2[i % NP], &mr2[nx(i)]) };
            black_box(o);
        }));
        b.run("g2_add", &par, "", |i| {
            black_box(&rs2[i % NP] + &rs2[nx(i)]);
        }, refs);
        let mut refs = vec![side("arkworks 0.5", "", |i| {
            black_box(ar2[i % NP].double());
        })];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "", |i| {
            let mut o = mcl::G2::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG2_dbl(&mut o, &mr2[i % NP]) };
            black_box(o);
        }));
        b.run("g2_double", &par, "", |i| {
            black_box(rs2[i % NP].double());
        }, refs);
        let mut refs = vec![side("arkworks 0.5", "into_affine", |i| {
            black_box(ar1[i % NP].into_affine());
        })];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "mclBnG1_normalize", |i| {
            let mut o = mcl::G1::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG1_normalize(&mut o, &mr1[i % NP]) };
            black_box(o);
        }));
        b.run("g1_coordinates", &par, "", |i| {
            black_box(rs1[i % NP].coordinates());
        }, refs);
        let mut refs = vec![side("arkworks 0.5", "into_affine", |i| {
            black_box(ar2[i % NP].into_affine());
        })];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "mclBnG2_normalize", |i| {
            let mut o = mcl::G2::default();
            // SAFETY: mcl on its own types.
            unsafe { mcl::mclBnG2_normalize(&mut o, &mr2[i % NP]) };
            black_box(o);
        }));
        b.run("g2_coordinates", &par, "", |i| {
            black_box(rs2[i % NP].coordinates());
        }, refs);

        // Subgroup checks (G1 of BN254 is all of E(F_p)).
        b.run("g1_is_in_subgroup", &format!("{curve}, points of G1"), "always true: E(F_p) has prime order", |i| {
            black_box(ps[i % NP].is_in_subgroup());
        }, vec![side("arkworks 0.5", "", |i| {
            black_box(ag1[i % NP].is_in_correct_subgroup_assuming_on_curve());
        })]);
        let mut refs = vec![side("arkworks 0.5", "", |i| {
            black_box(ag2[i % NP].is_in_correct_subgroup_assuming_on_curve());
        })];
        #[cfg(have_mcl)]
        refs.push(side(&mcl_name, "mclBnG2_isValidOrder", |i| {
            // SAFETY: mcl on its own types.
            black_box(unsafe { mcl::mclBnG2_isValidOrder(&mq[i % NP]) });
        }));
        b.run("g2_is_in_subgroup", &format!("{curve}, points of G2 (affine)"), "", |i| {
            black_box(qs[i % NP].is_in_subgroup());
        }, refs);

        // Cofactors: twist points outside G2 (from an ark random point of the twist).
        {
            let mut r = Rng(31);
            let mut os2: Vec<G2> = Vec::new();
            let mut oa2: Vec<ab::G2Affine> = Vec::new();
            while os2.len() < NP {
                let x = ab::Fq2::new(ab::Fq::from_le_bytes_mod_order(&r.bytes(32)), ab::Fq::from_le_bytes_mod_order(&r.bytes(32)));
                if let Some(p) = ab::G2Affine::get_point_from_x_unchecked(x, false) {
                    let t = c.tower();
                    let xx = a::math::field::Fp2::from_numbers(
                        t,
                        &Number::new(p.x.c0.into_bigint().to_bytes_le(), ByteOrder::LittleEndian),
                        &Number::new(p.x.c1.into_bigint().to_bytes_le(), ByteOrder::LittleEndian),
                    );
                    let yy = a::math::field::Fp2::from_numbers(
                        t,
                        &Number::new(p.y.c0.into_bigint().to_bytes_le(), ByteOrder::LittleEndian),
                        &Number::new(p.y.c1.into_bigint().to_bytes_le(), ByteOrder::LittleEndian),
                    );
                    os2.push(G2::new_on_curve(c, xx, yy).unwrap());
                    oa2.push(p);
                }
            }
            for i in 0..3 {
                assert!(!os2[i].is_in_subgroup());
                assert_eq!(ark_g2_eth(&oa2[i].clear_cofactor().into_group()), os2[i].mul_by_cofactor().to_ethereum(), "ark mul_by_cofactor");
                assert!(os2[i].clear_cofactor().is_in_subgroup());
            }
            let par = format!("{curve}, twist points outside G2");
            b.run("g2_clear_cofactor", &par, "Fuentes-Castaneda-Knapp-Rodriguez-Henriquez map", |i| {
                black_box(os2[i % NP].clear_cofactor());
            }, vec![]);
            b.run("g2_mul_by_cofactor", &par, "254-bit cofactor 2p - r", |i| {
                black_box(os2[i % NP].mul_by_cofactor());
            }, vec![side("arkworks 0.5", "clear_cofactor = product by the cofactor on BN254's G2", |i| {
                black_box(oa2[i % NP].clear_cofactor());
            })]);
            b.run("g2_psi", &format!("{curve}, projective input"), "", |i| {
                black_box(rs2[i % NP].psi());
            }, vec![]);
            b.run("g1_clear_cofactor", &format!("{curve}, points of G1"), "the identity map (cofactor 1)", |i| {
                black_box(ps[i % NP].clear_cofactor());
            }, vec![]);
        }

        // Ethereum's encodings.
        {
            for i in 0..4 {
                assert_eq!(ark_g1_eth(&ar1[i]), rs1[i].to_ethereum());
                assert_eq!(sb_g1_eth(sr1[i]).to_vec(), rs1[i].to_ethereum());
                assert_eq!(ark_g2_eth(&ar2[i]), rs2[i].to_ethereum());
            }
            let par = format!("{curve}, from a projective point");
            b.run("g1_to_ethereum", &par, "", |i| {
                black_box(rs1[i % NP].to_ethereum());
            }, vec![
                side("arkworks 0.5", "into_affine + big-endian coordinates", |i| {
                    black_box(ark_g1_eth(&ar1[i % NP]));
                }),
                side("substrate-bn 0.6", "AffineG1::from_jacobian + to_big_endian", |i| {
                    black_box(sb_g1_eth(sr1[i % NP]));
                }),
            ]);
            b.run("g2_to_ethereum", &par, "", |i| {
                black_box(rs2[i % NP].to_ethereum());
            }, vec![side("arkworks 0.5", "into_affine + big-endian coordinates", |i| {
                black_box(ark_g2_eth(&ar2[i % NP]));
            })]);
            let par = format!("{curve}, checked (field, curve; subgroup for G2)");
            b.run("g1_from_ethereum", &par, "", |i| {
                black_box(G1::from_ethereum(c, &g1e[i % NP]).unwrap());
            }, vec![
                side("arkworks 0.5", "Fq::from_bigint + is_on_curve", |i| {
                    black_box(ark_g1_from_eth(&g1e[i % NP]).unwrap());
                }),
                side("substrate-bn 0.6", "Fq::from_slice + AffineG1::new", |i| {
                    black_box(sb_g1(&g1e[i % NP]));
                }),
            ]);
            b.run("g2_from_ethereum", &par, "", |i| {
                black_box(G2::from_ethereum(c, &g2e[i % NP]).unwrap());
            }, vec![
                side("arkworks 0.5", "Fq::from_bigint + is_on_curve + subgroup", |i| {
                    black_box(ark_g2_from_eth(&g2e[i % NP]).unwrap());
                }),
                side("substrate-bn 0.6", "Fq::from_slice + AffineG2::new (curve and subgroup)", |i| {
                    black_box(sb_g2(&g2e[i % NP]));
                }),
            ]);
        }

        // Ethereum's precompiles on bytes.
        {
            let add_in: Vec<Vec<u8>> = (0..NP).map(|i| [rs1[i].to_ethereum(), rs1[(i + 1) % NP].to_ethereum()].concat()).collect();
            let mul_in: Vec<Vec<u8>> = (0..NP)
                .map(|i| {
                    let mut k = ks[(i * 3) % NK];
                    k.reverse();
                    [g1e[i].clone(), k.to_vec()].concat()
                })
                .collect();
            for i in 0..NP {
                let w = ethereum::ec_add(&add_in[i]).unwrap();
                assert_eq!(ark_ec_add(&add_in[i]), w);
                assert_eq!(sb_ec_add(&add_in[i]), w);
                let w = ethereum::ec_mul(&mul_in[i]).unwrap();
                assert_eq!(ark_ec_mul(&mul_in[i]), w);
                assert_eq!(sb_ec_mul(&mul_in[i]), w);
            }
            b.run("ec_add", &format!("{curve}, 128-byte input"), "", |i| {
                black_box(ethereum::ec_add(&add_in[i % NP]).unwrap());
            }, vec![
                side("arkworks 0.5", "decode, add, encode (revm's arkworks backend's steps)", |i| {
                    black_box(ark_ec_add(&add_in[i % NP]));
                }),
                side("substrate-bn 0.6", "decode, add, encode (revm's default backend's steps)", |i| {
                    black_box(sb_ec_add(&add_in[i % NP]));
                }),
            ]);
            b.run("ec_mul", &format!("{curve}, 96-byte input, 254-bit scalar"), "", |i| {
                black_box(ethereum::ec_mul(&mul_in[i % NP]).unwrap());
            }, vec![
                side("arkworks 0.5", "decode, multiply, encode", |i| {
                    black_box(ark_ec_mul(&mul_in[i % NP]));
                }),
                side("substrate-bn 0.6", "decode, multiply, encode", |i| {
                    black_box(sb_ec_mul(&mul_in[i % NP]));
                }),
            ]);
            for n in [1usize, 2, 4] {
                // Inputs whose check succeeds: pairs (aP, Q), (−P, aQ)… and a final pair to make it 1
                // when n is odd: (P, Q) and (−P, Q).
                let ins: Vec<Vec<u8>> = (0..NP)
                    .map(|i| {
                        let mut v = Vec::new();
                        let mut j = 0;
                        while j < n {
                            if n - j >= 2 {
                                let p = &ps[(i + j) % NP];
                                let q = &qs[(i + j) % NP];
                                let k = &ok[(i + j) % NK];
                                v.extend((p * k).to_ethereum());
                                v.extend(q.to_ethereum());
                                v.extend((-p).to_ethereum());
                                v.extend((q * k).to_ethereum());
                                j += 2;
                            } else {
                                v.extend(ps[(i + j) % NP].to_ethereum());
                                v.extend(qs[(i + j) % NP].to_ethereum());
                                j += 1;
                            }
                        }
                        v
                    })
                    .collect();
                for x in &ins {
                    let w = ethereum::ec_pairing(x).unwrap();
                    assert_eq!(ark_ec_pairing(x), w);
                    assert_eq!(sb_ec_pairing(x), w);
                    assert_eq!(w[31], (n % 2 == 0) as u8);
                }
                b.run("ec_pairing", &format!("{curve}, {n} pairs ({} bytes)", 192 * n), "", |i| {
                    black_box(ethereum::ec_pairing(&ins[i % NP]).unwrap());
                }, vec![
                    side("arkworks 0.5", "decode (subgroup check of G2), multi_pairing", |i| {
                        black_box(ark_ec_pairing(&ins[i % NP]));
                    }),
                    side("substrate-bn 0.6", "decode (subgroup check of G2), pairing_batch", |i| {
                        black_box(sb_ec_pairing(&ins[i % NP]));
                    }),
                ]);
            }
        }

        // GT.
        {
            let oe: Vec<Gt> = (0..NP).map(|i| c.pairing(&ps[i], &qs[(i + 1) % NP])).collect();
            let ae: Vec<PairingOutput<AE>> = (0..NP).map(|i| AE::pairing(ag1[i], ag2[(i + 1) % NP])).collect();
            let se: Vec<sb::Gt> = (0..NP).map(|i| sb::pairing(sp[i], sq[(i + 1) % NP])).collect();
            #[cfg(have_mcl)]
            let me: Vec<mcl::GT> = (0..NP)
                .map(|i| {
                    let mut g = mcl::GT::default();
                    // SAFETY: mcl on its own types.
                    unsafe { mcl::mclBn_pairing(&mut g, &mp[i], &mq[(i + 1) % NP]) };
                    g
                })
                .collect();
            for i in 0..3 {
                assert_eq!(ark_gt_hex(&(ae[i] * afr[i]).0), gt_hex(&oe[i].pow(&ok[i])));
                assert_eq!(ark_gt_hex(&ae[i].double().0), gt_hex(&oe[i].square()));
            }
            let par = format!("{curve}, 254-bit exponent < r");
            let mut refs = vec![
                side("arkworks 0.5", "cyclotomic_exp", |i| {
                    black_box(ae[i % NP] * afr[i % NK]);
                }),
                side("substrate-bn 0.6", "Gt::pow", |i| {
                    black_box(se[i % NP].pow(sfr[i % NK]));
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_pow (GLV)", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_pow(&mut g, &me[i % NP], &mfr[i % NK]) };
                black_box(g);
            }));
            b.run("gt_pow", &par, "", |i| {
                black_box(oe[i % NP].pow(&ok[i % NK]));
            }, refs);
            let par = curve.to_string();
            let mut refs = vec![
                side("arkworks 0.5", "", |i| {
                    black_box(ae[i % NP] + ae[nx(i)]);
                }),
                side("substrate-bn 0.6", "", |i| {
                    black_box(se[i % NP] * se[nx(i)]);
                }),
            ];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_mul(&mut g, &me[i % NP], &me[nx(i)]) };
                black_box(g);
            }));
            b.run("gt_mul", &par, "", |i| {
                black_box(&oe[i % NP] * &oe[nx(i)]);
            }, refs);
            let mut refs = vec![side("arkworks 0.5", "cyclotomic_square", |i| {
                black_box(ae[i % NP].double());
            })];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_sqr", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_sqr(&mut g, &me[i % NP]) };
                black_box(g);
            }));
            b.run("gt_square", &par, "", |i| {
                black_box(oe[i % NP].square());
            }, refs);
            let mut refs = vec![side("arkworks 0.5", "neg = cyclotomic_inverse", |i| {
                black_box(-ae[i % NP]);
            })];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_inv (conjugate)", |i| {
                let mut g = mcl::GT::default();
                // SAFETY: mcl on its own types.
                unsafe { mcl::mclBnGT_inv(&mut g, &me[i % NP]) };
                black_box(g);
            }));
            b.run("gt_inverse", &par, "", |i| {
                black_box(oe[i % NP].inverse());
            }, refs);
            let f12: Vec<_> = oe.iter().map(|e| e.value().clone()).collect();
            let mut refs = vec![];
            #[cfg(have_mcl)]
            refs.push(side(&mcl_name, "mclBnGT_isValid", |i| {
                // SAFETY: mcl on its own types.
                black_box(unsafe { mcl::mclBnGT_isValid(&me[i % NP]) });
            }));
            b.run("gt_from_fp12", &format!("{curve}, element of GT"), "includes a clone of the Fp12 (it takes it by value)", |i| {
                black_box(Gt::from_fp12(c, f12[i % NP].clone()).unwrap());
            }, refs);
            let _ = &mut b;
        }
    }
}
