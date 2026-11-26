// Curves over binary fields (a::math::ec::binary) on the 18 curves of SEC 2 against OpenSSL 3 and
// OpenSSL 4 (EC2M: EC_POINT_mul and friends), loaded at run time.
//
// Inputs: seeded random points of the prime-order subgroup (k·G), passed to OpenSSL as SEC 1
// uncompressed bytes; scalars uniform below the order n, as big-endian bytes for OpenSSL. Every
// output is compared with OpenSSL's before timing. OpenSSL holds points in affine coordinates and
// multiplies by the López–Dahab Montgomery ladder (with y recovered), with no precomputation for
// the generator on these curves.
#![allow(unused_must_use)]
#[path = "../common.rs"]
mod common;
#[path = "../ossl.rs"]
mod ossl;

use common::*;

use a::math::ec::binary::named;
use a::math::ec::binary::{BinaryPoint, BinaryPrepared};
use a::math::numbers::{ByteOrder, Number};
use ossl::{CP, Ossl, P};

const NP: usize = 16;
const NK: usize = 64;

struct Ref<'o> {
    o: &'o Ossl,
    g: P,
    ctx: P,
    pts: Vec<P>,
    ks: Vec<P>,
    ks2: Vec<P>,
    order: CP,
    _cof: CP,
    comp: Vec<Vec<u8>>,
    unc: Vec<Vec<u8>>,
    multi_pts: Vec<CP>,
    multi_ks: Vec<CP>,
}

type Op<'a> = Box<dyn Fn(&Ref, P, usize) + 'a>;

fn mk<'a, 'o: 'a>(rs: &'a [Ref<'o>], f: Op<'a>, note: &str) -> Vec<Side<'a>> {
    let f = std::rc::Rc::new(f);
    rs.iter()
        .map(|r| {
            let out = r.o.point(r.g);
            let f = f.clone();
            side(&r.o.name, note, move |i| f(r, out, i))
        })
        .collect()
}

fn main() {
    let refs: Vec<Ossl> = [3u32, 4].into_iter().filter_map(Ossl::load).collect();
    if refs.is_empty() {
        eprintln!("binary: no OpenSSL found; rows without references only");
    }
    let mut b = Bench::new("binary_curves");
    let only: Option<String> = std::env::args().nth(1);
    for name in named::NAMES {
        if let Some(o) = &only {
            if !name.contains(o.as_str()) {
                continue;
            }
        }
        curve(&mut b, name, &refs);
    }
}

fn curve(b: &mut Bench, name: &str, osls: &[Ossl]) {
    let d = named::by_name(name).unwrap();
    let n = d.order();
    let nbytes = number_be(n, 80).iter().skip_while(|&&x| x == 0).count();
    let nb: Vec<u8> = number_be(n, nbytes);
    let mask = 0xffu8 >> nb[0].leading_zeros();
    let mut rng = Rng(name.bytes().fold(17u64, |h, c| h.wrapping_mul(31).wrapping_add(c as u64)));
    let mut scalar = || loop {
        let mut be = rng.bytes(nbytes);
        be[0] &= mask;
        if be < nb && be.iter().any(|&x| x != 0) {
            return be;
        }
    };
    let kb: Vec<Vec<u8>> = (0..NK).map(|_| scalar()).collect();
    let kb2: Vec<Vec<u8>> = (0..NK).map(|_| scalar()).collect();
    let rb: Vec<Vec<u8>> = (0..NP).map(|_| scalar()).collect();
    let num = |be: &Vec<u8>| {
        let mut le = be.clone();
        le.reverse();
        Number::new(le, ByteOrder::LittleEndian)
    };
    let ks: Vec<Number> = kb.iter().map(num).collect();
    let ks2: Vec<Number> = kb2.iter().map(num).collect();
    let g = d.generator();
    let gen_pts: Vec<BinaryPoint> = rb.iter().map(|r| &g * &num(r)).collect();
    let unc: Vec<Vec<u8>> = gen_pts.iter().map(|p| p.to_bytes()).collect();
    let comp: Vec<Vec<u8>> = gen_pts.iter().map(|p| p.compress()).collect();
    // The points as a user gets them from bytes (affine).
    let ps: Vec<BinaryPoint> = unc.iter().map(|x| d.point_from_bytes(x).unwrap()).collect();
    let m = d.field().degree();
    let kind = if d.is_koblitz() { "Koblitz" } else { "random" };
    let label = format!("{name} (m = {m}, {kind})");
    let nbits = n.bits();

    let rs: Vec<Ref> = osls
        .iter()
        .filter_map(|o| {
            let g = o.group(name)?;
            let ctx = o.ctx();
            let pts: Vec<P> = unc.iter().map(|x| o.decode(g, x, ctx).unwrap()).collect();
            // SAFETY: the group's own order and cofactor.
            let (order, cof) = unsafe { ((o.EC_GROUP_get0_order)(g), (o.EC_GROUP_get0_cofactor)(g)) };
            let multi_pts: Vec<CP> = (0..64).map(|j| pts[j % NP] as CP).collect();
            let ksb: Vec<P> = kb.iter().map(|k| o.bn(k)).collect();
            let multi_ks: Vec<CP> = (0..64).map(|j| ksb[j % NK] as CP).collect();
            Some(Ref {
                o,
                g,
                ctx,
                pts,
                ks: ksb,
                ks2: kb2.iter().map(|k| o.bn(k)).collect(),
                order,
                _cof: cof,
                comp: comp.clone(),
                unc: unc.clone(),
                multi_pts,
                multi_ks,
            })
        })
        .collect();
    let fresh = |r: &Ref| r.o.point(r.g);

    // Checks against each OpenSSL.
    for r in &rs {
        let o = r.o;
        let out = fresh(r);
        for i in 0..4 {
            // SAFETY: OpenSSL on its own objects of this group.
            unsafe { (o.EC_POINT_mul)(r.g, out, core::ptr::null(), r.pts[i], r.ks[i], r.ctx) };
            assert_eq!(o.encode(r.g, out, ossl::UNCOMPRESSED, r.ctx), (&ps[i] * &ks[i]).to_bytes(), "{name} mul");
            // SAFETY: as above.
            unsafe { (o.EC_POINT_mul)(r.g, out, r.ks[i], core::ptr::null(), core::ptr::null(), r.ctx) };
            assert_eq!(o.encode(r.g, out, ossl::UNCOMPRESSED, r.ctx), d.mul_generator(&ks[i]).to_bytes(), "{name} mul_generator");
            // SAFETY: as above.
            unsafe { (o.EC_POINT_mul)(r.g, out, r.ks[i], r.pts[i], r.ks2[i], r.ctx) };
            assert_eq!(o.encode(r.g, out, ossl::UNCOMPRESSED, r.ctx), d.mul_add(&ks[i], &ks2[i], &ps[i]).to_bytes(), "{name} mul_add");
            // SAFETY: as above.
            unsafe { (o.EC_POINT_add)(r.g, out, r.pts[i], r.pts[i + 1], r.ctx) };
            assert_eq!(o.encode(r.g, out, ossl::UNCOMPRESSED, r.ctx), (&ps[i] + &ps[i + 1]).to_bytes(), "{name} add");
            // SAFETY: as above.
            unsafe { (o.EC_POINT_dbl)(r.g, out, r.pts[i], r.ctx) };
            assert_eq!(o.encode(r.g, out, ossl::UNCOMPRESSED, r.ctx), ps[i].double().to_bytes(), "{name} dbl");
            assert_eq!(o.encode(r.g, r.pts[i], ossl::COMPRESSED, r.ctx), ps[i].compress(), "{name} compress");
            assert_eq!(o.encode(r.g, r.pts[i], ossl::UNCOMPRESSED, r.ctx), ps[i].to_bytes(), "{name} to_bytes");
            let q = o.decode(r.g, &comp[i], r.ctx).unwrap();
            assert_eq!(o.encode(r.g, q, ossl::UNCOMPRESSED, r.ctx), d.point_from_bytes(&comp[i]).unwrap().to_bytes(), "{name} decompress");
            // SAFETY: frees the decoded point.
            unsafe { (o.EC_POINT_free)(q) };
        }
        if let Some(f) = o.EC_POINTs_mul {
            let terms: Vec<(&Number, &BinaryPoint)> = (0..64).map(|j| (&ks[j % NK], &ps[j % NP])).collect();
            // SAFETY: arrays of 64 points and 64 scalars of the group.
            unsafe { f(r.g, out, core::ptr::null(), 64, r.multi_pts.as_ptr(), r.multi_ks.as_ptr(), r.ctx) };
            assert_eq!(o.encode(r.g, out, ossl::UNCOMPRESSED, r.ctx), d.multi_mul(&terms).to_bytes(), "{name} multi_mul");
        }
        // SAFETY: frees the point.
        unsafe { (o.EC_POINT_free)(out) };
    }

    let kp = format!("{label}, {nbits}-bit scalar < n, point of the subgroup");
    b.run("mul", &kp, "", |i| {
        black_box(&ps[i % NP] * &ks[i % NK]);
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: OpenSSL on its own objects of this group.
        unsafe { (r.o.EC_POINT_mul)(r.g, out, core::ptr::null(), r.pts[i % NP], r.ks[i % NK], r.ctx) };
    }), "EC_POINT_mul(P, k): ladder, affine result"));
    b.run("mul_generator", &format!("{label}, {nbits}-bit scalar < n"), "table built once", |i| {
        black_box(d.mul_generator(&ks[i % NK]));
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe { (r.o.EC_POINT_mul)(r.g, out, r.ks[i % NK], core::ptr::null(), core::ptr::null(), r.ctx) };
    }), "EC_POINT_mul(G, k): ladder, no precomputation"));
    b.run("mul_add", &format!("{label}, a·G + b·P, {nbits}-bit scalars"), "", |i| {
        black_box(d.mul_add(&ks[i % NK], &ks2[i % NK], &ps[i % NP]));
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe { (r.o.EC_POINT_mul)(r.g, out, r.ks[i % NK], r.pts[i % NP], r.ks2[i % NK], r.ctx) };
    }), "EC_POINT_mul(G, a, P, b): wNAF"));
    {
        let terms: Vec<(&Number, &BinaryPoint)> = (0..64).map(|j| (&ks[(j * 7) % NK], &ps[j % NP])).collect();
        let rr: Vec<&Ref> = rs.iter().filter(|r| r.o.EC_POINTs_mul.is_some()).collect();
        let refs: Vec<Side> = rr
            .iter()
            .map(|r| {
                let out = r.o.point(r.g);
                let pts: Vec<CP> = (0..64).map(|j| r.pts[j % NP] as CP).collect();
                let kk: Vec<CP> = (0..64).map(|j| r.ks[(j * 7) % NK] as CP).collect();
                side(&r.o.name, "EC_POINTs_mul (deprecated API), 64 terms", move |_| {
                    // SAFETY: arrays of 64 points and scalars of the group.
                    unsafe { r.o.EC_POINTs_mul.unwrap()(r.g, out, core::ptr::null(), 64, pts.as_ptr(), kk.as_ptr(), r.ctx) };
                })
            })
            .collect();
        b.run("multi_mul", &format!("{label}, 64 terms, {nbits}-bit scalars"), "", |_| {
            black_box(d.multi_mul(&terms));
        }, refs);
    }
    let ap = format!("{label}, affine inputs from bytes");
    b.run("add", &ap, "López–Dahab result, no inversion", |i| {
        black_box(&ps[i % NP] + &ps[(i / NP + i + 1) % NP]);
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe { (r.o.EC_POINT_add)(r.g, out, r.pts[i % NP], r.pts[(i / NP + i + 1) % NP], r.ctx) };
    }), "EC_POINT_add: affine result (one field division)"));
    b.run("add+coordinates", &ap, "the sum then its affine coordinates (the output form of OpenSSL)", |i| {
        black_box((&ps[i % NP] + &ps[(i / NP + i + 1) % NP]).coordinates());
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe { (r.o.EC_POINT_add)(r.g, out, r.pts[i % NP], r.pts[(i / NP + i + 1) % NP], r.ctx) };
    }), "EC_POINT_add"));
    b.run("double", &ap, "López–Dahab result, no inversion", |i| {
        black_box(ps[i % NP].double());
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe { (r.o.EC_POINT_dbl)(r.g, out, r.pts[i % NP], r.ctx) };
    }), "EC_POINT_dbl: affine result (one field division)"));
    b.run("compress", &ap, "", |i| {
        black_box(ps[i % NP].compress());
    }, mk(&rs, Box::new(|r, _out, i| {
        black_box(r.o.encode(r.g, r.pts[i % NP], ossl::COMPRESSED, r.ctx));
    }), "EC_POINT_point2oct compressed"));
    b.run("to_bytes", &ap, "", |i| {
        black_box(ps[i % NP].to_bytes());
    }, mk(&rs, Box::new(|r, _out, i| {
        black_box(r.o.encode(r.g, r.pts[i % NP], ossl::UNCOMPRESSED, r.ctx));
    }), "EC_POINT_point2oct uncompressed"));
    let dp = format!("{label}, checked (on the curve)");
    b.run("point_from_bytes compressed", &dp, "", |i| {
        black_box(d.point_from_bytes(&comp[i % NP]).unwrap());
    }, mk(&rs, Box::new(|r, out, i| {
        let c = &r.comp[i % NP];
        // SAFETY: reads c into the point.
        assert_eq!(unsafe { (r.o.EC_POINT_oct2point)(r.g, out, c.as_ptr(), c.len(), r.ctx) }, 1);
    }), "EC_POINT_oct2point"));
    b.run("point_from_bytes uncompressed", &dp, "", |i| {
        black_box(d.point_from_bytes(&unc[i % NP]).unwrap());
    }, mk(&rs, Box::new(|r, out, i| {
        let c = &r.unc[i % NP];
        // SAFETY: reads c into the point.
        assert_eq!(unsafe { (r.o.EC_POINT_oct2point)(r.g, out, c.as_ptr(), c.len(), r.ctx) }, 1);
    }), "EC_POINT_oct2point"));
    b.run("in_subgroup", &format!("{label}, points of the subgroup"), "n·P = O", |i| {
        black_box(d.in_subgroup(&ps[i % NP]));
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe {
            (r.o.EC_POINT_mul)(r.g, out, core::ptr::null(), r.pts[i % NP], r.order, r.ctx);
            assert_eq!((r.o.EC_POINT_is_at_infinity)(r.g, out), 1);
        }
    }), "EC_POINT_mul by the order + EC_POINT_is_at_infinity"));
    let h = d.cofactor().to_str_radix(10);
    let dbls = if h == "2" { 1 } else { assert_eq!(h, "4"); 2 };
    b.run("clear_cofactor", &format!("{label}, cofactor {h}"), "", |i| {
        black_box(d.clear_cofactor(&ps[i % NP]));
    }, mk(&rs, Box::new(move |r, out, i| {
        // SAFETY: as above.
        unsafe {
            (r.o.EC_POINT_dbl)(r.g, out, r.pts[i % NP], r.ctx);
            if dbls == 2 {
                (r.o.EC_POINT_dbl)(r.g, out, out, r.ctx);
            }
        }
    }), "EC_POINT_dbl once or twice (h = 2 or 4): its fastest way; EC_POINT_mul by h runs a full-length ladder"));
    let xs: Vec<_> = ps.iter().map(|p| p.coordinates().unwrap().0).collect();
    b.run("x_mul", &kp, "x only, López–Dahab ladder", |i| {
        black_box(d.x_mul(&xs[i % NP], &ks[i % NK]));
    }, mk(&rs, Box::new(|r, out, i| {
        // SAFETY: as above.
        unsafe { (r.o.EC_POINT_mul)(r.g, out, core::ptr::null(), r.pts[i % NP], r.ks[i % NK], r.ctx) };
    }), "EC_POINT_mul: the same ladder plus y recovery (does more)"));
    // Ours only.
    let prep: Vec<BinaryPrepared> = ps.iter().take(4).map(|p| p.prepare(nbits as usize)).collect();
    b.run("prepared.mul", &kp, "a prepared point of the pool (prepare(bits) done once)", |i| {
        black_box(prep[i % 4].mul(&ks[i % NK]));
    }, vec![]);
    let pj: Vec<BinaryPoint> = (0..NP).map(|i| &ps[i] * &ks[i]).collect();
    b.run("coordinates", &format!("{label}, López–Dahab input"), "one inversion", |i| {
        black_box(pj[i % NP].coordinates());
    }, vec![]);
    b.run("prepare", &format!("{label}, bits = {nbits}"), "table of signed 4-bit windows", |i| {
        black_box(ps[i % NP].prepare(nbits as usize));
    }, vec![]);
    let _ = &mut Some(());
}
