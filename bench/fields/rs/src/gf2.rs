//! `Gf2_8` and `Gf2_128`: products, inverses and powers against NTL (GF2E), OpenSSL (BN_GF2m) and
//! FLINT (fq_nmod over GF(2)); GHASH against RustCrypto's ghash and polyval and OpenSSL's
//! AES-GCM (its additional data, which runs GHASH alone).

use crate::common::{self, Bench, Cmp};
use crate::ffi;
use crate::harness::{self, Op, Rng, Side, keep};
use a::math::field::{Gf2_8, Gf2_128};
use ghash::universal_hash::UniversalHash;

const COUNT: usize = 256;

struct Ours8 {
    coef: bool,
    xs: Vec<Gf2_8>,
    e: u64,
    acc: Gf2_8,
}

impl Ours8 {
    fn l(&self, x: Gf2_8) -> Vec<u64> {
        if self.coef { (0..8).map(|i| (x.to_byte() >> i) as u64 & 1).collect() } else { vec![x.to_byte() as u64] }
    }
}

impl Bench for Ours8 {
    fn name(&self) -> String {
        "ours".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (self.xs[i], self.xs[(i + 1) % COUNT]);
        Some(match op {
            Op::Mul => (self.l(x * y), 0),
            Op::Sqr => (self.l(x * x), 0),
            Op::Inv => (self.l(x.inverse().unwrap()), 0),
            Op::Div => (self.l(x / y), 0),
            Op::PowU | Op::Pow => (self.l(x.pow(self.e)), 0),
            Op::Sbox => (self.l(x.aes_sbox()), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut a = self.acc;
        let xs = &self.xs;
        match op {
            Op::Mul => (0..iters).for_each(|i| a *= xs[i as usize & m]),
            Op::Sqr => (0..iters).for_each(|_| a = a * a),
            Op::Inv => (0..iters).for_each(|i| a += xs[i as usize & m].inverse().unwrap()),
            Op::Div => (0..iters).for_each(|i| a /= xs[i as usize & m]),
            Op::PowU | Op::Pow => (0..iters).for_each(|i| a += xs[i as usize & m].pow(self.e)),
            Op::Sbox => (0..iters).for_each(|_| a = a.aes_sbox()),
            _ => {}
        }
        if a == Gf2_8::ZERO {
            a = xs[0];
        }
        self.acc = keep(a);
    }
}

struct Ours128 {
    xs: Vec<Gf2_128>,
    e: u64,
    acc: Gf2_128,
}

fn w128(x: Gf2_128) -> Vec<u64> {
    let p = x.polynomial();
    vec![p as u64, (p >> 64) as u64]
}

impl Bench for Ours128 {
    fn name(&self) -> String {
        "ours".into()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        let (x, y) = (self.xs[i], self.xs[(i + 1) % COUNT]);
        Some(match op {
            Op::Mul => (w128(x * y), 0),
            Op::Clmul => (w128(x.mul_with(y, a::math::field::Clmul::Portable)), 0),
            Op::Sqr => (w128(x.square()), 0),
            Op::Inv => (w128(x.inverse().unwrap()), 0),
            Op::Div => (w128(x / y), 0),
            Op::PowU => (w128(x.pow(self.e as u128)), 0),
            _ => return None,
        })
    }
    fn run(&mut self, op: Op, iters: u64) {
        let m = COUNT - 1;
        let mut a = self.acc;
        let xs = &self.xs;
        match op {
            Op::Mul => (0..iters).for_each(|i| a = a * xs[i as usize & m]),
            Op::Clmul => (0..iters).for_each(|i| a = a.mul_with(xs[i as usize & m], a::math::field::Clmul::Portable)),
            Op::Sqr => (0..iters).for_each(|_| a = a.square()),
            Op::Inv => (0..iters).for_each(|i| a = a + xs[i as usize & m].inverse().unwrap()),
            Op::Div => (0..iters).for_each(|i| a = a / xs[i as usize & m]),
            Op::PowU => (0..iters).for_each(|i| a = a + xs[i as usize & m].pow(self.e as u128)),
            _ => {}
        }
        if a.is_zero() {
            a = xs[0];
        }
        self.acc = keep(a);
    }
}

/// A reference's product, asked for under the name of our portable product.
struct Portable(crate::ffi::CRef);

impl Bench for Portable {
    fn name(&self) -> String {
        self.0.name.to_string()
    }
    fn out(&mut self, op: Op, i: usize) -> Option<(Vec<u64>, i32)> {
        self.0.out(if op == Op::Clmul { Op::Mul } else { op }, i)
    }
    fn run(&mut self, op: Op, iters: u64) {
        self.0.run(if op == Op::Clmul { Op::Mul } else { op }, iters)
    }
    fn note(&self, _op: Op) -> String {
        "the reference's own product".into()
    }
}

/// Runs the rows of `Gf2_8` and `Gf2_128`.
pub fn run(only: &[String]) {
    let want = |s: &str| only.is_empty() || only.iter().any(|o| o == s);
    if want("gf2_8") {
        gf2_8();
    }
    if want("gf2_128") {
        gf2_128();
    }
    if want("ghash") {
        ghash();
    }
}

fn gf2_8() {
    let group = "gf2_8";
    let mut rng = Rng(0x8);
    let bytes: Vec<u8> = (0..COUNT).map(|_| (rng.next() % 255 + 1) as u8).collect();
    let e = rng.next() >> 1;
    let xs: Vec<Gf2_8> = bytes.iter().map(|&b| Gf2_8::new(b)).collect();
    let exps = [8u32, 4, 3, 1, 0];
    // as one word: NTL GF2E and OpenSSL BN_GF2m
    let words: Vec<u64> = bytes.iter().map(|&b| b as u64).collect();
    let mut sides: Vec<Box<dyn Bench>> = vec![Box::new(Ours8 { coef: false, xs: xs.clone(), e, acc: xs[0] })];
    for r in ffi::bin_refs() {
        if let Some(c) = ffi::bin_new(&r, &exps, &words, COUNT, 1, e) {
            sides.push(Box::new(c));
        }
    }
    // as 8 coefficients: FLINT fq_nmod over GF(2)
    let coefs: Vec<u64> = bytes.iter().flat_map(|&b| (0..8).map(move |i| (b >> i) as u64 & 1)).collect();
    let g: Vec<u64> = (0..=8).map(|i| [1, 1, 0, 1, 1, 0, 0, 0, 1][i]).collect();
    let mut csides: Vec<Box<dyn Bench>> = vec![Box::new(Ours8 { coef: true, xs: xs.clone(), e, acc: xs[0] })];
    for r in ffi::ext_refs() {
        if let Some(c) = ffi::ext_new(&r, &[2], &g, 8, &coefs, COUNT, &[e]) {
            csides.push(Box::new(c));
        }
    }
    let params = "AES's GF(2^8) = GF(2)[x]/(x^8+x^4+x^3+x+1); log/exp tables";
    let ops = [(Op::Mul, "mul"), (Op::Sqr, "square"), (Op::Inv, "inverse"), (Op::Div, "div"), (Op::PowU, "pow")];
    for (op, name) in ops {
        let pr = if op == Op::PowU { format!("{params}; 63-bit exponent") } else { params.to_string() };
        common::compare(group, name, &pr, &mut sides, op, 64, Cmp::Exact, &[0]);
        let op2 = if op == Op::PowU { Op::Pow } else { op };
        if op != Op::Div {
            common::compare(group, name, &pr, &mut csides, op2, 64, Cmp::Exact, &[0]);
        }
    }
    let mut ours: Vec<Box<dyn Bench>> = vec![Box::new(Ours8 { coef: false, xs: xs.clone(), e, acc: xs[0] })];
    common::compare(group, "aes_sbox", "FIPS 197's S-box: the inverse and the affine map; a dependent chain", &mut ours, Op::Sbox, 8, Cmp::Exact, &[0]);
}

fn gf2_128() {
    let group = "gf2_128";
    let mut rng = Rng(0x128);
    let xs: Vec<Gf2_128> =
        (0..COUNT).map(|_| Gf2_128::from_polynomial(((rng.next() as u128) << 64) | rng.next() as u128 | 1)).collect();
    let e = rng.next() >> 1;
    let words: Vec<u64> = xs.iter().flat_map(|&x| w128(x)).collect();
    let mut sides: Vec<Box<dyn Bench>> = vec![Box::new(Ours128 { xs: xs.clone(), e, acc: xs[0] })];
    for r in ffi::bin_refs() {
        if let Some(c) = ffi::bin_new(&r, &[128, 7, 2, 1, 0], &words, COUNT, 2, e) {
            sides.push(Box::new(c));
        }
    }
    let params = format!("GCM's GF(2^128) = GF(2)[x]/(x^128+x^7+x^2+x+1); kernel {:?}", a::math::field::Clmul::detect());
    for (op, name) in [(Op::Mul, "mul"), (Op::Sqr, "square"), (Op::Inv, "inverse"), (Op::Div, "div"), (Op::PowU, "pow")] {
        let pr = if op == Op::PowU { format!("{params}; 63-bit exponent") } else { params.clone() };
        common::compare(group, name, &pr, &mut sides, op, 32, Cmp::Exact, &[0, 0]);
    }
    // our portable kernel (CPUs without PMULL or PCLMULQDQ), against the same references' products
    let mut psides: Vec<Box<dyn Bench>> = vec![Box::new(Ours128 { xs: xs.clone(), e, acc: xs[0] })];
    for r in ffi::bin_refs() {
        if let Some(c) = ffi::bin_new(&r, &[128, 7, 2, 1, 0], &words, COUNT, 2, e) {
            psides.push(Box::new(Portable(c)));
        }
    }
    common::compare(group, "mul_with(Clmul::Portable)", &params, &mut psides, Op::Clmul, 32, Cmp::Exact, &[0, 0]);
}

/// POLYVAL's field element for GHASH's, RFC 8452 appendix A: the reversed bytes times x.
fn mulx_polyval(b: [u8; 16]) -> [u8; 16] {
    let mut v = u128::from_le_bytes(b);
    let carry = v >> 127;
    v <<= 1;
    if carry == 1 {
        v ^= 0xc2000000000000000000000000000001;
    }
    v.to_le_bytes()
}

fn rev16(b: &[u8]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = b[15 - i];
    }
    r
}

fn ghash() {
    let group = "gf2_128";
    let mut rng = Rng(0x6ca);
    let key: [u8; 16] = core::array::from_fn(|_| rng.next() as u8);
    let iv: [u8; 12] = core::array::from_fn(|_| rng.next() as u8);
    #[allow(unused_mut)]
    let mut hb: [u8; 16] = core::array::from_fn(|_| rng.next() as u8);
    #[allow(unused_mut, unused_variables)]
    let mut ej0 = [0u8; 16];
    #[cfg(has_ossl)]
    // SAFETY: arrays of the sizes the shim reads and writes.
    unsafe {
        assert_eq!(ffi::ossl_gcm_keys(key.as_ptr(), iv.as_ptr(), hb.as_mut_ptr(), ej0.as_mut_ptr()), 1);
    }
    let h = Gf2_128::from_bytes(hb);
    let keyed = ghash::GHash::new(&ghash::Key::from(hb));
    let pkey = mulx_polyval(rev16(&hb));
    let pkeyed = polyval::Polyval::new(&polyval::Key::from(pkey));
    for len in [16usize, 64, 256, 1024, 16384, 1 << 20] {
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let rdata: Vec<u8> = data.chunks(16).flat_map(rev16).collect();
        // the checks
        let ours = Gf2_128::ghash(h, &data).to_bytes();
        let mut g = keyed.clone();
        g.update_padded(&data);
        let theirs: [u8; 16] = g.finalize().into();
        let mut names = vec!["ours".to_string()];
        let mut notes = vec![String::new()];
        let gh_ok = ours == theirs;
        let mut pv = pkeyed.clone();
        pv.update_padded(&rdata);
        let pv_out: [u8; 16] = pv.finalize().into();
        let pv_ok = rev16(&pv_out) == ours;
        let mut sides: Vec<Side> = vec![Side::new("ours", |n| {
            let mut a = Gf2_128::ZERO;
            for _ in 0..n {
                a = a + Gf2_128::ghash(h, &data);
            }
            keep(a);
        })];
        names.push("RustCrypto ghash 0.6".into());
        notes.push(if gh_ok { "keyed state cloned per message".into() } else { harness::mismatch("ghash crate") });
        sides.push(Side::new("ghash", |n| {
            for _ in 0..n {
                let mut g = keyed.clone();
                g.update_padded(&data);
                keep(g.finalize());
            }
        }));
        names.push("RustCrypto polyval 0.7".into());
        notes.push(if pv_ok {
            "POLYVAL on the byte-reversed blocks (RFC 8452), checked equal; keyed state cloned per message".into()
        } else {
            harness::mismatch("polyval crate")
        });
        sides.push(Side::new("polyval", |n| {
            for _ in 0..n {
                let mut g = pkeyed.clone();
                g.update_padded(&rdata);
                keep(g.finalize());
            }
        }));
        #[cfg(has_ossl)]
        let gcm = {
            // check: OpenSSL's tag over `data` as additional data = GHASH(H, A, {}) + E(K, J0)
            let mut tag = [0u8; 16];
            // SAFETY: the buffers are as long as given.
            unsafe { ffi::ossl_gcm_tag(key.as_ptr(), iv.as_ptr(), data.as_ptr(), data.len(), tag.as_mut_ptr()) };
            let want = Gf2_128::gcm_ghash(h, &data, &[]) + Gf2_128::from_bytes(ej0);
            names.push("OpenSSL AES-GCM".into());
            notes.push(if want.to_bytes() == tag {
                "EVP_EncryptUpdate of the bytes as additional data (GHASH alone) on one running message; tag checked"
                    .into()
            } else {
                harness::mismatch("OpenSSL GCM tag")
            });
            // SAFETY: a fresh context, freed below.
            unsafe { ffi::ossl_gcm_new(key.as_ptr(), iv.as_ptr()) }
        };
        #[cfg(has_ossl)]
        sides.push(Side::new("openssl", |n| {
            // SAFETY: a live context and a live buffer.
            unsafe { ffi::ossl_gcm_aad(gcm, data.as_ptr(), data.len(), n) }
        }));
        let stats = harness::measure(&mut sides);
        drop(sides);
        #[cfg(has_ossl)]
        // SAFETY: made above, freed once.
        unsafe {
            ffi::ossl_gcm_free(gcm)
        };
        let gbs = |ns: f64| format!("{:.2} GB/s", len as f64 / ns);
        let notes: Vec<String> = notes
            .iter()
            .enumerate()
            .map(|(k, n)| if k == 0 { format!("ours {}", gbs(stats[0].median)) } else { format!("{n}; ref {}", gbs(stats[k].median)) })
            .collect();
        harness::report(group, "ghash", &format!("{len} bytes a message (H set once)"), &names, &stats, &notes);
    }
    // GCM's GHASH(H, A, C) with the lengths block
    let a_: Vec<u8> = (0..16).map(|_| rng.next() as u8).collect();
    let c_: Vec<u8> = (0..1024).map(|_| rng.next() as u8).collect();
    let mut lens = [0u8; 16];
    lens[..8].copy_from_slice(&(128u64).to_be_bytes());
    lens[8..].copy_from_slice(&(8192u64).to_be_bytes());
    let mut g = keyed.clone();
    g.update_padded(&a_);
    g.update_padded(&c_);
    g.update_padded(&lens);
    let theirs: [u8; 16] = g.finalize().into();
    let ok = Gf2_128::gcm_ghash(h, &a_, &c_).to_bytes() == theirs;
    let mut sides: Vec<Side> = vec![
        Side::new("ours", |n| {
            for _ in 0..n {
                keep(Gf2_128::gcm_ghash(h, &a_, &c_));
            }
        }),
        Side::new("ghash", |n| {
            for _ in 0..n {
                let mut g = keyed.clone();
                g.update_padded(&a_);
                g.update_padded(&c_);
                g.update_padded(&lens);
                keep(g.finalize());
            }
        }),
    ];
    let stats = harness::measure(&mut sides);
    let note = if ok { "update_padded(A), update_padded(C), the lengths block, finalize".to_string() } else { harness::mismatch("gcm_ghash") };
    harness::report(
        group,
        "gcm_ghash",
        "A of 16 bytes, C of 1024 bytes",
        &["ours".into(), "RustCrypto ghash 0.6".into()],
        &stats,
        &[String::new(), note],
    );
}
