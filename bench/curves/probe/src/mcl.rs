// The few functions of mcl's C API (include/mcl/bn.h) the benchmarks call, for the library built
// with its default MCL_FP_BIT=384, MCL_FR_BIT=256 (one build covers BN254 and BLS12-381):
// MCLBN_FP_UNIT_SIZE = 6, MCLBN_FR_UNIT_SIZE = 4, MCLBN_COMPILED_TIME_VAR = 46.
#![allow(non_snake_case, non_camel_case_types, dead_code)]

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Fr {
    pub d: [u64; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Fp {
    pub d: [u64; 6],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Fp2 {
    pub d: [Fp; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct G1 {
    pub x: Fp,
    pub y: Fp,
    pub z: Fp,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct G2 {
    pub x: Fp2,
    pub y: Fp2,
    pub z: Fp2,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GT {
    pub d: [Fp; 12],
}
impl Default for GT {
    fn default() -> Self {
        GT { d: [Fp::default(); 12] }
    }
}

pub const BN_SNARK1: i32 = 4;
pub const BLS12_381: i32 = 5;
pub const COMPILED_TIME_VAR: i32 = 46;
pub const MAP_TO_MODE_HASH_TO_CURVE: i32 = 5;

unsafe extern "C" {
    pub fn mclBn_getVersion() -> i32;
    pub fn mclBn_init(curve: i32, ctv: i32) -> i32;
    pub fn mclBn_setETHserialization(enable: i32);
    pub fn mclBn_setMapToMode(mode: i32) -> i32;
    pub fn mclBn_verifyOrderG1(v: i32);
    pub fn mclBn_verifyOrderG2(v: i32);
    pub fn mclBnFr_setLittleEndian(x: *mut Fr, buf: *const u8, n: usize) -> i32;
    pub fn mclBnFp_setBigEndianMod(x: *mut Fp, buf: *const u8, n: usize) -> i32;
    pub fn mclBnG1_setStr(x: *mut G1, buf: *const u8, n: usize, mode: i32) -> i32;
    pub fn mclBnG2_setStr(x: *mut G2, buf: *const u8, n: usize, mode: i32) -> i32;
    pub fn mclBnG1_getStr(buf: *mut u8, n: usize, x: *const G1, mode: i32) -> usize;
    pub fn mclBnG2_getStr(buf: *mut u8, n: usize, x: *const G2, mode: i32) -> usize;
    pub fn mclBnGT_getStr(buf: *mut u8, n: usize, x: *const GT, mode: i32) -> usize;
    pub fn mclBnG1_serialize(buf: *mut u8, n: usize, x: *const G1) -> usize;
    pub fn mclBnG2_serialize(buf: *mut u8, n: usize, x: *const G2) -> usize;
    pub fn mclBnG1_deserialize(x: *mut G1, buf: *const u8, n: usize) -> usize;
    pub fn mclBnG2_deserialize(x: *mut G2, buf: *const u8, n: usize) -> usize;
    pub fn mclBnG1_isValid(x: *const G1) -> i32;
    pub fn mclBnG2_isValid(x: *const G2) -> i32;
    pub fn mclBnG1_isValidOrder(x: *const G1) -> i32;
    pub fn mclBnG2_isValidOrder(x: *const G2) -> i32;
    pub fn mclBnG1_isEqual(x: *const G1, y: *const G1) -> i32;
    pub fn mclBnG2_isEqual(x: *const G2, y: *const G2) -> i32;
    pub fn mclBnG1_add(z: *mut G1, x: *const G1, y: *const G1);
    pub fn mclBnG2_add(z: *mut G2, x: *const G2, y: *const G2);
    pub fn mclBnG1_dbl(y: *mut G1, x: *const G1);
    pub fn mclBnG2_dbl(y: *mut G2, x: *const G2);
    pub fn mclBnG1_neg(y: *mut G1, x: *const G1);
    pub fn mclBnG2_neg(y: *mut G2, x: *const G2);
    pub fn mclBnG1_normalize(y: *mut G1, x: *const G1);
    pub fn mclBnG2_normalize(y: *mut G2, x: *const G2);
    pub fn mclBnG1_mul(z: *mut G1, x: *const G1, y: *const Fr);
    pub fn mclBnG2_mul(z: *mut G2, x: *const G2, y: *const Fr);
    pub fn mclBnG1_hashAndMapToWithDst(x: *mut G1, buf: *const u8, n: usize, dst: *const u8, dn: usize) -> i32;
    pub fn mclBnG2_hashAndMapToWithDst(x: *mut G2, buf: *const u8, n: usize, dst: *const u8, dn: usize) -> i32;
    pub fn mclBnGT_isEqual(x: *const GT, y: *const GT) -> i32;
    pub fn mclBnGT_isValid(x: *const GT) -> i32;
    pub fn mclBnGT_mul(z: *mut GT, x: *const GT, y: *const GT);
    pub fn mclBnGT_sqr(y: *mut GT, x: *const GT);
    pub fn mclBnGT_inv(y: *mut GT, x: *const GT);
    pub fn mclBnGT_pow(z: *mut GT, x: *const GT, y: *const Fr);
    pub fn mclBn_pairing(z: *mut GT, x: *const G1, y: *const G2);
    pub fn mclBn_millerLoop(z: *mut GT, x: *const G1, y: *const G2);
    pub fn mclBn_millerLoopVec(z: *mut GT, x: *const G1, y: *const G2, n: usize);
    pub fn mclBn_finalExp(y: *mut GT, x: *const GT);
    pub fn mclBnFp_mul(z: *mut Fp, x: *const Fp, y: *const Fp);
    pub fn mclBnFp2_mul(z: *mut Fp2, x: *const Fp2, y: *const Fp2);
}

/// Initialises mcl for one curve (it keeps one curve per process).
pub fn init(curve: i32) {
    // SAFETY: mcl's initialisation, called once before any other mcl function.
    unsafe {
        assert_eq!(mclBn_init(curve, COMPILED_TIME_VAR), 0, "mclBn_init failed");
    }
}

pub fn version() -> String {
    // SAFETY: returns an int.
    let v = unsafe { mclBn_getVersion() };
    // 0xABCD = A.BC.D (include/mcl/op.hpp)
    format!("{:x}.{:02x}.{:x}", v >> 12, (v >> 4) & 0xff, v & 0xf)
}

fn get(f: impl Fn(*mut u8, usize) -> usize) -> String {
    let mut b = vec![0u8; 4096];
    let n = f(b.as_mut_ptr(), b.len());
    assert!(n > 0, "mcl getStr failed");
    b.truncate(n);
    String::from_utf8(b).unwrap()
}

/// "1 x y" (affine) in base 16, as mcl writes it with ioMode 16 after normalising, or "0".
pub fn g1_str(x: &G1) -> String {
    // SAFETY: writes at most n bytes into the buffer.
    get(|b, n| unsafe { mclBnG1_getStr(b, n, x, 16) })
}
pub fn g2_str(x: &G2) -> String {
    // SAFETY: as above.
    get(|b, n| unsafe { mclBnG2_getStr(b, n, x, 16) })
}
pub fn gt_str(x: &GT) -> String {
    // SAFETY: as above.
    get(|b, n| unsafe { mclBnGT_getStr(b, n, x, 16) })
}

pub fn g1_from_str(s: &str) -> G1 {
    let mut p = G1::default();
    // SAFETY: reads n bytes of s.
    let r = unsafe { mclBnG1_setStr(&mut p, s.as_ptr(), s.len(), 16) };
    assert_eq!(r, 0, "mclBnG1_setStr failed on {s}");
    p
}
pub fn g2_from_str(s: &str) -> G2 {
    let mut p = G2::default();
    // SAFETY: as above.
    let r = unsafe { mclBnG2_setStr(&mut p, s.as_ptr(), s.len(), 16) };
    assert_eq!(r, 0, "mclBnG2_setStr failed on {s}");
    p
}
pub fn fr_le(b: &[u8]) -> Fr {
    let mut x = Fr::default();
    // SAFETY: reads b.
    let r = unsafe { mclBnFr_setLittleEndian(&mut x, b.as_ptr(), b.len()) };
    assert_eq!(r, 0);
    x
}
