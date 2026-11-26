//! The parts of GMP's C interface the benchmarks call (mpn_* on raw limbs, mpz_* on numbers), and
//! of OpenSSL's BIGNUM. 64-bit limbs, LP64 (macOS and Linux on aarch64 and x86_64).

#![allow(dead_code, non_camel_case_types)]

use std::ffi::{c_char, c_int, c_long, c_ulong, c_void};

#[repr(C)]
pub struct Mpz {
    alloc: c_int,
    size: c_int,
    d: *mut u64,
}

#[cfg(has_gmp)]
unsafe extern "C" {
    pub fn __gmpn_add_n(rp: *mut u64, a: *const u64, b: *const u64, n: c_long) -> u64;
    pub fn __gmpn_sub_n(rp: *mut u64, a: *const u64, b: *const u64, n: c_long) -> u64;
    pub fn __gmpn_mul_n(rp: *mut u64, a: *const u64, b: *const u64, n: c_long);
    /// Internal (gmp-impl.h) but exported: the low n limbs of the product.
    pub fn __gmpn_mullo_n(rp: *mut u64, a: *const u64, b: *const u64, n: c_long);
    pub fn __gmpn_sqr(rp: *mut u64, a: *const u64, n: c_long);
    pub fn __gmpn_mul_1(rp: *mut u64, a: *const u64, n: c_long, b: u64) -> u64;
    pub fn __gmpn_tdiv_qr(
        qp: *mut u64,
        rp: *mut u64,
        qxn: c_long,
        np: *const u64,
        nn: c_long,
        dp: *const u64,
        dn: c_long,
    );
    pub fn __gmpn_divrem_1(qp: *mut u64, qxn: c_long, np: *const u64, nn: c_long, d: u64) -> u64;
    pub fn __gmpn_cmp(a: *const u64, b: *const u64, n: c_long) -> c_int;
    pub fn __gmpn_sqrtrem(r1: *mut u64, r2: *mut u64, sp: *const u64, n: c_long) -> c_long;
    pub fn __gmpn_lshift(rp: *mut u64, a: *const u64, n: c_long, cnt: u32) -> u64;
    pub fn __gmpn_rshift(rp: *mut u64, a: *const u64, n: c_long, cnt: u32) -> u64;
    pub fn __gmpn_sec_powm(
        rp: *mut u64,
        bp: *const u64,
        bn: c_long,
        ep: *const u64,
        enb: c_ulong,
        mp: *const u64,
        n: c_long,
        tp: *mut u64,
    );
    pub fn __gmpn_sec_powm_itch(bn: c_long, enb: c_ulong, n: c_long) -> c_long;

    pub fn __gmpz_init(x: *mut Mpz);
    pub fn __gmpz_init2(x: *mut Mpz, bits: c_ulong);
    pub fn __gmpz_clear(x: *mut Mpz);
    pub fn __gmpz_import(
        r: *mut Mpz,
        count: usize,
        order: c_int,
        size: usize,
        endian: c_int,
        nails: usize,
        op: *const c_void,
    );
    pub fn __gmpz_export(
        r: *mut c_void,
        count: *mut usize,
        order: c_int,
        size: usize,
        endian: c_int,
        nails: usize,
        op: *const Mpz,
    ) -> *mut c_void;
    pub fn __gmpz_add(r: *mut Mpz, a: *const Mpz, b: *const Mpz);
    pub fn __gmpz_sub(r: *mut Mpz, a: *const Mpz, b: *const Mpz);
    pub fn __gmpz_mul(r: *mut Mpz, a: *const Mpz, b: *const Mpz);
    pub fn __gmpz_mod(r: *mut Mpz, a: *const Mpz, m: *const Mpz);
    pub fn __gmpz_tdiv_qr(q: *mut Mpz, r: *mut Mpz, n: *const Mpz, d: *const Mpz);
    pub fn __gmpz_tdiv_q_ui(q: *mut Mpz, n: *const Mpz, d: c_ulong) -> c_ulong;
    pub fn __gmpz_powm(r: *mut Mpz, b: *const Mpz, e: *const Mpz, m: *const Mpz);
    pub fn __gmpz_invert(r: *mut Mpz, a: *const Mpz, m: *const Mpz) -> c_int;
    pub fn __gmpz_sqrt(r: *mut Mpz, a: *const Mpz);
    pub fn __gmpz_pow_ui(r: *mut Mpz, b: *const Mpz, e: c_ulong);
    pub fn __gmpz_get_str(s: *mut c_char, base: c_int, a: *const Mpz) -> *mut c_char;
    pub fn __gmpz_set_str(r: *mut Mpz, s: *const c_char, base: c_int) -> c_int;
    pub fn __gmpz_cmp(a: *const Mpz, b: *const Mpz) -> c_int;
    pub fn __gmpz_mul_2exp(r: *mut Mpz, a: *const Mpz, k: c_ulong);
    pub fn __gmpz_fdiv_q_2exp(r: *mut Mpz, a: *const Mpz, k: c_ulong);
    pub fn __gmpz_tdiv_r_2exp(r: *mut Mpz, a: *const Mpz, k: c_ulong);
}

/// An mpz_t that clears itself.
#[cfg(has_gmp)]
pub struct Z(pub Mpz);

#[cfg(has_gmp)]
impl Z {
    pub fn new() -> Z {
        let mut z = Mpz { alloc: 0, size: 0, d: std::ptr::null_mut() };
        unsafe { __gmpz_init2(&mut z, 64 * 160) };
        Z(z)
    }
    pub fn of(l: &[u64]) -> Z {
        let mut z = Z::new();
        unsafe { __gmpz_import(&mut z.0, l.len(), -1, 8, 0, 0, l.as_ptr().cast()) };
        z
    }
    pub fn limbs<const N: usize>(&self) -> [u64; N] {
        let mut out = vec![0u64; N + 160];
        let mut count = 0usize;
        unsafe { __gmpz_export(out.as_mut_ptr().cast(), &mut count, -1, 8, 0, 0, &self.0) };
        assert!(count <= N, "GMP's result does not fit");
        out[..N].try_into().unwrap()
    }
    pub fn p(&self) -> *const Mpz {
        &self.0
    }
    pub fn m(&mut self) -> *mut Mpz {
        &mut self.0
    }
}

#[cfg(has_gmp)]
impl Drop for Z {
    fn drop(&mut self) {
        unsafe { __gmpz_clear(&mut self.0) };
    }
}

// ---------------------------------------------------------------------------------------------
// OpenSSL's BIGNUM (libcrypto 3).

#[repr(C)]
pub struct BIGNUM {
    _p: [u8; 0],
}
#[repr(C)]
pub struct BN_CTX {
    _p: [u8; 0],
}
#[repr(C)]
pub struct BN_MONT_CTX {
    _p: [u8; 0],
}

#[cfg(has_openssl)]
unsafe extern "C" {
    pub fn BN_new() -> *mut BIGNUM;
    pub fn BN_free(a: *mut BIGNUM);
    pub fn BN_CTX_new() -> *mut BN_CTX;
    pub fn BN_CTX_free(c: *mut BN_CTX);
    pub fn BN_lebin2bn(s: *const u8, len: c_int, ret: *mut BIGNUM) -> *mut BIGNUM;
    pub fn BN_bn2lebinpad(a: *const BIGNUM, to: *mut u8, tolen: c_int) -> c_int;
    pub fn BN_mod_exp(
        r: *mut BIGNUM,
        a: *const BIGNUM,
        p: *const BIGNUM,
        m: *const BIGNUM,
        ctx: *mut BN_CTX,
    ) -> c_int;
    pub fn BN_mod_exp_mont(
        r: *mut BIGNUM,
        a: *const BIGNUM,
        p: *const BIGNUM,
        m: *const BIGNUM,
        ctx: *mut BN_CTX,
        mont: *mut BN_MONT_CTX,
    ) -> c_int;
    pub fn BN_mod_inverse(
        r: *mut BIGNUM,
        a: *const BIGNUM,
        n: *const BIGNUM,
        ctx: *mut BN_CTX,
    ) -> *mut BIGNUM;
    pub fn BN_MONT_CTX_new() -> *mut BN_MONT_CTX;
    pub fn BN_MONT_CTX_free(m: *mut BN_MONT_CTX);
    pub fn BN_MONT_CTX_set(m: *mut BN_MONT_CTX, modulus: *const BIGNUM, ctx: *mut BN_CTX) -> c_int;
    pub fn BN_mod_mul_montgomery(
        r: *mut BIGNUM,
        a: *const BIGNUM,
        b: *const BIGNUM,
        m: *mut BN_MONT_CTX,
        ctx: *mut BN_CTX,
    ) -> c_int;
    pub fn BN_mod_mul(
        r: *mut BIGNUM,
        a: *const BIGNUM,
        b: *const BIGNUM,
        m: *const BIGNUM,
        ctx: *mut BN_CTX,
    ) -> c_int;
}

#[cfg(has_openssl)]
pub struct Bn(pub *mut BIGNUM);

#[cfg(has_openssl)]
impl Bn {
    pub fn new() -> Bn {
        Bn(unsafe { BN_new() })
    }
    pub fn of(l: &[u64]) -> Bn {
        let bytes: Vec<u8> = l.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b = Bn::new();
        unsafe { BN_lebin2bn(bytes.as_ptr(), bytes.len() as c_int, b.0) };
        b
    }
    pub fn limbs<const N: usize>(&self) -> [u64; N] {
        let mut bytes = vec![0u8; 8 * N];
        let r = unsafe { BN_bn2lebinpad(self.0, bytes.as_mut_ptr(), (8 * N) as c_int) };
        assert!(r >= 0, "OpenSSL's result does not fit");
        let mut out = [0u64; N];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap());
        }
        out
    }
}

#[cfg(has_openssl)]
impl Drop for Bn {
    fn drop(&mut self) {
        unsafe { BN_free(self.0) };
    }
}
