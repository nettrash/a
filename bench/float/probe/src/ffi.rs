//! Hand-written bindings to GMP, MPFR 4.2, GNU MPC 1.3+ and (with `has_arb`) FLINT 3's Arb.
#![allow(non_camel_case_types, dead_code)]
use std::os::raw::{c_char, c_int, c_long, c_ulong, c_void};

pub type Rnd = c_int;
pub const RN: Rnd = 0;
pub const RZ: Rnd = 1;
pub const RU: Rnd = 2;
pub const RD: Rnd = 3;

#[repr(C)]
pub struct Mpfr {
    pub prec: c_long,
    pub sign: c_int,
    pub exp: c_long,
    pub d: *mut u64,
}
#[repr(C)]
pub struct Mpz {
    pub alloc: c_int,
    pub size: c_int,
    pub d: *mut u64,
}
#[repr(C)]
pub struct Mpq {
    pub num: Mpz,
    pub den: Mpz,
}
#[repr(C)]
pub struct Mpc {
    pub re: Mpfr,
    pub im: Mpfr,
}

pub type P = *mut Mpfr;
pub type C = *const Mpfr;

unsafe extern "C" {
    pub fn mpfr_init2(x: P, p: c_long);
    pub fn mpfr_clear(x: P);
    pub fn mpfr_set_prec(x: P, p: c_long);
    pub fn mpfr_set(r: P, x: C, rnd: Rnd) -> c_int;
    pub fn mpfr_add(r: P, a: C, b: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sub(r: P, a: C, b: C, rnd: Rnd) -> c_int;
    pub fn mpfr_mul(r: P, a: C, b: C, rnd: Rnd) -> c_int;
    pub fn mpfr_div(r: P, a: C, b: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sqr(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sqrt(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_fma(r: P, a: C, b: C, c: C, rnd: Rnd) -> c_int;
    pub fn mpfr_fms(r: P, a: C, b: C, c: C, rnd: Rnd) -> c_int;
    pub fn mpfr_rec_sqrt(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_cbrt(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_rootn_ui(r: P, a: C, n: c_ulong, rnd: Rnd) -> c_int;
    pub fn mpfr_pow_si(r: P, a: C, n: c_long, rnd: Rnd) -> c_int;
    pub fn mpfr_pow_z(r: P, a: C, n: *const Mpz, rnd: Rnd) -> c_int;
    pub fn mpfr_pow(r: P, a: C, b: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sum(r: P, t: *const P, n: c_ulong, rnd: Rnd) -> c_int;
    pub fn mpfr_dot(r: P, a: *const P, b: *const P, n: c_ulong, rnd: Rnd) -> c_int;
    pub fn mpfr_exp(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_exp2(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_exp10(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_expm1(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_log(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_log2(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_log10(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_log1p(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sin(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_cos(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_tan(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sec(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_csc(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_cot(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sin_cos(s: P, c: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_atan(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_asin(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_acos(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_atan2(r: P, y: C, x: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sinh(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_cosh(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_tanh(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_sinh_cosh(s: P, c: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_asinh(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_acosh(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_atanh(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_const_pi(r: P, rnd: Rnd) -> c_int;
    pub fn mpfr_const_log2(r: P, rnd: Rnd) -> c_int;
    pub fn mpfr_const_euler(r: P, rnd: Rnd) -> c_int;
    pub fn mpfr_const_catalan(r: P, rnd: Rnd) -> c_int;
    pub fn mpfr_free_cache();
    pub fn mpfr_get_d(a: C, rnd: Rnd) -> f64;
    pub fn mpfr_set_d(r: P, d: f64, rnd: Rnd) -> c_int;
    pub fn mpfr_set_si(r: P, d: c_long, rnd: Rnd) -> c_int;
    pub fn mpfr_set_ui(r: P, d: c_ulong, rnd: Rnd) -> c_int;
    pub fn mpfr_get_si(a: C, rnd: Rnd) -> c_long;
    pub fn mpfr_floor(r: P, a: C) -> c_int;
    pub fn mpfr_ceil(r: P, a: C) -> c_int;
    pub fn mpfr_trunc(r: P, a: C) -> c_int;
    pub fn mpfr_rint(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_frac(r: P, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_set_z(r: P, z: *const Mpz, rnd: Rnd) -> c_int;
    pub fn mpfr_get_z(z: *mut Mpz, a: C, rnd: Rnd) -> c_int;
    pub fn mpfr_set_q(r: P, q: *const Mpq, rnd: Rnd) -> c_int;
    pub fn mpfr_get_q(q: *mut Mpq, a: C);
    pub fn mpfr_set_z_2exp(r: P, z: *const Mpz, e: c_long, rnd: Rnd) -> c_int;
    pub fn mpfr_get_z_2exp(z: *mut Mpz, a: C) -> c_long;
    pub fn mpfr_get_str(s: *mut c_char, e: *mut c_long, b: c_int, n: usize, a: C, rnd: Rnd) -> *mut c_char;
    pub fn mpfr_free_str(s: *mut c_char);
    pub fn mpfr_strtofr(r: P, s: *const c_char, end: *mut *mut c_char, b: c_int, rnd: Rnd) -> c_int;
    pub fn mpfr_prec_round(r: P, p: c_long, rnd: Rnd) -> c_int;
    pub fn mpfr_cmp(a: C, b: C) -> c_int;
    pub fn mpfr_cmp_d(a: C, b: f64) -> c_int;
    pub fn mpfr_cmp_si(a: C, b: c_long) -> c_int;
    pub fn mpfr_add_si(r: P, a: C, b: c_long, rnd: Rnd) -> c_int;
    pub fn mpfr_mul_d(r: P, a: C, b: f64, rnd: Rnd) -> c_int;
    pub fn mpfr_div_z(r: P, a: C, b: *const Mpz, rnd: Rnd) -> c_int;
    pub fn mpfr_set_emax(e: c_long) -> c_int;
    pub fn mpfr_set_emin(e: c_long) -> c_int;
    pub fn mpfr_get_emax_max() -> c_long;
    pub fn mpfr_get_emin_min() -> c_long;
    pub fn mpfr_get_version() -> *const c_char;

    #[link_name = "__gmpz_init"]
    pub fn mpz_init(z: *mut Mpz);
    #[link_name = "__gmpz_clear"]
    pub fn mpz_clear(z: *mut Mpz);
    #[link_name = "__gmpz_import"]
    pub fn mpz_import(z: *mut Mpz, count: usize, order: c_int, size: usize, endian: c_int, nails: usize, op: *const c_void);
    #[link_name = "__gmpz_neg"]
    pub fn mpz_neg(r: *mut Mpz, a: *const Mpz);
    #[link_name = "__gmpz_set_si"]
    pub fn mpz_set_si(r: *mut Mpz, a: c_long);
    #[link_name = "__gmpq_init"]
    pub fn mpq_init(q: *mut Mpq);
    #[link_name = "__gmpq_clear"]
    pub fn mpq_clear(q: *mut Mpq);
    #[link_name = "__gmpq_set_z"]
    pub fn mpq_set_z(q: *mut Mpq, z: *const Mpz);
    #[link_name = "__gmpq_canonicalize"]
    pub fn mpq_canonicalize(q: *mut Mpq);
    #[link_name = "__gmpz_set"]
    pub fn mpz_set(r: *mut Mpz, a: *const Mpz);

    pub fn mpc_init2(z: *mut Mpc, p: c_long);
    pub fn mpc_clear(z: *mut Mpc);
    pub fn mpc_set(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_add(r: *mut Mpc, a: *const Mpc, b: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_sub(r: *mut Mpc, a: *const Mpc, b: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_mul(r: *mut Mpc, a: *const Mpc, b: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_div(r: *mut Mpc, a: *const Mpc, b: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_sqr(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_sqrt(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_abs(r: P, a: *const Mpc, rnd: Rnd) -> c_int;
    pub fn mpc_norm(r: P, a: *const Mpc, rnd: Rnd) -> c_int;
    pub fn mpc_add_fr(r: *mut Mpc, a: *const Mpc, b: C, rnd: c_int) -> c_int;
    pub fn mpc_sub_fr(r: *mut Mpc, a: *const Mpc, b: C, rnd: c_int) -> c_int;
    pub fn mpc_mul_fr(r: *mut Mpc, a: *const Mpc, b: C, rnd: c_int) -> c_int;
    pub fn mpc_div_fr(r: *mut Mpc, a: *const Mpc, b: C, rnd: c_int) -> c_int;
    pub fn mpc_mul_si(r: *mut Mpc, a: *const Mpc, b: c_long, rnd: c_int) -> c_int;
    pub fn mpc_mul_i(r: *mut Mpc, a: *const Mpc, sgn: c_int, rnd: c_int) -> c_int;
    pub fn mpc_neg(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_conj(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_exp(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_log(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_log10(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_arg(r: P, a: *const Mpc, rnd: Rnd) -> c_int;
    pub fn mpc_proj(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_sin(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_cos(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_tan(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_sin_cos(s: *mut Mpc, c: *mut Mpc, a: *const Mpc, rs: c_int, rc: c_int) -> c_int;
    pub fn mpc_sinh(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_cosh(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_tanh(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_asin(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_acos(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_atan(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_asinh(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_acosh(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_atanh(r: *mut Mpc, a: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_pow(r: *mut Mpc, a: *const Mpc, b: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_pow_fr(r: *mut Mpc, a: *const Mpc, b: C, rnd: c_int) -> c_int;
    pub fn mpc_pow_si(r: *mut Mpc, a: *const Mpc, b: c_long, rnd: c_int) -> c_int;
    pub fn mpc_dot(r: *mut Mpc, a: *const *mut Mpc, b: *const *mut Mpc, n: c_ulong, rnd: c_int) -> c_int;
    pub fn mpc_fma(r: *mut Mpc, a: *const Mpc, b: *const Mpc, c: *const Mpc, rnd: c_int) -> c_int;
    pub fn mpc_get_version() -> *const c_char;
}

/// An `mpfr_t` that clears itself.
pub struct MF(pub Mpfr);
impl MF {
    pub fn new(p: u64) -> MF {
        let mut x = std::mem::MaybeUninit::<Mpfr>::uninit();
        unsafe {
            mpfr_init2(x.as_mut_ptr(), p as c_long);
            MF(x.assume_init())
        }
    }
    #[inline(always)]
    pub fn p(&mut self) -> P {
        &mut self.0
    }
    #[inline(always)]
    pub fn c(&self) -> C {
        &self.0
    }
    pub fn copy(&self) -> MF {
        let mut r = MF::new(self.0.prec as u64);
        unsafe { mpfr_set(r.p(), self.c(), RN) };
        r
    }
}
impl Drop for MF {
    fn drop(&mut self) {
        unsafe { mpfr_clear(&mut self.0) }
    }
}

pub struct MZ(pub Mpz);
impl MZ {
    pub fn new() -> MZ {
        let mut z = std::mem::MaybeUninit::<Mpz>::uninit();
        unsafe {
            mpz_init(z.as_mut_ptr());
            MZ(z.assume_init())
        }
    }
    pub fn from_limbs(neg: bool, limbs: &[u64]) -> MZ {
        let mut z = MZ::new();
        unsafe {
            mpz_import(&mut z.0, limbs.len(), -1, 8, 0, 0, limbs.as_ptr() as *const c_void);
            if neg {
                let zp: *mut Mpz = &mut z.0;
                mpz_neg(zp, zp);
            }
        }
        z
    }
    pub fn limbs(&self) -> (bool, Vec<u64>) {
        let n = self.0.size.unsigned_abs() as usize;
        let v = unsafe { std::slice::from_raw_parts(self.0.d, n).to_vec() };
        (self.0.size < 0, v)
    }
}
impl Drop for MZ {
    fn drop(&mut self) {
        unsafe { mpz_clear(&mut self.0) }
    }
}

pub struct MQ(pub Mpq);
impl MQ {
    pub fn new() -> MQ {
        let mut q = std::mem::MaybeUninit::<Mpq>::uninit();
        unsafe {
            mpq_init(q.as_mut_ptr());
            MQ(q.assume_init())
        }
    }
}
impl Drop for MQ {
    fn drop(&mut self) {
        unsafe { mpq_clear(&mut self.0) }
    }
}

/// An `mpc_t` that clears itself.
pub struct MC(pub Mpc);
impl MC {
    pub fn new(p: u64) -> MC {
        let mut x = std::mem::MaybeUninit::<Mpc>::uninit();
        unsafe {
            mpc_init2(x.as_mut_ptr(), p as c_long);
            MC(x.assume_init())
        }
    }
    #[inline(always)]
    pub fn p(&mut self) -> *mut Mpc {
        &mut self.0
    }
    #[inline(always)]
    pub fn c(&self) -> *const Mpc {
        &self.0
    }
}
impl Drop for MC {
    fn drop(&mut self) {
        unsafe { mpc_clear(&mut self.0) }
    }
}

#[cfg(has_arb)]
pub mod arb {
    use super::*;
    #[repr(C)]
    pub struct Arb {
        pub w: [u64; 6],
    }
    #[repr(C)]
    pub struct ArbMat {
        pub entries: *mut Arb,
        pub r: c_long,
        pub c: c_long,
        pub stride: c_long,
    }
    pub type A = *mut Arb;
    pub type AC = *const Arb;
    unsafe extern "C" {
        pub fn arb_init(x: A);
        pub fn arb_clear(x: A);
        pub fn arf_set_mpfr(x: A, y: C);
        pub fn arf_get_mpfr(r: P, x: AC, rnd: Rnd) -> c_int;
        pub fn arb_set_ui(x: A, u: c_ulong);
        pub fn arb_exp(r: A, x: AC, p: c_long);
        pub fn arb_expm1(r: A, x: AC, p: c_long);
        pub fn arb_log(r: A, x: AC, p: c_long);
        pub fn arb_log1p(r: A, x: AC, p: c_long);
        pub fn arb_log_base_ui(r: A, x: AC, b: c_ulong, p: c_long);
        pub fn arb_pow(r: A, x: AC, y: AC, p: c_long);
        pub fn arb_sin(r: A, x: AC, p: c_long);
        pub fn arb_cos(r: A, x: AC, p: c_long);
        pub fn arb_tan(r: A, x: AC, p: c_long);
        pub fn arb_sec(r: A, x: AC, p: c_long);
        pub fn arb_csc(r: A, x: AC, p: c_long);
        pub fn arb_cot(r: A, x: AC, p: c_long);
        pub fn arb_sin_cos(s: A, c: A, x: AC, p: c_long);
        pub fn arb_atan(r: A, x: AC, p: c_long);
        pub fn arb_asin(r: A, x: AC, p: c_long);
        pub fn arb_acos(r: A, x: AC, p: c_long);
        pub fn arb_atan2(r: A, y: AC, x: AC, p: c_long);
        pub fn arb_sinh(r: A, x: AC, p: c_long);
        pub fn arb_cosh(r: A, x: AC, p: c_long);
        pub fn arb_tanh(r: A, x: AC, p: c_long);
        pub fn arb_sinh_cosh(s: A, c: A, x: AC, p: c_long);
        pub fn arb_asinh(r: A, x: AC, p: c_long);
        pub fn arb_acosh(r: A, x: AC, p: c_long);
        pub fn arb_atanh(r: A, x: AC, p: c_long);
        pub fn arb_sqrt(r: A, x: AC, p: c_long);
        pub fn arb_rsqrt(r: A, x: AC, p: c_long);
        pub fn arb_root_ui(r: A, x: AC, k: c_ulong, p: c_long);
        pub fn arb_const_pi(r: A, p: c_long);
        pub fn arb_const_e(r: A, p: c_long);
        pub fn arb_const_log2(r: A, p: c_long);
        pub fn arb_const_log10(r: A, p: c_long);
        pub fn arb_const_euler(r: A, p: c_long);
        pub fn arb_const_catalan(r: A, p: c_long);
        pub fn arb_mat_init(m: *mut ArbMat, r: c_long, c: c_long);
        pub fn arb_mat_clear(m: *mut ArbMat);
        pub fn arb_mat_mul(r: *mut ArbMat, a: *const ArbMat, b: *const ArbMat, p: c_long);
        pub fn flint_cleanup();
    }
    pub struct AB(pub Box<Arb>);
    impl AB {
        pub fn new() -> AB {
            let mut b = Box::new(Arb { w: [0; 6] });
            unsafe { arb_init(&mut *b) };
            AB(b)
        }
        pub fn from_mpfr(x: &MF) -> AB {
            let mut b = AB::new();
            unsafe { arf_set_mpfr(b.p(), x.c()) };
            b
        }
        #[inline(always)]
        pub fn p(&mut self) -> A {
            &mut *self.0
        }
        #[inline(always)]
        pub fn c(&self) -> AC {
            &*self.0
        }
        /// The midpoint rounded to nearest at `p` bits.
        pub fn mid(&self, p: u64) -> MF {
            let mut r = MF::new(p);
            unsafe { arf_get_mpfr(r.p(), self.c(), RN) };
            r
        }
    }
    impl Drop for AB {
        fn drop(&mut self) {
            unsafe { arb_clear(&mut *self.0) }
        }
    }
}
