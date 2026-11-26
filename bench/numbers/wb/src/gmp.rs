//! GMP's mpn layer.
pub type Limb = u64;
#[link(name = "gmp")]
unsafe extern "C" {
    #[link_name = "__gmpn_mul_n"]
    pub fn mpn_mul_n(r: *mut Limb, x: *const Limb, y: *const Limb, n: i64);
    #[link_name = "__gmpn_sqr"]
    pub fn mpn_sqr(r: *mut Limb, x: *const Limb, n: i64);
    #[link_name = "__gmpn_mul"]
    pub fn mpn_mul(r: *mut Limb, x: *const Limb, nx: i64, y: *const Limb, ny: i64) -> Limb;
    #[link_name = "__gmpn_gcd"]
    pub fn mpn_gcd(r: *mut Limb, x: *mut Limb, nx: i64, y: *mut Limb, ny: i64) -> i64;
    #[link_name = "__gmpn_gcd_1"]
    pub fn mpn_gcd_1(x: *const Limb, nx: i64, y: Limb) -> Limb;
    #[link_name = "__gmpn_mod_1"]
    pub fn mpn_mod_1(x: *const Limb, nx: i64, y: Limb) -> Limb;
    #[link_name = "__gmpn_addmul_1"]
    pub fn mpn_addmul_1(r: *mut Limb, x: *const Limb, n: i64, y: Limb) -> Limb;
    #[link_name = "__gmpn_mul_1"]
    pub fn mpn_mul_1(r: *mut Limb, x: *const Limb, n: i64, y: Limb) -> Limb;
}

#[repr(C)]
pub struct Mpz {
    alloc: i32,
    size: i32,
    d: *mut Limb,
}
#[link(name = "gmp")]
unsafe extern "C" {
    #[link_name = "__gmpz_init"]
    pub fn mpz_init(x: *mut Mpz);
    #[link_name = "__gmpz_clear"]
    pub fn mpz_clear(x: *mut Mpz);
    #[link_name = "__gmpz_import"]
    pub fn mpz_import(r: *mut Mpz, count: usize, order: i32, size: usize, endian: i32, nails: usize, op: *const u8);
    #[link_name = "__gmpz_jacobi"]
    pub fn mpz_jacobi(a: *const Mpz, b: *const Mpz) -> i32;
    #[link_name = "__gmpz_powm"]
    pub fn mpz_powm(r: *mut Mpz, b: *const Mpz, e: *const Mpz, m: *const Mpz);
    #[link_name = "__gmpz_gcd"]
    pub fn mpz_gcd(r: *mut Mpz, a: *const Mpz, b: *const Mpz);
}

pub struct Z(pub Mpz);
impl Z {
    pub fn new() -> Z {
        let mut z = Mpz { alloc: 0, size: 0, d: core::ptr::null_mut() };
        unsafe { mpz_init(&mut z) };
        Z(z)
    }
    pub fn from_limbs(l: &[u64]) -> Z {
        let mut z = Z::new();
        unsafe { mpz_import(&mut z.0, l.len(), -1, 8, 0, 0, l.as_ptr().cast()) };
        z
    }
    pub fn limbs(&self) -> Vec<u64> {
        let n = self.0.size.unsigned_abs() as usize;
        let mut v = vec![0u64; n];
        unsafe { core::ptr::copy_nonoverlapping(self.0.d, v.as_mut_ptr(), n) };
        v
    }
}
impl Drop for Z {
    fn drop(&mut self) {
        unsafe { mpz_clear(&mut self.0) };
    }
}
