// OpenSSL's libcrypto loaded at run time (dlopen, RTLD_LOCAL), so that OpenSSL 3 and OpenSSL 4
// can be timed side by side in one process. The libraries are found by the environment variables
// OPENSSL3_LIB / OPENSSL4_LIB, else at Homebrew's paths (macOS) or by soname (Linux).
#![allow(non_snake_case, dead_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, s: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "macos")]
const MODE: c_int = 2 | 4; // RTLD_NOW | RTLD_LOCAL
#[cfg(not(target_os = "macos"))]
const MODE: c_int = 2; // RTLD_NOW (RTLD_LOCAL is 0)

pub type P = *mut c_void;
pub type CP = *const c_void;

pub const COMPRESSED: c_int = 2;
pub const UNCOMPRESSED: c_int = 4;

pub struct Ossl {
    pub name: String,
    pub OBJ_sn2nid: unsafe extern "C" fn(*const c_char) -> c_int,
    pub EC_GROUP_new_by_curve_name: unsafe extern "C" fn(c_int) -> P,
    pub EC_GROUP_get0_generator: unsafe extern "C" fn(CP) -> CP,
    pub EC_GROUP_get0_order: unsafe extern "C" fn(CP) -> CP,
    pub EC_GROUP_get0_cofactor: unsafe extern "C" fn(CP) -> CP,
    pub EC_POINT_new: unsafe extern "C" fn(CP) -> P,
    pub EC_POINT_free: unsafe extern "C" fn(P),
    pub EC_POINT_oct2point: unsafe extern "C" fn(CP, P, *const u8, usize, P) -> c_int,
    pub EC_POINT_point2oct: unsafe extern "C" fn(CP, CP, c_int, *mut u8, usize, P) -> usize,
    pub EC_POINT_mul: unsafe extern "C" fn(CP, P, CP, CP, CP, P) -> c_int,
    pub EC_POINTs_mul: Option<unsafe extern "C" fn(CP, P, CP, usize, *const CP, *const CP, P) -> c_int>,
    pub EC_POINT_add: unsafe extern "C" fn(CP, P, CP, CP, P) -> c_int,
    pub EC_POINT_dbl: unsafe extern "C" fn(CP, P, CP, P) -> c_int,
    pub EC_POINT_is_at_infinity: unsafe extern "C" fn(CP, CP) -> c_int,
    pub BN_CTX_new: unsafe extern "C" fn() -> P,
    pub BN_bin2bn: unsafe extern "C" fn(*const u8, c_int, P) -> P,
    pub EVP_MD_fetch: unsafe extern "C" fn(P, *const c_char, *const c_char) -> P,
    pub EVP_MD_CTX_new: unsafe extern "C" fn() -> P,
    pub EVP_DigestInit_ex: unsafe extern "C" fn(P, CP, P) -> c_int,
    pub EVP_DigestUpdate: unsafe extern "C" fn(P, *const u8, usize) -> c_int,
    pub EVP_DigestFinal_ex: unsafe extern "C" fn(P, *mut u8, *mut c_uint) -> c_int,
    pub SHA224: Option<unsafe extern "C" fn(*const u8, usize, *mut u8) -> *mut u8>,
    pub SHA256: Option<unsafe extern "C" fn(*const u8, usize, *mut u8) -> *mut u8>,
    pub SHA384: Option<unsafe extern "C" fn(*const u8, usize, *mut u8) -> *mut u8>,
    pub SHA512: Option<unsafe extern "C" fn(*const u8, usize, *mut u8) -> *mut u8>,
}

fn candidates(major: u32) -> Vec<String> {
    let var = format!("OPENSSL{major}_LIB");
    if let Ok(v) = std::env::var(&var) {
        return vec![v];
    }
    if cfg!(target_os = "macos") {
        vec![
            format!("/opt/homebrew/opt/openssl@{major}/lib/libcrypto.{major}.dylib"),
            format!("/usr/local/opt/openssl@{major}/lib/libcrypto.{major}.dylib"),
        ]
    } else {
        vec![format!("libcrypto.so.{major}")]
    }
}

macro_rules! sym {
    ($h:expr, $name:literal) => {{
        let c = CString::new($name).unwrap();
        // SAFETY: dlsym on a handle from dlopen; the symbol's C type is the field's.
        let p = unsafe { dlsym($h, c.as_ptr()) };
        assert!(!p.is_null(), "{} missing", $name);
        // SAFETY: as above, a function pointer of the declared type.
        unsafe { core::mem::transmute::<*mut c_void, _>(p) }
    }};
}
macro_rules! opt_sym {
    ($h:expr, $name:literal) => {{
        let c = CString::new($name).unwrap();
        // SAFETY: as in sym!.
        let p = unsafe { dlsym($h, c.as_ptr()) };
        if p.is_null() {
            None
        } else {
            // SAFETY: as in sym!.
            Some(unsafe { core::mem::transmute::<*mut c_void, _>(p) })
        }
    }};
}

impl Ossl {
    /// OpenSSL `major` (3 or 4), or None when it is not installed.
    pub fn load(major: u32) -> Option<Ossl> {
        for path in candidates(major) {
            let c = CString::new(path.clone()).unwrap();
            // SAFETY: dlopen of a shared library by path.
            let h = unsafe { dlopen(c.as_ptr(), MODE) };
            if h.is_null() {
                continue;
            }
            let version: unsafe extern "C" fn(c_int) -> *const c_char = sym!(h, "OpenSSL_version");
            // SAFETY: OpenSSL_version(OPENSSL_VERSION) returns a static C string.
            let v = unsafe { CStr::from_ptr(version(0)) }.to_string_lossy().into_owned();
            let name = v.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
            return Some(Ossl {
                name,
                OBJ_sn2nid: sym!(h, "OBJ_sn2nid"),
                EC_GROUP_new_by_curve_name: sym!(h, "EC_GROUP_new_by_curve_name"),
                EC_GROUP_get0_generator: sym!(h, "EC_GROUP_get0_generator"),
                EC_GROUP_get0_order: sym!(h, "EC_GROUP_get0_order"),
                EC_GROUP_get0_cofactor: sym!(h, "EC_GROUP_get0_cofactor"),
                EC_POINT_new: sym!(h, "EC_POINT_new"),
                EC_POINT_free: sym!(h, "EC_POINT_free"),
                EC_POINT_oct2point: sym!(h, "EC_POINT_oct2point"),
                EC_POINT_point2oct: sym!(h, "EC_POINT_point2oct"),
                EC_POINT_mul: sym!(h, "EC_POINT_mul"),
                EC_POINTs_mul: opt_sym!(h, "EC_POINTs_mul"),
                EC_POINT_add: sym!(h, "EC_POINT_add"),
                EC_POINT_dbl: sym!(h, "EC_POINT_dbl"),
                EC_POINT_is_at_infinity: sym!(h, "EC_POINT_is_at_infinity"),
                BN_CTX_new: sym!(h, "BN_CTX_new"),
                BN_bin2bn: sym!(h, "BN_bin2bn"),
                EVP_MD_fetch: sym!(h, "EVP_MD_fetch"),
                EVP_MD_CTX_new: sym!(h, "EVP_MD_CTX_new"),
                EVP_DigestInit_ex: sym!(h, "EVP_DigestInit_ex"),
                EVP_DigestUpdate: sym!(h, "EVP_DigestUpdate"),
                EVP_DigestFinal_ex: sym!(h, "EVP_DigestFinal_ex"),
                SHA224: opt_sym!(h, "SHA224"),
                SHA256: opt_sym!(h, "SHA256"),
                SHA384: opt_sym!(h, "SHA384"),
                SHA512: opt_sym!(h, "SHA512"),
            });
        }
        None
    }

    pub fn bn(&self, be: &[u8]) -> P {
        // SAFETY: BN_bin2bn reads be.len() bytes and allocates a BIGNUM.
        let b = unsafe { (self.BN_bin2bn)(be.as_ptr(), be.len() as c_int, core::ptr::null_mut()) };
        assert!(!b.is_null());
        b
    }

    pub fn ctx(&self) -> P {
        // SAFETY: allocates a BN_CTX.
        unsafe { (self.BN_CTX_new)() }
    }

    pub fn group(&self, sn: &str) -> Option<P> {
        let c = CString::new(sn).unwrap();
        // SAFETY: OBJ_sn2nid reads a C string.
        let nid = unsafe { (self.OBJ_sn2nid)(c.as_ptr()) };
        if nid == 0 {
            return None;
        }
        // SAFETY: a group by its NID.
        let g = unsafe { (self.EC_GROUP_new_by_curve_name)(nid) };
        (!g.is_null()).then_some(g)
    }

    pub fn point(&self, g: CP) -> P {
        // SAFETY: a new point of the group.
        unsafe { (self.EC_POINT_new)(g) }
    }

    pub fn decode(&self, g: CP, bytes: &[u8], ctx: P) -> Option<P> {
        let p = self.point(g);
        // SAFETY: reads bytes.len() bytes into p.
        let ok = unsafe { (self.EC_POINT_oct2point)(g, p, bytes.as_ptr(), bytes.len(), ctx) };
        if ok == 1 {
            Some(p)
        } else {
            // SAFETY: frees the point allocated above.
            unsafe { (self.EC_POINT_free)(p) };
            None
        }
    }

    pub fn encode(&self, g: CP, p: CP, form: c_int, ctx: P) -> Vec<u8> {
        let mut buf = vec![0u8; 160];
        // SAFETY: writes at most buf.len() bytes.
        let n = unsafe { (self.EC_POINT_point2oct)(g, p, form, buf.as_mut_ptr(), buf.len(), ctx) };
        assert!(n > 0);
        buf.truncate(n);
        buf
    }
}
