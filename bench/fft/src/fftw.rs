//! FFTW 3 through its C interface, double (`fftw_`) and float (`fftwf_`).

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

pub type RawPlan = *mut c_void;

pub const FORWARD: c_int = -1;
pub const BACKWARD: c_int = 1;
pub const MEASURE: c_uint = 0;
pub const PATIENT: c_uint = 1 << 5;
pub const ESTIMATE: c_uint = 1 << 6;

pub const REDFT00: c_int = 3;
pub const REDFT01: c_int = 4;
pub const REDFT10: c_int = 5;
pub const REDFT11: c_int = 6;
pub const RODFT00: c_int = 7;
pub const RODFT01: c_int = 8;
pub const RODFT10: c_int = 9;
pub const RODFT11: c_int = 10;

macro_rules! fftw_api {
    ($m:ident, $t:ty, $pre:literal) => {
        pub mod $m {
            use super::*;
            #[allow(dead_code)]
            unsafe extern "C" {
                #[link_name = concat!($pre, "plan_dft_1d")]
                pub fn plan_dft_1d(n: c_int, i: *mut $t, o: *mut $t, sign: c_int, f: c_uint) -> RawPlan;
                #[link_name = concat!($pre, "plan_dft_2d")]
                pub fn plan_dft_2d(n0: c_int, n1: c_int, i: *mut $t, o: *mut $t, sign: c_int, f: c_uint) -> RawPlan;
                #[link_name = concat!($pre, "plan_dft_3d")]
                pub fn plan_dft_3d(n0: c_int, n1: c_int, n2: c_int, i: *mut $t, o: *mut $t, sign: c_int, f: c_uint) -> RawPlan;
                #[link_name = concat!($pre, "plan_many_dft")]
                pub fn plan_many_dft(
                    rank: c_int, n: *const c_int, howmany: c_int,
                    i: *mut $t, inembed: *const c_int, istride: c_int, idist: c_int,
                    o: *mut $t, onembed: *const c_int, ostride: c_int, odist: c_int,
                    sign: c_int, f: c_uint,
                ) -> RawPlan;
                #[link_name = concat!($pre, "plan_dft_r2c_1d")]
                pub fn plan_dft_r2c_1d(n: c_int, i: *mut $t, o: *mut $t, f: c_uint) -> RawPlan;
                #[link_name = concat!($pre, "plan_dft_c2r_1d")]
                pub fn plan_dft_c2r_1d(n: c_int, i: *mut $t, o: *mut $t, f: c_uint) -> RawPlan;
                #[link_name = concat!($pre, "plan_many_dft_r2c")]
                pub fn plan_many_dft_r2c(
                    rank: c_int, n: *const c_int, howmany: c_int,
                    i: *mut $t, inembed: *const c_int, istride: c_int, idist: c_int,
                    o: *mut $t, onembed: *const c_int, ostride: c_int, odist: c_int,
                    f: c_uint,
                ) -> RawPlan;
                #[link_name = concat!($pre, "plan_r2r_1d")]
                pub fn plan_r2r_1d(n: c_int, i: *mut $t, o: *mut $t, kind: c_int, f: c_uint) -> RawPlan;
                #[link_name = concat!($pre, "execute")]
                pub fn execute(p: RawPlan);
                #[link_name = concat!($pre, "destroy_plan")]
                pub fn destroy_plan(p: RawPlan);
                #[link_name = concat!($pre, "forget_wisdom")]
                pub fn forget_wisdom();
                #[link_name = concat!($pre, "import_wisdom_from_filename")]
                pub fn import_wisdom_from_filename(f: *const c_char) -> c_int;
                #[link_name = concat!($pre, "export_wisdom_to_filename")]
                pub fn export_wisdom_to_filename(f: *const c_char) -> c_int;
                #[link_name = concat!($pre, "version")]
                pub static version: [c_char; 0];
            }
        }
    };
}

fftw_api!(d, f64, "fftw_");
fftw_api!(s, f32, "fftwf_");

/// The FFTW functions of one precision, the pointers to its real type (complex arrays are
/// passed as pointers to their first real part: `Complex<T>` is `#[repr(C)] {re, im}`).
pub trait Fftw: Copy + 'static {
    const PRE: &'static str;
    unsafe fn dft_1d(n: usize, i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan;
    unsafe fn dft_2d(n0: usize, n1: usize, i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan;
    unsafe fn dft_3d(n: [usize; 3], i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan;
    unsafe fn many_dft(n: usize, howmany: usize, i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan;
    unsafe fn r2c_1d(n: usize, i: *mut Self, o: *mut Self, f: c_uint) -> RawPlan;
    unsafe fn c2r_1d(n: usize, i: *mut Self, o: *mut Self, f: c_uint) -> RawPlan;
    unsafe fn many_r2c(n: usize, howmany: usize, i: *mut Self, o: *mut Self, f: c_uint) -> RawPlan;
    unsafe fn r2r_1d(n: usize, i: *mut Self, o: *mut Self, kind: c_int, f: c_uint) -> RawPlan;
    unsafe fn exec(p: RawPlan);
    unsafe fn destroy(p: RawPlan);
    unsafe fn forget();
    unsafe fn import(path: &CStr) -> bool;
    unsafe fn export(path: &CStr) -> bool;
    fn version() -> String;
}

macro_rules! impl_fftw {
    ($t:ty, $m:ident, $pre:literal) => {
        impl Fftw for $t {
            const PRE: &'static str = $pre;
            unsafe fn dft_1d(n: usize, i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan {
                unsafe { $m::plan_dft_1d(n as c_int, i, o, sign, f) }
            }
            unsafe fn dft_2d(n0: usize, n1: usize, i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan {
                unsafe { $m::plan_dft_2d(n0 as c_int, n1 as c_int, i, o, sign, f) }
            }
            unsafe fn dft_3d(n: [usize; 3], i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan {
                unsafe { $m::plan_dft_3d(n[0] as c_int, n[1] as c_int, n[2] as c_int, i, o, sign, f) }
            }
            unsafe fn many_dft(n: usize, howmany: usize, i: *mut Self, o: *mut Self, sign: c_int, f: c_uint) -> RawPlan {
                let nn = n as c_int;
                unsafe {
                    $m::plan_many_dft(1, &nn, howmany as c_int, i, std::ptr::null(), 1, nn,
                        o, std::ptr::null(), 1, nn, sign, f)
                }
            }
            unsafe fn r2c_1d(n: usize, i: *mut Self, o: *mut Self, f: c_uint) -> RawPlan {
                unsafe { $m::plan_dft_r2c_1d(n as c_int, i, o, f) }
            }
            unsafe fn c2r_1d(n: usize, i: *mut Self, o: *mut Self, f: c_uint) -> RawPlan {
                unsafe { $m::plan_dft_c2r_1d(n as c_int, i, o, f) }
            }
            unsafe fn many_r2c(n: usize, howmany: usize, i: *mut Self, o: *mut Self, f: c_uint) -> RawPlan {
                let nn = n as c_int;
                let h = (n / 2 + 1) as c_int;
                unsafe {
                    $m::plan_many_dft_r2c(1, &nn, howmany as c_int, i, std::ptr::null(), 1, nn,
                        o, std::ptr::null(), 1, h, f)
                }
            }
            unsafe fn r2r_1d(n: usize, i: *mut Self, o: *mut Self, kind: c_int, f: c_uint) -> RawPlan {
                unsafe { $m::plan_r2r_1d(n as c_int, i, o, kind, f) }
            }
            unsafe fn exec(p: RawPlan) {
                unsafe { $m::execute(p) }
            }
            unsafe fn destroy(p: RawPlan) {
                unsafe { $m::destroy_plan(p) }
            }
            unsafe fn forget() {
                unsafe { $m::forget_wisdom() }
            }
            unsafe fn import(path: &CStr) -> bool {
                unsafe { $m::import_wisdom_from_filename(path.as_ptr()) != 0 }
            }
            unsafe fn export(path: &CStr) -> bool {
                unsafe { $m::export_wisdom_to_filename(path.as_ptr()) != 0 }
            }
            fn version() -> String {
                // SAFETY: FFTW's version string is a NUL-terminated static array.
                unsafe { CStr::from_ptr(std::ptr::addr_of!($m::version) as *const c_char) }
                    .to_string_lossy()
                    .into_owned()
            }
        }
    };
}

impl_fftw!(f64, d, "fftw_");
impl_fftw!(f32, s, "fftwf_");

/// An FFTW plan, destroyed when dropped.
pub struct Plan<T: Fftw> {
    pub raw: RawPlan,
    _t: std::marker::PhantomData<T>,
}

impl<T: Fftw> Plan<T> {
    /// Wraps a plan; `None` when FFTW returned no plan.
    pub fn wrap(raw: RawPlan) -> Option<Plan<T>> {
        if raw.is_null() {
            None
        } else {
            Some(Plan {
                raw,
                _t: std::marker::PhantomData,
            })
        }
    }
    #[inline]
    pub fn exec(&self) {
        // SAFETY: the plan's arrays outlive it (each caller keeps them alive alongside).
        unsafe { T::exec(self.raw) }
    }
}

impl<T: Fftw> Drop for Plan<T> {
    fn drop(&mut self) {
        // SAFETY: a live plan of this precision.
        unsafe { T::destroy(self.raw) }
    }
}

/// The wisdom file of a precision: `$FFTBENCH_WISDOM_DIR/<prefix>measure.wisdom`.
fn wisdom_path<T: Fftw>() -> Option<CString> {
    wisdom_file::<T>("measure")
}

fn wisdom_file<T: Fftw>(kind: &str) -> Option<CString> {
    let dir = std::env::var("FFTBENCH_WISDOM_DIR").ok()?;
    CString::new(format!("{dir}/{}{kind}.wisdom", T::PRE)).ok()
}

/// Whether FFTW_PATIENT plans are made too (`FFTBENCH_PATIENT=1`), for powers of two from 64
/// to 2^20 (its planning takes up to about half a minute a length; the wisdom is kept).
pub fn patient(n: usize) -> bool {
    std::env::var("FFTBENCH_PATIENT").as_deref() == Ok("1")
        && n.is_power_of_two()
        && (64..=1 << 20).contains(&n)
}

/// Runs `f` (which makes FFTW_PATIENT plans) with the patient wisdom loaded instead of the
/// measure wisdom, so that neither kind of plan is made from the other's wisdom.
pub fn with_patient<T: Fftw, R>(f: impl FnOnce() -> R) -> R {
    let (m, p) = (wisdom_file::<T>("measure"), wisdom_file::<T>("patient"));
    // SAFETY: valid C strings; the planner is used by this thread only.
    unsafe {
        if let Some(m) = &m {
            T::export(m);
        }
        T::forget();
        if let Some(p) = &p {
            T::import(p);
        }
        let r = f();
        if let Some(p) = &p {
            T::export(p);
        }
        T::forget();
        if let Some(m) = &m {
            T::import(m);
        }
        r
    }
}

pub fn load_wisdom<T: Fftw>() {
    if let Some(p) = wisdom_path::<T>() {
        // SAFETY: a valid C string; FFTW reads the file if it exists.
        let ok = unsafe { T::import(&p) };
        eprintln!("{}wisdom {:?}: {}", T::PRE, p, if ok { "loaded" } else { "none" });
    }
}

pub fn save_wisdom<T: Fftw>() {
    if let Some(p) = wisdom_path::<T>() {
        // SAFETY: a valid C string.
        unsafe { T::export(&p) };
    }
}
