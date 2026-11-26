//! C ABI for the SciPy driver (scipy_bench.py): the f64 cases' records and our batch
//! functions, so that SciPy's ufuncs and ours run on the same records in one process.
use crate::{Case, POOL, cases64, pool};
use std::sync::OnceLock;

fn reg() -> &'static Vec<Case<f64>> {
    static R: OnceLock<Vec<Case<f64>>> = OnceLock::new();
    R.get_or_init(cases64)
}

#[unsafe(no_mangle)]
pub extern "C" fn sb_ncases() -> usize {
    reg().len()
}

/// Writes "group\top\tparams\tnout\tscipy-spec\tmetric" (spec empty when none); returns its
/// length.
/// # Safety
/// `buf` must hold `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sb_case_info(i: usize, buf: *mut u8, cap: usize) -> usize {
    let c = &reg()[i];
    let s = format!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        c.group,
        c.op,
        c.params,
        c.nout,
        c.scipy.clone().unwrap_or_default(),
        c.metric.code()
    );
    let n = s.len().min(cap);
    unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), buf, n) };
    s.len()
}

/// The first `n` records of case `i` (the same sequence the Rust harness draws, cycled past
/// its pool of 16,384).
/// # Safety
/// `out` must hold `n` records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sb_case_inputs(i: usize, out: *mut [f64; 4], n: usize) {
    let p = pool(&reg()[i], POOL);
    let o = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for (k, r) in o.iter_mut().enumerate() {
        *r = p[k % POOL];
    }
}

/// Our batch function of case `i`.
/// # Safety
/// `inp` and `out` must hold `n` records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sb_case_ours(i: usize, inp: *const [f64; 4], out: *mut [f64; 4], n: usize) {
    let c = &reg()[i];
    let a = unsafe { std::slice::from_raw_parts(inp, n) };
    let b = unsafe { std::slice::from_raw_parts_mut(out, n) };
    (c.ours)(a, b);
}
