//! Buffers, random data, comparisons and the rows of the scoreboard.

use crate::timing::Stat;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::io::Write;
use std::ops::{Deref, DerefMut};

/// A zeroed buffer aligned to 64 bytes (the same placement for every contender).
pub struct ABuf<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> ABuf<T> {
    pub fn new(len: usize) -> ABuf<T> {
        let bytes = (len * size_of::<T>()).max(64);
        let layout = Layout::from_size_align(bytes, 64).unwrap();
        // SAFETY: non-zero size; zero bytes are a valid value of the float types used here.
        let ptr = unsafe { alloc_zeroed(layout) } as *mut T;
        assert!(!ptr.is_null());
        ABuf { ptr, len }
    }
    pub fn from(src: &[T]) -> ABuf<T> {
        let mut b = ABuf::new(src.len());
        b.copy_from_slice(src);
        b
    }
    pub fn ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: Copy> Drop for ABuf<T> {
    fn drop(&mut self) {
        let bytes = (self.len * size_of::<T>()).max(64);
        // SAFETY: allocated with this layout in `new`.
        unsafe { dealloc(self.ptr as *mut u8, Layout::from_size_align(bytes, 64).unwrap()) }
    }
}

impl<T: Copy> Deref for ABuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` initialised values.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> DerefMut for ABuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: `len` initialised values, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// xorshift64*, uniform in [−1, 1).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
    pub fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Relative L2 distance ‖a − b‖ / ‖a‖.
pub fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared outputs differ in length");
    let (mut d, mut s) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += (x - y) * (x - y);
        s += x * x;
    }
    if s == 0.0 { d.sqrt() } else { (d / s).sqrt() }
}

/// One row of the scoreboard: ours against one reference.
pub struct Out {
    pub rows: usize,
    pub failed: usize,
    pub tol_scale: f64,
}

pub fn fmt_ns(v: f64) -> String {
    if !v.is_finite() {
        return "NA".into();
    }
    let digits = if v >= 1000.0 {
        0
    } else if v >= 100.0 {
        1
    } else if v >= 10.0 {
        2
    } else {
        3
    };
    format!("{v:.digits$}")
}

pub fn fmt_ratio(r: f64) -> String {
    if !r.is_finite() || r <= 0.0 {
        return "NA".into();
    }
    // three significant digits
    let e = r.log10().floor() as i32;
    let d = (2 - e).max(0) as usize;
    format!("{r:.d$}")
}

impl Out {
    pub fn new() -> Out {
        Out {
            rows: 0,
            failed: 0,
            tol_scale: 1.0,
        }
    }

    /// Writes one row. `ours`/`theirs` are per call; `per` divides both (items per call).
    #[allow(clippy::too_many_arguments)]
    pub fn row(
        &mut self,
        group: &str,
        op: &str,
        params: &str,
        ours: Stat,
        reference: &str,
        theirs: Stat,
        per: f64,
        notes: &str,
    ) {
        let (o, t) = (ours.median / per, theirs.median / per);
        let mut n = notes.to_string();
        let wide: Vec<String> = [("ours", ours.iqr), ("ref", theirs.iqr)]
            .iter()
            .filter(|(_, q)| *q > 0.10)
            .map(|(w, q)| format!("{w} iqr {:.0}%", q * 100.0))
            .collect();
        if !wide.is_empty() {
            if !n.is_empty() {
                n.push_str("; ");
            }
            n.push_str(&wide.join(", "));
        }
        println!(
            "fft\t{group}\t{op}\t{params}\t{}\t{reference}\t{}\t{}\t{n}",
            fmt_ns(o),
            fmt_ns(t),
            fmt_ratio(o / t)
        );
        std::io::stdout().flush().ok();
        self.rows += 1;
    }
}

/// Checks that a reference computes the same thing as ours: relative L2 distance below `tol`.
/// Returns a note for the row ("" when it agrees); prints a warning when it does not.
pub fn check(what: &str, reference: &str, ours: &[f64], theirs: &[f64], tol: f64) -> (bool, String) {
    let e = rel(ours, theirs);
    if e.is_finite() && e <= tol {
        (true, String::new())
    } else {
        eprintln!("CHECK FAILED: {what} vs {reference}: rel {e:.3e} > {tol:.1e}");
        (false, format!("CHECK FAILED rel {e:.2e}"))
    }
}
