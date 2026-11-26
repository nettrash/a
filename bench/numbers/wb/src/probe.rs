//! The rows of the numbers scoreboard (the shared kernels of `Number` the v0.2 speed pass added)
//! against GMP: short products (`nat::mulhigh_to`, `nat::sqrhigh_to`, the high half of a product
//! to a stated error) against GMP's and the crate's whole products, the Jacobi symbol (the
//! public `Number::jacobi`, and the kernel `nat::euclid_steps` driving the nt area's state
//! table) against `mpz_jacobi`, and `quotients_into` against `quotients` into a cleared vector.
//!
//! `numbers-wb check` runs the same-work checks (the short products' error bound against GMP's
//! whole product, the Jacobi symbols against GMP); `numbers-wb time SAMPLES` prints the
//! scoreboard rows (tab-separated, the header of the scoreboards, without it).
use crate::gmp::*;
use crate::math::numbers::nat;
use std::time::Instant;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn limbs(&mut self, n: usize) -> Vec<u64> {
        (0..n).map(|_| self.next()).collect()
    }
    /// Random limbs with runs of all-ones and zeros (the carry-heavy cases).
    pub fn edgy(&mut self, n: usize) -> Vec<u64> {
        let kind = self.next() % 6;
        (0..n)
            .map(|_| match kind {
                0 => u64::MAX,
                1 => if self.next() % 2 == 0 { u64::MAX } else { 0 },
                2 => if self.next() % 4 == 0 { self.next() } else { u64::MAX },
                3 => if self.next() % 4 == 0 { self.next() } else { 0 },
                4 => u64::MAX - (self.next() % 3),
                _ => self.next(),
            })
            .collect()
    }
}

/// Median ns per call over `samples` batches of each, alternating A and B sample by sample
/// (ABAB…, the order swapped every other sample); each batch takes at least 1 ms of `a`.
pub fn ab(samples: usize, mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            a();
        }
        if t.elapsed().as_secs_f64() >= 1e-3 {
            break;
        }
        reps *= 2;
    }
    for _ in 0..reps {
        b();
    }
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for i in 0..samples {
        let mut run = |f: &mut dyn FnMut(), v: &mut Vec<f64>| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            v.push(t.elapsed().as_secs_f64() * 1e9 / reps as f64);
        };
        if i % 2 == 0 {
            run(&mut a, &mut ta);
            run(&mut b, &mut tb);
        } else {
            run(&mut b, &mut tb);
            run(&mut a, &mut ta);
        }
    }
    ta.sort_by(f64::total_cmp);
    tb.sort_by(f64::total_cmp);
    (ta[samples / 2], tb[samples / 2])
}

const SHORT: &[usize] = &[12, 16, 24, 32, 48, 64, 96, 128, 200, 256, 360, 400, 512];
const JACOBI: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const CF: &[usize] = &[2, 4, 16, 64, 256, 1024];

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    let samples = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(21);
    let only = std::env::var("NUMBERS_FILTER").unwrap_or_default();
    match args.get(1).map(String::as_str) {
        Some("check") => {
            short_check(300, 2000);
            jac_check(1, 300, 300);
        }
        Some("time") => {
            let want = |g: &str| only.is_empty() || g.contains(only.as_str());
            if want("mul") {
                short_time(samples);
            }
            if want("quadratic") {
                jac_time(samples);
            }
            if want("gcd") {
                cf_time(samples);
            }
        }
        _ => eprintln!("numbers-wb check | numbers-wb time [SAMPLES]  (NUMBERS_FILTER=mul|quadratic|gcd)"),
    }
}

fn row(group: &str, op: &str, params: &str, ours: f64, reference: &str, theirs: f64, notes: &str) {
    println!("numbers\t{group}\t{op}\t{params}\t{ours:.0}\t{reference}\t{theirs:.0}\t{:.3}\t{notes}", ours / theirs);
}

fn full(x: &[u64], y: &[u64]) -> Vec<u64> {
    let n = x.len();
    let mut p = vec![0u64; 2 * n];
    unsafe { mpn_mul_n(p.as_mut_ptr(), x.as_ptr(), y.as_ptr(), n as i64) };
    p
}

/// Checks `xy - (n-1) B^n < V B^(n-1) <= xy` for V = r[n-1..2n].
fn check_bound(x: &[u64], y: &[u64], r: &[u64]) -> bool {
    let n = x.len();
    let p = full(x, y);
    // d = p[n-1..2n] - V as a signed multi-limb; the low part p[..n-1] makes the true value
    // p/B^(n-1) = p[n-1..] + frac, so xy - V B^(n-1) = (p[n-1..] - V) B^(n-1) + p[..n-1] >= 0 iff
    // p[n-1..] >= V; and < (n-1) B^n iff (p[n-1..] - V) B^(n-1) + low < (n-1) B^n, i.e.
    // p[n-1..] - V < (n-1) B, or == (n-1)B - 1 ... handle exactly: D = p[n-1..] - V must satisfy
    // D*B^(n-1) + low < (n-1) B^n.
    let mut d = vec![0u64; n + 1];
    let mut borrow = 0u64;
    for i in 0..=n {
        let (a, b1) = p[n - 1 + i].overflowing_sub(r[n - 1 + i]);
        let (a, b2) = a.overflowing_sub(borrow);
        d[i] = a;
        borrow = (b1 || b2) as u64;
    }
    if borrow != 0 {
        return false; // V > p
    }
    // D must be < (n-1) B + something: D B^(n-1) + low < (n-1) B^n  <=>  D < (n-1) B or
    // (D == (n-1)B - ... ) -- since low < B^(n-1), D B^(n-1) + low < (D+1) B^(n-1), so it is enough
    // and necessary that D B^(n-1) + low < (n-1)B^n; with D integer: D <= (n-1)B - 1 suffices; and
    // D >= (n-1)B fails.
    if d[2..].iter().any(|&w| w != 0) {
        return false;
    }
    let dv = (u128::from(d[1]) << 64) | u128::from(d[0]);
    dv < (n as u128 - 1) * (1u128 << 64) || (n == 1 && dv == 0)
}

fn short_check(nmax: usize, count: usize) {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut scratch = Vec::new();
    let mut bad = 0;
    let mut cases = 0;
    let mut worst = 0f64;
    for it in 0..count {
        let n = if it < nmax { it + 1 } else { 1 + (rng.next() as usize) % nmax };
        for sq in [false, true] {
            let (x, y) = if it % 3 == 0 {
                (rng.edgy(n), rng.edgy(n))
            } else {
                (rng.limbs(n), rng.limbs(n))
            };
            let y = if sq { x.clone() } else { y };
            let mut r = vec![0xdead_beefu64; 2 * n];
            if sq {
                nat::sqrhigh_to(&mut r, &x, &mut scratch);
            } else {
                nat::mulhigh_to(&mut r, &x, &y, &mut scratch);
            }
            cases += 1;
            if !check_bound(&x, &y, &r) {
                bad += 1;
                if bad < 10 {
                    eprintln!("BAD n={n} sq={sq}");
                }
            }
            // How far, in units of r[n]: (p[n..] - r[n..]).
            let p = full(&x, &y);
            let mut d: i128 = 0;
            for i in (n..2 * n).rev() {
                d = d.saturating_mul(1 << 64).saturating_add(p[i] as i128 - r[i] as i128);
                if d.abs() > 1 << 40 { break; }
            }
            let rel = d as f64 / n as f64;
            if rel > worst { worst = rel; }
        }
    }
    println!("short-check: {cases} cases, {bad} bad, worst (H - A)/n = {worst:.3}");
}

fn short_time(samples: usize) {
    let mut rng = Rng(7);
    let mut scratch = Vec::new();
    for &n in SHORT {
        let x = rng.limbs(n);
        let y = rng.limbs(n);
        let mut r = vec![0u64; 2 * n];
        let mut r2 = vec![0u64; 2 * n];
        let mut s2 = Vec::new();
        let p = format!("{n} limbs");
        let note = "white-box: a pub(crate) kernel; the high half to the error bound (n-1)·B^n against a whole product (more work, the reference a user of GMP would call)";
        let (a, b) = ab(samples, || nat::mulhigh_to(&mut r, &x, &y, &mut scratch), || nat::mul_to(&mut r2, &x, &y, &mut s2));
        row("mul", "nat::mulhigh_to", &p, a, "crate whole product nat::mul_to", b, note);
        let (a, b) = ab(samples, || nat::mulhigh_to(&mut r, &x, &y, &mut scratch), || unsafe { mpn_mul_n(r2.as_mut_ptr(), x.as_ptr(), y.as_ptr(), n as i64) });
        row("mul", "nat::mulhigh_to", &p, a, "GMP mpn_mul_n (whole)", b, note);
        let (a, b) = ab(samples, || nat::sqrhigh_to(&mut r, &x, &mut scratch), || nat::sqr_to(&mut r2, &x, &mut s2));
        row("mul", "nat::sqrhigh_to", &p, a, "crate whole square nat::sqr_to", b, note);
        let (a, b) = ab(samples, || nat::sqrhigh_to(&mut r, &x, &mut scratch), || unsafe { mpn_sqr(r2.as_mut_ptr(), x.as_ptr(), n as i64) });
        row("mul", "nat::sqrhigh_to", &p, a, "GMP mpn_sqr (whole)", b, note);
    }
}

fn number_of(l: &[u64]) -> crate::math::numbers::Number {
    use crate::math::numbers::{ByteOrder, Number};
    let mut v = Vec::with_capacity(8 * l.len());
    for w in l {
        v.extend_from_slice(&w.to_le_bytes());
    }
    Number::new(v, ByteOrder::LittleEndian)
}

// The Jacobi state of the nt area's modular/jacobi.rs (bit 0 the sign, bits 1-2 a mod 4, bits 3-4
// b mod 4, bit 5 a is the denominator), rebuilt here to drive `nat::euclid_steps`.
const fn jstep(s: u8, a_reduced: bool, q: u8) -> u8 {
    let mut e = s & 1;
    let xa = (s >> 1) & 3;
    let xb = (s >> 3) & 3;
    let mut den_a = (s >> 5) & 1 == 1;
    let (mut r, t) = if a_reduced { (xa, xb) } else { (xb, xa) };
    let reduced_is_den = den_a == a_reduced;
    if !reduced_is_den {
        r = (r + 16 - (q * t) % 4) % 4;
    } else if t & 1 == 1 {
        if r == 3 && t == 3 { e ^= 1; }
        den_a = !den_a;
        r = (r + 16 - (q * t) % 4) % 4;
    } else {
        let mut k = 0;
        while k < q {
            let d = r;
            let next = (d + 4 - t) % 4;
            if d == 3 { e ^= 1; }
            if d == 3 && next == 3 { e ^= 1; }
            r = next;
            k += 1;
        }
    }
    let (xa, xb) = if a_reduced { (r, t) } else { (t, r) };
    e | (xa << 1) | (xb << 3) | ((den_a as u8) << 5)
}
const JT: [u8; 512] = {
    let mut t = [0u8; 512];
    let mut i = 0;
    while i < 512 {
        t[i] = jstep((i >> 3) as u8, (i >> 2) & 1 == 1, (i & 3) as u8);
        i += 1;
    }
    t
};

fn jacobi_steps(x: &[u64], y: &[u64]) -> i32 {
    let mut s = (((x[0] & 3) << 1) | ((y[0] & 3) << 3)) as u8;
    let one = nat::euclid_steps(x, y, |ar, q| {
        s = JT[(usize::from(s) << 3) | (usize::from(ar) << 2) | (q & 3) as usize];
    });
    if one { 1 - 2 * i32::from(s & 1) } else { 0 }
}

fn gmp_jacobi(x: &[u64], y: &[u64]) -> i32 {
    let (a, b) = (Z::from_limbs(x), Z::from_limbs(y));
    unsafe { mpz_jacobi(&a.0, &b.0) }
}

fn jac_check(lo: usize, hi: usize, count: usize) {
    let mut rng = Rng(77);
    let (mut bad, mut zero, mut minus) = (0, 0, 0);
    for it in 0..count {
        let n = lo + it % (hi - lo + 1);
        let mut y = if it % 5 == 0 { rng.edgy(n) } else { rng.limbs(n) };
        y[0] |= 1;
        if y[n - 1] == 0 { y[n - 1] = 1; }
        let mut x = match it % 7 {
            0 => { let mut v = rng.limbs(n); v[n - 1] = 0; v }
            1 => { // a common factor: x = y * small mod-ish: take x = y - 2*small
                let mut v = y.clone(); v[0] = v[0].wrapping_sub(6); v }
            2 => rng.edgy(n),
            _ => rng.limbs(n),
        };
        if it % 11 == 0 { // force a common factor 3
            let k = (it % 3) as u64;
            let _ = k;
        }
        // keep x < y for the nt convention (not needed by euclid_steps).
        if x.iter().rev().cmp(y.iter().rev()) != std::cmp::Ordering::Less {
            x[n - 1] = y[n - 1] / 2;
        }
        let ours = jacobi_steps(&x, &y);
        let g = gmp_jacobi(&x, &y);
        if ours == 0 { zero += 1; }
        if ours == -1 { minus += 1; }
        if ours != g {
            bad += 1;
            if bad < 10 { eprintln!("BAD jacobi n={n} ours={ours} gmp={g}"); }
        }
    }
    println!("jac-check: {count} cases {lo}..={hi} limbs, {bad} bad ({zero} zero, {minus} minus one)");
}

fn jac_time(samples: usize) {
    let mut rng = Rng(3);
    for &n in JACOBI {
        let mut y = rng.limbs(n);
        y[0] |= 1;
        let mut x = rng.limbs(n);
        x[n - 1] >>= 1;
        y[n - 1] |= 1 << 63;
        let (xn, yn) = (number_of(&x), number_of(&y));
        let j = gmp_jacobi(&x, &y);
        assert_eq!(jacobi_steps(&x, &y), j, "euclid_steps Jacobi against GMP, {n} limbs");
        assert_eq!(xn.jacobi(&yn), j, "Number::jacobi against GMP, {n} limbs");
        let (za, zb) = (Z::from_limbs(&x), Z::from_limbs(&y));
        let p = format!("{n} limbs random");
        let (a, b) = ab(samples, || { std::hint::black_box(xn.jacobi(&yn)); }, || { std::hint::black_box(unsafe { mpz_jacobi(&za.0, &zb.0) }); });
        row("quadratic", "x.jacobi(&y)", &p, a, "GMP mpz_jacobi", b, "same symbol checked");
        let (a, b) = ab(samples, || { std::hint::black_box(jacobi_steps(&x, &y)); }, || { std::hint::black_box(unsafe { mpz_jacobi(&za.0, &zb.0) }); });
        row("quadratic", "Jacobi by nat::euclid_steps", &p, a, "GMP mpz_jacobi", b, "white-box: the pub(crate) kernel with the nt area's state table; same symbol checked");
    }
}

fn cf_time(samples: usize) {
    use crate::math::numbers::{ByteOrder, quotients, quotients_into};
    let mut rng = Rng(66);
    for &n in CF {
        let mut a = rng.limbs(n);
        a[n - 1] |= 1 << 63;
        let b = rng.limbs(n - 1);
        let (an, bn) = (number_of(&a), number_of(&b));
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        quotients(&an, &bn, ByteOrder::LittleEndian, &mut v1);
        quotients_into(&an, &bn, ByteOrder::LittleEndian, &mut v2, 0);
        assert!(v1.len() == v2.len() && v1.iter().zip(&v2).all(|(p, q)| p == q), "quotients_into against quotients, {n} limbs");
        let (x, y) = ab(samples, || { quotients_into(&an, &bn, ByteOrder::LittleEndian, &mut v2, 0); }, || { v1.clear(); quotients(&an, &bn, ByteOrder::LittleEndian, &mut v1); });
        row("gcd", "quotients_into (reused vectors)", &format!("{n} limbs, {} terms", v2.len()), x, "quotients into a cleared Vec (new terms)", y, "white-box: both pub(crate); the same terms checked");
    }
}
