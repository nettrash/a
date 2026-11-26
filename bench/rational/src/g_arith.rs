// + - * / (new results and in place), comparisons, equality: random parts, common factors,
// a pool the predictor learns, and operands of different sizes.
use a::math::rational::Rational;
use malachite_q::Rational as MQ;
use num_rational::{BigRational, Ratio};

use crate::data::*;
use crate::ffi::*;
use crate::harness::row;
use crate::libs::*;

#[derive(Clone)]
pub struct Sz {
    pub label: String,
    pub bits: u64,
    pub exact: bool,
}

pub fn s16() -> Sz {
    Sz { label: "s".into(), bits: 16, exact: false }
}
pub fn w1() -> Sz {
    Sz { label: "1".into(), bits: 64, exact: false }
}
pub fn limbs_sz(n: u64) -> Sz {
    if n == 1 {
        return w1();
    }
    Sz { label: format!("{n}"), bits: 64 * n, exact: true }
}
impl Sz {
    pub fn limbs(&self) -> u64 {
        self.bits.div_ceil(64)
    }
    pub fn q(&self, rng: &mut Rng, signed: bool) -> Q {
        rand_q(rng, self.bits, self.bits, self.exact, signed)
    }
    fn part(&self, rng: &mut Rng, bits: u64) -> Vec<u8> {
        if self.exact { rng.exact(bits) } else { rng.upto(bits) }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    AddA,
    SubA,
    MulA,
    DivA,
    Cmp,
    Eq,
}

impl Op {
    pub fn name(self) -> &'static str {
        match self {
            Op::Add => "&a + &b",
            Op::Sub => "&a - &b",
            Op::Mul => "&a * &b",
            Op::Div => "&a / &b",
            Op::AddA => "a += &b",
            Op::SubA => "a -= &b",
            Op::MulA => "a *= &b",
            Op::DivA => "a /= &b",
            Op::Cmp => "cmp",
            Op::Eq => "==",
        }
    }
    fn code(self) -> i32 {
        match self {
            Op::Add => 0,
            Op::Sub => 1,
            Op::Mul => 2,
            Op::Div => 3,
            Op::Cmp => 4,
            Op::Eq => 5,
            Op::AddA => 6,
            Op::SubA => 7,
            Op::MulA => 8,
            Op::DivA => 9,
        }
    }
    fn res(self) -> Res {
        match self {
            Op::Cmp => Res::Sign,
            Op::Eq => Res::Bool,
            _ => Res::Q,
        }
    }
}

macro_rules! arms {
    ($op:expr, $name:expr, $a:expr, $b:expr, $k:expr) => {
        match $op {
            Op::Add => mk2($name, $a, $b, $k, |x, y| x + y),
            Op::Sub => mk2($name, $a, $b, $k, |x, y| x - y),
            Op::Mul => mk2($name, $a, $b, $k, |x, y| x * y),
            Op::Div => mk2($name, $a, $b, $k, |x, y| x / y),
            Op::AddA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t += y;
                t
            }),
            Op::SubA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t -= y;
                t
            }),
            Op::MulA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t *= y;
                t
            }),
            Op::DivA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t /= y;
                t
            }),
            Op::Cmp => mk2($name, $a, $b, $k, |x, y| x.cmp(y)),
            Op::Eq => mk2($name, $a, $b, $k, |x, y| x == y),
        }
    };
}

#[cfg(feature = "rug")]
macro_rules! rug_arms {
    ($op:expr, $name:expr, $a:expr, $b:expr, $k:expr) => {
        match $op {
            Op::Add => mk2($name, $a, $b, $k, |x, y| rug::Rational::from(x + y)),
            Op::Sub => mk2($name, $a, $b, $k, |x, y| rug::Rational::from(x - y)),
            Op::Mul => mk2($name, $a, $b, $k, |x, y| rug::Rational::from(x * y)),
            Op::Div => mk2($name, $a, $b, $k, |x, y| rug::Rational::from(x / y)),
            Op::AddA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t += y;
                t
            }),
            Op::SubA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t -= y;
                t
            }),
            Op::MulA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t *= y;
                t
            }),
            Op::DivA => mk2($name, $a, $b, $k, |x, y| {
                let mut t = x.clone();
                t /= y;
                t
            }),
            Op::Cmp => mk2($name, $a, $b, $k, |x, y| x.cmp(y)),
            Op::Eq => mk2($name, $a, $b, $k, |x, y| x == y),
        }
    };
}

#[derive(Clone, Copy)]
pub struct Refs {
    pub gmp: bool,
    pub flint: bool,
    pub mal: bool,
    pub num: bool,
    pub rug: bool,
    pub i64: bool,
}

pub const ALL: Refs = Refs { gmp: true, flint: true, mal: true, num: true, rug: true, i64: false };

/// One operation on the pairs (a[i], b[i]) against every reference in `refs`.
pub fn bin_row(group: &str, op: Op, params: &str, a: &[Q], b: &[Q], refs: Refs, note: &str) {
    let k = 32;
    let what = format!("{group} {} {params}", op.name());
    let oa: Vec<Rational> = a.iter().map(ours).collect();
    let ob: Vec<Rational> = b.iter().map(ours).collect();
    let ga = Gmp::new(a, true);
    let gb = Gmp::new(b, true);
    #[cfg(has_flint)]
    let (fa, fb) = (Flint::new(a, true), Flint::new(b, true));
    let (ma, mb): (Vec<MQ>, Vec<MQ>) = if refs.mal { (a.iter().map(mal).collect(), b.iter().map(mal).collect()) } else { (vec![], vec![]) };
    let (na, nb): (Vec<BigRational>, Vec<BigRational>) = if refs.num { (a.iter().map(num).collect(), b.iter().map(num).collect()) } else { (vec![], vec![]) };
    #[cfg(feature = "rug")]
    let (ra, rb): (Vec<rug::Rational>, Vec<rug::Rational>) = if refs.rug { (a.iter().map(rugq).collect(), b.iter().map(rugq).collect()) } else { (vec![], vec![]) };
    let to_i = |q: &Q| -> Ratio<i64> {
        let n = limbs(&q.num)[0] as i64;
        let d = limbs(&q.den)[0] as i64;
        Ratio::new_raw(if q.neg { -n } else { n }, d)
    };
    let (ia, ib): (Vec<Ratio<i64>>, Vec<Ratio<i64>>) = if refs.i64 { (a.iter().map(to_i).collect(), b.iter().map(to_i).collect()) } else { (vec![], vec![]) };
    let mut set = Set::new(&what, arms!(op, "a", &oa, &ob, k));
    if refs.gmp {
        set.add(gmp2("GMP mpq", &ga, &gb, op.code(), k, op.res()));
    }
    #[cfg(has_flint)]
    if refs.flint {
        set.add(flint2("FLINT fmpq", &fa, &fb, op.code(), k, op.res()));
    }
    if refs.mal {
        set.add(arms!(op, "malachite-q", &ma, &mb, k));
    }
    if refs.num {
        set.add(arms!(op, "num-rational BigRational", &na, &nb, k));
    }
    #[cfg(feature = "rug")]
    if refs.rug {
        set.add(rug_arms!(op, "rug Rational", &ra, &rb, k));
    }
    if refs.i64 {
        set.add(arms!(op, "num-rational Ratio<i64>", &ia, &ib, k));
    }
    row(group, op.name(), params, set.ents, 1.0, note);
}

fn mul_bytes(x: &[u8], y: &[u8]) -> Vec<u8> {
    le_bytes(&(&number(x, false) * &number(y, false)))
}

/// Pairs whose denominators share a factor of half their length.
fn common_add(rng: &mut Rng, sz: &Sz, n: usize) -> (Vec<Q>, Vec<Q>) {
    let h = sz.bits / 2;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let g = sz.part(rng, h);
        let x = mul_bytes(&g, &sz.part(rng, h));
        let y = mul_bytes(&g, &sz.part(rng, h));
        let s1 = rng.bool();
        let s2 = rng.bool();
        a.push(reduce(s1, sz.part(rng, sz.bits), x));
        b.push(reduce(s2, sz.part(rng, sz.bits), y));
    }
    (a, b)
}

/// Pairs where each numerator shares a factor of half its length with the other's denominator.
fn common_mul(rng: &mut Rng, sz: &Sz, n: usize) -> (Vec<Q>, Vec<Q>) {
    let h = sz.bits / 2;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let g1 = sz.part(rng, h);
        let g2 = sz.part(rng, h);
        let an = mul_bytes(&g1, &sz.part(rng, h));
        let bd = mul_bytes(&g1, &sz.part(rng, h));
        let bn = mul_bytes(&g2, &sz.part(rng, h));
        let ad = mul_bytes(&g2, &sz.part(rng, h));
        let s1 = rng.bool();
        let s2 = rng.bool();
        a.push(reduce(s1, an, ad));
        b.push(reduce(s2, bn, bd));
    }
    (a, b)
}

fn pair(rng: &mut Rng, x: &Sz, y: &Sz, n: usize, signed: bool) -> (Vec<Q>, Vec<Q>) {
    let a = (0..n).map(|_| x.q(rng, signed)).collect();
    let b = (0..n).map(|_| y.q(rng, signed)).collect();
    (a, b)
}

pub fn run(which: &str) {
    let mut rng = Rng::new(0xA417);
    let sizes = [1u64, 2, 4, 16, 64, 128, 1024, 8192];
    let mut all: Vec<Sz> = vec![s16()];
    all.extend(sizes.iter().map(|&n| limbs_sz(n)));
    let parts = |sz: &Sz| {
        if sz.label == "s" { "parts < 2^16".to_string() } else if sz.label == "1" { "parts < 2^64 (words)".to_string() } else { format!("parts of {} limbs", sz.label) }
    };
    if which == "arith" || which == "all" {
        for sz in &all {
            let n = pool_for(sz.limbs(), 4);
            let (a, mut b) = pair(&mut rng, sz, sz, n, true);
            // No zero divisor.
            for q in b.iter_mut() {
                if q.num == [0] {
                    *q = reduce(q.neg, vec![1], q.den.clone());
                }
            }
            let refs = Refs { num: sz.limbs() <= 1024, i64: sz.label == "s", ..ALL };
            let note = if sz.limbs() > 1024 { "num-rational left out (quadratic gcd, seconds an operation)" } else { "" };
            for op in [Op::Add, Op::Sub, Op::Mul, Op::Div] {
                bin_row("arith", op, &format!("{}; random signs; pool {n}", parts(sz)), &a, &b, refs, note);
            }
        }
    }
    if which == "inplace" || which == "all" {
        for lab in [0u64, 1, 2, 8, 64, 1024] {
            let sz = if lab == 0 { s16() } else { limbs_sz(lab) };
            let n = pool_for(sz.limbs(), 4);
            let (a, mut b) = pair(&mut rng, &sz, &sz, n, true);
            for q in b.iter_mut() {
                if q.num == [0] {
                    *q = reduce(q.neg, vec![1], q.den.clone());
                }
            }
            let refs = Refs { i64: sz.label == "s", ..ALL };
            for op in [Op::AddA, Op::SubA, Op::MulA, Op::DivA] {
                bin_row("inplace", op, &format!("{}; random; fresh copy of a; pool {n}", parts(&sz)), &a, &b, refs, "GMP/FLINT: mpq_set then mpq_op(r, r, b) into a variable that exists");
            }
            let (a, b) = common_add(&mut rng, &sz, n);
            bin_row("inplace", Op::AddA, &format!("{}; denominators share half their length; pool {n}", parts(&sz)), &a, &b, refs, "GMP/FLINT: mpq_set then mpq_add(r, r, b)");
            let (a, b) = common_mul(&mut rng, &sz, n);
            bin_row("inplace", Op::MulA, &format!("{}; numerators share half with the other's denominator; pool {n}", parts(&sz)), &a, &b, refs, "GMP/FLINT: mpq_set then mpq_mul(r, r, b)");
        }
    }
    if which == "pool512" || which == "all" {
        for sz in [s16(), w1()] {
            let (a, b) = pair(&mut rng, &sz, &sz, 512, true);
            let refs = Refs { num: false, rug: false, ..ALL };
            for op in [Op::Add, Op::Mul, Op::AddA, Op::MulA] {
                bin_row("pool512", op, &format!("{}; pool of 512 timed over and over (the predictor learns it)", parts(&sz)), &a, &b, refs, "");
            }
        }
    }
    if which == "mixed" || which == "all" {
        let combos: [(u64, u64); 8] = [(0, 2), (1, 2), (2, 1), (1, 4), (1, 16), (0, 64), (1, 64), (1, 1024)];
        for (x, y) in combos {
            let sx = if x == 0 { s16() } else { limbs_sz(x) };
            let sy = if y == 0 { s16() } else { limbs_sz(y) };
            let n = pool_for(sx.limbs().max(sy.limbs()), 4);
            let (a, b) = pair(&mut rng, &sx, &sy, n, true);
            let p = format!("a {} b {} limbs; random signs; pool {n}", sx.label, sy.label);
            for op in [Op::Add, Op::Mul, Op::AddA] {
                bin_row("mixed", op, &p, &a, &b, ALL, "");
            }
        }
    }
    if which == "cmp" || which == "all" {
        for lab in [0u64, 1, 2, 8, 64, 1024] {
            let sz = if lab == 0 { s16() } else { limbs_sz(lab) };
            let n = pool_for(sz.limbs(), 4);
            let (a, b) = pair(&mut rng, &sz, &sz, n, true);
            bin_row("cmp", Op::Cmp, &format!("{}; random signs; pool {n}", parts(&sz)), &a, &b, Refs { i64: sz.label == "s", ..ALL }, "");
            // Positive, all four parts of the same length (exact bits).
            let ex = Sz { exact: true, ..sz.clone() };
            let (a, b) = pair(&mut rng, &ex, &ex, n, false);
            bin_row("cmp", Op::Cmp, &format!("{}; positive, all parts of the same length; pool {n}", parts(&sz)), &a, &b, ALL, "");
            // Equality: random pairs, one in four equal.
            let (a, mut b) = pair(&mut rng, &sz, &sz, n, true);
            for i in (0..n).step_by(4) {
                b[i] = a[i].clone();
            }
            bin_row("cmp", Op::Eq, &format!("{}; random pairs, one in four equal; pool {n}", parts(&sz)), &a, &b, Refs { i64: sz.label == "s", ..ALL }, "");
        }
    }
}
