//! The test matrices: SuiteSparse-collection-like families, built deterministically.

use a::math::sparse::{CooMatrix, SparseMatrix};

/// A matrix in compressed sparse rows, canonical (sorted, no explicit zeros, no repeats).
#[derive(Clone)]
pub struct Csr {
    pub nrows: usize,
    pub ncols: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
}

impl Csr {
    pub fn nnz(&self) -> usize {
        self.idx.len()
    }
    pub fn from_ours(a: &SparseMatrix<f64>) -> Csr {
        Csr {
            nrows: a.rows(),
            ncols: a.cols(),
            ptr: a.row_pointers().to_vec(),
            idx: a.col_indices().to_vec(),
            val: a.values().to_vec(),
        }
    }
    pub fn ours(&self) -> SparseMatrix<f64> {
        SparseMatrix::from_csr(
            self.nrows,
            self.ncols,
            self.ptr.clone(),
            self.idx.clone(),
            self.val.clone(),
        )
    }
    pub fn transpose(&self) -> Csr {
        Csr::from_ours(&self.ours().transpose())
    }
    /// The lower triangle, diagonal included.
    pub fn lower(&self) -> Csr {
        Csr::from_ours(&self.ours().lower_triangle())
    }
    pub fn upper(&self) -> Csr {
        Csr::from_ours(&self.ours().upper_triangle())
    }
}

/// xorshift64*: the generator of every random matrix here.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 11) as u128 * n as u128 >> 53) as usize
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn build(n_rows: usize, n_cols: usize, t: Vec<(usize, usize, f64)>) -> Csr {
    Csr::from_ours(&SparseMatrix::from_coo(n_rows, n_cols, &t))
}

/// The five-point Laplacian of the k × k grid (Dirichlet), numbered by rows.
pub fn lap2d(k: usize) -> Csr {
    convdiff2d(k, 0.0, 0.0)
}

/// The five-point operator with 4 on the diagonal and −1 ∓ cx, −1 ∓ cy to the neighbours.
pub fn convdiff2d(k: usize, cx: f64, cy: f64) -> Csr {
    let n = k * k;
    let mut t = Vec::with_capacity(5 * n);
    for i in 0..k {
        for j in 0..k {
            let r = i * k + j;
            if i > 0 {
                t.push((r, r - k, -1.0 - cy));
            }
            if j > 0 {
                t.push((r, r - 1, -1.0 - cx));
            }
            t.push((r, r, 4.0));
            if j + 1 < k {
                t.push((r, r + 1, -1.0 + cx));
            }
            if i + 1 < k {
                t.push((r, r + k, -1.0 + cy));
            }
        }
    }
    build(n, n, t)
}

/// An anisotropic five-point Laplacian (weights 1 and 1.37 along the two axes): no multiple
/// eigenvalues, unlike the isotropic grid, so that eigensolvers asked for k of them find the same k.
pub fn ani2d(k: usize) -> Csr {
    let w = 1.37;
    let n = k * k;
    let mut t = Vec::with_capacity(5 * n);
    for i in 0..k {
        for j in 0..k {
            let r = i * k + j;
            if i > 0 {
                t.push((r, r - k, -w));
            }
            if j > 0 {
                t.push((r, r - 1, -1.0));
            }
            t.push((r, r, 2.0 + 2.0 * w));
            if j + 1 < k {
                t.push((r, r + 1, -1.0));
            }
            if i + 1 < k {
                t.push((r, r + k, -w));
            }
        }
    }
    build(n, n, t)
}

/// An anisotropic seven-point Laplacian (weights 1, 1.37, 1.71).
pub fn ani3d(k: usize) -> Csr {
    let (wx, wy, wz) = (1.71, 1.37, 1.0);
    let n = k * k * k;
    let mut t = Vec::with_capacity(7 * n);
    for x in 0..k {
        for y in 0..k {
            for z in 0..k {
                let r = (x * k + y) * k + z;
                if x > 0 {
                    t.push((r, r - k * k, -wx));
                }
                if y > 0 {
                    t.push((r, r - k, -wy));
                }
                if z > 0 {
                    t.push((r, r - 1, -wz));
                }
                t.push((r, r, 2.0 * (wx + wy + wz)));
                if z + 1 < k {
                    t.push((r, r + 1, -wz));
                }
                if y + 1 < k {
                    t.push((r, r + k, -wy));
                }
                if x + 1 < k {
                    t.push((r, r + k * k, -wx));
                }
            }
        }
    }
    build(n, n, t)
}

/// The seven-point Laplacian of the k × k × k grid.
pub fn lap3d(k: usize) -> Csr {
    convdiff3d(k, 0.0)
}

/// The seven-point operator with 6 on the diagonal and −1 ∓ c along each axis.
pub fn convdiff3d(k: usize, c: f64) -> Csr {
    let n = k * k * k;
    let mut t = Vec::with_capacity(7 * n);
    for x in 0..k {
        for y in 0..k {
            for z in 0..k {
                let r = (x * k + y) * k + z;
                if x > 0 {
                    t.push((r, r - k * k, -1.0 - c));
                }
                if y > 0 {
                    t.push((r, r - k, -1.0 - c));
                }
                if z > 0 {
                    t.push((r, r - 1, -1.0 - c));
                }
                t.push((r, r, 6.0));
                if z + 1 < k {
                    t.push((r, r + 1, -1.0 + c));
                }
                if y + 1 < k {
                    t.push((r, r + k, -1.0 + c));
                }
                if x + 1 < k {
                    t.push((r, r + k * k, -1.0 + c));
                }
            }
        }
    }
    build(n, n, t)
}

/// A structural matrix: the 27-point graph Laplacian of the k³ grid (plus 0.01·I) times a 3 × 3
/// positive definite block, as a Kronecker product — three unknowns a node, 81 elements a row
/// inside, symmetric positive definite.
pub fn elast3d(k: usize) -> Csr {
    let b = [[3.0, -1.0, 0.5], [-1.0, 3.0, -1.0], [0.5, -1.0, 3.0]];
    let nodes = k * k * k;
    let n = 3 * nodes;
    let mut t = Vec::with_capacity(81 * n);
    let id = |x: usize, y: usize, z: usize| (x * k + y) * k + z;
    for x in 0..k {
        for y in 0..k {
            for z in 0..k {
                let r = id(x, y, z);
                let mut deg = 0.0;
                for dx in -1i64..=1 {
                    for dy in -1i64..=1 {
                        for dz in -1i64..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                            if xx < 0 || yy < 0 || zz < 0 || xx >= k as i64 || yy >= k as i64 || zz >= k as i64 {
                                continue;
                            }
                            let c = id(xx as usize, yy as usize, zz as usize);
                            deg += 1.0;
                            for p in 0..3 {
                                for q in 0..3 {
                                    t.push((3 * r + p, 3 * c + q, -b[p][q]));
                                }
                            }
                        }
                    }
                }
                for p in 0..3 {
                    for q in 0..3 {
                        t.push((3 * r + p, 3 * r + q, (deg + 0.01) * b[p][q]));
                    }
                }
            }
        }
    }
    build(n, n, t)
}

/// A random unsymmetric matrix of `per_row` random off-diagonal columns a row, diagonally
/// dominant (nonsingular, no pivoting needed).
pub fn random_unsym(n: usize, per_row: usize, seed: u64) -> Csr {
    let mut rng = Rng::new(seed);
    let mut t = Vec::with_capacity((per_row + 1) * n);
    for i in 0..n {
        let mut s = 0.0;
        for _ in 0..per_row {
            let j = rng.below(n);
            let v = rng.unit() * 2.0 - 1.0;
            if j != i {
                s += v.abs();
                t.push((i, j, v));
            }
        }
        t.push((i, i, s + 1.0));
    }
    build(n, n, t)
}

/// A circuit-like matrix: a sparse random netlist with a power-law distribution of the node
/// degrees, a few supply nodes touching many others, unsymmetric values (controlled sources)
/// on a nearly symmetric pattern, a dominant diagonal of conductances.
pub fn circuit(n: usize, seed: u64) -> Csr {
    let mut rng = Rng::new(seed);
    let mut t = Vec::with_capacity(6 * n);
    let mut diag = vec![1e-3; n];
    let mut add_edge = |i: usize, j: usize, g: f64, t: &mut Vec<(usize, usize, f64)>, diag: &mut Vec<f64>| {
        if i == j {
            return;
        }
        t.push((i, j, -g));
        t.push((j, i, -g));
        diag[i] += g;
        diag[j] += g;
    };
    // A chain keeps it connected; then edges to nodes chosen with probability ∝ 1/rank.
    for i in 1..n {
        let g = 0.5 + rng.unit();
        add_edge(i - 1, i, g, &mut t, &mut diag);
    }
    for i in 0..n {
        let extra = (rng.unit() * 2.0) as usize;
        for _ in 0..extra {
            let u = rng.unit();
            let j = ((n as f64).powf(u) as usize).min(n - 1);
            let g = 0.1 + rng.unit();
            add_edge(i, j, g, &mut t, &mut diag);
        }
    }
    // Supply rails.
    for rail in 0..4 {
        let r = (rail * 7919) % n;
        for _ in 0..n / 100 {
            let j = rng.below(n);
            add_edge(r, j, 0.01 + 0.1 * rng.unit(), &mut t, &mut diag);
        }
    }
    // Controlled sources: one-sided couplings.
    for _ in 0..n / 10 {
        let i = rng.below(n);
        let j = rng.below(n);
        if i != j {
            t.push((i, j, 0.5 * (rng.unit() - 0.5)));
            diag[i] += 0.25;
        }
    }
    for (i, d) in diag.into_iter().enumerate() {
        t.push((i, i, d));
    }
    build(n, n, t)
}

/// A random graph Laplacian plus the identity (symmetric positive definite), `deg` random
/// edges a vertex on average.
pub fn random_spd(n: usize, deg: usize, seed: u64) -> Csr {
    let mut rng = Rng::new(seed);
    let mut t = Vec::with_capacity((2 * deg + 1) * n);
    let mut diag = vec![1.0; n];
    for i in 0..n {
        for _ in 0..deg / 2 {
            let j = rng.below(n);
            if j != i {
                let w = 0.5 + rng.unit();
                t.push((i, j, -w));
                t.push((j, i, -w));
                diag[i] += w;
                diag[j] += w;
            }
        }
    }
    for (i, d) in diag.into_iter().enumerate() {
        t.push((i, i, d));
    }
    build(n, n, t)
}

pub fn tridiag(n: usize) -> Csr {
    Csr::from_ours(&SparseMatrix::tridiagonal(&vec![-1.0; n - 1], &vec![2.0; n], &vec![-1.0; n - 1]))
}

/// The arrow: a full last row and column on a diagonal.
pub fn arrow(n: usize) -> Csr {
    let mut t = Vec::with_capacity(3 * n);
    for i in 0..n - 1 {
        t.push((i, i, 4.0));
        t.push((i, n - 1, 1.0));
        t.push((n - 1, i, 1.0));
    }
    t.push((n - 1, n - 1, n as f64));
    build(n, n, t)
}

/// A quasi-definite saddle point: [[L + I, Bᵀ], [B, −10⁻²·I]] with L the k × k Laplacian and B
/// of `m` constraints, each on a random point and its right-hand neighbour (local constraints,
/// as a discretised constraint has them).
pub fn saddle(k: usize, m: usize, seed: u64) -> Csr {
    let l = lap2d(k);
    let n = k * k;
    let mut rng = Rng::new(seed);
    let mut t = Vec::with_capacity(l.nnz() + n + 5 * m);
    for i in 0..n {
        for p in l.ptr[i]..l.ptr[i + 1] {
            t.push((i, l.idx[p], l.val[p]));
        }
        t.push((i, i, 1.0));
    }
    for c in 0..m {
        let r = n + c;
        let p = rng.below(n - 1);
        for j in [p, p + 1] {
            let v = rng.unit() + 0.5;
            t.push((r, j, v));
            t.push((j, r, v));
        }
        t.push((r, r, -1e-2));
    }
    build(n + m, n + m, t)
}

/// The smoothing of a curve: [I; 2·D], D the (n−1) × n first difference.
pub fn curve(n: usize) -> Csr {
    let mut t = Vec::with_capacity(3 * n);
    for i in 0..n {
        t.push((i, i, 1.0));
    }
    for i in 0..n - 1 {
        t.push((n + i, i, -2.0));
        t.push((n + i, i + 1, 2.0));
    }
    build(2 * n - 1, n, t)
}

/// A very tall band: m = 20·n rows, row r holding columns near r/20, ~3 elements.
pub fn tallband(n: usize, seed: u64) -> Csr {
    let mut rng = Rng::new(seed);
    let m = 20 * n;
    let mut t = Vec::with_capacity(3 * m);
    for r in 0..m {
        let c = r / 20;
        for _ in 0..3 {
            let off = rng.below(9) as i64 - 4;
            let j = (c as i64 + off).clamp(0, n as i64 - 1) as usize;
            t.push((r, j, rng.unit() * 2.0 - 1.0));
        }
    }
    build(m, n, t)
}

/// The least squares of a grid: [I; 2·Dx; 2·Dy] on the k × k grid.
pub fn gridls(k: usize) -> Csr {
    let n = k * k;
    let mut t = Vec::with_capacity(5 * n);
    let mut r = 0;
    for i in 0..n {
        t.push((r, i, 1.0));
        r += 1;
    }
    for i in 0..k {
        for j in 0..k - 1 {
            t.push((r, i * k + j, -2.0));
            t.push((r, i * k + j + 1, 2.0));
            r += 1;
        }
    }
    for i in 0..k - 1 {
        for j in 0..k {
            t.push((r, i * k + j, -2.0));
            t.push((r, (i + 1) * k + j, 2.0));
            r += 1;
        }
    }
    build(r, n, t)
}

/// A random tall matrix of m = 3·n rows and `per_row` elements a row, of full column rank (an
/// identity block on top).
pub fn random_tall(n: usize, per_row: usize, seed: u64) -> Csr {
    let mut rng = Rng::new(seed);
    let m = 3 * n;
    let mut t = Vec::with_capacity(per_row * m + n);
    for i in 0..n {
        t.push((i, i, 1.0 + rng.unit()));
    }
    for r in n..m {
        for _ in 0..per_row {
            t.push((r, rng.below(n), rng.unit() * 2.0 - 1.0));
        }
    }
    build(m, n, t)
}

/// The triplets of a finite-element assembly on the k × k grid of squares: every element adds
/// its 4 × 4 matrix, so each interior cell is given four times, in element order (repeats summed
/// by the assembly).
pub fn fe_triplets(k: usize) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let nodes = (k + 1) * (k + 1);
    let ke = [
        [4.0, -1.0, -2.0, -1.0],
        [-1.0, 4.0, -1.0, -2.0],
        [-2.0, -1.0, 4.0, -1.0],
        [-1.0, -2.0, -1.0, 4.0],
    ];
    let mut ri = Vec::with_capacity(16 * k * k);
    let mut ci = Vec::with_capacity(16 * k * k);
    let mut v = Vec::with_capacity(16 * k * k);
    for ey in 0..k {
        for ex in 0..k {
            let n0 = ey * (k + 1) + ex;
            let ids = [n0, n0 + 1, n0 + k + 2, n0 + k + 1];
            for p in 0..4 {
                for q in 0..4 {
                    ri.push(ids[p]);
                    ci.push(ids[q]);
                    v.push(ke[p][q] / 6.0);
                }
            }
        }
    }
    (nodes, ri, ci, v)
}

/// Random triplets in random order with repeats (a scattered assembly).
pub fn random_triplets(n: usize, nnz: usize, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut ri = Vec::with_capacity(nnz);
    let mut ci = Vec::with_capacity(nnz);
    let mut v = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        ri.push(rng.below(n));
        ci.push(rng.below(n));
        v.push(rng.unit());
    }
    (ri, ci, v)
}

/// A random permutation (Fisher–Yates).
pub fn random_perm(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        p.swap(i, j);
    }
    p
}

/// The matrix scrambled by a random symmetric permutation (a mesh numbered badly).
pub fn scrambled(a: &Csr, seed: u64) -> Csr {
    let p = random_perm(a.nrows, seed);
    Csr::from_ours(&a.ours().permute(&p, &p))
}

pub fn ones(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

pub fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect()
}

/// The suite by name, for the Python side (written out by `dump`).
pub fn by_name(name: &str) -> Csr {
    let (fam, arg) = name.split_once('-').unwrap_or((name, ""));
    let num = |s: &str| s.parse::<usize>().unwrap();
    match fam {
        "lap2d" => lap2d(num(arg)),
        "lap3d" => lap3d(num(arg)),
        "ani2d" => ani2d(num(arg)),
        "ani3d" => ani3d(num(arg)),
        "elast3d" => elast3d(num(arg)),
        "convdiff2d" => convdiff2d(num(arg), 1.5, 3.5),
        "upwind2d" => convdiff2d(num(arg), 0.3, 0.2),
        // Mild convection: nearly normal, eigenvalues known in closed form and well conditioned.
        "cd2d" => convdiff2d(num(arg), 0.05, 0.03),
        "convdiff3d" => convdiff3d(num(arg), 0.4),
        "randunsym" => random_unsym(num(arg), 8, 7),
        "circuit" => circuit(num(arg), 11),
        "randspd" => random_spd(num(arg), 4, 13),
        "tridiag" => tridiag(num(arg)),
        "arrow" => arrow(num(arg)),
        "saddle" => {
            let k = num(arg);
            saddle(k, k * k / 5, 17)
        }
        "curve" => curve(num(arg)),
        "tallband" => tallband(num(arg), 19),
        "gridls" => gridls(num(arg)),
        "randtall" => random_tall(num(arg), 4, 23),
        _ => panic!("unknown matrix {name}"),
    }
}

/// The COO assembly of a CSR matrix in a scrambled order (for the assembly rows).
pub fn _coo_of(a: &Csr) -> CooMatrix<f64> {
    a.ours().to_coo()
}
