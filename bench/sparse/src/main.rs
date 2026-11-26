//! The scoreboard probe of a::math::sparse. Usage: sparse-bench <group> [--quick]
//! Groups: ops io ordering chol lu qr iter eigen dump. Prints TSV rows (no header) to stdout.

mod conv;
mod mats;
mod ss;
#[cfg(have_accelerate)]
mod accel;
mod timing;
mod g_ops;
mod g_order;
mod g_chol;
mod g_lu;
mod g_qr;
mod g_iter;
mod g_eigen;
mod g_io;
mod kern;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    #[cfg(have_accelerate)]
    unsafe {
        accel::ac_single_threaded();
    }
    match args.get(1).map(String::as_str) {
        Some("ops") => g_ops::run(quick),
        Some("ordering") => g_order::run(quick),
        Some("chol") => g_chol::run(quick),
        Some("lu") => g_lu::run(quick),
        Some("qr") => g_qr::run(quick),
        Some("iter") => g_iter::run(quick),
        Some("eigen") => g_eigen::run(quick),
        Some("io") => g_io::run(quick),
        Some("kern") => kern::run(),
        Some("probe") => {
            // probe <chol|lu|qr> <matrix>: one factorisation of ours, its time and fill (for sizing).
            let a = mats::by_name(&args[3]);
            let o = a.ours();
            let t = std::time::Instant::now();
            let (nnz, extra) = match args[2].as_str() {
                "chol" => { let f = a::math::sparse::SparseCholesky::new(&o).unwrap(); (f.nnz(), f.supernodes()) }
                "lu" => { let f = a::math::sparse::SparseLu::new(&o); (f.nnz(), f.is_singular() as usize) }
                _ => { let f = a::math::sparse::SparseQr::new(&o); (f.nnz(), f.rank()) }
            };
            println!("{} {} n={} nnz={} -> {:.3} s, factor nnz {nnz}, {extra}", args[2], args[3], a.nrows, a.nnz(), t.elapsed().as_secs_f64());
        }
        Some("prof") => {
            // prof <amd|colamd|symamd|nd|etree|chol|lu|qr|lanczos> <matrix> <seconds>: loops one operation of ours
            // (for a sampling profiler).
            let a = mats::by_name(&args[3]);
            let a = if args[2] == "amd" || args[2] == "symamd" || args[2] == "nd" { mats::scrambled(&a, 5) } else { a };
            let o = a.ours();
            let secs: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10.0);
            let chol = (args[2] == "cholsolve").then(|| a::math::sparse::SparseCholesky::new(&o).unwrap());
            let rhs = mats::rand_vec(a.nrows, 3);
            if let Some(f) = &chol {
                println!("supernodes {} nnz {}", f.supernodes(), f.nnz());
            }
            let t = std::time::Instant::now();
            let mut reps = 0;
            while t.elapsed().as_secs_f64() < secs {
                match args[2].as_str() {
                    "cholsolve" => timing::keep(chol.as_ref().unwrap().solve_vec(&rhs)),
                    "amd" => timing::keep(a::math::sparse::amd(&o)),
                    "colamd" => timing::keep(a::math::sparse::colamd(&o)),
                    "symamd" => timing::keep(a::math::sparse::Ordering::Colamd.symmetric(&o)),
                    "nd" => timing::keep(a::math::sparse::nested_dissection(&o)),
                    "etree" => timing::keep(a::math::sparse::elimination_tree(&o)),
                    "chol" => timing::keep(a::math::sparse::SparseCholesky::new(&o)),
                    "lu" => timing::keep(a::math::sparse::SparseLu::new(&o)),
                    "qr" => timing::keep(a::math::sparse::SparseQr::new(&o)),
                    _ => timing::keep(a::math::sparse::lanczos_with(&o, 6, a::math::sparse::LanczosWhich::LargestAlgebraic,
                        &a::math::sparse::EigenOptions::new().with_basis(20).with_tolerance(1e-10))),
                }
                reps += 1;
            }
            println!("{reps} reps in {:.2} s", t.elapsed().as_secs_f64());
        }
        Some("qrcheck") => {
            // qrcheck [matrices…]: our R against SPQR's R of A·Q_c (our column order, fixed, no rank
            // detection), row by row up to the sign of the row, and the least-squares solutions.
            let names: Vec<String> = if args.len() > 2 { args[2..].to_vec() } else {
                ["curve-5000", "curve-50000", "tallband-500", "tallband-5000", "gridls-31", "gridls-100", "gridls-183", "randtall-1000", "lap2d-100"]
                    .iter().map(|s| s.to_string()).collect()
            };
            for name in &names {
                let a = mats::by_name(name);
                let o = a.ours();
                let f = a::math::sparse::SparseQr::new(&o);
                let q = f.column_permutation().to_vec();
                // A·Q_c in compressed columns: column k is column q[k] of A.
                let cs = ss::Csc32::of(&a);
                let mut p64 = vec![0i64];
                let (mut i64_, mut x) = (Vec::new(), Vec::new());
                for &old in &q {
                    let (b, e) = (cs.p[old] as usize, cs.p[old + 1] as usize);
                    i64_.extend(cs.i[b..e].iter().map(|&v| v as i64));
                    x.extend_from_slice(&cs.x[b..e]);
                    p64.push(i64_.len() as i64);
                }
                let (mut rp, mut ri, mut rx) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
                let rank = unsafe { ss::ss_spqr_r(a.nrows as i64, a.ncols as i64, p64.as_ptr(), i64_.as_ptr(), x.as_ptr(), &mut rp, &mut ri, &mut rx) };
                assert!(rank >= 0, "{name}: SPQR failed ({rank})");
                let n = a.ncols;
                let (rp, ri, rx) = unsafe {
                    let rp_s = std::slice::from_raw_parts(rp, n + 1).to_vec();
                    let nz = rp_s[n] as usize;
                    let out = (rp_s, std::slice::from_raw_parts(ri, nz).to_vec(), std::slice::from_raw_parts(rx, nz).to_vec());
                    ss::ss_free(rp as *mut _); ss::ss_free(ri as *mut _); ss::ss_free(rx as *mut _);
                    out
                };
                // SPQR's R by rows.
                let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
                for k in 0..n {
                    for t in rp[k] as usize..rp[k + 1] as usize {
                        rows[ri[t] as usize].push((k, rx[t]));
                    }
                }
                let r = f.r();
                let mut worst = 0.0f64;
                let mut flips = 0usize;
                for j in 0..n {
                    let mut d = std::collections::BTreeMap::<usize, f64>::new();
                    let ours: Vec<(usize, f64)> = r.row(j).collect();
                    let so = ours.iter().find(|e| e.0 == j).map_or(1.0, |e| e.1.signum());
                    let st = rows[j].iter().find(|e| e.0 == j).map_or(1.0, |e| e.1.signum());
                    if so != st { flips += 1; }
                    let norm = rows[j].iter().map(|e| e.1 * e.1).sum::<f64>().sqrt().max(1e-300);
                    for &(c, v) in &ours { *d.entry(c).or_default() += so * v; }
                    for &(c, v) in &rows[j] { *d.entry(c).or_default() -= st * v; }
                    worst = worst.max(d.values().fold(0.0f64, |m, v| m.max(v.abs())) / norm);
                }
                let b = mats::rand_vec(a.nrows, 61);
                let xo = f.solve_vec(&b).expect("ours refused");
                let h = unsafe { ss::ss_spqr(a.nrows as i64, a.ncols as i64, cs.p64().as_ptr(), cs.i64().as_ptr(), cs.x.as_ptr(), 2, -2.0) };
                let mut bb = b.clone();
                let mut xs = vec![0.0; n];
                unsafe { ss::ss_spqr_solve(h, a.nrows as i64, n as i64, 1, bb.as_mut_ptr(), xs.as_mut_ptr()); ss::ss_spqr_free(h) };
                let scale = xs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let dx = xo.iter().zip(&xs).fold(0.0f64, |m, (p, q)| m.max((p - q).abs())) / scale;
                println!("qrcheck\t{name}\trank ours {} SPQR {rank}\tR rows up to sign: max diff/row norm {worst:.2e} ({flips} rows of opposite sign)\tsolution vs SPQR(COLAMD) max rel {dx:.2e}", f.rank());
                assert!(worst < 1e-9 && dx < 1e-8 && f.rank() as i64 == rank, "{name}: disagreement");
            }
        }
        Some("dump") => {
            // dump <matrix name> <path>: the CSR arrays as raw little-endian u64/u64/f64 files.
            let a = mats::by_name(&args[2]);
            let base = &args[3];
            let w = |suffix: &str, bytes: Vec<u8>| std::fs::write(format!("{base}.{suffix}"), bytes).unwrap();
            w("shape", format!("{} {} {}", a.nrows, a.ncols, a.nnz()).into_bytes());
            w("ptr", a.ptr.iter().flat_map(|v| (*v as u64).to_le_bytes()).collect());
            w("idx", a.idx.iter().flat_map(|v| (*v as u64).to_le_bytes()).collect());
            w("val", a.val.iter().flat_map(|v| v.to_le_bytes()).collect());
        }
        _ => {
            eprintln!("usage: sparse-bench <ops|io|ordering|chol|lu|qr|iter|eigen|dump> [--quick]");
            std::process::exit(2);
        }
    }
}
