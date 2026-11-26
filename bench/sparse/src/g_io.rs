//! Matrix Market files, read from and written to memory. (fast_matrix_market through SciPy is
//! in py/scipy_ref.py.)

use crate::mats::{self, Csr};
use crate::timing::{Contender, keep, note, race, rows, selected};
use a::math::sparse::{
    MatrixMarketOptions, MatrixMarketSymmetry, SparseMatrix, read_matrix_market, to_matrix_market, to_matrix_market_with, write_matrix_market,
    write_matrix_market_with,
};

const G: &str = "io";

pub fn run(quick: bool) {
    let set: Vec<&str> = if quick { vec!["lap2d-316"] } else { vec!["lap2d-316", "lap2d-1000", "randunsym-1000000", "elast3d-30"] };
    for name in set.iter().filter(|n| selected(n)) {
        let a = mats::by_name(name);
        io_rows(name, &a, false);
        if !name.starts_with("randunsym") {
            io_rows(name, &a, true);
        }
    }
}

fn io_rows(name: &str, a: &Csr, sym: bool) {
    note(&format!("io {name} sym={sym}"));
    let o = a.ours();
    let opts = if sym { MatrixMarketOptions::new().with_symmetry(MatrixMarketSymmetry::Symmetric) } else { MatrixMarketOptions::new() };
    let text = to_matrix_market_with(&o, &opts).unwrap();
    let bytes = text.as_bytes();
    let pp = format!("{name} n={} nnz={} {} file {:.1} MB", a.nrows, a.nnz(), if sym { "symmetric" } else { "general" }, bytes.len() as f64 / 1e6);
    // Checks: the references read our file into the same matrix.
    let back = SparseMatrix::<f64>::read_matrix_market(bytes).unwrap();
    assert!(back == o);
    let tri: sprs::TriMat<f64> = sprs::io::read_matrix_market_from_bufread(&mut &bytes[..]).unwrap();
    assert!(crate::conv::mat_diff(a, &crate::conv::from_sprs(&tri.to_csr())) == 0.0, "sprs reads differently");
    let nac: nalgebra_sparse::CooMatrix<f64> = nalgebra_sparse::io::load_coo_from_matrix_market_str(&text).unwrap();
    assert!(crate::conv::mat_diff(a, &crate::conv::from_na(&nalgebra_sparse::CsrMatrix::from(&nac))) == 0.0, "nalgebra reads differently");
    // Reading.
    let mut c: Vec<Contender> = Vec::new();
    c.push(Contender::new("ours", move || keep(read_matrix_market::<f64>(bytes).unwrap())));
    c.push(Contender::new("sprs 0.11 read_matrix_market_from_bufread", move || {
        keep(sprs::io::read_matrix_market_from_bufread::<f64, usize, _>(&mut &bytes[..]).unwrap())
    }));
    {
        let t = &text;
        c.push(Contender::new("nalgebra-sparse 0.11 load_coo_from_matrix_market_str", move || keep(nalgebra_sparse::io::load_coo_from_matrix_market_str::<f64>(t).unwrap())));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "to triplets (CooMatrix) from memory; ours rounds each number once from its digits".into();
    notes[1] = "to triplets (TriMat)".into();
    notes[2] = "pest grammar parser".into();
    rows(G, "read_matrix_market (to COO)", &pp, &st, &notes);
    // Reading into compressed rows.
    let mut c: Vec<Contender> = Vec::new();
    c.push(Contender::new("ours", move || keep(SparseMatrix::<f64>::read_matrix_market(bytes).unwrap())));
    c.push(Contender::new("sprs 0.11 (read + TriMat::to_csr)", move || {
        keep(sprs::io::read_matrix_market_from_bufread::<f64, usize, _>(&mut &bytes[..]).unwrap().to_csr::<usize>())
    }));
    {
        let t = &text;
        c.push(Contender::new("nalgebra-sparse 0.11 (load + CsrMatrix::from)", move || {
            keep(nalgebra_sparse::CsrMatrix::from(&nalgebra_sparse::io::load_coo_from_matrix_market_str::<f64>(t).unwrap()))
        }));
    }
    let st = race(&mut c);
    rows(G, "SparseMatrix::read_matrix_market (to CSR)", &pp, &st, &vec![String::new(); st.len()]);
    // Writing.
    let sp = crate::conv::sprs_csr(a);
    let na = crate::conv::na_csr(a);
    let mut c: Vec<Contender> = Vec::new();
    {
        let (o, opts) = (&o, &opts);
        let cap = bytes.len() + 64;
        c.push(Contender::new("ours", move || {
            let mut out = Vec::with_capacity(cap);
            if sym {
                write_matrix_market_with(o, &mut out, opts).unwrap();
            } else {
                write_matrix_market(o, &mut out).unwrap();
            }
            keep(out);
        }));
    }
    if !sym {
        let sp = &sp;
        let cap = bytes.len() + 64;
        c.push(Contender::new("sprs 0.11 write_matrix_market_to_bufwrite", move || {
            let mut out = std::io::BufWriter::new(Vec::with_capacity(cap));
            sprs::io::write_matrix_market_to_bufwrite(&mut out, sp.view()).unwrap();
            keep(out);
        }));
        let na = &na;
        c.push(Contender::new("nalgebra-sparse 0.11 save_to_matrix_market_str", move || keep(nalgebra_sparse::io::save_to_matrix_market_str(na))));
    }
    if c.len() >= 2 {
        let st = race(&mut c);
        let mut notes = vec![String::new(); st.len()];
        notes[0] = "shortest round-trip digits ({:e})".into();
        rows(G, "write_matrix_market", &pp, &st, &notes);
    }
    let _ = to_matrix_market::<f64, SparseMatrix<f64>>;
}
