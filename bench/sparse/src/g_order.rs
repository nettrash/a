//! Orderings: their time, and their quality (the fill they leave, the band and profile).

use crate::mats::{self, Csr};
use crate::ss::{self, Csc32};
use crate::timing::{Contender, keep, note, race, row_raw, rows, selected};
use a::math::sparse::{
    SparseMatrix, amd, colamd, elimination_tree, nested_dissection, postorder, reverse_cuthill_mckee, Ordering,
};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SymbolicSparseColMat;

const G: &str = "ordering";

fn params(name: &str, a: &Csr) -> String {
    format!("{name} n={} nnz={}", a.nrows, a.nnz())
}

/// The pattern of A + Aᵀ without its diagonal, as sorted adjacency lists.
fn sym_adjacency(a: &Csr) -> Vec<Vec<usize>> {
    let n = a.nrows;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for k in a.ptr[i]..a.ptr[i + 1] {
            let j = a.idx[k];
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for l in adj.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    adj
}

/// nnz(L) of the Cholesky factor of P·(A + Aᵀ)·Pᵀ, diagonal included, by the elimination tree and
/// row subtrees (O(nnz(L)) time) — the same count for every ordering compared.
pub fn chol_fill(adj: &[Vec<usize>], p: &[usize]) -> u64 {
    let n = adj.len();
    let mut pinv = vec![0usize; n];
    for (k, &v) in p.iter().enumerate() {
        pinv[v] = k;
    }
    // Rows of the permuted lower triangle.
    let mut low: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (old, l) in adj.iter().enumerate() {
        let i = pinv[old];
        for &o in l {
            let j = pinv[o];
            if j < i {
                low[i].push(j);
            }
        }
    }
    // Elimination tree (Liu, path compression).
    const NONE: usize = usize::MAX;
    let mut parent = vec![NONE; n];
    let mut anc = vec![NONE; n];
    for i in 0..n {
        for &k0 in &low[i] {
            let mut k = k0;
            while k != NONE && k < i {
                let next = anc[k];
                anc[k] = i;
                if next == NONE {
                    parent[k] = i;
                    break;
                }
                k = next;
            }
        }
    }
    // Row counts by marking the row subtrees.
    let mut mark = vec![NONE; n];
    let mut count = n as u64;
    for i in 0..n {
        mark[i] = i;
        for &k0 in &low[i] {
            let mut k = k0;
            while mark[k] != i {
                mark[k] = i;
                count += 1;
                k = parent[k];
                if k == NONE {
                    break;
                }
            }
        }
    }
    count
}

/// The pattern of AᵀA (for the column orderings' fill).
fn ata_adjacency(a: &Csr) -> Vec<Vec<usize>> {
    let n = a.ncols;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..a.nrows {
        let r = &a.idx[a.ptr[i]..a.ptr[i + 1]];
        for &x in r {
            for &y in r {
                if x != y {
                    adj[x].push(y);
                }
            }
        }
    }
    for l in adj.iter_mut() {
        l.sort_unstable();
        l.dedup();
    }
    adj
}

fn band_profile(adj: &[Vec<usize>], p: &[usize]) -> (usize, u64) {
    let n = adj.len();
    let mut pinv = vec![0usize; n];
    for (k, &v) in p.iter().enumerate() {
        pinv[v] = k;
    }
    let mut band = 0;
    let mut profile = 0u64;
    for (old, l) in adj.iter().enumerate() {
        let i = pinv[old];
        let mut first = i;
        for &o in l {
            let j = pinv[o];
            band = band.max(i.abs_diff(j));
            first = first.min(j);
        }
        profile += (i - first + 1) as u64;
    }
    (band, profile)
}

fn is_perm(p: &[usize]) -> bool {
    let mut seen = vec![false; p.len()];
    p.iter().all(|&v| v < p.len() && !std::mem::replace(&mut seen[v], true))
}

fn to_usize(p: &[i32]) -> Vec<usize> {
    p.iter().map(|&v| v as usize).collect()
}

fn faer_pattern(a: &Csr) -> SymbolicSparseColMat<usize> {
    // The pattern of a (symmetric here) as compressed columns = its rows.
    let t = a.transpose();
    SymbolicSparseColMat::new_checked(a.nrows, a.ncols, t.ptr, None, t.idx)
}

fn faer_amd(pat: &SymbolicSparseColMat<usize>) -> Vec<usize> {
    let n = pat.nrows();
    let mut perm = vec![0usize; n];
    let mut inv = vec![0usize; n];
    let req = faer::sparse::linalg::amd::order_scratch::<usize>(n, pat.compute_nnz());
    let mut buf = MemBuffer::new(req);
    faer::sparse::linalg::amd::order(&mut perm, &mut inv, pat.as_ref(), Default::default(), MemStack::new(&mut buf)).unwrap();
    perm
}

fn faer_colamd(pat: &SymbolicSparseColMat<usize>) -> Vec<usize> {
    let (m, n) = (pat.nrows(), pat.ncols());
    let mut perm = vec![0usize; n];
    let mut inv = vec![0usize; n];
    let req = faer::sparse::linalg::colamd::order_scratch::<usize>(m, n, pat.compute_nnz());
    let mut buf = MemBuffer::new(req);
    faer::sparse::linalg::colamd::order(&mut perm, &mut inv, pat.as_ref(), Default::default(), MemStack::new(&mut buf)).unwrap();
    perm
}

pub fn run(quick: bool) {
    let sym: Vec<(&str, bool)> = if quick {
        vec![("lap2d-100", true), ("lap3d-22", false)]
    } else {
        vec![
            ("lap2d-316", true),
            ("lap2d-1000", true),
            ("lap3d-46", false),
            ("elast3d-16", false),
            ("randspd-100000", false),
            ("circuit-100000", false),
        ]
    };
    for (name, scramble) in sym.iter().filter(|(n, _)| selected(n)) {
        let mut a = mats::by_name(name);
        let label = if *scramble {
            a = mats::scrambled(&a, 5);
            format!("{name} scrambled")
        } else {
            name.to_string()
        };
        amd_rows(&label, &a);
        symamd_rows(&label, &a);
        if !name.starts_with("randspd") && !name.starts_with("circuit") {
            nd_rows(&label, &a);
        }
    }
    let cols: Vec<&str> = if quick {
        vec!["convdiff2d-100", "randtall-3000"]
    } else {
        vec!["convdiff2d-316", "circuit-10000", "randunsym-10000", "randtall-30000", "gridls-183", "convdiff3d-30"]
    };
    for name in cols.iter().filter(|n| selected(n)) {
        colamd_rows(name, &mats::by_name(name));
    }
    let rcm: Vec<&str> = if quick { vec!["lap2d-100"] } else { vec!["lap2d-316", "lap3d-46", "circuit-100000", "lap2d-1000"] };
    for name in rcm.iter().filter(|n| selected(n)) {
        let a = mats::scrambled(&mats::by_name(name), 6);
        rcm_rows(&format!("{name} scrambled"), &a);
    }
    for name in (if quick { vec!["lap2d-316"] } else { vec!["lap2d-1000", "lap3d-100", "elast3d-30"] }).into_iter().filter(|n| selected(n)) {
        etree_rows(name, &mats::by_name(name));
    }
}

fn amd_rows(label: &str, a: &Csr) {
    note(&format!("amd {label}"));
    let o = a.ours();
    let cs = Csc32::of(a);
    let pat = faer_pattern(a);
    let adj = sym_adjacency(a);
    let p_ours = amd(&o);
    let mut p_ss = vec![0i32; a.nrows];
    unsafe { ss::ss_amd(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), p_ss.as_mut_ptr()) };
    let p_ss = to_usize(&p_ss);
    let mut p_camd = vec![0i32; a.nrows];
    unsafe { ss::ss_camd(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), p_camd.as_mut_ptr()) };
    let p_camd = to_usize(&p_camd);
    let p_faer = faer_amd(&pat);
    assert!(is_perm(&p_ours) && is_perm(&p_ss) && is_perm(&p_faer) && is_perm(&p_camd));
    let f_ours = chol_fill(&adj, &p_ours);
    let pp = params(label, a);
    for (name, p) in [("SuiteSparse AMD 3.3", &p_ss), ("SuiteSparse CAMD 3.3 (no constraints)", &p_camd), ("faer 0.24 AMD", &p_faer)] {
        let f = chol_fill(&adj, p);
        row_raw(G, "amd fill nnz(L)", &format!("{pp} [values: nnz(L), not ns]"), f_ours as f64, name, f as f64, "quality: lower is better");
    }
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(amd(o))));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; a.nrows];
        c.push(Contender::new("SuiteSparse AMD 3.3", move || unsafe {
            keep(ss::ss_amd(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr()));
        }));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; a.nrows];
        c.push(Contender::new("SuiteSparse CAMD 3.3 (no constraints)", move || unsafe {
            keep(ss::ss_camd(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr()));
        }));
    }
    {
        let pat = &pat;
        c.push(Contender::new("faer 0.24 AMD", move || keep(faer_amd(pat))));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "ours ends with a postorder of the elimination tree".into();
    notes[1] = "int32; no postorder".into();
    notes[2] = "int32; no postorder".into();
    rows(G, "amd", &pp, &st, &notes);
}

fn symamd_rows(label: &str, a: &Csr) {
    note(&format!("symamd {label}"));
    let o = a.ours();
    let cs = Csc32::of(a);
    let adj = sym_adjacency(a);
    let p_ours = Ordering::Colamd.symmetric(&o);
    let mut p = vec![0i32; a.nrows + 1];
    unsafe { ss::ss_symamd(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), p.as_mut_ptr()) };
    let p_ss = to_usize(&p[..a.nrows]);
    assert!(is_perm(&p_ours) && is_perm(&p_ss));
    let pp = params(label, a);
    row_raw(G, "symamd fill nnz(L)", &format!("{pp} [values: nnz(L), not ns]"), chol_fill(&adj, &p_ours) as f64, "SuiteSparse SYMAMD 3.3", chol_fill(&adj, &p_ss) as f64, "quality: lower is better");
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(Ordering::Colamd.symmetric(o))));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; a.nrows + 1];
        c.push(Contender::new("SuiteSparse SYMAMD 3.3", move || unsafe {
            keep(ss::ss_symamd(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr()));
        }));
    }
    let st = race(&mut c);
    rows(G, "Ordering::Colamd.symmetric (SYMAMD)", &pp, &st, &vec![String::new(); st.len()]);
}

fn nd_rows(label: &str, a: &Csr) {
    note(&format!("nd {label}"));
    let o = a.ours();
    let cs = Csc32::of(a);
    let adj = sym_adjacency(a);
    let n = a.nrows;
    let p_ours = nested_dissection(&o);
    let mut p = vec![0i32; n];
    unsafe { ss::ss_metis(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), p.as_mut_ptr(), 0) };
    let p_metis = to_usize(&p);
    let (mut cp, mut cm) = (vec![0i32; n], vec![0i32; n]);
    unsafe { ss::ss_nesdis(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), p.as_mut_ptr(), cp.as_mut_ptr(), cm.as_mut_ptr()) };
    let p_nesdis = to_usize(&p);
    assert!(is_perm(&p_ours) && is_perm(&p_metis) && is_perm(&p_nesdis));
    let pp = params(label, a);
    let f_ours = chol_fill(&adj, &p_ours);
    let f_amd = chol_fill(&adj, &amd(&o));
    for (name, q) in [("METIS 5 NodeND (via CHOLMOD)", &p_metis), ("CHOLMOD NESDIS", &p_nesdis)] {
        row_raw(G, "nested_dissection fill nnz(L)", &format!("{pp} [values: nnz(L), not ns]"), f_ours as f64, name, chol_fill(&adj, q) as f64, &format!("quality: lower is better; ours amd gives {f_amd}"));
    }
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(nested_dissection(o))));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; n];
        c.push(Contender::new("METIS 5 NodeND (via CHOLMOD)", move || unsafe {
            keep(ss::ss_metis(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr(), 0));
        }));
    }
    {
        let s = &cs;
        let (mut p, mut cp, mut cm) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        c.push(Contender::new("CHOLMOD NESDIS", move || unsafe {
            keep(ss::ss_nesdis(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr(), cp.as_mut_ptr(), cm.as_mut_ptr()));
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[1] = "METIS bundled in CHOLMOD 5 (SuiteSparse_metis); multilevel".into();
    notes[2] = "METIS bisection + CAMD on the parts".into();
    rows(G, "nested_dissection", &pp, &st, &notes);
}

fn colamd_rows(name: &str, a: &Csr) {
    note(&format!("colamd {name}"));
    let o = a.ours();
    let cs = Csc32::of(a);
    let t = a.transpose();
    let fpat = SymbolicSparseColMat::new_checked(a.nrows, a.ncols, t.ptr.clone(), None, t.idx.clone());
    let p_ours = colamd(&o);
    let mut p = vec![0i32; a.ncols + 1];
    unsafe { ss::ss_colamd(cs.m, cs.n, cs.p.as_ptr(), cs.i.as_ptr(), p.as_mut_ptr()) };
    let p_ss = to_usize(&p[..a.ncols]);
    let p_faer = faer_colamd(&fpat);
    assert!(is_perm(&p_ours) && is_perm(&p_ss) && is_perm(&p_faer));
    let adj = ata_adjacency(a);
    let pp = format!("{name} {}×{} nnz={}", a.nrows, a.ncols, a.nnz());
    let f = chol_fill(&adj, &p_ours);
    for (rn, q) in [("SuiteSparse COLAMD 3.3", &p_ss), ("faer 0.24 COLAMD", &p_faer)] {
        row_raw(G, "colamd fill nnz(chol(AᵀA))", &format!("{pp} [values: nnz, not ns]"), f as f64, rn, chol_fill(&adj, q) as f64, "quality: lower is better");
    }
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(colamd(o))));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; a.ncols + 1];
        c.push(Contender::new("SuiteSparse COLAMD 3.3", move || unsafe {
            keep(ss::ss_colamd(s.m, s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr()));
        }));
    }
    {
        let fpat = &fpat;
        c.push(Contender::new("faer 0.24 COLAMD", move || keep(faer_colamd(fpat))));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "ours ends with a postorder of the column elimination tree".into();
    notes[1] = "int32; includes the copy into its workspace; postordered".into();
    rows(G, "colamd", &pp, &st, &notes);
}

fn rcm_rows(label: &str, a: &Csr) {
    note(&format!("rcm {label}"));
    let o = a.ours();
    let sp = crate::conv::sprs_csr(a);
    let adj = sym_adjacency(a);
    let p_ours = reverse_cuthill_mckee(&o);
    let ord = sprs::linalg::reverse_cuthill_mckee(sp.view());
    // sprs gives perm as new -> old in `perm` (its PermOwned's vec).
    let p_sprs: Vec<usize> = ord.perm.vec();
    assert!(is_perm(&p_ours) && is_perm(&p_sprs));
    let pp = params(label, a);
    let (bo, po) = band_profile(&adj, &p_ours);
    let (bs, ps) = band_profile(&adj, &p_sprs);
    let (bs2, ps2) = band_profile(&adj, &inverse(&p_sprs));
    let (bs, ps) = if ps2 < ps { (bs2, ps2) } else { (bs, ps) };
    row_raw(G, "reverse_cuthill_mckee band", &format!("{pp} [values: band width, not ns]"), bo as f64, "sprs 0.11 RCM", bs as f64, "quality: lower is better");
    row_raw(G, "reverse_cuthill_mckee profile", &format!("{pp} [values: profile, not ns]"), po as f64, "sprs 0.11 RCM", ps as f64, "quality: lower is better");
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(reverse_cuthill_mckee(o))));
    }
    {
        let sp = &sp;
        c.push(Contender::new("sprs 0.11 RCM", move || keep(sprs::linalg::reverse_cuthill_mckee(sp.view()))));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "ours from pseudo-peripheral vertices".into();
    rows(G, "reverse_cuthill_mckee", &pp, &st, &notes);
}

fn inverse(p: &[usize]) -> Vec<usize> {
    let mut q = vec![0; p.len()];
    for (k, &v) in p.iter().enumerate() {
        q[v] = k;
    }
    q
}

fn etree_rows(name: &str, a: &Csr) {
    note(&format!("etree {name}"));
    let o = a.ours();
    let n = a.nrows;
    let cs = Csc32::of(a);
    let par = elimination_tree(&o);
    let mut pc = vec![0i32; n];
    unsafe { ss::ss_etree(cs.n, cs.p.as_ptr(), cs.i.as_ptr(), pc.as_mut_ptr()) };
    for j in 0..n {
        let w = par[j].map(|v| v as i32).unwrap_or(-1);
        assert_eq!(w, pc[j], "etree differs at {j}");
    }
    let pp = params(name, a);
    let mut c: Vec<Contender> = Vec::new();
    {
        let o = &o;
        c.push(Contender::new("ours", move || keep(elimination_tree(o))));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; n];
        c.push(Contender::new("CHOLMOD etree", move || unsafe {
            keep(ss::ss_etree(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr()));
        }));
    }
    {
        let s = &cs;
        let mut p = vec![0i32; n];
        c.push(Contender::new("CSparse cs_etree", move || unsafe {
            keep(ss::ss_cs_etree(s.n, s.p.as_ptr(), s.i.as_ptr(), p.as_mut_ptr()));
        }));
    }
    let st = race(&mut c);
    let mut notes = vec![String::new(); st.len()];
    notes[0] = "Vec<Option<usize>> result".into();
    rows(G, "elimination_tree", &pp, &st, &notes);
    // postorder of that tree.
    let mut c: Vec<Contender> = Vec::new();
    {
        let par = &par;
        c.push(Contender::new("ours", move || keep(postorder(par))));
    }
    {
        let pc = &pc;
        let mut post = vec![0i32; n];
        c.push(Contender::new("CHOLMOD postorder", move || unsafe {
            keep(ss::ss_postorder(n as i32, pc.as_ptr(), post.as_mut_ptr()));
        }));
    }
    {
        let pc = &pc;
        let mut post = vec![0i32; n];
        c.push(Contender::new("CSparse cs_post", move || unsafe {
            keep(ss::ss_cs_post(n as i32, pc.as_ptr(), post.as_mut_ptr()));
        }));
    }
    let st = race(&mut c);
    rows(G, "postorder", &pp, &st, &vec![String::new(); st.len()]);
}

#[allow(dead_code)]
fn _unused(_: SparseMatrix<f64>) {}
