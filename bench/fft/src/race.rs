//! One measurement: our variants and the references, timed together, written as rows.

use crate::timing::{self, Cfg, Job};
use crate::util::Out;

pub struct Race<'a> {
    jobs: Vec<Job<'a>>,
    ours: Vec<(usize, String)>,
    refs: Vec<(String, String, usize, usize, Option<String>)>,
}

impl<'a> Race<'a> {
    pub fn new() -> Race<'a> {
        Race {
            jobs: Vec::new(),
            ours: Vec::new(),
            refs: Vec::new(),
        }
    }

    /// Adds one of our variants, named by its operation; returns its handle.
    pub fn ours(&mut self, op: &str, f: impl FnMut() + 'a) -> usize {
        self.jobs.push(Box::new(f));
        self.ours.push((self.jobs.len() - 1, op.to_string()));
        self.ours.len() - 1
    }

    /// Adds a reference compared with our variant `ours`; returns the row's handle.
    pub fn reference(
        &mut self,
        name: &str,
        notes: &str,
        ours: usize,
        f: impl FnMut() + 'a,
    ) -> usize {
        self.jobs.push(Box::new(f));
        self.refs
            .push((name.to_string(), notes.to_string(), ours, self.jobs.len() - 1, None));
        self.refs.len() - 1
    }

    /// Another row for the reference of the row `row` (timed once), against our variant `ours`,
    /// in the group `group`.
    pub fn pair(&mut self, row: usize, ours: usize, group: &str) {
        let (name, notes, _, job, _) = self.refs[row].clone();
        self.refs.push((name, notes, ours, job, Some(group.to_string())));
    }

    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Times everything and writes one row per reference; `per` is the items per call.
    pub fn finish(mut self, o: &mut Out, cfg: &Cfg, group: &str, params: &str, per: f64) {
        // FFTBENCH_FILTER="n=16|n=32 T=f64": rows whose params hold every word of one of the
        // alternatives (words separated by spaces, alternatives by `|`).
        if let Ok(f) = std::env::var("FFTBENCH_FILTER") {
            let words: Vec<&str> = params.split_whitespace().collect();
            let hit = f
                .split('|')
                .any(|alt| alt.split_whitespace().all(|w| words.contains(&w)));
            if !hit {
                return;
            }
        }
        if self.refs.is_empty() {
            eprintln!("  (no reference for {group} {params})");
            return;
        }
        let stats = timing::run(&mut self.jobs, cfg);
        for (name, notes, ours, job, g) in &self.refs {
            let (oj, op) = &self.ours[*ours];
            let g = g.as_deref().unwrap_or(group);
            o.row(g, op, params, stats[*oj], name, stats[*job], per, notes);
        }
    }
}
