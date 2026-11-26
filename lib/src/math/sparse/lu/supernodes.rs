//! The supernodal elimination of [`SparseLu`](super::SparseLu): the left-looking method of
//! Gilbert and Peierls, with the columns of `L` that share their rows held together as dense
//! blocks and the columns of the matrix taken a panel at a time, so that most of the arithmetic
//! is that of dense products.
//!
//! The method is the one of Demmel, Eisenstat, Gilbert, Li and Liu, *A supernodal approach to
//! sparse partial pivoting* (SIAM J. Matrix Anal. Appl. 20, 1999). A *supernode* is a run of
//! consecutive columns of `L` whose rows below the diagonal are the same, but for the pivotal
//! row of each next column, which is in the columns before it: their elements are one dense
//! block, a row for each of those rows, the pivotal rows of its own columns first and in order.
//! Whether a column joins the supernode of the one before is decided when its rows are known,
//! after its pivot is chosen: it does when the column before has an element in its pivotal row
//! — the column depends on it — and the two have as many rows that are not pivotal, which then
//! makes them the same rows.
//!
//! A column that depends on a column of a supernode depends on every later column of it,
//! because the earlier column of the supernode has an element in the pivotal row of each later
//! one. So the part of a column of `U` in the pivotal rows of a supernode is a dense *segment*,
//! from the first row the column reaches to the last row of the supernode, and the search that
//! finds what a column depends on goes over supernodes instead of columns. Its update by a
//! supernode is then a dense triangular solve of its segment with the block on the diagonal of
//! the supernode, and the product of the block below it with the segment, subtracted at the
//! rows of the supernode.
//!
//! The columns are taken in *panels* of [`PANEL`] consecutive ones. A panel's columns are
//! searched first in the supernodes before it, and each of those supernodes updates every
//! column of the panel that depends on it at once — the segments padded with zeros to one
//! length — so that the product is one of two dense blocks, which [`gemm`] computes at the
//! speed of a dense product. Then the columns of the panel are finished one after the other:
//! the search and the update of each by the supernodes of the columns of the panel before it,
//! its pivot, and whether it joins a supernode. The pivots are chosen as by the column method,
//! from the whole column once every update is in, so the factors are the same but for the
//! rounding of the sums, which the blocks group otherwise.
//!
//! The search over the supernodes is pruned as the column search is (Eisenstat and Liu): a
//! supernode is pruned by the first later column whose pivotal row it has an element in and
//! that depends on it, after which the search follows from it only the rows that were pivotal by
//! then — and of those only one for each supernode they are in, the pivotal row of its earliest
//! step, which takes the search to the same segment. The pruned rows are a copy, so the numbers
//! of a block stay in the order of its rows.
//!
//! What is kept of a supernode once the elimination ends are the columns of `L` and `U` that
//! the column method would have made, with the same rows, so everything that reads the factors
//! reads them as before.

use super::{Eliminated, Elimination, NONE, Pivoting, Watch};
use crate::math::complex::Float;
use crate::math::matrix::Scalar;
use crate::math::matrix::blas::{Op, Side, Uplo, gemm, trsm};

/// The columns of a panel (SuperLU's default is 12).
// Measured on an Apple M4 Max (`SparseLu::new` on the matrices of [`WIDEST`], `f64` and
// `Complex<f64>`; sweep 8 to 48, 24 against 16 twice): 12 to 24 are as fast within 2 % in `f64`,
// 24 taking up to 3 % off in `Complex<f64>` but 1.5 % more for the random matrix of order 1000 in
// `f64`; 8 costs up to 6 %, 48 up to 4 %.
pub(super) const PANEL: usize = 16;
const _: () = assert!(PANEL <= 64, "a panel's columns are the bits of a `u64`");

/// The most columns a supernode takes, which bounds the triangle of a solve.
// Measured on an Apple M4 Max (`SparseLu::new` on 2-D and 3-D grids, elasticity meshes, circuits
// and random matrices of orders 1000 to 100 000; sweep 64 to 1024): 128 and 256 are as fast
// (within 1 %); 64 loses up to 7 % (circuits, elasticity), 512 and 1024 up to 3 %.
const WIDEST: usize = 256;

/// The smallest update, in rows below times segment length times columns, that goes to `gemm`
/// rather than to loops.
// Measured on an Apple M4 Max (`SparseLu::new`, the matrices of [`WIDEST`], `f64` and
// `Complex<f64>`; sweep 64 to 4096): flat from 64 to 512 (within 1.4 %); 1024 loses up to 1.5 %,
// 4096 up to 7 %.
const BLOCKED: usize = 256;

/// The longest segment that is solved by loops rather than by `trsm`.
// Measured on an Apple M4 Max (`SparseLu::new`, the matrices of [`WIDEST`], `f64` and
// `Complex<f64>`; sweep 4 to 48): flat from 4 to 16 (within 1 %); 24 loses up to 1.2 %, 48 up
// to 2.5 %.
const SHORT: usize = 12;

/// The supernodes found so far.
struct Supernodes {
    /// The first step of each supernode.
    first: Vec<usize>,
    /// Where the rows of each supernode begin in `rows`, and where the last ends.
    row_ptrs: Vec<usize>,
    /// The rows of `A` of each supernode: the pivotal rows of its columns, in order, then the
    /// rows below them.
    rows: Vec<usize>,
    /// Where the block of each supernode begins in the elements.
    value_ptrs: Vec<usize>,
    /// Where the rows a search follows from each closed supernode begin in `followed`.
    follow_ptrs: Vec<usize>,
    /// Where they end: all its rows below the diagonal until it is pruned, then those that were
    /// pivotal by then.
    follow_ends: Vec<usize>,
    /// The rows a search follows from the closed supernodes.
    followed: Vec<usize>,
    /// Whether each supernode has been pruned.
    pruned: Vec<bool>,
    /// Whether the last supernode can still take columns.
    open: bool,
}

impl Supernodes {
    /// How many supernodes there are.
    fn count(&self) -> usize {
        self.first.len()
    }

    /// Whether `s` is the last supernode while it can still take columns.
    fn is_open(&self, s: usize) -> bool {
        self.open && s + 1 == self.count()
    }

    /// The columns of `s` once `done` steps are done.
    fn width(&self, s: usize, done: usize) -> usize {
        if s + 1 < self.count() {
            self.first[s + 1] - self.first[s]
        } else {
            done - self.first[s]
        }
    }

    /// The rows of `s`.
    fn rows_of(&self, s: usize) -> &[usize] {
        &self.rows[self.row_ptrs[s]..self.row_ptrs[s + 1]]
    }

    /// The rows a search follows from `s` once `done` steps are done: those below its diagonal
    /// while it is open, and the pruned copy of them once it is closed.
    fn followed_from(&self, s: usize, done: usize) -> &[usize] {
        if self.is_open(s) {
            &self.rows_of(s)[self.width(s, done)..]
        } else {
            &self.followed[self.follow_ptrs[s]..self.follow_ends[s]]
        }
    }

    /// Closes the open supernode `s` once `done` steps are done: the rows below its diagonal
    /// are copied to be followed by the searches, and pruned.
    fn close(&mut self, s: usize, done: usize) {
        let below = self.row_ptrs[s] + self.width(s, done)..self.row_ptrs[s + 1];
        self.follow_ptrs[s] = self.followed.len();
        self.followed.extend_from_slice(&self.rows[below]);
        self.follow_ends[s] = self.followed.len();
        self.open = false;
    }
}

/// The working memory of the searches of a panel.
struct Search {
    /// The step whose first search last reached each row: what tells a row the search has seen
    /// while it runs.
    mark: Vec<usize>,
    /// The place of each row among the rows the panel reaches, or [`NONE`].
    local: Vec<usize>,
    /// The rows the panel reaches, in the order of their places.
    reached: Vec<usize>,
    /// For each place, the columns of the panel that reach its row, a bit each.
    members: Vec<u64>,
    /// The place of each supernode among those the panel's first searches reach, or [`NONE`].
    slot: Vec<usize>,
    /// The supernode of each place.
    slots: Vec<usize>,
    /// For each place and each column of the panel, the first step of the supernode the column
    /// reaches, or [`NONE`]: [`PANEL`] to a place.
    starts: Vec<usize>,
    /// The first step any column of the panel reaches in the supernode of each place.
    low: Vec<usize>,
    /// The columns of the panel that reach the supernode of each place, a bit each.
    reaching: Vec<u64>,
    /// The places each column of the panel reaches.
    places_of: Vec<Vec<usize>>,
    /// The last step of the supernode of each place when the panel began.
    last: Vec<usize>,
    /// Whether the supernode of each place has been finished by a search.
    finished: Vec<bool>,
    /// The supernodes of the places in the order the searches finished them.
    order: Vec<usize>,
    /// The supernodes a search is in the middle of, and where it goes on in the rows of each.
    stack: Vec<(usize, usize)>,
    /// The rows of each column of the panel that are not pivotal (or were not, when the panel
    /// began): those of `L`, its pivotal row among them.
    structs: Vec<Vec<usize>>,
    /// How many of them each column's first search found.
    found: [usize; PANEL],
    /// The step whose second search last reached each supernode.
    inner_mark: Vec<usize>,
    /// The first step the second search reached in each supernode.
    inner_start: Vec<usize>,
    /// The supernodes the second search of a column reached, in the order it finished them.
    inner_order: Vec<usize>,
}

impl Search {
    /// Gives `row` a place among the rows of the panel, unless it has one.
    #[inline]
    fn place(&mut self, row: usize) {
        if self.local[row] == NONE {
            self.local[row] = self.reached.len();
            self.reached.push(row);
            self.members.push(0);
        }
    }

    /// Gives `row` a place, and records that column `c` reaches it.
    #[inline]
    fn join(&mut self, row: usize, c: usize) {
        self.place(row);
        self.members[self.local[row]] |= 1 << c;
    }

    /// Whether column `c` of the panel reaches `row`.
    #[inline]
    fn reaches(&self, row: usize, c: usize) -> bool {
        match self.local[row] {
            NONE => false,
            place => self.members[place] >> c & 1 == 1,
        }
    }

    /// The place of supernode `s`, given one if it has none: `last` is its last step.
    #[inline]
    fn slot_of(&mut self, s: usize, last: usize) -> usize {
        let t = self.slot[s];
        if t != NONE {
            return t;
        }
        let t = self.slots.len();
        self.slot[s] = t;
        self.slots.push(s);
        self.starts.extend([NONE; PANEL]);
        self.low.push(NONE);
        self.reaching.push(0);
        self.last.push(last);
        self.finished.push(false);
        t
    }

    /// Records that column `c` reaches step `k` of the supernode of place `t`: whether it is the
    /// first time it reaches that supernode.
    #[inline]
    fn reach_slot(&mut self, t: usize, c: usize, k: usize) -> bool {
        self.low[t] = self.low[t].min(k);
        let start = &mut self.starts[t * PANEL + c];
        if *start != NONE {
            *start = (*start).min(k);
            return false;
        }
        *start = k;
        self.reaching[t] |= 1 << c;
        self.places_of[c].push(t);
        true
    }

    /// The first search of column `c` of the panel that begins at step `p0`, from the rows
    /// `roots` of its column of `A`: the supernodes before the panel it depends on, with the
    /// first step of each it reaches, and the rows it reaches that are not pivotal.
    fn outer(&mut self, c: usize, p0: usize, roots: &[usize], g: &Graph<'_>) {
        let stamp = p0 + c;
        for &root in roots {
            if self.mark[root] == stamp {
                continue;
            }
            self.mark[root] = stamp;
            self.join(root, c);
            let k = g.step_of_row[root];
            if k == NONE {
                self.structs[c].push(root);
                continue;
            }
            let s = g.supernode_of[k];
            let t = self.slot_of(s, g.last(s, p0));
            if !self.reach_slot(t, c, k) {
                continue;
            }
            self.stack.push((s, 0));
            while let Some(&(s, at)) = self.stack.last() {
                let rows = g.supernodes.followed_from(s, p0);
                let mut at = at;
                let mut next = NONE;
                while at < rows.len() {
                    let row = rows[at];
                    at += 1;
                    if self.mark[row] == stamp {
                        continue;
                    }
                    self.mark[row] = stamp;
                    self.join(row, c);
                    let k = g.step_of_row[row];
                    if k == NONE {
                        self.structs[c].push(row);
                        continue;
                    }
                    let s2 = g.supernode_of[k];
                    let t2 = self.slot_of(s2, g.last(s2, p0));
                    if !self.reach_slot(t2, c, k) {
                        continue;
                    }
                    next = s2;
                    break;
                }
                if let Some(top) = self.stack.last_mut() {
                    top.1 = at;
                }
                if next == NONE {
                    self.stack.pop();
                    let t = self.slot[s];
                    if !self.finished[t] {
                        self.finished[t] = true;
                        self.order.push(s);
                    }
                } else {
                    self.stack.push((next, 0));
                }
            }
        }
    }

    /// The second search of column `c` of the panel, at step `j`: the supernodes of the panel's
    /// columns before it that it depends on, from the rows of the first search that have become
    /// pivotal since; the rows it reaches that the first did not are added to its rows.
    fn inner(&mut self, c: usize, j: usize, g: &Graph<'_>) {
        self.inner_order.clear();
        for at in 0..self.found[c] {
            let root = self.structs[c][at];
            let k = g.step_of_row[root];
            if k == NONE {
                continue;
            }
            let s = g.supernode_of[k];
            if self.inner_mark[s] == j {
                self.inner_start[s] = self.inner_start[s].min(k);
                continue;
            }
            self.inner_mark[s] = j;
            self.inner_start[s] = k;
            self.stack.push((s, 0));
            while let Some(&(s, at)) = self.stack.last() {
                let rows = g.supernodes.followed_from(s, j);
                let mut at = at;
                let mut next = NONE;
                while at < rows.len() {
                    let row = rows[at];
                    at += 1;
                    let k = g.step_of_row[row];
                    if k == NONE {
                        debug_assert!(self.local[row] != NONE);
                        if !self.reaches(row, c) {
                            self.join(row, c);
                            self.structs[c].push(row);
                        }
                        continue;
                    }
                    let s2 = g.supernode_of[k];
                    if self.inner_mark[s2] == j {
                        self.inner_start[s2] = self.inner_start[s2].min(k);
                        continue;
                    }
                    self.inner_mark[s2] = j;
                    self.inner_start[s2] = k;
                    next = s2;
                    break;
                }
                if let Some(top) = self.stack.last_mut() {
                    top.1 = at;
                }
                if next == NONE {
                    self.stack.pop();
                    self.inner_order.push(s);
                } else {
                    self.stack.push((next, 0));
                }
            }
        }
    }

    /// Forgets the panel.
    fn reset(&mut self) {
        for &row in &self.reached {
            self.local[row] = NONE;
        }
        for &s in &self.slots {
            self.slot[s] = NONE;
        }
        self.reached.clear();
        self.members.clear();
        self.slots.clear();
        self.starts.clear();
        self.low.clear();
        self.reaching.clear();
        for places in &mut self.places_of {
            places.clear();
        }
        self.last.clear();
        self.finished.clear();
        self.order.clear();
        for rows in &mut self.structs {
            rows.clear();
        }
    }
}

/// What the searches read of the elimination so far.
struct Graph<'a> {
    supernodes: &'a Supernodes,
    step_of_row: &'a [usize],
    supernode_of: &'a [usize],
}

impl Graph<'_> {
    /// The last step of supernode `s` once `done` steps are done.
    fn last(&self, s: usize, done: usize) -> usize {
        self.supernodes.first[s] + self.supernodes.width(s, done) - 1
    }
}

/// One supernode's update of some columns of the panel: its block (from where it begins, of
/// `height` rows), its rows, the steps `low..=last` of its columns that the update uses, and its
/// first step `first`.
struct Source<'a, T> {
    block: &'a [T],
    height: usize,
    rows: &'a [usize],
    first: usize,
    low: usize,
    last: usize,
}

/// The working memory of the updates.
struct Scratch<T> {
    /// The segments of the columns, one after the other.
    segments: Vec<T>,
    /// The product of the block below the diagonal with them.
    product: Vec<T>,
    /// The places among the panel's rows of the rows below the diagonal.
    places: Vec<usize>,
    /// For each column of the panel, the steps and the elements of its column of `U` found so
    /// far: a segment is final once its update is done.
    u_steps: Vec<Vec<usize>>,
    u_values: Vec<Vec<T>>,
}

/// Applies the update of `source` to the columns `columns` of `dense` — the panel's columns by
/// its rows, `ld` rows to a column — each given with the first step of its segment, which
/// begins at or after `source.low`: a dense triangular solve of the segments, then the product
/// of the block below the diagonal with them, subtracted at the rows below. The segments, final
/// now, are added to the columns of `U`.
fn update<T: Scalar>(
    source: &Source<'_, T>,
    columns: &[(usize, usize)],
    local: &[usize],
    dense: &mut [T],
    ld: usize,
    scratch: &mut Scratch<T>,
) {
    let Source {
        block,
        height,
        rows,
        first,
        low,
        last,
    } = *source;
    let m = last + 1 - low;
    let offset = low - first;
    let below = last + 1 - first;
    let under = height - below;
    let count = columns.len();
    let Scratch {
        segments,
        product,
        places,
        u_steps,
        u_values,
    } = scratch;
    places.clear();
    places.extend(rows[below..].iter().map(|&row| local[row]));
    if m == 1 {
        // One column of the block: an element of `U` times its column of `L`.
        let column = &block[offset * height + below..offset * height + height];
        let pivotal = local[rows[offset]];
        for &(c, _) in columns {
            let x = &mut dense[c * ld..(c + 1) * ld];
            let u = x[pivotal];
            u_steps[c].push(low);
            u_values[c].push(u);
            if u.is_zero() {
                continue;
            }
            for (&place, &l) in places.iter().zip(column) {
                x[place] = x[place].mul_sub(l, u);
            }
        }
        return;
    }
    // The segments, padded with zeros before where each begins.
    segments.clear();
    segments.resize(m * count, T::zero());
    for (q, &(c, start)) in columns.iter().enumerate() {
        let x = &dense[c * ld..(c + 1) * ld];
        for i in start - low..m {
            segments[q * m + i] = x[local[rows[offset + i]]];
        }
    }
    // The solve with the unit lower triangle of the diagonal block.
    let triangle = &block[offset * height + offset..];
    if m <= SHORT {
        for q in 0..count {
            let segment = &mut segments[q * m..(q + 1) * m];
            for i in columns[q].1 - low..m {
                let u = segment[i];
                if u.is_zero() {
                    continue;
                }
                let column = &triangle[i * height..];
                for r in i + 1..m {
                    segment[r] = segment[r].mul_sub(column[r], u);
                }
            }
        }
    } else {
        trsm(
            Side::Left,
            Uplo::Lower,
            Op::N,
            true,
            m,
            count,
            triangle,
            height,
            segments,
            m,
        );
    }
    for (q, &(c, start)) in columns.iter().enumerate() {
        let x = &mut dense[c * ld..(c + 1) * ld];
        for i in start - low..m {
            let u = segments[q * m + i];
            x[local[rows[offset + i]]] = u;
            u_steps[c].push(low + i);
            u_values[c].push(u);
        }
    }
    if under == 0 {
        return;
    }
    let lower = &block[offset * height + below..];
    if under * m * count >= BLOCKED {
        product.clear();
        product.resize(under * count, T::zero());
        gemm(
            under,
            count,
            m,
            T::one(),
            lower,
            height,
            Op::N,
            segments,
            m,
            Op::N,
            T::zero(),
            product,
            under,
        );
        for (q, &(c, _)) in columns.iter().enumerate() {
            let x = &mut dense[c * ld..(c + 1) * ld];
            for (&place, &p) in places.iter().zip(&product[q * under..(q + 1) * under]) {
                x[place] -= p;
            }
        }
    } else {
        for (q, &(c, start)) in columns.iter().enumerate() {
            let x = &mut dense[c * ld..(c + 1) * ld];
            for i in start - low..m {
                let u = segments[q * m + i];
                if u.is_zero() {
                    continue;
                }
                let column = &lower[i * height..i * height + under];
                for (&place, &l) in places.iter().zip(column) {
                    x[place] = x[place].mul_sub(l, u);
                }
            }
        }
    }
}

/// The columns of the matrix being eliminated, in the order of the elimination.
pub(super) struct Columns<'c, T> {
    /// The order: step `j` eliminates column `col_perm[j]` of `A`.
    pub(super) col_perm: &'c [usize],
    /// Column `c` of `A` is in `starts[c]..starts[c + 1]` of `rows` and `values`.
    pub(super) starts: &'c [usize],
    /// The rows of the elements of each column, ascending.
    pub(super) rows: &'c [usize],
    /// The elements.
    pub(super) values: &'c [T],
}

/// The elimination of the matrix whose columns are `columns`, its pivots chosen by `pivoting`,
/// from step `done` on: `before` has made the steps before it a column at a time, and each of
/// its columns becomes a supernode of one column, its search pruned as it was. With `watched` —
/// the watch, whether a pivot of the steps before was off the diagonal, and the largest
/// absolute value of an element of `A` — the watch is asked after every step, and `None` is
/// returned as soon as it says to stop.
pub(super) fn eliminate<T: Scalar>(
    columns: &Columns<'_, T>,
    pivoting: &Pivoting<'_, T::Real>,
    before: Elimination<T>,
    done: usize,
    watched: Option<(&mut Watch<'_>, bool, T::Real)>,
) -> Option<Eliminated<T>> {
    let &Columns {
        col_perm,
        starts,
        rows: a_rows,
        values: a_values,
    } = columns;
    let (mut watch, mut off_diagonal, largest_a) = match watched {
        Some((watch, off_diagonal, largest_a)) => (Some(watch), off_diagonal, largest_a),
        None => (None, false, T::Real::ZERO),
    };
    let n = col_perm.len();
    let zero = T::Real::ZERO;
    let Elimination {
        l_ptrs: old_ptrs,
        l_rows: old_rows,
        l_values: old_values,
        u_ptrs,
        u_rows,
        u_values,
        pivots,
        step_of_row,
        row_perm,
        followed: old_followed,
        pruned: old_pruned,
        free,
        deficient,
        finite,
        largest_u,
        ..
    } = before;
    let (mut u_ptrs, mut u_rows, mut u_values, mut pivots) = (u_ptrs, u_rows, u_values, pivots);
    let (mut step_of_row, mut row_perm, mut free) = (step_of_row, row_perm, free);
    let (mut deficient, mut finite, mut largest_u) = (deficient, finite, largest_u);
    let mut supernode_of: Vec<usize> = Vec::with_capacity(n);
    let mut sn = Supernodes {
        first: Vec::with_capacity(done),
        row_ptrs: Vec::with_capacity(done + 1),
        rows: Vec::with_capacity(a_rows.len() + old_rows.len() + done),
        value_ptrs: Vec::with_capacity(done),
        follow_ptrs: Vec::with_capacity(done),
        follow_ends: Vec::with_capacity(done),
        followed: Vec::new(),
        pruned: old_pruned,
        open: false,
    };
    let mut blocks: Vec<T> = Vec::with_capacity(a_rows.len() + old_rows.len() + done);
    // The columns made so far, each a closed supernode: its pivotal row and its rows of `L`,
    // which the search followed as far as `old_followed` says.
    sn.row_ptrs.push(0);
    for k in 0..done {
        let (begin, end) = (old_ptrs[k], old_ptrs[k + 1]);
        sn.first.push(k);
        sn.value_ptrs.push(blocks.len());
        sn.rows.push(row_perm[k]);
        sn.rows.extend_from_slice(&old_rows[begin..end]);
        sn.row_ptrs.push(sn.rows.len());
        blocks.push(pivots[k]);
        blocks.extend_from_slice(&old_values[begin..end]);
        sn.follow_ptrs.push(begin);
        sn.follow_ends.push(old_followed[k]);
        supernode_of.push(k);
    }
    // The elements of `L` made so far, for the watch.
    let mut l_made = old_rows.len();
    sn.followed = old_rows;
    drop(old_values);
    u_ptrs.reserve(n - done);
    pivots.reserve(n - done);
    row_perm.reserve(n - done);
    let mut search = Search {
        mark: vec![NONE; n],
        local: vec![NONE; n],
        reached: Vec::new(),
        members: Vec::new(),
        slot: vec![NONE; n],
        slots: Vec::new(),
        starts: Vec::new(),
        low: Vec::new(),
        reaching: Vec::new(),
        places_of: (0..PANEL).map(|_| Vec::new()).collect(),
        last: Vec::new(),
        finished: Vec::new(),
        order: Vec::new(),
        stack: Vec::new(),
        structs: (0..PANEL).map(|_| Vec::new()).collect(),
        found: [0; PANEL],
        inner_mark: vec![NONE; n],
        inner_start: vec![0; n],
        inner_order: Vec::new(),
    };
    let mut scratch = Scratch {
        segments: Vec::new(),
        product: Vec::new(),
        places: Vec::new(),
        u_steps: (0..PANEL).map(|_| Vec::new()).collect(),
        u_values: (0..PANEL).map(|_| Vec::new()).collect(),
    };
    let mut dense: Vec<T> = Vec::new();
    let mut columns: Vec<(usize, usize)> = Vec::with_capacity(PANEL);
    // For the pruning: the pruning that last met each supernode, and where its row was kept.
    let mut prunings = 0usize;
    let mut seen = vec![0usize; n];
    let mut kept_at = vec![0usize; n];
    let mut p0 = done;
    while p0 < n {
        let width = PANEL.min(n - p0);
        // The first searches, in the supernodes before the panel.
        {
            let g = Graph {
                supernodes: &sn,
                step_of_row: &step_of_row,
                supernode_of: &supernode_of,
            };
            for c in 0..width {
                let column = col_perm[p0 + c];
                search.outer(c, p0, &a_rows[starts[column]..starts[column + 1]], &g);
                search.found[c] = search.structs[c].len();
            }
            // The pivotal rows of the segments are rows of the panel too.
            for t in 0..search.slots.len() {
                for &row in &row_perm[search.low[t]..=search.last[t]] {
                    search.place(row);
                }
            }
        }
        // The panel's columns of `A`, then their updates by the supernodes before it, in an order
        // that has each after every supernode it depends on.
        let ld = search.reached.len();
        dense.clear();
        dense.resize(ld * width, T::zero());
        for c in 0..width {
            let column = col_perm[p0 + c];
            for at in starts[column]..starts[column + 1] {
                dense[c * ld + search.local[a_rows[at]]] = a_values[at];
            }
        }
        for &s in search.order.iter().rev() {
            let t = search.slot[s];
            columns.clear();
            let mut bits = search.reaching[t];
            while bits != 0 {
                let c = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                columns.push((c, search.starts[t * PANEL + c]));
            }
            let source = Source {
                block: &blocks[sn.value_ptrs[s]..],
                height: sn.row_ptrs[s + 1] - sn.row_ptrs[s],
                rows: sn.rows_of(s),
                first: sn.first[s],
                low: search.low[t],
                last: search.last[t],
            };
            update(
                &source,
                &columns,
                &search.local,
                &mut dense,
                ld,
                &mut scratch,
            );
        }
        // The columns of the panel one after the other.
        for c in 0..width {
            let j = p0 + c;
            {
                let g = Graph {
                    supernodes: &sn,
                    step_of_row: &step_of_row,
                    supernode_of: &supernode_of,
                };
                search.inner(c, j, &g);
            }
            for &s in search.inner_order.iter().rev() {
                let source = Source {
                    block: &blocks[sn.value_ptrs[s]..],
                    height: sn.row_ptrs[s + 1] - sn.row_ptrs[s],
                    rows: sn.rows_of(s),
                    first: sn.first[s],
                    low: search.inner_start[s],
                    last: sn.first[s] + sn.width(s, j) - 1,
                };
                let start = search.inner_start[s];
                update(
                    &source,
                    &[(c, start)],
                    &search.local,
                    &mut dense,
                    ld,
                    &mut scratch,
                );
            }
            let x = &dense[c * ld..(c + 1) * ld];
            let local = &search.local;
            let value = |row: usize| match local[row] {
                NONE => T::zero(),
                place => x[place],
            };
            // The pivot, as the column method chooses it.
            let diagonal = col_perm[j];
            let mut largest = -T::Real::ONE;
            let mut lowest = NONE;
            let mut free_rows = 0;
            for &row in &search.structs[c] {
                if step_of_row[row] != NONE {
                    continue;
                }
                free_rows += 1;
                let size = value(row).abs1();
                if size > largest || (size == largest && row < lowest) {
                    largest = size;
                    lowest = row;
                }
            }
            let diagonal_is_free = step_of_row[diagonal] == NONE;
            let pivot_row = if largest > zero {
                let size = value(diagonal).abs1();
                if diagonal_is_free && size > zero && size >= pivoting.diagonal * largest {
                    diagonal
                } else {
                    let free = search.structs[c]
                        .iter()
                        .filter(|&&row| step_of_row[row] == NONE)
                        .map(|&row| (row, value(row).abs1()));
                    pivoting.off_diagonal(j, largest, lowest, free)
                }
            } else {
                deficient = true;
                if diagonal_is_free {
                    diagonal
                } else if lowest != NONE {
                    lowest
                } else {
                    while step_of_row[free] != NONE {
                        free += 1;
                    }
                    free
                }
            };
            let pivot = value(pivot_row);
            let reciprocal = T::one() / pivot;
            let multiply = reciprocal.is_finite();
            let element = |v: T| {
                if v.is_zero() {
                    v
                } else if multiply {
                    v * reciprocal
                } else {
                    v / pivot
                }
            };
            let mut column_finite = pivot.is_finite();
            let mut column_largest = largest_u.max(pivot.abs());
            // Column `j` of `U`: the segments of the supernodes it depends on, which their
            // updates left.
            for &v in &scratch.u_values[c] {
                column_finite &= v.is_finite();
                column_largest = column_largest.max(v.abs());
            }
            u_rows.append(&mut scratch.u_steps[c]);
            u_values.append(&mut scratch.u_values[c]);
            u_ptrs.push(u_rows.len());
            // Column `j` of `L`: into the supernode of the column before, or a new one.
            let in_rows = search.reaches(pivot_row, c) && step_of_row[pivot_row] == NONE;
            let previous = sn.count().wrapping_sub(1);
            let joins = j > 0 && sn.open && in_rows && {
                let s = previous;
                let w = sn.width(s, j);
                let t = search.slot[s];
                let depends = search.inner_mark[s] == j
                    || (t != NONE && search.starts[t * PANEL + c] != NONE);
                depends && w < WIDEST && free_rows == sn.rows_of(s).len() - w
            };
            if joins {
                let s = previous;
                let w = sn.width(s, j);
                let base = sn.row_ptrs[s];
                let height = sn.row_ptrs[s + 1] - base;
                let at = sn.value_ptrs[s];
                debug_assert!(
                    sn.rows[base + w..base + height]
                        .iter()
                        .all(|&r| search.reaches(r, c) && step_of_row[r] == NONE),
                    "step {j}: a row of the supernode is not in the column"
                );
                let pos = (w..height)
                    .find(|&p| sn.rows[base + p] == pivot_row)
                    .unwrap_or(w);
                if pos != w {
                    sn.rows.swap(base + w, base + pos);
                    for q in 0..w {
                        blocks.swap(at + q * height + w, at + q * height + pos);
                    }
                }
                debug_assert_eq!(blocks.len(), at + w * height);
                blocks.extend(core::iter::repeat_n(T::zero(), w));
                blocks.push(pivot);
                for p in w + 1..height {
                    let e = element(value(sn.rows[base + p]));
                    column_finite &= e.is_finite();
                    blocks.push(e);
                }
                l_made += height - w - 1;
            } else {
                if j > 0 && sn.open {
                    sn.close(previous, j);
                }
                sn.first.push(j);
                sn.rows.push(pivot_row);
                sn.value_ptrs.push(blocks.len());
                blocks.push(pivot);
                for &row in &search.structs[c] {
                    if step_of_row[row] == NONE && row != pivot_row {
                        sn.rows.push(row);
                        let e = element(value(row));
                        column_finite &= e.is_finite();
                        blocks.push(e);
                    }
                }
                l_made += sn.rows.len() - sn.row_ptrs[sn.count() - 1] - 1;
                sn.row_ptrs.push(sn.rows.len());
                sn.follow_ptrs.push(0);
                sn.follow_ends.push(0);
                sn.pruned.push(false);
                sn.open = true;
            }
            finite &= column_finite;
            largest_u = column_largest;
            pivots.push(pivot);
            step_of_row[pivot_row] = j;
            row_perm.push(pivot_row);
            supernode_of.push(sn.count() - 1);
            if let Some(watch) = watch.as_deref_mut() {
                off_diagonal |= pivot_row != diagonal;
                let made = l_made + u_rows.len() + j + 1;
                if watch.exceeded(col_perm, j, off_diagonal, made, largest_u, largest_a) {
                    return None;
                }
            }
            // The pruning: a closed supernode this column depends on, with an element in its
            // pivotal row, need be followed only to the rows that are pivotal now.
            let mut prune = |s: usize| {
                if sn.is_open(s) || sn.pruned[s] {
                    return;
                }
                let (begin, end) = (sn.follow_ptrs[s], sn.follow_ends[s]);
                if !sn.followed[begin..end].contains(&pivot_row) {
                    return;
                }
                // Of the rows pivotal by now, one for each supernode they are in is enough: the
                // pivotal row of its earliest step, which takes a search to the same segment.
                prunings += 1;
                let mut kept = begin;
                for at in begin..end {
                    let row = sn.followed[at];
                    let k = step_of_row[row];
                    if k == NONE {
                        continue;
                    }
                    let s2 = supernode_of[k];
                    if seen[s2] == prunings {
                        let p = kept_at[s2];
                        if k < step_of_row[sn.followed[p]] {
                            sn.followed[p] = row;
                        }
                    } else {
                        seen[s2] = prunings;
                        kept_at[s2] = kept;
                        sn.followed[kept] = row;
                        kept += 1;
                    }
                }
                sn.follow_ends[s] = kept;
                sn.pruned[s] = true;
            };
            for &t in &search.places_of[c] {
                prune(search.slots[t]);
            }
            for &s in &search.inner_order {
                prune(s);
            }
        }
        search.reset();
        p0 += width;
    }
    // The columns of `L` out of the blocks.
    let mut l_ptrs = Vec::with_capacity(n + 1);
    l_ptrs.push(0);
    let total: usize = (0..sn.count())
        .map(|s| {
            let (w, h) = (sn.width(s, n), sn.rows_of(s).len());
            w * h - w * (w + 1) / 2
        })
        .sum();
    let mut l_rows = Vec::with_capacity(total);
    let mut l_values = Vec::with_capacity(total);
    for s in 0..sn.count() {
        let rows = sn.rows_of(s);
        let height = rows.len();
        let at = sn.value_ptrs[s];
        for q in 0..sn.width(s, n) {
            l_rows.extend_from_slice(&rows[q + 1..]);
            l_values.extend_from_slice(&blocks[at + q * height + q + 1..at + (q + 1) * height]);
            l_ptrs.push(l_rows.len());
        }
    }
    debug_assert_eq!(l_made, total);
    Some(Eliminated {
        l_ptrs,
        l_rows,
        l_values,
        u_ptrs,
        u_rows,
        u_values,
        pivots,
        step_of_row,
        row_perm,
        deficient,
        finite,
        largest_u,
    })
}
