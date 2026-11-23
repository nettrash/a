//! The pattern of a matrix as an undirected graph, held the way the matrix is: one array of
//! neighbours, and for every vertex the offset in it where that vertex's stretch begins.
//!
//! The orderings never read an element. They read which cells store one, and they read it
//! symmetrically — an edge joins `i` and `j` when `(i, j)` or `(j, i)` is stored — so the graph is
//! built once here, in time proportional to the stored elements, and every ordering works on the
//! same form of it.

use super::SparseMatrix;
use crate::math::matrix::Scalar;

/// The index that stands for "none" in the arrays of the orderings.
pub(super) const NONE: usize = usize::MAX;

/// An undirected graph without loops or repeated edges: the neighbours of every vertex, ascending.
pub(super) struct Graph {
    /// Where each vertex's neighbours begin in `neighbours`, of length one more than the vertices.
    offsets: Vec<usize>,
    /// The neighbours, vertex after vertex.
    neighbours: Vec<usize>,
}

impl Graph {
    /// The graph of the pattern of `A + Aᵀ` without its diagonal, for a square `a`.
    ///
    /// The columns of `a` are gathered by a counting pass, and each vertex's list is the merge of
    /// its row and its column, so the cost is the stored elements and no sort is needed.
    pub(super) fn of_symmetric_pattern<T: Scalar>(a: &SparseMatrix<T>) -> Self {
        debug_assert_eq!(a.rows(), a.cols());
        let n = a.rows();
        let (row_ptrs, cols) = (a.row_pointers(), a.col_indices());
        let (col_ptrs, rows) = transposed_pattern(a);
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbours = Vec::with_capacity(a.nnz());
        for v in 0..n {
            offsets.push(neighbours.len());
            let (mut p, row_end) = (row_ptrs[v], row_ptrs[v + 1]);
            let (mut q, col_end) = (col_ptrs[v], col_ptrs[v + 1]);
            loop {
                let across = if p < row_end { cols[p] } else { NONE };
                let down = if q < col_end { rows[q] } else { NONE };
                let w = across.min(down);
                if w == NONE {
                    break;
                }
                p += usize::from(across == w);
                q += usize::from(down == w);
                if w != v {
                    neighbours.push(w);
                }
            }
        }
        offsets.push(neighbours.len());
        Self {
            offsets,
            neighbours,
        }
    }

    /// The graph of the pattern of `AᵀA` without its diagonal: its vertices are the columns of
    /// `a`, and two are joined when some row stores an element in both. Nothing is multiplied, so
    /// nothing can cancel or underflow out of the pattern.
    ///
    /// The cost is the sum of the squares of the row lengths, in time, and the edges found, in
    /// memory: a row that is full makes the graph complete, which is the mathematics and not the
    /// method.
    pub(super) fn of_normal_equations<T: Scalar>(a: &SparseMatrix<T>) -> Self {
        let n = a.cols();
        let (row_ptrs, cols) = (a.row_pointers(), a.col_indices());
        let (col_ptrs, rows) = transposed_pattern(a);
        // The neighbours of each column in the order they are met, each once.
        let mut met = vec![NONE; n];
        let mut offsets = Vec::with_capacity(n + 1);
        let mut unsorted = Vec::new();
        for j in 0..n {
            offsets.push(unsorted.len());
            met[j] = j;
            for &i in &rows[col_ptrs[j]..col_ptrs[j + 1]] {
                for &c in &cols[row_ptrs[i]..row_ptrs[i + 1]] {
                    if met[c] != j {
                        met[c] = j;
                        unsorted.push(c);
                    }
                }
            }
        }
        offsets.push(unsorted.len());
        // The graph is symmetric, so handing every vertex to its neighbours in ascending order
        // writes every list in ascending order.
        let mut neighbours = vec![0usize; unsorted.len()];
        let mut cursor = offsets.clone();
        for j in 0..n {
            for &c in &unsorted[offsets[j]..offsets[j + 1]] {
                neighbours[cursor[c]] = j;
                cursor[c] += 1;
            }
        }
        Self {
            offsets,
            neighbours,
        }
    }

    /// The graph of the given lists: `offsets` of length one more than the vertices, and the
    /// neighbours of each vertex ascending, without the vertex itself and symmetric.
    pub(super) fn from_lists(offsets: Vec<usize>, neighbours: Vec<usize>) -> Self {
        debug_assert_eq!(offsets.last().copied(), Some(neighbours.len()));
        Self {
            offsets,
            neighbours,
        }
    }

    /// The number of vertices.
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The neighbours of `v`, ascending.
    #[inline]
    pub(super) fn neighbours(&self, v: usize) -> &[usize] {
        &self.neighbours[self.offsets[v]..self.offsets[v + 1]]
    }

    /// The number of neighbours of `v`.
    #[inline]
    pub(super) fn degree(&self, v: usize) -> usize {
        self.offsets[v + 1] - self.offsets[v]
    }

    /// Twice the number of edges: the length of all the lists together.
    #[inline]
    pub(super) fn entries(&self) -> usize {
        self.neighbours.len()
    }

    /// The connected component of every vertex, numbered from 0 in the order the components are
    /// first met, and how many there are.
    pub(super) fn components(&self) -> (Vec<usize>, usize) {
        let n = self.len();
        let mut labels = vec![NONE; n];
        let mut count = 0;
        let mut stack = Vec::new();
        for seed in 0..n {
            if labels[seed] != NONE {
                continue;
            }
            labels[seed] = count;
            stack.push(seed);
            while let Some(v) = stack.pop() {
                for &w in self.neighbours(v) {
                    if labels[w] == NONE {
                        labels[w] = count;
                        stack.push(w);
                    }
                }
            }
            count += 1;
        }
        (labels, count)
    }
}

/// How many level structures the search for a far vertex may build. The search of George and Liu
/// ends by itself, as a rule after two to four of them; the cap keeps its cost linear on a graph
/// built to make it crawl.
const MAX_SWEEPS: usize = 12;

/// The levels of a breadth-first search of a part of a graph: the vertices level after level, and
/// the marks that let one search follow another without clearing anything, so that a search costs
/// what it reaches and no more.
pub(super) struct Levels {
    /// The search that reached each vertex last.
    reached: Vec<usize>,
    /// The number of searches made.
    search: usize,
    /// The vertices of the last search, level after level.
    queue: Vec<usize>,
    /// Where each level begins in the queue, and where the last one ends.
    starts: Vec<usize>,
}

impl Levels {
    /// The working memory for searches of a graph of `n` vertices.
    pub(super) fn new(n: usize) -> Self {
        Self {
            reached: vec![0; n],
            search: 0,
            queue: Vec::new(),
            starts: Vec::new(),
        }
    }

    /// Builds the levels of a breadth-first search from `root` over the vertices that `inside`
    /// admits: the rooted level structure of `root`.
    pub(super) fn build(&mut self, graph: &Graph, root: usize, inside: &impl Fn(usize) -> bool) {
        self.search += 1;
        let search = self.search;
        self.queue.clear();
        self.starts.clear();
        self.queue.push(root);
        self.reached[root] = search;
        let mut begin = 0;
        loop {
            self.starts.push(begin);
            let end = self.queue.len();
            for k in begin..end {
                for &w in graph.neighbours(self.queue[k]) {
                    if self.reached[w] != search && inside(w) {
                        self.reached[w] = search;
                        self.queue.push(w);
                    }
                }
            }
            if self.queue.len() == end {
                self.starts.push(end);
                return;
            }
            begin = end;
        }
    }

    /// Finds a pseudo-peripheral vertex of the part of `seed` by the iteration of George and Liu,
    /// and leaves its level structure built: from any vertex, a vertex of least degree in the
    /// last level becomes the new root as long as its own structure is deeper.
    pub(super) fn far(
        &mut self,
        graph: &Graph,
        seed: usize,
        inside: &impl Fn(usize) -> bool,
    ) -> usize {
        let mut root = seed;
        self.build(graph, root, inside);
        for _ in 1..MAX_SWEEPS {
            let depth = self.depth();
            let candidate = self
                .level(depth)
                .iter()
                .copied()
                .min_by_key(|&v| (graph.degree(v), v))
                .unwrap_or(root);
            if candidate == root {
                break;
            }
            // A vertex of the last level is as far from the root as anything is, so its own
            // structure is at least as deep. It is the better root when it is deeper, and as good
            // a one when it is not, which ends the search.
            self.build(graph, candidate, inside);
            root = candidate;
            if self.depth() <= depth {
                break;
            }
        }
        root
    }

    /// The number of levels after the first: the eccentricity of the root.
    #[inline]
    pub(super) fn depth(&self) -> usize {
        self.starts.len() - 2
    }

    /// The vertices of level `j`.
    #[inline]
    pub(super) fn level(&self, j: usize) -> &[usize] {
        &self.queue[self.starts[j]..self.starts[j + 1]]
    }

    /// How many vertices the levels before `j` hold.
    #[inline]
    pub(super) fn before(&self, j: usize) -> usize {
        self.starts[j]
    }

    /// The vertices reached, level after level.
    #[inline]
    pub(super) fn vertices(&self) -> &[usize] {
        &self.queue
    }
}

/// The pattern of the transpose of `a`: see [`transposed_lists`].
pub(super) fn transposed_pattern<T: Scalar>(a: &SparseMatrix<T>) -> (Vec<usize>, Vec<usize>) {
    transposed_lists(a.cols(), a.row_pointers(), a.col_indices())
}

/// The pattern of the transpose of a pattern held by rows: for every one of the `cols` columns the
/// rows that hold it, ascending, as the offsets of the columns and the rows after one another.
pub(super) fn transposed_lists(
    cols: usize,
    row_ptrs: &[usize],
    col_indices: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut col_ptrs = vec![0usize; cols + 1];
    for &j in col_indices {
        col_ptrs[j + 1] += 1;
    }
    for j in 0..cols {
        col_ptrs[j + 1] += col_ptrs[j];
    }
    let mut cursor = col_ptrs.clone();
    let mut rows = vec![0usize; col_indices.len()];
    for i in 0..row_ptrs.len() - 1 {
        for &j in &col_indices[row_ptrs[i]..row_ptrs[i + 1]] {
            rows[cursor[j]] = i;
            cursor[j] += 1;
        }
    }
    (col_ptrs, rows)
}

/// The largest number of neighbours a vertex of a graph of `n` vertices may have before the
/// minimum degree orderings set it aside as dense: `10·√n`, and never less than 16. These are the
/// defaults of the AMD and COLAMD papers.
pub(super) fn dense_threshold(n: usize) -> usize {
    // The square root of a `usize` is exact enough in `f64`: the threshold is a heuristic.
    let root = (n as f64).sqrt();
    ((10.0 * root) as usize).max(16)
}
