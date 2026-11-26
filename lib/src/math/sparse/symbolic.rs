//! The symbolic analysis of a sparse Cholesky factorisation: all that the pattern of the matrix
//! decides about its factor, found before any arithmetic is done.
//!
//! The factor `L` of `P·A·Pᵀ = L·Lᴴ` has a nonzero wherever the pattern of `A` lets one arise,
//! whatever the elements are, so its shape can be laid out first and the arithmetic then run on
//! storage that never moves. [`Symbolic::analyse`] does that for [`SparseCholesky`] and
//! [`SparseLdl`] alike, in steps that together cost about the nonzeros of `A` (path compression
//! keeps the climbs of the tree nearly linear) and the row indices it writes, never `n²`:
//!
//! 1. The ordering gives the permutation `P`, and the pattern of the lower triangle of `P·A·Pᵀ`
//!    is laid out by columns and by rows.
//! 2. The elimination tree, by Liu's algorithm with path compression: the parent of column `j` is
//!    the row of the first nonzero below the diagonal of column `j` of `L`, and a column of `L`
//!    depends on the columns of its subtree alone. (J. W. H. Liu, "A compact row storage scheme for
//!    Cholesky factors using elimination trees", ACM TOMS 12, 1986; "The role of elimination trees
//!    in sparse factorization", SIAM J. Matrix Anal. Appl. 11, 1990.)
//! 3. The number of nonzeros of every column of `L`, by the algorithm of Gilbert, Ng and Peyton,
//!    which never forms a column to count it. ("An efficient algorithm to compute row and column
//!    counts for sparse Cholesky factorization", SIAM J. Matrix Anal. Appl. 15, 1994.)
//! 4. A postorder of the tree, in which the columns are renumbered: a subtree becomes a run of
//!    consecutive columns, which is what lets columns that belong together lie together. A
//!    postorder is an equivalent reordering — the same fill, the same operations — so it stays
//!    inside: [`Symbolic::permutation`] is the ordering as it was given, and the factor is handed
//!    out in that numbering.
//! 5. The supernodes: runs of consecutive columns with the same pattern below a dense triangle,
//!    which the numeric factorisation treats as one dense block. First the fundamental supernodes
//!    of Liu, Ng and Peyton ("On finding supernodes for sparse matrix computations", SIAM J.
//!    Matrix Anal. Appl. 14, 1993), then the relaxed amalgamation of Ashcraft and Grimes ("The
//!    influence of relaxed supernode partitions on the multifrontal method", ACM TOMS 15, 1989):
//!    see [`amalgamate`] for the rule.
//! 6. The rows below each supernode, written in ascending order by walking the row subtrees over
//!    the tree of supernodes, each row index written once.
//!
//! [`SparseCholesky`]: super::SparseCholesky
//! [`SparseLdl`]: super::SparseLdl

use super::{Ordering, SparseMatrix};
use crate::math::matrix::Scalar;

/// No vertex: the parent of a root, the end of a list, a mark not yet made.
pub(super) const NONE: usize = usize::MAX;

/// What the pattern of a Hermitian matrix and an ordering decide about the factor: the
/// permutation, the supernodes and the rows of each, and where each block of the factor is held.
///
/// Rows and columns have three numberings here. The *original* one is that of `A`. The *permuted*
/// one is that of `P·A·Pᵀ` for the fill-reducing `P`, in which the factor is handed out. The
/// *inner* one is the permuted one taken in a postorder of the elimination tree, in which the
/// factor is computed and held: everything below is in the inner numbering unless it says
/// otherwise.
#[derive(Clone, Debug)]
pub(super) struct Symbolic {
    /// The order of the matrix.
    pub(super) n: usize,
    /// The fill-reducing permutation as the ordering gave it: row `i` of `P·A·Pᵀ` is row
    /// `permutation[i]` of `A`.
    pub(super) permutation: Vec<usize>,
    /// The postorder: inner column `k` is column `postorder[k]` of `P·A·Pᵀ`.
    pub(super) postorder: Vec<usize>,
    /// The two together: inner column `k` is column `original[k]` of `A`.
    pub(super) original: Vec<usize>,
    /// The first column of each supernode, and `n` after the last: supernode `s` is the columns
    /// `first[s]..first[s + 1]`.
    pub(super) first: Vec<usize>,
    /// The supernode of each column.
    pub(super) supernode: Vec<usize>,
    /// Where the rows below each supernode begin in `rows`, and their number after the last.
    pub(super) row_ptrs: Vec<usize>,
    /// The rows below the columns of each supernode in which it has a nonzero, ascending: with the
    /// supernode's own columns, the rows of its dense block.
    pub(super) rows: Vec<usize>,
    /// Where the square block on the diagonal of each supernode begins in the factor's storage of
    /// them, `width × width` elements each.
    pub(super) diagonal_ptrs: Vec<usize>,
    /// Where the block below it begins in the factor's storage of those, `height × width`
    /// elements each, a row of it for each of the supernode's `rows`.
    pub(super) below_ptrs: Vec<usize>,
    /// The number of nonzeros of the factor, its diagonal counted: the sum of the column counts,
    /// without the zeros the amalgamation of supernodes adds to what is held.
    pub(super) nnz: usize,
}

/// The lower triangle of the matrix in the inner numbering, column after column: what the
/// numeric factorisation reads its elements from.
pub(super) struct Lower<T> {
    /// Where each column begins in `rows` and `values`, and their number after the last.
    pub(super) col_ptrs: Vec<usize>,
    /// The row of each element, not above its column.
    pub(super) rows: Vec<usize>,
    /// The elements, those of the diagonal real.
    pub(super) values: Vec<T>,
}

impl Symbolic {
    /// The analysis of the Hermitian matrix whose lower triangle is that of `a`, under the
    /// ordering. What `a` holds above its diagonal is not looked at, not even by the ordering.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, if the ordering doesn't give a permutation of its rows, or if
    /// the factor would hold more elements than a vector can.
    #[track_caller]
    pub(super) fn analyse<T: Scalar>(a: &SparseMatrix<T>, ordering: &Ordering) -> Self {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        let n = a.rows();
        let permutation = fill_reducing(a, ordering);
        let pattern = Pattern::new(a, &inverse_of(&permutation, n));
        let parent = elimination_tree(&pattern);
        let counts = column_counts(&pattern, &parent);
        // The child whose column is longest is visited last, next to its parent: of the children
        // it is the one whose pattern is nearest the parent's, and so the one worth merging.
        let postorder = postorder(&parent, Some(&heaviest_children(&parent, &counts)));
        let mut place = vec![0usize; n];
        for (k, &j) in postorder.iter().enumerate() {
            place[j] = k;
        }
        // The tree and the counts in the inner numbering.
        let mut inner_parent = vec![NONE; n];
        let mut inner_counts = vec![0usize; n];
        for (k, &j) in postorder.iter().enumerate() {
            inner_counts[k] = counts[j];
            if parent[j] != NONE {
                inner_parent[k] = place[parent[j]];
            }
        }
        let first = supernodes(&inner_parent, &inner_counts);
        let count = first.len() - 1;
        let mut supernode = vec![0usize; n];
        for s in 0..count {
            supernode[first[s]..first[s + 1]].fill(s);
        }
        // The rows below a supernode are those of its last column, the root of the subtree its
        // columns make, so their number is known before they are.
        let too_large = "the factor has more elements than a vector can hold";
        let mut row_ptrs = vec![0usize; count + 1];
        let mut diagonal_ptrs = vec![0usize; count + 1];
        let mut below_ptrs = vec![0usize; count + 1];
        for s in 0..count {
            let width = first[s + 1] - first[s];
            let height = inner_counts[first[s + 1] - 1] - 1;
            row_ptrs[s + 1] = row_ptrs[s].checked_add(height).expect(too_large);
            diagonal_ptrs[s + 1] = width
                .checked_mul(width)
                .and_then(|block| diagonal_ptrs[s].checked_add(block))
                .expect(too_large);
            below_ptrs[s + 1] = height
                .checked_mul(width)
                .and_then(|block| below_ptrs[s].checked_add(block))
                .expect(too_large);
        }
        let rows = rows_below(
            &pattern,
            &postorder,
            &place,
            &inner_parent,
            &first,
            &supernode,
            &row_ptrs,
        );
        // Every nonzero of the factor is held, so their number fits where the blocks did.
        let nnz = inner_counts.iter().sum();
        let original = postorder.iter().map(|&j| permutation[j]).collect();
        Self {
            n,
            permutation,
            postorder,
            original,
            first,
            supernode,
            row_ptrs,
            rows,
            diagonal_ptrs,
            below_ptrs,
            nnz,
        }
    }

    /// The number of supernodes.
    pub(super) fn supernodes(&self) -> usize {
        self.first.len() - 1
    }

    /// The lower triangle of `a` in the inner numbering, with its elements: an element that the
    /// two permutations carry above the diagonal goes to the cell across it, conjugated, as the
    /// matrix is Hermitian. The imaginary part of a diagonal element is left out, which a
    /// Hermitian matrix has none of.
    pub(super) fn lower<T: Scalar>(&self, a: &SparseMatrix<T>) -> Lower<T> {
        let n = self.n;
        let mut place = vec![0usize; n];
        for (k, &old) in self.original.iter().enumerate() {
            place[old] = k;
        }
        let (ptrs, cols, elements) = (a.row_pointers(), a.col_indices(), a.values());
        // The columns of a row ascend, so its part in the lower triangle is at its beginning.
        let lower_end = |r: usize| {
            ptrs[r]
                + cols[ptrs[r]..ptrs[r + 1]]
                    .iter()
                    .take_while(|&&c| c <= r)
                    .count()
        };
        let mut col_ptrs = vec![0usize; n + 1];
        for r in 0..n {
            for &c in &cols[ptrs[r]..lower_end(r)] {
                col_ptrs[place[r].min(place[c]) + 1] += 1;
            }
        }
        for j in 0..n {
            col_ptrs[j + 1] += col_ptrs[j];
        }
        let mut rows = vec![0usize; col_ptrs[n]];
        let mut values = vec![T::zero(); col_ptrs[n]];
        let mut cursor = col_ptrs[..n].to_vec();
        for r in 0..n {
            for p in ptrs[r]..lower_end(r) {
                let (c, x) = (cols[p], elements[p]);
                let (i, j) = (place[r], place[c]);
                let (row, col, x) = if c == r {
                    (i, i, T::from_real(x.re()))
                } else if i > j {
                    (i, j, x)
                } else {
                    (j, i, x.conj())
                };
                rows[cursor[col]] = row;
                values[cursor[col]] = x;
                cursor[col] += 1;
            }
        }
        Lower {
            col_ptrs,
            rows,
            values,
        }
    }
}

/// The fill-reducing permutation the ordering gives the lower triangle of `a`, `p[new] = old`.
///
/// # Panics
///
/// Panics if a given permutation isn't one of the rows of `a`.
#[track_caller]
fn fill_reducing<T: Scalar>(a: &SparseMatrix<T>, ordering: &Ordering) -> Vec<usize> {
    match ordering {
        // Neither looks at the matrix.
        Ordering::Natural | Ordering::Given(_) => ordering.symmetric(a),
        // The others read the pattern of `A + Aᵀ`. Only the lower triangle of `a` is the matrix,
        // so only it may steer them: whatever stands above the diagonal is taken out first.
        _ => ordering.symmetric(&a.lower_triangle()),
    }
}

/// The inverse of the permutation `p` of `0..n`: `inverse[p[k]] = k`.
///
/// # Panics
///
/// Panics if `p` isn't a permutation of `0..n`, which an ordering must give.
#[track_caller]
fn inverse_of(p: &[usize], n: usize) -> Vec<usize> {
    assert_eq!(p.len(), n, "the ordering isn't of the right length");
    let mut inverse = vec![NONE; n];
    for (new, &old) in p.iter().enumerate() {
        assert!(old < n, "the ordering is out of range");
        assert!(inverse[old] == NONE, "the ordering repeats an index");
        inverse[old] = new;
    }
    inverse
}

/// The pattern of the strict lower triangle of `P·A·Pᵀ`, by columns and by rows.
struct Pattern {
    /// Where the rows of each column begin in `col_rows`, and their number after the last.
    col_ptrs: Vec<usize>,
    /// The rows `i > j` of the nonzeros of each column `j`, in no particular order.
    col_rows: Vec<usize>,
    /// Where the columns of each row begin in `row_cols`, and their number after the last.
    row_ptrs: Vec<usize>,
    /// The columns `k < i` of the nonzeros of each row `i`, ascending.
    row_cols: Vec<usize>,
}

impl Pattern {
    /// The pattern of `P·A·Pᵀ` below its diagonal, from the lower triangle of `a` and the
    /// inverse of the permutation.
    fn new<T: Scalar>(a: &SparseMatrix<T>, inverse: &[usize]) -> Self {
        let n = a.rows();
        let (ptrs, cols) = (a.row_pointers(), a.col_indices());
        // An element below the diagonal of `A`, at `(r, c)`, is one of `P·A·Pᵀ` at the new indices
        // of the two, of which the larger is its row in the lower triangle. The columns of a row
        // ascend, so those below the diagonal come first.
        let below = |r: usize| {
            cols[ptrs[r]..ptrs[r + 1]]
                .iter()
                .take_while(move |&&c| c < r)
                .map(move |&c| (inverse[r].max(inverse[c]), inverse[r].min(inverse[c])))
        };
        let mut col_ptrs = vec![0usize; n + 1];
        let mut row_ptrs = vec![0usize; n + 1];
        for r in 0..n {
            for (i, j) in below(r) {
                col_ptrs[j + 1] += 1;
                row_ptrs[i + 1] += 1;
            }
        }
        for j in 0..n {
            col_ptrs[j + 1] += col_ptrs[j];
            row_ptrs[j + 1] += row_ptrs[j];
        }
        let mut col_rows = vec![0usize; col_ptrs[n]];
        let mut cursor = col_ptrs[..n].to_vec();
        for r in 0..n {
            for (i, j) in below(r) {
                col_rows[cursor[j]] = i;
                cursor[j] += 1;
            }
        }
        // The same by rows: the columns are visited in order, so each row's come out ascending.
        let mut row_cols = vec![0usize; row_ptrs[n]];
        cursor.copy_from_slice(&row_ptrs[..n]);
        for j in 0..n {
            for &i in &col_rows[col_ptrs[j]..col_ptrs[j + 1]] {
                row_cols[cursor[i]] = j;
                cursor[i] += 1;
            }
        }
        Self {
            col_ptrs,
            col_rows,
            row_ptrs,
            row_cols,
        }
    }

    /// The order of the matrix.
    fn order(&self) -> usize {
        self.col_ptrs.len() - 1
    }

    /// The rows below the diagonal in which column `j` has a nonzero.
    fn column(&self, j: usize) -> &[usize] {
        &self.col_rows[self.col_ptrs[j]..self.col_ptrs[j + 1]]
    }

    /// The columns before the diagonal in which row `i` has a nonzero.
    fn row(&self, i: usize) -> &[usize] {
        &self.row_cols[self.row_ptrs[i]..self.row_ptrs[i + 1]]
    }
}

/// The elimination tree of the pattern: `parent[j]` is the row of the first nonzero below the
/// diagonal of column `j` of the factor, or [`NONE`] where the column has none and `j` is a root.
///
/// Liu's algorithm. Row `i` of the factor has its nonzeros on the paths of the tree from the
/// nonzeros of row `i` of the matrix up to `i`, so the tree of the first `i` rows becomes that of
/// the first `i + 1` by hanging under `i` the root of every tree a nonzero of row `i` lies in.
/// The climb to a root is cut short by `ancestor`, which points every vertex passed at the vertex
/// the climb was made for: path compression, which makes the whole nearly linear in the nonzeros.
fn elimination_tree(pattern: &Pattern) -> Vec<usize> {
    let n = pattern.order();
    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    for i in 0..n {
        for &k in pattern.row(i) {
            let mut k = k;
            loop {
                let above = ancestor[k];
                if above == i {
                    // An earlier nonzero of this row has been up this path already.
                    break;
                }
                ancestor[k] = i;
                if above == NONE {
                    parent[k] = i;
                    break;
                }
                k = above;
            }
        }
    }
    parent
}

/// A postorder of the forest `parent`: `order[k]` is the vertex visited `k`-th, each vertex after
/// all of its descendants and each subtree in one run. The roots, and the children of a vertex,
/// are visited in ascending order, but for the child `last[v]` of `v`, where there is one, which
/// is visited after the others.
fn postorder(parent: &[usize], last: Option<&[usize]>) -> Vec<usize> {
    let n = parent.len();
    // The children of each vertex as a list, those of slot `n` the roots: `head` is the first
    // and `next` the one after. The vertices are pushed at the front from the largest down, so
    // each list comes out ascending; a child that goes last is put in before any other.
    let mut head = vec![NONE; n + 1];
    let mut next = vec![NONE; n];
    let goes_last = |j: usize| last.is_some_and(|last| parent[j] != NONE && last[parent[j]] == j);
    if let Some(last) = last {
        head[..n].copy_from_slice(last);
    }
    for j in (0..n).rev() {
        if !goes_last(j) {
            let p = if parent[j] == NONE { n } else { parent[j] };
            next[j] = head[p];
            head[p] = j;
        }
    }
    // Depth first with a stack of its own: a tree may be a path of `n` vertices. `head` is used
    // up as it goes, pointing at the next child still to visit.
    let mut order = Vec::with_capacity(n);
    let mut stack = Vec::new();
    let mut root = head[n];
    while root != NONE {
        stack.push(root);
        while let Some(&top) = stack.last() {
            let child = head[top];
            if child == NONE {
                order.push(top);
                stack.pop();
            } else {
                head[top] = next[child];
                stack.push(child);
            }
        }
        root = next[root];
    }
    order
}

/// The number of nonzeros of each column of the factor, its diagonal counted, by the algorithm
/// of Gilbert, Ng and Peyton.
///
/// Column `j` has a nonzero in row `i` exactly when `j` lies in the *row subtree* of `i`: the
/// part of the elimination tree made of the paths from the nonzeros of row `i` of the matrix up
/// to `i`. So the count of `j` is the number of row subtrees it lies in, and that is found
/// without walking any of them, by giving the vertices weights whose sum over the subtree of `j`
/// is 1 when `j` lies in the row subtree and 0 when it doesn't: `+1` at each leaf of the row
/// subtree, `−1` at the least common ancestor of each two consecutive leaves, where their two
/// paths join and would count twice, and `−1` at the parent of `i`, above which the row subtree
/// has ended. A row with nothing before its diagonal is the leaf of its own subtree; it is such a
/// row exactly when it is a leaf of the elimination tree. The counts are then the sums of the
/// weights over the subtrees.
///
/// With the columns taken in postorder, `j` is a leaf of the row subtree of `i` when no column
/// of the subtree of `j` has had a nonzero in row `i`: when the latest column that had one came
/// before the first descendant of `j`. The least common ancestors come from a forest of disjoint
/// sets in which a finished vertex is joined to its parent, so that the set of an earlier leaf is
/// named by its lowest ancestor not yet finished, which is the one it shares with `j`.
fn column_counts(pattern: &Pattern, parent: &[usize]) -> Vec<usize> {
    let n = parent.len();
    let order = postorder(parent, None);
    let mut place = vec![0usize; n];
    for (k, &j) in order.iter().enumerate() {
        place[j] = k;
    }
    // The sizes of the subtrees: the subtree of `j` is visited from `place[j] + 1 − size[j]`.
    let mut size = vec![1usize; n];
    for &j in &order {
        if parent[j] != NONE {
            size[parent[j]] += size[j];
        }
    }
    let mut weight = vec![0isize; n];
    for j in 0..n {
        if size[j] == 1 {
            weight[j] += 1;
        }
        if parent[j] != NONE {
            weight[parent[j]] -= 1;
        }
    }
    // For each row: one more than the place of the latest column with a nonzero in it, 0 before
    // there is one; and the latest leaf found of its row subtree.
    let mut latest = vec![0usize; n];
    let mut leaf = vec![NONE; n];
    let mut set: Vec<usize> = (0..n).collect();
    for &j in &order {
        let first_descendant = place[j] + 1 - size[j];
        for &i in pattern.column(j) {
            if latest[i] <= first_descendant {
                weight[j] += 1;
                if leaf[i] != NONE {
                    weight[find(&mut set, leaf[i])] -= 1;
                }
                leaf[i] = j;
            }
            latest[i] = place[j] + 1;
        }
        if parent[j] != NONE {
            set[j] = parent[j];
        }
    }
    for &j in &order {
        if parent[j] != NONE {
            let below = weight[j];
            weight[parent[j]] += below;
        }
    }
    weight
        .into_iter()
        .map(|w| {
            debug_assert!(w >= 1, "a column of the factor has its diagonal at least");
            w as usize
        })
        .collect()
}

/// The vertex that names the set of `x`, with every vertex passed pointed nearer to it.
fn find(set: &mut [usize], mut x: usize) -> usize {
    while set[x] != x {
        set[x] = set[set[x]];
        x = set[x];
    }
    x
}

/// For each vertex, the child whose column of the factor is longest, the later of equals, or
/// [`NONE`] for a leaf.
fn heaviest_children(parent: &[usize], counts: &[usize]) -> Vec<usize> {
    let mut heaviest = vec![NONE; parent.len()];
    for (j, &p) in parent.iter().enumerate() {
        if p != NONE && (heaviest[p] == NONE || counts[j] >= counts[heaviest[p]]) {
            heaviest[p] = j;
        }
    }
    heaviest
}

/// The supernodes of a postordered elimination tree with the column counts `counts`: the first
/// column of each, and `n` after the last.
///
/// A *fundamental* supernode is a longest run of consecutive columns in which each column but the
/// first is the parent of the one before it, has no other child, and has the pattern of the one
/// before it less that column's diagonal — one nonzero fewer. These are then *amalgamated*, from
/// the leaves up: a supernode takes in the supernode just before it when that is one of its
/// children and [`amalgamate`] allows it, and goes on doing so with what then stands before it.
/// Only a last child can be taken in, as only it lies next to its parent; the postorder has put
/// the child with the longest column there.
fn supernodes(parent: &[usize], counts: &[usize]) -> Vec<usize> {
    let n = parent.len();
    let mut children = vec![0usize; n];
    for &p in parent {
        if p != NONE {
            children[p] += 1;
        }
    }
    // The nonzeros of the columns before each, to count those of a run of columns at once.
    let mut before = vec![0u64; n + 1];
    for j in 0..n {
        before[j + 1] = before[j] + counts[j] as u64;
    }
    let mut first: Vec<usize> = Vec::new();
    let mut start = 0;
    for j in 1..=n {
        let fundamental =
            j < n && parent[j - 1] == j && children[j] == 1 && counts[j - 1] == counts[j] + 1;
        if fundamental {
            continue;
        }
        // The fundamental supernode `start..j` is complete; the columns `begin..j` are what it
        // has become once the supernodes before it have been taken in.
        let mut begin = start;
        while let Some(&child) = first.last() {
            let is_child = parent[begin - 1] != NONE && parent[begin - 1] < j;
            if !(is_child && amalgamate(child, begin, j, counts, &before)) {
                break;
            }
            begin = child;
            first.pop();
        }
        first.push(begin);
        start = j;
    }
    first.push(n);
    first
}

/// Whether the supernode of the columns `child..parent` is merged into its parent, the supernode
/// of the columns `parent..end`.
///
/// A merged supernode is held as one dense block: its columns in full from the diagonal down to
/// the last of them, and below that the rows of the parent. A column of the child that has no
/// nonzero in one of those rows holds a zero there, which costs memory and arithmetic, while
/// every merge saves the bookkeeping of a supernode and makes the dense blocks larger, which is
/// where the kernels are fast. The rule weighs the two, by the number of columns of the merged
/// supernode and the fraction of zeros among the elements it would hold, the zeros of earlier
/// merges counted:
///
/// * a merge that adds no zero is always made, and so is one that gives at most 4 columns;
/// * up to 16 columns, with less than 80 % zeros;
/// * up to 48 columns, with less than 10 % zeros;
/// * any number of columns, with less than 5 % zeros.
///
/// The thresholds are the defaults of CHOLMOD (Chen, Davis, Hager and Rajamanickam, "Algorithm
/// 887: CHOLMOD, supernodal sparse Cholesky factorization and update/downdate", ACM TOMS 35,
/// 2008, and its user guide: `nrelax` and `zrelax`), taken as a starting point and not tuned
/// here.
fn amalgamate(child: usize, parent: usize, end: usize, counts: &[usize], before: &[u64]) -> bool {
    let (child_width, parent_width) = (parent - child, end - parent);
    // The rows below a supernode are those of its last column, which lie in the columns of the
    // parent or in the rows below the parent.
    let below = counts[end - 1] - 1;
    let child_below = counts[parent - 1] - 1;
    debug_assert!(child_below <= parent_width + below);
    let added = child_width as f64 * (parent_width + below - child_below) as f64;
    let width = child_width + parent_width;
    if added == 0.0 || width <= 4 {
        return true;
    }
    // In floats, where the number of elements of a block cannot overflow.
    let held = width as f64 * (width + 1) as f64 / 2.0 + width as f64 * below as f64;
    let nonzeros = (before[end] - before[child]) as f64;
    let zeros = (held - nonzeros) / held;
    (width <= 16 && zeros < 0.8) || (width <= 48 && zeros < 0.1) || zeros < 0.05
}

/// The rows below each supernode in which it has a nonzero, ascending, at the offsets `row_ptrs`.
///
/// Row `i` belongs to the supernodes its row subtree passes through below the one that holds
/// `i`: for each nonzero of row `i` of the matrix, the supernodes on the way up the tree of
/// supernodes from the one of its column. The climb stops at a supernode this row has marked
/// already, so each row index is written once, and the rows are taken in order, so each
/// supernode receives its own in ascending order with nothing to sort.
///
/// # Panics
///
/// Panics if a supernode receives another number of rows than its column count has given it
/// room for, which would mean that the analysis contradicts itself.
#[allow(clippy::too_many_arguments)]
fn rows_below(
    pattern: &Pattern,
    postorder: &[usize],
    place: &[usize],
    parent: &[usize],
    first: &[usize],
    supernode: &[usize],
    row_ptrs: &[usize],
) -> Vec<usize> {
    let count = first.len() - 1;
    let mut rows = vec![0usize; row_ptrs[count]];
    let mut cursor = row_ptrs[..count].to_vec();
    let mut marked = vec![NONE; count];
    for (i, &permuted) in postorder.iter().enumerate() {
        let home = supernode[i];
        for &k in pattern.row(permuted) {
            let mut s = supernode[place[k]];
            while s != home && marked[s] != i {
                marked[s] = i;
                assert!(
                    cursor[s] < row_ptrs[s + 1],
                    "the symbolic analysis is inconsistent"
                );
                rows[cursor[s]] = i;
                cursor[s] += 1;
                // The parent of a supernode is the supernode of the parent of its last column,
                // which exists: the row lies above.
                s = supernode[parent[first[s + 1] - 1]];
            }
        }
    }
    assert!(
        cursor[..] == row_ptrs[1..],
        "the symbolic analysis is inconsistent"
    );
    rows
}
