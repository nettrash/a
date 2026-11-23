//! The symbolic analysis of the sparse QR decomposition: everything the pattern of the matrix
//! decides, found before any arithmetic so that the numeric phase allocates once.
//!
//! For `B = A·Q_c` the analysis finds, in this order,
//!
//! * the column elimination tree, which is the elimination tree of `BᴴB`, from the pattern of `B`
//!   alone: Liu's algorithm with path compression, in which a row stands for the path through
//!   its columns instead of the clique it makes in `BᴴB` ([`column_tree`]);
//! * the fronts ([`chains`]): chains of the tree whose rows of `R` have the same pattern but for
//!   their leading columns (fundamental supernodes), each with the columns its rows reach;
//! * a postorder of the fronts, and for each front its rows — those of `B` that begin in its
//!   pivot columns and those its children pass on — sorted by their first columns: the staircase
//!   the reflections follow. From it come the length of every Householder vector, the pattern of
//!   every row of `R`, the place of every contribution block in a stack and the peak of that
//!   stack ([`analyse`]).
//!
//! The patterns are those the rows really make. Where the matrix lacks the strong Hall property
//! they are smaller than the pattern of the Cholesky factor of `BᴴB`, which is only a bound: a
//! front that has no more rows than pivot columns passes nothing on, whatever columns it reaches.

use crate::math::matrix::Scalar;
use crate::math::sparse::SparseMatrix;

/// The index that stands for none: no parent, no front, no column.
pub(super) const NONE: usize = usize::MAX;

/// What the pattern of `A·Q_c` decides, the fronts numbered in the order they are factored,
/// which is a postorder of their tree.
pub(super) struct Symbolic {
    /// Where each column of the matrix went, `position[old] = new`: the inverse of the column
    /// permutation.
    pub position: Vec<usize>,
    /// The number of pivot columns of each front.
    pub pivots: Vec<usize>,
    /// Where the columns of each front begin in `columns`, and where the last ones end.
    pub column_ptr: Vec<usize>,
    /// The columns of each front, ascending, its pivot columns first.
    pub columns: Vec<usize>,
    /// Where the children of each front begin in `children`, and where the last ones end.
    pub child_ptr: Vec<usize>,
    /// The children of each front, in the order they are factored.
    pub children: Vec<usize>,
    /// Where the rows of each front begin in `coordinates`, and where the last ones end.
    pub row_ptr: Vec<usize>,
    /// The rows of each front as rows of the matrix, in the order of its staircase: the ones
    /// that begin in an earlier column of the front first.
    pub coordinates: Vec<usize>,
    /// The first reflection of each front, and the number of all of them.
    pub reflection_ptr: Vec<usize>,
    /// Where the vector of each reflection begins, and where the last one ends.
    pub vector_ptr: Vec<usize>,
    /// The front that takes each row of the matrix in, [`NONE`] for a row that stores nothing.
    pub entering: Vec<usize>,
    /// Where the contribution block of each front lies in the stack.
    pub block_ptr: Vec<usize>,
    /// The peak height of the stack of contribution blocks.
    pub stack: usize,
    /// The elements of the largest front, each column counted down to its step of the staircase.
    pub work: usize,
    /// The most columns of a front.
    pub widest: usize,
    /// The most rows of a front.
    pub tallest: usize,
    /// Where each row of `R` begins in `r_cols`, and where the last one ends.
    pub r_ptr: Vec<usize>,
    /// The columns of each row of `R`, ascending, the diagonal first.
    pub r_cols: Vec<usize>,
    /// The row of the matrix whose place each row of `R` takes once the reflections are applied.
    pub pivot_row: Vec<usize>,
}

/// The fronts in the order their first columns come.
struct Chains {
    /// The number of pivot columns of each front.
    pivots: Vec<usize>,
    /// Where the columns of each front begin in `columns`, and where the last ones end.
    column_ptr: Vec<usize>,
    /// The columns of each front, ascending, its pivot columns first.
    columns: Vec<usize>,
    /// The number of rows of each front.
    rows: Vec<usize>,
    /// The front each front passes its rows to, [`NONE`] for a root.
    parent: Vec<usize>,
}

/// The analysis of `a` with its columns in the order `permutation`, `permutation[new] = old`.
///
/// The cost is of the order of the stored elements of `a`, of `R` and of the contribution
/// blocks, with a sort of each front's columns.
///
/// # Panics
///
/// Panics if `permutation` isn't a permutation of the columns, or if `a` has fewer rows than
/// columns.
#[track_caller]
pub(super) fn analyse<T: Scalar>(a: &SparseMatrix<T>, permutation: &[usize]) -> Symbolic {
    let (m, n) = (a.rows(), a.cols());
    assert!(m >= n, "the matrix has fewer rows than columns");
    let position = inverse(permutation, n);
    let (pointers, indices) = (a.row_pointers(), a.col_indices());

    // The first column of each row in the new order, and the rows gathered by it: a row is
    // assembled into the front of that column and into no other.
    let mut bucket_ptr = vec![0usize; n + 1];
    let mut leftmost = vec![NONE; m];
    for (i, first) in leftmost.iter_mut().enumerate() {
        let row = &indices[pointers[i]..pointers[i + 1]];
        if let Some(least) = row.iter().map(|&j| position[j]).min() {
            *first = least;
            bucket_ptr[least + 1] += 1;
        }
    }
    for k in 0..n {
        bucket_ptr[k + 1] += bucket_ptr[k];
    }
    let mut bucket = vec![0usize; bucket_ptr[n]];
    let mut cursor = bucket_ptr.clone();
    for (i, &least) in leftmost.iter().enumerate() {
        if least != NONE {
            bucket[cursor[least]] = i;
            cursor[least] += 1;
        }
    }

    let tree = column_tree(m, n, pointers, indices, &position);
    let fronts = chains(&tree, pointers, indices, &position, &bucket_ptr, &bucket);

    // The children of each front, and a postorder: a front is factored once all its children
    // are, and their blocks are then the top of the stack.
    let count = fronts.pivots.len();
    let mut child_ptr = vec![0usize; count + 1];
    for &p in &fronts.parent {
        if p != NONE {
            child_ptr[p + 1] += 1;
        }
    }
    for f in 0..count {
        child_ptr[f + 1] += child_ptr[f];
    }
    let mut children = vec![0usize; child_ptr[count]];
    let mut cursor = child_ptr.clone();
    for (f, &p) in fronts.parent.iter().enumerate() {
        if p != NONE {
            children[cursor[p]] = f;
            cursor[p] += 1;
        }
    }
    let order = postorder(&fronts.parent, &child_ptr, &children);
    let mut place = vec![0usize; count];
    for (p, &f) in order.iter().enumerate() {
        place[f] = p;
    }

    let mut s = Symbolic {
        position,
        pivots: Vec::with_capacity(count),
        column_ptr: Vec::with_capacity(count + 1),
        columns: Vec::with_capacity(fronts.columns.len()),
        child_ptr: Vec::with_capacity(count + 1),
        children: Vec::with_capacity(children.len()),
        row_ptr: Vec::with_capacity(count + 1),
        coordinates: Vec::with_capacity(fronts.rows.iter().sum()),
        reflection_ptr: Vec::with_capacity(count + 1),
        vector_ptr: vec![0],
        entering: vec![NONE; m],
        block_ptr: Vec::with_capacity(count),
        stack: 0,
        work: 0,
        widest: 0,
        tallest: 0,
        r_ptr: vec![0; n + 1],
        r_cols: Vec::new(),
        pivot_row: vec![NONE; n],
    };
    s.column_ptr.push(0);
    s.child_ptr.push(0);
    s.row_ptr.push(0);
    s.reflection_ptr.push(0);

    // The place of each column in the front being laid out.
    let mut local = vec![0usize; n];
    // The rows of the matrix that hold a row of `R` in the end.
    let mut taken = vec![false; m];
    // The rows of the front being laid out, each with the column of the front it begins in.
    let mut entries: Vec<(usize, usize)> = Vec::new();
    let mut tally: Vec<usize> = Vec::new();
    // The height of the stack of contribution blocks.
    let mut top = 0usize;
    for (p, &f) in order.iter().enumerate() {
        let cols = &fronts.columns[fronts.column_ptr[f]..fronts.column_ptr[f + 1]];
        let (width, pivots) = (cols.len(), fronts.pivots[f]);
        for (l, &col) in cols.iter().enumerate() {
            local[col] = l;
        }

        // The rows of the matrix that begin in a pivot column, then the rows of the children's
        // blocks: row `u` of a block is zero before the block's column `u`.
        entries.clear();
        for (t, &col) in cols[..pivots].iter().enumerate() {
            for &i in &bucket[bucket_ptr[col]..bucket_ptr[col + 1]] {
                entries.push((t, i));
                s.entering[i] = p;
            }
        }
        // The children's blocks are the top of the stack, the first child's the lowest of them:
        // they are read before the block of this front is written over them.
        let mut base = top;
        for (index, &child) in children[child_ptr[f]..child_ptr[f + 1]].iter().enumerate() {
            let c = place[child];
            s.children.push(c);
            if index == 0 {
                base = s.block_ptr[c];
            }
            let reflections = s.reflection_ptr[c + 1] - s.reflection_ptr[c];
            let first = s.pivots[c];
            if reflections > first {
                let passed_cols = &s.columns[s.column_ptr[c] + first..s.column_ptr[c + 1]];
                let passed_rows = &s.coordinates[s.row_ptr[c] + first..s.row_ptr[c] + reflections];
                for (&row, &col) in passed_rows.iter().zip(passed_cols) {
                    entries.push((local[col], row));
                }
            }
        }
        let height = entries.len();
        debug_assert_eq!(
            height, fronts.rows[f],
            "the rows of a front were miscounted"
        );

        // A counting sort by the first column, which leaves in `tally[t]` the number of rows
        // that begin at or before column `t`.
        tally.clear();
        tally.resize(width + 1, 0);
        for &(key, _) in &entries {
            tally[key + 1] += 1;
        }
        for t in 0..width {
            tally[t + 1] += tally[t];
        }
        let begin = s.coordinates.len();
        s.coordinates.resize(begin + height, 0);
        for &(key, row) in &entries {
            s.coordinates[begin + tally[key]] = row;
            tally[key] += 1;
        }

        // The staircase: column `t` is nonzero in the rows above `end`, the rows that begin at
        // or before it and never fewer than `t + 1`, so that the reflection of column `t` has
        // its own row. A front of fewer rows than columns has a reflection for each row.
        let reflections = height.min(width);
        let mut size = 0usize;
        for (t, &below) in tally[..width].iter().enumerate() {
            let end = below.max(t + 1).min(height);
            size += end;
            if t < reflections {
                let last = s.vector_ptr[s.vector_ptr.len() - 1];
                s.vector_ptr.push(last + end - t);
            }
        }
        s.work = s.work.max(size);
        s.widest = s.widest.max(width);
        s.tallest = s.tallest.max(height);

        // The first rows of the triangle are rows of `R`, one for each pivot column that has a
        // row at all.
        for (t, &col) in cols[..pivots.min(height)].iter().enumerate() {
            let row = s.coordinates[begin + t];
            s.r_ptr[col + 1] = width - t;
            s.pivot_row[col] = row;
            taken[row] = true;
        }

        // The rest of the triangle is the contribution block, kept by columns: column `v` of it
        // has `v + 1` elements, and no more than the block has rows.
        let passed = reflections.saturating_sub(pivots);
        let block = passed * (passed + 1) / 2 + (width - pivots - passed) * passed;
        s.block_ptr.push(base);
        top = base + block;
        s.stack = s.stack.max(top);

        s.pivots.push(pivots);
        s.columns.extend_from_slice(cols);
        s.column_ptr.push(s.columns.len());
        s.child_ptr.push(s.children.len());
        s.row_ptr.push(s.coordinates.len());
        s.reflection_ptr.push(s.reflection_ptr[p] + reflections);
    }
    debug_assert_eq!(top, 0, "a root front left a contribution block");

    // The rows of `R` by columns of the matrix, each the tail of its front's columns.
    for j in 0..n {
        s.r_ptr[j + 1] += s.r_ptr[j];
    }
    s.r_cols = vec![0usize; s.r_ptr[n]];
    for p in 0..count {
        let cols = &s.columns[s.column_ptr[p]..s.column_ptr[p + 1]];
        let height = s.row_ptr[p + 1] - s.row_ptr[p];
        for (t, &col) in cols[..s.pivots[p].min(height)].iter().enumerate() {
            s.r_cols[s.r_ptr[col]..s.r_ptr[col + 1]].copy_from_slice(&cols[t..]);
        }
    }

    // A pivot column without a row has a zero row of `R`, and its column of `Q` may be any unit
    // vector orthogonal to the others: the image of a row no other column of `Q` stands on. With
    // at least as many rows as columns there are enough of them.
    let mut spare = (0..m).filter(|&i| !taken[i]);
    for row in s.pivot_row.iter_mut().filter(|row| **row == NONE) {
        *row = spare
            .next()
            .expect("a matrix with no fewer rows than columns has a row for every column");
    }
    s
}

/// The inverse of the column permutation, `position[old] = new`.
///
/// # Panics
///
/// Panics if `permutation` isn't a permutation of `0..n`.
#[track_caller]
fn inverse(permutation: &[usize], n: usize) -> Vec<usize> {
    assert_eq!(
        permutation.len(),
        n,
        "the ordering isn't a permutation of the columns"
    );
    let mut position = vec![NONE; n];
    for (new, &old) in permutation.iter().enumerate() {
        assert!(
            old < n && position[old] == NONE,
            "the ordering isn't a permutation of the columns"
        );
        position[old] = new;
    }
    position
}

/// The column elimination tree of `B = A·Q_c`, the elimination tree of `BᴴB`, as the parent of
/// each column, [`NONE`] for a root.
///
/// A row of `B` with elements in the columns `c₁ < c₂ < …` makes a clique of them in `BᴴB`,
/// whose elimination tree is the path `c₁ → c₂ → …`, and the tree of a union of cliques is the
/// tree of the union of their paths: so column `k` is linked, for each row it has an element in,
/// to the previous column of that row alone. The links are followed with path compression
/// (Liu, "A compact row storage scheme for Cholesky factors using elimination trees", ACM Trans.
/// Math. Softw. 12, 1986), which makes the whole tree nearly linear in the stored elements.
fn column_tree(
    m: usize,
    n: usize,
    pointers: &[usize],
    indices: &[usize],
    position: &[usize],
) -> Vec<usize> {
    // The pattern of `B` by columns: a counting sort of the elements by their new column.
    let mut start = vec![0usize; n + 1];
    for &j in indices {
        start[position[j] + 1] += 1;
    }
    for k in 0..n {
        start[k + 1] += start[k];
    }
    let mut rows = vec![0usize; indices.len()];
    let mut cursor = start.clone();
    for i in 0..m {
        for &j in &indices[pointers[i]..pointers[i + 1]] {
            let k = position[j];
            rows[cursor[k]] = i;
            cursor[k] += 1;
        }
    }

    let mut parent = vec![NONE; n];
    // A column on the way from each column to the root of its tree so far.
    let mut ancestor = vec![NONE; n];
    // The last column seen in each row.
    let mut previous = vec![NONE; m];
    for k in 0..n {
        for &i in &rows[start[k]..start[k + 1]] {
            // Climb from the row's previous column to the root of its tree, pointing everything
            // passed at `k`; a root below `k` becomes a child of `k`. `NONE` is above any column.
            let mut j = previous[i];
            while j < k {
                let above = ancestor[j];
                ancestor[j] = k;
                if above == NONE {
                    parent[j] = k;
                }
                j = above;
            }
            previous[i] = k;
        }
    }
    parent
}

/// The fronts of the decomposition, in the order of their first columns.
///
/// The row of `R` of column `k` reaches the columns of the rows of `B` that begin at `k` and
/// the columns of the rows its children in the tree pass on, a child that has no row left over
/// passing none. Column `k` joins the front of its only child when its row reaches exactly what
/// the child's reaches beyond the child itself — a fundamental supernode, which costs no stored
/// zero — and begins a front of its own otherwise.
fn chains(
    tree: &[usize],
    pointers: &[usize],
    indices: &[usize],
    position: &[usize],
    bucket_ptr: &[usize],
    bucket: &[usize],
) -> Chains {
    let n = tree.len();
    let mut found = Chains {
        pivots: Vec::new(),
        column_ptr: vec![0],
        columns: Vec::new(),
        rows: Vec::new(),
        parent: Vec::new(),
    };
    // The fronts whose last pivot column is a child of each column, as lists: the first of
    // them, and for each front the next one of the same list.
    let mut head = vec![NONE; n];
    let mut next: Vec<usize> = Vec::new();
    // The column whose row last took each column in.
    let mut mark = vec![NONE; n];
    let mut reach: Vec<usize> = Vec::new();
    let mut kids: Vec<usize> = Vec::new();
    for k in 0..n {
        reach.clear();
        reach.push(k);
        mark[k] = k;
        let own = &bucket[bucket_ptr[k]..bucket_ptr[k + 1]];
        for &i in own {
            for &j in &indices[pointers[i]..pointers[i + 1]] {
                let c = position[j];
                if mark[c] != k {
                    mark[c] = k;
                    reach.push(c);
                }
            }
        }
        kids.clear();
        let mut passed = 0usize;
        let mut f = head[k];
        while f != NONE {
            kids.push(f);
            let cols = &found.columns[found.column_ptr[f]..found.column_ptr[f + 1]];
            let left = found.rows[f]
                .min(cols.len())
                .saturating_sub(found.pivots[f]);
            if left > 0 {
                passed += left;
                for &c in &cols[found.pivots[f]..] {
                    if mark[c] != k {
                        mark[c] = k;
                        reach.push(c);
                    }
                }
            }
            f = next[f];
        }
        reach.sort_unstable();
        debug_assert_eq!(reach[0], k, "a row of R reaches a column before its own");

        let joined = match kids[..] {
            [child] => {
                let cols = &found.columns[found.column_ptr[child]..found.column_ptr[child + 1]];
                (cols[found.pivots[child]..] == reach[..]).then_some(child)
            }
            _ => None,
        };
        let front = match joined {
            Some(child) => {
                found.pivots[child] += 1;
                found.rows[child] += own.len();
                child
            }
            None => {
                let front = found.pivots.len();
                found.pivots.push(1);
                found.rows.push(own.len() + passed);
                found.parent.push(NONE);
                found.columns.extend_from_slice(&reach);
                found.column_ptr.push(found.columns.len());
                next.push(NONE);
                for &child in &kids {
                    found.parent[child] = front;
                }
                front
            }
        };
        if tree[k] != NONE {
            next[front] = head[tree[k]];
            head[tree[k]] = front;
        }
    }
    found
}

/// A postorder of the forest of fronts: every front after all the fronts below it, and the
/// fronts below one front next to each other.
fn postorder(parent: &[usize], child_ptr: &[usize], children: &[usize]) -> Vec<usize> {
    let mut order = Vec::with_capacity(parent.len());
    // The fronts from a root down to the one being visited, each with its next child to visit.
    let mut path: Vec<(usize, usize)> = Vec::new();
    for (root, &above) in parent.iter().enumerate() {
        if above != NONE {
            continue;
        }
        path.push((root, child_ptr[root]));
        while let Some(&mut (f, ref mut at)) = path.last_mut() {
            if *at < child_ptr[f + 1] {
                let child = children[*at];
                *at += 1;
                path.push((child, child_ptr[child]));
            } else {
                order.push(f);
                path.pop();
            }
        }
    }
    order
}
