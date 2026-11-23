//! The approximate minimum degree ordering, and the quotient graph it is computed on.
//!
//! Eliminating a vertex of the graph of a symmetric matrix joins all its neighbours to each other,
//! and the edges that were not there before are the fill of the factor. The minimum degree rule
//! eliminates, at every step, a vertex with the fewest neighbours. Done on the graph itself that
//! needs room for the fill, which is what the ordering is meant to avoid finding out; done on the
//! *quotient graph* it needs no more room than the matrix has.
//!
//! In a quotient graph an eliminated vertex is kept as an *element*, standing for the clique its
//! elimination made, with the list of the *variables* — the vertices still to eliminate — it
//! reaches. A variable is adjacent to the variables in its own list and to every variable in the
//! lists of its elements. When a variable becomes the pivot, the elements it is adjacent to are
//! merged into the new one (they are *absorbed*, and their lists are freed), so the lists never
//! hold more than they did at the start.
//!
//! What is implemented here is the algorithm of Amestoy, Davis and Duff, "An approximate minimum
//! degree ordering algorithm", SIAM J. Matrix Anal. Appl. 17 (1996), with all its parts:
//!
//! * the approximate external degree, an upper bound of the true one that costs a walk over the
//!   lists the pivot touches instead of a union of them;
//! * mass elimination and the detection of indistinguishable variables by hashing, which are
//!   eliminated together as one supervariable;
//! * aggressive absorption of an element whose variables all lie in the new one;
//! * one pool of lists with compaction when it is full;
//! * dense rows set aside, to be eliminated after the others, and a postorder of the assembly
//!   tree at the end.
//!
//! The order is then renumbered by a postorder of the elimination tree, which keeps the fill and
//! puts every subtree of that tree in one run of columns.
//!
//! The same elimination orders the columns of a rectangular matrix, where the rows are elements
//! from the start: [`colamd`](super::colamd) builds that graph and runs it here.

use super::SparseMatrix;
use super::graph::{Graph, NONE, dense_threshold};
use super::ordering::in_postorder;
use crate::math::matrix::Scalar;

/// The approximate minimum degree ordering of the pattern of `A + Aᵀ`: a permutation `p` with
/// `p[new] = old` that keeps the fill of a Cholesky factorisation of `P·A·Pᵀ` low.
///
/// The method is that of Amestoy, Davis and Duff (1996): minimum degree on the quotient graph,
/// with approximate external degrees, mass elimination, supervariables found by hashing and
/// aggressive absorption of elements. A row with more than `10·√n` elements off the diagonal (and
/// more than 16) is set aside and eliminated after the others, by its length, since it would fill
/// its neighbours in whatever was done with it; the final postorder then places it after what it
/// touches, which is not always last. The pivots come out in a postorder of the assembly
/// tree, the tree of the elements and of what absorbed them.
///
/// Last, the whole order is renumbered by a postorder of the elimination tree of `P·A·Pᵀ`, which
/// is not the assembly tree: an element absorbed aggressively, or a row set aside, hangs elsewhere
/// in it. The postorder leaves the fill as it is and puts the columns of every subtree of the
/// elimination tree next to each other, as supernodal and multifrontal factorisations want. It
/// keeps every row after all the rows it is joined to that came before it, so a row set aside
/// still follows everything it touches.
///
/// Only the pattern is read, as that of `A + Aᵀ` without its diagonal; the elements may be of any
/// kind, and the matrix need not be symmetric. The working memory is a small multiple of the
/// stored elements. The time has no useful worst-case bound; on the matrices of meshes it grows a
/// little faster than the stored elements.
///
/// The ordering is a heuristic: it finds a low fill, not the lowest, and which of two vertices of
/// equal degree goes first is settled by their place in the lists.
///
/// # Panics
///
/// Panics if the matrix isn't square.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, amd, symmetric_permutation};
///
/// // An arrow whose shaft comes first: eliminating it joins every row to every other.
/// let n = 6;
/// let a = SparseMatrix::from_fn(n, n, |i, j| {
///     if i == j { n as f64 } else if i == 0 || j == 0 { -1.0 } else { 0.0 }
/// });
/// let nonzeros = |m: &SparseMatrix<f64>| {
///     let l = m.to_matrix().cholesky().unwrap().l();
///     l.as_slice().iter().filter(|&&x| x != 0.0).count()
/// };
/// assert_eq!(nonzeros(&a), n * (n + 1) / 2); // the factor is full
///
/// let p = amd(&a);
/// assert_eq!(nonzeros(&symmetric_permutation(&a, &p)), 2 * n - 1); // and now has no fill
/// ```
#[must_use]
#[track_caller]
pub fn amd<T: Scalar>(a: &SparseMatrix<T>) -> Vec<usize> {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    amd_of(&Graph::of_symmetric_pattern(a))
}

/// [`amd`] of a graph.
pub(super) fn amd_of(graph: &Graph) -> Vec<usize> {
    let n = graph.len();
    let limit = dense_threshold(n);
    let dense = |v: usize| graph.degree(v) > limit;
    let mut quotient = QuotientGraph::new(n, n, graph.entries());
    for v in 0..n {
        if !dense(v) {
            let neighbours = graph.neighbours(v).iter().copied().filter(|&w| !dense(w));
            quotient.variable(v, [], neighbours, None);
        }
    }
    let mut order = quotient.eliminate();
    // The dense rows last, the shorter first: among themselves they are all but a clique by then.
    let mut rest: Vec<usize> = (0..n).filter(|&v| dense(v)).collect();
    rest.sort_unstable_by_key(|&v| (graph.degree(v), v));
    order.extend(rest);
    // The assembly tree is not the elimination tree: an element absorbed aggressively, or a dense
    // row put after the others, hangs elsewhere in the one than in the other. A postorder of the
    // elimination tree itself keeps the fill and puts each of its subtrees in one run.
    in_postorder(graph, &order)
}

/// What a node of the quotient graph is at the moment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Not part of the elimination: a row or column that was set aside.
    Outside,
    /// A variable still to be eliminated, the principal one of its supervariable: it has a list
    /// and a place in the degree lists.
    Variable,
    /// A variable that was merged into another, or eliminated along with a pivot. It has no list;
    /// its parent is what it went into.
    Merged,
    /// An element whose list is alive: an eliminated pivot, or a row given as an element.
    Element,
    /// An element absorbed into its parent, or one whose variables are all gone. It has no list.
    Absorbed,
}

/// A quotient graph and the minimum degree elimination of it.
///
/// The nodes `0..variables` may be variables, and any node may be an element. All lists live in
/// one pool. The list of a variable holds its elements first, `elements[i]` of them, and then the
/// variables it is adjacent to directly; the list of an element holds its variables.
pub(super) struct QuotientGraph {
    /// The nodes below this may be variables.
    variables: usize,
    /// The lists, one after another, with the garbage of the lists that died between them.
    pool: Vec<usize>,
    /// The first entry of the pool that no list has used.
    free: usize,
    /// Where the list of each node begins in the pool.
    start: Vec<usize>,
    /// The length of the list of each node.
    length: Vec<usize>,
    /// How many of the first entries of a variable's list are elements.
    elements: Vec<usize>,
    /// What each node is.
    kind: Vec<Kind>,
    /// The number of original variables a principal variable stands for.
    weight: Vec<usize>,
    /// Of a variable, the bound of its external degree; of an element, the weight of its list.
    degree: Vec<usize>,
    /// What a merged variable or an absorbed element went into.
    parent: Vec<usize>,
}

impl QuotientGraph {
    /// A graph of `nodes` nodes, none of them part of the elimination yet, with room for lists of
    /// `entries` entries in all.
    pub(super) fn new(nodes: usize, variables: usize, entries: usize) -> Self {
        // The lists never hold more than they do at the start, and a new element has fewer
        // variables than there are: with room for both, a compaction always makes room for the
        // next element. A fifth more keeps the compactions few.
        let capacity = entries + variables + entries / 5;
        Self {
            variables,
            pool: vec![0; capacity],
            free: 0,
            start: vec![0; nodes],
            length: vec![0; nodes],
            elements: vec![0; nodes],
            kind: vec![Kind::Outside; nodes],
            weight: vec![1; nodes],
            degree: vec![0; nodes],
            parent: vec![NONE; nodes],
        }
    }

    /// Makes the node `i` a variable adjacent to the given elements and variables. Its degree is
    /// the one given, or the number of its variables when none is.
    pub(super) fn variable(
        &mut self,
        i: usize,
        elements: impl IntoIterator<Item = usize>,
        variables: impl IntoIterator<Item = usize>,
        degree: Option<usize>,
    ) {
        debug_assert!(i < self.variables);
        self.kind[i] = Kind::Variable;
        self.start[i] = self.free;
        for e in elements {
            self.pool[self.free] = e;
            self.free += 1;
        }
        self.elements[i] = self.free - self.start[i];
        for j in variables {
            self.pool[self.free] = j;
            self.free += 1;
        }
        self.length[i] = self.free - self.start[i];
        self.degree[i] = degree.unwrap_or(self.length[i] - self.elements[i]);
    }

    /// Makes the node `e` an element of the given variables, each of weight 1.
    pub(super) fn element(&mut self, e: usize, variables: impl IntoIterator<Item = usize>) {
        self.kind[e] = Kind::Element;
        self.start[e] = self.free;
        for j in variables {
            self.pool[self.free] = j;
            self.free += 1;
        }
        self.length[e] = self.free - self.start[e];
        self.degree[e] = self.length[e];
    }

    /// Eliminates every variable, each time one of least approximate degree, and returns the
    /// variables in the order of a postorder of the assembly tree.
    pub(super) fn eliminate(mut self) -> Vec<usize> {
        let nodes = self.kind.len();
        let total = (0..self.variables)
            .filter(|&i| self.kind[i] == Kind::Variable)
            .count();
        let mut lists = DegreeLists::new(nodes, total);
        // Entered backwards, so that of the variables of equal degree the first comes first.
        for i in (0..self.variables).rev() {
            if self.kind[i] == Kind::Variable {
                debug_assert!(self.degree[i] < total);
                lists.insert(i, self.degree[i]);
            }
        }
        // Whether a variable is in the element being formed.
        let mut in_front = vec![false; nodes];
        // For the elements met in a step: the step, and the weight of their part outside the new
        // element.
        let mut seen = vec![NONE; nodes];
        let mut outside = vec![0usize; nodes];
        // The hash buckets of the variables of the new element, chained through `chain`.
        let mut bucket = vec![NONE; total];
        let mut bucket_of = vec![0usize; nodes];
        let mut chain = vec![NONE; nodes];
        // The marks of the comparison of two lists, each comparison with a mark of its own.
        let mut mark = vec![0usize; nodes];
        let mut comparison = 0usize;

        let mut pivots = Vec::with_capacity(total);
        let mut eliminated = 0;
        let mut lowest = 0;
        while eliminated < total {
            let step = pivots.len();
            while lists.head[lowest] == NONE {
                lowest += 1;
            }
            let me = lists.head[lowest];
            lists.remove(me, lowest);
            pivots.push(me);
            let mut pivot_weight = self.weight[me];
            eliminated += pivot_weight;

            // The new element: every variable the pivot reaches, through its elements and through
            // its own list. A pivot with no elements is turned into its element where it lies;
            // otherwise the element is gathered at the free end of the pool, which is compacted
            // first if that end is too near.
            let own = self.elements[me];
            if own > 0 {
                let mut bound = self.length[me] - own;
                for k in 0..own {
                    bound += self.length[self.pool[self.start[me] + k]];
                }
                if self.free + bound.min(total) > self.pool.len() {
                    self.compact();
                }
            }
            self.kind[me] = Kind::Element;
            let begin = if own > 0 { self.free } else { self.start[me] };
            let mut end = begin;
            let mut front_degree = 0;
            for k in 0..=own {
                let source = if k < own {
                    self.pool[self.start[me] + k]
                } else {
                    me
                };
                let (from, count) = if k < own {
                    (self.start[source], self.length[source])
                } else {
                    (self.start[me] + own, self.length[me] - own)
                };
                for p in from..from + count {
                    let i = self.pool[p];
                    if self.kind[i] == Kind::Variable && !in_front[i] {
                        in_front[i] = true;
                        front_degree += self.weight[i];
                        lists.remove(i, self.degree[i]);
                        self.pool[end] = i;
                        end += 1;
                    }
                }
                if k < own {
                    self.kind[source] = Kind::Absorbed;
                    self.parent[source] = me;
                }
            }
            self.start[me] = begin;

            // For every element a variable of the new one belongs to, the weight of its part
            // outside the new one: its weight, less that of each variable of the new one in it.
            for p in begin..end {
                let i = self.pool[p];
                let weight = self.weight[i];
                for q in self.start[i]..self.start[i] + self.elements[i] {
                    let e = self.pool[q];
                    if self.kind[e] != Kind::Element {
                        continue;
                    }
                    if seen[e] == step {
                        outside[e] -= weight;
                    } else {
                        seen[e] = step;
                        outside[e] = self.degree[e] - weight;
                    }
                }
            }

            // The list of every variable of the new element is rewritten: the absorbed elements
            // go, and so do the variables now reached through the new element; the new element
            // comes first. What is left gives the external degree, and a hash of the list.
            for p in begin..end {
                let i = self.pool[p];
                let first = self.start[i];
                let (own_elements, length) = (self.elements[i], self.length[i]);
                let mut out = first;
                let mut external = 0usize;
                let mut hash = 0usize;
                for q in first..first + own_elements {
                    let e = self.pool[q];
                    if self.kind[e] != Kind::Element {
                        continue;
                    }
                    if outside[e] > 0 {
                        external += outside[e];
                        hash = hash.wrapping_add(e);
                        self.pool[out] = e;
                        out += 1;
                    } else {
                        // Aggressive absorption: the element lies wholly inside the new one.
                        self.kind[e] = Kind::Absorbed;
                        self.parent[e] = me;
                    }
                }
                let kept = out - first;
                for q in first + own_elements..first + length {
                    let j = self.pool[q];
                    if self.kind[j] == Kind::Variable && !in_front[j] {
                        external += self.weight[j];
                        hash = hash.wrapping_add(j);
                        self.pool[out] = j;
                        out += 1;
                    }
                }
                if out == first {
                    // Mass elimination: the variable reaches nothing but the new element, so it
                    // can be eliminated with the pivot and cause no fill.
                    self.kind[i] = Kind::Merged;
                    self.parent[i] = me;
                    in_front[i] = false;
                    front_degree -= self.weight[i];
                    pivot_weight += self.weight[i];
                    eliminated += self.weight[i];
                } else {
                    // The pivot or one of its elements has left the list, so there is room for
                    // the new element at its front.
                    debug_assert!(out < first + length);
                    self.degree[i] = self.degree[i].min(external);
                    self.pool[out] = self.pool[first + kept];
                    self.pool[first + kept] = self.pool[first];
                    self.pool[first] = me;
                    self.length[i] = out - first + 1;
                    self.elements[i] = kept + 1;
                    let h = hash % total;
                    bucket_of[i] = h;
                    chain[i] = bucket[h];
                    bucket[h] = i;
                }
            }

            // Supervariables: two variables of the new element with the same list are adjacent
            // to the same variables for good, and are merged. Only those in one bucket are
            // compared.
            for p in begin..end {
                let i = self.pool[p];
                if self.kind[i] != Kind::Variable {
                    continue;
                }
                let mut a = bucket[bucket_of[i]];
                bucket[bucket_of[i]] = NONE;
                while a != NONE && chain[a] != NONE {
                    comparison += 1;
                    let (first, length) = (self.start[a], self.length[a]);
                    for q in first + 1..first + length {
                        mark[self.pool[q]] = comparison;
                    }
                    let mut before = a;
                    let mut b = chain[a];
                    while b != NONE {
                        let after = chain[b];
                        let list = &self.pool[self.start[b] + 1..self.start[b] + self.length[b]];
                        if self.length[b] == length
                            && self.elements[b] == self.elements[a]
                            && list.iter().all(|&x| mark[x] == comparison)
                        {
                            self.weight[a] += self.weight[b];
                            self.kind[b] = Kind::Merged;
                            self.parent[b] = a;
                            in_front[b] = false;
                            chain[before] = after;
                        } else {
                            before = b;
                        }
                        b = after;
                    }
                    a = chain[a];
                }
            }

            // The variables that are left get their degrees and go back into the degree lists,
            // and the new element keeps them alone.
            let remaining = total - eliminated;
            let mut out = begin;
            for p in begin..end {
                let i = self.pool[p];
                if self.kind[i] != Kind::Variable {
                    continue;
                }
                in_front[i] = false;
                let weight = self.weight[i];
                let degree = (self.degree[i] + front_degree - weight).min(remaining - weight);
                self.degree[i] = degree;
                lists.insert(i, degree);
                lowest = lowest.min(degree);
                self.pool[out] = i;
                out += 1;
            }
            self.length[me] = out - begin;
            self.degree[me] = front_degree;
            self.weight[me] = pivot_weight;
            if own > 0 {
                self.free = out;
            }
            if out == begin {
                // Nothing is left for it to be an element of: a root of the assembly forest.
                self.kind[me] = Kind::Absorbed;
            }
        }
        self.numbering(&pivots)
    }

    /// Moves the lists that are alive to the front of the pool, in the order they lie in.
    fn compact(&mut self) {
        let nodes = self.kind.len();
        // The first entry of every live list is put aside and replaced by a mark that names its
        // owner, which no entry can be mistaken for: every entry is a node.
        for i in 0..nodes {
            if matches!(self.kind[i], Kind::Variable | Kind::Element) && self.length[i] > 0 {
                let first = self.start[i];
                self.start[i] = self.pool[first];
                self.pool[first] = nodes + i;
            }
        }
        let (mut from, mut to) = (0, 0);
        while from < self.free {
            let entry = self.pool[from];
            if entry < nodes {
                from += 1;
                continue;
            }
            let i = entry - nodes;
            let length = self.length[i];
            self.pool[to] = self.start[i];
            self.pool.copy_within(from + 1..from + length, to + 1);
            self.start[i] = to;
            from += length;
            to += length;
        }
        self.free = to;
    }

    /// The variables in their final order: the pivots in a postorder of the assembly forest, each
    /// followed by the variables that were eliminated with it.
    fn numbering(mut self, pivots: &[usize]) -> Vec<usize> {
        let steps = pivots.len();
        let mut step_of = vec![NONE; self.variables];
        for (step, &pivot) in pivots.iter().enumerate() {
            step_of[pivot] = step;
        }
        // Every merged variable is found a pivot: the end of the chain of what it went into. The
        // chain is shortened as it is walked.
        let mut offsets = vec![0usize; steps + 1];
        for v in 0..self.variables {
            if self.kind[v] != Kind::Merged {
                continue;
            }
            let mut pivot = v;
            while self.kind[pivot] == Kind::Merged {
                pivot = self.parent[pivot];
            }
            let mut walk = v;
            while self.kind[walk] == Kind::Merged {
                let up = self.parent[walk];
                self.parent[walk] = pivot;
                walk = up;
            }
            offsets[step_of[pivot] + 1] += 1;
        }
        for step in 0..steps {
            offsets[step + 1] += offsets[step];
        }
        let mut members = vec![0usize; offsets[steps]];
        let mut cursor = offsets.clone();
        for v in 0..self.variables {
            if self.kind[v] == Kind::Merged {
                let step = step_of[self.parent[v]];
                members[cursor[step]] = v;
                cursor[step] += 1;
            }
        }
        // The assembly forest: the parent of a pivot is the pivot whose element absorbed its own.
        // The children of every pivot in the order they were eliminated, which the backward pass
        // gives.
        let mut first_child = vec![NONE; steps];
        let mut next_sibling = vec![NONE; steps];
        let mut roots = Vec::new();
        for step in (0..steps).rev() {
            let parent = self.parent[pivots[step]];
            if parent == NONE {
                roots.push(step);
            } else {
                next_sibling[step] = first_child[step_of[parent]];
                first_child[step_of[parent]] = step;
            }
        }
        let mut order = Vec::with_capacity(steps + members.len());
        let mut stack = Vec::new();
        for &root in roots.iter().rev() {
            stack.push(root);
            while let Some(&top) = stack.last() {
                let child = first_child[top];
                if child == NONE {
                    order.push(pivots[top]);
                    order.extend_from_slice(&members[offsets[top]..offsets[top + 1]]);
                    stack.pop();
                } else {
                    first_child[top] = next_sibling[child];
                    stack.push(child);
                }
            }
        }
        debug_assert_eq!(order.len(), steps + members.len());
        order
    }
}

/// The variables by degree, in doubly linked lists, so that one of least degree is found at once
/// and any is taken out at once.
struct DegreeLists {
    /// The first variable of each degree.
    head: Vec<usize>,
    /// The variable after each in its list.
    next: Vec<usize>,
    /// The variable before each in its list.
    previous: Vec<usize>,
}

impl DegreeLists {
    /// Empty lists for `nodes` nodes and the degrees below `degrees`.
    fn new(nodes: usize, degrees: usize) -> Self {
        Self {
            head: vec![NONE; degrees],
            next: vec![NONE; nodes],
            previous: vec![NONE; nodes],
        }
    }

    /// Puts `i` at the front of the list of `degree`.
    fn insert(&mut self, i: usize, degree: usize) {
        let first = self.head[degree];
        self.next[i] = first;
        self.previous[i] = NONE;
        if first != NONE {
            self.previous[first] = i;
        }
        self.head[degree] = i;
    }

    /// Takes `i` out of the list of `degree`, which it is in.
    fn remove(&mut self, i: usize, degree: usize) {
        let (before, after) = (self.previous[i], self.next[i]);
        if before == NONE {
            self.head[degree] = after;
        } else {
            self.next[before] = after;
        }
        if after != NONE {
            self.previous[after] = before;
        }
    }
}
