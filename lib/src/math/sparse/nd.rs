//! The nested dissection ordering.
//!
//! A set of vertices whose removal cuts a graph into two parts that no edge joins is a
//! *separator*. If the two parts are eliminated first and the separator last, no fill can appear
//! between the parts: the factor is as sparse there as the matrix. Each part is a smaller graph of
//! the same kind, so the same is done to it, and so on down — the dissection is *nested* — until
//! the parts are small enough for a minimum degree ordering to finish them. For the graphs of
//! meshes in the plane, whose separators have about `√n` vertices, the factor then has of the
//! order of `n·log n` elements (George, "Nested dissection of a regular finite element mesh",
//! SIAM J. Numer. Anal. 10, 1973), which no ordering improves on by more than a constant.
//!
//! The separators are found without any knowledge of a mesh, in two ways, and the smaller one is
//! taken. One is the automatic nested dissection of George and Liu ("An automatic nested
//! dissection algorithm for irregular finite element problems", SIAM J. Numer. Anal. 15, 1978):
//! the levels of a breadth-first search from a pseudo-peripheral vertex separate what comes before
//! them from what comes after, and one near the middle is taken. The other is a multilevel
//! bisection after Karypis and Kumar, in the `separator` module: the graph is coarsened by
//! matchings, cut, and the cut refined on the way back. Either is then made smaller where it can
//! be, by the matching method of Liu ("A graph partitioning algorithm by node separators", ACM
//! Trans. Math. Softw. 15, 1989): a largest matching between the separator and the vertices of one
//! part next to it shows whether some of the separator can be traded for fewer of those.

mod separator;

use super::SparseMatrix;
use super::amd::amd_of;
use super::graph::{Graph, Levels, NONE, dense_threshold};
use crate::math::matrix::Scalar;

/// The nested dissection ordering of the pattern of `A + Aᵀ`: a permutation `p` with
/// `p[new] = old` in which every separator is numbered after the two parts it separates.
///
/// The graph is cut recursively:
///
/// 1. A piece that is not connected is taken component by component.
/// 2. A connected piece gets two separators. One is the part of a level of the breadth-first
///    search from a pseudo-peripheral vertex that touches the next level (George and Liu): of the
///    levels that leave at least a quarter of the piece on either side, the one with the fewest
///    such vertices. The other comes from a multilevel bisection (Karypis and Kumar): the piece
///    is coarsened by matchings of its heaviest edges to a hundred or so vertices, cut there by
///    growing a part breadth-first from several vertices, and the cut is refined at every level
///    on the way back by the vertex-separator form of the method of Fiduccia and Mattheyses.
/// 3. Both are improved by matching (Liu): while a largest matching between a separator and its
///    neighbours in one of the parts leaves some of it unmatched, the smallest vertex cover of
///    that bipartite graph is a smaller separator, and replaces it. The smaller of the two
///    separators is kept, the more even cut of two of a size.
/// 4. The two parts are ordered in the same way, one after the other, and the separator after
///    both.
/// 5. A piece of at most 128 vertices is ordered by [`amd`](super::amd) on the graph it induces,
///    and so is a piece that has no separator worth the name: one in which every vertex is next
///    to every other, whose best separator holds more than an eighth of it, or whose cut leaves
///    less than an eighth of the rest on one side.
///
/// As in [`amd`](super::amd), a row with more than `10·√n` elements off the diagonal (and more
/// than 16) is set aside first and ordered last.
///
/// What this gains over minimum degree depends on the graph. On regular grids and on meshes in
/// three dimensions the fill is lower once there are some thousands of vertices, by more the
/// larger the matrix, and the elimination tree is short and wide — independent subtrees — where
/// minimum degree leaves a tall one: on the 128 × 128 grid it is 441 columns high against 787. On
/// irregular meshes of the plane of up to a few hundred thousand vertices, and on long thin
/// strips, minimum degree fills in less. So it does on trees and on random graphs of average
/// degree below three, which have small separators but little fill for a dissection to save: the
/// dissection fills in some 13 per cent more than minimum degree on random trees, and 25 to 77
/// per cent more on random graphs of average degree 1.5 to 2.5. A graph without small separators
/// — a random graph of average degree three or more, say — is handed to minimum degree by rule 5.
///
/// Only the pattern is read, as that of `A + Aᵀ` without its diagonal. On the graphs of meshes
/// every level of the dissection costs about the stored elements times their logarithm, and since
/// every cut leaves at least an eighth of a piece on either side there are at most
/// `log n / log(8/7)` levels. The refinement works out the gain of a move afresh from all the
/// neighbours of every vertex the move touches, though, so a vertex of high degree costs its
/// degree at every move next to it. On a graph of many vertices with nearly `10·√n` neighbours,
/// just short of being set aside, the time grows like `n^1.5`: on stars of that many leaves strung
/// on a path, a graph 64 times larger takes some 560 times as long.
///
/// # Panics
///
/// Panics if the matrix isn't square.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, elimination_tree, nested_dissection, symmetric_permutation};
///
/// // The five-point Laplacian of a 24 × 24 grid.
/// let k = 24;
/// let mut entries = Vec::new();
/// for y in 0..k {
///     for x in 0..k {
///         let v = y * k + x;
///         entries.push((v, v, 4.0));
///         if x + 1 < k {
///             entries.push((v, v + 1, -1.0));
///             entries.push((v + 1, v, -1.0));
///         }
///         if y + 1 < k {
///             entries.push((v, v + k, -1.0));
///             entries.push((v + k, v, -1.0));
///         }
///     }
/// }
/// let a = SparseMatrix::from_coo(k * k, k * k, &entries);
///
/// let p = nested_dissection(&a);
/// let b = symmetric_permutation(&a, &p);
/// // The last vertices are the first separator, and the two parts it leaves are independent:
/// // below it, the elimination tree splits in two.
/// let parent = elimination_tree(&b);
/// let root = k * k - 1;
/// assert_eq!(parent[root], None);
/// let mut top = root;
/// let children = |v: usize| (0..v).filter(|&c| parent[c] == Some(v)).count();
/// while children(top) == 1 {
///     top -= 1;
/// }
/// assert!(children(top) >= 2);
/// ```
#[must_use]
#[track_caller]
pub fn nested_dissection<T: Scalar>(a: &SparseMatrix<T>) -> Vec<usize> {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    nested_dissection_of(&Graph::of_symmetric_pattern(a))
}

/// A piece of at most this many vertices is ordered by minimum degree.
const LEAF: usize = 128;

/// How many times a separator may be improved by matching.
const ROUNDS: usize = 16;

/// The part before the separator.
const LEFT: u8 = 0;
/// The part after the separator.
const RIGHT: u8 = 1;
/// The separator.
const CUT: u8 = 2;

/// What is known of a piece waiting to be ordered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing: it may be in several components.
    Whole,
    /// It is connected.
    Connected,
    /// It is to be ordered by minimum degree as it is.
    Leaf,
}

/// A set of vertices that takes the places from `at` on in the order.
struct Piece {
    vertices: Vec<usize>,
    at: usize,
    state: State,
}

/// [`nested_dissection`] of a graph.
pub(super) fn nested_dissection_of(graph: &Graph) -> Vec<usize> {
    let n = graph.len();
    let limit = dense_threshold(n);
    let (sparse, mut dense): (Vec<usize>, Vec<usize>) =
        (0..n).partition(|&v| graph.degree(v) <= limit);
    dense.sort_unstable_by_key(|&v| (graph.degree(v), v));
    let mut work = Dissection {
        graph,
        piece: vec![NONE; n],
        pieces: 0,
        levels: Levels::new(n),
        level: vec![0; n],
        side: vec![LEFT; n],
        local: vec![0; n],
        met: vec![0; n],
        meetings: 0,
        order: vec![NONE; n],
    };
    work.order[sparse.len()..].copy_from_slice(&dense);
    // The recursion is kept on a stack of its own: the depth of a dissection is small when the
    // cuts are balanced, but nothing promises that they are.
    let mut stack = vec![Piece {
        vertices: sparse,
        at: 0,
        state: State::Whole,
    }];
    while let Some(piece) = stack.pop() {
        work.dissect(piece, &mut stack);
    }
    work.order
}

/// The working memory of a dissection.
struct Dissection<'a> {
    graph: &'a Graph,
    /// The piece every vertex is in; a number no piece has once the vertex has its place.
    piece: Vec<usize>,
    /// How many pieces have been numbered.
    pieces: usize,
    /// The breadth-first searches.
    levels: Levels,
    /// The level of every vertex of the piece being cut.
    level: Vec<usize>,
    /// The part of every vertex of the piece being cut.
    side: Vec<u8>,
    /// The number of a vertex among those it was collected with.
    local: Vec<usize>,
    /// The collection that met each vertex last.
    met: Vec<usize>,
    /// The number of collections made.
    meetings: usize,
    /// The ordering.
    order: Vec<usize>,
}

impl Dissection<'_> {
    /// Orders one piece, or cuts it into pieces that go on the stack.
    fn dissect(&mut self, piece: Piece, stack: &mut Vec<Piece>) {
        let Piece {
            mut vertices,
            at,
            state,
        } = piece;
        // A piece is told from everything else by a number of its own, so that no search of it
        // has to be told where it ends.
        let id = self.pieces;
        self.pieces += 1;
        for &v in &vertices {
            self.piece[v] = id;
        }
        if state == State::Leaf || vertices.len() <= LEAF {
            self.leaf(id, &mut vertices, at);
            return;
        }
        if state == State::Whole {
            let parts = self.components(id, &vertices);
            if parts.len() > 1 || parts[0].state == State::Leaf {
                let mut at = at;
                for part in parts {
                    let size = part.vertices.len();
                    stack.push(Piece { at, ..part });
                    at += size;
                }
                return;
            }
        }
        let Some(separator) = self.bisect(id, &vertices) else {
            self.leaf(id, &mut vertices, at);
            return;
        };
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for &v in &vertices {
            match self.side[v] {
                LEFT => left.push(v),
                RIGHT => right.push(v),
                _ => self.piece[v] = NONE,
            }
        }
        let parts = left.len() + right.len();
        self.order[at + parts..at + vertices.len()].copy_from_slice(&separator);
        let middle = at + left.len();
        stack.push(Piece {
            vertices: left,
            at,
            state: State::Whole,
        });
        stack.push(Piece {
            vertices: right,
            at: middle,
            state: State::Whole,
        });
    }

    /// Orders the piece by minimum degree on the graph it induces.
    fn leaf(&mut self, id: usize, vertices: &mut [usize], at: usize) {
        // In ascending order, so that the lists of the induced graph ascend too.
        vertices.sort_unstable();
        for (k, &v) in vertices.iter().enumerate() {
            self.local[v] = k;
        }
        let mut offsets = Vec::with_capacity(vertices.len() + 1);
        let mut neighbours = Vec::new();
        for &v in vertices.iter() {
            offsets.push(neighbours.len());
            for &w in self.graph.neighbours(v) {
                if self.piece[w] == id {
                    neighbours.push(self.local[w]);
                }
            }
        }
        offsets.push(neighbours.len());
        let induced = Graph::from_lists(offsets, neighbours);
        for (k, &l) in amd_of(&induced).iter().enumerate() {
            self.order[at + k] = vertices[l];
        }
        for &v in vertices.iter() {
            self.piece[v] = NONE;
        }
    }

    /// The connected components of the piece: the large ones each by itself, and the small ones
    /// together as one leaf, since minimum degree orders them as well at once as one by one.
    fn components(&mut self, id: usize, vertices: &[usize]) -> Vec<Piece> {
        self.meetings += 1;
        let meeting = self.meetings;
        let mut parts = Vec::new();
        let mut small = Vec::new();
        let mut stack = Vec::new();
        for &seed in vertices {
            if self.met[seed] == meeting {
                continue;
            }
            let mut part = vec![seed];
            self.met[seed] = meeting;
            stack.push(seed);
            while let Some(v) = stack.pop() {
                for &w in self.graph.neighbours(v) {
                    if self.piece[w] == id && self.met[w] != meeting {
                        self.met[w] = meeting;
                        part.push(w);
                        stack.push(w);
                    }
                }
            }
            if part.len() <= LEAF {
                small.extend(part);
            } else {
                parts.push(Piece {
                    vertices: part,
                    at: 0,
                    state: State::Connected,
                });
            }
        }
        if !small.is_empty() {
            parts.push(Piece {
                vertices: small,
                at: 0,
                state: State::Leaf,
            });
        }
        parts
    }

    /// Cuts a connected piece in two: sets the side of each of its vertices and returns the
    /// separator, or `None` when the piece has no separator worth cutting it by.
    ///
    /// Two separators are found, one from level sets and one by multilevel bisection, both
    /// improved by matching, and the smaller is kept, the more even one of two of a size. Level
    /// sets of a regular mesh are the straight cuts that suit its later dissections too; the
    /// multilevel cut finds what the level sets of an irregular mesh miss.
    fn bisect(&mut self, id: usize, vertices: &[usize]) -> Option<Vec<usize>> {
        let m = vertices.len();
        let by_levels = self.level_cut(id, vertices).map(|(separator, sizes)| {
            let sides: Vec<u8> = vertices.iter().map(|&v| self.side[v]).collect();
            (separator, sizes, sides)
        });
        let (mut separator, mut sizes) = self.multilevel_cut(id, vertices);
        if let Some((other, other_sizes, sides)) = by_levels {
            let key = |s: &[usize], z: [usize; 2]| (s.len(), z[0].abs_diff(z[1]));
            if key(&other, other_sizes) <= key(&separator, sizes) {
                for (&v, &side) in vertices.iter().zip(&sides) {
                    self.side[v] = side;
                }
                separator = other;
                sizes = other_sizes;
            }
        }
        debug_assert!(self.separates(id, vertices, &separator));
        // A separator of more than an eighth of the piece is no sign of the structure dissection
        // lives on: the graphs of meshes have far smaller ones, and on a graph without them
        // (a random graph of average degree three or more, say) minimum degree does better than
        // any dissection. A cut that leaves less than an eighth of the rest on one side gains
        // little, and refusing it keeps the dissection shallow.
        if 8 * separator.len() > m || 8 * sizes[0].min(sizes[1]) < m - separator.len() {
            return None;
        }
        Some(separator)
    }

    /// The separator a level of the breadth-first search from a pseudo-peripheral vertex gives,
    /// improved by matching, with the sizes of the two parts; `None` when the search has no level
    /// between two others. The sides of the vertices are set.
    fn level_cut(&mut self, id: usize, vertices: &[usize]) -> Option<(Vec<usize>, [usize; 2])> {
        let m = vertices.len();
        let (graph, piece) = (self.graph, &self.piece);
        let inside = |w: usize| piece[w] == id;
        self.levels.far(graph, vertices[0], &inside);
        let depth = self.levels.depth();
        if depth < 2 {
            // Every vertex is next to the root, and the root is as far out as any: no level lies
            // between two others.
            return None;
        }
        debug_assert_eq!(self.levels.vertices().len(), m);
        for j in 0..=depth {
            for &v in self.levels.level(j) {
                self.level[v] = j;
            }
        }
        // Of the levels between the first and the last, the vertices with a neighbour in the next
        // level are what separates the levels before from the levels after. The level is chosen
        // for having few of them while leaving a quarter of the piece on either side; if none
        // leaves that much, the one that leaves most.
        let mut best: Option<(usize, usize, usize)> = None;
        for j in 1..depth {
            let thin = self
                .levels
                .level(j)
                .iter()
                .filter(|&&v| {
                    graph
                        .neighbours(v)
                        .iter()
                        .any(|&w| inside(w) && self.level[w] == j + 1)
                })
                .count();
            let after = m - self.levels.before(j + 1);
            let smaller = after.min(m - after - thin);
            let balanced = 4 * smaller >= m;
            let better = match best {
                None => true,
                Some((_, best_thin, best_smaller)) => {
                    let best_balanced = 4 * best_smaller >= m;
                    if balanced != best_balanced {
                        balanced
                    } else if balanced {
                        (thin, best_smaller) < (best_thin, smaller)
                    } else {
                        smaller > best_smaller
                    }
                }
            };
            if better {
                best = Some((j, thin, smaller));
            }
        }
        let (cut, _, _) = best?;
        for j in 0..=depth {
            let side = match j.cmp(&cut) {
                core::cmp::Ordering::Less => LEFT,
                core::cmp::Ordering::Equal => CUT,
                core::cmp::Ordering::Greater => RIGHT,
            };
            for &v in self.levels.level(j) {
                self.side[v] = side;
            }
        }
        // Of the level, only the vertices with a neighbour in the next one separate anything: the
        // others go to the part before it, which they touch alone.
        let mut separator = Vec::new();
        let mut sizes = [self.levels.before(cut), m - self.levels.before(cut + 1)];
        for &v in self.levels.level(cut) {
            if graph
                .neighbours(v)
                .iter()
                .any(|&w| piece[w] == id && self.level[w] == cut + 1)
            {
                separator.push(v);
            } else {
                self.side[v] = LEFT;
                sizes[0] += 1;
            }
        }
        self.improve(id, &mut separator, &mut sizes);
        Some((separator, sizes))
    }

    /// The separator a multilevel bisection of the piece gives, improved by matching, with the
    /// sizes of the two parts. The sides of the vertices are set.
    fn multilevel_cut(&mut self, id: usize, vertices: &[usize]) -> (Vec<usize>, [usize; 2]) {
        for (k, &v) in vertices.iter().enumerate() {
            self.local[v] = k;
        }
        let mut offsets = Vec::with_capacity(vertices.len() + 1);
        let mut targets = Vec::new();
        for &v in vertices {
            offsets.push(targets.len());
            for &w in self.graph.neighbours(v) {
                if self.piece[w] == id {
                    targets.push(self.local[w]);
                }
            }
        }
        offsets.push(targets.len());
        let sides = separator::separate(separator::Weighted::new(offsets, targets));
        let mut separator = Vec::new();
        let mut sizes = [0, 0];
        for (&v, &side) in vertices.iter().zip(&sides) {
            self.side[v] = side;
            if side == CUT {
                separator.push(v);
            } else {
                sizes[usize::from(side)] += 1;
            }
        }
        debug_assert!(self.separates(id, vertices, &separator));
        self.improve(id, &mut separator, &mut sizes);
        (separator, sizes)
    }

    /// Improves the separator by matching, moving it into the larger part first, which evens the
    /// parts out, and into the smaller one when that gains nothing.
    fn improve(&mut self, id: usize, separator: &mut Vec<usize>, sizes: &mut [usize; 2]) {
        for _ in 0..ROUNDS {
            let larger = if sizes[0] >= sizes[1] { LEFT } else { RIGHT };
            if !self.advance(id, larger, separator, sizes)
                && !self.advance(id, 1 - larger, separator, sizes)
            {
                break;
            }
        }
    }

    /// Whether `separator` is what the sides of the piece say, and no edge joins the two parts.
    fn separates(&self, id: usize, vertices: &[usize], separator: &[usize]) -> bool {
        let cut = vertices.iter().filter(|&&v| self.side[v] == CUT).count();
        cut == separator.len()
            && separator.iter().all(|&v| self.side[v] == CUT)
            && vertices.iter().all(|&v| {
                self.side[v] != LEFT
                    || self
                        .graph
                        .neighbours(v)
                        .iter()
                        .all(|&w| self.piece[w] != id || self.side[w] != RIGHT)
            })
    }

    /// Moves the separator into the part `toward` where that makes it smaller, and returns
    /// whether it did.
    ///
    /// A set of vertices of the separator can be given to the other part if all their neighbours
    /// in `toward` take their place, and the separator shrinks when those are fewer. The largest
    /// such gain is what a largest matching between the separator and its neighbours in `toward`
    /// leaves unmatched, and the new separator is the smallest vertex cover of that bipartite
    /// graph.
    fn advance(
        &mut self,
        id: usize,
        toward: u8,
        separator: &mut Vec<usize>,
        sizes: &mut [usize; 2],
    ) -> bool {
        self.meetings += 1;
        let meeting = self.meetings;
        let mut offsets = Vec::with_capacity(separator.len() + 1);
        let mut edges = Vec::new();
        let mut next_to = Vec::new();
        for &x in separator.iter() {
            offsets.push(edges.len());
            for &w in self.graph.neighbours(x) {
                if self.piece[w] == id && self.side[w] == toward {
                    if self.met[w] != meeting {
                        self.met[w] = meeting;
                        self.local[w] = next_to.len();
                        next_to.push(w);
                    }
                    edges.push(self.local[w]);
                }
            }
        }
        offsets.push(edges.len());
        let Some((leaves, enters)) = smaller_cover(&offsets, &edges, next_to.len()) else {
            return false;
        };
        let leaving = leaves.iter().filter(|&&l| l).count();
        let entering = enters.iter().filter(|&&e| e).count();
        debug_assert!(entering < leaving);
        // The part moved into loses what enters the separator; the other gains what leaves it.
        // The move is not made if it leaves the smaller part smaller and below a quarter.
        let (to, from) = (usize::from(toward), usize::from(1 - toward));
        let m = sizes[0] + sizes[1] + separator.len();
        let smaller = (sizes[to] - entering).min(sizes[from] + leaving);
        if smaller < sizes[0].min(sizes[1]) && 4 * smaller < m {
            return false;
        }
        sizes[to] -= entering;
        sizes[from] += leaving;
        let mut moved = Vec::with_capacity(separator.len() + entering - leaving);
        for (k, &x) in separator.iter().enumerate() {
            if leaves[k] {
                self.side[x] = 1 - toward;
            } else {
                moved.push(x);
            }
        }
        for (k, &y) in next_to.iter().enumerate() {
            if enters[k] {
                self.side[y] = CUT;
                moved.push(y);
            }
        }
        *separator = moved;
        true
    }
}

/// The smallest vertex cover of a bipartite graph, when it is smaller than the left side.
///
/// The graph has the left vertices `0..offsets.len() - 1`, each with its neighbours among the
/// `right` right vertices in `edges`. A largest matching is found by the algorithm of Hopcroft and
/// Karp; by König's theorem the smallest cover has as many vertices as the matching has edges.
/// Returns `None` when the matching covers the left side, which is then a smallest cover itself.
/// Otherwise returns which left vertices are *not* in the cover and which right vertices *are*:
/// those an alternating path reaches from an unmatched left vertex.
fn smaller_cover(
    offsets: &[usize],
    edges: &[usize],
    right: usize,
) -> Option<(Vec<bool>, Vec<bool>)> {
    let left = offsets.len() - 1;
    let mut mate_of_left = vec![NONE; left];
    let mut mate_of_right = vec![NONE; right];
    // The layer of every left vertex in the search from the unmatched ones.
    let mut layer = vec![NONE; left];
    let mut cursor = vec![0usize; left];
    let mut queue = Vec::with_capacity(left);
    let mut path = Vec::new();
    loop {
        // The layers by breadth-first search along unmatched edges out and matched edges back,
        // down to the first layer from which an unmatched right vertex is seen.
        queue.clear();
        for x in 0..left {
            layer[x] = if mate_of_left[x] == NONE { 0 } else { NONE };
            if layer[x] == 0 {
                queue.push(x);
            }
        }
        let mut last = NONE;
        let mut head = 0;
        while head < queue.len() {
            let x = queue[head];
            head += 1;
            if layer[x] > last {
                break;
            }
            for &y in &edges[offsets[x]..offsets[x + 1]] {
                let other = mate_of_right[y];
                if other == NONE {
                    last = layer[x];
                } else if layer[other] == NONE {
                    layer[other] = layer[x] + 1;
                    queue.push(other);
                }
            }
        }
        if last == NONE {
            break;
        }
        // As many shortest augmenting paths as fit side by side, by depth-first search down the
        // layers; every edge is tried once.
        cursor.copy_from_slice(&offsets[..left]);
        for start in 0..left {
            if mate_of_left[start] != NONE {
                continue;
            }
            path.clear();
            path.push(start);
            while let Some(&x) = path.last() {
                if cursor[x] == offsets[x + 1] {
                    // Nothing leads on from here in this phase.
                    layer[x] = NONE;
                    path.pop();
                    continue;
                }
                let y = edges[cursor[x]];
                cursor[x] += 1;
                let other = mate_of_right[y];
                if other == NONE {
                    if layer[x] == last {
                        for &on_path in &path {
                            let via = edges[cursor[on_path] - 1];
                            mate_of_left[on_path] = via;
                            mate_of_right[via] = on_path;
                        }
                        break;
                    }
                } else if layer[x] != NONE && layer[other] == layer[x] + 1 {
                    path.push(other);
                }
            }
        }
    }
    if mate_of_left.iter().all(|&y| y != NONE) {
        return None;
    }
    let mut left_reached = vec![false; left];
    let mut right_reached = vec![false; right];
    queue.clear();
    for x in 0..left {
        if mate_of_left[x] == NONE {
            left_reached[x] = true;
            queue.push(x);
        }
    }
    while let Some(x) = queue.pop() {
        for &y in &edges[offsets[x]..offsets[x + 1]] {
            if !right_reached[y] {
                right_reached[y] = true;
                // A right vertex reached is matched, or the matching would not be a largest one.
                let other = mate_of_right[y];
                if other != NONE && !left_reached[other] {
                    left_reached[other] = true;
                    queue.push(other);
                }
            }
        }
    }
    Some((left_reached, right_reached))
}
