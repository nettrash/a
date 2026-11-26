//! Vertex separators by multilevel bisection, which is how
//! [`nested_dissection`](super::super::nested_dissection) cuts a piece of a graph in two.
//!
//! A cut found on the graph itself by local moves alone sees only its neighbourhood. So the graph
//! is first made smaller: the heaviest edges of a matching are contracted, over and over, until a
//! hundred or so vertices are left, each standing for as many vertices of the graph as its weight
//! and each edge for as many edges. That graph is cut from several starting vertices by growing
//! one part breadth-first to half the weight, the vertices next to it becoming the separator, and
//! the best of these cuts is carried back through the graphs, a level at a time, improved at every
//! level by the vertex-separator form of the method of Fiduccia and Mattheyses: a vertex of the
//! separator moves into one part, those of its neighbours that were in the other part move into
//! the separator, and the moves that leave the separator lightest are kept.
//!
//! This is the multilevel scheme of Karypis and Kumar, "A fast and high quality multilevel scheme
//! for partitioning irregular graphs", SIAM J. Sci. Comput. 20 (1998), with heavy-edge matching,
//! breadth-first growing and refinement of the separator itself at every level, as in their
//! ordering by nested dissection.

use super::{CUT, LEFT, RIGHT};
use crate::math::sparse::graph::NONE;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The coarsening stops at this many vertices.
const COARSEST: usize = 100;

/// How many starting vertices the cut of the coarsest graph is grown from.
const TRIES: usize = 6;

/// How many refinement passes a level gets at most; they stop once one gains nothing.
const PASSES: usize = 8;

/// A graph whose vertices and edges carry weights: a vertex of a coarse graph stands for as many
/// vertices of the graph being cut as its weight, an edge for as many edges.
pub(super) struct Weighted {
    /// Where the neighbours of each vertex begin in `targets`.
    offsets: Vec<usize>,
    /// The neighbours, vertex after vertex.
    targets: Vec<usize>,
    /// The weight of each edge of `targets`.
    edge_weights: Vec<usize>,
    /// The weight of each vertex.
    weights: Vec<usize>,
    /// The weight of all the vertices.
    total: usize,
}

impl Weighted {
    /// The graph of the given lists, every vertex and every edge of weight one: `offsets` of
    /// length one more than the vertices, and the neighbours of each vertex, without the vertex
    /// itself and symmetric.
    pub(super) fn new(offsets: Vec<usize>, targets: Vec<usize>) -> Self {
        let n = offsets.len() - 1;
        let edge_weights = vec![1; targets.len()];
        Self {
            offsets,
            targets,
            edge_weights,
            weights: vec![1; n],
            total: n,
        }
    }

    /// The number of vertices.
    fn len(&self) -> usize {
        self.weights.len()
    }

    /// The neighbours of `v` with the weights of the edges to them.
    fn edges(&self, v: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let (begin, end) = (self.offsets[v], self.offsets[v + 1]);
        self.targets[begin..end]
            .iter()
            .copied()
            .zip(self.edge_weights[begin..end].iter().copied())
    }

    /// The number of neighbours of `v`.
    fn degree(&self, v: usize) -> usize {
        self.offsets[v + 1] - self.offsets[v]
    }
}

/// The side of every vertex of the connected graph `g` — `LEFT`, `RIGHT`, or `CUT` for the
/// separator — such that no edge joins the two parts.
pub(super) fn separate(g: Weighted) -> Vec<u8> {
    let mut finer: Vec<(Weighted, Vec<usize>)> = Vec::new();
    let mut graph = g;
    while graph.len() > COARSEST {
        let (coarse, map) = coarsen(&graph);
        // A matching that shrinks the graph by less than a tenth has met a star or a clique,
        // where further levels would cost time and gain nothing.
        if 10 * coarse.len() > 9 * graph.len() {
            break;
        }
        finer.push((graph, map));
        graph = coarse;
    }
    let mut side = initial(&graph);
    for (fine, map) in finer.iter().rev() {
        // A coarse vertex in a part puts all it stands for there, so no edge joins the parts.
        side = map.iter().map(|&c| side[c]).collect();
        refine(fine, &mut side);
    }
    side
}

/// The graph with a matching of heavy edges contracted, and the vertex of it that every vertex
/// of `g` went into.
///
/// The vertices are visited from the fewest neighbours up, and each is matched with the
/// unmatched neighbour it shares the heaviest edge with, which keeps the heavy edges inside the
/// coarse vertices, out of reach of any cut. No coarse vertex may weigh more than one and a half
/// times its share of the coarsest graph, so that no single vertex is too heavy to balance.
fn coarsen(g: &Weighted) -> (Weighted, Vec<usize>) {
    let n = g.len();
    // From the fewest neighbours up, vertices of one degree in their order: a counting sort.
    let most = (0..n).map(|v| g.degree(v)).max().unwrap_or(0);
    let mut first = vec![0usize; most + 2];
    for v in 0..n {
        first[g.degree(v) + 1] += 1;
    }
    for d in 0..=most {
        first[d + 1] += first[d];
    }
    let mut order = vec![0usize; n];
    for v in 0..n {
        let d = g.degree(v);
        order[first[d]] = v;
        first[d] += 1;
    }
    let heaviest = (3 * g.total).div_ceil(2 * COARSEST).max(2);
    let mut mate = vec![NONE; n];
    for &v in &order {
        if mate[v] != NONE {
            continue;
        }
        let mut best = NONE;
        let mut best_weight = 0;
        for (u, w) in g.edges(v) {
            if mate[u] == NONE
                && g.weights[v] + g.weights[u] <= heaviest
                && (best == NONE
                    || w > best_weight
                    || (w == best_weight && g.weights[u] < g.weights[best]))
            {
                best = u;
                best_weight = w;
            }
        }
        if best == NONE {
            mate[v] = v;
        } else {
            mate[v] = best;
            mate[best] = v;
        }
    }
    // The coarse vertices in the order of the first of their vertices.
    let mut map = vec![NONE; n];
    let mut count = 0;
    for v in 0..n {
        if map[v] == NONE {
            map[v] = count;
            map[mate[v]] = count;
            count += 1;
        }
    }
    let mut offsets = Vec::with_capacity(count + 1);
    let mut targets = Vec::with_capacity(g.targets.len());
    let mut edge_weights = Vec::with_capacity(g.targets.len());
    let mut weights = Vec::with_capacity(count);
    // Where each coarse neighbour sits in the list being built, so that the edges to it add up.
    let mut slot = vec![NONE; count];
    for v in 0..n {
        if map[v] != weights.len() {
            continue;
        }
        let c = map[v];
        let begin = targets.len();
        offsets.push(begin);
        let pair = if mate[v] == v {
            [v, NONE]
        } else {
            [v, mate[v]]
        };
        let mut weight = 0;
        for fine in pair.into_iter().filter(|&f| f != NONE) {
            weight += g.weights[fine];
            for (u, w) in g.edges(fine) {
                let cu = map[u];
                if cu == c {
                    continue;
                }
                if slot[cu] == NONE {
                    slot[cu] = targets.len();
                    targets.push(cu);
                    edge_weights.push(w);
                } else {
                    edge_weights[slot[cu]] += w;
                }
            }
        }
        weights.push(weight);
        for &cu in &targets[begin..] {
            slot[cu] = NONE;
        }
    }
    offsets.push(targets.len());
    let coarse = Weighted {
        offsets,
        targets,
        edge_weights,
        weights,
        total: g.total,
    };
    (coarse, map)
}

/// The best of several cuts of the coarsest graph, each grown breadth-first from a starting
/// vertex to half the weight and refined.
fn initial(g: &Weighted) -> Vec<u8> {
    let n = g.len();
    let mut order = Vec::with_capacity(n);
    let mut reached = vec![NONE; n];
    // A far vertex, the last a search from vertex 0 reaches, and vertices spread over the rest.
    let mut seeds = vec![breadth_first(g, 0, &mut order, &mut reached, 0)];
    seeds.extend((0..TRIES - 1).map(|k| k * n / (TRIES - 1)));
    let mut best: Option<((usize, usize, usize), Vec<u8>)> = None;
    for (attempt, &seed) in seeds.iter().enumerate() {
        breadth_first(g, seed, &mut order, &mut reached, attempt + 1);
        let mut side = vec![RIGHT; n];
        let mut grown = 0;
        for &v in &order {
            if 2 * grown >= g.total {
                break;
            }
            side[v] = LEFT;
            grown += g.weights[v];
        }
        // The separator: the vertices of the second part next to the first.
        for v in 0..n {
            if side[v] == RIGHT && g.edges(v).any(|(u, _)| side[u] == LEFT) {
                side[v] = CUT;
            }
        }
        refine(g, &mut side);
        let key = score(g, &weights_of(g, &side));
        if best.as_ref().is_none_or(|(b, _)| key < *b) {
            best = Some((key, side));
        }
    }
    best.map_or_else(|| vec![LEFT; n], |(_, side)| side)
}

/// Lays the vertices reached from `seed` in `order`, nearest first, marking them with `mark`, and
/// returns the last.
fn breadth_first(
    g: &Weighted,
    seed: usize,
    order: &mut Vec<usize>,
    reached: &mut [usize],
    mark: usize,
) -> usize {
    order.clear();
    if g.len() == 0 {
        return 0;
    }
    order.push(seed);
    reached[seed] = mark;
    let mut head = 0;
    while head < order.len() {
        let v = order[head];
        head += 1;
        for (u, _) in g.edges(v) {
            if reached[u] != mark {
                reached[u] = mark;
                order.push(u);
            }
        }
    }
    order[order.len() - 1]
}

/// The weights of the two parts and of the separator.
fn weights_of(g: &Weighted, side: &[u8]) -> [usize; 3] {
    let mut weight = [0; 3];
    for (v, &s) in side.iter().enumerate() {
        weight[usize::from(s)] += g.weights[v];
    }
    weight
}

/// How good a cut is, smaller being better: how far its heavier part is beyond three fifths of
/// the whole, then the weight of its separator, then the difference of the parts.
fn score(g: &Weighted, weight: &[usize; 3]) -> (usize, usize, usize) {
    let largest = (3 * g.total).div_ceil(5);
    (
        weight[0].max(weight[1]).saturating_sub(largest),
        weight[2],
        weight[0].abs_diff(weight[1]),
    )
}

/// The weights of the neighbours of the vertices in each part, kept up to date while a refinement
/// moves vertices from part to part, so that a gain costs no walk over the neighbours.
///
/// A vertex's are summed when its gain is first asked for, and from then on every vertex that
/// changes sides adds its weight to, or takes it from, those of its neighbours that have them.
/// The gains are those summed afresh, to the unit.
struct Near {
    /// The weights of the neighbours in `LEFT` and in `RIGHT`, of the vertices that have them.
    weights: Vec<[usize; 2]>,
    /// Whether each vertex has them.
    known: Vec<bool>,
}

impl Near {
    fn new(n: usize) -> Self {
        Self {
            weights: vec![[0; 2]; n],
            known: vec![false; n],
        }
    }

    /// What moving the separator vertex `v` into the part `to` takes out of the separator: its
    /// own weight, less that of its neighbours in the other part, which move into the separator.
    fn gain(&mut self, g: &Weighted, side: &[u8], v: usize, to: u8) -> i64 {
        if !self.known[v] {
            let mut near = [0usize; 2];
            for (u, _) in g.edges(v) {
                let s = side[u];
                if s != CUT {
                    near[usize::from(s)] += g.weights[u];
                }
            }
            self.weights[v] = near;
            self.known[v] = true;
        }
        g.weights[v] as i64 - self.weights[v][usize::from(1 - to)] as i64
    }

    /// Records that `u` went from the part (or the separator) `from` to `to`.
    fn moved(&mut self, g: &Weighted, u: usize, from: u8, to: u8) {
        let w = g.weights[u];
        for (x, _) in g.edges(u) {
            if self.known[x] {
                let near = &mut self.weights[x];
                if from != CUT {
                    near[usize::from(from)] -= w;
                }
                if to != CUT {
                    near[usize::from(to)] += w;
                }
            }
        }
    }
}

/// Improves the cut `side` of `g` by passes of the vertex-separator Fiduccia-Mattheyses method.
///
/// A pass moves, one at a time, the separator vertex whose move gains most into a part — into the
/// lighter one when it can, so that the parts stay within three fifths of the whole — and locks
/// it; those of its neighbours that were in the other part join the separator. Moves that gain
/// nothing are made too, which lets a pass climb out of a local minimum, until a number of them
/// in a row has brought nothing; then the pass is undone back to the best cut it went through.
/// The gains come from [`Near`], kept up to date as vertices change sides, and a vertex whose
/// gain a move leaves as it was keeps its entry in the heap.
fn refine(g: &Weighted, side: &mut [u8]) {
    let n = g.len();
    let largest = (3 * g.total).div_ceil(5);
    let patience = (n / 50).clamp(25, 200);
    let mut weight = weights_of(g, side);
    // The vertices moved in a pass are locked for the rest of it: marked with the pass.
    let mut locked = vec![0usize; n];
    // Every gain pushed for a vertex and a part is stamped, and only the latest stamp counts:
    // the heaps are updated by pushing anew and skipping what is stale.
    let mut stamp = vec![[0u64; 2]; n];
    let mut heaps: [BinaryHeap<(i64, Reverse<usize>, u64)>; 2] =
        [BinaryHeap::new(), BinaryHeap::new()];
    let mut touched = vec![NONE; n];
    let mut near = Near::new(n);
    // The pass in which each vertex last had its gains pushed, and the gains its entries hold:
    // a vertex of the separator not locked has one entry in each heap that counts, and one whose
    // gain has not changed needs no new one.
    let mut entered = vec![0usize; n];
    let mut held = vec![[0i64; 2]; n];
    let mut moves: Vec<(usize, u8, usize)> = Vec::new();
    let mut pulled: Vec<usize> = Vec::new();
    let mut affected: Vec<usize> = Vec::new();
    let mut generation = 0;
    // The separator, in no particular order: the heaps give out the moves by their keys alone,
    // which no two entries share, whatever order they were pushed in.
    let mut cut: Vec<usize> = (0..n).filter(|&v| side[v] == CUT).collect();
    for pass in 1..=PASSES {
        for heap in &mut heaps {
            heap.clear();
        }
        moves.clear();
        pulled.clear();
        for &v in &cut {
            entered[v] = pass;
            for to in [LEFT, RIGHT] {
                let t = usize::from(to);
                stamp[v][t] += 1;
                held[v][t] = near.gain(g, side, v, to);
                heaps[t].push((held[v][t], Reverse(v), stamp[v][t]));
            }
        }
        let start = score(g, &weight);
        let mut best = start;
        let mut best_moves = 0;
        let mut idle = 0;
        loop {
            // The lighter part first: moving into it pulls vertices out of the heavier one.
            let first = if weight[0] <= weight[1] { LEFT } else { RIGHT };
            let mut chosen = None;
            'parts: for to in [first, 1 - first] {
                let t = usize::from(to);
                while let Some(&(gain, Reverse(v), s)) = heaps[t].peek() {
                    if side[v] != CUT || locked[v] == pass || s != stamp[v][t] {
                        heaps[t].pop();
                        continue;
                    }
                    // The part may not grow beyond three fifths of the whole, unless the move
                    // brings the heavier part nearer to it.
                    let into = weight[t] + g.weights[v];
                    let out = weight[1 - t] - (g.weights[v] as i64 - gain) as usize;
                    let over = |a: usize, b: usize| a.max(b).saturating_sub(largest);
                    if into > largest && over(into, out) >= over(weight[t], weight[1 - t]) {
                        // The part is full, for this vertex at least.
                        break;
                    }
                    heaps[t].pop();
                    chosen = Some((v, to));
                    break 'parts;
                }
            }
            let Some((v, to)) = chosen else {
                break;
            };
            let (t, other) = (usize::from(to), 1 - to);
            side[v] = to;
            near.moved(g, v, CUT, to);
            locked[v] = pass;
            weight[2] -= g.weights[v];
            weight[t] += g.weights[v];
            let begin = pulled.len();
            for (u, _) in g.edges(v) {
                if side[u] == other {
                    side[u] = CUT;
                    weight[usize::from(other)] -= g.weights[u];
                    weight[2] += g.weights[u];
                    pulled.push(u);
                }
            }
            for &u in &pulled[begin..] {
                near.moved(g, u, other, CUT);
            }
            moves.push((v, to, begin));
            // The gains that changed: of the separator neighbours of the moved vertex, of the
            // vertices that joined the separator, and of their separator neighbours.
            generation += 1;
            let mark = generation;
            affected.clear();
            for (u, _) in g.edges(v) {
                if side[u] == CUT && touched[u] != mark {
                    touched[u] = mark;
                    affected.push(u);
                }
            }
            for &u in &pulled[begin..] {
                for x in core::iter::once(u).chain(g.edges(u).map(|(x, _)| x)) {
                    if side[x] == CUT && touched[x] != mark {
                        touched[x] = mark;
                        affected.push(x);
                    }
                }
            }
            for &x in &affected {
                if locked[x] != pass {
                    // A vertex that has just joined the separator has no entry yet in this pass.
                    let fresh = entered[x] != pass;
                    entered[x] = pass;
                    for to in [LEFT, RIGHT] {
                        let t = usize::from(to);
                        let gain = near.gain(g, side, x, to);
                        if fresh || gain != held[x][t] {
                            stamp[x][t] += 1;
                            held[x][t] = gain;
                            heaps[t].push((gain, Reverse(x), stamp[x][t]));
                        }
                    }
                }
            }
            let now = score(g, &weight);
            if now < best {
                best = now;
                best_moves = moves.len();
                idle = 0;
            } else {
                idle += 1;
                if idle > patience {
                    break;
                }
            }
        }
        // Back to the best cut of the pass.
        while moves.len() > best_moves {
            let (v, to, begin) = moves.pop().expect("a move to undo");
            let other = usize::from(1 - to);
            for &u in &pulled[begin..] {
                side[u] = 1 - to;
                near.moved(g, u, CUT, 1 - to);
                weight[2] -= g.weights[u];
                weight[other] += g.weights[u];
            }
            pulled.truncate(begin);
            side[v] = CUT;
            near.moved(g, v, to, CUT);
            weight[usize::from(to)] -= g.weights[v];
            weight[2] += g.weights[v];
        }
        if best >= start {
            break;
        }
        // The separator now: what it was and what the kept moves pulled into it, less what they
        // moved out, each vertex once.
        generation += 1;
        let mark = generation;
        let mut now = Vec::with_capacity(cut.len());
        for &x in cut.iter().chain(&pulled) {
            if side[x] == CUT && touched[x] != mark {
                touched[x] = mark;
                now.push(x);
            }
        }
        cut = now;
    }
}
