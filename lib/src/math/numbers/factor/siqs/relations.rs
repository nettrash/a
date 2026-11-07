//! The relations: each the polynomial and the `x` it came from, the primes of the factor base that
//! divide `A·g(x)` (with repetition), the sign of `g(x)`, and up to two large primes.
//!
//! The relations with large primes are the edges of a graph whose vertices are the large primes and
//! 1 (a relation with one large prime `p` joins `p` and 1): every cycle of it is a set of relations
//! whose product has each large prime to an even power, and the number of independent cycles is the
//! number of edges less the number of vertices plus the number of components, kept up to date by a
//! union–find of the vertices as relations arrive (Lenstra and Manasse 1994, for two large primes).
//! The cycles themselves come at the end from a spanning forest of the graph, by breadth-first
//! search: each edge outside the forest closes one, with the paths of the forest from its ends to
//! their common ancestor.

use std::collections::{HashMap, HashSet};

/// A relation.
#[derive(Clone, Copy, Debug)]
pub(super) struct Relation {
    /// The number of its `A` in the list.
    pub(super) a: u32,
    /// The number of its polynomial for that `A`.
    pub(super) v: u32,
    /// The `x`.
    pub(super) x: i32,
    /// Whether `g(x)` is negative.
    pub(super) negative: bool,
    /// The large primes, 1 for none.
    pub(super) lp: [u32; 2],
    /// Where its factors start in the store's list.
    pub(super) start: u32,
    /// The number of its factors.
    pub(super) len: u32,
}

/// The relations found, and the count of what they make.
pub(super) struct Store {
    /// The relations.
    pub(super) rels: Vec<Relation>,
    /// The factors of all relations, as indices into the factor base.
    pub(super) factors: Vec<u32>,
    /// The values `|Ax + B| mod 2^64` of the relations, so that none is taken twice (two different
    /// ones with the same value, which is unlikely, cost a relation).
    seen: HashSet<u64>,
    /// The number of full relations.
    fulls: usize,
    /// The vertex of each large prime (1 is vertex 0).
    vertex: HashMap<u32, u32>,
    /// The union–find forest of the vertices.
    parent: Vec<u32>,
    /// The number of independent cycles.
    cycles: usize,
}

impl Store {
    /// No relations.
    pub(super) fn new() -> Store {
        Store {
            rels: Vec::new(),
            factors: Vec::new(),
            seen: HashSet::new(),
            fulls: 0,
            vertex: HashMap::new(),
            parent: vec![0],
            cycles: 0,
        }
    }

    /// The number of independent products with no large prime to an odd power: full relations and
    /// cycles.
    pub(super) fn count(&self) -> usize {
        self.fulls + self.cycles
    }

    /// The vertex of the large prime `p`, 1 for vertex 0.
    fn vertex_of(&mut self, p: u32) -> u32 {
        if p == 1 {
            return 0;
        }
        let next = self.parent.len() as u32;
        let v = *self.vertex.entry(p).or_insert(next);
        if v == next {
            self.parent.push(next);
        }
        v
    }

    /// The root of the vertex `x`, halving the path.
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let up = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = up;
            x = up;
        }
        x
    }

    /// Adds the relation with the factors `factors` and the value `key`, unless one with the same
    /// value is there: whether it was added.
    pub(super) fn add(&mut self, mut rel: Relation, factors: &[u32], key: u64) -> bool {
        if !self.seen.insert(key) {
            return false;
        }
        rel.start = self.factors.len() as u32;
        rel.len = factors.len() as u32;
        self.factors.extend_from_slice(factors);
        if rel.lp == [1, 1] {
            self.fulls += 1;
        } else {
            let u = self.vertex_of(rel.lp[0]);
            let v = self.vertex_of(rel.lp[1]);
            let (ru, rv) = (self.find(u), self.find(v));
            if ru == rv {
                self.cycles += 1;
            } else {
                self.parent[ru as usize] = rv;
            }
        }
        self.rels.push(rel);
        true
    }

    /// The factors of relation `r`.
    pub(super) fn factors_of(&self, r: usize) -> &[u32] {
        let rel = &self.rels[r];
        &self.factors[rel.start as usize..(rel.start + rel.len) as usize]
    }

    /// The sets of relations whose products have no large prime to an odd power: each full
    /// relation alone, and each independent cycle of the graph of the others.
    pub(super) fn columns(&self) -> Vec<Vec<u32>> {
        let mut out = Vec::with_capacity(self.count());
        // The vertices and edges of the graph.
        let nv = self.parent.len();
        let mut edges: Vec<(u32, u32, u32)> = Vec::new();
        for (r, rel) in self.rels.iter().enumerate() {
            if rel.lp == [1, 1] {
                out.push(vec![r as u32]);
            } else {
                let u = self.vertex_id(rel.lp[0]);
                let v = self.vertex_id(rel.lp[1]);
                edges.push((u, v, r as u32));
            }
        }
        // Adjacency lists.
        let mut deg = vec![0u32; nv + 1];
        for &(u, v, _) in &edges {
            deg[u as usize + 1] += 1;
            if u != v {
                deg[v as usize + 1] += 1;
            }
        }
        for i in 0..nv {
            deg[i + 1] += deg[i];
        }
        let mut adj = vec![0u32; deg[nv] as usize];
        let mut fill = deg.clone();
        for (e, &(u, v, _)) in edges.iter().enumerate() {
            adj[fill[u as usize] as usize] = e as u32;
            fill[u as usize] += 1;
            if u != v {
                adj[fill[v as usize] as usize] = e as u32;
                fill[v as usize] += 1;
            }
        }
        // The spanning forest: for each vertex the edge to its parent and its depth.
        const NONE: u32 = u32::MAX;
        let mut up = vec![NONE; nv];
        let mut depth = vec![NONE; nv];
        let mut tree = vec![false; edges.len()];
        let mut queue = Vec::new();
        for root in 0..nv {
            if depth[root] != NONE {
                continue;
            }
            depth[root] = 0;
            queue.clear();
            queue.push(root as u32);
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head] as usize;
                head += 1;
                for &e in &adj[deg[u] as usize..deg[u + 1] as usize] {
                    let (a, b, _) = edges[e as usize];
                    let other = if a as usize == u { b } else { a } as usize;
                    if depth[other] == NONE {
                        depth[other] = depth[u] + 1;
                        up[other] = e;
                        tree[e as usize] = true;
                        queue.push(other as u32);
                    }
                }
            }
        }
        // Each edge outside the forest closes a cycle.
        for (e, &(u, v, r)) in edges.iter().enumerate() {
            if tree[e] {
                continue;
            }
            let mut cycle = vec![r];
            let (mut a, mut b) = (u as usize, v as usize);
            let step = |x: usize, cycle: &mut Vec<u32>| -> usize {
                let f = up[x] as usize;
                let (p, q, rel) = edges[f];
                cycle.push(rel);
                if p as usize == x {
                    q as usize
                } else {
                    p as usize
                }
            };
            while depth[a] > depth[b] {
                a = step(a, &mut cycle);
            }
            while depth[b] > depth[a] {
                b = step(b, &mut cycle);
            }
            while a != b {
                a = step(a, &mut cycle);
                b = step(b, &mut cycle);
            }
            out.push(cycle);
        }
        out
    }

    /// The vertex of a large prime already seen.
    fn vertex_id(&self, p: u32) -> u32 {
        if p == 1 { 0 } else { self.vertex[&p] }
    }
}
