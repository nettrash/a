use core::cmp::Ordering::{Equal, Greater, Less};

/// Sorts a slice in place using double-ended selection sort.
///
/// Each pass scans the unsorted middle of the slice once, finds both its smallest and its
/// largest key, moves every element holding the smallest key to the front of that range and
/// every element holding the largest key to its back. Those elements are then in their final
/// places, so the next pass scans only what lies between them.
///
/// Three refinements keep the algorithm as cheap as selection sort can be:
///
/// * **Pairwise min-max scan.** Elements are taken two at a time: the two are compared with
///   each other first, then only the smaller one is compared with the running minimum and
///   only the larger one with the running maximum. A range of `m` elements costs
///   `⌈3m/2⌉ - 2` comparisons per pass instead of the `2m - 2` of two independent scans,
///   which is the fewest any algorithm can use to find both extremes.
/// * **Duplicate gathering ("bingo") for free.** Every comparison is a three-way
///   [`Ord::cmp`], so an element equal to the running minimum or maximum is recognised
///   without an extra comparison and joins that key's group. A pass therefore retires all
///   copies of two keys: input with `k` distinct keys finishes in at most `⌈k/2⌉` passes
///   instead of `⌊n/2⌋`. A pass in which every comparison reports [`Equal`] has proved the whole range
///   equal and ends the sort, again at no extra cost. An equal pair also skips the
///   comparison with the maximum whenever its minimum comparison already rules that out.
/// * **Lazy placement.** A running minimum or maximum that has no duplicate yet is only
///   remembered by index; elements are moved once a pass is over, or when a second copy of
///   the current extreme turns up. Input with distinct keys costs at most `n - 1` swaps,
///   exactly like classic selection sort, and gathering duplicates never costs a comparison.
///
/// Distinct keys get no benefit from the refinements beyond the pairwise scan, so the worst
/// case is any input with distinct keys, and on such input the number of comparisons
/// depends only on `n`: `n(3n - 2)/8` for even `n` and `3(n² - 1)/8` for odd `n`, about 25%
/// fewer than the `n(n - 1)/2` of classic selection sort.
///
/// # Properties
///
/// * **Not stable**: equal elements may be reordered.
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Not adaptive to order**: sorted input costs as much as any other with distinct keys.
///   It is adaptive to duplicates instead: the fewer distinct keys, the fewer passes.
///
/// | Case                          | Comparisons                         | Swaps          |
/// |-------------------------------|-------------------------------------|----------------|
/// | Best (all elements equal)     | `n - 1`                             | `0`            |
/// | `k` distinct keys             | `O(nk)`                             | `O(nk)`        |
/// | Worst (all keys distinct)     | `n(3n - 2)/8` or `3(n² - 1)/8`      | `≤ n - 1`      |
///
/// Gathering duplicates costs at most `2m` swaps in a pass over `m` elements, so input with
/// many repeated keys trades the `n - 1` swap bound for far fewer comparisons and passes.
/// Input that is already in non-decreasing order is never moved at all.
///
/// Selection sort is quadratic and is mainly of educational value; prefer
/// [`slice::sort_unstable`] for real workloads.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: elements are only ever exchanged with [`slice::swap`].
///
/// # Examples
///
/// ```
/// use a::sort::selection;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// selection(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // Three distinct keys: the first pass retires every 1 and every 3, the second the 2s.
/// let mut v = [3, 1, 2, 3, 1, 2, 2, 1];
/// selection(&mut v);
/// assert_eq!(v, [1, 1, 1, 2, 2, 2, 3, 3]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// selection(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn selection<T: Ord>(arr: &mut [T]) {
    // Everything before `lo` and from `hi` onwards is already in its final position.
    let (mut lo, mut hi) = (0, arr.len());
    while hi - lo > 1 {
        let mut pass = Pass::new(&mut arr[lo..hi]);
        pass.scan();
        match pass.finish() {
            Some((front, back)) => {
                lo += front;
                hi -= back;
            }
            None => return, // every element still unsorted is equal to every other
        }
    }
}

/// One pass of [`selection`] over the unsorted range `v`.
///
/// The scanned prefix `v[..end]` holds the elements equal to the running minimum (the min
/// group), the elements equal to the running maximum (the max group), and everything else in
/// no particular order. A group with a single element is only remembered by its index; once
/// it has two or more, it is kept contiguous:
///
/// * the min group occupies `v[..min_len]`, at the front, where it has to end up anyway;
/// * the max group occupies `v[mb..mb + max_len]`, somewhere after the min group.
///
/// While every element scanned so far is equal (`same`), they all belong to the min group and
/// the max fields are unused.
struct Pass<'a, T> {
    v: &'a mut [T],
    /// Number of elements scanned.
    end: usize,
    /// Whether every element scanned so far is equal.
    same: bool,
    /// Length of the contiguous min group at the front, or `0` while it is a single element.
    min_len: usize,
    /// Index of an element of the min group.
    mi: usize,
    /// Length of the contiguous max group, or `0` while it is a single element.
    max_len: usize,
    /// Start of the contiguous max group, when there is one.
    mb: usize,
    /// Index of an element of the max group.
    mx: usize,
    /// Indices of elements still being placed, kept up to date by [`Pass::swap`].
    held: [usize; 2],
}

impl<'a, T: Ord> Pass<'a, T> {
    fn new(v: &'a mut [T]) -> Self {
        Pass {
            v,
            end: 0,
            same: true,
            min_len: 0,
            mi: 0,
            max_len: 0,
            mb: 0,
            mx: 0,
            held: [0; 2],
        }
    }

    /// Scans the whole range, gathering the min and max groups.
    fn scan(&mut self) {
        let len = self.v.len();
        // An odd length starts from a lone element, an even one from a compared pair, so
        // the rest of the range always splits into whole pairs.
        if len % 2 == 1 {
            self.end = 1;
        } else {
            self.end = 2;
            match self.v[0].cmp(&self.v[1]) {
                Less => {
                    self.same = false;
                    self.mx = 1;
                }
                Greater => {
                    self.same = false;
                    self.mi = 1;
                }
                Equal => self.push_min(1),
            }
        }
        while self.end < len {
            let x = self.end;
            self.end += 2;
            self.pair(x, x + 1);
        }
    }

    /// Classifies the pair at `x` and `y` with at most three comparisons and places it.
    fn pair(&mut self, x: usize, y: usize) {
        // Before the first unequal pair, the minimum is also the maximum.
        let (min, max) = (self.mi, if self.same { self.mi } else { self.mx });
        match self.v[x].cmp(&self.v[y]) {
            Equal => {
                // Both elements share one key, so one comparison against the minimum often
                // settles both of them.
                self.held[0] = y;
                let joined_min = match self.v[x].cmp(&self.v[min]) {
                    Less => {
                        self.new_min(x);
                        true
                    }
                    Equal => {
                        self.push_min(x);
                        true
                    }
                    // Greater than the minimum while everything so far is equal, so it is
                    // also greater than the maximum.
                    Greater if self.same => {
                        self.new_max(x);
                        false
                    }
                    Greater => match self.v[x].cmp(&self.v[max]) {
                        Less => return, // both lie strictly between the extremes
                        Equal => {
                            self.push_max(x);
                            false
                        }
                        Greater => {
                            self.new_max(x);
                            false
                        }
                    },
                };
                let y = self.held[0];
                if joined_min {
                    self.push_min(y);
                } else {
                    self.push_max(y);
                }
            }
            order => {
                let (s, l) = if order == Less { (x, y) } else { (y, x) };
                // Both comparisons are made against the extremes as they were before the
                // pair, then the smaller element is placed before the larger one.
                let below = self.v[s].cmp(&self.v[min]);
                let above = self.v[l].cmp(&self.v[max]);
                self.held[0] = l;
                match below {
                    Less => self.new_min(s),
                    Equal => self.push_min(s),
                    Greater => {}
                }
                let l = self.held[0];
                match above {
                    Less => {}
                    // `l` equals the old maximum. If that was also the old minimum, `s` has
                    // just undercut it, which turned the old min group into the max group.
                    Equal => self.push_max(l),
                    Greater => self.new_max(l),
                }
            }
        }
    }

    /// Makes the element at `src` the new minimum; the old min group becomes ordinary.
    fn new_min(&mut self, src: usize) {
        if self.same {
            // Every element scanned so far was equal, and they are all larger than `src`:
            // they form the max group now, exactly where they already are.
            self.same = false;
            self.max_len = self.min_len;
            self.mb = 0;
            self.mx = self.mi;
        }
        self.min_len = 0;
        self.mi = src;
    }

    /// Makes the element at `src` the new maximum; the old max group becomes ordinary.
    fn new_max(&mut self, src: usize) {
        self.same = false;
        self.max_len = 0;
        self.mx = src;
    }

    /// Adds the element at `src`, which equals the minimum, to the min group.
    fn push_min(&mut self, src: usize) {
        self.held[1] = src;
        if self.min_len == 0 {
            // The minimum's only element so far opens the group at the front of the range.
            self.make_room_for_min();
            self.swap(0, self.mi);
            self.min_len = 1;
        }
        self.make_room_for_min();
        self.swap(self.min_len, self.held[1]);
        self.min_len += 1;
        self.mi = 0;
    }

    /// Frees the slot just after the min group if the max group starts there, by moving the
    /// max group's first element to just past its end.
    fn make_room_for_min(&mut self) {
        // The element being added to the min group belongs to neither group, and it lies
        // in the scanned prefix but not in front of the max group, so `mb + max_len` is
        // still inside the scanned prefix.
        if self.max_len > 0 && self.mb == self.min_len {
            self.swap(self.mb, self.mb + self.max_len);
            self.mb += 1;
            self.mx = self.mb;
        }
    }

    /// Adds the element at `src`, which equals the maximum, to the max group.
    fn push_max(&mut self, src: usize) {
        if self.max_len == 0 {
            // The maximum's only element so far opens the group where it already is.
            self.mb = self.mx;
            self.max_len = 1;
        }
        // Grow the group to the right while that stays inside the scanned prefix. When it
        // does not, the group ends the prefix and `src` lies before it but after the min
        // group, so the slot just before the group is free to take.
        let next = self.mb + self.max_len;
        let slot = if next < self.end {
            next
        } else {
            self.mb -= 1;
            self.mb
        };
        self.swap(slot, src);
        self.max_len += 1;
        self.mx = self.mb;
    }

    /// Moves the groups into their final places and returns how many elements were placed at
    /// the front and at the back, or `None` if the whole range turned out to be equal.
    fn finish(&mut self) -> Option<(usize, usize)> {
        if self.same {
            return None;
        }
        let len = self.v.len();
        // The max group goes first: the slot the minimum needs, `v[0]`, may lie inside it.
        let back = if self.max_len == 0 {
            self.swap(len - 1, self.mx);
            1
        } else {
            // Only the group's elements outside `v[target..]` move, each into a slot of
            // `v[target..]` that the group does not already occupy.
            let (start, count) = (self.mb, self.max_len);
            let target = len - count;
            let free = target.max(start + count);
            for k in 0..(target - start).min(count) {
                self.swap(start + k, free + k);
            }
            count
        };
        let front = if self.min_len == 0 {
            self.swap(0, self.mi);
            1
        } else {
            self.min_len
        };
        Some((front, back))
    }

    /// Swaps two elements and keeps every remembered index pointing at the same element.
    fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.v.swap(a, b);
        let [h0, h1] = &mut self.held;
        for index in [&mut self.mi, &mut self.mx, h0, h1] {
            if *index == a {
                *index = b;
            } else if *index == b {
                *index = a;
            }
        }
    }
}
