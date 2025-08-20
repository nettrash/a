/// Bubble sort.
pub fn bubble<T: Ord>(arr: &mut [T]) {
    // Early exit and reduced pass range via last-swap tracking.
    if arr.len() < 2 {
        return;
    }

    let mut n = arr.len();
    loop {
        let mut new_n = 0;
        for j in 1..n {
            if arr[j - 1] > arr[j] {
                arr.swap(j - 1, j);
                new_n = j; // last index where a swap happened
            }
        }
        if new_n == 0 {
            break; // already sorted
        }
        n = new_n; // next pass only needs to scan up to the last swap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_empty_slice() {
        let mut v: Vec<i32> = vec![];
        bubble(&mut v);
        assert_eq!(v, vec![]);
    }

    #[test]
    fn sorts_single_element() {
        let mut v = vec![42];
        bubble(&mut v);
        assert_eq!(v, vec![42]);
    }

    #[test]
    fn sorts_two_elements() {
        let mut v = vec![2, 1];
        bubble(&mut v);
        assert_eq!(v, vec![1, 2]);
    }

    #[test]
    fn sorts_already_sorted() {
        let mut v = vec![1, 2, 3, 4, 5];
        bubble(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sorts_reverse_sorted() {
        let mut v = vec![5, 4, 3, 2, 1];
        bubble(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sorts_with_duplicates() {
        let mut v = vec![3, 1, 2, 3, 2, 1, 0];
        bubble(&mut v);
        assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn sorts_strings() {
        let mut v = vec!["pear", "apple", "banana", "apple"];
        bubble(&mut v);
        assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
    }

    #[test]
    fn is_stable_for_equal_keys() {
        #[derive(Debug, Clone, Eq, PartialEq)]
        struct Pair {
            key: i32,
            idx: usize,
        }

        use std::cmp::Ordering;
        impl Ord for Pair {
            fn cmp(&self, other: &Self) -> Ordering {
                self.key.cmp(&other.key)
            }
        }
        impl PartialOrd for Pair {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut v = vec![
            Pair { key: 2, idx: 0 },
            Pair { key: 1, idx: 1 },
            Pair { key: 2, idx: 2 },
            Pair { key: 1, idx: 3 },
            Pair { key: 2, idx: 4 },
        ];

        bubble(&mut v);

        // After sorting by key, pairs with the same key should keep their original relative order.
        let ones: Vec<usize> = v.iter().filter(|p| p.key == 1).map(|p| p.idx).collect();
        let twos: Vec<usize> = v.iter().filter(|p| p.key == 2).map(|p| p.idx).collect();
        assert_eq!(ones, vec![1, 3]);
        assert_eq!(twos, vec![0, 2, 4]);
    }
}
