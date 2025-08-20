/// Bubble sort.
pub fn bubble_sort<T: Ord>(arr: &mut [T]) {
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