//! The pages under `docs/` describe each algorithm, with examples. Including them here, in
//! doctest builds only, makes `cargo test` compile and run those examples, so the pages cannot
//! fall behind the code. The pages are packaged with the crate, so this also works in a
//! published copy.
//!
//! A new page needs a line here.

#[doc = include_str!("../docs/sorting/default.md")]
struct SortingDefault;

#[doc = include_str!("../docs/sorting/bubble.md")]
struct SortingBubble;

#[doc = include_str!("../docs/sorting/selection.md")]
struct SortingSelection;

#[doc = include_str!("../docs/sorting/insertion.md")]
struct SortingInsertion;

#[doc = include_str!("../docs/sorting/cocktail_shaker.md")]
struct SortingCocktailShaker;

#[doc = include_str!("../docs/sorting/gnome.md")]
struct SortingGnome;

#[doc = include_str!("../docs/sorting/comb.md")]
struct SortingComb;

#[doc = include_str!("../docs/sorting/merge.md")]
struct SortingMerge;

#[doc = include_str!("../docs/sorting/quick.md")]
struct SortingQuick;

#[doc = include_str!("../docs/sorting/heap.md")]
struct SortingHeap;

#[doc = include_str!("../docs/sorting/tim.md")]
struct SortingTim;

#[doc = include_str!("../docs/sorting/intro.md")]
struct SortingIntro;

#[doc = include_str!("../docs/sorting/shell.md")]
struct SortingShell;

#[doc = include_str!("../docs/sorting/drift.md")]
struct SortingDrift;

#[doc = include_str!("../docs/sorting/counting.md")]
struct SortingCounting;

#[doc = include_str!("../docs/sorting/radix.md")]
struct SortingRadix;

#[doc = include_str!("../docs/sorting/bucket.md")]
struct SortingBucket;

#[doc = include_str!("../docs/sorting/pigeonhole.md")]
struct SortingPigeonhole;

#[doc = include_str!("../docs/sorting/tree.md")]
struct SortingTree;

#[doc = include_str!("../docs/sorting/cycle.md")]
struct SortingCycle;

#[doc = include_str!("../docs/sorting/bitonic.md")]
struct SortingBitonic;

#[doc = include_str!("../docs/sorting/odd_even.md")]
struct SortingOddEven;

#[doc = include_str!("../docs/sorting/stooge.md")]
struct SortingStooge;

#[doc = include_str!("../docs/sorting/sleep.md")]
struct SortingSleep;

#[doc = include_str!("../docs/math/numbers.md")]
struct MathNumbers;
