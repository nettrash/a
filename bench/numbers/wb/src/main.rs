//! White-box probe of the crate's `numbers` kernels: the crate's source (the repository's `lib/`)
//! compiled as modules of this binary, so that its `pub(crate)` items are reachable, checked
//! against GMP's `mpn`/`mpz` functions and timed beside them.
#![allow(dead_code, unused_imports, unused_macros, unused_variables, unused_mut)]
#![allow(clippy::all)]

#[path = "../../../../lib/src/math/mod.rs"]
pub mod math;
#[path = "../../../../lib/src/hash/mod.rs"]
pub mod hash;
#[path = "../../../../lib/src/sort/mod.rs"]
pub mod sort;

mod gmp;
mod probe;

fn main() {
    probe::main();
}
