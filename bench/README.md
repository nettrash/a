# Benchmarks of nettrash-a against its competitors

The harnesses of the v0.2 release speed pass: for every public operation that the v0.2 milestone
added (issues #15–#25 and #27), at representative sizes, the crate `a` (`nettrash-a`, built from
this repository's `lib/`) against every library that does the same thing. This folder is not part
of the published crate (it is outside `lib/`).

**The published figures** (the *Measured* tables of the pages in `lib/docs/math/`) come from an
**Apple M4 Max** (aarch64, macOS, one thread). The workflow
[`bench-x86.yml`](../.github/workflows/bench-x86.yml) runs the same harnesses on an x86_64 Linux
runner of GitHub Actions, by hand, for information: a shared virtual machine, noisier than the Mac.

## What a run produces

One scoreboard per area, `bench/results/<area>.tsv`, tab-separated, with the header

```
area	group	operation	params	ours_ns	reference	ref_ns	ratio	notes
```

one row per (operation, size, reference): `ours_ns` and `ref_ns` are median nanoseconds per
operation (or per item where `params` says so), `ratio` = ours / reference (above 1, we are
slower), and `notes` carries the same-work caveats (a reference that does less or more, the
accuracy compared, evaluation counts of iterative methods, a wide spread). Every reference that
does an operation has its own row. `bench/summarize.py [TSV ...]` prints, per group, the worst
ratio against the fastest reference and the number of losses (ratio > 1.05).

## How it measures

- **The same work.** Both sides compute the same thing from the same inputs, to the same precision
  or accuracy target, in the same output form; each harness compares the outputs before it times
  them and reports a mismatch (`MISMATCH` or `CHECK FAILED` in its log). Where a reference does
  less (a lower accuracy, no error checking, a weaker guarantee) or more, the row's notes say so.
- **Build profiles.** Our crate as a user gets it: each probe crate depends on `../../lib` by path
  and is built with `cargo build --release` and the default profile (opt-level 3, no LTO, default
  codegen units, no `target-cpu`); the Rust references the same way, pinned in each probe's
  `Cargo.toml` and `Cargo.lock`. The C and C++ references are the system's (Homebrew, or the
  distribution's packages); our own C/C++ drivers that call them are compiled at
  `-O3 -mcpu=native` (aarch64) or `-O3 -march=native` (x86_64). Some areas give a second figure for
  ours with `lto = "fat"`, `codegen-units = 1` (`PROFILE=fat`/`lto`, see below).
- **Timing.** `std::time::Instant`; each sample is a batch of at least 1 ms, 15 to 21 samples
  (per area), the **median** reported (and the spread when it is wide). Ours and the
  reference **alternate sample by sample in the same process** (ABAB…), so that load and clock
  changes hit both; fresh inputs per repetition where caches or branch predictors would flatter
  repeated identical inputs; a warm-up first. One thread on both sides unless an operation is
  defined as parallel; the BLAS-based references run with `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`. The SciPy rows of numerics/lp are timed in a
  separate process (not interleaved, said in their notes); those of special and sparse alternate
  with ours through a C ABI of the crate loaded by ctypes.
- **One timing process at a time.** Every timed command goes through `bench/slot.sh`, which lets
  one timed process run on the machine at a time (`BENCH_SLOTS=n` allows n); compiles need no slot.
  The workflow runs one shard per runner and times directly.

## Running it

```sh
bench/run_all.sh                        # every area in turn, into bench/results/
bench/run_all.sh fft poly               # some areas
bench/run_all.sh --build-only           # build every probe, time nothing
bench/run_all.sh --args "wmul wdiv" poly    # ARGS to the area's run.sh (here: two groups)
bench/poly/run.sh -o /tmp/poly.tsv gcd  # one area, one group, elsewhere
bench/summarize.py                      # the worst ratios of bench/results/*.tsv
```

A full run of every area takes most of a day on the M4 Max (the longest areas, poly, fields,
lattice and sparse, one and a half to two and a half hours each). Each `run.sh` prints its
progress on stderr; its raw parts and logs go to `bench/work/<area>/`.

**macOS** (as measured): Xcode's command-line tools, Rust stable, and Homebrew:

```sh
brew install gmp mpfr libmpc flint ntl pari fftw gsl boost primesieve primecount gmp-ecm msieve \
  m4ri fplll openssl@3 openssl@4 suite-sparse openblas
python3 -m pip install numpy scipy mpmath sympy
```

Apple's Accelerate (vDSP, LAPACK, Sparse Solvers) is part of macOS: its rows exist on macOS only.
`pkg-config` is optional there (the harnesses look in Homebrew's prefix).

**Linux x86_64** (Ubuntu): Rust stable and

```sh
sudo apt-get install build-essential pkg-config git m4 libgmp-dev libmpfr-dev libmpc-dev \
  libflint-dev libntl-dev libpari-dev pari-gp libfftw3-dev libgsl-dev libboost-dev \
  libprimesieve-dev libprimecount-dev libecm-dev libm4ri-dev libfplll-dev libssl-dev \
  libsuitesparse-dev libopenblas-dev liblapack-dev        # optional: msieve libmetis-dev
python3 -m pip install numpy scipy mpmath sympy
```

What differs from the Mac on Ubuntu 26.04 (the workflow's runner: FLINT 3.4.0, PARI 2.17.3,
NTL 11.5.1, MPC 1.3.1, M4RI 20240729, OpenSSL 3.5):

- `float`: gmp-mpfr-sys 1.7 (rug) accepts no MPC older than 1.4, so off macOS it builds the
  GMP 6.3.0, MPFR 4.2.2 and MPC 1.4.1 it bundles from source (Homebrew's versions; it needs `m4`
  and a few minutes once), and the probe's MPFR and MPC rows use that build too.
- FLINT 3.4 lacks three functions the shims call: the FLINT rows of poly's
  `real_root_intervals*` and `real_roots(prec)` (`fmpz_poly_isolate_real_roots`,
  `arb_fmpz_poly_real_roots`, FLINT 3.6) and of lattice's `smith_normal_form`
  (`fmpz_mat_snf_transform`, FLINT 3.5) are left out, and those rows say so in their notes.
- FLINT before 3.5 computes `fmpz_mat_snf` of a singular or non-square matrix by Kannan–Bachem,
  whose entries grow without bound (5 GB in two minutes on lattice's "30×30 rank 20"; on the
  runner it took the whole machine): lattice's `elementary_divisors` rows of those matrices have
  no FLINT figure there, and say so. Square nonsingular ones keep it.
- PARI before 2.19 can answer `bestappr(x, D)` with a convergent where a closer semiconvergent
  exists (fixed in 2.19.0): rational's `limit_denominator` rows with PARI say "not the same work";
  at 4096 limbs with a bound of half the length it needs more than 6 GB of stack, so that row has
  no PARI figure.
- msieve and OpenSSL 4 are not packaged: nt's msieve rows and curves' OpenSSL 4 rows are absent.

**Memory.** sparse's Cholesky command of one matrix peaks at up to about 6.2 GB (lap2d-1000;
measured on Linux, see `peak_gb` in `sparse/src/g_chol.rs`). A matrix whose peak, with a quarter
more, exceeds the memory available (`MemAvailable` of `/proc/meminfo`, `hw.memsize` on macOS,
`BENCH_MEM_GB=n` to set it) skips itself and leaves rows that say so, without figures. On the
x86 workflow's runners a guard in the Run step also kills the largest process when less than
1 GB stays available (`MEM_FLOOR_KB`): that group fails and says so in the log, and the shard
goes on, rather than the runner stopping with no log or artifact.

The references are found by `pkg-config` or in the standard prefixes (Homebrew's, `/usr/local`,
`/usr` and the multiarch directories). A reference that is missing loses its rows (the build
warns); an area whose main reference is missing writes a scoreboard with the header only and says
why (marked *required* below). mcl is not packaged: `bench/mcl.sh` clones
[herumi/mcl](https://github.com/herumi/mcl) at a pinned commit into `bench/ext/mcl` and builds
its static library (curves and fields call it; `MCL_DIR=dir` uses a build of your own, `NO_MCL=1`
leaves it out).

**Common settings** (environment): `BENCH_RESULTS` (default `bench/results`), `BENCH_WORK`
(default `bench/work`), `CARGO_TARGET_DIR` (default `bench/target`, one build directory for every
area), `CARGO_BUILD_JOBS` (default every core), `BENCH_SLOT` (the timing wrapper, default
`bench/slot.sh`; empty to time directly), `BENCH_SLOTS` (default 1).

## The areas

| area | what (`a::math::…`) | references | arguments of `run.sh` |
|---|---|---|---|
| `nt` | number theory of `numbers` (#15): sieves, π(x), primality, factoring (trial, ρ, p±1, ECM, SIQS), CRT, orders, discrete logs, Kronecker, square roots, arithmetic functions, sequences | primesieve, primecount, GMP, FLINT, PARI, GMP-ECM, msieve, primal, machine-prime, machine-factor, malachite | tasks (`ntbench list`); `NT_LONG=1` adds the rows of minutes, `NT_FILTER`, `NT_SAMPLES`, `NT_BUDGET`, `NT_SKIP` |
| `numbers` | the kernels of `Number` the pass added: short products, Jacobi, continued-fraction quotients (white-box: the crate's source compiled into the probe) | GMP (*required*) | `mul quadratic gcd` |
| `numerics` | roots, systems, quadrature, cubature, Gauss rules, ODEs, optimisation, L-BFGS-B, interpolation, AAA (#21) | GSL and Boost (*required*), argmin, ode_solvers, roots, SciPy | `roots systems quad cubature gauss ode opt lbfgsb interp aaa` |
| `lp` | `LinearProgram::solve` (#21; the numerics probe's group `lp`) | HiGHS 1.12 through SciPy (*required*) | — |
| `float` | `BigFloat`, `BigComplex` (#17) | MPFR, MPC (*required* on macOS; built from source elsewhere), rug, malachite-float, astro-float, dashu-float, Arb (FLINT) | `[--quick] [--filter s] [--skip s] [--samples n] [GROUP…]`; `PROFILE=fat` |
| `curves` | pairings (BLS12-381, BN254), hashing to curves, binary curves, SHA-2 (#24) | arkworks, blst, bls12_381, substrate-bn, mcl, OpenSSL 3 and 4, ring, sha2, RustCrypto curves, curve25519-dalek | `pairing h2c binary sha` |
| `special` | special functions (#20), f64 and f32 | statrs, special, puruspe, libm, lambert_w, the system libm, Boost.Math, GSL, SciPy | `[--quick] [GROUP…]` (`gamma … orthopoly f32 scipy`) |
| `rational` | `Rational` (#16, #27), rational and integer matrices | GMP (*required*), FLINT, PARI, malachite-q, num-rational, rug | groups (`ratbench list`); `LTO=1`, `SKIP_LONG=1` |
| `fft` | complex, real, DCT/DST, n-dimensional transforms, convolutions, plans (#19) | FFTW 3 (*required*), rustfft, realfft, rustdct, vDSP (macOS) | job filters (`"complex f64"`, …); `RUNS=n`, `PROFILE=fat` |
| `poly` | polynomials over word and multi-limb moduli, ℤ, ℚ, floats, GF(2) (#18) | FLINT, NTL, PARI, LAPACK (Accelerate on macOS), rustfft, realfft | `wmul wdiv mlimb zarith qarith farith gcd eval roots factor`; `PROFILE=lto` |
| `numtypes` | quaternions, dual numbers, fixed-width integers (#25) | glam, nalgebra, num-dual, ruint, crypto-bigint, bnum, malachite-nz, GMP, OpenSSL BN | `uint mont quat dual`; `PROFILE=lto` |
| `fields` | prime fields, extension fields, towers, binary fields, GF(2⁸)/GF(2¹²⁸)/GHASH (#24) | FLINT, NTL, GMP, OpenSSL, mcl, arkworks, blst, bls12_381, ghash, polyval | `prime ext tower binary gf2` |
| `lattice` | LLL, deep insertions, BKZ, enumeration, PSLQ, normal forms, exact linear algebra, `ModMatrix`, `BitMatrix` (#23) | FLINT, fplll, NTL, M4RI, PARI, mpmath | jobs `group[:filter]` (see `run.sh`); `LATBENCH_PROFILE=fat` |
| `sparse` | sparse storage and operations, Matrix Market I/O, orderings, Cholesky, LU, QR, Krylov, eigenvalues (#22) | SuiteSparse (*required*), Accelerate (macOS), faer, sprs, nalgebra-sparse, SciPy | `ops io ordering chol lu qr iter eigen py-…`; `QUICK=1`, `BENCH_MEM_GB` |

Every `run.sh` also takes `-o FILE` (the scoreboard's path).

## Versions on the M4 Max

The published figures were measured with: macOS 27.2, Apple clang 21.0, Rust 1.100.0; Homebrew's
GMP 6.3.0, MPFR 4.2.2, MPC 1.4.1, FLINT 3.6.0, NTL 11.6.0, PARI/GP 2.19.1, FFTW 3.3.11, GSL 2.8,
Boost 1.92.0, primesieve 12.18, primecount 9.0, GMP-ECM 7.0.7, msieve 1.53, M4RI 20260122,
fplll 5.5.0, OpenSSL 3.5.9 and 4.1.0, SuiteSparse 7.14.1 with OpenBLAS 0.3.34; mcl at commit
`a1c4e29e` (2026-11-18); Python 3.13 with NumPy 2.3.3, SciPy 1.18.0 (HiGHS 1.12.0), mpmath 1.3.0
and SymPy 1.14.0; the Rust references as pinned in each probe's `Cargo.toml` and `Cargo.lock`.
The machine's state during each campaign (load, date) is stated with each page's tables.

## What is not here

The phase-2 tools of the speed pass that compared the crate with an older copy of itself (A/B
against master, wired against unwired), the differential and confirmation probes of the audits,
profiling drivers and the summaries written for the pages: they need copies of earlier trees and
measure no row against a reference. The references are linked, never vendored; the drivers here
are our own code (MIT, as the crate).
