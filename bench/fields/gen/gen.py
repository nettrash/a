#!/usr/bin/env python3
"""Writes ../rs/src/consts.rs (the primes of the prime-field rows) and ../rs/src/arkf.rs (an
arkworks MontConfig for each prime no ark crate ships) from primes.txt, which primes.gp writes."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
rows = [l.split() for l in open(os.path.join(here, "primes.txt")) if l.strip()]
out = ["// Written by bench/gen/gen.py from primes.txt (PARI/GP, setrand(20261124)). Do not edit.", "",
       "/// (name, prime in hexadecimal, least quadratic non-residue, p mod 8, bits)",
       "pub const PRIMES: &[(&str, &str, u64, u64, u64)] = &["]
for name, p, g, m8, bits in rows:
    out.append(f'    ("{name}", "{int(p):x}", {g}, {m8}, {bits}),')
out.append("];")
open(os.path.join(here, "../rs/src/consts.rs"), "w").write("\n".join(out) + "\n")
# arkworks fields for the primes without a curve crate here (BN254 and BLS12-381 come from theirs).
skip = {"bn254", "bn254r", "bls381", "bls381r", "p600"}
ark = ["// Written by bench/gen/gen.py from primes.txt. Do not edit.", "#![allow(missing_docs)]", "",
       "use ark_ff::fields::{Fp, MontBackend, MontConfig};", ""]
for name, p, g, m8, bits in rows:
    if name in skip:
        continue
    n = (int(bits) + 63) // 64
    c = name.upper() + "Config"
    ark += ["#[derive(MontConfig)]", f'#[modulus = "{p}"]', f'#[generator = "{g}"]', f"pub struct {c};",
            f"pub type {name.upper()} = Fp<MontBackend<{c}, {n}>, {n}>;", ""]
open(os.path.join(here, "../rs/src/arkf.rs"), "w").write("\n".join(ark))
