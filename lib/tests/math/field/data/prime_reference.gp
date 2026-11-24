\\ Writes prime_reference.txt: for each prime p, the Legendre symbols (kronecker) and the smaller
\\ square roots (PARI's sqrt in Z/pZ, min(r, p - r)) of the inputs
\\   a_i = ((p \ 3) i^3 + (p \ 7) i + 1) mod p,   i = 0 .. n - 1,
\\ which tests/math/field/prime.rs computes again with Number.
\\ Run: gp -q prime_reference.gp > prime_reference.txt   (PARI/GP 2.19)
\\ PARI 2.19.0's sqrt of an intmod may have a negative representative, so the root is reduced
\\ into [0, p) with % p before the smaller one is taken.
term(p, i) = ((p \ 3) * i^3 + (p \ 7) * i + 1) % p;
hex(x) = Strprintf("%x", x);
emit(p, n) = {
  my(a, l = "", r);
  print("p ", hex(p), " ", n);
  for (i = 0, n - 1, a = term(p, i); l = concat(l, if (kronecker(a, p) == 1, "+", if (kronecker(a, p) == -1, "-", "0"))));
  print("l ", l);
  for (i = 0, n - 1,
    a = term(p, i);
    if (kronecker(a, p) == -1, print("-"),
      r = lift(sqrt(Mod(a, p))) % p; print(hex(min(r, p - r)))));
};
findq(bits, s) = {my(q = 2^(bits-s-1)+1); while(!isprime(q*2^s+1), q+=2); q*2^s+1};
{
  my(ps = [3, 5, 7, 41, 65537, 15*2^27+1, 3*2^30+1, 2^61-1, 2^64-2^32+1, precprime(2^64-1),
    nextprime(2^64+1), precprime(2^128-1), precprime(2^192-1), nextprime(2^192+1),
    precprime(2^320-1), findq(256, 3), findq(256, 50),
    findq(256, 56), findq(256, 64),
    21888242871839275222246405745257275088696311157297823662689037894645226208583,
    21888242871839275222246405745257275088548364400416034343698204186575808495617,
    0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab,
    0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001,
    2^256 - 2^32 - 977, 2^255 - 19, 2^256 - 2^224 + 2^192 + 2^96 - 1,
    2^384 - 2^128 - 2^96 + 2^32 - 1, 2^521 - 1, precprime(2^383-1), findq(384, 4),
    precprime(2^448-1), findq(512, 5), precprime(2^512-1), nextprime(2^512+1), precprime(2^570-1),
    nextprime(2^576+1),
    nextprime(3*2^598), findq(600, 20), findq(600, 90), nextprime(5*2^1497)]);
  for (k = 1, #ps, emit(ps[k], if (#binary(ps[k]) > 580, 8, 24)));
}
