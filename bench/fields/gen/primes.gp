\\ Fixed random primes of the general sizes of the fields scoreboard, and the least
\\ quadratic non-residue of each (arkworks' MontConfig needs a non-residue generator).
setrand(20261124);
nr(p) = my(g = 2); while(kronecker(g, p) != -1, g++); g;
show(name, p) = print(name, " ", p, " ", nr(p), " ", p % 8, " ", #binary(p));
show("p64", 2^64 - 2^32 + 1);
foreach([128, 192, 512, 576, 768, 1024, 2048, 4096], b, show(Str("p", b), nextprime(2^(b-1) + random(2^(b-1)))));
show("c25519", 2^255 - 19);
show("secp256k1", 2^256 - 2^32 - 977);
show("p256", 2^256 - 2^224 + 2^192 + 2^96 - 1);
show("p384", 2^384 - 2^128 - 2^96 + 2^32 - 1);
show("p521", 2^521 - 1);
show("p600", nextprime(2^599 + random(2^599)));
show("bn254", 21888242871839275222246405745257275088696311157297823662689037894645226208583);
show("bn254r", 21888242871839275222246405745257275088548364400416034343698204186575808495617);
show("bls381", 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab);
show("bls381r", 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001);
