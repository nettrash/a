\\ The numbers of points of the Koblitz curves y^2 + x y = x^3 + a x^2 + 1, a = 0 and 1, over
\\ GF(2^641) = GF(2)[x] / (x^641 + x^11 + 1), by PARI's point counting, for
\\ tests/math/ec/binary.rs (CARD641_A0, CARD641_A1); and the irreducibility of the four
\\ polynomials of its WIDE fields.   gp -q binary_card641.gp < /dev/null
g = ffgen(Mod(1, 2) * (x^641 + x^11 + 1), 't);
for (a = 0, 1, printf("a=%d %X\n", a, ellcard(ellinit([1, a, 0, 0, 1], g))));
P = [x^641 + x^11 + 1, x^1279 + x^216 + 1, x^2281 + x^715 + 1, x^3217 + x^67 + 1];
for (i = 1, #P, print(P[i], " irreducible: ", polisirreducible(Mod(1, 2) * P[i])));
quit
