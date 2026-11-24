\\ Reference values for tests/math/field/normal.rs: the minimal polynomials over GF(2) of Gauss
\\ periods, the normal elements of the Gaussian normal bases, by PARI/GP 2.19.
\\
\\ For p = Tm + 1 prime and k the order of 2 mod p, gamma is a primitive p-th root of unity in
\\ GF(2^k) (a random element to the power (2^k - 1)/p, not 1), u an element of order T mod p, and
\\ the Gauss period is sum_{j < T} gamma^(u^j). Each line: m, T (the lowest type when the line says
\\ "lowest"), and the minimal polynomial as a hexadecimal number, bit i the coefficient of y^i.
\\
\\   gp -q gen_normal.gp > normal_pari.txt
exists(m, T) = my(p = T*m + 1); isprime(p) && gcd(T*m / znorder(Mod(2, p)), m) == 1;
lowest(m) = my(T = 1); while (!exists(m, T), T++); T;
gauss_minpoly(m, T) = {
  my(p = T*m + 1, k = znorder(Mod(2, p)), g = ffgen(2^k, 'w), r, gam, u, b = 0);
  until(gam != 0 && gam != 1, r = random(g); gam = r^((2^k - 1) / p));
  u = lift(znprimroot(p)^((p - 1) / T));
  for (j = 0, T - 1, b += gam^lift(Mod(u, p)^j));
  subst(lift(minpoly(b)), 'x, 2);
}
default(parisizemax, 2^31);
setrand(20261123);
print("# gen_normal.gp: PARI/GP 2.19, minimal polynomials of Gauss periods.");
for (m = 2, 100, if (m % 8, my(T = lowest(m)); print("lowest ", m, " ", T, " ", Strprintf("%x", gauss_minpoly(m, T)))));
{
  foreach([[27, 6], [54, 3], [62, 6], [63, 6], [75, 10], [102, 6], [113, 2], [126, 3], [131, 2],
           [145, 10], [156, 13], [163, 4], [171, 12], [190, 10], [193, 4], [198, 22], [233, 2],
           [239, 2], [283, 6], [409, 4], [571, 10]], c,
    if (lowest(c[1]) != c[2], error("not the lowest type: ", c));
    print("lowest ", c[1], " ", c[2], " ", Strprintf("%x", gauss_minpoly(c[1], c[2]))));
  foreach([[7, 6], [7, 10], [11, 6], [163, 16], [233, 6], [5, 6], [9, 4], [10, 13], [10, 21],
           [12, 3], [12, 5], [14, 3], [14, 15]], c,
    if (!exists(c[1], c[2]), error("no such basis: ", c));
    print("type ", c[1], " ", c[2], " ", Strprintf("%x", gauss_minpoly(c[1], c[2]))));
}
