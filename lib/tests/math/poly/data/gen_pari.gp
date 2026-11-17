\\ The vectors of roots_pari.txt (the test against_pari of roots_real.rs): random integer polynomials (squarefree or
\\ not; the few on which polrootsreal fails skipped), PARI 2.19's polsturm count and polrootsreal at 160 bits printed to 40 digits.
\\ cd lib/tests/math/poly/data && gp -q gen_pari.gp > roots_pari.txt
setrand(20261115);
default(realprecision, 60);
{
cnt = 0;
for (t = 1, 200,
  if (cnt == 60, break);
  n = 1 + random(12);
  b = 1 + random(40);
  p = sum(i = 0, n, (random(2^(b+1)) - 2^b) * x^i);
  if (poldegree(p) < 1, next);
  if (t % 5 == 0, p = p * (x^2 - 2 - random(28))^2);
  if (t % 7 == 0, p = p * x^2);
  c = Vec(p); c = vector(#c, i, c[#c + 1 - i]);
  ok = 1;
  r = iferr(polrootsreal(p), E, ok = 0; []);
  if (!ok, next);
  cnt++;
  print("p ", strjoin(apply(z -> Str(z), c), " "));
  print("n ", polsturm(p));
  print("r ", strjoin(apply(z -> Str(z), vector(#r, i, if (r[i] == 0, "0", Strprintf("%.40g", r[i])))), " "));
);
}
quit
