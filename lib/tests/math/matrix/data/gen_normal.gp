\\ The Hermite and Smith normal forms by PARI/GP, for gen_normal.py, in FLINT's conventions.
\\
\\ PARI's mathnf is the form of the columns with the pivots at the bottom; FLINT's of the rows,
\\ with the pivots at the top. For B, A with its columns in the opposite order, FLINT's form
\\ without its zero rows is mathnf(B~)~ with its rows and its columns in the opposite order.
\\ matsnf gives the elementary divisors from the largest, padded with zeros in front to the
\\ number of rows: reversed, the first min(m, n) are FLINT's.

normal_rev(A) = my(m = #A~, n = #A); matrix(m, n, i, j, A[m + 1 - i, n + 1 - j]);

normal_hnf(A) = {
  my(m = #A~, n = #A, B, H, r);
  if (m == 0 || n == 0, return(matrix(m, n)));
  B = matrix(m, n, i, j, A[i, n + 1 - j]);
  H = normal_rev(mathnf(B~)~);
  r = #H~;
  matrix(m, n, i, j, if (i <= r, H[i, j], 0));
}

normal_snf(A) = {
  my(m = #A~, n = #A, d);
  if (m == 0 || n == 0, return([]));
  d = Vecrev(matsnf(A));
  vector(min(m, n), i, d[i]);
}

\\ The rows joined by ';', the elements of a row by ','.
normal_text(A) = {
  my(s = "");
  for (i = 1, #A~,
    if (i > 1, s = concat(s, ";"));
    for (j = 1, #A, if (j > 1, s = concat(s, ",")); s = concat(s, Str(A[i, j]))));
  s;
}

normal_vec(v) = {
  my(s = "");
  for (i = 1, #v, if (i > 1, s = concat(s, ",")); s = concat(s, Str(v[i])));
  s;
}

normal_print(A) = {
  print("H ", normal_text(normal_hnf(A)));
  print("S ", normal_vec(normal_snf(A)));
}
