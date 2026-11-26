// A second reference for gf2_reference.txt: NTL's mat_GF2 recomputes what it offers (products,
// ranks, determinants, inverses, kernel dimensions) for every line and reports disagreements.
//   c++ -O2 -std=c++17 gen_gf2_ntl.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lntl -lgmp \
//     -o gen_gf2_ntl && ./gen_gf2_ntl < gf2_reference.txt
#include <NTL/mat_GF2.h>

#include <iostream>
#include <sstream>
#include <string>

#include "gen_gf2.h"

static NTL::mat_GF2 to_ntl(const bm &a) {
  NTL::mat_GF2 m;
  m.SetDims(a.rows, a.cols);
  for (int i = 0; i < a.rows; ++i)
    for (int j = 0; j < a.cols; ++j)
      if (bm_get(&a, i, j)) m.put(i, j, 1);
  return m;
}

static uint64_t hash_ntl(const NTL::mat_GF2 &m, int rows, int cols) {
  bm a = bm_new(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      if (NTL::IsOne(m.get(i, j))) bm_flip(&a, i, j);
  uint64_t h = bm_hash(&a);
  free(a.w);
  return h;
}

static long rank_of(const NTL::mat_GF2 &m) {
  NTL::mat_GF2 c = m;
  return NTL::gauss(c);
}

static uint64_t hex(const std::string &s) { return std::stoull(s, nullptr, 16); }

int main() {
  std::string line;
  long lines = 0, checks = 0, bad = 0;
  auto check = [&](bool ok, const std::string &what) {
    ++checks;
    if (!ok) {
      ++bad;
      std::cerr << "disagree (" << what << "): " << line << "\n";
    }
  };
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string op;
    in >> op;
    ++lines;
    if (op == "mul") {
      std::string ka, kb, h;
      int m, l, n;
      unsigned long long sa, sb;
      in >> ka >> m >> l >> sa >> kb >> n >> sb >> h;
      bm a = gen(ka.c_str(), m, l, sa), b = gen(kb.c_str(), l, n, sb);
      NTL::mat_GF2 c;
      NTL::mul(c, to_ntl(a), to_ntl(b));
      check(hash_ntl(c, m, n) == hex(h), "product");
      free(a.w);
      free(b.w);
    } else if (op == "ech") {
      std::string k, hr, hn;
      int m, n;
      unsigned long long s;
      long rank;
      in >> k >> m >> n >> s >> rank >> hr >> hn;
      bm a = gen(k.c_str(), m, n, s);
      NTL::mat_GF2 x = to_ntl(a);
      check(rank_of(x) == rank, "rank");
      // The right kernel of A is the left kernel of Aᵀ: n − rank vectors.
      NTL::mat_GF2 t, kern;
      NTL::transpose(t, x);
      NTL::kernel(kern, t);
      check(kern.NumRows() == n - rank, "kernel dimension");
      free(a.w);
    } else if (op == "inv") {
      std::string k, h;
      int n;
      unsigned long long s;
      in >> k >> n >> s >> h;
      bm a = gen(k.c_str(), n, n, s);
      NTL::mat_GF2 x = to_ntl(a), y;
      NTL::GF2 d;
      if (n == 0) {
        d = 1;
      } else {
        NTL::inv(d, y, x);
      }
      check(NTL::IsOne(d) == (h != "singular"), "determinant");
      if (n > 0 && NTL::IsOne(d)) check(hash_ntl(y, n, n) == hex(h), "inverse");
      free(a.w);
    } else if (op == "solve") {
      std::string ka, kb, h;
      int m, n, k;
      unsigned long long sa, sb;
      long ra, rab;
      in >> ka >> m >> n >> sa >> kb >> k >> sb >> ra >> rab >> h;
      bm a = gen(ka.c_str(), m, n, sa), b;
      if (kb == "image") {
        bm x0 = gen("dense", n, k, sb);
        b = bm_naive_mul(&a, &x0);
        free(x0.w);
      } else {
        b = gen(kb.c_str(), m, k, sb);
      }
      NTL::mat_GF2 x = to_ntl(a), y = to_ntl(b), ab;
      ab.SetDims(m, n + k);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n + k; ++j) ab.put(i, j, j < n ? x.get(i, j) : y.get(i, j - n));
      check(rank_of(x) == ra, "rank of A");
      check(rank_of(ab) == rab, "rank of [A | B]");
      free(a.w);
      free(b.w);
    }
  }
  std::cout << lines << " lines, " << checks << " checks, " << bad << " disagreements\n";
  return bad != 0;
}
