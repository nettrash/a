// Reads a basis in fplll's format ("[[1 2]\n[3 4]]") from stdin, reduces its rows by NTL's LLL_FP
// with deep insertions (delta and deep as arguments; deep = 0 is plain LLL_FP) and writes the
// squared norm of the first nonzero row of the result. A helper of gen_deep_reference.py. Build:
//   c++ -O2 -std=c++17 gen_deep_ntl.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lntl -lgmp -o gen_deep_ntl
#include <NTL/LLL.h>
#include <NTL/mat_ZZ.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace NTL;

int main(int argc, char **argv) {
    double delta = argc > 1 ? std::atof(argv[1]) : 0.99;
    long deep = argc > 2 ? std::atol(argv[2]) : 0;
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(std::cin, line)) {
        for (char &c : line)
            if (c == '[' || c == ']') c = ' ';
        std::istringstream in(line);
        std::vector<std::string> row;
        std::string x;
        while (in >> x) row.push_back(x);
        if (!row.empty()) rows.push_back(row);
    }
    mat_ZZ B;
    B.SetDims(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); i++)
        for (size_t j = 0; j < rows[i].size(); j++) B[i][j] = conv<ZZ>(rows[i][j].c_str());
    LLL_FP(B, delta, deep);
    for (long i = 0; i < B.NumRows(); i++) {
        ZZ s;
        InnerProduct(s, B[i], B[i]);
        if (!IsZero(s)) {
            std::cout << s << std::endl;
            return 0;
        }
    }
    std::cout << 0 << std::endl;
}
