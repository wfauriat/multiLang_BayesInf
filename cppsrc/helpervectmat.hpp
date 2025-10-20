#ifndef HELPERVECTMAT_HPP
#define HELPERVECTMAT_HPP

#include <vector>
#include <limits>
#include <cmath>

namespace vhelp{

double det2x2(const std::vector<std::vector<double>>& mat);

std::vector<std::vector<double>> inv2x2(
    const std::vector<std::vector<double>>& mat);

std::vector<double> vec_mat_mult(
    const std::vector<double>& vec,
    const std::vector<std::vector<double>>& mat);

double vec_vec_mult(const std::vector<double>& vec1,
                    const std::vector<double>& vec2);

}

#endif