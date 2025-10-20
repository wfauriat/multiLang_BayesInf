#include "helpervectmat.hpp"


namespace vhelp{

double det2x2(const std::vector<std::vector<double>>& mat) 
{
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
}

std::vector<std::vector<double>> inv2x2(
    const std::vector<std::vector<double>>& mat) 
{
    double determinant = det2x2(mat);
    if (determinant == 0.0) {
        // Handle singular matrix: return values that lead to -infinity log-likelihood.
        return {{std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()},
                {std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()}};
    }

    double inv_det = 1.0 / determinant;
    std::vector<std::vector<double>> inv_mat(2, std::vector<double>(2));
    inv_mat[0][0] = mat[1][1] * inv_det;
    inv_mat[0][1] = -mat[0][1] * inv_det;
    inv_mat[1][0] = -mat[1][0] * inv_det;
    inv_mat[1][1] = mat[0][0] * inv_det;
    return inv_mat;
}

std::vector<double> vec_mat_mult(
    const std::vector<double>& vec, 
    const std::vector<std::vector<double>>& mat) 
{
    std::vector<double> result(2);
    result[0] = vec[0] * mat[0][0] + vec[1] * mat[1][0];
    result[1] = vec[0] * mat[0][1] + vec[1] * mat[1][1];
    return result;
}

double vec_vec_mult(
    const std::vector<double>& vec1, 
    const std::vector<double>& vec2) 
{
    return vec1[0] * vec2[0] + vec1[1] * vec2[1];
}

}