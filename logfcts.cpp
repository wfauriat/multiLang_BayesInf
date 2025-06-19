#include "logfcts.hpp"
#include "helpervectmat.hpp"
// #include <tuple>

namespace logfct {

std::vector<std::vector<double>> params_to_cov(
    double sigma_x, double sigma_y, double rho) 
{
    std::vector<std::vector<double>> cov(2, std::vector<double>(2));
    cov[0][0] = sigma_x * sigma_x;
    cov[1][1] = sigma_y * sigma_y;
    cov[0][1] = rho * sigma_x * sigma_y;
    cov[1][0] = rho * sigma_x * sigma_y;
    return cov;
}

Eigen::MatrixXd vectorToMatrix(const std::vector<std::vector<double>>& vect)
{
    Eigen::MatrixXd datav(vect.size(), vect[0].size());
    for (int i = 0; i < vect.size(); ++i) {
        for (int j = 0; j < vect[0].size(); ++j) {
            datav(i, j) = vect[i][j];
        }
    }
    return datav;
}

double logLikelihood(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    bool verbose)
{
    int D = x.rows();
    Eigen::VectorXd diff = x - mu;
    
    Eigen::LLT<Eigen::MatrixXd> lltOfSigma(sigma);
    if (lltOfSigma.info() != Eigen::Success) {
        if (verbose != false) {
            std::cerr <<
            "Error: Covariance matrix is not positive definite or singular." <<
            std::endl;
        }
        return -std::numeric_limits<double>::infinity();
    }
    double mahalanobis_squared = diff.transpose() * lltOfSigma.solve(diff);
    double log_det_sigma =
     2.0 * lltOfSigma.matrixLLT().diagonal().array().log().sum();
    double log_likelihood = -0.5 * (D * std::log(2.0 * M_PI) +
     log_det_sigma + mahalanobis_squared);

    return log_likelihood;
}

double logLikelihoodVect(
    const Eigen::MatrixXd& data,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    bool verbose)
{
    int N = data.rows();
    // int N = 1;
    int D = data.cols();

    if (N == 0) {
        return 0.0;
    }

    Eigen::LLT<Eigen::MatrixXd> lltOfSigma(sigma);
    if (lltOfSigma.info() != Eigen::Success) {
        if (verbose != false) {
            std::cerr <<
            "Error: Covariance matrix is not positive definite or singular." <<
            std::endl;
        }
        return -std::numeric_limits<double>::infinity();
    }
    double log_det_sigma = 
        2.0 * lltOfSigma.matrixLLT().diagonal().array().log().sum();
    
    double total_log_likelihood = 0.0;
    total_log_likelihood -= (double)N / 2.0 * (D * std::log(2.0 * M_PI) 
                                                + log_det_sigma);

    for (int i = 0; i < N; ++i) {
        Eigen::VectorXd diff = data.row(i).transpose() - mu; 
        double mahalanobis_squared = diff.transpose() * lltOfSigma.solve(diff);
        total_log_likelihood -= 0.5 * mahalanobis_squared;
    }

    return total_log_likelihood;
}

double log_prior(const std::vector<std::vector<double>>& bounds,
                 const std::vector<double>& point)
{
    bool inside = true;
    auto bound_it = bounds.begin();
    auto point_it = point.begin();
    for (;bound_it != bounds.end() && point_it != point.end();
     ++bound_it, ++point_it){
        if (*point_it <= (*bound_it)[0] || *point_it >= (*bound_it)[1])
        {
            inside=false;
        } 
    }
    return inside ? 0 : -std::numeric_limits<double>::infinity();
}


}