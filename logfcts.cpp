#include "logfcts.hpp"
#include "helpervectmat.hpp"


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
    Eigen::MatrixXd datav(vect.size(), 2);
    for (int i = 0; i < vect.size(); ++i) {
        for (int j = 0; j < 2; ++j) {
            datav(i, j) = vect[i][j];
        }
    }
    return datav;
}

double calculateSingleMultivariateGaussianLogLikelihood(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma)
{
    int D = x.rows();
    Eigen::VectorXd diff = x - mu;
    
    Eigen::LLT<Eigen::MatrixXd> lltOfSigma(sigma);
    if (lltOfSigma.info() != Eigen::Success) {
        std::cerr <<
         "Error: Covariance matrix is not positive definite or singular." <<
        std::endl;
        return -std::numeric_limits<double>::infinity();
    }
    double mahalanobis_squared = diff.transpose() * lltOfSigma.solve(diff);
    double log_det_sigma =
     2.0 * lltOfSigma.matrixLLT().diagonal().array().log().sum();
    double log_likelihood = -0.5 * (D * std::log(2.0 * M_PI) +
     log_det_sigma + mahalanobis_squared);

    return log_likelihood;
}

double calculateDatasetMultivariateGaussianLogLikelihood(
    const Eigen::MatrixXd& data,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma)
{
    int N = data.rows();
    int D = data.cols();

    if (N == 0) {
        return 0.0;
    }

    Eigen::LLT<Eigen::MatrixXd> lltOfSigma(sigma);
    if (lltOfSigma.info() != Eigen::Success) {
        std::cerr << 
        "Error: Covariance matrix is not positive definite or singular." << 
        std::endl;
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


double log_likelihood(const std::vector<std::vector<double>>& data,
                    double mu_x, double mu_y,
                    const std::vector<std::vector<double>>& cov) 
{
    int N = data.size();
    double determinant = vhelp::det2x2(cov);

    if (determinant <= 0) { // Covariance must be positive definite
        return -std::numeric_limits<double>::infinity();
    }

    double log_det_cov = std::log(determinant);
    std::vector<std::vector<double>> inv_cov = vhelp::inv2x2(cov);

    double sum_mahalanobis_dist_sq = 0.0;
    std::vector<double> mu = {mu_x, mu_y};

    for (const auto& point : data) {
        std::vector<double> diff = {point[0] - mu[0], point[1] - mu[1]};
        std::vector<double> temp = vhelp::vec_mat_mult(diff, inv_cov);
        sum_mahalanobis_dist_sq += vhelp::vec_vec_mult(temp, diff);
    }

    double log_lik = -0.5 * (N * (2 * std::log(2 * M_PI) 
                    + log_det_cov) 
                    + sum_mahalanobis_dist_sq);
    return log_lik;
}


double log_prior(double mu_x, double mu_y,
                 double sigma_x, double sigma_y,
                 double rho) 
{
    if (sigma_x <= 0 || sigma_y <= 0 || rho <= -1 || rho >= 1) {
        return -std::numeric_limits<double>::infinity();
    }
    return 0.0;
}


}