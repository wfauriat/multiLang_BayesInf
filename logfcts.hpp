#ifndef LOGFCTS_HPP
#define LOGFCTS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace logfct {

std::vector<std::vector<double>> params_to_cov(double sigma_x, double sigma_y, 
    double rho);

double calculateSingleMultivariateGaussianLogLikelihood(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    bool verbose=false);

double calculateDatasetMultivariateGaussianLogLikelihood(
    const Eigen::MatrixXd& data,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    bool verbose=false);

Eigen::MatrixXd vectorToMatrix(const std::vector<std::vector<double>>& vect);


double log_likelihood(const std::vector<std::vector<double>>& data,
                    double mu_x, double mu_y,
                    const std::vector<std::vector<double>>& cov);

double log_prior(double mu_x, double mu_y,
                 double sigma_x, double sigma_y,
                 double rho);

}

#endif