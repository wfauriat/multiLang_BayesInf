#ifndef LOGFCTS_HPP
#define LOGFCTS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace logfct {

std::vector<std::vector<double>> params_to_cov(double sigma_x, double sigma_y, 
    double rho);

double logLikelihood(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    bool verbose=false);

double logLikelihoodVect(
    const Eigen::MatrixXd& data,
    const Eigen::VectorXd& mu,
    const Eigen::MatrixXd& sigma,
    bool verbose=false);

Eigen::MatrixXd vectorToMatrix(const std::vector<std::vector<double>>& vect);

double log_prior(const std::vector<std::vector<double>>& bounds,
                 const std::vector<double>& point);

}

#endif