#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <stdexcept>

#include "helpervectmat.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


std::vector<std::vector<double>> params_to_cov(double sigma_x, double sigma_y, 
    double rho);

double log_likelihood(const std::vector<std::vector<double>>& data,
                    double mu_x, double mu_y,
                    const std::vector<std::vector<double>>& cov);

double log_prior(double mu_x, double mu_y,
                 double sigma_x, double sigma_y,
                 double rho);


int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d0(0.0, 1.0);
    std::uniform_real_distribution<> du(0.0, 1.0);

    double mu_1 = 3;
    double mu_2 = 1;
    double s_1 = 0.5;
    double s_2 = 0.1;
    double rho = 0.6;

    std::vector<double> mu_vec = {mu_1, mu_2};
    std::vector<std::vector<double>> true_cov = params_to_cov(
        s_1, s_2, rho);

    // Cholesky decomposition for sampling 
    // from a multivariate normal
    // L L^T = Cov
    // For 2x2: L = | l11 0  |
    //              | l21 l22|
    double l11 = std::sqrt(true_cov[0][0]);
    double l21 = true_cov[1][0] / l11;
    double l22 = std::sqrt(true_cov[1][1] - l21 * l21);

    int num_data_points;
    if (argc > 1) {
        try {
            num_data_points = std::stoi(argv[1]);
            if (num_data_points <= 0) {
                std::cerr << 
                "Error: Number of data points must be positive. Using default: " << 
                100 << std::endl;
                num_data_points = 100;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid argument for number of data points. Using default: " << 
            100 << std::endl;
            num_data_points = 100;
        } catch (const std::out_of_range& e) {
            std::cerr << "Error: Number of data points out of range. Using default: " << 
            100 << std::endl;
            num_data_points = 100;
        }
    } else { 
        num_data_points = 100;
    }
    std::vector<std::vector<double>> data(num_data_points,
                                             std::vector<double>(2));
    
    std::cout << "Generation of synthetic data :" << std::endl;

    for (int i=0; i < num_data_points; ++i){
        double z1 = d0(gen);
        double z2 = d0(gen);
        // x = mu + L * z
        data[i][0] = mu_vec[0] + l11 * z1;
        data[i][1] = mu_vec[1] + l21 * z1 + l22 * z2;
    };

    std::cout << "Visualisation of data sample :" << std::endl;

    for (int j=0; j < 2; ++j) {
        for (int i=0; i < 10; ++i) {
            std::cout << data[i][j] << " ";
        };
        std::cout << std::endl;
    };

    int num_iterations = 20000;
    int burn_in = 2000;
    int thin_factor = 10;

    double current_mu_1 = 0.0, current_mu_2 = 0.0;
    double current_s1 = 1.0, current_s2 = 1.0;
    double current_rho = 0.0;

    double proposal_sd_mu = 0.05;
    double proposal_sd_sigma = 0.05;
    double proposal_sd_rho = 0.02;

    std::vector<std::vector<double>> posterior_samples;
    std::cout << "Starting MCMC..." << std::endl;

    for (int i = 0; i < num_iterations; ++i) {
        double proposed_mu_1 = current_mu_1 + d0(gen) * proposal_sd_mu;
        double proposed_mu_2 = current_mu_2 + d0(gen) * proposal_sd_mu;
        double proposed_s1 = current_s1 + d0(gen) * proposal_sd_sigma;
        double proposed_s2 = current_s2 + d0(gen) * proposal_sd_sigma;
        double proposed_rho = current_rho + d0(gen) * proposal_sd_rho;

        std::vector<std::vector<double>> current_cov;
        std::vector<std::vector<double>> proposed_cov;
        double current_log_post;
        double proposed_log_post;
        double acceptance_ratio;

        current_cov = params_to_cov(current_s1, current_s2,
                                 current_rho);
        current_log_post = log_likelihood(data,
                            current_mu_1, current_mu_2,
                            current_cov) 
                            + log_prior(current_mu_1, current_mu_2,
                            current_s1, current_s2, current_rho);
        proposed_cov = params_to_cov(
            proposed_s1, proposed_s2, 
            proposed_rho);
        proposed_log_post = log_likelihood(data,
                     proposed_mu_1, proposed_mu_2, proposed_cov) 
                     + log_prior(proposed_mu_1, proposed_mu_2,
                     proposed_s1, proposed_s2, proposed_rho);

        acceptance_ratio = std::exp(proposed_log_post - current_log_post);

        if (du(gen) < acceptance_ratio) {
            current_mu_1 = proposed_mu_1;
            current_mu_2 = proposed_mu_2;
            current_s1 = proposed_s1;
            current_s2 = proposed_s2;
            current_rho = proposed_rho;
        }

        if (i >= burn_in && (i - burn_in) % thin_factor == 0) {
            posterior_samples.push_back(
                {current_mu_1, current_mu_2,
                current_s1, current_s2,
                current_rho});
        }

        if ((i + 1) % (num_iterations / 10) == 0) {
            std::cout << "Iteration " << i + 1 <<
                        "/" << num_iterations <<
                        " (Current: mu=[" <<
                        current_mu_1 << ", " << current_mu_2 <<
                        "], sigma_x=" << current_s1 << 
                        ", sigma_y=" << current_s2 << 
                        ", rho=" << current_rho << ")" <<
                        std::endl;
        }
    }
    std::cout << "MCMC finished." << std::endl;

    std::cout << "Visualisation of chain end :" << std::endl;

    int chainN = posterior_samples.size();

    for (int j=0; j < 2; ++j) {
        for (int i=chainN - 10; i < chainN; ++i) {
            std::cout << posterior_samples[i][j] << " ";
        };
        std::cout << std::endl;
    };

    double MAP_mu1 = 0, MAP_mu2 = 0, MAP_s1 = 0, MAP_s2 = 0, MAP_rho = 0;
    for (const auto& point : posterior_samples){
        MAP_mu1 += point[0];
        MAP_mu2 += point[1];
        MAP_s1 += point[2];
        MAP_s2 += point[3];
        MAP_rho += point[4];
    }
    MAP_mu1 /= posterior_samples.size();
    MAP_mu2 /= posterior_samples.size();
    MAP_s1 /= posterior_samples.size();
    MAP_s2 /= posterior_samples.size();
    MAP_rho /= posterior_samples.size();

    std::cout << "Data size : " << num_data_points << std::endl;

    std::cout << "MAP results :" << std::endl;
    std::cout << "MAP mu_1 : " << MAP_mu1 << std::endl;
    std::cout << "MAP mu_2 : " << MAP_mu2 << std::endl;
    std::cout << "MAP s_1 : " << MAP_s1 << std::endl;
    std::cout << "MAP s_2 : " << MAP_s2 << std::endl;
    std::cout << "MAP rho : " << MAP_rho << std::endl;

}



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
    // Flat priors with bounds
    if (sigma_x <= 0 || sigma_y <= 0 || rho <= -1 || rho >= 1) {
        return -std::numeric_limits<double>::infinity(); // Invalid parameters
    }
    // You can add more specific bounds if desired, e.g.,
    // if (std::abs(mu_x) > 100 || std::abs(mu_y) > 100 || sigma_x > 10 || sigma_y > 10) {
    //     return -std::numeric_limits<double>::infinity();
    // }
    return 0.0; // Flat prior within valid bounds
}