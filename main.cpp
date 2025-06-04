#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

#include "helpervectmat.hpp"
#include "logfcts.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


int main(int argc, char* argv[]) {

    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(123);
    std::normal_distribution<> d0(0.0, 1.0);
    std::uniform_real_distribution<> du(0.0, 1.0);

    //////////////////// DEFINITION OF TRUE PARAMETERS
    double mu_1 = 3, mu_2 = 1, s_1 = 0.5, s_2 = 0.1, rho = 0.9;

    std::vector<double> mu_vec = {mu_1, mu_2};
    std::vector<std::vector<double>> true_cov = logfct::params_to_cov(
        s_1, s_2, rho);

    double l11 = std::sqrt(true_cov[0][0]);
    double l21 = true_cov[1][0] / l11;
    double l22 = std::sqrt(true_cov[1][1] - l21 * l21);

    //////////////////// PARSING INPUT ARGS
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

    //////////////////// GENERATE SAMPLE DATA
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

    //////////////////// DEFENITION OF MCMC INFERENCE PARAMETERS
    int num_iterations = 100000;
    int burn_in = 2000;
    int thin_factor = 10;

    double proposal_sd_mu = 0.05;
    double proposal_sd_sigma = 0.05;
    double proposal_sd_rho = 0.02;

    //////////////////// DEFINITION OF START POINT
    double current_mu_1 = 0.0, current_mu_2 = 0.0;
    double current_s1 = 1.0, current_s2 = 1.0;
    double current_rho = 0.0;

    std::vector<std::vector<double>> posterior_samples;
    std::cout << "Starting MCMC..." << std::endl;

    for (int i = 0; i < num_iterations; ++i) 
    {
        //////////////////// GENERATION OF PROPOSAL
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
        Eigen::VectorXd curr_mu(2);
        Eigen::MatrixXd curr_cov(2, 2);
        Eigen::VectorXd prop_mu(2);
        Eigen::MatrixXd prop_cov(2, 2);
        Eigen::MatrixXd datav(num_data_points, 2);

        current_cov = logfct::params_to_cov(current_s1, current_s2,
                                 current_rho);
        proposed_cov = logfct::params_to_cov(proposed_s1, proposed_s2, 
        proposed_rho);
        curr_mu << current_mu_1, current_mu_2;
        curr_cov << current_cov[0][0], current_cov[0][1],
                    current_cov[1][0], current_cov[1][1];
        prop_mu << proposed_mu_1, proposed_mu_2;
        prop_cov << proposed_cov[0][0], proposed_cov[0][1],
            proposed_cov[1][0], proposed_cov[1][1];
        datav = logfct::vectorToMatrix(data);
        

        //////////////////// COMPUTATION OF LIKELIHOODS
        current_log_post = 
            logfct::calculateDatasetMultivariateGaussianLogLikelihood(
                datav, curr_mu, curr_cov);
        current_log_post += logfct::log_prior(current_mu_1, current_mu_2,
                    current_s1, current_s2, current_rho);
        proposed_log_post = 
            logfct::calculateDatasetMultivariateGaussianLogLikelihood(
                datav, prop_mu, prop_cov);
        proposed_log_post += logfct::log_prior(proposed_mu_1, proposed_mu_2,
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
        //////////////////// VISUALISATION OF INFERENCE ADVANCE
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

    //////////////////// POST TRAITMENT OF MCMC CHAIN
    std::cout << "MCMC finished." << std::endl;

    // std::cout << "Visualisation of chain end :" << std::endl;
    // int chainN = posterior_samples.size();
    // for (int j=0; j < 5; ++j) {
    //     for (int i=chainN - 10; i < chainN; ++i) {
    //         std::cout << posterior_samples[i][j] << " ";
    //     };
    //     std::cout << std::endl;
    // };

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

    //////////////////// DISPLAY OF RESULTS
    std::cout << "Data size : " << num_data_points << std::endl;
     std::cout << "MAP mu_1 : " << MAP_mu1 << std::endl;
    std::cout << "MAP mu_2 : " << MAP_mu2 << std::endl;
    std::cout << "MAP s_1 : " << MAP_s1 << std::endl;
    std::cout << "MAP s_2 : " << MAP_s2 << std::endl;
    std::cout << "MAP rho : " << MAP_rho << std::endl;


    ///////////////////// DATA OUTPUT
    std::string filenamedata = "data.txt";
    std::string filenameMAP = "MAP.txt";
    std::ofstream outputFiledata(filenamedata, std::ios::out | std::ios::trunc);
    std::ofstream outputFileMAP(filenameMAP, std::ios::out | std::ios::trunc);

    for (int i=0; i < data.size(); ++i) {
        outputFiledata << data[i][0] << "," << data[i][1] << std::endl;
    }
    outputFileMAP << MAP_mu1 << "," << MAP_mu2 << "," << 
                     MAP_s1  << "," << MAP_s2  << "," << MAP_rho << std::endl;


    outputFiledata.close();
    outputFileMAP.close();
}