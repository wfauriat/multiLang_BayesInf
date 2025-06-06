#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <limits>

#include "helpervectmat.hpp"
#include "logfcts.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<double> modeltrue(const std::vector<std::vector<double>>& x,
                const std::vector<double>& b);

std::vector<double> modelfit(const std::vector<std::vector<double>>& x,
                const std::vector<double>& b);

Eigen::VectorXd s2e(const std::vector<double>& x);

std::vector<double> rnv(const std::vector<double>& x,
                const std::vector<double>& prop_s);

double rnlv(double m, double s);

double invGaussLogLike(double x, double mu,
     double loc = 0.0, double scale = 1.0);

const auto covDiagMat = [](const double& stdV, int size){
    Eigen::MatrixXd cov(size, size);
    for (int i=0; i<size; ++i) {
        cov(i,i) = stdV;
    }
    return cov;
};


int main(int argc, char* argv[]) {

    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(123);
    std::normal_distribution<> d0(0.0, 1.0);
    std::uniform_real_distribution<> du(0.0, 1.0);

    //////////////////// DEFINITION OF APPLICATION / CALIBRATION CASE
    std::vector<double> b0 {2.0, -1.0, 2.0};
    std::vector<double> btest {1.5, -0.7, 1.5};
    double nslvl {0.2};
    double smod {0.2};
    std::vector<double> sinvgauss {0.4, 0.2};

    std::vector<double> xmes {0.0, 0.5, 1.0, 2.0, 2.5,
                              2.8, 4.0, 4.4, 5.2, 5.5};
    std::vector<std::vector<double>> xxmes(xmes.size(), {0.0, 0.0});
       for (int i=0; i<xmes.size(); ++i) {
        xxmes[i][0] = xmes[i];
        xxmes[i][1] = d0(gen);
    }
    std::vector<double> ymes = modeltrue(xxmes, b0);
        for (int i=0; i<xmes.size(); ++i) {
        ymes[i] += d0(gen) * nslvl;
    }
    std::vector<double> ytheo = modelfit(xxmes, b0);

    // auto covMat = covDiagMat(smod, xmes.size());
    // Eigen::MatrixXd covMat(xmes.size(), xmes.size());
    // for (int i=0; i<xmes.size(); ++i) {
    //     covMat(i,i) = nslvl;
    // }

    std::cout << "XMES :" << std::endl;
    for (const auto& t : xxmes) {std::cout << t[0] << " ";} 
    std::cout << std::endl;
    std::cout << "YMES :" << std::endl;
    for (const auto& t : ymes) {std::cout << t << " ";} 
    std::cout << std::endl;

    //////////////////// DEFENITION OF MCMC INFERENCE PARAMETERS
    int num_iterations = 22000;
    int burn_in = 2000;
    int thin_factor = 20;

    std::vector<std::vector<double>> bounds(3,{-5.0,5.0});
    std::vector<double> proposal_sd {0.2, 0.2, 0.05};
    double proposal_sinvg {0.05};

    std::vector<std::vector<double>> MCchain(num_iterations,
                                     std::vector<double>(
                                        bounds[0].size(), 0.0));
    std::vector<double> Schain(num_iterations, 0.0);
    std::vector<double> llchain(num_iterations, 0.0);
    
    MCchain[0] = btest;
    Schain[0] = sinvgauss[0] + sinvgauss[1];
    auto covMat = covDiagMat(Schain[0], xmes.size());
    llchain[0] = logfct::logLikelihood(s2e(ymes),
                    s2e(modelfit(xxmes, MCchain[0])), covMat);

    double llold = llchain[0];
    double lpold = logfct::log_prior(bounds, MCchain[0]);
    double lsold = invGaussLogLike(Schain[0], sinvgauss[0], sinvgauss[1]);  

    std::vector<double> xprop;
    double sp;
    double llprop, lpprop, lspp, ldiff;

    std::cout << "Starting MCMC..." << std::endl;
    for (int k=0;k<xprop.size();++k) {std::cout << MCchain[0][k] << " ";}
    std::cout << std::endl;

    for (int i=1; i<num_iterations; ++i)
    {
        //////////////////// GENERATION OF PROPOSAL
        xprop = rnv(MCchain[i-1], proposal_sd);
        sp = rnlv(Schain[i-1], proposal_sinvg);
        covMat = covDiagMat(sp, xmes.size());   
        llprop = logfct::logLikelihood(s2e(ymes),
                            s2e(modelfit(xxmes, xprop)), covMat);
        lpprop = logfct::log_prior(bounds, xprop);
        lspp = invGaussLogLike(sp, sinvgauss[0], sinvgauss[1]);

        ldiff = llprop + lpprop + lspp - llold - lpold - lsold;

        if (ldiff > std::log(du(gen))) {
            MCchain[i] = xprop;
            llchain[i] = llprop;
            Schain[i] = sp;
            llold = llprop;
            lpold = lpprop;
            lsold = lspp;
        }
        else {
            MCchain[i] = MCchain[i-1];
            Schain[i] = Schain[i-1];
            llchain[i] = llchain[i-1];
        }

        if ((i + 1) % (num_iterations / 10) == 0) {
            std::cout << "Iteration " << i + 1 <<
                        "/" << num_iterations << 
                        " (Current: b0=" << MCchain[i][0] << ", " <<
                        " b1=" << MCchain[i][1] << ", " <<
                        " b2=" << MCchain[i][2] <<
                        " s=" << Schain[i] << ")" << std::endl;
        }

        // if (i%thin_factor == 0) {
        //     std::cout << "llprop : " << llprop << " " <<
        //                  "llold : " << llold << " " <<
        //                 "ldiff : " << ldiff << " " << "sample : ";
        //     for (int k=0;k<xprop.size();++k) {std::cout << MCchain[i][k] << " ";}
        //     std::cout << std::endl;
        // }
    }

    // std::cout << "Inv gauss valid :" << std::endl;
    // for (int i=0;i<10;++i) {std::cout << rnlv(0.6, 0.05) << " ";} 
    // std::cout << std::endl;
    // std::cout << invGaussLogLike(0.6, 0.4, 0.2) << std::endl;

    /////////////////// DATA OUTPUT
    std::string filenamedata = "data.txt";
    std::string filenameLL = "LL.txt";
    std::ofstream outputFiledata(filenamedata, std::ios::out | std::ios::trunc);
    std::ofstream outputFileLL(filenameLL, std::ios::out | std::ios::trunc);

    for (int i=0; i < num_iterations; ++i) {
        if (i%thin_factor == 0 && i > burn_in) {
        outputFiledata << MCchain[i][0] << "," <<
                          MCchain[i][1] << "," << 
                          MCchain[i][2] << "," << 
                          Schain[i] << std::endl;
        outputFileLL << llchain[i] << std::endl;
        }
    }
    outputFiledata.close();
    outputFileLL.close();
}


/////////////////////////////// FUNCTIONS DEFINITION ////////////////
///////////////////////////////

std::vector<double> modeltrue(const std::vector<std::vector<double>>& x,
                const std::vector<double>& b)
{
    std::vector<double> result;
    for (const auto& t : x) {
        result.push_back(b[0] + b[1]*t[0] + b[2]*t[0]*t[0]+ 0.01*t[1]);
    }
    return result;
}

std::vector<double> modelfit(const std::vector<std::vector<double>>& x,
                const std::vector<double>& b)
{
    std::vector<double> result;
    for (const auto& t : x) {
        result.push_back(b[0] + b[1]*t[0] + b[2]*t[0]*t[0]);
    }
    return result;
}

Eigen::VectorXd s2e(const std::vector<double>& x) 
{
    Eigen::VectorXd eigV(x.size());
        for (int i=0; i<x.size();++i) {eigV(i)=x[i];}
        return eigV;
}

std::vector<double> rnv(const std::vector<double>& x,
                const std::vector<double>& prop_s){
    std::vector<double> prop(prop_s.size());
    std::normal_distribution<> d0(0.0, 1.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int k=0; k<prop_s.size(); ++k) {
        prop[k] = x[k] + d0(gen) * prop_s[k];
    }
    return prop;
};


double rnlv(double m, double s) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> d0(0.0, 1.0);
    double z = d0(gen);
    return std::exp(z * s + std::log(m));
}

double invGaussLogLike(double x, double mu,
     double loc, double scale) {
    if (mu <= 0 || scale <= 0 || (x - loc) <= 0) {
        return -std::numeric_limits<double>::infinity();
    }

    double y = (x - loc) / scale;

    double term1 = -0.5 * std::log(2.0 * M_PI);
    double term2 = -1.5 * std::log(y);
    double term3_numerator = std::pow(y - mu, 2);
    double term3_denominator = 2.0 * std::pow(mu, 2) * y;
    double term3 = -term3_numerator / term3_denominator;

    // Adjust for scale: log(f(y) / scale) = log(f(y)) - log(scale)
    double log_pdf_standardized = term1 + term2 + term3;
    double log_pdf_scaled = log_pdf_standardized - std::log(scale);

    return log_pdf_scaled;
}

//////////////////// PARSING INPUT ARGS
// int num_data_points;
// if (argc > 1) {
//     try {
//         num_data_points = std::stoi(argv[1]);
//         if (num_data_points <= 0) {
//             std::cerr << 
//             "Error: Number of data points must be positive. Using default: " << 
//             100 << std::endl;
//             num_data_points = 100;
//         }
//     } catch (const std::invalid_argument& e) {
//         std::cerr << "Error: Invalid argument for number of data points. Using default: " << 
//         100 << std::endl;
//         num_data_points = 100;
//     } catch (const std::out_of_range& e) {
//         std::cerr << "Error: Number of data points out of range. Using default: " << 
//         100 << std::endl;
//         num_data_points = 100;
//     }
// } else { 
//     num_data_points = 100;
// }
