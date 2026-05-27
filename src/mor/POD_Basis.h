#pragma once
#include <Eigen/Dense>
#include <vector>

class POD_Basis {
public:
    Eigen::MatrixXd snapshots;
    double energyThreshold;
    int model_order;

    POD_Basis(const Eigen::MatrixXd& snapshots, double energyThreshold = 1.0 - 1e-12, int model_order = -1);
    
    Eigen::MatrixXd buildReducedOrderModel(int n_blocks = 1, int k_svd_max = 1000, 
                                           bool filter_low_energy = false, double filter_min_energy = 1e-6);

private:
    int energyCriterionForTruncation(const Eigen::VectorXd& singularValues);
    
    Eigen::MatrixXd filter_low_energy_snapshots(const Eigen::MatrixXd& mats, double threshold);
    
    std::vector<std::pair<int, int>> divide_into_blocks(int n_snapshots, int n_blocks);
    
    void run_randomized_svd(const Eigen::MatrixXd& A, int n_components, int n_iter, int n_oversamples,
                            Eigen::MatrixXd& U, Eigen::VectorXd& S);
                            
    void incremental_pod_step(const Eigen::MatrixXd& U_current, const Eigen::VectorXd& S_current, 
                              const Eigen::MatrixXd& new_snapshots, int k_svd_max,
                              Eigen::MatrixXd& U_updated, Eigen::VectorXd& S_updated);
};