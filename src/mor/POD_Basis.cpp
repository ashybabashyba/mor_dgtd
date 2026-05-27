#include "POD_Basis.h"
#include <iostream>
#include <random>
#include <algorithm>

POD_Basis::POD_Basis(const Eigen::MatrixXd& snapshots, double energyThreshold, int model_order)
    : snapshots(snapshots), energyThreshold(energyThreshold), model_order(model_order) {}

int POD_Basis::energyCriterionForTruncation(const Eigen::VectorXd& singularValues) {
    Eigen::VectorXd sv_squared = singularValues.array().square(); // order=2 por defecto
    double total_energy = sv_squared.sum();
    double cumulative = 0.0;
    for (int i = 0; i < sv_squared.size(); ++i) {
        cumulative += sv_squared[i];
        if ((cumulative / total_energy) >= energyThreshold) return i + 1;
    }
    return singularValues.size();
}

Eigen::MatrixXd POD_Basis::filter_low_energy_snapshots(const Eigen::MatrixXd& mats, double threshold) {
    std::vector<int> keep_indices;
    for (int j = 0; j < mats.cols(); ++j) {
        if (mats.col(j).norm() >= threshold) keep_indices.push_back(j);
    }
    Eigen::MatrixXd filtered(mats.rows(), keep_indices.size());
    for (size_t i = 0; i < keep_indices.size(); ++i) {
        filtered.col(i) = mats.col(keep_indices[i]);
    }
    std::cout << "Filtered snapshots: " << mats.cols() << " -> " << filtered.cols() << std::endl;
    return filtered;
}

std::vector<std::pair<int, int>> POD_Basis::divide_into_blocks(int n_snapshots, int n_blocks) {
    int block_size = n_snapshots / n_blocks;
    std::vector<std::pair<int, int>> indices;
    for (int i = 0; i < n_blocks - 1; ++i) {
        indices.push_back({i * block_size, (i + 1) * block_size});
    }
    indices.push_back({(n_blocks - 1) * block_size, n_snapshots});
    return indices;
}

void POD_Basis::run_randomized_svd(const Eigen::MatrixXd& A, int n_components, int n_iter, int n_oversamples,
                                   Eigen::MatrixXd& U, Eigen::VectorXd& S) {
    int m = A.rows(); int n = A.cols();
    int k = std::min({n_components + n_oversamples, m, n});

    std::mt19937 gen(42); // Fixed seed 42 to match sklearn
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Omega(n, k);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < n; ++i) Omega(i, j) = dist(gen);

    std::cout << "[POD_Basis] Projecting snapshot matrix onto random subspace..." << std::endl;
    Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(A * Omega).householderQ() * Eigen::MatrixXd::Identity(m, k);

    // Safeguard: if n_iter is less than 20, set interval to 1 to print every iteration.
    // This prevents a division by zero error (i.e., n_iter / 20 = 0).
    int log_interval = (n_iter >= 20) ? (n_iter / 20) : 1;

    std::cout << "[POD_Basis] Starting power iterations (Total iterations: " << n_iter << ")..." << std::endl;
    for (int i = 0; i < n_iter; ++i) {
        Eigen::MatrixXd Q_z = Eigen::HouseholderQR<Eigen::MatrixXd>(A.transpose() * Q).householderQ() * Eigen::MatrixXd::Identity(n, k);
        Q = Eigen::HouseholderQR<Eigen::MatrixXd>(A * Q_z).householderQ() * Eigen::MatrixXd::Identity(m, k);

        // Check if we should log progress at this iteration step
        if ((i + 1) % log_interval == 0) {
            std::cout << "  -> Power iteration progress: " << (i + 1) << " / " << n_iter << std::endl;
        }
    }

    std::cout << "[POD_Basis] Projecting back and computing final BDCSVD on the small subspace..." << std::endl;
    Eigen::MatrixXd B = Q.transpose() * A;
    Eigen::BDCSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU);
    U = Q * svd.matrixU().leftCols(n_components);
    S = svd.singularValues().head(n_components);
}

void POD_Basis::incremental_pod_step(const Eigen::MatrixXd& U_current, const Eigen::VectorXd& S_current, 
                                      const Eigen::MatrixXd& new_snapshots, int k_svd_max,
                                      Eigen::MatrixXd& U_updated, Eigen::VectorXd& S_updated) {
    Eigen::MatrixXd C = U_current.transpose() * new_snapshots;
    Eigen::MatrixXd D = new_snapshots - (U_current * C);
    
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(D);
    Eigen::MatrixXd Q_new = qr.householderQ() * Eigen::MatrixXd::Identity(D.rows(), std::min(D.rows(), D.cols()));
    Eigen::MatrixXd R_new = Q_new.transpose() * D;

    Eigen::MatrixXd small_matrix = Eigen::MatrixXd::Zero(S_current.size() + R_new.rows(), S_current.size() + C.cols());
    small_matrix.topLeftCorner(S_current.size(), S_current.size()) = S_current.asDiagonal();
    small_matrix.topRightCorner(S_current.size(), C.cols()) = C;
    small_matrix.bottomRightCorner(R_new.rows(), C.cols()) = R_new;

    int k_components = std::min((int)small_matrix.rows(), (int)small_matrix.cols()) - 1;
    k_components = std::min(k_components, k_svd_max);

    Eigen::MatrixXd U_small;
    run_randomized_svd(small_matrix, k_components, 20, 10, U_small, S_updated);

    Eigen::MatrixXd U_combined(U_current.rows(), U_current.cols() + Q_new.cols());
    U_combined << U_current, Q_new;
    U_updated = U_combined * U_small;
}

Eigen::MatrixXd POD_Basis::buildReducedOrderModel(int n_blocks, int k_svd_max, bool filter_low_energy, double filter_min_energy) {
    Eigen::MatrixXd working_snapshots = snapshots;
    if (filter_low_energy) working_snapshots = filter_low_energy_snapshots(working_snapshots, filter_min_energy);

    int n_snapshots = working_snapshots.cols();
    int k_max_possible = std::min((int)working_snapshots.rows(), n_snapshots) - 1;
    int k_effective = std::min(k_svd_max, k_max_possible);

    Eigen::MatrixXd U; Eigen::VectorXd S;

    if (n_blocks == 1) {
        run_randomized_svd(working_snapshots, k_effective, 20, 10, U, S);
    } else {
        std::cout << "Building ROM incrementally with " << n_blocks << " blocks..." << std::endl;
        auto block_indices = divide_into_blocks(n_snapshots, n_blocks);
        
        for (int b = 0; b < n_blocks; ++b) {
            int start = block_indices[b].first; int end = block_indices[b].second;
            Eigen::MatrixXd block_snapshots = working_snapshots.block(0, start, working_snapshots.rows(), end - start);
            
            if (b == 0) {
                int k_block = std::min(k_effective, (int)block_snapshots.cols() - 1);
                run_randomized_svd(block_snapshots, k_block, 20, 10, U, S);
            } else {
                Eigen::MatrixXd U_next; Eigen::VectorXd S_next;
                incremental_pod_step(U, S, block_snapshots, k_effective, U_next, S_next);
                U = U_next; S = S_next;
            }
        }
    }

    int r = model_order;
    if (r <= 0) {
        r = energyCriterionForTruncation(S);
        std::cout << "Truncation rank (Energy threshold): " << r << std::endl;
    } else {
        std::cout << "Using fixed model_order: " << r << std::endl;
    }
    return U.leftCols(r);
}