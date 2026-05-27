#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>

class ModelOrderReduction {
public:
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    Eigen::SparseMatrix<double, Eigen::RowMajor> B;
    Eigen::SparseMatrix<double, Eigen::RowMajor> T;
    Eigen::MatrixXd u;
    std::string case_name;
    
    Eigen::MatrixXd Ur;
    Eigen::MatrixXd Ar;
    Eigen::MatrixXd Br;

    static const double A_rk[5];
    static const double B_rk[5];

    ModelOrderReduction(const Eigen::SparseMatrix<double, Eigen::RowMajor>& evolutionOperator,
                        const Eigen::MatrixXd& reducedBasis,
                        const Eigen::MatrixXd& input_snapshots = Eigen::MatrixXd(),
                        const Eigen::SparseMatrix<double, Eigen::RowMajor>& tfsf_operator = Eigen::SparseMatrix<double, Eigen::RowMajor>(),
                        const Eigen::SparseMatrix<double, Eigen::RowMajor>& tfsf_mapping = Eigen::SparseMatrix<double, Eigen::RowMajor>(),
                        const std::string& case_name = "");

    Eigen::VectorXd lserk4_step(const Eigen::VectorXd& x, double dt, const Eigen::VectorXd& u_current);
    Eigen::MatrixXd build_lserk4_operator(const Eigen::MatrixXd& A_mat, double dt);
    Eigen::VectorXd run_until_ROM(const Eigen::VectorXd& reducedInitialState, int finalIteration, double dt, int input_start_idx = 0);
    
    void export_reduced_state(const Eigen::VectorXd& reduced_state, int iteration, double dt);
    std::pair<double, Eigen::VectorXd> read_reduced_state(const std::string& file_path);
    void export_full_state(const Eigen::VectorXd& full_state, int iteration, double absolute_time);
    void reconstruct_all_states();
};