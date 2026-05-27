#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <string>
#include <vector>

class MFEMHandler {
public:
    std::string filename;
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix;

    MFEMHandler(const std::string& filename = "");
    
    Eigen::SparseMatrix<double, Eigen::RowMajor> load_as_sparse_matrix(const std::string& file = "");
    void print_info() const;
    
    Eigen::MatrixXd load_snapshots_from_folder(const std::string& folder_path, const std::string& prefix = "x");
    Eigen::MatrixXd load_inputs_from_folder(const std::string& folder_path, const std::string& prefix = "u");

private:
    std::vector<std::string> get_sorted_files(const std::string& folder_path, const std::string& prefix);
};