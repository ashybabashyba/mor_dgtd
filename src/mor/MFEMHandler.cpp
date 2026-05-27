#include "MFEMHandler.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <regex>
#include <algorithm>

namespace fs = std::filesystem;

MFEMHandler::MFEMHandler(const std::string& filename) : filename(filename) {}

Eigen::SparseMatrix<double, Eigen::RowMajor> MFEMHandler::load_as_sparse_matrix(const std::string& file) {
    std::string target = file.empty() ? filename : file;
    std::ifstream f(target);
    if (!f.is_open()) throw std::runtime_error("No se pudo abrir el archivo CSR: " + target);

    int height, width;
    f >> height >> width;

    std::vector<int> indptr(height + 1);
    for (int i = 0; i <= height; ++i) f >> indptr[i];

    int nnz = indptr.back();
    std::vector<int> indices(nnz);
    for (int i = 0; i < nnz; ++i) f >> indices[i];

    std::vector<double> data(nnz);
    for (int i = 0; i < nnz; ++i) f >> data[i];

    matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>(height, width);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);
    
    for (int r = 0; r < height; ++r) {
        for (int idx = indptr[r]; idx < indptr[r+1]; ++idx) {
            triplets.push_back(Eigen::Triplet<double>(r, indices[idx], data[idx]));
        }
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}

std::vector<std::string> MFEMHandler::get_sorted_files(const std::string& folder_path, const std::string& prefix) {
    std::vector<std::pair<int, std::string>> indexed_files;
    std::regex pattern(prefix + "_(\\d+)(?:\\.txt)?"); // Captura x_1 o x_1.txt

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(filename, match, pattern)) {
            int idx = std::stoi(match[1].str());
            indexed_files.push_back({idx, entry.path().string()});
        }
    }
    
    if (indexed_files.empty()) throw std::runtime_error("No se encontraron archivos con el prefijo: " + prefix);
    std::sort(indexed_files.begin(), indexed_files.end());
    
    std::vector<std::string> sorted_paths;
    for (const auto& p : indexed_files) sorted_paths.push_back(p.second);
    return sorted_paths;
}

Eigen::MatrixXd MFEMHandler::load_snapshots_from_folder(const std::string& folder_path, const std::string& prefix) {
    auto files = get_sorted_files(folder_path, prefix);
    
    std::ifstream first_file(files[0]);
    std::string dummy;
    int N;
    std::getline(first_file, dummy); // Saltar marca de tiempo/iteración
    first_file >> N;

    int num_snapshots = files.size();
    Eigen::MatrixXd snapshots(N, num_snapshots);

    for (int idx = 0; idx < num_snapshots; ++idx) {
        std::ifstream f(files[idx]);
        std::getline(f, dummy); f >> dummy; // Saltar las primeras dos líneas de cabecera
        for (int i = 0; i < N; ++i) {
            f >> snapshots(i, idx);
        }
    }
    return snapshots;
}

Eigen::MatrixXd MFEMHandler::load_inputs_from_folder(const std::string& folder_path, const std::string& prefix) {
    return -1.0 * load_snapshots_from_folder(folder_path, prefix); // Inversión del signo (u_*)
}