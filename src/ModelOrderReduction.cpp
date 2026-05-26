#include "ModelOrderReduction.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <algorithm>

namespace fs = std::filesystem;

const double ModelOrderReduction::A_rk[5] = {0.0, -0.417890474499852, -1.19215169464268, -1.69778469247153, -1.51418344425716};
const double ModelOrderReduction::B_rk[5] = {0.149659021999229, 0.379210312999627, 0.822955029386982, 0.699450455949122, 0.153057247968152};

ModelOrderReduction::ModelOrderReduction(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& evolutionOperator,
    const Eigen::MatrixXd& reducedBasis, const Eigen::MatrixXd& input_snapshots,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& tfsf_operator,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& tfsf_mapping, const std::string& case_name)
    : A(evolutionOperator), Ur(reducedBasis), u(input_snapshots), B(tfsf_operator), T(tfsf_mapping), case_name(case_name) 
{
    Ar = Ur.transpose() * (A * Ur);
    if (B.rows() > 0 && T.rows() > 0) {
        Br = Ur.transpose() * (B * T);
    }
}

Eigen::VectorXd ModelOrderReduction::lserk4_step(const Eigen::VectorXd& x, double dt, const Eigen::VectorXd& u_current) {
    Eigen::VectorXd y = x;
    Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
    for (int s = 0; s < 5; ++s) {
        Eigen::VectorXd rhs = Ar * y;
        if (Br.rows() > 0 && u_current.size() > 0) rhs += Br * u_current;
        res = A_rk[s] * res + dt * rhs;
        y = y + B_rk[s] * res;
    }
    return y;
}

Eigen::MatrixXd ModelOrderReduction::build_lserk4_operator(const Eigen::MatrixXd& A_mat, double dt) {
    int n = A_mat.rows();
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n, n);
    for (int s = 0; s < 5; ++s) {
        res = A_rk[s] * res + dt * (A_mat * R);
        R = R + B_rk[s] * res;
    }
    return R;
}

Eigen::VectorXd ModelOrderReduction::run_until_ROM(const Eigen::VectorXd& reducedInitialState, int finalIteration, double dt, int input_start_idx) {
    Eigen::VectorXd reducedState = reducedInitialState;
    bool use_forcing = (u.size() > 0 && B.rows() > 0 && T.rows() > 0);

    export_reduced_state(reducedState, 0, dt);

    if (!use_forcing) {
        Eigen::MatrixXd R = build_lserk4_operator(Ar, dt);
        for (int i = 0; i < finalIteration; ++i) {
            reducedState = R * reducedState;
            export_reduced_state(reducedState, i + 1, dt);
        }
    } else {
        int num_inputs = u.cols();
        for (int i = 0; i < finalIteration; ++i) {
            int idx_n = (input_start_idx + i) % num_inputs;
            Eigen::VectorXd u_n = u.col(idx_n);
            reducedState = lserk4_step(reducedState, dt, u_n);
            export_reduced_state(reducedState, i + 1, dt);
        }
    }
    return reducedState;
}

void ModelOrderReduction::export_reduced_state(const Eigen::VectorXd& reduced_state, int iteration, double dt) {
    if (case_name.empty()) return;
    fs::path export_dir = fs::path("Exports") / case_name / "xr";
    fs::create_directories(export_dir);
    std::ofstream f(export_dir / ("xr_" + std::to_string(iteration) + ".txt"));
    f << (iteration * dt) << "\n" << reduced_state.size() << "\n";
    for (int i = 0; i < reduced_state.size(); ++i) f << reduced_state[i] << "\n";
}

std::pair<double, Eigen::VectorXd> ModelOrderReduction::read_reduced_state(const std::string& file_path) {
    std::ifstream f(file_path);
    if (!f.is_open()) throw std::runtime_error("Fallo al leer estado reducido: " + file_path);
    double time; int size;
    f >> time >> size;
    Eigen::VectorXd data(size);
    for (int i = 0; i < size; ++i) f >> data[i]; // Nota: En Python leías complex, pero como guardas double, los leemos nativamente como double.
    return {time, data};
}

void ModelOrderReduction::export_full_state(const Eigen::VectorXd& full_state, int iteration, double absolute_time) {
    if (case_name.empty()) return;
    fs::path export_dir = fs::path("Exports") / case_name / "xfull";
    fs::create_directories(export_dir);
    std::ofstream f(export_dir / ("x_" + std::to_string(iteration)));
    f << absolute_time << "\n" << full_state.size() << "\n";
    for (int i = 0; i < full_state.size(); ++i) f << full_state[i] << "\n";
}

void ModelOrderReduction::reconstruct_all_states(const Eigen::MatrixXd& Ur_mat) {
    if (case_name.empty()) throw std::uint56_t(1); // Requiere case_name
    fs::path xr_dir = fs::path("Exports") / case_name / "xr";
    if (!fs::exists(xr_dir)) throw std::runtime_error("El directorio xr no existe");

    std::vector<std::pair<int, std::string>> files;
    std::regex pattern("xr_(\\d+)\\.txt");
    for (const auto& entry : fs::directory_iterator(xr_dir)) {
        std::string name = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(name, match, pattern)) {
            files.push_back({std::stoi(match[1].str()), entry.path().string()});
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& pair : files) {
        int iteration = pair.first;
        auto [absolute_time, reduced_state] = read_reduced_state(pair.second);
        Eigen::VectorXd full_state = Ur_mat * reduced_state;
        export_full_state(full_state, iteration, absolute_time);
    }
}