#pragma once

#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace mor::launcher {

struct SolverConfig {
    std::filesystem::path base_exports;
    
    std::string train_case;
    std::filesystem::path snapshot_probes;
    double energy_threshold;
    int model_order;
    int n_blocks;
    int k_svd_max;
    bool filter_low_energy;
    double filter_min_energy;
    
    std::string evolve_case;
    std::filesystem::path path_A;
    std::filesystem::path path_B;
    std::filesystem::path path_T;
    std::filesystem::path path_probes;
    
    double dt;
    double final_time;
    std::string case_name; 
    int total_iterations;
};

SolverConfig parse_and_validate_config(const nlohmann::json& config);
SolverConfig load_config_from_file(const std::string& json_path);

} 