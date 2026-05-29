#include "ConfigParser.h"
#include <fstream>
#include <stdexcept>

namespace mor::launcher {

SolverConfig parse_and_validate_config(const nlohmann::json& config) {
    if (!config.contains("exports_dir")) {
        throw std::runtime_error("JSON Error: Missing mandatory global key 'exports_dir'.");
    }
    if (!config.contains("training")) {
        throw std::runtime_error("JSON Error: Missing mandatory block 'training'.");
    }
    if (!config.contains("model_to_evolve")) {
        throw std::runtime_error("JSON Error: Missing mandatory block 'model_to_evolve'.");
    }
    if (!config.contains("solver_options")) {
        throw std::runtime_error("JSON Error: Missing mandatory block 'solver_options'.");
    }

    SolverConfig cfg;
    cfg.base_exports = config["exports_dir"].get<std::string>();
    
    auto train_cfg  = config["training"];
    auto evolve_cfg = config["model_to_evolve"];
    auto opts_cfg   = config["solver_options"];

    if (!train_cfg.contains("case_name")) {
        throw std::runtime_error("JSON Error: Missing 'case_name' inside 'training' block.");
    }
    cfg.train_case = train_cfg["case_name"].get<std::string>();
    cfg.snapshot_probes = cfg.base_exports / "single-core" / cfg.train_case / "MORStateProbes" / "MORState";

    cfg.energy_threshold = train_cfg.value("energy_threshold", 1.0 - 1e-12);
    cfg.model_order      = train_cfg.value("model_order", -1); 
    cfg.n_blocks         = train_cfg.value("n_blocks", 1);
    
    int default_k_svd    = (cfg.model_order > 0) ? (cfg.model_order + 20) : 1000;
    cfg.k_svd_max        = train_cfg.value("k_svd_max", default_k_svd);
    cfg.filter_low_energy   = train_cfg.value("filter_low_energy", false);
    cfg.filter_min_energy  = train_cfg.value("filter_min_energy", 1e-6);

    if (!evolve_cfg.contains("case_name")) {
        throw std::runtime_error("JSON Error: Missing 'case_name' inside 'model_to_evolve' block.");
    }
    cfg.evolve_case = evolve_cfg["case_name"].get<std::string>();
    
    cfg.path_A = cfg.base_exports / "Operators" / cfg.evolve_case / (cfg.evolve_case + "_global.csr");
    cfg.path_B = cfg.base_exports / "Operators" / cfg.evolve_case / (cfg.evolve_case + "_tfsf.csr");
    cfg.path_T = cfg.base_exports / "Operators" / cfg.evolve_case / (cfg.evolve_case + "_tfsf_mapping.csr");
    cfg.path_probes = cfg.base_exports / "single-core" / cfg.evolve_case / "MORStateProbes" / "MORState";

    if (!opts_cfg.contains("dt") || !opts_cfg.contains("final_time")) {
        throw std::runtime_error("JSON Error: Missing mandatory keys ('dt' or 'final_time') inside 'solver_options'.");
    }
    cfg.dt = opts_cfg["dt"].get<double>();
    cfg.final_time = opts_cfg["final_time"].get<double>();
    cfg.case_name = opts_cfg.value("case_name", ""); 
    cfg.total_iterations = static_cast<int>(cfg.final_time / cfg.dt);

    return cfg;
}

SolverConfig load_config_from_file(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON configuration file: " + json_path);
    }
    nlohmann::json config;
    file >> config;
    return parse_and_validate_config(config);
}

} 