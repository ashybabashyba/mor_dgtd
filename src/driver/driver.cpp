#include "driver.h"
#include "MFEMHandler.h"
#include "POD_Basis.h"
#include "ModelOrderReduction.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace mor::driver {

void run_simulation(const std::string& json_path) {
    // 1. Load and parse JSON configuration file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON configuration file: " + json_path);
    }
    json config;
    file >> config;

    MFEMHandler handler;

    // =================================================================
    // PHASE 1: TRAINING (Pure Ur generation via POD_Basis)
    // =================================================================
    std::cout << "[DRIVER] Phase 1: Loading training snapshots..." << std::endl;
    
    if (!config.contains("training") || !config["training"].contains("snapshots_folder")) {
        throw std::runtime_error("Missing mandatory key 'snapshots_folder' inside 'training' block.");
    }
    
    auto train_cfg = config["training"];

    fs::path base_snapshot_dir = train_cfg["snapshots_folder"].get<std::string>();
    fs::path snapshot_probes = base_snapshot_dir / "MORStateProbes" / "mor_state";

    std::cout << "  -> Looking for training Snapshots at: " << snapshot_probes << std::endl;

    // Load the snapshot matrix required for training via the handler
    Eigen::MatrixXd train_snapshots = handler.load_snapshots_from_folder(snapshot_probes.string());  

    // --- Dynamic optional arguments for POD_Basis constructor ---
    // Using json::value(key, default) to mimic Python's dict.get(key, default)
    double energy_threshold = train_cfg.value("energy_threshold", 1.0 - 1e-12);
    
    // Assuming -1 or 0 means "not set / automatic via energy" in your C++ class logic
    int model_order = train_cfg.value("model_order", -1); 

    std::cout << "[DRIVER] Initializing POD_Basis with parameters -> Energy Threshold: " 
              << energy_threshold << ", Model Order: " << model_order << std::endl;
              
    // Instantiate POD_Basis using the exact required constructor signature
    POD_Basis pod_processor(train_snapshots, energy_threshold, model_order);
    
    // --- Dynamic optional arguments for buildReducedOrderModel ---
    int n_blocks = train_cfg.value("n_blocks", 1);
    
    // If k_svd_max is not provided, we dynamically set a safe default based on model_order
    int default_k_svd = (model_order > 0) ? (model_order + 20) : 1000;
    int k_svd_max = train_cfg.value("k_svd_max", default_k_svd);
    
    bool filter_low_energy = train_cfg.value("filter_low_energy", false);
    double filter_min_energy = train_cfg.value("filter_min_energy", 1e-6);

    std::cout << "[DRIVER] Building ROM with parameters -> n_blocks: " << n_blocks 
              << ", k_svd_max: " << k_svd_max 
              << ", filter_low_energy: " << (filter_low_energy ? "true" : "false") 
              << " (" << filter_min_energy << ")" << std::endl;

    // Call the builder passing all resolved (user-defined or default) arguments.
    // Adjust the parameter names below to match your actual C++ buildReducedOrderModel signature.
    Eigen::MatrixXd Ur = pod_processor.buildReducedOrderModel(
        n_blocks, 
        k_svd_max, 
        filter_low_energy, 
        filter_min_energy
    );
    
    std::cout << "[DRIVER] -> Reduced Basis Ur successfully generated. Dim: " 
              << Ur.rows() << "x" << Ur.cols() << "\n" << std::endl;


    // =================================================================
    // PHASE 2: AUTOMATIC PATH DEDUCTION (Target Model to Evolve)
    // =================================================================
    std::cout << "[DRIVER] Phase 2: Deducing target model paths using convention..." << std::endl;
    auto evolve_cfg = config["model_to_evolve"];
    
    fs::path base_dir = evolve_cfg["base_folder"].get<std::string>();
    std::string prefix = evolve_cfg["case_prefix"].get<std::string>();

    // Reconstruct exact file paths following the MFEM data structure convention
    fs::path path_A = base_dir / (prefix + "_global.csr");
    fs::path path_B = base_dir / (prefix + "_tfsf.csr");
    fs::path path_T = base_dir / (prefix + "_tfsf_mapping.csr");
    fs::path path_probes = base_dir / "MORStateProbes" / "mor_state";

    std::cout << "  -> Looking for Matrix A at: " << path_A << std::endl;
    std::cout << "  -> Looking for Matrix B at: " << path_B << std::endl;
    std::cout << "  -> Looking for Matrix T at: " << path_T << std::endl;
    std::cout << "  -> Looking for Snapshots/Inputs at: " << path_probes << std::endl;

    // Load full-order operators as sparse matrices via the MFEM handler
    auto A = handler.load_as_sparse_matrix(path_A.string());
    auto B = handler.load_as_sparse_matrix(path_B.string());
    auto T = handler.load_as_sparse_matrix(path_T.string());
    
    // Load target states and boundary forcing inputs from the probes folder
    Eigen::MatrixXd target_snapshots = handler.load_snapshots_from_folder(path_probes.string());
    Eigen::MatrixXd target_inputs = handler.load_inputs_from_folder(path_probes.string());


    // =================================================================
    // PHASE 3: TIME EVOLUTION (Model Order Reduction Simulation)
    // =================================================================
    auto opts_cfg = config["solver_options"];
    double dt = opts_cfg["dt"].get<double>();
    double final_time = opts_cfg["final_time"].get<double>();
    std::string case_name = opts_cfg["case_name"].get<std::string>();

    std::cout << "\n[DRIVER] Phase 3: Initializing MOR for time evolution..." << std::endl;
    
    // Setup the MOR simulator. The projection Ar = Ur^T * A * Ur is performed internally.
    ModelOrderReduction mor_simulator(A, Ur, target_inputs, B, T, case_name);

    // Initial state in the full-order space (Column 0 of the target snapshots)
    Eigen::VectorXd initialState = target_snapshots.col(0);
    
    // Project initial state onto the reduced subspace: x_r(0) = Ur^T * x(0)
    Eigen::VectorXd reducedInitialState = Ur.transpose() * initialState;

    int total_iterations = static_cast<int>(final_time / dt);
    
    std::cout << "[DRIVER] Executing LSERK4 time integrator..." << std::endl;
    mor_simulator.run_until_ROM(reducedInitialState, total_iterations, dt, /*input_start_idx=*/0);

    std::cout << "[DRIVER] Process successfully completed for case: " << case_name << "!" << std::endl;
}

} // namespace mor::driver