#include "driver.h"
#include "MFEMHandler.h"
#include "POD_Basis.h"
#include "ModelOrderReduction.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <iomanip> // Requerido para std::setprecision

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace mor::driver {

void run_simulation(const std::string& json_path) {
    // =================================================================
    // STAGE 1: CONFIGURATION LOADING, VALIDATION & DEDUCTION
    // =================================================================
    
    // 1. Load and parse JSON configuration file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON configuration file: " + json_path);
    }
    json config;
    file >> config;

    // 2. Validate mandatory outer blocks
    if (!config.contains("training")) {
        throw std::runtime_error("JSON Error: Missing mandatory block 'training'.");
    }
    if (!config.contains("model_to_evolve")) {
        throw std::runtime_error("JSON Error: Missing mandatory block 'model_to_evolve'.");
    }
    if (!config.contains("solver_options")) {
        throw std::runtime_error("JSON Error: Missing mandatory block 'solver_options'.");
    }

    auto train_cfg = config["training"];
    auto evolve_cfg = config["model_to_evolve"];
    auto opts_cfg = config["solver_options"];

    // 3. Validate and Extract Training Configuration
    if (!train_cfg.contains("snapshots_folder")) {
        throw std::runtime_error("JSON Error: Missing mandatory key 'snapshots_folder' inside 'training' block.");
    }
    fs::path base_snapshot_dir = train_cfg["snapshots_folder"].get<std::string>();
    fs::path snapshot_probes = base_snapshot_dir / "MORStateProbes" / "mor_state";

    double energy_threshold = train_cfg.value("energy_threshold", 1.0 - 1e-12);
    int model_order = train_cfg.value("model_order", -1); 
    int n_blocks = train_cfg.value("n_blocks", 1);
    int default_k_svd = (model_order > 0) ? (model_order + 20) : 1000;
    int k_svd_max = train_cfg.value("k_svd_max", default_k_svd);
    bool filter_low_energy = train_cfg.value("filter_low_energy", false);
    double filter_min_energy = train_cfg.value("filter_min_energy", 1e-6);

    // 4. Validate and Extract Target Model Path Deduction Configuration
    if (!evolve_cfg.contains("base_folder")) {
        throw std::runtime_error("JSON Error: Missing mandatory key 'base_folder' inside 'model_to_evolve'.");
    }
    if (!evolve_cfg.contains("case_prefix")) {
        throw std::runtime_error("JSON Error: Missing mandatory key 'case_prefix' inside 'model_to_evolve'.");
    }
    fs::path base_dir = evolve_cfg["base_folder"].get<std::string>();
    std::string prefix = evolve_cfg["case_prefix"].get<std::string>();

    // Deduce file paths cleanly using standard cross-platform filesystem paths
    fs::path path_A = base_dir / (prefix + "_global.csr");
    fs::path path_B = base_dir / (prefix + "_tfsf.csr");
    fs::path path_T = base_dir / (prefix + "_tfsf_mapping.csr");
    fs::path path_probes = base_dir / "MORStateProbes" / "mor_state";

    // 5. Validate and Extract Solver Options (Only dt and final_time are mandatory)
    if (!opts_cfg.contains("dt") || !opts_cfg.contains("final_time")) {
        throw std::runtime_error("JSON Error: Missing mandatory keys ('dt' or 'final_time') inside 'solver_options'.");
    }
    double dt = opts_cfg["dt"].get<double>();
    double final_time = opts_cfg["final_time"].get<double>();
    
    // case_name is now completely optional. If missing, it defaults to an empty string ""
    std::string case_name = opts_cfg.value("case_name", "");
    int total_iterations = static_cast<int>(final_time / dt);

    // 6. Print all deduced info up front (Before execution)
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "               SIMULATION CONFIGURATION UPFRONT SUMMARY   " << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "[CONFIG] Training Snapshots Directory: " << snapshot_probes << std::endl;
    std::cout << "[CONFIG] POD Parameters -> Energy Threshold: " 
              << std::setprecision(14) << energy_threshold 
              << ", Model Order: " 
              << (model_order == -1 ? "Automatic via Energy" : std::to_string(model_order)) 
              << std::setprecision(6) // Opcional: restaurar la precisión por defecto de cout
              << std::endl;

    std::cout << "[CONFIG] ROM Parameters -> n_blocks: " << n_blocks << ", SVD maximum order (k_svd_max): " << k_svd_max 
              << ", filter_low_energy: " << (filter_low_energy ? "true" : "false") << " (" << filter_min_energy << ")" << std::endl;
    
    std::cout << "\n[CONFIG] Target Model Path Deduction:" << std::endl;
    std::cout << "  -> Matrix A path: " << path_A << std::endl;
    std::cout << "  -> Matrix B path: " << path_B << std::endl;
    std::cout << "  -> Matrix T path: " << path_T << std::endl;
    std::cout << "  -> States & Forcing inputs path: " << path_probes << std::endl;
    
    std::cout << "\n[CONFIG] Solver Parameters:" << std::endl;
    std::cout << "  -> Case Name: " << (case_name.empty() ? "[NOT SET - Exports Disabled]" : case_name) << std::endl;
    std::cout << "  -> Timestep (dt): " << dt << ", Final Time: " << final_time << " (" << total_iterations << " iterations)" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    // Initialize the shared data wrapper handler
    MFEMHandler handler;

    // =================================================================
    // STAGE 2: HEAVY COMPUTATIONAL PIPELINE EXECUTION
    // =================================================================

    // --- PHASE 1: TRAINING ---
    std::cout << "[DRIVER] Phase 1: Loading training snapshots matrix..." << std::endl;
    Eigen::MatrixXd train_snapshots = handler.load_snapshots_from_folder(snapshot_probes.string());  

    std::cout << "[DRIVER] Instantiating POD_Basis processor..." << std::endl;
    POD_Basis pod_processor(train_snapshots, energy_threshold, model_order);
    
    std::cout << "[DRIVER] Building Reduced Order Model Basis (Ur)..." << std::endl;
    Eigen::MatrixXd Ur = pod_processor.buildReducedOrderModel(
        n_blocks, 
        k_svd_max, 
        filter_low_energy, 
        filter_min_energy
    );
    std::cout << "[DRIVER] -> Subspace Matrix Ur generated. Dimensions: " << Ur.rows() << "x" << Ur.cols() << "\n" << std::endl;

    // --- PHASE 2: MODEL LOADING ---
    std::cout << "[DRIVER] Phase 2: Loading full-order operators & target snapshot data..." << std::endl;
    auto A = handler.load_as_sparse_matrix(path_A.string());
    auto B = handler.load_as_sparse_matrix(path_B.string());
    auto T = handler.load_as_sparse_matrix(path_T.string());
    
    Eigen::MatrixXd target_snapshots = handler.load_snapshots_from_folder(path_probes.string());
    Eigen::MatrixXd target_inputs = handler.load_inputs_from_folder(path_probes.string());

    // --- PHASE 3: TIME EVOLUTION ---
    std::cout << "[DRIVER] Phase 3: Initializing MOR Simulator..." << std::endl;
    ModelOrderReduction mor_simulator(A, Ur, target_inputs, B, T, case_name);

    // Grab the initial state vector at t=0 (Column 0)
    Eigen::VectorXd initialState = target_snapshots.col(0);
    
    // Project state to reduced coordinate space: x_r(0) = Ur^T * x(0)
    Eigen::VectorXd reducedInitialState = Ur.transpose() * initialState;

    std::cout << "[DRIVER] Executing LSERK4 time integrator..." << std::endl;
    mor_simulator.run_until_ROM(reducedInitialState, total_iterations, dt, /*input_start_idx=*/0);

    //  --- PHASE 4: AUTOMATIC POST-PROCESSING (Reconstruction) --- 
    if (!case_name.empty()) {
        std::cout << "\n[DRIVER] Phase 4: Automatically triggering snapshot reconstruction..." << std::endl;
        mor_simulator.reconstruct_all_states();
    } else {
        std::cout << "\n[DRIVER] Phase 4: Skipping reconstruction (case_name is empty, exports were disabled)." << std::endl;
    }

    std::cout << "\n[DRIVER] Process successfully completed for case: " 
              << (case_name.empty() ? "[No Name]" : case_name) << "!" << std::endl;
}

} 