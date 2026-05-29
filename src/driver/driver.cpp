#include "driver.h"
#include "ConfigParser.h" 
#include "MFEMHandler.h"
#include "POD_Basis.h"
#include "ModelOrderReduction.h"

#include <iostream>
#include <iomanip> 

namespace mor::driver {

void run_simulation(const std::string& json_path) {
    // =================================================================
    // STAGE 1: CONFIGURATION LOADING, VALIDATION & DEDUCTION 
    // =================================================================
    mor::launcher::SolverConfig cfg = mor::launcher::load_config_from_file(json_path);

    // =================================================================
    // SUMMARY PRINTING 
    // =================================================================
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "               SIMULATION CONFIGURATION UPFRONT SUMMARY   " << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "[CONFIG] DGTD Base Exports Directory: " << cfg.base_exports << std::endl;
    std::cout << "[CONFIG] Training Case (POD Basis Source): " << cfg.train_case << std::endl;
    std::cout << "  -> Training Snapshots from: " << cfg.snapshot_probes << std::endl;
    
    std::cout << "\n[CONFIG] Target Evolution Case (Model to Evolve): " << cfg.evolve_case << std::endl;
    std::cout << "  -> Matrix A path: " << cfg.path_A << std::endl;
    std::cout << "  -> Matrix B path: " << cfg.path_B << std::endl;
    std::cout << "  -> Matrix T path: " << cfg.path_T << std::endl;
    std::cout << "  -> Target Forcing/Inputs from: " << cfg.path_probes << std::endl;
    
    std::cout << "\n[CONFIG] POD Parameters -> Energy Threshold: " 
              << std::setprecision(14) << cfg.energy_threshold 
              << ", Model Order: " 
              << (cfg.model_order == -1 ? "Automatic via Energy" : std::to_string(cfg.model_order)) 
              << std::setprecision(6) << std::endl;
              
    std::cout << "[CONFIG] ROM Parameters -> n_blocks: " << cfg.n_blocks << ", k_svd_max: " << cfg.k_svd_max 
              << ", filter_low_energy: " << (cfg.filter_low_energy ? "true" : "false") << " (" << cfg.filter_min_energy << ")" << std::endl;
    
    std::cout << "\n[CONFIG] Solver Parameters:" << std::endl;
    std::cout << "  -> ROM Export Name: " << (cfg.case_name.empty() ? "[NOT SET - Exports Disabled]" : cfg.case_name) << std::endl;
    std::cout << "  -> Timestep (dt): " << cfg.dt << ", Final Time: " << cfg.final_time << " (" << cfg.total_iterations << " iterations)" << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    MFEMHandler handler;

    // =================================================================
    // STAGE 2: HEAVY COMPUTATIONAL PIPELINE EXECUTION
    // =================================================================

    // --- PHASE 1: TRAINING ---
    std::cout << "[DRIVER] Phase 1: Loading training snapshots matrix..." << std::endl;
    Eigen::MatrixXd train_snapshots = handler.load_snapshots_from_folder(cfg.snapshot_probes.string());  

    std::cout << "[DRIVER] Instantiating POD_Basis processor..." << std::endl;
    POD_Basis pod_processor(train_snapshots, cfg.energy_threshold, cfg.model_order);
    
    std::cout << "[DRIVER] Building Reduced Order Model Basis (Ur)..." << std::endl;
    Eigen::MatrixXd Ur = pod_processor.buildReducedOrderModel(
        cfg.n_blocks, 
        cfg.k_svd_max, 
        cfg.filter_low_energy, 
        cfg.filter_min_energy
    );
    std::cout << "[DRIVER] -> Subspace Matrix Ur generated. Dimensions: " << Ur.rows() << "x" << Ur.cols() << "\n" << std::endl;

    // --- PHASE 2: MODEL LOADING ---
    std::cout << "[DRIVER] Phase 2: Loading full-order operators & target snapshot data..." << std::endl;
    auto A = handler.load_as_sparse_matrix(cfg.path_A.string());
    auto B = handler.load_as_sparse_matrix(cfg.path_B.string());
    auto T = handler.load_as_sparse_matrix(cfg.path_T.string());
    
    Eigen::MatrixXd target_snapshots = handler.load_snapshots_from_folder(cfg.path_probes.string());
    Eigen::MatrixXd target_inputs = handler.load_inputs_from_folder(cfg.path_probes.string());

    // --- PHASE 3: TIME EVOLUTION ---
    std::cout << "[DRIVER] Phase 3: Initializing MOR Simulator..." << std::endl;
    ModelOrderReduction mor_simulator(A, Ur, target_inputs, B, T, cfg.case_name);

    Eigen::VectorXd initialState = target_snapshots.col(0);
    Eigen::VectorXd reducedInitialState = Ur.transpose() * initialState;

    std::cout << "[DRIVER] Executing LSERK4 time integrator..." << std::endl;
    mor_simulator.run_until_ROM(reducedInitialState, cfg.total_iterations, cfg.dt, /*input_start_idx=*/0);

    //  --- PHASE 4: AUTOMATIC POST-PROCESSING (Reconstruction) --- 
    if (!cfg.case_name.empty()) {
        std::cout << "\n[DRIVER] Phase 4: Automatically triggering snapshot reconstruction..." << std::endl;
        mor_simulator.reconstruct_all_states();
    } else {
        std::cout << "\n[DRIVER] Phase 4: Skipping reconstruction (case_name is empty, exports were disabled)." << std::endl;
    }

    std::cout << "\n[DRIVER] Process successfully completed for case: " 
              << (cfg.case_name.empty() ? "[No Name]" : cfg.case_name) << "!" << std::endl;
}

} 