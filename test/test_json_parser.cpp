#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "ConfigParser.h" // Acceso directo a tu módulo aislado

using json = nlohmann::json;

TEST(ConfigParserTest, SuccessfulParsingAndPathDeduction) {
    std::string raw_json = R"({
        "exports_dir": "external/dgtd/Exports",
  
        "training": {
            "case_name": "test_training",
            "energy_threshold": 0.99,
            "model_order": 200,
            "n_blocks": 5,
            "k_svd_max": 500,
            "filter_low_energy": true,
            "filter_min_energy": 1e-5
        },
        
        "model_to_evolve": {
            "case_name": "test_evolve"
        },
        
        "solver_options": {
            "dt": 0.025,
            "final_time": 10.0,
            "case_name": "test_Output"
      } 
    })";

    json j = json::parse(raw_json);
    mor::launcher::SolverConfig cfg = mor::launcher::parse_and_validate_config(j);

    EXPECT_EQ(cfg.base_exports.string(), "external/dgtd/Exports");
    EXPECT_EQ(cfg.train_case, "test_training");
    EXPECT_DOUBLE_EQ(cfg.energy_threshold, 0.99);
    EXPECT_EQ(cfg.model_order, 200);
    EXPECT_EQ(cfg.n_blocks, 5);
    EXPECT_EQ(cfg.k_svd_max, 500);
    EXPECT_TRUE(cfg.filter_low_energy);
    EXPECT_DOUBLE_EQ(cfg.filter_min_energy, 1e-5);
    EXPECT_EQ(cfg.evolve_case, "test_evolve");
    EXPECT_DOUBLE_EQ(cfg.dt, 0.025);
    EXPECT_DOUBLE_EQ(cfg.final_time, 10.0);
    EXPECT_EQ(cfg.case_name, "test_Output");
    EXPECT_EQ(cfg.total_iterations, 400);

    EXPECT_EQ(cfg.snapshot_probes.string(), "external/dgtd/Exports/single-core/test_training/MORStateProbes/MORState");
    EXPECT_EQ(cfg.path_A.string(), "external/dgtd/Exports/Operators/test_evolve/test_evolve_global.csr");
    EXPECT_EQ(cfg.path_B.string(), "external/dgtd/Exports/Operators/test_evolve/test_evolve_tfsf.csr");
    EXPECT_EQ(cfg.path_T.string(), "external/dgtd/Exports/Operators/test_evolve/test_evolve_tfsf_mapping.csr");
    EXPECT_EQ(cfg.path_probes.string(), "external/dgtd/Exports/single-core/test_evolve/MORStateProbes/MORState");
}

TEST(ConfigParserTest, FallbacksToDefaultsAndOptionals) {
    std::string raw_json = R"({
        "exports_dir": "external/dgtd/Exports",
        "training": { "case_name": "2D_TFSF_MOR_training" },
        "model_to_evolve": { "case_name": "2D_TFSF_MOR_evolve" },
        "solver_options": { "dt": 0.05, "final_time": 5.0 }
    })";

    json j = json::parse(raw_json);
    mor::launcher::SolverConfig cfg = mor::launcher::parse_and_validate_config(j);

    EXPECT_DOUBLE_EQ(cfg.energy_threshold, 1.0 - 1e-12);
    EXPECT_EQ(cfg.model_order, -1);
    EXPECT_EQ(cfg.n_blocks, 1);
    EXPECT_EQ(cfg.k_svd_max, 1000); // Al ser model_order = -1, default_k_svd debe ser 1000
    EXPECT_FALSE(cfg.filter_low_energy);
    EXPECT_DOUBLE_EQ(cfg.filter_min_energy, 1e-6);
    
    EXPECT_EQ(cfg.case_name, "");
    EXPECT_EQ(cfg.total_iterations, 100);
}

TEST(ConfigParserTest, DynamicKsvdMaxDeduction) {
    std::string raw_json = R"({
        "exports_dir": "external/dgtd/Exports",
        "training": { "case_name": "training", "model_order": 40 },
        "model_to_evolve": { "case_name": "evolve" },
        "solver_options": { "dt": 0.1, "final_time": 1.0 }
    })";

    json j = json::parse(raw_json);
    mor::launcher::SolverConfig cfg = mor::launcher::parse_and_validate_config(j);

    // Si model_order > 0, k_svd_max por defecto es model_order + 20 
    EXPECT_EQ(cfg.model_order, 40);
    EXPECT_EQ(cfg.k_svd_max, 60);
}


TEST(ConfigExceptionTest, MissingExportsDirThrows) {
    json j = R"({
        "training": {"case_name": "a"}, "model_to_evolve": {"case_name": "b"}, "solver_options": {"dt": 0.1, "final_time": 1}
    })"_json;
    EXPECT_THROW(mor::launcher::parse_and_validate_config(j), std::runtime_error);
}

TEST(ConfigExceptionTest, MissingTrainingBlockThrows) {
    json j = R"({
        "exports_dir": "path", "model_to_evolve": {"case_name": "b"}, "solver_options": {"dt": 0.1, "final_time": 1}
    })"_json;
    EXPECT_THROW(mor::launcher::parse_and_validate_config(j), std::runtime_error);
}

TEST(ConfigExceptionTest, MissingTrainingCaseNameThrows) {
    json j = R"({
        "exports_dir": "path", "training": {}, "model_to_evolve": {"case_name": "b"}, "solver_options": {"dt": 0.1, "final_time": 1}
    })"_json;
    EXPECT_THROW(mor::launcher::parse_and_validate_config(j), std::runtime_error);
}

TEST(ConfigExceptionTest, MissingModelToEvolveCaseNameThrows) {
    json j = R"({
        "exports_dir": "path", "training": {"case_name": "a"}, "model_to_evolve": {}, "solver_options": {"dt": 0.1, "final_time": 1}
    })"_json;
    EXPECT_THROW(mor::launcher::parse_and_validate_config(j), std::runtime_error);
}

TEST(ConfigExceptionTest, MissingSolverOptionsParametersThrows) {
    json j = R"({
        "exports_dir": "path", "training": {"case_name": "a"}, "model_to_evolve": {"case_name": "b"}, "solver_options": {"final_time": 1}
    })"_json; 
    EXPECT_THROW(mor::launcher::parse_and_validate_config(j), std::runtime_error);
}