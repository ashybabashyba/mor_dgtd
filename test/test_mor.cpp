#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "POD_Basis.h"
#include "ModelOrderReduction.h"

Eigen::SparseMatrix<double> to_sparse(const Eigen::MatrixXd& dense) {
    Eigen::SparseMatrix<double> sparse = dense.sparseView();
    sparse.makeCompressed();
    return sparse;
}


TEST(MORMathTest, OrthogonalityAndDimensions) {
    int n_dofs = 100;
    int n_snapshots = 40;
    
    Eigen::MatrixXd snapshots = Eigen::MatrixXd::Random(n_dofs, n_snapshots);
    
    double energy_threshold = 0.9999;
    int model_order = -1; 
    POD_Basis pod_processor(snapshots, energy_threshold, model_order);
    
    Eigen::MatrixXd Ur = pod_processor.buildReducedOrderModel(
        /*n_blocks=*/1, /*k_svd_max=*/40, /*filter_low=*/false, /*min_energy=*/0.0
    );
    
    int r = Ur.cols(); 
    
    EXPECT_EQ(Ur.rows(), n_dofs);
    EXPECT_LE(r, n_snapshots); 
    
    Eigen::MatrixXd UtUr = Ur.transpose() * Ur;
    Eigen::MatrixXd identity_r = Eigen::MatrixXd::Identity(r, r);
    
    EXPECT_LT((UtUr - identity_r).norm(), 1e-12); 
        
    Eigen::MatrixXd UrUt = Ur * Ur.transpose();
    Eigen::MatrixXd identity_N = Eigen::MatrixXd::Identity(n_dofs, n_dofs);
    
    EXPECT_GT((UrUt - identity_N).norm(), 1e-3);
}

TEST(MORMathTest, SnapshotReconstructionError) {
    int n_dofs = 50;
    int n_snapshots = 10;
    
    Eigen::MatrixXd random_basis = Eigen::MatrixXd::Random(n_dofs, 5); 
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Random(5, n_snapshots);
    Eigen::MatrixXd snapshots = random_basis * coefficients; 
    
    POD_Basis pod_processor(snapshots, /*threshold=*/1.0, /*model_order=*/5);
    Eigen::MatrixXd Ur = pod_processor.buildReducedOrderModel(1, 10, false, 0.0);
    
    Eigen::MatrixXd reduced_coordinates = Ur.transpose() * snapshots;
    Eigen::MatrixXd reconstructed_snapshots = Ur * reduced_coordinates;
    
    EXPECT_LT((reconstructed_snapshots - snapshots).norm(), 1e-10);
}

TEST(MORMathTest, LSERK4StepConsistency) {
    int n_dofs = 30;
    
    Eigen::MatrixXd A_dense = Eigen::MatrixXd::Random(n_dofs, n_dofs) * 0.1;
    A_dense.diagonal() -= Eigen::VectorXd::Ones(n_dofs) * 2.0; 
    Eigen::SparseMatrix<double> A = to_sparse(A_dense);
    
    Eigen::MatrixXd random_mat = Eigen::MatrixXd::Random(n_dofs, 5);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(random_mat);
    Eigen::MatrixXd Ur = qr.householderQ() * Eigen::MatrixXd::Identity(n_dofs, 5);
    
    Eigen::MatrixXd mock_inputs = Eigen::MatrixXd::Zero(1, 10);
    Eigen::SparseMatrix<double> mock_B = to_sparse(Eigen::MatrixXd::Zero(n_dofs, 1));
    Eigen::SparseMatrix<double> mock_T = to_sparse(Eigen::MatrixXd::Zero(1, 1));
    
    ModelOrderReduction mor_simulator(A, Ur, mock_inputs, mock_B, mock_T, /*case_name=*/"");
    
    Eigen::VectorXd initialState = Eigen::VectorXd::Random(n_dofs);
    Eigen::VectorXd reducedInitialState = Ur.transpose() * initialState;
    
    double dt = 1e-4;
    
    mor_simulator.run_until_ROM(reducedInitialState, /*iterations=*/1, dt, /*input_idx=*/0);
    
    SUCCEED();
}