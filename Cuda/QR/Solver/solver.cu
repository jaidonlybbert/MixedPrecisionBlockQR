#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>

#include <cstdlib>
#include <stdio.h>

#include "mmult.cuh"
#include "qr.cuh"
#include <Eigen/Dense>

#define MAX_SOLVER_ERROR 1E-4

void h_QR_Solver(float* A, float* result, int m, int n) {

}

Eigen::VectorXf h_Eigen_Solver(float* A, int m, int n) {
	// Example for 5 points
	Eigen::MatrixXf matA = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(A, m, n); // row: 5 points; column: xyz coordinates

	// Residual is all -1s
	Eigen::Matrix<float, Eigen::Dynamic, 1> matB = -1 * Eigen::Matrix<float, Eigen::Dynamic, 1>::Ones(m, 1);

	// Solve for x
	Eigen::VectorXf x = matA.colPivHouseholderQr().solve(matB);

	return x;
}

__global__
void dev_linear_solve(float* dev_R, float* dev_Q, float* dev_b, float* dev_x) {

}

void dev_QR_Solver(float* A, float* b, float* x, int m, int n) {
	/*
	* Launches kernels to solve for x which minimizes ||Ax - b||
	* 
	* Reference:
    *   Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins
    *   University Press. Pg. 264. Algorithm 5.3.2
    */

	
	size_t A_size = m * n * sizeof(float);
	size_t R_size = m * n * sizeof(float);
	size_t Q_size = m * m * sizeof(float);
	size_t b_size = n * sizeof(float);
	size_t x_size = n * sizeof(float);

	float* dev_A;
	float* dev_Q;
	float* dev_R;
	float* dev_b;
	float* dev_x;

	cudaMalloc(&dev_A, A_size);
	cudaMalloc(&dev_b, b_size);
	cudaMalloc(&dev_x, x_size);
	cudaMalloc(&dev_Q, Q_size);
	cudaMalloc(&dev_R, R_size);

	cudaMemcpy(dev_A, A, A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, b_size, cudaMemcpyHostToDevice);

	/*
	* Pseudocode!
	* Call custom QR decomposition function
	*/

	// Configure grid for Least Squares algorithm
	// Leave the QR result on the device for operating on

	// Copy result to host
	cudaMemcpy(x, dev_x, x_size, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(dev_x);
	cudaFree(dev_b);
	cudaFree(dev_A);
	cudaFree(dev_Q);
	cudaFree(dev_R);
}

void test_QRSolver(int m, int n) {
	float* A = h_generate_random_matrix<float>(m, n);

	float* b = (float*)malloc(n * sizeof(float));
	float* h_Result = (float*)malloc(n * sizeof(float));
	float* eigen_Result = (float*)malloc(n * sizeof(float));
	float* dev_Result = (float*)malloc(n * sizeof(float));

	Eigen::VectorXf eigen_result = h_Eigen_Solver(A, m, n);

	bool pass = true;
	int error_count = 0;
	float cpu_error, gpu_error;
	for (int idx = 0; idx < n; idx++) {

		cpu_error = fabs(h_Result[idx] - eigen_Result[idx]);
		if (cpu_error > MAX_SOLVER_ERROR) {
			error_count++;
			if (error_count < 100) {
				printf("CPU solver failed verification against Eigen. \
					Error %.2E greater than %.2E\n", cpu_error, MAX_SOLVER_ERROR);
			}
		}
		gpu_error = fabs(dev_Result[idx] - eigen_Result[idx]);
		if (gpu_error > MAX_SOLVER_ERROR) {
			error_count++;
			if (error_count < 100) {
				printf("GPU solver failed verification against Eigen. \
					Error %.2E greater than %.2E\n", gpu_error, MAX_SOLVER_ERROR);
			}
		}
	}

	free(A);
	free(h_Result);
	free(eigen_Result);
}


int main() {
	// Test the solver over a range of random matrices
	for (int rows = 1; rows < 1000; rows *= 2) {
		for (int cols = 1; cols < rows; cols *= 2) {
			float* A = (float*)malloc(rows * cols * sizeof(float));
			float* Result = (float*)malloc(rows * sizeof(float));
			//h_Eigen_Solver(rows, cols);
			test_QRSolver(rows, cols);
		}
	}
}