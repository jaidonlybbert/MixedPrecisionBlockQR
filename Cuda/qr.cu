#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>

__global__ 
void householder_qr(float *dev_A, int m, int n) {
    /*
    * Computes the QR decomposition of A using Householder reflectors.
    * 
    * Reference: 
    *   Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins 
    *   University Press. Pg. 249. Algorithm 5.2.1
    */

    // TODO: Copy A from global memory to shared memory
        //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
    dim3 GridDim(1, 1, 1);
    dim3 BlockDim(m, n, 1);

    // Iterate over columns
    for (int k = 0; k < n; k++) {
        /*
        * Compute householder vector
        */

        // Skip last transform is square matrix
        if (m == n && k == n - 1) {
            break;
        }

        // Copy the column as u
        int len = m - k;
        float* u = (float*)malloc((len) * sizeof(float));
        for (int i = 0; i < len; i++) {
            u[i] = dev_A[n * (i+k) + k];
        }

        // Create the householder vector from the column vector
        int sign = 0;
        if (u[0] >= 0) {
            sign = 1;
        }
        else if (u[0] < 0) {
            sign = -1;
        }

        // Get the magnitude of u
        float mag = 0;
        for (int i = 0; i < len; i++) {
            mag+=u[i] * u[i];
        }
        mag = sqrtf(mag);

        // Compute householder normal vector
        u[0] = sign * mag + u[0];
        // Normalize
        mag = 0;
        for (int i = 0; i < len; i++) {
            mag += u[i] * u[i];
        }
        mag = sqrtf(mag);
        for (int i = 0; i < len; i++) {
            u[i] /= mag;
        }

        /*
        * Update trailing matrix : A_k:m, k : n = A_k : m, k : n - 2V((V ^ T)(A_k:m, k : n)
        */

        // (V^T)(A_k:m,k:n)
        float* temp = (float*)malloc((n - k) * sizeof(float));
        for (int col = k; col < n; col++) {
            float inner_product = 0;
            for (int row = k; row < m; row++) {
                inner_product += u[row - k] * dev_A[row * n + col];
            }
            temp[col-k] = inner_product;
        }
        
        // (V)(V^T)(A_k:m,k:n)
        float* temp2 = (float*)malloc((n - k) * (m - k) * sizeof(float));
        for (int row = k; row < m; row++) {
            for (int col = k; col < n; col++) {
                temp2[(row - k) * (n - k) + (col - k)] = u[row-k] * temp[col-k];
                dev_A[row * n + col] = dev_A[row * n + col] - 2 * temp2[(row - k) * (n - k) + (col - k)];
            }
        }

        free(temp);
        free(temp2);
        free(u);
    }
}

__global__
void cpy_to_smem(float* dev_A, int m, int n) {
    /*
    * Collaboratively loads m-by-n memory array from global memory to shared memory
    */
}

int main() {
    int m = 3;
    int n = 3;

    // Initialize test matrix A input on Host
    float h_A_in[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41},
    };

    // Allocate Host memory for result
    float* h_A_out = (float*)malloc(m * n * sizeof(float));

    // Allocate device memory for input matrix
    float* dev_A;
    cudaMalloc(&dev_A, m * n * sizeof(float));

    // Copy input matrix to device Global memory
    cudaMemcpy(dev_A, h_A_in, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel to collaboratively copy input matrix from Global memory to Shared memory
    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(1, 1, 1);
    householder_qr<<<DimGrid, DimBlock>>>(dev_A, m, n);

    cudaDeviceSynchronize();

    cudaMemcpy(h_A_out, dev_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print input A
    printf("Input A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_A_in[i][j]);
        }
        printf("\n");
    }

    // Print output A
    printf("Result A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_A_out[i * n + j]);
        }
        printf("\n");
    }
    printf("Finished\n");

    return 0;
}