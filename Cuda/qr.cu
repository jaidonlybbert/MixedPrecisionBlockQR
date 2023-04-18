/*
Copyright (c) 2023 Jaidon Lybbert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
* CUDA implementation of the Block QR decomposition algorithm
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

__global__ 
void householder_qr(float *dev_A, int m, int n, int global_offset) {
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
    for (int k = global_offset; k < n; k++) {
        /*
        * Compute householder vector
        */

        // Skip last transform is square matrix
        if (m == n && k == n - 1) {
            break;
        }

        // Copy the column as u - can be done in parallel
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

        // Compute householder normal vector w_k
        u[0] = sign * mag + u[0]; // v overwrites u
        // Normalize
        mag = 0;
        for (int i = 0; i < len; i++) {
            mag += u[i] * u[i];
        }
        mag = sqrtf(mag);
        for (int i = 0; i < len; i++) {
            u[i] /= mag; // w_k overwrites v, here u = w_k = v_k = householder vector
        }

        /*
        * Update trailing matrix : A_k:m, k : n = A_k:m,k:n - 2V((V ^ T)(A_k:m, k : n)
        */

        // (V^T)(A_k:m,k:n) - vector matrix product
        float* temp = (float*)malloc((n - k) * sizeof(float));
        for (int col = k; col < n; col++) {
            float inner_product = 0;
            for (int row = k; row < m; row++) {
                inner_product += u[row - k] * dev_A[row * n + col];
            }
            temp[col-k] = inner_product;
        }
        
        // (A_k:m,k:n) - 2 * (V)(V^T)(A_k:m,k:n)
        float* temp2 = (float*)malloc((n - k) * (m - k) * sizeof(float));
        for (int row = k; row < m; row++) {
            for (int col = k; col < n; col++) {
                temp2[(row - k) * (n - k) + (col - k)] = u[row-k] * temp[col-k];
                dev_A[row * n + col] = dev_A[row * n + col] - 2 * temp2[(row - k) * (n - k) + (col - k)];
            }
        }

        // Copy householder vector (vk) to lower triangular portion of A
        for (int row = k + 1; row < k + len + 1; row++) {
            dev_A[row * n + k] = u[row - k - 1];
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


void h_wy_transform(float* h_A, int m, int n, int global_offset, int panel_width)
{
    float* W = (float*)malloc((m - global_offset) * panel_width * sizeof(float));
    float* Y = (float*)malloc((m - global_offset) * panel_width * sizeof(float));
    float* z = (float*)malloc((m - global_offset) * sizeof(float));
    float* W_Yt = (float*)malloc((m - global_offset) * (m - global_offset) * sizeof(float)); // temporary matrix W * Y^T

    // Y = w_1
    for (int i = 0; i < m - global_offset; i++) {
        Y[i * panel_width] = h_A[(i + global_offset + 1) * n + global_offset];
        W[i * panel_width] = 2 * h_A[(i + global_offset + 1) * n + global_offset];
    }

    // Iterate over columns of panel and update W, Y
    for (int i = 1; i < panel_width; i++) { // cols of panel
        // Calculate z = 2 * (I_m - WY^T)w_i

        // Im - WY^T (classic "triply-nested-loop")
        for (int row = 0; row < m - global_offset; row++) { // rows of W_Yt
            for (int col = 0; col < m - global_offset; col++) { // cols of W_Yt
                // compute each inner product
                float inner_product = 0;
                for (int idx = 0; idx < i; idx++) { // rows of W
                    inner_product += W[row * panel_width + idx] * Y[col * panel_width + idx];
                }
                if (row == col) { // Im is 1
                    W_Yt[row * (m - global_offset) + col] = 1 - inner_product; // Im - WY^T
                }
                else { // Im is zero
                    W_Yt[row * (m - global_offset) + col] = -inner_product;
                }
            }
        }

        // 2 * (Im - WY^T)w_i (matrix-vector product)
        for (int row = 0; row < (m - global_offset); row++) {
            float inner_product = 0;
            for (int col = i; col < (m - global_offset); col++) {
                inner_product += W_Yt[row * (m - global_offset) + col] * h_A[(global_offset + col + 1) * n + global_offset + i];
            }
            z[row] = 2 * inner_product;
        }

        // Copy z to W
        for (int idx = 0; idx < (m - global_offset); idx++) {
            if (idx < (i)) {
                Y[idx * panel_width + i] = 0;
            }
            else {
                Y[idx * panel_width + i] = h_A[(global_offset + idx + 1) * n + global_offset + i];
            }
            W[idx * panel_width + i] = z[idx];
        }
    }

    free(W);
    free(Y);
    free(z);
    free(W_Yt);
}

void test_wy_transform() {
    int m = 3;
    int n = 3;

    // Initialize test matrix A input on Host
    float h_A_in[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41},
    };

    float* h_A_out = (float*)malloc((m + 1) * n * sizeof(float)); // extra row gives room for storing householder vectors in lower triangular portion of A
    float* h_Q_out = (float*)malloc(m * m * sizeof(float));

    h_wy_transform((float*)h_A_in, m, n, 0, n);
}

void read_euroc_jacobian(const char filename[], int* rows, int* cols, double** matrix) {
    std::ifstream fin;

    std::string line;

    fin.open(filename);

    // Read first line to get dimensions
    getline(fin, line);

    std::cout << line << std::endl;
    int start = line.find(" ");
    int end = line.find(" ");

    std::string rows_str = line.substr(0, start);
    std::string cols_str = line.substr(start + 1, end);

    std::cout << rows_str << std::endl;
    std::cout << cols_str << std::endl;

    *cols = std::stoi(cols_str);
    *rows = std::stoi(rows_str);

    printf("Rows: %d\nCols: %d\n", *rows, *cols);

    // Allocate memory for matrix
    *matrix = (double*)malloc((*rows) * (*cols) * sizeof(double));

    for (int row = 0; row < (*rows); row++) {
        for (int col = 0; col < (*cols); col++) {
            (*matrix)[row * (*cols) + col] = (double)0.0;
        }
    }

    int linecount = 0;
    while (getline(fin, line)) {
        //std::cout << line << std::endl;

        std::wstring::size_type pos = line.find_first_not_of(' ');
        line = line.substr(pos);
        pos = line.find(' ');
        std::string row_idx_str = line.substr(0, pos);
        line = line.substr(pos);

        pos = line.find_first_not_of(' ');
        line = line.substr(pos);
        pos = line.find(' ');
        std::string col_idx_str = line.substr(0, pos);
        line = line.substr(pos);

        pos = line.find_first_not_of(' ');
        line = line.substr(pos);
        pos = line.find(' ');
        std::string val_str = line.substr(0, pos);

        //std::cout << row_idx_str << std::endl;
        //std::cout << col_idx_str << std::endl;
        //std::cout << val_str << std::endl;

        //printf("Row idx: %d\nCol idx: %d\nVal: %.15f\n", std::stoi(row_idx_str), std::stoi(col_idx_str), std::stod(val_str));

        int row_idx = std::stoi(row_idx_str);
        int col_idx = std::stoi(col_idx_str);
        double val = std::stod(val_str);

        (*matrix)[row_idx * (*cols) + col_idx] = val;
        linecount++;
    }

    printf("Total linecount: %d\n", linecount);
}

void test_householder();

int main() {
    test_householder();
}
//
//__global__
//void householder_qr(float* dev_A, int m, int n, int tau, int r) {
//    // For k = tau + 1 to r
//        // Copy vector u from A[k:m,k]
//        // Calculate v vector from u
//        // Calculate w = v - u
//        // Perform matrix update on block 
//}
//
__global__
void apply_qt(float* dev_A, float* dev_Q, int m, int n, int tau) {
    // Collaboratively load householder vectors vk from global memory to shared memory
    // Construct W, K from householder vectors
    // Construct Q
    // Collaboratively load matrix A to shared memory
    // Perform tiled GMMULT TensorCore warp-level mixed precision fused multiply add operations to update Q and A
    // Update matrix Q, A in global memory
}
//
void block_qr(float* dev_A, float* dev_Q, int m, int n, int r) {
    /*
    * Kernel to compute QR decomposition with Block QR algorithm
    */

    int k = 0;

    // initialize Q, lambda, k
    while (int lambda = 0 <= n) {
        // set panel offset
        int tau = (lambda + r - 1 < n) ? (lambda + r - 1) : n;
        k += 1;

        dim3 GridDim(1, 1, 1);
        dim3 BlockDim(m, n, 1);

        householder_qr<<<GridDim, BlockDim>>>(dev_A, m, n, lambda);

        cudaDeviceSynchronize();

        // Q is stored in factored form in lower triangular portion of dev_A
        // R is stored in upper triangular portion of dev_A
        apply_qt<<<GridDim, BlockDim>>>(dev_A, m, n, tau);

        cudaDeviceSynchronize();

        // increment panel offset
    }
}
//
///*
//* PSEUDOCODE
//*/
//
//int main() {
//    // load matrix A
//    // allocate device memory
//    // copy matrix A to device
//
//    block_qr<<<1, 1>>>(dev_A, m, n, r);
//
//    // copy result R and Q to host
//    // report results
//}

void test_qr_result(int n, int m) {
    // Create random matrix A widthxheight

    // Start timer
    // Call cuda QR decomposition on A -> Q, R
    // End timer

    // Perform backward error calc ||A - QR||/||A||

    // Print error
    // Print FLOPs = 4(m^2*n - mn^2 + n^3/3) / time
}

void test_householder() {
    int rows, cols;
    double* mtx;

    read_euroc_jacobian("C:\\Users\\jaido\\source\\MixedPrecisionBlockQR\\Cuda\\jacobians\\A_000000100.txt", &rows, &cols, &mtx);

    int m = 3;
    int n = 3;

    // Initialize test matrix A input on Host
    float h_A_in[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41},
    };

    float* h_A_out = (float*)malloc((m+1) * n * sizeof(float)); // extra row gives room for storing householder vectors in lower triangular portion of A
    float* h_Q_out = (float*)malloc(m * m * sizeof(float));

    // Allocate device memory for input matrix
    float* dev_A;
    float* dev_Q; // Matrix Q in A=QR

    cudaMalloc(&dev_Q, m * m * sizeof(float));
    cudaMalloc(&dev_A, (m+1) * n * sizeof(float));

    // Copy input matrix to device Global memory
    cudaMemcpy(dev_A, h_A_in, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel to collaboratively copy input matrix from Global memory to Shared memory
    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(1, 1, 1);
    householder_qr << <DimGrid, DimBlock >> > (dev_A, dev_Q, m, n, 0);

    cudaDeviceSynchronize();

    cudaMemcpy(h_A_out, dev_A, (m+1) * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q_out, dev_Q, m * m * sizeof(float), cudaMemcpyDeviceToHost);

    h_wy_transform(h_A_out, m, n, 0, n);

    // Print input A
    printf("Input A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_A_in[i][j]);
        }
        printf("\n");
    }

    // Print output A
    printf("Result V\\R:\n");
    for (int i = 0; i < m+1; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_A_out[i * n + j]);
        }
        printf("\n");
    }

    // Print output Q
    printf("Result Q:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.2f ", h_Q_out[i * n + j]);
        }
        printf("\n");
    }

    // Backward error
    float backward_err = 0;

    printf("Backward error: ||A-QR||/||A|| = %f\n", backward_err);
    printf("Finished\n");
}

void h_block_qr() {

}