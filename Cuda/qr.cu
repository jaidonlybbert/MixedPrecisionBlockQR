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
* 
* Conventions:
*   Functions prefixed by "h_" sequentially execute on the CPU (host)
*   Functions prefixed by "dev_" execute in parallel on the GPU (device)
*/


#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <cstdlib>
#include <vector>

#define TC_TILE_M 16
#define TC_TILE_N 16
#define TC_TILE_K 16

void h_write_results_to_log(int height, int width, float time_ms, float flops_per_second, float backward_error) {
    //write arguments to log file
    std::vector<double> params = { width * 1.0, height * 1.0, time_ms, flops_per_second, backward_error };
    std::string line;
    for (int i = 0; i < params.size(); i++)
    {
        line += std::to_string(params[i]);
        if (i != params.size() - 1) {
            line += ',';
        }
    }
    line += "\n";


    std::ofstream logFile;
    logFile.open("logFile.txt", std::ios::app);
    logFile << line;
    logFile.close();
}

void h_identity_mtx(float* I, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            if (row == col) {
                I[row * n + col] = 1;
            }
            else {
                I[row * n + col] = 0;
            }
        }
    }
}

void h_mmult(float* A, float* B, float* C, int m, int n, int k) {
    /*
    * A - mxk
    * B - kxn
    * C - mxn
    *
    * C = AB
    */

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float inner_product = 0;
            for (int inner_idx = 0; inner_idx < k; inner_idx++) {
                inner_product += A[row * k + inner_idx] * B[(inner_idx)*n + col];
            }
            C[row * n + col] = inner_product;
        }
    }
}

void h_mmult_transpose_A(float* A, float* B, float* C, int m) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            float inner_product = 0;
            for (int inner_idx = 0; inner_idx < m; inner_idx++) {
                inner_product += A[(inner_idx)* m + row] * B[(inner_idx)*m + col];
            }
            C[row * m + col] = inner_product;
        }
    }
}

void h_matrix_subtract(float* A, float* B, float* C, int m, int n) {
    /*
    * Dimensions all match, element-wise subtraction
    *
    * C = A - B
    */

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            C[row * n + col] = A[row * n + col] - B[row * n + col];
        }
    }
}

float h_matrix_norm(float* A, int m, int n) {
    /*
    * A shape: mxn
    *
    * norm = ||A||
    */

    float norm = 0;
    float squared_sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            squared_sum += A[row * n + col] * A[row * n + col];
        }
    }
    return sqrtf(squared_sum);
}

void h_matrix_cpy(float* A, float* B, int m, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            B[row * n + col] = A[row * n + col];
        }
    }
}

void h_strip_R_from_A(float* A, float* R, int m, int n) {
    /*
    * Removes householder vectors from lower triangular section of A
    */

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            if (row <= col) {
                R[row * n + col] = A[row * n + col];
            }
            else {
                R[row * n + col] = 0;
            }
        }
    }
}

float h_backward_error(float* A, float* R, float* Q, int m, int n) {
    // Computes || A - QR||/||A ||

    float* QR = (float*)malloc(m * n * sizeof(float));
    float* A_sub_QR = (float*)malloc(m * n * sizeof(float));
    bool pass = false;
    const double error_limit = 1.1920928955078125e-07;
    h_mmult((float*)Q, R, QR, m, n, m);
    h_matrix_subtract((float*)A, QR, A_sub_QR, m, n);

    float backward_error = (h_matrix_norm(A_sub_QR, m, n) / h_matrix_norm((float*)A, m, n));
    if (backward_error <= error_limit * m * h_matrix_norm((float*)A, m, n)){
            pass = true;
    }
    printf("||A - QR||/||A|| = %e Error Criteria: %s\n", backward_error, pass ? "True" : "False");
    free(QR);
    free(A_sub_QR);

    return backward_error;
}

float h_error_2(float* Q, int m) {

    // ||Q^T @ Q - Im||
    const double error_limit = 1.1920928955078125e-07;
    bool pass = false;
    float* Qt_Q = (float*)malloc(m * m * sizeof(float));
    float* Im = (float*)malloc(m * m * sizeof(float));
    float* Qt_Q_sub_Im = (float*)malloc(m * m * sizeof(float));

    h_mmult_transpose_A(Q, Q, Qt_Q, m);
    h_identity_mtx(Im, m, m);
    h_matrix_subtract(Qt_Q, Im, Qt_Q_sub_Im, m, m);

    float error = h_matrix_norm(Qt_Q_sub_Im, m, m);
    if (error <= error_limit * m){
            pass = true;
    }
    printf("||QT @ Q - Im|| = %e Error Criteria: %s\n", error, pass ? "True" : "False");

    free(Qt_Q);
    free(Im);
    free(Qt_Q_sub_Im);

    return error;
}

float h_error_3(float* R, int m, int n) {
    // Compute third type of error for QR result
    // ||L|| < m * 2E-23
    const double error_limit = 1.1920928955078125e-07;
    bool pass = false;
    float* L = (float*)malloc(m * n * sizeof(float));
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            if (col < row){
                L[row * n + col] = R[row * n + col];
            }
            else{
                L[row * n + col] = 0;
            }
        }
    }
    float error3 = (h_matrix_norm(L, m, n));
    if (error3 <= error_limit * m){
	    pass = true;
    }
    printf("||L|| = %e Error Criteria: %s\n", error3, pass ? "True" : "False");
    free(L);

    return error3;
}

void h_householder_qr(float* A, int m, int n, int global_offset, int panel_width) {
    /*
    * Computes the QR decomposition of A using Householder reflectors.
    *
    * Reference:
    *   Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins
    *   University Press. Pg. 249. Algorithm 5.2.1
    */

    // Iterate over columns
    int r = (panel_width + global_offset) > n ? n: panel_width + global_offset;
    for (int k = global_offset; k < r; k++) {
        /*
        * Compute householder vector
        */

        // Skip last transform if square matrix
        //if (m == n && k == n - 1) {
        //    break;
        //}

        // Copy the column as u - can be done in parallel
        int len = m - k;
        float* u = (float*)malloc((len) * sizeof(float));
        for (int i = 0; i < len; i++) {
            u[i] = A[n * (i + k) + k];
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
            mag += u[i] * u[i];
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
        * Update trailing matrix : A_k:m,k:r = A_k:m,k:r - 2V((V ^ T)(A_k:m,k:r)
        */

        // (V^T)(A_k:m,k:r) - vector matrix product
        float* temp = (float*)malloc((r - k) * sizeof(float));
        for (int col = k; col < r; col++) {
            float inner_product = 0;
            for (int row = k; row < m; row++) {
                inner_product += u[row - k] * A[row * n + col];
            }
            temp[col - k] = inner_product;
        }

        // (A_k:m,k:r) - 2 * (V)(V^T)(A_k:m,k:r)
        float* temp2 = (float*)malloc((r - k) * (m - k) * sizeof(float));
        for (int row = k; row < m; row++) {
            for (int col = k; col < r; col++) {
                temp2[(row - k) * (r - k) + (col - k)] = u[row - k] * temp[col - k];
                A[row * n + col] = A[row * n + col] - 2 * temp2[(row - k) * (r - k) + (col - k)];
            }
        }

        // Copy householder vector (vk) to lower triangular portion of A
        for (int row = k + 1; row < k + len + 1; row++) {
            A[row * n + k] = u[row - k - 1];
        }

        free(temp);
        free(temp2);
        free(u);
    }
}

void h_wy_transform(float* h_A, float** h_Q, int m, int n, int global_offset, int panel_width)
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

    // Im - WY^T (classic "triply-nested-loop")
    for (int row = 0; row < m - global_offset; row++) { // rows of W_Yt
        for (int col = 0; col < m - global_offset; col++) { // cols of W_Yt
            // compute each inner product
            float inner_product = 0;
            for (int idx = 0; idx < panel_width; idx++) { // cols of W
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

    free(W);
    free(Y);
    free(z);
    //free(W_Yt);
    *h_Q = W_Yt;
}

__global__ void global_mem_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB using global memory with CUDA
*
* Assumed a_width == b_height
*
* Dimensions of C are a_height x b_width
*/
{
    // row and column of the C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_height && col < b_width) {
        // Calculate the inner product of the row of A and column of B
        float innerProduct = 0;
        for (int i = 0; i < a_width; i++) {
            innerProduct += a_mtx[a_width * row + i] * b_mtx[b_width * i + col];
        }

        c_mtx[b_width * row + col] = innerProduct;
    }
}

#define TILE_WIDTH 3

__global__ void shared_mem_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB using shared memory with CUDA
*
* Assumed a_width == b_height
*
* Dimensions of C are a_height x b_width
*/
{
    // row and column of C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bds[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(a_width / (float)TILE_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * TILE_WIDTH + tx < a_width) && (row < a_height)) {
            ads[ty][tx] = a_mtx[row * a_width + i * TILE_WIDTH + tx];
        }

        if ((i * TILE_WIDTH + ty < a_width) && (col < b_width)) {
            bds[ty][tx] = b_mtx[(i * TILE_WIDTH + ty) * b_width + col];
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            if ((i * TILE_WIDTH + k) < a_width)
                pval += ads[ty][k] * bds[k][tx];
        }
        __syncthreads();
    }

    if (col < b_width && row < a_height) {
        c_mtx[row * b_width + col] = pval;
    }
}

__global__
void dev_tensorcore_mmult_1_warp(float* c_mtx, half* a_mtx, half* b_mtx) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    using namespace nvcuda;
    // Create fragments 
    wmma::fragment<wmma::matrix_a, TC_TILE_M, TC_TILE_N, TC_TILE_K, half, wmma::row_major> Amat;
    wmma::fragment<wmma::matrix_b, TC_TILE_M, TC_TILE_N, TC_TILE_K, half, wmma::row_major> Bmat;
    wmma::fragment<wmma::accumulator, TC_TILE_M, TC_TILE_N, TC_TILE_K, float, void> Cmat;

    // Initialize output to zero
    wmma::fill_fragment(Cmat, 0.0f);

    // Load inputs
    wmma::load_matrix_sync(Amat, a_mtx, TC_TILE_M);
    wmma::load_matrix_sync(Bmat, b_mtx, TC_TILE_K);

    // Perfrom matrix multiplication
    wmma::mma_sync(Cmat, Amat, Bmat, Cmat);

    // Store output
    wmma::store_matrix_sync(c_mtx, Cmat, TC_TILE_N, wmma::mem_row_major);
}


void test_tensorcore_mmult_gmem() {
    printf("\nTesting tensorcore 16x16x16 mmult...\n");

    __half* a_mtx = (__half*)malloc(16 * 16 * sizeof(__half));
    __half* b_mtx = (__half*)malloc(16 * 16 * sizeof(__half));
    float* c_mtx = (float*)malloc(16 * 16 * sizeof(float));

    // initialize matrices A, B, C
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            a_mtx[i * 16 + j] = (__half)(float)j;
            b_mtx[i * 16 + j] = (__half)(float)j;
            c_mtx[i * 16 + j] = (__half)0.0f;
        }
    }

    // Allocate device memory
    __half* dev_a;
    __half* dev_b;
    float* dev_c;

    cudaMalloc(&dev_a, 16 * 16 * sizeof(__half));
    cudaMalloc(&dev_b, 16 * 16 * sizeof(__half));
    cudaMalloc(&dev_c, 16 * 16 * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a_mtx, 16 * 16 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_mtx, 16 * 16 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c_mtx, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 1, 1); // one warp

    dev_tensorcore_mmult_1_warp << <gridDim, blockDim >> > (dev_c, dev_b, dev_a);

    cudaDeviceSynchronize();

    cudaMemcpy(c_mtx, dev_c, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);

    // test result
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            assert(c_mtx[i * 16 + j] == j * 120);
        }
    }

    printf("Test passed.\n");

}


__global__ 
void dev_householder_qr(float *dev_A, int m, int n, int global_offset) {
    /*
    * Computes the QR decomposition of A using Householder reflectors.
    * 
    * Reference: 
    *   Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins 
    *   University Press. Pg. 249. Algorithm 5.2.1
    */

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
        for (int i = 0; i < len; i++) { // TASK4 1 shashank: implement parallel algorithm in CUDA to replace for loop
            mag += u[i] * u[i];
        }
        mag = sqrtf(mag);
        for (int i = 0; i < len; i++) { // TASK5 1 shashank: implement parallel algorithm in CUDA to replace for loop
            u[i] /= mag; // w_k overwrites v, here u = w_k = v_k = householder vector
        }

        /*
        * Update trailing matrix : A_k:m, k : n = A_k:m,k:n - 2V((V ^ T)(A_k:m, k : n)
        */

        // (V^T)(A_k:m,k:n) - vector matrix product
        float* temp = (float*)malloc((n - k) * sizeof(float));
        for (int col = k; col < n; col++) { // TASK6 1 shashank: implement parallel algorithm in CUDA to replace for loop
            float inner_product = 0;
            for (int row = k; row < m; row++) {
                inner_product += u[row - k] * dev_A[row * n + col];
            }
            temp[col-k] = inner_product;
        }
        
        // (A_k:m,k:n) - 2 * (V)(V^T)(A_k:m,k:n)
        float* temp2 = (float*)malloc((n - k) * (m - k) * sizeof(float));
        for (int row = k; row < m; row++) { // TASK7 1 shashank: implement parallel algorithm in CUDA to replace for loop
            for (int col = k; col < n; col++) {
                temp2[(row - k) * (n - k) + (col - k)] = u[row-k] * temp[col-k];
                dev_A[row * n + col] = dev_A[row * n + col] - 2 * temp2[(row - k) * (n - k) + (col - k)];
            }
        }

        // Copy householder vector (vk) to lower triangular portion of A
        for (int row = k + 1; row < k + len + 1; row++) { // TASK8 1 shashank: implement parallel algorithm in CUDA to replace for loop
            dev_A[row * n + k] = u[row - k - 1];
        }

        free(temp);
        free(temp2);
        free(u);
    }
}

float* h_generate_random_matrix(int height, int width) {
    /*
    * Returns pointer to random float matrix of dimensions HeightxWidth
    */
    unsigned seed = time(0);
    srand(seed);
    float* matrix = (float*)malloc(height * width * sizeof(float));
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            matrix[row * width + col] = rand(); // randomize this number
        }
    }

    return matrix;
}

void read_euroc_jacobian(const char filename[], int* rows, int* cols, double** matrix) {
    /*
    * Reads text file containing jacobian matrices from the Euroc dataset, and returns pointer to matrix
    */

    std::ifstream fin;

    std::string line;

    fin.open(filename);

    if (!fin) {
        printf("File not found.");
    }

    assert(fin);

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


__global__
void dev_apply_qt_to_a(float* dev_A, float* dev_panel_Q, int m, int n, int tau, int lambda) {
    // Collaboratively load householder vectors vk from global memory to shared memory
    // Construct W, K from householder vectors
    // Construct Q
    // Collaboratively load matrix A to shared memory
    // Perform tiled GMMULT TensorCore warp-level mixed precision fused multiply add operations to update Q and A
    // Update matrix Q, A in global memory

    int row = blockIdx.y * blockDim.y + threadIdx.y + lambda;
    int col = blockIdx.x * blockDim.x + threadIdx.x + tau;

    if (row < m && row >= lambda && col < n && col >= tau) {
        float inner_product = 0;
        for (int inner_dim = 0; inner_dim < (m - lambda); inner_dim++) {
            inner_product += dev_panel_Q[(inner_dim) * (m - lambda) + (row - lambda)] * dev_A[(inner_dim)*n + col];
        }
        dev_A[row * n + col] = inner_product;
    }
}

__global__ 
void dev_apply_qt_to_a_tensorcore_gmem(half* dev_A, half* dev_panel_Q, int m, int n, int tau, int lambda) {

}

__global__
void dev_apply_qpanel_to_q(float* dev_Q, float* dev_Q_panel, int m, int lambda) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x + lambda;

    if (row >= 0 && row < m && col >= lambda && col < m) {
        float inner_product = 0;
        for (int inner_dim = 0; inner_dim < (m - lambda); inner_dim++) {
            inner_product += dev_Q[row * m + inner_dim + lambda] * dev_Q_panel[(inner_dim * (m - lambda)) + (col - lambda)];
        }
        dev_Q[row * m + col] = inner_product;
    }
}



void dev_block_qr(float* A, float* Q, int m, int n, int r) {
    /*
    * GPU code to compute QR decomposition with Block QR algorithm
    */

    float* panel_Q = NULL;
    int lambda = 0;
    while (lambda < n) { // panel starts at lambda
        int tau = (lambda + r < n) ? (lambda + r) : n; // panel ends at tau

        // Q is stored in factored form in lower triangular portion of dev_A
        // R is stored in upper triangular portion of dev_A
        h_householder_qr(A, m, n, lambda, r);

        // Get panel Q from factors - dim panel_Q: (m-lambda)x(m-lambda)
        h_wy_transform(A, &panel_Q, m, n, lambda, r); // TASK10 3 shashank: write cuda kernel to implement WY transform on GPU

        // Update matrix A = Q^T @ A
        float blockWidth = 32.;
        float blockHeight = 32.;

        float* dev_A;
        float* dev_Q;
        float* dev_panel_Q;

        cudaMalloc(&dev_A, m * n * sizeof(float));
        cudaMalloc(&dev_Q, m * m * sizeof(float));
        cudaMalloc(&dev_panel_Q, (m - lambda) * (m - lambda) * sizeof(float));

        cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_panel_Q, panel_Q, (m - lambda) * (m - lambda) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Q, Q, m * m * sizeof(float), cudaMemcpyHostToDevice);

        dim3 BlockDim((int)blockWidth, (int)blockHeight, 1);
        dim3 GridDim(ceil((n - tau) / blockWidth), ceil((m - lambda) / blockHeight), 1);

        // Update global Q
        dev_apply_qt_to_a<<<GridDim, BlockDim>>>(dev_A, dev_panel_Q, m, n, tau, lambda);

        dim3 BlockDim2((int)blockWidth, (int)blockHeight, 1);
        dim3 GridDim2(ceil((m - lambda) / blockWidth), ceil((m) / blockHeight), 1);
        dev_apply_qpanel_to_q << <GridDim2, BlockDim2 >> >(dev_Q, dev_panel_Q, m, lambda);

        cudaDeviceSynchronize();

        cudaMemcpy(A, dev_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Q, dev_Q, m * m * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(dev_A);
        cudaFree(dev_panel_Q);
        cudaFree(dev_Q);

        // increment panel offset
        lambda = tau;
    }
}

void test_dev_householder_qr() {
    printf("\nTesting GPU householder QR...\n");

    int rows, cols;
    double* mtx;

    //read_euroc_jacobian("C:\\Users\\jaido\\source\\MixedPrecisionBlockQR\\Cuda\\jacobians\\A_000000100.txt", &rows, &cols, &mtx);

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
    float* h_R = (float*)malloc(m * n * sizeof(float));

    // Allocate device memory for input matrix
    float* dev_A;
    float* dev_Q; // Matrix Q in A=QR

    //cudaMalloc(&dev_Q, m * m * sizeof(float));
    cudaMalloc(&dev_A, (m+1) * n * sizeof(float));

    // Copy input matrix to device Global memory
    cudaMemcpy(dev_A, h_A_in, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel to collaboratively copy input matrix from Global memory to Shared memory
    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(1, 1, 1);
    // Time execution of the following kernel call
    dev_householder_qr <<<DimGrid, DimBlock >> > (dev_A, m, n, 0);

    cudaDeviceSynchronize();

    cudaMemcpy(h_A_out, dev_A, (m+1) * n * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_Q_out, dev_Q, m * m * sizeof(float), cudaMemcpyDeviceToHost);

    h_wy_transform(h_A_out, &h_Q_out, m, n, 0, n);

    h_strip_R_from_A((float*)h_A_out, h_R, m, n);

    float backward_error = h_backward_error((float*)h_A_in, h_R, h_Q_out, m, n);
    float error3 = h_error_3(h_R, m, n);
    float error2 = h_error_2(h_Q_out, m);
    //printf("||A - QR||/||A|| = %e\n", backward_error);
    //printf("||QT @ Q - Im|| = %e\n", h_error_2(h_Q_out, m));
    //printf("||L|| = %e\n", error3);
    printf("GPU householder QR finished...\n");


    // Write results to log file
    //h_write_results_to_log()

    //h_wy_transform(h_A_out, m, n, 0, n);

}

void h_block_qr(float* A, float* Q, int m, int n, int r) {
    /*
    * Sequential version of the block QR algorithm, runs on CPU
    */

    // initialize Q, lambda, k
    //h_identity_mtx(Q, m, m);
    float* panel_Q = NULL;
    int lambda = 0;
    while (lambda < n) { // panel starts at lambda
        int tau = (lambda + r < n) ? (lambda + r) : n; // panel ends at tau

        // Q is stored in factored form in lower triangular portion of dev_A
        // R is stored in upper triangular portion of dev_A
        h_householder_qr(A, m, n, lambda, r);

        // Get panel Q from factors
        h_wy_transform(A, &panel_Q, m, n, lambda, r); // dim panel_Q: (m-lambda)x(m-lambda)

        // Update matrix A = Q^T @ A
        float* A_old = (float*)malloc(m * n * sizeof(float));
        memcpy(A_old, A, m * n * sizeof(float));
        for (int row = lambda; row < m; row++) {
            for (int col = tau; col < n; col++) {
                float inner_product = 0;
                for (int inner_dim = 0; inner_dim < (m - lambda); inner_dim++) {
                    inner_product += panel_Q[(inner_dim) * (m - lambda) + (row - lambda)] * A_old[(inner_dim) * n + col];
                }
                A[row * n + col] = inner_product;
            }
        }
        free(A_old);

        // Update global Q
        float* Q_old = (float*)malloc(m * m * sizeof(float)); 
        memcpy(Q_old, Q, m * m * sizeof(float));
        for (int row = 0; row < m; row++) {
            for (int col = lambda; col < m; col++) {
                float inner_product = 0;
                for (int inner_dim = 0; inner_dim < (m - lambda); inner_dim++) {
                    inner_product += Q_old[row * m + inner_dim + lambda] * panel_Q[(inner_dim * (m-lambda)) + (col-lambda)];
                }
                Q[row * m + col] = inner_product;
            }
        }
        free(Q_old);

        // increment panel offset
        lambda = tau;
    }
}



void test_h_mmult() {
    float A[3][3] = {
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3}
    };

    int m = 3;
    int n = 3;
    int k = 3;

    float* C = (float*)malloc(m * n * sizeof(float));

    h_mmult((float*)A, (float*)A, C, m, n, k);
}

void test_h_mmult_transpose_A() {
    float A[3][3] = {
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3}
    };

    float expected_result[3][3] = {
        {3, 6, 9},
        {6, 12, 18},
        {9, 18, 27}
    };

    int m = 3;
    int n = 3;
    int k = 3;

    float* C = (float*)malloc(m * n * sizeof(float));

    h_mmult_transpose_A((float*)A, (float*)A, C, m);

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            assert((C[row * n + col] - ((float*)expected_result)[row * n + col]) < 1E-8);
        }
    }
}

void test_h_householder_qr() {
    /*
    * Test host version of householder QR
    */

    // TASK14 3 alice: iterate over many matrix sizes, & test matrices from Tong
    printf("\nTesting sequential householder QR...\n");

    float A_in[6][4] = {
        {10,20,30,40},
        {32,32,44,55},
        {23,66,74,64},
        {67,28,46,26},
        {95,95,52,88},
        {75,53,96,47},
    };

    int m = 6;
    int n = 4;
    int r = 4;

    int global_offset = 0;

    float* Q = (float*)malloc(m * m * sizeof(float));
    float* R = (float*)malloc(m * n * sizeof(float));
    float* A_out = (float*)malloc((m + 1) * n * sizeof(float));

    h_matrix_cpy((float*)A_in, A_out, m, n);

    //h_block_qr((float*)A, Q, m, n, r);
    h_householder_qr((float*)A_out, m, n, global_offset, r);

    h_wy_transform(A_out, &Q, m, n, global_offset, r);

    h_strip_R_from_A((float*)A_out, R, m, n);

    float backward_error = h_backward_error((float*)A_in, R, Q, m, n);
    float error3 = h_error_3(R, m, n);
    float error2 = h_error_2(Q, m);
    //printf("||A - QR||/||A|| = %e\n", backward_error);
    //printf("||QT @ Q - Im|| = %e\n", h_error_2(Q, m));
    //printf("||L|| = %e\n", error3);
    printf("Sequential householder QR finished...\n");

    h_write_results_to_log(m, n, 0, 0, backward_error);


    // write results to log file
    free(Q);
    free(R);
    free(A_out);



}


void test_h_wy_transform() {
    // Initialize test matrix A input on Host
    // TASK16 Alice: iterate over many matrix sizes
    int m = 3;
    int n = 3;

    // TASK17 Alice: use h_generate_random_matrix to randomize input matrix
    float h_A_in[3][3] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41},
    };

    float* h_A_out = (float*)malloc((m + 1) * n * sizeof(float)); // extra row (m+1) gives room for storing householder vectors in lower triangular portion of A
    float* h_R = (float*)malloc(m * n * sizeof(float));
    float* h_Q_out = NULL; // pointer to Q is returned by h_wy_transform

    h_householder_qr((float*)h_A_in, m, n, 0, n);

    h_wy_transform((float*)h_A_out, &h_Q_out, m, n, 0, n);

    h_strip_R_from_A(h_A_out, h_R, m, n);

    float backward_error = h_backward_error((float*)h_A_in, h_R, h_Q_out, m, n);

    free(h_A_out);
    free(h_Q_out);
    free(h_R);
}


float h_qr_flops_per_second(float time_ms, int m, int n) {
    /*
    * Computes FLOPs / second for householder QR given matrix dimensions and execution time
    * 
    * TASK21 2 Mike: Verify equation and provide academic reference for equation (textbook or paper)
    */
    return (4. * (pow<float>(m, 2) * n - m * pow<float>(n, 2) + pow<float>(n, 3) / 3.)) / (time_ms * 1000);
}

void test_h_block_qr() {
    /*
    * Test host version of block QR
    */

    printf("\nTesting sequential block QR...\n");

    // use read_euroc_jacobian to load test matrices
    float A_in[6][4] = {
        {10,20,30,40},
        {32,32,44,55},
        {23,66,74,64},
        {67,28,46,26},
        {95,95,52,88},
        {75,53,96,47},
    };

    int m = 6;
    int n = 4;
    int r = 2;

    float* Q = (float*)malloc(m * m * sizeof(float));
    float* R = (float*)malloc(m * n * sizeof(float));
    float* A_out = (float*)malloc((m + 1) * n * sizeof(float));

    h_identity_mtx(Q, m, m);

    h_matrix_cpy((float*)A_in, A_out, m, n);

    float time_ms = 0; // Time how long the QR function takes to execute

    h_block_qr((float*)A_out, Q, m, n, r);

    float flops_per_second = h_qr_flops_per_second(time_ms, m, n);

    h_strip_R_from_A((float*)A_out, R, m, n);

    float backward_error = h_backward_error((float*)A_in, R, Q, m, n);
    float error2 = h_error_2(Q, m);
    float error3 = h_error_3(R, m, n);

    // write results to log file
    h_write_results_to_log(m, n, time_ms, flops_per_second, backward_error);

    printf("Sequential block QR finished...\n");
    //printf("||A - QR||/||A|| = %e\n", backward_error);
    free(Q);
    free(R);
    free(A_out);
}

void test_dev_block_qr() {
    /*
    * Test GPU version of block QR
    */

    printf("\nTesting GPU block QR...\n");

    // use read_euroc_jacobian to load test matrices
    float A_in[6][4] = {
        {10,20,30,40},
        {32,32,44,55},
        {23,66,74,64},
        {67,28,46,26},
        {95,95,52,88},
        {75,53,96,47},
    };

    int m = 6;
    int n = 4;
    int r = 2;

    float* Q = (float*)malloc(m * m * sizeof(float));
    float* R = (float*)malloc(m * n * sizeof(float));
    float* A_out = (float*)malloc((m + 1) * n * sizeof(float));

    h_identity_mtx(Q, m, m);

    h_matrix_cpy((float*)A_in, A_out, m, n);

    float time_ms = 0; // Time how long the QR function takes to execute

    dev_block_qr((float*)A_out, Q, m, n, r);

    float flops_per_second = h_qr_flops_per_second(time_ms, m, n);

    h_strip_R_from_A((float*)A_out, R, m, n);

    float backward_error = h_backward_error((float*)A_in, R, Q, m, n);
    float error2 = h_error_2(Q, m);
    float error3 = h_error_3(R, m, n);

    // write results to log file
    h_write_results_to_log(m, n, time_ms, flops_per_second, backward_error);

    printf("GPU block QR finished...\n");
   // printf("||A - QR||/||A|| = %e\n", backward_error);
    
    free(Q);
    free(R);
    free(A_out);
}

int main() {
//	std::out<< "testing" << endl;
    test_dev_householder_qr();
    test_h_mmult();
    test_h_mmult_transpose_A();
    test_h_householder_qr();
    test_h_block_qr();
    test_dev_block_qr();
    test_tensorcore_mmult_gmem();
}
