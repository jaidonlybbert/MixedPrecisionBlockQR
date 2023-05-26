/*
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

// CUDA includes
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

#define TC_TILE_M 16
#define TC_TILE_N 16
#define TC_TILE_K 16
#define WARP_SIZE 32

// Shared memory tile sizes for tiled global memory mmult
#define GMEM_MMULT_TILE_WIDTH 16
#define GMEM_MMULT_TILE_HEIGHT 16

// Thread block sizes for TC MMULT
// Shrink to get more shared memory per warp, expand to increase memory re-use
#define TC_MMULT_THREAD_BLOCK_WIDTH 128
#define TC_MMULT_THREAD_BLOCK_HEIGHT 4

// Thread block sizes for array cpy
#define CPY_ARRAY_BLOCK_WIDTH 32
#define CPY_ARRAY_BLOCK_HEIGHT 32

#define TOP_LEFT 1
#define BOTTOM_RIGHT 0

#define TILE_WIDTH 32

#define COLUMN_MAJOR 0
#define ROW_MAJOR 1

typedef void QR_FUNC(int, int, int, float*);
typedef void MMULT_FUNC(int, int, int);

void h_write_results_to_log(int height, int width, float time_ms, float flops_per_second, float backward_error, std::string file_name = "logFile") {
    //write arguments to log file
    std::vector<double> params = { height * 1.0, width * 1.0, time_ms, flops_per_second, backward_error };
    std::string path = "log/" + file_name + ".txt";
    std::ifstream file(path);
    std::string line = "";

    if (!file.good()) {
        line += "rows,cols,runtime,flops,error\n";
    }

    for (int i = 0; i < params.size(); i++)
    {
        line += std::to_string(params[i]);
        if (i != params.size() - 1) {
            line += ',';
        }
    }
    line += "\n";


    std::ofstream logFile;
    logFile.open(path, std::ios::app);
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

template <typename T>
T* h_generate_random_matrix(int height, int width) {
    /*
    * Returns pointer to random float matrix of dimensions HeightxWidth
    */
    unsigned seed = time(0);
    srand(seed);
    T* matrix = (T*)malloc(height * width * sizeof(T));
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float rand_fun = (float)((float)rand() / RAND_MAX);
            T random_num = (T)rand_fun;
            matrix[row * width + col] = random_num; // randomize this number
        }
    }

    return matrix;
}
template float* h_generate_random_matrix<float>(int, int);
template __half* h_generate_random_matrix<__half>(int, int);

template <typename T_A, typename T_B, typename T_C>
void h_mmult(T_A* A, T_B* B, T_C* C, int m, int n, int k) {
    /*
    * A - mxk
    * B - kxn
    * C - mxn
    *
    * C = AB
    */

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            T_C inner_product = 0;
            for (int inner_idx = 0; inner_idx < k; inner_idx++) {
                inner_product += (T_C)A[row * k + inner_idx] * (T_C)B[(inner_idx)*n + col];
            }
            C[row * n + col] = inner_product;
        }
    }
}

template void h_mmult<float, float, float>(float*, float*, float*, int, int, int);
template void h_mmult<__half, __half, float>(__half*, __half*, float*, int, int, int);

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

float h_qr_flops_per_second(float time_ms, int m, int n) {
    /*
    * Computes FLOPs / second for householder QR given matrix dimensions and execution time
    *
    * TASK21 2 Mike: Verify equation and provide academic reference for equation (textbook or paper)
    */
    return (4. * (pow<float>(m, 2) * n - m * pow<float>(n, 2) + pow<float>(n, 3) / 3.)) / (time_ms / 1000);
}

float h_backward_error(float* A, float* R, float* Q, int m, int n) {
    // Computes || A - QR||/||A ||

    float* QR = (float*)malloc(m * n * sizeof(float));
    float* A_sub_QR = (float*)malloc(m * n * sizeof(float));
    bool pass = false;
    const double error_limit = 1.1920928955078125e-07;
    h_mmult<float, float, float>((float*)Q, R, QR, m, n, m);
    h_matrix_subtract((float*)A, QR, A_sub_QR, m, n);

    float a_norm = h_matrix_norm((float*)A, m, n);

    float backward_error = (h_matrix_norm(A_sub_QR, m, n) / a_norm);
    if (backward_error <= error_limit * m){
            pass = true;
    }
    printf("||A - QR||/||A|| = %e Error Criteria: %s\n", backward_error, pass ? "True" : "False");
    free(QR);
    free(A_sub_QR);

    return backward_error;
}

float h_error_2(float* Q, int m) {

    // ||Q^T @ Q - Im||
    const double error_limit = pow<double>(2, -23);//1.1920928955078125e-07;
    bool pass = false;
    float* Qt_Q = (float*)malloc(m * m * sizeof(float));
    float* Im = (float*)malloc(m * m * sizeof(float));
    float* Qt_Q_sub_Im = (float*)malloc(m * m * sizeof(float));

    h_mmult_transpose_A(Q, Q, Qt_Q, m);
    h_identity_mtx(Im, m, m);
    h_matrix_subtract(Qt_Q, Im, Qt_Q_sub_Im, m, m);

    float max_error = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            if (Qt_Q_sub_Im[row * m + col] > max_error) {
                max_error = Qt_Q_sub_Im[row * m + col];
            }
        }
    }
    if (max_error <= error_limit * m){
            pass = true;
    }
    printf("||QT @ Q - Im|| = %E Error Criteria: %s\n", max_error, pass ? "True" : "False: should be less than ");

    if (!pass) {
        printf("%.2E\n", error_limit * m);
    }

    free(Qt_Q);
    free(Im);
    free(Qt_Q_sub_Im);

    return max_error;
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

    nvtxRangePush(__func__);

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
        if (mag == 0) {
            continue;
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

    nvtxRangePop();
}


void h_q_backward_accumulation(float* h_A, float** h_Q, int m, int n) {
    /*
    * "Backward accumulation" of Q from householder vectors stored in lower trapezoidal region
    *   of A, after householder QR
    *
    * Reference:
    *   Golub, Van Loan. Matrix Computations, Fourth Edition. The Johns Hopkins
    *   University Press. Pg. 238. Algorithm 5.1.5
    */

    // Initialize Q as identity
    *h_Q = (float*)malloc(m * m * sizeof(float));
    h_identity_mtx(*h_Q, m, m);

    // Declare temporary vectors
    float* v;
    float beta;

    for (int j = n - 1; j >= 0; j--) { // iterate over householder vectors stored in lower part of A
        int v_length = m - j; // v is the householder vector, smallest first
        v = (float*)malloc((m - j) * sizeof(float));

        // Q = (Im - 2v(v^T))Q
        // Q = Q_j:m,j:m - 2V @ ((V^T) @ Q_j:m,j:m)

        // (V^T) @ Q_j:m,j:m
        float* temp = (float*)malloc((m - j) * sizeof(float));
        for (int col = j; col < m; col++) {
            float inner_product = 0;
            for (int row = j; row < m; row++) {
                inner_product += h_A[(row + 1) * n + j] * (*h_Q)[row * m + col];
            }
            temp[col - j] = inner_product;
        }

        // Q_j:m,j:m = Q_j:m,j:m - 2 * V @ ((V^T) @ Q_j:m,j:m)
        for (int row = j; row < m; row++) {
            for (int col = j; col < m; col++) {
                (*h_Q)[row * m + col] = (*h_Q)[row * m + col] - 2.0 * h_A[(row + 1) * n + j] * temp[col - j];
            }
        }
    }
}

void h_wy_transform(float* h_A, float** h_Q, int m, int n, int global_offset, int panel_width)
{
    nvtxRangePush(__func__);
    float* W = (float*)malloc((m - global_offset) * panel_width * sizeof(float));
    float* Y = (float*)malloc((m - global_offset) * panel_width * sizeof(float));
    float* z = (float*)malloc((m - global_offset) * sizeof(float));
    float* W_Yt = (float*)malloc((m - global_offset) * (m - global_offset) * sizeof(float));

    // Dimensions of final result Im - WY^T, square
    int W_Yt_dim = m - global_offset;

    // Y = w_1
    for (int i = 0; i < W_Yt_dim; i++) {
        Y[i * panel_width] = h_A[(i + global_offset + 1) * n + global_offset];
        W[i * panel_width] = 2 * h_A[(i + global_offset + 1) * n + global_offset];
    }

    clock_t cycles = clock();

    // Iterate over columns of panel and update W, Y
    for (int i = 1; i < panel_width; i++) { // cols of panel
        // Calculate z = 2 * (I_m - WY^T)w_i
        // Im - WY^T (classic "triply-nested-loop")
        // Flops: (m-global_offset)x(m-global_offset)x(i)
        for (int row = 0; row < W_Yt_dim; row++) { // rows of W_Yt
            int row_offset = row * panel_width;
            for (int col = 0; col < W_Yt_dim; col++) { // cols of W_Yt
                int col_offset = col * panel_width;
                // compute each inner product
                float inner_product = 0;
                for (int idx = 0; idx < i; idx++) { // idx of columns of W
                    inner_product += W[row_offset + idx] * Y[col_offset + idx];
                }
                if (row == col) { // Im is 1
                    W_Yt[row * W_Yt_dim + col] = 1 - inner_product; // Im - WY^T
                }
                else { // Im is zero
                    W_Yt[row * W_Yt_dim + col] = -inner_product;
                }
            }
        }

        // 2 * (Im - WY^T)w_i (matrix-vector product)
        // Flops: (m-global_offset)x(m-global_offset-i)
        for (int row = 0; row < W_Yt_dim; row++) {
            float inner_product = 0;
            for (int col = i; col < W_Yt_dim; col++) {
                inner_product += W_Yt[row * W_Yt_dim + col] * h_A[(global_offset + col + 1) * n + global_offset + i];
            }
            z[row] = 2 * inner_product;
        }

        // Copy z to W
        // Flops: (m-global_offset)
        for (int idx = 0; idx < W_Yt_dim; idx++) {
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
    // Flops: (m-global_offset)x(m-global_offset)xpanel_width
    for (int row = 0; row < W_Yt_dim; row++) { // rows of W_Yt
        for (int col = 0; col < W_Yt_dim; col++) { // cols of W_Yt
            // compute each inner product
            float inner_product = 0;
            for (int idx = 0; idx < panel_width; idx++) { // cols of W
                inner_product += W[row * panel_width + idx] * Y[col * panel_width + idx];
            }
            if (row == col) { // Im is 1
                W_Yt[row * W_Yt_dim + col] = 1 - inner_product; // Im - WY^T
            }
            else { // Im is zero
                W_Yt[row * W_Yt_dim + col] = -inner_product;
            }
        }
    }

    free(W);
    free(Y);
    free(z);
    //free(W_Yt);
    *h_Q = W_Yt;
    nvtxRangePop();
}

template <typename T>
__global__
void dev_cpy_strided_array(T* dest, T* src, int dest_height, int dest_width, int src_height, int src_width, int mode) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int smaller_width;
    int smaller_height;
    int larger_width;
    int larger_height;

    if (src_width < dest_width) {
        smaller_width = src_width;
        larger_width = dest_width;
    }
    else {
        smaller_width = dest_width;
        larger_width = src_width;
    }

    if (src_height < dest_height) {
        smaller_height = src_height;
        larger_height = dest_height;
    }
    else {
        smaller_height = dest_height;
        larger_height = src_height;
    }


    if (mode == TOP_LEFT) {
        if (row < smaller_height && col < smaller_width) {
            dest[row * dest_width + col] = src[row * src_width + col];
        }
        else if (row < dest_height && col < dest_width) {
            dest[row * dest_width + col] = (T)0.0;
        }
    }
    else if (mode == BOTTOM_RIGHT) {
        int row_offset = (larger_height - smaller_height);
        int col_offset = (larger_width - smaller_width);

        if (row >= row_offset && col >= col_offset && row < dest_height && col < dest_width) {
            dest[row * dest_width + col] = src[(row - row_offset) * src_width + col - col_offset];
        }
    }

}

template __global__ void dev_cpy_strided_array<float>(float*, float*, int, int, int, int, int);

template <typename T>
void h_launch_cpy_strided_array(T* h_dest, T* h_src, int dest_height, int dest_width, int src_height, int src_width) {

    // Allocate device memory
    T* dev_dest;
    T* dev_src;

    cudaMalloc(&dev_dest, dest_width * dest_height * sizeof(T));
    cudaMalloc(&dev_src, src_width * src_height * sizeof(T));

    cudaMemcpy(dev_src, h_src, src_width * src_height * sizeof(T), cudaMemcpyHostToDevice);

    // Configure grid of thread blocks
    int grid_height = dest_height / CPY_ARRAY_BLOCK_HEIGHT +
        (dest_height % CPY_ARRAY_BLOCK_HEIGHT != 0); // Integer div. rounded up
    int grid_width = dest_width / CPY_ARRAY_BLOCK_WIDTH +
        (dest_width % CPY_ARRAY_BLOCK_WIDTH != 0);

    dim3 gridDim(grid_width, grid_height, 1);
    dim3 blockDim(CPY_ARRAY_BLOCK_WIDTH, CPY_ARRAY_BLOCK_HEIGHT, 1);
    dev_cpy_strided_array<T> << <gridDim, blockDim >> > (dev_dest, dev_src, dest_height, dest_width, src_height, src_width, TOP_LEFT);

    cudaDeviceSynchronize();

    cudaMemcpy(h_dest, dev_dest, dest_height * dest_width * sizeof(T), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dev_dest);
    cudaFree(dev_src);
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
void dev_wy_compute_z(float* dev_z, float* dev_W_Yt, float* dev_A, int m, int n, int global_offset, int W_Yt_dim, int column_offset)
/*
* Computes vector z for WY representation on the GPU
*
* z = (Im - W @ Yt) @ w
*
* Where w is the householder vector stored in the lower trapezoidal region of matrix A
*/
{
    // thread mapping to row of z
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float inner_product = 0;

    if (row < W_Yt_dim) {
        for (int col = column_offset; col < W_Yt_dim; col++) {
            inner_product += dev_W_Yt[row * W_Yt_dim + col] *
                dev_A[(global_offset + col + 1) * n + global_offset + column_offset];
        }
        dev_z[row] = 2 * inner_product;
    }
}

__global__
void dev_wy_copy_z_and_w(float* dev_z, float* dev_W, float* dev_Y, float* dev_A,
                         int m, int n, int W_Yt_dim, int column_offset, int panel_width, int global_offset) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < W_Yt_dim) {
        // Copy z
        dev_W[row * panel_width + column_offset] = dev_z[row];
        // Copy householder vector w from matrix A
        if (row < column_offset) {
            dev_Y[row * panel_width + column_offset] = 0.0;
        }
        else {
            dev_Y[row * panel_width + column_offset] = dev_A[(global_offset + row + 1) * n + global_offset + column_offset];
        }
    }
}

__global__ void shared_mem_mmult_transpose_b(float* c_mtx, float* a_mtx, float* b_mtx,
                                             int a_width, int a_height, int b_width,
                                             int b_layout) {
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
            bds[tx][ty] = b_mtx[(i * TILE_WIDTH + col) * a_width + ty];
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
void shared_mem_mmult_in_place(float* c_mtx, float* a_mtx, float* b_mtx, int m, int n, int k, int b_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB using shared memory with CUDA
*
* Dimensions:
* A : m x k
* B : b_height x b_width => operate on bottom-right corner k x n submatrix
* C : m x n
*/
{
    // row and column of C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // offsets for 'in-place' read from b matrix
    int b_row_offset = b_height - k;
    int b_col_offset = b_width - n;

    __shared__ float ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bds[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(k / (float)TILE_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * TILE_WIDTH + tx < k) && (row < m)) {
            ads[ty][tx] = a_mtx[row * k + i * TILE_WIDTH + tx];
        }

        if ((i * TILE_WIDTH + ty < k) && (col < n)) {
            bds[ty][tx] = b_mtx[(i * TILE_WIDTH + ty + b_row_offset) * b_width + (col + b_col_offset)];
        }

        __syncthreads();

        for (int idx = 0; idx < TILE_WIDTH; idx++) {
            if ((i * TILE_WIDTH + idx) < k)
                pval += ads[ty][idx] * bds[idx][tx];
        }
        __syncthreads();
    }

    if (col < n && row < m) {
        c_mtx[row * n + col] = pval;
    }

    __syncthreads();
}

__global__
void shared_mem_mmult_in_place_transpose_a(float* c_mtx, float* a_mtx, float* b_mtx, int m, int n, int k, int b_height, int b_width)
/*
* Computes result c matrix from the matrix multiplication C = AB using shared memory
*
* assumed m = k
*
* Dimensions:
* A : m x k
* B : b_height x b_width => operate on bottom-right corner k x n submatrix``1`1~!`
* C : m x n
*/
{
    // row and column of C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // offsets for 'in-place' read from b matrix
    int b_row_offset = b_height - k;
    int b_col_offset = b_width - n;

    __shared__ float ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bds[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(k / (float)TILE_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * TILE_WIDTH + tx < m) && (row < k)) {
            ads[tx][ty] = a_mtx[(i * TILE_WIDTH + tx) * k + row];
        }

        if ((i * TILE_WIDTH + ty < k) && (col < n)) {
            bds[ty][tx] = b_mtx[(i * TILE_WIDTH + ty + b_row_offset) * b_width + (col + b_col_offset)];
        }

        __syncthreads();

        for (int idx = 0; idx < TILE_WIDTH; idx++) {
            if ((i * TILE_WIDTH + idx) < m)
                pval += ads[idx][ty] * bds[idx][tx];
        }
        __syncthreads();
    }

    if (col < n && row < m) {
        c_mtx[row * n + col] = pval;
    }

    __syncthreads();
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

template <typename T_A, typename T_B, typename T_C>
__global__
void dev_tensorcore_mmult_tiled(T_C* c_mtx, T_A* a_mtx, T_B* b_mtx, int m, int n, int k) {
    /*
    * Tiled matrix multiply using warp matrix multiply-accumulate (wmma)
    *
    * The output matrix is divided into tiles (M-N-K), where each warp is responsible for computing one output tile
    */

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    using namespace nvcuda;

    // Determine warp index
    int warp_x = x / WARP_SIZE;
    int warp_y = y;

    int tiles_x = n / TC_TILE_N + (n % TC_TILE_N != 0);
    int tiles_y = m / TC_TILE_M + (m % TC_TILE_M != 0);
    int num_phases = k / TC_TILE_K + (k % TC_TILE_K != 0);

    if (warp_x < tiles_x && warp_y < tiles_y) {
        // Create fragments
        wmma::fragment<wmma::matrix_a, TC_TILE_M, TC_TILE_N, TC_TILE_K, T_A, wmma::row_major> Amat;
        wmma::fragment<wmma::matrix_b, TC_TILE_M, TC_TILE_N, TC_TILE_K, T_B, wmma::row_major> Bmat;
        wmma::fragment<wmma::accumulator, TC_TILE_M, TC_TILE_N, TC_TILE_K, T_C, void> Cmat;

        // Initialize output to zero
        wmma::fill_fragment(Cmat, 0.0f);

        // Compute tiled matrix multiply for warp
        for (int phase = 0; phase < num_phases; phase++) {
            // Load inputs
            T_A* a_idx = &a_mtx[warp_y * TC_TILE_M * k + phase * TC_TILE_K];
            T_B* b_idx = &b_mtx[phase * n * TC_TILE_K + warp_x * TC_TILE_N];

            wmma::load_matrix_sync(Amat, a_idx, k);
            wmma::load_matrix_sync(Bmat, b_idx, n);

            // Perfrom matrix multiplication, accumulate into C
            wmma::mma_sync(Cmat, Amat, Bmat, Cmat);
        }

        // Write output
        T_C* c_idx = &c_mtx[warp_y * n * TC_TILE_M + warp_x * TC_TILE_N];
        wmma::store_matrix_sync(c_idx, Cmat, n, wmma::mem_row_major);
    }
}

__global__
void dev_panel_wy_transform(float* dev_A, float* dev_Q, int m, int n, int global_offset, int panel_width) {

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



void test_dev_smem_mmult(int m, int n, int k, int b_layout) {
    printf("\nTesting GPU SMEM tiled mmult %dx%dx%d...\n", m, n, k);

    float* a_mtx = (float*)malloc(m * k * sizeof(float));
    float* b_mtx = (float*)malloc(k * n * sizeof(float));
    float* c_mtx = (float*)malloc(m * n * sizeof(float));

    // initialize matrix A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a_mtx[i * k + j] = (float)(float)j;
        }
    }

    if (b_layout == ROW_MAJOR) {
        // initialize matrix B
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                b_mtx[i * n + j] = (float)(float)j;
            }
        }
    }
    else {
        // initialize matrix B
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                b_mtx[i * k + j] = (float)i;
            }
        }
    }


    // initialize matrix C
    memset(c_mtx, 0.0, m * n * sizeof(float));

    // Allocate device memory
    float* dev_a;
    float* dev_b;
    float* dev_c;

    cudaMalloc(&dev_a, m * k * sizeof(float));
    cudaMalloc(&dev_b, k * n * sizeof(float));
    cudaMalloc(&dev_c, m * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a_mtx, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_mtx, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c_mtx, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim((int)ceil((float)n / TILE_WIDTH), (int)ceil((float)m / TILE_WIDTH), 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // one warp

    shared_mem_mmult_transpose_b << <gridDim, blockDim >> > (dev_c, dev_a, dev_b, k, m, n, b_layout);

    cudaDeviceSynchronize();

    cudaMemcpy(c_mtx, dev_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float row_sum = 0;
    for (int i = 0; i < k; i++) {
        row_sum += i;
    }

    // test result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            assert(abs(c_mtx[i * n + j] - (j * row_sum)) <= 1E-7 * row_sum * j * m);
        }
    }

    printf("Test passed.\n");
}

void test_dev_smem_mmult(int m, int n, int k) {
    printf("\nTesting GPU SMEM tiled mmult %dx%dx%d...\n", m, n, k);

    float* a_mtx = (float*)malloc(m * k * sizeof(float));
    float* b_mtx = (float*)malloc(k * n * sizeof(float));
    float* c_mtx = (float*)malloc(m * n * sizeof(float));

    // initialize matrix A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a_mtx[i * k + j] = (float)(float)j;
        }
    }

    // initialize matrix B
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b_mtx[i * n + j] = (float)(float)j;
        }
    }

    // initialize matrix C
    memset(c_mtx, 0.0, m * n * sizeof(float));

    // Allocate device memory
    float* dev_a;
    float* dev_b;
    float* dev_c;

    cudaMalloc(&dev_a, m * k * sizeof(float));
    cudaMalloc(&dev_b, k * n * sizeof(float));
    cudaMalloc(&dev_c, m * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a_mtx, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_mtx, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c_mtx, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim((int)ceil((float)n / TILE_WIDTH), (int)ceil((float)m / TILE_WIDTH), 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // one warp

    shared_mem_mmult << <gridDim, blockDim >> > (dev_c, dev_a, dev_b, k, m, n);

    cudaDeviceSynchronize();

    cudaMemcpy(c_mtx, dev_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float row_sum = 0;
    for (int i = 0; i < k; i++) {
        row_sum += i;
    }

    // test result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            assert(abs(c_mtx[i * n + j] - (j * row_sum)) <= 1E-7 * row_sum * j * m);
        }
    }

    printf("Test passed.\n");
}

void test_dev_smem_mmult_in_place(int m, int n, int k, int b_width, int b_height) {
    /*
    * Computes C = A @ B', where b' (mxk) is stored in the "bottom-right" submatrix of a larger matrix B
    * (b_height x b_width)
    */
    printf("\nTesting GPU SMEM tiled mmult (in-place) %dx%dx%d...\n", m, n, k);

    float* a_mtx = (float*)malloc(m * k * sizeof(float));
    float* b_mtx = (float*)malloc(b_height * b_width * sizeof(float));
    float* c_mtx = (float*)malloc(m * n * sizeof(float));

    // initialize matrix A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a_mtx[i * k + j] = (float)(float)j;
        }
    }

    memset(b_mtx, 0.0, b_width * b_height * sizeof(float));

    // initialize matrix B
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b_mtx[(i + (b_height - k)) * b_width + (j + (b_width - n))] = (float)(float)j;
        }
    }

    // initialize matrix C
    memset(c_mtx, 0.0, m * n * sizeof(float));

    // Allocate device memory
    float* dev_a;
    float* dev_b;
    float* dev_c;

    cudaMalloc(&dev_a, m * k * sizeof(float));
    cudaMalloc(&dev_b, b_height * b_width * sizeof(float));
    cudaMalloc(&dev_c, m * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a_mtx, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_mtx, b_height * b_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c_mtx, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim((int)ceil((float)n / TILE_WIDTH), (int)ceil((float)m / TILE_WIDTH), 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // one warp

    shared_mem_mmult_in_place << <gridDim, blockDim >> > (dev_c, dev_a, dev_b, m, n, k, b_height, b_width);

    cudaDeviceSynchronize();

    dim3 gridDim2((int)ceil((float)b_width / TILE_WIDTH), (int)ceil((float)b_height / TILE_WIDTH), 1);
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);
    dev_cpy_strided_array<float> << <gridDim2, blockDim2 >> >(dev_b, dev_c, b_height, b_width, m, n, BOTTOM_RIGHT);

    cudaDeviceSynchronize();

    cudaMemcpy(b_mtx, dev_b, b_height * b_width * sizeof(float), cudaMemcpyDeviceToHost);

    float row_sum = 0;
    for (int i = 0; i < k; i++) {
        row_sum += i;
    }

    // test result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            assert(abs(b_mtx[(i + (b_height - k)) * b_width + j + (b_width - n)] - (j * row_sum)) <= 1E-7 * row_sum * j * m);
        }
    }

    printf("Test passed.\n");
}

void test_dev_smem_mmult_in_place_transpose_a(int m, int n, int k, int b_width, int b_height) {
    /*
    * Computes C = A^T @ B', where b' (mxk) is stored in the "bottom-right" submatrix of a larger matrix B
    * (b_height x b_width), and A is transposed in memory
    */
    printf("\nTesting GPU SMEM tiled mmult (in-place & transposed A) %dx%dx%d...\n", m, n, k);

    float* a_mtx = (float*)malloc(m * k * sizeof(float));
    float* b_mtx = (float*)malloc(b_height * b_width * sizeof(float));
    float* c_mtx = (float*)malloc(m * n * sizeof(float));

    // initialize matrix A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a_mtx[i * k + j] = (float)(float)i;
        }
    }

    memset(b_mtx, 0.0, b_width * b_height * sizeof(float));

    // initialize matrix B
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b_mtx[(i + (b_height - k)) * b_width + (j + (b_width - n))] = (float)(float)j;
        }
    }

    // initialize matrix C
    memset(c_mtx, 0.0, m * n * sizeof(float));

    // Allocate device memory
    float* dev_a;
    float* dev_b;
    float* dev_c;

    cudaMalloc(&dev_a, m * k * sizeof(float));
    cudaMalloc(&dev_b, b_height * b_width * sizeof(float));
    cudaMalloc(&dev_c, m * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a_mtx, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_mtx, b_height * b_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c_mtx, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim((int)ceil((float)n / TILE_WIDTH), (int)ceil((float)m / TILE_WIDTH), 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // one warp

    shared_mem_mmult_in_place_transpose_a << <gridDim, blockDim >> > (dev_c, dev_a, dev_b, m, n, k, b_height, b_width);

    cudaDeviceSynchronize();

    dim3 gridDim2((int)ceil((float)b_width / TILE_WIDTH), (int)ceil((float)b_height / TILE_WIDTH), 1);
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);
    dev_cpy_strided_array<float> << <gridDim2, blockDim2 >> > (dev_b, dev_c, b_height, b_width, m, n, BOTTOM_RIGHT);

    cudaDeviceSynchronize();

    cudaMemcpy(b_mtx, dev_b, b_height * b_width * sizeof(float), cudaMemcpyDeviceToHost);

    float row_sum = 0;
    for (int i = 0; i < k; i++) {
        row_sum += i;
    }

    // test result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            assert(abs(b_mtx[(i + (b_height - k)) * b_width + j + (b_width - n)] - (j * row_sum)) <= 1E-7 * row_sum * j * m);
        }
    }

    printf("Test passed.\n");
}



template <typename T_A, typename T_B, typename T_C>
void h_launch_dev_tensorcore_mmult_tiled(T_A* a_mtx, T_B* b_mtx, T_C* c_mtx, int m, int n, int k) {
    /*
    * Performs tiled matrix multiply C = A@B with TensorCore
    *
    * Dimensions of A: mxk
    * Dimensions of B: kxn
    * Dimensions of C: mxn
    */

    // Allocation size must be integer multiple of TC tile size
    int m_padded = (m % TC_TILE_M) ? m + (TC_TILE_M - m % TC_TILE_M): m; // Padded height of A & C
    int n_padded = (n % TC_TILE_N) ? n + (TC_TILE_N - n % TC_TILE_N): n; // Padded width of B & C
    int k_padded = (k % TC_TILE_K) ? k + (TC_TILE_K - k % TC_TILE_K): k; // Padded inner dimension

    // Matrix sizes in bytes
    size_t a_bytes = m_padded * k_padded * sizeof(T_A);
    size_t b_bytes = k_padded * n_padded * sizeof(T_B);
    size_t c_bytes = m_padded * n_padded * sizeof(T_C);

    // Allocate host-side padded arrays
    T_A* h_a = (T_A*)malloc(a_bytes);
    T_B* h_b = (T_B*)malloc(b_bytes);
    T_C* h_c = (T_C*)malloc(c_bytes);

    // Pad arrays
    h_launch_cpy_strided_array<T_A>(h_a, a_mtx, m_padded, k_padded, m, k);
    h_launch_cpy_strided_array<T_B>(h_b, b_mtx, k_padded, n_padded, k, n);
    // Set output to zeros
    memset(h_c, 0, c_bytes);

    // Allocate input & output matrices on device
    T_A* dev_a;
    T_B* dev_b;
    T_C* dev_c;

    cudaMalloc(&dev_a, a_bytes);
    cudaMalloc(&dev_b, b_bytes);
    cudaMalloc(&dev_c, c_bytes);

    // Copy matrices from host to device
    cudaMemcpy(dev_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, h_c, c_bytes, cudaMemcpyHostToDevice);

    // Configure grid of "warp blocks" which overlay output C
    int warp_grid_height = m / TC_TILE_M + (m % TC_TILE_M != 0);
    int warp_grid_width = n / TC_TILE_N + (n % TC_TILE_N != 0);

    // Configure grid of thread blocks
    int grid_height = warp_grid_height / TC_MMULT_THREAD_BLOCK_HEIGHT +
                      (warp_grid_height % TC_MMULT_THREAD_BLOCK_HEIGHT != 0); // Integer div. rounded up
    int grid_width = warp_grid_width * WARP_SIZE / TC_MMULT_THREAD_BLOCK_WIDTH +
                     ((warp_grid_width * WARP_SIZE) % TC_MMULT_THREAD_BLOCK_WIDTH != 0);

    // Configure grid
    dim3 gridDim(grid_width, grid_height, 1);
    dim3 blockDim(TC_MMULT_THREAD_BLOCK_WIDTH, TC_MMULT_THREAD_BLOCK_HEIGHT, 1);

    dev_tensorcore_mmult_tiled<T_A, T_B, T_C> << <gridDim, blockDim >> > (dev_c, dev_a, dev_b, m_padded, n_padded, k_padded);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, dev_c, c_bytes, cudaMemcpyDeviceToHost);

    h_launch_cpy_strided_array<T_C>(c_mtx, h_c, m, n, m_padded, n_padded);
}

template void h_launch_dev_tensorcore_mmult_tiled<half, half, float>(half*, half*, float*, int, int, int);

__global__
void dev_wy_init(float* dev_A, float* dev_Y, float* dev_W, int global_offset, int n, int W_Yt_dim, int panel_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int linearIdx = row * TILE_WIDTH + col;

    if (linearIdx < W_Yt_dim) {
        dev_Y[linearIdx * panel_width] = dev_A[(linearIdx + global_offset + 1) * n + global_offset];
        dev_W[linearIdx * panel_width] = 2 * dev_A[(linearIdx + global_offset + 1) * n + global_offset];
    }
}

__global__
void dev_wy_compute_Im_sub_W_Yt(float* dev_W_Yt, float* dev_W, float* dev_Y,
                                int panel_width, int column_idx, int W_Yt_dim) {
    // row and column of C result
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bds[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y, tx = threadIdx.x;

    int phases = ceil(panel_width / (float)TILE_WIDTH);

    float pval = 0.0;
    for (int i = 0; i < phases; i++) {
        if ((i * TILE_WIDTH + tx < panel_width) && (row < W_Yt_dim)) {
            ads[ty][tx] = dev_W[row * panel_width + i * TILE_WIDTH + tx];
        }

        if ((i * TILE_WIDTH + ty < panel_width) && (col < W_Yt_dim)) {
            bds[tx][ty] = dev_Y[col * panel_width + (i * TILE_WIDTH) + ty];
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            if ((i * TILE_WIDTH + k) < column_idx)
                pval += ads[ty][k] * bds[tx][k];
        }
        __syncthreads();
    }

    if (col < W_Yt_dim && row < W_Yt_dim) {
        if (row == col) {
            dev_W_Yt[row * W_Yt_dim + col] = 1 - pval;
        }
        else {
            dev_W_Yt[row * W_Yt_dim + col] = -pval;
        }

    }

}

#define VECTOR_OP_1D_BLOCK_WIDTH 64

void dev_wy_transform(float* dev_A, float** dev_panel_Q, int m, int n, int global_offset, int panel_width) {
    nvtxRangePush(__func__);
    float* dev_W;
    float* dev_Y;
    float* dev_z;
    float* dev_W_Yt;

    cudaMalloc(&dev_W, (m - global_offset) * panel_width * sizeof(float));
    cudaMalloc(&dev_Y, (m - global_offset) * panel_width * sizeof(float));
    cudaMalloc(&dev_z, (m - global_offset) * sizeof(float));
    cudaMalloc(&dev_W_Yt, (m - global_offset) * (m - global_offset) * sizeof(float));

    // Dimensions of final result Im - WY^T, square
    int W_Yt_dim = m - global_offset;

    // Y = w_1
    // W = 2 * w_1
    // Configure grid of thread blocks
    int grid_dim = W_Yt_dim / VECTOR_OP_1D_BLOCK_WIDTH +
        (W_Yt_dim % VECTOR_OP_1D_BLOCK_WIDTH != 0); // Integer div. rounded up
    dim3 gridDim(grid_dim, 1, 1);
    dim3 blockDim(VECTOR_OP_1D_BLOCK_WIDTH, 1, 1);

    // Grid for matrix-matrix products
    dim3 mtx_gridDim((int)ceil((float)W_Yt_dim / TILE_WIDTH), (int)ceil((float)W_Yt_dim / TILE_WIDTH), 1);
    dim3 mtx_blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // CUDA grid for vector operations
    dim3 vec_gridDim(1, (int)ceil((float)W_Yt_dim / VECTOR_OP_1D_BLOCK_WIDTH), 1);
    dim3 vec_blockDim(1, VECTOR_OP_1D_BLOCK_WIDTH, 1);

    dev_wy_init << <gridDim, blockDim >> > (dev_A, dev_Y, dev_W, global_offset, n, W_Yt_dim, panel_width);
    cudaDeviceSynchronize();

    // Iterate over columns of panel and update W, Y
    for (int i = 1; i < panel_width; i++) { // cols of panel

        /*
         * Calculate z = 2 * (I_m - WY ^ T)w_i
         *  for each column in the panel using 3 CUDA kernels
         */

        // (Im - WY^T)
        dev_wy_compute_Im_sub_W_Yt<<<mtx_gridDim, mtx_blockDim >>>(dev_W_Yt, dev_W, dev_Y, panel_width, i, W_Yt_dim);
        cudaDeviceSynchronize();

        // 2 * (Im - WY^T)w_i (matrix-vector product)
        dev_wy_compute_z<<<vec_gridDim, vec_blockDim >>>(dev_z, dev_W_Yt, dev_A, m, n, global_offset, W_Yt_dim, i);
        cudaDeviceSynchronize();

        // Copy z & householder vector (w) to W & Y matrices
        dev_wy_copy_z_and_w << <vec_gridDim, vec_blockDim >> > (dev_z, dev_W, dev_Y, dev_A, m, n, W_Yt_dim, i, panel_width, global_offset);
        cudaDeviceSynchronize();
    }

    // Im - WY^T
    dev_wy_compute_Im_sub_W_Yt<<<mtx_gridDim, mtx_blockDim>>>(dev_W_Yt, dev_W, dev_Y, panel_width, panel_width, W_Yt_dim);
    cudaDeviceSynchronize();

    cudaFree(dev_W);
    cudaFree(dev_Y);
    cudaFree(dev_z);
    //free(W_Yt);
    *dev_panel_Q = dev_W_Yt;
    nvtxRangePop();
}

void test_template_tensorcore_mmult_tiled(int m, int n, int k) {
    printf("\nTesting template tensorcore tiled mmult (m, n, k) = (%dx%dx%d)...\n", m, n, k);

    // Initialize random matrices
    __half* a_mtx = h_generate_random_matrix<__half>(m, k);
    __half* b_mtx = h_generate_random_matrix<__half>(k, n);
    float* c_mtx_dev_result = (float*)malloc(m * n * sizeof(float));
    float* c_mtx_h_result = (float*)malloc(m * n * sizeof(float));

    memset(c_mtx_dev_result, 0.0, m * n * sizeof(float));
    memset(c_mtx_h_result, 0.0, m * n * sizeof(float));

    h_launch_dev_tensorcore_mmult_tiled<half, half, float>(a_mtx, b_mtx, c_mtx_dev_result, m, n, k);

    h_mmult<__half, __half, float>(a_mtx, b_mtx, c_mtx_h_result, m, n, k);

    // test result
    bool pass = true;
    float max_error = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float error = abs(c_mtx_h_result[i * n + j] - c_mtx_dev_result[i * n + j]);
            if (error > 5E-4) {
                pass = false;
                //printf("Error exceeded %.2E limit\n", 5E-4);
            }
            if (error > max_error) max_error = error;
        }
    }

    if (pass) {
        printf("Test passed. Max error: %.2E\n", max_error);
    }
    else {
        printf("Test failed. Max error: %.2E exceeded %.2E limit\n", max_error, 5E-4);
    }
}

void test_tensorcore_mmult_tiled() {
    printf("\nTesting tensorcore tiled mmult 32x32x32...\n");

    __half* a_mtx = (__half*)malloc(32 * 32 * sizeof(__half));
    __half* b_mtx = (__half*)malloc(32 * 32 * sizeof(__half));
    float* c_mtx = (float*)malloc(32 * 32 * sizeof(float));

    // initialize matrices A, B, C
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            a_mtx[i * 32 + j] = (__half)(float)j;
            b_mtx[i * 32 + j] = (__half)(float)j;
            c_mtx[i * 32 + j] = (__half)0.0f;
        }
    }

    // Allocate device memory
    __half* dev_a;
    __half* dev_b;
    float* dev_c;

    cudaMalloc(&dev_a, 32 * 32 * sizeof(__half));
    cudaMalloc(&dev_b, 32 * 32 * sizeof(__half));
    cudaMalloc(&dev_c, 32 * 32 * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a_mtx, 32 * 32 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b_mtx, 32 * 32 * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c_mtx, 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(64, 2, 1); // one warp

    dev_tensorcore_mmult_tiled << <gridDim, blockDim >> > (dev_c, dev_b, dev_a, 32, 32, 32);

    cudaDeviceSynchronize();

    cudaMemcpy(c_mtx, dev_c, 32 * 32 * sizeof(float), cudaMemcpyDeviceToHost);

    // test result
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            assert(c_mtx[i * 32 + j] == j * 496);
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




void read_euroc_jacobian(std::string filename, int* rows, int* cols, float** matrix) {
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

//     std::cout << line << std::endl;
    int start = line.find(" ");
    int end = line.find(" ");

    std::string rows_str = line.substr(0, start);
    std::string cols_str = line.substr(start + 1, end);

//     std::cout << rows_str << std::endl;
//     std::cout << cols_str << std::endl;

    *cols = std::stoi(cols_str);
    *rows = std::stoi(rows_str);

//     printf("Rows: %d\nCols: %d\n", *rows, *cols);

    // Allocate memory for matrix
    *matrix = (float*)malloc((*rows) * (*cols) * sizeof(float));

    for (int row = 0; row < (*rows); row++) {
        for (int col = 0; col < (*cols); col++) {
            (*matrix)[row * (*cols) + col] = (float)0.0;
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

//     printf("Total linecount: %d\n", linecount);
}




__global__
void dev_apply_qt_to_a(float* dev_A, float* dev_panel_Q, float* res_A, int m, int n, int tau, int lambda) {
    // Collaboratively load householder vectors vk from global memory to shared memory
    // Construct W, K from householder vectors
    // Construct Q
    // Collaboratively load matrix A to shared memory
    // Perform tiled GMMULT TensorCore warp-level mixed precision fused multiply add operations to update Q and A
    // Update matrix Q, A in global memory

    __shared__ float a_smem_tile[GMEM_MMULT_TILE_WIDTH][GMEM_MMULT_TILE_WIDTH];
    __shared__ float panel_q_smem_tile[GMEM_MMULT_TILE_WIDTH][GMEM_MMULT_TILE_WIDTH];

    // Row and column of the output result (A)
    int row = blockIdx.y * blockDim.y + threadIdx.y + lambda;
    int col = blockIdx.x * blockDim.x + threadIdx.x + tau;

    int panel_q_dim = (m - lambda); // panel_q is square matrix, shrinks for subsequent panels

    // Number of phases determined from block width
    int phases = panel_q_dim % GMEM_MMULT_TILE_WIDTH == 0 ?
        panel_q_dim / GMEM_MMULT_TILE_WIDTH : panel_q_dim / GMEM_MMULT_TILE_WIDTH + 1;

    // Traverse phases and perform matrix-multiply accumulate into inner_product for each thread

    // check thread maps to output matrix
    bool valid_row = (row >= lambda && row < m);
    bool valid_col = (col >= tau && col < n);

    // panel_Q[(inner_dim) * (m - lambda) + (row - lambda)] * A_old[(inner_dim + lambda) * n + col];

    float inner_product = 0;
    for (int p = 0; p < phases; p++) {
        // Check index doesn't exceed input bounds
        int a_idx_x = col;
        int a_idx_y = p * GMEM_MMULT_TILE_HEIGHT + row;
        int q_idx_x = (row - lambda);
        int q_idx_y = p * GMEM_MMULT_TILE_HEIGHT + (row-lambda);

        bool valid_idx_a = (a_idx_y < m);
        bool valid_idx_q = (q_idx_y < panel_q_dim);

        if (valid_idx_a && valid_idx_q && valid_row && valid_col) {
            // Collaboratively load data into smem
            a_smem_tile[threadIdx.y][threadIdx.x] = dev_A[a_idx_y * n + a_idx_x];
            panel_q_smem_tile[threadIdx.y][threadIdx.x] = dev_panel_Q[q_idx_y * panel_q_dim + q_idx_x];
        }

        __syncthreads();

        // Accumulate tile inner product
        if (valid_idx_a && valid_idx_q && valid_row && valid_col) {
            for (int i = 0; i < GMEM_MMULT_TILE_WIDTH; i++) {
                inner_product += panel_q_smem_tile[i][threadIdx.y] * a_smem_tile[i][threadIdx.x];
            }
        }

        __syncthreads();
    }


}

__global__
void dev_apply_qpanel_to_q(float* dev_Q, float* dev_Q_panel, float* dev_Q_result, int m, int lambda) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x + lambda;

    if (row >= 0 && row < m && col >= lambda && col < m) {
        float inner_product = 0;
        for (int inner_dim = 0; inner_dim < (m - lambda); inner_dim++) {
            inner_product += dev_Q[row * m + inner_dim + lambda] * dev_Q_panel[(inner_dim * (m - lambda)) + (col - lambda)];
        }
        dev_Q_result[row * m + col] = inner_product;
    }
}

__global__
void dev_apply_qpanel_to_q_tensorcore(float* dev_Q, float* dev_Q_panel, float* dev_Q_result, int m, int lambda) {

}

__global__
void dev_cpy_panel_result_a(float* dev_A, float* dev_A_panel_result, int m, int n, int tau, int lambda) {
    // Row and column of the output result (A)
    int row = blockIdx.y * blockDim.y + threadIdx.y + lambda;
    int col = blockIdx.x * blockDim.x + threadIdx.x + tau;

    int panel_a_height = (m - lambda); // panel_q is square matrix, shrinks for subsequent panels
    int panel_a_width = (n - tau);

    dev_A[row * n + col] = dev_A_panel_result[(row - lambda) * (panel_a_width) + (col - tau)];

    __syncthreads();
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
        h_householder_qr(A, m, n, lambda, tau-lambda);

        // Get panel Q from factors - dim panel_Q: (m-lambda)x(m-lambda)
        h_wy_transform(A, &panel_Q, m, n, lambda, tau-lambda); // TASK10 3 shashank: write cuda kernel to implement WY transform on GPU

        // Update matrix A = Q^T @ A
        float blockWidth = 32.;
        float blockHeight = 32.;

        float* dev_A;
        float* dev_Q;
        float* dev_panel_Q;
        float* dev_A_panel_result;
        float* dev_Q_result;

        cudaMalloc(&dev_A, m * n * sizeof(float));
        cudaMalloc(&dev_Q, m * m * sizeof(float));
        cudaMalloc(&dev_panel_Q, (m - lambda) * (m - lambda) * sizeof(float));
        cudaMalloc(&dev_A_panel_result, (m - lambda) * (n - tau) * sizeof(float));
        cudaMalloc(&dev_Q_result, m * m * sizeof(float));

        cudaMemcpy(dev_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_panel_Q, panel_Q, (m - lambda) * (m - lambda) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Q, Q, m * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Q_result, Q, m * m * sizeof(float), cudaMemcpyHostToDevice);

        dim3 BlockDim((int)blockWidth, (int)blockHeight, 1);
        dim3 GridDim(ceil((n - tau) / blockWidth), ceil((m - lambda) / blockHeight), 1);

        // Updates trailing matrix in place : A = Qt @ A
        shared_mem_mmult_in_place_transpose_a<<<GridDim, BlockDim>>>(dev_A_panel_result, dev_panel_Q, dev_A,
                                                        (m - lambda), (n - tau), (m - lambda), m, n);

        cudaDeviceSynchronize();

        dim3 gridDim2((int)ceil((float)n / TILE_WIDTH), (int)ceil((float)m / TILE_WIDTH), 1);
        dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);
        dev_cpy_strided_array<float> << <gridDim2, blockDim2 >> > (dev_A, dev_A_panel_result, m, n,
                                                                  (m - lambda), (n - tau), BOTTOM_RIGHT);

        cudaDeviceSynchronize();

        dim3 BlockDim3((int)blockWidth, (int)blockHeight, 1);
        dim3 GridDim3(ceil((m - lambda) / blockWidth), ceil((m) / blockHeight), 1);
        dev_apply_qpanel_to_q << <GridDim3, BlockDim3 >> >(dev_Q, dev_panel_Q, dev_Q_result, m, lambda);

        cudaDeviceSynchronize();

        cudaMemcpy(A, dev_A, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Q, dev_Q_result, m * m * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dev_A);
        cudaFree(dev_Q);
        cudaFree(dev_panel_Q);
        cudaFree(dev_A_panel_result);
        cudaFree(dev_Q_result);

        free(panel_Q);

        // increment panel offset
        lambda = tau;
    }
}

void dev_block_qr_wy(float* A, float* Q, int m, int n, int r) {
    /*
    * GPU code to compute QR decomposition with Block QR algorithm
    */

    // Pointers to device data
    float* dev_A;
    float* dev_Q;
    float* dev_panel_Q;
    float* dev_A_panel_result;
    float* dev_Q_result;

    // Data sizes
    size_t size_A = (m + 1) * n * sizeof(float);
    size_t size_Q = (m * m * sizeof(float));

    // Allocate memory on device
    cudaMalloc(&dev_A, size_A);
    cudaMalloc(&dev_Q, size_Q);
    cudaMalloc(&dev_Q_result, size_Q);

    // Move matrix A and Q (initialized as identity) to device
    cudaMemcpy(dev_Q, Q, size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Q_result, Q, size_Q, cudaMemcpyHostToDevice);

    int lambda = 0;
    while (lambda < n) { // panel starts at lambda
        int tau = (lambda + r < n) ? (lambda + r) : n; // panel ends at tau

        // Q is stored in factored form in lower triangular portion of dev_A
        // R is stored in upper triangular portion of dev_A
        h_householder_qr(A, m, n, lambda, tau - lambda);

        cudaMemcpy(dev_A, A, size_A, cudaMemcpyHostToDevice);

        // Allocate memory for A result
        size_t size_panel_A = (m - lambda) * (n - tau) * sizeof(float);
        cudaMalloc(&dev_A_panel_result, size_panel_A);

        // Perform WY transform to construct Q for panel from householder vectors stored in matrix A
        dev_wy_transform(dev_A, &dev_panel_Q, m, n, lambda, tau - lambda);

        // Update trailing matrix in place : A = Qt @ A
        float blockWidth = 32.;
        float blockHeight = 32.;
        dim3 BlockDim((int)blockWidth, (int)blockHeight, 1);
        dim3 GridDim(ceil((n - tau) / blockWidth), ceil((m - lambda) / blockHeight), 1);
        shared_mem_mmult_in_place_transpose_a << <GridDim, BlockDim >> > (dev_A_panel_result, dev_panel_Q, dev_A,
            (m - lambda), (n - tau), (m - lambda), m, n);
        cudaDeviceSynchronize();

        // Copy panel A result to matrix A
        dim3 gridDim2((int)ceil((float)n / TILE_WIDTH), (int)ceil((float)m / TILE_WIDTH), 1);
        dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);
        dev_cpy_strided_array<float> << <gridDim2, blockDim2 >> > (dev_A, dev_A_panel_result, m, n,
            (m - lambda), (n - tau), BOTTOM_RIGHT);
        cudaDeviceSynchronize();

        // Update Q matrix with panel Q
        dim3 BlockDim3((int)blockWidth, (int)blockHeight, 1);
        dim3 GridDim3(ceil((m - lambda) / blockWidth), ceil((m) / blockHeight), 1);
        dev_apply_qpanel_to_q << <GridDim3, BlockDim3 >> > (dev_Q, dev_panel_Q, dev_Q_result, m, lambda);
        cudaDeviceSynchronize();

        // Overwrite Q with result
        cudaMemcpy(dev_Q, dev_Q_result, size_Q, cudaMemcpyDeviceToDevice);

        cudaFree(dev_panel_Q);
        cudaFree(dev_A_panel_result);
        cudaMemcpy(A, dev_A, size_A, cudaMemcpyDeviceToHost);

        // increment panel offset
        lambda = tau;
    }

    cudaMemcpy(Q, dev_Q, size_Q, cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_Q);
    cudaFree(dev_Q_result);
}


void test_dev_householder_qr(int m, int n, int r) {
    printf("\nTesting GPU householder QR...\n");
    printf("Dimensions of A: %dx%d\n", m, n);

    float* h_A_in = h_generate_random_matrix<float>(m, n);

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
    clock_t cycles = clock(); // Time how long the QR function takes to execute
    dev_householder_qr <<<DimGrid, DimBlock >> > (dev_A, m, n, 0);
    cudaDeviceSynchronize();
    cycles = clock() - cycles;
    float time_ms = cycles * 1000 / CLOCKS_PER_SEC;
    float flops = h_qr_flops_per_second(time_ms, m, n);

    cudaMemcpy(h_A_out, dev_A, (m+1) * n * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_Q_out, dev_Q, m * m * sizeof(float), cudaMemcpyDeviceToHost);

    h_q_backward_accumulation(h_A_out, &h_Q_out, m, n);
    //h_wy_transform(h_A_out, &h_Q_out, m, n, 0, n);

    h_strip_R_from_A((float*)h_A_out, h_R, m, n);

    float backward_error = h_backward_error((float*)h_A_in, h_R, h_Q_out, m, n);
    float error3 = h_error_3(h_R, m, n);
    float error2 = h_error_2(h_Q_out, m);

    printf("GPU householder QR finished in %.2f ms...\n", time_ms);
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
        h_householder_qr(A, m, n, lambda, tau-lambda);

        // Get panel Q from factors
        h_wy_transform(A, &panel_Q, m, n, lambda, tau-lambda); // dim panel_Q: (m-lambda)x(m-lambda)

        // Update matrix A = Q^T @ A
        float* A_old = (float*)malloc(m * n * sizeof(float));
        memcpy(A_old, A, m * n * sizeof(float));
        for (int row = lambda; row < m; row++) {
            for (int col = tau; col < n; col++) {
                float inner_product = 0;
                for (int inner_dim = 0; inner_dim < (m - lambda); inner_dim++) {
                    inner_product += panel_Q[(inner_dim) * (m - lambda) + (row - lambda)] * A_old[(inner_dim + lambda) * n + col];
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
                    inner_product += Q_old[row * m + inner_dim + lambda] * panel_Q[(inner_dim * (m - lambda)) + (col - lambda)];
                }
                Q[row * m + col] = inner_product;
            }
        }
        free(Q_old);
        free(panel_Q);

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

void test_h_householder_qr(int m, int n, int r, float* A_in) {
    /*
    * Test host version of householder QR
    */

    // TASK14 3 alice: iterate over many matrix sizes, & test matrices from Tong
    printf("\nTesting sequential householder QR...\n");

    printf("Dimensions of A: %dx%d\n", m, n);


    int global_offset = 0;

    float* Q = (float*)malloc(m * m * sizeof(float));
    float* R = (float*)malloc(m * n * sizeof(float));
    float* A_out = (float*)malloc((m + 1) * n * sizeof(float));

    h_matrix_cpy((float*)A_in, A_out, m, n);


    //h_block_qr((float*)A, Q, m, n, r);
    auto start_time = std::chrono::high_resolution_clock::now();
    h_householder_qr(A_out, m, n, 0, n);
    h_q_backward_accumulation(A_out, &Q, m, n);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed_time =
    end_time - start_time;
    float time_ms = elapsed_time.count();

    float flops = h_qr_flops_per_second(time_ms, m, n);

    h_strip_R_from_A((float*)A_out, R, m, n);

    float backward_error = h_backward_error((float*)A_in, R, Q, m, n);
    float error3 = h_error_3(R, m, n);
    float error2 = h_error_2(Q, m);
    //printf("||A - QR||/||A|| = %e\n", backward_error);
    //printf("||QT @ Q - Im|| = %e\n", h_error_2(Q, m));
    //printf("||L|| = %e\n", error3);
    printf("Averaged %.2f GFLOPs\n", flops / 1E9);
    printf("Sequential householder finished in %.2f ms\n", time_ms);

    h_write_results_to_log(m, n, time_ms, flops / 1E9, backward_error * 1e8, "cpu_householder");


    // write results to log file
    free(Q);
    free(R);
    free(A_out);
}

void test_h_householder_qr() {

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

void test_dev_wy_compute_Im_sub_W_Yt(int W_Yt_dim, int panel_width, int current_column) {
    /*
    * Tests subroutine for WY representation on GPU, which computes matrix product W @ transpose(Y)
    */

    printf("\nTesting GPU SMEM (Im - W @ Yt) kernel %dx%d...\n", W_Yt_dim, panel_width);

    size_t W_size = W_Yt_dim * panel_width * sizeof(float);
    size_t Y_size = W_Yt_dim * panel_width * sizeof(float);
    size_t W_Yt_size = W_Yt_dim * W_Yt_dim * sizeof(float);

    float* W = (float*)malloc(W_size);
    float* Y = (float*)malloc(Y_size);
    float* dev_result_W_Yt = (float*)malloc(W_Yt_size);
    float* h_result_W_Yt = (float*)malloc(W_Yt_size);

    // initialize matrix A
    for (int i = 0; i < W_Yt_dim; i++) {
        for (int j = 0; j < panel_width; j++) {
            W[i * panel_width + j] = (float)i+1;
            Y[i * panel_width + j] = (float)i+1;
        }
    }

    memset(dev_result_W_Yt, 0.0, W_Yt_size);
    memset(h_result_W_Yt, 0.0, W_Yt_size);

    float* dev_W;
    float* dev_Y;
    float* dev_WYt;

    cudaMalloc(&dev_W, W_size);
    cudaMalloc(&dev_Y, Y_size);
    cudaMalloc(&dev_WYt, W_Yt_size);

    cudaMemcpy(dev_W, W, W_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Y, Y, Y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_WYt, dev_result_W_Yt, W_Yt_size, cudaMemcpyHostToDevice);

    // Configure grid
    dim3 gridDim((int)ceil((float)W_Yt_dim / TILE_WIDTH), (int)ceil((float)W_Yt_dim / TILE_WIDTH), 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Launch kernel
    dev_wy_compute_Im_sub_W_Yt << <gridDim, blockDim >> > (dev_WYt, dev_W, dev_Y, panel_width, current_column, W_Yt_dim);

    cudaDeviceSynchronize();

    cudaMemcpy(W, dev_W, W_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, dev_Y, Y_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dev_result_W_Yt, dev_WYt, W_Yt_size, cudaMemcpyDeviceToHost);

    // Run CPU version for comparison
    for (int row = 0; row < W_Yt_dim; row++) { // rows of W_Yt
        int row_offset = row * panel_width;
        for (int col = 0; col < W_Yt_dim; col++) { // cols of W_Yt
            int col_offset = col * panel_width;
            // compute each inner product
            float inner_product = 0;
            for (int idx = 0; idx < current_column; idx++) { // idx of columns of W
                inner_product += W[row_offset + idx] * Y[col_offset + idx];
            }
            if (row == col) { // Im is 1
                h_result_W_Yt[row * W_Yt_dim + col] = 1 - inner_product; // Im - WY^T
            }
            else { // Im is zero
                h_result_W_Yt[row * W_Yt_dim + col] = -inner_product;
            }
        }
    }

    // Compare GPU and CPU result
    bool pass = true;
    for (int row = 0; row < W_Yt_dim; row++) {
        for (int col = 0; col < W_Yt_dim; col++) {
            if (h_result_W_Yt[row * W_Yt_dim + col] != dev_result_W_Yt[row * W_Yt_dim + col]) {
                pass = false;
            }
        }
    }

    if (pass) {
        printf("Test passed.\n");
    }
    else {
        printf("Test failed.\n");
    }

    free(W);
    free(Y);
    free(dev_result_W_Yt);
    free(h_result_W_Yt);

    cudaFree(dev_W);
    cudaFree(dev_Y);
    cudaFree(dev_WYt);

}

void test_dev_wy_compute_z(int m, int n, int global_offset, int column_offset) {
    /*
    * Tests GPU kernel for WY representation, which computes matrix-vector product (Im - W @ Yt) @ w, where w is the
    * householder vector stored in the lower trapezoidal region of A
    */

    printf("\nTesting WY z computation...\n");
    printf("Dimensions of A (m, n): (%d, %d)\n", m, n);

    int W_Yt_dim = m - global_offset;

    size_t W_Yt_size = W_Yt_dim * W_Yt_dim * sizeof(float);
    size_t A_size = (m + 1) * n * sizeof(float);
    size_t w_size = W_Yt_dim * sizeof(float);

    float* dev_result_z = (float*)malloc(w_size);
    float* h_result_z = (float*)malloc(w_size);
    float* h_A_in = (float*)malloc(A_size);
    float* h_W_Yt = (float*)malloc(W_Yt_size);

    float* dev_A_in;
    float* dev_W_Yt;
    float* dev_z;

    cudaMalloc(&dev_A_in, A_size);
    cudaMalloc(&dev_W_Yt, W_Yt_size);
    cudaMalloc(&dev_z, w_size);

    // initialize matrix A
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n; j++) {
            h_A_in[i * n + j] = (float)i + 1;
        }
    }

    // initialize W_Yt matrix
    for (int row = 0; row < W_Yt_dim; row++) {
        for (int col = 0; col < W_Yt_dim; col++) {
            h_W_Yt[row * W_Yt_dim + col] = (float)col+1;
        }
    }

    cudaMemcpy(dev_A_in, h_A_in, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W_Yt, h_W_Yt, W_Yt_size, cudaMemcpyHostToDevice);

    dim3 gridDim(1, (int)ceil((float)W_Yt_dim / VECTOR_OP_1D_BLOCK_WIDTH), 1);
    dim3 blockDim(1, VECTOR_OP_1D_BLOCK_WIDTH, 1);
    dev_wy_compute_z<<<gridDim, blockDim>>>(dev_z, dev_W_Yt, dev_A_in, m, n, global_offset, W_Yt_dim, column_offset);

    cudaDeviceSynchronize();

    cudaMemcpy(dev_result_z, dev_z, w_size, cudaMemcpyDeviceToHost);

    // Compute with CPU for comparison
    for (int row = 0; row < W_Yt_dim; row++) {
        float inner_product = 0;
        for (int col = column_offset; col < W_Yt_dim; col++) {
            inner_product += h_W_Yt[row * W_Yt_dim + col] * h_A_in[(global_offset + col + 1) * n + global_offset + column_offset];
        }
        h_result_z[row] = 2 * inner_product;
    }

    // Compare CPU and GPU result
    bool pass = true;
    for (int row = 0; row < W_Yt_dim; row++) {
        if (h_result_z[row] != dev_result_z[row]) {
            pass = false;
        }
    }

    if (pass) {
        printf("Test passed.\n");
    }
    else {
        printf("Test failed.\n");
    }

    free(dev_result_z);
    free(h_result_z);
    free(h_A_in);
    free(h_W_Yt);

    cudaFree(dev_A_in);
    cudaFree(dev_W_Yt);
    cudaFree(dev_z);
}

void test_dev_wy_transform(int m, int n, int panel_width, int global_offset) {
    /*
    * Test GPU wy transform against CPU version
    */
    printf("\nTesting GPU WY transform...\n");
    printf("Dimensions (m, n, panel_width, global_offset): (%d,%d,%d,%d)\n", m, n, panel_width, global_offset);

    size_t A_size = (m+1) * n * sizeof(float);
    size_t panel_Q_size = (m-global_offset) * (m-global_offset) * sizeof(float);

    float* h_A = (float*)malloc(A_size);
    float* h_result_panel_Q = (float*)malloc(panel_Q_size);
    float* h_dev_result_panel_Q = (float*)malloc(panel_Q_size);

    unsigned seed = time(0);
    srand(seed);
    // initialize matrix A
    for (int row = 0; row < m + 1; row++) {
        for (int col = 0; col < n; col++) {
            h_A[row * n + col] = ((float)rand() / (RAND_MAX));
        }
    }

    float* dev_A;
    float* dev_panel_Q;

    cudaMalloc(&dev_A, A_size);
    cudaMemcpy(dev_A, h_A, A_size, cudaMemcpyHostToDevice);

    dev_wy_transform(dev_A, &dev_panel_Q, m, n, global_offset, panel_width);

    cudaMemcpy(h_dev_result_panel_Q, dev_panel_Q, panel_Q_size, cudaMemcpyDeviceToHost);

    h_wy_transform(h_A, &h_result_panel_Q, m, n, global_offset, panel_width);


    bool pass = true;
    for (int row = 0; row < m-global_offset; row++) {
        for (int col = 0; col < m-global_offset; col++) {
            if (h_dev_result_panel_Q[row * (m-global_offset) + col] !=
                                    h_result_panel_Q[row * (m-global_offset) + col]) {
                pass = false;
            }
        }
    }

    if (pass) {
        printf("Test passed.\n");
    }
    else {
        printf("Test failed.\n");
    }

    free(h_A);
    free(h_result_panel_Q);
    free(h_dev_result_panel_Q);

    cudaFree(dev_A);
    cudaFree(dev_panel_Q);
}


void test_h_block_qr(int m, int n, int r) {
    /*
    * Test host version of block QR
    */

    printf("\nTesting sequential block QR...\n");
    printf("Dimensions of A (m, n, r): (%d,%d,%d)\n", m, n, r);

    float* A_in = h_generate_random_matrix<float>(m, n);

    float* Q = (float*)malloc(m * m * sizeof(float));
    float* R = (float*)malloc(m * n * sizeof(float));
    float* A_out = (float*)malloc((m + 1) * n * sizeof(float));

    h_identity_mtx(Q, m, m);

    h_matrix_cpy((float*)A_in, A_out, m, n);

    clock_t cycles = clock(); // Time how long the QR function takes to execute
    h_block_qr((float*)A_out, Q, m, n, r);
    cycles = clock() - cycles;
    float time_ms = cycles * 1000 / CLOCKS_PER_SEC;

    float flops_per_second = h_qr_flops_per_second(time_ms, m, n);

    h_strip_R_from_A((float*)A_out, R, m, n);

    float backward_error = h_backward_error((float*)A_in, R, Q, m, n);
    float error2 = h_error_2(Q, m);
    float error3 = h_error_3(R, m, n);

    // write results to log file
    h_write_results_to_log(m, n, time_ms, flops_per_second, backward_error, "cpu_block");

    printf("Sequential block QR finished in %.2f ms...\n", time_ms);
    //printf("||A - QR||/||A|| = %e\n", backward_error);
    free(Q);
    free(R);
    free(A_out);
}

struct MatrixInfo {
    std::string filePath;
    int m;
    int n;
};
bool compareByRow(const MatrixInfo& item1, const MatrixInfo& item2) {
    return item1.m < item2.m;
}
std::vector<MatrixInfo> get_jacobians_test_matrixs() {
    std::string folderPath = "./jacobians";
    std::vector<MatrixInfo> list;
    for (int i = 100; i <= 22500; i += 100) {
        std::ostringstream oss;
        oss << "A_" << std::setfill('0') << std::setw(9) << i << ".txt";
        std::string filename = oss.str();
        std::string filePath = folderPath + "/" + filename;

        std::ifstream file(filePath);

        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            std::istringstream iss(line);
            int m, n;
            iss >> m >> n;
            MatrixInfo info;
            info.filePath = filePath;
            info.m = m;
            info.n = n;
            list.push_back(info);
            //read file conten if needed
            // while (std::getline(file, line)) {
            // }
            file.close();
        } else {
            std::cout << "can not open file:" << filePath << std::endl;
        }
    }
    std::sort(list.begin(), list.end(), compareByRow);
    std::vector<MatrixInfo> result;
    int matrixCount = 30;
    for (int i =0;i < list.size() && result.size() < matrixCount;i+= 2) {
        result.push_back(list[i]);
    }
    return result;
}

struct QRProblemSize {
    // A = QR problem set dimensions
    int m; // height of matrix A
    int n; // width of matrix A
    int r; // block QR panel width
};

# define NUM_STATIC_QR_TESTS 24
# define NUM_STATIC_MMULT_TESTS 15
struct MMULTProblemSize {
    // C = A @ B problem set dimensions
    // Dimensions of A: m x k
    // Dimensions of B: k x n
    // Dimensions of C: m x n
    int m;
    int n;
    int k;
};

void test_qr_by_random_matrix(QR_FUNC f) {
   QRProblemSize testDim[NUM_STATIC_QR_TESTS] = {
       {6, 4, 2},
       {6, 4, 1},
       {6, 4, 3},
       {12, 8, 4},
       {12, 8, 5},
       {12, 8, 6},
       {12, 8, 2},
       {12, 8, 8},
       {12, 8, 3},
       {24, 16, 8},
       {24, 16, 12},
       {60, 40, 8},
       {60, 40, 16},
       {80, 80, 16},
       {97, 90, 16},
       {100, 80, 16},
       {128, 80, 16},
       {129, 80, 16},
       {240, 160, 16},
       {600, 400, 16},
        {900, 900, 16},
        {1200, 1200, 16},
        {1500, 1500, 16},
        {1800, 1800, 32},
   };
    for (int i = 0; i < NUM_STATIC_QR_TESTS; i++) {
        float* A_in = h_generate_random_matrix(testDim[i].m, testDim[i].n);
        f(testDim[i].m, testDim[i].n, testDim[i].r, A_in);
    }
}

void test_qr(QR_FUNC f) {
    std::vector<MatrixInfo> list = get_jacobians_test_matrixs();
    for (const auto& item : list) {
        int m;
        int n;
        float* A_in;
        read_euroc_jacobian(item.filePath, &m, &n, &A_in);
        f(m, n, 16, A_in);
    }
}


void test_mmult(MMULT_FUNC f) {
    QRProblemSize testDim[NUM_STATIC_MMULT_TESTS] = {
        {6, 4, 2},
        {6, 4, 1},
        {6, 4, 3},
        {12, 8, 4},
        {12, 8, 5},
        {12, 8, 6},
        {12, 8, 2},
        {12, 8, 8},
        {12, 8, 3},
        {24, 16, 8},
        {24, 16, 12},
        {60, 40, 8},
        {240, 160, 16},
        {600, 400, 16},
        {600, 400, 600}
    };

    for (int i = 0; i < NUM_STATIC_MMULT_TESTS; i++) {
        f(testDim[i].m, testDim[i].n, testDim[i].r);
    }
}

void test_mmult_in_place() {
    MMULTProblemSize testDim[7] = {
        {6, 4, 6},
        {12, 8, 12},
        {24, 16, 24},
        {60, 40, 60},
        {240, 160, 240},
        {400, 300, 400},
        {300, 300, 300}
    };

    for (int i = 0; i < 7; i++) {
        test_dev_smem_mmult_in_place(testDim[i].m, testDim[i].n, testDim[i].k, 400, 400);
    }
}

void test_mmult_in_place_transpose_a() {
    MMULTProblemSize testDim[7] = {
    {6, 4, 6},
    {12, 8, 12},
    {24, 16, 24},
    {60, 40, 60},
    {240, 160, 240},
    {400, 300, 400},
    {300, 300, 300}
    };

    for (int i = 0; i < 7; i++) {
        test_dev_smem_mmult_in_place_transpose_a(testDim[i].m, testDim[i].n, testDim[i].k, 400, 400);
    }
}

void test_dev_block_qr(int m, int n, int r, float * A_in) {
    /*
    * Test GPU version of block QR
    */

    printf("\nTesting GPU block QR...\n");
    printf("Dimensions of A (m, n, r): (%d, %d, %d)\n", m, n, r);

    float* Q = (float*)malloc(m * m * sizeof(float));
    float* R = (float*)malloc(m * n * sizeof(float));
    float* A_out = (float*)malloc((m + 1) * n * sizeof(float));

    h_identity_mtx(Q, m, m);

    h_matrix_cpy((float*)A_in, A_out, m, n);

    auto start_time = std::chrono::high_resolution_clock::now(); // Time how long the QR function takes to execute
    //dev_block_qr((float*)A_out, Q1, m, n, r);
    dev_block_qr_wy(A_out, Q, m, n, r);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed_time =
        end_time - start_time;
    float time_ms = elapsed_time.count();

    float flops = h_qr_flops_per_second(time_ms, m, n);

    h_strip_R_from_A((float*)A_out, R, m, n);

    float backward_error = h_backward_error((float*)A_in, R, Q, m, n);
    float error2 = h_error_2(Q, m);
    float error3 = h_error_3(R, m, n);

    // write results to log file
    h_write_results_to_log(m, n, time_ms, flops / 1E9, backward_error * 1E8, "gpu_block");

    printf("GPU block QR finished...\n");
    printf("Averaged %.2f GFLOPs\n", flops / 1E9);
    printf("GPU Block QR finished in %.2f ms...\n", time_ms);

    free(Q);
    free(R);
    free(A_out);
    free(A_in);
}



int main() {
//     test_h_mmult();
//     test_h_mmult_transpose_A();
//
//     for (int rows = 10; rows < 1000; rows *= 2) {
//         for (int panel_width = 2; panel_width < 16; panel_width *= 2) {
//             for (int current_column = 1; current_column < panel_width; current_column *= 2) {
//                 test_dev_wy_compute_Im_sub_W_Yt(rows, panel_width, current_column);
//             }
//         }
//     }
//
//     for (int m = 10; m < 1000; m *= 2) {
//         for (int n = 2; n < 16; n *= 2) {
//             test_dev_wy_compute_z(m, n, n / 2, 0);
//         }
//     }
//
//     for (int m = 40; m < 2000; m *= 2) {
//         for (int n = m / 4; n < m; n *= 2) {
//             for (int global_offset = n / 4; global_offset < n; global_offset *= 2) {
//                 int panel_width;
//                 if (n-global_offset < 16) {
//                     panel_width = n-global_offset;
//                 }
//                 else if (m < 500) {
//                     panel_width = 16;
//                 }
//                 else {
//                     panel_width = 8;
//                 }
//                 test_dev_wy_transform(m, n, panel_width, global_offset);
//             }
//         }
//     }



    // test_qr_by_random_matrix(test_h_householder_qr);
    // test_qr_by_random_matrix(test_dev_block_qr);

    test_qr(test_h_householder_qr);
    test_qr(test_dev_block_qr);
    //test_qr(test_dev_householder_qr);
    //test_qr(test_h_block_qr);
    //test_mmult(test_dev_smem_mmult);
    //test_mmult_in_place();
    //test_mmult_in_place_transpose_a();
    //test_mmult(test_dev_smem_mmult_in_place);

    //test_dev_smem_mmult(6000, 4000, 6000);
    //test_tensorcore_mmult_gmem();
    //test_tensorcore_mmult_tiled();

    //for (int m = 20; m < 2000; m *= 2) {
    //    for (int n = m / 4; n < m; n *= 2) {
    //        for (int k = n / 4; k < n; k *= 2) {
    //            test_template_tensorcore_mmult_tiled(m, n, k);
    //        }
    //    }
    //}

    //test_template_tensorcore_mmult_tiled(65, 32, 16);
    //test_dev_block_qr_tensorcore_gmem();
}
