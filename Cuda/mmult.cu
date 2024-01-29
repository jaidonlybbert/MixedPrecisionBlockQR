#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>
#include "mmult.cuh"



void h_mmult_transpose_A(float* A, float* B, float* C, int m) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            float inner_product = 0;
            for (int inner_idx = 0; inner_idx < m; inner_idx++) {
                inner_product += A[(inner_idx)*m + row] * B[(inner_idx)*m + col];
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

void test_tensorcore_mmult_1_warp() {
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
    memset(c_mtx, 0, m * n * sizeof(float));

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
    memset(c_mtx, 0, m * n * sizeof(float));

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

    memset(b_mtx, 0, b_width * b_height * sizeof(float));

    // initialize matrix B
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b_mtx[(i + (b_height - k)) * b_width + (j + (b_width - n))] = (float)(float)j;
        }
    }

    // initialize matrix C
    memset(c_mtx, 0, m * n * sizeof(float));

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

    memset(b_mtx, 0, b_width * b_height * sizeof(float));

    // initialize matrix B
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b_mtx[(i + (b_height - k)) * b_width + (j + (b_width - n))] = (float)(float)j;
        }
    }

    // initialize matrix C
    memset(c_mtx, 0, m * n * sizeof(float));

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

typedef void MMULT_FUNC(int, int, int);
# define NUM_STATIC_MMULT_TESTS 15

void test_mmult(MMULT_FUNC f) {
    MMULTProblemSize testDim[NUM_STATIC_MMULT_TESTS] = {
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
        f(testDim[i].m, testDim[i].n, testDim[i].k);
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

    float* C = (float*)malloc(m * n * sizeof(float));

    h_mmult_transpose_A((float*)A, (float*)A, C, m);

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            assert((C[row * n + col] - ((float*)expected_result)[row * n + col]) < 1E-8);
        }
    }
}


