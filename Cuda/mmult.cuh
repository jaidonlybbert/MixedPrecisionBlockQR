#pragma once

#define TC_TILE_M 8
#define TC_TILE_N 32
#define TC_TILE_K 16
#define WARP_SIZE 32

// Shared memory tile sizes for tiled global memory mmult
#define GMEM_MMULT_TILE_WIDTH 16
#define GMEM_MMULT_TILE_HEIGHT 16

// Thread block sizes for TC MMULT
// Shrink to get more shared memory per warp, expand to increase memory re-use
#define TC_MMULT_THREAD_BLOCK_WIDTH 128
#define TC_MMULT_THREAD_BLOCK_HEIGHT 4


#define TILE_WIDTH 32

#define COLUMN_MAJOR 0
#define ROW_MAJOR 1

// Thread block sizes for array cpy
#define CPY_ARRAY_BLOCK_WIDTH 32
#define CPY_ARRAY_BLOCK_HEIGHT 32

#define TOP_LEFT 1
#define BOTTOM_RIGHT 0

template <typename T>
T* h_generate_random_matrix(int height, int width) {
    /*
    * Returns pointer to random float matrix of dimensions HeightxWidth
    */
    unsigned seed = time(0);
    srand(seed);
    T* matrix = (T*)malloc(height * width * sizeof(T));
    T random_num = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float rand_fun = (float)((float)rand() / RAND_MAX);

            if constexpr (std::is_same_v<T, unsigned char>) {
                rand_fun = abs(rand_fun * 12); // low value to prevent overflow
            }
            else if constexpr (std::is_same_v<T, signed char>) {
                rand_fun = (rand_fun * 12); // low value to prevent overflow
            }

            random_num = rand_fun;
            matrix[row * width + col] = random_num; // randomize this number
        }
    }

    return matrix;
}
template float* h_generate_random_matrix<float>(int, int);
template __half* h_generate_random_matrix<__half>(int, int);
template unsigned char* h_generate_random_matrix<unsigned char>(int, int);
template signed char* h_generate_random_matrix<signed char>(int, int);

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
            float inner_product = 0;
            for (int inner_idx = 0; inner_idx < k; inner_idx++) {
                inner_product += (float)A[row * k + inner_idx] * (float)B[(inner_idx)*n + col];
            }
            C[row * n + col] = (T_C)inner_product;
        }
    }
}

template void h_mmult<float, float, float>(float*, float*, float*, int, int, int);
template void h_mmult<__half, __half, float>(__half*, __half*, float*, int, int, int);
template void h_mmult<__half, __half, __half>(__half*, __half*, __half*, int, int, int);

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
    int m_padded = (m % TC_TILE_M) ? m + (TC_TILE_M - m % TC_TILE_M) : m; // Padded height of A & C
    int n_padded = (n % TC_TILE_N) ? n + (TC_TILE_N - n % TC_TILE_N) : n; // Padded width of B & C
    int k_padded = (k % TC_TILE_K) ? k + (TC_TILE_K - k % TC_TILE_K) : k; // Padded inner dimension

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


template <typename T_A, typename T_B, typename T_C>
void test_template_tensorcore_mmult_tiled(int m, int n, int k) {
    printf("\nTesting template tensorcore tiled mmult (m, n, k) = (%dx%dx%d)...\n", m, n, k);

    // Initialize random matrices
    T_A* a_mtx = h_generate_random_matrix<T_A>(m, k);
    T_B* b_mtx = h_generate_random_matrix<T_B>(k, n);
    T_C* c_mtx_dev_result = (T_C*)malloc(m * n * sizeof(T_C));
    memset(c_mtx_dev_result, 0.0, m * n * sizeof(T_C));
    T_C* c_mtx_h_result = (T_C*)malloc(m * n * sizeof(T_C));
    memset(c_mtx_h_result, 0.0, m * n * sizeof(T_C));


    h_launch_dev_tensorcore_mmult_tiled<T_A, T_B, T_C>(a_mtx, b_mtx, c_mtx_dev_result, m, n, k);

    h_mmult<T_A, T_B, T_C>(a_mtx, b_mtx, c_mtx_h_result, m, n, k);


    // test result
    bool pass = true;
    float max_error = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T_A, __half>) {
                float error = abs((float)c_mtx_h_result[i * n + j] - (float)c_mtx_dev_result[i * n + j]);
                if (error > 5E-4) {
                    pass = false;
                }
                if (error > max_error) max_error = error;
            }

            if constexpr (std::is_same_v<T_C, int>) {
                int error = abs((int)((int)c_mtx_h_result[i * n + j] - (int)c_mtx_dev_result[i * n + j]));
                if (error > 0) {
                    pass = false;
                }
                if (error > max_error) max_error = error;
            }
        }
    }

    if (pass) {
        printf("Test passed. Max error: %.2E\n", max_error);
    }
    else {
        printf("Test failed. Max error: %.2E exceeded %.2E limit\n", max_error, 5E-4);
    }
}

template void test_template_tensorcore_mmult_tiled<__half, __half, __half>(int, int, int);
template void test_template_tensorcore_mmult_tiled<__half, __half, float>(int, int, int);
template void test_template_tensorcore_mmult_tiled<unsigned char, unsigned char, int>(int, int, int);
template void test_template_tensorcore_mmult_tiled<signed char, signed char, int>(int, int, int);

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
struct MMULTProblemSize {
    // C = A @ B problem set dimensions
    // Dimensions of A: m x k
    // Dimensions of B: k x n
    // Dimensions of C: m x n
    int m;
    int n;
    int k;
};

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
    int k = 3;

    float* C = (float*)malloc(m * n * sizeof(float));

    h_mmult_transpose_A((float*)A, (float*)A, C, m);

    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            assert((C[row * n + col] - ((float*)expected_result)[row * n + col]) < 1E-8);
        }
    }
}