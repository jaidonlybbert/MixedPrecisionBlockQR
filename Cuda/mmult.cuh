#pragma once

// CUDA includes
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>

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


#define TILE_WIDTH 32

#define COLUMN_MAJOR 0
#define ROW_MAJOR 1

// Thread block sizes for array cpy
#define CPY_ARRAY_BLOCK_WIDTH 32
#define CPY_ARRAY_BLOCK_HEIGHT 32

#define TOP_LEFT 1
#define BOTTOM_RIGHT 0

template <typename T>
T* h_generate_random_matrix(int height, int width);
template float* h_generate_random_matrix<float>(int, int);
template __half* h_generate_random_matrix<__half>(int, int);
template unsigned char* h_generate_random_matrix<unsigned char>(int, int);
template signed char* h_generate_random_matrix<signed char>(int, int);

template <typename T_A, typename T_B, typename T_C>
void h_mmult(T_A* A, T_B* B, T_C* C, int m, int n, int k);
template void h_mmult<float, float, float>(float*, float*, float*, int, int, int);
template void h_mmult<__half, __half, float>(__half*, __half*, float*, int, int, int);
template void h_mmult<__half, __half, __half>(__half*, __half*, __half*, int, int, int);

void h_mmult_transpose_A(float* A, float* B, float* C, int m);

void h_matrix_subtract(float* A, float* B, float* C, int m, int n);

float h_matrix_norm(float* A, int m, int n);

void h_matrix_cpy(float* A, float* B, int m, int n);

void h_identity_mtx(float* I, int m, int n);

template <typename T>
__global__
void dev_cpy_strided_array(T* dest, T* src, int dest_height, int dest_width, int src_height, int src_width, int mode);

struct CopyMatrixParam {
    int dest_height;
    int dest_width;
    int src_height;
    int src_width;

    int src_row_start;
    int src_row_end;
    int src_col_start;
    int src_col_end;

    int dest_col_start;
    int dest_row_start;
};

template __global__ void dev_cpy_strided_array<float>(float*, float*, int, int, int, int, int);

template <typename T_SRC, typename T_DEST>
__global__
void dev_cpy_and_cast_array(T_DEST* dest, T_SRC* src, CopyMatrixParam param);
template __global__ void dev_cpy_and_cast_array<float, __half>(__half*, float*, CopyMatrixParam);

template <typename T>
void h_launch_cpy_strided_array(T* h_dest, T* h_src, int dest_height, int dest_width, int src_height, int src_width);

__global__ void global_mem_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width);

__global__ void shared_mem_mmult(float* c_mtx, float* a_mtx, float* b_mtx, int a_width, int a_height, int b_width);

__global__ void shared_mem_mmult_transpose_b(float* c_mtx, float* a_mtx, float* b_mtx,
    int a_width, int a_height, int b_width,
    int b_layout);

__global__
void shared_mem_mmult_in_place(float* c_mtx, float* a_mtx, float* b_mtx, int m, int n, int k, int b_height, int b_width);

__global__
void shared_mem_mmult_in_place_transpose_a(float* c_mtx, float* a_mtx, float* b_mtx, int m, int n, int k, int b_height, int b_width);

__global__
void dev_tensorcore_mmult_1_warp(float* c_mtx, half* a_mtx, half* b_mtx);

template <typename T_A, typename T_B, typename T_C>
__global__
void dev_tensorcore_mmult_tiled(T_C* c_mtx, T_A* a_mtx, T_B* b_mtx, int m, int n, int k);

void test_tensorcore_mmult_1_warp();



void test_dev_smem_mmult(int m, int n, int k, int b_layout);

void test_dev_smem_mmult(int m, int n, int k);

void test_dev_smem_mmult_in_place(int m, int n, int k, int b_width, int b_height);

void test_dev_smem_mmult_in_place_transpose_a(int m, int n, int k, int b_width, int b_height);

template <typename T_A, typename T_B, typename T_C>
void h_launch_dev_tensorcore_mmult_tiled(T_A* a_mtx, T_B* b_mtx, T_C* c_mtx, int m, int n, int k);

template void h_launch_dev_tensorcore_mmult_tiled<half, half, float>(half*, half*, float*, int, int, int);


template <typename T_A, typename T_B, typename T_C>
void test_template_tensorcore_mmult_tiled(int m, int n, int k);

template void test_template_tensorcore_mmult_tiled<__half, __half, __half>(int, int, int);
template void test_template_tensorcore_mmult_tiled<__half, __half, float>(int, int, int);
template void test_template_tensorcore_mmult_tiled<unsigned char, unsigned char, int>(int, int, int);
template void test_template_tensorcore_mmult_tiled<signed char, signed char, int>(int, int, int);

void test_tensorcore_mmult_tiled();

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

void test_mmult(MMULT_FUNC f);

void test_mmult_in_place();

void test_mmult_in_place_transpose_a();

void test_h_mmult();

void test_h_mmult_transpose_A();

template <typename T_SRC, typename T_DEST>
void h_launch_dev_cpy_and_cast_array(T_DEST* dev_dest, T_SRC* dev_src, CopyMatrixParam param);

template void h_launch_dev_cpy_and_cast_array<float, float>(float*, float*, CopyMatrixParam);
template void h_launch_dev_cpy_and_cast_array<float, __half>(__half*, float*, CopyMatrixParam);
template void h_launch_dev_cpy_and_cast_array<__half, float>(float*, __half*, CopyMatrixParam);

template <typename T_SRC, typename T_DEST>
void test_dev_cpy_and_cast_array(CopyMatrixParam param);

template void test_dev_cpy_and_cast_array<float, float>(CopyMatrixParam);
template void test_dev_cpy_and_cast_array<float, __half>(CopyMatrixParam);
template void test_dev_cpy_and_cast_array<__half, float>(CopyMatrixParam);