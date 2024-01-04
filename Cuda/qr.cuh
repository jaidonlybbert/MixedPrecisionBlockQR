#pragma once

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

#define NUM_STATIC_QR_TESTS 20
#define VECTOR_OP_1D_BLOCK_WIDTH 64

struct QRProblemSize {
    // A = QR problem set dimensions
    int m; // height of matrix A
    int n; // width of matrix A
    int r; // block QR panel width
};

struct MatrixInfo {
    std::string filePath;
    int m;
    int n;
};

typedef void QR_FUNC(int, int, int, float*);

void h_write_results_to_log(int height, int width, float time_ms, float flops_per_second, float backward_error, std::string file_name);

void h_strip_R_from_A(float* A, float* R, int m, int n);

float h_qr_flops_per_second(float time_ms, int m, int n);

float h_backward_error(float* A, float* R, float* Q, int m, int n, int precision_bits);

float h_q_error(float* Q, int m, int precision_bits);

float h_lower_trapezoid_error(float* R, int m, int n, int precision_bits);

void h_householder_qr(float* A, int m, int n, int global_offset, int panel_width);


void h_q_backward_accumulation(float* h_A, float** h_Q, int m, int n);

void h_wy_transform(float* h_A, float** h_Q, int m, int n, int global_offset, int panel_width);

__global__
void dev_wy_compute_z(float* dev_z, float* dev_W_Yt, float* dev_A, int m, int n, int global_offset, int W_Yt_dim, int column_offset);

__global__
void dev_wy_copy_z_and_w(float* dev_z, float* dev_W, float* dev_Y, float* dev_A,
    int m, int n, int W_Yt_dim, int column_offset, int panel_width, int global_offset);


__global__
void dev_wy_init(float* dev_A, float* dev_Y, float* dev_W, int global_offset, int n, int W_Yt_dim, int panel_width);

__global__
void dev_wy_compute_Im_sub_W_Yt(float* dev_W_Yt, float* dev_W, float* dev_Y,
    int panel_width, int column_idx, int W_Yt_dim);



void dev_wy_transform(float* dev_A, float** dev_panel_Q, int m, int n, int global_offset, int panel_width);

__global__
void dev_householder_qr(float* dev_A, int m, int n, int global_offset);


void read_euroc_jacobian(std::string filename, int* rows, int* cols, float** matrix);




__global__
void dev_apply_qt_to_a(float* dev_A, float* dev_panel_Q, float* res_A, int m, int n, int tau, int lambda);

__global__
void dev_apply_qpanel_to_q(float* dev_Q, float* dev_Q_panel, float* dev_Q_result, int m, int lambda);

__global__
void dev_apply_qpanel_to_q_tensorcore(float* dev_Q, float* dev_Q_panel, float* dev_Q_result, int m, int lambda);

__global__
void dev_cpy_panel_result_a(float* dev_A, float* dev_A_panel_result, int m, int n, int tau, int lambda);

void dev_block_qr(float* A, float* Q, int m, int n, int r);

void dev_block_qr_wy(float* A, float* Q, int m, int n, int r);

void dev_mixed_precision_block_qr(float* A, float* Q, int m, int n, int r);

void test_dev_householder_qr(int m, int n, int r);

void h_block_qr(float* A, float* Q, int m, int n, int r);


void test_h_householder_qr(int m, int n, int r, float* A_in);

void test_h_householder_qr();


void test_h_wy_transform();

void test_dev_wy_compute_Im_sub_W_Yt(int W_Yt_dim, int panel_width, int current_column);

void test_dev_wy_compute_z(int m, int n, int global_offset, int column_offset);

void test_dev_wy_transform(int m, int n, int panel_width, int global_offset);


void test_h_block_qr(int m, int n, int r, float* A_in);

bool compareByRow(const MatrixInfo& item1, const MatrixInfo& item2);

std::vector<MatrixInfo> get_jacobians_test_matrixs();

void test_qr_by_random_matrix(QR_FUNC f);

void test_qr(QR_FUNC f);

void test_dev_block_qr(int m, int n, int r, float* A_in);

void test_dev_mixed_precision_block_qr(int m, int n, int r, float* A_in);

void test_iterator_dev_wy_funcs();

void test_iterator_template_tensorcore_mmult_tiled();