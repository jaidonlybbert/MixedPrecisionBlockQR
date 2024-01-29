#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>

#include "qr.cuh"
#include "mmult.cuh"

int main() {

	//test_qr(test_dev_mixed_precision_block_qr);
	//return 0;
	test_iterator_dev_wy_funcs();
	/*
    test_iterator_template_tensorcore_mmult_tiled();

    test_qr_by_random_matrix(test_h_householder_qr);
    test_qr_by_random_matrix(test_dev_block_qr);
    test_qr_by_random_matrix(test_dev_mixed_precision_block_qr);

    test_qr(test_h_householder_qr);
    test_qr(test_dev_block_qr);
    test_qr(test_dev_mixed_precision_block_qr);
*/
    return 0;
}
