#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>

#include "qr.cuh"
#include "mmult.cuh"

int main() {

	test_qr(test_dev_mixed_precision_block_qr);
	return 0;
}
