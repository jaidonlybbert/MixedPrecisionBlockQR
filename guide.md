 ## Prerequisites
 * python3
 * Cuda 11+

 ## Build
 ```bash
 git clone git@github.com:jaidonlybbert/MixedPrecisionBlockQR.git
 cd MixedPrecisionBlockQR/Cuda/
 # you may need add cuda environment variables in linux platform before compile
nvcc -arch=sm_75 qr.cu
```

## Run test
1. Unzip the jacobians test matrix to `Cuda` folder  
2. Run test
    * linux: `./a.out`
    * windows `./a.exe`

## Draw performance plot
1. Install pyplot
2. run `cuda/performance/runtime.py`