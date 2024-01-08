cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MixedPrecisionBlockQR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 52)
endif()

# Add the executable
add_executable(MixedPrecisionBlockQR Cuda/qr.cu)

# Request that MixedPrecisionBlockQR is built with -std=c++20
target_compile_features(MixedPrecisionBlockQR PUBLIC cxx_std_20)

# We need to explicitly state that we need all CUDA files in the 
# project folder to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(MixedPrecisionBlockQR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
