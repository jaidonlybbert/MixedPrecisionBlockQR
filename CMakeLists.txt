cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MixedPrecisionBlockQR LANGUAGES CXX CUDA)
add_subdirectory(${CMAKE_HOME_DIRECTORY}/tests)

# Extract input data archive (jacobian test data)
set(QR_DATA_PATH "${CMAKE_HOME_DIRECTORY}/data")
set(QR_JACOBIAN_PATH "${QR_DATA_PATH}/jacobians")
file(ARCHIVE_EXTRACT 
	INPUT "${QR_DATA_PATH}/okvis_euroc_MH_05_jacobians.tar.gz"
	DESTINATION "${QR_JACOBIAN_PATH}"
	)
configure_file(${CMAKE_HOME_DIRECTORY}/Cuda/qr_config.h.in ${CMAKE_HOME_DIRECTORY}/Cuda/qr_config.h)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Add the executable
add_executable(MixedPrecisionBlockQR Cuda/qr.cu Cuda/mmult.cu Cuda/main.cu)

target_include_directories(MixedPrecisionBlockQR PUBLIC 
							"${CMAKE_HOME_DIRECTORY}/Cuda")
# Request that MixedPrecisionBlockQR is built with -std=c++20
target_compile_features(MixedPrecisionBlockQR PUBLIC cxx_std_20)

# We need to explicitly state that we need all CUDA files in the 
# project folder to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(MixedPrecisionBlockQR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
