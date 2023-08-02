# Mixed-Precision Block QR Decomposition
>A rectangular matrix **A** can be factored into a product of an orthogonal matrix **Q**, and an upper triangular matrix **R**; **A** = **QR**.

\- Van Loan Golub. Matrix Computations, Fourth Edition.

# Overview
The QR decomposition can be used as a non-linear least squares matrix solver. It is numerically stable, and well-suited for parallelism. These properties make it useful for processing matrix data from cameras and sensor arrays, such as for SLAM, background subtraction, radio communications, object encoding, point cloud visualization, and many more applications.

This project was started as a University of Washington graduate school project in collaboration with Amazon Lab126. The objective, a fast and correct parallel Block QR decomposition algorithm using half-precision FP16 matrix-matrix multiplies. The Mixed-precision Block QR algorithm is well-suited for large, wide matrices. Other QR algorithms are better suited for small or tall-and-skinny matrices. 

# Test Data
Our test data is derived from the Euroc-MAV dataset to emulate using the QR decomposition to perform a non-linear least squares optimization for robot pose estimation and bundle adjustment for SLAM applications. The implementation works for arbitrarily sized matrices. The dimensions of the largest matrices in our test data were on the order of 2000 x 2000.

# What this project is
* 100% free to use for anything
* A reference for future researchers implementing a parallel QR algorithm (including our references!)
* A way to test FP16 error performance for the Block QR algorithm using TensorCores
* A low-level C++ implementation using functional programming
    * Very few dependencies (just CUDA)
    * Adaptable to alternate parallel processors

# What this project isn't
* In development
    * Our results are published here, with instructions on installing and running the project for yourself. I will respond to issues and PRs about bugs or unclear instructions, but won't be adding features.
* The fastest parallel implementation of QR in CUDA
    * Use CuSolver if you just need a CUDA solver, it uses a more robust implementation of QR by the NVIDIA developers themselves
* Memory optimized
    * We met our performance targets for the research project, but many many things could be done more efficiently, especially with the HtoD memory traffic

# Dependencies (work in progress)
* NVIDIA GPU (compute capability >7.5)
* CUDA 12.
* Python
* CMAKE
* gcc

Windows
* Microsoft Visual Studio 2022

# Installation (work in progress)

All platforms
* Clone repo
* Download test data

Build for Linux

* Run CMAKE
* Execute from shell

Build for Windows

* Run CMAKE
* Open generated .sln
* Build and execute in release mode

All platforms
* Run Python script to generate results
    * Only after execution finishes and the results log is generated for Python to parse

# Our Test Results (work in progress)
## Error

## Execution Time

# Related Projects
* https://github.com/Orgline/LATER
* http://ceres-solver.org
* https://eigen.tuxfamily.org/
* https://docs.nvidia.com/cuda/cusolver/index.html 
* https://numpy.org/ 
* https://netlib.org/lapack/ 
* https://github.com/NVIDIA/cutlass 

# References
* Zhang, S., Baharlouei, E., & Wu, P. (2020). High Accuracy Matrix Computations on Neural Engines: A Study of QR Factorization and its Applications. Proceedings of the 29th International Symposium on High-Performance Parallel and Distributed Computing, 17–28. https://doi.org/10.1145/3369583.3392685 | [PDF](https://www2.cs.uh.edu/~panruowu/pdf/HPDC2020.pdf)
* Bouwmeester, H., Mathias Jacquelin, Langou, J., & Yves, R. (2011). Tiled QR factorization algorithms. arXiv.org. https://arxiv.org/pdf/1104.4475.pdf
* L. Minah Yang, Alyson Fox, and Geoffrey Sanders. Rounding error analysis of mixed precision block householder qr algorithms. SIAM Journal on Scientific Computing, 43(3):A1723–A1753, 2021.
* Van Loan Golub. Matrix Computations, Fourth Edition, pages 235–240, 246–251. The Johns Hopkins University Press, 2013.
* Bhaskar Dasgupta. Applied Mathematical Methods, chapters 10-12. Pearson, 1986.

