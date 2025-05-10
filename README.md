# Matrix Multiplication Implementations

This folder contains various C++ and CUDA C++ implementations for performing square matrix multiplication, showcasing different approaches from basic CPU execution to optimized GPU computations using custom kernels and NVIDIA's CUBLAS library

The primary goal is to explore and compare the performance characteristics of these different methods, particularly for large matrices

## Files and Implementations

### 1. `mat_mul_cpu.cpp` - CPU Implementation

* **Description:** This file provides a standard, generic matrix multiplication algorithm executed entirely on the CPU


* **Testing:** It includes 2 test cases, demonstrating multiplication for a 10000x10000 matrix. One is in sequential numbering while the other is in floating point operations, of course it took around ~ 120 minutes to calculate which is extremely slow.


### 2. `mat_mul_gpu.cu` - Naive GPU Implementation

* **Description:** This file contains a basic implementation of matrix multiplication accelerated on an NVIDIA GPU using CUDA


* **Approach (Naive):**
    1.  **Memory Transfer (Host to Device):** Input matrices are copied from CPU main memory (RAM) to GPU global memory (VRAM)
    2.  **GPU Computation:** A simple CUDA kernel performs the matrix multiplication on the GPU
    3.  **Memory Transfer (Device to Host):** The resulting matrix is copied back from GPU VRAM to CPU RAM

* **Considerations:** This version implements the fundamental steps of GPU computing but does not incorporate advanced optimization techniques. It primarily demonstrates the data flow and basic kernel execution, without deeply considering overheads like memory latency or thread divergence


* **Testing:** ~Approximately 12 seconds, or about 3% of the entire GPU theoretical GFLOPS

### 3. `mat_mul_gpu_opti.cu` - Optimized GPU Implementation & CUBLAS

This file presents two advanced approaches for GPU matrix multiplication: a custom-optimized CUDA kernel and the use of the NVIDIA CUBLAS library.

#### Part A: Optimized Custom CUDA Kernel

* **Description:** This section of the file focuses on a CUDA kernel specifically optimized for better performance than the naive GPU version


* **Key Optimizations at Kernel Level:**

    * **Memory Coalescing:** Data is read from global memory in a way that allows threads within a warp to access contiguous memory locations, maximizing memory transaction efficiency

    * **Sequential Addressing:** Threads are organized to access memory sequentially where possible, further improving global memory access patterns

    * **Tiling (Shared Memory Usage):** The matrices are divided into smaller blocks (tiles). These tiles are loaded into the GPU's fast on-chip shared memory. Computations are then performed on these tiles, significantly reducing reliance on slower global memory and increasing effective memory bandwidth by reusing data within shared memory ( The tiles are done by 32 x 32 )

* **Performance Note:** Despite these optimizations, it was observed that this custom kernel might still only utilize a fraction (6%) of the GPU's theoretical maximum compute capability 



#### Part B: CUBLAS Implementation

* **CUBLAS:** CUBLAS is a highly optimized Basic Linear Algebra Subprograms (BLAS) library provided by NVIDIA. It contains routines for matrix and vector operations that are fine-tuned for NVIDIA GPU architectures (# Vendor Locked In Syndrome)



* Using the CUBLAS ```cublassSgemm``` function to calculate the matirx multiplication, it was discovered that we leverage approximately 48~ 50% of compute capability. Of course, this is only for production grade but regardless it is fairly impressive 

## Performance Metrics: GFLOPS

A common metric to evaluate the performance of high-performance computing tasks like matrix multiplication is GFLOPS (Giga Floating-point Operations Per Second).

For a standard multiplication of two $N \times N$ matrices, the number of floating-point operations is approximately $2N^3 - N^2$ (or often simplified to $2N^3$ for large $N$).

* Each element of the resulting $N \times N$ matrix requires $N$ multiplications and $N-1$ additions.
* Total multiplications = $N^2 \times N = N^3$.
* Total additions = $N^2 \times (N-1) = N^3 - N^2$.
* Total operations = $N^3 + (N^3 - N^2) = 2N^3 - N^2$.

**Formula to Calculate GFLOPS:**

$$ \text{GFLOPS} = \frac{\text{Total Floating Point Operations}}{\text{Time Taken (seconds)} \times 10^9} $$

So, for matrix multiplication:

$$ \text{GFLOPS} = \frac{2N^3 - N^2}{\text{Time Taken (seconds)} \times 10^9} $$


## How to Use

1.  **Compile:**
    * CPU code (`mat_mul_cpu.cpp`): Use a C++ compiler (e.g., g++).
        ```bash
        g++ mat_mul_cpu.cpp -o mat_mul_cpu -O3
        ```
    * GPU code (`.cu` files): Use the NVIDIA CUDA Compiler (`nvcc`).
        ```bash
        # For naive GPU version
        nvcc mat_mul_gpu.cu -o mat_mul_gpu

        # For optimized GPU version (including CUBLAS)
        nvcc mat_mul_gpu_opti.cu -o mat_mul_gpu_opti -lcublas
        ```
        (Ensure the CUDA toolkit path is correctly set up in your environment variables).
2.  **Run:**
    ```bash
    ./mat_mul_cpu
    ./mat_mul_gpu
    ./mat_mul_gpu_opti
    ```

## Reference Links:
- https://medium.com/data-science/matrix-multiplication-on-the-gpu-e920e50207a8
- https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8 
- https://0mean1sigma.com/2678x-faster-how-gpus-supercharge-matrix-multiplication/
