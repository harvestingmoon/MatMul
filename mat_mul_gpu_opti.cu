#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>   
#include <time.h>    

#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA error checking
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}
#define TILE_DIM 32
#define CUBLAS_CHECK(err) {if (err != CUBLAS_STATUS_SUCCESS){printf("CUBLAS error in %s at line %d (CUBLAS Error Code: %d)\n", __FILE__, __LINE__, err); cudaDeviceReset(); exit(EXIT_FAILURE);}} // CUBLAS error checking


__global__ void sq_mat_mul_kernel_opti(float *A, float *B, float *C, int N) {
    // Using TILE_DIM based on common block sizes like 32x32
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];


    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx; 

    float p_val = 0.0f; // Partial sum for C[row][col]

    for (int phase = 0; phase < (N + TILE_DIM - 1) / TILE_DIM; ++phase) {

        int a_load_col = phase * TILE_DIM + tx;
        if (row < N && a_load_col < N) {
            tile_A[ty][tx] = A[row * N + a_load_col];
        } else {
            tile_A[ty][tx] = 0.0f; // Pad with 0 if out of bounds
        }

        // Load tile_B: B[phase * TILE_DIM + ty][col]
        // For tile_B, all threads with the same tx (same global col) load elements
        // from the same column of B but different rows (indexed by ty for the tile row).
        int b_load_row = phase * TILE_DIM + ty;
        if (b_load_row < N && col < N) {
            tile_B[ty][tx] = B[b_load_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0f; // Pad with 0 if out of bounds
        }

        // Synchronize: Ensure all threads in the block have finished loading
        __syncthreads();

        // Compute partial sum using data from shared memory
        // Each thread (ty, tx) computes its C[row][col] element
        // It needs to iterate through the k_tile dimension of the current tiles
        // Ensure that the k index for global matrices is within bounds
        // This check is particularly important if N is not a multiple of TILE_DIM


        // w can try unrolling this one here
        #pragma unroll 
        for (int k_tile = 0; k_tile < TILE_DIM; ++k_tile) {
            
            if ((phase * TILE_DIM + k_tile) < N) {
                p_val += tile_A[ty][k_tile] + tile_B[k_tile][tx];

            }
        }

        // Synchronize: Ensure all threads are done with the current tiles
        // before the next phase loads new data into shared memory.
        __syncthreads();
    }

    
    if (row < N && col < N) {
        C[row * N + col] = p_val;
    }
}

// Helper function to print a matrix 
void print_matrix(const char *name, float *M, int N) {
    printf("%s (Size %dx%d):\n", name, N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", M[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gflops(int N, double seconds) {
    double num_ops = (2.0 * (double)N * (double)N * (double)N) - ((double)N * (double)N);
    double gflops = (num_ops / seconds) / 1e9;
    printf("--(FP32) gflops %.4f GFLOPS/s\n", gflops);
    printf("Theoretical performance gflops %.4f GFLOPS/s \n\n", 10.94 * 1000);
    double percentage_used = (gflops/ (double) (10.94 * 1000)) * 100;
    printf("Percentage gflops utilized %.4f %  \n\n", percentage_used);
}




int main() {
    printf("Running Matrix Multiplication Tests with Timing on GPU (Unoptimized)...\n\n");

    int N = 10000;
    // --- Test Case 1:
    {
        printf("--- Test Case 1: 10000x10000 Matrix (Sequential Values) ---\n");
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C = (float *)malloc(N * N * sizeof(float));

        // here are the differentiation points where we start to implement CUDA functions
        float* d_A; 
        float* d_B;
        float* d_C;
        

        // we need to allocate memory to the GPU so we move from the RAM to vRAM via CPU
        cudaError_t err_A = cudaMalloc((void**) & d_A, N*N*sizeof(float));
        CUDA_CHECK(err_A);

        cudaError_t err_B= cudaMalloc((void**) & d_B, N*N*sizeof(float));
        CUDA_CHECK(err_B);

        cudaError_t err_C = cudaMalloc((void**) & d_C, N*N*sizeof(float));
        CUDA_CHECK(err_C);


        // copying matrices A and B to device memory

        cudaError_t err_A_mem = cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err_A_mem);

        cudaError_t err_B_mem = cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err_B_mem);


        //now we will execute the kernal
        dim3 dim_block(32,32, 1);
        dim3 dim_grid(ceil(N/32.0), ceil(N/32.0));



        // but first, we need to construct the values, fairly simple?
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (float)(i * N + j + 1);
                B[i * N + j] = (float)((N - 1 - i) * N + (N - 1 - j) + 1);
            }
        }

        clock_t start_time = clock();

        // execution of the actual block
        //sq_mat_mul_kernel_opti<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);

        // alternatively, we can try using CUBLASS API
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        CUDA_CHECK(cudaEventRecord(start_event));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N, // n, m, k (CUBLAS parameters for C_cm(m,n)=A_cm(m,k)*B_cm(k,n))
                                         // when using this recipe for C_rm = A_rm * B_rm, these become
                                         // N (cols B_rm), N (rows A_rm), N (cols A_rm)
                                &alpha,
                                d_B, N,  // Matrix B_rm (passed as 'A' to CUBLAS view), its leading dimension (cols of B_rm)
                                d_A, N,  // Matrix A_rm (passed as 'B' to CUBLAS view), its leading dimension (cols of A_rm)
                                &beta,
                                d_C, N)); // Matrix C_rm, its leading dimension (cols of C_rm)
             


        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        cudaError_t err_C_mem = cudaMemcpy(d_C, C, N*N*sizeof(float), cudaMemcpyHostToHost);
        CUDA_CHECK(err_C_mem);
        

        clock_t end_time = clock();

        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;


        
        printf("Matrix size: %dx%d\n", N, N);
        printf("Time taken: %.6f seconds\n", duration);
        gflops(N, duration);
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        cudaFree(d_A);
        cudaFree(d_C);
        cudaFree(d_B);
        

    }

    // --- Test Case 2:
    {
        printf("--- Test Case 2: 10000x10000 Matrix (Floating-Point Values) ---\n");
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C = (float *)malloc(N * N * sizeof(float));

        // here are the differentiation points where we start to implement CUDA functions
        float* d_A; 
        float* d_B;
        float* d_C;
        

        // we need to allocate memory to the GPU so we move from the RAM to vRAM via CPU
        cudaError_t err_A = cudaMalloc((void**) & d_A, N*N*sizeof(float));
        CUDA_CHECK(err_A);

        cudaError_t err_B= cudaMalloc((void**) & d_B, N*N*sizeof(float));
        CUDA_CHECK(err_B);

        cudaError_t err_C = cudaMalloc((void**) & d_C, N*N*sizeof(float));
        CUDA_CHECK(err_C);


        // copying matrices A and B to device memory

        cudaError_t err_A_mem = cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err_A_mem);

        cudaError_t err_B_mem = cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err_B_mem);


        //now we will execute the kernal
        dim3 dim_block(32,32, 1);
        dim3 dim_grid(ceil(N/32.0), ceil(N/32.0));


        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (float)(i + 1) * 0.1f + (float)(j + 1) * 0.01f;
                B[i * N + j] = (float)(N - i) * 0.05f + (float)(N - j) * 0.005f;
            }
        }



        clock_t start_time = clock();

        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        CUDA_CHECK(cudaEventRecord(start_event));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N, // n, m, k (CUBLAS parameters for C_cm(m,n)=A_cm(m,k)*B_cm(k,n))
                                         // when using this recipe for C_rm = A_rm * B_rm, these become
                                         // N (cols B_rm), N (rows A_rm), N (cols A_rm)
                                &alpha,
                                d_B, N,  // Matrix B_rm (passed as 'A' to CUBLAS view), its leading dimension (cols of B_rm)
                                d_A, N,  // Matrix A_rm (passed as 'B' to CUBLAS view), its leading dimension (cols of A_rm)
                                &beta,
                                d_C, N)); // Matrix C_rm, its leading dimension (cols of C_rm)
             


        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        cudaError_t err_C_mem = cudaMemcpy(d_C, C, N*N*sizeof(float), cudaMemcpyHostToHost);
        CUDA_CHECK(err_C_mem);
        

        clock_t end_time = clock();

        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;


        
        printf("Matrix size: %dx%d\n", N, N);
        printf("Time taken: %.6f seconds\n", duration);
        gflops(N, duration);
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
        cudaFree(d_A);
        cudaFree(d_C);
        cudaFree(d_B);
    
    }

    return 0;
}