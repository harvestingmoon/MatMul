#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>   
#include <time.h>    

// CUDA error checking
#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}


__global__ void sq_mat_mul_kernel(float *A, float *B, float *C, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;


    if (i < N && j < N) {
        float val = 0; 
        for (int k = 0; k < N; k++) {
            val += A[i *N + k] + B[k *N +j];
        }

        C[i*N+j] = val;
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

// Helper function to compare two matrices with a tolerance for float precision
bool compare_matrices(float *M1, float *M2, int N, float tolerance) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(M1[i * N + j] - M2[i * N + j]) > tolerance) {
                printf("Mismatch at C[%d][%d]: Expected %.4f, Got %.4f\n", i, j, M2[i*N+j], M1[i*N+j]);
                return false;
            }
        }
    }
    return true;
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

    // calculating the number of operations / second 
     
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
        sq_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);

        cudaError_t err_C_mem = cudaMemcpy(d_C, C, N*N*sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err_C_mem);

        clock_t end_time = clock();

        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;


        
        printf("Matrix size: %dx%d\n", N, N);
        printf("Time taken: %.6f seconds\n", duration);
        gflops(N, duration);
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

        // execution of the actual block
        sq_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);

        cudaError_t err_C_mem = cudaMemcpy(d_C, C, N*N*sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK(err_C_mem);

        clock_t end_time = clock();

        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;


        
        printf("Matrix size: %dx%d\n", N, N);
        printf("Time taken: %.6f seconds\n", duration);
        gflops(N, duration);
        cudaFree(d_A);
        cudaFree(d_C);
        cudaFree(d_B);

    
    }

    return 0;
}