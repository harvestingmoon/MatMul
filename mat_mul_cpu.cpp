#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <math.h>   
#include <time.h>    

// Corrected matrix multiplication function
void sq_mat_mul(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float val = 0;
            for (int k = 0; k < N; k++) {
                val += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = val;
        }
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

int main() {
    printf("Running Matrix Multiplication Tests with Timing...\n\n");

    int N = 100;
    // --- Test Case 1:
    {
        printf("--- Test Case 1: 1000x1000 Matrix (Sequential Values) ---\n");
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C = (float *)malloc(N * N * sizeof(float));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (float)(i * N + j + 1);
                B[i * N + j] = (float)((N - 1 - i) * N + (N - 1 - j) + 1);
            }
        }

        clock_t start_time = clock();
        sq_mat_mul(A, B, C, N);
        clock_t end_time = clock();

        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Matrix size: %dx%d\n", N, N);
        printf("Time taken: %.6f seconds\n\n", duration);


        free(A);
        free(B);
        free(C);
    }

    // --- Test Case 2:
    {
        printf("--- Test Case 2: 1000x1000 Matrix (Floating-Point Values) ---\n");
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C = (float *)malloc(N * N * sizeof(float));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (float)(i + 1) * 0.1f + (float)(j + 1) * 0.01f;
                B[i * N + j] = (float)(N - i) * 0.05f + (float)(N - j) * 0.005f;
            }
        }



        clock_t start_time = clock();
        sq_mat_mul(A, B, C, N);
        clock_t end_time = clock();

        double duration = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Matrix size: %dx%d\n", N, N);
        printf("Time taken: %.6f seconds\n\n", duration);

        free(A);
        free(B);
        free(C);
    }


    return 0;
}