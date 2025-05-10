#include <cuda.h>
#include <iostream>
#include <vector>
#include <cmath>   // For ceil
#include <iomanip> // For printing (std::fixed, std::setprecision)
#include <cstdio>  // For printf used in DRIVER_API_CHECK if you prefer

// CUDA Driver API error checking macro
#define DRIVER_API_CHECK(err)                                                                  \
    do {                                                                                       \
        if (err != CUDA_SUCCESS) {                                                             \
            const char *err_str;                                                               \
            cuGetErrorString(err, &err_str);                                                   \
            /* Using fprintf for cerr consistency, or std::cerr */                             \
            fprintf(stderr, "CUDA Driver API Error: %s in %s at line %d\n", err_str, __FILE__, \
                    __LINE__);                                                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

// Helper function to print a matrix (optional)
void print_matrix(const char *name, const float *M, int N_print) { // Renamed N to N_print
    std::cout << name << " (Size " << N_print << "x" << N_print << "):" << std::endl;
    for (int i = 0; i < N_print; i++) {
        for (int j = 0; j < N_print; j++) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << M[i * N_print + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    CUdeviceptr d_A, d_B, d_C; 

    CUevent start_event, stop_event;
    float milliseconds = 0;

   
    const int N = 1024; 
    std::cout << "Matrix size N = " << N << std::endl;
    size_t bytes = (size_t)N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N); 

    // Initialize host matrices A and B
    // (Using simple sequential values for demonstration)
    std::cout << "Initializing host matrices..." << std::endl;
    for (long long i = 0; i < (long long)N * N; ++i) { // Use long long for large N*N
        h_A[i] = (float)(i % N + 1); // Example data
        h_B[i] = (float)((i / N) + 1); // Example data
    }

    // 1. Initialize CUDA
    CUresult err = cuInit(0);
    DRIVER_API_CHECK(err);

    err = cuDeviceGet(&cuDevice, 0); // Get first device
    DRIVER_API_CHECK(err);

    err = cuCtxCreate(&cuContext, 0, cuDevice);
    DRIVER_API_CHECK(err);

    // Create CUDA events
    err = cuEventCreate(&start_event, CU_EVENT_DEFAULT);
    DRIVER_API_CHECK(err);
    err = cuEventCreate(&stop_event, CU_EVENT_DEFAULT);
    DRIVER_API_CHECK(err);

    // 2. Load PTX module
    const char* ptx_file_path = "mat_mul_gpu.ptx";
    std::cout << "Loading PTX file: " << ptx_file_path << "..." << std::endl;
    err = cuModuleLoad(&cuModule, ptx_file_path);
    DRIVER_API_CHECK(err);

    // 3. Get kernel handle
    const char* kernel_name = "_Z17sq_mat_mul_kernelPfS_S_i"; // Your mangled kernel name
    std::cout << "Getting kernel function: " << kernel_name << "..." << std::endl;
    err = cuModuleGetFunction(&cuFunction, cuModule, kernel_name);
    DRIVER_API_CHECK(err);

    std::cout << "PTX module loaded and kernel function retrieved successfully." << std::endl;

    // 4. Allocate device memory
    std::cout << "Allocating device memory (" << bytes / (1024.0*1024.0) << " MB each)..." << std::endl;
    err = cuMemAlloc(&d_A, bytes);
    DRIVER_API_CHECK(err);
    err = cuMemAlloc(&d_B, bytes);
    DRIVER_API_CHECK(err);
    err = cuMemAlloc(&d_C, bytes);
    DRIVER_API_CHECK(err);

    // Copy host matrices A and B to device memory
    std::cout << "Copying H_A and H_B from Host to Device..." << std::endl;
    err = cuMemcpyHtoD(d_A, h_A.data(), bytes);
    DRIVER_API_CHECK(err);
    err = cuMemcpyHtoD(d_B, h_B.data(), bytes);
    DRIVER_API_CHECK(err);

    // 5. Set up kernel parameters
    int N_kernel_arg = N; 
    void* args[] = { &d_A, &d_B, &d_C, &N_kernel_arg };

    // 6. Launch the kernel & Time it
    unsigned int blockDimX = 32;
    unsigned int blockDimY = 32;
    unsigned int blockDimZ = 1;

    unsigned int gridDimX = (unsigned int)std::ceil((float)N / blockDimX);
    unsigned int gridDimY = (unsigned int)std::ceil((float)N / blockDimY);
    unsigned int gridDimZ = 1;

    std::cout << "Launching kernel with Grid(" << gridDimX << "," << gridDimY << ") and Block("
              << blockDimX << "," << blockDimY << ")..." << std::endl;

    // Record start event
    err = cuEventRecord(start_event, 0); // 0 for default stream
    DRIVER_API_CHECK(err);

    err = cuLaunchKernel(cuFunction,
                         gridDimX, gridDimY, gridDimZ,
                         blockDimX, blockDimY, blockDimZ,
                         0,        
                         0,        
                         args,    
                         NULL);   
    DRIVER_API_CHECK(err);

    err = cuEventRecord(stop_event, 0);
    DRIVER_API_CHECK(err);
    err = cuEventSynchronize(stop_event);
    DRIVER_API_CHECK(err);

    // Calculate elapsed time
    err = cuEventElapsedTime(&milliseconds, start_event, stop_event);
    DRIVER_API_CHECK(err);

    std::cout << "Kernel execution finished." << std::endl;
    printf("Kernel execution time: %.6f ms (%.6f seconds)\n", milliseconds, milliseconds / 1000.0);


    // 7. Copy results (matrix C) from device to host
    std::cout << "Copying H_C from Device to Host..." << std::endl;
    err = cuMemcpyDtoH(h_C.data(), d_C, bytes);
    DRIVER_API_CHECK(err);

    if (N > 0) {
        std::cout << "Result H_C[0]: " << h_C[0] << std::endl;
        if (N*N > 1) std::cout << "Result H_C[last]: " << h_C[N*N-1] << std::endl;

    }


    // 8. Clean up
    std::cout << "Freeing device memory and CUDA events..." << std::endl;
    err = cuMemFree(d_A); DRIVER_API_CHECK(err);
    err = cuMemFree(d_B); DRIVER_API_CHECK(err);
    err = cuMemFree(d_C); DRIVER_API_CHECK(err);

    err = cuEventDestroy(start_event); DRIVER_API_CHECK(err);
    err = cuEventDestroy(stop_event); DRIVER_API_CHECK(err);

    err = cuModuleUnload(cuModule); DRIVER_API_CHECK(err);
    err = cuCtxDestroy(cuContext); DRIVER_API_CHECK(err);

    std::cout << "CUDA resources cleaned up. Program finished successfully." << std::endl;

    return 0;
}