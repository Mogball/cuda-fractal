#include <stdio.h>
#include <gpu_error.cuh>

/* Thread and blocks are one-dimensional
    ___________   ___________   ___________
   |0|1|2|3|4|5| |0|1|2|3|4|5| |0|1|2|3|4|5|
   Block 0,      Block 1,      Block 2,
   Block X: 6

   Global index:
   0,  1,  2,  3,  4,  5,
   6,  7,  8,  9,  10, 11,
   12, 13, 14, 15, 16, 17
*/

__global__
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check in case there are extra threads
    if (i < n) { y[i] = a * x[i] + y[i]; }
}

int main(void) {
    // Size of vectors for operation
    int N = 1 << 20;
    // Host arrays
    float *x, *y;
    // Device arrays
    float *d_x, *d_y;

    // Allocate host data
    x = (float *) malloc(N * sizeof(float));
    y = (float *) malloc(N * sizeof(float));

    // Allocate device data
    errchk( cudaMalloc(&d_x, N * sizeof(float)) );
    errchk( cudaMalloc(&d_y, N * sizeof(float)) );

    // Fill host array with dummy data
    for (int i = 0; i < N; ++i) {
        x[i] = 3.0f;
        y[i] = 4.0f;
    }

    // Copy data to device
    errchk( cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice) );
    errchk( cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice) );

    // Perform SAXPY kernel on elements, one thread per array index
    // 256 threads per block, and ensure that we round up
    // to the correct number of thread blocks
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
    errchk( cudaPeekAtLastError() );
    errchk( cudaDeviceSynchronize() );

    // Copy results back to host
    errchk( cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost) );

    // Compute max error
    float max_error = 0.0f;
    float expected = 10.0f; // 2 * 3 + 4
    for (int i = 0; i < N; ++i) {
        max_error = max(max_error, abs(y[i] - expected));
    }

    printf("Maximum error: %.5f\n", max_error);

    // Free memory
    free(x);
    free(y);
    errchk( cudaFree(d_x) );
    errchk( cudaFree(d_y) );
}
