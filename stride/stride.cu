#include <stdio.h>
#include <math.h>
#include <gpu_error.cuh>

__global__
void saxpy(int n, float a, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main(void) {
    int N = 1 << 20;
    float *x, *y;

    // Allocate unified memory accessible from host and device
    errchk( cudaMallocManaged(&x, N * sizeof(float)) );
    errchk( cudaMallocManaged(&y, N * sizeof(float)) );

    for (int i = 0; i < N; ++i) {
        x[i] = 3.0f;
        y[i] = 4.0f;
    }

    int blockSize = 1 << 8;
    int numBlocks = 1 << 10;
    saxpy<<<numBlocks, blockSize>>>(N, 2.0f, x, y);
    errchk( cudaPeekAtLastError() );
    errchk( cudaDeviceSynchronize() );

    float max_error = 0.0f;
    float expected = 10.0f;
    for (int i = 0; i < N; ++i) {
        max_error = max(max_error, fabs(y[i] - expected));
    }
    printf("Max error: %.5f\n", max_error);

    // Free memory
    errchk( cudaFree(x) );
    errchk( cudaFree(y) );
}
