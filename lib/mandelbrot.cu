#include <stdint.h>
#include <gpu_error.cuh>

#define MIN_X   (-2.0)
#define MAX_X   (+1.0)
#define MIN_Y   (-1.0)
#define MAX_Y   (+1.0)
#define RANGE_X (MAX_X - MIN_X)
#define RANGE_Y (MAX_Y - MIN_Y)

__device__
uint32_t argb(uint8_t r, uint8_t g, uint8_t b) {
    return (0xff << 24) | (r << 16) | (g << 8) | b;
}

__device__
uint32_t color_greyscale(long t, long T) {
    uint8_t grey = (uint8_t) (t / (double) T * 0xff);
    return argb(grey, grey, grey);
}

__device__
long mandelbrot_compute(double x0, double y0, long T) {
    double x = 0;
    double y = 0;
    double xSq = 0;
    double ySq = 0;
    long t = 0;
    while (xSq + ySq < 4 && t < T) {
        y = x * y;
        y += y;
        y += y0;
        x = xSq - ySq + x0;
        xSq = x * x;
        ySq = y * y;
        ++t;
    }
    return t;
}

__global__
void mandelbrot(int max_x, int max_y, long T, uint32_t *data) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    double xm = (double) max_x;
    double ym = (double) max_y;
    double x0, y0;
    long t;
    for (int i_x = index_x; i_x < max_x; i_x += stride_x) {
        for (int i_y = index_y; i_y < max_y; i_y += stride_y) {
            x0 = i_x / xm * RANGE_X + MIN_X;
            y0 = i_y / ym * RANGE_Y + MIN_Y;
            t = mandelbrot_compute(x0, y0, T);
            *(data + i_x + i_y * max_x) = color_greyscale(t, T);
        }
    }
}

void gpu_mandelbrot(int max_x, int max_y, long T, uint32_t *data) {
    uint32_t *device_data;
    errchk( cudaMalloc(&device_data, max_x * max_y * sizeof(uint32_t)) );

    dim3 block_dim(16, 16);
    dim3 block_num(
        (max_x + block_dim.x - 1) / block_dim.x,
        (max_y + block_dim.y - 1) / block_dim.y
    );
    mandelbrot<<<block_num, block_dim>>>(max_x, max_y, T, device_data);
    errchk( cudaPeekAtLastError()   );
    errchk( cudaDeviceSynchronize() );

    errchk( cudaMemcpy(data, device_data, max_x * max_y * sizeof(uint32_t), cudaMemcpyDeviceToHost) );
    errchk( cudaFree(device_data) );
}
