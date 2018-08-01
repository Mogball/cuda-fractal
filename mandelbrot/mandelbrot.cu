#include <stdint.h>
#include <gpu.h>
#include <gpu_error.cuh>

#include "color.cuh"

#define MAX_SPECTRUM    16
__constant__ color  c_C[MAX_SPECTRUM];
__constant__ color  c_D[MAX_SPECTRUM];
__constant__ double c_F[MAX_SPECTRUM];
__constant__ double c_Fcul[MAX_SPECTRUM];

static void color_prepare(size_t L, color *c, color *d, double *F, double *Fcul) {
    errchk( cudaMemcpyToSymbol(c_C,    c,    L * sizeof(color))  );
    errchk( cudaMemcpyToSymbol(c_d,    d,    L * sizeof(color))  );
    errchk( cudaMemcpyToSymbol(c_F,    F,    L * sizeof(double)) );
    errchk( cudaMemcpyToSymbol(c_Fcul, Fcul, L * sizeof(double)) );
}

/**
 * Device function to compute ARGB pixel.
 * @param r 8-bit red value
 * @param g 8-bit green value
 * @param b 8-bit blue value
 * @return ARGB pixel with max alpha
 */
__device__
uint32_t argb(uint8_t r, uint8_t g, uint8_t b) {
    return (0xff << 24) | (r << 16) | (g << 8) | b;
}

/**
 * Basic coloring with greyscale.
 * @param t escape time for the pixel
 * @param T max iterations
 * @return greyscale ARGB pixel
 */
__device__
uint32_t color_greyscale(long t, long T) {
    uint8_t grey = (uint8_t) (t / (double) T * 0xff);
    return argb(grey, grey, grey);
}

/**
 * Device function to compute Mandelbrot escape time for
 * a particular pixel, or coordinate (x0, y0).
 * @param x0 x-value
 * @param y0 y-value
 * @param T  max iterations
 * @return Mandelbrot escape time
 */
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

/**
 * Kernel to compute Mandelbrot value at pixels.
 * @param config fractal rendering config
 * @param data   array to store pixel data
 */
__global__
void mandelbrot(render_config config, uint32_t *data) {
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    double m = (double) config.max;
    double x0, y0;
    long t;
    for (int i_x = index_x; i_x < m; i_x += stride_x) {
        for (int i_y = index_y; i_y < m; i_y += stride_y) {
            x0 = (i_x / m - 0.5) * config.s + config.xc;
            y0 = (i_y / m - 0.5) * config.s + config.yc;
            t = mandelbrot_compute(x0, y0, config.T);
            *(data + i_x + i_y * config.max) = color_greyscale(t, config.T);
        }
    }
}

/**
 * Exposed function to used GPU to compute Mandelbrot data.
 * @param config fractal rendering configuration
 * @param data   array to store pixel data
 */
void gpu_mandelbrot(const render_config &config, uint32_t *data) {
    // Allocate device image data
    uint32_t *device_data;
    uint32_t num_pixels = config.max * config.max;
    uint32_t num_bytes = num_pixels * sizeof(uint32_t);
    errchk( cudaMalloc(&device_data, num_bytes) );

    // Launch Mandelbrot kernels
    dim3 block_dim(16, 16);
    dim3 block_num(
        (config.max + block_dim.x - 1) / block_dim.x,
        (config.max + block_dim.y - 1) / block_dim.y
    );
    mandelbrot<<<block_num, block_dim>>>(config, device_data);
    errchk( cudaPeekAtLastError()   );
    errchk( cudaDeviceSynchronize() );

    // Copy data into host memory
    errchk( cudaMemcpy(data, device_data, num_bytes, cudaMemcpyDeviceToHost) );
    errchk( cudaFree(device_data) );
}
