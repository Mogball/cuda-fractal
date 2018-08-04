#include <assert.h>
#include <stdint.h>
#include <gpu.h>
#include <gpu_error.cuh>

#include "color.cuh"

#define MAX_SPECTRUM    16
__constant__ color  c_C[MAX_SPECTRUM];
__constant__ color  c_D[MAX_SPECTRUM];
__constant__ double c_F[MAX_SPECTRUM];
__constant__ double c_Fcul[MAX_SPECTRUM];

static void color_prepare(size_t L,
    const color *c, const color *d,
    const double *F, const double *Fcul) {
    errchk( cudaMemcpyToSymbol(c_C, c, L * sizeof(color)) );
    errchk( cudaMemcpyToSymbol(c_D, d, L * sizeof(color)) );
    errchk( cudaMemcpyToSymbol(c_F, F, L * sizeof(double)) );
    errchk( cudaMemcpyToSymbol(c_Fcul, Fcul, (L + 1) * sizeof(double)) );
}

typedef void (*fprepare_t)(void);
static fprepare_t s_fprepare[] = {
    spec_prepare(greyScale),
    spec_prepare(redOrange),
    spec_prepare(blackGoldYellow),
    spec_prepare(blackYellowPurple),
    spec_prepare(blackYellowBlue)
};

static void color_prepare(render_color color) {
    (*s_fprepare[color])();
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

__device__
uint32_t spectrum_color(long t, long T) {
    double f = (T - t) / (double) T;
    int g = 0;
    while (c_Fcul[g] > f || f > c_Fcul[g + 1]) {
        ++g;
    }
    f -= c_Fcul[g];
    f /= c_F[g];
    int R = c_C[g].R + static_cast<int>(c_D[g].R * f);
    int G = c_C[g].G + static_cast<int>(c_D[g].G * f);
    int B = c_C[g].B + static_cast<int>(c_D[g].B * f);
    return argb(
        static_cast<uint8_t>(R),
        static_cast<uint8_t>(G),
        static_cast<uint8_t>(B));
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

__device__
long julia_compute(double x0, double y0, double tx, double ty, long T) {
    double x = x0;
    double y = y0;
    double xSq = x * x;
    double ySq = y * y;
    long t = 0;
    while (xSq + ySq < 4 && t < T) {
        y = x * y;
        y += y;
        y += ty;
        x = xSq - ySq + tx;
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
            *(data + i_x + i_y * config.max) = spectrum_color(t, config.T);
        }
    }
}

__global__
void julia(render_config config, julia_config jul, uint32_t *data) {
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
            t = julia_compute(x0, y0, jul.tx, jul.ty, config.T);
            *(data + i_x + i_y * config.max) = spectrum_color(t, config.T);
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

    // Prepare color tables
    color_prepare(config.color);

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

void gpu_julia(const render_config &config, uint32_t *data) {
    // Allocate device image data
    uint32_t *device_data;
    uint32_t num_pixels = config.max * config.max;
    uint32_t num_bytes = num_pixels * sizeof(uint32_t);
    errchk( cudaMalloc(&device_data, num_bytes) );

    // Prepare color tables
    color_prepare(config.color);

    // Grab extended configuration
    julia_config *jul = (julia_config *) config.extra;
    assert(NULL != jul);

    // Launch Julia kernels
    dim3 block_dim(16, 16);
    dim3 block_num(
        (config.max + block_dim.x - 1) / block_dim.x,
        (config.max + block_dim.y - 1) / block_dim.y
    );
    julia<<<block_num, block_dim>>>(config, *jul, device_data);
    errchk( cudaPeekAtLastError()   );
    errchk( cudaDeviceSynchronize() );

    // Copy data into host memory
    errchk( cudaMemcpy(data, device_data, num_bytes, cudaMemcpyDeviceToHost) );
    errchk( cudaFree(device_data) );
}
