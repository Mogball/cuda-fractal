#pragma once
#include <cstdint>

struct render_config {
    int max;   /* Image pixel width and height */
    long T;    /* Maximum iteration depth      */
    double xc; /* Fractal center x             */
    double yc; /* Fractal center y             */
    double s;  /* Fractal zoom factor          */
};

/**
 * Compute Mandelbrot values.
 * @param config fractal rendering config
 * @param data   output array for image ##AARRGGBB
 */
extern void gpu_mandelbrot(const render_config &config, uint32_t *data);
