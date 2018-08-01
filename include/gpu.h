#pragma once
#include <cstdint>

enum render_color {
    GreyScale,
    RedOrange,
    BlackGoldYellow,
    BlackYellowPurple,
    BlackYellowBlue
};

struct render_config {
    int max;   /* Image pixel width and height */
    long T;    /* Maximum iteration depth      */
    double xc; /* Fractal center x             */
    double yc; /* Fractal center y             */
    double s;  /* Fractal zoom factor          */

    render_color color; /* Image coloring type */

    void *extra; /* Fractal-specific configuration */
};

struct julia_config {
    double tx; /* Julia set focus x */
    double ty; /* Julia set focus y */
};

/**
 * Compute Mandelbrot values.
 * @param config fractal rendering config
 * @param data   output array for image ##AARRGGBB
 */
extern void gpu_mandelbrot(const render_config &config, uint32_t *data);

/**
 * Compute Julia values. Provide focus values with
 * the extra configuration.
 * @param config fractal rendering config
 * @param data   output array for image ##AARRGGBB
 */
extern void gpu_julia(const render_config &config, uint32_t *data);
