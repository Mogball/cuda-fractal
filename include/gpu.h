#pragma once
#include <cstdint>

/**
 * Compute mandelbot values.
 * @param max_x horizontal range [0, max_x)
 * @param max_y vertical   range [0, max_y)
 * @param T     maximum iteration depth
 * @param data  output array for image ##AARRGGBB
 */
extern void gpu_mandelbrot(int max_x, int max_y, long T, uint32_t *data);
