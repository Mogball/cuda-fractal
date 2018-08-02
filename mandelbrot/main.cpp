#include <gpu.h>
#include <bitmap.h>
#include <stdio.h>
#include <stdint.h>

constexpr int fractal_max = 1028;
constexpr int fractal_depth = 1 << 4;

int main(void) {
    static uint32_t data[fractal_max * fractal_max];
    render_config config = {fractal_max, fractal_depth, -0.5, 0.0, 3.0};
    printf("Running GPU kernel .... ");
    gpu_mandelbrot(config, data);
    printf("Done!\n");

    printf("Saving fractal image .. ");
    save_bitmap("fractal.bmp", fractal_max, fractal_max, data);
    printf("Done!\n");
}
