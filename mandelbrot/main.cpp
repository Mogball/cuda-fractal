#include <gpu.h>
#include <bitmap.h>
#include <stdio.h>
#include <stdint.h>

static constexpr int max_x = 6144;
static constexpr int max_y = 4096;
static constexpr int T = 1 << 16;

int main(void) {
    static uint32_t data[max_x * max_y];
    printf("Running GPU kernel .... ");
    gpu_mandelbrot(max_x, max_y, T, data);
    printf("Done!\n");

    printf("Saving fractal image .. ");
    save_bitmap("fractal.bmp", max_x, max_y, data);
    printf("Done!\n");
}
