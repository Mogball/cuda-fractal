#include <bitmap.h>
#include <cstdint>

using namespace std;

static constexpr uint32_t width = 400;
static constexpr uint32_t height = 400;

inline uint32_t argb(uint8_t r, uint8_t g, uint8_t b) {
    return (0xff << 24) | (r << 16) | (g << 8) | b;
}

int main(void) {
    uint32_t data[width * height];
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int a = y * width + x;
            if ((x > 50 && x < 350) && (y > 50 && y < 350)) {
                data[a] = argb(255, 255, 5);
            } else {
                data[a] = argb(55, 55, 55);
            }
        }
    }
    save_bitmap("black_border.bmp", width, height, data);
}
