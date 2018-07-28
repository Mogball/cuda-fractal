#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace std;

struct bitmap_file_header {
    uint8_t  bitmap_type[2];
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset_bits;
} __attribute__((packed));

struct bitmap_image_header {
    uint32_t size_header;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bit_count;
    uint32_t compression;
    uint32_t image_size;
    uint32_t ppm_x;
    uint32_t ppm_y;
    uint32_t clr_used;
    uint32_t clr_important;
} __attribute__((packed));

static_assert(sizeof(bitmap_file_header) == 14, "Expected sizeof(bitmap_file_header) == 14");
static_assert(sizeof(bitmap_image_header) == 40, "Expected sizeof(bitmap_image_header) == 50");

// Data is expected in ARGB format #AARRGGBB
void save_bitmap(const char *file_name, uint32_t width, uint32_t height, uint32_t *data) {
    FILE *image;
    uint32_t image_size = 3 * sizeof(uint8_t) * width * height;
    uint32_t file_size = sizeof(bitmap_file_header) + sizeof(bitmap_image_header) + image_size;
    bitmap_file_header bfh;
    bitmap_image_header bih;
    memset(&bfh, 0, sizeof(bfh));
    memset(&bih, 0, sizeof(bih));

    memcpy(&bfh.bitmap_type, "BM", 2);
    bfh.file_size = file_size;

    bih.size_header = sizeof(bih);
    bih.width = width;
    bih.height = height;
    bih.planes = 1;
    bih.bit_count = 24;
    bih.image_size = image_size;

    image = fopen(file_name, "wb");
    fwrite(&bfh, sizeof(uint8_t), sizeof(bfh), image);
    fwrite(&bih, sizeof(uint8_t), sizeof(bih), image);

    uint8_t color[3];
    for (int i = 0; i < width * height; ++i) {
        /* blue  */ color[0] = (data[i] & 0xff);
        /* green */ color[1] = (data[i] & 0xff00) >> 8;
        /* red   */ color[2] = (data[i] & 0xff0000) >> 16;
        fwrite(color, sizeof(uint8_t), sizeof(color), image);
    }
    fflush(image);
    fclose(image);
}
