#pragma once
#include <cstdint>

/**
 * Save bitmap image with given width and height.
 *
 * @param file_name name of bitmap file
 * @param width     image width  (pixels)
 * @param height    image height (pixels)
 * @param dpi       dots per inch (recommend 96)
 * @param data      image data as array of ##AARRGGBB
 */
extern void save_bitmap(const char *file_name, uint32_t width, uint32_t height, uint32_t *data);
