#include "color_defs.cuh"
#include "const_util.cuh"

#define spec_get(ci, field) ci##_##field
#define spec_F(ci) spec_get(ci, F)
#define spec_C(ci) spec_get(ci, C)
#define spec_D(ci) spec_get(ci, D)
#define spec_Fcul(ci) spec_get(ci, Fcul)
#define spec_color(ci) spec_get(ci, color)

#define spectrum_declare_fields(ci)     \
constexpr auto spec_F(ci) = get_F(ci);  \
constexpr auto spec_C(ci) = get_C(ci);  \
constexpr auto spec_D(ci) = get_D(ci);  \
constexpr auto spec_Fcul(ci) = get_Fcul(ci);

#define spectrum_declare(ci)                                         \
spectrum_declare_fields(ci);                                         \
static uint32_t spec_color(ci)(long t, long T) {                     \
    double f = t / (double) T;                                       \
    int g = 0;                                                       \
    while (spec_Fcul(ci)[g] > f || f > spec_Fcul(ci)[g + 1])         \
    { ++g; }                                                         \
    f -= spec_Fcul(ci)[g];                                           \
    f /= spec_F(ci)[g];                                              \
    int R = spec_C(ci)[g].R + static_cast<int>(spec_D(ci)[g].R * f); \
    int G = spec_C(ci)[g].G + static_cast<int>(spec_D(ci)[g].G * f); \
    int B = spec_C(ci)[g].B + static_cast<int>(spec_D(ci)[g].B * f); \
    return (0xff << 24) | (R << 16) | (G << 8) | B;                  \
}

spectrum_declare(greyScale);
spectrum_declare(redOrange);
spectrum_declare(blackGoldYellow);
spectrum_declare(blackYellowPurple);
spectrum_declare(blackYellowBlue);
