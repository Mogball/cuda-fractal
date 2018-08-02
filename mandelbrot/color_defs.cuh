#pragma once
#include <array>

struct color {
    __device__ color() {}
    constexpr color(int r, int g, int b) : R(r), G(g), B(b) {}
    int R, G, B;
};

struct color_index {
    constexpr color_index(color c1, color c2, double w) : C1(c1), C2(c2), W(w) {}
    color C1, C2;
    double W;
};

template<size_t L>
using indices = std::array<color_index, L>;

constexpr color black  {   0,   0,   0 };
constexpr color white  { 255, 255, 255 };
constexpr color red1   { 173,  21,  13 };
constexpr color red2   { 240,  99,  64 };
constexpr color red3   { 215,   0,   0 };
constexpr color red4   { 155,   0,   0 };
constexpr color orange { 245, 137 , 50 };
constexpr color grey   { 230, 230, 230 };
constexpr color gold1  { 111,  66,  43 };
constexpr color gold2  { 158,  90,  43 };
constexpr color gold3  { 216, 151,  41 };
constexpr color yellow { 255, 127,   0 };
constexpr color purple { 127,   0, 127 };
constexpr color lrange { 255, 200,   0 };
constexpr color blue   {   0,   0, 255 };
constexpr color dblue  {   0,   0, 128 };

constexpr indices<1> greyScale{
    color_index{ black, white, 1.0 }
};
constexpr indices<6> redOrange{
    color_index{ black,   red1, 0.2 },
    color_index{  red1,   red2, 0.2 },
    color_index{  red2,  white, 0.2 },
    color_index{ white,   red3, 0.5 },
    color_index{  red3,   red4, 0.3 },
    color_index{  red4, orange, 0.2 }
};
constexpr indices<6> blackGoldYellow{
    color_index{ black, gold1, 1.0 },
    color_index{ gold1, gold2, 0.7 },
    color_index{ gold2, gold3, 1.0 },
    color_index{ gold3, white, 0.2 },
    color_index{ white, gold2, 1.2 },
    color_index{ gold2, black, 0.5 }
};
constexpr indices<3> blackYellowPurple{
    color_index{  black, yellow, 1.0 },
    color_index{ yellow, purple, 1.0 },
    color_index{ purple,  black, 0.2 }
};
constexpr indices<5> blackYellowBlue{
    color_index{  black, lrange, 1.0 },
    color_index{ lrange,   grey, 1.0 },
    color_index{   grey,   blue, 1.0 },
    color_index{   blue,  dblue, 1.0 },
    color_index{  dblue,  black, 0.1 }
};

