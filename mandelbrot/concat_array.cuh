#pragma once
#include <array>

template<typename T, size_t N1, size_t... I1, size_t N2, size_t... I2>
constexpr std::array<T, N1 + N2> concat(
    const std::array<T, N1> &a1,
    const std::array<T, N2> &a2,
    std::index_sequence<I1...>,
    std::index_sequence<I2...>) {
    return { a1[I1]..., a2[I2]... };
}

template<typename T, size_t N1, size_t N2>
constexpr std::array<T, N1 + N2> concat(
    const std::array<T, N1> &a1,
    const std::array<T, N2> &a2) {
    return concat(a1, a2, std::make_index_sequence<N1>(), std::make_index_sequence<N2>());
}
