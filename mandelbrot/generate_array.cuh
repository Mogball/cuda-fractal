#pragma once
#include <utility>

template<typename T>
constexpr auto ret(const T &t) -> T { return t; }

template<typename Ret, typename Arg, template<size_t> typename Func, size_t... Is>
constexpr auto generate_array(const Arg arg, std::index_sequence<Is...>)
-> std::array<Ret, sizeof...(Is)> {
    return {{ ret(Func<Is>(arg))... }};
}

template<typename Ret, typename Arg, template<size_t> typename Func, size_t N>
constexpr auto generate_array(const Arg arg) {
    return generate_array<Ret, Arg, Func>(arg, std::make_index_sequence<N>());
}

/*template<typename T, size_t... Is>
constexpr auto generate_array(std::index_sequence<Is...>)
-> std::array<T, sizeof...(Is)> {
    return {{ ret(std::integral_constant<long, Is>::value)... }};
}

template<typename T, size_t N>
constexpr auto generate_array() {
    return generate_array<T>(std::make_index_sequence<N>());
}*/
