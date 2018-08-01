#pragma once
#include <array>
#include <tuple>
#include <utility>

template<typename Function, size_t... Indices>
constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>)
-> std::array<typename std::result_of<Function(size_t)>::type, sizeof...(Indices)>
{ return {{ f(Indices)... }}; }

template<int N, typename Function>
constexpr auto make_array(Function f)
-> std::array<typename std::result_of<Function(size_t)>::type, N>
{ return make_array_helper(f, std::make_index_sequence<N>{}); }
