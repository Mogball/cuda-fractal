#pragma once
#include "make_array.cuh"
#include "generate_array.cuh"

template<size_t g, size_t m>
struct cumulative {
    template<size_t L>
    static constexpr double apply(const indices<L> &a) {
        return a[g].W + cumulative<g + 1, m>::apply(a);
    }
};

template<size_t m>
struct cumulative<m, m> {
    template<size_t L>
    static constexpr double apply(const indices<L> &) {
        return 0;
    }
};

template<size_t L>
struct weight_get {
    constexpr explicit weight_get(const indices<L> &a, double t) : A(a), T(t) {}
    constexpr double operator()(size_t i) {
        return A[i].W / T;
    }
    const indices<L> &A;
    double T;
};

template<size_t L, size_t... Is>
constexpr auto generate_cul(const indices<L> &a, double tot, std::index_sequence<Is...>)
-> std::array<double, L + 1> {
    return {{ ret(cumulative<0, Is>::apply(a) / tot)... }};
}

template<size_t L>
constexpr auto generate_cul(const indices<L> &a, double tot)
-> std::array<double, L + 1> {
    return generate_cul<L>(a, tot, std::make_index_sequence<L + 1>());
}

template<size_t L>
struct color_get {
    constexpr explicit color_get(const indices<L> &a) : A(a) {}
    constexpr color operator()(size_t i) {
        return A[i].C1;
    }
    const indices<L> &A;
};

template<size_t L>
struct diff_get {
    constexpr explicit diff_get(const indices<L> &a) : A(a) {}
    constexpr color operator()(size_t i) {
        return {
            static_cast<int>(A[i].C2.R - A[i].C1.R),
            static_cast<int>(A[i].C2.G - A[i].C1.G),
            static_cast<int>(A[i].C2.B - A[i].C1.B)
        };
    }
    const indices<L> &A;
};

#define weight_sum(a) \
cumulative<0, a.size()>::apply(a)

#define get_F(a) \
make_array<a.size(), weight_get<a.size()>>(weight_get<a.size()>{a, weight_sum(a)})

#define get_Fcul(a) \
generate_cul<a.size()>(a, weight_sum(a))

#define get_C(a) \
make_array<a.size(), color_get<a.size()>>(color_get<a.size()>{a})

#define get_D(a) \
make_array<a.size(), diff_get<a.size()>>(diff_get<a.size()>{a})
