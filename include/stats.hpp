
#pragma once

#include <numeric>
#include <algorithm>
#include <cmath>

/*
 * Implementations of some simple statistics that are not provided 
 * by the standard library.
 */


namespace NR_stats {

template<typename Cont>
double mean(const Cont& c) {
    auto sum = std::accumulate(c.begin(), c.end(), 0.0);
    return sum / static_cast<double>(c.size());
}

template<typename Cont>
double variance(const Cont& c) {
    double m = mean(c);
    double s = std::accumulate(c.begin(), c.end(), 0.0,
                    [m](auto x, auto y) { return x + std::pow(y - m, 2); });
    return s / static_cast<decltype(s)>(c.size()); 
}

}
