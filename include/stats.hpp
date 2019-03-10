
#pragma once

#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>

/*
 * Implementations of some simple statistics that are not provided 
 * by the standard library (and I couldn't find a header-only lib that implemented
 * them...)
 */

namespace nr {
namespace stats {

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
    return s / static_cast<double>(c.size()); 
}

template<typename Cont>
typename Cont::value_type 
median(Cont c) {
    // don't need(?) to sort, but c shouldn't be too large anyway.
    // so the time shouldn't be too bad.
    
    if(c.size() == 0) {
        // median of empty container is undefined
        throw std::logic_error("cannot take median of empty container");
    }

    std::sort(c.begin(), c.end());

    if(c.size() % 2 == 0) { 
        // if there's an even number, take average of middle two.
        return (c[c.size() / 2 - 1] + c[c.size() / 2]) / 2; 
    }
    else {
        return c[c.size() / 2];
    }
}

}
}
