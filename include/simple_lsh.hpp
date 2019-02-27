

#pragma once

#include <Eigen/Core>
#include <random>

#include "normal_matrix.hpp"

/*
 * SimpleLSH family.
 */

namespace nr {

class SimpleLSH {
private:
    Eigen::MatrixXd a;
    int64_t bits;
    int64_t dim;

    Eigen::VectorXd bit_mask;

public:
    SimpleLSH(int64_t bits, int64_t dim): 
        a(Eigen::MatrixXd(bits, dim+1)),
        bits(bits),
        dim(dim),
        bit_mask(bits)
    {
        NormalMatrix nm;
        nm.fill_matrix(a); 

        fill_bit_mask();
    }

    int64_t bit_count() const {
        return bits;
    }

    int64_t dimension() const {
        return dim;
    }

    void fill_bit_mask() {
        // fill this->bit_mask with powers of 2.
        for(int64_t i = 0; i < bits; ++i) {
            bit_mask(i) = std::pow(2, i);
        }
    }

    template<typename T>
    T P(T input) {
        // symmetric transform that appends sqrt(1 - ||input||) to input
        T append(dim+1);
        append << input, std::sqrt(1 - input.norm()); // append sqrt to input.
        return append;
    }

    template<typename T>
    T numerals_to_bits(T input) {
        // if a value is positive, it's bit is 1, otherwise 0.
        // can be in place because input arg is a copy. 
        for(int64_t i = 0; i < input.rows(); ++i) {
            input(i) = input(i) > 0 ? 1 : 0; 
        }
        return input;
    }

    template<typename T>
    int64_t operator()(T input) {
        T simple = P(input);
        T prods = a * simple; 
        T bit_vect = numerals_to_bits(prods);
        return bit_vect.dot(bit_mask);
    }
};

}
