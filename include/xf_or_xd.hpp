
#pragma once

#include <Eigen/Core>
#include <type_traits>

/*
 * Helper classes to determine if something should use Xd or Xf based on
 * the component type. These could probably be combined into one that takes a
 * matrix or vector parameter, but I think this is more readable
 */

namespace nr {

template <typename Component>
class VectorXf_or_Xd : public std::conditional<
                           /*if*/ std::is_same<Component, double>::value,
                           /*then*/ Eigen::VectorXd,
                           /*else*/ Eigen::VectorXf> {};

template <typename Component>
class MatrixXf_or_Xd : public std::conditional<
                           /*if*/ std::is_same<Component, double>::value,
                           /*then*/ Eigen::MatrixXd,
                           /*else*/ Eigen::MatrixXf> {};

} // namespace nr
