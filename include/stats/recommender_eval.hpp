

#pragma once

namespace nr {
namespace stats {

template<typename Cont>
float recall(const Cont &actual, const Cont &predicted) {
  /*
  * this version of recall returns the fraction of predicted values that are
  * also in true.
  * each Cont should contain a pair of (vector, some id type)
  * this function checks if vectors are approximately the same.
  */
  size_t found = 0;
  for(auto& elem : predicted) {
    for(size_t i = 0; i < actual.size(); ++i) {
      if(elem.isApprox(actual.at(i))) {
        ++found;
      }
    }
  }
  return static_cast<float>(found) / static_cast<float>(actual.size());
}

} // stats
} // nr
