

#pragma once

namespace nr {
namespace stats {

/*
* Simple version of unique that assumes Sub has proper operator== defined.
*/

template <template<typename Sub> typename Cont, typename Sub>
std::pair<Cont<Sub>, Cont<size_t>> unique(const Cont<Sub> &c) {
  // return unique elements of the input and their indices into the input.
  // returned in order that the elements first appear in.
  // - I.e., unique(2, 2, 1, 3) -> {2, 1, 3}.
  Cont<Sub> unique_values(0);
  Cont<size_t> unique_idx(0);
  size_t idx = 0;
  for(const Sub &elem : c) {
    auto found_iter = std::find(unique_values.begin(), unique_values.end(), elem);

    // if elem is not in unique_values, insert it.
    if(found_iter == unique_values.end()) {
      unique_values.push_back(elem);
      unique_idx.push_back(idx);
    }
    ++idx; // increment index counter.
  }
  return std::make_pair(unique_values, unique_idx);
}

/*
* Version of unique that takes an equality comparator for Sub
*/

template<template<typename Sub> typename Cont, typename Sub, typename Equal>
bool contains(const Cont<Sub> &c, const Sub& item, Equal equal) {
  // checks if c contains item. Effectively the same as find.
  for(auto& elem : c) {
    if(equal(elem, item)) {
      return true;
    }
  }
  return false;
}

template <template<typename Sub> typename Cont, typename Sub, typename Equal>
std::pair<Cont<Sub>, Cont<size_t>> unique(const Cont<Sub> &c, Equal equal) {
  // return unique elements of the input and their indices into the input.
  // returned in order that the elements first appear in.
  // - I.e., unique(2, 2, 1, 3) -> {2, 1, 3}.
  Cont<Sub> unique_values(0);
  Cont<size_t> unique_idx(0);
  size_t idx = 0;
  for(const Sub &elem : c) {
    bool found = contains(unique_values, elem, equal);
    // if elem is not in unique_values, insert it!
    if(!found) {
      unique_values.push_back(elem);
      unique_idx.push_back(idx);
    }
    ++idx; // increment index counter.
  }
  return std::make_pair(unique_values, unique_idx);
}

} // stats
} // nr
