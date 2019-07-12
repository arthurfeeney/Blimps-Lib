
#pragma once

#include <algorithm>

namespace nr {
namespace stats {

template <typename Cont>
static void shift_left_from_inplace(size_t idx, Cont &c) {
  // starting at index, shift everything one index to the left.
  // Overwriting the first element.
  for (size_t i = 0; i < idx; ++i) {
    c.at(i) = c.at(i + 1);
  }
}

/*
 * Version of topk that uses operator<
 */

template <template <typename Sub> typename Cont, typename Sub>
std::optional<size_t> insert_inplace(Sub to_insert, Cont<Sub> &c) {
  // maintains sorted ordering and size of c. Removing smallest value
  // if to_insert is the smallest, it is not inserted.
  // returns index inserted at.

  // start from the largest element.
  for (auto i = c.end() - 1; i >= c.begin(); --i) {
    if (to_insert > *i) {
      // shift left and insert.
      shift_left_from_inplace(i - c.begin(), c);
      *i = to_insert;
      return i - c.begin();
    }
  }
  return {};
}

template <template <typename Sub> typename Cont, typename Sub>
std::pair<Cont<Sub>, Cont<size_t>> topk(int64_t k, const Cont<Sub> &c) {
  // returns a pair. The first element is the topk values of the input.
  // The second element of the pair contains the indices of the topk
  if (k < 1) {
    throw std::logic_error("stats::topk, k must be positive.");
  }
  if (static_cast<size_t>(k) > c.size()) {
    // if c is smaller than k, return the input.
    Cont<size_t> indices(c.size());
    std::iota(indices.begin(), indices.end(), 0);
    return std::make_pair(c, indices);
  }

  // insert first k values of c into topk.
  Cont<Sub> topk(k);

  Cont<size_t> topk_idx(k, 0);
  std::iota(topk_idx.begin(), topk_idx.end(), 0); // fill with initial indices.
  std::sort(topk_idx.begin(), topk_idx.end(),
            [&c](size_t x, size_t y) { return c.at(x) < c.at(y); });

  for (size_t idx = 0; idx < static_cast<size_t>(k); ++idx) {
    topk.at(idx) = c.at(topk_idx.at(idx));
  }

  for (size_t idx = k; idx < c.size(); ++idx) {
    std::optional<size_t> inserted = insert_inplace(c.at(idx), topk);
    if (inserted) {
      shift_left_from_inplace(inserted.value(), topk_idx);
      topk_idx.at(inserted.value()) = idx;
    }
  }
  return make_pair(topk, topk_idx);
}

/*
 * Version of topk that takes a comparator function, rather than using
 * operator<
 */

template <template <typename Sub> typename Cont, typename Sub, typename Greater>
std::optional<size_t> insert_inplace(Sub to_insert, Cont<Sub> &c,
                                     Greater greater) {
  // maintains sorted ordering and size of c. Removing smallest value
  // if to_insert is the smallest, it is not inserted.
  // returns index inserted at.

  // start from the largest element.
  for (auto i = c.end() - 1; i >= c.begin(); --i) {
    if (greater(to_insert, *i)) {
      // shift left and insert.
      shift_left_from_inplace(i - c.begin(), c);
      *i = to_insert;
      return i - c.begin();
    }
  }
  return {};
}

template <template <typename Sub> typename Cont, typename Sub, typename Less,
          typename Greater>
std::pair<Cont<Sub>, Cont<size_t>> topk(int64_t k, const Cont<Sub> &c,
                                        Less less, Greater greater) {
  // returns a pair. The first element is the topk values of the input.
  // The second element of the pair contains the indices of the topk
  if (k < 1) {
    throw std::logic_error("stats::topk, k must be positive.");
  }
  if (static_cast<size_t>(k) > c.size()) {
    // if c is smaller than k, return the input.
    Cont<size_t> indices(c.size());
    std::iota(indices.begin(), indices.end(), 0);
    return std::make_pair(c, indices);
  }

  Cont<Sub> topk(k);

  // initialize the topk indices with the first k indices
  // sort them so they are in the correct order.
  Cont<size_t> topk_idx(k, 0);
  std::iota(topk_idx.begin(), topk_idx.end(), 0);
  std::sort(topk_idx.begin(), topk_idx.end(),
            [&c, &less](size_t x, size_t y) { return less(c.at(x), c.at(y)); });

  // use the sorted indices to fill topk with initial values.
  // the first k values of c, but in sorted order.
  for (size_t idx = 0; idx < static_cast<size_t>(k); ++idx) {
    topk.at(idx) = c.at(topk_idx.at(idx));
  }

  // find the topk and track their indices.
  for (size_t idx = k; idx < c.size(); ++idx) {
    std::optional<size_t> inserted = insert_inplace(c.at(idx), topk, greater);
    if (inserted) {
      shift_left_from_inplace(inserted.value(), topk_idx);
      topk_idx.at(inserted.value()) = idx;
    }
  }
  return make_pair(topk, topk_idx);
}

} // namespace stats
} // namespace nr
