
#pragma once

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

template <template <typename Sub> typename Cont, typename Sub>
static std::optional<size_t> insert_inplace(Sub to_insert, Cont<Sub> &c) {
  // maintains sorted ordering and size of c. Removing smallest value
  // if to_insert is the smallest, it is not inserted.
  // returns index inserted at. -1 if not inserted.

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
  if (k > c.size()) {
    throw std::logic_error("stats::topk, k is greater than size of input.");
  }
  Cont<Sub> topk(k, std::numeric_limits<Sub>::min());
  Cont<size_t> topk_idx(k, 0);

  for (size_t idx = 0; idx < c.size(); ++idx) {
    std::optional<size_t> inserted = insert_inplace(c.at(idx), topk);
    if (inserted) {
      shift_left_from_inplace(inserted.value(), topk_idx);
      topk_idx.at(inserted.value()) = idx;
    }
  }
  return make_pair(topk, topk_idx);
}

} // namespace stats
} // namespace nr
