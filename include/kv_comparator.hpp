
#pragma once

/*
* comparator functions for the KV type used by nr_multiprobe.
* These are used by topk to find the k KV that have alarge inner products with
* the query.   
*/

namespace nr {

template<typename KV>
class KVLess {
private:
  using Vect = typename KV::first_type;
  Vect q;

public:
  KVLess(Vect q):q(q) {}

  bool operator()(KV x, KV y) {
    return q.dot(x.first) < q.dot(y.first);
  }
};

template<typename KV>
class KVGreater {
private:
  using Vect = typename KV::first_type;
  Vect q;

public:
  KVGreater(Vect q):q(q) {}

  bool operator()(KV x, KV y) {
    return q.dot(x.first) > q.dot(y.first);
  }
};
} // namespace nr
