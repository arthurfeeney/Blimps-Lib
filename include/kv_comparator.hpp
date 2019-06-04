
#pragma once

/*

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
