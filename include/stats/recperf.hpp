

namespace nr {
namespace stats {

template <template <typename Vect> typename Cont, typename Vect>
bool hit(size_t k, Vect good, const Cont<Vect> &others) {
  // rank good and others. if good is in topk, return 1/k.
}

} // namespace stats
} // namespace nr
