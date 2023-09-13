#pragma once

#include "TinySet_small.hpp"

namespace SSTGraph {

template <typename... Ts> class TinySetV {

private:
  TinySetV_small<Ts...> ts;
  typename TinySetV_small<Ts...>::extra_data d;

  using element_type =
      typename std::conditional<TinySetV_small<Ts...>::binary, std::tuple<el_t>,
                                std::tuple<el_t, Ts...>>::type;

  template <int I>
  using NthType = typename std::tuple_element<I, element_type>::type;

public:
  void print_cutoffs() {
    std::cout << "24: " << d.thresh_24 << std::endl;
    std::cout << "16: " << d.thresh_16 << std::endl;
    std::cout << "8: " << d.thresh_8 << std::endl;
  }
  inline void prefetch_pmas() { ts.prefetch_pmas(); }
  explicit TinySetV(uint32_t max_el) : ts() {
    d = ts.get_thresholds(max_el);
    // printf("the threshholds are %u, %u, %u\n", d.thresh_24, d.thresh_16,
    //        d.thresh_8);
  }
  TinySetV(const TinySetV &source) : ts(source.ts, source.d), d(source.d) {}
  TinySetV(TinySetV &&source) : ts(source.ts, source.d), d(source.d) {}
  TinySetV &operator=(TinySetV&& other) {
    if (this != &other) {
      d = other.d;
      ts = other.ts;
    }
    return *this;
  }

  TinySetV &operator=(const TinySetV& other) {
    if (this != &other) {
      d = other.d;
      new(&ts) TinySetV_small<Ts...>(other.ts, d);
    }
    return *this;
  }

  ~TinySetV() { ts.destroy(d); }
  bool has(el_t e) const { return ts.has(e, d); }
  auto value(el_t e) const { return ts.value(e, d); }
  void insert(element_type e) { ts.insert(e, d); }
  void remove(el_t e) { ts.remove(e, d); }
  void insert_batch(element_type *els, uint64_t n) {
    ts.insert_batch(els, n, d);
  }

  uint64_t get_size() const { return ts.get_size(d) + sizeof(d); }
  uint64_t get_n() const { return ts.get_n(); }
  template <size_t... Is> void print() const { ts.template print<Is...>(d); }
  void print_pmas() const { ts.print_pmas(d); }

  template <bool no_early_exit, size_t... Is, class F>
  void map(F f, bool parallel = false) const {
    ts.template map<no_early_exit, Is...>(f, d, parallel);
  }
  template <bool no_early_exit, size_t... Is, class F>
  void parallel_map(F f) const {
    ts.template map<no_early_exit, Is...>(f, d, true);
  }
  [[nodiscard]] uint8_t get_b() const { return ts.get_b(); }
};
} // namespace SSTGraph
