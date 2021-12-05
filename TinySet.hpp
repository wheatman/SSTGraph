#pragma once
#include "TinySet_small.hpp"

template <typename value_type = bool> class TinySetV {

private:
  TinySetV_small<value_type> ts;
  typename TinySetV_small<value_type>::extra_data d;

public:
  bool check_all_pmas() { return ts.check_all_pmas(d); }
  inline void prefetch_pmas() { ts.prefetch_pmas(); }
  explicit TinySetV(uint32_t max_el) : ts() {
    d = ts.get_thresholds(max_el);
    // printf("the threshholds are %u, %u, %u\n", d.thresh_24, d.thresh_16,
    //        d.thresh_8);
  }
  TinySetV(const TinySetV &source) : ts(source.ts, source.d), d(source.d) {}

  ~TinySetV() { ts.destroy(d); }
  bool has(el_t e) { return ts.has(e, d); }
  value_type value(el_t e) { return ts.value(e, d); }
  void insert(el_t e) { ts.insert(e, d); }
  void insert(el_t e, value_type v) { ts.insert({e, v}, d); }
  void remove(el_t e) { ts.remove(e, d); }
  void insert_batch(el_t *els, uint64_t n) { ts.insert_batch(els, n, d); }
  void insert_batch(el_t *els, value_type vs, uint64_t n) {
    ts.insert_batch(els, vs, n, d);
  }

  uint64_t get_size() { return ts.get_size(d) + sizeof(d); }
  uint64_t get_n() { return ts.get_n(); }
  uint64_t sum_keys() { return ts.sum_keys(d); }
  value_type sum_values() { return ts.sum_values(d); }

  void print() { ts.print(d); }
  void print_pmas() { ts.print_pmas(d); }
  class iterator {
  public:
    typename TinySetV_small<value_type>::iterator it;

    // for marking the end
    explicit iterator(uint32_t pma_index, uint8_t b)
        : it(typename TinySetV_small<value_type>::iterator(pma_index, b)) {}

    // only checks for not the end
    bool operator!=(const iterator &other) const { return it != other.it; }
    iterator &operator++() {
      ++it;
      return *this;
    }
    std::pair<el_t, value_type> operator*() const { return *it; }

    iterator(TinySetV<value_type> &t, uint32_t pma_index, uint8_t b)
        : it(typename TinySetV_small<value_type>::iterator(&t.ts, pma_index,
                                                           b)) {}
  };
  iterator begin() { return iterator(*this, ts.get_pma_count(d), ts.get_b()); }
  iterator end() { return iterator(ts.get_pma_count(d), ts.get_b()); }
};
