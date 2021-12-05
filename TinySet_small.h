#pragma once
#include "PMA.hpp"
#include "VertexSubset.hpp"
#include "helpers.h"
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <tuple>
#include <vector>
//#include <atomic>

template <typename value_type = bool>
class __attribute__((__packed__)) TinySetV_small {
  // class TinySetV_small {
  static constexpr bool binary = std::is_same<value_type, bool>::value;

  using item_type =
      typename std::conditional<binary, std::tuple<el_t>,
                                std::tuple<el_t, value_type>>::type;

  static constexpr uint32_t max_pma_index_size = 4;

  template <typename> friend class TinySetV;

  using pma_types = union __attribute__((__packed__)) pma_types_ {
    PMA<1, value_type> pma1;
    PMA<2, value_type> pma2;
    PMA<3, value_type> pma3;
    PMA<4, value_type> pma4;
  };

  using pma_data = union __attribute__((__packed__)) pma_data_ {
    pma_types *p = nullptr;
#if NO_INLINE_TINYSET == 1
    pma_types *d;
#else
    pma_types d[1];
#endif
    struct __attribute__((__packed__)) {
      uint8_t _spacers[sizeof(pma_types) - 1];
      uint8_t _space : 5;
      uint8_t B : 3;
    } data;

    pma_data_() {}
    ~pma_data_() {}
    [[nodiscard]] uint8_t get_b() const { return data.B; }
    void set_b(uint8_t b) { data.B = b; }
  };

public:
  using extra_data = struct extra_data_ {
    uint32_t thresh_24 = 0;
    uint32_t thresh_16 = 0;
    uint32_t thresh_8 = 0;
    uint32_t max_el = 0;
  };

private:
  pma_data pmas;
  uint32_t el_count = 0;

  static uint8_t find_b_for_el_count(uint32_t element_count,
                                     const extra_data &d);

  [[nodiscard]] uint64_t pma_region_size() const;
  [[nodiscard]] uint32_t which_pma(el_t e) const;
  [[nodiscard]] uint32_t small_element(el_t e) const;

  [[nodiscard]] uint32_t calc_pma_count(uint64_t max_el, uint32_t b) const;

  template <uint32_t old_b, uint32_t new_b>
  void change_index(uint32_t old_pma_count, const extra_data &d);

  template <typename p1, typename p2>
  [[nodiscard]] uint64_t
  set_intersection_count_body(const extra_data &this_d,
                              const TinySetV_small &other,
                              const extra_data &other_d, uint32_t early_end_A,
                              uint32_t early_end_B) const;

  template <class F, typename p>
  void map_set_type(F &f, const extra_data &d, bool parallel) const;

  template <int bytes> void print_pmas_internal(const extra_data &d) const;

  [[nodiscard]] uint32_t
  find_best_b_for_given_element_count(uint64_t element_count,
                                      const std::vector<uint32_t> &b_options,
                                      uint64_t max_val) const;

  [[nodiscard]] uint32_t get_pma_count(const extra_data &d) const {
    return calc_pma_count(d.max_el, pmas.get_b());
  }
  [[nodiscard]] bool check_all_pmas(const extra_data &d) const;

  inline void prefetch_pma_data() const {
    // all are the same so just use one
    pmas.d->pma4.prefetch_data();
  }
  inline void prefetch_pmas() const { __builtin_prefetch(pmas.p, 1, 0); }

  void print_pmas(const extra_data &d) const;

  template <int B> class iterator_internal {

    uint64_t which_pma;
    pma_types *pmas;
    typename PMA<B, value_type>::iterator it;
    uint64_t pma_count = 0;
    // for marking the end
  public:
    explicit iterator_internal(uint64_t pma_index)
        : which_pma(pma_index), pmas(nullptr), it(0) {}

    // only checks for not the end
    bool operator!=(const iterator_internal &other) const {
      return which_pma != other.which_pma;
    }
    iterator_internal &operator++() {
      ++it;
      while (
          which_pma < pma_count &&
          it >=
              reinterpret_cast<PMA<B, value_type> *>(&pmas[which_pma])->end()) {
        which_pma += 1;
        if (which_pma < pma_count) {
          it =
              reinterpret_cast<PMA<B, value_type> *>(&pmas[which_pma])->begin();
        }
      }
      return *this;
    }
    std::pair<el_t, value_type> operator*() const {
      if constexpr (binary) {
        return {((which_pma) << (B * 8U)) | (*it).first, true};
      } else {
        return {((which_pma) << (B * 8U)) | (*it).first, (*it).second};
      }
    }

    iterator_internal(TinySetV_small *t, uint32_t pma_count_)
        : which_pma(0), pmas((pma_count_ == 1) ? &t->pmas.d[0] : t->pmas.p),
          it(reinterpret_cast<PMA<B, value_type> *>(&pmas[which_pma])
                 ->begin()) {
      pma_count = pma_count_;
      while (
          which_pma < pma_count &&
          it >=
              reinterpret_cast<PMA<B, value_type> *>(&pmas[which_pma])->end()) {
        which_pma += 1;
        if (which_pma < pma_count) {
          it =
              reinterpret_cast<PMA<B, value_type> *>(&pmas[which_pma])->begin();
        }
      }
    }
  };

public:
  [[nodiscard]] uint8_t get_b() const { return pmas.get_b(); }

  [[nodiscard]] extra_data get_thresholds(uint64_t element_count) const;

  inline void prefetch_data(const extra_data &d) const {
    if (get_pma_count(d) == 1) {
      prefetch_pma_data();
    } else {
      prefetch_pmas();
    }
  }
  TinySetV_small();
  // NOTE make sure you call destroy
  ~TinySetV_small() = default;
  TinySetV_small(const TinySetV_small &source, const extra_data &d);
  void destroy(const extra_data &d);
  [[nodiscard]] bool has(el_t e, const extra_data &d) const;
  value_type value(el_t e, const extra_data &d) const;
  // TODO(wheatman) reimplement the inserts with edge_type like in SparseMatrix
  void insert(item_type e, const extra_data &d);
  void insert(el_t e, const extra_data &d) {
    static_assert(binary);
    return insert(std::make_tuple(e), d);
  }
  void remove(el_t e, const extra_data &d);
  void insert_batch(item_type *els, uint64_t n, const extra_data &d);

  [[nodiscard]] uint64_t get_size(const extra_data &d) const;
  [[nodiscard]] uint64_t get_n() const;
  [[nodiscard]] uint64_t sum_keys(const extra_data &d) const;
  [[nodiscard]] value_type sum_values(const extra_data &d) const;
  // returns number of buckets, max_count in any bucket, number of buckets that
  // should be stored dense, number of buckets that are empty
  [[nodiscard]] std::tuple<uint64_t, uint64_t, uint64_t>
  statistics(const extra_data &d) const;

  uint64_t Set_Intersection_Count(const extra_data &this_d,
                                  const TinySetV_small &other,
                                  const extra_data &other_d,
                                  uint32_t early_end_A,
                                  uint32_t early_end_B) const;
  void print(const extra_data &d) const;

  template <class F>
  void map(F &f, const extra_data &d, bool parallel = false) const;

  class iterator {

    iterator_internal<1> it_1;
    iterator_internal<2> it_2;
    iterator_internal<3> it_3;
    iterator_internal<4> it_4;
    uint8_t B;

  public:
    // for marking the end
    explicit iterator(uint64_t pma_index, uint8_t b)
        : it_1(pma_index), it_2(pma_index), it_3(pma_index), it_4(pma_index),
          B(b) {}

    // only checks for not the end
    bool operator!=(const iterator &other) const {
      switch (B) {
      case 1:
        return it_1 != other.it_1;
      case 2:
        return it_2 != other.it_2;
      case 3:
        return it_3 != other.it_3;
      case 4:
        return it_4 != other.it_4;
      }
      // should never happen
      return true;
    }
    iterator &operator++() {
      switch (B) {
      case 1:
        ++it_1;
        break;
      case 2:
        ++it_2;
        break;
      case 3:
        ++it_3;
        break;
      case 4:
        ++it_4;
        break;
      }
      return *this;
    }
    std::pair<el_t, value_type> operator*() const {
      switch (B) {
      case 1:
        return *it_1;
      case 2:
        return *it_2;
      case 3:
        return *it_3;
      case 4:
        return *it_4;
      }
      // should never happen
      return {};
    }

    iterator(TinySetV_small *t, uint32_t pma_count_, uint8_t b)
        : it_1(t, pma_count_), it_2(t, pma_count_), it_3(t, pma_count_),
          it_4(t, pma_count_), B(b) {}
  };
  iterator begin(extra_data d) const {
    return iterator((TinySetV_small *)this, get_pma_count(d), pmas.get_b());
  }
  iterator end(extra_data d) const {
    return iterator(get_pma_count(d), pmas.get_b());
  }
};
