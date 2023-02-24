#pragma once
#include "PMA.hpp"
#include "SizedInt.hpp"
#include "helpers.h"

#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <tuple>
#include <vector>
//#include <atomic>

namespace TimeSet_small_helpers {
struct extra_data {
  uint32_t thresh_24 = 0;
  uint32_t thresh_16 = 0;
  uint32_t thresh_8 = 0;
  uint32_t max_el = 0;
};

[[nodiscard]] uint32_t calc_pma_count(uint64_t max_el, uint32_t b);

} // namespace TimeSet_small_helpers

template <typename... Ts> class __attribute__((__packed__)) TinySetV_small {
  // class TinySetV_small {

  static constexpr bool get_if_binary() {
    if constexpr (sizeof...(Ts) == 0) {
      return true;
    } else {
      using FirstType = typename std::tuple_element<0, std::tuple<Ts...>>::type;
      return (sizeof...(Ts) == 1 && std::is_same_v<bool, FirstType>);
    }
  }

public:
  static constexpr bool binary = get_if_binary();

private:
  using element_type = typename std::conditional<binary, std::tuple<el_t>,
                                                 std::tuple<el_t, Ts...>>::type;

  using value_type =
      typename std::conditional<binary, std::tuple<>, std::tuple<Ts...>>::type;

  template <int I>
  using NthType = typename std::tuple_element<I, value_type>::type;

  static constexpr bool use_bit_array = false;

  static constexpr uint32_t max_pma_index_size = 4;

  // static constexpr bool use_low_index_pma = true;

public:
  using pma_types = union __attribute__((__packed__)) pma_types_ {
    PMA<sized_uint<1>, Ts...> pma1;
    PMA<sized_uint<2>, Ts...> pma2;
    PMA<sized_uint<3>, Ts...> pma3;
    PMA<sized_uint<4>, Ts...> pma4;
  };

  using pma_data = union __attribute__((__packed__)) pma_data_ {
    pma_types *p = nullptr;
#if NO_INLINE_TINYSET == 1
    pma_types *d;
#else
    pma_types d[1];
    struct __attribute__((__packed__)) {
      uint8_t _spacers[sizeof(pma_types) - 1];
      uint8_t _space : 5;
      uint8_t B : 3;
    } data;
#endif
    pma_data_() {}
    ~pma_data_() {}
    [[nodiscard]] uint8_t get_b() const { return data.B; }
    void set_b(uint8_t b) { data.B = b; }
  };

  using extra_data = TimeSet_small_helpers::extra_data;

private:
  pma_data pmas;

  // class empty_type {};
  // [[no_unique_address]]
  // typename std::conditional<use_low_index_pma, PMA<1, value_type>,
  //                           empty_type>::type low_index_pma;

  uint32_t el_count = 0;

  [[nodiscard]] uint64_t pma_region_size() const;
  [[nodiscard]] uint32_t which_pma(el_t e) const;
  [[nodiscard]] uint32_t small_element(el_t e) const;

  template <uint32_t old_b, uint32_t new_b>
  void change_index(uint32_t old_pma_count, const extra_data &d);

  template <typename p1, typename p2, bool intersection, bool no_early_exit,
            class F>
  void zip_body(const extra_data &this_d, const TinySetV_small &other,
                const extra_data &other_d, F f,
                el_t early_end_A = std::numeric_limits<el_t>::max(),
                el_t early_end_B = std::numeric_limits<el_t>::max()) const;

  template <bool intersection, bool no_early_exit, class F>
  void Zip(const extra_data &this_d, const TinySetV_small &other,
           const extra_data &other_d, F f,
           el_t early_end_A = std::numeric_limits<el_t>::max(),
           el_t early_end_B = std::numeric_limits<el_t>::max()) const;
  template <typename p1, typename p2>
  [[nodiscard]] uint64_t
  set_intersection_count_body(const extra_data &this_d,
                              const TinySetV_small &other,
                              const extra_data &other_d, uint32_t early_end_A,
                              uint32_t early_end_B) const;

  template <bool no_early_exit, typename p, size_t... Is, class F>
  void map_set_type(F f, const extra_data &d, bool parallel) const;

  template <int bytes> void print_pmas_internal(const extra_data &d) const;

public:
  [[nodiscard]] uint8_t get_b() const { return pmas.get_b(); }
  [[nodiscard]] uint32_t
  find_best_b_for_given_element_count(uint64_t element_count,
                                      const std::vector<uint32_t> &b_options,
                                      uint64_t max_val) const;

  [[nodiscard]] extra_data get_thresholds(uint64_t element_count) const;

  [[nodiscard]] uint32_t get_pma_count(const extra_data &d) const {
    return TimeSet_small_helpers::calc_pma_count(d.max_el, pmas.get_b());
  }

  [[nodiscard]] uint32_t *bit_array_start(const extra_data &d) const {
    if constexpr (use_bit_array) {
      return (uint32_t *)(pmas.p + get_pma_count(d));
    }
  }

  inline void prefetch_pma_data() const {
    // all are the same so just use one
    pmas.d->pma4.prefetch_data();
  }
  inline void prefetch_pmas() const { __builtin_prefetch(pmas.p, 1, 0); }
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
  void insert(element_type e, const extra_data &d);
  void remove(el_t e, const extra_data &d);
  void insert_batch(element_type *els, uint64_t n, const extra_data &d);

  [[nodiscard]] uint64_t get_size(const extra_data &d) const;
  [[nodiscard]] uint64_t get_n() const;

  // returns number of buckets, max_count in any bucket, number of buckets that
  // should be stored dense, number of buckets that are empty
  [[nodiscard]] std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, bool,
                           int64_t, int64_t>
  statistics(const extra_data &d, uint64_t self_index = 0) const;

  uint64_t Set_Intersection_Count(const extra_data &this_d,
                                  const TinySetV_small &other,
                                  const extra_data &other_d,
                                  uint32_t early_end_A,
                                  uint32_t early_end_B) const;
  template <size_t... Is> void print(const extra_data &d) const;
  void print_pmas(const extra_data &d) const;

  template <bool no_early_exit, size_t... Is, class F>
  void map(F f, const extra_data &d, bool parallel = false) const;
};
