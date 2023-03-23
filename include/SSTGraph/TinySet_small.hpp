#pragma once

#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <limits>
#include <tuple>
#include <vector>

#include "SSTGraph/PMA.hpp"
#include "SSTGraph/internal/BitArray.hpp"
#include "SSTGraph/internal/SizedInt.hpp"
#include "SSTGraph/internal/helpers.hpp"

namespace SSTGraph {

namespace TimeSet_small_helpers {
struct extra_data {
  uint32_t thresh_24 = 0;
  uint32_t thresh_16 = 0;
  uint32_t thresh_8 = 0;
  uint32_t max_el = 0;
};

[[nodiscard]] inline uint32_t calc_pma_count(uint64_t max_el, uint32_t b);

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
  void map_set_type(F &&f, const extra_data &d, bool parallel) const;

  template <int bytes> void print_pmas_internal(const extra_data &d) const;

public:
  [[nodiscard]] uint8_t get_b() const { return pmas.get_b(); }
  [[nodiscard]] uint32_t static find_best_b_for_given_element_count(
      uint64_t element_count, const std::vector<uint32_t> &b_options,
      uint64_t max_val);

  [[nodiscard]] static extra_data get_thresholds(uint64_t element_count);

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
  void map(F &&f, const extra_data &d, bool parallel = false) const;
};

namespace TimeSet_small_helpers {

inline uint8_t find_b_for_el_count(uint32_t element_count,
                                   const extra_data &d) {
  if (element_count < d.thresh_24) {
    return 4;
  } else if (element_count < d.thresh_16) {
    return 3;
  } else if (element_count < d.thresh_8) {
    return 2;
  } else {
    return 1;
  }
}
inline uint32_t calc_pma_count(uint64_t max_el, uint32_t b) {
  uint64_t n = max_el;
  uint64_t pma_count = std::max(1UL, n >> (b * 8));
  if ((pma_count << (b * 8)) < n) {
    pma_count += 1;
  }
  return pma_count;
}

} // namespace TimeSet_small_helpers

template <typename... Ts>
typename TinySetV_small<Ts...>::extra_data
TinySetV_small<Ts...>::get_thresholds(uint64_t element_count) {
  extra_data d = {};
  d.max_el = element_count;
  d.thresh_24 = 2;
  uint32_t current_b = 32;
  for (uint32_t el_count = 2; el_count < UINT32_MAX / 2; el_count *= 2) {
    uint32_t best_b =
        find_best_b_for_given_element_count(el_count, {4, 3, 2, 1}, d.max_el);
    if (best_b < current_b) {
      current_b = best_b;
      switch (current_b) {
      case 3:
        d.thresh_24 = el_count * 2;
        break;
      case 2:
        d.thresh_16 = el_count * 2;
        break;
      case 1:
        d.thresh_8 = el_count * 2;
        break;
      }
    }
  }
#if TINYSET_32 == 1
  d.thresh_24 = UINT32_MAX / 4;
#endif
  if (d.thresh_16 <= d.thresh_24) {
    d.thresh_16 = d.thresh_24 * 2;
  }
  if (d.thresh_8 <= d.thresh_16) {
    d.thresh_8 = d.thresh_16 * 2;
  }
  return d;
}

template <typename... Ts>
uint32_t TinySetV_small<Ts...>::find_best_b_for_given_element_count(
    uint64_t element_count, const std::vector<uint32_t> &b_options,
    uint64_t max_val) {
  assert(!b_options.empty());
  std::vector<uint64_t> size_guesses(b_options.size());
  for (uint64_t i = 0; i < b_options.size(); i++) {
    uint32_t b = b_options[i];
    uint64_t elements_space = (element_count * b) * 2; // mul 2 for extra spaces
    uint64_t num_pmas = TimeSet_small_helpers::calc_pma_count(max_val, b);
    // they are all the same size so just pick one
    uint64_t pma_space = num_pmas * sizeof(PMA<sized_uint<4>, Ts...>);
    if constexpr (use_bit_array) {
      pma_space += BitArray::bit_array_size(num_pmas);
    }
    if (num_pmas == 1) {
      pma_space = 0;
    }
    size_guesses[i] = elements_space + pma_space;
  }
  uint32_t best_b = b_options[0];
  uint64_t best_mem = size_guesses[0];
  for (uint64_t i = 1; i < b_options.size(); i++) {
    if (size_guesses[i] < best_mem) {
      best_mem = size_guesses[i];
      best_b = b_options[i];
    }
  }
  return best_b;
}

template <typename... Ts>
template <bool no_early_exit, typename p, size_t... Is, class F>
void TinySetV_small<Ts...>::map_set_type(F &&f, const extra_data &d,
                                         bool parallel) const {
  uint32_t iters = get_pma_count(d);
  const p *ps = reinterpret_cast<const p *>((iters == 1) ? &pmas.d[0] : pmas.p);
  uint64_t b_ = pmas.get_b();
  if (!parallel || iters == 1 || el_count <= 1000) {
    for (uint64_t i = 0; i < iters; i += 1) {
      if constexpr (use_bit_array) {
        if (iters > 1) {
          if (!bit_array_get(bit_array_start(d), i)) {
            continue;
          }
        }
      }
      uint64_t prefix = i << (b_ * 8);
      if constexpr (no_early_exit) {
        ps[i].template map<true, Is...>(f, prefix);
      } else {
        if (ps[i].template map<false, Is...>(f, prefix)) {
          break;
        }
      }
    }
  } else {
    ParallelTools::parallel_for(0, iters, [&](uint64_t i) {
      if constexpr (use_bit_array) {
        if (iters > 1) {
          if (!bit_array_get(bit_array_start(d), i)) {
            return;
          }
        }
      }
      uint64_t prefix = i << (b_ * 8);
      ps[i].template map<no_early_exit, Is...>(f, prefix);
    });
  }
}

template <typename... Ts>
template <bool no_early_exit, size_t... Is, class F>
void TinySetV_small<Ts...>::map(F &&f, const extra_data &d,
                                bool parallel) const {
  switch (pmas.get_b()) {
  case 1:
    return map_set_type<no_early_exit, PMA<sized_uint<1>, Ts...>, Is...>(
        f, d, parallel);
  case 2:
    return map_set_type<no_early_exit, PMA<sized_uint<2>, Ts...>, Is...>(
        f, d, parallel);
  case 3:
    return map_set_type<no_early_exit, PMA<sized_uint<3>, Ts...>, Is...>(
        f, d, parallel);
  case 4:
    return map_set_type<no_early_exit, PMA<sized_uint<4>, Ts...>, Is...>(
        f, d, parallel);
  }
}

enum which_done { FIRST, SECOND, BOTH, COMPLETE };

template <typename i1, typename i2>
static which_done set_intersection_count_body_intersection(
    uint64_t *count, i1 &it1, i2 &it2, i1 &it1_end, i2 &it2_end,
    uint64_t this_prefix, uint64_t other_prefix, uint64_t this_region_size,
    uint64_t other_region_size, uint32_t early_end_A, uint32_t early_end_B) {
  uint64_t local_count = 0;
  if ((it1 < it1_end) && (it2 < it2_end)) {
    uint64_t this_val = std::get<0>(*it1) + this_prefix;
    uint64_t other_val = std::get<0>(*it2) + other_prefix;
    if (this_val >= early_end_A || other_val >= early_end_B) {
      *count += local_count;
      return COMPLETE;
    }
    while (true) {
      local_count += (this_val == other_val);
      uint64_t original_this_val = this_val;
      if (this_val <= other_val) {
        ++it1;
        if (!(it1 < it1_end)) {
          break;
        }
        this_val = std::get<0>(*it1) + this_prefix;
        if (this_val >= early_end_A) {
          *count += local_count;
          return COMPLETE;
        }
      }
      if (other_val <= original_this_val) {
        ++it2;
        if (!(it2 < it2_end)) {
          break;
        }
        other_val = std::get<0>(*it2) + other_prefix;
        if (other_val >= early_end_B) {
          *count += local_count;
          return COMPLETE;
        }
      }
    }
  }
  *count += local_count;
  if (this_prefix + this_region_size == other_prefix + other_region_size) {
    return BOTH;
  }
  if (it1 < it1_end) {
    return SECOND;
  } else if (it2 < it2_end) {
    return FIRST;
  }
  return BOTH;
}

template <typename... Ts>
template <typename p1, typename p2>
uint64_t TinySetV_small<Ts...>::set_intersection_count_body(
    const extra_data &this_d, const TinySetV_small &other,
    const extra_data &other_d, uint32_t early_end_A,
    uint32_t early_end_B) const {
  static_assert(
      binary,
      "set_intersection_count only implemented for binary mode for now");
  uint64_t count = 0;
  uint64_t pma_count_this = get_pma_count(this_d);
  uint64_t pma_count_other = other.get_pma_count(other_d);
  uint64_t this_region_size = pma_region_size();
  uint64_t other_region_size = other.pma_region_size();
  uint64_t iter_this = 0;
  uint64_t iter_other = 0;
  const p1 *this_pmas = (pma_count_this == 1)
                            ? reinterpret_cast<const p1 *>(pmas.d)
                            : reinterpret_cast<const p1 *>(pmas.p);
  const p2 *other_pmas = (pma_count_other == 1)
                             ? reinterpret_cast<const p2 *>(other.pmas.d)
                             : reinterpret_cast<const p2 *>(other.pmas.p);

  bool this_fast = this_pmas[0].use_fast_iter();
  bool other_fast = other_pmas[0].use_fast_iter();

  auto pma_iterator_this = this_pmas[0].begin();
  auto pma_iterator_other = other_pmas[0].begin();
  auto pma_iterator_this_end = this_pmas[0].end();
  auto pma_iterator_other_end = other_pmas[0].end();
  auto pma_iterator_fast_this = this_pmas[0].template begin<true>();
  auto pma_iterator_fast_other = other_pmas[0].template begin<true>();
  auto pma_iterator_fast_this_end = this_pmas[0].template end<true>();
  auto pma_iterator_fast_other_end = other_pmas[0].template end<true>();

  uint64_t this_prefix = this_region_size * iter_this;
  uint64_t other_prefix = other_region_size * iter_other;

  which_done done;
  while (iter_this < pma_count_this && iter_other < pma_count_other) {
    if (this_fast) {
      if (other_fast) {
        done = set_intersection_count_body_intersection(
            &count, pma_iterator_fast_this, pma_iterator_fast_other,
            pma_iterator_fast_this_end, pma_iterator_fast_other_end,
            this_prefix, other_prefix, this_region_size, other_region_size,
            early_end_A, early_end_B);
      } else {
        done = set_intersection_count_body_intersection(
            &count, pma_iterator_fast_this, pma_iterator_other,
            pma_iterator_fast_this_end, pma_iterator_other_end, this_prefix,
            other_prefix, this_region_size, other_region_size, early_end_A,
            early_end_B);
      }
    } else {
      if (other_fast) {
        done = set_intersection_count_body_intersection(
            &count, pma_iterator_this, pma_iterator_fast_other,
            pma_iterator_this_end, pma_iterator_fast_other_end, this_prefix,
            other_prefix, this_region_size, other_region_size, early_end_A,
            early_end_B);
      } else {
        done = set_intersection_count_body_intersection(
            &count, pma_iterator_this, pma_iterator_other,
            pma_iterator_this_end, pma_iterator_other_end, this_prefix,
            other_prefix, this_region_size, other_region_size, early_end_A,
            early_end_B);
      }
    }
    switch (done) {
    case BOTH:
    case FIRST:
      iter_this += 1;
      this_prefix = this_region_size * iter_this;
      if (iter_this < pma_count_this) {
        pma_iterator_this = this_pmas[iter_this].begin();
        pma_iterator_this_end = this_pmas[iter_this].end();
        this_fast = this_pmas[iter_this].use_fast_iter();
        pma_iterator_fast_this = this_pmas[iter_this].template begin<true>();
        pma_iterator_fast_this_end = this_pmas[iter_this].template end<true>();
      } else {
        return count;
      }
      if (done == FIRST) {
        break;
      }
      [[fallthrough]];
    case SECOND:
      iter_other += 1;
      other_prefix = other_region_size * iter_other;
      if (iter_other < pma_count_other) {
        pma_iterator_other = other_pmas[iter_other].begin();
        pma_iterator_other_end = other_pmas[iter_other].end();
        other_fast = other_pmas[iter_other].use_fast_iter();
        pma_iterator_fast_other = other_pmas[iter_other].template begin<true>();
        pma_iterator_fast_other_end =
            other_pmas[iter_other].template end<true>();
      } else {
        return count;
      }
      break;
    case COMPLETE:
      return count;
    }
  }
  return count;
}

template <typename... Ts>
uint64_t TinySetV_small<Ts...>::Set_Intersection_Count(
    const extra_data &this_d, const TinySetV_small &other,
    const extra_data &other_d, uint32_t early_end_A,
    uint32_t early_end_B) const {
  static_assert(
      binary,
      "set_intersection_count only implemented for binary mode for now");
  switch (pmas.get_b()) {
  case 1:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<sized_uint<1>, Ts...>,
                                         PMA<sized_uint<1>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<sized_uint<1>, Ts...>,
                                         PMA<sized_uint<2>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<sized_uint<1>, Ts...>,
                                         PMA<sized_uint<3>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<sized_uint<1>, Ts...>,
                                         PMA<sized_uint<4>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    __builtin_unreachable();
    return 0;
  case 2:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<sized_uint<2>, Ts...>,
                                         PMA<sized_uint<1>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<sized_uint<2>, Ts...>,
                                         PMA<sized_uint<2>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<sized_uint<2>, Ts...>,
                                         PMA<sized_uint<3>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<sized_uint<2>, Ts...>,
                                         PMA<sized_uint<4>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    __builtin_unreachable();
    return 0;
  case 3:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<sized_uint<3>, Ts...>,
                                         PMA<sized_uint<1>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<sized_uint<3>, Ts...>,
                                         PMA<sized_uint<2>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<sized_uint<3>, Ts...>,
                                         PMA<sized_uint<3>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<sized_uint<3>, Ts...>,
                                         PMA<sized_uint<4>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    __builtin_unreachable();
    return 0;
  case 4:
    switch (other.pmas.get_b()) {
    case 1:
      return set_intersection_count_body<PMA<sized_uint<4>, Ts...>,
                                         PMA<sized_uint<1>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 2:
      return set_intersection_count_body<PMA<sized_uint<4>, Ts...>,
                                         PMA<sized_uint<2>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 3:
      return set_intersection_count_body<PMA<sized_uint<4>, Ts...>,
                                         PMA<sized_uint<3>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    case 4:
      return set_intersection_count_body<PMA<sized_uint<4>, Ts...>,
                                         PMA<sized_uint<4>, Ts...>>(
          this_d, other, other_d, early_end_A, early_end_B);
    }
    break;
  }
  __builtin_unreachable();
  return 0;
}

// TODO(wheatman) fix so that by default prints all fields
template <typename... Ts>
template <size_t... Is>
void TinySetV_small<Ts...>::print(const extra_data &d) const {
  printf("tinyset: element_count = %u, B = %u\n", el_count, +pmas.get_b());
  if (pmas.get_b() > 4) {
    std::cout << "B is too big" << std::endl;
    return;
  }
  if constexpr (binary || sizeof...(Is) == 0) {
    map<true>([&](el_t dest) { printf("%u, ", dest); }, d, false);
  } else {
    map<true, Is...>(
        [](el_t key, NthType<Is>... args) {
          std::cout << "(" << key;
          ((std::cout << ", " << args), ...);
          std::cout << "), ";
        },
        d, false);
  }
  printf("\n");
}

template <typename... Ts>
template <int bytes>
void TinySetV_small<Ts...>::print_pmas_internal(const extra_data &d) const {
  uint32_t pma_count = get_pma_count(d);
  if (pma_count == 1) {
    reinterpret_cast<const PMA<sized_uint<bytes>, Ts...> *>(&pmas.d[0])
        ->print_pma(0);
  } else {
    for (uint32_t i = 0; i < get_pma_count(d); i++) {
      if constexpr (bytes < 4) {
        reinterpret_cast<const PMA<sized_uint<bytes>, Ts...> *>(&pmas.p[i])
            ->print_pma(i << (bytes * 8U));
      } else {
        reinterpret_cast<const PMA<sized_uint<bytes>, Ts...> *>(&pmas.p[i])
            ->print_pma(i);
      }
    }
  }
}

template <typename... Ts>
void TinySetV_small<Ts...>::print_pmas(const extra_data &d) const {
  std::cout << "B = " << +pmas.get_b() << std::endl;
  switch (pmas.get_b()) {
  case 1:
    return print_pmas_internal<1>(d);
  case 2:
    return print_pmas_internal<2>(d);
  case 3:
    return print_pmas_internal<3>(d);
  case 4:
    return print_pmas_internal<4>(d);
  }
}

template <typename... Ts> uint64_t TinySetV_small<Ts...>::get_n() const {
  return el_count;
}

template <typename... Ts>
uint64_t TinySetV_small<Ts...>::get_size(const extra_data &d) const {
  uint64_t size = 0;
  uint64_t pma_count = get_pma_count(d);
  switch (pmas.get_b()) {
  case 1: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma1.get_size();
#else
      size += pmas.d[0].pma1.get_size() - sizeof(pmas.d[0].pma1);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma1.get_size();
      }
    }
    break;
  }
  case 2: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma2.get_size();
#else
      size += pmas.d[0].pma2.get_size() - sizeof(pmas.d[0].pma2);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma2.get_size();
      }
    }
    break;
  }
  case 3: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma3.get_size();
#else
      size += pmas.d[0].pma3.get_size() - sizeof(pmas.d[0].pma3);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma3.get_size();
      }
    }
    break;
  }
  case 4: {
    if (pma_count == 1) {
      // printf("pma_size = %lu\n", pmas.d->pma.get_size());
#if NO_INLINE_TINYSET == 1
      size += pmas.d->pma4.get_size();
#else
      size += pmas.d[0].pma4.get_size() - sizeof(pmas.d[0].pma4);
#endif
    } else {
      for (uint32_t i = 0; i < pma_count; i++) {
        size += pmas.p[i].pma4.get_size();
      }
    }
    break;
  }
  }
  if constexpr (use_bit_array) {
    if (pma_count > 1) {
      size += BitArray::bit_array_size(pma_count);
    }
  }
  return size + sizeof(TinySetV_small);
}

// returns number of buckets, max_count in any bucket, number of buckets that
// should be stored dense, number of buckets that are empty, if multiple buckets
// are being used and the number of bytes saved if we stored index 0-255 as 1
// byte objects
template <typename... Ts>
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, bool, int64_t, int64_t>
TinySetV_small<Ts...>::statistics(const extra_data &d,
                                  uint64_t self_index) const {
  uint32_t pma_count = get_pma_count(d);
  uint64_t max_count = 0;
  uint32_t worth_dense = 0;
  uint32_t empty = 0;
  bool multiple_buckets = pma_count > 1;
  int64_t possible_bytes_saved_first = 0;
  int64_t possible_bytes_saved_diagonal = 0;
  if (multiple_buckets) {
    possible_bytes_saved_diagonal -= sizeof(PMA<sized_uint<1>, Ts...>);
    possible_bytes_saved_first -= sizeof(PMA<sized_uint<1>, Ts...>);
    uint64_t bytes_saved_each = pmas.get_b() - 1;
    uint32_t start = 0;
    uint32_t end = d.max_el;
    if (self_index > 128) {
      start = self_index - 128;
    }
    if (self_index < end - 128) {
      end = self_index + 128;
    }
    for (uint32_t i = start; i < end; i++) {
      possible_bytes_saved_diagonal += bytes_saved_each * has(i, d);
    }
    for (uint32_t i = 0; i < 256; i++) {
      possible_bytes_saved_first += bytes_saved_each * has(i, d);
    }
  }
  switch (pmas.get_b()) {
  case 1: {
    const PMA<sized_uint<1>, Ts...> *ps =
        reinterpret_cast<const PMA<sized_uint<1>, Ts...> *>(
            (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      worth_dense += ps[i].stored_uncompressed();
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  case 2: {
    const PMA<sized_uint<2>, Ts...> *ps =
        reinterpret_cast<const PMA<sized_uint<2>, Ts...> *>(
            (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      worth_dense += ps[i].stored_uncompressed();
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  case 3: {
    const PMA<sized_uint<3>, Ts...> *ps =
        reinterpret_cast<const PMA<sized_uint<3>, Ts...> *>(
            (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      worth_dense += ps[i].stored_uncompressed();
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  case 4: {
    const PMA<sized_uint<4>, Ts...> *ps =
        reinterpret_cast<const PMA<sized_uint<4>, Ts...> *>(
            (pma_count == 1) ? &pmas.d[0] : pmas.p);
    for (uint32_t i = 0; i < pma_count; i++) {
      worth_dense += ps[i].stored_uncompressed();
      max_count = std::max(max_count, ps[i].get_n());
      if (ps[i].get_n() == 0) {
        empty += 1;
      }
    }
    break;
  }
  }
  if (empty > pma_count) {
    printf("empty = %u, pma_count = %u\n", empty, pma_count);
    print_pmas(d);
    exit(-1);
  }
  return {pma_count,
          max_count,
          worth_dense,
          empty,
          multiple_buckets,
          possible_bytes_saved_first,
          possible_bytes_saved_diagonal};
}

template <typename... Ts>
uint32_t TinySetV_small<Ts...>::which_pma(el_t e) const {
  if constexpr (max_pma_index_size == sizeof(uint64_t)) {
    if (pmas.get_b() == sizeof(e)) {
      return 0;
    }
  }
  return static_cast<uint64_t>(e) >> (pmas.get_b() * 8);
}

template <typename... Ts>
uint32_t TinySetV_small<Ts...>::small_element(el_t e) const {
  if constexpr (max_pma_index_size == sizeof(uint64_t)) {
    if (pmas.get_b() == sizeof(e)) {
      return e;
    }
  }
  return e & ((1UL << (pmas.get_b() * 8)) - 1);
}

template <typename... Ts>
uint64_t TinySetV_small<Ts...>::pma_region_size() const {
  if constexpr (max_pma_index_size == sizeof(uint64_t)) {
    if (pmas.get_b() == sizeof(uint64_t)) {
      return 0; // this is incorrect but we can't express the number with the
                // given bits, since it is 1 bigger than the max size
    }
  }
  return 1UL << (pmas.get_b() * 8);
}

// TODO(wheatman) fails with new_b = 64
// TODO(wheatman) this can be made a lot faster since we know they are already
// in order
template <typename... Ts>
template <uint32_t old_b, uint32_t new_b>
void TinySetV_small<Ts...>::change_index(uint32_t old_pma_count,
                                         const extra_data &d) {
  // printf("changing index from %u, to %u\n", old_b, new_b);
  const uint32_t new_pma_count =
      TimeSet_small_helpers::calc_pma_count(d.max_el, new_b);
  // printf("changing index from %u, to %u, old_count = %u, new_count = %u\n",
  //        old_b, new_b, old_pma_count, new_pma_count);
  pma_types *new_pmas = nullptr;
  PMA<sized_uint<new_b>, Ts...> new_pma_in_place;
  [[maybe_unused]] uint32_t *new_bit_array_start = nullptr;
  if (new_pma_count > 1) {
    if constexpr (use_bit_array) {
      new_pmas = (pma_types *)malloc(new_pma_count * sizeof(pma_types) +
                                     BitArray::bit_array_size(new_pma_count));
      new_bit_array_start = (uint32_t *)(new_pmas + new_pma_count);
      memset(new_bit_array_start, 0, BitArray::bit_array_size(new_pma_count));
    } else {
      new_pmas = (pma_types *)malloc(new_pma_count * sizeof(pma_types));
    }
    for (uint32_t i = 0; i < new_pma_count; i++) {
      new (&new_pmas[i]) PMA<sized_uint<new_b>, Ts...>();
    }
  }
  // TODO(wheatman) we know all of these inserts are to the end since the old
  // pma is sorted
  if (old_pma_count == 1) {
    auto it_end =
        reinterpret_cast<PMA<sized_uint<old_b>, Ts...> *>(&pmas.d[0])->end();
    auto it =
        reinterpret_cast<PMA<sized_uint<old_b>, Ts...> *>(&pmas.d[0])->begin();
    for (; it != it_end; ++it) {
      el_t val = std::get<0>(*it);
      if (new_pma_count == 1) {
        if constexpr (binary) {
          new_pma_in_place.insert(val);
        } else {
          new_pma_in_place.insert(std::tuple_cat(
              std::make_tuple(sized_uint<new_b>(val)), leftshift_tuple(*it)));
        }

        continue;
      }
      uint32_t pma_index = 0;
      uint32_t small_el = val;
      if constexpr (new_b < 4) {
        pma_index = val >> (new_b * 8);
        small_el &= (((1U << (new_b * 8)) - 1));
      }
      if constexpr (use_bit_array) {
        BitArray::bit_array_set(new_bit_array_start, pma_index);
      }
      if constexpr (binary) {
        reinterpret_cast<PMA<sized_uint<new_b>, Ts...> *>(&new_pmas[pma_index])
            ->insert(small_el);
      } else {
        reinterpret_cast<PMA<sized_uint<new_b>, Ts...> *>(&new_pmas[pma_index])
            ->insert(
                std::tuple_cat(std::make_tuple(sized_uint<new_b>(small_el)),
                               leftshift_tuple(*it)));
      }
    }
  } else {
    for (uint32_t i = 0; i < old_pma_count; i++) {
      auto it_end =
          reinterpret_cast<PMA<sized_uint<old_b>, Ts...> *>(&pmas.p[i])->end();
      auto it = reinterpret_cast<PMA<sized_uint<old_b>, Ts...> *>(&pmas.p[i])
                    ->begin();
      for (; it != it_end; ++it) {
        el_t val = std::get<0>(*it);
        if constexpr (old_b < 4) {
          val |= (i << (old_b * 8));
        }
        if (new_pma_count == 1) {
          if constexpr (binary) {
            new_pma_in_place.insert(val);
          } else {
            new_pma_in_place.insert(std::tuple_cat(
                std::make_tuple(sized_uint<new_b>(val)), leftshift_tuple(*it)));
          }
          continue;
        }
        uint32_t pma_index = 0;
        uint32_t small_el = val;
        if constexpr (new_b < 4) {
          pma_index = val >> (new_b * 8);
          small_el &= (((1U << (new_b * 8)) - 1));
        }
        if constexpr (use_bit_array) {
          BitArray::bit_array_set(new_bit_array_start, pma_index);
        }
        if constexpr (binary) {
          reinterpret_cast<PMA<sized_uint<new_b>, Ts...> *>(
              &new_pmas[pma_index])
              ->insert(small_el);
        } else {
          reinterpret_cast<PMA<sized_uint<new_b>, Ts...> *>(
              &new_pmas[pma_index])
              ->insert(
                  std::tuple_cat(std::make_tuple(sized_uint<new_b>(small_el)),
                                 leftshift_tuple(*it)));
        }
      }
    }
  }
  if (old_pma_count > 1) {
    for (uint32_t i = 0; i < old_pma_count; i++) {
      reinterpret_cast<PMA<sized_uint<old_b>, Ts...> *>(&pmas.p[i])
          ->~PMA<sized_uint<old_b>, Ts...>();
    }
    free(pmas.p);
  } else {
#if NO_INLINE_TINYSET == 1
    reinterpret_cast<PMA<old_b, value_type> *>(pmas.d)
        ->~PMA<old_b, value_type>();
#else
    reinterpret_cast<PMA<sized_uint<old_b>, Ts...> *>(&pmas.d[0])
        ->~PMA<sized_uint<old_b>, Ts...>();
#endif
  }
  if (new_pma_count == 1) {
    pmas.set_b(new_b);
#if NO_INLINE_TINYSET == 1
    pmas.d = (pma_types_ *)new PMA<sized_uint<new_b>, Ts...>(new_pma_in_place);
#else
    if constexpr (new_b == 1) {
      new (&pmas.d[0].pma1)
          PMA<sized_uint<new_b>, Ts...>(std::move(new_pma_in_place));
    }
    if constexpr (new_b == 2) {
      new (&pmas.d[0].pma2)
          PMA<sized_uint<new_b>, Ts...>(std::move(new_pma_in_place));
    }
    if constexpr (new_b == 3) {
      new (&pmas.d[0].pma3)
          PMA<sized_uint<new_b>, Ts...>(std::move(new_pma_in_place));
    }
    if constexpr (new_b == 4) {
      new (&pmas.d[0].pma4)
          PMA<sized_uint<new_b>, Ts...>(std::move(new_pma_in_place));
    }
#endif
  } else {
    pmas.set_b(new_b);
    pmas.p = new_pmas;
  }
  pmas.set_b(new_b);
}

template <typename... Ts>
void TinySetV_small<Ts...>::insert(element_type e, const extra_data &d) {
  uint32_t pma_count = get_pma_count(d);
  if constexpr (use_bit_array) {
    if (pma_count > 1) {
      bit_array_set(bit_array_start(d), which_pma(e));
    }
  }
  bool inserted = false;
  uint32_t which = which_pma(std::get<0>(e));
  std::get<0>(e) = small_element(std::get<0>(e));
  switch (pmas.get_b()) {
  case 1:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.insert(e);
    break;
  case 2:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.insert(e);
    break;
  case 3:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.insert(e);
    break;
  case 4:
    inserted = ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.insert(e);
    break;
  }
  el_count += inserted;
  if (inserted) {
    if (el_count == d.thresh_24) {
      change_index<4, 3>(pma_count, d);
    } else if (el_count == d.thresh_16) {
      change_index<3, 2>(pma_count, d);
    } else if (el_count == d.thresh_8) {
      change_index<2, 1>(pma_count, d);
    }
  }
}

template <typename... Ts>
void TinySetV_small<Ts...>::insert_batch(element_type *els, uint64_t n,
                                         const extra_data &d) {
  for (uint64_t i = 0; i < n; i++) {
    insert(els[i], d);
  }
}

template <typename... Ts>
void TinySetV_small<Ts...>::remove(el_t e, const extra_data &d) {
  uint32_t pma_count = get_pma_count(d);
  if constexpr (use_bit_array) {
    if (pma_count > 1) {
      if (!bit_array_get(bit_array_start(d), which_pma(e))) {
        return;
      }
    }
  }
  bool removed = false;
  uint32_t which = which_pma(e);
  el_t small = small_element(e);
  switch (pmas.get_b()) {
  case 1:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.remove(small);

    if constexpr (use_bit_array) {
      if (pma_count > 1) {
        if (removed && pmas.p[which].pma1.is_empty()) {
          bit_array_flip(bit_array_start(d), which);
        }
      }
    }
    break;
  case 2:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.remove(small);
    if constexpr (use_bit_array) {
      if (pma_count > 1) {
        if (removed && pmas.p[which].pma2.is_empty()) {
          bit_array_flip(bit_array_start(d), which);
        }
      }
    }
    break;
  case 3:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.remove(small);
    if constexpr (use_bit_array) {
      if (pma_count > 1) {
        if (removed && pmas.p[which].pma3.is_empty()) {
          bit_array_flip(bit_array_start(d), which);
        }
      }
    }
    break;
  case 4:
    removed =
        ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.remove(small);
    if constexpr (use_bit_array) {
      if (pma_count > 1) {
        if (removed && pmas.p[which].pma4.is_empty()) {
          bit_array_flip(bit_array_start(d), which);
        }
      }
    }
    break;
  }
  el_count -= removed;
  if (removed) {
    if (el_count == d.thresh_24 - 1) {
      change_index<3, 4>(pma_count, d);
    } else if (el_count == d.thresh_16 - 1) {
      change_index<2, 3>(pma_count, d);
    } else if (el_count == d.thresh_8 - 1) {
      change_index<1, 2>(pma_count, d);
    }
  }
}

template <typename... Ts>
bool TinySetV_small<Ts...>::has(el_t e, const extra_data &d) const {
  uint32_t which = which_pma(e);
  uint32_t pma_count = get_pma_count(d);
  if constexpr (use_bit_array) {
    if (pma_count > 1 && !bit_array_get(bit_array_start(d), which)) {
      return false;
    }
  }

  el_t small = small_element(e);
  switch (pmas.get_b()) {
  case 1:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.has(small);
  case 2:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.has(small);
  case 3:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.has(small);
  case 4:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.has(small);
  }
  // should never happen
  return false;
}

template <typename... Ts>
typename TinySetV_small<Ts...>::value_type
TinySetV_small<Ts...>::value(el_t e, const extra_data &d) const {
  uint32_t which = which_pma(e);
  uint32_t pma_count = get_pma_count(d);
  if constexpr (use_bit_array) {
    if (pma_count > 1 && !bit_array_get(bit_array_start(d), which)) {
      return 0;
    }
  }
  el_t small = small_element(e);
  switch (pmas.get_b()) {
  case 1:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma1.value(small);
  case 2:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma2.value(small);
  case 3:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma3.value(small);
  case 4:
    return ((pma_count == 1) ? &pmas.d[0] : pmas.p)[which].pma4.value(small);
  }
  // should never happen
  return {};
}

template <typename... Ts> TinySetV_small<Ts...>::TinySetV_small() {
#if NO_INLINE_TINYSET == 1
  pmas.d = (pma_types_ *)new PMA<sized_uint<4>, Ts...>();
#else
  new (&pmas.d[0]) PMA<sized_uint<4>, Ts...>();
#endif
  pmas.set_b(4);
}

template <typename... Ts>
TinySetV_small<Ts...>::TinySetV_small(const TinySetV_small &source,
                                      const extra_data &d) {
  el_count = source.el_count;
  uint32_t new_pma_count = source.get_pma_count(d);
  uint8_t b = source.pmas.get_b();
  pmas.set_b(b);
  if (new_pma_count > 1) {
    if constexpr (use_bit_array) {
      pmas.p = (pma_types *)malloc(new_pma_count * sizeof(pma_types) +
                                   BitArray::bit_array_size(new_pma_count));
      memset(bit_array_start(d), 0, BitArray::bit_array_size(new_pma_count));
    } else {
      pmas.p = (pma_types *)malloc(new_pma_count * sizeof(pma_types));
    }
    pmas.set_b(b);
    switch (pmas.get_b()) {
    case 1: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<sized_uint<1>, Ts...>(source.pmas.p[i].pma1);
        if constexpr (use_bit_array) {
          if (!source.pmas.p[i].pma1.is_empty()) {
            bit_array_set(bit_array_start(d), i + 1);
          }
        }
      }
      break;
    }
    case 2: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<sized_uint<2>, Ts...>(source.pmas.p[i].pma2);
        if constexpr (use_bit_array) {
          if (!source.pmas.p[i].pma2.is_empty()) {
            bit_array_set(bit_array_start(d), i + 1);
          }
        }
      }
      break;
    }
    case 3: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<sized_uint<3>, Ts...>(source.pmas.p[i].pma3);
        if constexpr (use_bit_array) {
          if (!source.pmas.p[i].pma3.is_empty()) {
            bit_array_set(bit_array_start(d), i + 1);
          }
        }
      }
      break;
    }
    case 4: {
      for (uint32_t i = 0; i < new_pma_count; i++) {
        new (&pmas.p[i]) PMA<sized_uint<4>, Ts...>(source.pmas.p[i].pma4);
        if constexpr (use_bit_array) {
          if (!source.pmas.p[i].pma4.is_empty()) {
            bit_array_set(bit_array_start(d), i + 1);
          }
        }
      }
      break;
    }
    }

  } else {
    switch (pmas.get_b()) {
    case 1:
      new (&pmas.d[0].pma1) PMA<sized_uint<1>, Ts...>(source.pmas.d[0].pma1);
      break;
    case 2:
      new (&pmas.d[0].pma2) PMA<sized_uint<2>, Ts...>(source.pmas.d[0].pma2);
      break;
    case 3:
      new (&pmas.d[0].pma3) PMA<sized_uint<3>, Ts...>(source.pmas.d[0].pma3);
      break;
    case 4:
      new (&pmas.d[0].pma4) PMA<sized_uint<4>, Ts...>(source.pmas.d[0].pma4);
      break;
    }
  }
  pmas.set_b(b);
}

template <typename... Ts>
void TinySetV_small<Ts...>::destroy(const extra_data &d) {
  if (get_pma_count(d) == 1) {
#if NO_INLINE_TINYSET == 1
    switch (pmas.get_b()) {
    case 1:
      pmas.d->pma1.~PMA<1, value_type>();
      break;
    case 2:
      pmas.d->pma2.~PMA<2, value_type>();
      break;
    case 3:
      pmas.d->pma3.~PMA<3, value_type>();
      break;
    case 4:
      pmas.d->pma4.~PMA<4, value_type>();
      break;
    }
#else
    switch (pmas.get_b()) {
    case 1:
      pmas.d[0].pma1.~PMA<sized_uint<1>, Ts...>();
      break;
    case 2:
      pmas.d[0].pma2.~PMA<sized_uint<2>, Ts...>();
      break;
    case 3:
      pmas.d[0].pma3.~PMA<sized_uint<3>, Ts...>();
      break;
    case 4:
      pmas.d[0].pma4.~PMA<sized_uint<4>, Ts...>();
      break;
    }

#endif
    return;
  }
  for (uint32_t i = 0; i < get_pma_count(d); i++) {
    switch (pmas.get_b()) {
    case 1:
      pmas.p[i].pma1.~PMA<sized_uint<1>, Ts...>();
      break;
    case 2:
      pmas.p[i].pma2.~PMA<sized_uint<2>, Ts...>();
      break;
    case 3:
      pmas.p[i].pma3.~PMA<sized_uint<3>, Ts...>();
      break;
    case 4:
      pmas.p[i].pma4.~PMA<sized_uint<4>, Ts...>();
      break;
    }
  }
  free(pmas.p);
}
} // namespace SSTGraph
