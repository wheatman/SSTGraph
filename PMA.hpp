
#pragma once
#include "VertexSubset.hpp"
#include "helpers.h"
#include "packedarray.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <type_traits>

template <int index_size, typename value_type = bool, int num_raw_blocks = 1,
          int num_dense_blocks = 512>
class __attribute__((__packed__)) PMA {
  // class PMA {
  static_assert(index_size >= 1 && index_size <= 4,
                "for now index size needs to be between 1 and 4 bytes");
  static_assert(sizeof(value_type) >= 1 && sizeof(value_type) <= 8,
                "for now value size needs to be between 1 and 8 bytes");
  static_assert(num_raw_blocks < num_dense_blocks,
                "num_raw_blocks must be less than num_dense_blocks");
  static_assert(num_raw_blocks == 1 || num_raw_blocks == 2 ||
                    num_raw_blocks == 4,
                "num_raw_blocks has only been tested with 1, 2, and 4");

  static constexpr bool binary = std::is_same<value_type, bool>::value;

  using item_type =
      typename std::conditional<binary, std::tuple<el_t>,
                                std::tuple<el_t, value_type>>::type;

  static constexpr int value_type_size = (!binary) * sizeof(value_type);

  static constexpr int total_size = value_type_size + index_size;

private:
  // TODO(wheatman) not tested with other NULL_VALUEs
  static constexpr uint32_t NULL_VAL = 0;
  using array_union = union __attribute__((__packed__)) array_union_ {
    block_t *p_data;
    block_t raw_data[num_raw_blocks] = {0};
  };
  class empty_type {};

  class pair_type {

    uint32_t idx = 0;
    [[no_unique_address]]
    typename std::conditional<binary, empty_type, value_type>::type val;

  public:
    static constexpr uint8_t size = index_size + (!binary) * sizeof(value_type);
    static constexpr uint8_t i_size = index_size;
    static constexpr uint8_t v_size = (!binary) * sizeof(value_type);
    pair_type(uint32_t i, value_type v) : idx(i), val(v) {
      static_assert(!binary);
    }
    pair_type() = default;
    pair_type(uint32_t i) : idx(i) { static_assert(binary); }
    pair_type(item_type e) : idx(std::get<0>(e)) {
      if constexpr (!binary) {
        val = std::get<1>(e);
      }
    }
    pair_type(uint8_t const *address_key,
              [[maybe_unused]] uint8_t const *address_value) {

      if constexpr (index_size == 1) {
        memcpy(&idx, address_key, i_size);
      } else if constexpr (index_size == 2) {
        memcpy(&idx, address_key, i_size);
      } else if constexpr (index_size == 4) {
        memcpy(&idx, address_key, i_size);
      } else {
        memcpy(&idx, address_key, 4);
        idx &= index_byte_mask;
      }

      if constexpr (!binary) {
        auto *value_address =
            reinterpret_cast<const value_type *>(address_value);
        memcpy(&val, value_address, v_size);
      }
    }

    void write(uint8_t *address_key,
               [[maybe_unused]] uint8_t *address_value) const {
      memcpy(address_key, &idx, i_size);
      if constexpr (!binary) {
        auto *value_address = reinterpret_cast<value_type *>(address_value);
        // *value_address = val;
        memcpy(value_address, &val, v_size);
      }
    }

    [[nodiscard]] uint32_t index() const { return idx; }
    [[nodiscard]] value_type value() const {
      if constexpr (binary) {
        return true;
      } else {
        return val;
      }
    }
  };
  static constexpr uint32_t get_index_byte_mask(uint8_t size = index_size) {
    if (size == 0) {
      return 0;
    }
    return 0xFFU | (get_index_byte_mask(size - 1) << 8U);
  }
  static_assert(get_index_byte_mask(1) == 0xFFU);
  static_assert(get_index_byte_mask(2) == 0xFFFFU);
  static_assert(get_index_byte_mask(3) == 0xFFFFFFU);
  static_assert(get_index_byte_mask(4) == 0xFFFFFFFFU);
  static constexpr uint32_t index_byte_mask = get_index_byte_mask(index_size);
  static inline bool haszerobyte(uint64_t v) {
    return (((v)-0x0101010101010101UL) & ~(v)&0x8080808080808080UL);
  }
  static inline bool hasvaluebyte(uint64_t x, uint8_t n) {
    return (haszerobyte((x) ^ (~0UL / 255 * (n))));
  }
  array_union array;
  [[no_unique_address]]
  typename std::conditional<binary, empty_type, value_type>::type val_of_0 = {};

  // returns the value of 0 if it exists
  // only used to simplify other functions and remove the if constexpr when it
  // makes things more complicated
  [[nodiscard]] inline value_type zero_value_if_exists() const {
    if constexpr (binary) {
      return 1;
    } else {
      return val_of_0;
    }
  }
  bool has_0 : 1;
  uint8_t real_logN : 7;        // max value 32
  uint32_t count_elements : 21; // we know by 524288 we will use 16 bit elements
  // this needs to be at the end
  uint8_t b_spot : 3; // B will go here at the tinyset level

  [[nodiscard]] inline bool stored_dense() const {
    return ((total_size * count_elements) < num_dense_blocks * sizeof(block_t));
  }
  [[nodiscard]] inline bool
  would_be_stored_dense(uint32_t element_count) const {
    return ((total_size * element_count) < num_dense_blocks * sizeof(block_t));
  }
  [[nodiscard]] inline bool stored_in_place() const {
    return (total_size * count_elements <= sizeof(array_union));
  }

  static constexpr uint32_t get_max_in_place() {
    uint32_t ret = sizeof(array_union) / pair_type::size;
    return ret;
  }

  static constexpr uint32_t max_in_place = get_max_in_place();

  [[nodiscard]] inline bool
  would_be_stored_in_place(uint32_t element_count) const {
    return (total_size * element_count <= sizeof(array_union));
  }

  static constexpr std::array<std::array<double, 32>, 32>
  get_upper_density_bound_table() {
    std::array<std::array<double, 32>, 32> res = {};
    for (uint32_t depth = 0; depth < 32; depth++) {
      for (uint32_t real_logN = 0; real_logN < 32; real_logN++) {
        uint32_t div = bsr_word_constexpr(
            (1U << real_logN) / (1U << (bsr_word_constexpr(real_logN))));
        if (div == 0) {
          res[depth][real_logN] = 0;
          continue;
        }
        double upper = 3.0 / 4.0 + ((.25 * depth) / div);
        double bound =
            (static_cast<double>(1U << bsr_word_constexpr(real_logN)) - 1) /
            (1U << bsr_word_constexpr(real_logN));
        if (upper > bound) {
          upper = bound + .001;
        }
        res[depth][real_logN] = upper;
      }
    }
    return res;
  }

  static constexpr std::array<std::array<double, 32>, 32>
  get_lower_density_bound_table() {
    std::array<std::array<double, 32>, 32> res = {};
    for (uint32_t depth = 0; depth < 32; depth++) {
      for (uint32_t real_logN = 0; real_logN < 32; real_logN++) {
        if (depth <= real_logN) {
          uint32_t div = bsr_word_constexpr(
              (1U << real_logN) / (1U << (bsr_word_constexpr(real_logN))));
          if (div == 0) {
            res[depth][real_logN] = 0;
            continue;
          }
          double lower = 1.0 / 4.0 - ((.125 * depth) / div);
          res[depth][real_logN] = lower;
        } else {
          res[depth][real_logN] = 0;
        }
      }
    }
    return res;
  }

  static constexpr std::array<std::array<double, 32>, 32> upper_density_bound =
      get_upper_density_bound_table();

  static constexpr std::array<std::array<double, 32>, 32> lower_density_bound =
      get_lower_density_bound_table();

  [[nodiscard]] uint32_t N() const { return 1U << real_logN; }

  [[nodiscard]] uint8_t loglogN() const { return bsr_word(real_logN); }
  [[nodiscard]] uint32_t logN() const { return (1U << loglogN()); }
  [[nodiscard]] uint32_t mask_for_leaf() const { return ~(logN() - 1); }
  [[nodiscard]] uint32_t H() const { return bsr_word(N() / logN()); }

  void double_list();
  void half_list();
  void slide_right(uint32_t index);
  void slide_left(uint32_t index);
  void redistribute(uint32_t index, uint32_t len);

  [[nodiscard]] double get_density(uint32_t index, uint32_t len) const;
  [[nodiscard]] uint32_t get_density_count(uint32_t index, uint32_t len) const;

  [[nodiscard]] uint32_t search(el_t e) const;
  [[nodiscard]] uint32_t search_no_nulls_branchless(el_t e) const;
  [[nodiscard]] uint32_t
  search_no_nulls_counting(el_t e, uint32_t start = 0,
                           uint32_t end = UINT32_MAX) const;

  void place(uint32_t index, pair_type e);
  void take(uint32_t index);

  [[nodiscard]] uint32_t find_prev_valid(uint32_t start) const;

  [[nodiscard]] bool check_no_full_leaves(uint32_t index, uint32_t len) const;

  [[nodiscard]] uint32_t find_leaf(uint32_t index) const {
    return index & mask_for_leaf();
  }

  // same as find_leaf, but does it for any level in the tree
  // index: index in array
  // len: length of sub-level.
  [[nodiscard]] uint32_t find_node(uint32_t index, uint32_t len) const {
    return (index / len) * len;
  }
  [[nodiscard]] uint32_t next_leaf(uint32_t index) const {
    return ((index >> loglogN()) + 1) << (loglogN());
  }
  [[nodiscard]] uint32_t next_leaf(uint32_t index, uint32_t llN) const {
    return ((index >> llN) + 1) << (llN);
  }

  [[nodiscard]] bool check_packed_left() const;
  void print_array(uint32_t start = 0, uint32_t end = UINT32_MAX) const;
  [[nodiscard]] bool has_no_zero_non_empty(el_t e) const;
  [[nodiscard]] value_type value_no_zero_non_empty(el_t e) const;

public:
  // Basic Usage
  PMA();
  PMA(const PMA &source);
  ~PMA();

  void print_pma(uint32_t prefix = 0) const;

  // Query state

  [[nodiscard]] uint64_t get_size() const;
  [[nodiscard]] uint64_t get_n() const { return count_elements + has_0; }

  [[nodiscard]] bool is_empty() const { return get_n() == 0; }
  [[nodiscard]] bool use_fast_iter() const { return stored_dense(); }

  // Query Data

  [[nodiscard]] bool has(el_t e) const;
  value_type value(el_t e) const;

  [[nodiscard]] uint64_t sum_keys(uint64_t prefix = 0) const;
  [[nodiscard]] uint64_t sum_keys_no_it() const;

  template <class F> bool map(F &f, uint64_t prefix) const;

  // Modifications

  bool insert(item_type e);
  bool insert(el_t e) {
    static_assert(binary);
    return insert(std::make_tuple(e));
  }
  bool remove(el_t e);

  // For Verification
  [[nodiscard]] bool check_pma(uint8_t guess_b,
                               uint64_t max_val = UINT32_MAX) const;

  // Needed for higher level structures
  void clean_no_free() {
    count_elements = 0;
    // b = 0;
    has_0 = false;
    real_logN = 0;
  }

  void prefetch_data() const { __builtin_prefetch(array.p_data, 0, 3); }

  void shallow_copy(const PMA *source) {
    memcpy(static_cast<void *>(this), source, sizeof(PMA));
  }

  class iterator {

    uint64_t length;
    const block_t *array;
    uint64_t place;
    uint8_t loglogN;
    value_type zero_val = 0;
    uint32_t n;

  public:
    explicit iterator(uint64_t pl)
        : length(0), array(nullptr), place(pl), loglogN(0), n(0) {}

    explicit iterator(const PMA &pma, value_type z_val, bool skip_zero = false)
        : length(pma.stored_dense() ? pma.count_elements : pma.N()),
          array(pma.stored_in_place() ? pma.array.raw_data : pma.array.p_data),
          place((pma.has_0 && (!skip_zero)) ? UINT64_MAX : 0),
          loglogN(pma.loglogN()), zero_val(z_val),
          n(pma.stored_in_place() ? PMA::max_in_place : pma.N()) {}

    bool operator==(const iterator &other) const {
      return (place == other.place);
    }
    bool operator!=(const iterator &other) const {
      return (place != other.place);
    }
    bool operator<(const iterator &other) const {
      return (place < other.place) || (place == UINT64_MAX);
    }
    bool operator>=(const iterator &other) const {
      return (place >= other.place) && (place != UINT64_MAX);
    }
    iterator &operator++() {
      place += 1;
      while (
          (place < length) &&
          (PackedArray_get<pair_type>(array, place, n).index() == NULL_VAL)) {
        place = ((place >> loglogN) + 1) << (loglogN);
      }
      return *this;
    }
    std::pair<el_t, value_type> operator*() const {
      if (place == UINT64_MAX) {
        if constexpr (binary) {
          return {0, true};
        } else {
          return {0, zero_val};
        }
      }
      pair_type p = PackedArray_get<pair_type>(array, place, n);
      if constexpr (binary) {
        return {p.index(), true};
      } else {
        return {p.index(), p.value()};
      }
    }
    ~iterator() = default;
  };

  iterator begin(bool skip_zero = false) const {
    if constexpr (binary) {
      return iterator(*this, has_0, skip_zero);
    } else {
      return iterator(*this, val_of_0, skip_zero);
    }
  }
  iterator end() const {
    uint64_t length = N();
    if (stored_dense()) {
      length = count_elements;
    }
    return iterator(length);
  }

  class iterator_fast {

    uint64_t length;
    const block_t *array;
    uint64_t place;
    uint32_t n;

  public:
    explicit iterator_fast(uint64_t pl)
        : length(0), array(nullptr), place(pl), n(0) {}
    explicit iterator_fast(const PMA &pma)
        : length(pma.count_elements),
          array(pma.stored_in_place() ? pma.array.raw_data : pma.array.p_data),
          place(0), n(pma.stored_in_place() ? PMA::max_in_place : pma.N()) {}
    bool operator==(const iterator_fast &other) const {
      return (place == other.place);
    }
    bool operator!=(const iterator_fast &other) const {
      return (place != other.place);
    }
    bool operator<(const iterator_fast &other) const {
      return (place < other.place);
    }
    iterator_fast &operator++() {
      place += 1;
      return *this;
    }
    std::pair<el_t, value_type> operator*() const {
      pair_type p = PackedArray_get<pair_type>(array, place, n);
      if constexpr (binary) {
        return {p.index(), true};
      } else {
        return {p.index(), p.value()};
      }
    }
    ~iterator_fast() = default;
  };
  iterator_fast begin_fast() const { return iterator_fast(*this); }
  iterator_fast end_fast(uint64_t offset = 0) const {
    uint64_t length = count_elements;
    return iterator_fast(length - offset);
  }
};

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::check_pma(
    uint8_t guess_b, uint64_t max_val) const {
  if (index_size != guess_b) {
    return false;
  }
  PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::iterator it =
      begin();
  for (; it < end(); ++it) {
    auto e = *it;
    if (e.first > max_val) {
      return false;
    }
  }
  return true;
}

struct SUM_KEYS_PMA {
  static constexpr bool no_early_exit = true;
  uint64_t result = 0;
  inline bool update(el_t key) {
    result += key;
    return false;
  }
};

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint64_t
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::sum_keys_no_it()
    const {
  struct SUM_KEYS_PMA v;
  map(v);
  return v.result;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
template <class F>
bool PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::map(
    F &f, uint64_t prefix) const {
  static_assert(
      std::is_invocable_v<decltype(&F::update), F &, uint32_t> ||
          std::is_invocable_v<decltype(&F::update), F &, uint32_t, value_type>,
      "update function must take in an index type and it can "
      "also take in a "
      "value_type");
  constexpr bool keys_only =
      std::is_invocable_v<decltype(&F::update), F &, uint32_t>;
  if (has_0) {
    if constexpr (F::no_early_exit) {
      if constexpr (keys_only) {
        f.update(prefix);
      } else {
        f.update(prefix, zero_value_if_exists());
      }
    } else {
      bool r;
      if constexpr (keys_only) {
        r = f.update(prefix);
      } else {
        r = f.update(prefix, zero_value_if_exists());
      }
      if (r) {
        return true;
      }
    }
  }
  uint32_t n = N();
  if (__builtin_expect(stored_dense(), 1)) { // also deals with in place
    const block_t *const data =
        (stored_in_place()) ? array.raw_data : array.p_data;
    if (stored_in_place()) {
      n = max_in_place;
    }

    for (uint32_t i = 0; i < count_elements; i++) {
      auto pair = PackedArray_get<pair_type>(data, i, n);
      if constexpr (F::no_early_exit) {
        if constexpr (keys_only) {
          f.update(pair.index() + prefix);
        } else {
          f.update(pair.index() + prefix, pair.value());
        }
      } else {
        bool r;
        if constexpr (keys_only) {
          r = f.update(pair.index() + prefix);
        } else {

          r = f.update(pair.index() + prefix, pair.value());
        }
        if (r) {
          return true;
        }
      }
    }
    return false;
  }
  // not stored dense
  for (uint32_t i = 0; i < N(); i++) {
    auto pair = PackedArray_get<pair_type>(array.p_data, i, n);
    if (pair.index() != NULL_VAL) {
      if constexpr (F::no_early_exit) {
        if constexpr (keys_only) {
          f.update(pair.index() + prefix);
        } else {
          f.update(pair.index() + prefix, pair.value());
        }
      } else {
        bool r;
        if constexpr (keys_only) {
          r = f.update(pair.index() + prefix);
        } else {
          r = f.update(pair.index() + prefix, pair.value());
        }
        if (r) {
          return true;
        }
      }
    }
  }
  return false;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint64_t
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::sum_keys(
    uint64_t prefix) const {
  uint64_t result = prefix * get_n();

  if (use_fast_iter()) {
    for (PMA<index_size, value_type, num_raw_blocks,
             num_dense_blocks>::iterator_fast it = begin_fast();
         it != end_fast(); ++it) {
      result += (*it).first;
    }
  } else {
    auto it_end = end();
    for (PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::iterator
             it = begin();
         it != it_end; ++it) {
      result += (*it).first;
    }
  }
  return result;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint32_t
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::find_prev_valid(
    uint32_t start) const {
  while (PackedArray_get<pair_type>(array.p_data, start, N()).index() ==
             NULL_VAL &&
         start > 0) {
    start--;
  }
  return start;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint64_t
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::get_size()
    const {
  uint64_t size = 0;
  if (!stored_in_place()) {
    size += PackedArray_get_size<pair_type>(N());
  }
  return size +
         sizeof(PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>);
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::print_array(
    uint32_t start, uint32_t end) const {
  if (end > N()) {
    end = N();
  }
  printf("N = %u, logN = %u, loglogN = %u, H = %u\n", N(), logN(), loglogN(),
         H());
  printf("count_elements %u, b = %u\n", count_elements, index_size);
  if (end - start > 500) {
    printf("too big to print\n");
    return;
  }
  if (has_0) {
    if constexpr (binary) {
      std::cout << "0, ";
    } else {
      std::cout << "(0, " << +val_of_0 << "), ";
    }
  }
  for (uint32_t i = start; i < end; i++) {
    if (PackedArray_get<pair_type>(array.p_data, i, N()).index() != NULL_VAL) {

      if constexpr (binary) {
        std::cout << PackedArray_get<pair_type>(array.p_data, i, N()).index()
                  << ", ";
      } else {
        std::cout << "("
                  << PackedArray_get<pair_type>(array.p_data, i, N()).index()
                  << ", "
                  << +PackedArray_get<pair_type>(array.p_data, i, N()).value()
                  << "), ";
      }
    }
  }
  printf("\n");
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::print_pma(
    uint32_t prefix) const {
  printf("PMA: prefix = %u, count_elements %u, index_size = %d, "
         "value_size = "
         "%d, pair_type::size = %u\n",
         prefix, count_elements, index_size, (int)pair_type::v_size,
         pair_type::size);
  if (has_0) {
    printf("the pma has 0\n");
    if constexpr (!binary) {
      std::cout << "the value of zero is " << +val_of_0 << std::endl;
    }
  }
  if (count_elements == 0) {
    printf("the pma is empty\n");
  } else if (stored_in_place()) {
    printf("the pma is only storing data in place\n");
    printf("the elements are:\n");
    for (uint32_t i = 0; i < count_elements; i++) {
      if constexpr (binary) {
        std::cout << prefix + PackedArray_get<pair_type>(&array.raw_data[0], i,
                                                         max_in_place)
                                  .index()
                  << ", ";
      } else {
        std::cout << "("
                  << prefix + PackedArray_get<pair_type>(&array.raw_data[0], i,
                                                         max_in_place)
                                  .index()
                  << ", "
                  << +PackedArray_get<pair_type>(&array.raw_data[0], i,
                                                 max_in_place)
                          .value()
                  << "), ";
      }
    }
    printf("\n");
  } else {
    printf("N = %u, logN = %u, loglogN = %u, H = %u\n", N(), logN(), loglogN(),
           H());
    for (uint32_t i = 0; i < N(); i += logN()) {
      for (uint32_t j = i; j < i + logN(); j++) {
        if (PackedArray_get<pair_type>(array.p_data, j, N()).index() !=
            NULL_VAL) {

          if constexpr (binary) {
            std::cout
                << prefix +
                       PackedArray_get<pair_type>(array.p_data, j, N()).index()
                << ", ";
          } else {
            std::cout
                << "("
                << prefix +
                       PackedArray_get<pair_type>(array.p_data, j, N()).index()
                << ", "
                << +PackedArray_get<pair_type>(array.p_data, j, N()).value()
                << "), ";
          }
        } else {
          std::cout << "_" << j << "_,";
        }
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint32_t PMA<index_size, value_type, num_raw_blocks,
             num_dense_blocks>::get_density_count(uint32_t index,
                                                  uint32_t len) const {
  uint32_t full = 0;
  for (uint32_t i = index; i < index + len; i += 4) {
    // TODO(wheatman) this can be made faster
    uint32_t add =
        (PackedArray_get<pair_type>(array.p_data, i, N()).index() != NULL_VAL) +
        (PackedArray_get<pair_type>(array.p_data, i + 1, N()).index() !=
         NULL_VAL) +
        (PackedArray_get<pair_type>(array.p_data, i + 2, N()).index() !=
         NULL_VAL) +
        (PackedArray_get<pair_type>(array.p_data, i + 3, N()).index() !=
         NULL_VAL);
    full += add;
  }
  return full;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
double
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::get_density(
    uint32_t index, uint32_t len) const {
  uint32_t full = get_density_count(index, len);
  double full_d = (double)full;
  return full_d / len;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks,
         num_dense_blocks>::check_no_full_leaves(uint32_t index,
                                                 uint32_t len) const {
  for (uint32_t i = index; i < index + len; i += logN()) {
    bool full = true;
    for (uint32_t j = i; j < i + logN(); j++) {
      if (PackedArray_get<pair_type>(array.p_data, j, N()).index() ==
          NULL_VAL) {
        full = false;
      }
    }
    if (full) {
      return false;
    }
  }
  return true;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks,
         num_dense_blocks>::check_packed_left() const {
  for (uint32_t i = 0; i < N(); i += logN()) {
    bool zero = false;
    for (uint32_t j = i; j < i + logN(); j++) {
      if (PackedArray_get<pair_type>(array.p_data, j, N()).index() ==
          NULL_VAL) {
        zero = true;
      } else if (zero) {
        for (uint32_t k = i; k < i + logN(); k++) {
          printf("arr[%u]=%u, ", k,
                 PackedArray_get<pair_type>(array.p_data, k, N()).index());
        }
        printf("\n");
        return false;
      }
    }
  }
  return true;
}

// Evenly redistribute elements in the pma, given a range to look into
// index: starting position in pma structure
// len: area to redistribute
// should already be locked
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks,
         num_dense_blocks>::redistribute(uint32_t index, uint32_t len) {
  // TODO(wheatman) if its small use the stack
  if (len == logN()) {
    return;
  }
  uint32_t j = 0;
  if (len > 1024) {
    block_t *space = create_PackedArray<pair_type>(len);
    // TODO(wheatman)
    // could get better cache behavior if I go back and forth with reading
    // and writing
    for (uint32_t i = index; i < index + len; i++) {
      pair_type item = PackedArray_get<pair_type>(array.p_data, i, N());
      PackedArray_set<pair_type>(space, j, len, item);
      // counting non-null
      j += (item.index() != NULL_VAL);
    }
    PackedArray_memset_0<pair_type>(array.p_data, index, index + len, N());
    assert(((double)j) / len <= ((double)(logN() - 1) / logN()));

    uint32_t num_leaves = len >> loglogN();
    uint32_t count_per_leaf = j / num_leaves;
    uint32_t extra = j % num_leaves;

    // parallizing does not make it faster
    for (uint32_t i = 0; i < num_leaves; i++) {
      uint32_t count_for_leaf = count_per_leaf + (i < extra);
      uint32_t in = index + (logN() * (i));
      uint32_t j2 = count_per_leaf * i + std::min(i, extra);
      for (uint32_t k = in; k < count_for_leaf + in; k++) {
        PackedArray_set<pair_type>(array.p_data, k, N(),
                                   PackedArray_get<pair_type>(space, j2, len));
        j2++;
      }
    }
    free(space);
  } else {
    pair_type *space = (pair_type *)malloc(len * sizeof(pair_type));
    // could get better cache behavior if I go back and forth with reading
    // and writing
    for (uint32_t i = index; i < index + len; i++) {
      pair_type item = PackedArray_get<pair_type>(array.p_data, i, N());
      space[j] = item;
      // counting non-null
      j += (item.index() != NULL_VAL);
    }
    PackedArray_memset_0<pair_type>(array.p_data, index, index + len, N());
    assert(((double)j) / len <= ((double)(logN() - 1) / logN()));

    uint32_t num_leaves = len >> loglogN();
    uint32_t count_per_leaf = j / num_leaves;
    uint32_t extra = j % num_leaves;

    for (uint32_t i = 0; i < num_leaves; i++) {
      uint32_t count_for_leaf = count_per_leaf + (i < extra);
      uint32_t in = index + (logN() * (i));
      uint32_t j2 = count_per_leaf * i + std::min(i, extra);
      for (uint32_t k = in; k < count_for_leaf + in; k++) {
        PackedArray_set<pair_type>(array.p_data, k, N(), space[j2]);
        j2++;
      }
    }
    free(space);
  }
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks,
         num_dense_blocks>::double_list() {
  array.p_data = PackedArray_double<pair_type>(array.p_data, N());
  real_logN += 1;
  redistribute(0, N());
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks,
         num_dense_blocks>::half_list() {
  block_t *new_array = create_PackedArray<pair_type>(N() / 2);
  uint32_t index = 0;
  auto it_end = end();
  for (PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::iterator
           it = begin(true);
       it != it_end; ++it) {
    if constexpr (binary) {
      PackedArray_set<pair_type>(new_array, index, N() / 2,
                                 pair_type((*it).first));
    } else {
      PackedArray_set<pair_type>(new_array, index, N() / 2,
                                 pair_type((*it).first, (*it).second));
    }
    index++;
  }
  free(array.p_data);
  array.p_data = new_array;
  real_logN -= 1;
  redistribute(0, N());
}

// index is the beginning of the sequence that you want to slide right.
// we wil always hold locks to the end of the leaf so we don't need to
// lock here
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::slide_right(
    uint32_t index) {
  PackedArray_slide_right<pair_type>(array.p_data, index, next_leaf(index),
                                     N());
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::slide_left(
    uint32_t index) {
  PackedArray_slide_left<pair_type>(array.p_data, index, next_leaf(index), N());
}

// algorithm taken from
// https://dirtyhandscoding.wordpress.com/2017/08/25/performance-comparison-linear-search-vs-binary-search/
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint32_t PMA<index_size, value_type, num_raw_blocks,
             num_dense_blocks>::search_no_nulls_counting(el_t e, uint32_t start,
                                                         uint32_t end) const {
  if (end == UINT32_MAX) {
    end = count_elements;
  }
  uint32_t cnt = start;
  uint32_t len = end - start;
  switch (len) {
  case 0:
    return 0;
  case 1:
    return (PackedArray_get<pair_type>(array.p_data, start, N()).index() < e);
  case 2:
    cnt += (PackedArray_get<pair_type>(array.p_data, start, N()).index() < e);
    cnt +=
        (PackedArray_get<pair_type>(array.p_data, start + 1, N()).index() < e);
    return cnt;
  case 3:
    cnt += (PackedArray_get<pair_type>(array.p_data, start, N()).index() < e);
    cnt +=
        (PackedArray_get<pair_type>(array.p_data, start + 1, N()).index() < e);
    cnt +=
        (PackedArray_get<pair_type>(array.p_data, start + 2, N()).index() < e);
    return cnt;
  default:
#if defined(__SSE4_2__)
    if constexpr (index_size == 3 && binary) {
      __m128i shuf =
          _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
      __m128i cnt_vec = _mm_setzero_si128();
      __m128i e_vec = _mm_set1_epi32(e);
      uint32_t i = start * 3;
      for (; i < end * 3 - 11; i += 12) {
        __m128i data_vector =
            _mm_loadu_si128((__m128i *)(((uint8_t *)array.p_data) + i));
        __m128i el_vector = _mm_shuffle_epi8(data_vector, shuf);
        cnt_vec = _mm_sub_epi32(cnt_vec, _mm_cmplt_epi32(el_vector, e_vec));
      }
      uint64_t bottom = _mm_extract_epi64(cnt_vec, 0);
      uint64_t top = _mm_extract_epi64(cnt_vec, 1);
      // we know these can be at most 24 bits so we have no worry of
      // overflow
      uint64_t tmp = bottom + top;
      tmp += (tmp >> 32U);
      cnt += (uint32_t)tmp;
      i /= 3;
      for (; i < end; i++) {
        cnt += (PackedArray_get<pair_type>(array.p_data, i, N()).index() < e);
      }
      return cnt;
    }
#endif
    uint32_t i = start;
    for (; i < end - 3; i += 4) {
      cnt += (PackedArray_get<pair_type>(array.p_data, i, N()).index() < e);
      cnt += (PackedArray_get<pair_type>(array.p_data, i + 1, N()).index() < e);
      cnt += (PackedArray_get<pair_type>(array.p_data, i + 2, N()).index() < e);
      cnt += (PackedArray_get<pair_type>(array.p_data, i + 3, N()).index() < e);
    }
    for (; i < end; i++) {
      cnt += (PackedArray_get<pair_type>(array.p_data, i, N()).index() < e);
    }
  }
  return cnt;
}

// needs to be larger than the smallest element
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint32_t PMA<index_size, value_type, num_raw_blocks,
             num_dense_blocks>::search_no_nulls_branchless(el_t e) const {
  static constexpr uint32_t cutoff = 64;
  if (count_elements <= cutoff) {
    return search_no_nulls_counting(e, 0, count_elements);
  }
  uint32_t pos = -1;
  uint32_t t = count_elements;
  uint32_t logstep = bsr_word(t);
  uint32_t first_step = t + 1 - (1U << logstep);
  uint32_t step = 1U << logstep;
  pos = (PackedArray_get<pair_type>(array.p_data, pos + step, N()).index() < e
             ? pos + first_step
             : pos);
  step >>= 1U;
  while (step >= cutoff) {
    pos = (PackedArray_get<pair_type>(array.p_data, pos + step, N()).index() < e
               ? pos + step
               : pos);
    step >>= 1U;
  }
  pos += 1;
  uint32_t end = pos + step * 2 - 1;
  return search_no_nulls_counting(e, pos, end);
}

// returns the index of the smallest element bigger than you in the range
// [start, end)
// if no such element is found, returns end (because insert shifts
// everything to the right)
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
uint32_t PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::search(
    el_t e) const {
  if (e <= PackedArray_get<pair_type>(array.p_data, 0, N()).index()) {
    return 0;
  }
  uint32_t s = 0;
  uint32_t t = N() - 1;
  if (stored_dense()) {
    return search_no_nulls_branchless(e);
  }
  uint32_t mid = (s + t) / 2;
  uint32_t llN = loglogN();
  while (s + 1 < t) {
    el_t item = PackedArray_get<pair_type>(array.p_data, mid, N()).index();

    // if is_null
    if (item == NULL_VAL) {
      // first check the next leaf
      uint32_t check = next_leaf(mid, llN);
      if (check > t) {
        // first check the next leaf
        t = mid;
        mid = (s + t) / 2;
        continue;
      }
      // if is_null
      if (PackedArray_get<pair_type>(array.p_data, check, N()).index() ==
          NULL_VAL) {
        // first check the next leaf
        uint32_t early_check = find_prev_valid(mid);

        if (early_check < s) {
          // first check the next leaf
          s = mid;
          mid = (s + t) / 2;
          continue;
        }
        check = early_check;
      }
      el_t item = PackedArray_get<pair_type>(array.p_data, check, N()).index();
      if (item == e) {
        // cleanup before return
        return check;
      } else if (e < item) {
        t = find_prev_valid(mid) + 1;
      } else {
        if (check == s) {
          s = check + 1;
        } else {
          s = check;
        }
        // otherwise, searched for item is more than current and we set
        // start
      }
      mid = (s + t) / 2;
      //__builtin_prefetch ((void *)&dests[mid], 0, 3);
      continue;
    }

    if (e < item) {
      t = mid; // if the searched for item is less than current item, set
               // end
      mid = (s + t) / 2;
    } else if (e > item) {
      s = mid;
      mid = (s + t) / 2;
      // otherwise, sesarched for item is more than current and we set start
    } else if (e == item) { // if we found it, return
      return mid;
    }
  }
  if (t < s) {
    s = t;
  }

  // trying to encourage the packed left property so if they are both null
  // go to the left
  if ((PackedArray_get<pair_type>(array.p_data, s, N()).index() == NULL_VAL) &&
      (PackedArray_get<pair_type>(array.p_data, t, N()).index() == NULL_VAL)) {
    t = s;
  }

  // handling the case where there is one element left
  // if you are leq, return start (index where elt is)
  // otherwise, return end (no element greater than you in the range)
  if (e <= PackedArray_get<pair_type>(array.p_data, s, N()).index() &&
      (PackedArray_get<pair_type>(array.p_data, s, N()).index() != NULL_VAL)) {
    t = s;
  }
  // really insure packed left
  // TODO(wheatman) we shouldn't need this
  while (!((find_leaf(t) == t) ||
           (PackedArray_get<pair_type>(array.p_data, t - 1, N()).index() !=
            NULL_VAL))) {
    t--;
  }
  return t;
}

// insert elem at index
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::place(
    uint32_t index, pair_type e) {
  if (e.index() == 0) {
    has_0 = true;
    return;
  }
  if (e.index() == NULL_VAL) {
    printf("you can't insert the null value\n");
    assert(false);
  }
  uint32_t level = H();
  uint32_t len = logN();

  // always deposit on the left
  if (PackedArray_get<pair_type>(array.p_data, index, N()).index() ==
      NULL_VAL) {
    PackedArray_set<pair_type>(array.p_data, index, N(), e);
  } else {
    slide_right(index);
    PackedArray_set<pair_type>(array.p_data, index, N(), e);
  }

  uint32_t node_index = find_leaf(index);
  double density = get_density(node_index, len);
  // spill over into next level up, node is completely full.
  if (density == 1) {
    len *= 2;
    level--;
    node_index = find_node(node_index, len);
  }

  // get density of the leaf you are in
  uint32_t density_count = get_density_count(node_index, len);
  density = ((double)density_count) / len;

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density >= upper_density_bound[level][real_logN]) {
    len *= 2;
    if (len <= N()) {
      level--;
      uint32_t new_node_index = find_node(node_index, len);

      if (new_node_index < node_index) {
        density_count += get_density_count(new_node_index, len / 2);
      } else {
        density_count += get_density_count(new_node_index + len / 2, len / 2);
      }
      node_index = new_node_index;
      density = ((double)density_count) / len;
    } else {
      double_list();
      return;
    }
  }
  assert(((double)get_density_count(node_index, len)) / len == density);
  assert(density < upper_density_bound[level][real_logN]);
  assert(density <= (((double)logN() - 1) / logN()));
  if (len > logN()) {
    redistribute(node_index, len);
  }
  assert(check_no_full_leaves(node_index, len));
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::insert(
    item_type e) {
  const el_t key = std::get<0>(e);
  if (key == 0) {
    bool inserted = !has_0;
    has_0 = true;
    if constexpr (!binary) {
      val_of_0 = std::get<1>(e);
    }
    return inserted;
  }
  if (stored_in_place()) {
    uint32_t first_greater = 0;
    for (uint32_t i = 0; i < count_elements; i++) {
      pair_type item =
          PackedArray_get<pair_type>(&array.raw_data[0], i, max_in_place);
      if (item.index() == key) {
        // update the value if its different
        if constexpr (!binary) {
          if (item.value() != std::get<1>(e)) {
            PackedArray_set<pair_type>(&array.raw_data[0], i, max_in_place,
                                       pair_type(e));
          }
        }
        return false;
      } else if (key > item.index()) {
        first_greater++;
      }
    }
    if (would_be_stored_in_place(count_elements + 1)) {
      if (first_greater == count_elements) {
        PackedArray_set<pair_type>(&array.raw_data[0], count_elements,
                                   max_in_place, pair_type(e));
      } else {
        PackedArray_insert<pair_type>(&array.raw_data[0], pair_type(e),
                                      count_elements + 1, first_greater,
                                      max_in_place, false);
      }
      count_elements += 1;
      return true;
    } else if (stored_in_place()) {
      // raw is full, insert all the elements and switch over
      int32_t raw_count_saved = count_elements;
      array_union raw_data_saved;
      memcpy(&raw_data_saved, array.raw_data, sizeof(array_union));
      while (count_elements > N()) {
        real_logN += 1;
      }
      array.p_data = create_PackedArray<pair_type>(N());
      for (int i = 0; i < raw_count_saved; i++) {
        block_t *old_array = array.p_data;
        auto pair = PackedArray_get<pair_type>(&raw_data_saved.raw_data[0], i,
                                               max_in_place);
        array.p_data =
            PackedArray_insert_end<pair_type>(array.p_data, pair, i, N());
        if (old_array != array.p_data) {
          real_logN += 1;
        }
      }
    }
  }
  uint32_t index = search(key);
  // if its already in the array
  if (index < N() &&
      PackedArray_get<pair_type>(array.p_data, index, N()).index() == key) {
    // update its value if needed
    if constexpr (!binary) {
      if (PackedArray_get<pair_type>(array.p_data, index, N()).value() !=
          std::get<1>(e)) {
        PackedArray_set<pair_type>(array.p_data, index, N(), pair_type(e));
      }
    }
    return false;
  }
  count_elements += 1;
  if (stored_dense()) {
    block_t *old_array = array.p_data;
    array.p_data = PackedArray_insert<pair_type>(
        array.p_data, pair_type(e), count_elements, index, N(), true);
    if (old_array != array.p_data) {
      real_logN += 1;
    }
  } else {
    if (would_be_stored_dense(count_elements - 1)) {
      double_list();
      index = search(key);
    }
    place(index, pair_type(e));
  }
  return true;
}

// remove elem at index
template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
void PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::take(
    uint32_t index) {
  uint32_t level = H();
  uint32_t len = logN();

  slide_left(index);

  uint32_t node_index = find_leaf(index);

  // get density of the leaf you are in
  uint32_t density_count = get_density_count(node_index, len);
  double density = ((double)density_count) / len;

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density <= lower_density_bound[level][real_logN]) {
    len *= 2;
    if (len <= N()) {
      level--;
      uint32_t new_node_index = find_node(node_index, len);

      if (new_node_index < node_index) {
        density_count += get_density_count(new_node_index, len / 2);
      } else {
        density_count += get_density_count(new_node_index + len / 2, len / 2);
      }
      node_index = new_node_index;
      density = ((double)density_count) / len;
    } else {
      half_list();
      return;
    }
  }
  if (len > logN()) {
    redistribute(node_index, len);
  }
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::remove(
    el_t e) {
  if (!has(e)) {
    return false;
  }
  if (e == 0) {
    bool removed = has_0;
    has_0 = false;
    return removed;
  }
  if (stored_in_place()) {
    for (uint32_t i = 0; i < count_elements; i++) {
      el_t item =
          PackedArray_get<pair_type>(&array.raw_data[0], i, max_in_place)
              .index();
      if (item == e) {
        PackedArray_remove<pair_type>(&array.raw_data[0], count_elements - 1, i,
                                      max_in_place, false);
        count_elements -= 1;
        return true;
      }
    }
  } else if (stored_dense()) {
    // TODO(wheatman) could be speed up by using overestimate and going
    // backwords for 24 bit items
    for (uint32_t i = 0; i < count_elements; i++) {
      el_t item = PackedArray_get<pair_type>(array.p_data, i, N()).index();
      if (item == e) {
        block_t *old_array = array.p_data;
        array.p_data = PackedArray_remove<pair_type>(
            array.p_data, count_elements - 1, i, N(), true);
        if (old_array != array.p_data) {
          real_logN -= 1;
        }
        count_elements -= 1;
        break;
      }
    }
    // if we should now be in place due to getting smaller
    if (stored_in_place()) {
      block_t *old_array = array.p_data;
      for (uint32_t i = 0; i < count_elements; i++) {
        PackedArray_set<pair_type>(
            &array.raw_data[0], i, max_in_place,
            PackedArray_get<pair_type>(old_array, i, N()));
      }
      free(old_array);
      return true;
    } else {
      return true;
    }
  }
  // if I am going to be moving back to a dense array do the remove and the
  // compress in one pass
  count_elements -= 1;
  if (stored_dense()) {
    block_t *new_array = create_PackedArray<pair_type>(count_elements);
    uint32_t new_size =
        PackedArray_get_size<pair_type>(count_elements) / pair_type::size;
    uint32_t index = 0;
    count_elements += 1;
    auto it_end = end();
    for (PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::iterator
             it = begin(true);
         it != it_end; ++it) {
      if ((*it).first != e) {
        if constexpr (binary) {
          PackedArray_set<pair_type>(new_array, index, new_size,
                                     pair_type((*it).first));
        } else {
          PackedArray_set<pair_type>(new_array, index, new_size,
                                     pair_type((*it).first, (*it).second));
        }

        index++;
      }
    }
    count_elements -= 1;
    uint64_t new_logN = bsr_word(count_elements);
    if (1U << new_logN < count_elements) {
      new_logN += 1;
    }
    real_logN = new_logN;
    free(array.p_data);
    array.p_data = new_array;
    return true;
  }
  count_elements += 1;
  uint32_t index = search(e);
  take(index);
  count_elements -= 1;
  return true;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks,
         num_dense_blocks>::has_no_zero_non_empty(el_t e) const {
  if (count_elements > 0 && stored_in_place()) {
    if constexpr (index_size == 1 && binary) {
      static_assert(num_raw_blocks * 8 == sizeof(array_union),
                    "this optimization assumes num_raw_blocks * 8 = "
                    "sizeof(array_union)");
      bool ret = false;
      for (int i = 0; i < num_raw_blocks; i++) {
        ret |= hasvaluebyte(array.raw_data[i], e);
      }
      return ret;
    }
    for (uint32_t i = 0; i < count_elements; i++) {
      if (e == PackedArray_get<pair_type>(&array.raw_data[0], i, max_in_place)
                   .index()) {
        return true;
      }
    }
    return false;
  }
  uint32_t index = search(e);
  return (index < N() &&
          PackedArray_get<pair_type>(array.p_data, index, N()).index() == e);
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
value_type PMA<index_size, value_type, num_raw_blocks,
               num_dense_blocks>::value_no_zero_non_empty(el_t e) const {
  if (count_elements > 0 && stored_in_place()) {
    for (uint32_t i = 0; i < count_elements; i++) {
      if (e == PackedArray_get<pair_type>(&array.raw_data[0], i, max_in_place)
                   .index()) {
        return PackedArray_get<pair_type>(&array.raw_data[0], i, max_in_place)
            .value();
      }
    }
    return false;
  }
  uint32_t index = search(e);
  if (index < N() &&
      PackedArray_get<pair_type>(array.p_data, index, N()).index() == e) {
    return PackedArray_get<pair_type>(array.p_data, index, N()).value();
  }
  return 0;
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
bool PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::has(
    el_t e) const {
  if (e == 0) {
    return has_0;
  }
  if (count_elements == 0) {
    return false;
  }
  return has_no_zero_non_empty(e);
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
value_type PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::value(
    el_t e) const {
  if constexpr (binary) {
    return has(e);
  } else {
    if (e == 0 && has_0) {
      return val_of_0;
    }
    if (count_elements == 0) {
      return 0;
    }
    return value_no_zero_non_empty(e);
  }
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::PMA()
    : has_0(false), real_logN(4), count_elements(0), b_spot(index_size) {}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::PMA(
    const PMA<index_size, value_type, num_raw_blocks, num_dense_blocks> &source)
    : b_spot(index_size) {
  if (source.stored_in_place()) {
    shallow_copy(&source);
    return;
  }
  // b = source.b;
  real_logN = source.real_logN;
  has_0 = source.has_0;
  count_elements = source.count_elements;
  if (stored_dense()) {
    uint32_t n = 1UL << bsr_word(source.count_elements);
    if (n <= source.count_elements) {
      n *= 2;
    }
    array.p_data = create_PackedArray<pair_type>(n);
    PackedArray_memcpy_aligned<pair_type>(array.p_data, source.array.p_data, n,
                                          N(), N());
  } else {
    array.p_data = create_PackedArray<pair_type>(N());
    PackedArray_memcpy_aligned<pair_type>(array.p_data, source.array.p_data,
                                          N(), N(), N());
  }
}

template <int index_size, typename value_type, int num_raw_blocks,
          int num_dense_blocks>
PMA<index_size, value_type, num_raw_blocks, num_dense_blocks>::~PMA<
    index_size, value_type, num_raw_blocks, num_dense_blocks>() {
  if (!stored_in_place()) {
    free(array.p_data);
  }
}