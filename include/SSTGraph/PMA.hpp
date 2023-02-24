
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <type_traits>

#include "SSTGraph/internal/SizedInt.hpp"
#include "SSTGraph/internal/helpers.hpp"
#include "StructOfArrays/soa.hpp"

namespace SSTGraph {

namespace PMA_helpers {
constexpr std::array<std::array<double, 32>, 32>
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

constexpr std::array<std::array<double, 32>, 32>
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
        double bound = 1.0 / (1U << bsr_word_constexpr(real_logN));
        if (lower <= bound) {
          lower = bound + .001;
        }
        res[depth][real_logN] = lower;
      } else {
        res[depth][real_logN] = 0;
      }
    }
  }
  return res;
}

constexpr std::array<std::array<double, 32>, 32> upper_density_bound =
    get_upper_density_bound_table();

constexpr std::array<std::array<double, 32>, 32> lower_density_bound =
    get_lower_density_bound_table();

static inline bool haszerobyte(uint64_t v) {
  return (((v)-0x0101010101010101UL) & ~(v)&0x8080808080808080UL);
}
static inline bool hasvaluebyte(uint64_t x, uint8_t n) {
  return (haszerobyte((x) ^ (~0UL / 255 * (n))));
}
} // namespace PMA_helpers

template <typename key_type, typename... Ts>
class __attribute__((__packed__)) PMA {

  static constexpr int num_raw_bytes = 8;
  static constexpr int num_dense_bytes = 4096;
  // static constexpr int num_dense_bytes = 64;

  static constexpr bool get_if_binary() {
    if constexpr (sizeof...(Ts) == 0) {
      return true;
    } else {
      using FirstType = typename std::tuple_element<0, std::tuple<Ts...>>::type;
      return (sizeof...(Ts) == 1 && std::is_same_v<bool, FirstType>);
    }
  }

  static_assert(std::is_same<key_type, sized_uint<sizeof(key_type)>>::value);

  static constexpr bool binary = get_if_binary();

  using element_type =
      typename std::conditional<binary, std::tuple<key_type>,
                                std::tuple<key_type, Ts...>>::type;

  using value_type =
      typename std::conditional<binary, std::tuple<>, std::tuple<Ts...>>::type;

  template <int I>
  using NthType = typename std::tuple_element<I, value_type>::type;
  static constexpr int num_types = std::tuple_size_v<element_type>;

  using SOA_type = typename std::conditional<binary, SOA<key_type>,
                                             SOA<key_type, Ts...>>::type;

  static constexpr int get_num_raw_items() {
    int num_raw_items = 0;
    while (SOA_type::get_size_static(num_raw_items + 1) <= num_raw_bytes) {
      num_raw_items += 1;
    }
    return num_raw_items;
  }

  static constexpr int get_num_dense_items() {
    int num_dense_items = 0;
    while (SOA_type::get_size_static(num_dense_items + 1) <= num_dense_bytes) {
      num_dense_items += 1;
    }
    return num_dense_items;
  }

  static constexpr int num_raw_items = get_num_raw_items();
  static constexpr int num_dense_items = get_num_dense_items();

  static_assert(sizeof(key_type) >= 1 && sizeof(key_type) <= 8,
                "for now index size needs to be between 1 and 8 bytes");

  static_assert(num_raw_items < num_dense_items,
                "num_raw_items must be less than num_dense_items");
  static_assert(num_raw_bytes == 8,
                "num_raw_bytes has only been tested with 8");

  static constexpr key_type NULL_VAL = {};

  template <class T> class unaligned_pointer {
    std::array<uint8_t, sizeof(T *)> data;

  public:
    constexpr operator T *() const {
      T *x = 0;
      std::memcpy(&x, data.data(), sizeof(T *));
      return x;
    }
    unaligned_pointer &operator=(T *other) noexcept {
      std::memcpy(data.data(), &other, sizeof(T *));
      return *this;
    }
  };

  using array_union = union __attribute__((__packed__)) array_union_ {
    unaligned_pointer<void> p_data;
    uint8_t raw_data[SOA_type::get_size_static(num_raw_items)] = {};
  };

  class empty_type {};

private:
  array_union array;

  uint8_t real_logN;            // max value 32
  uint32_t count_elements : 21; // we know by 524288 we will use 16 bit elements
  // this needs to be at the end
  uint8_t b_spot : 3; // B will go here at the tinyset level

  [[nodiscard]] inline bool stored_dense() const {
    return count_elements < num_dense_items;
  }
  [[nodiscard]] inline bool
  would_be_stored_dense(uint32_t element_count) const {
    return element_count < num_dense_items;
  }
  [[nodiscard]] inline bool stored_in_place() const {
    return count_elements <= num_raw_items;
  }

  [[nodiscard]] inline bool
  would_be_stored_in_place(uint32_t element_count) const {
    return element_count <= num_raw_items;
  }

  [[nodiscard]] uint32_t N() const { return 1U << real_logN; }
  [[nodiscard]] uint8_t loglogN() const { return bsr_word(real_logN); }
  [[nodiscard]] uint32_t logN() const { return (1U << loglogN()); }
  [[nodiscard]] uint32_t mask_for_leaf() const { return ~(logN() - 1); }
  [[nodiscard]] uint32_t H() const { return bsr_word(N() / logN()); }

  inline key_type get_key_array(uint32_t index) const {
    return std::get<0>(
        SOA_type::template get_static<0>(array.p_data, N(), index));
  }

  void double_list();
  void half_list();
  void slide_right(uint32_t index);
  void slide_left(uint32_t index);
  void redistribute(uint32_t index, uint32_t len);

  [[nodiscard]] double get_density(uint32_t index, uint32_t len) const;
  [[nodiscard]] uint32_t get_density_count(uint32_t index, uint32_t len) const;

  [[nodiscard]] uint32_t search(key_type e) const;
  [[nodiscard]] uint32_t search_no_nulls_branchless(key_type e) const;
  [[nodiscard]] uint32_t
  search_no_nulls_counting(key_type e, uint32_t start = 0,
                           uint32_t end = UINT32_MAX) const;

  void place(uint32_t index, element_type e);
  void take(uint32_t index);

  [[nodiscard]] uint32_t find_prev_valid(uint32_t start) const {
    while (get_key_array(start) == NULL_VAL && start > 0) {
      start--;
    }
    return start;
  }

  [[nodiscard]] bool check_no_full_leaves(uint32_t index, uint32_t len) const;

  // given index, return the starting index of the leaf it is in
  [[nodiscard]] uint32_t find_leaf(uint32_t index) const {
    return index & mask_for_leaf();
  }
  [[nodiscard]] uint32_t next_leaf(uint32_t index) const {
    return ((index >> loglogN()) + 1) << (loglogN());
  }
  [[nodiscard]] uint32_t next_leaf(uint32_t index, uint32_t llN) const {
    return ((index >> llN) + 1) << (llN);
  }

  void print_array(uint32_t start = 0, uint32_t end = UINT32_MAX) const;

public:
  static constexpr void print_details() {
    std::cout << "num_raw_bytes = " << num_raw_bytes << ", ";
    std::cout << "num_dense_bytes = " << num_dense_bytes << ", ";
    std::cout << "num_raw_items = " << num_raw_items << ", ";
    std::cout << "num_dense_items = " << num_dense_items << ", ";
    std::cout << "binary = " << binary << "\n";
    std::cout << "element_type = " << TypeName<element_type>() << "\n";
    std::cout << "value_type = " << TypeName<value_type>() << "\n";
    std::cout << "sizeof(array_union) = " << sizeof(array_union) << ", ";
    std::cout << "sizeof(PMA) = " << sizeof(PMA) << "\n\n";
  }
  [[nodiscard]] inline bool stored_uncompressed() const {
    if (stored_in_place()) {
      return false;
    }
    if constexpr (binary) {
      return 8 * sizeof(key_type) * N() >= (1UL << (sizeof(key_type) * 8U));
    } else {
      return SOA_type::get_size_static(N()) >=
             sizeof(value_type) * (1UL << (sizeof(key_type) * 8U));
    }
  }
  void clean_no_free() {
    count_elements = 0;
    // b = 0;
    real_logN = 0;
  }
  void prefetch_data() const { __builtin_prefetch(array.p_data, 0, 3); }

  void shallow_copy(const PMA *source) {
    memcpy(static_cast<void *>(this), source, sizeof(PMA));
  }
  PMA();
  PMA(const PMA &source);
  ~PMA();
  void print_pma(uint32_t prefix = 0) const;
  [[nodiscard]] bool has(key_type e) const;
  value_type value(key_type e) const;
  bool insert(element_type e);
  bool remove(key_type e);
  [[nodiscard]] uint64_t get_size() const;
  [[nodiscard]] uint64_t get_n() const { return count_elements; }
  [[nodiscard]] uint64_t sum_keys() const;
  [[nodiscard]] bool is_empty() const { return get_n() == 0; }
  [[nodiscard]] bool use_fast_iter() const { return stored_dense(); }

  template <bool no_early_exit, size_t... Is, class F>
  bool map(F f, uint64_t prefix = 0) const;

  template <bool dense = false> class iterator {

    uint64_t length;
    const void *array;
    uint64_t place;
    uint8_t loglogN;
    uint32_t n;

  public:
    explicit iterator(uint64_t pl)
        : length(0), array(nullptr), place(pl), loglogN(0), n(0) {}

    explicit iterator(const PMA &pma)
        : length(pma.stored_dense() ? pma.count_elements : pma.N()),
          array(pma.stored_in_place() ? pma.array.raw_data : pma.array.p_data),
          place(0), loglogN(pma.loglogN()),
          n(pma.stored_in_place() ? PMA::num_raw_items : pma.N()) {}

    bool operator==(const iterator &other) const {
      return (place == other.place);
    }
    bool operator!=(const iterator &other) const {
      return (place != other.place);
    }
    bool operator<(const iterator &other) const {
      return (place < other.place);
    }
    bool operator>=(const iterator &other) const {
      return (place >= other.place);
    }
    iterator &operator++() {
      place += 1;
      if constexpr (!dense) {
        while ((place < length) &&
               (std::get<0>(SOA_type::template get_static<0>(
                    array, n, place)) == NULL_VAL)) {
          place = ((place >> loglogN) + 1) << (loglogN);
        }
      }
      return *this;
    }
    element_type operator*() const {
      return SOA_type::get_static(array, n, place);
    }
    ~iterator() = default;
  };
  template <bool dense = false> iterator<dense> begin() const {
    return iterator<dense>(*this);
  }
  template <bool dense = false> iterator<dense> end() const {
    if constexpr (dense) {
      uint64_t length = count_elements;
      return iterator<dense>(length);
    } else {
      uint64_t length = N();
      if (stored_dense()) {
        length = count_elements;
      }
      return iterator<dense>(length);
    }
  }
};

template <typename key_type, typename... Ts>
uint64_t PMA<key_type, Ts...>::sum_keys() const {
  uint64_t result = 0;
  map<true>([&](key_type key) { result += key; });
  return result;
}

template <typename key_type, typename... Ts>
template <bool no_early_exit, size_t... Is, class F>
bool PMA<key_type, Ts...>::map(F f, uint64_t prefix) const {

  static_assert(std::is_invocable_v<decltype(&F::operator()), F &, uint32_t,
                                    NthType<Is>...>,
                "update function must match given types");
  auto f_add_prefix = [f, prefix](uint32_t key, auto... args) {
    return f(key + prefix, args...);
  };

  if (__builtin_expect(stored_dense(), 1)) { // also deals with in place
    const void *const data =
        (stored_in_place()) ? array.raw_data : array.p_data;
    uint32_t n = N();
    if (stored_in_place()) {
      n = num_raw_items;
    }

    for (uint32_t i = 0; i < count_elements; i++) {
      auto element = SOA_type::template get_static<0, (Is + 1)...>(data, n, i);
      if constexpr (no_early_exit) {
        std::apply(f_add_prefix, element);
      } else {
        if (std::apply(f_add_prefix, element)) {
          return true;
        }
      }
    }
    return false;
  }
  // not stored dense
  {
    auto element =
        SOA_type::template get_static<0, (Is + 1)...>(array.p_data, N(), 0);
    if constexpr (no_early_exit) {
      std::apply(f_add_prefix, element);
    } else {
      if (std::apply(f_add_prefix, element)) {
        return true;
      }
    }
  }
  for (uint32_t i = 1; i < N(); i++) {
    auto index = get_key_array(i);
    if (index != NULL_VAL) {
      auto element =
          SOA_type::template get_static<0, (Is + 1)...>(array.p_data, N(), i);
      if constexpr (no_early_exit) {
        std::apply(f_add_prefix, element);
      } else {
        if (std::apply(f_add_prefix, element)) {
          return true;
        }
      }
    }
  }
  return false;
}

template <typename key_type, typename... Ts>
uint64_t PMA<key_type, Ts...>::get_size() const {
  uint64_t size = 0;
  if (!stored_in_place()) {
    size += SOA_type::get_size_static(N());
  }
  return size + sizeof(PMA<key_type, Ts...>);
}

template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::print_array(uint32_t start, uint32_t end) const {
  if (end > N()) {
    end = N();
  }
  printf("N = %u, logN = %u, loglogN = %u, H = %u\n", N(), logN(), loglogN(),
         H());
  printf("count_elements %u, b = %lu\n", count_elements, sizeof(key_type));
  if (end - start > 500) {
    printf("too big to print\n");
    return;
  }
  // the first element can look like null while being 0
  std::apply(
      [](key_type key, auto... args) {
        if constexpr (binary) {
          std::cout << key << ", ";
        } else {
          std::cout << "(" << key << ", ";
          ((std::cout << ", " << args), ...);
          std::cout << "), ";
        }
      },
      SOA_type::get_static(array.p_data, N(), 0));
  SOA_type::map_range_static(
      array.p_data, N(),
      [](key_type key, auto... args) {
        if (key != NULL_VAL) {
          if constexpr (binary) {
            std::cout << key << ", ";
          } else {
            std::cout << "(" << key << ", ";
            ((std::cout << ", " << args), ...);
            std::cout << "), ";
          }
        }
      },
      1);
  printf("\n");
}

template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::print_pma(uint32_t prefix) const {
  printf("PMA: prefix = %u, count_elements %u\n", prefix, count_elements);
  printf("num raw items = %d, num_dense_items = %d\n", num_raw_items,
         num_dense_items);
  SOA_type::print_type_details();
  if (count_elements == 0) {
    printf("the pma is empty\n");
  } else if (stored_in_place()) {
    printf("the pma is only storing data in place\n");
    printf("the elements are:\n");

    SOA_type::map_range_static((void *)&array.raw_data[0], num_raw_items,
                               [prefix](key_type key, auto... args) {
                                 if constexpr (binary) {
                                   std::cout << key + prefix << ", ";
                                 } else {
                                   std::cout << "(" << key + prefix << ", ";
                                   ((std::cout << ", " << args), ...);
                                   std::cout << "), ";
                                 }
                               },
                               0, count_elements);
    printf("\n");
  } else {
    printf("N = %u, logN = %u, loglogN = %u, H = %u\n", N(), logN(), loglogN(),
           H());
    for (uint32_t i = 0; i < N(); i += logN()) {
      SOA_type::map_range_with_index_static(
          array.p_data, N(),
          [prefix](size_t index, key_type key, auto... args) {
            if (key != NULL_VAL || index == 0) {
              if constexpr (binary) {
                std::cout << key + prefix << ", ";
              } else {
                std::cout << "(" << key + prefix << ", ";
                ((std::cout << ", " << args), ...);
                std::cout << "), ";
              }
            } else {
              std::cout << "_" << index << "_,";
            }
          },
          i, i + logN());
      printf("\n");
    }
    printf("\n");
  }
}

template <typename key_type, typename... Ts>
uint32_t PMA<key_type, Ts...>::get_density_count(uint32_t index,
                                                 uint32_t len) const {
  uint32_t full = 0;
  // the later check will miss when the first element is 0
  if (index == 0 && count_elements > 0 && get_key_array(0) == NULL_VAL) {
    full += 1;
  }
  for (uint32_t i = index; i < index + len; i += 4) {
    // TODO(wheatman) this can be made faster
    uint32_t add =
        (get_key_array(i) != NULL_VAL) + (get_key_array(i + 1) != NULL_VAL) +
        (get_key_array(i + 2) != NULL_VAL) + (get_key_array(i + 3) != NULL_VAL);
    full += add;
  }
  return full;
}

template <typename key_type, typename... Ts>
double PMA<key_type, Ts...>::get_density(uint32_t index, uint32_t len) const {
  uint32_t full = get_density_count(index, len);
  double full_d = (double)full;
  return full_d / len;
}

template <typename key_type, typename... Ts>
bool PMA<key_type, Ts...>::check_no_full_leaves(uint32_t index,
                                                uint32_t len) const {
  for (uint32_t i = index; i < index + len; i += logN()) {
    bool full = true;
    for (uint32_t j = i; j < i + logN(); j++) {
      if (get_key_array(j) == NULL_VAL && j != 0) {
        full = false;
      }
    }
    if (full) {
      return false;
    }
  }
  return true;
}

// Evenly redistribute elements in the ofm, given a range to look into
// index: starting position in ofm structure
// len: area to redistribute
// should already be locked
template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::redistribute(uint32_t index, uint32_t len) {
  // TODO(wheatman) if its small use the stack
  // for small cases put on the stack
  if (len == logN()) {
    return;
  }
  uint32_t j = 0;

  SOA_type space = SOA_type(len);
  uint32_t end = index + len;

  // could get better cache behavior if I go back and forth with reading
  // and writing
  int done_first = 0;
  if (index == 0) {
    auto item = SOA_type::get_static(array.p_data, N(), 0);
    space.get(0) = item;
    done_first = 1;
    j += 1;
  }
  for (uint32_t i = index + done_first; i < end; i++) {
    auto item = SOA_type::get_static(array.p_data, N(), i);
    space.get(j) = item;
    // counting non-null
    j += (std::get<0>(item) != NULL_VAL);
  }
  SOA_type::map_range_static(
      array.p_data, N(),
      [](auto &...args) { std::forward_as_tuple(args...) = element_type(); },
      index, end);

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
      SOA_type::get_static(array.p_data, N(), k) = space.get(j2);
      j2++;
    }
  }
}

template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::double_list() {
  void *old_array = array.p_data;
  array.p_data = SOA_type::resize_static(old_array, N(), N() * 2);
  free(old_array);
  real_logN += 1;
  redistribute(0, N());
}

template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::half_list() {
  void *new_array = std::malloc(SOA_type::get_size_static(N() / 2));
  uint32_t index = 0;
  {
    auto elem = SOA_type::get_static(array.p_data, N(), 0);
    SOA_type::get_static(new_array, N() / 2, 0) = elem;
    index++;
  }
  for (uint64_t i = 1; i < N(); i++) {
    auto elem = SOA_type::get_static(array.p_data, N(), i);
    if (std::get<0>(elem) != NULL_VAL) {
      SOA_type::get_static(new_array, N() / 2, index) = elem;
      index++;
    }
  }

  free(array.p_data);
  array.p_data = new_array;
  real_logN -= 1;

  SOA_type::map_range_static(
      array.p_data, N(),
      [](auto &...args) { std::forward_as_tuple(args...) = element_type(); },
      index, N());

  redistribute(0, N());
}

template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::slide_right(uint32_t index) {
  for (uint64_t i = next_leaf(index) - 1; i > index; i--) {
    SOA_type::get_static(array.p_data, N(), i) =
        SOA_type::get_static(array.p_data, N(), i - 1);
  }
  SOA_type::get_static(array.p_data, N(), index) = element_type();
}

template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::slide_left(uint32_t index) {
  for (uint64_t i = index; i < next_leaf(index) - 1; i++) {
    SOA_type::get_static(array.p_data, N(), i) =
        SOA_type::get_static(array.p_data, N(), i + 1);
  }
  SOA_type::get_static(array.p_data, N(), next_leaf(index) - 1) =
      element_type();
}

// algorithm taken from
// https://dirtyhandscoding.wordpress.com/2017/08/25/performance-comparison-linear-search-vs-binary-search/
template <typename key_type, typename... Ts>
uint32_t PMA<key_type, Ts...>::search_no_nulls_counting(key_type e,
                                                        uint32_t start,
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

    return (get_key_array(start) < e);
  case 2:
    cnt += (get_key_array(start) < e);
    cnt += (get_key_array(start + 1) < e);
    return cnt;
  case 3:
    cnt += (get_key_array(start) < e);
    cnt += (get_key_array(start + 1) < e);
    cnt += (get_key_array(start + 2) < e);
    return cnt;
  default:
    /* TODO(wheatman) redo hand vectorized code
  #if defined(__SSE4_2__)
      if constexpr (index_size == 3 && binary) {
        // sees to be slower for now
  #if defined(__AVX512VL__) && defined(__AVX512F__) && 0
        if (len > 8) {
          __m256i shuf_long = _mm256_setr_epi8(
              0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1, 16, 17, 18,
              -1, 19, 20, 21, -2, 22, 23, 24, -1, 25, 26, 27, -1);
          __m256i shuf_short = _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 6);
          __m256i cnt_vec = _mm256_setzero_si256();
          __m256i e_vec = _mm256_set1_epi32(e);
          uint32_t i = start * 3;
          for (; i < end * 3 - 23; i += 24) {
            __m256i data_vector =
                _mm256_loadu_si256((__m256i *)(((uint8_t *)array.p_data) + i));
            // Log<uint8_t>(data_vector);
            __m256i el_vector = _mm256_maskz_shuffle_epi8(
                -1, _mm256_permutexvar_epi32(shuf_short, data_vector),
  shuf_long);
            // Log<uint32_t>(el_vector);
            cnt_vec =
                _mm256_sub_epi32(cnt_vec, _mm256_cmpgt_epi32(e_vec, el_vector));
          }
          cnt += sum_256<uint32_t, __m256i>(cnt_vec);
          i /= 3;
          // std::cout << cnt << " from vector, i = " << i << std::endl;
          for (; i < end; i++) {
            cnt += (PackedArray_get<pair_type>(array.p_data, i, N()).index() <
  e);
          }
          // std::cout << cnt << " after cleanup" << std::endl;
          if (cnt > 0 &&
              PackedArray_get<pair_type>(array.p_data, cnt - 1, N()).index() ==
                  e) {
            cnt -= 1;
          }
  #if 0
          uint32_t correct_cnt = start;
          for (uint32_t i = start; i < end; i++) {
            correct_cnt +=
                (PackedArray_get<pair_type>(array.p_data, i, N()).index() < e);
          }
          if (cnt != correct_cnt) {
            printf("issue, got %u, expected %u, el_count = %u\n", cnt,
                   correct_cnt, count_elements);
            printf("looking for %u\n", e);
            print_pma();
          } else {
            // printf("good count\n");
          }
  #endif
          return cnt;
        }
  #endif
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
        // uint32_t correct_cnt = _mm_extract_epi32(cnt_vec, 0) +
        // _mm_extract_epi32(cnt_vec, 1) + _mm_extract_epi32(cnt_vec, 2) +
        // _mm_extract_epi32(cnt_vec, 3);
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
  #if 0
        uint32_t correct_cnt = start;
        for (uint32_t i = start; i < end; i++) {
          correct_cnt +=
              (PackedArray_get<pair_type>(array.p_data, i, N()).index() < e);
        }
        if (cnt != correct_cnt) {
          printf("issue, got %u, expected %u, el_count = %u, looking for %u\n",
                 cnt, correct_cnt, count_elements, e);
          std::cout << start << ", " << end << std::endl;
          print_pma();
        } else {
          // printf("good count\n");
        }
  #endif
        return cnt;
      }
  //     if constexpr (index_size == 2 && binary) {
  //       __m128i cnt_vec = _mm_setzero_si128();
  //       __m128i e_vec = _mm_set1_epi32(e);
  //       for (uint32_t i = 0; i < count_elements / 4 + (count_elements % 4 >
  0);
  //            i++) {
  //         __m64 el_vector_i = *((__m64 *)(array.p_data + i));
  //         __m128i el_vector =
  //             _mm_cvtepu16_epi32(_mm_set_epi64(_mm_setzero_si64(),
  //             el_vector_i));
  //         cnt_vec = _mm_sub_epi32(cnt_vec, _mm_cmplt_epi32(el_vector,
  e_vec));
  //       }
  //       uint64_t bottom = _mm_extract_epi64(cnt_vec, 0);
  //       uint64_t top = _mm_extract_epi64(cnt_vec, 1);
  //       // we know these can be at most 16 bits so we have no worry of
  //       // overflow
  //       uint64_t tmp = bottom + top;
  //       tmp += (tmp >> 32U);
  //       cnt += (uint32_t)tmp;
  //       cnt -= offset[count_elements % 4];
  // #if 0
  //         uint32_t correct_cnt = 0;
  //         for (uint32_t i = 0; i < count_elements; i++) {
  //           correct_cnt +=
  //               (PackedArray_get<pair_type>(array.p_data, i, N()).index() <
  e);
  //         }
  //         if (cnt != correct_cnt) {
  //           printf("issue, got %u, expected %u, el_count = %u\n", cnt,
  //                  correct_cnt, count_elements);
  //         } else {
  //           // printf("good count\n");
  //         }
  // #endif
  //       return cnt;
  //     }
  #endif
  */
    uint32_t i = start;
    for (; i < end - 3; i += 4) {
      cnt += (get_key_array(i) < e);
      cnt += (get_key_array(i + 1) < e);
      cnt += (get_key_array(i + 2) < e);
      cnt += (get_key_array(i + 3) < e);
    }
    for (; i < end; i++) {
      cnt += (get_key_array(i) < e);
    }
  }
  return cnt;
}

// needs to be larger than the smallest element
template <typename key_type, typename... Ts>
uint32_t PMA<key_type, Ts...>::search_no_nulls_branchless(key_type e) const {
  static constexpr uint32_t cutoff = 64;
  if (count_elements <= cutoff) {
    return search_no_nulls_counting(e, 0, count_elements);
  }
  uint32_t pos = -1;
  uint32_t t = count_elements;
  uint32_t logstep = bsr_word(t);
  uint32_t first_step = t + 1 - (1U << logstep);
  uint32_t step = 1U << logstep;
  // std::cout << pos << ", " << t << ", " << logstep << ", " << first_step <<
  // ", "
  //           << step << std::endl;
  pos = (get_key_array(pos + step) < e ? pos + first_step : pos);
  step >>= 1U;
  while (step >= cutoff) {
    pos = (get_key_array(pos + step) < e ? pos + step : pos);
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
template <typename key_type, typename... Ts>
uint32_t PMA<key_type, Ts...>::search(key_type e) const {
  if (e <= get_key_array(0)) {
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
    key_type item = get_key_array(mid);

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
      if (get_key_array(check) == NULL_VAL) {
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
      key_type item = get_key_array(check);
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
      // cleanup before return
      return mid;
    }
  }
  if (t < s) {
    s = t;
  }

  // trying to encourage the packed left property so if they are both null
  // go to the left
  if ((get_key_array(s) == NULL_VAL) && (get_key_array(t) == NULL_VAL)) {
    t = s;
  }

  // handling the case where there is one element left
  // if you are leq, return start (index where elt is)
  // otherwise, return end (no element greater than you in the range)
  if (e <= get_key_array(s) && (get_key_array(s) != NULL_VAL)) {
    t = s;
  }
  // really insure packed left
  // TODO(wheatman) we shouldn't need this
  while (!((find_leaf(t) == t) || (get_key_array(t - 1) != NULL_VAL))) {
    t--;
  }
  if (t == 0 && e > get_key_array(0)) {
    t = 1;
  }
  return t;
}

// insert elem at index
template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::place(uint32_t index, element_type e) {

  uint32_t level = H();
  uint32_t len = logN();

  // always deposit on the left
  if (get_key_array(index) != NULL_VAL) {
    slide_right(index);
  }

  SOA_type::get_static(array.p_data, N(), index) = e;

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
  while (density >= PMA_helpers::upper_density_bound[level][real_logN]) {
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
  assert(density < PMA_helpers::upper_density_bound[level][real_logN]);
  assert(density <= (((double)logN() - 1) / logN()));
  if (len > logN()) {
    redistribute(node_index, len);
  }
  assert(check_no_full_leaves(node_index, len));
}

template <typename key_type, typename... Ts>
bool PMA<key_type, Ts...>::insert(element_type e) {
  const key_type key = std::get<0>(e);
  if (stored_in_place()) {
    // printf("storing in place\n");
    uint32_t first_greater = 0;
    for (uint32_t i = 0; i < count_elements; i++) {
      element_type item =
          SOA_type::get_static(&array.raw_data[0], num_raw_items, i);
      if (std::get<0>(item) == key) {
        // update the value if its different
        if constexpr (!binary) {
          if (leftshift_tuple(item) != leftshift_tuple(e)) {
            SOA_type::get_static(&array.raw_data[0], num_raw_items, i) = e;
          }
        }
        assert(has(std::get<0>(e)));
        return false;
      } else if (key > std::get<0>(item)) {
        first_greater++;
      }
    }
    if (would_be_stored_in_place(count_elements + 1)) {
      // printf("insert in place first_greater = %u, count_elements = %u\n",
      //        first_greater, count_elements);
      if (first_greater == count_elements) {
        SOA_type::get_static(&array.raw_data[0], num_raw_items,
                             count_elements) = e;
      } else {

        for (uint64_t i = count_elements; i > first_greater; i--) {
          SOA_type::get_static(&array.raw_data[0], num_raw_items, i) =
              SOA_type::get_static(&array.raw_data[0], num_raw_items, i - 1);
        }
        SOA_type::get_static(&array.raw_data[0], num_raw_items, first_greater) =
            e;
      }
      count_elements += 1;
      assert(has(std::get<0>(e)));
      return true;
    } else if (stored_in_place()) {
      // printf("switching from in place to dense\n");
      // raw is full, insert all the elements and switch over
      array_union raw_data_saved;
      memcpy(&raw_data_saved, array.raw_data, sizeof(array_union));
      while (count_elements > N()) {
        real_logN += 1;
      }
      array.p_data = SOA_type::resize_static(&raw_data_saved.raw_data[0],
                                             num_raw_items, N());
      // print_array();
    }
  }
  uint32_t index = search(key);
  // if its already in the array
  if (index < N() && get_key_array(index) == key) {
    // update its value if needed
    if constexpr (!binary) {
      if (leftshift_tuple(SOA_type::get_static(array.p_data, N(), index)) !=
          leftshift_tuple(e)) {
        SOA_type::get_static(array.p_data, N(), index) = e;
      }
    }
    if (count_elements == 0) {
      count_elements = 1;
      assert(has(std::get<0>(e)));
      return true;
    }
    assert(has(std::get<0>(e)));
    return false;
  }
  count_elements += 1;
  if (stored_dense()) {
    // printf("inserting into dense\n");
    // TODO(wheatman) could combine resize and insert
    void *old_array = array.p_data;
    if (count_elements > N()) {
      // printf("double the size of dense\n");
      array.p_data = SOA_type::resize_static(old_array, N(), 2 * N());
      real_logN += 1;
      free(old_array);
    }
    for (uint64_t i = count_elements - 1; i > index; i--) {
      SOA_type::get_static(array.p_data, N(), i) =
          SOA_type::get_static(array.p_data, N(), i - 1);
    }

    SOA_type::get_static(array.p_data, N(), index) = e;
  } else {
    if (would_be_stored_dense(count_elements - 1)) {
      // printf("switching from dense to pma\n");
      double_list();
      index = search(key);
    }
    // printf("inserting into pma\n");
    place(index, e);
  }
  assert(has(std::get<0>(e)));
  return true;
}

// remove elem at index
template <typename key_type, typename... Ts>
void PMA<key_type, Ts...>::take(uint32_t index) {
  uint32_t level = H();
  uint32_t len = logN();

  slide_left(index);

  uint32_t node_index = find_leaf(index);

  // get density of the leaf you are in
  uint32_t density_count = get_density_count(node_index, len);
  double density = ((double)density_count) / len;

  // while density too high, go up the implicit tree
  // go up to the biggest node above the density bound
  while (density <= PMA_helpers::lower_density_bound[level][real_logN]) {
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

template <typename key_type, typename... Ts>
bool PMA<key_type, Ts...>::remove(key_type e) {
  if (!has(e)) {
    return false;
  }
  if (stored_in_place()) {
    // printf("removing element from in place\n");
    for (uint32_t i = 0; i < count_elements; i++) {
      if (std::get<0>(SOA_type::template get_static<0>(
              &array.raw_data[0], num_raw_items, i)) == e) {
        // printf("found the elment in position %u\n", i);

        for (uint64_t j = i; j < count_elements; j++) {
          SOA_type::get_static(&array.raw_data[0], num_raw_items, j) =
              SOA_type::get_static(&array.raw_data[0], num_raw_items, j + 1);
        }
        SOA_type::get_static(&array.raw_data[0], num_raw_items,
                             count_elements - 1) = element_type();
        count_elements -= 1;
        assert(!has(e));
        return true;
      }
    }
    // printf("should never happen\n");
  } else if (stored_dense()) {
    // printf("remove from dense\n");
    // TODO(wheatman) could be speed up by using overestimate and going
    // backwords
    for (uint32_t i = 0; i < count_elements; i++) {
      key_type item = get_key_array(i);
      if (item == e) {
        // TODO(wheatman) could combine resize and slide left
        void *old_array = array.p_data;

        for (uint64_t j = i; j < count_elements - 1; j++) {
          SOA_type::get_static(old_array, N(), j) =
              SOA_type::get_static(old_array, N(), j + 1);
        }
        SOA_type::get_static(old_array, N(), count_elements - 1) =
            element_type();

        count_elements -= 1;

        if (count_elements <= N() / 2) {
          array.p_data = SOA_type::resize_static(old_array, N(), N() / 2);
          free(old_array);
          real_logN -= 1;
        }
        break;
      }
    }
    // if we should now be in place due to getting smaller
    if (stored_in_place()) {
      // printf("going back to in place\n");
      void *old_array = array.p_data;
      for (uint32_t i = 0; i < count_elements; i++) {
        SOA_type::get_static(&array.raw_data[0], num_raw_items, i) =
            SOA_type::get_static(old_array, N(), i);
      }
      free(old_array);
      assert(!has(e));
      return true;
    } else {
      assert(!has(e));
      return true;
    }
  }
  // if I am going to be moving back to a dense array do the remove and the
  // compress in one pass
  count_elements -= 1;
  if (stored_dense()) {
    // printf("going back to dense, count element = %u\n", count_elements);
    // print_pma();
    if (count_elements <= N() / 2) {
      // if we also need to half the array size
      uint64_t old_N = N();
      uint64_t new_logN = bsr_word(count_elements);
      if (1U << new_logN < count_elements) {
        new_logN += 1;
      }
      real_logN = new_logN;

      void *new_array = std::malloc(SOA_type::get_size_static(N()));
      // printf("N() = %u\n", N());

      uint32_t index = 0;
      {
        element_type element = SOA_type::get_static(array.p_data, old_N, 0);
        if (std::get<0>(element) != e) {
          // printf("index = %u\n", index);
          SOA_type::get_static(new_array, N(), index) = element;
          index++;
        }
      }
      for (uint64_t i = 1; i < old_N; i++) {
        element_type element = SOA_type::get_static(array.p_data, old_N, i);
        if (std::get<0>(element) != e && std::get<0>(element) != NULL_VAL) {
          // printf("index = %u\n", index);
          SOA_type::get_static(new_array, N(), index) = element;
          index++;
        }
      }
      for (uint64_t i = index; i < N(); i++) {
        SOA_type::get_static(new_array, N(), i) = element_type();
      }
      free(array.p_data);
      array.p_data = new_array;
    } else {
      // if we don't need to half the array we just compress the elements in one
      // pass
      uint32_t index = 0;
      {
        element_type element = SOA_type::get_static(array.p_data, N(), 0);
        if (std::get<0>(element) != e) {
          SOA_type::get_static(array.p_data, N(), index) = element;
          index++;
        }
      }
      for (uint64_t i = 1; i < N(); i++) {
        element_type element = SOA_type::get_static(array.p_data, N(), i);
        if (std::get<0>(element) != e && std::get<0>(element) != NULL_VAL) {
          SOA_type::get_static(array.p_data, N(), index) = element;
          index++;
        }
      }
      for (uint64_t i = index; i < N(); i++) {
        SOA_type::get_static(array.p_data, N(), i) = element_type();
      }
    }
    assert(!has(e));
    return true;
  }
  // printf("remove from pma\n");
  count_elements += 1;
  uint32_t index = search(e);
  take(index);
  count_elements -= 1;
  assert(!has(e));
  return true;
}

template <typename key_type, typename... Ts>
bool PMA<key_type, Ts...>::has(key_type e) const {
  if (count_elements == 0) {
    return false;
  }
  if (stored_in_place()) {
    /* TODO(wheatman) add back this optimization
    if constexpr (index_size == 1 && binary) {
      static_assert(num_raw_blocks * 8 == sizeof(array_union),
                    "this optimization assumes num_raw_blocks * 8 = "
                    "sizeof(array_union)");
      bool ret = false;
      for (int i = 0; i < num_raw_blocks; i++) {
        ret |= PMA_helpers::hasvaluebyte(array.raw_data[i], e);
      }
      return ret;
    }
    */
    for (uint32_t i = 0; i < count_elements; i++) {
      if (e == std::get<0>(SOA_type::template get_static<0>(
                   &array.raw_data[0], num_raw_items, i))) {
        return true;
      }
    }
    return false;
  }
  uint32_t index = search(e);
  return (index < N() && get_key_array(index) == e);
}

template <typename key_type, typename... Ts>
typename PMA<key_type, Ts...>::value_type
PMA<key_type, Ts...>::value(key_type e) const {
  if constexpr (binary) {
    return has(e);
  } else {
    if (count_elements == 0) {
      return {};
    }
    if (stored_in_place()) {
      for (uint32_t i = 0; i < count_elements; i++) {
        element_type element =
            SOA_type::get_static(&array.raw_data[0], num_raw_items, i);
        if (e == std::get<0>(element)) {
          return leftshift_tuple(element);
        }
      }
      return {};
    }
    uint32_t index = search(e);
    if (index < N()) {
      element_type element = SOA_type::get_static(array.p_data, N(), index);
      if (e == std::get<0>(element)) {
        return leftshift_tuple(element);
      }
    }
    return {};
  }
}

template <typename key_type, typename... Ts>
PMA<key_type, Ts...>::PMA()
    : real_logN(4), count_elements(0), b_spot(sizeof(key_type)) {}

template <typename key_type, typename... Ts>
PMA<key_type, Ts...>::~PMA<key_type, Ts...>() {
  if (!stored_in_place()) {
    free(array.p_data);
  }
}
} // namespace SSTGraph
