#include <concepts>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <unordered_set>

#include "cxxopts.hpp"

#include "SSTGraph/PMA.hpp"
#include "SSTGraph/internal/helpers.hpp"

using namespace SSTGraph;

template <uint32_t b>
[[nodiscard]] int pma_update_test_templated(uint32_t el_count,
                                            bool check = false) {
  uint32_t max = 0;
  if constexpr (b == 4) {
    max = UINT32_MAX;
  } else {
    max = 1U << (b * 8U);
  }
  PMA<sized_uint<b>> pma;
  el_count = std::min(max - 1, el_count);

  std::vector<uint32_t> random_numbers = create_random_data(el_count, max);

  std::unordered_set<uint32_t> checker;

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    pma.insert(random_numbers[i]);
    // pma.print_pma();
    if (check) {
      checker.insert(random_numbers[i]);
      if (!pma.has(random_numbers[i])) {
        printf("don't have something we inserted while inserting elements\n");
        return -1;
      }
    }
  }
  for (uint32_t i = 0; i < el_count; i++) {
    if (check) {
      if (!pma.has(random_numbers[i])) {
        printf("don't have something we inserted after inserting elements, "
               "index was %u\n",
               i);
        return -1;
      }
    }
  }
  // pma.print_pma();
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    // printf("trying to remove %u\n", random_numbers[i]);
    pma.remove(random_numbers[i]);
    // pma.print_pma();
    if (check) {
      checker.erase(random_numbers[i]);
      if (pma.has(random_numbers[i])) {
        printf("have something we removed while removing elements, tried to "
               "remove %u\n",
               random_numbers[i]);
        return -1;
      }
      for (auto el : checker) {
        if (!pma.has(el)) {
          printf("we removed %u when we shouldn't have\n", el);
          return -1;
        }
      }
    }
  }
  if (check && pma.get_n() != 0) {
    printf("still have elements when we shouldn't\n");
    return -1;
  }
  uint64_t remove_duration = get_usecs() - start;
  printf("insert duration = %lu, remove duration = %lu\n", insert_duration,
         remove_duration);
  return 0;
}

[[nodiscard]] int pma_update_test(uint32_t el_count, bool check = false) {
  int r = 0;
  r = pma_update_test_templated<1>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_update_test_templated<2>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_update_test_templated<3>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_update_test_templated<4>(el_count, check);
  if (r) {
    return r;
  }
  return 0;
}

template <uint32_t index_size, typename value_type>
[[nodiscard]] int pma_map_test_templated(uint32_t el_count,
                                         bool check = false) {
  uint32_t max_key = std::numeric_limits<uint32_t>::max();
  if constexpr (index_size < 4) {
    max_key = 1UL << (index_size * 8U);
  }
  PMA<sized_uint<index_size>, value_type> pma;
  el_count = std::min(max_key - 1, el_count);
  std::unordered_map<uint32_t, value_type> random_pairs = {};

  auto keys = create_random_data(el_count, max_key);
  auto values = create_random_data<value_type>(el_count);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    uint32_t key = keys[i];
    value_type value = values[i];

    // std::cout << "trying to insert (" << key << ", " << +value << ")"
    //           << std::endl;

    pma.insert({key, value});
    // pma.print_pma();
    if (check) {
      random_pairs.insert_or_assign(key, value);
      if (!pma.has(key)) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               key);
        pma.print_pma();
        return -1;
      }
      if (std::get<0>(pma.value(key)) != value) {
        printf("FAILED: value doesn't match while inserting elements\n");
        std::cout << std::get<0>(pma.value(key)) << ", " << value << std::endl;
        pma.print_pma();
        return -1;
      }
    }
  }
  for (auto &pair : random_pairs) {
    if (check) {
      if (!pma.has(pair.first)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               pair.first);
        return -1;
      }
      if (std::get<0>(pma.value(pair.first)) != pair.second) {
        printf("FAILED: value doesn't match after inserting elements\n");
        std::cout << "key is " << +pair.first << std::endl;
        std::cout << +std::get<0>(pma.value(pair.first)) << ", " << +pair.second
                  << std::endl;

        pma.print_pma();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  for (auto &pair : random_pairs) {
    // std::cout << "trying to remove: " << pair.first << std::endl;
    pma.remove(pair.first);
    // pma.print_pma();
    // printf("#######\n");
    if (check) {
      if (pma.has(pair.first)) {
        printf("have something we removed while removing elements, tried to "
               "remove %u\n",
               pair.first);
        return -1;
      }
    }
  }
  if (check && pma.get_n() != 0) {
    printf("still have elements when we shouldn't\n");
    pma.print_pma();
    return -1;
  }
  uint64_t remove_duration = get_usecs() - start;
  printf("insert duration = %lu, remove duration = %lu\n", insert_duration,
         remove_duration);
  return 0;
}

[[nodiscard]] int pma_map_test(uint32_t el_count, bool check = false) {
  int r = 0;
  r = pma_map_test_templated<1, uint8_t>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_test_templated<2, uint32_t>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_test_templated<3, float>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_test_templated<4, double>(el_count, check);
  if (r) {
    return r;
  }
  return 0;
}

template <uint32_t b>
[[nodiscard]] int pma_map_soa_test_templated(uint32_t el_count,
                                             bool check = false) {
  uint32_t max = 0;
  if constexpr (b == 4) {
    max = UINT32_MAX;
  } else {
    max = 1U << (b * 8);
  }
  PMA<sized_uint<b>, uint8_t, uint16_t, uint32_t, uint64_t> pma_soa;
  using tup_type = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>;
  el_count = std::min(max - 1, el_count);
  std::vector<uint32_t> random_numbers = create_random_data(el_count, max);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    // printf("trying to insert %u\n", random_numbers[i]);
    pma_soa.insert({random_numbers[i], i, 2 * i, 3 * i, 4 * i});
    // pma.print_pma();
    // printf("\n");
    if (check) {
      if (!pma_soa.has(random_numbers[i])) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               random_numbers[i]);
        pma_soa.print_pma();
        return -1;
      }
      if (pma_soa.value(random_numbers[i]) !=
          tup_type(i, 2 * i, 3 * i, 4 * i)) {
        printf("FAILED: have the wrong value while inserting "
               "elements, %u\n",
               random_numbers[i]);
        std::cout << "got " << pma_soa.value(random_numbers[i]);
        std::cout << "\nexpected " << tup_type(i, 2 * i, 3 * i, 4 * i);
        std::cout << "\n";
        pma_soa.print_pma();
        return -1;
      }
    }
  }
  std::map<uint32_t, std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>> checker;
  if (check) {
    for (uint32_t i = 0; i < el_count; i++) {
      checker.insert_or_assign(random_numbers[i],
                               tup_type(i, 2 * i, 3 * i, 4 * i));
    }
    for (const auto &[key, value] : checker) {
      if (!pma_soa.has(key)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               key);
        return -1;
      }
      if (pma_soa.value(key) != value) {
        printf("FAILED: have the wrong value after inserting "
               "elements, %u\n",
               key);
        std::cout << "got " << pma_soa.value(key);
        std::cout << "\nexpected " << value;
        std::cout << "\n";
        pma_soa.print_pma();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum_keys = pma_soa.sum_keys();
  uint64_t sum_key_duration = get_usecs() - start;
  start = get_usecs();
  uint8_t sum_byte = 0;
  pma_soa.template map<true, 0>([&sum_byte]([[maybe_unused]] uint32_t key,
                                            uint8_t val) { sum_byte += val; });
  uint64_t sum_byte_duration = get_usecs() - start;
  start = get_usecs();
  uint16_t sum_short = 0;
  pma_soa.template map<true, 1>(
      [&sum_short]([[maybe_unused]] uint32_t key, uint16_t val) {
        sum_short += val;
      });
  uint64_t sum_short_duration = get_usecs() - start;
  start = get_usecs();
  uint32_t sum_int = 0;
  pma_soa.template map<true, 2>([&sum_int]([[maybe_unused]] uint32_t key,
                                           uint32_t val) { sum_int += val; });
  uint64_t sum_int_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum_long = 0;
  pma_soa.template map<true, 3>([&sum_long]([[maybe_unused]] uint32_t key,
                                            uint64_t val) { sum_long += val; });
  uint64_t sum_long_duration = get_usecs() - start;
  printf("max, %.9u, b , %.2u, el_count , %.9u, average_insert_time, %.10f, "
         "sum_key_time, %.6f, sum_byte_time, %.6f, sum_short_time, %.6f, "
         "sum_int_time, %.6f, sum_long_time, %.6f\n",
         max, b, el_count, ((double)insert_duration) / (1000000 * el_count),
         ((float)sum_key_duration) / 1000000,
         ((float)sum_byte_duration) / 1000000,
         ((float)sum_short_duration) / 1000000,
         ((float)sum_int_duration) / 1000000,
         ((float)sum_long_duration) / 1000000);
  if (check) {

    uint64_t sum_key_check = 0;
    uint8_t sum_byte_check = 0;
    uint16_t sum_short_check = 0;
    uint32_t sum_int_check = 0;
    uint64_t sum_long_check = 0;
    for (auto s : checker) {
      sum_key_check += s.first;
      sum_byte_check += std::get<0>(s.second);
      sum_short_check += std::get<1>(s.second);
      sum_int_check += std::get<2>(s.second);
      sum_long_check += std::get<3>(s.second);
    }
    if (sum_key_check != sum_keys) {
      printf("FAILED: got bad result in pma key sum in "
             "pma_map_soa_test_templated, got %lu, "
             "expect %lu\n",
             sum_keys, sum_key_check);
      return -1;
    }
    if (sum_byte_check != sum_byte) {
      printf("FAILED: got bad result in pma byte sum in "
             "pma_map_soa_test_templated, got %u, "
             "expect %u\n",
             +sum_byte, +sum_byte_check);
      return -1;
    }
    if (sum_short_check != sum_short) {
      printf("FAILED: got bad result in pma short sum in "
             "pma_map_soa_test_templated, got %u, "
             "expect %u\n",
             +sum_short, +sum_short_check);
      return -1;
    }
    if (sum_int_check != sum_int) {
      printf("FAILED: got bad result in pma int sum in "
             "pma_map_soa_test_templated, got %u, "
             "expect %u\n",
             sum_int, sum_int_check);
      return -1;
    }
    if (sum_long_check != sum_long) {
      printf("FAILED: got bad result in pma long sum in "
             "pma_map_soa_test_templated, got %lu, "
             "expect %lu\n",
             sum_long, sum_long_check);
      return -1;
    }
  }
  return 0;
}

[[nodiscard]] int pma_map_soa_test(uint32_t el_count, bool check = false) {
  int r = 0;
  r = pma_map_soa_test_templated<1>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_soa_test_templated<2>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_soa_test_templated<3>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_soa_test_templated<4>(el_count, check);
  if (r) {
    return r;
  }
  return 0;
}

template <uint32_t b>
[[nodiscard]] int pma_map_tuple_test_templated(uint32_t el_count,
                                               bool check = false) {
  uint32_t max = 0;
  if constexpr (b == 4) {
    max = UINT32_MAX;
  } else {
    max = 1U << (b * 8U);
  }
  using tup_type = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>;
  PMA<sized_uint<b>, tup_type> pma_soa;

  el_count = std::min(max - 1, el_count);
  std::vector<uint32_t> random_numbers = create_random_data(el_count, max);
  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    // printf("trying to insert %u\n", random_numbers[i]);
    pma_soa.insert({random_numbers[i], {i, 2 * i, 3 * i, 4 * i}});
    // pma.print_pma();
    // printf("\n");
    if (check) {
      if (!pma_soa.has(random_numbers[i])) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               random_numbers[i]);
        pma_soa.print_pma();
        return -1;
      }
      if (std::get<0>(pma_soa.value(random_numbers[i])) !=
          tup_type(i, 2 * i, 3 * i, 4 * i)) {
        printf("FAILED: have the wrong value while inserting "
               "elements, %u\n",
               random_numbers[i]);
        std::cout << "got " << pma_soa.value(random_numbers[i]);
        std::cout << "\nexpected " << tup_type(i, 2 * i, 3 * i, 4 * i);
        std::cout << "\n";
        pma_soa.print_pma();
        return -1;
      }
    }
  }
  std::map<uint32_t, std::tuple<uint8_t, uint16_t, uint32_t, uint64_t>> checker;
  if (check) {
    for (uint32_t i = 0; i < el_count; i++) {
      checker.insert_or_assign(random_numbers[i],
                               tup_type(i, 2 * i, 3 * i, 4 * i));
    }
    for (const auto &[key, value] : checker) {

      if (!pma_soa.has(key)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               key);
        return -1;
      }
      if (std::get<0>(pma_soa.value(key)) != value) {
        printf("FAILED: have the wrong value after inserting "
               "elements, %u\n",
               key);
        std::cout << "got " << pma_soa.value(key);
        std::cout << "\nexpected " << value;
        std::cout << "\n";
        pma_soa.print_pma();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum_keys = pma_soa.sum_keys();
  uint64_t sum_key_duration = get_usecs() - start;
  start = get_usecs();
  uint8_t sum_byte = 0;
  pma_soa.template map<true, 0>(
      [&sum_byte]([[maybe_unused]] uint32_t key,
                  std::tuple<uint8_t, uint16_t, uint32_t, uint64_t> val) {
        sum_byte += std::get<0>(val);
      });
  uint64_t sum_byte_duration = get_usecs() - start;
  start = get_usecs();
  uint16_t sum_short = 0;
  pma_soa.template map<true, 0>(
      [&sum_short]([[maybe_unused]] uint32_t key,
                   std::tuple<uint8_t, uint16_t, uint32_t, uint64_t> val) {
        sum_short += std::get<1>(val);
      });
  uint64_t sum_short_duration = get_usecs() - start;
  start = get_usecs();
  uint32_t sum_int = 0;
  pma_soa.template map<true, 0>(
      [&sum_int]([[maybe_unused]] uint32_t key,
                 std::tuple<uint8_t, uint16_t, uint32_t, uint64_t> val) {
        sum_int += std::get<2>(val);
      });
  uint64_t sum_int_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum_long = 0;
  pma_soa.template map<true, 0>(
      [&sum_long]([[maybe_unused]] uint32_t key,
                  std::tuple<uint8_t, uint16_t, uint32_t, uint64_t> val) {
        sum_long += std::get<3>(val);
      });
  uint64_t sum_long_duration = get_usecs() - start;
  printf("max, %.9u, b , %.2u, el_count , %.9u, average_insert_time, %.10f, "
         "sum_key_time, %.6f, sum_byte_time, %.6f, sum_short_time, %.6f, "
         "sum_int_time, %.6f, sum_long_time, %.6f\n",
         max, b, el_count, ((double)insert_duration) / (1000000 * el_count),
         ((float)sum_key_duration) / 1000000,
         ((float)sum_byte_duration) / 1000000,
         ((float)sum_short_duration) / 1000000,
         ((float)sum_int_duration) / 1000000,
         ((float)sum_long_duration) / 1000000);
  if (check) {

    uint64_t sum_key_check = 0;
    uint8_t sum_byte_check = 0;
    uint16_t sum_short_check = 0;
    uint32_t sum_int_check = 0;
    uint64_t sum_long_check = 0;
    for (auto s : checker) {
      sum_key_check += s.first;
      sum_byte_check += std::get<0>(s.second);
      sum_short_check += std::get<1>(s.second);
      sum_int_check += std::get<2>(s.second);
      sum_long_check += std::get<3>(s.second);
    }
    if (sum_key_check != sum_keys) {
      printf("FAILED: got bad result in pma key sum in "
             "pma_map_tuple_test_templated, got %lu, "
             "expect %lu\n",
             sum_keys, sum_key_check);
      return -1;
    }
    if (sum_byte_check != sum_byte) {
      printf("FAILED: got bad result in pma byte sum in "
             "pma_map_tuple_test_templated, got %u, "
             "expect %u\n",
             +sum_byte, +sum_byte_check);
      return -1;
    }
    if (sum_short_check != sum_short) {
      printf("FAILED: got bad result in pma short sum in "
             "pma_map_tuple_test_templated, got %u, "
             "expect %u\n",
             +sum_short, +sum_short_check);
      return -1;
    }
    if (sum_int_check != sum_int) {
      printf("FAILED: got bad result in pma int sum in "
             "pma_map_tuple_test_templated, got %u, "
             "expect %u\n",
             sum_int, sum_int_check);
      return -1;
    }
    if (sum_long_check != sum_long) {
      printf("FAILED: got bad result in pma long sum in "
             "pma_map_tuple_test_templated, got %lu, "
             "expect %lu\n",
             sum_long, sum_long_check);
      return -1;
    }
  }
  return 0;
}

[[nodiscard]] int pma_map_tuple_test(uint32_t el_count, bool check = false) {
  int r = 0;
  r = pma_map_tuple_test_templated<1>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_tuple_test_templated<2>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_tuple_test_templated<3>(el_count, check);
  if (r) {
    return r;
  }
  r = pma_map_tuple_test_templated<4>(el_count, check);
  if (r) {
    return r;
  }
  return 0;
}

int main(int argc, char *argv[]) {

  cxxopts::Options options("PMAtester",
                           "allows testing diferent attributes of the PMA");

  options.positional_help("Help Text");

  // clang-format off
  options.add_options()
    ("sizes", "sizeof different objects")
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("v, verify", "verify the results of the test, might be much slower")
    ("pma_update_test", "time updating a pma")
    ("pma_map_test","time updating a pma with values")
    ("pma_map_soa_test","time iupdating a pma with multiple values")
    ("pma_map_tuple_test","time iupdating a pma with multiple values")
    ("help","Print help");
  // clang-format on

  auto result = options.parse(argc, argv);
  uint32_t el_count = result["el_count"].as<int>();
  bool verify = result["verify"].as<bool>();

  if (result["sizes"].as<bool>()) {
    printf("PMA<sized_uint<4>>\n");
    PMA<sized_uint<4>>::print_details();

    printf("PMA<sized_uint<4>, uint8_t>\n");
    PMA<sized_uint<4>, uint8_t>::print_details();

    printf("PMA<sized_uint<4>, double>\n");
    PMA<sized_uint<4>, double>::print_details();

    printf("PMA<sized_uint<4>, uint8_t, int, uint8_t>\n");
    PMA<sized_uint<4>, uint8_t, int, uint8_t>::print_details();

    printf("PMA<sized_uint<4>, std::tuple<double, int, uint8_t>>\n");
    PMA<sized_uint<4>, std::tuple<double, int, uint8_t>>::print_details();
    return 0;
  }

  if (result["pma_update_test"].as<bool>()) {
    return pma_update_test(el_count, verify);
  }
  if (result["pma_map_test"].as<bool>()) {
    return pma_map_test(el_count, verify);
  }
  if (result["pma_map_soa_test"].as<bool>()) {
    return pma_map_soa_test(el_count, verify);
  }
  if (result["pma_map_tuple_test"].as<bool>()) {
    return pma_map_tuple_test(el_count, verify);
  }
  return 0;
}
