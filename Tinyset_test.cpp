#include <random>
#include <set>

#include "cxxopts.hpp"

#include "SSTGraph/TinySet.hpp"
#include "SSTGraph/internal/helpers.hpp"

using namespace SSTGraph;

[[nodiscard]] int TinySet_add_test(uint32_t max, uint32_t el_count,
                                   bool check = false) {
  TinySetV ts(max);
  std::vector<uint32_t> random_numbers = create_random_data(el_count, max - 1);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    ts.insert(random_numbers[i]);
    if (check) {
      // printf("inserting %u\n", random_numbers[i]);
      if (!ts.has(random_numbers[i])) {
        printf("don't have something we inserted while inserting elements\n");
        // ts.print();
        return -1;
      }
    }
  }
  for (uint32_t i = 0; i < el_count; i++) {
    if (check) {
      if (!ts.has(random_numbers[i])) {
        printf("don't have something we inserted after inserting elements, "
               "index was %u, number was %u\n",
               i, random_numbers[i]);
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  // printf("pma_sum = %lu\n", ts.pmas[0].pma32.sum());
  uint64_t sum = 0;
  ts.map<true>([&sum](el_t key) { sum += key; });
  uint64_t sum_duration = get_usecs() - start;
  printf("max, %.10u, el_count , %.9u, average_insert_time milles, %.10f, "
         "sum_time milles, %.6f\n",
         max, el_count, ((double)insert_duration) / (double)(1000UL * el_count),
         ((float)sum_duration) / (double)1000UL);
  if (check) {
    std::set<uint32_t> checker(random_numbers.begin(), random_numbers.end());
    uint64_t sum_check = 0;
    for (auto s : checker) {
      sum_check += s;
    }
    if (sum_check != sum) {
      printf("got bad result in TS sum in TinySet_add_test, got %lu, expect "
             "%lu\n",
             sum, sum_check);
      ts.print_pmas();
      ts.print();
      for (auto s : checker) {
        printf("%u, ", s);
      }
      printf("\n");
      return -1;
    }
  }
  return 0;
}

int TinySet_add_test_fast(uint32_t max, uint32_t el_count) {
  TinySetV ts(max);
  std::vector<uint32_t> random_numbers = create_random_data(el_count, max - 1);
  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    ts.insert(random_numbers[i]);
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum = 0;
  ts.map<true>([&sum](el_t key) { sum += key; });
  uint64_t sum_duration = get_usecs() - start;
  printf("max, %.10u, el_count , %.9u, fill_frac, %f, "
         "average_insert_time, %.10f, sum_time, %.6f, sum, %lu\n",
         max, el_count, ((double)el_count) / max,
         ((float)insert_duration) / (double)(1000000UL * el_count),
         ((float)sum_duration) / (double)1000000, sum);
  return 0;
}

[[nodiscard]] int tinyset_remove_test(uint32_t el_count, uint32_t max_val,
                                      bool check = false) {
  TinySetV ts(max_val);
  el_count = std::min(max_val - 1, el_count);
  std::vector<uint32_t> random_numbers =
      create_random_data(el_count, max_val - 1);

  std::unordered_set<uint32_t> checker;

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    ts.insert(random_numbers[i]);
    if (check) {
      checker.insert(random_numbers[i]);
      // printf("adding %u\n", random_numbers[i]);
      if (!ts.has(random_numbers[i])) {
        printf("don't have something we inserted while inserting elements\n");
        return -1;
      }
    }
  }
  for (uint32_t i = 0; i < el_count; i++) {
    if (check) {
      if (!ts.has(random_numbers[i])) {
        printf("don't have something we inserted after inserting elements, "
               "index was %u\n",
               i);
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    ts.remove(random_numbers[i]);
    // ts.print_pmas();
    if (check) {
      checker.erase(random_numbers[i]);
      // printf("removing %u\n", random_numbers[i]);
      if (ts.has(random_numbers[i])) {
        printf("have something we removed while removing elements, tried to "
               "remove %u\n",
               random_numbers[i]);
        return -1;
      }
      for (auto el : checker) {
        if (!ts.has(el)) {
          printf("we removed %u when we shouldn't have\n", el);
          return -1;
        }
      }
    }
  }
  if (check && ts.get_n() != 0) {
    printf("still have elements when we shouldn't\n");
    return -1;
  }
  uint64_t remove_duration = get_usecs() - start;
  printf("insert duration = %lu, remove duration = %lu\n", insert_duration,
         remove_duration);
  return 0;
}

[[nodiscard]] int sorting_test(uint32_t max, uint32_t fill,
                               bool verify = false) {
  std::vector<uint32_t> numbers;
  uint32_t fill_scale = 1000;
  std::random_device r;
  std::mt19937 rng(r());

  std::uniform_int_distribution<uint32_t> dist(0, fill_scale);
  for (uint32_t i = 0; i < max; i++) {
    if (dist(rng) < fill) {
      numbers.push_back(i);
    }
  }
  std::shuffle(numbers.begin(), numbers.end(), rng);
  TinySetV ts(max);
  uint64_t start = get_usecs();
  for (auto item : numbers) {
    ts.insert(item);
  }
  uint64_t end_insert = get_usecs();
  std::vector<uint32_t> ts_numbers(ts.get_n());
  uint64_t i = 0;
  ts.map<true>([&](el_t dest) { ts_numbers[i++] = dest; }, false);
  uint64_t end_iter = get_usecs();
  std::sort(numbers.begin(), numbers.end());
  uint64_t end_sort = get_usecs();

  if (verify) {
    if (ts_numbers != numbers) {
      printf("sorting FAILED\n");
      for (auto item : numbers) {
        if (!ts.has(item)) {
          printf("missing %u\n", item);
        }
      }
      return -1;
    }
    printf("verified\n");
  }
  printf("took %lu to insert, %lu to iterate, %lu total, std::sort took %lu\n",
         end_insert - start, end_iter - end_insert, end_iter - start,
         end_sort - end_iter);
  return 0;
}

[[nodiscard]] int edge_case(uint32_t max, uint32_t fill) {
  std::vector<uint32_t> numbers;
  uint32_t fill_scale = 1000;
  std::random_device r;
  std::mt19937 rng(r());

  std::uniform_int_distribution<uint32_t> dist(0, fill_scale);
  for (uint32_t i = 0; i < max; i++) {
    if (dist(rng) < fill || (i & 0xFFFF) == 0) {
      numbers.push_back(i);
    }
  }
  std::shuffle(numbers.begin(), numbers.end(), rng);
  TinySetV ts(max);
  std::vector<uint32_t> checks;
  for (size_t i = 0; i < numbers.size(); i++) {
    auto item = numbers[i];
    // printf("inserting %u, i = %zu, count = %lu\n", item, i, ts.get_n());
    ts.insert(item);
    if (!ts.has(item)) {
      printf("missing something new while inserting, %u\n", item);
      return -1;
    }
    for (auto check : checks) {
      if (!ts.has(check)) {
        printf("missing something old while inserting, %u\n", check);
        return -1;
      }
    }
    if ((item & 0xFFFF) == 0) {
      checks.push_back(item);
    }
  }
  int missing = 0;
  for (auto item : numbers) {
    if (!ts.has(item)) {
      printf("missing %u after inserting\n", item);
      missing += 1;
    }
  }

  return missing;
}

void tinyset_size_test(uint32_t el_count, uint32_t max_val) {
  TinySetV ts(max_val);
  el_count = std::min(max_val - 1, el_count);
  std::random_device r;
  std::mt19937 rng(r());

  std::uniform_int_distribution<uint32_t> dist(0, max_val);
  while (ts.get_n() < el_count) {
    uint32_t val = dist(rng);
    ts.insert(val);
    if (__builtin_popcount(ts.get_n()) == 1) {
      printf("(%lu, %f)\n", ts.get_n(),
             ((double)ts.get_size() * 8) / ts.get_n());
    }
  }
}

template <typename value_type>
[[nodiscard]] int tinyset_map_insert_test_templated(uint32_t el_count,
                                                    uint32_t max_key,
                                                    bool check = false) {
  TinySetV<value_type> ts(max_key);
  std::unordered_map<uint32_t, value_type> random_pairs;
  auto keys = create_random_data(el_count, max_key - 1);
  auto values = create_random_data<value_type>(el_count);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    uint32_t key = keys[i];
    value_type value = values[i];
    // std::cout << "trying to insert (" << key << ", " << +value << ")"
    //           << std::endl;

    ts.insert({key, value});
    // ts.print_pmas();
    if (check) {
      random_pairs.insert_or_assign(key, value);
      if (!ts.has(key)) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               key);
        ts.template print<0>();
        ts.print_pmas();
        return -1;
      }
      if (ts.value(key) != std::make_tuple(value)) {
        printf("FAILED: value doesn't match while inserting elements\n");
        std::cout << ts.value(key) << ", " << std::make_tuple(value)
                  << std::endl;
        ts.template print<0>();
        ts.print_pmas();
        return -1;
      }
    }
  }
  if (check) {
    for (auto &pair : random_pairs) {

      if (!ts.has(pair.first)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               pair.first);
        return -1;
      }
      if (ts.value(pair.first) != std::make_tuple(pair.second)) {
        printf("FAILED: value doesn't match after inserting elements\n");
        std::cout << "key is " << +pair.first << std::endl;
        std::cout << ts.value(pair.first) << ", "
                  << std::make_tuple(pair.second) << std::endl;

        ts.template print<0>();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum_keys = 0;
  ts.template map<true>([&sum_keys](el_t key) { sum_keys += key; });
  value_type sum_values = 0;
  ts.template map<true, 0>(
      [&sum_values]([[maybe_unused]] el_t key, value_type value) {
        sum_values += value;
      });
  printf("sum_keys = %lu: ", sum_keys);
  std::cout << "sum_keys = " << +sum_values << ": ";
  uint64_t sum_duration = get_usecs() - start;
  if (check) {
    uint64_t correct_sum_keys = 0;
    value_type correct_sum_values = 0;
    for (auto &pair : random_pairs) {
      correct_sum_keys += pair.first;
      correct_sum_values += pair.second;
    }
    if (sum_keys != correct_sum_keys) {
      printf("\nFAILED: sum_keys doesn't match, got %lu, expected %lu\n",
             sum_keys, correct_sum_keys);
      return -1;
    }
    if (!approximatelyEqual(sum_values, correct_sum_values,
                            std::numeric_limits<float>::epsilon() * 10000)) {
      printf("\nFAILED: sum_values doesn't match\n");
      std::cout << "got " << +sum_values << " expected " << +correct_sum_values
                << std::endl;
      // ts.print();
      // for (auto &pair : random_pairs) {
      //   std::cout << "{" << pair.first << ", " << +pair.second << "}"
      //             << ", ";
      // }
      // std::cout << std::endl;
      return -1;
    }
  }

  printf("value_size %.2lu, el_count , %.9u, "
         "average_insert_time, "
         "%.10f, sum time = %lu\n",
         sizeof(value_type), el_count,
         ((float)insert_duration) / (float)(1000000 * el_count),
         sum_duration / 1000);
  return 0;
}

template <typename v1, typename v2, typename v3>
[[nodiscard]] int tinyset_map_soa_insert_test_templated(uint32_t el_count,
                                                        uint32_t max_key,
                                                        bool check = false) {
  TinySetV<v1, v2, v3> ts(max_key);
  auto keys = create_random_data(el_count, max_key - 1);
  auto values1 = create_random_data<v1>(el_count);
  auto values2 = create_random_data<v2>(el_count);
  auto values3 = create_random_data<v3>(el_count);
  std::unordered_map<uint32_t, std::tuple<v1, v2, v3>> random_pairs;

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    uint32_t key = keys[i];
    v1 value1 = values1[i];
    v2 value2 = values2[i];
    v3 value3 = values3[i];
    random_pairs[key] = {value1, value2, value3};
    // std::cout << "trying to insert (" << key << ", " << +value << ")"
    //           << std::endl;

    ts.insert({key, value1, value2, value3});
    // ts.print_pmas();
    if (check) {
      if (!ts.has(key)) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               key);
        ts.template print<0, 1, 2>();
        ts.print_pmas();
        return -1;
      }
      if (ts.value(key) != std::make_tuple(value1, value2, value3)) {
        printf("FAILED: value doesn't match while inserting elements\n");
        std::cout << ts.value(key) << ", "
                  << std::make_tuple(value1, value2, value3) << std::endl;
        ts.template print<0, 1, 2>();
        ts.print_pmas();
        return -1;
      }
    }
  }
  for (auto &pair : random_pairs) {
    if (check) {
      if (!ts.has(pair.first)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               pair.first);
        return -1;
      }
      if (ts.value(pair.first) != pair.second) {
        printf("FAILED: value doesn't match after inserting elements\n");
        std::cout << "key is " << +pair.first << std::endl;
        std::cout << ts.value(pair.first) << ", " << pair.second << std::endl;
        ;
        ts.template print<0, 1, 2>();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  uint64_t sum_keys = 0;
  ts.template map<true>([&sum_keys](el_t key) { sum_keys += key; });
  v1 sum_v1 = 0;
  v2 sum_v2 = 0;
  v3 sum_v3 = 0;
  ts.template map<true, 0, 1, 2>(
      [&sum_v1, &sum_v2, &sum_v3]([[maybe_unused]] el_t key, v1 val1, v2 val2,
                                  v3 val3) {
        sum_v1 += val1;
        sum_v2 += val2;
        sum_v3 += val3;
      });
  printf("sum_keys = %lu: ", sum_keys);
  std::cout << "sum_values = " << std::make_tuple(sum_v1, sum_v2, sum_v3)
            << ": ";
  uint64_t sum_duration = get_usecs() - start;
  if (check) {
    uint64_t correct_sum_keys = 0;
    v1 correct_sum_v1 = 0;
    v2 correct_sum_v2 = 0;
    v3 correct_sum_v3 = 0;
    for (auto &pair : random_pairs) {
      correct_sum_keys += pair.first;
      correct_sum_v1 += std::get<0>(pair.second);
      correct_sum_v2 += std::get<1>(pair.second);
      correct_sum_v3 += std::get<2>(pair.second);
    }
    if (sum_keys != correct_sum_keys) {
      printf("\nFAILED: sum_keys doesn't match, got %lu, expected %lu\n",
             sum_keys, correct_sum_keys);
      return -1;
    }
    if (!approximatelyEqual(sum_v1, correct_sum_v1,
                            std::numeric_limits<float>::epsilon() * 10000)) {
      printf("\nFAILED: sum_values 1 doesn't match\n");
      std::cout << "got " << +sum_v1 << " expected " << +correct_sum_v1
                << std::endl;
      return -1;
    }
    if (!approximatelyEqual(sum_v2, correct_sum_v2,
                            std::numeric_limits<float>::epsilon() * 10000)) {
      printf("\nFAILED: sum_values 2 doesn't match\n");
      std::cout << "got " << +sum_v2 << " expected " << +correct_sum_v2
                << std::endl;
      return -1;
    }
    if (!approximatelyEqual(sum_v3, correct_sum_v3,
                            std::numeric_limits<float>::epsilon() * 10000)) {
      printf("\nFAILED: sum_values 3 doesn't match\n");
      std::cout << "got " << +sum_v3 << " expected " << +correct_sum_v3
                << std::endl;
      return -1;
    }
  }

  printf("el_count , %.9u, "
         "average_insert_time, "
         "%.10f, sum time = %lu\n",
         el_count, ((float)insert_duration) / (float)(1000000 * el_count),
         sum_duration / 1000);
  return 0;
}

[[nodiscard]] int
tinyset_map_insert_test(uint32_t el_count,
                        uint32_t max_key = std::numeric_limits<uint32_t>::max(),
                        bool check = false) {
  int r = 0;
  r = tinyset_map_insert_test_templated<uint8_t>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_insert_test_templated<uint32_t>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_insert_test_templated<float>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_insert_test_templated<double>(el_count, max_key, check);

  if (r) {
    return r;
  }
  r = tinyset_map_soa_insert_test_templated<uint8_t, uint16_t, uint32_t>(
      el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_soa_insert_test_templated<uint8_t, uint64_t, double>(
      el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_soa_insert_test_templated<float, uint16_t, uint64_t>(
      el_count, max_key, check);
  return r;
}

template <typename value_type>
[[nodiscard]] int tinyset_map_remove_test_templated(uint32_t el_count,
                                                    uint32_t max_key,
                                                    bool check = false) {
  TinySetV<value_type> ts(max_key);
  std::unordered_map<uint32_t, value_type> random_pairs;

  auto keys = create_random_data(el_count, max_key - 1);
  auto values = create_random_data<value_type>(el_count);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    uint32_t key = keys[i];
    value_type value = values[i];

    // std::cout << "trying to insert (" << key << ", " << +value << ")"
    //           << std::endl;

    ts.insert({key, value});
    if (check) {
      random_pairs[key] = value;
      if (!ts.has(key)) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               key);
        ts.print();
        return -1;
      }
      if (ts.value(key) != std::make_tuple(value)) {
        printf("FAILED: value doesn't match while inserting elements\n");
        std::cout << ts.value(key) << ", " << std::make_tuple(value)
                  << std::endl;
        ts.print();
        return -1;
      }
    }
  }
  if (check) {
    for (auto &pair : random_pairs) {
      if (!ts.has(pair.first)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               pair.first);
        return -1;
      }
      if (ts.value(pair.first) != std::make_tuple(pair.second)) {
        printf("FAILED: value doesn't match after inserting elements\n");
        std::cout << "key is " << +pair.first << std::endl;
        std::cout << ts.value(pair.first) << ", "
                  << std::make_tuple(pair.second) << std::endl;
        ;
        ts.print();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  for (auto &pair : random_pairs) {
    ts.remove(pair.first);
    if (check) {
      if (ts.has(pair.first)) {
        printf("have something we removed while removing elements, tried to "
               "remove %u\n",
               pair.first);
        return -1;
      }
    }
  }
  if (check && ts.get_n() != 0) {
    printf("still have elements when we shouldn't\n");
    return -1;
  }
  uint64_t remove_duration = get_usecs() - start;
  printf("insert duration = %lu, remove duration = %lu\n", insert_duration,
         remove_duration);
  return 0;
}

template <typename v1, typename v2, typename v3>
[[nodiscard]] int tinyset_map_soa_remove_test_templated(uint32_t el_count,
                                                        uint32_t max_key,
                                                        bool check = false) {
  TinySetV<v1, v2, v3> ts(max_key);
  std::unordered_map<uint32_t, std::tuple<v1, v2, v3>> random_pairs;

  auto keys = create_random_data(el_count, max_key - 1);
  auto values1 = create_random_data<v1>(el_count);
  auto values2 = create_random_data<v2>(el_count);
  auto values3 = create_random_data<v3>(el_count);

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    uint32_t key = keys[i];
    v1 value1 = values1[i];
    v2 value2 = values2[i];
    v3 value3 = values3[i];
    random_pairs[key] = {value1, value2, value3};
    // std::cout << "trying to insert (" << key << ", " << +value << ")"
    //           << std::endl;

    ts.insert({key, value1, value2, value3});
    if (check) {
      if (!ts.has(key)) {
        printf("FAILED: don't have something we inserted while inserting "
               "elements, %u\n",
               key);
        ts.print();
        return -1;
      }
      if (ts.value(key) != std::make_tuple(value1, value2, value3)) {
        printf("FAILED: value doesn't match while inserting elements\n");
        std::cout << ts.value(key) << ", "
                  << std::make_tuple(value1, value2, value3) << std::endl;
        ts.print();
        return -1;
      }
    }
  }
  for (auto &pair : random_pairs) {
    if (check) {
      if (!ts.has(pair.first)) {
        printf("FAILED: don't have something we inserted after inserting "
               "elements, "
               "key was %u\n",
               pair.first);
        return -1;
      }
      if (ts.value(pair.first) != pair.second) {
        printf("FAILED: value doesn't match after inserting elements\n");
        std::cout << "key is " << +pair.first << std::endl;
        std::cout << ts.value(pair.first) << ", " << pair.second << std::endl;
        ;
        ts.print();
        return -1;
      }
    }
  }
  uint64_t insert_duration = get_usecs() - start;
  start = get_usecs();
  for (auto &pair : random_pairs) {
    ts.remove(pair.first);
    if (check) {
      if (ts.has(pair.first)) {
        printf("have something we removed while removing elements, tried to "
               "remove %u\n",
               pair.first);
        return -1;
      }
    }
  }
  if (check && ts.get_n() != 0) {
    printf("still have elements when we shouldn't\n");
    return -1;
  }
  uint64_t remove_duration = get_usecs() - start;
  printf("insert duration = %lu, remove duration = %lu\n", insert_duration,
         remove_duration);
  return 0;
}

[[nodiscard]] int
tinyset_map_remove_test(uint32_t el_count,
                        uint32_t max_key = std::numeric_limits<uint32_t>::max(),
                        bool check = false) {
  int r = 0;
  r = tinyset_map_remove_test_templated<uint8_t>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_remove_test_templated<uint32_t>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_remove_test_templated<float>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_remove_test_templated<double>(el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_soa_remove_test_templated<uint8_t, int, double>(
      el_count, max_key, check);
  if (r) {
    return r;
  }
  r = tinyset_map_soa_remove_test_templated<double, float, long>(
      el_count, max_key, check);
  return r;
}

void test_tinyset_size_file_out(uint64_t max_size) {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
  if (max_size > std::numeric_limits<uint32_t>::max()) {
    max_size = std::numeric_limits<uint32_t>::max();
  }
  std::vector<std::string> header;
  std::vector<std::vector<double>> sizes(max_size + 1);
  {
    TinySetV s(max_size + 1);
    s.print_cutoffs();
  }

  {
    header.push_back("sequence");
    sizes[0].push_back(0);
    TinySetV s(max_size + 1);
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(i);
      sizes[i].push_back(static_cast<double>(s.get_size()) / i);
    }
  }
  {
    header.push_back("uniform_random.csv");
    std::vector<uint32_t> elements = create_random_data<uint32_t>(
        max_size, std::numeric_limits<uint32_t>::max(), seed);
    sizes[0].push_back(0);
    TinySetV s(std::numeric_limits<uint32_t>::max());
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(elements[i - 1]);
      sizes[i].push_back(static_cast<double>(s.get_size()) / s.get_n());
    }
  }

  std::ofstream myfile;
  myfile.open("sizes.csv");
  const char delim = ',';
  myfile << Join(header, delim) << std::endl;
  for (const auto &row : sizes) {
    std::vector<std::string> stringVec;
    for (const auto &e : row) {
      stringVec.push_back(std::to_string(e));
    }
    myfile << Join(stringVec, delim) << std::endl;
  }
}

int main(int argc, char *argv[]) {

  cxxopts::Options options("TinySetTester",
                           "allows testing diferent attributes of the tinyset");

  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("sizes", "sizeof different objects")
    ("v, verify", "verify the results of the test, might be much slower")
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("tinyset_map_add_test", "time inserting into a tinyset with values")
    ("fill_amount", "amount to fill out of 1000", cxxopts::value<std::vector<int>>()->default_value("1,10,50,100"))
    ("tinyset_map_remove_test", "time removing from a tinyset with values")
    ("tinyset_remove_test", "test removing from a tinyset")
    ("tinyset_add_test", "time inserting into a tinyset")
    ("tinyset_add_test_fast", "time inserting into a tinyset, no checks for comparing insert and sum time")
    ("sorting","tests how fast a tinyset can sort a random set of numbers")
    ("edge_case","test an edge_case I found")
    ("size", "tests big tinyset is with different dirtibutions")
    ("m," "max_val", "max value to insert", cxxopts::value<int>()->default_value("2147483647"))
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  uint32_t el_count = result["el_count"].as<int>();
  uint32_t max_val = result["max_val"].as<int>();
  bool verify = result["verify"].as<bool>();

  std::vector<int> fills = {1, 10, 50, 100};
  if (result.count("fill_amount") > 0) {
    fills = result["fill_amount"].as<std::vector<int>>();
  }

  if (result["sizes"].as<bool>()) {
    printf("sizeof(TinySetV<bool>) = %zu\n", sizeof(TinySetV<bool>));
    printf("sizeof(TinySetV<uint32_t>) = %zu\n", sizeof(TinySetV<uint32_t>));
    printf("sizeof(TinySetV<double>) = %zu\n", sizeof(TinySetV<double>));
    printf("sizeof(TinySetV_small<bool>) = %zu\n",
           sizeof(TinySetV_small<bool>));
    printf("sizeof(TinySetV_small<uint32_t>) = %zu\n",
           sizeof(TinySetV_small<uint32_t>));
    printf("sizeof(TinySetV_small<double>) = %zu\n",
           sizeof(TinySetV_small<double>));
    printf("sizeof(TinySetV_small<char, int double>) = %zu\n",
           sizeof(TinySetV_small<char, int, double>));
  }

  if (result.count("sorting") > 0) {
    for (auto fill : fills) {
      int ret = sorting_test(max_val, fill, verify);
      if (ret) {
        return ret;
      }
    }
    return 0;
  }
  if (result.count("edge_case") > 0) {
    for (auto fill : fills) {
      int ret = edge_case(max_val, fill);
      if (ret) {
        return ret;
      }
    }
    return 0;
  }
  if (result.count("size") > 0) {
    tinyset_size_test(el_count, max_val);
    // test_tinyset_size_file_out(el_count);
    return 0;
  }

  if (result["tinyset_map_add_test"].as<bool>()) {
    return tinyset_map_insert_test(el_count, max_val, verify);
  }
  if (result["tinyset_map_remove_test"].as<bool>()) {
    return tinyset_map_remove_test(el_count, max_val, verify);
  }

  if (result["tinyset_remove_test"].as<bool>()) {
    return tinyset_remove_test(el_count, max_val, verify);
  }
  if (result["tinyset_add_test"].as<bool>()) {
    return TinySet_add_test(max_val, el_count, verify);
  }
  if (result["tinyset_add_test_fast"].as<bool>()) {
    for (uint64_t max = 1U << 20U; max <= (1UL << 30U); max *= 2) {
      for (uint64_t el_count = 1U << 16U; el_count <= max / 2; el_count *= 2) {
        int ret = TinySet_add_test_fast(max, el_count);
        if (ret) {
          return ret;
        }
      }
    }
    return 0;
  }
  return 0;
}
