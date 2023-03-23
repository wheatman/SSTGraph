#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <tuple>
#include <type_traits>
#include <vector>

#include "parlay/primitives.h"

#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"

#include "SSTGraph/TinySet_small.hpp"
#include "SSTGraph/internal/BitArray.hpp"
#include "SSTGraph/internal/helpers.hpp"

namespace SSTGraph {

template <bool is_csr_ = true, typename... Ts> class SparseMatrixV {

  static constexpr bool get_if_binary() {
    if constexpr (sizeof...(Ts) == 0) {
      return true;
    } else {
      using FirstType = typename std::tuple_element<0, std::tuple<Ts...>>::type;
      return (sizeof...(Ts) == 1 && std::is_same_v<bool, FirstType>);
    }
  }

  static constexpr bool binary = get_if_binary();

private:
  using element_type =
      typename std::conditional<binary, std::tuple<el_t, el_t>,
                                std::tuple<el_t, el_t, Ts...>>::type;

  using value_type =
      typename std::conditional<binary, std::tuple<>, std::tuple<Ts...>>::type;

  template <int I>
  using NthType = typename std::tuple_element<I, value_type>::type;

  TinySetV_small<Ts...> *lines;
  typename TinySetV_small<Ts...>::extra_data ts_data;
  uint32_t line_count = 0;
  uint32_t line_width;

public:
  [[nodiscard]] uint32_t get_line_count() const { return line_count; }
  [[nodiscard]] uint32_t get_line_width() const { return line_width; }
  [[nodiscard]] size_t num_nodes() const { return line_count; }
  [[nodiscard]] uint32_t getDegree(uint32_t node) const {
    return lines[node].get_n();
  }

  [[nodiscard]] uint64_t common_neighbors(uint32_t node1, uint32_t node2,
                                          bool early_end = false) const {
    static_assert(binary, "common_neighbors only works in binary mode for now");

    if (early_end) {
      return lines[node1].Set_Intersection_Count(ts_data, lines[node2], ts_data,
                                                 node1, node2);
    }
    return lines[node1].Set_Intersection_Count(ts_data, lines[node2], ts_data,
                                               line_width + 1, line_width + 1);
  }

  SparseMatrixV(el_t height, el_t width) {
    if constexpr (is_csr()) {
      line_count = height;
      line_width = width;
    } else {
      line_count = width;
      line_width = height;
    }
    lines = (TinySetV_small<Ts...> *)malloc(line_count *
                                            sizeof(TinySetV_small<Ts...>));
    ts_data = TinySetV_small<Ts...>::get_thresholds(line_width);
    for (el_t i = 0; i < line_count; i++) {
      new (&lines[i]) TinySetV_small<Ts...>();
    }
  }
  SparseMatrixV(const SparseMatrixV &source)
      : ts_data(source.ts_data), line_count(source.line_count),
        line_width(source.line_width) {
    lines = (TinySetV_small<Ts...> *)malloc(line_count *
                                            sizeof(TinySetV_small<Ts...>));
    ParallelTools::parallel_for(0, line_count, [&](el_t i) {
      new (&lines[i]) TinySetV_small<Ts...>(source.lines[i], ts_data);
    });
  }
  SparseMatrixV(SparseMatrixV &&source)
      : ts_data(source.ts_data), line_count(source.line_count),
        line_width(source.line_width) {
    source.line_count = 0;
    lines = source.lines;
    source.lines = nullptr;
  }
  SparseMatrixV &operator=(const SparseMatrixV &source) {
    if (this != &source) {
      if (line_count > 0) {
        for (uint32_t i = 0; i < line_count; i++) {
          lines[i].destroy(ts_data);
        }
        free(lines);
      }
      ts_data = source.ts_data;
      line_count = source.line_count;
      line_width = source.line_width;

      lines = (TinySetV_small<Ts...> *)malloc(line_count *
                                              sizeof(TinySetV_small<Ts...>));
      ParallelTools::parallel_for(0, line_count, [&](el_t i) {
        new (&lines[i]) TinySetV_small<Ts...>(source.lines[i], ts_data);
      });
    }
    return *this;
  }
  SparseMatrixV &operator=(SparseMatrixV &&source) {
    if (this != &source) {
      if (line_count > 0) {
        for (uint32_t i = 0; i < line_count; i++) {
          lines[i].destroy(ts_data);
        }
        free(lines);
      }
      ts_data = source.ts_data;
      line_count = source.line_count;
      source.line_count = 0;

      lines = source.lines;
      source.lines = nullptr;
    }
    return *this;
  }
  ~SparseMatrixV() {
    for (uint32_t i = 0; i < line_count; i++) {
      lines[i].destroy(ts_data);
    }
    free(lines);
  }
  static constexpr bool is_csr() { return is_csr_; }
  static constexpr bool is_csc() { return !is_csr_; }
  static constexpr bool is_binary() { return binary; }
  [[nodiscard]] uint32_t get_rows() const {
    if constexpr (is_csr()) {
      return get_line_count();
    }
    return get_line_width();
  }
  [[nodiscard]] uint32_t get_cols() const {
    if constexpr (is_csr()) {
      return get_line_width();
    }
    return get_line_count();
  }
  [[nodiscard]] bool has(el_t row, el_t col) const;
  value_type value(el_t row, el_t col) const;
  void insert(element_type e);
  void insert(el_t row, el_t col, auto... vals) { insert({row, col, vals...}); }
  void remove(el_t row, el_t col);
  void insert_batch(element_type *edges, uint64_t n);
  void remove_batch(std::tuple<el_t, el_t> *edges, uint64_t n);

  std::vector<uint64_t> degree_order_map() const;

  SparseMatrixV rewrite_graph(std::vector<uint64_t> new_label_map) const;

  template <bool no_early_exit, size_t... Is, class F>
  void map_line(F &&f, uint64_t i, bool parallel) const {
    lines[i].template map<no_early_exit, Is...>(f, ts_data, parallel);
  }

  template <bool no_early_exit, class F, size_t... Is>
  void map_neighbors_impl(
      uint64_t i, F &&f, [[maybe_unused]] void *d, bool parallel,
      [[maybe_unused]] std::integer_sequence<size_t, Is...> int_seq) const {

    constexpr bool keys_only = sizeof...(Is) == 0;
    if constexpr (keys_only) {
      static_assert(std::is_invocable_v<F, uint64_t, uint64_t>,
                    "update function must match given types");
    } else {
      if constexpr (binary) {
        // if its binary and we are calling on types, we will arbitarily pass in
        // something of an interger type (they will have value 1) and the
        // function will then convert this to whatever type it wants
        static_assert(
            std::is_invocable_v<F, uint64_t, uint64_t, decltype(Is)...>,
            "update function must match given types");
      } else {
        static_assert(
            std::is_invocable_v<F, uint64_t, uint64_t, NthType<Is>...>,
            "update function must match given types");
      }
    }

    if constexpr (keys_only) {
      lines[i].template map<no_early_exit>([&](el_t el) { return f(i, el); },
                                           ts_data, parallel);
    } else {
      if constexpr (binary) {
        lines[i].template map<no_early_exit>(
            [&](el_t el) { return f(i, el, (Is >= 0)...); }, ts_data, parallel);
      } else {
        lines[i].template map<no_early_exit, Is...>(
            [&](el_t el, auto... args) { return f(i, el, args...); }, ts_data,
            parallel);
      }
    }
  }

  template <class F, size_t... Is>
  void map_neighbors(uint64_t i, F &&f, [[maybe_unused]] void *d,
                     bool parallel) const {

    if constexpr (sizeof...(Is) > 0) {
      map_neighbors_impl<F::no_early_exit, Is...>(i, f, d, parallel, {});
    } else {
      map_neighbors_impl<F::no_early_exit>(
          i, f, d, parallel, std::make_index_sequence<sizeof...(Ts)>{});
    }
  }

  template <class F, size_t... Is>
  void map_neighbors_early_exit(uint64_t i, F &&f, [[maybe_unused]] void *d,
                                bool parallel) const {

    if constexpr (sizeof...(Is) > 0) {
      map_neighbors_impl<false, Is...>(i, f, d, parallel, {});
    } else {
      map_neighbors_impl<false>(i, f, d, parallel,
                                std::make_index_sequence<sizeof...(Ts)>{});
    }
  }

  template <class F, size_t... Is>
  void map_neighbors_no_early_exit(uint64_t i, F &&f, [[maybe_unused]] void *d,
                                   bool parallel) const {

    if constexpr (sizeof...(Is) > 0) {
      map_neighbors_impl<true, Is...>(i, f, d, parallel, {});
    } else {
      map_neighbors_impl<true>(i, f, d, parallel,
                               std::make_index_sequence<sizeof...(Ts)>{});
    }
  }

  template <class F, size_t... Is>
  void map_range_impl(
      F &&f, uint64_t start_node, uint64_t end_node, [[maybe_unused]] void *d,
      [[maybe_unused]] std::integer_sequence<size_t, Is...> int_seq) const {
    constexpr bool keys_only = sizeof...(Is) == 0;
    if constexpr (keys_only) {
      static_assert(std::is_invocable_v<F, uint64_t, uint64_t>,
                    "update function must match given types");
    } else {
      if constexpr (binary) {
        static_assert(
            std::is_invocable_v<F, uint64_t, uint64_t, decltype(Is)...>,
            "update function must match given types");
      } else {
        static_assert(
            std::is_invocable_v<F, uint64_t, uint64_t, NthType<Is>...>,
            "update function must match given types");
      }
    }
    for (uint64_t i = start_node; i < end_node - 1; i++) {
      lines[i + 1].prefetch_data(ts_data);
      if constexpr (keys_only) {
        lines[i].template map<F::no_early_exit>(
            [&](el_t el) { return f(i, el); }, ts_data, false);
      } else {
        if constexpr (binary) {
          lines[i].template map<F::no_early_exit>(
              [&](el_t el) { return f(i, el, (Is >= 0)...); }, ts_data, false);
        } else {
          lines[i].template map<F::no_early_exit, Is...>(
              [&](el_t el, auto... args) { return f(i, el, args...); }, ts_data,
              false);
        }
      }
    }
    if constexpr (keys_only) {
      lines[end_node - 1].template map<F::no_early_exit>(
          [&](el_t el) { return f(end_node - 1, el); }, ts_data, false);
    } else {
      if constexpr (binary) {
        lines[end_node - 1].template map<F::no_early_exit>(
            [&](el_t el) { return f(end_node - 1, el, (Is >= 0)...); }, ts_data,
            false);
      } else {
        lines[end_node - 1].template map<F::no_early_exit, Is...>(
            [&](el_t el, auto... args) { return f(end_node - 1, el, args...); },
            ts_data, false);
      }
    }
  }

  template <class F, size_t... Is>
  void map_range(F &&f, uint64_t start_node, uint64_t end_node,
                 [[maybe_unused]] void *d) const {
    if constexpr (sizeof...(Is) > 0) {
      map_range_impl<Is...>(f, start_node, end_node, d, {});
    } else {
      map_range_impl(f, start_node, end_node, d,
                     std::make_index_sequence<sizeof...(Ts)>{});
    }
  }

  [[nodiscard]] uint64_t get_memory_size() const;
  [[nodiscard]] uint64_t M() const;
  [[nodiscard]] SparseMatrixV<true, Ts...> convert_to_csr();
  [[nodiscard]] SparseMatrixV<false, Ts...> convert_to_csc();

  template <size_t... Is> void print_arrays() const {
    if constexpr (is_csr()) {
      printf("matrix stored in csr\n");
    }
    if constexpr (is_csc()) {
      printf("matrix stored in csc\n");
    }
    printf("there are %lu elements\n", M());
    for (uint32_t line = 0; line < line_count; line++) {
      printf("line %u\n", line);
      lines[line].template print<Is...>(ts_data);
      uint64_t sum_key = 0;
      lines[line].template map<true>([&sum_key](el_t key) { sum_key += key; },
                                     ts_data);
      printf("sum = %lu\n", sum_key);
    }
  }

  void print_statistics() const {
    if constexpr (is_csr()) {
      printf("matrix stored in csr\n");
    }
    if constexpr (is_csc()) {
      printf("matrix stored in csc\n");
    }
    printf("there are %lu elements\n", M());
    printf("there are %u rows and %u columns\n", get_rows(), get_cols());
    printf("The graph uses %lu bytes of memory\n", get_memory_size());

    using stats_data_t = struct stats_data_t {
      uint32_t counts = 0;
      uint64_t max_num_elements = 0;
      uint64_t should_be_dense = 0;
      uint64_t empty = 0;
      uint64_t total_buckets = 0;
      uint64_t multiple_buckets = 0;
      int64_t bytes_saved_first = 0;
      int64_t bytes_saved_diagonal = 0;
      void update(const stats_data_t &other) {
        counts += other.counts;
        max_num_elements = std::max(max_num_elements, other.max_num_elements);
        should_be_dense += other.should_be_dense;
        empty += other.empty;
        total_buckets += other.total_buckets;
        multiple_buckets += other.multiple_buckets;
        bytes_saved_first += other.bytes_saved_first;
        bytes_saved_diagonal += other.bytes_saved_diagonal;
      }
    };

    std::array<ParallelTools::Reducer<stats_data_t>, 5> stats_data(
        {ParallelTools::Reducer<stats_data_t>(),
         ParallelTools::Reducer<stats_data_t>(),
         ParallelTools::Reducer<stats_data_t>(),
         ParallelTools::Reducer<stats_data_t>(),
         ParallelTools::Reducer<stats_data_t>()});

    ParallelTools::parallel_for(0, line_count, [&](uint32_t i) {
      uint32_t b = lines[i].get_b();
      auto stats_per_line = lines[i].statistics(ts_data, i);
      stats_data[b].update(
          {1, std::get<1>(stats_per_line), std::get<2>(stats_per_line),
           std::get<3>(stats_per_line), std::get<0>(stats_per_line),
           std::get<4>(stats_per_line), std::get<5>(stats_per_line),
           std::get<6>(stats_per_line)});
    });
    for (uint32_t b = 1; b < 5; b++) {
      stats_data_t final_stats = stats_data[b].get();
      printf("there are %u rows with b = %u, and the max number of elements in "
             "any is %lu, giving a density of %f\n",
             final_stats.counts, b, final_stats.max_num_elements,
             static_cast<double>(final_stats.max_num_elements) /
                 (1UL << (b * 8)));
      printf("\t%lu buckets should be stored dense out of %lu total buckets\n",
             final_stats.should_be_dense, final_stats.total_buckets);
      printf("\t%lu buckets are empty out of %lu total buckets\n",
             final_stats.empty, final_stats.total_buckets);
      printf("\t%lu have multiple buckets out of %u total rows\n",
             final_stats.multiple_buckets, final_stats.counts);
      printf("\twould save %ld bytes by storing first 256 separately\n",
             final_stats.bytes_saved_first);
      printf("\twould save %ld bytes by storing 256 diagonal separately\n",
             final_stats.bytes_saved_diagonal);
    }

    std::vector<ParallelTools::Reducer_sum<uint64_t>> counts_per_top_bit(33);
    ParallelTools::parallel_for(0, line_count, [&](uint32_t i) {
      lines[i].template map<true>(
          [&](el_t dest) {
            if (dest == 0) {
              counts_per_top_bit[0].inc();
            } else {
              counts_per_top_bit[bsr_word(dest) + 1].inc();
            }
          },
          ts_data, false);
    });
    printf("count of outgoing edges which uses x bits\n");
    uint64_t prefix_sum = 0;
    for (int i = 0; i < 33; i++) {
      prefix_sum += counts_per_top_bit[i].get();
      printf("%d, %lu, total so far = %lu\n", i, counts_per_top_bit[i].get(),
             prefix_sum);
    }
  }

  template <size_t... Is> void print_line(uint32_t line) {
    lines[line].template print<Ts...>(ts_data);
  }

  // sums the values of the dests
  // doesn't have much meaning in either the graph or matrix world
  // used to touch every value and benchmarks operations
  // when order is not null touched the rows in the specified order
  uint64_t touch_all_sum(uint64_t const *const order = nullptr) const {
    ParallelTools::Reducer_sum<uint64_t> sum;
    if (order == nullptr) {
      ParallelTools::parallel_for(0, line_count, [&](uint64_t i) {
        uint64_t sum_local = 0;
        lines[i].template map<true>(
            [&sum_local](el_t key) { sum_local += key; }, ts_data);
        sum.add(sum_local);
      });
    } else {
      ParallelTools::parallel_for(0, line_count, [&](uint64_t i) {
        uint64_t sum_local = 0;
        lines[order[i]].template map<true>(
            [&sum_local](el_t key) { sum_local += key; }, ts_data);
        sum.add(sum_local);
      });
    }
    return sum.get();
  }

  static SparseMatrixV make_watts_strogatz_graph(uint64_t N, uint64_t K,
                                                 double beta) {
    static_assert(binary);

    SparseMatrixV graph(N, N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dis(0, 1.0);
    std::uniform_int_distribution<uint64_t> int_dist(0, N - 1);
    std::vector<element_type> edges(N * K * 2);
    ParallelTools::parallel_for(0, N, [&](uint64_t i) {
      uint64_t start = 0;
      if (i > (K / 2)) {
        start = i - (K / 2);
      }
      uint64_t end = i + (K / 2);
      for (uint64_t j = start; j < end; j++) {
        uint64_t dest = j % N;
        if (real_dis(gen) < beta) {
          dest = int_dist(gen);
        }
        edges[i * K * 2 + 2 * (j - start)] = {i, dest};
        edges[i * K * 2 + 2 * (j - start) + 1] = {dest, i};
      }
    });
    graph.insert_batch(edges.data(), edges.size());
    return graph;
  }

private:
  template <typename it, typename Key>
  static void sort_batch(it e, uint64_t batch_size, Key &&key) {
    // if this isn't just a set
    if constexpr (!binary) {
      ParallelTools::sort(
          e, e + batch_size,
          [&key](const auto &a, const auto &b) { return key(a) < key(b); });
      return;
    } else {

      if constexpr (!std::is_integral_v<el_t>) {
        ParallelTools::sort(
            e, e + batch_size,
            [&key](const auto &a, const auto &b) { return key(a) < key(b); });
        return;
      } else {
        if (batch_size > 1000) {
          // TODO find out why this doesn't work
          std::vector<element_type> data_vector;
          wrapArrayInVector(e, batch_size, data_vector);
          parlay::integer_sort_inplace(data_vector, key);
          releaseVectorWrapper(data_vector);
        } else {
          ParallelTools::sort(
              e, e + batch_size,
              [&key](const auto &a, const auto &b) { return key(a) < key(b); });
        }
      }
    }
  }
};

template <bool is_csr_, typename... Ts>
uint64_t SparseMatrixV<is_csr_, Ts...>::M() const {
  uint64_t size = 0;
  for (el_t i = 0; i < line_count; i++) {
    size += lines[i].get_n();
  }
  return size;
}

template <bool is_csr_, typename... Ts>
uint64_t SparseMatrixV<is_csr_, Ts...>::get_memory_size() const {
  uint64_t size = 0;
  for (uint32_t i = 0; i < line_count; i++) {
    size += lines[i].get_size(ts_data);
  }
  return size + sizeof(SparseMatrixV);
}

template <bool is_csr_, typename... Ts>
void __attribute__((noinline))
SparseMatrixV<is_csr_, Ts...>::insert(element_type e) {
  if constexpr (is_csr()) {
    lines[std::get<0>(e)].insert(leftshift_tuple(e), ts_data);
  } else {
    lines[std::get<1>(e)].insert(remove_second_from_tuple(e), ts_data);
  }
}

template <bool is_csr_, typename... Ts>
void __attribute__((noinline))
SparseMatrixV<is_csr_, Ts...>::remove(el_t row, el_t col) {
  if constexpr (is_csr()) {
    lines[row].remove(col, ts_data);
  } else {
    lines[col].remove(row, ts_data);
  }
}

template <bool is_csr_, typename... Ts>
void __attribute__((noinline))
SparseMatrixV<is_csr_, Ts...>::insert_batch(element_type *edges, uint64_t n) {
  uint64_t n_workers = ParallelTools::getWorkers();
  uint64_t p = std::min(std::max(1UL, n / 100), n_workers);
  std::vector<uint64_t> indxs(p + 1);
  indxs[0] = 0;
  indxs[p] = n;
  if (is_csr()) {
    if (n <= 100) {
      for (uint64_t idx = 0; idx < n; idx++) {
        // Not including self loops to compare to aspen
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        if (x == y) {
          continue;
        }
        lines[x].insert(leftshift_tuple(edges[idx]), ts_data);
      }
      return;
    }
    sort_batch(edges, n, [](const element_type &e) { return std::get<0>(e); });

    // printf("done sorting\n");
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * n) / p;
      el_t start_val = std::get<0>(edges[start]);
      while (std::get<0>(edges[start]) == start_val) {
        start += 1;
        if (start == n) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](uint64_t i) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];

      for (; idx < end; idx++) {

        // Not including self loops to compare to aspen
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        if (x == y) {
          continue;
        }
        lines[x].insert(leftshift_tuple(edges[idx]), ts_data);
      }
    });
  } else {
    if (n <= 100) {
      for (uint64_t idx = 0; idx < n; idx++) {
        // Not including self loops to compare to aspen
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        if (x == y) {
          continue;
        }
        lines[y].insert(remove_second_from_tuple(edges[idx]), ts_data);
      }
      return;
    }
    sort_batch(edges, n, [](const element_type &e) { return std::get<1>(e); });
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * n) / p;
      el_t start_val = std::get<1>(edges[start]);
      while (std::get<1>(edges[start]) == start_val) {
        start += 1;
        if (start == n - 1) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](uint64_t i) {
      for (uint64_t idx = indxs[i]; idx < indxs[i + 1]; idx++) {
        // el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[y].insert(remove_second_from_tuple(edges[idx]), ts_data);
      }
    });
  }
}

template <bool is_csr_, typename... Ts>
void __attribute__((noinline))
SparseMatrixV<is_csr_, Ts...>::remove_batch(std::tuple<el_t, el_t> *edges,
                                            uint64_t n) {
  uint64_t n_workers = ParallelTools::getWorkers();
  uint64_t p = std::min(std::max(1UL, n / 100), n_workers);
  std::vector<uint64_t> indxs(p + 1);
  indxs[0] = 0;
  indxs[p] = n;
  if (is_csr()) {
    if (n <= 100) {
      for (uint64_t idx = 0; idx < n; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[x].remove(y, ts_data);
      }
      return;
    }
    sort_batch(edges, n, [](const element_type &e) { return std::get<0>(e); });
    // printf("done sorting\n");
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * n) / p;
      el_t start_val = std::get<0>(edges[start]);
      while (std::get<0>(edges[start]) == start_val) {
        start += 1;
        if (start == n) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, p, [&](uint64_t i) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];
      for (; idx < end; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[x].remove(y, ts_data);
      }
    });
  } else {
    if (n <= 100) {
      for (uint64_t idx = 0; idx < n; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[y].remove(x, ts_data);
      }
      return;
    }
    sort_batch(edges, n, [](const element_type &e) { return std::get<1>(e); });
    for (uint64_t i = 1; i < p; i++) {
      uint64_t start = (i * n) / p;
      el_t start_val = std::get<1>(edges[start]);
      while (std::get<1>(edges[start]) == start_val) {
        start += 1;
        if (start == n - 1) {
          break;
        }
      }
      indxs[i] = start;
    }
    ParallelTools::parallel_for(0, line_count, [&](uint64_t i) {
      for (uint64_t idx = indxs[i]; idx < indxs[i + 1]; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[y].remove(x, ts_data);
      }
    });
  }
}

template <bool is_csr_, typename... Ts>
bool SparseMatrixV<is_csr_, Ts...>::has(el_t row, el_t col) const {
  if constexpr (is_csr()) {
    return lines[row].has(col, ts_data);
  }
  return lines[col].has(row, ts_data);
}

template <bool is_csr_, typename... Ts>
typename SparseMatrixV<is_csr_, Ts...>::value_type
SparseMatrixV<is_csr_, Ts...>::value(el_t row, el_t col) const {
  if constexpr (is_csr()) {
    return lines[row].value(col, ts_data);
  }
  return lines[col].value(row, ts_data);
}

template <bool is_csr_, typename... Ts>
std::vector<uint64_t> SparseMatrixV<is_csr_, Ts...>::degree_order_map() const {
  std::vector<std::pair<uint64_t, uint64_t>> label_map(get_rows());
  ParallelTools::parallel_for(0, get_rows(), [&](uint64_t i) {
    label_map[i] = {getDegree(i), i};
  });
  std::sort(label_map.begin(), label_map.end(), std::greater<>());
  std::vector<uint64_t> label_map_final(get_rows());
  ParallelTools::parallel_for(0, get_rows(), [&](uint64_t i) {
    label_map_final[label_map[i].second] = i;
  });
  return label_map_final;
}

template <bool is_csr_, typename... Ts>
SparseMatrixV<is_csr_, Ts...> SparseMatrixV<is_csr_, Ts...>::rewrite_graph(
    std::vector<uint64_t> new_label_map) const {
  static_assert(binary, "only implemented for binary graphs");
  SparseMatrixV<is_csr_, Ts...> new_matrix =
      SparseMatrixV(get_rows(), get_cols());

  ParallelTools::parallel_for(0, get_rows(), [&](uint64_t i) {
    lines[i].template map<true>(
        [&](el_t dest) {
          new_matrix.insert({new_label_map[i], new_label_map[dest]});
        },
        ts_data, false);
  });
  return new_matrix;
}

} // namespace SSTGraph
