#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/BellmanFord.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"
#include "EdgeMapVertexMap/algorithms/TC.h"
#include "EdgeMapVertexMap/algorithms/Touchall.h"

#include "ParallelTools/parallel.h"

#include "SSTGraph/PMA.hpp"
#include "SSTGraph/SparseMatrix.hpp"
#include "SSTGraph/TinySet.hpp"
#include "SSTGraph/TinySet_small.hpp"
#include "SSTGraph/internal/BitArray.hpp"
#include "SSTGraph/internal/helpers.hpp"
#include "SSTGraph/internal/rmat_util.h"

namespace SSTGraph {

int timing_inserts(uint64_t max_size) {
  printf("std::set, b, 32,");
  uint64_t start = get_usecs();
  std::set<uint32_t> s32;
  uint64_t end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    s32.insert(i);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  uint32_t sum = 0;
  for (auto el : s32) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);

  printf("std::set, b, 16,");
  start = get_usecs();
  std::set<uint16_t> s16;
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    s16.insert(i);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  for (auto el : s16) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);

  printf("std::unordered_set, b, 32, ");
  start = get_usecs();
  std::unordered_set<uint32_t> us32;
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    us32.insert(i);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  for (auto el : us32) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);

  printf("std::unordered_set, b, 16, ");
  start = get_usecs();
  std::unordered_set<uint16_t> us16;
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    us16.insert(i);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  for (auto el : us16) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);

  if (max_size <= 100000) {
    printf("std::vector, b , 32,");
    start = get_usecs();
    std::vector<uint32_t> v;
    end = get_usecs();
    printf("creation, %lu, ", end - start);
    start = get_usecs();
    for (uint32_t i = 0; i < max_size; i++) {
      v.insert(v.begin(), i);
    }
    end = get_usecs();
    printf("insertion, %lu,", end - start);
    start = get_usecs();
    sum = 0;
    for (auto el : v) {
      sum += el;
    }
    end = get_usecs();
    printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  }
  if (max_size <= 100000) {
    printf("std::vector, b , 16,");
    start = get_usecs();
    std::vector<uint16_t> v;
    end = get_usecs();
    printf("creation, %lu, ", end - start);
    start = get_usecs();
    for (uint32_t i = 0; i < max_size; i++) {
      v.insert(v.begin(), i);
    }
    end = get_usecs();
    printf("insertion, %lu,", end - start);
    start = get_usecs();
    sum = 0;
    for (auto el : v) {
      sum += el;
    }
    end = get_usecs();
    printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  }
  if (max_size <= 100000) {
    printf("std::vector, b , 8,");
    start = get_usecs();
    std::vector<uint8_t> v;
    end = get_usecs();
    printf("creation, %lu, ", end - start);
    start = get_usecs();
    for (uint32_t i = 0; i < max_size; i++) {
      v.insert(v.begin(), i);
    }
    end = get_usecs();
    printf("insertion, %lu,", end - start);
    start = get_usecs();
    sum = 0;
    for (auto el : v) {
      sum += el;
    }
    end = get_usecs();
    printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  }
  for (uint8_t b = 8; b <= 32; b += 8) {
    if ((1UL << b) < max_size) {
      continue;
    }
    printf("PMA, b, %u, ", b);
    start = get_usecs();
    PMA<sized_uint<4>> pma;
    end = get_usecs();
    printf("creation, %lu, ", end - start);
    start = get_usecs();
    for (uint32_t i = 0; i < max_size; i++) {
      pma.insert(i);
    }
    end = get_usecs();
    printf("insertion, %lu,", end - start);
    start = get_usecs();
    sum = pma.sum_keys();
    end = get_usecs();
    printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  }

  printf("TinySet");
  start = get_usecs();
  TinySetV ts(max_size);
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    ts.insert(i);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  ts.map<true>([&sum](el_t key) { sum += key; });
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  // for (uint8_t b = 8; b <= 32; b += 8) {
  //   printf("PackedArray, b, %u,", b);
  //   start = get_usecs();
  //   block_t *array = create_PackedArray<0>(32);
  //   end = get_usecs();
  //   printf("creation, %lu, ", end - start);
  //   start = get_usecs();
  //   for (uint32_t i = 0; i < max_size; i++) {
  //     array = PackedArray_insert<0>(array, i, i, 0);
  //   }
  //   end = get_usecs();
  //   printf("insertion, %lu,", end - start);
  //   start = get_usecs();
  //   sum = 0;
  //   for (uint32_t i = 0; i < max_size; i++) {
  //     sum += PackedArray_get<0>(array, i, b);
  //   }
  //   end = get_usecs();
  //   printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  // }
  return 0;
}

int timing_random_inserts(uint64_t max_size, uint64_t num_inserts) {
  srand(0);
  printf("std::set, b, 32,");
  uint64_t start = get_usecs();
  std::set<uint32_t> s32;
  uint64_t end = get_usecs();
  printf("creation, %lu, ", end - start);
  auto random_numbers = create_random_data<uint32_t>(num_inserts, max_size);
  start = get_usecs();
  for (uint32_t i = 0; i < num_inserts; i++) {
    s32.insert(random_numbers[i]);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  uint32_t sum = 0;
  for (auto el : s32) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);

  srand(0);
  printf("std::unordered_set, b, 32, ");
  start = get_usecs();
  std::unordered_set<uint32_t> us32;
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < num_inserts; i++) {
    us32.insert(random_numbers[i]);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  for (auto el : us32) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);

  srand(0);
  printf("TinySet");
  start = get_usecs();
  TinySetV ts(max_size);
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < num_inserts; i++) {
    ts.insert(random_numbers[i]);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  ts.map<true>([&sum](el_t key) { sum += key; });
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  srand(0);
  printf("BitArray, ");
  start = get_usecs();
  BitArray bitarray = BitArray(max_size);
  end = get_usecs();
  printf("creation, %lu, ", end - start);
  start = get_usecs();
  for (uint32_t i = 0; i < num_inserts; i++) {
    bitarray.set(random_numbers[i]);
  }
  end = get_usecs();
  printf("insertion, %lu,", end - start);
  start = get_usecs();
  sum = 0;
  for (uint32_t i = 0; i < max_size; i++) {
    if (bitarray.get(i)) {
      sum += i;
    }
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %u\n", end - start, sum);
  return 0;
}

void perf_test_tinyset(uint32_t N) {
  printf("testing tinyset N = %u\n", N);
  uint64_t start = get_usecs();
  TinySetV ts(N);
  for (uint32_t i = 0; i < N; i++) {
    ts.insert(i);
    if (!ts.has(i)) {
      printf("don't have element %u, stopping\n", i);
      return;
    }
  }
  for (uint32_t i = 0; i < N; i++) {
    if (!ts.has(i)) {
      printf("don't have element %u, stopping\n", i);
      return;
    }
  }
  uint64_t duration = get_usecs() - start;
  printf("it took %lu seconds\n", duration / 1000000);
  start = get_usecs();
  uint64_t sum = 0;
  ts.map<true>([&sum](el_t key) { sum += key; });
  uint64_t duration2 = get_usecs() - start;
  printf("it took %lu ms to sum\n", duration2 / 1000);
  printf("%u, %lu, %lu, %lu\n", N, duration, ts.get_size(), sum);
}
void perf_test_set(uint32_t N) {
  printf("testing set N = %u\n", N);
  uint64_t start = get_usecs();
  std::set<uint32_t> ts;
  for (uint32_t i = 0; i < N; i++) {
    ts.insert(i);
  }
  for (uint32_t i = 0; i < N; i++) {
    if (!ts.count(i)) {
      printf("don't have element %u, stopping\n", i);
      return;
    }
  }
  uint64_t duration = get_usecs() - start;
  printf("it took %lu seconds\n", duration / 1000000);
  start = get_usecs();
  uint64_t sum = 0;
  for (uint32_t i : ts) {
    sum += i;
  }
  uint64_t duration2 = get_usecs() - start;
  printf("it took %lu ms to sum\n", duration2 / 1000);
  printf("%u, %lu, %lu, %lu\n", N, duration, ts.size(), sum);
}
void perf_test_unordered_set(uint32_t N) {
  printf("testing unordered_set N = %u\n", N);
  uint64_t start = get_usecs();
  std::unordered_set<uint32_t> ts;
  for (uint32_t i = 0; i < N; i++) {
    ts.insert(i);
  }
  for (uint32_t i = 0; i < N; i++) {
    if (!ts.count(i)) {
      printf("don't have element %u, stopping\n", i);
      return;
    }
  }
  uint64_t duration = get_usecs() - start;
  printf("it took %lu seconds\n", duration / 1000000);
  start = get_usecs();
  uint64_t sum = 0;
  for (uint32_t i : ts) {
    sum += i;
  }
  uint64_t duration2 = get_usecs() - start;
  printf("it took %lu ms to sum\n", duration2 / 1000);
  printf("%u, %lu, %lu, %lu\n", N, duration, ts.size(), sum);
}

bool real_graph(const std::string &filename, [[maybe_unused]] bool symetric,
                int iters = 20, uint32_t start_node = 0,
                uint32_t max_batch_size = 100000,
                bool use_degree_order = false) {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t> *edges =
      get_edges_from_file(filename, &num_edges, &num_nodes);

  printf("done reading in the file, n = %u, m = %lu\n", num_nodes, num_edges);
  uint64_t start = get_usecs();
  SparseMatrixV<true> g(num_nodes, num_nodes);
  uint64_t end = get_usecs();
  printf("creation took %lums\n", (end - start) / 1000);
  start = get_usecs();

  uint64_t bfs_milles = 0;
  uint64_t pr_milles = 0;
  uint64_t bc_milles = 0;
  uint64_t cc_milles = 0;
  uint64_t tc_milles = 0;
  uint64_t bf_milles = 0;
  uint64_t add_batch[10] = {0};
  /*
  for (uint32_t i = 0; i < num_edges; i++) {
    //printf("adding edge %u, (%u, %u)\n", i, srcs[i], dests[i]);
    g.insert(edges[i].x, edges[i].y);
  }
  */

  // uint32_t local_batch_size = 1000;
  uint32_t local_batch_size = num_edges / 1000;
  if (num_edges > 10000) {
    local_batch_size = 10000;
  }
  if (num_edges > 100000) {
    local_batch_size = 100000;
  }
  if (num_edges > 10000000) {
    local_batch_size = 10000000;
  }
  if (num_edges > 50000000) {
    local_batch_size = 50000000;
  }
  if (num_edges > 100000000) {
    local_batch_size = 100000000;
  }
  uint64_t i = 0;
  if (num_edges > local_batch_size) {
    for (; i < num_edges - local_batch_size; i += local_batch_size) {
      g.insert_batch(edges + i, local_batch_size);
      // fprintf(stderr, "num_edges added = %lu\n", i + local_batch_size);
    }
  }
  g.insert_batch(edges + i, num_edges % local_batch_size);

  end = get_usecs();
  // g.insert_batch(edges, num_edges);
  free(edges);
  printf("inserting the edges took %lums\n", (end - start) / 1000);
  // uint64_t insert_time = end - start;
  uint64_t size = g.get_memory_size();
  printf("size = %lu bytes, number of edges = %lu, number of nodes = %u\n",
         size, g.M(), num_nodes);
  SparseMatrixV<true> degree_g = g.rewrite_graph(g.degree_order_map());
  if (use_degree_order) {
    printf("using degree ordered relabeled graph\n");
    g.~SparseMatrixV();
    g = degree_g;
    printf("size = %lu bytes, number of edges = %lu, number of nodes = %u\n",
           size, g.M(), num_nodes);
  }

#if 1
  start = get_usecs();
  uint64_t sum1 = g.touch_all_sum();
  end = get_usecs();
  printf("sum of all the edges was = %lu time to count %lu micros\n", sum1,
         end - start);
  start = get_usecs();
  uint64_t sum2 = EdgeMapVertexMap::TouchAll(g);
  end = get_usecs();
  printf("sum of all the edges was = %lu time to count %lu micros\n", sum2,
         end - start);

  int32_t parallel_bfs_result2_ = 0;
  uint64_t parallel_bfs_time2 = 0;

  for (int i = 0; i < iters; i++) {
    start = get_usecs();
    int32_t *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
    end = get_usecs();
    parallel_bfs_result2_ += parallel_bfs_result[0];
    if (i == 0 && parallel_bfs_result != nullptr) {
      uint64_t reached = 0;
      for (uint32_t j = 0; j < num_nodes; j++) {
        reached += parallel_bfs_result[j] != -1;
      }
      printf("the bfs from source %u, reached %lu vertices\n", start_node,
             reached);
    }
    std::vector<uint32_t> depths(num_nodes, UINT32_MAX);
    ParallelTools::parallel_for(0, num_nodes, [&](uint32_t j) {
      uint32_t current_depth = 0;
      int32_t current_parent = j;
      if (parallel_bfs_result[j] < 0) {
        return;
      }
      while (current_parent != parallel_bfs_result[current_parent]) {
        current_depth += 1;
        current_parent = parallel_bfs_result[current_parent];
      }
      depths[j] = current_depth;
    });
    std::ofstream myfile;
    myfile.open("bfs.out");
    for (unsigned int i = 0; i < num_nodes; i++) {
      myfile << depths[i] << "\n";
    }
    myfile.close();

    free(parallel_bfs_result);
    parallel_bfs_time2 += (end - start);
  }
  // printf("bfs took %lums, parent of 0 = %d\n", (bfs_time)/(1000*iters),
  // bfs_result_/iters);
  printf("parallel_bfs with edge_map took %lums, parent of 0 = %d\n",
         parallel_bfs_time2 / (1000 * iters), parallel_bfs_result2_ / iters);
  bfs_milles = parallel_bfs_time2 / (1000 * iters);

  start = get_usecs();
  auto *values3 = EdgeMapVertexMap::PR_S<double>(g, 10);
  end = get_usecs();
  printf("pagerank with MAPS took %lums, value of 0 = %f, for %d iters\n",
         (end - start) / (1000), values3[0], iters);
  pr_milles = (end - start) / (1000);
  std::ofstream myfile;
  myfile.open("pr.out");
  for (unsigned int i = 0; i < num_nodes; i++) {
    myfile << values3[i] << "\n";
  }
  myfile.close();
  free(values3);

  start = get_usecs();
  double *values4 = nullptr;
  double dep_0 = 0;
  for (int i = 0; i < iters; i++) {
    if (values4 != nullptr) {
      free(values4);
    }
    values4 = EdgeMapVertexMap::BC(g, start_node);
    dep_0 += values4[0];
  }
  end = get_usecs();
  printf("BC took %lums, value of 0 = %f\n", (end - start) / (1000 * iters),
         dep_0 / iters);
  bc_milles = (end - start) / (1000 * iters);
  if (values4 != nullptr) {
    std::ofstream myfile;
    myfile.open("bc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values4[i] << "\n";
    }
    myfile.close();
    free(values4);
  }

  start = get_usecs();
  uint32_t *values5 = nullptr;
  uint32_t id_0 = 0;
  for (int i = 0; i < iters; i++) {
    if (values5) {
      free(values5);
    }
    values5 = EdgeMapVertexMap::CC(g);
    id_0 += values5[0];
  }
  end = get_usecs();
  printf("CC took %lums, value of 0 = %u\n", (end - start) / (1000 * iters),
         id_0 / iters);
  cc_milles = (end - start) / (1000 * iters);
  if (values5 != nullptr) {
    std::unordered_map<uint32_t, uint32_t> components;
    for (uint32_t i = 0; i < num_nodes; i++) {
      components[values5[i]] += 1;
    }
    printf("there are %zu components\n", components.size());
    uint32_t curent_max = 0;
    uint32_t curent_max_key = 0;
    for (auto p : components) {
      if (p.second > curent_max) {
        curent_max = p.second;
        curent_max_key = p.first;
      }
    }
    printf("the element with the biggest component is %u, it has %u members "
           "to its component\n",
           curent_max_key, curent_max);
    std::ofstream myfile;
    myfile.open("cc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values5[i] << "\n";
    }
    myfile.close();
  }

  free(values5);
#if 1
  start = get_usecs();
  EdgeMapVertexMap::TC(g);
  end = get_usecs();
  printf("TC took %lums\n", (end - start) / (1000));
  tc_milles = (end - start) / (1000);
#endif
  start = get_usecs();
  int32_t *bf_values = nullptr;
  int32_t val_0 = 0;
  for (int i = 0; i < iters; i++) {
    if (bf_values != nullptr) {
      free(bf_values);
    }
    bf_values = EdgeMapVertexMap::BF(g, start_node);
    val_0 += bf_values[0];
  }
  end = get_usecs();
  printf("BF took %lums, value of 0 = %d\n", (end - start) / (1000 * iters),
         val_0 / iters);
  bf_milles = (end - start) / (1000 * iters);
  if (bf_values != nullptr) {
    std::ofstream myfile;
    myfile.open("bf.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << bf_values[i] << "\n";
    }
    myfile.close();
    free(bf_values);
  }
#endif
// batch updates
#if 1
  auto r = random_aspen();
  uint32_t counter = 0;
  for (uint32_t b_size = 10; b_size <= max_batch_size; b_size *= 10) {
    double batch_insert_time = 0;
    double batch_remove_time = 0;
    for (int it = 0; it < iters; it++) {
      // uint64_t size = g.get_memory_size();
      // printf("size start = %lu\n", size);
      double a = 0.5;
      double b = 0.1;
      double c = 0.1;
      size_t nn = 1UL << (log2_up(num_nodes) - 1);
      auto rmat = rMat<uint32_t>(nn, r.ith_rand(0), a, b, c);
      std::vector<std::tuple<el_t, el_t>> es(b_size);
      ParallelTools::parallel_for(0, b_size, [&](uint32_t i) {
        std::pair<uint32_t, uint32_t> edge = rmat(i);
        es[i] = {edge.first, edge.second};
      });
      // std::unordered_map<uint32_t, uint32_t> count_per_vertex;
      // std::map<uint32_t, uint32_t> count_per_count;
      // for (auto x : es) {
      //   count_per_vertex[x.x]++;
      // }
      // for (auto x : count_per_vertex) {
      //   count_per_count[x.second]++;
      // }
      // for (auto x : count_per_count) {
      //   std::cout << x.second << " vertices had " << x.first << "
      //   elements"
      //             << std::endl;
      // }
      start = get_usecs();
      g.insert_batch(es.data(), b_size);
      end = get_usecs();
      batch_insert_time += (double)(end - start);
      // size = g.get_memory_size();
      // printf("size end = %lu\n", size);
      start = get_usecs();
      g.remove_batch(es.data(), b_size);
      end = get_usecs();
      batch_remove_time += (double)(end - start);
    }
    batch_insert_time /= (1000000 * iters);
    batch_remove_time /= (1000000 * iters);
    printf("batch_size = %d, time to insert = %f seconds, throughput = %4.2e "
           "updates/second\n",
           b_size, batch_insert_time, b_size / (batch_insert_time));
    printf("batch_size = %d, time to remove = %f seconds, throughput = %4.2e "
           "updates/second\n",
           b_size, batch_remove_time, b_size / (batch_remove_time));
    add_batch[counter] = b_size / (batch_insert_time * 1000000);
    counter += 1;
  }
#endif
  printf("%s, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu %lu\n",
         filename.c_str(), bfs_milles, pr_milles, bc_milles, cc_milles,
         bf_milles, tc_milles, add_batch[0], add_batch[1], add_batch[2],
         add_batch[3], add_batch[4]);

  return true;
}

template <typename value_type>
bool real_graph_weights(const std::string &filename,
                        [[maybe_unused]] bool symetric, int iters = 20,
                        uint32_t start_node = 0,
                        [[maybe_unused]] uint32_t max_batch_size = 100000) {
  std::cout << "value_type is " << TypeName<value_type>() << std::endl;
  using edge_type =
      typename std::conditional<std::is_same<value_type, bool>::value,
                                std::tuple<el_t, el_t>,
                                std::tuple<el_t, el_t, value_type>>::type;
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  edge_type *edges =
      get_edges_from_file<value_type>(filename, &num_edges, &num_nodes);

  printf("done reading in the file, n = %u, m = %lu\n", num_nodes, num_edges);
  uint64_t start = get_usecs();
  SparseMatrixV<true, value_type> g(num_nodes, num_nodes);
  uint64_t end = get_usecs();
  printf("creation took %lums\n", (end - start) / 1000);
  start = get_usecs();

  uint32_t local_batch_size = num_edges / 1000;
  if (num_edges > 10000) {
    local_batch_size = 10000;
  }
  if (num_edges > 100000) {
    local_batch_size = 100000;
  }
  if (num_edges > 10000000) {
    local_batch_size = 10000000;
  }
  if (num_edges > 50000000) {
    local_batch_size = 50000000;
  }
  if (num_edges > 100000000) {
    local_batch_size = 100000000;
  }
  if (num_edges > 500000000) {
    local_batch_size = 500000000;
  }
  uint64_t i = 0;
  if (num_edges > local_batch_size) {
    for (; i < num_edges - local_batch_size; i += local_batch_size) {
      g.insert_batch(edges + i, local_batch_size);
      fprintf(stderr, "num_edges added = %lu\n", i + local_batch_size);
    }
  }
  g.insert_batch(edges + i, num_edges % local_batch_size);

  end = get_usecs();
  // g.insert_batch(edges, num_edges);
  free(edges);
  printf("inserting the edges took %lums\n", (end - start) / 1000);
  // uint64_t insert_time = end - start;
  uint64_t size = g.get_memory_size();
  printf("size = %lu bytes, number of edges = %lu, number of nodes = %u\n",
         size, g.M(), num_nodes);
  // g.print_statistics();

#if 1
  start = get_usecs();
  uint64_t sum1 = 0;
  for (int i = 0; i < iters; i++) {
    sum1 += g.touch_all_sum();
  }
  end = get_usecs();
  printf("sum of all the edges was = %lu time to count %lu micros\n",
         sum1 / iters, (end - start) / iters);
  start = get_usecs();
  uint64_t sum2 = 0;
  for (int i = 0; i < iters; i++) {
    sum2 += EdgeMapVertexMap::TouchAll(g);
  }
  end = get_usecs();
  printf("sum of all the edges was = %lu time to count %lu micros\n",
         sum2 / iters, (end - start) / iters);
#endif
#if 1
  start = get_usecs();
  int32_t *bf_values = nullptr;
  int32_t val_0 = 0;
  for (int i = 0; i < iters; i++) {
    if (bf_values != nullptr) {
      free(bf_values);
    }
    bf_values = EdgeMapVertexMap::BF(g, start_node);
    val_0 += bf_values[0];
  }
  end = get_usecs();
  printf("BF took %lums, value of 0 = %d\n", (end - start) / (1000 * iters),
         val_0 / iters);
  if (bf_values != nullptr) {
    std::ofstream myfile;
    myfile.open("bf_" + TypeName<value_type>() + ".out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << bf_values[i] << "\n";
    }
    myfile.close();
    free(bf_values);
  }
#endif
// batch updates
#if 0
  auto r = random_aspen();
  for (uint32_t b_size = 10; b_size <= max_batch_size; b_size *= 10) {
    double batch_insert_time = 0;
    double batch_remove_time = 0;
    for (int it = 0; it < iters; it++) {
      // uint64_t size = g.get_memory_size();
      // printf("size start = %lu\n", size);
      double a = 0.5;
      double b = 0.1;
      double c = 0.1;
      size_t nn = 1UL << (log2_up(num_nodes) - 1);
      auto rmat = rMat<uint32_t>(nn, r.ith_rand(0), a, b, c);
      std::vector<edge_type> es(b_size);
      ParallelTools::parallel_for(0, b_size, [&](uint32_t i) {
        std::pair<uint32_t, uint32_t> edge = rmat(i);
        if constexpr (std::is_same_v<value_type, bool>) {
          es[i] = {edge.first, edge.second};
        } else {
          es[i] = {edge.first, edge.second, static_cast<value_type>(1)};
        }
      });
      start = get_usecs();
      g.insert_batch(es.data(), b_size);
      end = get_usecs();
      batch_insert_time += (double)(end - start);
      // size = g.get_memory_size();
      // printf("size end = %lu\n", size);
      start = get_usecs();
      g.remove_batch(es.data(), b_size);
      end = get_usecs();
      batch_remove_time += (double)(end - start);
    }
    batch_insert_time /= (1000000 * iters);
    batch_remove_time /= (1000000 * iters);
    printf("batch_size = %d, time to insert = %f seconds, throughput = %4.2e "
           "updates/second\n",
           b_size, batch_insert_time, b_size / (batch_insert_time));
    printf("batch_size = %d, time to remove = %f seconds, throughput = %4.2e "
           "updates/second\n",
           b_size, batch_remove_time, b_size / (batch_remove_time));
  }
#endif
  return true;
}

bool real_graph_static_test(const std::string &filename,
                            [[maybe_unused]] bool symetric, int iters = 10,
                            uint32_t start_node = 0,
                            const std::string &run_info = "") {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t> *edges =
      get_edges_from_file(filename, &num_edges, &num_nodes);

  if (num_nodes == 0) {
    printf("graphs needs to have non zero number of nodes\n");
    free(edges);
    return false;
  }

  SparseMatrixV<true, bool> g(num_nodes, num_nodes);

  uint32_t local_batch_size = num_edges / 1000;
  if (num_edges > 10000000) {
    local_batch_size = 10000000;
  }
  if (num_edges > 100000000) {
    local_batch_size = 100000000;
  }
  if (num_edges > 500000000) {
    local_batch_size = 500000000;
  }
  uint64_t i = 0;
  if (num_edges > local_batch_size) {
    for (; i < num_edges - local_batch_size; i += local_batch_size) {
      g.insert_batch(edges + i, local_batch_size);
      fprintf(stderr, "num_edges added = %lu\n", i + local_batch_size);
    }
  }
  g.insert_batch(edges + i, num_edges % local_batch_size);

  // g.insert_batch(edges, num_edges);
  free(edges);
  uint64_t start = 0;
  uint64_t end = 0;
  if (g.get_rows() == 0) {
    printf("graph has no vertices\n");
    return false;
  }
#if 1
  auto *values3 = EdgeMapVertexMap::PR_S<double>(g, 10);
  free(values3);
  start = get_usecs();
  for (int i = 0; i < iters; i++) {
    auto *values3 = EdgeMapVertexMap::PR_S<double>(g, 10);
    free(values3);
  }
  end = get_usecs();
  printf("tinyset, %d, PageRank, %d, %s, %s, %f\n", iters, start_node,
         filename.c_str(), run_info.c_str(),
         ((double)(end - start)) / (1000000 * iters));
  fprintf(stderr, "PR done\n");

  int32_t *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
  free(parallel_bfs_result);
  start = get_usecs();
  for (int i = 0; i < iters; i++) {
    int32_t *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
    free(parallel_bfs_result);
  }
  end = get_usecs();
  printf("tinyset, %d, BFS, %d, %s, %s, %f\n", iters, start_node,
         filename.c_str(), run_info.c_str(),
         ((double)(end - start)) / (1000000 * iters));
  fprintf(stderr, "BFS done\n");

  double *values4 = EdgeMapVertexMap::BC(g, start_node);
  free(values4);
  start = get_usecs();
  for (int i = 0; i < iters; i++) {
    values4 = EdgeMapVertexMap::BC(g, start_node);
    free(values4);
  }
  end = get_usecs();
  printf("tinyset, %d, BC, %d, %s, %s, %f\n", iters, start_node,
         filename.c_str(), run_info.c_str(),
         ((double)(end - start)) / (1000000 * iters));
  fprintf(stderr, "BC done\n");
  uint32_t *values5 = EdgeMapVertexMap::CC(g);
  start = get_usecs();
  for (int i = 0; i < iters; i++) {
    if (values5) {
      free(values5);
    }
    values5 = EdgeMapVertexMap::CC(g);
  }
  end = get_usecs();
  printf("tinyset, %d, Components, %d, %s, %s, %f\n", iters, start_node,
         filename.c_str(), run_info.c_str(),
         ((double)(end - start)) / (1000000 * iters));
  fprintf(stderr, "CC done\n");
  free(values5);
#endif
#if 1
  EdgeMapVertexMap::TC(g);
  start = get_usecs();
  for (int i = 0; i < iters; i++) {
    EdgeMapVertexMap::TC(g);
  }
  end = get_usecs();
  printf("tinyset, %d, TC, %d, %s, %s, %f\n", iters, start_node,
         filename.c_str(), run_info.c_str(),
         ((double)(end - start)) / (1000000 * iters));
#endif
  return true;
}
} // namespace SSTGraph
