#include "PMA.hpp"
#include "SparseMatrix.hpp"
#include "TinySet.hpp"
#include "TinySet_small.hpp"
#include "cxxopts.hpp"
#include "test.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {

  cxxopts::Options options("SSTGraph runner",
                           "allows running different experiments on SSTGraph");

  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("sizes", "sizeof different objects")
    ("fill_amount", "amount to fill out of 1000", cxxopts::value<std::vector<int>>()->default_value("1,10,50,100"))
    ("r,rows", "how many rows", cxxopts::value<int>()->default_value("10000"))
    ("m," "max_val", "max value to insert", cxxopts::value<int>()->default_value("2147483647"))
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("iters","number of iterations to run graph kernals",cxxopts::value<int>()->default_value("20"))
    ("bfs_src","what node to start bfs from",cxxopts::value<int>()->default_value("0"))
    ("max_batch","max batch_size to insert for ramt gen edges",cxxopts::value<int>()->default_value("100000"))
    ("p","probability",cxxopts::value<double>()->default_value(".001"))
    ("v, verify", "verify the results of the test, might be much slower")
    ("timing_inserts","run a quick test of inserts with a bunch of different data structures")
    ("timing_random_inserts", "run a quick test of inserts with a bunch of different data structures random items up to some value")
    ("perf_test_tinyset", "time insertions and sum for tinyset")
    ("pma_add_test", "time inserting into a pma")
    ("pma_map_add_test","time inserting into a pma with values")
    ("pma_map_remove_test", "time removing from a pma with values")
    ("tinyset_map_add_test", "time inserting into a tinyset with values")
    ("tinyset_map_remove_test", "time removing from a tinyset with values")
    ("matrix_values_add_remove_test","adding and removing from a matrix with values")
    ("pma_remove_test","test removing from a pma")
    ("tinyset_remove_test", "test removing from a tinyset")
    ("tinyset_add_test", "time inserting into a tinyset")
    ("tinyset_add_test_fast", "time inserting into a tinyset, no checks for comparing insert and sum time")
    ("real","runs with a graph from a file",cxxopts::value<std::string>())
    ("real_weights", "runs with a weighted graph from a file", cxxopts::value<std::string>())
    ("static","runs with a graph from a file",cxxopts::value<std::string>())
    ("rewrite","shuffles the node labels of a graph file and makes a new file", cxxopts::value<std::string>())
    ("degree","prints the degree distibution", cxxopts::value<std::string>())
    ("run_info", "extra print to static_test, used to show which cores were used", cxxopts::value< std::string>() ->implicit_value("-1"))
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result["help"].as<bool>()) {
    std::cout << options.help({"", "Group"}) << std::endl;
  }
  if (result["sizes"].as<bool>()) {
    printf("sizeof(PMA<4>) = %zu\n", sizeof(PMA<4>));
    printf("sizeof(PMA<4, uint8_t>) = %zu\n", sizeof(PMA<4, uint8_t>));
    printf("sizeof(PMA<4, double>) = %zu\n", sizeof(PMA<4, double>));
    printf("sizeof(TinySetV<bool>) = %zu\n", sizeof(TinySetV<bool>));
    printf("sizeof(TinySetV<uint32_t>) = %zu\n", sizeof(TinySetV<uint32_t>));
    printf("sizeof(TinySetV<double>) = %zu\n", sizeof(TinySetV<double>));
    printf("sizeof(TinySetV_small<bool>) = %zu\n",
           sizeof(TinySetV_small<bool>));
    printf("sizeof(TinySetV_small<uint32_t>) = %zu\n",
           sizeof(TinySetV_small<uint32_t>));
    printf("sizeof(TinySetV_small<double>) = %zu\n",
           sizeof(TinySetV_small<double>));
    printf("sizeof(SparseMatrixV<true, bool>) = %zu\n",
           sizeof(SparseMatrixV<true, bool>));
    printf("sizeof(SparseMatrixV<false, uint32_t>) = %zu\n",
           sizeof(SparseMatrixV<false, uint32_t>));
    printf("sizeof(SparseMatrixV<false, double>) = %zu\n",
           sizeof(SparseMatrixV<false, double>));
  }
  std::vector<int> fills = {1, 10, 50, 100};
  if (result.count("fill_amount") > 0) {
    fills = result["fill_amount"].as<std::vector<int>>();
  }
  uint32_t rows = result["rows"].as<int>();
  uint32_t el_count = result["el_count"].as<int>();
  uint32_t iters = result["iters"].as<int>();
  uint32_t max_val = result["max_val"].as<int>();
  bool verify = result["verify"].as<bool>();
  uint32_t bfs_src = result["bfs_src"].as<int>();
  uint32_t max_batch = result["max_batch"].as<int>();

  if (result.count("real") > 0) {
    real_graph(result["real"].as<std::string>(), true, iters, bfs_src,
               max_batch);
    return 0;
  }
  if (result.count("real_weights") > 0) {
    real_graph_weights<bool>(result["real_weights"].as<std::string>(), true,
                             iters, bfs_src, max_batch);
    real_graph_weights<uint8_t>(result["real_weights"].as<std::string>(), true,
                                iters, bfs_src, max_batch);
    real_graph_weights<uint16_t>(result["real_weights"].as<std::string>(), true,
                                 iters, bfs_src, max_batch);
    real_graph_weights<uint32_t>(result["real_weights"].as<std::string>(), true,
                                 iters, bfs_src, max_batch);
    real_graph_weights<uint64_t>(result["real_weights"].as<std::string>(), true,
                                 iters, bfs_src, max_batch);
    return 0;
  }
  if (result.count("static") > 0) {
    real_graph_static_test(result["static"].as<std::string>(), true, iters,
                           bfs_src, result["run_info"].as<std::string>());
    return 0;
  }
  if (result.count("degree") > 0) {
    get_graph_distribution(result["degree"].as<std::string>());
    return 0;
  }
  if (result.count("rewrite") > 0) {
    rewrite_graph(result["rewrite"].as<std::string>());
    return 0;
  }

  if (result["timing_inserts"].as<bool>()) {
    timing_inserts(el_count);
  }
  if (result["timing_random_inserts"].as<bool>()) {
    timing_random_inserts(max_val, el_count);
  }
  if (result["perf_test_tinyset"].as<bool>()) {
    perf_test_tinyset(el_count);
  }

  if (result["pma_add_test"].as<bool>()) {
    PMA_add_test<1>(el_count, verify);
    PMA_add_test<2>(el_count, verify);
    PMA_add_test<3>(el_count, verify);
    PMA_add_test<4>(el_count, verify);
  }
  if (result["pma_map_add_test"].as<bool>()) {
    PMA_map_insert_test(el_count, verify);
  }
  if (result["pma_map_remove_test"].as<bool>()) {
    PMA_map_remove_test(el_count, verify);
  }
  if (result["tinyset_map_add_test"].as<bool>()) {
    tinyset_map_insert_test(el_count, max_val, verify);
  }
  if (result["tinyset_map_remove_test"].as<bool>()) {
    tinyset_map_remove_test(el_count, max_val, verify);
  }
  if (result["matrix_values_add_remove_test"].as<bool>()) {
    matrix_values_add_remove_test(el_count, rows, verify);
  }
  if (result["pma_remove_test"].as<bool>()) {
    PMA_remove_test<1>(el_count, verify);
    PMA_remove_test<2>(el_count, verify);
    PMA_remove_test<3>(el_count, verify);
    PMA_remove_test<4>(el_count, verify);
  }
  if (result["tinyset_remove_test"].as<bool>()) {
    tinyset_remove_test(el_count, max_val, verify);
  }
  if (result["tinyset_add_test"].as<bool>()) {
    TinySet_add_test(max_val, el_count, verify);
  }
  if (result["tinyset_add_test_fast"].as<bool>()) {
    for (uint64_t max = 1U << 20U; max <= (1UL << 30U); max *= 2) {
      for (uint64_t el_count = 1U << 16U; el_count <= max / 2; el_count *= 2) {
        TinySet_add_test_fast(max, el_count);
      }
    }
  }
  return 0;
}
