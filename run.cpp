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

  cxxopts::Options options("Graph tester", "Runns tests on different graphs");

  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("sizes", "sizeof different objects")
    ("m," "max_val", "max value to insert", cxxopts::value<int>()->default_value("2147483647"))
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("iters","number of iterations to run graph kernals",cxxopts::value<int>()->default_value("20"))
    ("bfs_src","what node to start bfs from",cxxopts::value<int>()->default_value("0"))
    ("max_batch","max batch_size to insert for ramt gen edges",cxxopts::value<int>()->default_value("100000"))
    ("timing_inserts","run a quick test of inserts with a bunch of different data structures")
    ("timing_random_inserts", "run a quick test of inserts with a bunch of different data structures random items up to some value")
    ("perf_test_tinyset", "time insertions and sum for tinyset")
    ("perf_test_set","time insertions and sum for set")
    ("perf_test_unorderedset", "time insertions and sum for unorderedset")
    ("real","runs with a graph from a file",cxxopts::value<std::string>())
    ("real_weights", "runs with a weighted graph from a file", cxxopts::value<std::string>())
    ("static","runs with a graph from a file",cxxopts::value<std::string>())
    ("run_info", "extra print to static_test, used to show which cores were used", cxxopts::value< std::string>() ->implicit_value("-1"))
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  uint32_t el_count = result["el_count"].as<int>();
  uint32_t iters = result["iters"].as<int>();
  uint32_t max_val = result["max_val"].as<int>();
  uint32_t bfs_src = result["bfs_src"].as<int>();
  uint32_t max_batch = result["max_batch"].as<int>();

  if (result["help"].as<bool>()) {
    std::cout << options.help({"", "Group"}) << std::endl;
  }
  if (result["sizes"].as<bool>()) {
    printf("sizeof(PMA<sized_uint<4>>) = %zu\n", sizeof(PMA<sized_uint<4>>));
    printf("sizeof(PMA<sized_uint<4>, uint8_t>) = %zu\n",
           sizeof(PMA<sized_uint<4>, uint8_t>));
    printf("sizeof(PMA<sized_uint<4>, double>) = %zu\n",
           sizeof(PMA<sized_uint<4>, double>));
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

    // test_tinyset_size_file_out(el_count);
  }

  if (result.count("real") > 0) {
    real_graph(result["real"].as<std::string>(), true, iters, bfs_src,
               max_batch);
    return 0;
  }
  if (result.count("real_weights") > 0) {
    // real_graph_weights<bool>(result["real_weights"].as<std::string>(), true,
    //                          iters, bfs_src, max_batch);
    // real_graph_weights<uint8_t>(result["real_weights"].as<std::string>(),
    // true,
    //                             iters, bfs_src, max_batch);
    // real_graph_weights<uint16_t>(result["real_weights"].as<std::string>(),
    // true,
    //                              iters, bfs_src, max_batch);
    real_graph_weights<uint32_t>(result["real_weights"].as<std::string>(), true,
                                 iters, bfs_src, max_batch);
    // real_graph_weights<uint64_t>(result["real_weights"].as<std::string>(),
    // true,
    //                              iters, bfs_src, max_batch);
    return 0;
  }

  if (result.count("static") > 0) {
    real_graph_static_test(result["static"].as<std::string>(), true, iters,
                           bfs_src, result["run_info"].as<std::string>());
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
  if (result["perf_test_set"].as<bool>()) {
    perf_test_set(el_count);
  }
  if (result["perf_test_unorderedset"].as<bool>()) {
    perf_test_unordered_set(el_count);
  }

  return 0;
}
