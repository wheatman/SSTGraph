#include "cxxopts.hpp"

#include "SSTGraph/SparseMatrix.hpp"
#include "SSTGraph/internal/rmat_util.h"

using namespace SSTGraph;

void stats_graph(const std::string &filename) {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t> *edges =
      get_edges_from_file(filename, &num_edges, &num_nodes);

  SparseMatrixV<true, bool> g(num_nodes, num_nodes);
  g.insert_batch(edges, num_edges);
  g.print_statistics();
  printf("checking with degree ordered relabeled graph\n");
  auto degree_g = g.rewrite_graph(g.degree_order_map());
  degree_g.print_statistics();
}

void get_graph_distribution(const std::string &filename) {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t> *edges =
      get_edges_from_file(filename, &num_edges, &num_nodes);

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
  std::map<uint64_t, uint64_t> degrees;
  for (uint64_t i = 0; i < num_nodes; i++) {
    degrees[g.getDegree(i)] += 1;
  }
  for (auto pair : degrees) {
    printf("%lu, %lu\n", pair.first, pair.second);
  }
}

void rewrite_graph(const std::string &filename) {
  printf("rewriting graph %s\n", filename.c_str());
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t> *edges =
      get_edges_from_file(filename, &num_edges, &num_nodes);
  printf("num_nodes = %u\n", num_nodes);
  std::vector<uint32_t> new_node_ids(num_nodes, 0);
  for (uint32_t i = 0; i < num_nodes; i++) {
    new_node_ids[i] = i;
  }
  std::mt19937 rng;
  rng.seed(0);
  std::shuffle(new_node_ids.begin(), new_node_ids.end(), rng);
  printf("node 35 in the old graph is node %u in the new\n", new_node_ids[35]);
  std::string f_name = filename + "el.shuf";
  FILE *fw = fopen(f_name.c_str(), "w");
  if (fw == nullptr) {
    printf("file was not opened\n");
    free(edges);
    return;
  }
  for (uint64_t i = 0; i < num_edges; i++) {
    fprintf(fw, "%u   %u\n", new_node_ids[std::get<0>(edges[i])],
            new_node_ids[std::get<1>(edges[i])]);
  }
  free(edges);
  // return 0;
  fclose(fw);
  printf("finished writing %s\n", (filename + "el.shuf").c_str());
}

void write_weighted_graph(const std::string &filename) {
  printf("rewriting graph %s\n", filename.c_str());
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t, uint32_t> *edges =
      get_edges_from_file<uint32_t>(filename, &num_edges, &num_nodes);

  SparseMatrixV<true, uint32_t> g(num_nodes, num_nodes);
  g.insert_batch(edges, num_edges);
  g.print_statistics();
  printf("num_nodes = %u\n", num_nodes);
  std::vector<uint32_t> new_node_ids(num_nodes, 0);
  std::string f_name = filename + ".adj";
  FILE *fw = fopen(f_name.c_str(), "w");
  if (fw == nullptr) {
    printf("file was not opened\n");
    free(edges);
    return;
  }
  fprintf(fw, "WeightedAdjacencyGraph\n");
  fprintf(fw, "%u\n", num_nodes);
  fprintf(fw, "%lu\n", g.M());
  uint64_t offset = 0;
  for (uint64_t i = 0; i < num_nodes; i++) {
    fprintf(fw, "%lu\n", offset);
    offset += g.getDegree(i);
  }

  for (uint64_t i = 0; i < num_nodes; i++) {
    g.map_line<true>([&](el_t dest) { fprintf(fw, "%u\n", dest); }, i, false);
  }
  for (uint64_t i = 0; i < num_nodes; i++) {
    g.map_line<true, 0>([&]([[maybe_unused]] el_t dest,
                            uint32_t val) { fprintf(fw, "%u\n", val); },
                        i, false);
  }
  free(edges);
  // return 0;
  fclose(fw);
  printf("finished writing %s\n", f_name.c_str());
}

void write_unweighted_graph(const std::string &filename) {
  printf("rewriting graph %s\n", filename.c_str());
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  std::tuple<el_t, el_t, uint32_t> *edges =
      get_edges_from_file<uint32_t>(filename, &num_edges, &num_nodes);

  SparseMatrixV<true, uint32_t> g(num_nodes, num_nodes);
  g.insert_batch(edges, num_edges);
  g.print_statistics();
  printf("num_nodes = %u\n", num_nodes);
  std::vector<uint32_t> new_node_ids(num_nodes, 0);
  std::string f_name = filename + ".unweighted.adj";
  FILE *fw = fopen(f_name.c_str(), "w");
  if (fw == nullptr) {
    printf("file was not opened\n");
    free(edges);
    return;
  }
  fprintf(fw, "AdjacencyGraph\n");
  fprintf(fw, "%u\n", num_nodes);
  fprintf(fw, "%lu\n", g.M());
  uint64_t offset = 0;
  for (uint64_t i = 0; i < num_nodes; i++) {
    fprintf(fw, "%lu\n", offset);
    offset += g.getDegree(i);
  }

  for (uint64_t i = 0; i < num_nodes; i++) {
    g.map_line<true>([&](el_t dest) { fprintf(fw, "%u\n", dest); }, i, false);
  }
  free(edges);
  // return 0;
  fclose(fw);
  printf("finished writing %s\n", f_name.c_str());
}

void make_ER_graph(uint32_t nodes, double p, bool symetrize,
                   const std::string &filename) {
  SparseMatrixV<true, bool> g(nodes, nodes);
  ParallelTools::parallel_for(0, nodes, [&](uint32_t i) {
    std::mt19937 gen(i);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (uint32_t j = 0; j < nodes; j++) {
      if (dis(gen) <= p) {
        g.insert({i, j});
      }
    }
  });
  if (symetrize) {
    std::vector<std::tuple<el_t, el_t>> edges;
    for (uint32_t i = 0; i < nodes; i++) {
      g.map_line<true>(
          [&](el_t dest) {
            if (dest != i) {
              edges.emplace_back(dest, i);
            }
          },
          i, false);
    }
    g.insert_batch(edges.data(), edges.size());
  }
  printf("num_nodes = %u, num_edges = %lu\n", g.get_rows(), g.M());
  FILE *fw = fopen(filename.c_str(), "w");
  if (fw == nullptr) {
    printf("file was not opened\n");
    return;
  }
  fprintf(fw, "AdjacencyGraph\n");
  fprintf(fw, "%u\n", nodes);
  fprintf(fw, "%lu\n", g.M());
  uint64_t offset = 0;
  for (uint64_t i = 0; i < nodes; i++) {
    fprintf(fw, "%lu\n", offset);
    offset += g.getDegree(i);
  }

  for (uint64_t i = 0; i < nodes; i++) {
    g.map_line<true>([&](el_t dest) { fprintf(fw, "%u\n", dest); }, i, false);
  }
  fclose(fw);
}

void rmat_distribution_info(uint64_t num_nodes, uint64_t b_size) {
  std::cout << "num nodes = " << num_nodes << " batch size = " << b_size
            << std::endl;
  auto r = random_aspen();
  double a = 0.5;
  double b = 0.1;
  double c = 0.1;
  size_t nn = 1UL << (log2_up(num_nodes) - 1);
  auto rmat = rMat<uint32_t>(nn, r.ith_rand(0), a, b, c);
  std::vector<std::pair<el_t, el_t>> es(b_size);
  for (uint32_t i = 0; i < b_size; i++) {
    es[i] = rmat(i);
  }
  std::sort(es.begin(), es.end());
  std::vector<uint64_t> counts(num_nodes);
  counts[es[0].first]++;
  for (uint64_t i = 1; i < es.size(); i++) {
    auto last = es[i - 1];
    auto current = es[i];
    if (current == last) {
      continue;
    }
    counts[es[i].first]++;
  }
  std::map<uint64_t, uint64_t> histogram;
  for (auto item : counts) {
    histogram[item]++;
  }
  for (auto item : histogram) {
    std::cout << item.second << " nodes have " << item.first << " elements"
              << std::endl;
  }
  std::cout << std::endl;
}

void watts_strogatz_graph(uint64_t N, uint64_t K, double beta) {
  const auto g = SparseMatrixV<>::make_watts_strogatz_graph(N, K, beta);
  g.print_statistics();
}

int main(int argc, char *argv[]) {

  cxxopts::Options options("Graph Utils", "access to some graph utilities");

  options.positional_help("Help Text");
  // clang-format off
  options.add_options()
    ("stats","print the stats from a graph from a file",cxxopts::value<std::string>())
    ("degree","prints the degree distibution", cxxopts::value<std::string>())
    ("write_weighted_adj","writes a weighted graph out in adj format", cxxopts::value<std::string>())
    ("write_adj","writes a graph out in adj format", cxxopts::value<std::string>())
    ("rewrite","shuffles the node labels of a graph file and makes a new file", cxxopts::value<std::string>())
    ("make_er","make an er graph, need to specify rows and p",cxxopts::value<std::string>())
    ("rmat_distrib","statistics of an rmat distribution",cxxopts::value<std::string>())
    ("p","probability",cxxopts::value<double>()->default_value(".001"))
    ("r,rows", "how many rows", cxxopts::value<int>()->default_value("10000"))
    ("batch_size", "batch_size", cxxopts::value<int>()->default_value("10000"))
    ("watts_strogatz", "make a run statstics on a watts_strogatz graph")
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("help","Print help");
  // clang-format on
  auto result = options.parse(argc, argv);
  double p = result["p"].as<double>();
  uint32_t rows = result["rows"].as<int>();
  uint32_t batch_size = result["batch_size"].as<int>();
  uint32_t el_count = result["el_count"].as<int>();

  if (result.count("stats") > 0) {
    stats_graph(result["stats"].as<std::string>());
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

  if (result.count("write_weighted_adj") > 0) {
    write_weighted_graph(result["write_weighted_adj"].as<std::string>());
    return 0;
  }

  if (result.count("write_adj") > 0) {
    write_unweighted_graph(result["write_adj"].as<std::string>());
    return 0;
  }

  if (result.count("make_er") > 0) {
    make_ER_graph(rows, p, false, result["make_er"].as<std::string>());
    return 0;
  }

  if (result.count("rmat_distrib") > 0) {
    rmat_distribution_info(rows, batch_size);
    return 0;
  }

  if (result.count("watts_strogatz") > 0) {
    watts_strogatz_graph(rows, el_count, p);
  }
}
