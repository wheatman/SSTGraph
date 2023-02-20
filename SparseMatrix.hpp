#pragma once
#include "BitArray.hpp"
#include "TinySet_small.hpp"
#include "VertexSubset.hpp"
#include "helpers.h"
#include "integerSort/blockRadixSort.h"
#include "parallel.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <tuple>
#include <type_traits>
#include <vector>

template <bool is_csr_ = true, typename value_type = bool> class SparseMatrixV {
  static constexpr bool binary = std::is_same<value_type, bool>::value;

private:
  using edge_type =
      typename std::conditional<binary, std::tuple<el_t, el_t>,
                                std::tuple<el_t, el_t, value_type>>::type;
  TinySetV_small<value_type> *lines;
  typename TinySetV_small<value_type>::extra_data ts_data;
  uint32_t line_count;
  uint32_t line_width;

  template <class F, bool output> struct MAP_SPARSE {
  private:
    const el_t src;
    F &f;
    VertexSubset &output_vs;

  public:
    static constexpr bool no_early_exit = true;

    MAP_SPARSE(const el_t src, F &f, VertexSubset &output_vs)
        : src(src), f(f), output_vs(output_vs) {}

    inline bool update(el_t dest, [[maybe_unused]] value_type val) {
      constexpr bool no_vals =
          std::is_invocable_v<decltype(&F::update), F &, uint32_t, uint32_t>;
      if (f.cond(dest) == 1) {
        if constexpr (output) {
          bool r;
          if constexpr (no_vals) {
            r = f.updateAtomic(src, dest);
          } else {
            r = f.updateAtomic(src, dest, val);
          }
          if (r) {
            output_vs.insert_sparse(dest);
          }
        } else {
          if constexpr (no_vals) {
            f.updateAtomic(src, dest);
          } else {
            f.updateAtomic(src, dest, val);
          }
        }
      }
      return false;
    }
  };

  template <class F, bool output> struct EDGE_MAP_SPARSE {
    static_assert(is_csr_, "EdgeMap only works with csr");

  private:
    const SparseMatrixV &G;
    VertexSubset &output_vs;
    F f;
    const typename TinySetV_small<value_type>::extra_data &d;

  public:
    EDGE_MAP_SPARSE(const SparseMatrixV &G_, VertexSubset &output_vs_, F f_,
                    const typename TinySetV_small<value_type>::extra_data &d_)
        : G(G_), output_vs(output_vs_), f(f_), d(d_) {}
    inline bool update(el_t val) {
      struct MAP_SPARSE<F, output> ms(val, f, output_vs);
      // openmp is doing wrong things with nested parallelism so disable the
      // inner parallel portion
#if OPENMP == 1
      G.lines[val].template map<MAP_SPARSE<F, output>>(ms, d, false);
#else
      G.lines[val].template map<MAP_SPARSE<F, output>>(ms, d, true);
#endif
      return false;
    }
  };

  template <class F, bool output> struct VERTEX_MAP {
    static_assert(is_csr_, "VertexMap only works with csr");

  private:
    const VertexSubset &vs;
    VertexSubset &output_vs;
    F f;

  public:
    VERTEX_MAP(const VertexSubset &vs_, VertexSubset &output_vs_, F f_)
        : vs(vs_), output_vs(output_vs_), f(f_) {}
    inline bool update(el_t val) {
      if constexpr (output) {
        if (f(val) == 1) {
          output_vs.insert(val);
        }
      } else {
        f(val);
      }
      return false;
    }
  };

  template <class F, bool output>
  VertexSubset EdgeMapSparse(const VertexSubset &vertext_subset, F f) const {
    static_assert(is_csr_, "EdgeMap only works with csr");
    VertexSubset vs = (vertext_subset.sparse())
                          ? vertext_subset
                          : vertext_subset.convert_to_sparse();
    if constexpr (output) {
      VertexSubset output_vs = VertexSubset(vs, false);
      struct EDGE_MAP_SPARSE<F, output> v(*this, output_vs, f, ts_data);
      vs.map_sparse(v);
      output_vs.finalize();
      if (!vertext_subset.sparse()) {
        vs.del();
      }
      return output_vs;
    } else {
      VertexSubset null_vs = VertexSubset();
      struct EDGE_MAP_SPARSE<F, output> v(*this, null_vs, f, ts_data);
      vs.map_sparse(v);
      if (!vertext_subset.sparse()) {
        vs.del();
      }
      return null_vs;
    }
  }

  template <class F, bool output, bool vs_all> struct MAP_DENSE {
  private:
    const el_t dest;
    F &f;
    const VertexSubset &vs;
    VertexSubset &output_vs;

  public:
    static constexpr bool no_early_exit = false;

    MAP_DENSE(const el_t dest, F &f, const VertexSubset &vs,
              VertexSubset &output_vs)
        : dest(dest), f(f), vs(vs), output_vs(output_vs) {}

    inline bool update(el_t src, [[maybe_unused]] value_type val) {
      constexpr bool no_vals =
          std::is_invocable_v<decltype(&F::update), F &, uint32_t, uint32_t>;
      bool has = true;
      if constexpr (!vs_all) {
        has = vs.has_dense_no_all(src);
      }
      if (has) {
        bool r;
        if constexpr (no_vals) {
          r = f.update(src, dest);
        } else {
          r = f.update(src, dest, val);
        }
        if constexpr (output) {
          if (r) {
            output_vs.insert_dense(dest);
          }
        }
        if (f.cond(dest) == 0) {
          return true;
        }
      }
      return false;
    }
  };

  template <class F, bool output, bool vs_all>
  VertexSubset EdgeMapDense(const VertexSubset &vertext_subset, F f) const {
    static_assert(is_csr_, "EdgeMap only works with csr");
    VertexSubset vs = (vertext_subset.sparse())
                          ? vertext_subset.convert_to_dense()
                          : vertext_subset;
    if constexpr (output) {
      VertexSubset output_vs = VertexSubset(vs, false);
      // needs a grainsize of at least 512
      // so writes to the bitvector storing the next vertex set are going to
      // different cache lines
      parallel_for(uint64_t i_ = 0; i_ < get_rows(); i_ += 512) {
        uint64_t end = std::min(i_ + 512, (uint64_t)get_rows());
        uint64_t i = i_;
        for (; i < end - 1; i++) {
          lines[i + 1].prefetch_data(ts_data);
          if (f.cond(i) == 1) {
            MAP_DENSE<F, output, vs_all> md(i, f, vs, output_vs);
            lines[i].template map<MAP_DENSE<F, output, vs_all>>(md, ts_data,
                                                                false);
          }
        }
        // dealing with the last iteration outside
        // skips needs to branch for the prefetch to avoid out of bounds
        if (f.cond(i) == 1) {
          MAP_DENSE<F, output, vs_all> md(i, f, vs, output_vs);
          lines[i].template map<MAP_DENSE<F, output, vs_all>>(md, ts_data,
                                                              false);
        }
      }
      if (vertext_subset.sparse()) {
        vs.del();
      }
      return output_vs;
    } else {
      VertexSubset null_vs = VertexSubset();
      // needs a grainsize of at least 512
      // so writes to the bitvector storing the next vertex set are going to
      // different cache lines
      parallel_for(uint64_t i_ = 0; i_ < get_rows(); i_ += 512) {
        uint64_t end = std::min(i_ + 512, (uint64_t)get_rows());
        uint64_t i = i_;
        for (; i < end - 1; i++) {
          lines[i + 1].prefetch_data(ts_data);
          if (f.cond(i) == 1) {
            MAP_DENSE<F, output, vs_all> md(i, f, vs, null_vs);
            lines[i].template map<MAP_DENSE<F, output, vs_all>>(md, ts_data,
                                                                false);
          }
        }
        // dealing with the last iteration outside
        // skips needs to branch for the prefetch to avoind out of bounds
        if (f.cond(i) == 1) {
          MAP_DENSE<F, output, vs_all> md(i, f, vs, null_vs);
          lines[i].template map<MAP_DENSE<F, output, vs_all>>(md, ts_data,
                                                              false);
        }
      }
      if (vertext_subset.sparse()) {
        vs.del();
      }
      return null_vs;
    }
  }

public:
  [[nodiscard]] uint32_t get_line_count() const { return line_count; }
  [[nodiscard]] uint32_t get_line_width() const { return line_width; }
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

  SparseMatrixV(el_t height, el_t width);
  SparseMatrixV(const SparseMatrixV &source)
      : ts_data(source.ts_data), line_count(source.line_count),
        line_width(source.line_width) {
    lines = (TinySetV_small<value_type> *)malloc(
        line_count * sizeof(TinySetV_small<value_type>));
    parallel_for(el_t i = 0; i < line_count; i++) {
      new (&lines[i]) TinySetV_small<value_type>(source.lines[i], ts_data);
    }
  }
  ~SparseMatrixV();
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
  void insert(el_t row, el_t col);
  void insert(el_t row, el_t col, value_type val);
  void remove(el_t row, el_t col);
  void insert_batch(edge_type *edges, uint64_t n);
  void remove_batch(edge_type *edges, uint64_t n);

  template <class F>
  VertexSubset edgeMap(VertexSubset &vs, F f, bool output = true,
                       uint32_t threshold = 20) const {
    static_assert(is_csr_, "EdgeMap only works in with csr");
    if (output) {
      if (vs.complete()) {
        if (get_rows() / threshold <= vs.get_n()) {
          auto out = EdgeMapDense<F, true, true>(vs, f);
          return out;
        } else {
          auto out = EdgeMapSparse<F, true>(vs, f);
          return out;
        }
      } else {
        if (get_rows() / threshold <= vs.get_n()) {
          auto out = EdgeMapDense<F, true, false>(vs, f);
          return out;
        } else {
          auto out = EdgeMapSparse<F, true>(vs, f);
          return out;
        }
      }
    } else {
      if (vs.complete()) {
        if (get_rows() / threshold <= vs.get_n()) {
          auto out = EdgeMapDense<F, false, true>(vs, f);
          return out;
        } else {
          auto out = EdgeMapSparse<F, false>(vs, f);
          return out;
        }
      } else {
        if (get_rows() / threshold <= vs.get_n()) {
          auto out = EdgeMapDense<F, false, false>(vs, f);
          return out;
        } else {
          auto out = EdgeMapSparse<F, false>(vs, f);
          return out;
        }
      }
    }
  }

  template <class F>
  VertexSubset vertexMap(VertexSubset &vs, F f, bool output = true) const {
    static_assert(is_csr_, "vertexMap only works with csr");

    if (output) {
      VertexSubset output_vs = VertexSubset(vs, false);
      struct VERTEX_MAP<F, true> v(vs, output_vs, f);
      vs.map(v);
      output_vs.finalize();
      return output_vs;
    } else {
      // output is empty
      VertexSubset null_vs = VertexSubset();
      struct VERTEX_MAP<F, false> v(vs, null_vs, f);
      vs.map(v);
      return null_vs;
    }
  }

  [[nodiscard]] uint64_t get_memory_size() const;
  [[nodiscard]] uint64_t M() const;
  [[nodiscard]] SparseMatrixV<true, value_type> convert_to_csr();
  [[nodiscard]] SparseMatrixV<false, value_type> convert_to_csc();

  void print_arrays() const {
    if constexpr (is_csr()) {
      printf("matrix stored in csr\n");
    }
    if constexpr (is_csc()) {
      printf("matrix stored in csc\n");
    }
    printf("there are %lu elements\n", M());
    for (uint32_t line = 0; line < line_count; line++) {
      printf("line %u\n", line);
      lines[line].print(ts_data);
      printf("sum = %lu\n", lines[line].sum_keys(ts_data));
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

    uint32_t counts[5] = {0};
    uint64_t max_num_elements[5] = {0};
    uint64_t empty[5] = {0};
    uint64_t total_buckets[5] = {0};
    for (uint32_t i = 0; i < line_count; i++) {
      counts[lines[i].get_b()]++;
      uint32_t b = lines[i].get_b();
      auto stats_per_line = lines[i].statistics(ts_data);
      total_buckets[b] += std::get<0>(stats_per_line);
      max_num_elements[b] =
          std::max(max_num_elements[b], std::get<1>(stats_per_line));
      empty[b] += std::get<2>(stats_per_line);
    }
    for (uint32_t b = 1; b < 5; b++) {
      printf("there are %u rows with b = %u, and the max number of elements in "
             "any is %lu, giving a density of %f\n",
             counts[b], b, max_num_elements[b],
             static_cast<double>(max_num_elements[b]) / (1UL << (b * 8)));
      printf("\t%lu buckets are empty out of %lu total buckets "
             "with b "
             "= %u\n",
             empty[b], total_buckets[b], b);
    }
  }

  void print_line(uint32_t line) { lines[line].print(ts_data); }

  // sums the values of the dests
  // doesn't have much meaning in either the graph or matrix world
  // used to touch every value and benchmarks operations
  // when order is not null touched the rows in the specified order
  uint64_t touch_all_sum(uint64_t const *const order = nullptr) const {
    uint64_t sum = 0;
    int n_workers = getWorkers();
    std::vector<int64_t> sum_vector(n_workers * 8, 0);
    if (order == nullptr) {
      parallel_for(uint64_t i = 0; i < line_count; i++) {
        uint32_t worker_num = getWorkerNum();
        sum_vector[8 * worker_num] += lines[i].sum_keys(ts_data);
      }
      for (auto s : sum_vector) {
        sum += s;
      }
    } else {
      parallel_for(uint64_t i = 0; i < line_count; i++) {
        uint32_t worker_num = getWorkerNum();
        sum_vector[8 * worker_num] += lines[order[i]].sum_keys(ts_data);
      }
      for (auto s : sum_vector) {
        sum += s;
      }
    }
    return sum;
  }
};

template <bool is_csr_, typename value_type>
uint64_t SparseMatrixV<is_csr_, value_type>::M() const {
  uint64_t size = 0;
  for (el_t i = 0; i < line_count; i++) {
    size += lines[i].get_n();
  }
  return size;
}

template <bool is_csr_, typename value_type>
uint64_t SparseMatrixV<is_csr_, value_type>::get_memory_size() const {
  uint64_t size = 0;
  for (uint32_t i = 0; i < line_count; i++) {
    size += lines[i].get_size(ts_data);
  }
  return size + sizeof(SparseMatrixV);
}

template <bool is_csr_, typename value_type>
void __attribute__((noinline))
SparseMatrixV<is_csr_, value_type>::insert(el_t row, el_t col) {
  if constexpr (is_csr()) {
    lines[row].insert(col, ts_data);
  } else {
    lines[col].insert(row, ts_data);
  }
}

template <bool is_csr_, typename value_type>
void __attribute__((noinline))
SparseMatrixV<is_csr_, value_type>::insert(el_t row, el_t col, value_type val) {
  if constexpr (is_csr()) {
    lines[row].insert({col, val}, ts_data);
  } else {
    lines[col].insert({row, val}, ts_data);
  }
}

template <bool is_csr_, typename value_type>
void __attribute__((noinline))
SparseMatrixV<is_csr_, value_type>::remove(el_t row, el_t col) {
  if constexpr (is_csr()) {
    lines[row].remove(col, ts_data);
  } else {
    lines[col].remove(row, ts_data);
  }
}

template <bool is_csr_, typename value_type>
void __attribute__((noinline))
SparseMatrixV<is_csr_, value_type>::insert_batch(edge_type *edges, uint64_t n) {
  uint64_t n_workers = getWorkers();
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
        if constexpr (binary) {
          lines[x].insert(y, ts_data);
        } else {
          lines[x].insert({y, std::get<2>(edges[idx])}, ts_data);
        }
      }
      return;
    }
#if (OPENMP == 0) && (CILK == 0)
    std::sort(edges, &edges[n]);
#else
    integerSort_x(edges, n, line_width);
#endif
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
    parallel_for(uint64_t i = 0; i < p; i++) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];

      for (; idx < end; idx++) {

        // Not including self loops to compare to aspen
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        if (x == y) {
          continue;
        }
        if constexpr (binary) {
          lines[x].insert(y, ts_data);
        } else {
          lines[x].insert({y, std::get<2>(edges[idx])}, ts_data);
        }
      }
    }
  } else {
    if (n <= 100) {
      for (uint64_t idx = 0; idx < n; idx++) {
        // Not including self loops to compare to aspen
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        if (x == y) {
          continue;
        }
        if constexpr (binary) {
          lines[y].insert(x, ts_data);
        } else {
          lines[y].insert({x, std::get<2>(edges[idx])}, ts_data);
        }
      }
      return;
    }
#if (OPENMP == 0) && (CILK == 0)
    std::sort(edges, &edges[n], [](auto const &t1, auto const &t2) {
      return get<1>(t1) < get<1>(t2);
    });
#else
    integerSort_y(edges, n, line_width);
#endif
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
    parallel_for(uint64_t i = 0; i < p; i++) {
      for (uint64_t idx = indxs[i]; idx < indxs[i + 1]; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        if constexpr (binary) {
          lines[y].insert(x, ts_data);
        } else {
          lines[y].insert({x, std::get<2>(edges[idx])}, ts_data);
        }
      }
    }
  }
}

template <bool is_csr_, typename value_type>
void __attribute__((noinline))
SparseMatrixV<is_csr_, value_type>::remove_batch(edge_type *edges, uint64_t n) {
  uint64_t n_workers = getWorkers();
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
#if (OPENMP == 0) && (CILK == 0)
    std::sort(edges, &edges[n]);
#else
    integerSort_x(edges, n, line_width);
#endif
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
    parallel_for(uint64_t i = 0; i < p; i++) {
      uint64_t idx = indxs[i];
      uint64_t end = indxs[i + 1];
      for (; idx < end; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[x].remove(y, ts_data);
      }
    }
  } else {
    if (n <= 100) {
      for (uint64_t idx = 0; idx < n; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[y].remove(x, ts_data);
      }
      return;
    }
#if (OPENMP == 0) && (CILK == 0)
    std::sort(edges, &edges[n], [](auto const &t1, auto const &t2) {
      return get<1>(t1) < get<1>(t2);
    });
#else
    integerSort_y(edges, n, line_width);
#endif
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
    parallel_for(uint64_t i = 0; i < p; i++) {
      for (uint64_t idx = indxs[i]; idx < indxs[i + 1]; idx++) {
        el_t x = std::get<0>(edges[idx]);
        el_t y = std::get<1>(edges[idx]);
        lines[y].remove(x, ts_data);
      }
    }
  }
}

template <bool is_csr_, typename value_type>
bool SparseMatrixV<is_csr_, value_type>::has(el_t row, el_t col) const {
  if constexpr (is_csr()) {
    return lines[row].has(col, ts_data);
  }
  return lines[col].has(row, ts_data);
}

template <bool is_csr_, typename value_type>
value_type SparseMatrixV<is_csr_, value_type>::value(el_t row, el_t col) const {
  if constexpr (is_csr()) {
    return lines[row].value(col, ts_data);
  }
  return lines[col].value(row, ts_data);
}

template <bool is_csr_, typename value_type>
SparseMatrixV<is_csr_, value_type>::SparseMatrixV(el_t height, el_t width) {
  ts_data.thresh_24 = 2;
  ts_data.thresh_16 = 4;
  ts_data.thresh_8 = 8;
  if constexpr (is_csr()) {
    ts_data.max_el = width;
    line_count = height;
    line_width = width;
  } else {
    ts_data.max_el = height;
    line_count = width;
    line_width = height;
  }
  lines = (TinySetV_small<value_type> *)malloc(
      line_count * sizeof(TinySetV_small<value_type>));
  ts_data = lines[0].get_thresholds(ts_data.max_el);
  for (el_t i = 0; i < line_count; i++) {
    new (&lines[i]) TinySetV_small<value_type>();
  }
}

template <bool is_csr_, typename value_type>
SparseMatrixV<is_csr_, value_type>::~SparseMatrixV<is_csr_, value_type>() {
  for (uint32_t i = 0; i < line_count; i++) {
    lines[i].destroy(ts_data);
  }
  free(lines);
}
