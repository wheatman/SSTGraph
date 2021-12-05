#pragma once
// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Triangle counting code (assumes a symmetric graph, so pass the "-s"
// flag). This is not optimized (no ordering heuristic is used)--for
// optimized code, see "Multicore Triangle Computations Without
// Tuning", ICDE 2015. Currently only works with uncompressed graphs,
// and not with compressed graphs.
#include "../SparseMatrix.hpp"

struct countF { // for edgeMap
  const SparseMatrixV<true, bool> &G;
  std::vector<uint64_t> &counts;
  countF(const SparseMatrixV<true, bool> &G_, std::vector<uint64_t> &_counts)
      : G(G_), counts(_counts) {}
  inline bool update(uint32_t s, uint32_t d) {
    if (s > d) { // only count "directed" triangles
      counts[8 * getWorkerNum()] += G.common_neighbors(s, d, true);
    }
    return true;
  }
  inline bool updateAtomic(uint32_t s, uint32_t d) {
    if (s > d) { // only count "directed" triangles
      counts[8 * getWorkerNum()] += G.common_neighbors(s, d, true);
    }
    return true;
  }
  inline bool cond([[maybe_unused]] uint32_t d) { return true; } // does nothing
};

void TC(const SparseMatrixV<true, bool> &G) {
  uint32_t n = G.get_rows();
  std::vector<uint64_t> counts(getWorkers() * 8, 0);
  VertexSubset Frontier(0, n, true); // frontier contains all vertices

  G.edgeMap(Frontier, countF(G, counts), false);
  uint64_t count = 0;
  for (int i = 0; i < getWorkers(); i++) {
    count += counts[i * 8];
  }
  printf("triangle count = %ld\n", count);
}
